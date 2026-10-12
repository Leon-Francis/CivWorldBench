"""Synthesise a semantic STRATEGY DIRECTIVE for the ego civ from the engine's own AI plan state.

Design rule (§6.2 of doc 12): the ego civ's OWN hidden plan fields are revealed to it -- you know
your own plan -- while every other civ's same fields stay hidden and remain task-2 probe targets.
So a directive may only ever read `probes.*[ego]`, never `probes.*[other]`.

Provenance discipline, per field:
  t1-INTENT (safe: it is the ego's hidden state at the anchor, not an outcome)
    B_attitude[ego][x].{love, countdown, ask_peace, ask_alliance, ask_ceasefire}
    D_intent[ego].{wonder_city, research.goal_name}, per-city {ai.founder_want, ai.urgency}
    unit_orders[ego] destinations of moving military units  (`destination` is intent; `at` is an
    output -- v1's action path leaked the trajectory through `at (x,y)`)
  FUTURE-INTENT (aggregated per-turn intent over [t1+1, t1+h]; same category as v1's intent path,
    must pass a leakage gate before use)
    production mix shares, tax/science split
  NEVER: any count, any event, any research-goal sequence length (~= techs_learned)

    .venv/bin/python bench/measure/directive_synth.py --world <id> --t1 <turn> --civ <name>
    .venv/bin/python bench/measure/directive_synth.py --samples 4
"""
from __future__ import annotations
import argparse, collections, gzip, json, pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bundle")); sys.path.insert(0, str(ROOT / "tools"))
from bench.measure.action_profile import profile, l1, PROD_CATS
from bench.spec_v2 import HZ_BY_BUDGET, LOVE_BINS
from bench.interval_v2 import _near

# Bins from the corpus distribution, `bench/measure/out/plan_fields_gate.json` (80 worlds):
#   ai.founder_want  21% > 0, p50(nonzero)=198, p75=340, p90=439
#   ai.urgency       61% > 0, p50=16, p75=66, p90=211
FW_BINS = (1, 340)
URG_BINS = (16, 211)
# ★ GATE 2026-09-12: `countdown` does NOT mean what its name suggests.  Measured on 11,236
# (ego, other, anchor) pairs, share that actually declares war within 10 turns:
#   countdown 1-10 : 48.1%   (27x the no-plan baseline)  <- a real war plan
#   countdown == 0 :  5.4%   <- NOT "ready to strike now"; the countdown is inactive/expired
#   countdown == -1:  1.8%   <- baseline
# and countdown takes many values < -1 (a different counter, semantics undecoded).
# So only 1 <= countdown <= ~10 may be rendered as an intention to attack.
CD_MIN, CD_MAX = 1, 12
SHARE = [(0.02, None), (0.10, "a little"), (0.30, "some"), (0.45, "about a third"),
         (0.60, "about half"), (0.80, "most"), (1.01, "almost all")]
PLABEL = {"expansion": "settlers for new cities", "worker": "workers and engineers",
          "military": "military units", "civilian": "diplomats, caravans and transports",
          "infrastructure": "city buildings", "wonder": "wonders",
          "coinage": "coinage (gold instead of a build)"}


def share_word(x):
    for hi, w in SHARE:
        if x < hi:
            return w
    return "almost all"


def love_word(v):
    lo, hi = LOVE_BINS[0][1], LOVE_BINS[2][0]
    return "hostile towards" if v <= lo else ("friendly towards" if v >= hi else "indifferent to")


def clusters(dests, xsize, cap=3):
    """Merge intended destinations within MOVE_RADIUS into 'around (x,y)', biggest first."""
    out = []
    for d in dests:
        for c in out:
            if _near(c[0], d, xsize):
                c[1] += 1
                break
        else:
            out.append([d, 1])
    out.sort(key=lambda c: -c[1])
    return out[:cap]


BLOCK, REGIME_TH, MAX_PHASES = 20, 0.30, 3


def dir_label(fp, has_target):
    """Direction label from the production mix (never from wonder_city alone)."""
    if fp.get("military", 0) >= 0.55:
        return "go to war" if has_target else "arm heavily"
    if has_target:
        return "prepare for war"
    if fp.get("expansion", 0) >= 0.12:
        return "keep expanding"
    if fp.get("wonder", 0) >= 0.05:
        return "build wonders"
    if fp.get("infrastructure", 0) >= 0.40:
        return "build up its existing cities"
    return "hold its position"


def phases(recs, ego, t1, h, ruleset):
    """Segment [t1+1, t1+h] into <= MAX_PHASES regimes.  Measured on the corpus
    (`bench/measure/out/regime_scale.json`): a 250-turn window holds a median of 2 postures
    (p75 3, p90 5), and consecutive 20-turn blocks differ by only 0.114 (median).  So one
    directive over the whole horizon is an average of ~2 regimes -- 2-3 phases is the honest
    granularity, and it keeps the description short."""
    bs = []
    for a in range(t1 + 1, t1 + h + 1, BLOCK):
        b = min(a + BLOCK - 1, t1 + h)
        pr = profile(recs, ego, a, b, ruleset)
        if pr:
            bs.append((a, b, pr))
    if not bs:
        return []
    segs = [[bs[0]]]
    for prev, cur in zip(bs, bs[1:]):
        (segs[-1].append(cur) if l1(prev[2], cur[2]) <= REGIME_TH else segs.append([cur]))
    # keep the MAX_PHASES longest segments, in time order; merge the rest into their predecessor
    while len(segs) > MAX_PHASES:
        i = min(range(len(segs)), key=lambda j: len(segs[j]))
        j = i - 1 if i else 1
        segs[min(i, j)] = segs[i] + segs[j] if i < j else segs[j] + segs[i]
        del segs[max(i, j)]
    out = []
    for seg in segs:
        a, b = seg[0][0], seg[-1][1]
        pr = profile(recs, ego, a, b, ruleset)
        if pr:
            out.append((a, b, pr))
    return out


def synth(rec, recs, world, ego, t1, h, multi=True, sources="C2"):
    """`sources` selects how much the directive is allowed to read (doc 12 §6.7):
       C0  only the ego's explicit actions at t1        -- no leakage, but largely inferable
       C1  + the ego's hidden PLAN fields at t1         -- forward-looking by construction, zero
                                                           future data read; recommended default
       C2  + the realised posture over [t1+1, t1+h]     -- strongest, needs a leakage gate
    Traits are never included: they are task-2 targets, they are a disposition rather than a plan,
    and keeping them out preserves the "strategy wording vs mechanism wording" contrast."""
    """Returns (text, provenance dict)."""
    pb = rec["probes"]
    att = (pb.get("B_attitude") or {}).get(ego) or {}
    intent = (pb.get("D_intent") or {}).get(ego) or {}
    acts = (rec.get("actions") or {}).get(ego) or {}
    xsize = int((rec.get("geo") or {}).get("xsize") or 0)
    prov = {}

    # --- war aim: t1 countdown only (intent).  countdown >= 0 == a war is being prepared.
    targets = [(o, d.get("countdown")) for o, d in att.items()
               if o != ego and isinstance(d.get("countdown"), int) and CD_MIN <= d["countdown"] <= CD_MAX]
    targets.sort(key=lambda kv: kv[1])
    prov["countdown"] = {o: c for o, c in targets}

    # --- expansion appetite and wonder ambition: t1 city intent
    cities = intent.get("cities") or {}
    fw = max([int(c.get("ai.founder_want") or 0) for c in cities.values()] or [0])
    urg = max([int(c.get("ai.urgency") or 0) for c in cities.values()] or [0])
    wonder_city = intent.get("wonder_city") or 0
    goal = ((intent.get("research") or {}).get("goal_name") or "").strip()
    prov.update({"max_founder_want": fw, "max_urgency": urg, "wonder_city_id": wonder_city,
                 "research_goal": goal})

    # --- future posture: aggregated per-turn intent
    pf = profile(recs, ego, t1 + 1, t1 + h, world["ruleset"])
    pp = profile(recs, ego, t1 - 40, t1, world["ruleset"])
    prov["future_prod"] = {k: round(v, 3) for k, v in (pf or {"prod": {}})["prod"].items() if v >= 0.02}
    prov["visible_prod"] = {k: round(v, 3) for k, v in (pp or {"prod": {}})["prod"].items() if v >= 0.02}
    prov["drift"] = round(l1(pp, pf), 3) if (pp and pf) else None

    # --- offensive axis: intended destinations of moving military units at t1
    from bench.spec_v2 import mil_map_for
    from sandboxes.freeciv.state import base_ruleset
    mil = mil_map_for(base_ruleset(rec.get("ruleset") or "civ2civ3"))
    # only MILITARY units define an axis: settlers/workers wandering is not an offensive plan
    dests = [tuple(o["destination"]) for o in (acts.get("unit_orders") or [])
             if o.get("moving") and o.get("destination") and mil.get(o.get("type"))]
    cl = clusters(dests, xsize)
    prov["axis"] = [(tuple(c[0]), c[1]) for c in cl]

    # ---------------- render ----------------
    fp = prov["future_prod"]
    # the label must follow the production mix; `wonder_city` alone said "build wonders" for a civ
    # that spent 83% of its production on troops (2026-09-12)
    if fp.get("military", 0) >= 0.55:
        direction = ("go to war" if targets else "arm heavily, without having chosen a target yet")
    elif targets:
        direction = "prepare for war"
    elif fp.get("expansion", 0) >= 0.12:
        direction = "keep expanding"
    elif fp.get("wonder", 0) >= 0.05:
        direction = "build wonders"
    elif fp.get("infrastructure", 0) >= 0.40:
        direction = "build up the cities it already has"
    else:
        direction = "hold its position"

    snap = profile(recs, ego, t1, t1, world["ruleset"])
    prov["t1_prod"] = {k: round(v, 2) for k, v in (snap or {"prod": {}})["prod"].items() if v >= 0.02}
    if sources != "C2":
        L = [f"--- STRATEGY: {ego}, from turn {t1+1} on ---"]
        d = dir_label(prov["t1_prod"], bool(targets)) if snap else "hold its position"
        if targets:
            d = "prepare for war"
        elif fw >= FW_BINS[1]:
            d = "keep expanding"
        L.append(f"{ego} is set on this course: {d}.")
        if snap and sources == "C0":
            it = sorted(((v, k) for k, v in prov["t1_prod"].items()), reverse=True)
            pt = [f"{share_word(v)} to {PLABEL[k]}" for v, k in it if share_word(v)]
            L.append(f"  Right now its cities are building " + "; ".join(pt)
                     + f"; budget {round(snap['tax'])}% taxes / {round(snap['sci'])}% research.")
        if sources == "C1":
            if goal and goal != "A_UNSET":
                L.append(f"  Research aim: {goal}.")
            if targets:
                o, c = targets[0]
                ax = ", ".join(f"around ({x},{y})" for (x, y), n in cl) if cl else "no single axis yet"
                L.append(f"  War aim: it intends to attack {o} and will move in about {c} turns; "
                         f"its forces are converging {ax}.")
            elif cl:
                L.append("  Forces: its troops are concentrating "
                         + ", ".join(f"around ({x},{y})" for (x, y), n in cl) + ".")
            if fw >= FW_BINS[1]:
                L.append("  Settlement: it is still strongly set on founding new cities.")
            elif fw >= FW_BINS[0]:
                L.append("  Settlement: it would still like one or two more cities.")
            if urg >= URG_BINS[1]:
                L.append("  Defence: one of its cities is under heavy defensive pressure.")
            elif urg >= URG_BINS[0]:
                L.append("  Defence: one of its cities feels exposed.")
            nb = []
            for o, d2 in att.items():
                if o == ego:
                    continue
                w = love_word(int(d2.get("love") or 0))
                wants = [n for n, k in (("peace", "ask_peace"), ("an alliance", "ask_alliance"),
                                        ("a cease-fire", "ask_ceasefire")) if d2.get(k)]
                if w == "indifferent to" and not wants:
                    continue
                t = f"{w} {o}" + (f", and would accept {' and '.join(wants)}" if wants else "")
                nb.append(t)
            if nb:
                L.append("  Neighbours: it is " + "; ".join(nb) + ".")
        return "\n".join(L), prov
    ph = phases(recs, ego, t1, h, world["ruleset"]) if multi else []
    prov["phases"] = [(a, b, {k: round(v, 2) for k, v in pr["prod"].items() if v >= 0.02}) for a, b, pr in ph]
    L = [f"--- STRATEGY: {ego}, turns {t1+1}-{t1+h} ---"]
    if len(ph) > 1:
        L.append(f"{ego} follows this plan, in {len(ph)} phases:")
        for i, (a, b, pr) in enumerate(ph, 1):
            it = sorted(((v, k) for k, v in pr["prod"].items() if v >= 0.02), reverse=True)
            pt = [f"{share_word(v)} to {PLABEL[k]}" for v, k in it if share_word(v)]
            L.append(f"  Phase {i} (turns {a}-{b}): {dir_label(pr['prod'], False)}"
                     f" — production goes " + "; ".join(pt)
                     + f"; budget about {round(pr['tax'])}% taxes / {round(pr['sci'])}% research.")
    else:
        L.append(f"{ego} holds to the following plan for the whole period.")
        items = sorted(((v, k) for k, v in prov["future_prod"].items()), reverse=True)
        parts = [f"{share_word(v)} to {PLABEL[k]}" for v, k in items if share_word(v)]
        L.append(f"  Direction: {direction}. City production goes " + "; ".join(parts) + ".")
        if pf:
            L.append(f"  Budget: about {round(pf['tax'])}% taxes, {round(pf['sci'])}% research"
                     + (f", {round(pf['lux'])}% luxuries." if round(pf['lux']) else "."))
    if goal and goal != "A_UNSET":
        L.append(f"  Research aim: {goal}.")
    if targets:
        o, c = targets[0]
        when = f"will move in about {c} turns"
        ax = ", ".join(f"around ({x},{y})" for (x, y), n in cl) if cl else "no single axis yet"
        L.append(f"  War aim: {ego} intends to attack {o} and {when}; its forces are converging {ax}.")
        for o2, c2 in targets[1:]:
            L.append(f"            it is also preparing against {o2} (in about {c2} turns).")
    elif cl:
        ax = ", ".join(f"around ({x},{y})" for (x, y), n in cl)
        # never assert "no attack is coming": countdown == -1 only means no countdown is set NOW,
        # and a negative claim about the future can be falsified by the 1b war template
        L.append(f"  Forces: its troops are concentrating {ax}.")
    # raw AI scales are engine-internal and not comparable across worlds -> bins from the corpus
    # distribution (bench/measure/out/plan_fields_gate.json)
    if fw >= FW_BINS[1]:
        L.append("  Settlement: it is still strongly set on founding new cities.")
    elif fw >= FW_BINS[0]:
        L.append("  Settlement: it would still like one or two more cities.")
    if urg >= URG_BINS[1]:
        L.append("  Defence: one of its cities is under heavy defensive pressure.")
    elif urg >= URG_BINS[0]:
        L.append("  Defence: one of its cities feels exposed.")
    # Only non-neutral attitudes and explicit diplomatic wishes are stated.  Listing "indifferent
    # to X" for every pair reads as noise AND reveals a task-2 love label for nothing.
    nb = []
    for o, d in att.items():
        if o == ego:
            continue
        w = love_word(int(d.get("love") or 0))
        wants = [n for n, k in (("peace", "ask_peace"), ("an alliance", "ask_alliance"),
                                ("a cease-fire", "ask_ceasefire")) if d.get(k)]
        if w == "indifferent to" and not wants:
            continue
        t = f"{w} {o}" if w != "indifferent to" else f"indifferent to {o}"
        if wants:
            t += f", and would accept {' and '.join(wants)}"
        nb.append(t)
    if nb:
        L.append("  Neighbours: it is " + "; ".join(nb) + ".")
    return "\n".join(L), prov


def load(C, wid, lo, hi):
    recs = {}
    for line in gzip.open(C / wid / "world.jsonl.gz", "rt"):
        r = json.loads(line)
        if lo <= r["t"] <= hi:
            recs[r["t"]] = r
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data_v2/bench")
    ap.add_argument("--world"); ap.add_argument("--t1", type=int); ap.add_argument("--civ")
    ap.add_argument("--samples", type=int, default=0)
    ap.add_argument("--sources", default="C2", choices=("C0", "C1", "C2"))
    a = ap.parse_args()
    C = ROOT / a.corpus
    man = {w["world_id"]: w for w in json.load(open(C / "manifest.json"))["worlds"]}

    if a.samples:
        # pick a spread: worlds of different tier/ruleset, prefer anchors where a war is planned
        picks, seen_kind = [], collections.Counter()
        for wid, w in man.items():
            if len(picks) >= a.samples * 3:
                break
            meta = json.load(open(C / wid / "meta.json"))
            ck = [int(pathlib.Path(c).stem[1:]) for c in meta["branch_checkpoints"]]
            picks.append((wid, w, ck))
        shown = 0
        for wid, w, cks in picks:
            hz = HZ_BY_BUDGET[int(w["report_endturn"])]; h = hz[-1]
            recs = load(C, wid, min(cks) - 40, max(cks) + h)
            for t1 in cks:
                if t1 not in recs or shown >= a.samples:
                    continue
                civs = [p["name"] for p in recs[t1]["state"]["players"] if p.get("is_alive")]
                for ego in civs:
                    att = (recs[t1]["probes"].get("B_attitude") or {}).get(ego) or {}
                    war = any(isinstance(d.get("countdown"), int) and d["countdown"] >= 0
                              for o, d in att.items() if o != ego)
                    kind = "war" if war else "peace"
                    if seen_kind[kind] >= a.samples // 2 + 1 or shown >= a.samples:
                        continue
                    txt, prov = synth(recs[t1], recs, w, ego, t1, h)
                    print("=" * 92)
                    print(f"[{kind}] {wid}  t1={t1}  h=+{h}  ruleset={w['ruleset']} "
                          f"ailevel={w['ailevel']} civs={len(civs)}")
                    print("=" * 92)
                    print(txt)
                    print(f"\n  [核对，不进 prompt] 未来生产 {prov['future_prod']}")
                    print(f"  [核对] 可见窗口生产 {prov['visible_prod']}  漂移={prov['drift']}")
                    print(f"  [核对] countdown {prov['countdown']}  founder_want {prov['max_founder_want']} "
                          f"urgency {prov['max_urgency']}  wonder_city {prov['wonder_city_id']}")
                    print(f"  [核对] 目的地簇 {prov['axis']}\n")
                    seen_kind[kind] += 1; shown += 1
                    break
        return
    hz = HZ_BY_BUDGET[int(man[a.world]["report_endturn"])]; h = hz[-1]
    recs = load(C, a.world, a.t1 - 40, a.t1 + h)
    txt, prov = synth(recs[a.t1], recs, man[a.world], a.civ, a.t1, h, sources=a.sources)
    print(txt)


if __name__ == "__main__":
    main()
