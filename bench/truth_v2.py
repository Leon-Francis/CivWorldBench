"""Ground truth for the v2 tasks, from mainline records (1a/1b/1c) -- 09 §0.

All functions take `recs: dict[int, record]` covering the turns they need.  The v2
corpus stores one record per turn with the merged engine+diff event stream in
`rec["events"]`, so window questions ("at any turn between t1 and t1+h") are
evaluated on every turn, not on endpoints.
"""
from __future__ import annotations

from typing import Any

from bench.spec_v2 import (BASE_RATE_HI, BASE_RATE_LO, LEADER_MAX_GAP, MIN_CELL_N, REL_MAX_GAP, TEMPLATES_1B,
                           derive, mil_map_for, task_vars)
from sandboxes.freeciv.state import base_ruleset


def alive(rec: dict) -> list[dict]:
    """Living NON-barbarian players: the civilizations questions are asked about.  Barbarian and
    pirate players (probes.A_static.ai_barb_type != None) have no score, no research and can be
    respawned; the review run 2026-09-15 found a pirate among the 1a targets.  They still appear in
    the reports and still count in n_enemies (glossary)."""
    A = (rec.get("probes") or {}).get("A_static") or {}
    return [p for p in rec["state"]["players"] if p.get("is_alive")
            and ((A.get(p["name"]) or {}).get("ai_barb_type") or "None") == "None"
            and p.get("nation") not in ("Barbarian", "Pirate")]


def player(rec: dict, name: str) -> dict | None:
    return next((p for p in rec["state"]["players"] if p["name"] == name), None)


def slot_names(rec: dict) -> dict[str, str]:
    """`diplomacy` is keyed by player slot ("player3"); map to names."""
    return {f"player{i}": p["name"] for i, p in enumerate(rec["state"]["players"])}


def relations(rec: dict, civ: str) -> dict[str, str]:
    """{other civ name: diplstate} for one civ (contacted only)."""
    p = player(rec, civ)
    if p is None:
        return {}
    sn = slot_names(rec)
    return {sn[k]: v for k, v in (p.get("diplomacy") or {}).items()
            if k in sn and sn[k] != civ and v not in ("Never met", "No Contact")}


# ---------------------------------------------------------------- 1a --------
def mil_of(rec: dict) -> dict[str, bool]:
    return mil_map_for(base_ruleset(rec.get("ruleset") or "civ2civ3"))


def truth_1a(recs: dict, t1: int, hz: list[int], civs: list[str]) -> dict:
    mil = mil_of(recs[t1])
    out: dict[str, dict] = {}
    for h in hz:
        r = recs.get(t1 + h)
        out[str(h)] = {}
        for c in civs:
            p = player(r, c) if r else None
            lvl_t1 = derive(player(recs[t1], c), mil, recs[t1])
            out[str(h)][c] = task_vars(derive(p, mil, r) if p and p.get("is_alive") else None, lvl_t1)
    return out


# ---------------------------------------------------------------- 1b --------
def _events(recs: dict, a: int, b: int):
    """Events that happen between the state at turn a and the state at turn b.
    Engine events recorded at t happen during turn t (after the snapshot) -> records a .. b-1;
    diff events recorded at t describe the transition into state t (happened during t-1) ->
    records a+1 .. b.  (Both measured 2026-09-15; see build_bench_v2.events_of.)"""
    for t in range(a, b + 1):
        r = recs.get(t)
        if not r:
            continue
        for e in r.get("events") or []:
            te = t - 1 if e.get("src") == "diff" else t
            if a <= te < b:
                yield te, e


def items_1b(recs: dict, t1: int, hz: list[int], civs: list[str],
             cells: dict | None = None, budget: int | None = None) -> list[dict]:
    """Every askable (template, civ, h[, other]) item with its truth.

    `cells` (optional): the corpus cell table (bench/v2/cells_1b.json, keyed
    `kind|budget|h|t1` since 2026-09-14, values {"rate", "n"}) -- items whose corpus cell
    rate is outside 10-90% or rests on < MIN_CELL_N anchors are dropped (09 §2.4).  When
    None, everything is emitted (that is how the base rates themselves are measured).
    ⚠️ This is only the FIRST stage: build_bench_v2.finalize_1b re-applies the rule on the
    question set's own per-(set, cond) rates, which is what the scorer uses as reference.
    `leader` is exempt (multi-class, persistence reference)."""
    r1 = recs[t1]
    mil = mil_of(r1)
    items = []

    def ok(kind, h):
        if cells is None or kind == "leader":
            return True
        c = cells.get(f"{kind}|{budget}|{h}|{t1}") or cells.get(f"{kind}|{budget}|{h}")
        if isinstance(c, dict):
            br, n = c.get("rate"), c.get("n") or 0
        else:
            br, n = c, MIN_CELL_N
        return br is not None and n >= MIN_CELL_N and BASE_RATE_LO <= br <= BASE_RATE_HI

    for civ in civs:
        p1 = player(r1, civ)
        if not p1 or not p1.get("is_alive"):
            continue
        d1 = derive(p1, mil, r1)
        rel1 = relations(r1, civ)
        for h in hz:
            tt = t1 + h
            r2 = recs.get(tt)
            if r2 is None:
                continue
            p2 = player(r2, civ)
            d2 = derive(p2, mil, r2) if p2 else None
            base = {"civ": civ, "h": h, "t1": t1, "tt": tt}
            # ---- windows over every turn in (t1, tt]
            golds, mils, govs = [], [], []
            for t in range(t1 + 1, tt + 1):
                pt = player(recs[t], civ) if t in recs else None
                if pt is None:
                    continue
                golds.append(int(pt["gold"]))
                mils.append(sum(1 for u in (pt.get("units") or []) if mil.get(u.get("type") or u.get("name"), False)))
                govs.append(pt.get("government"))
            evs = list(_events(recs, t1, tt))
            founded = any(e["type"] == "city_founded" and e.get("player") == civ for _, e in evs)
            captured = any(e["type"] == "city_transferred" and e.get("player") == civ
                           and e.get("from_player") != civ for _, e in evs)
            lost = any((e["type"] == "city_destroyed" and e.get("player") == civ)
                       or (e["type"] == "city_transferred" and e.get("from_player") == civ) for _, e in evs)
            newally = any(e["type"] == "alliance_formed" and civ in (e.get("player"), e.get("other")) for _, e in evs)
            govch = any(e["type"] == "government_change" and e.get("player") == civ for _, e in evs) \
                or any(g != p1.get("government") for g in govs)
            wond = bool(p2) and len(p2.get("wonders") or []) > len(p1.get("wonders") or [])
            for kind, fam, min_h, text in TEMPLATES_1B:
                if h < min_h or not ok(kind, h):
                    continue
                if kind == "bankrupt":
                    items.append({**base, "kind": kind, "truth": any(g == 0 for g in golds),
                                  "text": text.format(civ=civ, t1=t1, tt=tt)})
                elif kind == "newcity":
                    items.append({**base, "kind": kind, "truth": founded, "text": text.format(civ=civ, t1=t1, tt=tt)})
                elif kind == "capture":
                    items.append({**base, "kind": kind, "truth": captured, "text": text.format(civ=civ, t1=t1, tt=tt)})
                elif kind == "lostcity":
                    items.append({**base, "kind": kind, "truth": lost, "text": text.format(civ=civ, t1=t1, tt=tt)})
                elif kind == "mildrop":
                    if d1["n_military_units"] >= 5 and mils:
                        items.append({**base, "kind": kind, "base": d1["n_military_units"],
                                      "truth": min(mils) <= 0.7 * d1["n_military_units"],
                                      "text": text.format(civ=civ, t1=t1, tt=tt, base=d1["n_military_units"])})
                elif kind == "wonder":
                    items.append({**base, "kind": kind, "truth": wond, "text": text.format(civ=civ, t1=t1, tt=tt)})
                elif kind == "moreenemies":
                    if d2 is not None:
                        items.append({**base, "kind": kind, "base": d1["n_enemies"],
                                      "truth": d2["n_enemies"] > d1["n_enemies"],
                                      "text": text.format(civ=civ, t1=t1, tt=tt, base=d1["n_enemies"])})
                elif kind == "newally":
                    items.append({**base, "kind": kind, "truth": newally, "text": text.format(civ=civ, t1=t1, tt=tt)})
                elif kind == "joinwar":
                    # opportunities at t1: ally B at war with C, civ not at war with C
                    if r2 is None:
                        continue
                    rel2 = relations(r2, civ)
                    seen = set()
                    for ally, st in rel1.items():
                        if st != "Alliance":
                            continue
                        for other, st2 in relations(r1, ally).items():
                            if st2 == "War" and other != civ and rel1.get(other) not in (None, "War") and other not in seen:
                                seen.add(other)
                                items.append({**base, "kind": kind, "ally": ally, "other": other,
                                              "truth": rel2.get(other) == "War",
                                              "text": text.format(civ=civ, ally=ally, other=other, t1=t1, tt=tt)})
                elif kind == "govchange":
                    items.append({**base, "kind": kind, "truth": govch, "text": text.format(civ=civ, t1=t1, tt=tt)})
                elif kind == "stillahead":
                    if d2 is None:
                        continue
                    for other in civs:
                        if other <= civ:      # each unordered pair once, civ = the one ahead
                            continue
                        po1 = player(r1, other); po2 = player(r2, other)
                        if not po1 or not po1.get("is_alive") or not po2:
                            continue
                        a, b = int(p1["score"]), int(po1["score"])
                        hi_, lo_ = (civ, other) if a > b else (other, civ)
                        sh, sl = max(a, b), min(a, b)
                        if sh == 0 or (sh - sl) / sh > REL_MAX_GAP or sh == sl:
                            continue
                        ph2 = player(r2, hi_); pl2 = player(r2, lo_)
                        items.append({**base, "kind": kind, "civ": hi_, "other": lo_, "base": sh, "base2": sl,
                                      "gap": (sh - sl) / sh,
                                      "truth": int(ph2["score"]) > int(pl2["score"]),
                                      "text": text.format(civ=hi_, other=lo_, t1=t1, tt=tt, base=sh, base2=sl)})
        # (loop over civs ends)
    # ---- leader question: one per (anchor, h)
    for h in hz:
        tt = t1 + h
        r2 = recs.get(tt)
        if r2 is None or not ok("leader", h):
            continue
        sc = sorted(((int(p["score"]), p["name"]) for p in alive(r1)), reverse=True)
        if len(sc) < 2 or sc[0][0] == 0 or (sc[0][0] - sc[1][0]) / sc[0][0] > LEADER_MAX_GAP:
            continue
        sc2 = sorted(((int(p["score"]), p["name"]) for p in alive(r2)), reverse=True)
        text = dict((k, t) for k, _, _, t in TEMPLATES_1B)["leader"]
        items.append({"kind": "leader", "h": h, "t1": t1, "tt": tt, "civ": "*",
                      "civs": [n for _, n in sc], "truth": sc2[0][1],
                      "text": text.format(tt=tt)})
    return items


# ---------------------------------------------- 1b truth under an intervention -----
def _mil_count(p: dict, mil) -> int:
    return sum(1 for u in (p.get("units") or []) if mil.get(u.get("type") or u.get("name"), False))


def _city_owner_map(rec: dict) -> dict[str, str]:
    return {c["name"]: p["name"] for p in rec["state"]["players"] for c in (p.get("cities") or [])}


def truth_1b_from_states(items: list[dict], states: dict[int, dict], rec_t1: dict, mil) -> list:
    """Re-evaluate the mainline 1b items on an arm's states (saves every 10 turns).

    Windows are sampled at 10-turn resolution here (the arm has no per-turn record),
    so brief events between samples can be missed -- stated in the manifest as the
    task-3 truth resolution.  Same questions, same wording, different world."""
    t1 = rec_t1["t"]
    own1 = _city_owner_map(rec_t1)
    out = []
    for it in items:
        tt = it["tt"]; civ = it["civ"]
        samples = [t for t in sorted(states) if t1 < t <= tt]
        r2 = states.get(tt)
        p1 = player(rec_t1, civ) if civ != "*" else None
        p2 = player(r2, civ) if (r2 is not None and civ != "*") else None
        kind = it["kind"]
        if kind == "leader":
            if r2 is None:
                out.append(None); continue
            sc2 = sorted(((int(p["score"]), p["name"]) for p in alive(r2)), reverse=True)
            out.append(sc2[0][1] if sc2 else None); continue
        if r2 is None or p1 is None:
            out.append(None); continue
        if kind == "bankrupt":
            out.append(any(int(player(states[t], civ)["gold"]) == 0 for t in samples if player(states[t], civ)))
        elif kind == "newcity":
            names1 = set(own1)
            out.append(any(c["name"] not in names1 for c in (p2.get("cities") or [])) if p2 else False)
        elif kind == "capture":
            got = False
            prev = own1
            for t in samples:
                cur = _city_owner_map(states[t])
                if any(o == civ and prev.get(n) not in (None, civ) for n, o in cur.items()):
                    got = True; break
                prev = cur
            out.append(got)
        elif kind == "lostcity":
            lost = False; prev = own1
            for t in samples:
                cur = _city_owner_map(states[t])
                if any(o == civ and cur.get(n) != civ for n, o in prev.items()):
                    lost = True; break
                prev = cur
            out.append(lost)
        elif kind == "mildrop":
            base = it.get("base") or _mil_count(p1, mil)
            vals = [_mil_count(player(states[t], civ), mil) for t in samples if player(states[t], civ)]
            out.append(bool(vals) and min(vals) <= 0.7 * base)
        elif kind == "wonder":
            out.append(bool(p2) and len(p2.get("wonders") or []) > len(p1.get("wonders") or []))
        elif kind == "moreenemies":
            out.append(bool(p2) and derive(p2, mil, r2)["n_enemies"] > derive(p1, mil, rec_t1)["n_enemies"])
        elif kind == "newally":
            allies1 = {o for o, s in relations(rec_t1, civ).items() if s == "Alliance"}
            got = False
            for t in samples:
                if any(s == "Alliance" and o not in allies1 for o, s in relations(states[t], civ).items()):
                    got = True; break
            out.append(got)
        elif kind == "joinwar":
            out.append(relations(r2, civ).get(it["other"]) == "War")
        elif kind == "govchange":
            g1 = p1.get("government")
            out.append(any((player(states[t], civ) or {}).get("government") not in (None, g1) for t in samples))
        elif kind == "stillahead":
            po2 = player(r2, it["other"])
            out.append(bool(p2 and po2) and int(p2["score"]) > int(po2["score"]))
        else:
            out.append(None)
    return out
