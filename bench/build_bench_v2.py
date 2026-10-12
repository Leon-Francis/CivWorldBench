"""v2/v3 question builder: one file per (world, anchor) under bench/runs/<run>/questions/.

Anchors are the branch points recorded in each world's meta (4 per world, in
[60, report_endturn]); horizons follow the world's turn budget (spec_v2).  Conditions:
the two main views on every anchor, four ablations on world subsets (09 §6.2).

Tasks per anchor and condition (09 §8.7, revised 2026-09-14/15 = v3, see 14/15 号文档):
  1a  11 quantities x all civs x horizons, quantiles           truth: mainline records
  1b  templates whose (kind, budget, h, t1) corpus rate is 10-90% AND whose per-(set, cond)
      in-sample rate is 10-90% on >= 200 items (second stage: finalize_1b)
                                                               truth: mainline records/events
  2   personality RANKING per dimension x rankable civs, attitude class x pairs,
      rule bins only where the window holds evidence               truth: probes / manifest
      (the only task WITHOUT the STRATEGIES block: plans/attitudes are its targets)
  ★ v3 conditioning: a STRATEGIES block (each AI's t1 plan, C1 of doc 13) follows the frames
      in the 1a / 1b / 3 prompts -- all civs in view 1, the viewer alone in view 2
  3   the anchor's sampled operator + its placebo (+ a no-intervention control when it
      differs from the main 1a/1b): ONE prompt per sub-task, only in-band horizons
                                                               truth: arm savegames (arm_state)
  (1c retired 2026-09-12.)

    .venv/bin/python bench/build_bench_v2.py --corpus data_v2/bench --run v3-bench --jobs 40
"""
from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "bundle")); sys.path.insert(0, str(ROOT / "tools"))

from sandboxes.freeciv.state import base_ruleset                       # noqa: E402
from bench import tasks_v2 as T                                        # noqa: E402
from bench.arm_state import arm_records, arm_records_light, arm_save_turns   # noqa: E402
from bench.interventions_v2 import Unparseable, describe_and_edit      # noqa: E402
from bench.render import build_A, build_H, render_stale_report, stat_report  # noqa: E402
from bench.spec_v2 import (ABLATION_WORLDS, BASE_RATE_HI, BASE_RATE_LO, CONDITIONS, HZ_BY_BUDGET,   # noqa: E402
                           LOVE_BINS, MIN_CELL_N, OPS, PARAM_BINS, PARAM_EVIDENCE, PARAMS_ASKED, TRAITS,
                           VARS, SCORE_VARS, bin_of, frames_for, hz_task3, met_civs, op_base_of, task_vars)
from bench.truth_v2 import (alive, items_1b, mil_of, player, relations,  # noqa: E402
                            truth_1a, truth_1b_from_states)
from bench.strategy_v3 import strategy_block  # noqa: E402

CELLS = None


def load_cells():
    global CELLS
    if CELLS is None:
        p = ROOT / "bench/v2/cells_1b.json"          # keyed kind|budget|h|t1 -> {"rate","n"} (2026-09-14)
        CELLS = json.load(open(p)) if p.exists() else {}
    return CELLS


def load_recs(wd: Path, lo: int, hi: int) -> dict:
    recs = {}
    with gzip.open(wd / "world.jsonl.gz", "rt") as f:
        for l in f:
            d = json.loads(l)
            if lo <= d["t"] <= hi:
                recs[d["t"]] = d
            if d["t"] > hi:
                break
    return recs


def events_of(recs: dict, lo: int, hi: int) -> list[dict]:
    """The full event stream (engine + diff) with `turn` set.  interval_v2 needs the per-unit
    events (unit_built, action_finished_*, unit_lost); the oracle condition filters to
    ORACLE_EVENT_TYPES itself."""
    ev = []
    for t in range(lo, hi + 1):
        r = recs.get(t)
        if r:
            for e in r.get("events") or []:
                e = {k: (_demojibake(v) if isinstance(v, str) else v) for k, v in e.items()}
                # two timing conventions in one stream (measured 2026-09-15): an ENGINE event recorded at
                # turn t happened during turn t (after the turn-t snapshot); a DIFF event recorded at t
                # describes the transition INTO state t, i.e. it happened during turn t-1.  Normalise
                # everything to "the turn during which it happened".
                e["turn"] = t - 1 if e.get("src") == "diff" else t
                if e.get("src") == "diff":
                    e.pop("engine_turn", None)
                ev.append(e)
    return ev


def _demojibake(s: str) -> str:
    """Engine-log names arrive double-encoded (`JÃ³zef PiÅ\x82sudski`, `BiaÅ\x82ystok`) while the frames
    spell them right, so the narrative treated one civ as two and printed self-attacks (review run
    2026-09-15).  Undo the Latin-1 re-decode when it round-trips to valid UTF-8."""
    if s.isascii():
        return s
    try:
        fixed = s.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return s
    return fixed if fixed != s else s


def median_civ(rec: dict) -> str:
    sc = sorted(alive(rec), key=lambda p: int(p["score"]))
    return sc[len(sc) // 2]["name"]


def param_evidence(ev: list[dict], lo: int, t1: int, civs: set[str]) -> dict[str, int]:
    """Evidence counts in the observation window (lo, t1] for the world-rule questions:
    city size changes (visible as size differences between the frames' city tables) and
    technologies completed (narrated in the INTERVAL block) -- restricted to `civs` (all civs in
    the omniscient view; the viewer alone in the POMDP view, whose interval and city tables only
    cover what it can know).  Plague is not counted: it is neither narrated nor rendered."""
    n = {"foodbox": 0, "sciencebox": 0}
    for e in ev:
        t = e.get("turn")
        if t is None or not (lo <= t < t1) or e.get("player") not in civs:
            continue
        if e.get("type") == "city_size_change":
            n["foodbox"] += 1
        elif e.get("type") == "tech_discovered":
            n["sciencebox"] += 1
    return n


def _task2(rec: dict, world: dict, civs: list[str], evidence: dict[str, int],
           scope: list[str] | None = None) -> tuple[str, dict, list, list, list]:
    """-> (question, truth, rankable civs, pairs, params asked).

    truth["traits"] = {dim: {civ: effective value}} (v3 ranking; the scorer derives the pairwise
    order); truth["love"] / truth["params"] unchanged from v2."""
    if scope is not None:
        civs = [c for c in civs if c in scope]
    A = rec["probes"].get("A_static") or {}; Batt = rec["probes"].get("B_attitude") or {}
    t2_civs = [c for c in civs if A.get(c) and A[c].get("traits") and (A[c].get("ai_barb_type") or "None") == "None"]
    truth_tr = ({d: {c: int(A[c]["traits"][d]["effective"]) for c in t2_civs} for d in TRAITS}
                if len(t2_civs) >= 2 else {})
    pairs, truth_love = [], {}
    for a in t2_civs:
        for b, st in sorted(relations(rec, a).items()):
            if b not in t2_civs:
                continue
            lv = ((Batt.get(a) or {}).get(b) or {}).get("love")
            if lv is None:
                continue
            pairs.append((a, b)); truth_love[f"{a}->{b}"] = ["hostile", "neutral", "friendly"][bin_of(lv, LOVE_BINS)]
    # cap 24, round-robin over source civs so no civ is dropped wholesale (review run 2026-09-15)
    by_src = collections.defaultdict(list)
    for a_, b_ in pairs:
        by_src[a_].append((a_, b_))
    rr = []
    while len(rr) < 24 and any(by_src.values()):
        for a_ in list(by_src):
            if by_src[a_] and len(rr) < 24:
                rr.append(by_src[a_].pop(0))
    pairs = rr; truth_love = {f"{a}->{b}": truth_love[f"{a}->{b}"] for a, b in pairs}
    params = list(PARAMS_ASKED)                          # v3: `_illness` never asked (no observable evidence)
    truth_par = {}
    for pn in params:
        rule = PARAM_EVIDENCE.get(pn)
        if rule is not None and evidence.get(pn, 0) < rule[1]:
            continue                                     # v3: no evidence in the window -> not asked
        val = (world.get("params") or {}).get(pn)
        if val is None:
            continue
        truth_par[pn] = bin_of(float(val), PARAM_BINS[pn])
    params = [p for p in params if p in truth_par]
    rank_civs = t2_civs if len(t2_civs) >= 2 else []
    return (T.q_2(rank_civs, pairs, params, rec["t"]),
            {"traits": truth_tr, "love": truth_love, "params": truth_par}, rank_civs, pairs, params)


def _arms_at(wd: Path, t1: int, arms_dir: str = "arms") -> dict:
    out = {}
    for f in (wd / arms_dir).glob(f"t{t1}_*.json"):
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        if d.get("status") not in ("OK", "OK_SHORT"):
            continue
        # key by the FILE name, not the json's `arm` field: the +90 trait placebos are symlinks to the
        # +45 placebo files (identical +3 edit), whose json still says `L4-trait-aggr__placebo`
        # (2026-09-15: keying by `arm` lost 239 of 265 trait anchors on the first +90 build)
        name = f.stem.split("_", 1)[1]
        d = dict(d, arm=name)
        out[name] = (d, f.with_suffix(""))
    return out


_GOVTECH = re.compile(r"(?:GOVTECH|GIVETECH)_(?P<tech>[A-Za-z][A-Za-z .'_-]*?)=")


def _pick_real(arms: dict, wid: str, t1: int, budget: int, rec_t1: dict | None = None):
    """The anchor's real operator: among the OPS arms whose real AND placebo reached the
    operator's own in-band horizon, a deterministic hash pick (several may exist after the
    NA-fallback re-sampling and the single-dimension trait re-collection).  Returns
    (arm_json, arm_dir, hz3) or None."""
    cands = []
    for name, (aj, d) in sorted(arms.items()):
        op = aj.get("op") or {}
        if not op or aj.get("placebo_of") or op_base_of(op.get("id") or "") not in OPS or "__placebo" in name:
            continue
        hz3 = hz_task3(budget, op.get("horizon_band") or (0, 10**9))
        if not hz3:
            continue
        if op_base_of(op.get("id")) == "L2-gov-indirect" and rec_t1 is not None:
            # review run 2026-09-15: "acquires Monarchy" for a civ that already knew every technology --
            # the arm is a no-op and cannot be a counterfactual
            m = _GOVTECH.search(op.get("applied") or "")
            tg = next((q for q in rec_t1["state"]["players"] if q["name"] == op.get("target")), None)
            if m and tg and m["tech"].strip().replace("_", " ") in (tg.get("techs") or []):
                continue
        need = t1 + max(hz3)
        pl = arms.get(name + "__placebo")
        if pl and int(aj.get("turn_reached") or 0) >= need and int(pl[0].get("turn_reached") or 0) >= need:
            cands.append((aj, d, hz3))
    if not cands:
        return None
    i = int(hashlib.sha1(f"{wid}|{t1}".encode()).hexdigest()[:8], 16) % len(cands)
    return cands[i]


def build_anchor(args) -> dict:
    corpus, world, t1, run_dir, conds, tasks = args[:6]
    arms_dir = args[6] if len(args) > 6 else "arms"        # "arms_pt": per-turn saves (plan B, 14 §B16-2)
    do1, do2, do3 = "1" in tasks, "2" in tasks, "3" in tasks
    C = Path(corpus); wd = C / world["world_id"]; wid = world["world_id"]
    dst = Path(run_dir) / "questions" / f"{wid}_{t1}.jsonl.gz"
    if dst.exists():
        return {"world": wid, "t1": t1, "skipped": True}
    budget = int(world["report_endturn"]); hz = list(HZ_BY_BUDGET[budget])
    frames = frames_for(t1)
    need_k5 = "k5" in conds and t1 >= 161
    lo = frames_for(t1, 5)[0] if need_k5 else frames[0]
    recs = load_recs(wd, lo, t1 + max(hz))
    if t1 not in recs:
        return {"world": wid, "t1": t1, "error": "records missing"}
    for r in recs.values():
        r["_world"] = world            # render.with_cost: research line `[accumulated/needed]` needs sciencebox + ailevel
    # a world that ended (conquest) before t1 + max(hz) keeps the anchor; horizons past the
    # end simply have no truth (1a lines None, 1b items skipped)
    rs = base_ruleset(recs[t1]["ruleset"])
    civs = [p["name"] for p in alive(recs[t1])]
    ev = events_of(recs, lo, t1 + max(hz))          # record lo's events belong to the first interval (see render.build_H)
    arms = _arms_at(wd, t1, arms_dir)
    real = _pick_real(arms, wid, t1, budget, recs[t1]) if do3 else None
    viewer = real[0]["op"]["target"] if real else median_civ(recs[t1])

    def viewer_ok(v: str) -> bool:
        # the viewer must exist (alive, with a belief probe) in every observation frame: a civ
        # born from a civil war after frames[0] has no stale report to render there
        return all(v in ((recs[t].get("probes") or {}).get("C_belief") or {})
                   and any(p["name"] == v and p.get("is_alive") for p in recs[t]["state"]["players"])
                   for t in frames if t in recs)
    if viewer not in civs or not viewer_ok(viewer):
        ok = sorted((p for p in alive(recs[t1]) if viewer_ok(p["name"])), key=lambda p: int(p["score"]))
        viewer = ok[len(ok) // 2]["name"] if ok else median_civ(recs[t1])
    cells = load_cells()
    truth1a = truth_1a(recs, t1, hz, civs)
    mil0 = mil_of(recs[t1])
    from bench.spec_v2 import derive as _derive
    lvl_t1 = {cv: _derive(player(recs[t1], cv), mil0, recs[t1]) for cv in civs}
    base1a = {cv: task_vars(lvl_t1[cv], lvl_t1[cv]) for cv in civs}         # persistence: levels at t1, increments 0
    from bench.render import pomdp_known
    base_pomdp = pomdp_known(recs[t1], viewer, civs, mil0)                  # view 2: what the viewer can read off
    # values at the observation frames (trend-extrapolation baseline in score_v2)
    base_hist = {cv: {t: (_derive(player(recs[t], cv), mil0, recs[t]) if (player(recs[t], cv) or {}).get("is_alive") else None)
                      for t in frames if t in recs} for cv in civs}
    # view 2: the trend baseline may only use what the viewer could see at each frame
    hist_pomdp = {t: pomdp_known(recs[t], viewer, civs, mil0) for t in frames if t in recs and viewer in ((recs[t].get("probes") or {}).get("C_belief") or {})}
    base_hist_pomdp = {cv: {t: hist_pomdp[t].get(cv) for t in hist_pomdp} for cv in civs}
    items = items_1b(recs, t1, hz, civs, cells=cells, budget=budget)
    evid = param_evidence(ev, frames[0], t1, set(civs))
    q2, truth2, t2_civs, pairs, params2 = _task2(recs[t1], world, civs, evid)
    # POMDP set (view 2): everything is asked about V(t1) = viewer + civs it has met
    V = [c for c in met_civs(recs[t1], viewer) if c in civs]      # met civs minus barbarians (alive() excludes them)
    truth1a_P = {h: {c: v for c, v in d.items() if c in V} for h, d in truth1a.items()}
    # review run 2026-09-15: `mildrop` / `moreenemies` print the civ's military count / enemy count and
    # `stillahead` prints both scores -- numbers the viewer's stale report does not show for foreign
    # civs.  In the POMDP view those three templates are asked only about the viewer itself
    # (`stillahead` never: it always reveals a foreign score).
    def _pomdp_ok(i):
        if i["civ"] == "*":
            return all(c in V for c in i.get("civs", []))
        if i["civ"] not in V or i.get("other", i["civ"]) not in V or i.get("ally", i["civ"]) not in V:
            return False
        if i["kind"] == "stillahead":
            return False
        if i["kind"] in ("mildrop", "moreenemies") and i["civ"] != viewer:
            return False
        return True
    items_P = [i for i in items if _pomdp_ok(i)]
    evid_P = param_evidence(ev, frames[0], t1, {viewer})
    q2_P, truth2_P, t2_civs_P, pairs_P, params2_P = _task2(recs[t1], world, civs, evid_P, scope=V)

    # ---- task 3 data (shared by conditions)
    arm_pack = None
    if real and do3:
        a_real, d_real, hz3 = real
        arm_id = a_real["arm"]                                 # e.g. L4-trait-aggr90
        oid = op_base_of(a_real["op"]["id"])                   # benchmark operator name, e.g. L4-trait-aggr
        tgt = a_real["op"].get("target")                       # task 3 view-2: the intervened civ,
        if tgt and viewer_ok(tgt):                             # but only if it is observable in every frame
            viewer = tgt                                       # (a civ born from a civil war is not)
        plc = arms.get(arm_id + "__placebo"); fac = arms.get("factual"); chk = arms.get("factual_check")
        # the anchor's two identical control reruns must agree, else no effect size from it means
        # anything.  `fac` is deliberately NOT required to reach the far horizon: horizons it misses
        # give None cells and score_v2 skips them.  Compare only the turns the arm actually resolves
        # at -- exactly what `branch.reduce_world` does (comparing every saved turn also flags the
        # terminal game-over save, which excluded 1,549 of 1,900 anchors on the first try).
        det_ok = False
        if fac and chk:
            fd = fac[0].get("digests") or {}; cd = chk[0].get("digests") or {}
            want = {str(t) for t in (fac[0].get("resolved_at_turn") or {}).values()}
            shared = set(fd) & set(cd) & (want or set(fd))
            det_ok = bool(shared) and all(fd[t] == cd[t] for t in shared)
        if plc and fac and det_ok:
            turns = sorted({t1 + h for h in hz3} | {t for t in range(t1 + 10, t1 + max(hz3) + 1, 10)})
            packs = {}
            for name, (aj, adir) in (("real", (a_real, d_real)), ("placebo", plc), ("factual", fac)):
                st = arm_records(Path(str(adir)), wid, turns, with_probes=True)
                for r in st.values():
                    r["_world"] = world
                if arms_dir != "arms":
                    # per-turn arms: light state records at every other turn in (t1, t1+max h], so the
                    # 1b window items ("at any turn between ...") are evaluated per turn like the main 1b
                    rest = [t for t in range(t1 + 1, t1 + max(hz3) + 1) if t not in st]
                    st.update(arm_records_light(Path(str(adir)), wid, rest))
                packs[name] = {"json": aj, "states": st}
            arm_pack = (oid, packs, hz3)
    # mainline truth at the task-3 horizons (h=30 etc. are not in the main 1a grid)
    truth1a_3 = truth_1a(recs, t1, arm_pack[2], civs) if arm_pack else None

    def arm_truth_1a(states, hz_, civs_c):
        from bench.spec_v2 import derive
        out = {str(h): {} for h in hz_}
        for h in hz_:
            r = states.get(t1 + h)
            for cv in civs_c:
                p = player(r, cv) if r else None
                out[str(h)][cv] = task_vars(derive(p, mil0, r) if (p and p.get("is_alive")) else None, lvl_t1[cv])
        return out

    # ---- v3 strategy conditioning (C1, doc 13 §5; user 2026-09-15): every civ's t1 plan in the
    # omniscient view, the viewer's own plan in the POMDP view.  Goes into the PREDICTION tasks
    # (1a, 1b, 3) only -- task 2 asks for hidden state and must not see plans/attitudes (13 §5.3).
    strat_all, strat_prov = strategy_block(recs[t1], world, civs, 1)
    strat_v, _ = strategy_block(recs[t1], world, [viewer], 2)
    rows = []
    for cid in conds:
        c = CONDITIONS[cid]; view = c["view"]; primer = rs if c["primer"] else ""
        if c["k"] == 5 and t1 < 161:
            continue
        fr = frames if c["k"] == 3 else ([t1] if c["k"] == 1 else frames_for(t1, 5))
        pomdp = view == 2
        civs_c = V if pomdp else civs
        truth1a_c = truth1a_P if pomdp else truth1a
        items_c = items_P if pomdp else items
        q2_c, truth2_c, t2_civs_c, pairs_c, params_c = (q2_P, truth2_P, t2_civs_P, pairs_P, params2_P) if pomdp else (q2, truth2, t2_civs, pairs, params2)
        H, st = build_H(recs, fr, viewer, view, ev, ruleset=primer, world=world)
        body = H
        if c["oracle"]:
            # same news style as INTERVAL (user 2026-09-10), civ-level sentences only
            from bench.interval_v2 import interval_block
            A = interval_block(recs, t1, t1 + max(hz), ev, mil0, view, viewer, rs, world, major_only=True)
            body += (f"\n===== FUTURE EVENTS (oracle)   turn {t1+1}..{t1+max(hz)}   "
                     + ("(all civilizations)" if view == 1 else f"(only what {viewer} can know)") + " =====\n" + A + "\n")
        body_s = body + "\n" + (strat_v if pomdp else strat_all)         # prediction tasks: observations + plans
        common = dict(world=wid, set=world.get("split"), tier=world["tier"], budget=budget, t1=t1, cond=cid,
                      view=view, viewer=(viewer if view == 2 else None), k=c["k"], primer=c["primer"],
                      oracle=c["oracle"], think=c["think"], n_events=st["n_events"], n_events_hidden=st["n_events_hidden"],
                      scope=(V if pomdp else None), pomdp=pomdp, strategy="C1")
        bases = dict(base={cv: base1a[cv] for cv in civs_c},
                     base_pomdp=({cv: base_pomdp.get(cv, {}) for cv in civs_c} if pomdp else None),
                     base_hist={cv: base_hist[cv] for cv in civs_c},                  # omniscient history: ONE trend reference for both views
                     base_hist_visible=({cv: base_hist_pomdp[cv] for cv in civs_c} if pomdp else None))   # what the viewer saw (visible/hidden split)

        def emit(task, question, truth, max_tok, **extra):
            b = body if task == "2" else body_s                          # task 2 never sees the plans
            rows.append({**common, "task": task, "prompt": b + "\n" + question, "truth": truth,
                         "max_tokens": int(max_tok), **({"strategy": None} if task == "2" else {}), **extra})

        def strip_items(its):
            return [{k: v for k, v in i.items() if k != "truth"} for i in its]

        if do1:
            emit("1a", T.q_1a(civs_c, t1, hz), truth1a_c, len(civs_c) * len(VARS) * len(hz) * 32 * 1.6, horizons=hz, civs=civs_c, **bases)
        if items_c and do1:
            emit("1b", T.q_1b(items_c), [i["truth"] for i in items_c], len(items_c) * 16 * 1.6, items=strip_items(items_c))
        if not c["oracle"]:
            if do2 and (t2_civs_c or pairs_c or params_c):
                emit("2", q2_c, truth2_c, (4 * (len(t2_civs_c) + 2) + len(pairs_c) + len(params_c)) * 40 + 200,
                     civs=t2_civs_c, pairs=pairs_c, params=params_c,
                     param_evidence=(evid_P if pomdp else evid))
            if arm_pack:
                oid, packs, hz3 = arm_pack
                items3 = [i for i in items_c if i["h"] in hz3]        # in-band 1b items (same wording as the main 1b)
                same_1a = hz3 == hz                                    # control 1a byte-identical to the main 1a?
                same_1b = len(items3) == len(items_c)
                truth1a_3c = {str(h): {cv: v for cv, v in d.items() if cv in civs_c} for h, d in truth1a_3.items()}
                truth_fac_1a = arm_truth_1a(packs["factual"]["states"], hz3, civs_c)
                t1b_fac = truth_1b_from_states(items3, packs["factual"]["states"], recs[t1], mil0) if items3 else []
                meta3 = dict(op_base=oid, layer=(packs["real"]["json"]["op"].get("layer")),
                             op_target=packs["real"]["json"]["op"].get("target"), op_other=packs["real"]["json"]["op"].get("other"),
                             horizon_band=packs["real"]["json"]["op"].get("horizon_band"), score_vars=list(SCORE_VARS.get(oid, ())),
                             horizons=hz3, civs=civs_c, control_ref={"1a": "main" if same_1a else "control", "1b": "main" if same_1b else "control"})
                # -- the no-intervention control, same format, only where it differs from the main grid
                if not same_1a:
                    rows.append({**common, **meta3, **bases, "task": "3", "sub": "1a", "arm_kind": "control", "op": oid, "is_placebo": False,
                                 "channel": None, "intervention": None,
                                 "prompt": body_s + "\n" + T.q_1a(civs_c, t1, hz3),
                                 "truth": {"1a": truth1a_3c}, "factual_arm": {"1a": truth_fac_1a},
                                 "max_tokens": int(len(civs_c) * len(VARS) * len(hz3) * 32 * 1.6)})
                if items3 and not same_1b:
                    rows.append({**common, **meta3, "task": "3", "sub": "1b", "arm_kind": "control", "op": oid, "is_placebo": False,
                                 "channel": None, "intervention": None,
                                 "prompt": body_s + "\n" + T.q_1b(items3), "items": strip_items(items3),
                                 "truth": {"1b": [i["truth"] for i in items3]}, "factual_arm": {"1b": t1b_fac},
                                 "max_tokens": int(len(items3) * 16 * 1.6)})
                for name in ("real", "placebo"):
                    aj = packs[name]["json"]; states = packs[name]["states"]
                    try:
                        ed, desc, channel = describe_and_edit(recs[t1], aj["op"])
                    except Unparseable as e:
                        rows.append({**common, "task": "3", "error": f"unparseable: {e}", "op": aj["op"]["id"]})
                        continue
                    pre = body_s
                    if ed is not None:
                        b0 = stat_report(recs[t1], rs, False) if view == 1 else render_stale_report(recs[t1], viewer, False, rs)
                        b1 = stat_report(ed, rs, False) if view == 1 else render_stale_report(ed, viewer, False, rs)
                        if b0 in body_s and b1 != b0:
                            pre = body_s.replace(b0, b1, 1)
                        else:
                            channel = "text"      # last frame not found verbatim (view-2 quirk); fall back
                    head = pre + "\n" + T.q_3_preamble(desc, t1) + "\n\n"
                    arm_meta = dict(meta3, op=aj["op"]["id"], is_placebo=(name == "placebo"), arm_kind=name,
                                    channel=channel, intervention=desc)
                    rows.append({**common, **arm_meta, **bases, "task": "3", "sub": "1a",
                                 "prompt": head + T.q_1a(civs_c, t1, hz3),
                                 "truth": {"1a": arm_truth_1a(states, hz3, civs_c)},
                                 "factual_arm": {"1a": truth_fac_1a}, "mainline": {"1a": truth1a_3c},
                                 "max_tokens": int(len(civs_c) * len(VARS) * len(hz3) * 32 * 1.6)})
                    if items3:
                        rows.append({**common, **arm_meta, "task": "3", "sub": "1b",
                                     "prompt": head + T.q_1b(items3), "items": strip_items(items3),
                                     "truth": {"1b": truth_1b_from_states(items3, states, recs[t1], mil0)},
                                     "factual_arm": {"1b": t1b_fac}, "mainline": {"1b": [i["truth"] for i in items3]},
                                     "max_tokens": int(len(items3) * 16 * 1.6)})
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp")
    with gzip.open(tmp, "wt") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.rename(dst)
    return {"world": wid, "t1": t1, "rows": len(rows), "by_task": dict(collections.Counter(r["task"] + (r.get("sub") or "") for r in rows)),
            "n_1b": len(items), "task3": bool(arm_pack), "op": (arm_pack[0] if arm_pack else None), "viewer": viewer,
            "n_war_plans": sum(1 for v in strat_prov.values() if v["countdown"]),
            "prompt_chars": int(sum(len(r.get("prompt", "")) for r in rows) / max(1, len(rows)))}


def ablation_subsets(worlds: list[dict], all_basic: bool = False) -> dict[str, set]:
    """Ablation conditions run on deterministic subsets of the basic set, spread over budgets.

    `all_basic` (--ablation_all, 2026-09-12 user request): emit every ablation condition for EVERY
    basic world instead of the 60/60/40/40-world subsets, so the ablations sit on the same footing
    as the main table.  Build it into a separate run dir -- the frozen question sets must not move.
    """
    basic = sorted((w for w in worlds if w.get("split") == "bench-basic"), key=lambda w: w["world_id"])
    if all_basic:
        ids = {w["world_id"] for w in basic}
        return {cid: set(ids) for cid in ABLATION_WORLDS}
    by = collections.defaultdict(list)
    for w in basic:
        by[int(w["report_endturn"])].append(w["world_id"])
    out = {}
    budgets = sorted(by)
    for cid, n in ABLATION_WORLDS.items():
        per, rem = divmod(n, len(budgets))
        out[cid] = {w for i, b in enumerate(budgets) for w in by[b][:per + (1 if i < rem else 0)]}
    return out


def finalize_1b(run_dir: Path, jobs: int = 40) -> dict:
    """Second-stage 1b filter (v3, user 2026-09-14): re-apply the 10-90% rule on the question set's
    OWN per-(set, cond, template, budget, h, t1) rates, require MIN_CELL_N items per cell, and
    rewrite every 1b prompt (main grid and task 3) without the dropped items.  Writes the cell
    table the scorer reads (bench/v2/cells_1b_insample_<run>.json).

    Why a second stage: the corpus table (stage 1, cells_1b.json) is measured on all 475 worlds'
    mainline; the generalisation set is a different 250 worlds and the POMDP view asks only about
    the viewer's neighbours, so the per-setting rates differ (residual 0.030-0.042) and a cell
    legal in the corpus can be near-certain in one setting.  The reference the scorer uses IS the
    in-sample rate, so the filter must be applied to the same table."""
    qdir = run_dir / "questions"
    files = sorted(qdir.glob("*.jsonl.gz"))
    acc = collections.defaultdict(lambda: [0, 0])
    for f in files:
        for l in gzip.open(f, "rt"):
            r = json.loads(l)
            if r.get("task") != "1b" or r.get("error"):
                continue
            for it, y in zip(r["items"], r["truth"]):
                if it["kind"] == "leader" or y is None:
                    continue
                k = (r["set"], r["cond"], it["kind"], str(r["budget"]), str(it["h"]), str(r["t1"]))
                acc[k][0] += int(bool(y)); acc[k][1] += 1
    cells = {k: {"rate": y / n, "n": n} for k, (y, n) in acc.items() if n}
    run = run_dir.name
    (ROOT / f"bench/v2/cells_1b_insample_{run}.json").write_text(
        json.dumps({"|".join(k): v for k, v in cells.items()}, indent=1))

    corpus = load_cells()
    fallback = collections.Counter()

    def ref_cell(r, it):
        """The reference cell the scorer will use (mirror of score_v2.cell_1b): the (set, cond) in-sample
        cell -- ablation conditions share the (set, v1) cell -- when it has >= MIN_CELL_N items, else the
        corpus (kind|budget|h|t1) cell."""
        cond_ref = r["cond"] if r["cond"] in ("v1", "v2") else "v1"
        c = cells.get((r["set"], cond_ref, it["kind"], str(r["budget"]), str(it["h"]), str(r["t1"])))
        if c and c["n"] >= MIN_CELL_N:
            return c, "insample"
        return corpus.get(f"{it['kind']}|{r['budget']}|{it['h']}|{r['t1']}"), "corpus"

    def legal(r, it):
        if it["kind"] == "leader":
            return True
        c, src = ref_cell(r, it)
        fallback[src] += 1
        return bool(c) and (c.get("n") or 0) >= MIN_CELL_N and BASE_RATE_LO <= c["rate"] <= BASE_RATE_HI

    stats = collections.Counter(); dropped_cells = collections.Counter()
    for f in files:
        rows = [json.loads(l) for l in gzip.open(f, "rt")]
        out = []; changed = False
        for r in rows:
            is_1b = (r.get("task") == "1b") or (r.get("task") == "3" and r.get("sub") == "1b")
            if r.get("error") or not is_1b:
                out.append(r); continue
            keep = [i for i, it in enumerate(r["items"]) if legal(r, it)]
            stats["items_before"] += len(r["items"]); stats["items_after"] += len(keep)
            for i, it in enumerate(r["items"]):
                if i not in keep:
                    dropped_cells[(it["kind"], r["budget"], it["h"], r["t1"])] += 1
            if len(keep) == len(r["items"]):
                out.append(r); continue
            changed = True
            if not keep:
                stats["rows_dropped"] += 1; continue
            items = [r["items"][i] for i in keep]
            head = r["prompt"].split("=====  TASK 1B")[0]
            r["prompt"] = head + T.q_1b(items)
            r["items"] = items
            r["max_tokens"] = int(len(items) * 16 * 1.6)
            if r["task"] == "1b":
                r["truth"] = [r["truth"][i] for i in keep]
            else:
                for key in ("truth", "factual_arm", "mainline"):
                    if key in r and isinstance(r[key], dict) and "1b" in r[key]:
                        r[key]["1b"] = [r[key]["1b"][i] for i in keep]
            out.append(r)
        if changed:
            tmp = f.with_suffix(".tmp")
            with gzip.open(tmp, "wt") as g:
                for r in out:
                    g.write(json.dumps(r, ensure_ascii=False) + "\n")
            tmp.rename(f)
    bad = {"|".join(map(str, k)): v for k, v in cells.items()
           if k[1] in ("v1", "v2") and v["n"] >= MIN_CELL_N and not (BASE_RATE_LO <= v["rate"] <= BASE_RATE_HI)}
    thin = sum(1 for k, v in cells.items() if k[1] in ("v1", "v2") and v["n"] < MIN_CELL_N)
    summary = {"cells": len(cells), "cells_illegal_rate": len(bad), "cells_thin_main": thin, "illegal": bad,
               "reference_used": dict(fallback), **stats,
               "dropped_by_corpus_cell": {"|".join(map(str, k)): v for k, v in sorted(dropped_cells.items(), key=lambda kv: -kv[1])[:40]}}
    (run_dir / "finalize_1b.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    print(f"finalize_1b: {len(cells)} in-sample cells ({thin} thin in v1/v2 -> corpus reference), {len(bad)} out-of-range "
          f"-> items {stats['items_before']} -> {stats['items_after']}, rows dropped {stats['rows_dropped']}, reference used {dict(fallback)}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data_v2/bench")
    ap.add_argument("--run", default="v2-bench")
    ap.add_argument("--jobs", type=int, default=40)
    ap.add_argument("--limit", type=int, default=0, help="first N worlds (smoke)")
    ap.add_argument("--conds", default="main+ablations", help="'main', 'main+ablations' or comma list")
    ap.add_argument("--tasks", default="123", help="which task groups to emit: any of 1 (1a/1b), 2, 3")
    ap.add_argument("--no_finalize", action="store_true", help="skip the second-stage in-sample 1b filter (smoke builds)")
    ap.add_argument("--sets", default="", help="comma list of splits to build, e.g. bench-basic (default: all)")
    ap.add_argument("--arms_dir", default="arms", help="per-world arm directory: 'arms' (10-turn saves) or 'arms_pt' (per-turn, plan B)")
    ap.add_argument("--ablation_all", action="store_true",
                    help="消融条件对**全部**基础集世界出题（默认只有 60/60/40/40 个世界）；"
                         "务必建到独立的 --run 目录，别动冻结题集")
    a = ap.parse_args()
    C = Path(a.corpus); run_dir = ROOT / "bench/runs" / a.run; run_dir.mkdir(parents=True, exist_ok=True)
    man = json.load(open(C / "manifest.json"))
    worlds = [w for w in man["worlds"] if (C / w["world_id"] / "world.jsonl.gz").exists() and (C / w["world_id"] / "meta.json").exists()]
    if a.sets:
        keep = set(a.sets.split(","))
        worlds = [w for w in worlds if w.get("split") in keep]      # v2 eval: bench-basic only (user 2026-09-10)
    if a.limit:
        # one of each budget first, then the rest
        seen, pick = set(), []
        for w in worlds:
            if w["report_endturn"] not in seen:
                pick.append(w); seen.add(w["report_endturn"])
        pick += [w for w in worlds if w not in pick]
        worlds = pick[:a.limit]
    subsets = ablation_subsets(man["worlds"], all_basic=a.ablation_all)
    jobs = []
    for w in worlds:
        meta = json.loads((C / w["world_id"] / "meta.json").read_text())
        anchors = sorted(int(Path(c).stem[1:]) for c in meta.get("branch_checkpoints") or [])
        for t1 in anchors:
            # Evaluation sets (user 2026-09-10): omniscient view (v1) on all 475 worlds =
            # basic set (225) + generalisation set (250); the POMDP view (v2) and the
            # ablations only on the 225 basic worlds.
            basic = w.get("split") == "bench-basic"
            if a.conds == "main":
                conds = ["v1", "v2"] if basic else ["v1"]
            elif a.conds == "main+ablations":
                conds = (["v1", "v2"] if basic else ["v1"]) + [cid for cid, ws in subsets.items() if w["world_id"] in ws]
            else:
                conds = [x for x in a.conds.split(",") if x]
            jobs.append((str(C), w, t1, str(run_dir), conds, a.tasks, a.arms_dir))
    print(f"{len(worlds)} worlds, {len(jobs)} anchors -> {run_dir}")
    stats = collections.Counter(); errs = []; n3 = 0; chars = []; ops3 = collections.Counter()
    with ProcessPoolExecutor(a.jobs) as ex:
        for r in ex.map(build_anchor, jobs, chunksize=1):
            if r.get("error"):
                errs.append(r); continue
            if r.get("skipped"):
                stats["skipped"] += 1; continue
            stats["anchors"] += 1; n3 += r["task3"]; chars.append(r["prompt_chars"])
            if r.get("op"):
                ops3[r["op"]] += 1
            for k, v in r["by_task"].items():
                stats[k] += v
    print("built:", dict(stats), "| anchors with task 3:", n3, "| ops:", dict(ops3), "| mean prompt chars:", int(sum(chars) / max(1, len(chars))))
    if errs:
        print("errors:", len(errs), errs[:5])
    fin = None if a.no_finalize else finalize_1b(run_dir, a.jobs)
    (run_dir / "build_meta.json").write_text(json.dumps({
        "corpus": a.corpus, "arms_dir": a.arms_dir, "n_worlds": len(worlds), "n_anchors": len(jobs), "tasks": a.tasks, "sets": a.sets or "all", "stats": dict(stats), "errors": errs,
        "ops3": dict(ops3), "finalize_1b": (None if fin is None else {k: v for k, v in fin.items() if k not in ("illegal", "dropped_by_corpus_cell")}),
        "spec_digest": hashlib.sha256((ROOT / "bench/spec_v2.py").read_bytes()).hexdigest()[:16],
        "tasks_digest": hashlib.sha256((ROOT / "bench/tasks_v2.py").read_bytes()).hexdigest()[:16],
        "ablation_subsets": {k: sorted(v) for k, v in subsets.items()}}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
