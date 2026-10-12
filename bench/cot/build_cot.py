"""Build the CoT-lens question set from a frozen run (v3-bench): a stratified anchor subset, every
prediction prompt (1a, 1b, task-3 arms + controls) in three externalisation formats, plus the
intermediate truth every evidence line is checked against.

    .venv/bin/python bench/cot/build_cot.py --src v3-bench --run cot-pilot --per_cell 3 \
        --cells bench-basic:v1,bench-gen:v1,bench-basic:v2 --fmts free,table,code

Rows are the frozen rows with these additions (nothing is removed, so score_v2 still scores the
answer part unchanged):
  fmt          free | table | code
  prompt       free: byte-identical to the frozen prompt; table/code: frozen prompt + OUTPUT PROTOCOL
  max_tokens   + EXTRA_TOKENS_*
  traj         {turn: {civ: {11 LEVEL keys, "government", "relations": {other: rel}, "techs_new": [..]} | None}}
               for turn = t1 (persistence) and t1+10k up to t1+hmax; task-3 arm rows from the arm saves
  owners       {turn: {city: owner}}            (O-line truth)
  techs_t1     {civ: [tech, ...]}               (T-line truth = in techs_t1 or in techs_new at that turn)
  frames_t     the observation turns
  world_json   code format only: the structured observation the program receives (edited record for
               edit-channel arms; `intervention` text for text-channel arms)
Task 2 is not part of the lens (it asks about hidden state, not about the future; 12 §5).
"""
from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "bundle")); sys.path.insert(0, str(ROOT / "tools"))

from bench.arm_state import arm_records, arm_records_light                # noqa: E402
from bench.build_bench_v2 import _arms_at, _pick_real, load_recs           # noqa: E402
from bench.cot.cot_spec import (EXTRA_TOKENS_CODE, EXTRA_TOKENS_CODE_V31, EXTRA_TOKENS_TABLE, LEVEL_KEYS,  # noqa: E402
                                PROTOCOL_VERSION, evidence_turns, protocol_code, protocol_code_v31, protocol_table)
from bench.interventions_v2 import describe_and_edit                      # noqa: E402
from bench.spec_v2 import derive, op_base_of                              # noqa: E402
from bench.truth_v2 import _city_owner_map, mil_of, player, relations     # noqa: E402

TASK_HEADERS = ("=====  TASK 1A", "=====  TASK 1B", "=====  COUNTERFACTUAL")


def civ_state(rec: dict, civ: str, mil) -> dict | None:
    p = player(rec, civ)
    if p is None or not p.get("is_alive"):
        return None
    lv = derive(p, mil, rec)
    return {**{k: lv[k] for k in LEVEL_KEYS}, "government": p.get("government"),
            "relations": relations(rec, civ), "techs": sorted(p.get("techs") or []),
            "cities": [{"name": c["name"], "size": int(c["size"])} for c in (p.get("cities") or [])]}


def traj_truth(states: dict, t1: int, turns: list[int], civs: list[str], mil) -> tuple[dict, dict, dict]:
    """states: {turn: record}.  -> (traj, owners, techs_t1)."""
    techs_t1 = {}
    for c in civs:
        p = player(states[t1], c)
        techs_t1[c] = sorted(p.get("techs") or []) if p else []
    traj, owners = {}, {}
    for t in [t1] + turns:
        r = states.get(t)
        if r is None:
            continue
        traj[str(t)] = {}
        for c in civs:
            s = civ_state(r, c, mil)
            if s is None:
                traj[str(t)][c] = None
            else:
                base = set(techs_t1[c])
                traj[str(t)][c] = {**{k: s[k] for k in LEVEL_KEYS}, "government": s["government"],
                                   "relations": s["relations"], "techs_new": [x for x in s["techs"] if x not in base]}
        owners[str(t)] = _city_owner_map(r)
    return traj, owners, techs_t1


def full_state(rec: dict, civ: str, mil) -> dict | None:
    """v3.1 `world` state: everything the text frames show, structured (cities, units by type, rates,
    research, techs, wonders) on top of the 11 levels."""
    p = player(rec, civ)
    if p is None or not p.get("is_alive"):
        return None
    lv = derive(p, mil, rec)
    inc = p.get("income") or {}
    units = collections.Counter((u.get("type") or u.get("name") or "?") for u in (p.get("units") or []))
    cities = []
    for c in (p.get("cities") or []):
        imp = c.get("improvements")
        if isinstance(imp, str):
            try:
                imp = eval(imp, {"__builtins__": {}}, {})     # the corpus stores the dict repr
            except Exception:
                imp = {}
        built = sorted(k for k, v in (imp or {}).items() if v) if isinstance(imp, dict) else (list(imp) if isinstance(imp, list) else [])
        cities.append({"name": c["name"], "size": int(c["size"]), "production": c.get("production"),
                       "food_stock": int(c.get("food_stock") or 0), "shield_stock": int(c.get("shield_stock") or 0),
                       "founded": int(c.get("founded") or 0), "improvements": built})
    return {**{k: lv[k] for k in LEVEL_KEYS}, "government": p.get("government"), "relations": relations(rec, civ),
            "research_now": p.get("research_now") or p.get("research"),
            "rates": {"tax": inc.get("tax_pct"), "science": inc.get("science_pct"), "luxury": inc.get("luxury_pct")},
            "income": {"trade": inc.get("trade"), "production": inc.get("production"), "science_output": inc.get("science_output")},
            "techs": sorted(p.get("techs") or []), "wonders": sorted(p.get("wonders") or []),
            "cities": cities, "units": dict(sorted(units.items()))}


def traj_pt(states: dict, t1: int, hmax: int, civs: list[str], mil) -> dict:
    """Per-turn truth for v3.1 evidence (any turn) and event detection, compact:
    {"turns": [t1+1..t1+hmax], "levels": {civ: {key: [v or None per turn]}}, "government": {civ: [...]},
     "n_wonders": {civ: [...]}, "wonders_new": {civ: [[turn, name], ...]}, "relations": {civ: {other: [...]}},
     "owners_changes": [[turn, city, new_owner or None]], "missing_turns": [...]}"""
    turns = list(range(t1 + 1, t1 + hmax + 1))
    lv = {c: {k: [] for k in LEVEL_KEYS} for c in civs}; gov = {c: [] for c in civs}; nw = {c: [] for c in civs}
    wnew = {c: [] for c in civs}; rel = {c: collections.defaultdict(list) for c in civs}
    prev_w = {}; prev_own = _city_owner_map(states[t1]) if t1 in states else {}
    own_changes = []; missing = []
    for c in civs:
        p0 = player(states[t1], c) if t1 in states else None
        prev_w[c] = set(p0.get("wonders") or []) if p0 else set()
    for t in turns:
        r = states.get(t)
        if r is None:
            missing.append(t)
            for c in civs:
                for k in LEVEL_KEYS:
                    lv[c][k].append(None)
                gov[c].append(None); nw[c].append(None)
                for o in civs:
                    if o != c:
                        rel[c][o].append(None)
            continue
        for c in civs:
            s = civ_state(r, c, mil)
            if s is None:
                for k in LEVEL_KEYS:
                    lv[c][k].append(None)
                gov[c].append(None); nw[c].append(None)
                for o in civs:
                    if o != c:
                        rel[c][o].append(None)
                continue
            for k in LEVEL_KEYS:
                lv[c][k].append(s[k])
            gov[c].append(s["government"])
            p = player(r, c); w = set(p.get("wonders") or [])
            nw[c].append(len(w))
            for x in sorted(w - prev_w[c]):
                wnew[c].append([t, x])
            prev_w[c] = w
            for o in civs:
                if o != c:
                    rel[c][o].append(s["relations"].get(o))
        own = _city_owner_map(r)
        for city in set(own) | set(prev_own):
            if own.get(city) != prev_own.get(city):
                own_changes.append([t, city, own.get(city)])
        prev_own = own
    return {"turns": turns, "levels": lv, "government": gov, "n_wonders": nw, "wonders_new": wnew,
            "relations": {c: dict(d) for c, d in rel.items()}, "owners_changes": own_changes, "missing_turns": missing}


def world_json_full(recs: dict, frames: list[int], t1: int, hz: list[int], civs: list[str], ruleset: str, mil,
                    rec_t1_override: dict | None = None, intervention: str | None = None) -> dict:
    fr = {}
    for t in frames:
        r = recs[t] if not (rec_t1_override is not None and t == t1) else rec_t1_override
        fr[str(t)] = {c: full_state(r, c, mil) for c in civs}
    out = {"turn": t1, "horizons": list(hz), "ruleset": ruleset, "civs": list(civs), "frames": fr}
    if intervention:
        out["intervention"] = intervention
    return out


def world_json(recs: dict, frames: list[int], t1: int, hz: list[int], civs: list[str], ruleset: str, mil,
               rec_t1_override: dict | None = None, intervention: str | None = None) -> dict:
    fr = {}
    for t in frames:
        r = recs[t] if not (rec_t1_override is not None and t == t1) else rec_t1_override
        fr[str(t)] = {c: civ_state(r, c, mil) for c in civs}
    out = {"turn": t1, "horizons": list(hz), "ruleset": ruleset, "civs": list(civs), "frames": fr}
    if intervention:
        out["intervention"] = intervention
    return out


def _insert_protocol(prompt: str, proto: str) -> str:
    return prompt.rstrip("\n") + "\n\n" + proto


def build_anchor(args) -> dict:
    src_file, dst_file, corpus, fmts = args[:4]
    tasks = args[4] if len(args) > 4 else ("1a", "1b", "3")
    arms_dir = args[5] if len(args) > 5 else "arms"          # "arms_pt": per-turn arms (plan B, 14 §B16-2)
    v31 = bool(args[6]) if len(args) > 6 else False          # protocol v3.1: per-turn table, stochastic simulator code
    aligned = bool(args[7]) if len(args) > 7 else False
    conds = set(args[8]) if len(args) > 8 else None
    rows = [json.loads(l) for l in gzip.open(src_file, "rt")]
    rows = [r for r in rows if not r.get("error") and r["task"] in tasks and (conds is None or r["cond"] in conds)]
    if aligned:
        rows = [r for r in rows if r["cond"] == "v1" or (r["cond"] == "v2" and r["set"] == "bench-basic")]
    if not rows:
        return {"file": Path(src_file).name, "rows": 0}
    r0 = rows[0]; wid, t1 = r0["world"], int(r0["t1"])
    wd = Path(corpus) / wid
    hmax_all = max(max(r["horizons"]) if r.get("horizons") else max(i["h"] for i in r["items"]) for r in rows)
    from bench.spec_v2 import frames_for
    frames = frames_for(t1)            # 3 observation turns (k1/k5 rows exist only in ablations, not here)
    recs = load_recs(wd, frames[0], t1 + hmax_all)
    rec1 = recs[t1]; mil = mil_of(rec1); ruleset = rec1.get("ruleset") or "civ2civ3"
    from sandboxes.freeciv.state import base_ruleset
    rs = base_ruleset(ruleset)
    # task-3 arm states, same deterministic pick as the frozen build
    arm_pack = None
    if any(r["task"] == "3" for r in rows) and not aligned:
        arms = _arms_at(wd, t1, arms_dir)
        budget = int(r0["budget"])
        real = _pick_real(arms, wid, t1, budget, rec1)
        if real is not None:
            a_real, d_real, hz3 = real
            plc = arms.get(a_real["arm"] + "__placebo")
            turns3 = evidence_turns(t1, max(hz3))
            if v31:                                          # per-turn arm saves (plan B): light records at every turn
                turns3 = list(range(t1 + 1, t1 + max(hz3) + 1))
                loader = lambda d: arm_records_light(Path(str(d)), wid, turns3)          # noqa: E731
            else:
                loader = lambda d: arm_records(Path(str(d)), wid, turns3, with_probes=False)   # noqa: E731
            arm_pack = {"real": (a_real, loader(d_real)),
                        "placebo": (plc[0], loader(plc[1])) if plc else None,
                        "hz3": hz3}
    arms = _arms_at(wd, t1, arms_dir) if aligned else {}
    arm_cache = {}
    reference = next((r for r in rows if r["task"] == "3" and r.get("arm_kind") == "real"), {})
    reference = dict(reference, all_answer_horizons=sorted({h for r in rows for h in r.get("horizons", [])}))

    def aligned_arm(row, factual=False):
        if factual or row.get("arm_kind") == "control":
            names = ["factual"] if "factual" in arms else []
        else:
            names = [name for name, (aj, _) in arms.items()
                     if (aj.get("op") or {}).get("id") == row.get("op")
                     and (aj.get("op") or {}).get("target") == row.get("op_target")]
            if row.get("arm_kind") == "placebo":
                real_names = [name for name, (aj, _) in arms.items()
                              if (aj.get("op") or {}).get("id") == reference.get("op")
                              and (aj.get("op") or {}).get("target") == reference.get("op_target")]
                # Frozen +90 rows can retain the +45 placebo id; the paired file is authoritative.
                names = [n + "__placebo" for n in real_names if n + "__placebo" in arms
                         and op_base_of(arms[n + "__placebo"][0]["op"]["id"]) == row.get("op_base")
                         and arms[n + "__placebo"][0]["op"].get("target") == row.get("op_target")]
        if len(names) != 1:
            raise ValueError(f"{wid}:{t1}: ambiguous/missing frozen arm {row.get('op')} factual={factual}: {names}")
        name = names[0]
        if name not in arm_cache:
            aj, directory = arms[name]
            max_h = max(max(r.get("horizons") or [i["h"] for i in r["items"]]) for r in rows if r["task"] == "3")
            states = arm_records_light(Path(directory), wid, list(range(t1 + 1, t1 + max_h + 1)))
            arm_cache[name] = (aj, dict(states))
            arm_cache[name][1][t1] = rec1
        return name, arm_cache[name]
    out = []
    stats = collections.Counter()
    for r in rows:
        civs = list(r["civs"]) if r.get("civs") else sorted({i["civ"] for i in r["items"] if i["civ"] != "*"} | {c for i in r["items"] for c in i.get("civs", [])})
        if r["task"] in ("1a",) or (r["task"] == "3" and r.get("sub") == "1a"):
            hz = list(r["horizons"])
        else:
            hz = sorted({int(i["h"]) for i in r["items"]})
        hmax = max(hz)
        turns = evidence_turns(t1, hmax)
        kind = r.get("arm_kind")
        required = []
        if aligned:
            from bench.cot.aligned import evidence_cells
            required = evidence_cells(r, reference)
            turns = sorted(set(turns) | {c["turn"] for c in required})
        mainline_pt = None
        factual_traj = factual_pt = None
        arm_name = None
        if aligned and r["task"] == "3":
            arm_name, (aj, states) = aligned_arm(r)
            traj, owners, techs_t1 = traj_truth(states, t1, turns, civs, mil)
            pt = traj_pt(states, t1, hmax, civs, mil)
            _, (_, factual_states) = aligned_arm(r, factual=True)
            factual_traj, _, _ = traj_truth(factual_states, t1, turns, civs, mil)
            factual_pt = traj_pt(factual_states, t1, hmax, civs, mil)
            mainline_pt = traj_pt(recs, t1, hmax, civs, mil) if kind == "control" else None
            if kind == "control":
                rec_over, interv = None, None
            else:
                rec_over, interv, _ = describe_and_edit(rec1, aj["op"])
            wj_src = recs
        elif r["task"] == "3" and kind in ("real", "placebo"):
            if arm_pack is None or arm_pack.get(kind) is None:
                stats["arm_missing"] += 1; continue
            aj, states = arm_pack[kind]
            if op_base_of(aj["op"]["id"]) != r.get("op_base"):
                stats["arm_mismatch"] += 1; continue
            states = dict(states); states[t1] = rec1
            traj, owners, techs_t1 = traj_truth(states, t1, turns, civs, mil)
            pt = traj_pt(states, t1, hmax, civs, mil) if v31 else None
            try:
                ed, desc, channel = describe_and_edit(rec1, aj["op"])
            except Exception:
                ed, desc = None, r.get("intervention")
            wj_src = dict(recs); rec_over = ed
            interv = desc
        else:
            traj, owners, techs_t1 = traj_truth(recs, t1, turns, civs, mil)
            pt = traj_pt(recs, t1, hmax, civs, mil) if v31 else None
            wj_src = recs; rec_over = None; interv = None
        has_gloss = "=====  TASK 1A" in r["prompt"]
        cf = r["task"] == "3" and (not aligned or kind != "control")
        for fmt in fmts:
            q = dict(r); q["fmt"] = fmt
            q["traj"] = traj; q["owners"] = owners; q["techs_t1"] = techs_t1; q["frames_t"] = frames
            q["ev_turns"] = turns; q["hmax"] = hmax; q["ev_civs"] = civs
            if v31:
                q["traj_pt"] = pt; q["protocol"] = PROTOCOL_VERSION
                q["wonders_t1"] = {c: sorted((player(rec1, c) or {}).get("wonders") or []) for c in civs}
            if fmt == "table":
                q["prompt"] = _insert_protocol(r["prompt"], protocol_table(t1, hmax, civs, has_gloss, cf, per_turn=v31,
                                                                            answer_hz=(hz if has_gloss else None)))
                q["max_tokens"] = int(r["max_tokens"]) + EXTRA_TOKENS_TABLE
            elif fmt == "code":
                if r.get("view") != 1:
                    continue          # code only in the omniscient view (a masked POMDP world dict is not built)
                if v31:
                    task_kind = "1a" if has_gloss else "1b"
                    q["prompt"] = _insert_protocol(r["prompt"], protocol_code_v31(t1, hmax, hz, civs, frames, rs, cf, has_gloss, task_kind))
                    q["max_tokens"] = int(r["max_tokens"]) + EXTRA_TOKENS_CODE_V31
                    q["world_json"] = world_json_full(wj_src, frames, t1, hz, civs, rs, mil, rec_over, interv)
                else:
                    if not has_gloss:
                        continue      # v3.0: code only for 1a-type prompts
                    q["prompt"] = _insert_protocol(r["prompt"], protocol_code(t1, hmax, hz, civs, frames, rs, cf))
                    q["max_tokens"] = int(r["max_tokens"]) + EXTRA_TOKENS_CODE
                    q["world_json"] = world_json(wj_src, frames, t1, hz, civs, rs, mil, rec_over, interv)
            if aligned:
                from bench.cot.aligned import PROTOCOL, table_requirement
                q.update(protocol=PROTOCOL, evidence_cells=required, evidence_max_lines=max(40, len(required) + 8),
                         truth_source=(f"{arms_dir}/t{t1}_{arm_name}" if arm_name else "mainline"))
                if factual_pt is not None:
                    q.update(factual_traj=factual_traj, factual_traj_pt=factual_pt, mainline_traj_pt=mainline_pt,
                             process_control_ref=(r.get("control_ref") or {}).get("1a"))
                if fmt == "table":
                    q["prompt"] += table_requirement(required)
                    q["max_tokens"] = int(r["max_tokens"]) + max(EXTRA_TOKENS_TABLE, (len(required) + 8) * 48)
                if fmt == "code":
                    q["backtest_world_json"] = world_json_full(recs, frames, t1, hz, civs, rs, mil)
                    q["prompt"] = q["prompt"].replace(
                        "the counterfactual world (the intervention is already applied in `world`)",
                        "the counterfactual world described by `world[\"intervention\"]` "
                        "(visible state edits are applied in `world`; also model the described mechanism changes)")
            out.append(q); stats[f"{r['task']}{r.get('sub') or ''}|{fmt}"] += 1
    dst = Path(dst_file); dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp")
    with gzip.open(tmp, "wt") as f:
        for q in out:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    tmp.rename(dst)
    return {"file": dst.name, "rows": len(out), "stats": dict(stats)}


def select_anchors(src_dir: Path, cells: list[tuple[str, str]], per_cell: int, mid_only: bool = True) -> list[tuple[Path, str]]:
    """Deterministic stratified pick: per (set, cond, budget) `per_cell` anchors that carry task 3,
    preferring the 2nd/3rd anchor of a world (mid-game) and rotating over operators."""
    anchors = json.load(open(ROOT / "bench/v2/anchors.json"))["anchors"]
    meta = []
    for f in sorted(src_dir.glob("*.jsonl.gz")):
        wid, t1 = f.stem.replace(".jsonl", "").rsplit("_", 1); t1 = int(t1)
        idx = anchors.get(wid, []).index(t1) if t1 in anchors.get(wid, []) else -1
        rows = [json.loads(l) for l in gzip.open(f, "rt")]
        conds = {r["cond"] for r in rows}
        st = rows[0]["set"]; budget = int(rows[0]["budget"])
        ops = {r.get("op_base") for r in rows if r["task"] == "3" and r.get("arm_kind") == "real"}
        has3b = any(r["task"] == "3" and r.get("sub") == "1b" for r in rows)
        meta.append(dict(f=f, wid=wid, t1=t1, idx=idx, conds=conds, set=st, budget=budget, ops=ops, has3b=has3b))
    picked = []
    for st, cond in cells:
        for budget in (400, 800, 1200):
            cands = [m for m in meta if m["set"] == st and cond in m["conds"] and m["budget"] == budget and m["ops"]]
            if mid_only:
                mid = [m for m in cands if m["idx"] in (1, 2)]
                cands = mid or cands
            cands.sort(key=lambda m: hashlib.sha1(f"{m['wid']}|{m['t1']}|cot".encode()).hexdigest())
            chosen, seen_ops = [], collections.Counter()
            for m in cands:                     # rotate operators: least-used op first
                if len(chosen) >= per_cell:
                    break
                op = next(iter(m["ops"]))
                if seen_ops[op] >= max(1, per_cell // 3) and any(seen_ops[o] < seen_ops[op] for o in {x for c in cands for x in c["ops"]}):
                    continue
                chosen.append(m); seen_ops[op] += 1
            for m in cands:
                if len(chosen) >= per_cell:
                    break
                if m not in chosen:
                    chosen.append(m)
            picked += [(m["f"], cond) for m in chosen]
    return picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="v3-bench")
    ap.add_argument("--run", default="cot-pilot")
    ap.add_argument("--corpus", default="data_v2/bench")
    ap.add_argument("--per_cell", type=int, default=3, help="anchors per (set, cond, budget)")
    ap.add_argument("--cells", default="bench-basic:v1,bench-gen:v1,bench-basic:v2")
    ap.add_argument("--fmts", default="free,table,code")
    ap.add_argument("--anchors", default="", help="file with anchor file names to use instead of the stratified pick")
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--tasks", default="1a,1b,3", help="which source tasks to externalise (e.g. 1a,1b before the arms are final)")
    ap.add_argument("--arms_dir", default="arms", help="per-world arm directory for task-3 truth: arms (10-turn) or arms_pt (per-turn)")
    ap.add_argument("--v31", action="store_true", help="protocol v3.1: per-turn table (+W lines), stochastic simulator code for 1a/1b/3, full world dict, per-turn truth")
    ap.add_argument("--aligned", action="store_true", help="v3.2 fixed numeric cells and frozen-arm/factual process truth")
    a = ap.parse_args()
    if a.aligned:
        a.v31 = True
    src = ROOT / "bench/runs" / a.src / "questions"; run_dir = ROOT / "bench/runs" / a.run
    (run_dir / "questions").mkdir(parents=True, exist_ok=True)
    cells = [tuple(c.split(":")) for c in a.cells.split(",") if c]
    if a.anchors:
        names = [l.strip() for l in open(a.anchors) if l.strip()]
        picked = [(src / n, c) for n in names for c in {cc for _, cc in cells}]
    else:
        picked = select_anchors(src, cells, a.per_cell)
    # one output file per anchor; the conditions wanted per anchor are the union over cells
    want = collections.defaultdict(set)
    for f, cond in picked:
        want[f].add(cond)
    fmts = [x for x in a.fmts.split(",") if x]
    jobs = []
    for f, conds in sorted(want.items()):
        # pre-filter rows by cond into a temp source? simpler: build all conds in the file, then drop
        jobs.append((str(f), str(run_dir / "questions" / f.name), a.corpus, fmts, sorted(conds), tuple(a.tasks.split(",")), a.arms_dir, a.v31, a.aligned))
    print(f"{len(jobs)} anchors -> {run_dir}")
    stats = collections.Counter(); files = []
    with ProcessPoolExecutor(a.jobs) as ex:
        for res in ex.map(_build_filtered, jobs, chunksize=1):
            files.append(res["file"]); stats["rows"] += res["rows"]
            for k, v in (res.get("stats") or {}).items():
                stats[k] += v
    print("built:", dict(stats))
    (run_dir / "build_meta.json").write_text(json.dumps({
        "src": a.src, "cells": a.cells, "per_cell": a.per_cell, "fmts": fmts, "tasks": a.tasks, "arms_dir": a.arms_dir, "v31": a.v31, "aligned": a.aligned, "n_anchors": len(jobs),
        "anchors": [Path(j[0]).name for j in jobs], "stats": dict(stats),
        "cot_spec_digest": __import__("bench.cot.cot_spec", fromlist=["digest"]).digest()}, ensure_ascii=False, indent=1))
    (run_dir / "anchors.txt").write_text("\n".join(Path(j[0]).name for j in jobs) + "\n")


def _build_filtered(args):
    src_file, dst_file, corpus, fmts, conds = args[:5]
    tasks = args[5] if len(args) > 5 else ("1a", "1b", "3"); arms_dir = args[6] if len(args) > 6 else "arms"
    v31 = args[7] if len(args) > 7 else False
    aligned = args[8] if len(args) > 8 else False
    res = build_anchor((src_file, dst_file, corpus, fmts, tasks, arms_dir, v31, aligned, conds))
    # drop conditions that were not asked for (build_anchor emits every cond in the file)
    rows = [json.loads(l) for l in gzip.open(dst_file, "rt")]
    keep = [r for r in rows if r["cond"] in conds]
    if len(keep) != len(rows):
        tmp = Path(dst_file).with_suffix(".tmp")
        with gzip.open(tmp, "wt") as f:
            for r in keep:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        tmp.rename(dst_file)
        res["rows"] = len(keep)
    res["stats"] = dict(collections.Counter(f"{r['task']}{r.get('sub') or ''}|{r['fmt']}" for r in keep))
    return res


if __name__ == "__main__":
    main()
