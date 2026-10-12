"""v2 leakage and sanity gates over a built question set.

 ① no world_id in any prompt              (trait/param lookup key)
 ② no ruleset variant name (`_ill`) or hidden parameter setting names in the prompt
 ③ task 2 targets are never barbarians
 ④ (retired with 1c) no 1c rows exist
 ⑤ observation part mentions no turn beyond t1 (except the oracle condition)
 ⑥ POMDP rows: the observation is shorter than the omniscient one and asks only about met civs
 ⑦ task 3: every real arm has its placebo in the same anchor & condition; every operator
    appears; intervention text never names the arm id
 ⑧ prompt + max_tokens <= max_model_len (tokenizer count)
 ⑨ v3: every 1b item (main grid and task 3) sits in a per-(set, cond, template, budget, h, t1)
    in-sample cell with rate in 10-90% and n >= MIN_CELL_N; `leader` exempt   ← the gate that was
    missing when 30/264 out-of-range cells slipped through the frozen v2 set
 ⑩ v3: every task-3 horizon lies inside the operator's effect band
 ⑪ v3: task-2 personality rows rank >= 2 civs and carry one truth value per civ and dimension;
    world-rule questions appear only with the evidence flag; `_illness` never asked
 ⑫ v3: the STRATEGIES block is present in every 1a/1b/3 prompt and ABSENT from every task-2
    prompt (13 §5.3 mutual exclusion); in the POMDP view it names only the viewer's plan

    .venv/bin/python bench/verify_bench_v2.py --run v2-bench --jobs 40
"""
from __future__ import annotations

import argparse
import collections
import gzip
import json
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from bench.spec_v2 import BASE_RATE_HI, BASE_RATE_LO, MIN_CELL_N, OPS, TRAITS, PARAM_EVIDENCE, PARAM_DISPLAY  # noqa: E402

MML = 262144
TK = None
FORBIDDEN = ("_ill", "civ2civ3_ill", "foodbox", "sciencebox", "disasters=", "razechance", "onsetbarbs", "rulesetdir")


def _tok():
    global TK
    if TK is None:
        from tokenizers import Tokenizer
        import glob
        cands = glob.glob(str(ROOT / "model_states/Qwen3.8-Flash-Next/tokenizer.json")) + glob.glob(str(ROOT / "model_states/Qwen3.5-9B-Base/tokenizer.json"))
        TK = Tokenizer.from_file(cands[0])
    return TK


def check_file(path: Path) -> dict:
    tk = _tok(); bad = []; stats = collections.Counter(); lens = []
    arms = collections.defaultdict(set); ops_seen = collections.Counter()
    obs_len = {}; cells_1b = []
    for line in gzip.open(path, "rt"):
        r = json.loads(line)
        if r.get("error"):
            stats["rows_error"] += 1; continue
        p = r["prompt"]; obs = p.split("=====  TASK")[0].split("=====  COUNTERFACTUAL")[0]
        stats["rows"] += 1
        if r["world"] in p:
            bad.append(f"① {r['world']} {r['task']}: prompt contains world_id")
        low = p.lower()
        for w in FORBIDDEN:
            if w in low and not (w in ("disasters=",) ):
                # "disasters" appears legitimately in the WP-disasters intervention sentence and task-2 labels
                if w in ("foodbox", "sciencebox", "razechance", "onsetbarbs", "rulesetdir", "_ill", "civ2civ3_ill"):
                    bad.append(f"② {r['world']} {r['task']} {r['cond']}: prompt contains '{w}'")
                    break
        if r["task"] == "2" and any(("Pirate" in c or "Barbarian" in c) for c in (r.get("civs") or [])):
            bad.append(f"③ {r['world']}: task 2 targets include barbarians")
        if r["task"] == "1c":
            bad.append(f"④ {r['world']}: 1c row present (retired)")
        if r["task"] == "1b" or (r["task"] == "3" and r.get("sub") == "1b"):
            for it in r["items"]:
                if it["kind"] != "leader":
                    cells_1b.append((r["task"], r["set"], r["cond"], it["kind"], str(r["budget"]), str(it["h"]), str(r["t1"])))
        if r["task"] == "3":
            band = r.get("horizon_band") or [0, 10**9]
            if any(not (band[0] <= h <= band[1]) for h in (r.get("horizons") or [])):
                bad.append(f"⑩ {r['world']} {r['cond']} {r['op']}: horizons {r['horizons']} outside band {band}")
        has_strat = "=====  STRATEGIES" in p
        if r["task"] == "2" and has_strat:
            bad.append(f"⑫ {r['world']} {r['cond']}: task-2 prompt contains the STRATEGIES block")
        if r["task"] in ("1a", "1b", "3") and not has_strat:
            bad.append(f"⑫ {r['world']} {r['cond']} {r['task']}: STRATEGIES block missing")
        if has_strat and r.get("pomdp"):
            blk = p.split("=====  STRATEGIES")[1].split("=====  ")[0]
            plans = [ln for ln in blk.split("\n") if ln.endswith(".") and " is set on this course: " in ln]
            if len(plans) != 1 or not plans[0].startswith(r["viewer"]):
                bad.append(f"⑫ {r['world']} {r['cond']} {r['task']}: POMDP strategy block names {len(plans)} civs")
        if r["task"] == "2":
            tr = (r["truth"].get("traits") or {})
            if tr:
                civs = r.get("civs") or []
                if len(civs) < 2 or set(tr) != set(TRAITS) or any(set(v) != set(civs) for v in tr.values()):
                    bad.append(f"⑪ {r['world']} {r['cond']}: personality truth malformed ({len(civs)} civs, dims {sorted(tr)})")
            elif len(r.get("civs") or []) >= 2:
                bad.append(f"⑪ {r['world']} {r['cond']}: {len(r['civs'])} rankable civs but no personality truth")
            for pn in (r["truth"].get("params") or {}):
                rule = PARAM_EVIDENCE.get(pn, "never")
                if rule == "never":
                    bad.append(f"⑪ {r['world']} {r['cond']}: parameter {pn} asked (not in PARAM_EVIDENCE)")
                elif rule is not None and (r.get("param_evidence") or {}).get(pn, 0) < rule[1]:
                    bad.append(f"⑪ {r['world']} {r['cond']}: {pn} asked without evidence {r.get('param_evidence')}")
        if not r.get("oracle"):
            fut = [int(x) for x in re.findall(r"\bturn (\d+)\b", obs) if int(x) > r["t1"]]
            if fut:
                bad.append(f"⑤ {r['world']} {r['cond']} {r['task']}: observation mentions future turn {fut[:3]}")
        if r["task"] == "1a":
            obs_len[(r["cond"],)] = len(obs)
            if r.get("pomdp"):
                if r.get("scope") is not None and set(r["civs"]) - set(r["scope"]):
                    bad.append(f"⑥ {r['world']} {r['cond']}: POMDP asks about unmet civs")
        if r["task"] == "3":
            arms[(r["cond"], r["op_base"], r.get("sub"))].add(r.get("arm_kind") or ("placebo" if r["is_placebo"] else "real"))
            if r.get("arm_kind", "real") == "real" and r.get("sub", "1a") == "1a":
                ops_seen[r["op_base"]] += 1
            if r["op"] in p or r["op_base"] in p:
                bad.append(f"⑦ {r['world']} {r['cond']}: prompt contains operator id {r['op']}")
        n = len(tk.encode(p).ids); tot = n + r["max_tokens"]
        lens.append((r["task"], r["cond"], r["budget"], n, tot))
        if tot > MML:
            bad.append(f"⑧ {r['world']} {r['cond']} {r['task']}: {n:,}+{r['max_tokens']:,}={tot:,} > {MML:,}")
    for (cond, op, sub), kinds in arms.items():
        if not {"real", "placebo"} <= kinds:
            bad.append(f"⑦ {path.name} {cond} {op} {sub}: arm pair incomplete {sorted(kinds)}")
    if ("v1",) in obs_len and ("v2",) in obs_len and obs_len[("v2",)] >= obs_len[("v1",)]:
        bad.append(f"⑥ {path.name}: POMDP observation not shorter than omniscient ({obs_len[('v2',)]} vs {obs_len[('v1',)]})")
    return {"file": path.name, "bad": bad, "stats": dict(stats), "lens": lens, "ops": dict(ops_seen), "cells_1b": cells_1b}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", default="v2-bench"); ap.add_argument("--jobs", type=int, default=40)
    a = ap.parse_args()
    run = ROOT / "bench/runs" / a.run
    files = sorted((run / "questions").glob("*.jsonl.gz"))
    bad = []; lens = []; ops = collections.Counter(); rows = 0; cells = collections.Counter(); used = collections.Counter()
    with ProcessPoolExecutor(a.jobs) as ex:
        for r in ex.map(check_file, files, chunksize=4):
            bad += r["bad"]; lens += r["lens"]; ops.update(r["ops"]); rows += r["stats"].get("rows", 0)
            for c in r["cells_1b"]:
                used[c] += 1
    # ⑨ in-sample cells: rate from the main-grid 1b truth, checked for every item that uses the cell
    yes = collections.Counter(); n = collections.Counter()
    for f in files:
        for l in gzip.open(f, "rt"):
            r = json.loads(l)
            if r.get("task") != "1b" or r.get("error"):
                continue
            for it, y in zip(r["items"], r["truth"]):
                if it["kind"] != "leader" and y is not None:
                    k = (r["set"], r["cond"], it["kind"], str(r["budget"]), str(it["h"]), str(r["t1"]))
                    n[k] += 1; yes[k] += int(bool(y))
    cp = ROOT / "bench/v2/cells_1b.json"
    corpus = json.load(open(cp)) if cp.exists() else {}
    for (task, *k), cnt in sorted(used.items()):
        st, cond, kind, budget, h, t1 = k
        kr = (st, cond if cond in ("v1", "v2") else "v1", kind, budget, h, t1)     # mirror of score_v2.cell_1b
        if n[kr] >= MIN_CELL_N:
            rate, nn, src = yes[kr] / n[kr], n[kr], "insample"
        else:
            c = corpus.get(f"{kind}|{budget}|{h}|{t1}") or {}
            rate, nn, src = c.get("rate"), c.get("n") or 0, "corpus"
        if rate is None or nn < MIN_CELL_N or not (BASE_RATE_LO <= rate <= BASE_RATE_HI):
            bad.append(f"⑨ task {task} cell {'|'.join(k)} ({src}): rate {rate} n={nn} used by {cnt} items")
    by = collections.defaultdict(list)
    for task, cond, budget, n, tot in lens:
        by[(task, cond, budget)].append((n, tot))
    print(f"{len(files)} anchor files, {rows} rows, {len(bad)} gate violations")
    for b in bad[:30]:
        print("  ", b)
    print("\nprompt tokens (median / p95 / max total incl. max_tokens) by task, cond, budget:")
    import numpy as np
    for k in sorted(by):
        ns = np.array([x[0] for x in by[k]]); ts = np.array([x[1] for x in by[k]])
        print(f"  {k[0]:<3} {k[1]:<7} {k[2]:>5}  n={len(ns):>5}  prompt med {np.median(ns):>8,.0f}  p95 {np.percentile(ns,95):>8,.0f}  max total {ts.max():>8,.0f}")
    print("\noperators in task 3 (rows):", dict(ops), "| missing:", [o for o in OPS if o not in ops])
    (run / "verify.json").write_text(json.dumps({"n_files": len(files), "rows": rows, "violations": bad, "ops": dict(ops)}, ensure_ascii=False, indent=1))
    print("\nGATES:", "PASS" if not bad else f"FAIL ({len(bad)})")


if __name__ == "__main__":
    main()
