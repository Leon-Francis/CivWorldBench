"""Audit built aligned prompts against frozen answers and paired process truth."""
from __future__ import annotations

import argparse
import collections
import gzip
import json
from pathlib import Path

from bench.cot.score_cot import key_of, pt_state, task_value

ROOT = Path(__file__).resolve().parents[2]


def verify(run):
    counts = collections.Counter()
    errors = []
    for path in sorted((run / "questions").glob("*.jsonl.gz")):
        rows = [json.loads(x) for x in gzip.open(path, "rt")]
        lookup = {key_of(q): q for q in rows}
        if len(lookup) != len(rows):
            errors.append(f"{path.name}: duplicate question keys")
        for q in rows:
            counts[q["fmt"]] += 1
            counts["rows"] += 1
            where = f"{path.name}:{key_of(q)}"
            if q["protocol"] != "3.2-aligned":
                errors.append(f"{where}: wrong protocol")
            required = {(c["civ"], c["var"], c["turn"]) for c in q["evidence_cells"]}
            if len(required) != len(q["evidence_cells"]):
                errors.append(f"{where}: duplicate required cells")
            for c, v, t in required:
                if not q["t1"] < t <= q["t1"] + q["hmax"] or t - q["t1"] in q.get("horizons", []):
                    errors.append(f"{where}: invalid intermediate cell")
                counts["required_cells"] += 1
            if q["fmt"] == "code":
                table = lookup[key_of(dict(q, fmt="table"))]
                if q["evidence_cells"] != table["evidence_cells"]:
                    errors.append(f"{where}: Table/Code required cells differ")
                for world_key in ("world_json", "backtest_world_json"):
                    w = q[world_key]
                    if max(map(int, w["frames"])) > q["t1"]:
                        errors.append(f"{where}: future frame leaked into {world_key}")
                if q["backtest_world_json"].get("intervention"):
                    errors.append(f"{where}: backtest intervention leakage")
            if q["task"] == "3":
                if not q.get("factual_traj_pt") or "arms_pt/" not in q["truth_source"]:
                    errors.append(f"{where}: missing rerun factual truth")
                if q.get("sub") == "1a" and q.get("arm_kind") in ("real", "placebo"):
                    main = (q.get("control_ref") or {}).get("1a") == "main"
                    ctrl = next((r for r in rows if r["cond"] == q["cond"] and r["fmt"] == q["fmt"]
                                 and ((main and r["task"] == "1a") or
                                      (not main and r["task"] == "3" and r.get("sub") == "1a"
                                       and r.get("arm_kind") == "control" and r.get("op_base") == q.get("op_base")))), None)
                    if ctrl is None or ctrl["evidence_cells"] != q["evidence_cells"]:
                        errors.append(f"{where}: missing/misaligned control predictions")
                    counts["paired_numeric_prompts"] += 1
            if q["task"] != "1a" and q.get("sub") != "1a":
                continue
            truth = q["truth"] if q["task"] == "1a" else q["truth"]["1a"]
            if q.get("arm_kind") == "control":
                truth = q["factual_arm"]["1a"]
            for h, civs in truth.items():
                for civ, variables in civs.items():
                    if variables is None:
                        continue
                    st = pt_state(q, civ, int(q["t1"]) + int(h))
                    base = q["traj"][str(q["t1"])].get(civ)
                    if st is None or base is None:
                        counts["missing_truth_states"] += 1
                        continue
                    for var, expected in variables.items():
                        if expected is None:
                            continue
                        actual = task_value(st, base, var)
                        counts["truth_comparisons"] += 1
                        if actual is None or abs(actual - expected) > 1e-6:
                            errors.append(f"{where}: frozen truth differs {civ} {h} {var}: {actual} != {expected}")
        counts["anchors"] += 1
    return {"counts": dict(counts), "errors": errors, "ok": bool(counts["anchors"]) and not errors}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    a = ap.parse_args()
    run = ROOT / "bench/runs" / a.run
    report = verify(run)
    (run / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({**report, "errors": report["errors"][:15]}, indent=2))
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
