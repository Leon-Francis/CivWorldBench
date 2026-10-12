"""Paired state/effect process diagnostics; never changes formal benchmark scores."""
from __future__ import annotations

import argparse
import collections
import gzip
import json
from pathlib import Path

import numpy as np

from bench import score_v2 as SV
from bench.cot.aligned import intermediate_offsets
from bench.cot.score_cot import key_of, pt_state, task_value
from bench.parse_v2 import parse_quantiles

ROOT = Path(__file__).resolve().parents[2]


def effect_error(var, pred, control_pred, truth, control_truth):
    scale = SV.norm(var)
    estimated = (SV.tf(var, pred) - SV.tf(var, control_pred)) / scale
    observed = (SV.tf(var, truth) - SV.tf(var, control_truth)) / scale
    return abs(estimated - observed), abs(observed), estimated, observed


def level(q, civ, var, turn, source="traj_pt"):
    st = pt_state(dict(q, traj_pt=q.get(source)), civ, turn)
    base = q["traj"][str(q["t1"])].get(civ)
    return task_value(st, base, var) if st is not None and base is not None else None


def spearman(x, y):
    if len(x) < 3 or len(set(x)) < 2 or len(set(y)) < 2:
        return None
    def ranks(v):
        _, inverse, counts = np.unique(v, return_inverse=True, return_counts=True)
        return (np.cumsum(counts) - (counts - 1) / 2)[inverse]
    return float(np.corrcoef(ranks(x), ranks(y))[0, 1])


def association(rows, nboot, seed=20260923):
    if not rows:
        return {"n": 0, "n_worlds": 0}
    worlds = sorted({r["world"] for r in rows})
    indices = {w: [i for i, r in enumerate(rows) if r["world"] == w] for w in worlds}
    values = np.array([[r[k] for k in ("l1_process_error", "l1_answer_error", "process_error", "answer_error")] for r in rows])

    def compute(v):
        r1 = spearman(v[:, 0], v[:, 1]); r3 = spearman(v[:, 2], v[:, 3])
        return r1, r3, r3 - r1 if r1 is not None and r3 is not None else None

    point = compute(values)
    rng = np.random.default_rng(seed)
    boot = [[], [], []]
    for _ in range(nboot):
        sample = rng.choice(worlds, len(worlds), replace=True)
        v = values[[i for w in sample for i in indices[w]]]
        for j, x in enumerate(compute(v)):
            if x is not None:
                boot[j].append(x)
    out = {"n": len(rows), "n_worlds": len(worlds)}
    for j, name in enumerate(("rho_L1", "rho_L3", "rho_difference")):
        out[name] = {"value": point[j], "ci95": np.percentile(boot[j], [2.5, 97.5]).tolist() if boot[j] else None}
    # Residual associations remove differences in task composition without using model outcomes to select cells.
    groups = collections.defaultdict(list)
    for i, row in enumerate(rows):
        groups[(row["set"], row["cond"], row["op_base"], row["var"], row["h"])].append(i)
    residuals = []
    for ids in groups.values():
        if len(ids) >= 3:
            v = values[ids]
            residuals.extend(v - np.mean(v, axis=0))
    out["composition_adjusted"] = dict(zip(("rho_L1", "rho_L3", "rho_difference"), compute(np.array(residuals)))) if residuals else None
    if all("process_zero_error" in r and "answer_zero_error" in r for r in rows):
        strength = np.array([[r["process_zero_error"], r["answer_zero_error"]] for r in rows])
        adjusted = []
        for ids in groups.values():
            # Within-task adjustment for the observed effect at both intermediate and final times.
            if len(ids) >= 6:
                x = np.column_stack([np.ones(len(ids)), np.log1p(strength[ids])])
                y = values[ids]
                adjusted.extend(y - x @ np.linalg.lstsq(x, y, rcond=None)[0])
        out["effect_strength_adjusted"] = {
            "n": len(adjusted),
            "correlations": dict(zip(("rho_L1", "rho_L3", "rho_difference"), compute(np.array(adjusted)))) if adjusted else None,
        }
        out["zero_effect_reference"] = {
            "process_win_fraction": float(np.mean(values[:, 2] < strength[:, 0])),
            "answer_win_fraction": float(np.mean(values[:, 3] < strength[:, 1])),
            "mean_process_error": float(np.mean(values[:, 2])),
            "mean_zero_process_error": float(np.mean(strength[:, 0])),
            "mean_answer_error": float(np.mean(values[:, 3])),
            "mean_zero_answer_error": float(np.mean(strength[:, 1])),
        }
    return out


def cell_key(r):
    return tuple(r.get(k) for k in ("world", "t1", "cond", "fmt", "task", "sub", "arm_kind", "op_base", "civ", "var", "turn"))


def analyze(run, raw, nboot):
    cells = [json.loads(x) for x in gzip.open(run / f"cot_cells_{raw}.jsonl.gz", "rt")]
    numeric = {cell_key(r): r for r in cells if r.get("kind") == "V" and r.get("status") == "ok" and r.get("required")}
    state_units = []; effects = []; coverage = collections.Counter()
    cost = collections.defaultdict(lambda: collections.Counter())
    formal_effect_mask = {}
    for fmt in ("table", "code"):
        path = run / f"scored_{raw}.ans.{fmt}.jsonl.gz"
        if path.exists():
            for line in gzip.open(path, "rt"):
                r = json.loads(line)
                if r.get("kind") == "3" and r.get("sub") == "1a":
                    k = (fmt, r["world"], r["t1"], r["cond"], r["op"], r["arm_kind"], r["civ"], r["var"], r["h"])
                    formal_effect_mask[k] = bool(r.get("in_band") and r.get("scored_var") and r.get("truth_moved") and "skill_arm" in r)
    for path in sorted((run / "questions").glob("*.jsonl.gz")):
        if not (run / raw / path.name).exists():
            continue
        questions = [json.loads(x) for x in gzip.open(path, "rt")]
        outputs = [json.loads(x) for x in gzip.open(run / raw / path.name, "rt")]
        for o in outputs:
            c = cost[o["fmt"]]
            c["requests"] += 1; c["generated_tokens"] += o.get("n_gen", 0)
            c["prompt_tokens"] += o.get("n_prompt", 0); c["rounds"] += o.get("n_rounds", 1)
            c["truncated"] += o.get("finish") == "length"
            c["errors"] += str(o.get("finish", "")).startswith("error:")
            c["runnable"] += bool(o.get("runnable"))
        answers = {}
        for fmt in ("free", "table", "code"):
            ap = run / f"{raw}.ans.{fmt}" / path.name
            if ap.exists():
                answers.update({key_of(o): parse_quantiles(o["output"]) for x in gzip.open(ap, "rt") if (o := json.loads(x))})
        by_q = {}
        for q in questions:
            if not q.get("evidence_cells") or q["fmt"] == "free":
                continue
            pred = answers.get(key_of(q), {})
            q_units = {}
            for civ, var in sorted({(c["civ"], c["var"]) for c in q["evidence_cells"]}):
                required = {c["turn"] for c in q["evidence_cells"] if c["civ"] == civ and c["var"] == var}
                for h in q["horizons"]:
                    coverage["state_expected"] += 1
                    turns = [q["t1"] + t for t in intermediate_offsets([h]) if q["t1"] + t in required]
                    source = "mainline_traj_pt" if q.get("arm_kind") == "control" else "traj_pt"
                    rows = [numeric.get(cell_key(dict(q, civ=civ, var=var, turn=t))) for t in turns]
                    ys = [level(q, civ, var, t, source) for t in turns]
                    final_truth = q["truth"] if q["task"] == "1a" else q["truth"]["1a"]
                    y = ((final_truth.get(str(h)) or {}).get(civ) or {}).get(var)
                    p = pred.get((civ, var, int(h)))
                    if not turns or any(r is None for r in rows) or any(v is None for v in ys) or y is None or p is None:
                        coverage["state_missing"] += 1; continue
                    pe = float(np.mean([abs(SV.tf(var, r["value"]) - SV.tf(var, v)) / SV.norm(var) for r, v in zip(rows, ys)]))
                    ae = abs(SV.tf(var, p[2]) - SV.tf(var, y)) / SV.norm(var)
                    tr = [SV.trend_forecast((q.get("base_hist") or {}).get(civ) or {}, var, q["t1"], t - q["t1"]) for t in turns]
                    trend = SV.trend_forecast((q.get("base_hist") or {}).get(civ) or {}, var, q["t1"], int(h))
                    er = float(np.mean([abs(SV.tf(var, z) - SV.tf(var, v)) / SV.norm(var) for z, v in zip(tr, ys)])) if all(z is not None for z in tr) else None
                    ar = abs(SV.tf(var, trend) - SV.tf(var, y)) / SV.norm(var) if trend is not None else None
                    u = {k: q.get(k) for k in ("world", "t1", "set", "cond", "fmt", "task", "sub", "arm_kind", "op_base")}
                    u.update(civ=civ, var=var, h=int(h), process_error=pe, answer_error=ae,
                             process_reference_error=er, answer_reference_error=ar, n_intermediate=len(turns))
                    state_units.append(u); q_units[(civ, var, int(h))] = u
            by_q[key_of(q)] = q_units
        for q in questions:
            if q["fmt"] == "free" or q.get("sub") != "1a" or q.get("arm_kind") not in ("real", "placebo"):
                continue
            ctrl = next((r for r in questions if r["fmt"] == q["fmt"] and r["cond"] == q["cond"] and
                         ((q["control_ref"]["1a"] == "main" and r["task"] == "1a") or
                          (q["control_ref"]["1a"] == "control" and r.get("arm_kind") == "control" and r.get("sub") == "1a"
                           and r.get("op_base") == q.get("op_base")))), None)
            if ctrl is None:
                coverage["missing_control_prompts"] += 1; continue
            pred = answers.get(key_of(q), {}); ctrl_pred = answers.get(key_of(ctrl), {})
            for civ, var in sorted({(c["civ"], c["var"]) for c in q["evidence_cells"]}):
                required = {c["turn"] for c in q["evidence_cells"] if c["civ"] == civ and c["var"] == var}
                for h in q["horizons"]:
                    coverage["effects_expected"] += 1
                    turns = [q["t1"] + t for t in intermediate_offsets([h]) if q["t1"] + t in required]
                    pe = []; ze = []
                    for t in turns:
                        r = numeric.get(cell_key(dict(q, civ=civ, var=var, turn=t)))
                        c = numeric.get(cell_key(dict(ctrl, civ=civ, var=var, turn=t)))
                        y = level(q, civ, var, t); yf = level(q, civ, var, t, "factual_traj_pt")
                        if r is None or c is None or y is None or yf is None:
                            break
                        e, z, _, _ = effect_error(var, r["value"], c["value"], y, yf)
                        pe.append(e); ze.append(z)
                    k = (civ, var, int(h))
                    y = ((q["truth"]["1a"].get(str(h)) or {}).get(civ) or {}).get(var)
                    yf = ((q["factual_arm"]["1a"].get(str(h)) or {}).get(civ) or {}).get(var)
                    l1 = by_q.get(key_of(ctrl), {}).get(k)
                    if not turns or len(pe) != len(turns) or k not in pred or k not in ctrl_pred or y is None or yf is None or l1 is None:
                        coverage["effects_missing"] += 1; continue
                    e, z, ph, yt = effect_error(var, pred[k][2], ctrl_pred[k][2], y, yf)
                    u = {key: q.get(key) for key in ("world", "t1", "set", "cond", "fmt", "op_base", "arm_kind")}
                    u.update(civ=civ, var=var, h=int(h), process_error=float(np.mean(pe)), process_zero_error=float(np.mean(ze)),
                             answer_error=e, answer_zero_error=z, predicted_effect=ph, observed_effect=yt,
                             l1_process_error=l1["process_error"], l1_answer_error=l1["answer_error"],
                             moved=y != yf, in_band=q["horizon_band"][0] <= h <= q["horizon_band"][1],
                             scored_var=civ == q.get("op_target") and var in q.get("score_vars", []))
                    u["formal_capture_eligible"] = formal_effect_mask.get((q["fmt"], q["world"], q["t1"], q["cond"], q["op_base"], q["arm_kind"], civ, var, int(h)), False)
                    effects.append(u)
    grouped = collections.defaultdict(list)
    for u in effects:
        if u["formal_capture_eligible"] and u["op_base"] != "L4-love":
            grouped[(u["fmt"], u["cond"], u["arm_kind"])].append(u)
    common = collections.defaultdict(dict)
    for u in state_units:
        if u["cond"] == "v1":
            common[tuple(u[k] for k in ("world", "t1", "task", "sub", "arm_kind", "op_base", "civ", "var", "h"))][u["fmt"]] = u
    paired = [v for v in common.values() if "table" in v and "code" in v]
    report = {"raw": raw, "coverage": dict(coverage), "cost": dict(cost), "common_state_units": len(paired),
              "associations": {"|".join(k): association(v, nboot) for k, v in grouped.items()},
              "limitations": ["Associations are observational, not training effects or internal-reasoning faithfulness.",
                              "Code intermediate and final outputs share a generated simulator.",
                              "Matched L1 includes unedited observational controls at L3-specific horizons."]}
    report["paired_point_forecasts"] = {}
    for scope in ("L1", "L3-state"):
        pairs = [p for p in paired if (p["table"]["task"] == "1a" if scope == "L1" else p["table"]["arm_kind"] == "real")]
        results = {}
        for fmt in ("table", "code"):
            by_setting = collections.defaultdict(list)
            for p in pairs:
                by_setting[p[fmt]["set"]].append(p[fmt])
            settings = {}
            for st, units in by_setting.items():
                valid = [u for u in units if u["process_reference_error"] is not None and u["answer_reference_error"] is not None]
                quadrants = collections.Counter()
                for u in valid:
                    quadrants[f"A{'+' if u['answer_error'] < u['answer_reference_error'] else '-'}E{'+' if u['process_error'] < u['process_reference_error'] else '-'}"] += 1
                setting = {"n": len(valid), "quadrants": {k: quadrants[k] / len(valid) if valid else None
                           for k in ("A+E+", "A+E-", "A-E+", "A-E-")}}
                for name in ("process", "answer"):
                    groups = collections.defaultdict(lambda: [0.0, 0.0])
                    for u in valid:
                        family = next(f for f, vs in SV.FAMILIES.items() if u["var"] in vs)
                        groups[(u["world"], family)][0] += u[f"{name}_error"]
                        groups[(u["world"], family)][1] += u[f"{name}_reference_error"]
                    world = collections.defaultdict(list)
                    for (w, _), (num, den) in groups.items():
                        if den > 0:
                            world[w].append(float(np.clip(1 - num / den, -1, 1)))
                    setting[f"{name}_skill"] = float(np.mean([np.mean(v) for v in world.values()])) if world else None
                settings[st] = setting
            results[fmt] = {"settings": settings, "aggregate": {
                k: float(np.mean([settings[st][k] for st in ("bench-basic", "bench-gen")]))
                if all(st in settings and settings[st][k] is not None for st in ("bench-basic", "bench-gen")) else None
                for k in ("process_skill", "answer_skill")}}
        report["paired_point_forecasts"][scope] = results
    summary = run / f"cot_scores_{raw}.json"
    if summary.exists():
        report["formal_answer_scores"] = json.loads(summary.read_text())["answer"]
    for name, units in (("state_units", state_units), ("effect_units", effects)):
        with gzip.open(run / f"aligned_{name}_{raw}.jsonl.gz", "wt") as fh:
            for u in units:
                fh.write(json.dumps(u) + "\n")
    (run / f"aligned_analysis_{raw}.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--bootstrap", type=int, default=2000)
    a = ap.parse_args()
    print(json.dumps(analyze(ROOT / "bench/runs" / a.run, a.raw, a.bootstrap), indent=2))


if __name__ == "__main__":
    main()
