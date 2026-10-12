"""Offline comparison of normalized paired-effect errors with published CAPTURE."""
import collections
from concurrent.futures import ProcessPoolExecutor
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from bench.spec_v2 import LOG_VARS
from paper.gen_tables import LINES, MIN_OP_CELLS, SETTINGS, TAG_ALIAS, m_l3

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "bench/runs/v3-bench-br"
OUT = ROOT / "bench/measure/out/effect_error_comparison"


def transform(variable, value):
    return math.copysign(math.log1p(abs(value)), value) if variable in LOG_VARS else value


def summarize(rows, bootstrap=1000):
    counts = collections.Counter(key[2] for key, truth, prediction in rows)
    rows = [row for row in rows if counts[row[0][2]] >= MIN_OP_CELLS]
    if not rows:
        return None
    operators = sorted({key[2] for key, truth, prediction in rows})
    worlds = sorted({key[0] for key, truth, prediction in rows})
    operator_index = {name: index for index, name in enumerate(operators)}
    world_index = {name: index for index, name in enumerate(worlds)}
    sums = np.zeros((len(worlds), len(operators), 10))
    errors = []
    for key, truth, prediction in rows:
        error = abs(prediction - truth)
        errors.append(error)
        category = (6 if prediction == 0 else 7 if prediction * truth < 0
                    else 8 if abs(prediction) > abs(truth) else 9)
        values = np.array([1, error, error**2, abs(truth), truth**2,
                           float(prediction * truth > 0), 0, 0, 0, 0])
        values[category] = error
        sums[world_index[key[0]], operator_index[key[2]]] += values

    def metrics(totals):
        means = totals / totals[:, :1]
        capture = 1 - totals[:, 1] / totals[:, 3]
        return dict(delta_mae=float(means[:, 1].mean()),
                    delta_rmse=float(np.sqrt(means[:, 2]).mean()),
                    zero_mae=float(means[:, 3].mean()),
                    zero_rmse=float(np.sqrt(means[:, 4]).mean()),
                    mae_minus_zero=float((means[:, 1] - means[:, 3]).mean()),
                    capture=float(np.clip(capture, -1, 1).mean()),
                    capture_unclipped=float(capture.mean()))

    totals = sums.sum(axis=0)
    result = metrics(totals)
    result.update(n_cells=len(rows), n_worlds=len(worlds), n_operators=len(operators),
                  cell_error_median=float(np.median(errors)), cell_error_p90=float(np.quantile(errors, .9)))
    result["error_components"] = dict(zip(
        ("no_response", "wrong_direction", "same_direction_overshoot", "same_direction_other"),
        map(float, (totals[:, 6:] / totals[:, :1]).mean(axis=0))))
    result["by_operator"] = {
        operator: dict(metrics(totals[index:index+1]), n_cells=int(totals[index, 0]),
                       n_worlds=int(np.count_nonzero(sums[:, index, 0])))
        for index, operator in enumerate(operators)
    }
    result["interval_supported"] = all(value["n_worlds"] >= 2 for value in result["by_operator"].values())
    if bootstrap and result["interval_supported"]:
        rng = np.random.default_rng(0)
        draws = collections.defaultdict(list)
        for _ in range(bootstrap):
            weights = rng.exponential(size=len(worlds))
            for name, value in metrics((sums * weights[:, None, None]).sum(axis=0)).items():
                draws[name].append(value)
        result["ci95"] = {name: list(map(float, np.quantile(values, [.025, .975])))
                          for name, values in draws.items()}
        result["interval_method"] = "paired world-level Bayesian bootstrap; exponential weights; 1000 draws"
    elif bootstrap:
        result["interval_warning"] = "Interval omitted: at least one operator has only one world; fixed-operator bootstrap cannot estimate its between-world uncertainty."
    return result


def read_model(tag):
    alias = TAG_ALIAS.get(tag, tag)
    score_path = RUN / f"scores_raw_{alias}.json"
    if not score_path.exists():
        alias += "_sub120"
        score_path = RUN / f"scores_raw_{alias}.json"
    dump = RUN / f"scored_raw_{alias}.jsonl.gz"
    score_bytes = score_path.read_bytes()
    scores = json.loads(score_bytes)
    data = collections.defaultdict(dict)
    eligible = collections.Counter()
    rejected = collections.Counter()
    settings = {key for key, label in SETTINGS}
    with gzip.open(dump, "rt") as source:
        for line in source:
            row = json.loads(line)
            if (row.get("kind") != "3" or row.get("sub") != "1a"
                    or row.get("arm_kind") != "real" or not row.get("in_band")
                    or not row.get("scored_var") or row.get("truth_moved") is not True):
                continue
            setting = f"{row['set']}|{row['cond']}"
            if setting not in settings:
                continue
            eligible[setting] += 1
            fields = ("p50", "p50_ignore", "y", "y_fac", "norm")
            if ("skill_arm" not in row or any(row.get(field) is None or not math.isfinite(row[field])
                                             for field in fields) or row["norm"] <= 0):
                rejected[setting] += 1
                continue
            truth = (transform(row["var"], row["y"]) - transform(row["var"], row["y_fac"])) / row["norm"]
            prediction = (transform(row["var"], row["p50"]) - transform(row["var"], row["p50_ignore"])) / row["norm"]
            key = tuple(row[field] for field in ("world", "t1", "op", "civ", "var", "h"))
            if key in data[setting]:
                raise ValueError(f"Duplicate cell: {tag} {setting} {key}")
            if truth == 0:
                raise ValueError(f"Moved cell has zero transformed effect: {key}")
            data[setting][key] = (truth, prediction)
    results = {}
    for setting in settings:
        result = summarize([(key, *values) for key, values in data[setting].items()])
        published = m_l3(scores, setting)[0]
        if result is not None:
            if published is None or not math.isclose(result["capture"], published, abs_tol=1e-10):
                raise ValueError(f"CAPTURE mismatch: {tag} {setting}: {result['capture']} != {published}")
            result.update(published_capture=published, scored_eligible_rows=eligible[setting],
                          rejected_rows=rejected[setting], valid_before_operator_filter=len(data[setting]))
        results[setting] = result
    print(f"Loaded and verified {tag}: {alias}", flush=True)
    return tag, dict(data), results, dict(score_file=str(score_path), dump_file=str(dump),
                                         score_sha256=hashlib.sha256(score_bytes).hexdigest())


def common_results(data, tags, anchors=None):
    output = {}
    for setting, label in SETTINGS:
        subsets = [{key for key in data[tag][setting]
                    if anchors is None or f"{key[0]}_{key[1]}.jsonl.gz" in anchors} for tag in tags]
        common = sorted(set.intersection(*subsets))
        for key in common:
            truths = [data[tag][setting][key][0] for tag in tags]
            if not all(math.isclose(truth, truths[0], rel_tol=1e-10, abs_tol=1e-12) for truth in truths):
                raise ValueError(f"Common-cell truth mismatch: {setting} {key}")
        output[setting] = dict(n_common_before_operator_filter=len(common),
                               cells=[list(key) for key in common],
                               models={tag: summarize([(key, *data[tag][setting][key]) for key in common])
                                       for tag in tags})
    return output


def main():
    tags = [tag for name, tag, index in LINES]
    data, available, provenance = {}, {}, {}
    with ProcessPoolExecutor(max_workers=4) as pool:
        for tag, rows, results, source in pool.map(read_model, tags):
            data[tag], available[tag], provenance[tag] = rows, results, source
    anchors = set((ROOT / "bench/v2/api_subset_60of120.txt").read_text().split())
    model_tags = [tag for tag in tags if tag != "fable"]
    common = common_results(data, model_tags, anchors)
    expert_common = common_results(data, tags, anchors)
    full_tags = [tag for tag in model_tags if tag not in ("opus5", "gpt56sol")
                 and "_sub120" not in provenance[tag]["score_file"]]
    full_common = common_results(data, full_tags)
    output = dict(available=available, common_api60=common, common_with_expert=expert_common,
                  common_full_open_models=full_common,
                  provenance=provenance, min_operator_cells=MIN_OP_CELLS,
                  scope="Real-minus-control, target-band, truth-moved, same valid cells as current CAPTURE",
                  aggregation="Equal operator mean; normalized transformed units; no error clipping",
                  caveats=["Available-set rows have different sample coverage; do not treat them as a matched ranking.",
                           "Common intersections may be small and omit operators; counts are reported.",
                           "Observed paired stochastic effects, not noise-free conditional mean effects."])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    lines = ["# Offline effect-error comparison", "", output["scope"], "", output["aggregation"], "",
             "MAE/RMSE: lower is better. Zero is mathematical perfection, not an estimated attainable minimum.",
             "Each zero-effect baseline uses exactly the same cells and operator weights as its model.",
             "Intervals: 1000 paired world-level Bayesian bootstrap draws. No formal scores changed.", ""]
    groups = [("Available set", {setting: {tag: available[tag][setting] for tag in tags}
                                  for setting, label in SETTINGS}),
              ("API60 common model cells", {setting: value["models"] for setting, value in common.items()}),
              ("Full open-model common cells", {setting: value["models"] for setting, value in full_common.items()}),
              ("Common cells including expert", {setting: value["models"] for setting, value in expert_common.items()})]
    for title, group in groups:
        for setting, models in group.items():
            lines += [f"## {title}: {setting}", "",
                      "| Model | Cells / worlds / ops | MAE | Zero MAE | MAE minus zero (95% interval) | RMSE | Zero RMSE | CAPTURE |",
                      "|---|---|---:|---:|---|---:|---:|---:|"]
            for tag, result in models.items():
                if result is None:
                    lines.append(f"| {tag} | No eligible operators | | | | | | |")
                    continue
                interval = result.get("ci95", {}).get("mae_minus_zero")
                interval_text = (f"[{interval[0]:+.4f}, {interval[1]:+.4f}]" if interval
                                 else "[not estimated: single-world operator]")
                lines.append(f"| {tag} | {result['n_cells']} / {result['n_worlds']} / {result['n_operators']} | "
                             f"{result['delta_mae']:.4f} | {result['zero_mae']:.4f} | "
                             f"{result['mae_minus_zero']:+.4f} {interval_text} | "
                             f"{result['delta_rmse']:.4f} | {result['zero_rmse']:.4f} | {result['capture']:+.4f} |")
            lines.append("")
    lines += ["## Caveats", ""] + [f"- {item}" for item in output["caveats"]]
    (OUT / "report.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
