"""Audit effect magnitudes conditional on correct directions in frozen panel outputs."""
import collections
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from bench.measure.effect_error_comparison import transform
from paper.gen_tables import LINES, MIN_OP_CELLS, ROOT, SETTINGS, m_l3, score_path


OUT = ROOT / "bench/measure/out/section5/direction_magnitude"
FIELDS = (
    "cells", "missing", "zero", "active", "correct", "wrong", "under_half",
    "within_factor_two", "over_twice", "correct_error", "correct_truth",
    "correct_prediction", "error", "truth", "over_true", "under_true",
)
INDEX = {name: index for index, name in enumerate(FIELDS)}


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def contributions(truth, prediction):
    if not math.isfinite(truth) or truth == 0:
        raise ValueError("Expected a finite, nonzero true effect")
    values = dict.fromkeys(FIELDS, 0.0)
    values.update(cells=1, truth=abs(truth))
    if prediction is None:
        values.update(missing=1, error=2 * abs(truth))
    else:
        if not math.isfinite(prediction):
            raise ValueError("Expected a finite prediction")
        values["error"] = abs(prediction - truth)
        if prediction == 0:
            values["zero"] = 1
        else:
            values["active"] = 1
            if prediction * truth > 0:
                values.update(correct=1, correct_error=values["error"], correct_truth=abs(truth),
                              correct_prediction=abs(prediction))
                category = ("under_half" if abs(prediction) < .5 * abs(truth)
                            else "over_twice" if abs(prediction) > 2 * abs(truth)
                            else "within_factor_two")
                values[category] = 1
                values["over_true"] = float(abs(prediction) > abs(truth))
                values["under_true"] = float(abs(prediction) < abs(truth))
            else:
                values["wrong"] = 1
    return np.array([values[name] for name in FIELDS])


def metrics(totals, correct_operators):
    combined = totals.sum(axis=0)

    def ratio(numerator, denominator):
        return float(combined[INDEX[numerator]] / combined[INDEX[denominator]]) if combined[INDEX[denominator]] else None

    correct_totals = totals[correct_operators]
    conditional = (1 - correct_totals[:, INDEX["correct_error"]] / correct_totals[:, INDEX["correct_truth"]]
                   if len(correct_totals) else np.array([]))
    return dict(
        direction_valid_active=ratio("correct", "active"),
        correct_among_valid=(float(combined[INDEX["correct"]] / (combined[INDEX["cells"]] - combined[INDEX["missing"]]))
                             if combined[INDEX["cells"]] > combined[INDEX["missing"]] else None),
        under_half=ratio("under_half", "correct"),
        within_factor_two=ratio("within_factor_two", "correct"),
        over_twice=ratio("over_twice", "correct"),
        over_true=ratio("over_true", "correct"),
        under_true=ratio("under_true", "correct"),
        correct_mae=ratio("correct_error", "correct"),
        correct_zero_mae=ratio("correct_truth", "correct"),
        correct_capture=float(np.clip(conditional, -1, 1).mean()) if len(conditional) else None,
        correct_capture_unclipped=float(conditional.mean()) if len(conditional) else None,
        capture=float(np.clip(1 - totals[:, INDEX["error"]] / totals[:, INDEX["truth"]], -1, 1).mean()),
    )


def summarize(rows, bootstrap=1000):
    counts = collections.Counter(key[2] for key, truth, prediction in rows)
    rows = [row for row in rows if counts[row[0][2]] >= MIN_OP_CELLS]
    if not rows:
        return None
    operators = sorted({key[2] for key, truth, prediction in rows})
    worlds = sorted({key[0] for key, truth, prediction in rows})
    operator_index = {name: index for index, name in enumerate(operators)}
    world_index = {name: index for index, name in enumerate(worlds)}
    sums = np.zeros((len(worlds), len(operators), len(FIELDS)))
    for key, truth, prediction in rows:
        sums[world_index[key[0]], operator_index[key[2]]] += contributions(truth, prediction)
    totals = sums.sum(axis=0)
    correct_operators = totals[:, INDEX["correct"]] >= MIN_OP_CELLS
    result = metrics(totals, correct_operators)
    relative_magnitudes = [abs(prediction) / abs(truth) for key, truth, prediction in rows
                           if prediction is not None and prediction * truth > 0]
    if not all(math.isfinite(value) and value > 0 for value in relative_magnitudes):
        raise ValueError("Invalid correct-sign relative magnitude")
    result["relative_magnitude"] = (
        dict(zip(("q25", "median", "q75"), map(float, np.quantile(relative_magnitudes, [.25, .5, .75]))))
        if relative_magnitudes else None)
    result.update(n_worlds=len(worlds), n_operators=len(operators),
                  n_correct_operators=int(correct_operators.sum()),
                  counts={name: int(totals[:, INDEX[name]].sum()) for name in FIELDS[:9]},
                  correct_capture_cells=int(totals[correct_operators, INDEX["correct"]].sum()),
                  by_operator={})
    for operator, index in operator_index.items():
        result["by_operator"][operator] = dict(
            metrics(totals[index:index+1], correct_operators[index:index+1]),
            n_worlds=int(np.count_nonzero(sums[:, index, INDEX["cells"]])),
            n_correct_worlds=int(np.count_nonzero(sums[:, index, INDEX["correct"]])),
            counts={name: int(totals[index, INDEX[name]]) for name in FIELDS[:9]})
    if bootstrap:
        rng = np.random.default_rng(20260923)
        draws = collections.defaultdict(list)
        for _ in range(bootstrap):
            weights = rng.exponential(size=len(worlds))
            trial = metrics((sums * weights[:, None, None]).sum(axis=0), correct_operators)
            for name, value in trial.items():
                if value is not None:
                    draws[name].append(value)
        intervals = {name: list(map(float, np.quantile(values, [.025, .975]))) for name, values in draws.items()}
        if any(value["n_worlds"] < 2 for value in result["by_operator"].values()):
            intervals.pop("capture", None)
        if any(value["n_correct_worlds"] < 2 for value in result["by_operator"].values()
               if value["counts"]["correct"] >= MIN_OP_CELLS):
            intervals.pop("correct_capture", None)
            intervals.pop("correct_capture_unclipped", None)
        result["ci95"] = intervals
    return result


def read_model(tag):
    source_path = score_path(tag)
    dump_path = source_path.with_name(source_path.name.replace("scores_", "scored_").replace(".json", ".jsonl.gz"))
    hashes = {str(path.relative_to(ROOT)): digest(path) for path in (source_path, dump_path)}
    scores = json.loads(source_path.read_text())
    data = collections.defaultdict(dict)
    excluded = collections.Counter()
    settings = {setting for setting, label in SETTINGS}
    with gzip.open(dump_path, "rt") as source:
        for line in source:
            row = json.loads(line)
            if (row.get("kind") != "3" or row.get("sub") != "1a" or row.get("arm_kind") != "real"
                    or not row.get("in_band") or not row.get("scored_var") or row.get("truth_moved") is not True):
                continue
            setting = f"{row['set']}|{row['cond']}"
            if setting not in settings:
                continue
            if any(row.get(field) is None for field in ("y", "y_fac")):
                continue
            if not math.isfinite(row["norm"]) or row["norm"] <= 0:
                raise ValueError(f"Invalid normalization: {tag}")
            missing = any(row.get(field) is None for field in ("p50", "p50_ignore"))
            if not missing and "skill_arm" not in row:
                excluded[setting] += 1
                continue
            truth = (transform(row["var"], row["y"]) - transform(row["var"], row["y_fac"])) / row["norm"]
            prediction = (None if missing else (transform(row["var"], row["p50"]) - transform(row["var"], row["p50_ignore"])) / row["norm"])
            key = tuple(row[field] for field in ("world", "t1", "op", "civ", "var", "h"))
            if key in data[setting]:
                raise ValueError(f"Duplicate effect cell: {tag} {key}")
            contributions(truth, prediction)
            data[setting][key] = (truth, prediction)
    results = {}
    for setting in sorted(settings):
        rows = [(key, *values) for key, values in data[setting].items()]
        result = summarize(rows)
        expected_capture, expected_direction, _ = m_l3(scores, setting)
        if result is None or not math.isclose(result["capture"], expected_capture, abs_tol=1e-10):
            raise ValueError(f"Main capture mismatch: {tag} {setting}")
        weighted_direction = 0.0
        direction_weight = 0
        for operator in sorted({key[2] for key, truth, prediction in rows}):
            operator_rows = [row for row in rows if row[0][2] == operator]
            total = sum((contributions(truth, prediction) for key, truth, prediction in operator_rows))
            denominator = total[INDEX["active"]] + total[INDEX["missing"]]
            if denominator:
                weighted_direction += total[INDEX["correct"]] / denominator * len(operator_rows)
                direction_weight += len(operator_rows)
        reconstructed = weighted_direction / direction_weight if direction_weight else None
        if not math.isclose(reconstructed, expected_direction, abs_tol=1e-10):
            raise ValueError(f"Main direction mismatch: {tag} {setting}")
        result["published_capture"] = expected_capture
        result["published_direction_raw"] = expected_direction
        result["excluded_valid_outside_main_mask"] = excluded[setting]
        thresholds = {}
        large_rows = []
        for operator in sorted({key[2] for key, truth, prediction in rows}):
            selected = [row for row in rows if row[0][2] == operator]
            threshold = float(np.quantile([abs(truth) for key, truth, prediction in selected], .25))
            thresholds[operator] = threshold
            large_rows.extend(row for row in selected if abs(row[1]) > threshold)
        result["above_operator_q25"] = summarize(large_rows, bootstrap=0)
        result["q25_thresholds"] = thresholds
        results[setting] = result
        print(tag, setting, "direction", round(result["direction_valid_active"], 3),
              "relative magnitude", result["relative_magnitude"], flush=True)
    for path in (source_path, dump_path):
        if digest(path) != hashes[str(path.relative_to(ROOT))]:
            raise ValueError(f"Source changed during analysis: {path}")
    return results, hashes


def write_report(result):
    lines = ["# Direction and effect magnitude", "", result["scope"], "", *result["protocol"], ""]
    for setting, label in SETTINGS:
        lines += [f"## {label}", "", "| Model | Eligible / missing / zero / correct | Direction, valid nonzero | Capture, correct sign | <0.5x | 0.5-2x | >2x | Magnitude median [Q25, Q75] |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for name, tag, aa in LINES:
            value = result["models"][tag][setting]
            counts = value["counts"]
            magnitude = value["relative_magnitude"]
            lines.append(f"| {tag} | {counts['cells']} / {counts['missing']} / {counts['zero']} / {counts['correct']} | "
                         f"{value['direction_valid_active']:.3f} | {value['correct_capture']:+.3f} | "
                         f"{value['under_half']:.1%} | {value['within_factor_two']:.1%} | {value['over_twice']:.1%} | "
                         f"{magnitude['median']:.1%} [{magnitude['q25']:.1%}, {magnitude['q75']:.1%}] |")
        lines.append("")
    (OUT / "report.md").write_text("\n".join(lines) + "\n")


def main():
    result = dict(models={}, provenance={},
                  scope="Current main-table sources, all three settings, real arms, target-band, truth-moved cells.",
                  protocol=[
                      "Main CAPTURE and raw Direction are reconstructed and checked against current score files.",
                      "The inherited main scoring mask excludes valid predictions without skill_arm, including zero-CRPS control cases. Counts are recorded; no scoring definition is changed.",
                      "Diagnostic direction is cell-weighted among valid nonzero responses, with a 0.5 random-sign reference. Missing and zero responses are reported separately.",
                      "Magnitude bins are conditional on correct direction and use the same transformed, normalized effect space as CAPTURE. Factor two is a descriptive tolerance, not a success criterion.",
                      "No per-cell division is needed for the bins. Same-sign predictions above twice the true magnitude have larger absolute error than predicting zero effect.",
                      "Relative magnitude is abs(predicted effect)/abs(realized effect) on correct-sign cells, equally weighted by cell. Median and Q25-Q75 use linear sample quantiles; the interval describes dispersion, not confidence. No ratios are clipped.",
                      "Correct-sign CAPTURE is a selected-subset diagnostic, not a corrected main score or evidence of deployable improvement. Operators require at least three correct-sign cells.",
                      "Intervals use 1000 world-level exponential-weight bootstrap draws with fixed eligible operators. Operator-macro intervals are omitted for singleton-world operators.",
                      "Sensitivity excludes effects at or below each operator's 25th percentile, set from all eligible truth magnitudes before selecting correct predictions.",
                      "Model samples differ. The analysis does not establish a matched model ranking, a noise-free effect, or the benefit of a calibration intervention.",
                  ])
    for name, tag, aa in LINES:
        result["models"][tag], source_hashes = read_model(tag)
        result["provenance"].update(source_hashes)
    result["analysis_sha256"] = digest(Path(__file__))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    write_report(result)
    print(f"Wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
