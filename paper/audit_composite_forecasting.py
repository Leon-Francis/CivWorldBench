"""Fixed-weight composite diagnostics; no fitting and no manuscript changes."""
import hashlib
import itertools
import json
import math
from statistics import StatisticsError, correlation, mean

from paper.build_capability_correlations import METRICS
from paper.build_external_forecasting_audit import OUTPUT
from paper.gen_tables import SETTINGS, ceiling, rules_ceiling


SCHEMES = {
    "levels_equal": dict(quantity=1/6, event=1/6, trait=1/6, rules=1/6,
                         capture=1/9, direction=1/9, specificity=1/9),
    "columns_equal": dict.fromkeys(METRICS, 1/7),
    "accuracy_only": dict.fromkeys(METRICS[:-1], 1/6),
    "forecast_only": dict(quantity=.5, event=.5),
}


def reference_ranges(references, setting):
    upper = references[setting]
    return dict(quantity=(0, upper["macro"]), event=(0, upper["1b"]),
                trait=(0, upper["traits"]), rules=(0, rules_ceiling(references, setting)),
                capture=(-1, upper["capture"]), direction=(0, upper["dir"]), specificity=(0, 1))


def normalize(values, ranges):
    normalized = {}
    for metric in METRICS:
        value = values[metric]
        floor, upper = ranges[metric]
        if value is None or not math.isfinite(value) or not floor < upper:
            raise ValueError(f"Invalid score or reference range: {metric}")
        normalized[metric] = (value - floor) / (upper - floor)
    return normalized


def composite(values, weights):
    if not weights or any(weight < 0 or not math.isfinite(weight) for weight in weights.values()):
        raise ValueError("Weights must be finite and nonnegative")
    if not math.isclose(sum(weights.values()), 1):
        raise ValueError("Weights must sum to one")
    if any(metric not in values or not math.isfinite(values[metric]) for metric in weights):
        raise ValueError("Every weighted metric must be present and finite")
    return 100 * sum(values[metric] * weight for metric, weight in weights.items())


def association(local, external):
    if len(local) != len(external) or not 3 <= len(local) <= 8:
        raise ValueError("Exact diagnostic requires 3--8 paired models")
    if any(value is None or not math.isfinite(value) for value in local + external):
        raise ValueError("Missing/nonfinite paired score")
    try:
        pearson = correlation(local, external)
        spearman = correlation(local, external, method="ranked")
    except StatisticsError:
        return dict(n=len(local), pearson=None, spearman=None, reason="Constant score vector")
    null = [correlation(local, permuted) for permuted in itertools.permutations(external)]
    return dict(n=len(local), pearson=pearson, spearman=spearman,
                permutation_p=sum(abs(value) >= abs(pearson) - 1e-12 for value in null) / len(null))


def build(audit, ranges):
    schemes = {}
    for name, weights in SCHEMES.items():
        scores = {}
        for tag, record in audit["civladder_records"].items():
            scores[tag] = {setting: composite(normalize(record["values"][setting], ranges[setting]), weights)
                           for setting, _ in SETTINGS}
            scores[tag]["settings_mean"] = mean(scores[tag].values())
        comparisons = []
        for variant in audit["diagnostics"]:
            tags = variant["tags"]
            external = [row["Overall"] for row in variant["external_rows"]]
            comparisons.append(dict(tags=tags, external_rows=variant["external_rows"],
                associations={setting: association([scores[tag][setting] for tag in tags], external)
                              for setting in [key for key, _ in SETTINGS] + ["settings_mean"]}))
        schemes[name] = dict(weights=weights, scores=scores, comparisons=comparisons)
    return schemes


def render(result):
    lines = ["# Fixed-weight Composite Diagnostic", "",
             "Exploratory only. Weights are fixed design choices, not fitted to external scores.",
             "No main-table score, figure, or manuscript has been replaced.", "",
             "All components use (score - displayed floor)/(displayed ceiling - floor),",
             "without clipping or sample-min/max scaling. The aggregate is multiplied by 100.",
             "This is an index, not accuracy or a common probability of task completion.", "",
             "- levels_equal: each level has weight 1/3; equal weights within each level.",
             "- columns_equal: all seven metrics have weight 1/7.",
             "- accuracy_only: six metrics have weight 1/6; specificity excluded as a selectivity diagnostic.",
             "- forecast_only: quantity and event each have weight 1/2; NOT an overall ladder score.", "",
             "settings_mean averages the three settings per model. It is still three model observations,",
             "not nine independent observations. No external rows are pooled across configurations.", "",
             "## Pearson Correlations", "",
             "Ranges below span both published Opus configurations, not confidence intervals.", "",
             "| Scheme | Omni-ID | Ego | OOD | Settings mean |",
             "| --- | ---: | ---: | ---: | ---: |"]
    keys = [key for key, _ in SETTINGS] + ["settings_mean"]
    for name, scheme in result["schemes"].items():
        cells = []
        for setting in keys:
            values = [item["associations"][setting]["pearson"] for item in scheme["comparisons"]]
            cells.append("undefined" if any(value is None for value in values)
                         else f"{min(values):.3f} to {max(values):.3f}")
        lines.append("| " + name + " | " + " | ".join(cells) + " |")
    lines.extend(["", "## Interpretation", "",
                  "These are three-model in-sample descriptions, with mismatched assistance and anchor scopes.",
                  "Pearson measures linear agreement of paired model scores, not equality of marginal distributions.",
                  "Neither weights nor an affine calibration were fitted. No held-out validation was performed.",
                  "The exact permutation p-values in the JSON are unadjusted diagnostics, not question-level uncertainty.",
                  "Prior results on these models were already known; this is not a preregistered confirmatory analysis.",
                  "Do not select the largest correlation and present it as independent external validation.", "",
                  "A benchmark index requires a content-based weighting decision, frozen reference scales, and",
                  "validation on additional models under matched protocols. Learning weights instead requires a",
                  "separate calibration panel and held-out models/model families; three or six models cannot",
                  "reliably support a seven-metric fitted index. Specificity exclusion is a sensitivity check,",
                  "not a conclusion justified by the observed external correlation.", "",
                  "Reproduce: `.venv/bin/python -m paper.audit_composite_forecasting`."])
    return "\n".join(lines) + "\n"


def main():
    path = OUTPUT / "audit.json"
    contents = path.read_bytes()
    audit = json.loads(contents)
    references = ceiling()
    ranges = {setting: reference_ranges(references, setting) for setting, _ in SETTINGS}
    result = dict(status="Exploratory fixed-weight candidates; no fitted or adopted final score",
                  input_sha256=hashlib.sha256(contents).hexdigest(), references=ranges,
                  schemes=build(audit, ranges))
    (OUTPUT / "composite_audit.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    report = render(result)
    (OUTPUT / "composite_audit.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
