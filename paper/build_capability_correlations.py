"""Audit the main-table capability association without changing benchmark scores."""
import hashlib
import json
import math
from statistics import StatisticsError, correlation

from paper.gen_tables import (
    LINES, ROOT, SETTINGS, adjusted_accuracy, ceiling, load, score_path,
    m_1a, m_1b, m_l3, m_rules, m_traits, rules_ceiling,
)


METRICS = ("quantity", "event", "trait", "rules", "capture", "direction", "specificity")
LABELS = (r"\QNT", r"\EVT", r"\TRT", r"\RUL", "Capture", "Direction", "Specificity")


def collect_panel():
    records = []
    references = ceiling()
    perfect = load("perfect")
    for name, tag, index in LINES:
        if index is None:
            continue
        path = score_path(tag)
        contents = path.read_bytes()
        scores = json.loads(contents)
        values = {}
        for setting, label in SETTINGS:
            capture, direction, specificity = m_l3(scores, setting)
            values[setting] = dict(zip(METRICS, (
                m_1a(scores, setting), m_1b(scores, setting),
                adjusted_accuracy(m_traits(scores, setting), .5, references[setting]["traits"]),
                adjusted_accuracy(m_rules(scores, setting)[0], m_rules(perfect, setting)[1],
                                  rules_ceiling(references, setting)), capture,
                adjusted_accuracy(direction, .5, references[setting]["dir"]), specificity,
            )))
        records.append(dict(
            name=name, tag=tag, index=index, values=values,
            source=str(path.relative_to(ROOT)), sha256=hashlib.sha256(contents).hexdigest(),
        ))
    return records


def summarize(records):
    if len(records) < 3:
        raise ValueError("At least three model lines are required")
    associations = {}
    for setting, label in SETTINGS:
        associations[setting] = {}
        for metric in METRICS:
            indices = [record["index"] for record in records]
            values = [record["values"][setting][metric] for record in records]
            if any(value is None or not math.isfinite(value) for value in indices + values):
                raise ValueError(f"Incomplete panel for {setting}: {metric}")
            try:
                rho = correlation(indices, values, method="ranked")
            except StatisticsError as error:
                raise ValueError(f"Undefined rank correlation for {setting}: {metric}") from error
            associations[setting][metric] = dict(rho=rho, n=len(records))
    return associations


def render_table(associations, count):
    lines = [
        r"\begin{table}[t]\centering\small",
        r"\caption{Spearman rank correlations between the external capability index and",
        rf"the unrounded main-table scores across {count} unassisted model lines in each setting.",
        r"All seven metrics are included; tied values receive average ranks. The assisted",
        r"\expert\ line is excluded. These are descriptive model-level associations, not",
        r"matched-anchor comparisons or significance tests.}",
        r"\label{tab:capabilitycorr}",
        r"\begin{tabular}{@{}lrrr@{}}\toprule",
        r"Metric & \omniID & \ego & \omniOOD \\ \midrule",
    ]
    for metric, label in zip(METRICS, LABELS):
        values = [f"${associations[setting][metric]['rho']:+.2f}$" for setting, _ in SETTINGS]
        lines.append(label + " & " + " & ".join(values) + r" \\")
    lines.extend([r"\bottomrule\end{tabular}", r"\end{table}"])
    return "\n".join(lines) + "\n"


def main():
    records = collect_panel()
    associations = summarize(records)
    result = dict(
        method="Spearman correlation; average ranks for ties; one observation per model",
        index_source="Artificial Analysis Intelligence Index v4.3 as recorded in paper.gen_tables.LINES",
        scope="Descriptive main-table audit; no expert, pooled settings, p-values, or composite score",
        records=records, correlations=associations,
    )
    output = ROOT / "bench/measure/out/section5/capability_correlations.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    (ROOT / "paper/tables/capability_correlations.tex").write_text(render_table(associations, len(records)))
    print(json.dumps(associations, indent=2))


if __name__ == "__main__":
    main()
