"""Compare existing model scores across separate public forecasting snapshots."""
import csv
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import io
import itertools
import json
from statistics import StatisticsError, correlation, mean
import re

from paper.audit_composite_forecasting import (
    SCHEMES, association, composite, normalize, reference_ranges,
)
from paper.build_capability_correlations import METRICS, collect_panel
from paper.build_external_forecasting_audit import OUTPUT, parse_forecastbench
from paper.gen_tables import (
    ROOT, RUN, SETTINGS, TAG_ALIAS, adjusted_accuracy, ceiling, load,
    m_1a, m_1b, m_l3, m_rules, m_traits, rules_ceiling,
)


SETTING_KEYS = [key for key, _ in SETTINGS] + ["settings_mean"]
SETTING_LABELS = dict(SETTINGS) | {"settings_mean": "Settings mean"}
FB_NAMES = {
    "opus5": "claude-opus-5-adaptive-thinking-max-web-search-code-execution-128k",
    "kimi-k3": "kimi-k3-max-128k",
    "gpt56sol": "gpt-5.6-sol-standard-medium-web-search-code-execution",
    "gemma4-31b": "gemma-4-31b-it",
}
FX_NAMES = {"claude-opus-5": "opus5", "Kimi K3": "kimi-k3", "gpt-5.6-sol": "gpt56sol"}
PA_NAMES = {"glm-5.3": "glm-5.3", "kimi-k3": "kimi-k3", "gpt-5.6-sol": "gpt56sol"}


def markdown_rows(content):
    table = "\n".join(line for line in content.splitlines() if line.startswith("|"))
    rows = [[cell.strip() for cell in row[1:-1]]
            for row in csv.reader(io.StringIO(table), delimiter="|", quoting=csv.QUOTE_NONE)]
    if len(rows) < 2:
        raise ValueError("No markdown table found")
    header = rows[0]
    output = []
    for row in rows[2:]:
        if len(row) != len(header):
            raise ValueError("Inconsistent markdown table columns")
        output.append(dict(zip(header, row)))
    return output


def forecastbench_groups(rows, with_inkling=False):
    names = dict(FB_NAMES)
    if with_inkling:
        names["inkling"] = "Inkling with search"
    groups = {tag: [] for tag in names}
    for row in rows:
        for tag, model in names.items():
            if row["Model"].rstrip("\u2020") == model:
                groups[tag].append(dict(model=row["Model"], score=row["Overall"], source_row=row))
    return groups


def futurex_groups(content):
    groups = {tag: [] for tag in FX_NAMES.values()}
    for row in markdown_rows(content):
        tag = FX_NAMES.get(row["MODEL"])
        if tag is not None:
            groups[tag].append(dict(model=row["MODEL"], framework=row["FRAMEWORK"],
                                    score=float(row["SCORE"]), source_row=row))
    return groups


def prophet_groups(content):
    groups = {tag: [] for tag in PA_NAMES.values()}
    for row in markdown_rows(content):
        match = re.search(r"\]\(https://www\.prophetarena\.co/model/([^\s)]+)", row["Model"])
        tag = PA_NAMES.get(match.group(1)) if match else None
        if tag is None or row["Type"] != "Fixed Context":
            continue
        skill_key = next(key for key in row if key.startswith("Skill "))
        value = re.match(r"[+-]?\d+(?:\.\d+)?", row[skill_key])
        if value is None:
            raise ValueError("Missing Prophet Arena skill value")
        groups[tag].append(dict(model=match.group(1), framework=row["Type"],
                                score=float(value.group()), reported_skill=row[skill_key], source_row=row))
    return groups


def variants(groups):
    if not groups or any(not rows for rows in groups.values()):
        raise ValueError("Missing model group; do not silently shrink the model panel")
    return [dict(zip(groups, chosen)) for chosen in itertools.product(*groups.values())]


def sensitivity(local, external, tags):
    result = association(local, external)
    result["leave_one_out"] = {}
    for index, tag in enumerate(tags):
        reduced_local = local[:index] + local[index + 1:]
        reduced_external = external[:index] + external[index + 1:]
        try:
            pearson = correlation(reduced_local, reduced_external)
            spearman = correlation(reduced_local, reduced_external, method="ranked")
        except StatisticsError:
            pearson = spearman = None
        result["leave_one_out"][tag] = dict(n=len(reduced_local), pearson=pearson, spearman=spearman)
    return result


def model_scores(panel, ranges):
    scores = {}
    for tag, record in panel.items():
        scores[tag] = {name: {} for name in list(SCHEMES) + list(METRICS)}
        for setting, _ in SETTINGS:
            values = normalize(record["values"][setting], ranges[setting])
            for scheme, weights in SCHEMES.items():
                scores[tag][scheme][setting] = composite(values, weights)
            for metric in METRICS:
                scores[tag][metric][setting] = 100 * values[metric]
        for values in scores[tag].values():
            values["settings_mean"] = mean(values.values())
    return scores


def compare(groups, scores):
    output = []
    tags = list(groups)
    for selected in variants(groups):
        external = [selected[tag]["score"] for tag in tags]
        results = {}
        for name in list(SCHEMES) + list(METRICS):
            results[name] = {setting: sensitivity(
                [scores[tag][name][setting] for tag in tags], external, tags) for setting in SETTING_KEYS}
        output.append(dict(tags=tags, external_rows=selected, results=results))
    return output


def verify_anchors(path, expected):
    observed = set()
    counts = Counter()
    with gzip.open(path, "rt") as stream:
        for line in stream:
            record = json.loads(line)
            observed.add(f"{record['world']}_{record['t1']}.jsonl.gz")
            counts[record["kind"]] += 1
    if observed != expected:
        raise ValueError(f"Anchor mismatch in {path.name}: missing={expected-observed}, extra={observed-expected}")
    return dict(anchors=len(observed), record_counts=dict(counts),
                scored_source=str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
                scored_sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def collect_common30(tags, references):
    anchor_path = ROOT / "bench/v2/human_line_30.txt"
    expected = {line.strip() for line in anchor_path.read_text().splitlines() if line.strip()}
    if len(expected) != 30:
        raise ValueError("Expected the frozen 30-anchor subset")
    perfect = load("perfect")
    records = {}
    for tag in sorted(tags):
        alias = TAG_ALIAS.get(tag, tag)
        path = RUN / f"scores_raw_{alias}_h30.json"
        contents = path.read_bytes()
        scores = json.loads(contents)
        coverage = verify_anchors(RUN / f"scored_raw_{alias}_h30.jsonl.gz", expected)
        values = {}
        for setting, _ in SETTINGS:
            capture, direction, specificity = m_l3(scores, setting)
            values[setting] = dict(zip(METRICS, (
                m_1a(scores, setting), m_1b(scores, setting),
                adjusted_accuracy(m_traits(scores, setting), .5, references[setting]["traits"]),
                adjusted_accuracy(m_rules(scores, setting)[0], m_rules(perfect, setting)[1],
                                  rules_ceiling(references, setting)), capture,
                adjusted_accuracy(direction, .5, references[setting]["dir"]), specificity,
            )))
        records[tag] = dict(tag=tag, values=values, source=str(path.relative_to(ROOT)),
                            sha256=hashlib.sha256(contents).hexdigest(), coverage=coverage)
    return dict(records=records, anchors=sorted(expected),
                anchor_source=str(anchor_path.relative_to(ROOT)),
                anchor_sha256=hashlib.sha256(anchor_path.read_bytes()).hexdigest())


def number_range(values):
    valid = [value for value in values if value is not None]
    if not valid:
        return "undefined"
    low, high = min(valid), max(valid)
    return f"{low:+.3f}" if abs(high - low) < 1e-12 else f"{low:+.3f} to {high:+.3f}"


def render(result):
    lines = ["# Existing-results External-validity Analysis", "",
             "Exploratory only; no model inference, learned weights, or manuscript changes.",
             "Source snapshots are dated 2026-09-22. Each dataset is analyzed separately.", "",
             "The same four previously defined weight schemes are retained. Each metric is normalized",
             "with the displayed floor/ceiling; each setting is computed separately, and settings_mean",
             "averages three settings per model without treating them as independent observations.", "",
             "## Cross-benchmark Pearson Correlations", "",
             "This table uses settings_mean. Ranges reflect all eligible published configurations, not CIs.", "",
             "| Dataset | N models | Variants | Levels equal | Columns equal | Accuracy only | Forecast only |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, dataset in result["datasets"].items():
        comparisons = dataset["comparisons"]
        cells = [number_range([item["results"][scheme]["settings_mean"]["pearson"]
                               for item in comparisons]) for scheme in SCHEMES]
        lines.append(f"| {name} | {len(dataset['groups'])} | {len(comparisons)} | " + " | ".join(cells) + " |")
    for name, dataset in result["datasets"].items():
        comparisons = dataset["comparisons"]
        lines.extend(["", "## " + name, "", dataset["scope"], "", "### Model Scores", "",
                      "| Model | External score | CivLadder levels-equal index (settings mean) |",
                      "| --- | ---: | ---: |"])
        for tag, rows in dataset["groups"].items():
            lines.append(f"| {tag} | {number_range([row['score'] for row in rows])} | "
                         f"{result['model_scores'][tag]['levels_equal']['settings_mean']:.3f} |")
        lines.extend(["", "### Setting Sensitivity: Levels Equal", "",
                      "| Setting | Pearson | Spearman | Exact permutation p |",
                      "| --- | ---: | ---: | ---: |"])
        for setting in SETTING_KEYS:
            cells = [number_range([item["results"]["levels_equal"][setting][metric]
                                   for item in comparisons]) for metric in ("pearson", "spearman", "permutation_p")]
            lines.append("| " + SETTING_LABELS[setting] + " | " + " | ".join(cells) + " |")
        lines.extend(["", "### Drop One Model: Levels Equal, Settings Mean", "",
                      "| Omitted model | Remaining models | Pearson |", "| --- | ---: | ---: |"])
        for tag in dataset["groups"]:
            values = [item["results"]["levels_equal"]["settings_mean"]["leave_one_out"][tag]["pearson"]
                      for item in comparisons]
            lines.append(f"| {tag} | {len(dataset['groups']) - 1} | {number_range(values)} |")
        lines.extend(["", "### Individual Metrics: Settings Mean", "",
                      "| Metric | Pearson | Spearman |", "| --- | ---: | ---: |"])
        for metric in METRICS:
            cells = [number_range([item["results"][metric]["settings_mean"][measure]
                                   for item in comparisons]) for measure in ("pearson", "spearman")]
            lines.append("| " + metric + " | " + " | ".join(cells) + " |")
    lines.extend(["", "## Common 30-anchor Sensitivity", "",
                  "All six relevant models' scored-record files contain exactly the frozen 30 anchors.",
                  "Existing score summaries are re-extracted with current metric functions; no inference",
                  "or scorer rerun. Common anchors do not imply identical valid answer cells. Full-panel",
                  "reference scales are held fixed. These are smaller-sample sensitivity checks, not replacements.", "",
                  "| Dataset | Levels equal | Columns equal | Accuracy only | Forecast only |",
                  "| --- | ---: | ---: | ---: | ---: |"])
    for name, comparisons in result["common30"]["comparisons"].items():
        cells = [number_range([item["results"][scheme]["settings_mean"]["pearson"]
                               for item in comparisons]) for scheme in SCHEMES]
        lines.append("| " + name + " | " + " | ".join(cells) + " |")
    lines.extend(["", "## Limits", "",
                  "- These are model-name candidates; checkpoint and serving equivalence are not independently certified.",
                  "- All 11 unassisted local model records are retained. Matching does not use local performance.",
                  "- No pooling across datasets or variants; repeated frameworks are not extra models.",
                  "- Main-panel anchor scopes differ. The common30 check controls anchor identity but not valid output sets.",
                  "- Exact permutation p-values are unadjusted, conditional on this convenience panel, and do not",
                  "  propagate uncertainty of leaderboard scores or quantify held-out generalization.",
                  "- Two-point leave-one-out correlations are necessarily +/-1 and are not evidence of robustness.",
                  "- Pearson measures paired linear agreement, not equality of full score distributions.",
                  "- ECHO: no matching current-panel rows established in the saved snapshot.",
                  "- Metaculus: search-index leads for Opus 5, Kimi K3, GPT-5.6 Sol exist, but no verified complete",
                  "  leaderboard snapshot is available. No search-snippet scores enter this calculation.", "",
                  "Reproduce: `.venv/bin/python -m paper.analyze_external_validity`."])
    return "\n".join(lines) + "\n"


def main():
    sources = json.loads((OUTPUT / "sources.json").read_text())
    for filename, metadata in sources.items():
        if hashlib.sha256((OUTPUT / filename).read_bytes()).hexdigest() != metadata["sha256"]:
            raise ValueError(f"Source hash mismatch: {filename}")
    panel = {row["tag"]: row for row in collect_panel()}
    references = ceiling()
    ranges = {setting: reference_ranges(references, setting) for setting, _ in SETTINGS}
    scores = model_scores(panel, ranges)
    fb_rows = parse_forecastbench((OUTPUT / "forecastbench_tournament.js").read_text())
    futurex = json.loads((OUTPUT / "futurex.json").read_text())["data"]["content"]
    prophet = json.loads((OUTPUT / "prophet_arena.json").read_text())["data"]["content"]
    definitions = {
        "ForecastBench tournament": (forecastbench_groups(fb_rows),
            "Overall Brier Index, four models. Opus max and GPT standard-medium variants are retained; "
            "search/code assistance differs. No baseline/tournament score mixing."),
        "FutureX week 3": (futurex_groups(futurex),
            "Weekly score on 63 events. All seven Kimi frameworks retained; Opus X-Nevar and "
            "GPT FrontierAgent each supply one row. Frameworks are NOT matched across models. "
            "Generic GPT5.6, verifier-suffixed labels and multi-model ensembles are excluded."),
        "Prophet Arena fixed context": (prophet_groups(prophet),
            "Binary-event Fixed Context board; published market-anchored skill theta, not raw Brier, "
            "market edge or betting return. Model-specific question coverage differs; theta is rounded."),
        "ForecastBench + tentative Inkling": (forecastbench_groups(fb_rows, with_inkling=True),
            "Sensitivity only: adds 'Inkling with search'. Exact checkpoint correspondence to local "
            "Inkling-NVFP4 is unresolved; this five-model panel is NOT the primary result."),
    }
    datasets = {name: dict(scope=scope, groups=groups, comparisons=compare(groups, scores))
                for name, (groups, scope) in definitions.items()}
    common30 = collect_common30({tag for groups, _ in definitions.values() for tag in groups}, references)
    common30["model_scores"] = model_scores(common30["records"], ranges)
    common30["comparisons"] = {name: compare(groups, common30["model_scores"])
                               for name, (groups, _) in definitions.items()}
    result = dict(created_at_utc=datetime.now(timezone.utc).isoformat(), sources=sources,
                  weights=SCHEMES, ranges=ranges, local_records=panel, model_scores=scores,
                  datasets=datasets, common30=common30)
    (OUTPUT / "cross_benchmark.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    report = render(result)
    (OUTPUT / "cross_benchmark.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
