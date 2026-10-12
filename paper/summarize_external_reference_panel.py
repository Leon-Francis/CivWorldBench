"""External references with unmatched-framework FutureX comparisons withdrawn."""
from statistics import correlation

from paper.aggregate_external_forecasting import (
    ROOT, EXTERNAL, aggregate, compare_configs, number_range, read, write, digest,
)
from paper.analyze_futureeval_pdf import association_with_coverage
from paper.gen_tables import LINES


OUTPUT = EXTERNAL / "reference_panel_sameframework_20260923"
BASE = EXTERNAL / "fullscope_aggregation_20260923/analysis.json"
FUTUREEVAL = EXTERNAL / "futureeval_pdf_20260923/analysis.json"


def aa_scores():
    return {tag: score for _, tag, score in LINES if score is not None}


def simple_association(local, external):
    tags = sorted(set(local) & set(external))
    if len(tags) < 3:
        raise ValueError("Too few common models")
    local_values = [local[tag] for tag in tags]
    external_values = [external[tag] for tag in tags]
    return dict(n=len(tags), tags=tags, pearson=correlation(local_values, external_values),
                spearman=correlation(local_values, external_values, method="ranked"))


def futurex_stages(base, paired):
    groups = base["groups"]["futurex4"]
    original_groups = {tag: rows for tag, rows in groups.items() if tag != "gpt-6-astra"}
    corrected_original_scope = dict(base["local"]["new_mixed"])
    for tag in ("opus5", "gpt56sol"):
        corrected_original_scope[tag] = paired["records"]["new60"][tag]["index"]["overall"]
    definitions = [
        ("historical_three_old", original_groups, base["local"]["old_mixed"]),
        ("add_astra_old_policy", groups, base["local"]["old_mixed"]),
        ("correct_policy_same_scope", groups, corrected_original_scope),
        ("expand_api190_new", groups, base["local"]["new_mixed"]),
        ("original_three_new", original_groups, base["local"]["new_mixed"]),
    ]
    return {name: dict(n=len(group), local={tag: scores[tag] for tag in group},
                       comparisons=compare_configs(group, scores)) for name, group, scores in definitions}


def reference_boards(base, futureeval, track):
    if track not in ("forecastbench4", "forecastbench_preliminary5"):
        raise ValueError("Use one ForecastBench track at a time")
    return {track: base["model_level_board_medians"][track],
            "futureeval5": {tag: row["score"] for tag, row in futureeval["matched"].items()}}


def futurex_framework_audit(groups):
    frameworks = {}
    for tag, rows in groups.items():
        for row in rows:
            name = row["source_row"]["FRAMEWORK"]
            frameworks.setdefault(name, {})[tag] = row["score"]
    return dict(frameworks=frameworks,
                maximum_matched_models=max((len(rows) for rows in frameworks.values()), default=0),
                names_are_not_configuration_verification=True,
                status="No verified shared-framework panel; historical correlations are withdrawn as model-validity evidence")


def render(result):
    lines = ["# Primary External Reference Review", "",
             "Prophet is excluded from this primary review and its aggregates at the user's request.",
             "Its previously observed results remain in the source audits as sensitivity evidence.",
             "This exclusion was made after inspecting outcomes, not a preregistered eligibility criterion.",
             "FutureX is excluded from references and aggregates: neither old nor new comparisons held framework fixed.",
             "Other boards remain exploratory candidates; organizer identity or AskNews naming alone does not verify all configurations.",
             "AA is general capability, not a forecasting board, and is never included in the forecasting aggregate.", "",
             "## Current Maximum-scope SWFI Correlations", "",
             "| Reference | Models | Pearson | Spearman |", "|---|---:|---:|---:|"]
    for name, item in result["references"].items():
        rows = item["new"]
        lines.append(f"| {name} | {item['n']} | " + " | ".join(
            number_range([row[metric] for row in rows]) for metric in ("pearson", "spearman")) + " |")
    lines += ["", "Ranges are across eligible external configurations, not confidence intervals.",
              "AA covers only the 11 recorded main-table models; no new API model AA value is inferred.", "",
              "## Withdrawn FutureX Comparisons (Audit Only)", "",
              "The historical 0.565--0.972 arithmetic is reproducible, but mixed frameworks invalidate its use as model-validity evidence.",
              f"Maximum exact-name framework overlap in the local panel: {result['futurex_framework_audit']['maximum_matched_models']} model(s).",
              "Similar FrontierAgent names were not silently treated as the same framework/configuration.",
              "Sequential changes below are descriptive, not an order-independent causal attribution.", "",
              "| Stage | Models | Pearson |", "|---|---:|---:|"]
    for name, item in result["futurex_stages"].items():
        lines.append(f"| {name} | {item['n']} | " + number_range([row["pearson"] for row in item["comparisons"]]) + " |")
    lines += ["", "## FutureX External Configurations", "",
              "| Model | Framework | Score |", "|---|---|---:|"]
    for tag, rows in result["futurex_groups"].items():
        for row in rows:
            lines.append(f"| {tag} | {row['source_row']['FRAMEWORK']} | {row['score']:.2f} |")
    lines += ["", "Kimi contributes seven framework alternatives and Astra two: 14 four-model combinations.",
              "These are not 14 independent models or repeated runs. The interval combines different agent systems.", "",
              "Taking each model's configuration median instead gives new SWFI Pearson "
              f"{result['futurex_median']['new']['pearson']:+.3f}; this is a fixed sensitivity, not a best-row choice.", "",
              "## Exploratory Aggregates Without Prophet or FutureX", "",
              "| ForecastBench track | Method | Minimum boards | Models | Old Pearson | New Pearson |",
              "|---|---|---:|---:|---:|---:|"]
    for track, methods in result["aggregates"].items():
        for method, thresholds in methods.items():
            for threshold, item in thresholds.items():
                old, new = item["old"], item["new"]
                lines.append(f"| {track} | {method} | {threshold} | {new['n']} | "
                             f"{old['pearson']:+.3f} | {new['pearson']:+.3f} |")
    lines += ["", "Available-board averages have heterogeneous coverage. Requiring two boards leaves only",
              "the original Opus/GPT/Kimi trio. Neither aggregate is a new independent validation set.",
              "Higher post-exclusion correlation cannot establish a reliable real-world proxy by itself.",
              "No paper table or frozen metric/source result was modified."]
    return "\n".join(lines) + "\n"


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    base, futureeval = read(BASE), read(FUTUREEVAL)
    if digest(BASE) != futureeval["previous_sha256"]:
        raise ValueError("Upstream audit changed")
    paired = read(ROOT / base["prior_source"])
    if digest(ROOT / base["prior_source"]) != base["prior_sha256"]:
        raise ValueError("Upstream score comparison changed")
    for name, expected in paired["provenance"].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"Input changed: {name}")
    aa = aa_scores()
    local = base["local"]
    references = {name: dict(n=len(base["groups"][name]), old=base["per_board"]["old_mixed"][name],
                             new=base["per_board"]["new_mixed"][name])
                  for name in ("forecastbench4", "forecastbench_preliminary5")}
    references["futureeval5"] = dict(n=len(futureeval["matched"]), old=[futureeval["associations"]["old_mixed"]],
                                     new=[futureeval["associations"]["new_mixed"]])
    references["AA_index_reference"] = dict(n=len(aa), old=[simple_association(local["old_mixed"], aa)],
                                             new=[simple_association(local["new_mixed"], aa)])
    aggregates = {}
    for track in ("forecastbench4", "forecastbench_preliminary5"):
        boards = reference_boards(base, futureeval, track)
        aggregates[track] = {}
        for method in ("zscore", "midrank"):
            index = aggregate(boards, method)
            aggregates[track][method] = {}
            for threshold in (1, 2):
                targets = {tag: row["score"] for tag, row in index.items() if row["coverage"] >= threshold}
                aggregates[track][method][str(threshold)] = dict(
                    records={tag: index[tag] for tag in targets},
                    old=association_with_coverage(local["old_mixed"], targets),
                    new=association_with_coverage(local["new_mixed"], targets))
    result = dict(references=references, futurex_stages=futurex_stages(base, paired),
                  futurex_groups=base["groups"]["futurex4"], aggregates=aggregates,
                  futurex_framework_audit=futurex_framework_audit(base["groups"]["futurex4"]),
                  futurex_median={label: simple_association(local[scope], base["model_level_board_medians"]["futurex4"])
                                  for label, scope in (("old", "old_mixed"), ("new", "new_mixed"))},
                  aa=dict(values=aa, source="paper/gen_tables.py:LINES, recorded AA Intelligence Index v4.3",
                          source_sha256=digest(ROOT / "paper/gen_tables.py"),
                          not_a_live_refresh=True, interpretation="General capability reference, not real forecasting validity"),
                  provenance={str(path.relative_to(ROOT)): digest(path) for path in
                              (BASE, FUTUREEVAL, ROOT / "paper/summarize_external_reference_panel.py")},
                  excluded=dict(prophet="Post-hoc exclusion from primary summary; original results retained",
                                futurex="Framework mismatch; old and new correlations withdrawn from evidence"))
    write(OUTPUT / "analysis.json", result)
    (OUTPUT / "analysis.md").write_text(render(result))
    print(render(result))


if __name__ == "__main__":
    main()
