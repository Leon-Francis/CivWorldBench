"""Maximum-local-coverage and external-only aggregation diagnostics, not paper updates."""
import itertools
from statistics import mean, median, pstdev

from paper.analyze_external_validity import FB_NAMES, number_range, sensitivity, variants
from paper.compare_external_validity_versions import (
    EXTERNAL, ROOT, external_groups, parse_forecastbench, read, write, digest,
)


OUTPUT = EXTERNAL / "fullscope_aggregation_20260923"
PRIOR = EXTERNAL / "version_comparison_20260923/comparison.json"
PRELIMINARY_NAMES = FB_NAMES | {
    "gemini-3.7-flash": "gemini-3.7-flash-high-web-search-code-execution",
}


def preliminary_groups(rows):
    output = {tag: [] for tag in PRELIMINARY_NAMES}
    for row in rows:
        for tag, name in PRELIMINARY_NAMES.items():
            if row["Model"].rstrip("\u2020") == name:
                output[tag].append(dict(score=row["Dataset"], source_row=row))
    variants(output)
    return output


def standardize(scores, method):
    values = list(scores.values())
    if len(values) < 2 or pstdev(values) == 0:
        raise ValueError("Board needs distinct scores from at least two models")
    if method == "zscore":
        center, scale = mean(values), pstdev(values)
        return {tag: (value - center) / scale for tag, value in scores.items()}
    if method == "midrank":
        return {tag: (sum(other < value for other in values) +
                      (sum(other == value for other in values) - 1) / 2) / (len(values) - 1)
                for tag, value in scores.items()}
    raise ValueError(method)


def aggregate(boards, method):
    standardized = {name: standardize(scores, method) for name, scores in boards.items()}
    tags = sorted(set().union(*(set(scores) for scores in boards.values())))
    records = {}
    for tag in tags:
        available = {name: scores[tag] for name, scores in standardized.items() if tag in scores}
        records[tag] = dict(score=mean(available.values()), coverage=len(available), components=available)
    return records


def diagnostic(local, target):
    tags = sorted(set(local) & set(target))
    if len(tags) < 3:
        return dict(n=len(tags), tags=tags, reason="Too few models; no interpretable correlation test")
    return dict(tags=tags, **sensitivity([local[tag] for tag in tags], [target[tag] for tag in tags], tags))


def compare_configs(groups, local):
    return [diagnostic(local, {tag: row["score"] for tag, row in selected.items()})
            for selected in variants(groups)]


def render(result):
    lines = ["# Maximum-coverage and Cross-board Diagnostics", "",
             "Uses existing maximum approved local scoring scope, not the common60 subset.",
             "Local SWFI scales/weights remain frozen. External snapshots remain dated 2026-09-22.",
             "Preliminary Dataset scores are analyzed separately from Tournament Overall; never spliced.", "",
             "## Per-board Pearson", "", "| Board | Models | Old | New |", "|---|---:|---:|---:|"]
    for name, groups in result["groups"].items():
        values = [number_range([row["pearson"] for row in result["per_board"][scope][name]])
                  for scope in ("old_mixed", "new_mixed")]
        lines.append(f"| {name} | {len(groups)} | " + " | ".join(values) + " |")
    lines += ["", "## Prophet Fixed Context", "",
              "The public +/- values are displayed uncertainty; current methodology calls theta uncertainty conditional SE, not a full CI.",
              "Predicted 1-Brier is fitted on a shared reference, NOT raw own-event accuracy.", "",
              "| Model | Theta | Predicted 1-Brier | Resolved / predicted | New SWFI |",
              "|---|---:|---:|---:|---:|"]
    for tag, rows in result["groups"]["prophet7"].items():
        row = rows[0]["source_row"]
        theta = row[next(key for key in row if key.startswith("Skill "))]
        predicted = row[next(key for key in row if key.startswith("Predicted "))].split(" measured")[0]
        events = row["Events resolved / predicted"]
        lines.append(f"| {tag} | {theta} | {predicted} | {events} | {result['local']['new_mixed'][tag]:.3f} |")
    lines += ["", "## External-only Aggregation", "",
              "Within each board, take the median of all eligible configurations per model; then standardize",
              "across its exact local-overlap models. Average observed boards equally. Missing is not zero.",
              "Correlate only models covered by at least two boards. This remains a mixed-coverage exploratory index:",
              "the score need not be comparable across different board subsets; standardization can magnify noise.",
              "No CivLadder score enters construction, normalization, or model eligibility.",
              "Tournament and Preliminary are alternative versions of ONE board, never independent votes.", "",
              "| ForecastBench track | Aggregate | Models with >=2 boards | Old r | New r | New rho | New p |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for track, methods in result["aggregates"].items():
        for method, item in methods.items():
            old, new = item["associations"]["old_mixed"], item["associations"]["new_mixed"]
            lines.append(f"| {track} | {method} | {new['n']} | {old['pearson']:+.3f} | "
                         f"{new['pearson']:+.3f} | {new['spearman']:+.3f} | {new['permutation_p']:.3f} |")
    lines += ["", "## Limitations", "",
              "Coverage is sparse and not missing at random; only GPT-5.6 Sol and Kimi K3 span all three boards.",
              "Board-specific domains, tools, thinking budgets and uncertainty differ. Median does not remove harness confounding.",
              "Counts refer to distinct model names, not provider independence. Tests do not propagate event uncertainty.",
              "An average cannot establish a common forecasting factor if its constituent boards disagree.",
              "Old high correlations can be reported as historical exploratory results, not selected in place of corrected scores."]
    return "\n".join(lines) + "\n"


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    prior = read(PRIOR)
    for name, expected in prior["provenance"].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"Prior audit input changed: {name}; update local scores before comparing")
    groups = external_groups(read(EXTERNAL / "prophet_arena.json")["data"]["content"],
                             read(EXTERNAL / "futurex.json")["data"]["content"],
                             parse_forecastbench((EXTERNAL / "forecastbench_tournament.js").read_text()))
    del groups["prophet_new4"]
    groups["forecastbench_preliminary5"] = preliminary_groups(
        parse_forecastbench((EXTERNAL / "forecastbench_preliminary.js").read_text()))
    local = {scope: {tag: record["index"]["overall"] for tag, record in prior["records"][scope].items()}
             for scope in ("old_mixed", "new_mixed")}
    point_boards = {name: {tag: median(row["score"] for row in rows) for tag, rows in group.items()}
                    for name, group in groups.items()}
    per_board = {scope: {name: compare_configs(group, scores) for name, group in groups.items()}
                 for scope, scores in local.items()}
    tentative = {}
    for track, group_name, column in (("tournament", "forecastbench4", "Overall"),
                                       ("preliminary", "forecastbench_preliminary5", "Dataset")):
        candidates = [dict(score=row[column], source_row=row)
                      for row in parse_forecastbench((EXTERNAL / f"forecastbench_{track}.js").read_text())
                      if row["Model"] == "Inkling with search"]
        expanded = groups[group_name] | {"inkling": candidates}
        tentative[track] = dict(note="Unverified external checkpoint versus local NVFP4; sensitivity only",
                                associations={scope: compare_configs(expanded, scores) for scope, scores in local.items()})
    aggregates = {}
    for track in ("forecastbench4", "forecastbench_preliminary5"):
        boards = {name: point_boards[name] for name in (track, "futurex4", "prophet7")}
        aggregates[track] = {}
        for method in ("zscore", "midrank"):
            index = aggregate(boards, method)
            eligible = {tag: value["score"] for tag, value in index.items() if value["coverage"] >= 2}
            associations = {scope: diagnostic(scores, eligible) for scope, scores in local.items()}
            leave_board_out = {}
            for omitted in boards:
                reduced = aggregate({name: scores for name, scores in boards.items() if name != omitted}, method)
                target = {tag: reduced[tag]["score"] for tag in eligible if tag in reduced}
                leave_board_out[omitted] = dict(
                    note="Fixed eligible model panel; remaining coverage may fall to one board",
                    associations={scope: diagnostic(scores, target) for scope, scores in local.items()})
            aggregates[track][method] = dict(records=index, associations=associations, leave_board_out=leave_board_out)
    pairwise = {f"{first} vs {second}": diagnostic(point_boards[first], point_boards[second])
                for first, second in itertools.combinations(point_boards, 2)
                if {first, second} != {"forecastbench4", "forecastbench_preliminary5"}}
    coverage = {tag: {name: tag in group for name, group in groups.items()} for tag in local["new_mixed"]}
    result = dict(prior_source=str(PRIOR.relative_to(ROOT)), prior_sha256=digest(PRIOR),
                  code_sha256=digest(ROOT / "paper/aggregate_external_forecasting.py"), groups=groups,
                  local=local, coverage=coverage, per_board=per_board, aggregates=aggregates,
                  board_pairwise=pairwise, model_level_board_medians=point_boards,
                  tentative_inkling=tentative)
    write(OUTPUT / "analysis.json", result)
    (OUTPUT / "analysis.md").write_text(render(result))
    print(render(result))


if __name__ == "__main__":
    main()
