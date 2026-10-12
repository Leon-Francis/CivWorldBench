"""Export manuscript evidence from current scores and archived external snapshots."""
from statistics import median

from paper import gen_tables as tables
from paper.aggregate_external_forecasting import EXTERNAL, compare_configs, read, write
from paper.audit_composite_forecasting import reference_ranges
from paper.build_sandbox_index import calculate, digest, extract
from paper.build_main_table import render_compact, render_full
from paper.summarize_external_reference_panel import aa_scores, simple_association


PAIRED = EXTERNAL / "version_comparison_20260923/comparison.json"
BASE = EXTERNAL / "fullscope_aggregation_20260923/analysis.json"
FUTURE = EXTERNAL / "futureeval_pdf_20260923/analysis.json"
HISTORY = EXTERNAL / "prophet_history_20260923/analysis.json"
SPEC = tables.ROOT / "bench/measure/out/section5/swfi_v1/spec.json"
OUTPUT = EXTERNAL / "manuscript_20260923"
LABELS = {
    "forecastbench4": "ForecastBench Tournament (Overall)",
    "forecastbench_preliminary5": "ForecastBench Preliminary (Dataset)",
    "futureeval": "Metaculus FutureEval (Avg Score)",
    "prophet_skill": "Prophet Fixed Context (SKILL)",
    "aa": "AA Intelligence Index v4.3 (reference)",
}


def interval(values, digits=3):
    low, high = min(values), max(values)
    left, right = f"{low:.{digits}f}", f"{high:.{digits}f}"
    return left if left == right else left + "--" + right


def check_alignment(paired, spec):
    if paired["spec"] != spec:
        raise ValueError("Frozen SWFI specification changed")
    for name, expected in paired["provenance"].items():
        if digest(tables.ROOT / name) != expected:
            raise ValueError(f"Analysis input changed: {name}")
    for setting in spec["setting_weights"]:
        current = reference_ranges(tables.ceiling(), setting)
        if {key: list(value) for key, value in current.items()} != spec["ranges"][setting]:
            raise ValueError(f"Main-table reference scales changed: {setting}")
    sources = {}
    for _, tag, aa in tables.LINES:
        if aa is None:
            continue
        path = tables.score_path(tag)
        row = paired["records"]["new_mixed"][tag]
        if path != tables.ROOT / row["source"]:
            raise ValueError(f"Main-table score source changed: {tag}")
        values = extract(tables.load(tag), spec)
        if values != row["values"] or calculate(values, spec) != row["index"]:
            raise ValueError(f"Main-table metrics differ from SWFI: {tag}")
        sources[tag] = dict(path=str(path.relative_to(tables.ROOT)), sha256=digest(path))
    scores = {tag: tables.load(tag) for _, tag, _ in tables.LINES}
    rendered = tables.table_main(scores, tables.ceiling(), tables.SETTINGS, "tab:main", tables.CAP_MAIN)
    if rendered != (tables.OUT_T / "main_all.tex").read_text():
        raise ValueError("Rendered main table is stale")
    for name, expected in (("main_compact.tex", render_compact(scores, tables.ceiling(), spec)),
                           ("main_full.tex", render_full(scores, tables.ceiling()))):
        if (tables.OUT_T / name).read_text() != expected:
            raise ValueError(f"Published table is stale: {name}")
    return dict(main_models=len(sources), checked_components=len(sources) * 21,
                maximum_absolute_difference=0.0, main_table_byte_identical=True, sources=sources)


def build():
    paired, spec = read(PAIRED), read(SPEC)
    alignment = check_alignment(paired, spec)
    base, future, history = read(BASE), read(FUTURE), read(HISTORY)
    for artifact in (base, future, history):
        for name, expected in artifact.get("provenance", {}).items():
            if digest(tables.ROOT / name) != expected:
                raise ValueError(f"External analysis source changed: {name}")
    if base["prior_sha256"] != digest(PAIRED) or future["previous_sha256"] != digest(BASE):
        raise ValueError("External analysis does not refer to current score snapshot")
    groups = {key: base["groups"][key] for key in ("forecastbench4", "forecastbench_preliminary5")}
    groups["futureeval"] = {tag: [dict(score=row["score"])] for tag, row in future["matched"].items()}
    for metric in ("skill",):
        groups["prophet_" + metric] = {
            tag: [dict(score=row[metric])] for tag, row in history["matched"].items()}
    local = {scope: {tag: row["index"]["overall"] for tag, row in paired["records"][scope].items()}
             for scope in ("new_mixed", "new60")}
    associations = {scope: {key: compare_configs(group, scores) for key, group in groups.items()}
                    for scope, scores in local.items()}
    associations["new_mixed"]["aa"] = [simple_association(local["new_mixed"], aa_scores())]
    matched_aa = {}
    for key, group in groups.items():
        shared = {tag: rows for tag, rows in group.items() if tag in aa_scores()}
        if len(shared) >= 3:
            matched_aa[key] = dict(n=len(shared), swfi=compare_configs(shared, local["new_mixed"]),
                                   aa=compare_configs(shared, aa_scores()))
    figure_targets = {key: {tag: median(row["score"] for row in rows) for tag, rows in group.items()}
                      for key, group in groups.items()}
    figure_targets["aa"] = aa_scores()
    figure_associations = {key: simple_association(local["new_mixed"], target)
                           for key, target in figure_targets.items()}
    return dict(alignment=alignment, associations=associations, matched_aa=matched_aa,
                figure_targets=figure_targets, figure_associations=figure_associations,
                groups=groups, local=local, names={tag: name for name, tag, _ in tables.LINES},
                aa=aa_scores(), previous_binary_skill=history["previous_binary_skill_common_models"],
                provenance={str(path.relative_to(tables.ROOT)): digest(path)
                            for path in (PAIRED, BASE, FUTURE, HISTORY, SPEC)})


def render_main(result):
    lines = [r"\begin{table}[t]\centering\small",
             r"\caption{Associations between fixed-weight SWFI and external model scores.",
             r"$n$ counts matched models. Ranges span eligible external configurations.",
             r"The two ForecastBench tracks overlap. AA provides a general-capability reference.",
             r"\Cref{app:externalvalidity} describes the protocols and sensitivity analyses.}",
             r"\label{tab:externalvalidity}",
             r"\begin{tabular}{@{}lrrr@{}}\toprule",
             r"External reference & $n$ & Pearson $r$ & Spearman $\rho$ \\ \midrule"]
    for key in ("forecastbench4", "forecastbench_preliminary5", "futureeval", "prophet_skill", "aa"):
        rows = result["associations"]["new_mixed"][key]
        if key == "aa":
            lines.append(r"\midrule")
        lines.append(f"{LABELS[key]} & {rows[0]['n']} & " +
                     interval([row["pearson"] for row in rows]) + " & " +
                     interval([row["spearman"] for row in rows]) + r" \\")
    return "\n".join(lines + [r"\bottomrule\end{tabular}", r"\end{table}", ""])


def render_matches(result):
    names = dict(result["names"], **{"gemini-3.7-flash": "Gemini 3.7 Flash", "gpt-6-astra": "GPT-6 Astra",
                                     "glm-5.3-flash": "GLM-5.3 Flash", "hy3": "Hunyuan Hy3"})
    lines = [r"\begin{table}[t]\centering\small",
             r"\caption{SWFI and external scores for exact model matches. T/P denotes ForecastBench",
             r"Tournament/Preliminary and FE denotes FutureEval. SKILL uses Prophet's Fixed Context track.",
             r"Dashes indicate unavailable matches. External scores retain their native units.}",
             r"\label{tab:externalmatches}", r"\setlength{\tabcolsep}{4pt}",
             r"\begin{tabular}{@{}lrrrrrr@{}}\toprule",
             r"Model & SWFI & FB T & FB P & FE & SKILL & AA \\ \midrule"]
    for tag, score in sorted(result["local"]["new_mixed"].items(), key=lambda item: -item[1]):
        cells = [names[tag], f"{score:.2f}"]
        for key in ("forecastbench4", "forecastbench_preliminary5", "futureeval", "prophet_skill"):
            matches = result["groups"][key].get(tag)
            cells.append(interval([row["score"] for row in matches], 2) if matches else "--")
        cells.append(str(result["aa"].get(tag, "--")))
        lines.append(" & ".join(cells) + r" \\")
    return "\n".join(lines + [r"\bottomrule\end{tabular}", r"\end{table}", ""])


def make_figure(result):
    plt = tables._mpl(academic=True)
    figure, axis = plt.subplots(figsize=(3.4, 1.35))
    axis.set_position([.40, .18, .58, .66])
    labels = ["FB Tournament", "FB Preliminary", "FutureEval", "Prophet SKILL"]
    keys = ["forecastbench4", "forecastbench_preliminary5", "futureeval", "prophet_skill"]
    values = [result["figure_associations"][key]["pearson"] for key in keys]
    bars = axis.barh([3, 2, 1, 0], values, height=.52, color="#367E83", linewidth=0, zorder=3)
    axis.bar_label(bars, labels=[f"{value:.3f}" for value in values], padding=3, fontsize=9)
    reference = result["figure_associations"]["aa"]["pearson"]
    axis.axvline(reference, color="#B37553", linewidth=1.2, linestyle=(0, (3, 2)),
                 zorder=4, label=f"AA reference: {reference:.3f}")
    axis.set_xlim(0, 1.23)
    axis.set_ylim(-.5, 3.5)
    axis.set_yticks([3, 2, 1, 0], labels)
    axis.set_xticks([0, .5, 1.0], ["0", "0.5", "1.0"])
    axis.tick_params(axis="y", length=0, pad=4, labelsize=10)
    axis.tick_params(axis="x", length=2, pad=2, labelsize=9)
    axis.grid(axis="x", color="#D7D7D7", linewidth=.45)
    axis.set_axisbelow(True)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.spines["bottom"].set_color("#999999")
    axis.legend(loc="lower right", bbox_to_anchor=(1, 1.03), fontsize=9,
                frameon=False, borderaxespad=0, handletextpad=.5)
    tables._save(figure, tables.OUT_F / "external_validity.pdf")
    plt.close(figure)


def render_sensitivity(result):
    lines = [r"\begin{table}[t]\centering\small",
             r"\caption{External-validity sensitivity. Drop-one ranges omit each model in turn and include all",
             r"eligible configurations. Exact two-sided permutation tests are exploratory and unadjusted.}",
             r"\label{tab:externalsensitivity}", r"\setlength{\tabcolsep}{4pt}",
             r"\begin{tabular}{@{}lrr@{}}\toprule",
             r"External reference & Drop-one $r$ & Permutation $p$ \\ \midrule"]
    for key in result["groups"]:
        rows = result["associations"]["new_mixed"][key]
        drops = [value["pearson"] for row in rows for value in row["leave_one_out"].values()]
        lines.append(LABELS[key] + " & " + interval(drops) + " & " +
                     interval([row["permutation_p"] for row in rows], 4) + r" \\")
    lines += [r"\bottomrule\end{tabular}", r"\end{table}", "",
              r"\begin{table}[t]\centering\small",
              r"\caption{Same-model comparison with AA: each predictor is correlated with the external outcome.",
              r"This differs from the SWFI--AA association in \cref{fig:externalvalidity}.",
              r"Only panels with at least three recorded AA values are included.}",
              r"\label{tab:externalAA}", r"\begin{tabular}{@{}lrrr@{}}\toprule",
              r"External reference & $n$ & SWFI $r$ & AA $r$ \\ \midrule"]
    for key, row in result["matched_aa"].items():
        lines.append(LABELS[key] + f" & {row['n']} & " +
                     interval([value["pearson"] for value in row["swfi"]]) + " & " +
                     interval([value["pearson"] for value in row["aa"]]) + r" \\")
    return "\n".join(lines + [r"\bottomrule\end{tabular}", r"\end{table}", ""])


def main():
    result = build()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write(OUTPUT / "analysis.json", result)
    for name, renderer in (("external_validity", render_main), ("external_matches", render_matches),
                           ("external_sensitivity", render_sensitivity)):
        (tables.OUT_T / f"{name}.tex").write_text(renderer(result))
    make_figure(result)
    print(result["alignment"])


if __name__ == "__main__":
    main()
