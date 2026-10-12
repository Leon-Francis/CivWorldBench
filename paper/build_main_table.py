"""Render compact main results and full-setting appendix without changing scoring."""
import json

from paper import gen_tables as tables
from paper.audit_composite_forecasting import reference_ranges
from paper.build_sandbox_index import calculate, digest, extract


SPEC = tables.ROOT / "bench/measure/out/section5/swfi_v1/spec.json"
OUTPUT = tables.ROOT / "bench/measure/out/section5/main_table_compact.json"
OMNI = "bench-basic|v1"
METRICS = ["quantity", "event", "trait", "rules", "capture", "direction", "specificity"]
CAPTION = (
    r"Model performance on \bench: three-setting SWFI and \omniID\ diagnostics. "
    r"Bold denotes the best model score, excluding the expert reference. "
    r"Percentages indicate reference-normalized completion. "
    r"Full three-setting results are reported in \cref{tab:main_full}."
)


def collect(scores, references, spec):
    for setting in spec["setting_weights"]:
        current = {key: list(value) for key, value in reference_ranges(references, setting).items()}
        if current != spec["ranges"][setting]:
            raise ValueError(f"Current reference differs from frozen SWFI: {setting}")
    records = []
    for name, tag, aa in tables.LINES:
        score = scores.get(tag)
        if score is None:
            records.append(dict(name=name, tag=tag, aa=aa, values=dict.fromkeys(METRICS), overall=None))
            continue
        values = extract(score, spec)
        complete = all(value is not None for components in values.values() for value in components.values())
        records.append(dict(name=name, tag=tag, aa=aa, values=values[OMNI],
                            overall=calculate(values, spec)["overall"] if complete else None))
    return records


def render_compact(scores, references, spec):
    records = collect(scores, references, spec)
    best = {}
    for metric in ["overall", *METRICS]:
        candidates = [record["overall"] if metric == "overall" else record["values"][metric]
                      for record in records if record["tag"] != tables.EXPERT_TAG]
        best[metric] = max((value for value in candidates if value is not None), default=None)
    lines = [r"\begin{table}[!b]\centering\scriptsize",
             r"\setlength{\tabcolsep}{1.5pt}\renewcommand{\arraystretch}{1.12}",
             r"\caption{" + CAPTION + "}", r"\label{tab:main}",
             r"\begin{tabular}{@{}>{\raggedright\arraybackslash}p{0.16\linewidth}"
             r">{\centering\arraybackslash}p{0.045\linewidth}"
             r">{\raggedleft\arraybackslash}p{0.075\linewidth}"
             r">{\raggedleft\arraybackslash}p{\dimexpr(0.72\linewidth-32.1pt)/7\relax}"
             r">{\raggedleft\arraybackslash}p{\dimexpr(0.72\linewidth-32.1pt)/7+3.6pt\relax}"
             r"*{4}{>{\raggedleft\arraybackslash}p{\dimexpr(0.72\linewidth-32.1pt)/7\relax}}"
             r">{\raggedleft\arraybackslash}p{\dimexpr(0.72\linewidth-32.1pt)/7+1.5pt\relax}@{}}",
             r"\toprule",
             r" & & & \multicolumn{2}{c}{\Lone\ association} & \multicolumn{2}{c}{\Ltwo\ belief} & "
             r"\multicolumn{3}{c}{\Lthree\ intervention} \\",
             r"\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-10}",
             r"Model & AA & SWFI & \mbox{Quantity\,$\uparrow$} & \mbox{Event\,$\uparrow$} & "
             r"\mbox{Trait\,$\uparrow$} & \mbox{Rules\,$\uparrow$} & \mbox{Capture\,$\uparrow$} & "
             r"\mbox{Direction\,$\uparrow$} & \mbox{Specificity\,$\uparrow$} \\",
             r"\midrule",
             r"\floorline & \textemdash & 0.00 & +0.00 & +0.000 & 0.00 & 0.00 & $-$1.00 & 0.00 & +0.00 \\",
             r"\tabledash"]
    for record in records:
        if record["tag"] == tables.EXPERT_TAG:
            lines.append(r"\tabledash")
        index = tables._cellfmt(record["overall"], 2, False)
        if record["overall"] is not None and record["overall"] == best["overall"] and record["tag"] != tables.EXPERT_TAG:
            index = r"\textbf{" + index + "}"
        name = "Opus~5" if record["tag"] == "opus5" else record["name"]
        if record["tag"] in ("gpt56sol", "opus5"):
            name += "-Medium"
        cells = [name, str(record["aa"]) if record["aa"] is not None else r"\textemdash", index]
        for metric in METRICS:
            value = record["values"][metric]
            cell = tables._cellfmt(value, 3 if metric == "event" else 2,
                                  metric not in ("trait", "rules", "direction"))
            if value is not None:
                if value == best[metric] and record["tag"] != tables.EXPERT_TAG:
                    cell = r"\textbf{" + cell + "}"
                floor, upper = spec["ranges"][OMNI][metric]
                completion = 100 * (value - floor) / (upper - floor)
                shade = "tableRose" if completion < 0 else "tableTeal"
                cell += f"\\headroompct{{{shade}}}{{{completion:.0f}}}"
            cells.append(r"\mbox{" + cell + "}")
        lines.append(" & ".join(cells) + r" \\")
    ceiling_cells = [r"\ceilingline", r"\textemdash", "100.00"]
    for metric in METRICS:
        ceiling_cells.append(tables._cellfmt(spec["ranges"][OMNI][metric][1],
                                            3 if metric == "event" else 2,
                                            metric not in ("trait", "rules", "direction", "specificity")))
    lines.extend([r"\tabledash", " & ".join(ceiling_cells) + r" \\",
                  r"\bottomrule\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def render_full(scores, references):
    caption = (
        r"Complete results across three settings. Bold denotes the best model score, "
        r"excluding the expert reference. Percentages report reference-normalized completion. "
        r"Missing or unparseable answers receive the failure penalties in \cref{app:metrics}. "
        r"\Cref{app:referencelines} specifies the numerical reference assumptions."
    )
    return tables.table_main(scores, references, tables.SETTINGS, "tab:main_full", caption)


def main():
    spec = json.loads(SPEC.read_text())
    scores = {tag: tables.load(tag) for _, tag, _ in tables.LINES}
    references = tables.ceiling()
    for name, content in (("main_compact.tex", render_compact(scores, references, spec)),
                          ("main_full.tex", render_full(scores, references))):
        path = tables.OUT_T / name
        if not path.exists() or path.read_text() != content:
            path.write_text(content)
    records = collect(scores, references, spec)
    provenance = {str(tables.score_path(record["tag"]).relative_to(tables.ROOT)):
                  digest(tables.score_path(record["tag"])) for record in records if scores[record["tag"]] is not None}
    OUTPUT.write_text(json.dumps(dict(records=records, spec_sha256=digest(SPEC), provenance=provenance), indent=2) + "\n")
    print(f"Main table: {len(records)} model/reader rows, Omni-ID diagnostics and unchanged three-setting SWFI")


if __name__ == "__main__":
    main()
