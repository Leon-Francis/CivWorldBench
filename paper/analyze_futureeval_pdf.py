"""Extract the user-supplied FutureEval PDF and extend external-only diagnostics."""
from statistics import correlation

import pymupdf

from paper.aggregate_external_forecasting import (
    ROOT, EXTERNAL, aggregate, diagnostic, read, write, digest,
)


PDF = ROOT / "Top Model Leaderboards _ Metaculus.pdf"
OUTPUT = EXTERNAL / "futureeval_pdf_20260923"
PREVIOUS = EXTERNAL / "fullscope_aggregation_20260923/analysis.json"
MATCHES = {
    "metac-claude-opus-5+asknews": "opus5",
    "metac-kimi-k3+asknews": "kimi-k3",
    "metac-gpt-5-6-sol-high+asknews": "gpt56sol",
    "metac-nemotron-3-ultra+asknews": "nemotron3-ultra",
    "metac-gpt-oss-120b+asknews": "gpt-oss-120b",
}


def parse_blocks(blocks, page):
    output = []
    for block in blocks:
        fields = block[4].strip().splitlines()
        if not fields or not fields[0].isdigit():
            continue
        if len(fields) == 5 and fields[2].startswith("metac-"):
            rank, model, slug, forecasts, score = fields
            output.append(dict(rank=int(rank), model=model, slug=slug, forecasts=int(forecasts),
                               score=float(score), page=page, bbox=list(block[:4])))
        elif len(fields) == 4 and fields[1].startswith("Metaculus "):
            rank, model, forecasts, score = fields
            output.append(dict(rank=int(rank), model=model, slug=None, forecasts=int(forecasts),
                               score=float(score), page=page, bbox=list(block[:4])))
        else:
            raise ValueError(f"Unexpected numeric-leading PDF block: {fields}")
    return output


def extract_pdf(path):
    rows = []
    with pymupdf.open(path) as document:
        metadata = document.metadata
        pages = len(document)
        for page_number, page in enumerate(document, 1):
            rows.extend(parse_blocks(page.get_text("blocks"), page_number))
    if [row["rank"] for row in rows] != list(range(1, 100)):
        raise ValueError("Expected consecutive PDF ranks 1..99; inspect changed document")
    slugs = [row["slug"] for row in rows if row["slug"]]
    if len(slugs) != len(set(slugs)):
        raise ValueError("Duplicate model slug in PDF")
    return dict(metadata=metadata, pages=pages, rows=rows)


def select_models(rows):
    groups = {tag: [] for tag in MATCHES.values()}
    for row in rows:
        tag = MATCHES.get(row["slug"])
        if tag:
            groups[tag].append(row)
    if any(len(rows) != 1 for rows in groups.values()):
        raise ValueError("Missing or duplicate FutureEval match")
    return {tag: rows[0] for tag, rows in groups.items()}


def association_with_coverage(local, targets):
    tags = sorted(set(local) & set(targets))
    if len(tags) <= 8:
        return diagnostic(local, targets)
    return dict(n=len(tags), tags=tags,
                pearson=correlation([local[tag] for tag in tags], [targets[tag] for tag in tags]),
                spearman=correlation([local[tag] for tag in tags], [targets[tag] for tag in tags], method="ranked"),
                note="Point correlation only; heterogeneous board coverage, no inferential p-value")


def render(result):
    lines = ["# FutureEval PDF Analysis", "",
             "Source: user-supplied 7-page PDF printed 2026-09-23 15:30 China time; 99 rows, 97 model rows.",
             "Uses unified forecasting Avg Score based on log scores, NOT Brier or a probability.",
             "Uses maximum available approved local scopes; frozen SWFI weights and reference ranges.", "",
             "| Model | PDF page | Forecasts | FutureEval | Old SWFI | New SWFI |",
             "|---|---:|---:|---:|---:|---:|"]
    for tag, row in result["matched"].items():
        lines.append(f"| {row['model']} | {row['page']} | {row['forecasts']} | {row['score']:.2f} | "
                     f"{result['local']['old_mixed'][tag]:.3f} | {result['local']['new_mixed'][tag]:.3f} |")
    lines += ["", "| Version | Pearson | Spearman | Exact two-sided p | Drop-one Pearson |",
              "|---|---:|---:|---:|---|"]
    for scope, association in result["associations"].items():
        dropped = [row["pearson"] for row in association["leave_one_out"].values()]
        lines.append(f"| {scope} | {association['pearson']:.3f} | {association['spearman']:.3f} | "
                     f"{association['permutation_p']:.4f} | {min(dropped):.3f} to {max(dropped):.3f} |")
    lines += ["", "## Four-board Exploratory Aggregate", "",
              "External-only per-board configuration median, then z-score or midrank, then equal available-board mean.",
              "ForecastBench Preliminary and Tournament are alternative single-source tracks, never two votes.",
              "Coverage >=1 pools DIFFERENT external tests across models and is not a validated common scale.",
              "Coverage >=2 is stricter but still not a complete common panel; all four boards overlap on only two models.", "",
              "| ForecastBench track | Method | Minimum boards | N | Old Pearson | New Pearson | New Spearman |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for track, methods in result["aggregates"].items():
        for method, aggregate_record in methods.items():
            for threshold, associations in aggregate_record["associations"].items():
                old, new = associations["old_mixed"], associations["new_mixed"]
                lines.append(f"| {track} | {method} | {threshold} | {new['n']} | "
                             f"{old['pearson']:+.3f} | {new['pearson']:+.3f} | {new['spearman']:+.3f} |")
    lines += ["", "## Boundaries", "",
              "Gemma 4 in this PDF is metac-gemma-4-26b, not local 31B. DeepSeek V4 Flash is not V4.1 Flash.",
              "Gemini 3.6 Flash is not 3.7 Flash. GLM 5.2 is not 5.3. Grok 4.6 has no completed local run.",
              "Matched external rows use AskNews and some High reasoning settings; local configuration is not identical.",
              "PDF gives forecast counts but no score uncertainty or event-level records. This is a model-label association.",
              "Scores from other boards are dated one day earlier. No new-model/unknown-event holdout is added by this PDF.",
              "Positive five-model evidence is useful, but not proof of a reliable universal real-world proxy."]
    return "\n".join(lines) + "\n"


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    previous = read(PREVIOUS)
    if digest(ROOT / previous["prior_source"]) != previous["prior_sha256"]:
        raise ValueError("Previous local comparison changed")
    prior = read(ROOT / previous["prior_source"])
    for name, expected in prior["provenance"].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"Local/source inputs changed: {name}")
    extracted = extract_pdf(PDF)
    matched = select_models(extracted["rows"])
    target = {tag: row["score"] for tag, row in matched.items()}
    local = previous["local"]
    associations = {scope: diagnostic(scores, target) for scope, scores in local.items()}
    aggregates = {}
    for track in ("forecastbench4", "forecastbench_preliminary5"):
        boards = {name: previous["model_level_board_medians"][name] for name in (track, "futurex4", "prophet7")}
        boards["futureeval5"] = target
        aggregates[track] = {}
        for method in ("zscore", "midrank"):
            index = aggregate(boards, method)
            results = {}
            for threshold in (1, 2):
                eligible = {tag: row["score"] for tag, row in index.items() if row["coverage"] >= threshold}
                results[str(threshold)] = {scope: association_with_coverage(scores, eligible)
                                          for scope, scores in local.items()}
            aggregates[track][method] = dict(records=index, associations=results)
    result = dict(pdf_source=str(PDF.relative_to(ROOT)), pdf_sha256=digest(PDF),
                  source_url="https://www.metaculus.com/futureeval/leaderboard/",
                  code_sha256=digest(ROOT / "paper/analyze_futureeval_pdf.py"),
                  previous_source=str(PREVIOUS.relative_to(ROOT)), previous_sha256=digest(PREVIOUS),
                  extracted=extracted, matched=matched, local=local, associations=associations, aggregates=aggregates)
    write(OUTPUT / "analysis.json", result)
    (OUTPUT / "analysis.md").write_text(render(result))
    with pymupdf.open(PDF) as document:
        for number in (0, 3, 4, 5):
            document[number].get_pixmap(matrix=pymupdf.Matrix(1.5, 1.5)).save(str(OUTPUT / f"page_{number + 1}.png"))
    print(render(result))


if __name__ == "__main__":
    main()
