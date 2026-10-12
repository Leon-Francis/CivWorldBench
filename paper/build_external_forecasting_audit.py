"""Snapshot public forecasting sources and audit overlap; never run model inference."""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
from statistics import StatisticsError, correlation
from urllib.request import Request, urlopen

from paper.build_capability_correlations import METRICS, collect_panel
from paper.gen_tables import ROOT, SETTINGS


OUTPUT = ROOT / "bench/measure/out/section5/external_forecasting"
SOURCES = {
    "forecastbench_protocol.html": "https://www.forecastbench.org/leaderboards/",
    **{f"forecastbench_{track}.js":
       f"https://www.forecastbench.org/assets/js/leaderboard_{track}_full.js"
       for track in ("baseline", "tournament", "preliminary")},
    "futurex.json": "https://futurex.live/",
    "echo.json": "https://echo.unipat.ai/leaderboard",
    "prophet_arena.json": "https://www.prophetarena.co/leaderboard/forecast",
    "metaculus.json": "https://www.metaculus.com/futureeval/",
}
TARGETS = {
    "opus5": r"claude-opus-5(?:-|$)",
    "glm-5.3": r"glm-5\.3$",
    "kimi-k3": r"kimi-k3(?:-|$)",
    "dsv41-flash": r"deepseek-v4\.1-flash(?:-|$)",
    "q38-27b": r"qwen3\.8-27b(?:-|$)",
    "gemma4-31b": r"gemma-4-31b-it$",
}


def snapshot(output):
    output.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for filename, url in SOURCES.items():
        started = datetime.now(timezone.utc).isoformat()
        if filename.endswith(".json"):
            request = Request("https://r.jina.ai/" + url, headers={"Accept": "text/plain"})
            with urlopen(request, timeout=60) as response:
                payload = {"data": {"content": response.read().decode("utf-8"), "url": url}}
            contents = (json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode()
        else:
            with urlopen(url, timeout=30) as response:
                contents = response.read()
        (output / filename).write_bytes(contents)
        manifest[filename] = dict(url=url, retrieved_at_utc=started,
                                  sha256=hashlib.sha256(contents).hexdigest())
        (output / "sources.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print("Retrieved", filename, flush=True)


def parse_forecastbench(contents):
    match = re.search(r"const data\s*=\s*(\[.*?\]);", contents, re.DOTALL)
    if match is None:
        raise ValueError("ForecastBench data declaration not found")
    rows = ast.literal_eval(match.group(1))
    if not isinstance(rows, list) or not rows:
        raise ValueError("Empty or invalid ForecastBench table")
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("Model"), str):
            raise ValueError("Invalid ForecastBench row")
        for key in ("Dataset", "Overall"):
            if key in row and not math.isfinite(row[key]):
                raise ValueError("Nonfinite ForecastBench score")
    return rows


def candidate_rows(rows):
    return {tag: [row for row in rows if re.fullmatch(pattern + r".*", row["Model"].lower().rstrip("\u2020"))]
            for tag, pattern in TARGETS.items()}


def rank_diagnostic(external, local):
    if len(external) != len(local) or len(local) < 3:
        raise ValueError("At least three paired models required")
    if any(value is None or not math.isfinite(value) for value in external + local):
        raise ValueError("Missing/nonfinite model score")
    rho = correlation(external, local, method="ranked")
    null = [correlation(permuted, local, method="ranked")
            for permuted in itertools.permutations(external)]
    leave_one_out = []
    for index in range(len(local)):
        try:
            value = correlation(external[:index] + external[index + 1:],
                                local[:index] + local[index + 1:], method="ranked")
        except StatisticsError:
            value = None
        leave_one_out.append(value)
    return dict(n=len(local), rho=rho,
                exact_two_sided_permutation_p=sum(abs(value) >= abs(rho) - 1e-12
                                                 for value in null) / len(null),
                leave_one_model_out_rho=leave_one_out)


def build(output):
    manifest = json.loads((output / "sources.json").read_text())
    if set(manifest) != set(SOURCES):
        raise ValueError("Incomplete source manifest; finish snapshot collection before analysis")
    for filename, metadata in manifest.items():
        if hashlib.sha256((output / filename).read_bytes()).hexdigest() != metadata["sha256"]:
            raise ValueError(f"Snapshot hash mismatch: {filename}")
    coverage = {track: candidate_rows(parse_forecastbench(
        (output / f"forecastbench_{track}.js").read_text()))
        for track in ("baseline", "tournament", "preliminary")}
    panel = {record["tag"]: record for record in collect_panel() if record["tag"] in TARGETS}
    tags = [tag for tag, rows in coverage["tournament"].items() if rows]
    diagnostics = []
    if 3 <= len(tags) <= 8:
        for chosen in itertools.product(*(coverage["tournament"][tag] for tag in tags)):
            external = [row["Overall"] for row in chosen]
            result = {}
            for setting, label in SETTINGS:
                result[setting] = {metric: rank_diagnostic(
                    external, [panel[tag]["values"][setting][metric] for tag in tags])
                    for metric in METRICS}
            diagnostics.append(dict(tags=tags, external_rows=chosen, correlations=result))
    result = dict(
        status="Exploratory coverage audit only; no six-model external validity estimate",
        metric="ForecastBench tournament Overall Brier Index (higher is better)",
        warning="Name candidates are not verified checkpoint/protocol matches. Tournament assistance differs. "
                "All candidate row combinations retained; no best-row selection. Local anchor scopes differ. "
                "Permutation p-values are unadjusted model-label diagnostics, not event uncertainty.",
        forecastbench_candidates=coverage, civladder_records=panel, diagnostics=diagnostics,
        sources=manifest,
    )
    (output / "audit.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({track: {tag: len(rows) for tag, rows in matches.items()}
                      for track, matches in coverage.items()}, indent=2))
    for diagnostic in diagnostics:
        print(json.dumps(diagnostic["correlations"][SETTINGS[0][0]], indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--refresh", action="store_true", help="Fetch public sources; overwrites snapshots")
    args = parser.parse_args()
    if args.refresh:
        snapshot(args.output)
    build(args.output)


if __name__ == "__main__":
    main()
