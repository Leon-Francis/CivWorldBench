"""Regenerate released benchmark results with CPU-only, offline stages."""
from __future__ import annotations

import argparse
from collections import Counter
import contextlib
import gzip
import importlib
import io
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "tools", ROOT / "bundle", ROOT):
    sys.path.insert(0, str(directory))
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

STAGES = ("main", "details", "cot", "appendix", "usability", "external", "score")
SNAPSHOTS = {
    "main": ["bench/measure/out/section5/main_table_compact.json"],
    "details": ["bench/measure/out/section5/results.json"],
    "cot": ["paper/cot_results.json"],
    "appendix": ["bench/measure/out/section5/appendix_summary.json"],
    "usability": ["paper/usability_evidence.json", "paper/usability_figure_data.json"],
}


def read_json(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def numeric_values(value, path=()):
    if isinstance(value, bool):
        return {}
    if isinstance(value, (int, float)):
        return {path: value}
    result = {}
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"provenance", "source_hashes"} or "sha256" in key or "digest" in key:
                continue
            result.update(numeric_values(item, (*path, key)))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            result.update(numeric_values(item, (*path, str(index))))
    return result


def compare_numbers(expected, actual, label):
    before, after = numeric_values(expected), numeric_values(actual)
    if before.keys() != after.keys():
        differences = sorted(before.keys() ^ after.keys())
        raise ValueError(f"{label}: numeric fields changed: {differences[:5]}")
    for key, value in before.items():
        if not math.isclose(value, after[key], rel_tol=1e-9, abs_tol=1e-9):
            location = "/".join(key)
            raise ValueError(f"{label}: {location}: expected {value}, got {after[key]}")
    return len(before)


def call_main(module):
    importlib.import_module(module).main()


def main_tables():
    from paper import gen_tables as gt

    call_main("paper.build_main_table")
    scores = {tag: gt.load(tag) for _, tag, _ in gt.LINES}
    references = gt.ceiling()
    for filename, text in (("raw_accuracies.tex", gt.table_raw_accuracies(scores)),
                           ("l3_capture.tex", gt.table_l3(scores))):
        (gt.OUT_T / filename).write_text(text, encoding="utf-8")
    lines = [(name.replace("~", " ").replace("\\expert", "Expert"), tag)
             for name, tag, _ in gt.LINES]
    radar_tags = {"gpt-oss-120b", "tk", "kimi-k3", "opus5", gt.EXPERT_TAG}
    gt.radar([(name, tag) for name, tag in lines if tag in radar_tags], scores,
             references, gt.OUT_F / "radar.pdf")
    gt.fig_1b_templates(lines, scores, references, gt.OUT_F / "onebee_templates.pdf")


def detailed_results():
    call_main("paper.build_results_analysis")
    call_main("paper.build_direction_magnitude")


def usability_results():
    call_main("paper.build_usability_results")
    call_main("paper.build_usability_figures")


def external_results():
    from paper import build_external_validity_tables as external
    from paper import gen_tables as gt
    from paper.build_sandbox_index import calculate, extract

    result = read_json("bench/measure/out/section5/external_forecasting/manuscript_20260923/analysis.json")
    spec = read_json("bench/measure/out/section5/swfi_v1/spec.json")
    checked = 0
    for _, tag, _ in gt.LINES:
        if tag not in result["local"]["new_mixed"]:
            continue
        current = calculate(extract(gt.load(tag), spec), spec)["overall"]
        expected = result["local"]["new_mixed"][tag]
        if not math.isclose(current, expected, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError(f"External numerical snapshot disagrees with current SWFI: {tag}")
        checked += 1
    for filename, renderer in (("external_validity", external.render_main),
                               ("external_matches", external.render_matches),
                               ("external_sensitivity", external.render_sensitivity)):
        (gt.OUT_T / (filename + ".tex")).write_text(renderer(result), encoding="utf-8")
    external.make_figure(result)
    return {"current_main_models_checked": checked, "numeric_values_checked": checked,
            "source": "Packaged parsed external metrics and archived association estimates"}


def scorer_smoke():
    from bench import score_v2

    run = ROOT / "bench/runs/v3-bench-br"
    question = next(path for path in sorted((run / "questions").glob("*.jsonl.gz"))
                    if (run / "raw_tk" / path.name).is_file())
    score_v2.build_insample_cells("v3-bench-br")
    actual = score_v2.score_anchor((question, run / "raw_tk" / question.name, None, True))
    with gzip.open(question, "rt", encoding="utf-8") as stream:
        anchor = json.loads(next(stream))
    expected = []
    with gzip.open(run / "scored_raw_tk_parsefail_v2.jsonl.gz", "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["world"] == anchor["world"] and row["t1"] == anchor["t1"]:
                expected.append(row)
    if not actual or len(actual) != len(expected):
        raise ValueError(f"Scorer row count differs: {len(actual)} versus {len(expected)}")
    identity_fields = ("world", "t1", "set", "cond", "kind", "sub", "op", "arm_kind",
                       "control_ref", "target", "civ", "var", "tpl", "h")

    def identity(row):
        return json.dumps([row.get(field) for field in identity_fields], ensure_ascii=True)

    expected_ids = Counter(identity(row) for row in expected)
    actual_ids = Counter(identity(row) for row in actual)
    if expected_ids != actual_ids:
        raise ValueError("Scorer row identities differ from the released records")
    if any(count != 1 for count in expected_ids.values()):
        raise ValueError("Scorer smoke anchor has ambiguous duplicate row identities")
    expected.sort(key=identity)
    actual.sort(key=identity)
    checked = compare_numbers(expected, actual, "one-anchor scorer")
    return {"anchor": question.name, "rows": len(actual), "numeric_values_checked": checked,
            "rows_by_kind": dict(Counter(row["kind"] for row in actual))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stages", default=",".join(STAGES),
                        help="Comma-separated stages: " + ",".join(STAGES))
    parser.add_argument("--check", action="store_true",
                        help="Compare regenerated numeric JSON values with the packaged snapshots")
    args = parser.parse_args()
    selected = list(dict.fromkeys(args.stages.split(",")))
    unknown = set(selected) - set(STAGES)
    if unknown:
        parser.error("Unknown stages: " + ", ".join(sorted(unknown)))
    if (ROOT / "data/index.json").exists() and not any(
            (ROOT / "bench/runs/v3-bench-br/questions").glob("*.jsonl.gz")):
        parser.error("Unpack the data first: python tools/unpack.py")
    if not (ROOT / "data/index.json").is_file():
        parser.error("Dataset: In preparation. This code release contains no benchmark data. Numerical reproduction requires the matching data release.")
    os.chdir(ROOT)
    for path in ("paper/tables", "paper/figures", "outputs"):
        (ROOT / path).mkdir(parents=True, exist_ok=True)
    snapshots = {path: read_json(path) for stage in selected
                 for path in SNAPSHOTS.get(stage, [])} if args.check else {}
    functions = {
        "main": main_tables,
        "details": detailed_results,
        "cot": lambda: call_main("paper.build_cot_results"),
        "appendix": lambda: call_main("paper.build_appendix_summary"),
        "usability": usability_results,
        "external": external_results,
        "score": scorer_smoke,
    }
    report = {"stages": {}, "check": args.check,
              "numeric_tolerance": {"relative": 1e-9, "absolute": 1e-9},
              "scope": "Offline CPU reproduction; no inference, API calls, or engine simulations"}
    for stage in selected:
        print(f"Running {stage}...", flush=True)
        started = time.monotonic()
        with contextlib.redirect_stdout(io.StringIO()):
            details = functions[stage]()
        checked = sum(compare_numbers(snapshots[path], read_json(path), path)
                      for path in SNAPSHOTS.get(stage, []) if path in snapshots)
        if isinstance(details, dict):
            checked += details.get("numeric_values_checked", 0)
        report["stages"][stage] = {"seconds": round(time.monotonic() - started, 3),
                                   "numeric_values_checked": checked, "details": details}
        (ROOT / "outputs/reproduction.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(f"Completed {stage}; checked {checked} numeric values.", flush=True)
    print("Results: paper/tables/, paper/figures/, outputs/reproduction.json")


if __name__ == "__main__":
    main()
