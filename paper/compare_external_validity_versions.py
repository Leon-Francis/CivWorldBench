"""Paired offline score-policy audit with frozen SWFI scales and external snapshots."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import gzip
import json
import math
from pathlib import Path
import re
from statistics import mean

from bench.score_v2 import build_insample_cells, score_anchor, summarize
from paper.analyze_external_validity import (
    forecastbench_groups, markdown_rows, number_range, sensitivity, variants,
)
from paper.audit_composite_forecasting import SCHEMES, composite
from paper.build_external_forecasting_audit import parse_forecastbench
from paper.build_sandbox_index import calculate, digest, extract
from paper import gen_tables as tables


ROOT = tables.ROOT
RUN = tables.RUN
EXTERNAL = ROOT / "bench/measure/out/section5/external_forecasting"
FROZEN = ROOT / "bench/measure/out/section5/swfi_v1"
OUTPUT = EXTERNAL / "version_comparison_20260923"
NEW = {
    "gemini-3.7-flash": "swfi_v1_gemini37flash_c60",
    "gpt-6-astra": "swfi_v1_gpt6astra_c60",
    "glm-5.3-flash": "swfi_v1_glm53flash_c60",
    "hy3": "swfi_v1_hy3_c60",
}
EXISTING = ("opus5", "gpt56sol", "kimi-k3", "gemma4-31b", "glm-5.3")
PROPHET_NAMES = {name: name for name in ("glm-5.3", "kimi-k3", *NEW)} | {"gpt-5.6-sol": "gpt56sol"}
FUTUREX_NAMES = {"claude-opus-5": "opus5", "Kimi K3": "kimi-k3",
                "gpt-5.6-sol": "gpt56sol", "gpt-6-astra": "gpt-6-astra"}


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def anchor_id(row):
    return f"{row['world']}_{row['t1']}.jsonl.gz"


def select_rows(rows, anchors):
    selected = [row for row in rows if anchor_id(row) in anchors]
    observed = {anchor_id(row) for row in selected}
    if observed != set(anchors):
        raise ValueError(f"Missing selected anchors: {set(anchors) - observed}")
    return selected


def external_groups(prophet, futurex, forecastbench):
    groups = {tag: [] for tag in PROPHET_NAMES.values()}
    for row in markdown_rows(prophet):
        match = re.search(r"\]\(https://www\.prophetarena\.co/model/([^\s)]+)", row["Model"])
        tag = PROPHET_NAMES.get(match.group(1)) if match else None
        if tag is None or row["Type"] != "Fixed Context":
            continue
        skill = row[next(key for key in row if key.startswith("Skill "))]
        value = re.match(r"[+-]?\d+(?:\.\d+)?", skill)
        if value is None:
            raise ValueError("Missing Prophet theta")
        groups[tag].append(dict(score=float(value.group()), source_row=row))
    if any(len(rows) != 1 for rows in groups.values()):
        raise ValueError("Expected exactly one fixed-context Prophet row per model")
    future = {tag: [] for tag in FUTUREX_NAMES.values()}
    for row in markdown_rows(futurex):
        tag = FUTUREX_NAMES.get(row["MODEL"])
        if tag:
            future[tag].append(dict(score=float(row["SCORE"]), source_row=row))
    return dict(prophet7=groups, prophet_new4={tag: groups[tag] for tag in NEW},
                futurex4=future, forecastbench4=forecastbench_groups(forecastbench))


def heldout_prediction(records, external, fit, baseline):
    rows = []
    for tag in NEW:
        actual = external[tag][0]["score"]
        predicted = fit["slope"] * records[tag]["index"]["overall"] + fit["intercept"]
        rows.append(dict(tag=tag, actual=actual, predicted=predicted, baseline=baseline))
    errors = {}
    for name in ("predicted", "baseline"):
        residuals = [row[name] - row["actual"] for row in rows]
        errors[name] = dict(mae=mean(abs(value) for value in residuals),
                            rmse=math.sqrt(mean(value * value for value in residuals)))
    errors["rmse_skill_vs_constant"] = 1 - errors["predicted"]["rmse"] / errors["baseline"]["rmse"]
    return dict(rows=rows, errors=errors, fit=fit, baseline=baseline)


def summarize_job(job):
    cache = OUTPUT / "cache" / f"{job['version']}_{job['tag']}.json"
    if cache.exists():
        saved = read(cache)
        if saved["fingerprint"] == job["fingerprint"]:
            print(f"CACHE {job['version']} {job['tag']}", flush=True)
            return saved["records"]
        print(f"REFRESH {job['version']} {job['tag']}: audit inputs changed", flush=True)
    union = set().union(*[set(value) for value in job["subsets"].values()])
    rows = []
    if job["raw"]:
        build_insample_cells("v3-bench-br")
        for name in sorted(union):
            rows.extend(score_anchor((RUN / "questions" / name, Path(job["path"]) / name, None, True)))
    else:
        with gzip.open(job["path"], "rt") as stream:
            for line in stream:
                row = json.loads(line)
                if anchor_id(row) in union:
                    rows.append(row)
    spec = job["spec"]
    output = {}
    for label, anchors in job["subsets"].items():
        selected = select_rows(rows, anchors)
        summary = summarize(selected, 200, unparsed_as_wrong=job["version"] == "new")
        values = extract(summary, spec)
        output[label] = dict(tag=job["tag"], values=values, index=calculate(values, spec),
                             anchors=len(anchors), record_count=len(selected),
                             policy=summary.get("unparsed_policy"), source=job["path"],
                             input_fingerprint=job["fingerprint"])
    write(cache, dict(fingerprint=job["fingerprint"], records=output))
    print(f"DONE {job['version']} {job['tag']}: " +
          str({label: round(value['index']['overall'], 4) for label, value in output.items()}), flush=True)
    return output


def compare(groups, records, spec):
    tags = list(groups)
    results = []
    for chosen in variants(groups):
        target = [chosen[tag]["score"] for tag in tags]
        schemes = {}
        for scheme, weights in SCHEMES.items():
            by_setting = {}
            local_scores = {}
            for tag in tags:
                normalized = calculate(records[tag]["values"], spec)["normalized"]
                local_scores[tag] = {setting: composite(values, weights)
                                     for setting, values in normalized.items()}
                local_scores[tag]["settings_mean"] = sum(
                    local_scores[tag][setting] * weight for setting, weight in spec["setting_weights"].items())
            for setting in [*spec["setting_weights"], "settings_mean"]:
                local = [local_scores[tag][setting] for tag in tags]
                by_setting[setting] = sensitivity(local, target, tags)
            schemes[scheme] = by_setting
        results.append(dict(tags=tags, external_rows=chosen, results=schemes))
    return results


def render(result):
    lines = ["# SWFI: Paired Version Comparison", "",
             "Offline diagnostic only. No inference, weight fitting, or manuscript changes.",
             "SWFI v1 numeric ranges, level weights and three-setting weights are fixed.",
             "Old = preserved scoring artifacts. New = corrected parser plus unparsed-as-wrong-v2.",
             "External snapshots stay fixed at 2026-09-22; model aliases are exact snapshot labels, not independently verified API identities.",
             "Ranges below span ALL eligible external configurations; they are not confidence intervals.", "",
             "## Matched-panel Associations", "",
             "| Scope | Dataset | Models | Pearson | Spearman | Exact two-sided p |",
             "|---|---|---:|---:|---:|---:|"]
    for scope, datasets in result["comparisons"].items():
        for name, comparisons in datasets.items():
            metrics = [number_range([item["results"]["levels_equal"]["settings_mean"][key]
                                    for item in comparisons])
                       for key in ("pearson", "spearman", "permutation_p")]
            lines.append(f"| {scope} | {name} | {len(comparisons[0]['tags'])} | " + " | ".join(metrics) + " |")
    lines += ["", "## Historical Mixed-scope Check (Descriptive Only)", "",
              "These correlations use each model's original main-table scope, NOT matched anchors.", "",
              "| Scope | Dataset | Pearson |", "|---|---|---:|"]
    for scope, datasets in result["mixed_scope_comparisons"].items():
        for name, comparisons in datasets.items():
            value = number_range([item["results"]["levels_equal"]["settings_mean"]["pearson"] for item in comparisons])
            lines.append(f"| {scope} | {name} | {value} |")
    lines += ["", "## Same 60 Anchors, Same Weights", "",
              "| Model | Old SWFI | New SWFI | Delta |", "|---|---:|---:|---:|"]
    for tag in result["records"]["old60"]:
        old = result["records"]["old60"][tag]["index"]["overall"]
        new = result["records"]["new60"][tag]["index"]["overall"]
        lines.append(f"| {tag} | {old:.3f} | {new:.3f} | {new-old:+.3f} |")
    lines += ["", "## Truly Held-out Local Model Outcomes", "",
              "Apply the old-three-model affine calibration, frozen before new local outcomes, to all four new models.",
              "External outcomes were already visible before local evaluation; this is not an unseen-future-events test.",
              "Do not refit weights, slope or intercept on these four outcomes.", "",
              "| Policy | Prediction MAE | Constant MAE | Prediction RMSE | Constant RMSE | RMSE skill |",
              "|---|---:|---:|---:|---:|---:|"]
    for scope, value in result["heldout"].items():
        errors = value["errors"]
        lines.append(f"| {scope} | {errors['predicted']['mae']:.4f} | {errors['baseline']['mae']:.4f} | "
                     f"{errors['predicted']['rmse']:.4f} | {errors['baseline']['rmse']:.4f} | "
                     f"{errors['rmse_skill_vs_constant']:+.3f} |")
    lines += ["", "The new-policy row deliberately keeps the old calibration and is a policy-transfer sensitivity,",
              "not a newly preregistered calibration. Exact permutation tests are descriptive: model families",
              "are not independent samples and external estimates are rounded/noisy. No multiplicity correction.",
              "Internal scorer bootstrap intervals are NOT SWFI or external-correlation confidence intervals.", "",
              "## Interpretation", "",
              "A favorable association on a four-model board is convergent evidence, not validation of a reliable proxy.",
              "Assess cross-board consistency, held-out calibration, settings, and leave-one-model/family sensitivity together.",
              "The common190 comparison is available only for ForecastBench's four overlapping models.",
              "ECHO and Metaculus are excluded for insufficient verified exact overlap. Grok's incomplete run is excluded.",
              "Mixed-scope panel scores are retained in JSON for provenance, not used as matched validation."]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    (OUTPUT / "cache").mkdir(parents=True, exist_ok=True)
    paths = [FROZEN / name for name in ("spec.json", "scores.json", "inputs.json")]
    paths += [ROOT / name for name in ("bench/score_v2.py", "bench/parse_v2.py", "paper/gen_tables.py",
              "paper/build_sandbox_index.py", "paper/audit_composite_forecasting.py", "paper/analyze_external_validity.py",
              "paper/compare_external_validity_versions.py", "bench/spec_v2.py", "bench/score_1c.py")]
    reference_paths = [ROOT / name for name in (
        "bench/v2/norms.json", "bench/v2/climatology.json", "bench/v2/cells_1b.json",
        "bench/v2/cells_1b_pooled.json", "bench/v2/cells_1b_insample_v3-bench-br.json",
        "data_v2/bench/manifest.json", "bench/runs/v3-bench-br/parsefail_v2_review.json")]
    paths += reference_paths
    spec = read(FROZEN / "spec.json")
    panel = {row["tag"]: row for row in read(FROZEN / "scores.json")["records"]}
    calibration_path = RUN / "api_swfi60/calibration_existing3.json"
    calibration = read(calibration_path)
    paths.append(calibration_path)
    anchors = {}
    for label, filename, count in (("60", "api_subset_60of120.txt", 60), ("190", "api_subset_190.txt", 190)):
        path = ROOT / "bench/v2" / filename
        paths.append(path)
        anchors[label] = set(path.read_text().split())
        if len(anchors[label]) != count:
            raise ValueError("Unexpected manifest size")
    if not anchors["60"] <= anchors["190"]:
        raise ValueError("Expanded set must contain all original anchors")
    if digest(paths[-2]) != calibration["anchors_sha256"] or digest(FROZEN / "spec.json") != calibration["spec_sha256"]:
        raise ValueError("Calibration scales or anchors changed")
    for tag in ("opus5", "gpt56sol"):
        for name in sorted(anchors["60"]):
            original, expanded = RUN / f"raw_{tag}" / name, RUN / f"raw_{tag}_api190" / name
            if digest(original) != digest(expanded):
                raise ValueError(f"Expanded run changed an original response: {tag} {name}")
            paths.extend((original, expanded))
    sources = read(EXTERNAL / "sources.json")
    paths.append(EXTERNAL / "sources.json")
    for name, source in sources.items():
        path = EXTERNAL / name
        if digest(path) != source["sha256"]:
            raise ValueError(f"External snapshot changed: {name}")
        paths.append(path)
    groups = external_groups(read(EXTERNAL / "prophet_arena.json")["data"]["content"],
                             read(EXTERNAL / "futurex.json")["data"]["content"],
                             parse_forecastbench((EXTERNAL / "forecastbench_tournament.js").read_text()))
    records = {scope: {} for scope in ("old60", "new60", "new190", "old_mixed", "new_mixed")}
    for tag, row in panel.items():
        path = ROOT / row["source"]
        if digest(path) != row["sha256"]:
            raise ValueError(f"Legacy summary changed: {path}")
        paths.append(path)
        records["old_mixed"][tag] = dict(values=row["values"], index=calculate(row["values"], spec),
                                         anchors=row["coverage"]["anchors"])
        new_path = RUN / f"scores_raw_{tables.PANEL_SCORE_TAGS[tag]}.json"
        paths.append(new_path)
        values = extract(read(new_path), spec)
        records["new_mixed"][tag] = dict(values=values, index=calculate(values, spec),
                                         source=str(new_path.relative_to(ROOT)))
    jobs = []
    for tag in EXISTING:
        for version in ("old", "new"):
            source = (ROOT / panel[tag]["source"] if version == "old"
                      else RUN / f"scores_raw_{tables.PANEL_SCORE_TAGS[tag]}.json")
            scored = source.with_name(source.name.replace("scores_raw_", "scored_raw_").replace(".json", ".jsonl.gz"))
            if version == "old" and digest(scored) != panel[tag]["coverage"]["scored_sha256"]:
                raise ValueError(f"Legacy scored rows changed: {tag}")
            paths.append(scored)
            subsets = {"60": sorted(anchors["60"])}
            if version == "new" and tag != "glm-5.3":
                subsets["190"] = sorted(anchors["190"])
            jobs.append(dict(tag=tag, version=version, path=str(scored), raw=False, subsets=subsets, spec=spec,
                             fingerprint={str(scored): digest(scored)}))
    for tag, raw_tag in NEW.items():
        index_path = RUN / "api_swfi60" / tag / "index.json"
        old = read(index_path)
        paths += [index_path, ROOT / old["source"]]
        if digest(ROOT / old["source"]) != old["source_sha256"] or old["spec_sha256"] != digest(FROZEN / "spec.json"):
            raise ValueError(f"Old new-model result changed: {tag}")
        records["old60"][tag] = dict(values=old["values"], index=calculate(old["values"], spec), anchors=60)
        records["old_mixed"][tag] = records["old60"][tag]
        raw_path = RUN / f"raw_{raw_tag}"
        selected_paths = [directory / name for directory in (raw_path, RUN / "questions") for name in sorted(anchors["60"])]
        paths += selected_paths
        jobs.append(dict(tag=tag, version="new", path=str(raw_path), raw=True, subsets={"60": sorted(anchors["60"])},
                         spec=spec, fingerprint={str(path): digest(path) for path in selected_paths}))
    provenance = {str(path.relative_to(ROOT)): digest(path) for path in sorted(set(paths))}
    code_hashes = {path: value for path, value in provenance.items() if path.endswith(".py") or path.endswith("spec.json") or path.endswith(".txt")}
    code_hashes.update({str(path.relative_to(ROOT)): digest(path) for path in reference_paths})
    for job in jobs:
        job["fingerprint"].update(code_hashes)
    with ProcessPoolExecutor(args.workers) as executor:
        for job, output in zip(jobs, executor.map(summarize_job, jobs)):
            for label, value in output.items():
                records[job["version"] + label][job["tag"]] = value
    for row in calibration["records"]:
        reconstructed = records["old60"][row["tag"]]["index"]["overall"]
        if not math.isclose(reconstructed, row["swfi"], abs_tol=1e-9):
            raise ValueError(f"Legacy aggregation no longer reproduces frozen calibration: {row['tag']}")
    for tag in NEW:
        records["new_mixed"][tag] = records["new60"][tag]
    comparisons = {scope: {name: compare(group, records[scope], spec) for name, group in groups.items()}
                   for scope in ("old60", "new60")}
    comparisons["new190"] = dict(forecastbench4=compare(groups["forecastbench4"], records["new190"], spec))
    mixed_comparisons = {scope: {name: compare(group, records[scope], spec) for name, group in groups.items()}
                        for scope in ("old_mixed", "new_mixed")}
    original_three = {tag: rows for tag, rows in groups["prophet7"].items() if tag not in NEW}
    original_checks = {scope: compare(original_three, records[scope], spec)
                       for scope in ("old60", "new60", "old_mixed", "new_mixed")}
    heldout = {scope: heldout_prediction(records[scope], groups["prophet7"], calibration["fits"]["swfi"],
                                         calibration["mean_baseline"]) for scope in ("old60", "new60")}
    family_groups = {family: {tag: rows for tag, rows in groups["prophet7"].items() if tag not in excluded}
                     for family, excluded in {"GLM": ["glm-5.3", "glm-5.3-flash"],
                                               "GPT": ["gpt56sol", "gpt-6-astra"]}.items()}
    family_checks = {scope: {family: compare(group, records[scope], spec)
                             for family, group in family_groups.items()} for scope in ("old60", "new60")}
    for name, expected in provenance.items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"Input changed during analysis: {name}")
    result = dict(generated_at=datetime.now(timezone.utc).isoformat(), spec=spec, provenance=provenance,
                  records=records, comparisons=comparisons, heldout=heldout, leave_family_out=family_checks,
                  original_three_checks=original_checks, mixed_scope_comparisons=mixed_comparisons,
                  sources=sources, uncertainty="Point-score association, not event/anchor uncertainty propagation")
    write(OUTPUT / "comparison.json", result)
    (OUTPUT / "comparison.md").write_text(render(result))
    print(render(result), flush=True)


if __name__ == "__main__":
    main()
