"""Freeze and export SWFI v1 from existing CivLadder scores; no inference."""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import inspect
import json
from pathlib import Path
from statistics import mean

from paper import gen_tables as tables
from paper.audit_composite_forecasting import composite, normalize, reference_ranges
from paper.build_capability_correlations import METRICS, collect_panel


OUTPUT = tables.ROOT / "bench/measure/out/section5/swfi_v1"
LEVELS = {"L1": ["quantity", "event"], "L2": ["trait", "rules"],
          "L3": ["capture", "direction", "specificity"]}


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def extraction_signature():
    names = ["m_1a", "m_1b", "m_traits", "m_rules", "m_l3", "_l3_effect", "_l3_shift",
             "adjusted_accuracy"]
    payload = {name: inspect.getsource(getattr(tables, name)) for name in names}
    payload.update(PARAMS=tables.PARAMS, TRAITS=tables.TRAITS, MIN_OP_CELLS=tables.MIN_OP_CELLS)
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def freeze():
    if (OUTPUT / "spec.json").exists() or (OUTPUT / "inputs.json").exists():
        raise FileExistsError("SWFI v1 already exists; never silently redefine a frozen index")
    references = tables.ceiling()
    perfect = tables.load("perfect")
    panel = collect_panel()
    source_paths = [tables.CEIL, tables.TRAIT_CEIL, tables.TRAIT_CEIL_V3, tables.RULES_CEIL,
                    tables.DIR_CEIL, tables.EFFECT_CEIL, tables.RUN / "scores_raw_perfect.json",
                    tables.ROOT / "bench/score_v2.py", tables.ROOT / "bench/spec_v2.py"]
    spec = dict(
        name="Sandbox-World Forecasting Index", abbreviation="SWFI", version="1.0",
        frozen_at=datetime.now(timezone.utc).isoformat(), benchmark="CivLadder v3-bench-br",
        scope="Freeciv sandbox world-model tasks; not a validated real-world forecasting proxy",
        levels=LEVELS, level_weights=dict.fromkeys(LEVELS, 1/3),
        metric_weights={metric: 1/(3*len(metrics)) for metrics in LEVELS.values() for metric in metrics},
        setting_weights={key: 1/3 for key, _ in tables.SETTINGS},
        setting_labels=dict(tables.SETTINGS),
        ranges={key: reference_ranges(references, key) for key, _ in tables.SETTINGS},
        raw_accuracy_baselines={key: dict(trait=.5, direction=.5, rules=tables.m_rules(perfect, key)[1])
                                for key, _ in tables.SETTINGS},
        scaling="100 * equal-setting mean of equal-level means of (score-floor)/(ceiling-floor)",
        missing_policy="Reject undefined components; no imputation or weight redistribution. Inherits existing scorer coverage.",
        clipping="No additional clipping; CAPTURE and SPECIFICITY retain their existing metric-level clipping",
        interpretation="0 and 100 are reference endpoints, not guaranteed bounds or a success probability",
        exploration_tags=[record["tag"] for record in panel],
        prior_exposure="Existing model and external outcomes were viewed before definition; only new local model outcomes are held out",
        extraction_signature=extraction_signature(),
        provenance={str(path.relative_to(tables.ROOT)): digest(path) for path in source_paths if path.exists()},
    )
    inputs = [{key: record[key] for key in ("name", "tag", "source", "sha256")} for record in panel]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_json(OUTPUT / "spec.json", spec)
    write_json(OUTPUT / "inputs.json", inputs)


def extract(scores, spec):
    output = {}
    for setting in spec["setting_weights"]:
        capture, direction, specificity = tables.m_l3(scores, setting)
        values = dict(quantity=tables.m_1a(scores, setting), event=tables.m_1b(scores, setting),
                      trait=tables.m_traits(scores, setting), rules=tables.m_rules(scores, setting)[0],
                      capture=capture, direction=direction, specificity=specificity)
        for metric, baseline in spec["raw_accuracy_baselines"][setting].items():
            values[metric] = tables.adjusted_accuracy(values[metric], baseline, spec["ranges"][setting][metric][1])
        output[setting] = values
    return output


def calculate(values, spec):
    if set(values) != set(spec["setting_weights"]):
        raise ValueError("Every frozen setting must be present")
    settings = {}
    levels = {}
    normalized = {}
    for setting, components in values.items():
        if set(components) != set(METRICS):
            raise ValueError("Every frozen metric must be present")
        normalized[setting] = normalize(components, spec["ranges"][setting])
        settings[setting] = composite(normalized[setting], spec["metric_weights"])
        levels[setting] = {level: 100*mean(normalized[setting][metric] for metric in metrics)
                           for level, metrics in spec["levels"].items()}
    overall = composite(settings, spec["setting_weights"])/100
    return dict(overall=overall, settings=settings, levels=levels, normalized=normalized)


def coverage(path):
    anchors = set()
    settings = {}
    counts = Counter()
    with gzip.open(path, "rt") as stream:
        for line in stream:
            row = json.loads(line)
            anchor = f"{row['world']}_{row['t1']}.jsonl.gz"
            anchors.add(anchor)
            settings.setdefault(f"{row['set']}|{row['cond']}", set()).add(anchor)
            counts[row["kind"]] += 1
    if not anchors:
        raise ValueError(f"Empty scored records: {path}")
    return dict(anchors=len(anchors), anchor_ids=sorted(anchors),
                anchors_by_setting={key: len(value) for key, value in settings.items()},
                scored_records_by_kind=dict(counts), scored_sha256=digest(path),
                note="Observed scored anchors, not proof of all requested prompts or valid cells")


def export():
    spec_path = OUTPUT / "spec.json"
    spec = json.loads(spec_path.read_text())
    if extraction_signature() != spec["extraction_signature"]:
        raise ValueError("Metric extraction changed since v1 freeze; review/version before exporting")
    inputs_path = OUTPUT / "inputs.json"
    inputs = json.loads(inputs_path.read_text())
    rows = []
    for record in inputs:
        path = tables.ROOT / record["source"]
        if digest(path) != record["sha256"]:
            raise ValueError(f"Changed input: {path}; register a new result snapshot explicitly")
        scored = path.with_name(path.name.replace("scores_raw_", "scored_raw_").replace(".json", ".jsonl.gz"))
        values = extract(json.loads(path.read_text()), spec)
        rows.append(dict(record, name=record["name"].replace("~", " "), values=values,
                         scores=calculate(values, spec), coverage=coverage(scored)))
        print(f"Read {record['tag']}: {rows[-1]['coverage']['anchors']} scored anchors", flush=True)
    rows.sort(key=lambda record: (-record["scores"]["overall"], record["tag"]))
    result = dict(spec_sha256=digest(spec_path), inputs_sha256=digest(inputs_path),
                  status="Descriptive existing-model snapshot with mixed evaluation scopes; not matched-panel validation",
                  records=rows)
    write_json(OUTPUT / "scores.json", result)
    keys = list(spec["setting_weights"])
    with (OUTPUT / "scores.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["model", "tag", "SWFI", *keys, "scored_anchors"])
        for record in rows:
            writer.writerow([record["name"], record["tag"], record["scores"]["overall"],
                             *[record["scores"]["settings"][key] for key in keys], record["coverage"]["anchors"]])
    lines = ["# SWFI v1: Existing-model Snapshot", "", result["status"], "",
             "Fixed levels and settings, each weighted 1/3. Not accuracy; no additional clipping.",
             "Anchor counts reflect scored records, not complete/valid prompt coverage. No confidence intervals estimated.", "",
             "| Model | SWFI | Omni-ID | Ego | OOD | Scored anchors |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for record in rows:
        numbers = [record["scores"]["overall"]] + [record["scores"]["settings"][key] for key in keys]
        lines.append("| " + record["name"] + " | " + " | ".join(f"{value:.2f}" for value in numbers)
                     + f" | {record['coverage']['anchors']} |")
    (OUTPUT / "scores.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", action="store_true", help="Create v1 once; refuses to overwrite")
    args = parser.parse_args()
    if args.freeze:
        freeze()
    export()


if __name__ == "__main__":
    main()
