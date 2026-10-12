"""Prepare the authorized 22-overlap + 8-new Sol subset without API calls."""
from __future__ import annotations

import collections
import gzip
import hashlib
import json
import os
import shutil
from pathlib import Path
from statistics import mean

from bench.cot.run_cot_v31 import key_of
from bench.cot.verify_aligned import verify
from bench.run_infer_v2 import SYSTEM

ROOT = Path(__file__).resolve().parents[2]
PARENT = ROOT / "bench/runs/cot-v32-190"
SOURCE = ROOT / "bench/runs/v3-bench-br"
RUN = ROOT / "bench/runs/cot-v32-sol30"
RAW = "raw_gpt56sol"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path):
    with gzip.open(path, "rt") as stream:
        return [json.loads(line) for line in stream]


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        stream.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def save_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with gzip.open(temporary, "wt") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    temporary.replace(path)


def select():
    metadata = json.loads((PARENT / "selection.json").read_text())["anchors"]
    by_name = {row["file"]: row for row in metadata}
    api = set((ROOT / "bench/v2/api_subset_190.txt").read_text().split())
    fixed = sorted(set(by_name) & api)
    assert len(fixed) == 22
    assert collections.Counter(by_name[n]["stratum"][0] for n in fixed) == {
        "bench-basic": 17, "bench-gen": 5}
    selected = list(fixed)
    gen = [r for r in metadata if r["stratum"][0] == "bench-gen"]
    dimensions = [lambda r: r["stratum"][2], lambda r: r["phase"]]
    targets = [collections.Counter(fn(r) for r in gen) for fn in dimensions]
    # Fixed basic coverage already exceeds the proportional quota; all eight additions are OOD.
    for _ in range(8):
        def rank(row):
            proposed = [by_name[n] for n in selected if by_name[n]["stratum"][0] == "bench-gen"] + [row]
            error = 0.0
            for fn, target in zip(dimensions, targets):
                count = collections.Counter(fn(r) for r in proposed)
                error += sum((count[k] - 13 * v / len(gen)) ** 2 / (13 * v / len(gen))
                             for k, v in target.items())
            tie = hashlib.sha256(("sol30-20260924|" + row["file"]).encode()).hexdigest()
            return error, tie
        chosen = min((r for r in gen if r["file"] not in selected), key=rank)
        selected.append(chosen["file"])
    return fixed, selected[22:], by_name


def main():
    fixed, added, metadata = select()
    names = sorted(fixed + added)
    manifest_path = RUN / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        assert manifest["anchors"] == names
        for name, digest in manifest["questions"].items():
            assert sha(RUN / "questions" / name) == digest
        print(json.dumps({"already_prepared": True, "run": str(RUN)}))
        return
    (RUN / "questions").mkdir(parents=True, exist_ok=True)
    questions, reuse, provenance, inventory = {}, {}, {}, collections.Counter()
    ratios = []
    historical_output = collections.defaultdict(list)
    for name in names:
        source_path = PARENT / "questions" / name
        destination = RUN / "questions" / name
        if destination.exists() and sha(destination) != sha(source_path):
            raise ValueError(f"Different question copy: {name}")
        shutil.copyfile(source_path, destination)
        rows = read_rows(destination)
        questions[name] = rows
        inventory.update(q["fmt"] for q in rows)
        source_questions = {key_of(dict(q, fmt="free")): q for q in read_rows(SOURCE / "questions" / name)}
        for q in rows:
            original = source_questions[key_of(dict(q, fmt="free"))]
            if q["fmt"] == "free":
                assert q["prompt"] == original["prompt"]
                assert q["max_tokens"] == original["max_tokens"]
                assert (q.get("system") or SYSTEM) == (original.get("system") or SYSTEM)
        if name not in fixed:
            continue
        raw_path = SOURCE / "raw_gpt56sol_api190" / name
        old = {key_of(dict(r, fmt="free")): r for r in read_rows(raw_path)}
        copied = []
        for q in rows:
            if q["fmt"] != "free":
                continue
            r = old[key_of(q)]
            assert r.get("model") == "gpt-5.6-sol" and not str(r.get("finish", "")).startswith("error")
            assert r.get("output", "").strip()
            for field in ("world", "t1", "cond", "task", "sub", "arm_kind", "op", "horizons", "items", "control_ref"):
                if field in r:
                    assert r[field] == q.get(field), (name, field)
            record = dict(r, fmt="free", protocol=q["protocol"], reuse_source=str(raw_path.relative_to(ROOT)),
                          reuse_source_sha256=sha(raw_path))
            copied.append(record)
            ratios.append(r["n_prompt"] / (len(q["prompt"]) + len(q.get("system") or SYSTEM)))
            historical_output[(q["task"], q.get("sub"))].append(r["n_gen"])
        save_rows(RUN / RAW / name, copied)
        reuse[name] = len(copied)
        provenance[name] = {"raw": str(raw_path.relative_to(ROOT)), "raw_sha256": sha(raw_path),
                            "question_sha256": sha(SOURCE / "questions" / name)}
    verification = verify(RUN)
    atomic_json(RUN / "verification.json", verification)
    if not verification["ok"]:
        raise ValueError(verification["errors"][:5])
    char_ratio = mean(ratios)
    proxies = {}
    missing_free = [q for n in added for q in questions[n] if q["fmt"] == "free"]
    for tag in ("flash-next", "q38-27b", "gemma4-31b"):
        totals = collections.defaultdict(collections.Counter)
        for name in names:
            local = {key_of(r): r for r in read_rows(PARENT / f"raw_{tag}" / name)}
            for q in questions[name]:
                if q["fmt"] == "free" and name in fixed:
                    continue
                fmt = q["fmt"]
                estimated_input = (len(q["prompt"]) + len(q.get("system") or SYSTEM)) * char_ratio
                if fmt == "free":
                    output = mean(historical_output[(q["task"], q.get("sub"))])
                    calls = 1
                else:
                    r = local[key_of(q)]
                    assert not str(r.get("finish", "")).startswith("error")
                    output = r["n_gen"]
                    calls = r.get("n_rounds", 1)
                    if fmt == "code":
                        estimated_input *= r["n_prompt"] / r["rounds"][0]["n_prompt"]
                totals[fmt].update(rows=1, calls=calls, input_tokens=estimated_input, output_tokens=output)
        proxies[tag] = {}
        for fmt, values in totals.items():
            proxies[tag][fmt] = dict(values, usd_input5=(values["input_tokens"] * 5 + values["output_tokens"] * 30) / 1_050_000,
                                     usd_all_input_cachewrite=(values["input_tokens"] * 6.25 + values["output_tokens"] * 30) / 1_050_000)
    quote = {"currency": "USD", "tokens_per_unit": 1_050_000,
             "prices": {"input": 5, "output": 30, "cache_write": 6.25, "cache_read": .5},
             "input_tokens_per_character": char_ratio, "reuse_free_rows": sum(reuse.values()),
             "new_free_rows": len(missing_free), "table_rows": inventory["table"], "code_rows": inventory["code"],
             "new_rows": len(missing_free) + inventory["table"] + inventory["code"],
             "proxies": proxies,
             "limitations": "Output/repair counts are local-model proxies, not measured Sol CoT usage. Cache savings assumed zero. Retries extra."}
    atomic_json(RUN / "cost_estimate.json", quote)
    manifest = {"protocol": "3.2-aligned", "parent_run": "cot-v32-190", "anchors": names,
                "reused_free_anchors": fixed, "new_free_anchors": added,
                "selection": "All 22 API190/CoT190 intersections plus 8 OOD additions, greedily balance budget and phase; SHA256 tie break sol30-20260924. No outcomes used.",
                "distributions": {"split": dict(collections.Counter(metadata[n]["stratum"][0] for n in names)),
                                  "phase": dict(collections.Counter(metadata[n]["phase"] for n in names))},
                "parent_manifest_sha256": sha(PARENT / "manifest.json"),
                "api_anchor_list_sha256": sha(ROOT / "bench/v2/api_subset_190.txt"),
                "questions": {n: sha(RUN / "questions" / n) for n in names},
                "question_digest": hashlib.sha256(b"".join(bytes.fromhex(sha(RUN / "questions" / n)) for n in names)).hexdigest(),
                "free_provenance": provenance, "reuse_free_rows": sum(reuse.values()), "counts": dict(inventory),
                "reuse_note": "Free prompts/budgets match frozen source; historical raw predates per-request prompt hashes. API provider provenance is retained, not a guarantee of an unchanged provider snapshot."}
    (RUN / "anchors.txt").write_text("\n".join(names) + "\n")
    (RUN / "new_free_anchors.txt").write_text("\n".join(sorted(added)) + "\n")
    atomic_json(manifest_path, manifest)
    atomic_json(RUN / "status.json", {"state": "prepared", "paid_calls_started": False, "counts": dict(inventory)})
    print(json.dumps({"run": str(RUN), "counts": dict(inventory), "reuse_free": sum(reuse.values()),
                      "new_free": len(missing_free), "new_rows": quote["new_rows"],
                      "cost_proxy_usd": {tag: round(sum(v["usd_all_input_cachewrite"] for v in fmts.values()), 2)
                                         for tag, fmts in proxies.items()}}, indent=2))


if __name__ == "__main__":
    main()
