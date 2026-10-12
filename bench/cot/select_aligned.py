"""Result-blind anchor sampling and a separate historical development set."""
from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--development", action="store_true")
    ap.add_argument("--seed", type=int, default=20260923)
    a = ap.parse_args()
    registry = ROOT / "bench/v2/anchors.json"
    anchors = json.loads(registry.read_text())["anchors"]
    src = ROOT / "bench/runs/v3-bench-br/questions"
    rng = random.Random(a.seed)
    selected = []
    if a.development:
        candidates = (ROOT / "bench/runs/cot-v31-main/anchors.txt").read_text().splitlines()
        groups = collections.defaultdict(list)
        for name in candidates:
            groups[name.split("-")[2]].append(name)
        for tier in sorted(groups):
            rng.shuffle(groups[tier])
            selected.extend({"file": name, "development": True} for name in groups[tier][:2])
        if len(selected) != 8:
            raise ValueError(f"Expected 8 development anchors, found {len(selected)}")
    else:
        strata = collections.defaultdict(list)
        for world, turns in sorted(anchors.items()):
            with gzip.open(src / f"{world}_{turns[0]}.jsonl.gz", "rt") as fh:
                meta = json.loads(next(fh))
            strata[(meta["set"], meta["tier"], meta["budget"])].append(world)
        for split, target in (("bench-basic", 90), ("bench-gen", 100)):
            keys = sorted(k for k in strata if k[0] == split)
            total = sum(len(strata[k]) for k in keys)
            quotas = {k: target * len(strata[k]) // total for k in keys}
            ranked = sorted(keys, key=lambda k: (-(target * len(strata[k]) % total), k))
            for k in ranked[:target - sum(quotas.values())]:
                quotas[k] += 1
            for key in keys:
                worlds = list(strata[key]); rng.shuffle(worlds)
                phases = list(range(4)); rng.shuffle(phases)
                for i, world in enumerate(worlds[:quotas[key]]):
                    turns = anchors[world]
                    if len(turns) != 4:
                        raise ValueError(f"Unexpected phase count for {world}")
                    phase = phases[i % 4]
                    selected.append({"file": f"{world}_{turns[phase]}.jsonl.gz", "world": world,
                                     "phase": phase, "stratum": list(key),
                                     "inclusion_probability": quotas[key] / len(worlds) / 4})
        assert len(selected) == 190 and len({r["world"] for r in selected}) == 190
    run = ROOT / "bench/runs" / a.run
    run.mkdir(parents=True, exist_ok=True)
    manifest = {"seed": a.seed, "development": a.development,
                "registry_sha256": hashlib.sha256(registry.read_bytes()).hexdigest(), "anchors": selected}
    path = run / "selection.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError("Refusing to replace a different anchor selection")
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    (run / "anchors.txt").write_text("\n".join(sorted(r["file"] for r in selected)) + "\n")
    print(f"Selected {len(selected)} anchors -> {run}")


if __name__ == "__main__":
    main()
