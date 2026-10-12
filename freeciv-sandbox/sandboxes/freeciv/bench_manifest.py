"""Manifest for the v2 *benchmark* corpus (designed 2026-09-09, 09 §8.7).

Two sets, one collection, disjoint seeds from the training pool (>= 500000):

* basic  (`bench-basic`): S/M/L x 75, exactly the training pool's in-distribution
  generator (hidden parameters sampled the same way).  Measures basic game
  understanding.
* general (`bench-gen`): 250 worlds whose configuration is drawn *jointly* at random
  within feasible ranges -- ruleset, AI level, map size, civ count (200-700 tiles per
  civ), hidden parameters over the training range widened by 30%, 30% team worlds.
  Measures real generalisation; skill is regressed on the axes.

Every world is a branch world with 4 anchors (branch points) in [60, report_endturn];
the mainline runs on to max_horizon so every anchor has mainline truth at every
horizon.

    .venv/bin/python -m sandboxes.freeciv.bench_manifest --out data_v2/bench
"""
import argparse, collections, json, math, random
from pathlib import Path
from typing import Any

from .worlds import (ILLNESS_RULESETDIR, TIER_BUDGET, WorldCfg, _log_uniform,
                     apply_params, build_manifest, sample_params)

BASIC_SEED = 500000
GEN_SEED = 510000
AILEVELS = ("novice", "easy", "normal", "hard", "cheating")
TEAM_LAYOUTS_BY_N = {           # equal-size teams only; 5 and 7 civs never team up
    4: [[0, 0, 1, 1]],
    6: [[0, 0, 1, 1, 2, 2], [0, 0, 0, 1, 1, 1]],
    8: [[0, 0, 1, 1, 2, 2, 3, 3], [0, 0, 0, 0, 1, 1, 1, 1]],
    9: [[0, 0, 0, 1, 1, 1, 2, 2, 2]],
    10: [[0, 0, 0, 0, 0, 1, 1, 1, 1, 1], [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]],
}


def budget_for(tiles: int) -> tuple[int, int]:
    """Turn budget by map area, matching the training tiers (S 768 -> 400, M 2048 -> 800,
    L 3840 -> 1200); max_horizon = report_endturn + 300 as in TIER_BUDGET."""
    if tiles <= 1000:
        return TIER_BUDGET["S"]
    if tiles <= 2500:
        return TIER_BUDGET["M"]
    return TIER_BUDGET["L"]


def sample_params_wide(rng: random.Random) -> dict[str, Any]:
    """Training ranges widened by ~30% on both sides (all values legal for 3.1.6)."""
    return {
        "disasters": 0 if rng.random() < 0.15 else _log_uniform(rng, 5, 200),
        "foodbox": _log_uniform(rng, 42, 208),
        "sciencebox": _log_uniform(rng, 42, 208),
        "razechance": rng.randint(0, 100),
        "globalwarming_percent": 1 if rng.random() < 0.15 else _log_uniform(rng, 21, 520),
        "barbarians": "DISABLED" if rng.random() < 0.15 else rng.choice(["NORMAL", "FREQUENT", "HORDES"]),
        "onsetbarbs": rng.randint(28, 325),
        "civilwarsize": 999 if rng.random() < 0.2 else rng.randint(4, 26),
        "migration": rng.choice(["DISABLED", "ENABLED"]),
        "mgr_distance": rng.choice([2, 4, 6]),
        "mgr_worldchance": rng.choice([0, 10, 30, 40]),
        "mgr_nationchance": rng.choice([0, 50, 100]),
        "_illness": rng.choice([0, 10, 30]),
    }


def gen_world(seed: int, rng: random.Random) -> WorldCfg:
    ruleset = rng.choice(["civ2civ3", "classic"])
    ailevel = rng.choice(AILEVELS)
    while True:
        w = rng.randint(24, 64)
        h = int(round(w * 4 / 3))
        n = rng.randint(3, 10)
        if 200 <= w * h / n <= 700:
            break
    teams = None
    if rng.random() < 0.30 and n in TEAM_LAYOUTS_BY_N:
        teams = list(rng.choice(TEAM_LAYOUTS_BY_N[n]))
    re_, mh = budget_for(w * h)
    cfg = WorldCfg(tier="G", seed=seed, ruleset=ruleset, ailevel=ailevel, aifill=n,
                   xsize=w, ysize=h, report_endturn=re_, max_horizon=mh, teams=teams)
    drawn = sample_params_wide(rng)
    if ruleset != "civ2civ3":
        drawn["_illness"] = 10                   # classic has no illness variants
    apply_params(cfg, drawn)
    if ruleset != "civ2civ3":
        cfg.rulesetdir = ruleset
    cfg.split = "bench-gen"
    cfg.ood_axis = "gen"
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_basic", type=int, default=75, help="worlds per S/M/L tier")
    ap.add_argument("--n_gen", type=int, default=250)
    ap.add_argument("--branch_points", type=int, default=4)
    a = ap.parse_args()

    # ---- basic set: the training generator, fresh seeds
    m = build_manifest(3 * a.n_basic, 0, tiers=("S", "M", "L"), randomize_params=True,
                       start_seed=BASIC_SEED, branch_worlds=0)
    worlds: list[dict[str, Any]] = []
    for j in m["worlds"]:
        j = dict(j); j["split"] = "bench-basic"; j["branch"] = True; j["branch_points"] = a.branch_points
        j["ood_axis"] = "basic"
        worlds.append(j)
    # ---- generalisation set: joint random configuration
    rng = random.Random(GEN_SEED)
    for i in range(a.n_gen):
        cfg = gen_world(GEN_SEED + i, rng)
        j = cfg.to_json(); j["world_id"] = cfg.world_id(); j["branch"] = True; j["branch_points"] = a.branch_points
        worlds.append(j)
    ids = [w["world_id"] for w in worlds]
    assert len(ids) == len(set(ids)), "duplicate world_id"

    out = {
        "version": m.get("version"), "kind": "bench-v2", "created_turns": m.get("created_turns"),
        "counts": dict(collections.Counter(w["split"] for w in worlds)),
        "branch_worlds": sum(w["branch"] for w in worlds),
        "branch_by_tier": dict(collections.Counter(w["tier"] for w in worlds if w["branch"])),
        "worlds_by_tier": dict(collections.Counter(w["tier"] for w in worlds)),
        "ood_by_axis": dict(collections.Counter(w.get("ood_axis") or "" for w in worlds)),
        "design": {"basic": {"per_tier": a.n_basic, "tiers": ["S", "M", "L"], "seed_from": BASIC_SEED},
                   "general": {"n": a.n_gen, "seed_from": GEN_SEED, "map_width": [24, 64], "civs": [3, 10],
                               "tiles_per_civ": [200, 700], "ailevels": list(AILEVELS), "rulesets": ["civ2civ3", "classic"],
                               "team_share": 0.30, "params": "training range widened 30%"}},
        "worlds": worlds,
    }
    Path(a.out).mkdir(parents=True, exist_ok=True)
    p = Path(a.out) / "manifest.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    g = [w for w in worlds if w["split"] == "bench-gen"]
    print(f"wrote {p}: {len(worlds)} worlds {out['counts']}; by tier {out['worlds_by_tier']}")
    print("general set:", "rulesets", dict(collections.Counter(w['ruleset'] for w in g)),
          "| ailevel", dict(collections.Counter(w['ailevel'] for w in g)),
          "| teamed", sum(1 for w in g if w['teams']),
          "| civs", dict(sorted(collections.Counter(w['aifill'] for w in g).items())),
          "| budget", dict(collections.Counter(w['report_endturn'] for w in g)))
    tiles = [w["xsize"] * w["ysize"] / w["aifill"] for w in g]
    print(f"tiles/civ: min {min(tiles):.0f} median {sorted(tiles)[len(tiles)//2]:.0f} max {max(tiles):.0f}")


if __name__ == "__main__":
    main()
