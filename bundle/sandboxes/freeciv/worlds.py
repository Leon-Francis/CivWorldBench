"""World configurations, ids, and the train/valid/test split.

Three things live here because they all have to agree with each other: what a
world is configured as, what it is called, and which split it lands in.

**Map size has to be stated explicitly.**  The collector used to `set size 4`,
and the resulting world came out 32x64 = 2048 tiles, i.e. `size` 2.  Reading the
savegame's own `[settings]` explains it: `mapsize` defaults to `PLAYER` under
civ2civ3, so the map is sized from tiles-per-player and `size` is ignored
outright.  Every tier therefore sets `mapsize XYSIZE` with explicit `xsize` /
`ysize`, and `world_id` carries the dimensions **as read back from the produced
savegame**, not as requested -- otherwise a silently-resized world would be
mislabelled in the corpus.

**Tiers exist for world diversity, not to control report size.**  The corpus
stores structured state and renders reports at sample-build time, so a long
world does not produce an untrainable example -- the sample builder chooses which
`(t, delta)` pairs to render and under what token ceiling.  What the tiers buy is
genuinely different dynamics: a 3-civ world on a small map is not a small version
of an 8-civ world, it is a different game.

**`traitdistribution EVEN` is on for every world.**  Under the `FIXED` default
every civ gets trait 50/50/50/50, which makes the strongest probe target in the
savegame a constant -- nothing to infer.  ⚠️ It also changes the behaviour
distribution, so pre-EVEN data (the `w888` pilot) cannot be mixed with this
corpus.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .runner import BASE_SETTINGS, run
from .savegame import Savegame


@dataclass(frozen=True)
class Tier:
    name: str
    xsize: int
    ysize: int
    aifill: int


TIERS: dict[str, Tier] = {
    "S": Tier("S", 24, 32, 3),
    "M": Tier("M", 32, 64, 6),      # the shape every measurement so far was taken on
    "L": Tier("L", 48, 80, 8),
    "XL": Tier("XL", 64, 96, 10),   # held out for the OOD split
}

# (report_endturn, max_horizon) per tier.  Set by how long a world stays alive,
# not by what it costs -- cost is explicitly not a constraint here.
#
# Measured to T800: S stops producing events around T360 (tech tree finished,
# two civs left), so 400 is where its content actually ends.  M never went quiet.
# L never went quiet either and its living-civ count *grew* from 8 to 11 as civil
# wars and barbarian leaders spawned new players, so it gets the longest budget.
#
# ⚠️ L at 1200 is beyond the measured range (the liveliness run stopped at T801).
# Whether it stays interesting from T800 to T1200 is an assumption, not a reading.
# Branch points per tier.  Not a constant, because the budgets span 400 to 1200
# turns and four points across a 1200-turn world is very sparse -- but more
# importantly because **operator applicability moves with the game phase**: fog
# only bites while a civ is still exploring, barbarians only after the ruleset's
# onset turn, give-tech only while techs remain unresearched.  Spreading the
# points lets every operator collect replicates inside its own regime and record
# NA outside it.
TIER_POINTS: dict[str, int] = {"S": 4, "M": 6, "L": 6, "XL": 4}

TIER_BUDGET: dict[str, tuple[int, int]] = {
    "S": (400, 700),
    "M": (800, 1100),
    "L": (1200, 1500),
    "XL": (400, 700),
}

# Axes the in-distribution grid is built from.  Anything NOT listed here is
# available to test-ood, which is the point: an OOD world differs from training
# in a way no amount of seed variation can produce.
TRAIN_RULESET = "civ2civ3"
TRAIN_TIERS = ("S", "M", "L")
TRAIN_AILEVEL = "hard"

OOD_RULESET = "classic"
OOD_TIERS = ("XL",)
OOD_AILEVELS = ("cheating", "novice")

SPLITS = ("train", "valid", "test-iid", "test-ood")


@dataclass
class WorldCfg:
    tier: str
    seed: int
    ruleset: str = TRAIN_RULESET
    ailevel: str = TRAIN_AILEVEL
    aifill: int | None = None
    report_endturn: int = 250
    max_horizon: int = 480
    # ★ One checkpoint per turn.
    #
    # At every-5-turns, the `actions` of the four intervening turns are gone for
    # good -- and unit orders and production choices change every turn.  A stride-5
    # training example would then be conditioned on a fifth of the decisions that
    # actually drove the interval, so the corpus would not be information-complete
    # at any stride.  Per-turn checkpoints make every stride reconstructible.
    save_every: int = 1
    split: str = "train"
    xsize: int | None = None          # filled from the tier unless overridden
    ysize: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def resolved(self) -> tuple[int, int, int]:
        t = TIERS[self.tier]
        return (self.xsize or t.xsize, self.ysize or t.ysize,
                self.aifill if self.aifill is not None else t.aifill)

    def world_id(self, actual: tuple[int, int] | None = None) -> str:
        x, y, n = self.resolved()
        if actual:
            x, y = actual
        return (f"fc-{self.ruleset}-{self.tier}-{x}x{y}-p{n}-"
                f"{self.ailevel}-s{self.seed}")

    def settings(self) -> dict[str, Any]:
        """Server settings for a fresh game.

        `mapsize XYSIZE` must come before `xsize` / `ysize` or the dimensions are
        overwritten by the player-count sizing.

        `endturn` is `report_endturn`.  It used to be `max_horizon` so the
        mainline would supply the factual arm for free -- but a reload does not
        reproduce the uninterrupted trajectory (measured: it diverges within five
        turns of the fork), so each branch point runs its own factual arm and the
        mainline has no reason to outrun the reported window.  `max_horizon` now
        only bounds how far a branch arm may roll.
        """
        x, y, n = self.resolved()
        s = dict(BASE_SETTINGS)
        s.update({
            "mapseed": self.seed,
            "gameseed": self.seed,
            "mapsize": "XYSIZE",
            "xsize": x,
            "ysize": y,
            "aifill": n,
            "traitdistribution": "EVEN",
            "saveturns": self.save_every,
            "endturn": self.report_endturn,
        })
        s.update(self.extra)
        return s

    def commands(self) -> list[str]:
        """`hard` / `cheating` / ... are commands, not settings."""
        return [self.ailevel]

    def to_json(self) -> dict[str, Any]:
        x, y, n = self.resolved()
        return {"tier": self.tier, "seed": self.seed, "ruleset": self.ruleset,
                "ailevel": self.ailevel, "xsize": x, "ysize": y, "aifill": n,
                "report_endturn": self.report_endturn,
                "max_horizon": self.max_horizon,
                "save_every": self.save_every, "split": self.split}


# --------------------------------------------------------------------------
# the manifest
# --------------------------------------------------------------------------

def assign_split(world_id: str, weights=((("train"), 80), ("valid", 10),
                                         ("test-iid", 10))) -> str:
    """Deterministic split from the id, so re-running the generator is stable.

    Splitting by world rather than by checkpoint is not a style choice: two
    checkpoints of the same game share the map, the civs and most of the state,
    so putting T60 in train and T65 in valid leaks almost everything.
    """
    h = int(hashlib.sha1(world_id.encode()).hexdigest()[:8], 16) % 100
    acc = 0
    for name, w in weights:
        acc += w
        if h < acc:
            return name
    return weights[-1][0]


def build_manifest(n_train_pool: int, n_ood: int, *, start_seed: int = 100000,
                   branch_worlds: int = 0, report_endturn: int = 0,
                   max_horizon: int = 0,
                   branch_tiers: tuple[str, ...] | None = None,
                   branch_per_tier: int | dict[str, int] = 0) -> dict[str, Any]:
    """`n_train_pool` worlds are drawn over the in-distribution grid and split
    deterministically into train / valid / test-iid; `n_ood` more use held-out
    configurations and are all test-ood.

    Branch treatment is where ~90% of the compute goes, and its cost differs by
    an order of magnitude across tiers (0.228 s/turn on S against 2.795 on L), so
    `branch_tiers` / `branch_per_tier` let it be allocated per tier rather than
    sampled uniformly.  Sampling uniformly would quietly hand most of the budget
    to whichever expensive tier the hash happened to pick.

    Within a tier the choice is still by id hash, so it is deterministic and
    unbiased with respect to seed.
    """
    worlds: list[WorldCfg] = []
    seed = start_seed
    for i in range(n_train_pool):
        tier = TRAIN_TIERS[i % len(TRAIN_TIERS)]
        # Explicit arguments win; otherwise each tier gets its own measured budget
        # rather than one number applied to worlds of very different lifespans.
        re_, mh = TIER_BUDGET[tier]
        cfg = WorldCfg(tier=tier, seed=seed,
                       report_endturn=report_endturn or re_,
                       max_horizon=max_horizon or mh)
        cfg.split = assign_split(cfg.world_id())
        worlds.append(cfg)
        seed += 1
    for i in range(n_ood):
        ood_tier = OOD_TIERS[i % len(OOD_TIERS)]
        re_, mh = TIER_BUDGET[ood_tier]
        cfg = WorldCfg(tier=ood_tier, seed=seed,
                       ruleset=OOD_RULESET,
                       ailevel=OOD_AILEVELS[i % len(OOD_AILEVELS)],
                       report_endturn=report_endturn or re_,
                       max_horizon=max_horizon or mh,
                       split="test-ood")
        worlds.append(cfg)
        seed += 1

    # Branch treatment goes to a subset, chosen deterministically within each
    # tier and spread across splits so that counterfactual data exists on both
    # sides of the split rather than only in train.
    def _h(c: WorldCfg) -> str:
        return hashlib.sha1(c.world_id().encode()).hexdigest()

    if branch_per_tier and branch_tiers:
        # `branch_per_tier` may be one number for every tier, or a per-tier dict.
        # Per-tier is what the budget actually needs: an L arm costs ~14x an S arm,
        # so "the same count everywhere" is not the same spend everywhere.
        per = (branch_per_tier if isinstance(branch_per_tier, dict)
               else {t: branch_per_tier for t in branch_tiers})
        for tier in branch_tiers:
            k = int(per.get(tier, 0))
            if k <= 0:
                continue
            pool = sorted((c for c in worlds if c.tier == tier), key=_h)
            for c in pool[:k]:
                c.extra["branch"] = True
    else:
        for c in sorted(worlds, key=_h)[:branch_worlds]:
            c.extra["branch"] = True

    return {
        "created_turns": {"report_endturn": report_endturn,
                          "max_horizon": max_horizon},
        "counts": {s: sum(1 for c in worlds if c.split == s) for s in SPLITS},
        "branch_worlds": sum(1 for c in worlds if c.extra.get("branch")),
        "branch_by_tier": {t: sum(1 for c in worlds
                                  if c.extra.get("branch") and c.tier == t)
                           for t in sorted({c.tier for c in worlds})},
        "worlds_by_tier": {t: sum(1 for c in worlds if c.tier == t)
                           for t in sorted({c.tier for c in worlds})},
        "worlds": [{**c.to_json(), "world_id": c.world_id(),
                    "branch": bool(c.extra.get("branch")),
                    "branch_points": TIER_POINTS[c.tier]} for c in worlds],
    }


def cfg_from_json(j: dict[str, Any]) -> WorldCfg:
    c = WorldCfg(tier=j["tier"], seed=j["seed"], ruleset=j["ruleset"],
                 ailevel=j["ailevel"], aifill=j["aifill"],
                 report_endturn=j["report_endturn"],
                 max_horizon=j["max_horizon"], save_every=j["save_every"],
                 split=j["split"], xsize=j["xsize"], ysize=j["ysize"])
    if j.get("branch"):
        c.extra["branch"] = True
    return c


# --------------------------------------------------------------------------
# P-1 / P-2
# --------------------------------------------------------------------------

def preflight(tiers: list[str], root: Path, turns: int = 3) -> list[dict]:
    """Start each tier and read back what the server actually built.

    Checks the two things that would silently corrupt the whole corpus:
    the map is the size we asked for, and `traitdistribution EVEN` really
    randomises the personality vector instead of leaving it at 50/50/50/50.
    """
    out = []
    for name in tiers:
        cfg = WorldCfg(tier=name, seed=4242, report_endturn=turns,
                       max_horizon=turns)
        wd = root / f"pf_{name}"
        shutil.rmtree(wd, ignore_errors=True)
        t0 = time.time()
        r = run(wd, {**cfg.settings(), "saveturns": 1, "endturn": turns},
                commands=cfg.commands(), timeout=900)
        row: dict[str, Any] = {"tier": name, "wall_s": round(time.time() - t0, 1),
                               "requested": cfg.resolved()[:2]}
        if not r.ok:
            row.update(ok=False, detail=r.message)
            out.append(row)
            continue
        sg = Savegame.load(r.final)
        x, y = int(sg.get("map", "xsize", -1)), int(sg.get("map", "ysize", -1))
        if x < 0:                       # older layout: dimensions only in settings
            x = len(str(sg.get("map", "t0000", "")))
            y = sum(1 for k in sg.sections.get("map", {})
                    if k.startswith("t") and k[1:].isdigit())
        tv = [str(v) for v in sg.get("savefile", "trait_vector", [])]
        traits = {}
        for sec in sg.players():
            t = sg.table(sec, "trait")
            if t is None:
                continue
            traits[str(sg.get(sec, "name", sec))] = [
                int(t.rows[i][0]) + int(t.rows[i][1])
                for i in range(min(len(tv), len(t.rows)))]
        vals = [v for vs in traits.values() for v in vs]
        row.update(ok=True, actual=(x, y),
                   size_ok=(x, y) == cfg.resolved()[:2],
                   players=len(sg.players()),
                   trait_vector=tv, traits=traits,
                   traits_varied=bool(vals) and len(set(vals)) > 1,
                   world_id=cfg.world_id((x, y)))
        out.append(row)
        shutil.rmtree(wd, ignore_errors=True)
    return out


# --------------------------------------------------------------------------
# P-0b
# --------------------------------------------------------------------------

def liveliness(tier: str, seed: int, turns: int, root: Path,
               save_every: int = 10) -> dict[str, Any]:
    """Run one world long and report how alive it stays, turn by turn.

    This is what sets the turn budget.  A world that has lost four of six civs
    and stopped producing events is spending compute to generate nothing, and
    the point where that happens is a measurement, not a guess.  Wall-clock per
    block is recorded in the same pass because cost per turn rises with unit
    count -- 250 turns at 142 s does not extrapolate.
    """
    from .events import diff_events
    from .state import extract

    cfg = WorldCfg(tier=tier, seed=seed, report_endturn=turns, max_horizon=turns,
                   save_every=save_every)
    wd = root / f"live_{tier}_{seed}"
    shutil.rmtree(wd, ignore_errors=True)
    t0 = time.time()
    r = run(wd, cfg.settings(), commands=cfg.commands(), timeout=14400)
    wall = time.time() - t0
    if not r.ok:
        return {"tier": tier, "seed": seed, "ok": False, "detail": r.message,
                "wall_s": round(wall, 1)}

    rows, prev = [], None
    for p in r.saves:
        sg = Savegame.load(p)
        ws = extract(sg, "live")
        ev = diff_events(prev, ws) if prev else []
        alive = sum(1 for q in ws.players if q.is_alive)
        rows.append({
            "t": sg.turn, "alive": alive,
            "cities": sum(len(q.cities) for q in ws.players),
            "units": sum(len(q.units) for q in ws.players),
            "techs": sum(len(q.techs) for q in ws.players),
            "events": len(ev),
            "score_spread": (max(q.score for q in ws.players)
                             - min(q.score for q in ws.players)) if ws.players else 0,
        })
        prev = ws
    shutil.rmtree(wd, ignore_errors=True)
    return {"tier": tier, "seed": seed, "ok": True, "wall_s": round(wall, 1),
            "turns": r.turn_reached,
            "sec_per_turn": round(wall / max(1, r.turn_reached), 3), "rows": rows}


# --------------------------------------------------------------------------
# cli
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preflight", action="store_true", help="P-1 + P-2")
    ap.add_argument("--liveliness", action="store_true", help="P-0b")
    ap.add_argument("--manifest", help="write a manifest to this path")
    ap.add_argument("--tiers", default="S,M,L,XL")
    ap.add_argument("--worlds", type=int, default=3)
    ap.add_argument("--turns", type=int, default=800)
    ap.add_argument("--branch_worlds", type=int, default=0)
    ap.add_argument("--ood", type=int, default=0)
    ap.add_argument("--report_endturn", type=int, default=250)
    ap.add_argument("--max_horizon", type=int, default=480)
    ap.add_argument("--root",
                    default="data_v2/worlds")
    args = ap.parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)

    if args.preflight:
        rows = preflight([t for t in args.tiers.split(",") if t], root)
        print(f"{'tier':<5} {'requested':<12} {'actual':<12} {'size':<6} "
              f"{'civs':<5} {'traits vary':<12} {'wall':<7} world_id")
        for r in rows:
            if not r.get("ok"):
                print(f"{r['tier']:<5} {str(r['requested']):<12} {'FAIL':<12} "
                      f"{'-':<6} {'-':<5} {'-':<12} {r['wall_s']:<7} {r['detail'][:50]}")
                continue
            print(f"{r['tier']:<5} {str(r['requested']):<12} {str(r['actual']):<12} "
                  f"{'ok' if r['size_ok'] else 'MISMATCH':<6} {r['players']:<5} "
                  f"{str(r['traits_varied']):<12} {r['wall_s']:<7} {r['world_id']}")
        for r in rows:
            if r.get("ok") and r.get("traits"):
                print(f"  {r['tier']} traits {r['trait_vector']}:")
                for n, v in list(r["traits"].items())[:4]:
                    print(f"      {n:<20} {v}")
        bad = [r for r in rows if not r.get("ok") or not r.get("size_ok")
               or not r.get("traits_varied")]
        print(f"\nP-1/P-2: {'PASS' if not bad else 'FAIL on ' + ','.join(r['tier'] for r in bad)}")
        return

    if args.liveliness:
        tiers = [t for t in args.tiers.split(",") if t]
        for i in range(args.worlds):
            tier = tiers[i % len(tiers)]
            res = liveliness(tier, 7000 + i, args.turns, root)
            if not res.get("ok"):
                print(f"[{tier} s{7000+i}] FAILED: {res['detail']}")
                continue
            print(f"\n[{tier} s{7000+i}] reached T{res['turns']} in "
                  f"{res['wall_s']}s ({res['sec_per_turn']}s/turn)")
            print(f"  {'t':>5} {'alive':>5} {'cities':>6} {'units':>6} "
                  f"{'techs':>6} {'events':>6} {'spread':>7}")
            for row in res["rows"][::max(1, len(res["rows"]) // 16)]:
                print(f"  {row['t']:>5} {row['alive']:>5} {row['cities']:>6} "
                      f"{row['units']:>6} {row['techs']:>6} {row['events']:>6} "
                      f"{row['score_spread']:>7}")
        return

    if args.manifest:
        m = build_manifest(args.worlds, args.ood, branch_worlds=args.branch_worlds,
                           report_endturn=args.report_endturn,
                           max_horizon=args.max_horizon)
        Path(args.manifest).parent.mkdir(parents=True, exist_ok=True)
        Path(args.manifest).write_text(json.dumps(m, ensure_ascii=False, indent=1))
        print(f"wrote {args.manifest}: {m['counts']}, "
              f"{m['branch_worlds']} branch worlds")
        return

    ap.error("pass --preflight, --liveliness or --manifest")


if __name__ == "__main__":
    main()
