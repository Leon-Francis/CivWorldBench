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
import math
import random
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
    # v2: team worlds.  Same map and civ count as M so the team effect is
    # attributable against the M baseline; `teams` on the cfg says who is with whom.
    "MT": Tier("MT", 32, 64, 6),
    # v2 benchmark generalisation set: size / civ count are explicit per world
    # (bench_manifest.gen_world); this entry only gives the tier a name.
    "G": Tier("G", 32, 64, 6),
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
TIER_POINTS: dict[str, int] = {"S": 4, "M": 6, "L": 6, "XL": 4, "MT": 6, "G": 4}

TIER_BUDGET: dict[str, tuple[int, int]] = {
    "S": (400, 700),
    "M": (800, 1100),
    "L": (1200, 1500),
    "XL": (400, 700),
    "MT": (800, 1100),
    "G": (800, 1100),     # placeholder; bench_manifest sets the budget per world by map area
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

# Explicitly created players for team worlds (aifill cannot assign teams).  Their
# nations are still drawn at random by the server; only the leader names are fixed.
TEAM_PLAYER_NAMES = ("Alpha", "Bravo", "Charlie", "Delta", "Echo", "Foxtrot",
                     "Golf", "Hotel", "India", "Juliet")
TEAM_LAYOUTS = {"2v2v2": [0, 0, 1, 1, 2, 2], "3v3": [0, 0, 0, 1, 1, 1]}

# Illness is a ruleset parameter, not a server setting, so it is selected by
# choosing one of three baked ruleset directories (see docker/Dockerfile).
ILLNESS_RULESETDIR = {0: "civ2civ3_ill0", 10: "civ2civ3", 30: "civ2civ3_ill30"}


def _log_uniform(rng: random.Random, lo: float, hi: float) -> int:
    return int(round(math.exp(rng.uniform(math.log(lo), math.log(hi)))))


def sample_params(rng: random.Random, *, extreme: bool = False) -> dict[str, Any]:
    """Hidden world parameters for one world (09 §4.4 / §8.4).

    Continuous draws on a log scale so the corpus covers the range densely; the
    benchmark bins them (3-4 ordered bins) for the probe task.  `extreme=True`
    is the held-out OOD segment (`ood-params`): values outside the in-distribution
    range, so OOD also tests unseen rule strengths.  Every value is a valid `set`
    argument (ranges checked against common/game.h of freeciv 3.1.6).
    """
    if not extreme:
        return {
            # 15% of worlds have no disasters at all; otherwise 5..150 (default 10)
            "disasters": 0 if rng.random() < 0.15 else _log_uniform(rng, 5, 150),
            "foodbox": _log_uniform(rng, 60, 160),          # default 100
            "sciencebox": _log_uniform(rng, 60, 160),       # default 100
            "razechance": rng.randint(0, 80),               # default 20
            "globalwarming_percent": (1 if rng.random() < 0.15
                                      else _log_uniform(rng, 30, 400)),  # default 100; min is 1
            "barbarians": ("DISABLED" if rng.random() < 0.15
                           else rng.choice(["NORMAL", "FREQUENT"])),
            "onsetbarbs": rng.randint(40, 250),             # default 60
            "civilwarsize": (999 if rng.random() < 0.2 else rng.randint(6, 20)),  # default 10
            # `migration` is the master switch and defaults to DISABLED -- without
            # it the two mgr_* chances are dead parameters.  Half the worlds have
            # it on; the chances only matter (and are only scored) in those.
            "migration": rng.choice(["DISABLED", "ENABLED"]),
            # default 0 = only between overlapping city work areas, which never
            # happens with AI spacing (P3 pilot: dead parameter) -- open it up
            "mgr_distance": rng.choice([2, 4]),
            "mgr_worldchance": rng.choice([0, 10, 30]),     # default 10
            "mgr_nationchance": rng.choice([0, 50, 100]),   # default 50
            "_illness": rng.choice([0, 10, 30]),            # -> rulesetdir, not a `set`
        }
    return {
        "disasters": rng.randint(150, 250),
        "foodbox": rng.choice([rng.randint(45, 60), rng.randint(160, 190)]),
        "sciencebox": rng.choice([rng.randint(45, 60), rng.randint(160, 190)]),
        "razechance": rng.randint(80, 100),
        "globalwarming_percent": rng.randint(400, 700),
        "barbarians": "HORDES",
        "onsetbarbs": rng.randint(250, 350),
        "civilwarsize": rng.randint(3, 5),
        "migration": "ENABLED",
        "mgr_distance": 6,
        "mgr_worldchance": rng.choice([0, 60]),
        "mgr_nationchance": rng.choice([0, 100]),
        "_illness": rng.choice([0, 30]),
    }


def apply_params(cfg: "WorldCfg", drawn: dict[str, Any]) -> None:
    """Put a `sample_params` draw onto a cfg: the illness level picks the ruleset
    directory, everything else is a server setting."""
    d = dict(drawn)
    ill = d.pop("_illness", 10)
    if cfg.ruleset == "civ2civ3":
        cfg.rulesetdir = ILLNESS_RULESETDIR[ill]
    cfg.params = d
    cfg.params["_illness"] = ill        # kept in the manifest as the true value


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
    # ---- v2 ----
    # ⚠️ v1 never passed the ruleset to the server: `ruleset` was only a label in
    # the world_id, and every "classic" world actually ran civ2civ3.  `rulesetdir`
    # is what is really sent (`set rulesetdir`).  It may be a variant directory
    # (civ2civ3_ill0 / civ2civ3_ill30) while `ruleset` keeps the family name.
    rulesetdir: str | None = None
    # Per-world randomised server settings (the hidden world parameters the
    # benchmark's task 2 asks a model to infer).  Every key is a `set`-able
    # server setting; they persist in the savegame's [settings] so branch arms
    # inherit them on reload.  Round-tripped through to_json/cfg_from_json --
    # the event-replay gate depends on reproducing them exactly.
    params: dict[str, Any] = field(default_factory=dict)
    # Team assignment per created player (e.g. [0,0,1,1,2,2] = 2v2v2).  None =
    # no teams (aifill creates the players).
    teams: list[int] | None = None
    # Which OOD axis this world varies (manifest bookkeeping only).
    ood_axis: str = ""
    # Pilot-only: appended to world_id so the same seed can run under several
    # parameter conditions side by side (P3).  Empty in the formal corpus --
    # world_id must not encode parameters there.
    tag: str = ""

    def resolved(self) -> tuple[int, int, int]:
        t = TIERS[self.tier]
        return (self.xsize or t.xsize, self.ysize or t.ysize,
                self.aifill if self.aifill is not None else t.aifill)

    def world_id(self, actual: tuple[int, int] | None = None) -> str:
        x, y, n = self.resolved()
        if actual:
            x, y = actual
        return (f"fc-{self.ruleset}-{self.tier}-{x}x{y}-p{n}-"
                f"{self.ailevel}-s{self.seed}" + (f"-{self.tag}" if self.tag else ""))

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
            # v2 (bench design, 2026-09-09): the mainline runs on to max_horizon so
            # every anchor <= report_endturn has mainline truth for every horizon
            # (the docstring in collect.py always promised this; settings() never
            # did it).  report_endturn is now the *anchor ceiling* only.
            "endturn": self.max_horizon,
            # v2: actually select the ruleset (see the field's note).
            "rulesetdir": self.rulesetdir or self.ruleset,
        })
        # v2: hidden world parameters.  Merged last so they override the
        # defaults above if a key collides (none does today).  Keys starting with
        # "_" are bookkeeping (e.g. the illness level, which selects rulesetdir).
        s.update({k: v for k, v in self.params.items() if not k.startswith("_")})
        return s

    def commands(self) -> list[str]:
        """`hard` / `cheating` / ... are commands, not settings.

        Team worlds: `aifill` does not assign teams and `team` only works before
        `start` on an existing player, so the players are created explicitly and
        teamed here.  `serv_script` emits `set` lines, then these, then `start`,
        which is exactly the required order.  The skill-level command goes first
        so the created players inherit it.
        """
        cmds = [self.ailevel]
        if self.teams:
            for i, _ in enumerate(self.teams):
                cmds.append(f"create {TEAM_PLAYER_NAMES[i]}")
            for i, t in enumerate(self.teams):
                cmds.append(f"team {TEAM_PLAYER_NAMES[i]} {t}")
        return cmds

    def to_json(self) -> dict[str, Any]:
        x, y, n = self.resolved()
        return {"tier": self.tier, "seed": self.seed, "ruleset": self.ruleset,
                "ailevel": self.ailevel, "xsize": x, "ysize": y, "aifill": n,
                "report_endturn": self.report_endturn,
                "max_horizon": self.max_horizon,
                "save_every": self.save_every, "split": self.split,
                # v2 -- everything the server run depends on must round-trip.
                "rulesetdir": self.rulesetdir or self.ruleset,
                "params": dict(self.params),
                "teams": list(self.teams) if self.teams else None,
                "ood_axis": self.ood_axis, "tag": self.tag}


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


def build_manifest(n_train_pool: int, n_ood: int, *, start_seed: int = 200000,
                   branch_worlds: int = 0, report_endturn: int = 0,
                   max_horizon: int = 0,
                   branch_tiers: tuple[str, ...] | None = None,
                   branch_per_tier: int | dict[str, int] = 0,
                   n_team: int = 0, team_branch: int = 0,
                   ood_batches: dict[str, int] | None = None,
                   ood_branch: dict[str, int] | None = None,
                   randomize_params: bool = True,
                   tiers: tuple[str, ...] | None = None) -> dict[str, Any]:
    """v2 world set.

    * `n_train_pool` worlds over S/M/L, split 80/10/10 by id hash; every world
      draws its hidden parameters (`sample_params`) from a per-world RNG seeded
      off its seed, so the manifest is reproducible.
    * `n_ood` worlds in **single-axis** batches against the M baseline (草案 §2.2):
      ruleset / mapsize / civcount / ailevel change one thing each, `allvary`
      changes all four (= v1's ood), `params` uses the extreme parameter segment.
      Counts come from `ood_batches`; defaults sum to n_ood proportionally.
    * `n_team` team worlds (tier MT, 2v2v2 and 3v3 alternating), in-distribution split.
    * Branch treatment per tier / per ood batch / for team worlds, chosen by id hash.
    """
    worlds: list[WorldCfg] = []
    seed = start_seed

    def _draw(cfg: WorldCfg, extreme: bool = False) -> None:
        if randomize_params:
            apply_params(cfg, sample_params(random.Random(cfg.seed * 7919 + 17),
                                            extreme=extreme))
        else:
            cfg.rulesetdir = cfg.ruleset

    cycle = tuple(tiers) if tiers else TRAIN_TIERS
    for i in range(n_train_pool):
        tier = cycle[i % len(cycle)]
        re_, mh = TIER_BUDGET[tier]
        cfg = WorldCfg(tier=tier, seed=seed,
                       report_endturn=report_endturn or re_,
                       max_horizon=max_horizon or mh)
        _draw(cfg)
        cfg.split = assign_split(cfg.world_id())
        worlds.append(cfg)
        seed += 1

    for i in range(n_team):
        layout = "2v2v2" if i % 2 == 0 else "3v3"
        re_, mh = TIER_BUDGET["MT"]
        cfg = WorldCfg(tier="MT", seed=seed, report_endturn=report_endturn or re_,
                       max_horizon=max_horizon or mh, teams=list(TEAM_LAYOUTS[layout]))
        _draw(cfg)
        cfg.split = assign_split(cfg.world_id())
        cfg.ood_axis = ""
        worlds.append(cfg)
        seed += 1

    if n_ood:
        default = {"ruleset": 25, "mapsize": 25, "civcount": 25, "ailevel": 25,
                   "allvary": 75, "params": 25}
        batches = dict(ood_batches or default)
        tot = sum(batches.values())
        scale = n_ood / tot if tot else 0
        batches = {k: int(round(v * scale)) for k, v in batches.items()}
        m_re, m_mh = TIER_BUDGET["M"]
        for axis, k in batches.items():
            for j in range(k):
                kw: dict[str, Any] = dict(tier="M", seed=seed, split="test-ood",
                                          report_endturn=report_endturn or m_re,
                                          max_horizon=max_horizon or m_mh)
                extreme = False
                if axis == "ruleset":
                    kw["ruleset"] = OOD_RULESET
                elif axis == "mapsize":
                    kw.update(xsize=TIERS["XL"].xsize, ysize=TIERS["XL"].ysize)
                elif axis == "civcount":
                    kw["aifill"] = TIERS["XL"].aifill
                elif axis == "ailevel":
                    kw["ailevel"] = OOD_AILEVELS[j % len(OOD_AILEVELS)]
                elif axis == "allvary":
                    xl_re, xl_mh = TIER_BUDGET["XL"]
                    kw.update(tier="XL", ruleset=OOD_RULESET,
                              ailevel=OOD_AILEVELS[j % len(OOD_AILEVELS)],
                              report_endturn=report_endturn or xl_re,
                              max_horizon=max_horizon or xl_mh)
                elif axis == "params":
                    extreme = True
                cfg = WorldCfg(**kw)
                cfg.ood_axis = axis
                _draw(cfg, extreme=extreme)
                worlds.append(cfg)
                seed += 1

    def _h(c: WorldCfg) -> str:
        return hashlib.sha1(c.world_id().encode()).hexdigest()

    if branch_per_tier and branch_tiers:
        per = (branch_per_tier if isinstance(branch_per_tier, dict)
               else {t: branch_per_tier for t in branch_tiers})
        for tier in branch_tiers:
            k = int(per.get(tier, 0))
            if k <= 0:
                continue
            pool = sorted((c for c in worlds if c.tier == tier and not c.ood_axis
                           and c.tier != "MT"), key=_h)
            for c in pool[:k]:
                c.extra["branch"] = True
    elif branch_worlds:
        for c in sorted((c for c in worlds if not c.ood_axis), key=_h)[:branch_worlds]:
            c.extra["branch"] = True
    if team_branch:
        for c in sorted((c for c in worlds if c.tier == "MT"), key=_h)[:team_branch]:
            c.extra["branch"] = True
    for axis, k in (ood_branch or {}).items():
        for c in sorted((c for c in worlds if c.ood_axis == axis), key=_h)[:k]:
            c.extra["branch"] = True

    return {
        "version": 2,
        "created_turns": {"report_endturn": report_endturn,
                          "max_horizon": max_horizon},
        "counts": {s: sum(1 for c in worlds if c.split == s) for s in SPLITS},
        "branch_worlds": sum(1 for c in worlds if c.extra.get("branch")),
        "branch_by_tier": {t: sum(1 for c in worlds
                                  if c.extra.get("branch") and c.tier == t)
                           for t in sorted({c.tier for c in worlds})},
        "worlds_by_tier": {t: sum(1 for c in worlds if c.tier == t)
                           for t in sorted({c.tier for c in worlds})},
        "ood_by_axis": {a: sum(1 for c in worlds if c.ood_axis == a)
                        for a in sorted({c.ood_axis for c in worlds if c.ood_axis})},
        "worlds": [{**c.to_json(), "world_id": c.world_id(),
                    "branch": bool(c.extra.get("branch")),
                    "branch_points": TIER_POINTS[c.tier]} for c in worlds],
    }


def cfg_from_json(j: dict[str, Any]) -> WorldCfg:
    c = WorldCfg(tier=j["tier"], seed=j["seed"], ruleset=j["ruleset"],
                 ailevel=j["ailevel"], aifill=j["aifill"],
                 report_endturn=j["report_endturn"],
                 max_horizon=j["max_horizon"], save_every=j["save_every"],
                 split=j["split"], xsize=j["xsize"], ysize=j["ysize"],
                 rulesetdir=j.get("rulesetdir"), params=dict(j.get("params") or {}),
                 teams=list(j["teams"]) if j.get("teams") else None,
                 ood_axis=j.get("ood_axis", ""), tag=j.get("tag", ""))
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
                       max_horizon=turns,
                       teams=list(TEAM_LAYOUTS["2v2v2"]) if name == "MT" else None)
        # v2: exercise the parameter path too, with a fixed draw per tier.
        apply_params(cfg, sample_params(random.Random(4242)))
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
        # v2 checks: the ruleset the server really loaded, the randomised
        # settings echoed back from [settings], and team membership.
        st = sg.table("settings", "set")
        echoed = {r[0]: r[1] for r in st.rows} if st is not None else {}
        # Booleans are sent as ENABLED/DISABLED but saved as TRUE/FALSE.
        _norm = {"ENABLED": "TRUE", "DISABLED": "FALSE"}
        params_ok = all(_norm.get(str(v), str(v)) == _norm.get(str(echoed.get(k)), str(echoed.get(k)))
                        for k, v in cfg.params.items() if not k.startswith("_"))
        row.update(rulesetdir=sg.ruleset, rulesetdir_ok=(sg.ruleset == (cfg.rulesetdir or cfg.ruleset)),
                   params_ok=params_ok, params=dict(cfg.params),
                   teams=[str(sg.get(sec, "team_no", "")) for sec in sg.players()])
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
              f"{'civs':<5} {'traits vary':<12} {'ruleset':<16} {'params':<7} {'wall':<7} world_id")
        for r in rows:
            if not r.get("ok"):
                print(f"{r['tier']:<5} {str(r['requested']):<12} {'FAIL':<12} "
                      f"{'-':<6} {'-':<5} {'-':<12} {r['wall_s']:<7} {r['detail'][:50]}")
                continue
            print(f"{r['tier']:<5} {str(r['requested']):<12} {str(r['actual']):<12} "
                  f"{'ok' if r['size_ok'] else 'MISMATCH':<6} {r['players']:<5} "
                  f"{str(r['traits_varied']):<12} "
                  f"{(r['rulesetdir'] + ('' if r['rulesetdir_ok'] else '!!')):<16} "
                  f"{('ok' if r['params_ok'] else 'MISMATCH'):<7} {r['wall_s']:<7} {r['world_id']}")
            if r.get("teams") and any(t not in ("", "0") for t in r["teams"]):
                print(f"      teams: {r['teams']}")
        for r in rows:
            if r.get("ok") and r.get("traits"):
                print(f"  {r['tier']} traits {r['trait_vector']}:")
                for n, v in list(r["traits"].items())[:4]:
                    print(f"      {n:<20} {v}")
        bad = [r for r in rows if not r.get("ok") or not r.get("size_ok")
               or not r.get("traits_varied") or not r.get("rulesetdir_ok")
               or not r.get("params_ok")]
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
