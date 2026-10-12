"""The counterfactual operator library: 21 operators, each with a matched placebo.

Two things this module is built around, both of them measured rather than assumed.

**1. Change a mechanism, not a stock.**  `+500 gold` was the pilot intervention
and its effect profile shows why it is the weakest kind: H30 diff 487, H150 diff
**10**.  The AI spends the money and the perturbation is absorbed.  Compare
`do(trait)`, which changed how the civ decides: over 120 turns it went 4 vs 8
cities, 80 vs 123 score, and the gap *widened* monotonically.  The library is
therefore layered, and the layer is recorded on every arm so effect profiles can
be compared within a layer:

    L1 stock       1 hop, absorbed within a few turns.  Kept as a lower bound.
    L2 constraint  2-3 hops, changes the feasible set.
    L3 endowment   many hops, permanent, cannot be absorbed.
    L4 mechanism   changes the decision function itself; effect accumulates.

**2. Every operator needs a same-type placebo, not an empty one.**  In a
deterministic world an empty intervention is byte-identical to factual, so it
cannot measure the perturbation of *making a change of this kind at all*.
§2.5.5b measured the difference: giving Pottery -- a tech that unlocks no
government -- moved T200 score from 82 to 74, almost exactly where the real
`give Monarchy` arm landed (73).  Reading the real arm alone would have credited
that decline to the government change.  It was path noise.

So each operator declares:

  `mutate` or `lua`   how the intervention is applied
  `placebo`           a same-type change that does NOT engage the mechanism
  `mechanism_var`     the variable the operator is supposed to move -- the
                      placebo must leave it alone, which is what P-5 checks
  `verify`            did the effect survive N turns (gate 4)
  `expected_path`     the transduction chain, written down BEFORE measuring, so
                      "operator did nothing" can be told apart from "operator
                      worked by a different route" -- the second is the more
                      valuable finding
  `horizon_band`      where this layer's effect is attributable

⚠️ Eight operators from the original design are absent because they were measured
to be dead, and removing them is the point of having gate 4:

    gov            five paths all failed; the Lua assignment mutates a copy
    wondercity     the AI recomputes it every turn
    dc-poison      poisoning all 39 memories changed no unit position at all
    civil_war      the C layer rejects the call, precondition unknown
    rates          AI recomputes rates.* each turn -- the arm came out
                   byte-identical to factual, so the write is wholly inert
    goal           same, for research.goal_name
    urgency        same, for city.ai.urgency
    climate        edit.climate_change moved none of the six warming/cooling
                   fields; five call signatures were tried

Note the pattern in five of those eight: **any field the AI recomputes at the
start of its turn is a dead handle**, no matter that the savegame accepts the
write.  `government_name`, `adv.wonder_city`, `rates.*`, `research.goal_name`
and `city.ai.urgency` are all of that kind.  The working operators either change
something the engine treats as world state (gold, cities, units, terrain,
resources, wonders) or go through an official `edit.*` entry point.

⚠️ `lua cmd` takes ONE line and **discards syntax errors silently**.  Never
judge a Lua operator by its log output -- `unleash_barbarians` printed nothing
and worked.  Read the state back.
"""

from __future__ import annotations

import dataclasses

from dataclasses import dataclass, field
from typing import Any, Callable

from .geo import RESOURCE_EXTRAS, MapInfo
from .savegame import Savegame
from .state import WorldState


# --------------------------------------------------------------------------
# target selection
# --------------------------------------------------------------------------

@dataclass
class Ctx:
    """What an operator needs to know about the checkpoint it acts on.

    Targets are chosen by *score rank*, not by player index, so that "intervene
    on the median civ" means the same thing in every world.  Picking `player0`
    everywhere would mean the operator lands on a runaway leader in one world
    and a doomed rump state in the next, and the effect distribution would be
    measuring that instead of the operator.
    """
    sg: Savegame
    ws: WorldState
    mi: MapInfo
    target_pid: int
    target_name: str
    other_pid: int          # the target's strongest rival, for pairwise ops
    other_name: str


def make_ctx(sg: Savegame, ws: WorldState, mi: MapInfo, rank: int = -1) -> Ctx:
    """`rank` -1 selects the median living civ; 0 the leader."""
    alive = [(i, p) for i, p in enumerate(ws.players) if p.is_alive and p.cities]
    if not alive:
        alive = list(enumerate(ws.players))
    ranked = sorted(alive, key=lambda ip: -ip[1].score)
    idx = len(ranked) // 2 if rank < 0 else min(rank, len(ranked) - 1)
    tpid, tp = ranked[idx]
    rivals = [ip for ip in ranked if ip[0] != tpid]
    # v2: `other` is the strongest rival the target is NOT already at war with.
    # L4-love drives love toward hostility, and on a pair already at War the
    # relation has nothing left to flip (81% of v1's L4-love arms showed no
    # movement on `at_war` for exactly this reason).  Falls back to the
    # strongest rival when every relation is War.
    def _rel(i):
        return (tp.diplomacy or {}).get(f"player{i}", "")
    peaceful = [ip for ip in rivals if _rel(ip[0]) not in ("War", "Never met", "")]
    opid, op = (peaceful or rivals)[0] if rivals else (tpid, tp)
    return Ctx(sg=sg, ws=ws, mi=mi, target_pid=tpid, target_name=tp.name,
               other_pid=opid, other_name=op.name)


# --------------------------------------------------------------------------
# the operator record
# --------------------------------------------------------------------------

@dataclass
class Op:
    id: str
    layer: str
    cond_text: Callable[[Ctx], str]
    expected_path: list[str]
    horizon_band: tuple[int, int]
    mechanism_var: Callable[[Savegame, Ctx], Any]
    mutate: Callable[[Savegame, Ctx], str] | None = None
    lua: Callable[[Ctx], list[str]] | None = None
    verify: Callable[[Savegame, Savegame, Ctx], tuple[bool, str]] | None = None
    placebo_of: str | None = None
    # How a placebo is matched to its real operator, which decides how P-5
    # checks it:
    #   "type"       same edit, mechanism deliberately not engaged  -> the
    #                mechanism variable must not move at all
    #   "magnitude"  same edit and same mechanism, negligible size  -> the
    #                mechanism variable moves, but by far less than the real arm
    placebo_kind: str = ""
    # Which resolved variables this operator is *supposed* to move.  The generic
    # distance averages over every civ and every variable, so ~71 of 72
    # (civ, var) pairs contribute only path noise -- and because that noise sits
    # in both the real and the placebo arm, it compresses the ratio toward 1
    # regardless of how large the real effect is.  Scoring on the declared
    # variables removes that shared floor.
    score_vars: tuple[str, ...] = ()
    # How this operator's effect is measured.  "distance" compares `score_vars`
    # against the matched placebo; "termination" compares how often the world
    # resolves before the horizon.  `L2-city-kill` needs the second: on the
    # resolved variables it is indistinguishable from razing the same number of
    # small cities (0.96), but it ends worlds at 8.9% against the placebo's 2.2%.
    score_by: str = "distance"
    notes: str = ""
    # v2: a third channel besides `mutate` (edit the save) and `lua` (script at
    # start): server settings to `set` after the checkpoint is loaded.  This is how
    # the world-parameter operators (`WP-*`) intervene -- `do(disasters x3)` is a
    # settings change, not a save edit.  `mutate` still supplies the description.
    settings: Callable[[Savegame, Ctx], dict[str, Any]] | None = None

    def is_placebo(self) -> bool:
        return self.placebo_of is not None


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------

def _sec(ctx: Ctx) -> str:
    return f"player{ctx.target_pid}"


def _lua_for_player(pid: int, body: str, tag: str) -> str:
    """Wrap a body in the single-line player loop that the server accepts."""
    return (f'for p in players_iterate() do if p.id=={pid} then '
            f'log.normal("{tag}=%s", tostring(pcall(function() {body} end))) '
            f'end end')


def _plane_and_bit(extras_vector: list[str], name: str) -> tuple[str, int]:
    idx = extras_vector.index(name)
    return f"e{idx // 4:02d}", idx % 4


def _land_tiles(ctx: Ctx, owner: str | None, want_terrain=None, need_empty=False):
    """Walk the map yielding (x, y) under an ownership / terrain filter."""
    mi = ctx.mi
    for y in range(mi.ysize):
        row = mi.terrain[y] if y < len(mi.terrain) else ""
        for x in range(min(mi.xsize, len(row))):
            tok = mi.owner[y][x] if (y < len(mi.owner) and x < len(mi.owner[y])) else "-"
            if owner is not None and tok != owner:
                continue
            if owner is None and tok.isdigit():
                continue                      # "unowned" means unowned
            tname = mi.terrain_names.get(row[x], "")
            if want_terrain and tname not in want_terrain:
                continue
            ex = mi.extras[y][x] if y < len(mi.extras) else []
            if need_empty and ex:
                continue
            yield x, y


def _capital_xy(ctx: Ctx) -> tuple[int, int] | None:
    p = ctx.ws.players[ctx.target_pid]
    if not p.cities:
        return None
    cap = max(p.cities, key=lambda c: c.size)
    return cap.x, cap.y


def _remote_water(ctx: Ctx) -> tuple[int, int] | None:
    """A water tile as far from any city as we can cheaply find."""
    mi = ctx.mi
    cities = [(c.x, c.y) for p in ctx.ws.players for c in p.cities]
    best, bestd = None, -1
    for y in range(0, mi.ysize, 2):
        row = mi.terrain[y] if y < len(mi.terrain) else ""
        for x in range(0, min(mi.xsize, len(row)), 2):
            if mi.terrain_names.get(row[x], "") not in ("Ocean", "Deep Ocean", "Lake"):
                continue
            d = min((abs(x - cx) + abs(y - cy) for cx, cy in cities), default=999)
            if d > bestd:
                best, bestd = (x, y), d
    return best


# --------------------------------------------------------------------------
# mechanism readers -- what each operator is supposed to move
# --------------------------------------------------------------------------

def mv_gold(sg, ctx):    return int(sg.get(_sec(ctx), "gold", 0))
def mv_ncities(sg, ctx): return int(sg.get(_sec(ctx), "ncities", 0))
def mv_nunits(sg, ctx):  return int(sg.get(_sec(ctx), "nunits", 0))
def mv_rates(sg, ctx):   return tuple(int(sg.get(_sec(ctx), f"rates.{k}", 0))
                                      for k in ("tax", "science", "luxury"))
def mv_gov(sg, ctx):     return str(sg.get(_sec(ctx), "government_name", ""))


def mv_holds_gov_tech(sg, ctx):
    """Whether the target holds the government-unlocking tech this operator grants.

    Gate 4 has to read the *proximal* link.  Using the government itself was
    wrong: the arm demonstrably diverged (the tech was granted) yet the civ had
    not switched government 20 turns later, and the gate called that "the edit
    never took".  §2.5.5b already established the right division of labour --
    the government *path* is what the branch data scores, the tech grant is what
    tells us the intervention landed.
    """
    pick = pick_gov_tech(ctx)
    if pick is None:
        return None
    tech, _gov = pick
    tv = [str(x) for x in sg.get("savefile", "technology_vector", [])]
    r = sg.table("research", "r")
    if r is None or ctx.target_pid >= len(r.rows) or tech not in tv:
        return None
    mask = str(r.rows[ctx.target_pid][-1])
    i = tv.index(tech)
    return mask[i] == "1" if i < len(mask) else None


def mv_has_tech(name: str):
    """Whether the target holds one named tech.

    The tech *count* cannot be the mechanism variable for a give-tech operator:
    the placebo also gives a tech, so both arms move the count by one and the
    check would pass for a placebo that is doing exactly what it must not do.
    """
    def read(sg, ctx):
        r = sg.table("research", "r")
        tv = [str(x) for x in sg.get("savefile", "technology_vector", [])]
        if r is None or ctx.target_pid >= len(r.rows) or name not in tv:
            return None
        mask = str(r.rows[ctx.target_pid][-1])
        i = tv.index(name)
        return mask[i] == "1" if i < len(mask) else None
    return read


def mv_holds_picked(candidates: list[str]):
    """Whether the target holds the tech this operator would have granted.

    Must read the same choice the operator made, so it re-runs `pick_tech`
    against the *checkpoint* context (`ctx.sg`), not against the savegame being
    inspected -- otherwise the granted tech would already be held and the reader
    would pick a different one.
    """
    def read(sg, ctx):
        tech = pick_tech(ctx, candidates)
        if tech is None:
            return None
        tv = [str(x) for x in sg.get("savefile", "technology_vector", [])]
        r = sg.table("research", "r")
        if r is None or ctx.target_pid >= len(r.rows) or tech not in tv:
            return None
        mask = str(r.rows[ctx.target_pid][-1])
        i = tv.index(tech)
        return mask[i] == "1" if i < len(mask) else None
    return read


def mv_techs(sg, ctx):
    r = sg.table("research", "r")
    if r is None or ctx.target_pid >= len(r.rows):
        return 0
    return str(r.rows[ctx.target_pid][-1]).count("1")


def mv_bulbs(sg, ctx):
    r = sg.table("research", "r")
    if r is None or ctx.target_pid >= len(r.rows):
        return 0
    return int(dict(zip(r.columns, r.rows[ctx.target_pid])).get("bulbs", 0))


def mv_goal(sg, ctx):
    r = sg.table("research", "r")
    if r is None or ctx.target_pid >= len(r.rows):
        return ""
    return str(dict(zip(r.columns, r.rows[ctx.target_pid])).get("goal_name", ""))


def mv_total_city_size(sg, ctx):
    """Summed city size of the target.

    `ncities` cannot be the mechanism variable for city-kill: the matched
    placebo also removes exactly one city, so both arms move the count by one
    and the check cannot separate them.  What the operator is really doing is
    removing *productive capacity*, and razing the largest city removes far more
    of it than razing the smallest.
    """
    c = sg.table(f"player{ctx.target_pid}", "c")
    if c is None:
        return 0
    j = c.col("size")
    return sum(int(r[j]) for r in c.rows)


def mv_worked_resources(sg, ctx):
    """Resources on the tiles the target held **at the checkpoint**.

    The tile set is frozen at intervention time on purpose.  Counting "resources
    in the target's current territory" drifts: borders expand over twenty turns
    and swallow the far-away tiles the placebo edited, so a placebo that
    correctly changed nothing anybody was using read as having engaged the
    mechanism.  A frozen tile set makes the reading a property of the edit
    rather than of subsequent border movement.
    """
    from .geo import extract_map
    from .state import extract
    tok = str(ctx.target_pid)
    tiles = [(x, y) for y in range(ctx.mi.ysize)
             for x in range(ctx.mi.xsize)
             if (y < len(ctx.mi.owner) and x < len(ctx.mi.owner[y])
                 and ctx.mi.owner[y][x] == tok)]
    ws = extract(sg, "mv")
    mi = extract_map(sg, [p.name for p in ws.players])
    n = 0
    for x, y in tiles:
        if y >= len(mi.extras) or x >= len(mi.extras[y]):
            continue
        n += sum(1 for e in mi.extras[y][x] if e in RESOURCE_EXTRAS)
    return n


def mv_units_lost(sg, ctx):
    """Cumulative units lost -- the right reading for an external shock.

    `nunits` is too noisy: the AI builds and loses units constantly, so a remote
    barbarian spawn that touched nobody still moved the live count by one while
    the real arm moved it by two.  `units_lost` only rises when this civ actually
    loses something.
    """
    return int(sg.get(f"score{ctx.target_pid}", "units_lost", 0))


def mv_owned_resources(sg, ctx):
    """Resource tiles inside the target's territory -- re-read, not cached."""
    from .geo import extract_map
    from .state import extract
    ws = extract(sg, "mv")
    mi = extract_map(sg, [p.name for p in ws.players])
    tok = str(ctx.target_pid)
    n = 0
    for y in range(mi.ysize):
        for x in range(mi.xsize):
            o = mi.owner[y][x] if (y < len(mi.owner) and x < len(mi.owner[y])) else "-"
            if o != tok:
                continue
            n += sum(1 for e in mi.extras[y][x] if e in RESOURCE_EXTRAS)
    return n


def mv_total_resources(sg, ctx):
    from .geo import extract_map
    from .state import extract
    ws = extract(sg, "mv")
    mi = extract_map(sg, [p.name for p in ws.players])
    return sum(1 for y in range(mi.ysize) for x in range(mi.xsize)
               for e in mi.extras[y][x] if e in RESOURCE_EXTRAS)


def mv_owned_mountains(sg, ctx):
    from .geo import extract_map
    from .state import extract
    ws = extract(sg, "mv")
    mi = extract_map(sg, [p.name for p in ws.players])
    inv = {v: k for k, v in mi.terrain_names.items()}
    m = inv.get("Mountains")
    tok = str(ctx.target_pid)
    if not m:
        return -1
    n = 0
    for y in range(mi.ysize):
        row = mi.terrain[y] if y < len(mi.terrain) else ""
        for x in range(min(mi.xsize, len(row))):
            o = mi.owner[y][x] if (y < len(mi.owner) and x < len(mi.owner[y])) else "-"
            if o == tok and row[x] == m:
                n += 1
    return n


def _has_improvement(sg, ctx, name: str) -> bool:
    names = [str(x) for x in sg.get("savefile", "improvement_vector", [])]
    if name not in names:
        return False
    bit = names.index(name)
    c = sg.table(_sec(ctx), "c")
    if c is None:
        return False
    for row in c.rows:
        mask = str(row[c.col("improvements")])
        if bit < len(mask) and mask[bit] == "1":
            return True
    return False


def mv_great_library(sg, ctx): return _has_improvement(sg, ctx, "Great Library")
def mv_barracks(sg, ctx):      return _has_improvement(sg, ctx, "Barracks")


def mv_explored(sg, ctx):
    from .geo import explored_fraction, extract_map
    from .state import extract
    ws = extract(sg, "mv")
    mi = extract_map(sg, [p.name for p in ws.players])
    if ctx.target_pid >= len(ws.players):
        return 0.0
    return round(explored_fraction(mi, ws.players[ctx.target_pid].name), 4)


def mv_warming(sg, ctx):
    return (int(sg.get("game", "globalwarming", 0)), int(sg.get("game", "heating", 0)))


def mv_trait(dim: str):
    """`effective = val + mod` for one trait dimension (§2.5.6).

    ★ 2026-09-14: added so the four single-dimension trait operators can each verify
    their own dimension.  `mv_trait_aggr` is kept as an alias for the existing
    two-dimension `L4-trait`, whose arms are already on disk.
    """
    def f(sg, ctx):
        tv = [str(x) for x in sg.get("savefile", "trait_vector", [])]
        t = sg.table(_sec(ctx), "trait")
        if t is None or dim not in tv:
            return None
        r = t.rows[tv.index(dim)]
        return int(r[0]) + int(r[1])
    f.__name__ = f"mv_trait_{dim.lower()}"
    return f


mv_trait_aggr = mv_trait("Aggressive")


def mv_love(sg, ctx):
    ai = sg.table(_sec(ctx), "ai")
    if ai is None or ctx.other_pid >= len(ai.rows):
        return None
    return int(ai.rows[ctx.other_pid][ai.col("love")])


def mv_diplstate(sg, ctx):
    d = sg.table(_sec(ctx), "diplstate")
    if d is None or ctx.other_pid >= len(d.rows):
        return None
    return str(d.rows[ctx.other_pid][d.col("current")])


def mv_urgency(sg, ctx):
    c = sg.table(_sec(ctx), "c")
    if c is None or not c.rows:
        return 0
    j = c.col("ai.urgency")
    return sum(int(r[j]) for r in c.rows)


def mv_unit_positions(sg, ctx):
    u = sg.table(_sec(ctx), "u")
    if u is None:
        return ()
    xi, yi, ii = u.col("x"), u.col("y"), u.col("id")
    return tuple(sorted((int(r[ii]), int(r[xi]), int(r[yi])) for r in u.rows))


# --------------------------------------------------------------------------
# L1 -- stock
# --------------------------------------------------------------------------

# Tech lists the give-tech operators draw from.  A tech the civ already holds
# makes `give_tech` a no-op, and the first run of the gate caught exactly that:
# the median civ at T120 already had Bronze Working, Alphabet AND Monarchy, so
# three operators reported REVERTED when nothing had been attempted.  The choice
# is therefore made against the checkpoint, and an operator with no applicable
# choice reports SKIP rather than pretending to have intervened.
MILITARY_TECHS = ["Bronze Working", "Iron Working", "Horseback Riding",
                  "Mathematics", "Construction", "Feudalism", "Chivalry",
                  "Gunpowder", "Metallurgy", "Conscription"]
# Used by the paired placebo: a tech the civ lacks that is not on the military
# list.  ⚠️ We do NOT claim these unlock nothing -- that would need the ruleset's
# requirement graph.  The verified claim is narrower and sufficient: the
# operator's mechanism variable is "holds the specific granted tech", and a
# different tech leaves it false.
NEUTRAL_TECHS = ["Alphabet", "Pottery", "Ceremonial Burial", "Code of Laws",
                 "Writing", "Mysticism", "Astronomy", "Seafaring",
                 "Polytheism", "Banking", "Medicine", "Sanitation"]
# (tech, government it unlocks) from civ2civ3/governments.ruleset.
GOV_TECHS = [("Monarchy", "Monarchy"), ("The Republic", "Republic"),
             ("Communism", "Communism"), ("Theology", "Fundamentalism"),
             ("Economics", "Federation"), ("Democracy", "Democracy")]


def _held_techs(ctx: Ctx) -> set[str]:
    r = ctx.sg.table("research", "r")
    tv = [str(x) for x in ctx.sg.get("savefile", "technology_vector", [])]
    if r is None or ctx.target_pid >= len(r.rows):
        return set()
    mask = str(r.rows[ctx.target_pid][-1])
    return {tv[i] for i, ch in enumerate(mask) if ch == "1" and i < len(tv)}


def _tech_vector(ctx: Ctx) -> list[str]:
    return [str(x) for x in ctx.sg.get("savefile", "technology_vector", [])]


def pick_tech(ctx: Ctx, candidates: list[str]) -> str | None:
    """First candidate the target lacks, or None if it already holds them all."""
    have, tv = _held_techs(ctx), _tech_vector(ctx)
    return next((t for t in candidates if t in tv and t not in have), None)


def pick_gov_tech(ctx: Ctx) -> tuple[str, str] | None:
    """A government-unlocking tech the target lacks, for a government it is not
    already running.  Giving Monarchy to a civ already in Monarchy does nothing,
    which is what the first gate run measured."""
    have, tv = _held_techs(ctx), _tech_vector(ctx)
    cur = str(ctx.sg.get(f"player{ctx.target_pid}", "government_name", ""))
    for tech, gov in GOV_TECHS:
        if tech in tv and tech not in have and gov != cur:
            return tech, gov
    return None


def _add_gold(amount: int):
    def f(sg: Savegame, ctx: Ctx) -> str:
        sec = _sec(ctx)
        before = int(sg.get(sec, "gold", 0))
        sg.set_scalar(sec, "gold", max(0, before + amount))
        return f"{ctx.target_name} gold {before} -> {max(0, before + amount)}"
    return f


def _set_gold(value: int):
    def f(sg: Savegame, ctx: Ctx) -> str:
        sec = _sec(ctx)
        before = int(sg.get(sec, "gold", 0))
        sg.set_scalar(sec, "gold", value)
        return f"{ctx.target_name} gold {before} -> {value}"
    return f


def _gold_swing(big: bool):
    """L1-gold-swing: one operator slot, both directions.  Even branch turn -> the
    treasury gains 500; odd branch turn -> it is emptied (user 2026-09-09: the two
    one-directional gold operators were redundant; merge, do not drop one).  The
    placebo mirrors the direction at a negligible magnitude (+5 / -5)."""
    def f(sg: Savegame, ctx: Ctx) -> str:
        even = ctx.sg.turn % 2 == 0
        if big:
            return (_add_gold(500) if even else _set_gold(0))(sg, ctx)
        return (_add_gold(5) if even else _add_gold(-5))(sg, ctx)
    return f


def _gold_swing_text(big: bool):
    def t(c: Ctx) -> str:
        even = c.sg.turn % 2 == 0
        if big:
            return f"{c.target_name} 的国库突然增加了 500 金" if even else f"{c.target_name} 的国库被清空"
        return f"{c.target_name} 的国库增加了 5 金" if even else f"{c.target_name} 的国库减少了 5 金"
    return t


def _give_tech(candidates: list[str], n: int = 1):
    """Grant the first `n` candidate techs the target lacks.

    ⚠️ When n > 1 and the placebo also grants n techs, `n_techs` is matched by
    construction and cannot discriminate the arms -- the pilot measured L1-tech at
    0.99 partly for this reason.  Such operators must be scored on what the techs
    *unlock* (units built, army size), not on the tech count.
    """
    def f(ctx: Ctx) -> list[str]:
        have, tv = _held_techs(ctx), _tech_vector(ctx)
        picks = [t for t in candidates if t in tv and t not in have][:n]
        if not picks:
            return []
        return [_lua_for_player(
            ctx.target_pid,
            f'edit.give_tech(p, find.tech_type("{t}"), -1, false, "researched")',
            f"GIVETECH_{t}") for t in picks]
    return f


def _give_gov_tech(ctx: Ctx) -> list[str]:
    pick = pick_gov_tech(ctx)
    if pick is None:
        return []
    tech, _gov = pick
    body = f'edit.give_tech(p, find.tech_type("{tech}"), -1, false, "researched")'
    return [_lua_for_player(ctx.target_pid, body, f"GOVTECH_{tech}")]


def _give_bulbs(base: int, per_tech: int = 0):
    """Grant research points, scaled by how far the civ has already got.

    A flat 200 completes a tech in the opening and completes nothing by turn 160 --
    research cost climbs with the number of techs held, so a fixed amount silently
    turns into a no-op mid-game (measured: the arm diverged, proving the bulbs were
    granted, yet `n_techs` was unchanged five turns later).  Overshoot is harmless:
    surplus bulbs carry into the next tech.
    """
    def f(ctx: Ctx) -> list[str]:
        n = base + per_tech * len(_held_techs(ctx))
        return [_lua_for_player(ctx.target_pid, f"edit.give_bulbs(p, {n})", "BULBS")]
    return f


# --------------------------------------------------------------------------
# L2 -- constraint
# --------------------------------------------------------------------------

def _kill_city(which: str, fraction: float = 0.5):
    """Raze a fraction of a civ's cities -- and clear everything pointing at them.

    `which` = 'largest' or 'smallest'.  Both arms raze the **same number** of
    cities and differ only in which ones.  That pairing is deliberate: when the
    operator razed one city and its placebo razed one city, `n_cities` moved
    identically in both arms and the comparison measured almost nothing (pilot
    ratio 0.92).  With the count matched, population and score become the clean
    discriminators between losing your best cities and losing your worst.

    Two kinds of bookkeeping are mandatory.

    **The count scalar.**  freeciv stores `ncities` separately and refuses a file
    whose city table length disagrees with it.

    *** Every reference to each razed city id.**  The first version deleted the
    row, fixed the count, and stopped.  It passed the four-gate at a turn-120
    checkpoint and then **segfaulted the server on load** (rc=139 in
    `_load_players`) at turn 235, where the world is denser.  Counting references
    on that checkpoint explains it: razing one city left 8 tiles in `map.worked`
    still claimed by it, 6 units still homed there, and 5 `dc` memories elsewhere.

    The crash is the lucky outcome.  The dangerous one is a mutated file that
    loads *anyway* and runs with dangling city links, because then the operator
    silently measures a corrupted world instead of a razed city.

    WARNING: clearing the obvious five reference kinds was **not enough**, and
    guessing which one remained produced two wrong theories (the Palace; an
    `aifill` player being added -- that log line is printed for every player on
    every load).  What found it was a brute-force scan of the mutated file for the
    razed id: a *foreign* city held a **trade route** to it, and an AI bodyguard's
    `charge` field pointed at it (`charge` holds either a unit id or a city id).
    Cross-player references are the ones intuition misses.
    """
    def f(sg: Savegame, ctx: Ctx) -> str:
        sec = _sec(ctx)
        c = sg.table(sec, "c")
        if c is None or len(c.rows) < 2:
            return ""
        rows = c.as_dicts()
        k = min(max(1, int(len(rows) * fraction)), len(rows) - 1)
        order = sorted(range(len(rows)), key=lambda i: -int(rows[i].get("size", 0)))
        if which != "largest":
            order = order[::-1]
        victims = [rows[i] for i in order[:k]]

        cleared = {"worked": 0, "source": 0, "homed": 0, "wonder_city": 0,
                   "charge": 0, "traderoute": 0, "dc": 0}
        razed = []

        for victim in victims:
            vid = int(victim["id"])
            razed.append(f"{victim['name']}({victim['size']})")

            # 1. tiles it worked, and territory claims radiating from it
            msec = sg.sections.get("map", {})
            for prefix in ("worked", "source"):
                y = 0
                while f"{prefix}{y:04d}" in msec:
                    cells = [q.strip() for q in
                             str(msec[f"{prefix}{y:04d}"]).split(",")]
                    for x, val in enumerate(cells):
                        if val == str(vid):
                            sg.set_csv_cell("map", prefix, y, x, "-")
                            cleared[prefix] += 1
                    y += 1

            # 2. units homed there -> homeless (0 is a state real units have)
            for s2 in sg.players():
                u = sg.table(s2, "u")
                if u is None:
                    continue
                j = u.col("homecity")
                for i in range(len(u.rows)):
                    if int(u.rows[i][j]) == vid:
                        sg.set_cell(s2, "u", i, "homecity", 0)
                        cleared["homed"] += 1

            # 3. the AI's chosen wonder-building city
            for s2 in sg.players():
                if int(sg.get(s2, "adv.wonder_city", 0) or 0) == vid:
                    sg.set_scalar(s2, "adv.wonder_city", 0)
                    cleared["wonder_city"] += 1

            # 4. AI bodyguard assignments -- `charge` may hold a city id
            for s2 in sg.players():
                u = sg.table(s2, "u")
                if u is None:
                    continue
                for col in ("charge", "texaicharge"):
                    if col not in u.columns:
                        continue
                    j = u.col(col)
                    for i in range(len(u.rows)):
                        try:
                            ref = int(u.rows[i][j])
                        except (TypeError, ValueError):
                            continue
                        if ref == vid:
                            sg.set_cell(s2, "u", i, col, 0)
                            cleared["charge"] += 1

            # 5. * trade routes to it, in EVERY civ's cities -- the one that
            #    actually crashed the server.  0 is the empty state.
            for s2 in sg.players():
                cc = sg.table(s2, "c")
                if cc is None:
                    continue
                for col in [q for q in cc.columns if q.startswith("traderoute")]:
                    j = cc.col(col)
                    for i in range(len(cc.rows)):
                        try:
                            ref = int(cc.rows[i][j])
                        except (TypeError, ValueError):
                            continue
                        if ref == vid:
                            sg.set_cell(s2, "c", i, col, 0)
                            cleared["traderoute"] += 1

            # 6. every civ's remembered copy of it.  `delete_rows` re-parses,
            #    so the table is re-fetched per civ rather than cached.
            for s2 in sg.players():
                d = sg.table(s2, "dc")
                if d is None:
                    continue
                drop = [i for i in range(len(d.rows))
                        if int(d.rows[i][d.col("id")]) == vid]
                if not drop:
                    continue
                sg.delete_rows(s2, "dc", drop)
                cleared["dc"] += len(drop)
                if sg.get(s2, "dc_total") is not None:
                    sg.set_scalar(s2, "dc_total",
                                  max(0, int(sg.get(s2, "dc_total")) - len(drop)))

            # 7. finally the city row itself
            c2 = sg.table(sec, "c")
            pick = next((i for i in range(len(c2.rows))
                         if int(c2.rows[i][c2.col("id")]) == vid), None)
            if pick is None:
                continue
            sg.delete_rows(sec, "c", [pick])
            sg.set_scalar(sec, "ncities", int(sg.get(sec, "ncities")) - 1)

        if not razed:
            return ""
        return (f"razed {len(razed)} {which} cities of {ctx.target_name} "
                f"[{', '.join(razed)}]; cleared "
                + ", ".join(f"{v} {k}" for k, v in cleared.items() if v))
    return f


def _wipe_units(all_of_them: bool):
    def f(sg: Savegame, ctx: Ctx) -> str:
        sec = _sec(ctx)
        u = sg.table(sec, "u")
        if u is None or not u.rows:
            return ""
        n = len(u.rows)
        drop = list(range(n)) if all_of_them else [n - 1]
        sg.delete_rows(sec, "u", drop)
        sg.set_scalar(sec, "nunits", n - len(drop))
        return f"deleted {len(drop)}/{n} units of {ctx.target_name}"
    return f


def _shift_rates(amount: int):
    """Move `amount` percentage points from research into the treasury.

    The three rates must still sum to 100 or the savegame fails its own
    invariant check, and each government caps a single rate, so the shift is
    clamped rather than assumed to fit.
    """
    def f(sg: Savegame, ctx: Ctx) -> str:
        sec = _sec(ctx)
        tax = int(sg.get(sec, "rates.tax", 0))
        sci = int(sg.get(sec, "rates.science", 0))
        lux = int(sg.get(sec, "rates.luxury", 0))
        move = min(amount, sci)
        if move <= 0:
            return ""
        sg.set_scalar(sec, "rates.science", sci - move)
        sg.set_scalar(sec, "rates.tax", tax + move)
        return (f"{ctx.target_name} rates tax {tax}->{tax+move}, "
                f"science {sci}->{sci-move}, luxury {lux}")
    return f


# Relations that mean "not fighting".  Writing "Peace" over an Alliance is a
# downgrade the engine simply undoes -- measured: at a checkpoint where the pair
# was allied, the operator read back Alliance five turns later and the arm was
# scored REVERTED.  So the operator flips the relation instead of assuming war.
_PEACEFUL = {"Peace", "Alliance", "Cease-fire", "Armistice", "Team"}


def _flip_relation(a_pid_key: str = "target"):
    """Invert a pair's diplomatic state: fighting -> Peace, otherwise -> War.

    Both sides must be written.  `diplstate` is stored per player, so editing one
    side leaves the pair inconsistent.
    """
    def f(sg: Savegame, ctx: Ctx) -> str:
        a, b = ctx.target_pid, ctx.other_pid
        ns = sg.players()
        d = sg.table(ns[a], "diplstate")
        if d is None or b >= len(d.rows):
            return ""
        cur = str(d.rows[b][d.col("current")])
        if cur == "Never met":
            return ""
        new = "Peace" if cur not in _PEACEFUL else "War"
        for me, them in ((a, b), (b, a)):
            dd = sg.table(ns[me], "diplstate")
            if dd is None or them >= len(dd.rows):
                return ""
            sg.set_cell(ns[me], "diplstate", them, "current", new)
        return (f"{ctx.target_name} and {ctx.other_name}: {cur} -> {new}")
    return f


def _peace_elsewhere(sg: Savegame, ctx: Ctx) -> str:
    """Placebo for the diplomacy operator: the same edit, between two *other* civs.

    The first design looked for a never-met pair, but by mid-game everyone has
    met everyone and the arm just skipped.  Forcing peace between two third
    parties is the better control: identical edit, identical mechanism type, and
    the target's own strategic position -- which is what `mechanism_var` reads --
    is untouched.
    """
    ns = sg.players()
    others = [i for i in range(len(ns)) if i not in (ctx.target_pid, ctx.other_pid)]
    for a in others:
        for b in others:
            if a >= b:
                continue
            da, db = sg.table(ns[a], "diplstate"), sg.table(ns[b], "diplstate")
            if da is None or db is None or b >= len(da.rows) or a >= len(db.rows):
                continue
            cur = str(da.rows[b][da.col("current")])
            if cur == "Never met":
                continue
            new = "Peace" if cur not in _PEACEFUL else "War"
            sg.set_cell(ns[a], "diplstate", b, "current", new)
            sg.set_cell(ns[b], "diplstate", a, "current", new)
            return (f"flipped third parties player{a}/player{b}: "
                    f"{cur} -> {new} (placebo)")
    return ""


# --------------------------------------------------------------------------
# L3 -- endowment
# --------------------------------------------------------------------------

def _add_resources(where: str, n: int = 12, res: str = "Wine"):
    """Put a natural resource on n tiles.

    `where` = 'worked' (inside the target's economy) or 'nowhere' (unowned land far
    from everyone) -- the placebo makes the same bit-plane edit somewhere that
    cannot feed anybody's city.

    Two corrections, both from measurements:

    ★ **'worked' means worked, not merely owned.**  Filtering on ownership alone
    left the operator inert in 5 of 10 arms: early on, a civ owns far more land than
    its cities can work, so the resource landed on tiles nobody was using and the
    world did not diverge at all.  Worked tiles are taken first, owned-but-unworked
    only as a fallback.

    ★ **Only tiles that already carry a resource are skipped.**  Requiring "no
    extras at all" excluded nearly every worked tile, because a worked tile normally
    has irrigation or a road -- the operator was placing 1-5 resources where it
    asked for 12.  A road does not stop a tile from having ore.

    ★ **Both arms place min(quota, real capacity, placebo capacity).**  Capping only
    the placebo was not enough: late in a game the map fills up and it is the
    *unowned* land that runs out, so the real arm placed 12 where the placebo could
    manage 1.  Matching requires the smaller of the two capacities, applied to both.
    Below a floor of 4 tiles the operator is not meaningfully applicable and returns
    NA rather than a token edit.
    """
    FLOOR = 4
    GOOD = {"Hills", "Grassland", "Plains"}

    def _own_candidates(ctx: Ctx):
        mi = ctx.mi
        my_ids = {str(c.id) for c in ctx.ws.players[ctx.target_pid].cities}
        for want_worked in (True, False):          # worked tiles first
            for y in range(mi.ysize):
                row = mi.terrain[y] if y < len(mi.terrain) else ""
                for x in range(min(mi.xsize, len(row))):
                    o = (mi.owner[y][x] if (y < len(mi.owner)
                         and x < len(mi.owner[y])) else "-")
                    if o != str(ctx.target_pid):
                        continue
                    if mi.terrain_names.get(row[x], "") not in GOOD:
                        continue
                    if any(e in RESOURCE_EXTRAS for e in mi.extras[y][x]):
                        continue
                    w = (mi.worked[y][x] if (y < len(mi.worked)
                         and x < len(mi.worked[y])) else "-")
                    if (w in my_ids) != want_worked:
                        continue
                    yield x, y

    def f(sg: Savegame, ctx: Ctx) -> str:
        ev = [str(x) for x in sg.get("savefile", "extras_vector", [])]
        if res not in ev:
            return ""
        plane, bit = _plane_and_bit(ev, res)
        cap_real = sum(1 for _ in _own_candidates(ctx))
        cap_plac = sum(1 for _ in _land_tiles(ctx, None, want_terrain=GOOD,
                                              need_empty=True))
        quota = min(n, cap_real, cap_plac)
        if quota < FLOOR:
            return ""
        src = (_own_candidates(ctx) if where == "worked"
               else _land_tiles(ctx, None, want_terrain=GOOD, need_empty=True))
        placed = 0
        coords: list[str] = []
        for x, y in src:
            if placed >= quota:
                break
            sg.set_bitplane_bit("map", plane, y, x, bit, True)
            placed += 1
            coords.append(f"{x},{y}")
        # v2: the coordinates are part of `applied` so the benchmark can replay
        # the edit into the report (edit channel) instead of falling back to text.
        return (f"added {res} to {placed} tiles ({where}) at [{' '.join(coords)}]"
                if placed else "")
    return f


# Tile improvements the civ built.  Stripping these alongside the resource is
# what makes the operator bite: removing a resource alone barely registered
# (measured 0.54, below its placebo) because the AI simply works another tile.
# Taking the irrigation and mines with it removes the developed surface itself.
INFRA_EXTRAS = ("Irrigation", "Farmland", "Mine", "Oil Well", "Oil Platform",
                "Road", "Railroad", "Maglev", "Buoy")


def _strip_resources(where: str, limit: int | None = None, infra: bool = True):
    def f(sg: Savegame, ctx: Ctx) -> str:
        ev = [str(x) for x in sg.get("savefile", "extras_vector", [])]
        owner = str(ctx.target_pid) if where == "owned" else None
        kill = set(RESOURCE_EXTRAS) | (set(INFRA_EXTRAS) if infra else set())
        removed = 0
        touched: list[str] = []
        for x, y in _land_tiles(ctx, owner):
            hit = False
            for e in ctx.mi.extras[y][x]:
                if e not in kill or e not in ev:
                    continue
                plane, bit = _plane_and_bit(ev, e)
                sg.set_bitplane_bit("map", plane, y, x, bit, False)
                removed += 1
                hit = True
            if hit:
                touched.append(f"{x},{y}")
            if limit and removed >= limit:
                break
        return (f"stripped {removed} resource/infrastructure extras ({where}) "
                f"at [{' '.join(touched)}]" if removed else "")
    return f


def _flatten(where: str, n: int = 20):
    def f(sg: Savegame, ctx: Ctx) -> str:
        inv = {v: k for k, v in ctx.mi.terrain_names.items()}
        mtn, plains = inv.get("Mountains"), inv.get("Plains")
        if not mtn or not plains:
            return ""
        owner = str(ctx.target_pid) if where == "owned" else None
        changed = 0
        for x, y in _land_tiles(ctx, owner, want_terrain={"Mountains"}):
            sg.set_char("map", "t", y, x, plains)
            changed += 1
            if changed >= n:
                break
        return f"flattened {changed} mountain tiles ({where})" if changed else ""
    return f


def _grant_improvement(name: str):
    def f(sg: Savegame, ctx: Ctx) -> str:
        names = [str(x) for x in sg.get("savefile", "improvement_vector", [])]
        if name not in names:
            return ""
        bit = names.index(name)
        sec = _sec(ctx)
        c = sg.table(sec, "c")
        if c is None or not c.rows:
            return ""
        # biggest city, so a wonder lands somewhere it can plausibly matter
        rows = c.as_dicts()
        # Biggest city first so a wonder lands where it can plausibly matter,
        # but fall through to the others: the placebo building is often already
        # present in the capital, and returning "" there would make the arm skip
        # for a reason that has nothing to do with the operator.
        order = sorted(range(len(rows)), key=lambda i: -int(rows[i].get("size", 0)))
        pick = None
        for i in order:
            m = str(c.rows[i][c.col("improvements")])
            if bit < len(m) and m[bit] == "0":
                pick = i
                break
        if pick is None:
            return ""
        mask = list(str(c.rows[pick][c.col("improvements")]))
        mask[bit] = "1"
        sg.set_cell(sec, "c", pick, "improvements", "".join(mask))
        return f"granted {name} to {rows[pick]['name']} of {ctx.target_name}"
    return f


def _grant_improvements(names: list[str]):
    """Grant several improvements, each to the biggest city that lacks it.

    v2 replacement for a single `_grant_improvement("Great Library")`: one wonder
    moved culture/score by 0.002 normalised units, below any model's output
    granularity.  Great wonders are world-unique, so a wonder already built by
    *anyone* is skipped rather than duplicated.  `applied` lists every grant so
    the benchmark can edit the report accordingly.
    """
    def f(sg: Savegame, ctx: Ctx) -> str:
        vec = [str(x) for x in sg.get("savefile", "improvement_vector", [])]
        sec = _sec(ctx)
        c = sg.table(sec, "c")
        if c is None or not c.rows:
            return ""
        # wonders built anywhere in the world (any player's city has the bit)
        built_anywhere: set[str] = set()
        for psec in sg.players():
            pc = sg.table(psec, "c")
            if pc is None:
                continue
            for r in pc.rows:
                m = str(r[pc.col("improvements")])
                for i, ch in enumerate(m):
                    if ch == "1" and i < len(vec):
                        built_anywhere.add(vec[i])
        granted: list[str] = []
        for name in names:
            if name not in vec or name in GREAT_WONDER_NAMES and name in built_anywhere:
                continue
            bit = vec.index(name)
            rows = c.as_dicts()
            order = sorted(range(len(rows)), key=lambda i: -int(rows[i].get("size", 0)))
            for i in order:
                m = str(c.rows[i][c.col("improvements")])
                if bit < len(m) and m[bit] == "0":
                    mask = list(m)
                    mask[bit] = "1"
                    sg.set_cell(sec, "c", i, "improvements", "".join(mask))
                    granted.append(f"{name}@{rows[i]['name']}")
                    break
        return (f"granted {len(granted)} buildings to {ctx.target_name}: "
                + "; ".join(granted)) if granted else ""
    return f


GREAT_WONDER_NAMES = {
    "Apollo Program", "A.Smith's Trading Co.", "Colossus", "Copernicus' Observatory",
    "Cure For Cancer", "Darwin's Voyage", "Eiffel Tower", "Great Library", "Great Wall",
    "Hanging Gardens", "Hoover Dam", "Isaac Newton's College", "J.S. Bach's Cathedral",
    "King Richard's Crusade", "Leonardo's Workshop", "Lighthouse", "Magellan's Expedition",
    "Manhattan Project", "Marco Polo's Embassy", "Michelangelo's Chapel", "Oracle",
    "Pyramids", "Shakespeare's Theater", "Statue of Liberty", "Sun Tzu's War Academy",
    "United Nations", "Women's Suffrage", "Statue of Zeus", "Temple of Artemis",
    "Mausoleum of Mausolos", "Internet",
}


def _setting(sg: Savegame, name: str, default: Any) -> Any:
    """Current value of a server setting from the savegame's [settings] table."""
    t = sg.table("settings", "set")
    if t is None:
        return default
    for r in t.rows:
        if str(r[0]) == name:
            # booleans are saved as TRUE/FALSE but `set` takes ENABLED/DISABLED
            return {"TRUE": "ENABLED", "FALSE": "DISABLED"}.get(str(r[1]).upper(), r[1])
    return default


def _wp_scale(name: str, factor: float, default: int, lo: int, hi: int):
    """World-parameter operator: multiply a numeric server setting.

    Returns (mutate, settings): `mutate` only describes (it does not touch the
    save); `settings` is what `run_to_turn` gets as `extra`.  The description is
    relative ("x3") because that is the only phrasing a model can judge without
    seeing the absolute scale (09 §5.2).
    """
    def desc(sg: Savegame, ctx: Ctx) -> str:
        cur = int(_setting(sg, name, default) or default)
        new = int(max(lo, min(hi, round(cur * factor))))
        if new == cur:
            return ""
        return f"setting {name} {cur} -> {new} (x{factor:g})"
    def sett(sg: Savegame, ctx: Ctx) -> dict[str, Any]:
        cur = int(_setting(sg, name, default) or default)
        return {name: int(max(lo, min(hi, round(cur * factor))))}
    return desc, sett


def _wp_set(name: str, value: Any, default: Any):
    def desc(sg: Savegame, ctx: Ctx) -> str:
        cur = _setting(sg, name, default)
        if str(cur) == str(value):
            return ""
        return f"setting {name} {cur} -> {value}"
    def sett(sg: Savegame, ctx: Ctx) -> dict[str, Any]:
        return {name: value}
    return desc, sett


def _wp_multi(values: dict[str, Any], defaults: dict[str, Any]):
    """World-parameter operator over several settings at once (e.g. the
    `migration` master switch plus its chances).  NA only if every value is
    already in place."""
    def desc(sg: Savegame, ctx: Ctx) -> str:
        parts = []
        for name, value in values.items():
            cur = _setting(sg, name, defaults.get(name))
            if str(cur) != str(value):
                parts.append(f"{name} {cur} -> {value}")
        return ("setting " + ", ".join(parts)) if parts else ""
    def sett(sg: Savegame, ctx: Ctx) -> dict[str, Any]:
        return dict(values)
    return desc, sett


# WP-migration: `migration` is the master switch (default DISABLED; 3.1 GEN_BOOL
# takes ENABLED/DISABLED).  Both arms switch it on; the real arm at the maximum
# rates, the placebo at rates that almost never fire.
# ⚠️ `mgr_distance` (default 0) is *added to the city radius*: with 0, citizens
# only migrate between cities whose work areas overlap, which AI city spacing
# never produces -- pilot P3 2026-09-09: migration ENABLED at 100%/30% chance
# was byte-for-byte identical to default over 6 x 400 turns.  Both arms
# therefore also open the distance; the placebo keeps the chances negligible.
_MIGRATION_DEFAULTS = {"migration": "DISABLED", "mgr_turninterval": 5,
                       "mgr_nationchance": 50, "mgr_worldchance": 10, "mgr_distance": 0}
_MIGRATION_HIGH = {"migration": "ENABLED", "mgr_turninterval": 1, "mgr_distance": 4,
                   "mgr_nationchance": 100, "mgr_worldchance": 50}
_MIGRATION_LOW = {"migration": "ENABLED", "mgr_turninterval": 100, "mgr_distance": 4,
                  "mgr_nationchance": 1, "mgr_worldchance": 0}


def _mv_setting(name: str, default: Any):
    def mv(sg, ctx):
        return _setting(sg, name, default)
    return mv


def _reveal(scope: str, n: int = 20, max_known: float = 0.9):
    """Reveal map tiles to one civ.

    ⚠️ Guarded on how much the civ already knows.  Measured at a turn-235
    checkpoint the target was already at 99.5% explored, so "reveal everything"
    changed nothing and the arm looked like a failed operator instead of an
    inapplicable one.  Returning "" records it as NA, which is the truth and
    shows up in the coverage statistics.
    """
    def f(sg: Savegame, ctx: Ctx) -> str:
        pid = ctx.target_pid
        from .geo import explored_fraction
        if explored_fraction(ctx.mi, ctx.target_name) >= max_known:
            return ""
        plane, bit = f"k{pid // 4:02d}", pid % 4
        mi = ctx.mi
        seen = mi.known.get(ctx.target_name)
        count = 0
        for y in range(mi.ysize):
            for x in range(mi.xsize):
                if scope == "all":
                    sg.set_bitplane_bit("map", plane, y, x, bit, True)
                    count += 1
                    continue
                if seen is None or seen[y][x] or count >= n:
                    continue
                # placebo: only tiles already touching known ground, so the civ
                # learns nothing it could not have scouted next turn
                adj = any(0 <= y + dy < mi.ysize and 0 <= x + dx < mi.xsize
                          and seen[y + dy][x + dx]
                          for dy in (-1, 0, 1) for dx in (-1, 0, 1))
                if not adj:
                    continue
                sg.set_bitplane_bit("map", plane, y, x, bit, True)
                count += 1
        # v2: an edge placebo that found nothing to reveal is NA, not a valid arm.
        return (f"revealed {count} tiles to {ctx.target_name} ({scope})"
                if count else "")
    return f


def _climate(n: int):
    def f(ctx: Ctx) -> list[str]:
        return [f'log.normal("CLIMATE=%s", tostring(pcall(edit.climate_change, '
                f'edit.GLOBAL_WARMING, {n})))']
    return f


# --------------------------------------------------------------------------
# L4 -- mechanism
# --------------------------------------------------------------------------

def _trait_mod(traits: list[str], amount: int):
    """`edit.trait_mod` is the official handle and writes the `mod` column.

    §2.5.6 established `effective = val + mod`; the server protects `val`.
    Verified here: the write persists and is not clamped (`Aggressive=50+40`
    survived, and even `+200` was stored verbatim).

    ⚠️ But persistence is not effect.  The pilot measured this operator's
    distance from factual as **exactly 0.0 at every horizon out to +200** on a
    tier-S world, while `L1-gold` and `L4-love` on the same checkpoint moved.
    Two confounds are known and neither is resolved yet:

      * §2.5.2's large measured effect (4 vs 8 cities over 120 turns) pushed
        **two** traits, on a **tier-M** world, by writing `val` -- not one trait
        via `mod` on a 3-civ map.
      * n=2 per cell in the pilot; a single small world cannot separate a weak
        effect from none.

    So the operator now pushes the same pair §2.5.2 did, through the official
    interface.  Whether trait interventions move macro outcomes is a question for
    the scaled run, where each operator gets one replicate per world.
    """
    def f(ctx: Ctx) -> list[str]:
        return [_lua_for_player(ctx.target_pid,
                                f'edit.trait_mod(p, "{t}", {amount})',
                                f"TRAIT_{t}")
                for t in traits]
    return f


def _set_love(value: int, delta: bool = False, flip: bool = False,
              hostile_only: bool = False):
    """Rewrite one civ's regard for another.

    ★ The engine is ASYMMETRIC, and this cost 42% of the operator's arms on the
    full corpus.  Reading the inert arms back showed a clean pattern:

        -1000 -> +1000   inert  (turning hatred into affection changes nothing)
        +116  -> -1000   works  (turning affection into hatred starts wars)

    Which makes sense: an AI does not stop a war because it suddenly likes someone
    -- the armies, the borders and the territorial claims are all still there --
    but it will start one because it suddenly does not.

    So `flip` (drive to whichever extreme is opposite) was the wrong design: half
    its draws were the useless direction.  `hostile_only` always drives toward
    hostility and reports NA when the pair is already hostile, which is the honest
    encoding of "this operator has nothing left to do here".
    """
    def f(sg: Savegame, ctx: Ctx) -> str:
        sec = _sec(ctx)
        ai = sg.table(sec, "ai")
        if ai is None or ctx.other_pid >= len(ai.rows):
            return ""
        before = int(ai.rows[ctx.other_pid][ai.col("love")])
        if hostile_only:
            if before <= -900:
                return ""          # already hostile: nothing to move
            new = -1000
        elif flip:
            new = 1000 if before < 0 else -1000
        elif delta:
            new = max(-1000, min(1000, before + value))
        else:
            new = value
        if new == before:
            return ""
        sg.set_cell(sec, "ai", ctx.other_pid, "love", new)
        return f"{ctx.target_name} love toward {ctx.other_name} {before} -> {new}"
    return f


def _barbarians(where: str, n: int = 3):
    """Unleash barbarians on `n` tiles.

    ★ Gated on the ruleset's barbarian onset turn.  civ2civ3 sets
    `onsetbarbs = 120`, and before that turn `edit.unleash_barbarians` is a
    complete no-op -- the arm comes out byte-identical to factual.  The pilot put
    branch points at t25/t95/t165/t235, so **half its barbarian arms did nothing
    at all**, which is why the measured effect came out at 0.48, i.e. below its own
    placebo.  That number was an average over 50% empty interventions, not a
    statement about the operator.

    One spawn was also too weak to separate from a remote placebo, so the shock is
    three tiles around the anchor rather than one.
    """
    def f(ctx: Ctx) -> list[str]:
        onset = 120
        st = ctx.sg.table("settings", "set")
        if st is not None:
            for r in st.as_dicts():
                if str(r.get("name")) == "onsetbarbs":
                    try:
                        onset = int(r.get("value"))
                    except (TypeError, ValueError):
                        pass
        if ctx.ws.turn < onset:
            return []                      # recorded as NA, which is the truth
        base = _capital_xy(ctx) if where == "capital" else _remote_water(ctx)
        if base is None:
            return []
        bx, by = base
        # a small ring around the anchor, clamped into the map
        offs = [(0, 0), (2, 1), (-2, -1), (1, -2), (-1, 2)][:n]
        out = []
        for dx, dy in offs:
            x = max(0, min(ctx.mi.xsize - 1, bx + dx))
            y = max(0, min(ctx.mi.ysize - 1, by + dy))
            out.append(f'log.normal("BARB=%s", tostring(pcall('
                       f'edit.unleash_barbarians, find.tile({x},{y}))))')
        return out
    return f


def _teleport(distance: str, whole_army: bool = True):
    """Relocate the civ's units.

    Teleporting a single unit measured 1.19 -- barely above its one-tile placebo,
    which is unsurprising: one displaced unit out of thirty changes nothing the
    resolved variables can see.  Moving the whole army onto a rival's capital is a
    real change of the strategic situation; the placebo shifts the same army by
    one tile, so the *act* of mass relocation is controlled for.
    """
    def f(ctx: Ctx) -> list[str]:
        p = ctx.ws.players[ctx.target_pid]
        if not p.units:
            return []
        if distance == "far":
            other = ctx.ws.players[ctx.other_pid]
            if not other.cities:
                return []
            tgt = max(other.cities, key=lambda c: c.size)
            dest = f"find.tile({tgt.x},{tgt.y})"
        else:
            dest = "find.tile(un.tile.x + 1, un.tile.y)"
        stop = "" if whole_army else " break"
        body = (f'for un in p:units_iterate() do '
                f'edit.unit_teleport(un, {dest}){stop} end')
        return [_lua_for_player(ctx.target_pid, body, f"TELEPORT_{distance}")]
    return f


def _set_goal(kind: str):
    """Rewrite the long-term research goal.

    The real arm points the civ at a military branch; the placebo points it at a
    neighbouring tech in the branch it was already pursuing, so the *act* of
    rewriting the goal is controlled for but the strategic direction is not
    changed.
    """
    MILITARY = ["Feudalism", "Chivalry", "Gunpowder", "Bronze Working", "Iron Working"]

    def f(sg: Savegame, ctx: Ctx) -> str:
        r = sg.table("research", "r")
        if r is None or ctx.target_pid >= len(r.rows):
            return ""
        tv = [str(x) for x in sg.get("savefile", "technology_vector", [])]
        row = dict(zip(r.columns, r.rows[ctx.target_pid]))
        before = str(row.get("goal_name", ""))
        if kind == "military":
            pick = next((t for t in MILITARY if t in tv and t != before), None)
        else:
            now = str(row.get("now_name", ""))
            pick = now if now and now != before else None
        if not pick:
            return ""
        sg.set_cell("research", "r", ctx.target_pid, "goal_name", pick)
        return f"{ctx.target_name} research goal {before} -> {pick} ({kind})"
    return f


def _set_urgency(value: int, delta: bool = False):
    def f(sg: Savegame, ctx: Ctx) -> str:
        sec = _sec(ctx)
        c = sg.table(sec, "c")
        if c is None or not c.rows:
            return ""
        j = c.col("ai.urgency")
        for i in range(len(c.rows)):
            before = int(c.rows[i][j])
            sg.set_cell(sec, "c", i, "ai.urgency", before + value if delta else value)
        return f"{ctx.target_name} ai.urgency on {len(c.rows)} cities -> " \
               f"{'+' if delta else '='}{value}"
    return f


# --------------------------------------------------------------------------
# gate-4 verifiers
# --------------------------------------------------------------------------

def _moved(reader):
    def vf(before: Savegame, after: Savegame, ctx: Ctx):
        b, a = reader(before, ctx), reader(after, ctx)
        return a != b, f"{b} -> {a}"
    return vf


def _increased(reader):
    def vf(before: Savegame, after: Savegame, ctx: Ctx):
        b, a = reader(before, ctx), reader(after, ctx)
        try:
            return a > b, f"{b} -> {a}"
        except TypeError:
            return False, f"uncomparable {b} -> {a}"
    return vf


def _decreased(reader):
    def vf(before: Savegame, after: Savegame, ctx: Ctx):
        b, a = reader(before, ctx), reader(after, ctx)
        try:
            return a < b, f"{b} -> {a}"
        except TypeError:
            return False, f"uncomparable {b} -> {a}"
    return vf


def _became_true(reader):
    def vf(before: Savegame, after: Savegame, ctx: Ctx):
        b, a = reader(before, ctx), reader(after, ctx)
        return (not b) and bool(a), f"{b} -> {a}"
    return vf


# --------------------------------------------------------------------------
# the library
# --------------------------------------------------------------------------

def _op(**kw) -> Op:
    return Op(**kw)


REAL_OPS: list[Op] = [
    # ---- L1: stock.  Kept as the lower bound an operator must beat. ----
    _op(id="L1-gold", score_vars=("gold",), layer="L1", mutate=_add_gold(500),
        cond_text=lambda c: f"{c.target_name} 的国库突然增加了 500 金",
        expected_path=["国库", "购买单位/建筑", "短期产出"],
        horizon_band=(5, 30), mechanism_var=mv_gold, verify=_moved(mv_gold),
        notes="measured: H30 diff 487, H150 diff 10 -- absorbed"),
    _op(id="L1-gold-neg", score_vars=("gold",), layer="L1", mutate=_set_gold(0),
        cond_text=lambda c: f"{c.target_name} 的国库被清空",
        expected_path=["破产", "建筑失修", "产能下降"],
        horizon_band=(5, 30), mechanism_var=mv_gold, verify=_moved(mv_gold)),
    _op(id="L1-gold-swing", score_vars=("gold",), layer="L1", mutate=_gold_swing(True),
        cond_text=_gold_swing_text(True),
        expected_path=["国库", "购买单位/建筑 或 破产失修", "短期产出"],
        horizon_band=(5, 30), mechanism_var=mv_gold, verify=_moved(mv_gold),
        notes="merged L1-gold (+500, even branch turn) and L1-gold-neg (emptied, odd turn); v2 bench uses this one"),
    _op(id="L1-bulbs", score_vars=("n_techs",), layer="L1", lua=_give_bulbs(400, 120),
        cond_text=lambda c: f"{c.target_name} 的研究突然取得重大进展",
        expected_path=["研究点", "立刻完成一项科技", "科技数"],
        horizon_band=(5, 60), mechanism_var=mv_techs, verify=_increased(mv_techs)),

    # ---- L2: constraint ----
    _op(id="L2-city-kill", score_vars=("population", "score"), score_by="termination", layer="L2", mutate=_kill_city("largest", 0.5),
        cond_text=lambda c: f"{c.target_name} 一半的城市被摧毁（最大的那些）",
        expected_path=["城市", "产能与人口", "军力", "战争结果"],
        horizon_band=(5, 150), mechanism_var=mv_total_city_size,
        verify=_decreased(mv_total_city_size)),
    _op(id="L2-gov-indirect", score_vars=("government",), layer="L2", lua=_give_gov_tech,
        cond_text=lambda c: f"{c.target_name} 提前掌握了一种新政体的知识",
        expected_path=["科技", "可选政体集扩大", "AI 自主决策", "政体改变"],
        horizon_band=(5, 60), mechanism_var=mv_holds_gov_tech,
        verify=_became_true(mv_holds_gov_tech),
        notes="★ §2.5.5b: pulled the Monarchy switch 30 turns earlier. "
              "Score on the government path, NOT on far-horizon score -- the "
              "placebo landed at the same far-horizon value."),

    # ---- L3: endowment ----
    _op(id="L3-res-add", score_vars=("population", "score"), layer="L3", mutate=_add_resources("worked", n=40),
        cond_text=lambda c: f"{c.target_name} 的领土上出现了新的资源",
        expected_path=["资源", "耕作产出", "人口与产能", "全面国力"],
        horizon_band=(30, 300), mechanism_var=mv_worked_resources,
        verify=_increased(mv_total_resources)),
    _op(id="L3-res-strip", score_vars=("population", "score"), layer="L3",
        mutate=_strip_resources("owned", limit=60),
        cond_text=lambda c: f"{c.target_name} 领土上已开发地块的资源与基础设施被剥除",
        expected_path=["资源与基础设施", "耕作产出", "人口与产能"],
        horizon_band=(30, 300), mechanism_var=mv_worked_resources,
        verify=_decreased(mv_total_resources),
        notes="v2: restored from DROPPED with the placebo back at limit=30 (v1 "
              "strengthened both arms together and could not tell them apart)"),
    _op(id="L3-wonder", score_vars=("culture", "score"), layer="L3",
        mutate=_grant_improvements(["Great Library", "Pyramids", "Colossus"]),
        cond_text=lambda c: f"{c.target_name} 建成了三座大奇观",
        expected_path=["奇观", "全局效应", "长期科技/文化优势"],
        horizon_band=(30, 300), mechanism_var=mv_great_library,
        verify=_became_true(mv_great_library),
        notes="v2: three wonders instead of one -- one Great Library moved "
              "culture/score by 0.002 normalised units, below output granularity"),
    _op(id="L3-fog", score_vars=("explored",), layer="L3", mutate=_reveal("all"),
        cond_text=lambda c: f"{c.target_name} 突然认识了整个世界的地图",
        expected_path=["地图认知", "扩张与外交决策", "扩张方向"],
        # ⚠️ measured ABSORBED at a T120 checkpoint: the civ explores the whole
        # map on its own within 20 turns, so factual catches up.  Attributable
        # only from an early checkpoint, before natural exploration completes.
        horizon_band=(5, 60), mechanism_var=mv_explored,
        verify=_increased(mv_explored)),

    # ---- L4: mechanism.  The layer whose effect accumulates. ----
    _op(id="L4-trait", score_vars=("n_cities", "n_units"), layer="L4", lua=_trait_mod(["Aggressive", "Expansionist"], 45),
        cond_text=lambda c: f"{c.target_name} 的统治集团变得远为好战",
        expected_path=["人格", "扩张与宣战倾向", "城市数与军力", "国力"],
        horizon_band=(50, 300), mechanism_var=mv_trait_aggr, verify=_moved(mv_trait_aggr),
        notes="★★ §2.5.2 measured 120 turns, cities 4 vs 8, score 80 vs 123, "
              "monotone -- on tier-M. ⚠️ the tier-S pilot measured effect 0.0 at "
              "every horizon; unresolved, see _trait_mod"),
    # ★★ 2026-09-14 (user decision): four SINGLE-dimension trait operators.
    #
    # Why they did not exist before: `L4-trait` pushes **Aggressive AND Expansionist
    # together** (a compromise -- the pilot measured a single dimension via `mod` as
    # effect 0.0 on one tier-S world with n=2, while §2.5.2's large effect had come from
    # a two-trait, tier-M, write-`val` configuration; see _trait_mod).  `Trader` and
    # `Builder` never had operators at all, because in v1 they were structurally inert
    # (Builder integer division, Trader no effect surface) -- both fixed by
    # v2-builder.patch / v2-engine.patch, but nobody went back to add the operators.
    #
    # The bundle destroys this operator's most informative reading: its two scored
    # variables map one-per-dimension (n_military_units <- Aggressive, n_cities <-
    # Expansionist), so bundled you can only read "did anything move", never "does the
    # model know which dimension drives which quantity".  These four restore the
    # **directional dissociation** test: each intervention has its own expected variable
    # AND its own expected sign.
    #
    # Each is gated by the same criterion as the pilot: real vs matched placebo on the
    # TARGET civ only, relative effect >= 5% and signal/noise >= 3x.  A dimension that
    # fails stays out and is written up as a negative result.
    _op(id="L4-trait-aggr", score_vars=("n_military_units", "n_enemies"), layer="L4",
        lua=_trait_mod(["Aggressive"], 45),
        cond_text=lambda c: f"{c.target_name} 的统治集团变得远为好战（只改这一维）",
        expected_path=["Aggressive", "宣战与拒和", "军队规模与敌国数"],
        horizon_band=(50, 300), mechanism_var=mv_trait("Aggressive"), verify=_moved(mv_trait("Aggressive")),
        notes="single-dim split of L4-trait; expected: military up, cities ~unchanged"),
    _op(id="L4-trait-expa", score_vars=("n_cities", "city_size_total"), layer="L4",
        lua=_trait_mod(["Expansionist"], 45),
        cond_text=lambda c: f"{c.target_name} 的统治集团变得远为热衷开疆（只改这一维）",
        expected_path=["Expansionist", "建城意愿", "城市数与城市总规模"],
        horizon_band=(50, 300), mechanism_var=mv_trait("Expansionist"), verify=_moved(mv_trait("Expansionist")),
        notes="single-dim split of L4-trait; expected: cities up, military ~unchanged"),
    _op(id="L4-trait-trade", score_vars=("gold", "n_techs"), layer="L4",
        lua=_trait_mod(["Trader"], 45),
        cond_text=lambda c: f"{c.target_name} 的统治集团变得远为重商（只改这一维）",
        expected_path=["Trader", "商队与贸易路线", "金库与科研"],
        horizon_band=(50, 300), mechanism_var=mv_trait("Trader"), verify=_moved(mv_trait("Trader")),
        notes="v2-engine.patch fixed the Trader integer division; never had an operator before"),
    _op(id="L4-trait-build", score_vars=("city_size_total", "n_military_units"), layer="L4",
        lua=_trait_mod(["Builder"], 45),
        cond_text=lambda c: f"{c.target_name} 的统治集团变得远为偏重营建（只改这一维）",
        expected_path=["Builder", "建筑优先于单位", "城市规模上升、军队下降"],
        horizon_band=(50, 300), mechanism_var=mv_trait("Builder"), verify=_moved(mv_trait("Builder")),
        notes="v2-builder.patch fixed the Builder integer division (Spearman +0.36 after); "
              "never had an operator before. ⚠️ expected sign on n_military_units is NEGATIVE"),
    # ★ 2026-09-15 剂量反应：与 L4-trait-aggr 同锚点、同维、mod +90（安慰剂复用 +3 臂）。
    _op(id="L4-trait-aggr90", score_vars=('n_military_units', 'n_enemies'), layer="L4",
        lua=_trait_mod(["Aggressive"], 90),
        cond_text=lambda c: f"{c.target_name} 的统治集团变得远为好战（只改这一维）",
        expected_path=["Aggressive", "同 L4-trait-aggr", "剂量 +90"],
        horizon_band=(50, 300), mechanism_var=mv_trait("Aggressive"), verify=_moved(mv_trait("Aggressive")),
        notes="dose-response twin of L4-trait-aggr: +90 instead of +45"),
    # ★ 2026-09-15 剂量反应：与 L4-trait-expa 同锚点、同维、mod +90（安慰剂复用 +3 臂）。
    _op(id="L4-trait-expa90", score_vars=('n_cities', 'city_size_total'), layer="L4",
        lua=_trait_mod(["Expansionist"], 90),
        cond_text=lambda c: f"{c.target_name} 的统治集团变得远为热衷开疆（只改这一维）",
        expected_path=["Expansionist", "同 L4-trait-expa", "剂量 +90"],
        horizon_band=(50, 300), mechanism_var=mv_trait("Expansionist"), verify=_moved(mv_trait("Expansionist")),
        notes="dose-response twin of L4-trait-expa: +90 instead of +45"),
    # ★ 2026-09-15 剂量反应：与 L4-trait-trade 同锚点、同维、mod +90（安慰剂复用 +3 臂）。
    _op(id="L4-trait-trade90", score_vars=('gold', 'n_techs'), layer="L4",
        lua=_trait_mod(["Trader"], 90),
        cond_text=lambda c: f"{c.target_name} 的统治集团变得远为重商（只改这一维）",
        expected_path=["Trader", "同 L4-trait-trade", "剂量 +90"],
        horizon_band=(50, 300), mechanism_var=mv_trait("Trader"), verify=_moved(mv_trait("Trader")),
        notes="dose-response twin of L4-trait-trade: +90 instead of +45"),
    # ★ 2026-09-15 剂量反应：与 L4-trait-build 同锚点、同维、mod +90（安慰剂复用 +3 臂）。
    _op(id="L4-trait-build90", score_vars=('city_size_total', 'n_military_units'), layer="L4",
        lua=_trait_mod(["Builder"], 90),
        cond_text=lambda c: f"{c.target_name} 的统治集团变得远为偏重营建（只改这一维）",
        expected_path=["Builder", "同 L4-trait-build", "剂量 +90"],
        horizon_band=(50, 300), mechanism_var=mv_trait("Builder"), verify=_moved(mv_trait("Builder")),
        notes="dose-response twin of L4-trait-build: +90 instead of +45"),
    # P5 2026-09-09: at_war alone never moved (69% Δ≠0, effect 0); the love edit does move
    # love[target->other] (97%), the pair's diplstate (28% vs placebo 1%), n_enemies (44% vs 11%),
    # n_allies (28% vs 2%).  Score on the relation counts.
    _op(id="L4-love", score_vars=("n_enemies", "n_allies", "n_peace"), layer="L4", mutate=_set_love(0, hostile_only=True),
        cond_text=lambda c: f"{c.target_name} 对 {c.other_name} 产生了不可调和的敌意",
        expected_path=["态度", "外交动作", "战和", "发展路径"],
        horizon_band=(50, 300), mechanism_var=mv_love, verify=_moved(mv_love),
        notes="★ §2.5.3: forcing +1000 turned War into Cease-fire for 14+ turns. "
              "⚠️ but at scale the +1000 direction is mostly inert (42% of arms); "
              "only the hostile direction bites. Now hostile_only."),

    # ---- WP: world parameters (v2).  do(setting) on reload; see Op.settings ----
    _op(id="WP-disasters", score_vars=("n_improvements", "population"), layer="WP",
        mutate=_wp_scale("disasters", 3.0, 10, 0, 1000)[0],
        settings=_wp_scale("disasters", 3.0, 10, 0, 1000)[1],
        cond_text=lambda c: "灾害发生频率变为原来的 3 倍",
        expected_path=["灾害频率", "建筑损毁/库存清空/人口损失", "发展速度"],
        horizon_band=(30, 300), mechanism_var=_mv_setting("disasters", 10)),
    # ⚠️ `foodbox` (and sciencebox, razechance, globalwarming_percent) are
    # SSET_RULES: locked once the game has started, so a mid-game `set` is a
    # silent no-op (verify_ops 2026-09-09: REVERTED).  Only SSET_RULES_FLEXIBLE
    # settings can be operators: disasters, barbarians/onsetbarbs, migration/mgr_*,
    # civilwarsize, revolen, contactturns, restrictinfra, unreachableprotects.
    _op(id="WP-migration", score_vars=("max_city_size", "total_city_size", "population"), layer="WP",
        mutate=_wp_multi(_MIGRATION_HIGH, _MIGRATION_DEFAULTS)[0],
        settings=_wp_multi(_MIGRATION_HIGH, _MIGRATION_DEFAULTS)[1],
        cond_text=lambda c: "居民迁徙被开启并设为高频：每回合小城的居民都有很大概率迁往更宜居的城市（包括他国）",
        expected_path=["迁徙频率", "城市规模重新分布（大城更大、小城萎缩）", "人口与产能格局"],
        horizon_band=(30, 300), mechanism_var=_mv_setting("mgr_nationchance", 50)),
    _op(id="WP-barbarians", score_vars=("units_lost", "n_cities"), layer="WP",
        mutate=_wp_set("barbarians", "HORDES", "NORMAL")[0],
        settings=_wp_set("barbarians", "HORDES", "NORMAL")[1],
        cond_text=lambda c: "蛮族出现的频率变为最高档（原来的数倍）",
        expected_path=["蛮族入侵频率", "单位损失与边境城市", "军费"],
        horizon_band=(30, 300), mechanism_var=_mv_setting("barbarians", "NORMAL")),
]

# ---- matched placebos: same edit path, mechanism deliberately not engaged ----
PLACEBOS: list[Op] = [
    _op(id="L1-gold__placebo", score_vars=("gold",), layer="L1", placebo_of="L1-gold", placebo_kind="magnitude", mutate=_add_gold(5),
        cond_text=lambda c: f"{c.target_name} 的国库增加了 5 金",
        expected_path=["国库（量级可忽略）"], horizon_band=(5, 30),
        mechanism_var=mv_gold),
    _op(id="L1-gold-neg__placebo", score_vars=("gold",), layer="L1", placebo_of="L1-gold-neg", placebo_kind="magnitude",
        mutate=_add_gold(-5),
        cond_text=lambda c: f"{c.target_name} 的国库减少了 5 金",
        expected_path=["国库（量级可忽略）"], horizon_band=(5, 30),
        mechanism_var=mv_gold),
    _op(id="L1-gold-swing__placebo", score_vars=("gold",), layer="L1", placebo_of="L1-gold-swing", placebo_kind="magnitude",
        mutate=_gold_swing(False), cond_text=_gold_swing_text(False),
        expected_path=["国库（量级可忽略）"], horizon_band=(5, 30), mechanism_var=mv_gold),
    _op(id="L1-bulbs__placebo", score_vars=("n_techs",), layer="L1", placebo_of="L1-bulbs", placebo_kind="type", lua=_give_bulbs(5, 0),
        cond_text=lambda c: f"{c.target_name} 的研究略有进展",
        expected_path=["研究点（不足以完成任何科技）"], horizon_band=(5, 60),
        mechanism_var=mv_techs),

    _op(id="L2-city-kill__placebo", score_vars=("population", "score"), score_by="termination", layer="L2", placebo_of="L2-city-kill", placebo_kind="magnitude",
        mutate=_kill_city("smallest", 0.5),
        cond_text=lambda c: f"{c.target_name} 一半的城市被摧毁（最小的那些）",
        expected_path=["城市数（边际城市，产能损失极小）"], horizon_band=(5, 150),
        mechanism_var=mv_total_city_size),
    _op(id="L2-gov-indirect__placebo", score_vars=("government",), layer="L2", placebo_of="L2-gov-indirect", placebo_kind="type",
        lua=_give_tech(NEUTRAL_TECHS),
        cond_text=lambda c: f"{c.target_name} 提前掌握了一项与政体无关的技术",
        expected_path=["科技（不解锁任何政体）"], horizon_band=(5, 60),
        mechanism_var=mv_holds_gov_tech,
        notes="✅ the placebo that refuted the far-horizon story in §2.5.5b"),

    _op(id="L3-res-add__placebo", score_vars=("population", "score"), layer="L3", placebo_of="L3-res-add", placebo_kind="type",
        mutate=_add_resources("nowhere", n=40),
        cond_text=lambda c: "世界上某些无人区出现了新的资源",
        expected_path=["资源（无人可耕作）"], horizon_band=(30, 300),
        mechanism_var=mv_worked_resources),
    _op(id="L3-res-strip__placebo", score_vars=("population", "score"), layer="L3", placebo_of="L3-res-strip", placebo_kind="type",
        mutate=_strip_resources("unowned", limit=30),
        cond_text=lambda c: "无人区的一些资源消失了",
        expected_path=["资源（无人耕作）"], horizon_band=(30, 300),
        mechanism_var=mv_worked_resources),
    _op(id="L3-wonder__placebo", score_vars=("culture", "score"), layer="L3", placebo_of="L3-wonder", placebo_kind="type",
        mutate=_grant_improvements(["Barracks", "Granary", "Temple"]),
        cond_text=lambda c: f"{c.target_name} 建成了三座普通建筑",
        expected_path=["建筑（只有本城效应，无全局效应）"], horizon_band=(30, 300),
        mechanism_var=mv_great_library),
    _op(id="L3-fog__placebo", score_vars=("explored",), layer="L3", placebo_of="L3-fog", placebo_kind="magnitude", mutate=_reveal("edge", 20),
        cond_text=lambda c: f"{c.target_name} 探明了边界外的一小片地区",
        expected_path=["地图认知（20 格，本就相邻）"], horizon_band=(30, 300),
        mechanism_var=mv_explored),

    _op(id="L4-trait__placebo", score_vars=("n_cities", "n_units"), layer="L4", placebo_of="L4-trait", placebo_kind="magnitude",
        lua=_trait_mod(["Aggressive", "Expansionist"], 3),
        cond_text=lambda c: f"{c.target_name} 的统治集团略微强硬了一些",
        expected_path=["人格（+3，可忽略）"], horizon_band=(50, 300),
        mechanism_var=mv_trait_aggr),
    _op(id="L4-trait-aggr__placebo", score_vars=("n_military_units", "n_enemies"), layer="L4",
        placebo_of="L4-trait-aggr", placebo_kind="magnitude", lua=_trait_mod(["Aggressive"], 3),
        cond_text=lambda c: f"{c.target_name} 的统治集团略微强硬了一些",
        expected_path=["Aggressive（+3，可忽略）"], horizon_band=(50, 300),
        mechanism_var=mv_trait("Aggressive")),
    _op(id="L4-trait-expa__placebo", score_vars=("n_cities", "city_size_total"), layer="L4",
        placebo_of="L4-trait-expa", placebo_kind="magnitude", lua=_trait_mod(["Expansionist"], 3),
        cond_text=lambda c: f"{c.target_name} 的统治集团略微更想开疆了一些",
        expected_path=["Expansionist（+3，可忽略）"], horizon_band=(50, 300),
        mechanism_var=mv_trait("Expansionist")),
    _op(id="L4-trait-trade__placebo", score_vars=("gold", "n_techs"), layer="L4",
        placebo_of="L4-trait-trade", placebo_kind="magnitude", lua=_trait_mod(["Trader"], 3),
        cond_text=lambda c: f"{c.target_name} 的统治集团略微更重商了一些",
        expected_path=["Trader（+3，可忽略）"], horizon_band=(50, 300),
        mechanism_var=mv_trait("Trader")),
    _op(id="L4-trait-build__placebo", score_vars=("city_size_total", "n_military_units"), layer="L4",
        placebo_of="L4-trait-build", placebo_kind="magnitude", lua=_trait_mod(["Builder"], 3),
        cond_text=lambda c: f"{c.target_name} 的统治集团略微更偏重营建了一些",
        expected_path=["Builder（+3，可忽略）"], horizon_band=(50, 300),
        mechanism_var=mv_trait("Builder")),
    _op(id="L4-trait-aggr90__placebo", score_vars=('n_military_units', 'n_enemies'), layer="L4",
        placebo_of="L4-trait-aggr90", placebo_kind="magnitude", lua=_trait_mod(["Aggressive"], 3),
        cond_text=lambda c: f"{c.target_name} 的统治集团略微强硬了一些",
        expected_path=["Aggressive（+3，可忽略）"], horizon_band=(50, 300),
        mechanism_var=mv_trait("Aggressive")),
    _op(id="L4-trait-expa90__placebo", score_vars=('n_cities', 'city_size_total'), layer="L4",
        placebo_of="L4-trait-expa90", placebo_kind="magnitude", lua=_trait_mod(["Expansionist"], 3),
        cond_text=lambda c: f"{c.target_name} 的统治集团略微更想开疆了一些",
        expected_path=["Expansionist（+3，可忽略）"], horizon_band=(50, 300),
        mechanism_var=mv_trait("Expansionist")),
    _op(id="L4-trait-trade90__placebo", score_vars=('gold', 'n_techs'), layer="L4",
        placebo_of="L4-trait-trade90", placebo_kind="magnitude", lua=_trait_mod(["Trader"], 3),
        cond_text=lambda c: f"{c.target_name} 的统治集团略微更重商了一些",
        expected_path=["Trader（+3，可忽略）"], horizon_band=(50, 300),
        mechanism_var=mv_trait("Trader")),
    _op(id="L4-trait-build90__placebo", score_vars=('city_size_total', 'n_military_units'), layer="L4",
        placebo_of="L4-trait-build90", placebo_kind="magnitude", lua=_trait_mod(["Builder"], 3),
        cond_text=lambda c: f"{c.target_name} 的统治集团略微更偏重营建了一些",
        expected_path=["Builder（+3，可忽略）"], horizon_band=(50, 300),
        mechanism_var=mv_trait("Builder")),
    _op(id="L4-love__placebo", score_vars=("n_enemies", "n_allies", "n_peace"), layer="L4", placebo_of="L4-love", placebo_kind="magnitude",
        mutate=_set_love(-30, delta=True),
        cond_text=lambda c: f"{c.target_name} 对 {c.other_name} 稍有不满",
        expected_path=["态度（−30，不足以改变外交）"], horizon_band=(50, 300),
        mechanism_var=mv_love),

    _op(id="WP-disasters__placebo", score_vars=("n_improvements", "population"), layer="WP", placebo_of="WP-disasters", placebo_kind="magnitude",
        mutate=_wp_scale("disasters", 1.1, 10, 0, 1000)[0],
        settings=_wp_scale("disasters", 1.1, 10, 0, 1000)[1],
        cond_text=lambda c: "灾害发生频率变为原来的 1.1 倍",
        expected_path=["灾害频率（可忽略）"], horizon_band=(30, 300),
        mechanism_var=_mv_setting("disasters", 10)),
    _op(id="WP-migration__placebo", score_vars=("max_city_size", "total_city_size", "population"), layer="WP", placebo_of="WP-migration", placebo_kind="magnitude",
        mutate=_wp_multi(_MIGRATION_LOW, _MIGRATION_DEFAULTS)[0],
        settings=_wp_multi(_MIGRATION_LOW, _MIGRATION_DEFAULTS)[1],
        cond_text=lambda c: "居民迁徙被开启，但概率设得极低（几乎不会发生）",
        expected_path=["迁徙（可忽略）"], horizon_band=(30, 300),
        mechanism_var=_mv_setting("mgr_nationchance", 50)),
    _op(id="WP-barbarians__placebo", score_vars=("units_lost", "n_cities"), layer="WP", placebo_of="WP-barbarians", placebo_kind="magnitude",
        mutate=_wp_set("onsetbarbs", 9999, 60)[0],
        settings=_wp_set("onsetbarbs", 9999, 60)[1],
        cond_text=lambda c: "蛮族的出现时间被推迟到很久以后",
        expected_path=["蛮族（实际上不再出现）"], horizon_band=(30, 300),
        mechanism_var=_mv_setting("barbarians", "NORMAL"),
        notes="type-ish placebo: same lever family, opposite/negligible direction"),
]

# --------------------------------------------------------------------------
# dropped after two screening rounds (2026-08-26)
# --------------------------------------------------------------------------
# Each of these cleared all four gates -- they load, run, hold their edit -- and
# was still removed, because clearing the gates only says an operator *does
# something*.  What the library needs is an operator that does something its own
# matched placebo does not, inside the window it claims to act in.  Measured with
# n=300 in the first round and again after the magnitude of each was increased:
#
#   operator        round 1   round 2 (strengthened)      reading
#   L1-tech            0.99   0.99  (4 techs, scored on   the AI does not turn
#                                    n_units/units_built)  military techs into
#                                                          army any faster
#   L4-barbarians      0.48   0.98  (onsetbarbs guard +   barbarians anywhere
#                                    3 tiles)              roam; a remote spawn
#                                                          costs comparable units
#   L2-diplo           1.31   1.24  (flip instead of      never reached the
#                                    forcing Peace)        strong band
#   L4-teleport        1.19   1.24  (whole army)          same
#   L2-unit-wipe       1.34   n/a   (already deletes all) same
#   L3-res-strip       0.54   0.26  (+ infrastructure)    ** CONFOUNDED **
#
# WARNING on L3-res-strip: its 0.26 is NOT evidence the operator is inert.  The
# placebo's `limit` was raised 30 -> 60 in the same change, so BOTH arms were
# strengthened and the ratio fell for that reason.  A matched placebo may never be
# strengthened alongside its operator.  Restoring this operator means putting the
# placebo back first and re-measuring; until then the question is open.
#
# The implementations stay here on purpose.  A measured negative is as expensive
# to obtain as a positive, and the next person to consider "why not just raze a
# city / unleash barbarians / teleport an army" deserves the numbers rather than
# having to rediscover them.
DROPPED_OPS: dict[str, str] = {
    "L1-tech": "0.99 both rounds; military techs do not become army faster",
    "L1-tech__placebo": "0.99 both rounds; military techs do not become army faster",
    "L2-diplo": "1.24 after switching to a relation flip",
    "L2-diplo__placebo": "1.24 after switching to a relation flip",
    "L2-unit-wipe": "1.34; cannot be strengthened further",
    "L2-unit-wipe__placebo": "1.34; cannot be strengthened further",
    "L3-terrain": "v2: dropped -- 1.19 in v1, and at benchmark scale its true effect "
                  "(0.006 normalised) sat below any model's output granularity",
    "L3-terrain__placebo": "v2: dropped with L3-terrain",
    "L4-barbarians": "0.98 after the onsetbarbs guard removed the empty arms",
    "L4-barbarians__placebo": "0.98 after the onsetbarbs guard removed the empty arms",
    "L4-teleport": "1.24 with the whole army relocated",
    "L4-teleport__placebo": "1.24 with the whole army relocated",
}


# `empty` exists only as a pipeline self-check.  In a deterministic world it is
# byte-identical to the mainline continuation, which is exactly what makes it a
# strong assertion and a useless control.
EMPTY = _op(id="empty", layer="none", placebo_of="__pipeline__",
            mutate=lambda sg, ctx: "no change",
            cond_text=lambda c: "没有对世界做任何改动",
            expected_path=[], horizon_band=(0, 0), mechanism_var=lambda sg, c: None)

LIBRARY: list[Op] = REAL_OPS + PLACEBOS
BY_ID: dict[str, Op] = {o.id: o for o in LIBRARY + [EMPTY]}


# --------------------------------------------------------------------------
# replaying the v1 corpus
# --------------------------------------------------------------------------

# Operators whose implementation was **improved after the v1 corpus was already
# collected**.  Replaying one of those arms has to reproduce the intervention that
# actually produced it, not the better one: the arm's `resolved` scalars are
# already on disk, and if the replay intervenes differently, the harvested events
# describe a different counterfactual than the numbers they would be paired with.
# Nothing would error -- the effect sizes would still compute, on mismatched data.
#
# `L4-love` used `flip` (drive to whichever extreme is opposite).  Reading the
# full corpus back showed that half of its draws were the useless direction
# (`-1000 -> +1000` is inert: an AI does not call off a war because it suddenly
# likes someone), so the library moved to `hostile_only`.  The corpus therefore
# contains both directions -- measured 225 hostile / 112 friendly / 71 NA over 40
# worlds -- and only the old form reproduces it.
#
# This is deliberately not a blanket "pin the whole library": each entry is one
# named operator with a stated reason, and `harvest_arm_job` **verifies** the
# substitution by comparing the operator's own `applied` string against the one
# recorded at collection time.  A wrong entry here fails the gate rather than
# silently producing mismatched data.
_REPLAY_MUTATE: dict[str, Callable[[Savegame, Ctx], str]] = {
    "L4-love": _set_love(0, flip=True),
}


def replay_op(op_id: str) -> Op | None:
    """The operator as it existed when the v1 corpus was collected."""
    base = BY_ID.get(op_id)
    if base is None or op_id not in _REPLAY_MUTATE:
        return base
    return dataclasses.replace(base, mutate=_REPLAY_MUTATE[op_id])


def placebo_for(op_id: str) -> Op | None:
    return BY_ID.get(f"{op_id}__placebo")


def apply_op(op: Op, sg: Savegame, ctx: Ctx) -> tuple[str, list[str]]:
    """Returns (description, lua lines).  An empty description means the operator
    was not applicable to this checkpoint and the arm should be skipped rather
    than recorded as a no-op."""
    desc, lua = "", []
    if op.mutate is not None:
        desc = op.mutate(sg, ctx)
    if op.lua is not None:
        lua = list(op.lua(ctx))
        if not desc:
            desc = f"lua: {'; '.join(lua)[:160]}" if lua else ""
    return desc, lua


def apply_settings(op: Op, sg: Savegame, ctx: Ctx) -> dict[str, Any]:
    """Server settings this operator wants `set` after the checkpoint loads
    (world-parameter operators).  Empty for every save-editing operator."""
    return dict(op.settings(sg, ctx)) if op.settings is not None else {}


def op_to_json(op: Op, ctx: Ctx, desc: str) -> dict[str, Any]:
    return {
        "id": op.id, "layer": op.layer, "placebo_of": op.placebo_of,
        "target": ctx.target_name, "other": ctx.other_name,
        "applied": desc, "expected_path": op.expected_path,
        "horizon_band": list(op.horizon_band), "notes": op.notes,
    }
