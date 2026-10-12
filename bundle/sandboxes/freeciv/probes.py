"""Hidden state the engine maintains and the report never shows.

The report carries what the world *looks like*.  This module carries what the
engine *knows and intends* -- the variables a world model would have to infer
rather than read.  Nine groups, all read from the savegame, all verified present
on freeciv 3.1.6 / civ2civ3.  Design and per-field measurements are in
`docs/freeciv/探针状态与反事实算子设计.md` §1.1.

  A  episode-static hidden structure   trait vector, ai level/type, style
  B  attitudes toward others           ai.love and the diplomacy intent flags
  C  beliefs and memory                the `dc` table, embassies, exploration
  D  development intent                wonder city, per-city wants, research path
  E  event clocks                      last_war, turns_alive, per-city stamps
  F  global accumulators               warming/cooling, world tech set, RNG
  G  unit-level hidden fields          veteran level, transport graph, pending decisions
  H  city citizen layer                specialists and the citizen bitstring
  I  scoreboard quantities not in the report   happiness split, cumulative war losses

**Nothing here goes into any report variant.**  The moment a probe target is
rendered it stops being hidden and the probe measures nothing.

Three of these are the ones worth building tasks on, and the reasons are
measured, not assumed:

  * `A.trait` is resampled per episode under `traitdistribution EVEN`, is
    invisible in the report, has an exact 4-dimensional ground truth, and drives
    behaviour (§2.5.2: forcing it produced 4 vs 8 cities over 120 turns, an
    effect that *accumulated* rather than decayed).
  * `B.love` evolves continuously in [-1000, +1000] and causally moves diplomacy
    (§2.5.3: War -> Cease-fire for 14+ turns) -- but it is deliberately *not* a
    sufficient statistic for war: three pairs declared war at positive love.
    That is what makes it a non-trivial probe rather than a giveaway.
  * `C.dc` is the AI's memory of foreign cities and is 33-77% stale against the
    truth sitting in the same savegame.  ⚠️ §2.5.4 showed that poisoning it does
    not change AI behaviour, so it is a memory cache, not a decision input:
    valid as a probe target, dead as an intervention handle.

`I` is not in the original design; it was found while writing this module.
`[scoreN]` holds engine-computed civ aggregates the report does not render, and
`units_built` / `units_killed` / `units_lost` are *cumulative history* -- they
cannot be recovered from any single frame, which makes them a genuinely
different kind of probe target from the rest.
"""

from __future__ import annotations

import hashlib
from typing import Any

from .savegame import Savegame

# The `texai.*` mirror tables are the tex AI's copies of the classic AI's
# fields.  Every world we collect runs `ai_type="classic"`, so they are dead
# weight; recorded only as a presence flag rather than duplicated in full.
_SKIP_TABLE_PREFIXES = ("texai",)

_UNIT_PROBE_FIELDS = [
    "veteran", "homecity", "transported_by", "ferryboat", "server_side_agent",
    "action_decision", "action_decision_tile_x", "action_decision_tile_y",
    "born", "moved", "done_moving", "facing", "nationality", "fuel",
    "battlegroup", "carrying", "charge", "bodyguard", "ord_map", "ord_city",
    "activity_count", "activity_tgt", "paradropped", "stay",
]

_CITY_INTENT_FIELDS = [
    "ai.urgency", "ai.building_turn", "ai.building_wait", "ai.founder_turn",
    "ai.founder_want", "ai.founder_boat", "current_want",
    "changed_from_kind", "changed_from_name", "before_change_shields",
    "caravan_shields", "disbanded_shields", "last_turns_shield_surplus",
    "wl_length",
]

_CITY_CLOCK_FIELDS = [
    "turn_last_built", "turn_plague", "anarchy", "rapture", "was_happy",
    "history", "did_buy", "did_sell", "steal", "airlift", "original",
]

# Per-city fields with a fixed name.  The nationality columns are NOT here: see
# `_citizens` -- their names depend on which nationalities exist, so they have to
# be discovered from the table rather than listed.
_CITY_CITIZEN_FIELDS = ["nspe0", "nspe1", "nspe2", "style",
                        "city_radius_sq", "cma_enabled"]

_SCOREBOARD_FIELDS = [
    "happy", "content", "unhappy", "angry",
    "specialists0", "specialists1", "specialists2",
    "pollution", "literacy", "landarea", "settledarea", "spaceship",
    "units_built", "units_killed", "units_lost", "units", "wonders",
]


def _row_subset(row: dict[str, Any], fields: list[str]) -> dict[str, Any]:
    return {f: row[f] for f in fields if f in row}


def extract_probes(sg: Savegame, mapinfo=None) -> dict[str, Any]:
    """All nine groups.  `mapinfo` is optional and only feeds the exploration
    fraction; the fog-of-war mask itself already travels in the `geo` field."""
    names = [str(sg.get(s, "name", s)) for s in sg.players()]
    trait_vector = [str(x) for x in sg.get("savefile", "trait_vector", [])]
    tech_vector = [str(x) for x in sg.get("savefile", "technology_vector", [])]

    out: dict[str, Any] = {
        "A_static": {}, "B_attitude": {}, "C_belief": {}, "D_intent": {},
        "E_clock": {}, "F_global": _global_probes(sg, tech_vector),
        "G_units": {}, "H_citizens": {}, "I_scoreboard": {},
    }

    # The true city roster, used to score how stale each AI's memory is.
    real_cities = _real_cities(sg)

    research_rows = []
    r_tbl = sg.table("research", "r")
    if r_tbl is not None:
        research_rows = r_tbl.as_dicts()

    for sec in sg.players():
        pid = int(sec[6:])
        name = names[pid]

        out["A_static"][name] = _static(sg, sec, trait_vector)
        out["B_attitude"][name] = _attitude(sg, sec, names)
        out["C_belief"][name] = _belief(sg, sec, names, real_cities, mapinfo, name)
        out["D_intent"][name] = _intent(sg, sec, research_rows, pid)
        out["E_clock"][name] = _clock(sg, sec)
        out["G_units"][name] = _units(sg, sec)
        out["H_citizens"][name] = _citizens(sg, sec)
        out["I_scoreboard"][name] = _scoreboard(sg, pid)

    return out


# --------------------------------------------------------------------------
# A -- episode-static
# --------------------------------------------------------------------------

def _static(sg: Savegame, sec: str, trait_vector: list[str]) -> dict[str, Any]:
    """★ `trait` is the strongest probe target in the whole savegame.

    Two columns, and §2.5.6 established what they mean: `val` is the base value
    the server protects, `mod` is a modifier, and the **effective value is
    val + mod**.  `edit.trait_mod` writes `mod`, which is why it persists while
    writing `val` directly does not go through the engine.
    """
    t = sg.table(sec, "trait")
    traits: dict[str, Any] = {}
    if t is not None:
        for i, tname in enumerate(trait_vector):
            if i >= len(t.rows):
                break
            row = dict(zip(t.columns, t.rows[i]))
            val, mod = int(row.get("val", 0)), int(row.get("mod", 0))
            traits[tname] = {"val": val, "mod": mod, "effective": val + mod}
    return {
        "traits": traits,
        "ai_level": sg.get(sec, "ai.level"),
        "ai_type": sg.get(sec, "ai_type"),
        "ai_barb_type": sg.get(sec, "ai.barb_type"),
        "style": sg.get(sec, "style_by_name"),
        "nation": sg.get(sec, "nation"),
        "has_texai_mirror": sg.table(sec, "texai") is not None,
    }


# --------------------------------------------------------------------------
# B -- attitudes
# --------------------------------------------------------------------------

def _attitude(sg: Savegame, sec: str, names: list[str]) -> dict[str, Any]:
    """Row i of the `ai` table is this civ's stance toward player i."""
    ai = sg.table(sec, "ai")
    if ai is None:
        return {}
    out = {}
    for i, row in enumerate(ai.rows):
        if i >= len(names):
            break
        out[names[i]] = dict(zip(ai.columns, row))
    return out


# --------------------------------------------------------------------------
# C -- beliefs and memory
# --------------------------------------------------------------------------

def _real_cities(sg: Savegame) -> dict[int, dict[str, Any]]:
    """Ground truth for every city in the world, keyed by city id."""
    truth = {}
    for sec in sg.players():
        owner = str(sg.get(sec, "name", sec))
        c = sg.table(sec, "c")
        if c is None:
            continue
        for row in c.as_dicts():
            truth[int(row["id"])] = {
                "owner": owner, "name": str(row.get("name", "")),
                "size": int(row.get("size", 0)),
                "x": int(row.get("x", 0)), "y": int(row.get("y", 0)),
            }
    return truth


def _belief(sg: Savegame, sec: str, names: list[str],
            real: dict[int, dict[str, Any]], mapinfo, myname: str) -> dict[str, Any]:
    dc = sg.table(sec, "dc")
    rows = dc.as_dicts() if dc is not None else []

    # ★ The derived reading that makes `dc` interesting: how wrong is the
    # memory, measured against the truth in the same file.  Tracking the world
    # and tracking what someone *remembers* of the world are different tasks,
    # and this number is the size of the gap.
    stale = fresh = missing = 0
    examples = []
    for r in rows:
        cid = int(r.get("id", -1))
        t = real.get(cid)
        if t is None:
            missing += 1
            continue
        if int(r.get("size", -1)) != t["size"]:
            stale += 1
            if len(examples) < 8:
                examples.append({"city": t["name"], "believed": int(r.get("size", -1)),
                                 "actual": t["size"]})
        else:
            fresh += 1
    n_cmp = stale + fresh

    dip = sg.table(sec, "diplstate")
    diplo = {}
    if dip is not None:
        for i, row in enumerate(dip.rows):
            if i >= len(names):
                break
            d = dict(zip(dip.columns, row))
            # `current` is the relation the report already prints; the rest of
            # this row is not in any report.
            d.pop("current", None)
            diplo[names[i]] = d

    explored = None
    if mapinfo is not None:
        from .geo import explored_fraction
        explored = round(explored_fraction(mapinfo, myname), 4)

    return {
        "dc": rows,
        "dc_total": sg.get(sec, "dc_total"),
        "dc_staleness": {
            "remembered": len(rows), "compared": n_cmp,
            "size_stale": stale, "size_fresh": fresh, "no_longer_exists": missing,
            "stale_fraction": round(stale / n_cmp, 4) if n_cmp else None,
            "examples": examples,
        },
        "diplstate_hidden": diplo,
        "explored_fraction": explored,
    }


# --------------------------------------------------------------------------
# D -- development intent
# --------------------------------------------------------------------------

def _intent(sg: Savegame, sec: str, research_rows: list[dict], pid: int) -> dict[str, Any]:
    """`goal_name` vs `now_name` is the interesting pair here.

    The report shows the long-term goal; `now_name` -- the intermediate tech
    actually being researched toward it -- is not shown.  Recovering `now` from
    `goal` requires knowing the tech tree, so it is a probe with an exact answer
    that needs real domain reasoning rather than pattern matching.

    ⚠️ `adv.wonder_city` is a valid probe target but a dead intervention handle:
    §2.5.1 measured it being recomputed by the AI every turn (129 -> 129).
    """
    # research rows are per TEAM (`number` = team_no), not per player -- see state.extract (2026-09-15)
    team_no = int(sg.get(sec, "team_no", pid) or 0)
    res = next((rr for rr in research_rows if int(rr.get("number", -1)) == team_no), None)
    if res is None:
        res = research_rows[pid] if pid < len(research_rows) else {}
    cities = {}
    c = sg.table(sec, "c")
    if c is not None:
        for row in c.as_dicts():
            cities[str(row.get("name", "?"))] = _row_subset(row, _CITY_INTENT_FIELDS)
    return {
        "wonder_city": sg.get(sec, "adv.wonder_city"),
        "research": {
            "goal_name": res.get("goal_name"),
            "now_name": res.get("now_name"),
            "bulbs": res.get("bulbs"),
            "bulbs_before": res.get("bulbs_before"),
            "futuretech": res.get("futuretech"),
            "got_tech": res.get("got_tech"),
        },
        "cities": cities,
    }


# --------------------------------------------------------------------------
# E -- clocks
# --------------------------------------------------------------------------

def _clock(sg: Savegame, sec: str) -> dict[str, Any]:
    cities = {}
    c = sg.table(sec, "c")
    if c is not None:
        for row in c.as_dicts():
            cities[str(row.get("name", "?"))] = _row_subset(row, _CITY_CLOCK_FIELDS)
    return {
        "last_war": sg.get(sec, "last_war"),
        "revolution_finishes": sg.get(sec, "revolution_finishes"),
        "turns_alive": sg.get(sec, "turns_alive"),
        "idle_turns": sg.get(sec, "idle_turns"),
        "hut_count": sg.get(sec, "hut_count"),
        "cities": cities,
    }


# --------------------------------------------------------------------------
# F -- global accumulators
# --------------------------------------------------------------------------

def _global_probes(sg: Savegame, tech_vector: list[str]) -> dict[str, Any]:
    """⚠️ The RNG state is stored as a digest only.

    `[random]` holds `index_J/K/X` plus `table0..N`, and it is the reason
    save/reload/continue is bit-identical.  Recording it in full would add a
    kilobyte of incompressible hex per checkpoint for no probe value -- nothing
    can be inferred about a Mersenne-style table from behaviour.  The digest is
    enough to tell two branches apart.
    """
    adv = str(sg.get("game", "global_advances", ""))
    world_techs = [tech_vector[i] for i, ch in enumerate(adv)
                   if ch == "1" and i < len(tech_vector)]
    rnd = sg.sections.get("random", {})
    tables = "".join(str(v) for k, v in sorted(rnd.items()) if k.startswith("table"))
    return {
        "globalwarming": sg.get("game", "globalwarming"),
        "heating": sg.get("game", "heating"),
        "warminglevel": sg.get("game", "warminglevel"),
        "nuclearwinter": sg.get("game", "nuclearwinter"),
        "cooling": sg.get("game", "cooling"),
        "coolinglevel": sg.get("game", "coolinglevel"),
        "world_techs": world_techs,
        "world_tech_count": len(world_techs),
        "rng": {
            "index_J": rnd.get("index_J"), "index_K": rnd.get("index_K"),
            "index_X": rnd.get("index_X"),
            "table_digest": hashlib.md5(tables.encode()).hexdigest() if tables else None,
        },
    }


# --------------------------------------------------------------------------
# G / H / I
# --------------------------------------------------------------------------

def _units(sg: Savegame, sec: str) -> list[dict[str, Any]]:
    u = sg.table(sec, "u")
    if u is None:
        return []
    out = []
    for row in u.as_dicts():
        d = {"id": row.get("id"), "type": row.get("type_by_name")}
        d.update(_row_subset(row, _UNIT_PROBE_FIELDS))
        out.append(d)
    return out


def _citizens(sg: Savegame, sec: str) -> dict[str, Any]:
    """Per-city population composition.

    Decoded (verified on 35/35 cities, `sum(citizenN) == size`):

        citizenN        citizens whose **nationality** is player slot N
        nspe0/1/2       specialists of each kind; specialists are part of `size`,
                        not additional to it
        city_radius_sq  squared work radius
        style           city graphics style, episode-static
        cma_enabled     0 for every AI city observed, so the 14 `cma_*` columns
                        are recorded as this one flag rather than in full

    The nationality split is the point of this group: a captured city keeps the
    losing civ's citizens, so `size - citizen<own>` is the foreign population --
    the direct antecedent of unrest and revolt.

    ⚠️ **The column names are not fixed.** freeciv writes a `citizenN` column only
    for nationalities that actually occur in that player's cities, so the set
    differs per player: in one measured savegame player0 had `citizen0,citizen1`,
    player1 had `citizen0,citizen1,citizen2`, and player2 had only `citizen2`.
    An earlier version listed `citizen0` as a fixed field, which meant the value
    was captured only when nationality-0 citizens happened to be present -- 23%
    of city-turns in the v1 corpus -- and for most owners it was a *foreign*
    count rather than their own.  The nationality signal was therefore unusable.
    Discover the columns instead of naming them, and record `own_slot` so "own
    vs foreign" needs no outside knowledge.
    """
    c = sg.table(sec, "c")
    if c is None:
        return {}
    nat_cols = [x for x in c.columns if str(x).startswith("citizen")
                and str(x)[7:].isdigit()]
    own = sec[6:] if sec.startswith("player") else ""
    out = {}
    for row in c.as_dicts():
        d = _row_subset(row, _CITY_CITIZEN_FIELDS)
        d["citizens"] = {k[7:]: row[k] for k in nat_cols if k in row}
        d["own_slot"] = own
        out[str(row.get("name", "?"))] = d
    return out


def _scoreboard(sg: Savegame, pid: int) -> dict[str, Any]:
    """`[scoreN]` fields the report does not render.

    ★ `units_built` / `units_killed` / `units_lost` are cumulative counters --
    they encode the civ's entire military history and cannot be reconstructed
    from any single frame, unlike everything else here.
    """
    sec = f"score{pid}"
    return {f: sg.get(sec, f) for f in _SCOREBOARD_FIELDS if sg.get(sec, f) is not None}


def main():
    import argparse
    import json

    ap = argparse.ArgumentParser(description="dump probe state for one savegame")
    ap.add_argument("savegame")
    ap.add_argument("--group", default=None, help="only this group, e.g. A_static")
    ap.add_argument("--player", default=None)
    args = ap.parse_args()

    from .geo import extract_map
    from .state import extract

    sg = Savegame.load(args.savegame)
    ws = extract(sg, "probe")
    mi = extract_map(sg, [p.name for p in ws.players])
    pr = extract_probes(sg, mi)

    if args.group:
        pr = {args.group: pr[args.group]}
    if args.player:
        pr = {g: (v.get(args.player) if isinstance(v, dict) and args.player in v else v)
              for g, v in pr.items()}
    print(json.dumps(pr, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
