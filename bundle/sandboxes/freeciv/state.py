"""Frozen WorldState schema for the freeciv sandbox.

This is the substrate-agnostic state contract.  Field names are deliberately
not freeciv-specific (no `player0`, no `government_name`) so that a Grid2Op or
Covasim dataset can land in the same format later and be jointly trained on.
`extract()` maps a savegame into this shape; `delta()` computes the change
between two adjacent world steps, which is one of the training targets.

Timing note:  the values below are the *observable* state the world model is
asked to track -- the things that change turn to turn.  The freeciv savegame
keeps far more (unit orders, AI cma buffers, rally points, ...) but that is
the model's *private* state; the forecast task is about this projection.
"""

from __future__ import annotations

import re

import json
from dataclasses import dataclass, asdict
from typing import Any

from .savegame import Savegame

# Freeciv gives population in thousands; scorelog reports the citizen sum.
# We keep thousands in the typed state (monotonic, comparable across worlds).
_POP_UNIT = 1000

# Great wonders, as opposed to ordinary city improvements.  Both live in the
# same `improvement_vector` bitset, so the only way to tell them apart is by
# name.  A great wonder is world-unique, which is why it belongs on the player
# rather than being buried in a per-city improvements dict.
GREAT_WONDERS = {
    "Apollo Program", "A.Smith's Trading Co.", "Colossus",
    "Copernicus' Observatory", "Cure For Cancer", "Darwin's Voyage",
    "Eiffel Tower", "Great Library", "Great Wall", "Hanging Gardens",
    "Hoover Dam", "Isaac Newton's College", "J.S. Bach's Cathedral",
    "King Richard's Crusade", "Leonardo's Workshop", "Lighthouse",
    "Magellan's Expedition", "Manhattan Project", "Marco Polo's Embassy",
    "Michelangelo's Chapel", "Mausoleum of Mausolos", "Statue of Zeus",
    "Temple of Artemis", "Pyramids", "Internet", "Shakespeare's Theatre",
    "Statue of Liberty", "Sun Tzu's War Academy", "United Nations",
    "Women's Suffrage",
}


@dataclass
class City:
    id: int
    name: str
    x: int
    y: int
    size: int
    owner: str
    production: str          # currently_building_name
    food_stock: int
    shield_stock: int        # progress toward the current build
    founded: int             # turn founded
    improvements: dict[str, bool]   # city_radius improvements mask decoded via ruleset


@dataclass
class Player:
    name: str
    nation: str
    is_alive: bool
    government: str
    gold: int
    income: dict[str, int]            # trade/production/tax split, see extract()
    cities: list[City]
    techs: list[str]
    research: str                     # current research goal
    bulbs_last: int                   # ⚠️ misnamed: research.bulbs = ACCUMULATED bulbs toward the current tech
    population: int                   # thousands
    score: int
    culture: int
    military_units: int               # ⚠️ misnamed: savegame `nunits` = ALL units (verified 2026-09-10); count military by type
    units: list[dict[str, Any]]
    diplomacy: dict[str, str]         # other-player-name -> state (War/Peace/...)
    wonders: list[str]                # great wonders this civ has completed
    wonders_lost: list[str]           # great wonders destroyed while theirs
    small_wonders: list[str]          # e.g. Palace -- world-unique per player
    # v2: the tech being researched right now (`research` above is the long-term
    # goal).  Defaulted so v1 records without the field still rehydrate.
    research_now: str = ""
    # v2: team slot (`team_no` in the save; every player is on a team, possibly
    # alone).  Team worlds put several players on one slot.
    team_no: int = 0


@dataclass
class WorldState:
    sim: str
    ruleset: str
    world_id: str
    turn: int
    year: int
    players: list[Player]
    events: list[dict[str, Any]]      # this step's event records

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


def _decode_improvements(mask: str, names: list[str]) -> dict[str, bool]:
    """City improvements are a bitset in the savegame; name is given per ruleset."""
    return {names[i]: bool(int(c)) for i, c in enumerate(mask) if i < len(names)}


def extract(sg: Savegame, world_id: str) -> WorldState:
    """Map a savegame into the frozen WorldState shape."""
    # The research table is one row per player and is read once, outside the
    # player loop.  It used to be read inside and probed with `"entries" in
    # dir()`, which relied on the local scope leaking the loop variable and
    # raised IndexError whenever a player index ran past the table -- a latent
    # parse failure that would have poisoned a whole batch rather than erroring
    # on one world.
    r_tbl = sg.table("research", "r")
    research_rows = r_tbl.as_dicts() if r_tbl is not None else []
    tech_vector = [str(x) for x in sg.get("savefile", "technology_vector", [])]

    players = []
    for sec in sg.players():
        pid = int(sec[6:])
        name = str(sg.get(sec, "name", sec))
        cities = []
        c_tbl = sg.table(sec, "c")
        if c_tbl is not None:
            names = [str(x) for x in sg.get("savefile", "improvement_vector", [])]
            for row in c_tbl.as_dicts():
                cities.append(City(
                    id=int(row.get("id", -1)),
                    name=str(row.get("name", "?")),
                    x=int(row.get("x", 0)), y=int(row.get("y", 0)),
                    size=int(row.get("size", 0)),
                    owner=name,
                    production=str(row.get("currently_building_name", "")),
                    food_stock=int(row.get("food_stock", 0)),
                    shield_stock=int(row.get("shield_stock", 0)),
                    founded=int(row.get("turn_founded", 0)),
                    improvements=_decode_improvements(
                        str(row.get("improvements", "")), names[: len(str(row.get("improvements", "")))]),
                ))

        techs = []
        research_row: dict[str, Any] = {}
        # ★ 2026-09-15: the [research] table has ONE ROW PER RESEARCH TEAM (`number` = team id;
        # freeciv pools research inside a team), not one per player.  Indexing it by player slot was
        # right only while every player sat in its own team; in the 60 /team worlds of the v2 bench
        # corpus it handed civs another team's tech list ("technologies (0)" for a civ fielding AEGIS
        # cruisers).  Found by the 2026-09-15 review run.  Map by the player's team_no.
        team_no = int(sg.get(sec, "team_no", pid) or 0)
        ridx = next((i for i, rr in enumerate(research_rows) if int(rr.get("number", -1)) == team_no), None)
        if ridx is None and pid < len(research_rows):
            ridx = pid                                      # pre-team saves without `number`
        if ridx is not None:
            # The tech *bitset* is the final column of each row: the header
            # declares `done` but the real mask is 88 chars and sits past the
            # declared columns.  Decoded against `technology_vector`.
            research_row = research_rows[ridx]
            mask = str(r_tbl.rows[ridx][-1])
            techs = [tech_vector[i] for i, c in enumerate(mask)
                     if c == "1" and i < len(tech_vector)]
            techs = [t for t in techs if t != "A_NONE"]

        # units: the [playerN] `u` table carries position/type/hp per unit.
        units = []
        u_tbl = sg.table(sec, "u")
        if u_tbl is not None:
            for row in u_tbl.as_dicts():
                units.append({
                    "id": int(row.get("id", 0)),
                    "x": int(row.get("x", 0)),
                    "y": int(row.get("y", 0)),
                    "type": str(row.get("type_by_name", "")),
                    "hp": int(row.get("hp", 0)),
                    "moves": int(row.get("moves", 0)),
                    "activity": int(row.get("activity", 0)),
                })

        # diplomacy: row i of [playerN].diplstate is N's relation to player#i.
        diplomacy = {}
        d_tbl = sg.table(sec, "diplstate")
        if d_tbl is not None and len(d_tbl.rows) == len(sg.players()):
            for i, row in enumerate(d_tbl.rows):
                diplomacy.setdefault(f"player{i}", str(row[0]))

        # Wonders: great wonders are world-unique and belong on the player, not
        # buried in a per-city improvements dict.  `lost_wonders` is a parallel
        # bitset of wonders destroyed while this civ held them -- a real event
        # in the world's history that the per-city view cannot express.
        impr_names = [str(x) for x in sg.get("savefile", "improvement_vector", [])]
        wonders, small = [], []
        for c in cities:
            for k, v in c.improvements.items():
                if not v:
                    continue
                if k in GREAT_WONDERS:
                    wonders.append(k)
                elif k in ("Palace", "Ecclesiastical Palace"):
                    small.append(k)
        lost_mask = str(sg.get(sec, "lost_wonders", ""))
        wonders_lost = [impr_names[i] for i, ch in enumerate(lost_mask)
                        if ch == "1" and i < len(impr_names)]

        # Income: freeciv does not store a single "income" scalar, so it is
        # reported as the components the game actually computes.  `bnp` is the
        # civ's total trade and `mfg` its total production (both from [scoreN],
        # cross-checked against the scorelog); the tax split says how trade is
        # divided between treasury, research and luxury.  Reporting the split
        # rather than a derived net figure avoids inventing a number the
        # simulator never computed -- the maintenance side is not in the save.
        income = {
            "trade": int(sg.get(f"score{pid}", "bnp", 0)),
            "production": int(sg.get(f"score{pid}", "mfg", 0)),
            "science_output": int(sg.get(f"score{pid}", "techout", 0)),
            "tax_pct": int(sg.get(sec, "rates.tax", 0)),
            "science_pct": int(sg.get(sec, "rates.science", 0)),
            "luxury_pct": int(sg.get(sec, "rates.luxury", 0)),
        }

        players.append(Player(
            name=name,
            nation=str(sg.get(sec, "nation", "")),
            is_alive=bool(sg.get(sec, "is_alive", True)),
            government=str(sg.get(sec, "government_name", "")),
            gold=int(sg.get(sec, "gold", 0)),
            income=income,
            cities=cities,
            techs=techs,
            research=str(research_row.get("goal_name", "") or ""),
            bulbs_last=int(research_row.get("bulbs", 0) or 0),
            research_now=str(research_row.get("now_name", "") or ""),
            team_no=int(sg.get(sec, "team_no", 0) or 0),
            population=int(sg.get(f"score{pid}", "population", 0)),
            score=int(sg.get(f"score{pid}", "total", 0)),
            culture=int(sg.get(f"score{pid}", "culture", 0)),
            military_units=int(sg.get(sec, "nunits", 0)),
            units=units,
            diplomacy=diplomacy,
            wonders=sorted(wonders),
            wonders_lost=sorted(wonders_lost),
            small_wonders=sorted(set(small)),
        ))

    return WorldState(
        sim="freeciv",
        ruleset=sg.ruleset,
        world_id=world_id,
        turn=sg.turn,
        year=sg.year,
        players=players,
        events=[],
    )


def delta(prev: WorldState, cur: WorldState) -> dict[str, Any]:
    """Change between two adjacent steps, for the structured-transition target.

    Keyed by player *name* rather than index, because the player roster is not
    fixed: a civ can be eliminated mid-game, and freeciv also *adds* players --
    barbarian leaders spawn, and a civil war splits an existing civ into two.
    Both directions are recorded explicitly (`eliminated` / `appeared`) so a
    roster change reads as an event rather than as silently missing data.
    """
    d = {"turn": {"from": prev.turn, "to": cur.turn},
         "players": {}, "roster": {"eliminated": [], "appeared": []}}
    prev_by = {p.name: p for p in prev.players}
    cur_by = {p.name: p for p in cur.players}

    for name, p in prev_by.items():
        q = cur_by.get(name)
        if q is None:
            d["players"][name] = {"eliminated": True}
            d["roster"]["eliminated"].append(name)
            continue
        d["players"][name] = {
            "gold": q.gold - p.gold,
            "population": q.population - p.population,
            "score": q.score - p.score,
            "culture": q.culture - p.culture,
            "cities": len(q.cities) - len(p.cities),
            "techs": len(q.techs) - len(p.techs),
            "military_units": q.military_units - p.military_units,
            "trade": q.income.get("trade", 0) - p.income.get("trade", 0),
            "production": q.income.get("production", 0) - p.income.get("production", 0),
            "government": {"from": p.government, "to": q.government}
                          if p.government != q.government else None,
            "new_cities": sorted({c.name for c in q.cities} - {c.name for c in p.cities}),
            "lost_cities": sorted({c.name for c in p.cities} - {c.name for c in q.cities}),
            "new_techs": sorted(set(q.techs) - set(p.techs)),
            "new_wonders": sorted(set(q.wonders) - set(p.wonders)),
            "city_sizes": {c.name: c.size for c in q.cities},
        }

    # Players present now but not before: barbarians spawning, or a civil war
    # splitting off a new civ.  Their whole state is the delta.
    for name, q in cur_by.items():
        if name in prev_by:
            continue
        d["roster"]["appeared"].append(name)
        d["players"][name] = {
            "appeared": True,
            "gold": q.gold, "population": q.population, "score": q.score,
            "cities": len(q.cities), "techs": len(q.techs),
            "government": {"from": None, "to": q.government},
            "new_cities": sorted(c.name for c in q.cities),
        }
    return d


def base_ruleset(name: str) -> str:
    """`civ2civ3_ill30` -> `civ2civ3`.  v2 records carry the *ruleset directory*
    in `ruleset` (the illness variants are separate directories; meta.json has
    `rulesetdir` too).  The rules in force, the primer to feed and the report
    header all want the base name -- printing the variant would leak the hidden
    illness parameter, which is a task-2 probe target."""
    return re.sub(r"_ill\d+$", "", name or "")
