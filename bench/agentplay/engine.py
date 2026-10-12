"""Engine step for agent-play: one turn of a human-controlled civilization.

The ego civ is a *human* player (no `ai` flag) in an otherwise unchanged AI world.  A turn is:

    savegame(T)  --edit-->  start.sav  --lua-->  freeciv-server runs turn T  -->  savegame(T+1)

  * civ-level knobs (tax rates, research goal, government change, city production, buying) are savegame
    edits -- for a human player nothing recomputes them, so unlike the AI-player "dead handles" they hold;
  * unit actions are `lua cmd` lines executed before `start`: `edit.unit_move` (one adjacent step at a time,
    charging the terrain cost, the same rule the engine applies), `edit.perform_action` (Found City, Fortify,
    Build Road/Irrigation/Mine, Attack, Help Wonder, ...), `edit.unit_kill` (disband);
  * server-side automation is available exactly as to a human at a client: `server_side_agent` 1 =
    auto-worker, activity 11 = auto-explore, activity 7 = sentry.

Every Lua action logs `AGENT|<kind>|<unit>|<detail>|<true/false>` on the server console (stdout; **not** the
-l log file), and Lua runtime errors appear there as `lua error:` blocks -- the driver turns both into the
per-action feedback the model sees next turn.  Measured cost of one step on an S/M map: 0.7-1.1 s.

⚠️ A savegame edit the loader rejects makes the server start a *new random game* with the same settings and
no error in the log file.  `step()` therefore checks that the player roster after the run matches the
roster before it, and raises otherwise.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "freeciv-sandbox"))
sys.path.insert(0, str(ROOT))
from sandboxes.freeciv.engine_events import hook_lua, parse_log     # noqa: E402
from sandboxes.freeciv.runner import BASE_SETTINGS, serv_script            # noqa: E402

# ★ agent-play runs on the v3 image (v2 + patches/v3-restart-phase.patch): without it a phase resumed from a
# savegame skips the AI's first activities, the AI's settlers never get a build-city task, and the opponents
# stop expanding (doc 17 §6.7.3).  The corpus collection still uses the v2 image, hence the separate variable.
DOCKER_IMG = os.environ.get("WM_AGENTPLAY_IMG", "wm/freeciv-server:3.1.6-v3")
from sandboxes.freeciv.savegame import Savegame                     # noqa: E402

from bench.agentplay.hexmap import Board                            # noqa: E402
from bench.agentplay.rules import Rules                             # noqa: E402
from bench.agentplay.sites import is_legal_site, land_sites_from, sea_sites_from  # noqa: E402

ACTIVITY = {"idle": 0, "mine": 3, "irrigate": 4, "fortified": 5, "sentry": 7, "pillage": 9, "explore": 11,
            "transform": 12, "fortifying": 15, "cultivate": 21, "plant": 22}
SSA_NONE, SSA_AUTOWORKER, SSA_AUTOEXPLORE = 0, 1, 2
MAX_RATE = {"Anarchy": 100, "Tribal": 60, "Despotism": 60, "Monarchy": 70, "Communism": 80, "Fundamentalism": 70,
            "Federation": 90, "Republic": 80, "Democracy": 90}
# unit action -> (Lua action rule name, target kind)
TILE_ACTIONS = {"found_city": "Found City", "irrigate": "Build Irrigation", "mine": "Build Mine", "road": "Build Road",
                "cultivate": "Cultivate", "plant": "Plant", "transform": "Transform Terrain", "pillage": "Pillage"}
SELF_ACTIONS = {"fortify": "Fortify"}
CITY_ACTIONS = {"join_city": "Join City", "help_wonder": "Help Wonder", "trade_route": "Establish Trade Route",
                "home_city": "Home City"}
REVOLUTION_TURNS = 2       # revolentype RANDQUICK draws 1-5; a fixed 2 keeps arms comparable


@dataclass
class UnitCmd:
    uid: int
    kind: str                                   # move|found_city|join_city|fortify|sentry|explore|auto_work|irrigate|mine|road|
                                                # cultivate|plant|transform|pillage|attack|conquer|help_wonder|trade_route|home_city|disband|idle
    target: tuple[int, int] | None = None       # tile for move/attack/conquer/found_city-at/help_wonder(city tile)


@dataclass
class CityCmd:
    name: str
    build: str | None = None
    buy: bool = False


@dataclass
class TurnPlan:
    research_goal: str | None = None
    research_now: str | None = None
    rates: tuple[int, int, int] | None = None   # tax, science, luxury
    government: str | None = None
    cities: list[CityCmd] = field(default_factory=list)
    units: list[UnitCmd] = field(default_factory=list)
    treaties: list[dict] = field(default_factory=list)      # [{"with": civ name, "type": "ceasefire"|"peace"|"war"}]


@dataclass
class StepResult:
    ok: bool
    save: Path | None
    turn: int
    feedback: list[str]                         # human-readable, one line per accepted/rejected action
    lua_results: list[dict[str, Any]]           # parsed AGENT| lines
    lua_errors: list[str]
    events: list[dict[str, Any]]                # engine events during the turn (all civs; filter in observe)
    seconds: float
    stdout: str = ""


class Ego:
    """The ego player's view of the savegame tables we edit."""

    def __init__(self, sg: Savegame, idx: int):
        self.sg, self.idx, self.sec = sg, idx, f"player{idx}"
        self.name = str(sg.get(self.sec, "name"))
        self.team = int(sg.get(self.sec, "team_no", idx))
        self.gov = str(sg.get(self.sec, "government_name"))
        self.gold = int(sg.get(self.sec, "gold", 0))
        u = sg.table(self.sec, "u"); c = sg.table(self.sec, "c")
        self.units = {int(r["id"]): r for r in (u.as_dicts() if u else [])}
        self.unit_row = {int(r["id"]): i for i, r in enumerate(u.as_dicts() if u else [])}
        self.cities = {str(r["name"]): r for r in (c.as_dicts() if c else [])}
        self.city_row = {str(r["name"]): i for i, r in enumerate(c.as_dicts() if c else [])}
        rt = sg.table("research", "r")
        self.research_row = None
        for i, r in enumerate(rt.as_dicts() if rt else []):
            if int(r["number"]) == self.team:
                self.research_row = i; self.research = r
        tv = list(sg.get("savefile", "technology_vector") or [])
        done = str(self.research.get("done", "")) if self.research_row is not None else ""
        self.known_techs = {t for t, b in zip(tv, done) if b == "1" and t != "A_NONE"}
        self.is_human = "ai" not in [str(x) for x in (sg.get(self.sec, "flags") or [])] if sg.get(self.sec, "flags") is not None else True

    def all_units_by_tile(self) -> dict[tuple[int, int], list[tuple[str, dict]]]:
        out: dict[tuple[int, int], list] = {}
        for sec in self.sg.players():
            t = self.sg.table(sec, "u")
            if not t:
                continue
            owner = str(self.sg.get(sec, "name"))
            for r in t.as_dicts():
                out.setdefault((int(r["x"]), int(r["y"])), []).append((owner, r))
        return out

    def all_cities_by_tile(self) -> dict[tuple[int, int], tuple[str, dict]]:
        out = {}
        for sec in self.sg.players():
            t = self.sg.table(sec, "c")
            if not t:
                continue
            owner = str(self.sg.get(sec, "name"))
            for r in t.as_dicts():
                out[(int(r["x"]), int(r["y"]))] = (owner, r)
        return out


def make_human(sg: Savegame, idx: int) -> bool:
    """Drop the `ai` flag line of player idx (an empty flags vector is *no line*).  Returns True if changed."""
    sec = f"player{idx}"
    key = (sec, "flags")
    if key not in sg._scalar_line:
        return False
    li = sg._scalar_line[key]
    sg.lines[li] = "; agentplay: flags removed -> human player"
    del sg._scalar_line[key]; sg.sections[sec].pop("flags", None)
    return True


def _insert_after(sg: Savegame, sec: str, key_after: str, line: str) -> None:
    """Insert a raw line right after scalar `sec.key_after`.  Do this LAST: it shifts line indices."""
    li = sg._scalar_line[(sec, key_after)]
    sg.lines.insert(li + 1, line)


def buy_price(remaining: int, stock: int, great_wonder: bool) -> int:
    cost = 2 * remaining + (remaining * remaining) // 20
    if stock == 0:
        cost *= 2
    if great_wonder:
        cost *= 2
    return cost


def _lua(kind: str, uid: int, detail: str, expr: str) -> str:
    """One single-line Lua statement: find the unit, run expr, log the result under the AGENT tag."""
    return (f"do local p=find.player({{PID}}) local u=find.unit(p,{uid}) if u==nil then log.normal('AGENT|{kind}|{uid}|{detail}|nounit') "
            f"else local ok={expr} log.normal('AGENT|{kind}|{uid}|{detail}|%s', tostring(ok)) end end")


def plan_to_edits(sg: Savegame, ego: Ego, rules: Rules, board: Board, plan: TurnPlan, turn: int,
                  standing: dict[int, dict[str, Any]] | None = None) -> tuple[list[str], list[str], list[str]]:
    """Apply the civ-level edits to `sg` in place and build the Lua lines for unit actions.
    Returns (lua_lines, feedback, warnings).  `standing` holds multi-turn orders (goto and what to do on arrival);
    it is updated in place: entries are added for new gotos and removed when finished or impossible."""
    fb: list[str] = []; lua: list[str] = []; warn: list[str] = []
    standing = standing if standing is not None else {}
    sec = ego.sec
    pending_gov_line = None

    # ---- rates ----
    if plan.rates is not None:
        tax, sci, lux = plan.rates
        mx = MAX_RATE.get(ego.gov, 100)
        if tax + sci + lux != 100 or any(v % 10 or v < 0 for v in (tax, sci, lux)):
            fb.append(f"REJECTED rates {tax}/{sci}/{lux}: must be multiples of 10 summing to 100")
        elif max(tax, sci, lux) > mx:
            fb.append(f"REJECTED rates {tax}/{sci}/{lux}: {ego.gov} caps every rate at {mx}%")
        else:
            sg.set_scalar(sec, "rates.tax", tax); sg.set_scalar(sec, "rates.science", sci); sg.set_scalar(sec, "rates.luxury", lux)
            fb.append(f"OK rates tax {tax}% science {sci}% luxury {lux}%")

    # ---- research ----
    if ego.research_row is not None:
        if plan.research_goal is not None:
            g = plan.research_goal
            if g not in rules.techs:
                fb.append(f"REJECTED research goal '{g}': unknown technology")
            elif g in ego.known_techs:
                fb.append(f"REJECTED research goal '{g}': already known")
            else:
                sg.set_cell("research", "r", ego.research_row, "goal_name", g)
                fb.append(f"OK research goal {g} (the server researches the prerequisites in order)")
        if plan.research_now is not None:
            n = plan.research_now
            if n not in rules.researchable(ego.known_techs):
                fb.append(f"REJECTED research now '{n}': not researchable yet (prerequisites missing or already known)")
            else:
                sg.set_cell("research", "r", ego.research_row, "now_name", n)
                fb.append(f"OK now researching {n}")

    # ---- government ----
    if plan.government is not None:
        g = plan.government
        reqs = rules.governments.get(g)
        if reqs is None:
            fb.append(f"REJECTED government '{g}': no such government")
        elif g == ego.gov:
            fb.append(f"REJECTED government '{g}': already in effect")
        elif ego.gov == "Anarchy":
            fb.append("REJECTED government change: a revolution is already under way")
        elif any(r["type"].lower() == "tech" and r["name"] not in ego.known_techs for r in reqs):
            need = ", ".join(r["name"] for r in reqs if r["type"].lower() == "tech")
            now = str(ego.research.get("now_name", "")) if ego.research_row is not None else ""
            fb.append(f"REJECTED government '{g}': requires the technology {need}, which you have NOT discovered yet"
                      + (f" (you are researching it now, {ego.research.get('bulbs')} bulbs stored)" if now in need else ""))
        else:
            sg.set_scalar(sec, "government_name", "Anarchy")
            sg.set_scalar(sec, "revolution_finishes", turn + REVOLUTION_TURNS)
            pending_gov_line = f'target_government_name="{g}"'
            fb.append(f"OK revolution started: Anarchy for {REVOLUTION_TURNS} turns, then {g}")

    # ---- cities ----
    great = {b.name for b in rules.buildings.values() if b.genus == "GreatWonder"}
    for cc in plan.cities:
        if cc.name not in ego.cities:
            fb.append(f"REJECTED city '{cc.name}': not one of your cities"); continue
        row = ego.city_row[cc.name]; c = ego.cities[cc.name]
        if cc.build is not None:
            item = cc.build
            kind = "UnitType" if item in rules.units else "Building" if item in rules.buildings else None
            if kind is None:
                fb.append(f"REJECTED {cc.name} build '{item}': unknown unit or building"); continue
            cur_kind, cur_name = str(c["currently_building_kind"]), str(c["currently_building_name"])
            if (cur_kind, cur_name) != (kind, item):
                stock = int(c["shield_stock"])
                cat_old = "W" if cur_name in great else cur_kind
                cat_new = "W" if item in great else kind
                if cat_old != cat_new and stock > 0 and not bool(c.get("did_buy")):
                    stock //= 2                                   # freeciv's 50% category-change penalty
                    sg.set_cell(sec, "c", row, "shield_stock", stock)
                sg.set_cell(sec, "c", row, "currently_building_kind", kind)
                sg.set_cell(sec, "c", row, "currently_building_name", item)
                c = dict(c); c["shield_stock"] = stock; c["currently_building_kind"] = kind; c["currently_building_name"] = item
                fb.append(f"OK {cc.name} now builds {item}" + (" (half the stored shields lost: category change)" if cat_old != cat_new and stock > 0 else ""))
        if cc.buy:
            item = str(c["currently_building_name"]); cost = rules.cost_of(item); stock = int(c["shield_stock"])
            if cost is None:
                fb.append(f"REJECTED buy in {cc.name}: cannot price '{item}'")
            elif bool(c.get("did_buy")):
                fb.append(f"REJECTED buy in {cc.name}: already bought something this turn")
            elif stock >= cost:
                fb.append(f"REJECTED buy in {cc.name}: {item} completes anyway")
            else:
                price = buy_price(cost - stock, stock, item in great)
                if price > ego.gold:
                    fb.append(f"REJECTED buy in {cc.name}: {item} costs {price} gold, treasury {ego.gold}")
                else:
                    ego.gold -= price
                    sg.set_scalar(sec, "gold", ego.gold)
                    sg.set_cell(sec, "c", row, "shield_stock", cost); sg.set_cell(sec, "c", row, "did_buy", True)
                    fb.append(f"OK bought {item} in {cc.name} for {price} gold (completes at turn change)")

    # ---- units ----
    units_by_tile = ego.all_units_by_tile(); cities_by_tile = ego.all_cities_by_tile()
    own_city_tiles = {(int(c["x"]), int(c["y"])) for c in ego.cities.values()}
    board.enterable_cities = set(own_city_tiles)
    board.blocked = {t for t, lst in units_by_tile.items() if any(o != ego.name for o, _ in lst)} | \
                    {t for t, (o, _) in cities_by_tile.items() if o != ego.name}

    def set_unit(uid: int, activity: int | None = None, ssa: int | None = None):
        row = ego.unit_row[uid]
        if activity is not None:
            sg.set_cell(sec, "u", row, "activity", activity); sg.set_cell(sec, "u", row, "activity_count", 0)
            sg.set_cell(sec, "u", row, "activity_tgt", -1)     # a stale target with a target-less activity trips an engine assertion
        if ssa is not None:
            sg.set_cell(sec, "u", row, "server_side_agent", ssa)

    # new explicit commands override standing orders
    for uc in plan.units:
        if uc.uid not in ego.units:
            fb.append(f"REJECTED unit {uc.uid}: not one of your units"); continue
        u = ego.units[uc.uid]; ut = rules.units.get(str(u["type_by_name"]))
        k = uc.kind
        if k == "idle":
            standing.pop(uc.uid, None); set_unit(uc.uid, activity=0, ssa=SSA_NONE)
            fb.append(f"OK unit {uc.uid} {u['type_by_name']}: orders cleared"); continue
        if k == "sentry":
            standing.pop(uc.uid, None); set_unit(uc.uid, activity=ACTIVITY["sentry"], ssa=SSA_NONE)
            fb.append(f"OK unit {uc.uid} {u['type_by_name']}: sentry"); continue
        if k == "explore":
            standing.pop(uc.uid, None); set_unit(uc.uid, activity=ACTIVITY["explore"], ssa=SSA_NONE)   # the server sets ssa=2 itself; setting it here leaves the unit idle
            pax = [str(o) for o, r in ego.units.items() if int(r.get("transported_by", -1)) == uc.uid]
            fb.append(f"OK unit {uc.uid} {u['type_by_name']}: auto-explore" + (f" (WARNING: it carries units {', '.join(pax)}, they travel with it)" if pax else "")); continue
        if k == "auto_work":
            if ut is None or not ({"Settlers", "Workers"} & set(ut.flags) or "Workers" in str(u["type_by_name"])):
                fb.append(f"REJECTED unit {uc.uid} {u['type_by_name']}: only Workers/Settlers/Engineers can auto-work"); continue
            standing.pop(uc.uid, None); set_unit(uc.uid, activity=0, ssa=SSA_AUTOWORKER)
            fb.append(f"OK unit {uc.uid} {u['type_by_name']}: auto-worker"); continue
        if k == "disband":
            standing.pop(uc.uid, None)
            lua.append(_lua("disband", uc.uid, "", "(function() edit.unit_kill(u, 'disbanded', nil) return true end)()"))
            continue
        if k == "fortify":
            standing.pop(uc.uid, None)
            if ut is None or not ut.military or "Cant_Fortify" in ut.flags:
                fb.append(f"REJECTED unit {uc.uid} {u['type_by_name']}: cannot fortify (not a military unit); use sentry"); continue
            if int(u["activity"]) in (ACTIVITY["fortified"], ACTIVITY["fortifying"]):
                set_unit(uc.uid, ssa=SSA_NONE); fb.append(f"OK unit {uc.uid} {u['type_by_name']}: already fortified"); continue
            set_unit(uc.uid, activity=ACTIVITY["fortifying"], ssa=SSA_NONE)     # becomes Fortified at the next turn change
            fb.append(f"OK unit {uc.uid} {u['type_by_name']}: fortifying"); continue
        if k == "found_city" and uc.target is None:
            if ut is None or "Cities" not in ut.flags:
                fb.append(f"REJECTED unit {uc.uid} {u['type_by_name']}: only Settlers can found cities"); continue
            here = (int(u["x"]), int(u["y"])); aboard = int(u.get("transported_by", -1)) != -1
            if not aboard and is_legal_site(board, board.geo, here, cities_by_tile, ego.name, rules.citymindist):
                standing.pop(uc.uid, None); set_unit(uc.uid, activity=0, ssa=SSA_NONE)
                lua.append(_lua(k, uc.uid, "here", f"edit.perform_action(u, find.action('Found City'), u.tile)")); continue
            # not a legal spot here: turn it into "go to the best legal site and found" (resolved below, boat-aware)
            set_unit(uc.uid, activity=0, ssa=SSA_NONE)
            standing[uc.uid] = {"kind": "found_city", "target": None, "since": turn}
            fb.append(f"NOTE unit {uc.uid} Settlers: this tile is not a legal city site; heading for the best legal site instead")
            continue
        if k in TILE_ACTIONS and uc.target is None:
            if int(u.get("transported_by", -1)) != -1:
                fb.append(f"REJECTED unit {uc.uid} {u['type_by_name']} {k}: it is aboard boat {u['transported_by']}; give a land tile as 'to' (the boat must be next to it)"); continue
            standing.pop(uc.uid, None); set_unit(uc.uid, activity=0, ssa=SSA_NONE)
            lua.append(_lua(k, uc.uid, "here", f"edit.perform_action(u, find.action('{TILE_ACTIONS[k]}'), u.tile)")); continue
        if k in CITY_ACTIONS and uc.target is None:
            here = (int(u["x"]), int(u["y"]))
            if here not in cities_by_tile:
                fb.append(f"REJECTED unit {uc.uid} {k}: give the city's tile as target, or stand in the city"); continue
            cid = int(cities_by_tile[here][1]["id"]); owner_idx = [str(sg.get(s, "name")) for s in sg.players()].index(cities_by_tile[here][0])
            standing.pop(uc.uid, None); set_unit(uc.uid, ssa=SSA_NONE)
            lua.append(_lua(k, uc.uid, str(cid), f"edit.perform_action(u, find.action('{CITY_ACTIONS[k]}'), find.city(find.player({owner_idx}),{cid}))")); continue
        if k in ("move", "attack", "conquer", "found_city", "join_city", "help_wonder", "trade_route", "home_city",
                 "irrigate", "mine", "road", "cultivate", "plant", "transform", "pillage"):
            if uc.target is None:
                fb.append(f"REJECTED unit {uc.uid} {k}: needs a target tile"); continue
            tx, ty = uc.target
            if not (0 <= tx < board.geo.xsize and 0 <= ty < board.geo.ysize):
                fb.append(f"REJECTED unit {uc.uid} {k}: tile {tx},{ty} is off the map"); continue
            if not board.is_known(tx, ty):
                fb.append(f"REJECTED unit {uc.uid} {k}: tile {tx},{ty} is unexplored"); continue
            if k == "found_city":
                near = [(board.geo.distance((tx, ty), t), o, c["name"]) for t, (o, c) in cities_by_tile.items() if board.geo.distance((tx, ty), t) < rules.citymindist]
                if near:
                    d, o, n = min(near); fb.append(f"REJECTED unit {uc.uid} found_city -> {tx},{ty}: {n} ({o}) is only {d} tiles away (minimum {rules.citymindist})"); continue
                own = board.owner_at(tx, ty)
                if own not in (None, ego.name):
                    fb.append(f"REJECTED unit {uc.uid} found_city -> {tx},{ty}: the tile belongs to {own}; cities can only be founded on unclaimed or your own land"); continue
                if ut is None or "Cities" not in ut.flags:
                    fb.append(f"REJECTED unit {uc.uid} {u['type_by_name']}: only Settlers can found cities"); continue
            if k in ("attack", "conquer"):
                if ut is None or not ut.military:
                    fb.append(f"REJECTED unit {uc.uid} {u['type_by_name']}: not a military unit, it cannot attack"); continue
                foreign_here = [o for o, _ in units_by_tile.get((tx, ty), []) if o != ego.name] + ([cities_by_tile[(tx, ty)][0]] if (tx, ty) in cities_by_tile and cities_by_tile[(tx, ty)][0] != ego.name else [])
                if not foreign_here:
                    fb.append(f"REJECTED unit {uc.uid} {k} -> {tx},{ty}: no foreign unit or city on that tile"); continue
            if k == "join_city":
                if (tx, ty) not in cities_by_tile:
                    fb.append(f"REJECTED unit {uc.uid} join_city -> {tx},{ty}: no city there"); continue
                o, c = cities_by_tile[(tx, ty)]; pop = ut.pop_cost if ut else 0
                imprs = list(sg.get("savefile", "improvement_vector") or []); have = {n for n, b in zip(imprs, str(c["improvements"])) if b == "1"}
                cap = 16 if any(n.startswith("Aqueduct") for n in have) else 8
                if pop <= 0:
                    fb.append(f"REJECTED unit {uc.uid} {u['type_by_name']}: this unit cannot join a city"); continue
                if int(c["size"]) + pop > cap:
                    fb.append(f"REJECTED unit {uc.uid} join_city -> {c['name']}: size {c['size']} + {pop} exceeds the {cap} limit (needs an Aqueduct)"); continue
            set_unit(uc.uid, activity=0, ssa=SSA_NONE)
            standing[uc.uid] = {"kind": k, "target": (tx, ty), "since": turn}
            continue
        fb.append(f"REJECTED unit {uc.uid}: unknown order '{k}'")

    # execute standing orders (new and carried over): path one turn's worth of moves, then the arrival action.
    # A unit aboard a boat is carried: the BOAT is driven to a water tile next to the target, then the unit steps ashore.
    boats_driven: set[int] = set()
    # own boats with free capacity: a land unit may walk onto them (the old unit_move auto-embarks)
    pax_count: dict[int, int] = {}
    for x in ego.units.values():
        if int(x.get("transported_by", -1)) != -1:
            pax_count[int(x["transported_by"])] = pax_count.get(int(x["transported_by"]), 0) + 1
    board.boat_tiles = {(int(x["x"]), int(x["y"])) for bid_, x in ego.units.items()
                        if (rules.units.get(str(x["type_by_name"])) and rules.units[str(x["type_by_name"])].transport_cap > pax_count.get(bid_, 0))}
    for uid in list(standing):
        so = standing.get(uid)
        if so is None:                       # popped while iterating (a boat taken over by its passenger's order)
            continue
        if uid not in ego.units:
            standing.pop(uid); fb.append(f"NOTE unit {uid}: no longer exists, its orders are dropped"); continue
        u = ego.units[uid]; ut = rules.units.get(str(u["type_by_name"]))
        uclass = ut.uclass if ut else "Land"; igter = bool(ut and ut.igter)
        here = (int(u["x"]), int(u["y"])); k = so["kind"]; moves = int(u["moves"])
        tb = int(u.get("transported_by", -1)); boat = ego.units.get(tb) if tb != -1 else None
        # an inherited AI destination is often a ferry point or an illegal spot: fall back to the automatic site choice
        if so.get("inherited") and k == "found_city" and so.get("target") is not None and \
                not is_legal_site(board, board.geo, tuple(so["target"]), cities_by_tile, ego.name, rules.citymindist):
            so["target"] = None
        so.pop("inherited", None)
        # --- resolve an automatic founding target (target None) ---
        if so.get("target") is None:
            if k != "found_city":
                standing.pop(uid); continue
            reserved = [tuple(o["target"]) for i2, o in standing.items() if i2 != uid and o.get("kind") == "found_city" and o.get("target")]
            def free(site):
                return all(board.geo.distance(site, r_) >= rules.citymindist for r_ in reserved)
            if boat is not None:
                bt = rules.units.get(str(boat["type_by_name"])); bclass = bt.uclass if bt else "Trireme"
                cands = [c for c in sea_sites_from(board, board.geo, (int(boat["x"]), int(boat["y"])), bclass, cities_by_tile, ego.name, rules.citymindist, board.extras) if free(c[2])]
                if not cands:
                    fb.append(f"FAILED unit {uid} Settlers: no legal coastal city site reachable by boat {tb}; explore first or give a target"); standing.pop(uid); continue
                so["target"] = cands[0][2]
            else:
                cands = [c for c in land_sites_from(board, board.geo, [here], cities_by_tile, ego.name, rules.citymindist, board.extras) if free(c[2])]
                if not cands:
                    # no site on this landmass: walk to the nearest own boat with room and embark; the sea search runs next turn
                    best_b = None
                    for bt_ in board.boat_tiles:
                        pth = board.path(uclass, here, bt_, igter=igter)
                        if pth is not None and (best_b is None or len(pth) < len(best_b[1])):
                            best_b = (bt_, pth)
                    if best_b is None:
                        fb.append(f"FAILED unit {uid} Settlers: no legal city site reachable on foot from {here[0]},{here[1]} and no boat with room nearby; build a Trireme or explore"); standing.pop(uid); continue
                    so["target"] = best_b[0]; so["embark"] = True
                    fb.append(f"OK unit {uid} Settlers: no site left on this landmass, walking to the boat at {best_b[0][0]},{best_b[0][1]} to sail for one")
                else:
                    so["target"] = cands[0][2]
            if not so.get("embark"):
                fb.append(f"OK unit {uid} Settlers: will found a city at {so['target'][0]},{so['target'][1]} (nearest good legal site)")
        tgt = tuple(so["target"])
        if so.get("embark") and tb != -1:                    # aboard now: choose a coastal site next turn
            so["target"] = None; so.pop("embark", None); fb.append(f"OK unit {uid} {u['type_by_name']}: aboard boat {tb}, choosing a coastal site"); continue
        attack_like = k in ("attack", "conquer")
        adjacent_goal = attack_like or k in ("help_wonder", "trade_route", "home_city", "join_city") and tgt in cities_by_tile and cities_by_tile[tgt][0] != ego.name
        # --- aboard a boat: drive the boat, then step ashore ---
        if boat is not None and here != tgt:
            bid = tb; bt = rules.units.get(str(boat["type_by_name"])); bclass = bt.uclass if bt else "Trireme"
            bpos = (int(boat["x"]), int(boat["y"])); bmoves = int(boat["moves"]) if bid not in boats_driven else 0
            landings = [nb for _, nb in board.geo.neighbours(*tgt) if board.is_known(*nb) and board.native_for(bclass, *nb) and nb not in board.blocked]
            if bpos in landings or board.geo.adjacent(bpos, tgt):
                cost = board.step_cost(uclass, bpos, tgt, "E", igter)
                if tgt in board.blocked and not attack_like:
                    fb.append(f"WAIT unit {uid} {u['type_by_name']}: landing tile {tgt[0]},{tgt[1]} is occupied"); continue
                lua.append(_lua("move", uid, f"{tgt[0]},{tgt[1]}", f"edit.unit_move(u, find.tile({tgt[0]},{tgt[1]}), {min(cost, moves)})"))
                moves -= min(cost, moves)
                if k == "found_city" and moves > 0:
                    standing.pop(uid); lua.append(_lua(k, uid, f"{tgt[0]},{tgt[1]}", f"edit.perform_action(u, find.action('Found City'), find.tile({tgt[0]},{tgt[1]}))"))
                elif k == "move":
                    standing.pop(uid); fb.append(f"OK unit {uid} {u['type_by_name']}: lands at {tgt[0]},{tgt[1]}")
                else:
                    fb.append(f"OK unit {uid} {u['type_by_name']}: lands at {tgt[0]},{tgt[1]}, {k} next turn")
                continue
            if not landings:
                standing.pop(uid); fb.append(f"FAILED unit {uid} {u['type_by_name']} {k} -> {tgt[0]},{tgt[1]}: no water tile next to it for boat {bid}"); continue
            best = None
            for w in landings:
                pth = board.path(bclass, bpos, w)
                if pth is not None and (best is None or sum(c for _, _, c in pth) < sum(c for _, _, c in best)):
                    best = pth
            if best is None:
                standing.pop(uid); fb.append(f"FAILED unit {uid} {u['type_by_name']} {k} -> {tgt[0]},{tgt[1]}: boat {bid} has no known sea route there"); continue
            if bid in boats_driven:
                fb.append(f"WAIT unit {uid} {u['type_by_name']}: boat {bid} is already carrying out another order this turn"); continue
            boats_driven.add(bid); set_unit(bid, activity=0, ssa=SSA_NONE); standing.pop(bid, None)
            steps = []
            for dirname, tile, cost in best:
                if bmoves <= 0:
                    break
                steps.append((tile, min(cost, bmoves))); bmoves -= min(cost, bmoves)
            for tile, cost in steps:
                lua.append(_lua("move", bid, f"{tile[0]},{tile[1]}", f"edit.unit_move(u, find.tile({tile[0]},{tile[1]}), {cost})"))
            fb.append(f"OK unit {uid} {u['type_by_name']}: boat {bid} sails towards {tgt[0]},{tgt[1]} ({len(best) - len(steps)} sea steps left after this turn)")
            continue
        # --- on land / at sea under own power ---
        if here == tgt or (adjacent_goal and board.geo.adjacent(here, tgt)):
            path = []
        else:
            path = board.path(uclass, here, tgt, igter=igter, allow_goal_blocked=attack_like or adjacent_goal)
            if path is None and k == "found_city" and so.get("retries", 0) < 3:
                so["target"] = None; so["retries"] = so.get("retries", 0) + 1
                fb.append(f"NOTE unit {uid} Settlers: {tgt[0]},{tgt[1]} is not reachable; picking another site next turn"); continue
            if path is None:
                standing.pop(uid)
                tname = board.rules.by_char.get(board.tchar(*tgt), {}).get("name", "?")
                why = "the tile is occupied by a foreign unit or city" if tgt in board.blocked and not (attack_like or adjacent_goal) else \
                      f"{tgt[0]},{tgt[1]} is {tname}, not terrain this unit can enter" if not board.native_for(uclass, *tgt) and not (attack_like or adjacent_goal) else \
                      "no known route over explored tiles (use a boat for other islands)"
                fb.append(f"FAILED unit {uid} {u['type_by_name']} {k} -> {tgt[0]},{tgt[1]}: {why}"); continue
            if adjacent_goal:
                path = path[:-1]                                    # stop next to the target
        steps = []
        for dirname, tile, cost in path:
            if moves <= 0:
                break
            steps.append((tile, min(cost, moves))); moves -= min(cost, moves)
        for tile, cost in steps:
            lua.append(_lua("move", uid, f"{tile[0]},{tile[1]}", f"edit.unit_move(u, find.tile({tile[0]},{tile[1]}), {cost})"))
        arrived = (len(steps) == len(path))
        if arrived and moves > 0 or (arrived and k in ("move",)):
            if k == "move":
                standing.pop(uid); fb.append(f"OK unit {uid} {u['type_by_name']}: will reach {tgt[0]},{tgt[1]} this turn" if steps else f"OK unit {uid} {u['type_by_name']}: already at {tgt[0]},{tgt[1]}")
            elif k == "attack":
                standing.pop(uid)
                if not any(o != ego.name for o, _ in units_by_tile.get(tgt, [])) and not (tgt in cities_by_tile and cities_by_tile[tgt][0] != ego.name):
                    fb.append(f"NOTE unit {uid} {u['type_by_name']}: nothing left to attack at {tgt[0]},{tgt[1]}, order dropped")
                else:
                    lua.append(_lua("attack", uid, f"{tgt[0]},{tgt[1]}", f"edit.perform_action(u, find.action('Attack'), find.tile({tgt[0]},{tgt[1]}))"))
            elif k == "conquer":
                standing.pop(uid)
                if not steps or steps[-1][0] != tgt:
                    lua.append(_lua("move", uid, f"{tgt[0]},{tgt[1]}", f"edit.unit_move(u, find.tile({tgt[0]},{tgt[1]}), {max(1, moves)})"))
            elif k in TILE_ACTIONS:
                if so.get("embark"):
                    continue                                    # stepping onto the boat is the whole move this turn
                if k == "found_city" and not is_legal_site(board, board.geo, tgt, cities_by_tile, ego.name, rules.citymindist):
                    so["target"] = None; fb.append(f"NOTE unit {uid} Settlers: {tgt[0]},{tgt[1]} is no longer a legal city site (a city appeared nearby); picking another site next turn"); continue
                standing.pop(uid); lua.append(_lua(k, uid, f"{tgt[0]},{tgt[1]}", f"edit.perform_action(u, find.action('{TILE_ACTIONS[k]}'), find.tile({tgt[0]},{tgt[1]}))"))
            elif k in CITY_ACTIONS:
                standing.pop(uid)
                if tgt in cities_by_tile:
                    owner, crow = cities_by_tile[tgt]; owner_idx = [str(sg.get(s, "name")) for s in sg.players()].index(owner)
                    lua.append(_lua(k, uid, str(crow["id"]), f"edit.perform_action(u, find.action('{CITY_ACTIONS[k]}'), find.city(find.player({owner_idx}),{int(crow['id'])}))"))
                else:
                    fb.append(f"FAILED unit {uid} {k}: no city at {tgt[0]},{tgt[1]}")
        else:
            fb.append(f"OK unit {uid} {u['type_by_name']}: en route to {tgt[0]},{tgt[1]} ({len(path) - len(steps)} steps left after this turn)")

    # ---- treaties (proxy diplomacy, 2026-09-17): the engine has no treaty API for an unconnected human, so the
    # driver applies the AI's own acceptance logic in simplified form and edits both diplstate rows ----
    for tr in plan.treaties:
        other, kind = str(tr.get("with", "")).strip(), str(tr.get("type", "")).strip().lower()
        names = [str(sg.get(s2, "name")) for s2 in sg.players()]
        if other not in names or other == ego.name:
            fb.append(f"REJECTED treaty with '{other}': no such civilization"); continue
        j = names.index(other); ns = sg.players()
        d_me, d_them = sg.table(ego.sec, "diplstate"), sg.table(ns[j], "diplstate")
        if d_me is None or d_them is None or j >= len(d_me.rows) or ego.idx >= len(d_them.rows):
            fb.append(f"REJECTED treaty with {other}: no diplomatic record"); continue
        cur = str(d_me.rows[j][d_me.col("current")])
        if cur == "Never met":
            fb.append(f"REJECTED treaty with {other}: you have never met them"); continue
        ai_t = sg.table(ns[j], "ai"); love = countdown = None
        if ai_t is not None and ego.idx < len(ai_t.rows):
            row = dict(zip(ai_t.columns, ai_t.rows[ego.idx])); love = int(row.get("love", 0) or 0); countdown = int(row.get("countdown", -1) or -1)
        def set_state(new: str, turns_left: int):
            for me_i, them_i in ((ego.idx, j), (j, ego.idx)):
                sg.set_cell(ns[me_i], "diplstate", them_i, "current", new)
                sg.set_cell(ns[me_i], "diplstate", them_i, "turns_left", turns_left)
                if new in ("Peace", "Armistice", "Alliance"):
                    sg.set_cell(ns[me_i], "diplstate", them_i, "closest", new if new != "Armistice" else "Peace")
        if kind == "war":
            if cur == "War":
                fb.append(f"REJECTED war on {other}: already at war")
            else:
                set_state("War", 0); fb.append(f"OK you declared war on {other}")
        elif kind == "ceasefire":
            if cur != "War":
                fb.append(f"REJECTED ceasefire with {other}: you are not at war (state {cur})")
            elif love is not None and (love < -100 or countdown not in (None, -1)):
                fb.append(f"REFUSED: {other} rejects a ceasefire" + (" (they are preparing to fight you)" if countdown not in (None, -1) else " (they are hostile to you)"))
            else:
                set_state("Cease-fire", 16); fb.append(f"OK ceasefire with {other} (16 turns; make peace before it lapses back into war)")
        elif kind == "peace":
            if cur in ("Peace", "Armistice", "Alliance"):
                fb.append(f"REJECTED peace with {other}: already at {cur}")
            elif cur == "War":
                fb.append(f"REFUSED: {other} wants a ceasefire before talking peace")
            elif love is not None and love < 100:
                fb.append(f"REFUSED: {other} is not friendly enough for a peace treaty yet")
            else:
                set_state("Armistice", 16); fb.append(f"OK armistice with {other}: becomes Peace in 16 turns")
        else:
            fb.append(f"REJECTED treaty '{kind}' with {other}: use ceasefire, peace or war")

    if pending_gov_line is not None:
        _insert_after(sg, sec, "government_name", pending_gov_line)
    return lua, fb, warn


_AGENT = re.compile(r"AGENT\|([^|]*)\|([^|]*)\|([^|]*)\|(\S*)")



def drop_phantom_events(events: list[dict[str, Any]], sg_after: Savegame) -> list[dict[str, Any]]:
    """Per-turn stepping artefact (2026-09-18): the AI's `city_built` + `unit_lost(reason=used)` signals fire for foundings
    the engine then rolls back within the same turn -- the city never exists in the T+1 save and the Settlers unit is still
    alive (s500030: 227 phantom foundings by one AI in 100 turns; the mainline corpus has 7 real ones).  Keep only events
    whose objects survive in the save the turn produced: a `city_founded` needs its city id in the save; a `unit_lost`
    with reason `used` needs its unit id to be GONE from the save."""
    city_ids: set[int] = set(); unit_ids: set[int] = set()
    for sec in sg_after.players():
        for key, bag in (("c", city_ids), ("u", unit_ids)):
            t = sg_after.table(sec, key)
            for r in (t.as_dicts() if t else []):
                try:
                    bag.add(int(r["id"]))
                except (KeyError, ValueError, TypeError):
                    pass
    out = []
    for e in events:
        if e.get("type") == "city_founded" and e.get("city_id") is not None and int(e["city_id"]) not in city_ids:
            continue
        if e.get("type") == "unit_lost" and e.get("reason") == "used" and e.get("unit_id") is not None and int(e["unit_id"]) in unit_ids:
            continue
        out.append(e)
    return out

def run_turn(workdir: Path, sg: Savegame, ego_idx: int, lua_lines: list[str], turn: int, timeout: int = 600,
             image: str = DOCKER_IMG, retries: int = 5) -> StepResult:
    """Write the edited save, run exactly one turn with the hooks + our Lua, return the result.

    ★ Turn loop after a load: begin_phase(is_new=FALSE) skips the per-turn refresh (move points, activities,
    AI first activities); the refresh for turn T+1 happens in begin_phase(TRUE) and the autosave of T+1 is written
    right after it.  So we run to endturn T+1 and take the T+1 *autosave* -- the same point a normal reload resumes
    from -- not the game-over save (which would leave our units with last turn's leftover move points forever).

    ★ Failure modes seen 2026-09-17: a rejected savegame edit, or a transient "Could not find a readable
    techs.ruleset" when several containers start at once, both make the server start a NEW random game from turn 1
    (no error in the exit code).  Detected by the missing T+1 autosave / a T0001 autosave / a changed roster;
    retried `retries` times, then raised."""
    import time
    t0 = time.time()
    workdir = Path(workdir).resolve()
    roster = [str(sg.get(s, "name")) for s in sg.players()]
    last_err = ""
    for attempt in range(retries + 1):
        shutil.rmtree(workdir / "saves", ignore_errors=True); (workdir / "saves").mkdir(parents=True)
        (workdir / "game.log").unlink(missing_ok=True)          # the server appends; a stale log would replay last turn's events
        sg.write(workdir / "start.sav")
        settings = dict(BASE_SETTINGS); settings.update({"saveturns": 1, "endturn": turn + 1})
        lua = list(hook_lua()) + [l.replace("{PID}", str(ego_idx)) for l in lua_lines]
        (workdir / "run.serv").write_text(serv_script(settings, lua))
        # ★ 2026-09-17: reads of ruleset files from the image's upper overlay layer fail intermittently for minutes at a
        # time ("Could not find a readable techs.ruleset" -> the server silently starts a new game).  Bind-mounting the
        # ruleset directories from the host copy (bench/agentplay/rulesets, byte-identical to the image) bypasses the layer.
        rs_root = ROOT / "bench/agentplay/rulesets"
        mounts = []
        for d in sorted(rs_root.iterdir()):
            if d.is_dir() and (d / "game.ruleset").exists():
                mounts += ["-v", f"{d}:/opt/freeciv/share/freeciv/{d.name}:ro"]
        cmd = ["docker", "run", "--rm", "--user", f"{os.getuid()}:{os.getgid()}", "-v", f"{workdir}:/work", *mounts, image, "freeciv-server",
               "-f", "/work/start.sav", "-r", "/work/run.serv", "-e", "-s", "/work/saves", "-l", "/work/game.log"]
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, encoding="latin-1", timeout=timeout)
            out = p.stdout + p.stderr
        except subprocess.TimeoutExpired:
            last_err = f"ENGINE timeout after {timeout}s"; continue
        log = (workdir / "game.log").read_text(encoding="latin-1") if (workdir / "game.log").exists() else ""
        saves = sorted((workdir / "saves").glob(f"freeciv-T{turn + 1:04d}-*-auto.sav"))
        # only LOAD failures count (engine assertions logged at level 1 during play are not: the game goes on)
        bad = [l for l in log.splitlines() if "Could not find a readable" in l or "Failure loading savegame" in l or "Error loading" in l]
        if not saves or (workdir / "saves" / f"freeciv-T0001-Y-4000-auto.sav").exists() or bad:
            last_err = f"no T{turn + 1} autosave / new game started (rc={p.returncode}): {' | '.join(bad[:2])[:300]} {out[-200:]}"
            time.sleep(5 * 2 ** attempt); continue          # 5, 10, 20, 40, 80 s: the failure seen so far lasted ~2 min
        new = saves[-1]
        sg2 = Savegame.load(new)
        roster2 = [str(sg2.get(s, "name")) for s in sg2.players()]
        # the ego must still be there under the same slot; barbarian/pirate players may vanish (server_remove_player) and
        # civil wars may append players, so only the ego slot is checked (a new-game fallback is caught above anyway)
        if len(roster2) <= ego_idx or roster2[ego_idx] != roster[ego_idx]:
            last_err = f"roster changed {roster} -> {roster2}"; time.sleep(2); continue
        results = []; errors = []; lines = out.splitlines()
        for i, l in enumerate(lines):
            m = _AGENT.search(l)
            if m:
                results.append({"kind": m.group(1), "uid": int(m.group(2)) if m.group(2).isdigit() else m.group(2), "detail": m.group(3), "ok": m.group(4)})
            elif l.startswith("lua error") and 'Signal "' not in " ".join(lines[i:i + 3]):   # hook_lua's unit_lost_with_transport is absent in this build; harmless
                errors.append("\n".join(lines[i:i + 3]))
        by_turn, _ = parse_log(log)
        events = [e for t, es in by_turn.items() for e in es]
        events = drop_phantom_events(events, sg2)
        return StepResult(True, new, sg2.turn, [], results, errors, events, time.time() - t0, out)
    raise RuntimeError(f"engine step failed after {retries + 1} attempts: {last_err}")


def feedback_from_results(res: StepResult, ego: Ego) -> list[str]:
    """Turn the AGENT| console lines into sentences the model can read."""
    fb = []
    for r in res.lua_results:
        u = ego.units.get(r["uid"]) if isinstance(r["uid"], int) else None
        name = f"unit {r['uid']}" + (f" {u['type_by_name']}" if u else "")
        if r["ok"] == "nounit":
            fb.append(f"FAILED {name} {r['kind']}: the unit no longer exists")
        elif r["kind"] == "move":
            fb.append(f"{'moved' if r['ok'] == 'true' else 'COULD NOT move'} {name} to {r['detail']}")
        else:
            fb.append(f"{'OK' if r['ok'] == 'true' else 'FAILED'} {name} {r['kind']}" + (f" at {r['detail']}" if r["detail"] not in ("", "here") else ""))
    for e in res.lua_errors:
        fb.append("ENGINE lua error: " + e.replace("\n", " ")[:200])
    return fb
