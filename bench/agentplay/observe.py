"""What the agent sees each turn: the fog-of-war world report (the benchmark's own `pomdp` renderer, so the
agent reads exactly the observation format the world model was evaluated on) plus an EGO PANEL with the
things a player at a client would have in front of them and the report leaves out: production progress and
the build menu per city, research options, unit status with ids and standing orders, alerts (idle units,
foreign units near you, what finished last turn) and the engine's verdict on last turn's orders.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "bundle")); sys.path.insert(0, str(ROOT))
from sandboxes.freeciv.geo import extract_map      # noqa: E402
from sandboxes.freeciv.report import render        # noqa: E402
from sandboxes.freeciv.savegame import Savegame    # noqa: E402
from sandboxes.freeciv.state import extract        # noqa: E402

from bench.agentplay.engine import Ego             # noqa: E402
from bench.agentplay.hexmap import Board, Geometry # noqa: E402
from bench.agentplay.rules import Rules            # noqa: E402

ACTIVITY_NAME = {0: "idle", 1: "cleaning pollution", 3: "mining", 4: "irrigating", 5: "fortified", 6: "building fortress", 7: "sentry",
                 9: "pillaging", 10: "goto", 11: "exploring", 12: "transforming", 15: "fortifying", 16: "cleaning fallout",
                 18: "building base", 19: "building road", 20: "converting", 21: "cultivating", 22: "planting"}


def _tile_pred_factory(board: Board, mi, x: int, y: int):
    tiles = [(x, y)] + [t for _, t in board.geo.neighbours(x, y)]

    def pred(req_type: str, name: str) -> bool:
        for tx, ty in tiles:
            ch = board.tchar(tx, ty); info = board.rules.by_char.get(ch, {})
            if req_type == "TerrainClass" and info.get("class") == name:
                return True
            if req_type == "Terrain" and info.get("name") == name:
                return True
            if req_type == "Extra" and name in mi.extras[ty][tx]:
                return True
            if req_type == "TerrainFlag":
                if name == "FreshWater" and info.get("name") in ("Lake",) or name == "FreshWater" and "River" in mi.extras[ty][tx]:
                    return True
        return False
    return pred


def build_view(sg: Savegame, world_id: str, ego_idx: int, rules: Rules, standing: dict[int, dict[str, Any]],
               feedback: list[str], events: list[dict[str, Any]], max_menu: int = 40, gold_hist: list[int] | None = None) -> dict[str, Any]:
    """Returns {"report": str, "panel": str, "alerts": list[str], "ego": Ego, "board": Board, "mi": MapInfo, "ws": WorldState}."""
    ws = extract(sg, world_id)
    names = [p.name for p in ws.players]
    mi = extract_map(sg, names)
    ego = Ego(sg, ego_idx)
    me = ws.players[ego_idx]
    geo = Geometry(mi.xsize, mi.ysize)
    board = Board(geo, rules.terrain, mi.terrain, mi.extras, mi.known[ego.name], owner=mi.owner, player_names=names)
    report = render(ws, "pomdp", view_player=ego.name, mapinfo=mi)

    L: list[str] = []
    turn = sg.turn
    L.append(f"### EGO PANEL  (you are {ego.name}; turn {turn})")
    sc = f"score{ego_idx}"
    L.append(f"score {me.score} · gold {ego.gold} · government {ego.gov}" +
             (f" (revolution ends turn {sg.get(ego.sec, 'revolution_finishes')})" if ego.gov == "Anarchy" else "") +
             f" · rates tax {sg.get(ego.sec, 'rates.tax')}% science {sg.get(ego.sec, 'rates.science')}% luxury {sg.get(ego.sec, 'rates.luxury')}%"
             f" (max {__import__('bench.agentplay.engine', fromlist=['MAX_RATE']).MAX_RATE.get(ego.gov, 100)}% under {ego.gov})")
    L.append(f"trade {sg.get(sc, 'bnp')} · production {sg.get(sc, 'mfg')} · science {sg.get(sc, 'techout')} bulbs/turn · "
             f"units built {sg.get(sc, 'units_built')} killed {sg.get(sc, 'units_killed')} lost {sg.get(sc, 'units_lost')}")
    if ego.research_row is not None:
        r = ego.research
        goal = r["goal_name"] if r["goal_name"] not in ("A_UNSET", "", None) else "(none)"
        L.append(f"research: now {r['now_name']} ({r['bulbs']} bulbs stored) · goal {goal} · researchable now: " + ", ".join(rules.researchable(ego.known_techs)))
    govs = [g for g, reqs in rules.governments.items() if g not in ("Anarchy", ego.gov) and all(q["type"].lower() != "tech" or q["name"] in ego.known_techs for q in reqs)]
    L.append("governments available: " + (", ".join(govs) or "none besides the current one"))

    # ---- cities ----
    wonders_gone = set()
    for sec in sg.players():
        t = sg.table(sec, "c")
        if t:
            imprs = list(sg.get("savefile", "improvement_vector") or [])
            for c in t.as_dicts():
                for name, b in zip(imprs, str(c["improvements"])):
                    if b == "1" and name in rules.buildings and rules.buildings[name].genus == "GreatWonder":
                        wonders_gone.add(name)
    imprs = list(sg.get("savefile", "improvement_vector") or [])
    own_small = set()
    for c in ego.cities.values():
        own_small |= {n for n, b in zip(imprs, str(c["improvements"])) if b == "1" and n in rules.buildings and rules.buildings[n].genus == "SmallWonder"}
    L.append("")
    L.append("### YOUR CITIES: production and options")
    for cname, c in ego.cities.items():
        x, y = int(c["x"]), int(c["y"])
        have = {n for n, b in zip(imprs, str(c["improvements"])) if b == "1"}
        item = str(c["currently_building_name"]); cost = rules.cost_of(item); stock = int(c["shield_stock"]); surplus = int(c.get("last_turns_shield_surplus", 0))
        eta = "?" if cost is None else ("done next turn" if stock >= cost else (f"~{math.ceil((cost - stock) / surplus)} turns" if surplus > 0 else "stalled (no surplus)"))
        coastal = any(board.rules.is_water(board.tchar(*t)) for _, t in geo.neighbours(x, y))
        um = rules.unit_menu(ego.known_techs, have, coastal)
        bm = [b for b in rules.building_menu(ego.known_techs, have, int(c["size"]), wonders_gone, _tile_pred_factory(board, mi, x, y)) if b.name not in own_small]
        L.append(f"- {cname} ({x},{y}) size {c['size']}: building {item} {stock}/{cost} shields, +{surplus}/turn, {eta}"
                 + (" · bought this turn" if bool(c.get("did_buy")) else "") + (f" · disorder" if int(c.get('anarchy', 0)) > 0 else "")
                 + f"; has: {', '.join(sorted(have)) or 'nothing'}")
        L.append("    can build units: " + ", ".join(f"{u.name}[{u.build_cost}]" for u in um[:max_menu]))
        L.append("    can build: " + ", ".join(f"{b.name}[{b.build_cost}]" + ("*" if b.genus == "GreatWonder" else "") for b in bm[:max_menu]))
    # ---- units ----
    L.append("")
    L.append("### YOUR UNITS: status  (id type @x,y hp moves-left/turn · activity · orders)  -- ONLY these ids can be ordered; ids in the report's FOREIGN UNITS section belong to other civilizations")
    idle = []
    for uid, u in ego.units.items():
        ut = rules.units.get(str(u["type_by_name"]))
        act = ACTIVITY_NAME.get(int(u["activity"]), f"activity{u['activity']}")
        ssa = int(u["server_side_agent"])
        auto = " · AUTO-WORKER" if ssa == 1 else " · AUTO-EXPLORE" if ssa == 2 else ""
        so = standing.get(uid)
        order = (f" · standing order: {so['kind']} -> {so['target'][0]},{so['target'][1]}" if so and so.get("target") else
                 f" · standing order: {so['kind']} (site chosen automatically)" if so else "")
        mv = f"{int(u['moves']) / rules.terrain.move_fragments:g}/{ut.move_rate if ut else '?'}"
        tb = int(u.get("transported_by", -1))
        home = f" · ABOARD boat {tb}" if tb != -1 else ""
        pax = [f"{o} {r['type_by_name']}" for o, r in ego.units.items() if int(r.get("transported_by", -1)) == uid]
        if pax:
            home += " · carrying " + ", ".join(pax)
        tname = rules.terrain.by_char.get(board.tchar(int(u["x"]), int(u["y"])), {}).get("name", "?")
        L.append(f"- {uid} {u['type_by_name']} @{u['x']},{u['y']} ({tname}) hp {u['hp']} mv {mv} · {act}{auto}{order}{home}")
        if int(u["activity"]) == 0 and ssa == 0 and so is None:
            idle.append(f"{uid} {u['type_by_name']} @{u['x']},{u['y']}")
    # ---- candidate city sites (a client's "city site advisor": known land, >= citymindist from every known city,
    # reachable on foot from one of our cities or settlers; ranked by terrain) ----
    sites = city_sites(board, geo, rules, ego, mi)
    L.append("")
    L.append("### CANDIDATE CITY SITES  (explored land, at least %d tiles from any known city, reachable on foot; best first)" % rules.citymindist)
    L.append("- " + "; ".join(sites) if sites else "- none within reach (explore, or carry Settlers by boat)")
    # ---- alerts ----
    alerts: list[str] = []
    if idle:
        alerts.append(f"{len(idle)} unit(s) without orders: " + "; ".join(idle))
    aboard = [f"{uid} {u['type_by_name']} aboard {u['transported_by']}" for uid, u in ego.units.items() if int(u.get("transported_by", -1)) != -1]
    if aboard:
        alerts.append("units at sea aboard boats (move the boat next to land, then order the unit onto a land tile): " + "; ".join(aboard))
    own_tiles = {(int(u["x"]), int(u["y"])) for u in ego.units.values()} | {(int(c["x"]), int(c["y"])) for c in ego.cities.values()}
    rel = {names[int(k[6:])]: v for k, v in me.diplomacy.items() if k.startswith("player") and int(k[6:]) < len(names)}
    near = []
    for sec in sg.players():
        if sec == ego.sec:
            continue
        owner = str(sg.get(sec, "name")); t = sg.table(sec, "u")
        if not t or not bool(sg.get(sec, "is_alive", True)):
            continue
        for u in t.as_dicts():
            pos = (int(u["x"]), int(u["y"]))
            if not board.is_known(*pos):
                continue
            d = min(geo.distance(pos, o) for o in own_tiles) if own_tiles else 99
            if d <= 2:
                near.append(f"{owner} ({rel.get(owner, 'unknown')}) {u['type_by_name']} @{pos[0]},{pos[1]} at distance {d}")
    if near:
        alerts.append("foreign units close to you: " + "; ".join(near[:12]))
    mine = [e for e in events if any(str(v) == ego.name or (isinstance(v, str) and v.endswith(f"#{ego_idx}")) for v in e.values())
            or e.get("type") in ("city_built", "unit_built", "building_built", "city_size_change") and _owned(e, ego)]
    if mine:
        alerts.append("last turn: " + "; ".join(_event_text(e) for e in mine[:15]))
    if ego.gov == "Anarchy":
        alerts.append("you are in Anarchy (revolution in progress)")
    # ---- advisor: objective facts a client shows on its dialogs ----
    adv = []
    mil_here: dict[tuple[int, int], int] = {}
    for u in ego.units.values():
        ut = rules.units.get(str(u["type_by_name"]))
        if ut is not None and ut.military and int(u.get("transported_by", -1)) == -1:
            mil_here[(int(u["x"]), int(u["y"]))] = mil_here.get((int(u["x"]), int(u["y"])), 0) + 1
    undefended = [n for n, c in ego.cities.items() if mil_here.get((int(c["x"]), int(c["y"])), 0) == 0]
    if undefended:
        adv.append(f"cities with NO military unit inside: {', '.join(undefended)}")
    enemies_near = []
    for sec in sg.players():
        if sec == ego.sec:
            continue
        owner = str(sg.get(sec, "name"))
        if rel.get(owner) != "War":
            continue
        t = sg.table(sec, "u")
        for u in (t.as_dicts() if t else []):
            pos = (int(u["x"]), int(u["y"]))
            if not board.is_known(*pos):
                continue
            ut = rules.units.get(str(u["type_by_name"]))
            if ut is None or not ut.military:
                continue
            for n, c in ego.cities.items():
                d = geo.distance(pos, (int(c["x"]), int(c["y"])))
                if d <= 3:
                    enemies_near.append(f"{u['type_by_name']} ({owner}) {d} tiles from {n}")
    if enemies_near:
        adv.append("ENEMY military near your cities: " + "; ".join(enemies_near[:8]))
    n_settlers = sum(1 for u in ego.units.values() if str(u["type_by_name"]) == "Settlers")
    adv.append(f"you have {len(ego.cities)} cities and {n_settlers} Settlers; the AI civilizations typically reach 8-15 cities by turn 100 -- expansion is the main score lever")
    if gold_hist and len(gold_hist) >= 5:
        dg = gold_hist[-1] - gold_hist[-5]
        if dg < 0:
            adv.append(f"treasury fell by {-dg} gold over the last 4 turns (now {gold_hist[-1]}); raise taxes or stop buying before it hits 0 (buildings get sold)")
    disorder = [n for n, c in ego.cities.items() if int(c.get("anarchy", 0)) > 0]
    if disorder:
        adv.append("cities in civil disorder: " + ", ".join(disorder))
    # ---- siege arithmetic (doc 17 §4.4): rule-based estimate of whether the garrison / walls hold ----
    try:
        from bench.agentplay.siege import threat_block
        siege = threat_block(sg, rules, board, geo, ego, rel)
    except Exception as e:                         # noqa: BLE001
        siege = [f"(siege arithmetic unavailable: {type(e).__name__}: {e})"]
    L.append("")
    L.append("### ADVISOR (facts)")
    L.extend(f"- {a}" for a in adv)
    if siege:
        L.append("")
        L.append("### " + siege[0])
        L.extend(siege[1:])
    L.append("")
    L.append("### ALERTS")
    L.extend(f"- {a}" for a in alerts) if alerts else L.append("- nothing new")
    if feedback:
        L.append("")
        L.append("### RESULT OF YOUR LAST ORDERS")
        L.extend(f"- {f}" for f in feedback)
    return {"report": report, "panel": "\n".join(L), "alerts": alerts, "advisor": adv, "siege": siege, "ego": ego, "board": board, "mi": mi, "ws": ws, "idle": idle,
            "sites": sites, "geo": geo, "rel": rel}


def _owned(e: dict, ego: Ego) -> bool:
    c = str(e.get("city", ""))
    return any(c.split("#")[0] == n for n in ego.cities) or str(e.get("owner", "")).split("#")[0] == ego.name


def _event_text(e: dict) -> str:
    t = e.get("type", "?")
    keep = {k: str(v).split("#")[0] for k, v in e.items() if k not in ("type", "turn", "engine_turn") and v not in (None, "")}
    return t + " " + " ".join(f"{k}={v}" for k, v in keep.items())


TERRAIN_SCORE = {"Grassland": 3, "Plains": 3, "Hills": 3, "Desert": 1, "Tundra": 1, "Forest": 1, "Jungle": 1, "Swamp": 1}


def city_sites(board: Board, geo: Geometry, rules: Rules, ego: Ego, mi, max_dist: int = 8, top: int = 10) -> list[str]:
    cities = {t: (o, c) for t, (o, c) in ego.all_cities_by_tile().items() if board.is_known(*t)}
    units_by_tile = ego.all_units_by_tile()
    board.blocked = {t for t, lst in units_by_tile.items() if any(o != ego.name for o, _ in lst)} | {t for t, (o, _) in cities.items() if o != ego.name}
    board.enterable_cities = {t for t, (o, _) in cities.items() if o == ego.name}
    origins = [(int(c["x"]), int(c["y"]), n) for n, c in ego.cities.items()]
    origins += [(int(u["x"]), int(u["y"]), f"Settlers {uid}") for uid, u in ego.units.items()
                if str(u["type_by_name"]) == "Settlers" and int(u.get("transported_by", -1)) == -1]
    if not origins:
        return []
    # multi-source BFS in hex steps (not move cost) from every origin over known, native, unblocked land
    import collections
    dist: dict[tuple[int, int], tuple[int, str]] = {}
    dq = collections.deque()
    for x, y, n in origins:
        dist[(x, y)] = (0, n); dq.append((x, y))
    while dq:
        cur = dq.popleft(); d, src = dist[cur]
        if d >= max_dist:
            continue
        for _, nb in geo.neighbours(*cur):
            if nb in dist or not board.is_known(*nb) or nb in board.blocked or not board.native_for("Small Land", *nb):
                continue
            dist[nb] = (d + 1, src); dq.append(nb)
    out = []
    for t, (d, src) in dist.items():
        if d == 0 and t in cities:
            continue
        if any(geo.distance(t, c) < rules.citymindist for c in cities):
            continue
        if board.owner_at(*t) not in (None, ego.name):
            continue
        tname = rules.terrain.by_char.get(board.tchar(*t), {}).get("name", "?")
        sc = TERRAIN_SCORE.get(tname, 0)
        if sc == 0:
            continue
        ex = set(mi.extras[t[1]][t[0]]); tags = []
        if "River" in ex:
            sc += 1; tags.append("river")
        if any(board.rules.is_water(board.tchar(*nb)) for _, nb in geo.neighbours(*t)):
            sc += 1; tags.append("coastal")
        spec = [e for e in ex if e not in ("River", "Road", "Railroad", "Irrigation", "Mine", "Farmland", "Hut", "Ruins", "Fort", "Fortress", "Pollution", "Fallout", "Buoy", "Airbase", "Airstrip", "Maglev", "Oil Well", "Oil Platform")]
        if spec:
            sc += 1; tags.append(spec[0])
        out.append((-sc, d, t, tname, tags, src))
    out.sort()
    return [f"({t[0]},{t[1]}) {tname}{', ' + ', '.join(tags) if tags else ''} · {d} tiles from {src}" for _, d, t, tname, tags, src in out[:top]]
