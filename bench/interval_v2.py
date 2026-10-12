"""INTERVAL block as chronological news, five turns per paragraph (user 2026-09-10).

Each paragraph `turn a-b:` is a sequence of plain sentences about what happened in
those turns, in this order: diplomacy, cities changing hands, governments, disasters,
research, units built (per civ, by type), buildings / wonders completed, tile improvements,
military redeployments (per civ, merged by destination: which unit types ended up
where, whatever their origin), and combat (attacker, defender, losses, HP lost as a
share of the units' maximum HP).

Sources: engine events (`rec["events"]`) for everything but movement and hp, the
per-turn records for unit positions / hit points.  Coordinates: the event stream's
`target_x/y` are Lua map coordinates; records use native coordinates; the native tile
is decoded from `target_id` (= y * xsize + x).  Unit max HP and building genus
(building / great wonder / small wonder) come from the ruleset copies in
bench/v2/rulesets/.

A `building_built` event for a building the city already had at the start of that
turn is dropped: it is the AI selling and re-buying under treasury pressure, seen as
"completed Barracks III" four times in twenty turns (user 2026-09-10: dedupe).

View 2 (POMDP): world-level sentences pass `visible_to`; the viewer's own activity is
complete; foreign production, buildings, tile work and movements are omitted; combat is
reported only when the viewer is a party.
"""
from __future__ import annotations

import collections
from functools import lru_cache
from pathlib import Path

from bench.render import visible_to
from sandboxes.freeciv.ruleset import parse

RULESETS = Path(__file__).resolve().parent / "v2/rulesets"
ATTACK_ACTIONS = ("Attack", "Suicide Attack", "Nuke Units", "Nuke City", "Explode Nuclear")
BOMBARD = ("Bombard",)
CAPTURE = ("Capture Units",)
TILE_WORK = {"Build Road": ("road", "roads"), "Build Irrigation": ("irrigation", "irrigation"), "Build Mine": ("mine", "mines"),
             "Clean Pollution": ("pollution clean-up", "pollution clean-ups"), "Clean Fallout": ("fallout clean-up", "fallout clean-ups"),
             "Transform Terrain": ("terrain transformation", "terrain transformations"), "Cultivate": ("cultivation", "cultivations"),
             "Plant": ("planting", "plantings")}
DIPL = ("war_declared", "ceasefire_agreed", "ceasefire_expired", "peace_treaty_signed", "peace_established",
        "alliance_formed", "alliance_ended", "first_contact", "diplstate_change")
MOVE_CAP = 25                    # destination clusters listed per civ per paragraph
MOVE_RADIUS = 3                  # destinations within this Chebyshev distance (x wraps) merge into "around (x,y)"


def _near(p, q, xsize: int) -> bool:
    dx = abs(p[0] - q[0])
    if xsize:
        dx = min(dx, xsize - dx)
    return max(dx, abs(p[1] - q[1])) <= MOVE_RADIUS


@lru_cache(maxsize=None)
def unit_max_hp(ruleset: str) -> dict[str, int]:
    u = parse(RULESETS / ruleset / "units.ruleset")
    return {str(v["name"]): int(v["hitpoints"]) for k, v in u.items() if k.startswith("unit_") and "hitpoints" in v}


@lru_cache(maxsize=None)
def building_genus(ruleset: str) -> dict[str, str]:
    b = parse(RULESETS / ruleset / "buildings.ruleset")
    return {str(v["name"]): str(v["genus"]) for k, v in b.items() if k.startswith("building_") and "genus" in v}


def _t(e):
    return e.get("engine_turn") if e.get("engine_turn") is not None else e.get("turn")


def _dipl(e) -> str:
    a, b, t = e.get("player"), e.get("other"), e["type"]
    return {"war_declared": f"{a} declared war on {b}", "ceasefire_agreed": f"{a} and {b} agreed a cease-fire",
            "ceasefire_expired": f"the cease-fire between {a} and {b} ended and they are at war again",
            "peace_treaty_signed": f"{a} and {b} signed a peace treaty", "peace_established": f"peace took effect between {a} and {b}",
            "alliance_formed": f"{a} and {b} formed an alliance", "alliance_ended": f"the alliance between {a} and {b} ended",
            "first_contact": f"{a} and {b} made first contact"}.get(t, f"{a}'s relation with {b} changed from {e.get('from')} to {e.get('to')}")


def _units(c: collections.Counter) -> str:
    """'4 Marines, 1 Diplomat' from a type counter (order: most numerous first)."""
    return ", ".join(f"{n} {k}" for k, n in c.most_common())


def _join(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def _units_index(rec: dict) -> tuple[dict, dict, dict]:
    """unit id -> (owner, type, x, y, hp); (x,y) -> [(owner, type, id, hp)]; (x,y) -> (owner, city name)."""
    by_id, by_tile, city_at = {}, collections.defaultdict(list), {}
    for p in rec["state"]["players"]:
        for u in p.get("units") or []:
            by_id[u["id"]] = (p["name"], u["type"], u["x"], u["y"], u.get("hp"))
            by_tile[(u["x"], u["y"])].append((p["name"], u["type"], u["id"], u.get("hp")))
        for c in p.get("cities") or []:
            city_at[(c["x"], c["y"])] = (p["name"], c["name"])
    return by_id, by_tile, city_at


def _city_buildings(rec: dict) -> dict[int, set]:
    out = {}
    for p in rec["state"]["players"]:
        for c in p.get("cities") or []:
            imp = c.get("improvements") or c.get("buildings") or {}
            out[c["id"]] = {k for k, v in imp.items() if v} if isinstance(imp, dict) else set(imp)
    return out


def interval_block(recs: dict, a: int, b: int, ev: list[dict], mil: dict, view: int, viewer: str,
                   ruleset: str = "civ2civ3", world: dict | None = None, window: int = 5,
                   major_only: bool = False) -> str:
    """`major_only=True` (oracle ablation's FUTURE EVENTS): diplomacy, cities, governments,
    disasters and research only -- no production, buildings, tile work, movements or combat,
    so the oracle tells the model *what happened* at civ level without handing it the
    per-unit outcome stream."""
    idx = {t: _units_index(recs[t]) for t in range(a, b + 2) if t in recs}
    blds = {t: _city_buildings(recs[t]) for t in range(a, b + 2) if t in recs}
    xsize = int((recs[a].get("geo") or {}).get("xsize") or 0)
    maxhp = unit_max_hp(ruleset); genus = building_genus(ruleset)
    lost_at = collections.defaultdict(set)          # (turn, owner) -> unit ids lost by kill/capture
    for e in ev:
        if e["type"] == "unit_lost" and e.get("reason") in ("killed", "captured", "eliminated") and a <= (_t(e) or 0) < b:
            lost_at[(_t(e), e.get("player"))].add(e.get("unit_id"))

    _sn = {f"player{i}": q["name"] for i, q in enumerate(recs[a]["state"]["players"])}
    _rel = {(q["name"], _sn[k]): v for q in recs[a]["state"]["players"] for k, v in (q.get("diplomacy") or {}).items() if k in _sn}

    def allied(x, y) -> bool:
        return _rel.get((x, y)) in ("Alliance", "Team") or _rel.get((y, x)) in ("Alliance", "Team")

    def vis(e) -> bool:
        return view == 1 or visible_to(e, viewer, recs[a])

    def own_or_seen(civ) -> bool:
        return view == 1 or civ == viewer

    def at(t):
        return idx.get(t, ({}, {}, {}))

    reported_bld = set()
    out = ["Events (five-turn paragraphs):"]
    # events of record t happen after the turn-t snapshot -> the interval covers turns a .. b-1
    for w0 in range(a, b, window):
        w1 = min(w0 + window - 1, b - 1)
        evs = [e for e in ev if w0 <= (_t(e) or 0) <= w1]
        S: list[str] = []
        # --- diplomacy
        seen = set()
        for e in evs:
            if e["type"] in DIPL and vis(e):
                k = (_t(e), e["type"], frozenset((e.get("player"), e.get("other"))))
                if k not in seen:
                    seen.add(k); S.append(_dipl(e))
        # --- cities
        for e in evs:
            if e["type"] == "city_founded" and e.get("player") and e.get("city") and vis(e):
                S.append(f"{e['player']} founded the city of {e['city']}")
            elif e["type"] == "city_transferred" and e.get("player") and e.get("city") and vis(e):
                how = {"conquest": "by conquest", "civil_war": "in a civil war", "incited": "after being incited to revolt"}.get(e.get("reason"), "")
                S.append(f"the city of {e['city']} passed from {e.get('from_player', '?')} to {e['player']} {how}".strip())
            elif e["type"] == "city_destroyed" and e.get("city") and vis(e):
                S.append((f"{e['player']}'s city of " if e.get("player") else "the city of ") + f"{e['city']} was destroyed")
        # --- governments
        for e in evs:
            if e["type"] == "government_change" and e.get("player") and vis(e):
                S.append(f"{e['player']} changed its government from {e.get('from')} to {e.get('to')}")
        # --- disasters
        dis = collections.defaultdict(lambda: collections.defaultdict(list))
        for e in evs:
            if e["type"] == "disaster_occurred" and e.get("player") and vis(e):
                dis[e["player"]][e.get("disaster")].append(e.get("city"))
        for civ, d in dis.items():
            for kind, cities in d.items():
                # the same city can be hit several times in one window: say so once ("Shanghai (twice)")
                cnt = collections.Counter(cities)
                names = [f"{c} ({'twice' if n == 2 else str(n) + ' times'})" if n > 1 else c for c, n in cnt.items()]
                if len(cnt) == 1:
                    c0, n0 = next(iter(cnt.items()))
                    art = "an" if kind[:1].lower() in "aeiou" else "a"
                    S.append(f"{civ}'s city of {c0} suffered " + (f"{art} {kind}" if n0 == 1 else f"{kind}s {'twice' if n0 == 2 else str(n0) + ' times'}"))
                else:
                    S.append(f"{civ}'s cities {_join(names)} suffered {kind}s")
        # --- research
        tech = collections.defaultdict(lambda: collections.defaultdict(list))
        for e in evs:
            if e["type"] == "tech_discovered" and e.get("player") and e.get("tech") and vis(e):
                name = "Future Tech" if e["tech"] in ("Future", "A_FUTURE") else e["tech"]
                tech[e["player"]][e.get("how") or "researched"].append(name)
        for civ, d in tech.items():
            parts = []
            for how, ts in d.items():
                if ts.count("Future Tech") > 1:
                    ts = [t for t in ts if t != "Future Tech"] + [f"Future Tech ({ts.count('Future Tech')} times)"]
                noun = "technology" if len(ts) == 1 else "technologies"
                verb = {"researched": "discovered", "stolen": "stole", "traded": "acquired by treaty", "conquered": "captured"}.get(how, f"acquired ({how})")
                parts.append(f"{verb} the {noun} {_join(ts)}")
            S.append(f"{civ} " + " and ".join(parts))
        if major_only:
            S = [x[:1].upper() + x[1:] for x in S]
            label = f"turn {w0}" if w0 == w1 else f"turn {w0}-{w1}"
            out.append(f"{label}: " + (". ".join(S) + "." if S else "Nothing notable."))
            continue
        # --- units disbanded / starved (review run 2026-09-15: rosters shrank with no narrated cause)
        gone = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))
        for e in evs:
            if e["type"] == "unit_lost" and e.get("player") and e.get("reason") in ("disbanded", "starved") and own_or_seen(e["player"]):
                gone[e["player"]][e["reason"]][e.get("unit") or e.get("type") or "unit"] += 1
        for civ, d in gone.items():
            # engine reason "starved" covers any upkeep shortfall (food OR shields), so say "upkeep"
            parts = [("disbanded " if why == "disbanded" else "lost to upkeep shortfalls ") + _units(c) for why, c in d.items()]
            S.append(f"{civ} " + " and ".join(parts))
        # --- units built
        prod = collections.defaultdict(collections.Counter)          # civ -> unit type counter (all cities pooled)
        for e in evs:
            if e["type"] == "unit_built" and e.get("player") and own_or_seen(e["player"]):
                prod[e["player"]][e.get("unit")] += 1
        for civ, c in prod.items():
            n = sum(c.values())
            S.append(f"{civ}'s cities built new units: " + _units(c) + (" unit" if n == 1 else " units"))
        # --- buildings and wonders (skip sell-and-rebuy: the city already had it at turn start)
        bld = collections.defaultdict(lambda: {"Improvement": [], "GreatWonder": [], "SmallWonder": [], "Special": []})
        for e in evs:
            if e["type"] == "building_built" and e.get("player") and own_or_seen(e["player"]):
                key = (e.get("city_id"), e.get("building"))
                if e.get("building") in blds.get(_t(e), {}).get(e.get("city_id"), set()) or key in reported_bld:
                    continue
                reported_bld.add(key)
                bld[e["player"]][genus.get(e.get("building"), "Improvement")].append(f"{e.get('building')} in {e.get('city')}")
        for civ, g in bld.items():
            parts = []
            if g["Improvement"] or g["Special"]:
                items = g["Improvement"] + g["Special"]
                parts.append(("the building " if len(items) == 1 else "the buildings ") + _join(items))
            if g["GreatWonder"]:
                parts.append(("the great wonder " if len(g["GreatWonder"]) == 1 else "the great wonders ") + _join(g["GreatWonder"]))
            if g["SmallWonder"]:
                parts.append(("the small wonder " if len(g["SmallWonder"]) == 1 else "the small wonders ") + _join(g["SmallWonder"]))
            if parts:
                S.append(f"{civ} completed " + " and ".join(parts))
        # --- tile work
        tw = collections.defaultdict(collections.Counter)
        for e in evs:
            if e["type"] == "action_finished_unit_tile" and e.get("action") in TILE_WORK and e.get("player") and e.get("success", True) and own_or_seen(e["player"]):
                tw[e["player"]][(e.get("actor"), e["action"])] += 1
        for civ, c in tw.items():
            by_type = collections.defaultdict(list)
            for (typ, act), n in c.most_common():
                sg, pl = TILE_WORK[act]
                by_type[typ].append(f"{n} {sg if n == 1 else pl}")
            S.append(f"{civ}'s " + "; ".join(f"{typ} units built {_join(v)}" for typ, v in by_type.items()))
        # --- redeployments: per civ, merged by destination
        first, last = {}, {}
        for t in range(w0, w1 + 2):
            if t in idx:
                for uid, (own, typ, x, y, hp) in at(t)[0].items():
                    if mil.get(typ, False):
                        first.setdefault(uid, (own, typ, (x, y))); last[uid] = (own, typ, (x, y))
        dest = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))   # civ -> dest -> type counter
        for uid, (own, typ, p0) in first.items():
            own1, typ1, p1 = last[uid]
            if p0 != p1 and own_or_seen(own1):
                dest[own1][p1][typ1] += 1
        for civ, dd in dest.items():
            ranked = sorted(dd.items(), key=lambda kv: -sum(kv[1].values()))
            clusters: list[list] = []                       # [center, type counter, n_dests]; greedy, radius MOVE_RADIUS
            for p, c in ranked:
                for cl in clusters:
                    if _near(cl[0], p, xsize):
                        cl[1].update(c); cl[2] += 1; break
                else:
                    clusters.append([p, collections.Counter(c), 1])
            parts = [f"{_units(c)} to " + (f"({p[0]},{p[1]})" if k == 1 else f"around ({p[0]},{p[1]})") for p, c, k in clusters[:MOVE_CAP]]
            rest = sum(sum(c.values()) for _, c, _ in clusters[MOVE_CAP:])
            S.append(f"{civ} moved military units: " + "; ".join(parts) + (f"; and {rest} more units elsewhere" if rest else ""))
        # --- combat
        combat: dict = {}
        hits = collections.defaultdict(set); def_hp_seen = set(); def_hp_by = {}
        for e in evs:
            if e["type"] != "action_finished_unit_units" or e.get("action") not in ATTACK_ACTIONS + BOMBARD + CAPTURE or not e.get("player"):
                continue
            t = _t(e); att = e["player"]; tid = e.get("target_id")
            if tid is None or not xsize:
                continue
            key = (tid % xsize, tid // xsize)
            defenders = [d for d in (at(t)[1].get(key) or at(t - 1)[1].get(key) or []) if d[0] != att]
            city = at(t)[2].get(key) or at(t - 1)[2].get(key)
            if not defenders and not (city and city[0] != att):
                continue
            dciv = defenders[0][0] if defenders else city[0]
            if view == 2 and viewer not in (att, dciv):
                continue
            g = combat.setdefault((att, e.get("actor"), e["action"], dciv, key),
                                  {"att_ids": set(), "dead": collections.Counter(), "att_dead": 0, "att_hp": 0, "def_hp": 0,
                                   "def_maxhp": 0, "stack": None, "dead_ids": set(), "city": city[1] if city and city[0] == dciv else None})
            g["att_ids"].add(e.get("actor_id"))
            stack = collections.Counter(d[1] for d in defenders)
            if g["stack"] is None:
                g["stack"] = stack
                g["def_maxhp"] = sum(maxhp.get(d[1], 10) for d in defenders)
            a_hp0 = (at(t)[0].get(e.get("actor_id")) or (None,) * 5)[4]
            a_hp1 = (at(t + 1)[0].get(e.get("actor_id")) or (None,) * 5)[4]
            if e.get("actor_id") in lost_at.get((t, att), set()):
                g["att_dead"] += 1
            elif a_hp0 is not None and a_hp1 is not None and a_hp1 < a_hp0:
                g["att_hp"] += a_hp0 - a_hp1
            dead = {d[2]: d[1] for d in defenders if d[2] in lost_at.get((t, dciv), set())}
            g["dead"].update(v for k, v in dead.items() if k not in g["dead_ids"]); g["dead_ids"].update(dead)
            hits[(t, key)].add((att, e.get("actor"), e["action"], dciv, key))
            if (t, key) not in def_hp_seen:                       # defenders' hp loss is per turn+tile, counted once
                def_hp_seen.add((t, key))
                loss = 0
                for d in defenders:
                    if d[2] not in dead and d[3] is not None:
                        d1 = (at(t + 1)[0].get(d[2]) or (None,) * 5)[4]
                        if d1 is not None and d1 < d[3]:
                            loss += d[3] - d1
                def_hp_by[(t, key)] = (dciv, loss, sum(maxhp.get(d[1], 10) for d in defenders))
        shared = collections.defaultdict(lambda: [0, 0])          # (dciv, tile) -> [hp lost, max hp] under several attacker groups
        for (t, key), (dciv, loss, mx) in def_hp_by.items():
            if not loss:
                continue
            gs = hits[(t, key)]
            if len(gs) == 1:
                combat[next(iter(gs))]["def_hp"] += loss
            else:
                shared[(dciv, key)][0] += loss; shared[(dciv, key)][1] = max(shared[(dciv, key)][1], mx)
        covered_dead = set()
        for (att, actor, action, dciv, key), g in combat.items():
            if not allied(att, dciv):
                covered_dead |= set(g["dead_ids"])
        for (att, actor, action, dciv, key), g in combat.items():
            if allied(att, dciv):
                # an ally cannot attack an ally: the tile-based attacker inference picked the wrong unit
                # (review run 2026-09-15: "Marie's Frigate attacked Khosrau's Workers" while allied)
                continue
            n_att = len(g["att_ids"])
            verb = "bombarded" if action in BOMBARD else "captured" if action in CAPTURE else "attacked"
            subj = f"{att}'s {actor} unit" + ("s" if n_att > 1 else "")
            where = f"at ({key[0]},{key[1]})"
            if g["city"]:
                obj = f"{dciv}'s city of {g['city']} {where}" + (f" (defended by {_units(g['stack'])})" if g["stack"] else " (undefended)")
            elif sum(g["stack"].values()) == 1:
                obj = f"{dciv}'s {next(iter(g['stack']))} unit {where}"
            else:
                obj = f"{dciv}'s units {where} ({_units(g['stack'])})"
            s = f"{subj} {verb} {obj}"
            tail = []
            if g["dead"]:
                n = sum(g["dead"].values())
                tail.append(f"{dciv} lost {_units(g['dead'])}" + (" unit" if n == 1 else " units"))
            if g["att_dead"]:
                tail.append(f"{att} lost {g['att_dead']} {actor} unit" + ("s" if g["att_dead"] > 1 else ""))
            hp = []
            if g["att_hp"] and n_att:
                pct = round(100 * g["att_hp"] / (n_att * maxhp.get(actor, 10)))
                hp.append(f"the attacker{'s' if n_att > 1 else ''} lost about {min(pct, 100)}% HP")
            if g["def_hp"] and g["def_maxhp"]:
                pct = round(100 * g["def_hp"] / g["def_maxhp"])
                hp.append(f"the defender{'s' if sum(g['stack'].values()) > 1 else ''} lost about {min(pct, 100)}% HP")
            if hp:
                tail.append(" and ".join(hp))
            if tail:
                s += "; " + "; ".join(tail)
            S.append(s)
        for (dciv, key), (loss, mx) in shared.items():
            if mx:
                S.append(f"under these attacks {dciv}'s units at ({key[0]},{key[1]}) lost about {min(round(100 * loss / mx), 100)}% HP in total")
        # --- losses not explained by any attack sentence (barbarian raids and other unlogged actors --
        # review run 2026-09-15: rosters shrank by 8-12 units with nothing narrated)
        unexplained = collections.defaultdict(collections.Counter)
        for e in evs:
            if e["type"] == "unit_lost" and e.get("reason") in ("killed", "captured") and e.get("player") \
                    and e.get("unit_id") not in covered_dead and own_or_seen(e["player"]):
                unexplained[e["player"]][e.get("unit") or e.get("type") or "unit"] += 1
        for civ, c in unexplained.items():
            S.append(f"{civ} also lost {_units(c)} in fighting not described above")
        label = f"turn {w0}" if w0 == w1 else f"turn {w0}-{w1}"
        S = [x[:1].upper() + x[1:] for x in S]
        out.append(f"{label}: " + (". ".join(S) + "." if S else "Nothing notable."))
    return "\n".join(out)
