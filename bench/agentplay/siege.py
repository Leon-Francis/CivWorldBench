"""Siege arithmetic for the agent-play panel and the forecaster's prompt (doc 17 §4.4, 2026-09-19).

dlg16 showed the forecaster rewarding "buy walls, add a garrison" with a lower city-loss probability while the
measured loss rate on exactly those revised routes was the highest of the run (0.45 vs 0.15 predicted, §6.7.2).
Neither side ever computed whether the reinforcement could change the outcome.  This module does that from the
ruleset and the savegame, so both the agent and the forecaster read the same numbers instead of guessing:

  * defence of a city = the best defender's `defense` x veteran x fortified x terrain x City Walls (civ2civ3:
    +100% vs land units, and Great Wall / Coastal Defense are ignored -- stated in the block);
  * threat = every KNOWN hostile military unit within `reach` turns of the city, by its `attack` and move rate;
  * outcome = a sequential-combat estimate: each attacker fights the best defender, per-round hit chance
    A/(A+D), hit points and firepower from the ruleset, and the city falls when every defender is dead;
  * the same estimate is repeated for two counterfactual garrisons -- "+ City Walls" and "+ one more of your best
    defender" -- which is exactly the decision the agent keeps making.

Everything is an estimate of the engine's own combat model, not the engine itself: no terrain-specific attack
bonuses, no veteran promotion during the fight, no bombardment, no stack kill.  The block says so.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

WALLS_BONUS = 1.0          # civ2civ3: City Walls +100% vs land units
FORTIFIED_BONUS = 0.5      # effect_fortified
GREAT_WALL = "Great Wall"
WALLS = "City Walls"
REACH_TURNS = 4            # how far ahead a threat is listed


@dataclass
class Defender:
    uid: int
    type: str
    defense: float
    hp: int
    fp: int
    vet: float
    fortified: bool


def _vet_power(sg, kind: str, level: int) -> float:
    """Veteran power factor; the civ2civ3 table is 100/150/175/200 for land units."""
    return (1.0, 1.5, 1.75, 2.0)[min(max(level, 0), 3)]


def has_building(sg, city: dict, name: str) -> bool:
    """`improvements` is a bit string over savefile.improvement_vector, not a list of names."""
    iv = list(sg.get("savefile", "improvement_vector") or [])
    return any(n == name and b == "1" for n, b in zip(iv, str(city.get("improvements") or "")))


def defender_strength(d: "Defender", has_walls: bool, terrain_bonus: float = 0.0) -> float:
    s = d.defense * d.vet * (1 + FORTIFIED_BONUS if d.fortified else 1.0) * (1 + terrain_bonus)
    return s * (1 + WALLS_BONUS) if has_walls else s


def city_defence(sg, rules, ego, city: dict, terrain_bonus: float = 0.0) -> tuple[list[Defender], float]:
    """Defenders inside the city tile and the effective defence strength of the best one."""
    pos = (int(city["x"]), int(city["y"]))
    has_walls = has_building(sg, city, WALLS)
    ds: list[Defender] = []
    for uid, u in ego.units.items():
        if (int(u["x"]), int(u["y"])) != pos or int(u.get("transported_by", -1)) != -1:
            continue
        ut = rules.units.get(str(u["type_by_name"]))
        if ut is None or ut.defense <= 0:
            continue
        ds.append(Defender(uid, ut.name, float(ut.defense), int(u.get("hp", ut.hitpoints) or ut.hitpoints), int(ut.firepower),
                           _vet_power(sg, "land", int(u.get("veteran", 0) or 0)), int(u.get("activity", 0)) in (5, 15)))
    best = max((defender_strength(d, has_walls, terrain_bonus) for d in ds), default=0.0)
    return ds, best


def _fight(rng, a_str: float, a_hp: int, a_fp: int, d_str: float, d_hp: int, d_fp: int) -> tuple[bool, int, int]:
    """One duel by the engine's round model.  Returns (attacker won, attacker hp left, defender hp left)."""
    if d_str <= 0:
        return True, a_hp, 0
    p = a_str / (a_str + d_str)
    while a_hp > 0 and d_hp > 0:
        if rng.random() < p:
            d_hp -= a_fp
        else:
            a_hp -= d_fp
    return d_hp <= 0, max(0, a_hp), max(0, d_hp)


def assault(attackers: list[dict], defenders: list[Defender], has_walls: bool, trials: int = 400, seed: int = 7) -> dict[str, Any]:
    """Sequential assault, Monte Carlo: every attacker in turn fights the city's currently strongest defender
    (damage carries over, as in the engine); the city falls when no defender is left."""
    import random
    if not defenders:
        return {"p_city_falls": 1.0}
    rng = random.Random(seed)
    falls = 0
    for _ in range(trials):
        pool = [[defender_strength(d, has_walls), d.hp, d.fp] for d in defenders]
        for a in attackers:
            if not pool:
                break
            pool.sort(key=lambda x: -x[0])
            won, _, dhp = _fight(rng, a["strength"], a["hp"], a["fp"], pool[0][0], pool[0][1], pool[0][2])
            if won:
                pool.pop(0)
            else:
                pool[0][1] = dhp
        if not pool:
            falls += 1
    return {"p_city_falls": round(falls / trials, 2)}


def threat_block(sg, rules, board, geo, ego, rel: dict[str, str], reach: int = REACH_TURNS) -> list[str]:
    """One block of text; empty list when no ego city is within reach of a known hostile stack."""
    threats: dict[str, list[dict]] = {}
    for sec in sg.players():
        if sec == ego.sec:
            continue
        owner = str(sg.get(sec, "name"))
        if rel.get(owner) not in ("War", "Cease-fire", "No Contact", "Armistice", None):
            continue
        hostile = rel.get(owner) == "War"
        t = sg.table(sec, "u")
        for u in (t.as_dicts() if t else []):
            pos = (int(u["x"]), int(u["y"]))
            if not board.is_known(*pos):
                continue
            ut = rules.units.get(str(u["type_by_name"]))
            if ut is None or ut.attack <= 0 or not ut.military:
                continue
            for n, c in ego.cities.items():
                d = geo.distance(pos, (int(c["x"]), int(c["y"])))
                eta = max(1, -(-d // max(1, ut.move_rate)))
                if eta <= reach and (hostile or d <= 2):
                    threats.setdefault(n, []).append(
                        {"owner": owner, "type": str(u["type_by_name"]), "strength": float(ut.attack) * _vet_power(sg, "land", int(u.get("veteran", 0) or 0)),
                         "hp": int(u.get("hp", ut.hitpoints) or ut.hitpoints), "fp": int(ut.firepower), "eta": eta,
                         "at_war": hostile})
    if not threats:
        return []
    cities_safe = 0
    L = ["SIEGE ARITHMETIC (computed by the driver from the ruleset, not guessed.  Each number is CONDITIONAL: it is the",
         " chance the city falls IF that stack assaults it -- per-round hit chance attack/(attack+defence), ruleset hit points",
         " and firepower, City Walls +100% vs land, fortified +50%, veteran 1.0/1.5/1.75/2.0; no terrain attack bonus, no",
         " promotion mid-fight.  ★ Measured base rate over 32 played games of this kind: a city listed here was actually lost",
         " within the next 20 turns in only 6% of cases (2% where the conditional number was under 0.3, 8% where it was above",
         " 0.7) -- most stacks never assault.  So read a high number as 'this city cannot survive an assault', not as",
         " 'this city will fall', and weigh the cost of reinforcing against that 6%.)"]
    shown = 0
    for n, ats in sorted(threats.items(), key=lambda kv: -len(kv[1])):
        c = ego.cities[n]
        ds, best = city_defence(sg, rules, ego, c)
        has_walls = has_building(sg, c, WALLS)
        ats = sorted(ats, key=lambda a: (a["eta"], -a["strength"]))[:8]
        now = assault(ats, ds, has_walls)
        if now["p_city_falls"] < 0.05 and ds:
            continue                                   # safe cities are not listed (keeps the block short)
        shown += 1
        if shown > 6:
            L.append("- (more cities under threat not shown)")
            break
        walls = assault(ats, ds, True) if not has_walls else None
        # counterfactual 2: one more of the best defender we can build, fortified
        try:
            buildable = [u for u in rules.unit_menu(ego.known_techs, set(), False) if u.defense > 0 and u.military and u.uclass == "Land"]
        except Exception:                          # noqa: BLE001
            buildable = []
        extra = max(buildable, key=lambda u: u.defense) if buildable else None
        ds2 = list(ds)
        if extra is not None:
            ds2.append(Defender(-1, extra.name, float(extra.defense),
                                int(extra.hitpoints), int(extra.firepower), 1.0, True))
        more = assault(ats, ds2, has_walls) if extra else None
        both = assault(ats, ds2, True) if (extra and not has_walls) else None
        L.append(f"- {n} (size {c.get('size')}, {'walls' if has_walls else 'NO walls'}): garrison "
                 + (", ".join(f"{d.type}{' fortified' if d.fortified else ''} def {d.defense:g} hp {d.hp}" for d in ds) or "EMPTY")
                 + f"; best effective defence {best:.1f}")
        L.append(f"    threat: " + "; ".join(f"{a['type']} ({a['owner']}, {'war' if a['at_war'] else 'not at war'}) att {a['strength']:.1f} in ~{a['eta']}t" for a in ats[:5]))
        L.append(f"    P(falls IF assaulted) now {now['p_city_falls']:.2f}"
                 + (f" | with City Walls {walls['p_city_falls']:.2f}" if walls else " | walls already built")
                 + (f" | with one more {extra.name} fortified {more['p_city_falls']:.2f}" if more else "")
                 + (f" | with both {both['p_city_falls']:.2f}" if both else ""))
    return L if shown else []


# ---------------------------------------------------------------------------------------------------------------
# v10 (doc 17 §4.10, 2026-09-20): the CONDITIONAL risk of a route, computed instead of asked.
#
# Measured over 408 planning points of the v9 run: the forecaster's P(ego loses a city within 20 turns) has a
# median spread of 0.04 across the three routes it is comparing, is identical for all three at 32% of planning
# points (44% after turn 150), and its correlation with the route's own declared defender count is -0.012.  Its
# MARGINAL forecast is fine (AUC 0.678, reliability 0.15 / 0.25 / 0.43 across forecast bands) -- it knows which
# SITUATIONS are dangerous.  It cannot tell which PLAN is dangerous in a given situation, which is exactly the
# ability the benchmark measures as weakest (L3), and it is the ability this protocol depends on.  So the driver
# computes that one number from the ruleset and hands it over as a fact, the same way it already hands over the
# siege arithmetic of the CURRENT garrison.
# ---------------------------------------------------------------------------------------------------------------

def _threats(sg, rules, board, geo, ego, rel: dict[str, str], reach: int = REACH_TURNS) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for sec in sg.players():
        if sec == ego.sec:
            continue
        owner = str(sg.get(sec, "name"))
        if rel.get(owner) not in ("War", "Cease-fire", "No Contact", "Armistice", None):
            continue
        hostile = rel.get(owner) == "War"
        t = sg.table(sec, "u")
        for u in (t.as_dicts() if t else []):
            pos = (int(u["x"]), int(u["y"]))
            if not board.is_known(*pos):
                continue
            ut = rules.units.get(str(u["type_by_name"]))
            if ut is None or ut.attack <= 0 or not ut.military:
                continue
            for n, c in ego.cities.items():
                d = geo.distance(pos, (int(c["x"]), int(c["y"])))
                eta = max(1, -(-d // max(1, ut.move_rate)))
                if eta <= reach and (hostile or d <= 2):
                    out.setdefault(n, []).append(
                        {"owner": owner, "type": str(u["type_by_name"]),
                         "strength": float(ut.attack) * _vet_power(sg, "land", int(u.get("veteran", 0) or 0)),
                         "hp": int(u.get("hp", ut.hitpoints) or ut.hitpoints), "fp": int(ut.firepower), "eta": eta})
    return out


def _best_defender(rules, ego):
    try:
        cand = [u for u in rules.unit_menu(ego.known_techs, set(), False) if u.defense > 0 and u.military and u.uclass == "Land"]
    except Exception:                                  # noqa: BLE001
        cand = []
    return max(cand, key=lambda u: u.defense) if cand else None


def route_risk(sg, rules, board, geo, ego, rel: dict[str, str], routes: list[dict], letters: list[str],
               horizon: int = 20) -> list[str]:
    """One line per route: the expected number of cities lost IF every threatening stack assaults, under the
    garrison that route promises.  `routes[i]["contract_spec"]["build"]["defenders"]` is the promise; extra
    defenders are placed in the most endangered cities first, City Walls are added where the route's text asks
    for them."""
    threats = _threats(sg, rules, board, geo, ego, rel)
    if not threats:
        return ["ROUTE RISK (computed by the driver from the ruleset, NOT a forecast): no foreign stack that this ruler can see is",
                f" within {REACH_TURNS} turns of any of its cities, so on EVERY one of these routes the expected number of cities",
                " lost to an assault in that window is 0.00.  A defender built now prevents nothing in this window; if a route",
                " spends on garrisons anyway, that spending has to be justified by what it expects LATER, not by this block."]
    extra_type = _best_defender(rules, ego)
    L = [f"ROUTE RISK (computed by the driver from the ruleset, NOT a forecast).  For each route: the garrison it promises,",
         f" and the expected number of your cities that fall IF the stacks now within {REACH_TURNS} turns of them assault.",
         " Same combat model as the SIEGE ARITHMETIC block.  Extra defenders are placed in your most endangered cities first."]
    base = None
    for L_, rt in zip(letters, routes):
        spec = rt.get("contract_spec") or {}
        n_def = int((spec.get("build") or {}).get("defenders", 0) or 0)
        wants_walls = "wall" in (rt.get("plan") or "").lower()
        # rank the threatened cities by how badly they do as they stand
        ranked = []
        for n, ats in threats.items():
            c = ego.cities[n]
            ds, _ = city_defence(sg, rules, ego, c)
            hw = has_building(sg, c, WALLS)
            ats8 = sorted(ats, key=lambda a: (a["eta"], -a["strength"]))[:8]
            ranked.append((assault(ats8, ds, hw)["p_city_falls"], n, ats8, ds, hw))
        ranked.sort(reverse=True)
        left = n_def
        exp = 0.0; worst = []
        for p0, n, ats8, ds, hw in ranked:
            ds2 = list(ds)
            add = min(left, 2)                      # at most two of the promised defenders go to one city
            left -= add
            for _ in range(add):
                if extra_type is not None:
                    ds2.append(Defender(-1, extra_type.name, float(extra_type.defense), int(extra_type.hitpoints),
                                        int(extra_type.firepower), 1.0, True))
            p = assault(ats8, ds2, hw or wants_walls)["p_city_falls"]
            exp += p
            worst.append((n, p0, p))
        worst.sort(key=lambda x: -x[2])
        if base is None:
            base = exp
        L.append(f"- {L_}: promises {n_def} defender(s)" + (" + City Walls" if wants_walls else "")
                 + f" -> expected cities lost {exp:.2f}" + (f" (vs {base:.2f} on route {letters[0]})" if L_ != letters[0] else "")
                 + "; worst: " + ", ".join(f"{n} {p0:.2f}->{p:.2f}" for n, p0, p in worst[:3]))
    return L


def plan_risk(sg, rules, board, geo, ego, rel: dict[str, str], plans: list[dict], labels: list[str]) -> list[str]:
    """v12: the same arithmetic, but the garrison comes from the plan's own per-city numbers, so the answer is
    about THIS plan rather than about a generic 'N more defenders'."""
    threats = _threats(sg, rules, board, geo, ego, rel)
    if not threats:
        return ["PLAN RISK (driver arithmetic, not a forecast): no foreign stack you can see is within "
                f"{REACH_TURNS} turns of any of your cities, so every one of these plans has an expected loss of "
                "0.00 cities to an assault in that window.  A defender built now prevents nothing in this window."]
    extra = _best_defender(rules, ego)
    L = ["PLAN RISK (computed by the driver from the ruleset, NOT a forecast).  For each plan: the expected number of",
         f" your cities that fall IF the stacks now within {REACH_TURNS} turns of them assault, given the garrison that",
         " plan asks for in each city (a city the plan does not mention keeps the garrison it has now)."]
    for lab, p in zip(labels, plans):
        want = {str(k): int(v) for k, v in (p.get("garrison") or {}).items()}
        walls = "wall" in (str(p.get("production")) + str(p.get("military", ""))).lower()
        exp = 0.0; rows = []
        for n, ats in threats.items():
            c = ego.cities[n]
            ds, _ = city_defence(sg, rules, ego, c)
            hw = has_building(sg, c, WALLS)
            ats8 = sorted(ats, key=lambda a: (a["eta"], -a["strength"]))[:8]
            p0 = assault(ats8, ds, hw)["p_city_falls"]
            ds2 = list(ds)
            for _ in range(max(0, want.get(n, len(ds)) - len(ds))):
                if extra is not None:
                    ds2.append(Defender(-1, extra.name, float(extra.defense), int(extra.hitpoints),
                                        int(extra.firepower), 1.0, True))
            p1 = assault(ats8, ds2, hw or walls)["p_city_falls"]
            exp += p1
            rows.append((n, p0, p1, want.get(n, len(ds)), len(ds)))
        rows.sort(key=lambda r: -max(r[1], r[2]))
        L.append(f"- {lab}: expected cities lost {exp:.2f}; " +
                 "; ".join(f"{n}: now {len(ds) if False else a} defender(s) -> plan asks {w}, P(falls) {p0:.2f}->{p1:.2f}"
                           for n, p0, p1, w, a in rows[:4]))
    return L
