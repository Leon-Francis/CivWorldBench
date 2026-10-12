"""v3 strategy conditioning (C1, doc 13 §5): each AI's PLAN at t1, read from its hidden belief
fields and rendered as text, inserted after the observation frames of the prediction tasks.

C1 reads ONLY turn-t1 fields -- research goal, war countdown (1..12 = a real plan, gate 13 §3.1),
founder_want / urgency bins, intended destinations of moving military units, non-neutral
attitudes and explicit diplomatic wishes.  No future record is opened, so nothing about the
outcome leaks; the fields are forward-looking by construction.

Scope (user 2026-09-15): omniscient view -> every alive civ's plan; POMDP view -> the viewer's
own plan only.  Mutual exclusion (13 §5.3): task 2 (infer hidden state) NEVER gets this block,
because attitudes and plans are among its targets.  Traits are never rendered (13 §5.4).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT, ROOT / "bundle", ROOT / "tools"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from bench.measure.directive_synth import (CD_MAX, CD_MIN, FW_BINS, URG_BINS, clusters,  # noqa: E402
                                           dir_label, love_word)
from bench.measure.action_profile import profile  # noqa: E402
from bench.spec_v2 import mil_map_for  # noqa: E402
from sandboxes.freeciv.state import base_ruleset  # noqa: E402

# ai/default/aidata.h (freeciv S3_1) `enum war_reason` -- verified against the source 2026-09-15;
# only rendered next to an ACTIVE countdown (1..12), where it explains the plan
WAR_REASON = {0: "it dislikes how that civilization behaves", 1: "to stop that civilization's space programme",
              2: "it has a pretext", 3: "out of hatred", 4: "to honour an alliance", 5: None}
HEADER = "=====  STRATEGIES   as of turn {t1}   {scope}  ====="
INTRO = ("Each AI civilization follows an internal plan. The lines below state, for {who}, what that "
         "plan is at turn {t1}: research aim, war preparations, where its military, settler and caravan "
         "parties are heading, appetite for new cities, defensive pressure, pending government changes, "
         "and the attitudes it has formed. Plans can change later.")


def civ_plan(rec: dict, world: dict, ego: str) -> tuple[list[str], dict]:
    """C1 plan lines for one civ from the t1 record only (mirror of directive_synth.synth, sources=C1)."""
    pb = rec["probes"]
    att = (pb.get("B_attitude") or {}).get(ego) or {}
    intent = (pb.get("D_intent") or {}).get(ego) or {}
    acts = (rec.get("actions") or {}).get(ego) or {}
    xsize = int((rec.get("geo") or {}).get("xsize") or 0)
    t1 = rec["t"]
    # current relations (slot -> name); a "war plan" against an ally or team-mate is an engine
    # artefact (review run 2026-09-15: /team worlds showed Alpha planning to attack Bravo, its own
    # team-mate, "to honour an alliance") and is not rendered
    me = next((q for q in rec["state"]["players"] if q["name"] == ego), {})
    slot_name = {f"player{i}": q["name"] for i, q in enumerate(rec["state"]["players"])}
    current = {slot_name[k]: v for k, v in (me.get("diplomacy") or {}).items() if k in slot_name}
    targets = sorted(((o, d.get("countdown")) for o, d in att.items()
                      if o != ego and isinstance(d.get("countdown"), int) and CD_MIN <= d["countdown"] <= CD_MAX
                      and current.get(o) not in ("Alliance", "Team")),
                     key=lambda kv: kv[1])
    cities = intent.get("cities") or {}
    fw = max([int(c.get("ai.founder_want") or 0) for c in cities.values()] or [0])
    urg = max([int(c.get("ai.urgency") or 0) for c in cities.values()] or [0])
    goal = ((intent.get("research") or {}).get("goal_name") or "").strip()
    mil = mil_map_for(base_ruleset(rec.get("ruleset") or "civ2civ3"))
    # ⚠️ 70% of units flagged `moving` have destination == position (fortified units whose `go` flag is
    # set; 58,104 vs 25,225 measured 2026-09-15).  Only a destination away from the unit is a plan.
    moving = [o for o in (acts.get("unit_orders") or [])
              if o.get("moving") and o.get("destination") and list(o["destination"]) != list(o["at"])]
    dests = [tuple(o["destination"]) for o in moving if mil.get(o.get("type"))]
    cl = clusters(dests, xsize)
    # expansion plan: settler parties on their way to found cities (zero future data: t1 orders)
    settlers = [tuple(o["destination"]) for o in moving if o.get("type") in ("Settlers", "Migrants")]
    scl = clusters(settlers, xsize)
    # trade plan: caravans/freight heading for a city -- whose?
    city_at = {(c["x"], c["y"]): (c["name"], pl["name"]) for pl in rec["state"]["players"] for c in (pl.get("cities") or [])}
    car_own, car_foreign = 0, {}
    for o in moving:
        if o.get("type") in ("Caravan", "Freight"):
            hit = city_at.get(tuple(o["destination"]))
            if hit is None:
                continue
            if hit[1] == ego:
                car_own += 1
            else:
                car_foreign[hit[1]] = car_foreign.get(hit[1], 0) + 1
    # government: an anarchy already under way ends at a known turn (E_clock.revolution_finishes)
    clock = ((pb.get("E_clock") or {}).get(ego) or {})
    rev = clock.get("revolution_finishes", -1)
    rev_in = (int(rev) - t1) if isinstance(rev, int) and rev >= t1 else None
    snap = profile({t1: rec}, ego, t1, t1, world["ruleset"])
    t1_prod = {k: round(v, 2) for k, v in (snap or {"prod": {}})["prod"].items() if v >= 0.02}
    d = dir_label(t1_prod, bool(targets)) if snap else "hold its position"
    if targets:
        d = "prepare for war"
    elif fw >= FW_BINS[1]:
        d = "keep expanding"
    L = [f"{ego} is set on this course: {d}."]
    if goal and goal != "A_UNSET":
        L.append(f"  Research aim: {goal}.")
    if targets:
        o, c = targets[0]
        ax = ", ".join(f"around ({x},{y})" for (x, y), n in cl) if cl else "not yet on the move"
        why = WAR_REASON.get((att.get(o) or {}).get("war_reason"))
        L.append(f"  War aim: it intends to attack {o} and will move in about {c} turns"
                 + (f" ({why})" if why else "") + f"; its forces are {ax}.")
        for o2, c2 in targets[1:]:
            L.append(f"           it is also preparing against {o2} (in about {c2} turns).")
    elif cl:
        L.append("  Forces: military units are on the move towards " + ", ".join(f"({x},{y})" for (x, y), n in cl) + ".")
    if scl:
        L.append(f"  Expansion: {len(settlers)} settler part{'y is' if len(settlers) == 1 else 'ies are'} on the way to found "
                 f"{'a new city' if len(settlers) == 1 else 'new cities'} around " + ", ".join(f"({x},{y})" for (x, y), n in scl) + ".")
    if car_own or car_foreign:
        parts = []
        if car_foreign:
            parts.append("to " + " and ".join(f"{k}'s cities ({v})" for k, v in sorted(car_foreign.items(), key=lambda kv: -kv[1])))
        if car_own:
            parts.append(f"to its own cities ({car_own})")
        L.append("  Trade: caravans are under way " + " and ".join(parts) + ".")
    if rev_in is not None:
        L.append(f"  Government: it is in anarchy; its new government takes office in {rev_in} turn{'s' if rev_in != 1 else ''}.")
    if fw >= FW_BINS[1]:
        L.append("  Settlement: it is still strongly set on founding new cities.")
    elif fw >= FW_BINS[0]:
        L.append("  Settlement: it would still like one or two more cities.")
    if urg >= URG_BINS[1]:
        L.append("  Defence: one of its cities is under heavy defensive pressure.")
    elif urg >= URG_BINS[0]:
        L.append("  Defence: one of its cities feels exposed.")
    nb = []
    for o, d2 in att.items():
        if o == ego or current.get(o) == "Team":       # team-mates: permanent relation, attitude is noise
            continue
        w = love_word(int(d2.get("love") or 0))
        wants = [n for n, k in (("peace", "ask_peace"), ("an alliance", "ask_alliance"),
                                ("a cease-fire", "ask_ceasefire")) if d2.get(k)]
        if w == "indifferent to" and not wants:
            continue
        nb.append(f"{w} {o}" + (f", and would accept {' and '.join(wants)}" if wants else ""))
    if nb:
        L.append("  Neighbours: it is " + "; ".join(nb) + ".")
    prov = {"countdown": dict(targets), "founder_want": fw, "urgency": urg, "research_goal": goal,
            "axis": [(list(c[0]), c[1]) for c in cl], "direction": d, "n_attitudes": len(nb),
            "settlers_en_route": len(settlers), "caravans_own": car_own, "caravans_foreign": car_foreign,
            "anarchy_ends_in": rev_in}
    return L, prov


def strategy_block(rec: dict, world: dict, egos: list[str], view: int) -> tuple[str, dict]:
    """The block inserted after the observation frames.  view 1: all `egos`; view 2: `egos` = [viewer]."""
    t1 = rec["t"]
    scope = "(every civilization's plan)" if view == 1 else f"(only {egos[0]}'s own plan is known)"
    who = "every civilization" if view == 1 else egos[0]
    out = [HEADER.format(t1=t1, scope=scope), INTRO.format(who=who, t1=t1), ""]
    prov = {}
    for ego in egos:
        L, pv = civ_plan(rec, world, ego)
        out += L
        prov[ego] = pv
    return "\n".join(out) + "\n", prov
