"""v2 benchmark specification -- frozen constants and truth derivations.

Everything the builder, the baselines and the scorer must agree on lives here, so
that "what is being measured" is one file with one digest.  Sources: 09 §1 (1a),
§2 (1b), §3 (1c), §4 + §8.1 (task 2), §5 + §8.5 (task 3), §6 (conditions), §8.7
(corpus design).  Anchor horizons follow the world's *turn budget*, not its tier
name, because the generalisation set (tier "G") mixes budgets.
"""
from __future__ import annotations

import math
from typing import Any

# ---------------------------------------------------------------- 1a --------
# 1a task variables.  Three are INCREMENTS over the horizon window (user 2026-09-11 01:00):
# score / culture / n_techs are smooth cumulative quantities whose linear extrapolation is
# near-optimal, so asking for the level made the task "copy the trend"; asking for the gain
# from t1 to t1+h removes the copy-the-current-value degenerate answer and makes the
# question about the growth rate.  `derive()` still returns the 11 LEVEL quantities (keys
# below in LEVEL_OF); `task_vars()` converts a (level_h, level_t1) pair into the task vector.
VARS = ("gold", "population", "score_gain", "culture_gain", "n_cities", "city_size_total",
        "techs_learned", "n_military_units", "n_civilian_units", "n_enemies", "n_allies")
DELTA_VARS = {"score_gain": "score", "culture_gain": "culture", "techs_learned": "n_techs"}
LEVEL_OF = {v: DELTA_VARS.get(v, v) for v in VARS}

FAMILIES = {
    "economy": ("gold",),
    "scale": ("population", "n_cities", "city_size_total"),
    "tech_culture": ("techs_learned", "culture_gain"),
    "military": ("n_military_units", "n_civilian_units"),
    "diplomacy": ("n_enemies", "n_allies"),
    "overall": ("score_gain",),
}


def task_vars(level_h: dict | None, level_t1: dict) -> dict:
    """The 11 task quantities at horizon h from the level dicts (derive output) at t1+h and t1.
    A dead civ (level_h None) gives None everywhere (scored as missing)."""
    if level_h is None:
        return {v: None for v in VARS}
    out = {}
    for v in VARS:
        lv = LEVEL_OF[v]
        out[v] = (level_h[lv] - level_t1[lv]) if v in DELTA_VARS else level_h[lv]
    return out

GLOSSARY = """Quantity definitions (read these; several are not what the name suggests):
  gold               treasury at that turn (not income)
  population         total population in thousands
  score_gain         CHANGE in the game score from turn {t1} to that turn (score then minus
                     score now; may be negative)
  culture_gain       culture points accumulated from turn {t1} to that turn (then minus now)
  n_cities           number of cities owned
  city_size_total    sum of the sizes of all cities owned
  techs_learned      number of technologies learned from turn {t1} to that turn (count then
                     minus count now); repeated "Future Technology" advances do NOT increase
                     the count, so a civilization that already knows every technology stays at 0
  n_military_units   number of military units alive (attack strength > 0)
  n_civilian_units   number of non-military units alive (settlers, workers, caravans,
                     diplomats, transports, ...)
  n_enemies          number of LIVING contacted players this civilization is AT WAR with --
                     barbarian and pirate players count while they are alive; destroyed
                     civilizations do not
  n_allies           number of LIVING contacted players this civilization is ALLIED with
                     (the permanent Team relation of team games is not an alliance and is
                     not counted)"""

# horizons by turn budget (report_endturn); S/XL-like 400, M-like 800, L-like 1200
# h=30 dropped 2026-09-10 night (user, after the 20% pilot): at 30 turns most variables have not
# moved, persistence is near-exact and the winnable space is too small.
# Second revision the same night: the shortest of each budget dropped as well -> two horizons per
# budget.  1c uses the same two (task 3's 1c takes the first).
HZ_BY_BUDGET = {400: (60, 120), 800: (120, 250), 1200: (200, 300)}
# 1c retired 2026-09-12 (user); the constant stays only so old runs' export code still imports.
H1C_BY_BUDGET = {400: (60, 120), 800: (120, 250), 1200: (200, 300)}
# every arm resolves its state at these horizons (branch.HORIZONS); task 3 may ask any of them
ARM_HORIZONS = (30, 60, 90, 120, 150, 200, 250, 300)


def hz_task3(budget: int, band) -> list[int]:
    """v3 (user 2026-09-14): task 3 asks ONLY horizons inside the operator's effect band.

    Take the budget's standard horizons that fall in the band; if none does, ask the single
    longest arm-resolved horizon inside the band.  Measured reason (14 §B2): `L1-gold-swing`
    (band 5-30) was never scored because the shortest standard horizon is 60, and for the
    short-band operators h=30 is where the real/placebo contrast is strongest (113% vs 1.8%
    of level at h=30, ratio 68x; 7x by h=120)."""
    lo, hi = int(band[0]), int(band[1])
    hz = [h for h in HZ_BY_BUDGET[int(budget)] if lo <= h <= hi]
    if hz:
        return hz
    inb = [h for h in ARM_HORIZONS if lo <= h <= hi]
    return [max(inb)] if inb else []
# 1a variables scored on log1p scale (heavy-tailed: AI treasuries reach 10^6 while typical civs hold
# 10^2-10^3; a global constant normalisation lets a handful of hoarders decide a family's skill)
LOG_VARS = ("gold", "population", "culture_gain")
K, DH, DH_EARLY = 3, 40, 20        # K frames DH apart; DH_EARLY when t1 - 80 < 1 (09 §7.3)


def frames_for(t1: int, k: int = K) -> list[int]:
    dh = DH if t1 - (k - 1) * DH >= 1 else DH_EARLY
    return [t1 - (k - 1 - i) * dh for i in range(k)]


def met_civs(rec: dict, viewer: str) -> list[str]:
    """POMDP question scope V(t1): the viewer plus every civ it has met (diplstate
    left `Never met`; contact may have lapsed, an embassy is NOT required).  Only the
    viewer's own block is fully observed; for the others the model must infer hidden
    quantities from what the stale report shows -- that is the point of the set."""
    pl = rec["state"]["players"]; sn = {f"player{i}": p["name"] for i, p in enumerate(pl)}
    me = next((p for p in pl if p["name"] == viewer), None)
    if me is None:
        return [viewer]
    alive = {p["name"] for p in pl if p.get("is_alive")}
    seen = [sn[k] for k, v in (me.get("diplomacy") or {}).items()
            if k in sn and sn[k] != viewer and v not in ("Never met", "No Contact") and sn[k] in alive]
    return [viewer] + sorted(seen)


def budget_of(world_json: dict) -> int:
    return int(world_json["report_endturn"])


def derive(player: dict, mil_map: dict[str, bool] | None = None, rec: dict | None = None) -> dict[str, int]:
    """The 11 variables of one civ from a corpus record's player dict.

    ⚠️ `player["military_units"]` is the savegame's `nunits` = ALL units (the field is
    misnamed; verified 2026-09-10), so military/civilian are counted by unit type
    with the ruleset's attack table (`tools.bench_report.unit_is_military`).
    `rec` (2026-09-15): the record the player comes from; when given, relations with players that
    are no longer alive are not counted in n_enemies / n_allies (a destroyed civ keeps its War row
    in the diplstate table forever; the review run found dead civs inflating the enemy count).
    Barbarian / pirate players DO count while alive -- the glossary says so."""
    dipl = player.get("diplomacy") or {}
    dead = set()
    if rec is not None:
        dead = {f"player{i}" for i, q in enumerate(rec["state"]["players"]) if not q.get("is_alive")}
    units = player.get("units") or []
    if mil_map is None:
        mil_map = _MIL_DEFAULT
    mil = sum(1 for u in units if mil_map.get(u.get("type") or u.get("name"), False))
    return {
        "gold": int(player["gold"]), "population": int(player["population"]),
        "score": int(player["score"]), "culture": int(player["culture"]),
        "n_cities": len(player.get("cities") or []),
        "city_size_total": sum(int(c["size"]) for c in (player.get("cities") or [])),
        "n_techs": len(player.get("techs") or []),
        "n_military_units": mil,
        "n_civilian_units": len(units) - mil,
        "n_enemies": sum(1 for k, v in dipl.items() if v == "War" and k not in dead),
        "n_allies": sum(1 for k, v in dipl.items() if v == "Alliance" and k not in dead),
    }


def mil_map_for(ruleset: str) -> dict[str, bool]:
    from bench_report import unit_is_military      # tools/
    return unit_is_military(ruleset)


_MIL_DEFAULT = None
try:
    _MIL_DEFAULT = mil_map_for("civ2civ3")
except Exception:      # tools not on path (e.g. importing the spec alone)
    _MIL_DEFAULT = {}


# arm `resolved` dicts use these names for the same quantities
RESOLVED_ALIAS = {"city_size_total": "total_city_size", "n_military_units": "military_units",
                  "n_civilian_units": "civilian_units"}


def derive_from_resolved(r: dict) -> dict[str, Any]:
    out = {}
    for v in VARS:
        out[v] = r.get(RESOLVED_ALIAS.get(v, v))
    return out


# ---------------------------------------------------------------- 1b --------
# (kind, family, min_h, text template).  Truth functions live in truth_1b().
TEMPLATES_1B = (
    ("bankrupt",   "economy",      60, "does {civ}'s treasury fall to 0 gold at any turn between turn {t1} and turn {tt}?"),
    ("newcity",    "scale",        60, "does {civ} found at least one new city between turn {t1} and turn {tt}?"),
    ("capture",    "scale_mil",     0, "does {civ} capture at least one city from another civilization between turn {t1} and turn {tt}?"),
    ("lostcity",   "military",      0, "does {civ} lose at least one city (destroyed or captured) between turn {t1} and turn {tt}?"),
    ("mildrop",    "military",      0, "does {civ}'s number of military units fall by 30% or more from its turn-{t1} level ({base}) at any turn between turn {t1} and turn {tt}?"),
    ("wonder",     "tech_culture",  0, "does {civ} complete at least one great wonder between turn {t1} and turn {tt}?"),
    ("moreenemies","diplomacy",     0, "is {civ} at war with MORE civilizations at turn {tt} than at turn {t1} ({base})?"),
    ("newally",    "diplomacy",     0, "does {civ} form at least one new alliance between turn {t1} and turn {tt}?"),
    ("govchange",  "overall",       0, "does {civ} change its form of government at any turn between turn {t1} and turn {tt}?"),
    ("stillahead", "relative",      0, "at turn {t1}, {civ} ({base}) has a higher score than {other} ({base2}); is {civ}'s score still higher than {other}'s at turn {tt}?"),
    ("leader",     "relative",      0, "which civilization has the highest score at turn {tt}? (answer with one probability per civilization; they must sum to 1)"),
)
REL_MAX_GAP = 0.25        # relative "still ahead" only for pairs within 25% at t1
LEADER_MAX_GAP = 0.10     # "who leads" only when the leader's margin is <= 10%
BASE_RATE_LO, BASE_RATE_HI = 0.10, 0.90
# v3: the 10-90% rule is applied per (set, cond, template, budget, h, t1) on the question set's
# OWN truth (in-sample climatology, the same reference the scorer uses), and a cell must have
# at least MIN_CELL_N items.  The frozen v2 set filtered on rates pooled over anchors and let
# 30/264 out-of-range cells through (govchange|1200|300|t60 at 99.2%); those 12.4% of the items
# carried 41.9% of the model's Brier deficit (14 §3.4).  `leader` is multi-class and exempt.
MIN_CELL_N = 200
# `joinwar` deleted 2026-09-14 (user): ideal-observer winnable space +0.2%, the event is triggered
# by another civ's decision.  `leader` kept but reported separately (0-2 Brier scale, persistence ref).


# ---------------------------------------------------------------- 1c --------
COMPONENTS_1C = ("techs_new", "unit_mix", "city_exist", "city_size", "city_prod",
                 "city_bldg_new", "stock", "dipl_transitional")
TRANSITIONAL = ("Cease-fire", "Armistice", "Peace")


# ------------------------------------------------------------- task 2 --------
TRAITS = ("Aggressive", "Expansionist", "Trader", "Builder")
TRAIT_BINS = ((10, 56), (57, 103), (104, 150))         # low / mid / high over the 10-150 range
# v3 (user 2026-09-14): the personality question is a WITHIN-WORLD RANKING per dimension
# (`Aggressive: A > B = C > D`), scored by pairwise ranking accuracy (ties 0.5, chance 0.5).
# Absolute bins are kept only as an appendix reading: an ideal observer with 33 privileged
# behaviour features reaches 0.337-0.367 3-class accuracy (chance 0.333) but 0.565-0.609
# pairwise (14 §3.2) -- the signal lives in the relative ordering, and the model's gap is
# only attributable there.  Requires >= 2 rankable civs; single-civ scopes skip the block.
TRAIT_GLOSS = {"Aggressive": "starts and prolongs wars, refuses ceasefires",
               "Expansionist": "founds cities", "Trader": "builds caravans and trade routes",
               "Builder": "prefers buildings over units"}
# v3 (user 2026-09-14): a world-rule question is asked only when the observation window holds
# evidence a reader could use; `disasters` is unconditional (disaster events are narrated).
# Measured reason: only disasters was identifiable (0.558 vs majority 0.418); illness/foodbox/
# sciencebox sat at -0.011..+0.015 because nothing in a 3-frame window reveals them (14 §B6).
# ⚠️ `_illness` is NOT asked at all in v3: plague is never narrated (interval_v2 has no plague or
# famine sentence) and never rendered (no city illness field), so a plague hit leaves only an
# unexplained size drop in the city table -- indistinguishable from famine.  No observable
# evidence exists in any window; the question is dropped and written up as a negative result.
PARAM_EVIDENCE = {"disasters": None,                    # always (disasters are narrated)
                  "foodbox": ("city_size_change", 1),   # >= 1 city grew or shrank (city tables)
                  "sciencebox": ("tech_discovered", 2)} # >= 2 technologies completed (narrated)
PARAMS_ASKED = ("disasters", "foodbox", "sciencebox")
LOVE_BINS = ((-1000, -250), (-249, 249), (250, 1000))  # hostile / neutral / friendly
# world parameters: three bins over the training range (log-spaced for the log-uniform ones)
PARAM_BINS = {
    "disasters": ((0, 12), (13, 40), (41, 10**9)),         # 0 and low / default-ish / high
    "foodbox": ((0, 82), (83, 117), (118, 10**9)),
    "sciencebox": ((0, 82), (83, 117), (118, 10**9)),
    "_illness": ((0, 0), (10, 10), (30, 30)),
}
PARAM_DISPLAY = {"disasters": "disaster frequency", "foodbox": "city growth speed",
                 "sciencebox": "technology speed", "_illness": "plague severity"}
PARAM_LABELS = {
    "disasters": ("rare or none", "normal", "frequent"),
    "foodbox": ("cities grow fast", "normal", "cities grow slowly"),
    "sciencebox": ("technology comes fast", "normal", "technology comes slowly"),
    "_illness": ("no plague", "normal plague", "severe plague"),
}


def bin_of(x: float, bins) -> int:
    for i, (lo, hi) in enumerate(bins):
        if lo <= x <= hi:
            return i
    return 0 if x < bins[0][0] else len(bins) - 1


# ------------------------------------------------------------- task 3 --------
OPS = ("L1-gold-swing", "L1-bulbs", "L2-city-kill", "L2-gov-indirect", "L3-fog",
       "L4-trait-aggr", "L4-trait-expa", "L4-trait-trade", "L4-trait-build", "L4-love", "WP-disasters")
# v3 changes (user 2026-09-14): `L3-wonder` deleted (every variable d <= 0.23, declared ones
# 0.09/0.05: granting three wonders has no measurable consequence); `L4-trait` (Aggressive AND
# Expansionist +45 in one edit, prompt named only one of them) replaced by four single-dimension
# operators collected 2026-09-14/15 (14 §B11).  Other operators' score_vars unchanged (user).
# ★ 2026-09-15 (user): the four trait operators are collected at dose +90 (arm ids `L4-trait-*90`);
# the benchmark keeps the dose-free ids above as the operator names.  Dose-response on the same 298
# anchors (14 §B11-3.7): Aggressive->n_enemies direction 71-85% at +45 vs 87-91% at +90,
# Builder->buildings 58-66% vs 63-75%; Expansionist/Trader unchanged (coin flips at both doses).
OP_ALIAS = {f"L4-trait-{k}90": f"L4-trait-{k}" for k in ("aggr", "expa", "trade", "build")}


def op_base_of(arm_op_id: str) -> str:
    return OP_ALIAS.get(arm_op_id.split("__placebo")[0], arm_op_id.split("__placebo")[0])


# 1a quantities each operator is expected to move (task 3 score A, "effect capture");
# the arm json does not carry score_vars, and `government` (L2-gov-indirect's engine-side
# readout) is not a 1a quantity -- its effect is read through 1b govchange (score C)
SCORE_VARS = {
    "L1-gold-swing": ("gold",),
    "L1-bulbs": ("techs_learned",),
    # n_cities dropped 2026-09-16 (user review of L3): the placebo razes the SAME NUMBER of (smallest) cities by design,
    # so n_cities moves identically in both arms (measured A real +0.44 vs placebo +0.40, direction 0.92 / 0.92) and only
    # rewards reading the intervention text; population and total city size are the discriminating variables
    # (real +0.06 / +0.64 vs placebo -0.16 / +0.42).
    "L2-city-kill": ("city_size_total", "population"),
    # B12: real ("acquire Communism") and placebo ("acquire Polytheism") both add exactly one
    # tech, so techs_learned moves identically in both arms and cancels in the real-placebo
    # contrast; keep only the downstream quantity.
    "L2-gov-indirect": ("score_gain",),
    "L3-fog": ("n_cities", "score_gain"),
    # single-dimension traits: the declared mechanism variables (interventions.py).  Measured on
    # the +45 arms (n=74-79, 14 §B11-3.5): only Aggressive->n_enemies (77-85% in the expected
    # direction) and Builder->buildings are directionally consistent; the rest diverge from the
    # factual arm (d 0.5-0.9) without a consistent sign.  Kept as declared (user 2026-09-15:
    # whether the lever can act -- e.g. no free land left -- is itself for the model to judge).
    "L4-trait-aggr": ("n_enemies", "n_military_units"),
    "L4-trait-expa": ("n_cities", "city_size_total"),
    "L4-trait-trade": ("gold", "techs_learned"),
    "L4-trait-build": ("city_size_total", "n_military_units"),
    "L4-love": ("n_enemies", "n_allies"),
    "WP-disasters": ("population", "city_size_total"),
}
# operators whose intervention is written into the t1 report (edit channel); the
# rest are told in one sentence with a magnitude the model can judge (text channel)
EDIT_OPS = ("L1-gold-swing", "L1-bulbs", "L2-gov-indirect", "L2-city-kill")

# ---------------------------------------------------------- conditions -------
CONDITIONS = {
    "v1": {"view": 1, "primer": 0, "k": 3, "oracle": 0, "think": 0},   # omniscient, main
    "v2": {"view": 2, "primer": 0, "k": 3, "oracle": 0, "think": 0},   # single-civ, main
    "oracle": {"view": 1, "primer": 0, "k": 3, "oracle": 1, "think": 0},
    "k1": {"view": 1, "primer": 0, "k": 1, "oracle": 0, "think": 0},
    "k5": {"view": 1, "primer": 0, "k": 5, "oracle": 0, "think": 0},   # only anchors with t1 >= 161 (160-turn window)
    "primer": {"view": 1, "primer": 1, "k": 3, "oracle": 0, "think": 0},
    "think": {"view": 1, "primer": 0, "k": 3, "oracle": 0, "think": 1},
}
MAIN_CONDITIONS = ("v1", "v2")
ABLATION_WORLDS = {"oracle": 60, "k1": 60, "primer": 40, "think": 40}   # per condition, spread over tiers
# "k5" (five frames) exists as a condition but is NOT scheduled: user 2026-09-10 kept the
# 3-frame main grid and dropped the K=5 ablation for now (L-tier K=5 prompts reach ~196K
# tokens; measured in 09 §8.7.2).

# Events listed in the premise's INTERVAL blocks.  The v2 engine stream is per unit
# (action_started/finished_*, unit_built, unit_lost, city_size_change ...): 86% of a
# 40-turn interval's characters (measured 2026-09-10 on a G world: 62K of 72K), and the
# interval narrative already aggregates production / movement / losses per civ.  So the
# Events list carries only the civilisation-level events.
MAJOR_EVENT_TYPES = ("war_declared", "ceasefire_agreed", "ceasefire_expired", "peace_treaty_signed",
                     "peace_established", "alliance_formed", "alliance_ended", "first_contact", "diplstate_change",
                     "city_founded", "city_destroyed", "city_transferred", "government_change",
                     "tech_discovered", "building_built", "disaster_occurred", "achievement_gained",
                     "wonder_built", "hut_enter")
# Oracle ablation: the full future event stream of an M/L anchor is 260-320K tokens
# (verify 2026-09-10), so the oracle gives only the *major* event types -- diplomacy,
# cities, government, wonders, techs -- not unit production/loss and unit actions.
ORACLE_EVENT_TYPES = ("war_declared", "ceasefire_agreed", "ceasefire_expired", "peace_treaty_signed",
                      "peace_established", "alliance_formed", "alliance_ended", "first_contact",
                      "city_founded", "city_destroyed", "city_transferred", "government_change",
                      "tech_discovered", "building_built", "disaster_occurred", "achievement_gained")

SYSTEM = ("You are analysing a Freeciv game. You are given observations of the world at "
          "several past turns, and a narrative of what happened between them. Answer only "
          "in the exact output format requested. Do not explain your reasoning outside the "
          "requested format, and do not omit any requested line.")
