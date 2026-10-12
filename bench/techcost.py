"""Exact research cost of the tech a civ is researching, recomputed from the record
(freeciv 3.1.6 `research_total_bulbs_required`, read 2026-09-10):

  base   = base_tech_cost * num_reqs                       ("Linear", civ2civ3: 10)
         = base_tech_cost * (1+n) * sqrt(1+n) / 2           ("Classic", classic: 20)
           where num_reqs = the tech itself + all recursive prerequisites
  x Tech_Cost_Factor effect (civ2civ3 3, classic 1)
  - leak (tech_leakage "Embassies", civ2civ3 only):
           base * (#alive civs that know the tech AND with whom we have an embassy)
                * tech_leak_pct / (#alive civs) / 100
  x AI science_cost / 100 (skill-level handicap)
  x sciencebox / 100 ;  at least 1

Verified against the corpus (bulbs before + income - bulbs after at a discovery), see
bench/measure/out/techcost_check.txt.  Used for the report line
`research: <tech> [<accumulated bulbs>/<cost>]` (user 2026-09-10).
"""
from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path

from sandboxes.freeciv.ruleset import parse

ROOT = Path(__file__).resolve().parent.parent
RULESETS = ROOT / "bench/v2/rulesets"
STYLE = {"civ2civ3": ("linear", 10, 3, "embassies"), "classic": ("classic", 20, 1, "none")}
TECH_LEAK_PCT = 100                       # ruleset default (neither ruleset sets tech_leak_pct)
SCIENCE_COST = {"away": 100, "novice": 250, "easy": 100, "normal": 100, "hard": 100, "cheating": 100, "experimental": 100}  # ai/difficulty.c


@lru_cache(maxsize=None)
def tech_tree(ruleset: str) -> dict[str, tuple[str, ...]]:
    t = parse(RULESETS / ruleset / "techs.ruleset")
    tree = {}
    for k, v in t.items():
        if k.startswith("advance_") and v.get("name"):
            reqs = tuple(r for r in (str(v.get("req1", "None")), str(v.get("req2", "None"))) if r not in ("None", "", "Never"))
            tree[str(v["name"])] = reqs
    return tree


@lru_cache(maxsize=None)
def num_reqs(ruleset: str, tech: str) -> int:
    tree = tech_tree(ruleset)
    seen = set(); stack = [tech]
    while stack:
        x = stack.pop()
        if x in seen or x not in tree:
            continue
        seen.add(x); stack.extend(tree[x])
    return len(seen)                      # includes the tech itself


def base_cost(ruleset: str, tech: str) -> float:
    style, base, factor, _ = STYLE[ruleset]
    n = num_reqs(ruleset, tech)
    if n == 0:
        return 0.0
    if style == "linear":
        c = base * n
    else:
        c = base * (1.0 + n) * math.sqrt(1.0 + n) / 2
    return max(c, base) * factor


def research_cost(rec: dict, civ: str, tech: str, ruleset: str, sciencebox: int, ailevel: str = "hard") -> int:
    """Bulbs needed by `civ` for `tech` at this record's state."""
    if ruleset not in STYLE or not tech or tech in ("None", "none"):
        return 0
    if tech == "A_FUTURE":
        # is_future_tech -> style CIV1CIV2: base_tech_cost * techs_researched (then the same factors)
        me = next((p for p in rec["state"]["players"] if p["name"] == civ), None)
        n = len((me or {}).get("techs") or []) + 1
        bc = STYLE[ruleset][1] * n * STYLE[ruleset][2]
    else:
        bc = base_cost(ruleset, tech)
    if STYLE[ruleset][3] == "embassies":
        players = [p for p in rec["state"]["players"] if p.get("is_alive")]
        emb = ((rec.get("probes") or {}).get("C_belief") or {}).get(civ) or {}
        ds = emb.get("diplstate_hidden") or {}
        k = sum(1 for p in players if p["name"] != civ and tech in (p.get("techs") or [])
                and bool((ds.get(p["name"]) or {}).get("embassy")))
        if players:
            leak = bc * k * TECH_LEAK_PCT / len(players) / 100.0
            bc = 0.0 if leak > bc else bc - leak
    bc = bc * SCIENCE_COST.get(ailevel, 100) / 100.0
    bc = bc * sciencebox / 100.0
    return max(int(round(bc)), 1)
