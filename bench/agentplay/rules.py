"""Ruleset knowledge for the agent-play driver: what a city can build, what can be researched, how
units move.  Parsed from the ruleset directories copied out of the engine image
(`bench/agentplay/rulesets/{civ2civ3,classic}`), with the sandbox's own `.ruleset` parser.

The checks here are a *filter for the menu we show the model* and a first-line validator; the engine
remains the authority (an illegal production choice is replaced by the server's advisor at turn change,
and the driver reports that as a rejected action).  Requirement types covered: Tech, Building (City),
TerrainClass/Terrain/TerrainFlag/Extra (Adjacent, via a caller-supplied tile predicate), MinSize.
Anything else is treated as satisfied.
"""
from __future__ import annotations

import sys
import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "freeciv-sandbox"))
from sandboxes.freeciv.ruleset import parse, sections  # noqa: E402

from bench.agentplay.hexmap import TerrainRules  # noqa: E402

RULESET_DIR = ROOT / "bench/agentplay/rulesets"


def _parse_joined(path: Path) -> dict:
    """The sandbox parser drops the continuation lines of a multi-line list that has no backslash
    (`flags = "A", "B",` / `        "C"`): join a line ending in a comma with the next one first.
    (Found 2026-09-17: Settlers lost their "Cities" flag -> 118 founding orders rejected.)"""
    import tempfile
    raw = path.read_text(encoding="utf-8", errors="replace").split("\n")
    out: list[str] = []
    for ln in raw:
        body = ln.split(";", 1)[0].rstrip()
        if out and out[-1].split(";", 1)[0].rstrip().endswith(",") and not body.lstrip().startswith("[") and "=" not in body.split('"')[0]:
            out[-1] = out[-1].split(";", 1)[0].rstrip() + " " + ln.strip()
        else:
            out.append(ln)
    # Each load owns its file until parsing finishes, including concurrent loads.
    with tempfile.TemporaryDirectory(prefix="agentplay_rules_") as directory:
        tmp = Path(directory) / path.name
        tmp.write_text("\n".join(out), encoding="utf-8")
        result = parse(tmp)
    if not result:
        raise ValueError(f"Empty parsed ruleset: {path}")
    return result


def _lst(v: Any) -> list[str]:
    if v is None or v == "":
        return []
    return [str(x) for x in (v if isinstance(v, list) else [v])]


def _rn(s: dict) -> str:
    """Rule name = what savegames, reports and Lua use; `name` is the translated display name (civ2civ3: Amphitheater vs Colosseum)."""
    return str(s.get("rule_name") or s.get("name") or "")


def _reqs(tbl: Any) -> list[dict[str, Any]]:
    if tbl is None or tbl == "" or not hasattr(tbl, "as_dicts"):
        return []
    out = []
    for r in tbl.as_dicts():
        out.append({"type": str(r.get("type", "")), "name": str(r.get("name", "")), "range": str(r.get("range", "")),
                    "present": (str(r.get("present", "TRUE")).upper() != "FALSE")})
    return out


@dataclass
class UnitType:
    name: str
    uclass: str
    tech_req: str
    obsolete_by: str
    impr_req: str
    build_cost: int
    pop_cost: int
    attack: int
    defense: int
    hitpoints: int
    firepower: int
    move_rate: int
    transport_cap: int
    uk_shield: int
    uk_food: int
    uk_gold: int
    flags: list[str]
    roles: list[str]

    @property
    def military(self) -> bool:
        return "NonMil" not in self.flags and self.attack > 0

    @property
    def igter(self) -> bool:
        return "IgTer" in self.flags


@dataclass
class Building:
    name: str
    genus: str                # GreatWonder / SmallWonder / Improvement / Special
    reqs: list[dict[str, Any]]
    obsolete_by: list[dict[str, Any]]
    build_cost: int
    upkeep: int


@dataclass
class Rules:
    name: str
    units: dict[str, UnitType]
    buildings: dict[str, Building]
    techs: dict[str, dict[str, str]]          # name -> {req1, req2, root_req}
    governments: dict[str, list[dict]]        # name -> reqs
    terrain: TerrainRules
    citymindist: int = 3
    unitclass_flags: dict[str, list[str]] = field(default_factory=dict)

    # ---- tech tree ----
    def researchable(self, known: set[str]) -> list[str]:
        out = []
        for t, r in self.techs.items():
            if t in known or t == "None":
                continue
            if all(q in ("None", "") or q in known for q in (r["req1"], r["req2"])) and (r["root_req"] in ("None", "") or r["root_req"] in known):
                out.append(t)
        return sorted(out)

    def tech_known_or_none(self, t: str, known: set[str]) -> bool:
        return t in ("None", "", None) or t in known

    def unit_menu(self, known: set[str], city_impr: set[str], coastal: bool) -> list[UnitType]:
        out = []
        for u in self.units.values():
            if not self.tech_known_or_none(u.tech_req, known):
                continue
            if u.obsolete_by not in ("None", "") and u.obsolete_by in self.units and self.tech_known_or_none(self.units[u.obsolete_by].tech_req, known):
                continue                       # an upgrade is available: hide the obsolete type
            if u.impr_req not in ("None", "") and u.impr_req not in city_impr:
                continue
            if u.uclass in ("Sea", "Trireme") and not coastal:
                continue
            if "Barbarian" in u.roles or "BarbarianOnly" in u.flags or u.name in ("Barbarian Leader", "Leader"):
                continue
            if "NoBuild" in u.flags:
                continue
            out.append(u)
        return out

    def building_menu(self, known: set[str], city_impr: set[str], city_size: int, wonders_gone: set[str],
                      tile_pred: Callable[[str, str], bool]) -> list[Building]:
        """`tile_pred(req_type, name)` answers TerrainClass/Terrain/TerrainFlag/Extra 'Adjacent' requirements
        for this city's tile and its neighbours."""
        out = []
        for b in self.buildings.values():
            if b.name in city_impr or b.genus == "Special" or b.name in wonders_gone:
                continue
            if any(o["type"] == "Tech" and o["name"] in known and o["present"] for o in b.obsolete_by):
                continue
            ok = True
            for r in b.reqs:
                t, n, present = r["type"], r["name"], r["present"]
                if t == "Tech":
                    sat = n in known
                elif t == "Building":
                    sat = n in city_impr
                elif t == "MinSize":
                    sat = city_size >= int(n)
                elif t in ("TerrainClass", "Terrain", "TerrainFlag", "Extra"):
                    sat = tile_pred(t, n)
                else:
                    sat = True
                if sat != present:
                    ok = False; break
            if ok:
                out.append(b)
        return out

    def cost_of(self, item: str) -> int | None:
        if item in self.units:
            return self.units[item].build_cost
        if item in self.buildings:
            return self.buildings[item].build_cost
        return None


def rules_audit(rules: Rules) -> dict:
    counts = {key: len(getattr(rules, key)) for key in ("units", "buildings", "techs", "governments", "unitclass_flags")}
    counts["terrain"] = len(rules.terrain.by_char)
    if not all(counts.values()):
        raise ValueError(f"Incomplete {rules.name} rules: {counts}")
    encoded = json.dumps(asdict(rules), sort_keys=True, ensure_ascii=True).encode()
    return {"ruleset": rules.name, "counts": counts, "sha256": hashlib.sha256(encoded).hexdigest()}


def load_rules(name: str) -> Rules:
    d = RULESET_DIR / name
    U = _parse_joined(d / "units.ruleset"); B = _parse_joined(d / "buildings.ruleset"); T = _parse_joined(d / "techs.ruleset")
    G = _parse_joined(d / "governments.ruleset"); TR = _parse_joined(d / "terrain.ruleset"); GM = _parse_joined(d / "game.ruleset")
    units = {}
    for _, s in sections(U, "unit"):
        if not s.get("name"):
            continue
        units[_rn(s)] = UnitType(
            name=_rn(s), uclass=str(s.get("class", "Land")), tech_req=str(s.get("tech_req", "None")),
            obsolete_by=str(s.get("obsolete_by", "None")), impr_req=str(s.get("impr_req", "None")),
            build_cost=int(s.get("build_cost", 0)), pop_cost=int(s.get("pop_cost", 0)), attack=int(s.get("attack", 0)),
            defense=int(s.get("defense", 0)), hitpoints=int(s.get("hitpoints", 10)), firepower=int(s.get("firepower", 1)),
            move_rate=int(s.get("move_rate", 1)), transport_cap=int(s.get("transport_cap", 0)),
            uk_shield=int(s.get("uk_shield", 0)), uk_food=int(s.get("uk_food", 0)), uk_gold=int(s.get("uk_gold", 0)),
            flags=_lst(s.get("flags")), roles=_lst(s.get("roles")))
    uclass_flags = {_rn(s).split(":")[-1]: _lst(s.get("flags")) for _, s in sections(U, "unitclass") if s.get("name")}
    buildings = {}
    for _, s in sections(B, "building"):
        if not s.get("name"):
            continue
        buildings[_rn(s)] = Building(name=_rn(s), genus=str(s.get("genus", "Improvement")),
                                            reqs=_reqs(s.get("reqs")), obsolete_by=_reqs(s.get("obsolete_by")),
                                            build_cost=int(s.get("build_cost", 0)), upkeep=int(s.get("upkeep", 0)))
    techs = {}
    for _, s in sections(T, "advance"):
        if s.get("name"):
            techs[_rn(s)] = {"req1": str(s.get("req1", "None")), "req2": str(s.get("req2", "None")),
                                    "root_req": str(s.get("root_req", "None"))}
    govs = {_rn(s): _reqs(s.get("reqs")) for _, s in sections(G, "government") if s.get("name")}
    by_char = {}
    for _, s in sections(TR, "terrain"):
        if s.get("identifier"):
            by_char[str(s["identifier"])] = {"name": _rn(s), "class": str(s.get("class", "Land")),
                                             "movement_cost": int(s.get("movement_cost", 1))}
    params = TR.get("parameters", {})
    road_cost = 2; river_cost = 2
    for _, s in sections(TR, "extra"):
        if str(s.get("name", "")) == "Road" and s.get("move_cost") not in (None, ""):
            road_cost = int(s["move_cost"])
        if str(s.get("name", "")) == "River" and s.get("move_cost") not in (None, ""):
            river_cost = int(s["move_cost"])
    for _, s in sections(TR, "road"):
        if str(s.get("name", "")) == "Road" and s.get("move_cost") not in (None, ""):
            road_cost = int(s["move_cost"])
        if str(s.get("name", "")) == "River" and s.get("move_cost") not in (None, ""):
            river_cost = int(s["move_cost"])
    terrain = TerrainRules(move_fragments=int(params.get("move_fragments", 3)), igter_cost=int(params.get("igter_cost", 1)),
                           by_char=by_char, road_cost=road_cost, river_cost=river_cost)
    citymindist = 3
    st = GM.get("settings", {}).get("set")
    if st is not None and hasattr(st, "as_dicts"):
        for r in st.as_dicts():
            if str(r.get("name")) == "citymindist":
                citymindist = int(r.get("value", 3))
    return Rules(name=name, units=units, buildings=buildings, techs=techs, governments=govs, terrain=terrain,
                 citymindist=citymindist, unitclass_flags=uclass_flags)


if __name__ == "__main__":
    for n in ("civ2civ3", "classic"):
        r = load_rules(n)
        print(n, "units", len(r.units), "buildings", len(r.buildings), "techs", len(r.techs), "govs", list(r.governments),
              "frags", r.terrain.move_fragments, "road", r.terrain.road_cost, "river", r.terrain.river_cost, "citymindist", r.citymindist)
        known = {"Alphabet", "Bronze Working", "Ceremonial Burial", "Code of Laws", "Construction", "Currency", "Horseback Riding",
                 "Iron Working", "Literacy", "Map Making", "Masonry", "Mathematics", "Monarchy", "Pottery", "The Wheel", "Trade",
                 "Warrior Code", "Writing"}
        print("  researchable:", r.researchable(known))
        print("  units:", [u.name for u in r.unit_menu(known, set(), coastal=True)])
        print("  buildings:", [b.name for b in r.building_menu(known, set(), 3, set(), lambda t, n: t == "TerrainClass" and n == "Oceanic")])
        print("  terrain:", {c: (v["name"], v["class"], v["movement_cost"]) for c, v in r.terrain.by_char.items()})
