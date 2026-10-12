"""Task 3 (v2): turn an arm's recorded `applied` string into (a) an English one-line
description with the magnitude the model can judge and (b) where the variable is in
the report, an edited copy of the t1 record so the last observation frame shows the
intervened value (09 §5.2: edit channel for gold / bulbs / techs / cities / wonders,
relative text for the rest).

Deviation from v1's silent edit: the edit is ALSO stated in the counterfactual
preamble.  Task 3 now asks the full 1a/1b/1c set and scores "clean" counterfactual
behaviour (unaffected quantities must not move), which only makes sense if the model
knows an intervention happened.  The placebo pair still controls for wording.
"""
from __future__ import annotations

import copy
import re

_GOLD = re.compile(r"^(?P<civ>.+?) gold (?P<a>\d+) -> (?P<b>\d+)$")
_BULBS = re.compile(r"give_bulbs\(p, (?P<n>\d+)\)")
_CITYKILL = re.compile(r"^razed (?P<k>\d+) (?P<which>largest|smallest) cities of (?P<civ>.+?) \[(?P<list>.*?)\]")
_TECH = re.compile(r"(?:GOVTECH|GIVETECH)_(?P<tech>[A-Za-z][A-Za-z .'_-]*?)=")
_FOG = re.compile(r"^revealed (?P<n>\d+) tiles to (?P<civ>.+?) \((?P<how>\w+)\)")
_WONDER = re.compile(r"^granted (?P<k>\d+) buildings to (?P<civ>.+?): (?P<list>.+)$")
_LOVE = re.compile(r"^(?P<civ>.+?) love toward (?P<other>.+?) (?P<a>-?\d+) -> (?P<b>-?\d+)$")
_TRAIT = re.compile(r'trait_mod\(p, "(?P<dim>\w+)", (?P<n>-?\d+)\)')
_SETTING = re.compile(r"^setting (?P<name>\w+) (?P<a>\d+) -> (?P<b>\d+) \(x(?P<f>[\d.]+)\)")


class Unparseable(Exception):
    pass


def _player(rec: dict, name: str) -> dict | None:
    return next((p for p in rec["state"]["players"] if p["name"] == name), None)


def describe_and_edit(rec_t1: dict, op: dict) -> tuple[dict | None, str, str]:
    """Returns (edited record or None, description, channel 'edit'|'text')."""
    oid = (op.get("id") or "").split("__placebo")[0]
    applied = (op.get("applied") or "").strip()
    tgt = op.get("target")

    if oid == "L1-gold-swing":
        m = _GOLD.match(applied)
        if not m:
            raise Unparseable(applied)
        out = copy.deepcopy(rec_t1); p = _player(out, m["civ"])
        if p is None or int(p["gold"]) != int(m["a"]):
            raise Unparseable(f"gold before mismatch: {applied}")
        p["gold"] = int(m["b"])
        return out, f"{m['civ']}'s treasury changes from {m['a']} to {m['b']} gold.", "edit"

    if oid == "L1-bulbs":
        m = _BULBS.search(applied)
        if not m:
            raise Unparseable(applied)
        out = copy.deepcopy(rec_t1); p = _player(out, tgt)
        if p is None:
            raise Unparseable(f"no target {tgt}")
        before = int(p.get("bulbs_last") or 0)
        p["bulbs_last"] = before + int(m["n"])
        return out, (f"{tgt} gains {m['n']} research points toward its current research "
                     f"(its accumulated bulbs go from {before} to {before + int(m['n'])})."), "edit"

    if oid == "L2-city-kill":
        m = _CITYKILL.match(applied)
        if not m:
            raise Unparseable(applied)
        names = [x.split("(")[0].strip() for x in m["list"].split(",") if x.strip()]
        out = copy.deepcopy(rec_t1); p = _player(out, m["civ"])
        if p is None:
            raise Unparseable(f"no civ {m['civ']}")
        keep = [c for c in p["cities"] if c["name"] not in names]
        if len(keep) == len(p["cities"]):
            raise Unparseable(f"cities not found: {names}")
        gone = {c["name"] for c in p["cities"]} - {c["name"] for c in keep}
        p["cities"] = keep
        # units homed in a razed city are recorded in the save; the report only
        # counts units, so nothing else to edit
        return out, (f"{m['civ']}'s {'largest' if m['which']=='largest' else 'smallest'} "
                     f"{len(gone)} of its cities are destroyed: {', '.join(sorted(gone))}."), "edit"

    if oid == "L2-gov-indirect":
        m = _TECH.search(applied)
        if not m:
            raise Unparseable(applied)
        tech = m["tech"].strip().replace("_", " ")
        out = copy.deepcopy(rec_t1); p = _player(out, tgt)
        if p is None:
            raise Unparseable(f"no target {tgt}")
        if tech not in p["techs"]:
            p["techs"] = sorted(list(p["techs"]) + [tech])
        return out, f"{tgt} acquires the technology {tech}.", "edit"

    if oid == "L3-wonder":
        m = _WONDER.match(applied)
        if not m:
            raise Unparseable(applied)
        out = copy.deepcopy(rec_t1); p = _player(out, m["civ"])
        if p is None:
            raise Unparseable(f"no civ {m['civ']}")
        done = []
        for item in m["list"].split(";"):
            if "@" not in item:
                continue
            what, city = [x.strip() for x in item.split("@", 1)]
            for c in p["cities"]:
                if c["name"] == city and isinstance(c.get("improvements"), dict):
                    c["improvements"][what] = True
                    done.append(f"{what} in {city}")
        if not done:
            raise Unparseable(f"no city matched: {applied}")
        return out, f"{m['civ']} completes {len(done)} building{'s' if len(done) > 1 else ''}: {'; '.join(done)}.", "edit"

    if oid == "L3-fog":
        m = _FOG.match(applied)
        if not m:
            raise Unparseable(applied)
        how = "the entire map, every tile" if m["how"] == "all" else f"{m['n']} tiles just beyond the edge of its explored area"
        return None, f"{m['civ']} suddenly learns the geography of {how} ({m['n']} tiles revealed).", "text"

    if oid == "L4-love":
        m = _LOVE.match(applied)
        if not m:
            raise Unparseable(applied)
        return None, (f"{m['civ']}'s hidden attitude toward {m['other']} changes from {m['a']} to {m['b']} "
                      f"on the -1000 (hostile) .. +1000 (friendly) scale."), "text"

    if oid.startswith("L4-trait"):
        # v3: single-dimension operators L4-trait-{aggr,expa,trade,build}[90]; the old two-dimension
        # `L4-trait` wrote two trait_mod calls and this regex reported only the first (14 §决定3).
        ms = list(_TRAIT.finditer(applied))
        if not ms:
            raise Unparseable(applied)
        gloss = {"Aggressive": "starts and prolongs wars, refuses ceasefires", "Expansionist": "founds cities",
                 "Trader": "builds caravans and trade routes", "Builder": "prefers buildings over units"}
        parts = [f"{m['dim']} ({gloss.get(m['dim'], '')}) rises by {int(m['n'])}" for m in ms]
        return None, (f"{tgt}'s hidden {' and '.join(parts)} on the 10..150 personality scale where 50 is "
                      f"average; {'this value is' if len(ms) == 1 else 'these values are'} fixed for the rest of the game "
                      f"and nothing else about {tgt} changes."), "text"

    if oid == "WP-disasters":
        m = _SETTING.match(applied)
        if not m:
            raise Unparseable(applied)
        # the absolute setting values (10 -> 30) are exactly task 2's hidden "disaster frequency" answer;
        # keep the magnitude, drop the numbers (review run 2026-09-15)
        return None, f"From now on natural disasters strike this world {m['f']} times as often as before.", "text"

    raise Unparseable(f"no handler for {oid}: {applied}")
