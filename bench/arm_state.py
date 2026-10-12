"""Rebuild corpus-style records from a counterfactual arm's savegames.

Arms keep a savegame every 10 turns (`arms/t<t>_<op>/saves/*.sav.zst`).  Task 3 truth
is computed from these -- NOT from the arm's `resolved` dict, whose
`military_units` is the misnamed savegame `nunits` (all units).  The same code path
that built the mainline records (state.extract / geo.extract_map / probes.extract_probes)
is used, so the arm records render with the same bench renderer.
"""
from __future__ import annotations

import dataclasses
import re
import tempfile
from pathlib import Path

from sandboxes.freeciv.geo import extract_map, geo_to_json
from sandboxes.freeciv.probes import extract_probes
from sandboxes.freeciv.savegame import Savegame
from sandboxes.freeciv.shard import decompress_file
from sandboxes.freeciv.state import extract

_T = re.compile(r"-T(\d{4})-")


def arm_save_turns(arm_dir: Path) -> dict[int, Path]:
    out = {}
    for p in (arm_dir / "saves").glob("*.sav.zst"):
        m = _T.search(p.name)
        if m:
            out[int(m.group(1))] = p
    return out


def arm_record(sav: Path, world_id: str, with_probes: bool = True) -> dict:
    with tempfile.NamedTemporaryFile(suffix=".sav", delete=True) as tmp:
        decompress_file(sav, Path(tmp.name))
        sg = Savegame.load(Path(tmp.name))
        ws = extract(sg, world_id)
        mi = extract_map(sg, [p.name for p in ws.players])
        rec = {"t": ws.turn, "year": ws.year, "ruleset": ws.ruleset, "world_id": world_id,
               "state": dataclasses.asdict(ws), "geo": geo_to_json(mi), "events": [],
               "probes": extract_probes(sg, mi) if with_probes else {}}
    return rec


def arm_record_light(sav: Path, world_id: str) -> dict:
    """State only (no geo, no probes): what the 1b window checks need at intermediate turns
    (gold, government, city owners, units, diplomacy).  ~5x cheaper than `arm_record`."""
    with tempfile.NamedTemporaryFile(suffix=".sav", delete=True) as tmp:
        decompress_file(sav, Path(tmp.name))
        sg = Savegame.load(Path(tmp.name))
        ws = extract(sg, world_id)
        return {"t": ws.turn, "year": ws.year, "ruleset": ws.ruleset, "world_id": world_id,
                "state": dataclasses.asdict(ws), "geo": None, "events": [], "probes": {}, "_light": True}


def arm_records_light(arm_dir: Path, world_id: str, turns: list[int]) -> dict[int, dict]:
    """Light records at exactly the requested turns (per-turn arms only: no nearest-save fallback)."""
    saves = arm_save_turns(arm_dir)
    return {t: arm_record_light(saves[t], world_id) for t in turns if t in saves}


def arm_records(arm_dir: Path, world_id: str, turns: list[int], with_probes: bool = True) -> dict[int, dict]:
    """Records at the requested turns (nearest saved turn at or before each, since
    saves are every 10 turns and horizons are multiples of 10)."""
    saves = arm_save_turns(arm_dir)
    out = {}
    for t in turns:
        cand = [s for s in saves if s <= t]
        if not cand:
            continue
        st = max(cand)
        if t - st > 9:
            continue
        out[t] = arm_record(saves[st], world_id, with_probes)
        out[t]["t"] = t
    return out
