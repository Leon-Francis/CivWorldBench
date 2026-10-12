"""Map extraction: terrain, resources, improvements, ownership.

Freeciv is a spatial game, and the first version of the report left the map out
entirely -- a forecaster could not see who borders whom, who is boxed in, where
the resources are, or whether a war is physically possible.  ForecastBench-Sim
itself shipped a territory PNG every 10 turns for exactly this reason.

The `[map]` section stores ten parallel grids, all `ysize` rows of `xsize`
columns.  Everything here is read from the savegame's own vocabulary tables, not
hardcoded, because they are ruleset-defined:

  `tNNNN`         terrain, one character per tile.  The character->name mapping
                  is `[savefile] terrident`, so a ruleset that renames or adds
                  terrain still decodes.  (An earlier version of this module
                  hardcoded the mapping and had ' ' and 'i' swapped.)
  `eNN_NNNN`      extras, as ten bit-planes of hex nibbles: plane p, bit b is
                  index p*4+b into `[savefile] extras_vector`.  With 38 entries
                  this one layer carries BOTH the natural resources (Gold, Iron,
                  Coal, Fish, Gems, Wheat, Wine, Oil, Whales, ...) and the
                  terrain improvements (Irrigation, Mine, Road, Railroad,
                  River, Fortress, Hut, Pollution, ...).
  `ownerNNNN`     territory owner per tile, player index or '-'.
  `eownerNNNN`    owner of the extras on that tile (can differ from the tile).
  `workedNNNN`    id of the city working the tile, or '-'.  This is what turns
                  land into food and shields, so it is the economic layer.
  `sourceNNNN`    the tile or city id a territory claim radiates from -- so a
                  border is not just "who owns this" but "whose city projects
                  here", which is what makes border pressure legible.
  `kNN_NNNN`      known/explored mask, same hex-nibble bit-plane layout as
                  extras but with one bit per player: plane p, bit b is player
                  p*4+b.  This is the fog-of-war layer, so it is what makes a
                  per-player `partial` report honest rather than guessed at.
  `placingNNNN`   the extra currently being built on the tile, and
  `infra_turnsNNNN`  how many turns remain -- in-progress terraforming, i.e. a
                  commitment already made whose effect has not landed yet.
  `label_X_Y`     named map features ("Mud Volcanoes", "Deep Trench").

`extract_map` decodes the lot and derives the facts a forecaster can act on;
the raw grids are kept on the MapInfo so `geo_to_json` can record them in full.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .savegame import Savegame

# Terrain the classic-derived rulesets treat as water.  Resolved against
# `terrident` names rather than raw characters so the check survives a ruleset
# that reassigns identifiers.
WATER_NAMES = {"Lake", "Ocean", "Deep Ocean", "Inaccessible"}

# Extras that are natural resources (a property of the land) as opposed to
# player-built infrastructure.  Splitting them matters because resources are
# fixed endowments -- they explain *why* a civ is where it is -- while
# infrastructure is an outcome of past decisions.
RESOURCE_EXTRAS = {
    "Gold", "Iron", "Game", "Furs", "Coal", "Fish", "Fruit", "Gems", "Buffalo",
    "Wheat", "Oasis", "Peat", "Pheasant", "Resources", "Ivory", "Silk",
    "Spice", "Whales", "Wine", "Oil",
}
# Naturally occurring but not a worked resource.
NATURAL_EXTRAS = {"River", "Hut", "Ruins"}


@dataclass
class Territory:
    """Per-civilization spatial summary."""
    owner: str
    tiles: int
    land_tiles: int
    worked_tiles: int
    bbox: tuple[int, int, int, int]        # x0, y0, x1, y1
    centroid: tuple[float, float]
    terrain_mix: dict[str, int] = field(default_factory=dict)
    resources: dict[str, int] = field(default_factory=dict)
    infrastructure: dict[str, int] = field(default_factory=dict)
    neighbours: list[str] = field(default_factory=list)


@dataclass
class MapInfo:
    xsize: int
    ysize: int
    # raw grids, recorded in full
    terrain: list[str]                       # ysize strings of xsize chars
    owner: list[list[str]]
    extras: list[list[list[str]]]            # [y][x] -> list of extra names
    worked: list[list[str]]
    eowner: list[list[str]]                  # owner of the tile's extras
    source: list[list[str]]                  # id the territory claim comes from
    placing: list[list[str]]                 # extra under construction
    infra_turns: list[list[str]]             # turns left on that construction
    labels: dict[str, str]                   # "x,y" -> feature name
    known: dict[str, list[list[bool]]]   # player name -> [y][x] explored
    # vocabularies read from the savegame
    terrain_names: dict[str, str]            # char -> terrain name
    extras_vector: list[str]
    # derived
    territory: dict[str, Territory] = field(default_factory=dict)
    total_land: int = 0
    world_resources: dict[str, int] = field(default_factory=dict)
    unclaimed_resources: dict[str, int] = field(default_factory=dict)
    startpos: list[dict[str, Any]] = field(default_factory=list)


def _rows(sec: dict, prefix: str, numeric_only: bool = True) -> list[str]:
    keys = [k for k in sec if k.startswith(prefix)]
    if numeric_only:
        keys = [k for k in keys if k[len(prefix):].isdigit()]
    return [str(sec[k]) for k in sorted(keys)]


def _csv_rows(sec: dict, prefix: str) -> list[list[str]]:
    out = []
    for row in _rows(sec, prefix):
        out.append([c.strip() for c in row.split(",")])
    return out


def _decode_extras(sec: dict, vector: list[str], xsize: int, ysize: int):
    """Ten hex-nibble bit-planes -> per-tile list of extra names."""
    planes = sorted({k.split("_")[0] for k in sec
                     if len(k) > 3 and k[0] == "e" and "_" in k and k[1:3].isdigit()})
    grid = [[[] for _ in range(xsize)] for _ in range(ysize)]
    for pi, p in enumerate(planes):
        for y in range(ysize):
            row = str(sec.get(f"{p}_{y:04d}", ""))
            for x in range(min(xsize, len(row))):
                try:
                    nib = int(row[x], 16)
                except ValueError:
                    continue
                if not nib:
                    continue
                for b in range(4):
                    if nib & (1 << b):
                        idx = pi * 4 + b
                        if idx < len(vector):
                            grid[y][x].append(vector[idx])
    return grid


def _decode_known(sec: dict, player_names: list[str], xsize: int, ysize: int):
    """`kNN_NNNN` bit-planes -> per-player explored mask.

    One bit per player, so this is the fog of war: which tiles each civ has
    actually seen.  A `partial` report can then be restricted to what its
    viewpoint really knows instead of leaking the full map."""
    planes = sorted({k.split("_")[0] for k in sec
                     if len(k) > 3 and k[0] == "k" and "_" in k and k[1:3].isdigit()})
    out = {n: [[False] * xsize for _ in range(ysize)] for n in player_names}
    for pi, p in enumerate(planes):
        for y in range(ysize):
            row = str(sec.get(f"{p}_{y:04d}", ""))
            for x in range(min(xsize, len(row))):
                try:
                    nib = int(row[x], 16)
                except ValueError:
                    continue
                for b in range(4):
                    if nib & (1 << b):
                        idx = pi * 4 + b
                        if idx < len(player_names):
                            out[player_names[idx]][y][x] = True
    return out


def extract_map(sg: Savegame, player_names: list[str]) -> MapInfo:
    sec = sg.sections.get("map", {})

    terrain = _rows(sec, "t")
    ysize = len(terrain)
    xsize = len(terrain[0]) if terrain else 0

    # terrain character -> name, from the savegame's own table
    terrain_names: dict[str, str] = {}
    ti = sg.table("savefile", "terrident")
    if ti is not None:
        for row in ti.as_dicts():
            terrain_names[str(row.get("identifier", ""))] = str(row.get("name", ""))

    extras_vector = [str(x) for x in sg.get("savefile", "extras_vector", [])]
    extras = _decode_extras(sec, extras_vector, xsize, ysize)
    owner = _csv_rows(sec, "owner")
    worked = _csv_rows(sec, "worked")
    eowner = _csv_rows(sec, "eowner")
    source = _csv_rows(sec, "source")
    placing = _csv_rows(sec, "placing")
    infra_turns = _csv_rows(sec, "infra_turns")
    labels = {}
    for k, v in sec.items():
        if k.startswith("label_"):
            parts = k[6:].split("_")
            if len(parts) == 2 and all(q.isdigit() for q in parts):
                labels[f"{parts[0]},{parts[1]}"] = str(v)
    known = _decode_known(sec, player_names, xsize, ysize)

    startpos = []
    sp = sg.table("map", "startpos")
    if sp is not None:
        startpos = sp.as_dicts()

    mi = MapInfo(xsize=xsize, ysize=ysize, terrain=terrain, owner=owner,
                 extras=extras, worked=worked, eowner=eowner, source=source,
                 placing=placing, infra_turns=infra_turns, labels=labels,
                 known=known, terrain_names=terrain_names,
                 extras_vector=extras_vector, startpos=startpos)
    _derive(mi, player_names)
    _fill_neighbours(mi, player_names)
    return mi


def _is_water(mi: MapInfo, ch: str) -> bool:
    return mi.terrain_names.get(ch, "") in WATER_NAMES


def _derive(mi: MapInfo, player_names: list[str]) -> None:
    acc: dict[str, dict[str, Any]] = {}
    for y in range(mi.ysize):
        trow = mi.terrain[y]
        for x in range(min(mi.xsize, len(trow))):
            ch = trow[x]
            tname = mi.terrain_names.get(ch, ch)
            water = _is_water(mi, ch)
            if not water:
                mi.total_land += 1

            tile_extras = mi.extras[y][x] if y < len(mi.extras) else []
            for e in tile_extras:
                if e in RESOURCE_EXTRAS:
                    mi.world_resources[e] = mi.world_resources.get(e, 0) + 1

            tok = mi.owner[y][x] if (y < len(mi.owner) and x < len(mi.owner[y])) else "-"
            if not tok.isdigit() or int(tok) >= len(player_names):
                for e in tile_extras:
                    if e in RESOURCE_EXTRAS:
                        mi.unclaimed_resources[e] = mi.unclaimed_resources.get(e, 0) + 1
                continue

            name = player_names[int(tok)]
            a = acc.setdefault(name, {"n": 0, "land": 0, "worked": 0, "xs": [], "ys": [],
                                      "terr": {}, "res": {}, "infra": {}})
            a["n"] += 1
            a["land"] += int(not water)
            a["xs"].append(x)
            a["ys"].append(y)
            a["terr"][tname] = a["terr"].get(tname, 0) + 1
            if (y < len(mi.worked) and x < len(mi.worked[y])
                    and mi.worked[y][x] not in ("-", "")):
                a["worked"] += 1
            for e in tile_extras:
                if e in RESOURCE_EXTRAS:
                    a["res"][e] = a["res"].get(e, 0) + 1
                elif e not in NATURAL_EXTRAS:
                    a["infra"][e] = a["infra"].get(e, 0) + 1

    for name, a in acc.items():
        xs, ys = a["xs"], a["ys"]
        mi.territory[name] = Territory(
            owner=name, tiles=a["n"], land_tiles=a["land"], worked_tiles=a["worked"],
            bbox=(min(xs), min(ys), max(xs), max(ys)),
            centroid=(round(sum(xs) / len(xs), 1), round(sum(ys) / len(ys), 1)),
            terrain_mix=dict(sorted(a["terr"].items(), key=lambda kv: -kv[1])),
            resources=dict(sorted(a["res"].items(), key=lambda kv: -kv[1])),
            infrastructure=dict(sorted(a["infra"].items(), key=lambda kv: -kv[1])),
        )


def _fill_neighbours(mi: MapInfo, player_names: list[str]) -> None:
    """Two civs are neighbours if any of their tiles touch (8-connectivity).

    The single most forecast-relevant spatial fact: a civ can only attack, or be
    attacked by, someone it can reach.  A war declaration between two civs that
    do not border each other means something very different from one between
    neighbours.  (Caveat: this is land adjacency of *claimed* territory, so two
    civs facing each other across one tile of ocean read as non-adjacent even
    though a navy could cross.)
    """
    pairs: set[tuple[str, str]] = set()
    for y in range(len(mi.owner)):
        for x in range(len(mi.owner[y])):
            tok = mi.owner[y][x]
            if not tok.isdigit():
                continue
            a = int(tok)
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ny, nx = y + dy, x + dx
                    if not (0 <= ny < len(mi.owner)) or not (0 <= nx < len(mi.owner[ny])):
                        continue
                    tok2 = mi.owner[ny][nx]
                    if not tok2.isdigit():
                        continue
                    b = int(tok2)
                    if a == b or a >= len(player_names) or b >= len(player_names):
                        continue
                    pairs.add(tuple(sorted((player_names[a], player_names[b]))))
    for a, b in pairs:
        if a in mi.territory:
            mi.territory[a].neighbours.append(b)
        if b in mi.territory:
            mi.territory[b].neighbours.append(a)
    for t in mi.territory.values():
        t.neighbours.sort()


def ascii_map(mi: MapInfo, player_names: list[str], cell: int = 4) -> str:
    """Downsample the owner grid into a compact territory overview.

    Each character covers a `cell` x `cell` block and shows the index of
    whichever civ holds the most tiles in it, `~` for water, `.` for unclaimed
    land.  At cell=4 a 32x64 map becomes 8x16 -- small enough for a text report,
    coarse enough to still show who is where.  The full-resolution grids travel
    in the record's `geo` field alongside it.
    """
    out = []
    for by in range(0, mi.ysize, cell):
        row = []
        for bx in range(0, mi.xsize, cell):
            counts: dict[str, int] = {}
            land = water = 0
            for y in range(by, min(by + cell, mi.ysize)):
                trow = mi.terrain[y]
                for x in range(bx, min(bx + cell, mi.xsize)):
                    if x < len(trow):
                        if _is_water(mi, trow[x]):
                            water += 1
                        else:
                            land += 1
                    if y < len(mi.owner) and x < len(mi.owner[y]):
                        tok = mi.owner[y][x]
                        if tok.isdigit():
                            counts[tok] = counts.get(tok, 0) + 1
            if counts:
                row.append(max(counts, key=lambda k: counts[k]))
            elif water > land:
                row.append("~")
            else:
                row.append(".")
        out.append("".join(row))
    return "\n".join(out)


def terrain_map(mi: MapInfo) -> str:
    """The full-resolution terrain grid, as the savegame writes it."""
    return "\n".join(mi.terrain)


def resource_map(mi: MapInfo, cell: int = 4) -> str:
    """Downsampled resource density: count of resource tiles per cell block."""
    out = []
    for by in range(0, mi.ysize, cell):
        row = []
        for bx in range(0, mi.xsize, cell):
            n = 0
            for y in range(by, min(by + cell, mi.ysize)):
                for x in range(bx, min(bx + cell, mi.xsize)):
                    if y < len(mi.extras) and x < len(mi.extras[y]):
                        n += sum(1 for e in mi.extras[y][x] if e in RESOURCE_EXTRAS)
            row.append("." if n == 0 else (str(n) if n < 10 else "+"))
        out.append("".join(row))
    return "\n".join(out)


def known_map(mi: MapInfo, viewer: str, player_names: list[str], cell: int = 4) -> str:
    """The territory map as `viewer` actually knows it.

    Unexplored blocks render as `?`.  This is what keeps the `partial` report
    honest: without it a per-player report would still be quoting the full map,
    which is precisely the leak the POMDP framing is meant to avoid."""
    seen = mi.known.get(viewer)
    out = []
    for by in range(0, mi.ysize, cell):
        row = []
        for bx in range(0, mi.xsize, cell):
            counts: dict[str, int] = {}
            land = water = unknown = 0
            for y in range(by, min(by + cell, mi.ysize)):
                trow = mi.terrain[y]
                for x in range(bx, min(bx + cell, mi.xsize)):
                    if seen is not None and not seen[y][x]:
                        unknown += 1
                        continue
                    if x < len(trow):
                        if _is_water(mi, trow[x]):
                            water += 1
                        else:
                            land += 1
                    if y < len(mi.owner) and x < len(mi.owner[y]):
                        tok = mi.owner[y][x]
                        if tok.isdigit():
                            counts[tok] = counts.get(tok, 0) + 1
            if unknown > (land + water):
                row.append("?")
            elif counts:
                row.append(max(counts, key=lambda k: counts[k]))
            elif water > land:
                row.append("~")
            else:
                row.append(".")
        out.append("".join(row))
    return "\n".join(out)


def explored_fraction(mi: MapInfo, viewer: str) -> float:
    g = mi.known.get(viewer)
    if not g:
        return 0.0
    return sum(sum(r) for r in g) / float(mi.xsize * mi.ysize or 1)


def geo_to_json(mi: MapInfo) -> dict[str, Any]:
    """Full record payload -- raw grids included, per the decision to record
    everything and worry about volume later."""
    import dataclasses
    return {
        "xsize": mi.xsize, "ysize": mi.ysize, "total_land": mi.total_land,
        "terrain_legend": mi.terrain_names,
        "extras_vector": mi.extras_vector,
        "grids": {
            "terrain": mi.terrain,
            "owner": [",".join(r) for r in mi.owner],
            "worked": [",".join(r) for r in mi.worked],
            "eowner": [",".join(r) for r in mi.eowner],
            "source": [",".join(r) for r in mi.source],
            "placing": [",".join(r) for r in mi.placing],
            "infra_turns": [",".join(r) for r in mi.infra_turns],
            "known": {n: ["".join("1" if v else "0" for v in row) for row in grid]
                      for n, grid in mi.known.items()},
            # extras as an explicit per-tile name list: verbose but decoded once
            # here rather than re-deriving the bit-plane layout downstream.
            "extras": [[mi.extras[y][x] for x in range(mi.xsize)] for y in range(mi.ysize)],
        },
        "startpos": mi.startpos,
        "labels": mi.labels,
        "world_resources": mi.world_resources,
        "unclaimed_resources": mi.unclaimed_resources,
        "territory": {k: dataclasses.asdict(v) for k, v in mi.territory.items()},
    }
