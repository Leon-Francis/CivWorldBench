"""Map geometry for the agent-play driver: iso-hex adjacency, passability, move costs, paths.

Every world in this corpus uses the rulesets' default topology `WRAPX|WRAPY|ISO|HEX` (checked in the
savegame `[settings]` table).  Savegames, the renderer and the Lua API all speak *native* coordinates
(x, y); adjacency, however, is defined on freeciv's *map* coordinates, and on an iso-hex map only six of
the eight compass directions exist (SE and NW are removed).  This module does the conversion once so the
rest of the driver can treat the map as a graph of native (x, y) nodes.

  native -> map : mx = (ny + (ny & 1)) // 2 + nx ;  my = ny - mx + xsize
  map -> native : ny = mx + my - xsize          ;  nx = (2*mx - ny - (ny & 1)) // 2
(server/../common/map.h NATIVE_TO_MAP_POS / MAP_TO_NATIVE_POS for the ISO case)

Move costs are in *fragments* (`move_fragments`: 6 in civ2civ3, 3 in classic) so they can be compared
with the unit's `moves` field in the savegame and passed to `edit.unit_move(unit, tile, movecost)`.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Callable, Iterable

# DIR8 steps in MAP coordinates, iso-hex subset: NE and SW do not exist (common/map.c
# map_vector_to_real_distance: "Iso-hex: you can't move NE or SW").  Verified against the engine 2026-09-17 with the
# validating 10-argument edit.unit_move: N/E/SE/S/W/NW moved (sq_distance 1), NE/SW were refused (sq_distance 4).
_DIRS_ISOHEX = {"N": (0, -1), "E": (1, 0), "SE": (1, 1), "S": (0, 1), "W": (-1, 0), "NW": (-1, -1)}
# Freeciv's "cardinal" directions on iso-hex (river movement is cardinal-only): the non-diagonal four in map coords.
_CARDINAL = {"N", "E", "S", "W"}

WATER_CLASSES = {"Oceanic"}


@dataclass
class Geometry:
    xsize: int
    ysize: int
    wrapx: bool = True
    wrapy: bool = True

    def to_map(self, nx: int, ny: int) -> tuple[int, int]:
        mx = (ny + (ny & 1)) // 2 + nx
        return mx, ny - mx + self.xsize

    def to_native(self, mx: int, my: int) -> tuple[int, int]:
        ny = mx + my - self.xsize
        nx = (2 * mx - ny - (ny & 1)) // 2
        return nx, ny

    def normalize(self, nx: int, ny: int) -> tuple[int, int] | None:
        if self.wrapx:
            nx %= self.xsize
        elif not 0 <= nx < self.xsize:
            return None
        if self.wrapy:
            ny %= self.ysize
        elif not 0 <= ny < self.ysize:
            return None
        return nx, ny

    def neighbours(self, nx: int, ny: int) -> list[tuple[str, tuple[int, int]]]:
        """[(dir_name, (nx, ny))] for the six iso-hex neighbours, wrapped."""
        mx, my = self.to_map(nx, ny)
        out = []
        for d, (dx, dy) in _DIRS_ISOHEX.items():
            n = self.normalize(*self.to_native(mx + dx, my + dy))
            if n is not None:
                out.append((d, n))
        return out

    def adjacent(self, a: tuple[int, int], b: tuple[int, int]) -> bool:
        return any(n == tuple(b) for _, n in self.neighbours(*a))

    def distance(self, a: tuple[int, int], b: tuple[int, int]) -> int:
        """Hex distance (moves) between two native positions, honouring wrap: BFS-free closed form
        on map coordinates with the iso-hex metric (cube distance on the 6-direction lattice)."""
        best = None
        ax, ay = self.to_map(*a)
        for wx in ((-1, 0, 1) if self.wrapx else (0,)):
            for wy in ((-1, 0, 1) if self.wrapy else (0,)):
                bx, by = self.to_map(b[0] + wx * self.xsize, b[1] + wy * self.ysize)
                dx, dy = bx - ax, by - ay
                # freeciv map_vector_to_real_distance for iso-hex: a step can change dx and dy together only in the
                # same direction (SE/NW), so opposite-sign vectors cost |dx|+|dy|, same-sign ones max(|dx|,|dy|)
                d = abs(dx) + abs(dy) if (dx < 0 < dy) or (dy < 0 < dx) else max(abs(dx), abs(dy))
                best = d if best is None else min(best, d)
        return best


@dataclass
class TerrainRules:
    """What the path-finder needs from terrain.ruleset, keyed by the one-char identifier used in the
    savegame/renderer grids."""
    move_fragments: int
    igter_cost: int
    by_char: dict[str, dict] = field(default_factory=dict)   # char -> {name, class, movement_cost}
    road_cost: int = 2                                        # fragments along a road
    river_cost: int = 2                                       # fragments along a river (cardinal only)

    def terrain_class(self, ch: str) -> str:
        return self.by_char.get(ch, {}).get("class", "Land")

    def is_water(self, ch: str) -> bool:
        return self.terrain_class(ch) == "Oceanic"

    def is_inaccessible(self, ch: str) -> bool:
        return self.by_char.get(ch, {}).get("name", "") == "Inaccessible"


@dataclass
class Board:
    """A snapshot of the tiles as the ego knows them: terrain chars, extras, known mask, blockers."""
    geo: Geometry
    rules: TerrainRules
    terrain: list[str]                    # [y] -> row string of terrain chars
    extras: list[list[list[str]]]         # [y][x] -> extra names
    known: list[list[bool]]               # ego's explored mask
    blocked: set[tuple[int, int]] = field(default_factory=set)   # tiles the ego may not enter (foreign units/cities)
    enterable_cities: set[tuple[int, int]] = field(default_factory=set)  # own cities (land units can always stand there)
    owner: list[list[str]] | None = None                          # territory grid tokens ('.', '?', digit = player index)
    player_names: list[str] = field(default_factory=list)
    boat_tiles: set[tuple[int, int]] = field(default_factory=set)  # own transports with free capacity: land units may step onto them (embark)

    def owner_at(self, x: int, y: int) -> str | None:
        if self.owner is None:
            return None
        tok = self.owner[y][x]
        return self.player_names[int(tok)] if tok.isdigit() and int(tok) < len(self.player_names) else None

    def tchar(self, x: int, y: int) -> str:
        return self.terrain[y][x]

    def is_known(self, x: int, y: int) -> bool:
        return bool(self.known[y][x])

    def native_for(self, unit_class: str, x: int, y: int) -> bool:
        ch = self.tchar(x, y)
        if self.rules.is_inaccessible(ch):
            return False
        water = self.rules.is_water(ch)
        if unit_class in ("Sea", "Trireme"):
            return water or (x, y) in self.enterable_cities
        if unit_class in ("Air", "Helicopter", "Missile"):
            return True
        return (not water) or (x, y) in self.boat_tiles

    def step_cost(self, unit_class: str, a: tuple[int, int], b: tuple[int, int], dirname: str,
                  igter: bool = False) -> int:
        """Fragments to move from a to b (adjacent).  Terrain cost of the destination, or the road/river
        shortcut when both tiles carry it.  Sea units: one move per tile."""
        r = self.rules
        if unit_class in ("Sea", "Trireme", "Air", "Helicopter", "Missile"):
            return r.move_fragments
        ea, eb = set(self.extras[a[1]][a[0]]), set(self.extras[b[1]][b[0]])
        base = int(r.by_char.get(self.tchar(*b), {}).get("movement_cost", 1)) * r.move_fragments
        if igter:
            base = min(base, r.igter_cost)
        if {"Road", "Railroad", "Maglev"} & ea and {"Road", "Railroad", "Maglev"} & eb:
            return min(base, r.road_cost)
        if "River" in ea and "River" in eb and dirname in _CARDINAL:
            return min(base, r.river_cost)
        return base

    def dist_map(self, unit_class: str, start: tuple[int, int], max_cost: int = 90, igter: bool = False) -> dict:
        """Dijkstra from `start` over known, native, unblocked tiles: {tile: (cost_fragments, prev_tile)} (start included)."""
        start = tuple(start)
        dist = {start: (0, None)}; pq = [(0, start)]
        while pq:
            d, cur = heapq.heappop(pq)
            if d > dist[cur][0] or d > max_cost:
                continue
            for dirname, nb in self.geo.neighbours(*cur):
                if not self.is_known(*nb) or nb in self.blocked or not self.native_for(unit_class, *nb):
                    continue
                nd = d + self.step_cost(unit_class, cur, nb, dirname, igter)
                if nd <= max_cost and nd < dist.get(nb, (1 << 30,))[0]:
                    dist[nb] = (nd, cur); heapq.heappush(pq, (nd, nb))
        return dist

    def path(self, unit_class: str, start: tuple[int, int], goal: tuple[int, int],
             igter: bool = False, allow_goal_blocked: bool = False,
             max_nodes: int = 20000) -> list[tuple[str, tuple[int, int], int]] | None:
        """Dijkstra over known, native, unblocked tiles.  Returns [(dir, tile, cost_fragments)] or None.
        `allow_goal_blocked` lets the path end on a blocked tile (an attack / conquest target)."""
        start, goal = tuple(start), tuple(goal)
        if start == goal:
            return []
        dist = {start: 0}; prev: dict[tuple[int, int], tuple[tuple[int, int], str, int]] = {}
        pq = [(0, start)]; seen = 0
        while pq:
            d, cur = heapq.heappop(pq)
            if d > dist.get(cur, 1 << 30):
                continue
            if cur == goal:
                break
            seen += 1
            if seen > max_nodes:
                return None
            for dirname, nb in self.geo.neighbours(*cur):
                if not self.is_known(*nb):
                    continue
                if nb in self.blocked and not (allow_goal_blocked and nb == goal):
                    continue
                if not self.native_for(unit_class, *nb) and not (allow_goal_blocked and nb == goal):
                    continue
                c = self.step_cost(unit_class, cur, nb, dirname, igter)
                nd = d + c
                if nd < dist.get(nb, 1 << 30):
                    dist[nb] = nd; prev[nb] = (cur, dirname, c)
                    heapq.heappush(pq, (nd, nb))
        if goal not in prev:
            return None
        out = []; cur = goal
        while cur != start:
            p, dirname, c = prev[cur]
            out.append((dirname, cur, c)); cur = p
        return out[::-1]
