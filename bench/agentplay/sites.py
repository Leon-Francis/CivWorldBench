"""City-site logic shared by the panel (advisor list) and the engine (automatic targets for `found_city`).

A legal site = explored land the unit class can stand on, not occupied by a foreign unit/city, at least
`citymindist` (real distance) from every city the ego knows, on unclaimed or own territory, and not a
terrain that scores 0 (Mountains, Glacier).  Ranking = terrain score + river + coastal + special resource.
"""
from __future__ import annotations

import collections

from bench.agentplay.hexmap import Board, Geometry

TERRAIN_SCORE = {"Grassland": 3, "Plains": 3, "Hills": 3, "Desert": 1, "Tundra": 1, "Forest": 1, "Jungle": 1, "Swamp": 1}
_NOT_SPECIAL = {"River", "Road", "Railroad", "Irrigation", "Mine", "Farmland", "Hut", "Ruins", "Fort", "Fortress", "Pollution",
                "Fallout", "Buoy", "Airbase", "Airstrip", "Maglev", "Oil Well", "Oil Platform"}


def site_score(board: Board, geo: Geometry, t: tuple[int, int], extras) -> tuple[int, str, list[str]]:
    tname = board.rules.by_char.get(board.tchar(*t), {}).get("name", "?")
    sc = TERRAIN_SCORE.get(tname, 0)
    if sc == 0:
        return 0, tname, []
    ex = set(extras[t[1]][t[0]]); tags = []
    if "River" in ex:
        sc += 1; tags.append("river")
    if any(board.rules.is_water(board.tchar(*nb)) for _, nb in geo.neighbours(*t)):
        sc += 1; tags.append("coastal")
    spec = [e for e in ex if e not in _NOT_SPECIAL]
    if spec:
        sc += 1; tags.append(spec[0])
    return sc, tname, tags


def is_legal_site(board: Board, geo: Geometry, t: tuple[int, int], known_cities: dict, ego_name: str, citymindist: int) -> bool:
    if not board.is_known(*t) or t in board.blocked or not board.native_for("Small Land", *t):
        return False
    if any(geo.distance(t, c) < citymindist for c in known_cities):
        return False
    if board.owner_at(*t) not in (None, ego_name):
        return False
    return TERRAIN_SCORE.get(board.rules.by_char.get(board.tchar(*t), {}).get("name", "?"), 0) > 0


def land_sites_from(board: Board, geo: Geometry, origins: list[tuple[int, int]], known_cities: dict, ego_name: str,
                    citymindist: int, extras, max_dist: int = 10) -> list[tuple[int, int, tuple[int, int]]]:
    """[(dist_steps, -score, tile)] sorted best-first, reachable on foot from any origin (hex steps, not move cost)."""
    dist: dict[tuple[int, int], int] = {}
    dq = collections.deque()
    for o in origins:
        dist[tuple(o)] = 0; dq.append(tuple(o))
    while dq:
        cur = dq.popleft(); d = dist[cur]
        if d >= max_dist:
            continue
        for _, nb in geo.neighbours(*cur):
            if nb in dist or not board.is_known(*nb) or nb in board.blocked or not board.native_for("Small Land", *nb):
                continue
            dist[nb] = d + 1; dq.append(nb)
    out = []
    for t, d in dist.items():
        if is_legal_site(board, geo, t, known_cities, ego_name, citymindist):
            sc, _, _ = site_score(board, geo, t, extras)
            out.append((d, -sc, t))
    # prefer close *and* good: rank by (steps - score) then steps
    out.sort(key=lambda x: (x[0] + x[1], x[0]))
    return out


def sea_sites_from(board: Board, geo: Geometry, boat_tile: tuple[int, int], boat_class: str, known_cities: dict, ego_name: str,
                   citymindist: int, extras, max_cost: int = 90) -> list[tuple[int, int, tuple[int, int], tuple[int, int]]]:
    """Coastal legal sites reachable by a boat: [(boat_cost_frags, -score, site, landing_water_tile)] best-first."""
    dm = board.dist_map(boat_class, boat_tile, max_cost=max_cost)
    best: dict[tuple[int, int], tuple[int, tuple[int, int]]] = {}
    for w, (cost, _) in dm.items():
        for _, nb in geo.neighbours(*w):
            if nb in best and best[nb][0] <= cost:
                continue
            if is_legal_site(board, geo, nb, known_cities, ego_name, citymindist):
                best[nb] = (cost, w)
    out = []
    for site, (cost, w) in best.items():
        sc, _, _ = site_score(board, geo, site, extras)
        out.append((cost, -sc, site, w))
    out.sort(key=lambda x: (x[0] / 6 + x[1], x[0]))
    return out
