"""Prototype for the benchmark's C tier: an observation built from the ENGINE'S OWN
stale memory, not from the current truth.

Why this tier exists.  Tiers A (omniscient) and B (single-civ) both show the *current*
truth for whatever passes the mask, so a single frame is still nearly a sufficient
statistic and a belief stays redundant -- the exact structure that made v1's belief
useless.  Tier C breaks that: foreign cities are rendered from `probes.C_belief[viewer].dc`
(freeciv's `dumb_city` memory, measured 33-77% stale), and foreign units are shown only
where the viewer can *currently* see them.  Now one frame is provably insufficient and
integrating several frames is the only way to recover the truth -- which is what a belief
is for.

Two facts had to be established before this was worth building, and both are measured here:

  * `dc` really is stale.  At tier-S t=201 the viewer remembers 20 foreign cities and
    35% of the remembered sizes are wrong (New York: believed 3, actual 11).
  * The corpus has **no current-vision mask** -- `geo.grids.known` is *cumulative* and hits
    99% by t=201, i.e. it masks nothing. So current vision is reconstructed here from the
    viewer's own units and cities, which is freeciv's actual rule (vision radius around
    own units/cities). Measured: 32.8% of the map against known's 99.0%, hiding 56 of 84
    foreign units.

Standalone on purpose: `report.py` is being edited by the v2 work, so this proves
feasibility without touching it. The clean integration is noted at the bottom.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "bundle"))

# freeciv's vision radius is expressed as radius_sq; 2 is the usual unit/city value in
# civ2civ3, which makes the visible disc {(dx,dy) : dx^2+dy^2 <= radius_sq}.
VISION_RADIUS_SQ = 2


def vision_mask(me: dict, X: int, Y: int, radius_sq: int = VISION_RADIUS_SQ) -> set[tuple[int, int]]:
    """Tiles the viewer can see RIGHT NOW: the discs around its own units and cities.

    Reconstructed rather than read, because the corpus stores only the cumulative
    `known` grid.  x wraps (freeciv maps are cylindrical), y does not.
    """
    out: set[tuple[int, int]] = set()
    r = int(radius_sq ** 0.5) + 1
    for src in list(me.get("units") or []) + list(me.get("cities") or []):
        cx, cy = src["x"], src["y"]
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                if dx * dx + dy * dy <= radius_sq and 0 <= cy + dy < Y:
                    out.add(((cx + dx) % X, cy + dy))
    return out


def owner_name(idx: int, players: list) -> str:
    """Decode `dc.owner` -- the owner AS REMEMBERED.

    ⚠️ It disagrees with the current truth for 38 of 1,184 sampled entries, and that is
    the point, not a decoding bug: I first "verified" this mapping by checking the
    remembered owner against the city's *present* owner and concluded the mapping was
    broken.  It is not.  Tracing one case: St. Louis passed Grant -> Zulu (t123) -> Grant
    (t192), and at t201 Dinis still remembers it as Zulu.  So `dc` is stale in OWNERSHIP as
    well as in size -- a richer form of partial observability than the size drift alone,
    and one a belief can only fix by integrating several frames.

    Player id is the index into `state.players` (players carry no `id` field; verified by
    tracing captures, and no `dc.owner` exceeds the player count).
    """
    return players[idx]["name"] if 0 <= idx < len(players) else f"player#{idx}"


def improvements_from_bits(bits: str, legend: list[str] | None) -> list[str]:
    """`dc.improvements` is a bit string over the improvement vocabulary.

    ⚠️ Without the savegame's improvement order this cannot be decoded to names, and the
    corpus does not carry that vocabulary in `geo`. So the prototype reports the COUNT of
    set bits, which is what a viewer could tell by looking (walls/wonders are visible),
    and flags the gap rather than inventing names.
    """
    n = sum(1 for c in bits if c == "1")
    if legend is None:
        return [f"{n} known buildings"] if n else []
    return [legend[i] for i, c in enumerate(bits) if c == "1" and i < len(legend)]


def render_stale(rec: dict, viewer: str) -> tuple[str, dict]:
    """Tier-C observation for `viewer`, plus the stats that make it auditable."""
    st, geo, pr = rec["state"], rec["geo"], rec["probes"]
    X, Y = geo["xsize"], geo["ysize"]
    cb = pr["C_belief"][viewer]
    me = next(p for p in st["players"] if p["name"] == viewer)
    vis = vision_mask(me, X, Y)
    known = geo["grids"]["known"][viewer]

    L = [f"## WORLD REPORT (as {viewer} believes it)",
         f"turn {rec['t']}, year {rec['year']}",
         f"ruleset {rec['ruleset']}",   # NOTE: no world_id -- it is a trait lookup key
         ""]

    # -- own civilization: full truth, it is the viewer's own books ----------
    L += ["### MY CIVILIZATION", f"{me['name']} of the {me['nation']}",
          f"  government: {me['government']}", f"  treasury: {me['gold']} gold",
          f"  technologies ({len(me['techs'])}): " + ", ".join(sorted(me["techs"])),
          f"  cities: {len(me['cities'])}   units: {len(me['units'])}", ""]

    # -- diplomacy: only what the engine says this viewer knows --------------
    L.append("### DIPLOMATIC RELATIONS (as known to me)")
    for other, ds in (cb.get("diplstate_hidden") or {}).items():
        if other == viewer:
            continue
        L.append(f"- {other}: {ds.get('closest','?')}"
                 + ("; embassy" if ds.get("embassy") else "")
                 + (f"; first contact turn {ds['first_contact_turn']}"
                    if ds.get("first_contact_turn") else ""))
    L.append("")

    # -- ★ foreign cities from MEMORY, not from truth ------------------------
    L += ["### FOREIGN CITIES (from memory -- may be out of date)",
          "owner | city | x,y | size(remembered) | walls | capital | buildings"]
    for e in sorted(cb["dc"], key=lambda z: (z["y"], z["x"])):
        owner = owner_name(e["owner"], st["players"])   # as REMEMBERED, may be stale
        bld = improvements_from_bits(e.get("improvements", ""), None)
        L.append(f"{owner} | {e['name']} | {e['x']},{e['y']} | {e['size']} | "
                 f"{'yes' if e.get('walls') else 'no'} | {e.get('capital','Not')} | "
                 + (bld[0] if bld else "none"))
    L.append("")

    # -- foreign units: ONLY inside current vision --------------------------
    L += ["### FOREIGN UNITS IN SIGHT (this turn only; not remembered)",
          "per civ: TYPE id@x,y"]
    n_all = n_seen = 0
    for p in st["players"]:
        if p["name"] == viewer:
            continue
        got = []
        for u in p.get("units") or []:
            n_all += 1
            if (u["x"], u["y"]) in vis:
                n_seen += 1
                got.append(f"{u['type']} {u['id']}@{u['x']},{u['y']}")
        L.append(f"{p['name']}: " + ("; ".join(got) if got else "(none in sight)"))
    L.append("")

    # -- map: unexplored tiles masked (cumulative known is the right mask here,
    #    terrain does not change once seen) --------------------------------
    L += [f"### THE MAP  ({X} wide x {Y} tall)", "terrain; '?' = never explored"]
    for y, row in enumerate(geo["grids"].get("terrain") or geo.get("terrain") or []):
        kr = known[y] if y < len(known) else ""
        L.append("".join(ch if (x < len(kr) and str(kr[x]) not in "0False")
                         else "?" for x, ch in enumerate(row)))
    L.append("")

    by_id = {c["id"]: p["name"] for p in st["players"] for c in p.get("cities") or []}
    n_owner_stale = sum(1 for e in cb["dc"]
                        if e["id"] in by_id
                        and by_id[e["id"]] != owner_name(e["owner"], st["players"]))
    n_gone = sum(1 for e in cb["dc"] if e["id"] not in by_id)
    stats = {"viewer": viewer, "t": rec["t"],
             "dc_owner_stale": n_owner_stale,
             "dc_no_longer_exists": n_gone,
             "dc_cities": len(cb["dc"]),
             "dc_stale_fraction": cb["dc_staleness"]["stale_fraction"],
             "explored_fraction": cb["explored_fraction"],
             "vision_tiles": len(vis), "map_tiles": X * Y,
             "vision_fraction": round(len(vis) / (X * Y), 4),
             "foreign_units_total": n_all, "foreign_units_visible": n_seen,
             "foreign_units_hidden": n_all - n_seen}
    return "\n".join(L), stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", default="fc-civ2civ3-S-24x32-p3-hard-s100000")
    ap.add_argument("--t", type=int, default=201)
    ap.add_argument("--viewer", default=None)
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()
    rec = None
    with gzip.open(ROOT / f"corpus/{a.world}/world.jsonl.gz", "rt") as f:
        for line in f:
            d = json.loads(line)
            if d["t"] == a.t:
                rec = d
                break
    if rec is None:
        raise SystemExit(f"no record at t={a.t}")
    viewer = a.viewer or list(rec["probes"]["C_belief"])[0]
    txt, stats = render_stale(rec, viewer)
    print(json.dumps(stats, ensure_ascii=False, indent=1))
    print(f"\n报告 {len(txt)} 字符")
    if a.show:
        print("=" * 78)
        print(txt)


if __name__ == "__main__":
    main()
