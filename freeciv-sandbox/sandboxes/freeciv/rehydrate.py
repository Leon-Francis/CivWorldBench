"""Rebuild the typed objects from the stored JSON, so reports can be rendered later.

The corpus stores structured state only -- `state`, `actions`, `geo`, `events`,
`probes` -- and *not* the rendered report text.  Rendering is a pure function of
those fields, it costs 2 ms, and keeping it out of the corpus buys the thing
that actually matters: **the report format stays a training-time decision.**
Changing the renderer later is a few CPU-minutes, not a re-collection of every
world.

This module is the inverse of the `asdict` / `geo_to_json` / `actions.to_json`
serialisers.  It has to be exact, because the failure mode is silent:

    stored at collection:   Diplomat #1301 at (10, 26) heading to (9, 27)
    naive rehydration:      Diplomat #1301 at [10, 26] heading to [9, 27]

`UnitOrder.at` and `.destination` are tuples; JSON has no tuple, so they come
back as lists, and `f"{u.at}"` renders differently.  Same character count, no
exception, a different corpus.  Every tuple field therefore has to be restored
explicitly, and `--selftest` asserts byte equality over a whole world rather
than trusting that it looks right.

The grid layers need the same care in the other direction: `geo_to_json` flattens
the CSV grids to `",".join(row)` strings and the fog-of-war mask to `"0101"`
strings, so they have to be split back into the shapes `report.py` indexes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .actions import PlayerActions, UnitOrder
from .geo import MapInfo, Territory
from .state import City, Player, WorldState
from .state import base_ruleset


# --------------------------------------------------------------------------
# state
# --------------------------------------------------------------------------

def state_from_json(j: dict[str, Any]) -> WorldState:
    players = []
    for p in j.get("players", []):
        p = dict(p)
        p["cities"] = [City(**c) for c in p.get("cities", [])]
        players.append(Player(**p))
    return WorldState(
        sim=j["sim"], ruleset=base_ruleset(j["ruleset"]), world_id=j["world_id"],
        turn=j["turn"], year=j["year"], players=players,
        events=j.get("events", []),
    )


# --------------------------------------------------------------------------
# geo
# --------------------------------------------------------------------------

def _csv_grid(rows: list[str]) -> list[list[str]]:
    """`geo_to_json` stored these as `",".join(row)`; undo that."""
    return [r.split(",") for r in rows]


def _bool_grid(rows: list[str]) -> list[list[bool]]:
    return [[c == "1" for c in row] for row in rows]


def geo_from_json(j: dict[str, Any]) -> MapInfo:
    g = j["grids"]
    territory = {}
    for name, t in (j.get("territory") or {}).items():
        t = dict(t)
        # bbox and centroid are tuples on the dataclass and lists in JSON.  The
        # render path does not read them, but a half-restored object is a trap
        # for whatever reads it next.
        t["bbox"] = tuple(t["bbox"])
        t["centroid"] = tuple(t["centroid"])
        territory[name] = Territory(**t)
    return MapInfo(
        xsize=j["xsize"], ysize=j["ysize"],
        terrain=g["terrain"],
        owner=_csv_grid(g["owner"]),
        extras=g["extras"],
        worked=_csv_grid(g["worked"]),
        eowner=_csv_grid(g["eowner"]),
        source=_csv_grid(g["source"]),
        placing=_csv_grid(g["placing"]),
        infra_turns=_csv_grid(g["infra_turns"]),
        labels=j.get("labels", {}),
        known={n: _bool_grid(rows) for n, rows in (g.get("known") or {}).items()},
        terrain_names=j["terrain_legend"],
        extras_vector=j["extras_vector"],
        territory=territory,
        total_land=j.get("total_land", 0),
        world_resources=j.get("world_resources", {}),
        unclaimed_resources=j.get("unclaimed_resources", {}),
        startpos=j.get("startpos", []),
    )


# --------------------------------------------------------------------------
# actions
# --------------------------------------------------------------------------

def actions_from_json(j: dict[str, Any]) -> dict[str, PlayerActions]:
    """★ The tuple restoration here is load-bearing -- see the module docstring."""
    out: dict[str, PlayerActions] = {}
    for name, a in j.items():
        a = dict(a)
        orders = []
        for u in a.get("unit_orders", []):
            u = dict(u)
            u["at"] = tuple(u["at"])
            u["destination"] = tuple(u["destination"]) if u["destination"] is not None else None
            orders.append(UnitOrder(**u))
        a["unit_orders"] = orders
        out[name] = PlayerActions(**a)
    return out


# --------------------------------------------------------------------------
# one call site for the whole record
# --------------------------------------------------------------------------

def record_to_objects(rec: dict[str, Any],
                      events: list[dict[str, Any]] | None = None):
    """`world_step` record -> (WorldState with events attached, MapInfo, actions).

    Events are attached to the state before returning because the report's final
    section reads `ws.events`; rendering without this step silently emits an
    empty chronicle (a bug that already happened once in the collector).

    `events` overrides the record's own field and is how the merged stream from
    `event_stream.py` reaches the renderer.  The in-record field is a fallback
    for a world with no `events.jsonl.gz`: it predates the `events.py` fixes, so
    it reports a conquest as a destruction plus a founding and names diplomatic
    counterparties by savegame slot.
    """
    ws = state_from_json(rec["state"])
    ws.events = events if events is not None else rec.get("events", [])
    mi = geo_from_json(rec["geo"]) if rec.get("geo") else None
    acts = actions_from_json(rec.get("actions") or {})
    return ws, mi, acts


# --------------------------------------------------------------------------
# P-0: byte equality, asserted rather than assumed
# --------------------------------------------------------------------------

def selftest_savedir(save_dir: Path, variants=("full", "minimal")) -> dict[str, Any]:
    """Parse every savegame, render directly and via JSON, compare bytes.

    This is the gate the corpus design rests on: if a report rebuilt from the
    stored fields differs from the one the collector would have written, then
    dropping the text from the corpus loses information and we have to store it
    after all.
    """
    import dataclasses

    from .actions import extract_actions, to_json as actions_to_json
    from .geo import extract_map, geo_to_json
    from .report import render, render_actions
    from .savegame import Savegame
    from .state import extract

    saves = sorted(save_dir.glob("*.sav"))
    if not saves:
        raise SystemExit(f"no .sav files under {save_dir}")

    fails: list[str] = []
    n = 0
    for p in saves:
        sg = Savegame.load(p)
        ws = extract(sg, "selftest")
        mi = extract_map(sg, [q.name for q in ws.players])
        acts = extract_actions(sg)

        # Round-trip exactly as the corpus will: through json.dumps/loads, so
        # that any type the serialiser cannot express shows up here.
        sj = json.loads(json.dumps(dataclasses.asdict(ws), ensure_ascii=False))
        gj = json.loads(json.dumps(geo_to_json(mi), ensure_ascii=False))
        aj = json.loads(json.dumps(actions_to_json(acts), ensure_ascii=False))

        ws2, mi2 = state_from_json(sj), geo_from_json(gj)
        ws2.events = ws.events
        acts2 = actions_from_json(aj)

        for v in variants:
            view = ws.players[0].name if ws.players else None
            a = render(ws, v, view, mi)
            b = render(ws2, v, view, mi2)
            n += 1
            if a != b:
                fails.append(f"{p.name} report.{v}: {_first_diff(a, b)}")

        a = render_actions(acts)
        b = render_actions(acts2)
        n += 1
        if a != b:
            fails.append(f"{p.name} actions_text: {_first_diff(a, b)}")

    return {"checkpoints": len(saves), "comparisons": n,
            "failures": fails, "ok": not fails}


def _first_diff(a: str, b: str) -> str:
    la, lb = a.split("\n"), b.split("\n")
    for i, (x, y) in enumerate(zip(la, lb)):
        if x != y:
            return f"line {i}\n      direct={x[:100]!r}\n      rehydr={y[:100]!r}"
    if len(la) != len(lb):
        return f"line count {len(la)} vs {len(lb)}"
    return "differ but no line diff found (trailing bytes?)"


def main():
    ap = argparse.ArgumentParser(description="P-0: report round-trip byte equality")
    ap.add_argument("--world", help="world dir containing saves/ (e.g. data/freeciv/demo2/w888)")
    ap.add_argument("--saves", help="a saves directory directly")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if not args.selftest:
        ap.error("nothing to do; pass --selftest")
    if args.saves:
        save_dir = Path(args.saves)
    elif args.world:
        save_dir = Path(args.world) / "saves"
    else:
        ap.error("pass --world or --saves")

    r = selftest_savedir(save_dir)
    print(f"[P-0] {save_dir}")
    print(f"  checkpoints: {r['checkpoints']}, comparisons: {r['comparisons']}")
    if r["ok"]:
        print(f"  ALL BYTE-IDENTICAL -- corpus may store structured state only")
    else:
        print(f"  {len(r['failures'])} FAILURES:")
        for f in r["failures"][:10]:
            print(f"    {f}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
