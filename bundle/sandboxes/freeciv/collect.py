"""Mainline collection: one world -> structured world_step records.

What this writes, per world:

    <out>/<world_id>/world.jsonl.gz   one record per checkpoint up to report_endturn
    <out>/<world_id>/factual.json     per-civ scalars + digest at EVERY checkpoint
    <out>/<world_id>/ckpt/*.sav       only the savegames branch arms fork from
    <out>/<world_id>/meta.json        config, actual map size, timings, counts

Four decisions worth stating, because each replaces something the earlier version
did differently.

**No rendered text.**  `reports` and `actions_text` are gone.  Rendering is a
pure function of `state` + `geo` + `events`, it costs 2 ms, and
`rehydrate.selftest_savedir` verifies the rebuild is byte-identical over a whole
world.  Keeping the text out of the corpus is worth 45% of the bytes, but the
real reason is that it makes the report format a training-time decision instead
of something frozen into 2000 worlds.

**The mainline runs past the reported window.**  Records stop at
`report_endturn`; the rollout continues to `max_horizon`.  Those extra turns are
the factual arm every counterfactual branch is measured against, so it cannot
disagree with the mainline -- it *is* the mainline.

**The corpus is stride-complete, and stride is a training-time choice.**  One
checkpoint per turn, each carrying its full `state` / `actions` / `geo` /
`events`.  So any stride is reconstructible: `report_t`, `actions_t` and the full
action path over any interval all come out of records that are already there.

`delta` is therefore **not** stored at a few baked-in strides any more.  It is a
pure function of two stored states, so `build_samples` computes it for whatever
stride is asked for.  Baking in {5, 20, 50} would have frozen a training decision
into the corpus for no benefit -- and at every-5-turns spacing the intervening
turns' `actions` were being dropped outright, which made the corpus incomplete at
*every* stride, not just the long ones (unit orders and production choices change
turn by turn).

The stride ladder is real and worth having: measured on 132 worlds, predicting
"same as last checkpoint" leaves a persistence error of 0.41% of climatology at
stride 5, 4.74% at 20, and 27.79% at 50.
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import shutil
import time
from pathlib import Path
from typing import Any

from .actions import extract_actions, to_json as actions_to_json
from .branch import resolve_state
from .events import diff_events
from .geo import extract_map, geo_to_json
from .probes import extract_probes
from .runner import run
from .savegame import Savegame
from .state import extract
from .worlds import WorldCfg


def _state_to_j(ws):
    import dataclasses
    return dataclasses.asdict(ws)


def collect_world(cfg: WorldCfg, out_root: Path, *, branch_points: int = 0,
                  keep_saves: bool = False,
                  branch_turns: tuple[int, ...] = ()) -> dict[str, Any]:
    """Run and parse one world.  Returns a summary dict."""
    from .branch import choose_points

    wid_req = cfg.world_id()
    wdir = out_root / wid_req
    work = wdir / "work"
    shutil.rmtree(work, ignore_errors=True)
    wdir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    r = run(work, cfg.settings(), commands=cfg.commands(), timeout=21600)
    sim_s = time.time() - t0
    if not r.ok:
        return {"world_id": wid_req, "ok": False, "detail": r.message}

    # The id carries the dimensions the server actually built, which is not
    # always what was asked for -- `mapsize` defaults to PLAYER sizing under
    # civ2civ3 and silently ignores an explicit size.
    first = Savegame.load(r.saves[0])
    xs, ys = int(first.get("map", "xsize", -1)), int(first.get("map", "ysize", -1))
    wid = cfg.world_id((xs, ys)) if xs > 0 else wid_req
    if wid != wid_req:
        new = out_root / wid
        new.mkdir(parents=True, exist_ok=True)
        work = new / "work"
        shutil.move(str(wdir / "work"), str(work))
        wdir = new

    t1 = time.time()
    records: list[dict[str, Any]] = []
    states: list[Any] = []
    factual: dict[str, Any] = {}
    digests: dict[str, str] = {}
    ckpt_by_turn: dict[int, Path] = {}

    prev_ws = None
    for p in r.saves:
        sg = Savegame.load(p)
        t = sg.turn
        ws = extract(sg, wid)
        ev = diff_events(prev_ws, ws) if prev_ws else []
        # Events must be attached before anything reads the state: the report's
        # final section is driven off `ws.events`, and a renderer run before this
        # emits an empty chronicle without complaining.
        ws.events = ev
        prev_ws = ws

        # Cheap enough to do at every checkpoint, including the post-report ones:
        # this is the factual arm.
        factual[str(t)] = resolve_state(sg, wid)
        digests[str(t)] = sg.state_digest()

        if t > cfg.report_endturn:
            continue

        mi = extract_map(sg, [q.name for q in ws.players])
        records.append({
            "world_id": wid,
            "sim": "freeciv-3.1.6",
            "ruleset": sg.ruleset,
            "split": cfg.split,
            "t": t,
            "year": sg.year,
            "state": _state_to_j(ws),
            "actions": actions_to_json(extract_actions(sg)),
            "geo": geo_to_json(mi),
            "events": ev,
            # Hidden state.  Never rendered into any report -- a probe target
            # that appears in the input is not a probe target.
            "probes": extract_probes(sg, mi),
        })
        states.append(ws)
        ckpt_by_turn[t] = p

    parse_s = time.time() - t1

    shard = wdir / "world.jsonl.gz"
    with gzip.open(shard, "wt") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    (wdir / "factual.json").write_text(json.dumps(
        {"world_id": wid, "resolved": factual, "digests": digests},
        ensure_ascii=False))

    # Retain only the savegames branch arms will fork from; a world's saves are
    # ~70 MB of text and 150 concurrent workers would otherwise fill the disk.
    kept: list[str] = []
    if branch_points > 0:
        # Explicit turns win.  Some operators are only applicable in a specific
        # regime -- fog needs a civ that has not finished exploring, barbarians
        # need the ruleset's onset turn to have passed -- so a verification run has
        # to be able to place its branch points rather than take them evenly.
        if branch_turns:
            points = [t for t in branch_turns if t in ckpt_by_turn]
        else:
            points = choose_points(list(ckpt_by_turn), n_points=branch_points,
                                   report_endturn=cfg.report_endturn,
                                   max_horizon=cfg.max_horizon)
        ck = wdir / "ckpt"
        ck.mkdir(exist_ok=True)
        for bt in points:
            dst = ck / f"T{bt:04d}.sav"
            shutil.copy(ckpt_by_turn[bt], dst)
            kept.append(str(dst))
    if not keep_saves:
        shutil.rmtree(work, ignore_errors=True)

    meta = {
        "world_id": wid, "requested_id": wid_req, "config": cfg.to_json(),
        "xsize": xs, "ysize": ys,
        "turn_reached": r.turn_reached,
        "checkpoints_total": len(r.saves), "records": len(records),
        "sim_seconds": round(sim_s, 1), "parse_seconds": round(parse_s, 1),
        "shard_bytes": shard.stat().st_size,
        "branch_checkpoints": kept,
        "events_total": sum(len(rec["events"]) for rec in records),
    }
    (wdir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    return {"world_id": wid, "ok": True, **meta}


# --------------------------------------------------------------------------
# retrofit: harvest the engine event stream onto an already-collected world
# --------------------------------------------------------------------------

def harvest_world(world_dir: Path, *, restore_ckpt: bool = False,
                  timeout: int = 21600) -> dict[str, Any]:
    """Re-run one already-collected world with the Lua hooks attached.

    Deliberately separate from `collect_world`: none of the state parsing is
    needed here (the states are already in the shard and a re-run reproduces them
    byte-identically), and the collector is the one piece of this pipeline that
    2,100 worlds already depend on -- adding a mode to it would put a working
    collector at risk for no gain.

    The config comes from the world's own `meta.json`, and branch turns come from
    its recorded `branch_checkpoints` rather than from `choose_points`: re-deriving
    the points would risk choosing different ones, and the arms have to fork from
    exactly the turns they originally forked from.

    `restore_ckpt` re-creates `ckpt/*.sav`, which `orchestrate` deleted after the
    original pass 3.  That forces `save_every=1` for the run, because branch turns
    are not on any period -- so only the 340 branch worlds pay the write cost.
    """
    from .engine_events import hook_lua, parse_log
    from .event_stream import derive_diff, load as load_events, merge, write as write_events
    from .worlds import cfg_from_json

    wdir = Path(world_dir)
    meta_p = wdir / "meta.json"
    if not meta_p.exists():
        return {"world_id": wdir.name, "ok": False, "detail": "no meta.json"}
    meta = json.loads(meta_p.read_text())
    cfg = cfg_from_json(meta["config"])

    # Branch turns as originally chosen, e.g. ".../ckpt/T0021.sav" -> 21.
    branch_turns = sorted(int(Path(c).stem[1:]) for c in meta.get("branch_checkpoints") or [])
    if restore_ckpt and not branch_turns:
        return {"world_id": wdir.name, "ok": False,
                "detail": "restore_ckpt asked for but meta records no branch checkpoints"}

    settings = cfg.settings()
    # Writing savegames is pure cost here unless the checkpoints are wanted:
    # the states are already in the corpus.  One periodic save still has to be
    # produced, because `run()` reads the turn actually reached off a savegame.
    settings["saveturns"] = 1 if restore_ckpt else max(1, cfg.report_endturn)

    work = wdir / "work_harvest"
    shutil.rmtree(work, ignore_errors=True)
    t0 = time.time()
    r = run(work, settings, commands=cfg.commands(), lua=hook_lua(), timeout=timeout)
    sim_s = time.time() - t0
    if not r.ok:
        shutil.rmtree(work, ignore_errors=True)
        return {"world_id": wdir.name, "ok": False, "detail": r.message}

    engine, attached = parse_log(r.log)
    if not attached:
        # `lua cmd` swallows syntax errors without output, so a harvest with no
        # events is ambiguous unless the hooks announce themselves.  Refuse to
        # write, rather than record an empty stream as a success.
        shutil.rmtree(work, ignore_errors=True)
        return {"world_id": wdir.name, "ok": False,
                "detail": "hook self-test line absent: Lua hooks never attached"}

    kept: list[str] = []
    if restore_ckpt:
        by_turn = {}
        for sp in r.saves:
            m = re.search(r"T(\d+)", sp.name)
            if m:
                by_turn[int(m.group(1))] = sp
        missing = [t for t in branch_turns if t not in by_turn]
        if missing:
            shutil.rmtree(work, ignore_errors=True)
            return {"world_id": wdir.name, "ok": False,
                    "detail": f"re-run produced no save at branch turns {missing}"}
        ck = wdir / "ckpt"
        ck.mkdir(exist_ok=True)
        for bt in branch_turns:
            dst = ck / f"T{bt:04d}.sav"
            shutil.copy(by_turn[bt], dst)
            kept.append(str(dst))

    shutil.rmtree(work, ignore_errors=True)

    # Keep the diff half already on disk if pass ① ran; otherwise derive it now
    # so the world ends up with a complete stream either way.
    existing = load_events(wdir)
    if existing:
        diff = {t: [e for e in evs if e.get("src") != "engine"]
                for t, evs in existing.items()}
    else:
        diff = derive_diff(wdir / "world.jsonl.gz")
    write_events(wdir, merge(diff, engine))

    n = sum(len(v) for v in engine.values())
    return {"world_id": wdir.name, "ok": True, "engine_events": n,
            "turns": len(engine), "ckpt_restored": len(kept),
            "sim_seconds": round(sim_s, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="M")
    ap.add_argument("--seed", type=int, default=90001)
    ap.add_argument("--ruleset", default="civ2civ3")
    ap.add_argument("--ailevel", default="hard")
    ap.add_argument("--report_endturn", type=int, default=0,
                    help="0 = the tier's measured budget")
    ap.add_argument("--max_horizon", type=int, default=0)
    ap.add_argument("--save_every", type=int, default=1)
    ap.add_argument("--branch_points", type=int, default=0)
    ap.add_argument("--keep_saves", action="store_true")
    ap.add_argument("--out", default="data_v2/bench")
    args = ap.parse_args()

    from .worlds import TIER_BUDGET
    re_, mh = TIER_BUDGET[args.tier]
    cfg = WorldCfg(tier=args.tier, seed=args.seed, ruleset=args.ruleset,
                   ailevel=args.ailevel,
                   report_endturn=args.report_endturn or re_,
                   max_horizon=args.max_horizon or mh,
                   save_every=args.save_every)
    res = collect_world(cfg, Path(args.out), branch_points=args.branch_points,
                        keep_saves=args.keep_saves)
    if not res.get("ok"):
        raise SystemExit(f"FAILED: {res.get('detail')}")
    print(json.dumps({k: v for k, v in res.items() if k != "branch_checkpoints"},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
