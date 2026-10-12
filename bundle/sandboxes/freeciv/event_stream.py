"""The merged event stream: one `events.jsonl.gz` per world.

Two producers write into this one file, because neither alone is complete
(measured against freeciv 3.1.6, see the `events.py` docstring):

    src="engine"   cities, units, techs, buildings, actions, disasters,
                   achievements -- typed Lua signals, carrying the causal
                   attribution the state cannot show (who razed the city, which
                   unit attacked, why a unit died)
    src="diff"     government changes and diplomacy -- the engine has **no**
                   `government_changed` and **no** `war_declared` signal, so
                   state diffing is their only source

`world.jsonl.gz` is deliberately **not** rewritten.  Its `events` field predates
the `events.py` fixes and is superseded by this file; readers should call
`load()` and fall back to the in-record field only for a world with no
`events.jsonl.gz`.  Rewriting 88 GB of shards to correct one field would buy
nothing over a ~3 MB sidecar and would cost a full re-migration.

Diff events are a *pure function of the stored state*, exactly like the report:
the corpus keeps every turn, so they can be re-derived whenever `events.py`
changes at no simulation cost.  That is what makes correcting them a
minutes-long job rather than a re-collection.

File format, one line per turn, ascending:

    {"t": 96, "events": [{...,"src":"engine"}, {...,"src":"diff"}]}
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from .events import diff_events
from .rehydrate import state_from_json

EVENTS_FILE = "events.jsonl.gz"

# Event types the engine reports better than a state diff can, because it names
# the actor.  Once engine events exist for a turn, the diff-derived copies of
# these are dropped rather than duplicated; everything else the diff produces
# (government, diplomacy) is kept because the engine has no signal for it.
ENGINE_COVERED = frozenset({
    "city_founded", "city_destroyed", "city_transferred", "tech_discovered",
})


# --------------------------------------------------------------------------
# read / write
# --------------------------------------------------------------------------

def path_for(world_dir: Path) -> Path:
    return Path(world_dir) / EVENTS_FILE


def load(world_dir: Path) -> dict[int, list[dict[str, Any]]] | None:
    """`{turn: [event, ...]}`, or None when the world has no stream yet."""
    p = path_for(world_dir)
    if not p.exists():
        return None
    out: dict[int, list[dict[str, Any]]] = {}
    with gzip.open(p, "rt") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            out[int(r["t"])] = r.get("events") or []
    return out


def write(world_dir: Path, by_turn: dict[int, list[dict[str, Any]]]) -> Path:
    """Atomic: write beside the target and rename, so a killed job (this runs
    unattended for days) never leaves a truncated gzip that reads as an empty
    event stream."""
    p = path_for(world_dir)
    tmp = p.with_suffix(p.suffix + ".part")
    with gzip.open(tmp, "wt") as f:
        for t in sorted(by_turn):
            f.write(json.dumps({"t": t, "events": by_turn[t]},
                               ensure_ascii=False) + "\n")
    os.replace(tmp, p)
    return p


# --------------------------------------------------------------------------
# producers
# --------------------------------------------------------------------------

def derive_diff(shard: Path) -> dict[int, list[dict[str, Any]]]:
    """Re-derive the state-diff events for one world from its corpus shard.

    Streams the shard: a tier-L shard is ~1 GB decompressed and only two
    consecutive states are needed at a time.
    """
    by_turn: dict[int, list[dict[str, Any]]] = {}
    prev = None
    with gzip.open(shard, "rt") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            ws = state_from_json(rec["state"])
            t = int(rec["t"])
            if prev is not None:
                by_turn[t] = [dict(e, src="diff") for e in diff_events(prev, ws)]
            else:
                # The first record has no predecessor, so no diff exists for it.
                # Recorded as empty rather than omitted, so "no events" and "not
                # covered" stay distinguishable.
                by_turn[t] = []
            prev = ws
    return by_turn


def merge(diff_by_turn: dict[int, list[dict[str, Any]]],
          engine_by_turn: dict[int, list[dict[str, Any]]]
          ) -> dict[int, list[dict[str, Any]]]:
    """Engine events first (they carry the attribution), then the diff events
    the engine has no signal for.

    A turn with engine coverage drops its diff-derived `ENGINE_COVERED` events;
    a turn without engine coverage keeps them, so a partially harvested corpus
    degrades to the ① behaviour instead of losing city events entirely.
    """
    out: dict[int, list[dict[str, Any]]] = {}
    for t in sorted(set(diff_by_turn) | set(engine_by_turn)):
        eng = engine_by_turn.get(t) or []
        dif = diff_by_turn.get(t) or []
        if eng:
            dif = [e for e in dif if e.get("type") not in ENGINE_COVERED]
        out[t] = [dict(e, src="engine") for e in eng] + \
                 [dict(e, src="diff") for e in dif]
    return out


# --------------------------------------------------------------------------
# batch: pass ① over a whole corpus
# --------------------------------------------------------------------------

def _one(args: tuple[str, bool]) -> dict[str, Any]:
    world_dir, force = args
    d = Path(world_dir)
    shard = d / "world.jsonl.gz"
    try:
        existing = load(d) if not force else None
        engine = {}
        if existing:
            # Keep any engine events already harvested (pass ② may have run) and
            # replace only the diff half.
            for t, evs in existing.items():
                keep = [e for e in evs if e.get("src") == "engine"]
                if keep:
                    engine[t] = keep
        diff = derive_diff(shard)
        write(d, merge(diff, engine) if engine else
                 {t: [dict(e, src="diff") for e in evs] for t, evs in diff.items()})
        n = sum(len(v) for v in diff.values())
        return {"world": d.name, "ok": True, "turns": len(diff),
                "diff_events": n, "engine_turns": len(engine)}
    except Exception as e:
        return {"world": d.name, "ok": False, "detail": f"{type(e).__name__}: {e}"}


def main():
    ap = argparse.ArgumentParser(
        description="Rebuild the diff half of events.jsonl.gz across a corpus")
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 8)
    ap.add_argument("--force", action="store_true",
                    help="discard engine events already in the stream too")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    corpus = Path(args.corpus)
    # ⚠️ Same corpus lock as the collector and the harvest.  Without it, two
    # rebuilds raced here for 41 minutes: a stray `--force` run kept overwriting
    # streams with output from an older revision of this file, and `--force`
    # discards the engine half, so worlds silently regressed while a second run
    # was writing them correctly.  Every writer to the corpus takes the lock.
    from .orchestrate import take_lock

    lock = take_lock(corpus)
    try:
        _run(corpus, args)
    finally:
        if lock.exists():
            lock.unlink()


def _run(corpus: Path, args) -> None:
    dirs = sorted(d for d in corpus.iterdir()
                  if d.is_dir() and (d / "world.jsonl.gz").exists())
    if args.limit:
        dirs = dirs[:args.limit]
    print(f"[event_stream] {len(dirs)} worlds, {args.jobs} jobs")

    t0 = time.time()
    done = fail = ev = 0
    fails: list[str] = []
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        futs = [ex.submit(_one, (str(d), args.force)) for d in dirs]
        for i, fu in enumerate(as_completed(futs), 1):
            r = fu.result()
            if r["ok"]:
                done += 1
                ev += r["diff_events"]
            else:
                fail += 1
                fails.append(f"{r['world']}: {r['detail']}")
            if i % 200 == 0 or i == len(futs):
                el = time.time() - t0
                eta = el / i * (len(futs) - i)
                print(f"  {i}/{len(futs)}  ok={done} fail={fail}  "
                      f"events={ev:,}  {el:.0f}s elapsed, ETA {eta:.0f}s")

    print(f"[event_stream] ok={done} fail={fail} diff_events={ev:,} "
          f"in {time.time()-t0:.0f}s")
    for f in fails[:10]:
        print(f"  FAIL {f}")
    if fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
