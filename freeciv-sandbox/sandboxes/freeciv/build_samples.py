"""Corpus -> training samples.  This is where reports get rendered.

The corpus stores structured state; rendering happens here, which is what makes
the report format a training-time decision.  Rebuilding every sample set from
2000 worlds costs minutes of CPU, so changing the renderer is cheap and changing
the collection is not.

Four sets, one per training target:

  text-transition   render(state_t, geo_t, events_t) + render_actions(actions_t)
                    -> render(state_{t+d}, ...)
  struct-transition  state_t + actions_t -> delta[d]
  counterfactual     premise report at t_branch + cond_text -> resolved
  probe              a window of reports -> hidden probe targets

Two guards, both of which exist because the failure they catch is silent:

  * **Byte-equality spot check.**  A sampled fraction of every build re-renders
    from a second independent rehydration and asserts equality.  The known trap
    is JSON having no tuple type: `UnitOrder.at` comes back as a list and
    `f"{u.at}"` prints `[10, 26]` instead of `(10, 26)` -- same length, no error,
    different corpus.
  * **Renderer version pinning.**  `render_manifest.json` records a digest of the
    renderer sources, so a sample set can always be traced to the code that
    produced it.  (Digest rather than a git hash: this tree is not a git repo.)

Token ceilings apply here and nowhere else.  A `(t, d)` pair whose rendered
example exceeds the ceiling is dropped and counted, never silently truncated --
a truncated report is a *wrong* report, and it would teach the model that cities
stop existing past some index.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

from .rehydrate import actions_from_json, geo_from_json, record_to_objects, state_from_json
from .report import render, render_actions

SETS = ("text-transition", "struct-transition", "counterfactual", "probe")

# Probe targets worth a task.  Chosen for the three properties that make a probe
# meaningful: invisible in the report, exact ground truth, and not recoverable
# from a single frame by a shortcut.  See §1.2 of the probe design doc.
PROBE_TARGETS = {
    # episode-static personality: only inferable from a whole behaviour stream
    "trait": lambda pr, civ: (pr["A_static"].get(civ) or {}).get("traits"),
    # continuous hidden attitude; deliberately NOT a sufficient statistic for war
    "love": lambda pr, civ: {k: v.get("love")
                             for k, v in (pr["B_attitude"].get(civ) or {}).items()},
    # how wrong this civ's memory of the world is
    "dc_staleness": lambda pr, civ: (pr["C_belief"].get(civ) or {}).get("dc_staleness"),
    # the intermediate tech being researched toward the visible goal
    "research_now": lambda pr, civ: (pr["D_intent"].get(civ) or {}).get("research", {}).get("now_name"),
    # the AI's chosen wonder-building city (0 = none, itself informative)
    "wonder_city": lambda pr, civ: (pr["D_intent"].get(civ) or {}).get("wonder_city"),
    # cumulative military history, unrecoverable from any single frame
    "war_history": lambda pr, civ: {
        k: (pr["I_scoreboard"].get(civ) or {}).get(k)
        for k in ("units_built", "units_killed", "units_lost")},
}


def renderer_digest() -> dict[str, str]:
    here = Path(__file__).parent
    out = {}
    for name in ("report.py", "rehydrate.py", "build_samples.py"):
        p = here / name
        if p.exists():
            out[name] = hashlib.sha1(p.read_bytes()).hexdigest()[:12]
    return out


def iter_shards(corpus: Path, need_branch: bool = False) -> Iterator[tuple[str, Path]]:
    """World shards, optionally only those that also have branch data.

    ★ `need_branch` exists because the counterfactual set only concerns the 340
    branch worlds, yet the first version walked all 2100 -- decompressing and
    parsing the whole 89.8 GB corpus so that 1760 of those worlds could return
    nothing.  That turned a build into a 5.4-hour job and undermined the reason the
    corpus stores no rendered text: rebuilding is supposed to be cheap.
    """
    for d in sorted(corpus.iterdir()):
        shard = d / "world.jsonl.gz"
        if not (d.is_dir() and shard.exists()):
            continue
        if need_branch and not (d / "branch.jsonl.gz").exists():
            continue
        yield d.name, shard


def attach_events(world_dir: Path, recs) -> None:
    """Overwrite each record's `events` with the merged stream, in place.

    Done at the single point where records enter the pipeline so that every
    downstream reader (`_report_of`, `ws_of`, `_spotcheck`) keeps working
    unchanged.  The in-shard field is only used for a world with no
    `events.jsonl.gz`: it predates the `events.py` fixes, so it reports a
    conquest as a destruction plus a founding, names diplomatic counterparties
    by savegame slot, and has no engine attribution at all.
    """
    from .event_stream import load as load_events

    by_turn = load_events(world_dir)
    if by_turn is None:
        return
    it = recs.values() if isinstance(recs, dict) else recs
    for r in it:
        r["events"] = by_turn.get(int(r["t"]), [])


def load_records(shard: Path) -> list[dict]:
    with gzip.open(shard, "rt") as f:
        recs = [json.loads(line) for line in f]
    attach_events(shard.parent, recs)
    return recs


def load_records_at(shard: Path, turns: set[int]) -> dict[int, dict]:
    """Only the records at `turns`.  A tier-L shard is ~1 GB decompressed and the
    counterfactual set needs 4-6 of its ~1200 records."""
    out: dict[int, dict] = {}
    with gzip.open(shard, "rt") as f:
        for line in f:
            # Cheap pre-filter: parse fully only when the turn is wanted.  `"t":`
            # appears once per record as a top-level key.
            i = line.find('"t":')
            if i < 0:
                continue
            j = line.find(",", i)
            try:
                t = int(line[i + 4:j])
            except ValueError:
                continue
            if t in turns:
                out[t] = json.loads(line)
    attach_events(shard.parent, out)
    return out


def _report_of(rec: dict, variant: str = "full") -> str:
    ws, mi, _ = record_to_objects(rec)
    return render(ws, variant, ws.players[0].name if ws.players else None, mi)


def _actions_of(rec: dict) -> str:
    return render_actions(actions_from_json(rec.get("actions") or {}))


def _spotcheck(rec: dict, variant: str) -> str | None:
    """Independent second rehydration; returns a message on mismatch."""
    a = _report_of(rec, variant)
    ws = state_from_json(json.loads(json.dumps(rec["state"])))
    ws.events = rec.get("events", [])
    mi = geo_from_json(json.loads(json.dumps(rec["geo"])))
    b = render(ws, variant, ws.players[0].name if ws.players else None, mi)
    if a != b:
        return f"report mismatch at t={rec['t']}"
    if _actions_of(rec) != render_actions(
            actions_from_json(json.loads(json.dumps(rec["actions"] or {})))):
        return f"actions mismatch at t={rec['t']}"
    return None


# --------------------------------------------------------------------------
# the four builders
# --------------------------------------------------------------------------

def build_text_transition(recs: list[dict], deltas: list[int], max_chars: int,
                          variant: str, with_path: bool) -> tuple[list[dict], Counter]:
    by_t = {r["t"]: i for i, r in enumerate(recs)}
    cache: dict[int, str] = {}

    def rep(i: int) -> str:
        if i not in cache:
            cache[i] = _report_of(recs[i], variant)
        return cache[i]

    out, stats = [], Counter()
    for i, r in enumerate(recs):
        for d in deltas:
            j = by_t.get(r["t"] + d)
            if j is None:
                stats["no_pair"] += 1
                continue
            inp, act, tgt = rep(i), _actions_of(r), rep(j)
            n = len(inp) + len(act) + len(tgt)
            if n > max_chars:
                stats["over_ceiling"] += 1
                continue
            sample = {
                "world_id": r["world_id"], "split": r["split"], "t": r["t"],
                "delta": d, "next_t": recs[j]["t"], "variant": variant,
                "report_t": inp, "actions_t": act, "report_next": tgt,
                "chars": n,
            }
            if with_path and d > 1:
                # For a long stride the turn-t decisions do not describe the whole
                # interval.  Every intervening turn is in the corpus, so the path
                # is assembled here rather than duplicated into each record.
                sample["actions_path"] = [
                    _actions_of(recs[by_t[t]])
                    for t in range(r["t"] + 1, recs[j]["t"]) if t in by_t]
            out.append(sample)
            stats["kept"] += 1
    return out, stats


def build_struct_transition(recs: list[dict], deltas: list[int],
                            with_path: bool = False) -> tuple[list[dict], Counter]:
    """`delta` is computed here, not read from the corpus.

    It is a pure function of two stored states, so baking a fixed set of strides
    into the corpus would only freeze a training decision into it.  Computing on
    demand means a new stride costs a rebuild, not a re-collection.
    """
    from .state import delta as state_delta

    by_t = {r["t"]: r for r in recs}
    cache: dict[int, Any] = {}

    def ws_of(t: int):
        if t not in cache:
            w = state_from_json(by_t[t]["state"])
            w.events = by_t[t].get("events", [])
            cache[t] = w
        return cache[t]

    out, stats = [], Counter()
    for r in recs:
        for d in deltas:
            nxt = by_t.get(r["t"] + d)
            if nxt is None:
                stats["no_pair"] += 1
                continue
            sample = {
                "world_id": r["world_id"], "split": r["split"], "t": r["t"],
                "delta": d, "next_t": nxt["t"],
                "state_t": r["state"], "actions_t": r["actions"],
                "target_delta": state_delta(ws_of(r["t"]), ws_of(nxt["t"])),
            }
            if with_path and d > 1:
                # Every intervening turn's decisions, which the corpus has because
                # it checkpoints every turn.  Without these a long-stride example is
                # conditioned on a fraction of what actually drove the interval.
                sample["actions_path"] = [by_t[t]["actions"]
                                          for t in range(r["t"] + 1, nxt["t"])
                                          if t in by_t]
            out.append(sample)
            stats["kept"] += 1
    return out, stats


def build_counterfactual(world_dir: Path, recs: list[dict] | None, max_chars: int,
                         variant: str) -> tuple[list[dict], Counter]:
    """One sample per (branch point, arm).

    The premise report is rendered from the mainline record at `t_branch`, so it
    is literally the same report the factual side of the pair sees -- storing a
    copy in the branch group could only ever introduce a discrepancy.
    """
    bp = world_dir / "branch.jsonl.gz"
    out, stats = [], Counter()
    if not bp.exists():
        return out, stats
    # Two passes over the small branch file instead of holding the whole corpus
    # shard: first collect which mainline turns are needed, then read only those.
    want = set()
    with gzip.open(bp, "rt") as f:
        for line in f:
            want.add(json.loads(line)["premise"]["from_world_step"])
    by_t = (load_records_at(world_dir / "world.jsonl.gz", want)
            if recs is None else {r["t"]: r for r in recs})
    with gzip.open(bp, "rt") as f:
        for line in f:
            g = json.loads(line)
            rec = by_t.get(g["t_branch"])
            if rec is None:
                stats["no_premise_record"] += 1
                continue
            premise = _report_of(rec, variant)
            actions = _actions_of(rec)
            for arm in g["arms"]:
                if arm.get("status") != "OK":
                    stats[f"arm_{arm.get('status','?')}"] += 1
                    continue
                n = len(premise) + len(actions) + len(arm.get("cond_text", ""))
                if n > max_chars:
                    stats["over_ceiling"] += 1
                    continue
                out.append({
                    "world_id": g["world_id"], "split": rec["split"],
                    "t_branch": g["t_branch"], "horizons": g["horizons"],
                    "report_t": premise, "actions_t": actions,
                    "arm": arm["arm"], "layer": arm.get("layer"),
                    "placebo_of": arm.get("placebo_of"),
                    "placebo_kind": arm.get("placebo_kind"),
                    "cond_text": arm.get("cond_text"),
                    "expected_path": (arm.get("op") or {}).get("expected_path"),
                    "horizon_band": (arm.get("op") or {}).get("horizon_band"),
                    "target": (arm.get("op") or {}).get("target"),
                    "resolved": arm.get("resolved"),
                    "factual": g.get("factual"),
                })
                stats["kept"] += 1
    return out, stats


def build_probe(recs: list[dict], window: int, stride: int, max_chars: int,
                variant: str) -> tuple[list[dict], Counter]:
    """A window of consecutive reports for one civ -> that civ's hidden state.

    A window rather than a single frame because the most valuable target
    (`trait`) is episode-static: it can only be read off a behaviour stream, and
    a one-frame probe would be measuring something else entirely.
    """
    out, stats = [], Counter()
    for i in range(0, max(0, len(recs) - window + 1), stride):
        chunk = recs[i:i + window]
        reports = [_report_of(r, variant) for r in chunk]
        n = sum(len(x) for x in reports)
        if n > max_chars:
            stats["over_ceiling"] += 1
            continue
        last = chunk[-1]
        for civ in [p["name"] for p in last["state"]["players"]]:
            targets = {}
            for key, fn in PROBE_TARGETS.items():
                try:
                    targets[key] = fn(last["probes"], civ)
                except Exception:
                    targets[key] = None
            out.append({
                "world_id": last["world_id"], "split": last["split"],
                "t_from": chunk[0]["t"], "t_to": last["t"], "civ": civ,
                "reports": reports, "targets": targets, "chars": n,
            })
            stats["kept"] += 1
    return out, stats


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def build(corpus: Path, outdir: Path, setname: str, *, deltas: list[int],
          max_chars: int, variant: str, spotcheck_rate: float,
          with_path: bool, window: int, stride: int) -> dict[str, Any]:
    outdir.mkdir(parents=True, exist_ok=True)
    writers: dict[str, Any] = {}
    stats: Counter = Counter()
    mismatches: list[str] = []
    rng = random.Random(0)
    t0 = time.time()
    n_worlds = 0

    # The counterfactual set touches only branch worlds, and needs a handful of
    # records from each -- so it neither walks every world nor loads whole shards.
    cf = setname == "counterfactual"
    try:
        for world_id, shard in iter_shards(corpus, need_branch=cf):
            recs = None if cf else load_records(shard)
            if not cf and not recs:
                continue
            n_worlds += 1
            for r in (recs or []):
                if rng.random() < spotcheck_rate:
                    m = _spotcheck(r, variant)
                    stats["spotchecks"] += 1
                    if m:
                        mismatches.append(f"{world_id}: {m}")

            if setname == "text-transition":
                samples, s = build_text_transition(recs, deltas, max_chars,
                                                   variant, with_path)
            elif setname == "struct-transition":
                samples, s = build_struct_transition(recs, deltas, with_path)
            elif setname == "counterfactual":
                samples, s = build_counterfactual(shard.parent, recs, max_chars,
                                                  variant)
            elif setname == "probe":
                samples, s = build_probe(recs, window, stride, max_chars, variant)
            else:
                raise ValueError(setname)
            stats.update(s)

            for sm in samples:
                sp = sm.get("split", "train")
                if sp not in writers:
                    writers[sp] = gzip.open(outdir / f"{sp}.jsonl.gz", "wt")
                writers[sp].write(json.dumps(sm, ensure_ascii=False) + "\n")
    finally:
        for w in writers.values():
            w.close()

    manifest = {
        "set": setname, "corpus": str(corpus), "worlds": n_worlds,
        "deltas": deltas, "max_chars": max_chars, "variant": variant,
        "window": window, "stride": stride, "with_actions_path": with_path,
        "renderer": renderer_digest(),
        "stats": dict(stats),
        "spotcheck_mismatches": mismatches[:20],
        "byte_equality_ok": not mismatches,
        "seconds": round(time.time() - t0, 1),
        "files": {sp: (outdir / f"{sp}.jsonl.gz").stat().st_size
                  for sp in writers if (outdir / f"{sp}.jsonl.gz").exists()},
    }
    (outdir / "render_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1))
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", default=None, help="default <corpus>/../samples")
    ap.add_argument("--set", dest="setname", default="text-transition",
                    choices=list(SETS) + ["all"])
    ap.add_argument("--deltas", default="1,5,20,50,100")
    ap.add_argument("--max_chars", type=int, default=400000)
    ap.add_argument("--variant", default="full")
    ap.add_argument("--spotcheck_rate", type=float, default=0.05)
    ap.add_argument("--actions_path", action="store_true")
    ap.add_argument("--window", type=int, default=6)
    ap.add_argument("--stride", type=int, default=6)
    args = ap.parse_args()

    corpus = Path(args.corpus)
    base = Path(args.out) if args.out else corpus.parent / "samples"
    deltas = [int(x) for x in args.deltas.split(",") if x]
    names = list(SETS) if args.setname == "all" else [args.setname]

    for name in names:
        m = build(corpus, base / name, name, deltas=deltas,
                  max_chars=args.max_chars, variant=args.variant,
                  spotcheck_rate=args.spotcheck_rate,
                  with_path=args.actions_path, window=args.window,
                  stride=args.stride)
        tot = sum(m["files"].values())
        print(f"[{name}] worlds={m['worlds']} kept={m['stats'].get('kept',0)} "
              f"bytes={tot/1e6:.1f}MB {m['seconds']}s "
              f"byte_equality={'OK' if m['byte_equality_ok'] else 'FAILED'}")
        for k, v in sorted(m["stats"].items()):
            if k != "kept":
                print(f"    {k}: {v}")
        if not m["byte_equality_ok"]:
            for x in m["spotcheck_mismatches"]:
                print(f"    MISMATCH {x}")
            raise SystemExit(1)


if __name__ == "__main__":
    main()
