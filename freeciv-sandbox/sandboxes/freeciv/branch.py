"""Counterfactual branches: one arm per process, resolved variables only.

A branch is a fork of a mainline checkpoint with one operator applied, rolled
forward and read at a set of horizons.  Three decisions shape this module.

**The factual arm IS run, from the same checkpoint as every other arm.**  It was
originally taken from the mainline's own later checkpoints -- the mainline runs
past the reported window, so the numbers were already there for free.  The empty
arm's self-check refuted that, and the measurement is worth stating:

    reload A  vs  reload B          SAME   (byte-identical at every turn)
    reload    vs  continuous run    DIFF   (already at the first checkpoint
                                            5 turns after the fork)

`scripts/freeciv_determinism_check.sh` only ever compared reload against reload,
so "DETERMINISTIC" meant paired branches are exactly comparable -- which is true
and is what the design needs -- but never that a reload reproduces the
uninterrupted trajectory.  It does not: loading perturbs the RNG state
immediately (`[random] index_J/K/X` and all eight tables differ, and map extras
diverge within five turns).

So a factual arm read off the mainline would differ from its own intervention
arms for a reason that has nothing to do with the intervention, and every effect
size would be contaminated by it.  Each branch point therefore runs its own
factual arm through exactly the same reload path.  Cost is ~3% more arms.

**Only `resolved` is stored, never reports or full state.**  A branch group with
40 arms x 8 horizons would otherwise dominate the corpus, and the branch point's
report is recoverable by rendering the mainline record at `t_branch`.

**Arms are flat jobs.**  The previous implementation opened a
`ProcessPoolExecutor` inside a worker that was itself in a pool, so the real
concurrency was the product of two limits and nobody was tracking it.  Here one
arm is one job and the orchestrator owns the single pool.
"""

from __future__ import annotations

import gzip
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .geo import extract_map
from .interventions import (BY_ID, EMPTY, LIBRARY, Ctx, apply_op, apply_settings, make_ctx,
                            op_to_json, replay_op)
from .runner import cleanup, run_to_turn
from .savegame import Savegame
from .state import extract

# What a forecaster can be asked about at a horizon.  Deliberately a small set
# of civ-level scalars: `resolved` is multiplied by arms x horizons x worlds, so
# anything stored here is stored tens of thousands of times.
# What a forecaster can be asked about at a horizon.  Kept small because it is
# multiplied by arms x horizons x worlds -- but two additions earned their place:
# the pilot could not score `fog` or `barbarians` correctly without them, and
# variables cannot be added retroactively (re-scoring means re-simulating).
#
#   units_lost / units_killed / units_built  cumulative war history.  `n_units`
#       is the wrong reading for an external shock: the AI rebuilds, so the live
#       count converges while losses do not.  Measured: barbarians scored 0.48 on
#       n_units, which says nothing about whether the shock landed.
#   explored  the fog-of-war operator's own mechanism variable.  Scored on
#       n_cities it came out at 0.72 -- exploration to expansion is a long chain
#       and the placebo matches it.
RESOLVED_VARS = ["gold", "population", "score", "culture", "government",
                 "n_cities", "n_techs", "n_units", "at_war",
                 "units_lost", "units_killed", "units_built",
                 "n_wonders", "explored", "alive",
                 # v2 (see resolve_state)
                 "n_enemies", "n_allies", "n_peace", "n_ceasefire", "n_contacted",
                 "military_units", "civilian_units", "total_city_size",
                 "max_city_size", "n_improvements", "n_wonders_lost",
                 "bulbs", "research_goal", "research_now",
                 "tax_pct", "science_pct", "luxury_pct",
                 "trade", "production", "science_output",
                 "pairs"]          # dict: other civ -> diplstate/embassy/shared_vision/love/countdown/war_reason

HORIZONS = [30, 60, 90, 120, 150, 200, 250, 300]
BRANCH_MIN_TURN = 60          # v2, see choose_points


def resolve_state(sg: Savegame, world_id: str) -> dict[str, dict[str, Any]]:
    """Per-civ scalars at one turn, plus the pairwise diplomacy/love matrices.

    v2 lists everything the benchmark can be asked about (09 §1.1, §5.2, §8.5):
    the v1 fourteen, the 1a variable set (military/civilian units, total city
    size, allies/peace/enemies), research progress, rates, and -- keyed under
    `pairs` -- this civ's `diplstate` and `love` toward every other civ.  The
    pair matrices are what `L4-love` is scored on and what every cooperation
    counterfactual needs; v1 stored only `at_war`, and 81% of L4-love arms then
    showed no effect because a count cannot see one relation flipping.

    Adding a variable here is free at collection time and impossible afterwards
    (re-scoring means re-simulating every arm).

    The fog-of-war fraction needs the map decoded (~10 ms); worth it for L3-fog.
    """
    ws = extract(sg, world_id)
    explored = {}
    try:
        from .geo import explored_fraction, extract_map
        mi = extract_map(sg, [p.name for p in ws.players])
        explored = {p.name: round(explored_fraction(mi, p.name), 4)
                    for p in ws.players}
    except Exception:
        pass
    names = [p.name for p in ws.players]
    secs = sg.players()
    out = {}
    for i, p in enumerate(ws.players):
        sc = f"score{i}"
        sec = secs[i] if i < len(secs) else f"player{i}"
        # --- pairwise: row j of [playerN].diplstate / .ai is N's stance toward j ---
        pairs: dict[str, dict[str, Any]] = {}
        d_tbl, ai_tbl = sg.table(sec, "diplstate"), sg.table(sec, "ai")
        for j, other in enumerate(names):
            if j == i:
                continue
            row: dict[str, Any] = {}
            if d_tbl is not None and j < len(d_tbl.rows):
                dr = dict(zip(d_tbl.columns, d_tbl.rows[j]))
                row["diplstate"] = str(dr.get("current", ""))
                row["embassy"] = bool(dr.get("embassy", False))
                row["shared_vision"] = bool(dr.get("gives_shared_vision", False))
            if ai_tbl is not None and j < len(ai_tbl.rows):
                ar = dict(zip(ai_tbl.columns, ai_tbl.rows[j]))
                row["love"] = int(ar.get("love", 0) or 0)
                row["countdown"] = int(ar.get("countdown", -1) or -1)
                row["war_reason"] = int(ar.get("war_reason", 5) or 5)
            if row:
                pairs[other] = row
        rel = [v.get("diplstate", "") for v in pairs.values()]
        sizes = [c.size for c in p.cities]
        n_units = len(p.units)
        out[p.name] = {
            # v1 fourteen (unchanged semantics)
            "gold": p.gold, "population": p.population, "score": p.score,
            "culture": p.culture, "government": p.government,
            "n_cities": len(p.cities), "n_techs": len(p.techs),
            "n_units": n_units,
            "at_war": sum(1 for v in rel if v == "War"),
            "units_lost": int(sg.get(sc, "units_lost", 0) or 0),
            "units_killed": int(sg.get(sc, "units_killed", 0) or 0),
            "units_built": int(sg.get(sc, "units_built", 0) or 0),
            "n_wonders": len(p.wonders),
            "explored": explored.get(p.name),
            "alive": p.is_alive,
            # v2 additions
            "n_enemies": sum(1 for v in rel if v == "War"),
            "n_allies": sum(1 for v in rel if v == "Alliance"),
            "n_peace": sum(1 for v in rel if v in ("Peace", "Armistice")),
            "n_ceasefire": sum(1 for v in rel if v == "Cease-fire"),
            "n_contacted": sum(1 for v in rel if v and v != "Never met"),
            "military_units": int(p.military_units),
            "civilian_units": max(0, n_units - int(p.military_units)),
            "total_city_size": sum(sizes), "max_city_size": max(sizes) if sizes else 0,
            "n_improvements": sum(sum(1 for v in c.improvements.values() if v)
                                  for c in p.cities),
            "n_wonders_lost": len(p.wonders_lost),
            "bulbs": int(p.bulbs_last), "research_goal": p.research,
            "research_now": getattr(p, "research_now", ""),
            "tax_pct": p.income.get("tax_pct", 0),
            "science_pct": p.income.get("science_pct", 0),
            "luxury_pct": p.income.get("luxury_pct", 0),
            "trade": p.income.get("trade", 0), "production": p.income.get("production", 0),
            "science_output": p.income.get("science_output", 0),
            "pairs": pairs,
        }
    return out


def _nearest(by_turn: dict[int, Any], target: int, tol: int):
    """Nearest recorded turn within `tol`, or None.

    Returning None rather than the closest available value matters: a horizon
    resolved from a turn 40 away is not that horizon, and silently substituting
    it would put a shorter branch's numbers into a longer branch's column.
    """
    if not by_turn:
        return None, None
    t = min(by_turn, key=lambda k: abs(k - target))
    return (t, by_turn[t]) if abs(t - target) <= tol else (None, None)


@dataclass
class ArmJob:
    world_id: str
    world_dir: str
    t_branch: int
    op_id: str                # "" = no intervention (factual / its duplicate)
    checkpoint: str
    end_turn: int
    save_every: int = 10
    # Distinct from `op_id` because two arms apply the same (empty) operator and
    # must not collide on disk: `factual` is the control every other arm is
    # measured against, `factual_check` is a second identical run whose only job
    # is to assert reload-vs-reload determinism still holds for this world.
    arm_name: str = ""

    def name(self) -> str:
        return self.arm_name or self.op_id or "factual"

    def out_path(self) -> Path:
        return Path(self.world_dir) / "arms" / f"t{self.t_branch}_{self.name()}.json"


def run_arm_job(job: ArmJob) -> dict[str, Any]:
    """Fork, intervene, roll out, resolve, delete the savegames.

    Idempotent: an existing output file means this arm is done, which is what
    makes `--resume` work after a crash without re-simulating everything.
    """
    out = job.out_path()
    if out.exists():
        return {"arm": job.name(), "t_branch": job.t_branch, "status": "CACHED"}
    out.parent.mkdir(parents=True, exist_ok=True)

    wd = Path(job.world_dir) / "work" / f"t{job.t_branch}_{job.name()}"
    shutil.rmtree(wd, ignore_errors=True)
    wd.mkdir(parents=True, exist_ok=True)

    ckpt = Path(job.checkpoint)
    sg = Savegame.load(ckpt)
    ws = extract(sg, job.world_id)
    mi = extract_map(sg, [p.name for p in ws.players])
    ctx = make_ctx(sg, ws, mi)

    op = BY_ID[job.op_id] if job.op_id else EMPTY
    desc, lua = "", []
    try:
        desc, lua = apply_op(op, sg, ctx)
    except Exception as e:
        rec = {"arm": job.name(), "t_branch": job.t_branch, "status": "FAIL",
               "detail": f"apply raised: {e}"}
        out.write_text(json.dumps(rec, ensure_ascii=False))
        return rec
    if not desc:
        # Not applicable to this checkpoint.  Recorded rather than silently
        # dropped, so coverage statistics can tell "operator never ran here"
        # from "operator ran and did nothing".
        rec = {"arm": job.name(), "t_branch": job.t_branch, "status": "NA",
               "detail": "operator not applicable at this checkpoint"}
        out.write_text(json.dumps(rec, ensure_ascii=False))
        shutil.rmtree(wd, ignore_errors=True)
        return rec

    start = wd / "start.sav"
    sg.write(start)
    # v2: the engine event hooks ride along with the operator's own lua so every
    # arm gets an attributed event stream (who razed which city, unit_lost reason).
    from .engine_events import hook_lua, parse_log
    # v2: world-parameter operators intervene through server settings applied
    # after the checkpoint loads (`Op.settings`); save-editing operators return {}.
    extra = apply_settings(op, sg, ctx)
    r = run_to_turn(wd, start, job.end_turn, save_every=job.save_every,
                    lua=list(lua) + hook_lua(), extra=extra or None, timeout=7200)
    # ⚠️ A short run is NOT a failed arm.  freeciv stops when the world resolves
    # -- on a 3-civ map one civ can finish the others off well before the
    # horizon -- and the turns it did produce are valid data.  Discarding them
    # would be worse than lossy: the loss lands preferentially on the operators
    # strong enough to end a game, so far-horizon effect sizes would be computed
    # only on the runs where the operator did NOT decide the world.  That is
    # selection bias pointing straight at "no effect".
    #
    # So the arm is kept, the reachable horizons are resolved, and *why* it
    # stopped is recorded (`alive_at_end`) so the truncation can be analysed
    # rather than silently averaged over.
    if not r.saves:
        rec = {"arm": job.name(), "t_branch": job.t_branch, "status": "FAIL",
               "detail": r.message, "applied": desc}
        shutil.rmtree(wd, ignore_errors=True)
        out.write_text(json.dumps(rec, ensure_ascii=False))
        return rec
    engine_events, hooks_attached = parse_log(r.log)

    by_turn: dict[int, Any] = {}
    digests: dict[int, str] = {}
    for p in r.saves:
        s = Savegame.load(p)
        by_turn[s.turn] = resolve_state(s, job.world_id)
        digests[s.turn] = s.state_digest()

    final_state = by_turn.get(max(by_turn), {}) if by_turn else {}
    alive_at_end = sum(1 for v in final_state.values() if v.get("alive"))
    ended_early = r.turn_reached < job.end_turn - job.save_every

    resolved, at_turn = {}, {}
    for h in HORIZONS:
        t, v = _nearest(by_turn, job.t_branch + h, job.save_every)
        if v is None:
            continue
        resolved[str(h)] = v
        at_turn[str(h)] = t

    rec = {
        "arm": job.name(), "op_id": op.id,
        "layer": op.layer, "placebo_of": op.placebo_of,
        "placebo_kind": op.placebo_kind, "t_branch": job.t_branch,
        "status": "OK", "op": op_to_json(op, ctx, desc),
        # Truncation bookkeeping.  `alive_at_end <= 1` means the world resolved;
        # anything else with `ended_early` means the run stopped for a reason we
        # have not explained and the arm should be treated with suspicion.
        "ended_early": ended_early,
        "alive_at_end": alive_at_end,
        "horizons_missing": [h for h in HORIZONS
                             if str(h) not in resolved and job.t_branch + h
                             <= job.end_turn],
        "cond_text": op.cond_text(ctx),
        "resolved": resolved, "resolved_at_turn": at_turn,
        "turn_reached": r.turn_reached,
        # Kept so the empty-arm self-check can be verified from the record
        # alone, without re-running anything.
        "digests": digests,
        "hooks_attached": hooks_attached,
        "engine_events_total": sum(len(v) for v in engine_events.values()),
    }
    out.write_text(json.dumps(rec, ensure_ascii=False))
    # v2: keep the arm's savegames (compressed) and its engine event stream.
    # Disk is no longer the constraint it was for v1; the process inside an arm
    # (not just its horizons) becomes analysable and re-forkable.
    keep_dir = Path(job.world_dir) / "arms" / f"t{job.t_branch}_{job.name()}"
    keep_dir.mkdir(parents=True, exist_ok=True)
    from .shard import compress_file
    for sp in r.saves:
        compress_file(sp, keep_dir / "saves" / (sp.name + ".zst"))
    with open(keep_dir / "events.jsonl", "w") as fh:
        for t in sorted(engine_events):
            fh.write(json.dumps({"t": t, "events": engine_events[t]},
                                ensure_ascii=False) + "\n")
    shutil.rmtree(wd, ignore_errors=True)
    return {"arm": job.name(), "t_branch": job.t_branch,
            "status": "OK_SHORT" if ended_early else "OK",
            "horizons": len(resolved), "alive_at_end": alive_at_end}


# --------------------------------------------------------------------------
# retrofit: harvest the engine event stream for an arm already on disk
# --------------------------------------------------------------------------

def harvest_arm_job(job: ArmJob) -> dict[str, Any]:
    """Re-run one already-resolved arm with the Lua hooks and store its events.

    **Self-verifying, per arm.**  An arm is only accepted if both of these
    reproduce what is already on disk:

      * `applied` -- the operator's own description of the mutation it made.
        Operators are deterministic functions of the checkpoint (the target is
        the median-score living civ, nothing samples), so a different string
        means the regenerated checkpoint is not the one the arm forked from.
      * `resolved` -- the scalars at every horizon.  These are a function of the
        whole trajectory, so equality is the proof that the harvested events
        describe *this* arm and not a near-miss.

    That is why `save_every` is left at the job's original value rather than
    raised to save write bandwidth: the resolution turns have to line up for the
    comparison to mean anything, and a per-arm correctness check is worth far
    more than the I/O it costs.

    Arms recorded as NA or FAIL have no trajectory and are skipped, not failed.
    """
    from .engine_events import hook_lua, parse_log

    stored_p = job.out_path()
    if not stored_p.exists():
        return {"arm": job.name(), "t_branch": job.t_branch, "status": "NO_ARM",
                "detail": "no resolved arm on disk to harvest against"}
    stored = json.loads(stored_p.read_text())
    if stored.get("status") != "OK":
        return {"arm": job.name(), "t_branch": job.t_branch,
                "status": "SKIP_" + str(stored.get("status")),
                "detail": "arm has no trajectory"}

    out = events_path_for(job)
    if out.exists():
        return {"arm": job.name(), "t_branch": job.t_branch, "status": "CACHED"}
    out.parent.mkdir(parents=True, exist_ok=True)

    ckpt = Path(job.checkpoint)
    if not ckpt.exists():
        return {"arm": job.name(), "t_branch": job.t_branch, "status": "NO_CKPT",
                "detail": f"{ckpt} missing; run the mainline harvest first"}

    wd = Path(job.world_dir) / "work" / f"h_t{job.t_branch}_{job.name()}"
    shutil.rmtree(wd, ignore_errors=True)
    wd.mkdir(parents=True, exist_ok=True)

    sg = Savegame.load(ckpt)
    ws = extract(sg, job.world_id)
    mi = extract_map(sg, [p.name for p in ws.players])
    ctx = make_ctx(sg, ws, mi)
    # The operator **as it was when this arm was collected** -- see
    # `interventions.replay_op`.  A library improvement made after collection
    # would otherwise intervene differently on replay, and the events would then
    # belong to a different counterfactual than the `resolved` numbers on disk.
    op = (replay_op(job.op_id) or BY_ID[job.op_id]) if job.op_id else EMPTY
    try:
        desc, lua = apply_op(op, sg, ctx)
    except Exception as e:
        shutil.rmtree(wd, ignore_errors=True)
        return {"arm": job.name(), "t_branch": job.t_branch, "status": "FAIL",
                "detail": f"apply raised: {e}"}

    want_desc = (stored.get("op") or {}).get("applied")
    if want_desc is not None and desc != want_desc:
        shutil.rmtree(wd, ignore_errors=True)
        return {"arm": job.name(), "t_branch": job.t_branch, "status": "MISMATCH_OP",
                "detail": f"applied differs\n  stored: {want_desc[:200]}\n  replay: {desc[:200]}"}

    start = wd / "start.sav"
    sg.write(start)
    r = run_to_turn(wd, start, job.end_turn, save_every=job.save_every,
                    lua=list(lua) + hook_lua(), timeout=7200)
    if not r.saves:
        shutil.rmtree(wd, ignore_errors=True)
        return {"arm": job.name(), "t_branch": job.t_branch, "status": "FAIL",
                "detail": r.message}

    engine, attached = parse_log(r.log)
    if not attached:
        shutil.rmtree(wd, ignore_errors=True)
        return {"arm": job.name(), "t_branch": job.t_branch, "status": "NO_HOOKS",
                "detail": "hook self-test line absent: Lua hooks never attached"}

    by_turn = {s.turn: resolve_state(s, job.world_id)
               for s in (Savegame.load(p) for p in r.saves)}
    replay: dict[str, Any] = {}
    for h in HORIZONS:
        t, v = _nearest(by_turn, job.t_branch + h, job.save_every)
        if v is not None:
            replay[str(h)] = v
    shutil.rmtree(wd, ignore_errors=True)

    if replay != stored.get("resolved"):
        bad = sorted({h for h in set(replay) | set(stored.get("resolved") or {})
                      if replay.get(h) != (stored.get("resolved") or {}).get(h)})
        return {"arm": job.name(), "t_branch": job.t_branch,
                "status": "MISMATCH_RESOLVED",
                "detail": f"horizons differing: {bad}"}

    with gzip.open(out, "wt") as f:
        for t in sorted(engine):
            f.write(json.dumps({"t": t, "events": engine[t]},
                               ensure_ascii=False) + "\n")
    return {"arm": job.name(), "t_branch": job.t_branch, "status": "OK",
            "turns": len(engine),
            "engine_events": sum(len(v) for v in engine.values())}


def events_path_for(job: ArmJob) -> Path:
    return (Path(job.world_dir) / "arms_events"
            / f"t{job.t_branch}_{job.name()}.jsonl.gz")


def reduce_world_events(world_dir: Path) -> dict[str, Any]:
    """Fold `arms_events/*.jsonl.gz` into one `branch_events.jsonl.gz`.

    Mirrors `reduce_world`: the per-arm files are the unit of work and of resume,
    the folded file is the unit of transfer.
    """
    wd = Path(world_dir)
    src = wd / "arms_events"
    if not src.exists():
        return {"world": wd.name, "arms": 0}
    outp = wd / "branch_events.jsonl.gz"
    tmp = outp.with_suffix(outp.suffix + ".part")
    n = ev = 0
    with gzip.open(tmp, "wt") as f:
        for p in sorted(src.glob("*.jsonl.gz")):
            stem = p.name[:-len(".jsonl.gz")]
            t_branch, _, arm = stem[1:].partition("_")
            for line in gzip.open(p, "rt"):
                if not line.strip():
                    continue
                r = json.loads(line)
                f.write(json.dumps({"t_branch": int(t_branch), "arm": arm,
                                    "t": r["t"], "events": r["events"]},
                                   ensure_ascii=False) + "\n")
                ev += len(r["events"])
            n += 1
    os.replace(tmp, outp)
    return {"world": wd.name, "arms": n, "events": ev,
            "bytes": outp.stat().st_size}


# --------------------------------------------------------------------------
# planning and reduction
# --------------------------------------------------------------------------

def choose_points(ckpt_turns: list[int], *, n_points: int, report_endturn: int,
                  max_horizon: int) -> list[int]:
    """Pick branch points out of the mainline's checkpoint turns.

    A point must sit inside the reported window -- otherwise the premise report
    it is conditioned on does not exist in the corpus -- and far enough below
    `max_horizon` that at least the nearest horizon is reachable.  The 20-turn
    lead-in keeps interventions off the opening, where every civ is one settler
    and there is nothing yet for an operator to act on.
    """
    turns = sorted(ckpt_turns)
    if not turns:
        return []
    # v2: the earliest point is turn 60, not first_ckpt+20 (=21).  A t=21 branch
    # point was unusable by the benchmark (its warm-up needs 40-80 turns of history
    # before the anchor), and 60-150 is where alliances form and the expansionist
    # trait still has land to act on -- the regime v1 had no branch point in.
    lo = max(turns[0] + 20, BRANCH_MIN_TURN)
    hi = min(report_endturn, max_horizon - min(HORIZONS))
    usable = [t for t in turns if lo <= t <= hi]
    if not usable:
        return []
    step = max(1, len(usable) // max(n_points, 1))
    return usable[::step][:n_points]


def arms_for_points(world_id: str, world_dir: Path,
                    ckpt_by_turn: dict[int, Path], *, max_horizon: int,
                    save_every: int = 10,
                    only_ops: tuple[str, ...] = (),
                    ops_per_point: int = 0,
                    fallback_na: bool = False, need_h: int = 0) -> list[ArmJob]:
    """Enumerate one job per (given branch point, operator).

    `fallback_na` (2026-09-10 arms census): the sampled operator is skipped when its arm
    or its placebo already exists on disk and is unusable -- status NA / FAIL, or an
    OK arm that stopped before `t_branch + need_h` (game over or the L2-city-kill
    segfault) -- and the next operator in the same cyclic order is sampled instead.
    Deterministic and resumable: the on-disk outcomes decide, so a re-run makes the same
    choice.  Measured before the fix: 396 of 1,900 anchors had no usable
    (factual, real, placebo) triple -- L3-fog / L3-wonder / L2-gov-indirect are NA at
    about half of the anchors (nothing to reveal / no wonder left / civ already has the
    government tech), so sampling one operator blindly loses the anchor for task 3.

    `ops_per_point` > 0 (v2 bench, user 2026-09-10): instead of every operator at
    every anchor, each anchor gets that many *sampled* real operators (plus their
    placebos).  The sample is deterministic -- a world-specific offset into the
    operator list, advancing one per anchor -- so the nine operators are balanced
    across the corpus and a resumed run makes the same choice.  Cuts arms 5x while
    keeping ~200 anchors per operator; task 3 then has one counterfactual question
    per anchor, the same number of test points as tasks 1 and 2.

    Takes the points as already chosen -- it must not re-filter them.  Feeding
    already-selected points back through `choose_points` silently dropped the
    first one, because its 20-turn lead-in was then measured from that point
    itself rather than from the start of the game: a run that should have queued
    69 arms queued 35 and looked like it had succeeded.

    Every operator gets its matched placebo at every point.  Without the pair,
    §2.5.5b showed the far-horizon reading is not attributable to the operator.
    """
    jobs: list[ArmJob] = []
    for i, bt in enumerate(sorted(ckpt_by_turn)):
        end = min(max_horizon, bt + max(HORIZONS))
        common = dict(world_id=world_id, world_dir=str(world_dir), t_branch=bt,
                      checkpoint=str(ckpt_by_turn[bt]), end_turn=end,
                      save_every=save_every)
        # The control, run through the identical reload path as the operators.
        jobs.append(ArmJob(op_id="", arm_name="factual", **common))
        # `only_ops` names real operators; each one's matched placebo comes along
        # automatically, because an operator without its placebo cannot be scored.
        keep = None
        if only_ops:
            keep = set(only_ops) | {f"{o}__placebo" for o in only_ops}
        if ops_per_point > 0:
            reals = [o.id for o in LIBRARY if not o.placebo_of and (keep is None or o.id in keep)]
            import hashlib
            off = int(hashlib.sha1(world_id.encode()).hexdigest()[:8], 16) % len(reals)
            i = sorted(ckpt_by_turn).index(bt)
            start = off + i * ops_per_point
            if fallback_na:
                arms_dir = Path(world_dir) / "arms"
                need_turn = bt + (need_h or max(HORIZONS))

                def unusable(op_id: str) -> bool:
                    for name in (op_id, f"{op_id}__placebo"):
                        pj = arms_dir / f"t{bt}_{name}.json"
                        if not pj.exists():
                            continue                    # not run yet: fine
                        try:
                            d = json.loads(pj.read_text())
                        except Exception:
                            return True
                        if d.get("status") not in ("OK", "OK_SHORT") or int(d.get("turn_reached") or 0) < need_turn:
                            return True
                    return False
                # Replacement order (2026-09-11, after two rounds still produced fresh NAs):
                # `unusable()` can only see arms already on disk, so a blind rotation keeps assigning
                # the STAGE-DEPENDENT operators (L3-fog NA once the map is explored, L3-wonder once the
                # wonders are built, L2-gov-indirect once the gov techs are known -- 65-98% NA past
                # turn 300) to anchors that never tried them.  So: first the sampled operator, then the
                # ones measured to be applicable almost everywhere (rotated by an anchor hash for
                # balance), and only then the stage-dependent ones.
                STAGE_DEPENDENT = ("L3-fog", "L3-wonder", "L2-gov-indirect")
                safe = [o for o in reals if o not in STAGE_DEPENDENT]
                rot = int(hashlib.sha1(f"{world_id}:{bt}".encode()).hexdigest()[:8], 16) % max(1, len(safe))
                order = [reals[(start + k) % len(reals)] for k in range(ops_per_point)] + \
                        [safe[(rot + k) % len(safe)] for k in range(len(safe))] + \
                        [o for o in reals if o in STAGE_DEPENDENT]
                chosen = set()
                for cand in order:
                    if len(chosen) >= ops_per_point:
                        break
                    if not unusable(cand):
                        chosen.add(cand)
                if not chosen:                          # every operator unusable here: keep the original sample
                    chosen = {reals[(start + k) % len(reals)] for k in range(ops_per_point)}
            else:
                chosen = {reals[(start + k) % len(reals)] for k in range(ops_per_point)}
            keep = chosen | {f"{o}__placebo" for o in chosen}
        for op in LIBRARY:
            if keep is not None and op.id not in keep:
                continue
            jobs.append(ArmJob(op_id=op.id, arm_name=op.id, **common))
        # v2: a second identical control at EVERY point (v1 ran it at the first
        # point only, so determinism was never verified for 1,440/1,780 groups).
        # It must come out byte-identical to `factual`; if it does not, no effect
        # size from this point means anything.  Its `resolved` is also kept: the
        # difference between two reruns is the benchmark's RNG floor.
        jobs.append(ArmJob(op_id="", arm_name="factual_check", **common))
    return jobs


def reduce_world(world_dir: Path, world_id: str) -> list[dict]:
    """Collect this world's arm files into one `branch_group` per branch point.

    Factual values come from the `factual` arm at the same branch point, not from
    the mainline: a reload does not reproduce the uninterrupted trajectory (see
    the module docstring), so pairing against the mainline would fold that
    difference into every effect size.
    """
    arms_dir = world_dir / "arms"
    if not arms_dir.exists():
        return []
    by_point: dict[int, list[dict]] = {}
    for f in sorted(arms_dir.glob("t*_*.json")):
        rec = json.loads(f.read_text())
        by_point.setdefault(int(rec["t_branch"]), []).append(rec)

    groups = []
    for bt, arms in sorted(by_point.items()):
        by_name = {a["arm"]: a for a in arms}
        fa = by_name.get("factual")
        chk = by_name.get("factual_check")
        selfcheck, diverged_at = None, []
        if fa and chk and fa.get("status") == "OK" and chk.get("status") == "OK":
            # Compare only the turns a horizon actually resolves from.
            #
            # Comparing every saved turn is a stricter claim than the design
            # needs, and it produced eight false alarms: those worlds agreed at
            # all thirty substantive checkpoints and differed **only in the
            # terminal game-over save**, one turn past the farthest horizon.
            # That save carries end-of-session metadata, so it is not evidence
            # about the trajectory -- and no data is ever read from it.
            #
            # Which turns diverged is recorded rather than collapsed into a
            # boolean, so a real failure can be diagnosed without re-running.
            want = {str(t) for t in (fa.get("resolved_at_turn") or {}).values()}
            shared = (set(fa.get("digests", {})) & set(chk.get("digests", {}))
                      & (want or set(fa.get("digests", {}))))
            if shared:
                diverged_at = sorted((int(t) for t in shared
                                      if fa["digests"][t] != chk["digests"][t]))
                selfcheck = not diverged_at
        horizons = sorted({int(h) for a in arms for h in a.get("resolved", {})})
        # v2: per-arm `diverged` -- did this arm's world ever differ from the
        # factual arm at a horizon turn?  A real arm with diverged=False means
        # the edit never took (verify_ops' REVERTED, caught in production); a
        # placebo with diverged=False is an empty placebo, useless as a control.
        fd = (fa or {}).get("digests") or {}
        want_f = {str(t) for t in ((fa or {}).get("resolved_at_turn") or {}).values()}
        for a in arms:
            ad = a.get("digests") or {}
            shared = set(fd) & set(ad) & (want_f or set(fd))
            a["diverged"] = (any(fd[t] != ad[t] for t in shared) if shared else None)
        groups.append({
            "world_id": world_id, "t_branch": bt, "horizons": horizons,
            # The premise report is NOT stored: render the mainline world_step at
            # t_branch instead, which guarantees it is the same report the model
            # sees on the factual side.
            "premise": {"from_world_step": bt},
            "factual": (fa or {}).get("resolved") or {},
            "factual_status": (fa or {}).get("status", "MISSING"),
            # reload-vs-reload determinism, checked on this world specifically,
            # at the turns horizons are read from
            "determinism_ok": selfcheck,
            "determinism_diverged_turns": diverged_at,
            "arms": [{k: v for k, v in a.items() if k != "digests"}
                     for a in arms if a["arm"] not in ("factual", "factual_check")],
            "n_ended_early": sum(1 for a in arms if a.get("ended_early")),
        })
    return groups
