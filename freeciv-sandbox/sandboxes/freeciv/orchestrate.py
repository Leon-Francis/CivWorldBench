"""Flat scheduler for corpus collection.

Three passes, one process pool, no nesting:

  1. **mainline** -- one job per world.  Produces `world.jsonl.gz`,
     `factual.json`, and the retained branch-point savegames.
  2. **arms** -- one job per (world, branch point, operator).  This is ~98% of
     the compute, and flattening it is the point of this module: the previous
     code opened a pool inside a pooled worker, so actual concurrency was the
     product of two limits and nothing tracked it.  Here every arm is a peer and
     the pool size is the only knob.
  3. **reduce** -- collect each world's arm files into `branch.jsonl.gz`,
     pairing every arm against the factual values the mainline already produced.

Resume is per-output-file, not per-run: a world with a `world.jsonl.gz` is
skipped, an arm with its `.json` is skipped.  A crash therefore costs the arms
that were mid-flight, not the batch.  This matters because the batch is measured
in core-days.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from .branch import ArmJob, arms_for_points, reduce_world, run_arm_job
from .collect import collect_world
from .savegame import Savegame
from .worlds import build_manifest, cfg_from_json


# --------------------------------------------------------------------------
# pass 1
# --------------------------------------------------------------------------

def take_lock(out: Path) -> Path:
    """Corpus-level mutual exclusion, shared with the event harvest.

    Two runs over the same corpus destroy each other: they share the per-arm
    `work/` directories, so one `rmtree`s what the other just created.  The
    observed damage was ~600 arms failing with mount errors, PermissionError and
    SIGKILL -- failures that look like simulator bugs and are not -- plus
    root-owned leftover directories that ordinary permissions cannot remove.

    Returns the lock path; the caller must unlink it in a `finally`.
    """
    lock = out / "orchestrate.lock"
    if lock.exists():
        try:
            other = int(lock.read_text().split()[0])
        except (ValueError, IndexError):
            other = -1
        alive = False
        if other > 0:
            try:
                os.kill(other, 0)
                alive = True
            except (ProcessLookupError, PermissionError):
                alive = False
        if alive:
            raise SystemExit(
                f"another job (pid {other}) is already running on {out}. "
                f"Two runs over one corpus corrupt each other's arms; wait for "
                f"it, or kill it and remove {lock}.")
        print(f"[lock] stale lock from pid {other} -> taking over")
    lock.write_text(f"{os.getpid()}\n")
    return lock


def _mainline_job(wj: dict[str, Any], out_root: str, branch_points: int,
                  branch_turns: tuple[int, ...] = ()) -> dict:
    cfg = cfg_from_json(wj)
    wdir = Path(out_root) / wj["world_id"]
    if (wdir / "world.jsonl.gz").exists() and (wdir / "meta.json").exists():
        return {"world_id": wj["world_id"], "ok": True, "cached": True}
    try:
        # Per-tier point count from the manifest; the CLI value is only a fallback
        # so an older manifest still runs.
        n_points = int(wj.get("branch_points") or branch_points)
        return collect_world(
            cfg, Path(out_root),
            branch_points=n_points if wj.get("branch") else 0,
            branch_turns=branch_turns if wj.get("branch") else ())
    except Exception as e:
        return {"world_id": wj["world_id"], "ok": False, "detail": f"{type(e).__name__}: {e}"}


# --------------------------------------------------------------------------
# pass 2 planning
# --------------------------------------------------------------------------

BENCH_NEED_H = {400: 120, 800: 250, 1200: 300}     # bench/spec_v2.HZ_BY_BUDGET max horizon per turn budget


def _arms_for_world(out_root: Path, world_id: str, cfg,
                    only_ops: tuple[str, ...] = (), ops_per_point: int = 0,
                    fallback_na: bool = False) -> list[ArmJob]:
    wdir = out_root / world_id
    ck = wdir / "ckpt"
    if not ck.exists():
        return []
    by_turn = {}
    for p in sorted(ck.glob("*.sav")):
        by_turn[Savegame.load(p).turn] = p
    # The retained savegames ARE the branch points -- collect_world already
    # selected them.  Re-running the selection here would drop the earliest one.
    return arms_for_points(world_id, wdir, by_turn, max_horizon=cfg.max_horizon,
                           only_ops=only_ops, ops_per_point=ops_per_point,
                           fallback_na=fallback_na,
                           need_h=BENCH_NEED_H.get(int(getattr(cfg, "report_endturn", 0) or 0), 0))


def _parse_per_tier(spec: str):
    if ":" not in str(spec):
        return int(spec)
    out = {}
    for part in str(spec).split(","):
        if not part.strip():
            continue
        k, v = part.split(":")
        out[k.strip()] = int(v)
    return out


def free_gb(path: Path) -> float:
    return shutil.disk_usage(path).free / 2**30


def _run_arm(payload: dict) -> dict:
    try:
        return run_arm_job(ArmJob(**payload))
    except Exception as e:
        return {"arm": payload.get("op_id") or "empty",
                "t_branch": payload.get("t_branch"),
                "status": "FAIL", "detail": f"{type(e).__name__}: {e}"}


# --------------------------------------------------------------------------
# pass 3
# --------------------------------------------------------------------------

def reduce_all(out_root: Path, manifest: dict) -> dict[str, int]:
    n_groups = n_worlds = 0
    checks = {"ok": 0, "bad": 0, "unknown": 0}
    for wj in manifest["worlds"]:
        if not wj.get("branch"):
            continue
        wdir = out_root / wj["world_id"]
        groups = reduce_world(wdir, wj["world_id"])
        if not groups:
            continue
        with gzip.open(wdir / "branch.jsonl.gz", "wt") as f:
            for g in groups:
                f.write(json.dumps(g, ensure_ascii=False) + "\n")
                v = g.get("determinism_ok")
                checks["ok" if v is True else ("bad" if v is False else "unknown")] += 1
        n_groups += len(groups)
        n_worlds += 1
    return {"worlds": n_worlds, "groups": n_groups,
            **{f"determinism_{k}": v for k, v in checks.items()}}


# --------------------------------------------------------------------------
# cli
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", help="path to a manifest.json (created if absent "
                                       "when --worlds is given)")
    ap.add_argument("--out", default="data_v2/bench")
    ap.add_argument("--slots", type=int, default=200)
    ap.add_argument("--worlds", type=int, default=0, help="build a manifest of this size")
    ap.add_argument("--ood", type=int, default=0)
    ap.add_argument("--branch_worlds", type=int, default=0)
    ap.add_argument("--branch_tiers", default=None,
                    help="allocate branch treatment per tier, e.g. S,M,L "
                         "(branch cost differs ~12x across tiers)")
    ap.add_argument("--branch_per_tier", default="0",
                    help="one number for all tiers, or per-tier as "
                         "'S:110,M:105,L:105,XL:20' (an L arm costs ~14x an S arm, "
                         "so equal counts are not equal spend)")
    ap.add_argument("--branch_points", type=int, default=4)
    ap.add_argument("--branch_turns", default=None,
                    help="explicit branch-point turns, e.g. 60,160,260; some "
                         "operators are only applicable in one regime")
    ap.add_argument("--ops_per_point", type=int, default=0,
                    help="v2 bench: sample this many real operators (+placebos) per anchor instead of all")
    ap.add_argument("--fallback_na", action="store_true",
                    help="with --ops_per_point: skip a sampled operator whose arm/placebo on disk is NA/FAIL/short and take the next one")
    ap.add_argument("--only_ops", default=None,
                    help="restrict arms to these real operators (placebos follow)")
    # 0 means "let each tier use its own measured budget from worlds.TIER_BUDGET".
    # These used to default to 250/480, which were passed explicitly into
    # build_manifest and silently overrode the per-tier budgets -- every world came
    # out 250 turns long instead of 400/800/1200.  A default that overrides a
    # deliberate per-tier setting is worse than no default.
    ap.add_argument("--report_endturn", type=int, default=0,
                    help="0 = per-tier budget from worlds.TIER_BUDGET")
    ap.add_argument("--max_horizon", type=int, default=0)
    ap.add_argument("--min_free_gb", type=float, default=300.0,
                    help="stop submitting work (pass 1 and pass 2) below this much free space")
    ap.add_argument("--keep_ckpt", action="store_true", default=True,
                    help="(v2: always on) keep the branch-point savegames after reduce")
    ap.add_argument("--n_team", type=int, default=0, help="team worlds (tier MT) in a new manifest")
    ap.add_argument("--team_branch", type=int, default=0)
    ap.add_argument("--ood_branch", default="",
                    help="branch worlds per ood axis, e.g. 'ruleset:5,mapsize:5,civcount:5,ailevel:5,allvary:20,params:5'")
    ap.add_argument("--start_seed", type=int, default=200000)
    ap.add_argument("--no_params", action="store_true",
                    help="pilot P1/P2: keep every world at default server settings (no hidden parameters)")
    ap.add_argument("--tiers", default="",
                    help="override the in-distribution tier cycle, e.g. 'S,M,L' or 'S'")
    ap.add_argument("--skip_arms", action="store_true")
    ap.add_argument("--only_reduce", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # ⚠️ One job per corpus, enforced rather than assumed -- see take_lock().
    lock = take_lock(out)

    mpath = Path(args.manifest) if args.manifest else out / "manifest.json"

    if mpath.exists():
        manifest = json.loads(mpath.read_text())
        print(f"[manifest] {mpath} -> {manifest['counts']}, "
              f"{manifest['branch_worlds']} branch worlds")
    else:
        if not args.worlds:
            ap.error("no manifest and --worlds not given")
        manifest = build_manifest(
            args.worlds, args.ood, branch_worlds=args.branch_worlds,
            report_endturn=args.report_endturn, max_horizon=args.max_horizon,
            branch_tiers=tuple(x for x in (args.branch_tiers or "").split(",") if x)
                         or None,
            branch_per_tier=_parse_per_tier(args.branch_per_tier),
            n_team=args.n_team, team_branch=args.team_branch,
            ood_branch=_parse_per_tier(args.ood_branch) if args.ood_branch else None,
            start_seed=args.start_seed, randomize_params=not args.no_params,
            tiers=tuple(x for x in args.tiers.split(",") if x) or None)
        mpath.write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
        print(f"[manifest] wrote {mpath} -> {manifest['counts']}")
        print(f"[manifest] worlds by tier {manifest['worlds_by_tier']}, "
              f"{manifest['branch_worlds']} branch worlds "
              f"{manifest['branch_by_tier']}")

    worlds = manifest["worlds"]

    if not args.only_reduce:
        # ---- pass 1 ----
        t0 = time.time()
        done = fail = cached = 0
        with ProcessPoolExecutor(max_workers=args.slots) as ex:
            bturns = tuple(int(x) for x in (args.branch_turns or "").split(",") if x)
            futs = {ex.submit(_mainline_job, wj, str(out), args.branch_points,
                              bturns): wj for wj in worlds}
            halted1 = False
            for n1, fut in enumerate(as_completed(futs), 1):
                r = fut.result()
                if n1 % 50 == 0:
                    gb = free_gb(out)
                    print(f"  [pass1] {n1}/{len(futs)} done, {gb:.0f}GB free", flush=True)
                    if gb < args.min_free_gb and not halted1:
                        halted1 = True
                        print(f"  [pass1] ⚠️ HALTING NEW WORK: only {gb:.0f}GB free", flush=True)
                        for f in futs:
                            f.cancel()
                if not r.get("ok"):
                    fail += 1
                    print(f"  [mainline FAIL] {r['world_id']}: {r.get('detail')}",
                          flush=True)
                elif r.get("cached"):
                    cached += 1
                else:
                    done += 1
                    print(f"  [mainline] {r['world_id']} T{r['turn_reached']} "
                          f"{r['records']} recs {r['shard_bytes']/1e6:.2f}MB "
                          f"sim={r['sim_seconds']}s parse={r['parse_seconds']}s",
                          flush=True)
        print(f"[pass1] {done} collected, {cached} cached, {fail} failed "
              f"in {time.time()-t0:.0f}s")

        # ---- pass 2 ----
        if not args.skip_arms:
            per_tier: dict[str, list[dict]] = {}
            for wj in worlds:
                if not wj.get("branch"):
                    continue
                cfg = cfg_from_json(wj)
                only = tuple(x for x in (args.only_ops or "").split(",") if x)
                for j in _arms_for_world(out, wj["world_id"], cfg, only, args.ops_per_point, args.fallback_na):
                    per_tier.setdefault(wj["tier"], []).append(j.__dict__)
            # Interleave tiers round-robin.  A time-bounded run that is cut short
            # would otherwise lose whichever tier happens to sort last, and
            # per-tier comparison is one of the things this data is for -- e.g.
            # whether L4-trait's null effect is specific to the small maps.
            # Arm cost spans ~12x across tiers, so submission order is also the
            # only lever on what a partial run ends up containing.
            jobs: list[dict] = []
            order = sorted(per_tier)
            for i in range(max((len(v) for v in per_tier.values()), default=0)):
                for t in order:
                    if i < len(per_tier[t]):
                        jobs.append(per_tier[t][i])
            print(f"[pass2] {len(jobs)} arms across "
                  f"{sum(1 for w in worlds if w.get('branch'))} worlds, "
                  f"interleaved by tier: "
                  + ", ".join(f"{t}={len(v)}" for t, v in sorted(per_tier.items())))
            t0 = time.time()
            counts: dict[str, int] = {}
            halted = False
            with ProcessPoolExecutor(max_workers=args.slots) as ex:
                futs = [ex.submit(_run_arm, j) for j in jobs]
                for n, fut in enumerate(as_completed(futs), 1):
                    r = fut.result()
                    counts[r["status"]] = counts.get(r["status"], 0) + 1
                    if r["status"] == "FAIL":
                        print(f"  [arm FAIL] {r['arm']} @t{r.get('t_branch')}: "
                              f"{r.get('detail')}", flush=True)
                    if n % 200 == 0:
                        el = time.time() - t0
                        eta = el / n * (len(jobs) - n)
                        gb = free_gb(out)
                        print(f"  [pass2] {n}/{len(jobs)} {counts} "
                              f"{el/3600:.1f}h elapsed, ETA {eta/3600:.1f}h, "
                              f"{gb:.0f}GB free", flush=True)
                        # A multi-day unattended run must not fill the disk: each
                        # in-flight arm holds tens of MB of savegames and the loss
                        # on a full disk is the whole batch, not one arm.  Arms
                        # already queued keep running; nothing new is submitted.
                        if gb < args.min_free_gb and not halted:
                            halted = True
                            print(f"  [pass2] ⚠️ HALTING NEW WORK: only {gb:.0f}GB "
                                  f"free (< {args.min_free_gb}). Queued arms will "
                                  f"finish; re-run with --resume once space is "
                                  f"reclaimed.", flush=True)
                            for f in futs:
                                f.cancel()
            print(f"[pass2] {counts} in {(time.time()-t0)/3600:.2f}h"
                  + ("  (HALTED on low disk)" if halted else ""))

    # ---- pass 3 ----
    try:
        print(f"[pass3] {reduce_all(out, manifest)}")
        if not args.keep_ckpt:
            # The branch-point savegames exist only so arms can fork from them.
            # Once every arm is reduced they are dead weight.
            freed = 0
            for wj in manifest["worlds"]:
                ck = out / wj["world_id"] / "ckpt"
                if not ck.exists():
                    continue
                for f in ck.glob("*.sav"):
                    freed += f.stat().st_size
                    f.unlink()
            print(f"[cleanup] removed branch-point savegames, "
                  f"{freed/2**30:.2f} GB reclaimed")
    finally:
        if lock.exists():
            lock.unlink()


if __name__ == "__main__":
    main()
