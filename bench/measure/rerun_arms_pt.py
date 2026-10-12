"""Re-run every task-3 arm of v3-bench with a savegame EVERY turn (plan B, user 2026-09-15).

Why: arms saved every 10 turns, so the task-3 1b truth for window templates ("at any turn between
t1 and t1+h ...") was sampled at 10-turn resolution -- `bankrupt` was under-detected in 36% of items,
`newcity` in 10% (14 §B16).  Per-turn saves make the task-3 1b truth definitionally identical to the
main 1b.  Savegames are 0.2-2 MB plain / ~50 KB zstd, so 816k turns cost ~40 GB compressed.

Arms are re-created exactly as `branch.run_arm_job` does (same checkpoint, same operator, same
`make_ctx` target choice, same reload path); the intervention text (`op.applied`) is asserted equal
to the stored arm json, and the state digests at the old 10-turn save points are compared with the
old arm.  ⚠️ They do NOT match, and that is expected: `endturn` and `saveturns` both feed the engine's RNG
stream (measured 2026-09-15 on s510034/t145: end_turn=t1+300 & save_every=10 reproduces the old arm
byte-for-byte; end_turn=t1+60 or save_every=1 gives a different trajectory).  Per-turn arms are therefore a
fresh roll of the same intervention; `factual_check` is re-run under the new settings as the determinism gate.

Output: `data_v2/bench/<world>/arms_pt/t<t1>_<arm>{.json,/saves/*.sav.zst,/events.jsonl}`; `factual_check`
is re-run per anchor so build_bench_v2's determinism gate works under the new settings.
"""
from __future__ import annotations

import argparse
import collections
import gzip
import json
import os
import re
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "data_v2/bench"
RUN = ROOT / "bench/runs/v3-bench"
SANDBOX = ROOT / "freeciv-sandbox"
JOBS = ROOT / "bench/v2/arms_pt_jobs.json"
ARM_DIR = "arms_pt"
ARM_HORIZONS = (30, 60, 90, 120, 150, 200, 250, 300)

_SB = {}
M32 = 0xFFFFFFFF


def fc_srand(seed: int):
    """utility/rand.c fc_srand(): LCG-filled 56-word table, j=0 k=31 x=55, 10,000 warm-up draws (= ceiling_v3)."""
    v = [0] * 56; v[0] = seed & M32
    for i in range(1, 56):
        v[i] = (3 * v[i - 1] + 257) & M32
    j, k, x = 0, 31, 55
    for _ in range(10000):
        bail = 0
        while True:
            new = (v[j] + v[k]) & M32
            x = (x + 1) % 56; j = (j + 1) % 56; k = (k + 1) % 56
            v[x] = new; bail += 1
            if new <= M32 - 1 or bail > 10000:
                break
    return v, j, k, x


def reseed_sg(sg, seed: int):
    """Replace the savegame's `[random]` state by fc_srand(seed): a fresh, engine-exact RNG state."""
    v, j, k, x = fc_srand(seed)
    sg.set_scalar("random", "index_J", j); sg.set_scalar("random", "index_K", k); sg.set_scalar("random", "index_X", x)
    for i in range(8):
        sg.set_scalar("random", f"table{i}", " ".join(f"{w:8x}" for w in v[7 * i:7 * i + 7]))


def sb():
    """v2 sandbox modules, imported once in the main thread (see ceiling_v3)."""
    if not _SB:
        sys.path.insert(0, str(SANDBOX))
        for m in [m for m in sys.modules if m.startswith("sandboxes")]:
            del sys.modules[m]
        from sandboxes.freeciv.runner import run_to_turn
        from sandboxes.freeciv.engine_events import hook_lua, parse_log
        from sandboxes.freeciv.savegame import Savegame
        from sandboxes.freeciv.shard import compress_file
        from sandboxes.freeciv.interventions import BY_ID, EMPTY, apply_op, apply_settings, make_ctx, op_to_json
        from sandboxes.freeciv.state import extract
        from sandboxes.freeciv.geo import extract_map
        _SB.update(locals())
    return _SB


def cmd_jobs(a):
    jobs = {}
    for p in sorted((RUN / "questions").glob("*.jsonl.gz")):
        for l in gzip.open(p, "rt"):
            r = json.loads(l)
            if r["task"] == "3" and r["cond"] == "v1" and r.get("sub") == "1a" and r.get("arm_kind") == "real":
                key = (r["world"], r["t1"])
                jobs[key] = {"world": r["world"], "t1": r["t1"], "op": r["op"], "end_turn": r["t1"] + max(r["horizons"]),
                             "budget": r["budget"], "hz3": r["horizons"]}
    out = []
    for j in jobs.values():
        # factual_check re-run too: `endturn` and `saveturns` both change the RNG stream (measured 2026-09-15:
        # only end_turn=t1+300 / save_every=10 reproduces the old arms byte-for-byte), so the old check's
        # digests cannot gate the per-turn factual -- the determinism gate needs its own twin under the new settings
        for arm in (j["op"], j["op"] + "__placebo", "factual", "factual_check"):
            old = CORPUS / j["world"] / "arms" / f"t{j['t1']}_{arm}.json"
            out.append({**j, "arm": arm, "old_json": str(old), "old_exists": old.exists()})
    JOBS.write_text(json.dumps(out, indent=0))
    turns = sum(j["end_turn"] - j["t1"] for j in out)
    print(f"{len(jobs)} anchors, {len(out)} arms, {turns:,} turns; old json missing: {sum(not j['old_exists'] for j in out)}")
    print(collections.Counter(j["end_turn"] - j["t1"] for j in out))


def run_one(job: dict) -> dict:
    S = sb()
    wid, t1, arm = job["world"], job["t1"], job["arm"]
    wdir = CORPUS / wid
    outj = wdir / ARM_DIR / f"t{t1}_{arm}.json"
    if outj.exists():
        return {**json.load(open(outj)), "status_now": "CACHED"}
    keep = wdir / ARM_DIR / f"t{t1}_{arm}"
    wd = wdir / "work_pt" / f"t{t1}_{arm}"
    if job.get("claim"):
        # second runner (shortest-first): the work dir is the claim; an existing one means the first
        # runner is already simulating this arm
        try:
            wd.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            return {"status_now": "INFLIGHT", "elapsed_s": 0}
    else:
        shutil.rmtree(wd, ignore_errors=True); wd.mkdir(parents=True)
    t0 = time.time()
    old = json.load(open(job["old_json"])) if job["old_exists"] else {}
    Savegame = S["Savegame"]
    sg = Savegame.load(wdir / "ckpt" / f"T{t1:04d}.sav")
    ws = S["extract"](sg, wid)
    mi = S["extract_map"](sg, [p.name for p in ws.players])
    ctx = S["make_ctx"](sg, ws, mi)
    op = S["BY_ID"][arm] if arm not in ("factual", "factual_check") else S["EMPTY"]
    desc, lua = S["apply_op"](op, sg, ctx)
    if job.get("seed") is not None:
        # arms whose anchor hit the engine's trade-route crash under the reload dice: same checkpoint, same
        # intervention, a fresh RNG state shared by all four arms of the anchor (14 §B16-2)
        reseed_sg(sg, int(job["seed"]))
    rec = {"arm": arm, "seed": job.get("seed"), "op_id": op.id, "layer": op.layer, "placebo_of": op.placebo_of, "placebo_kind": op.placebo_kind,
           "t_branch": t1, "op": S["op_to_json"](op, ctx, desc), "end_turn": job["end_turn"], "save_every": 1,
           "end_turn_bumped": bool(job.get("bumped")),
           "applied_match": (desc == (old.get("op") or {}).get("applied") and ctx.target_name == (old.get("op") or {}).get("target")) if old else None}
    if not desc:
        rec["status"] = "NA"
    else:
        start = wd / "start.sav"; sg.write(start)
        extra = S["apply_settings"](op, sg, ctx)
        r = S["run_to_turn"](wd, start, job["end_turn"], save_every=1, lua=list(lua) + S["hook_lua"](),
                             extra=extra or None, timeout=job.get("timeout", 14400))
        if not r.saves:
            rec.update(status="FAIL", detail=r.message[-300:])
        else:
            events, hooks = S["parse_log"](r.log)
            digests, kept = {}, []
            (keep / "saves").mkdir(parents=True, exist_ok=True)
            for p in r.saves:
                m = re.search(r"-T(\d{4})-", p.name)
                if not m:
                    continue
                t = int(m.group(1))
                if t <= t1:
                    continue
                if (t - t1) % 10 == 0 or t == r.turn_reached:      # old arms saved at t1+10k
                    digests[str(t)] = Savegame.load(p).state_digest()
                S["compress_file"](p, keep / "saves" / (p.name + ".zst"))
                kept.append(t)
            with open(keep / "events.jsonl", "w") as fh:
                for t in sorted(events):
                    fh.write(json.dumps({"t": t, "events": events[t]}, ensure_ascii=False) + "\n")
            od = old.get("digests") or {}
            shared = sorted(set(od) & set(digests), key=int)
            rec.update(status=("OK" if r.turn_reached >= job["end_turn"] else "OK_SHORT"), turn_reached=r.turn_reached, message=r.message[-300:],
                       digests=digests, resolved_at_turn={str(h): t1 + h for h in ARM_HORIZONS if t1 + h <= r.turn_reached},
                       hooks_attached=hooks, engine_events_total=sum(len(v) for v in events.values()),
                       n_saves=len(kept), digest_shared=len(shared), digest_equal=sum(od[t] == digests[t] for t in shared),
                       cond_text=old.get("cond_text"))
    rec["elapsed_s"] = round(time.time() - t0, 1)
    shutil.rmtree(wd, ignore_errors=True)
    outj.parent.mkdir(parents=True, exist_ok=True)
    outj.write_text(json.dumps(rec, ensure_ascii=False))
    return {**rec, "status_now": rec["status"]}


def cmd_run(a):
    sb()
    jobs = json.load(open(JOBS))
    if a.reseed:
        seeds = {(k[0], k[1], k[2]): k[3] for k in json.load(open(a.reseed))}
        jobs = [dict(j, seed=seeds[(j["world"], j["t1"], j["arm"])]) for j in jobs if (j["world"], j["t1"], j["arm"]) in seeds]
    if a.bump:
        # arms that stopped early under the new dice (server crash): `endturn` enters the RNG stream,
        # so end_turn+1 is a fresh roll of the same intervention that usually avoids the crash turn
        keys = {tuple(k) for k in json.load(open(a.bump))}
        jobs = [dict(j, end_turn=j["end_turn"] + 1, bumped=True) if (j["world"], j["t1"], j["arm"]) in keys else j for j in jobs]
    if a.limit:
        jobs = jobs[:a.limit]
    jobs = [j for j in jobs if not (CORPUS / j["world"] / ARM_DIR / f"t{j['t1']}_{j['arm']}.json").exists()]
    jobs.sort(key=lambda j: -(j["end_turn"] - j["t1"]) * (1 + int(re.search(r"-(\d+)x(\d+)-", j["world"]).group(1)) ** 2 / 600))
    if a.reverse:
        jobs = [dict(j, claim=True) for j in reversed(jobs)
                if not (CORPUS / j["world"] / "work_pt" / f"t{j['t1']}_{j['arm']}").exists()]
    print(f"{len(jobs)} arms to run, {a.slots} slots", flush=True)
    t0 = time.time(); done = 0; st = collections.Counter(); mism = 0; deq = [0, 0]
    with ThreadPoolExecutor(a.slots) as ex:
        futs = {ex.submit(run_one, j): j for j in jobs}
        for f in as_completed(futs):
            j = futs[f]
            try:
                r = f.result()
            except Exception as e:      # noqa
                r = {"status_now": "EXC", "detail": repr(e)[-300:], "elapsed_s": 0}
                print("  EXC:", r["detail"], flush=True)
            done += 1; st[r["status_now"]] += 1
            if r.get("applied_match") is False:
                mism += 1
            deq[0] += r.get("digest_equal") or 0; deq[1] += r.get("digest_shared") or 0
            rate = done / (time.time() - t0)
            print(f"[{done}/{len(jobs)}] {j['world']}_{j['t1']} {j['arm']} {r['status_now']} {r.get('elapsed_s', 0):.0f}s "
                  f"reached {r.get('turn_reached')} saves {r.get('n_saves')} dig {r.get('digest_equal')}/{r.get('digest_shared')} "
                  f"{dict(st)} applied-mismatch {mism} digests {deq[0]}/{deq[1]} elapsed {(time.time() - t0) / 3600:.2f}h "
                  f"ETA {(len(jobs) - done) / max(rate, 1e-9) / 3600:.2f}h", flush=True)
    (CORPUS / "arms_pt_done.json").write_text(json.dumps({"done": done, "status": dict(st), "applied_mismatch": mism,
                                                          "digest_equal": deq[0], "digest_shared": deq[1], "hours": (time.time() - t0) / 3600}))


def cmd_harvest(a):
    """Finish the bookkeeping of arms whose runner was killed: a work dir with a complete run (terminal
    game-over save at end_turn+1 present) but no json -> digests, compress, events, json; no re-simulation."""
    S = sb(); Savegame = S["Savegame"]; n = 0
    for j in json.load(open(JOBS)):
        wid, t1, arm = j["world"], j["t1"], j["arm"]; wdir = CORPUS / wid
        outj = wdir / ARM_DIR / f"t{t1}_{arm}.json"; wd = wdir / "work_pt" / f"t{t1}_{arm}"
        if outj.exists() or not (wd / "saves").exists():
            continue
        saves = sorted((wd / "saves").glob("*.sav"))
        turns = [int(m.group(1)) for p_ in saves for m in [re.search(r"-T(\d{4})-", p_.name)] if m]
        if not turns or max(turns) <= j["end_turn"]:
            continue                                   # still running, or crashed short: leave it to the retry
        sg = Savegame.load(wdir / "ckpt" / f"T{t1:04d}.sav"); ws = S["extract"](sg, wid)
        mi = S["extract_map"](sg, [q.name for q in ws.players]); ctx = S["make_ctx"](sg, ws, mi)
        op = S["BY_ID"][arm] if arm not in ("factual", "factual_check") else S["EMPTY"]
        desc, _ = S["apply_op"](op, sg, ctx)
        old = json.load(open(j["old_json"])) if j["old_exists"] else {}
        events, hooks = S["parse_log"]((wd / "game.log").read_text(encoding="latin-1")) if (wd / "game.log").exists() else ({}, False)
        keep = wdir / ARM_DIR / f"t{t1}_{arm}"; (keep / "saves").mkdir(parents=True, exist_ok=True)
        digests, kept = {}, []
        for p_ in saves:
            t = int(re.search(r"-T(\d{4})-", p_.name).group(1))
            if t <= t1:
                continue
            if (t - t1) % 10 == 0 or t == max(turns):
                digests[str(t)] = Savegame.load(p_).state_digest()
            S["compress_file"](p_, keep / "saves" / (p_.name + ".zst")); kept.append(t)
        with open(keep / "events.jsonl", "w") as fh:
            for t in sorted(events):
                fh.write(json.dumps({"t": t, "events": events[t]}, ensure_ascii=False) + "\n")
        rec = {"arm": arm, "op_id": op.id, "layer": op.layer, "placebo_of": op.placebo_of, "placebo_kind": op.placebo_kind,
               "t_branch": t1, "op": S["op_to_json"](op, ctx, desc), "end_turn": j["end_turn"], "save_every": 1, "end_turn_bumped": False,
               "applied_match": (desc == (old.get("op") or {}).get("applied") and ctx.target_name == (old.get("op") or {}).get("target")) if old else None,
               "status": "OK", "turn_reached": max(turns), "digests": digests,
               "resolved_at_turn": {str(h): t1 + h for h in ARM_HORIZONS if t1 + h <= max(turns)},
               "hooks_attached": hooks, "engine_events_total": sum(len(v) for v in events.values()),
               "n_saves": len(kept), "cond_text": old.get("cond_text"), "harvested": True, "elapsed_s": None}
        outj.write_text(json.dumps(rec, ensure_ascii=False)); shutil.rmtree(wd, ignore_errors=True); n += 1
        print("harvested", wid, t1, arm, "reached", max(turns), flush=True)
    print(f"harvested {n} arms")


def cmd_check(a):
    st = collections.Counter(); mism = []; deq = [0, 0]; short = []
    for j in json.load(open(JOBS)):
        p = CORPUS / j["world"] / ARM_DIR / f"t{j['t1']}_{j['arm']}.json"
        if not p.exists():
            st["MISSING"] += 1; continue
        r = json.load(open(p)); st[r["status"]] += 1
        if r.get("applied_match") is False:
            mism.append((j["world"], j["t1"], j["arm"]))
        deq[0] += r.get("digest_equal") or 0; deq[1] += r.get("digest_shared") or 0
        if r.get("status") == "OK_SHORT":
            short.append((j["world"], j["t1"], j["arm"], r.get("turn_reached"), j["end_turn"]))
    print(dict(st)); print("applied mismatches:", len(mism), mism[:10]); print("digests equal/shared:", deq)
    print("short:", len(short), short[:10])


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("jobs")
    s = sub.add_parser("run"); s.add_argument("--slots", type=int, default=180); s.add_argument("--limit", type=int, default=0)
    s.add_argument("--reverse", action="store_true", help="second runner: shortest arms first, skip arms another runner has claimed")
    s.add_argument("--reseed", default="", help="json list of [world, t1, arm, seed]: run ONLY these arms, with the [random] state set to fc_srand(seed)")
    s.add_argument("--bump", default="", help="json list of [world, t1, arm] to re-run with end_turn+1 (short arms)")
    sub.add_parser("check"); sub.add_parser("harvest")
    a = ap.parse_args()
    {"jobs": cmd_jobs, "run": cmd_run, "check": cmd_check, "harvest": cmd_harvest}[a.cmd](a)


if __name__ == "__main__":
    main()
