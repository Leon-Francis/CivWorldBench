"""Irreducible-noise ceiling for the v3 benchmark (tasks 1a / 1b), from K reseeded factual reruns.

Why.  The report's ④ line ("可达上限 ≈ +0.78") came from ONE factual rerun per v2 task-3 anchor,
scored as a point forecast against the mainline, halved by the CRPS identity, 1a only.  Here:

  * v3 anchors, stratified by (set, budget), one anchor per sampled world;
  * K reseeded reruns of the SAME checkpoint (the savegame's `[random]` state is replaced by
    `fc_srand(seed)` computed exactly as utility/rand.c does) plus the existing `factual` arm
    -> K+1 exchangeable trajectories from the anchor state, plus the mainline;
  * ideal-observer loss estimated with the FAIR (finite-ensemble-unbiased) CRPS / Brier, i.e. the
    expected score of a forecaster that knows the true predictive distribution;
  * aggregated with the scorer's own machinery (trend reference, norms, family macro, in-sample 1b
    cells, world bootstrap), for both the omniscient scope (v1) and the POMDP scope (v2).

Sub-commands:  sample -> smoke -> run -> truth -> score.
"""
from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "data_v2/bench"
import os as _os
OUT = ROOT / _os.environ.get("CEIL_OUT", "data_v2/ceiling")          # CEIL_OUT=data_v2/ceiling_pt for the per-turn re-run
SAVE_EVERY = int(_os.environ.get("CEIL_SAVE_EVERY", "10"))
RUN = ROOT / "bench/runs/v3-bench"
ANCHORS = ROOT / "bench/v2/ceiling_anchors.json"
SANDBOX = ROOT / "freeciv-sandbox"          # the v2 runner (image 3.1.6-v2); bundle/ holds the v1 runner
M32 = 0xFFFFFFFF
DIP = ("n_enemies", "n_allies")


# ----------------------------------------------------------------------------- RNG reseed
def fc_srand(seed: int):
    """utility/rand.c fc_srand(): LCG-filled 56-word table, j=0 k=31 x=55, 10,000 warm-up draws."""
    v = [0] * 56
    v[0] = seed & M32
    for i in range(1, 56):
        v[i] = (3 * v[i - 1] + 257) & M32
    j, k, x = 0, 31, 55
    for _ in range(10000):                    # fc_rand(MAX_UINT32): divisor 1, max MAX_UINT32-1
        bail = 0
        while True:
            new = (v[j] + v[k]) & M32
            x = (x + 1) % 56; j = (j + 1) % 56; k = (k + 1) % 56
            v[x] = new
            bail += 1
            if new <= M32 - 1 or bail > 10000:
                break
    return v, j, k, x


def random_section(seed: int) -> bytes:
    v, j, k, x = fc_srand(seed)
    lines = ["[random]", "saved=TRUE", f"index_J={j}", f"index_K={k}", f"index_X={x}"]
    for i in range(8):
        lines.append(f'table{i}="' + " ".join(f"{w:8x}" for w in v[7 * i:7 * i + 7]) + '"')
    return ("\n".join(lines) + "\n\n").encode()


def reseed(sav: bytes, seed: int) -> bytes:
    a = sav.index(b"\n[random]\n") + 1
    b = sav.index(b"\n[", a)
    while sav[b:b + 2] != b"\n[" or sav[b - 1:b] != b"\n":       # first section header after [random]
        b = sav.index(b"\n[", b + 1)
    return sav[:a] + random_section(seed) + sav[b + 1:]


def seed_of(wid: str, t1: int, k: int) -> int:
    return int(hashlib.sha1(f"{wid}|{t1}|{k}".encode()).hexdigest()[:8], 16)


# ----------------------------------------------------------------------------- sample
def cmd_sample(a):
    from bench.spec_v2 import HZ_BY_BUDGET
    man = json.load(open(CORPUS / "manifest.json"))
    W = {w["world_id"]: w for w in man["worlds"]}
    anchors = collections.defaultdict(list)
    for p in sorted((RUN / "questions").glob("*.jsonl.gz")):
        wid, t1 = p.name[:-9].rsplit("_", 1)
        w = W[wid]
        anchors[(w["split"], int(w["report_endturn"]))].append((wid, int(t1)))
    out = []
    for (st, bud), lst in sorted(anchors.items()):
        worlds = sorted({w for w, _ in lst})
        m = min(a.per_stratum, len(worlds))
        pick = [worlds[round(i * (len(worlds) - 1) / max(1, m - 1))] for i in range(m)] if m > 1 else worlds[:1]
        for i, wid in enumerate(pick):
            ts = sorted(t for w, t in lst if w == wid)
            t1 = ts[i % len(ts)]
            hz = list(HZ_BY_BUDGET[bud])
            out.append({"world": wid, "set": st, "budget": bud, "t1": t1, "hz": hz, "end_turn": t1 + max(hz),
                        "ckpt": str(CORPUS / wid / "ckpt" / f"T{t1:04d}.sav"), "k": a.k})
    ANCHORS.write_text(json.dumps(out, indent=1))
    c = collections.Counter((o["set"], o["budget"]) for o in out)
    print(f"{len(out)} anchors x {a.k} reseeds = {len(out) * a.k} runs -> {ANCHORS.relative_to(ROOT)}")
    for k_, n in sorted(c.items()):
        print(f"  {k_}: {n} anchors, t1 = {sorted(o['t1'] for o in out if (o['set'], o['budget']) == k_)}")


# ----------------------------------------------------------------------------- run
_SB = {}
def _sandbox_imports():
    """Import the v2 runner ONCE (main thread): deleting sys.modules entries from worker threads raced
    against each other on the first launch and every early job died with an import KeyError."""
    if not _SB:
        sys.path.insert(0, str(SANDBOX))
        for m in [m for m in sys.modules if m.startswith("sandboxes")]:
            del sys.modules[m]
        from sandboxes.freeciv.runner import run_to_turn            # noqa
        from sandboxes.freeciv.engine_events import hook_lua        # noqa
        from sandboxes.freeciv.savegame import Savegame             # noqa
        from sandboxes.freeciv.shard import compress_file           # noqa
        _SB.update(run_to_turn=run_to_turn, hook_lua=hook_lua, Savegame=Savegame, compress_file=compress_file)
    return _SB["run_to_turn"], _SB["hook_lua"], _SB["Savegame"], _SB["compress_file"]


def arm_dir(wid: str, t1: int, k: int) -> Path:
    return OUT / wid / "arms" / f"t{t1}_reseed{k}"


def run_one(job: dict) -> dict:
    """One reseeded factual rerun: ckpt -> reseed -> roll to end_turn -> keep the 10-turn saves (.zst)."""
    run_to_turn, hook_lua, Savegame, compress_file = _sandbox_imports()
    wid, t1, k = job["world"], job["t1"], job["k"]
    adir = arm_dir(wid, t1, k); outj = adir.with_suffix(".json")
    if outj.exists():
        return {**json.load(open(outj)), "status_now": "CACHED"}
    wd = OUT / wid / "work" / f"t{t1}_reseed{k}"
    shutil.rmtree(wd, ignore_errors=True); wd.mkdir(parents=True)
    seed = seed_of(wid, t1, k)
    (wd / "start.sav").write_bytes(reseed(Path(job["ckpt"]).read_bytes(), seed))
    t0 = time.time()
    r = run_to_turn(wd, wd / "start.sav", job["end_turn"], save_every=SAVE_EVERY, lua=hook_lua(), timeout=job.get("timeout", 10800))
    rec = {"arm": f"reseed{k}", "t_branch": t1, "seed": seed, "end_turn": job["end_turn"], "save_every": SAVE_EVERY,
           "turn_reached": r.turn_reached, "elapsed_s": round(time.time() - t0, 1), "message": r.message[-300:]}
    if not r.saves:
        rec["status"] = "FAIL"
    else:
        (adir / "saves").mkdir(parents=True, exist_ok=True)
        kept = []
        for p in r.saves:
            m = re.search(r"-T(\d{4})-", p.name)
            if not m or int(m.group(1)) <= t1:
                continue
            compress_file(p, adir / "saves" / (p.name + ".zst"))
            kept.append(int(m.group(1)))
        rec["saves"] = sorted(kept)
        rec["status"] = "OK" if r.turn_reached >= job["end_turn"] - 10 else "OK_SHORT"
    shutil.rmtree(wd, ignore_errors=True)
    outj.parent.mkdir(parents=True, exist_ok=True)
    outj.write_text(json.dumps(rec))
    return {**rec, "status_now": rec["status"]}


def cmd_smoke(a):
    """Two seeds + a repeat of the first, 20 turns, on the smallest sampled world: reseeds must differ
    from each other and from the factual arm, and the repeat must be byte-identical (determinism)."""
    run_to_turn, hook_lua, Savegame, compress_file = _sandbox_imports()
    anchors = json.load(open(ANCHORS))
    an = min((x for x in anchors if x["budget"] == 400), key=lambda x: x["world"])
    wid, t1 = an["world"], an["t1"]
    res = {}
    for tag, k in (("s1", 1001), ("s2", 1002), ("s1b", 1001)):
        wd = OUT / "_smoke" / f"{wid}_{t1}_{tag}"
        shutil.rmtree(wd, ignore_errors=True); wd.mkdir(parents=True)
        (wd / "start.sav").write_bytes(reseed(Path(an["ckpt"]).read_bytes(), seed_of(wid, t1, k)))
        t0 = time.time()
        r = run_to_turn(wd, wd / "start.sav", t1 + 20, save_every=10, lua=hook_lua(), timeout=1800)
        res[tag] = {Savegame.load(p).turn: Savegame.load(p).state_digest() for p in r.saves}
        print(tag, r.message, f"{time.time() - t0:.0f}s", res[tag])
    fac = CORPUS / wid / "arms" / f"t{t1}_factual" / "saves"
    from sandboxes.freeciv.shard import decompress_file
    import tempfile
    fd = {}
    for p in sorted(fac.glob("*.zst")):
        t = int(re.search(r"-T(\d{4})-", p.name).group(1))
        if t1 < t <= t1 + 20:
            with tempfile.NamedTemporaryFile(suffix=".sav") as tmp:
                decompress_file(p, Path(tmp.name)); fd[t] = Savegame.load(Path(tmp.name)).state_digest()
    print("factual", fd)
    ok = res["s1"] == res["s1b"] and res["s1"] != res["s2"] and all(res["s1"].get(t) != d for t, d in fd.items())
    print("SMOKE", "PASS" if ok else "FAIL", "(repeat identical; seeds differ; differ from factual)")
    shutil.rmtree(OUT / "_smoke", ignore_errors=True)


def cmd_run(a):
    _sandbox_imports()
    anchors = json.load(open(ANCHORS))
    jobs = [dict(an, k=k) for an in anchors for k in range(an["k"])]
    jobs = [j for j in jobs if not arm_dir(j["world"], j["t1"], j["k"]).with_suffix(".json").exists()]
    # longest first so the tail of the schedule is short jobs
    jobs.sort(key=lambda j: -(j["end_turn"] - j["t1"]) * (1 + int(re.search(r"-(\d+)x(\d+)-", j["world"]).group(1)) ** 2 / 600))
    print(f"{len(jobs)} runs to do, {a.slots} slots", flush=True)
    t0 = time.time(); done = 0; st = collections.Counter(); el = []
    with ThreadPoolExecutor(a.slots) as ex:
        futs = {ex.submit(run_one, j): j for j in jobs}
        for f in as_completed(futs):
            j = futs[f]
            try:
                r = f.result()
            except Exception as e:      # noqa
                r = {"status_now": "EXC", "message": repr(e)[-300:], "elapsed_s": 0}
                print("  EXC:", r["message"], flush=True)
            done += 1; st[r["status_now"]] += 1; el.append(r.get("elapsed_s") or 0)
            rate = done / (time.time() - t0)
            print(f"[{done}/{len(jobs)}] {j['world']}_{j['t1']} r{j['k']} {r['status_now']} {r.get('elapsed_s', 0):.0f}s "
                  f"reached {r.get('turn_reached')}  {dict(st)}  elapsed {(time.time() - t0) / 3600:.2f}h "
                  f"ETA {(len(jobs) - done) / max(rate, 1e-9) / 3600:.2f}h", flush=True)
    (OUT / "run_done.json").write_text(json.dumps({"done": done, "status": dict(st), "hours": (time.time() - t0) / 3600}))


# ----------------------------------------------------------------------------- truth
def _q_rows(wid: str, t1: int) -> dict:
    rows = {}
    for l in gzip.open(RUN / "questions" / f"{wid}_{t1}.jsonl.gz", "rt"):
        r = json.loads(l)
        if r["task"] in ("1a", "1b") and r["cond"] in ("v1", "v2"):
            rows[(r["task"], r["cond"])] = {k: v for k, v in r.items() if k != "prompt"}
    return rows


def truth_one(an: dict) -> dict:
    from bench.arm_state import arm_records
    from bench.build_bench_v2 import load_recs
    from bench.spec_v2 import derive, task_vars
    from bench.truth_v2 import mil_of, player, truth_1b_from_states
    wid, t1, hz = an["world"], an["t1"], an["hz"]
    dst = OUT / "truth" / f"{wid}_{t1}.json"
    if dst.exists():
        return {"world": wid, "t1": t1, "cached": True}
    q = _q_rows(wid, t1)
    q1a, q1b = q[("1a", "v1")], q.get(("1b", "v1"))
    recs = load_recs(CORPUS / wid, t1, t1 + max(hz))
    civs = q1a["civs"]; mil0 = mil_of(recs[t1])
    lvl_t1 = {cv: derive(player(recs[t1], cv), mil0, recs[t1]) for cv in civs}
    turns = sorted({t1 + h for h in hz} | set(range(t1 + 10, t1 + max(hz) + 1, 10)))
    items = (q1b or {}).get("items") or []
    traj = {"factual": CORPUS / wid / "arms" / f"t{t1}_factual"}
    for k in range(an["k"]):
        d = arm_dir(wid, t1, k)
        if d.with_suffix(".json").exists() and json.load(open(d.with_suffix(".json"))).get("status", "").startswith("OK"):
            traj[f"reseed{k}"] = d
    tr1a, tr1b, reached = {}, {}, {}
    from bench.arm_state import arm_records_light, arm_save_turns
    for name, d in traj.items():
        states = arm_records(d, wid, turns, with_probes=True)
        if SAVE_EVERY == 1 and name != "factual":
            # per-turn arms: light state records at every other turn so the 1b window items are evaluated
            # per turn, exactly like the mainline truth (removes the bankrupt/newcity resolution gap of 14 §B16)
            rest = [t for t in arm_save_turns(d) if t1 < t <= t1 + max(hz) and t not in states]
            states.update(arm_records_light(d, wid, rest))
        reached[name] = max(states) if states else None
        out = {}
        for h in hz:
            r = states.get(t1 + h); out[str(h)] = {}
            for cv in civs:
                p = player(r, cv) if r else None
                out[str(h)][cv] = task_vars(derive(p, mil0, r) if (p and p.get("is_alive")) else None, lvl_t1[cv])
        tr1a[name] = out
        tr1b[name] = truth_1b_from_states(items, states, recs[t1], mil0) if items else []
    # the mainline itself sampled at the arms' 10-turn resolution (window items can differ from the
    # per-turn benchmark truth): the exchangeable comparison for 1b
    if items:
        mt = [t for t in recs if t1 < t <= t1 + max(hz)] if SAVE_EVERY == 1 else turns
        tr1b["main10"] = truth_1b_from_states(items, {t: recs[t] for t in mt if t in recs}, recs[t1], mil0)
    res = {**{k: an[k] for k in ("world", "set", "budget", "t1", "hz")}, "civs": civs,
           "scope_v2": (q.get(("1a", "v2")) or {}).get("civs"),
           "q1a": {k: q1a.get(k) for k in ("truth", "base", "base_hist", "world", "t1", "civs")},
           "items": items, "truth_1b": (q1b or {}).get("truth"),
           "items_v2": (q.get(("1b", "v2")) or {}).get("items"),
           "arms_1a": tr1a, "arms_1b": tr1b, "reached": reached}
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp"); tmp.write_text(json.dumps(res)); tmp.rename(dst)
    return {"world": wid, "t1": t1, "arms": len(traj), "reached": reached}


def cmd_truth(a):
    anchors = json.load(open(ANCHORS))
    with ProcessPoolExecutor(a.jobs) as ex:
        for r in ex.map(truth_one, anchors):
            print(r, flush=True)


# ----------------------------------------------------------------------------- score
def fair_crps(xs, y):
    """Unbiased estimate of the CRPS of the distribution the ensemble is drawn from (Ferro 2014)."""
    K = len(xs)
    m = sum(abs(x - y) for x in xs) / K
    if K < 2:
        return m
    d = sum(abs(xs[i] - xs[j]) for i in range(K) for j in range(i + 1, K)) / (K * (K - 1) / 2)
    return m - d / 2


def fair_brier(ps, y):
    """ps: 0/1 ensemble members (or per-class vectors); fair Brier = (p-y)^2 - p(1-p)/(K-1)."""
    K = len(ps)
    if isinstance(ps[0], dict):
        cls = set().union(*ps)
        s = 0.0
        for c in cls:
            p = sum(x.get(c, 0.0) for x in ps) / K
            s += (p - float(y == c)) ** 2 - (p * (1 - p) / (K - 1) if K > 1 else 0.0)
        return s
    p = sum(ps) / K
    return (p - float(y)) ** 2 - (p * (1 - p) / (K - 1) if K > 1 else 0.0)


def _median(xs):
    s = sorted(xs); n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def records_1a(T: dict) -> list[dict]:
    from bench.score_v2 import _1a_lines, crps5, tf, TAUS
    import numpy as np
    q = T["q1a"]; recs = []
    skel = _1a_lines(q, {}, q["truth"], q["base"] or {}, T["budget"], base_hist=q["base_hist"])
    arms = T["arms_1a"]; names = sorted(arms)
    for r in skel:
        civ, var, h = r["civ"], r["var"], str(r["h"])
        xs_raw = {n: arms[n][h][civ][var] for n in names
                  if isinstance(arms[n].get(h, {}).get(civ, {}).get(var), (int, float))}
        if len(xs_raw) < 3:
            continue
        nz = r["norm"]; y = r["y"]; yt = tf(var, y)
        xt = [tf(var, x) / nz for x in xs_raw.values()]; yn = yt / nz
        K = len(xt)
        rec = {k: r.get(k) for k in ("civ", "var", "h", "y", "norm", "crps_trend", "crps_ref", "ae_trend", "hit_trend")}
        rec.update(world=T["world"], set=T["set"], budget=T["budget"], t1=T["t1"], K=K,
                   in_v2=(T["scope_v2"] is not None and civ in T["scope_v2"]))
        rec["pit"] = (sum(x < yn for x in xt) + 0.5 * sum(x == yn for x in xt)) / K if K >= 8 else None
        rec["fair"] = fair_crps(xt, yn)
        rec["fair_loo"] = sum(fair_crps([x for j, x in enumerate(xt) if j != i], xt[i]) for i in range(K)) / K
        rec["pt_mean"] = sum(abs(x - yn) for x in xt) / K
        rec["pair"] = sum(abs(xt[i] - xt[j]) for i in range(K) for j in range(i + 1, K)) / (K * (K - 1) / 2)
        if "factual" in xs_raw:
            rec["pt_fac"] = abs(tf(var, xs_raw["factual"]) / nz - yn); rec["half_fac"] = rec["pt_fac"] / 2
        qs = [float(v) for v in np.quantile([tf(var, x) for x in xs_raw.values()], TAUS)]
        rec["ensq"] = crps5(qs, yt) / nz
        rec["ensq_raw"] = [float(v) for v in np.quantile(list(xs_raw.values()), TAUS)]
        med = _median(list(xs_raw.values()))
        rec["ae_med"] = abs(med - y); rec["hit_med"] = int(round(med)) == int(y)
        xr = list(xs_raw.values())
        rec["ae_med_loo"] = sum(abs(_median([x for j, x in enumerate(xr) if j != i]) - xr[i]) for i in range(K)) / K
        if "factual" in xs_raw:
            rec["ae_fac"] = abs(xs_raw["factual"] - y)
        recs.append(rec)
    return recs


def records_1b(T: dict) -> list[dict]:
    from bench.score_v2 import cell_1b, brier
    items, truth = T["items"], T["truth_1b"] or []
    arms = T["arms_1b"]; names = sorted(n for n in arms if n != "main10")
    v2keys = {json.dumps(i, sort_keys=True) for i in (T["items_v2"] or [])}
    recs = []
    for i, (it, y) in enumerate(zip(items, truth)):
        if y is None:
            continue
        xs = {n: arms[n][i] for n in names if i < len(arms[n]) and arms[n][i] is not None}
        if len(xs) < 3:
            continue
        rec = dict(world=T["world"], set=T["set"], budget=T["budget"], t1=T["t1"], tpl=it["kind"], civ=it["civ"], h=it["h"],
                   K=len(xs), in_v2=json.dumps(it, sort_keys=True) in v2keys, y=y)
        if it["kind"] == "leader":
            civs = it["civs"]
            rec["brier_base"] = sum((float(c == civs[0]) - float(c == y)) ** 2 for c in civs)
            vec = [{x: 1.0} for x in xs.values()]
            rec["fair"] = fair_brier(vec, y)
            rec["fair_loo"] = sum(fair_brier([v for j, v in enumerate(vec) if j != k], list(xs.values())[k]) for k in range(len(vec))) / len(vec)
            if "factual" in xs:
                rec["pt_fac"] = sum((float(c == xs["factual"]) - float(c == y)) ** 2 for c in civs); rec["half_fac"] = rec["pt_fac"] / 2
            cnt = collections.Counter(xs.values()); K = len(xs)
            rec["ens_p"] = {c: (cnt.get(c, 0) + 0.5) / (K + 0.5 * len(civs)) for c in civs}
            rec["ens"] = sum((rec["ens_p"].get(c, 0.0) - float(c == y)) ** 2 for c in civs)
            m10 = arms.get("main10", [None] * len(items))[i]
            if m10 is not None:
                rec["y10"] = m10; rec["fair_m10"] = fair_brier(vec, m10)
        else:
            cell = cell_1b(it["kind"], T["budget"], it["h"], T["t1"], T["set"], "v1")
            if not cell or cell.get("rate") is None:
                continue
            y = bool(y); rec["y"] = y; rec["rate"] = cell["rate"]
            rec["brier_base"] = brier(cell["rate"], y)
            bs = [float(bool(x)) for x in xs.values()]; K = len(bs)
            rec["fair"] = fair_brier(bs, y)
            rec["fair_loo"] = sum(fair_brier([b for j, b in enumerate(bs) if j != k], bs[k]) for k in range(K)) / K
            rec["pt_mean"] = sum((b - float(y)) ** 2 for b in bs) / K
            if "factual" in xs:
                rec["pt_fac"] = (float(bool(xs["factual"])) - float(y)) ** 2; rec["half_fac"] = rec["pt_fac"] / 2
            rec["ens_p"] = (sum(bs) + 0.5) / (K + 1); rec["ens"] = brier(rec["ens_p"], y)
            m10 = arms.get("main10", [None] * len(items))[i]
            if m10 is not None:
                rec["y10"] = bool(m10); rec["fair_m10"] = fair_brier(bs, bool(m10))
        recs.append(rec)
    return recs


VARIANTS_1A = {          # name -> (loss key for CRPS vars, loss key for the diplomacy MAE vars)
    "ideal_fair": ("fair", "ae_med"),            # ideal observer vs the benchmark (mainline) truth
    "ideal_fair_loo": ("fair_loo", "ae_med_loo"),  # ideal observer vs an exchangeable rerun as truth
    "ensemble_q5": ("ensq", "ae_med"),           # realisable K-member ensemble (5 quantiles)
    "one_rerun_point": ("pt_fac", "ae_fac"),     # the old ④ method: one factual rerun as a point forecast
    "one_rerun_half": ("half_fac", "ae_med"),    # ... halved by the CRPS identity (report 1.1)
    "rerun_point_mean": ("pt_mean", "ae_med"),   # E|X-X'| estimate averaged over all reruns
}
VARIANTS_1B = ["ideal_fair", "ideal_fair_nobankrupt", "ideal_fair_loo", "ideal_fair_main10", "ensemble_laplace", "one_rerun_point", "one_rerun_half", "rerun_point_mean"]
_1B_KEY = {"ideal_fair": "fair", "ideal_fair_nobankrupt": "fair", "ideal_fair_loo": "fair_loo", "ideal_fair_main10": "fair_m10", "ensemble_laplace": "ens",
           "one_rerun_point": "pt_fac", "one_rerun_half": "half_fac", "rerun_point_mean": "pt_mean"}


def macro_1a(rows, loss, dip_loss, nboot):
    from bench.score_v2 import boot_mean, FAMILIES
    import numpy as np
    fam_of = {v: fn for fn, vs in FAMILIES.items() for v in vs}
    pw = collections.defaultdict(lambda: collections.defaultdict(lambda: [0.0, 0.0]))
    pwv = collections.defaultdict(lambda: collections.defaultdict(lambda: [0.0, 0.0]))
    for r in rows:
        num, den = (r.get(dip_loss), r.get("ae_trend")) if r["var"] in DIP else (r.get(loss), r.get("crps_ref"))
        if num is None or den is None:
            continue
        acc = pw[r["world"]][fam_of[r["var"]]]; acc[0] += num; acc[1] += den
        acc = pwv[r["world"]][r["var"]]; acc[0] += num; acc[1] += den
    sk = lambda a: float(np.clip(1 - a[0] / a[1], -1, 1)) if a[1] > 0 else None
    vals, blocks = [], []
    for w in sorted(pw):
        f = [sk(a) for a in pw[w].values()]; f = [x for x in f if x is not None]
        if f:
            vals.append(float(np.mean(f))); blocks.append(w)
    by_fam = {}
    for fn in FAMILIES:
        xs = [sk(pw[w][fn]) for w in pw if fn in pw[w]]; xs = [x for x in xs if x is not None]
        if xs:
            by_fam[fn] = float(np.mean(xs))
    by_fam_eq = {}
    for fn, vs in FAMILIES.items():
        per_w = []
        for w in pwv:
            xs = [sk(pwv[w][v]) for v in vs if v in pwv[w]]; xs = [x for x in xs if x is not None]
            if xs:
                per_w.append(float(np.mean(xs)))
        if per_w:
            by_fam_eq[fn] = float(np.mean(per_w))
    return {"macro": boot_mean(vals, blocks, nboot), "by_family": by_fam, "by_family_macro": by_fam_eq, "n_worlds": len(blocks)}


def cmd_score(a):
    from bench.score_v2 import build_insample_cells, boot_mean
    import numpy as np
    build_insample_cells("v3-bench")
    Ts = [json.loads(p.read_text()) for p in sorted((OUT / "truth").glob("*.json"))]
    R1a, R1b = [], []
    for T in Ts:
        R1a += records_1a(T); R1b += records_1b(T)
    print(f"{len(Ts)} anchors, {len(R1a)} 1a cells, {len(R1b)} 1b items; K per cell: {collections.Counter(r['K'] for r in R1a)}")
    out = {"n_anchors": len(Ts), "anchors": [(T["world"], T["t1"]) for T in Ts], "1a": {}, "1b": {}, "1a_cells": {}, "1b_cells": {},
           "exchangeability": {}, "1b_resolution": {}}
    for st in sorted({r["set"] for r in R1a}):
        for cond in ("v1", "v2"):
            rows = [r for r in R1a if r["set"] == st and (cond == "v1" or r["in_v2"])]
            if not rows:
                continue                      # the generalisation set has no POMDP (v2) rows
            out["1a"][f"{st}|{cond}"] = {name: macro_1a(rows, l, d, a.bootstrap) for name, (l, d) in VARIANTS_1A.items()}
            out["1a"][f"{st}|{cond}"]["n_cells"] = len(rows)
            rb = [r for r in R1b if r["set"] == st and (cond == "v1" or r["in_v2"])]
            d = {}
            for name in VARIANTS_1B:
                key = _1B_KEY[name]
                rr = [r for r in rb if r.get(key) is not None and r.get("brier_base") is not None
                      and not (name.endswith("nobankrupt") and r["tpl"] == "bankrupt")]
                if not rr:
                    continue
                d[name] = {"gain": boot_mean([r["brier_base"] - r[key] for r in rr], [r["world"] for r in rr], a.bootstrap),
                           "skill_ratio": float(1 - sum(r[key] for r in rr) / max(1e-9, sum(r["brier_base"] for r in rr))),
                           "brier": float(np.mean([r[key] for r in rr])), "brier_base": float(np.mean([r["brier_base"] for r in rr])), "n": len(rr)}
            out["1b"][f"{st}|{cond}"] = d
        # exchangeability: reruns vs mainline (pt_mean) against rerun vs rerun (pair), pooled over cells
        rs = [r for r in R1a if r["set"] == st and r["var"] not in DIP]
        out["exchangeability"][st] = {
            "mean|x_k - y_main| / mean|x_k - x_l|": float(sum(r["pt_mean"] for r in rs) / max(1e-12, sum(r["pair"] for r in rs))),
            "mean|x_fac - y_main| / mean|x_k - x_l|": float(sum(r["pt_fac"] for r in rs if "pt_fac" in r) / max(1e-12, sum(r["pair"] for r in rs if "pt_fac" in r))),
            "n_cells": len(rs)}
        # rank (PIT) of the mainline among the reruns: uniform on [0,1] (mean 0.5, 10% in each tail) iff the
        # continuous mainline is exchangeable with the reloaded reruns
        pit = [((r["pit"], r["world"], r["var"], r["t1"])) for r in R1a if r["set"] == st and r.get("pit") is not None]
        def _pit(sel):
            xs = [x for x, *_ in sel]
            return {"mean": boot_mean(xs, [w for _, w, *_ in sel], a.bootstrap), "frac_le_0.1": float(np.mean([x <= 0.1 for x in xs])),
                    "frac_ge_0.9": float(np.mean([x >= 0.9 for x in xs])), "n": len(xs)} if xs else None
        out["exchangeability"][st]["mainline_pit"] = _pit(pit)
        out["exchangeability"][st]["mainline_pit_by_var"] = {v: _pit([x for x in pit if x[2] == v]) for v in sorted({x[2] for x in pit})}
        out["exchangeability"][st]["mainline_pit_by_t1"] = {str(t): _pit([x for x in pit if x[3] == t]) for t in sorted({x[3] for x in pit})}
        rb = [r for r in R1b if r["set"] == st and "y10" in r]
        out["1b_resolution"][st] = {"items": len(rb), "main10 != per-turn truth": float(np.mean([r["y10"] != r["y"] for r in rb])) if rb else None,
                                    "by_template": {k: float(np.mean([r["y10"] != r["y"] for r in rb if r["tpl"] == k])) for k in sorted({r["tpl"] for r in rb})}}
    # cells: floor ratio (ideal loss / trend loss, ratio of sums) per (set, budget, h, var) and per (set, budget, h)
    for keyf in (("set", "budget", "h", "var"), ("set", "budget", "h"), ("set", "var")):
        g = collections.defaultdict(lambda: [0.0, 0.0, 0.0, 0])
        for r in R1a:
            if r["var"] in DIP or r.get("crps_ref") is None:
                continue
            k = "|".join(str(r[x]) for x in keyf); acc = g[k]
            acc[0] += r["fair"]; acc[1] += r["crps_ref"]; acc[2] += r["fair_loo"]; acc[3] += 1
        out["1a_cells"]["|".join(keyf)] = {k: {"floor_ratio": v[0] / v[1] if v[1] > 0 else None, "ceiling": 1 - v[0] / v[1] if v[1] > 0 else None,
                                              "ceiling_loo": 1 - v[2] / v[1] if v[1] > 0 else None, "n": v[3]} for k, v in sorted(g.items())}
    g = collections.defaultdict(lambda: [0.0, 0.0, 0, 0.0, 0.0])
    for r in R1b:
        acc = g[f"{r['set']}|{r['tpl']}"]; acc[0] += r["fair"]; acc[1] += r["brier_base"]; acc[2] += 1; acc[3] += r["fair_loo"]; acc[4] += r.get("fair_m10", r["fair"])
    out["1b_cells"] = {k: {"ceiling_skill_ratio": 1 - v[0] / v[1] if v[1] > 0 else None, "gain": (v[1] - v[0]) / v[2], "n": v[2],
                           "ceiling_loo": 1 - v[3] / v[1] if v[1] > 0 else None, "ceiling_main10": 1 - v[4] / v[1] if v[1] > 0 else None} for k, v in sorted(g.items())}
    out["save_every"] = SAVE_EVERY
    (ROOT / ("bench/v2/ceiling_v3.json" if SAVE_EVERY == 10 else "bench/v2/ceiling_v3_pt.json")).write_text(json.dumps(out, indent=1, ensure_ascii=False))
    with gzip.open(OUT / "ceiling_records.jsonl.gz", "wt") as f:
        for r in R1a:
            f.write(json.dumps({**r, "kind": "1a"}) + "\n")
        for r in R1b:
            f.write(json.dumps({**r, "kind": "1b"}) + "\n")
    # ---- print
    print("\n===== 1a ceiling (family macro skill vs trend; world bootstrap) =====")
    for key, d in out["1a"].items():
        print(f"{key:<18}" + "  ".join(f"{n}: {d[n]['macro']['mean']:+.3f} [{d[n]['macro']['lo']:+.2f},{d[n]['macro']['hi']:+.2f}]" for n in ("ideal_fair", "ideal_fair_loo", "ensemble_q5", "one_rerun_half")))
        print(f"{'':<18}families(ideal_fair) " + " ".join(f"{fn}:{x:+.2f}" for fn, x in d["ideal_fair"]["by_family"].items()) + f"   n_cells {d['n_cells']} worlds {d['ideal_fair']['n_worlds']}")
    print("\n===== 1b ceiling (Brier gain over in-sample base rate; world bootstrap) =====")
    for key, d in out["1b"].items():
        print(f"{key:<18}" + "  ".join(f"{n}: {d[n]['gain']['mean']:+.4f} [{d[n]['gain']['lo']:+.3f},{d[n]['gain']['hi']:+.3f}] ratio {d[n]['skill_ratio']:+.2f}" for n in ("ideal_fair", "ideal_fair_loo", "ideal_fair_main10", "ensemble_laplace") if n in d))
    for st, e in out["exchangeability"].items():
        p_ = e["mainline_pit"]
        print(f"\nexchangeability {st}: reruns-vs-mainline / rerun-vs-rerun distance = {e['mean|x_k - y_main| / mean|x_k - x_l|']:.3f}; "
              f"mainline PIT mean {p_['mean']['mean']:.3f} [{p_['mean']['lo']:.3f},{p_['mean']['hi']:.3f}] tails {p_['frac_le_0.1']:.2f}/{p_['frac_ge_0.9']:.2f} (n={p_['n']})")
        print("   PIT by var:", {v: round(x["mean"]["mean"], 2) for v, x in e["mainline_pit_by_var"].items() if x})
        print("   PIT by t1:", {t: round(x["mean"]["mean"], 2) for t, x in e["mainline_pit_by_t1"].items() if x})
    print("1b resolution (10-turn vs per-turn mainline truth):", json.dumps(out["1b_resolution"]))
    print("\nper (set,budget,h) ceiling:", {k: round(v["ceiling"], 3) for k, v in out["1a_cells"]["set|budget|h"].items()})
    print("per (set,var) ceiling:", {k: round(v["ceiling"], 3) for k, v in out["1a_cells"]["set|var"].items()})
    if a.raw:
        write_raw(Ts, R1a, R1b, a.raw)


def write_raw(Ts, R1a, R1b, tag):
    """Realisable K-member ensemble as model outputs (5 empirical quantiles / Laplace frequencies),
    so `score_v2 --run v3-bench --raw raw_<tag>` scores it through the real pipeline on these anchors."""
    from bench.spec_v2 import VARS
    d = RUN / f"raw_{tag}"; d.mkdir(exist_ok=True)
    by = collections.defaultdict(list)
    for r in R1a:
        by[(r["world"], r["t1"])].append(r)
    byb = collections.defaultdict(dict)
    for r in R1b:
        byb[(r["world"], r["t1"])][(r["tpl"], r["civ"], r["h"], r["y"] if r["tpl"] == "leader" else None)] = r
    n = 0
    for T in Ts:
        key = (T["world"], T["t1"]); q = _q_rows(*key)
        rows = []
        for cond in ("v1", "v2"):
            q1a = q.get(("1a", cond)); q1b = q.get(("1b", cond))
            if q1a:
                cells = {(r["civ"], r["var"], r["h"]): r for r in by[key]}
                lines = []
                for h in q1a["horizons"]:
                    for c in q1a["civs"]:
                        for v in VARS:
                            r = cells.get((c, v, h))
                            vals = r["ensq_raw"] if r else [((q1a["base"].get(c) or {}).get(v) or 0)] * 5
                            lines.append(f"{c} | {v} | +{h} | " + " | ".join(f"{x:.4g}" for x in vals))
                rows.append({"world": T["world"], "tier": "G", "t1": T["t1"], "cond": cond, "task": "1a", "view": 1 if cond == "v1" else 2,
                             "output": "\n".join(lines), "finish": "stop", "parse_rate": 1.0})
            if q1b:
                items = q1b["items"]; lines = []
                for i, it in enumerate(items, 1):
                    r = next((x for x in by_items(R1b, key) if x["tpl"] == it["kind"] and x["civ"] == it["civ"] and x["h"] == it["h"]), None)
                    if it["kind"] == "leader":
                        p = (r or {}).get("ens_p") or {c: (1.0 if k == 0 else 0.0) for k, c in enumerate(it["civs"])}
                        lines.append(f"Q{i} " + "; ".join(f"{c}={p.get(c, 0.0):.3f}" for c in it["civs"]))
                    else:
                        lines.append(f"Q{i} {(r or {}).get('ens_p', (r or {}).get('rate', 0.5)):.3f}")
                rows.append({"world": T["world"], "tier": "G", "t1": T["t1"], "cond": cond, "task": "1b", "view": 1 if cond == "v1" else 2,
                             "output": "\n".join(lines), "finish": "stop", "parse_rate": 1.0})
        with gzip.open(d / f"{T['world']}_{T['t1']}.jsonl.gz", "wt") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        n += 1
    print(f"wrote {n} anchor files -> {d.relative_to(ROOT)};  score: .venv/bin/python bench/score_v2.py --run v3-bench --raw raw_{tag}")


_BY_ITEMS = {}
def by_items(R1b, key):
    if not _BY_ITEMS:
        for r in R1b:
            _BY_ITEMS.setdefault((r["world"], r["t1"]), []).append(r)
    return _BY_ITEMS.get(key, [])


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample"); s.add_argument("--per_stratum", type=int, default=16); s.add_argument("--k", type=int, default=8)
    sub.add_parser("smoke")
    s = sub.add_parser("run"); s.add_argument("--slots", type=int, default=96)
    s = sub.add_parser("truth"); s.add_argument("--jobs", type=int, default=48)
    s = sub.add_parser("score"); s.add_argument("--bootstrap", type=int, default=2000); s.add_argument("--raw", default="ens")
    a = ap.parse_args()
    {"sample": cmd_sample, "smoke": cmd_smoke, "run": cmd_run, "truth": cmd_truth, "score": cmd_score}[a.cmd](a)


if __name__ == "__main__":
    main()
