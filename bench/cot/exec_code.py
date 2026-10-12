"""Sandboxed execution of a model-written `predict(world)` (CoT lens, format C).

    from bench.cot.exec_code import run_predict, sensitivity
    res = run_predict(code, world_json)      # {"ok", "error", "trajectory", "quantiles", "secs"}

Isolation: a fresh `python3 -I -S` subprocess (no site-packages, so numpy & co. are unavailable --
the protocol says standard library only), CPU limit, address-space limit, wall-clock timeout, empty
environment, temp cwd.  This guards against accidents (infinite loops, huge allocations), not
against adversaries; the outputs are only ever compared with the truth.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time

RUNNER = r'''
import json, sys, resource, math
resource.setrlimit(resource.RLIMIT_AS, (4 << 30, 4 << 30))
resource.setrlimit(resource.RLIMIT_CPU, (20, 20))
inp = json.loads(sys.stdin.read())
class KD(dict):
    """dict whose digit keys are ints but that also answers str lookups (review 2026-09-16: 2 of 3
    programs died with KeyError: 345 because JSON had made the frame turns strings)."""
    def _alt(self, k):
        if isinstance(k, str) and k.lstrip("-").isdigit():
            return int(k)
        if isinstance(k, int):
            return str(k)
        return None
    def __getitem__(self, k):
        if dict.__contains__(self, k):
            return dict.__getitem__(self, k)
        a = self._alt(k)
        if a is not None and dict.__contains__(self, a):
            return dict.__getitem__(self, a)
        raise KeyError(k)
    def __contains__(self, k):
        return dict.__contains__(self, k) or (self._alt(k) is not None and dict.__contains__(self, self._alt(k)))
    def get(self, k, d=None):
        try:
            return self[k]
        except KeyError:
            return d
def wrap(o):
    if isinstance(o, dict):
        return KD({(int(k) if isinstance(k, str) and k.lstrip("-").isdigit() else k): wrap(v) for k, v in o.items()})
    if isinstance(o, list):
        return [wrap(v) for v in o]
    return o
ns = {"__name__": "__model__", "math": math}
try:
    exec(compile(inp["code"], "<model>", "exec"), ns)
    fn = ns.get("predict")
    if fn is None:
        raise RuntimeError("no function named predict")
    out = fn(wrap(inp["world"]))
    def fix(o):
        if isinstance(o, dict):
            return {str(k): fix(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [fix(v) for v in o]
        if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
            return None
        return o
    print(json.dumps({"ok": True, "result": fix(out)}, ensure_ascii=False))
except BaseException as e:      # noqa: BLE001
    print(json.dumps({"ok": False, "error": f"{type(e).__name__}: {str(e)[:400]}"}))
'''


def run_predict(code: str, world: dict, wall: float = 30.0) -> dict:
    t0 = time.time()
    with tempfile.TemporaryDirectory() as td:
        try:
            p = subprocess.run([sys.executable, "-I", "-S", "-c", RUNNER],
                               input=json.dumps({"code": code, "world": world}, ensure_ascii=False),
                               capture_output=True, text=True, timeout=wall, cwd=td, env={"PATH": "/usr/bin:/bin"})
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": "timeout", "secs": time.time() - t0}
    secs = time.time() - t0
    line = (p.stdout or "").strip().split("\n")[-1] if p.stdout and p.stdout.strip() else ""
    try:
        d = json.loads(line)
    except Exception:
        return {"ok": False, "error": f"no json on stdout (rc={p.returncode}): {(p.stderr or '')[-300:]}", "secs": secs}
    if not d.get("ok"):
        return {"ok": False, "error": d.get("error", "?"), "secs": secs}
    res = d.get("result")
    if not isinstance(res, dict):
        return {"ok": False, "error": f"predict returned {type(res).__name__}, not dict", "secs": secs}
    return {"ok": True, "trajectory": res.get("trajectory") or {}, "quantiles": res.get("quantiles") or {}, "secs": secs}


def perturbed(world: dict, factor: float = 1.25) -> dict:
    """Scale every numeric level of every civ in every frame (city sizes included)."""
    import copy
    w = copy.deepcopy(world)
    for t, civs in w.get("frames", {}).items():
        for c, s in civs.items():
            if not isinstance(s, dict):
                continue
            for k, v in list(s.items()):
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    s[k] = int(round(v * factor))
            for city in s.get("cities") or []:
                city["size"] = max(1, int(round(city["size"] * factor)))
    return w


def sensitivity(code: str, world: dict, factor: float = 1.25) -> dict | None:
    """Fraction of (civ, quantity, horizon) p50s that change when the inputs are scaled -- a
    'world model' that ignores its inputs returns 0.  None when either run fails."""
    a = run_predict(code, world); b = run_predict(code, perturbed(world, factor))
    if not (a.get("ok") and b.get("ok")):
        return None
    n = ch = 0
    for civ, d in (a["quantiles"] or {}).items():
        for var, hh in (d or {}).items():
            for h, qa in (hh or {}).items():
                qb = ((b["quantiles"].get(civ) or {}).get(var) or {}).get(h)
                if isinstance(qa, list) and isinstance(qb, list) and len(qa) == 5 and len(qb) == 5:
                    n += 1; ch += (qa[2] != qb[2])
    return {"n": n, "frac_changed": (ch / n if n else None), "factor": factor}


# ============================================================ v3.1: stochastic ensemble ==========
RUNNER_ENSEMBLE = r'''
import json, sys, resource, math, random, inspect, time
resource.setrlimit(resource.RLIMIT_AS, (6 << 30, 6 << 30))
inp = json.loads(sys.stdin.read())
resource.setrlimit(resource.RLIMIT_CPU, (int(inp["cpu"]), int(inp["cpu"])))
''' + RUNNER.split("inp = json.loads(sys.stdin.read())", 1)[1].split("ns = {", 1)[0] + r'''
KEEP = set(inp["keep"])
def slim(traj):
    """keep only the scored fields of each state; JSON-safe."""
    out = {}
    if not isinstance(traj, dict):
        return traj
    for t, per in traj.items():
        if not isinstance(per, dict):
            continue
        row = {}
        for civ, st in per.items():
            if not isinstance(st, dict):
                continue
            s2 = {}
            for k in KEEP:
                if k in st:
                    v = st[k]
                    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                        v = None
                    elif isinstance(v, dict):
                        v = {str(a): b for a, b in v.items()}
                    elif isinstance(v, (list, tuple, set)):
                        # city / wonder lists: programs often keep the input dicts -> reduce to the name
                        v = [x if isinstance(x, (str, int, float)) else (str(x.get("name")) if isinstance(x, dict) and "name" in x else str(x)) for x in v]
                    s2[k] = v
            row[str(civ)] = s2
        out[str(t)] = row
    return out
ns = {"__name__": "__model__", "math": math, "random": random}
runs, errors = [], []
try:
    exec(compile(inp["code"], "<model>", "exec"), ns)
    fn = ns.get("predict")
    if fn is None:
        raise RuntimeError("no function named predict")
    try:
        nparam = len(inspect.signature(fn).parameters)
    except (TypeError, ValueError):
        nparam = 2
    W = wrap(inp["world"])
    for seed in range(int(inp["seed0"]), int(inp["seed0"]) + int(inp["n"])):
        t0 = time.time()
        try:
            rng = random.Random(seed)
            out = fn(W, rng) if nparam >= 2 else fn(W)
            if not isinstance(out, dict):
                raise RuntimeError(f"predict returned {type(out).__name__}, not dict")
            runs.append({"seed": seed, "trajectory": slim(out.get("trajectory") or {}), "secs": round(time.time() - t0, 3)})
        except BaseException as e:      # noqa: BLE001
            errors.append({"seed": seed, "error": f"{type(e).__name__}: {str(e)[:400]}"})
            if len(errors) >= 3 and not runs:
                break
        if nparam < 2:
            break                       # deterministic program: one run says it all
    print(json.dumps({"ok": bool(runs), "runs": runs, "errors": errors, "deterministic": nparam < 2}, ensure_ascii=False))
except BaseException as e:      # noqa: BLE001
    import traceback
    tb = traceback.format_exc().strip().split("\n")
    print(json.dumps({"ok": False, "runs": [], "errors": [{"seed": None, "error": f"{type(e).__name__}: {str(e)[:400]}", "trace": tb[-4:]}]}))
'''

KEEP_FIELDS = ("gold", "population", "score", "culture", "n_cities", "city_size_total", "n_techs",
               "n_military_units", "n_civilian_units", "n_enemies", "n_allies", "government", "relations", "cities", "wonders")


def run_ensemble(code: str, world: dict, n: int = 50, seed0: int = 0, wall: float = 180.0, cpu: int = 150) -> dict:
    """Execute `predict(world, rng)` n times (seeds seed0..).  -> {"ok", "runs": [{seed, trajectory, secs}],
    "errors": [...], "deterministic", "secs"}.  A legacy one-argument predict(world) is run once."""
    t0 = time.time()
    with tempfile.TemporaryDirectory() as td:
        try:
            p = subprocess.run([sys.executable, "-I", "-S", "-c", RUNNER_ENSEMBLE],
                               input=json.dumps({"code": code, "world": world, "n": n, "seed0": seed0, "cpu": cpu, "keep": KEEP_FIELDS},
                                                ensure_ascii=False),
                               capture_output=True, text=True, timeout=wall, cwd=td, env={"PATH": "/usr/bin:/bin"})
        except subprocess.TimeoutExpired:
            return {"ok": False, "runs": [], "errors": [{"error": f"timeout after {wall:.0f}s for {n} runs"}], "secs": time.time() - t0}
    secs = time.time() - t0
    line = (p.stdout or "").strip().split("\n")[-1] if p.stdout and p.stdout.strip() else ""
    try:
        d = json.loads(line)
    except Exception:
        return {"ok": False, "runs": [], "errors": [{"error": f"no json on stdout (rc={p.returncode}): {(p.stderr or '')[-400:]}"}], "secs": secs}
    d["secs"] = secs
    return d


def check_program(code: str, world: dict, t1: int, hmax: int, civs: list[str], n: int = 3) -> tuple[bool, str, dict]:
    """Quick validity check used by the repair loop: runs `n` seeds and inspects the structure.
    -> (ok, feedback text for the model, stats).  Feedback names the problem concretely; it never
    mentions the truth."""
    res = run_ensemble(code, world, n=n, seed0=1000, wall=60.0, cpu=40)
    stats = {"exec_ok": bool(res.get("ok")), "secs": res.get("secs"), "deterministic": res.get("deterministic")}
    if not res.get("ok"):
        e = (res.get("errors") or [{}])[0]
        msg = e.get("error", "unknown error")
        tr = "\n".join(e.get("trace") or [])
        return False, (f"Your program failed when executed:\n{msg}\n{tr}\n"
                       "Fix it and output the complete corrected program again in one ```python block."), stats
    probs = []
    need_turns = set(range(t1 + 1, t1 + hmax + 1))
    for run in res["runs"][:n]:
        traj = run["trajectory"] or {}
        turns = set()
        for tk in traj:
            try:
                turns.add(int(float(tk)))
            except ValueError:
                pass
        missing = sorted(need_turns - turns)
        if missing:
            probs.append(f"trajectory lacks {len(missing)} of the {len(need_turns)} required turns (e.g. {missing[:5]}); every turn from {t1 + 1} to {t1 + hmax} is required")
        last = traj.get(str(t1 + hmax)) or traj.get(t1 + hmax) or {}
        cm = {str(c).casefold(): c for c in last}
        miss_c = [c for c in civs if c.casefold() not in cm]
        if len(miss_c) == len(civs) and last:
            probs.append(f"the civilization names in the trajectory ({list(last)[:3]}) do not match world['civs'] ({civs[:3]})")
        for c, st in list(last.items())[:3]:
            miss_k = [k for k in ("gold", "population", "score", "culture", "n_cities", "city_size_total", "n_techs",
                                  "n_military_units", "n_civilian_units", "n_enemies", "n_allies", "government", "relations", "cities", "wonders") if k not in st]
            if miss_k:
                probs.append(f"state of {c} at turn {t1 + hmax} lacks keys {miss_k}")
            bad = [k for k in ("gold", "population", "score", "culture", "n_cities", "city_size_total", "n_techs", "n_military_units", "n_civilian_units", "n_enemies", "n_allies")
                   if k in st and not isinstance(st[k], (int, float))]
            if bad:
                probs.append(f"non-numeric values for {bad} in the state of {c}")
            if "cities" in st and not isinstance(st["cities"], list):
                probs.append(f"'cities' of {c} must be a list of city names")
            if "relations" in st and not isinstance(st["relations"], dict):
                probs.append(f"'relations' of {c} must be a dict {{other civ: relation}}")
        break                                           # structure is the same across seeds
    slow = [r["secs"] for r in res["runs"] if r.get("secs", 0) > 2.0]
    if slow:
        probs.append(f"one call of predict took {max(slow):.1f}s; it must finish within 2 seconds")
    if res.get("deterministic"):
        probs.append("predict takes a single argument; it must be `predict(world, rng)` and use `rng` for uncertain outcomes")
    stats["n_problems"] = len(probs)
    if probs:
        seen = []
        for x in probs:
            if x not in seen:
                seen.append(x)
        return False, ("Your program ran, but its output does not follow the protocol:\n- " + "\n- ".join(seen[:6])
                       + "\nFix it and output the complete corrected program again in one ```python block."), stats
    return True, "", stats


# ============================================================ v3.1: back-test on the observed frames ==========
LEVELS_BT = ("gold", "population", "score", "culture", "n_cities", "city_size_total", "n_techs",
             "n_military_units", "n_civilian_units", "n_enemies", "n_allies")


def backtest(code: str, world: dict, n: int = 5) -> tuple[bool, str, dict]:
    """Start the simulator at the SECOND-LAST observed frame with the frames up to it, run to the last observed
    frame, compare the ensemble median with what was observed (all of it is in the prompt -- no truth leaks).
    -> (ok, feedback, stats).  A cell is flagged when |pred - obs| exceeds half the observed movement plus 10% of
    the level (minimum 2 units; 20 gold).  Feedback lists the worst cells; ok when fewer than 2 cells are flagged."""
    frames = sorted(int(k) for k in world["frames"])
    if len(frames) < 2:
        return True, "", {"skipped": "one frame"}
    f_start, f_end = frames[-2], frames[-1]
    wb = {"turn": f_start, "horizons": [f_end - f_start], "ruleset": world.get("ruleset"), "civs": world["civs"],
          "frames": {str(t): world["frames"][str(t)] for t in frames[:-1]}}
    res = run_ensemble(code, wb, n=n, seed0=2000, wall=60.0, cpu=40)
    stats = {"exec_ok": bool(res.get("ok")), "start": f_start, "end": f_end}
    if not res.get("ok"):
        e = (res.get("errors") or [{}])[0]
        return False, (f"Back-test: your program failed when started at turn {f_start} with the {len(frames) - 1} frame(s) up to that turn "
                       f"(it must work with any number of frames, taking the latest as the start state):\n{e.get('error', '?')}\n"
                       "Fix it and output the complete corrected program again in one ```python block."), stats
    obs_end = world["frames"][str(f_end)]; obs_start = world["frames"][str(f_start)]
    rows = []
    for civ in world["civs"]:
        o2, o1 = obs_end.get(civ), obs_start.get(civ)
        if not o2 or not o1:
            continue
        for k in LEVELS_BT:
            vals = []
            for r in res["runs"]:
                per = r["trajectory"].get(str(f_end)) or {}
                st = per.get(civ) or next((v for c_, v in per.items() if str(c_).casefold() == civ.casefold()), None)
                if st and isinstance(st.get(k), (int, float)):
                    vals.append(float(st[k]))
            if not vals:
                rows.append((civ, k, None, o2[k], o1[k], 9.0)); continue
            vals.sort(); pred = vals[len(vals) // 2]
            move = abs(float(o2[k]) - float(o1[k])); tol = 0.5 * move + 0.1 * abs(float(o2[k])) + (20 if k == "gold" else 2)
            err = abs(pred - float(o2[k]))
            rows.append((civ, k, pred, o2[k], o1[k], err / tol))
    flagged = [r for r in rows if r[5] > 1.0]
    stats.update(n_cells=len(rows), n_flagged=len(flagged), worst=[(r[0], r[1], r[2], r[3], round(r[5], 2)) for r in sorted(rows, key=lambda r: -r[5])[:8]])
    if len(flagged) < 2:
        return True, "", stats
    lines = []
    for civ, k, pred, o2, o1, ratio in sorted(flagged, key=lambda r: -r[5])[:8]:
        ps = "no value" if pred is None else f"{pred:g}"
        lines.append(f"- {civ} | {k}: your median at turn {f_end} = {ps}, observed = {o2:g} (it was {o1:g} at turn {f_start})")
    fb = (f"Back-test: I started your program at turn {f_start} with the frames up to that turn and ran it to turn {f_end}, "
          f"which you observed.  {len(flagged)} of {len(rows)} quantities disagree badly with the observation; the worst:\n"
          + "\n".join(lines) + "\nCalibrate your rates and mechanisms on the observed frames so the program reproduces the "
          "observed movement, then output the complete corrected program again in one ```python block.")
    return False, fb, stats
