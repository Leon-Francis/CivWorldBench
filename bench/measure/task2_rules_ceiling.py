#!/usr/bin/env python
"""Attainable ceiling of the L2 world-rule questions (disasters / foodbox / sciencebox bins).

* sciencebox: the prompt's research line prints the cost of the current research, which the
  renderer computes from the ruleset formula and the hidden sciencebox, so a reader who knows the
  formula inverts it exactly -> identifiable, ceiling 1.0 wherever the question is asked.
* foodbox: the city table prints each city's food stock; the granary of a size-s city is a
  ruleset formula times foodbox, so the stock maximum before a growth event (the question is
  only asked when growth events are in the window) identifies foodbox -> ceiling 1.0.
* disasters: the only stochastic one.  The interval narratives list every disaster, and the
  frames give the city counts, so the observer sees a Poisson count over the window's
  city-turns.  Ceiling = accuracy of the best 1-D classifier on the observed log-rate,
  thresholds chosen leave-world-out, evaluated on exactly the asked questions (POMDP questions
  see only the viewer's cities and disasters).

    .venv/bin/python bench/measure/task2_rules_ceiling.py --run v3-bench-br --corpus data_v2/bench --jobs 40
"""
from __future__ import annotations
import argparse, collections, gzip, json, math, pathlib
from concurrent.futures import ProcessPoolExecutor
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
BINS = ((0, 12), (13, 40), (41, 10 ** 9))       # spec_v2.PARAM_BINS["disasters"]


def questions(run):
    """-> {world: [(t1, cond, viewer, truth_bin, set)]} for every asked disasters question"""
    by = collections.defaultdict(list)
    for p in (ROOT / "bench/runs" / run / "questions").glob("*.jsonl.gz"):
        for l in gzip.open(p, "rt"):
            r = json.loads(l)
            if r.get("task") != "2" or "disasters" not in (r.get("params") or []):
                continue
            tb = (r.get("truth") or {}).get("params", {}).get("disasters")
            if tb is None:
                continue
            by[r["world"]].append((r["t1"], r["cond"], r.get("viewer"), int(tb), r["set"]))
    return by


def job(arg):
    """per world: for each asked question, (count, city_turns) in the observation window"""
    wd, qs = arg
    need = {}
    for t1, cond, viewer, tb, st in qs:
        lo = t1 - (80 if t1 >= 81 else 40)
        need[(t1, cond)] = (lo, viewer)
    counts = collections.defaultdict(int); cturns = collections.defaultdict(int)
    city_owner = {}
    for l in gzip.open(pathlib.Path(wd) / "world.jsonl.gz", "rt"):
        d = json.loads(l); t = d["t"]
        players = d["state"]["players"]
        for p in players:
            for c in p.get("cities") or []:
                city_owner[c["id"]] = p["name"]
        for (t1, cond), (lo, viewer) in need.items():
            if not (lo <= t < t1):
                continue
            for p in players:
                if not p.get("is_alive"):
                    continue
                if viewer and p["name"] != viewer:
                    continue
                cturns[(t1, cond)] += len(p.get("cities") or [])
            for e in d.get("events") or []:
                if e.get("type") != "disaster_occurred":
                    continue
                civ = e.get("player") or city_owner.get(e.get("city_id"))
                if viewer and civ != viewer:
                    continue
                counts[(t1, cond)] += 1
    out = []
    for t1, cond, viewer, tb, st in qs:
        out.append((pathlib.Path(wd).name, t1, cond, st, tb, counts[(t1, cond)], cturns[(t1, cond)]))
    return out


def best_thresholds(x, y):
    """two thresholds on x maximising 3-class accuracy (grid over observed values)"""
    xs = np.unique(x); best = (-1, None)
    cand = np.r_[-np.inf, (xs[:-1] + xs[1:]) / 2, np.inf]
    for i in range(len(cand)):
        for j in range(i, len(cand)):
            pred = np.where(x <= cand[i], 0, np.where(x <= cand[j], 1, 2))
            acc = (pred == y).mean()
            if acc > best[0]:
                best = (acc, (cand[i], cand[j]))
    return best[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="v3-bench-br")
    ap.add_argument("--corpus", default="data_v2/bench")
    ap.add_argument("--jobs", type=int, default=40)
    ap.add_argument("--folds", type=int, default=8)
    ap.add_argument("--out", default="bench/v2/rules_ceiling.json")
    a = ap.parse_args()
    qs = questions(a.run)
    worlds = [(str(ROOT / a.corpus / w), v) for w, v in qs.items() if (ROOT / a.corpus / w / "world.jsonl.gz").exists()]
    print(f"{len(worlds)} worlds, {sum(len(v) for _, v in worlds)} asked disasters questions")
    rows = []
    with ProcessPoolExecutor(min(a.jobs, len(worlds))) as ex:
        for r in ex.map(job, worlds, chunksize=1):
            rows += r
    W = np.array([r[0] for r in rows]); T = np.array([r[1] for r in rows]); C = np.array([r[2] for r in rows])
    S = np.array([r[3] for r in rows]); Y = np.array([r[4] for r in rows]); N = np.array([r[5] for r in rows], float)
    E = np.array([r[6] for r in rows], float)
    X = np.log((N + 0.5) / np.maximum(E, 1.0))          # observed log-rate per city-turn
    wids = sorted(set(W)); folds = [set(wids[i::a.folds]) for i in range(a.folds)]
    P = np.zeros_like(Y)
    for f in folds:
        te = np.isin(W, list(f)); tr = ~te
        th = best_thresholds(X[tr], Y[tr])
        P[te] = np.where(X[te] <= th[0], 0, np.where(X[te] <= th[1], 1, 2))
    out = {}
    for st in sorted(set(S)):
        for cond in sorted(set(C)):
            m = (S == st) & (C == cond)
            if not m.any():
                continue
            acc = float((P[m] == Y[m]).mean()); maj = float(max((Y[m] == k).mean() for k in (0, 1, 2)))
            out[f"{st}|{cond}"] = {"disasters_ceiling": acc, "majority": maj, "n": int(m.sum()),
                                  "median_city_turns": float(np.median(E[m])), "median_count": float(np.median(N[m]))}
            print(f"{st}|{cond:<4} disasters ideal-observer acc {acc:.3f} (majority {maj:.3f}, n={int(m.sum())}, "
                  f"median exposure {np.median(E[m]):.0f} city-turns, median count {np.median(N[m]):.0f})")
    out["foodbox_ceiling"] = 1.0; out["sciencebox_ceiling"] = 1.0
    out["note"] = ("foodbox and sciencebox are identifiable by formula from the rendered food stocks and research cost; "
                   "disasters ceiling = leave-world-out 1-D classifier on the window's observed disaster rate")
    (ROOT / a.out).write_text(json.dumps(out, indent=1))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
