#!/usr/bin/env python
"""Attainable ceiling of the v3 personality probe (pairwise within-world ranking).

Same ideal observer as task2_trait_ceiling.py (33 privileged behaviour statistics of the 80
turns before t1, from pilot_ideal_observer.feats), on the v3 anchors, with two extensions:
  * a richer observer: ridge on the features plus their squares (numpy only);
  * pairwise accuracy also restricted to pairs whose true trait gap is >= 20 / 40 / 60, and the
    same restriction applied to the scored model lines, so ceiling and model are compared on
    identical pairs.

    .venv/bin/python bench/measure/task2_trait_ceiling_v3.py --run v3-bench-br --corpus data_v2/bench --jobs 40
"""
from __future__ import annotations
import argparse, collections, gzip, json, pathlib, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "bundle")); sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "bench/measure"))
from pilot_ideal_observer import NAMES, feats, ridge, pred            # noqa: E402

GAPS = (0, 20, 40, 60)
RAWS = ("raw_tk", "raw_opus5", "raw_gpt56sol", "raw_fable")


def job(arg):
    wd, anchors = arg
    try:
        return feats(wd, anchors)
    except Exception as e:
        print(f"!! {wd}: {e}", file=sys.stderr); return []


def bench_anchors(run):
    by = collections.defaultdict(list)
    for p in (ROOT / "bench/runs" / run / "questions").glob("*.jsonl.gz"):
        w, t1 = p.name[:-9].rsplit("_", 1); by[w].append(int(t1))
    return {w: sorted(v) for w, v in by.items()}      # t1=60 anchors have a 40-turn window; feats handles the clamp


def pair_acc(groups, gap_min):
    num = den = 0.0
    for vals in groups:
        for i in range(len(vals)):
            for j in range(i + 1, len(vals)):
                (pi, ti), (pj, tj) = vals[i], vals[j]
                if abs(ti - tj) < max(gap_min, 1):
                    continue
                den += 1; num += 0.5 if pi == pj else float((pi > pj) == (ti > tj))
    return (num / den if den else None), int(den)


def model_pairs(run, raw):
    """{(trait, set): [(gap, hit)]} from a scored file, cond v1"""
    p = ROOT / "bench/runs" / run / f"scored_{raw}.jsonl.gz"
    out = collections.defaultdict(list)
    if not p.exists():
        return out
    for l in gzip.open(p, "rt"):
        r = json.loads(l)
        if r.get("kind") != "2" or not str(r.get("target", "")).startswith("trait_") or r.get("cond") != "v1":
            continue
        if r.get("gap") is None or r.get("hit") is None:
            continue
        out[(r["target"][6:], r["set"])].append((int(r["gap"]), float(r["hit"])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="v3-bench-br")
    ap.add_argument("--corpus", default="data_v2/bench")
    ap.add_argument("--jobs", type=int, default=40)
    ap.add_argument("--folds", type=int, default=8)
    ap.add_argument("--out", default="bench/v2/trait_ceiling_v3.json")
    a = ap.parse_args()
    anc = bench_anchors(a.run)
    worlds = [(str(p.parent), anc[p.parent.name]) for p in (ROOT / a.corpus).glob("*/world.jsonl.gz") if p.parent.name in anc]
    print(f"{len(worlds)} worlds")
    rows = []
    with ProcessPoolExecutor(min(a.jobs, len(worlds))) as ex:
        for r in ex.map(job, worlds, chunksize=1):
            rows += r
    wids = sorted({r[0] for r in rows})
    X = np.array([r[3] for r in rows], float); Y = np.array([r[4] for r in rows], float); wid = np.array([r[0] for r in rows])
    print(f"{len(wids)} worlds · {len(rows)} (civ, t1) samples · {X.shape[1]} features")
    folds = [set(wids[i::a.folds]) for i in range(a.folds)]
    sets = {w: ("bench-gen" if "-G-" in w else "bench-basic") for w in wids}

    def fit(Xf, lam):
        P = np.zeros_like(Y)
        for f in folds:
            te = np.isin(wid, list(f)); tr = ~te
            mu, sd = Xf[tr].mean(0), Xf[tr].std(0) + 1e-9
            for j in range(4):
                P[te, j] = pred((Xf[te] - mu) / sd, ridge((Xf[tr] - mu) / sd, Y[tr, j], lam))
        return P
    P_lin = fit(X, 100.0)
    P_sq = fit(np.c_[X, X ** 2], 300.0)
    out = {"observers": {}, "models": {}, "gaps": list(GAPS)}
    for name, P in (("linear", P_lin), ("quadratic", P_sq)):
        res = {}
        for j, n in enumerate(NAMES):
            g = collections.defaultdict(list)
            for r, pv in zip(rows, P[:, j]):
                g[(r[0], r[2])].append((pv, r[4][j]))
            for st in ("bench-basic", "bench-gen", "all"):
                grp = [v for (w, t), v in g.items() if st == "all" or sets[w] == st]
                for gm in GAPS:
                    acc, dn = pair_acc(grp, gm)
                    res[f"{n}|{st}|gap>={gm}"] = {"acc": acc, "n_pairs": dn}
        out["observers"][name] = res
        print(f"\n== ideal observer ({name}), pairwise accuracy by minimum true gap")
        for st in ("bench-basic", "bench-gen"):
            print(f"  {st}: " + " | ".join(
                f"gap>={gm}: " + "/".join(f"{res[f'{n}|{st}|gap>={gm}']['acc']:.2f}" for n in NAMES) for gm in GAPS))
    for raw in RAWS:
        mp = model_pairs(a.run, raw)
        if not mp:
            continue
        res = {}
        for (n, st), lst in mp.items():
            for gm in GAPS:
                v = [h for gp, h in lst if gp >= max(gm, 1)]
                res[f"{n}|{st}|gap>={gm}"] = {"acc": (sum(v) / len(v) if v else None), "n_pairs": len(v)}
        out["models"][raw] = res
        print(f"\n== {raw}")
        for st in ("bench-basic", "bench-gen"):
            print(f"  {st}: " + " | ".join(
                f"gap>={gm}: " + "/".join((lambda x: f"{x:.2f}" if x is not None else "-")(res.get(f'{n}|{st}|gap>={gm}', {}).get('acc'))
                                          for n in NAMES) for gm in GAPS))
    (ROOT / a.out).write_text(json.dumps(out, indent=1))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
