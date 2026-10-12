#!/usr/bin/env python
"""Attainable ceiling of the L3 *direction* score.

`truth_moved` is |y_real - y_control| > 0, so a "moved" cell includes cells that only moved
through the chaotic divergence a reload + edit induces, not through the intervention's
mechanism.  On such cells the sign of the true shift is not predictable even by a forecaster
who knows the engine.  The placebo arm (same operator kind, negligible magnitude) is a sample of
exactly that divergence noise, so:

    d      = slog(y_arm) - slog(y_control)           slog(x) = sign(x) * log1p(|x|)  (the 1a transform)
    sigma² = mean over cells of d_placebo²           per (operator, variable), pooled over worlds
    mu²    = max(d_real² - sigma², 0)                 the part of the real shift that is not noise
    ceiling(cell) = Phi(|mu| / sigma)                 P(sign of the observed shift == sign of mu)

The setting's ceiling is the mean over the real arm's moved cells (the same cells and weights the
direction score uses).  Truth fields only; no model output enters.

    .venv/bin/python bench/measure/l3_direction_ceiling.py --run v3-bench-br --raw raw_tk
"""
from __future__ import annotations
import argparse, collections, gzip, json, math, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]


def slog(x):
    return math.copysign(math.log1p(abs(x)), x)


def phi(z):
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="v3-bench-br")
    ap.add_argument("--raw", default="raw_tk", help="any scored file: only truth fields are read")
    ap.add_argument("--out", default="bench/v2/l3_direction_ceiling.json")
    a = ap.parse_args()
    p = ROOT / "bench/runs" / a.run / f"scored_{a.raw}.jsonl.gz"
    real, plac = {}, {}
    for l in gzip.open(p, "rt"):
        r = json.loads(l)
        if r.get("kind") != "3" or r.get("sub") != "1a" or r.get("y") is None or r.get("y_fac") is None:
            continue
        if not (r.get("in_band") and r.get("scored_var")):      # score_v2 scores A / direction on these cells only
            continue
        key = (r["world"], r["t1"], r["cond"], r["op"], r["civ"], r["var"], r["h"])
        d = slog(float(r["y"])) - slog(float(r["y_fac"]))
        (plac if r.get("is_placebo") else real)[key] = (r["set"], r["op"], r["var"], d, bool(r.get("truth_moved")))
    # noise scale per (op, var) from the placebo arm, all cells (moved or not)
    acc = collections.defaultdict(list)
    for (_, op, var, d, _m) in plac.values():
        acc[(op, var)].append(d * d)
    sigma = {k: math.sqrt(sum(v) / len(v)) for k, v in acc.items() if v}
    out = {}
    by = collections.defaultdict(list); byop = collections.defaultdict(list)
    for (w, t1, cond, op, civ, var, h), (st, _op, _var, d, moved) in real.items():
        if not moved:
            continue
        s = sigma.get((op, var))
        if s is None:
            continue
        if s == 0:
            c = 1.0
        else:
            mu = math.sqrt(max(d * d - s * s, 0.0)); c = phi(mu / s)
        by[f"{st}|{cond}"].append(c); byop[f"{st}|{cond}|{op}"].append(c)
    for k, v in sorted(by.items()):
        out[k] = {"direction_ceiling": sum(v) / len(v), "n_moved": len(v),
                  "frac_identifiable": sum(1 for c in v if c > 0.75) / len(v)}
        print(f"{k:<20} direction ceiling {out[k]['direction_ceiling']:.3f}  n_moved={len(v)}  "
              f"cells with P>0.75: {out[k]['frac_identifiable']:.2f}")
    out["by_op"] = {k: {"direction_ceiling": sum(v) / len(v), "n": len(v)} for k, v in sorted(byop.items())}
    out["sigma"] = {f"{op}|{var}": s for (op, var), s in sorted(sigma.items())}
    (ROOT / a.out).write_text(json.dumps(out, indent=1))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
