#!/usr/bin/env python
"""Attainable ceiling of the L3 *effect-capture* score S (score_v2 `A_effect_S`, 2026-09-21).

    S = 1 - sum|s_hat - d| / sum|d|      over the real arm's moved, in-band, scored cells

with d = tf(y_real) - tf(y_control) the true shift and s_hat the predicted one, both / norm.  A
forecaster who knew the engine would predict the *mechanism* part mu of the shift; the observed
d = mu + eps also carries the reload + RNG divergence eps.  The placebo arm (same operator kind,
negligible edit) samples exactly eps, so per (operator, variable), pooled over worlds:

    sigma^2 = mean d_placebo^2
    mu^2    = max(mean d_real^2 - sigma^2, 0)
    E|s_hat - d| for the oracle = E|eps| = sigma * sqrt(2/pi)   (Gaussian eps)
    ceiling(setting) = 1 - sum_cells sigma_(op,var) sqrt(2/pi) / sum_cells |d|

Same cells and weights as the score.  Truth fields only; no model output enters.

    .venv/bin/python bench/measure/l3_effect_ceiling.py --run v3-bench-br --raw raw_tk
"""
from __future__ import annotations
import argparse, collections, gzip, json, math, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "bundle"), str(ROOT / "tools"), str(ROOT)]
from bench.spec_v2 import LOG_VARS  # noqa: E402


def tf(var, x):
    x = float(x)
    return math.copysign(math.log1p(abs(x)), x) if var in LOG_VARS else x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="v3-bench-br")
    ap.add_argument("--raw", default="raw_tk", help="any scored file: only truth fields are read")
    ap.add_argument("--out", default="bench/v2/l3_effect_ceiling.json")
    a = ap.parse_args()
    p = ROOT / "bench/runs" / a.run / f"scored_{a.raw}.jsonl.gz"
    real, plac = {}, {}
    for l in gzip.open(p, "rt"):
        r = json.loads(l)
        if r.get("kind") != "3" or r.get("sub") != "1a" or r.get("y") is None or r.get("y_fac") is None:
            continue
        if not (r.get("in_band") and r.get("scored_var")):
            continue
        key = (r["world"], r["t1"], r["cond"], r["op"], r["civ"], r["var"], r["h"])
        d = (tf(r["var"], r["y"]) - tf(r["var"], r["y_fac"])) / r["norm"]
        (plac if r.get("is_placebo") else real)[key] = (r["set"], r["op"], r["var"], d, bool(r.get("truth_moved")))
    acc = collections.defaultdict(list)
    for (_, op, var, d, _m) in plac.values():
        acc[(op, var)].append(d * d)
    sigma = {k: math.sqrt(sum(v) / len(v)) for k, v in acc.items() if v}
    by = collections.defaultdict(lambda: [0.0, 0.0, 0]); byop = collections.defaultdict(lambda: [0.0, 0.0, 0])
    for (w, t1, cond, op, civ, var, h), (st, _op, _var, d, moved) in real.items():
        if not moved or (op, var) not in sigma:
            continue
        noise = sigma[(op, var)] * math.sqrt(2 / math.pi)
        for tgt in (by[f"{st}|{cond}"], byop[f"{st}|{cond}|{op}"]):
            tgt[0] += noise; tgt[1] += abs(d); tgt[2] += 1
    out = {k: {"effect_ceiling": 1 - n / d, "n_moved": c} for k, (n, d, c) in by.items() if d > 0}
    out["by_op"] = {k: {"effect_ceiling": 1 - n / d, "n": c} for k, (n, d, c) in byop.items() if d > 0}
    out["sigma"] = {f"{op}|{var}": s for (op, var), s in sigma.items()}
    # macro over operators (what gen_tables reports), per setting
    for k in list(by):
        ops = [v["effect_ceiling"] for kk, v in out["by_op"].items() if kk.startswith(k + "|")]
        out[k]["effect_ceiling_macro"] = sum(ops) / len(ops) if ops else None
    (ROOT / a.out).write_text(json.dumps(out, indent=1))
    for k, v in sorted(out.items()):
        if "|" in k and k.count("|") == 1:
            print(f"{k:16s} pooled {v['effect_ceiling']:.3f}  macro-over-ops {v['effect_ceiling_macro']:.3f}  (moved cells {v['n_moved']})")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
