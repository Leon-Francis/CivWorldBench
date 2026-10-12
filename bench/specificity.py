"""Paired, bounded real/placebo response specificity (independent of accuracy)."""
import collections
import math

import numpy as np


VERSION = "paired-cell-max-operator-macro-v1"


def contrast(real, placebo):
    scale = max(real, placebo)
    return (real - placebo) / scale if scale else 0.0


def summarize_specificity(records, nboot=2000):
    pairs = collections.defaultdict(dict)
    for r in records:
        if r.get("kind") != "3" or r.get("sub") != "1a" or r.get("arm_kind") == "control":
            continue
        if "shift" not in r or not math.isfinite(r["shift"]) or r["shift"] < 0:
            continue
        key = tuple(r.get(k) for k in ("set", "cond", "op", "world", "t1", "civ", "var", "h"))
        arm = bool(r["is_placebo"])
        if arm in pairs[key]:
            raise ValueError(f"Duplicate specificity cell: {key}, placebo={arm}")
        pairs[key][arm] = (r["shift"], r.get("p50_ignore"))

    cells = collections.defaultdict(list)
    available = collections.defaultdict(lambda: [0, 0])
    mismatched = collections.Counter()
    for key, arms in pairs.items():
        setting = "|".join(key[:2])
        for arm in arms:
            available[setting][int(arm)] += 1
        if len(arms) != 2:
            continue
        real, placebo = arms[False], arms[True]
        if real[1] is None or real[1] != placebo[1]:
            mismatched[setting] += 1
            continue
        cells[setting].append((key[2], key[3], contrast(real[0], placebo[0]), max(real[0], placebo[0]) == 0))

    result = {}
    for setting, counts in sorted(available.items()):
        rows = cells[setting]
        ops = sorted({r[0] for r in rows})
        worlds = sorted({r[1] for r in rows})
        oi, wi = {x: i for i, x in enumerate(ops)}, {x: i for i, x in enumerate(worlds)}
        sums = np.zeros((len(worlds), len(ops)))
        nums = np.zeros_like(sums)
        for op, world, value, _ in rows:
            sums[wi[world], oi[op]] += value
            nums[wi[world], oi[op]] += 1
        means = sums.sum(axis=0) / nums.sum(axis=0) if rows else np.array([])
        mean = float(means.mean()) if rows else None
        lo = hi = None
        if rows and nboot:
            rng = np.random.default_rng(0)
            draws = []
            for _ in range(nboot):
                idx = rng.integers(0, len(worlds), len(worlds))
                n = nums[idx].sum(axis=0)
                # Retain the observed operator set in each bootstrap replicate.
                if np.all(n > 0):
                    draws.append(float((sums[idx].sum(axis=0) / n).mean()))
            if draws:
                lo, hi = map(float, np.quantile(draws, [0.025, 0.975]))
        result[setting] = dict(version=VERSION, mean=mean, lo=lo, hi=hi,
                               n_pairs=len(rows), n_worlds=len(worlds), n_operators=len(ops),
                               n_real=counts[0], n_placebo=counts[1],
                               n_control_mismatch=mismatched[setting],
                               n_both_zero=sum(r[3] for r in rows),
                               by_operator={op: dict(mean=float(means[i]), n_pairs=int(nums[:, i].sum()))
                                            for i, op in enumerate(ops)})
    return result
