"""Data-quality checks on the corpus.  No model, no GPU.

The governing lesson is from `docs/HANDOFF_沙盒实验.md` §4: before blaming an
architecture, ask whether the signal is in the data at all.  Every check here is
cheap and is a *prerequisite* for a training run, not a nice-to-have.

  render_roundtrip   reports rebuilt from the corpus must be byte-identical.
                     The corpus stores no text, so if this fails the corpus has
                     lost information and nothing downstream is trustworthy.
  predictability     persistence ("same as last checkpoint") against climatology,
                     per stride.  A persistence error near zero means the task is
                     trivial at that stride, not that the model is good.
  effect_size    ★   for each operator, its distance from factual AND its matched
                     placebo's distance from factual, per horizon.  The ratio is
                     the number that matters: §2.5.5b measured a placebo landing
                     at 74 where the real arm landed at 73 and factual at 82, so
                     the real arm's absolute distance proved nothing at all.
                     ratio ~ 1 => the operator is buying path noise, drop it.
  empty_arm          the empty arm must be byte-identical to the mainline.
  probe_baseline  ★  how well report-derivable features alone predict a probe
                     target.  A learned probe only means something if it beats
                     this; without the baseline a high R^2 says nothing.
  world_split        no world_id may appear in two splits.
  coverage           event types, operator applicability, sizes.
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def iter_worlds(corpus: Path, sample: int = 0, seed: int = 0):
    """All world dirs, or a deterministic sample of `sample` of them.

    Sampling exists because the corpus outgrew the exhaustive scan: 2100 worlds is
    90 GB gzipped, roughly 600 GB of JSON to parse, and a coverage histogram or a
    persistence baseline is identical to three significant figures on a couple of
    hundred worlds.  The checks that must NOT be sampled -- effect sizes, inert
    arms, determinism -- read `branch.jsonl.gz`, which is small, so they stay
    exhaustive.
    """
    ds = [d for d in sorted(corpus.iterdir())
          if d.is_dir() and (d / "world.jsonl.gz").exists()]
    if sample and sample < len(ds):
        ds = sorted(random.Random(seed).sample(ds, sample))
    yield from ds


def stream_jsonl(p: Path):
    """Yield records one at a time.  `load_jsonl` materialises a whole shard --
    up to ~500 MB for a tier-L world -- which is pointless for a single pass."""
    with gzip.open(p, "rt") as f:
        for line in f:
            yield json.loads(line)


def load_jsonl(p: Path) -> list[dict]:
    with gzip.open(p, "rt") as f:
        return [json.loads(line) for line in f]


# --------------------------------------------------------------------------
# 1. the corpus must be renderable
# --------------------------------------------------------------------------

def render_roundtrip(corpus: Path, rate: float = 0.1, seed: int = 0,
                     sample: int = 0) -> dict:
    from .build_samples import _spotcheck
    rng = random.Random(seed)
    n = 0
    bad: list[str] = []
    for d in iter_worlds(corpus, sample):
        for rec in stream_jsonl(d / "world.jsonl.gz"):
            if rng.random() >= rate:
                continue
            n += 1
            m = _spotcheck(rec, "full")
            if m:
                bad.append(f"{d.name}: {m}")
    return {"checked": n, "mismatches": bad[:10], "ok": not bad}


# --------------------------------------------------------------------------
# 2. is there anything to predict
# --------------------------------------------------------------------------

STRIDES = (1, 5, 20, 50, 100)


def predictability(corpus: Path, var: str = "score", sample: int = 0) -> dict:
    """How hard is the transition, measured WITHIN each world and per tier.

    ⚠️ The first version normalised the persistence error by the **corpus-wide**
    variance of `var`.  On 2100 heterogeneous worlds spanning 400 to 1200 turns
    that denominator is enormous -- measured climatology MSE 1,065,056 against the
    pilot's 18,815, a 57x jump driven by tier-L scores running away by T1200 -- so
    every stride came out "trivial" (Delta=50 fell from 30% to 1.78%) even though
    the numerator barely moved.  The task did not get easier; the yardstick got
    longer.

    Cross-world marginal variance answers "how different are two random worlds",
    which is not the question.  Three readings that are:

      within_frac   persistence MSE / the variance of `var` *inside the same
                    world*.  Scale-free per world, so tiers are comparable.
      rel_change    median |var(t+d) - var(t)| / max(1, var(t)).  Reads directly
                    as "over d turns this quantity moves by X% of its value".
      persist_rmse  the raw error, per tier.  Absolute numbers matter when the
                    denominators are this treacherous.
    """
    import math
    from collections import defaultdict as dd

    per: dict[tuple[str, int], list[float]] = dd(list)   # (tier, stride) -> sq err
    rel: dict[tuple[str, int], list[float]] = dd(list)   # (tier, stride) -> rel chg
    wvar: dict[str, list[float]] = dd(list)              # tier -> within-world var

    for d in iter_worlds(corpus, sample):
        tier = d.name.split("-")[2]
        series: dict[str, dict[int, float]] = dd(dict)
        for r in stream_jsonl(d / "world.jsonl.gz"):
            for pl in r["state"]["players"]:
                v = pl.get(var)
                if isinstance(v, (int, float)):
                    series[pl["name"]][r["t"]] = float(v)
        for civ, ser in series.items():
            vals = list(ser.values())
            if len(vals) > 2:
                m = sum(vals) / len(vals)
                wvar[tier].append(sum((x - m) ** 2 for x in vals) / len(vals))
            for st in STRIDES:
                for t, a in ser.items():
                    b = ser.get(t + st)
                    if b is None:
                        continue
                    per[(tier, st)].append((b - a) ** 2)
                    rel[(tier, st)].append(abs(b - a) / max(1.0, abs(a)))

    out: dict[str, Any] = {"var": var, "tiers": {}}
    for tier in sorted(wvar):
        wv = float(np.mean(wvar[tier])) if wvar[tier] else float("nan")
        row = {"within_world_var": round(wv, 1), "strides": {}}
        for st in STRIDES:
            e = per.get((tier, st), [])
            rc = rel.get((tier, st), [])
            if not e:
                continue
            mse = float(np.mean(e))
            row["strides"][str(st)] = {
                "persist_rmse": round(math.sqrt(mse), 1),
                "within_frac": round(mse / wv, 4) if wv and wv == wv else None,
                "rel_change_median": round(float(np.median(rc)), 4) if rc else None,
                "n": len(e),
            }
        out["tiers"][tier] = row
    return out


def report_change(corpus: Path, strides=(1, 5, 20, 50, 100), per_tier: int = 20,
                  per_world: int = 6, seed: int = 0) -> dict:
    """How much of `report_{t+d}` is NOT already in `report_t`.

    This replaces an earlier `score`-based "difficulty ladder" that was confounded
    and should not be used.  That metric divided the persistence error by the
    within-world variance of `score`, and `score` grows monotonically all game --
    so the denominator was dominated by the growth trend rather than by
    unpredictability.  The giveaway: persistence/drift came out at 2.8-4.9x and was
    **flat across strides** (4.73, 4.88, 4.52), which means it was measuring the
    slope of the trend, not how hard the step is.  It also described one scalar out
    of nineteen while the actual task is to emit a 20-150 KB report.

    This measure has no denominator to get wrong.  The training target IS the next
    report, so "what fraction of its lines are new" is exactly the part the model
    must produce by reasoning instead of by copying.  Reported per section as well,
    because a section that is both large and static inflates the total while
    contributing no learnable signal.
    """
    import difflib
    import re

    from .build_samples import _report_of

    rng = random.Random(seed)
    agg: dict[tuple[str, int], list[float]] = defaultdict(list)
    sect: dict[tuple[str, int, str], list[float]] = defaultdict(list)
    share: dict[tuple[str, str], list[float]] = defaultdict(list)

    def sections(text: str) -> dict[str, list[str]]:
        out, cur = {}, "HEADER"
        buf: list[str] = []
        for line in text.split("\n"):
            m = re.match(r"^### (.+)", line)
            if m:
                out[cur] = buf
                cur = m.group(1).split("(")[0].strip()
                buf = []
            else:
                buf.append(line)
        out[cur] = buf
        return out

    def new_frac(a: list[str], b: list[str]) -> float:
        """Share of b's lines that are not matched in a."""
        if not b:
            return 0.0
        sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
        matched = sum(bl.size for bl in sm.get_matching_blocks())
        return 1.0 - matched / len(b)

    by_tier: dict[str, list[Path]] = defaultdict(list)
    for d in sorted(corpus.iterdir()):
        if (d / "world.jsonl.gz").exists():
            by_tier[d.name.split("-")[2]].append(d)

    for tier, ds in by_tier.items():
        for d in rng.sample(ds, min(per_tier, len(ds))):
            recs = {r["t"]: r for r in stream_jsonl(d / "world.jsonl.gz")}
            turns = sorted(recs)
            if len(turns) < max(strides) + 2:
                continue
            picks = rng.sample(turns[: -max(strides)], min(per_world, len(turns) - max(strides)))
            cache: dict[int, str] = {}

            def rep(t: int) -> str:
                if t not in cache:
                    cache[t] = _report_of(recs[t], "full")
                return cache[t]

            for t in picks:
                a = rep(t)
                sa = sections(a)
                tot_a = len(a.split("\n"))
                for name, lines in sa.items():
                    share[(tier, name)].append(len(lines) / max(1, tot_a))
                for st_ in strides:
                    if t + st_ not in recs:
                        continue
                    b = rep(t + st_)
                    agg[(tier, st_)].append(new_frac(a.split("\n"), b.split("\n")))
                    sb = sections(b)
                    for name, lines in sb.items():
                        sect[(tier, st_, name)].append(new_frac(sa.get(name, []), lines))

    out: dict[str, Any] = {"tiers": {}}
    for tier in sorted(by_tier):
        row: dict[str, Any] = {"overall": {}, "by_section": {}, "section_share": {}}
        for st_ in strides:
            v = agg.get((tier, st_), [])
            if v:
                row["overall"][str(st_)] = round(float(np.median(v)), 4)
        names = sorted({k[2] for k in sect if k[0] == tier})
        for name in names:
            row["section_share"][name] = round(
                float(np.mean(share.get((tier, name), [0]))), 4)
            row["by_section"][name] = {
                str(st_): round(float(np.median(sect[(tier, st_, name)])), 4)
                for st_ in strides if sect.get((tier, st_, name))}
        out["tiers"][tier] = row
    return out


def targeted_effect(corpus: Path) -> dict:
    """Score each operator on the variables it declares, three ways.

    The generic distance averages 9 variables over every civ.  For an operator
    that moves one variable on one civ, ~71 of 72 (civ, var) pairs carry only path
    noise -- and since that noise appears in the real arm AND its placebo, it
    compresses the ratio toward 1 no matter how big the real effect is.

    Three readings, because they answer different questions:

      target   the operator's own variables, on the civ it acted on.
               "Did it move what it was aimed at?"  -- attribution.
      world    the operator's own variables, summed over every civ.
               "How much did those quantities move anywhere?"  Needed because
               the knock-on effect can be larger elsewhere: in one measured
               branch, razing the target's city cost the target 1 city while a
               *third* civ lost 3.
      early    fraction of arms whose world resolved before the horizon.  An
               operator that ends games is showing a very large effect, and it
               truncates its own far-horizon average, so the rate has to be read
               as an effect rather than as missing data.
    """
    from .interventions import BY_ID

    acc: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    early: Counter = Counter()
    total: Counter = Counter()
    # ★ Two effect sizes per operator, not one.
    #
    # An arm whose `resolved` equals `factual` at every horizon is inert: the
    # operator ran and the world did not move.  Measured on the full corpus that is
    # 42% of `L4-love` arms and 17% of `L4-trait` arms, so the unconditional mean
    # answers "how strong is this operator on average" while mixing in a large
    # block of zeros -- two different questions collapsed into one number.
    #
    # The conditional mean drops a PAIR when either side is inert.  Dropping only
    # the real arm would break the pairing, which is the one thing the whole design
    # rests on.
    inert: dict[tuple[str, str], set] = defaultdict(set)
    cond: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    keyed: dict[tuple[str, str, str, str], list] = {}

    def d(a: dict, b: dict, vars_: tuple, only: str | None) -> float | None:
        civs = [only] if only else sorted(set(a) & set(b))
        tot, n = 0.0, 0
        for civ in civs:
            if civ not in a or civ not in b:
                continue
            for k in vars_:
                x, y = a[civ].get(k), b[civ].get(k)
                if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                    tot += abs(float(x) - float(y)) / max(1.0, abs(float(y)))
                    n += 1
                elif x is not None and y is not None:
                    tot += 0.0 if x == y else 1.0
                    n += 1
        return tot / n if n else None

    for wd in iter_worlds(corpus):
        bp = wd / "branch.jsonl.gz"
        if not bp.exists():
            continue
        for g in load_jsonl(bp):
            fact = g.get("factual") or {}
            for arm in g["arms"]:
                if arm.get("status") != "OK":
                    continue
                op = BY_ID.get(arm["arm"])
                if op is None or not op.score_vars:
                    continue
                base = arm.get("placebo_of") or arm["arm"]
                kind = "placebo" if arm.get("placebo_of") else "real"
                total[(base, kind)] += 1
                if arm.get("ended_early"):
                    early[(base, kind)] += 1
                tgt = (arm.get("op") or {}).get("target")
                # Identity of this pair, so the real and placebo arms of the same
                # (world, branch point) can be matched up afterwards.
                pair_id = (g["world_id"], g["t_branch"])
                shared = [h for h in (arm.get("resolved") or {}) if h in fact]
                if shared and all((arm["resolved"])[h] == fact[h] for h in shared):
                    inert[(base, kind)].add(pair_id)
                for h, res in (arm.get("resolved") or {}).items():
                    f = fact.get(h)
                    if not f:
                        continue
                    lo, hi = op.horizon_band
                    if not (lo <= int(h) <= hi):
                        continue
                    for scope, only in (("target", tgt), ("world", None)):
                        v = d(res, f, op.score_vars, only)
                        if v is not None:
                            acc[(base, kind, scope)].append(v)
                            keyed.setdefault((base, kind, scope, str(pair_id)),
                                             []).append(v)

    # Conditional means: keep only pairs where NEITHER arm was inert.
    for (base, kind, scope, pid), vals in keyed.items():
        dead = inert.get((base, "real"), set()) | inert.get((base, "placebo"), set())
        if eval(pid) in dead:
            continue
        cond[(base, kind, scope)].extend(vals)

    rows = []
    for base in sorted({k[0] for k in acc} | {k[0] for k in total}):
        row = {"op": base, "layer": BY_ID[base].layer,
               "vars": ",".join(BY_ID[base].score_vars),
               "score_by": BY_ID[base].score_by,
               "band": list(BY_ID[base].horizon_band)}
        for scope in ("target", "world"):
            r = acc.get((base, "real", scope), [])
            p = acc.get((base, "placebo", scope), [])
            rm = float(np.mean(r)) if r else None
            pm = float(np.mean(p)) if p else None
            row[scope] = round(rm / pm, 2) if (rm and pm) else None
            row[f"{scope}_real"] = round(rm, 4) if rm is not None else None
            rc = cond.get((base, "real", scope), [])
            pc = cond.get((base, "placebo", scope), [])
            rcm = float(np.mean(rc)) if rc else None
            pcm = float(np.mean(pc)) if pc else None
            row[f"{scope}_cond"] = round(rcm / pcm, 2) if (rcm and pcm) else None
        nt = total.get((base, "real"), 0)
        ni = len(inert.get((base, "real"), set()))
        row["inert_frac"] = round(ni / nt, 3) if nt else None
        for kind in ("real", "placebo"):
            t = total.get((base, kind), 0)
            row[f"early_{kind}"] = round(early.get((base, kind), 0) / t, 4) if t else None
        rows.append(row)
    return {"rows": rows}


def inert_arm_check(corpus: Path) -> dict:
    """★ How many real arms produced a world identical to factual.

    A real arm that does not diverge did nothing at all, and averaging it in points
    at the wrong conclusion: `L4-barbarians` measured 0.48 -- below its own placebo
    -- purely because half its arms sat before the ruleset's `onsetbarbs` turn and
    were silent no-ops.  Any non-zero rate here means an operator is inert in some
    regime and its effect size is being diluted by empty interventions.

    Detected without the savegames: an arm whose `resolved` equals the factual
    values at every horizon it reached did not move the world.
    """
    inert: Counter = Counter()
    total: Counter = Counter()
    for wd in iter_worlds(corpus):
        bp = wd / "branch.jsonl.gz"
        if not bp.exists():
            continue
        for g in load_jsonl(bp):
            fact = g.get("factual") or {}
            for arm in g["arms"]:
                if arm.get("status") != "OK" or arm.get("placebo_of"):
                    continue
                res = arm.get("resolved") or {}
                shared = [h for h in res if h in fact]
                if not shared:
                    continue
                total[arm["arm"]] += 1
                if all(res[h] == fact[h] for h in shared):
                    inert[arm["arm"]] += 1
    return {"by_op": {k: f"{inert[k]}/{total[k]}" for k in sorted(total) if inert[k]},
            "total_inert": sum(inert.values()), "total_arms": sum(total.values())}


def empty_arm_check(corpus: Path) -> dict:
    """Reload-vs-reload determinism, per branch group.

    The paired design needs exactly one property: two identical reloads of a
    checkpoint agree at the turns horizons are read from.  `factual_check` asserts
    it per world, and the turns compared are restricted to resolution turns --
    an earlier version compared every saved turn and flagged eight worlds that
    agreed everywhere except the terminal game-over save, which no horizon uses.
    """
    c = Counter()
    turns: list[int] = []
    for d in iter_worlds(corpus):
        bp = d / "branch.jsonl.gz"
        if not bp.exists():
            continue
        for g in load_jsonl(bp):
            v = g.get("determinism_ok")
            c["identical" if v is True else
              ("DIVERGED" if v is False else "not checked")] += 1
            turns += g.get("determinism_diverged_turns") or []
    out = dict(c)
    if turns:
        out["diverged_turns"] = sorted(set(turns))[:12]
    return out


# --------------------------------------------------------------------------
# 4. ★ probe shortcut baseline
# --------------------------------------------------------------------------

def probe_baseline(corpus: Path, target: str = "trait", sample: int = 0) -> dict:
    """Least-squares R^2 from report-visible features to a probe target.

    This is the number a learned probe has to beat.  If "cities grew fast"
    already predicts the Expansionist trait, then a probe reporting high R^2 has
    demonstrated nothing about the model's internal state -- which is exactly the
    trap §1.2 of the probe design doc warns about.
    """
    X, Y, names = [], [], None
    for d in iter_worlds(corpus, sample):
        first = last = None
        for r in stream_jsonl(d / "world.jsonl.gz"):
            if first is None:
                first = r
            last = r
        if last is None:
            continue
        pr = last.get("probes") or {}
        for p in last["state"]["players"]:
            civ = p["name"]
            tr = (pr.get("A_static", {}).get(civ) or {}).get("traits")
            if not tr:
                continue
            if names is None:
                names = sorted(tr)
            p0 = next((q for q in first["state"]["players"] if q["name"] == civ), None)
            # Everything here is readable off the reports: rosters, sizes, techs.
            X.append([
                len(p["cities"]),
                sum(c["size"] for c in p["cities"]),
                len(p["units"]),
                len(p["techs"]),
                p["gold"], p["score"], p["culture"],
                len(p["cities"]) - (len(p0["cities"]) if p0 else 0),
                sum(1 for v in p["diplomacy"].values() if v == "War"),
            ])
            Y.append([tr[k]["effective"] for k in names])
    if len(X) < 6 or names is None:
        return {"note": f"not enough worlds for a baseline (n={len(X)})"}
    A = np.array(X, dtype=float)
    A = np.hstack([A, np.ones((len(A), 1))])
    B = np.array(Y, dtype=float)
    coef, *_ = np.linalg.lstsq(A, B, rcond=None)
    pred = A @ coef
    ss_res = ((B - pred) ** 2).sum(axis=0)
    ss_tot = ((B - B.mean(axis=0)) ** 2).sum(axis=0)
    r2 = 1 - ss_res / np.where(ss_tot == 0, np.nan, ss_tot)
    return {"target": target, "n": len(X), "dims": names,
            "r2_from_report_features": {k: round(float(v), 3)
                                        for k, v in zip(names, r2)},
            "note": "a learned probe must beat these to mean anything"}


# --------------------------------------------------------------------------
# 5. splits and coverage
# --------------------------------------------------------------------------

def world_split(corpus: Path) -> dict:
    """Split membership from the first record of every world -- exhaustive, and
    cheap because it stops after one line per shard."""
    by_split: dict[str, set[str]] = defaultdict(set)
    for d in sorted(corpus.iterdir()):
        sh = d / "world.jsonl.gz"
        if not sh.exists():
            continue
        for r in stream_jsonl(sh):
            by_split[r.get("split", "?")].add(r["world_id"])
            break
    leaks = []
    keys = sorted(by_split)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            inter = by_split[a] & by_split[b]
            if inter:
                leaks.append({"a": a, "b": b, "n": len(inter)})
    return {"counts": {k: len(v) for k, v in by_split.items()},
            "leaks": leaks, "ok": not leaks}


def coverage(corpus: Path, sample: int = 0) -> dict:
    ev = Counter()
    arm_status = Counter()
    n_rec = n_world = 0
    total_bytes = 0
    tmin, tmax = 10**9, -1
    # Every world's shard size and record count come from meta.json -- no need to
    # decompress 90 GB to count lines.
    all_bytes = 0
    all_recs = 0
    for d in sorted(corpus.iterdir()):
        m = d / "meta.json"
        sh = d / "world.jsonl.gz"
        if m.exists() and sh.exists():
            all_bytes += sh.stat().st_size
            try:
                all_recs += int(json.loads(m.read_text())["records"])
            except Exception:
                pass
    for d in iter_worlds(corpus, sample):
        n_world += 1
        shard = d / "world.jsonl.gz"
        total_bytes += shard.stat().st_size
        for r in stream_jsonl(shard):
            n_rec += 1
            tmin, tmax = min(tmin, r["t"]), max(tmax, r["t"])
            for e in r["events"]:
                ev[e["type"]] += 1
    # arm statuses over EVERY branch world -- cheap and not sampleable
    for d in sorted(corpus.iterdir()):
        bp = d / "branch.jsonl.gz"
        if bp.exists():
            for g in stream_jsonl(bp):
                for a in g["arms"]:
                    arm_status[f"{a['arm']}:{a.get('status')}"] += 1
    na = Counter(k.split(":")[0] for k, v in arm_status.items()
                 if k.endswith(":NA") for _ in range(v))
    return {"worlds": n_world, "records": n_rec,
            "all_worlds": len([1 for d in corpus.iterdir() if (d / "meta.json").exists()]),
            "all_records": all_recs, "all_bytes": all_bytes,
            "sampled": bool(sample),
            "bytes": total_bytes,
            "bytes_per_record": int(total_bytes / n_rec) if n_rec else 0,
            "turn_range": (tmin, tmax) if tmax >= 0 else None,
            "events": dict(ev.most_common()),
            "events_per_record": round(sum(ev.values()) / n_rec, 2) if n_rec else 0,
            "arm_status": dict(Counter(k.split(":")[1]
                                       for k in arm_status.elements())),
            "ops_not_applicable": dict(na)}


# --------------------------------------------------------------------------
# cli
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="corpus dir")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--roundtrip_rate", type=float, default=0.1)
    ap.add_argument("--sample_worlds", type=int, default=200,
                    help="worlds sampled for the corpus-wide statistics "
                         "(coverage, predictability, probe baseline, round-trip). "
                         "0 = every world. Branch-based checks always use all.")
    args = ap.parse_args()
    corpus = Path(args.data)
    if (corpus / "corpus").exists():
        corpus = corpus / "corpus"

    print(f"[validate] {corpus}")
    cov = coverage(corpus, args.sample_worlds)
    print(f"\n-- coverage --")
    print(f"  CORPUS TOTAL: {cov['all_worlds']} worlds, {cov['all_records']} records, "
          f"{cov['all_bytes']/1e9:.2f} GB")
    print(f"  sampled for the statistics below: {cov['worlds']} worlds, "
          f"{cov['records']} records ({cov['bytes_per_record']/1024:.1f} KB/record)")
    print(f"  turns {cov['turn_range']}, {cov['events_per_record']} events/record")
    print(f"  events: {cov['events']}")
    if cov["arm_status"]:
        print(f"  arms: {cov['arm_status']}")
    if cov["ops_not_applicable"]:
        print(f"  operators not applicable: {cov['ops_not_applicable']}")

    sp = world_split(corpus)
    print(f"\n-- splits --\n  {sp['counts']}  leak={'NO' if sp['ok'] else sp['leaks']}")

    rt = render_roundtrip(corpus, args.roundtrip_rate, sample=args.sample_worlds)
    print(f"\n-- render roundtrip (the corpus stores no text) --")
    print(f"  checked {rt['checked']} records: "
          f"{'ALL BYTE-IDENTICAL' if rt['ok'] else 'MISMATCHES ' + str(rt['mismatches'])}")

    pr = predictability(corpus, sample=args.sample_worlds)
    if pr and "climatology_mse" in pr:
        print(f"\n-- predictability of {pr['var']} "
              f"(climatology MSE {pr['climatology_mse']}) --")
        for s, v in pr["strides"].items():
            flag = "TOO EASY" if (v["frac_of_climatology"] or 1) < 0.02 else "ok"
            print(f"  stride {s:>3}: persistence MSE {v['persistence_mse']:>9} "
                  f"= {v['frac_of_climatology']*100:>6.2f}% of climatology  [{flag}]")

    ea = empty_arm_check(corpus)
    if ea:
        print(f"\n-- reload determinism at resolution turns (P-6) --\n  {ea}")
        if ea.get("DIVERGED"):
            print("  ⚠️ FAILED: paired arms are not comparable in those worlds")

    ia = inert_arm_check(corpus)
    print(f"\n-- ★ inert real arms (world identical to factual = the operator did "
          f"nothing) --")
    print(f"  {ia['total_inert']}/{ia['total_arms']} arms"
          + (f"  ⚠️ {ia['by_op']}" if ia["by_op"] else "  -> none, good"))

    es = effect_size(corpus)
    if es["rows"]:
        print(f"\n-- effect size: real vs matched placebo (distance from factual) --")
        print(f"  {'operator':<20} {'L':<3} {'h':>4} {'real':>8} {'placebo':>8} "
              f"{'ratio':>6} {'n':>3}")
        for r in es["rows"]:
            print(f"  {r['op']:<20} {r['layer']:<3} {r['horizon']:>4} "
                  f"{str(r['real']):>8} {str(r['placebo']):>8} "
                  f"{str(r['ratio']):>6} {r['n']:>3}")
        if es.get("ended_early"):
            print(f"  arms that ended before their horizon (world resolved -- "
                  f"a large effect, and it truncates that operator's own "
                  f"far-horizon average): {es['ended_early']}")
        # Judge each operator inside the window it claims to act in.
        # Flagging any operator with a single sub-1.2 horizon condemned 15 of 17,
        # including L4-love (ratio 3-4.8 throughout) -- because every operator has
        # some far horizon where path noise dominates.  That is the finding, not
        # a defect: L1 is *designed* to be absorbed.
        from .interventions import BY_ID
        best: dict[str, tuple] = {}
        for r in es["rows"]:
            op = BY_ID.get(r["op"])
            if op is None or r["ratio"] is None:
                continue
            lo, hi = op.horizon_band
            if not (lo <= r["horizon"] <= hi):
                continue
            cur = best.setdefault(r["op"], [0.0, 0.0, 0])
            cur[0] += r["ratio"]
            cur[1] = max(cur[1], r["ratio"])
            cur[2] += 1
        print(f"\n  within each operator's declared attribution window:")
        print(f"  {'operator':<20} {'band':<12} {'mean':>6} {'peak':>6}  verdict")
        for opid in sorted(best, key=lambda k: -best[k][0] / max(1, best[k][2])):
            tot, peak, n = best[opid]
            mean = tot / max(1, n)
            v = ("strong" if mean >= 2.0 else
                 "usable" if mean >= 1.3 else "DROP -- ~= its placebo")
            band = str(list(BY_ID[opid].horizon_band))
            print(f"  {opid:<20} {band:<12} {mean:>6.2f} {peak:>6.2f}  {v}")

    te = targeted_effect(corpus)
    if te["rows"]:
        dist = [r for r in te["rows"] if r.get("score_by") != "termination"]
        term = [r for r in te["rows"] if r.get("score_by") == "termination"]
        print(f"\n-- ★ targeted effect: scored on each operator's own variables, "
              f"inside its own attribution window --")
        print(f"  {'operator':<20}{'L':<4}{'tgt':>7}{'tgt|act':>9}{'world':>7}"
              f"{'wld|act':>9}{'inert':>7}  variables")
        for r in sorted(dist, key=lambda x: -(x.get("target_cond") or x["target"] or 0)):
            def f2(k):
                v = r.get(k)
                return f"{v:.2f}" if v is not None else "-"
            inf = f"{r['inert_frac']*100:.0f}%" if r.get("inert_frac") is not None else "-"
            print(f"  {r['op']:<20}{r['layer']:<4}{f2('target'):>7}"
                  f"{f2('target_cond'):>9}{f2('world'):>7}{f2('world_cond'):>9}"
                  f"{inf:>7}  {r['vars']}")
        print("  tgt/world = real-arm distance / matched-placebo distance.")
        print("  '|act' columns drop every PAIR where either arm was inert, so they")
        print("  answer 'how strong when it bites' rather than mixing in the no-ops.")
        print("  inert = share of real arms that left the world identical to factual.")
        if term:
            # These operators are indistinguishable from their placebo on the
            # resolved variables and are kept for a different outcome entirely:
            # how often the world stops before the horizon.  Reporting them in the
            # distance table would read as a failing operator.
            print(f"\n-- operators scored on world termination instead of distance --")
            print(f"  {'operator':<20}{'L':<4}{'early(real)':>12}"
                  f"{'early(plac)':>12}{'ratio':>8}")
            for r in term:
                a, b = r.get("early_real"), r.get("early_placebo")
                ra = f"{a/b:.1f}x" if (a and b) else "-"
                print(f"  {r['op']:<20}{r['layer']:<4}"
                      f"{(f'{a*100:.1f}%' if a is not None else '-'):>12}"
                      f"{(f'{b*100:.1f}%' if b is not None else '-'):>12}{ra:>8}")

    if args.full:
        pb = probe_baseline(corpus, sample=args.sample_worlds)
        print(f"\n-- probe shortcut baseline --")
        print(f"  {pb}")


if __name__ == "__main__":
    main()
