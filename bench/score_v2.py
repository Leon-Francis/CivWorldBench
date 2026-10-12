"""v2 scorer: questions/ + raw/ -> per-line records -> stratified readings with
world-block bootstrap.  Never pools across sets (basic / general), budgets or horizons.

1a   normalised CRPS; skill vs persistence and vs climatology; family macro; calibration
1b   Brier per template vs the cell base rate; multiclass Brier for "leader"
1c   component metrics vs copy-forward (score_1c)
2    accuracy per hidden target vs chance (1/3) and vs the run's majority label
3    (A) effect: skill_arm on the operator's score_vars for its target inside the band,
         reference = the model's OWN 1a answer in the same condition ("ignore")
     (B) cleanliness: on (civ, var, h) where the arm truth equals the factual-arm truth,
         how far the counterfactual p50 moved from the model's factual p50 (normalised)
     (C) counterfactual 1b Brier and 1c components against the arm truth, reference =
         the model's factual answers scored against the same arm truth
     all reported for real vs placebo, headline = paired real - placebo difference

    .venv/bin/python bench/score_v2.py --run v2-bench --bootstrap 2000
"""
from __future__ import annotations

import argparse
import collections
import gzip
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pathlib
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "bundle")); sys.path.insert(0, str(ROOT / "tools"))
from bench.parse_v2 import parse_binary, parse_multi, parse_quantiles, parse_task2, split_task3   # noqa: E402
from bench.score_1c import HEADLINE, components, skill as skill_1c                                # noqa: E402
from bench.spec_v2 import FAMILIES, PARAM_DISPLAY, TRAITS, VARS                                   # noqa: E402

TAUS = (0.10, 0.25, 0.50, 0.75, 0.90)
NORMS = {k: v["norm"] for k, v in json.load(open(ROOT / "bench/v2/norms.json")).items()} if (ROOT / "bench/v2/norms.json").exists() else {}
CLIM = json.load(open(ROOT / "bench/v2/climatology.json")) if (ROOT / "bench/v2/climatology.json").exists() else {}
CELLS = json.load(open(ROOT / "bench/v2/cells_1b.json")) if (ROOT / "bench/v2/cells_1b.json").exists() else {}
_CELLS_POOLED = (json.load(open(ROOT / "bench/v2/cells_1b_pooled.json"))
                 if (ROOT / "bench/v2/cells_1b_pooled.json").exists() else {})
TECH_CEILING = 87       # both rulesets: 87 real technologies; beyond that only Future Technology repeats
MIN_CELL_N = 200        # 低于此样本量的 (模板,预算,h,t1) 格回落到 t1 池化率，见 cell_1b()


_INSAMPLE = {}      # (set, cond, kind, budget, h, t1) -> rate；由 build_insample_cells() 填


def build_insample_cells(run: str, force: bool = False):
    """★ A4b（2026-09-14，我拍板选(i)）：参照线再按 (set, cond) 分开，用**题目集自己的边缘率**。

    为什么必须分开：三个设置的题目人群不同 ——
      * 泛化集是**另外 250 个世界**（一半 classic、配置外扩），事件率天然不同；
      * 自我视角与基础集同一批世界，但**只问视角文明见过的文明**（多半是邻居，更容易打起来），
        边缘率也不同。
    实测共用一条率的残余错配是 0.030（泛化集）~0.042（基础集/自我视角）。

    为什么只能内样本：「问哪些文明」是**出题时**才决定的，语料里没有这个信息，
    所以自我视角的率只能在题目集自己身上数。这就是标准的 climatology 基线做法
    （气象上的气候学也是从同一份历史里数的），但**论文必须声明**：
    「1b 的参照线是逐设置的内样本边缘率」。它让参照线更准、因此对模型更严
    （实测增益比共用一条时再低 0.002–0.004）。

    缓存到 bench/v2/cells_1b_insample_<run>.json。
    """
    global _INSAMPLE
    cache = ROOT / f"bench/v2/cells_1b_insample_{run}.json"
    if cache.exists() and not force:
        _INSAMPLE = {tuple(k.split("|")): v for k, v in json.loads(cache.read_text()).items()}
        return _INSAMPLE
    acc = collections.defaultdict(lambda: [0, 0])
    for p_ in sorted((ROOT / "bench/runs" / run / "questions").glob("*.jsonl.gz")):
        for l in gzip.open(p_, "rt"):
            r = json.loads(l)
            if r.get("task") != "1b":
                continue
            for it, y in zip(r["items"], r["truth"]):
                if it["kind"] == "leader" or y is None:
                    continue
                k = (r["set"], r["cond"], it["kind"], str(r["budget"]), str(it["h"]), str(r["t1"]))
                acc[k][0] += int(bool(y)); acc[k][1] += 1
    _INSAMPLE = {k: {"rate": y / n, "n": n} for k, (y, n) in acc.items() if n}
    cache.write_text(json.dumps({"|".join(k): v for k, v in _INSAMPLE.items()}, indent=1))
    return _INSAMPLE


def cell_1b(kind: str, budget, h, t1, st=None, cond=None):
    """1b 的参照单元格。★ 2026-09-14：键加了 t1。

    原键 `(模板, 预算, h)` 把每个预算下 4 个锚点的事件率池在一起，而开局与中后期天差地别
    （t1=60 时 `newcity` 0.916、`govchange` 0.968、`moreenemies` 0.913；其余锚点 0.173/0.492/0.323）。
    平均 |池化率 − 正确率| 在首锚点上是 **0.295**，其余锚点 0.096 ⇒ 25% 的格参照线根本没对齐，
    而 1b 的全部负值恰好来自这 25%（基础集：首锚点 −0.068 vs 其余 +0.0115）。详见 10 §3.16。
    找不到带 t1 的键时回落到池化键，这样旧的 cells_1b.json 也还能跑。
    """
    if st is not None and _INSAMPLE:
        # ablation conditions (oracle/k1/primer/think) run on 40-60 basic worlds: their own cells are
        # thin, but they ask the SAME questions on the same population as the omniscient main view,
        # so they share the (set, v1) cell (v3, 2026-09-15)
        cond_ref = cond[:2] if cond and cond[:2] in ("v1", "v2") else "v1"     # v1br/v2br (base-rate ablation) share the main cells
        c = _INSAMPLE.get((st, cond_ref, kind, str(budget), str(h), str(t1)))
        if c and c.get("n", 0) >= MIN_CELL_N:
            return c
    c = CELLS.get(f"{kind}|{budget}|{h}|{t1}")
    # ⚠️ 样本量守卫：按 t1 切开后 46/288 格的 n<200，最薄的 `joinwar|400|*|60` 只有 9 个样本
    # （条件模板：t1=60 时很少有文明满足"别人正在打仗"/"有军队可掉"）。用 9 个样本估出的率
    # 是噪声，当参照线只会抬高它自己的 Brier、白送模型分数。n<MIN_CELL_N 时回落到池化率。
    if c is None or (c.get("n") or 0) < MIN_CELL_N:
        c = CELLS.get(f"{kind}|{budget}|{h}") or _CELLS_POOLED.get(f"{kind}|{budget}|{h}") or c
    return c
# ★ 2026-09-14（用户决定）：classic 规则集没有 history 积累，culture 恒为 0（实测 10 个 classic
# 世界 1230/1230 个 (回合,文明) 全为 0；civ2civ3 只有 7.5% 为 0，都在开局）。这些格真值 0、
# 趋势 0、CRPS 0，占着 1/11 的变量位却不贡献分数，且开思考后模型会「推理」出非零值而被罚
# （泛化集 tech_culture 族 −0.14→−0.45）。⇒ 模型照常预测，**评分时跳过**。
_MAN = ROOT / "data_v2/bench/manifest.json"
WORLD_RULESET = ({w["world_id"]: (w.get("ruleset") or "civ2civ3") for w in json.load(open(_MAN))["worlds"]}
                 if _MAN.exists() else {})
DEGENERATE = {("classic", "culture_gain")}


def degenerate(world: str, var: str) -> bool:
    return (WORLD_RULESET.get(world, "civ2civ3"), var) in DEGENERATE


def crps5(q, y):
    s = 0.0
    for t, c in zip(TAUS, q):
        s += (y - c) * t if y >= c else (c - y) * (1 - t)
    return 2 * s / len(TAUS)


def brier(p, y):
    return (p - (1.0 if y else 0.0)) ** 2


from bench.spec_v2 import LOG_VARS, LEVEL_OF, DELTA_VARS, VARS  # noqa: E402


def norm(var):
    return 1.0 if var in LOG_VARS else float(NORMS.get(LEVEL_OF.get(var, var)) or 1.0)


def tf(var, x):
    """Value transform before CRPS: signed log1p for the heavy-tailed variables (user 2026-09-10)."""
    import math
    x = float(x)
    return math.copysign(math.log1p(abs(x)), x) if var in LOG_VARS else x


def trend_forecast(hist: dict, var: str, t1: int, h: int):
    """Trend reference for a task variable: linear extrapolation of the last two observation
    frames (t1-DH -> t1) of the underlying LEVEL quantity, clipped at 0; for the increment
    variables (score_gain, culture_gain, techs_learned) the extrapolated level minus the level
    at t1.  None when a frame is missing (view 2: the viewer never saw that quantity)."""
    lv = LEVEL_OF.get(var, var)
    hist = {int(t): v for t, v in hist.items()}          # JSON keys are strings
    ts = sorted(t for t, v in hist.items() if v and isinstance(v.get(lv), (int, float)))
    if len(ts) < 2 or ts[-1] != t1:
        return None
    a, b = ts[-2], ts[-1]
    va, vb = float(hist[a][lv]), float(hist[b][lv])
    level = max(0.0, vb + (vb - va) / (b - a) * h)
    return level - vb if var in DELTA_VARS else level


def clim_q(budget, h, var):
    c = CLIM.get(f"{budget}|{h}|{var}")
    return c["q"] if c else None


def _1a_lines(q, got, truth, base, budget, ignore=None, other_truth=None, base_pomdp=None, base_hist=None):
    """Per-line 1a records.  `ignore`: the model's factual quantiles (task 3 reference).
    `other_truth`: the factual-arm truth (task 3 cleanliness).

    Reference forecast `crps_ref` (2026-09-10, after the 20% pilot): view 1 = omniscient
    persistence; view 2 = what the viewer can read off its stale report (`base_pomdp`:
    own civ true values, foreign civs' remembered cities / units in sight / embassy fields),
    climatology where the viewer has no value at all.  Scoring view 2 against omniscient
    persistence gave skill -2.9: the model is asked about gold/score/techs it cannot see."""
    recs = []
    for h, per in truth.items():
        for civ, vals in per.items():
            for var, y in (vals or {}).items():
                if not isinstance(y, (int, float)) or isinstance(y, bool):
                    continue
                if degenerate(q.get("world", ""), var):
                    continue                      # classic 世界的 culture_gain：真值恒 0，不计分
                if var == "techs_learned" and base_hist:
                    # review run 2026-09-15: a civ that already knows all 87 technologies at t1 can only
                    # answer 0 (Future Technology repeats are not counted) -- a constant, not a forecast
                    lv = (base_hist.get(civ) or {}).get(str(q["t1"])) or {}
                    if isinstance(lv, dict) and (lv.get("n_techs") or 0) >= TECH_CEILING:
                        continue
                k = (civ, var, int(h)); nz = norm(var); yt = tf(var, y)
                r = {"civ": civ, "var": var, "h": int(h), "y": float(y), "norm": nz}
                b = (base.get(civ) or {}).get(var)
                if isinstance(b, (int, float)):
                    r["crps_pers"] = crps5([tf(var, b)] * 5, yt) / nz
                if base_hist:
                    tr_ = trend_forecast(base_hist.get(civ) or {}, var, int(q["t1"]), int(h))
                    if tr_ is not None:
                        r["crps_trend"] = crps5([tf(var, tr_)] * 5, yt) / nz
                cq = clim_q(budget, int(h), var)
                if cq:
                    r["crps_clim"] = crps5([tf(var, c) for c in cq], yt) / nz
                if base_pomdp is not None:
                    # view 2 (user 2026-09-10): the reference is climatology -- the viewer cannot see the
                    # other civs' gold/score/techs, so omniscient persistence is not a fair yardstick.
                    # `crps_pomdp` (what the viewer can read off its stale report) is kept as a diagnostic.
                    bp = (base_pomdp.get(civ) or {}).get(var)
                    if isinstance(bp, (int, float)):
                        r["crps_pomdp"] = crps5([tf(var, bp)] * 5, yt) / nz
                # headline reference for BOTH views = linear trend of the TRUE history (user
                # 2026-09-11 01:30): one yardstick for every cell, view 2 included -- the viewer cannot
                # compute it for foreign civs, and that gap IS the cost of partial observability
                # (reported separately as the visible/hidden split and the v2/v1 CRPS ratio).
                if "crps_trend" in r:
                    r["crps_ref"] = r["crps_trend"]; r["ref"] = "trend"
                # n_enemies / n_allies (near-constant small integers): accuracy + MAE of the median,
                # model vs trend, instead of a CRPS ratio (user 2026-09-11)
                if var in ("n_enemies", "n_allies"):
                    n_civs = len(q.get("civs") or []) or 2
                    if k in got:
                        r["hit"] = int(round(got[k][2])) == int(y); r["ae"] = abs(got[k][2] - float(y))
                    if base_hist:
                        tr2 = trend_forecast(base_hist.get(civ) or {}, var, int(q["t1"]), int(h))
                        if tr2 is not None:
                            tr2 = min(max(tr2, 0.0), float(n_civs - 1))      # a relation count cannot exceed the other civs
                            r["hit_trend"] = int(round(tr2)) == int(y); r["ae_trend"] = abs(tr2 - float(y))
                if k in got:
                    g = got[k]
                    r["crps"] = crps5([tf(var, c) for c in g], yt) / nz; r["p50"] = g[2]
                    r["cover"] = g[0] <= y <= g[4]; r["below_p10"] = y < g[0]; r["above_p90"] = y > g[4]
                if ignore is not None and k in ignore:
                    r["crps_ignore"] = crps5([tf(var, c) for c in ignore[k]], yt) / nz; r["p50_ignore"] = ignore[k][2]
                if other_truth is not None:
                    yf = ((other_truth.get(h) or {}).get(civ) or {}).get(var)
                    if isinstance(yf, (int, float)):
                        r["y_fac"] = float(yf); r["truth_moved"] = abs(float(yf) - float(y)) > 0
                recs.append(r)
    return recs


def score_anchor(args) -> list[dict]:
    # `fpath` (optional third element): the model's answers to the SAME anchor's factual questions,
    # living in the tasks-1/2 run.  Task 3's reference is the model's own factual prediction, so
    # without it every task-3 metric is None (found by the self-test 2026-09-11: the task-3 question
    # set is a separate run dir, so its own raw has no 1a/1b/1c rows to compare against).
    arguments = tuple(args)
    qpath, rpath = arguments[:2]
    fpath = arguments[2] if len(arguments) > 2 else None
    unparsed_as_wrong = arguments[3] if len(arguments) > 3 else True
    def K(r):    # v3: task-3 rows are (arm, sub-task) prompts and the control arm shares the op id
        return (r["cond"], r["task"], r.get("op"), r.get("h"), r.get("sub"), r.get("arm_kind"))
    Q = {}
    for l in gzip.open(qpath, "rt"):
        r = json.loads(l)
        Q[K(r)] = r
    outs = {}
    if fpath is not None and pathlib.Path(fpath).exists():
        for l in gzip.open(fpath, "rt"):
            o = json.loads(l)
            outs[K(o)] = o
    for l in gzip.open(rpath, "rt"):
        o = json.loads(l)
        outs[K(o)] = o
    if unparsed_as_wrong:
        conditions = {key[0] for key in outs} or {key[0] for key in Q if key[0] in ("v1", "v2")}
        for key, question in Q.items():
            if key[0] in conditions and key not in outs:
                outs[key] = {field: question[field] for field in
                             ("cond", "task", "op", "op_base", "h", "sub", "arm_kind") if field in question}
                outs[key].update(output="", parse_rate=0.0, finish="missing")
    recs = []
    # factual answers per cond, needed by task 3 references
    fact_1a = {k[0]: parse_quantiles(o["output"]) for k, o in outs.items() if k[1] == "1a"}
    fact_1b = {}
    fact_1c = {}
    # v3 control arms (same format as the arm prompts, no intervention): per (cond, op_base, sub)
    ctrl_1a = {(k[0], o.get("op_base") or k[2]): parse_quantiles(o["output"])
               for k, o in outs.items() if k[1] == "3" and k[5] == "control" and k[4] == "1a"}
    ctrl_1b = {}
    for k, o in outs.items():
        if k[1] == "3" and k[5] == "control" and k[4] == "1b":
            q = Q.get(k); n = len(q.get("items") or []) if q else 0
            ctrl_1b[(k[0], o.get("op_base") or k[2])] = (parse_binary(o["output"], n), parse_multi(o["output"], n))
    for (c, t, op, h, *_), o in outs.items():
        if t == "1b":
            q = Q.get((c, t, op, h, None, None))
            if q is None:      # task-3 run: the 1b answer comes from the tasks-1/2 run; count items
                q3 = next((v for (cc, tt, *_), v in Q.items() if cc == c and tt == "3"), None)
                n = len(q3.get("items") or []) if q3 else 0
            else:
                n = len(q["truth"])
            fact_1b[c] = (parse_binary(o["output"], n), parse_multi(o["output"], n))
        if t == "1c":
            fact_1c[(c, h)] = o["output"]
    for key, o in outs.items():
        q = Q.get(key)
        if q is None or q.get("error"):
            continue
        base = {k: q[k] for k in ("world", "set", "tier", "budget", "t1", "cond", "view", "pomdp")}
        base["parse_rate"] = o.get("parse_rate"); base["truncated"] = o.get("finish") == "length"
        t = q["task"]
        if t == "1a":
            got = parse_quantiles(o["output"])
            q2 = Q.get(("v2", "1a", None, None, None, None))
            vis = None
            if q2 and q2.get("base_hist_visible"):
                # all 11 task variables -- n_enemies / n_allies ARE visible for the viewer's own civ
                # (its report has the DIPLOMATIC RELATIONS section) and hidden for foreign civs
                vis = {(civ, var) for civ, hist in q2["base_hist_visible"].items() for var in VARS
                       if trend_forecast(hist or {}, var, int(q2["t1"]), 1) is not None}
            for r in _1a_lines(q, got, q["truth"], q.get("base") or {}, q["budget"], base_pomdp=q.get("base_pomdp"), base_hist=q.get("base_hist")):
                if vis is not None:
                    r["visible_v2"] = (r["civ"], r["var"]) in vis      # v1 rows: cell a POMDP viewer could see; v2 rows: what it saw
                recs.append(dict(base, kind="1a", **r))
        elif t == "1b":
            n = len(q["truth"]); pb = parse_binary(o["output"], n); pm = parse_multi(o["output"], n)
            for i, (item, y) in enumerate(zip(q["items"], q["truth"]), 1):
                r = dict(base, kind="1b", tpl=item["kind"], civ=item["civ"], h=item["h"])
                cell = cell_1b(item["kind"], q["budget"], item["h"], q["t1"], q["set"], q["cond"])
                if item["kind"] == "leader":
                    civs = item["civs"]; r["truth"] = y
                    r["brier_base"] = sum((float(c == civs[0]) - float(c == y)) ** 2 for c in civs)   # persistence: t1 leader
                    if i in pm:
                        r["brier"] = sum((pm[i].get(c, 0.0) - float(c == y)) ** 2 for c in civs)
                else:
                    r["truth"] = bool(y)
                    if cell and cell.get("rate") is not None:
                        r["brier_base"] = brier(cell["rate"], bool(y)); r["rate"] = float(cell["rate"])
                    if i in pb:
                        r["brier"] = brier(pb[i], bool(y)); r["p"] = pb[i]
                recs.append(r)
        elif t == "1c":
            comp = components(o["output"], q["truth"], q["copy_forward"], viewer=q.get("viewer") if q["view"] == 2 else None)
            for k, v in comp.items():
                if not isinstance(v, dict):
                    continue
                recs.append(dict(base, kind="1c", comp=k, h=q["h"], model=v["model"], cf=v["cf"], n=v["n"],
                                 skill=skill_1c(k, v["model"], v["cf"])))
        elif t == "2":
            got = parse_task2(o["output"])
            tr_truth = q["truth"]["traits"] or {}
            if tr_truth and isinstance(next(iter(tr_truth.values())), dict):
                # v3 ranking: one record per (dimension, civ pair with different truth); hit = 1 / 0.5 (tie) / 0
                for dim, vals in tr_truth.items():
                    groups = got["traits"].get(dim)
                    pos = {c: gi for gi, g in enumerate(groups) for c in g} if groups else {}
                    civs = sorted(vals)
                    for i in range(len(civs)):
                        for j in range(i + 1, len(civs)):
                            a_, b_ = civs[i], civs[j]
                            if vals[a_] == vals[b_]:
                                continue
                            y = a_ if vals[a_] > vals[b_] else b_                  # the truly higher one
                            hit = None
                            if a_ in pos and b_ in pos:
                                hit = 0.5 if pos[a_] == pos[b_] else float((pos[a_] < pos[b_]) == (vals[a_] > vals[b_]))
                            recs.append(dict(base, kind="2", target=f"trait_{dim}", civ=f"{a_}|{b_}", y=y,
                                             pred=(None if hit is None else ("tie" if hit == 0.5 else (y if hit == 1 else (b_ if y == a_ else a_)))),
                                             hit=hit, gap=abs(vals[a_] - vals[b_]), chance=0.5))
            else:
                for civ, tr in tr_truth.items():
                    for dim, y in tr.items():
                        p = (got["traits_bins"].get(civ) or {}).get(dim)
                        recs.append(dict(base, kind="2", target=f"trait_{dim}", civ=civ, y=y, pred=p, hit=(None if p is None else p == y)))
            for pair, y in q["truth"]["love"].items():
                p = got["love"].get(pair)
                recs.append(dict(base, kind="2", target="love", civ=pair, y=y, pred=p, hit=(None if p is None else p == y)))
            labels_by_name = {PARAM_DISPLAY[k].lower(): k for k in PARAM_DISPLAY}
            from bench.spec_v2 import PARAM_LABELS
            def _norm_param(pn, lab):
                """review 2026-09-16: 4.5% of disaster answers said `none` / `rare` / `none / rare` and 3% of technology answers
                `technology comes normal(ly)` / `fast` -- unambiguous synonyms of the three allowed labels, counted as wrong before."""
                if lab is None:
                    return None
                l_ = lab.strip().lower().rstrip(".")
                labels = [x.lower() for x in PARAM_LABELS[pn]]
                if l_ in labels:
                    return l_
                if "normal" in l_:
                    return "normal"
                if any(w in l_ for w in ("none", "rare")):
                    return next((x for x in labels if "rare" in x), l_)
                if "fast" in l_ or "frequent" in l_:
                    return next((x for x in labels if "fast" in x or "frequent" in x), l_)
                if "slow" in l_:
                    return next((x for x in labels if "slow" in x), l_)
                return l_
            for pn, ybin in q["truth"]["params"].items():
                disp = PARAM_DISPLAY[pn].lower(); lab = _norm_param(pn, got["params"].get(disp))
                y = PARAM_LABELS[pn][ybin].lower()
                recs.append(dict(base, kind="2", target=f"param_{pn}", civ="*", y=y, pred=lab, hit=(None if lab is None else lab == y)))
        elif t == "3":
            if q.get("arm_kind") == "control":
                continue                                   # reference only, scored through the arms
            sub = q.get("sub")
            cref = q.get("control_ref") or {}
            if sub == "1a":                                # v3: this prompt IS the 1a block
                blocks = {"1a": o["output"]}
            elif sub == "1b":
                blocks = {"1b": o["output"]}
            else:
                blocks = split_task3(o["output"])          # v2 three-block prompt
            got = parse_quantiles(blocks.get("1a", ""))
            if sub and cref.get("1a") == "control":
                ign = ctrl_1a.get((q["cond"], q["op_base"]))
            else:
                ign = fact_1a.get(q["cond"])
            common = dict(base, kind="3", op=q["op_base"], is_placebo=bool(q["is_placebo"]), channel=q.get("channel"),
                          layer=q.get("layer"), target=q.get("op_target"), arm_kind=q.get("arm_kind") or ("placebo" if q["is_placebo"] else "real"),
                          control_ref=cref)
            band = q.get("horizon_band") or [0, 10**9]; sv = set(q.get("score_vars") or [])
            for r in (_1a_lines(q, got, q["truth"]["1a"], q.get("base") or {}, q["budget"], ignore=ign, base_pomdp=q.get("base_pomdp"), base_hist=q.get("base_hist"),
                                other_truth=(q.get("factual_arm") or {}).get("1a")) if "1a" in q["truth"] else []):
                r["in_band"] = band[0] <= r["h"] <= band[1]
                r["scored_var"] = r["civ"] == q.get("op_target") and r["var"] in sv
                if "crps" in r and "crps_ignore" in r and r["crps_ignore"] > 0:
                    r["skill_arm"] = 1 - r["crps"] / r["crps_ignore"]
                if "p50" in r and "p50_ignore" in r:
                    # same transform as the CRPS (log1p for the heavy-tailed vars), else gold alone
                    # decides the number: the self-test measured 207 vs the others' O(0.1)
                    r["shift"] = abs(tf(r["var"], r["p50"]) - tf(r["var"], r["p50_ignore"])) / r["norm"]
                recs.append(dict(common, sub="1a", **r))
            items = q.get("items") or []; t1b = q["truth"].get("1b") or []; n = len(items)
            if n and t1b:
                pb = parse_binary(blocks.get("1b", ""), n); pm = parse_multi(blocks.get("1b", ""), n)
                if sub and cref.get("1b") == "control":
                    fb, fm = ctrl_1b.get((q["cond"], q["op_base"]), ({}, {}))
                else:
                    fb, fm = fact_1b.get(q["cond"], ({}, {}))
                for i, (item, y) in enumerate(zip(items, t1b), 1):
                    if y is None:
                        continue
                    r = dict(common, sub="1b", tpl=item["kind"], civ=item["civ"], h=item["h"])
                    if item["kind"] == "leader":
                        civs = item["civs"]
                        if i in pm:
                            r["brier"] = sum((pm[i].get(c, 0.0) - float(c == y)) ** 2 for c in civs)
                        if i in fm:
                            r["brier_ignore"] = sum((fm[i].get(c, 0.0) - float(c == y)) ** 2 for c in civs)
                    else:
                        if i in pb:
                            r["brier"] = brier(pb[i], bool(y))
                        if i in fb:
                            r["brier_ignore"] = brier(fb[i], bool(y))
                    recs.append(r)
            if q["truth"].get("1c") and blocks.get("1c"):        # v2 only
                cf_pred = fact_1c.get((q["cond"], q.get("h_1c")))
                viewer = q.get("viewer") if q["view"] == 2 else None
                # copy-forward for the arm 1c = the (edited) t1 report is not stored; use the
                # model's factual 1c answer as the "ignore" reference and the mainline copy-forward
                # is not applicable here -> report model vs ignore only
                comp_m = components(blocks["1c"], q["truth"]["1c"], q["truth"]["1c"], viewer=viewer)
                comp_i = components(cf_pred or "", q["truth"]["1c"], q["truth"]["1c"], viewer=viewer) if cf_pred else {}
                for k, v in comp_m.items():
                    if not isinstance(v, dict):
                        continue
                    vi = comp_i.get(k) or {}
                    recs.append(dict(common, sub="1c", comp=k, h=q.get("h_1c"), model=v["model"], ignore=vi.get("model"), n=v["n"]))
    return recs


# --------------------------------------------------------------------------
def boot_mean(vals, blocks, n=2000, seed=0):
    vals = np.asarray(vals, float); blocks = np.asarray(blocks)
    if len(vals) == 0:
        return None
    ub = np.unique(blocks); idx = {b: np.where(blocks == b)[0] for b in ub}
    rng = np.random.default_rng(seed); means = []
    for _ in range(n):
        pick = rng.choice(ub, len(ub), replace=True)
        sel = np.concatenate([idx[b] for b in pick])
        means.append(vals[sel].mean())
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"mean": float(vals.mean()), "lo": float(lo), "hi": float(hi), "n": int(len(vals)), "n_worlds": int(len(ub))}


def boot_ratio(num, den, blocks, n=2000, seed=0):
    """1 − Σnum/Σden，按 block bootstrap。

    ⚠️ 为什么必须有这个函数（2026-09-14）：任务 3 的 `skill_arm = 1 − crps/crps_ignore` 是一个
    **上有界 1、下无界**的逐格比值（`crps_ignore→0` 时 →−∞）。对它取算术平均会被左尾完全支配：
    实测真干预的均值是 −0.562 而中位数是 **+0.048**、胜率 53%，L2-city-kill 更极端
    （均值 −0.392 / 中位 +0.423 / 胜率 75%）。项目自己的 HANDOFF_沙盒实验 §5 就写过
    「重尾的逐样本差要用 block-bootstrap CI + 逐样本胜率 + 中位样本，不要加权均值」，
    这里违反了自己的规则。1a 的族技能一直用的是比值口径（1 − Σ模型/Σ参照），任务 3 也必须一致。
    """
    num = np.asarray(num, float); den = np.asarray(den, float); blocks = np.asarray(blocks)
    if len(num) == 0 or den.sum() <= 0:
        return None
    ub = np.unique(blocks); idx = {b: np.where(blocks == b)[0] for b in ub}
    rng = np.random.default_rng(seed); vals = []
    for _ in range(n):
        pick = rng.choice(ub, len(ub), replace=True)
        sel = np.concatenate([idx[b] for b in pick])
        d = den[sel].sum()
        if d > 0:
            vals.append(1 - num[sel].sum() / d)
    if not vals:
        return None
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return {"mean": float(1 - num.sum() / den.sum()), "lo": float(lo), "hi": float(hi),
            "n": int(len(num)), "n_worlds": int(len(ub))}


def quantity_skill_loss(row, unparsed_as_wrong=True):
    """Return model/reference losses for QNT; an unparsed cell contributes skill -1."""
    if row["var"] in ("n_enemies", "n_allies"):
        model, reference = row.get("ae"), row.get("ae_trend")
    else:
        model, reference = row.get("crps"), row.get("crps_ref")
    if reference is None:
        return None
    if model is None and unparsed_as_wrong:
        model = 2 * reference
    return (model, reference) if model is not None else None


def summarize(recs: list[dict], nboot: int, unparsed_as_wrong=True) -> dict:
    out = {}
    df = collections.defaultdict(list)
    for r in recs:
        df[r["kind"]].append(r)
    # ---- 1a
    a = df.get("1a", [])
    for grp_key in ("set", "cond"):
        pass
    res = {}
    for (st, cond, budget, h, var), rows in _group(a, ("set", "cond", "budget", "h", "var")).items():
        rows = [r for r in rows if "crps" in r]
        if not rows:
            continue
        # skill = 1 - mean(crps) / mean(crps_reference)  (ratio of means; the earlier mean of per-row
        # ratios is unbounded when the reference is nearly exact and produced -3.9 for one family)
        def skill(key):
            rr = [r for r in rows if r.get(key) is not None]
            if not rr:
                return None
            den = float(np.mean([r[key] for r in rr]))
            return float(1 - np.mean([r["crps"] for r in rr]) / den) if den > 0 else None
        refs = collections.Counter(r.get("ref") for r in rows if r.get("ref"))

        def mean_of(key):
            xs = [r[key] for r in rows if r.get(key) is not None]
            return float(np.mean(xs)) if xs else None

        def winrate(key):
            xs = [r["crps"] < r[key] for r in rows if r.get(key) is not None]
            return float(np.mean(xs)) if xs else None
        res[f"{st}|{cond}|{budget}|{h}|{var}"] = {
            "crps": float(np.mean([r["crps"] for r in rows])),
            "crps_pers": mean_of("crps_pers"), "crps_trend": mean_of("crps_trend"), "crps_clim": mean_of("crps_clim"),
            "skill_pers": skill("crps_pers"), "skill_trend": skill("crps_trend"),
            "skill_clim": skill("crps_clim"), "skill_ref": skill("crps_ref"), "skill_pomdp": skill("crps_pomdp"), "ref": dict(refs),
            "win_vs_pers": winrate("crps_pers"), "win_vs_trend": winrate("crps_trend"),
            "log_scale": var in LOG_VARS,
            "acc": mean_of("hit"), "mae": mean_of("ae"), "acc_trend": mean_of("hit_trend"), "mae_trend": mean_of("ae_trend"),
            "n_ref": sum(1 for r in rows if r.get("crps_ref") is not None),
            "coverage": float(np.mean([r["cover"] for r in rows])),
            "below_p10": float(np.mean([r["below_p10"] for r in rows])), "above_p90": float(np.mean([r["above_p90"] for r in rows])),
            "n": len(rows)}
    out["1a_cells"] = res
    # family macro per (set, cond): average per-var skill within family, then across families; bootstrap by world
    fam = {}
    for (st, cond), rows in _group(a, ("set", "cond")).items():
        # per (world, family): skill = 1 - sum(crps) / sum(crps_ref), clipped to [-1, 1]; macro = mean over
        # families then over worlds (bootstrap by world).  Clipping keeps one family (economy: gold at a
        # short horizon barely moves, so persistence is near-exact) from dominating the macro.
        per_world_fam = collections.defaultdict(lambda: collections.defaultdict(lambda: [0.0, 0.0]))
        per_world_sub = collections.defaultdict(lambda: collections.defaultdict(lambda: [0.0, 0.0]))
        for r in rows:
            f = next((fn for fn, vs in FAMILIES.items() if r["var"] in vs), None)
            losses = quantity_skill_loss(r, unparsed_as_wrong)
            if losses is None:
                continue
            model_loss, reference_loss = losses
            per_world_fam[r["world"]][f][0] += model_loss
            per_world_fam[r["world"]][f][1] += reference_loss
            if r.get("visible_v2"):
                per_world_sub[r["world"]][f][0] += model_loss
                per_world_sub[r["world"]][f][1] += reference_loss
        worlds = sorted(per_world_fam)
        MACRO_FAMS = list(FAMILIES)          # six families; score_gain is an increment now, kept as its own family

        def fam_skill(acc):
            return float(np.clip(1 - acc[0] / acc[1], -1.0, 1.0)) if acc[1] > 0 else None
        def macro_of(pw):
            vals = []; blocks = []
            for w in sorted(pw):
                fams = [fam_skill(pw[w][fn]) for fn in MACRO_FAMS if fn in pw[w]]
                fams = [x for x in fams if x is not None]
                if fams:
                    vals.append(np.mean(fams)); blocks.append(w)
            return vals, blocks
        vals, blocks = macro_of(per_world_fam)
        vals_sub, blocks_sub = macro_of(per_world_sub)
        def fam_means(pw):
            out_ = {}
            for fn in FAMILIES:
                xs = [fam_skill(pw[w][fn]) for w in sorted(pw) if fn in pw[w]]
                xs = [x for x in xs if x is not None]
                if xs:
                    out_[fn] = float(np.mean(xs))
            return out_
        by_fam = fam_means(per_world_fam); by_fam_vis = fam_means(per_world_sub)
        # ★ A8（2026-09-14）：族技能是在**族内跨变量池化**的比值，而各变量的归一化常数
        # （bench/v2/norms.json，每变量一个全局常数）决定了它们在池化里的权重。实测极不均衡：
        #   scale: population 85.3% / n_cities 9.0% / city_size_total 5.7%
        #   tech_culture: culture_gain 94.7% / techs_learned 5.3%
        # 也就是「规模族」其实是 population 的分数、「科技文化族」其实是 culture 的分数。
        # military/diplomacy 反而均衡（46/53、46/54）。平凡线用同一套权重，所以模型 vs 平凡线的
        # 对照仍然公平——受影响的是「这个族在测什么」的解释。因此额外报一份**逐变量等权**的宏平均。
        def fam_macro(rs, vis_only=False):
            per_wv = collections.defaultdict(lambda: collections.defaultdict(lambda: [0.0, 0.0]))
            for r in rs:
                if vis_only and not r.get("visible_v2"):
                    continue
                # ⚠️ 必须与 by_family 用同一个量：外交两个变量走 MAE（|p50−y|），其余走 CRPS。
                # 2026-09-14 初版这里对所有变量一律用 crps，等于把外交族偷换成另一个指标，
                # 于是 by_family_macro 的外交族凭空高出 +0.389（+0.273 → +0.662）——那不是权重效应。
                losses = quantity_skill_loss(r, unparsed_as_wrong)
                if losses is None:
                    continue
                num, den = losses
                acc = per_wv[r["world"]][r["var"]]
                acc[0] += num; acc[1] += den
            out_ = {}
            for fn, vs in FAMILIES.items():
                per_world = []
                for w in sorted(per_wv):
                    xs = [fam_skill(per_wv[w][v]) for v in vs if v in per_wv[w]]
                    xs = [x for x in xs if x is not None]
                    if xs:
                        per_world.append(float(np.mean(xs)))       # 族内变量等权
                if per_world:
                    out_[fn] = float(np.mean(per_world))
            return out_
        by_fam_macro = fam_macro(rows); by_fam_macro_vis = fam_macro(rows, True)
        # diplomacy family (n_enemies, n_allies): accuracy / MAE, model vs trend, world bootstrap
        # ★ 2026-09-14：外交族的 MAE 口径（只看 p50）与 CRPS 口径（全分布）给出**相反**的结论：
        # 基础集全知下模型 MAE 技能 +0.34 < 平凡线 +0.41，但 CRPS 技能 +0.79 > 平凡线 +0.75；
        # 自我视角下两种口径都是模型输（MAE −0.27 vs +0.35；CRPS +0.62 vs +0.73）。
        # ⇒「外交族不如什么都不变」这条结论必须限定为**点估计**，且强版本只在自我视角成立。
        # 因此把 CRPS 口径的外交族技能一并存下来。
        _dv = [r for r in rows if r["var"] in ("n_enemies", "n_allies")
               and r.get("crps") is not None and r.get("crps_ref") is not None]
        _dvv = [r for r in _dv if r.get("visible_v2")]
        def _ratio(rr):
            d = sum(r["crps_ref"] for r in rr)
            return float(1 - sum(r["crps"] for r in rr) / d) if d > 0 else None
        dip_crps = {"skill_crps": _ratio(_dv), "n": len(_dv),
                    "skill_crps_visible": _ratio(_dvv), "n_visible": len(_dvv)}
        dip = [r for r in rows if r["var"] in ("n_enemies", "n_allies") and "hit" in r]
        dip_summary = {"acc": boot_mean([float(r["hit"]) for r in dip], [r["world"] for r in dip], nboot),
                       "mae": boot_mean([r["ae"] for r in dip], [r["world"] for r in dip], nboot),
                       "acc_trend": float(np.mean([r["hit_trend"] for r in dip if "hit_trend" in r])) if any("hit_trend" in r for r in dip) else None,
                       "mae_trend": float(np.mean([r["ae_trend"] for r in dip if "ae_trend" in r])) if any("ae_trend" in r for r in dip) else None,
                       "n": len(dip)} if dip else None
        n_all = sum(1 for r in rows if "crps" in r and r["var"] not in ("n_enemies", "n_allies"))
        n_ref = sum(1 for r in rows if "crps" in r and r.get("crps_ref") is not None and r["var"] not in ("n_enemies", "n_allies"))
        fam[f"{st}|{cond}"] = {"diplomacy_crps": dip_crps,                  # 外交族的全分布口径（对照 MAE 口径）
                               "by_family_macro": by_fam_macro,             # A8：族内变量等权
                               "by_family_macro_visible": by_fam_macro_vis,
                               "macro_skill": boot_mean(vals, blocks, nboot),
                               "macro_skill_visible": boot_mean(vals_sub, blocks_sub, nboot),   # cells the POMDP viewer can see (own civ + foreign cities/units)
                               "by_family_visible": by_fam_vis,
                               "score_skill": by_fam.get("overall"),
                               "by_family": by_fam, "diplomacy_acc_mae": dip_summary,
                               "ref_coverage": (n_ref / n_all if n_all else None),
                               "reference": "trend extrapolation of the observation frames (view 2: viewer-visible history only); per-family skill clipped to [-1,1]; n_enemies/n_allies reported as accuracy/MAE"}
    out["1a_family"] = fam
    # POMDP information gap: same (world, t1, civ, var, h) cell answered under v1 and v2 -> CRPS ratio,
    # split by whether the viewer had a visible history for that cell (no external reference needed)
    v1 = {(r["world"], r["t1"], r["civ"], r["var"], r["h"]): r["crps"] for r in a if r["cond"] == "v1" and "crps" in r}
    gap = collections.defaultdict(lambda: [0.0, 0.0, 0])
    for r in a:
        if r["cond"] != "v2" or "crps" not in r:
            continue
        k = (r["world"], r["t1"], r["civ"], r["var"], r["h"])
        if k in v1:
            g = gap[(r["set"], "visible" if r.get("visible_v2") else "hidden")]
            g[0] += r["crps"]; g[1] += v1[k]; g[2] += 1
    out["1a_pomdp_gap"] = {f"{st}|{kind}": {"crps_v2_over_v1": (g[0] / g[1] if g[1] > 0 else None), "n": g[2]} for (st, kind), g in gap.items()}
    # ---- 1b
    b = df.get("1b", []); res = {}
    if unparsed_as_wrong:
        b = [dict(row, brier=2.0 if row["tpl"] == "leader" else 1.0)
             if row.get("brier") is None else row for row in b]
    for (st, cond, budget, tpl, h), rows in _group(b, ("set", "cond", "budget", "tpl", "h")).items():
        rows = [r for r in rows if "brier" in r and "brier_base" in r]
        if not rows:
            continue
        res[f"{st}|{cond}|{budget}|{tpl}|{h}"] = {"brier": float(np.mean([r["brier"] for r in rows])),
                                                  "brier_base": float(np.mean([r["brier_base"] for r in rows])),
                                                  "skill": 1 - float(np.mean([r["brier"] for r in rows])) / max(1e-9, float(np.mean([r["brier_base"] for r in rows]))),
                                                  "n": len(rows)}
    out["1b_cells"] = res
    hb = {}
    for (st, cond), rows in _group(b, ("set", "cond")).items():
        rows = [r for r in rows if "brier" in r and "brier_base" in r]
        hb[f"{st}|{cond}"] = boot_mean([r["brier_base"] - r["brier"] for r in rows], [r["world"] for r in rows], nboot)
    out["1b_headline_brier_gain"] = hb
    # ---- 1c
    c = df.get("1c", []); res = {}
    for (st, cond, budget, h, comp), rows in _group(c, ("set", "cond", "budget", "h", "comp")).items():
        rows = [r for r in rows if r.get("skill") is not None]
        if rows:
            res[f"{st}|{cond}|{budget}|{h}|{comp}"] = {"skill": float(np.mean([r["skill"] for r in rows])),
                                                       "model": float(np.mean([r["model"] for r in rows])),
                                                       "cf": float(np.mean([r["cf"] for r in rows])), "n": len(rows)}
    out["1c_cells"] = res
    hc = {}
    for (st, cond), rows in _group(c, ("set", "cond")).items():
        per_w = collections.defaultdict(lambda: collections.defaultdict(list))
        for r in rows:
            if r.get("skill") is not None and r["comp"] in HEADLINE:
                per_w[r["world"]][r["comp"]].append(r["skill"])
        vals = [np.mean([np.mean(v) for v in d.values()]) for d in per_w.values()]
        hc[f"{st}|{cond}"] = boot_mean(vals, list(per_w.keys()), nboot)
    out["1c_headline"] = hc
    # ---- 2
    t2 = df.get("2", []); res = {}
    if unparsed_as_wrong:
        t2 = [dict(row, hit=0.0) if row.get("hit") is None else row for row in t2]
    for (st, cond, target), rows in _group(t2, ("set", "cond", "target")).items():
        rows = [r for r in rows if r.get("hit") is not None]
        if not rows:
            continue
        pairwise = any(r.get("chance") == 0.5 for r in rows)          # v3 ranking: pairwise accuracy, chance 0.5
        maj = None if pairwise else collections.Counter(r["y"] for r in rows).most_common(1)[0][1] / len(rows)
        res[f"{st}|{cond}|{target}"] = {"acc": boot_mean([float(r["hit"]) for r in rows], [r["world"] for r in rows], nboot),
                                        "chance": 0.5 if pairwise else 1 / 3, "majority": maj, "n": len(rows),
                                        "metric": "pairwise ranking accuracy (ties 0.5)" if pairwise else "3-class accuracy",
                                        "tie_rate": (float(np.mean([r["hit"] == 0.5 for r in rows])) if pairwise else None)}
    out["2"] = res
    # ---- 3
    t3 = [r for r in df.get("3", []) if r.get("arm_kind") != "control"]; res = {}
    for (st, cond, op, plc), rows in _group(t3, ("set", "cond", "op", "is_placebo")).items():
        # ★ 2026-09-14（用户决定）：A 只在**真值确实动了**的格上算。测量发现带内格里 23% 的
        # (score_var, h) 上真臂真值与无干预对照逐位相同（`L4-love` 66%、`L3-fog` 24%），
        # 零效应格上任何非零反应都被罚，于是 A 变成"奖励惯性"。旧口径保留为 A_effect_all_cells
        # 以便对照。判据 `truth_moved` 由 `_1a_lines(other_truth=...)` 逐格写入。
        def _eff(rs, moved_only):
            return [r for r in rs if r.get("sub") == "1a" and r.get("in_band") and r.get("scored_var")
                    and "skill_arm" in r and (r.get("truth_moved") is True if moved_only else True)]
        eff_rows = _eff(rows, True); eff_all = _eff(rows, False)
        missing_effects = ([row for row in rows if row.get("sub") == "1a" and row.get("in_band")
                           and row.get("scored_var") and row.get("truth_moved") is True
                           and row.get("y") is not None and row.get("y_fac") is not None
                           and (row.get("p50") is None or row.get("p50_ignore") is None)]
                          if unparsed_as_wrong else [])
        eff = [r["skill_arm"] for r in eff_rows]
        # ★ A5（2026-09-14）：旧的 B 是 |反事实 p50 − 事实 p50| 的平均偏移，**结构上是坏的**。
        # 判据 truth_moved=False 的意思是「臂真值 == **事实臂**真值」，而模型的事实答案瞄准的是
        # **主线**——事实臂与主线是两个不同的世界（重载后 RNG 立刻分岔）。于是这个指标在拿
        # 「瞄准臂的答案」减「瞄准主线的答案」，把两个世界的 RNG 差异当成幻觉副作用：
        # 实测 raw_perfect（反事实答案 100% 等于臂真值）在这些格上也改了 55.6% 的数、平均偏移
        # **0.173**，而 Flash 思考是 0.193 —— 只高出 0.020。人类那份 0.003 是靠**照抄**事实答案
        # 得到的，比完美作答者还低 ⇒ 旧口径奖励的是「不动」而不是「答对」。
        # 新口径与 A 同形：在**不该受影响**的格上，看反事实答案相对自己的事实答案，
        # 对臂真值的误差是变大还是变小。
        #   +1 = 完美（反事实答案就是臂真值）  0 = 照抄事实答案  <0 = 幻觉出了副作用
        _bl = [r for r in rows if r.get("sub") == "1a" and r.get("truth_moved") is False
               and not r.get("scored_var") and r.get("crps") is not None and r.get("crps_ignore") is not None]
        b_skill = boot_ratio([r["crps"] for r in _bl], [r["crps_ignore"] for r in _bl],
                             [r["world"] for r in _bl], nboot)
        _chg = [r for r in rows if r.get("sub") == "1a" and r.get("truth_moved") is False
                and r.get("p50") is not None and r.get("p50_ignore") is not None]
        clean = [r["shift"] for r in rows if r.get("sub") == "1a" and r.get("truth_moved") is False and "shift" in r]
        moved = [r["shift"] for r in rows if r.get("sub") == "1a" and r.get("truth_moved") is True and "shift" in r]
        all_sk = [r["skill_arm"] for r in rows if r.get("sub") == "1a" and "skill_arm" in r]
        b3 = [r["brier_ignore"] - r["brier"] for r in rows if r.get("sub") == "1b" and "brier" in r and "brier_ignore" in r]
        if unparsed_as_wrong:
            b3.extend(-2.0 if row["tpl"] == "leader" else -1.0 for row in rows
                      if row.get("sub") == "1b" and
                      (row.get("brier") is None or row.get("brier_ignore") is None))
        c3 = collections.defaultdict(list)
        for r in rows:
            if r.get("sub") == "1c" and r.get("model") is not None and r.get("ignore") is not None:
                c3[r["comp"]].append((r["model"], r["ignore"]))
        # ★ 2026-09-14：A 的**主口径改成比值** `1 − Σcrps/Σcrps_ignore`（与 1a 族技能一致），
        # 逐格均值 A_effect_skill_arm 降为对照，并同时给中位数与胜率。理由见 boot_ratio 的 docstring。
        eff_dir = [(r, np.sign((r.get("p50") or 0) - (r.get("p50_ignore") or 0)),
                    np.sign((r.get("y") or 0) - (r.get("y_fac") or 0)))
                   for r in eff_rows if r.get("p50") is not None and r.get("p50_ignore") is not None
                   and r.get("y") is not None and r.get("y_fac") is not None]
        n_mv = sum(1 for _, m, _ in eff_dir if m != 0)
        # ★ 2026-09-21（用户拍板，14 号 B23 补 10/11）：capture 的主口径改成**只看效应**的 S。
        # 旧 A_effect_ratio = 1 − Σcrps/Σcrps_ignore 把「效应对不对」和「对照答案本来准不准」混在一起：
        # 对照误差 |e| 与真效应 |d| 同量级且符号随机，于是 ŝ = d 精确命中也只能拿 +0.17…+0.40，而在对照
        # 错得最狠的格上乱动一大步、恰好往真值挪的线反而最高（GPT-5.6 效应最差 −1.15、旧 C 最高 +0.19）。
        #   S = 1 − Σ|ŝ − d| / Σ|d|，ŝ = tf(p50) − tf(p50_ignore)，d = tf(y) − tf(y_fac)，都除以 norm（与 CRPS 同一空间）
        #   0 = 没动，1 = 每格精确，<0 = 动得比不动还离真位移远；与对照答案的水平无关。上限见 l3_effect_ceiling.py。
        _sd = [(abs((tf(r["var"], r["p50"]) - tf(r["var"], r["p50_ignore"])) / r["norm"]
                    - (tf(r["var"], r["y"]) - tf(r["var"], r["y_fac"])) / r["norm"]),
                abs((tf(r["var"], r["y"]) - tf(r["var"], r["y_fac"])) / r["norm"]), r["world"])
               for r, _m, _t in eff_dir]
        for row in missing_effects:
            effect = abs(tf(row["var"], row["y"]) - tf(row["var"], row["y_fac"])) / row["norm"]
            _sd.append((2 * effect, effect, row["world"]))
        direction_count = n_mv + len(missing_effects)
        res[f"{st}|{cond}|{op}|{'placebo' if plc else 'real'}"] = {
            "A_effect_S": boot_ratio([a for a, _, _ in _sd], [b for _, b, _ in _sd], [w for _, _, w in _sd], nboot),  # ★ 主口径
            "A_effect_ratio": boot_ratio([r["crps"] for r in eff_rows],
                                         [r["crps_ignore"] for r in eff_rows],
                                         [r["world"] for r in eff_rows], nboot),      # 旧主口径（2026-09-14 → 09-21），仅对照
            "A_effect_median": float(np.median(eff)) if eff else None,
            "A_effect_winrate": float(np.mean([v > 0 for v in eff])) if eff else None,
            "A_dir_correct": (sum(1 for _, m, t in eff_dir if m != 0 and m == t) / direction_count) if direction_count else None,
            "A_frac_moved": (n_mv / len(eff_dir)) if eff_dir else None,
            "A_effect_skill_arm": boot_mean(eff, [r["world"] for r in eff_rows], nboot),  # 旧口径（逐格均值，重尾）
            "A_n_moved": len(eff_rows) + len(missing_effects), "A_n_cells": len(eff_all) + len(missing_effects),
            "A_effect_all_cells": boot_mean([r["skill_arm"] for r in eff_all],
                                            [r["world"] for r in eff_all], nboot),   # 旧口径，含零效应格
            "A_all_lines_skill_arm": float(np.mean(all_sk)) if all_sk else None,
            "B_skill_unmoved": b_skill,                        # ★ 新主口径，见上方注释
            "B_frac_touched": (float(np.mean([r["p50"] != r["p50_ignore"] for r in _chg])) if _chg else None),
            "B_n_unmoved": len(_bl),
            "B_clean_shift_unmoved": float(np.mean(clean)) if clean else None,   # 旧口径（结构上有偏，仅对照）
            "B_shift_moved": float(np.mean(moved)) if moved else None,
            "B_n_unmoved": len(clean), "B_n_moved": len(moved),
            "C_1b_brier_gain": float(np.mean(b3)) if b3 else None, "C_1b_n": len(b3),
            "C_1c": {k: {"model": float(np.mean([m for m, _ in v])), "ignore": float(np.mean([i for _, i in v])), "n": len(v)} for k, v in c3.items()},
        }
    out["3"] = res
    from bench.specificity import summarize_specificity
    out["3_specificity"] = summarize_specificity(df.get("3", []), nboot)
    if unparsed_as_wrong:
        response_coverage = collections.defaultdict(collections.Counter)
        for row in df.get("3", []):
            if row.get("sub") == "1a" and row.get("arm_kind") != "control":
                setting = f"{row['set']}|{row['cond']}"
                response_coverage[setting]["total"] += 1
                response_coverage[setting]["valid"] += int(row.get("shift") is not None)
        out["3_response_coverage"] = {
            setting: dict(counts, fraction=counts["valid"] / counts["total"], failure_score=-1.0)
            for setting, counts in sorted(response_coverage.items())
        }
        coverage = collections.defaultdict(collections.Counter)
        for row in recs:
            setting = f"{row['set']}|{row['cond']}"
            if row["kind"] == "1a":
                missing = row.get("crps") is None
                metric = "QNT"
            elif row["kind"] == "1b":
                missing = row.get("brier") is None
                metric = "EVT"
            elif row["kind"] == "2":
                missing = row.get("hit") is None
                metric = row["target"]
            elif row["kind"] == "3" and row.get("sub") == "1a":
                missing = row.get("p50") is None or row.get("p50_ignore") is None
                metric = "INT_quantity"
            else:
                continue
            coverage[setting][f"{metric}_total"] += 1
            coverage[setting][f"{metric}_unparsed"] += int(missing)
        out["unparsed_policy"] = {
            "version": "unparsed-as-wrong-v2",
            "quantity": "Missing loss contributes twice the reference loss before existing family aggregation and clipping; no synthetic prediction.",
            "events": "Missing binary/multiclass Brier loss is 1/2 respectively.",
            "attributes": "Missing hit is zero; retain the pair or label in the denominator.",
            "capture": "Missing arm/control effect contributes twice the absolute true effect to the error numerator.",
            "direction": "Missing arm/control effect counts as an incorrect direction.",
            "specificity": "Weight the observed response-selectivity score by valid response coverage; missing responses receive -1. No prediction shifts are imputed.",
            "diagnostics": "Raw CRPS and calibration still require actual forecasts. Task-3 event comparisons with a missing arm/control receive the minimum gain (-1 binary, -2 multiclass).",
            "coverage": {key: dict(value) for key, value in sorted(coverage.items())},
        }
    return out


def _group(rows, keys):
    g = collections.defaultdict(list)
    for r in rows:
        g[tuple(r.get(k) for k in keys)].append(r)
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="v2-bench")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--jobs", type=int, default=32)
    ap.add_argument("--raw", default="raw", help="subdir with model outputs")
    ap.add_argument("--anchors", default="", help="file with anchor file names (one per line); score only those "
                                                 "-- use it to compare two models on exactly the same anchors")
    ap.add_argument("--tag", default="", help="suffix for the output file name, e.g. _common")
    ap.add_argument("--cells_run", default="", help="take the in-sample 1b reference cells from this run instead of --run (ablation runs built on a subset)")
    failure_policy = ap.add_mutually_exclusive_group()
    failure_policy.add_argument("--unparsed-as-wrong", dest="unparsed_as_wrong", action="store_true",
                                help="count unparseable answer cells as failures (default)")
    failure_policy.add_argument("--skip-unparsed", dest="unparsed_as_wrong", action="store_false",
                                help="reproduce the legacy valid-answer-only policy")
    ap.set_defaults(unparsed_as_wrong=True)
    a = ap.parse_args()
    run = ROOT / "bench/runs" / a.run
    if not any((run / "questions").glob("*.jsonl.gz")):
        ap.error("Dataset: In preparation. This code release contains no benchmark data. Supply a separate local run before scoring.")
    if not all((NORMS, CLIM, CELLS, WORLD_RULESET)):
        ap.error("Scoring requires matching normalization, climatology, event cells, "
                 "and world-to-ruleset calibration inputs; see docs/PROTOCOL.md.")
    # A4b：1b 的参照线 = 逐设置的内样本边缘率（首次调用会扫一遍 questions 并缓存）
    ins = build_insample_cells(a.cells_run or a.run)
    print(f"1b in-sample reference cells: {len(ins)}  (per set/cond/template/budget/h/t1)")
    # ⚠️ 子进程要能看到它：ProcessPoolExecutor 用 fork，模块级全局在 fork 时已填好即可。
    # task-3 runs need the model's factual answers from the tasks-1/2 run (same anchor, same model)
    fac_dir = (ROOT / "bench/runs/v2-bench-t12" / a.raw) if a.run != "v2-bench-t12" else None
    keep = None
    if a.anchors:
        keep = {l.strip() for l in open(a.anchors) if l.strip()}
    pairs = [(q, run / a.raw / q.name, (fac_dir / q.name) if fac_dir else None, a.unparsed_as_wrong)
             for q in sorted((run / "questions").glob("*.jsonl.gz"))
             if (run / a.raw / q.name).exists() and (keep is None or q.name in keep)]
    if fac_dir:
        have = sum(1 for _, _, f, _ in pairs if f and f.exists())
        print(f"factual reference from {fac_dir.relative_to(ROOT)}: {have}/{len(pairs)} anchors")
    print(f"{len(pairs)} anchors with outputs")
    recs = []
    with ProcessPoolExecutor(a.jobs) as ex:
        for r in ex.map(score_anchor, pairs, chunksize=2):
            recs.extend(r)
    with gzip.open(run / f"scored_{a.raw}{a.tag}.jsonl.gz", "wt") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summ = summarize(recs, a.bootstrap, unparsed_as_wrong=a.unparsed_as_wrong)
    (run / f"scores_{a.raw}{a.tag}.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1))
    print("\n===== 主指标 =====")
    for key in sorted(summ["1a_family"]):
        f = summ["1a_family"][key]; m = f["macro_skill"]; ms = f.get("macro_skill_visible"); d = f.get("diplomacy_acc_mae") or {}
        b = summ["1b_headline_brier_gain"].get(key); c = summ["1c_headline"].get(key)
        fams = " ".join(f"{fn}:{x:+.2f}" for fn, x in f["by_family"].items())
        print(f"{key:<20} 1a macro {m['mean']:+.3f} [{m['lo']:+.2f},{m['hi']:+.2f}]"
              + (f"  (visible-cells {ms['mean']:+.3f})" if ms else "")
              + f"  dipl MAE {d.get('mae', {}).get('mean', float('nan')):.2f} (trend {d.get('mae_trend', float('nan')):.2f})"
              + f"  | 1b gain {b['mean']:+.4f} [{b['lo']:+.3f},{b['hi']:+.3f}]" if b else ""
              ) 
        print(f"{'':<20} families {fams}" + (f"  | 1c {c['mean']:+.3f} [{c['lo']:+.3f},{c['hi']:+.3f}]" if c else ""))
    print("POMDP gap (CRPS v2 / v1 on identical cells):", {k: (round(v["crps_v2_over_v1"], 2) if v["crps_v2_over_v1"] else None, v["n"]) for k, v in summ.get("1a_pomdp_gap", {}).items()})
    t2 = collections.defaultdict(dict)
    for key, v in summ["2"].items():
        st_cond, tgt = key.rsplit("|", 1); t2[st_cond][tgt] = (v["acc"]["mean"], v["majority"])
    for k, d in sorted(t2.items()):
        print(f"{k:<20} task2 " + " ".join(f"{t.replace('trait_', '').replace('param_', '')}:{a:.2f}({(mj if mj is not None else float('nan')):.2f})" for t, (a, mj) in d.items()))
    print("\n1a family macro skill vs trend:", {k: (round(v["macro_skill"]["mean"], 3) if v["macro_skill"] else None) for k, v in summ["1a_family"].items()})
    print("1b headline Brier gain:", {k: (round(v["mean"], 4) if v else None) for k, v in summ["1b_headline_brier_gain"].items()})
    print("1c headline:", {k: (round(v["mean"], 3) if v else None) for k, v in summ["1c_headline"].items()})
    print("task 2:", {k: round(v["acc"]["mean"], 3) for k, v in summ["2"].items()})
    for k, v in summ["3"].items():
        e = v["A_effect_skill_arm"]
        print(f"task 3 {k}: A={e['mean']:+.3f} (n={e['n']})" if e else f"task 3 {k}: A=None",
              f"| B unmoved shift={v['B_clean_shift_unmoved']} moved={v['B_shift_moved']} | C 1b gain={v['C_1b_brier_gain']}")
    print("wrote", run / f"scores_{a.raw}{a.tag}.json")


if __name__ == "__main__":
    main()
