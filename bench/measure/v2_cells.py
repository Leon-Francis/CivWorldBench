"""v2 bench: 1b base rates per (template, budget, h) and 1a normalisation constants,
measured on the benchmark corpus itself (09 §2.4, §1.1 / T1, §8.7: a single set,
the base rates are measured on it directly).

Writes bench/v2/cells_1b.json, bench/v2/norms.json, bench/v2/anchors.json.

    .venv/bin/python bench/measure/v2_cells.py --corpus data_v2/bench --jobs 40
"""
import argparse, collections, gzip, json, pathlib, sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "bundle")); sys.path.insert(0, str(ROOT / "tools"))
from bench.spec_v2 import HZ_BY_BUDGET, VARS, derive  # noqa: E402
from bench.truth_v2 import alive, items_1b, mil_of, player  # noqa: E402


def load_recs(wd: pathlib.Path, need: set[int]) -> dict:
    recs = {}
    with gzip.open(wd / "world.jsonl.gz", "rt") as f:
        for l in f:
            d = json.loads(l)
            if d["t"] in need:
                recs[d["t"]] = d
            if d["t"] > max(need):
                break
    return recs


def one(args):
    wd, budget = args
    wd = pathlib.Path(wd)
    meta = json.loads((wd / "meta.json").read_text())
    anchors = sorted(int(pathlib.Path(c).stem[1:]) for c in meta.get("branch_checkpoints") or [])
    hz = list(HZ_BY_BUDGET[budget])
    if not anchors:
        return {"world": wd.name, "anchors": [], "cells": {}, "vals": {}}
    need = set()
    for t1 in anchors:
        need |= set(range(t1, t1 + max(hz) + 1))
    recs = load_recs(wd, need)
    cells = collections.defaultdict(lambda: [0, 0]); vals = collections.defaultdict(list)
    for t1 in anchors:
        if t1 not in recs:
            continue
        civs = [p["name"] for p in alive(recs[t1])]
        mil = mil_of(recs[t1])
        for it in items_1b(recs, t1, hz, civs):
            # ★ 2026-09-14：键里必须带 t1。原键 (模板, 预算, h) 把开局与中后期池在一起，
            # 而事件率天差地别：t1=60 时 `newcity` 0.916 / `govchange` 0.968 / `moreenemies` 0.913，
            # 其余锚点分别是 0.173 / 0.492 / 0.323。平均偏差在首锚点上是 0.295，
            # 于是 1b 的参照线在 25% 的格上根本没对齐（见 10_论文写作备忘 §3.16）。
            # t1 每个预算下固定 4 个值（400: 60/145/230/315，800: 60/245/430/615，
            # 1200: 60/345/630/915），所以按精确 t1 分组即可，不需要分箱。
            k = f"{it['kind']}|{budget}|{it['h']}|{t1}"
            cells[k][1] += 1
            if it["kind"] == "leader":
                # multi-class: "base rate" = share won by the t1 leader (persistence)
                cells[k][0] += int(it["truth"] == it["civs"][0])
            else:
                cells[k][0] += int(bool(it["truth"]))
        for h in hz:
            r = recs.get(t1 + h)
            if not r:
                continue
            for c in civs:
                p = player(r, c)
                if p and p.get("is_alive"):
                    for v, x in derive(p, mil, r).items():
                        vals[v].append(x)
                        vals[f"{budget}|{h}|{v}"].append(x)          # climatology cell
    return {"world": wd.name, "budget": budget, "anchors": anchors,
            "cells": {k: v for k, v in cells.items()}, "vals": {k: v for k, v in vals.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data_v2/bench")
    ap.add_argument("--jobs", type=int, default=40)
    a = ap.parse_args()
    C = pathlib.Path(a.corpus)
    man = json.load(open(C / "manifest.json"))
    jobs = [(str(C / w["world_id"]), int(w["report_endturn"])) for w in man["worlds"]
            if (C / w["world_id"] / "world.jsonl.gz").exists() and (C / w["world_id"] / "meta.json").exists()]
    print(f"{len(jobs)} worlds")
    cells = collections.defaultdict(lambda: [0, 0]); vals = collections.defaultdict(list); anchors = {}
    with ProcessPoolExecutor(a.jobs) as ex:
        for r in ex.map(one, jobs, chunksize=2):
            anchors[r["world"]] = r["anchors"]
            for k, (y, n) in r["cells"].items():
                cells[k][0] += y; cells[k][1] += n
            for v, xs in r["vals"].items():
                vals[v].extend(xs)
    out = pathlib.Path("bench/v2"); out.mkdir(exist_ok=True)
    cell_rates = {k: {"rate": y / n if n else None, "n": n, "yes": y} for k, (y, n) in sorted(cells.items())}
    json.dump(cell_rates, open(out / "cells_1b.json", "w"), indent=1)
    # 同时写一份 t1 池化的旧口径，供跨版本对账（`score_v2` 找不到带 t1 的键时也回落到它）
    pooled = collections.defaultdict(lambda: [0, 0])
    for k, (y, n) in cells.items():
        tpl, bud, h, _t1 = k.split("|")
        p = pooled[f"{tpl}|{bud}|{h}"]; p[0] += y; p[1] += n
    json.dump({k: {"rate": y / n if n else None, "n": n, "yes": y} for k, (y, n) in sorted(pooled.items())},
              open(out / "cells_1b_pooled.json", "w"), indent=1)
    norms = {}
    # ⚠️ 2026-09-14：`vals` 的键是 `derive()` 产出的**底层量名**（score / culture / n_techs），
    # 而 `VARS` 是**任务变量名**（score_gain / culture_gain / techs_learned）。原来这里写
    # `for v in VARS` 会在三个增量变量上拿到空数组并 IndexError —— 也就是本文件的 norms 分支
    # 自 VARS 改名后一直是坏的（盘上的 norms.json 是更早版本产出的，键正是底层量名）。
    # 改成遍历实际有数据的底层量名。
    LEVELS = [v for v in sorted({k for k in vals if "|" not in k}) if len(vals[v]) > 0]
    for v in LEVELS:
        x = np.array(vals[v], float)
        p1, p99 = np.percentile(x, 1), np.percentile(x, 99)
        rng = p99 - p1
        mag = 10 ** int(np.floor(np.log10(max(rng, 1))))
        norms[v] = {"p1": float(p1), "p99": float(p99), "range": float(rng), "norm": int(np.ceil(rng / mag) * mag), "n": len(x)}
    json.dump(norms, open(out / "norms.json", "w"), indent=1)
    # climatology: unconditional quantiles of the value at t1+h per (budget, h, var) -- the
    # second 1a baseline (09 §1.3); the scorer reads the five quantiles as a forecast
    clim = {}
    for k, xs in vals.items():
        if "|" in k and len(xs) >= 20:
            q = np.percentile(np.array(xs, float), [10, 25, 50, 75, 90])
            clim[k] = {"q": [float(x) for x in q], "n": len(xs)}
    json.dump(clim, open(out / "climatology.json", "w"), indent=1)
    print(f"climatology cells: {len(clim)}")
    json.dump({"corpus": a.corpus, "n_worlds": len(anchors), "n_anchors": sum(len(v) for v in anchors.values()),
               "anchors": anchors}, open(out / "anchors.json", "w"), indent=1)
    print(f"anchors: {sum(len(v) for v in anchors.values())} over {len(anchors)} worlds")
    print("\n1a normalisation (p1-p99 range, rounded):")
    for v in LEVELS:
        print(f"  {v:<18} p1 {norms[v]['p1']:>10.0f}  p99 {norms[v]['p99']:>10.0f}  norm {norms[v]['norm']:>8}")
    # ★ 键带了 t1 之后，10–90% 的筛选必须**逐 t1** 判：同一个 (模板,预算,h) 在不同锚点上
    # 可能一个在窗内、一个超出 90%（`newcity|400|60`：t1=60 时 0.885，t1=315 时 0.077）。
    print("\n1b base rates by (template | budget | h | t1)   x = 落在 10-90% 之外，v3 应剔除:")
    t1s = sorted({int(k.split("|")[3]) for k in cell_rates})
    for kind in sorted({k.split("|")[0] for k in cell_rates}):
        for budget in (400, 800, 1200):
            row = []
            for h in HZ_BY_BUDGET[budget]:
                for t1 in t1s:
                    c = cell_rates.get(f"{kind}|{budget}|{h}|{t1}")
                    if c and c["n"]:
                        flag = "" if 0.10 <= c["rate"] <= 0.90 else "x"
                        thin = "!" if c["n"] < 200 else ""
                        row.append(f"t{t1}/+{h}:{c['rate']:.2f}{flag}{thin}")
            if row:
                print(f"  {kind:<12} {budget:>5}  " + "  ".join(row))


if __name__ == "__main__":
    main()
