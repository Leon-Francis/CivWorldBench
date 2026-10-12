"""Ideal-observer features and ridge regression for the four AI traits.

Extracts behavioral statistics over an 80-turn window from a caller-supplied
corpus. The task-2 ceiling analysis imports NAMES, feats, ridge, and pred from
this module. No corpus records, fitted weights, or measured results are bundled.

Standalone usage with independently supplied inputs:
  python bench/measure/pilot_ideal_observer.py --corpus path/to/corpus \
      --anchors 81,121,161,201,241,281,321,361 --folds 6
"""
import argparse, collections, gzip, json, pathlib, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from bench_report import unit_is_military  # noqa: E402

NAMES = ("Aggressive", "Expansionist", "Trader", "Builder")
CARAVANS = {"Caravan", "Freight"}
MIL = None


def feats(world_dir: str, anchors: list[int]):
    global MIL
    if MIL is None:
        MIL = unit_is_military("civ2civ3")
    need = set()
    for t1 in anchors:
        need |= {t1 - 80, t1 - 40, t1}
    recs, wars_by = {}, collections.defaultdict(int)
    built_by = collections.defaultdict(int)     # (player, t, "b"|"u") -> count from engine events
    cf_by = collections.defaultdict(int)        # (player, t) -> ceasefires agreed (either side)
    war_pt = collections.defaultdict(int)       # (player, t) -> number of pairs at war this turn
    for l in gzip.open(f"{world_dir}/world.jsonl.gz", "rt"):
        d = json.loads(l)
        city_owner = {c.get("id"): p["name"] for p in d["state"]["players"] for c in (p.get("cities") or [])}
        for e in d.get("events") or []:
            if e.get("type") == "war_declared" and e.get("player"):
                wars_by[(e["player"], d["t"])] += 1
            elif e.get("type") == "building_built":
                # v2 engine event carries city_id but (until the hook is fixed) no
                # owner: resolve through this frame's city roster.
                owner = e.get("player") or city_owner.get(e.get("city_id"))
                if owner:
                    built_by[(owner, d["t"], "b")] += 1
            elif e.get("type") == "unit_built" and e.get("player"):
                built_by[(e["player"], d["t"], "u")] += 1
            elif e.get("type") == "ceasefire_agreed":
                for who in (e.get("player"), e.get("other")):
                    if who:
                        cf_by[(who, d["t"])] += 1
        for p in d["state"]["players"]:
            if p.get("is_alive"):
                war_pt[(p["name"], d["t"])] = sum(1 for v in (p.get("diplomacy") or {}).values() if "War" in str(v))
        if d["t"] in need:
            recs[d["t"]] = d
        if d["t"] > max(need):
            break
    tr = {}
    for d in recs.values():
        for c, a in (d["probes"].get("A_static") or {}).items():
            if a and a.get("traits") and (a.get("ai_barb_type") or "None") == "None":
                tr[c] = {k: v["effective"] for k, v in a["traits"].items()}
    rows = []
    for t1 in anchors:
        fr = [recs.get(t1 - 80), recs.get(t1 - 40), recs.get(t1)]
        if any(f is None for f in fr):
            continue
        for civ in tr:
            ps = [next((p for p in f["state"]["players"] if p["name"] == civ), None) for f in fr]
            if any(p is None or not p.get("is_alive") for p in ps):
                continue

            def wh(f):
                for g in f["probes"].values():
                    if isinstance(g, dict) and isinstance(g.get(civ), dict) and "units_built" in g[civ]:
                        return g[civ]
                return {}
            w0, w2 = wh(fr[0]), wh(fr[2])

            def utype(u):
                return u.get("type") or u.get("name")

            def mil_n(p):
                return sum(1 for u in p["units"] if MIL.get(utype(u), False))

            def car_n(p):
                return sum(1 for u in p["units"] if utype(u) in CARAVANS)

            def prod_mil(p):
                c = p["cities"]
                return (sum(1 for x in c if MIL.get(x.get("production"), False)) / len(c)) if c else 0

            def prod_car(p):
                c = p["cities"]
                return (sum(1 for x in c if x.get("production") in CARAVANS) / len(c)) if c else 0

            def rel(p, word):
                return sum(1 for v in (p.get("diplomacy") or {}).values() if word in str(v))

            def nimp(p):
                return sum(sum(1 for v in (c.get("improvements") or {}).values() if v) for c in p["cities"])
            p0, p1, p2 = ps
            decl = sum(v for (c, t), v in wars_by.items() if c == civ and t1 - 80 < t <= t1)
            nb = sum(v for (c, t, k), v in built_by.items() if c == civ and k == "b" and t1 - 80 < t <= t1)
            nu = sum(v for (c, t, k), v in built_by.items() if c == civ and k == "u" and t1 - 80 < t <= t1)
            build_ratio = nb / max(1, nb + nu)
            ncf = sum(v for (c, t), v in cf_by.items() if c == civ and t1 - 80 < t <= t1)
            war_share = sum(war_pt.get((civ, t), 0) for t in range(t1 - 79, t1 + 1)) / 80.0   # mean # enemies over window
            x = [len(p2["cities"]) - len(p0["cities"]), len(p2["cities"]),
                 len(p2["units"]) - len(p0["units"]), mil_n(p2) - mil_n(p0), mil_n(p2) / max(1, len(p2["units"])),
                 prod_mil(p2), prod_mil(p1), len(p2["techs"]) - len(p0["techs"]),
                 p2["gold"] - p0["gold"], p2["score"] - p0["score"], p2["culture"] - p0["culture"],
                 p2["population"] - p0["population"], rel(p2, "War"), rel(p2, "War") - rel(p0, "War"), rel(p2, "Alliance"),
                 (w2.get("units_built", 0) - w0.get("units_built", 0)), (w2.get("units_killed", 0) - w0.get("units_killed", 0)),
                 (w2.get("units_lost", 0) - w0.get("units_lost", 0)), nimp(p2) - nimp(p0),
                 (p2.get("income") or {}).get("tax_pct", 0), (p2.get("income") or {}).get("science_pct", 0),
                 (p2.get("income") or {}).get("trade", 0), t1, len(fr[2]["state"]["players"]),
                 # v2 additions
                 car_n(p2) / max(1, len(p2["units"])), prod_car(p2), decl, rel(p2, "Peace"),
                 nb, nu, build_ratio, ncf, war_share]
            rows.append((pathlib.Path(world_dir).name, civ, t1, x, [tr[civ][k] for k in NAMES]))
    return rows


def job(a):
    try:
        return feats(*a)
    except Exception as e:  # noqa: BLE001
        print("  !", a[0], e, file=sys.stderr)
        return []


def ridge(X, Y, lam):
    Xb = np.c_[X, np.ones(len(X))]
    A = Xb.T @ Xb + lam * np.eye(Xb.shape[1]); A[-1, -1] -= lam
    return np.linalg.solve(A, Xb.T @ Y)


def pred(X, w):
    return np.c_[X, np.ones(len(X))] @ w


def r2(y, p):
    return 1 - ((y - p) ** 2).sum() / max(1e-9, ((y - y.mean()) ** 2).sum())


def poly(X):
    n = X.shape[1]; iu = np.triu_indices(n, 1)
    return np.concatenate([X, X ** 2, X[:, iu[0]] * X[:, iu[1]]], 1)


def rank(a):
    o = np.argsort(a); r = np.empty(len(a)); r[o] = np.arange(len(a)); return r


def spearman(a, b):
    ra, rb = rank(np.asarray(a, float)), rank(np.asarray(b, float))
    if ra.std() == 0 or rb.std() == 0:
        return np.nan
    return np.corrcoef(ra, rb)[0, 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--anchors", default="81,121,161,201,241,281,321,361")
    ap.add_argument("--folds", type=int, default=6)
    ap.add_argument("--jobs", type=int, default=20)
    ap.add_argument("--tiers", default="", help="comma list to restrict, e.g. S,M")
    a = ap.parse_args()
    anchors = [int(x) for x in a.anchors.split(",")]
    worlds = sorted(str(p.parent) for p in pathlib.Path(a.corpus).glob("*/world.jsonl.gz"))
    if a.tiers:
        keep = set(a.tiers.split(","))
        worlds = [w for w in worlds if pathlib.Path(w).name.split("-")[2] in keep]
    rows = []
    with ProcessPoolExecutor(min(a.jobs, len(worlds))) as ex:
        for r in ex.map(job, [(w, anchors) for w in worlds]):
            rows += r
    wids = sorted({r[0] for r in rows})
    print(f"corpus {a.corpus}: {len(wids)} worlds · {len(rows)} (civ, t1) samples · anchors {anchors}")
    if len(wids) < 3:
        print("too few worlds"); return
    X = np.array([r[3] for r in rows], float); Y = np.array([r[4] for r in rows], float)
    # --- trait range actually present (P2: should span ~10-150, not 30-90)
    print("trait range in corpus:", {n: (int(Y[:, j].min()), int(Y[:, j].max()), round(float(Y[:, j].std()), 1))
                                     for j, n in enumerate(NAMES)})
    # --- Trader surface: caravan share
    car = X[:, 24]; prodc = X[:, 25]
    print(f"caravan share of units: mean {car.mean():.3%} · frac of samples with any caravan {np.mean(car > 0):.1%} · "
          f"cities producing caravans {prodc.mean():.2%}")
    print(f"war declarations per civ per 80 turns: mean {X[:, 26].mean():.2f} · frac>0 {np.mean(X[:, 26] > 0):.1%}")
    # --- leave-world-out K-fold
    folds = [set(wids[i::a.folds]) for i in range(a.folds)]
    wid = np.array([r[0] for r in rows])
    P_lin = np.zeros_like(Y); P_q = np.zeros_like(Y)
    for f in folds:
        te = np.isin(wid, list(f)); trn = ~te
        mu, sd = X[trn].mean(0), X[trn].std(0) + 1e-9
        Xtr, Xte = (X[trn] - mu) / sd, (X[te] - mu) / sd
        Ptr, Pte = poly(Xtr), poly(Xte); pm, ps = Ptr.mean(0), Ptr.std(0) + 1e-9
        Ptr, Pte = (Ptr - pm) / ps, (Pte - pm) / ps
        for j in range(4):
            P_lin[te, j] = pred(Xte, ridge(Xtr, Y[trn, j], 100.0))
            P_q[te, j] = pred(Pte, ridge(Ptr, Y[trn, j], 10000.0))
    print(f"\n{'trait':<14}{'linear R²':>10}{'quad R²':>9}{'pearson r':>11}{'MAE':>7}{'std':>7}{'within-world Spearman':>22}")
    out = {}
    for j, n in enumerate(NAMES):
        by = collections.defaultdict(list)
        for r, p in zip(rows, P_lin[:, j]):
            by[(r[0], r[2])].append((p, r[4][j]))
        rh = [spearman([u for u, _ in v], [w for _, w in v]) for v in by.values() if len(v) >= 3]
        rh = [x for x in rh if x == x]
        r2l, r2q = r2(Y[:, j], P_lin[:, j]), r2(Y[:, j], P_q[:, j])
        pr = np.corrcoef(Y[:, j], P_lin[:, j])[0, 1]
        print(f"{n:<14}{r2l:>10.3f}{r2q:>9.3f}{pr:>11.3f}{np.abs(Y[:, j]-P_lin[:, j]).mean():>7.1f}{Y[:, j].std():>7.1f}"
              f"{(np.median(rh) if rh else float('nan')):>+22.3f}")
        out[n] = {"r2_linear": float(r2l), "r2_quad": float(r2q), "pearson": float(pr),
                  "spearman_within_world_median": float(np.median(rh)) if rh else None}
    # --- the two single-feature sanity correlations the source audit predicts
    print("\nsingle-feature correlations (expected sign if the patch works):")
    print(f"  Expansionist ~ new cities/80t      r = {np.corrcoef(Y[:, 1], X[:, 0])[0, 1]:+.3f}")
    print(f"  Builder      ~ improvements/80t    r = {np.corrcoef(Y[:, 3], X[:, 18])[0, 1]:+.3f}")
    print(f"  Builder      ~ frac cities on mil  r = {np.corrcoef(Y[:, 3], X[:, 5])[0, 1]:+.3f}")
    print(f"  Builder      ~ buildings/(bld+units) built (events) r = {np.corrcoef(Y[:, 3], X[:, 30])[0, 1]:+.3f}"
          f"   (mean ratio {X[:, 30].mean():.3f}, buildings/80t {X[:, 28].mean():.1f}, units/80t {X[:, 29].mean():.1f})")
    print(f"  Trader       ~ caravan share       r = {np.corrcoef(Y[:, 2], X[:, 24])[0, 1]:+.3f}")
    print(f"  Aggressive   ~ war declarations    r = {np.corrcoef(Y[:, 0], X[:, 26])[0, 1]:+.3f}")
    print(f"  Aggressive   ~ n at war            r = {np.corrcoef(Y[:, 0], X[:, 12])[0, 1]:+.3f}")
    print(f"  Aggressive   ~ mean enemies/80t     r = {np.corrcoef(Y[:, 0], X[:, 32])[0, 1]:+.3f}")
    print(f"  Aggressive   ~ ceasefires agreed    r = {np.corrcoef(Y[:, 0], X[:, 31])[0, 1]:+.3f}   (mean {X[:, 31].mean():.2f}/80t)")
    print(f"  Aggressive   ~ military unit share  r = {np.corrcoef(Y[:, 0], X[:, 4])[0, 1]:+.3f}")
    print(f"  Aggressive   ~ units killed/80t     r = {np.corrcoef(Y[:, 0], X[:, 16])[0, 1]:+.3f}")
    print(f"  Aggressive   ~ units lost/80t       r = {np.corrcoef(Y[:, 0], X[:, 17])[0, 1]:+.3f}")
    # --- Aggressive by tercile: the readable version of the correlations above
    ag = Y[:, 0]; lo, hi = np.percentile(ag, 33.3), np.percentile(ag, 66.7)
    groups = [("aggr low", ag <= lo), ("aggr mid", (ag > lo) & (ag <= hi)), ("aggr high", ag > hi)]
    cols = [("enemies", 32), ("war decl", 26), ("ceasefires", 31), ("mil share", 4), ("killed", 16), ("lost", 17), ("new cities", 0)]
    print(f"\nAggressive terciles (per civ per 80-turn window; cut at {lo:.0f} / {hi:.0f}):")
    print(f"{'group':<10}{'n':>5}" + "".join(f"{c:>12}" for c, _ in cols))
    for name, m in groups:
        print(f"{name:<10}{m.sum():>5}" + "".join(f"{X[m, j].mean():>12.2f}" for _, j in cols))
    outp = pathlib.Path("bench/measure/out") / (pathlib.Path(a.corpus).name + "_ideal_observer.json")
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps({"corpus": a.corpus, "n_worlds": len(wids), "n_samples": len(rows),
                                "anchors": anchors, "results": out}, ensure_ascii=False, indent=1))
    print("wrote", outp)


if __name__ == "__main__":
    main()
