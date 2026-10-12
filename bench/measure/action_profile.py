"""Can the corpus's `actions` field be turned into an abstract STRATEGY COMMITMENT?

`actions` holds turn-t *intent* per civ: tax/science/luxury split, research goal, per-city
production target (+ kind), and per-unit orders.  The question for the action-conditional task
(L3/L4 in doc 12) is not whether the field is rich -- it is -- but:

  1. is a coarse profile of it STABLE enough over a horizon to be called a commitment?
  2. is the FUTURE profile predictable from the VISIBLE past window?  If yes, conditioning on it
     adds nothing and a null result would be uninterpretable (redundancy, not inability).
  3. does the profile DISCRIMINATE civs / worlds?  A profile everyone shares carries no signal.
  4. is it outcome-free?  Only shares and postures may enter -- never counts (v1's lesson: a
     per-unit action path leaked positions, which are outputs).

    .venv/bin/python bench/measure/action_profile.py --corpus data_v2/bench --worlds 60
"""
from __future__ import annotations
import argparse, collections, glob, gzip, json, math, pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bundle")); sys.path.insert(0, str(ROOT / "tools"))
from bench.spec_v2 import mil_map_for                                    # noqa: E402
from bench.interval_v2 import building_genus, RULESETS                   # noqa: E402
from sandboxes.freeciv.ruleset import parse                              # noqa: E402
from functools import lru_cache

PROD_CATS = ("expansion", "worker", "military", "civilian", "infrastructure", "wonder", "coinage")


@lru_cache(maxsize=None)
def unit_class(ruleset: str) -> dict[str, str]:
    """name -> expansion | worker | military | civilian, from the ruleset's own tables."""
    u = parse(RULESETS / ruleset / "units.ruleset")
    mil = mil_map_for(ruleset)
    out = {}
    for k, v in u.items():
        if not k.startswith("unit_") or "name" not in v:
            continue
        name = str(v["name"])
        flags = str(v.get("flags") or "") + " " + str(v.get("roles") or "")
        if mil.get(name):
            out[name] = "military"
        elif "Cities" in flags:
            out[name] = "expansion"
        elif "Settlers" in flags or "Workers" in flags:
            out[name] = "worker"
        else:
            out[name] = "civilian"
    return out


def prod_cat(target: str, kind: str, ucls: dict, genus: dict) -> str | None:
    if target in (None, "", "None"):
        return None
    if target == "Coinage":
        return "coinage"
    if kind == "UnitType":
        return ucls.get(target, "civilian")
    g = (genus.get(target) or "Improvement").lower()
    return "wonder" if "wonder" in g else "infrastructure"


def profile(recs: dict, civ: str, a: int, b: int, ruleset: str) -> dict | None:
    """Coarse, outcome-free strategy profile of `civ` over turns [a, b]."""
    ucls, genus = unit_class(ruleset), building_genus(ruleset)
    mix = collections.Counter(); city_turns = 0
    tax = sci = lux = 0.0; n = 0
    goals = collections.Counter()
    moving = fort = idle = 0; orders = 0; queued = 0
    for t in range(a, b + 1):
        r = recs.get(t)
        act = ((r or {}).get("actions") or {}).get(civ)
        if not act:
            continue
        n += 1
        tax += act.get("tax") or 0; sci += act.get("science") or 0; lux += act.get("luxury") or 0
        if act.get("research_goal") and act["research_goal"] != "A_UNSET":
            goals[act["research_goal"]] += 1
        kinds = act.get("production_kind") or {}
        for city, tgt in (act.get("production") or {}).items():
            c = prod_cat(tgt, kinds.get(city, "UnitType"), ucls, genus)
            if c:
                mix[c] += 1; city_turns += 1
        for o in act.get("unit_orders") or []:
            orders += 1
            queued += int(o.get("queued_orders") or 0)
            if o.get("moving"):
                moving += 1
            elif str(o.get("activity")) in ("Fortified", "Fortifying"):
                fort += 1
            else:
                idle += 1
    if not n or not city_turns:
        return None
    return {"prod": {c: mix[c] / city_turns for c in PROD_CATS},
            "tax": tax / n, "sci": sci / n, "lux": lux / n,
            "n_goals": len(goals), "goal_top": goals.most_common(1)[0][0] if goals else None,
            "posture": {"moving": moving / max(orders, 1), "fortified": fort / max(orders, 1),
                        "idle": idle / max(orders, 1)},
            "queued_per_unit": queued / max(orders, 1), "turns": n, "city_turns": city_turns}


def l1(p: dict, q: dict) -> float:
    return sum(abs(p["prod"][c] - q["prod"][c]) for c in PROD_CATS) / 2   # total-variation, [0,1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data_v2/bench")
    ap.add_argument("--worlds", type=int, default=60)
    ap.add_argument("--h", type=int, default=120)
    ap.add_argument("--back", type=int, default=40)
    a = ap.parse_args()
    C = ROOT / a.corpus
    man = json.load(open(C / "manifest.json"))["worlds"]
    man = [w for w in man if (C / w["world_id"] / "world.jsonl.gz").exists()][:: max(1, len(man) // a.worlds)]
    rows = []; per_world_spread = []
    for w in man[: a.worlds]:
        wid = w["world_id"]; rs = w["ruleset"]
        meta = json.load(open(C / wid / "meta.json"))
        anchors = [int(pathlib.Path(c).stem[1:]) for c in meta["branch_checkpoints"]]
        recs = {}
        need = set()
        for t1 in anchors:
            need |= set(range(t1 - a.back, t1 + a.h + 1))
        for line in gzip.open(C / wid / "world.jsonl.gz", "rt"):
            r = json.loads(line)
            if r["t"] in need:
                recs[r["t"]] = r
        for t1 in anchors:
            civs = [p["name"] for p in recs.get(t1, {}).get("state", {}).get("players", []) if p.get("is_alive")]
            profs = {}
            for c in civs:
                past = profile(recs, c, t1 - a.back, t1, rs)
                fut = profile(recs, c, t1 + 1, t1 + a.h, rs)
                if not past or not fut:
                    continue
                profs[c] = (past, fut)
                rows.append((wid, t1, c, past, fut, l1(past, fut)))
            # spread between civs inside one anchor (discriminability)
            cs = list(profs)
            for i in range(len(cs)):
                for j in range(i + 1, len(cs)):
                    per_world_spread.append(l1(profs[cs[i]][1], profs[cs[j]][1]))
    print(f"{len(man[:a.worlds])} 世界 · {len(rows)} 个 (锚点, 文明) 画像对 · back={a.back} h={a.h}")

    def q(v, p):
        v = sorted(v); return v[min(len(v) - 1, int(p * len(v)))] if v else float("nan")

    drift = [r[5] for r in rows]
    print(f"\n[1] 未来窗口 vs 可见窗口的生产结构漂移（总变差距离，0=完全一致，1=完全不同）")
    print(f"    p10 {q(drift,.1):.3f}  中位 {q(drift,.5):.3f}  p90 {q(drift,.9):.3f}")
    print(f"[2] 同一锚点内不同文明之间的生产结构差异（可辨别性）")
    print(f"    p10 {q(per_world_spread,.1):.3f}  中位 {q(per_world_spread,.5):.3f}  p90 {q(per_world_spread,.9):.3f}")
    print(f"    ⇒ 若 [1] 明显小于 [2]，则「未来画像」几乎可从可见窗口推出，条件化是冗余的")

    print("\n[3] 生产结构的边缘分布（未来窗口，全体 civ-锚点）")
    for c in PROD_CATS:
        v = [r[4]["prod"][c] for r in rows]
        print(f"    {c:15s} 均值 {sum(v)/len(v):.3f}  中位 {q(v,.5):.3f}  p90 {q(v,.9):.3f}  >0 的比例 {sum(1 for x in v if x>0)/len(v):.2f}")

    print("\n[4] 财政与姿态的稳定性（|未来均值 − 过去均值|）")
    for k in ("tax", "sci", "lux"):
        d = [abs(r[4][k] - r[3][k]) for r in rows]
        print(f"    {k:5s} 中位漂移 {q(d,.5):.1f} 个百分点  p90 {q(d,.9):.1f}")
    for k in ("moving", "fortified", "idle"):
        d = [abs(r[4]["posture"][k] - r[3]["posture"][k]) for r in rows]
        print(f"    {k:9s} 中位漂移 {q(d,.5):.3f}  p90 {q(d,.9):.3f}")
    d = [r[4]["n_goals"] for r in rows]
    print(f"    未来窗口内不同研究目标个数: 中位 {q(d,.5):.0f}  p90 {q(d,.9):.0f}  ⚠️ 这个量与 techs_learned 相关，不能进 prompt")

    out = ROOT / "bench/measure/out/action_profile.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"drift": drift, "spread": per_world_spread,
                               "rows": [[r[0], r[1], r[2], r[3], r[4], r[5]] for r in rows[:2000]]},
                              ensure_ascii=False))
    print("\nwrote", out)


if __name__ == "__main__":
    main()
