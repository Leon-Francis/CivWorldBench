"""Measured base rates of the 1b events over games that have actually been played (WM v6, doc 17 §4.7).

The v3 benchmark's clearest single win was giving the forecaster the reference base rate in the 1b prompt: Brier
skill went from negative to positive (`v3-bench-br`).  The agent-play forecaster never had that, and the case
analysis (§6.8.1) showed exactly the biases you would predict without it -- bankruptcy forecast at 0.09 and never
observed, city loss on a "reinforce" route forecast at 0.14 and realized at 0.29, wonders forecast at 0.29 and
realized at 0.11.  This script measures the rates from the games themselves and writes the block that v6 puts in
front of every forecast question.

  .venv/bin/python bench/agentplay/base_rates.py --games bench/agentplay/games/none16v5 ... --jobs 40
  -> bench/agentplay/base_rates.json   (+ the prompt block on stdout)
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "bundle"))
from bench.agentplay.wm import History, current_levels                     # noqa: E402
from bench.truth_v2 import mil_of, truth_1b_from_states                    # noqa: E402
from bench.spec_v2 import TEMPLATES_1B, met_civs                           # noqa: E402

_TPL = {k: txt for k, _, _, txt in TEMPLATES_1B}
EGO_EVENTS = ("lostcity", "newcity", "mildrop", "bankrupt", "moreenemies", "govchange", "wonder", "capture")
OTHER_EVENTS = ("moreenemies", "capture", "lostcity", "newcity")
OUT = ROOT / "bench/agentplay/base_rates.json"


def one_game(args: tuple[str, int, int]) -> dict:
    gdir, step, hz_max = Path(args[0]), args[1], args[2]
    try:
        recs = [json.loads(l) for l in open(gdir / "turns.jsonl")]
        summ = json.load(open(gdir / "summary.json"))
    except Exception as e:                                     # noqa: BLE001
        return {"game": gdir.name, "error": f"{type(e).__name__}: {e}"}
    world, start, ego_i = summ["world"], int(summ["start"]), int(summ["ego"])
    ego = summ.get("ego_name"); end = recs[-1]["turn"] + 1
    hist = History(world, gdir, start, ego_i)
    hits: dict[str, list[int]] = defaultdict(list)
    try:
        states = {t: hist.record(t) for t in range(start, end + 1)}
    except Exception as e:                                     # noqa: BLE001
        return {"game": gdir.name, "error": f"records: {type(e).__name__}: {e}"}
    for t1 in range(start, end - 20 + 1, step):
        rec1 = states[t1]
        try:
            civs = met_civs(rec1, ego)
            levels = current_levels(rec1, civs)
        except Exception:                                      # noqa: BLE001
            continue
        items = []
        for civ in civs:
            lv = levels.get(civ, {})
            for h in (20, 40):
                if t1 + h > end:
                    continue
                for k in (EGO_EVENTS if civ == ego else OTHER_EVENTS):
                    if civ != ego and h != 20:
                        continue
                    base = lv.get("n_military_units") if k == "mildrop" else lv.get("n_enemies") if k == "moreenemies" else ""
                    items.append({"kind": k, "civ": civ, "h": h, "t1": t1, "tt": t1 + h, "base": base,
                                  "text": _TPL[k].format(civ=civ, t1=t1, tt=t1 + h, base=base, other="", base2="")})
        if not items:
            continue
        try:
            truths = truth_1b_from_states(items, states, rec1, mil_of(rec1))
        except Exception:                                      # noqa: BLE001
            continue
        for it, tr in zip(items, truths):
            if tr is None:
                continue
            hits[f"{'ego' if it['civ'] == ego else 'other'}|{it['kind']}|{it['h']}"].append(int(bool(tr)))
    return {"game": gdir.name, "arm": summ.get("arm"), "counts": {k: [sum(v), len(v)] for k, v in hits.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", nargs="+", required=True, help="game dirs or arm dirs")
    ap.add_argument("--jobs", type=int, default=40); ap.add_argument("--step", type=int, default=10)
    a = ap.parse_args()
    dirs: list[Path] = []
    for g in a.games:
        p = Path(g)
        if (p / "turns.jsonl").exists():
            dirs.append(p)
        else:
            dirs += [q for q in sorted(p.iterdir()) if (q / "turns.jsonl").exists() and (q / "summary.json").exists()]
    print(f"{len(dirs)} games", flush=True)
    pool: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    per_arm: dict[str, dict] = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    bad = 0
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        for r in ex.map(one_game, [(str(d), a.step, 40) for d in dirs]):
            if r.get("error"):
                bad += 1; print(f"  {r['game']}: {r['error']}", flush=True); continue
            for k, (h, n) in r["counts"].items():
                pool[k][0] += h; pool[k][1] += n
                per_arm[r.get("arm") or "?"][k][0] += h; per_arm[r.get("arm") or "?"][k][1] += n
    rates = {k: {"rate": round(v[0] / v[1], 3), "n": v[1]} for k, v in sorted(pool.items()) if v[1] >= 20}
    out = {"n_games": len(dirs) - bad, "step": a.step, "rates": rates,
           "per_arm": {arm: {k: [v[0], v[1]] for k, v in sorted(d.items())} for arm, d in per_arm.items()}}
    json.dump(out, open(OUT, "w"), indent=1)
    print(json.dumps(rates, indent=1))
    print("\n" + block())


def block() -> str:
    """The prompt block v6 puts in front of every forecast question."""
    if not OUT.exists():
        return ""
    d = json.load(open(OUT))
    r = d["rates"]
    def row(k: str) -> str:
        v = r.get(k)
        return "" if v is None else f"{v['rate']:.2f}"
    L = ["=====  MEASURED BASE RATES  (how often each of these actually happened, over "
         f"{d['n_games']} played games of exactly this kind: the same worlds, the same takeover turn, the same "
         "length, rulers played by a model like the one you are advising)  =====",
         "These are frequencies, not opinions. A forecast far from the base rate needs a reason you can name from",
         "the frames. Under-forecasting a common event and over-forecasting a rare one are equally wrong.",
         "",
         f"{'event (does it happen in the window?)':<44} {'within 20 turns':>16} {'within 40 turns':>16}"]
    names = {"lostcity": "the ruler loses at least one city", "newcity": "the ruler founds at least one city",
             "mildrop": "the ruler's military falls by 30% or more", "bankrupt": "the ruler's treasury hits 0",
             "moreenemies": "the ruler is at war with more civs", "govchange": "the ruler changes government",
             "wonder": "the ruler completes a great wonder", "capture": "the ruler captures a city"}
    for k, nm in names.items():
        a20, a40 = row(f"ego|{k}|20"), row(f"ego|{k}|40")
        if a20 or a40:
            L.append(f"{nm:<44} {a20:>16} {a40:>16}")
    L.append("")
    for k, nm in (("lostcity", "a neighbour loses a city"), ("newcity", "a neighbour founds a city"),
                  ("capture", "a neighbour captures a city"), ("moreenemies", "a neighbour gains an enemy")):
        a20 = row(f"other|{k}|20")
        if a20:
            L.append(f"{nm:<44} {a20:>16}")
    L.append("=====  END OF BASE RATES  =====")
    return "\n".join(L) + "\n\n"


if __name__ == "__main__":
    main()
