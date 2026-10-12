"""End-to-end self-test of the v2 scoring pipeline without a model.

Writes two synthetic output sets for a built run:
  raw_perfect   answers generated from the truth (1a: five equal quantiles at the truth;
                1b: probability 1/0; 1c: the truth report; 2: the truth labels; 3: the
                arm truth) -> every skill must come out at its ceiling
  raw_persist   answers equal to the t1 state (1a: base values; 1b: 0.5 / t1 leader;
                1c: copy-forward; 2: 'mid'/'neutral'/'normal'; 3: same as its factual)
                -> 1a skill vs persistence must be ~0, 1c skills 0, task-3 A ~0

    .venv/bin/python bench/measure/v2_selftest.py --run v2-smoke
    .venv/bin/python bench/score_v2.py --run v2-smoke --raw raw_perfect --bootstrap 200
"""
import argparse, gzip, json, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from bench.spec_v2 import PARAM_DISPLAY, PARAM_LABELS, TRAITS, VARS  # noqa: E402


def lines_1a(truth: dict, civs, hz, base=None) -> str:
    out = []
    for h in hz:
        for c in civs:
            for v in VARS:
                y = (truth.get(str(h)) or {}).get(c, {}).get(v)
                if base is not None:
                    y = (base.get(c) or {}).get(v)
                if y is None:
                    y = 0
                out.append(f"{c} | {v} | +{h} | " + " | ".join(str(y) for _ in range(5)))
    return "\n".join(out)


def lines_1b(items, truth, mode) -> str:
    out = []
    for i, (it, y) in enumerate(zip(items, truth), 1):
        if it["kind"] == "leader":
            civs = it["civs"]
            if mode == "perfect":
                out.append(f"Q{i} " + "; ".join(f"{c}={1.0 if c == y else 0.0}" for c in civs))
            else:
                out.append(f"Q{i} " + "; ".join(f"{c}={1.0 if k == 0 else 0.0}" for k, c in enumerate(civs)))
        else:
            out.append(f"Q{i} {1.0 if (mode == 'perfect' and y) else (0.0 if mode == 'perfect' else 0.5):.2f}")
    return "\n".join(out)


def answer(r: dict, mode: str) -> str:
    t = r["task"]
    if t == "1a":
        return lines_1a(r["truth"], r["civs"], r["horizons"], None if mode == "perfect" else r["base"])
    if t == "1b":
        return lines_1b(r["items"], r["truth"], mode)
    if t == "1c":
        return r["truth"] if mode == "perfect" else r["copy_forward"]
    if t == "2":
        tr, lv, pa = r["truth"]["traits"], r["truth"]["love"], r["truth"]["params"]
        L = []
        if tr and isinstance(next(iter(tr.values())), dict):          # v3 ranking truth {dim: {civ: value}}
            for d in TRAITS:
                if mode == "perfect":
                    order = sorted(tr[d].items(), key=lambda kv: -kv[1])
                    L.append(f"{d}: " + " > ".join(c for c, _ in order))
                else:                                                  # persistence-like: everybody tied
                    L.append(f"{d}: " + " = ".join(r["civs"]))
        elif tr:                                                       # v2 bins
            L += [f"{c} | " + " | ".join(f"{d} {tr[c][d] if mode == 'perfect' else 'mid'}" for d in TRAITS) for c in r["civs"]]
        L += [f"{p} | {lv[p] if mode == 'perfect' else 'neutral'}" for p in lv]
        L += [f"{PARAM_DISPLAY[k]} | {PARAM_LABELS[k][b] if mode == 'perfect' else PARAM_LABELS[k][1]}" for k, b in pa.items()]
        return "\n".join(L)
    if t == "3":
        if r.get("sub"):                                               # v3: one sub-task per prompt
            if r["sub"] == "1a":
                if mode == "perfect":
                    return lines_1a(r["truth"]["1a"], r["civs"], r["horizons"])
                ref = r.get("mainline", {}).get("1a") or r["truth"]["1a"]      # control rows carry mainline truth in `truth`
                return lines_1a(ref, r["civs"], r["horizons"], r["base"])
            if mode == "perfect":
                return lines_1b(r["items"], r["truth"]["1b"], "perfect")
            return lines_1b(r["items"], r.get("mainline", {}).get("1b") or r["truth"]["1b"], "persist")
        if mode == "perfect":
            a = lines_1a(r["truth"]["1a"], r["civs"], r["horizons"])
            b = lines_1b(r["items"], r["truth"]["1b"], "perfect") if r.get("items") else ""
            c = r["truth"]["1c"] or ""
        else:      # ignore the intervention: same as the factual answers (persistence here)
            a = lines_1a(r["mainline"]["1a"], r["civs"], r["horizons"], r["base"])
            b = lines_1b(r["items"], r["mainline"]["1b"], "persist") if r.get("items") else ""
            c = ""
        return "##### 1A\n" + a + ("\n##### 1B\n" + b if b else "") + ("\n##### 1C\n" + c if c else "")
    return ""


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", default="v2-smoke"); a = ap.parse_args()
    run = ROOT / "bench/runs" / a.run
    for mode in ("perfect", "persist"):
        out = run / f"raw_{mode}"; out.mkdir(exist_ok=True)
        for q in sorted((run / "questions").glob("*.jsonl.gz")):
            rows = []
            for l in gzip.open(q, "rt"):
                r = json.loads(l)
                if r.get("error"):
                    continue
                rows.append({k: r[k] for k in ("world", "tier", "t1", "cond", "task", "view")}
                            | {k: r[k] for k in ("op", "op_base", "is_placebo", "h", "h_1c", "sub", "arm_kind", "control_ref") if k in r}
                            | {"output": answer(r, mode), "finish": "stop", "parse_rate": 1.0})
            with gzip.open(out / q.name, "wt") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(mode, "written:", len(list(out.glob("*.jsonl.gz"))), "files")


if __name__ == "__main__":
    main()
