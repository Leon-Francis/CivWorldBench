"""Assemble the annotator packs' answers into a raw output dir of the run, scored like any model.

  .venv/bin/python bench/measure/human_line/assemble_line.py --run v3-bench-br --packs bench/runs/human_line/packs --tag fable
  .venv/bin/python bench/score_v2.py --run v3-bench-br --raw raw_fable --bootstrap 500

Rows follow run_infer_v2's output schema (fields copied from the question row + output/parse_rate); missing answers
are reported and left out (score_v2 then counts them as missing).
"""
from __future__ import annotations

import argparse
import collections
import gzip
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from bench.parse_v2 import parse_rate  # noqa: E402
from bench.measure.human_line.export_pack import row_key  # noqa: E402

KEEP = ("world", "tier", "t1", "cond", "task", "view", "set", "budget", "pomdp", "viewer", "primer", "oracle", "k",
        "op", "op_base", "is_placebo", "channel", "layer", "score_vars", "horizon_band", "op_target", "h", "h_1c",
        "sub", "arm_kind", "control_ref", "fmt", "horizons", "civs", "items", "pairs", "think")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="v3-bench-br"); ap.add_argument("--packs", default="bench/runs/human_line/packs")
    ap.add_argument("--tag", default="fable")
    a = ap.parse_args()
    packs = sorted((ROOT / a.packs).glob("*__v*"))
    out = ROOT / "bench/runs" / a.run / f"raw_{a.tag}"; out.mkdir(exist_ok=True)
    by_file = collections.defaultdict(list); missing = []; rates = collections.defaultdict(list); n_ok = 0
    for pk in packs:
        meta = json.loads((pk / "keys.json").read_text()); anchor = meta["anchor"]
        rows = {row_key(r): r for r in (json.loads(l) for l in gzip.open(ROOT / "bench/runs" / a.run / "questions" / anchor, "rt")) if r["cond"] == meta["cond"]}
        for k in meta["order"]:
            f = pk / "answers" / f"{k}.txt"
            if not f.exists() or not f.read_text().strip():
                missing.append(f"{pk.name}/{k}"); continue
            q = rows[k]; text = f.read_text().strip("\n")
            pr = parse_rate(q, text)
            rates[q["task"] + ("-" + q["sub"] if q.get("sub") else "")].append(pr); n_ok += 1
            by_file[anchor].append({kk: q[kk] for kk in KEEP if kk in q} | {"output": text, "n_gen": len(text) // 4, "n_prompt": len(q["prompt"]) // 4,
                                                                          "n_cached": 0, "think_tok": 0, "finish": "stop", "parse_rate": pr, "annotator": a.tag})
    for anchor, recs in by_file.items():
        with gzip.open(out / anchor, "wt") as fh:
            for r in recs:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"answers {n_ok} · anchors {len(by_file)} · missing {len(missing)} -> {out}")
    for t, v in sorted(rates.items()):
        print(f"  parse rate {t}: {sum(v)/len(v):.2f} (n={len(v)})")
    if missing:
        print("  missing:", missing[:20], "..." if len(missing) > 20 else "")


if __name__ == "__main__":
    main()
