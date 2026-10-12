"""Build annotator packs for the human-analogue line (2026-09-17).

  .venv/bin/python bench/measure/human_line/export_pack.py --run v3-bench-br --anchors bench/v2/human_line_30.txt \
        --out bench/runs/human_line/packs

One pack per (anchor, cond).  A pack contains ONLY prompt text -- no truth fields, no base_hist, nothing the
model does not see:
  MANUAL.md                the expert manual (copied)
  INSTRUCTIONS.md          what to do, which files to write
  WORLD.txt                the system message + the full 1a prompt of this cond (frames, intervals, strategies, TASK 1A)
  rows/<key>.txt           every other row as either "identical to WORLD.txt up to line N, then:" + tail, or a
                           unified diff against WORLD.txt (edit-channel arms change a frame; task 2 has no STRATEGIES)
  answers/                 the annotator writes <key>.txt here (answer lines only)
  keys.json                row keys -> question identity (world, t1, cond, task, sub, op, arm_kind) for assembly
"""
from __future__ import annotations

import argparse
import difflib
import gzip
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from bench.run_infer_v2 import SYSTEM  # noqa: E402

TASK_LABEL = {"1a": "TASK 1A (quantiles)", "1b": "TASK 1B (event probabilities)", "2": "TASK 2 (hidden state)",
              "3": "TASK 3 (counterfactual arm)"}


def row_key(r: dict) -> str:
    k = r["task"] + (f"-{r['sub']}" if r.get("sub") else "")
    if r["task"] == "3":
        k += f"__{str(r.get('op')).split('__placebo')[0]}__{r.get('arm_kind')}"     # placebo arms: op id already ends in __placebo
    return k


def render_relative(base: str, text: str, base_name: str) -> str:
    b, t = base.split("\n"), text.split("\n")
    n = 0
    while n < min(len(b), len(t)) and b[n] == t[n]:
        n += 1
    if n >= 0.5 * len(b):
        return (f"[[ Lines 1..{n} of this prompt are IDENTICAL to lines 1..{n} of {base_name} (same frames, intervals and "
                f"strategies). Read them there. The prompt continues below. ]]\n\n" + "\n".join(t[n:]))
    diff = list(difflib.unified_diff(b, t, fromfile=base_name, tofile="this prompt", lineterm="", n=2))
    return (f"[[ This prompt is {base_name} with the changes below (unified diff; '-' lines are removed, '+' lines added). "
            f"Everything not mentioned in the diff is identical. ]]\n\n" + "\n".join(diff))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="v3-bench-br")
    ap.add_argument("--anchors", default="bench/v2/human_line_30.txt")
    ap.add_argument("--out", default="bench/runs/human_line/packs")
    ap.add_argument("--manual", default="bench/measure/human_line/FREECIV_EXPERT_MANUAL.md")
    ap.add_argument("--conds", default="v1,v2")
    a = ap.parse_args()
    out = ROOT / a.out; out.mkdir(parents=True, exist_ok=True)
    names = [l.strip() for l in open(ROOT / a.anchors) if l.strip()]
    conds = a.conds.split(",")
    index = []
    for n in names:
        rows = [json.loads(l) for l in gzip.open(ROOT / "bench/runs" / a.run / "questions" / n, "rt")]
        for cond in conds:
            rs = [r for r in rows if r["cond"] == cond and not r.get("error")]
            if not rs:
                continue
            base = next((r for r in rs if r["task"] == "1a"), None)
            if base is None:
                continue
            pk = out / f"{n[:-9]}__{cond}"; (pk / "rows").mkdir(parents=True, exist_ok=True); (pk / "answers").mkdir(exist_ok=True)
            shutil.copy(ROOT / a.manual, pk / "MANUAL.md")
            world = f"=====  SYSTEM MESSAGE  =====\n{base.get('system') or SYSTEM}\n\n=====  USER MESSAGE (this is the full TASK 1A prompt)  =====\n{base['prompt']}"
            (pk / "WORLD.txt").write_text(world)
            keys = {}
            for r in rs:
                k = row_key(r)
                keys[k] = {kk: r.get(kk) for kk in ("world", "t1", "cond", "task", "sub", "op", "arm_kind", "op_base", "max_tokens", "h")}
                if r is base:
                    (pk / "rows" / f"{k}.txt").write_text(f"[[ The prompt for this row is WORLD.txt itself: answer its TASK 1A section. ]]\n")
                else:
                    body = render_relative(base["prompt"], r["prompt"], "WORLD.txt")
                    (pk / "rows" / f"{k}.txt").write_text(f"=====  {TASK_LABEL[r['task']]}   row key: {k}  =====\n\n{body}")
            order = sorted(keys, key=lambda k: ({"1a": 0, "1b": 1, "2": 2, "3": 3}[keys[k]["task"]], k))
            (pk / "keys.json").write_text(json.dumps({"anchor": n, "cond": cond, "order": order, "rows": keys}, ensure_ascii=False, indent=1))
            instr = f"""# Instructions for this pack   ({n[:-9]}, condition {cond})

You are an expert Freeciv player answering the CIVLADDER prediction tasks for ONE world at ONE moment.
Everything you may use is in this directory; never read anything outside it.

1. Read MANUAL.md once (game rules, AI behaviour, how to read the prompt, per-task method).
2. Read WORLD.txt: the system message and the full TASK 1A prompt (observation frames, interval narratives,
   strategies, then the 1a question).  Take notes: per-civ regime, rates between frames, land situation, wars.
3. Answer the rows in this order (files in rows/):
{chr(10).join(f"   - rows/{k}.txt   -> answers/{k}.txt" for k in order)}
   A row file either says the prompt is WORLD.txt itself, or gives the tail that follows the shared prefix, or a
   unified diff against WORLD.txt (task-3 arms edit a frame; task 2 has no STRATEGIES block).  For every task-3
   arm, answer consistently with your own factual answers (the control arm IS the factual world).
4. Use a calculator (python) for rates and extrapolations.  Write ONLY the answer lines into answers/<key>.txt,
   exactly in the format the task demands (1a: `<civ> | <quantity> | +<h> | p10 | p25 | p50 | p75 | p90`, all
   civs x quantities x horizons; 1b: `Q<n> <p>` per question, leader questions `Q<n> A=..; B=..`; task 2: the
   ranking / classification lines the prompt specifies; task 3: as its sub-task 1a or 1b).  No commentary,
   no headers, no blank lines between answer lines.
5. After writing all answers, print a short summary (one line per row: key, number of lines written).
"""
            (pk / "INSTRUCTIONS.md").write_text(instr)
            index.append({"pack": pk.name, "anchor": n, "cond": cond, "n_rows": len(keys), "world_chars": len(world)})
    (out / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=1))
    print(f"{len(index)} packs -> {out}; rows {sum(i['n_rows'] for i in index)}; WORLD chars median {sorted(i['world_chars'] for i in index)[len(index)//2]}")


if __name__ == "__main__":
    main()
