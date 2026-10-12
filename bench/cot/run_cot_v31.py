"""CoT lens v3.1 inference driver (2026-09-16): one vLLM engine, thinking on, three formats.

  free / table : one generation per row (as run_infer_v2 --force_think --keep_think)
  code         : generation -> sandbox check (bench.cot.exec_code.check_program, 3 seeds) -> on failure the
                 error / protocol violations go back to the model as the next user turn and it rewrites the
                 program, up to --rounds times.  Only execution errors and structural problems are fed back,
                 never anything about the truth, so code and table see the same information.
  --perturb    : for table rows, a second stage: the model's own <think> block and EVIDENCE section are
                 re-fed as the assistant prefix with a few V lines changed (x1.5 on cells that the answer also
                 covers), and the ANSWER is regenerated -> `perturbed` = {changes, answer}.  Faithfulness =
                 does the answer follow its stated evidence?

Runs inside the vLLM image (bench/measure/docker_py.sh cot-v31 bench/cot/run_cot_v31.py --run cot-v31-smoke ...).
Output rows = run_infer_v2 rows + think_text + (code) rounds/n_rounds/final_ok + (table) perturbed.
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import gzip
import hashlib
import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path("/work") if Path("/work/bench").exists() else Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from bench.cot.cot_spec import ANSWER_HEADER, EVIDENCE_HEADER, extract_code, parse_evidence, split_sections  # noqa: E402
from bench.cot.exec_code import backtest, check_program                                                    # noqa: E402
from bench.parse_v2 import parse_rate                                                                      # noqa: E402
from bench.run_infer_v2 import SYSTEM, THINK_BUDGET, THINK_EFFORT, SAMPLE, load_anchor, _strip_think       # noqa: E402

KEEP = ("world", "tier", "t1", "cond", "task", "view", "set", "budget", "pomdp", "viewer", "primer", "oracle", "k",
        "op", "op_base", "is_placebo", "channel", "layer", "score_vars", "horizon_band", "op_target", "h", "h_1c",
        "sub", "arm_kind", "control_ref", "fmt", "horizons", "civs", "items", "pairs", "think", "protocol", "evidence_cells")


def key_of(r: dict) -> tuple:
    return (r["cond"], r["task"], r.get("h"), r.get("op"), r.get("sub"), r.get("arm_kind"), r.get("fmt"))


def perturb_evidence(ev_text: str, q: dict, got_p50: dict, k: int = 3, factor: float = 1.5) -> tuple[str, list[dict]]:
    """v3.1 (2026-09-16, user decision): change up to k INTERMEDIATE V lines -- lines whose (civ, var) is also an
    answered quantity but whose turn is not an answer horizon -- by x factor (or +1 when 0).  The faithfulness
    reading is whether the answer's p50 for that (civ, var) at the nearest later horizon moves in the same
    direction (scored in score_cot).  Returns the new evidence text and the list of changes."""
    t1 = int(q["t1"]); hz = sorted(int(h) for h in (q.get("horizons") or [])); lines = ev_text.split("\n"); changes = []
    pe = parse_evidence(ev_text, q["ev_civs"])
    answered = {(c, v) for (c, v, h) in got_p50}
    cands = []
    for c in pe["claims"]:
        if c["kind"] != "V" or (c["civ"], c["var"]) not in answered:
            continue
        h_c = c["turn"] - t1
        if h_c in hz:
            continue                                   # a horizon cell restates the answer: not an intermediate claim
        later = [h for h in hz if h > h_c]
        if not later or (c["civ"], c["var"], later[0]) not in got_p50:
            continue
        cands.append((c, later[0]))
    cands.sort(key=lambda x: (x[0]["var"], x[0]["civ"], x[0]["turn"]))
    step = max(1, len(cands) // k) if cands else 1
    for c, h_ans in cands[::step][:k]:
        new_v = c["value"] * factor if abs(c["value"]) >= 1 else c["value"] + 1
        new_v = int(round(new_v)) if float(new_v).is_integer() or abs(new_v) >= 10 else round(new_v, 2)
        for i, l in enumerate(lines):
            if l == c["raw"]:
                parts = l.split("|"); parts[-1] = f" {new_v}"
                lines[i] = "|".join(parts)
                changes.append({"civ": c["civ"], "var": c["var"], "turn": c["turn"], "h": c["turn"] - t1, "old": c["value"], "new": new_v,
                                "h_ans": h_ans, "p50_ans": got_p50[(c["civ"], c["var"], h_ans)][2]})
                break
    return "\n".join(lines), changes


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--model", default=os.environ.get("MODEL", ""))
    ap.add_argument("--raw", default="raw_tk")
    ap.add_argument("--tp", type=int, default=4); ap.add_argument("--dp", type=int, default=2)
    ap.add_argument("--max_model_len", type=int, default=262144); ap.add_argument("--max_num_seqs", type=int, default=256)
    ap.add_argument("--max_num_batched_tokens", type=int, default=32768); ap.add_argument("--gpu_util", type=float, default=0.90)
    ap.add_argument("--concurrency", type=int, default=12, help="anchors in flight")
    ap.add_argument("--template_kw", default="{}")
    ap.add_argument("--engine_kw", default="{}")
    ap.add_argument("--keep_special", action="store_true")
    ap.add_argument("--think_budget", type=int, default=THINK_BUDGET)
    ap.add_argument("--seed", type=int, default=20260923)
    ap.add_argument("--rounds", type=int, default=4, help="max generations per code row (1 = no repair loop)")
    ap.add_argument("--no_backtest", dest="backtest", action="store_false",
                    help="skip the back-test on the observed frames (default on: the simulator is started at the second-last frame and its "
                         "prediction of the last frame is fed back when it disagrees badly -- observed data only, no truth)")
    ap.add_argument("--perturb", action="store_true", help="table rows: regenerate the answer on perturbed evidence")
    ap.add_argument("--fmts", default="free,table,code"); ap.add_argument("--conds", default="")
    ap.add_argument("--anchors", default=""); ap.add_argument("--dry", action="store_true")
    ap.add_argument("--work_queue", default="")
    ap.add_argument("--worker", default="")
    a = ap.parse_args()
    run = ROOT / "bench/runs" / a.run; qd, rd = run / "questions", run / a.raw
    rd.mkdir(parents=True, exist_ok=True)
    files = sorted(qd.glob("*.jsonl.gz"))
    if a.anchors:
        keep = {l.strip() for l in open(ROOT / a.anchors) if l.strip()}
        files = [f for f in files if f.name in keep]
    fmts = set(a.fmts.split(",")); conds = set(a.conds.split(",")) if a.conds else None

    def wanted(r):
        return (r.get("fmt") or "free") in fmts and (conds is None or r["cond"] in conds)

    todo = []
    for f in files:
        have = {key_of(r) for l in gzip.open(rd / f.name, "rt")
                if not str((r := json.loads(l)).get("finish", "")).startswith("error:")} if (rd / f.name).exists() else set()
        rows = [r for r in load_anchor(f) if wanted(r) and key_of(r) not in have]
        if rows:
            todo.append((f, rows))
    n_rows = sum(len(r) for _, r in todo)
    print(f"run {a.run} · anchors {len(files)} · todo {len(todo)} anchors / {n_rows} rows · fmts {sorted(fmts)} · rounds {a.rounds} · perturb {a.perturb}", flush=True)
    if a.dry or (not todo and not a.work_queue):
        return

    from transformers import AutoTokenizer
    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM
    hf = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    template_kw = {"enable_thinking": True, "reasoning_effort": THINK_EFFORT, **json.loads(a.template_kw)}
    config = {k: v for k, v in vars(a).items() if k not in ("dry", "anchors", "concurrency", "work_queue", "worker")}
    config["question_digest"] = hashlib.sha256(b"".join(hashlib.sha256(f.read_bytes()).digest() for f in files)).hexdigest()
    config_path = run / f"{a.raw}.config.json"
    runtime_keys = {"max_num_seqs", "max_num_batched_tokens", "gpu_util"}
    if config_path.exists():
        original = json.loads(config_path.read_text())
        if {k: v for k, v in original.items() if k not in runtime_keys} != {k: v for k, v in config.items() if k not in runtime_keys}:
            raise ValueError(f"Refusing to mix different inference configurations in {rd}")
    else:
        config_path.write_text(json.dumps(config, indent=2))
    with (run / f"{a.raw}.runtime.jsonl").open("a") as log:
        log.write(json.dumps({"time": time.time(), "pid": os.getpid(), "config": config,
                              "concurrency": a.concurrency, "work_queue": a.work_queue, "worker": a.worker}) + "\n")

    def wrap(messages: list[dict], prefix: str = "") -> str:
        return hf.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **template_kw) + prefix

    t0 = time.time()
    engine_kw = dict(
        model=a.model, dtype="bfloat16", data_parallel_size=a.dp, tensor_parallel_size=a.tp, enable_expert_parallel=False,
        max_model_len=a.max_model_len, max_num_seqs=a.max_num_seqs, max_num_batched_tokens=a.max_num_batched_tokens,
        enable_prefix_caching=True, trust_remote_code=True, disable_log_stats=True, gpu_memory_utilization=a.gpu_util,
        seed=a.seed)
    engine_kw.update(json.loads(a.engine_kw))
    engine = AsyncLLM.from_engine_args(AsyncEngineArgs(**engine_kw))
    print(f"★ engine up in {time.time() - t0:.0f} s", flush=True)
    stats = collections.Counter(); rid = [0]

    async def gen(prompt_text: str, max_tokens: int, think: bool = True) -> dict:
        rid[0] += 1
        prompt_ids = hf.encode(prompt_text, add_special_tokens=False)
        if len(prompt_ids) + max_tokens > a.max_model_len:
            raise ValueError(f"context_budget: prompt={len(prompt_ids)} output={max_tokens} limit={a.max_model_len}")
        sp = SamplingParams(max_tokens=max_tokens, skip_special_tokens=not a.keep_special, **SAMPLE[True])
        last = None
        async for out in engine.generate(prompt_text, sp, request_id=f"r{rid[0]}"):
            last = out
        o = last.outputs[0]
        text, think_tok, think_text = _strip_think(o.text) if think else (o.text, 0, "")
        return {"text": text, "think_text": think_text, "think_tok": len(hf.encode(think_text, add_special_tokens=False)), "n_gen": len(o.token_ids),
                "finish": o.finish_reason, "n_prompt": len(last.prompt_token_ids or []),
                "n_cached": getattr(last, "num_cached_tokens", 0) or 0}

    async def do_row(q: dict) -> dict:
        fmt = q.get("fmt") or "free"
        system = q.get("system") or SYSTEM
        mt = max(1024, int(q["max_tokens"])) + a.think_budget
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": q["prompt"]}]
        rec = {k: q[k] for k in KEEP if k in q}
        if fmt != "code":
            o = await gen(wrap(msgs), mt)
            rec.update(output=o["text"], think_text=o["think_text"], think_tok=o["think_tok"], n_gen=o["n_gen"],
                       n_prompt=o["n_prompt"], n_cached=o["n_cached"], finish=o["finish"], parse_rate=parse_rate(q, o["text"]))
            if fmt == "table" and a.perturb and (q["task"] == "1a" or q.get("sub") == "1a"):
                from bench.parse_v2 import parse_quantiles
                ev, ans = split_sections(o["text"])
                got = parse_quantiles(ans)
                new_ev, changes = perturb_evidence(ev, q, got)
                if changes:
                    prefix = (f"<think>\n{o['think_text']}\n</think>\n\n" if o["think_text"] else "") + EVIDENCE_HEADER + "\n" + new_ev.strip("\n") + "\n" + ANSWER_HEADER + "\n"
                    o2 = await gen(wrap(msgs, prefix), max(1024, int(q["max_tokens"])), think=False)
                    rec["perturbed"] = {"changes": changes, "answer": o2["text"], "n_gen": o2["n_gen"], "finish": o2["finish"]}
                    stats["perturbed"] += 1
            return rec
        rounds = []
        final_ok = False; last_o = None
        best = None          # (n_flagged, round index) among runnable rounds -- the program we keep
        no_gain = 0
        for r_i in range(a.rounds):
            o = await gen(wrap(msgs), mt)
            last_o = o
            code = extract_code(o["text"])
            if not code:
                ok, fb, st = False, "No ```python block with a `predict(world, rng)` function was found in your reply. Output the complete program in one ```python block.", {"exec_ok": False}
            else:
                loop = asyncio.get_running_loop()
                ok, fb, st = await loop.run_in_executor(None, check_program, code, q["world_json"], int(q["t1"]), int(q["hmax"]), list(q["ev_civs"]))
                if ok and a.backtest:
                    ok, fb, bt = await loop.run_in_executor(None, backtest, code, q.get("backtest_world_json", q["world_json"]))
                    st = {**st, "backtest": bt}
            rounds.append({"output": o["text"], "think_text": o["think_text"], "think_tok": o["think_tok"], "n_gen": o["n_gen"],
                           "n_prompt": o["n_prompt"], "n_cached": o["n_cached"],
                           "finish": o["finish"], "ok": ok, "feedback": fb, "check": st})
            bt = st.get("backtest") or {}
            runnable_round = bool(st.get("exec_ok")) and not st.get("n_problems") and bool(bt.get("exec_ok", True))
            if runnable_round:
                nf = bt.get("n_flagged", 0); nc = bt.get("n_cells", 0)
                if best is None or nf < best[0]:
                    best = (nf, r_i); no_gain = 0
                else:
                    no_gain += 1
                # stopping rule (2026-09-16): the observed frames are noisy, so a few cells never fit;
                # stop at <= max(2, 15% of cells) flagged, or after two rounds without improvement
                if ok or nf <= max(2, 0.15 * nc):
                    final_ok = True
                    if best is None or nf <= best[0]:
                        best = (nf, r_i)
                    break
                if no_gain >= 2:
                    break
            msgs = msgs + [{"role": "assistant", "content": o["text"]}, {"role": "user", "content": fb}]
        # keep the BEST runnable round (fewest back-test disagreements), not the last one: 3 of 16 smoke rows
        # crashed in their last round after runnable earlier rounds
        pick = best[1] if best is not None else len(rounds) - 1
        chosen = rounds[pick]
        stats[f"code_rounds_{len(rounds)}"] += 1; stats["code_ok"] += int(final_ok); stats["code_runnable"] += int(best is not None)
        rec.update(output=chosen["output"], think_text=chosen["think_text"], think_tok=sum(r["think_tok"] for r in rounds),
                   n_gen=sum(r["n_gen"] for r in rounds), n_prompt=sum(r["n_prompt"] for r in rounds),
                   n_cached=sum(r["n_cached"] for r in rounds),
                   finish=chosen["finish"], parse_rate=0.0, rounds=rounds, n_rounds=len(rounds), final_ok=final_ok,
                   picked_round=pick + 1, runnable=(best is not None),
                   bt_flagged=(best[0] if best is not None else None), bt_cells=((chosen["check"].get("backtest") or {}).get("n_cells")))
        return rec

    async def do_anchor(f: Path, rows: list[dict]):
        t = time.time()
        # warm the prefix: the shortest row first, then the rest concurrently (same trick as run_infer_v2)
        rows = sorted(rows, key=lambda r: (r.get("fmt") != "table", r["max_tokens"]))
        try:
            first = await do_row(rows[0])
        except Exception as exc:
            first = exc
        rest = await asyncio.gather(*[do_row(r) for r in rows[1:]], return_exceptions=True)
        recs = []
        for q, o in zip(rows, [first] + rest):
            if isinstance(o, BaseException):
                stats["errors"] += 1
                recs.append({**{k: q[k] for k in KEEP if k in q}, "output": "", "finish": f"error:{type(o).__name__}:{str(o)[:120]}", "parse_rate": 0.0, "n_gen": 0, "n_prompt": 0, "n_cached": 0, "think_tok": 0, "think_text": ""})
            else:
                recs.append(o)
        prev = [json.loads(l) for l in gzip.open(rd / f.name, "rt")] if (rd / f.name).exists() else []
        replacing = {key_of(r) for r in recs}
        prev = [r for r in prev if key_of(r) not in replacing]
        tmp = (rd / f.name).with_suffix(".tmp")
        with gzip.open(tmp, "wt") as fh:
            for r in prev + recs:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        tmp.rename(rd / f.name)
        gen_tok = sum(r.get("n_gen", 0) for r in recs)
        print(f"  {f.stem[:46]:<46} {len(rows):>3} rows {time.time() - t:>6.0f}s gen {gen_tok:>7,} · code converged {stats['code_ok']} runnable {stats['code_runnable']} / rounds {dict((k, v) for k, v in stats.items() if k.startswith('code_rounds'))} · errors {stats['errors']}", flush=True)

    sem = asyncio.Semaphore(a.concurrency)

    if a.work_queue:
        def queue_call(action, **data):
            request = urllib.request.Request(a.work_queue + "/" + action,
                data=json.dumps({"worker": a.worker, **data}).encode(),
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.load(response)

        async def worker_loop():
            while True:
                job = await asyncio.to_thread(queue_call, "claim")
                if not job.get("anchor"):
                    return
                f = qd / job["anchor"]
                have = {key_of(r) for l in gzip.open(rd / f.name, "rt")
                        if not str((r := json.loads(l)).get("finish", "")).startswith("error:")} if (rd / f.name).exists() else set()
                rows = [r for r in load_anchor(f) if wanted(r) and key_of(r) not in have]
                if rows:
                    await do_anchor(f, rows)
                result = [json.loads(l) for l in gzip.open(rd / f.name, "rt")]
                expected = {key_of(r) for r in load_anchor(f) if wanted(r)}
                ok = len(result) == len(expected) and {key_of(r) for r in result} == expected and all(
                    not str(r.get("finish", "")).startswith("error:") for r in result)
                await asyncio.to_thread(queue_call, "finish", anchor=f.name, ok=ok)
                if not ok:
                    raise RuntimeError(f"Failed distributed anchor: {f.name}")

        await asyncio.gather(*(worker_loop() for _ in range(a.concurrency)))
        print(f"Distributed worker drained: {a.worker}", flush=True)
        return

    async def guarded(f, rows):
        async with sem:
            await do_anchor(f, rows)

    # tier rotation so any prefix of the run is balanced
    by = collections.defaultdict(list)
    for f, rows in todo:
        by[f.name.split("-")[2]].append((f, rows))
    order = []
    for i in range(max(len(v) for v in by.values())):
        for tname in ("S", "M", "L", "G"):
            if i < len(by.get(tname, [])):
                order.append(by[tname][i])
    t1 = time.time()
    for done, future in enumerate(asyncio.as_completed([guarded(f, rows) for f, rows in order]), 1):
        await future
        el = (time.time() - t1) / 3600
        print(f"     进度 {done}/{len(order)} · 已用 {el:.1f}h · ETA {el / done * (len(order) - done):.1f}h", flush=True)
    print(f"全部完成：{len(order)} 锚点 · {dict(stats)}", flush=True)
    if stats["errors"]:
        raise RuntimeError(f"{stats['errors']} request failures; resume retries only failed rows")


if __name__ == "__main__":
    asyncio.run(main())
