"""Run frozen questions through a configured OpenAI/Anthropic-compatible API.

Set BENCHMARK_API_KEY and BENCHMARK_API_BASE_URL in the environment. The endpoint
must support the selected model's API shape and authorization protocol. Published
sampling settings and resumable raw-response schemas are retained. Credentials
are read only from the environment and are never written to response records.
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import gzip
import json
import os
import random
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from bench.parse_v2 import parse_rate  # noqa: E402
from bench.run_infer_v2 import SYSTEM, load_anchor  # noqa: E402  (constants only; no vllm import at module level)

GATEWAY = os.environ.get("BENCHMARK_API_BASE_URL", "https://api.openai.com")
THINK_BUDGET = 16384          # 与本地面板一致：答案预算之外再给这么多思考 token
EFFORT = "medium"
RETRIES = 6
TIMEOUT_S = 1800.0            # 170K token 的 prompt + 16K 思考，慢的时候要十几分钟

def _api_of(model: str) -> str:
    return "anthropic" if model.startswith("claude") else "openai"

def _scrub(s: str) -> str:
    # 任何异常文本里都不该出现密钥；保险起见把 Bearer 后面的 token 抹掉
    import re
    return re.sub(r"(Bearer\s+)\S+", r"\1***", s)

def key_from(path=None):
    key = os.environ.get("BENCHMARK_API_KEY", "").strip()
    if not key:
        raise RuntimeError("Set BENCHMARK_API_KEY in the environment")
    return key

def key_of(r: dict) -> tuple:
    return (r["cond"], r["task"], r.get("h"), r.get("op"), r.get("sub"), r.get("arm_kind"), r.get("fmt"))

# --------------------------------------------------------------------------
MIN_CACHE_CHARS = 6000        # 网关要求可缓存前缀 ≥1024 token；短于这个的前缀不打缓存标记

def shared_prefix(prompts: list[str]) -> str:
    """同一 (锚点, 条件) 下各行 prompt 的最长公共前缀，回退到行边界。
    v3 的 prompt = [primer][H 帧][STRATEGIES][题目]，任务 2 没有 STRATEGIES，所以公共前缀 = 帧块（约 89% 的字符）。"""
    if not prompts:
        return ""
    p = prompts[0]
    for q in prompts[1:]:
        i, m = 0, min(len(p), len(q))
        while i < m and p[i] == q[i]:
            i += 1
        p = p[:i]
    cut = p.rfind("\n")
    return p[:cut + 1] if cut > 0 else ""

def build_request(api: str, model: str, row: dict, prefix: str = "", cache_key: str = "") -> tuple[str, dict]:
    """prefix：该行所属组的公共前缀（见 shared_prefix）。给了就把它作为独立的缓存块 ——
    Anthropic 用 cache_control(ephemeral)（写 1.25×、读 0.1×），OpenAI 兼容口靠自动前缀缓存 + prompt_cache_key 路由。
    组内先单跑一行把缓存写热，再放其余行（见 do_group），否则并发的首批请求都各付一次写。"""
    system = row.get("system") or SYSTEM
    mt = max(1024, int(row["max_tokens"]))
    use_cache = bool(prefix) and len(prefix) >= MIN_CACHE_CHARS and row["prompt"].startswith(prefix)
    if api == "anthropic":
        if use_cache:
            content = [{"type": "text", "text": prefix, "cache_control": {"type": "ephemeral"}},
                       {"type": "text", "text": row["prompt"][len(prefix):]}]
        else:
            content = row["prompt"]
        body = {"model": model, "max_tokens": mt + THINK_BUDGET,
                "thinking": {"type": "adaptive"}, "output_config": {"effort": EFFORT},
                "system": system,
                "messages": [{"role": "user", "content": content}]}
        return "/v1/messages", body
    body = {"model": model, "reasoning_effort": EFFORT,
            "max_completion_tokens": mt + THINK_BUDGET,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": row["prompt"]}]}
    if use_cache and cache_key:
        body["prompt_cache_key"] = cache_key
    return "/v1/chat/completions", body

def parse_response(api: str, js: dict) -> dict:
    """-> {text, think_text, think_tok, n_gen, n_prompt, finish}"""
    if api == "anthropic":
        text, think = [], []
        for blk in js.get("content", []):
            if blk.get("type") == "text":
                text.append(blk.get("text", ""))
            elif blk.get("type") in ("thinking", "redacted_thinking"):
                think.append(blk.get("thinking", "") or "")
        u = js.get("usage", {}) or {}
        stop = js.get("stop_reason") or ""
        finish = {"end_turn": "stop", "max_tokens": "length", "stop_sequence": "stop"}.get(stop, stop or "stop")
        think_text = "\n".join(think)
        # 网关返回 usage.output_tokens_details.thinking_tokens（adaptive 思考不回传原文时这是唯一的思考量）
        think_tok = int(((u.get("output_tokens_details") or {}).get("thinking_tokens", 0)) or 0) or (len(think_text) // 4)
        return {"text": "\n".join(text).strip(), "think_text": think_text, "think_tok": think_tok,
                "n_gen": int(u.get("output_tokens", 0) or 0), "n_prompt": int(u.get("input_tokens", 0) or 0),
                "n_cached": int(u.get("cache_read_input_tokens", 0) or 0),
                "n_cache_write": int(u.get("cache_creation_input_tokens", 0) or 0), "finish": finish}
    ch = (js.get("choices") or [{}])[0]
    msg = ch.get("message", {}) or {}
    u = js.get("usage", {}) or {}
    det = u.get("completion_tokens_details") or {}
    think_text = msg.get("reasoning_content") or msg.get("reasoning") or ""
    think_tok = int(det.get("reasoning_tokens", 0) or 0) or (len(think_text) // 4)
    return {"text": (msg.get("content") or "").strip(), "think_text": think_text, "think_tok": think_tok,
            "n_gen": int(u.get("completion_tokens", 0) or 0), "n_prompt": int(u.get("prompt_tokens", 0) or 0),
            "n_cached": int((u.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0),
            "n_cache_write": 0, "finish": ch.get("finish_reason") or "stop"}

# --------------------------------------------------------------------------
async def main():
    global EFFORT
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="v3-bench-br")
    ap.add_argument("--model", required=True, help="claude-opus-5 | gpt-5.6-sol | …（网关模型名）")
    ap.add_argument("--raw", required=True, help="输出子目录，如 raw_opus5")
    ap.add_argument("--anchors", default="bench/v2/api_subset_120.txt")
    ap.add_argument("--conds", default="v1,v2")
    ap.add_argument("--tasks", default="", help="只跑这些任务，逗号分隔（3 按子题写 3-1a / 3-1b）")
    ap.add_argument("--concurrency", type=int, default=4, help="同时在飞的请求数（网关侧要求别开太高）")
    ap.add_argument("--cache", action="store_true",
                    help="开前缀缓存（默认关）。实测 2026-09-16：网关把请求散到多个后端，Opus 5 的缓存命中只有 ~30%%"
                         "（顺序发也一样），写 1.25× / 读 0.1× 的盈亏点是 22%%，基本白忙；GPT-5.6 Sol 经网关看不到任何缓存字段。")
    ap.add_argument("--limit_rows", type=int, default=0, help="调试：每个锚点只跑前 N 行")
    ap.add_argument("--max_anchors", type=int, default=0, help="调试：只跑前 N 个锚点")
    ap.add_argument("--gateway", default=GATEWAY)
    ap.add_argument("--key_file", default="")
    ap.add_argument("--keep_think", action="store_true", help="把思考原文存进 think_text（CoT 透镜用）")
    ap.add_argument("--effort", default="medium", help="思考力度：Anthropic output_config.effort / OpenAI reasoning_effort")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    EFFORT = a.effort

    api = _api_of(a.model)
    run = ROOT / "bench/runs" / a.run
    qd, rd = run / "questions", run / a.raw
    rd.mkdir(parents=True, exist_ok=True)
    files = sorted(qd.glob("*.jsonl.gz"))
    if a.anchors:
        keep = {l.strip() for l in open(ROOT / a.anchors) if l.strip()}
        files = [f for f in files if f.name in keep]
    if a.max_anchors:
        files = files[:a.max_anchors]
    conds = set(a.conds.split(",")) if a.conds else None
    tasks = set(a.tasks.split(",")) if a.tasks else None

    def wanted(r: dict) -> bool:
        if tasks is not None and (r["task"] + ("-" + r["sub"] if r.get("sub") else "")) not in tasks:
            return False
        return conds is None or r["cond"] in conds

    todo, n_rows = [], 0
    for f in files:
        have = set()
        if (rd / f.name).exists():
            have = {key_of(json.loads(l)) for l in gzip.open(rd / f.name, "rt")}
        rows = [r for r in load_anchor(f) if wanted(r) and key_of(r) not in have]
        if a.limit_rows:
            rows = rows[:a.limit_rows]
        if rows:
            todo.append((f, rows)); n_rows += len(rows)
    est_in = sum(len(r["prompt"]) for _, rows in todo for r in rows) / 3.3
    print(f"model {a.model} ({api}) · effort {EFFORT} · run {a.run} · 锚点 {len(files)} · 待跑 {len(todo)} 锚点 / {n_rows} 行 "
          f"· 估 prompt tokens {est_in/1e6:.1f}M · raw -> {rd}", flush=True)
    if a.dry or not todo:
        return
    key = key_from(a.key_file)
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    if api == "anthropic":
        headers["anthropic-version"] = "2023-06-01"
    sem = asyncio.Semaphore(a.concurrency)
    stats = collections.Counter()
    client = httpx.AsyncClient(base_url=a.gateway, headers=headers, timeout=httpx.Timeout(TIMEOUT_S, connect=30.0))

    async def one_call(row: dict, prefix: str = "", cache_key: str = "") -> dict:
        path, body = build_request(api, a.model, row, prefix, cache_key)
        last_err = "unknown"
        for attempt in range(RETRIES):
            try:
                async with sem:
                    resp = await client.post(path, json=body)
                if resp.status_code == 200:
                    out = parse_response(api, resp.json())
                    stats["ok"] += 1
                    return out
                last_err = f"http{resp.status_code}:{_scrub(resp.text[:160])}"
                if resp.status_code in (400, 401, 403, 404, 413, 422):     # 不可重试
                    break
            except Exception as e:                       # noqa: BLE001
                last_err = f"{type(e).__name__}:{_scrub(str(e))[:120]}"
            stats["retry"] += 1
            await asyncio.sleep(min(120, 5 * 2 ** attempt) + random.random() * 3)
        stats["fail"] += 1
        print(f"    ⚠️ 失败 {row['world']} t{row['t1']} {row['cond']}/{row['task']}: {last_err}", flush=True)
        return {"text": "", "think_text": "", "think_tok": 0, "n_gen": 0, "n_prompt": 0, "n_cached": 0,
                "n_cache_write": 0, "finish": f"error:{last_err.split(':')[0]}"}

    async def do_group(f: Path, rows: list[dict], idxs: list[int], outs: list):
        prefix = shared_prefix([rows[i]["prompt"] for i in idxs]) if (a.cache and len(idxs) > 1) else ""
        ck = f"{f.stem}-{rows[idxs[0]]['cond']}"
        idxs = sorted(idxs, key=lambda i: rows[i]["max_tokens"])      # 最短答案的那行先跑，把前缀写进缓存
        outs[idxs[0]] = await one_call(rows[idxs[0]], prefix, ck)
        rest = await asyncio.gather(*[one_call(rows[i], prefix, ck) for i in idxs[1:]])
        for i, o in zip(idxs[1:], rest):
            outs[i] = o

    async def do_anchor(f: Path, rows: list[dict]):
        t = time.time()
        outs = [None] * len(rows)
        groups = collections.defaultdict(list)
        for i, r in enumerate(rows):
            groups[r["cond"]].append(i)
        await asyncio.gather(*[do_group(f, rows, g, outs) for g in groups.values()])
        prev = [json.loads(l) for l in gzip.open(rd / f.name, "rt")] if (rd / f.name).exists() else []
        recs = []
        for r, o in zip(rows, outs):
            pr = parse_rate(r, o["text"])
            stats["gen"] += o["n_gen"]; stats["prompt"] += o["n_prompt"]; stats["think"] += o["think_tok"]
            stats["cached"] += o["n_cached"]; stats["cache_write"] += o.get("n_cache_write", 0)
            recs.append({k: r[k] for k in ("world", "tier", "t1", "cond", "task", "view")}
                        | {k: r[k] for k in ("set", "budget", "pomdp", "viewer", "primer", "oracle", "k",
                                             "op", "op_base", "is_placebo", "channel", "layer",
                                             "score_vars", "horizon_band", "op_target", "h", "h_1c",
                                             "sub", "arm_kind", "control_ref", "fmt",
                                             "horizons", "civs", "items", "pairs", "think")
                           if k in r}
                        | ({"think_text": o.get("think_text", "")} if a.keep_think else {})
                        | {"output": o["text"], "n_gen": o["n_gen"], "n_prompt": o["n_prompt"],
                           "n_cached": o["n_cached"], "n_cache_write": o.get("n_cache_write", 0),
                           "think_tok": o["think_tok"], "finish": o["finish"],
                           "parse_rate": pr, "model": a.model})
        tmp = (rd / f.name).with_suffix(".tmp")
        with gzip.open(tmp, "wt") as fh:
            for r in prev + recs:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        tmp.rename(rd / f.name)
        dt = time.time() - t
        print(f"  {f.stem[:44]:<44} {len(rows):>3} 调用 {dt:>6.0f}s · 解析 "
              f"{sum(x['parse_rate'] for x in recs)/len(recs):.2f} · 累计 prompt {stats['prompt']/1e6:.1f}M "
              f"(缓存读 {100*stats['cached']/max(1,stats['prompt']):.0f}%) "
              f"gen {stats['gen']/1e3:.0f}K think {stats['think']/1e3:.0f}K · 失败 {stats['fail']}", flush=True)

    t0 = time.time()
    # 锚点之间也并发（每个锚点 5–20 行；全局并发由 sem 控制），按 tier 轮转让任何时刻的子集都均衡
    by = collections.defaultdict(list)
    for f, rows in todo:
        by[f.name.split("-")[2]].append((f, rows))
    order = []
    for i in range(max(len(v) for v in by.values())):
        for tname in ("S", "M", "L", "G"):
            if i < len(by.get(tname, [])):
                order.append(by[tname][i])
    anchor_sem = asyncio.Semaphore(max(1, a.concurrency // 2))     # 同时最多处理 concurrency/2 个锚点

    async def guarded(f, rows):
        async with anchor_sem:
            await do_anchor(f, rows)
    for i in range(0, len(order), 8):
        await asyncio.gather(*[guarded(f, rows) for f, rows in order[i:i + 8]])
        done = min(i + 8, len(order)); el = (time.time() - t0) / 3600
        print(f"     进度 {done}/{len(order)} · 已用 {el:.1f}h · ETA {el/done*(len(order)-done):.1f}h", flush=True)
    await client.aclose()
    print(f"全部完成：{len(order)} 锚点 · ok {stats['ok']} · 重试 {stats['retry']} · 失败 {stats['fail']} · "
          f"prompt {stats['prompt']:,} (缓存读 {stats['cached']:,} · 缓存写 {stats['cache_write']:,}) · "
          f"gen {stats['gen']:,} · think≈{stats['think']:,}", flush=True)

if __name__ == "__main__":
    asyncio.run(main())
