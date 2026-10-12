"""单引擎推理驱动：pilot → 主网格 → 思考消融 → oracle，全在**一个**长驻进程里。

⚠️ 冷启动实测 **2,528 s ≈ 42 min**（DP 的两个 EngineCore 各自 compile + capture CUDA graph）。
所以这个脚本只起一次引擎，四个阶段共用它，断点续跑按锚点落文件。崩了重启只重算未完成的锚点。

★ **按锚点成组提交。** 一个锚点的 8–9 个条件 × 26 次调用一起下去，做完再下一个锚点。
散着提交 prefix cache 会被 LRU 顶掉，tier L 那 99K token 的 `[primer][H]` 就要反复重算 ——
那是几十倍的差别。

★ **两套解释器严格分离**：这个文件只读 `build_bench.py` 产出的 prompt JSONL，
**不 import 任何 `sandboxes.*` 或 `bench.render`**。nightly 那套依赖集是自洽的
（自带 torch 2.13.0+cu130 / transformers 5.16.1），被 `.venv` 的包污染后的现象是
"Model architectures ['Qwen4ExpForConditionalGeneration'] failed to be inspected"，
看着像不支持这个模型（手册 §1）。

⚠️ 判断是否卡死看 counter 有没有走，**不要看 tqdm 的 `s/it`** —— 那是平滑平均，加载阶段
曾显示 "ETA 87 分钟" 而真实耗时 9 分钟。日志里
`No available shared memory broadcast block found in 60 seconds` 与 CuteDSL 的
`AuxData cannot be converted to a JitArgument` **都是噪声不是错误**。
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import gzip
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# 模型卡的两套采样参数。非思考是主网格用的；思考只在子集上跑任务 2/3。
SAMPLE = {
    False: dict(temperature=0.7, top_p=0.80, top_k=20, presence_penalty=1.5),
    True:  dict(temperature=1.0, top_p=0.95, top_k=20),
}

# 与 bench/tasks.py 的 SYSTEM 一字不差 —— 这里重抄一份是为了让本文件**不 import**
# bench.tasks（它会拉进 sandboxes 侧的依赖，而 nightly 环境必须保持自洽，见模块 docstring）。
# ★ 思考预算与 reasoning effort，两者都是实测定的。
#
# 模板的默认 effort 是 **xhigh**，实测在 100K 上下文上 8,192 token 的思考块**根本不够收尾**：
# pilot 的 tier-S 锚点里任务 3 有 139/152 条顶到上限、答案一个字没输出，解析率 0.16。
# （非思考的 1a/1b/1c 同时是 1.00、零截断 —— 所以主网格没问题，坏的只有思考消融。）
#
# `chat_template.jinja` 暴露了 `reasoning_effort ∈ {xhigh, medium, low}`。取 **medium** +
# 16,384 的预算：xhigh 在这个上下文长度上不可负担（6,400 次调用 × 8K+ 思考 token 还收不了尾）。
# ⇒ 报告里必须写明"思考臂是 medium effort"，不能说成"开了思考"就完了。
THINK_BUDGET = 16384
THINK_EFFORT = "medium"

# ★ 思考消融只覆盖任务 2 与 **L4 算子**（`L4-trait` / `L4-love` 及其 placebo）。
# 理由：L4 干预的是**隐状态**，正是推理该起作用的地方，也是设计文档 §1.5 说的最有价值的两个；
# 把 19 个任务 3 调用全开思考要多花一倍时间去测那些 L1/L3 的可见变量干预。
THINK_ALL = False      # set by --think_all
FORCE_THINK = False    # set by --force_think
NO_THINK = False       # set by --no_think (ablation 2026-09-16: thinking OFF for every selected row)


def wants_think(row: dict) -> bool:
    # ★ 2026-09-12: `--think_all` 把思考扩到 1a/1b/1c。默认仍只覆盖任务 2 与 L4 算子（见上），
    # 那些 1a/1b/1c 的 think 行在默认模式下与 v1 逐字节相同、会被 `wanted()` 跳过；开了这个开关
    # 它们才真正带思考跑一遍，用来回答「测试时推理对标量预测有没有用」。输出务必写到单独的
    # --raw 目录，别污染冻结的 raw。
    # ★ 2026-09-12（用户决定）：thinking 要成为**基础设置**而不只是消融项。`think` 条件的题只出了
    # 160 个锚点，所以全量思考的做法是拿**同一批 v1/v2 题**开思考再跑一遍：prompt 逐字节不变，
    # 只有 chat template 的 enable_thinking 与采样参数变。输出必须写到单独的 --raw 目录。
    if NO_THINK:
        return False
    if FORCE_THINK:
        return True
    if not row.get("think"):
        return False
    return True if THINK_ALL else row["task"] in ("2", "3")

SYSTEM = ("You are analysing a Freeciv game. You are given observations of the world at "
          "several past turns, and a narrative of what happened between them. Answer only "
          "in the exact output format requested. Do not explain your reasoning outside the "
          "requested format, and do not omit any requested line.")


# --------------------------------------------------------------------------
# v2: parsers live in bench/parse_v2.py (shared with score_v2)
from bench.parse_v2 import parse_binary, parse_multi, parse_quantiles, parse_rate, parse_task2, split_task3  # noqa: E402,F401


# --------------------------------------------------------------------------
def load_anchor(path: Path) -> list[dict]:
    return [json.loads(l) for l in gzip.open(path, "rt")]


def prefill_of(row: dict) -> str:
    """Non-thinking rows only.  Task 3's answer is three blocks; pre-filling the first
    header keeps the model from opening with prose (v1 lesson: an un-anchored task-3
    answer was 80% prose and 20% parsable).  1a/1b/1c/2 start with a table anyway."""
    if row.get("think"):
        return ""
    if row["task"] == "3" and not row.get("sub"):      # v2 three-block prompt only; v3 rows are single-task
        return "##### 1A\n"
    return ""


KEEP_THINK = False     # set by --keep_think (CoT lens, 2026-09-15): store the <think> block as `think_text`
CLAMP_WINDOW = False   # set by --clamp_window (panel 2026-09-17): shrink max_tokens so prompt+answer fit max_model_len
KEEP_SPECIAL = False   # set by --keep_special (panel 2026-09-17): Gemma 4 marks thinking with the SPECIAL tokens
                       # `<|channel>thought` … `<channel|>`; with skip_special_tokens the thought text is glued onto the
                       # answer and _strip_think cannot see the boundary -- keep them in the decoded text instead


def _strip_think(text: str) -> tuple[str, int, str]:
    """thinking 模式会先输出 `<think>…</think>`。返回 (正文, 思考 token 粗估, 思考原文)。
    ⚠️ 2026-09-15 之前思考原文被直接丢弃 —— raw_tk 里没有任何 CoT 文本可评；CoT 透镜要 --keep_think。"""
    if "</think>" in text:
        a, b = text.split("</think>", 1)
        for t in ("<response>", "</response>",          # Kimi K3 (2026-09-17) wraps the answer channel in these tags
                  "<｜end▁of▁sentence｜>"):             # DeepSeek V4.1 EOS survives --keep_special
            b = b.replace(t, "")
        return b.strip(), len(a) // 4, a.replace("<think>", "", 1).strip()
    # gpt-oss (panel, 2026-09-17, harmony format, needs --keep_special): `<|channel|>analysis<|message|>…<|end|>
    # <|start|>assistant<|channel|>final<|message|>ANSWER`; the analysis channel is the reasoning
    if "<|channel|>final<|message|>" in text:
        a, b = text.rsplit("<|channel|>final<|message|>", 1)
        if "<|channel|>analysis<|message|>" in a:
            a = a.split("<|channel|>analysis<|message|>", 1)[1]
        a = a.replace("<|end|>", "").replace("<|start|>assistant", "").strip()
        for t in ("<|return|>", "<|end|>", "<|call|>"):
            b = b.replace(t, "")
        return b.strip(), len(a) // 4, a
    # Kimi K3 (panel, 2026-09-17, --keep_special): XTML tags decode as `<|open|> think <|sep|> … <|close|> think <|sep|>
    # <|open|> response <|sep|> ANSWER <|close|> response <|sep|>`; the prompt already opened the think tag
    if re.search(r"<\|close\|>\s*think\s*<\|sep\|>", text):
        a, b = re.split(r"<\|close\|>\s*think\s*<\|sep\|>", text, 1)
        b = re.sub(r"<\|open\|>\s*response\s*<\|sep\|>", "", b)
        b = re.split(r"<\|close\|>\s*response", b, 1)[0]
        b = re.sub(r"<\|(open|close|sep)\|>\s*(message|think|response)?\s*", "", b)
        return b.strip(), len(a) // 4, a.strip()
    # Inkling (panel, 2026-09-17, --keep_special): `<|content_thinking|>…<|end_message|><|message_model|><|content_text|>ANSWER<|end_message|>`
    if "<|content_text|>" in text:
        a, b = text.rsplit("<|content_text|>", 1)
        a = a.split("<|content_thinking|>", 1)[-1] if "<|content_thinking|>" in a else ""
        for t in ("<|end_message|>", "<|message_model|>", "<|content_model_end_sampling|>", "<|endoftext|>"):
            a = a.replace(t, ""); b = b.replace(t, "")
        return b.strip(), len(a) // 4, a.strip()
    # Gemma 4 (panel, 2026-09-17): thinking is a channel block `<|channel>thought\n…<channel|>` before the answer
    if "<channel|>" in text:
        a, b = text.rsplit("<channel|>", 1)
        a = a.split("<|channel>thought", 1)[-1] if "<|channel>thought" in a else a
        return b.strip(), len(a) // 4, a.strip()
    return text, 0, ""



def _marker(path, text: str) -> None:
    """Best-effort status file.  On the borrowed H20 nodes the run dir is a shared NAS where a marker written
    by one node's uid is unwritable by the other; that must not crash a finished run (2026-09-18)."""
    try:
        path.write_text(text)
    except OSError as e:
        print(f"⚠️ could not write {path.name}: {e} (results are complete; only the marker is missing)", flush=True)


async def main():
    global THINK_BUDGET          # referenced by an argparse default below, so declare it first
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="v2-bench")
    ap.add_argument("--pilot", type=int, default=4, help="pilot 用几个锚点（每 tier 一个）")
    ap.add_argument("--min_parse", type=float, default=0.80, help="pilot 解析率闸门")
    ap.add_argument("--max_model_len", type=int, default=262144)
    ap.add_argument("--max_num_seqs", type=int, default=256)
    # ★ 2026-09-10 夜实测：vLLM 在这套配置下默认 chunked prefill 的 max_num_batched_tokens=2048，
    # 每步只能预填 2K token。我们的 prompt 中位 45K、并发 640 个请求 ⇒ 完全被 prefill 卡住
    # （25 分钟 7M prompt token ≈ 4.7K tok/s），GPU 利用率忽高忽低。调到 32K。
    ap.add_argument("--max_num_batched_tokens", type=int, default=32768)
    # ---- 8x L20D 275 GB, run inside the vllm-openai:qwen38-flash-next
    # image (bench/infer_docker.sh).  One engine process, TP x DP = 8.  Flash-Next: TP 4 x DP 2
    # (the recipe qwen_vllm_testing/RESULTS.md validated on this box); Qwen3.8-27B: TP 1 x DP 8.
    ap.add_argument("--model", default=os.environ.get("MODEL", ""), help="模型目录（容器内路径）")
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--dp", type=int, default=2)
    ap.add_argument("--gpu_util", type=float, default=0.90, help="gpu_memory_utilization（卡独占时用）")
    ap.add_argument("--kv_gib", type=float, default=0.0, help=">0 时改为按字节给 KV 显存（共享卡）")
    ap.add_argument("--attn", default="", help="attention backend；空 = 让 vLLM 自选")
    ap.add_argument("--raw", default="raw", help="输出子目录；第二个模型用别的名字，如 raw_q38-27b")
    ap.add_argument("--only", default="", help="只跑这些 tier，逗号分隔")
    # ---- staged runs (user 2026-09-10 evening: first results needed next morning)
    ap.add_argument("--conds", default="", help="只跑这些条件，逗号分隔，如 v1,v2；空 = 全部")
    ap.add_argument("--frac", type=float, default=1.0,
                    help="每个 (集合, tier) 内按世界抽样的比例（排序后等距取），如 0.2；同一比例每次选同一批世界")
    # ★ 同时在飞的锚点数。pilot 实测一个锚点只有 27 个请求 ⇒ DP2 每个引擎 ~13 条序列，decode 只有
    # ~1,000 tok/s、ETA 32 h；KV 每组 227 GB（≈10M token）远没用满。4 个锚点并发仍按锚点成组提交，
    # 每个锚点自己的 [primer][H] 前缀照样连续命中。
    ap.add_argument("--concurrency", type=int, default=4, help="主体阶段同时处理的锚点数")
    ap.add_argument("--dry", action="store_true", help="不起引擎，只报计划")
    ap.add_argument("--think_all", action="store_true",
                    help="把 thinking 扩到 1a/1b/1c（默认只有任务 2 与 L4 算子）；请配合单独的 --raw 目录")
    ap.add_argument("--force_think", action="store_true",
                    help="对选中的所有行强制开思考（用于把 thinking 当基础设置跑全量 v1/v2）；"
                         "prompt 与非思考版逐字节相同，务必写到单独的 --raw 目录")
    ap.add_argument("--no_think", action="store_true",
                    help="对选中的所有行强制关闭思考（思考 vs 非思考消融；输出写到单独的 --raw 目录）")
    ap.add_argument("--anchors", default="", help="只跑这个文件里列出的锚点文件名（每行一个），用于消融子集")
    ap.add_argument("--tasks", default="", help="只跑这些任务，逗号分隔；任务 3 按子题写 3-1a / 3-1b（如 1b,3-1b）")
    ap.add_argument("--template_py", default="",
                    help="模型没有 Jinja 模板时（DeepSeek V4.1），给一个 Python 文件，定义 wrap(system, user, think) -> str；优先于 apply_chat_template")
    ap.add_argument("--think_budget", type=int, default=THINK_BUDGET,
                    help="思考 token 预算（默认 16384）。2026-09-18 用户要求 DeepSeek-V4.1 解析率 ≥90%%：它在 16K 下 40%% 的调用把预算烧在思考里，给 65536")
    ap.add_argument("--template_kw", default="",
                    help='额外 apply_chat_template kwargs，JSON，如 \'{"thinking": true, "thinking_effort": "high"}\'（Kimi K3 的自定义模板）')
    ap.add_argument("--engine_kw", default="",
                    help='额外 AsyncEngineArgs 字段，JSON，如 \'{"moe_backend":"marlin"}\'（gpt-oss 在 L20D 上要 Marlin MXFP4 路径）')
    ap.add_argument("--clamp_window", action="store_true",
                    help="把 max_tokens 压到 max_model_len 以内（128K 窗口的模型用；prompt 本身超窗仍让它失败）")
    ap.add_argument("--keep_special", action="store_true",
                    help="解码时保留特殊 token（Gemma 4 的思考边界是特殊 token；Qwen 系不需要）")
    ap.add_argument("--keep_think", action="store_true",
                    help="把 <think> 块原文存进输出行的 think_text（CoT 透镜用；默认只存 token 粗估）")
    a = ap.parse_args()
    global THINK_ALL, FORCE_THINK, KEEP_THINK
    THINK_ALL = bool(a.think_all)
    FORCE_THINK = bool(a.force_think)
    global NO_THINK
    NO_THINK = bool(a.no_think)
    KEEP_THINK = bool(a.keep_think)
    THINK_BUDGET = int(a.think_budget)
    if THINK_BUDGET != 16384:
        print(f"⚠️ thinking budget {THINK_BUDGET:,} (panel default 16,384) -- report this deviation", flush=True)
    global KEEP_SPECIAL, CLAMP_WINDOW
    KEEP_SPECIAL = bool(a.keep_special)
    CLAMP_WINDOW = bool(a.clamp_window)
    if not a.model:
        sys.exit("--model 或 MODEL 环境变量必须给")

    run = ROOT / f"bench/runs/{a.run}"
    qd, rd = run / "questions", run / a.raw
    rd.mkdir(parents=True, exist_ok=True)
    files = sorted(qd.glob("*.jsonl.gz"))
    if a.only:
        keep = set(a.only.split(","))
        files = [f for f in files if f.name.split("-")[2] in keep]
    if a.anchors:
        ap_ = Path(a.anchors)
        if not ap_.is_absolute() and not ap_.exists():
            ap_ = ROOT / a.anchors                       # inside the container the repo is mounted at /work = ROOT
        keep_files = {l.strip() for l in open(ap_) if l.strip()}
        files = [f for f in files if f.name in keep_files]
        print(f"--anchors: {len(files)} anchor files", flush=True)
    if a.frac < 1.0:
        # deterministic stratified subset: within each tier, sort the world ids and take every k-th
        by_w = collections.defaultdict(set)
        for f in files:
            by_w[f.name.split("-")[2]].add(f.name.rsplit("_", 1)[0])
        step = max(1, round(1.0 / a.frac))
        chosen = {w for t, ws in by_w.items() for w in sorted(ws)[::step]}
        files = [f for f in files if f.name.rsplit("_", 1)[0] in chosen]
        print(f"--frac {a.frac}: {len(chosen)} worlds, {len(files)} anchors", flush=True)
    conds = set(a.conds.split(",")) if a.conds else None
    tasks = set(a.tasks.split(",")) if a.tasks else None

    def wanted(r: dict) -> bool:
        if tasks is not None and (r["task"] + ("-" + r["sub"] if r.get("sub") else "")) not in tasks:
            return False
        # ⚠️ two independent gates hide the think 1a/1b/1c rows: this one and `wants_think`.
        # `--think_all` must open BOTH -- opening only wants_think ran task 2 alone (2026-09-12).
        if r["cond"] == "think" and r["task"] not in ("2", "3") and not THINK_ALL:
            return False            # byte-identical to v1 without thinking (user 2026-09-10)
        return conds is None or r["cond"] in conds

    def key_of(r: dict) -> tuple:
        # v3: task-3 rows are one prompt per (arm, sub-task) and a control arm shares the op id
        return (r["cond"], r["task"], r.get("h"), r.get("op"), r.get("sub"), r.get("arm_kind"), r.get("fmt"))

    # row-level resume: an anchor is "todo" if any wanted row has no output yet (earlier staged
    # runs may have written only some conditions of the file)
    todo = []
    for f in files:
        if not (rd / f.name).exists():
            todo.append(f); continue
        have = {key_of(json.loads(l)) for l in gzip.open(rd / f.name, "rt")}
        if any(key_of(r) not in have for r in load_anchor(f) if wanted(r)):
            todo.append(f)
    # pilot：每 tier 一个最小的锚点，先验证格式与吞吐
    by_tier = collections.defaultdict(list)
    for f in todo:
        by_tier[f.name.split("-")[2]].append(f)
    pilot = [by_tier[t][0] for t in ("S", "M", "L", "G") if by_tier[t]][:a.pilot]
    # ★★ 按 tier **轮转**，不按文件名排序。字母序会把 105 个 tier-L 世界全排在最前面
    # （实测 tier L 每锚点约 800 s ⇒ 约 23 小时），tier S/XL 要等到最后 —— 任何中断都
    # 只留下 tier L，而**逐 tier 对比是整个设计的核心**（1c 的 copy-forward 地板随 tier
    # 从 54.8% 单调降到 35.0%，per-tier horizon 是按游戏长度比例对齐的，绝不能跨 tier 池化）。
    # 轮转之后任何时刻停下来，手上都是四个 tier 均衡的可用子集。
    rest_by = collections.defaultdict(list)
    for f in todo:
        if f not in pilot:
            rest_by[f.name.split("-")[2]].append(f)
    rest = []
    for i in range(max((len(v) for v in rest_by.values()), default=0)):
        for t in ("S", "M", "L", "G"):
            if i < len(rest_by.get(t, [])):
                rest.append(rest_by[t][i])
    print("  轮转后前 8 个锚点的 tier:",
          [f.name.split("-")[2] for f in rest[:8]], flush=True)
    print(f"锚点 {len(files)} · 待跑 {len(todo)} · pilot {len(pilot)} · 主体 {len(rest)}",
          flush=True)
    if a.dry:
        return

    from transformers import AutoTokenizer
    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM

    # ★★ 必须走 chat template。这是个 **instruct** 模型（带 chat_template.jinja 与
    # `enable_thinking` 开关）。把 prompt 当纯文本喂进去会同时坏三件事：system 消息不生效、
    # 没有 `<|im_start|>` 标记（模型落到 completion 模式）、**`enable_thinking` 完全失效**
    # —— 于是"思考 vs 非思考"这个消融根本不存在，而从输出上看不出来。
    hf = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)   # Kimi K3 ships a custom tokenizer class
    print(f"model {a.model} · TP {a.tp} x DP {a.dp} · max_model_len {a.max_model_len:,} · raw -> {rd}", flush=True)

    tmpl_kw = json.loads(a.template_kw) if a.template_kw else {}
    custom_wrap = None
    if a.template_py:
        import importlib.util as _ilu
        _spec = _ilu.spec_from_file_location("custom_template", a.template_py)
        _mod = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_mod)
        custom_wrap = _mod.wrap
        print(f"custom prompt template: {a.template_py}", flush=True)

    def wrap(user: str, think: bool, system: str = "") -> str:
        if custom_wrap is not None:
            return custom_wrap(system or SYSTEM, user, think)
        kw = {"reasoning_effort": THINK_EFFORT} if think else {}
        kw.update(tmpl_kw)
        return hf.apply_chat_template(
            [{"role": "system", "content": system or SYSTEM}, {"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True, enable_thinking=think, **kw)

    t0 = time.time()
    eng_kw = dict(model=a.model, dtype="bfloat16",
                  data_parallel_size=a.dp, tensor_parallel_size=a.tp,
                  enable_expert_parallel=False,                   # DP+TP+EP 三者同开必崩
                  max_model_len=a.max_model_len, max_num_seqs=a.max_num_seqs,
                  max_num_batched_tokens=a.max_num_batched_tokens,
                  enable_prefix_caching=True,                     # ★ 同锚点共享 [primer][H]
                  trust_remote_code=True, disable_log_stats=True)
    if a.kv_gib > 0:
        eng_kw["kv_cache_memory_bytes"] = int(a.kv_gib * (1 << 30))   # 共享机器：给字节数
    else:
        eng_kw["gpu_memory_utilization"] = a.gpu_util                # 独占：给比例
    if a.attn:
        eng_kw["attention_backend"] = a.attn
    if a.engine_kw:
        eng_kw.update(json.loads(a.engine_kw)); print("extra engine kwargs:", a.engine_kw, flush=True)
    engine = AsyncLLM.from_engine_args(AsyncEngineArgs(**eng_kw))
    print(f"\n★ engine up in {time.time()-t0:.0f} s\n", flush=True)

    tok_stats = collections.Counter()
    parse_stats = collections.defaultdict(list)

    async def one_call(row, rid):
        think = wants_think(row)
        # ★★ 思考模式必须额外给思考块的预算。`max_tokens` 是按**答案**长度定的
        # （任务 3 只有 100–300 token），而 thinking 的模板注入了
        # "Reasoning effort is set to xhigh" —— 思考块会吃光预算，答案输出为空，
        # 于是整个思考消融静默变成"模型答不出来"。同时给所有任务一个 256 的下限。
        # ⚠️ 下限 1024 而不是 256。任务 3 占全部题目的 82%，而它的 `max_tokens` 是按**答案**
        # 长度定的（几个变量 × 几个 horizon ≈ 100–300 token）。实测这个模型在任务 2 上会先写
        # 一段"We need answer in exact format only. Need analyze…"的散文再作答 —— 256 的预算
        # 会被前言吃光、解析率归零，而那会静默毁掉 82% 的数据。生成遇到 EOS 就停，
        # 所以对老实作答的调用这个下限**不花额外成本**，它只在模型跑偏时救场。
        mt = max(1024, row["max_tokens"]) + (THINK_BUDGET if think else 0)
        # system 消息是常量、在最前面 ⇒ 不影响 [primer][H] 的 prefix 共享
        pre = "" if think else prefill_of(row)
        if CLAMP_WINDOW:
            # panel 2026-09-17 (gpt-oss 128K): keep prompt + budget inside max_model_len when the prompt itself fits;
            # an over-long prompt is left alone so the engine rejects it and the row counts as a failed call
            n_in = len(hf(wrap(row["prompt"], think, row.get("system") or "") + pre, add_special_tokens=False)["input_ids"])
            if n_in < a.max_model_len:
                mt = max(64, min(mt, a.max_model_len - n_in - 8))
        sp = SamplingParams(max_tokens=mt, skip_special_tokens=not KEEP_SPECIAL, **SAMPLE[think])
        last = None
        async for out in engine.generate(wrap(row["prompt"], think, row.get("system") or "") + pre,
                                         sp, request_id=rid):
            last = out
        o = last.outputs[0]
        text, think_tok, think_text = _strip_think(o.text)
        if pre:
            text = pre + text        # 前缀补回去，否则解析器看不到首行
        return {"text": text, "n_gen": len(o.token_ids), "think_tok": think_tok,
                "think_text": (think_text if KEEP_THINK else ""),
                "finish": o.finish_reason,
                "n_prompt": len(last.prompt_token_ids or []),
                "n_cached": getattr(last, "num_cached_tokens", 0) or 0}

    async def do_anchor(f: Path, tag: str) -> dict:
        # the `think` condition only differs from v1 on tasks 2/3 (wants_think); its 1a/1b/1c
        # rows are byte-identical v1 prompts run without thinking -- skipped (user 2026-09-10)
        prev = [json.loads(l) for l in gzip.open(rd / f.name, "rt")] if (rd / f.name).exists() else []
        have = {key_of(r) for r in prev}
        rows = [r for r in load_anchor(f) if wanted(r) and key_of(r) not in have]
        if not rows:
            return {"dt": 0.0, "gen": 0, "n": 0}
        t = time.time()
        # 按条件成组，让同一个 [primer][H] 前缀连续命中
        rows.sort(key=lambda r: (r["cond"], r["task"] != "1a"))
        # ★ return_exceptions=True 是这个长跑能无人值守的前提。默认下**任何一个**请求抛异常
        # （prompt 超限、引擎瞬时错误、取消）都会让 gather 上抛，穿过 main 把引擎一起带走 ——
        # 15 小时的跑可能在第 3 小时死掉而四张卡空转到早上。失败的请求记成空输出，
        # 让它在评分里表现为"解析失败"（有专门的口径报缺失率），而不是让整批消失。
        # ★ 前缀预热（2026-09-10 夜实测）：同一 (锚点, 条件) 的 5 道题共享 [primer][H] 前缀，
        # 但一起提交时它们**同时**预填，谁也命中不了谁 —— 64 并发下缓存命中率从 69% 掉到 19%，
        # 每个前缀被算 5 遍。改为每组先单独跑输出最短的那道（任务 2，~350 token），让前缀进缓存，
        # 再放其余 4 道；不同条件、不同锚点之间仍然并发。
        async def one_group(idxs: list[int]):
            idxs = sorted(idxs, key=lambda i: rows[i]["max_tokens"])
            out = {}
            out[idxs[0]] = await _safe(one_call(rows[idxs[0]], f"{f.stem}_{idxs[0]}"))
            rest = await asyncio.gather(*[_safe(one_call(rows[i], f"{f.stem}_{i}")) for i in idxs[1:]])
            out.update(dict(zip(idxs[1:], rest)))
            return out

        async def _safe(coro):
            try:
                return await coro
            except BaseException as e:      # noqa: BLE001 -- recorded as a failed call below
                return e

        groups = collections.defaultdict(list)
        for i, r in enumerate(rows):
            groups[r["cond"]].append(i)
        parts = await asyncio.gather(*[one_group(g) for g in groups.values()])
        merged = {}
        for d in parts:
            merged.update(d)
        res = [merged[i] for i in range(len(rows))]
        nerr = sum(1 for o in res if isinstance(o, BaseException))
        if nerr:
            first = next(o for o in res if isinstance(o, BaseException))
            print(f"    ⚠️ {f.stem}: {nerr}/{len(rows)} 个请求失败，首个："
                  f"{type(first).__name__}: {str(first)[:160]}", flush=True)
        res = [o if not isinstance(o, BaseException)
               else {"text": "", "n_gen": 0, "think_tok": 0, "think_text": "",
                     "finish": f"error:{type(o).__name__}", "n_prompt": 0, "n_cached": 0}
               for o in res]
        dt = time.time() - t
        recs = []
        for r, o in zip(rows, res):
            pr = parse_rate(r, o["text"])
            tok_stats["gen"] += o["n_gen"]; tok_stats["prompt"] += o["n_prompt"]
            tok_stats["cached"] += o["n_cached"]; tok_stats["think"] += o["think_tok"]
            # ★ 解析率按**模式**分开累计：思考消融只占 12% 的题，不该让它的问题
            # 中止 299 世界的主网格（第一次 pilot 就是这样被 0.31 误判的）。
            parse_stats[("think" if wants_think(r) else "plain", r["task"])].append(pr)
            recs.append({k: r[k] for k in ("world", "tier", "t1", "cond", "task", "view")}
                        | {k: r[k] for k in ("set", "budget", "pomdp", "viewer", "primer", "oracle", "k",
                                             "op", "op_base", "is_placebo", "channel", "layer",
                                             "score_vars", "horizon_band", "op_target", "h", "h_1c",
                                             "sub", "arm_kind", "control_ref", "fmt",
                                             "horizons", "civs", "items", "pairs", "think")
                           if k in r}
                        | ({"think_text": o.get("think_text", "")} if KEEP_THINK else {})
                        | {"output": o["text"], "n_gen": o["n_gen"],
                           "n_prompt": o["n_prompt"], "n_cached": o["n_cached"],
                           "think_tok": o["think_tok"], "finish": o["finish"],
                           "parse_rate": pr})
        tmp = (rd / f.name).with_suffix(".tmp")
        with gzip.open(tmp, "wt") as fh:
            for r in prev + recs:                  # keep rows from earlier staged runs
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        tmp.rename(rd / f.name)
        gen = sum(o["n_gen"] for o in res)
        print(f"  [{tag}] {f.stem[:44]:<44} {len(rows):>4} 调用 {dt:>7.1f}s "
              f"gen {gen:>7,} ({gen/dt:>6.0f} tok/s) 解析 "
              f"{sum(parse_rate(r,o['text']) for r,o in zip(rows,res))/len(rows):.2f}",
              flush=True)
        return {"dt": dt, "gen": gen, "n": len(rows)}

    # ---------------- pilot ----------------
    print("=" * 78 + f"\npilot：{len(pilot)} 个锚点\n" + "=" * 78, flush=True)
    pt = time.time(); pg = 0; pn = 0
    for f in pilot:
        r = await do_anchor(f, "pilot")
        pg += r["gen"]; pn += r["n"]
    pdt = time.time() - pt
    plain = [x for k, v in parse_stats.items() if k[0] == "plain" for x in v]
    thk = [x for k, v in parse_stats.items() if k[0] == "think" for x in v]
    # ★ 闸门只看非思考的主网格；`--force_think` 时全部行都是思考臂（plain 为空），那就按思考臂算
    # （2026-09-15：v3 首次启动因 plain 为空被判 0.00 中止，白跑 20 分钟 pilot）
    overall = sum(plain) / len(plain) if plain else (sum(thk) / len(thk) if thk else 0.0)
    print(f"\npilot 用时 {pdt:.0f}s · 生成 {pg:,} tok ({pg/pdt:.0f} tok/s)"
          f" · prompt {tok_stats['prompt']:,} · 缓存命中 {tok_stats['cached']:,}"
          f" ({100*tok_stats['cached']/max(1,tok_stats['prompt']):.0f}%)")
    print("逐（模式, 任务）解析率：" + " ".join(
        f"{k[0]}/{k[1]} {sum(v)/len(v):.2f}(n={len(v)})" for k, v in sorted(parse_stats.items())))
    print(f"闸门口径 = **非思考**主网格解析率 {overall:.2f}"
          + (f" · 思考臂 {sum(thk)/len(thk):.2f}（n={len(thk)}，不参与闸门）" if thk else ""))
    # ⚠️ `--pilot 0` 时 plain 为空、overall 退化成 0.0，会误判成"解析率不合格"直接中止。
    # pilot 已经在 tier S/M 上过了（解析率 1.00），重启时跳过它是刻意的。
    if pilot and overall < a.min_parse:
        print(f"\n✗ 解析率 {overall:.2f} < 闸门 {a.min_parse} —— 中止。样本：", flush=True)
        for f in pilot[:1]:
            for r in load_anchor(f)[:1]:
                pass
        (run / "pilot_failed.json").write_text(json.dumps(
            {"overall_plain": overall,
             "by_mode_task": {f"{k[0]}/{k[1]}": sum(v)/len(v) for k, v in parse_stats.items()}},
            ensure_ascii=False, indent=1))
        engine.shutdown()
        return
    if pilot:
        print(f"\n✓ 解析率 {overall:.2f} ≥ {a.min_parse} —— 继续主体\n", flush=True)
    else:
        print("\n（--pilot 0：跳过 pilot，沿用上一轮 tier S/M 的解析率 1.00）\n", flush=True)
    _marker(run / "pilot.json", json.dumps(
        {"overall_parse_plain": overall,
         "parse_think": (sum(thk)/len(thk) if thk else None),
         "think_effort": THINK_EFFORT, "think_budget": THINK_BUDGET,
         "by_mode_task": {f"{k[0]}/{k[1]}": sum(v)/len(v) for k, v in parse_stats.items()},
         "gen_tok_s": pg/pdt, "cache_hit": tok_stats['cached']/max(1,tok_stats['prompt']),
         "n_calls": pn, "wall_s": pdt}, ensure_ascii=False, indent=1))

    # ---------------- 主体 ----------------
    print("=" * 78 + f"\n主体：{len(rest)} 个锚点\n" + "=" * 78, flush=True)
    t1 = time.time(); done = 0; failed = []
    sem = asyncio.Semaphore(max(1, a.concurrency))

    async def run_one(i: int, f: Path):
        nonlocal done
        async with sem:
            try:
                await do_anchor(f, f"{i+1}/{len(rest)}")
            except Exception as e:      # 一个坏锚点不该让剩下 200 个不跑
                failed.append((f.name, f"{type(e).__name__}: {e}"))
                print(f"  ✗ 锚点 {f.name} 整体失败：{type(e).__name__}: {e}", flush=True)
            done += 1
            el = time.time() - t1
            eta = el / done * (len(rest) - done)
            print(f"     进度 {done}/{len(rest)} · 已用 {el/3600:.1f}h · ETA {eta/3600:.1f}h",
                  flush=True)

    await asyncio.gather(*[run_one(i, f) for i, f in enumerate(rest)])
    print(f"\n全部完成：{done} 个锚点（失败 {len(failed)}）· 生成 {tok_stats['gen']:,} tok"
          f" · 思考 {tok_stats['think']:,} tok"
          f" · prompt {tok_stats['prompt']:,} · 缓存命中 "
          f"{100*tok_stats['cached']/max(1,tok_stats['prompt']):.0f}%")
    if failed:
        print("失败的锚点（重跑本脚本会自动只补这些）：")
        for n, e in failed[:10]:
            print(f"  {n}  {e}")
    _marker(run / "infer_done.json", json.dumps(
        {"n_anchors_done": done, "failed": failed, "tok": dict(tok_stats),
         "cache_hit": tok_stats["cached"] / max(1, tok_stats["prompt"])},
        ensure_ascii=False, indent=1))
    engine.shutdown()


if __name__ == "__main__":        # DP 会 spawn worker 进程，必须有守卫
    asyncio.run(main())
