"""v3 review export: for each evaluation setting, a few anchors with EVERY question type,
written as markdown review sheets (question part + truth + how it is scored) plus the full
prompt as .prompt.txt.  For the user's step-2 check ("构造目前的题目，采样给我看").

    .venv/bin/python bench/export_samples_v3.py --run v3-bench --per_set 3
"""
from __future__ import annotations

import argparse
import collections
import gzip
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from bench.spec_v2 import SCORE_VARS, SYSTEM, TRAITS  # noqa: E402

SETS = [("bench-basic", "v1", "基础集·全知"), ("bench-gen", "v1", "泛化集·全知"), ("bench-basic", "v2", "基础集·自我视角(POMDP)")]

NOTE = {
 "1a": "**评分**：每个 (文明, 量, h) 一行五分位 → 归一化 CRPS；技能 = 1 − Σ模型CRPS / Σ趋势外推CRPS（外推用三帧的最后两帧线性外推，"
       "自我视角只用视角文明能看到的量）。classic 世界的 culture_gain 不计分（恒为 0）。gold/population/culture_gain 在 log1p 尺度上算。",
 "1b": "**评分**：Brier 对参照线 = 该 (设置, 条件, 模板, 预算, h, t1) 格在**本题集内**的发生率（表中 `cell` 列）。"
       "头条口径改为相对 Brier 技能 = (参照Brier − 模型Brier)/参照Brier。`leader` 题单列、不进总分（参照线 = 持续法）。"
       "所有格的发生率都在 10–90% 且 n≥200，由 finalize_1b + 闸门 ⑨ 保证。",
 "2": "**评分**：人格 = 世界内成对排序准确率（只看真值不同的文明对，`=` 算 0.5，基准 0.5；理想观察者 0.565–0.609）；"
      "态度 = 三分类准确率对多数类；世界规则 = 三分类准确率对多数类，只在有证据的锚点上问（`param_evidence`）。瘟疫题已删除（无可观测证据）。",
 "3": "**评分**：只问效应带内的 h。A 效应捕捉 = 1 − ΣCRPS(模型 vs 真臂真值)/ΣCRPS(对照答案 vs 真臂真值)，只在算子 `score_vars` × 目标文明 × 真值确实变动的格上；"
      "对照答案 = 同格式无干预题的模型答案（`control_ref`：与主网格 1a 相同时直接用主网格 1a）。B 干净性 = 在真值未变动且非计分的格上，"
      "1 − ΣCRPS(模型)/ΣCRPS(对照)（+1 完美、0 照抄、<0 幻觉副作用）。真臂减安慰剂臂是干净读数。方向正确率另报。",
}


def fmt_truth(r) -> str:
    t = r["task"]
    if t == "1a":
        out = ["| civ | h | " + " | ".join(k for k in next(iter(next(iter(r["truth"].values())).values())).keys()) + " |"]
        out.append("|" + "---|" * (2 + 11))
        for h, d in r["truth"].items():
            for civ, v in d.items():
                out.append(f"| {civ} | +{h} | " + " | ".join("—" if x is None else str(x) for x in v.values()) + " |")
        return "\n".join(out)
    if t == "1b":
        return None      # rendered with the items table
    if t == "2":
        L = []
        for dim, d in (r["truth"].get("traits") or {}).items():
            order = sorted(d.items(), key=lambda kv: -kv[1])
            L.append(f"- **{dim}** 真值排序: " + " > ".join(f"{c} ({v})" for c, v in order))
        for pair, lab in (r["truth"].get("love") or {}).items():
            L.append(f"- 态度 {pair}: **{lab}**")
        for pn, b in (r["truth"].get("params") or {}).items():
            L.append(f"- 规则 {pn}: 档位 {b}（0/1/2 = 低/中/高）  证据计数 {(r.get('param_evidence') or {}).get(pn, '—')}")
        return "\n".join(L)
    if t == "3":
        sub = r["sub"]
        if sub == "1a":
            hz = r["horizons"]; tgt = r.get("op_target"); sv = r.get("score_vars") or []
            L = [f"算子 `{r['op_base']}` · 臂 **{r.get('arm_kind')}** · 效应带 {r.get('horizon_band')} · 问的 h {hz} · 目标文明 **{tgt}** · 计分变量 {sv} · 对照来源 {r.get('control_ref')}",
                 "", "目标文明在计分变量上的三条真值（真臂 / 事实臂 / 主线）：", "", "| h | var | 真臂 | 事实臂 | 主线 |", "|---|---|---|---|---|"]
            for h in hz:
                for v in sv:
                    a = (r["truth"]["1a"].get(str(h)) or {}).get(tgt) or {}
                    f = ((r.get("factual_arm") or {}).get("1a", {}).get(str(h)) or {}).get(tgt) or {}
                    m = ((r.get("mainline") or {}).get("1a", {}).get(str(h)) or {}).get(tgt) or {}
                    L.append(f"| +{h} | {v} | {a.get(v)} | {f.get(v)} | {m.get(v)} |")
            L += ["", "全部文明的真臂真值：", ""]
            first = next(iter(next(iter(r["truth"]["1a"].values())).values()))
            L.append("| civ | h | " + " | ".join(first.keys()) + " |"); L.append("|" + "---|" * 13)
            for h, d in r["truth"]["1a"].items():
                for civ, v in d.items():
                    L.append(f"| {civ} | +{h} | " + " | ".join("—" if x is None else str(x) for x in v.values()) + " |")
            return "\n".join(L)
        return None
    return json.dumps(r["truth"], ensure_ascii=False)[:3000]


def items_table(r, cells) -> str:
    L = ["| Q | 模板 | 文明 | h | 真值 | 事实臂 | 主线 | cell 发生率 (n) |", "|---|---|---|---|---|---|---|---|"]
    tr = r["truth"] if r["task"] == "1b" else r["truth"]["1b"]
    fa = (r.get("factual_arm") or {}).get("1b") if r["task"] == "3" else None
    ml = (r.get("mainline") or {}).get("1b") if r["task"] == "3" else None
    for i, (it, y) in enumerate(zip(r["items"], tr), 1):
        c = cells.get("|".join([r["set"], r["cond"], it["kind"], str(r["budget"]), str(it["h"]), str(r["t1"])]))
        cs = f"{c['rate']:.2f} ({c['n']})" if c else "—" if it["kind"] != "leader" else "多分类"
        L.append(f"| Q{i} | {it['kind']} | {it['civ']} | +{it['h']} | {y} | {fa[i-1] if fa else '—'} | {ml[i-1] if ml else '—'} | {cs} |")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="v3-bench"); ap.add_argument("--per_set", type=int, default=3)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    run = ROOT / "bench/runs" / a.run
    out = pathlib.Path(a.out) if a.out else run / "samples_review"
    out.mkdir(parents=True, exist_ok=True)
    cp = ROOT / f"bench/v2/cells_1b_insample_{a.run}.json"
    cells = json.load(open(cp)) if cp.exists() else {}
    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(str(ROOT / "model_states/Qwen3.8-Flash-Next/tokenizer.json"))
    files = sorted((run / "questions").glob("*.jsonl.gz"))
    # index anchors: (set, cond) -> [(budget, t1, file)] restricted to anchors that have task 3
    by = collections.defaultdict(list)
    for f in files:
        rows = [json.loads(l) for l in gzip.open(f, "rt")]
        if not rows:
            continue
        has3 = {r["cond"] for r in rows if r["task"] == "3" and r.get("arm_kind") == "real"}
        ops = {r["op_base"] for r in rows if r["task"] == "3"}
        for st, cond, _ in SETS:
            if rows[0]["set"] == st and cond in has3:
                by[(st, cond)].append((rows[0]["budget"], rows[0]["t1"], f, tuple(sorted(ops))))
    index = ["# v3 题目抽样审阅\n", f"run `{a.run}` · 每个设置 {a.per_set} 个锚点（预算 400/800/1200 各一，取第 2–3 个锚点即中局），每锚点列出全部题型。",
             "每道题一份 `.md`（题面部分 + 真值 + 评分口径）和一份 `.prompt.txt`（完整 prompt = system 消息 + user 消息，含观测帧）。",
             "所有题共用同一条 system 消息（`spec_v2.SYSTEM`，推理时由 `run_infer_v2` 发送；思考模式下只多开 enable_thinking）：\n", "```", SYSTEM, "```\n"]
    used_ops = collections.Counter()
    for st, cond, label in SETS:
        cands = sorted(by[(st, cond)])
        pick = []
        for budget in (400, 800, 1200):
            cs = [c for c in cands if c[0] == budget]
            if not cs:
                continue
            # prefer anchors whose operator is not yet shown, mid-game anchor (2nd of the world)
            cs.sort(key=lambda c: (min(used_ops[o] for o in c[3]) if c[3] else 9, c[1]))
            mid = [c for c in cs if c[1] > 100] or cs
            pick.append(mid[len(mid) // 3] if len(mid) > 2 else mid[0])
            for o in pick[-1][3]:
                used_ops[o] += 1
        index.append(f"\n## {label}  (`{st}` / `{cond}`)\n")
        for budget, t1, f, ops in pick[:a.per_set]:
            rows = [json.loads(l) for l in gzip.open(f, "rt")]
            wid = rows[0]["world"]
            index.append(f"\n### {wid}  t₁={t1}  预算 {budget}  算子 {', '.join(ops)}\n")
            index.append("| 题 | 类型 | prompt tokens | 其中题面 | max_tokens | 文件 |\n|---|---|---|---|---|---|")
            for r in rows:
                if r["cond"] != cond or r.get("error"):
                    continue
                if r["task"] == "3":
                    kind = f"3·{r['sub']}·{r['arm_kind']}"
                else:
                    kind = r["task"]
                name = f"{label.split('·')[0]}_{cond}_{budget}_t{t1}_{kind.replace('·', '-')}"
                qpos = r["prompt"].find("=====  COUNTERFACTUAL")
                if qpos < 0:
                    qpos = r["prompt"].find("=====  TASK")
                obs, q = r["prompt"][:qpos], r["prompt"][qpos:]
                n_all = len(tk.encode(r["prompt"]).ids); n_q = len(tk.encode(q).ids)
                # exactly what run_infer_v2 sends: one system message + one user message (chat template applied by vLLM)
                (out / f"{name}.prompt.txt").write_text("===== SYSTEM MESSAGE =====\n" + SYSTEM + "\n\n===== USER MESSAGE =====\n" + r["prompt"])
                md = [f"# {kind} · {label} · {wid} · t₁={t1}\n",
                      f"prompt {n_all:,} tokens（观测部分 {n_all - n_q:,}，题面 {n_q:,}）· max_tokens {r['max_tokens']:,} · 完整 prompt 见 `{name}.prompt.txt`\n",
                      NOTE[r["task"]], "\n## 题面（观测帧之后的部分）\n", "```", q, "```", "\n## 真值\n"]
                if r["task"] == "1b" or (r["task"] == "3" and r["sub"] == "1b"):
                    if r["task"] == "3":
                        md.append(f"算子 `{r['op_base']}` · 臂 **{r['arm_kind']}** · 问的 h {r['horizons']} · 对照来源 {r.get('control_ref')}\n")
                    md.append(items_table(r, cells))
                else:
                    md.append(fmt_truth(r) or "")
                (out / f"{name}.md").write_text("\n".join(md))
                index.append(f"| {kind} | {r['task']} | {n_all:,} | {n_q:,} | {r['max_tokens']:,} | [{name}.md]({name}.md) |")
    (out / "README.md").write_text("\n".join(index))
    print("written", out, "files:", len(list(out.glob('*.md'))))


if __name__ == "__main__":
    main()
