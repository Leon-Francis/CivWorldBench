#!/usr/bin/env python
"""Generate the result tables and figures of the paper from score files.

Reads the approved failure-scored files in PANEL_SCORE_TAGS for every registered
line, bench/v2/ceiling_v3_pt.json (re-seeded-rerun ceiling), bench/v2/trait_ceiling.json
(ideal-observer ceiling for the personality probe) and
bench/runs/cot-v32-{190,sol30}/ (aligned process lens); writes

  paper/tables/main_all.tex       -- main table: every line once on 1a / 1b / L2 / L3 (direction)
  paper/tables/l3_capture.tex     -- appendix: L3 effect-capture ratios real (placebo) per line
  paper/tables/families.tex       -- 1a skill per quantity family (not used in the main text)
  paper/figures/radar.pdf         -- skill profile, every line once, axes = % of the way from the
                                     reference predictor to the attainable ceiling
  paper/figures/onebee_templates.pdf -- 1b skill ratio per event template, lines vs ceiling
  paper/figures/cot.pdf           -- process lens: mismatch, effect-error association, Capture

Lines whose score file does not exist yet are still listed with empty cells, so everything can
be regenerated as the panel completes.  Run from the repository root:
    .venv/bin/python paper/gen_tables.py
"""
from __future__ import annotations
import json, math, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "bench/runs/v3-bench-br"
CEIL = ROOT / "bench/v2/ceiling_v3_pt.json"
TRAIT_CEIL = ROOT / "bench/v2/trait_ceiling.json"
COT = ROOT / "bench/runs/cot-v31-main/cot_scores_raw_tk.json"
TRAIT_CEIL_V3 = ROOT / "bench/v2/trait_ceiling_v3.json"     # ideal observer on the v3 anchors (per set)
RULES_CEIL = ROOT / "bench/v2/rules_ceiling.json"           # disasters: Poisson-limited observer; foodbox/sciencebox: 1.0
DIR_CEIL = ROOT / "bench/v2/l3_direction_ceiling.json"      # direction: placebo-divergence-limited
# Ceilings fixed by decision 2026-09-21 (doc 14 B23 补 10), overriding the measured files above:
#   TRT  0.66 = the ideal observer on pairs whose true traits differ by >= 40 points (trait_ceiling_v3.json,
#               gap>=40, upper end of \ceilTraitsGapForty); the all-pairs value 0.59 is kept in the JSON.
#   dir  0.92 = direction identifiability on the cells whose true shift exceeds the placebo divergence
#               (l3_direction_ceiling.json, |d| > sigma); the all-moved-cells value 0.71 is kept in the JSON.
CEIL_OVERRIDE = {"traits": 0.66, "dir": 0.92}
EFFECT_CEIL = ROOT / "bench/v2/l3_effect_ceiling.json"      # capture (S): oracle-mean forecaster vs placebo divergence
OUT_T = ROOT / "paper/tables"; OUT_F = ROOT / "paper/figures"
SETTINGS = [("bench-basic|v1", "omni-ID"), ("bench-basic|v2", "ego"), ("bench-gen|v1", "OOD")]
FAMS = ["economy", "scale", "tech_culture", "military", "diplomacy", "overall"]
FAM_LABEL = {"economy": "economy", "scale": "scale", "tech_culture": "tech--culture",
             "military": "military", "diplomacy": "diplomacy", "overall": "overall score"}
PARAMS = ["param_disasters", "param_foodbox", "param_sciencebox"]
TRAITS = ["trait_Aggressive", "trait_Expansionist", "trait_Trader", "trait_Builder"]
TEMPLATES = ["leader", "wonder", "mildrop", "capture", "stillahead", "lostcity", "newcity",
             "moreenemies", "bankrupt", "govchange", "newally"]
TEMPLATE_LABEL = {"leader": "leads on score", "wonder": "completes wonder", "mildrop": "army −30%",
                  "capture": "captures city", "stillahead": "stays ahead", "lostcity": "loses city",
                  "newcity": "founds city", "moreenemies": "more enemies", "bankrupt": "treasury → 0",
                  "govchange": "changes govt", "newally": "new alliance"}

# (display name, score tag). Open models are scored on the full set; the closed models and the
# expert line use the same stratified 190-anchor common subset.
EXPERT_TAG = "expert-v33"
LINES = [   # (display name, score tag, Artificial Analysis Intelligence Index v4.3); weakest to strongest by the index,
            # then the expert line; the ceiling row closes the table
    ("gpt-oss-120b", "gpt-oss-120b", 12),
    ("Gemma~4 31B", "gemma4-31b", 15),
    ("Nemotron~3 Ultra", "nemotron3-ultra", 23),
    ("Inkling", "inkling", 26),
    ("Qwen3.8-27B", "q38-27b", 34),
    ("GPT-5.6~Sol", "gpt56sol", 39),
    ("DeepSeek-V4.1-Flash", "dsv41-flash", 40),
    ("Qwen3.8-Flash-Next", "tk", 40),
    ("Kimi~K3", "kimi-k3", 44),
    ("GLM-5.3", "glm-5.3", 45),
    ("Claude Opus~5", "opus5", 45),
    ("\\expert", EXPERT_TAG, None),
]
AA_INDEX = {t: i for _, t, i in LINES}
# categorical palette (fixed order, validated for CVD separation on light surfaces)
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
COLOR_BY_TAG = {"tk": "#2a78d6", "tk_api60": "#2a78d6", "opus5": "#eb6834",
                "gpt56sol": "#1baf7a", EXPERT_TAG: "#4a3aa7", "q38-27b": "#eda100",
                "gpt-oss-120b": "#666666", "kimi-k3": "#008300"}
TEXT_PRIMARY, TEXT_SECONDARY, GRID = "#0b0b0b", "#52514e", "#e6e5e1"


def color(tag, i):
    return COLOR_BY_TAG.get(tag, COLORS[i % len(COLORS)])


# raw-dir tag -> score-file tag differences (the Inkling line ran the NVFP4 checkpoint)
TAG_ALIAS = {"inkling": "inkling-nvfp4", EXPERT_TAG: "gpt56_v33_full60_all60"}
PANEL_SCORE_TAGS = {
    "gpt-oss-120b": "gpt-oss-120b_parsefail_v2",
    "gemma4-31b": "gemma4-31b_parsefail_v2",
    "nemotron3-ultra": "nemotron3-ultra_parsefail_v2",
    "inkling": "inkling-nvfp4_parsefail_v2",
    "q38-27b": "q38-27b_parsefail_v2",
    "gpt56sol": "gpt56sol_api190_parsefail_v2",
    "dsv41-flash": "dsv41-flash_parsefail_v2",
    "tk": "tk_parsefail_v2",
    "kimi-k3": "kimi-k3_parsefail_v2",
    "glm-5.3": "glm-5.3",
    "opus5": "opus5_api190_parsefail_v2",
    EXPERT_TAG: "gpt56_v33_api190_parsefail_v2",
}
COMMON60_SCORE_TAGS = {
    "tk_api60": "tk_api60", "opus5": "opus5", "gpt56sol": "gpt56sol",
    EXPERT_TAG: "gpt56_v33_full60_all60",
}
# lines whose numbers currently come from the frozen 120-anchor subset (`scores_raw_<tag>_sub120.json`);
# no marker in the table (user decision 2026-09-18: every subset line is replaced by its full-set file before submission)
SUBSET_TAGS = set()


def score_path(tag):
    if tag in PANEL_SCORE_TAGS:
        return RUN / f"scores_raw_{PANEL_SCORE_TAGS[tag]}.json"
    t = TAG_ALIAS.get(tag, tag)
    p = RUN / f"scores_raw_{t}.json"
    if p.exists():
        return p
    q = RUN / f"scores_raw_{t}_sub120.json"
    if q.exists():
        SUBSET_TAGS.add(tag)
        return q
    return None


def load(tag):
    path = score_path(tag)
    if path is None:
        return None
    scores = json.loads(path.read_text())
    if tag in PANEL_SCORE_TAGS and scores.get("unparsed_policy", {}).get("version") != "unparsed-as-wrong-v2":
        raise ValueError(f"Unexpected scoring policy: {path}")
    return scores


def load_common60():
    return {tag: json.loads((RUN / f"scores_raw_{source}.json").read_text())
            for tag, source in COMMON60_SCORE_TAGS.items()}


def fmt(x, nd=2, sign=True):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "\\pend"
    return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"


def triple(vals, nd=2, sign=True):
    if all(v is None for v in vals):
        return "\\pend"
    return " / ".join(fmt(v, nd, sign) for v in vals)


# ---------------------------------------------------------------- extraction
def m_1a(s, key):
    f = (s or {}).get("1a_family", {}).get(key)
    return f["macro_skill"]["mean"] if f and f.get("macro_skill") else None


def m_1a_fam(s, key, fam):
    f = (s or {}).get("1a_family", {}).get(key)
    return f["by_family_macro"].get(fam) if f else None


def m_1b(s, key):
    b = (s or {}).get("1b_headline_brier_gain", {}).get(key)
    return b["mean"] if b else None


def m_1b_template(s, key, tpl):
    """Pooled Brier skill ratio of one template in one setting: 1 - sum(brier*n)/sum(base*n)."""
    st, cond = key.split("|")
    num = den = 0.0
    for k, v in (s or {}).get("1b_cells", {}).items():
        p = k.split("|")
        if p[0] == st and p[1] == cond and p[3] == tpl and v.get("n"):
            num += v["brier"] * v["n"]; den += v["brier_base"] * v["n"]
    return 1 - num / den if den else None


def m_love(s, key):
    t = (s or {}).get("2", {}).get(f"{key}|love")
    return (t["acc"]["mean"], t.get("majority")) if t else (None, None)


def m_rules(s, key):
    accs, majs = [], []
    for p in PARAMS:
        t = (s or {}).get("2", {}).get(f"{key}|{p}")
        if t and t.get("acc"):
            accs.append(t["acc"]["mean"]); majs.append(t.get("majority") or 1 / 3)
    return (sum(accs) / len(accs), sum(majs) / len(majs)) if accs else (None, None)


def m_traits(s, key):
    accs = [(s or {}).get("2", {}).get(f"{key}|{t}", {}).get("acc", {}).get("mean") for t in TRAITS]
    accs = [a for a in accs if a is not None]
    return sum(accs) / len(accs) if accs else None


def m_l3(s, key):
    """Pooled over operators: (effect capture real, direction real, specificity).

    Effect capture (2026-09-21, user decision) is the effect-only score `A_effect_S` of score_v2,
    S = 1 - sum|s_hat - d| / sum|d| over the moved cells of an operator (0 = the forecast did not move,
    1 = every predicted shift equals the true one, <0 = moving was worse than not moving), then the
    UNWEIGHTED MEAN OVER OPERATORS that have at least MIN_OP_CELLS moved cells, each operator's S
    clipped to [-1, 1] first -- the sum-ratio pooled over everything let one (operator, quantity)
    carry half the number, and an unclipped overshoot on one operator (Gemma: -42 on L4-trait-expa)
    would do the same to the macro-average.  The old CRPS-ratio capture
    (`A_effect_ratio`) is still in the JSON as a diagnostic.
    Specificity restores 1 - mean(placebo shift) / mean(real shift), pooled over
    cells and clipped to [-1, 1]. It is undefined when the real mean is zero.
    """
    t = (s or {}).get("3", {})
    dnum = dden = 0.0
    for k, v in t.items():
        if not (k.startswith(key + "|") and k.endswith("|real")):
            continue
        w = v.get("A_n_moved") or 0
        d = v.get("A_dir_correct")
        if d is not None and w:
            dnum += d * w; dden += w
    cap = _l3_effect(s, key, "real")
    real_shift = _l3_shift(s, key, "real")
    placebo_shift = _l3_shift(s, key, "placebo")
    spec = (max(-1.0, min(1.0, 1.0 - placebo_shift / real_shift))
            if real_shift is not None and real_shift > 0 and placebo_shift is not None else None)
    coverage = (s or {}).get("3_response_coverage", {}).get(key)
    if coverage:
        if coverage["fraction"] == 0:
            spec = coverage["failure_score"]
        elif spec is not None:
            spec = coverage["fraction"] * spec + (1 - coverage["fraction"]) * coverage["failure_score"]
    return (cap, dnum / dden if dden else None, spec)


MIN_OP_CELLS = 3     # an operator enters the capture macro-average only with this many moved cells


def _l3_effect(s, key, arm):
    """Macro-average over operators of score_v2's `A_effect_S` (see m_l3)."""
    t = (s or {}).get("3", {}); vals = []
    for k, v in t.items():
        if not (k.startswith(key + "|") and k.endswith("|" + arm)):
            continue
        r = v.get("A_effect_S")
        if r and (r.get("n") or 0) >= MIN_OP_CELLS:
            vals.append(max(-1.0, min(1.0, r["mean"])))   # S is unbounded below (overshoot); -1 = a full reversal
    return sum(vals) / len(vals) if vals else None


def ceiling():
    c = json.load(open(CEIL))
    out = {}
    for key, _ in SETTINGS:
        a = c["1a"][key]["ideal_fair"]; b = c["1b"][key]["ideal_fair_loo"]
        out[key] = {"macro": a["macro"]["mean"], "fam": a["by_family_macro"], "1b": b["gain"]["mean"]}
    tc = json.load(open(TRAIT_CEIL))
    pair = [tc[t.split("_")[1]]["pair_ideal"] for t in TRAITS]
    out["traits"] = sum(pair) / len(pair); out["traits_range"] = (min(pair), max(pair))
    if TRAIT_CEIL_V3.exists():                      # per-set ideal observer on the v3 anchors (linear, all pairs)
        obs = json.load(open(TRAIT_CEIL_V3))["observers"]["linear"]
        allp = []
        for key, _ in SETTINGS:
            st = key.split("|")[0]
            vals = [obs[f"{t.split('_')[1]}|{st}|gap>=0"]["acc"] for t in TRAITS]
            out[key]["traits"] = sum(vals) / len(vals); allp += vals
        out["traits"] = sum(allp) / len(allp); out["traits_range"] = (min(allp), max(allp))
    else:
        for key, _ in SETTINGS:
            out[key]["traits"] = out["traits"]
    dc = json.load(open(DIR_CEIL)) if DIR_CEIL.exists() else {}
    for key, _ in SETTINGS:
        out[key]["dir"] = dc.get(key, {}).get("direction_ceiling", 1.0)
    ec = json.load(open(EFFECT_CEIL)) if EFFECT_CEIL.exists() else {}
    for key, _ in SETTINGS:
        out[key]["capture"] = (ec.get(key) or {}).get("effect_ceiling_macro", 1.0)
    if CEIL_OVERRIDE:                                # decision 2026-09-21, see the constant's comment
        for key, _ in SETTINGS:
            out[key]["traits"] = CEIL_OVERRIDE.get("traits", out[key]["traits"])
            out[key]["dir"] = CEIL_OVERRIDE.get("dir", out[key]["dir"])
        if "traits" in CEIL_OVERRIDE:
            out["traits"] = CEIL_OVERRIDE["traits"]
    rc = json.load(open(RULES_CEIL)) if RULES_CEIL.exists() else {}
    out["_rules_raw"] = rc
    cells = c["1b_cells"]
    out["1b_templates"] = {st: {tpl: cells[f"{st}|{tpl}"] for tpl in TEMPLATES if f"{st}|{tpl}" in cells}
                           for st in ("bench-basic", "bench-gen")}
    return out


def rules_ceiling(ceil, key):
    """Equal parameter weights, matching m_rules and independent of model coverage."""
    rc = ceil.get("_rules_raw") or {}
    dis = rc.get(key, {}).get("disasters_ceiling")
    if dis is None:
        return 1.0
    return (dis + rc.get("foodbox_ceiling", 1.0) + rc.get("sciencebox_ceiling", 1.0)) / 3


# ---------------------------------------------------------------- tables
def _cellfmt(x, nd=2, sign=True):
    if x is None:
        return "\\pend"
    t = f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"
    return t.replace("-", "$-$")


def adjusted_accuracy(value, floor, upper):
    """Map the accuracy reference to zero while preserving the ceiling value."""
    if value is None:
        return None
    if not 0 <= floor < upper:
        raise ValueError("Accuracy references must satisfy 0 <= floor < ceiling")
    return upper * (value - floor) / (upper - floor)


def level_scores(s, key, ceil):
    """Per-level summary: mean fraction of the way from the reference predictor to the attainable ceiling over the
    level's tasks (L1: Forecast, Event; L2: Trait, Rules; L3: Intervene direction).  None if nothing scored."""
    v = {ax: normalised_metric(s, key, ceil, ax) for ax in ("macro", "1b", "traits", "rules", "l3")}
    def mean(keys):
        xs = [v[k] for k in keys if v[k] is not None]
        return sum(xs) / len(xs) if xs else None
    return mean(("macro", "1b")), mean(("traits", "rules")), mean(("l3",))


def _pct(x):
    return "\pend" if x is None else f"{100*x:.0f}".replace("-", "$-$")


def table_main(scores, ceil, settings, label, caption, size="\\scriptsize"):
    """Rows = lines, with a separate AA index column and one shaded block per setting;
    columns grouped by level.  Within a block the best model value of each column is bold."""
    L = []
    L.append("\\begin{table}[t]\\centering" + size)
    L.append("\\setlength{\\tabcolsep}{1.5pt}\\renewcommand{\\arraystretch}{1.12}")
    L.append("\\caption{" + caption + "}")
    L.append("\\label{" + label + "}")
    L.append("\\begin{tabular}{@{}p{0.195\\linewidth}>{\\centering\\arraybackslash}p{0.045\\linewidth}*{6}{>{\\raggedleft\\arraybackslash}p{\\dimexpr(0.76\\linewidth-27pt)/7\\relax}}>{\\raggedleft\\arraybackslash}p{\\dimexpr(0.76\\linewidth-27pt)/7+3pt\\relax}@{}}")
    L.append("\\toprule")
    L.append(" & & \\multicolumn{2}{c}{\\Lone\\ association} & \\multicolumn{2}{c}{\\Ltwo\\ belief} & "
             "\\multicolumn{3}{c}{\\Lthree\\ intervention} \\\\")
    L.append("\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-9}")
    L.append("Line & \\shortstack{AA\\\\Index} & \\QNT$\\uparrow$ & \\EVT$\\uparrow$ & \\TRT$\\uparrow$ & \\RUL$\\uparrow$ & "
             "\\textsc{Capture}$\\uparrow$ & \\textsc{Direction}$\\uparrow$ & \\textsc{Specificity}$\\uparrow$ \\\\")
    s_floor = load("perfect")  # Majority labels from the complete truth set, independent of model parsing.
    pretty = {"omni-ID": "\\omniID", "ego": "\\ego", "OOD": "\\omniOOD"}
    for bi, (k, lab) in enumerate(settings):
        L.append("\\midrule" if bi == 0 else "\\addlinespace[2pt]\\midrule")
        L.append(f"\\rowcolor{{tableTeal!12}} \\multicolumn{{9}}{{@{{}}l}}{{\\textcolor{{tableTeal}}{{{pretty[lab]}}}}} \\\\")
        rule_floor = m_rules(s_floor, k)[1]
        floor = {"a": 0, "b": 0, "t": 0, "r": 0, "c": -1, "d": 0, "s": 0}
        upper = {"a": ceil[k]["macro"], "b": ceil[k]["1b"], "t": ceil[k]["traits"],
                 "r": rules_ceiling(ceil, k), "c": ceil[k]["capture"], "d": ceil[k]["dir"], "s": 1}
        L.append(f"\\rowcolor{{black!3}} \\floorline & \\textemdash & +0.00 & +0.000 & 0.00 & 0.00 & $-$1.00 & 0.00 & +0.00 \\\\")
        L.append("\\tabledash")
        rows, best = [], {}
        for name, tag, aa in LINES:
            sc = scores.get(tag)
            c, d, sp = m_l3(sc, k)
            v = {"a": m_1a(sc, k), "b": m_1b(sc, k),
                 "t": adjusted_accuracy(m_traits(sc, k), 0.5, upper['t']),
                 "r": adjusted_accuracy(m_rules(sc, k)[0], rule_floor, upper['r']),
                 "c": c, "d": adjusted_accuracy(d, 0.5, upper['d']), "s": sp}
            label_ = name + " & " + (r"\textemdash" if aa is None else str(aa))
            rows.append((label_, tag, v))
            if tag != EXPERT_TAG:
                for key, x in v.items():
                    if x is not None and (best.get(key) is None or x > best[key]):
                        best[key] = x
        for ri, (label_, tag, v) in enumerate(rows):
            def cell(key, nd=2, sign=True):
                x = v[key]
                t = _cellfmt(x, nd, sign)
                if x is None:
                    return t
                if tag != EXPERT_TAG and best.get(key) == x:
                    t = f"\\textbf{{{t}}}"
                pct = 100 * (x - floor[key]) / (upper[key] - floor[key])
                tint = "tableRose" if pct < 0 else "tableTeal"
                return t + f"\\headroompct{{{tint}}}{{{pct:.0f}}}"
            if tag == EXPERT_TAG:
                L.append("\\tabledash")
                L.append("\\rowcolor{tableTeal!7}")
            elif ri % 2 == 1:
                L.append("\\rowcolor{black!2}")
            if all(x is None for x in v.values()):
                L.append(f"{label_} & \\multicolumn{{7}}{{c}}{{\\pend}} \\\\")
                continue
            L.append(f"{label_} & {cell('a')} & {cell('b', 3)} & {cell('t', 2, False)} & {cell('r', 2, False)} & "
                     f"{cell('c')} & {cell('d', 2, False)} & {cell('s')} \\\\")
        L.append("\\tabledash")
        L.append(f"\\rowcolor{{tableTeal!4}} \\ceilingline & \\textemdash & {_cellfmt(ceil[k]['macro'])} & {_cellfmt(ceil[k]['1b'], 3)} & "
                 f"{_cellfmt(ceil[k]['traits'], 2, False)} & {_cellfmt(rules_ceiling(ceil, k), 2, False)} & "
                 f"{_cellfmt(ceil[k]['capture'])} & {_cellfmt(ceil[k]['dir'], 2, False)} & 1.00 \\\\")
    L.append("\\bottomrule\\end{tabular}")
    L.append("\\end{table}")
    return "\n".join(L) + "\n"


CAP_MAIN = ("Main results; all metrics are defined in \\cref{sec:protocol} and higher is better.  "
            "Missing or unparseable answers count as failures for every line. "
            "The AA Index column reports the Artificial Analysis Intelligence Index (v4.3), which orders the model rows.  "
            "Trait, rules, and direction use reference-adjusted scores (\\cref{eq:adjustedaccuracy}), not raw accuracies.  "
            "Each block closes with the attainable ceiling of that setting; the best model value of each column is bold, "
            "and a dash marks a run in progress. "
            "Small percentages show $(s-f)/(c-f)$ relative to the displayed floor $f$ and ceiling $c$, "
            "without clipping; teal denotes nonnegative headroom and rose denotes below-floor performance. "
            "\\floorline lists column-wise references, not a common baseline: "
            "trend extrapolation (QNT), the supplied event base rate (EVT), random pair ordering (TRT), "
            "the majority label per rule (RUL), the clipped lower endpoint $-1$ (capture), random nonzero signs "
            "(direction), and equal nonzero responses to real and placebo edits (specificity). "
            "Zero capture still means no improvement over predicting no effect.")


def table_raw_accuracies(scores):
    lines = [r"\begin{table}[t]\centering\small",
             r"\caption{Underlying accuracies before the main table's reference adjustment.",
             r"Triples are \omniID/\ego/\omniOOD. Raw references are 0.50 for trait and",
             r"direction, and the equal-parameter majority accuracy for rules.}",
             r"\label{tab:rawaccuracies}",
             r"\begin{tabular}{@{}lccc@{}}\toprule",
             r"Line & Trait accuracy & Rule accuracy & Direction accuracy \\ \midrule"]
    perfect = load("perfect")
    references = triple([m_rules(perfect, key)[1] for key, _ in SETTINGS], sign=False)
    lines.append(r"Raw reference & 0.50 / 0.50 / 0.50 & " + references + r" & 0.50 / 0.50 / 0.50 \\ \midrule")
    for name, tag, _ in LINES:
        scores_by_setting = [[m_traits(scores.get(tag), key), m_rules(scores.get(tag), key)[0],
                              m_l3(scores.get(tag), key)[1]] for key, _ in SETTINGS]
        cells = [triple([values[index] for values in scores_by_setting], sign=False)
                 for index in range(3)]
        lines.append(name + " & " + " & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule\end{tabular}", r"\end{table}"])
    return "\n".join(lines) + "\n"


def table_l3(scores):
    """Appendix: the intervention level in full -- effect capture on the real arm and on its matched
    placebo, direction agreement, and the mean absolute shift of the forecast on each arm."""
    L = []
    L.append("\\begin{table}[t]\\centering\\scriptsize")
    L.append("\\setlength{\\tabcolsep}{5pt}")
    L.append("\\caption{Intervention diagnostics across three settings. Capture and shift are reported for real arms "
             "with matched placebos in parentheses. Shift is the mean absolute change from the model's "
             "no-intervention forecast in the normalized units of \\cref{sub:scores}. Specificity applies "
             "the missing-response penalty to the clipped shift ratio defined in \\cref{eq:spec,app:metrics}. Triples are "
             "\\omniID\\ / \\ego\\ / \\omniOOD.}")
    L.append("\\label{tab:l3}")
    L.append("\\begin{tabular}{@{}lccc@{}}")
    L.append("\\toprule")
    L.append("Line & capture real (placebo) & direction & shift real (placebo) \\\\")
    L.append("\\midrule")
    for name, tag, aa in LINES:
        sc = scores.get(tag)
        caps, dirs, shifts = [], [], []
        for k, _ in SETTINGS:
            c, d, _sp = m_l3(sc, k)
            p = _l3_capture_placebo(sc, k)
            sr, sp2 = _l3_shift(sc, k, "real"), _l3_shift(sc, k, "placebo")
            caps.append("\\pend" if c is None else f"{_cellfmt(c)} ({_cellfmt(p)})")
            dirs.append(_cellfmt(d, 2, False))
            shifts.append("\\pend" if sr is None else f"{_cellfmt(sr, 2, False)} ({_cellfmt(sp2, 2, False)})")
        L.append(f"{name} & {' / '.join(caps)} & {' / '.join(dirs)} & {' / '.join(shifts)} \\\\")
    L.append("\\bottomrule\\end{tabular}")
    L.append("\\end{table}")
    return "\n".join(L) + "\n"


def _l3_capture_placebo(s, key):
    return _l3_effect(s, key, "placebo")


def _l3_shift(s, key, arm):
    t = (s or {}).get("3", {}); sn = sd = 0.0
    for k, v in t.items():
        if not (k.startswith(key + "|") and k.endswith("|" + arm)):
            continue
        for f, nf in (("B_shift_moved", "B_n_moved"), ("B_clean_shift_unmoved", "B_n_unmoved")):
            x = v.get(f); w = v.get(nf) or 0
            if x is not None and w:
                sn += x * w; sd += w
    return sn / sd if sd else None




def table_families(scores, ceil):
    tables = []
    for index, (setting, label) in enumerate(SETTINGS):
        lines = [r"\begin{table}[t]\centering\scriptsize",
                 r"\setlength{\tabcolsep}{4pt}",
                 r"\caption{Quantity-family skill against the trend in "+label+
                 r". Variables are equal-weighted within each family for this diagnostic; these columns need not average to the headline score. The reference row uses re-seeded reruns.}",
                 r"\label{tab:fam"+("" if index == 0 else str(index))+"}",
                 r"\begin{tabular}{@{}lcccccc@{}}\toprule",
                 "Line & "+" & ".join(FAM_LABEL[family] for family in FAMS)+r" \\ \midrule"]
        for name, tag, _ in LINES:
            values = [fmt(m_1a_fam(scores.get(tag), setting, family)) for family in FAMS]
            lines.append(name+" & "+" & ".join(values)+r" \\")
        lines.extend([r"\midrule", r"\ceilingline & "+" & ".join(fmt(ceil[setting]["fam"][family]) for family in FAMS)+r" \\",
                      r"\bottomrule\end{tabular}\end{table}"])
        tables.append("\n".join(lines))
    return "\n".join(tables)+"\n"


# ---------------------------------------------------------------- figures
def _mpl(academic=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    style = {"font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
                         "mathtext.fontset": "stix", "font.size": 9, "axes.titlesize": 9,
                         "axes.titleweight": "normal", "axes.labelsize": 9, "axes.linewidth": .6,
                         "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
                         "xtick.major.width": .6, "ytick.major.width": .6,
                         "xtick.direction": "out", "ytick.direction": "out",
                         "lines.linewidth": 1, "grid.linewidth": .5, "grid.alpha": .5,
                         "hatch.linewidth": .45,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "figure.facecolor": "white",
                         "axes.facecolor": "white", "text.color": "black", "axes.labelcolor": "black"}
    plt.rcParams.update(style if academic else {key: matplotlib.rcParamsDefault[key] for key in style})
    if not academic:
        plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "legend.fontsize": 7.5, "pdf.fonttype": 42})
    return plt


def _save(fig, out):
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=110, bbox_inches="tight")   # preview only


AXES = [("economy", "economy"), ("scale", "scale"), ("tech_culture", "tech–culture"), ("military", "military"),
        ("diplomacy", "diplomacy"), ("overall", "overall\nscore"), ("1b", "events"),
        ("rules", "rules"), ("traits", "traits"), ("l3", "intervention\ndirection")]


def normalised(s, key, ceil):
    """Every axis as the fraction of the way from the reference predictor (0) to the attainable ceiling (1)."""
    v = []
    for ax, _ in AXES:
        if ax in FAMS:
            x = m_1a_fam(s, key, ax); v.append(None if x is None else x / ceil[key]["fam"][ax])
        elif ax == "1b":
            x = m_1b(s, key); v.append(None if x is None else x / ceil[key]["1b"])
        elif ax == "love":
            x, maj = m_love(s, key); v.append(None if x is None else (x - maj) / (1 - maj))
        elif ax == "rules":
            x, maj = m_rules(s, key); c = rules_ceiling(ceil, key)
            v.append(None if x is None else (x - maj) / max(c - maj, 1e-9))
        elif ax == "traits":
            x = m_traits(s, key); v.append(None if x is None else (x - 0.5) / (ceil[key]["traits"] - 0.5))
        elif ax == "l3":
            _, d, _ = m_l3(s, key); v.append(None if d is None else (d - 0.5) / (ceil[key]["dir"] - 0.5))
    return v


def radar(lines, scores, ceil, out, title_note=""):
    import numpy as np
    plt = _mpl()
    lines = [(n, t) for n, t in lines if scores.get(t)]
    if not lines:
        return False
    n = len(AXES); ang = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist(); ang += ang[:1]
    fig, axs = plt.subplots(1, 3, figsize=(12.6, 4.3), subplot_kw=dict(polar=True))
    for ax, (key, lab) in zip(axs, SETTINGS):
        ax.plot(ang, [1] * (n + 1), color="black", lw=1.2, ls="--", label="attainable ceiling (100%)" if ax is axs[0] else None)
        for i, (name, tag) in enumerate(lines):
            v = normalised(scores[tag], key, ceil)
            if all(x is None for x in v):
                continue
            vv = [max(0.0, min(1.0, x)) if x is not None else 0.0 for x in v]; vv += vv[:1]
            ax.plot(ang, vv, color=color(tag, i), lw=1.6, marker="o", ms=3, label=name if ax is axs[0] else None)
            ax.fill(ang, vv, color=color(tag, i), alpha=0.06)
        ax.set_xticks(ang[:-1]); ax.set_xticklabels([l for _, l in AXES], fontsize=7.5)
        ax.set_ylim(0, 1.08); ax.set_yticks([0.25, 0.5, 0.75, 1.0]); ax.set_yticklabels(["", "50%", "", "100%"], fontsize=7)
        ax.set_title(lab, pad=12)
    fig.legend(loc="lower center", ncol=min(4, len(lines) + 1), frameon=False, bbox_to_anchor=(0.5, -0.10))
    if title_note:
        fig.suptitle(title_note, fontsize=8, y=0.995, color="gray")
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    _save(fig, out); plt.close(fig)
    return True


def ladder_map(lines, scores, ceil, out, title_note=""):
    """Rows = lines (weakest to strongest), columns = task axes, one panel per setting; cell = fraction of
    the way from the reference predictor (0) to the attainable ceiling (1), clipped to [0, 1]; the value is printed."""
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap
    plt = _mpl()
    lines = [(n, t) for n, t in lines if scores.get(t)]
    if not lines:
        return False
    cmap = LinearSegmentedColormap.from_list("seq_blue", ["#f4f7fc", "#c9daf3", "#8fb3e6", "#4f88d4", "#1f5fb4", "#123f7d"])
    fig, axs = plt.subplots(1, 3, figsize=(12.6, 0.36 * len(lines) + 1.4), sharey=True)
    for ax, (key, lab) in zip(axs, SETTINGS):
        M = np.full((len(lines), len(AXES)), np.nan)
        for i, (name, tag) in enumerate(lines):
            v = normalised(scores[tag], key, ceil)
            for j, x in enumerate(v):
                if x is not None:
                    M[i, j] = x
        shown = np.clip(np.nan_to_num(M, nan=0.0), 0, 1)
        ax.imshow(shown, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        for i in range(len(lines)):
            for j in range(len(AXES)):
                x = M[i, j]
                if np.isnan(x):
                    ax.text(j, i, "\u00b7", ha="center", va="center", color=TEXT_SECONDARY, fontsize=7)
                    continue
                txt = ("100+" if x > 1.005 else f"{100*x:.0f}") if x >= 0 else "<0"
                ax.text(j, i, txt, ha="center", va="center", fontsize=6.8,
                        color="white" if shown[i, j] > 0.55 else TEXT_PRIMARY)
        ax.set_xticks(range(len(AXES))); ax.set_xticklabels([l.replace("\n", " ") for _, l in AXES], rotation=35, ha="right", fontsize=7.5)
        ax.set_yticks(range(len(lines))); ax.set_yticklabels([n for n, _ in lines], fontsize=7.5)
        ax.set_title({"omni-ID": "Omni-ID", "ego": "Ego", "OOD": "Omni-OOD"}.get(lab, lab), fontsize=9, pad=6, color=TEXT_PRIMARY)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(length=0)
        # white gaps between cells
        for k in range(1, len(AXES)):
            ax.axvline(k - 0.5, color="white", lw=1.2)
        for k in range(1, len(lines)):
            ax.axhline(k - 0.5, color="white", lw=1.2)
    if title_note:
        fig.suptitle(title_note, fontsize=8, y=1.0, color=TEXT_SECONDARY)
    fig.tight_layout(w_pad=1.0)
    _save(fig, out); plt.close(fig)
    return True


METRICS = [("macro", "Forecast"), ("1b", "Event"), ("traits", "Trait"), ("rules", "Rules"), ("l3", "Intervene\ndir.")]


def normalised_metric(s, key, ceil, ax):
    if ax == "macro":
        x = m_1a(s, key); return None if x is None else x / ceil[key]["macro"]
    return normalised(s, key, ceil)[[a for a, _ in AXES].index(ax)]


def ladder_map_compact(lines, scores, ceil, out):
    """One panel: rows = lines, columns = 5 metrics x 3 settings; cell = fraction of the way from reference to ceiling."""
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap
    plt = _mpl()
    lines = [(n, t) for n, t in lines if scores.get(t)]
    if not lines:
        return False
    cmap = LinearSegmentedColormap.from_list("seq_blue", ["#f4f7fc", "#c9daf3", "#8fb3e6", "#4f88d4", "#1f5fb4", "#123f7d"])
    cols = [(k, lab, m, ml) for k, lab in SETTINGS for m, ml in METRICS]
    M = np.full((len(lines), len(cols)), np.nan)
    for i, (name, tag) in enumerate(lines):
        for j, (k, _, m, _) in enumerate(cols):
            x = normalised_metric(scores[tag], k, ceil, m)
            if x is not None:
                M[i, j] = x
    fig, ax = plt.subplots(figsize=(7.2, 0.30 * len(lines) + 1.15))
    shown = np.clip(np.nan_to_num(M, nan=0.0), 0, 1)
    ax.imshow(shown, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for i in range(len(lines)):
        for j in range(len(cols)):
            x = M[i, j]
            if np.isnan(x):
                ax.text(j, i, "\u00b7", ha="center", va="center", color=TEXT_SECONDARY, fontsize=7); continue
            txt = ("100+" if x > 1.005 else f"{100*x:.0f}") if x >= 0 else "<0"
            ax.text(j, i, txt, ha="center", va="center", fontsize=6.8, color="white" if shown[i, j] > 0.55 else TEXT_PRIMARY)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels([ml for _, _, _, ml in cols], fontsize=6.8)
    ax.set_yticks(range(len(lines))); ax.set_yticklabels([n for n, _ in lines], fontsize=7.5)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    for k in range(1, len(cols)):
        ax.axvline(k - 0.5, color="white", lw=2.6 if k % len(METRICS) == 0 else 1.0)
    for k in range(1, len(lines)):
        ax.axhline(k - 0.5, color="white", lw=1.0)
    for g, (_, lab) in enumerate(SETTINGS):
        ax.text(g * len(METRICS) + (len(METRICS) - 1) / 2, -0.95, {"omni-ID": "Omni-ID", "ego": "Ego", "OOD": "Omni-OOD"}[lab],
                ha="center", va="bottom", fontsize=8.5, color=TEXT_PRIMARY, fontweight="bold")
    ax.set_ylim(len(lines) - 0.5, -1.4)
    fig.tight_layout()
    _save(fig, out); plt.close(fig)
    return True


def fig_1b_templates(lines, scores, ceil, out):
    """Per-template EVT skill for at most four lines (main model, the two closed models, the expert) plus the ceiling."""
    import numpy as np
    plt = _mpl()
    keep = ("tk", "gpt56sol", "opus5", EXPERT_TAG)
    lines = [(n, t) for n, t in lines if t in keep and scores.get(t)]
    fig, axs = plt.subplots(1, 2, figsize=(12.0, 3.1), sharey=True)
    for ax, (key, st, lab) in zip(axs, [("bench-basic|v1", "bench-basic", "Omni-ID"), ("bench-gen|v1", "bench-gen", "Omni-OOD")]):
        x = np.arange(len(TEMPLATES)); w = 0.78 / max(1, len(lines))
        for i, (name, tag) in enumerate(lines):
            vals = [m_1b_template(scores[tag], key, tpl) for tpl in TEMPLATES]
            ax.bar(x + (i - (len(lines) - 1) / 2) * w, [v if v is not None else 0 for v in vals], w * 0.92,
                   color=color(tag, i), label=name if ax is axs[0] else None, linewidth=0)
        cl = [ceil["1b_templates"][st].get(tpl, {}).get("ceiling_loo") for tpl in TEMPLATES]
        ax.plot(x, [c if c is not None else np.nan for c in cl], "_", color=TEXT_PRIMARY, ms=16, mew=1.6,
                label="ceiling (leave-one-out reruns)" if ax is axs[0] else None)
        ax.axhline(0, color=TEXT_SECONDARY, lw=0.8)
        ax.set_xticks(x); ax.set_xticklabels([TEMPLATE_LABEL[t] for t in TEMPLATES], rotation=35, ha="right", fontsize=7.5)
        ax.set_title(lab, fontsize=9, color=TEXT_PRIMARY); ax.set_ylim(-0.45, 1.0)
        ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.spines["left"].set_color(GRID); ax.spines["bottom"].set_color(GRID)
    axs[0].set_ylabel("Brier skill vs shown base rate", color=TEXT_SECONDARY)
    fig.legend(loc="upper center", ncol=len(lines) + 1, frameon=False, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout()
    _save(fig, out); plt.close(fig)


def fig_cot(out):
    from paper.build_cot_results import collect, plot
    return plot(collect(), out)


def fig_cot_v31(out):
    import numpy as np
    if not COT.exists():
        return False
    c = json.loads(COT.read_text())
    plt = _mpl(academic=True)
    from matplotlib.ticker import PercentFormatter

    teal, rust, pale_teal, gray = "#367E83", "#B37553", "#A6CDD0", "#D9D8D3"
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.25))
    for axis, position in zip(axs, ([.07, .33, .52, .53], [.75, .33, .24, .53])):
        axis.set_position(position)
        axis.tick_params(labelsize=7.5, length=2, pad=3)
    ax = axs[0]
    order = [("table", "1a"), ("code", "1a"), ("table", "3-1a"), ("code", "3-1a"), ("table", "1b"), ("code", "1b")]
    keys = ["answer_right/evidence_right", "answer_right/evidence_wrong", "answer_wrong/evidence_right", "answer_wrong/evidence_wrong"]
    labs = ["Answer+ / Evidence+", "Answer+ / Evidence-", "Answer- / Evidence+", "Answer- / Evidence-"]
    cols = [teal, rust, pale_teal, gray]
    xs = np.arange(len(order)); bottom = np.zeros(len(order))
    for k, lab, col in zip(keys, labs, cols):
        vals = np.array([c["quadrants"].get(f"{f}|{r}", {}).get(k, 0.0) for f, r in order])
        ax.bar(xs, vals, 0.7, bottom=bottom, color=col, label=lab, linewidth=0); bottom += vals
    ax.set_xticks(xs, [form.title() for form, task in order], fontsize=7)
    for position, label in ((.5, "Quantity"), (2.5, "INT (quantity)"), (4.5, "Events")):
        ax.text(position, -.16, label, transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=7)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, .5, 1])
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.set_ylabel("Eligible units", fontsize=8, labelpad=3)
    ax.set_title("(a) Joint outcomes", fontsize=9, pad=6)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -.32), ncol=2, frameon=False,
              fontsize=7, handlelength=1.1, handletextpad=.4, columnspacing=.8, borderaxespad=0)
    ax = axs[1]
    fmts = ["table", "code"]
    shared_settings = ("bench-basic|v1", "bench-gen|v1")
    ans, evd = [], []
    for form in fmts:
        answer_scores = [c["answer"].get(form, {}).get(setting, {}).get("1a_macro")
                         for setting in shared_settings]
        evidence_scores = [c["evidence"].get(f"{setting}|{form}|1a", {}).get("V_intermediate", {}).get("macro_skill_trend")
                           for setting in shared_settings]
        ans.append(float(np.mean(answer_scores)) if all(v is not None for v in answer_scores) else None)
        evd.append(float(np.mean(evidence_scores)) if all(v is not None for v in evidence_scores) else None)
    x = np.arange(len(fmts))
    ax.bar(x - 0.16, [value if value is not None else np.nan for value in ans], 0.32,
           color=teal, label="Answer", linewidth=0)
    ax.bar(x + 0.16, [value if value is not None else np.nan for value in evd], 0.32,
           color=rust, label="Evidence", linewidth=0)
    ax.axhline(0, color=TEXT_SECONDARY, lw=0.8)
    ax.set_xlim(-0.6, len(fmts) - 0.4)
    ax.set_xticks(x, [form.title() for form in fmts])
    ax.set_ylim(-.22, .22)
    ax.set_yticks([-.2, 0, .2])
    ax.set_ylabel("Skill vs trend", fontsize=8, labelpad=3)
    ax.set_title("(b) Answer and evidence skill", fontsize=9, pad=6)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -.32), ncol=1, frameon=False,
              fontsize=7, handlelength=1.1, handletextpad=.4, borderaxespad=0)
    for axis in axs:
        for sp in ("top", "right"):
            axis.spines[sp].set_visible(False)
        axis.spines["left"].set_linewidth(.5)
        axis.spines["bottom"].set_linewidth(.5)
        axis.yaxis.grid(True, color=GRID, lw=.5)
        axis.set_axisbelow(True)
    _save(fig, out); plt.close(fig)
    return True


def main():
    OUT_T.mkdir(parents=True, exist_ok=True)
    tags = {t for _, t, _ in LINES}
    scores = {t: load(t) for t in tags}
    print("score files present:", sorted(t for t, s in scores.items() if s))
    spec_values = [m_l3(scores[t], k)[2] for _, t, _ in LINES if t != EXPERT_TAG and scores[t]
                   for k, _ in SETTINGS]
    spec_values = [v for v in spec_values if v is not None]
    spec_macros = {"specModelsRange": f"{fmt(min(spec_values))} to {fmt(max(spec_values))}",
                   "specExpert": triple([m_l3(scores.get(EXPERT_TAG), k)[2] for k, _ in SETTINGS])}
    for tag, prefix in (("gpt56sol", "gpt"), ("opus5", "opus"), (EXPERT_TAG, "hum")):
        spec_macros[prefix + "OneA"] = triple([m_1a(scores[tag], key) for key, _ in SETTINGS])
        spec_macros[prefix + "OneB"] = triple([m_1b(scores[tag], key) for key, _ in SETTINGS], nd=3)
    numbers = ROOT / "paper/numbers.tex"
    text = numbers.read_text()
    for name, value in spec_macros.items():
        pattern = r"\\newcommand\{\\" + name + r"\}\{[^\n]*\}"
        text, n = re.subn(pattern, lambda _: "\\newcommand{\\" + name + "}{" + value + "}", text)
        if n != 1:
            raise ValueError(f"Expected one numbers.tex macro: {name}, got {n}")
    numbers.write_text(text)
    ceil = ceiling()
    (OUT_T / "main_all.tex").write_text(table_main(scores, ceil, SETTINGS, "tab:main", CAP_MAIN))
    (OUT_T / "raw_accuracies.tex").write_text(table_raw_accuracies(scores))
    (OUT_T / "l3_capture.tex").write_text(table_l3(scores))
    (OUT_T / "families.tex").write_text(table_families(scores, ceil))
    allx = [(n.replace("~", " ").replace("\\expert", "Expert"), t) for n, t, _ in LINES]
    ladder_map_compact(allx, scores, ceil, OUT_F / "ladder_map.pdf")
    ladder_map(allx, scores, ceil, OUT_F / "ladder_map_families.pdf")
    radar_tags = {"gpt-oss-120b", "tk", "kimi-k3", "opus5", EXPERT_TAG}
    radar([(name, tag) for name, tag in allx if tag in radar_tags], scores, ceil, OUT_F / "radar.pdf")
    fig_1b_templates(allx, scores, ceil, OUT_F / "onebee_templates.pdf")
    ok = fig_cot(OUT_F / "cot.pdf")
    print("wrote tables/main_all l3_capture families; figures/radar onebee_templates", "cot" if ok else "(no cot)")
    print(f"trait ceiling (ideal observer, pairwise): {ceil['traits']:.3f} range {ceil['traits_range']}")
    for k, lab in SETTINGS:
        print(f"  {lab:<8} ceilings: traits {ceil[k]['traits']:.3f}  rules {rules_ceiling(ceil, k):.3f}  direction {ceil[k]['dir']:.3f}  capture {ceil[k]['capture']:.3f}")


if __name__ == "__main__":
    main()
