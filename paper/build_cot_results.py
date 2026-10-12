"""Render Section 6 from completed aligned runs, without inference or rescoring."""
import hashlib
import gzip
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean

from bench.cot.analyze_aligned import spearman
from bench import score_v2 as SV
from paper.build_results_analysis import table
from paper import gen_tables as gt

MODELS = [("flash-next", "Flash-Next", "Flash", 190),
          ("q38-27b", "Qwen3.8-27B", "27B", 190),
          ("gemma4-31b", "Gemma 4 31B", "Gemma", 190),
          ("kimi-k3", "Kimi K3", "Kimi", 190),
          ("gpt56sol", "GPT-5.6 Sol", "Sol", 30)]
FORMATS = ("free", "table", "code")
SETTINGS = ("bench-basic|v1", "bench-gen|v1")
QUADRANTS = ("A+E+", "A+E-", "A-E+", "A-E-")
OUTPUT = gt.ROOT / "paper"
REFERENCE_SPEC = gt.ROOT / "bench/measure/out/section5/swfi_v1/spec.json"
MAIN_TABLE_RESULTS = gt.ROOT / "bench/measure/out/section5/main_table_compact.json"


def answer_process_rates(state_units, free_answers):
    def unit_key(unit):
        return tuple(unit.get(key) for key in
                     ("world", "t1", "set", "cond", "task", "sub", "arm_kind", "op_base", "civ", "var", "h"))

    paired = defaultdict(dict)
    for unit in state_units:
        if unit["cond"] == "v1" and (unit["task"] == "1a" or unit["arm_kind"] == "real"):
            paired[unit_key(unit)][unit["fmt"]] = unit
    free = {}
    for answer in free_answers:
        if answer["cond"] != "v1" or not (answer["kind"] == "1a" or
                (answer["kind"] == "3" and answer.get("sub") == "1a" and answer.get("arm_kind") == "real")):
            continue
        if any(answer.get(key) is None for key in ("p50", "y", "norm")):
            continue
        error = abs(SV.tf(answer["var"], answer["p50"]) - SV.tf(answer["var"], answer["y"])) / answer["norm"]
        if math.isfinite(error):
            key = unit_key(dict(answer, task=answer["kind"], op_base=answer.get("op")))
            if key in free:
                raise ValueError(f"Duplicate Free answer unit: {key}")
            free[key] = error
    groups = defaultdict(list)
    for key, formats in paired.items():
        if key not in free or not all(fmt in formats for fmt in ("table", "code")):
            continue
        if any(unit.get(field) is None or not math.isfinite(unit[field])
               for unit in formats.values() for field in
               ("answer_error", "process_error", "answer_reference_error", "process_reference_error")):
            continue
        table_unit = formats["table"]
        if any(not math.isclose(table_unit[field], formats["code"][field])
               for field in ("answer_reference_error", "process_reference_error")):
            raise ValueError(f"Mismatched Table/Code references: {key}")
        outcomes = {"free": (free[key] < table_unit["answer_reference_error"], None)}
        for fmt in ("table", "code"):
            unit = formats[fmt]
            positive = unit["answer_error"] < unit["answer_reference_error"]
            outcomes[fmt] = (positive, positive and unit["process_error"] >= unit["process_reference_error"])
        scope = "L1" if table_unit["task"] == "1a" else "L3-state"
        groups[(scope, table_unit["set"])].append(outcomes)
    by_scope = {}
    for scope in ("L1", "L3-state"):
        by_scope[scope] = {}
        for setting in SETTINGS:
            units = groups[(scope, setting.split("|")[0])]
            if not units:
                raise ValueError(f"No common Free/Table/Code units for {scope}, {setting}")
            by_scope[scope][setting] = dict(n=len(units), formats={
                fmt: dict(answer_positive=mean(unit[fmt][0] for unit in units),
                          answer_positive_process_negative=None if fmt == "free" else
                          mean(unit[fmt][1] for unit in units)) for fmt in FORMATS})
    aggregate = {fmt: {
        metric: None if fmt == "free" and metric == "answer_positive_process_negative" else
        mean(by_scope[scope][setting]["formats"][fmt][metric]
             for scope in by_scope for setting in SETTINGS)
        for metric in ("answer_positive", "answer_positive_process_negative")} for fmt in FORMATS}
    return dict(aggregate=aggregate, by_scope=by_scope,
                definition="Equal L1/L3-state and ID/OOD means on common valid Free/Table/Code units. "
                "Positive means strictly lower point error than trend, not exact accuracy or SWFI completion. "
                "Free process quality is unassessed.")


def capture_completion(values, ranges):
    return mean(100 * (values[setting]["capture"] - ranges[setting]["capture"][0]) /
                (ranges[setting]["capture"][1] - ranges[setting]["capture"][0])
                for setting in SETTINGS)


def process_capture_association(records, nboot=20000):
    import numpy as np

    points = [dict(tag=row["tag"], name=row["name"], fmt=fmt, anchors=row["anchors"],
                   process=row["analysis"]["paired_point_forecasts"]["L3-state"][fmt]["aggregate"]["process_skill"],
                   capture=row["answers"][fmt]["aggregate"]["capture"],
                   capture_completion=row["answers"][fmt]["aggregate"]["capture_completion"],
                   l1_process=row["analysis"]["paired_point_forecasts"]["L1"][fmt]["aggregate"]["process_skill"],
                   l1_completion=row["answers"][fmt]["aggregate"]["l1_completion"])
              for row in records for fmt in ("table", "code")]

    def correlations(sample, process_key="process", answer_key="capture"):
        process = [point[process_key] for point in sample]
        capture = [point[answer_key] for point in sample]
        pearson = float(np.corrcoef(process, capture)[0, 1]) if np.ptp(process) > 0 and np.ptp(capture) > 0 else None
        return dict(n=len(sample), pearson=pearson, spearman=spearman(process, capture))

    model_means = [dict(tag=row["tag"], name=row["name"], short=row["short"], **{
        key: mean(point[key] for point in points if point["tag"] == row["tag"])
        for key in ("process", "capture", "capture_completion", "l1_process", "l1_completion")}) for row in records]
    result = dict(points=points, pooled=correlations(points),
                  by_format={fmt: correlations([point for point in points if point["fmt"] == fmt])
                             for fmt in ("table", "code")},
                  model_means=correlations(model_means),
                  model_mean_points=model_means,
                  model_means_completion=correlations(model_means, answer_key="capture_completion"),
                  l1_model_means=correlations(model_means, "l1_process", "l1_completion"),
                  definition="L3 branch-state process skill versus raw Capture; equal ID/OOD means; Table and Code only.")
    values = np.array([[point["process"], point["capture"]] for point in points]).reshape(len(records), 2, 2)
    indices = np.random.default_rng(20260924).integers(0, len(records), size=(nboot, len(records)))
    samples = values[indices].reshape(nboot, 2*len(records), 2)
    centered = samples - samples.mean(axis=1, keepdims=True)
    numerator = (centered[:, :, 0]*centered[:, :, 1]).sum(axis=1)
    denominator = np.sqrt((centered[:, :, 0]**2).sum(axis=1)*(centered[:, :, 1]**2).sum(axis=1))
    valid = denominator > 0
    estimates = numerator[valid] / denominator[valid]
    result["pooled"]["pearson_ci95"] = np.quantile(estimates, [.025, .975]).tolist() if len(estimates) else None
    result["bootstrap"] = dict(unit="model, retaining both formats", draws=nboot, seed=20260924,
                               valid_draws=int(valid.sum()))
    mean_values = np.array([[point["process"], point["capture_completion"]] for point in model_means])
    mean_samples = mean_values[indices]
    centered = mean_samples - mean_samples.mean(axis=1, keepdims=True)
    numerator = (centered[:, :, 0]*centered[:, :, 1]).sum(axis=1)
    denominator = np.sqrt((centered[:, :, 0]**2).sum(axis=1)*(centered[:, :, 1]**2).sum(axis=1))
    valid = denominator > 0
    estimates = numerator[valid] / denominator[valid]
    result["model_means_completion"]["pearson_ci95"] = np.quantile(estimates, [.025, .975]).tolist() if len(estimates) else None
    result["model_means_completion"]["valid_bootstrap_draws"] = int(valid.sum())
    return result


def collect():
    records, provenance = [], {}

    def read(path):
        data = path.read_bytes()
        provenance[str(path.relative_to(gt.ROOT))] = hashlib.sha256(data).hexdigest()
        return json.loads(data)

    def read_rows(path):
        data = path.read_bytes()
        provenance[str(path.relative_to(gt.ROOT))] = hashlib.sha256(data).hexdigest()
        return [json.loads(line) for line in gzip.decompress(data).splitlines() if line.strip()]

    ranges = read(REFERENCE_SPEC)["ranges"]
    swfi = {row["tag"]: row["overall"] for row in read(MAIN_TABLE_RESULTS)["records"]}
    for tag, name, short, anchors in MODELS:
        run = gt.ROOT / "bench/runs" / ("cot-v32-sol30" if tag == "gpt56sol" else "cot-v32-190")
        assert len(list((run / f"raw_{tag}").glob("*.jsonl.gz"))) == anchors
        analysis = read(run / f"aligned_analysis_raw_{tag}.json")
        assert all(analysis["cost"][f]["errors"] == 0 for f in FORMATS)
        answers = {}
        for fmt in FORMATS:
            scores = read(run / f"scores_raw_{tag}.ans.{fmt}.json")
            answers[fmt] = {s: dict(quantity=gt.m_1a(scores, s), event=gt.m_1b(scores, s),
                                   capture=gt.m_l3(scores, s)[0]) for s in SETTINGS}
            answers[fmt]["aggregate"] = {m: mean(answers[fmt][s][m] for s in SETTINGS)
                                         for m in ("quantity", "event", "capture")}
            answers[fmt]["aggregate"]["capture_completion"] = capture_completion(answers[fmt], ranges)
            answers[fmt]["aggregate"]["l1_completion"] = mean(
                mean(100 * (answers[fmt][setting][metric] - ranges[setting][metric][0]) /
                     (ranges[setting][metric][1] - ranges[setting][metric][0])
                     for metric in ("quantity", "event")) for setting in SETTINGS)
        for formats in analysis["paired_point_forecasts"].values():
            for result in formats.values():
                result["aggregate"]["quadrants"] = {
                    q: mean(result["settings"][s.split("|")[0]]["quadrants"][q] for s in SETTINGS)
                    for q in QUADRANTS}
                assert abs(sum(result["aggregate"]["quadrants"].values()) - 1) < 1e-10
        model_swfi = swfi["tk" if tag == "flash-next" else tag]
        if model_swfi is None or not math.isfinite(model_swfi):
            raise ValueError(f"Missing main-table SWFI for {tag}")
        records.append(dict(tag=tag, name=name, short=short, anchors=anchors, swfi=model_swfi,
                            answers=answers, analysis=analysis,
                            answer_process=answer_process_rates(
                                read_rows(run / f"aligned_state_units_raw_{tag}.jsonl.gz"),
                                read_rows(run / f"scored_raw_{tag}.ans.free.jsonl.gz"))))
    records.sort(key=lambda row: (row["swfi"], row["tag"]))
    return dict(records=records, provenance=provenance, reference_ranges=ranges,
                process_capture=process_capture_association(records),
                aggregation="State skill, quadrants and Capture: equal Omni-ID/OOD means; own full sample per model.",
                protocol="3.2-aligned; formal L3 Capture unchanged")


def num(value):
    return f"${value:+.3f}$"


def render_tables(report):
    answers, process, associations, coverage, settings = [], [], [], [], []
    positive_rates = []
    for row in report["records"]:
        a = row["analysis"]
        name = row["name"]
        for fmt in FORMATS:
            v = row["answers"][fmt]["aggregate"]
            answers.append([name, fmt.title(), *[num(v[k]) for k in ("quantity", "event", "capture")]])
            rates = row["answer_process"]["aggregate"][fmt]
            positive_rates.append([name, fmt.title(), f"{100*rates['answer_positive']:.2f}",
                                   "--" if fmt == "free" else
                                   f"{100*rates['answer_positive_process_negative']:.2f}"])
        for fmt in ("table", "code"):
            l1, l3 = [a["paired_point_forecasts"][s][fmt]["aggregate"] for s in ("L1", "L3-state")]
            process.append([name, fmt.title(), num(l1["process_skill"]), num(l3["process_skill"]),
                            f"{100*l1['quadrants']['A+E-']:.2f}", f"{100*l3['quadrants']['A+E-']:.2f}"])
            assoc = a["associations"][f"{fmt}|v1|real"]
            lo, hi = assoc["rho_difference"]["ci95"]
            adjusted = assoc["effect_strength_adjusted"]["correlations"]
            associations.append([name, fmt.title(), str(assoc["n"]),
                                 f"{assoc['rho_L1']['value']:.3f}", f"{assoc['rho_L3']['value']:.3f}",
                                 f"$[{lo:+.3f}, {hi:+.3f}]$", f"{adjusted['rho_L3']:.3f}" if adjusted else "--"])
        c = a["coverage"]
        coverage.append([name, str(row["anchors"]),
                         "/".join(str(a["cost"][f]["requests"]) for f in FORMATS),
                         f"{c['state_missing']}/{c['state_expected']}",
                         f"{c['effects_missing']}/{c['effects_expected']}",
                         f"{a['cost']['code']['runnable']}/{a['cost']['code']['requests']}"])
        for scope in ("L1", "L3-state"):
            for setting, label in (("bench-basic", "ID"), ("bench-gen", "OOD")):
                t, c = [a["paired_point_forecasts"][scope][f]["settings"][setting] for f in ("table", "code")]
                assert t["n"] == c["n"]
                settings.append([name, scope, label, str(t["n"]), num(t["process_skill"]), num(c["process_skill"]),
                                 f"{100*t['quadrants']['A+E-']:.2f}", f"{100*c['quadrants']['A+E-']:.2f}"])
    outputs = {
        "cot_answer_positive": table(positive_rates,
            "Answer-positive rates on common Free, Table, and Code units, averaging L1 and "
            "L3 branch-state scopes and Omni-ID/OOD equally. A+P- shares use the same denominator. "
            "Free reasoning has no prescribed process score.",
            "tab:cotanswerpositive", "llrr", r"Model & Protocol & A+ (\%) & A+P- (\%)"),
        "cot_aligned_answers": table(answers,
            "Final-answer scores averaged equally over Omni-ID and OOD. "
            "Each model uses 190 anchors except GPT-5.6 Sol (30). Protocols use the same anchors within each model.",
            "tab:cotalignedanswers", "llrrr", r"Model & Protocol & \QNT & \EVT & Capture"),
        "cot_aligned_process": table(process,
            "Intermediate-state skill and joint answer-positive/process-negative shares. "
            "Percentages use all eligible matched units. L3-state evaluates intervention-branch states.",
            "tab:cotalignedprocess", "llrrrr",
            r"Model & Protocol & L1 skill & L3-state skill & L1 A+P- (\%) & L3-state A+P- (\%)"),
        "cot_aligned_associations": table(associations,
            "Intermediate/final error associations on matched intervention units. Intervals for "
            r"$\rho_{L3}-\rho_{L1}$ resample worlds. Adjusted L3 removes task composition and observed effect strength. "
            "These are unit-level error diagnostics, not correlations of model-level Capture scores.",
            "tab:cotalignedassociations", "llrrrrr",
            r"Model & Protocol & Units & $\rho_{L1}$ & $\rho_{L3}$ & Difference CI & Adjusted L3"),
        "cot_aligned_coverage": table(coverage,
            "Completed output census and diagnostic coverage. F/T/C counts include all executed settings. "
            "Missing state/effect units are counted before protocol matching.",
            "tab:cotalignedcoverage", "lrrrrr",
            r"Model & Anchors & Outputs F/T/C & State missing & Effect missing & Runnable"),
        "cot_aligned_settings": table(settings,
            "Setting-level process diagnostics on common valid Table/Code units within each model and scope. "
            "Aggregate figures give the settings equal weight.",
            "tab:cotalignedsettings", "lllrrrrr",
            r"Model & Scope & Setting & Units & Table skill & Code skill & T A+P- (\%) & C A+P- (\%)"),
    }
    association_summary = report["process_capture"]
    summary_rows = []
    for label, summary in (("Model--protocol pairs", association_summary["pooled"]),
                           ("Table only", association_summary["by_format"]["table"]),
                           ("Code only", association_summary["by_format"]["code"]),
                           ("Model means, raw Capture", association_summary["model_means"]),
                           ("Model means, Capture completion", association_summary["model_means_completion"]),
                           ("Model means, L1 process and L1 completion", association_summary["l1_model_means"])):
        summary_rows.append([label, str(summary["n"]), num(summary["pearson"]), num(summary["spearman"])])
    outputs["cot_process_capture"] = table(summary_rows,
        "Associations between process and final-answer scores. Rows use L3 state skill and Capture "
        "unless marked L1. Model means average Table and Code before computing correlations. "
        "Completion follows the frozen main-table normalization. These are descriptive associations.",
        "tab:cotprocesscapture", "lrrr", r"Aggregation & Points & Pearson $r$ & Spearman $\rho$")
    for name, content in outputs.items():
        (OUTPUT / "tables" / f"{name}.tex").write_text(content)


def render_numbers(report):
    rows = report["records"]
    sol = next(r for r in rows if r["tag"] == "gpt56sol")
    mismatch = [100*r["answer_process"]["aggregate"][f]["answer_positive_process_negative"]
                for r in rows for f in ("table", "code")]
    rhos = [r["analysis"]["associations"][f"{f}|v1|real"]["rho_L3"]["value"]
            for r in rows for f in ("table", "code")]
    vals = dict(cotAlignedAnchors="190", cotSolAnchors="30",
                cotMismatchRange=f"{min(mismatch):.1f}--{max(mismatch):.1f}",
                cotEffectRhoRange=f"{min(rhos):.2f}--{max(rhos):.2f}")
    association = report["process_capture"]["pooled"]
    vals.update(cotProcessCapturePearson=f"{association['pearson']:.2f}",
                cotProcessCaptureSpearman=f"{association['spearman']:.2f}",
                cotProcessCaptureLow=f"{association['pearson_ci95'][0]:+.2f}",
                cotProcessCaptureHigh=f"{association['pearson_ci95'][1]:+.2f}")
    model_association = report["process_capture"]["model_means_completion"]
    vals.update(cotModelCapturePearson=f"{model_association['pearson']:.2f}",
                cotModelCaptureSpearman=f"{model_association['spearman']:.2f}",
                cotModelCaptureLow=f"{model_association['pearson_ci95'][0]:+.2f}",
                cotModelCaptureHigh=f"{model_association['pearson_ci95'][1]:+.2f}")
    for fmt in FORMATS:
        vals[f"cotSol{fmt.title()}Capture"] = f"{sol['answers'][fmt]['aggregate']['capture']:.3f}"
    vals["cotSolCodeGain"] = f"{sol['answers']['code']['aggregate']['capture']-sol['answers']['free']['aggregate']['capture']:.3f}"
    (OUTPUT / "cot_numbers.tex").write_text("% Generated by paper.build_cot_results.\n" + "".join(
        "\\newcommand{\\" + k + "}{" + v + "}\n" for k, v in vals.items()))


def plot(report, out):
    import numpy as np
    from matplotlib.colors import to_rgb
    from matplotlib.patches import Patch
    from matplotlib.ticker import PercentFormatter

    plt = gt._mpl(academic=True)
    rows = sorted(report["records"], key=lambda row: (row["swfi"], row["tag"]))
    positions = np.arange(len(rows))
    colors = dict(free="#656565", table="#367E83", code="#B37553")
    pale_colors = {fmt: tuple(.18*channel+.82 for channel in to_rgb(color))
                   for fmt, color in colors.items()}
    markers = dict(free="o", table="s", code="^")
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.25),
                             gridspec_kw={"width_ratios": [1.2, 1, 1.15]})
    positive_values = []
    for index, fmt in enumerate(FORMATS):
        rates = [row["answer_process"]["aggregate"][fmt] for row in rows]
        total = np.array([100*rate["answer_positive"] for rate in rates])
        mismatch = np.array([100*(rate["answer_positive_process_negative"] or 0) for rate in rates])
        if np.any(mismatch > total) or np.any(mismatch < 0) or np.any(total > 100) or np.any(total < 0):
            raise ValueError("Invalid answer/process rate decomposition")
        positive_values.extend(total)
        axes[0].bar(positions+(index-1)*.24, total-mismatch, width=.205,
                    color=colors[fmt], edgecolor="none", linewidth=0,
                    label=f"_{fmt}_answer_positive", zorder=3)
        if fmt != "free":
            axes[0].bar(positions+(index-1)*.24, mismatch, bottom=total-mismatch, width=.205,
                        color=pale_colors[fmt], edgecolor=colors[fmt], linewidth=0,
                        hatch="////", hatch_linewidth=.5, label="_nolegend_", zorder=3)
    axes[0].set(title="(a) Answer and process performance", ylabel=r"Answer-positive rate $\uparrow$",
                ylim=(20, min(100, math.ceil(max(positive_values)/10)*10+5)))
    axes[0].yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    association = report["process_capture"]
    points = association["model_mean_points"]
    process_values = np.array([point["process"] for point in points])
    capture_values = np.array([point["capture_completion"] for point in points])
    axes[1].scatter(process_values, capture_values, color=colors["table"], s=30,
                    edgecolor="white", linewidth=.4, zorder=3)
    label_positions = {
        "q38-27b": (.80, .05, "right"),
        "gemma4-31b": (.40, .35, "right"),
        "flash-next": (.80, .17, "right"),
        "kimi-k3": (.99, .25, "right"),
        "gpt56sol": (.97, .64, "right"),
    }
    for point in points:
        horizontal, vertical, alignment = label_positions[point["tag"]]
        axes[1].annotate(point["name"], (point["process"], point["capture_completion"]),
                         xytext=(horizontal, vertical), textcoords="axes fraction", fontsize=7,
                         ha=alignment, va="center",
                         bbox=dict(facecolor="white", edgecolor="none", pad=.2),
                         arrowprops=dict(arrowstyle="-", color="#888888", lw=.5, shrinkA=2, shrinkB=3),
                         zorder=4)
    if np.ptp(process_values) > 0:
        slope, intercept = np.polyfit(process_values, capture_values, 1)
        limits = np.array([process_values.min(), process_values.max()])
        axes[1].plot(limits, slope*limits+intercept, color="#858585", linewidth=.8,
                     linestyle="--", label="_model_mean_fit", zorder=2)
    axes[1].set(title="(b) Process quality and Capture", xlabel=r"L3 process skill $\uparrow$",
                ylabel=r"Capture completion $\uparrow$", xlim=(process_values.min()-.02, process_values.max()+.025),
                ylim=(capture_values.min()-6, capture_values.max()+8))
    axes[1].set_xticks([-.05, 0, .05, .10])
    axes[1].yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    axes[1].text(.04, .96, f"$r={association['model_means_completion']['pearson']:.2f}$, "
                 f"$\\rho={association['model_means_completion']['spearman']:.2f}$",
                 transform=axes[1].transAxes, fontsize=8, va="top")
    completion_values = []
    for fmt in FORMATS:
        values = [row["answers"][fmt]["aggregate"]["capture_completion"] for row in rows]
        completion_values.extend(values)
        axes[2].plot(positions, values, color=colors[fmt], marker=markers[fmt], markersize=4.5,
                     markeredgecolor="white", markeredgewidth=.4, linewidth=1,
                     label=fmt.title(), zorder=3)
    axes[2].set(title="(c) Reasoning-protocol gains", ylabel=r"Capture completion $\uparrow$",
                ylim=(min(20, math.floor(min(completion_values)/10)*10),
                      max(completion_values)+18))
    axes[2].yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    for axis in (axes[0], axes[2]):
        axis.set_xticks(positions, [row["short"] for row in rows])
        axis.tick_params(axis="x", labelsize=7)
        axis.set_xlim(-.6, len(rows)-.4)
    for axis in axes:
        axis.tick_params(axis="y", length=2, pad=3)
        axis.tick_params(axis="x", length=2, pad=3)
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#DDDDDD", lw=.5)
        axis.set_axisbelow(True)
    legend_style = dict(loc="upper left", frameon=False, fontsize=7, handlelength=1,
                        handleheight=.7, handletextpad=.4, borderaxespad=.3)
    format_legend = axes[0].legend(
        handles=[Patch(facecolor=colors[fmt], edgecolor="none", label=fmt.title()) for fmt in FORMATS],
        bbox_to_anchor=(.01, 1), ncol=3, columnspacing=1, **legend_style)
    axes[0].add_artist(format_legend)
    axes[0].legend(
        handles=[Patch(facecolor="#F2F3F3", edgecolor="#778084", linewidth=0,
                       hatch="////", hatch_linewidth=.5,
                       label="Process at/below reference")],
        bbox_to_anchor=(.01, .865), **legend_style)
    axes[2].legend(loc="upper left", ncol=3, frameon=False, fontsize=7,
                    handlelength=1.1, handletextpad=.3, columnspacing=.65, borderaxespad=.4)
    fig.tight_layout(w_pad=1.2)
    gt._save(fig, Path(out))
    plt.close(fig)
    return True


def main():
    report = collect()
    render_tables(report)
    render_numbers(report)
    plot(report, OUTPUT / "figures/cot.pdf")
    (OUTPUT / "cot_results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print("Rendered Section 6: four models on 190 anchors and Sol on 30; L3 uses Capture.")


if __name__ == "__main__":
    main()
