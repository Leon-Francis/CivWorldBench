"""Render the direction-conditioned magnitude audit without changing benchmark scores."""
import json
import math

from bench.measure.direction_magnitude import OUT
from paper.gen_tables import EXPERT_TAG, LINES, ROOT, SETTINGS, _mpl, _save


SHORT_NAMES = {
    "gpt-oss-120b": "gpt-oss-120b", "gemma4-31b": "Gemma 31B",
    "nemotron3-ultra": "Nemotron Ultra", "inkling": "Inkling", "q38-27b": "Qwen 27B",
    "gpt56sol": "Sol-Medium", "dsv41-flash": "DeepSeek Flash", "tk": "Flash-Next",
    "kimi-k3": "Kimi K3", "glm-5.3": "GLM-5.3", "opus5": "Opus-Medium",
    EXPERT_TAG: "Expert",
}

FIGURE_COLORS = {
    "gpt-oss-120b": "#60646C",
    "kimi-k3": "#4C78A8",
    "glm-5.3": "#367E83",
    "tk": "#71964B",
    "q38-27b": "#9A913E",
    "nemotron3-ultra": "#C17F9B",
    "dsv41-flash": "#5B9EB0",
    "gpt56sol": "#B66D48",
    "opus5": "#8B6F9E",
}


def make_figure(result):
    plt = _mpl(academic=True)
    from matplotlib.ticker import PercentFormatter

    tags = list(FIGURE_COLORS)
    figure = plt.figure(figsize=(3.1, 1.9))
    main_axis = figure.add_axes([.18, .44, .625, .52])
    opus_axis = figure.add_axes([.84, .44, .13, .52], sharey=main_axis)
    main_axis.set_xlim(.584, .652)
    opus_axis.set_xlim(.680, .694144)
    plot_names = {**SHORT_NAMES, "gpt56sol": "GPT-5.6 Sol", "opus5": "Opus 5"}
    magnitudes = []
    handles = []
    for tag in tags:
        value = result["models"][tag]["bench-basic|v1"]
        magnitude = value["relative_magnitude"]
        magnitudes.append(magnitude)
        position = (value["direction_valid_active"], magnitude["median"])
        axis = opus_axis if tag == "opus5" else main_axis
        color = FIGURE_COLORS[tag]
        axis.plot([position[0], position[0]], [magnitude["q25"], magnitude["q75"]],
                  color=color, alpha=.65, linewidth=1.1, zorder=2)
        axis.plot([position[0], position[0]], [magnitude["q25"], magnitude["q75"]],
                  linestyle="none", marker="_", color=color, alpha=.8,
                  markersize=4, markeredgewidth=.7, zorder=2)
        marker = {"tk": "s", "dsv41-flash": "D", "nemotron3-ultra": "^"}.get(tag, "o")
        handles.append(axis.scatter(*position, color=color, s=24, marker=marker, edgecolor="white",
                                    linewidth=.5, zorder=4, label=plot_names[tag]))
    upper = max(3.5, math.ceil(max(value["q75"] for value in magnitudes) * 1.1 * 2) / 2)
    main_axis.set_xticks([.59, .61, .63, .65])
    opus_axis.set_xticks([.69])
    main_axis.set_ylabel("Relative effect\nmagnitude", labelpad=3)
    figure.text(.575, .285, r"Correct direction $\uparrow$", ha="center")
    figure.legend(handles=handles, loc="lower center", bbox_to_anchor=(.52, .035),
                  ncol=3, frameon=False, fontsize=7.5, handlelength=.8,
                  handletextpad=.3, columnspacing=.6, labelspacing=.2)
    for axis in (main_axis, opus_axis):
        axis.set_ylim(0, upper)
        axis.set_yticks(list(range(math.floor(upper) + 1)))
        axis.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        axis.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        axis.tick_params(length=2, pad=3)
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_axisbelow(True)
        axis.grid(color="#DDDDDD", linewidth=.5)
        axis.axhline(1, color="#9B6042", linestyle="--", linewidth=.9, label="Magnitude match 100%")
        for mark in axis.collections:
            for direction, magnitude in mark.get_offsets():
                if not (axis.get_xlim()[0] < direction < axis.get_xlim()[1]
                        and 0 < magnitude < upper):
                    raise ValueError("A model point falls outside the display bounds")
    opus_axis.spines["left"].set_visible(False)
    opus_axis.tick_params(axis="y", left=False, labelleft=False)
    opus_axis.text(.694, 1.08, "100%", fontsize=8, ha="right", color="#9B6042")
    for edge in (.805, .84):
        figure.add_artist(plt.Line2D([edge - .006, edge + .006], [.43, .45],
                                    transform=figure.transFigure, color="#333333", linewidth=.8))
    if not all(0 < value["q25"] <= value["median"] <= value["q75"] < upper
               for value in magnitudes):
        raise ValueError("A model interquartile range falls outside the display bounds")
    _save(figure, ROOT / "paper/figures/direction_magnitude.pdf")
    plt.close(figure)


def interval(value, name, percent=True):
    point = value[name]
    if point is None:
        return "--"
    bounds = value.get("ci95", {}).get(name)
    if percent:
        point_text = f"{100*point:.0f}"
        return point_text + (f" [{100*bounds[0]:.0f}, {100*bounds[1]:.0f}]" if bounds else "")
    point_text = f"{point:+.2f}"
    return point_text + (f" [{bounds[0]:+.2f}, {bounds[1]:+.2f}]" if bounds else r"$^{*}$")


def make_tables(result):
    blocks = []
    for position, (setting, label) in enumerate(SETTINGS):
        lines = [r"\begin{table}[t]\centering\scriptsize",
                 r"\setlength{\tabcolsep}{2pt}\renewcommand{\arraystretch}{1.12}",
                 r"\caption{Direction-conditioned magnitude diagnostics in " + label +
                 r". Counts give eligible, missing, zero-response, and correct-sign cells. "
                 r"Direction is the percentage correct among valid nonzero responses. "
                 r"Magnitude-bin percentages condition on a correct sign. "
                 r"Brackets give 95\% world-bootstrap intervals. $S_{+}$ is selected-subset capture, "
                 r"not a replacement main score. An asterisk marks an omitted interval due to a singleton-world operator.}",
                 r"\label{tab:directionmagnitude" + str(position) + "}",
                 r"\begin{tabular}{@{}lrrrrrr@{}}\toprule",
                 r"Model & Counts & Direction & $<0.5\times$ & $0.5$--$2\times$ & $>2\times$ & $S_{+}$ \\",
                 r"\midrule"]
        for name, tag, aa in LINES:
            value = result["models"][tag][setting]
            counts = value["counts"]
            row = [SHORT_NAMES[tag], "/".join(str(counts[key]) for key in ("cells", "missing", "zero", "correct")),
                   interval(value, "direction_valid_active"), f"{100*value['under_half']:.0f}",
                   f"{100*value['within_factor_two']:.0f}", interval(value, "over_twice"),
                   interval(value, "correct_capture", percent=False)]
            if tag == EXPERT_TAG:
                lines.append(r"\midrule")
            lines.append(" & ".join(row) + r" \\")
        lines.extend([r"\bottomrule\end{tabular}", r"\end{table}"])
        blocks.append("\n".join(lines))
    (ROOT / "paper/tables/direction_magnitude.tex").write_text("\n\n".join(blocks) + "\n")
    lines = [r"\begin{table}[t]\centering\scriptsize",
             r"\setlength{\tabcolsep}{3pt}",
             r"\caption{Magnitude sensitivity in \omniID. All percentages condition on a correct sign. "
             r"The sensitivity filter removes effects at or below each operator's lower quartile "
             r"before selecting correct-sign predictions. Excluded counts refer instead to valid "
             r"predictions omitted by the inherited main scoring mask.}",
             r"\label{tab:magnitudesensitivity}",
             r"\begin{tabular}{@{}lrrrrrr@{}}\toprule",
             r"Model & $<1\times$ & $>1\times$ & $>2\times$ & Filtered $>2\times$ & Filtered $n_{+}$ & Excluded \\",
             r"\midrule"]
    for name, tag, aa in LINES:
        value = result["models"][tag]["bench-basic|v1"]
        sensitivity = value["above_operator_q25"]
        row = [SHORT_NAMES[tag], *[f"{100*value[key]:.1f}" for key in ("under_true", "over_true", "over_twice")],
               f"{100*sensitivity['over_twice']:.1f}", str(sensitivity["counts"]["correct"]),
               str(value["excluded_valid_outside_main_mask"])]
        lines.append(" & ".join(row) + r" \\")
    lines.extend([r"\bottomrule\end{tabular}", r"\end{table}"])
    (ROOT / "paper/tables/magnitude_sensitivity.tex").write_text("\n".join(lines) + "\n")
    lines = [r"\begin{table}[t]\centering\scriptsize",
             r"\setlength{\tabcolsep}{5pt}",
             r"\caption{Relative effect magnitudes conditional on correct direction. "
             r"Entries give the median and [25th, 75th] percentiles, all in percentages. "
             r"The ratio uses absolute predicted and realized effects in the scoring space. "
             r"100\% denotes a magnitude match. Brackets describe sample dispersion, not confidence intervals.}",
             r"\label{tab:relativemagnitude}",
             r"\begin{tabular}{@{}lrrr@{}}\toprule",
             r"Model & \omniID & \ego & \omniOOD \\", r"\midrule"]
    for name, tag, aa in LINES:
        display_name = {"gpt56sol": "GPT-5.6 Sol", "opus5": "Opus 5"}.get(tag, SHORT_NAMES[tag])
        row = [display_name]
        for setting, label in SETTINGS:
            value = result["models"][tag][setting]["relative_magnitude"]
            row.append(f"{100*value['median']:.1f} [{100*value['q25']:.1f}, {100*value['q75']:.1f}]")
        lines.append(" & ".join(row) + r" \\")
    lines.extend([r"\bottomrule\end{tabular}", r"\end{table}"])
    (ROOT / "paper/tables/relative_magnitude.tex").write_text("\n".join(lines) + "\n")


def main():
    result = json.loads((OUT / "results.json").read_text())
    make_figure(result)
    make_tables(result)
    values = [result["models"][tag]["bench-basic|v1"]["over_twice"]
              for name, tag, aa in LINES if tag != EXPERT_TAG]
    (ROOT / "paper/direction_magnitude_numbers.tex").write_text(
        r"\newcommand{\magnitudeOvershootRange}{" + f"{100*min(values):.0f}--{100*max(values):.0f}" + r"\%}" + "\n")
    print("Wrote direction-magnitude figure, appendix tables, and number macro")


if __name__ == "__main__":
    main()
