"""Render compact appendix displays without changing scores or finalized figures."""
import hashlib
import json

import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from paper import gen_tables as tables
from paper.build_results_analysis import write_mechanism_table


OUTPUT = tables.ROOT / "bench/measure/out/section5/appendix_summary.json"
MECHANISM = tables.ROOT / "bench/measure/out/section5/results.json"
FAMILY_LABELS = ["Economy", "Scale", "Tech./culture", "Military", "Diplomacy", "Score"]
SETTING_LABELS = ["Omni-ID", "Ego", "Omni-OOD"]


def collect(scores, references):
    names = []
    for name, tag, _ in tables.LINES:
        names.append("Expert" if tag == tables.EXPERT_TAG else
                     "Opus 5" if tag == "opus5" else name.replace("~", " "))
    names.append("Ceiling")
    values = []
    for setting, _ in tables.SETTINGS:
        panel = [[tables.m_1a_fam(scores.get(tag), setting, family) for family in tables.FAMS]
                 for _, tag, _ in tables.LINES]
        panel.append([references[setting]["fam"][family] for family in tables.FAMS])
        values.append(panel)
    return dict(names=names, families=tables.FAMS, settings=[setting for setting, _ in tables.SETTINGS],
                values=values)


def plot(report):
    plt = tables._mpl(academic=True)
    values = np.array(report["values"], dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size and (finite.min() < -1 or finite.max() > 1):
        raise ValueError("Family skill outside the fixed display range")
    cmap = LinearSegmentedColormap.from_list(
        "family_skill", ["#BC777F", "#FAFAFA", "#367E83"]).with_extremes(bad="#E2E2E2")
    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    figure, axes = plt.subplots(1, 3, figsize=(6.75, 3.55), sharey=True)
    figure.subplots_adjust(left=.185, right=.985, top=.89, bottom=.26, wspace=.085)
    for panel, axis, title in zip(values, axes, SETTING_LABELS):
        image = axis.imshow(np.ma.masked_invalid(panel), cmap=cmap, norm=norm, aspect="auto")
        axis.set_title(title, fontsize=9, pad=6)
        axis.set_xticks(range(len(FAMILY_LABELS)), FAMILY_LABELS, rotation=45, ha="right", fontsize=7)
        axis.set_yticks(range(len(report["names"])), report["names"], fontsize=8)
        axis.tick_params(length=0, pad=3)
        for spine in axis.spines.values():
            spine.set_visible(False)
        for boundary in (len(report["names"])-2.5, len(report["names"])-1.5):
            axis.axhline(boundary, color="#777777", linewidth=.65)
        for row, column in np.ndindex(panel.shape):
            value = panel[row, column]
            label = f"{value:.2f}" if np.isfinite(value) else "--"
            color = "white" if np.isfinite(value) and abs(value) > .7 else "#242424"
            axis.text(column, row, label, ha="center", va="center", fontsize=7.5, color=color)
    scale_axis = figure.add_axes([.365, .075, .36, .025])
    scale = figure.colorbar(image, cax=scale_axis, orientation="horizontal", ticks=[-1, -.5, 0, .5, 1])
    scale.set_label("Skill relative to trend", fontsize=7, labelpad=2)
    scale.ax.tick_params(labelsize=7, length=2, pad=2)
    scale.outline.set_linewidth(.4)
    return figure


def main():
    paths = [tables.score_path(tag) for _, tag, _ in tables.LINES] + [tables.CEIL, MECHANISM]
    provenance = {str(path.relative_to(tables.ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in paths if path is not None}
    scores = {tag: tables.load(tag) for _, tag, _ in tables.LINES}
    report = collect(scores, tables.ceiling())
    figure = plot(report)
    tables._save(figure, tables.OUT_F / "quantity_families.pdf")
    import matplotlib.pyplot as plt
    plt.close(figure)
    mechanism = json.loads(MECHANISM.read_text())["mechanism"]
    write_mechanism_table(mechanism)
    for source, expected in provenance.items():
        if hashlib.sha256((tables.ROOT / source).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Source changed during rendering: {source}")
    OUTPUT.write_text(json.dumps(dict(report, provenance=provenance), indent=2, allow_nan=False) + "\n")
    print("Appendix: three-setting family heatmap and compact mechanism table")


if __name__ == "__main__":
    main()
