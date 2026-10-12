"""Render a paired-gain plot and compact benchmark comparison from validated data."""
import hashlib
import json
from pathlib import Path

import numpy as np

from paper.audit_composite_forecasting import normalize
from paper.build_main_table import OMNI, SPEC as CAPABILITY_SPEC
from paper.gen_tables import _mpl, _save

ROOT = Path(__file__).resolve().parents[1]
GAMES = ROOT / "bench/agentplay/games"
FULL = GAMES / "deliberative150_r2"
CAPABILITY = ROOT / "bench/measure/out/section5/main_table_compact.json"
LOCAL = GAMES / "usability_replication_20260925/analysis.json"
FIGURES = ROOT / "paper/figures"
TAGS = {"flash": "tk", "kimi": "kimi-k3", "sol": "gpt56sol", "opus": "opus5"}
NAMES = {"none": "None", "flash": "Flash-Next", "kimi": "Kimi K3", "sol": "Sol", "opus": "Opus"}
COLORS = {"none": "#5D6168", "flash": "#71964B", "kimi": "#4C78A8", "sol": "#B66D48", "opus": "#8B6F9E"}


def load(path, provenance):
    content = path.read_bytes()
    provenance[str(path.relative_to(ROOT))] = hashlib.sha256(content).hexdigest()
    return json.loads(content)


def collect():
    provenance = {}
    manifest = load(FULL / "manifest.json", provenance)
    reference = load(FULL / "paired_results.json", provenance)
    assert (FULL / "VERIFIED_DONE").exists()
    worlds = manifest["worlds"]
    assert len(worlds) == len(set(worlds)) == 76
    turns = np.arange(manifest["start"] + 1, manifest["start"] + manifest["turns"] + 1)
    arrays = {}
    for arm in ("none", "flash", "kimi"):
        rows = []
        for world in worlds:
            path = FULL / arm / (world.rsplit("-", 1)[1] + "_e0") / "summary.json"
            summary = load(path, provenance)
            assert summary["world"] == world
            assert summary["n_turns"] == 150
            trajectory = np.asarray(summary["score_traj"], dtype=float)
            np.testing.assert_array_equal(trajectory[:, 0], turns)
            assert np.isfinite(trajectory[:, 1]).all()
            rows.append(trajectory[:, 1])
        arrays[arm] = np.asarray(rows)
    for turn in (160, 210):
        index = int(np.flatnonzero(turns == turn)[0])
        for arm in arrays:
            np.testing.assert_allclose(arrays[arm][:, index].mean(), reference[str(turn)]["means"][arm])
        for arm in ("flash", "kimi"):
            delta = arrays[arm][:, index] - arrays["none"][:, index]
            np.testing.assert_allclose(delta.mean(), reference[str(turn)][arm + "_vs_none"]["mean"])
            np.testing.assert_allclose(delta.std(ddof=1) / np.sqrt(76), reference[str(turn)][arm + "_vs_none"]["se"])
    weights = np.random.default_rng(20260925).multinomial(76, np.full(76, 1 / 76), size=5000) / 76
    trajectories = {"n": 76, "turns": turns.tolist(), "arms": {}}
    for arm, scores in arrays.items():
        row = {"mean": scores.mean(axis=0).tolist()}
        if arm != "none":
            paired = scores - arrays["none"]
            bootstrap = weights @ paired
            row["paired_mean"] = paired.mean(axis=0).tolist()
            row["paired_ci95"] = np.quantile(bootstrap, [.025, .975], axis=0).tolist()
        trajectories["arms"][arm] = row
    capability_data = load(CAPABILITY, provenance)
    capability_spec = load(CAPABILITY_SPEC, provenance)
    if capability_data["spec_sha256"] != provenance[str(CAPABILITY_SPEC.relative_to(ROOT))]:
        raise ValueError("Capability scores and normalization specification do not match")
    records = {row["tag"]: row for row in capability_data["records"]}
    utility = load(LOCAL, provenance)
    assert utility["n_worlds"] == 36
    models = {}
    for arm, tag in TAGS.items():
        record = records[tag]
        normalized = normalize(record["values"], capability_spec["ranges"][OMNI])
        level_percent = {
            level: float(100 * np.mean([normalized[metric] for metric in metrics]))
            for level, metrics in capability_spec["levels"].items()
        }
        effect = utility["pooled_four_repeats"][arm + "_vs_self"]
        assert effect["n"] == 36
        models[arm] = {"quantity": record["values"]["quantity"], "trait": record["values"]["trait"],
                       "capture": record["values"]["capture"], "swfi": record["overall"],
                       "level_percent": level_percent,
                       "gain": effect["mean"], "gain_se": effect["se"], "gain_ci95": effect["bootstrap95"]}
        assert all(np.isfinite(models[arm][key]) for key in ("quantity", "trait", "capture", "swfi", "gain", "gain_se"))
    return {"full_game": trajectories, "local_models": models, "provenance": provenance,
            "capability_aggregation": {"setting": OMNI, "levels": capability_spec["levels"],
                                       "scaling": "100 * within-level mean of (score-floor)/(ceiling-floor); no clipping"},
            "intervals": "pointwise 95% world-bootstrap; local vertical bars reuse frozen four-execution intervals",
            "scope": "Descriptive four-model comparisons; no regression, jitter, outcome filtering, or smoothing"}


def style(axis):
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(length=2.5, pad=3)
    axis.set_axisbelow(True)
    axis.grid(axis="y", color="#D5D8DC", linewidth=.5)


def trajectories_figure(data):
    plt = _mpl(academic=True)
    figure, axis = plt.subplots(figsize=(3.20, 1.28))
    figure.subplots_adjust(left=.15, right=.98, bottom=.30, top=.96)
    turns = np.asarray(data["turns"])
    lower_limit, upper_limit = -1.5, 7.5
    for arm in ("flash", "kimi"):
        row = data["arms"][arm]
        assert lower_limit < min(row["paired_mean"]) <= max(row["paired_mean"]) < upper_limit
        lower, upper = row["paired_ci95"]
        axis.fill_between(turns, lower, upper, color=COLORS[arm], alpha=.14, linewidth=0)
        axis.plot(turns, row["paired_mean"], color=COLORS[arm], linewidth=1.5, label=NAMES[arm])
        axis.scatter(turns[-1], row["paired_mean"][-1], s=14, color=COLORS[arm], zorder=5)
        axis.annotate(f"{row['paired_mean'][-1]:+.2f}", (turns[-1], row["paired_mean"][-1]),
                      xytext=(-3, -1 if arm == "flash" else -1), textcoords="offset points",
                      ha="right", color=COLORS[arm], fontsize=8)
    style(axis)
    axis.set_xlim(60, 214)
    axis.set_ylim(lower_limit, upper_limit)
    axis.set_xticks([60, 100, 140, 180, 210])
    axis.set_yticks([0, 2, 4, 6])
    axis.set_ylabel("Score gain vs None", labelpad=3)
    axis.axvline(160, color="#B0B0B0", linestyle=":", linewidth=.7)
    axis.axhline(0, color="#65686C", linewidth=.85, linestyle="--")
    axis.legend(loc="upper left", ncol=2,
                frameon=False, columnspacing=1.2, handlelength=1.7, borderaxespad=.5)
    _save(figure, FIGURES / "usability_trajectories.pdf")
    plt.close(figure)


def capability_table(models):
    lines = [r"\begin{tabular}{@{}lrrrr@{}}", r"\toprule",
             r"Model & \Lone & \Ltwo & \Lthree & $\Delta$ score \\",
             r"\midrule"]
    best = {level: max(row["level_percent"][level] for row in models.values()) for level in ("L1", "L2", "L3")}
    for arm, row in models.items():
        cells = [NAMES[arm]]
        for level in best:
            value = row["level_percent"][level]
            cell = f"{value:.1f}"
            if value == best[level]:
                cell = r"\mathbf{" + cell + "}"
            cells.append("$" + cell + r"\%$")
        cells.append(f"${row['gain']:+.2f}$")
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (ROOT / "paper/tables/usability_compact.tex").write_text("\n".join(lines) + "\n")


def main():
    data = collect()
    trajectories_figure(data["full_game"])
    capability_table(data["local_models"])
    data["display"] = {"gain_axis_limits": [-1.5, 7.5], "confidence_bands_clipped": True,
                       "mean_curves_clipped": False,
                       "capabilities": "Omni-ID level completion percentages; bold level maxima only"}
    (ROOT / "paper/usability_figure_data.json").write_text(json.dumps(data, indent=2) + "\n")
    print("Validated 228 trajectories; rendered the compact gain plot and capability table.")


if __name__ == "__main__":
    main()
