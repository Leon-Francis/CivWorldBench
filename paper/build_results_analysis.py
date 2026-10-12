"""Build paired analyses for Section 5 from existing scored predictions only."""
import collections
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from bench.spec_v2 import FAMILIES
from bench.measure.effect_error_comparison import transform
from paper.gen_tables import LINES, SETTINGS, TRAITS, PARAMS, load, _mpl, _save

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "bench/runs"
OUT = ROOT / "bench/measure/out/section5"
FAMILY = {variable: family for family, variables in FAMILIES.items() for variable in variables}
DIRECT = {"L1-gold-swing": {"gold"}, "L1-bulbs": {"techs_learned"},
          "L2-city-kill": {"n_cities", "city_size_total"}, "L2-gov-indirect": {"techs_learned"}}
LABELS = {"tk": "Flash-Next", "q38-27b": "Qwen27B", "gemma4-31b": "Gemma31B"}
CHANNEL_LABELS = {"visible_short": "Visible, earliest", "visible_long": "Visible, later",
                  "other_channels": "Other channels"}
ABLATION_LABELS = {"k1": "One frame", "oracle": "Future narrative", "primer": "Rules primer",
                   "belief": "Study notes", "nothink": "Thinking off", "base_rate": "Base rate supplied"}


def read(path, kinds=("1a", "1b", "2", "3"), reference=None):
    groups = collections.defaultdict(dict)
    occurrences = collections.Counter()
    with gzip.open(path, "rt") as source:
        for line in source:
            row = json.loads(line)
            kind = row["kind"]
            if kind not in kinds:
                continue
            setting = (row["set"], row["cond"], kind)
            key = (row["world"], row["t1"], row.get("op"), row.get("arm_kind"),
                   row.get("civ"), row.get("var") or row.get("tpl") or row.get("target"), row.get("h"))
            if kind == "1b":
                occurrence = occurrences[(setting, key)]
                occurrences[(setting, key)] += 1
                key = (*key, occurrence)
            if kind == "3":
                if (row.get("sub") != "1a" or not row.get("in_band") or not row.get("scored_var")
                        or row.get("truth_moved") is not True):
                    continue
                if reference is not None:
                    cond = {"k1mech": "v1", "k1mech_ego": "v2"}[row["cond"]]
                    matched = reference.get((row["set"], cond, kind), {}).get(key)
                    if matched is None:
                        continue
                    row["p50_ignore"] = matched[6]
                elif "skill_arm" not in row:
                    continue
                if any(row.get(field) is None for field in ("p50", "p50_ignore", "y", "y_fac")):
                    continue
                truth = (transform(row["var"], row["y"]) - transform(row["var"], row["y_fac"])) / row["norm"]
                prediction = (transform(row["var"], row["p50"]) - transform(row["var"], row["p50_ignore"])) / row["norm"]
                value = (abs(prediction-truth), abs(truth), float(prediction*truth > 0),
                         float(prediction != 0), row["y"], row["y_fac"], row["p50_ignore"])
            elif kind == "1a":
                fields = ("ae", "ae_trend") if row["var"] in ("n_enemies", "n_allies") else ("crps", "crps_ref")
                if any(row.get(field) is None for field in fields):
                    continue
                value = (row[fields[0]], row[fields[1]], row["y"])
            elif kind == "1b":
                if row.get("brier") is None or row.get("brier_base") is None:
                    continue
                value = (row["brier_base"] - row["brier"], row["truth"])
            else:
                if row.get("hit") is None:
                    continue
                value = (row["hit"], row["y"])
            if key in groups[setting]:
                raise ValueError(f"Duplicate {path}: {setting}, {key}")
            groups[setting][key] = value
    print(f"Read {path}", flush=True)
    return groups


def paired_interval(base, variant, worlds):
    delta = np.asarray(variant)-np.asarray(base)
    rng = np.random.default_rng(2026)
    draws = [float(np.mean(delta[rng.integers(len(worlds), size=len(worlds))])) for _ in range(1000)]
    return dict(base=float(np.mean(base)), variant=float(np.mean(variant)), delta=float(delta.mean()),
                ci95=list(map(float, np.quantile(draws, [.025, .975]))), n_worlds=len(worlds),
                interval_method="paired world bootstrap, 1000 draws")


def paired_qnt(base, variant):
    common = sorted(base.keys() & variant.keys())
    sums = collections.defaultdict(lambda: collections.defaultdict(lambda: np.zeros((2, 2))))
    for key in common:
        if base[key][2] != variant[key][2]:
            raise ValueError(f"Quantity truth mismatch: {key}")
        if not math.isclose(base[key][1], variant[key][1], abs_tol=1e-10):
            raise ValueError(f"Quantity reference mismatch: {key}")
        sums[key[0]][FAMILY[key[5]]] += np.asarray([base[key][:2], variant[key][:2]])
    worlds, values = [], []
    for world, families in sorted(sums.items()):
        skills = [np.clip(1-totals[:, 0]/totals[:, 1], -1, 1)
                  for totals in families.values() if np.all(totals[:, 1] > 0)]
        if skills:
            worlds.append(world)
            values.append(np.mean(skills, axis=0))
    values = np.asarray(values)
    result = paired_interval(values[:, 0], values[:, 1], worlds)
    result.update(n_cells=len(common), n_anchors=len({key[:2] for key in common}),
                  base_valid=len(base), variant_valid=len(variant))
    return result


def paired_event(base, variant):
    common = sorted(base.keys() & variant.keys())
    worlds = sorted({key[0] for key in common})
    index = {world: offset for offset, world in enumerate(worlds)}
    totals = np.zeros((len(worlds), 3))
    for key in common:
        if base[key][1] != variant[key][1]:
            raise ValueError(f"Event truth mismatch: {key}")
        totals[index[key[0]]] += (base[key][0], variant[key][0], 1)
    point = totals.sum(axis=0)
    rng = np.random.default_rng(2026)
    draws = []
    for _ in range(1000):
        trial = totals[rng.integers(len(worlds), size=len(worlds))].sum(axis=0)
        draws.append((trial[1]-trial[0])/trial[2])
    return dict(base=float(point[0]/point[2]), variant=float(point[1]/point[2]),
                delta=float((point[1]-point[0])/point[2]), ci95=list(map(float, np.quantile(draws, [.025, .975]))),
                n_worlds=len(worlds), n_cells=len(common), n_anchors=len({key[:2] for key in common}),
                interval_method="paired world bootstrap, 1000 draws")


def paired_capture(base, variant, arm="real"):
    common = sorted(key for key in base.keys() & variant.keys() if key[3] == arm)
    counts = collections.Counter(key[2] for key in common)
    common = [key for key in common if counts[key[2]] >= 3]
    if not common:
        return None
    worlds = sorted({key[0] for key in common})
    operators = sorted({key[2] for key in common})
    world_index = {world: index for index, world in enumerate(worlds)}
    operator_index = {operator: index for index, operator in enumerate(operators)}
    totals = np.zeros((len(worlds), len(operators), 2, 5))
    mismatched_controls = 0
    for key in common:
        if base[key][4:6] != variant[key][4:6]:
            raise ValueError(f"Arm truth mismatch: {key}")
        mismatched_controls += base[key][6] != variant[key][6]
        totals[world_index[key[0]], operator_index[key[2]]] += np.asarray(
            [(*base[key][:4], 1), (*variant[key][:4], 1)])

    def metric(sums):
        return np.clip(1-sums[:, :, 0]/sums[:, :, 1], -1, 1).mean(axis=0)

    summed = totals.sum(axis=0)
    points = metric(summed)
    operator_worlds = [int(np.count_nonzero(totals[:, index, 0, 4])) for index in range(len(operators))]
    result = dict(base=float(points[0]), variant=float(points[1]), delta=float(points[1]-points[0]),
                  n_cells=len(common), n_worlds=len(worlds), n_operators=len(operators),
                  n_anchors=len({key[:2] for key in common}), mismatched_control_predictions=int(mismatched_controls),
                  base_valid=sum(key[3] == arm for key in base), variant_valid=sum(key[3] == arm for key in variant),
                  by_operator={operator: dict(base=float(np.clip(1-summed[index, 0, 0]/summed[index, 0, 1], -1, 1)),
                                              variant=float(np.clip(1-summed[index, 1, 0]/summed[index, 1, 1], -1, 1)),
                                              n_cells=int(summed[index, 0, 4]), n_worlds=operator_worlds[index])
                               for index, operator in enumerate(operators)})
    if min(operator_worlds) >= 2:
        rng = np.random.default_rng(2026)
        draws = []
        for _ in range(1000):
            weights = rng.exponential(size=len(worlds))
            trial = metric((totals*weights[:, None, None, None]).sum(axis=0))
            draws.append(trial[1]-trial[0])
        result["ci95"] = list(map(float, np.quantile(draws, [.025, .975])))
        result["interval_method"] = "paired world Bayesian bootstrap, 1000 draws; fixed operators"
    else:
        result["ci95"] = None
        result["interval_warning"] = "Some operators have only one world; between-world uncertainty is not estimable."
    return result


def write_mechanism_table(mechanism):
    rows = []
    for tag, settings in mechanism.items():
        for setting, arms in settings.items():
            for arm, value in arms.items():
                rows.append([LABELS[tag], dict(SETTINGS)[setting], arm,
                             f"{value['n_cells']}/{value['n_operators']}",
                             number(value["base"]), number(value["variant"]), number(value["delta"])])
    content = table(rows,
        "Paired Capture with mechanism cards and a shared no-intervention prediction. "
        "Counts denote cells/operators. Real and placebo arms use separate moved-cell sets.",
        "tab:mechanism", "llllrrr", "Model & Setting & Arm & Cells/ops & Base & Card & Change")
    (ROOT / "paper/tables/mechanism_paired.tex").write_text(content)
    return content


def esc(text):
    return "".join({"_": r"\_", "%": r"\%", "&": r"\&", "$": r"\$", "#": r"\#",
                    "{": r"\{", "}": r"\}"}.get(char, char) for char in text)


def render_mechanism_cards(cards):
    return "\n\n".join(
        r"\paragraph{" + esc(card["title"]) + ".} " + esc(card["card"])
        for operator, card in cards.items() if not operator.startswith("_")) + "\n"


def number(value):
    return f"{value:+.3f}"


def fingerprint(path):
    with path.open("rb") as source:
        return dict(path=str(path.relative_to(ROOT)), sha256=hashlib.file_digest(source, "sha256").hexdigest())


def table(rows, caption, label, columns, header):
    return "\n".join([r"\begin{table}[t]\centering\scriptsize", r"\caption{"+caption+"}",
                       r"\label{"+label+"}", r"\begin{tabular}{@{}"+columns+r"@{}}\toprule",
                       header+r" \\ \midrule", *[" & ".join(row)+r" \\" for row in rows],
                       r"\bottomrule\end{tabular}", r"\end{table}", ""])


def shortest_horizons(path):
    shortest = {}
    with gzip.open(path, "rt") as source:
        for line in source:
            row = json.loads(line)
            if (row["kind"] == "3" and row.get("sub") == "1a" and row["set"] == "bench-basic"
                    and row["cond"] == "v1" and row.get("in_band") and row.get("scored_var")
                    and row.get("arm_kind") == "real"):
                group = (row["world"], row["t1"], row["op"])
                shortest[group] = min(shortest.get(group, row["h"]), row["h"])
    return shortest


def split_capture_channels(rows, shortest):
    groups = {name: {} for name in CHANNEL_LABELS}
    for key, value in rows.items():
        if key[3] != "real":
            continue
        if key[-1] < shortest[key[:3]]:
            raise ValueError(f"Target precedes its declared earliest horizon: {key}")
        if key[5] not in DIRECT.get(key[2], set()):
            group = "other_channels"
        else:
            group = "visible_short" if key[-1] == shortest[key[:3]] else "visible_long"
        groups[group][key] = value
    return groups


def write_channel_table(channels):
    rows = []
    for group, value in channels.items():
        for operator, part in value["by_operator"].items():
            rows.append([CHANNEL_LABELS[group], esc(operator), str(part["n_cells"]),
                         str(part["n_worlds"]), number(part["base"])])
    (ROOT/"paper/tables/direct_capture.tex").write_text(table(rows,
        "Flash-Next real-arm CAPTURE by channel, relative horizon, and operator in Omni-ID. "
        "Earliest/later horizons are defined within each world, anchor, and operator; "
        "other channels include all eligible horizons and are not necessarily unobservable. "
        "The main score-variable and horizon masks are unchanged; groups are not composition-matched.",
        "tab:directcapture", "llrrr", "Group & Operator & Cells & Worlds & Capture"))


def update_channels():
    path = RUNS/"v3-bench-br/scored_raw_tk.jsonl.gz"
    result_path = OUT/"results.json"
    result = json.loads(result_path.read_text())
    source = fingerprint(path)
    if source["sha256"] != result["provenance"]["main"]["sha256"]:
        raise ValueError("Main scored source changed; rebuild the full analysis before updating channels")
    rows = read(path, ("3",))[("bench-basic", "v1", "3")]
    groups = split_capture_channels(rows, shortest_horizons(path))
    result["channel_capture"] = {name: paired_capture(group, group) for name, group in groups.items()}
    result_path.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    write_channel_table(result["channel_capture"])
    make_figures(result, intervention_only=True)
    print(json.dumps({name: {field: value[field] for field in ("base", "n_cells", "n_operators")}
                      for name, value in result["channel_capture"].items()}, indent=2))


def make_figures(result, intervention_only=False):
    plt = _mpl(academic=True)
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    teal, rust = "#367E83", "#B37553"
    figure, axis = plt.subplots(figsize=(3.1, 1.55))
    axis.set_position([.43, .26, .43, .70])
    axis.set_xlim(-.72, .36)
    axis.set_ylim(-.55, 2.55)
    axis.set_xticks([-.6, -.3, 0, .3])
    axis.set_yticks([2, 1, 0], ["Visible channel\n(earliest horizon)",
                                    "Visible channel\n(later horizons)", "Other channels\n(all horizons)"])
    axis.set_xlabel("Capture", labelpad=2)
    axis.tick_params(axis="y", length=0, pad=5)
    axis.tick_params(axis="x", length=2, pad=2)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.axvline(0, color="black", linewidth=.6)
    axis.set_axisbelow(True)
    axis.grid(axis="x", color="#CCCCCC", linewidth=.5)
    for name, position in (("visible_short", 2), ("visible_long", 1), ("other_channels", 0)):
        value = result["channel_capture"][name]
        axis.barh(position, value["base"], color=teal, height=.32, zorder=3)
        axis.text(1.06, position+.12, number(value["base"]),
                  transform=axis.get_yaxis_transform(), ha="left", va="center", fontsize=9)
        axis.text(1.06, position-.23, f"n={value['n_cells']:,}",
                  transform=axis.get_yaxis_transform(), ha="left", va="center", fontsize=7, color="#555555")
    _save(figure, ROOT / "paper/figures/intervention_channels.pdf")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(3.4, 1.6))
    axis.set_position([.30, .21, .56, .76])
    tags = list(LABELS)
    changes = np.array([[result["mechanism"][tag][setting]["real"]["delta"]
                         for setting, label in SETTINGS] for tag in tags])
    limit = max(.25, math.ceil(float(np.max(np.abs(changes)))*20)/20)
    palette = LinearSegmentedColormap.from_list("capture_change", [rust, "#FAFAFA", teal])
    heatmap = axis.imshow(changes, cmap=palette, norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit),
                             aspect="auto", interpolation="nearest")
    axis.set_xticks(range(3), ["Omni-ID", "Ego", "OOD"])
    axis.set_yticks(range(3), LABELS.values())
    axis.tick_params(length=0, pad=4)
    axis.spines[:].set_visible(False)
    for row, tag in enumerate(tags):
        for column, (setting, label) in enumerate(SETTINGS):
            paired = result["mechanism"][tag][setting]["real"]
            value = paired["delta"]
            text_color = "black"
            axis.text(column, row-.13, "0.000" if round(value, 3) == 0 else number(value),
                         ha="center", va="center", fontsize=9, color=text_color)
            axis.text(column, row+.23, f"n={paired['n_cells']}",
                         ha="center", va="center", fontsize=7, color=text_color)
    color_axis = figure.add_axes([.90, .21, .018, .76])
    colorbar = figure.colorbar(heatmap, cax=color_axis, ticks=[-limit, 0, limit])
    colorbar.ax.tick_params(labelsize=7, width=.4, length=2, pad=2)
    colorbar.outline.set_linewidth(.4)
    _save(figure, ROOT / "paper/figures/mechanism_cards.pdf")
    plt.close(figure)
    if intervention_only:
        return
    make_information_figure(result)


def make_information_figure(result):
    plt = _mpl(academic=True)
    from matplotlib.ticker import FuncFormatter

    teal, rust = "#367E83", "#B37553"
    figure, axes = plt.subplots(2, 1, figsize=(3.5, 1.3))
    axes[0].set_position([.30, .55, .67, .38])
    axes[1].set_position([.30, .11, .67, .19])
    axes[0].set_title(r"(a) Quantity: $\Delta$ skill", loc="left", pad=2)
    axes[1].set_title(r"(b) Event: $\Delta$ Brier gain", loc="left", pad=2)
    axes[0].set_xlim(-.19, .14)
    axes[0].set_xticks([-.15, 0, .10])
    axes[1].set_xlim(-.004, .064)
    axes[1].set_xticks([0, .03, .06])
    specifications = [(axes[0], "qnt_ablations", teal, [
        ("k1", "One frame"), ("primer", "Rules primer"),
        ("oracle", "Future narrative"), ("belief", "Study notes")]),
        (axes[1], "event_ablations", rust, [("belief", "Study notes"), ("base_rate", "Base rate shown")])]
    for axis, group, shade, entries in specifications:
        axis.set_ylim(len(entries)-.5, -.5)
        axis.set_yticks(range(len(entries)), [label for name, label in entries])
        axis.tick_params(axis="y", length=0, pad=4, labelsize=9)
        axis.tick_params(axis="x", length=2, pad=2)
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.axvline(0, color="#555555", linewidth=.6, linestyle="--")
        axis.set_axisbelow(True)
        axis.grid(axis="x", color="#CCCCCC", linewidth=.5)
        axis.xaxis.set_major_formatter(FuncFormatter(lambda value, position: "0" if value == 0 else f"{value:.2f}"))
        for position, (name, label) in enumerate(entries):
            value = result[group][name]
            low, high = value["ci95"]
            axis.plot([low, high], [position, position], color=shade, linewidth=1.1, zorder=3)
            axis.plot([low, high], [position, position], linestyle="none", marker="|",
                      color=shade, markersize=5, markeredgewidth=.8, zorder=3)
            axis.scatter(value["delta"], position, color=shade, s=18, zorder=4)
    _save(figure, ROOT / "paper/figures/information_ablations.pdf")
    plt.close(figure)


def write_hidden_details():
    scores = {tag: load(tag) for name, tag, index in LINES}
    for group_name, targets in (("traits", TRAITS), ("rules", PARAMS)):
        caption = ("Pairwise ranking accuracy by personality trait." if group_name == "traits"
                   else "Classification accuracy by world rule.")
        target_labels = {"param_disasters": "Disaster frequency", "param_foodbox": "Growth cost",
                         "param_sciencebox": "Research cost"}
        rows = []
        for name, tag, index in LINES:
            values = []
            for target in targets:
                values.append(" / ".join(f"{scores[tag]['2'][setting+'|'+target]['acc']['mean']:.2f}" for setting, label in SETTINGS))
            rows.append([name, *values])
        (ROOT/f"paper/tables/{group_name}_detail.tex").write_text(table(rows,
            caption + " Triples are Omni-ID / Ego / Omni-OOD, using the main-table samples and failure penalties.",
            "tab:"+group_name+"detail", "l"+"c"*len(targets), "Model & "+" & ".join(
                esc(target_labels.get(target, target.replace("trait_", ""))) for target in targets)))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    paths = dict(main=RUNS/"v3-bench-br/scored_raw_tk.jsonl.gz",
                 old=RUNS/"v3-bench/scored_raw_tk_abl60.jsonl.gz",
                 belief=RUNS/"v3-abl-belief/scored_raw_tk.jsonl.gz",
                 nothink=RUNS/"v3-bench/scored_raw_nt.jsonl.gz",
                 base_rate=RUNS/"v3-abl-baserate/scored_raw_tk.jsonl.gz")
    data = {name: read(path) for name, path in paths.items()}
    baseline = data["main"][("bench-basic", "v1", "1a")]
    qnt = {name: paired_qnt(baseline, data["main"][("bench-basic", name, "1a")])
           for name in ("k1", "oracle", "primer")}
    qnt["belief"] = paired_qnt(data["old"][("bench-basic", "v1", "1a")], data["belief"][("bench-basic", "belief", "1a")])
    qnt["nothink"] = paired_qnt(data["old"][("bench-basic", "v1", "1a")], data["nothink"][("bench-basic", "v1", "1a")])
    event = {name: paired_event(data["old"][("bench-basic", "v1", "1b")], data[name][("bench-basic", cond, "1b")])
             for name, cond in (("belief", "belief"), ("base_rate", "v1br"))}
    mechanism = {}
    for tag in LABELS:
        paths[f"mechanism_{tag}"] = RUNS/f"v3-int-know/scored_raw_{tag}.jsonl.gz"
        paths[f"mechanism_base_{tag}"] = paths["main"] if tag == "tk" else RUNS/f"v3-bench-br/scored_raw_{tag}_eb60.jsonl.gz"
        base = data["main"] if tag == "tk" else read(paths[f"mechanism_base_{tag}"], ("3",))
        card = read(paths[f"mechanism_{tag}"], ("3",), reference=base)
        mechanism[tag] = {}
        for setting, label in SETTINGS:
            split, cond = setting.split("|")
            card_cond = "k1mech" if cond == "v1" else "k1mech_ego"
            mechanism[tag][setting] = {arm: paired_capture(base[(split, cond, "3")], card[(split, card_cond, "3")], arm)
                                       for arm in ("real", "placebo")}
    study_capture = {arm: paired_capture(data["old"][("bench-basic", "v1", "3")],
                                         data["belief"][("bench-basic", "belief", "3")], arm)
                     for arm in ("real", "placebo")}
    main_rows = data["main"][("bench-basic", "v1", "3")]
    channels = split_capture_channels(main_rows, shortest_horizons(paths["main"]))
    direct = channels["visible_short"]
    other = channels["visible_long"] | channels["other_channels"]
    result = dict(qnt_ablations=qnt, event_ablations=event, mechanism=mechanism, study_capture=study_capture,
                  direct_capture={name: paired_capture(rows, rows) for name, rows in (("direct", direct), ("other", other))},
                  channel_capture={name: paired_capture(rows, rows) for name, rows in channels.items()},
                  provenance={name: fingerprint(path) for name, path in paths.items()},
                  protocol="Same valid cells per paired comparison; original QNT, EVT and CAPTURE aggregation. E-B uses each model's main-run control prediction in both conditions. No new inference.")
    (OUT/"results.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    make_figures(result)
    rows = []
    for arm, value in study_capture.items():
        interval = value["ci95"]
        rows.append([arm, str(value["n_cells"]), str(value["n_worlds"]), str(value["n_operators"]),
                     number(value["base"]), number(value["variant"]), number(value["delta"]),
                     f"[{number(interval[0])}, {number(interval[1])}]" if interval else "--"])
    (ROOT/"paper/tables/study_capture.tex").write_text(table(rows,
        "Study notes: paired CAPTURE on each arm's common valid moved cells in Omni-ID. Both arm and no-intervention predictions are note-conditioned. Intervals use paired world Bayesian bootstrap weights.",
        "tab:studycapture", "lrrrrrrl", r"Arm & Cells & Worlds & Ops & Base & Notes & Change & 95\% interval"))
    write_channel_table(result["channel_capture"])
    cards = json.loads((ROOT/"bench/v2/mechanism_cards.json").read_text())
    (ROOT/"paper/tables/mechanism_card_text.tex").write_text(render_mechanism_cards(cards))
    rows = []
    for name, value in qnt.items():
        rows.append([esc(ABLATION_LABELS[name]), r"\QNT", str(value["n_worlds"]), str(value["n_cells"]), number(value["base"]),
                     number(value["variant"]), number(value["delta"]), f"[{number(value['ci95'][0])}, {number(value['ci95'][1])}]"])
    for name, value in event.items():
        rows.append([esc(ABLATION_LABELS[name]), r"\EVT", str(value["n_worlds"]), str(value["n_cells"]), number(value["base"]),
                     number(value["variant"]), number(value["delta"]), f"[{number(value['ci95'][0])}, {number(value['ci95'][1])}]"])
    (ROOT/"paper/tables/paired_ablations.tex").write_text(table(rows,
        "Paired information ablations of Flash-Next in Omni-ID. Each row uses its own common valid cells; intervals are paired world bootstraps. Event comparisons here use a no-base-rate baseline.",
        "tab:abl", "llrrrrrl", r"Condition & Task & Worlds & Cells & Base & Added/ablated & Change & 95\% interval"))
    write_mechanism_table(mechanism)
    rows = []
    for tag, settings in mechanism.items():
        for setting, arms in settings.items():
            for operator, value in arms["real"]["by_operator"].items():
                rows.append([LABELS[tag], dict(SETTINGS)[setting], esc(operator), str(value["n_cells"]),
                             number(value["base"]), number(value["variant"])])
    grouped = collections.defaultdict(list)
    for row in rows:
        grouped[row[0]].append(row[1:])
    (ROOT/"paper/tables/mechanism_operators.tex").write_text("\n".join(table(part,
        f"{label}: paired real-arm CAPTURE by operator. Operator values are clipped exactly as in the main table.",
        "tab:mechops"+str(index), "llrrr", "Setting & Operator & Cells & Base & Card")
        for index, (label, part) in enumerate(grouped.items())))
    write_hidden_details()
    macros = {"studyPairedBase": number(qnt["belief"]["base"]), "studyPairedScore": number(qnt["belief"]["variant"]),
              "studyPairedGain": number(qnt["belief"]["delta"]),
              "studyPairedInterval": f"[{number(qnt['belief']['ci95'][0])}, {number(qnt['belief']['ci95'][1])}]",
              "directCapture": number(result["direct_capture"]["direct"]["base"]),
              "otherCapture": number(result["direct_capture"]["other"]["base"])}
    for tag, prefix in (("tk", "flash"), ("q38-27b", "qwen"), ("gemma4-31b", "gemma")):
        value = mechanism[tag]["bench-basic|v1"]["real"]
        macros[prefix+"CardBase"] = number(value["base"])
        macros[prefix+"CardScore"] = number(value["variant"])
    (ROOT/"paper/analysis_numbers.tex").write_text("% Generated by paper/build_results_analysis.py.\n"+
        "\n".join("\\newcommand{\\"+name+"}{"+value+"}" for name, value in macros.items())+"\n")
    print(json.dumps({"qnt": qnt, "event": event, "mechanism": mechanism, "study_capture": study_capture,
                      "direct": result["direct_capture"]}, indent=1), flush=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channels-only", action="store_true", help="Update only Figure 3 and its channel breakdown")
    if parser.parse_args().channels_only:
        update_channels()
    else:
        main()
