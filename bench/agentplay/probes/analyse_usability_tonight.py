"""Anchor-paired local utility, shared-pool predictions and execution diagnostics."""
import itertools
import json
from pathlib import Path
import random
import statistics
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from bench.agentplay.probes.usability_tonight import OUT, CONDITIONS, MODELS, summarize, write


def estimate(values):
    if not values:
        return {"n": 0}
    rng = random.Random(20260924)
    samples = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(5000))
    return {"n": len(values), "mean": statistics.mean(values), "se": statistics.stdev(values) / len(values)**0.5 if len(values) > 1 else None,
            "bootstrap95": [samples[125], samples[4874]], "wins": sum(value > 0 for value in values)}


def main():
    result = summarize()
    anchors, scores = result["complete_anchors"], result["scores"]
    report = {"n": len(anchors), "anchors": anchors, "scope": "exploratory 20-turn diagnostic; four models do not identify independent L2/L3 predictive effects",
              "utility": {}, "forecasts": {}, "execution": {}, "headroom": {}}
    for condition in CONDITIONS[1:]:
        for control in ("initial", "self"):
            if condition == control:
                continue
            report["utility"][condition + "_vs_" + control] = estimate([
                statistics.mean(scores[condition][anchor]) - statistics.mean(scores[control][anchor]) for anchor in anchors])
    crossfit, noise = [], []
    for anchor in anchors:
        heldout = []
        for train, test in ((0, 1), (1, 0)):
            best = max(scores[condition][anchor][train] for condition in CONDITIONS)
            tied = [condition for condition in CONDITIONS if scores[condition][anchor][train] == best]
            heldout.append(statistics.mean(scores[condition][anchor][test] for condition in tied) - scores["initial"][anchor][test])
        crossfit.append(statistics.mean(heldout))
        noise.append(statistics.mean(abs(scores[condition][anchor][0] - scores[condition][anchor][1]) for condition in CONDITIONS))
    report["headroom"] = {"crossfit_empirical_selector_gain": estimate(crossfit), "same_plan_repeat_absolute_gap": estimate(noise),
                          "caveat": "two repeats give a noisy empirical selector, not a true oracle or proof of zero headroom"}
    for model in MODELS:
        absolute, differences, direction, selection = [], [], [], []
        for anchor in anchors:
            forecast = json.loads((OUT / "forecasts" / anchor / (model + ".json")).read_text())
            predicted = {forecast["order"][entry["plan_index"]]: entry["score_p10_p50_p90"][1] for entry in forecast["forecasts"]}
            actual = {condition: statistics.mean(scores[condition][anchor]) for condition in CONDITIONS}
            absolute.append(statistics.mean(abs(predicted[condition] - actual[condition]) for condition in CONDITIONS))
            pairs = list(itertools.combinations(CONDITIONS, 2))
            differences.append(statistics.mean(abs((predicted[first] - predicted[second]) - (actual[first] - actual[second])) for first, second in pairs))
            moved = [(first, second) for first, second in pairs if actual[first] != actual[second]]
            if moved:
                direction.append(statistics.mean((predicted[first] - predicted[second]) * (actual[first] - actual[second]) > 0 for first, second in moved))
            best = max(predicted.values())
            selection.append(statistics.mean(actual[condition] for condition in CONDITIONS if predicted[condition] == best) - actual["initial"])
        report["forecasts"][model] = {"absolute_p50_mae": estimate(absolute), "pairwise_delta_mae": estimate(differences),
            "pairwise_direction": estimate(direction), "prediction_selected_gain": estimate(selection),
            "caveat": "local pairwise diagnostics are not the benchmark L3 Capture or causal effects of individual commands"}
    for condition in CONDITIONS:
        rejected, failed_checks, uncheckable, text_changed = [], [], [], []
        for anchor in anchors:
            initial = json.loads((OUT / "plans" / anchor / "initial.json").read_text())["plan"]
            plan = json.loads((OUT / "plans" / anchor / (condition + ".json")).read_text())["plan"]
            text_changed.append(plan != initial)
            rows = []
            for repeat in (0, 1):
                rows.extend(json.loads(line) for line in (OUT / "branches" / anchor / condition / str(repeat) / "turns.jsonl").read_text().splitlines())
            rejected.append(sum(any(str(item).startswith(("REJECTED", "FAILED", "COULD NOT")) for item in row.get("feedback_next", [])) for row in rows) / max(1, len(rows)))
            checks = [check for row in rows for check in row.get("orders", {}).get("checkpoint_observations", [])]
            failed_checks.append(sum(check["status"] == "failed" for check in checks))
            uncheckable.append(sum(check["status"] == "not_machine_checkable" for check in checks))
        report["execution"][condition] = {"turns_with_rejected_orders": estimate(rejected),
            "failed_checkpoint_observations": estimate(failed_checks), "uncheckable_observations": estimate(uncheckable),
            "changed_plan_text": sum(text_changed), "caveat": "text change is not semantic adoption; checkpoint counts are repeated observations, not distinct failures"}
    write(OUT / "analysis.json", report)
    print(json.dumps({key: report[key] for key in ("n", "utility", "headroom")}, indent=2))


if __name__ == "__main__":
    main()
