"""Independent execution replication of frozen plans, queued after validation."""
import argparse
import concurrent.futures as futures
import fcntl
import hashlib
import itertools
import json
import os
from pathlib import Path
import select
import statistics
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from bench.agentplay.probes import usability_tonight as base
from bench.agentplay.probes.analyse_usability_tonight import estimate

SOURCE = ROOT / "bench/agentplay/games/usability_extension_20260924"
OUT = ROOT / "bench/agentplay/games/usability_replication_20260925"
SCRIPT = Path(__file__).resolve()
BASE_HASH = hashlib.sha256(Path(base.__file__).read_bytes()).hexdigest()
CONDITIONS = base.CONDITIONS


def configure():
    base.OUT = OUT
    base.__file__ = str(SCRIPT)


def wait_for_validation():
    if (SOURCE / "VERIFIED_DONE.json").exists():
        return
    pid = json.loads((SOURCE / "supervisor.json").read_text())["pid"]
    try:
        descriptor = os.pidfd_open(pid)
    except ProcessLookupError:
        descriptor = None
    if descriptor is not None:
        try:
            select.select([descriptor], [], [])
        finally:
            os.close(descriptor)
    if not (SOURCE / "VERIFIED_DONE.json").exists():
        raise RuntimeError("Preceding validation stopped without verified completion")


def freeze():
    source = SOURCE / "validation36"
    report = json.loads((source / "analysis.json").read_text())
    manifest = json.loads((source / "manifest.json").read_text())
    assert report["n"] == len(manifest["anchors"]) == 36
    for category in ("anchors", "plans"):
        directory = OUT / category
        directory.mkdir(parents=True, exist_ok=True)
        for item in (source / category).iterdir():
            target = directory / item.name
            if not target.exists():
                target.symlink_to(item.resolve(), target_is_directory=item.is_dir())
    prior = {model: report["utility"][model + "_vs_self"] for model in base.MODELS}
    signals = [model for model, result in prior.items() if result["bootstrap95"][0] > 0]
    reason = ("Positive individual intervals require independent execution replication, not a new positive-selection search."
              if signals else "No clear individual WM advantage; test execution variance and out-of-sample forecast selection.")
    base.write(OUT / "preceding_analysis.json", {"at": base.now(), "utility": prior,
               "headroom": report["headroom"], "followup_reason": reason,
               "caveat": "Individual intervals do not control four-model multiple comparisons."})
    path = OUT / "manifest.json"
    frozen = {"created": base.now(), "anchors": manifest["anchors"], "conditions": CONDITIONS,
              "execution_turns": 20, "repetitions": 2, "agent": "flash-next",
              "source": str(source), "new_worlds": False,
              "purpose": "Frozen-plan execution replication; not independent strategy-generation replication",
              "primary": "Each WM versus self-review on the two new executions; all four reported",
              "secondary": "Four-execution pooled utility, old forecast selection on new outcomes, local forecast stability",
              "paid_api_calls": 0, "code_sha256": hashlib.sha256(SCRIPT.read_bytes()).hexdigest(),
              "base_code_sha256": BASE_HASH}
    if path.exists():
        existing = json.loads(path.read_text())
        assert existing["code_sha256"] == frozen["code_sha256"]
        assert existing["base_code_sha256"] == BASE_HASH
    else:
        base.write(path, frozen)
    return manifest["anchors"]


def prepare(anchor):
    plans = {condition: json.loads((OUT / "plans" / anchor["id"] / (condition + ".json")).read_text())
             for condition in CONDITIONS}
    with futures.ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda model: base.predict_pool(anchor, model, plans), ("flash", "kimi")))
    base.write(OUT / "ready" / (anchor["id"] + ".json"), {"at": base.now(), "repeat_forecasts_before_new_execution": True})
    return anchor


def score_pair(directory, anchor, condition):
    scores = []
    for repeat in (0, 1):
        path = directory / "branches" / anchor / condition / str(repeat) / "summary.json"
        row = json.loads(path.read_text())
        assert row["n_turns"] == 20 or row.get("game_over")
        scores.append(row["final"]["score"])
    return scores


def predicted(directory, anchor, model):
    row = json.loads((directory / "forecasts" / anchor / (model + ".json")).read_text())
    return {row["order"][item["plan_index"]]: item["score_p10_p50_p90"][1] for item in row["forecasts"]}


def analyse():
    source = SOURCE / "validation36"
    anchors = [item["id"] for item in json.loads((OUT / "manifest.json").read_text())["anchors"]]
    old = {anchor: {condition: score_pair(source, anchor, condition) for condition in CONDITIONS} for anchor in anchors}
    new = {anchor: {condition: score_pair(OUT, anchor, condition) for condition in CONDITIONS} for anchor in anchors}
    report = {"n_worlds": len(anchors), "scope": "Same frozen plans, independent agent executions; not new worlds or fresh advice",
              "replication_utility": {}, "pooled_four_repeats": {}, "old_forecasts_on_new_execution": {},
              "forecast_stability": {}}
    for condition in CONDITIONS[1:]:
        for control in ("initial", "self"):
            if condition == control:
                continue
            key = condition + "_vs_" + control
            report["replication_utility"][key] = estimate([
                statistics.mean(new[anchor][condition]) - statistics.mean(new[anchor][control]) for anchor in anchors])
            report["pooled_four_repeats"][key] = estimate([
                statistics.mean(old[anchor][condition] + new[anchor][condition]) -
                statistics.mean(old[anchor][control] + new[anchor][control]) for anchor in anchors])
    headroom = []
    for anchor in anchors:
        heldout = []
        for training, testing in ((old, new), (new, old)):
            means = {condition: statistics.mean(training[anchor][condition]) for condition in CONDITIONS}
            best = max(means.values())
            selected = [condition for condition in CONDITIONS if means[condition] == best]
            heldout.append(statistics.mean(statistics.mean(testing[anchor][condition]) for condition in selected)
                           - statistics.mean(testing[anchor]["initial"]))
        headroom.append(statistics.mean(heldout))
    report["two_train_two_test_selector_gain"] = estimate(headroom)
    for model in base.MODELS:
        gains, directions, delta_errors, stable = [], [], [], []
        for anchor in anchors:
            prediction = predicted(source, anchor, model)
            actual = {condition: statistics.mean(new[anchor][condition]) for condition in CONDITIONS}
            best = max(prediction.values())
            selected = [condition for condition in CONDITIONS if prediction[condition] == best]
            gains.append(statistics.mean(actual[condition] for condition in selected) - actual["initial"])
            pairs = list(itertools.combinations(CONDITIONS, 2))
            delta_errors.append(statistics.mean(abs(prediction[first] - prediction[second] - actual[first] + actual[second]) for first, second in pairs))
            unequal = [(first, second) for first, second in pairs if actual[first] != actual[second]]
            if unequal:
                directions.append(statistics.mean((prediction[first] - prediction[second]) * (actual[first] - actual[second]) > 0 for first, second in unequal))
            if model in ("flash", "kimi"):
                repeat = predicted(OUT, anchor, model)
                stable.append(statistics.mean(abs(prediction[condition] - repeat[condition]) for condition in CONDITIONS))
        report["old_forecasts_on_new_execution"][model] = {
            "selected_gain": estimate(gains), "direction": estimate(directions), "pairwise_delta_mae": estimate(delta_errors)}
        if stable:
            report["forecast_stability"][model] = {"repeat_p50_absolute_gap": estimate(stable)}
    base.write(OUT / "analysis.json", report)
    print(json.dumps(report), flush=True)


def run():
    configure()
    anchors = freeze()
    pending = [json.loads((OUT / "anchors" / (item["id"] + ".json")).read_text()) for item in anchors]
    for sweep in range(2):
        failures, completed = [], []
        print(base.now(), "sweep", sweep + 1, flush=True)
        with futures.ThreadPoolExecutor(max_workers=48) as execution, futures.ThreadPoolExecutor(max_workers=3) as preparation:
            ready = {preparation.submit(prepare, anchor): anchor for anchor in pending}
            jobs = []
            for future in futures.as_completed(ready):
                anchor = ready[future]
                try:
                    future.result()
                except Exception as error:
                    failures.append({"anchor": anchor["id"], "error": str(error)})
                    continue
                for index, condition in enumerate(CONDITIONS):
                    for repeat in (0, 1):
                        jobs.append(execution.submit(base.run_branch, (anchor, condition, repeat, base.ENDPOINTS[(index + repeat) % 2])))
            for future in futures.as_completed(jobs):
                result = future.result()
                completed.append(result)
                base.append(OUT / "completed.jsonl", result)
        valid = len(completed) == 432 and all(item["valid"] for item in completed) and not failures
        base.write(OUT / "exit.json", {"at": base.now(), "valid": valid, "branches": len(completed), "failures": failures})
        if valid:
            analyse()
            base.write(OUT / "VERIFIED_DONE.json", {"at": base.now(), "branches": 432})
            return
    raise RuntimeError("Replication incomplete after bounded retries")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("watch", "fork", "analyse"))
    parser.add_argument("--anchor")
    parser.add_argument("--condition")
    parser.add_argument("--repeat", type=int)
    parser.add_argument("--endpoint")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.mode == "fork":
        configure()
        manifest = json.loads((OUT / "manifest.json").read_text())
        assert manifest["code_sha256"] == hashlib.sha256(SCRIPT.read_bytes()).hexdigest()
        assert manifest["base_code_sha256"] == BASE_HASH
        anchor = json.loads((OUT / "anchors" / (args.anchor + ".json")).read_text())
        base.fork(anchor, args.condition, args.repeat, args.endpoint, 20)
    elif args.mode == "analyse":
        analyse()
    else:
        with (OUT / "supervisor.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                print(base.now(), "waiting for validated predecessor", flush=True)
                wait_for_validation()
                print(base.now(), "predecessor complete; analysing and starting replication", flush=True)
                run()
            except Exception as error:
                base.write(OUT / "SUPERVISOR_FAILED.json", {"at": base.now(), "error": str(error)})
                raise
