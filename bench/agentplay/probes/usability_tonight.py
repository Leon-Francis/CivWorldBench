"""Small, frozen decision-point experiment; forecasts never receive branch outcomes."""
from __future__ import annotations

import argparse
import concurrent.futures as futures
import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import tempfile
import threading
import time

ROOT = Path(__file__).resolve().parents[3]
TEMP = ROOT / "bench/agentplay/runtime_tmp"
TEMP.mkdir(exist_ok=True)
for variable in ("TMPDIR", "TMP", "TEMP"):
    os.environ[variable] = str(TEMP)
tempfile.tempdir = str(TEMP)
sys.path.insert(0, str(ROOT))

from bench.agentplay import run_game
from bench.agentplay.agent import Agent
from bench.agentplay import deliberative as dialogue
from bench.agentplay.deliberative import DeliberativeAgent, PLAN, checked_json, normalize_proposals
from bench.agentplay.run_deliberative_game import clean_siege
from bench.agentplay.wm import History, WorldModel, SYSTEM

OUT = ROOT / "bench/agentplay/games/usability_tonight_20260924"
SOURCE = ROOT / "bench/agentplay/games/deliberative150_r2"
MODELS = {"flash": "flash-next", "kimi": "kimi-k3", "sol": "gpt-5.6-sol", "opus": "claude-opus-5"}
ENDPOINTS = ["http://localhost:8011/v1", "http://127.0.0.1:8011/v1"]
CONDITIONS = ["initial", "self", *MODELS]
API_SLOTS = threading.Semaphore(4)
LOG_LOCK = threading.Lock()


def now():
    return datetime.datetime.now().astimezone().isoformat()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".new")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def append(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with LOG_LOCK, path.open("a") as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + "\n")


def clean_view(*args, **kwargs):
    view = ORIGINAL_VIEW(*args, **kwargs)
    view["rules"] = args[3] if len(args) > 3 else kwargs["rules"]
    old = "\n".join(view.get("siege") or [])
    view["siege"] = clean_siege(view.get("siege") or [])
    if old:
        view["panel"] = view["panel"].replace(old, "\n".join(view["siege"]))
    DeliberativeAgent.current_view = view
    return view


ORIGINAL_VIEW = run_game.build_view


def freeze():
    path = OUT / "manifest.json"
    if path.exists():
        return json.loads(path.read_text())
    manifest = json.loads((SOURCE / "manifest.json").read_text())
    worlds = list(manifest["worlds"])
    random.Random(20260924).shuffle(worlds)
    anchors = []
    for index, world in enumerate(worlds[:12]):
        source = SOURCE / "none" / (world.rsplit("-", 1)[1] + "_e0")
        sessions = [json.loads(line) for line in (source / "planning.jsonl").read_text().splitlines()]
        target = [80, 120, 170][index // 4]
        session = min((entry for entry in sessions if 70 <= entry["turn"] <= 185),
                      key=lambda entry: (abs(entry["turn"] - target), entry["turn"]))
        turn = session["turn"]
        records = [json.loads(line) for line in (source / "turns.jsonl").read_text().splitlines()]
        previous = next(row for row in records if row["turn"] == turn - 1)
        initial = session["rounds"][0]["plans"][0]
        question = PLAN + '\nPropose one to three genuinely useful plans under your own judgment. One is enough; do not invent alternatives to fill slots. Reply {"plans":[...]}.'
        user = session["initial_calls"][0]["messages"][1]["content"]
        assert user.endswith("\n" + question), "unrecognized historical planning prompt"
        context = user[:-len("\n" + question)]
        anchor = {"id": f"a{index:02d}_{world.rsplit('-', 1)[1]}_t{turn}", "world": world,
                  "source": str(source), "turn": turn, "target_turn": target,
                  "initial_candidate_count": len(session["rounds"][0]["plans"]),
                  "source_revision": session.get("code_revision", "original"),
                  "save": previous["save"], "save_sha256": hashlib.sha256(Path(previous["save"]).read_bytes()).hexdigest(),
                  "note": previous["note"], "feedback": previous["feedback_next"], "standing": previous["standing"],
                  "initial_plan": initial, "context": context,
                  "system": session["initial_calls"][0]["messages"][0]["content"]}
        write(OUT / "anchors" / (anchor["id"] + ".json"), anchor)
        anchors.append({key: anchor[key] for key in ("id", "world", "turn", "target_turn", "save_sha256")})
    result = {"created": now(), "sampling_seed": 20260924, "sampling": "12 shuffled world IDs; four each near T80/T120/T170; no score selection",
              "anchors": anchors, "conditions": CONDITIONS, "models": MODELS, "agent": "flash-next",
              "repetitions": 2, "execution_turns": 20, "goal_turn": 210,
              "execution": "fixed adopted plan, free tactics, no additional advisory/replanning calls",
              "randomness": "same engine checkpoint; independent stochastic agent calls, no claim of independent engine seeds",
              "budget_usd": 600, "batch_cap_usd": 90,
              "billing_reservation_rates_per_million": {"input": 20, "output": 100},
              "pricing_note": "Conservative reservation ceilings, not verified provider rates; historical gateway quote 6.25 input/write and 25 output. No cache discount assumed.",
              "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    write(path, result)
    return result


class BudgetWM(WorldModel):
    def __init__(self, model, anchor_id, stage):
        endpoint = "http://127.0.0.1:8011/v1" if model == "kimi-k3" else ENDPOINTS[0]
        super().__init__(endpoint, model, "medium", timeout=600)
        self.anchor_id, self.stage = anchor_id, stage

    def call(self, prompt, max_tokens):
        if self.api == "local":
            result = super().call(prompt, max_tokens)
            append(OUT / "calls.jsonl", {"anchor": self.anchor_id, "stage": self.stage,
                   "model": self.model, "prompt": prompt, "result": result})
            return result
        import bench.run_infer_api as api
        api.EFFORT = "medium"
        path, body = api.build_request(self.api, self.model, {"prompt": prompt, "system": SYSTEM, "max_tokens": min(max_tokens, 8192)})
        output_limit = min(max_tokens, 8192) + api.THINK_BUDGET
        input_bound = len((SYSTEM + prompt).encode()) + 4096
        reservation = input_bound * 20 / 1e6 + output_limit * 100 / 1e6
        group = int(self.anchor_id[1:3]) // 4
        batch = f"{self.model}:{self.stage}:{group}"
        for attempt in range(2):
            with API_SLOTS:
                ledger = OUT / "budget.json"
                with (OUT / "budget.lock").open("a") as lock:
                    fcntl.flock(lock, fcntl.LOCK_EX)
                    state = json.loads(ledger.read_text()) if ledger.exists() else {"reserved_usd": 0, "batches": {}}
                    if state["reserved_usd"] + reservation > 600 or state["batches"].get(batch, 0) + reservation > 90:
                        raise RuntimeError(f"budget reservation limit: {batch}")
                    state["reserved_usd"] += reservation
                    state["batches"][batch] = state["batches"].get(batch, 0) + reservation
                    write(ledger, state)
                started = time.time()
                try:
                    response = self.client.post(path, json=body)
                    if response.status_code == 200:
                        parsed = api.parse_response(self.api, response.json())
                        result = {"answer": parsed["text"], "prompt_tokens": parsed["n_prompt"],
                                  "completion_tokens": parsed["n_gen"], "finish": parsed["finish"], "seconds": time.time() - started}
                        append(OUT / "calls.jsonl", {"anchor": self.anchor_id, "stage": self.stage,
                            "model": self.model, "batch": batch, "reservation_usd": reservation,
                            "prompt": prompt, "usage": response.json().get("usage"), "result": result})
                        return result
                    error = "http" + str(response.status_code)
                except Exception as exception:
                    error = type(exception).__name__
                append(OUT / "api_errors.jsonl", {"at": now(), "anchor": self.anchor_id,
                       "model": self.model, "stage": self.stage, "error": error, "reservation_usd": reservation})
        raise RuntimeError("API unavailable: " + error)


def local_json(anchor, question, validator, endpoint=ENDPOINTS[0]):
    agent = Agent(endpoint=endpoint, naive=True)
    return checked_json(lambda messages: agent.call(messages, max_tokens=16000),
                        [{"role": "system", "content": anchor["system"]},
                         {"role": "user", "content": anchor["context"] + "\n" + question}], validator)


def make_plan(anchor, condition):
    destination = OUT / "plans" / anchor["id"] / (condition + ".json")
    if destination.exists():
        return json.loads(destination.read_text())
    original = anchor["initial_plan"]
    if condition == "initial":
        result = {"plan": original, "rationale": "Original agent proposal, without additional review.", "advice": None, "calls": []}
    else:
        if condition == "self":
            def review(value):
                if not isinstance(value.get("review"), str) or not value["review"].strip():
                    raise ValueError("nonempty review required")
            advice, advice_calls = local_json(anchor,
                'Review this plan for feasibility and possible improvements. Do not produce numerical future forecasts. '
                'Reply {"review":"your analysis"}.\nPLAN:\n' + json.dumps(original), review)
        else:
            adviser = DeliberativeAgent.__new__(DeliberativeAgent)
            adviser.config = {"end": 210}
            adviser.memo = ""
            adviser.history = History(anchor["world"], Path(anchor["source"]), 60, 0)
            state = run_game.Savegame.load(Path(anchor["save"]))
            rules = run_game.load_rules("civ2civ3")
            adviser.current_view = clean_view(state, anchor["world"], 0, rules,
                {int(key): value for key, value in anchor["standing"].items()}, anchor["feedback"], [])
            adviser.forecaster = BudgetWM(MODELS[condition], anchor["id"], "advice")
            advice, advice_calls = adviser.forecast(anchor["context"], [original], anchor["turn"], 1)
        def revision(value):
            normalize_proposals(value)
            if len(value["plans"]) != 1 or not isinstance(value.get("rationale"), str) or not value["rationale"].strip():
                raise ValueError("one complete plan and nonempty rationale required")
        decision, calls = local_json(anchor, PLAN + '\nReview the original plan and this advice. Keep, edit or replace it under your judgment. '
            'Return {"plans":[ONE_COMPLETE_PLAN],"rationale":"decision and reasons"}. '
            'No mandatory changes. This is the final revision for this local evaluation.\nORIGINAL:\n' + json.dumps(original) +
            "\nADVICE:\n" + json.dumps(advice), revision)
        result = {"plan": decision["plans"][0], "rationale": decision["rationale"], "advice": advice,
                  "calls": calls, "advice_calls": advice_calls}
    result.update(anchor=anchor["id"], condition=condition, created=now())
    write(destination, result)
    return result


def predict_pool(anchor, condition, plans):
    destination = OUT / "forecasts" / anchor["id"] / (condition + ".json")
    if destination.exists():
        return json.loads(destination.read_text())
    order = list(CONDITIONS)
    random.Random(20260924 + int(anchor["id"][1:3])).shuffle(order)
    forecaster = BudgetWM(MODELS[condition], anchor["id"], "pool")
    results, calls = [], []
    for start in (0, 3):
        subset = order[start:start + 3]
        candidates = [{"plan_index": start + index, "plan": plans[label]["plan"]} for index, label in enumerate(subset)]
        def validate(value):
            items = value.get("forecasts")
            if not isinstance(items, list) or sorted(item["plan_index"] for item in items) != list(range(start, start + len(subset))):
                raise ValueError("one forecast per supplied plan index")
            for item in items:
                quantiles = item.get("score_p10_p50_p90")
                if not isinstance(quantiles, list) or len(quantiles) != 3 or any(type(number) not in (int, float) or not math.isfinite(number) or number < 0 for number in quantiles) or quantiles != sorted(quantiles):
                    raise ValueError("three ordered nonnegative finite score quantiles required")
        prompt = (anchor["context"] + "\nForecast the TOTAL SCORE at turn " + str(anchor["turn"] + 20) +
            ". Each candidate is executed for 20 turns by the same Flash-Next agent, with tactical freedom and "
            "no additional strategic consultation. No execution results are available. Account for execution uncertainty. "
            'Return {"forecasts":[{"plan_index":0,"score_p10_p50_p90":[0,0,0],"reason":"evidence"}]}. '
            "These labels reveal no author.\nCANDIDATES:\n" + json.dumps(candidates))
        value, trace = checked_json(lambda messages: forecaster.call("\n\n".join(message["content"] for message in messages), 6000),
                                    [{"role": "user", "content": prompt}], validate)
        results.extend(value["forecasts"])
        calls.extend(trace)
    result = {"anchor": anchor["id"], "model": condition, "order": order, "forecasts": results, "calls": calls, "created": now()}
    write(destination, result)
    return result


class FixedPlanAgent(DeliberativeAgent):
    def deliberate(self, context, turn):
        self.last_planning = turn
        self.replan = False


def fork(anchor, condition, repeat, endpoint, turns):
    directory = OUT / "branches" / anchor["id"] / condition / str(repeat)
    directory.mkdir(parents=True, exist_ok=True)
    plan = json.loads((OUT / "plans" / anchor["id"] / (condition + ".json")).read_text())
    assert hashlib.sha256(Path(anchor["save"]).read_bytes()).hexdigest() == anchor["save_sha256"]
    if not (directory / "planning.jsonl").exists():
        append(directory / "planning.jsonl", {"turn": anchor["turn"], "final_plan": plan["plan"],
            "rationale": plan["rationale"], "discussion": [{"advice": plan["advice"]}], "wm_memo": ""})
    DeliberativeAgent.config = {"out": str(directory), "world": anchor["world"], "start": anchor["turn"],
                              "ego": 0, "end": 210, "wm_model": "", "wm_endpoint": ""}
    run_game.Agent = FixedPlanAgent
    run_game.build_view = clean_view
    sys.argv = [sys.argv[0], "--world", anchor["world"], "--out", str(directory), "--start", str(anchor["turn"]),
                "--turns", str(turns), "--arm", "none", "--agent_naive", "1", "--endpoint", endpoint,
                "--init_save", anchor["save"], "--init_note", anchor["note"], "--from_game", anchor["source"]]
    run_game.main()
    summary = json.loads((directory / "summary.json").read_text())
    assert summary["n_turns"] == turns or summary.get("game_over"), "incomplete fork"


def run_branch(task):
    anchor, condition, repeat, endpoint = task
    directory = OUT / "branches" / anchor["id"] / condition / str(repeat)
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "summary.json").exists():
        summary = json.loads((directory / "summary.json").read_text())
        if summary["n_turns"] == 20 or summary.get("game_over"):
            return {"anchor": anchor["id"], "condition": condition, "repeat": repeat, "valid": True}
    for attempt in range(3):
        with (directory / "run.log").open("a") as stream:
            process = subprocess.Popen([str(ROOT / ".venv/bin/python"), __file__, "fork", "--anchor", anchor["id"],
                "--condition", condition, "--repeat", str(repeat), "--endpoint", endpoint], stdout=stream, stderr=subprocess.STDOUT)
            append(OUT / "processes.jsonl", {"pid": process.pid, "anchor": anchor["id"], "condition": condition,
                                            "repeat": repeat, "attempt": attempt + 1, "endpoint": endpoint})
            code = process.wait()
        if code == 0:
            return {"anchor": anchor["id"], "condition": condition, "repeat": repeat, "valid": True}
    return {"anchor": anchor["id"], "condition": condition, "repeat": repeat, "valid": False}


def prepare_anchor(anchor):
    print(now(), "prepare", anchor["id"], flush=True)
    plans = {}
    with futures.ThreadPoolExecutor(max_workers=6) as pool:
        jobs = {pool.submit(make_plan, anchor, condition): condition for condition in CONDITIONS}
        for future in futures.as_completed(jobs):
            plans[jobs[future]] = future.result()
    with futures.ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda condition: predict_pool(anchor, condition, plans), MODELS))
    write(OUT / "ready" / (anchor["id"] + ".json"), {"ready": now(), "forecasts_frozen_before_execution": True})
    print(now(), "ready", anchor["id"], flush=True)
    return anchor


def summarize():
    manifest = freeze()
    scores = {condition: {} for condition in CONDITIONS}
    for item in manifest["anchors"]:
        for condition in CONDITIONS:
            paths = [OUT / "branches" / item["id"] / condition / str(repeat) / "summary.json" for repeat in (0, 1)]
            if all(path.exists() for path in paths):
                rows = [json.loads(path.read_text()) for path in paths]
                if all(row["n_turns"] == 20 or row.get("game_over") for row in rows):
                    scores[condition][item["id"]] = [row["final"]["score"] for row in rows]
    complete = sorted(set.intersection(*(set(values) for values in scores.values())))
    result = {"updated": now(), "complete_anchors": complete, "n": len(complete), "scores": scores,
              "scope": "20-turn local diagnostic, not T210 end-to-end utility"}
    for condition in CONDITIONS[1:]:
        differences = [statistics.mean(scores[condition][anchor]) - statistics.mean(scores["initial"][anchor]) for anchor in complete]
        if differences:
            result[condition + "_vs_initial"] = {"mean": statistics.mean(differences),
                "se": statistics.stdev(differences) / len(differences)**0.5 if len(differences) > 1 else None,
                "wins": sum(value > 0 for value in differences)}
    write(OUT / "results.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["freeze", "run", "fork", "summarize"])
    parser.add_argument("--anchor")
    parser.add_argument("--condition", choices=CONDITIONS)
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--endpoint", default=ENDPOINTS[0])
    parser.add_argument("--turns", type=int, default=20)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = freeze()
    if args.mode == "freeze":
        print(json.dumps({key: value for key, value in manifest.items() if key != "anchors"}, indent=2))
        return
    if args.mode == "summarize":
        print(json.dumps(summarize(), indent=2))
        return
    if args.mode == "fork":
        anchor = json.loads((OUT / "anchors" / (args.anchor + ".json")).read_text())
        fork(anchor, args.condition, args.repeat, args.endpoint, args.turns)
        return
    import httpx
    for endpoint in ENDPOINTS:
        response = httpx.get(endpoint + "/models", timeout=15)
        response.raise_for_status()
    assert manifest["code_sha256"] == hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "source changed after freeze"
    anchors = [json.loads((OUT / "anchors" / (item["id"] + ".json")).read_text()) for item in manifest["anchors"]]
    failures, completed = [], []
    with futures.ThreadPoolExecutor(max_workers=48) as execution, futures.ThreadPoolExecutor(max_workers=3) as preparation:
        preparation_jobs = {preparation.submit(prepare_anchor, anchor): anchor for anchor in anchors}
        branch_jobs = []
        for job in futures.as_completed(preparation_jobs):
            anchor = preparation_jobs[job]
            try:
                job.result()
            except Exception as error:
                failure = {"anchor": anchor["id"], "stage": "prepare", "error": str(error)[:500]}
                failures.append(failure)
                append(OUT / "failures.jsonl", failure)
                continue
            for index, condition in enumerate(CONDITIONS):
                for repeat in (0, 1):
                    endpoint = ENDPOINTS[(index + repeat) % 2]
                    branch_jobs.append(execution.submit(run_branch, (anchor, condition, repeat, endpoint)))
        for job in futures.as_completed(branch_jobs):
            result = job.result()
            completed.append(result)
            append(OUT / "completed.jsonl", result)
            summarize()
    valid = len(completed) == 144 and all(item["valid"] for item in completed) and not failures
    write(OUT / "exit.json", {"finished": now(), "valid": valid, "branches": len(completed), "preparation_failures": failures})
    if valid:
        (OUT / "VERIFIED_DONE").write_text(now() + "\n")
    else:
        raise RuntimeError("Some anchors/branches remain incomplete; see failures and exit.json")


if __name__ == "__main__":
    main()
