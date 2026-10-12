"""Resumable gateway inference for frozen aligned CoT rows and Code backtests."""
from __future__ import annotations

import argparse
import asyncio
import collections
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import time

import httpx

from bench.cot.cot_spec import extract_code
from bench.cot.exec_code import backtest, check_program
from bench.cot.prepare_sol30 import atomic_json, read_rows, save_rows, sha
from bench.cot.run_cot_v31 import KEEP, key_of
from bench.parse_v2 import parse_rate
from bench import run_infer_api as api

ROOT = Path(__file__).resolve().parents[2]
MODEL = "gpt-5.6-sol"
UNIT = 1_050_000


class BudgetExceeded(RuntimeError):
    pass


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def charge(output):
    cached = min(output["n_prompt"], output.get("n_cached", 0))
    return ((output["n_prompt"] - cached) * 6.25 + cached * .5 + output["n_gen"] * 30) / UNIT


def reservation(body):
    # UTF-8 bytes bound text-token counts conservatively; also reserve the entire output cap.
    prompt_bound = sum(len(m["content"].encode()) + 256 for m in body["messages"])
    return (prompt_bound * 6.25 + body["max_completion_tokens"] * 30) / UNIT


async def do_row(q, generate, check=check_program, history=backtest):
    rec = {k: q[k] for k in KEEP if k in q}
    messages = [{"role": "system", "content": q.get("system") or api.SYSTEM},
                {"role": "user", "content": q["prompt"]}]
    if q["fmt"] != "code":
        out = await generate(q, messages, 0)
        rec.update({k: v for k, v in out.items() if k != "text"})
        rec.update(output=out["text"], parse_rate=parse_rate(q, out["text"]))
        return rec
    rounds, best, no_gain, final_ok = [], None, 0, False
    for index in range(4):
        out = await generate(q, messages, index)
        code = extract_code(out["text"])
        if not code:
            ok, feedback, details = False, "No ```python block with a `predict(world, rng)` function was found in your reply. Output the complete program in one ```python block.", {"exec_ok": False}
        else:
            ok, feedback, details = await asyncio.to_thread(check, code, q["world_json"], int(q["t1"]),
                                                          int(q["hmax"]), list(q["ev_civs"]))
            if ok:
                ok, feedback, bt = await asyncio.to_thread(history, code, q.get("backtest_world_json", q["world_json"]))
                details = {**details, "backtest": bt}
        rounds.append({**{k: v for k, v in out.items() if k != "text"}, "output": out["text"],
                       "ok": ok, "feedback": feedback, "check": details})
        bt = details.get("backtest") or {}
        runnable = bool(details.get("exec_ok")) and not details.get("n_problems") and bool(bt.get("exec_ok", True))
        if runnable:
            flagged, cells = bt.get("n_flagged", 0), bt.get("n_cells", 0)
            if best is None or flagged < best[0]:
                best, no_gain = (flagged, index), 0
            else:
                no_gain += 1
            if ok or flagged <= max(2, .15 * cells):
                final_ok = True
                if best is None or flagged <= best[0]:
                    best = (flagged, index)
                break
            if no_gain >= 2:
                break
        messages += [{"role": "assistant", "content": out["text"]}, {"role": "user", "content": feedback}]
    pick = best[1] if best is not None else len(rounds) - 1
    chosen = rounds[pick]
    rec.update(output=chosen["output"], think_text=chosen["think_text"], finish=chosen["finish"],
               parse_rate=0.0, rounds=rounds, n_rounds=len(rounds), final_ok=final_ok,
               picked_round=pick + 1, runnable=best is not None, bt_flagged=best[0] if best else None,
               bt_cells=(chosen["check"].get("backtest") or {}).get("n_cells"), model=MODEL,
               provider_model=chosen.get("provider_model"))
    for field in ("think_tok", "n_gen", "n_prompt", "n_cached", "n_cache_write"):
        rec[field] = sum(r.get(field, 0) for r in rounds)
    return rec


class JournalClient:
    def __init__(self, directory, client, budget, secret="", stop=None):
        self.directory, self.client, self.budget, self.secret = directory, client, budget, secret
        self.directory.mkdir(parents=True, exist_ok=True)
        self.entries = {str(p.relative_to(directory)): json.loads(p.read_text())
                        for p in directory.glob("*/attempt_*.json")}
        for name, entry in self.entries.items():
            if entry["state"] == "reserved":
                entry.update(state="uncertain", interrupted_before_response=True)
                atomic_json(directory / name, entry)
        self.lock = asyncio.Lock()
        self.stop = stop

    def accounting(self):
        complete = [e for e in self.entries.values() if e["state"] == "complete"]
        unknown = [e for e in self.entries.values() if e["state"] in ("reserved", "uncertain")]
        return {"completed_calls": len(complete), "uncertain_calls": len(unknown),
                "usage_cost_upper_usd": sum(e["cost_upper_usd"] for e in complete),
                "reserved_or_uncertain_usd": sum(e["reserved_usd"] for e in unknown),
                "committed_usd": sum(e["cost_upper_usd"] for e in complete) + sum(e["reserved_usd"] for e in unknown),
                "budget_usd": self.budget}

    async def generate(self, q, messages, round_index):
        endpoint, body = api.build_request("openai", MODEL, q)
        body["messages"] = messages
        request_hash = digest(body)
        directory = self.directory / request_hash
        directory.mkdir(exist_ok=True)
        cached = sorted(directory.glob("attempt_*.json"))
        for path in cached:
            entry = json.loads(path.read_text())
            if entry["state"] == "complete":
                return self.decode(entry, request_hash)
        atomic_json(directory / "request.json", {"endpoint": endpoint, "body": body, "sha256": request_hash,
                    "row_key": key_of(q), "world": q["world"], "t1": q["t1"], "round": round_index + 1})
        reserve = reservation(body)
        for attempt in range(6):
            async with self.lock:
                if self.stop is not None and self.stop.is_set():
                    raise RuntimeError("Run stopped after another worker failed")
                if self.accounting()["committed_usd"] + reserve > self.budget:
                    raise BudgetExceeded("Insufficient budget for the next request reservation")
                path = directory / f"attempt_{time.time_ns()}.json"
                entry = {"state": "reserved", "reserved_usd": reserve, "started_at": timestamp(),
                         "request_sha256": request_hash}
                atomic_json(path, entry)
                self.entries[str(path.relative_to(self.directory))] = entry
            try:
                response = await self.client.post(endpoint, json=body)
                if response.status_code == 200:
                    payload = response.json()
                    out = api.parse_response("openai", payload)
                    if not payload.get("usage") or out["n_prompt"] <= 0:
                        raise ValueError("Provider did not return usable token accounting")
                    entry.update(state="complete", response=payload, cost_upper_usd=charge(out), completed_at=timestamp())
                    atomic_json(path, entry)
                    return self.decode(entry, request_hash)
                retry = response.status_code in (408, 409, 429) or response.status_code >= 500
                # Explicit rejected requests are not billed. Ambiguous server failures retain reservations.
                entry.update(state="rejected" if 400 <= response.status_code < 500 and response.status_code != 408 else "uncertain",
                             http_status=response.status_code, error=f"HTTP {response.status_code}")
                error = entry["error"]
            except (httpx.TransportError, ValueError) as exception:
                retry = True
                error = str(exception).replace(self.secret, "[REDACTED]") if self.secret else str(exception)
                entry.update(state="uncertain", error=f"{type(exception).__name__}: {error[:180]}")
            atomic_json(path, entry)
            if not retry or attempt == 5:
                raise RuntimeError(f"Gateway request failed: {error}")
            await asyncio.sleep(min(60, 5 * 2 ** attempt))
        raise RuntimeError("Retry limit exhausted")

    @staticmethod
    def decode(entry, request_hash):
        payload = entry["response"]
        provider = payload.get("model")
        if provider != MODEL and not (isinstance(provider, str) and provider.startswith(MODEL + "-")):
            raise RuntimeError(f"Unexpected provider model: {provider!r}")
        if not payload.get("choices"):
            raise RuntimeError("Provider returned no choices")
        return {**api.parse_response("openai", payload), "model": MODEL, "provider_model": provider,
                "request_sha256": request_hash}


async def run(args):
    directory = ROOT / "bench/runs" / args.run
    manifest = json.loads((directory / "manifest.json").read_text())
    raw = directory / "raw_gpt56sol"
    raw.mkdir(exist_ok=True)
    config = {"model": MODEL, "reasoning_effort": "medium", "think_budget": api.THINK_BUDGET,
              "rounds": 4, "backtest": True, "perturb": False, "question_digest": manifest["question_digest"],
              "model_endpoint": api.GATEWAY,
              "implementation_sha256": {str(p.relative_to(ROOT)): sha(p) for p in
                  (Path(__file__), ROOT / "bench/run_infer_api.py", ROOT / "bench/cot/exec_code.py",
                   ROOT / "bench/cot/prepare_sol30.py")}}
    path = directory / "api_config.json"
    if path.exists() and json.loads(path.read_text()) != config:
        previous = json.loads(path.read_text())
        semantic = lambda c: {k: v for k, v in c.items() if k != "implementation_sha256"}
        if sha(path) != args.previous_config_sha256 or semantic(previous) != semantic(config):
            raise ValueError("Refusing an unaudited implementation change or a different API protocol")
        if not args.dry:
            atomic_json(directory / f"api_config.previous.{sha(path)}.json", previous)
    work, expected = [], collections.Counter()
    for name, question_sha in manifest["questions"].items():
        qpath = directory / "questions" / name
        if sha(qpath) != question_sha:
            raise ValueError(f"Question hash changed: {name}")
        have = {key_of(r) for r in read_rows(raw / name)
                if not str(r.get("finish", "")).startswith("error")} if (raw / name).exists() else set()
        for q in read_rows(qpath):
            expected[q["fmt"]] += 1
            if key_of(q) not in have:
                work.append((name, q))
    work.sort(key=lambda item: (item[0], {"table": 0, "free": 1, "code": 2}[item[1]["fmt"]], item[1]["max_tokens"]))
    print(json.dumps({"remaining_rows": len(work), "expected": dict(expected), "budget_usd": args.budget_usd}), flush=True)
    if args.dry:
        return
    if not 1 <= args.concurrency <= 6:
        raise ValueError("Gateway concurrency must be between 1 and 6")
    if args.budget_usd <= 0:
        raise ValueError("A positive authorized budget is required")
    atomic_json(path, config)
    atomic_json(directory / f"api_runtime.{time.time_ns()}.json",
                {"started_at": timestamp(), "pid": os.getpid(), "budget_usd": args.budget_usd,
                 "concurrency": args.concurrency, "config_sha256": sha(path),
                 "previous_config_sha256": args.previous_config_sha256})
    key = api.key_from("")
    status_path = directory / "status.json"
    started = time.time()
    counters, errors = collections.Counter(), []
    queue = asyncio.Queue()
    for item in work:
        queue.put_nowait(item)
    stop = asyncio.Event()
    drain = asyncio.Event()
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(signum, drain.set)
    locks = collections.defaultdict(asyncio.Lock)
    async with httpx.AsyncClient(base_url=api.GATEWAY, headers={"Authorization": f"Bearer {key}"},
                                 timeout=httpx.Timeout(1800, connect=30)) as client:
        journal = JournalClient(directory / "api_calls", client, args.budget_usd, key, stop)

        def update(state):
            if state == "running" and drain.is_set():
                state = "draining"
            report = {"state": state, "updated_at": timestamp(), "pid": os.getpid(), "run": args.run,
                      "concurrency": args.concurrency,
                      "counts": dict(expected), "session_rows_done": sum(counters.values()),
                      "session_rows_total": len(work), "session_format_counts": dict(counters),
                      "elapsed_seconds": time.time() - started, "accounting": journal.accounting(), "errors": errors}
            atomic_json(status_path, report)
            return report

        async def worker():
            while not stop.is_set() and not drain.is_set():
                try:
                    name, q = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                try:
                    record = await do_row(q, journal.generate)
                    async with locks[name]:
                        path = raw / name
                        previous = read_rows(path) if path.exists() else []
                        previous = [r for r in previous if key_of(r) != key_of(record)]
                        save_rows(path, previous + [record])
                    counters[q["fmt"]] += 1
                    update("running")
                    print(f"{sum(counters.values())}/{len(work)} {q['fmt']} {name} rounds={record.get('n_rounds', 1)} cost<={journal.accounting()['committed_usd']:.2f}", flush=True)
                except Exception as exc:
                    message = str(exc).replace(key, "[REDACTED]")
                    errors.append({"anchor": name, "key": key_of(q), "type": type(exc).__name__, "error": message[:300]})
                    stop.set()
                finally:
                    queue.task_done()

        async def heartbeat():
            while True:
                update("running")
                await asyncio.sleep(30)

        update("running")
        heart = asyncio.create_task(heartbeat())
        try:
            await asyncio.gather(*(worker() for _ in range(args.concurrency)))
        finally:
            heart.cancel()
            await asyncio.gather(heart, return_exceptions=True)
        if errors:
            update("budget_exhausted" if any(e["type"] == "BudgetExceeded" for e in errors) else "failed")
            raise RuntimeError("Inference stopped; inspect status.json; cached API responses remain resumable")
        if drain.is_set():
            update("drained")
            return
        for name in manifest["anchors"]:
            expected_keys = {key_of(q) for q in read_rows(directory / "questions" / name)}
            actual = read_rows(raw / name)
            if len(actual) != len(expected_keys) or {key_of(r) for r in actual} != expected_keys:
                raise ValueError(f"Incomplete output coverage: {name}")
        update("inference_complete")
        if not args.no_score:
            update("scoring")
            commands = [
                [str(ROOT / ".venv/bin/python"), "-m", "bench.cot.score_cot", "--run", args.run,
                 "--raw", "raw_gpt56sol", "--cells_run", "v3-bench-br", "--fact_run", args.run,
                 "--jobs", "8", "--bootstrap", "2000"],
                [str(ROOT / ".venv/bin/python"), "-m", "bench.cot.analyze_aligned", "--run", args.run,
                 "--raw", "raw_gpt56sol", "--bootstrap", "2000"],
            ]
            for index, command in enumerate(commands):
                with (directory / f"score_{index}.log").open("a") as log:
                    process = await asyncio.create_subprocess_exec(*command, cwd=ROOT, stdout=log, stderr=log)
                    if await process.wait() != 0:
                        errors.append({"stage": "scoring", "log": f"score_{index}.log"})
                        update("scoring_failed")
                        raise RuntimeError("Scoring failed")
            update("complete")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="cot-v32-sol30")
    parser.add_argument("--budget_usd", type=float, required=True)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--dry", action="store_true")
    parser.add_argument("--no_score", action="store_true")
    parser.add_argument("--previous_config_sha256", default="",
                        help="Audited prior configuration for a semantics-preserving implementation update")
    args = parser.parse_args()
    directory = ROOT / "bench/runs" / args.run
    with (directory / "api_runner.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        asyncio.run(run(args))


if __name__ == "__main__":
    main()
