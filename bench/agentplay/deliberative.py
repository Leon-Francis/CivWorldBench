"""Operational planning and observed execution checks, isolated from legacy runs."""
from __future__ import annotations

import json
import os
from pathlib import Path

from bench.agentplay.agent import Agent, extract_json, to_plan
from bench.agentplay.dialogue import _H, notes_block
from bench.agentplay.wm import History, WorldModel

TRACE_PATH = None
PLAN = """A plan is a JSON object with:
intent: detailed strategy and reasoning, not a slogan;
steps: a list of concrete actions in temporal order, naming existing cities and unit IDs,
production sequences, research, spending, settlement destinations and escorts where relevant;
assumptions: observable facts the plan depends on, with deadlines and consequences if false;
contingencies: what changes if an assumption fails, including alternative enemy targets;
checkpoints: optional machine-checkable observations, each with kind (unit_at, city_garrison,
or city_owned), by_turn, and on_failure (what to reconsider). unit_at also has unit and at:[x,y];
city_garrison has city and minimum; city_owned has city. These are observations, NOT constraints.
Use only checkpoints relevant to your plan, not arbitrary quotas. A future unit without an ID
belongs in steps/assumptions, not a fabricated ID. Write full plans, not partial patches.
Plans declare intent: the driver does NOT automatically execute their steps or queues.
Exact structural example (replace content with your actual plan):
{"intent":"detailed reasoning", "steps":["specific action"],
 "assumptions":["specific observable premise"], "contingencies":["if premise fails, response"],
 "checkpoints":[]}
Keep these five keys and their types. Checkpoints may be empty; never invent entries just to fill them.
"""

REASONING = """Review feasibility against the CURRENT panel and engine feedback. Separate units
inside a city from adjacent, travelling, embarked, or merely planned units. For a redeployment,
consider arrival time AND the cities losing those defenders. Attack destinations are uncertain;
opponents may change targets. Do not assume a declared reinforcement has happened.
Distinguish P(assault given current evidence) from P(loss given assault). A historical loss
frequency is neither the former nor a cap on current risk, especially after changing defenses.
Check economic and growth opportunities as well as military dangers; do not prescribe a fixed
style. Weigh losses of multiple cities, population, production and research, not just a binary
loss event. Calculator outputs are limited combat approximations, not guaranteed outcomes.
"""


def require_plan(plan):
    if not isinstance(plan, dict) or not isinstance(plan.get("intent"), str) or not plan["intent"].strip():
        raise ValueError("plan needs a nonempty intent")
    for key in ("steps", "assumptions", "contingencies", "checkpoints"):
        if not isinstance(plan.get(key), list):
            raise ValueError(f"plan.{key} must be a list")
    if not plan["steps"]:
        raise ValueError("plan needs concrete steps")
    for check in plan["checkpoints"]:
        if not isinstance(check, dict) or check.get("kind") not in ("unit_at", "city_garrison", "city_owned"):
            raise ValueError("invalid checkpoint kind")
        if type(check.get("by_turn")) is not int or not isinstance(check.get("on_failure"), str):
            raise ValueError("checkpoint needs integer by_turn and on_failure")
        if check["kind"] == "unit_at":
            if type(check.get("unit")) is not int or not isinstance(check.get("at"), list) or len(check["at"]) != 2 or any(type(v) is not int for v in check["at"]):
                raise ValueError("unit_at requires integer unit and [x,y]")
        else:
            if not isinstance(check.get("city"), str):
                raise ValueError("checkpoint requires city name")
            if check["kind"] == "city_garrison" and (type(check.get("minimum")) is not int or check["minimum"] < 0):
                raise ValueError("city_garrison requires nonnegative minimum")
    return plan


def observations(plan, view, turn):
    ego, rules = view["ego"], view["rules"]
    result = [{"checkpoint": check, "actual": None, "status": "not_machine_checkable"}
              for check in plan.get("unverified_checkpoints", [])]
    for check in plan.get("checkpoints", []):
        city = ego.cities.get(check.get("city"))
        if check["kind"] == "unit_at":
            unit = ego.units.get(check["unit"])
            actual = [int(unit["x"]), int(unit["y"])] if unit else None
            arrived = actual == check["at"] and int(unit.get("transported_by", -1)) < 0 if unit else False
            passed = arrived
        elif check["kind"] == "city_owned":
            actual = city is not None
            passed = actual
        else:
            actual = sum(1 for unit in ego.units.values()
                         if city and (unit["x"], unit["y"]) == (city["x"], city["y"])
                         and int(unit.get("transported_by", -1)) < 0
                         and rules.units.get(str(unit["type_by_name"]))
                         and rules.units[str(unit["type_by_name"])].military)
            passed = city is not None and actual >= check["minimum"]
        result.append({"checkpoint": check, "actual": actual, "status": "unresolved_missing_unit" if check["kind"] == "unit_at" and actual is None else "met" if passed else
                       "failed" if turn >= check["by_turn"] else "pending"})
    return result


def normalize_execution(value):
    if not any(key in value for key in ("cities", "units", "rates", "research_now", "research_goal",
                                        "government", "treaties", "strategy_note", "end_turns")):
        raise ValueError("response must contain gameplay orders, not only review metadata")
    missing = []
    review = value.get("execution_review")
    if not isinstance(review, dict):
        missing.append("execution_review")
    else:
        missing.extend("execution_review." + key for key in
                       ("observed", "unmet_assumptions", "changes") if key not in review)
    if type(value.get("replan_requested")) is not bool:
        missing.append("replan_requested")
    value["metadata_missing"] = missing
    to_plan(value)


def retain_memo(value, previous):
    value["memo_updated"] = isinstance(value.get("memo"), str) and bool(value["memo"].strip())
    if not value["memo_updated"]:
        value["memo"] = previous


def normalize_proposals(value):
    if "plans" not in value and "intent" in value and "steps" in value:
        plan = {key: value[key] for key in
                ("intent", "steps", "assumptions", "contingencies", "checkpoints") if key in value}
        value["plans"] = [plan]
    if not isinstance(value.get("plans"), list) or not 1 <= len(value["plans"]) <= 3:
        raise ValueError("provide one to three sensible plans")
    for plan in value["plans"]:
        if not isinstance(plan, dict):
            raise ValueError("plan must be an object")
        for key in ("steps", "assumptions", "contingencies"):
            content = plan.get(key)
            if isinstance(content, (str, dict)) and content:
                plan[key] = [content]
            elif key != "steps" and content is None:
                plan[key] = []
                plan.setdefault("missing_plan_fields", []).append(key)
        checks = plan.get("checkpoints")
        if checks is None:
            checks = []
        if not isinstance(checks, list):
            checks = [checks]
        supported = []
        for check in checks:
            try:
                require_plan({**plan, "checkpoints": [check]})
            except (ValueError, TypeError, KeyError):
                plan.setdefault("unverified_checkpoints", []).append(check)
            else:
                supported.append(check)
        plan["checkpoints"] = supported
        require_plan(plan)


def checked_json(call, messages, validate):
    attempts = []
    original_messages = messages
    for attempt in range(3):
        output = call(messages)
        output["messages"] = messages
        attempts.append(output)
        if TRACE_PATH is not None:
            with TRACE_PATH.open("a") as stream:
                stream.write(json.dumps({"attempt": attempt, "call": output}, ensure_ascii=False) + "\n")
        if str(output.get("finish", "")).startswith("error"):
            raise RuntimeError(f"inference failed: {output['finish']}")
        try:
            value = extract_json(output["answer"])
            validate(value)
            return value, attempts
        except (ValueError, KeyError, TypeError) as error:
            if TRACE_PATH is not None:
                with TRACE_PATH.open("a") as stream:
                    stream.write(json.dumps({"validation_error": str(error)}) + "\n")
            messages = original_messages + [
                {"role": "assistant", "content": output["answer"]},
                {"role": "user", "content":
                 f"Repair this JSON response. Validation error: {error}. Preserve its decisions, commands, "
                 "numbers and reasoning. Do not reconsider strategy or add alternatives. Return only the "
                 "complete corrected JSON object following the original schema, with properly escaped "
                 "strings and delimiters. Supply missing fields using the original context. "
                 "Do not shorten or drop other fields."}]
    raise RuntimeError("Invalid planning JSON after three attempts: " + messages[-1]["content"])


class DeliberativeAgent(Agent):
    config = None
    current_view = None

    def __init__(self, *args, **kwargs):
        global TRACE_PATH
        super().__init__(*args, **kwargs)
        self.system = self.system.replace(
            "- Diplomacy is passive: you cannot propose treaties; AI civs may declare war on you or offer nothing.  Defend accordingly.",
            "- Diplomacy: request ceasefire, peace or war with the treaties field. The AI may reject a request; inspect the actual feedback.")
        self.directory = Path(self.config["out"])
        TRACE_PATH = self.directory / "calls.jsonl"
        self.history = History(self.config["world"], self.directory, self.config["start"], self.config["ego"])
        self.forecaster = WorldModel(self.config["wm_endpoint"], self.config["wm_model"], "medium") if self.config["wm_model"] else None
        self.sessions = self.directory / "planning.jsonl"
        self.course = None
        self.memo = ""
        self.last_planning = -999
        self.replan = False
        if self.sessions.exists():
            last = json.loads(self.sessions.read_text().splitlines()[-1])
            self.course, self.memo = last, last["wm_memo"]
            self.last_planning = last["turn"]
        turns = self.directory / "turns.jsonl"
        if turns.exists():
            last = json.loads(turns.read_text().splitlines()[-1])
            self.replan = bool((last.get("orders") or {}).get("replan_requested"))

    def agent_json(self, context, question, validator, execution=False):
        system = self.system if execution else self.system.split("\n\n# ORDER FORMAT")[0].replace(
            "You reply with\none JSON object of orders for this turn.",
            "In this planning call, return the requested planning JSON, not turn orders.")
        if not execution:
            system += "\nThis call is strategic planning. Follow the planning/revision JSON schema in the user message; do not output turn orders."
        else:
            system += ('\nThe ORDER FORMAT also includes execution_review: '
                       '{"observed":"evidence of progress", "unmet_assumptions":"premises not met", '
                       '"changes":"tactical changes and reasons"}, and replan_requested: true or false. '
                       'Always include the boolean; use true when requesting a new strategic discussion. '
                       'These accompany, and do not replace, your gameplay orders.')
        return checked_json(lambda messages: self.call(messages, max_tokens=24000),
                            [{"role": "system", "content": system},
                             {"role": "user", "content": context + "\n" + question}], validator)

    def context(self, report, panel, note, turn):
        prior = ""
        if self.course:
            prior = json.dumps({key: self.course[key] for key in ("turn", "final_plan", "rationale", "discussion")}, ensure_ascii=False)
        checks = observations(self.course["final_plan"], self.current_view, turn) if self.course else []
        return (f"TURN {turn}; final score is measured at T{self.config['end']}.\n"
                f"WORLD REPORT\n{report}\nPANEL AND LAST ENGINE FEEDBACK\n{panel}\n"
                f"PREVIOUS ADOPTED PLAN AND DISCUSSION (not replaced by the scratchpad)\n{prior or 'none'}\n"
                f"CURRENT OBSERVED CHECKPOINTS\n{json.dumps(checks, ensure_ascii=False)}\n"
                f"TACTICAL SCRATCHPAD\n{note}\n{REASONING}")

    def forecast(self, context, plans, turn, round_number):
        horizons = sorted({min(turn + 20, self.config["end"]), min(turn + 40, self.config["end"]), self.config["end"]})
        question = f"""Act as a world-model adviser, not the ruler. Forecast each candidate in order.
Evaluate intended execution AND plausible execution failures. Do not assume its premises are true.
Return JSON {{"forecasts":[{{"plan_index":0, "outcomes":[{{"turn":{horizons[0]},
"score_p10_p50_p90":[0,0,0], "cities_lost_p10_p50_p90":[0,0,0],
"citizens_lost_p10_p50_p90":[0,0,0], "p_any_city_lost":0.0, "p_two_or_more_cities_lost":0.0}}],
"assumptions_audit":"arrival times, withdrawals and the cities exposed, production/research/budget feasibility; cite evidence and uncertainty",
"opponent_responses":"plausible targets including alternatives after redeployment; current evidence, earliest arrival and attack probability, not historical averages",
"failure_chain":"what can cause cascading losses or failure to grow, and magnitude",
"advice":"specific improvements, tradeoffs, growth opportunities, what evidence would change your advice"}}],
"memo":"updated factual belief about this game; separate observations from hypotheses"}}.
Provide one outcome at EACH absolute turn {horizons}. Scores are TOTAL scores, losses are
cumulative gross losses from now, not net counts. Quantiles must be ordered; probabilities 0..1.
Index identifies the input only, not a required final choice. Do not invent precise timing when unknown.
Current round {round_number}. The final horizon is uncertain: expose that uncertainty.
CANDIDATES\n{json.dumps(plans, ensure_ascii=False)}"""

        def validate(value):
            forecasts = value.get("forecasts")
            if not isinstance(forecasts, list) or len(forecasts) != len(plans):
                raise ValueError("one forecast required per plan")
            if sorted(item["plan_index"] for item in forecasts) != list(range(len(plans))):
                raise ValueError("plan indices must match candidates")
            for item in forecasts:
                if sorted(outcome["turn"] for outcome in item["outcomes"]) != horizons:
                    raise ValueError("missing forecast horizon")
                for outcome in item["outcomes"]:
                    for key in ("score_p10_p50_p90", "cities_lost_p10_p50_p90", "citizens_lost_p10_p50_p90"):
                        values = outcome[key]
                        if not isinstance(values, list) or len(values) != 3 or any(type(v) not in (int, float) or v < 0 for v in values) or values != sorted(values):
                            raise ValueError(f"invalid quantiles {key}")
                    if not 0 <= outcome["p_two_or_more_cities_lost"] <= outcome["p_any_city_lost"] <= 1:
                        raise ValueError("inconsistent loss probabilities")
            retain_memo(value, self.memo)

        frames = sorted({max(1, turn - 20), max(1, turn - 10), turn})
        history, _, _, _ = _H(self.history, frames, self.current_view["ego"].name)
        body = notes_block() + "\n" + history + "\n" + context + "\nPREVIOUS WM BELIEF\n" + self.memo + "\n" + question
        forecast, calls = checked_json(lambda messages: self.forecaster.call(
            "\n\n".join(message["content"] for message in messages), 16000),
            [{"role": "user", "content": body}], validate)
        missing = []
        for item in forecast["forecasts"]:
            for key in ("assumptions_audit", "opponent_responses", "failure_chain", "advice"):
                if isinstance(item.get(key), (list, dict)) and item[key]:
                    item[key] = json.dumps(item[key], ensure_ascii=False)
                if not isinstance(item.get(key), str) or not item[key].strip():
                    missing.append({"plan_index": item["plan_index"], "field": key})
        if missing:
            def supplement(value):
                expected = {(item["plan_index"], item["field"]) for item in missing}
                entries = value.get("supplements")
                if not isinstance(entries, list) or len(entries) != len(expected):
                    raise ValueError("provide exactly the requested supplements")
                if {(entry["plan_index"], entry["field"]) for entry in entries} != expected:
                    raise ValueError("supplement indices and fields must match the request")
                if any(not isinstance(entry.get("text"), str) or not entry["text"].strip() for entry in entries):
                    raise ValueError("each supplement needs nonempty text")
            request = (body + "\nYOUR FORECAST (keep its numbers unchanged):\n" + json.dumps(forecast) +
                       "\nSupply ONLY the missing analysis fields: " + json.dumps(missing) +
                       '\nReply {"supplements":[{"plan_index":0,"field":"advice","text":"analysis"}]}. '
                       "Use the original evidence and candidates. Do not repeat the forecast tables.")
            extra, extra_calls = checked_json(lambda messages: self.forecaster.call(
                "\n\n".join(message["content"] for message in messages), 8000),
                [{"role": "user", "content": request}], supplement)
            indexed = {item["plan_index"]: item for item in forecast["forecasts"]}
            for entry in extra["supplements"]:
                indexed[entry["plan_index"]][entry["field"]] = entry["text"]
            forecast["supplemented_fields"] = missing
            calls.extend(extra_calls)
        return forecast, calls

    def deliberate(self, context, turn):
        def proposals(value):
            normalize_proposals(value)

        initial, calls = self.agent_json(context, PLAN + "\nPropose one to three genuinely useful plans under your own judgment. One is enough; do not invent alternatives to fill slots. Reply {\"plans\":[...]}.", proposals)
        plans = initial["plans"]
        rounds, discussion = [], []
        final_plan = None
        for round_number in range(1, 4 if self.forecaster else 2):
            if self.forecaster:
                forecast, wm_calls = self.forecast(context + "\nCURRENT DISCUSSION\n" + json.dumps(discussion, ensure_ascii=False), plans, turn, round_number)
                self.memo = forecast["memo"]
            else:
                forecast, wm_calls = None, []
            last_round = round_number == 3 or self.forecaster is None
            question = PLAN + "\nReview your candidates and the evidence. You may edit, drop or merge freely. Finalize as soon as satisfied; no mandatory revision. "
            question += "This is the last round: return the full final plan, not an index. " if last_round else "You may request another forecast of revised plans. "
            question += '\nAlways return the same outer structure: {"final":true,"plans":[PLAN],"rationale":"why this plan, rejected alternatives, uncertainty and what would invalidate it"}. If requesting more discussion, set final=false and include one to three revised plans. If final=true, plans must contain EXACTLY ONE complete plan. Keep intent, steps, assumptions, contingencies and checkpoints INSIDE each plan object. Put rationale at the outer level.\n'
            question += "CANDIDATES\n" + json.dumps(plans, ensure_ascii=False) + "\nWM ADVICE\n" + json.dumps(forecast, ensure_ascii=False)

            def revision(value):
                if type(value.get("final")) is not bool or not isinstance(value.get("rationale"), str) or not value["rationale"].strip():
                    raise ValueError("explicit final boolean and rationale required")
                if last_round and not value["final"]:
                    raise ValueError("last round requires a final plan")
                proposals(value)
                if value["final"] and len(value["plans"]) != 1:
                    raise ValueError("final=true requires exactly one complete plan in plans")

            decision, agent_calls = self.agent_json(context + "\nCURRENT DISCUSSION\n" + json.dumps(discussion, ensure_ascii=False), question, revision)
            rounds.append({"round": round_number, "plans": plans, "forecast": forecast, "decision": decision,
                           "wm_calls": wm_calls, "agent_calls": agent_calls})
            discussion.append({"plans": plans, "forecast": forecast, "decision": decision})
            if decision["final"]:
                final_plan = decision["plans"][0]
                break
            plans = decision["plans"]
        self.course = {"turn": turn, "final_plan": final_plan, "rationale": decision["rationale"],
                       "discussion": discussion, "rounds": rounds, "initial_calls": calls,
                       "wm_memo": self.memo, "wm_model": self.config["wm_model"],
                       "code_revision": os.environ.get("WM_AGENTPLAY_REVISION", "original")}
        with self.sessions.open("a") as stream:
            stream.write(json.dumps(self.course, ensure_ascii=False) + "\n")
        self.last_planning = turn
        self.replan = False

    def decide(self, report, panel, note, turn, extra=""):
        context = self.context(report, panel, note, turn)
        if self.course is None or turn - self.last_planning >= 10 or (self.replan and turn - self.last_planning >= 2):
            self.deliberate(context, turn)
            context = self.context(report, panel, note, turn)
        instructions = ("Execute your adopted plan using the ACTUAL positions and last-order results above. "
                        "The plan is not a command queue and does not auto-execute. Tactical changes are free. "
                        "Include execution_review: {observed, unmet_assumptions, changes}, explaining evidence "
                        "for progress and changes, and replan_requested (boolean) if new evidence requires strategic "
                        "discussion next turn. Do not count rejected orders or in-transit units as completed. "
                        "If a checkpoint is wrong, explain and revise the plan rather than blindly satisfying it.")
        def execution(value):
            normalize_execution(value)

        obj, calls = self.agent_json(context, instructions + "\nGive this turn's orders in the ORDER FORMAT with these additional review fields.", execution, execution=True)
        plan, feedback = to_plan(obj)
        result = {"plan": plan, "feedback": feedback, "obj": obj, "parse_ok": True,
                  "note": str(obj.get("strategy_note") or note), "user_chars": len(context) + len(instructions),
                  "llm": {**calls[-1], "attempts": len(calls),
                          "prompt_tokens": sum(call.get("prompt_tokens", 0) for call in calls),
                          "completion_tokens": sum(call.get("completion_tokens", 0) for call in calls),
                          "seconds": sum(call.get("seconds", 0) for call in calls)}}
        self.replan = obj.get("replan_requested") is True
        result["end_turns"] = 1
        obj["end_turns"] = 1
        obj["plan_turn"] = self.last_planning
        obj["code_revision"] = os.environ.get("WM_AGENTPLAY_REVISION", "original")
        obj["checkpoint_observations"] = observations(self.course["final_plan"], self.current_view, turn)
        return result
