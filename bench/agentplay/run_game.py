"""Play one game: an LLM takes over one civilization at a bench-world checkpoint and plays N turns.

  PYTHONPATH=$PWD .venv/bin/python bench/agentplay/run_game.py --world fc-civ2civ3-S-24x32-p3-hard-s500001 \
        --ego 0 --start 60 --turns 100 --out bench/agentplay/games/pilot/s500001_e0 [--endpoint http://localhost:8011/v1]

Writes <out>/turns.jsonl (one line per engine turn: orders, feedback, state summary, LLM usage), <out>/saves/ (every
autosave), <out>/summary.json (score trajectory + the mainline AI's own score for the same civ and turns as reference).
Resumable: restarts from the last autosave in <out>/saves and the last turns.jsonl line.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "bundle"))
from sandboxes.freeciv.savegame import Savegame                                   # noqa: E402

from bench.agentplay.agent import Agent                                           # noqa: E402
from bench.agentplay.engine import Ego, TurnPlan, feedback_from_results, make_human, plan_to_edits, run_turn  # noqa: E402
from bench.agentplay.observe import build_view                                    # noqa: E402
from bench.agentplay.rules import load_rules, rules_audit                         # noqa: E402
from bench.agentplay.wm import History, WorldModel, decision_as_plan, utility   # noqa: E402
from bench.agentplay.dialogue import Dialogue                                     # noqa: E402
from bench.agentplay.contract import Contract, parse_contract                     # noqa: E402
from bench.agentplay.siege import route_risk, plan_risk                            # noqa: E402
from bench.agentplay.plan import PlanRunner, render_plan, validate_plan, prune_to_cities, carry_forward  # noqa: E402
import concurrent.futures as cf                                                   # noqa: E402
import random                                                                     # noqa: E402

BENCH = ROOT / "data_v2/bench"


def takeover(sg: Savegame, idx: int) -> dict[int, dict]:
    """Human control: drop the ai flag, cancel the AI's automation and gotos so every unit starts awaiting orders
    (fortified/sentried units keep their posture).  Returns the standing orders a human would inherit from the
    units' displayed destinations: Settlers with a goto -> found a city there (or the nearest legal site), others -> move."""
    make_human(sg, idx)
    sec = f"player{idx}"; t = sg.table(sec, "u")
    inherited: dict[int, dict] = {}
    for i, r in enumerate(t.as_dicts() if t else []):
        if str(r.get("go")).upper() == "TRUE":
            tgt = (int(r["goto_x"]), int(r["goto_y"]))
            if str(r["type_by_name"]) == "Settlers":
                inherited[int(r["id"])] = {"kind": "found_city", "target": tgt, "since": sg.turn, "inherited": True}
            elif str(r["type_by_name"]) not in ("Workers", "Engineers"):
                inherited[int(r["id"])] = {"kind": "move", "target": tgt, "since": sg.turn, "inherited": True}
        sg.set_cell(sec, "u", i, "server_side_agent", 0)
        if int(r["activity"]) in (10, 11):
            sg.set_cell(sec, "u", i, "activity", 0); sg.set_cell(sec, "u", i, "activity_count", 0)
        if str(r.get("go")).upper() == "TRUE":
            sg.set_cell(sec, "u", i, "go", False)
        if int(r.get("orders_length", 0) or 0) > 0:
            sg.set_cell(sec, "u", i, "orders_length", 0); sg.set_cell(sec, "u", i, "orders_index", 0)
            sg.set_cell(sec, "u", i, "orders_list", "-"); sg.set_cell(sec, "u", i, "dir_list", "-"); sg.set_cell(sec, "u", i, "activity_list", "-")
    return inherited


def _event_kinds(alert: str) -> list[str]:
    return [seg.strip().split(" ")[0] for seg in alert.split(": ", 1)[1].split("; ")]


def state_summary(sg: Savegame, idx: int) -> dict:
    ego = Ego(sg, idx); sc = f"score{idx}"
    return {"turn": sg.turn, "score": int(sg.get(sc, "total", 0) or 0), "gold": ego.gold, "gov": ego.gov,
            "n_cities": len(ego.cities), "n_units": len(ego.units), "n_techs": len(ego.known_techs),
            "citizens": sum(int(c["size"]) for c in ego.cities.values()), "techout": int(sg.get(sc, "techout", 0) or 0),
            "bnp": int(sg.get(sc, "bnp", 0) or 0), "mfg": int(sg.get(sc, "mfg", 0) or 0),
            "units_lost": int(sg.get(sc, "units_lost", 0) or 0), "units_killed": int(sg.get(sc, "units_killed", 0) or 0),
            "alive": bool(sg.get(f"player{idx}", "is_alive", True))}


def mainline_reference(world: str, idx: int, turns: list[int]) -> dict[int, dict]:
    """The same civ's score/gold/cities on the AI-played mainline at the given turns (corpus records)."""
    out = {}; want = set(turns)
    with gzip.open(BENCH / world / "world.jsonl.gz", "rt") as f:
        for l in f:
            r = json.loads(l)
            if r["t"] in want:
                p = r["state"]["players"][idx]
                out[r["t"]] = {"score": p["score"], "gold": p["gold"], "n_cities": len(p["cities"]), "n_techs": len(p["techs"]),
                               "citizens": sum(c["size"] for c in p["cities"])}
                if len(out) == len(want):
                    break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", required=True); ap.add_argument("--ego", type=int, default=0)
    ap.add_argument("--start", type=int, default=60); ap.add_argument("--turns", type=int, default=100)
    ap.add_argument("--out", required=True)
    ap.add_argument("--endpoint", default="http://localhost:8011/v1"); ap.add_argument("--model", default="flash-next")
    ap.add_argument("--effort", default="medium"); ap.add_argument("--max_tokens", type=int, default=20000)
    ap.add_argument("--arm", default="none", help="none | wm: every --wm_every turns the agent proposes candidate plans, the world model "
                                                 "(same LLM, benchmark 1a format) forecasts each, the agent decides with the forecasts in view | "
                                                 "wm_outlook: every --wm_every turns one forecast of the agent's current plan is shown as a threat/outlook bulletin | "
                                                 "wm2: 3 diverse candidates, ONE comparative forecast (dense frames, +10/+20, 3 samples, calibration appendix), commit to one")
    ap.add_argument("--wm_every", type=int, default=10); ap.add_argument("--wm_endpoint", default="")
    ap.add_argument("--wm_model", default="", help="forecaster model; default = the agent's model. claude-*/gpt-* go through the univ gateway")
    ap.add_argument("--wm_samples", type=int, default=3, help="wm2/bok: forecaster samples per decision point (median aggregation)")
    ap.add_argument("--k", type=int, default=4, help="bok: number of sampled decisions per decision point")
    ap.add_argument("--select", default="wm", help="bok: how the driver picks among the K decisions: wm | random | self")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dlg_rounds", type=int, default=3, help="dlg: max forecast rounds per planning point (agent may revise and ask again)")
    ap.add_argument("--dlg_alert_cooldown", type=int, default=5, help="dlg: min turns between alert-triggered dialogues")
    ap.add_argument("--agent_naive", type=int, default=0, help="1 = agent without the expert manual and strong-play priors (doc 17 §4.6)")
    ap.add_argument("--dlg_targets", type=int, default=0, help="dlg: also ask the forecaster who will attack which city (v5.1)")
    ap.add_argument("--contract_turns", type=int, default=10, help="dlg6: how many turns a committed route's contract binds")
    ap.add_argument("--dlg_alerts", type=int, default=0, help="dlg: 1 = also open a dialogue on enemy-near / city-lost alerts (v4 behaviour); 0 = every --wm_every turns only (v5)")
    # --- headroom measurement (arm "replay"): re-run one of the K candidates of a finished best-of-K decision point
    #     and let the same agent carry on, so the TRUE spread across candidates (and the noise floor) can be measured.
    ap.add_argument("--from_game", default="", help="replay: source game dir (its turns.jsonl holds the candidates)")
    ap.add_argument("--from_turn", type=int, default=0, help="replay: the decision turn to fork at")
    ap.add_argument("--cand", type=int, default=0, help="replay: which candidate (0-based) to execute on that turn")
    ap.add_argument("--init_save", default="", help="fork: start from this savegame instead of the world's T{start} checkpoint (no takeover edits)")
    ap.add_argument("--init_note", default="", help="fork: strategy note in force at the first turn (a route's plan text)")
    ap.add_argument("--max_skip", type=int, default=5)
    a = ap.parse_args()

    out = Path(a.out); (out / "saves").mkdir(parents=True, exist_ok=True); (out / "work").mkdir(exist_ok=True)
    meta = json.load(open(BENCH / a.world / "meta.json"))
    ruleset = meta["rulesetdir"].split("_")[0] if meta.get("rulesetdir") else meta["config"]["ruleset"]
    rules = load_rules(ruleset)
    audit = rules_audit(rules)
    expected_rules = os.environ.get("WM_RULES_EXPECTED_SHA256")
    if expected_rules and audit["sha256"] != expected_rules:
        raise RuntimeError(f"Rules fingerprint mismatch: {audit['sha256']} != {expected_rules}")
    (out / "rules_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    agent = Agent(a.endpoint, a.model, ruleset, a.max_tokens, a.effort, naive=bool(a.agent_naive))
    wm = WorldModel(a.wm_endpoint or a.endpoint, a.wm_model or a.model, a.effort) if a.arm in ("wm", "wm_outlook", "wm2", "dlg", "dlg6", "dlg7", "plan", "brief") or (a.arm == "bok" and a.select == "wm") else None
    hist = History(a.world, out, a.start, a.ego) if wm is not None else None
    rng = random.Random(a.seed * 1000003 + a.start)
    dlg_end = a.start + a.turns
    dlg = Dialogue(wm, hist, targets=bool(a.dlg_targets), end=dlg_end,
                   v6=(a.arm in ("dlg6", "dlg7", "plan"))) if a.arm in ("dlg", "dlg6", "dlg7", "plan", "brief") else None
    runner = PlanRunner(rules) if a.arm == "plan" else None
    dec_log = ""
    contract = Contract(rules, a.contract_turns, mode=("hard" if a.arm == "dlg6" else "soft")) if a.arm in ("dlg6", "dlg7") else None
    last_dlg_turn = -999
    course: dict = {}          # ★ v7: the route the agent committed to, kept VERBATIM between planning points
    log_path = out / "turns.jsonl"
    prev = [json.loads(l) for l in open(log_path)] if log_path.exists() else []
    if prev:
        last = prev[-1]; save = Path(last["save"]); turn = last["turn"] + 1
        standing = {int(k): v for k, v in last["standing"].items()}; note = last["note"]; feedback = last["feedback_next"]
        events = []; skip_left = last.get("skip_left", 0)
        print(f"resuming at turn {turn} from {save.name}", flush=True)
    elif a.arm == "replay":
        # fork: the autosave that turn `from_turn` started from, plus the standing orders in force at that moment
        src = [json.loads(l) for l in open(Path(a.from_game) / "turns.jsonl")]
        prev_r = next(r for r in src if r["turn"] == a.from_turn - 1)
        dec_r = next(r for r in src if r["turn"] == a.from_turn)
        cands = dec_r["bok"]["cand_orders"]
        if a.cand >= len(cands) or cands[a.cand] is None:
            sys.exit(f"candidate {a.cand} missing at turn {a.from_turn}")
        forced_obj = cands[a.cand]
        src_save = Path(prev_r["save"]); save = out / "saves" / src_save.name
        shutil.copy(src_save, save)
        standing = {int(k): v for k, v in (prev_r["standing"] or {}).items()}
        turn = a.from_turn; note = str(forced_obj.get("strategy_note") or prev_r.get("note") or "")
        feedback = prev_r.get("feedback_next") or []
        events = []; skip_left = 0
    elif a.init_save:
        # ★ strategic headroom / oracle fork (doc 17 §6.9): resume a played game at turn `start` with a given course
        #   in force.  The save is already a human-controlled position, so no takeover edits are applied.
        src = Path(a.init_save); save = out / "saves" / src.name
        shutil.copy(src, save)
        turn = a.start; note = a.init_note
        standing = {}
        if a.from_game:
            src_recs = [json.loads(l) for l in open(Path(a.from_game) / "turns.jsonl")]
            prev_r = next((r for r in src_recs if r["turn"] == a.start - 1), None)
            if prev_r:
                standing = {int(k): v for k, v in (prev_r.get("standing") or {}).items()}
        feedback = ["You are resuming with the course in your strategy note; keep following it unless it fails."]
        events = []; skip_left = 0
    else:
        ck = BENCH / a.world / "ckpt" / f"T{a.start:04d}.sav"
        sg = Savegame.load(ck); standing = takeover(sg, a.ego); save = out / "saves" / f"T{a.start:04d}-takeover.sav"; sg.write(save)
        turn = a.start; note = ""
        feedback = ["You have just taken control of this civilization. Units that were already travelling keep their destinations as standing orders "
                    "(Settlers will found a city where they were heading, or at the nearest legal site); everything else awaits your orders."]
        events = []; skip_left = 0
    end = (a.from_turn if a.arm == "replay" else a.start) + a.turns
    forced_obj = locals().get("forced_obj")
    ego_name = None; prev_idle: set = set(); prev_near: set = set(); gold_hist: list[int] = []
    wm_history: list[dict] = []; pending_wm = None
    if prev:
        for r0 in prev:
            if r0.get("course"):
                course = r0["course"]
            if r0.get("decision_log"):
                dec_log = r0["decision_log"]
            if runner is not None and (r0.get("wm") or {}).get("final_plan"):
                runner.adopt(r0["wm"]["final_plan"], r0["turn"])
                runner.done = dict((r0.get("plan_done") or {}))
    if prev and contract is not None:
        for r0 in prev:
            c0 = r0.get("contract")
            if c0 and c0.get("start") == r0["turn"]:
                contract.commit(r0["turn"], c0.get("letter", ""), c0.get("name", ""), c0.get("spec"))
        last_c = prev[-1].get("contract") or {}
        if contract.active and last_c.get("start") == contract.start:
            contract.built = dict(last_c.get("built") or contract.built); contract.founded = int(last_c.get("founded", 0))
            contract.breaks = list(last_c.get("breaks") or [])
        if contract.active and not contract.in_force(turn):
            contract.spec = None
    if prev and dlg is not None:
        dlg.restore(prev)
        last_dlg_turn = max([r0["turn"] for r0 in prev if (r0.get("wm") or {}).get("kind") == "dialogue"] or [-999])
    if prev and a.arm == "wm2":          # rebuild the calibration history from the log on resume
        for r0 in prev:
            w0 = r0.get("wm")
            if w0 and w0.get("chosen"):
                p50 = {tuple([k.split("|")[0], int(k.split("|")[1])]): v for k, v in (w0.get("ego_p50") or {}).get(w0["chosen"], {}).items()}
                wm_history.append({"turn": r0["turn"], "chosen": w0["chosen"], "p50": p50, "levels": w0.get("levels_ego") or {}})
    while turn < end:
        t0 = time.time()
        sg = Savegame.load(save); ego = Ego(sg, a.ego); ego_name = ego.name
        gold_hist.append(ego.gold)
        view = build_view(sg, a.world, a.ego, rules, standing, feedback, events, gold_hist=gold_hist[-6:])
        if dlg is not None and dlg.standing:
            view["panel"] += "\n\n### FORECASTER (standing, from your last planning dialogue)\n- " + dlg.standing
        if contract is not None:
            cp = contract.panel(turn, ego.gold)
            if cp:
                view["panel"] += "\n\n" + cp
        if runner is not None and runner.plan:
            pb = runner.progress_block(turn, ego)
            if pb:
                view["panel"] += "\n\n" + pb
        # wake the model early only on news: failed/rejected orders, newly idle units, newly adjacent foreign units,
        # or an engine event about us other than routine growth
        failed = [f for f in feedback if f.startswith(("FAILED", "COULD NOT", "REJECTED", "PARSE", "LLM"))]
        idle_ids = {x.split()[0] for x in view["idle"]}
        near_now = {x for a in view["alerts"] if a.startswith("foreign units close") for x in a.split(": ", 1)[1].split("; ")}
        news = [a for a in view["alerts"] if a.startswith("last turn:") and not all(k in ("city_size_change", "unit_built", "building_built") for k in _event_kinds(a))]
        wake = bool(failed) or bool(idle_ids - prev_idle) or bool(near_now - prev_near) or bool(news) \
            or any("found_city" in f or "attack" in f or "conquer" in f for f in feedback)
        consult = skip_left <= 0 or wake
        prev_idle, prev_near = idle_ids, near_now
        rec = {"turn": turn, "consulted": consult, "woken": bool(wake and skip_left > 0), "alerts": view["alerts"], "feedback_in": feedback, "forecaster_line": (dlg.standing if dlg is not None else None)}
        extra = ""
        course_block = ""
        if a.arm == "plan" and runner is not None and runner.plan and (turn - a.start) % a.wm_every != 0:
            ms = runner.milestone_check(turn, state_summary(sg, a.ego))
            tail = ("\n\nFollow it.  Your cities are already set to their queues, so spend this turn on what the "
                    "plan cannot do by itself: move the units it needs moved, send settlers (leave \"to\" out and "
                    "the driver picks the best legal site for you), and BUY anything that has to arrive sooner than "
                    "the queue will deliver it -- gold you are still holding at the end of the game scores nothing.  "
                    "If something has changed and you are leaving the plan, say so in \"plan_change\": \"<what "
                    "changed and what you do instead>\".")
            course_block = (((ms + "\n\n") if ms else "")
                            + "## THE PLAN YOU ADOPTED AT TURN %d (the driver runs its production queue for you)\n" % runner.start
                            + render_plan(runner.plan)
                            + ("\n\n## YOUR DECISION LOG\n" + dec_log if dec_log else "")
                            + tail)
        if a.arm == "dlg7" and course and turn > course["turn"]:
            # ★ v7 (2026-09-19, user decision): the per-turn `strategy_note` is rewritten by the agent every turn, so a
            #   course agreed at a planning point erodes within a few turns -- which is why three routes that read very
            #   differently produced indistinguishable games (§6.9).  The committed route is therefore pinned VERBATIM in
            #   front of the orders on every turn until the next planning point.  Nothing is enforced; the agent may
            #   depart, and says so in "course_change", which is logged.
            course_block = (f"## THE COURSE YOU COMMITTED TO AT TURN {course['turn']} (route {course['letter']}: {course['name']})\n"
                            "You chose this route after comparing the forecaster's predictions for three genuinely different plans.\n"
                            "It is your plan until the next planning point at turn "
                            f"{course['turn'] + a.wm_every}.  Follow it in DIRECTION -- what you expand, what you build, what you\n"
                            "research, what you spend on -- and use this turn's orders for the tactics it needs.  Your standing note\n"
                            "below is a tactical scratchpad; it does not replace this course.\n\n"
                            + course["plan"] +
                            "\n\nIf the situation has genuinely changed and you are leaving this course, say so: put "
                            "\"course_change\": \"<one line: what changed and what you are doing instead>\" in your orders.\n")
        if a.arm == "brief" and (turn - a.start) % a.wm_every == 0:
            # ★ brief (doc 17 §4.5): one forecast of the course already in force, shown as a standing block.
            consult = True; tp = time.time()
            mu = dlg.memo_update(turn, ego.name)
            br = dlg.brief(turn, ego.name, note, siege_text="\n".join(view.get("siege") or []))
            view["panel"] = view["panel"].split("\n\n### FORECASTER (standing")[0] + "\n\n### FORECASTER (updated this turn)\n- " + br["text"]
            extra = ""
            rec_wm = {"kind": "brief", "memo": mu, "brief": br, "seconds": time.time() - tp}
        elif a.arm == "plan" and (turn - a.start) % a.wm_every == 0:
            # ★ WM v12 (doc 17 Section 4.12): the dialogue refines a PLAN until the agent calls it final.
            consult = True; tp = time.time(); last_dlg_turn = turn
            LAB = "ABC"
            mu = dlg.memo_update(turn, ego.name)
            pw = agent.plan_write(view["report"], view["panel"], dec_log, turn, n_plans=3)
            plans = pw["plans"]
            rounds = []; parts = []; final = False; changes = ""; new_log = ""
            if not plans and runner.plan:
                plans = [runner.plan]
            for rnd in range(1, a.dlg_rounds + 1):
                if not plans:
                    break
                labels = list(LAB[:len(plans)])
                checks = []
                for L_, pl in zip(labels, plans):
                    for b_ in prune_to_cities(pl, ego):
                        checks.append(f"PLAN {L_}: {b_} (the driver has dropped that entry).")
                    try:
                        checks += validate_plan(pl, f"PLAN {L_}", sg, rules, view["board"], view["geo"], ego, view["mi"], turn)
                    except Exception as e:                            # noqa: BLE001
                        print(f"turn {turn}: validate_plan failed ({type(e).__name__}: {e})", flush=True)
                try:
                    rr = plan_risk(sg, rules, view["board"], view["geo"], ego, view["rel"], plans, labels)
                except Exception as e:                                # noqa: BLE001
                    rr = []; print(f"turn {turn}: plan_risk failed ({type(e).__name__}: {e})", flush=True)
                f = dlg.forecast_plans(turn, ego.name, plans, labels, rounds,
                                       siege_text="\n".join(view.get("siege") or []), sites=view.get("sites"),
                                       risk_text="\n".join(rr), checks=checks)
                rounds.append({"round": rnd, "routes": [{"name": pl.get("name", ""), "plan": render_plan(pl)} for pl in plans],
                               "plans": plans, "letters": labels, "forecast": f, "checks": checks, "risk": rr})
                parts.append(f"### Round {rnd}: your plan(s)\n"
                             + "\n\n".join(f"PLAN {L_}\n{render_plan(pl)}" for L_, pl in zip(labels, plans))
                             + f"\n\n### Round {rnd}: the forecaster and the driver's checks\n{f['text']}")
                if rnd == a.dlg_rounds:
                    break
                rc = agent.plan_revise(view["report"], view["panel"], dec_log, turn, "\n\n".join(parts), rnd,
                                       a.dlg_rounds, must_revise=(rnd == 1))
                rounds[-1]["decision"] = {k: v for k, v in rc.items() if k != "llm"}
                rounds[-1]["decision_llm"] = {k: v for k, v in rc["llm"].items() if k != "think"}
                parts.append(f"### Round {rnd}: your revision\n{'FINAL' if rc['final'] else 'revised'} -- {rc['changes']}")
                if rc["plans"]:
                    prev_by_name = {pl.get("name", ""): pl for pl in plans}
                    carried = []
                    for i_, np_ in enumerate(rc["plans"]):
                        base = prev_by_name.get(np_.get("name", "")) or (plans[i_] if i_ < len(plans) else plans[0])
                        carried += carry_forward(np_, base)
                    if carried:
                        parts.append("### The driver: " + "; ".join(sorted(set(carried)))
                                     + ".  Restate a section only when you want it changed.")
                    plans = rc["plans"]
                if rc.get("decision_log"):
                    new_log = rc["decision_log"]
                changes = rc["changes"] or changes
                if rc["final"] and rnd >= 2:
                    final = True; break
                if rc["final"] and rnd == 1:
                    # ★ at least two rounds: the first forecast is the first time the agent sees any of this, and a
                    #   plan waved through on the round it was written has not been through a dialogue at all.
                    parts.append("### The driver: a plan cannot be finalised in the first round; one more round.")
            chosen = plans[0] if plans else None
            if chosen is not None:
                runner.adopt(chosen, turn)
            if new_log:
                dec_log = new_log
            ms = runner.milestone_check(turn, state_summary(sg, a.ego)) if runner is not None else ""
            extra = ("\n\n".join(parts)
                     + "\n\n## THE PLAN YOU HAVE ADOPTED (the driver will run its production queue every turn)\n"
                     + (render_plan(chosen) if chosen else "(none)")
                     + "\n\nGive this turn's orders so that this plan starts moving: the units it needs on their way, "
                       "anything it buys now, the rates and research it sets.  Cities are set to their queue "
                       "automatically; name a city only if you want something other than its queue this turn.")
            if not rounds:
                print(f"turn {turn}: no plan could be read at this planning point; keeping the previous plan", flush=True)
            rec_wm = {"kind": "plan", "arm": a.arm, "memo": mu, "rounds": rounds, "letters": list(LAB[:len(plans)]),
                      "final": final, "changes": changes, "final_plan": chosen, "n_plans_first": len(pw["plans"]),
                      "write_llm": {k: v for k, v in pw["llm"].items() if k != "think"},
                      "seconds": time.time() - tp}
            if ms:
                extra = ms + "\n\n" + extra
        elif a.arm in ("dlg6", "dlg7") and (turn - a.start) % a.wm_every == 0:
            # ★ WM v6 (doc 17 §4.7): routes carry BINDING contracts, the forecaster is given the measured base rates,
            #   the scoring rule and the end turn, and must answer the SAFE WINDOW / GROWTH question first.
            consult = True; tp = time.time(); last_dlg_turn = turn
            from bench.agentplay.dialogue import LETTERS as DLG_LETTERS
            siege_on = bool(view.get("siege"))

            def _with_contract(rs):
                out_ = []
                for r_ in rs:
                    spec = parse_contract(r_.get("contract"), rules)
                    txt = r_["plan"]
                    if spec:
                        q_ = spec.get("build") or {}
                        txt += ("\nBINDING CONTRACT (the driver enforces this for the next "
                                f"{a.contract_turns} turns; orders that break it are rejected): "
                                + (f"tax/science/luxury {'/'.join(map(str, spec['rates']))}; " if spec.get("rates") else "")
                                + (f"research goal {spec['research_goal']}; " if spec.get("research_goal") else "")
                                + (f"government {spec['government']}; " if spec.get("government") else "")
                                + ("completes " + ", ".join(f"{v_} {k_}" for k_, v_ in q_.items() if v_) + "; " if q_ else "")
                                + ("starts NOTHING in: " + ", ".join(k_ for k_ in ("settlers", "workers", "defenders", "army", "buildings", "wonder")
                                                                     if int(q_.get(k_, 0)) <= 0) + "; " if q_ else "")
                                + (f"founds {spec['found_cities']} cities; " if spec.get("found_cities") else "")
                                + (f"spends gold down to {spec['max_treasury']}" if spec.get("max_treasury") is not None else ""))
                    out_.append({**r_, "plan": txt[:2400], "contract_spec": spec})
                return out_

            mu = dlg.memo_update(turn, ego.name)
            pr = agent.plan_routes6(view["report"], view["panel"], note, turn, siege_on, a.contract_turns, a.dlg_rounds)
            routes = _with_contract(pr["routes"]) or [{"name": "current course", "plan": (note or "Course: continue as before.")[:1800], "contract_spec": None}]
            if a.arm == "dlg7" and len(routes) > 1:
                # ★ v11: 66% of the v9 commits went to route A, which was always the agent's own opening
                #   intention.  The letters are shuffled so the label carries no information about authorship.
                rng.shuffle(routes)
            rounds = []; parts = []; used = 0
            for rnd in range(1, a.dlg_rounds + 1):
                letters = list(DLG_LETTERS[used:used + len(routes)]); used += len(routes)
                try:
                    rr = route_risk(sg, rules, view["board"], view["geo"], ego, view["rel"], routes, letters)
                except Exception as e:                                   # noqa: BLE001
                    rr = []; print(f"turn {turn}: route_risk failed ({type(e).__name__}: {e})", flush=True)
                risk_text = "\n".join(rr)
                f = dlg.forecast(turn, ego.name, routes, letters, rounds, siege_text="\n".join(view.get("siege") or []),
                                 sites=view.get("sites"), risk_text=risk_text)
                routes_txt = "\n".join(f"- {L_}: {r_['name']} -- {r_['plan']}" for L_, r_ in zip(letters, routes))
                rounds.append({"round": rnd, "routes": routes, "letters": letters, "forecast": f})
                parts.append(f"### Round {rnd}: your routes\n{routes_txt}\n\n### Round {rnd}: the forecaster\n{f['text']}")
                if rnd == a.dlg_rounds:
                    break
                rc = agent.revise_or_commit6(view["report"], view["panel"], note, turn, "\n\n".join(parts), rnd, a.dlg_rounds)
                rounds[-1]["decision"] = {k: v for k, v in rc.items() if k != "llm"}
                rounds[-1]["decision_llm"] = {k: v for k, v in rc["llm"].items() if k != "think"}
                parts.append(f"### Round {rnd}: your reply\n{rc['action']}" + (f" route {rc['route']}" if rc.get("route") else "") + f" -- {rc['why']}")
                if rc["action"] != "revise":
                    break
                routes = _with_contract(rc["routes"])
            all_letters = [L_ for r_ in rounds for L_ in r_["letters"]]
            transcript = f"## Planning dialogue (turn {turn})\n" + "\n\n".join(parts)
            pick = None
            if a.arm == "dlg7":
                pk = agent.choose_route(view["report"], view["panel"], note, turn, transcript, all_letters, a.contract_turns)
                pick = pk["route"]
                chosen_plan = ""
                for r_ in rounds:
                    if pick in r_["letters"]:
                        chosen_plan = r_["routes"][r_["letters"].index(pick)]["plan"]
                        break
                extra = (transcript + f"\n\nYou have just committed to route {pick}. Your reason: {pk['why']}\n"
                         f"That route is your course until turn {turn + a.contract_turns}:\n{chosen_plan}\n\n"
                         "Write it into your strategy_note and give this turn's orders in line with it.  The driver "
                         "will show this course to you every turn and track what you actually build against it; it "
                         "enforces nothing, but departing from it is a decision you should state.") if pick else (
                         transcript + "\n\nCommit to one route, write it into your strategy_note, and give this turn's orders.")
            else:
                extra = (transcript
                         + f"\n\nDecide now: commit to one route (letters {', '.join(all_letters)}) -- put its letter in \"chosen_route\", write the "
                           "long-term plan you commit to into strategy_note, and give this turn's orders."
                         + f"  ★ The contract of the route you choose becomes BINDING until turn {turn + a.contract_turns}.")
            rec_wm = {"kind": "dialogue", "v6": True, "arm": a.arm, "siege": siege_on, "memo": mu,
                      "pick": pick, "pick_llm": ({k: v for k, v in pk["llm"].items() if k != "think"} if a.arm == "dlg7" and pick is not None else None),
                      "pick_why": (pk["why"] if a.arm == "dlg7" else ""),
                      "mandates": pr.get("mandates"), "route_calls": pr.get("calls"),
                      "routes_llm": {k: v for k, v in pr["llm"].items() if k != "think"},
                      "rounds": rounds, "letters": all_letters, "seconds": time.time() - tp}
        elif a.arm == "dlg" and ((turn - a.start) % a.wm_every == 0 or
                               (a.dlg_alerts and turn - last_dlg_turn >= a.dlg_alert_cooldown and
                                (any(x.startswith(("ENEMY military", "cities with NO military")) for x in view.get("advisor", []))
                                 or any("last turn:" in x and ("city_transferred" in x or "city_destroyed" in x) for x in view["alerts"])))):
            # ★ planning dialogue (doc 17 §4.3): memo update -> routes -> forecast -> (revise -> forecast)* -> commit
            consult = True; tp = time.time(); last_dlg_turn = turn
            from bench.agentplay.dialogue import LETTERS as DLG_LETTERS
            mu = dlg.memo_update(turn, ego.name)
            pr = agent.plan_routes(view["report"], view["panel"], note, turn)
            routes = pr["routes"] or [{"name": "current course", "plan": (note or "Course: continue as before.")[:1800]}]
            rounds: list[dict] = []; parts: list[str] = []; used = 0
            for rnd in range(1, a.dlg_rounds + 1):
                letters = list(DLG_LETTERS[used:used + len(routes)]); used += len(routes)
                f = dlg.forecast(turn, ego.name, routes, letters, rounds, siege_text="\n".join(view.get("siege") or []))
                routes_txt = "\n".join(f"- {L_}: {r['name']} -- {r['plan']}" for L_, r in zip(letters, routes))
                rounds.append({"round": rnd, "routes": routes, "letters": letters, "forecast": f})
                parts.append(f"### Round {rnd}: your routes\n{routes_txt}\n\n### Round {rnd}: the forecaster\n{f['text']}")
                if rnd == a.dlg_rounds:
                    break
                rc = agent.revise_or_commit(view["report"], view["panel"], note, turn, "\n\n".join(parts), rnd, a.dlg_rounds)
                rounds[-1]["decision"] = {k: v for k, v in rc.items() if k != "llm"}
                rounds[-1]["decision_llm"] = {k: v for k, v in rc["llm"].items() if k != "think"}
                parts.append(f"### Round {rnd}: your reply\n{rc['action']}" + (f" route {rc['route']}" if rc.get("route") else "") + f" -- {rc['why']}")
                if rc["action"] != "revise":
                    break
                routes = rc["routes"]
            all_letters = [L_ for r in rounds for L_ in r["letters"]]
            extra = (f"## Planning dialogue (turn {turn})\n" + "\n\n".join(parts)
                     + f"\n\nDecide now: commit to one route (letters {', '.join(all_letters)}) -- put its letter in \"chosen_route\", write the "
                       "long-term plan you commit to into strategy_note (it stays in force until you change it), and give this turn's orders.")
            rec_wm = {"kind": "dialogue", "memo": mu, "routes_llm": {k: v for k, v in pr["llm"].items() if k != "think"},
                      "rounds": rounds, "letters": all_letters, "seconds": time.time() - tp}
        elif wm is not None and a.arm == "wm_outlook" and (turn - a.start) % a.wm_every == 0:
            # outlook mode: no candidate plans; one forecast for the agent's CURRENT plan, shown as an outlook every wm_every turns
            consult = True; tp = time.time()
            plan_txt = note or "Development: continue as now.\nWar aim: none."
            adv = wm.advise(hist, turn, ego.name, [{"name": "your current plan", "plan": plan_txt[:1500]}])
            extra = ("## World-model outlook if you continue your current plan (quantiles at +20 and +40 turns; *_gain = change from now)\n"
                     + adv["text"] + "\n\nRead the other civilizations' military and n_enemies lines as threat indicators; if your own n_cities "
                     "p10 is below your current city count, the model expects you may lose cities -- adjust defence or seek a ceasefire.")
            rec_wm = {"candidates": [{"name": "current plan", "plan": plan_txt[:1500]}], "wm_calls": adv.get("calls"), "wm_prompt_chars": adv.get("prompt_chars"),
                      "forecast_text": adv["text"], "seconds": time.time() - tp}
        elif wm is not None and a.arm == "wm2" and (turn - a.start) % a.wm_every == 0:
            # v2: 3 diverse candidates -> ONE comparative forecast (dense frames, +10/+20, 3 samples, calibration appendix) -> commit
            consult = True; tp = time.time()
            prop = agent.propose(view["report"], view["panel"], note, turn, v2=True)
            cands = prop["candidates"]
            if cands:
                adv = wm.advise_multi(hist, turn, ego.name, cands, wm_history, n_samples=a.wm_samples)
                cal_lines = []
                for h_ in wm_history[-3:]:
                    dt = turn - h_["turn"]; lv = adv.get("levels_ego") or {}
                    cells = []
                    for v in ("score_gain", "n_cities", "gold", "n_military_units"):
                        pv = h_["p50"].get((v, dt))
                        if pv is None or not lv:
                            continue
                        from bench.spec_v2 import LEVEL_OF
                        act = lv.get(LEVEL_OF[v]); act = act - h_["levels"].get(LEVEL_OF[v], 0) if v.endswith("_gain") and act is not None else act
                        cells.append(f"{v}: forecast {pv:g} vs actual {act}")
                    if cells:
                        cal_lines.append(f"- turn {h_['turn']} ({h_['chosen']}): " + "; ".join(cells))
                extra = ("## Candidate plans you proposed\n" + "\n".join(f"- {L_}: {c['name']} -- {c['plan']}" for L_, c in zip(adv["letters"], cands))
                         + "\n\n## World-model forecast, candidates side by side (+10 / +20 turns)\n" + adv["text"]
                         + ("\n\n## How the world model's earlier forecasts turned out\n" + "\n".join(cal_lines) if cal_lines else "")
                         + "\n\nCOMMIT to one candidate (you may adjust details): put its letter in \"chosen_candidate\", write the plan into your "
                           "strategy note, and give this turn's orders accordingly.")
                rec_wm = {"candidates": cands, "letters": adv["letters"], "propose_llm": {k: v for k, v in prop["llm"].items() if k != "think"},
                          "wm_calls": adv["calls"], "wm_prompt_chars": adv["prompt_chars"], "forecast_text": adv["text"], "ego_p50": {L_: {f"{v}|{h}": x for (v, h), x in d.items()} for L_, d in adv["ego_p50"].items()},
                          "levels_ego": adv["levels_ego"], "sections_parsed": adv["sections_parsed"], "sections_expected": adv["sections_expected"], "seconds": time.time() - tp}
                pending_wm = (cands, adv)
            else:
                extra = ""; rec_wm = {"candidates": [], "propose_llm": {k: v for k, v in prop["llm"].items() if k != "think"}, "seconds": time.time() - tp}; pending_wm = None
        elif wm is not None and (turn - a.start) % a.wm_every == 0:
            consult = True
            tp = time.time()
            prop = agent.propose(view["report"], view["panel"], note, turn)
            adv = wm.advise(hist, turn, ego.name, prop["candidates"]) if prop["candidates"] else {"text": "(no candidates were proposed)", "calls": []}
            extra = ("## Candidate plans you proposed and the world model's forecasts\n"
                     + "\n".join(f"- {c['name']}: {c['plan']}" for c in prop["candidates"])
                     + "\n\nForecasts (quantiles of each quantity at +20 and +40 turns; *_gain = change from now):\n" + adv["text"]
                     + "\n\nChoose (or blend) a plan, write it into your strategy note, and give this turn's orders accordingly.")
            rec_wm = {"candidates": prop["candidates"], "propose_llm": {k: v for k, v in prop["llm"].items() if k != "think"},
                      "wm_calls": adv.get("calls"), "wm_prompt_chars": adv.get("prompt_chars"), "forecast_text": adv["text"], "seconds": time.time() - tp}
        else:
            rec_wm = None
        if a.arm == "replay" and forced_obj is not None and turn == a.from_turn:
            from bench.agentplay.agent import to_plan
            consult = True
            plan, fb_parse = to_plan(forced_obj)
            note = str(forced_obj.get("strategy_note") or note)
            rec_wm = None; skip_left = 0
            rec.update(orders=forced_obj, parse_ok=True, end_turns=1, note=note, replayed_candidate=a.cand)
            forced_obj = None
        elif a.arm == "bok" and (turn - a.start) % a.wm_every == 0:
            # ★ best-of-K: K decisions sampled from the UNCHANGED agent prompt; the driver picks one (wm / random / self);
            #   the agent never sees the forecaster.  Non-decision turns are identical to the baseline.
            consult = True; tp = time.time()
            with cf.ThreadPoolExecutor(max_workers=a.k) as ex:
                ds = list(ex.map(lambda _: agent.decide(view["report"], view["panel"], note, turn), range(a.k)))
            ds = [d_ for d_ in ds if d_["parse_ok"]] or ds[:1]
            letters = "ABCDE"[:len(ds)]
            cand_texts = [decision_as_plan(d_, ego.units, rules) for d_ in ds]
            pick = 0; sel_info: dict = {"select": a.select, "k": len(ds)}
            if a.select == "wm" and len(ds) > 1:
                cands = [{"name": f"decision {L_}", "plan": t} for L_, t in zip(letters, cand_texts)]
                adv = wm.advise_multi(hist, turn, ego.name, cands, wm_history, n_samples=a.wm_samples)
                utils = {L_: utility(adv["ego_q"].get(L_, {}), adv.get("levels_ego") or {}) for L_ in letters}
                best_L = max(letters, key=lambda L_: utils[L_][0])
                if utils[best_L][0] == float("-inf"):
                    best_L = rng.choice(letters); sel_info["fallback"] = "no forecast -> random"
                pick = letters.index(best_L)
                sel_info.update(utilities={L_: [utils[L_][0], utils[L_][1]] for L_ in letters}, forecast_text=adv["text"], wm_calls=adv["calls"],
                                ego_q=adv["ego_q"], levels_ego=adv["levels_ego"], sections_parsed=adv["sections_parsed"], sections_expected=adv["sections_expected"])
                wm_history.append({"turn": turn, "chosen": f"{best_L}: decision", "p50": {tuple([k_.split("|")[0], int(k_.split("|")[1])]): v[2] for k_, v in adv["ego_q"].get(best_L, {}).items()},
                                   "levels": dict(adv.get("levels_ego") or {})})
            elif a.select == "self" and len(ds) > 1:
                rk = agent.rank(view["report"], view["panel"], note, turn, cand_texts)
                pick = letters.index(rk["best"]) if rk["best"] else rng.randrange(len(ds))
                sel_info.update(best=rk["best"], rank_llm={k_: v for k_, v in rk["llm"].items() if k_ != "think"})
            else:
                pick = rng.randrange(len(ds))
            d = ds[pick]
            sel_info.update(chosen=letters[pick], candidates=cand_texts, seconds=time.time() - tp,
                            cand_llm=[{k_: v for k_, v in d_["llm"].items() if k_ not in ("think",)} for d_ in ds],
                            cand_orders=[d_.get("obj") for d_ in ds])
            rec_wm = sel_info
            plan, fb_parse, note = d["plan"], d["feedback"], d["note"]
            skip_left = 0
            rec.update(llm=d["llm"], orders=d["obj"], parse_ok=d["parse_ok"], end_turns=1, note=note, user_chars=d["user_chars"], bok=sel_info)
        elif consult:
            d = agent.decide(view["report"], view["panel"], note, turn, (course_block + "\n" + extra) if course_block else extra)
            plan, fb_parse, note = d["plan"], d["feedback"], d["note"]
            skip_left = d["end_turns"] - 1
            rec.update(llm=d["llm"], orders=d["obj"], parse_ok=d["parse_ok"], end_turns=d["end_turns"], note=note, user_chars=d["user_chars"])
            if rec_wm is not None:
                rec["wm"] = rec_wm
            if a.arm in ("dlg6", "dlg7") and rec_wm is not None:
                ch = rec_wm.get("pick") or str((d.get("obj") or {}).get("chosen_route", "")).strip().upper()[:1]
                if ch not in rec_wm["letters"]:
                    ch = rec_wm["rounds"][-1]["letters"][0]; rec["wm"]["chosen_fallback"] = True
                cm = dlg.commit(turn, ch, rec_wm["rounds"], ego.name)
                rec["wm"]["chosen"] = cm["letter"]; rec["wm"]["standing"] = cm["standing"]
                rec["wm"]["seconds"] = time.time() - tp
                for r_ in rec_wm["rounds"]:
                    if cm["letter"] in r_["letters"]:
                        rt_ = r_["routes"][r_["letters"].index(cm["letter"])]
                        contract.commit(turn, cm["letter"], rt_["name"], rt_.get("contract_spec"))
                        course = {"turn": turn, "letter": cm["letter"], "name": rt_["name"],
                                  "plan": rt_["plan"], "mandate": rt_.get("mandate", "")}
                        break
            if a.arm == "dlg" and rec_wm is not None:
                ch = str((d.get("obj") or {}).get("chosen_route", "")).strip().upper()[:1]
                if ch not in rec_wm["letters"]:
                    ch = rec_wm["rounds"][-1]["letters"][0]; rec["wm"]["chosen_fallback"] = True
                cm = dlg.commit(turn, ch, rec_wm["rounds"], ego.name)
                rec["wm"]["chosen"] = cm["letter"]; rec["wm"]["standing"] = cm["standing"]
                rec["wm"]["seconds"] = time.time() - tp
            if a.arm == "wm2" and rec_wm is not None and pending_wm is not None:
                cands, adv = pending_wm
                ch = str((d.get("obj") or {}).get("chosen_candidate", "")).strip().upper()[:1]
                if ch not in adv["letters"]:
                    ch = adv["letters"][0]
                    rec["wm"]["chosen_fallback"] = True
                rec["wm"]["chosen"] = ch
                wm_history.append({"turn": turn, "chosen": f"{ch}: {cands[adv['letters'].index(ch)]['name'][:40]}",
                                   "p50": dict(adv["ego_p50"].get(ch, {})), "levels": dict(adv.get("levels_ego") or {})})
                pending_wm = None
        else:
            plan, fb_parse = TurnPlan(), []
            skip_left -= 1
            rec.update(note=note)
        if contract is not None and consult:
            fb_c = contract.enforce(plan, turn, rec.get("orders"), bool(view.get("siege")), set(ego.known_techs))
            if fb_c:
                fb_parse = list(fb_parse) + fb_c
        if contract is not None:
            rec["contract"] = contract.as_record(turn)
        if runner is not None and runner.plan:
            rec["plan_done"] = dict(runner.done)
            rec["decision_log"] = dec_log
            pc = str((rec.get("orders") or {}).get("plan_change") or "").strip()
            if pc:
                rec["plan_change"] = pc[:400]
        if course:
            rec["course"] = course
            cc = str((rec.get("orders") or {}).get("course_change") or "").strip()
            if cc:
                rec["course_change"] = cc[:400]
        if runner is not None and runner.plan:
            named = {c.name for c in plan.cities}
            for cc in runner.buy_orders(ego):
                if cc.name not in named:
                    plan.cities.append(cc); named.add(cc.name)
            for cc in runner.default_cities(ego):
                if cc.name not in named:
                    plan.cities.append(cc)
            b = (runner.plan.get("budget") or {})
            if plan.rates is None and b.get("rates"):
                plan.rates = tuple(b["rates"])
            r_ = (runner.plan.get("research") or {})
            if not plan.research_goal and r_.get("goal"):
                plan.research_goal = r_["goal"]
        lua, fb, warn = plan_to_edits(sg, ego, rules, view["board"], plan, turn, standing)
        res = None
        game_over = ""
        for attempt in range(5):
            try:
                res = run_turn(out / "work", sg, a.ego, lua, turn); break
            except RuntimeError as e:                 # transient engine failure (see engine.run_turn): wait and retry the whole step
                # ★ 2026-09-19: at the 150-turn horizon the game can simply END -- `check_for_game_over` fires
                #   (conquest victory once the last rival civ is gone; the remaining players are barbarians), the
                #   server removes every player and writes no T+1 autosave.  That is a WIN, not a failure, and the
                #   old code exited 3 and lost the game -- which would have dropped the best games of whichever arm
                #   plays better, i.e. a bias exactly against the effect we are measuring.
                #   The engine's own log is no help: when the load ends the game the server falls back to pregame
                #   and `start` begins a FRESH game, which overwrites work/game.log.  So decide it from the
                #   savegame itself, by the engine's rule: barbarians (nation "Pirate" / "Barbarian") are not
                #   civilizations, and the game is over once at most one civilization is still alive.
                alive = [sec for sec in sg.players()
                         if str(sg.get(sec, "is_alive", "")).upper() not in ("FALSE", "0")
                         and str(sg.get(sec, "nation", "")) not in ("Pirate", "Barbarian")]
                if len(alive) <= 1:
                    won = alive == [f"player{a.ego}"]
                    game_over = ("conquest victory for the ego" if won else
                                 ("the ego is the only civilization left" if alive else "no civilization left"))
                    if not won and alive:
                        game_over = f"the game ended with {sg.get(alive[0], 'name')} as the last civilization"
                    break
                print(f"turn {turn}: engine step failed ({str(e)[:160]}); retrying in 120 s", flush=True); time.sleep(120)
        if res is None and game_over:
            print(f"turn {turn}: THE GAME ENDED -- {game_over}", flush=True)
            rec.update(game_over=game_over, feedback_next=[], standing={str(k): v for k, v in standing.items()},
                       engine_s=0.0, wall_s=time.time() - t0, save=str(save), skip_left=skip_left,
                       state=state_summary(Savegame.load(save), a.ego), n_lua=len(lua))
            open(log_path, "a").write(json.dumps(rec, ensure_ascii=False) + "\n")
            break
        if res is None:
            print(f"turn {turn}: giving up after 5 step attempts", flush=True); sys.exit(3)
        if not res.ok:
            print(f"turn {turn}: ENGINE FAILURE {res.feedback}", flush=True)
            rec.update(engine_error=res.feedback); open(log_path, "a").write(json.dumps(rec, ensure_ascii=False) + "\n")
            sys.exit(2)
        fb_all = fb_parse + fb + feedback_from_results(res, ego)
        shutil.copy(res.save, out / "saves" / res.save.name)
        with open(out / "events.jsonl", "a") as fh:                    # engine events of this turn (WM arm builds INTERVAL blocks from them)
            fh.write(json.dumps({"turn": turn, "events": res.events}, ensure_ascii=False) + "\n")
        save = out / "saves" / res.save.name
        events = res.events
        if contract is not None:
            contract.observe(res.events, a.ego)
        if runner is not None:
            runner.observe(res.events, a.ego)
        st = state_summary(Savegame.load(save), a.ego)
        rec.update(feedback_next=fb_all, standing={str(k): v for k, v in standing.items()}, engine_s=res.seconds, wall_s=time.time() - t0,
                   save=str(save), skip_left=skip_left, state=st, n_lua=len(lua))
        open(log_path, "a").write(json.dumps(rec, ensure_ascii=False) + "\n")
        n_rej = sum(1 for f in fb_all if f.startswith(("REJECTED", "FAILED", "COULD NOT")))
        print(f"T{turn:>4} {'LLM' if consult else '---'} score {st['score']:>4} gold {st['gold']:>5} cities {st['n_cities']:>2} units {st['n_units']:>3} techs {st['n_techs']:>2} "
              f"| orders {len(plan.units)}u/{len(plan.cities)}c rejected {n_rej} | {rec.get('llm', {}).get('completion_tokens', 0):>6} tok {rec.get('llm', {}).get('seconds', 0):>5.0f}s | {time.time() - t0:.0f}s", flush=True)
        feedback = fb_all
        turn += 1
    # summary
    recs = [json.loads(l) for l in open(log_path)]
    traj = [(r["turn"] + 1, r["state"]["score"]) for r in recs]
    ref = mainline_reference(a.world, a.ego, [a.start] + [t for t, _ in traj if t % 10 == 0 or t == end])
    ended = next((r.get("game_over") for r in recs if r.get("game_over")), "")
    summ = {"world": a.world, "ego": a.ego, "ego_name": ego_name, "start": a.start, "end": end, "arm": a.arm, "model": a.model, "wm_model": a.wm_model or a.model,
            "game_over": ended, "ended_turn": (recs[-1]["turn"] if ended else None),
            "final": recs[-1]["state"], "score_traj": traj, "mainline_ref": {str(k): v for k, v in sorted(ref.items())},
            "n_turns": len(recs), "n_consulted": sum(1 for r in recs if r["consulted"]),
            "prompt_tokens": sum(r.get("llm", {}).get("prompt_tokens", 0) for r in recs),
            "completion_tokens": sum(r.get("llm", {}).get("completion_tokens", 0) for r in recs),
            "parse_failures": sum(1 for r in recs if r["consulted"] and not r.get("parse_ok")),
            "rejected_orders": sum(sum(1 for f in r["feedback_next"] if f.startswith(("REJECTED", "FAILED", "COULD NOT"))) for r in recs),
            "wall_s": sum(r["wall_s"] for r in recs)}
    json.dump(summ, open(out / "summary.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps({k: summ[k] for k in ("final", "n_consulted", "parse_failures", "rejected_orders", "wall_s")}, ensure_ascii=False))
    print("mainline AI reference:", summ["mainline_ref"])


if __name__ == "__main__":
    main()
