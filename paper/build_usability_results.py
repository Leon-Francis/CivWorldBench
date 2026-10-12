"""Reproduce the paper's selector comparisons and case-study evidence."""
import hashlib
import json
from pathlib import Path
import statistics

from bench.agentplay.probes.analyse_usability_tonight import estimate
from bench.agentplay.probes.usability_replication import CONDITIONS, score_pair
from bench.agentplay.run_game import Savegame
from bench.agentplay.observe import Ego

ROOT = Path(__file__).resolve().parents[1]
GAMES = ROOT / "bench/agentplay/games"
SOURCE = GAMES / "usability_extension_20260924/validation36"
REPEAT = GAMES / "usability_replication_20260925"
CASE = "a20_s500198_t173"
BAD_CASE = "a33_s500075_t172"


def selected_score(training, testing, pool):
    best = max(statistics.mean(training[condition]) for condition in pool)
    return statistics.mean(statistics.mean(testing[condition]) for condition in pool
                           if statistics.mean(training[condition]) == best)


def case_dialogue():
    selections = {
        "initial_loss": ("initial", ("plan", "steps", 2),
                         "Icara\u00ed and Siri Jibe have empty garrisons and P=1.00 regardless of actions; accept their loss."),
        "initial_build": ("initial", ("plan", "steps", 7),
                          "Do NOT build anything at Icara\u00ed or Siri Jibe (likely lost)."),
        "initial_buy": ("initial", ("plan", "steps", 9), "Do NOT buy anything."),
        "wm_risk": ("opus", ("advice", "forecasts", 0, "assumptions_audit"),
                    "'P=1.00' is P(loss GIVEN assault), not P(assault)"),
        "wm_arrival": ("opus", ("advice", "forecasts", 0, "assumptions_audit"),
                       "Realistic near-term capture risk is therefore moderate, not certain."),
        "wm_defender": ("opus", ("advice", "forecasts", 0, "advice"),
                        "an empty city is captured by any single unit, a walled city with even one defender forces a fight."),
        "wm_capuan": ("opus", ("advice", "forecasts", 0, "advice"),
                      "Reconsider the Capuan triage: it is the most likely city to be assaulted NEXT turn (adjacent Riflemen), and one more fortified Musketeer takes P(loss|assault) from 0.81 to 0.03 per the calculator"),
        "wm_buy": ("opus", ("advice", "forecasts", 0, "advice"),
                   "consider buying it if gold allows (~60-90 gold) rather than hoarding 83 gold that will drip away anyway."),
        "agent_buy": ("opus", ("plan", "steps", 0),
                      "T173 BUY: Buy Capuan Musketeers (20/40 shields remaining)."),
        "agent_move": ("opus", ("plan", "steps", 1),
                       "T173 DEFENSE: Order Musketeer 520 from Par\u00e1yba (23,2) to Icara\u00ed (21,1)."),
        "agent_build": ("opus", ("plan", "steps", 2),
                        "switch from Colosseum [45/70 stored] to Musketeers [40]."),
    }
    return load_case_dialogue(CASE, "opus", selections, 193)


def bad_case_dialogue():
    selections = {
        "initial_research": ("initial", ("plan", "intent"), "PRIMARY LEVER: tech at 69 bulbs/turn."),
        "initial_budget": ("initial", ("plan", "intent"),
                           "Treasury 665, -13/turn, ends ~T210 at ~160"),
        "wm_budget": ("sol", ("advice", "forecasts", 0, "advice"),
                      "Begin raising taxes if the treasury falls below roughly 300-350 or if the observed deficit remains above 10 gold/turn; waiting for 200 risks forced building sales."),
        "wm_defense": ("sol", ("advice", "forecasts", 0, "advice"),
                       "Oviedo and Santander particularly benefit from City Walls or a second defender"),
        "agent_budget": ("sol", ("plan", "intent"),
                         "Must raise taxes to 60/30/10 by ~T182-185 when treasury nears 400 to prevent forced building sales."),
        "agent_defense": ("sol", ("plan", "intent"),
                          "DEFENSE: City Walls (0 upkeep, +100% defense) on Oviedo, Santander, Valencia"),
        "agent_tax_check": ("sol", ("plan", "steps", 2),
                            "T182 TREASURY CHECK: if gold below 450 or deficit exceeds 20/turn, raise rates to **60/30/10**"),
    }
    return load_case_dialogue(BAD_CASE, "sol", selections, 192)


def load_case_dialogue(anchor_id, adviser, selections, end_turn):
    paths = {condition: SOURCE / "plans" / anchor_id / (condition + ".json")
             for condition in ("initial", "self", adviser)}
    paths["anchor"] = SOURCE / "anchors" / (anchor_id + ".json")
    sources = {name: json.loads(path.read_text()) for name, path in paths.items()}
    assert sources["anchor"]["initial_plan"] == sources["initial"]["plan"]
    assisted = sources[adviser]
    assert json.loads(assisted["advice_calls"][0]["answer"])["forecasts"] == assisted["advice"]["forecasts"]
    assert json.loads(assisted["calls"][0]["answer"])["plans"][0] == assisted["plan"]
    excerpts = {}
    for name, (source, field, quote) in selections.items():
        value = sources[source]
        for key in field:
            value = value[key]
        if quote not in value:
            raise ValueError(f"Case excerpt does not match source: {name}")
        excerpts[name] = {"source": str(paths[source].relative_to(ROOT)), "field": list(field), "text": quote}
    anchor = sources["anchor"]
    saved = Savegame.load(Path(anchor["save"]))
    return {"excerpts": excerpts,
            "source_hashes": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                              for path in paths.values()},
            "initial_state": {"turn": anchor["turn"], "score": int(saved.sections["score0"]["total"]),
                              "cities": len(Ego(saved, 0).cities), "gold": int(saved.sections["player0"]["gold"])},
            "forecast": next(outcome for outcome in assisted["advice"]["forecasts"][0]["outcomes"]
                             if outcome["turn"] == end_turn)}


def collect_case_runs(anchor_id):
    case = {}
    for condition in CONDITIONS:
        runs = []
        for directory in (SOURCE, REPEAT):
            for repeat in (0, 1):
                path = directory / "branches" / anchor_id / condition / str(repeat) / "turns.jsonl"
                rows = [json.loads(line) for line in path.read_text().splitlines()]
                saved = Savegame.load(Path(rows[-1]["save"]))
                runs.append({"records": str(path.relative_to(ROOT)), "records_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                             "final_state": rows[-1]["state"], "score_components": saved.sections["score0"],
                             "remaining_cities": list(Ego(saved, 0).cities),
                             "score_trajectory": [[row["state"]["turn"], row["state"]["score"]] for row in rows],
                             "state_trajectory": [row["state"] for row in rows],
                             "feedback_by_turn": {str(row["turn"]): row["feedback_next"] for row in rows},
                             "first_orders": rows[0]["orders"], "first_feedback": rows[0]["feedback_next"]})
        case[condition] = runs
    return case


def tex_escape(value):
    replacements = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
                    "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
                    "^": r"\textasciicircum{}", "|": r"\textbar{}", "\u00ed": r"\'{i}",
                    "\u00e1": r"\'{a}", "\u00e3": r"\~{a}"}
    return "".join(replacements.get(character, character) for character in value)


def render_case_dialogue(dialogue, runs):
    def quote(name):
        return "``" + tex_escape(dialogue["excerpts"][name]["text"]) + "''"

    state = dialogue["initial_state"]
    forecast = dialogue["forecast"]
    quantiles = ", ".join(str(value) for value in forecast["score_p10_p50_p90"])
    lines = [r"\begin{figure*}[ht]", r"\begin{promptbox}[A decision reversal: accept city losses or attempt targeted defense?]",
             rf"\textbf{{Checkpoint.}} Turn {state['turn']}: "
             rf"score {state['score']}, {state['cities']} cities, {state['gold']} gold. "
             r"One consultation with Opus, followed by a Flash-Next revision and 20 turns of execution.",
             r"\par\textbf{Agent (Flash-Next): original plan.}\par",
             quote("initial_loss") + r"\par " + quote("initial_build") + " " + quote("initial_buy"),
             r"\par\textbf{World model (Opus): forecast and advice.}\par",
             rf"Original-plan forecast at T193: score $[p_{{10}},p_{{50}},p_{{90}}]=[{quantiles}]$; "
             rf"$P(\text{{any city lost}})={forecast['p_any_city_lost']:.2f}$.\par",
             quote("wm_risk") + r" $[\ldots]$ " + quote("wm_arrival") + r"\par",
             quote("wm_defender") + r"\par",
             "``" + tex_escape(dialogue["excerpts"]["wm_capuan"]["text"]) + r" $[\ldots]$ "
             + tex_escape(dialogue["excerpts"]["wm_buy"]["text"]) + "''",
             r"\par\textbf{Agent (Flash-Next): revised plan.}\par",
             quote("agent_buy") + r"\par " + quote("agent_move") + r"\par",
             "Siri Jibe: " + quote("agent_build"),
             r"\par\textbf{Execution feedback (first execution, T173).}\par"]
    feedback = ["OK bought Musketeers in Capuan for 60 gold (completes at turn change)",
                "OK Siri Jibe now builds Musketeers (half the stored shields lost: category change)",
                "moved unit 520 Musketeers to 21,1"]
    for entry in feedback:
        assert entry in runs["opus"][0]["first_feedback"]
        lines.append(r"{\footnotesize\ttfamily " + tex_escape(entry) + r"}\par")
    lines += [r"\textbf{Measured score trajectory (mean of four executions per frozen plan).}\par",
              r"{\centering\setlength{\tabcolsep}{8pt}", r"\begin{tabular}{lrrrrr}", r"\toprule",
              r"Plan & T173 & T178 & T183 & T188 & T193 \\", r"\midrule"]
    for condition, name in (("initial", "Original"), ("self", "Self-review"), ("opus", "Opus-assisted")):
        scores = [state["score"]] + [statistics.mean(dict(run["score_trajectory"])[turn] for run in runs[condition])
                                    for turn in (178, 183, 188, 193)]
        lines.append(name + " & " + " & ".join(f"{value:.2f}" for value in scores) + r" \\")
    benefit = statistics.mean(run["final_state"]["score"] for run in runs["opus"]) - statistics.mean(
        run["final_state"]["score"] for run in runs["initial"])
    lines += [r"\bottomrule\end{tabular}\par}",
              rf"At T193, the revised plan reduces mean score loss by \textbf{{{benefit:.2f} points}} relative to the original.",
              r"\end{promptbox}",
              r"\caption{Recorded agent--WM exchange and outcomes in the targeted-defense case. "
              r"Quotations preserve model outputs, with $[\ldots]$ marking omissions. "
              r"The forecast concerns the original plan, and 0.81/0.03 are estimates from the shared siege calculator. "
              r"Recorded executions provide outcome evidence.}",
              r"\label{fig:ap_case_dialogue}", r"\end{figure*}", ""]
    return "\n".join(lines)


def render_bad_case_dialogue(dialogue, runs):
    def quote(name):
        return "``" + tex_escape(dialogue["excerpts"][name]["text"]) + "''"

    state = dialogue["initial_state"]
    forecast = dialogue["forecast"]
    quantiles = ", ".join(str(value) for value in forecast["score_p10_p50_p90"])
    lines = [r"\begin{figure*}[ht]", r"\begin{promptbox}[A failed revision: precaution without a realized score benefit]",
             rf"\textbf{{Checkpoint.}} Turn {state['turn']}: "
             rf"score {state['score']}, {state['cities']} cities, {state['gold']} gold. "
             r"One consultation with Sol, followed by a Flash-Next revision and 20 turns of execution.",
             r"\par\textbf{Agent (Flash-Next): original plan.}\par",
             quote("initial_research") + r"\par " + quote("initial_budget"),
             r"\par\textbf{World model (Sol): forecast and advice.}\par",
             rf"Original-plan forecast at T192: score $[p_{{10}},p_{{50}},p_{{90}}]=[{quantiles}]$; "
             rf"$P(\text{{any city lost}})={forecast['p_any_city_lost']:.2f}$.\par",
             quote("wm_budget") + r"\par " + quote("wm_defense"),
             r"\par\textbf{Agent (Flash-Next): revised plan.}\par",
             quote("agent_budget") + r"\par " + quote("agent_defense"),
             r"\par\textbf{Execution feedback (first execution).}\par"]
    for turn, feedback in ((176, "OK rates tax 60% science 30% luxury 10%"),
                           (177, "OK Valencia now builds City Walls"),
                           (179, "OK Oviedo now builds City Walls")):
        assert feedback in runs["sol"][0]["feedback_by_turn"][str(turn)]
        lines.append(rf"T{turn}: {{\footnotesize\ttfamily " + tex_escape(feedback) + r"}\par")
    lines += [r"\textbf{Measured score trajectory (mean of four executions per frozen plan).}\par",
              r"{\centering\setlength{\tabcolsep}{8pt}", r"\begin{tabular}{lrrrrr}", r"\toprule",
              r"Plan & T172 & T177 & T182 & T187 & T192 \\", r"\midrule"]
    for condition, name in (("initial", "Original"), ("self", "Self-review"), ("sol", "Sol-assisted")):
        scores = [state["score"]] + [statistics.mean(dict(run["score_trajectory"])[turn] for run in runs[condition])
                                    for turn in (177, 182, 187, 192)]
        assert all(snapshot["n_cities"] == state["cities"]
                   for run in runs[condition] for snapshot in run["state_trajectory"])
        lines.append(name + " & " + " & ".join(f"{value:.2f}" for value in scores) + r" \\")
    science = {condition: statistics.mean(snapshot["techout"] for run in runs[condition]
                                         for snapshot in run["state_trajectory"])
               for condition in ("initial", "sol")}
    gold = {condition: statistics.mean(run["final_state"]["gold"] for run in runs[condition])
            for condition in ("initial", "sol")}
    minimum_gold = min(snapshot["gold"] for run in runs["initial"] for snapshot in run["state_trajectory"])
    difference = statistics.mean(run["final_state"]["score"] for run in runs["sol"]) - statistics.mean(
        run["final_state"]["score"] for run in runs["initial"])
    assert difference < 0
    lines += [r"\bottomrule\end{tabular}\par}",
              rf"The revision yields \textbf{{{-difference:.2f} fewer score points}} at T192. "
              rf"All three conditions retain all {state['cities']} cities throughout. "
              rf"Mean science output is {science['initial']:.1f} versus {science['sol']:.1f} bulbs/turn "
              rf"(original versus Sol-assisted); final treasury is {gold['initial']:.1f} versus {gold['sol']:.1f}. "
              rf"The original executions never end a turn below {minimum_gold} gold.",
              r"\end{promptbox}",
              r"\caption{The precautionary revision retained more gold but reduced research output and final score. "
              r"Quotations reproduce logged advice and plans. The forecast concerns the original plan, "
              r"while outcomes reflect the complete revision.}",
              r"\label{fig:ap_bad_case_dialogue}", r"\end{figure*}", ""]
    return "\n".join(lines)


def main():
    anchors = json.loads((REPEAT / "manifest.json").read_text())["anchors"]
    assert len({item["world"] for item in anchors}) == 36
    contrasts = {key: [] for key in ("all_vs_initial", "all_vs_uniform", "all_vs_self", "all_vs_no_wm_pool")}
    model_differences = {key: [] for key in ("opus_vs_kimi", "sol_vs_kimi")}
    per_world = []
    cases = {}
    for item in anchors:
        anchor = item["id"]
        old = {condition: score_pair(SOURCE, anchor, condition) for condition in CONDITIONS}
        new = {condition: score_pair(REPEAT, anchor, condition) for condition in CONDITIONS}
        values = {key: [] for key in contrasts}
        for training, testing in ((old, new), (new, old)):
            selected = selected_score(training, testing, CONDITIONS)
            values["all_vs_initial"].append(selected - statistics.mean(testing["initial"]))
            values["all_vs_self"].append(selected - statistics.mean(testing["self"]))
            values["all_vs_uniform"].append(selected - statistics.mean(statistics.mean(testing[condition]) for condition in CONDITIONS))
            values["all_vs_no_wm_pool"].append(selected - selected_score(training, testing, ("initial", "self")))
        means = {condition: statistics.mean(old[condition] + new[condition]) for condition in CONDITIONS}
        model_differences["opus_vs_kimi"].append(means["opus"] - means["kimi"])
        model_differences["sol_vs_kimi"].append(means["sol"] - means["kimi"])
        paired = {key: statistics.mean(value) for key, value in values.items()}
        for key, value in paired.items():
            contrasts[key].append(value)
        per_world.append({"anchor": anchor, "scores": {condition: old[condition] + new[condition] for condition in CONDITIONS}, "contrasts": paired})
        if anchor in (CASE, BAD_CASE):
            cases[anchor] = collect_case_runs(anchor)
    dialogue = case_dialogue()
    bad_dialogue = bad_case_dialogue()
    (ROOT / "paper/figures/usability_case_dialogue.tex").write_text(render_case_dialogue(dialogue, cases[CASE]))
    (ROOT / "paper/figures/usability_bad_case_dialogue.tex").write_text(render_bad_case_dialogue(bad_dialogue, cases[BAD_CASE]))
    results = {"n": 36, "selector": {key: estimate(values) for key, values in contrasts.items()},
               "model_contrasts": {key: estimate(values) for key, values in model_differences.items()},
               "per_world": per_world, "case": {"anchor": CASE, "runs": cases[CASE], "dialogue": dialogue},
               "bad_case": {"anchor": BAD_CASE, "runs": cases[BAD_CASE], "dialogue": bad_dialogue},
               "scope": "Post-hoc candidate-pool diagnostic; outcome-informed selection, not deployable WM selection. Cases selected retrospectively; not representative."}
    destination = ROOT / "paper/usability_evidence.json"
    destination.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps({key: results[key] for key in ("n", "selector", "model_contrasts")}, indent=2))


if __name__ == "__main__":
    main()
