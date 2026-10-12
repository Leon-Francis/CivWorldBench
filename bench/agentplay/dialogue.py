"""WM v4 -- the planning DIALOGUE with a belief memo (doc 17 §4.3; user decisions 2026-09-18).

Best-of-K showed that at the granularity "K versions of one turn's orders" there is nothing to select (headroom ≈ +1.2
score / decision point).  The dialogue moves the world model to the lever that persists: the agent's long-term plan
(`strategy_note`), which shapes every following turn.  Two things are new against the earlier WM arms:

  belief   the forecaster carries TWO layers of belief, both prepended to every one of its prompts:
             (1) ANALYST NOTES -- the fixed general prior written once after studying 3 training-pool S worlds
                 (bench/runs/v3-abl-belief/analyst_notes.txt; the B18-④ ablation, 1a skill +0.27 -> +0.35);
             (2) a GAME MEMO about THIS game -- written at the takeover from the pre-takeover trajectory
                 (frames t-40, t-20, t + INTERVAL news, POMDP view) and rewritten at every planning point from
                 the previous memo + the two newest frames + the news + how its own earlier forecasts turned out.
                 The memo is the forecaster's standing note, exactly as `strategy_note` is the agent's.
  rounds   round r (r = 1..3): the agent's routes (round 1: its intention + exactly two alternatives it is weighing;
           later rounds: 1-2 revised routes) -> the forecaster answers, per route, (i) 1a quantiles at +20/+40,
           (ii) 1b event probabilities (the benchmark's templates: ego at +20/+40, each neighbour at +20),
           (iii) a written ASSESSMENT -> the agent commits, or revises and asks again (at most 3 rounds).
           The committed route's forecast is kept as a standing FORECASTER line in the panel until the next dialogue
           and becomes the calibration material of the next memo update.

Every item the forecaster answers is scorable afterwards against the game's own per-turn records (bench.truth_v2),
which is the F3 material; every transcript is kept verbatim for case studies.
"""
from __future__ import annotations

import difflib
import re
import time
from pathlib import Path
from typing import Any

from bench.agentplay.wm import (History, WorldModel, current_levels, events_of, frames_dense, met_civs,
                                strategies_multi, T, VARS, build_H, LEVEL_OF)
from bench.parse_v2 import parse_binary, parse_quantiles
from bench.spec_v2 import TEMPLATES_1B

from bench.agentplay.base_rates import block as base_rate_block

ROOT = Path(__file__).resolve().parents[2]
NOTES_PATH = ROOT / "bench/runs/v3-abl-belief/analyst_notes.txt"
HZ_DLG = (20, 40)
LETTERS = "ABCDEFGHIJ"
MEMO_WORDS = 700
EGO_EVENTS = ("lostcity", "newcity", "mildrop", "bankrupt", "moreenemies", "govchange", "wonder", "capture")
OTHER_EVENTS = ("moreenemies", "capture", "lostcity", "newcity")
SHOW_VARS = ("score_gain", "n_cities", "city_size_total", "gold", "techs_learned", "n_military_units", "n_enemies")
_TPL = {k: txt for k, _, _, txt in TEMPLATES_1B}


# ------------------------------------------------------------------ v6: what the ruler is actually judged on
SCORE_RULES = """=====  HOW THE RULER IS SCORED  (facts, not advice)  =====
{viewer} is judged by ONE number: its Freeciv score at turn {end}, which is {left} turns from now.  The formula is
    1 per citizen  +  1 per happy citizen  +  2 per technology known  +  5 per great wonder
    +  units built / 10  +  units killed / 3  +  culture / 1000.
Gold in the treasury and units alive do not appear in it.  Two costs the ruler pays that are easy to miss:
  * the treasury pays the upkeep of every unit and every building, and when it reaches zero the engine SELLS them;
  * one Settlers unit costs 30 shields AND 2 CITIZENS, taken out of the city that builds it, the moment it is built.
    Those 2 points come back only if it founds a city that then grows.
{room}Work out for yourself what each route is worth at turn {end}.  Where two routes differ, say which one is ahead
at turn {end}, not only at +20.
=====  END OF SCORING  =====

"""


def room_block(sites: list[str] | None) -> str:
    """★ v9 (doc 17 §6.15): the ruler's own panel lists the legal city sites the driver can find; the forecaster
    never saw them, and kept advising expansion on maps that were full.  Measured over 32 played games: by turn 200
    there are 0 legal sites in 8 of 10 games, yet the advised arm built 15.7 Settlers in the last 50 turns -- about
    31 citizens, i.e. 31 points, converted into units with nowhere to go."""
    if sites is None:
        return ""
    n = len(sites)
    if n == 0:
        return ("  * EXPANSION ROOM: the driver finds NO legal city site for this ruler right now (a site must be known\n"
                "    land at least 3 tiles from every known city, reachable on foot).  A Settlers unit built now founds\n"
                "    nothing and the 2 citizens it costs are simply gone.\n")
    return f"  * EXPANSION ROOM: the driver finds {n} legal city site(s) for this ruler right now: {'; '.join(sites[:6])}.\n"


def score_block(viewer: str, t1: int, end: int, sites: list[str] | None = None) -> str:
    return SCORE_RULES.format(viewer=viewer, end=end, left=max(0, end - t1), room=room_block(sites))


# ------------------------------------------------------------------ belief layer 1: the analyst notes
def notes_block() -> str:
    notes = NOTES_PATH.read_text().strip()
    return ("=====  ANALYST NOTES  =====\nBefore this game you studied several other games of this kind and wrote the following notes. "
            "Use them together with the observations below.\n\n" + notes + "\n\n=====  END OF NOTES  =====\n\n")


# ------------------------------------------------------------------ belief layer 2: the game memo
MEMO_FIRST = """=====  YOUR TASK: write a GAME MEMO  =====
You are the forecaster for {viewer} in THIS game for the next 100 turns.  Above are three observation frames of the
game so far (turns {frames}) and the news between them, as {viewer} sees them (fog of war).  Write a GAME MEMO
(at most {w} words) that you will re-read and update every ~10 turns before forecasting.  Numbered sections:
1. {viewer}: trend of cities, total city size, gold, science and military across the frames; what limits it now.
2. Each other civilization {viewer} has met, one paragraph each: posture (expanding / arming / at war with whom),
   military trend across the frames, its relation to {viewer} and where that relation is heading, WHICH of
   {viewer}'s cities it could reach and in roughly how many turns, and a personality guess (tentative).
3. World regime: who is at war with whom, the technology pace, and which quantities in this game run above or
   below what the ANALYST NOTES say is typical.
4. Watch list: 3-6 concrete things that would change your forecasts if they happen, each with the turn range in
   which you expect it.
Cite numbers from the frames.  Mark unsure claims (tentative).  No tables.  Plain text, nothing else."""

MEMO_NEXT = """=====  YOUR TASK: update your GAME MEMO  =====
Above are: the GAME MEMO you wrote at turn {prev}; the two newest observation frames (turns {prev} and {t1}) and the
news between them; and, where available, how your forecasts at earlier planning points compare with what actually
happened.  Rewrite the memo (at most {w} words) with the same numbered sections 1-4, then add:
5. What changed since turn {prev}: the concrete differences between the old memo and the new frames.
6. Forecast record: where your earlier forecasts were right or wrong, and what you correct in your beliefs because of it.
Keep what still holds; drop or correct what the new frames contradict.  Cite numbers.  Plain text, nothing else."""


def memo_block(memo: str, memo_turn: int) -> str:
    return (f"=====  GAME MEMO  (your standing notes about THIS game, written at turn {memo_turn})  =====\n"
            + memo.strip() + "\n=====  END OF GAME MEMO  =====\n\n")


def _H(hist: History, frames: list[int], viewer: str) -> tuple[str, dict, list[str], dict]:
    recs = hist.window(frames[0], frames[-1])
    ev = events_of(recs, frames[0], frames[-1])
    world = dict(hist.meta["config"]); world.setdefault("ruleset", hist.meta["config"].get("ruleset", "civ2civ3"))
    H, _ = build_H(recs, frames, viewer, 2, ev, ruleset="", world=world)
    t1 = frames[-1]
    civs = met_civs(recs[t1], viewer)
    try:
        levels = current_levels(recs[t1], civs)
    except Exception:                              # noqa: BLE001
        levels = {}
    return H, recs, civs, levels


# ------------------------------------------------------------------ the forecast question
def event_items(viewer: str, civs: list[str], t1: int, hz, levels: dict) -> list[dict]:
    """1b items in the benchmark's wording: the ego at every horizon, each neighbour at the first one (same list for every route)."""
    items = []
    for civ in civs:
        kinds = EGO_EVENTS if civ == viewer else OTHER_EVENTS
        hzs = list(hz) if civ == viewer else [hz[0]]
        lv = levels.get(civ, {})
        for h in hzs:
            for k in kinds:
                base = lv.get("n_military_units") if k == "mildrop" else lv.get("n_enemies") if k == "moreenemies" else ""
                items.append({"kind": k, "civ": civ, "h": h, "t1": t1, "tt": t1 + h, "base": base,
                              "text": _TPL[k].format(civ=civ, t1=t1, tt=t1 + h, base=base, other="", base2="")})
    return items


def q_dialogue(civs: list[str], viewer: str, t1: int, hz, letters: list[str], items: list[dict], revised: bool,
               siege: bool) -> str:
    """`civs` is the scope of the 1a table.  v5 asks it for the VIEWER ONLY: with three routes and every met civ the
    answer ran past 250 table lines and Flash-Next stopped after the second route (smoke tests 2026-09-19).  The
    neighbours are still forecast, as event probabilities in the 1b block."""
    n1a = len(civs) * len(VARS) * len(hz)
    head = T.q_1a(civs, t1, list(hz)).partition("You are uncertain")[0]
    qs = "\n".join(f"Q{i}  {it['text']}" for i, it in enumerate(items, 1))
    rev = ("These routes are REVISIONS made after your previous forecast in this session (shown above): keep your forecasts "
           "consistent for everything the revision does not change, and let them differ where it does.\n" if revised else "")
    sg = ("A SIEGE ARITHMETIC block above gives, for every threatened city, the driver's estimate of the chance it falls to the "
          "stack that can reach it, and the same estimate with City Walls and with one more fortified defender.  Those three "
          "numbers are computed from the ruleset, not guessed: use them, and do not assign a route a lower city-loss "
          "probability than the block supports just because the route says it will reinforce.\n" if siege else "")
    return (head +
            f"There are {len(letters)} routes {viewer} is weighing (labelled {', '.join(letters)}) in the STRATEGIES block.  The other "
            "civilizations do not know which will be chosen.  " + rev + sg +
            "Use your GAME MEMO and the ANALYST NOTES.  Answer FOUR tasks, in this order and format:\n\n"
            f"### <letter>\n"
            f"one line per (civilization, quantity, horizon): `<civilization> | <quantity> | +<horizon> | p10 | p25 | p50 | p75 | p90` "
            f"-- exactly {n1a} lines (quantities: {', '.join(VARS)}; horizons: {', '.join('+' + str(h) for h in hz)}).\n"
            f"### 1B-<letter>\n"
            f"one line `Q<n> <probability of YES>` (0.00-1.00) for each of the {len(items)} questions below, as they would come out "
            f"if {viewer} follows that route.\n\n" + qs + "\n\n"
            "Numbers only in the table lines, no units, no commas inside numbers.")


def q_end(viewer: str, t1: int, end: int, letters: list[str]) -> str:
    """★ v11 (doc 17 §4.11, user request 2026-09-20).  The forecast table stopped at +20/+40 while the ruler is
    judged at turn `end`.  Everything whose payoff arrives later than +40 -- an army that prevents a loss two
    windows away, a wonder that takes 30 turns to build, culture that accumulates -- was therefore invisible to
    the comparison, and the measured deficits were exactly those three (units, wonders, culture).  So the ruler's
    TOTAL score at the judging turn is now asked for directly, per route."""
    return ("\n### END SCORE\n"
            f"One line per route, exactly this shape: `<letter> | p10 | p50 | p90 | one sentence naming the single "
            f"thing that decides it`.  These are {viewer}'s TOTAL SCORE at turn {end} (not a gain, not a change): "
            "citizens + happy citizens + 2 per technology + 5 per wonder + units built/10 + units killed/3 + "
            f"culture/1000, as it will stand at turn {end} if {viewer} follows that route from now on.  "
            f"Exactly {len(letters)} lines, one for each of {', '.join(letters)}.  "
            "This is the number the ruler is actually judged on, so think about what each route still owns at turn "
            f"{end}, not only about the next 20 turns: a city lost at turn {end - 30} is gone for good, a wonder "
            "started now may or may not finish, and an army that is never attacked scores nothing but keeps "
            "everything else alive.")


def parse_end(txt: str, letters: list[str]) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for ln in txt.splitlines():
        parts = [x.strip().strip("*` ") for x in ln.split("|")]
        if len(parts) < 4 or parts[0].upper() not in letters or parts[0].upper() in out:
            continue
        vals = []
        for x in parts[1:4]:
            m = re.search(r"[-+]?\d+(?:\.\d+)?", x.replace(",", ""))
            vals.append(float(m.group(0)) if m else float("nan"))
        if sum(v == v for v in vals) < 3:
            continue
        out[parts[0].upper()] = vals
    return out


def q_cost(viewer: str, t1: int, hz, letters: list[str], routes: list[dict], siege: bool) -> str:
    return ("You have just forecast these routes.  Now answer two short questions about them.  "
            + ("The SIEGE ARITHMETIC block above gives, for every threatened city, the driver's rule-based estimate of the chance "
               "it falls, and the same estimate with City Walls and with one more fortified defender.  Those numbers are computed "
               "from the ruleset, not guessed: a reinforcement the block says does not change the odds is money spent for nothing.\n"
               if siege else "")
            + "\n### COST\n"
            f"one line per route: `<letter> | gold spent in the next {hz[0]} turns | shields diverted from growth | "
            f"settler-turns given up | change in P({viewer} loses a city by turn {t1 + hz[0]}) against route {letters[0]} | "
            "one sentence saying what the spending buys, citing a number from the frames or the siege block`.  "
            "A route that spends nothing writes 0.  The probability change is signed (-0.20, +0.05, 0.00).  "
            f"Exactly {len(letters)} lines.\n"
            "### RISKS\n"
            "For each route, 1-2 sentences: the single biggest risk on that route (name the civilization / city / turn range) and "
            "what would have to be true for the route to work.  Do NOT rank the routes and do NOT say which one you would choose: "
            f"{viewer} decides.  Be concrete; cite numbers.")


def q_safe(viewer: str, t1: int, hz, end: int, letters: list[str]) -> str:
    """v6 (doc 17 §4.7).  The v5 question was asymmetric: RISKS was compulsory, growth was nowhere, so the
    forecaster's output was always a list of things that could go wrong and the ruler answered it with taxes at
    60/40, a treasury of 300 and City Walls.  The two games the dialogue won (s500030, s500024) were the ones where
    the forecaster said "keep expanding".  So v6 asks for the safe window FIRST, and makes growth advice compulsory
    whenever the window is open."""
    return ("\n### SAFE WINDOW\n"
            f"One line, exactly this shape: `P(any city of {viewer} is attacked before turn {t1 + hz[0]}) | the earliest "
            f"turn an attack could land | which city | which civilization`.  Base it on where foreign units actually were "
            "in the three frames and how fast they move, on the SIEGE ARITHMETIC block if there is one, and on the "
            "measured base rates above -- not on what anyone says they intend.\n"
            "### GROWTH\n"
            f"If that probability is 0.25 or lower, {viewer} is in a SAFE WINDOW and you MUST write, in 4-8 lines, the "
            f"fastest way for {viewer} to have the most SCORE at turn {end}: which cities to grow and how, which "
            "buildings in which city, what tax/science split, how much of the gold it is holding to spend and how much "
            "to keep as the working reserve (name both numbers), and -- only if the EXPANSION ROOM line above says "
            "there are sites -- how many settlers and to which sites.  Every item must end in citizens, technologies or "
            f"a wonder by turn {end}; if it cannot, say so instead of proposing it.  Then say which of the routes "
            f"{', '.join(letters)} comes closest to that, and what it is still wasting.\n"
            f"If the probability is above 0.25, write instead the smallest insurance that buys {viewer} back into a safe "
            "window -- the cheapest thing, named, with its cost -- and then the same growth plan for after it.\n"
            "Do NOT rank the routes overall and do NOT say which route you would choose.")


def parse_safe(txt: str) -> dict:
    for ln in txt.splitlines():
        parts = [x.strip().strip("*` ") for x in ln.split("|")]
        if len(parts) < 2:
            continue
        m = re.search(r"[01]?\.\d+|\b[01]\b", parts[0])
        if not m:
            continue
        tm = re.search(r"\d{2,4}", parts[1])
        return {"p_attacked": float(m.group(0)), "turn": int(tm.group(0)) if tm else None,
                "city": (parts[2][:40] if len(parts) > 2 else ""), "civ": (parts[3][:40] if len(parts) > 3 else "")}
    return {}


def q_targets(viewer: str, t1: int, hz, civs: list[str]) -> str:
    others = [c for c in civs if c != viewer]
    if not others:
        return ""
    return ("\n### TARGETS\n"
            f"One line per civilization {viewer} has met: `<civilization> | P(it attacks a city of {viewer} before turn "
            f"{t1 + hz[0]}) | the city it would hit first | the turn it would arrive | what it would bring`.  "
            f"Exactly {len(others)} lines, one for each of: {', '.join(others)}.  "
            "Base it on where its units were in the three frames and where they were moving, not on its stated intentions.")


def parse_targets(txt: str, civs: list[str]) -> dict[str, dict]:
    out = {}
    for ln in txt.splitlines():
        parts = [x.strip().strip("*` ") for x in ln.split("|")]
        if len(parts) < 3:
            continue
        civ = next((c for c in civs if parts[0][:18].lower() == c[:18].lower()), None)
        if civ is None or civ in out:
            continue
        m = re.search(r"[01]?\.\d+|[01]\b", parts[1])
        if not m:
            continue
        turn = re.search(r"\d{2,4}", parts[3]) if len(parts) > 3 else None
        out[civ] = {"p_attack": float(m.group(0)), "city": parts[2][:40],
                    "turn": int(turn.group(0)) if turn else None,
                    "with": (parts[4][:120] if len(parts) > 4 else "")}
    return out


def parse_cost(txt: str, letters: list[str]) -> dict[str, dict]:
    """Rows `<letter> | gold | shields | settler-turns | signed dP | why`.  The header is often dropped, so the rows
    are recognised by shape and the whole answer can be scanned."""
    def num(x: str) -> float:
        m = re.search(r"[-+]?\d+(?:\.\d+)?", x.replace(",", ""))
        return float(m.group(0)) if m else float("nan")

    out: dict[str, dict] = {}
    for ln in txt.splitlines():
        parts = [x.strip().strip("*` ") for x in ln.split("|")]
        if len(parts) < 5 or parts[0].upper() not in letters or parts[0].upper() in out:
            continue
        vals = [num(x) for x in parts[1:5]]
        if sum(v == v for v in vals) < 3:                   # at least three of the four columns are numeric
            continue
        out[parts[0].upper()] = {"gold": vals[0], "shields": vals[1], "settler_turns": vals[2],
                                 "d_lostcity": vals[3], "why": (parts[5][:300] if len(parts) > 5 else "")}
    return out


def parse_dialogue(answer: str, letters: list[str], items: list[dict], civs_scope: list[str] | None = None) -> dict[str, Any]:
    """{"q": {L: {(civ,var,h): quantiles}}, "ev": {L: {i: p}}, "cost": {L: {...}}, "risks": str}"""
    parts = re.split(r"^\s*#{1,4}\s*(1B-[A-J]|COST|RISKS|TARGETS|ASSESSMENT|SAFE[ _-]?WINDOW|GROWTH|END[ _-]?SCORE|[A-J])\b.*$", answer, flags=re.M)
    q, ev, cost, risks, targets, safe, growth, endsc = {}, {}, {}, "", {}, {}, "", {}
    civs_scope = civs_scope or []
    for i in range(1, len(parts) - 1, 2):
        tag, txt = parts[i].strip().upper(), parts[i + 1]
        if tag in ("RISKS", "ASSESSMENT"):
            risks = (risks + "\n" + txt.strip()).strip()
        elif tag == "COST":
            cost = parse_cost(txt, letters) or cost
        elif tag == "TARGETS":
            targets = parse_targets(txt, civs_scope) or targets
        elif tag.replace("_", " ").replace("-", " ") == "SAFE WINDOW":
            safe = parse_safe(txt) or safe
        elif tag == "GROWTH":
            growth = txt.strip()[:3000]
        elif tag.replace("_", " ").replace("-", " ") == "END SCORE":
            endsc = parse_end(txt, letters) or endsc
        elif tag.startswith("1B-") and tag[3:] in letters:
            ev[tag[3:]] = parse_binary(txt, len(items))
        elif tag in letters:
            q[tag] = parse_quantiles(txt)
    civs_scope = civs_scope or []
    if not cost:                                            # the model often drops the "### COST" header
        cost = parse_cost(answer, letters)
    if not risks and "RISK" in answer.upper():
        risks = answer[answer.upper().index("RISK"):][:4000]
    if not targets and civs_scope:
        targets = parse_targets(answer, civs_scope)
    if not endsc:
        m = re.search(r"END[ _-]?SCORE", answer, flags=re.I)
        if m:
            endsc = parse_end(answer[m.end():], letters)
    return {"q": q, "ev": ev, "cost": cost, "risks": risks, "targets": targets, "safe": safe, "growth": growth,
            "end_score": endsc}


def render_for_agent(viewer: str, routes: list[dict], letters: list[str], civs: list[str], parsed: dict, items: list[dict],
                     hz, levels: dict, risk_text: str = "") -> str:
    L = []
    now = levels.get(viewer, {})
    es = parsed.get("end_score") or {}
    if es:
        L.append(f"★ THE NUMBER YOU ARE JUDGED ON -- the forecaster's estimate of your TOTAL SCORE at the final turn, per route")
        for L_, r in zip(letters, routes):
            v = es.get(L_)
            if v:
                L.append(f"   {L_} {r['name'][:28]:<28} p10 {v[0]:g} / p50 {v[1]:g} / p90 {v[2]:g}")
        best = max(es.items(), key=lambda kv: kv[1][1], default=None)
        if best and len(es) > 1:
            worst = min(es.items(), key=lambda kv: kv[1][1])
            L.append(f"   -> on its own numbers, {best[0]} is ahead of {worst[0]} by {best[1][1] - worst[1][1]:g} points at the final turn.")
        L.append("")
    if risk_text:
        L.append(risk_text); L.append("")
    sw = parsed.get("safe") or {}
    if sw:
        L.append(f"SAFE WINDOW: P(any of your cities is attacked before T{items[0]['tt'] if items else '?'}) = {sw['p_attacked']:.2f}"
                 + (f"; earliest ~T{sw['turn']}" if sw.get("turn") else "")
                 + (f"; first target {sw['city']}" if sw.get("city") else "")
                 + (f"; from {sw['civ']}" if sw.get("civ") else "")
                 + ("  -- YOU ARE IN A SAFE WINDOW: the forecaster's growth plan below is what it costs you to ignore."
                    if sw["p_attacked"] <= 0.25 else ""))
    if parsed.get("growth"):
        L.append("")
        L.append("HOW TO TURN WHAT YOU HAVE INTO SCORE (the forecaster's growth plan; gold in your treasury scores nothing):")
        L.append(parsed["growth"][:2500])
        L.append("")
    L.append(f"now: cities {now.get('n_cities', '?')}, gold {now.get('gold', '?')}, military units {now.get('n_military_units', '?')}, enemies {now.get('n_enemies', '?')}")
    L.append("your quantities (p10 / p50 / p90)      " + "".join(f"| {L_}: {r['name'][:20]:<20} " for L_, r in zip(letters, routes)))
    for v in SHOW_VARS:
        for h in hz:
            row = f"{v:<16} +{h:<3} "
            for L_ in letters:
                qq = parsed["q"].get(L_, {}).get((viewer, v, h))
                row += f"| {'' if qq is None else f'{qq[0]:g} / {qq[2]:g} / {qq[4]:g}':<24} "
            L.append(row)
    L.append("event probabilities (P of YES):")
    for i, it in enumerate(items, 1):
        cells = []
        for L_ in letters:
            p = parsed["ev"].get(L_, {}).get(i)
            cells.append("" if p is None else f"{p:.2f}")
        if any(cells):
            L.append(f"  {it['kind']:<12} {it['civ'][:18]:<18} by T{it['tt']:<4} " + "  ".join(f"{L_} {c:>4}" for L_, c in zip(letters, cells)))
    for c in civs:
        if c == viewer:
            continue
        cells = []
        for L_ in letters:
            qq = parsed["q"].get(L_, {})
            mil = qq.get((c, "n_military_units", hz[0])); sg = qq.get((c, "score_gain", hz[0])); nc = qq.get((c, "n_cities", hz[0]))
            cells.append(f"{L_}: mil {'' if mil is None else f'{mil[2]:g}'} cities {'' if nc is None else f'{nc[2]:g}'} Δscore {'' if sg is None else f'{sg[2]:g}'}")
        L.append(f"{c[:18]:<18} (+{hz[0]}, p50)  " + " | ".join(cells))
    if parsed.get("cost"):
        L.append("")
        L.append("what each route costs (forecaster's figures; dP = change in P(lose a city) vs the first route)")
        for L_ in letters:
            c = parsed["cost"].get(L_)
            if c:
                L.append(f"  {L_}: gold {c['gold']:g}, shields {c['shields']:g}, settler-turns {c['settler_turns']:g}, "
                         f"dP(lose city) {c['d_lostcity']:+.2f} -- {c['why']}")
    if parsed.get("targets"):
        L.append("")
        L.append("who the forecaster expects to attack you (from where their units were and were heading):")
        for civ, t in parsed["targets"].items():
            L.append(f"  {civ[:18]:<18} P(attacks you by +{hz[0]}) {t['p_attack']:.2f}"
                     + (f", first target {t['city']}" if t.get("city") else "")
                     + (f", around turn {t['turn']}" if t.get("turn") else "")
                     + (f", with {t['with']}" if t.get("with") else ""))
    if parsed.get("risks"):
        L.append("")
        L.append("RISKS (the forecaster's words; it is not allowed to rank the routes):")
        L.append(parsed["risks"][:3000])
    return "\n".join(L)


def prior_rounds_block(t1: int, rounds: list[dict]) -> str:
    if not rounds:
        return ""
    L = [f"=====  EARLIER ROUNDS OF THIS PLANNING SESSION (turn {t1})  =====",
         "The routes in the STRATEGIES block are revisions the ruler made after reading your forecasts below."]
    for r in rounds:
        L.append(f"--- round {r['round']}: routes " + ", ".join(r["letters"]))
        for L_, rt in zip(r["letters"], r["routes"]):
            L.append(f"{L_}: {rt['name']} -- " + " / ".join(ln.strip() for ln in rt["plan"].splitlines() if ln.strip())[:900])
        L.append("your forecast (as the ruler saw it):")
        L.append(r["forecast"]["text"][:3500])
        if r.get("decision"):
            L.append(f"the ruler's reply: {r['decision'].get('action')} -- {str(r['decision'].get('why', ''))[:400]}")
    return "\n".join(L) + "\n\n"


def similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, (a or "").split(), (b or "").split()).ratio()


# ------------------------------------------------------------------ the dialogue object (one per game)
class Dialogue:
    def __init__(self, wm: WorldModel, hist: History, hz=HZ_DLG, memo_words: int = MEMO_WORDS, targets: bool = False,
                 end: int = 0, v6: bool = False):
        self.wm, self.hist, self.hz, self.memo_words, self.want_targets = wm, hist, hz, memo_words, targets
        self.end, self.v6 = end, v6
        self.notes = notes_block()
        self.base_rates = base_rate_block() if v6 else ""
        self.memo: str = ""; self.memo_turn: int | None = None
        self.history: list[dict] = []           # committed forecasts: {turn, letter, name, p50 {(civ,var,h)}, levels, items, ev {i: p}}
        self.standing: str = ""                 # the panel line(s) for the turns between dialogues

    # ---- resume
    def restore(self, prev_recs: list[dict]) -> None:
        for r in prev_recs:
            w = r.get("wm") or {}
            if w.get("kind") not in ("dialogue", "brief"):
                continue
            if w.get("memo", {}).get("memo"):
                self.memo, self.memo_turn = w["memo"]["memo"], r["turn"]
            if w.get("kind") == "brief":
                self.standing = (w.get("brief") or {}).get("text", "")
                continue
            if w.get("chosen") and w.get("rounds"):
                self._commit_from_record(r["turn"], w)
        if self.history and not self.standing:
            self.standing = self.history[-1]["standing"]

    # ---- belief layer 2
    def memo_update(self, t1: int, viewer: str) -> dict[str, Any]:
        t0 = time.time()
        if self.memo_turn is None:
            frames = sorted({max(1, t1 - 40), max(1, t1 - 20), t1})
            H, recs, civs, levels = _H(self.hist, frames, viewer)
            body = (self.notes + self.base_rates + (score_block(viewer, t1, self.end) if self.v6 and self.end else "")
                    + H + "\n" + MEMO_FIRST.format(viewer=viewer, frames=", ".join(map(str, frames)), w=self.memo_words))
            cal = ""
        else:
            frames = sorted({self.memo_turn, t1}) if self.memo_turn < t1 else [t1]
            H, recs, civs, levels = _H(self.hist, frames, viewer)
            cal = self.calibration(t1, viewer, levels)
            body = (self.notes + self.base_rates + (score_block(viewer, t1, self.end) if self.v6 and self.end else "")
                    + memo_block(self.memo, self.memo_turn) + H + "\n" + (cal + "\n" if cal else "")
                    + MEMO_NEXT.format(prev=self.memo_turn, t1=t1, w=self.memo_words))
        out = self.wm.call(body, int(self.memo_words * 2.2))
        new = out["answer"].strip()
        sim = similarity(self.memo, new) if self.memo else None
        ok = len(new.split()) >= 80
        rec = {"turn": t1, "prev_turn": self.memo_turn, "frames": frames, "memo": new if ok else self.memo, "ok": ok, "similarity": sim,
               "calibration": cal, "prompt_chars": len(body), "call": {**out, "answer": out["answer"][:8000]}, "seconds": time.time() - t0}
        if ok:
            self.memo, self.memo_turn = new, t1
        return rec

    def calibration(self, t1: int, viewer: str, levels: dict) -> str:
        """What the forecaster said at earlier planning points about the route that was then committed, next to what happened:
        1a p50 vs actual for horizons that have elapsed, and 1b probabilities vs realized truth for windows that have closed."""
        from bench.truth_v2 import mil_of, truth_1b_from_states
        if not self.history or viewer not in levels:
            return ""
        L = ["=====  FORECAST RECORD: what you predicted at earlier planning points for the route that was followed, next to what happened  =====",
             f"(read before writing; values observed at turn {t1})"]
        now = levels[viewer]
        for h_ in self.history[-4:]:
            dt = t1 - h_["turn"]
            cells = []
            for v in ("score_gain", "n_cities", "gold", "techs_learned", "n_military_units", "n_enemies"):
                p = h_["p50"].get((viewer, v, dt))
                if p is None:
                    continue
                act = now.get(LEVEL_OF[v])
                if v in ("score_gain", "techs_learned", "culture_gain") and act is not None:
                    act = act - h_["levels"].get(LEVEL_OF[v], 0)
                if act is not None:
                    cells.append(f"{v} +{dt}: forecast {p:g}, actual {act:g}")
            if cells:
                L.append(f"turn {h_['turn']} (route {h_['letter']}: {h_['name'][:40]}): " + "; ".join(cells))
            closed = [it for it in h_["items"] if it["tt"] <= t1 and h_["ev"].get(it["i"]) is not None]
            if closed:
                try:
                    rec1 = self.hist.record(h_["turn"])
                    states = {t: self.hist.record(t) for t in range(h_["turn"], max(it["tt"] for it in closed) + 1)}
                    truths = truth_1b_from_states(closed, states, rec1, mil_of(rec1))
                    lines = []
                    for it, tr in zip(closed, truths):
                        if tr is None:
                            continue
                        lines.append(f"{it['kind']} {it['civ']} by T{it['tt']}: P={h_['ev'][it['i']]:.2f} -> {'YES' if tr else 'no'}")
                    if lines:
                        L.append(f"turn {h_['turn']} events: " + "; ".join(lines))
                except Exception as e:                 # noqa: BLE001
                    L.append(f"turn {h_['turn']} events: (could not be evaluated: {type(e).__name__})")
        return "\n".join(L) + "\n" if len(L) > 2 else ""

    # ---- one forecast call (one round)
    def forecast(self, t1: int, viewer: str, routes: list[dict], letters: list[str], rounds: list[dict],
                 siege_text: str = "", sites: list[str] | None = None, risk_text: str = "") -> dict[str, Any]:
        """Two calls in parallel over the same context: the benchmark-format 1a/1b tables, and the short COST/RISKS
        block.  They were one call in v4; at three routes the answer ran past 250 lines and the model silently
        dropped the tail (smoke test 2026-09-19: sections A/1B-A/B/1B-B/C written, everything after them missing)."""
        import concurrent.futures as cf
        t0 = time.time()
        # ★ v11: the second horizon follows the judging turn instead of being pinned at +40.
        if self.v6 and self.end:
            far = max(40, min(80, self.end - t1))
            self.hz = (20, far) if far > 20 else (20,)
        frames = frames_dense(t1)
        H, recs, civs, levels = _H(self.hist, frames, viewer)
        items = event_items(viewer, civs, t1, self.hz, levels)
        siege_block = ("=====  SIEGE ARITHMETIC  (computed by the driver from the ruleset, same block the ruler sees)  =====\n"
                       + siege_text + "\n=====  END OF SIEGE ARITHMETIC  =====\n\n") if siege_text else ""
        if risk_text:
            siege_block += ("=====  ROUTE RISK  (computed by the driver for THESE routes; arithmetic, not a forecast --\n"
                            "       do not contradict it, and make your own city-loss probabilities agree with it)  =====\n"
                            + risk_text + "\n=====  END OF ROUTE RISK  =====\n\n")
        head = (self.notes + self.base_rates
                + (score_block(viewer, t1, self.end, sites) if self.v6 and self.end else ""))
        ctx = (head + (memo_block(self.memo, self.memo_turn) if self.memo else "") + H + "\n"
               + strategies_multi(viewer, t1, routes, letters) + "\n" + siege_block + prior_rounds_block(t1, rounds))
        body = ctx + q_dialogue([viewer], viewer, t1, self.hz, letters, items, revised=bool(rounds), siege=bool(siege_text))
        body_c = ctx + q_cost(viewer, t1, self.hz, letters, routes, bool(siege_text))
        if self.v6:
            body_c += q_end(viewer, t1, self.end, letters)
            body_c += q_safe(viewer, t1, self.hz, self.end, letters)
        if self.want_targets:
            body_c += q_targets(viewer, t1, self.hz, civs)
        max_tok = int((len(letters) * (len(VARS) * len(self.hz) * 32 + len(items) * 12) + 2500) * 1.6)
        with cf.ThreadPoolExecutor(max_workers=2) as ex:
            fa = ex.submit(self.wm.call, body, max_tok)
            fb = ex.submit(self.wm.call, body_c, 2500 + (700 if self.want_targets else 0) + (1900 if self.v6 else 0))
            out, out_c = fa.result(), fb.result()
        parsed = parse_dialogue(out["answer"], letters, items, civs)
        parsed_c = parse_dialogue(out_c["answer"], letters, items, civs)
        parsed["cost"] = parsed_c["cost"] or parsed["cost"]
        parsed["risks"] = parsed_c["risks"] or parsed["risks"]
        parsed["targets"] = parsed_c["targets"] or parsed["targets"]
        parsed["safe"] = parsed_c.get("safe") or parsed.get("safe") or {}
        parsed["growth"] = parsed_c.get("growth") or parsed.get("growth") or ""
        parsed["end_score"] = parsed_c.get("end_score") or parsed.get("end_score") or {}
        text = render_for_agent(viewer, routes, letters, civs, parsed, items, self.hz, levels, risk_text)
        n_ok = sum(1 for L_ in letters if parsed["q"].get(L_))
        return {"text": text, "call": {**out, "answer": out["answer"][:16000]},
                "call_cost": {**out_c, "answer": out_c["answer"][:6000]},
                "parsed": {"q": {L_: {f"{c}|{v}|{h}": list(x) for (c, v, h), x in d.items()} for L_, d in parsed["q"].items()},
                           "ev": {L_: {str(i): p for i, p in d.items()} for L_, d in parsed["ev"].items()},
                           "cost": parsed["cost"], "risks": parsed["risks"], "targets": parsed["targets"],
                           "safe": parsed["safe"], "growth": parsed["growth"], "end_score": parsed["end_score"]},
                "items": items, "letters": letters, "civs": civs, "levels_ego": levels.get(viewer, {}),
                "prompt_chars": len(body), "sections_ok": n_ok, "cost_ok": len(parsed["cost"]),
                "seconds": time.time() - t0}


    # ======================================================================================================
    # v12: forecast PLANS and say what to change in them (doc 17 Section 4.12)
    # ======================================================================================================
    def forecast_plans(self, t1: int, viewer: str, plans: list[dict], labels: list[str], rounds: list[dict],
                       siege_text: str = "", sites: list[str] | None = None, risk_text: str = "",
                       checks: list[str] | None = None) -> dict[str, Any]:
        import concurrent.futures as cf
        from bench.agentplay.plan import render_plan
        t0 = time.time()
        if self.end:
            far = max(40, min(80, self.end - t1))
            self.hz = (20, far) if far > 20 else (20,)
        frames = frames_dense(t1)
        H, recs, civs, levels = _H(self.hist, frames, viewer)
        items = event_items(viewer, civs, t1, self.hz, levels)

        pblock = [f"=====  THE PLANS {viewer} IS WEIGHING AT TURN {t1}  =====",
                  "Each is a concrete plan its ruler intends to execute: the driver will run the production queues as"
                  " written.  Read them as declared policy."]
        for L_, pl in zip(labels, plans):
            pblock.append(f"--- PLAN {L_}")
            pblock.append(render_plan(pl))
        pblock.append("=====  END OF PLANS  =====\n")
        blocks = "\n".join(pblock) + "\n"
        if siege_text:
            blocks += ("=====  SIEGE ARITHMETIC  (driver, from the ruleset; same block the ruler sees)  =====\n"
                       + siege_text + "\n=====  END  =====\n\n")
        if risk_text:
            blocks += ("=====  PLAN RISK  (driver arithmetic for THESE plans' garrisons; it is not a forecast --\n"
                       "       do not contradict it, and make your own city-loss probabilities agree with it)  =====\n"
                       + risk_text + "\n=====  END  =====\n\n")
        if checks:
            blocks += ("=====  THE DRIVER'S CHECKS ON THESE PLANS  (facts from the map and the ruleset: illegal city\n"
                       "       sites, queues that do not fit the shields, units that cannot arrive in time)  =====\n"
                       + "\n".join(checks) + "\n=====  END  =====\n\n")

        head = self.notes + self.base_rates + (score_block(viewer, t1, self.end, sites) if self.end else "")
        ctx = (head + (memo_block(self.memo, self.memo_turn) if self.memo else "") + H + "\n"
               + blocks + prior_rounds_block(t1, rounds))
        n1a = len(VARS) * len(self.hz)
        qs = "\n".join(f"Q{i}  {it['text']}" for i, it in enumerate(items, 1))
        body = (ctx + f"{viewer} is weighing the {len(labels)} plans above (labelled {', '.join(labels)}).  Forecast each of "
                "them as if it were followed from now on.  Answer in this order and format:\n\n"
                f"### <label>\none line per (quantity, horizon): `{viewer} | <quantity> | +<horizon> | p10 | p25 | p50 | p75 | p90` "
                f"-- exactly {n1a} lines (quantities: {', '.join(VARS)}; horizons: {', '.join('+' + str(h) for h in self.hz)}).\n"
                f"### 1B-<label>\none line `Q<n> <probability of YES>` for each of the {len(items)} questions below, as they "
                f"would come out under that plan.\n\n" + qs + "\n\nNumbers only in the table lines.")

        body_c = (ctx + "You have just forecast these plans.  Now answer the questions that decide what the ruler does next.\n"
                  + q_end(viewer, t1, self.end, labels)
                  + "\n### COST\n"
                  f"one line per plan: `<label> | gold spent in the next {self.hz[0]} turns | shields diverted from growth | "
                  f"settler-turns | change in P({viewer} loses a city by T{t1 + self.hz[0]}) against plan {labels[0]} | what "
                  "the spending buys, citing a number`.\n"
                  + q_safe(viewer, t1, self.hz, self.end, labels)
                  + "\n### ADVICE\n"
                  "For EACH plan, 3 to 6 lines, each one pointing at a specific line of that plan and saying what to put "
                  "there instead: a city and what it should build in place of what is queued, a settler and a different "
                  "tile, a garrison number, a rate, a research step, a trigger that is missing.  For each change, say what "
                  f"you expect it to be worth in points at turn {self.end} and why.  Use the driver's checks and the plan "
                  "risk above -- those are facts, so a plan that contradicts them must be changed, not defended.  "
                  "Then ONE closing line: which plan you would play and what single thing decides it.  "
                  "If a plan is not worth forecasting again, say so and why.")
        if self.want_targets:
            body_c += q_targets(viewer, t1, self.hz, civs)

        max_tok = int((len(labels) * (n1a * 32 + len(items) * 12) + 2500) * 1.6)
        with cf.ThreadPoolExecutor(max_workers=2) as ex:
            fa = ex.submit(self.wm.call, body, max_tok)
            fb = ex.submit(self.wm.call, body_c, 4200 + (700 if self.want_targets else 0))
            out, out_c = fa.result(), fb.result()
        parsed = parse_dialogue(out["answer"], labels, items, civs)
        pc = parse_dialogue(out_c["answer"], labels, items, civs)
        for k in ("cost", "risks", "targets", "safe", "growth", "end_score"):
            parsed[k] = pc.get(k) or parsed.get(k) or ({} if k in ("cost", "targets", "safe", "end_score") else "")
        adv = ""
        m = re.search(r"#{1,4}\s*ADVICE\b", out_c["answer"], flags=re.I)
        if m:
            adv = out_c["answer"][m.end():].strip()[:5000]
        parsed["advice"] = adv

        L = []
        es = parsed.get("end_score") or {}
        if es:
            L.append(f"THE NUMBER YOU ARE JUDGED ON -- forecast TOTAL SCORE at turn {self.end}, per plan")
            for L_, pl in zip(labels, plans):
                v = es.get(L_)
                if v:
                    L.append(f"   {L_} {pl.get('name','')[:30]:<30} p10 {v[0]:g} / p50 {v[1]:g} / p90 {v[2]:g}")
            L.append("")
        if risk_text:
            L.append(risk_text); L.append("")
        if checks:
            L.append("THE DRIVER'S CHECKS ON YOUR PLANS (facts, not opinion):")
            L += ["  " + c for c in checks]
            L.append("")
        L.append(render_for_agent(viewer, [{"name": p_.get("name", ""), "plan": ""} for p_ in plans],
                                  labels, civs, parsed, items, self.hz, levels))
        if adv:
            L.append("")
            L.append("WHAT THE FORECASTER SAYS TO CHANGE:")
            L.append(adv[:4000])
        text = "\n".join(L)
        return {"text": text, "call": {**out, "answer": out["answer"][:16000]},
                "call_cost": {**out_c, "answer": out_c["answer"][:9000]},
                "parsed": {"q": {L_: {f"{c}|{v}|{h}": list(x) for (c, v, h), x in d.items()} for L_, d in parsed["q"].items()},
                           "ev": {L_: {str(i): p_ for i, p_ in d.items()} for L_, d in parsed["ev"].items()},
                           "cost": parsed["cost"], "risks": parsed["risks"], "targets": parsed["targets"],
                           "safe": parsed["safe"], "growth": parsed["growth"], "end_score": parsed["end_score"],
                           "advice": adv},
                "items": items, "letters": labels, "civs": civs, "levels_ego": levels.get(viewer, {}),
                "prompt_chars": len(body) + len(body_c), "sections_ok": sum(1 for L_ in labels if parsed["q"].get(L_)),
                "seconds": time.time() - t0}

    # ---- the cheap variant: one call every few turns on the course the agent is already following
    def brief(self, t1: int, viewer: str, note: str, siege_text: str = "") -> dict[str, Any]:
        """`brief` arm: no routes, no rounds.  One forecast of the course in force (the agent's own strategy note),
        rendered as a standing FORECASTER block.  Costs one call instead of the dialogue's four to eight, so it can
        run every 5 turns instead of every 10 -- the opposite trade from the v5 dialogue."""
        t0 = time.time()
        frames = frames_dense(t1)
        H, recs, civs, levels = _H(self.hist, frames, viewer)
        items = [it for it in event_items(viewer, civs, t1, self.hz, levels) if it["civ"] == viewer and it["h"] == self.hz[0]]
        route = [{"name": "the course in force", "plan": (note or "Course: continue as before.")[:1500]}]
        siege_block = ("=====  SIEGE ARITHMETIC  (computed by the driver from the ruleset, same block the ruler sees)  =====\n"
                       + siege_text + "\n=====  END OF SIEGE ARITHMETIC  =====\n\n") if siege_text else ""
        qs = "\n".join(f"Q{i}  {it['text']}" for i, it in enumerate(items, 1))
        q = (f"{viewer} is following the course in the STRATEGIES block.  Forecast what happens to {viewer} if it keeps "
             f"following it, in three short sections.\n\n### A\n"
             f"one line per quantity: `{viewer} | <quantity> | +{self.hz[0]} | p10 | p25 | p50 | p75 | p90` -- exactly "
             f"{len(VARS)} lines (quantities: {', '.join(VARS)}).\n"
             f"### 1B-A\none line `Q<n> <probability of YES>` for each of the {len(items)} questions:\n{qs}\n"
             + q_targets(viewer, t1, self.hz, civs) +
             "\n### RISKS\nAt most 4 sentences: the one thing most likely to cost {v} score in the next {h} turns, and the one "
             "opportunity it is missing.  Cite numbers from the frames or the siege block.  Do not write a plan.".format(v=viewer, h=self.hz[0]))
        body = (self.notes + (memo_block(self.memo, self.memo_turn) if self.memo else "") + H + "\n"
                + strategies_multi(viewer, t1, route, ["A"]) + "\n" + siege_block + q)
        out = self.wm.call(body, len(VARS) * 32 + len(items) * 12 + 1800)
        parsed = parse_dialogue(out["answer"], ["A"], items, civs)
        L = []
        now = levels.get(viewer, {})
        L.append(f"forecast if you keep your current course (p10 / p50 / p90 at +{self.hz[0]} turns; now: cities "
                 f"{now.get('n_cities', '?')}, gold {now.get('gold', '?')}, military {now.get('n_military_units', '?')})")
        for v in SHOW_VARS:
            qq = (parsed["q"].get("A") or {}).get((viewer, v, self.hz[0]))
            if qq:
                L.append(f"  {v:<16} {qq[0]:g} / {qq[2]:g} / {qq[4]:g}")
        ev = parsed["ev"].get("A") or {}
        cells = [f"{it['kind']} {ev[i]:.2f}" for i, it in enumerate(items, 1) if ev.get(i) is not None]
        if cells:
            L.append(f"  events by T{t1 + self.hz[0]}: " + ", ".join(cells))
        for civ, tg in (parsed.get("targets") or {}).items():
            if tg["p_attack"] >= 0.2:
                L.append(f"  {civ} attacks you p {tg['p_attack']:.2f}" + (f", first target {tg['city']}" if tg.get("city") else "")
                         + (f", ~T{tg['turn']}" if tg.get("turn") else "") + (f", with {tg['with']}" if tg.get("with") else ""))
        if parsed.get("risks"):
            L.append("  " + " ".join(parsed["risks"].split())[:700])
        self.standing = "\n- ".join(L)
        self.history.append({"turn": t1, "letter": "A", "name": "course in force",
                             "p50": {(viewer, v, self.hz[0]): (parsed["q"].get("A") or {}).get((viewer, v, self.hz[0]), [None] * 5)[2]
                                     for v in VARS if (parsed["q"].get("A") or {}).get((viewer, v, self.hz[0]))},
                             "levels": dict(now), "items": [{**it, "i": i} for i, it in enumerate(items, 1)],
                             "ev": {i: p for i, p in ev.items()}, "standing": self.standing})
        return {"text": self.standing, "call": {**out, "answer": out["answer"][:8000]},
                "parsed": {"q": {"A": {f"{c}|{v}|{h}": list(x) for (c, v, h), x in (parsed["q"].get("A") or {}).items()}},
                           "ev": {"A": {str(i): p for i, p in ev.items()}}, "targets": parsed.get("targets") or {},
                           "risks": parsed.get("risks", "")},
                "levels_ego": now, "prompt_chars": len(body), "seconds": time.time() - t0}

    # ---- the committed route: calibration material + the standing panel line
    def commit(self, t1: int, letter: str, rounds: list[dict], viewer: str) -> dict[str, Any]:
        for r in rounds:
            if letter in r["letters"]:
                f = r["forecast"]; rt = r["routes"][r["letters"].index(letter)]
                break
        else:
            r = rounds[-1]; letter = r["letters"][0]; f = r["forecast"]; rt = r["routes"][0]
        p50 = {}
        for k, x in (f["parsed"]["q"].get(letter) or {}).items():
            c, v, h = k.split("|"); p50[(c, v, int(h))] = x[2]
        ev = {int(i): p for i, p in (f["parsed"]["ev"].get(letter) or {}).items()}
        items = [{**it, "i": i} for i, it in enumerate(f["items"], 1)]
        ego_items = [(it, ev.get(it["i"])) for it in items if it["civ"] == viewer and ev.get(it["i"]) is not None]
        warn = sorted(ego_items, key=lambda x: -x[1])
        parts = [f"{it['kind']} by T{it['tt']} {p:.2f}" for it, p in warn[:6]]
        tg = (f.get("parsed", {}) or {}).get("targets") or {}
        top = max(tg.items(), key=lambda kv: kv[1]["p_attack"], default=None)
        tg_txt = (f"; {top[0]} attacks {top[1]['city']} p {top[1]['p_attack']:.2f}"
                  + (f" ~T{top[1]['turn']}" if top[1].get("turn") else "")) if top and top[1]["p_attack"] >= 0.3 else ""
        sg = [f"Δscore +{h} p50 {p50[(viewer, 'score_gain', h)]:g}" for h in self.hz if (viewer, "score_gain", h) in p50]
        sw = (f.get("parsed", {}) or {}).get("safe") or {}
        sw_txt = ""
        if sw:
            sw_txt = (f"; P(attacked before T{t1 + self.hz[0]}) {sw['p_attacked']:.2f}"
                      + (" -- SAFE WINDOW: spend, do not hoard" if sw["p_attacked"] <= 0.25 else ""))
        standing = (f"FORECASTER (planning dialogue at turn {t1}, your committed route {letter} '{rt['name'][:40]}'): "
                    + "; ".join(sg + parts) + tg_txt + sw_txt) if (sg or parts) else ""
        entry = {"turn": t1, "letter": letter, "name": rt["name"], "p50": p50, "levels": dict(f.get("levels_ego") or {}),
                 "items": items, "ev": ev, "standing": standing}
        self.history.append(entry); self.standing = standing
        return {"letter": letter, "standing": standing}

    def _commit_from_record(self, t1: int, w: dict) -> None:
        rounds = w["rounds"]
        for r in rounds:
            if w["chosen"] in r["letters"]:
                f = r["forecast"]; rt = r["routes"][r["letters"].index(w["chosen"])]; break
        else:
            return
        p50 = {}
        for k, x in (f["parsed"]["q"].get(w["chosen"]) or {}).items():
            c, v, h = k.split("|"); p50[(c, v, int(h))] = x[2]
        ev = {int(i): p for i, p in (f["parsed"]["ev"].get(w["chosen"]) or {}).items()}
        items = [{**it, "i": i} for i, it in enumerate(f["items"], 1)]
        self.history.append({"turn": t1, "letter": w["chosen"], "name": rt["name"], "p50": p50, "levels": dict(f.get("levels_ego") or {}),
                             "items": items, "ev": ev, "standing": w.get("standing", "")})
