"""The LLM player: one chat call per decision turn, JSON orders out.

System prompt = role + the Freeciv expert manual (bench/measure/human_line) + the ruleset primer + the order
schema.  User message = the agent's own standing strategy note + the fog-of-war world report + the EGO PANEL
(bench/agentplay/observe.py) + the instruction.  Thinking is on (Qwen3.8 chat template, medium effort); the
reply's <think> part is stripped and kept for the log.

Orders are a JSON object (see SCHEMA); anything the driver cannot execute is rejected with a reason that the
model sees next turn under RESULT OF YOUR LAST ORDERS.
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from bench.agentplay.engine import CityCmd, TurnPlan, UnitCmd   # noqa: E402

MANUAL = ROOT / "bench/measure/human_line/FREECIV_EXPERT_MANUAL.md"
PRIMER = {"civ2civ3": ROOT / "freeciv-sandbox/ruleset_primer/RULES_civ2civ3.md",
          "classic": ROOT / "freeciv-sandbox/ruleset_primer/RULES_classic.md"}

ROLE = """You are playing Freeciv 3.1 as the ruler of one civilization.  The other civilizations are the game's built-in AI.
The game runs turn by turn; each turn you receive the world as your civilization currently knows it (fog of war applies),
a panel with your cities, units, research and alerts, and the engine's verdict on your previous orders.  You reply with
one JSON object of orders for this turn.  Your objective is the highest possible **score** for your civilization at the
end of the game (score = citizens + happy citizens + 2 per technology + 5 per wonder + units built/10 + units killed/3 +
culture/1000).  Growth, technology and culture are the main levers; wonders help; losing cities is catastrophic.

How this interface differs from a normal client:
- Coordinates are the (x,y) pairs printed in the report; the map wraps in both directions and is a hex grid: a tile's six
  neighbours are the ones the pathfinder uses, you never need to compute them.  You give *destinations*; the driver finds
  the route over explored tiles and moves the unit each turn until it arrives (a standing order), then performs the
  arrival action (found a city, attack, join a city, help a wonder, build irrigation/mine/road, ...).
- Workers/Settlers can be delegated to the server's auto-worker ("auto_work"); military units and boats can "explore"
  automatically.  Fortified units in cities defend them.
- City production: name any item in that city's "can build" lists; changing between a unit and a building loses half
  the stored shields.  "buy": true completes the current item at once for gold (price shown by the engine only after the
  fact; roughly 2x the missing shields plus a quadratic term, doubled when nothing is stored yet).
- Research: set a long-term "research_goal"; the server researches prerequisites in order.  "research_now" is optional
  and must be researchable right now; changing the current research throws away the bulbs already stored, so set it
  rarely (normally only the goal).
- Tax/science/luxury rates are multiples of 10 summing to 100, each capped by your government.
- A government change ("government") means 2 turns of Anarchy first.
- Diplomacy is passive: you cannot propose treaties; AI civs may declare war on you or offer nothing.  Defend accordingly.
- A unit without orders does nothing.  Units that are "idle" every turn waste upkeep.  Units already fortified or
  auto-working keep doing so unless you give them a new order.
- Founding a city needs a Settlers unit on a land tile at least 3 tiles from any other city (the engine rejects closer
  sites); Migrants can only join cities.
- Units marked ABOARD are on a boat at sea: they cannot act until the boat is adjacent to land; then order the unit to
  a land tile (it disembarks).  Boats sent to "explore" carry their passengers away with them.
- Read the map grid carefully: the column header gives x (tens row, then units row), each printed row starts with y.
  Terrain letters: ' ' Ocean, ':' Deep Ocean, '+' Lake are water; g p d h f j m s t a are land; '?' unexplored.
  The panel's CANDIDATE CITY SITES list is computed by the driver and is reliable; prefer it over reading the grid.

What strong play looks like in these worlds (the AI you are up against does this):
- EXPAND FIRST.  Keep every city producing Settlers until you have 8-12 cities; "found_city" without "to" sends a
  Settlers unit (even one aboard a boat) to the nearest good legal site automatically -- use it liberally.
- Every city keeps 1-2 defensive units inside (Phalanx/Pikemen/Musketeers, fortified); build City Walls in cities an
  enemy can reach.  Never attack a unit whose defence beats your attack; do not send Phalanxes to attack.
- Keep the treasury positive: taxes 30-50%, luxury 0% unless a city is in disorder; buy only when it decides a fight.
- Research toward Monarchy early (then Currency/Trade/Philosophy); change government as soon as a better one is known.
- Libraries/Temples/Marketplaces after the expansion wave, not before.  Wonders only in your best production city.
- Diplomacy: you can ask for a ceasefire (then peace) via "treaties"; the AI accepts when it is not hostile and has no
  war plan against you.  Ending a war you cannot win is worth more than any building.

Be decisive and economical: only mention units and cities whose orders you want to change.  Use "end_turns" > 1 when
nothing needs your attention for a while (the driver still executes standing orders every turn and will wake you up
early on alerts)."""

SCHEMA = """Reply with exactly one JSON object inside a ```json fence, no other text after it:

```json
{
  "strategy_note": "one paragraph you want to read again next turn: your plan, priorities, threats (rewrite freely)",
  "end_turns": 1,                          // 1-5: how many turns to run before consulting you again
  "research_goal": "Monotheism",           // optional
  "research_now": "Mysticism",             // optional, must be researchable now
  "rates": [30, 70, 0],                    // optional: [tax, science, luxury]
  "government": "Republic",                // optional: starts a revolution
  "treaties": [ {"with": "Oleg", "type": "ceasefire"} ],   // optional: ceasefire | peace | war
  "cities": [ {"name": "Berlin", "build": "Library"}, {"name": "Essen", "build": "Settlers", "buy": true} ],
  "units": [
    {"id": 159, "do": "found_city", "to": [7, 26]},   // go there and found a city; omit "to" = nearest good legal site (works from a boat too)
    {"id": 138, "do": "explore"},
    {"id": 104, "do": "auto_work"},
    {"id": 160, "do": "fortify"},                     // military units only; "sentry" for others
    {"id": 161, "do": "move", "to": [9, 4]},
    {"id": 162, "do": "attack", "to": [9, 24]},       // move adjacent and attack the unit/city on that tile
    {"id": 163, "do": "conquer", "to": [9, 24]},      // move into an undefended enemy city
    {"id": 164, "do": "irrigate"},                    // also: mine, road, cultivate, plant, transform, pillage (here or with "to")
    {"id": 165, "do": "help_wonder", "to": [8, 29]},  // caravans: help the wonder built in the city on that tile
    {"id": 166, "do": "join_city", "to": [8, 29]},
    {"id": 167, "do": "disband"},
    {"id": 168, "do": "idle"}                         // cancel standing orders / automation
  ]
}
```
Only include the keys you want to change.  Unit ids and city names must be exactly as printed in the panel.
Your visible reply must be the ```json block and nothing else (think before it, not after it)."""


def system_prompt(ruleset: str, naive: bool = False) -> str:
    """`naive` (2026-09-19, doc 17 §4.6): the agent gets the interface, the ruleset reference and the order format, but
    NOT the expert manual and NOT the "what strong play looks like" priors.  Motivation: with the full priors the agent's
    own routes are noise-equivalent (§6.9) and no forecaster can add anything; a naive agent makes real strategic
    mistakes, which is the one setting in which a world model with world knowledge can be worth something."""
    primer = PRIMER[ruleset].read_text(encoding="utf-8")
    if naive:
        role = ROLE.split("\nWhat strong play looks like")[0].rstrip() + "\n\n" + ROLE[ROLE.index("Be decisive and economical"):]
        return role + "\n\n# RULESET REFERENCE\n\n" + primer + "\n\n# ORDER FORMAT\n\n" + SCHEMA
    manual = MANUAL.read_text(encoding="utf-8")
    # the manual was written for annotators predicting AI behaviour; its sections 12-15 are about answering prediction
    # questions and would only confuse a player -- keep 0-11 (game mechanics, AI behaviour, difficulty)
    cut = manual.find("## 12.")
    if cut > 0:
        manual = manual[:cut]
    primer = PRIMER[ruleset].read_text(encoding="utf-8")
    return (ROLE + "\n\n# GAME MANUAL (how this engine and its AI behave)\n\n" + manual +
            "\n\n# RULESET REFERENCE\n\n" + primer + "\n\n# ORDER FORMAT\n\n" + SCHEMA)


def strip_think(text: str) -> tuple[str, str]:
    """(answer, thinking).  vLLM returns the thinking inline, terminated by </think> (the opening tag is in the template)."""
    if "</think>" in text:
        th, ans = text.rsplit("</think>", 1)
        return ans.strip(), th.replace("<think>", "").strip()
    return text.strip(), ""


def extract_json(text: str) -> dict[str, Any]:
    m = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S)
    cands = m[::-1] if m else []
    if not cands:
        i, j = text.find("{"), text.rfind("}")
        if i >= 0 and j > i:
            cands = [text[i:j + 1]]
    last_err = None
    for c in cands:
        for attempt in (c, re.sub(r"//[^\n]*", "", c), re.sub(r",\s*([}\]])", r"\1", re.sub(r"//[^\n]*", "", c))):
            try:
                o = json.loads(attempt)
                if isinstance(o, dict):
                    return o
            except Exception as e:                 # noqa: BLE001
                last_err = e
    raise ValueError(f"no JSON object found ({last_err})")


def to_plan(o: dict[str, Any]) -> tuple[TurnPlan, list[str]]:
    """JSON -> TurnPlan; malformed entries become feedback lines instead of exceptions."""
    fb: list[str] = []
    plan = TurnPlan()
    if isinstance(o.get("research_goal"), str):
        plan.research_goal = o["research_goal"].strip()
    if isinstance(o.get("research_now"), str):
        plan.research_now = o["research_now"].strip()
    if o.get("rates") is not None:
        try:
            t, s, l = [int(x) for x in o["rates"]]
            plan.rates = (t, s, l)
        except Exception:                          # noqa: BLE001
            fb.append(f"REJECTED rates {o['rates']!r}: give [tax, science, luxury]")
    if isinstance(o.get("government"), str):
        plan.government = o["government"].strip()
    for t in o.get("treaties") or []:
        if isinstance(t, dict) and t.get("with") and t.get("type"):
            plan.treaties.append({"with": str(t["with"]), "type": str(t["type"])})
    for c in o.get("cities") or []:
        if not isinstance(c, dict) or "name" not in c:
            fb.append(f"REJECTED city entry {c!r}"); continue
        plan.cities.append(CityCmd(str(c["name"]).strip(), build=(str(c["build"]).strip() if c.get("build") else None), buy=bool(c.get("buy", False))))
    for u in o.get("units") or []:
        if not isinstance(u, dict) or "id" not in u or "do" not in u:
            fb.append(f"REJECTED unit entry {u!r}: needs id and do"); continue
        try:
            uid = int(u["id"])
        except Exception:                          # noqa: BLE001
            fb.append(f"REJECTED unit entry {u!r}: id must be an integer"); continue
        tgt = None
        if u.get("to") is not None:
            try:
                tgt = (int(u["to"][0]), int(u["to"][1]))
            except Exception:                      # noqa: BLE001
                fb.append(f"REJECTED unit {uid}: 'to' must be [x, y]"); continue
        plan.units.append(UnitCmd(uid, str(u["do"]).strip().lower().replace("-", "_").replace(" ", "_"), tgt))
    return plan, fb


class Agent:
    def __init__(self, endpoint: str = "http://localhost:8011/v1", model: str = "flash-next", ruleset: str = "civ2civ3",
                 max_tokens: int = 20000, effort: str = "medium", timeout: float = 900.0, naive: bool = False):
        self.endpoint, self.model, self.max_tokens, self.effort = endpoint, model, max_tokens, effort
        self.system = system_prompt(ruleset, naive)
        from bench.agentplay.rules import load_rules
        self.rules = load_rules(ruleset)
        self.client = httpx.Client(timeout=httpx.Timeout(timeout, connect=30.0))

    def call(self, messages: list[dict], max_tokens: int | None = None) -> dict[str, Any]:
        body = {"model": self.model, "messages": messages, "max_tokens": max_tokens or self.max_tokens,
                "temperature": 1.0, "top_p": 0.95, "top_k": 20,
                "chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": self.effort}}
        t0 = time.time()
        for attempt in range(4):
            try:
                r = self.client.post(self.endpoint + "/chat/completions", json=body)
                if r.status_code == 200:
                    js = r.json(); ch = js["choices"][0]
                    ans, think = strip_think(ch["message"].get("content") or "")
                    u = js.get("usage", {})
                    return {"answer": ans, "think": think, "finish": ch.get("finish_reason"), "prompt_tokens": u.get("prompt_tokens", 0),
                            "completion_tokens": u.get("completion_tokens", 0), "seconds": time.time() - t0}
                err = f"http{r.status_code}: {r.text[:200]}"
            except Exception as e:                 # noqa: BLE001
                err = f"{type(e).__name__}: {e}"
            time.sleep(5 * (attempt + 1))
        return {"answer": "", "think": "", "finish": f"error:{err}", "prompt_tokens": 0, "completion_tokens": 0, "seconds": time.time() - t0}

    PROPOSE = """This is a planning turn.  Before giving orders, write 2 or 3 clearly different candidate plans for the next
~20 turns.  A world-model forecaster will read each candidate as your civilization's declared plan and predict the
consequences (your score, cities, gold, technologies, military, enemies, and the other civilizations' trajectories);
you will then see the forecasts and decide.  Each candidate must be written as the plan lines a forecaster can use:
  Development: <expand / build up cities / prepare for war / trade and research>, with the concrete production mix
  Research aim: <technology>
  War aim: <none | attack X in about N turns with which units>
  Settlement: <how many new cities, where (tile coordinates from CANDIDATE CITY SITES)>
  Budget: <tax/science/luxury>
  Defence: <what garrisons where>
Reply with one ```json object: {"candidates": [{"name": "...", "plan": "Development: ...\\nResearch aim: ...\\nWar aim: ...\\nSettlement: ...\\nBudget: ...\\nDefence: ..."}, ...]}"""

    PROPOSE2 = """This is a planning turn.  Write exactly 3 candidate plans for the next 20 turns that are GENUINELY DIFFERENT
in what they do -- e.g. one expansion-heavy, one defence/military-heavy, one economy/technology-heavy -- each with
concrete numbers (how many Settlers/defenders/buildings, which cities, which sites from CANDIDATE CITY SITES, what tax rates).
A world-model forecaster will read them side by side as your declared plan and predict the consequences (your score,
cities, gold, technologies, military, enemies, and the other civilizations' trajectories) at +10 and +20 turns; it will
also show you how its earlier forecasts compared with what actually happened.  You will then have to COMMIT to one of them.
Each candidate = the six plan lines a forecaster can use:
  Development: <expand / build up cities / prepare for war / trade and research>, with the concrete production mix
  Research aim: <technology>
  War aim: <none | attack X in about N turns with which units>
  Settlement: <how many new cities, where (tile coordinates from CANDIDATE CITY SITES)>
  Budget: <tax/science/luxury>
  Defence: <what garrisons where>
Reply with one ```json object: {"candidates": [{"name": "...", "plan": "Development: ...\\nResearch aim: ...\\nWar aim: ...\\nSettlement: ...\\nBudget: ...\\nDefence: ..."}, ...]}"""

    def propose(self, report: str, panel: str, note: str, turn: int, v2: bool = False) -> dict[str, Any]:
        user = (f"# TURN {turn} (planning turn)\n\n## Your standing strategy note\n{note or '(none yet)'}\n\n## World report\n{report}\n\n## Panel\n{panel}\n\n"
                + (self.PROPOSE2 if v2 else self.PROPOSE))
        out = self.call([{"role": "system", "content": self.system}, {"role": "user", "content": user}])
        cands = []
        try:
            obj = extract_json(out["answer"])
            for c in obj.get("candidates") or []:
                if isinstance(c, dict) and c.get("plan"):
                    cands.append({"name": str(c.get("name", f"plan {len(cands) + 1}"))[:60], "plan": str(c["plan"])[:1500]})
        except ValueError:
            pass
        return {"candidates": cands[:3], "llm": out}

    ROUTES = """This is a planning point.  A forecaster (a model of this world that has followed the whole game so far) will read
your routes as your declared plan and predict, for each: your quantities at +20 and +40 turns, the probability of concrete
events (losing a city, founding cities, a military collapse, bankruptcy, new enemies, a government change, a wonder, a
capture), the same for your neighbours, what each route COSTS (gold, shields, settler-turns) and how much it changes your
city-loss risk, and the biggest risk on each route.  It will not tell you which route to take.  You will see all of it and
may then revise your routes and ask again (up to 3 rounds) before you commit.
Write exactly THREE routes: A = the course you currently intend to follow for the next 20-40 turns; B and C = two
alternatives you are genuinely weighing.  Each must differ from A in at least one major decision.
If the panel has a SIEGE ARITHMETIC block (a city of yours can fall), then B and C must each answer that threat with a
DIFFERENT response than A, chosen from this list and named in a `Response:` line:
  reinforce   -- more defenders / City Walls in the threatened city (the block tells you whether that changes the odds)
  abandon     -- accept the loss; move out what can be saved, spend nothing on the city, invest in growth elsewhere
  relocate    -- do not found / re-found at that site; send the Settlers to a safer site instead
  diplomacy   -- seek a ceasefire or peace first, before anything else
  preempt     -- attack the approaching stack with what you have before it reaches the city
  ignore      -- keep the current plan unchanged and take the risk
Each route = the plan lines a forecaster can use:
  Course: <one paragraph: what you do and in what order, with turn targets>
  Response: <one word from the list above, or none>
  Research aim: <technology / chain>
  War aim: <none | attack X in about N turns with which units | seek ceasefire with X>
  Settlement: <how many new cities, where (from CANDIDATE CITY SITES)>
  Budget: <tax/science/luxury>
  Defence: <what garrisons where>
Reply with one ```json object: {"routes": [{"name": "...", "response": "...", "plan": "Course: ...\\nResponse: ...\\nResearch aim: ...\\nWar aim: ...\\nSettlement: ...\\nBudget: ...\\nDefence: ..."}, {...}, {...}]}"""

    ROUTES6 = """This is a planning point.  The route you commit to becomes your COURSE for the next {lock} turns: the driver
will put its text in front of you verbatim on every turn until the next planning point, and will track what you actually
build against the numbers you write into it.  Nothing is enforced -- the tactics of each turn are yours, and you may leave
the course if the situation changes -- but it is the direction you are committing to, so write routes you would actually
follow, and make them DIFFERENT from each other: two routes whose numbers are the same are the same route.

A forecaster (a model of this world that has followed the whole game) will read the three routes and predict for each:
your quantities at +20 and +40 turns, the probability of concrete events (losing a city, founding cities, a military
collapse, bankruptcy, new enemies, a government change, a wonder, a capture), the same for your neighbours, what each
route COSTS -- and whether you are in a SAFE WINDOW, in which case it must tell you the fastest way to turn everything
you hold into score.  It knows the measured frequency of each of those events in games like this one.  It will not tell
you which route to take.  You may revise and ask again (up to {n} rounds) before you commit.

Write exactly THREE routes:
  A = the course you currently intend to follow for the next 20-40 turns.
  B, C = two alternatives you are genuinely weighing; each must differ from A in at least one major decision AND in its
  contract numbers.
{extra}

Each route is the plan lines a forecaster can use, plus a contract:
  Course: <one paragraph: what you do and in what order, with turn targets>
  Response: <reinforce | abandon | relocate | diplomacy | preempt | ignore | growth>
  Research aim: <technology / chain>
  War aim: <none | attack X in about N turns with which units | seek ceasefire with X>
  Settlement: <how many new cities, where (tile coordinates from CANDIDATE CITY SITES)>
  Budget: <tax/science/luxury>
  Defence: <what garrisons where>
and a "contract" object with the numbers the driver will hold you to:
  "plan_numbers": {{"rates": [tax, science, luxury], "research_goal": "<technology>", "found_cities": <n>,
                "build": {{"settlers": n, "workers": n, "defenders": n, "army": n, "buildings": n, "wonder": n}},
                "max_treasury": <the gold level above which you consider yourself to be hoarding>}}
"build" is what you intend to COMPLETE in the next {lock} turns; the driver counts what the engine actually finishes and
shows you both numbers every turn, so you can see whether the course you chose is the course you are playing.
★ These numbers are how the forecaster tells your routes apart.  Write them concretely: if two routes would produce the
same cities, the same units and the same research, they are one route and the comparison tells you nothing.
★ Gold in your treasury is worth zero points at the end of the game, so a surplus you never convert into citizens,
technologies or a wonder is wasted -- but the treasury also pays upkeep, and if it reaches zero the engine SELLS your
units and buildings to cover it.  "max_treasury" is the level above which you want to be buying production; set it
ABOVE the working reserve your units and buildings need, not at zero.
Reply with one ```json object:
{{"routes": [{{"name": "...", "response": "...", "plan": "Course: ...\\nResponse: ...\\nResearch aim: ...\\nWar aim: ...\\nSettlement: ...\\nBudget: ...\\nDefence: ...", "plan_numbers": {{...}}}}, {{...}}, {{...}}]}}"""

    SIEGE_CLAUSE = """The panel has a SIEGE ARITHMETIC block: a city of yours can fall.  B and C must each answer that threat with a
DIFFERENT response than A, named in the `Response:` line -- reinforce (more defenders / City Walls; the block tells you
whether that actually changes the odds), abandon (accept the loss, save what can be saved, spend nothing there),
relocate (do not found / re-found at that site), diplomacy (seek a ceasefire first), preempt (attack the stack before
it arrives), ignore (keep the plan and take the risk).  The block also gives the measured fact that a city listed there
was actually lost within 20 turns in only 6% of cases: reinforcing is insurance, and insurance has a price."""

    GROWTH_CLAUSE = """The panel has NO SIEGE ARITHMETIC block: no city of yours can currently fall.  Therefore one of the three routes
MUST be an all-in growth route with `Response: growth`.  That route spends the treasury down to its "max_treasury"
(a working reserve, not zero -- an empty treasury makes the engine sell your units), plans 0 in the "army" category,
and puts every shield and every surplus coin into SCORE: bigger cities, faster research, a wonder -- and new cities
only where the panel's CANDIDATE CITY SITES actually lists a site.  ★ A Settlers unit costs you 2 CITIZENS the moment
it is built; with no site to found on, that is 2 points thrown away, so read that list before promising settlers.  Write its numbers so that it is genuinely the most aggressive growth plan you can
survive -- if it turns out to be wrong, the forecaster will say so."""

    MANDATES = {
        "growth": ("MAXIMUM GROWTH", """Write the plan that ends the game with the most citizens and the most technologies, and nothing else.
Spend the treasury down -- gold scores nothing at the end.  Found every city you can reach, grow the ones you have, and buy the
buildings that grow them.  Build no offensive units and no wonder unless it pays for itself in this window.  Garrison only what
would otherwise be lost this window."""),
        "consolidate": ("CONSOLIDATE AND OUT-RESEARCH", """Write the plan that stops expanding and compounds what you already hold: buildings that raise
trade and science in your best cities, a research chain chosen for what it unlocks, trade routes and caravans, a wonder if your
best production city can finish one.  Accept a smaller empire that is richer per city."""),
        "security": ("SECURITY FIRST", """Write the plan that makes your position unloseable for the next 20-40 turns: garrisons and City Walls where
the arithmetic says they change the odds, a ceasefire or peace with whoever is closest to attacking you, and -- if the numbers
favour it -- destroying the threatening stack before it arrives.  Growth is secondary in this plan."""),
    }

    ONE_ROUTE = """This is a planning point.  You are writing ONE candidate plan for the next 20-40 turns, under a mandate given below.
Two other candidates are being written at the same time under DIFFERENT mandates, and you will see all three side by side, with a
world-model forecaster's predictions for each, before you choose.

Your mandate for THIS candidate: **{title}**
{mandate}

Write the most committed version of this mandate that your civilization can actually survive.  ★ Do NOT hedge: a plan that does a
little of everything is the plan you would have written anyway, and it tells you nothing when you compare it with the others.
If this mandate is wrong for your situation, that is what the forecasts are for -- write it properly and let the comparison decide.
{siege}
Answer with the plan lines a forecaster can use, plus the numbers you intend to hold yourself to:
  Course: <one paragraph: what you do and in what order, with turn targets>
  Response: <reinforce | abandon | relocate | diplomacy | preempt | ignore | growth>
  Research aim: <technology / chain>
  War aim: <none | attack X in about N turns with which units | seek ceasefire with X>
  Settlement: <how many new cities, where (from CANDIDATE CITY SITES)>
  Budget: <tax/science/luxury>
  Defence: <what garrisons where>
and a "plan_numbers" object -- your own intent, which the driver will track against what actually gets built, but will NOT enforce:
  {{"rates": [tax, science, luxury], "research_goal": "<technology>", "found_cities": <n>,
    "build": {{"settlers": n, "workers": n, "defenders": n, "army": n, "buildings": n, "wonder": n}},
    "max_treasury": <the gold level above which you consider yourself to be hoarding>}}
★ THIS IS NOT A TURN OF ORDERS.  Do not output "units", "cities" or "end_turns" -- no unit moves, no production orders.
You are writing a PLAN that a forecaster will read.  The only keys allowed are name, response, plan and plan_numbers.
Reply with one ```json object: {{"name": "...", "response": "...", "plan": "Course: ...\\nResponse: ...\\nResearch aim: ...\\nWar aim: ...\\nSettlement: ...\\nBudget: ...\\nDefence: ...", "plan_numbers": {{...}}}}"""

    PLAN_LINES = ("Course:", "Response:", "Research aim:", "War aim:", "Settlement:", "Budget:", "Defence:")

    @classmethod
    def _route_from(cls, obj: dict[str, Any] | None, answer: str, key: str) -> dict[str, Any] | None:
        """Accept the several shapes the model actually returns: the plan object, {"routes": [...]}, {"route": {...}},
        or -- when it slipped into the ORDER FORMAT and answered with a turn of orders -- the plan lines from the raw
        text.  (Measured 2026-09-19: 1 of 3 mandate calls came back as unit/city orders.)"""
        o = obj if isinstance(obj, dict) else {}
        for k in ("route", "plan_object"):
            if isinstance(o.get(k), dict):
                o = o[k]; break
        if isinstance(o.get("routes"), list) and o["routes"] and isinstance(o["routes"][0], dict):
            o = o["routes"][0]
        plan = o.get("plan") if isinstance(o.get("plan"), str) else None
        if not plan:
            lines = [ln.strip() for ln in (answer or "").splitlines()
                     if any(ln.strip().lstrip("*# -").startswith(h) for h in cls.PLAN_LINES)]
            plan = "\n".join(lines) if len(lines) >= 3 else None
        if not plan:
            return None
        return {"name": str(o.get("name", key))[:60], "plan": plan[:1800],
                "response": str(o.get("response", "none"))[:20], "mandate": key,
                "contract": o.get("plan_numbers") or o.get("contract")}

    SIEGE_NOTE = """The panel has a SIEGE ARITHMETIC block: a city of yours can fall.  Whatever your mandate, your plan must say in its
`Response:` line what it does about that city, and the block's own numbers tell you whether reinforcing changes the odds -- a city
listed there was actually lost within 20 turns in only 6% of cases, so insurance has a price and you should name it.
"""

    def plan_routes7(self, report: str, panel: str, note: str, turn: int, siege: bool) -> dict[str, Any]:
        """★ v7 (doc 17 §4.8): three routes from three INDEPENDENT calls under three different mandates.

        v5/v6 asked one call for three routes.  Measured (§6.9): the three came out 95% different in text and
        indistinguishable in outcome -- one model, one sample, one prior, writing three variations of the same idea.
        Three independent samples, each told to write the most committed version of a DIFFERENT mandate and told not
        to hedge, is diversity by construction rather than by instruction."""
        import concurrent.futures as cf
        keys = ["growth", "consolidate", "security"]
        head = (f"# TURN {turn} (planning point)\n\n## Your standing strategy note\n{note or '(none yet)'}\n\n"
                f"## World report\n{report}\n\n## Panel\n{panel}\n\n")

        def one(k: str) -> dict:
            title, mandate = self.MANDATES[k]
            q = self.ONE_ROUTE.format(title=title, mandate=mandate, siege=(self.SIEGE_NOTE if siege else ""))
            msgs = [{"role": "system", "content": self.system}, {"role": "user", "content": head + q}]
            out = self.call(msgs)
            try:
                o = extract_json(out["answer"])
            except ValueError:
                o = None
            rt = self._route_from(o, out["answer"], k)
            if rt is None:                                   # one repair round, as `decide` does
                out2 = self.call(msgs + [{"role": "assistant", "content": out["answer"][-4000:]},
                                         {"role": "user", "content": "That was a turn of orders, not a plan. Reply again with ONLY the "
                                          "```json object {\"name\", \"response\", \"plan\", \"plan_numbers\"}, where \"plan\" is the seven "
                                          "lines Course/Response/Research aim/War aim/Settlement/Budget/Defence as one string."}])
                out["completion_tokens"] += out2["completion_tokens"]; out["prompt_tokens"] += out2["prompt_tokens"]
                out["seconds"] += out2["seconds"]; out["repaired"] = True
                try:
                    o2 = extract_json(out2["answer"])
                except ValueError:
                    o2 = None
                rt = self._route_from(o2, out2["answer"], k)
            return {"mandate": k, "llm": out, "route": rt}

        with cf.ThreadPoolExecutor(max_workers=3) as ex:
            outs = list(ex.map(one, keys))
        return {"routes": [o["route"] for o in outs if o["route"]], "llm": outs[0]["llm"],
                "calls": [{k: v for k, v in o["llm"].items() if k != "think"} for o in outs],
                "mandates": [o["mandate"] for o in outs if o["route"]]}

    REVISE6 = """Above is the planning dialogue so far (this was round {r} of at most {n}).  Decide:
- COMMIT if one of the routes (by its letter) is what you will follow -- reply {{"action": "commit", "route": "<letter>", "why": "..."}}
- REVISE if the forecasts show a risk or an opportunity worth a changed plan: write 1 or 2 revised routes, each with its
  own full plan block AND its own "plan_numbers" object, and say which forecast, safe-window line, growth plan or
  siege-arithmetic line motivated each change -- reply
  {{"action": "revise", "why": "...", "routes": [{{"name": "...", "response": "...", "plan": "Course: ...", "plan_numbers": {{...}}}}]}}
Weigh what you are being told: a reinforcement the siege block says does not change the odds is money spent for nothing;
a safe window you spend on insurance is score you will not have at the end; gold you are still holding at the end of the
game scores nothing.  Reply with exactly one ```json object."""

    def plan_routes6(self, report: str, panel: str, note: str, turn: int, siege: bool, lock: int, n_rounds: int) -> dict[str, Any]:
        q = self.ROUTES6.format(lock=lock, n=n_rounds, extra=(self.SIEGE_CLAUSE if siege else self.GROWTH_CLAUSE))
        user = (f"# TURN {turn} (planning point)\n\n## Your standing strategy note\n{note or '(none yet)'}\n\n"
                f"## World report\n{report}\n\n## Panel\n{panel}\n\n" + q)
        out = self.call([{"role": "system", "content": self.system}, {"role": "user", "content": user}])
        routes = []
        try:
            obj = extract_json(out["answer"])
            for r in obj.get("routes") or []:
                if isinstance(r, dict) and r.get("plan"):
                    routes.append({"name": str(r.get("name", f"route {len(routes) + 1}"))[:60], "plan": str(r["plan"])[:1800],
                                   "response": str(r.get("response", "none"))[:20], "contract": r.get("plan_numbers") or r.get("contract")})
        except ValueError:
            pass
        return {"routes": routes[:3], "llm": out}

    def revise_or_commit6(self, report: str, panel: str, note: str, turn: int, transcript: str, rnd: int, n_rounds: int) -> dict[str, Any]:
        user = (f"# TURN {turn} (planning point, round {rnd})\n\n## Your standing strategy note\n{note or '(none yet)'}\n\n"
                f"## World report\n{report}\n\n## Panel\n{panel}\n\n## Planning dialogue so far\n{transcript}\n\n"
                + self.REVISE6.format(r=rnd, n=n_rounds))
        out = self.call([{"role": "system", "content": self.system}, {"role": "user", "content": user}])
        action, route, why, routes = "commit", None, "", []
        try:
            obj = extract_json(out["answer"])
            action = str(obj.get("action", "commit")).strip().lower()
            why = str(obj.get("why", ""))[:1500]
            route = str(obj.get("route", "")).strip().upper()[:1] or None
            for r in obj.get("routes") or []:
                if isinstance(r, dict) and r.get("plan"):
                    routes.append({"name": str(r.get("name", f"revised {len(routes) + 1}"))[:60], "plan": str(r["plan"])[:1800],
                                   "response": str(r.get("response", "none"))[:20], "contract": r.get("plan_numbers") or r.get("contract")})
        except ValueError:
            action = "commit"
        if action != "revise" or not routes:
            action = "commit"; routes = []
        return {"action": action, "route": route, "why": why, "routes": routes[:2], "llm": out}

    REVISE = """Above is the planning dialogue so far (this was round {r} of at most {n}).  Decide:
- COMMIT if one of the routes (by its letter) is what you will follow -- reply {{"action": "commit", "route": "<letter>", "why": "..."}}
- REVISE if the forecasts show a risk or an opportunity worth a changed plan: write 1 or 2 revised routes (full plans in the
  same format, each a concrete change of an earlier route -- and if a threatened city is involved, a response the earlier
  routes did not try), and say which forecast or siege-arithmetic line motivated each change.  Weigh the COST line: a
  reinforcement that the siege block says does not change the odds is money spent for nothing -- reply
  {{"action": "revise", "why": "...", "routes": [{{"name": "...", "plan": "Course: ...\\nResearch aim: ...\\nWar aim: ...\\nSettlement: ...\\nBudget: ...\\nDefence: ..."}}]}}
Reply with exactly one ```json object."""

    def plan_routes(self, report: str, panel: str, note: str, turn: int) -> dict[str, Any]:
        user = (f"# TURN {turn} (planning point)\n\n## Your standing strategy note\n{note or '(none yet)'}\n\n## World report\n{report}\n\n## Panel\n{panel}\n\n" + self.ROUTES)
        out = self.call([{"role": "system", "content": self.system}, {"role": "user", "content": user}])
        routes = []
        try:
            obj = extract_json(out["answer"])
            for r in obj.get("routes") or []:
                if isinstance(r, dict) and r.get("plan"):
                    routes.append({"name": str(r.get("name", f"route {len(routes) + 1}"))[:60], "plan": str(r["plan"])[:1800],
                                   "response": str(r.get("response", "none"))[:20]})
        except ValueError:
            pass
        return {"routes": routes[:3], "llm": out}

    def revise_or_commit(self, report: str, panel: str, note: str, turn: int, transcript: str, rnd: int, n_rounds: int) -> dict[str, Any]:
        user = (f"# TURN {turn} (planning point, round {rnd})\n\n## Your standing strategy note\n{note or '(none yet)'}\n\n## World report\n{report}\n\n## Panel\n{panel}\n\n"
                f"## Planning dialogue so far\n{transcript}\n\n" + self.REVISE.format(r=rnd, n=n_rounds))
        out = self.call([{"role": "system", "content": self.system}, {"role": "user", "content": user}])
        action, route, why, routes = "commit", None, "", []
        try:
            obj = extract_json(out["answer"])
            action = str(obj.get("action", "commit")).strip().lower()
            why = str(obj.get("why", ""))[:1500]
            route = str(obj.get("route", "")).strip().upper()[:1] or None
            for r in obj.get("routes") or []:
                if isinstance(r, dict) and r.get("plan"):
                    routes.append({"name": str(r.get("name", f"revised {len(routes) + 1}"))[:60], "plan": str(r["plan"])[:1800],
                                   "response": str(r.get("response", "none"))[:20]})
        except ValueError:
            action = "commit"
        if action != "revise" or not routes:
            action = "commit"; routes = []
        return {"action": action, "route": route, "why": why, "routes": routes[:2], "llm": out}

    CHOOSE = """The planning dialogue above is finished.  Choose the route you will follow for the next {lock} turns.
★ Before you answer, look at three things in the forecaster's reply and quote the numbers you used:
  1. THE NUMBER YOU ARE JUDGED ON -- each route's forecast TOTAL SCORE at the final turn.  That is the number that
     decides whether you did well, not the +20 gain.
  2. ROUTE RISK -- the driver's arithmetic for how many of your cities fall on each route if the stacks it can see
     assault them.  It is computed from the ruleset, not guessed.
  3. What each route costs and what the forecaster says it is still wasting.
A route that is 5 points better at +20 and 20 points worse at the final turn is the wrong route.
Reply with exactly one ```json object and nothing else:
{{"route": "<letter>", "why": "<one or two sentences that quote the final-turn scores and the route risk you compared>"}}"""

    def choose_route(self, report: str, panel: str, note: str, turn: int, transcript: str, letters: list[str],
                     lock: int) -> dict[str, Any]:
        """★ v11 (doc 17 §4.11).  Measured on the v9 runs: at 32% of planning points the agent's ORDER object came
        back without a usable `chosen_route`, so the driver silently fell back to the first route and the whole
        dialogue was discarded; and of the choices it did make, 66% were route A, its own opening intention, while
        the two alternatives it had written itself took 8.5% between them.  Asking for the choice on its own, in a
        small call that must quote the numbers, removes both the parse failure and most of the anchoring."""
        user = (f"# TURN {turn} (planning point, commit)\n\n## Your standing strategy note\n{note or '(none yet)'}\n\n"
                f"## Panel\n{panel}\n\n## Planning dialogue\n{transcript}\n\n" + self.CHOOSE.format(lock=lock))
        out = self.call([{"role": "system", "content": self.system}, {"role": "user", "content": user}])
        route, why = None, ""
        try:
            o = extract_json(out["answer"])
            r = str(o.get("route", "")).strip().upper()[:1]
            if r in letters:
                route = r
            why = str(o.get("why", ""))[:600]
        except ValueError:
            pass
        if route is None:                                   # last resort: the first letter mentioned in the reply
            import re as _re
            m = _re.search(r"\b([A-J])\b", out["answer"][-400:] or "")
            if m and m.group(1) in letters:
                route = m.group(1)
        return {"route": route, "why": why, "llm": out}

    # ======================================================================================================
    # v12: the planning dialogue works on PLANS, not on a menu of letters (doc 17 §4.12)
    # ======================================================================================================
    PLAN_SCHEMA = """{{
  // ---------- THE THINKING.  Free text, as long as you want.  ALL reasoning, caveats, conditionals and
  //            alternatives you considered go here.  Nothing here is parsed, so write it for yourself.
  "name": "short name",
  "intent": "what this plan is for, what it bets on, what would make it wrong, what you considered and rejected,
             and anything the fields below cannot express",
  "notes": {{"<city>": "why that city's queue is what it is"}},

  // ---------- THE EXECUTABLE CORE.  Bare values only -- no explanations, no parentheses, no arrows.
  //            The driver reads these literally: it sets cities to these exact names every turn, checks these
  //            tiles against the map, and computes your city-loss risk from these garrison numbers.
  "production": {{"<city>": [{{"item": "Settlers", "by_turn": {t1}}}, {{"item": "Phalanx", "by_turn": 0}}],
                 "<city>": [{{"item": "City Walls", "by_turn": {t2}}}]}},
  "settlement": [{{"unit": <numeric id of an existing Settlers, or 0>, "escort": <numeric id or 0>,
                  "target": [x, y] or null, "by_turn": {t2}, "why": "short"}}],
  "found_cities": <how many new cities you intend in the next 20 turns>,
  "buy": [{{"city": "<city>", "item": "<item>", "why": "why it cannot wait"}}],
  "garrison": {{"<city>": 2, "<city>": 1}},
  "military": "what you build beyond the garrison and what you will NOT build",
  "research": {{"now": "<a technology name or empty>", "goal": "<a technology name>"}},
  "budget": {{"rates": [tax, science, luxury], "treasury_floor": <gold you will not spend below>}},
  "diplomacy": "who you approach and when, or none",
  "triggers": ["if <observable event>, <what you do>"],
  "milestones": [{{"by_turn": {t3}, "expect": "concrete state: cities, citizens, techs, what is built"}}]
}}"""

    PLAN_WRITE = """This is a planning point (turn {turn}).  You are going to write your plan for the next 20 to 40 turns,
then a world-model forecaster will predict what happens under it and tell you what to change, and you will revise it.
What comes out of this conversation is the plan you actually play.

★ The driver EXECUTES the production queue you write.  Every turn each of your cities is set to the next unfinished
item of its queue, until you change it in that turn's orders.  So write queues you mean, in the order you mean them.

Write {n_plans} candidate plan(s) -- one is fine if you only see one sensible course; write two or three only if you are
genuinely torn, and then make them differ in what they DO, not in how they are described.

Every plan must be operational, at the level of detail you would need to play it without thinking again:
  * name the cities by the exact names in the panel and the units by the exact ids;
  * a production queue per city you intend to direct, in order, with the turn you expect each item;
  * for each settler: which unit, escorted by which unit, to which tile from CANDIDATE CITY SITES, by which turn;
  * how many defenders you want sitting in each city that can be reached by anyone;
  * the research you switch to now and the goal you steer to;
  * tax/science/luxury and the treasury level you will not spend below;
  * triggers: concrete observable events and what you will do when they happen;
  * milestones: what you expect to be true at a stated turn, so you can find out later whether you were right.
A plan that says "expand and build up the economy" is not a plan.  A plan that says "Hattusas: Settlers by T{t1},
then Phalanx, then Library; Settlers 201 escorted by Phalanx 188 founds (14,22) by T{t2}" is a plan.

★ The split matters.  "intent" and "notes" are yours: write as much as you want there, including the parts of your
thinking that do not fit any field.  Everything below them is read literally by a program:
   - "item" must be EXACTLY a name from that city's can-build list -- "Settlers", not "Settlers (28/30 stored)" and
     not "then Workers".  A name the driver cannot find is a line of your plan that will not run, and it will tell
     you so.  The turn goes in "by_turn"; the reason goes in "notes".
   - in "settlement", LEAVE `target` as null unless you specifically want one tile.  With no target the driver walks
     the settler to the nearest good legal site itself, re-choosing as the map opens up, and that is both faster and
     more reliable than a tile you fix now -- a tile can stop being legal before the settler arrives.  Name a tile
     only when the reason is in `why`.  `unit` is a bare number, 0 if the settler does not exist yet.
   - "buy" is for things that must not wait: the driver rush-buys them as soon as the treasury allows while staying
     above your treasury_floor.  Gold you never spend scores nothing, so say what it is for.
   - "research" takes bare technology names.
★ When you revise, restate only the sections you are changing; anything you leave out keeps its previous content.

Reply with one ```json object: {{"plans": [PLAN, ...]}} where each PLAN is
{schema}"""

    PLAN_REVISE = """Above is the forecaster's reply to your plan(s), together with the driver's own arithmetic (which is
computed from the ruleset and the map, not predicted: treat it as fact).  This was round {r} of at most {n}.

Do one of these, and say which:
  * EDIT a plan -- change the lines the forecaster or the driver's checks showed to be wrong or wasteful, and send the
    whole plan back with those lines changed;
  * DROP a plan you now think is not worth forecasting again;
  * MERGE two plans into one;
  * FINALISE -- you are done, and the single plan you send is what you will play.
{must}
Whatever you do, name the specific finding you are responding to (a probability, an expected loss, an illegal site, a
queue that does not fit in the shields) and what you changed because of it.  If you are keeping something the
forecaster criticised, say why you are keeping it.

Reply with one ```json object:
{{"final": true|false,
  "changes": "what you changed and which finding drove each change; if you kept something criticised, why",
  "plans": [PLAN, ...]        // the full plans as they now stand; exactly one plan when final is true
  {logline}}}
where PLAN is the same schema as before."""

    LOGLINE = (",\n  \"decision_log\": \"600-900 characters you will re-read on EVERY turn until "
               "the next planning point: what you committed to and why, the numbers you relied on, the main "
               "risk the forecaster named and the turn range it expected it in, what you rejected and why, "
               "and what would tell you this decision was wrong\"")

    def plan_write(self, report: str, panel: str, log: str, turn: int, n_plans: int = 2) -> dict[str, Any]:
        sch = self.PLAN_SCHEMA.format(t1=turn + 4, t2=turn + 10, t3=turn + 20)
        q = self.PLAN_WRITE.format(turn=turn, n_plans=("one or two" if n_plans <= 2 else "one to three"),
                                   t1=turn + 4, t2=turn + 10, schema=sch)
        user = (f"# TURN {turn} (planning point)\n\n## Your decision log from the last planning point\n{log or '(none yet)'}\n\n"
                f"## World report\n{report}\n\n## Panel\n{panel}\n\n" + q)
        from bench.agentplay.plan import parse_plan
        msgs = [{"role": "system", "content": self.system}, {"role": "user", "content": user}]
        out = self.call(msgs, max_tokens=max(self.max_tokens, 32000))

        def harvest(ans: str) -> list[dict]:
            got = []
            try:
                o = extract_json(ans)
            except ValueError:
                return got
            raw = o.get("plans") or ([o] if o.get("production") or o.get("intent") else [])
            for x in raw[:3]:
                pp = parse_plan(x, self.rules)
                if pp:
                    got.append(pp)
            return got

        plans = harvest(out["answer"])
        if not plans:
            # ★ measured: 5% of first rounds came back with finish_reason "length" -- the free-form intent ran the
            #   reply past the token budget and the JSON was cut off mid-object, so the planning point had no plan
            #   at all.  Ask again, shorter, once.
            out2 = self.call(msgs + [{"role": "assistant", "content": (out["answer"] or "")[-2000:]},
                                     {"role": "user", "content": "Your reply did not contain a complete JSON object "
                                      "(it was cut off or malformed). Send it again: keep \"intent\" under 150 words "
                                      "per plan, write at most two plans, and make sure the JSON closes."}],
                             max_tokens=max(self.max_tokens, 32000))
            out["completion_tokens"] += out2["completion_tokens"]; out["prompt_tokens"] += out2["prompt_tokens"]
            out["seconds"] += out2["seconds"]; out["repaired"] = True
            plans = harvest(out2["answer"])
            out["retry_answer"] = out2["answer"][:8000]
        return {"plans": plans, "llm": out}

    def plan_revise(self, report: str, panel: str, log: str, turn: int, transcript: str, rnd: int, n_rounds: int,
                    must_revise: bool) -> dict[str, Any]:
        must = ("★ This is the first round, so FINALISE is not available yet: the forecaster has just told you things you\n"
                "did not know when you wrote the plan.  Either change something because of it, or drop a plan -- and if\n"
                "you genuinely think nothing should change, say so in `changes` and send the plan back unchanged with\n"
                "final false.\n" if must_revise else "")
        q = self.PLAN_REVISE.format(r=rnd, n=n_rounds, must=must, logline=self.LOGLINE)
        user = (f"# TURN {turn} (planning point, round {rnd})\n\n## Your decision log from the last planning point\n{log or '(none yet)'}\n\n"
                f"## Panel\n{panel}\n\n## The planning conversation so far\n{transcript}\n\n" + q)
        out = self.call([{"role": "system", "content": self.system}, {"role": "user", "content": user}],
                        max_tokens=max(self.max_tokens, 32000))
        plans = []; final = False; changes = ""; dlog = ""
        try:
            o = extract_json(out["answer"])
            final = bool(o.get("final"))
            changes = str(o.get("changes", ""))[:1500]
            dlog = str(o.get("decision_log", ""))[:1200]
            from bench.agentplay.plan import parse_plan
            for x in (o.get("plans") or [])[:3]:
                pp = parse_plan(x, self.rules)
                if pp:
                    plans.append(pp)
        except ValueError:
            pass
        return {"final": final, "plans": plans, "changes": changes, "decision_log": dlog, "llm": out}

    def rank(self, report: str, panel: str, note: str, turn: int, cand_texts: list[str]) -> dict[str, Any]:
        """Control for best-of-K: the same model, without any forecast, picks which of its K sampled decisions is best."""
        letters = "ABCDE"[:len(cand_texts)]
        user = (f"# TURN {turn}\n\n## Your standing strategy note\n{note or '(none yet)'}\n\n## World report\n{report}\n\n## Panel\n{panel}\n\n"
                "## Candidate decisions for this turn (each = strategy note + orders)\n" + "\n\n".join(f"### {L_}\n{t}" for L_, t in zip(letters, cand_texts))
                + "\n\nWhich candidate gives the highest score at the end of the game? Reply with one ```json object: {\"best\": \"<letter>\", \"why\": \"...\"}")
        out = self.call([{"role": "system", "content": self.system}, {"role": "user", "content": user}])
        best = None
        try:
            obj = extract_json(out["answer"]); b = str(obj.get("best", "")).strip().upper()[:1]
            if b in letters:
                best = b
        except ValueError:
            pass
        return {"best": best, "llm": out}

    def decide(self, report: str, panel: str, note: str, turn: int, extra: str = "") -> dict[str, Any]:
        user = (f"# TURN {turn}\n\n## Your standing strategy note\n{note or '(none yet)'}\n\n## World report\n{report}\n\n## Panel\n{panel}\n\n"
                + (extra + "\n\n" if extra else "")
                + "Give your orders for this turn as one ```json object (see ORDER FORMAT).")
        msgs = [{"role": "system", "content": self.system}, {"role": "user", "content": user}]
        out = self.call(msgs)
        plan, fb, obj = TurnPlan(), [], None
        parse_ok = False
        if out["answer"]:
            try:
                obj = extract_json(out["answer"]); parse_ok = True
            except ValueError as e:
                # one repair round: show the model its own reply and the error
                msgs2 = msgs + [{"role": "assistant", "content": out["answer"]},
                                {"role": "user", "content": f"Your reply could not be parsed as a JSON order object ({e}). Reply again with only the ```json object."}]
                out2 = self.call(msgs2)
                out["completion_tokens"] += out2["completion_tokens"]; out["prompt_tokens"] += out2["prompt_tokens"]; out["seconds"] += out2["seconds"]
                out["retry_answer"] = out2["answer"]
                try:
                    obj = extract_json(out2["answer"]); parse_ok = True
                except ValueError as e2:
                    fb.append(f"PARSE FAILURE: your orders could not be read ({e2}); the turn ran with no new orders")
        else:
            fb.append(f"LLM call failed ({out['finish']}); the turn ran with no new orders")
        if obj is not None:
            plan, fb2 = to_plan(obj); fb += fb2
        end_turns = 1
        note_new = note
        if obj is not None:
            try:
                end_turns = max(1, min(5, int(obj.get("end_turns", 1))))
            except Exception:                      # noqa: BLE001
                end_turns = 1
            if isinstance(obj.get("strategy_note"), str) and obj["strategy_note"].strip():
                note_new = obj["strategy_note"].strip()
        return {"plan": plan, "feedback": fb, "obj": obj, "parse_ok": parse_ok, "end_turns": end_turns, "note": note_new, "llm": out, "user_chars": len(user)}
