"""WM v6: a committed route is a BINDING CONTRACT, not a paragraph in a memo (doc 17 §4.7, 2026-09-19).

The forking experiment (§6.9) measured the problem this fixes: the agent's three routes differ in 95% of their
text, but executing each of them for 20-40 turns produces a spread no larger than re-running the SAME route --
because a route was only a sentence in `strategy_note`, and the following twenty per-turn decisions were taken by
the same model with the same priors.  "Abandon the city" got played as "hold it a bit longer"; "all-in expansion"
got played as "expand and also build two Phalanxes".  With nothing separating the routes in outcome space, no
forecaster can add value: there is nothing to select.

So in v6 a route carries a machine-readable contract, and committing to it locks, for the next `turns` turns:

  rates            the tax/science/luxury split (the driver overrides any other rates the agent orders);
  research_goal    the long-term research target (a different goal is rejected);
  government       optional: the government the route commits to (any other revolution is rejected);
  build            a per-category quota of things to COMPLETE (settlers / workers / defenders / army /
                   buildings / wonder / other).  A category absent from the quota may not be started at all;
                   a category whose quota is already met may not be started again.  Completions are counted
                   from the engine's own `unit_built` / `building_built` events, not from what was ordered;
  max_treasury     the gold level above which the route says the ruler is hoarding (reported, not forced --
                   the driver cannot buy on the agent's behalf without inventing a decision).

The agent may break a contract by putting `"break_contract": "<reason>"` in its orders, and that is accepted ONLY
on a turn where the panel carries a SIEGE ARITHMETIC block (a city of the ego can actually fall).  Every break is
logged, so "how often did the contract bind" is a measurable property of the run rather than an assumption.
"""
from __future__ import annotations

from typing import Any

CATEGORIES = ("settlers", "workers", "defenders", "army", "buildings", "wonder", "other")


def categorise(target: str, rules) -> str:
    """Which contract category a production target belongs to.  Unknown names fall back to "other" (the engine is
    the authority on whether a name is buildable at all; this only decides which quota it consumes)."""
    t = (target or "").strip()
    u = rules.units.get(t)
    if u is not None:
        f = set(u.flags or [])
        if "Cities" in f or (u.pop_cost or 0) > 0 and u.attack == 0 and u.defense <= 1:
            return "settlers"
        if "Settlers" in f or "Workers" in (u.roles or []) or t in ("Workers", "Engineers"):
            return "workers"
        if u.military or u.attack > 0:
            return "defenders" if u.defense >= u.attack else "army"
        return "other"
    b = rules.buildings.get(t)
    if b is not None:
        return "wonder" if "Wonder" in str(b.genus) else "buildings"
    return "other"


def parse_contract(obj: Any, rules) -> dict[str, Any] | None:
    """The `contract` object of a route, validated.  Returns None when the route did not write a usable one."""
    if not isinstance(obj, dict):
        return None
    c: dict[str, Any] = {}
    r = obj.get("rates")
    if isinstance(r, (list, tuple)) and len(r) == 3:
        try:
            t, s, l = (int(x) for x in r)
            if t + s + l == 100 and min(t, s, l) >= 0:
                c["rates"] = [t, s, l]
        except Exception:                                  # noqa: BLE001
            pass
    if isinstance(obj.get("research_goal"), str) and obj["research_goal"].strip():
        c["research_goal"] = obj["research_goal"].strip()
    if isinstance(obj.get("government"), str) and obj["government"].strip():
        c["government"] = obj["government"].strip()
    q: dict[str, int] = {}
    b = obj.get("build")
    if not isinstance(b, dict):                        # the model sometimes writes a sentence here
        b = {}
    for k, v in b.items():
        k2 = str(k).strip().lower()
        if k2 in CATEGORIES:
            try:
                q[k2] = max(0, int(v))
            except Exception:                              # noqa: BLE001
                continue
    if q:
        c["build"] = q
    try:
        c["found_cities"] = max(0, int(obj.get("found_cities", 0)))
    except Exception:                                      # noqa: BLE001
        c["found_cities"] = 0
    try:
        c["max_treasury"] = max(0, int(obj["max_treasury"]))
    except Exception:                                      # noqa: BLE001
        pass
    return c or None


class Contract:
    """The contract in force, its counters, and the enforcement of a TurnPlan against it."""

    def __init__(self, rules, turns: int = 10, mode: str = "hard"):
        self.rules, self.turns, self.mode = rules, turns, mode
        self.spec: dict[str, Any] | None = None
        self.letter = ""; self.name = ""; self.start = -1; self.until = -1
        self.built: dict[str, int] = {}
        self.founded = 0
        self.breaks: list[dict] = []
        self.blocked: list[str] = []

    # ---- lifecycle
    def commit(self, turn: int, letter: str, name: str, spec: dict[str, Any] | None) -> None:
        self.spec, self.letter, self.name = spec, letter, name
        self.start, self.until = turn, turn + self.turns
        self.built = {k: 0 for k in CATEGORIES}
        self.founded = 0

    @property
    def active(self) -> bool:
        return self.spec is not None

    def in_force(self, turn: int) -> bool:
        return self.active and self.start <= turn < self.until

    # ---- counting what the engine actually completed
    def observe(self, events: list[dict], ego_id: int) -> None:
        if not self.active:
            return
        for e in events or []:
            if int(e.get("player_id", -1)) != ego_id:
                continue
            ty = e.get("type")
            if ty == "unit_built":
                self.built[categorise(str(e.get("unit") or ""), self.rules)] = self.built.get(categorise(str(e.get("unit") or ""), self.rules), 0) + 1
            elif ty == "building_built":
                self.built[categorise(str(e.get("building") or ""), self.rules)] = self.built.get(categorise(str(e.get("building") or ""), self.rules), 0) + 1
            elif ty == "city_founded":
                self.founded += 1

    # ---- enforcement
    def fulfilled(self) -> bool:
        """Every category the route promised has been completed: the build lock has done its job and is released
        (otherwise the last turns of a window are a straitjacket -- the agent can change nothing and the engine just
        repeats whatever each city was already producing).  Found in the v6 smoke test, 2026-09-19."""
        q = (self.spec or {}).get("build")
        return bool(q) and all(self.built.get(k, 0) >= v for k, v in q.items() if v > 0)

    def goal_done(self, known: set[str] | None) -> bool:
        """The contract's research goal is already known: keeping it locked makes every bulb from here on a waste."""
        g = (self.spec or {}).get("research_goal")
        return bool(g) and known is not None and g in known

    def enforce(self, plan, turn: int, obj: dict | None, siege: bool, known: set[str] | None = None) -> list[str]:
        """Mutates `plan` in place; returns the feedback lines the agent will read next turn."""
        fb: list[str] = []
        if not self.in_force(turn):
            return fb
        if self.mode != "hard":
            # ★ v7 (2026-09-19, user decision): the declaration is NOT enforced.  Hard enforcement was measured to
            #   trap the agent: 60% of committed routes wrote a settlers quota of 0, the agent then asked for
            #   settlers anyway and was refused up to 33 times in one game, and the games where that happened ended
            #   4.2 cities behind their own control.  A plan should steer the next ten turns, not handcuff them.
            #   What stays is the MEASUREMENT: what the route promised, what the engine actually completed, and the
            #   gap between them -- that is how we find out whether choosing a route changes anything at all.
            for c in plan.cities:
                if not c.build:
                    continue
                cat = categorise(c.build, self.rules)
                q = (self.spec or {}).get("build") or {}
                if int(q.get(cat, 0)) <= 0:
                    self.blocked.append(f"T{turn} {c.name} {c.build} ({cat}, off-plan)")
            return fb
        brk = str((obj or {}).get("break_contract") or "").strip()
        if brk:
            if siege:
                self.breaks.append({"turn": turn, "why": brk[:300], "granted": True})
                self.spec = None
                return [f"CONTRACT BROKEN at turn {turn} (a city of yours can fall, so the break is granted): {brk[:200]}. "
                        "You are free until the next planning point."]
            self.breaks.append({"turn": turn, "why": brk[:300], "granted": False})
            fb.append("CONTRACT BREAK REFUSED: you may only break the contract on a turn when the panel shows a SIEGE "
                      "ARITHMETIC block. Your contract still binds.")
        s = self.spec or {}
        if s.get("rates") and plan.rates is not None and list(plan.rates) != list(s["rates"]):
            fb.append(f"CONTRACT: rates {list(plan.rates)} replaced by the {s['rates']} you committed to in route {self.letter} "
                      f"(locked until turn {self.until}).")
            plan.rates = tuple(s["rates"])
        if s.get("rates") and plan.rates is None:
            plan.rates = tuple(s["rates"])
        goal_done = self.goal_done(known)
        if goal_done and plan.research_goal:
            fb.append(f"CONTRACT: the research goal '{s['research_goal']}' you committed to is already known, so the "
                      "research lock is released; your new goal stands.")
        if not goal_done and s.get("research_goal") and plan.research_goal and plan.research_goal != s["research_goal"]:
            fb.append(f"CONTRACT: research goal '{plan.research_goal}' rejected; route {self.letter} committed to "
                      f"'{s['research_goal']}' until turn {self.until}.")
            plan.research_goal = s["research_goal"]
        if not goal_done and s.get("research_goal") and not plan.research_goal:
            plan.research_goal = s["research_goal"]
        if plan.research_now and s.get("research_goal") and not goal_done:
            fb.append(f"CONTRACT: 'research_now' {plan.research_now} dropped; changing the current research throws away "
                      "stored bulbs and your contract fixes the goal.")
            plan.research_now = None
        if plan.government and s.get("government") and plan.government != s["government"]:
            fb.append(f"CONTRACT: revolution to '{plan.government}' rejected; route {self.letter} committed to "
                      f"'{s['government']}'.")
            plan.government = None
        elif plan.government and "government" not in s:
            fb.append(f"CONTRACT: revolution to '{plan.government}' rejected; route {self.letter} did not commit to a "
                      "government change, and two turns of Anarchy is not something to start off-plan.")
            plan.government = None
        q = s.get("build")
        if q is not None and self.fulfilled():
            fb.append("CONTRACT: every category of your build quota is complete, so the production lock is released for "
                      f"the rest of this window (until turn {self.until}); the budget and research goal still bind.")
            q = None
        if q is not None:
            keep = []
            for c in plan.cities:
                if not c.build:
                    keep.append(c); continue
                cat = categorise(c.build, self.rules)
                quota = int(q.get(cat, 0))
                done = int(self.built.get(cat, 0))
                if quota <= 0:
                    fb.append(f"CONTRACT: {c.name} may not start {c.build} -- route {self.letter} promised nothing in the "
                              f"category '{cat}' (quota: " + ", ".join(f"{k} {v}" for k, v in q.items()) + f"), locked until turn {self.until}.")
                    self.blocked.append(f"T{turn} {c.name} {c.build} ({cat}, quota 0)")
                    continue
                if done >= quota:
                    fb.append(f"CONTRACT: {c.name} may not start {c.build} -- the '{cat}' quota of route {self.letter} is "
                              f"already met ({done}/{quota}).")
                    self.blocked.append(f"T{turn} {c.name} {c.build} ({cat}, {done}/{quota})")
                    continue
                keep.append(c)
            plan.cities = keep
        return fb

    # ---- what the agent sees every turn
    def panel(self, turn: int, gold: int) -> str:
        if not self.in_force(turn):
            return ""
        s = self.spec or {}
        hard = self.mode == "hard"
        L = ([f"### CONTRACT IN FORCE (route {self.letter} '{self.name[:40]}', committed at turn {self.start}, binding until turn {self.until})",
              "You committed to this route and the driver enforces it: orders that break it are rejected, not executed."]
             if hard else
             [f"### THE COURSE YOU COMMITTED TO (route {self.letter} '{self.name[:40]}', chosen at turn {self.start}, in force until turn {self.until})",
              "You chose this route after reading the forecasts. Nothing here is enforced -- you may depart from it whenever "
              "the situation warrants -- but it is the plan you committed to, and a plan you abandon by default is not a plan."])
        lock = "locked at" if hard else "you committed to"
        if s.get("rates"):
            L.append(f"- budget {lock} tax/science/luxury {'/'.join(map(str, s['rates']))}")
        if s.get("research_goal"):
            L.append(f"- research goal {lock} {s['research_goal']}")
        if s.get("government"):
            L.append(("- the only government change allowed is to " if hard else "- the government change you planned: ") + s["government"])
        q = s.get("build")
        if q is not None:
            done = ", ".join(f"{k} {self.built.get(k, 0)}/{v}" for k, v in q.items())
            L.append(f"- what you promised to build / what the engine has completed since turn {self.start}: {done}")
            if not hard:
                off = len([x for x in self.blocked if x.startswith(f"T{turn}") or True])
                if off:
                    L.append(f"- you have started {off} things that were not in this plan; that is allowed, but say why in your note if it is a change of course")
            elif self.fulfilled():
                L.append("- ★ the whole quota is complete: the production lock is RELEASED for the rest of this window, "
                         "build what you like (the budget and research goal still bind)")
            else:
                closed = [k for k in CATEGORIES if int(q.get(k, 0)) <= 0] + [k for k, v in q.items() if v > 0 and self.built.get(k, 0) >= v]
                L.append(f"- you may NOT start anything in these categories: {', '.join(sorted(set(closed)))}")
        if s.get("found_cities"):
            L.append(f"- cities promised: {self.founded}/{s['found_cities']} founded so far")
        if s.get("max_treasury") is not None and gold > int(s["max_treasury"]):
            L.append(f"- ★ TREASURY {gold} is above the {s['max_treasury']} your route committed to. Gold in the treasury "
                     "scores NOTHING at the end of the game: buy production this turn or lower taxes at the next planning point.")
        if hard:
            L.append("- to break the contract you must write \"break_contract\": \"<reason>\" in your orders, and that is only "
                     "granted on a turn when the panel shows a SIEGE ARITHMETIC block.")
        return "\n".join(L)

    def as_record(self, turn: int) -> dict:
        return {"letter": self.letter, "name": self.name, "spec": self.spec, "start": self.start, "until": self.until,
                "built": dict(self.built), "founded": self.founded, "in_force": self.in_force(turn),
                "breaks": list(self.breaks), "blocked": list(self.blocked[-20:])}
