"""WM v12: the object the dialogue refines is a PLAN, not a choice between labels (doc 17 §4.12, user 2026-09-20).

Every version up to v11 asked the agent for three routes and then asked it to pick one.  Measured, that protocol
degenerated: the three routes were 95% different in text and indistinguishable in outcome (§6.9), the forecaster's
conditional probabilities did not separate them (spread 0.04, correlation with the routes' own garrison -0.012),
72% of planning points used a single round, and 60% of the picks were the agent's own opening intention.  A
multiple-choice format cannot produce a better plan; it can only select among plans nobody improved.

v12 keeps the object and throws away the menu.  The agent writes one to three CANDIDATE PLANS at the operational
granularity it would actually need to play -- a production queue per city, settlers with escorts and target tiles
and deadlines, a garrison per city, a research chain, a budget with a treasury floor, named triggers, and
milestones it expects to hit.  The forecaster predicts each, and says what to CHANGE, pointing at lines.  The
agent then deletes, edits or merges, and asks again, until it declares the plan final.  What comes out is a plan,
not a letter.

Because the plan is concrete, the driver can do three things it could never do with prose:
  * CHECK it -- is that tile a legal city site, does that queue fit in the shields that city makes, do the rates
    sum to 100 under this government;
  * COST it -- the garrison the plan names goes straight into the siege arithmetic (`siege.plan_risk`);
  * RUN it -- the production queue is applied as each city's default order every turn, so a plan that is written
    is a plan that executes, while any order the agent gives that turn still overrides it.
"""
from __future__ import annotations

import math
import re
from typing import Any

from bench.agentplay.engine import CityCmd
from bench.agentplay.sites import is_legal_site

SECTIONS = ("intent", "production", "settlement", "garrison", "military", "research", "budget",
            "diplomacy", "triggers", "milestones")


def _int(x, d=0):
    try:
        return int(x)
    except Exception:                                        # noqa: BLE001
        m = re.search(r"-?\d+", str(x))
        return int(m.group(0)) if m else d


def _tile(x) -> tuple[int, int] | None:
    if isinstance(x, (list, tuple)) and len(x) >= 2:
        try:
            return (int(x[0]), int(x[1]))
        except Exception:                                    # noqa: BLE001
            return None
    m = re.search(r"\(?\s*(\d+)\s*,\s*(\d+)\s*\)?", str(x))
    return (int(m.group(1)), int(m.group(2))) if m else None


def _as_list(x) -> list:
    """A field the schema declares as a list often comes back as one string; iterating that string would split it
    into characters (measured in the v12 smoke test: "Barbarians..." became six triggers "B","a","r","b","a","r")."""
    if x is None:
        return []
    if isinstance(x, str):
        parts = [t.strip() for t in re.split(r"\n+|(?<=[.;])\s+", x) if t.strip()]
        return parts or [x.strip()]
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def match_item(name: str, rules) -> str | None:
    """Resolve what the model wrote to a real production target, or None.

    Measured on the v12 smoke tests: the model writes queue entries as prose ("Settlers (28/30 kept) -> then Worker
    -> new Settler founds at (17,31)"), and the engine needs one exact rule name.  Anything that cannot be resolved
    is DROPPED from the queue the driver executes and named in the checks, because a queue the driver runs must not
    contain a sentence."""
    if not rules:
        return name or None
    n = str(name or "").strip()
    n = re.split(r"[(\[]", n)[0].strip().strip(" .,-–—>")
    n = re.sub(r"^(then|next|and then|followed by|new|another|a|an|the)\s+", "", n, flags=re.I).strip()
    if not n:
        return None
    pool = {**rules.units, **rules.buildings}
    if n in pool:
        return n
    low = {k.lower(): k for k in pool}
    if n.lower() in low:
        return low[n.lower()]
    for cand in (n + "s", n[:-1] if n.endswith("s") else n):
        if cand.lower() in low:
            return low[cand.lower()]
    return None


def buildable(name: str, rules) -> bool:
    return match_item(name, rules) is not None


def parse_plan(obj: Any, rules=None) -> dict | None:
    """Normalise one plan object.  Everything is optional: a plan that only says what to build is still a plan,
    it just gets fewer checks."""
    if not isinstance(obj, dict):
        return None
    p: dict[str, Any] = {"name": str(obj.get("name", "plan"))[:60],
                         "intent": str(obj.get("intent") or obj.get("reasoning") or "")[:4000]}
    notes = {}
    if isinstance(obj.get("notes"), dict):
        for k, v in obj["notes"].items():
            notes[str(k).strip()] = str(v)[:400]
    p["notes"] = notes
    p["found_cities"] = max(0, _int(obj.get("found_cities"), 0))
    buys = []
    for b_ in _as_list(obj.get("buy")):
        if isinstance(b_, dict) and b_.get("city") and b_.get("item"):
            buys.append({"city": str(b_["city"]).strip(), "item": str(b_["item"]).strip(), "why": str(b_.get("why", ""))[:160]})
    p["buy"] = buys[:4]
    prod = {}; dropped: list[str] = []
    src = obj.get("production")
    if isinstance(src, dict):
        for city, q in src.items():
            if isinstance(q, str):
                q = [x.strip() for x in re.split(r"->|→|,|;", q) if x.strip()]
            if isinstance(q, list):
                items = []
                for it in q[:8]:
                    if isinstance(it, dict):
                        nm = str(it.get("item") or it.get("build") or "").strip()
                        by = _int(it.get("by_turn") or it.get("turn"), 0)
                    else:
                        nm = str(it).strip(); by = 0
                        m = re.search(r"\(?T?(\d{2,4})\)?\s*$", nm)
                        if m:
                            by = int(m.group(1)); nm = nm[:m.start()].strip()
                    nm = nm.strip(" .-–→")
                    if not nm:
                        continue
                    res = match_item(nm, rules) if rules is not None else nm
                    if res is None:
                        dropped.append(f"{city}: '{nm[:50]}'")
                        continue
                    items.append({"item": res, "by_turn": by, "ok": True})
                if items:
                    prod[str(city).strip()] = items
    p["production"] = prod
    p["dropped_items"] = dropped[:8]
    st = []
    for s in _as_list(obj.get("settlement")):
        if isinstance(s, dict):
            t = _tile(s.get("target") or s.get("tile") or s.get("to"))
            st.append({"unit": _int(s.get("unit"), 0), "escort": _int(s.get("escort"), 0),
                       "target": (list(t) if t else None), "by_turn": _int(s.get("by_turn") or s.get("turn"), 0),
                       "why": str(s.get("why", ""))[:200]})
        elif isinstance(s, str):
            t = _tile(s)
            st.append({"unit": 0, "escort": 0, "target": (list(t) if t else None), "by_turn": 0, "why": s[:200]})
    p["settlement"] = st
    g = {}
    if isinstance(obj.get("garrison"), dict):
        for city, n in obj["garrison"].items():
            g[str(city).strip()] = max(0, _int(n, 0))
    p["garrison"] = g
    p["military"] = str(obj.get("military", ""))[:600]
    r = obj.get("research")
    if isinstance(r, dict):
        rr = {"now": str(r.get("now", ""))[:60], "goal": str(r.get("goal", ""))[:60]}
    else:
        rr = {"now": "", "goal": str(r or "")[:60]}
    if rules is not None:                                    # only a real tech name may reach the engine
        for k in ("now", "goal"):
            v = rr[k].strip()
            if v and v not in rules.techs:
                hit = next((t for t in rules.techs if t.lower() in v.lower()), "")
                rr[k] = hit
    p["research"] = rr
    b = obj.get("budget") if isinstance(obj.get("budget"), dict) else {}
    rates = b.get("rates")
    if isinstance(rates, (list, tuple)) and len(rates) == 3:
        rates = [_int(x) for x in rates]
        rates = rates if sum(rates) == 100 and min(rates) >= 0 else None
    else:
        rates = None
    p["budget"] = {"rates": rates, "treasury_floor": _int(b.get("treasury_floor"), -1)}
    p["diplomacy"] = str(obj.get("diplomacy", ""))[:400]
    p["triggers"] = [str(x)[:220] for x in _as_list(obj.get("triggers"))][:6]
    ms = []
    for m in _as_list(obj.get("milestones")):
        if isinstance(m, dict):
            ms.append({"by_turn": _int(m.get("by_turn") or m.get("turn"), 0), "expect": str(m.get("expect", ""))[:300]})
        elif isinstance(m, str):
            mm = re.search(r"T?(\d{2,4})", m)
            ms.append({"by_turn": int(mm.group(1)) if mm else 0, "expect": m[:300]})
    p["milestones"] = ms[:5]
    return p


def carry_forward(new: dict, old: dict) -> list[str]:
    """★ Measured in the v12 smoke test: asked to revise, the model returned a plan with only `name` and `intent`,
    silently dropping the production queue -- the part the driver executes.  A revision is an EDIT, so a section the
    revision does not mention keeps its previous content, and the agent is told that is what happened."""
    notes = []
    for k, empty in (("production", {}), ("garrison", {}), ("settlement", []), ("triggers", []),
                     ("milestones", []), ("buy", []), ("notes", {})):
        if not new.get(k) and old.get(k):
            new[k] = old[k]
            notes.append(f"your revision did not restate `{k}`, so the driver kept the previous one")
    for k in ("military", "diplomacy"):
        if not new.get(k) and old.get(k):
            new[k] = old[k]
    if not new.get("found_cities") and old.get("found_cities"):
        new["found_cities"] = old["found_cities"]
    r_new, r_old = new.get("research") or {}, old.get("research") or {}
    for k in ("now", "goal"):
        if not r_new.get(k) and r_old.get(k):
            r_new[k] = r_old[k]
    new["research"] = r_new
    b_new, b_old = new.get("budget") or {}, old.get("budget") or {}
    if not b_new.get("rates") and b_old.get("rates"):
        b_new["rates"] = b_old["rates"]
    if b_new.get("treasury_floor", -1) < 0 and b_old.get("treasury_floor", -1) >= 0:
        b_new["treasury_floor"] = b_old["treasury_floor"]
    new["budget"] = b_new
    return notes


def prune_to_cities(p: dict, ego) -> list[str]:
    """Drop production/garrison entries for cities that do not exist, and report them."""
    bad = []
    for key in ("production", "garrison"):
        d = p.get(key) or {}
        for c in list(d):
            if c not in ego.cities:
                bad.append(f"{key}: '{c}' is not one of your cities")
                d.pop(c)
    return bad


def render_plan(p: dict, header: str = "") -> str:
    L = ([header] if header else []) + [f"Name: {p.get('name','plan')}"]
    if p.get("intent"):
        L.append(f"Intent: {p['intent']}")
    if p.get("notes"):
        for c, n_ in p["notes"].items():
            L.append(f"Note on {c}: {n_}")
    if p.get("production"):
        L.append("Production queue (in order, per city):")
        for c, q in p["production"].items():
            L.append("  " + c + ": " + " -> ".join(f"{i['item']}" + (f" (by T{i['by_turn']})" if i["by_turn"] else "") for i in q))
    if p.get("settlement"):
        L.append("Settlement:")
        for s in p["settlement"]:
            L.append(f"  Settlers {s['unit'] or '?'}" + (f" escorted by {s['escort']}" if s["escort"] else "")
                     + (f" -> found at ({s['target'][0]},{s['target'][1]})" if s.get("target")
                        else " -> found at the best legal site the driver finds")
                     + (f" by T{s['by_turn']}" if s["by_turn"] else "") + (f" -- {s['why']}" if s["why"] else ""))
    if p.get("found_cities"):
        L.append(f"New cities intended in the next 20 turns: {p['found_cities']}")
    for b_ in p.get("buy") or []:
        L.append(f"Rush-buy: {b_['item']} in {b_['city']} -- {b_['why']}")
    if p.get("garrison"):
        L.append("Garrison: " + ", ".join(f"{c} {n}" for c, n in p["garrison"].items()))
    if p.get("military"):
        L.append(f"Military: {p['military']}")
    r = p.get("research") or {}
    if r.get("now") or r.get("goal"):
        L.append(f"Research: now {r.get('now') or '(unchanged)'}, goal {r.get('goal') or '(unchanged)'}")
    b = p.get("budget") or {}
    if b.get("rates") or b.get("treasury_floor", -1) >= 0:
        L.append("Budget: " + (f"tax/science/luxury {'/'.join(map(str, b['rates']))}" if b.get("rates") else "rates unchanged")
                 + (f"; treasury floor {b['treasury_floor']}" if b.get("treasury_floor", -1) >= 0 else ""))
    if p.get("diplomacy"):
        L.append(f"Diplomacy: {p['diplomacy']}")
    for t in p.get("triggers") or []:
        L.append(f"Trigger: {t}")
    for m in p.get("milestones") or []:
        L.append(f"Milestone by T{m['by_turn']}: {m['expect']}")
    return "\n".join(L)


# ------------------------------------------------------------------ the driver's factual check of a plan
def _cost_of(item: str, rules) -> int | None:
    u = rules.units.get(item)
    if u is not None:
        return int(u.build_cost)
    b = rules.buildings.get(item)
    return int(b.build_cost) if b is not None else None


def validate_plan(p: dict, label: str, sg, rules, board, geo, ego, mi, turn: int) -> list[str]:
    """Facts about this plan that the driver can establish from the ruleset and the map.  Never an opinion."""
    out: list[str] = []
    for d_ in (p.get("dropped_items") or []):
        out.append(f"{label}: the driver could not read {d_} as something buildable, so it is NOT in the queue it "
                   "runs.  Write the bare item name and put notes in \"intent\".")
    known = {t: o for t, (o, c) in ego.all_cities_by_tile().items() if board.is_known(*t)}
    units_by_tile = ego.all_units_by_tile()
    board.blocked = {t for t, lst in units_by_tile.items() if any(o != ego.name for o, _ in lst)} | {t for t, o in known.items() if o != ego.name}
    # --- settlement targets
    for s in p.get("settlement") or []:
        if not s.get("target"):
            continue                                       # the driver picks the site; nothing to check
        t = (int(s["target"][0]), int(s["target"][1]))
        if not board.is_known(*t):
            out.append(f"{label}: the site ({t[0]},{t[1]}) is on a tile you have not explored; it cannot be planned yet.")
            continue
        near = [(c, geo.distance(t, c)) for c in known]
        near.sort(key=lambda x: x[1])
        if not is_legal_site(board, geo, t, known, ego.name, rules.citymindist):
            why = "too close to a known city" if near and near[0][1] < rules.citymindist else "not usable land you may settle"
            d = f" (nearest known city is {near[0][1]} tiles away, minimum is {rules.citymindist})" if near else ""
            out.append(f"{label}: founding at ({t[0]},{t[1]}) is ILLEGAL -- {why}{d}. The engine will refuse it.")
        elif s.get("unit"):
            u = ego.units.get(int(s["unit"]))
            if u is None:
                out.append(f"{label}: unit {s['unit']} does not exist; the settler for ({t[0]},{t[1]}) has to come from somewhere.")
            else:
                d = geo.distance((int(u["x"]), int(u["y"])), t)
                mv = max(1, int(rules.units[str(u["type_by_name"])].move_rate))
                eta = turn + math.ceil(d / mv)
                if s.get("by_turn") and eta > s["by_turn"]:
                    out.append(f"{label}: Settlers {s['unit']} is {d} tiles from ({t[0]},{t[1]}), so it arrives about T{eta}, not T{s['by_turn']}.")
    # --- production queue feasibility
    for cname, q in (p.get("production") or {}).items():
        c = ego.cities.get(cname)
        if c is None:
            out.append(f"{label}: you have no city called '{cname}'.")
            continue
        surplus = max(1, int(c.get("last_turns_shield_surplus", 1) or 1))
        stock = int(c.get("shield_stock", 0) or 0)
        # ★ v13: one line per city instead of one per item.  Measured on the v12 run: this check fired 4.34 times
        #   per planning point (87% of everything the driver said), so the dialogue was spent correcting schedules
        #   instead of discussing strategy.  The schedule is still reported -- once, as a whole.
        t_cur = turn; eta = []
        for i, it in enumerate(q):
            cost = _cost_of(it["item"], rules)
            if cost is None:
                out.append(f"{label}: {cname}'s queue contains '{it['item'][:60]}', which is not the name of anything this "
                           "ruleset can build. The driver will skip it.")
                continue
            need = max(0, cost - (stock if i == 0 else 0))
            t_cur += math.ceil(need / surplus)
            eta.append((it["item"], t_cur, it.get("by_turn") or 0))
        if eta:
            late = [e for e in eta if e[2] and e[1] > e[2] + 1]
            line = (f"{label}: {cname} makes {surplus} shields/turn, so its queue lands about "
                    + ", ".join(f"{i_} T{t_}" for i_, t_, _ in eta[:4]) + ".")
            if late:
                line += f"  You wrote earlier turns for {len(late)} of them ({late[0][0]} T{late[0][2]})."
            out.append(line)
    # --- budget
    b = p.get("budget") or {}
    if b.get("rates"):
        cap = 60
        try:
            from bench.agentplay.rules import gov_max_rate
            cap = gov_max_rate(rules, ego.gov)
        except Exception:                                    # noqa: BLE001
            pass
        if max(b["rates"]) > cap:
            out.append(f"{label}: under {ego.gov} no rate may exceed {cap}%, so {b['rates']} will be clipped by the engine.")
    if b.get("treasury_floor", -1) >= 0 and ego.gold < b["treasury_floor"]:
        out.append(f"{label}: your treasury is {ego.gold}, already below the floor of {b['treasury_floor']} this plan sets.")
    return out


# ------------------------------------------------------------------ running the plan
class PlanRunner:
    """Holds the plan in force, applies its production queue as each city's default order, and checks milestones."""

    def __init__(self, rules):
        self.rules = rules
        self.plan: dict | None = None
        self.bought: set = set()
        self.start = -1
        self.done: dict[str, int] = {}
        self.checked: set[int] = set()

    def adopt(self, plan: dict | None, turn: int) -> None:
        self.plan, self.start = plan, turn
        self.done = {c: 0 for c in (plan or {}).get("production", {})}
        self.checked = set(); self.bought = set()

    def observe(self, events: list[dict], ego_id: int) -> None:
        if not self.plan:
            return
        for e in events or []:
            if int(e.get("player_id", -1)) != ego_id:
                continue
            nm = str(e.get("unit") or e.get("building") or "")
            city = str(e.get("city") or "")
            q = (self.plan.get("production") or {}).get(city)
            if not q or e.get("type") not in ("unit_built", "building_built"):
                continue
            i = self.done.get(city, 0)
            if i < len(q) and q[i]["item"].lower() == nm.lower():
                self.done[city] = i + 1

    def default_cities(self, ego) -> list[CityCmd]:
        """★ user decision 2026-09-20: the plan RUNS.  Every turn each city is set to the next unfinished item of
        its queue; any city the agent names in its own orders overrides this."""
        out = []
        if not self.plan:
            return out
        for city, q in (self.plan.get("production") or {}).items():
            c = ego.cities.get(city)
            if c is None:
                continue
            i = self.done.get(city, 0)
            if i >= len(q):
                continue
            while i < len(q) and not q[i].get("ok", True):
                i += 1
            if i >= len(q):
                continue
            want = q[i]["item"]
            if str(c.get("currently_building_name", "")).lower() == want.lower():
                continue
            out.append(CityCmd(city, build=want))
        return out

    def buy_orders(self, ego) -> list[CityCmd]:
        """The plan's rush-buys, executed as soon as the treasury allows above the plan's own floor."""
        out = []
        if not self.plan:
            return out
        floor = int((self.plan.get("budget") or {}).get("treasury_floor", 0) or 0)
        if ego.gold <= floor:
            return out
        for b in (self.plan.get("buy") or []):
            c = ego.cities.get(b["city"])
            if c is None or b["city"] in self.bought:
                continue
            item = match_item(b["item"], self.rules)
            if item is None:
                continue
            out.append(CityCmd(b["city"], build=item, buy=True))
            self.bought.add(b["city"])
        return out

    def progress_block(self, turn: int, ego) -> str:
        if not self.plan:
            return ""
        L = [f"### THE PLAN YOU ARE EXECUTING (adopted at turn {self.start})",
             "The driver sets each city to the next item of this queue every turn.  Name a city in your own orders "
             "and that overrides the plan for that city, this turn."]
        for city, q in (self.plan.get("production") or {}).items():
            i = self.done.get(city, 0)
            done = ", ".join(x["item"] for x in q[:i]) or "-"
            nxt = q[i]["item"] if i < len(q) else "(queue finished)"
            L.append(f"- {city}: done [{done}] -> now building {nxt}" + (f", then {', '.join(x['item'] for x in q[i+1:i+3])}" if i + 1 < len(q) else ""))
        g = self.plan.get("garrison") or {}
        if g:
            have = {}
            for city, c in ego.cities.items():
                pos = (int(c["x"]), int(c["y"]))
                have[city] = sum(1 for u in ego.units.values()
                                 if (int(u["x"]), int(u["y"])) == pos and self.rules.units.get(str(u["type_by_name"]))
                                 and self.rules.units[str(u["type_by_name"])].defense > 0)
            L.append("- garrison planned/actual: " + ", ".join(f"{c} {n}/{have.get(c, 0)}" for c, n in g.items()))
        b = self.plan.get("budget") or {}
        if b.get("treasury_floor", -1) >= 0:
            L.append(f"- treasury floor {b['treasury_floor']}, you have {ego.gold}"
                     + ("  <-- BELOW YOUR OWN FLOOR" if ego.gold < b["treasury_floor"] else ""))
        for t in self.plan.get("triggers") or []:
            L.append(f"- trigger you set: {t}")
        return "\n".join(L)

    def milestone_check(self, turn: int, state: dict) -> str:
        """When a milestone comes due, the driver puts the agent's own expectation next to what actually happened."""
        if not self.plan:
            return ""
        out = []
        for i, m in enumerate(self.plan.get("milestones") or []):
            if m["by_turn"] and turn >= m["by_turn"] and i not in self.checked:
                self.checked.add(i)
                out.append(f"### MILESTONE DUE (you set it at turn {self.start})\n"
                           f"- by T{m['by_turn']} you expected: {m['expect']}\n"
                           f"- actually at T{turn}: score {state['score']}, {state['n_cities']} cities, "
                           f"{state['citizens']} citizens, {state['n_techs']} techs, {state['gold']} gold, "
                           f"{state['n_units']} units")
        return "\n".join(out)
