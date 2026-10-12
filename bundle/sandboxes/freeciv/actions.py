"""Action extraction -- the decisions each civ has made, readable at turn t.

This module exists because of a real modelling error in the first version of
the collector: the event stream (derived by diffing two adjacent states) was
treated as the action term.  It is not.

    events   = what already happened between t and t+1   -> part of the OUTPUT
    actions  = what each civ has chosen to do, at t       -> a legal INPUT

Feeding `events` in as a condition leaks the answer.  The real actions have to
come from the *intent* fields of the turn-t savegame, which the freeciv AI
writes down before the turn resolves:

  1. tax policy      `[playerN] rates.tax / rates.science / rates.luxury`
                     A three-way split summing to 100.  A pure policy lever.
  2. research goal   `[research] r` -> goal_name per player
  3. city production `[playerN] c` -> currently_building_name / _kind, and
                     `changed_from_name` (what it was building before a switch,
                     which is itself a visible decision)
  4. unit orders     `[playerN] u` -> go / goto_x / goto_y / orders_length /
                     activity.  A unit with `go=TRUE` has a destination it has
                     not reached yet: an order issued but not yet executed.

All four are present in the turn-t file, contain no future information, and are
exactly the levers a strategy-level agent would pull -- which is the point.  The
action space and the intervention-operator space are then the same space, so
"predict the consequence of an action" and "predict the consequence of do(X)"
become one learning problem rather than two.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .savegame import Savegame


@dataclass
class UnitOrder:
    unit_id: int
    type: str
    at: tuple[int, int]
    activity: str                 # decoded via activities_vector
    moving: bool                  # `go` flag
    destination: tuple[int, int] | None
    queued_orders: int            # orders_length


@dataclass
class PlayerActions:
    """Everything one civ has decided, as of turn t."""
    player: str
    tax: int
    science: int
    luxury: int
    research_goal: str
    production: dict[str, str]          # city name -> what it is building
    production_kind: dict[str, str]     # city name -> Building | UnitType
    switched: list[str]                 # cities that changed production
    unit_orders: list[UnitOrder] = field(default_factory=list)

    def summary(self) -> str:
        """One line, for the report's action section."""
        moving = sum(1 for u in self.unit_orders if u.moving)
        return (f"tax {self.tax}/sci {self.science}/lux {self.luxury}, "
                f"researching {self.research_goal or 'nothing'}, "
                f"{len(self.production)} cities building, "
                f"{moving}/{len(self.unit_orders)} units under orders")


def extract_actions(sg: Savegame) -> dict[str, PlayerActions]:
    """Per-player action set from a single turn-t savegame."""
    acts: dict[str, PlayerActions] = {}
    activities = [str(x) for x in sg.get("savefile", "activities_vector", [])]
    research = sg.table("research", "r")
    r_rows = research.as_dicts() if research else []

    for sec in sg.players():
        pid = int(sec[6:])
        name = str(sg.get(sec, "name", sec))

        goal = ""
        if pid < len(r_rows):
            goal = str(r_rows[pid].get("goal_name", "") or "")

        production, kinds, switched = {}, {}, []
        c_tbl = sg.table(sec, "c")
        if c_tbl is not None:
            for row in c_tbl.as_dicts():
                cname = str(row.get("name", "?"))
                cur = str(row.get("currently_building_name", ""))
                production[cname] = cur
                kinds[cname] = str(row.get("currently_building_kind", ""))
                # `changed_from_name` differing from the current build means the
                # civ switched what this city makes -- a visible decision.
                prev = str(row.get("changed_from_name", ""))
                if prev and prev != cur:
                    switched.append(cname)

        orders = []
        u_tbl = sg.table(sec, "u")
        if u_tbl is not None:
            for row in u_tbl.as_dicts():
                act_i = int(row.get("activity", 0))
                go = bool(row.get("go", False))
                dest = (int(row.get("goto_x", 0)), int(row.get("goto_y", 0))) if go else None
                orders.append(UnitOrder(
                    unit_id=int(row.get("id", 0)),
                    type=str(row.get("type_by_name", "")),
                    at=(int(row.get("x", 0)), int(row.get("y", 0))),
                    activity=activities[act_i] if act_i < len(activities) else str(act_i),
                    moving=go,
                    destination=dest,
                    queued_orders=int(row.get("orders_length", 0)),
                ))

        acts[name] = PlayerActions(
            player=name,
            tax=int(sg.get(sec, "rates.tax", 0)),
            science=int(sg.get(sec, "rates.science", 0)),
            luxury=int(sg.get(sec, "rates.luxury", 0)),
            research_goal=goal,
            production=production,
            production_kind=kinds,
            switched=switched,
            unit_orders=orders,
        )
    return acts


def to_json(acts: dict[str, PlayerActions]) -> dict[str, Any]:
    import dataclasses
    return {k: dataclasses.asdict(v) for k, v in acts.items()}
