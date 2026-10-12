"""Derive the state-diff half of the event stream.

The savegame does not store a turn-by-turn log of *what happened* -- it stores
the state -- so events have to be inferred by comparing consecutive states.
**This module is only half the picture.**  The freeciv server also emits typed
Lua signals carrying causal attribution the state can never show (who razed the
city, which unit attacked, why a unit died); those are harvested by
`engine_events.py`.  The two are complementary, not alternatives:

    engine signals  cities / units / techs / buildings / actions / disasters
    state diff      government changes and diplomacy -- **measured**: freeciv
                    3.1.6 has no `government_changed` and no `war_declared`
                    signal, so this module is their only source

Three defects found by comparing this module against the engine log, all fixed
here, all with measured numbers (see `docs/freeciv/`):

1. **A conquest was reported as `city_destroyed` + `city_founded`.**  The city
   diff ran *per player*, so a city changing hands vanished from the loser's id
   set and appeared in the winner's.  4 of 4 conquests in the verified world
   were mislabelled: 4 of 6 `city_destroyed` and 4 of 41 `city_founded` events
   were fabrications, and the two real destructions were buried among them.
   Fixed by tracking city ids **globally** -- an owner change is a transfer.

2. **`other` was a savegame slot id, not a civ.**  `[playerN].diplstate` is
   indexed by slot, so events read `"other": "player1"` and `report.py` rendered
   that string into the corpus.  Slot -> name is safe: the self row is always
   the sentinel `Never met` (34,729/34,729), the matrix is symmetric
   (129,599/129,599 pairs), and the roster never shrinks, so `players[i]`
   corresponds to `playerI`.

3. **The two diplomacy labels were mostly wrong, and dropped a quarter of the
   transitions.**  `war_declared` covered `Never met -> War` (13.3%, that is
   *first contact*) and `Cease-fire -> War` (37.8%, a cease-fire *expiring*);
   only 5.3% were an actual declaration.  `peace_treaty` was 43.6%
   `War -> Cease-fire`, which is a truce, not peace.  And `Alliance` /
   `Armistice` were absent from the vocabulary entirely, so alliance formation
   and alliance betrayal produced no event at all -- 24.6% of all transitions
   were silently invisible.  Fixed by naming the transition faithfully and
   carrying `from`/`to` on every event, with a `diplstate_change` catch-all so
   an unmapped transition still shows up.

Diplomacy is mutual, so each transition is emitted once per side; the per-view
report filters on the viewer.

**4. `turn` on a diff event means "first visible in the state at this turn", and
that is all a diff can honestly claim.**  Measured against the engine's own turn
column inside `[event_cache]`, the old convention (`turn = cur.turn`) was late by
one for `city_founded` (30/30), `tech_discovered` (57/58) and `government_change`
(14/16), yet exact for `city_destroyed` and `first_contact` (6/6) -- the offset
depends on where in freeciv's turn sequence the change lands.  Rather than fit a
per-type constant to one world, events carry:

    turn          the state boundary the change was first visible at.  Exact by
                  construction, because that is what a diff observes.
    engine_turn   when the engine's own record gives it.  `city_founded` gets it
                  from the savegame's `turn_founded`, verified 30/30; every event
                  from `engine_events.py` has it because the signal fires on the
                  turn itself.

`report.py` prefers `engine_turn` and otherwise renders "by turn N", which is
true, instead of "turn N", which was not.
"""
from __future__ import annotations

from typing import Any

from .state import WorldState


# Diplomacy: (from, to) -> event type.  Every transition measured in the v1
# corpus is mapped; `diplstate_change` catches anything a future ruleset adds so
# a new transition shows up as an event rather than as silence.  freeciv's
# `Armistice` is the countdown state after a peace treaty is signed (units must
# leave each other's territory first), which is why `Cease-fire -> Armistice` is
# the signing and `Armistice -> Peace` is it taking effect.
_HOSTILE_FROM = ("Peace", "Armistice", "Alliance")
DIPL_TRANSITIONS: dict[tuple[str, str], str] = {
    **{(f, "War"): "war_declared" for f in _HOSTILE_FROM},
    ("War", "Cease-fire"): "ceasefire_agreed",
    ("Cease-fire", "War"): "ceasefire_expired",
    ("Cease-fire", "Armistice"): "peace_treaty_signed",
    ("Armistice", "Peace"): "peace_established",
    ("Peace", "Alliance"): "alliance_formed",
    ("Armistice", "Alliance"): "alliance_formed",
    ("Alliance", "Armistice"): "alliance_ended",
}

# The sentinel every player's own diplstate row carries (verified 34,729/34,729
# in the v1 corpus).  A row equal to this on the *self* index is not a relation.
SELF_DIPLSTATE = "Never met"


def _dipl_type(prev_st: str, st: str) -> str:
    """Name a diplstate transition without interpreting it into the wrong thing."""
    if prev_st == "Never met":
        # Meeting a civ.  In rulesets whose default relation is war this looks
        # like a declaration but nobody declared anything.
        return "first_contact"
    return DIPL_TRANSITIONS.get((prev_st, st), "diplstate_change")


def _slot_names(ws: WorldState) -> list[str]:
    """`playerN` -> civ name.  Slots are contiguous and never freed (an
    eliminated civ stays in the save with `is_alive=FALSE`), so position in
    `ws.players` is the slot number."""
    return [p.name for p in ws.players]


def diff_events(prev: WorldState, cur: WorldState) -> list[dict[str, Any]]:
    ev = []
    prev_by = {p.name: p for p in prev.players}
    names = _slot_names(cur)

    for i, curp in enumerate(cur.players):
        name = curp.name
        prevp = prev_by.get(name)
        if prevp is None:
            continue

        # government change
        if prevp.government != curp.government:
            ev.append({
                "type": "government_change",
                "player": name, "from": prevp.government, "to": curp.government,
                "turn": cur.turn,
            })

        # tech discovered: list diff
        got = set(curp.techs) - set(prevp.techs)
        for t in sorted(got):
            ev.append({"type": "tech_discovered", "player": name,
                       "tech": t, "turn": cur.turn})

        # diplomacy change
        for slot, st in curp.diplomacy.items():
            if slot == f"player{i}":        # own row: not a relation
                continue
            prev_st = prevp.diplomacy.get(slot)
            if prev_st is None or prev_st == st:
                continue
            try:
                other = names[int(slot[6:])]
            except (ValueError, IndexError):
                # A slot with no matching player section.  Report the raw slot
                # rather than dropping the event, so the gap is visible.
                other = slot
            ev.append({"type": _dipl_type(prev_st, st), "player": name,
                       "other": other, "from": prev_st, "to": st,
                       "turn": cur.turn})

    # Cities: tracked by id **globally**, not per player.  A city that changes
    # hands keeps its id, so an owner change is a transfer -- doing this per
    # player reported it as the loser destroying a city and the winner founding
    # one (4/4 conquests mislabelled; see the module docstring).
    def _by_id(ws: WorldState) -> dict[int, tuple[str, Any]]:
        return {c.id: (p.name, c) for p in ws.players for c in p.cities}

    pc, cc = _by_id(prev), _by_id(cur)
    for cid in sorted(cc.keys() - pc.keys()):
        owner, c = cc[cid]
        e = {"type": "city_founded", "player": owner, "city": c.name,
             "city_id": cid, "size": c.size, "turn": cur.turn}
        # The savegame records the founding turn itself, so this one diff event
        # can be exact rather than "first visible at".  Verified against the
        # engine on 30/30 foundings.
        if c.founded:
            e["engine_turn"] = c.founded
        ev.append(e)
    for cid in sorted(pc.keys() - cc.keys()):
        owner, c = pc[cid]
        ev.append({"type": "city_destroyed", "player": owner, "city": c.name,
                   "city_id": cid, "turn": cur.turn})
    for cid in sorted(pc.keys() & cc.keys()):
        was, _ = pc[cid]
        now, c = cc[cid]
        if was != now:
            # `player` is the acquirer so the per-view filter shows it to them;
            # `from_player` is checked too so the loser also sees it.
            ev.append({"type": "city_transferred", "player": now,
                       "from_player": was, "city": c.name, "city_id": cid,
                       "size": c.size, "turn": cur.turn})

    return ev


def wonders_of(ws: WorldState) -> list[str]:
    """Aggregate already-built wonders across players (for H0 checks)."""
    out = []
    for p in ws.players:
        for c in p.cities:
            for k, v in c.improvements.items():
                if v and k in _WONDER_NAMES:
                    out.append((p.name, c.name, k))
    return out


_WONDER_NAMES = {
    "Apollo Program", "A.Smith's Trading Co.", "Colossus", "Copernicus' Observatory",
    "Cure For Cancer", "Darwin's Voyage", "Eiffel Tower", "Great Library",
    "Great Wall", "Hanging Gardens", "Hoover Dam", "Isaac Newton's College",
    "J.S. Bach's Cathedral", "King Richard's Crusade", "Leonardo's Workshop",
    "Lighthouse", "Magellan's Expedition", "Manhattan Project",
    "Marco Polo's Embassy", "Michelangelo's Chapel", "Mausoleum of Mausolos",
    "Statue of Zeus", "Temple of Artemis", "Pyramids", "Internet",
    "Shakespeare's Theatre", "Statue of Liberty", "Sun Tzu's War Academy",
    "United Nations", "Women's Suffrage",
}
