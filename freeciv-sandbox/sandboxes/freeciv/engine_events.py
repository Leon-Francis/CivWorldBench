"""Harvest the engine's own event stream via freeciv's Lua signals.

The savegame stores *state*, so `events.py` has to infer what happened by
diffing it -- and a diff cannot see an actor.  The server, however, emits typed
signals with the causal parties as arguments, and a headless run can be made to
log them:

    city_destroyed(city, loser, destroyer)          <- who razed it
    city_transferred(city, from, to, reason)        <- reason="conquest"
    action_started_unit_city(action, actor, target)  <- "Conquer City", Phalanx#199
    unit_lost(unit, owner, reason)                  <- killed / city_lost / starved
                                                       / disbanded / used

Measured facts this module depends on (all verified on `wm/freeciv-server:3.1.6`):

  * 23 signals fire in a normal autogame.  `government_changed`, `war_declared`,
    `combat` and `turn_end` **do not exist** -- `signal.connect` returns false --
    so government and diplomacy stay the job of `events.py`.  The two halves are
    merged by `event_stream.py`.
  * Signals still fire on a **loaded** savegame, so branch arms can be harvested.
  * The emitted stream is **byte-identical across reload-vs-reload** once the
    log's wall-clock prefix is stripped, so a factual and a counterfactual arm
    remain comparable.
  * A fresh re-run with the same seed reproduces the stored trajectory
    byte-identically, which is what lets these events be retrofitted onto an
    already-collected corpus instead of re-collecting it.

⚠️ Two traps, both of which cost real time to find:

  1. `lua cmd` takes ONE line and **discards syntax errors with no output**, so a
     malformed hook is indistinguishable from a missing API.  Never conclude the
     hooks are attached because the run succeeded -- `parse_log` returns an
     explicit `hooks_attached` flag (see WITNESS), and a harvest without it is a
     failure, not an empty world.
  2. **`log.normal` produces nothing during script-read time (T000).**  Hooks
     must therefore be `function` definitions that run later, from inside a
     signal; any probe or diagnostic has to be emitted from a handler, not at
     the top level.  A top-level `log.normal` looks like a broken log system.

Movement is excluded deliberately: `unit_moved` plus the `Unit Move` /
`Transport Disembark` actions are 92% of all emitted lines, and unit positions
are already in the state at every turn.  ⚠️ `Found City`, `Build Road`,
`Build Irrigation` and `Build Mine` share the `action_*_unit_tile` signal with
`Unit Move`, so movement is filtered **by action name inside the Lua handler**
-- dropping the whole signal would drop city founding with it.
"""

from __future__ import annotations

import re
from typing import Any

PREFIX = "WMEV"

# How we know the hooks actually attached.  `lua cmd` discards syntax errors with
# no output, so "no events" is otherwise indistinguishable from "hooks never ran".
#
# ⚠️ This used to be a dedicated one-shot handler guarded by a Lua global
# (`wm_done`).  That was wrong in a way worth recording: **freeciv persists scalar
# Lua globals into the savegame** (`[script] vars=$wm_done=true$`).  So the guard
# leaked into every savegame the harvest wrote -- including the branch-point
# checkpoints -- which meant (a) the guard was already set when an arm forked from
# one, so the census never fired and every arm reported "hooks missing", and worse
# (b) the instrumentation had contaminated the fork point itself.
#
# `turn_begin` fires unconditionally on every turn, so its presence is a complete
# witness that costs no state at all.  **Create no Lua globals.**
WITNESS = "turn_begin"

# Signals harvested.  `turn_begin` is included because it is the only reliable
# in-band turn marker; the log's own `[Tnnn]` prefix is used as a cross-check.
SIGNALS: tuple[str, ...] = (
    "turn_begin",
    "city_built", "city_destroyed", "city_transferred", "city_size_change",
    "unit_built", "unit_lost", "unit_lost_with_transport",
    "tech_researched", "building_built",
    "disaster_occurred", "achievement_gained", "nuke_exploded", "hut_enter",
    "hut_frighten",
    "action_started_unit_unit", "action_finished_unit_unit",
    "action_started_unit_units", "action_finished_unit_units",
    "action_started_unit_city", "action_finished_unit_city",
    "action_started_unit_tile", "action_finished_unit_tile",
    "action_started_unit_self", "action_finished_unit_self",
)

# Pure movement.  Filtered by name because the `unit_tile` signal also carries
# Found City / Build Road / Build Irrigation / Build Mine.
EXCLUDE_ACTIONS: tuple[str, ...] = (
    "Unit Move", "Transport Disembark", "Transport Disembark 2",
)

# Argument names per signal, in the order the server passes them.  Taken from the
# handlers the shipped rulesets themselves connect (e.g. civ2civ3's
# `city_destroyed_callback(city, loser, destroyer)`) and confirmed by logging
# `select("#", ...)` for every signal.
ARGS: dict[str, tuple[str, ...]] = {
    "turn_begin": ("turn", "year"),
    "city_built": ("city",),
    "city_destroyed": ("city", "loser", "destroyer"),
    "city_transferred": ("city", "from_player", "to_player", "reason"),
    "city_size_change": ("city", "change", "reason"),
    "unit_built": ("unit", "city"),
    "unit_lost": ("unit", "owner", "reason"),
    "unit_lost_with_transport": ("unit", "transport", "owner"),
    "tech_researched": ("tech", "player", "how"),
    "building_built": ("building", "city"),
    "disaster_occurred": ("disaster", "city", "had_effect"),
    "achievement_gained": ("achievement", "player", "first"),
    "nuke_exploded": ("tile", "player"),
    "hut_enter": ("unit", "extra"),
    "hut_frighten": ("unit", "extra"),
}
for _s in SIGNALS:
    if _s.startswith("action_started_"):
        ARGS[_s] = ("action", "actor", "target")
    elif _s.startswith("action_finished_"):
        ARGS[_s] = ("action", "success", "actor", "target")


# --------------------------------------------------------------------------
# the Lua side
# --------------------------------------------------------------------------

# Describe one signal argument without knowing its type.  Objects are probed for
# the accessors freeciv exposes, most specific first; ids are emitted alongside
# names because names collide (two civs can share a leader name across worlds,
# and city names repeat once a site is resettled).
_DESCRIBE = (
    'function wmd(v) if v==nil then return "nil" end '
    'local t=type(v) if t~="userdata" then return tostring(v) end '
    'local ok,r=pcall(function() return v.utype:rule_name().."#"..string.format("%d",v.id) end) '
    'if ok and r~=nil then return r end '
    'ok,r=pcall(function() return v.name.."#"..string.format("%d",v.id) end) '
    'if ok and r~=nil then return r end '
    'ok,r=pcall(function() return v.name end) if ok and r~=nil then return r end '
    'ok,r=pcall(function() return v:rule_name() end) if ok and r~=nil then return r end '
    # A tile: emit coordinates as well as the id.  The id alone is freeciv's
    # native tile index; it *is* `y * xsize + x` (verified: index 916 on a 32-wide
    # map is (20,28), where the city was founded), but that formula is undocumented
    # and topology-dependent, so relying on it to join an action to the map is
    # fragile.  Coordinates make the join direct.
    'ok,r=pcall(function() return "~"..string.format("%d",v.x)..","..string.format("%d",v.y)'
    '.."#"..string.format("%d",v.id) end) if ok and r~=nil then return r end '
    'ok,r=pcall(function() return "#"..string.format("%d",v.id) end) if ok and r~=nil then return r end '
    'return "?" end'
)

# Several signals never pass the acting civ: `city_built(city)` has no founder,
# `building_built(building, city)` no builder, `action_*` only the unit.  The
# owning player is reachable from the object, so it is derived here and appended
# as a named extra -- without it the engine half of the stream cannot be attributed
# to a civ at all, which is the whole reason for harvesting it.
_OWNER = (
    'function wmo(v) if v==nil then return nil end '
    'local ok,r=pcall(function() return v.owner.name.."#"..string.format("%d",v.owner.id) end) '
    'if ok and r~=nil then return r end return nil end'
)

# Movement test, inlined into each handler as a chain of comparisons rather than
# a lookup table.  A table would be a Lua global, and although freeciv was
# measured to persist only *scalar* globals into the savegame, relying on that is
# how the `wm_done` contamination happened.  Create no globals at all.
_EXCL_TEST = " or ".join(f'an=="{a}"' for a in EXCLUDE_ACTIONS)


# Which argument the acting civ is derived from, 1-based; 0 = do not derive.
#
# ⚠️ This used to be "the first argument that has an owner", which is wrong on
# every `action_finished_*` whose actor was consumed by the action: with the actor
# gone the scan fell through to the *target*, so `Steal Tech` reported
# `player = <the victim>` on `finished` and `player = <the thief>` on `started`.
# The same field meant two different things depending on whether the unit survived.
# Deriving only from the actor makes `player` always the actor's civ, and absent
# when the actor is gone -- pair with the matching `started` event, whose actor is
# always present.
OWNER_ARG: dict[str, int] = {
    "turn_begin": 0,
    "tech_researched": 0,          # arg 2 already *is* the player
    "city_transferred": 0,         # from_player / to_player are explicit
    "unit_lost": 0,                # arg 2 already is the owner
    # v2 (2026-09-09): arg 1 of these is a *type* object with no `.owner`
    # (Building_Type / Disaster / Achievement), so the old default of 1 derived
    # nothing and `building_built` came out with no civ at all -- 36 of ~100
    # per-civ events per 80 turns were unattributable.  The city is arg 2.
    "building_built": 2,
    "disaster_occurred": 2,
    "achievement_gained": 0,       # arg 2 already is the player
    "nuke_exploded": 0,            # arg 2 already is the player
}
for _s in SIGNALS:
    if _s in OWNER_ARG:
        continue
    OWNER_ARG[_s] = 3 if _s.startswith("action_finished_") else (
        2 if _s.startswith("action_started_") else 1)


def _handler(sig: str) -> str:
    """One-line handler.  Emits `WMEV|<sig>|<arg>|<arg>|...` in engine order."""
    body = ('local n=select("#",...) local s="" '
            'for i=1,n do s=s.."|"..wmd((select(i,...))) end ')
    if sig.startswith("action_"):
        # First argument is the action; drop pure movement here so it never
        # reaches the log (92% of all lines).
        body = ('local a=select(1,...) '
                'local an=nil local ok,r=pcall(function() return a:rule_name() end) '
                f'if ok then an=r end if an~=nil and ({_EXCL_TEST}) then return end ') + body
    oi = OWNER_ARG.get(sig, 0)
    if oi:
        body += (f'local o=nil if n>={oi} then o=wmo((select({oi},...))) end '
                 'if o~=nil then s=s.."|@owner="..o end ')
    return f'function wmh_{sig}(...) {body} log.normal("{PREFIX}|{sig}%s", s) end'


def hook_lua() -> list[str]:
    """`lua cmd` lines: define handlers, connect them, then arrange a census.

    ⚠️ Order matters and every line must stay single-line -- see the module
    docstring on how `lua cmd` fails.
    """
    lines = [_DESCRIBE, _OWNER]
    for s in SIGNALS:
        lines.append(_handler(s))
    for s in SIGNALS:
        lines.append(f'signal.connect("{s}","wmh_{s}")')
    return lines


# --------------------------------------------------------------------------
# the parsing side
# --------------------------------------------------------------------------

# The engine and the state-diff stream name two things differently.  Normalising
# to the corpus vocabulary keeps one name per concept -- otherwise a consumer has
# to know both spellings, `ENGINE_COVERED` in `event_stream.py` stops matching,
# and the same founding appears twice under two names.  The engine's extra fields
# (`how` for a tech: researched / stolen / traded) survive the rename.
RENAME = {"city_built": "city_founded", "tech_researched": "tech_discovered"}

# Field aliases, same reason: `report.py` and every diff-derived event use
# `player` for the civ an event is about.
FIELD_ALIAS = {
    # city_transferred(city, from_player, to_player, reason)
    "city_transferred": {"to_player": "player"},
    # city_destroyed(city, loser, destroyer) -- the civ it happened to is the loser
    "city_destroyed": {"loser": "player"},
    # unit_lost(unit, owner, reason)
    "unit_lost": {"owner": "player"},
}

# freeciv log line:  `3: [T096 - 2026/09/01 17:54:06] in luascript_log_vargs() ...: WMEV|...`
_LINE = re.compile(r"^\d+: \[T(\d+) - [^\]]*\].*?: " + PREFIX + r"\|(.*)$")

# `Phalanx#199` / `Sipje#288` -- name and savegame id in one token.
_NAMED_ID = re.compile(r"^(.*)#(-?\d+)(?:\.0)?$")
# `~20,28#916` -- a tile: coordinates plus its native index.  A distinct sigil
# from `@`, which marks a named extra (`@owner=...`) and is split on `=`.
_TILE = re.compile(r"^~(-?\d+),(-?\d+)#(-?\d+)$")


def _atom(tok: str) -> Any:
    if tok == "nil":
        return None
    if tok == "true":
        return True
    if tok == "false":
        return False
    # Lua numbers arrive from `tostring` as `102.0`; keep integers as integers.
    if re.fullmatch(r"-?\d+\.0", tok):
        return int(float(tok))
    if re.fullmatch(r"-?\d+", tok):
        return int(tok)
    return tok


def _field(name: str, tok: str) -> dict[str, Any]:
    """One argument -> one to three record fields (`x`, `x_id`, `x_x`/`x_y`)."""
    m = _TILE.match(tok)
    if m:
        return {f"{name}_x": int(m.group(1)), f"{name}_y": int(m.group(2)),
                f"{name}_id": int(m.group(3))}
    m = _NAMED_ID.match(tok)
    if m and m.group(1):
        return {name: m.group(1), f"{name}_id": int(m.group(2))}
    if tok.startswith("#"):
        return {f"{name}_id": int(float(tok[1:]))}
    return {name: _atom(tok)}


def parse_log(log: str) -> tuple[dict[int, list[dict[str, Any]]], bool]:
    """`(events_by_turn, hooks_attached)`.

    Order **within** a turn is the engine's own emission order and is preserved:
    an `Attack` immediately followed by `unit_lost(reason="killed")` is how the
    kill is attributed, so reordering would destroy the attribution.

    `hooks_attached` is true once a `turn_begin` line has been seen -- see
    WITNESS.  A harvest without it produced no events because the hooks never
    ran, which is not the same as a world in which nothing happened, and
    `lua cmd` reports neither.
    """
    by_turn: dict[int, list[dict[str, Any]]] = {}
    attached = False
    for line in log.splitlines():
        m = _LINE.match(line)
        if not m:
            continue
        turn, rest = int(m.group(1)), m.group(2)
        parts = rest.split("|")
        sig, vals = parts[0], parts[1:]
        if sig == WITNESS:
            attached = True
            by_turn.setdefault(turn, [])
            continue
        names = ARGS.get(sig, ())
        # An engine event happens on the turn the signal fires, so its turn is
        # exact -- verified +0 against `[event_cache]`'s own turn column on 30/30
        # foundings and 58/58 techs.  `engine_turn` is set as well so consumers
        # need only one rule: prefer `engine_turn`, fall back to `turn`.
        rec: dict[str, Any] = {"type": RENAME.get(sig, sig), "turn": turn,
                               "engine_turn": turn}
        alias = FIELD_ALIAS.get(sig, {})
        pos = 0
        extras: list[tuple[str, str]] = []
        for v in vals:
            if v.startswith("@"):
                k, _, val = v[1:].partition("=")
                extras.append((k, val))
                continue
            nm = names[pos] if pos < len(names) else f"arg{pos}"
            pos += 1
            rec.update(_field(alias.get(nm, nm), v))
        for k, val in extras:
            # Derived in Lua, so it must never beat a value the signal passed
            # explicitly: at the moment `city_transferred` fires, `city.owner`
            # may be either side, while the signal's own `to_player` is exact.
            f = _field("player" if k == "owner" else k, val)
            for fk, fv in f.items():
                rec.setdefault(fk, fv)
        by_turn.setdefault(turn, []).append(rec)
    return by_turn, attached
