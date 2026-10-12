"""Render a WorldState into the world report the forecaster reads.

Design rule, and the reason this file was rewritten: **the report carries direct
observations only.**  Anything a reader could compute from what is already on the
page does not belong on it -- tile counts, land share, centroids, bounding boxes,
adjacency.  Those were in an earlier version and they were doing the model's work
for it: adjacency in particular is exactly the kind of inference we want to test,
so handing it over as a bullet point destroys the measurement.

What that leaves is the raw material:

  * the full-resolution map -- terrain, per-tile resources, roads and other
    extras, territory ownership, named features
  * where every city and every unit is, and what state it is in
  * each civ's own condition: government, treasury, economy, techs, wonders
  * the events of this turn

Three variants:

  full     everything above, omniscient.
  partial  one civ's viewpoint, masked by the engine's own fog-of-war bitmap.
           Unexplored tiles are literally unknown, not merely omitted -- so a
           civ that has never scouted the far continent cannot see who lives
           there.  This is the POMDP variant.
  minimal  events only, no map and no rosters.  The floor.

**Actions are rendered separately, by `render_actions`, and are NOT part of the
report.** The transition is

    report_t  +  actions_t   ->   report_{t+1}

and the two reports are structurally identical: same sections, same order, both
carrying their own `EVENTS THIS INTERVAL`.  Actions sit outside that structure
because they are not an observation of the world -- they are the dynamics acting
on it.  Folding them into the report would make the input and the output
asymmetric (the output has no "next turn's decisions" to show) and would blur
the thing we most need kept separate: what the world *is* versus what is being
*done to it*.

Events do belong in the report, on both sides.  A report at turn t describes the
world as of t, and "what happened since the last checkpoint" is part of that
description -- it is observable at t, unlike the decisions that will only resolve
later.
"""

from __future__ import annotations

from .state import WorldState


def render(ws: WorldState, variant: str = "full", view_player: str | None = None,
           mapinfo=None, actions=None) -> str:
    """`actions` is accepted and ignored -- use `render_actions` instead.  Kept
    in the signature so an old call site fails loudly in review rather than
    silently producing a report with decisions folded back in."""
    if variant == "full":
        return _full(ws, mapinfo)
    if variant == "partial":
        return _partial(ws, view_player, mapinfo)
    if variant == "minimal":
        return _minimal(ws)
    raise ValueError(f"unknown report variant {variant!r}")


# --------------------------------------------------------------------------
# shared pieces
# --------------------------------------------------------------------------

def _year(y: int) -> str:
    return f"{abs(y)} BCE" if y < 0 else f"{y} AD"


def _header(ws: WorldState, extra: str = "") -> list[str]:
    t = f"## WORLD REPORT{extra}"
    return [t, f"turn {ws.turn}, year {_year(ws.year)}",
            f"ruleset {ws.ruleset}, world {ws.world_id}"]


def _civ_block(p, indent: str = "") -> list[str]:
    """One civ's own condition.  Every line is a stored value."""
    inc = p.income or {}
    L = [f"{indent}{p.name} of the {p.nation}"]
    L.append(f"{indent}  government: {p.government}")
    L.append(f"{indent}  treasury: {p.gold} gold")
    L.append(f"{indent}  trade {inc.get('trade', 0)}, production "
             f"{inc.get('production', 0)}, science output {inc.get('science_output', 0)}")
    L.append(f"{indent}  revenue allocation: {inc.get('tax_pct', 0)}% treasury, "
             f"{inc.get('science_pct', 0)}% research, {inc.get('luxury_pct', 0)}% luxury")
    L.append(f"{indent}  population: {p.population} thousand")
    L.append(f"{indent}  score {p.score}, culture {p.culture}")
    L.append(f"{indent}  technologies ({len(p.techs)}): "
             + (", ".join(p.techs) if p.techs else "none"))
    if p.wonders:
        L.append(f"{indent}  great wonders: {', '.join(p.wonders)}")
    if p.small_wonders:
        L.append(f"{indent}  small wonders: {', '.join(p.small_wonders)}")
    if p.wonders_lost:
        L.append(f"{indent}  wonders destroyed while theirs: {', '.join(p.wonders_lost)}")
    if not p.is_alive:
        L.append(f"{indent}  status: eliminated")
    return L


def _city_line(c, owner: str | None = None) -> str:
    impr = sorted(k for k, v in c.improvements.items() if v)
    who = f", {owner}" if owner else ""
    return (f"- {c.name} at ({c.x},{c.y}){who}: size {c.size}, "
            f"founded turn {c.founded}, building {c.production or 'nothing'}, "
            f"food store {c.food_stock}, shield store {c.shield_stock}, "
            f"buildings: {', '.join(impr) if impr else 'none'}")


def _unit_line(u, owner: str | None = None) -> str:
    who = f", {owner}" if owner else ""
    return (f"- {u['type']} #{u['id']} at ({u['x']},{u['y']}){who}: "
            f"{u['hp']} hp, {u['moves']} moves left")


def _events(ws: WorldState, indent: str = "") -> list[str]:
    if not ws.events:
        return [f"{indent}(nothing recorded this interval)"]
    return [f"{indent}- {_fmt_event(e)}" for e in ws.events]


# Diplomacy phrasing.  Every one of these also renders the raw `from -> to`,
# because the label is a reading of the transition and the transition is the
# fact: an earlier version called `Never met -> War` a declaration of war and
# `War -> Cease-fire` a peace treaty, and both went into the corpus as prose.
_DIPL_PHRASE = {
    "first_contact": "made first contact with",
    "war_declared": "declared war on",
    "ceasefire_agreed": "agreed a cease-fire with",
    "ceasefire_expired": "saw its cease-fire with {other} expire",
    "peace_treaty_signed": "signed a peace treaty with",
    "peace_established": "is now at peace with",
    "alliance_formed": "allied with",
    "alliance_ended": "ended its alliance with",
    "diplstate_change": "changed relations with",
}


def _when(e) -> str:
    """`turn N` when the turn is known exactly, `by turn N` when it is not.

    A state diff observes that something changed between two saves, not when
    inside that interval -- and the old code claimed the later turn regardless,
    which measured one late for foundings, techs and government changes.  So the
    prose only promises what the source can support.
    """
    et = e.get("engine_turn")
    return f"turn {et}" if et is not None else f"by turn {e.get('turn', '?')}"


def _fmt_event(e) -> str:
    t, p = e["type"], e.get("player", "?")
    if t == "tech_discovered":
        return f"{_when(e)}: {p} discovered {e['tech']}"
    if t == "city_founded":
        return f"{_when(e)}: {p} founded {e['city']} (size {e.get('size', '?')})"
    if t == "city_destroyed":
        return f"{_when(e)}: {p} lost {e['city']}"
    if t == "city_transferred":
        return (f"{_when(e)}: {e['city']} passed from "
                f"{e.get('from_player', '?')} to {p} (size {e.get('size', '?')})")
    if t == "government_change":
        return f"{_when(e)}: {p} changed government from {e['from']} to {e['to']}"
    if t in _DIPL_PHRASE:
        other, phrase = e.get("other", "?"), _DIPL_PHRASE[t]
        head = (f"{p} {phrase.format(other=other)}" if "{other}" in phrase
                else f"{p} {phrase} {other}")
        return f"{_when(e)}: {head} ({e.get('from', '?')} -> {e.get('to', '?')})"
    return f"{_when(e)}: {t} ({p})"


def render_actions(actions, only: str | None = None) -> str:
    """The decisions in force this turn -- the transition's dynamics term.

    Rendered on its own, never inside a report.  These come from the intent
    fields of the turn-t savegame (tax split, research goal, per-city production,
    unit orders), which freeciv writes before the turn resolves, so they contain
    no future information.  They are what makes the world change; the report is
    what the world currently looks like.
    """
    if not actions:
        return ""
    L = [f"## ACTIONS IN FORCE"]
    L.append("")
    L.extend(_actions_block(actions, only))
    return "\n".join(L)


def _actions_block(actions, only: str | None = None) -> list[str]:
    if not actions:
        return []
    L = []
    for name, a in actions.items():
        if only and name != only:
            continue
        L.append(f"{name}:")
        L.append(f"  revenue allocation: {a.tax}% treasury, {a.science}% research, "
                 f"{a.luxury}% luxury")
        L.append(f"  researching: {a.research_goal or 'nothing'}")
        if a.production:
            L.append("  city production orders:")
            for city, what in a.production.items():
                kind = a.production_kind.get(city, "")
                mark = "  (switched this turn)" if city in a.switched else ""
                L.append(f"    {city}: {what}"
                         + (f" [{kind}]" if kind else "") + mark)
        else:
            L.append("  city production orders: none (no cities)")
        moving = [u for u in a.unit_orders if u.moving]
        holding = [u for u in a.unit_orders if not u.moving]
        if moving:
            L.append("  units under movement orders:")
            for u in moving:
                L.append(f"    {u.type} #{u.unit_id} at {u.at} heading to "
                         f"{u.destination}"
                         + (f", {u.queued_orders} queued orders" if u.queued_orders else ""))
        if holding:
            L.append("  units holding position:")
            for u in holding:
                L.append(f"    {u.type} #{u.unit_id} at {u.at}, {u.activity}")
        L.append("")
    return L


# --------------------------------------------------------------------------
# full
# --------------------------------------------------------------------------

def _full(ws: WorldState, mi=None) -> str:
    L = _header(ws)
    L.append("")

    if mi is not None:
        L.append(f"### THE MAP  ({mi.xsize} wide x {mi.ysize} tall)")
        L.append("")
        L.append("terrain, one character per tile:")
        L.append("  " + ", ".join(f"'{c}'={n}" for c, n in mi.terrain_names.items()))
        L.append("")
        L.extend(_grid_with_axis(mi.terrain, mi.xsize))
        L.append("")
        L.append("territory, the civ holding each tile ('.' = unclaimed):")
        L.extend(_grid_with_axis(_owner_chars(mi, ws), mi.xsize))
        L.append("  " + ", ".join(f"'{i}'={p.name}" for i, p in enumerate(ws.players)))
        L.append("")
        L.extend(_tile_detail(mi))
        if mi.labels:
            L.append("")
            L.append("named map features:")
            for xy, nm in sorted(mi.labels.items(),
                                 key=lambda kv: (int(kv[0].split(",")[1]),
                                                 int(kv[0].split(",")[0]))):
                x, y = xy.split(",")
                L.append(f"- {nm} at ({x},{y})")
        L.append("")

    L.append("### CIVILIZATIONS")
    L.append("")
    for p in ws.players:
        L.extend(_civ_block(p))
        L.append("")

    L.append("### DIPLOMATIC RELATIONS")
    idx = {f"player{i}": p.name for i, p in enumerate(ws.players)}
    for p in ws.players:
        rel = [f"{idx.get(k, k)}: {v}" for k, v in p.diplomacy.items()
               if idx.get(k) != p.name]
        L.append(f"- {p.name} -> " + ("; ".join(rel) if rel else "no relations"))
    L.append("")

    L.append("### CITIES")
    any_city = False
    for p in ws.players:
        for c in p.cities:
            L.append(_city_line(c, p.name))
            any_city = True
    if not any_city:
        L.append("(none)")
    L.append("")

    L.append("### UNITS")
    any_unit = False
    for p in ws.players:
        for u in p.units:
            L.append(_unit_line(u, p.name))
            any_unit = True
    if not any_unit:
        L.append("(none)")
    L.append("")

    L.append("### EVENTS THIS INTERVAL")
    L.extend(_events(ws))
    return "\n".join(L)


def _owner_chars(mi, ws) -> list[str]:
    """Ownership grid as one character per tile, for side-by-side reading."""
    rows = []
    for y in range(mi.ysize):
        row = []
        for x in range(mi.xsize):
            tok = mi.owner[y][x] if (y < len(mi.owner) and x < len(mi.owner[y])) else "-"
            row.append(tok if tok.isdigit() and int(tok) < 10 else ".")
        rows.append("".join(row))
    return rows


def _grid_with_axis(rows: list[str], xsize: int) -> list[str]:
    """Print a grid with x/y rulers so a coordinate can actually be located."""
    tens = "    " + "".join(str((x // 10) % 10) if x % 5 == 0 else " "
                            for x in range(xsize))
    ones = "    " + "".join(str(x % 10) for x in range(xsize))
    out = [tens, ones]
    for y, r in enumerate(rows):
        out.append(f"{y:3d} {r}")
    return out


def _tile_detail(mi) -> list[str]:
    """Every tile that has anything on it beyond bare terrain.

    Resources, rivers, roads, mines, huts -- listed per tile with coordinates,
    plus which city works it and whose claim reaches it.  This is the layer that
    explains why a civ is where it is, and it is a direct reading of the save,
    not a summary.
    """
    L = ["tiles with resources or improvements "
         "(coordinates, what is on them, who works them):"]
    n = 0
    for y in range(mi.ysize):
        for x in range(mi.xsize):
            ex = mi.extras[y][x] if y < len(mi.extras) else []
            if not ex:
                continue
            bits = [f"({x},{y})", ", ".join(ex)]
            w = (mi.worked[y][x] if (y < len(mi.worked) and x < len(mi.worked[y]))
                 else "-")
            if w not in ("-", ""):
                bits.append(f"worked by city #{w}")
            eo = (mi.eowner[y][x] if (y < len(mi.eowner) and x < len(mi.eowner[y]))
                  else "-")
            if eo not in ("-", ""):
                bits.append(f"extras owned by civ {eo}")
            src = (mi.source[y][x] if (y < len(mi.source) and x < len(mi.source[y]))
                   else "-")
            if src not in ("-", ""):
                bits.append(f"claim from #{src}")
            pl = (mi.placing[y][x] if (y < len(mi.placing) and x < len(mi.placing[y]))
                  else "-")
            if pl not in ("-", ""):
                it = (mi.infra_turns[y][x]
                      if (y < len(mi.infra_turns) and x < len(mi.infra_turns[y])) else "?")
                bits.append(f"building extra {pl}, {it} turns left")
            L.append("- " + "; ".join(bits))
            n += 1
    if n == 0:
        L.append("- (none)")
    return L


# --------------------------------------------------------------------------
# partial -- one civ's viewpoint, masked by the engine's fog of war
# --------------------------------------------------------------------------

def _partial(ws: WorldState, view: str | None, mi=None) -> str:
    if view is None:
        view = ws.players[0].name if ws.players else ""
    me = next((p for p in ws.players if p.name == view), None)
    L = _header(ws, f"  (as known to {view})")
    L.append("")
    if me is None:
        return "\n".join(L)

    seen = mi.known.get(view) if mi is not None else None

    if mi is not None:
        L.append(f"### THE MAP AS YOU KNOW IT  ({mi.xsize} x {mi.ysize})")
        L.append("")
        L.append("terrain ('?' = you have not explored this tile):")
        L.append("  " + ", ".join(f"'{c}'={n}" for c, n in mi.terrain_names.items()))
        L.append("")
        masked = []
        for y in range(mi.ysize):
            row = []
            for x in range(mi.xsize):
                if seen is not None and not seen[y][x]:
                    row.append("?")
                else:
                    row.append(mi.terrain[y][x] if x < len(mi.terrain[y]) else " ")
            masked.append("".join(row))
        L.extend(_grid_with_axis(masked, mi.xsize))
        L.append("")
        L.append("territory you have seen ('.' = unclaimed, '?' = unexplored):")
        oc = _owner_chars(mi, ws)
        mo = []
        for y in range(mi.ysize):
            row = []
            for x in range(mi.xsize):
                row.append("?" if (seen is not None and not seen[y][x]) else oc[y][x])
            mo.append("".join(row))
        L.extend(_grid_with_axis(mo, mi.xsize))
        L.append("  " + ", ".join(f"'{i}'={p.name}" for i, p in enumerate(ws.players)))
        L.append("")
        L.append("tiles you have seen that carry resources or improvements:")
        cnt = 0
        for y in range(mi.ysize):
            for x in range(mi.xsize):
                if seen is not None and not seen[y][x]:
                    continue
                ex = mi.extras[y][x] if y < len(mi.extras) else []
                if not ex:
                    continue
                w = (mi.worked[y][x] if (y < len(mi.worked) and x < len(mi.worked[y]))
                     else "-")
                extra = f"; worked by city #{w}" if w not in ("-", "") else ""
                L.append(f"- ({x},{y}) {', '.join(ex)}{extra}")
                cnt += 1
        if cnt == 0:
            L.append("- (none)")
        if mi.labels:
            named = [(xy, nm) for xy, nm in mi.labels.items()
                     if seen is None or seen[int(xy.split(",")[1])][int(xy.split(",")[0])]]
            if named:
                L.append("")
                L.append("named features you have found:")
                for xy, nm in named:
                    x, y = xy.split(",")
                    L.append(f"- {nm} at ({x},{y})")
        L.append("")

    L.append("### YOUR CIVILIZATION")
    L.extend(_civ_block(me))
    L.append("")

    L.append("### YOUR CITIES")
    if me.cities:
        for c in me.cities:
            L.append(_city_line(c))
    else:
        L.append("(you hold no cities)")
    L.append("")

    L.append("### YOUR UNITS")
    if me.units:
        for u in me.units:
            L.append(_unit_line(u))
    else:
        L.append("(none)")
    L.append("")

    L.append("### CIVILIZATIONS YOU HAVE MET")
    idx = {f"player{i}": p.name for i, p in enumerate(ws.players)}
    met = False
    for key, st in me.diplomacy.items():
        other = idx.get(key, key)
        if other == me.name or st == "Never met":
            continue
        met = True
        L.append(f"- {other}: {st}")
    if not met:
        L.append("(you have met no one)")
    L.append("")

    L.append("### FOREIGN UNITS AND CITIES IN YOUR SIGHT")
    spotted = False
    for p in ws.players:
        if p.name == me.name:
            continue
        for c in p.cities:
            if seen is None or (c.y < len(seen) and c.x < len(seen[c.y]) and seen[c.y][c.x]):
                L.append(_city_line(c, p.name))
                spotted = True
        for u in p.units:
            uy, ux = u["y"], u["x"]
            if seen is None or (uy < len(seen) and ux < len(seen[uy]) and seen[uy][ux]):
                L.append(_unit_line(u, p.name))
                spotted = True
    if not spotted:
        L.append("(nothing foreign in sight)")
    L.append("")

    L.append("### EVENTS YOU ARE AWARE OF")
    # A civ knows about its own events, about anything that happened *to* it
    # (losing a city to a conqueror), and about every diplomatic transition --
    # those are mutual, and `diff_events` emits one copy per side anyway.
    vis = [e for e in ws.events
           if e.get("player") == view
           or e.get("from_player") == view
           or e["type"] in _DIPL_PHRASE]
    if vis:
        for e in vis:
            L.append(f"- {_fmt_event(e)}")
    else:
        L.append("(nothing you know of)")
    return "\n".join(L)


# --------------------------------------------------------------------------
# minimal
# --------------------------------------------------------------------------

def _minimal(ws: WorldState) -> str:
    L = _header(ws)
    L.append("")
    L.append("### EVENTS THIS INTERVAL")
    L.extend(_events(ws))
    return "\n".join(L)
