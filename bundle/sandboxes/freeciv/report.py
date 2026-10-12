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

Four variants:

  full     everything above, omniscient.
  compact  byte-for-byte the same *information* as `full`, with the tile-detail
           block re-encoded.  See `_tile_detail_compact`.
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
    if variant == "compact":
        return _full(ws, mapinfo, compact=True)
    if variant == "partial":
        return _partial(ws, view_player, mapinfo)
    if variant == "pomdp":
        # partial's content in compact's encoding, and no world_id in the header
        return _partial(ws, view_player, mapinfo, compact=True)
    if variant == "minimal":
        return _minimal(ws)
    raise ValueError(f"unknown report variant {variant!r}")


# --------------------------------------------------------------------------
# shared pieces
# --------------------------------------------------------------------------

def _year(y: int) -> str:
    return f"{abs(y)} BCE" if y < 0 else f"{y} AD"


def _header(ws: WorldState, extra: str = "", world_id: bool = True) -> list[str]:
    """The three-line header.

    `world_id=False` drops the `world <id>` field.  That is not cosmetic: `trait` is
    episode-static, so a world id in the observation lets a model satisfy any
    probe-supervision loss by memorising `world_id -> trait` instead of inferring
    anything -- the same shape as the predecessor project's query-to-level lookup
    shortcut.  Any variant used with probe supervision must drop it.
    """
    t = f"## WORLD REPORT{extra}"
    L = [t, f"turn {ws.turn}, year {_year(ws.year)}"]
    L.append(f"ruleset {ws.ruleset}, world {ws.world_id}" if world_id
             else f"ruleset {ws.ruleset}")
    return L


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


def _city_row(c, owner: str) -> str:
    """`_city_line`'s facts as a table row.  Same fields, same order, no labels --
    the labels are the same nine words repeated once per city.

    Buildings are `; `-separated, not `, `: civ2civ3 ships two improvements whose
    names contain a comma (`Aqueduct, River` and `Aqueduct, Lake`), so the
    comma-joined list `full` emits cannot be read back unambiguously -- three
    buildings and four are the same string.  Checked against every improvement,
    unit type, city name and production target in the corpus: `,` is the only one
    of `, ; | / @ + :` that occurs inside a name.
    """
    impr = sorted(k for k, v in c.improvements.items() if v)
    return " | ".join([
        owner, c.name, f"{c.x},{c.y}", str(c.size), str(c.founded),
        c.production or "nothing", str(c.food_stock), str(c.shield_stock),
        "; ".join(impr) if impr else "none"])


# The compact table/line formats, hoisted so `_full` and `_partial` cannot drift.
# A silent divergence here would mean the POMDP variant is encoded differently from
# the variant its numbers get compared against.
CITY_HEADER = ("owner | city | x,y | size | founded | building | food | shields "
               "| buildings (';'-separated)")
UNIT_HEADER = "per civ, in order: TYPE id@x,y hp/moves-left"


def _unit_cell(u) -> str:
    return f"{u['type']} {u['id']}@{u['x']},{u['y']} {u['hp']}/{u['moves']}"


def _unit_line(u, owner: str | None = None) -> str:
    who = f", {owner}" if owner else ""
    return (f"- {u['type']} #{u['id']} at ({u['x']},{u['y']}){who}: "
            f"{u['hp']} hp, {u['moves']} moves left")


def _events(ws: WorldState, indent: str = "") -> list[str]:
    if not ws.events:
        return [f"{indent}(nothing recorded this interval)"]
    return [f"{indent}- {_fmt_event(e)}" for e in ws.events]


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
        # size is always 1 at founding (measured 263/263), so printing it is noise;
        # `city_transferred` keeps it because there the range is 1-7 and it says
        # whether a large or a small city changed hands.
        return f"{_when(e)}: {p} founded {e['city']}"
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
    if only is not None and only not in actions:
        # A silent empty block is the failure mode here: the caller believes it asked
        # for one civ's decisions and gets a heading with nothing under it, which reads
        # exactly like "this civ made no decisions".
        raise ValueError(
            f"render_actions(only={only!r}) matches no civ in this action block "
            f"({sorted(actions)}) -- refusing to emit an empty ACTIONS section")
    # ★ Name the viewer in the heading.  Without it a single-civ block is
    # indistinguishable from a one-player world, and that distinction is the whole
    # point of the POMDP variant.
    L = ["## ACTIONS IN FORCE" + (f"  (yours, {only})" if only else "")]
    L.append("")
    L.extend(_actions_block(actions, only))
    return "\n".join(L)


# --------------------------------------------------------------------------
# the action path, as intent changes only
# --------------------------------------------------------------------------

INTENT_LEGEND = (
    "action path -- only what CHANGED since the previous turn is listed.\n"
    "  research X     this civ's long-term research goal is now X\n"
    "  tax a/b/c      revenue split changed to a% treasury / b% research / c% luxury\n"
    "  build C=W      city C is now producing W\n"
    "  U id -> x,y    unit id (a U) has been ordered to move to tile x,y\n"
    "  U id arrived   unit id has a move order whose destination is the tile it is\n"
    "                 already standing on (order exists and is already satisfied)\n"
    "  U id Idle      unit id is holding position, with that activity\n"
    "Every line is attributed to its owning civ.  No unit positions appear."
)


def intent_state(actions, only: str | None = None) -> tuple[dict, dict]:
    """The intent an action block expresses, with everything else stripped.

    Two exclusions, both deliberate, because `actions_path` as originally rendered
    violates the same rule that keeps `events` out of the input:

    * **No `at`.**  A unit's position on an intervening turn is a *consequence* of
      its order.  Re-rendering the whole block per turn therefore handed over
      Delta-1 partial frames of the trajectory being predicted.
    * **No `destination` when it equals `at`.**  Measured at 60.1% of units with the
      `go` flag set.  For those the "destination" is the tile the unit is standing
      on, so emitting it republishes the position through the back door -- and it
      makes every unit that moved look like its intent changed, which is most of
      what made a naive diff large.  That case carries one bit: the order exists and
      is already satisfied.
    """
    if only is not None and only not in actions:
        raise ValueError(f"intent_state(only={only!r}) matches no civ in {sorted(actions)}")
    units, pol = {}, {}
    for civ, a in actions.items():
        # ★ Filter here, upstream of the delta, so `cur` and `prev` are consistently
        # scoped.  Filtering inside `render_intent_delta` instead would leave `per_civ`
        # populated from the other civs and make `prev` and `cur` disagree about which
        # civs exist -- every other civ's first turn would then read as a change.
        if only is not None and civ != only:
            continue
        pol[civ] = {"tax": (a.tax, a.science, a.luxury),
                    "research": a.research_goal,
                    "prod": dict(a.production)}
        for o in a.unit_orders:
            dest = tuple(o.destination) if o.destination else None
            arrived = bool(o.moving and dest is not None and dest == tuple(o.at))
            units[(civ, o.unit_id)] = {
                "type": o.type, "activity": o.activity, "moving": o.moving,
                "dest": None if arrived else dest, "arrived": arrived}
    return units, pol


def render_intent_delta(turn: int, cur: tuple[dict, dict],
                        prev: tuple[dict, dict] | None,
                        roster: bool = False) -> str:
    """One turn of the action path: what changed in intent, grouped by civ.

    `roster=True` additionally reports units and cities appearing or disappearing.
    ⚠️ Those are *outcomes*, not intent -- a unit appearing is the result of a
    production order, a unit disappearing is the result of combat or of a Settlers
    founding a city (visible in the data as `Settlers 117 arrived` then
    `Settlers 117 gone`).  Reporting them re-introduces the leak this encoding
    exists to remove, so it is off by default and is a switch rather than part of
    the format.  With `roster=False` only entities already present on the previous
    turn are reported, for units and cities alike.
    """
    if prev is None:
        return ""
    u, pol = cur
    pu, pp = prev

    def tag(v):
        if v["moving"] and v["dest"]:
            return f" -> {v['dest'][0]},{v['dest'][1]}"
        if v["arrived"]:
            return " arrived"
        return f" {v['activity']}"

    per_civ: dict[str, list[str]] = {}
    for key, v in u.items():
        if pu.get(key) == v:
            continue
        if not roster and key not in pu:
            continue
        per_civ.setdefault(key[0], []).append(f"{v['type']} {key[1]}{tag(v)}")
    if roster:
        for key, pv in pu.items():
            if key not in u:
                per_civ.setdefault(key[0], []).append(f"{pv['type']} {key[1]} gone")

    L = []
    for civ, v in pol.items():
        o = pp.get(civ) or {}
        head = []
        if o.get("tax") != v["tax"]:
            head.append("tax %d/%d/%d" % v["tax"])
        if o.get("research") != v["research"]:
            head.append(f"research {v['research']}")
        old = o.get("prod") or {}
        ch = [f"{c}={w}" for c, w in v["prod"].items()
              if old.get(c) != w and (roster or c in old)]
        if ch:
            head.append("build " + "; ".join(ch))
        if roster:
            gone = [c for c in old if c not in v["prod"]]
            if gone:
                head.append("city lost " + "; ".join(gone))
        orders = per_civ.get(civ) or []
        if not head and not orders:
            continue
        L.append(f"  {civ}: " + ("; ".join(head) if head else "-"))
        if orders:
            L.append("    orders: " + "; ".join(orders))
    return f"turn {turn}\n" + "\n".join(L) if L else ""


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

def _full(ws: WorldState, mi=None, compact: bool = False) -> str:
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
        L.extend(_tile_detail_compact(mi) if compact else _tile_detail(mi))
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
    if compact:
        L.append(CITY_HEADER)
    any_city = False
    for p in ws.players:
        for c in p.cities:
            L.append(_city_row(c, p.name) if compact else _city_line(c, p.name))
            any_city = True
    if not any_city:
        L.append("(none)")
    L.append("")

    L.append("### UNITS")
    any_unit = False
    if compact:
        # One line per civ, units `;`-separated, in the same order as `full`.
        # `full` spends 34 tokens per unit on a tier-L world with 1,591 of them --
        # 48% of the whole report -- almost all of it the owner name and the words
        # "at", "hp", "moves left" repeated once per unit.
        L.append(UNIT_HEADER)
        for p in ws.players:
            if not p.units:
                continue
            any_unit = True
            L.append(f"{p.name}: " + "; ".join(_unit_cell(u) for u in p.units))
    else:
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


def _mask_grid(rows, seen, fill: str = "?") -> list[str]:
    """Replace every cell the viewer has not explored with `fill`.

    `rows` may be a list of strings (terrain) or a list of lists of chars (owner
    codes); both occur.  Written once here because `_partial` had this loop inlined
    twice with slightly different bounds handling.
    """
    out = []
    for y, row in enumerate(rows):
        cells = []
        for x in range(len(row)):
            cells.append(fill if not seen[y][x] else row[x])
        out.append("".join(cells))
    return out


def _grid_with_axis(rows: list[str], xsize: int) -> list[str]:
    """Print a grid with x/y rulers so a coordinate can actually be located."""
    tens = "    " + "".join(str((x // 10) % 10) if x % 5 == 0 else " "
                            for x in range(xsize))
    ones = "    " + "".join(str(x % 10) for x in range(xsize))
    out = [tens, ones]
    for y, r in enumerate(rows):
        out.append(f"{y:3d} {r}")
    return out


_CODE_ALPHABET = ("abcdefghijklmnopqrstuvwxyz"
                  "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")


def _tile_detail_compact(mi, seen=None) -> list[str]:
    """The same tile facts as `_tile_detail`, with the scaffolding removed.

    Measured on a tier-L world at t=1200, the prose block is 89K of the map
    section's 98K, and only 39% of it is payload: 12% is the `(x,y)` prefix
    repeated on every line, 23% is `worked by city #`, 21% is `claim from #`, and
    the extra names are spelled out in full every time they occur.  A 48x80 world
    at t=1200 lists ~1,250 tiles, so every byte of per-line boilerplate is paid
    1,250 times.

    So: coordinates become position within a `y=` row, extra names become codes
    against an emitted legend, and the remaining fields keep one-letter tags.
    Nothing is dropped -- `worked` / `eowner` / `source` / `placing` /
    `infra_turns` all survive, because they are engine state rather than anything
    a reader could recompute, and deciding they are unnecessary is a separate
    decision from deciding they are verbose.

    A grid-per-layer encoding was considered and rejected: the extra layers are
    sparse (1,250 of 3,840 tiles), so a grid spends a character on every empty
    tile and comes out *larger* than the coordinate list it replaces.
    """
    # ⚠️ Collect the legend from VISIBLE tiles only.  Scanning all of `mi.extras`
    # would enumerate every extra that exists anywhere on the map, including on
    # tiles the viewer has never explored -- the legend would leak what the mask is
    # there to hide, and it is the one line of this block a reader would never
    # suspect.
    names = sorted({e
                    for y, row in enumerate(mi.extras)
                    for x, cell in enumerate(row)
                    if seen is None or seen[y][x]
                    for e in cell})
    code = {}
    for i, nm in enumerate(names):
        if i < len(_CODE_ALPHABET):
            code[nm] = _CODE_ALPHABET[i]
        else:  # more than 62 distinct extras: two characters, still deterministic
            a, b = divmod(i - len(_CODE_ALPHABET), len(_CODE_ALPHABET))
            code[nm] = "~" + _CODE_ALPHABET[b] if a == 0 else f"~{a}{_CODE_ALPHABET[b]}"

    L = ["tiles with resources or improvements, one row per map row:",
         "  format  x:codes[ wCITY][ eCIV][ sSOURCE][ +EXTRA/TURNS]  "
         "-- w=worked by city, e=extras owner, s=claim source, +=under construction"]
    if names:
        L.append("  codes: " + ", ".join(f"{code[n]}={n}" for n in names))
    else:
        L.append("  codes: (none)")

    n = 0
    for y in range(mi.ysize):
        cells = []
        for x in range(mi.xsize):
            if seen is not None and not seen[y][x]:
                continue          # the viewer has never explored this tile
            ex = mi.extras[y][x] if y < len(mi.extras) else []
            if not ex:
                continue
            bits = f"{x}:" + "".join(code[e] for e in ex)
            w = (mi.worked[y][x] if (y < len(mi.worked) and x < len(mi.worked[y]))
                 else "-")
            if w not in ("-", ""):
                bits += f" w{w}"
            eo = (mi.eowner[y][x] if (y < len(mi.eowner) and x < len(mi.eowner[y]))
                  else "-")
            if eo not in ("-", ""):
                bits += f" e{eo}"
            src = (mi.source[y][x] if (y < len(mi.source) and x < len(mi.source[y]))
                   else "-")
            if src not in ("-", ""):
                bits += f" s{src}"
            pl = (mi.placing[y][x] if (y < len(mi.placing) and x < len(mi.placing[y]))
                  else "-")
            if pl not in ("-", ""):
                it = (mi.infra_turns[y][x]
                      if (y < len(mi.infra_turns) and x < len(mi.infra_turns[y])) else "?")
                bits += f" +{pl}/{it}"
            cells.append(bits)
            n += 1
        if cells:
            L.append(f"y={y}  " + "  ".join(cells))
    if n == 0:
        L.append("(none)")
    return L


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

def _partial(ws: WorldState, view: str | None, mi=None, compact: bool = False) -> str:
    """One civilisation's point of view.

    `compact=True` is the `pomdp` variant: the same *content* as `partial` in the same
    *encoding* as `compact`.  Both halves are load-bearing.  Measured, `partial` as
    originally written is **1.19-1.29x LARGER** than `compact` on the same frame,
    because it masks the omniscient view but keeps `full`'s per-line prose -- so the
    single-civ mask saves less than the verbose encoding costs.  A POMDP variant that
    is bigger than the omniscient one it replaces is not usable as a training input.

    ⚠️ This is not "`compact` minus a mask": the tile block below carries only
    `(x,y)` + extras + `worked`, dropping `eowner` / `source` / `placing` /
    `infra_turns`.  That is deliberate -- those are engine bookkeeping a viewer has
    no channel for -- but it means `pomdp` and `compact` are not comparable field for
    field.

    ⚠️ **Accepted infidelity in this version.** Foreign cities and units are gated on
    the *cumulative* `known` mask and then rendered with their *current* truth, so a
    tile walked past 200 turns ago still reports live city size and unit hp: memory
    shaped as observation.  The engine's own answer is in `probes.C_belief[viewer]`
    (`embassy` / `gives_shared_vision` licence the scalars, `dc` is the civ's actual
    remembered foreign cities and is 33-77% stale), but `render` has no `probes`
    parameter and `rehydrate.record_to_objects` drops `rec["probes"]`.  Left to a
    later pass; must be stated as a boundary on any conclusion drawn here.
    """
    if view is None:
        view = ws.players[0].name if ws.players else ""
    me = next((p for p in ws.players if p.name == view), None)
    if me is None:
        raise ValueError(
            f"view_player {view!r} is not in this world's players "
            f"({[p.name for p in ws.players]}) -- refusing to render a viewpoint for "
            f"a civ that does not exist")
    # ★ Refuse to fall through to an omniscient report.  `mi.known.get(view)` returning
    # None -- a missing geo block, or a viewer name that does not match a key -- used to
    # make every mask site below pass everything through, producing a fully omniscient
    # report labelled "as known to X".  Nothing downstream could have detected that.
    if mi is None:
        raise ValueError("partial/pomdp needs mapinfo: without it there is no fog mask "
                         "and the result would be an omniscient report")
    seen = mi.known.get(view)
    if seen is None:
        raise ValueError(
            f"no explored-tile mask for {view!r} in mapinfo.known "
            f"(keys: {sorted(mi.known)[:6]}...) -- refusing to render an omniscient "
            f"report labelled as a viewpoint")
    L = _header(ws, f"  (as known to {view})", world_id=not compact)
    L.append("")

    if True:
        L.append(f"### THE MAP AS YOU KNOW IT  ({mi.xsize} x {mi.ysize})")
        L.append("")
        L.append("terrain ('?' = you have not explored this tile):")
        L.append("  " + ", ".join(f"'{c}'={n}" for c, n in mi.terrain_names.items()))
        L.append("")
        L.extend(_grid_with_axis(_mask_grid(mi.terrain, seen), mi.xsize))
        L.append("")
        L.append("territory you have seen ('.' = unclaimed, '?' = unexplored):")
        L.extend(_grid_with_axis(_mask_grid(_owner_chars(mi, ws), seen), mi.xsize))
        # ⚠️ Only name the civs this viewer has actually met.  The omniscient legend
        # here used to list every player in the world, which announced the existence
        # and the name of civs the viewer has never encountered.
        idx0 = {f"player{i}": p.name for i, p in enumerate(ws.players)}
        met_names = {idx0.get(k, k) for k, st in me.diplomacy.items()
                     if st != "Never met" and idx0.get(k) != me.name}
        L.append("  " + ", ".join(f"'{i}'={p.name}" for i, p in enumerate(ws.players)
                                  if p.name == me.name or p.name in met_names))
        L.append("")
        if compact:
            L.extend(_tile_detail_compact(mi, seen))
        else:
            L.append("tiles you have seen that carry resources or improvements:")
            cnt = 0
            for y in range(mi.ysize):
                for x in range(mi.xsize):
                    if not seen[y][x]:
                        continue
                    ex = mi.extras[y][x] if y < len(mi.extras) else []
                    if not ex:
                        continue
                    w = (mi.worked[y][x]
                         if (y < len(mi.worked) and x < len(mi.worked[y])) else "-")
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
        if compact:
            L.append(CITY_HEADER)
            for c in me.cities:
                L.append(_city_row(c, me.name))
        else:
            for c in me.cities:
                L.append(_city_line(c))
    else:
        L.append("(you hold no cities)")
    L.append("")

    L.append("### YOUR UNITS")
    if me.units:
        if compact:
            L.append(UNIT_HEADER)
            L.append(f"{me.name}: " + "; ".join(_unit_cell(u) for u in me.units))
        else:
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
    def _on_seen(y, x) -> bool:
        return y < len(seen) and x < len(seen[y]) and bool(seen[y][x])
    f_cities, f_units = [], {}
    for p in ws.players:
        if p.name == me.name:
            continue
        for c in p.cities:
            if _on_seen(c.y, c.x):
                f_cities.append((c, p.name))
        # ★ Filter this civ's units BEFORE deciding whether to emit a line for it.
        # Testing `if not p.units` first would emit "CivName: " with an empty tail for
        # a civ whose every unit is out of sight -- which announces that the civ exists
        # and has units somewhere, i.e. exactly what the mask is for.
        vis = [u for u in p.units if _on_seen(u["y"], u["x"])]
        if vis:
            f_units[p.name] = vis
    if f_cities or f_units:
        if compact:
            if f_cities:
                L.append(CITY_HEADER)
                for c, owner in f_cities:
                    L.append(_city_row(c, owner))
            if f_units:
                L.append(UNIT_HEADER)
                for owner, us in f_units.items():
                    L.append(f"{owner}: " + "; ".join(_unit_cell(u) for u in us))
        else:
            for c, owner in f_cities:
                L.append(_city_line(c, owner))
            for owner, us in f_units.items():
                for u in us:
                    L.append(_unit_line(u, owner))
    else:
        L.append("(nothing foreign in sight)")
    L.append("")

    L.append("### EVENTS YOU ARE AWARE OF")
    vis = [e for e in ws.events
           if e.get("player") == view
           or e.get("from_player") == view          # 征服的另一方也该看见
           or e["type"] in _DIPL_PHRASE]            # 外交事件双方都涉及
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
