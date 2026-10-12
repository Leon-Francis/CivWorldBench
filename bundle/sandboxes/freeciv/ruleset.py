"""Read freeciv `.ruleset` files, and render a static rules primer.

**Why this exists.**  The corpus records `ruleset: "civ2civ3"` and nothing else --
no tech tree, no unit stats, no government effects.  A forecaster asked to predict
the next state therefore has to *already know* civ2civ3, and when it gets a
prediction wrong there is no way to tell "cannot simulate the dynamics" from "does
not know what Alphabet unlocks".  That confound sits under every reading taken
from the corpus.  The rules are static per ruleset, so the fix is one preamble per
ruleset -- the same treatment the map gets.

⚠️ **The format is not the savegame format**, which is why `savegame.py` cannot be
reused: savegames write `key=value` with no spaces, rulesets write `name\t= _("X")`.
Measured differences that all have to be handled, and a naive parser gets wrong:

  * `key = value` with arbitrary whitespace around `=`
  * trailing `; comment` **outside quotes** (`shield = 3 ; glacier used to have 4`)
  * `_("...")` gettext wrappers, and `?qualifier:` prefixes inside them
    (`_("?unitclass:Small Land")` is the class named "Small Land")
  * 527 backslash line continuations, mostly inside `helptext`
  * multi-line `{ "col", ... / rows / }` tables -- **a building's tech prerequisite
    lives in one of these**, so a scalar-only parser silently reports every
    improvement as having no requirement
  * a section header may be any `[name]`.  A first attempt only opened a new
    section on the prefix it wanted, so the 20 `[resource_*]` and 38 `[extra_*]`
    blocks that follow the terrains were merged into the *last* terrain, which
    then reported a resource's name and a stray comment as its own fields.

**Governments are handled differently, on purpose.**  Their numbers are not in
`governments.ruleset`; they are spread over `effects.ruleset` as a requirement
vector DSL -- 492 effects in civ2civ3, 113 of them government-conditioned, with
present/absent negation and commented-out lines.  Summarising that by hand would
be *my reading* of the rules, and a confidently wrong rule in the preamble is
worse than no rule at all.  So:

  * civ2civ3 ships a numeric government table written by the ruleset authors in
    `README.civ2civ3`.  It is copied **verbatim**.
  * classic has no such table, so its government section is the authors' own
    `helptext` plus a faithful, uninterpreted listing of every effect whose
    requirements name a government.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

SECTION_RE = re.compile(r"^\[([A-Za-z0-9_]+)\]\s*$")
KV_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$")


class Table:
    """A `key = { "col", ... }` block: columns plus rows of atoms."""

    def __init__(self, columns: list[Any], rows: list[list[Any]]):
        self.columns, self.rows = columns, rows

    def as_dicts(self) -> list[dict[str, Any]]:
        return [dict(zip([str(c) for c in self.columns], r)) for r in self.rows]


def _strip_comment(line: str) -> str:
    """Drop a trailing `;` comment, but not a `;` inside a quoted string."""
    out, q = [], False
    for ch in line:
        if ch == '"':
            q = not q
        if ch == ";" and not q:
            break
        out.append(ch)
    return "".join(out).rstrip()


def _atoms(text: str) -> list[Any]:
    """Split a comma-separated value list, respecting quotes."""
    parts, buf, q = [], [], False
    for ch in text:
        if ch == '"':
            q = not q
            buf.append(ch)
        elif ch == "," and not q:
            parts.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    parts.append("".join(buf))
    return [_atom(p) for p in parts if p.strip() != ""]


def _atom(tok: str) -> Any:
    t = tok.strip()
    # `_("...")` gettext, possibly several concatenated with commas (handled by
    # the caller splitting first).
    m = re.match(r'^_\(\s*(.*?)\s*\)$', t, re.S)
    if m:
        t = m.group(1).strip()
    if t.startswith('"') and t.endswith('"') and len(t) >= 2:
        t = t[1:-1]
    # `?unitclass:Small Land` -- the qualifier is a translation hint, not part of
    # the name.
    t = re.sub(r"^\?[A-Za-z_]+:", "", t)
    if t == "TRUE":
        return True
    if t == "FALSE":
        return False
    try:
        return int(t)
    except ValueError:
        pass
    try:
        return float(t)
    except ValueError:
        return t


def parse(path: Path) -> dict[str, dict[str, Any]]:
    """`{section: {key: value}}`, values being atoms, lists or Tables."""
    raw = Path(path).read_text(encoding="utf-8", errors="replace").split("\n")

    # Join backslash continuations first, so a multi-line helptext is one logical
    # line and cannot be mistaken for a new key.
    lines: list[str] = []
    acc: str | None = None
    for ln in raw:
        cur = ln.rstrip("\r")
        if acc is not None:
            acc += cur.rstrip("\\") if cur.endswith("\\") else cur
            if not cur.endswith("\\"):
                lines.append(acc)
                acc = None
            continue
        if cur.rstrip().endswith("\\"):
            acc = cur.rstrip()[:-1]
        else:
            lines.append(cur)
    if acc is not None:
        lines.append(acc)

    out: dict[str, dict[str, Any]] = {}
    sec: dict[str, Any] | None = None
    i, n = 0, len(lines)
    while i < n:
        line = lines[i]
        bare = _strip_comment(line).strip()
        i += 1
        if not bare:
            continue
        m = SECTION_RE.match(bare)
        if m:
            # ★ Any header opens a new section.  Matching only a wanted prefix
            # merges every following block into the last match.
            sec = out.setdefault(m.group(1), {})
            continue
        if sec is None:
            continue
        m = KV_RE.match(bare)
        if not m:
            continue
        key, val = m.group(1), m.group(2).strip()
        if val.startswith("{") or val == "":
            # A table: `key =` then `{ "col", ... }` on following lines, or
            # `key = { "col", ...` on this one.
            body = val[1:] if val.startswith("{") else ""
            if not body:
                while i < n and not _strip_comment(lines[i]).strip().startswith("{"):
                    nxt = _strip_comment(lines[i]).strip()
                    if not nxt:
                        i += 1
                        continue
                    break
                if i < n and _strip_comment(lines[i]).strip().startswith("{"):
                    body = _strip_comment(lines[i]).strip()[1:]
                    i += 1
                else:
                    sec[key] = ""
                    continue
            cols = _atoms(body)
            rows = []
            while i < n:
                cur = _strip_comment(lines[i]).strip()
                i += 1
                if cur.startswith("}") or cur == "}":
                    break
                if not cur:
                    continue
                rows.append(_atoms(cur))
            sec[key] = Table(cols, rows)
            continue
        vals = _atoms(val)
        sec[key] = vals[0] if len(vals) == 1 else vals
    return out


def sections(data: dict[str, dict[str, Any]], prefix: str) -> list[tuple[str, dict]]:
    return [(k, v) for k, v in data.items() if k.startswith(prefix + "_")]


def _norm(v: Any) -> str:
    """`None` in a ruleset field means "no requirement"; render it as such rather
    than as the literal string `None`, which reads like a technology."""
    return "-" if v in (None, "None", "", "Never") else str(v)


def _tech_of(reqs: Any) -> str:
    """The Tech requirement out of a `reqs` table, or '-'.

    Buildings and units state prerequisites as rows of
    `{"type","name","range","present"}`, so a scalar read finds nothing.
    """
    if not isinstance(reqs, Table):
        return "-"
    got = [r.get("name") for r in reqs.as_dicts()
           if r.get("type") == "Tech" and r.get("present", True) is not False]
    return ", ".join(str(g) for g in got) if got else "-"


# --------------------------------------------------------------------------
# the primer
# --------------------------------------------------------------------------

def _gov_section(root: Path, name: str) -> list[str]:
    """Government rules, from the most authoritative source each ruleset has."""
    L = [f"## GOVERNMENTS ({name})"]
    readme = root / f"README.{name}"
    if readme.exists():
        txt = readme.read_text(encoding="utf-8", errors="replace").split("\n")
        start = next((i for i, l in enumerate(txt) if l.strip() == "GOVERNMENTS:"), None)
        if start is not None:
            end = next((j for j in range(start + 1, len(txt))
                        if re.match(r"^[A-Z][A-Z ]+:$", txt[j].strip())), len(txt))
            L.append("(verbatim from README.%s -- the ruleset authors' own table)" % name)
            L += [l for l in txt[start + 1:end]]
            return L

    # No authored table.  Give the authors' prose, then a faithful listing of the
    # effects whose requirements name a government -- uninterpreted, because the
    # requirement DSL is exactly what should not be paraphrased.
    govs = parse(root / "governments.ruleset")
    L.append("(helptext verbatim from governments.ruleset)")
    for _sid, g in sections(govs, "government"):
        if "name" not in g:
            continue
        ht = g.get("helptext", "")
        ht = " ".join(str(ht).split()) if not isinstance(ht, list) else " ".join(
            " ".join(str(x).split()) for x in ht)
        L.append(f"  {g['name']}: {ht}")
    L.append("")
    L.append("Effects conditioned on a government (verbatim, not paraphrased):")
    eff = parse(root / "effects.ruleset")
    for _eid, e in sections(eff, "effect"):
        reqs = e.get("reqs")
        if not isinstance(reqs, Table):
            continue
        rows = reqs.as_dicts()
        if not any(r.get("type") == "Gov" for r in rows):
            continue
        cond = "; ".join(
            f"{r.get('type')}={r.get('name')}"
            + ("" if r.get("present", True) is not False else " (absent)")
            for r in rows)
        L.append(f"  {e.get('type')} = {e.get('value')}   when {cond}")
    return L


def primer(root: Path) -> str:
    root = Path(root)
    name = root.name
    L = [f"# RULES OF THIS WORLD (freeciv ruleset: {name})",
         "Static for the whole trajectory.  Stated once, like the map.", ""]

    techs = parse(root / "techs.ruleset")
    L.append("## TECHNOLOGY TREE")
    L.append("technology: prerequisites")
    n_t = 0
    for _sid, t in sorted(sections(techs, "advance"), key=lambda kv: str(kv[1].get("name", ""))):
        if "name" not in t or t["name"] == "None":
            continue
        n_t += 1
        reqs = [str(t.get(k, "None")) for k in ("req1", "req2")]
        reqs = [r for r in reqs if r not in ("None", "", "Never")]
        L.append(f"  {t['name']}: {' + '.join(reqs) if reqs else '(available from the start)'}")

    units = parse(root / "units.ruleset")
    L.append("")
    L.append("## UNITS")
    L.append("unit | class | needs tech | cost | attack | defense | hp | firepower | moves | obsoleted by")
    n_u = 0
    for _sid, u in sections(units, "unit"):
        if "name" not in u:
            continue
        n_u += 1
        L.append("  {} | {} | {} | {} | {} | {} | {} | {} | {} | {}".format(
            u["name"], u.get("class", "?"), _norm(u.get("tech_req")),
            u.get("build_cost", "?"), u.get("attack", "?"), u.get("defense", "?"),
            u.get("hitpoints", "?"), u.get("firepower", "?"), u.get("move_rate", "?"),
            u.get("obsolete_by", "-")))

    blds = parse(root / "buildings.ruleset")
    L.append("")
    L.append("## CITY IMPROVEMENTS")
    L.append("improvement | genus | needs tech | cost | upkeep")
    n_b = 0
    for _sid, b in sections(blds, "building"):
        if "name" not in b:
            continue
        n_b += 1
        L.append("  {} | {} | {} | {} | {}".format(
            b["name"], b.get("genus", "?"), _tech_of(b.get("reqs")),
            b.get("build_cost", "?"), b.get("upkeep", "?")))

    terr = parse(root / "terrain.ruleset")
    L.append("")
    L.append("## TERRAIN")
    L.append("terrain | move cost | defense bonus % | food/shield/trade | mining | irrigation")
    n_g = 0
    for _sid, t in sections(terr, "terrain"):
        if "name" not in t or t["name"] in ("Inaccessible",):
            continue
        n_g += 1
        L.append("  {} | {} | {} | {}/{}/{} | +{} shield | +{} food".format(
            t["name"], t.get("movement_cost", "?"), t.get("defense_bonus", "?"),
            t.get("food", "?"), t.get("shield", "?"), t.get("trade", "?"),
            t.get("mining_shield_incr", 0), t.get("irrigation_food_incr", 0)))

    L.append("")
    L += _gov_section(root, name)
    L.append("")
    L.append(f"(counts: {n_t} technologies, {n_u} units, {n_b} improvements, {n_g} terrains)")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description="Render a ruleset primer")
    ap.add_argument("--root", required=True, help="a ruleset directory")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    txt = primer(Path(args.root))
    if args.out:
        Path(args.out).write_text(txt, encoding="utf-8")
    print(txt if not args.out else
          f"{args.out}: {len(txt.splitlines())} lines, {len(txt):,} chars, "
          f"~{len(txt)//4:,} tokens")


if __name__ == "__main__":
    main()
