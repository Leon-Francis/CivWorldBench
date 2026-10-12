"""Freeciv savegame reader and surgical mutator.

Freeciv writes savegames as a "section file": `[section]` headers, `key=value`
scalars, and multi-line tables of the form

    key={"col_a","col_b"
    1,"foo"
    2,"bar"
    }

Two deliberate design choices:

1. **Reading is structured, writing is surgical.**  The mutator edits the
   original lines in place rather than re-serialising the parsed structure.
   Round-tripping a 8800-line file through a hand-written serialiser is a
   silent-corruption risk we do not need to take: the server has to be able to
   load the mutated file, and a single formatting drift breaks it.  Every
   mutation is therefore a targeted replacement of one line.

2. **The version is asserted, not guessed.**  The savegame format is version
   specific and only partly documented.  `[savefile] version` is checked
   against SUPPORTED_VERSIONS and anything else raises, because a parser that
   silently mis-reads a field is worse than one that refuses to run.

The `[random]` section carries the full RNG state (`index_J/K/X`, `table0..N`).
That is why a save/reload/continue is bit-identical (see
`scripts/freeciv_determinism_check.sh`), and it is also the handle for the
opposite operation: perturb it to draw a different future from the same
checkpoint, which is how the Monte-Carlo floor in `validate.py` is built.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SUPPORTED_VERSIONS = {50}          # [savefile] version, freeciv 3.1.x
SUPPORTED_REVISIONS = {"3.1.6"}

# Wall-clock metadata with no bearing on game state.  Excluded from state
# comparisons so that two rollouts of the same branch compare equal.
VOLATILE_KEYS = {("game", "last_turn_change_time")}

_SECTION_RE = re.compile(r"^\[([^]]+)\]\s*$")
_KV_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_.,]*)=(.*)$")


@dataclass
class Table:
    """A `key={"col",...\\n row\\n row\\n}` block."""

    columns: list[str]
    rows: list[list[Any]] = field(default_factory=list)
    header_line: int = -1              # line index of the `key={"col",...` line
    row_lines: list[int] = field(default_factory=list)

    def col(self, name: str) -> int:
        return self.columns.index(name)

    def get(self, row: int, name: str) -> Any:
        return self.rows[row][self.col(name)]

    def as_dicts(self) -> list[dict[str, Any]]:
        return [dict(zip(self.columns, r)) for r in self.rows]


def _parse_atom(tok: str) -> Any:
    tok = tok.strip()
    if tok.startswith('"') and tok.endswith('"') and len(tok) >= 2:
        return tok[1:-1]
    if tok == "TRUE":
        return True
    if tok == "FALSE":
        return False
    try:
        return int(tok)
    except ValueError:
        pass
    try:
        return float(tok)
    except ValueError:
        return tok


def _split_csv(line: str) -> list[str]:
    """Split on commas that are not inside double quotes."""
    out, buf, in_q = [], [], False
    for ch in line:
        if ch == '"':
            in_q = not in_q
            buf.append(ch)
        elif ch == "," and not in_q:
            out.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    out.append("".join(buf))
    return out


class Savegame:
    """Parsed savegame with the raw lines retained for surgical mutation."""

    def __init__(self, lines: list[str], path: Path | None = None):
        self.lines = lines
        self.path = path
        self.sections: dict[str, dict[str, Any]] = {}
        self._scalar_line: dict[tuple[str, str], int] = {}
        self._tables: dict[tuple[str, str], Table] = {}
        self._parse()
        self._check_version()

    # -- construction ------------------------------------------------------

    @classmethod
    def load(cls, path: str | Path) -> "Savegame":
        p = Path(path)
        text = p.read_text(encoding="utf-8", errors="surrogateescape")
        return cls(text.split("\n"), p)

    def _parse(self) -> None:
        sec = None
        i = 0
        n = len(self.lines)
        while i < n:
            raw = self.lines[i]
            line = raw.rstrip("\r")
            m = _SECTION_RE.match(line)
            if m:
                sec = m.group(1)
                self.sections.setdefault(sec, {})
                i += 1
                continue
            if sec is None or not line.strip():
                i += 1
                continue
            m = _KV_RE.match(line)
            if not m:
                i += 1               # continuation row of a table, already consumed
                continue
            key, val = m.group(1), m.group(2)
            if val.startswith("{"):
                tbl, i = self._parse_table(val[1:], i)
                self.sections[sec][key] = tbl
                self._tables[(sec, key)] = tbl
                continue
            if val.startswith("$"):          # dollar-quoted (lua vars); keep raw
                self.sections[sec][key] = val
                self._scalar_line[(sec, key)] = i
                i += 1
                continue
            parts = _split_csv(val)
            self.sections[sec][key] = (
                _parse_atom(parts[0]) if len(parts) == 1 else [_parse_atom(p) for p in parts]
            )
            self._scalar_line[(sec, key)] = i
            i += 1

    def _parse_table(self, header_rest: str, start: int) -> tuple[Table, int]:
        cols = [_parse_atom(c) for c in _split_csv(header_rest)] if header_rest.strip() else []
        tbl = Table(columns=[str(c) for c in cols], header_line=start)
        i = start + 1
        while i < len(self.lines):
            line = self.lines[i].rstrip("\r")
            if line.strip() == "}":
                i += 1
                break
            tbl.rows.append([_parse_atom(c) for c in _split_csv(line)])
            tbl.row_lines.append(i)
            i += 1
        return tbl, i

    def _check_version(self) -> None:
        sf = self.sections.get("savefile", {})
        ver, rev = sf.get("version"), sf.get("revision")
        if ver not in SUPPORTED_VERSIONS:
            raise ValueError(
                f"unsupported savegame version {ver!r} (supported: {SUPPORTED_VERSIONS}). "
                "Refusing to guess -- the format is version specific."
            )
        if rev not in SUPPORTED_REVISIONS:
            raise ValueError(
                f"unsupported freeciv revision {rev!r} (supported: {SUPPORTED_REVISIONS})"
            )

    # -- reading -----------------------------------------------------------

    def get(self, section: str, key: str, default: Any = None) -> Any:
        return self.sections.get(section, {}).get(key, default)

    def table(self, section: str, key: str) -> Table | None:
        v = self.sections.get(section, {}).get(key)
        return v if isinstance(v, Table) else None

    def players(self) -> list[str]:
        return sorted(
            (s for s in self.sections if re.fullmatch(r"player\d+", s)),
            key=lambda s: int(s[6:]),
        )

    @property
    def turn(self) -> int:
        return int(self.get("game", "turn", -1))

    @property
    def year(self) -> int:
        return int(self.get("game", "year", 0))

    @property
    def ruleset(self) -> str:
        return str(self.get("savefile", "rulesetdir", ""))

    # -- surgical mutation -------------------------------------------------

    def set_scalar(self, section: str, key: str, value: Any) -> None:
        idx = self._scalar_line.get((section, key))
        if idx is None:
            raise KeyError(f"no scalar {section}.{key} to replace")
        self.lines[idx] = f"{key}={_fmt(value)}"
        self.sections[section][key] = value

    def set_cell(self, section: str, table_key: str, row: int, col: str, value: Any) -> None:
        tbl = self.table(section, table_key)
        if tbl is None:
            raise KeyError(f"no table {section}.{table_key}")
        j = tbl.col(col)
        tbl.rows[row][j] = value
        line_idx = tbl.row_lines[row]
        cells = _split_csv(self.lines[line_idx].rstrip("\r"))
        cells[j] = _fmt(value)
        self.lines[line_idx] = ",".join(cells)

    def delete_rows(self, section: str, table_key: str, rows: list[int]) -> int:
        """Delete rows from a table and renumber the line index.

        Needed by the destructive interventions (raze a city, wipe an army),
        which cannot be expressed as a cell edit.  Rows are removed
        highest-index-first so the earlier `row_lines` stay valid while we work.
        Returns the number of rows removed.

        The caller is responsible for any *count* scalar that has to agree with
        the new row count (`ncities`, `nunits`): the savegame stores both, and a
        table whose length disagrees with its count is rejected on load.
        """
        tbl = self.table(section, table_key)
        if tbl is None:
            raise KeyError(f"no table {section}.{table_key}")
        drop = sorted(set(rows), reverse=True)
        for r in drop:
            if r < 0 or r >= len(tbl.rows):
                raise IndexError(f"row {r} out of range for {section}.{table_key}")
            del self.lines[tbl.row_lines[r]]
            del tbl.rows[r]
            del tbl.row_lines[r]
        # Every line index recorded anywhere after a deletion point shifted up.
        self._reindex()
        return len(drop)

    def _reindex(self) -> None:
        """Re-derive all line indices after lines were inserted or removed."""
        self.sections = {}
        self._scalar_line = {}
        self._tables = {}
        self._parse()

    def set_bitplane_bit(self, section: str, plane_prefix: str, y: int, x: int,
                         bit: int, value: bool) -> None:
        """Flip one bit in a hex-nibble bit-plane row.

        The extras layer (`eNN_YYYY`) and the fog-of-war layer (`kNN_YYYY`) are
        stored as one hex digit per tile, four bits per digit.  Setting a
        resource on a tile means finding which plane holds that extra's index,
        then flipping bit (index % 4) of the nibble at column x.
        """
        key = f"{plane_prefix}_{y:04d}"
        row = self.get(section, key)
        if row is None:
            raise KeyError(f"no bit-plane row {section}.{key}")
        row = str(row)
        if x >= len(row):
            raise IndexError(f"x={x} beyond row width {len(row)}")
        nib = int(row[x], 16)
        nib = (nib | (1 << bit)) if value else (nib & ~(1 << bit))
        self.set_scalar(section, key, row[:x] + format(nib, "x") + row[x + 1:])

    def set_csv_cell(self, section: str, prefix: str, y: int, x: int, value: Any) -> None:
        """Replace one comma-separated cell in a per-row grid.

        Covers `ownerYYYY`, `workedYYYY`, `eownerYYYY`, `sourceYYYY` and friends,
        which are CSV rows rather than bit-planes."""
        key = f"{prefix}{y:04d}"
        row = self.get(section, key)
        if row is None:
            raise KeyError(f"no grid row {section}.{key}")
        cells = [c.strip() for c in str(row).split(",")]
        if x >= len(cells):
            raise IndexError(f"x={x} beyond row width {len(cells)}")
        cells[x] = str(value)
        self.set_scalar(section, key, ",".join(cells))

    def set_char(self, section: str, prefix: str, y: int, x: int, ch: str) -> None:
        """Replace one character in a plain character grid (the terrain layer)."""
        key = f"{prefix}{y:04d}"
        row = self.get(section, key)
        if row is None:
            raise KeyError(f"no grid row {section}.{key}")
        row = str(row)
        if x >= len(row):
            raise IndexError(f"x={x} beyond row width {len(row)}")
        self.set_scalar(section, key, row[:x] + ch + row[x + 1:])

    def write(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("\n".join(self.lines), encoding="utf-8", errors="surrogateescape")
        return p

    # -- comparison --------------------------------------------------------

    def state_digest(self) -> str:
        """md5 over the file with volatile metadata removed."""
        import hashlib

        drop = {k for _, k in VOLATILE_KEYS}
        body = "\n".join(
            ln for ln in self.lines if not any(ln.startswith(k + "=") for k in drop)
        )
        return hashlib.md5(body.encode("utf-8", "surrogateescape")).hexdigest()


def _fmt(v: Any) -> str:
    if isinstance(v, bool):
        return "TRUE" if v else "FALSE"
    if isinstance(v, str):
        return v if v.startswith("$") else f'"{v}"'
    return str(v)
