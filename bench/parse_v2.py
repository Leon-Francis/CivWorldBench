"""Output parsers shared by run_infer_v2 (parse-rate gate) and score_v2 (scoring)."""
from __future__ import annotations

import re

# ★ 2026-09-18: the quantile field also accepts `pNN` labels (`p10 150 | p25 300 | …`).  Inkling writes
# them; every other panel model gives bare numbers.  A label is cosmetic, not a different prediction, so
# tolerating it measures forecasting rather than formatting -- `PQ_LABEL` strips them before the numbers
# are read.  All models are re-scored with this rule so no line gets a private parser.
Q5 = re.compile(r"^\s*(?P<civ>[^|]+?)\s*\|\s*(?P<var>[a-z_]+)\s*\|\s*\+?(?P<h>\d+)\s*\|"
                r"\s*(?P<p>[-\d.eE+ |,pP]+?)\s*$")
PQ_LABEL = re.compile(r"\bp\d{1,2}\b\s*", re.I)
QB = re.compile(r"^\s*Q(?P<n>\d+)\s*[:.)]?\s*(?P<p>[01]?\.?\d+)\s*$", re.M)
QM = re.compile(r"^\s*Q(?P<n>\d+)\s*[:.)]?\s*(?P<body>[^=\n]+=\s*[\d.]+(?:\s*;\s*[^=\n]+=\s*[\d.]+)*)\s*$", re.M)
T2_TRAIT = re.compile(r"^\s*(?P<civ>[^|]+?)\s*\|\s*Aggressive\s+(?P<a>low|mid|high)\s*\|\s*Expansionist\s+(?P<e>low|mid|high)"
                      r"\s*\|\s*Trader\s+(?P<t>low|mid|high)\s*\|\s*Builder\s+(?P<b>low|mid|high)\s*$", re.M | re.I)   # v2 bins (old runs)
# v3 ranking line: `Aggressive: A > B = C > D`
T2_RANK = re.compile(r"^\s*\**\s*(?P<dim>Aggressive|Expansionist|Trader|Builder)\s*\**\s*[:：]\s*(?P<body>.+?)\s*$", re.M | re.I)
T2_LOVE = re.compile(r"^\s*(?P<a>[^|>]+?)\s*->\s*(?P<b>[^|]+?)\s*\|\s*(?P<lab>hostile|neutral|friendly)\s*$", re.M | re.I)
T2_PARAM = re.compile(r"^\s*(?P<name>[^|]+?)\s*\|\s*(?P<lab>[^|\n]+?)\s*$", re.M)
BLOCK = re.compile(r"^\s*#{3,6}\s*(1A|1B|1C)\b.*$", re.M | re.I)


def parse_quantiles(text: str) -> dict:
    out = {}
    for line in text.split("\n"):
        m = Q5.match(line)
        if not m:
            continue
        nums = [x for x in re.split(r"[|\s]+", PQ_LABEL.sub("", m["p"]).strip()) if x]
        try:
            v = [float(x.replace(",", "")) for x in nums]
        except ValueError:
            continue
        if len(v) != 5:
            continue
        out[(m["civ"].strip(), m["var"], int(m["h"]))] = sorted(v)
    return out


def parse_binary(text: str, n: int) -> dict:
    out = {}
    for m in QB.finditer(text):
        i = int(m["n"])
        try:
            p = float(m["p"])
        except ValueError:
            continue
        if 1 <= i <= n and 0.0 <= p <= 1.0:
            out[i] = p
    return out


def parse_multi(text: str, n: int) -> dict:
    """`Q<n> A=0.5; B=0.3; C=0.2` -> {n: {A: 0.5, ...}} (renormalised)."""
    out = {}
    for m in QM.finditer(text):
        i = int(m["n"])
        if not (1 <= i <= n):
            continue
        d = {}
        for part in m["body"].split(";"):
            if "=" in part:
                k, v = part.rsplit("=", 1)
                try:
                    d[k.strip()] = max(0.0, float(v))
                except ValueError:
                    pass
        s = sum(d.values())
        if d and s > 0:
            out[i] = {k: v / s for k, v in d.items()}
    return out


def parse_rank(body: str) -> list[list[str]]:
    """`A > B = C > D` -> [[A], [B, C], [D]] (groups from highest to lowest)."""
    groups = []
    for grp in re.split(r"\s*>\s*", body.strip()):
        names = []
        for part in re.split(r"\s*=\s*", grp):
            name = part.strip().strip("`*")
            if len(name) >= 2 and name[0] == name[-1] and name[0] in "\"'":
                name = name[1:-1]
            if name.strip():
                names.append(name.strip())
        if names:
            groups.append(names)
    return groups


def parse_task2(text: str) -> dict:
    """v3: traits -> {dim: [[civ, ...], ...]} ranking groups; v2 bins are still parsed
    (`traits_bins`) so old runs score."""
    bins = {m["civ"].strip(): {"Aggressive": m["a"].lower(), "Expansionist": m["e"].lower(),
                               "Trader": m["t"].lower(), "Builder": m["b"].lower()} for m in T2_TRAIT.finditer(text)}
    ranks = {}
    for m in T2_RANK.finditer(text):
        dim = m["dim"].capitalize()
        if dim not in ranks:                    # first occurrence wins
            ranks[dim] = parse_rank(m["body"])
    love = {f"{m['a'].strip()}->{m['b'].strip()}": m["lab"].lower() for m in T2_LOVE.finditer(text)}
    params = {}
    for m in T2_PARAM.finditer(text):
        name = m["name"].strip().lower()
        if "->" in m.group(0) or "Aggressive" in m.group(0) or T2_RANK.match(m.group(0)):
            continue
        params[name] = m["lab"].strip().lower()
    return {"traits": ranks, "traits_bins": bins, "love": love, "params": params}


def split_task3(text: str) -> dict[str, str]:
    """Blocks `##### 1A` / `##### 1B` / `##### 1C` -> {"1a": text, "1b": text, "1c": text}."""
    marks = [(m.start(), m.end(), m.group(1).lower()) for m in BLOCK.finditer(text)]
    out = {}
    for i, (s, e, k) in enumerate(marks):
        end = marks[i + 1][0] if i + 1 < len(marks) else len(text)
        out[k] = text[e:end].strip("\n")
    if not marks:                                   # no headers: treat everything as 1a
        out["1a"] = text
    return out


def parse_rate(row: dict, text: str) -> float:
    t = row["task"]
    if t == "study":                      # belief-state ablation (2026-09-16): free text, no format to parse
        return 1.0 if len(text.strip()) > 200 else 0.0
    if t == "1a":
        want = len(row["civs"]) * 11 * len(row["horizons"])
        return min(1.0, len(parse_quantiles(text)) / max(1, want))
    if t == "1b":
        n = len(row["truth"]); nb = len(parse_binary(text, n)) + len(parse_multi(text, n))
        return min(1.0, nb / max(1, n))
    if t == "1c":
        return 1.0 if "### CIVILIZATIONS" in text else 0.0
    if t == "2":
        got = parse_task2(text)
        tr = row["truth"].get("traits") or {}
        ranking = bool(tr) and isinstance(next(iter(tr.values())), dict) and len(row["civs"]) >= 2 \
            and set(tr) <= {"Aggressive", "Expansionist", "Trader", "Builder"}
        if ranking:                                   # v3: 4 ranking lines
            want = 4 + len(row.get("pairs") or []) + len((row["truth"].get("params") or {}))
            have = len(got["traits"]) + len(got["love"]) + len(got["params"])
        else:                                         # v2: one bins line per civ
            want = len(row["civs"]) + len(row.get("pairs") or []) + len((row["truth"].get("params") or {}))
            have = len(got["traits_bins"]) + len(got["love"]) + len(got["params"])
        return min(1.0, have / max(1, want))
    if t == "3":
        sub = row.get("sub")
        if sub == "1a":                               # v3: one task per prompt
            want = len(row["civs"]) * 11 * len(row["horizons"])
            return min(1.0, len(parse_quantiles(text)) / max(1, want))
        if sub == "1b":
            n = len(row.get("items") or [])
            return min(1.0, (len(parse_binary(text, n)) + len(parse_multi(text, n))) / max(1, n))
        b = split_task3(text)                         # v2 three-block prompt (old runs)
        want = len(row["civs"]) * 11 * len(row["horizons"])
        r1a = min(1.0, len(parse_quantiles(b.get("1a", ""))) / max(1, want))
        n = len(row.get("items") or [])
        r1b = (min(1.0, (len(parse_binary(b.get("1b", ""), n)) + len(parse_multi(b.get("1b", ""), n))) / n) if n else None)
        r1c = (1.0 if "### CIVILIZATIONS" in b.get("1c", "") else 0.0) if row.get("truth", {}).get("1c") else None
        parts = [x for x in (r1a, r1b, r1c) if x is not None]
        return sum(parts) / len(parts)
    return 0.0
