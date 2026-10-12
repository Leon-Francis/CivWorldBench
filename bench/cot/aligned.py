"""Predeclared numeric evidence cells for the Section 6 paired diagnostics."""
from __future__ import annotations

from bench.spec_v2 import VARS

PROTOCOL = "3.2-aligned"


def intermediate_offsets(horizons, excluded=()):
    finals = set(map(int, horizons)) | set(map(int, excluded))
    return sorted({int(h) * n // 4 for h in horizons for n in (1, 2, 3)} - finals - {0})


def evidence_cells(row, reference):
    if row["task"] != "1a" and row.get("sub") != "1a":
        return []
    civs = row.get("civs") or []
    target = reference.get("op_target")
    target = target if target in civs else (civs[0] if civs else None)
    variables = reference.get("score_vars") or ["score_gain", "n_military_units"]
    variables = [v for v in variables if v in VARS]
    return [{"civ": target, "var": v, "turn": int(row["t1"]) + t}
            for t in intermediate_offsets(row["horizons"], reference.get("all_answer_horizons", []))
            for v in variables] if target else []


def table_requirement(cells):
    lines = "\n".join(f"  V | {c['civ']} | {c['var']} | {c['turn']} | <your prediction>" for c in cells)
    return ("\nREQUIRED NUMERIC EVIDENCE: include exactly one numeric prediction for each cell below, "
            "using the task's quantity definitions. Replace each placeholder with your predicted number. "
            "Include all required cells first; you may add up to 8 other evidence lines. "
            "Do not duplicate a required cell. These requirements replace the general 8-40 line limit.\n"
            + lines + "\n") if cells else ""
