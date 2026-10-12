"""Score the CoT lens: evidence lines against the intermediate truth, answers through score_v2, and
the joint readings (quadrants, consistency, selection profile, format cost).

    .venv/bin/python bench/cot/score_cot.py --run cot-pilot --raw raw_tk
    .venv/bin/python bench/cot/score_cot.py --run cot-pilot --raw raw_tk --free_ref v3-bench:raw_tk

Writes  bench/runs/<run>/cot_cells_<raw>.jsonl.gz   one record per evidence claim (+ one `meta` per prompt)
        bench/runs/<run>/<raw>.ans.<fmt>/            answer-only raw dirs (score_v2 scores them unchanged)
        bench/runs/<run>/scored_<raw>.ans.<fmt>.jsonl.gz / scores_...json   (score_v2 output per format)
        bench/runs/<run>/cot_scores_<raw>.json       the lens readings

Evidence scoring, per claim kind (all references are the SAME as 1a's so the scales match):
  V   err = |tf(v) - tf(y)| / norm  (tf = signed log1p for gold/population/culture_gain; a point value
      is the degenerate 5-quantile forecast, whose CRPS is exactly the absolute error)
      refs: persistence (turn-t1 level, 0 for the gain quantities) and trend (two-frame linear
      extrapolation to that turn, `score_v2.trend_forecast`)
      skill = 1 - Σerr / Σerr_ref  (ratio of sums, per family; n_enemies/n_allies use |Δ| like score_v2)
      flags: dup_answer (the cell is also an answered (civ, quantity, horizon)); moved (truth != persistence)
  D/G/T/O   hit vs truth; `trivial` when the claim equals the turn-t1 state (a persistence claim);
      persistence's own hit rate is the reference
Consistency:
  1a  every V line at an answered (civ, var, h) must equal the answer p50 (rel. gap); gains monotone
  1b  deterministic implication from the claims to each item (newcity <- n_cities rises, moreenemies <-
      n_enemies at tt vs base, govchange <- G differs, stillahead/leader <- score_gain lines, ...);
      the model's p vs its own implied yes/no
Quadrants: per (prompt, civ) for 1a-type prompts: answer beats trend? x evidence beats trend?; per
  item for 1b: answer beats base rate? x implied value correct?
"""
from __future__ import annotations

import argparse
import collections
import gzip
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "bundle")); sys.path.insert(0, str(ROOT / "tools"))
from bench.cot.cot_spec import (LEVEL_KEYS, N_ENSEMBLE, WONDERS, extract_code, norm_name, parse_evidence,   # noqa: E402
                                split_sections)
from bench.cot.exec_code import run_ensemble, run_predict                               # noqa: E402
from bench.parse_v2 import parse_binary, parse_multi, parse_quantiles                    # noqa: E402
from bench.spec_v2 import DELTA_VARS, FAMILIES, LEVEL_OF, VARS, op_base_of              # noqa: E402
from bench import score_v2 as SV                                                         # noqa: E402

FMTS = ("free", "table", "code")


def key_of(r: dict) -> tuple:
    return (r["cond"], r["task"], r.get("op"), r.get("h"), r.get("sub"), r.get("arm_kind"), r.get("fmt") or "free")


# ------------------------------------------------------------ claims --------
def task_value(level: dict, level_t1: dict, var: str):
    lv = LEVEL_OF[var]
    if level is None or level.get(lv) is None:
        return None
    return (level[lv] - level_t1[lv]) if var in DELTA_VARS else level[lv]


def claims_from_code(text: str, q: dict) -> tuple[list[dict], dict, str]:
    """Execute the program; trajectory -> V/D/G claims, quantiles -> 1a answer text."""
    meta = {"code_found": False, "exec_ok": False, "exec_error": None, "exec_secs": None}
    code = extract_code(text)
    if not code:
        return [], meta, ""
    meta["code_found"] = True
    res = run_predict(code, q["world_json"])
    meta["exec_secs"] = res.get("secs")
    if not res.get("ok"):
        meta["exec_error"] = res.get("error"); return [], meta, ""
    meta["exec_ok"] = True
    t1 = int(q["t1"]); civs = list(q["ev_civs"]); cmap = {norm_name(c): c for c in civs}
    lv1 = {c: (q["traj"][str(t1)].get(c) or {}) for c in civs}
    claims = []
    for tk, per in (res["trajectory"] or {}).items():
        try:
            t = int(float(tk))
        except ValueError:
            continue
        if not isinstance(per, dict):
            continue
        for cname, st in per.items():
            civ = cmap.get(norm_name(str(cname)))
            if civ is None or not isinstance(st, dict):
                continue
            for var in VARS:
                lv = LEVEL_OF[var]
                v = st.get(lv)
                if isinstance(v, (int, float)) and not isinstance(v, bool) and lv1[civ].get(lv) is not None:
                    tv = (v - lv1[civ][lv]) if var in DELTA_VARS else v
                    claims.append({"kind": "V", "civ": civ, "var": var, "turn": t, "value": float(tv), "raw": "code"})
            if isinstance(st.get("government"), str):
                claims.append({"kind": "G", "civ": civ, "turn": t, "value": st["government"], "raw": "code"})
            for oname, rel in (st.get("relations") or {}).items():
                other = cmap.get(norm_name(str(oname)))
                if other and isinstance(rel, str):
                    claims.append({"kind": "D", "civ": civ, "other": other, "turn": t, "value": rel, "raw": "code"})
    lines = []
    for cname, d in (res["quantiles"] or {}).items():
        civ = cmap.get(norm_name(str(cname)))
        if civ is None or not isinstance(d, dict):
            continue
        # accept both orientations: {civ: {var: {h: [5]}}} (protocol) and {civ: {h: {var: [5]}}} (review 2026-09-16:
        # a program that nests horizon first was executable but its answer came out empty)
        flat = []
        for k1, hh in d.items():
            if not isinstance(hh, dict):
                continue
            for k2, qs in hh.items():
                if k1 in VARS:
                    flat.append((k1, k2, qs))
                elif str(k2) in VARS:
                    flat.append((str(k2), k1, qs))
        for var, hh in [(v_, {h_: qs_}) for v_, h_, qs_ in flat]:
            for h, qs in hh.items():
                try:
                    h = int(float(str(h).lstrip("+")))
                except ValueError:
                    continue
                if isinstance(qs, list) and len(qs) == 5 and all(isinstance(x, (int, float)) for x in qs):
                    lines.append(f"{civ} | {var} | +{h} | " + " | ".join(f"{float(x):g}" for x in sorted(qs)))
    meta["n_traj_turns"] = len(res["trajectory"] or {})
    return claims, meta, "\n".join(lines)


# ------------------------------------------------------------ v3.1 helpers --
def pt_state(q: dict, civ: str, turn: int) -> dict | None:
    """Truth state of `civ` at any turn in (t1, t1+hmax] from the per-turn record (`traj_pt`); None if dead/missing."""
    pt = q.get("traj_pt"); t1 = int(q["t1"])
    if not pt:
        return None
    i = turn - t1 - 1
    if i < 0 or i >= len(pt["turns"]):
        return None
    lv = pt["levels"].get(civ)
    if lv is None or lv["gold"][i] is None:
        return None
    st = {k: lv[k][i] for k in LEVEL_KEYS}
    st["government"] = pt["government"][civ][i]
    st["relations"] = {o: v[i] for o, v in (pt["relations"].get(civ) or {}).items() if v[i] is not None}
    st["n_wonders"] = pt["n_wonders"][civ][i]
    return st


def pt_owners(q: dict, turn: int) -> dict:
    own = dict((q.get("owners") or {}).get(str(q["t1"]), {}))
    for t, city, new in (q.get("traj_pt") or {}).get("owners_changes", []):
        if t <= turn:
            if new is None:
                own.pop(city, None)
            else:
                own[city] = new
    return own


def pt_wonders(q: dict, civ: str, turn: int) -> set:
    w = set((q.get("wonders_t1") or {}).get(civ) or [])
    for t, name in ((q.get("traj_pt") or {}).get("wonders_new") or {}).get(civ, []):
        if t <= turn:
            w.add(name)
    return w



def _city_names(lst) -> list[str]:
    """Programs return city lists as names or as dicts with a "name" (review 2026-09-16 smoke: 998 unknown_city)."""
    out = []
    for c in (lst or []):
        if isinstance(c, dict):
            n = c.get("name")
            if n is not None:
                out.append(str(n))
        elif isinstance(c, (str, int)):
            out.append(str(c))
    return out


def canon_wonder(name: str) -> str | None:
    """Map a written wonder name onto the ruleset list: exact, else unique prefix / substring
    ("Shakespeare" -> "Shakespeare's Theatre"); None if it is not a wonder."""
    nm = norm_name(str(name)).replace("’", "'")
    if not nm:
        return None
    canon = {norm_name(w): w for w in WONDERS}
    if nm in canon:
        return canon[nm]
    hits = [w for k, w in canon.items() if k.startswith(nm) or nm in k or k.split("'")[0] == nm.split("'")[0] and len(nm) > 4]
    return hits[0] if len(hits) == 1 else None


def _mode(xs):
    xs = [x for x in xs if x is not None]
    return collections.Counter(xs).most_common(1)[0][0] if xs else None


def _turn_keys(traj: dict) -> list[int]:
    """Numeric turn keys of a trajectory; programs sometimes add non-turn keys ('_events', 'meta') -- skip them
    (cot-v31-main scoring crashed on '_events', 2026-09-17)."""
    out = []
    for k in traj:
        try:
            out.append(int(float(k)))
        except (TypeError, ValueError):
            continue
    return out


def _run_state(traj: dict, turn: int, cmap: dict) -> dict:
    per = traj.get(str(turn)) or {}
    return {cmap[norm_name(str(k))]: v for k, v in per.items() if norm_name(str(k)) in cmap and isinstance(v, dict)}


def items_1b_from_runs(q: dict, runs: list[dict], cmap: dict) -> list[str]:
    """Event frequencies over the ensemble -> `Q<n> p` lines (leader: `Q<n> A=..; B=..`).  Detectors mirror
    truth_v2.truth_1b_from_states on the simulated trajectory.  p = (k + 0.5) / (n + 1)."""
    t1 = int(q["t1"]); lv1 = q["traj"][str(t1)]
    own1 = {norm_name(k): v for k, v in (q.get("owners") or {}).get(str(t1), {}).items()}
    w1 = q.get("wonders_t1") or {}
    lines = []
    for i, it in enumerate(q["items"], 1):
        civ, tt, kind = it["civ"], int(it["tt"]), it["kind"]
        s1 = lv1.get(civ) if civ != "*" else None
        if kind == "leader":
            cnt = collections.Counter(); n = 0
            for run in runs:
                st = _run_state(run["trajectory"], tt, cmap)
                sc = {c: st[c].get("score") for c in it["civs"] if c in st and isinstance(st[c].get("score"), (int, float))}
                if sc:
                    cnt[max(sc, key=sc.get)] += 1; n += 1
            if n:
                lines.append(f"Q{i} " + "; ".join(f"{c}={(cnt[c] + 0.5 / len(it['civs'])) / (n + 0.5):.3f}" for c in it["civs"]))
            continue
        if s1 is None:
            continue
        k = n = 0
        for run in runs:
            traj = run["trajectory"]; turns = sorted(t for t in _turn_keys(traj) if t1 < t <= tt)
            if not turns:
                continue
            sts = {t: _run_state(traj, t, cmap).get(civ) for t in turns}
            if all(v is None for v in sts.values()):
                continue
            n += 1
            def series(key):
                return [sts[t].get(key) for t in turns if sts[t] is not None and isinstance(sts[t].get(key), (int, float))]
            last = sts.get(tt)
            hit = False
            evs = [str(e) for t in turns if sts[t] for e in (sts[t].get("events") or []) if isinstance(e, (str, int))]
            has_events = any(isinstance(sts[t].get("events"), list) for t in turns if sts[t])
            # city-name consistency: if the program re-invented names (fewer than half of the turn-t1 names survive at
            # the first turn) the name-based detectors are meaningless -> fall back to n_cities / events
            mine1 = {c for c, o in own1.items() if o == civ}
            first = next((sts[t] for t in turns if sts[t]), None)
            names_ok = bool(mine1) and first is not None and isinstance(first.get("cities"), list) and \
                len(mine1 & {norm_name(c) for c in _city_names(first.get("cities"))}) >= len(mine1) / 2
            if kind == "bankrupt":
                hit = any(v <= 0 for v in series("gold"))
            elif kind == "newcity":
                if has_events:
                    hit = "founded_city" in evs
                elif names_ok:
                    names1 = set(own1)
                    hit = any(norm_name(c) not in names1 for t in turns if sts[t] for c in _city_names(sts[t].get("cities")))
                else:
                    hit = any(v > s1["n_cities"] for v in series("n_cities"))
            elif kind == "capture":
                if has_events:
                    hit = "captured_city" in evs
                elif names_ok:
                    hit = any(own1.get(norm_name(c)) not in (None, civ) for t in turns if sts[t] for c in _city_names(sts[t].get("cities")))
                else:
                    hit = False
            elif kind == "lostcity":
                if has_events:
                    hit = "lost_city" in evs
                elif names_ok:
                    hit = any(mine1 - {norm_name(c) for c in _city_names(sts[t].get("cities"))} for t in turns if sts[t] and isinstance(sts[t].get("cities"), list))
                else:
                    hit = any(v < s1["n_cities"] for v in series("n_cities"))
            elif kind == "mildrop":
                base = it.get("base") or s1["n_military_units"]
                hit = any(v <= 0.7 * base for v in series("n_military_units"))
            elif kind == "wonder":
                base = {norm_name(x) for x in (w1.get(civ) or [])}
                hit = ("completed_wonder" in evs) if has_events else \
                    any(norm_name(str(w)) not in base for t in turns if sts[t] for w in (sts[t].get("wonders") or []))
            elif kind == "moreenemies":
                base = it.get("base", s1["n_enemies"])
                hit = last is not None and isinstance(last.get("n_enemies"), (int, float)) and last["n_enemies"] > base
            elif kind == "newally":
                al1 = {o for o, r in (s1.get("relations") or {}).items() if r == "Alliance"}
                hit = any(str(r) == "Alliance" and cmap.get(norm_name(str(o)), o) not in al1 for t in turns if sts[t] for o, r in (sts[t].get("relations") or {}).items()) \
                    or any(v > s1["n_allies"] for v in series("n_allies")) or ("new_alliance" in evs)
            elif kind == "govchange":
                hit = any(str(sts[t].get("government")).casefold() != str(s1.get("government")).casefold() for t in turns if sts[t] and sts[t].get("government")) \
                    or ("changed_government" in evs)
            elif kind == "stillahead":
                o = _run_state(traj, tt, cmap).get(it["other"])
                hit = last is not None and o is not None and isinstance(last.get("score"), (int, float)) and isinstance(o.get("score"), (int, float)) and last["score"] > o["score"]
            k += bool(hit)
        if n:
            lines.append(f"Q{i} {(k + 0.5) / (n + 1):.3f}")
    return lines


def claims_from_ensemble(text: str, q: dict) -> tuple[list[dict], dict, str]:
    """v3.1: run the stochastic simulator N_ENSEMBLE times; ensemble mean/mode at the 10-turn grid -> claims;
    ensemble percentiles -> 1a answer, event frequencies -> 1b answer."""
    meta = {"code_found": False, "exec_ok": False, "exec_error": None, "exec_secs": None, "n_runs": 0}
    code = extract_code(text)
    if not code:
        return [], meta, ""
    meta["code_found"] = True
    res = run_ensemble(code, q["world_json"], n=N_ENSEMBLE)
    meta["exec_secs"] = res.get("secs"); meta["deterministic"] = bool(res.get("deterministic"))
    runs = [r for r in (res.get("runs") or []) if r.get("trajectory")]
    meta["n_runs"] = len(runs); meta["n_run_errors"] = len(res.get("errors") or [])
    if not runs:
        meta["exec_error"] = ((res.get("errors") or [{}])[0]).get("error", "no runs"); return [], meta, ""
    meta["exec_ok"] = True
    t1 = int(q["t1"]); civs = list(q["ev_civs"]); cmap = {norm_name(c): c for c in civs}
    lv1 = {c: (q["traj"][str(t1)].get(c) or {}) for c in civs}
    claims = []
    cover = []
    for t in q["ev_turns"]:
        sts = [_run_state(r["trajectory"], int(t), cmap) for r in runs]
        cover.append(float(np.mean([bool(s) for s in sts])))
        for civ in civs:
            per = [s.get(civ) for s in sts if s.get(civ)]
            if len(per) < max(1, len(runs) // 2) or not lv1[civ]:
                continue
            for var in VARS:
                lv = LEVEL_OF[var]
                vals = [p[lv] for p in per if isinstance(p.get(lv), (int, float)) and not isinstance(p.get(lv), bool)]
                if vals and lv1[civ].get(lv) is not None:
                    v = float(np.mean(vals)); tv = (v - lv1[civ][lv]) if var in DELTA_VARS else v
                    claims.append({"kind": "V", "civ": civ, "var": var, "turn": int(t), "value": tv, "raw": "code"})
            g = _mode([p.get("government") for p in per if isinstance(p.get("government"), str)])
            if g:
                claims.append({"kind": "G", "civ": civ, "turn": int(t), "value": g, "raw": "code"})
            rel = collections.defaultdict(list)
            for p in per:
                for o, r in (p.get("relations") or {}).items():
                    oc = cmap.get(norm_name(str(o)))
                    if oc and isinstance(r, str):
                        rel[oc].append(r)
            for oc, rs in rel.items():
                claims.append({"kind": "D", "civ": civ, "other": oc, "turn": int(t), "value": _mode(rs), "raw": "code"})
            cities = collections.Counter(c for p in per for c in _city_names(p.get("cities")))
            for city, k in cities.items():
                if k >= len(per) / 2:
                    claims.append({"kind": "O", "civ": civ, "turn": int(t), "value": city, "raw": "code"})
            wond = collections.Counter(canon_wonder(w) or str(w) for p in per for w in (p.get("wonders") or []) if isinstance(w, str))
            for w, k in wond.items():
                if k >= len(per) / 2:
                    claims.append({"kind": "W", "civ": civ, "turn": int(t), "value": w, "raw": "code"})
    meta["turn_coverage"] = float(np.mean(cover)) if cover else 0.0
    meta["uses_events"] = any(isinstance(st.get("events"), list) for r in runs[:3] for per in r["trajectory"].values() if isinstance(per, dict) for st in per.values() if isinstance(st, dict))
    # answers
    lines = []; iqr = []
    if q["task"] == "1a" or q.get("sub") == "1a":
        for civ in q["civs"]:
            for var in VARS:
                lv = LEVEL_OF[var]
                for h in q["horizons"]:
                    vals = []
                    for r in runs:
                        st = _run_state(r["trajectory"], t1 + int(h), cmap).get(civ)
                        if st and isinstance(st.get(lv), (int, float)) and not isinstance(st.get(lv), bool) and lv1.get(civ, {}).get(lv) is not None:
                            vals.append((st[lv] - lv1[civ][lv]) if var in DELTA_VARS else st[lv])
                    if vals:
                        qs = np.percentile(vals, [10, 25, 50, 75, 90])
                        lines.append(f"{civ} | {var} | +{h} | " + " | ".join(f"{float(x):g}" for x in qs))
                        iqr.append(abs(SV.tf(var, qs[3]) - SV.tf(var, qs[1])) / SV.norm(var))
        meta["ens_iqr_norm_mean"] = float(np.mean(iqr)) if iqr else None
    else:
        lines = items_1b_from_runs(q, runs, cmap)
        ps = [float(l.split()[1]) for l in lines if "=" not in l]
        meta["ens_p_extreme_frac"] = float(np.mean([p < 0.05 or p > 0.95 for p in ps])) if ps else None
    meta["n_traj_turns"] = len(_turn_keys(runs[0]["trajectory"]))
    return claims, meta, "\n".join(lines)


# ------------------------------------------------------------ scoring -------
def score_claims(claims: list[dict], q: dict, cap: bool = False) -> list[dict]:
    t1 = int(q["t1"]); traj = q["traj"]; owners = q.get("owners") or {}; techs_t1 = q.get("techs_t1") or {}
    turns_ok = set(int(t) for t in q["ev_turns"])
    hz = set(int(h) for h in (q.get("horizons") or [])) if (q["task"] == "1a" or q.get("sub") == "1a") else set()
    base_hist = q.get("base_hist") or {}
    lv1 = traj[str(t1)]
    out = []
    from bench.cot.cot_spec import MAX_LINES
    try:
        from bench.techcost import tech_tree
        all_techs = {norm_name(x) for x in tech_tree(SV.WORLD_RULESET.get(q["world"], "civ2civ3"))}
    except Exception:
        all_techs = set()
    for i_c, c in enumerate(claims):
        r = {"kind": c["kind"], "civ": c["civ"], "turn": c["turn"], "h": c["turn"] - t1}
        if cap and i_c >= q.get("evidence_max_lines", MAX_LINES):
            r["status"] = "over_cap"; out.append(r); continue          # table protocol: 8-40 lines; extra lines are not scored
        per_turn = bool(q.get("traj_pt")) and c["kind"] != "T"       # v3.1: any turn has a truth (T stays on the grid)
        if per_turn:
            if not (t1 < c["turn"] <= t1 + int(q.get("hmax") or 0)):
                r["status"] = "bad_turn"; out.append(r); continue
            st = pt_state(q, c["civ"], c["turn"])
        elif q.get("traj_pt") and c["kind"] == "T" and c["turn"] not in turns_ok:
            # v3.1 T line at an off-grid turn: known at the previous grid turn -> True; unknown at the next -> False;
            # learned in between -> ambiguous (not scored)
            if not (t1 < c["turn"] <= t1 + int(q.get("hmax") or 0)):
                r["status"] = "bad_turn"; out.append(r); continue
            g_prev = max([t for t in turns_ok if t <= c["turn"]] + [t1]); g_next = min([t for t in turns_ok if t >= c["turn"]] or [None])
            nm = norm_name(c["value"]); known1 = {norm_name(x) for x in techs_t1.get(c["civ"], [])}
            def known_at(g):
                if g == t1:
                    return nm in known1
                s_ = (traj.get(str(g)) or {}).get(c["civ"])
                return None if s_ is None else (nm in known1 or nm in {norm_name(x) for x in (s_.get("techs_new") or [])})
            kp, kn = known_at(g_prev), (known_at(g_next) if g_next else None)
            if kp:
                y = True
            elif kn is False:
                y = False
            else:
                r["status"] = "ambiguous"; r["value"] = c["value"]; out.append(r); continue
            r.update(value=c["value"], status="ok", y=y, hit=y, trivial=(nm in known1), pers_hit=((nm in known1) == y), moved=(nm not in known1), off_grid=True)
            out.append(r); continue
        elif c["turn"] not in turns_ok:
            # review 2026-09-16: models wrote off-grid turns (346, 400 with t1=345) and lost their turning points;
            # snap to the nearest grid turn when it is within half a step, and flag it
            near = min(turns_ok, key=lambda t: abs(t - c["turn"])) if turns_ok else None
            if near is not None and abs(near - c["turn"]) <= 5:
                r.update(turn=near, h=near - t1, snapped=True, turn_raw=c["turn"]); c = dict(c, turn=near)
            else:
                r["status"] = "bad_turn"; out.append(r); continue
        if not per_turn:
            st = (traj.get(str(c["turn"])) or {}).get(c["civ"])
        s1 = lv1.get(c["civ"])
        if st is None or s1 is None:
            r["status"] = "dead"; out.append(r); continue
        if c["kind"] == "V":
            var = c["var"]; r["var"] = var; r["value"] = c["value"]
            if SV.degenerate(q["world"], var):
                r["status"] = "degenerate"; out.append(r); continue
            y = task_value(st, s1, var)
            if y is None:
                r["status"] = "dead"; out.append(r); continue
            if var == "techs_learned" and (s1.get("n_techs") or 0) >= SV.TECH_CEILING:
                r["status"] = "tech_ceiling"; out.append(r); continue       # same skip as score_v2 (all 87 techs known at t1)
            nz = SV.norm(var); yt = SV.tf(var, y)
            pers = task_value(s1, s1, var)                       # level at t1, 0 for gains
            r.update(status="ok", y=float(y), norm=nz, err=abs(SV.tf(var, c["value"]) - yt) / nz,
                     err_pers=abs(SV.tf(var, pers) - yt) / nz, moved=(float(y) != float(pers)),
                     dup_answer=(c["turn"] - t1) in hz)
            tr = SV.trend_forecast(base_hist.get(c["civ"]) or {}, var, t1, c["turn"] - t1)
            if tr is not None:
                if var in ("n_enemies", "n_allies"):
                    tr = min(max(tr, 0.0), float(max(1, len(q.get("ev_civs") or []) - 1)))
                r["err_trend"] = abs(SV.tf(var, tr) - yt) / nz
            if var in ("n_enemies", "n_allies"):
                r["hit"] = int(round(c["value"])) == int(y); r["ae"] = abs(c["value"] - float(y))
                r["ae_pers"] = abs(float(pers) - float(y))
                if tr is not None:
                    r["ae_trend"] = abs(tr - float(y))
        elif c["kind"] == "D":
            r["other"] = c["other"]; r["value"] = c["value"]
            y = (st.get("relations") or {}).get(c["other"]); p1 = (s1.get("relations") or {}).get(c["other"])
            if y is None:
                r["status"] = "no_truth"; out.append(r); continue
            r.update(status="ok", y=y, hit=(y.casefold() == c["value"].casefold()), pers=p1,
                     trivial=(p1 is not None and p1.casefold() == c["value"].casefold()),
                     pers_hit=(p1 is not None and p1 == y), moved=(p1 != y))
        elif c["kind"] == "G":
            r["value"] = c["value"]; y = st.get("government"); p1 = s1.get("government")
            r.update(status="ok", y=y, hit=(str(y).casefold() == c["value"].casefold()), pers=p1,
                     trivial=(str(p1).casefold() == c["value"].casefold()), pers_hit=(p1 == y), moved=(p1 != y))
        elif c["kind"] == "T":
            r["value"] = c["value"]; nm = norm_name(c["value"])
            known1 = {norm_name(x) for x in techs_t1.get(c["civ"], [])}
            new = {norm_name(x) for x in (st.get("techs_new") or [])}
            all_names = {norm_name(x) for cv in techs_t1 for x in techs_t1[cv]} | {norm_name(x) for tt in traj.values() for s in (tt or {}).values() if s for x in (s.get("techs_new") or [])}
            if nm not in all_names and nm not in known1:
                # not learned by anyone in the window: still a valid claim (truth False) if it is a real technology
                # of the ruleset (review 2026-09-16: the civ's CURRENT research target was unscorable); tolerate
                # a plural/singular slip ("Computer" / "Computers")
                cand = {nm, nm + "s", nm[:-1] if nm.endswith("s") else nm}
                if not (cand & all_techs):
                    r["status"] = "unknown_tech"; out.append(r); continue
            y = nm in known1 or nm in new
            r.update(status="ok", y=bool(y), hit=bool(y), trivial=(nm in known1), pers_hit=(nm in known1) == y,
                     moved=(nm not in known1))
        elif c["kind"] == "W":
            r["value"] = c["value"]; nm = norm_name(canon_wonder(c["value"]) or c["value"])
            have_t = {norm_name(x) for x in pt_wonders(q, c["civ"], c["turn"])}
            have_1 = {norm_name(x) for x in (q.get("wonders_t1") or {}).get(c["civ"], [])}
            known = {norm_name(x) for x in WONDERS} | {norm_name(x) for cv in (q.get("wonders_t1") or {}) for x in pt_wonders(q, cv, t1 + int(q.get("hmax") or 0))}
            if nm not in known:
                r["status"] = "unknown_wonder"; out.append(r); continue
            y = nm in have_t
            r.update(status="ok", y=bool(y), hit=bool(y), trivial=(nm in have_1), pers_hit=((nm in have_1) == y), moved=(nm not in have_1))
        elif c["kind"] == "O":
            r["value"] = c["value"]; nm = norm_name(c["value"])
            own_t = {norm_name(k): v for k, v in ((pt_owners(q, c["turn"]) if per_turn else (owners.get(str(c["turn"])) or {})).items())}
            own_1 = {norm_name(k): v for k, v in (owners.get(str(t1)) or {}).items()}
            if nm not in own_t and nm not in own_1:
                r["status"] = "unknown_city"; out.append(r); continue
            y = own_t.get(nm) == c["civ"]; p1 = own_1.get(nm)
            r.update(status="ok", y=bool(y), owner=own_t.get(nm), hit=bool(y), pers=p1,
                     trivial=(p1 == c["civ"]), pers_hit=((p1 == c["civ"]) == y), moved=(p1 != own_t.get(nm)))
        out.append(r)
    return out


def implied_1b(claims: list[dict], q: dict) -> list:
    """Per item: True / False / None (no applicable evidence).

    Soundness rule (self-test 2026-09-15): the evidence is a sparse set of waypoints (turns t1+10k), so
    for WINDOW events ("at any turn between t1 and tt") only a YES can be implied soundly (a waypoint
    that shows the event's footprint proves it happened); a NO is never implied for them, because the
    event may sit between two waypoints.  ENDPOINT items (moreenemies, stillahead, leader -- defined
    at turn tt) are implied both ways from the tt waypoint.  With perfect evidence every implication
    is correct (measured: lostcity 2/33 and capture 1/42 wrong -- re-transfers between waypoints).
    `newcity` gets no implication: its footprint (n_cities rose) is founding OR capture (24/34 wrong)."""
    t1 = int(q["t1"]); traj = q["traj"]; lv1 = traj[str(t1)]
    V = collections.defaultdict(dict)          # (civ, var) -> {turn: value}
    D = collections.defaultdict(dict); G = collections.defaultdict(dict); O = collections.defaultdict(dict); Wc = collections.defaultdict(list)
    w1 = q.get("wonders_t1") or {}
    for c in claims:
        if c["kind"] == "V":
            V[(c["civ"], c["var"])][c["turn"]] = c["value"]
        elif c["kind"] == "D":
            D[(c["civ"], c["other"])][c["turn"]] = c["value"]
            D[(c["other"], c["civ"])].setdefault(c["turn"], c["value"])   # diplstate is symmetric (review: missed newally)
        elif c["kind"] == "G":
            G[c["civ"]][c["turn"]] = c["value"]
        elif c["kind"] == "O":
            O[norm_name(c["value"])][c["turn"]] = c["civ"]
        elif c["kind"] == "W":
            Wc[c["civ"]].append((c["turn"], norm_name(c["value"])))
    own1 = {norm_name(k): v for k, v in (q.get("owners") or {}).get(str(t1), {}).items()}
    out = []
    for it in q["items"]:
        civ, tt, kind = it["civ"], int(it["tt"]), it["kind"]
        s1 = lv1.get(civ) if civ != "*" else None
        upto = lambda d: {t: v for t, v in d.items() if t1 < t <= tt}     # noqa: E731
        imp = None
        if kind == "newcity":
            imp = None        # a rising n_cities is founding OR capture: 24/34 false positives under perfect evidence (self-test)
        elif kind == "lostcity" and s1:
            lost_o = any(own1.get(city) == civ and o != civ for city, per in O.items() for t, o in per.items() if t1 < t <= tt)
            imp = True if (lost_o or any(v < s1["n_cities"] for v in upto(V[(civ, "n_cities")]).values())) else None
        elif kind == "capture":
            got = any(own1.get(city) not in (None, civ) and o == civ for city, per in O.items() for t, o in per.items() if t1 < t <= tt)
            imp = True if got else None
        elif kind == "mildrop" and s1:
            base = it.get("base") or s1["n_military_units"]
            imp = True if any(v <= 0.7 * base for v in upto(V[(civ, "n_military_units")]).values()) else None
        elif kind == "bankrupt":
            imp = True if any(v <= 0 for v in upto(V[(civ, "gold")]).values()) else None
        elif kind == "moreenemies" and s1:
            vals = V[(civ, "n_enemies")]; base = it.get("base", s1["n_enemies"])
            imp = (vals[tt] > base) if tt in vals else None
        elif kind == "newally" and s1:
            rel1 = s1.get("relations") or {}
            new_d = any(v == "Alliance" and rel1.get(o) != "Alliance" for (a, o), per in D.items() if a == civ for t, v in per.items() if t1 < t <= tt)
            imp = True if (new_d or any(v > s1["n_allies"] for v in upto(V[(civ, "n_allies")]).values())) else None
        elif kind == "govchange" and s1:
            imp = True if any(g.casefold() != str(s1.get("government")).casefold() for g in upto(G[civ]).values()) else None
        elif kind == "wonder":
            base = {norm_name(x) for x in (w1.get(civ) or [])}
            imp = True if any(t1 < t <= tt and nm not in base for t, nm in Wc.get(civ, [])) else None   # v3.1 W lines; v3.0 had none
        elif kind == "stillahead":
            a, b = civ, it["other"]; va = V[(a, "score_gain")].get(tt); vb = V[(b, "score_gain")].get(tt)
            if va is not None and vb is not None and lv1.get(a) and lv1.get(b):
                imp = (lv1[a]["score"] + va) > (lv1[b]["score"] + vb)
        elif kind == "leader":
            sc = {c: lv1[c]["score"] + V[(c, "score_gain")][tt] for c in it["civs"] if lv1.get(c) and tt in V[(c, "score_gain")]}
            imp = max(sc, key=sc.get) if len(sc) == len(it["civs"]) and sc else None
        out.append(imp)
    return out


ENDPOINT_1B = ("moreenemies", "stillahead", "leader")


def consistency_1a(claims: list[dict], got: dict, q: dict) -> dict:
    """got: parsed answer quantiles {(civ, var, h): [5]}."""
    t1 = int(q["t1"]); gaps = []; n_dup = 0
    for c in claims:
        if c["kind"] != "V":
            continue
        k = (c["civ"], c["var"], c["turn"] - t1)
        if k in got:
            n_dup += 1
            p50 = got[k][2]; nz = SV.norm(c["var"])
            gaps.append(abs(SV.tf(c["var"], c["value"]) - SV.tf(c["var"], p50)) / nz)
    mono_viol = 0; mono_n = 0
    by = collections.defaultdict(dict)
    for c in claims:
        if c["kind"] == "V" and c["var"] == "techs_learned":      # the only gain that cannot fall (score and culture can)
            by[(c["civ"], c["var"])][c["turn"]] = c["value"]
    for (civ, var), per in by.items():
        for h in (q.get("horizons") or []):
            if (civ, var, int(h)) in got:
                per = dict(per); per[t1 + int(h)] = got[(civ, var, int(h))][2]
        ts = sorted(per)
        for a, b in zip(ts, ts[1:]):
            mono_n += 1; mono_viol += per[b] < per[a] - 1e-9
    return {"n_dup": n_dup, "dup_gap_mean": (float(np.mean(gaps)) if gaps else None),
            "dup_exact": (float(np.mean([g == 0 for g in gaps])) if gaps else None),
            "mono_n": mono_n, "mono_violations": mono_viol}      # techs_learned only


# ------------------------------------------------------------ driver --------
def process_anchor(args) -> tuple[list[dict], dict[str, list[dict]]]:
    qpath, rpath, extracted = args
    Q = {key_of(json.loads(l)): json.loads(l) for l in gzip.open(qpath, "rt")}
    outs = {}
    for l in gzip.open(rpath, "rt"):
        o = json.loads(l); outs[key_of(o)] = o
    cells = []; ans_rows = collections.defaultdict(list)
    # 1b rows carry no base_hist (the frozen 1b prompt has none); the anchor's 1a row of the same cond does,
    # so V lines in 1b/3-1b evidence get the same trend reference (review 2026-09-16: trend was "—" in every 1b sheet)
    hist_by_cond = {kk[0]: qq.get("base_hist") for kk, qq in Q.items() if kk[1] == "1a" and qq.get("base_hist")}
    for k, o in outs.items():
        q = Q.get(k)
        if q is None:
            continue
        if not q.get("base_hist") and hist_by_cond.get(k[0]):
            q = dict(q, base_hist=hist_by_cond[k[0]])
        fmt = k[-1]; text = o.get("output") or ""
        meta = {"kind": "meta", "fmt": fmt, "finish": o.get("finish"), "n_gen": o.get("n_gen"), "think_tok": o.get("think_tok"),
                "think_chars": len(o.get("think_text") or "")}
        base = {kk: q.get(kk) for kk in ("world", "set", "tier", "budget", "t1", "cond", "view", "task", "sub", "arm_kind", "op_base", "is_placebo", "op_target", "score_vars", "horizon_band", "hmax")}
        base["fmt"] = fmt; base["truncated"] = o.get("finish") == "length"
        if fmt == "table":
            ev_text, ans_text = split_sections(text, flexible=q.get("protocol") == "3.2-aligned")
            pe = parse_evidence(ev_text, q["ev_civs"]); claims = pe["claims"]
            meta.update(n_lines=pe["n_lines"], n_bad=pe["n_bad"], bad=pe["bad"][:10], has_headers=("=====  EVIDENCE" in text and "=====  ANSWER" in text))
        elif fmt == "code":
            if q.get("protocol") in ("3.1", "3.2-aligned"):
                claims, cm, ans_text = claims_from_ensemble(text, q)
                meta.update(n_rounds=o.get("n_rounds"), final_ok=o.get("final_ok"), picked_round=o.get("picked_round"),
                            bt_flagged=o.get("bt_flagged"), bt_cells=o.get("bt_cells"))
            else:
                claims, cm, ans_text = claims_from_code(text, q)
            meta.update(cm); meta.update(n_lines=len(claims), n_bad=0)
        else:
            ans_text = text; claims = []
            ex = (extracted or {}).get(json.dumps([k[0], k[1], k[2], k[3], k[4], k[5]]))
            if ex:
                pe = parse_evidence(ex, q["ev_civs"]); claims = pe["claims"]
                meta.update(n_lines=pe["n_lines"], n_bad=pe["n_bad"], extracted=True)
        sc = score_claims(claims, q, cap=(fmt == "table"))
        required = {(c["civ"], c["var"], c["turn"]) for c in q.get("evidence_cells", [])}
        seen_required = set()
        for r in sc:
            ek = (r.get("civ"), r.get("var"), r.get("turn"))
            r["required"] = ek in required
            if r["required"] and ek in seen_required:
                r["status"] = "duplicate"
            if r["required"]:
                seen_required.add(ek)
            cells.append(dict(base, kind_rec="cell", **r))
        meta["required_cells"] = len(required)
        meta["missing_required_cells"] = len(required - seen_required)
        meta["n_claims"] = len(claims); meta["n_scored"] = sum(1 for r in sc if r.get("status") == "ok")
        meta["by_kind"] = dict(collections.Counter(c["kind"] for c in claims))
        meta["status"] = dict(collections.Counter(r.get("status") for r in sc))
        # answer part -> derived raw for score_v2
        ans_row = {kk: v for kk, v in o.items() if kk not in ("output", "think_text")}
        ans_row["output"] = ans_text.strip("\n"); ans_row["parse_rate"] = __import__("bench.parse_v2", fromlist=["parse_rate"]).parse_rate(q, ans_text)
        ans_rows[fmt].append(ans_row)
        meta["ans_parse_rate"] = ans_row["parse_rate"]
        if q["task"] == "1a" or q.get("sub") == "1a":
            got = parse_quantiles(ans_text)
            meta["consistency_1a"] = consistency_1a(claims, got, q)
            if o.get("perturbed"):
                got2 = parse_quantiles(o["perturbed"].get("answer") or ""); t1 = int(q["t1"]); fol = []
                for ch in o["perturbed"].get("changes") or []:
                    # v3.1: the perturbed line is an INTERMEDIATE claim; the answer cell read is the same (civ, var) at
                    # the nearest later horizon (h_ans).  followed = the answer moved in the direction of the change by
                    # at least 5% of its size (or 1 unit for counts); a legacy same-cell change is scored as before.
                    h_ans = ch.get("h_ans", ch["h"]); k = (ch["civ"], ch["var"], h_ans)
                    if k in got and k in got2:
                        p0, p1 = got[k][2], got2[k][2]; d_ev = float(ch["new"]) - float(ch["old"])
                        rel = (p1 - p0) / max(abs(p0), 1.0)
                        same_dir = (p1 - p0) * d_ev > 0
                        fol.append({"var": ch["var"], "h": ch["h"], "h_ans": h_ans, "p50_before": p0, "p50_after": p1,
                                    "ev_old": ch["old"], "ev_new": ch["new"], "rel_move": rel,
                                    "moved_frac": ((p1 - p0) / (float(ch["new"]) - p0) if h_ans == ch["h"] and float(ch["new"]) != p0 else None),
                                    "followed": bool(same_dir and (abs(rel) >= 0.05 or abs(p1 - p0) >= 1))})
                meta["perturb"] = {"n_changes": len(o["perturbed"].get("changes") or []), "n_scored": len(fol), "items": fol}
        if q["task"] == "1b" or q.get("sub") == "1b":
            n = len(q["items"]); pb = parse_binary(ans_text, n); pm = parse_multi(ans_text, n)
            imp = implied_1b(claims, q)
            truth = q["truth"] if q["task"] == "1b" else q["truth"]["1b"]
            items = []
            for i, (it, y, im) in enumerate(zip(q["items"], truth, imp), 1):
                rec = {"i": i, "kind": it["kind"], "civ": it["civ"], "h": it["h"], "implied": im, "truth": y}
                if it["kind"] == "leader":
                    rec["p_implied"] = (pm.get(i) or {}).get(im) if im else None
                else:
                    rec["p"] = pb.get(i)
                items.append(rec)
            meta["items_1b"] = items
        cells.append(dict(base, kind_rec="meta", **meta))
    return cells, ans_rows


def _ratio(num, den):
    den = float(sum(den)); return (1 - float(sum(num)) / den) if den > 0 else None


def summarize(cells: list[dict], scored_ans: dict[str, list[dict]]) -> dict:
    out = {}
    metas = [c for c in cells if c["kind_rec"] == "meta"]; cl = [c for c in cells if c["kind_rec"] == "cell"]
    groups = collections.defaultdict(list)
    for m in metas:
        groups[(m["set"], m["cond"], m["fmt"], m["task"] + (m.get("sub") or ""))].append(m)
    # ---- compliance & census
    comp = {}
    for g, ms in sorted(groups.items()):
        d = {"n_prompts": len(ms), "truncated": float(np.mean([m["truncated"] for m in ms])),
             "ans_parse_rate": float(np.mean([m.get("ans_parse_rate") or 0 for m in ms])),
             "n_claims_mean": float(np.mean([m["n_claims"] for m in ms])),
             "n_bad_mean": float(np.mean([m.get("n_bad") or 0 for m in ms])),
             "n_gen_mean": float(np.mean([m.get("n_gen") or 0 for m in ms]))}
        if g[2] == "table":
            d["has_headers"] = float(np.mean([bool(m.get("has_headers")) for m in ms]))
        if g[2] == "code":
            d["code_found"] = float(np.mean([bool(m.get("code_found")) for m in ms]))
            d["exec_ok"] = float(np.mean([bool(m.get("exec_ok")) for m in ms]))
            d["exec_errors"] = collections.Counter((m.get("exec_error") or "").split(":")[0] for m in ms if m.get("exec_error")).most_common(5)
            if any(m.get("n_runs") is not None for m in ms):          # v3.1 ensemble diagnostics
                d["n_runs_mean"] = float(np.mean([m.get("n_runs") or 0 for m in ms]))
                d["deterministic_frac"] = float(np.mean([bool(m.get("deterministic")) for m in ms]))
                d["turn_coverage_mean"] = float(np.mean([m.get("turn_coverage") or 0 for m in ms]))
                rr_ = [m.get("n_rounds") for m in ms if m.get("n_rounds")]
                d["n_rounds_mean"] = float(np.mean(rr_)) if rr_ else None
                d["final_ok_frac"] = float(np.mean([bool(m.get("final_ok")) for m in ms]))
                bf = [m["bt_flagged"] / m["bt_cells"] for m in ms if m.get("bt_flagged") is not None and m.get("bt_cells")]
                d["bt_flagged_frac_mean"] = float(np.mean(bf)) if bf else None
                d["uses_events_frac"] = float(np.mean([bool(m.get("uses_events")) for m in ms]))
                iq = [m["ens_iqr_norm_mean"] for m in ms if m.get("ens_iqr_norm_mean") is not None]
                d["ens_iqr_norm_mean"] = float(np.mean(iq)) if iq else None
                pe = [m["ens_p_extreme_frac"] for m in ms if m.get("ens_p_extreme_frac") is not None]
                d["ens_p_extreme_frac"] = float(np.mean(pe)) if pe else None
        bk = collections.Counter(); stt = collections.Counter()
        for m in ms:
            bk.update(m.get("by_kind") or {}); stt.update(m.get("status") or {})
        d["claims_by_kind"] = dict(bk); d["claim_status"] = dict(stt)
        comp["|".join(g)] = d
    out["compliance"] = comp
    # ---- evidence accuracy
    ev = {}
    for g, _ in sorted(groups.items()):
        rows = [c for c in cl if (c["set"], c["cond"], c["fmt"], c["task"] + (c.get("sub") or "")) == g and c.get("status") == "ok"]
        if not rows:
            continue
        d = {}
        V = [r for r in rows if r["kind"] == "V"]
        Vn = [r for r in V if not r.get("dup_answer")]
        d["V_n"] = len(V); d["V_n_intermediate"] = len(Vn)
        d["V_frac_dup_answer"] = float(np.mean([r.get("dup_answer", False) for r in V])) if V else None
        d["V_frac_moved"] = float(np.mean([r["moved"] for r in V])) if V else None
        d["V_turn_rel_mean"] = float(np.mean([r["h"] / max(1, r.get("hmax") or 1) for r in V])) if V else None
        for label, rs in (("all", V), ("intermediate", Vn)):
            fam = {}
            for fn, vs in FAMILIES.items():
                rr = [r for r in rs if r["var"] in vs]
                if fn == "diplomacy":
                    rr = [r for r in rr if "ae_trend" in r]
                    fam[fn] = {"n": len(rr), "skill_trend": _ratio([r["ae"] for r in rr], [r["ae_trend"] for r in rr]) if rr else None,
                               "skill_pers": _ratio([r["ae"] for r in rr], [r["ae_pers"] for r in rr]) if rr else None,
                               "acc": float(np.mean([r["hit"] for r in rr])) if rr else None}
                else:
                    rt = [r for r in rr if "err_trend" in r]
                    fam[fn] = {"n": len(rr), "skill_trend": _ratio([r["err"] for r in rt], [r["err_trend"] for r in rt]) if rt else None,
                               "skill_pers": _ratio([r["err"] for r in rr], [r["err_pers"] for r in rr]) if rr else None,
                               "win_vs_trend": float(np.mean([r["err"] < r["err_trend"] for r in rt])) if rt else None}
            vals = [f["skill_trend"] for f in fam.values() if f["skill_trend"] is not None]
            d[f"V_{label}"] = {"by_family": fam, "macro_skill_trend": float(np.mean(np.clip(vals, -1, 1))) if vals else None}
        for k in ("D", "G", "T", "O", "W"):
            rr = [r for r in rows if r["kind"] == k]
            if rr:
                nt = [r for r in rr if not r.get("trivial")]
                d[k] = {"n": len(rr), "acc": float(np.mean([r["hit"] for r in rr])), "acc_pers": float(np.mean([r["pers_hit"] for r in rr])),
                        "frac_trivial": float(np.mean([bool(r.get("trivial")) for r in rr])),
                        "n_nontrivial": len(nt), "acc_nontrivial": (float(np.mean([r["hit"] for r in nt])) if nt else None),
                        "frac_truth_moved": float(np.mean([r["moved"] for r in rr]))}
        ev["|".join(g)] = d
    out["evidence"] = ev
    # ---- consistency
    cons = {}
    for g, ms in sorted(groups.items()):
        d = {}
        c1 = [m["consistency_1a"] for m in ms if m.get("consistency_1a")]
        if c1:
            dup = [x["dup_gap_mean"] for x in c1 if x["dup_gap_mean"] is not None]
            d["1a_dup_cells"] = int(sum(x["n_dup"] for x in c1)); d["1a_dup_gap_mean"] = float(np.mean(dup)) if dup else None
            ex = [x["dup_exact"] for x in c1 if x["dup_exact"] is not None]; d["1a_dup_exact_frac"] = float(np.mean(ex)) if ex else None
            # (no monotonicity metric: n_techs falls in the mainline -- 173 drops in 3 worlds, civ2civ3 tech loss --
            #  so not even techs_learned is monotone; the raw counts stay in the meta records for inspection)
        items = [it for m in ms for it in (m.get("items_1b") or [])]
        if items:
            bi = [it for it in items if it["kind"] != "leader" and it.get("p") is not None]
            with_imp = [it for it in bi if it["implied"] is not None]
            d["1b_items"] = len(bi); d["1b_frac_with_implication"] = (len(with_imp) / len(bi)) if bi else None
            d["1b_frac_with_implication_endpoint"] = (float(np.mean([it["implied"] is not None for it in bi if it["kind"] in ENDPOINT_1B])) if any(it["kind"] in ENDPOINT_1B for it in bi) else None)
            if with_imp:
                d["1b_p_when_implied_yes"] = float(np.mean([it["p"] for it in with_imp if it["implied"]])) if any(it["implied"] for it in with_imp) else None
                d["1b_p_when_implied_no"] = float(np.mean([it["p"] for it in with_imp if not it["implied"]])) if any(not it["implied"] for it in with_imp) else None
                d["1b_agree_rate"] = float(np.mean([(it["p"] > 0.5) == bool(it["implied"]) for it in with_imp]))
                d["1b_implied_correct"] = float(np.mean([bool(it["implied"]) == bool(it["truth"]) for it in with_imp if it["truth"] is not None]))
            ld = [it for it in items if it["kind"] == "leader" and it["implied"]]
            if ld:
                d["leader_p_on_implied"] = float(np.mean([it["p_implied"] or 0 for it in ld])); d["leader_implied_correct"] = float(np.mean([it["implied"] == it["truth"] for it in ld]))
        if d:
            cons["|".join(g)] = d
        pf = [it for m in ms for it in ((m.get("perturb") or {}).get("items") or [])]
        if pf:
            d["perturb_n"] = len(pf); d["perturb_followed_frac"] = float(np.mean([x["followed"] for x in pf]))
            mv = [x["rel_move"] for x in pf if x.get("rel_move") is not None]
            d["perturb_rel_move_median"] = float(np.median(mv)) if mv else None
            d["perturb_unmoved_frac"] = float(np.mean([x["p50_after"] == x["p50_before"] for x in pf]))
            cons["|".join(g)] = d
    out["consistency"] = cons
    # ---- quadrants (1a-type: per prompt x civ; 1b: per item)
    quad = {}
    for fmt, srows in scored_ans.items():
        a_rows = [r for r in srows if r["kind"] == "1a" and "crps" in r and r.get("crps_ref") is not None]
        ans_civ = collections.defaultdict(lambda: [0.0, 0.0])
        for r in a_rows:
            k = (r["set"], r["cond"], r["world"], r["t1"], "1a", None, None, r["civ"]); ans_civ[k][0] += r["crps"]; ans_civ[k][1] += r["crps_ref"]
        a3 = [r for r in srows if r["kind"] == "3" and r.get("sub") == "1a" and "crps" in r and r.get("crps_ref")]
        ev_civ = collections.defaultdict(lambda: [0.0, 0.0])
        for c in cl:
            if c["fmt"] != fmt or c["kind"] != "V" or c.get("status") != "ok" or "err_trend" not in c or c.get("dup_answer"):
                continue
            k = (c["set"], c["cond"], c["world"], c["t1"], c["task"], c.get("sub"), c.get("arm_kind"), c["civ"]); ev_civ[k][0] += c["err"]; ev_civ[k][1] += c["err_trend"]
        qd = collections.Counter(); n = 0
        for k, (an, ad) in ans_civ.items():
            if k not in ev_civ or ad <= 0 or ev_civ[k][1] <= 0:
                continue
            a_ok = an < ad; e_ok = ev_civ[k][0] < ev_civ[k][1]; n += 1
            qd[("answer_right" if a_ok else "answer_wrong") + "/" + ("evidence_right" if e_ok else "evidence_wrong")] += 1
        quad[f"{fmt}|1a"] = {"n_units": n, **{k: v / n for k, v in qd.items()}} if n else {"n_units": 0}
        # task-3 (sub 1a) quadrant: per (arm prompt, civ), answer vs its trend reference x evidence vs trend
        ans3 = collections.defaultdict(lambda: [0.0, 0.0])
        for r in a3:
            k = (r["set"], r["cond"], r["world"], r["t1"], "3", "1a", r.get("arm_kind"), op_base_of(r.get("op") or ""), r["civ"]); ans3[k][0] += r["crps"]; ans3[k][1] += r.get("crps_ref") or 0.0
        ev3 = collections.defaultdict(lambda: [0.0, 0.0])
        for c in cl:
            if c["fmt"] != fmt or c["kind"] != "V" or c.get("status") != "ok" or "err_trend" not in c or c.get("dup_answer") or c["task"] != "3":
                continue
            k = (c["set"], c["cond"], c["world"], c["t1"], "3", c.get("sub"), c.get("arm_kind"), c.get("op_base"), c["civ"]); ev3[k][0] += c["err"]; ev3[k][1] += c["err_trend"]
        qd = collections.Counter(); n = 0
        for k, (an, ad) in ans3.items():
            if k not in ev3 or ad <= 0 or ev3[k][1] <= 0:
                continue
            a_ok = an < ad; e_ok = ev3[k][0] < ev3[k][1]; n += 1
            qd[("answer_right" if a_ok else "answer_wrong") + "/" + ("evidence_right" if e_ok else "evidence_wrong")] += 1
        quad[f"{fmt}|3-1a"] = {"n_units": n, **{k: v / n for k, v in qd.items()}} if n else {"n_units": 0}
        # 1b quadrant
        b_rows = {(r["set"], r["cond"], r["world"], r["t1"], r["kind"], r.get("arm_kind"), r["tpl"], r["civ"], r["h"]): r for r in srows if r["kind"] in ("1b", "3") and r.get("sub", "1b") == "1b" and "brier" in r and r.get("brier_base") is not None}
        qd = collections.Counter(); n = 0
        for m in metas:
            if m["fmt"] != fmt or not (m["task"] == "1b" or (m["task"] == "3" and m.get("sub") == "1b")):
                continue
            for it in m.get("items_1b") or []:
                if it["implied"] is None or it["kind"] == "leader" or it["truth"] is None:
                    continue
                r = b_rows.get((m["set"], m["cond"], m["world"], m["t1"], m["task"], m.get("arm_kind"), it["kind"], it["civ"], it["h"]))
                if r is None:
                    continue
                a_ok = r["brier"] < r["brier_base"]; e_ok = bool(it["implied"]) == bool(it["truth"]); n += 1
                qd[("answer_right" if a_ok else "answer_wrong") + "/" + ("evidence_right" if e_ok else "evidence_wrong")] += 1
        quad[f"{fmt}|1b"] = {"n_units": n, **{k: v / n for k, v in qd.items()}} if n else {"n_units": 0}
    out["quadrants"] = quad
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="cot-pilot")
    ap.add_argument("--raw", default="raw_tk")
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--bootstrap", type=int, default=200)
    ap.add_argument("--cells_run", default="", help="frozen run defining event reference cells")
    ap.add_argument("--extracted", default="", help="dir with <anchor>.json {key: evidence text} for the free format (claims extracted by a reader)")
    ap.add_argument("--fact_run", default="", help="run holding the same anchors' 1a/1b answers in the same formats (e.g. cot-main-l1 for cot-main-l3): "
                                                  "task-3 A/B need the model's own factual answers as reference")
    a = ap.parse_args()
    run = ROOT / "bench/runs" / a.run; qd = run / "questions"; rd = run / a.raw
    files = sorted(q for q in qd.glob("*.jsonl.gz") if (rd / q.name).exists())
    print(f"{len(files)} anchors with outputs in {rd}")
    ex_dir = Path(a.extracted) if a.extracted else None
    jobs = []
    for q in files:
        ex = None
        if ex_dir and (ex_dir / (q.name.replace(".jsonl.gz", ".json"))).exists():
            ex = json.load(open(ex_dir / (q.name.replace(".jsonl.gz", ".json"))))
        jobs.append((q, rd / q.name, ex))
    from concurrent.futures import ProcessPoolExecutor
    cells = []; ans = collections.defaultdict(lambda: collections.defaultdict(list))
    with ProcessPoolExecutor(a.jobs) as ex:
        for (c, ar), (q, _, _) in zip(ex.map(process_anchor, jobs, chunksize=1), jobs):
            cells.extend(c)
            for fmt, rows in ar.items():
                ans[fmt][q.name].extend(rows)
    with gzip.open(run / f"cot_cells_{a.raw}.jsonl.gz", "wt") as f:
        for c in cells:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    # answer-only raws -> score_v2, per format
    SV.build_insample_cells(a.cells_run or a.run)
    scored = {}
    for fmt, per_file in ans.items():
        d = run / f"{a.raw}.ans.{fmt}"; d.mkdir(exist_ok=True)
        for fn, rows in per_file.items():
            with gzip.open(d / fn, "wt") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        fd = (ROOT / "bench/runs" / a.fact_run / f"{a.raw}.ans.{fmt}") if a.fact_run else None
        pairs = [(qd / fn, d / fn, (fd / fn) if fd else None) for fn in sorted(per_file)]
        recs = []
        with ProcessPoolExecutor(a.jobs) as ex:
            for r in ex.map(SV.score_anchor, pairs, chunksize=1):
                recs.extend(r)
        with gzip.open(run / f"scored_{a.raw}.ans.{fmt}.jsonl.gz", "wt") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        summ = SV.summarize(recs, a.bootstrap)
        (run / f"scores_{a.raw}.ans.{fmt}.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1))
        scored[fmt] = recs
    out = summarize(cells, scored)
    # answer skill per format (format cost), from score_v2's family macro
    out["answer"] = {}
    for fmt in scored:
        summ = json.load(open(run / f"scores_{a.raw}.ans.{fmt}.json"))
        out["answer"][fmt] = {k: {"1a_macro": (v["macro_skill"] or {}).get("mean"), "1a_by_family": v.get("by_family"),
                                  "1b_gain": (summ["1b_headline_brier_gain"].get(k) or {}).get("mean")}
                              for k, v in summ["1a_family"].items()}
        out["answer"][fmt]["task3"] = {k: {"A_capture": (v.get("A_effect_S") or {}).get("mean"), "A_ratio": (v["A_effect_ratio"] or {}).get("mean"), "A_dir": v.get("A_dir_correct"), "B": (v["B_skill_unmoved"] or {}).get("mean"), "n": v.get("A_n_moved")}
                                       for k, v in summ["3"].items()}
    (run / f"cot_scores_{a.raw}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    # ---- print
    print("\n===== compliance =====")
    for k, v in out["compliance"].items():
        print(f"{k:<40} prompts {v['n_prompts']:>3}  claims/prompt {v['n_claims_mean']:5.1f}  bad {v['n_bad_mean']:4.1f}  ans-parse {v['ans_parse_rate']:.2f}  trunc {v['truncated']:.2f}"
              + (f"  exec_ok {v['exec_ok']:.2f}" if "exec_ok" in v else "") + f"  kinds {v['claims_by_kind']}  status {v['claim_status']}")
    print("\n===== evidence accuracy =====")
    for k, v in out["evidence"].items():
        vi = v.get("V_intermediate") or {}
        fams = " ".join(f"{fn}:{(f['skill_trend'] if f['skill_trend'] is not None else float('nan')):+.2f}/{(f['skill_pers'] if f['skill_pers'] is not None else float('nan')):+.2f}(n{f['n']})" for fn, f in (vi.get("by_family") or {}).items() if f["n"])
        extra = " ".join(f"{kk}:acc {v[kk]['acc']:.2f}/pers {v[kk]['acc_pers']:.2f} trivial {v[kk]['frac_trivial']:.2f} n{v[kk]['n']}" for kk in ("D", "G", "T", "O") if kk in v)
        print(f"{k:<40} V macro(trend) {vi.get('macro_skill_trend') if vi.get('macro_skill_trend') is not None else float('nan'):+.3f} [family: vs trend / vs persistence]  n_int {v.get('V_n_intermediate')} dup {v.get('V_frac_dup_answer')}  moved {v.get('V_frac_moved')}\n{'':<40} {fams}\n{'':<40} {extra}")
    print("\n===== consistency =====")
    for k, v in out["consistency"].items():
        print(k, json.dumps(v))
    print("\n===== quadrants =====")
    for k, v in out["quadrants"].items():
        print(k, json.dumps({kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items()}))
    print("\n===== answer skill by format =====")
    for fmt, d in out["answer"].items():
        for k, v in d.items():
            if k != "task3":
                print(f"{fmt:<6} {k:<20} 1a macro {v['1a_macro']}  1b gain {v['1b_gain']}")
        for k, v in d["task3"].items():
            print(f"{fmt:<6} task3 {k:<40} A {v['A_ratio']} dir {v['A_dir']} B {v['B']} n {v['n']}")
    print("wrote", run / f"cot_scores_{a.raw}.json")


if __name__ == "__main__":
    main()
