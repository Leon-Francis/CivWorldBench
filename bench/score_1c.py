"""1c component scoring (09 §3.3): parse a bench report into structured pieces and
score each fine-grained component with the metric that fits its type, against the
truth report, with copy-forward (the t1 report) as the floor.

Components (per civ, then averaged):
  techs_new       F1 of the predicted NEW tech set vs the true new set (relative to t1)
  unit_mix        L1/2 distance between normalised unit-type distributions (0 same .. 1)
  city_exist      Jaccard of surviving t1 city names; new-city count MAE
  city_size       MAE of size over cities matched by name
  city_prod       accuracy of the production item over matched cities
  city_bldg_new   per-city F1 of newly added buildings (relative to t1), mean over cities
  stock           MAE of food and shield stock over matched cities
  dipl            accuracy of the relation state per directed pair, split by the t1 state
                  (transitional Cease-fire/Armistice/Peace vs stable War/Alliance)
Skill (lower-better metrics) = 1 - model/copyforward; for F1/accuracy/Jaccard the
improvement over copy-forward is reported alongside the raw value.
"""
from __future__ import annotations

import re
from collections import Counter

_CIV = re.compile(r"^(?P<name>\S.*?) of the (?P<nation>.+)$")
_TECHS = re.compile(r"^\s*technologies \((?P<n>\d+)\):\s*(?P<list>.*)$")
_UNITS = re.compile(r"^\s*units \((?P<n>\d+)\):\s*(?P<list>.*)$")
_CITY = re.compile(r"^\s{4}(?P<name>[^|]+?)\s*\|\s*(?P<xy>[^|]*)\|\s*(?P<size>-?\d+)\s*\|\s*(?P<founded>[^|]*)\|\s*"
                   r"(?P<prod>[^|]*)\|\s*(?P<food>-?\d+)\s*\|\s*(?P<shields>-?\d+)\s*\|\s*(?P<bld>[^|]*)(\|.*)?$")
_DIPL = re.compile(r"^- (?P<a>[^:]+?) -> (?P<rest>.+)$")


def parse_report(text: str, viewer: str | None = None) -> dict:
    """`viewer` is needed for view-2 (stale) reports, whose diplomacy lines read
    `- X: <state>; embassy; ...` from the viewer's side instead of `- A -> X: s; ...`."""
    civs: dict[str, dict] = {}
    dipl: dict[tuple[str, str], str] = {}
    cur = None; section = None
    for raw in text.split("\n"):
        line = raw.rstrip()
        s = line.strip()
        if s.startswith("### CIVILIZATIONS"):
            section = "civ"; continue
        if s.startswith("### DIPLOMATIC"):
            section = "dipl"; continue
        if s.startswith("### THE MAP") or s.startswith("=====") or s.startswith("#####"):
            section = None; continue
        if section == "civ":
            if s in ("-- mine --", "-- foreign (what I can observe) --"):
                continue
            m = _CIV.match(line)
            if m and not line.startswith(" ") and "|" not in line:
                cur = m["name"].strip(); civs[cur] = {"techs": set(), "units": Counter(), "cities": {}}
                continue
            if cur is None:
                continue
            m = _TECHS.match(line)
            if m:
                civs[cur]["techs"] = {x.strip() for x in m["list"].split(",") if x.strip() and x.strip() != "none"}
                continue
            m = _UNITS.match(line)
            if m:
                c = Counter()
                for part in m["list"].split(","):
                    mm = re.match(r"^\s*(?P<t>.+?)\s+x(?P<k>\d+)\s*$", part)
                    if mm:
                        c[mm["t"].strip()] += int(mm["k"])
                civs[cur]["units"] = c
                continue
            m = _CITY.match(line)
            if m and m["name"].strip() != "city":
                bld = {x.strip() for x in m["bld"].split(";") if x.strip() and x.strip() != "none"}
                civs[cur]["cities"][m["name"].strip()] = {"size": int(m["size"]), "prod": m["prod"].strip(),
                                                          "food": int(m["food"]), "shields": int(m["shields"]), "bld": bld}
        elif section == "dipl":
            m = _DIPL.match(s)
            if m:
                for part in m["rest"].split(";"):
                    if ":" in part:
                        b, st = part.rsplit(":", 1)
                        dipl[(m["a"].strip(), b.strip())] = st.strip()
            elif s.startswith("- ") and ":" in s:
                other, rest = s[2:].split(":", 1)
                dipl[(viewer or "me", other.strip())] = rest.split(";")[0].strip()
    return {"civs": civs, "dipl": dipl}


def _f1(pred: set, truth: set) -> float | None:
    if not pred and not truth:
        return None                       # nothing to predict, nothing predicted: not scored
    tp = len(pred & truth)
    if tp == 0:
        return 0.0
    p = tp / len(pred); r = tp / len(truth)
    return 2 * p * r / (p + r)


def _l1half(a: Counter, b: Counter) -> float:
    na, nb = sum(a.values()), sum(b.values())
    if na == 0 and nb == 0:
        return 0.0
    keys = set(a) | set(b)
    return 0.5 * sum(abs((a[k] / na if na else 0) - (b[k] / nb if nb else 0)) for k in keys)


def components(pred_text: str, truth_text: str, base_text: str, viewer: str | None = None) -> dict:
    """Per-component metrics for one report.  `base_text` is the t1 report (copy-forward
    floor and the reference for 'new' sets).  Returns {component: {"model": x, "cf": y,
    "n": count}} with None where the component is undefined."""
    P, Tt, B = parse_report(pred_text, viewer), parse_report(truth_text, viewer), parse_report(base_text, viewer)
    out = {}

    def add(k, model, cf, n):
        out[k] = {"model": model, "cf": cf, "n": n}

    civs = [c for c in Tt["civs"] if c in B["civs"]]
    if viewer is not None:
        civs = [c for c in civs if c == viewer]
    # techs_new
    f1_m, f1_c, n = [], [], 0
    for c in civs:
        new_t = Tt["civs"][c]["techs"] - B["civs"][c]["techs"]
        new_p = (P["civs"].get(c, {}).get("techs", set()) or set()) - B["civs"][c]["techs"]
        fm = _f1(new_p, new_t); fc = _f1(set(), new_t)
        if fm is not None:
            f1_m.append(fm); f1_c.append(fc if fc is not None else 0.0); n += 1
    add("techs_new", _mean(f1_m), _mean(f1_c), n)
    # unit_mix
    dm, dc = [], []
    for c in civs:
        tu = Tt["civs"][c]["units"]
        if sum(tu.values()) == 0:
            continue
        dm.append(_l1half(P["civs"].get(c, {}).get("units", Counter()) or Counter(), tu)); dc.append(_l1half(B["civs"][c]["units"], tu))
    add("unit_mix", _mean(dm), _mean(dc), len(dm))
    # cities
    jm, jc, mae_m, mae_c, acc_m, acc_c, bf_m, bf_c, st_m, st_c, new_m, new_c = ([] for _ in range(12))
    for c in civs:
        bc, tc, pc = B["civs"][c]["cities"], Tt["civs"][c]["cities"], P["civs"].get(c, {}).get("cities", {}) or {}
        base_names = set(bc)
        surv_t = set(tc) & base_names; surv_p = set(pc) & base_names
        if base_names:
            jm.append(_jacc(surv_p, surv_t)); jc.append(_jacc(base_names, surv_t))
        new_t = len(set(tc) - base_names); new_p = len(set(pc) - base_names)
        new_m.append(abs(new_p - new_t)); new_c.append(abs(0 - new_t))
        for name in surv_t:
            t = tc[name]; b = bc[name]; p = pc.get(name)
            mae_c.append(abs(b["size"] - t["size"])); acc_c.append(float(b["prod"] == t["prod"]))
            st_c.append((abs(b["food"] - t["food"]) + abs(b["shields"] - t["shields"])) / 2)
            newb_t = t["bld"] - b["bld"]
            f = _f1(set(), newb_t)
            if f is not None:
                bf_c.append(f)
            if p is None:                       # city missing from the prediction: worst case
                mae_m.append(abs(t["size"]) if t["size"] else 1); acc_m.append(0.0)
                st_m.append((abs(t["food"]) + abs(t["shields"])) / 2)
                if f is not None:
                    bf_m.append(0.0)
            else:
                mae_m.append(abs(p["size"] - t["size"])); acc_m.append(float(p["prod"] == t["prod"]))
                st_m.append((abs(p["food"] - t["food"]) + abs(p["shields"] - t["shields"])) / 2)
                fm = _f1(p["bld"] - b["bld"], newb_t)
                if f is not None:
                    bf_m.append(fm if fm is not None else 0.0)
    add("city_exist", _mean(jm), _mean(jc), len(jm))
    add("city_new_mae", _mean(new_m), _mean(new_c), len(new_m))
    add("city_size", _mean(mae_m), _mean(mae_c), len(mae_m))
    add("city_prod", _mean(acc_m), _mean(acc_c), len(acc_m))
    add("city_bldg_new", _mean(bf_m), _mean(bf_c), len(bf_m))
    add("stock", _mean(st_m), _mean(st_c), len(st_m))
    # diplomacy by t1 state
    tr_m, tr_c, sb_m, sb_c = [], [], [], []
    for pair, st_t in Tt["dipl"].items():
        st_b = B["dipl"].get(pair)
        if st_b is None:
            continue
        if viewer is not None and viewer not in pair:
            continue
        st_p = P["dipl"].get(pair)
        hit_m = float(st_p == st_t); hit_c = float(st_b == st_t)
        if st_b in ("Cease-fire", "Armistice", "Peace"):
            tr_m.append(hit_m); tr_c.append(hit_c)
        else:
            sb_m.append(hit_m); sb_c.append(hit_c)
    add("dipl_transitional", _mean(tr_m), _mean(tr_c), len(tr_m))
    add("dipl_stable", _mean(sb_m), _mean(sb_c), len(sb_m))
    out["_parsed_civs"] = len(P["civs"])
    return out


LOWER_BETTER = {"unit_mix", "city_new_mae", "city_size", "stock"}
HEADLINE = ("techs_new", "unit_mix", "city_exist", "city_size", "city_prod", "city_bldg_new", "dipl_transitional")


def skill(comp: str, model: float | None, cf: float | None) -> float | None:
    if model is None or cf is None:
        return None
    if comp in LOWER_BETTER:
        return 1 - model / cf if cf > 0 else (0.0 if model == 0 else -1.0)
    return model - cf                      # improvement over copy-forward for F1 / acc / Jaccard


def _mean(xs):
    return sum(xs) / len(xs) if xs else None


def _jacc(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a | b) else 1.0
