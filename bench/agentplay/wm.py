"""World-model advice for the agent (arm "wm"): the same LLM, in the benchmark's forecasting role.

At a decision point the agent proposes candidate plans; for each candidate this module builds a CIVLADDER-format
1a prompt -- [K=3 observation frames as the ego sees them (view 2, POMDP), INTERVAL news between them,
a STRATEGIES block that states the candidate as the ego's own plan, the 1a question] -- exactly as
`bench/build_bench_v2.py` does for the frozen benchmark (same renderers: bench.render.build_H / interval_v2,
same glossary and answer format), asks the model for five quantiles per (civ, quantity, horizon), and turns the
answer into a short table the agent can compare.

Records for turns before the takeover come from the corpus (`world.jsonl.gz`); records for played turns are rebuilt
from the game's autosaves exactly as collect.py builds them (state / actions / geo / probes + merged engine & diff
events), so the frames and news are the real history of THIS game.
"""
from __future__ import annotations

import concurrent.futures as cf
import dataclasses
import gzip
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "bundle", ROOT / "tools", ROOT / "freeciv-sandbox"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from sandboxes.freeciv.actions import extract_actions                    # noqa: E402
from sandboxes.freeciv.events import diff_events                          # noqa: E402
from sandboxes.freeciv.event_stream import merge as merge_events          # noqa: E402
from sandboxes.freeciv.geo import extract_map, geo_to_json                # noqa: E402
from sandboxes.freeciv.probes import extract_probes                       # noqa: E402
from sandboxes.freeciv.rehydrate import state_from_json                  # noqa: E402
from sandboxes.freeciv.savegame import Savegame                           # noqa: E402
from sandboxes.freeciv.shard import read_text                             # noqa: E402
from sandboxes.freeciv.state import extract                               # noqa: E402
from sandboxes.freeciv.actions import to_json as actions_to_json          # noqa: E402

import bench.tasks_v2 as T                                                # noqa: E402
from bench.build_bench_v2 import events_of                                # noqa: E402
from bench.parse_v2 import parse_quantiles                                # noqa: E402
from bench.render import build_H                                          # noqa: E402
from bench.run_infer_v2 import SYSTEM, THINK_BUDGET                       # noqa: E402
from bench.spec_v2 import LEVEL_OF, VARS, frames_for, met_civs            # noqa: E402
from bench.strategy_v3 import HEADER, INTRO                               # noqa: E402

BENCH = ROOT / "data_v2/bench"
HZ = (20, 40)          # decision horizons for the advice (the frozen benchmark asks +60/+120 on these worlds)


class History:
    """Per-turn records of one game: corpus before the takeover, rebuilt from autosaves after it."""

    def __init__(self, world: str, game_dir: Path, start: int, ego_idx: int):
        self.world, self.game_dir, self.start, self.ego_idx = world, Path(game_dir), start, ego_idx
        self.meta = json.load(open(BENCH / world / "meta.json"))
        self.split = self.meta["config"].get("split", "bench")
        self.corpus: dict[int, dict] = {}
        with gzip.open(BENCH / world / "world.jsonl.gz", "rt") as f:
            for l in f:
                r = json.loads(l)
                if r["t"] < start:
                    self.corpus[r["t"]] = r
        self.built: dict[int, dict] = {}
        self._ws: dict[int, Any] = {}

    def _engine_events(self) -> dict[int, list[dict]]:
        out: dict[int, list[dict]] = {}
        p = self.game_dir / "events.jsonl"
        if p.exists():
            for l in open(p):
                r = json.loads(l)
                for e in r["events"]:
                    t = int(e.get("engine_turn") or e.get("turn") or r["turn"])
                    out.setdefault(t, []).append(e)
        return out

    def _save(self, t: int) -> Savegame:
        if t < self.start:
            zst = sorted((BENCH / self.world / "saves").glob(f"freeciv-T{t:04d}-*.sav.zst"))
            if not zst:
                raise FileNotFoundError(f"no corpus save for turn {t} in {self.world}")
            return Savegame(read_text(zst[0]).split("\n"), zst[0])
        if t == self.start:
            cand = sorted((self.game_dir / "saves").glob(f"T{t:04d}-takeover.sav"))
        else:
            cand = sorted((self.game_dir / "saves").glob(f"freeciv-T{t:04d}-*-auto.sav"))
        if not cand:
            raise FileNotFoundError(f"no save for turn {t}")
        return Savegame.load(cand[-1])

    def _ws_at(self, t: int):
        if t not in self._ws:
            sg = self._save(t); self._ws[t] = (sg, extract(sg, self.world))
        return self._ws[t]

    def record(self, t: int) -> dict:
        if t in self.corpus:
            return self.corpus[t]
        if t in self.built:
            return self.built[t]
        eng = self._engine_events()
        sg, ws = self._ws_at(t)
        try:
            _, prev_ws = self._ws_at(t - 1) if t - 1 >= 1 else (None, None)
        except FileNotFoundError:
            # Archived corpus states remain usable when their raw saves were not retained.
            if t - 1 not in self.corpus:
                raise
            prev_ws = state_from_json(self.corpus[t - 1]["state"])
        ev_diff = diff_events(prev_ws, ws) if prev_ws is not None else []
        ev = merge_events({t: ev_diff}, {t: eng.get(t, [])}).get(t, [])
        ws.events = ev
        mi = extract_map(sg, [q.name for q in ws.players])
        rec = {"world_id": self.world, "sim": "freeciv-3.1.6", "ruleset": sg.ruleset, "split": self.split, "t": t, "year": sg.year,
               "state": dataclasses.asdict(ws), "actions": actions_to_json(extract_actions(sg)), "geo": geo_to_json(mi),
               "events": ev, "probes": extract_probes(sg, mi)}
        self.built[t] = rec
        return rec

    def window(self, lo: int, hi: int) -> dict[int, dict]:
        return {t: self.record(t) for t in range(max(1, lo), hi + 1)}


def strategies_block(viewer: str, t1: int, name: str, plan: str) -> str:
    scope = f"(only {viewer}'s own plan is known)"
    lines = [HEADER.format(t1=t1, scope=scope), INTRO.format(who=viewer, t1=t1), "",
             f"{viewer} is set on this course: {name}."]
    for ln in plan.strip().splitlines():
        lines.append("  " + ln.strip())
    return "\n".join(lines) + "\n"


def wm_prompt(hist: History, t1: int, viewer: str, cand_name: str, cand_plan: str, hz=HZ) -> tuple[str, int, list[str]]:
    frames = frames_for(t1)
    recs = hist.window(frames[0], t1)
    ev = events_of(recs, frames[0], t1)
    world = dict(hist.meta["config"]); world.setdefault("ruleset", hist.meta["config"].get("ruleset", "civ2civ3"))
    H, _ = build_H(recs, frames, viewer, 2, ev, ruleset="", world=world)
    civs = met_civs(recs[t1], viewer)
    body = H + "\n" + strategies_block(viewer, t1, cand_name, cand_plan) + "\n" + T.q_1a(civs, t1, list(hz))
    max_tok = int(len(civs) * len(VARS) * len(hz) * 32 * 1.6)
    return body, max_tok, civs


def current_levels(rec: dict, civs: list[str]) -> dict[str, dict[str, float]]:
    """Level of every 1a variable at t1 for the *_gain conversions and for showing 'now' next to the forecast."""
    from bench.spec_v2 import derive           # the benchmark's own level extractor (truth_v2 uses it)
    from bench.truth_v2 import mil_of, player
    mil = mil_of(rec)
    out = {}
    for c in civs:
        p = player(rec, c)
        if p is not None:
            out[c] = derive(p, mil, rec)
    return out


def forecast_table(answer: str, viewer: str, civs: list[str], t1: int, hz, levels: dict[str, dict[str, float]] | None) -> str:
    q = parse_quantiles(answer)
    if not q:
        return "(the world model returned nothing parseable)"
    L = []
    show = ["score_gain", "n_cities", "city_size_total", "gold", "techs_learned", "culture_gain", "n_military_units", "n_enemies", "n_allies"]
    L.append(f"{viewer} (you)" + (f" -- now: " + ", ".join(f"{v} {levels[viewer].get(LEVEL_OF[v], '?'):g}" for v in ("n_cities", "city_size_total", "gold", "n_military_units") if levels and viewer in levels) if levels else ""))
    for v in show:
        cells = []
        for h in hz:
            k = (viewer, v, h)
            if k in q:
                p = q[k]
                cells.append(f"+{h}: {p[0]:g} / {p[2]:g} / {p[4]:g}")
        if cells:
            L.append(f"  {v:<18} " + "   ".join(cells) + "   (p10 / p50 / p90)")
    for c in civs:
        if c == viewer:
            continue
        cells = []
        for v in ("score_gain", "n_cities", "n_military_units", "n_enemies"):
            vals = [f"+{h} {q[(c, v, h)][2]:g}" for h in hz if (c, v, h) in q]
            if vals:
                cells.append(f"{v} " + ", ".join(vals))
        if cells:
            L.append(f"{c}: " + " · ".join(cells) + "  (p50)")
    return "\n".join(L)


class WorldModel:
    """The forecaster.  `model` "flash-next" -> the local vLLM server; "claude-*" / "gpt-*" -> the univ gateway
    (same request shapes as bench/run_infer_api.py: adaptive thinking + effort for Anthropic, reasoning_effort for OpenAI;
    the key is read from  and never logged)."""

    def __init__(self, endpoint: str = "http://localhost:8011/v1", model: str = "flash-next", effort: str = "medium", timeout: float = 1800.0):
        import httpx
        self.endpoint, self.model, self.effort = endpoint, model, effort
        self.api = "anthropic" if model.startswith("claude") else "openai" if model.startswith("gpt") else "local"
        headers = {}
        if self.api != "local":
            from bench.run_infer_api import GATEWAY, key_from
            self.endpoint = GATEWAY
            headers = {"Authorization": f"Bearer {key_from('')}", "Content-Type": "application/json"}
            if self.api == "anthropic":
                headers["anthropic-version"] = "2023-06-01"
        self.client = httpx.Client(timeout=httpx.Timeout(timeout, connect=30.0), headers=headers,
                                  base_url=(self.endpoint if self.api != "local" else ""))

    def call(self, prompt: str, max_tokens: int) -> dict[str, Any]:
        if self.api != "local":
            return self._call_api(prompt, max_tokens)
        body = {"model": self.model, "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
                "max_tokens": max_tokens + THINK_BUDGET, "temperature": 1.0, "top_p": 0.95, "top_k": 20,
                "chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": self.effort}}
        if self.model.lower().startswith("kimi"):
            # vLLM's Kimi API accepts low/high/max, unlike the bundled tokenizer.
            kimi_effort = "high" if self.effort == "medium" else self.effort
            body["chat_template_kwargs"] = {"thinking": True, "thinking_effort": kimi_effort}
        t0 = time.time()
        for attempt in range(3):
            try:
                r = self.client.post(self.endpoint + "/chat/completions", json=body)
                if r.status_code == 200:
                    js = r.json(); txt = js["choices"][0]["message"].get("content") or ""
                    ans = txt.rsplit("</think>", 1)[-1].strip()
                    u = js.get("usage", {})
                    return {"answer": ans, "prompt_tokens": u.get("prompt_tokens", 0), "completion_tokens": u.get("completion_tokens", 0),
                            "finish": js["choices"][0].get("finish_reason"), "seconds": time.time() - t0}
                err = f"http{r.status_code}: {r.text[:1000]}"
            except Exception as e:                 # noqa: BLE001
                err = f"{type(e).__name__}: {e}"
            time.sleep(5 * (attempt + 1))
        return {"answer": "", "prompt_tokens": 0, "completion_tokens": 0, "finish": f"error:{err}", "seconds": time.time() - t0}

    def _call_api(self, prompt: str, max_tokens: int) -> dict[str, Any]:
        import bench.run_infer_api as R
        R.EFFORT = self.effort
        path, body = R.build_request(self.api, self.model, {"prompt": prompt, "system": SYSTEM, "max_tokens": max_tokens})
        t0 = time.time(); err = ""
        for attempt in range(4):
            try:
                r = self.client.post(path, json=body)
                if r.status_code == 200:
                    o = R.parse_response(self.api, r.json())
                    return {"answer": o["text"], "prompt_tokens": o["n_prompt"], "completion_tokens": o["n_gen"], "finish": o["finish"], "seconds": time.time() - t0}
                err = f"http{r.status_code}:{R._scrub(r.text[:120])}"
                if r.status_code in (400, 401, 403, 404, 413, 422):
                    break
            except Exception as e:                 # noqa: BLE001
                err = f"{type(e).__name__}"
            time.sleep(10 * (attempt + 1))
        return {"answer": "", "prompt_tokens": 0, "completion_tokens": 0, "finish": f"error:{err}", "seconds": time.time() - t0}

    def advise(self, hist: History, t1: int, viewer: str, candidates: list[dict[str, str]], hz=HZ) -> dict[str, Any]:
        """candidates: [{"name", "plan"}] -> {"text": table for the agent, "calls": [...], "prompt_chars": ...}"""
        prompts = [wm_prompt(hist, t1, viewer, c["name"], c["plan"], hz) for c in candidates]
        rec = hist.record(t1); civs = prompts[0][2]
        try:
            levels = current_levels(rec, civs)
        except Exception:                          # noqa: BLE001
            levels = None
        with cf.ThreadPoolExecutor(max_workers=len(prompts)) as ex:
            outs = list(ex.map(lambda pm: self.call(pm[0], pm[1]), prompts))
        blocks = []
        for c, (body, _, civs_c), o in zip(candidates, prompts, outs):
            blocks.append(f"### Candidate: {c['name']}\n" + forecast_table(o["answer"], viewer, civs_c, t1, hz, levels))
        return {"text": "\n\n".join(blocks), "calls": [{**o, "answer": o["answer"][:4000]} for o in outs],
                "prompt_chars": [len(p[0]) for p in prompts], "civs": civs, "hz": list(hz)}


# ============================================================================================================
# v2 advice (2026-09-17 afternoon): what the pilots said was wrong with v1 and what changes here
#   * absolute quantiles per candidate were optimistic and not discriminative -> ONE prompt holds all candidates
#     and the model answers them side by side (relative judgement), sampled N times and aggregated (median);
#   * frames 20-40 turns apart hid the current dynamics -> dense recent frames (t1-20, t1-10, t1), horizons +10/+20
#     matching the 10-turn decision cadence;
#   * nobody ever told the model how its last forecast turned out -> a CALIBRATION appendix with the previous
#     decision points' p50 forecasts for the chosen plan next to what actually happened (also shown to the agent).
# ============================================================================================================
HZ2 = (10, 20)
FRAMES2 = (20, 10, 0)          # offsets before t1
LETTERS = "ABCDE"


def frames_dense(t1: int) -> list[int]:
    return [max(1, t1 - d) for d in FRAMES2]


def q_1a_multi(civs: list[str], t1: int, hz: list[int], letters: list[str]) -> str:
    base = T.q_1a(civs, t1, hz)
    n = len(civs) * len(VARS) * len(hz)
    head, _, tail = base.partition("You are uncertain")
    return (head + f"There are {len(letters)} candidate plans for {civs[0]} above (labelled {', '.join(letters)}).  The other civilizations do not "
            "know which one will be chosen.  Forecast the world under EACH candidate, side by side, and let the candidates differ only where "
            "the plan really makes a difference.\n\n"
            "You are uncertain" + tail.replace(f"Output exactly {n} lines and nothing else.",
            f"For each candidate output a header line `### <letter>` followed by exactly {n} lines in that format; "
            f"{len(letters)} sections, {len(letters) * n} data lines, nothing else."))


def strategies_multi(viewer: str, t1: int, candidates: list[dict], letters: list[str]) -> str:
    scope = f"(only {viewer}'s own plan is known; {len(candidates)} candidate plans are under consideration)"
    out = [HEADER.format(t1=t1, scope=scope), INTRO.format(who=viewer, t1=t1), ""]
    for L_, c in zip(letters, candidates):
        out.append(f"Candidate {L_} -- {viewer} would be set on this course: {c['name']}.")
        out += ["  " + ln.strip() for ln in c["plan"].strip().splitlines()]
        out.append("")
    return "\n".join(out)


def calibration_block(history: list[dict], levels_now: dict, viewer: str, t1: int) -> str:
    """history: [{turn, chosen, p50: {(var,h): value}, levels: {var: level at that turn}}] from earlier decision points."""
    if not history or viewer not in levels_now:
        return ""
    L = ["=====  CALIBRATION: earlier forecasts for the plan that was actually followed, next to what happened  =====",
         f"(read before forecasting; the columns are the p50 you gave then and the value observed at turn {t1})"]
    now = levels_now[viewer]
    for h_ in history[-3:]:
        dt = t1 - h_["turn"]
        if dt <= 0:
            continue
        cells = []
        for v in ("score_gain", "n_cities", "gold", "techs_learned", "n_military_units", "n_enemies"):
            p = h_["p50"].get((v, dt)) if (v, dt) in h_["p50"] else None
            if p is None:
                continue
            lvl = LEVEL_OF[v]
            actual = now.get(lvl)
            if v in ("score_gain", "techs_learned", "culture_gain") and actual is not None:
                actual = actual - h_["levels"].get(lvl, 0)
            if actual is not None:
                cells.append(f"{v} +{dt}: forecast {p:g}, actual {actual:g}")
        if cells:
            L.append(f"turn {h_['turn']} ({h_['chosen']}): " + "; ".join(cells))
    return "\n".join(L) + "\n" if len(L) > 2 else ""


def wm_prompt_multi(hist: History, t1: int, viewer: str, candidates: list[dict], history: list[dict], hz=HZ2) -> tuple[str, int, list[str], list[str]]:
    frames = frames_dense(t1)
    recs = hist.window(frames[0], t1)
    ev = events_of(recs, frames[0], t1)
    world = dict(hist.meta["config"]); world.setdefault("ruleset", hist.meta["config"].get("ruleset", "civ2civ3"))
    H, _ = build_H(recs, frames, viewer, 2, ev, ruleset="", world=world)
    civs = met_civs(recs[t1], viewer)
    letters = list(LETTERS[:len(candidates)])
    try:
        levels = current_levels(recs[t1], civs)
    except Exception:                              # noqa: BLE001
        levels = {}
    cal = calibration_block(history, levels, viewer, t1)
    body = H + "\n" + strategies_multi(viewer, t1, candidates, letters) + "\n" + (cal + "\n" if cal else "") + q_1a_multi(civs, t1, list(hz), letters)
    max_tok = int(len(candidates) * (len(civs) * len(VARS) * len(hz) * 32 * 1.6 + 40))
    return body, max_tok, civs, letters


def parse_multi(answer: str, letters: list[str]) -> dict[str, dict]:
    import re
    parts = re.split(r"^\s*#{1,4}\s*([A-E])\b.*$", answer, flags=re.M)
    out = {}
    # parts = [pre, letter, text, letter, text, ...]
    for i in range(1, len(parts) - 1, 2):
        L_ = parts[i].strip().upper()
        if L_ in letters:
            out[L_] = parse_quantiles(parts[i + 1])
    return out


def aggregate(samples: list[dict]) -> dict:
    """median across samples of each quantile, per key"""
    import statistics
    keys = set().union(*[s.keys() for s in samples]) if samples else set()
    out = {}
    for k in keys:
        vals = [s[k] for s in samples if k in s]
        if not vals:
            continue
        out[k] = tuple(statistics.median(v[i] for v in vals) for i in range(5))
    return out


def table_multi(fc: dict[str, dict], candidates: list[dict], letters: list[str], viewer: str, civs: list[str], hz, levels: dict) -> str:
    L = []
    now = levels.get(viewer, {})
    L.append(f"now: cities {now.get('n_cities', '?')}, city sizes total {now.get('city_size_total', '?')}, gold {now.get('gold', '?')}, "
             f"military units {now.get('n_military_units', '?')}, enemies {now.get('n_enemies', '?')}")
    hdr = "quantity (yours)     " + "".join(f"| {L_}: {c['name'][:22]:<22} " for L_, c in zip(letters, candidates))
    L.append(hdr)
    for v in ("score_gain", "n_cities", "city_size_total", "gold", "techs_learned", "n_military_units", "n_enemies"):
        for h in hz:
            row = f"{v:<14} +{h:<4} "
            for L_ in letters:
                q = fc.get(L_, {}).get((viewer, v, h))
                row += f"| {'' if q is None else f'{q[0]:g} / {q[2]:g} / {q[4]:g}':<26} "
            L.append(row)
    L.append("(cells are p10 / p50 / p90; *_gain = change from now)")
    others = [c for c in civs if c != viewer]
    for c in others:
        row = f"{c[:20]:<20} "
        for L_ in letters:
            q = fc.get(L_, {})
            mil = q.get((c, "n_military_units", hz[-1])); en = q.get((c, "n_enemies", hz[-1])); sg = q.get((c, "score_gain", hz[-1]))
            row += f"| mil {'' if mil is None else f'{mil[2]:g}'} enemies {'' if en is None else f'{en[2]:g}'} Δscore {'' if sg is None else f'{sg[2]:g}'}".ljust(29) + " "
        L.append(row)
    L.append(f"(other civilizations at +{hz[-1]}, p50)")
    # one-line reading per candidate
    reads = []
    for L_, c in zip(letters, candidates):
        q = fc.get(L_, {})
        sg = q.get((viewer, "score_gain", hz[-1])); nc = q.get((viewer, "n_cities", hz[-1]))
        risk = "city-loss risk" if nc is not None and now.get("n_cities") is not None and nc[1] < now["n_cities"] else "no city-loss signal"
        reads.append(f"{L_}: expected Δscore +{hz[-1]} = {'' if sg is None else f'{sg[2]:g}'} ({risk})")
    L.append("reading: " + " · ".join(reads))
    return "\n".join(L)


def advise_multi(self: "WorldModel", hist: History, t1: int, viewer: str, candidates: list[dict], history: list[dict],
                 n_samples: int = 3, hz=HZ2) -> dict[str, Any]:
    body, max_tok, civs, letters = wm_prompt_multi(hist, t1, viewer, candidates, history, hz)
    with cf.ThreadPoolExecutor(max_workers=n_samples) as ex:
        outs = list(ex.map(lambda _: self.call(body, max_tok), range(n_samples)))
    per = [parse_multi(o["answer"], letters) for o in outs]
    fc = {L_: aggregate([p[L_] for p in per if L_ in p and p[L_]]) for L_ in letters}
    try:
        levels = current_levels(hist.record(t1), civs)
    except Exception:                              # noqa: BLE001
        levels = {}
    text = table_multi(fc, candidates, letters, viewer, civs, hz, levels)
    n_ok = sum(1 for p in per for L_ in letters if p.get(L_))
    ego_p50 = {L_: {(v, h): fc[L_][(viewer, v, h)][2] for (cv, v, h) in fc[L_] if cv == viewer} for L_ in letters}
    ego_q = {L_: {f"{v}|{h}": list(fc[L_][(viewer, v, h)]) for (cv, v, h) in fc[L_] if cv == viewer} for L_ in letters}
    return {"text": text, "calls": [{**o, "answer": o["answer"][:6000]} for o in outs], "prompt_chars": [len(body)], "civs": civs,
            "hz": list(hz), "letters": letters, "ego_p50": ego_p50, "ego_q": ego_q, "levels_ego": levels.get(viewer, {}), "sections_parsed": n_ok,
            "sections_expected": n_samples * len(letters)}


WorldModel.advise_multi = advise_multi


def decision_as_plan(d: dict, ego_units: dict, rules) -> str:
    """Render one sampled agent decision (strategy note + this turn's orders) as plan lines for the STRATEGIES block.
    Only what a forecaster can use: the stated course, production changes, research, rates, government, treaties and the
    unit orders summarised by kind."""
    import collections
    plan = d["plan"]; obj = d.get("obj") or {}
    L = []
    note = (d.get("note") or "").strip().replace("\n", " ")
    if note:
        L.append("Course: " + note[:700])
    if plan.research_goal or plan.research_now:
        L.append("Research aim: " + " / ".join(x for x in (plan.research_now, plan.research_goal) if x))
    if plan.rates:
        L.append(f"Budget: tax {plan.rates[0]}% science {plan.rates[1]}% luxury {plan.rates[2]}%")
    if plan.government:
        L.append(f"Government: revolution to {plan.government}")
    if plan.treaties:
        L.append("Diplomacy: " + "; ".join(f"{t['type']} with {t['with']}" for t in plan.treaties))
    if plan.cities:
        L.append("Production this turn: " + "; ".join(f"{c.name} -> {c.build}" + (" (buy)" if c.buy else "") for c in plan.cities if c.build or c.buy))
    kinds = collections.Counter()
    for u in plan.units:
        ut = ego_units.get(u.uid, {}).get("type_by_name", "unit")
        kinds[f"{ut} {u.kind}" + (f" -> {u.target[0]},{u.target[1]}" if u.target and u.kind in ("attack", "conquer", "found_city", "move") else "")] += 1
    if kinds:
        L.append("Unit orders this turn: " + "; ".join(f"{k}" + (f" x{n}" if n > 1 else "") for k, n in kinds.most_common(12)))
    return "\n".join(L) if L else "Course: continue as before; no new orders this turn."


def utility(ego_q: dict, levels_now: dict, hz=HZ2, loss_penalty: float = 5.0) -> tuple[float, str]:
    """U = p50 score_gain(+H) - penalty * max(0, cities_now - p25 n_cities(+H)).  Returns (U, explanation)."""
    H = hz[-1]
    sg = ego_q.get(f"score_gain|{H}"); nc = ego_q.get(f"n_cities|{H}")
    if not sg:
        return float("-inf"), "no forecast"
    u = sg[2]; risk = 0.0
    now_c = levels_now.get("n_cities")
    if nc and now_c is not None:
        risk = max(0.0, now_c - nc[1])
        u -= loss_penalty * risk
    return u, f"p50 Δscore(+{H}) {sg[2]:g}" + (f", city-loss risk {risk:g}" if risk else "")
