"""Process-correctness (CoT) lens -- frozen schema of *checkable intermediate claims*.

Doc: bundle/docs/freeciv/16_过程正确性（CoT）透镜：设计与实测.md (12 §5).

Idea: when the model answers a prediction question (1a / 1b / task 3) it must first EXTERNALISE the
intermediate predictions its answer rests on, in a closed vocabulary whose every line has a truth
value in the game record (mainline: every turn; arms: every 10 turns).  The same normalisation and
references as 1a (norms.json, log1p, trend / persistence) score the lines, so "process accuracy"
and "answer accuracy" are on one scale and the answer-right/evidence-wrong quadrant ("lucky") is
measurable.

Three externalisation formats, same questions, same truth:
  free   the frozen v3 prompt, thinking on, the <think> block is KEPT (form A; claims extracted later)
  table  EVIDENCE lines then the ANSWER block (form B, this module's grammar)
  code   one Python function `predict(world)` executed by us; its trajectory = evidence lines,
         its quantiles = the 1a answer (form C, 1a-type prompts only)

Evidence line grammar (fields separated by `|`; turn = t1 + 10k, t1 < turn <= t1 + hmax):
  V | <civ> | <quantity> | <turn> | <value>          one of the 11 task quantities at that turn
  D | <civ A> -> <civ B> | <turn> | <relation>       A's diplomatic state towards B at that turn
  G | <civ> | <turn> | <government>
  T | <civ> | <turn> | <technology>                  civ knows the technology at that turn
  O | <civ> | <turn> | <city>                        civ owns the city at that turn
The letter Q is deliberately NOT used (1b answers are `Q<n> <p>` lines).
"""
from __future__ import annotations

import hashlib
import re
import unicodedata

from bench.spec_v2 import GLOSSARY, VARS

STEP = 10
KINDS = ("V", "D", "G", "T", "O", "W")          # W (wonder completed) added in v3.1, 2026-09-16
PROTOCOL_VERSION = "3.1"
RELATIONS = ("War", "Cease-fire", "Armistice", "Peace", "Alliance", "No Contact", "Never met", "Team")
GOVERNMENTS = ("Anarchy", "Tribal", "Despotism", "Monarchy", "Communism", "Fundamentalism",
               "Federation", "Republic", "Democracy")
LEVEL_KEYS = ("gold", "population", "score", "culture", "n_cities", "city_size_total", "n_techs",
              "n_military_units", "n_civilian_units", "n_enemies", "n_allies")
WONDERS = ("Apollo Program", "A.Smith's Trading Co.", "Colossus", "Copernicus' Observatory", "Cure For Cancer", "Darwin's Voyage",
           "Eiffel Tower", "Great Library", "Great Wall", "Hanging Gardens", "Hoover Dam", "Isaac Newton's College", "J.S. Bach's Cathedral",
           "King Richard's Crusade", "Leonardo's Workshop", "Lighthouse", "Magellan's Expedition", "Manhattan Project", "Marco Polo's Embassy",
           "Michelangelo's Chapel", "Oracle", "Temple of Artemis", "Pyramids", "Shakespeare's Theatre", "Statue of Liberty", "Statue of Zeus",
           "Sun Tzu's War Academy", "United Nations", "Women's Suffrage", "Mausoleum of Mausolos", "Internet")
EVIDENCE_HEADER = "=====  EVIDENCE  ====="
ANSWER_HEADER = "=====  ANSWER  ====="
MIN_LINES, MAX_LINES = 8, 40
EXTRA_TOKENS_TABLE = 1600          # 40 lines x ~25 tokens + headers, on top of the answer budget
EXTRA_TOKENS_CODE = 6000           # a program is longer than the table it replaces


def evidence_turns(t1: int, hmax: int) -> list[int]:
    return [t1 + STEP * k for k in range(1, hmax // STEP + 1)]


def norm_name(s: str) -> str:
    """Civ / city / tech names: casefold, strip accents and quotes, collapse spaces."""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", s.strip().strip("`*\"'").casefold())


# ----------------------------------------------------------------- prompts --
def protocol_table(t1: int, hmax: int, civs: list[str], has_glossary: bool, counterfactual: bool,
                   per_turn: bool = True, answer_hz: list[int] | None = None) -> str:
    """v3.1 (per_turn=True, 2026-09-16): any turn in (t1, t1+hmax] is allowed -- the mainline record is
    per turn and, since plan B, so are the arm saves; a W line (wonder completed) is added.  per_turn=False
    reproduces the v3.0 grid protocol of cot-pilot / cot-main-*."""
    turns = evidence_turns(t1, hmax)
    if per_turn:
        tl = f"any turn from {t1 + 1} to {t1 + hmax}"
    else:
        tl = ", ".join(str(t) for t in turns) if len(turns) <= 8 else f"{turns[0]}, {turns[1]}, ..., {turns[-1]}"
    world = "this counterfactual world" if counterfactual else "this world"
    gl = "" if has_glossary else ("\nQuantities for V lines (same definitions as in the prediction tasks):\n"
                                 + GLOSSARY.replace("{t1}", str(t1)) + "\n")
    return f"""=====  OUTPUT PROTOCOL: evidence first, then the answer  =====
Before the answer, write the intermediate predictions your answer relies on. Each evidence line is one
claim about {world} at a turn AFTER turn {t1}, and every line will be checked against the actual game
record. Allowed turns: {tl}{"" if per_turn else f" (turn {t1} plus a multiple of 10, up to turn {t1 + hmax})"}.
Use ONLY these {"six" if per_turn else "five"} line forms, one claim per line, fields separated by ` | `:
  V | <civilization> | <quantity> | <turn> | <value>          one of the {len(VARS)} quantities ({', '.join(VARS)}) at that turn, same definitions as in the task; a single number
  D | <civilization A> -> <civilization B> | <turn> | <relation>   A's diplomatic state towards B at that turn: one of War, Cease-fire, Armistice, Peace, Alliance
  G | <civilization> | <turn> | <government>                  the form of government at that turn
  T | <civilization> | <turn> | <technology>                  the civilization knows that technology at that turn
  O | <civilization> | <turn> | <city>                        the civilization owns that city at that turn
{"  W | <civilization> | <turn> | <wonder>                      the civilization has completed that great wonder by that turn" + chr(10) if per_turn else ""}{gl}Write between {MIN_LINES} and {MAX_LINES} evidence lines. Prefer the claims that actually decide your answer:
turning points (wars and peace, conquests and lost cities, changes of government, key technologies) and
the intermediate levels of the quantities that move most. Do not restate the turn-{t1} observations:
every line must be about a later turn.{(" V lines are for INTERMEDIATE turns: do not put a V line at turn " + " or ".join(str(t1 + h) for h in answer_hz) + " -- those cells are already your answer below.") if (per_turn and answer_hz) else ""} Civilizations ({len(civs)}): {', '.join(civs)}.

Then write the answer exactly as specified in the task above. Output format:
{EVIDENCE_HEADER}
<evidence lines>
{ANSWER_HEADER}
<the answer lines exactly as specified above, nothing else>"""


def protocol_code(t1: int, hmax: int, hz: list[int], civs: list[str], frames: list[int], ruleset: str,
                  counterfactual: bool) -> str:
    turns = evidence_turns(t1, hmax)
    world = "the counterfactual world (the intervention is already applied in `world`)" if counterfactual else "this world"
    return f"""=====  OUTPUT PROTOCOL: a program instead of numbers  =====
Do not write the numbers directly. Write ONE Python 3 function `predict(world)` that computes the
prediction for {world} from a structured copy of the observations above. We will execute it
(standard library only, no network, no files, must finish within 20 seconds) and score its outputs
exactly like a direct answer. Its intermediate trajectory is checked against the actual game record.

`world` is a dict:
  world["turn"]      = {t1}
  world["horizons"]  = {hz}
  world["ruleset"]   = "{ruleset}"
  world["civs"]      = {civs}
  world["frames"]    = {{turn: {{civ: state}}}} for the observation turns {frames}
      state = {{"gold": int, "population": int, "score": int, "culture": int, "n_cities": int,
               "city_size_total": int, "n_techs": int, "n_military_units": int, "n_civilian_units": int,
               "n_enemies": int, "n_allies": int, "government": str,
               "relations": {{other civilization: "War"|"Cease-fire"|"Armistice"|"Peace"|"Alliance"}},
               "techs": [technology names], "cities": [{{"name": str, "size": int}}]}}
      (levels, exactly as printed in the frames; `score`, `culture`, `n_techs` are the running totals)

`predict(world)` must return a dict with two keys:
  "trajectory": {{turn: {{civ: {{"gold", "population", "score", "culture", "n_cities", "city_size_total",
                 "n_techs", "n_military_units", "n_civilian_units", "n_enemies", "n_allies",
                 "government", "relations": {{other: relation}}}}}}}}
                 for every turn in {turns} -- LEVELS at that turn (running totals for score / culture / n_techs)
  "quantiles":  {{civ: {{quantity: {{horizon: [p10, p25, p50, p75, p90]}}}}}} for the {len(VARS)} task quantities
                 ({', '.join(VARS)}) exactly as defined in the task (score_gain, culture_gain and
                 techs_learned are counted from turn {t1}), every civilization, every horizon in {hz}
Read the numbers from `world`; do not hard-code values that could be computed from it -- the same
function may be executed on other worlds and on this world with modified inputs. Keys may be ints or
strings. Output only the code, in one ```python fenced block, nothing else."""


EXTRA_TOKENS_CODE_V31 = 12000       # a turn-by-turn simulator is longer than the extrapolation programs of v3.0
N_ENSEMBLE = 50                    # runs per program; quantiles / probabilities are read off the ensemble
STATE_KEYS_OUT = LEVEL_KEYS + ("government", "relations", "cities", "wonders")


def protocol_code_v31(t1: int, hmax: int, hz: list[int], civs: list[str], frames: list[int], ruleset: str,
                      counterfactual: bool, has_glossary: bool, task_kind: str) -> str:
    """v3.1 code protocol (2026-09-16): `predict(world, rng)` is a stochastic turn-by-turn simulator; the
    harness runs it N_ENSEMBLE times and reads quantiles (1a) / event frequencies (1b) off the ensemble.
    `world` is the FULL structured copy of the frames (cities, units by type, rates, research, techs,
    wonders), not the 11-quantity summary of v3.0.  task_kind: "1a" | "1b" (which answer the ensemble feeds)."""
    world = "the counterfactual world (the intervention is already applied in `world`)" if counterfactual else "this world"
    gl = "" if has_glossary else ("\nQuantity definitions (same as in the prediction tasks):\n" + GLOSSARY.replace("{t1}", str(t1)) + "\n")
    if task_kind == "1a":
        how = (f"For every civilization, quantity and horizon in {hz} we take the ensemble's 10/25/50/75/90th "
               "percentiles as your quantile answer (gains counted from turn {t1}).")
    else:
        how = ("For every yes/no question in the task we take the fraction of runs in which the event happens as "
               "your probability; for the leader question, the fraction of runs in which each civilization has the "
               "highest score at that turn.  A program that never uses `rng` can only answer 0 or 1.")
    return f"""=====  OUTPUT PROTOCOL: a simulator instead of numbers  =====
Do not write the numbers directly. Write a Python 3 program that defines `predict(world, rng)` and SIMULATES
{world} forward turn by turn, from turn {t1} to turn {t1 + hmax}, returning the state of every civilization at
every turn.  `rng` is a `random.Random` instance -- draw from it for every uncertain outcome (growth, battles,
diplomacy, research, disasters).  We execute the program {N_ENSEMBLE} times with different seeds; the ensemble
is scored exactly like a direct answer, and its turn-by-turn trajectory is checked against the actual game
record.  {how.replace("{t1}", str(t1))}

`world` is a dict:
  world["turn"] = {t1}; world["horizons"] = {hz}; world["ruleset"] = "{ruleset}"; world["civs"] = {civs}
  world["frames"] = {{turn: {{civ: state}}}} for the observation turns {frames} (exactly what the text above shows)
      state = {{"gold", "population", "score", "culture", "n_cities", "city_size_total", "n_techs",
               "n_military_units", "n_civilian_units", "n_enemies", "n_allies": int,
               "government": str, "relations": {{other civ: "War"|"Cease-fire"|"Armistice"|"Peace"|"Alliance"}},
               "research_now": str, "rates": {{"tax", "science", "luxury": percent}},
               "income": {{"trade", "production", "science_output": int per turn}},
               "techs": [known technologies], "wonders": [great wonders owned],
               "cities": [{{"name", "size", "production", "food_stock", "shield_stock", "founded", "improvements": [...]}}],
               "units": {{unit type: count}}}}
{('  world["intervention"] = the intervention text' + chr(10)) if counterfactual else ""}
`predict(world, rng)` must return {{"trajectory": {{turn: {{civ: state}}}}}} for EVERY turn from {t1 + 1} to {t1 + hmax},
where state has at least: {", ".join(LEVEL_KEYS)} (levels / running totals at that turn),
"government", "relations" ({{other: relation}}), "cities" (the list of city NAMES owned at that turn: start from the
turn-{t1} names, drop a name when the city is lost, add the name when a city is founded or captured), "wonders"
(great wonder names owned).  Optionally add "events": a list of the events of THAT turn for that civilization,
using only these words: founded_city, captured_city, lost_city, completed_wonder, changed_government,
new_alliance, declared_war, made_peace.  A civilization that is destroyed may be omitted from later turns.
{gl}
CALIBRATE BEFORE YOU EXTRAPOLATE.  The observation frames are data: derive your rates and coefficients from them
(gold change per turn, research speed, growth of population and city sizes, unit build and loss rates, culture and
score per turn, all per civilization) instead of inventing constants, and make your mechanisms reproduce the
observed movement between the frames before they are used to go beyond turn {t1}.  We test this: the program is
also started at an EARLIER observed turn with only the frames up to that turn, and its prediction of the last
frame is compared with what was actually observed; large disagreements are reported back to you.  The program
must therefore work with any number of frames (one or more) -- take the latest frame in `world["frames"]` as the
start state and use the earlier ones, when present, for calibration.

The spread between runs must reflect your real uncertainty: at each horizon the 10th-90th percentile range of the
ensemble should be as wide as the interval you would give if you answered directly, not a few percent around one
path.  Runs that all follow the same trajectory are over-confident and are scored as such.

Rules: standard library only, no files, no network; one call of predict must finish within 2 seconds.  Read the
numbers from `world`; do not hard-code values that could be computed from it -- the same program will be run on
modified inputs.  Keys may be ints or strings.  Output only the code, in one ```python fenced block, nothing
else.  If the program fails, its output is incomplete, or its back-test disagrees badly with the observed frames,
you will get the details back and may fix it."""


# ------------------------------------------------------------------ parser --
_LINE = re.compile(r"^\s*(?P<k>[VDGTOW])\s*\|\s*(?P<rest>.+?)\s*$")
_TURN = re.compile(r"^(?:turn\s*)?\+?(\d+)$", re.I)


def split_sections(text: str, flexible: bool = False) -> tuple[str, str]:
    """-> (evidence text, answer text).  Tolerates missing headers: without an EVIDENCE header the
    evidence part is empty; without an ANSWER header everything after the evidence lines is the answer."""
    if flexible:
        text = re.sub(r"(?m)^[ \t]*={3,}[ \t]*EVIDENCE[ \t]*={3,}[ \t]*$", EVIDENCE_HEADER, text)
        text = re.sub(r"(?m)^[ \t]*={3,}[ \t]*ANSWER[ \t]*={3,}[ \t]*$", ANSWER_HEADER, text)
    ev_i = text.find(EVIDENCE_HEADER); an_i = text.find(ANSWER_HEADER)
    if ev_i >= 0 and an_i > ev_i:
        return text[ev_i + len(EVIDENCE_HEADER):an_i], text[an_i + len(ANSWER_HEADER):]
    if an_i >= 0:
        return text[:an_i], text[an_i + len(ANSWER_HEADER):]
    if ev_i >= 0:
        body = text[ev_i + len(EVIDENCE_HEADER):]
        lines = body.split("\n"); ev, rest, in_ev = [], [], True
        for l in lines:
            if in_ev and (_LINE.match(l) or not l.strip()):
                ev.append(l)
            else:
                in_ev = False; rest.append(l)
        return "\n".join(ev), "\n".join(rest)
    return "", text


def _turn(s: str):
    m = _TURN.match(s.strip())
    return int(m.group(1)) if m else None


def _num(s: str):
    s = s.strip().replace(",", "")
    m = re.match(r"^[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", s)
    try:
        return float(m.group(0)) if m else None
    except ValueError:
        return None


def parse_evidence(text: str, civs: list[str]) -> dict:
    """-> {"claims": [...], "n_lines": int, "n_bad": int, "bad": [...]}.

    Each claim: {"kind", "civ", "turn", ...kind fields..., "raw"}.  Civ names are matched against
    `civs` after norm_name(); a line naming an unknown civ, an unknown quantity, a malformed turn or
    a non-numeric value is a bad line (kept in `bad` for the census, not scored)."""
    cmap = {norm_name(c): c for c in civs}
    claims, bad, n = [], [], 0
    for raw in text.split("\n"):
        if not raw.strip():
            continue
        m = _LINE.match(raw)
        if not m:
            if "|" in raw:                       # looks like an attempt
                n += 1; bad.append({"raw": raw, "why": "no kind letter"})
            continue
        n += 1
        k = m["k"]; f = [x.strip() for x in m["rest"].split("|")]
        try:
            if k == "V":
                if len(f) != 4:
                    raise ValueError("V needs 4 fields")
                if _turn(f[1]) is not None and f[2].strip().lower() in VARS:
                    f[1], f[2] = f[2], f[1]                      # `V | civ | turn | quantity | value` (10 lines/sheet in review)
                if _turn(f[2]) is None and _turn(f[3]) is not None and _num(f[2]) is not None:
                    f[2], f[3] = f[3], f[2]                      # review 2026-09-16: value/turn swapped, keep the claim
                civ = cmap.get(norm_name(f[0])); var = f[1].strip().lower(); t = _turn(f[2]); v = _num(f[3])
                if civ is None:
                    raise ValueError("unknown civ")
                if var not in VARS:
                    raise ValueError("unknown quantity")
                if t is None or v is None:
                    raise ValueError("bad turn/value")
                claims.append({"kind": "V", "civ": civ, "var": var, "turn": t, "value": v, "raw": raw})
            elif k == "D":
                if len(f) != 3 or "->" not in f[0]:
                    raise ValueError("D needs `A -> B | turn | relation`")
                a, b = [x.strip() for x in f[0].split("->", 1)]
                if _turn(f[1]) is None and _turn(f[2]) is not None:
                    f[1], f[2] = f[2], f[1]                      # relation/turn swapped
                ca, cb = cmap.get(norm_name(a)), cmap.get(norm_name(b)); t = _turn(f[1])
                rel = next((r for r in RELATIONS if r.casefold() == f[2].strip().casefold().replace("ceasefire", "cease-fire")), None)
                if ca is None or cb is None:
                    raise ValueError("unknown civ")
                if t is None or rel is None:
                    raise ValueError("bad turn/relation")
                claims.append({"kind": "D", "civ": ca, "other": cb, "turn": t, "value": rel, "raw": raw})
            elif k in ("G", "T", "O", "W"):
                if len(f) != 3:
                    raise ValueError(f"{k} needs 3 fields")
                if _turn(f[1]) is None and _turn(f[2]) is not None:
                    f[1], f[2] = f[2], f[1]                      # review 2026-09-16: `O | civ | city | turn` -- the model's
                                                                 # conquest/tech turning points were being dropped as bad lines
                civ = cmap.get(norm_name(f[0])); t = _turn(f[1]); val = f[2].strip().strip("`*\"")
                if len(val) > 1 and val[0] == val[-1] == "'":
                    val = val[1:-1]                              # only symmetric quoting; `Waka'` is a real city name
                if civ is None:
                    raise ValueError("unknown civ")
                if t is None or not val:
                    raise ValueError("bad turn/value")
                if k == "G":
                    g = next((x for x in GOVERNMENTS if x.casefold() == val.casefold()), None)
                    if g is None:
                        raise ValueError("unknown government")
                    val = g
                claims.append({"kind": k, "civ": civ, "turn": t, "value": val, "raw": raw})
        except ValueError as e:
            bad.append({"raw": raw, "why": str(e)})
    return {"claims": claims, "n_lines": n, "n_bad": len(bad), "bad": bad}


_CODE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)


def extract_code(text: str) -> str | None:
    blocks = _CODE.findall(text)
    if blocks:
        return max(blocks, key=len)
    if "def predict" in text:
        return text[text.find("def predict"):]
    return None


def digest() -> str:
    import pathlib
    return hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()[:16]
