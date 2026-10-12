"""v2 question templates.  Frozen once the benchmark is published (digest in manifest).

Same three discipline rules as v1 (bench/tasks.py): every quantity is defined, the
number of output lines is stated, and no turn outside the requested horizons and
no world_id is ever mentioned.  v2 asked the FULL 1a/1b/1c set under an intervention in one prompt; v3 (2026-09-14/15,
14 §3) splits task 3 into one 1a prompt and one 1b prompt per arm plus a same-format
no-intervention control, asks only in-band horizons, drops 1c, turns the personality
question into a within-world ranking and gates the world-rule questions on evidence.
"""
from __future__ import annotations

from bench.spec_v2 import GLOSSARY, PARAM_DISPLAY, PARAM_LABELS, TRAIT_GLOSS, TRAITS, VARS


def q_1a(civs: list[str], t1: int, hz: list[int]) -> str:
    n = len(civs) * len(VARS) * len(hz)
    turns = ", ".join(f"turn {t1 + h} (+{h})" for h in hz)
    return f"""=====  TASK 1A: predict future quantities  =====
Predict the value of each quantity below at each of these future turns: {turns}.

{GLOSSARY.replace("{t1}", str(t1))}

You are uncertain, so express each prediction as five quantiles.
Output exactly one line per (civilization, quantity, horizon), in this format:
  <civilization> | <quantity> | +<horizon> | p10 | p25 | p50 | p75 | p90

Quantities ({len(VARS)}): {', '.join(VARS)}
Civilizations ({len(civs)}): {', '.join(civs)}
Horizons ({len(hz)}): {', '.join('+' + str(h) for h in hz)}

Output exactly {n} lines and nothing else. Numbers only, no units, no commas inside numbers."""


def q_1b(items: list[dict]) -> str:
    lines = []
    for i, it in enumerate(items, 1):
        if it["kind"] == "leader":
            lines.append(f"Q{i}  {it['text']}  Civilizations: {', '.join(it['civs'])}.")
        else:
            lines.append(f"Q{i}  {it['text']}")
    n_multi = sum(1 for it in items if it["kind"] == "leader")
    fmt = ("For yes/no questions output one line `Q<n> <probability of YES>` with a number between "
           "0.00 and 1.00.\n")
    if n_multi:
        fmt += ("For 'which civilization' questions output one line `Q<n> <civilization>=<probability>; "
                "<civilization>=<probability>; ...` covering every listed civilization, probabilities "
                "summing to 1.\n")
    return ("=====  TASK 1B: event probabilities  =====\n" + fmt +
            "\n" + "\n".join(lines) + f"\n\nOutput exactly {len(items)} lines and nothing else.")


def q_2(civs: list[str], pairs: list[tuple[str, str]], params: list[str], t1: int) -> str:
    """civs: the rankable civilizations (>= 2, else the personality block is omitted)."""
    blocks, n_lines, letters = [], 0, iter("ABC")
    if len(civs) >= 2:
        gl = "; ".join(f"{d} ({TRAIT_GLOSS[d]})" for d in TRAITS)
        blocks.append(f"""({next(letters)}) PERSONALITY. Each civilization's AI has four fixed personality values, each between
10 and 150 (50 is average): {gl}. For EACH of the four values, rank the {len(civs)} civilizations
listed below from highest to lowest. Output one line per personality value, in this format:
  <personality value>: <civilization> > <civilization> > ...
Write `=` instead of `>` between two civilizations you consider equal
(e.g. `<civilization> > <civilization> = <civilization> > ...`).
Every line must name all {len(civs)} civilizations exactly once.
Civilizations ({len(civs)}): {', '.join(civs)}
Personality values (4): {', '.join(TRAITS)}""")
        n_lines += 4
    if pairs:
        pair_lines = ", ".join(f"{a}->{b}" for a, b in pairs)      # every pair: the answer keys must be knowable (review run 2026-09-15)
        blocks.append(f"""({next(letters)}) ATTITUDE. Each AI keeps a hidden attitude score toward every civilization it has met
(from -1000 hostile to +1000 friendly). Classify the attitude of the FIRST civilization
toward the SECOND as hostile (<= -250), neutral or friendly (>= +250), one line per pair:
  <civilization A> -> <civilization B> | <hostile|neutral|friendly>
Pairs ({len(pairs)}): {pair_lines}""")
        n_lines += len(pairs)
    if params:
        plines = "\n".join(f"  {PARAM_DISPLAY[p]}: {' / '.join(PARAM_LABELS[p])}" for p in params)
        blocks.append(f"""({next(letters)}) WORLD RULES. This world was generated with hidden rule settings. Classify each:
{plines}
Output one line per setting, using the setting name exactly as written above:
  <setting> | <one of its three labels exactly as written above>""")
        n_lines += len(params)
    parts = []
    if len(civs) >= 2:
        parts.append("4 personality")
    if pairs:
        parts.append(f"{len(pairs)} attitude")
    if params:
        parts.append(f"{len(params)} rules")
    kinds = len(blocks)
    return (f"""=====  TASK 2: infer what is hidden  =====
Everything above is what can be observed up to turn {t1}. {'Three' if kinds == 3 else ('Two' if kinds == 2 else 'One')} kind{'s' if kinds > 1 else ''} of hidden state
drive{'s' if kinds == 1 else ''} this world; none of {'them' if kinds > 1 else 'it'} appears in any report.

""" + "\n\n".join(blocks) + f"""

Output exactly {n_lines} line{"s" if n_lines != 1 else ""} ({' + '.join(parts)}) and nothing else.""")


def q_3_preamble(intervention: str, t1: int) -> str:
    return f"""=====  COUNTERFACTUAL  =====
Suppose that at turn {t1}, immediately after the last observation frame, the following
intervention takes place and nothing else changes:
  {intervention}
Answer the task below for this counterfactual world. Predict everything, including
quantities and civilizations the intervention should NOT affect: in those cases your
answer should match what you would predict without the intervention."""
