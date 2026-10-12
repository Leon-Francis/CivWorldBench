# Freeciv 3.1 expert manual for the CIVLADDER human-analogue line

Version 3.3, 2026-09-21, for the "expert with tools" annotator line (one decision point per annotator).
Changes in 3.3, all measured on worlds outside this benchmark: §3 the granary table replaced by the true
foodbox-100 granary and a divide-and-compare rule (the old table used per-bin medians, which over-called
"cities grow slowly"); §10 disasters counted per 1,000 city-turns with the 0.85 / 1.65 cut-offs and the
zero-or-one-sentence rule; §14 task 2 now carries the numeric thresholds every world-rule question is decided
by and the corrected personality proxies (military-per-city is *below chance* for Aggressive; Settlers built
tells you nothing about Expansionist); §14 task 3 carries a measured table of what each intervention does,
split by ruleset, with a rule for when to move the median at all; §14 task 1b states how far to move from the
shown base rate and which six facts carry more or less weight than instinct suggests.
Changes in v3 from v2: §4 culture mechanics corrected (no interest in this ruleset; achievements; captured cities carry
their culture), Future Technology scoring; §5 research rewritten around the tech-count timelines (the bulbs ÷ cost
method under-predicts 3–5× early in a game); §3/§7 city capture frequencies; §13 measured 80% bands per quantity
and horizon; §14 the early-game acceleration rule, the interval-width rules and a mandatory self-check.
Scope: what a player who has watched several hundred AI-vs-AI Freeciv games — and read the AI's source —
knows about the rules, the default AI's decision logic, and the way these particular worlds behave. It contains
rules, mechanisms and behavioural regularities. It contains NO per-world truths, no benchmark statistics beyond
what every prompt already shows (the base rates), and nothing about how any model scored. Sources: the
civ2civ3 / classic ruleset primers (`ruleset_primer/RULES_*.md`), the Freeciv 3.1 server and `ai/default`
sources as audited in docs 09 and 13, the engine patches these worlds were run with, and the rendering code
that produced the prompts.

---

## 0. What these worlds are

- Freeciv 3.1.6, ruleset **civ2civ3** (most worlds) or **classic** (part of the generalisation set). Classic
  has no culture (culture stays 0 and `culture_gain` is not scored), a different technology tree (see the
  primer), no Migrants / Ecclesiastical Palace / Fanatics, and cheaper, Civ2-style tech costs early but steeper
  later. Everything else in this manual applies to both unless marked.
- All players are the stock **default AI** at one difficulty level per world (novice, easy, normal, hard or
  cheating). No human plays. Maps from 24×32 (3 civs, 400 turns) through 32×64 (6 civs, 800 turns) to 48×80
  (8 civs, 1200 turns) and odd sizes in the generalisation set (up to 10 civs); the game runs to its turn
  budget with no victory condition, so nobody "wins early" and wars never end the game.
- **Engine patches** in these worlds (they matter):
  1. Two civilizations that meet start at **Peace** (stock Freeciv starts at War).
  2. AI personality (`trait`, 10–150, default 50) is wired into behaviour: **Expansionist** scales the wish to
     build Settlers and found cities; **Trader** scales the wish to build Caravans and trade routes;
     **Aggressive** scales how fast the AI turns on a target and how easily it declares war; **Builder**
     discounts military production relative to buildings.
  3. Ceasefire is easier to reach (love threshold −600 instead of −400) and love recovers during ceasefire;
     grudges against a chosen war target accumulate only before the war starts and scale with Aggressive.
  4. National trade routes yield revenue, so caravans are built and sent to the civ's own cities too.
- **Hidden world parameters** (differ per world; task 2 asks for three-way bins of the first four):
  `disasters` (frequency of random disasters), `foodbox` (food needed to grow, 100 = normal),
  `sciencebox` (bulb cost of technologies, 100 = normal), `illness` (plague: off / normal / high).
  Also varied but not asked: barbarian onset turn and frequency, `razechance` (chance each building is lost
  when a city is captured), `civilwarsize` (cities needed before losing the capital can split the civ),
  `globalwarming_percent`. Migration is disabled. Some generalisation worlds are **team games**: teammates are
  in a permanent "Team" relation, share research and never fight; Team is not an alliance for `n_allies`.

## 1. The turn, and what the AI recomputes every turn

Each turn every city collects food, shields and trade from its worked tiles. Trade is split by the civ's
rates (tax → gold, luxury → happiness, science → bulbs; the maximum rate depends on government: 60% Tribal /
Despotism, 70% Monarchy / Fundamentalism, 80% Republic / Communism, 90% Democracy / Federation). Food surplus
fills the granary and the city grows when it is full; shields fill the production box; bulbs accumulate on the
research target. Units move, fight, fortify, build terrain improvements. Then the AI **re-plans from scratch**:
tax rates, every city's build order, settler targets, research goal, government, unit movements, and its
attitude to every other player. It has no long memory beyond a handful of counters (war countdown, ceasefire
timer, a few "ai love" values). Consequently AI trajectories are **state-driven and trend-stable at the scale of
20–60 turns** (a posture such as "arm heavily" typically holds for 40–100 turns) but noisy turn to turn, and a
plan visible at t₁ is only a good guide for roughly the first third of a long horizon.

## 2. The AI's economy: taxes, gold, upkeep

- **Rates.** The AI sets science as high as the government allows while keeping the treasury from
  falling; when gold is low or draining it moves points to tax; under Republic/Democracy with spare income it
  sometimes buys luxury to celebrate. Rates therefore swing with the treasury: a civ shown at 90% tax is broke
  or arming; a civ at 60–70% science is comfortable.
- **Treasury (`gold`) is a stock.** Income = trade × tax rate (+ Palace bonus in the capital: +75% under
  Despotism, +50% under Monarchy; + tithes under Fundamentalism; + trade-route revenue). Spending = unit upkeep
  in gold under **Despotism, Monarchy, Fundamentalism (×2), Democracy (×2), Federation (×2)** — 1 coin per unit
  beyond the free ones per city (2–3 free); building upkeep 1–4 coins each (Temple 1, Library 1, Marketplace 0,
  Aqueduct 1, Courthouse 1, University 2, Bank 1, Cathedral 3, Amphitheater 3, Factory 3, power plants 4,
  Barracks 1; Communism pays nothing for 1-coin buildings); rush-buying defenders in threatened cities;
  diplomat bribes and incitements. **Tribal, Republic and Communism pay unit upkeep in shields**, not gold.
- **Upkeep shortfalls are visible.** When a city cannot pay its units' shields, or the treasury cannot pay
  gold upkeep, units are disbanded: the narrative says "X lost to upkeep shortfalls 1 Musketeers, 1 Settlers".
  Frequent lines of this kind mean the civ is over its economic capacity; its military count is capped and its
  treasury is pinned near zero. When gold runs out entirely the AI sells a building.
- **Consequences for prediction.** Gold is the most volatile quantity. Typical patterns: (a) a peaceful civ
  under Monarchy/Republic with 30–40% tax accumulates steadily, hundreds per 100 turns, thousands late in big
  games; (b) an arming civ under a gold-upkeep government drains to 0 within 20–50 turns and then hovers at
  0–50 with periodic disbanding; (c) a government change (anarchy: 0% tax and science for 1–5 turns) or a lost
  war produces a sharp drop. "Treasury falls to 0 at any turn in the window" (1b `bankrupt`) is common for
  militarised gold-upkeep civs, uncommon for shield-upkeep governments with small armies, and unlikely for a civ
  sitting on 1,000+ with a positive trend and no war.
- Trade grows with worked ocean, river and road tiles, Colossus, Republic/Democracy (+1 trade on land tiles
  already producing trade), Marketplace/Library/University/Bank multiply the split. Corruption rises with
  distance from the capital (doubled until The Corporation); Courthouses cut it; Communism/Federation have no
  distance term. Waste (shields) works the same way, doubled until Trade.

## 3. Cities: growth, population, founding, capture

- **Population in thousands = Σ over cities of 5·size·(size+1)** (size 1 → 10, 3 → 60, 5 → 150, 8 → 360,
  10 → 550, 12 → 780, 16 → 1,360, 20 → 2,100). Verified exactly on the corpus. Predict the size list, then
  convert; population is convex in size, so a few big cities dominate it.
- **Growth.** The granary must fill; in civ2civ3 the food box grows with size
  (roughly 14–16 food at size 1 up to 60+ at size 15, times `foodbox`/100). **The largest food stock you
  ever see for a city of a given size is one short of its granary, so it measures `foodbox` directly.**
  At `foodbox` = 100 the granary is a step function of size, measured on the corpus:

  | | size 1–5 | size 6–7 | size 8–9 |
  |---|---|---|---|
  | civ2civ3 granary at foodbox 100 | ≈ 22 | ≈ 33 | ≈ 45 |

  and in classic it is close to (size + 1) × 10.8 — about 32 at size 2, 43 at 3, 54 at 4, 65 at 5. So

      foodbox ≈ 100 × (largest food stock you can find at that size) / (the standard above)

  ⚠️ **This is a lower bound from any one city**, because most cities grow or stagnate well before the
  granary fills: a single city typically shows only 60–90% of its box. Take the maximum over **every** city
  of that size band in **all three frames**, and prefer a city that is size-capped (at 8 without an
  Aqueduct, at 16 without a Sewer) or starving, since those sit at the cap for many turns and read the box
  exactly. Two or three independent readings that agree are worth far more than one. A small city with +3 food grows
  every ~5–8 turns; a size-8 city with +2 food every ~25 turns. Hard caps: **size 8 without an Aqueduct**
  (Aqueduct, Lake / Aqueduct, River are cheap and frequent; the ordinary Aqueduct needs Construction), **size
  16 without a Sewer System** (Sanitation). Unhappiness: base 4 content citizens, one fewer for each step of
  `Empire Size` beyond the government's base (10 Despotism, 12 Tribal, 14 Monarchy, 16 Republic, 20
  Fundamentalism, 24 Federation, 28 Communism, 32 Democracy); temples, amphitheaters, luxury and martial law
  offset it. A city with more unhappy than happy citizens falls into disorder (no output) — the AI avoids this
  by raising luxury or building temples, so growth slows in big empires.
- `city_size_total` grows near-linearly in the expansion phase (new size-1 cities + growth) and saturates
  once land is full and cities reach their caps; drops come from lost cities, plague, famine, starvation when
  the AI reassigns workers, and pollution/global warming late.
- **Founding.** Only Settlers found cities (Migrants join). Each city computes a "founder want": the value of
  the best free site within reach (food/shield/trade of the tiles, river/coast, distance, no overlap with
  existing work areas, minimum city distance 3 in civ2civ3 so packing is dense). When the want is high the city
  builds Settlers and they walk (or take a boat) to the site; the STRATEGIES block shows these as "Expansion: N
  settler parties are on the way to found new cities around (x,y)". Expansionist personality scales the want.
  **Land is the binding constraint**: read the territory map — a civ whose borders touch neighbours or coast on
  all sides founds ≤1 city per 100 turns from now on (only fill-ins and rare overseas sites); a civ with open
  tundra/desert/glacier nearby founds a few poor cities slowly; a civ with open grassland/plains/river land and
  Settlers in hand founds one every 8–20 turns. Settlers in the unit list at t₁ (e.g. "Settlers x7") are cities
  about to be founded — the 1b `newcity` template counts founding only, not capture.
- **Capture.** A city falls when attackers beat every defender. Defence = unit defence × terrain (hills
  +50%, river +25%) × fortified (+50%) × veteran (+50% per level) × **City Walls ×3 against land units**
  (Great Wall gives walls everywhere until Metallurgy; Coastal Defense doubles against ships). AI armies are
  mostly defenders (1–3 per city) and the AI attacks with stacks of its best attacker (Legion/Catapult →
  Knights/Crusaders → Musketeers as pseudo-attackers → Cannon/Cavalry → Artillery/Armor). Individual sieges of
  a walled city are long and bloody ("also lost 5 Musketeers in fighting" for many turns), **but cities change
  hands far more often than a single siege suggests**: over hundreds of AI games, a civ that is at war with at
  least one civilization at t₁ captures at least one city within 60 / 120 / 200 / 300 turns in roughly
  **20% / 35% / 45% / 50%** of cases, and *loses* at least one city in **30% / 45% / 55% / 65%** — because a
  war lasts long enough for a weak spot to be found (size-1–3 border cities, cities with one defender, cities cut
  off from the capital), because pirates take coastal cities, and because captured cities are retaken.
  What moves these numbers: early game (t₁ < 200: capture 40 / 55 / 70 / 75%) vs late (t₁ ≥ 500: 15 / 25 / 30 /
  40%); size of the civ (≤3 cities: 3–12%; 16+ cities: 30–65%); a tech lead of ≥5 techs over the world median
  (40 / 55 / 70 / 75%) vs a lag of ≥5 (10 / 15 / 20 / 25%); and momentum — a civ that gained cities in the last
  40 turns captures again 40 / 60 / 70 / 75% of the time, one that lost cities recently loses again 55 / 70 / 80 /
  85%. A civ at peace with every civilization is not safe either (lost a city within 120 turns: ~45%, mostly to
  barbarians/pirates and to wars that start inside the window). Size-1 cities are destroyed instead of captured;
  `razechance` destroys some buildings of a captured city. Pirates (barbarians landing from the sea) do capture
  weakly defended coastal cities — the narrative then says "The city of X passed from A to <pirate leader> by
  conquest" — and such a city is usually retaken within a few turns (a capture event for the original owner:
  **retaking your own city counts as a capture**, and a capture from a barbarian player counts too).
- **Civil war.** If the capital is captured and the civ has at least `civilwarsize` cities, it may split into
  two civs. Rare; when it happens a new player name appears and city counts drop by half.

## 4. Score and culture

- **Game score** = total citizens + happy citizens (counted a second time) + 2 per known technology +
  **4.5 per Future Technology** (2 as a technology + 2.5 future bonus; `techs_learned` does NOT count Future
  Technologies, the score does) + 5 per great wonder + units built/10 + units killed/3 + a culture term
  (+ spaceship parts). Fitted on the corpus (R² 0.99): **2.4 per tech, 1.1 per citizen, 4.5 per wonder,
  0.021 per culture point** (≈ culture/45). So in civ2civ3 culture is the largest score driver after the first
  200 turns: a civ with 5,000 culture gets ~110 points from it, one with 30,000 gets ~650, and
  `score_gain` over a horizon ≈ 2.4·Δtechs + 1.1·Δ(total city size) + 4.5·wonders + Δculture/45 + units
  built/10 (+ 4.5 per Future Tech once the tree is finished). Score essentially never falls except through lost
  cities (their citizens and their culture leave with them) or tech loss; over 120 turns only ~1% of civs that
  keep all their cities end lower, but 11% of civs that lose a city do. (Classic: no culture term; the classic
  late game is driven almost entirely by Future Techs and city growth.)
- **Typical score levels** (median civ, 10–90% range in brackets; use them as priors when a civ's own trend is
  short or the game is young). civ2civ3, hard AI: turn 60 → 29 (18–49), 100 → 73 (41–113), 140 → 122 (70–184),
  180 → 171 (99–258), 220 → 221 (127–347), 300 → 331 (173–559), 380 → 437 (210–775), 460 → 535 (240–1,050),
  540 → 625 (259–1,350), 700 → 812 (287–1,918), 900 → 1,058 (325–2,690), 1,100 → 1,307 (354–3,413). Cheating
  AIs run ~1.6× these, novice ~0.75×, easy/normal ≈ hard. Classic hard: 60 → 35, 100 → 96, 140 → 142, 180 →
  179, 220 → 212, 300 → 264, 380 → 314, 460 → 384, 540 → 496, 700 → 665, 900 → 916, 1,100 → 1,639 (the late
  rise is Future Techs). Per city, a hard civ2civ3 civ scores ~9 at turn 100, ~30 at 300, ~60 at 540, ~100 at
  900: score grows faster than linearly for the whole game, because culture and techs accumulate.
- **Culture (civ2civ3 only) — how it actually accrues in these worlds.** Every turn each city adds **history
  points** from its buildings — Temple 1, Library 1, Palace 1, Ecclesiastical Palace 1, Colosseum (Amphitheater) 2,
  University 2, Cathedral 3, Research Lab 3 — and each great wonder adds 2–6 per turn (ancient wonders 2, medieval
  3–4, modern 6). **There is no interest on accumulated history in this ruleset** (`history_interest_pml = 0`) and
  no "performance" term: a civ's culture is exactly the sum of what its cities' buildings have produced so far,
  plus one-off **achievement grants**: +100 for the first foreign shore sighted (every civ has it by turn ~60,
  which is why every civ shows ~160 culture at turn 60), and **+1,000 each** for Entire Map Known, Literate (100%
  literacy), Multicultural (citizens of two nationalities — usually after a capture or migrants), Metropolis (a
  size-20 city) and Cultured City (a city with 1,000 culture). Culture therefore rises in a ramp that steepens as
  buildings are added, with occasional +1,000 steps. **Culture belongs to the city that produced it: a captured
  city takes its accumulated culture to the new owner**, so `culture_gain` is negative for ~14% of civs that lose a
  city, and conquest of a big old city moves thousands of points at once.
- **Typical culture levels** (median hard civ, 10–90%): turn 100 → 205 (200–1,200), 140 → 735 (240–1,530),
  180 → 1,440 (410–2,540), 220 → 2,230 (1,170–4,580), 300 → 4,630 (1,600–10,400), 380 → 7,800 (2,200–18,000),
  460 → 11,100 (2,600–27,600), 540 → 13,800 (2,800–39,000), 700 → 21,000 (3,400–58,000), 900 → 29,400 (4,300–
  101,000), 1,100 → 36,400 (4,800–131,000). Cheating AIs ~2×, novice ~0.6×. The per-turn rate of a mid-game
  civ (turn 180–380) is ~25 (per city ~2); late (700+) ~47 (per city ~4). **Early in a game the current rate is
  useless for extrapolation**: from t₁ ≤ 150 the culture gained over the next 120 / 200 / 300 turns is typically
  **8× / 10× / 12×** (10–90%: 0.7–21× / 1–34× / 1.1–44×) what the last 40 turns' rate × h would give, because
  the civ goes from a Palace and two Temples to 20–40 culture buildings plus wonders and two or three
  achievements. From t₁ 150–350 the multiple is ~1.4–1.5× (10–90%: 0.35–5×); from t₁ ≥ 380 it is ~1.0–1.1×
  (0.25–2×).

## 5. Research

- Bulbs per turn = "science output" in the frame (already after Library/University multipliers). The frame
  shows `research: <target> [bulbs so far / cost]  goal: <goal>`. In civ2civ3 the cost is **linear**: 30 bulbs
  × (number of technologies in the target's prerequisite chain including itself) × sciencebox/100 × the
  difficulty factor (novice AI pays 2.5×; all other levels 1×), minus a discount for every civ you have an embassy
  with that already knows the tech ("tech leakage"). So a 10th-generation tech costs ~300, a 20th ~600, a 40th
  ~1,200, a 60th ~1,800 at sciencebox 100. In classic the curve is Civ2-style and steeper: ~365 at 10, ~960 at 20,
  ~1,700 at 30, ~2,600 at 40. Use the cost only for the *next* tech (when does the current research complete;
  when does the goal arrive) and for the sciencebox question in task 2.
- **Recovering `sciencebox` exactly.** The printed cost is `base(chain length) × sciencebox/100`, times 2.5
  for a novice AI, minus the leakage discount. Leakage is zero when no civ this one has an embassy with
  already knows the target, which is the normal case for a tech leader and for every civ early in a game.
  The base costs are fixed and small for the first few generations:

  | prerequisite chain length | 1 | 2 | 3 | 4 | 5 | 6 |
  |---|---|---|---|---|---|---|
  | civ2civ3 base | 30 | 60 | 90 | 120 | 150 | 180 |
  | classic base | 28 | 52 | 80 | 112 | 147 | 185 |
  | techs at that depth | Alphabet, Bronze Working, Ceremonial Burial, Horseback Riding, Masonry, Pottery, Warrior Code | Code of Laws, Currency, Map Making, Mysticism, The Wheel, Writing | Iron Working, Mathematics, Polytheism | Literacy, Monarchy, Seafaring, Trade (classic: Construction, Literacy, Monarchy, Seafaring) | Construction, The Republic (classic: The Republic, Trade) | Astronomy (classic: Astronomy, Feudalism) |

  So a civ2civ3 civ researching **Masonry for a printed 39** has `sciencebox = 100 × 39/30 = 130`; one
  researching **Bronze Working for 26** has 87. Find any civ whose current research is in the table, divide,
  and you have the number. Dividing a late tech's cost by its generation is far less reliable (leakage and
  the exact chain length both bite): measured, the root-tech route is exact while the "cost ÷ known techs"
  shortcut classifies the bin right only 61% of the time.
- ⚠️ **Do not forecast `techs_learned` as science output × h ÷ cost.** That method assumes today's bulb rate
  and today's cost, and both change: science output of a civ at turn 60–140 is **2.5× higher 120 turns later
  and 5× higher 300 turns later** (median; 10–90%: 0.7–8× and 0.5–23×), at turn 180–340 it is 1.5× / 2× higher,
  and only from turn ~400 on is it flat. Measured against the truth, bulbs ÷ cost under-predicts the techs learned
  from t₁ ≤ 140 by a factor of **5** (median, any horizon) and from t₁ 180–340 by 2–2.7×. Both tech sources that
  are not research (treaties, theft, conquest, Philosophy, Great Library, Darwin) are also invisible to it.
- **Forecast `techs_learned` from counts and the timeline instead.** (1) Count the technologies the civ gained
  between the frames (the difference in "known techs", or the "discovered / acquired by treaty / stole" lines per
  interval) → techs per 40 turns. (2) Look up where the civ sits on the timeline below and how many of the **87
  technologies** remain (Future Technology does not add to the count). (3) Combine: early in the game (fewer than
  ~40 techs known, t₁ < 200) the count rate holds or rises slightly for the next 100–200 turns (multiple of
  rate × h over 60 / 120 / 200 / 300 turns: median 1.1 / 1.0 / 0.9 / 0.8, 10–90% 0.6–2.3 / 0.5–2.3 / 0.5–2.0 /
  0.4–1.8); mid-game (40–75 known) it decays (median 0.8 / 0.75 / 0.65 / 0.5 of rate × h); with 75+ known the
  remaining tree is the binding constraint (median next-100-turn gain 2–3, and it cannot exceed 87 − known).
- **Timelines — median techs known (10–90%) by turn, per difficulty** (all civs, hundreds of games):
  civ2civ3 **hard**: t60 7 (4–11), t100 16 (10–23), t140 26 (16–35), t180 34 (22–44), t220 42 (28–52), t300 52
  (39–70), t380 61 (47–84), t460 70 (50–87), t540 78 (52–87), t700 83 (58–87), t900 84 (61–87).
  civ2civ3 **cheating**: t60 14, t100 29, t140 49, t180 65, t220 81, t300 87 (the whole tree by turn ~300).
  civ2civ3 **easy**: t100 19, t140 28, t180 39, t220 45, t300 56, t380 70, t460 78, t540 85.
  civ2civ3 **novice**: t100 7, t140 12, t180 14, t220 21, t300 34, t380 40, t460 47, t540 52, t700 70, t900 87.
  classic **hard**: t60 9, t100 26, t140 38, t180 48, t220 58, t300 71, t380 83, t460 87; classic **normal**:
  t100 24, t140 35, t180 45, t220 53, t300 65, t380 75, t460 86; classic **easy**: t100 21, t140 32, t180 39,
  t220 44, t300 60, t380 71, t460 81; classic **novice**: t100 8, t140 15, t180 24, t220 30, t300 39, t380 45,
  t460 56, t540 62, t700 76; classic **cheating**: t100 32, t140 42, t180 50, t220 58, t300 70, t380 80, t460 87.
  So a hard civ learns ~25 techs per 100 turns until turn ~200, ~10 per 100 turns from 300 to 500, and finishes
  the tree around turn 500–700; a **first-frame civ at turn 60 with 7 techs will typically know ~55 at turn 300
  and ~70 at turn 360** — i.e. 45–65 techs learned over h = 300, not 25. Civs above the median at t₁ stay above
  it (rank is persistent); the 10–90% band of the timeline is a fair p10–p90 when the civ's own count rate is
  ambiguous. Team games (shared research) run at the fast end.
- Techs per 100 turns depend on the era far more than on size: at turn 60–140 the median is ~20 for any civ
  with 4+ cities (14 for a 1–3-city civ); at 180–340 it is 9–11 (6 for tiny civs); at 380–660 it is 0–2 for most
  (the tree is nearly done) and up to 10 for laggards catching up by treaty.
- The AI's **goal** is a distant tech it wants for a unit, building or government (Monarchy, Currency → Trade,
  Bronze Working → Colossus, Philosophy for the free tech, The Republic, Sanitation, Invention → Leonardo /
  Gunpowder, Navigation for ships, Conscription/Tactics when militarised, Railroad/Industrialization for
  economy, Sanitation for growth). It researches the cheapest prerequisite path to it; when reached it picks a
  new goal. "Research aim" in STRATEGIES is that goal.
- Other tech sources: Philosophy first-discovery bonus; Great Library (any tech two other civs know);
  Darwin's Voyage; **treaty exchange between allies** (very common — the narrative says "X acquired by treaty
  the technology Y"; an ally of the tech leader catches up several techs per 50 turns); **stealing by
  Diplomats/Spies** ("X stole the technology Y" — the AI builds many Diplomats and steals actively from
  neighbours); conquest of a city (one tech). A civ far behind with an advanced ally or many diplomats can gain
  6–10 techs in 60 turns; an isolated leader gains only what it researches.
- **Tech upkeep and tech loss (civ2civ3).** Knowing many techs costs bulbs every turn; a civ whose bulb
  balance goes negative (science output collapses after losing cities, anarchy, Fundamentalism's −50%) can
  **lose a technology**. `techs_learned` may therefore be negative for a civ in a collapse (10% of late-game
  civ-windows show −1 to −3), and `n_techs` is not monotone. A civ that knows all 87 techs researches "Future
  Technology": it does not raise the count but adds 4.5 score each; a finished civ2civ3 civ completes ~2 (0–6)
  Future Techs per 100 turns, a finished classic civ ~10–20 per 100 turns (cheap in classic — this is what drives
  classic scores late).
## 6. What the AI builds in its cities, and its military

- Each turn each city runs the **danger assessment**: enemy units within a few turns' march, weighted by
  attack strength, minus its own defence (walls, fortified defenders). Under danger the city builds (and may
  rush-buy) the best available defender, then City Walls / Coastal Defense; "Defence: one of its cities feels
  exposed" in STRATEGIES means at least one city currently has positive danger.
- Otherwise the **military advisor** and the **domestic advisor** bid: military wants attackers when there is
  a war target (see §7), explorers early, ferries for overseas targets, and a floor of 1–3 defenders per city
  (Warriors → Phalanx → Pikemen → Musketeers → Riflemen → Alpine Troops / Mech. Inf. as techs arrive; coastal
  cities also get a ship); domestic wants Settlers (founder want), Workers (terrain improvement want; the AI
  keeps ~1 Workers per 1–2 cities and they irrigate/mine/road constantly — "X's Workers units built 2
  irrigation and 2 mines"), Caravans (for wonders in the wonder city and, in these worlds, for trade routes;
  Trader scales this), buildings by effect (Granary, Temple when unhappy, Library/Marketplace by output,
  Aqueduct near the size cap, Harbor for ocean cities, Barracks in militarised civs — Barracks are built
  everywhere when "arm heavily"), and great wonders in the best production city when the want is high.
  **Builder** discounts the military bid (a high-Builder civ builds proportionally more buildings, a low one
  more units). What a city is building now is in the frame ("building" column) with its shield stock, so the
  next 10–20 turns of production are readable: (cost − stock) / production per city.
- **Military unit count** (`n_military_units` = units with attack > 0; Triremes, Transports, Carriers,
  Workers, Settlers, Caravans, Diplomats, Explorers are civilian). It scales with cities: ≈1.5–2 per city for a
  peaceful civ, 3–5 per city for a rich civ at war or "arming heavily" (a 15-city civ can hold 60–80). It
  **jumps up** when an arming posture starts (10–30 units per 40 turns for a mid-size civ) and **falls** when
  upkeep can't be paid (disbanding), when an offensive fails (attacker losses of 5–15 units per 5-turn
  paragraph are typical in a siege), or when cities are lost (units homed there disband). The 1b `mildrop`
  template (≥30% below the t₁ level at any turn in the window) is a bankruptcy-or-lost-war signal: for a
  peaceful civ with a healthy treasury it is unlikely; for a civ already logging "lost to upkeep shortfalls"
  lines, or on the losing side of a war, it is likely even within 60 turns.
- **Civilian counts** move in bursts: Settlers appear during expansion and vanish when they found (so many
  Settlers at t₁ + free land ⇒ n_civilian down, n_cities up over the next 20 turns); Caravans accumulate under
  Trader-heavy civs and vanish on arrival; Workers grow slowly; Diplomats pile up in AI civs (dozens late).
- Obsolete units are kept (the AI upgrades only with Leonardo's Workshop or spare gold), so old Phalanxes
  linger for hundreds of turns.
- **Combat arithmetic** for judging sieges: attack vs defence as above, hit points 10/20/30 and firepower 1;
  a Musketeers (3 attack) attacking a fortified Musketeers (3 defence) behind walls (×3) wins ~1 in 20 fights,
  which is why AI wars between equal-tech neighbours are long stalemates with heavy attacker losses; Cannon
  (8) / Artillery (10) / Cavalry (8) against Musketeers behind walls (13.5) still lose more often than win;
  Armor (10) / Howitzer (12, ignores walls) against Riflemen (4) is when cities start falling. Ships bombarding
  a walled city do little. Barbarian landings (Legions / Musketeers / Cannon later) can take a size-2–4 coastal
  city with 1 defender.

## 7. Diplomacy: how the AI actually behaves

- **States.** Never met → (contact) **Peace** in these worlds → War by declaration; War → Cease-fire (16
  turns) → peace treaty (Armistice 16 turns → Peace) or lapse back to War; Peace → Alliance by proposal.
  Contact happens when units or borders meet; on big maps distant civs may stay "never met" for hundreds of
  turns and are not counted in anything.
- **Love** (−1000…+1000) is recomputed every turn: +15 in any non-war state (+10 more when allied, +7.5 per
  shared enemy), −15 per turn at war and −5 per enemy unit inside the border (capped), 4% decay towards 0,
  +40 per turn for cheating-level AIs, and a **grudge**: once an AI picks another player as its war target it
  subtracts up to 25·Aggressive/50 per turn until it declares. Thresholds: ceasefire needs both sides above
  −600; peace treaty needs love > 125 (and is proposed only in the last 4 turns of a ceasefire); alliance needs
  love ≥ 250 from Peace with no war countdown. Above 800 a non-ally loses 10 per turn.
- **Target choice and declaration.** Each turn each AI scores every player as a war target: value of their
  cities (size, buildings, gold-equivalent) discounted by distance and defence, boosted by hatred (low love),
  by an existing war of an ally against them, and by "casus belli" behaviour (their units inside its borders).
  The best target with positive desire gets a **countdown**; STRATEGIES prints it as "War aim: it intends to
  attack X and will move in about N turns". About half of printed countdowns lead to a declaration within 10
  turns; the rest are cancelled when love recovers or the target allies with someone strong. Reasons the AI
  attaches to wars: hatred, excuse (behaviour), alliance (dragged in by an ally — very common), space race
  (everyone attacks a spaceship builder), none.
- **Wars are sticky.** Once at war, love sits near −1000 and the ceasefire threshold is rarely met unless
  the front goes quiet for ~20 turns; after a ceasefire, peace needs love > 125 within 16 turns, which
  usually fails → "The cease-fire between A and B ended and they are at war again" (a frequent narrative line).
  Peace treaties between long-time enemies do happen but are a small minority of pairs late in the game.
- **Alliances are common and sticky.** A fifth to two fifths of contacted pairs are allied mid-game; allies
  share vision, **exchange technologies by treaty**, join each other's wars, and rarely break up (breaking
  needs love to collapse). Blocs form: the typical late picture on a 6–8 civ map is two or three blocs, most
  cross-bloc pairs at war, most intra-bloc pairs allied. A new alliance (1b `newally`) mostly arises from a
  Peace pair that shares an enemy; it needs ~20+ turns of Peace to build love, so a pair that just made peace
  is a candidate 30–60 turns later, a pair with a war countdown is not. Measured: a civ with **no** Peace pair
  at t₁ forms a new alliance within 60 / 120 / 200 / 300 turns in 5 / 9 / 13 / 17% of cases; with at least one
  Peace pair, 45 / 57 / 64 / 69% (55 / 66 / 71 / 75% if it is also not at war with anyone); cheating-level
  worlds sit at 25–43% regardless because everybody loves everybody already.
- **Counts.** `n_enemies` = living contacted players at war with the civ, **barbarians and pirates included
  while alive** (a pirate landing party is a temporary enemy; when its last unit dies the count drops);
  destroyed civs drop out. `n_allies` = living allies (Team members are not allies). Over 60 turns these
  counts usually stay put or move by ±1; ±2 happens when a bloc war starts (several declarations in one
  paragraph) or when barbarians appear. Measured over 60 / 120 / 200 / 300 turns: `n_enemies` unchanged 56 /
  53 / 49 / 45%, −1 in 17–20%, +1 in 18–19%, |Δ| ≥ 2 in 9–16%; `n_allies` unchanged 85 / 79 / 73 / 66%, +1 in
  6–11%, −1 in 4–9%. "More enemies at the end than now" (1b `moreenemies`, an endpoint question): 24–28% of all
  civ-windows, but ~50–60% for a civ currently at war with nobody (it will meet someone or be dragged in) and
  only ~18% for a civ already at war with two or more (wars end as often as they start).
- **Aggressive personality** in practice: high-Aggressive AIs declare on a neighbour within a few dozen turns
  of contact and re-declare after ceasefires; low-Aggressive AIs (≤ ~20) almost never start wars but are
  pulled in by allies. In STRATEGIES, "hostile towards X" is love < −250, "friendly" is love > 250; many
  "hostile" pairs are already at war, so read the frame's relation table too.

## 8. Governments

- civ2civ3 AI path: Tribal or Despotism at start → **Monarchy** as soon as Monarchy is known (most civs by
  turn 60–120) → later **Republic** (peaceful, trade-rich), **Communism** or **Fundamentalism** (at war, big
  empire), **Democracy** / **Federation** (rich, peaceful, late). The AI evaluates every available government's
  effects on its current situation each turn and switches when a clear improvement appears — in practice on
  the turn a new government tech (or the Statue of Liberty) arrives, or when a long war makes Republic's war
  weariness or Monarchy's gold upkeep intolerable. Most civs change government **0–2 times per 200 turns**;
  a civ already in a government suited to its situation keeps it for hundreds of turns. Each change passes
  through **Anarchy for 1–5 turns** (random): science and taxes stop, gold and bulbs dip, culture stalls;
  STRATEGIES shows "Government: it is in anarchy; its new government takes office in N turns" when a change is
  under way. Barbarian/pirate players flip Tribal ↔ Monarchy noisily; ignore them.
- 1b `govchange` in the window — measured frequencies over 60 / 120 / 200 / 300 turns by the government at
  t₁: **Despotism 60 / 69 / 75 / 81%** (and 83 / 94 / 97 / 99% when the civ knows fewer than 12 techs: Monarchy
  is coming), Tribal 18 / 24 / 29 / 35%, **Monarchy 38 / 53 / 62 / 69%** (Monarchy at peace with everyone 61 /
  85 / 95 / 98% — it will switch to Republic/Democracy; Monarchy at war 34 / 48 / 56 / 63%), Communism 14 / 21 /
  28 / 35%, Fundamentalism 14 / 21 / 28 / 35%, **Republic 89 / 97 / 99 / 99%** (the AI leaves Republic almost
  immediately, usually for Democracy or, at war, Communism/Fundamentalism), Democracy 50 / 66 / 77 / 86%,
  Federation 57 / 76 / 86 / 91%. Sharpen with the research aim (a government tech due inside the window → up)
  and with war (Republic/Democracy at war → up; Communism/Fundamentalism at war → stays). Anarchy at t₁
  ("Government: it is in anarchy") counts as a change when the new government takes office inside the window.

## 9. Wonders

- Great wonders cost 200 (ancient) – 300/400 (medieval/renaissance) – 600 (modern) shields; one per world.
  The AI designates a **wonder city** (best production, usually the capital region), builds wonders there in
  sequence, and helps with Caravans ("Help Wonder") — caravans "under way to its own cities" in STRATEGIES often
  mean this. From the frame: if a city's `building` column shows a wonder with stock s and the city makes p
  shields, completion is in ≈ (cost − s)/p turns (caravans shorten it). A civ with the "build wonders" plan
  completes one every 30–60 turns mid-game; other civs complete one occasionally when a wonder tech arrives
  (Colossus, Pyramids, Great Library, Temple of Artemis early; Michelangelo, Leonardo, Copernicus, King
  Richard's, Sun Tzu mid; Newton, Bach, Magellan, Darwin, Statue of Liberty, Eiffel later). Wonders give +5
  score, culture, and love (Eiffel, United Nations). Measured: a civ with a wonder in some city's `building` column at t₁
  completes at least one within 60 / 120 / 200 / 300 turns in **61 / 68 / 69 / 69%** of cases (the missing third
  loses the race to another civ, which then switches production); with no wonder in production, 6 / 10 / 14 /
  17% overall — 8 / 13 / 17 / 20% for civs with 8+ cities and 0–2% for civs with fewer than 4 cities.

## 10. Barbarians, pirates, disasters, illness, other hazards

- **Barbarians** (`onsetbarbs`, typically turn 60–250): land uprisings near poorly defended cities and
  **pirate landings** on coasts (a leader plus 3–8 Legions/Musketeers/Cannon in a boat). Their player names
  are pirate names (Blackbeard, Cofresí, Kidd, Morgan, Bonny…); they appear in the narrative attacking units and
  cities and in the frames' relation tables as enemies of everyone. While a barbarian player has units it
  counts in every contacted civ's `n_enemies`; it usually dies within 10–30 turns. Small coastal cities with a
  single defender are the ones they take.
- **Disasters** (`disasters` setting, chance per city per turn): Earthquake (destroys a building), Fire
  (destroys a building), Flood (river cities lose food stock or a citizen), Plague (city loses a citizen),
  Famine (granary emptied, growth reset), Industrial Accident (Mfg. Plant: pollution). The narrative reports
  them as their own sentence naming the disaster and the city. Disasters knock `city_size_total`,
  `population` and `culture` (lost buildings) by small steps.
  **Counting them identifies the setting.** Count the disaster sentences over the whole observation window
  and divide by (number of cities you can see × number of turns the window covers), per 1,000 city-turns.
  Measured on 90 worlds over an 80-turn window, the three settings barely overlap: **rare or none** median
  0.0, **normal** median ~0.9, **frequent** median ~3.4. Classifying at **below 0.85 / 0.85-1.65 / above
  1.65 per 1,000 city-turns** gets the bin right 88% of the time against a 40% majority class. Two extra
  rules for small counts, where the rate is unreliable: **no disaster sentence anywhere in the window means
  rare or none** (94% of such worlds), and a **single** sentence still favours rare or none (67%) unless the
  window is short. Frequent worlds essentially never show fewer than two. Do the arithmetic; do not judge
  "several" by eye. Famine and plague
  sentences are NOT disasters (famine is a food-box effect and plague is never narrated) -- count only the
  named disaster lines.
- **Illness** (civ2civ3, off / normal / high): plague chance grows with city size above 3, is cut 30% by an
  Aqueduct (any kind) and raised by pollution; a plague costs the city one citizen and is reported as such.
  In a high-illness world big cities without aqueducts lose citizens repeatedly and `city_size_total`
  fluctuates instead of growing.
- **Pollution and global warming**: late game (Factories, Power Plants, size > 15) produces pollution; if
  unattended, global warming turns coasts into swamp/jungle/ocean — visible only in 1200-turn games' last
  third; it lowers food and city sizes gradually.
- **Famine/starvation** also occurs without disasters when the AI shifts workers to shields in wartime; the
  city shrinks by one and refills.

## 11. Difficulty levels

- **novice**: research costs 2.5×; expansion capped (few cities); builds many defenders, poor at attacking;
  naive diplomacy; slow. **easy**: near-normal costs, still limited expansion and defensive. **normal**: mild
  handicaps (rates, target choice, no plane attacks). **hard**: no handicaps — the reference AI. **cheating**:
  no handicaps plus +40 love per turn from everyone (so cheating worlds are friendlier: more peace and
  alliances, fewer wars) and some economic bonuses. Difficulty is in the world id in the frames' header context
  (e.g. `-hard-`, `-novice-`); use it to scale growth and war expectations.

## 12. How the prompt is written (so you read it right)

- **Frames**: three snapshots at t₁−2Δ, t₁−Δ, t₁ (Δ = 40 turns usually; 20 early). Frame 1 carries the maps
  (terrain and territory); frames 2–3 do not (terrain is static). Each civ block: government, treasury,
  per-turn trade/production/science output, rates ("revenue allocation"), population, score, culture, known
  techs and the research line with cost, wonders, territory size and resources, the city table (name, x,y,
  size, founded turn, building now, food stock, shield stock, buildings, worked-tile resources), and the unit
  roster by type with military/civilian counts. In the single-civ view (condition v2) only the viewer's block
  is complete; other civs are shown as the viewer knows them.
- **INTERVAL** paragraphs (five turns each) between frames. Vocabulary you will see: "changed its government
  from A to B", "discovered the technology X", "acquired by treaty the technology X" (from an ally), "stole the
  technology X", "lost to upkeep shortfalls N <units>", "cities built new units: …", "completed the buildings …
  and the great wonder X in Y", "Workers units built 2 irrigation and 1 mine", "moved military units: 29
  Musketeers, 9 Phalanx to around (19,26)" (a stack forming = an offensive or a defence), "attacked …'s city of
  X (defended by …); the attacker lost about 60% HP", "also lost N units in fighting not described above",
  "declared war on", "signed a cease-fire / peace treaty / alliance", "The cease-fire between A and B ended and
  they are at war again", "The city of X passed from A to B by conquest", "founded the city of X", disasters,
  plague, civil war. Count these per civ per interval to estimate rates (units built per 40 turns, techs per 40
  turns, cities founded, losses).
- **STRATEGIES** block at t₁ (present in tasks 1a/1b/3, absent in task 2): per civ, a course ("arm heavily",
  "build up its existing cities", "build wonders", "keep expanding", "hold its position"), "Research aim",
  optional "War aim: it intends to attack X and will move in about N turns", "Forces: military units are on the
  move towards (x,y)" (destinations reveal fronts — compare with the territory map to see whose border they
  approach), "Expansion: N settler parties … around (x,y)", "Trade: caravans are under way to …", "Defence: one
  of its cities feels exposed", "Government: in anarchy …", and "Neighbours: friendly/hostile towards …".
- **Barbarian players** appear in the narrative and sometimes in relation lines but are never question
  targets; they do count in `n_enemies` while alive.
- In **team games** the research line is per team: two teammates show the same target and the same bulbs, and
  the bulbs stored can exceed the printed cost for a turn or two (the tech is granted at turn change). Do not
  read that as a data error; count the techs gained instead.
- **Task 3 arms**: the prompt is the same world with an intervention described in one sentence, e.g.
  "X's treasury changes from 210 to 1,210 gold" (an edit to the t₁ frame — the frame itself is edited),
  "X gains 400 research points toward its current research", "X's largest 2 cities are destroyed: …",
  "X acquires the technology Y" / "X completes 3 buildings: …", "X suddenly learns the geography of the entire
  map", "X's hidden attitude toward Y changes from −300 to +600 on the −1000..+1000 scale", "X's hidden
  Expansionist rises by 90 on the 10..150 personality scale", "From now on natural disasters strike this world 3
  times as often". The **control** arm has no intervention (identical to the factual world); the **placebo**
  arm has a tiny edit of the same kind (a few gold, +3 on a trait).

## 13. What moves fast and what moves slowly — the measured 80% bands

Over hundreds of AI games, the value at t₁ + h divided by the value at t₁ (level quantities), or the gain over
the horizon divided by (last-40-turn rate × h) (increment quantities), falls between these 10% and 90%
quantiles. Read them as the width a well-calibrated p10–p90 needs **when you have no specific mechanism
evidence**; when you do (a war aim, a Settlers stack, a wonder in production, "arm heavily"), the median moves but
the width rarely shrinks below two thirds of these bands on a log scale.

| Quantity | h = 60 | h = 120 | h = 200 | h = 300 | median drift | drivers |
|---|---|---|---|---|---|---|
| gold (× level) | 0.20 – 4.3 | 0.12 – 6.8 | 0.08 – 11 | 0.06 – 20 | ×1.1–1.2, fat right tail | tax rate, upkeep type, war, anarchy |
| population (× level) | 0.80 – 1.57 | 0.69 – 2.0 | 0.57 – 2.65 | 0.46 – 3.5 | ×1.04 → ×1.16 | sizes (convex), aqueducts, war, plague |
| n_cities (× level) | 0.89 – 1.17 | 0.78 – 1.27 | 0.67 – 1.40 | 0.50 – 1.52 | ×1.00 (flat median) | free land, settlers, war |
| city_size_total (× level) | 0.86 – 1.33 | 0.75 – 1.55 | 0.64 – 1.83 | 0.51 – 2.18 | ×1.02 → ×1.09 | food, caps, happiness, war |
| n_military_units (× level) | 0.81 – 1.71 | 0.73 – 2.3 | 0.65 – 3.1 | 0.57 – 4.25 | ×1.04 → ×1.19 | posture, war, gold |
| n_civilian_units (× level) | 0.73 – 1.50 | 0.60 – 1.74 | 0.50 – 2.0 | 0.42 – 2.4 | ×1.00 → ×1.08 | settlers, caravans, diplomats |
| techs_learned, t₁ ≤ 150 (× rate·h) | 0.6 – 2.3 | 0.5 – 2.3 | 0.5 – 2.0 | 0.4 – 1.8 | ×1.1 → ×0.8 | see §5 timeline |
| techs_learned, t₁ 150–350 (× rate·h) | 0.3 – 1.6 | 0.3 – 1.5 | 0.2 – 1.4 | 0.1 – 1.2 | ×0.8 → ×0.5 | remaining tree |
| culture_gain, t₁ ≤ 150 (× rate·h) | 0.35 – 19 | 0.7 – 21 | 1.0 – 34 | 1.1 – 44 | ×3 → ×12 | buildings, achievements |
| culture_gain, t₁ 150–350 (× rate·h) | 0.3 – 3.9 | 0.35 – 4.9 | 0.35 – 5.6 | 0.35 – 6.1 | ×1.2 → ×1.55 | buildings, wonders |
| culture_gain, t₁ ≥ 380 (× rate·h) | 0.4 – 1.7 | 0.3 – 1.8 | 0.25 – 2.0 | 0.25 – 2.1 | ×1.0–1.1 | lost cities (negative tail) |
| score_gain, t₁ ≤ 150 (× rate·h) | 0.6 – 3.0 | 0.65 – 3.1 | 0.6 – 3.4 | 0.6 – 3.7 | ×1.3 → ×1.45 | culture + techs accelerating |
| score_gain, t₁ 150–350 (× rate·h) | 0.45 – 1.9 | 0.4 – 1.95 | 0.4 – 2.0 | 0.35 – 2.2 | ×1.05 | same |
| score_gain, t₁ ≥ 380 (× rate·h) | 0.35 – 1.65 | 0.35 – 1.65 | 0.25 – 1.7 | 0.25 – 1.7 | ×1.0 | classic: Future Techs, right tail 2–3× |
| n_enemies / n_allies | ±1 in ~35% / ~10% | | | | discrete | §7 |

Note the asymmetry: the right tails (growth, arming, hoarding) are much longer than the left ones for every
quantity except cities and civilian units at war. Rules of thumb: use the **rate between frames 2 and 3** for
growth quantities and check it against frames 1→2; damp city counts and sizes toward saturation when the
territory map shows no free land; treat gold as a distribution around the recent level with a mass near 0 for
over-extended gold-upkeep civs and a fat right tail for rich peaceful ones; widen every interval with the horizon;
make intervals asymmetric where the mechanism is (gold, n_cities in war, military at war, every increment early
in a game).

Measured gold bands by regime (h = 120, × level): gold-upkeep government at peace 0.07–6.9 (treasury hits 0 at
some turn: 8%), gold-upkeep at war 0.11–4.9 (15%), shield-upkeep at peace 0.08–8.0 (17%), shield-upkeep at war
0.10–4.0 (27%). Treasury hits 0 inside 60 / 120 / 200 / 300 turns: gold < 50 at t₁ → 38 / 55 / 66 / 74%;
50–300 → 14 / 25 / 36 / 45%; 300–1,000 → 3 / 9 / 19 / 28%; ≥ 1,000 → 0 / 2 / 5 / 9%. Military falls ≥30% below
the t₁ level at some turn (1b `mildrop`): 8 / 13 / 18 / 22% at peace, 10 / 17 / 23 / 28% at war, 17 / 26 / 32 /
37% when gold < 50 under a gold-upkeep government. New city founded: with Settlers in hand 54 / 66 / 74 / 80%,
with no Settlers but cities founded in the last 40 turns 55 / 69 / 78 / 84%, with neither and t₁ ≥ 300 16 / 25 /
33 / 41%.

## 14. Task-by-task working method

**1a (quantiles, 11 quantities × civs × horizons).** For each civ classify the regime from frames +
STRATEGIES: expanding / consolidating / arming / at war winning / at war losing / boxed-in peaceful /
collapsing. Compute per-turn rates from frames 2→3 with a calculator, extrapolate, then adjust for: land
saturation and size caps; research goal (a government tech → anarchy dip; Sanitation → growth; military techs →
unit type upgrade in production); posture ("arm heavily" → military up 30–100%, gold down, buildings flat);
war aims and hostile pairs within the horizon (both sides' military up, gold down, cities may change hands,
n_enemies +1 for both, allies +1 too); ally tech treaties (laggard's techs up).

- **The early-game acceleration rule (t₁ ≤ ~150, or any civ that is still expanding).** The three increment
  quantities (`techs_learned`, `culture_gain`, `score_gain`) are NOT linear in the current rate: output grows with
  every city, Library, Temple and wonder, so from a young position the realised gain over 120–300 turns is
  typically 1.4× (score), 1.0× but from a much higher base than bulbs ÷ cost suggests (techs — use the §5
  timeline) and 8–12× (culture) the last-40-turn rate × h. Build these forecasts from the **timelines in §4–§5**
  (median level at t₁ + h for this difficulty, scaled by where the civ sits relative to the median now) and
  cross-check with the rate; when the two disagree, the timeline wins for h ≥ 120. For `city_size_total` and
  `population` from a young position, expect the total size to roughly triple between turn 60 and turn 140 and then flatten (median hard-AI
  total city size 10 → 28 → 44 → 52 → 58 → 63 at turns 60 → 100 → 140 → 180 → 220 → 300; cities 5 → 8 → 10 → 10 →
  10 → 11).
- **Late-game positions (t₁ ≥ 400).** Techs: bounded by 87 − known; culture and score: the current rate is a
  good median (×1.0–1.1); classic score_gain: count Future Techs (4.5 each) — a finished classic civ gains
  100–250 score per 100 turns from them alone.
- **Interval widths.** p50 = your estimate. p10 / p90 = you would be surprised to be outside one time in five —
  and in practice annotators are surprised far more often than that, always on the high side. Enforce these
  minimums after setting p50: (i) never narrower than two thirds of the §13 band on a log scale; (ii) for the
  increment quantities and for gold, `n_military_units` and `population`, the **upper** half-width (p90 − p50)
  is at least **twice** the lower half-width, and for increments p90 ≥ 2 × p50 at h ≥ 120 when t₁ < 400 (≥ 2.5 × p50 from
  an early position; ≥ 1.6 × p50 late in a game); (iii) p10 of an increment is at most 0.5 × p50 (0 or negative when cities can be lost — culture
  and score fall with lost cities; techs can be lost in a collapse); (iv) widen with the horizon: a 300-turn band
  is ≥ 2× the 60-turn band on a log scale; (v) p25 / p75 sit roughly a third of the way from p50 to p10 / p90.
  Sharpness is rewarded only when you are right: the cost of a truth beyond p90 is far larger than the cost of a
  slightly wide band.
- **Self-check before writing 1a.** For every civ: (a) does `techs_learned` at the longest horizon put the civ at a
  plausible point of the §5 timeline for its difficulty (known + p50 within the 10–90% band there)? (b) does
  `score_gain` reconcile with 2.4·Δtechs + 1.1·Δsize + Δculture/45 + 4.5·wonders using your own Δ forecasts?
  (c) is `culture_gain` from a young position several times the naive rate × h? (d) is each p90 of an increment
  ≥ 2 × p50 and each p90 of gold / military / population comfortably above the trend? (e) is `n_cities` p10 below
  the current count for any civ at war or with a hostile neighbour? Fix what fails.

**1b (probabilities; every question shows its base rate).** The base rate you are shown is not a global
average: it is the rate of this exact template, budget, horizon and turn measured on the question set
itself. It already contains everything a table can tell you. Your only way to beat it is to say which
*particular* civ in *this* world the event happens to. So start there and move on evidence that is
specific to the civ in front of you, using the measured conditional frequencies in §3, §7, §8, §9 and §13
to decide how far.

- **How far to move.** Strong, mechanical evidence (a wonder at 180/200 shields with three caravans one
  tile away; a stack of 20 Cannon next to an undefended size-2 city; a government tech completing in six
  turns) is worth ±0.25 to ±0.35. Suggestive evidence (a war aim, a falling treasury, a Peace pair with a
  shared enemy) is worth ±0.10 to ±0.15. Nothing in the window that bears on this civ is worth ±0.00:
  answer the base rate and move on. Never leave 0.03–0.97.
- **Six things that carry more or less weight than a player expects.** Each was fitted on worlds outside
  this benchmark, controlling for the other facts, so these are the directions to trust when your instinct
  disagrees. (1) For `bankrupt` the treasury **level** is almost the whole story, and a gold-upkeep
  government slightly *lowers* the probability once the level is known — those civs are already managing it.
  (2) For `newcity` what matters is whether the civ **founded** a city in the last 40 turns; Settlers sitting
  in the roster are weak evidence by comparison. (3) For `capture` and `lostcity`, how **small** the civ is
  dominates everything, and "at war" adds nothing once size and momentum are known, because it is the small
  civs that are at war. (4) For `wonder`, being tiny (≤ 4 cities) suppresses it more strongly than a queued
  wonder raises it. (5) For `govchange` the government the civ is in right now decides the answer. (6) For
  `newally`, whether a Peace pair exists at all is worth more than everything else combined.
- ⚠️ **Piling up scary-looking evidence does not multiply.** Measured on worlds outside this benchmark,
  the treasury falls to zero within 60 / 120 / 200 / 300 turns for **0.38 / 0.55 / 0.66 / 0.74** of civs
  holding under 50 gold, and for **0.35 / 0.50 / 0.61 / 0.69** of civs that hold under 50 gold *and* are
  draining *and* pay gold upkeep *and* are at war. The extra evidence changes nothing, because an AI at the
  floor is already managing it: it raises taxes, sells a building, disbands units and survives. Read the
  treasury level, apply the §13 table, and stop. The same restraint applies to any civ that merely looks
  doomed on an economic reading.
- **Where evidence does stack.** Losing a city: at war with under 2 military per city **and** having lost a
  city in the last 40 turns runs 0.68 / 0.80 / 0.88 / 0.92, against 0.25 / 0.36 / 0.46 / 0.55 for a civ at
  war with 3+ per city. Founding a city: Settlers in hand **and** a city founded in the last 40 turns runs
  0.76 / 0.85 / 0.91 / 0.93, against 0.33 / 0.47 / 0.57 / 0.65 for Settlers in hand but nothing founded
  since turn 300. Capture: the momentum, tech-lead and size conditions in §3 compound. In these three a
  confident 0.85 is justified; in `bankrupt`, `wonder`, `mildrop` and `newally` it is not.
- ⚠️ **Do not read the paragraph above as "always answer near the base rate".** Half of these templates
  reward a large, confident move: `capture`, `lostcity`, `newcity`, `govchange` and `stillahead`/`leader`
  are decided by things the frames show plainly — stacks on a border, a tiny army, Settlers and free land,
  the government the civ is in now, two score curves you can extrapolate — and on those the base rate is a
  poor answer and a deviation of 0.25–0.35 is routinely right. The restraint is specific to the four
  templates where dramatic evidence does not translate into a higher rate. Timid everywhere scores no
  better than reckless everywhere.
- **Template pointers.** bankrupt — treasury level and upkeep type, and little else (§13); newcity —
  Settlers, the "Expansion" line, free land on the territory map, recent foundings; **capture** — do not
  talk yourself down because AI sieges are slow: over 200–300 turns half of the civs at war capture
  something, three quarters if they are early, lead on tech or gained cities recently, and retaking your own
  city or taking one from pirates both count (§3); lostcity — an active war, a small or lagging civ, recent
  losses, pirates for small coastal cities, and note that a civ at peace with everyone still loses one about
  45% of the time within 120 turns; mildrop — upkeep-shortfall lines, a losing war, bankruptcy; wonder — a
  wonder actually in a production queue is 0.61 / 0.68 / 0.69 / 0.69 and little else moves it, so do not go
  above ~0.8 on a queue alone; moreenemies (endpoint!) — a civ at war with nobody is *more* likely to gain
  an enemy than one already at war with two (§7); newally — Peace pairs with shared enemies and time to
  build love (§7); govchange — start from the §8 table for the government it is in now, then adjust for the
  research aim and for war; stillahead / leader — build both scores forward from 2.4·techs + 1.1·sizes +
  culture/45 and compare, rather than assuming the leader stays ahead.
- Window questions ("at any turn between") deserve higher probabilities than the matching endpoint question.

**2 (hidden state).** Three separate questions with very different winnable space.

- **Personality ranking.** The four values are hidden and only weakly visible: even an observer with
  privileged behaviour statistics reaches 0.59 pairwise accuracy against 0.50 chance. Three of the four
  dimensions can be taken to that ceiling with one arithmetic rule each; the fourth cannot be read at all.
  Measured on 90 worlds outside this benchmark, 7,400 ranked pairs per dimension, ranking every civ by the
  quantity in bold:

  | dimension | rule (all quantities standardised **within the world** before adding) | pairwise accuracy |
  |---|---|---|
  | **Expansionist** | **number of cities − average city size** | **0.600** |
  | | number of cities alone | 0.579 |
  | | Settlers built per city | 0.503 (no signal) |
  | **Aggressive** | **number of civilizations it is at war with right now** | **0.596** |
  | | military units per city | 0.470 (**below chance — do not use it**) |
  | **Builder** | **buildings standing per city − military units per city** | **0.571** |
  | | buildings standing per city alone | 0.555 |
  | | buildings completed in the intervals | 0.499 (no signal) |
  | **Trader** | Caravans built per city | 0.513 (barely a signal) |
  | | buildings standing per city | 0.459 (below chance) |

  Four of those rows are worth reading twice, because each contradicts the obvious play. **Settlers built
  tells you nothing about Expansionist**; what does is having many cities *and* small ones — an expansionist
  spreads thin, so subtract the average city size. **Military per city is anti-correlated with Aggressive**:
  the civs with big armies per city are the small, frightened ones. What identifies an aggressive AI is
  simply how many wars it is in at this moment. For Builder, count the buildings that are *standing* in the
  city tables, not the ones completed in the narrative, and subtract the army. **Trader is not readable**:
  no visible quantity separates it, so rank it on Caravans if you must, expect to be near chance, and do
  not spend time on it.
  Standardise each quantity across the civs of this world (subtract the mean, divide by the spread) before
  combining two of them, rank on the result, and break ties with `=`.
- **Attitude (`love`).** The prompt states the thresholds, so this part is nearly deterministic: at war →
  hostile; allied → friendly; otherwise read the STRATEGIES "Neighbours" line, which names the pairs above
  +250 and below −250 and is informative by omission — a pair it does not name is neutral. A recent
  ceasefire is still hostile; a long peace with no shared enemy is neutral.
- **World rules.** The prompt gives three labels but not their numeric cut-offs. They are: **city growth
  speed** `foodbox` ≤ 82 fast / 83–117 normal / ≥ 118 slowly; **technology speed** `sciencebox` ≤ 82 fast /
  83–117 normal / ≥ 118 slowly; **disaster frequency** ≤ 12 rare or none / 13–40 normal / ≥ 41 frequent.
  Two of the three can be computed rather than guessed: recover `sciencebox` from a root technology's
  printed cost (§5 — divide by 30 in civ2civ3, 28 in classic, and by 2.5 more for a novice AI) and
  `foodbox` from the largest food stock at a given city size (§3 table). For disasters, count the disaster
  sentences per 1,000 city-turns and use the 0.85 / 1.65 cut-offs in §10 (zero or one sentence in the whole
  window means rare or none). Each is a two-minute arithmetic
  job and each is worth a whole question. The characteristic error here is *adjacent-bin*: computing the
  right number and then guessing the label. Compute the number, then read the label off the thresholds
  above. Fall back on the middle bin only when a parameter leaves no trace at all in the window.

**3 (counterfactual arms).** You answer the same quantiles for the same civs in a world where one thing
was changed. Only the intervened civ's designated quantities are scored, and only at the horizons the
question asks. Three rules decide the score.

1. **The control arm is the factual world.** Copy your own factual answer, unchanged.
2. **The placebo arm is normally a negligible edit** (a few gold, +3 on a trait, one cheap tech) and is
   answered as factual — but read its sentence. The city-destruction placebo razes the same *number* of
   (smallest) cities as the real arm, which for a five-city civ is not negligible; answer that one by
   mechanism like any other arm.
3. **The real arm: move the intervened civ's scored quantities by the amount below, and leave everything
   else — every other civ, every other quantity — exactly at your factual numbers.** Moving things the
   intervention does not touch is penalised directly.

**How much each intervention actually moves the target.** Measured on the engine's own re-runs of these
operators, over the cells where the outcome differed from the factual re-run at all. A level quantity is
given as a multiplier on the factual value, an increment as an absolute difference; the spread is the
inter-quartile range, which is also the right width for your interval.

| intervention | quantity | horizon | median | p25 – p75 | moves at all |
|---|---|---|---|---|---|
| **cities destroyed** (`L2-city-kill`), civ2civ3 | n_cities | 60 / 120 / 150 | **×0.40 / ×0.46 / ×0.59** | 0.00–0.75 | always |
| | city_size_total | 60 / 120 / 150 | **×0.23 / ×0.42 / ×0.37** | 0.00–0.64 | always |
| | population | 60 / 120 / 150 | **×0.23 / ×0.34 / ×0.25** | 0.00–0.57 | always |
| **cities destroyed**, **classic** | n_cities | 120 / 150 | ×0.69 / ×0.80 | 0.50–1.06 | always |
| | city_size_total | 120 / 150 | ×0.83 / ×0.72 | 0.54–1.06 | always |
| | population | 120 / 150 | ×0.84 / ×0.69 | 0.56–1.10 | always |
| **research points granted** (`L1-bulbs`) | techs_learned | 60 | **+6** civ2civ3 / **+3** classic | +3 … +8 / +2 … +6 | 68% |
| **map revealed** (`L3-fog`) | score_gain | 60 | **+20** civ2civ3 / **+4** classic | +15 … +36 / −4 … +12 | 96% |
| | n_cities | 60 | ×1.15 / ×1.10 | 0.83–1.30 | 71% |
| **government tech granted** (`L2-gov-indirect`) | score_gain | 60 | **+5** | −6 … +18 | 97% |
| **treasury emptied or filled** (`L1-gold-swing`) | gold | 30 | **×1.09** | 0.34–1.96 | always |
| **+90 Builder** | n_military_units | 60–300 | **×0.84 … ×0.97** | 0.62–1.27 | 90% |
| | city_size_total | 60–300 | ×0.98 … ×1.04 | 0.76–1.25 | 95% |
| **+90 Aggressive** | n_enemies | 120–300 | **×1.5 … ×2.0** | 1.0–2.0 | 40–55% |
| | n_military_units | 120–300 | ×0.88 … ×0.93 | 0.62–1.21 | 70% |
| **+90 Trader** | gold | 60–300 | ×0.97 … ×1.19 | 0.24–3.87 | 88% |
| | techs_learned | 60–300 | −1 … +2 | −7 … +4 | 40% |
| **+90 Expansionist** | n_cities | 120–300 | **×0.91 … ×1.08** | 0.80–1.20 | 66% |
| | city_size_total | 60–300 | ×0.95 … ×1.03 | 0.81–1.12 | 75% |
| **attitude → ±600** (`L4-love`) | n_enemies | 60–300 | ×0.75 … ×1.5 | 0.5–2.0 | 30–55% |
| | n_allies | 120–300 | ×0.50 … ×1.33 | 0.0–2.0 | 21–32% |
| **disasters ×3** (`WP-disasters`) | population | 60–300 | ×0.92 … ×1.03 | 0.74–1.32 | 96% |
| | city_size_total | 60–300 | ×0.97 … ×1.02 | 0.82–1.27 | 90% |

**Use the last column to decide whether to move the median at all.** If the intervention changes the
outcome in ~85% of cells or more (city-kill, fog, gov-indirect, gold-swing, Builder, disasters), apply the
median factor to your median and set the interval to the quoted inter-quartile range. If it changes the
outcome in only 30–70% of cells (Aggressive, the attitude edit, Expansionist, half of Trader's rows), the
intervention does nothing at all in most worlds: **keep your factual median and widen to the quoted range
instead**. Moving a median for an effect that usually does not happen loses more than it gains.

Read that table as three groups, and note that four of its rows contradict what the mechanism suggests:

- **Large and worth moving hard.** Destroying two cities is the biggest operator in the set: it takes the
  civ down to roughly half its cities and a quarter to a half of its citizens, and it does not recover.
  Granting research points is worth about five technologies; revealing the map is worth about nineteen
  score **in civ2civ3**, which is the **Entire Map Known** achievement (+1,000 culture ≈ +22 score) rather
  than any change in the AI's play — and that is why it is worth only about +4 in **classic**, where there
  is no culture for the achievement to grant. The same split matters for destroyed cities: in civ2civ3 the
  victim keeps roughly a quarter to a half of its citizens, in classic about three quarters, because classic
  cities regrow from a smaller food box and there is no culture stock to lose with them. Check the ruleset
  line before applying either row. +90 Builder really does cut the army, by 10–20%.
- **Real but one-sided and unreliable.** +90 Aggressive raises the war count when it does anything, but it
  does nothing at all in half the cells, and it *lowers* the military slightly (more wars means more
  losses, which outweighs the extra building). The attitude edit moves the relation counts in only a third
  of cells. For these, move the median a little and widen the interval a lot rather than committing.
- **Near zero at the median — do not move the median.** ⚠️ Emptying or filling the treasury has **no net
  effect on gold by turn 30** (median ×1.09): the AI spends a windfall and rebuilds from a wipe-out, so the
  right answer is your factual median with a much wider band. ⚠️ **+90 Expansionist does not add cities**
  (median ×0.91 to ×1.08, and the 60-turn median is slightly *below* 1) — land, not personality, binds,
  and the earlier advice to add cities for this operator is wrong. +90 Trader leaves gold near its factual
  median with an enormous right tail. Tripling the disaster rate moves population by a few percent.

The most common failure is timidity: the typical answer moves the median by under 1% of its value where
the truth moved by 5% or more, which scores almost the same as not answering the intervention at all. When
the table says the effect is real, apply it in full.

## 15. Discipline

- Compute rates from the frames with a calculator; never fill a column with one round number for every civ.
- Use the vocabulary and mechanics above, not memories of specific named worlds; do not consult other anchors.
- Follow the output format of each task exactly (line count, order, separators, numbers only, no commentary).
