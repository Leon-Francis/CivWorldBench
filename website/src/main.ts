import "./fonts.css";
import "./style.css";
import "./refinements.css";
import {
  createIcons,
  ArrowUpRight,
  ArrowDown,
  ArrowRight,
  ArrowLeft,
  ZoomIn,
  ZoomOut,
  Play,
  Pause,
  ChevronLeft,
  ChevronRight,
  Menu,
  X,
  Grid3x3,
  Database,
  Github,
  FileText,
  Copy,
  Check,
} from "lucide";
import publication from "../../PUBLICATION.json";
import bibtex from "../../CITATION.bib?raw";

type Point = {
  turn: number;
  score: number;
  cities: number;
  gold: number;
  alive?: boolean;
  min?: number;
  max?: number;
};
type MapPosition = { x: number; y: number; visible: boolean };
type RecordedUnit = {
  id: number;
  type: string;
  owner: number;
  isOwn: boolean;
  screen: MapPosition;
};
type UnitMove = {
  id: number;
  type: string;
  owner: number;
  isOwn: boolean;
  wraps?: boolean;
  from: { screen: MapPosition };
  to: { screen: MapPosition };
};
type Frame = Point & {
  src: string;
  execution: number;
  ownUnits?: number;
  units?: RecordedUnit[];
  unitOwnership?: { owner: number; count: number }[];
  territories?: {
    owner: number;
    path: string;
    borderPath: string;
    tiles: number;
    landTiles: number;
  }[];
  cityOwnership?: { owner: number; count: number }[];
  changes?: {
    fromTurn: number | null;
    moved: UnitMove[];
    appeared: RecordedUnit[];
    disappeared: RecordedUnit[];
    own: { moved: number; appeared: number; disappeared: number };
  };
};
type Plan = {
  id: string;
  label: string;
  color: string;
  finalScore: number;
  trajectory: Point[];
  executionCount: number;
  frames?: Frame[];
};
type Case = {
  id: string;
  title: string;
  nation: string;
  framesSrc: string;
  startTurn: number;
  endTurn: number;
  takeaway: string;
  quotes: { role: string; turn: number; text: string; context?: string }[];
  plans: Plan[];
  factions?: { id: number; name: string; color: string; isOwn: boolean }[];
};
type CaseFrames = {
  schemaVersion: number;
  caseId: string;
  territoryStates: { territories: NonNullable<Frame["territories"]> }[];
  plans: { id: string; frames: (Frame & { territoryState: number })[] }[];
};
type Model = {
  id: string;
  name: string;
  anchors: number;
  reference: boolean;
  swfi: number;
  settings: Record<
    string,
    { values: Record<string, number>; levels: Record<string, number> }
  >;
};
type Evidence = {
  schemaVersion: number;
  benchmark: { anchors: number; worlds: number; models: number };
  settings: { id: string; label: string; description: string }[];
  models: Model[];
  insights: {
    process: {
      id: string;
      name: string;
      anchors: number;
      format: string;
      answerAbove: number;
      answerAboveProcessBelow: number;
      processSkill: number;
      captureCompletion: number;
    }[];
    directionMagnitude: {
      id: string;
      name: string;
      setting: string;
      overTwice: number;
      overTwiceCi95: [number, number];
      nWorlds: number;
      counts: { correct: number };
    }[];
  };
  cases: Case[];
};

const app = document.querySelector<HTMLDivElement>("#app")!;
const icon = (name: string) =>
  `<i data-lucide="${name}" aria-hidden="true"></i>`;
const icons = () =>
  createIcons({
    icons: {
      ArrowUpRight,
      ArrowDown,
      ArrowRight,
      ArrowLeft,
      ZoomIn,
      ZoomOut,
      Play,
      Pause,
      ChevronLeft,
      ChevronRight,
      Menu,
      X,
      Grid3x3,
      Database,
      Github,
      FileText,
      Copy,
      Check,
    },
  });
const esc = (v: unknown) =>
  String(v).replace(
    /[&<>"']/g,
    (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        c
      ]!,
  );
const fmt = (n: number, digits = 2) =>
  Number.isFinite(n) ? n.toFixed(digits) : "N/A";
const signed = (n: number, digits = 2) =>
  `${n > 0 ? "+" : ""}${fmt(n, digits)}`;
const publicUrl = (value: string | undefined) => {
  if (!value) return null;
  try {
    const url = new URL(value);
    return url.protocol === "https:" && !url.username && !url.password
      ? url.href
      : null;
  } catch {
    return null;
  }
};
const $ = <T extends HTMLElement = HTMLElement>(selector: string) =>
  document.querySelector<T>(selector)!;
const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)");
let data: Evidence;
let activeCase = 0;
let activePlan = 0;
let activeTurn = 0;
let timer: ReturnType<typeof setInterval> | null = null;
let tableSetting = "omni-id";
let tableMetric = "swfi";
let tableSearch = "";
let protocol = "table";
let mapZoom = 1;
let showUnitChanges = true;
let showTerritories = true;
let caseRenderVersion = 0;
const caseLoads = new Map<string, Promise<void>>();

const method = [
  {
    level: "L1",
    name: "Association",
    color: "teal",
    question: "What happens next?",
    example: "How many military units will this civilization have in 60 turns?",
    detail:
      "Predict future quantities and events from the observation history.",
    evidence: "Recorded future states",
    signal: "Forecasting quantities & events",
  },
  {
    level: "L2",
    name: "Belief",
    color: "blue",
    question: "What is happening out of sight?",
    example: "Which civilization is more aggressive?",
    detail:
      "Infer hidden personalities and rules from their observable consequences.",
    evidence: "Hidden engine parameters",
    signal: "Inferring traits & rules",
  },
  {
    level: "L3",
    name: "Intervention",
    color: "coral",
    question: "What would an action change?",
    example: "If A gains The Republic, what is its score in 60 turns?",
    detail:
      "Grant The Republic technology, then compare A's score with and without the edit at the same forecast horizon.",
    evidence: "Paired engine rollouts",
    signal: "Predicting action effects",
  },
];

function render() {
  const authors = publication.authors as {
    name: string;
    affiliations: number[];
    corresponding?: boolean;
    url?: string;
  }[];
  const codeUrl = `https://github.com/${publication.github.repo_id}`;
  const paperUrl = publication.paper.url;
  app.innerHTML = `
    <a class="skip-link" href="#method">Skip to research</a>
    <header class="site-header"><a class="brand" href="#">${icon("grid-3x3")}<span>CivWorldBench</span></a>
      <nav aria-label="Main navigation" id="main-nav"><a href="#method">The benchmark</a><a href="#findings">Insights</a><a href="#cases">In the world</a><a href="#results">Results</a><a href="#resources">Resources ${icon("arrow-up-right")}</a></nav>
      <button class="icon-button mobile-menu" aria-label="Open navigation" aria-expanded="false" aria-controls="main-nav">${icon("menu")}</button>
    </header>
    <main>
      <section class="hero" aria-labelledby="hero-title">
        <img class="hero-map" src="./media/hero.webp" width="2600" height="880" fetchpriority="high" alt="A recorded Freeciv world: settlements and rival civilizations across an island map" />
        <div class="hero-shade"></div>
        <div class="hero-content"><p class="eyebrow light">LANGUAGE WORLD MODELS / A STRUCTURED SANDBOX</p><h1 id="hero-title">CivWorldBench</h1>
          <p class="hero-line">Beyond what happens next.</p><p class="hero-description">Understanding hidden states. Predicting the effects of actions.<br class="desktop-only" /> Evaluating world models in a living multi-agent world.</p>
          <div class="hero-actions"><a class="button primary" href="${esc(paperUrl)}">Read the paper ${icon("file-text")}</a><a class="button glass" href="${esc(codeUrl)}">Code ${icon("github")}</a></div>
        </div>
        <div class="hero-footer"><span>FREECIV / RECORDED WORLD STATE / TURN 173</span><span>Observable behavior. Hidden mechanisms.</span></div>
      </section>
      <div class="research-credits"><div class="container"><div class="authors">${authors.map((a) => `<span>${publicUrl(a.url) ? `<a href="${esc(publicUrl(a.url))}">${esc(a.name)}</a>` : esc(a.name)}<sup>${a.affiliations.join(",")}${a.corresponding ? "*" : ""}</sup></span>`).join("")}</div><div class="affiliations">${publication.affiliations.map((a) => `<span><sup>${a.id}</sup>${esc(a.name)}</span>`).join("")}</div><p class="contact-line">* Corresponding author: Kam-Fai Wong <span>Contact: <a href="mailto:${esc(publication.contact.email)}">${esc(publication.contact.email)}</a></span></p></div></div>
      <div class="stats-band"><div class="container stats"><div><strong>475</strong><span>simulated worlds</span></div><div><strong>1,900</strong><span>decision points</span></div><div><strong>11</strong><span>language models</span></div><div><strong>3</strong><span>levels of evaluation</span></div></div></div>
      <section id="method" class="section container"><div class="section-heading"><div><p class="eyebrow">01 / THE BENCHMARK</p><h2>One world.<br />Three kinds of understanding.</h2></div><p class="section-intro">A convincing forecast is only part of a world model. CivWorldBench separates what a model can predict, what it can infer, and what it understands about changing a world.</p></div>
        <div class="method-tabs" role="tablist" aria-label="Evaluation levels">${method.map((m, i) => `<button id="level-${i}" role="tab" aria-selected="${i === 0}" aria-controls="method-panel" tabindex="${i === 0 ? 0 : -1}" class="method-tab ${m.color}" data-level="${i}"><span>${m.level}</span><strong>${m.name}</strong><small>${m.signal}</small></button>`).join("")}</div>
        <div class="method-panel" id="method-panel" role="tabpanel" aria-labelledby="level-0"></div>
        <div class="protocol-strip"><span>Global & player-scoped observation</span><span>In-distribution & held-out rules</span><span>Engine-verifiable ground truth</span></div>
      </section>
      <section id="findings" class="findings-band"><div class="container section"><p class="eyebrow">02 / FOUR INSIGHTS</p><h2>What prediction scores leave out.</h2>
        <article class="insight"><div class="insight-copy"><span class="index teal-text">01</span><h3>Good forecasts do not imply<br class="desktop-only" /> a good model of the world.</h3><p>Models change rank across levels. GLM-5.3 leads Sol on quantity forecasting, while Sol is stronger at hidden-trait inference and intervention direction.</p><p class="implication">Evaluate the structure of capability, not only a single forecasting score.</p></div><figure class="chart-figure"><div class="figure-head"><h4>Same models. Different strengths.</h4><span>Omni-ID</span></div><div id="separation-chart"></div><figcaption>Within-task comparisons; each metric has its own reference. GLM: 1,900 anchors. Sol: 190 anchors.</figcaption></figure></article>
        <article class="insight"><div class="insight-copy"><span class="index coral-text">02</span><h3>The right direction.<br />The wrong magnitude.</h3><p>A model can anticipate an increase and still badly misjudge its size. Even among correct-direction predictions, substantial fractions exceed twice the true effect.</p><p class="implication">Action-conditioned forecasts need effect calibration, beyond sign accuracy.</p></div><figure class="chart-figure"><div class="figure-head"><h4>Severe effect overestimation</h4><span>Omni-ID</span></div><div id="magnitude-chart"></div><figcaption>Fraction exceeding 2× the true effect among correct-direction quantity predictions (n). Whiskers: 95% intervals from 1,000 world-level exponential-weight bootstrap draws. This is a conditional error diagnostic, not overall accuracy; samples differ across models.</figcaption></figure></article>
        <article class="insight"><div class="insight-copy"><span class="index blue-text">03</span><h3>A strong answer can hide<br class="desktop-only" /> a weak reasoning process.</h3><p>Some predictions beat the final-answer reference without beating the intermediate-state reference. Table and Code make those intermediate forecasts independently testable.</p><p class="implication">Verify the process as well as its endpoint. These are inference protocols, not process-supervised training results.</p></div><figure class="chart-figure"><div class="figure-head"><h4>Answers above reference</h4><div class="segmented" aria-label="Reasoning protocol"><button data-protocol="table" aria-pressed="true">Table</button><button data-protocol="code" aria-pressed="false">Code</button></div></div><div class="legend"><span><i class="swatch blue-bg"></i>Process above reference</span><span><i class="swatch hatched"></i>Process at/below reference</span></div><div id="process-chart"></div><figcaption>Equal L1/L3-state and Omni-ID/Omni-OOD means on common valid Free/Table/Code predictions. Four models: 190 anchors each; Sol: 30.</figcaption></figure></article>
      </div></section>
      <section id="cases" class="section container"><div class="section-heading"><div><p class="eyebrow">03 / IN THE WORLD</p><h2>150 turns.<br />Three paths through a world.</h2></div><div class="section-intro"><span class="index coral-text">04</span><p>An early lead can disappear. A defensive failure can accumulate. Two complete recorded games reveal how adviser outcomes diverge over time.</p></div></div>
        <div class="case-tabs" role="tablist" aria-label="Decision case">${data.cases.map((c, i) => `<button id="case-tab-${i}" role="tab" aria-selected="${i === 0}" aria-controls="case-content" tabindex="${i === 0 ? 0 : -1}" data-case="${i}"><span class="case-number">0${i + 1}</span><span><strong>${esc(c.nation)}</strong><small>${esc(c.title)}</small></span>${icon("arrow-up-right")}</button>`).join("")}</div>
        <div id="case-content" role="tabpanel" aria-labelledby="case-tab-0"></div>
        <p class="evidence-note">These two worlds were selected retrospectively to illustrate contrasting adviser outcomes. Across all 76 worlds, mean final gains over None are +6.51 for Flash (95% interval [-3.97, 16.62]) and +3.32 for Kimi ([-8.59, 14.04]). These world-bootstrap intervals include zero; the examples do not represent average gains.</p>
      </section>
      <section id="results" class="results-band"><div class="section container"><div class="section-heading"><div><p class="eyebrow">04 / FROZEN RESULTS</p><h2>A capability profile,<br />not just a rank.</h2></div><p class="section-intro">Compare the paper's evaluated models across information settings. The Sandbox-World Forecasting Index (SWFI) summarizes the three settings; individual metrics preserve the differences it compresses.</p></div>
        <div class="result-controls"><label>Setting<select id="result-setting">${data.settings.map((s) => `<option value="${s.id}">${esc(s.label)}</option>`).join("")}</select></label><label>Sort by<select id="result-sort"><option value="swfi">SWFI (all settings)</option><option value="quantity">Quantity</option><option value="trait">Trait</option><option value="direction">Direction</option><option value="capture">Capture</option></select></label><label class="search-label">Model<input id="result-search" type="search" placeholder="Filter models" autocomplete="off" /></label></div>
        <p id="setting-description" class="setting-description"></p><div class="table-scroll" tabindex="0" role="region" aria-label="Model results, horizontally scrollable"><table><thead><tr><th scope="col">Model</th><th scope="col">SWFI</th><th scope="col" class="teal-text">Quantity</th><th scope="col" class="teal-text">Event</th><th scope="col" class="blue-text">Trait</th><th scope="col" class="blue-text">Rules</th><th scope="col" class="coral-text">Direction</th><th scope="col" class="coral-text">Capture</th></tr></thead><tbody id="result-body"></tbody></table></div>
        <p id="result-count" class="small-note" aria-live="polite"></p><p class="small-note">Higher is better. Zero is the task-specific reference, not a universal chance score. Direction and Capture measure different aspects of intervention prediction. The expert row is a rules-informed model harness, not a human participant.</p>
      </div></section>
      <section id="resources" class="section container resources"><div><p class="eyebrow">05 / RESEARCH RESOURCES</p><h2>From a question<br />to verifiable evidence.</h2><p>Task definitions, evaluation protocols, scoring code, and the research behind CivWorldBench.</p></div><div class="resource-list"><a class="resource-row" href="${esc(codeUrl)}">${icon("github")}<div><h3>Code</h3><p>Evaluation, scoring, and engine tools.</p></div>${icon("arrow-up-right")}</a><div class="resource-row dataset-resource">${icon("database")}<div><h3>Dataset</h3><p>Benchmark questions, responses, and process evidence.</p></div><span class="resource-state">In preparation</span></div><a class="resource-row" href="${esc(paperUrl)}">${icon("file-text")}<div><h3>Paper</h3><p>${esc(publication.title)}</p></div>${icon("arrow-up-right")}</a></div><div class="citation"><div class="figure-head"><h3>Citation</h3><button id="copy-citation" class="icon-button" title="Copy BibTeX" aria-label="Copy BibTeX">${icon("copy")}</button></div><pre><code id="bibtex">${esc(bibtex)}</code></pre></div></section>
    </main><footer><div class="container footer-inner"><a class="brand" href="#">${icon("grid-3x3")}CivWorldBench</a><p>Structured worlds. Verifiable predictions.</p><a href="./media/ATTRIBUTION.md">Asset credits ${icon("arrow-up-right")}</a></div></footer>`;
  renderMethod(0);
  renderSeparation();
  renderMagnitude();
  renderProcess();
  renderCase();
  renderResults();
  bind();
  icons();
}

function renderMethod(index: number) {
  const m = method[index];
  $("#method-panel").setAttribute("aria-labelledby", `level-${index}`);
  $("#method-panel").innerHTML =
    `<div class="method-question ${m.color}"><span class="eyebrow">${m.level} / ${m.name.toUpperCase()}</span><h3>${m.question}</h3><p>${m.detail}</p></div><div class="method-example"><span class="tiny-label">EXAMPLE TASK</span><blockquote>${m.example}</blockquote>${index === 2 ? '<p class="intervention-baseline">Compare with its own no-intervention forecast</p><div class="intervention-branches"><span>No edit</span><span>Real edit</span><span>Placebo</span></div>' : ""}<div class="ground-truth">${icon("check")}<span>Verified against <strong>${m.evidence.toLowerCase()}</strong></span></div></div>`;
  document.querySelectorAll<HTMLButtonElement>("[data-level]").forEach((b) => {
    const on = Number(b.dataset.level) === index;
    b.setAttribute("aria-selected", String(on));
    b.tabIndex = on ? 0 : -1;
  });
  icons();
}

function renderSeparation() {
  const selected = [
    data.models.find((m) => m.name.includes("GLM"))!,
    data.models.find((m) => m.name.includes("Sol"))!,
  ];
  const metrics = [
    ["quantity", "Quantity / skill"],
    ["trait", "Trait / adjusted accuracy"],
    ["direction", "Direction / adjusted agreement"],
  ];
  $("#separation-chart").innerHTML =
    `<div class="legend"><span><i class="swatch teal-bg"></i>GLM-5.3</span><span><i class="swatch blue-bg"></i>GPT-5.6 Sol</span></div>${metrics
      .map(
        ([metric, label]) =>
          `<div class="compare-group"><div class="bar-label">${label}</div>${selected
            .map((m, i) => {
              const v = m.settings["omni-id"].values[metric];
              return `<div class="compare-row"><span class="bar-origin"></span><span class="compare-bar ${i ? "blue-bg" : "teal-bg"}" style="width:${(Math.abs(v) / 0.5) * 76}%;${v < 0 ? "transform:translateX(-100%)" : ""}"></span><strong>${signed(v)}</strong></div>`;
            })
            .join("")}</div>`,
      )
      .join("")}`;
}

function renderMagnitude() {
  const rows = data.insights.directionMagnitude
    .filter((m) => m.setting === "omni-id")
    .sort((a, b) => a.overTwice - b.overTwice);
  $("#magnitude-chart").innerHTML =
    `<div class="error-direction"><span>${icon("arrow-left")} Lower is better</span><span>More overestimation ${icon("arrow-right")}</span></div><div class="error-axis"><span>0%</span><span>25%</span><span>50%</span></div>${rows.map((m) => `<div class="error-row" data-rate="${m.overTwice}"><span title="${m.counts.correct} correct-direction predictions; ${m.nWorlds} eligible worlds">${esc(m.name)}<small>n = ${m.counts.correct}</small></span><div class="error-track" role="img" aria-label="${esc(m.name)}: ${fmt(m.overTwice * 100, 1)}% overestimation rate, 95% interval ${fmt(m.overTwiceCi95[0] * 100, 1)} to ${fmt(m.overTwiceCi95[1] * 100, 1)} percent"><b class="ci-whisker" style="left:${(m.overTwiceCi95[0] / 0.5) * 100}%;width:${((m.overTwiceCi95[1] - m.overTwiceCi95[0]) / 0.5) * 100}%"></b><i class="error-dot" style="left:${(m.overTwice / 0.5) * 100}%"></i></div><strong>${fmt(m.overTwice * 100, 1)}%</strong></div>`).join("")}<p class="error-axis-title">Severe overestimation rate (%)</p>`;
}

function renderProcess() {
  const rows = data.insights.process.filter((m) => m.format === protocol);
  $("#process-chart").innerHTML = rows
    .map(
      (m) =>
        `<div class="process-row"><div><span>${esc(m.name)}</span><strong>${fmt(m.answerAbove * 100, 1)}%</strong></div><div class="process-track"><span class="blue-bg" style="width:${(m.answerAbove - m.answerAboveProcessBelow) * 100}%"></span><span class="hatched" style="width:${m.answerAboveProcessBelow * 100}%" title="${fmt(m.answerAboveProcessBelow * 100, 1)}% of all matched predictions"></span></div><small>${fmt(m.answerAboveProcessBelow * 100, 1)}% with process at/below reference</small></div>`,
    )
    .join("");
  document
    .querySelectorAll<HTMLButtonElement>("[data-protocol]")
    .forEach((b) =>
      b.setAttribute("aria-pressed", String(b.dataset.protocol === protocol)),
    );
}

function loadCaseFrames(c: Case): Promise<void> {
  const cached = caseLoads.get(c.id);
  if (cached) return cached;
  const pending = (async () => {
    const embedded = document.getElementById(`case-data-${c.id}`);
    const chunk: CaseFrames = embedded
      ? JSON.parse(embedded.textContent!)
      : await fetch(`./${c.framesSrc}`).then((response) => {
          if (!response.ok) throw new Error("Case unavailable");
          return response.json();
        });
    if (
      chunk.schemaVersion !== 2 ||
      chunk.caseId !== c.id ||
      chunk.plans.length !== c.plans.length
    ) {
      throw new Error("Case evidence mismatch");
    }
    const framesByPlan = c.plans.map((plan) => {
      const source = chunk.plans.find((item) => item.id === plan.id);
      if (!source || source.frames.length !== c.endTurn - c.startTurn + 1) {
        throw new Error("Incomplete recorded turns");
      }
      return source.frames.map((frame, index) => {
        const state = chunk.territoryStates[frame.territoryState];
        if (
          frame.turn !== c.startTurn + index ||
          !state ||
          frame.score !== plan.trajectory[index]?.score
        ) {
          throw new Error("Recorded state mismatch");
        }
        return { ...frame, territories: state.territories };
      });
    });
    c.plans.forEach((plan, index) => {
      plan.frames = framesByPlan[index];
    });
  })().catch((error) => {
    caseLoads.delete(c.id);
    throw error;
  });
  caseLoads.set(c.id, pending);
  return pending;
}

function renderCase() {
  const c = data.cases[activeCase];
  const version = ++caseRenderVersion;
  const content = $("#case-content");
  content.setAttribute("aria-labelledby", `case-tab-${activeCase}`);
  document.querySelectorAll<HTMLButtonElement>("[data-case]").forEach((b) => {
    const on = Number(b.dataset.case) === activeCase;
    b.setAttribute("aria-selected", String(on));
    b.tabIndex = on ? 0 : -1;
  });
  if (c.plans.every((plan) => plan.frames)) {
    content.removeAttribute("aria-busy");
    renderLoadedCase();
    return;
  }
  content.setAttribute("aria-busy", "true");
  content.innerHTML =
    '<div class="case-load-state" role="status">Loading recorded turns...</div>';
  loadCaseFrames(c)
    .then(() => {
      if (version !== caseRenderVersion) return;
      content.removeAttribute("aria-busy");
      renderLoadedCase();
    })
    .catch(() => {
      if (version !== caseRenderVersion) return;
      content.removeAttribute("aria-busy");
      content.innerHTML = `<div class="case-load-state"><p role="alert">The recorded turns could not be loaded.</p><button class="button" id="retry-case">Retry ${icon("arrow-right")}</button></div>`;
      $("#retry-case").addEventListener("click", renderCase);
      icons();
    });
}

function renderLoadedCase() {
  const c = data.cases[activeCase];
  mapZoom = 1;
  $("#case-content").innerHTML =
    `<div class="case-story"><h3>${esc(c.title)}</h3><p>${esc(c.takeaway)}</p></div>
    <div class="case-workspace"><div class="replay"><div class="replay-toolbar"><div class="segmented plan-select" aria-label="World-model adviser">${c.plans.map((p, i) => `<button data-plan="${i}" aria-pressed="${i === activePlan}">${esc(p.label)}</button>`).join("")}</div><span class="recorded-label">WORLD-MODEL ADVISER / RECORDED GLOBAL STATE</span></div>
    <div class="map-ownership-bar"><div id="faction-legend" class="faction-legend" aria-label="Civilizations"></div><label class="territory-toggle"><input type="checkbox" id="show-territories" ${showTerritories ? "checked" : ""}/>Territories</label></div>
    <div class="map-viewport"><div class="map-stage" id="map-stage" role="region" aria-label="Recorded world map" tabindex="-1"><div class="map-canvas" id="map-canvas"><img id="case-map" width="1408" height="792" alt=""/><svg id="territory-overlay" viewBox="0 0 1408 792" aria-hidden="true"></svg><svg id="unit-changes-overlay" viewBox="0 0 1408 792" aria-hidden="true"></svg></div></div><span class="map-turn" id="map-turn"></span><div class="map-zoom"><button class="icon-button" id="zoom-out" aria-label="Zoom out on map" title="Zoom out on map" disabled>${icon("zoom-out")}</button><button class="icon-button" id="zoom-in" aria-label="Zoom in on map" title="Zoom in on map">${icon("zoom-in")}</button></div></div>
    <div class="replay-controls"><button class="icon-button" id="play-replay" aria-label="Play recorded turns" title="Play recorded turns">${icon("play")}</button><button class="icon-button step-control" id="prev-turn" aria-label="Previous turn" title="Previous turn">${icon("chevron-left")}</button><input id="turn-slider" type="range" aria-label="Recorded turn" min="${c.startTurn}" max="${c.endTurn}" value="${c.startTurn + activeTurn}" step="1"/><button class="icon-button step-control" id="next-turn" aria-label="Next turn" title="Next turn">${icon("chevron-right")}</button><output id="turn-output" for="turn-slider"></output></div>
    <div class="frame-stats" id="frame-stats" aria-live="polite"></div><div class="unit-changes-panel"><div class="unit-changes-heading"><span id="unit-changes-turn"></span><label><input type="checkbox" id="show-unit-changes" ${showUnitChanges ? "checked" : ""}/>Unit changes</label></div><div id="unit-changes-summary" aria-live="polite"></div></div></div>
    <aside class="case-dialogue"><p class="eyebrow">SELECTED DELIBERATIONS</p>${c.quotes
      .slice(0, 4)
      .map(
        (q) =>
          `<blockquote class="quote ${q.role.startsWith("None") ? "" : "adviser"}"><span>${esc(q.role)} / T${q.turn}</span><p>${esc(q.text)}</p>${q.context ? `<p class="quote-context">${esc(q.context)}</p>` : ""}</blockquote>`,
      )
      .join("")}<p class="small-note">Excerpts are model claims. Maps and scores show recorded engine outcomes.</p></aside></div>
    <div class="case-outcomes"><figure><div class="figure-head"><h4>How the games unfold</h4><span>Civilization score</span></div><div id="case-chart"></div><figcaption>One recorded execution per adviser, from the same T60 state. The Flash-Next acting agent is shared across None, Flash and Kimi; only the world-model adviser differs. Maps show global engine state, including information withheld from the agent.</figcaption></figure><div class="outcome-summary"><p class="eyebrow">AT TURN ${c.endTurn}</p><div id="outcome-values"></div><p class="small-note">Final civilization score and surviving cities. Case-level gains are not population estimates.</p></div></div>`;
  document.querySelectorAll<HTMLButtonElement>("[data-plan]").forEach((b) =>
    b.addEventListener("click", () => {
      activePlan = Number(b.dataset.plan);
      renderFrame();
    }),
  );
  $("#turn-slider").addEventListener("input", (e) => {
    stopPlayback();
    activeTurn = Number((e.target as HTMLInputElement).value) - c.startTurn;
    renderFrame();
  });
  $("#prev-turn").addEventListener("click", () => {
    stopPlayback();
    activeTurn = Math.max(0, activeTurn - 1);
    renderFrame();
  });
  $("#next-turn").addEventListener("click", () => {
    stopPlayback();
    activeTurn = Math.min(c.endTurn - c.startTurn, activeTurn + 1);
    renderFrame();
  });
  $("#play-replay").addEventListener("click", togglePlayback);
  $("#show-unit-changes").addEventListener("change", (e) => {
    showUnitChanges = (e.target as HTMLInputElement).checked;
    renderUnitChanges(c.plans[activePlan].frames![activeTurn]);
  });
  $("#show-territories").addEventListener("change", (e) => {
    showTerritories = (e.target as HTMLInputElement).checked;
    renderTerritories(c.plans[activePlan].frames![activeTurn]);
  });
  $("#zoom-in").addEventListener("click", () =>
    setMapZoom(Math.min(3, mapZoom + 0.5)),
  );
  $("#zoom-out").addEventListener("click", () =>
    setMapZoom(Math.max(1, mapZoom - 0.5)),
  );
  const mapStage = $("#map-stage");
  let drag: { x: number; y: number; left: number; top: number } | null = null;
  mapStage.addEventListener("pointerdown", (e) => {
    if (mapZoom === 1 || e.pointerType !== "mouse" || e.button !== 0) return;
    e.preventDefault();
    drag = {
      x: e.clientX,
      y: e.clientY,
      left: mapStage.scrollLeft,
      top: mapStage.scrollTop,
    };
    mapStage.setPointerCapture(e.pointerId);
    mapStage.classList.add("dragging");
  });
  mapStage.addEventListener("pointermove", (e) => {
    if (!drag) return;
    mapStage.scrollLeft = drag.left - (e.clientX - drag.x);
    mapStage.scrollTop = drag.top - (e.clientY - drag.y);
  });
  const endDrag = () => {
    drag = null;
    mapStage.classList.remove("dragging");
  };
  mapStage.addEventListener("pointerup", endDrag);
  mapStage.addEventListener("pointercancel", endDrag);
  $("#outcome-values").innerHTML = c.plans
    .map((p) => {
      const final = p.trajectory[p.trajectory.length - 1];
      return `<div class="outcome-row"><div><i class="swatch" style="background:${p.color}"></i><span>${esc(p.label)}</span><strong>${fmt(p.finalScore, 0)}</strong></div><small>${final.cities} ${final.cities === 1 ? "city" : "cities"}${final.alive === false ? " / eliminated" : ""}</small></div>`;
    })
    .join("");
  renderCaseChart();
  renderFrame();
  $(".unit-changes-panel").insertAdjacentHTML(
    "beforeend",
    '<p class="map-evidence-note">Territories follow saved tile ownership. Unclaimed tiles have no territory tint. Arrows join recorded positions, not routes.</p>',
  );
  icons();
}

function renderFrame() {
  const c = data.cases[activeCase],
    p = c.plans[activePlan];
  const f = p.frames?.[activeTurn];
  if (!f) return;
  const img = $<HTMLImageElement>("#case-map");
  img.src = `./${f.src}`;
  img.alt = `${p.label}, turn ${f.turn}: ${f.cities} cities${f.ownUnits === undefined ? "" : `, ${f.ownUnits} units in your civilization`}, civilization score ${f.score}. Recorded execution ${f.execution}.`;
  $("#map-turn").textContent =
    `TURN ${f.turn}${f.alive === false ? " / ELIMINATED" : ""}`;
  $("#turn-output").textContent = `${f.turn} / ${c.endTurn}`;
  $<HTMLInputElement>("#turn-slider").value = String(f.turn);
  $("#frame-stats").innerHTML =
    `<span><strong>${f.score}</strong> score</span><span><strong>${f.cities}</strong> cities</span><span><strong>${f.gold}</strong> gold</span>${f.ownUnits === undefined ? "" : `<span><strong>${f.ownUnits}</strong> units</span>`}`;
  renderUnitChanges(f);
  renderTerritories(f);
  $<HTMLButtonElement>("#prev-turn").disabled = activeTurn === 0;
  $<HTMLButtonElement>("#next-turn").disabled = f.turn === c.endTurn;
  document
    .querySelectorAll<HTMLButtonElement>("[data-plan]")
    .forEach((b) =>
      b.setAttribute(
        "aria-pressed",
        String(Number(b.dataset.plan) === activePlan),
      ),
    );
  const cursor = document.querySelector<SVGLineElement>("#chart-cursor");
  if (cursor) {
    const x = 45 + (activeTurn / (c.endTurn - c.startTurn)) * 620;
    cursor.setAttribute("x1", String(x));
    cursor.setAttribute("x2", String(x));
  }
  if (p.frames?.[activeTurn + 1]) {
    const next = new Image();
    next.src = `./${p.frames[activeTurn + 1].src}`;
  }
}

function renderTerritories(frame: Frame) {
  const factions = data.cases[activeCase].factions ?? [];
  const territories = frame.territories ?? [];
  const overlay = document.querySelector<SVGSVGElement>("#territory-overlay")!;
  overlay.style.display = showTerritories ? "" : "none";
  overlay.innerHTML = territories
    .map((territory) => {
      const faction = factions.find((f) => f.id === territory.owner);
      if (!faction) return "";
      return `<g data-owner="${territory.owner}" style="color:${esc(faction.color)}"><path class="territory-fill" d="${esc(territory.path)}"/><path class="territory-border-halo" d="${esc(territory.borderPath)}"/><path class="territory-border" d="${esc(territory.borderPath)}"/></g>`;
    })
    .join("");
  $("#faction-legend").innerHTML = factions
    .map((faction) => {
      const count = frame.cityOwnership?.find(
        (c) => c.owner === faction.id,
      )?.count;
      const active =
        faction.isOwn ||
        count === undefined ||
        count > 0 ||
        territories.some((t) => t.owner === faction.id && t.tiles > 0) ||
        frame.unitOwnership?.some((u) => u.owner === faction.id && u.count > 0);
      if (!active) return "";
      return `<div class="faction-key" data-owner="${faction.id}" style="--faction-color:${esc(faction.color)}"><i aria-hidden="true"></i><span><strong>${esc(faction.name)}${faction.isOwn ? " <small>(you)</small>" : ""}</strong>${count === undefined ? "" : `<span title="Cities across the recorded world">${count} ${count === 1 ? "city" : "cities"}</span>`}</span></div>`;
    })
    .join("");
}

function setMapZoom(zoom: number) {
  const stage = $("#map-stage");
  const ratio = zoom / mapZoom;
  const left = (stage.scrollLeft + stage.clientWidth / 2) * ratio;
  const top = (stage.scrollTop + stage.clientHeight / 2) * ratio;
  mapZoom = zoom;
  $("#map-canvas").style.width = `${zoom * 100}%`;
  stage.scrollLeft = left - stage.clientWidth / 2;
  stage.scrollTop = top - stage.clientHeight / 2;
  stage.tabIndex = zoom > 1 ? 0 : -1;
  stage.classList.toggle("zoomed", zoom > 1);
  stage.setAttribute("aria-label", `Recorded world map, ${zoom * 100}% zoom`);
  $<HTMLButtonElement>("#zoom-out").disabled = zoom === 1;
  $<HTMLButtonElement>("#zoom-in").disabled = zoom === 3;
}

function renderUnitChanges(frame: Frame) {
  const overlay = document.querySelector<SVGSVGElement>(
    "#unit-changes-overlay",
  )!;
  const changes = frame.changes;
  $("#unit-changes-turn").textContent =
    changes?.fromTurn == null
      ? "INITIAL STATE"
      : `T${changes.fromTurn} → T${frame.turn} / YOUR UNITS`;
  $("#unit-changes-summary").innerHTML =
    changes?.fromTurn == null
      ? `<span>${frame.ownUnits ?? ""} units in your civilization</span>`
      : `<span class="change-moved"><i></i><strong>${changes.own.moved}</strong> moved</span><span class="change-appeared"><i></i><strong>${changes.own.appeared}</strong> appeared</span><span class="change-disappeared"><i></i><strong>${changes.own.disappeared}</strong> no longer recorded</span>`;
  overlay.replaceChildren();
  overlay.style.display = showUnitChanges ? "" : "none";
  if (!changes || changes.fromTurn === null || !showUnitChanges) return;
  const point = (p: MapPosition) => ({ x: p.x * 1408, y: p.y * 792 });
  const ring = (p: MapPosition, kind: string, id: number) => {
    const { x, y } = point(p);
    return `<circle class="unit-change ${kind}" data-unit-id="${id}" cx="${x}" cy="${y}" r="20"/>`;
  };
  // Lines join consecutive snapshots; the saved states do not specify travelled routes.
  overlay.innerHTML =
    `<defs><marker id="move-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#f4f7a0"/></marker></defs>` +
    changes.moved
      .filter((u) => u.isOwn)
      .map((u) => {
        const a = point(u.from.screen),
          b = point(u.to.screen);
        const line =
          u.from.screen.visible && u.to.screen.visible && !u.wraps
            ? `<path class="unit-move" data-unit-id="${u.id}" d="M ${a.x} ${a.y} L ${b.x} ${b.y}" marker-end="url(#move-arrow)"/>`
            : "";
        return (
          line + (u.to.screen.visible ? ring(u.to.screen, "moved", u.id) : "")
        );
      })
      .join("") +
    changes.appeared
      .filter((u) => u.isOwn && u.screen.visible)
      .map((u) => ring(u.screen, "appeared", u.id))
      .join("") +
    changes.disappeared
      .filter((u) => u.isOwn && u.screen.visible)
      .map((u) => {
        const { x, y } = point(u.screen);
        return `<path class="unit-change disappeared" data-unit-id="${u.id}" d="M ${x - 13} ${y - 13} L ${x + 13} ${y + 13} M ${x + 13} ${y - 13} L ${x - 13} ${y + 13}"/>`;
      })
      .join("");
}

function renderCaseChart() {
  const c = data.cases[activeCase];
  const values = c.plans.flatMap((p) => p.trajectory.map((v) => v.score));
  const min = Math.floor((Math.min(...values) - 5) / 10) * 10;
  const max = Math.ceil((Math.max(...values) + 5) / 10) * 10;
  const x = (t: number) =>
    45 + ((t - c.startTurn) / (c.endTurn - c.startTurn)) * 620;
  const y = (v: number) => 235 - ((v - min) / (max - min)) * 210;
  const line = (pts: Point[]) =>
    pts
      .map(
        (v, i) =>
          `${i ? "L" : "M"}${x(v.turn).toFixed(1)},${y(v.score).toFixed(1)}`,
      )
      .join(" ");
  $("#case-chart").innerHTML =
    `<div class="legend">${c.plans.map((p) => `<span><i class="swatch" style="background:${p.color}"></i>${esc(p.label)}</span>`).join("")}</div><svg class="trajectory-chart" viewBox="0 0 700 270" role="img" aria-label="Civilization scores from turn 60 to 210, one recorded execution per adviser"><title>150 turns: None, Flash and Kimi civilization scores</title>${Array.from(
      { length: 5 },
      (_, i) => {
        const v = min + ((max - min) * i) / 4;
        return `<line x1="45" x2="665" y1="${y(v)}" y2="${y(v)}" class="gridline"/><text x="32" y="${y(v) + 4}" text-anchor="end">${fmt(v, 0)}</text>`;
      },
    ).join(
      "",
    )}${[60, 100, 140, 180, 210].map((t) => `<text x="${x(t)}" y="257" text-anchor="middle">T${t}</text>`).join("")}${c.plans.map((p) => `<path data-plan-line="${esc(p.id)}" d="${line(p.trajectory)}" stroke="${p.color}" fill="none" stroke-width="3"/>`).join("")}<line id="chart-cursor" x1="45" x2="45" y1="20" y2="235" stroke="#546568" stroke-dasharray="4 4"/></svg>`;
}

function stopPlayback() {
  if (timer) clearInterval(timer);
  timer = null;
  const button = document.querySelector<HTMLButtonElement>("#play-replay");
  if (button) {
    button.innerHTML = icon("play");
    button.setAttribute("aria-label", "Play recorded turns");
    button.title = "Play recorded turns";
    icons();
  }
}
function togglePlayback() {
  if (timer) {
    stopPlayback();
    return;
  }
  const c = data.cases[activeCase];
  if (activeTurn >= c.endTurn - c.startTurn) {
    activeTurn = 0;
    renderFrame();
  }
  $("#play-replay").innerHTML = icon("pause");
  $("#play-replay").setAttribute("aria-label", "Pause recorded turns");
  $("#play-replay").title = "Pause recorded turns";
  icons();
  timer = setInterval(
    () => {
      activeTurn += 1;
      renderFrame();
      if (activeTurn >= c.endTurn - c.startTurn) stopPlayback();
    },
    reducedMotion.matches ? 1600 : 850,
  );
}

function renderResults() {
  const models = data.models
    .filter((m) => m.name.toLowerCase().includes(tableSearch.toLowerCase()))
    .sort((a, b) =>
      tableMetric === "swfi"
        ? b.swfi - a.swfi
        : b.settings[tableSetting].values[tableMetric] -
          a.settings[tableSetting].values[tableMetric],
    );
  const metricKeys = [
    "quantity",
    "event",
    "trait",
    "rules",
    "direction",
    "capture",
  ];
  $("#result-body").innerHTML = models.length
    ? models
        .map(
          (m) =>
            `<tr><th scope="row">${esc(m.name)}${m.reference ? '<span class="reference-tag">reference harness</span>' : ""}</th><td class="score-cell">${signed(m.swfi, 1)}</td>${metricKeys.map((k) => `<td class="${m.settings[tableSetting].values[k] < 0 ? "negative" : ""}">${signed(m.settings[tableSetting].values[k], k === "event" ? 3 : 2)}</td>`).join("")}</tr>`,
        )
        .join("")
    : '<tr><td colspan="8" class="empty-state">No models match this filter.</td></tr>';
  $("#setting-description").textContent = data.settings.find(
    (s) => s.id === tableSetting,
  )!.description;
  $("#result-count").textContent =
    `${models.length} evaluated model / reference rows. Opus, Sol and the expert harness use a 190-anchor subset; other models use 1,900 anchors.`;
}

function tabKeys(selector: string) {
  document.querySelectorAll<HTMLButtonElement>(selector).forEach((b) =>
    b.addEventListener("keydown", (e) => {
      if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(e.key)) return;
      e.preventDefault();
      const buttons = [
        ...document.querySelectorAll<HTMLButtonElement>(selector),
      ];
      let next = buttons.indexOf(b) + (e.key === "ArrowLeft" ? -1 : 1);
      if (e.key === "Home") next = 0;
      if (e.key === "End") next = buttons.length - 1;
      const target = buttons[(next + buttons.length) % buttons.length];
      target.click();
      target.focus();
    }),
  );
}
function bind() {
  $("#copy-citation").addEventListener("click", async () => {
    const value = $("#bibtex").textContent!;
    try {
      if (!navigator.clipboard) throw new Error("Clipboard unavailable");
      await navigator.clipboard.writeText(value);
    } catch {
      const selection = getSelection();
      const range = document.createRange();
      range.selectNodeContents($("#bibtex"));
      selection?.removeAllRanges();
      selection?.addRange(range);
      if (!document.execCommand("copy")) return;
    }
    $("#copy-citation").setAttribute("aria-label", "BibTeX copied");
    $("#copy-citation").innerHTML = icon("check");
    icons();
  });
  document
    .querySelectorAll<HTMLButtonElement>("[data-level]")
    .forEach((b) =>
      b.addEventListener("click", () => renderMethod(Number(b.dataset.level))),
    );
  document.querySelectorAll<HTMLButtonElement>("[data-protocol]").forEach((b) =>
    b.addEventListener("click", () => {
      protocol = b.dataset.protocol!;
      renderProcess();
    }),
  );
  document.querySelectorAll<HTMLButtonElement>("[data-case]").forEach((b) =>
    b.addEventListener("click", () => {
      stopPlayback();
      activeCase = Number(b.dataset.case);
      activeTurn = 0;
      activePlan = 0;
      renderCase();
    }),
  );
  tabKeys("[data-level]");
  tabKeys("[data-case]");
  $("#result-setting").addEventListener("change", (e) => {
    tableSetting = (e.target as HTMLSelectElement).value;
    renderResults();
  });
  $("#result-sort").addEventListener("change", (e) => {
    tableMetric = (e.target as HTMLSelectElement).value;
    renderResults();
  });
  $("#result-search").addEventListener("input", (e) => {
    tableSearch = (e.target as HTMLInputElement).value;
    renderResults();
  });
  $(".mobile-menu").addEventListener("click", () => {
    const open = $(".mobile-menu").getAttribute("aria-expanded") !== "true";
    $(".mobile-menu").setAttribute("aria-expanded", String(open));
    $(".mobile-menu").setAttribute(
      "aria-label",
      open ? "Close navigation" : "Open navigation",
    );
    $("#main-nav").classList.toggle("open", open);
  });
  document.querySelectorAll("#main-nav a").forEach((a) =>
    a.addEventListener("click", () => {
      $("#main-nav").classList.remove("open");
      $(".mobile-menu").setAttribute("aria-expanded", "false");
      $(".mobile-menu").setAttribute("aria-label", "Open navigation");
    }),
  );
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) stopPlayback();
  });
  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((e) => {
        if (e.isIntersecting) {
          document
            .querySelectorAll("#main-nav a")
            .forEach((a) =>
              a.classList.toggle(
                "active",
                a.getAttribute("href") === `#${e.target.id}`,
              ),
            );
        }
      });
    },
    { rootMargin: "-15% 0px -60% 0px" },
  );
  document
    .querySelectorAll("main section[id]")
    .forEach((section) => observer.observe(section));
}

app.innerHTML =
  '<div class="loading-state"><strong>CivWorldBench</strong><p>Loading recorded evidence...</p></div>';
const embeddedEvidence = document.getElementById("evidence-data");
const evidencePromise: Promise<Evidence> = embeddedEvidence
  ? Promise.resolve(JSON.parse(embeddedEvidence.textContent!))
  : fetch("./evidence.json").then((response) => {
      if (!response.ok) throw new Error("Evidence unavailable");
      return response.json();
    });
evidencePromise
  .then((e) => {
    data = e;
    render();
  })
  .catch(() => {
    app.innerHTML =
      '<main class="loading-state"><h1>CivWorldBench</h1><p>The evidence file could not be loaded.</p><button onclick="location.reload()">Retry</button></main>';
  });
