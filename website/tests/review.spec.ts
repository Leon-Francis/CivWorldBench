import { test, expect, type Page, type TestInfo } from "@playwright/test";
import { mkdirSync, readFileSync } from "node:fs";
import { resolve } from "node:path";

const evidence = JSON.parse(
  readFileSync(resolve("public/evidence.json"), "utf8"),
);
const sectionScreenshotStyle =
  ".site-header, .skip-link { visibility: hidden !important; }";

type ScreenPosition = { x: number; y: number; visible: boolean };
type Unit = {
  id: number;
  owner: number;
  isOwn: boolean;
  screen: ScreenPosition;
};
type Faction = { id: number; name: string; color: string; isOwn: boolean };
type Frame = {
  turn: number;
  src: string;
  score: number;
  alive: boolean;
  ownUnits: number;
  unitOwnership: { owner: number; count: number }[];
  territories: {
    owner: number;
    path: string;
    borderPath: string;
    tiles: number;
    landTiles: number;
  }[];
  cityOwnership: { owner: number; count: number }[];
  changes: {
    fromTurn: number | null;
    own: { moved: number; appeared: number; disappeared: number };
    moved: {
      id: number;
      isOwn: boolean;
      wraps: boolean;
      from: { screen: ScreenPosition };
      to: { screen: ScreenPosition };
    }[];
    appeared: Unit[];
    disappeared: Unit[];
  };
};

function caseFrames(caseIndex: number, planIndex: number): Frame[] {
  const c = evidence.cases[caseIndex];
  const chunk = JSON.parse(
    readFileSync(resolve("public", c.framesSrc), "utf8"),
  );
  expect(chunk.schemaVersion).toBe(2);
  expect(chunk.caseId).toBe(c.id);
  const plan = chunk.plans.find(
    (item: { id: string }) => item.id === c.plans[planIndex].id,
  );
  return plan.frames.map((frame: Frame & { territoryState: number }) => ({
    ...frame,
    ...chunk.territoryStates[frame.territoryState],
  }));
}

async function selectTurn(page: Page, turn: number) {
  await page
    .locator("#turn-slider")
    .evaluate((element: HTMLInputElement, value) => {
      element.value = String(value);
      element.dispatchEvent(new Event("input", { bubbles: true }));
    }, turn);
  await expect(page.locator("#turn-output")).toHaveText(`${turn} / 210`);
}

function screenshotPath(info: TestInfo, filename: string) {
  const directory = resolve(
    process.env.CIVWORLD_SCREENSHOT_DIR ||
      "test-results/screenshots",
    info.title.replace(/[^a-z0-9]+/gi, "-").toLowerCase(),
  );
  mkdirSync(directory, { recursive: true });
  return resolve(directory, filename);
}

async function expectRecordedTerritories(
  page: Page,
  frame: Frame,
  factions: Faction[],
) {
  const overlay = page.locator("#territory-overlay");
  await expect(overlay).toHaveAttribute("viewBox", "0 0 1408 792");
  await expect(overlay.locator("g[data-owner]")).toHaveCount(
    frame.territories.length,
  );
  for (const territory of frame.territories) {
    const group = overlay.locator(`g[data-owner="${territory.owner}"]`);
    await expect(group.locator(".territory-fill")).toHaveAttribute(
      "d",
      territory.path,
    );
    await expect(group.locator(".territory-border")).toHaveAttribute(
      "d",
      territory.borderPath,
    );
    const border = await group
      .locator(".territory-border")
      .evaluate((path: SVGPathElement) => ({
        length: path.getTotalLength(),
        vectorEffect: getComputedStyle(path).vectorEffect,
      }));
    expect(Number.isFinite(border.length)).toBeTruthy();
    if (territory.borderPath.trim()) expect(border.length).toBeGreaterThan(0);
    else expect(border.length).toBe(0);
    expect(border.vectorEffect).toBe("non-scaling-stroke");
  }
  const activeFactions = factions.filter(
    (faction) =>
      faction.isOwn ||
      frame.cityOwnership.some(
        (city) => city.owner === faction.id && city.count > 0,
      ) ||
      frame.territories.some(
        (territory) => territory.owner === faction.id && territory.tiles > 0,
      ) ||
      frame.unitOwnership.some(
        (units) => units.owner === faction.id && units.count > 0,
      ),
  );
  await expect(page.locator("#faction-legend .faction-key")).toHaveCount(
    activeFactions.length,
  );
  for (const faction of activeFactions) {
    const legend = page.locator(
      `#faction-legend .faction-key[data-owner="${faction.id}"]`,
    );
    const cities = frame.cityOwnership.find(
      (city) => city.owner === faction.id,
    );
    expect(cities).toBeDefined();
    await expect(legend.locator("strong")).toHaveText(
      `${faction.name}${faction.isOwn ? " (you)" : ""}`,
    );
    await expect(legend.locator(":scope > span > span")).toHaveText(
      `${cities!.count} ${cities!.count === 1 ? "city" : "cities"}`,
    );
  }
}

async function screenshotDifference(page: Page, before: Buffer, after: Buffer) {
  return page.evaluate(
    async ({ before, after }) => {
      async function pixels(encoded: string) {
        const image = await createImageBitmap(
          await (await fetch(`data:image/png;base64,${encoded}`)).blob(),
        );
        const canvas = document.createElement("canvas");
        canvas.width = image.width;
        canvas.height = image.height;
        const context = canvas.getContext("2d")!;
        context.drawImage(image, 0, 0);
        return context.getImageData(0, 0, image.width, image.height);
      }
      const a = await pixels(before),
        b = await pixels(after);
      if (a.width !== b.width || a.height !== b.height)
        throw new Error("Map geometry changed while toggling territories");
      let changed = 0;
      for (let i = 0; i < a.data.length; i += 4) {
        const delta =
          Math.abs(a.data[i] - b.data[i]) +
          Math.abs(a.data[i + 1] - b.data[i + 1]) +
          Math.abs(a.data[i + 2] - b.data[i + 2]);
        if (delta > 12) changed++;
      }
      return {
        changedPixels: changed,
        changedFraction: changed / (a.width * a.height),
      };
    },
    { before: before.toString("base64"), after: after.toString("base64") },
  );
}

async function expectRecordedUnits(page: Page, frame: Frame) {
  await expect(page.locator("#frame-stats")).toContainText(
    `${frame.ownUnits} units`,
  );
  await expect(page.locator("#case-map")).toHaveAttribute(
    "alt",
    new RegExp(`${frame.ownUnits} units in your civilization`),
  );
  const changes = frame.changes;
  if (changes.fromTurn === null) {
    await expect(page.locator("#unit-changes-turn")).toHaveText(
      "INITIAL STATE",
    );
    await expect(page.locator(".unit-move, .unit-change")).toHaveCount(0);
    return;
  }
  await expect(page.locator("#unit-changes-turn")).toContainText(
    `T${changes.fromTurn}`,
  );
  await expect(page.locator("#unit-changes-turn")).toContainText(
    `T${frame.turn}`,
  );
  for (const [kind, label] of [
    ["moved", "moved"],
    ["appeared", "appeared"],
    ["disappeared", "no longer recorded"],
  ] as const) {
    const own = changes[kind];
    expect(changes.own[kind]).toBe(own.length);
    await expect(
      page.locator(`#unit-changes-summary .change-${kind}`),
    ).toHaveText(`${own.length} ${label}`);
  }

  const expectedIds: Record<string, number[]> = {
    ".unit-move": changes.moved
      .filter(
        (unit) =>
          unit.from.screen.visible && unit.to.screen.visible && !unit.wraps,
      )
      .map((unit) => unit.id),
    ".unit-change.moved": changes.moved
      .filter((unit) => unit.to.screen.visible)
      .map((unit) => unit.id),
    ".unit-change.appeared": changes.appeared
      .filter((unit) => unit.screen.visible)
      .map((unit) => unit.id),
    ".unit-change.disappeared": changes.disappeared
      .filter((unit) => unit.screen.visible)
      .map((unit) => unit.id),
  };
  for (const [selector, ids] of Object.entries(expectedIds)) {
    expect(
      await page
        .locator(`#unit-changes-overlay ${selector}`)
        .evaluateAll((elements) =>
          elements
            .map((element) => Number(element.getAttribute("data-unit-id")))
            .sort((a, b) => a - b),
        ),
    ).toEqual(ids.sort((a, b) => a - b));
  }
}

for (const viewport of [
  { width: 1440, height: 1000 },
  { width: 390, height: 844 },
  { width: 360, height: 740 },
]) {
  test(`recorded evidence and layout at ${viewport.width}px`, async ({
    page,
    baseURL,
  }, info) => {
    await page.setViewportSize(viewport);
    const errors: string[] = [];
    const external: string[] = [];
    page.on("pageerror", (error) => errors.push(error.message));
    page.on("request", (request) => {
      if (new URL(request.url()).origin !== new URL(baseURL!).origin)
        external.push(request.url());
    });
    await page.goto("/");
    await expect(
      page.getByRole("heading", { name: "CivWorldBench", exact: true }),
    ).toBeVisible();
    await page.evaluate(() => document.fonts.ready);
    expect(
      await page
        .locator(".hero-map")
        .evaluate(
          (img: HTMLImageElement) => img.complete && img.naturalWidth > 1000,
        ),
    ).toBeTruthy();
    expect(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= innerWidth + 1,
      ),
    ).toBeTruthy();
    const hero = await page.locator(".hero").boundingBox();
    const heroActions = await page.locator(".hero-actions").boundingBox();
    const footer = await page.locator(".hero-footer").boundingBox();
    expect(heroActions!.y + heroActions!.height).toBeLessThan(footer!.y);
    expect(hero!.y + hero!.height).toBeLessThan(viewport.height);
    await page.evaluate(async () => {
      await Promise.all(Array.from(document.images, (image) => image.decode()));
      await new Promise<void>((resolve) =>
        requestAnimationFrame(() => requestAnimationFrame(() => resolve())),
      );
    });
    await page.screenshot({
      path: screenshotPath(info, "page.png"),
      fullPage: true,
    });
    await page.locator('[data-level="2"]').click();
    await expect(page.locator("#method-panel blockquote")).toHaveText(
      "If A gains The Republic, what is its score in 60 turns?",
    );
    await expect(page.locator("#method-panel")).toContainText(
      "Compare with its own no-intervention forecast",
    );
    await expect(page.locator(".intervention-branches span")).toHaveText([
      "No edit",
      "Real edit",
      "Placebo",
    ]);
    await expect(page.locator("#method-panel")).toContainText(
      "Paired engine rollouts",
      { ignoreCase: true },
    );
    await page.locator("#method-panel").screenshot({
      path: screenshotPath(info, "intervention.png"),
    });
    await page.locator('[data-level="2"]').press("ArrowLeft");
    await expect(page.locator("#method-panel")).toContainText(
      "hidden personalities",
    );
    const magnitude = evidence.insights.directionMagnitude
      .filter((model: { setting: string }) => model.setting === "omni-id")
      .sort(
        (a: { overTwice: number }, b: { overTwice: number }) =>
          a.overTwice - b.overTwice,
      );
    await expect(page.locator(".error-direction")).toContainText(
      "Lower is better",
    );
    await expect(page.locator(".error-direction")).toContainText(
      "More overestimation",
    );
    await expect(page.locator("#magnitude-chart .error-dot")).toHaveCount(12);
    await expect(page.locator("#magnitude-chart .ci-whisker")).toHaveCount(12);
    expect(
      await page
        .locator("#magnitude-chart .error-row")
        .evaluateAll((rows) =>
          rows.map((row) => Number(row.getAttribute("data-rate"))),
        ),
    ).toEqual(magnitude.map((model: { overTwice: number }) => model.overTwice));
    for (let index = 0; index < magnitude.length; index++) {
      const model = magnitude[index];
      const row = page.locator("#magnitude-chart .error-row").nth(index);
      await expect(row).toContainText(model.name);
      await expect(row).toContainText(`n = ${model.counts.correct}`);
      await expect(row.locator(".error-track")).toHaveAttribute(
        "aria-label",
        `${model.name}: ${(model.overTwice * 100).toFixed(1)}% overestimation rate, 95% interval ${(model.overTwiceCi95[0] * 100).toFixed(1)} to ${(model.overTwiceCi95[1] * 100).toFixed(1)} percent`,
      );
      const ci = await row
        .locator(".ci-whisker")
        .evaluate((element: HTMLElement) => ({
          left: parseFloat(element.style.left),
          width: parseFloat(element.style.width),
        }));
      expect(ci.left).toBeCloseTo(model.overTwiceCi95[0] * 200, 4);
      expect(ci.width).toBeCloseTo(
        (model.overTwiceCi95[1] - model.overTwiceCi95[0]) * 200,
        4,
      );
    }
    await page.locator("#magnitude-chart").screenshot({
      path: screenshotPath(info, "overestimation.png"),
    });
    const table = await page.locator("#process-chart").innerText();
    await page.getByRole("button", { name: "Code", exact: true }).click();
    expect(await page.locator("#process-chart").innerText()).not.toBe(table);
    for (let ci = 0; ci < 2; ci++) {
      await page.locator(`[data-case="${ci}"]`).click();
      const c = evidence.cases[ci];
      expect(c.startTurn).toBe(60);
      expect(c.endTurn).toBe(210);
      expect(c.plans.map((plan: { id: string }) => plan.id)).toEqual([
        "none",
        "flash",
        "kimi",
      ]);
      await expect(page.locator("#case-chart [data-plan-line]")).toHaveCount(3);
      await expect(page.locator("#case-chart path")).toHaveCount(3);
      if (viewport.width <= 600) {
        const renderedTickSize = await page
          .locator("#case-chart svg text")
          .first()
          .evaluate((text: SVGTextElement) => {
            const matrix = text.getScreenCTM()!;
            return parseFloat(getComputedStyle(text).fontSize) * matrix.d;
          });
        expect(renderedTickSize).toBeGreaterThanOrEqual(8);
      }
      await expect(page.locator("#cases")).not.toContainText(
        /Original plan|Self-review|Opus-assisted|Sol-assisted|four executions|four repeats|execution 3|mean score|mean trajectory|mean of/i,
      );
      if (c.id === "defensive-collapse") {
        const contexts = c.quotes
          .filter((quote: { context?: string }) => Boolean(quote.context))
          .map((quote: { context: string }) => quote.context);
        expect(contexts.join(" ")).toMatch(/reject/i);
        await expect(page.locator("#case-content .quote-context")).toHaveText(
          contexts,
        );
        await expect(page.locator("#case-content .quote-context")).toBeVisible();
      }
      for (let pi = 0; pi < 3; pi++) {
        await expect(page.locator(`[data-plan="${pi}"]`)).toContainText(
          c.plans[pi].label,
        );
        await page.locator(`[data-plan="${pi}"]`).click();
        await page.locator("#turn-slider").focus();
        await page.locator("#turn-slider").press("End");
        const frames = caseFrames(ci, pi);
        expect(frames.map((frame) => frame.turn)).toEqual(
          Array.from({ length: 151 }, (_, index) => index + 60),
        );
        expect(c.plans[pi].executionCount).toBe(1);
        expect(c.plans[pi].trajectory).toHaveLength(151);
        const curve = page.locator(
          `#case-chart [data-plan-line="${c.plans[pi].id}"]`,
        );
        await expect(curve).toHaveAttribute("fill", "none");
        const frame = frames.at(-1)!;
        expect(c.plans[pi].finalScore).toBe(frame.score);
        await expect(page.locator("#case-map")).toHaveAttribute(
          "src",
          `./${frame.src}`,
        );
        await expect(page.locator("#frame-stats")).toContainText(
          `${frame.score} score`,
        );
        await expectRecordedUnits(page, frame);
        await expectRecordedTerritories(page, frame, c.factions);
        await expect
          .poll(() =>
            page
              .locator("#case-map")
              .evaluate(
                (img: HTMLImageElement) =>
                  img.complete && img.naturalWidth === 1408,
              ),
          )
          .toBeTruthy();
      }
      await page.locator("#case-content").screenshot({
        path: screenshotPath(info, `${c.id}.png`),
        style: sectionScreenshotStyle,
      });
    }
    await page.locator("#case-content").screenshot({
      path: screenshotPath(info, "case.png"),
      style: sectionScreenshotStyle,
    });
    expect(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= innerWidth + 1,
      ),
    ).toBeTruthy();
    await expect(page.locator("#results thead th")).toHaveCount(8);
    await expect(
      page.getByRole("columnheader", { name: "Anchors", exact: true }),
    ).toHaveCount(0);
    await page.locator("#result-setting").selectOption("omni-ood");
    await page.locator("#result-search").fill("Sol");
    await expect(page.locator("#result-body tr")).toHaveCount(1);
    await expect(
      page.locator("#result-body tr > th, #result-body tr > td"),
    ).toHaveCount(8);
    const sol = evidence.models.find((m: { name: string }) =>
      m.name.includes("Sol"),
    );
    await expect(page.locator("#result-body")).toContainText(
      sol.settings["omni-ood"].values.trait.toFixed(2),
    );
    await page.locator("#result-search").fill("no-such-model");
    await expect(page.locator("#result-body")).toContainText("No models match");
    await expect(page.locator("#result-body td")).toHaveAttribute(
      "colspan",
      "8",
    );
    expect(errors).toEqual([]);
    expect(external).toEqual([]);
  });
}

test("playback, keyboard controls and reduced motion", async ({ page }) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto("/");
  await page.locator("#cases").scrollIntoViewIfNeeded();
  await expect(page.locator("#turn-output")).toHaveText("60 / 210");
  await page.locator("#turn-slider").focus();
  await page.locator("#turn-slider").press("ArrowRight");
  await expect(page.locator("#turn-output")).toHaveText("61 / 210");
  await page
    .getByRole("button", { name: "Play recorded turns", exact: true })
    .click();
  await expect(page.locator("#turn-output")).not.toHaveText("61 / 210");
  await page
    .getByRole("button", { name: "Pause recorded turns", exact: true })
    .click();
  const stopped = await page.locator("#turn-output").innerText();
  await page.waitForTimeout(1800);
  await expect(page.locator("#turn-output")).toHaveText(stopped);
  await page.locator('[data-case="0"]').focus();
  await page.locator('[data-case="0"]').press("ArrowRight");
  await expect(page.locator("#turn-output")).toHaveText("60 / 210");
  await expect(page.locator('[data-case="1"]')).toHaveAttribute(
    "aria-selected",
    "true",
  );
});

test("lazy case loading keeps the latest selection and records late outcomes", async ({
  page,
}, info) => {
  const requested: string[] = [];
  let releaseFirst!: () => void;
  let firstRequested!: () => void;
  const firstStarted = new Promise<void>((resolve) => {
    firstRequested = resolve;
  });
  const holdFirst = new Promise<void>((resolve) => {
    releaseFirst = resolve;
  });
  await page.route("**/media/*/frames.json", async (route) => {
    const path = new URL(route.request().url()).pathname;
    requested.push(path);
    if (path.endsWith(evidence.cases[0].framesSrc)) {
      firstRequested();
      await holdFirst;
    }
    await route.continue();
  });

  try {
    await page.goto("/", { waitUntil: "domcontentloaded" });
    await firstStarted;
    expect(requested).toEqual([`/${evidence.cases[0].framesSrc}`]);
    await page.locator('[data-case="1"]').click();
    await expect(page.locator("#turn-output")).toHaveText("60 / 210");
    const secondStart = caseFrames(1, 0)[0];
    await expect(page.locator("#case-map")).toHaveAttribute(
      "src",
      `./${secondStart.src}`,
    );
    const firstResponse = page.waitForResponse((response) =>
      response.url().endsWith(evidence.cases[0].framesSrc),
    );
    releaseFirst();
    await firstResponse;
    await page.waitForLoadState("networkidle");
    await expect(page.locator('[data-case="1"]')).toHaveAttribute(
      "aria-selected",
      "true",
    );
    await expect(page.locator("#case-map")).toHaveAttribute(
      "src",
      `./${secondStart.src}`,
    );

    await page.locator('[data-plan="1"]').click();
    await selectTurn(page, 186);
    const eliminated = caseFrames(1, 1).find((frame) => frame.turn === 186)!;
    expect(eliminated.score).toBe(91);
    expect(eliminated.alive).toBe(false);
    await expect(page.locator("#frame-stats")).toContainText("91 score");
    await expectRecordedTerritories(
      page,
      eliminated,
      evidence.cases[1].factions,
    );
    const own = evidence.cases[1].factions.find(
      (faction: Faction) => faction.isOwn,
    );
    await expect(
      page.locator(`#faction-legend [data-owner="${own.id}"]`),
    ).toContainText("0 cities");
    await selectTurn(page, 187);
    await expect(page.locator("#frame-stats")).toContainText("85 score");
    await page.locator("#case-content").screenshot({
      path: screenshotPath(info, "recorded-elimination.png"),
      style: sectionScreenshotStyle,
    });

    await page.locator('[data-case="0"]').click();
    await expect(page.locator("#turn-output")).toHaveText("60 / 210");
    await page.locator('[data-plan="1"]').click();
    await selectTurn(page, 183);
    const lateLead = caseFrames(0, 1).find((frame) => frame.turn === 183)!;
    expect(lateLead.score).toBe(145);
    await expect(page.locator("#frame-stats")).toContainText("145 score");
    await expect(page.locator("#case-map")).toHaveAttribute(
      "src",
      `./${lateLead.src}`,
    );
    await expectRecordedTerritories(page, lateLead, evidence.cases[0].factions);
    await page.locator("#case-content").screenshot({
      path: screenshotPath(info, "recorded-late-lead.png"),
      style: sectionScreenshotStyle,
    });
    await page.locator('[data-case="1"]').click();
    await expect(page.locator("#turn-output")).toHaveText("60 / 210");
    expect(requested).toEqual(
      evidence.cases.map((c: { framesSrc: string }) => `/${c.framesSrc}`),
    );
  } finally {
    releaseFirst();
  }
});

test("a failed case download can be retried without blocking the page", async ({
  page,
}) => {
  let attempts = 0;
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.route(`**/${evidence.cases[0].framesSrc}`, async (route) => {
    attempts++;
    if (attempts === 1) {
      await route.fulfill({ status: 503, body: "Temporarily unavailable" });
    } else {
      await route.continue();
    }
  });
  await page.goto("/");
  await expect(page.locator("#retry-case")).toBeVisible();
  await expect(
    page.getByRole("heading", { name: "CivWorldBench", exact: true }),
  ).toBeVisible();
  await page.locator('[data-level="2"]').click();
  await expect(page.locator("#method-panel blockquote")).toHaveText(
    "If A gains The Republic, what is its score in 60 turns?",
  );
  await page.locator("#retry-case").click();
  await expect(page.locator("#turn-output")).toHaveText("60 / 210");
  await expect(page.locator("#case-map")).toHaveAttribute(
    "src",
    `./${caseFrames(0, 0)[0].src}`,
  );
  await expect(page.locator("#retry-case")).toHaveCount(0);
  expect(attempts).toBe(2);
  expect(errors).toEqual([]);
});

for (const touch of [false, true]) {
  test(`unit evidence, zoom and ${touch ? "touch" : "mouse"} map navigation`, async ({
    browser,
  }, info) => {
    const context = await browser.newContext({
      viewport: touch
        ? { width: 390, height: 844 }
        : { width: 1440, height: 1000 },
      isMobile: touch,
      hasTouch: touch,
    });
    const page = await context.newPage();
    await page.goto("http://127.0.0.1:4173/");
    await page.locator('[data-plan="0"]').click();
    const frames = caseFrames(0, 0);
    await expectRecordedUnits(page, frames[0]);
    await expectRecordedTerritories(
      page,
      frames[0],
      evidence.cases[0].factions,
    );
    const frame: Frame = frames
      .slice(1)
      .sort(
        (a: Frame, b: Frame) =>
          b.changes.own.moved +
          b.changes.own.appeared +
          b.changes.own.disappeared -
          a.changes.own.moved -
          a.changes.own.appeared -
          a.changes.own.disappeared,
      )[0];
    await page
      .locator("#turn-slider")
      .evaluate((element: HTMLInputElement, turn) => {
        element.value = String(turn);
        element.dispatchEvent(new Event("input", { bubbles: true }));
      }, frame.turn);
    await expectRecordedUnits(page, frame);
    await expectRecordedTerritories(page, frame, evidence.cases[0].factions);
    expect(frame.changes.own.moved).toBeGreaterThan(0);
    const summary = await page.locator("#unit-changes-summary").innerText();
    const frameSrc = await page.locator("#case-map").getAttribute("src");
    await page.locator("#show-unit-changes").uncheck();
    await expect(page.locator("#unit-changes-overlay")).toBeHidden();
    await expect(page.locator("#unit-changes-summary")).toHaveText(summary, {
      useInnerText: true,
    });
    await page.locator("#show-unit-changes").check();
    await expectRecordedUnits(page, frame);

    await expect(page.locator("#show-territories")).toBeChecked();
    await expect
      .poll(() =>
        page
          .locator("#case-map")
          .evaluate(
            (image: HTMLImageElement) =>
              image.complete && image.naturalWidth === 1408,
          ),
      )
      .toBeTruthy();
    const withTerritories = await page
      .locator(".map-viewport")
      .screenshot({ path: screenshotPath(info, "territories-on.png") });
    await page.locator("#show-territories").uncheck();
    await expect(page.locator("#territory-overlay")).toBeHidden();
    await expect(page.locator("#case-map")).toHaveAttribute("src", frameSrc!);
    await expect(page.locator("#turn-slider")).toHaveValue(String(frame.turn));
    const withoutTerritories = await page
      .locator(".map-viewport")
      .screenshot({ path: screenshotPath(info, "territories-off.png") });
    const difference = await screenshotDifference(
      page,
      withTerritories,
      withoutTerritories,
    );
    expect(difference.changedFraction).toBeGreaterThan(0.01);
    console.log(
      `${touch ? "Mobile" : "Desktop"} territory layer: ${JSON.stringify(difference)}`,
    );

    await page.locator('[data-plan="1"]').click();
    await expect(page.locator("#show-territories")).not.toBeChecked();
    await expect(page.locator("#territory-overlay")).toBeHidden();
    const otherPlanFrame = caseFrames(0, 1).find(
      (item) => item.turn === frame.turn,
    )!;
    await expectRecordedTerritories(
      page,
      otherPlanFrame,
      evidence.cases[0].factions,
    );
    await page.locator('[data-case="1"]').click();
    await expect(page.locator("#show-territories")).not.toBeChecked();
    await expect(page.locator("#territory-overlay")).toBeHidden();
    await expectRecordedTerritories(
      page,
      caseFrames(1, 0)[0],
      evidence.cases[1].factions,
    );
    await page.locator('[data-case="0"]').click();
    await page.locator('[data-plan="0"]').click();
    await page
      .locator("#turn-slider")
      .evaluate((element: HTMLInputElement, turn) => {
        element.value = String(turn);
        element.dispatchEvent(new Event("input", { bubbles: true }));
      }, frame.turn);
    await expect(page.locator("#show-territories")).not.toBeChecked();
    await page.locator("#show-territories").check();
    await expectRecordedTerritories(page, frame, evidence.cases[0].factions);

    const stage = page.locator("#map-stage");
    await stage.scrollIntoViewIfNeeded();
    const before = await stage.boundingBox();
    await expect(page.locator("#zoom-out")).toBeDisabled();
    await page.locator("#zoom-in").click();
    await page.locator("#zoom-in").click();
    const after = await stage.boundingBox();
    expect(after!.width).toBeCloseTo(before!.width, 1);
    expect(after!.height).toBeCloseTo(before!.height, 1);
    const mapSize = await page.locator("#map-canvas").boundingBox();
    expect(mapSize!.width).toBeCloseTo(after!.width * 2, 1);
    expect(mapSize!.height / mapSize!.width).toBeCloseTo(792 / 1408, 3);
    const territorySize = await page
      .locator("#territory-overlay")
      .boundingBox();
    expect(territorySize!.width).toBeCloseTo(mapSize!.width, 1);
    expect(territorySize!.height).toBeCloseTo(mapSize!.height, 1);
    await expect(page.locator("#territory-overlay")).toHaveCSS(
      "overflow",
      "hidden",
    );
    await expectRecordedTerritories(page, frame, evidence.cases[0].factions);
    await expect(page.locator("#case-map")).toHaveAttribute("src", frameSrc!);
    await expect(stage).toHaveAttribute("tabindex", "0");
    const initialScroll = await stage.evaluate((element) => element.scrollLeft);
    const x = after!.x + after!.width * 0.6;
    const y = after!.y + after!.height * 0.5;
    if (touch) {
      const cdp = await context.newCDPSession(page);
      await cdp.send("Input.dispatchTouchEvent", {
        type: "touchStart",
        touchPoints: [{ x, y }],
      });
      for (let step = 1; step <= 5; step++) {
        await cdp.send("Input.dispatchTouchEvent", {
          type: "touchMove",
          touchPoints: [{ x: x - 12 * step, y }],
        });
      }
      await cdp.send("Input.dispatchTouchEvent", {
        type: "touchEnd",
        touchPoints: [],
      });
      await cdp.detach();
    } else {
      await page.mouse.move(x, y);
      await page.mouse.down();
      await page.mouse.move(x - 60, y, { steps: 5 });
      await page.mouse.up();
    }
    await expect
      .poll(() => stage.evaluate((element) => element.scrollLeft))
      .toBeGreaterThan(initialScroll + 20);
    await stage.screenshot({ path: screenshotPath(info, "units-zoomed.png") });
    await page.locator("#zoom-out").click();
    await page.locator("#zoom-out").click();
    await expect(page.locator("#zoom-out")).toBeDisabled();
    await expect(stage).toHaveAttribute("tabindex", "-1");
    expect(await stage.evaluate((element) => element.scrollLeft)).toBe(0);
    expect(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= innerWidth + 1,
      ),
    ).toBeTruthy();
    await context.close();
  });
}

for (const width of [1440, 390]) {
  test(`public resources and author information ${width}`, async ({ page }, info) => {
    await page.setViewportSize({ width, height: 1000 });
    await page.goto("/");
    await expect(page.locator(".research-credits .authors > span")).toHaveCount(10);
    await expect(page.locator(".research-credits .affiliations > span")).toHaveCount(6);
    await expect(page.locator(".contact-line")).toContainText("angli@link.cuhk.edu.hk");
    await expect(page.locator(".dataset-resource")).toContainText("In preparation");
    await expect(page.locator(".dataset-resource a")).toHaveCount(0);
    await expect(page.locator('a[href*="huggingface.co"], .evidence-download')).toHaveCount(0);
    await expect(page.locator("#resources")).not.toContainText("PUBLIC RELEASE PENDING");
    await expect(page.locator("#bibtex")).toContainText("Wong, Kam-Fai");
    await expect(page.locator("#bibtex")).toContainText("eprint = {2610.16450}");
    await expect(page.locator("#bibtex")).toContainText("primaryClass = {cs.CL}");
    await expect(page.locator(".hero-actions a").first()).toHaveAttribute("href", "https://arxiv.org/abs/2610.16450");
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1)).toBeTruthy();
    await page.screenshot({ path: screenshotPath(info, "public-top.png") });
    await page.locator(".research-credits").screenshot({ path: screenshotPath(info, "authors.png") });
    await page.locator("#resources").screenshot({ path: screenshotPath(info, "resources.png") });
  });
}
