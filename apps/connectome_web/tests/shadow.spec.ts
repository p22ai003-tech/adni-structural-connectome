import { expect, test } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";

const genericSections = [
  ["demographics", "Demographics"],
  ["novel-findings", "Novel Findings"],
  ["global-dti", "Global DTI"],
  ["local-roi-dti", "Local / ROI DTI"],
  ["global-graph", "Global Graph"],
  ["node-metrics", "Node Metrics"],
  ["coupling", "Coupling"],
  ["coupling-aal", "Coupling-AAL"],
  ["brain-age", "Brain Age"],
  ["lr-sr", "LR / SR"],
  ["edr-exceptions", "EDR Exceptions"],
  ["ml-diagnostics", "ML Diagnostics"],
  ["delay", "Delay"],
  ["advanced", "Advanced"],
  ["network-analysis", "Network Analysis"],
  ["functional-pending", "Functional Pending"],
] as const;

const specialisedStatus: Record<string, string> = {
  demographics: "analysis subjects",
  "novel-findings": "FDR-supported catalog rows",
  "coupling-aal": "tested AAL3 regions",
  "ml-diagnostics": "prediction tasks",
  "network-analysis": "network–metric summaries",
};

test("overview, LR-SR, matrix and pipeline routes render without browser errors", async ({
  page,
}) => {
  const browserErrors: string[] = [];
  page.on("pageerror", (error) => browserErrors.push(error.message));
  page.on("console", (message) => {
    if (message.type() === "error") {
      browserErrors.push(message.text());
    }
  });

  await page.goto("/next/#/");
  await expect(
    page.getByRole("heading", {
      name: /From microstructure to network geometry/i,
    }),
  ).toBeVisible();
  await expect(page.getByText("530", { exact: true }).first()).toBeVisible();

  await page.goto("/next/#/section/lr-sr-analysis");
  await expect(
    page.getByRole("heading", {
      name: "LR–SR Structural Pathway Analysis",
    }),
  ).toBeVisible();
  await expect(
    page.getByRole("heading", { name: "TRACT-LENGTH DISTRIBUTIONS" }),
  ).toBeVisible();
  await expect(page.locator(".js-plotly-plot").first()).toBeVisible({
    timeout: 60_000,
  });
  await expect(
    page
      .getByText("AD has higher exception strength than MCI.", {
        exact: true,
      })
      .first(),
  ).toBeVisible({ timeout: 60_000 });
  await expect(
    page
      .getByRole("columnheader", { name: "Pairwise Holm p" })
      .first(),
  ).toBeVisible();
  await expect(
    page.getByText("POPULATION MEDIAN", { exact: true }),
  ).toBeVisible();
  await expect(
    page.getByText("0 < L ≤ 94.41 mm", { exact: true }),
  ).toBeVisible();
  await expect(
    page.getByRole("columnheader", { name: "Low outliers" }),
  ).toHaveCount(0);
  await expect(
    page.getByRole("columnheader", { name: "High outliers" }),
  ).toHaveCount(0);
  await expect(
    page.getByRole("columnheader", { name: "STRENGTH MEASURES" }),
  ).toBeVisible();
  await expect(
    page.getByRole("columnheader", {
      name: "EXCEPTION BURDEN / RATE",
    }),
  ).toBeVisible();

  await page.goto("/next/#/section/sc-matrix-viewer");
  await expect(
    page.getByRole("heading", {
      name: "Structural Connectome Matrix Viewer",
    }),
  ).toBeVisible();
  await expect(page.getByText("Positive edges", { exact: true })).toBeVisible({
    timeout: 30_000,
  });
  await expect(page.locator(".js-plotly-plot").first()).toBeVisible();

  await page.goto("/next/#/pipeline");
  await expect(
    page.getByRole("heading", { name: "530-Subject Pipeline Monitor" }),
  ).toBeVisible();
  await expect(page.getByText("Pretract complete", { exact: true })).toBeVisible(
    { timeout: 30_000 },
  );
  await expect(
    page.getByText("Existing AAL3 complete", { exact: true }),
  ).toBeVisible();

  expect(browserErrors).toEqual([]);
});

test("every generic analysis route resolves its complete output catalog", async ({
  page,
}) => {
  const browserErrors: string[] = [];
  page.on("pageerror", (error) => browserErrors.push(error.message));
  page.on("console", (message) => {
    if (message.type() === "error") {
      browserErrors.push(message.text());
    }
  });

  for (const [sectionId, heading] of genericSections) {
    await page.goto(`/next/#/section/${sectionId}`);
    await expect(
      page.getByRole("heading", { name: heading, exact: true }),
    ).toBeVisible();
    await expect(
      page.getByText(
        specialisedStatus[sectionId] ?? "auditable outputs",
        { exact: true },
      ),
    ).toBeVisible();
    await expect(page.getByText("Indexing section outputs…")).toBeHidden({
      timeout: 30_000,
    });
    await expect(page.getByRole("alert")).toHaveCount(0);
  }

  expect(browserErrors).toEqual([]);
});

test("specialised ML, network, coupling-AAL and findings workspaces expose their scientific contracts", async ({
  page,
}) => {
  await page.goto("/next/#/section/ml-diagnostics");
  await expect(
    page.getByRole("heading", { name: "Cross-validated performance" }),
  ).toBeVisible();
  await expect(page.getByText("prediction tasks", { exact: true })).toBeVisible();
  await expect(page.locator(".js-plotly-plot").first()).toBeVisible({
    timeout: 60_000,
  });

  await page.goto("/next/#/section/network-analysis");
  await expect(
    page.getByRole("heading", { name: "Network-resolved distributions" }),
  ).toBeVisible();
  await expect(page.getByText("Mapping boundary:", { exact: true })).toBeVisible();
  await expect(page.locator(".js-plotly-plot").first()).toBeVisible({
    timeout: 60_000,
  });

  await page.goto("/next/#/section/coupling-aal");
  await expect(
    page.getByRole("heading", { name: "AAL3 regional contrast ranking" }),
  ).toBeVisible();
  await expect(page.getByText("BH-FDR q < 0.05", { exact: true })).toBeVisible();
  await expect(page.getByText("166", { exact: true }).first()).toBeVisible();

  await page.goto("/next/#/section/novel-findings");
  await expect(
    page.getByRole("heading", {
      name: "Literature-grounded finding cards",
    }),
  ).toBeVisible();
  await expect(page.locator(".finding-card")).toHaveCount(4);
  await expect(page.getByText("Claim boundary:", { exact: true })).toBeVisible();
});

test("LR-SR group colors, medians, consolidated contrasts and feature bullets are preserved", async ({
  page,
}) => {
  await page.goto("/next/#/section/lr-sr-analysis");
  await expect(page.locator(".js-plotly-plot").first()).toBeVisible({
    timeout: 60_000,
  });
  await expect(
    page.getByText("AD has higher exception strength than MCI.", {
      exact: true,
    }).first(),
  ).toBeVisible();
  await expect(page.locator(".table-cell-list li").first()).toBeVisible();
  const plotState = await page.locator(".js-plotly-plot").evaluateAll(
    (nodes) =>
      nodes.map((node) => {
        const plot = node as HTMLElement & {
          data?: Array<Record<string, unknown>>;
          layout?: {
            annotations?: Array<{ text?: string }>;
          };
        };
        return {
          names: (plot.data ?? []).map((trace) => trace.name),
          colors: (plot.data ?? []).map((trace) => {
            const marker = trace.marker as
              | { color?: unknown }
              | undefined;
            const line = trace.line as
              | { color?: unknown }
              | undefined;
            return marker?.color ?? line?.color ?? null;
          }),
          annotations: (plot.layout?.annotations ?? []).map(
            (annotation) => annotation.text ?? "",
          ),
        };
      }),
  );
  const serialized = JSON.stringify(plotState);
  expect(serialized).toContain("#38bdf8");
  expect(serialized).toContain("#a3e635");
  expect(serialized).toContain("#fb7185");
  expect(serialized.toLowerCase()).toContain("median");
  const lrStrengthAxes = await page
    .locator(".js-plotly-plot")
    .evaluateAll((nodes) =>
      nodes
        .filter((node) =>
          String(
            (
              node as HTMLElement & {
                layout?: { title?: { text?: string } };
              }
            ).layout?.title?.text,
          ).includes("LR STRENGTH BY DIAGNOSIS"),
        )
        .map((node) => {
          const plot = node as HTMLElement & {
            data?: Array<{ boxpoints?: unknown }>;
            _fullLayout?: { yaxis?: { range?: number[] } };
          };
          return {
            range: plot._fullLayout?.yaxis?.range ?? [],
            boxpoints: (plot.data ?? []).map(
              (trace) => trace.boxpoints,
            ),
          };
        }),
    );
  expect(lrStrengthAxes).toHaveLength(1);
  expect(lrStrengthAxes[0].range[1]).toBeLessThan(4);
  expect(lrStrengthAxes[0].boxpoints).toEqual([false, false, false]);
  expect(
    page.getByRole("heading", {
      name: "LR exception versus non-exception strength across CN, MCI and AD",
    }),
  ).toBeVisible();
  await expect(
    page.getByText("GROUP MEDIAN STRENGTH PROFILES — NO SUBTRACTION", {
      exact: true,
    }),
  ).toHaveCount(0);
  await expect(
    page.getByRole("columnheader", { name: "Omnibus Holm p" }),
  ).toHaveCount(0);
  await expect(
    page.getByRole("columnheader", { name: "Supported after correction" }),
  ).toHaveCount(0);
  await expect(
    page.getByRole("columnheader", {
      name: "Prespecified 4-test Holm p",
    }),
  ).toHaveCount(0);
  await expect(
    page.getByText("How the mean + 3 SD flag works for one subject", {
      exact: true,
    }),
  ).toHaveCount(0);
  await expect(
    page.getByRole("heading", {
      name: "Does exception strength change disproportionately with diagnosis?",
    }),
  ).toBeVisible();
  await expect(page.getByText("0.0343", { exact: true }).first()).toBeVisible();
  await expect(
    page.getByText("CURRENT UNADJUSTED RESULT", { exact: true }),
  ).toHaveCount(0);
  await expect(
    page.getByText("Global edge-class × diagnosis p", { exact: true }),
  ).toHaveCount(0);
  await expect(
    page.getByText("WHAT IS ACTUALLY TESTED", { exact: true }),
  ).toHaveCount(0);
  await expect(
    page.getByText("BIOMARKER GATE", { exact: true }),
  ).toHaveCount(0);
  const significantRow = page
    .locator("tr.table-row-significant")
    .filter({ hasText: "CN vs AD" });
  await expect(significantRow).toBeVisible();
  await expect(significantRow).toContainText("0.0343");
  const interactionLayout = await page
    .locator(".interaction-results-grid")
    .evaluate((grid) => {
      const plot = grid.querySelector(".plot-shell");
      const table = grid.querySelector(".interaction-table-column");
      if (!plot || !table) return null;
      const plotRect = plot.getBoundingClientRect();
      const tableRect = table.getBoundingClientRect();
      return {
        topDifference: Math.abs(plotRect.top - tableRect.top),
        tableOverflow:
          table.scrollHeight - table.clientHeight,
      };
    });
  expect(interactionLayout).not.toBeNull();
  expect(interactionLayout!.topDifference).toBeLessThan(3);
  expect(interactionLayout!.tableOverflow).toBeLessThanOrEqual(1);
  const familyInventory = page.locator(".feature-family-inventory");
  await expect(familyInventory.locator("tbody tr")).toHaveCount(6);
  const inventoryOverflow = await familyInventory
    .locator(".data-table-scroll")
    .evaluate((table) => ({
      vertical: table.scrollHeight - table.clientHeight,
      expanded: table.classList.contains("show-all-rows"),
    }));
  expect(inventoryOverflow.expanded).toBe(true);
  expect(inventoryOverflow.vertical).toBeLessThanOrEqual(1);
});

test("plot titles, legends and plotting regions do not geometrically overlap", async ({
  page,
}) => {
  const routes = [
    "/next/#/section/demographics",
    "/next/#/section/global-dti",
    "/next/#/section/local-roi-dti",
    "/next/#/section/global-graph",
    "/next/#/section/node-metrics",
    "/next/#/section/coupling",
    "/next/#/section/coupling-aal",
    "/next/#/section/brain-age",
    "/next/#/section/lr-sr-analysis",
    "/next/#/section/edr-exceptions",
    "/next/#/section/ml-diagnostics",
    "/next/#/section/delay",
    "/next/#/section/advanced",
    "/next/#/section/network-analysis",
  ];
  for (const route of routes) {
    await page.goto(route);
    await expect(page.getByText("Loading connectome contracts…")).toBeHidden({
      timeout: 60_000,
    });
    const firstPlot = page.locator(".js-plotly-plot").first();
    if (await firstPlot.count()) {
      await expect(firstPlot).toBeVisible({ timeout: 60_000 });
    }
    const overlaps = await page.locator(".js-plotly-plot").evaluateAll(
      (plots) =>
        plots.flatMap((plot, index) => {
          const title = plot.querySelector<SVGGElement>(".g-gtitle");
          const legend = plot.querySelector<SVGGElement>(".legend");
          if (!title || !legend) return [];
          const first = title.getBoundingClientRect();
          const second = legend.getBoundingClientRect();
          const width = Math.max(
            0,
            Math.min(first.right, second.right) -
              Math.max(first.left, second.left),
          );
          const height = Math.max(
            0,
            Math.min(first.bottom, second.bottom) -
              Math.max(first.top, second.top),
          );
          const issues: Array<Record<string, number | string>> = [];
          if (width > 1 && height > 1) {
            issues.push({
              index,
              kind: "title-legend",
              overlapArea: width * height,
            });
          }
          const plotNode = plot as HTMLElement & {
            _fullLayout?: { _size?: { t?: number } };
          };
          const topMargin = Number(plotNode._fullLayout?._size?.t);
          if (Number.isFinite(topMargin)) {
            const plottingTop =
              plot.getBoundingClientRect().top + topMargin;
            const legendGap = plottingTop - second.bottom;
            if (legendGap < 8) {
              issues.push({
                index,
                kind: "legend-plotting-region",
                gap: legendGap,
              });
            }
          }
          return issues;
        }),
    );
    expect(overlaps, route).toEqual([]);
  }
});

test("representative page families have no automatically detectable WCAG A/AA violations", async ({
  page,
}) => {
  const routes = [
    "/next/#/",
    "/next/#/section/demographics",
    "/next/#/section/lr-sr-analysis",
    "/next/#/section/sc-matrix-viewer",
    "/next/#/pipeline",
  ];

  for (const route of routes) {
    await page.goto(route);
    await expect(page.getByText("Loading connectome contracts…")).toBeHidden({
      timeout: 60_000,
    });
    await expect(page.getByRole("alert")).toHaveCount(0);

    const results = await new AxeBuilder({ page })
      .withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"])
      .analyze();

    expect(
      results.violations,
      `${route}: ${results.violations
        .map((violation) => `${violation.id} (${violation.nodes.length})`)
        .join(", ")}`,
    ).toEqual([]);
  }
});

test("specialised mobile routes stay within the viewport", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  const routes = [
    "lr-sr-analysis",
    "ml-diagnostics",
    "network-analysis",
    "coupling-aal",
    "novel-findings",
  ];
  for (const section of routes) {
    await page.goto(`/next/#/section/${section}`);
    await expect(page.locator("main")).toBeVisible({ timeout: 60_000 });
    const dimensions = await page.evaluate(() => ({
      viewport: window.innerWidth,
      document: document.documentElement.scrollWidth,
      body: document.body.scrollWidth,
    }));
    expect(dimensions.document, section).toBeLessThanOrEqual(
      dimensions.viewport + 1,
    );
    expect(dimensions.body, section).toBeLessThanOrEqual(
      dimensions.viewport + 1,
    );
  }
});
