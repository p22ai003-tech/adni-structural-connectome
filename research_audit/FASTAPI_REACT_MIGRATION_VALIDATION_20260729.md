# FastAPI + React migration validation

Generated: 2026-07-29 11:56 UTC; updated 14:51 UTC  
Verdict: implementation complete in protected shadow mode; public-root
promotion is intentionally not authorized.

## Implemented application

- All 18 analysis sections and the live 530-subject pipeline monitor resolve
  through the React application.
- The read-only FastAPI core exposes authoritative subject, node, network,
  coupling-AAL, ML, novelty, LR/SR, matrix, artifact and pipeline contracts.
- Demographics, subject-level DTI/graph/coupling/brain-age/delay/advanced
  analyses, node-level DTI/graph analyses, matrix QC, LR-SR work-along,
  Coupling-AAL, ML diagnostics, network analysis and novelty synthesis have
  dedicated scientific workspaces.
- Every section retains the complete allow-listed source-artifact browser and
  authenticated raw-row visibility/download path.
- The existing Streamlit application and scientific refresh pipeline were not
  changed.

## Requested LR/EDR corrections

1. CN, MCI and AD consistently use cyan, green and coral in plots.
2. A global Plotly margin/title/legend contract is applied.
3. Automated geometry checks find no title/legend intersections on every
   plot-bearing migrated route.
4. LR exception and non-exception plots show full-data median labels; the
   one-subject demonstration and raw-subtraction display were replaced by
   cohort-level direct-strength and edge-class-interaction views.
5. Box-plot outlier points are hidden and the LR strength and interaction
   panels use
   per-group/edge-class 1.5-IQR visual trimming; inferential rows and medians
   remain full-data.
6. The direct inference table spells out the measure and direction and
   contains six rows: exception and non-exception strength by CN-MCI, CN-AD
   and MCI-AD. The three edge-class-interaction pairs are displayed separately.
7. Feature-family subfeatures are displayed as bulleted lists.
8. The six-category diagnosis-by-edge-class box plot is the sole direct
   strength graphic; the redundant group-median profile was removed.
9. A log-axis annotation bug discovered during visual review was fixed. Before
   correction, raw median annotation coordinates expanded the displayed axis
   to approximately 10^51; the corrected axis uses log10 annotation
   coordinates and retains raw values in the box traces and labels.
10. A subject-level edge-class-by-diagnosis interaction now formally asks
    whether diagnosis changes exception strength disproportionately to
    non-exception strength. The global rank-based log-strength interaction is
    p=0.0334; CN-AD survives three-pair Holm correction at p=0.0343 with a
    small Cliff effect, while MCI-AD is not supported at p=0.2164. The
    interface explicitly labels this exploratory and not a validated
    biomarker.
11. The tract-length panel now displays the complete-cohort median
    (124.04 mm) and the stored SR/MR/LR ranges in a yellow summary strip above
    the distributions. Low- and high-outlier count columns were removed from
    the visible table without changing the source calculations.
12. The EDR summary table now has two explicit, color-coded column groups:
    strength measures (median non-exception and exception measures) and
    exception burden/rate (edge count, pooled rate, median count/case and
    median subject rate).
13. Horizontal-legend clearance is enforced against the plotting region, not
    only the title. The browser regression check traverses every plot-bearing
    migrated route and requires both title/legend nonintersection and at least
    eight rendered pixels between the legend and data region.
14. No held-out-site EDR classifier was added. That design tests site
    transportability rather than the edge-class-by-diagnosis scientific
    question; the interaction analysis remains the applicable current test.
15. The interaction p-value metric strip and the three explanatory result
    cards were removed as requested; the exact p-values remain in the
    inferential table.
16. The interaction separation box plot and its complete three-row pairwise
    table now share one desktop row. The Holm-significant CN-AD row is
    highlighted in yellow and explicitly keyed as adjusted p<0.05.
17. The high-level feature-family inventory is fully expanded with all six
    rows visible and no vertical scroll container.

## Automated evidence

- Python core/API/reference/security suite: **23 passed**.
- Real-Chromium Playwright suite: **7 passed**.
  - overview, LR-SR, matrix and pipeline flows;
  - all 18 analysis routes;
  - ML, network, Coupling-AAL and novelty specialized workspaces;
  - LR colors, medians, visual trim and feature bullets;
  - title/legend/data-region geometry across all plot routes;
  - WCAG A/AA automation on representative page families;
  - no document-level overflow at 390-pixel width on the five specialized
    routes.
- TypeScript typecheck: passed.
- Production Vite build: passed.
- `npm audit --audit-level=high`: zero vulnerabilities.
- JSON contracts reject non-standard NaN/Infinity on LR, ML, network,
  findings, features and pipeline endpoints.
- Artifact download bytes match the allow-listed source exactly; encoded path
  traversal is rejected.
- Mutation verbs are rejected by the read-only API.

## Bounded load check

Direct localhost ApacheBench results:

- health: 200 requests, concurrency 10, 0 failed, 1,260 requests/s;
- network distribution (4,804 rows): 20 requests, concurrency 4, 0 failed,
  26.2 requests/s;
- pipeline release status: 20 requests, concurrency 4, 0 failed,
  62.9 requests/s.

This is a bounded single-host smoke profile, not an internet-scale capacity
certification.

## Deployment and rollback evidence

- Streamlit reference remains on `127.0.0.1:8501` and the public HTTPS root.
- FastAPI/React remains on `127.0.0.1:8001`, exposed at `/api/` and `/next/`.
- Unauthenticated public root and preview requests both return 401 under the
  existing nginx boundary.
- The shadow service was stopped and restored as a rollback rehearsal.
  Streamlit health stayed 200 while the shadow was stopped; React returned 200
  after restoration.
- Both Recovery4 HCP379 workers remained active throughout.

## Frozen build identity

- `dist/index.html` SHA-256:
  `be964452d9709619871bced390c2946630706e801446c40bf9c34e0a9b04ec23`
- `dist/assets/index-BF8NC3bw.js` SHA-256:
  `e6191b0b90cdcff46ec1cb972129dc8135f4f513c40b60680d060f43d38600f4`
- `dist/assets/index-DrFQ4qtM.css` SHA-256:
  `674aefe3e16c86eaf5aa63ecdd8fc2a68b99001d4b459b3527a16eea82189f32`

## Non-blocking observations

- The production JavaScript bundle is approximately 1.70 MB before gzip
  (553 KB gzip), so Vite reports a chunk-size optimization warning. It does
  not fail the build or current load checks.
- The Python suite reports one Starlette TestClient deprecation warning. It
  does not affect runtime correctness.

## Remaining promotion gate

Only two user-controlled actions remain:

1. human visual/keyboard acceptance of the protected `/next/` shadow;
2. explicit authorization to promote React from `/next/` to `/`.

Until both occur, Streamlit remains the authoritative public dashboard.
