# connectome_web — the React web app

## What is this folder?

This is the newer front end for the study's results: a web app that runs in
your browser. It is written in TypeScript (JavaScript with type checks) using
React (a library for building web pages out of components), and it is
packaged with Vite (a tool that turns the source files into a small set of
files a browser can load).

The web app holds no data. Every number it shows comes from the data server in
`apps/connectome_api/` (addresses under `/api/v1/`). So the data server must be
running, and the analysis must have been run before that.

## Do I need it?

- **Yes**, if you want the newer interface, or you want to change it.
- **No**, if the Streamlit dashboard (`apps/connectome_dashboard/`) is enough
  for you. Both show the same result files.
- **No**, for running the analysis. Nothing in the analysis needs Node.js.

## What's inside

| File or folder | What it is |
|---|---|
| `package.json` | The app's name, its commands (`npm run ...`) and the exact library versions it needs. |
| `package-lock.json` | The full, locked list of every library and sub-library, so `npm ci` installs exactly the same versions everywhere. |
| `index.html` | The single HTML page the app starts from. |
| `vite.config.ts` | Vite settings: the app lives under `/next/`, and during development requests to `/api` are forwarded to `http://127.0.0.1:8001`. |
| `tsconfig.json`, `tsconfig.app.json`, `tsconfig.node.json` | TypeScript settings for the app code and for the Vite settings file. |
| `playwright.config.ts` | Settings for the browser tests (Chromium, one worker, server at `http://127.0.0.1:8001`). |
| `src/main.tsx` | Starting point: sets up data fetching and draws the app. |
| `src/App.tsx` | The frame: side menu, top bar, and which page to show for each address. |
| `src/api.ts` | Fetches JSON from `/api/v1/...` and formats numbers, file sizes and CSV downloads. |
| `src/types.ts` | TypeScript descriptions of the data server's answers. |
| `src/generated/api-schema.ts` | The same descriptions, generated automatically from the data server (`npm run generate:api-types`). Do not edit by hand. |
| `src/pages/` | One file per page: overview, demographics, LR-SR analysis, matrix viewer, ML diagnostics, network analysis, coupling-AAL, novel findings, pipeline monitor, and a general section page. |
| `src/components/` | Building blocks the pages share: tables, plots, file browser, loading and error boxes. |
| `src/styles.css` | All the styling. |
| `src/declarations.d.ts` | Tells TypeScript about the Plotly charting bundle. |
| `tests/shadow.spec.ts` | Seven browser tests that open the pages and check headings, numbers, charts, layout and accessibility. |
| `test-results/` | Created by the browser tests. Not part of the repository (it is in `.gitignore`); see "Clean up after every test run". |

Made by you, and never committed (both are in `.gitignore`):

| Folder | Made by | What it is |
|---|---|---|
| `node_modules/` | `npm ci` | The downloaded libraries (about 340 packages). |
| `dist/` | `npm run build` | The finished app: `index.html` plus one JavaScript and one CSS file. The data server shows this folder at `/next/`. |

## Before you start

1. **Node.js 18 or newer**, with npm (the package installer that comes with
   Node.js). Download it from https://nodejs.org (the "LTS" version is fine).
   Check:

   ```bash
   node --version
   npm --version
   ```

   This README was checked with Node.js `v18.20.6` and npm `10.8.2`.

2. **The Python side is installed** (Install section of the
   [top-level README.md](../../README.md)) and the venv is on:

   ```bash
   source .venv/bin/activate
   ```

3. **The analysis has been run** (see
   [connectome_analysis/README.md](../../connectome_analysis/README.md)):

   ```bash
   python -m connectome_analysis.run_analysis --all --resume
   ```

4. **The data server works** (see
   [connectome_api/README.md](../connectome_api/README.md)).

## How to run it

There are two ways. Pick **Way 1** to just use the app. Pick **Way 2** if you
want to change the code and see your changes straight away.

### Way 1: build once, let the data server show it

**Step 1. Go into this folder and install the libraries.**

```bash
cd apps/connectome_web
npm ci
```

What you should see at the end (earlier `npm warn deprecated ...` lines can be
ignored; they do not stop the install):

```
added 344 packages, and audited 345 packages in 5s

28 packages are looking for funding
  run `npm fund` for details

3 vulnerabilities (1 high, 2 critical)

To address all issues, run:
  npm audit fix

Run `npm audit` for details.
```

The install worked if the output contains a line starting with
`added ... packages`. The vulnerability lines are expected. **Do NOT run
`npm audit fix`**: it changes `package-lock.json`, so you would no longer get
the tested versions, and the build can break.

`npm ci` installs exactly what `package-lock.json` lists. Use it instead of
`npm install`, which may pick newer versions and change the lock file.

**Step 2. Build the app.**

```bash
npm run build
```

What you should see:

```
vite v6.4.3 building for production...
✓ 102 modules transformed.
dist/index.html                     0.53 kB │ gzip:   0.32 kB
dist/assets/index-rcMX2TaY.css     24.87 kB │ gzip:   6.33 kB
dist/assets/index-Dcpsu-Od.js   1,735.54 kB │ gzip: 566.34 kB

(!) Some chunks are larger than 500 kB after minification. ...
✓ built in 7.26s
```

The "chunks are larger than 500 kB" note is normal; the charting library is
large.

**Step 3. Go back to the repository root and start the data server.**

```bash
cd ../..
python -m uvicorn apps.connectome_api.main:app --host 127.0.0.1 --port 8001
```

The data server looks for `apps/connectome_web/dist/` when it starts. If it
was already running before you built, stop it (`Ctrl+C`) and start it again.

**Step 4. Open http://127.0.0.1:8001/next/ in your browser.**

You should see the heading "From microstructure to network geometry, one
auditable analysis surface." and, in the top bar, the number of subjects.

### Way 2: development server (changes appear as you save)

You need two terminals.

**Terminal 1**, repository root: start the data server on port 8001. The
development server always forwards data requests to port 8001.

```bash
source .venv/bin/activate
python -m uvicorn apps.connectome_api.main:app --host 127.0.0.1 --port 8001
```

**Terminal 2**: start the development server.

```bash
cd apps/connectome_web
npm ci            # only the first time
npm run dev
```

What you should see:

```
  VITE v6.4.3  ready in 160 ms

  ➜  Local:   http://127.0.0.1:5173/next/
```

Open http://127.0.0.1:5173/next/. When you save a file in `src/`, the page
updates by itself. Press `Ctrl+C` in each terminal to stop.

### Preview the built app without the data server serving it

```bash
cd apps/connectome_web
npm run build
npm run preview
```

What you should see:

```
  ➜  Local:   http://127.0.0.1:4173/next/
```

Data requests are forwarded to port 8001 here too, so the data server must be
running.

## Checks

Run these inside `apps/connectome_web/`.

**Type check** (finds mistakes in the TypeScript without building):

```bash
npm run typecheck
```

What you should see when there are no problems (nothing after these two
lines):

```
> connectome-dashboard-web@1.0.0 typecheck
> tsc -b --pretty false
```

**Browser tests** (open the pages in a real Chromium browser):

```bash
npx playwright install chromium     # only the first time: downloads the browser
npm run test:e2e
```

The tests talk to `http://127.0.0.1:8001`, so first build the app (Way 1,
steps 1 to 3) and keep the data server running.

The tests look for the study's published numbers (for example 530 subjects),
so they need the complete study results. Test 6 also checks the Pipeline
monitor page for accessibility problems. Without the imaging server's progress
files that page shows "UNKNOWN", "NA" and empty tables, and the accessibility
check flags them. So on any machine other than the project's server, even
with the complete results, expect this at the end:

```
  1 failed
    [chromium] › tests/shadow.spec.ts:417:1 › representative page families have no automatically detectable WCAG A/AA violations
  6 passed (39.2s)
```

Just above it, the failure report says
`Error: /next/#/pipeline: color-contrast (1), scrollable-region-focusable (1)`.

On the project's server the result is:

```
Running 7 tests using 1 worker

  ✓  1 [chromium] › tests/shadow.spec.ts:31:1 › overview, LR-SR, matrix and pipeline routes render without browser errors (12.5s)
  ...
  ✓  7 [chromium] › tests/shadow.spec.ts:448:1 › specialised mobile routes stay within the viewport (788ms)

  7 passed (35.2s)
```

**Clean up after every test run.** When a test fails, Playwright saves a
screenshot (`test-failed-1.png`), a page description (`error-context.md`) and
a recording (`trace.zip`) in a sub-folder of `test-results/`. These files can
contain participant identifiers and participant-level pages, which the ADNI
Data Use Agreement does not allow you to share. `test-results/` is listed in
`.gitignore`, so git leaves it alone, but delete these files after every run
anyway, and never copy them anywhere shared:

```bash
rm -rf test-results/*/
```

**Regenerate the API descriptions** after the data server's answers change:

```bash
npm run generate:api-types
```

This reads `http://127.0.0.1:8001/api/openapi.json`, so the data server must
be running. It uses `npx` to fetch the tool `openapi-typescript` (version
7.10.1), so the first time it needs an internet connection. It rewrites
`src/generated/api-schema.ts`, a file that is part of the repository. What you
should see:

```
✨ openapi-typescript 7.10.1
🚀 http://127.0.0.1:8001/api/openapi.json → src/generated/api-schema.ts [357.2ms]
```

Afterwards, look at what changed:

```bash
git diff src/generated/api-schema.ts
```

No output means the data server's answers have not changed.

## Pages

The part of the address after `#` picks the page. Examples use
`http://127.0.0.1:8001/next/`.

| Address | Page |
|---|---|
| `#/` | Overview: cohort size and a card for every section |
| `#/section/demographics` | Group counts, age, sex, clinical scores |
| `#/section/lr-sr-analysis` | LR-SR = long-range / short-range connections. Tract lengths, the short/long-range split, EDR exceptions (connections much stronger than their length predicts; EDR = exponential distance rule), feature families |
| `#/section/lr-sr-models` | Cognition models on the tiered connectome features |
| `#/section/sc-matrix-viewer` | One participant's 166 x 166 matrix, any of the nine types |
| `#/section/ml-diagnostics` | Cross-validated model results (recorded, not re-trained) |
| `#/section/network-analysis` | Results summarised by brain network |
| `#/section/coupling-aal` | Region-level coupling rankings |
| `#/section/novel-findings` | The findings catalogue and finding cards |
| `#/section/<id>` | Any other section: its charts, tables and downloadable files |
| `#/pipeline` | Imaging progress (works only on the imaging server) |

The side menu lists every section. The top-bar badge says "API ready" when
`/api/v1/health/ready` answers 200.

## Inputs and outputs

- **Inputs:** only the data server's answers, fetched from the same address
  the page was loaded from (`/api/v1/...`). The app reads no files itself.
- **Outputs:** the `dist/` folder when you build. In the browser, some tables
  have a download button that saves a CSV file to your computer. Nothing is
  written on the server.

## Settings you can change

| Setting | Where | What it does | Default |
|---|---|---|---|
| Data server address for development | `vite.config.ts`, `server.proxy["/api"]` | Where `npm run dev` and `npm run preview` forward `/api` requests. | `http://127.0.0.1:8001` |
| Base path | `vite.config.ts`, `base` | The folder the app expects to live in. The data server serves it at `/next/`; change both together. | `/next/` |
| Development port | `package.json`, the `dev` script | Port for `npm run dev`. | `5173` |
| Preview port | `package.json`, the `preview` script | Port for `npm run preview`. | `4173` |
| Test server address | `playwright.config.ts`, `baseURL` | Where the browser tests look for the app. | `http://127.0.0.1:8001` |
| API description source | `package.json`, the `generate:api-types` script | Where the API descriptions are downloaded from. | `http://127.0.0.1:8001/api/openapi.json` |

Where the results come from is set on the data server, not here. See
[connectome_api/README.md](../connectome_api/README.md#settings-you-can-change).

## Logins and passwords

The web app has no login. It sends requests to the same address it was loaded
from, together with whatever login the browser already holds. On your own
computer there is none, and the servers listen on `127.0.0.1` so only you can
reach them. On the project's shared server, nginx asks for a username and
password before the page loads (see `deploy/FASTAPI_REACT_RUNBOOK.md`). The
credentials file lives outside the repository (by convention its location is
in `CONNECTOME_CREDENTIALS_FILE`); never copy it into the repository.

The side-menu link "Open Streamlit reference" points to `/`. That works only
on the shared server, where nginx serves the Streamlit dashboard at `/`. On
your computer, open the dashboard at its own address instead
(http://127.0.0.1:8501).

## If something goes wrong

**The page says "Could not load this view" and "Request failed (500)", and the
`npm run dev` terminal shows:**

```
[vite] http proxy error: /api/v1/sections
Error: connect ECONNREFUSED 127.0.0.1:8001
```

The data server is not running on port 8001. Start it (Way 2, terminal 1) and
reload the page.

**The page says "Could not load this view" and
`[Errno 2] No such file or directory: '.../00_master/master_cohort.csv'`**
The data server runs, but it cannot find the results. Run the analysis, or
point the data server at the right folder with `SC_ANALYSIS_ROOT`.

**Some boxes say "Could not load this view" with messages like
"LR/SR tract-length thresholds are unavailable" or
"network x measure ranked table sources are unavailable"**
Only part of the analysis has been run. Run
`python -m connectome_analysis.run_analysis --all --resume` and restart the
data server.

**The "Pipeline monitor" page shows sources as unavailable, or "Request failed (404)"**
Expected on any machine other than the one that ran the imaging: that page
reads imaging progress files that are not part of the repository. All other
pages work.

**The top bar says "API checking" and never "API ready"**
The data server's readiness check (`/api/v1/health/ready`) is failing: it
cannot find the analysis results or the connectome matrices. Open
http://127.0.0.1:8001/api/v1/health/ready in the browser; the message names
what is missing. See the data server README for the fix.

**http://127.0.0.1:8001/next/ shows `{"detail":"Not Found"}`**
The app has not been built, or it was built after the data server started.
Run `npm run build`, then restart the data server.

**`npm: command not found` or `node: command not found`**
Node.js is not installed, or your terminal was opened before you installed it.
Install Node.js 18 or newer and open a new terminal.

**`npm ci` prints `npm warn deprecated ...` or "3 vulnerabilities (1 high, 2 critical)"**
These are warnings from the library registry. The install still finishes; the
output should contain a line starting with `added ... packages`. Do NOT run
`npm audit fix`: it changes `package-lock.json`. If you already did, undo it
with `git checkout -- package-lock.json` and run `npm ci` again.

**Browser tests: "1 failed, 6 passed", with
`Error: /next/#/pipeline: color-contrast (1), scrollable-region-focusable (1)`**
Expected on any machine other than the project's server: the Pipeline monitor
page has no imaging progress files to show, and the accessibility check
(test 6) flags the empty page. Delete `test-results/*/` afterwards (see
[Checks](#checks)).
