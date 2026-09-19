# connectome_api — the data server

## What is this folder?

This is a small web server, written with FastAPI (a Python library for
building web servers). It reads the analysis result files and hands them out
as JSON (a text format that programs exchange) when a program asks for them
at an address such as `http://127.0.0.1:8001/api/v1/cohort/summary`. An
*API* (application programming interface) is exactly this: a fixed list of
questions a program may ask, and the shape of each answer.

It is **read-only**. It never writes, changes or deletes a file. It also
serves the built React web app (`apps/connectome_web/`) at `/next/`.

## Do I need it?

- **Yes**, if you want the React web app. The web app gets every number from
  this server.
- **Yes**, if you want to fetch results from your own script or notebook in a
  tidy, documented form.
- **No**, if you use the Streamlit dashboard (`apps/connectome_dashboard/`).
  It reads the result files directly.

## What's inside

| File | What it does |
|---|---|
| `main.py` | The whole server: every address it answers, its error handling, and the read-only rule. It reads results through `connectome_dashboard_core/`. |
| `schemas.py` | The shape of the answers: the standard envelope (`schema_version`, `data`, `warnings`, `provenance`), the health answer and the error answer. |
| `__init__.py` | Marks the folder as a Python package so it can be started as `apps.connectome_api.main`. |
| `requirements.txt` | A short, stand-alone package list for a machine that runs only this server. You do not need it if you installed the top-level `requirements.txt`. It is pinned separately, so a few versions differ from the top-level install (see [Before you start](#before-you-start)). |

## Before you start

1. The project is installed and the venv is on (see the Install section of the
   [top-level README.md](../../README.md)):

   ```bash
   source .venv/bin/activate
   ```

   The top-level `requirements.txt` already contains FastAPI, uvicorn (the
   program that runs a FastAPI server) and httpx (used by the tests).

   Machine that only runs this server? A smaller install also works:

   ```bash
   python3.12 -m venv .venv
   .venv/bin/pip install -r apps/connectome_api/requirements.txt
   ```

   This smaller list is a lighter, separately pinned set. It installs
   FastAPI 0.140.13, `uvicorn[standard]` (uvicorn with optional speed-ups)
   and pytest 8, where the top-level install uses FastAPI 0.141.1, plain
   uvicorn and pytest 9.1.1. The server works with either set. Use one
   install or the other in a venv, not both.

2. The analysis has been run, so the results folder exists. See
   [connectome_analysis/README.md](../../connectome_analysis/README.md):

   ```bash
   python -m connectome_analysis.run_analysis --all --resume
   ```

3. Optional: build the web app once, so the server can show it at `/next/`.
   See [connectome_web/README.md](../connectome_web/README.md).

## How to run it

All commands are typed from the repository root.

**Step 1. Start the server.**

```bash
python -m uvicorn apps.connectome_api.main:app --host 127.0.0.1 --port 8001
```

What you should see:

```
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8001 (Press CTRL+C to quit)
```

The terminal stays busy while the server runs. Open a second terminal for the
next steps. Every request is logged in the first terminal, for example:

```
... INFO connectome_api request_id=... method=GET path=/api/v1/health/live status=200 elapsed_ms=1.4
```

**Step 2. Check that it is alive.**

```bash
curl http://127.0.0.1:8001/api/v1/health/live
```

What you should see:

```
{"status":"ok","service":"connectome-api","schema_version":"1.0"}
```

(`curl` fetches a web address in the terminal. You can also paste the address
into your browser.)

**Step 3. Check that it can see your results.**

```bash
curl http://127.0.0.1:8001/api/v1/cohort/summary
```

What you should see (the start of a long line):

```
{"schema_version":"1.0","data":{"subjects":530,"group_counts":{"CN":251,"MCI":201,"AD":78},"columns":[...
```

If you see `{"detail":"[Errno 2] No such file or directory: ...master_cohort.csv"...}`
instead, the server cannot find the results. See
[If something goes wrong](#if-something-goes-wrong).

**Step 4. Look around.** Open http://127.0.0.1:8001/api/docs in your browser.
It lists every address, lets you fill in the parameters and press
"Try it out". The same list, as a machine-readable file, is at
http://127.0.0.1:8001/api/openapi.json.

**Step 5. Stop the server.** Press `Ctrl+C` in the first terminal.

### For developers: restart automatically on code changes

```bash
python -m uvicorn apps.connectome_api.main:app --host 127.0.0.1 --port 8001 \
  --reload --reload-dir apps/connectome_api --reload-dir connectome_dashboard_core
```

What you should see first:

```
INFO:     Will watch for changes in these directories: ['<repo>/apps/connectome_api', '<repo>/connectome_dashboard_core']
INFO:     Uvicorn running on http://127.0.0.1:8001 (Press CTRL+C to quit)
INFO:     Started reloader process [...] using StatReload
```

## What it answers

Every address starts with `http://127.0.0.1:8001`. Words in `{braces}` are
filled in by you. `?name` marks an optional setting added after a `?`.
Several settings are joined with `&`, for example `?offset=0&limit=2`.

Two abbreviations used below:
- **LR/SR** = long-range / short-range connections, split by the length of the
  fibre tract between two brain regions.
- **EDR** = the exponential distance rule: longer connections are usually
  weaker. An *EDR exception* is a connection much stronger than its length
  predicts.

**Worked example with settings.** Put the whole address in double quotes:

```bash
curl "http://127.0.0.1:8001/api/v1/models/features/eda?offset=0&limit=2"
```

What you should see (the start of a long line):

```
{"schema_version":"1.0","data":{"subjects":530,"offset":0,"limit":2,"total_rows":2299,"columns":["Feature","Dashboard family",...
```

The quotes matter. Without them the terminal treats `&` as "run this in the
background": it prints something like `[1] 12345`, and the server never sees
`limit=2` (you get the default 200 rows instead). Any address that contains
`?` or `&` must be in double quotes.

| Group | Address | What you get |
|---|---|---|
| health | `/api/v1/health/live` | "ok" whenever the server runs |
| health | `/api/v1/health/ready` | "ready" when the results can be found, otherwise status 503 and a list of what is missing |
| metadata | `/api/v1/metadata/app`, `/api/v1/metadata/provenance` | what the server is, which folders it reads, how many result files it indexed |
| sections | `/api/v1/sections` | the 18 dashboard sections, with file counts and metric lists |
| sections | `/api/v1/sections/{section_id}/artifacts` | every result file of one section (id, name, size, type) |
| sections | `/api/v1/sections/{section_id}/metrics`, `.../metrics/{metric}` | the per-participant metrics of a section, and their values |
| section analysis | `/api/v1/sections/{section_id}/analysis/profile`, `.../subject/{metric}`, `.../nodes/{metric}`, `.../nodes/{metric}/ranking` (`?group_a&group_b`), `.../nodes/{metric}/repeated` (`?top_n`), `.../nodes/{metric}/{node}` | group comparisons of one metric, over participants or over the 166 brain regions |
| files | `/api/v1/artifacts/{artifact_id}/table` (`?offset&limit`), `.../content`, `.../download` | one result file: a page of a CSV table, the text of a JSON or Markdown file, or the file itself |
| cohort | `/api/v1/cohort/summary`, `/api/v1/demographics/summary` | group counts and demographics |
| LR/SR | `/api/v1/lr-sr/tract-length/summary`, `.../ranges`, `.../distribution` (`?panel`) | tract-length distributions and the short/medium/long range cut-offs |
| LR/SR | `/api/v1/lr-sr/exceptions/config`, `.../summary`, `.../inference`, `.../subjects` (`?measure`), `/api/v1/lr-sr/exceptions/{subject_id}/example` | EDR exception results. `.../subjects` lists one row per participant, with their identifiers. |
| models | `/api/v1/models/features/families`, `/api/v1/models/features/eda` (`?offset&limit&family&status&search`), `/api/v1/models/tasks`, `/api/v1/models/tasks/{task}` | the recorded machine-learning results. Nothing is re-trained. |
| networks | `/api/v1/networks/mapping` (`?detail`), `/api/v1/networks/exception-specificity`, `/api/v1/networks/measure-ranked` (`?contrast`: `cn_ad` by default, or `cn_mci`, `mci_ad`), `/api/v1/networks/analysis/catalog`, `.../distribution` (`?scheme&family&metric`), `.../affectedness`, `.../blocks` | results summarised by brain network |
| coupling | `/api/v1/coupling-aal/catalog`, `/api/v1/coupling-aal/ranking` | region-level structure/microstructure coupling |
| findings | `/api/v1/findings/summary` | the findings catalogue and finding cards |
| matrices | `/api/v1/connectomes/subjects` | which participants have which matrix types |
| matrices | `/api/v1/connectomes/{subject_id}/{weight}/summary`, `.../matrix`, `.../edges` (`?positive_only&offset&limit`) | one participant's 166 x 166 matrix: summary numbers, the full matrix, or a sorted list of connections |
| pipeline | `/api/v1/pipeline/release-status`, `/api/v1/pipeline/live-status`, `/api/v1/pipeline/subjects` | imaging progress files. These exist only on the imaging server (see below). |

`{section_id}` is one of: `demographics`, `novel-findings`, `global-dti`,
`local-roi-dti`, `global-graph`, `sc-matrix-viewer`, `node-metrics`,
`coupling`, `coupling-aal`, `brain-age`, `lr-sr`, `lr-sr-analysis`,
`edr-exceptions`, `ml-diagnostics`, `delay`, `advanced`, `network-analysis`,
`functional-pending`.

`{weight}` is one of: `fd_sum`, `count`, `count_invnodevol`, `invlen_mean`,
`len_mean`, `fa_mean`, `md_mean`, `ad_mean`, `rd_mean`.

`{subject_id}` is an ADNI participant identifier such as `XXX_S_0001`. Take
real ones from `/api/v1/connectomes/subjects`; never write them into shared
documents.

### The shape of every answer

Apart from the two health addresses and the file download, every answer is
wrapped the same way:

```json
{
  "schema_version": "1.0",
  "data": { "...": "the actual result" },
  "warnings": [],
  "provenance": {
    "generated_utc": "2026-09-18T15:18:48.189377+00:00",
    "sources": [{"path": ".../00_master/master_cohort.csv", "available": true, "mtime_ns": 1789713800993956157}],
    "calculation": "a short description, when there is one"
  }
}
```

`provenance.sources` names the result files the answer was read from, so you
can always trace a number back to its file. Numbers that are not finite (NaN,
infinity) are sent as `null`.

## Inputs

The server reads, and never writes:

| Input | Where | Notes |
|---|---|---|
| Analysis results | `$SC_ANALYSIS_ROOT` (default `data/derivatives/qc/analysis_cohort`) | The numbered section folders (`00_master/`, `02_demographics/`, ... `20_exception_specificity/`). `00_master/master_cohort.csv` is required by most addresses. |
| Connectome matrices | `$SC_CONNECTOMES_DIR` (default `data/derivatives/connectomes`) | `SC_AAL166_<SUBJECT>_I<IMAGEID>_<type>.csv`, headerless 166 x 166, comma-separated. Used by the matrix addresses and the LR/SR tract-length and exception addresses. |
| Atlas labels | `atlas/AAL/aal3_labels_166.csv` (shipped with the repository) | Region names shown next to matrix rows, keyed on the matrix row. |
| Built web app | `apps/connectome_web/dist/` | Served at `/next/` if the folder exists when the server starts. |
| Imaging progress files | `research_audit/outputs/...` and `$SC_QC_ROOT/sc_matrix_qc/` (`SC_QC_ROOT` = the quality-control folder, default `$SC_DERIV_ROOT/qc`) | Only on the imaging server. Used by the three `pipeline` addresses. |

Only files ending in `.csv`, `.json`, `.md`, `.png`, `.svg` or `.pdf` inside
the results folder can be listed or downloaded. Links (symlinks) and paths
that try to leave the results folder are refused.

## Outputs

None. The server does not write files. Any request other than reading (GET,
HEAD, OPTIONS) is refused:

```bash
curl -X POST http://127.0.0.1:8001/api/v1/health/live
```

```
{"detail":"This service is read-only","request_id":"4d18424d-82ec-438e-b74a-971a4a49463b"}
```

## Settings you can change

| Setting | Where | What it does | Default |
|---|---|---|---|
| `CONNECTOME_ANALYSIS_ROOT` | environment variable | Results folder. Wins over `SC_ANALYSIS_ROOT`. | `$SC_ANALYSIS_ROOT` |
| `SC_ANALYSIS_ROOT` | environment variable (see `sc_config.py`) | Results folder, shared with the analysis. | `$SC_DERIV_ROOT/qc/analysis_cohort` |
| `CONNECTOME_MATRICES_DIR` | environment variable | Matrix folder. Wins over `SC_CONNECTOMES_DIR`. | `$SC_CONNECTOMES_DIR` |
| `SC_CONNECTOMES_DIR` | environment variable | Matrix folder, shared with the analysis. | `$SC_DERIV_ROOT/connectomes` |
| `CONNECTOME_PROJECT_ROOT` | environment variable | Repository root, used to find `atlas/` and `research_audit/`. | the repository |
| `--host` | uvicorn option | Which network address to listen on. Keep `127.0.0.1`. | `127.0.0.1` |
| `--port` | uvicorn option | Port number. The web app's development server and its tests expect `8001`. | `8000` if you leave it out |
| `?limit` on paged addresses | in the address | Rows per page. | 200 (tables), 500 (edges); at most 5000 (2500 for `/models/features/eda`) |

Example: point the server at a second results folder.

```bash
export SC_ANALYSIS_ROOT="$HOME/sc_trial/analysis_cohort"
python -m uvicorn apps.connectome_api.main:app --host 127.0.0.1 --port 8001
```

Keep result folders outside the repository (or inside the git-ignored
`data/`): they hold participant-level data and must never be committed.

Settings are read once, at start-up. Restart the server after changing one.
Most result files are re-read automatically when they change on disk. After a
new analysis run, restart the server anyway to be sure every page shows the
new files.

## Logins and passwords

The server has no login of its own. On your computer it listens on
`127.0.0.1`, so only you can reach it. Do not start it with
`--host 0.0.0.0`: it hands out participant-level data to anyone who can reach
the port.

On the project's shared server, nginx (a web server) sits in front of it and
asks for a username and password; see `deploy/FASTAPI_REACT_RUNBOOK.md`. The
credentials file is kept outside the repository; its location is recorded, by
convention, in the environment variable `CONNECTOME_CREDENTIALS_FILE`. No code
reads that variable. Never copy the credentials into the repository.

## Tests

```bash
python -m pytest tests/dashboard_parity -q
```

**These 23 tests only pass on the project's own server. On any other machine
most of them fail, and that is expected; they are not a check for new
users.** They compare the server's answers with frozen reference values kept
in `research_audit/outputs/`, a folder that is not part of the repository, at
a location fixed in `tests/dashboard_parity/conftest.py`. One of them
(`test_pipeline_poll_payload_is_compact_and_read_only`) also needs the imaging
server's progress files. To check your own setup, use Steps 2 and 3 above
instead.

On the project's server the result is:

```
.......................                                                  [100%]
23 passed in 15.18s
```

The tests use FastAPI's built-in test client, so they do not need a running
server and do not use any port.

## If something goes wrong

**`ModuleNotFoundError: No module named 'apps'`**
You started uvicorn from a folder other than the repository root. `cd` to the
repository root (the folder that holds `sc_config.py`) and start it again.

**`No module named uvicorn`**
The venv is off, or the packages are missing. Run `source .venv/bin/activate`,
then `pip install -r requirements.txt`.

**`ERROR:    [Errno 98] error while attempting to bind on address ('127.0.0.1', 8001): [errno 98] address already in use`**
Another program, often an earlier copy of this server, is using port 8001.
Stop it (`Ctrl+C` in its terminal), or use another port with `--port`.

**`{"detail":"[Errno 2] No such file or directory: '.../00_master/master_cohort.csv'", ...}`** (status 404)
The results folder is empty or wrong. Run the analysis, or point the server at
the right folder with `SC_ANALYSIS_ROOT`. Check where it looks with
`curl http://127.0.0.1:8001/api/v1/metadata/provenance`.

**`{"detail":"LR/SR tract-length thresholds are unavailable", ...}`** or
**`{"detail":"network x measure ranked table sources are unavailable", ...}`** (status 404)
Only part of the analysis has been run. Run
`python -m connectome_analysis.run_analysis --all --resume`.

**`/api/v1/health/ready` answers 503 with `required sources unavailable: [...]`**
The message lists every missing item:
- `.../00_master/master_cohort.csv`: run the analysis, or fix `SC_ANALYSIS_ROOT`.
- the matrix folder: fix `SC_CONNECTOMES_DIR`.

Those two are the only things readiness needs. Once both exist it answers 200,
and the web app's top bar changes from "API checking" to "API ready".

**The pipeline addresses say a source is "unavailable", or `/api/v1/pipeline/subjects` answers 404**
`/api/v1/pipeline/release-status`, `/api/v1/pipeline/live-status` and
`/api/v1/pipeline/subjects` report on the imaging runs. They read progress
files that exist only on the machine that ran the imaging (under
`research_audit/outputs/`, which is not part of the repository). Elsewhere the
first two answer normally and list each missing source under `warnings`, and
`/api/v1/pipeline/subjects?source=hcp379` or `?source=recovery-ledger` answers
404 with "... source is not available on this machine". That is expected; every
results address still works.

**`http://127.0.0.1:8001/next/` answers `{"detail":"Not Found"}`**
The web app has not been built, or it was built after the server started.
Run `npm run build` in `apps/connectome_web/` (see its README), then restart
the server.

**`http://127.0.0.1:8001/` answers `{"detail":"Not Found"}`**
Normal. The server has nothing at `/`. Use `/next/` or `/api/docs`.

**Status 422, for example `"msg":"Input should be 'fd_sum', 'count' or 'count_invnodevol'"`**
A value in the address is not allowed (here `?measure=nope`). The message
says which value and what is allowed. `/api/docs` lists the allowed values.

**`{"detail":"unknown section: not-real", ...}`** (status 404)
The `{section_id}` is misspelt. `/api/v1/sections` lists the valid ones.
