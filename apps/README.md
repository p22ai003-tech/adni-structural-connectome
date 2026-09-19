# apps/ — the results viewers

## What is this folder?

The programs in this folder show the study's results in a web browser:
charts, tables, and the connectome matrix of a single participant. They do
**not** calculate anything new. They only read the result files that the
analysis pipeline (`connectome_analysis/`) has already written, and draw them.

You start a viewer on your own computer, then open an address such as
`http://127.0.0.1:8501` in your web browser. `127.0.0.1` means "this
computer". Nobody else can see the page.

## Do I need it?

- **Yes**, if you want to click through the results instead of opening CSV
  files one by one.
- **No**, if you only need the numbers. Every table the viewers show is a
  plain CSV file in the analysis results folder (`$SC_ANALYSIS_ROOT`, see
  [Where the viewers read results from](#where-the-viewers-read-results-from)).
- The analysis does not depend on this folder. It runs without any viewer.

## What's inside

| Folder | What it is | Built with | Address when running |
|---|---|---|---|
| [`connectome_dashboard/`](connectome_dashboard/README.md) | The complete, older dashboard. The easiest one to start: one command. | Streamlit (a Python tool that turns a script into a web page) | http://127.0.0.1:8501 |
| [`connectome_api/`](connectome_api/README.md) | A data server. Programs ask it for results ("give me the demographics table") and it answers in JSON (a text format that programs exchange). | FastAPI (a Python web-server library) | http://127.0.0.1:8001/api/docs |
| [`connectome_web/`](connectome_web/README.md) | The newer web app. Same results as the dashboard, faster pages. It gets all its data from the data server, so the data server must be running. | React and TypeScript, built with Node.js | http://127.0.0.1:8001/next/ |
| [`connectome_monitor/`](connectome_monitor/README.md) | A one-page progress screen for the imaging pipeline. Only useful on the computer that runs the imaging. | Streamlit | http://127.0.0.1:8502 (any free port) |

One more piece lives at the repository root:
[`connectome_dashboard_core/`](../connectome_dashboard_core/README.md) is the
Python library the data server uses to find and read the result files.

## How the pieces fit together

```
connectome matrices + cohort tables
            |
            v
python -m connectome_analysis.run_analysis --all       (the analysis, see connectome_analysis/README.md)
            |
            |  writes CSV, JSON and PNG files into
            v
$SC_ANALYSIS_ROOT   (default: data/derivatives/qc/analysis_cohort)
      |                                   |
      | read directly                     | read through connectome_dashboard_core/
      v                                   v
connectome_dashboard/  (Streamlit)    connectome_api/  (FastAPI, port 8001)
port 8501                                 |
                                          |  JSON at http://127.0.0.1:8001/api/v1/...
                                          v
                                     connectome_web/  (React, shown at /next/)
```

The Streamlit dashboard and the React web app are two separate front ends
over the same result files. You only need one of them.

## Before you start

1. **Install the project.** Follow the Install section of the
   [top-level README.md](../README.md). In short:

   ```bash
   python3.12 -m venv .venv
   .venv/bin/pip install -r requirements.txt
   ```

   A *venv* (virtual environment) is a private Python installation for this
   project. `requirements.txt` already includes the data server (FastAPI and
   uvicorn).

2. **Open a terminal in the repository root** (the folder that contains
   `sc_config.py`) and switch the venv on. Do this in every new terminal:

   ```bash
   source .venv/bin/activate
   ```

   A *terminal* is the window where you type commands. Every command in these
   READMEs is typed from the repository root unless a step says `cd`.

3. **Get the data.** The input data come from ADNI (the Alzheimer's Disease
   Neuroimaging Initiative). You must download them yourself from
   https://adni.loni.usc.edu under your own ADNI Data Use Agreement. They are
   never part of this repository and must never be committed to git. How the
   connectome matrices are made from the raw images: see the Imaging section
   of the top-level [README.md](../README.md).

4. **Run the analysis first.** The viewers show only what the analysis wrote.
   With no results they show an error page.

   ```bash
   python sc_doctor.py --analysis     # preflight: paths, packages, cohort tables
   python sc_cohort.py check          # checks the cohort tables
   python -m connectome_analysis.run_analysis --all --resume
   ```

   Two words used here:
   - *Preflight* = a check you run before the real work. `sc_doctor.py` looks
     for missing folders, packages and files and tells you what to fix.
   - *Cohort tables* = the CSV files in `cohort/` that list each participant,
     their diagnostic group and which images the analysis uses (`dti.csv`,
     `mri.csv`, `dti_master.csv`, `mri_master.csv`). `sc_cohort.py build`
     makes them from your ADNI downloads.

   What you should see from the first two commands (paths shortened):

   ```
   ========================================================================
   no problems found.
   ```

   ```
   cohort folder: <repo>/cohort

     ok       dti.csv            1114 rows
     ok       mri.csv            1114 rows
     ok       dti_master.csv     8481 rows
     ok       mri_master.csv    27396 rows

   no problems found.
   ```

   The full analysis run is described in
   [connectome_analysis/README.md](../connectome_analysis/README.md).

5. **Install the extra pieces for the viewer you chose:**
   - Streamlit dashboard or monitor: `pip install -r requirements/dashboard.txt`
   - React web app: Node.js 18 or newer (see
     [connectome_web/README.md](connectome_web/README.md)).
   - Data server only: nothing extra.

## Quick start (pick one)

### Option A: the Streamlit dashboard (one terminal)

```bash
pip install -r requirements/dashboard.txt
python -m streamlit run apps/connectome_dashboard/connectome_app.py \
  --server.address 127.0.0.1 --server.port 8501 \
  --server.headless true --browser.gatherUsageStats false
```

What you should see:

```
  You can now view your Streamlit app in your browser.

  URL: http://127.0.0.1:8501
```

Open http://127.0.0.1:8501. The browser tab says "Connectome C Dashboard";
the big heading on the page says "Analysis of Structural connectomes". Below
it, "Cohort Counts" shows one box per group (CN, MCI, AD) plus a "Total
(dense)" box, and three bar charts. The third chart, "Active AAL3 rescue
cohort", shows 0 for every group on most machines. That is normal. Press
`Ctrl+C` in the terminal to stop the dashboard.

### Option B: the data server and the React web app

Build the web app once (needs Node.js):

```bash
cd apps/connectome_web
npm ci
npm run build
cd ../..
```

`npm ci` ends with a few lines about "funding" and "vulnerabilities". They are
expected. Do **not** run `npm audit fix`: it rewrites `package-lock.json` and
can break the build. Details in
[connectome_web/README.md](connectome_web/README.md).

Then start the data server:

```bash
python -m uvicorn apps.connectome_api.main:app --host 127.0.0.1 --port 8001
```

What you should see:

```
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8001 (Press CTRL+C to quit)
```

Open http://127.0.0.1:8001/next/ for the web app, or
http://127.0.0.1:8001/api/docs for a clickable list of every data request.
Press `Ctrl+C` to stop.

## Where the viewers read results from

An *environment variable* is a named setting that your terminal passes to
every program it starts. `export NAME=value` sets one for the current terminal
only. The viewers read their settings once, when they start, so restart a
viewer after you change one.

| What | Data server, web app, core library | Streamlit dashboard | Default location |
|---|---|---|---|
| Analysis results | `CONNECTOME_ANALYSIS_ROOT`, and if that is not set, `SC_ANALYSIS_ROOT` | `CONNECTOME_ANALYSIS_ROOT` only | `data/derivatives/qc/analysis_cohort` |
| Connectome matrices | `CONNECTOME_MATRICES_DIR`, and if that is not set, `SC_CONNECTOMES_DIR` | `CONNECTOME_MATRICES_DIR` only | `data/derivatives/connectomes` |
| Repository root | `CONNECTOME_PROJECT_ROOT` | not settable: always the repository that holds the app | the repository |

The `SC_*` variables are resolved in `sc_config.py`. Their defaults chain from
`SC_DATA_ROOT` (default `<repo>/data`) through `SC_DERIV_ROOT` (default
`$SC_DATA_ROOT/derivatives`). To see what they resolve to on your machine:

```bash
python -c "import sc_config; print(sc_config.describe())"
```

**The Streamlit dashboard does not read the `SC_*` variables.** If you moved
your data with `SC_DATA_ROOT`, `SC_DERIV_ROOT`, `SC_ANALYSIS_ROOT` or
`SC_CONNECTOMES_DIR`, copy the resolved folders into the two `CONNECTOME_*`
variables before you start it:

```bash
export CONNECTOME_ANALYSIS_ROOT="$(python -c 'import sc_config; print(sc_config.paths().analysis_root)')"
export CONNECTOME_MATRICES_DIR="$(python -c 'import sc_config; print(sc_config.paths().connectomes_dir)')"
```

The matrix files follow one naming rule:
`<connectomes_dir>/SC_AAL166_<SUBJECT>_I<IMAGEID>_<type>.csv`, a 166 x 166
table of numbers with no header, where `<type>` is one of `count`, `fd_sum`,
`len_mean`, `fa_mean`, `md_mean`, `rd_mean`, `ad_mean`, `count_invnodevol`,
`invlen_mean`.

The viewers show three diagnostic groups: CN (cognitively normal), MCI (mild
cognitive impairment) and AD (Alzheimer's disease). By the project's grouping
rule, participants labelled EMCI, LMCI or SMC in ADNI are counted as MCI.

## Ports

A *port* is a numbered door on your computer that a server listens on. Two
programs cannot use the same port at the same time.

| Port | Used by |
|---|---|
| 8001 | the data server (`connectome_api`). The web app's development server and its browser tests expect it here. |
| 5173 | the web app's development server (`npm run dev`) |
| 4173 | the web app's preview server (`npm run preview`) |
| 8501 | the Streamlit dashboard |
| 8502 | a suggested port for the monitor, so it does not clash with the dashboard |

## Logins and passwords

None of the viewers has a login screen of its own.

- **On your own computer** every command above listens on `127.0.0.1` only,
  so only you can open the pages. Keep it that way: do not change the
  address to `0.0.0.0`. That would let anyone on your network see
  participant-level data. The same happens if you leave
  `--server.address 127.0.0.1` out of a Streamlit command: without it,
  Streamlit listens on every network connection of your computer.
- **On the project's shared server** a web server called nginx sits in front
  of the viewers and asks for a username and password (HTTP "basic auth").
  The setup is described in `deploy/README_connectome_dashboard.md` and
  `deploy/FASTAPI_REACT_RUNBOOK.md`.
- **The credentials file lives outside the repository.** By convention its
  location is kept in the environment variable `CONNECTOME_CREDENTIALS_FILE`.
  No program in this repository reads that variable; it is a pointer for
  people. Never copy the credentials into the repository and never paste
  them into an issue, a README or a chat. As a safety net, `.gitignore`
  refuses file names that contain `credential` or `secret`.

## Checking the viewers against the published results

```bash
python -m pytest tests/dashboard_parity -q
```

**These 23 tests only pass on the project's own server. On any other machine
most of them fail, and that is expected; they are not a check for new
users.** They compare the data server and the core library with frozen
reference values that are kept in `research_audit/outputs/`, a folder that is
not part of the repository, at a location fixed in
`tests/dashboard_parity/conftest.py`. Some tests also read the imaging
server's progress files. To check your own setup, use
`python sc_doctor.py --analysis` and the health address of the data server
instead (see [connectome_api/README.md](connectome_api/README.md)).

On the project's server the result is:

```
.......................                                                  [100%]
23 passed in 15.18s
```

## If something goes wrong

| What you see | Why | Fix |
|---|---|---|
| `No module named streamlit` | The dashboard packages are not installed, or the venv is not switched on. | `source .venv/bin/activate`, then `pip install -r requirements/dashboard.txt` |
| `No module named uvicorn` | Same, for the data server. | `source .venv/bin/activate`, then `pip install -r requirements.txt` |
| `ModuleNotFoundError: No module named 'apps'` | The data server was started from the wrong folder. | `cd` to the repository root and start it again. |
| `Port 8501 is not available` or `address already in use` | Another program already uses that port. | Stop the other program, or choose another port. |
| `Analysis root not found: ...` (Streamlit) | The dashboard cannot find the results folder. | Run the analysis, or set `CONNECTOME_ANALYSIS_ROOT` (see above). |
| `Could not load this view` (web app) | The data server is not running, or the results are missing. | See [connectome_web/README.md](connectome_web/README.md#if-something-goes-wrong). |

Each viewer's README has a longer list with the exact messages.
