# connectome_dashboard — the Streamlit dashboard

## What is this folder?

This is the original, complete results dashboard. It is one Python script,
`connectome_app.py`, run by Streamlit (a tool that turns a Python script into
a web page). It reads the analysis result files (CSV tables, JSON files,
figures) and the connectome matrices, and shows them as charts and tables in
your browser, grouped into 18 sections such as Demographics, Global DTI,
LR-SR Analysis, ML Diagnostics and SC Matrix Viewer.

It only reads. Browsing the dashboard never changes a result file.

The folder also keeps an older command, `refresh_connectome_dashboard_data.py`,
that re-runs the analysis. It now simply passes your request on to the real
analysis runner (`python -m connectome_analysis.run_analysis`).

## Do I need it?

- **Yes**, if you want to look at the results in a browser and prefer a single
  command. This is the simplest viewer to start.
- **No**, if you use the newer React web app (`apps/connectome_web/` together
  with `apps/connectome_api/`). Both show the same result files.
- **No**, for producing results. The analysis runs without it.

## What's inside

| File | What it does |
|---|---|
| `connectome_app.py` | The dashboard. One large Streamlit script: the sidebar, the 18 section pages, the matrix viewer, and a "Live Pipeline Monitor" view for the imaging server. |
| `refresh_connectome_dashboard_data.py` | Old entry point for re-running the analysis. Forwards to `python -m connectome_analysis.run_analysis` (see [Refreshing the results](#refreshing-the-results)). |
| `run_connectome_dashboard_refresh_loop.sh` | Runs the refresh command again and again, with a pause in between. Meant for the shared server. |
| `start_connectome_dashboard_refresh_tmux.sh` | Starts that loop inside a `tmux` session (a terminal that keeps running after you log out) named `connectome_dashboard_refresh`. Use it only on the project's server; see [The refresh loop](#the-refresh-loop-shared-server-only). |

## Before you start

1. The project is installed (Install section of the
   [top-level README.md](../../README.md)) and the venv is on:

   ```bash
   source .venv/bin/activate
   ```

2. Install the dashboard's extra packages (Streamlit, Plotly, Altair,
   scikit-image). They are kept separate because the analysis does not need
   them:

   ```bash
   pip install -r requirements/dashboard.txt
   ```

3. The analysis has been run, so the results folder exists (see
   [connectome_analysis/README.md](../../connectome_analysis/README.md)):

   ```bash
   python -m connectome_analysis.run_analysis --all --resume
   ```

## How to run it

**Step 1. Start the dashboard** from the repository root:

```bash
python -m streamlit run apps/connectome_dashboard/connectome_app.py \
  --server.address 127.0.0.1 --server.port 8501 \
  --server.headless true --browser.gatherUsageStats false
```

What the options mean:

- `--server.address 127.0.0.1`: only this computer can open the page.
- `--server.port 8501`: the port (numbered door) to listen on.
- `--server.headless true`: do not try to open a browser or ask questions in
  the terminal.
- `--browser.gatherUsageStats false`: do not send usage statistics to
  Streamlit.

What you should see:

```
  You can now view your Streamlit app in your browser.

  URL: http://127.0.0.1:8501
```

The terminal stays busy while the dashboard runs.

**Step 2. Open http://127.0.0.1:8501 in your browser.** The first load can
take a little while. The browser tab says "Connectome C Dashboard"; the big
heading on the page says "Analysis of Structural connectomes". Below it you
should see "Cohort Counts" with one box per group (CN, MCI, AD) and a
"Total (dense)" box, then three bar charts:

- subjects per group, from `02_demographics/group_counts.csv`;
- final connectomes per group, from `01_qc/data_completeness.csv`;
- "Active AAL3 rescue cohort". This one counts participants in extra imaging
  batches, read from a fixed folder on the imaging server (environment
  variable `AAL3_RESCUE_BATCH_ROOT`). When there are no such batches, or the
  folder does not exist, it shows 0 for every group. The zeros are normal.

Below that is a grid of section buttons. Click one to open that section.

**Step 3. Use the sidebar.** Click the `>>` arrow at the top left to open it.

- **View**: "📊 Analysis" (the results, the default) or "🔴 Live Pipeline
  Monitor" (imaging progress; works only on the imaging server).
- **Controls** (click to expand):
  - the results folder being read (`Root: ...`);
  - "Show subject-level/raw tables": off by default, so tables with one row
    per participant stay hidden;
  - "Default live charts per section": how many charts each section draws
    (1 to 12, default 4);
  - "Hide visual outliers in live charts": on by default. It only hides
    extreme points in the charts; tables and statistics are unchanged;
  - "Refresh inventory": re-scan the results folder now instead of waiting.

**Step 4. Stop the dashboard** with `Ctrl+C` in its terminal.

## Inputs

| Input | Where | Required? |
|---|---|---|
| Analysis results | `$CONNECTOME_ANALYSIS_ROOT` (default `data/derivatives/qc/analysis_cohort`) | Yes. Every section reads its numbered folder, for example `00_master/`, `03_global_microstructure_live/`, `12_length_delay/`, `18_ml_diagnostics/`, `19_network_analysis/`. |
| Connectome matrices | `$CONNECTOME_MATRICES_DIR` (default `data/derivatives/connectomes`) | For the SC Matrix Viewer and the tract-length views. Files: `SC_AAL166_<SUBJECT>_I<IMAGEID>_<type>.csv`, headerless 166 x 166, comma-separated. |
| Atlas | `atlas/AAL/aal3_labels_166.csv`, `atlas/AAL/AAL3v1_1mm.nii.gz` (shipped with the repository) | Yes, for region names and brain positions. |
| Extra model-search runs | `$CONNECTOME_ENHANCED_ML_ROOT` (default `outputs/`) | No. Those panels stay empty without it. |
| Brain mask for 3-D views | `$CONNECTOME_MNI_BRAIN_MASK` | No. Falls back to a standard FSL mask if found, else to the AAL atlas. |
| Imaging progress files | fixed locations on the imaging server | Only for the "Live Pipeline Monitor" view. |

## Outputs

Browsing writes nothing. The dashboard only reads files.

(On the imaging server, the "🔄" buttons in the Live Pipeline Monitor view
start a script that recalculates imaging progress figures. They do nothing
useful anywhere else.)

## Settings you can change

| Setting | Where | What it does | Default |
|---|---|---|---|
| `CONNECTOME_ANALYSIS_ROOT` | environment variable | The results folder to show. | `<repo>/data/derivatives/qc/analysis_cohort` |
| `CONNECTOME_MATRICES_DIR` | environment variable | The matrix folder. | `<repo>/data/derivatives/connectomes` |
| `CONNECTOME_ENHANCED_ML_ROOT` | environment variable | Folder with optional extra model-search runs. | `<repo>/outputs` |
| `CONNECTOME_MNI_BRAIN_MASK` | environment variable | A brain-mask image for the 3-D views. | not set |
| `--server.port` | Streamlit option | Port number. | `8501` |
| `--server.address` | Streamlit option | Who may connect. Keep `127.0.0.1`. | `127.0.0.1` in the command above. If you leave this option out, Streamlit listens on all network interfaces and anyone on your network can open the page. Always keep `--server.address 127.0.0.1`. |
| Sidebar controls | in the page | See Step 3 above. | as listed |

**Important: this dashboard does not read the `SC_*` variables** from
`sc_config.py`. If you keep your data somewhere else and told the analysis so
with `SC_DATA_ROOT`, `SC_DERIV_ROOT`, `SC_ANALYSIS_ROOT` or
`SC_CONNECTOMES_DIR`, copy the resolved folders into the dashboard's own
variables before starting it:

```bash
export CONNECTOME_ANALYSIS_ROOT="$(python -c 'import sc_config; print(sc_config.paths().analysis_root)')"
export CONNECTOME_MATRICES_DIR="$(python -c 'import sc_config; print(sc_config.paths().connectomes_dir)')"
python -m streamlit run apps/connectome_dashboard/connectome_app.py \
  --server.address 127.0.0.1 --server.port 8501 \
  --server.headless true --browser.gatherUsageStats false
```

Then open the sidebar, expand "Controls", and check that `Root:` shows the
folder you expect.

The results folder is re-scanned about once a minute. Press "Refresh
inventory" in the sidebar to re-scan at once. Environment variables are read
only at start-up: restart the dashboard after changing one.

## Refreshing the results

To produce or update the results, use the analysis runner directly:

```bash
python -m connectome_analysis.run_analysis --all --resume
```

The older command in this folder still works and passes your request on:

```bash
python apps/connectome_dashboard/refresh_connectome_dashboard_data.py --mode quick
```

What you should see (paths shortened):

```
[refresh] delegating to run_analysis: --group cohort --lock-timeout-sec 5 --config <tmp>/sc_refresh_.../analysis.yaml
connectomes   : <repo>/data/derivatives/connectomes
cohort tables : <repo>/cohort
...
analysis root : <repo>/data/derivatives/qc/analysis_cohort
plan          : 7 stage(s)
...
  [run]   demographics ...
.../scipy/stats/_stats_py.py:8819: RuntimeWarning: divide by zero encountered in scalar divide
...
  [ok]    demographics  (1.2s)
...
ok=7   total 9.7s
```

During the demographics stage you will see several `RuntimeWarning` lines
from scipy (the statistics library), such as `divide by zero encountered in
scalar divide` and `p-value cannot be estimated ...`. They are expected. The
run succeeded if it ends with `ok=7`.

What its options become:

| Option | Passed on as | Meaning |
|---|---|---|
| `--mode quick` (default) | `--group cohort` | Only the 7 cohort and QC stages. |
| `--mode full` | `--all` | Every stage. |
| `--resume` | `--resume` | Skip stages whose outputs are present and current. |
| `--deriv-root DIR` | `--deriv-root DIR` | Derivatives folder. |
| `--snapshot-mode provisional\|final` (default `provisional`) | a one-run copy of `configs/analysis.yaml` | The `analysis_snapshot` stage records which inputs a run used. `provisional` marks a working run; `final` marks the frozen state behind a report. |
| `--edge-perms N`, `--brain-age-repeats N`, `--strict-tracks-count` | a one-run copy of `configs/analysis.yaml` | Stage settings. |
| `--with-literature`, `--literature-mailto EMAIL` | `--stage literature_retrieval` | Also run the optional literature stage. |
| `--lock-timeout-sec N` (default 5) | `--lock-timeout-sec N` | How long to wait if another run holds the lock. |

The results go wherever `SC_ANALYSIS_ROOT` points. To write a trial run
somewhere else, set it first, for example
`export SC_ANALYSIS_ROOT="$HOME/sc_trial/analysis_cohort"`. Keep result
folders outside the repository (or inside the git-ignored `data/`): they hold
participant-level data and must never be committed.

### The refresh loop (shared server only)

`run_connectome_dashboard_refresh_loop.sh` runs the refresh command, waits,
and repeats until you press `Ctrl+C`. Its defaults are set for the project's
server, so on any other machine set these environment variables first:

| Variable | What it is | Server default |
|---|---|---|
| `PROJECT_ROOT` | the repository | the server's checkout |
| `PYTHON` | the Python to use | the server's venv |
| `LOG_DIR` | where each run's log file goes | the server's results `logs/` folder |
| `MODE` | `quick` or `full` | `quick` |
| `INTERVAL` | seconds to wait between runs | `300` |
| `SNAPSHOT_MODE` | `provisional` or `final` | `provisional` |

Example, from the repository root:

```bash
PROJECT_ROOT="$PWD" PYTHON="$PWD/.venv/bin/python" LOG_DIR="$PWD/logs/dashboard_refresh" \
  apps/connectome_dashboard/run_connectome_dashboard_refresh_loop.sh
```

(`logs/` is ignored by git, so the log files cannot be committed by accident.)

Each pass writes `dashboard_refresh_<mode>_<time>.log` into `LOG_DIR` and ends
with a line like `[2026-09-18T15:27:07Z] sleeping 300s`. On a laptop you
rarely need this: results change only when you re-run the analysis.

**Use `start_connectome_dashboard_refresh_tmux.sh` only on the project's
server.** It passes `PROJECT_ROOT`, `APP_ROOT`, `MODE`, `INTERVAL` and
`SNAPSHOT_MODE` on to the loop, but not `PYTHON` or `LOG_DIR`. If a `tmux`
server is already running, the loop then falls back to the project server's
own Python and log folder. Elsewhere, run the loop script directly with
`PROJECT_ROOT`, `PYTHON` and `LOG_DIR` set, as shown above.

## Logins and passwords

The dashboard has no login of its own. With `--server.address 127.0.0.1`, only
your own computer can open it. Do not change that to `0.0.0.0`; with
"Show subject-level/raw tables" switched on, the page shows one row per
participant.

On the project's shared server, nginx (a web server) sits in front of the
dashboard and asks for a username and password; see
`deploy/README_connectome_dashboard.md`. The credentials file is kept outside
the repository, and by convention its location is kept in the environment
variable `CONNECTOME_CREDENTIALS_FILE`. No code reads that variable. Never copy
the credentials into the repository.

## If something goes wrong

**`.../python: No module named streamlit`**
The dashboard packages are not installed, or the venv is off. Run
`source .venv/bin/activate` and `pip install -r requirements/dashboard.txt`.

**`Port 8501 is not available`**
Something else already uses port 8501, often an earlier copy of the dashboard.
Stop it with `Ctrl+C` in its terminal, or start this one with
`--server.port 8502`.

**The page shows a red box: `Analysis root not found: ...`**
The dashboard cannot find the results folder named in the message. Either the
analysis has not been run, or your results are somewhere else. Set
`CONNECTOME_ANALYSIS_ROOT` (see [Settings](#settings-you-can-change)) and
restart.

**A section is empty or has fewer charts than expected**
Only part of the analysis has been run. Run
`python -m connectome_analysis.run_analysis --all --resume`, then press
"Refresh inventory".

**"Live Pipeline Monitor" shows `Live status not available yet: [Errno 2] No such file or directory: ...`
or `The HCP379-v2 status artifacts are not available.`**
Expected on any machine other than the imaging server: that view reads
imaging progress files that are not part of the repository. Switch the
sidebar back to "📊 Analysis".

**The terminal prints `Collecting usage statistics. To deactivate, set browser.gatherUsageStats to false.`**
Harmless. Add `--browser.gatherUsageStats false` to the command to turn it off.
