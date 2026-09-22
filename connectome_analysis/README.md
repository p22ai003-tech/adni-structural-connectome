# connectome_analysis: the analysis and machine-learning pipeline

## What is this folder?

This folder holds the second half of the project: the **analysis**. It starts
from connectome matrices that already exist, one set per participant, and
produces every statistics table, machine-learning result and figure that the
thesis and the dashboard report.

A **connectome matrix** is a table of 166 rows and 166 columns. Each row and
each column is one brain region from the AAL3 atlas. The number in row *i*,
column *j* says something about the white-matter fibres between region *i* and
region *j*: how many there are, how long they are, or how healthy they look.

The analysis is split into 37 small steps called **stages**. Each stage reads
some files, does one job, and writes its results into a numbered folder. One
command, `python -m connectome_analysis.run_analysis`, runs the stages in the
right order. That command is called **the runner**.

## Do I need it?

- **Yes**, if you have connectome matrices and cohort tables and want the
  results: group comparisons of CN, MCI and AD, the network tables, the
  machine-learning models and the tables the dashboard shows.
- **No**, if you want to make the matrices from MRI images. That is the
  imaging half. See the Imaging section of the top-level `README.md`. This
  folder does not describe how matrices are made.
- **No**, if you only want to look at results that someone else already
  computed. Point the dashboard at their analysis folder instead: set
  `CONNECTOME_ANALYSIS_ROOT` to that folder before you start the dashboard.
  The Streamlit dashboard reads only its own `CONNECTOME_*` variables, not
  `SC_ANALYSIS_ROOT`. See
  [apps/connectome_dashboard/README.md](../apps/connectome_dashboard/README.md).

## Words used in this README

**Computer words**

| Word | Meaning |
|---|---|
| repository root | The top folder of this project, the one holding `README.md` and `sc_config.py`. Run every command from there. |
| venv | A private Python installation for this project, in the folder `.venv`. You switch it on with `source .venv/bin/activate`. |
| environment variable | A named setting that the terminal hands to every program it starts, such as `SC_CONNECTOMES_DIR`. Set it for the rest of the terminal session with `export SC_CONNECTOMES_DIR=/path/to/matrices`, or for one command only by writing it in front: `SC_CONNECTOMES_DIR=/path/to/matrices python sc_doctor.py`. See its value with `echo $SC_CONNECTOMES_DIR`. |
| exit status | A number every command leaves behind when it ends: 0 means success, anything else means a problem. Type `echo $?` straight after a command to see it. |
| YAML | A plain-text format for settings: `key: value` lines, with indentation (spaces at the start of a line) showing which setting belongs under which. `configs/analysis.yaml` is a YAML file. |
| tmux | A terminal program that keeps your commands running when your window closes or your connection drops. Start one with `tmux new -s analysis`, leave it running with Ctrl-b then d, and come back with `tmux attach -t analysis`. A machine restart still stops it. |
| stage | One step of the analysis, for example `demographics` or `edr_exceptions`. |
| group (of stages) | Stages come in five groups: `cohort`, `measures`, `inference`, `ml`, `build`. |
| DAG | "Directed acyclic graph": a list of stages where each stage names the stages it needs first. The runner reads this list and works out a safe order. |
| analysis root | The folder all results go into. Default: `$SC_DERIV_ROOT/qc/analysis_cohort`. |
| derivatives | Files computed from the raw images: processed images, the connectome matrices, quality tables. They live under `$SC_DERIV_ROOT`. |
| ledger | A CSV file the runner writes at the end of every run: one row per stage, with its status and time. |

**Study words**

| Word | Meaning |
|---|---|
| subject | One participant. ADNI gives each a Subject ID of the form `NNN_S_NNNN` (three digits, `_S_`, then four or five digits). This README writes `<SUBJECT>` or `XXX_S_0001` instead of a real one. |
| cohort | The set of subjects being analysed. The **cohort tables** are the CSV files that list them, with their group, age, sex and test scores. |
| QC | Quality control: checks that data are usable before they are analysed. |
| CN, MCI, AD | Cognitively normal, mild cognitive impairment, Alzheimer's disease: the three diagnostic groups compared. |
| MMSE | Mini-Mental State Examination: a 30-point test of memory and thinking. Lower means more impairment. |
| CDR | Clinical Dementia Rating. The Global CDR is 0 (normal), 0.5 (very mild), 1 (mild), 2 (moderate) or 3 (severe). |
| FAQ, GDSCALE, NPI-Q | Three more questionnaires in the ADNI tables: Functional Activities Questionnaire (everyday tasks), Geriatric Depression Scale (mood) and Neuropsychiatric Inventory Questionnaire (behaviour). |
| APOE | A gene. Everyone carries two copies (alleles), recorded in the `APOE A1` and `APOE A2` columns. The e4 form is linked to a higher risk of Alzheimer's disease. |
| tractography | Using diffusion MRI to trace the likely paths of white-matter fibres through the brain. This happens in the imaging half, not here. |
| streamline | One traced fibre path produced by tractography. The `count` matrix counts them. |
| fibre density, `fd_sum` | Each streamline carries a weight that estimates how much real fibre it stands for. `fd_sum` adds up those weights for all streamlines joining two regions. It is the main measure of connection strength here. |
| MRtrix3, `tckinfo` | MRtrix3 is the diffusion-MRI software used by the imaging half. `tckinfo` is its tool that reports how many streamlines a track file holds. The analysis does not need it. |
| FA, MD, AxD, RD | Four diffusion measures of white-matter tissue: fractional anisotropy, mean diffusivity, axial diffusivity and radial diffusivity. This README writes axial diffusivity as AxD so it is never confused with the AD group. In file names it is `ad_mean`. |
| EDR | Exponential distance rule: longer connections are weaker. An "EDR exception" is a connection much stronger than its length predicts. |
| SR, MR, LR | Short-range, mid-range and long-range connections, split by fibre length. |
| FDR | False discovery rate, a correction used when many tests are run at once. |
| cross-validation | A fair way to test a model: split the subjects into k parts, train on k − 1 of them, test on the part left out, and repeat until every part has been the test part once. |
| AUC, PR-AUC | Two scores from 0 to 1 for how well a model tells groups apart: the area under the ROC curve and the area under the precision–recall curve. Higher is better; an AUC of 0.5 is guessing. |
| SHAP | A method that says how much each input pushed a machine-learning prediction up or down. |

## What's inside

**The runner**

| File | What it does |
|---|---|
| `run_analysis.py` | The runner. Reads the stage list, picks the stages you asked for, checks their inputs, runs them in order and writes the ledger. |
| `stages.py` | The stage list (the DAG). For each of the 37 stages: its group, a one-line summary, the stages it needs, and the files it must produce. |
| `relabel_edr_rois.py` | A repair tool for old `17_edr_exceptions` files that carry wrong region names. See [Repairing region names](#repairing-region-names-relabel_edr_roispy). |
| `analysis_config.py` | Works out the analysis folders from `sc_config.py`, and sets the plot style. |

**Modules the runner calls** (one line each; the stage names are in brackets)

| File | What it does |
|---|---|
| `analysis_cohort.py` | Builds the master cohort table from the cohort CSVs and the matrix folder; counts, demographics and data completeness (`master_cohort`, `qc_snapshot`, `demographics`, `data_completeness`). |
| `analysis_graph.py` | Matrix availability per subject and per group (`connectome_qc`). Its other functions are older versions not used by the runner. |
| `analysis_live_connectome.py` | Per-subject measures read straight from the matrices: whole-brain and per-region FA/MD/AxD/RD, graph measures (strength, degree, efficiency), coupling between them, and the brain-age model (the `live_*` stages). |
| `analysis_network.py` | Groups the 166 regions into brain networks and anatomical systems and compares the groups network by network (`network_analysis`). |
| `analysis_edges.py` | Loads a stack of matrices for the cohort; tests every connection (edge) between groups, with a permutation correction (`edgewise_fd_sum`). |
| `analysis_length_delay.py` | Short-, medium- and long-range connections by length tertiles (`lr_sr`); a conduction-delay estimate from fibre length (`delay`). |
| `analysis_edr_exceptions.py` | Finds EDR exception edges in every subject and compares them across groups (`edr_exceptions`). |
| `analysis_advanced.py` | Fits the EDR model on the CN group and measures how far each subject departs from it (`advanced_structural`). |
| `analysis_clinical.py` | Collects older covariate-adjusted clinical tables into one place, if they exist (`clinical`). |
| `analysis_findings.py` | Writes every FDR-significant network result, and every network measure that changes with age, as one row of a findings catalogue (`findings_catalog`). |
| `analysis_mediation.py` | Mediation tests: does measure X affect outcome Y through measure M? (`mediation`). |
| `analysis_ml_diagnostics.py` | The machine-learning sweep: diagnosis, clinical-score prediction and CDR classification from connectome features (`ml_diagnostics`). |
| `analysis_clinical_outcome_search.py` | A wider model search against MMSE and CDR (`clinical_outcome_search`). Also runnable on its own, see [Running the model search on its own](#running-the-model-search-on-its-own). |
| `literature_openalex.py` | Looks up papers for each finding on OpenAlex, a free paper index (`literature_retrieval`, optional). |

**Shared helpers**

| File | What it does |
|---|---|
| `analysis_stats.py` | Statistical tests used everywhere: Kruskal–Wallis, Brunner–Munzel, Cliff's delta, permutation tests, BH-FDR, Friedman and Wilcoxon tests between networks. |
| `analysis_plots.py` | Saving tables, writing the short `*_inference.md` notes, and the standard plots. |

**Not called by the runner** (kept for other tools or for history)

| File | What it is |
|---|---|
| `analysis_brain_age.py`, `analysis_coupling.py`, `analysis_dti.py` | Older versions of the brain-age, coupling and microstructure analyses. They read tables from an older folder layout. Replaced by `analysis_live_connectome.py`. |
| `analysis_exports.py` | Helpers for writing Excel bundles and export manifests. |
| `analysis_sc_matrix_qc.py` | A detailed matrix quality audit used by scripts in `scripts/`. |
| `analysis_ml_goal_search.py`, `analysis_ml_neural_benchmarks.py` | Stand-alone model-search experiments with their own command lines. |
| `novelty_report.py` | Assembles `20_findings/novelty_report.md` (read by the dashboard) from `novelty_cards.json`, the findings catalogue and the literature records. Nothing in the runner makes `novelty_cards.json`, and the folder it reads is fixed inside the file. |
| `hcp379_verified_handoff.py`, `hcp379_balanced_verified_handoff.py` | Read-only loaders for a separate HCP-379 atlas dataset. Not part of this analysis. |

The `build` group of stages runs scripts that live in a different folder,
`hcp_analysis/` (files named `build_*.py`). The runner starts each one as a
separate program. See `hcp_analysis/README.md` for what each script does.

## Before you start

You need four things.

1. **The Python environment.** Follow the Install section of the top-level
   `README.md`. In short, from the repository root:

   ```bash
   python3.12 -m venv .venv
   .venv/bin/pip install -r requirements.txt
   source .venv/bin/activate
   ```

   Python 3.12 is required. Every command below assumes the venv is switched
   on and that you are in the repository root.

2. **Connectome matrices**: a set of CSV files for each subject. How they are
   made is in the Imaging section of the top-level `README.md`. The exact file
   names and format are under [Inputs](#inputs). The analysis looks for them
   in `$SC_CONNECTOMES_DIR`, which is `data/derivatives/connectomes` unless you
   set it. A fresh copy of the repository has no `data/` folder, so choose one
   of these two ways:

   - **Copy them into the default folder** (simplest). Replace
     `/path/to/your/matrices` with the folder that holds your files:

     ```bash
     mkdir -p data/derivatives/connectomes
     cp /path/to/your/matrices/SC_AAL166_*.csv data/derivatives/connectomes/
     ```

   - **Leave them where they are** and tell the analysis where to look:

     ```bash
     mkdir -p data
     export SC_CONNECTOMES_DIR=/path/to/your/matrices
     ```

     `export` lasts until you close the terminal, so type it again in every
     new terminal. The `data/` folder is still needed: the results are written
     below it, and Step 1 reports a missing `data/` as a problem.

3. **Cohort tables** in `$SC_COHORT_DIR` (default `cohort/`): four CSV files
   built by `python sc_cohort.py build` from spreadsheets you export from ADNI.
   See `python sc_cohort.py --help`.

4. **Optional: the exclusions file** `configs/exclusions.local.yaml`. Without
   it the run still works, but the machine-learning numbers will differ from
   the published ones. See [The exclusions file](#the-exclusions-file).

**About the data.** ADNI images and clinical tables come only from
[adni.loni.usc.edu](https://adni.loni.usc.edu/), under your own ADNI Data Use
Agreement. They are not in this repository. Never commit matrices, cohort
tables, the exclusions file or any result folder to git. `.gitignore` already
blocks the usual locations (`data/`, `cohort/`, `*.local.yaml`, `SC_AAL*_*.csv`).

## How to run it

### Step 1. Check the machine

```bash
python sc_doctor.py --analysis
```

`sc_doctor.py` (the "doctor") checks the paths, the Python packages, the cohort
tables and the exclusions file. What you should see (paths shortened to
`<repo>`):

```
<repo>/sc_doctor.py:148: DeprecationWarning: Accessing jsonschema.__version__ is deprecated ...
  _print(OK, mod, getattr(m, "__version__", ""))
========================================================================
pipeline preflight
========================================================================

Paths
------------------------------------------------------------------------
  ok    project_root               <repo>
  ok    data_root                  <repo>/data
  ...
  ok    connectomes_dir            <repo>/data/derivatives/connectomes
  ok    cohort_dir                 <repo>/cohort
...
Cohort tables
------------------------------------------------------------------------
  ok    dti.csv                    <repo>/cohort/dti.csv
  ok    mri.csv                    <repo>/cohort/mri.csv
  ok    dti_master.csv             <repo>/cohort/dti_master.csv
  ok    mri_master.csv             <repo>/cohort/mri_master.csv
  ok    exclusions                 1 subject(s) from <repo>/configs/exclusions.local.yaml

========================================================================
no problems found.
```

How to read it:

- **The two `DeprecationWarning` lines at the top are harmless.** They come
  from a Python package and do not mean anything is wrong.
- **Fix every `MISS` line before going on.** For example
  `MISS  data_root  <repo>/data   <- set SC_DATA_ROOT` means the `data/`
  folder does not exist: see [Before you start](#before-you-start), item 2. A
  missing cohort table shows as `MISS  dti.csv ... <- build with: python
  sc_cohort.py build --exports <folder>`.
- **These `warn` lines are normal** for someone who only runs the analysis:
  `raw_images_root`, `raw_dwi_root`, `raw_t1_root`, `deriv_root`, `qc_root`,
  `analysis_root`, `run_state_dir` and `manifest` (folders for the imaging
  half, or folders the runner creates itself), `tabpfn` and `pytorch_tabnet`
  (optional packages that `requirements.txt` does not install), and
  `exclusions` (see [The exclusions file](#the-exclusions-file)).
- **One `warn` line is not normal:** `warn  connectomes_dir ... (absent;
  created on demand)`. It means the matrix folder does not exist, so the
  analysis has no matrices. The doctor still says `no problems found.` in this
  case, so look for it yourself.

The doctor's exit status is 0 when nothing required is missing and 1
otherwise.

The doctor does not look inside the matrix folder. Check that the matrices are
really there by counting the `count` files, one per subject:

```bash
ls data/derivatives/connectomes/SC_AAL166_*_count.csv | wc -l
```

If you set `SC_CONNECTOMES_DIR`, use this instead:

```bash
ls "$SC_CONNECTOMES_DIR"/SC_AAL166_*_count.csv | wc -l
```

It prints one number: how many subjects have a matrix (530 on the machine
this README was tested on). If it prints
`ls: cannot access ...: No such file or directory` and then `0`, the folder is
empty or the files are named differently; see [Inputs](#inputs).

### Step 2. Check the cohort tables

```bash
python sc_cohort.py check
```

What you should see:

```
cohort folder: <repo>/cohort

  ok       dti.csv            <rows> rows
  ok       mri.csv            <rows> rows
  ok       dti_master.csv     <rows> rows
  ok       mri_master.csv     <rows> rows

no problems found.
```

If the tables are missing you see this instead, and the exit status is 1:

```
cohort folder: <repo>/cohort

  MISSING  dti.csv
  MISSING  mri.csv
  MISSING  dti_master.csv
  MISSING  mri_master.csv

4 problem(s).
```

Build them with `python sc_cohort.py build --exports <folder with your ADNI
exports>` (see `python sc_cohort.py --help`), or, if they are somewhere else,
point `SC_COHORT_DIR` at that folder. Then run the check again.

### Step 3. Look at the stage list

```bash
python -m connectome_analysis.run_analysis --list
```

This prints all 37 stages in the order they will run, grouped, with what each
needs and makes. It reads no data. The start looks like this:

```
--- cohort ----------------------------------------------------
   0. analysis_snapshot
      Freeze the input snapshot (provisional or final).
      needs: -
      makes: 1 file(s), e.g. 00_master/analysis_snapshot.csv
   1. master_cohort
      Build the master cohort table; the spine of every later stage.
      needs: analysis_snapshot
      makes: 1 file(s), e.g. 00_master/master_cohort.csv
...
  36. build_shap_overall               [script]
      Seed-averaged SHAP over all features; supersedes the earlier tables.
      needs: build_ml_explain_v2, build_explain_v3
      makes: 6 file(s), e.g. 20_exception_specificity/ml_shap_class_group.csv

37 stages.
```

`[script]` marks a stage that runs a `hcp_analysis/build_*.py` script as a
separate program. `[optional]` marks the two stages that `--all` leaves out:
`literature_retrieval`, and `clinical_outcome_search`, an exploratory model
search that took about 7½ hours of a 14-hour full run and whose output nothing
else reads.

### Step 4. Do a dry run

A **dry run** prints the plan and checks where files will go, but computes
nothing.

**Before you press Enter:** a dry run is not completely silent. It creates the
analysis root and its basic sub-folders (`00_master/`, `exports/`, `logs/`,
`figures/`, `tables/`, `99_appendix/`), and it **replaces** the ledger
`exports/stage_status.csv` and `00_master/run_analysis_status.json`. On a first
run that does not matter. If the analysis root already holds results from an
earlier run and you want to keep their ledger, add
`--analysis-root <some other folder>` to the command.

```bash
python -m connectome_analysis.run_analysis --all --dry-run
```

What you should see:

```
connectomes   : <repo>/data/derivatives/connectomes
cohort tables : <repo>/cohort
exclusions    : 1 subject(s) from <repo>/configs/exclusions.local.yaml
analysis root : <repo>/data/derivatives/qc/analysis_cohort
config        : <repo>/configs/analysis.yaml
plan          : 36 stage(s) [dry-run]

  [dry]   analysis_snapshot  -> 1 output(s)
  [dry]   master_cohort  -> 1 output(s)
  ...
  ...
  [dry]   build_shap_overall  -> 6 output(s)

dry-run=35   total 0.3s
ledger: <repo>/data/derivatives/qc/analysis_cohort/exports/stage_status.csv
```

Read the first five lines carefully. They say which matrices, which cohort
tables, which exclusions file and which output folder the real run will use.

`--all` selects 35 stages, not 37: the two optional stages are left out
unless you name them. `--group` and `--from` leave them out too; only `--stage`
runs one:

```bash
python -m connectome_analysis.run_analysis --stage clinical_outcome_search
```

### Step 5. Run everything

**Before you press Enter**, read these four points:

1. **It takes hours**: about 5.5 hours when it was measured for this README
   (see [How long it takes](#how-long-it-takes)). Start it inside `tmux` so it
   keeps going if your window closes or your connection drops. A new `tmux`
   window may not have your settings, so switch the venv on again there, and
   repeat any `export SC_...` lines you use:

   ```bash
   tmux new -s analysis
   source .venv/bin/activate
   ```

2. **Keep a log.** The command below saves everything the run prints in
   `analysis_run.log` (git ignores `*.log` files) as well as showing it on the
   screen. If the run is stopped, this file is the only record of where it got
   to: a stopped run writes no ledger, and the ledger left on disk is the one
   from the *previous* run. In the command, `2>&1` sends error messages to the
   same place as normal output, `| tee analysis_run.log` copies the output into
   the file, and `-u` makes Python print each line at once. Without `-u` the
   lines in the file come out of order: `[FAIL]` and `[BLOCK]` lines can land
   above lines that were printed before them.
3. **It overwrites three figures in `docs/figs/`** (`fig4_shap_mmse.png`,
   `fig5_ice_mmse.png`, `fig6_mmse_buckets.png`). Git tracks that folder, so
   `git status` will list them as changed afterwards. To send them somewhere
   else, use the recipe in
   [Trying it without touching existing results](#trying-it-without-touching-existing-results).
4. **One stage is known to fail**: `build_range_restricted_networks`. Two
   commands finish the job afterwards; see below.

```bash
python -u -m connectome_analysis.run_analysis --all 2>&1 | tee analysis_run.log
```

Because of `| tee`, `echo $?` now shows the exit status of `tee` (normally 0).
To see the runner's own exit status, type `echo ${PIPESTATUS[0]}` (in the bash shell) straight
after the run ends.

Each stage prints a `[run]` line when it starts and an `[ok]` line with its
time when it finishes:

```
plan          : 36 stage(s)

  [run]   analysis_snapshot ...
  [ok]    analysis_snapshot  (0.2s)
  [run]   master_cohort ...
  [ok]    master_cohort  (4.8s)
  ...
  [run]   network_analysis ...
  [ok]    network_analysis  (1045.5s)
  ...
```

At the end it prints a count of each outcome and the path of the ledger. When
every stage succeeds it looks like this:

```
ok=36   total ...s
ledger: <analysis root>/exports/stage_status.csv
```

The exit status is 0 when every stage is `ok` (or `skipped`), and 1 when any
stage is `failed`, `blocked` or `incomplete`. A failed stage does not stop the
run: the runner goes on with every stage that does not depend on it.

With the current code one stage, `build_range_restricted_networks`, fails every
time the runner starts it. What happens next depends on the analysis root:

- **In a new, empty analysis root** the nine `build_*` stages after it are
  blocked, and the run ends with `blocked=9  failed=1  ok=26`.
- **In an analysis root that already holds results from an earlier run** the
  nine stages are *not* blocked. The runner only checks that their input files
  exist, and the old network tables from the earlier run are still there. So
  they run on those old tables, and the run can end with `failed=1  ok=35`.

Either way, **whenever `build_range_restricted_networks` fails, run the two
commands under
[If something goes wrong](#if-something-goes-wrong)** before you use the
results. They rebuild the network tables and then every stage after them.

You will also see many warning lines while it works, for example
`RuntimeWarning: divide by zero encountered in scalar divide` from `scipy`, or
`ConvergenceWarning` from `scikit-learn` during `ml_diagnostics`. They come
from tests on small or constant groups of values and from models that stop at
their iteration limit. They do not mean a stage failed; only a `[FAIL]`,
`[BLOCK]` or `[WARN]` line does.

### Step 6. Check the ledger

The **ledger** is a CSV file with one row per stage: its status, how long it
took, and the error message if it failed. The last line of the run output
gives its path. With the default settings:

```bash
cat data/derivatives/qc/analysis_cohort/exports/stage_status.csv
```

```
stage,group,status,seconds,finished_utc,produced
analysis_snapshot,cohort,ok,0.23,2026-09-18T06:43:16Z,1
master_cohort,cohort,ok,4.78,2026-09-18T06:43:21Z,1
...
```

The first five columns are always there. After them come whichever of these
apply to the run: `produced` (files written), `expected` (files promised, for
an incomplete stage), `reason` (why a stage was skipped), `missing` (how many
inputs a blocked stage lacked) and `error` (the message of a failed stage).
Each run replaces the ledger; it does not add to it. The ledger is written
only when a run reaches its end, so after a stopped run the file on disk still
describes the run before it. A short summary of the same run is in
`00_master/run_analysis_status.json`.

### Step 7. Continue after a stop, or redo part of the work

Pick the case that fits:

| Situation | Command |
|---|---|
| One stage failed, you fixed the cause, and you want it and everything after it | `python -m connectome_analysis.run_analysis --from <failed stage> --skip literature_retrieval` |
| The run was stopped (Ctrl-C, closed window, machine restart) during stage X | the same, with `--from X`. Find X in your log with `grep '\[run\]' analysis_run.log \| tail -1` (it prints a line like `  [run]   ml_diagnostics ...`). Do not look in the ledger: a stopped run writes none, so the ledger on disk is the one from the previous run. |
| You want to redo one stage only | `python -m connectome_analysis.run_analysis --stage <stage>` |
| You have new or changed matrices, cohort tables or parameters | `python -m connectome_analysis.run_analysis --all` (no `--resume`) |

`--from X` does not re-run the stages before X; their result files must still
be on disk. See [Other ways to choose stages](#other-ways-to-choose-stages).

To keep logging a long re-run, write it the same way as in Step 5, for example
`python -u -m connectome_analysis.run_analysis --from X --skip literature_retrieval 2>&1 | tee -a analysis_run.log`.
`tee -a` adds to the end of the log instead of replacing it.

**What `--resume` does.** With `--resume` the runner skips a stage (and
prints `[skip]`) when all of its declared output files exist and none of them
is older than the output files of the stages it needs. The decision is made
from the files on disk, so deleting a result file is enough to make its stage
run again. On a folder where nothing has changed, every stage is skipped:

```
$ python -m connectome_analysis.run_analysis --group cohort --resume
plan          : 7 stage(s) [resume]

  [skip]  analysis_snapshot
  [skip]  master_cohort
  ...
  [skip]  functional_placeholder

skipped=7   total 0.3s
```

`--resume` is not a way to save time after a complete run. Four things to
know:

1. **It only compares files inside the analysis root.** It does not notice new
   matrices, changed cohort tables, a changed `configs/analysis.yaml` or
   changed code. After any of those, run without `--resume`.
2. **Re-running one stage often re-runs everything after it.** Most stages
   use the master cohort table, which the runner keeps in memory. If
   `master_cohort` was skipped and a later stage has to run, the runner first
   rebuilds the table and prints `(recomputing master_cohort for its in-memory
   result)`. That rewrites `00_master/master_cohort.csv`, so every later stage
   that reads it now looks out of date and runs as well. Real example, after
   deleting `01_qc/data_completeness.csv`:

   ```
   $ python -m connectome_analysis.run_analysis --group cohort --resume
     [skip]  analysis_snapshot
     [skip]  master_cohort
     [skip]  qc_snapshot
     [run]   data_completeness ...
             (recomputing master_cohort for its in-memory result)
     [ok]    data_completeness  (9.6s)
     [run]   demographics ...
     [ok]    demographics  (1.5s)
     [run]   connectome_qc ...
     [ok]    connectome_qc  (0.0s)
     [run]   functional_placeholder ...
     [ok]    functional_placeholder  (0.0s)
   ok=4  skipped=3   total 11.1s
   ```

   Running the same command once more then re-runs `qc_snapshot` and every
   stage after it, because the table they read is now newer than their
   outputs.
3. **A complete run is never "all current".** `live_brain_age` rewrites the
   whole-brain microstructure files in `03_global_microstructure_live/`, so
   right after a complete run `--all --resume` finds `live_multimetric` out of
   date. Running it rebuilds the cohort table and the two microstructure
   stages it reads in memory, and because of point 2 nearly every later stage
   then runs again. In a test with six stages, the `--resume` run took 279
   seconds, longer than the 258 seconds of the first run:

   ```
     [skip]  live_node_microstructure
     [run]   live_multimetric ...
             (recomputing master_cohort for its in-memory result)
             (recomputing live_global_microstructure for its in-memory result)
             (recomputing live_node_microstructure for its in-memory result)
     [ok]    live_multimetric  (160.0s)
     [run]   live_brain_age ...
     [ok]    live_brain_age  (118.5s)
   ok=2  skipped=4   total 278.5s
   ```
4. **`clinical_outcome_search` always runs again**, because it declares no
   output files for the runner to check.

A dry run with `--resume` shows only the stages that are out of date *before*
anything runs. It cannot show the knock-on re-runs from point 2, because it
rebuilds nothing.

### Other ways to choose stages

Whatever you select, the runner sorts it into a safe order; you never have to
list stages in the right order yourself.

Every command below is complete: type it as it stands, from the repository
root.

| You want | Command |
|---|---|
| One stage | `python -m connectome_analysis.run_analysis --stage demographics` |
| Several stages (repeat the flag) | `python -m connectome_analysis.run_analysis --stage lr_sr --stage delay` |
| One stage and everything it needs | `python -m connectome_analysis.run_analysis --stage mediation --with-deps` |
| One group | `python -m connectome_analysis.run_analysis --group measures` (groups: `cohort`, `measures`, `inference`, `ml`, `build`). Unlike `--all`, `--group inference` includes the optional `literature_retrieval`. |
| A stage and every stage listed after it | `python -m connectome_analysis.run_analysis --from edr_exceptions --skip literature_retrieval` |
| Everything except one stage | `python -m connectome_analysis.run_analysis --all --skip clinical_outcome_search` |
| The optional literature lookup | `python -m connectome_analysis.run_analysis --stage literature_retrieval` (needs a contact e-mail, see [Settings](#settings-you-can-change)) |

Add `--dry-run` to any of them to see the plan first without running
anything.

Three details:

- **`--stage`, `--group` and `--from` never add earlier stages by
  themselves.** The results of the stages before the ones you picked must
  already be on disk, or the stages you picked are reported as `[BLOCK]`. Only
  `--group cohort` works on an empty analysis root, because it holds the first
  stages. For example, `--group measures` on an empty analysis root gives
  `blocked=7`.
- `--stage X` alone runs only X. Add `--with-deps` to run what X needs first:

  ```
  $ python -m connectome_analysis.run_analysis --stage mediation --with-deps --dry-run
  plan          : 7 stage(s) [dry-run]

    [dry]   analysis_snapshot  -> 1 output(s)
    [dry]   master_cohort  -> 1 output(s)
    [dry]   live_node_microstructure  -> 2 output(s)
    [dry]   live_node_graph  -> 2 output(s)
    [dry]   network_analysis  -> 4 output(s)
    [dry]   live_brain_age  -> 2 output(s)
    [dry]   mediation  -> 1 output(s)

  dry-run=7   total 0.4s
  ```

- `--from X` takes X and every stage that comes after it in the `--list`
  order (20 stages for `--from edr_exceptions`). That includes everything that
  depends on X, and also some stages that do not (for `--from edr_exceptions`:
  `delay`, `clinical`, `advanced_structural` and `mediation`), and the optional
  `literature_retrieval`. On an empty analysis root every one of the 20
  stages is `[BLOCK]`. Keep `--skip literature_retrieval` unless you have set
  a contact e-mail.

### Trying it without touching existing results

Use this when an analysis folder with results you want to keep already exists,
or when you want a trial run that leaves `git status` unchanged. The matrices
and cohort tables are only read, never changed. Two things normally write
outside the analysis root, and the recipe below handles both:

- **`clinical_outcome_search`** always reads and writes
  `$SC_DERIV_ROOT/qc/analysis_cohort/18_ml_diagnostics/`, whatever
  `--analysis-root` says. The recipe skips it.
- **The `build` stages** save three figures in `docs/figs/`, which git tracks.
  The recipe sends them to `data/figs_try/` instead, through the `figs_dir`
  setting.

Three commands, from the repository root:

```bash
cp configs/analysis.yaml configs/analysis.local.yaml
echo "figs_dir: data/figs_try" >> configs/analysis.local.yaml
python -u -m connectome_analysis.run_analysis --all \
    --analysis-root ~/sc_analysis_try \
    --config configs/analysis.local.yaml \
    --skip clinical_outcome_search 2>&1 | tee analysis_try.log
```

What each one does:

1. Makes your own copy of the settings file. Git ignores every file ending in
   `.local.yaml`, so the copy never shows up in `git status`.
2. Adds the line `figs_dir: data/figs_try` at the very end of the copy, with no
   spaces in front, so it is a top-level setting. `tail -1
   configs/analysis.local.yaml` should print exactly that line. A relative
   path like this one is taken from the repository root, and git ignores
   everything under `data/`. Do not use `~` in this setting: it is not
   expanded here.
3. Runs everything except `clinical_outcome_search`, writing the results to
   the folder `sc_analysis_try` in your home folder (`~`), and keeps a log.
   The header it prints should show `analysis root : .../sc_analysis_try` and
   `config        : configs/analysis.local.yaml`.

The same `build_range_restricted_networks` failure happens here. When you
follow the fix under [If something goes wrong](#if-something-goes-wrong), put
`SC_ANALYSIS_ROOT="$HOME/sc_analysis_try"` in front of the first command, and
add `--analysis-root ~/sc_analysis_try --config configs/analysis.local.yaml` to
the second.

## How long it takes

These times were measured for this README with the 530-subject cohort, on a
64-core Linux server with 123 GB of memory that was running other heavy jobs
at the same time (load average 80 to 110). Your times depend on how many CPU
cores you have and on what else the computer is doing.

| Group | Stages | Time | Slowest stage |
|---|---|---|---|
| `cohort` | 7 | about 10 seconds | `master_cohort` (5 to 13 seconds) |
| `measures` | 7 | about 20 minutes | `network_analysis` (17 minutes) |
| `inference` | 9 (without `literature_retrieval`) | about 9 minutes | `edr_exceptions` (3.3 minutes) |
| `ml` | 2 | about 4.5 hours | `ml_diagnostics` (2 hours 23 minutes) and `clinical_outcome_search` (2 hours 13 minutes) |
| `build` | 11 | about 34 minutes | `build_simple_ml` (14 minutes), `build_shap_overall` (6.5 minutes) |
| everything (`--all`) | 36 | about 5.5 hours | |

In an earlier run the `measures` group took about 26 minutes and
`network_analysis` alone about 22 minutes, so expect some spread.

Good to know:

- The statistics tables (`cohort`, `measures` and `inference` groups) are
  ready after about half an hour. To make only those, run
  `python -m connectome_analysis.run_analysis --group cohort --group measures --group inference --skip literature_retrieval`.
  (`--group inference` includes the optional `literature_retrieval` stage, so
  skip it unless you have set a contact e-mail.)
- The `ml` and `build` groups need the `edr_exceptions` and
  `network_analysis` results, so they always come last.
- If your terminal window might close during those hours (for example a
  remote connection that drops), start the run inside `tmux`, as shown in
  [Step 5](#step-5-run-everything). A run whose terminal closes is stopped,
  like Ctrl-C. Keep the log file from Step 5 so you can see where a stopped
  run got to.

## Inputs

### Connectome matrices

Folder: `$SC_CONNECTOMES_DIR` (default `$SC_DERIV_ROOT/connectomes`, which is
`data/derivatives/connectomes` if nothing is set).

One file per subject per matrix type, named

```
SC_AAL166_<SUBJECT>_I<IMAGEID>_<type>.csv
```

for example `SC_AAL166_XXX_S_0001_I123456_fd_sum.csv`. `<SUBJECT>` is the ADNI
subject ID and `<IMAGEID>` the ADNI image ID of the diffusion scan (digits
only). Two naming rules matter, because `master_cohort` finds each subject's
matrices by looking for `SC_AAL166_<Subject ID>_I*_count.csv`:

- `<SUBJECT>` must be written exactly as in the `Subject ID` column of
  `cohort/dti.csv`. A subject whose files are spelled differently is not
  found and is left out.
- The `_I<IMAGEID>` part must be there. A file named without it, such as
  `SC_AAL166_<SUBJECT>_count.csv`, is not found.

Each file is plain text: 166 lines of 166 numbers separated by commas, with no
header row and no row names. Line *k* of the file, and number *k* on each line,
belong to the region whose `new_id` is *k* (1 to 166) in
`atlas/AAL/aal3_node_map_166.csv`. That file's `name` column gives the
region's name.

| `<type>` | What each cell holds | Used by |
|---|---|---|
| `count` | Streamline count between the two regions | Cohort density check (`master_cohort`), network density, ML features |
| `fd_sum` | Summed fibre density (the main connection weight) | Graph measures, edge-wise tests, EDR, length analyses, networks, ML |
| `len_mean` | Mean streamline length in mm | Length splits, delay, EDR, ML |
| `fa_mean`, `md_mean`, `rd_mean`, `ad_mean` | Mean FA / MD / RD / AxD along the streamlines | Microstructure, coupling, networks, ML |
| `count_invnodevol`, `invlen_mean` | Count scaled by region volume; mean inverse length | Counted in the inventory; not read by the runner's stages |

If a subject is missing one of the microstructure types, that subject is
simply left out of the analyses that need it.

### Cohort tables

Folder: `$SC_COHORT_DIR` (default `cohort/`). Build them with
`python sc_cohort.py build --exports <folder with your ADNI exports>`.

| File | One row per | Columns that must exist |
|---|---|---|
| `dti.csv` | subject (the DTI scan used) | `Subject ID`, `Research Group`, `Image ID`, `Study Date`, `Phase`, `Sex`, `Age` |
| `mri.csv` | subject (the T1 scan) | `Subject ID` |
| `dti_master.csv` | every DTI visit | `Subject ID`, `Study Date`, `MMSE Total Score`, `Global CDR` |
| `mri_master.csv` | every MRI visit | `Subject ID`, `Study Date`, `MMSE Total Score`, `Global CDR` |

`APOE A1`, `APOE A2`, `Weight`, `FAQ Total Score`, `GDSCALE Total Score` and
`NPI-Q Total Score` in `dti.csv` are used when present.

### Who is analysed

The `master_cohort` stage decides which subjects enter every later stage:

- **Diagnostic groups.** EMCI, LMCI and SMC are grouped with MCI. Only CN, MCI
  and AD are analysed; other labels are dropped.
- **Matrix density.** A subject is kept only if its `count` matrix is found and
  at least 60 % of the off-diagonal cells are non-zero (the "dense" cohort).

The result is `00_master/master_cohort.csv`. `00_master/qc_snapshot.csv` gives
the counts per group.

### Files shipped with the repository

| File | Used for |
|---|---|
| `atlas/AAL/aal3_node_map_166.csv` | Region name (`name`) for each matrix index 1–166 (`new_id`). |
| `atlas/AAL/AAL3_network_mapping.csv` | Network and anatomical system for each matrix index. |
| `configs/analysis.yaml` | Parameters (see [Settings](#settings-you-can-change)). |

Always take region names from `aal3_node_map_166.csv`, never from
`atlas/AAL/AAL3_labels.csv`. The second file is keyed on the original atlas
value, which is not the same as the matrix index from row 35 onward (the
atlas has no values 35 and 36, so matrix row 35 is atlas value 37).

### Optional inputs

The run works without these; the affected stage just has less to report.

| File | Read by | If absent |
|---|---|---|
| `configs/exclusions.local.yaml` (or `$SC_EXCLUSIONS`) | `build_exception_specificity`, `build_simple_ml` | No subject is excluded (see [The exclusions file](#the-exclusions-file)). |
| `$SC_QC_ROOT/sc_matrix_qc/sc_matrix_qc_decisions.csv` | `master_cohort` | The QC columns in `master_cohort.csv` say `NOT_RUN`. |
| `$SC_QC_ROOT/metrics_enriched.xlsx` (or `metrics_excel_new.xlsx`, `metrics_excel.xlsx`) | `master_cohort` | Those extra per-subject columns are left out. |
| Imaging intermediates under `$SC_DERIV_ROOT` (`biascorr_1/`, `dwi_t1_bbr/`, `fod/`, `tracks/`, `dti/`, `parc/`) | `data_completeness` | Those columns of `01_qc/data_completeness.csv` count 0. |
| `$SC_QC_ROOT/analysis/analysis4_clinical_revised/` | `clinical` | `14_clinical_sensitivity/clinical_summary_files.csv` is empty. |

(`$SC_QC_ROOT` defaults to `$SC_DERIV_ROOT/qc`.)

## Outputs

Everything goes into the analysis root: `$SC_ANALYSIS_ROOT`, by default
`$SC_DERIV_ROOT/qc/analysis_cohort`. Each stage writes into the numbered
folder its readers (other stages and the dashboard) expect, so there is no copy
step. Tables are CSV files. Most folders also hold PNG plots and a short
`*_inference.md` note explaining the tests used.

Most folders contain a pair of files per measure:
`<measure>_descriptives.csv` (count, mean, SD, median, quartiles, min and max
per group) and `<measure>_pairwise.csv` (CN vs MCI, CN vs AD and MCI vs AD,
with Brunner–Munzel and Mann–Whitney p-values, FDR q-values and Cliff's delta
effect sizes).

| Folder | Stage(s) | What is in it |
|---|---|---|
| `00_master/` | `analysis_snapshot`, `master_cohort`, `qc_snapshot` | `master_cohort.csv` (one row per analysed subject: group, age, sex, clinical scores, which matrix types exist, density); `analysis_snapshot.csv` (provisional or final, and when); `qc_snapshot.csv` (subjects per group); matrix inventories; `run_analysis_status.json` (summary of the last run). |
| `01_qc/` | `data_completeness` | How many subjects per group have each imaging intermediate and a final matrix, overall and per subject. |
| `02_demographics/` | `demographics` | `group_counts.csv` (subjects, mean age, number of women per group); age and clinical-scale (MMSE, CDR, FAQ, GDSCALE, NPI-Q) comparisons; `clinical_summary.csv`. |
| `03_global_microstructure_live/` | `live_global_microstructure` (rewritten by `live_brain_age`) | Whole-brain FA, MD, AxD, RD per subject (mean, median and spread over all connections) and group tests. |
| `04_node_microstructure_live/` | `live_node_microstructure` | The same four measures for each of the 166 regions (`live_node_microstructure_long.csv`), with a test per region. |
| `05_connectome_qc/` | `connectome_qc` | Which matrix types exist per subject and how many per group. |
| `05_multimetric_live/` | `live_multimetric` | The whole-brain and per-region microstructure summaries side by side, and a p-value heat map. |
| `06_global_graph_live/` | `live_global_graph` | Graph measures of each subject's `fd_sum` network: mean and total strength, density, global efficiency, characteristic path length; group tests. |
| `07_node_graph_live/` | `live_node_graph` | Strength, degree and nodal efficiency for each region, with a test per region. |
| `09_coupling_live/` | `live_coupling` | Per subject, how strongly a region's graph measure goes with its microstructure (Spearman correlation across regions), for 12 pairings; group tests. |
| `10_edgewise_fd_sum/` | `edgewise_fd_sum` | A test for every connection: Kruskal–Wallis across the three groups with FDR (`edges_fd_sum_omnibus.csv`), Welch t per pair of groups (`edges_fd_sum_CNvsAD.csv` etc.), and clusters of connections with a permutation p-value (`nbs_*_components.csv`). |
| `11_brain_age_live/` | `live_brain_age` | Brain age predicted from graph and microstructure measures; per subject the brain-age gap (BAG = predicted minus real age) and an age-corrected BAG; model accuracy; group tests. |
| `12_length_delay/` | `lr_sr` | Connections split into short, medium and long by the 33.3 % and 66.7 % points of the CN length distribution (`lr_sr_thresholds.csv`); weights and efficiency per length class, and within versus between hemispheres. |
| `13_delay/` | `delay` | Travel-time estimate for each connection (length divided by 6 mm/ms): mean, median, 90th percentile, delay burden, per subject and per region. |
| `14_clinical_sensitivity/` | `clinical` | An index of older covariate-adjusted clinical tables, if present (see Optional inputs). |
| `15_advanced_structural/` | `advanced_structural` | The EDR model fitted on CN (`edr_model_parameters.csv`); each subject's departure from it for short and long connections; a ranking of connections that change steadily from CN to MCI to AD. |
| `17_edr_exceptions/` | `edr_exceptions` | Length ranges from the CN quartiles (`edr_exception_thresholds.csv`); for every subject, which connections are EDR exceptions (weight above its length bin's mean + 3 SD), at subject, region and connection level; group tests; region rankings. |
| `18_ml_diagnostics/` | `ml_diagnostics`, `clinical_outcome_search` | The feature table used for machine learning, cross-validated models for diagnosis, MMSE and CDR, their predictions, performance and feature importance. `ml_model_status.csv` lists every model tried; a few rows say `unavailable_optional_dependency` (model packages the project does not install), which is expected. The sub-folder `clinical_outcome_model_search/` holds the wider model search. |
| `19_network_analysis/` | `network_analysis`, `functional_placeholder`, `build_range_restricted_networks` | Region-to-network mapping used (`network_mapping_used.csv`). Two sub-folders, one per grouping: `functional/` (seven Yeo networks plus subcortical, cerebellar and brainstem) and `anatomical/` (anatomical systems). Each has per-network microstructure, graph and coupling measures per subject and group tests, network-versus-network comparisons, connectivity within and between networks, and `network_affectedness_summary.csv` ranking the networks by CN-vs-AD effect. `functional/` also gets the short-/long-range and exception network tables from the build stage, and a note (`README_functional.txt`) that these networks are a mapping of structural data, not fMRI. |
| `20_findings/` | `findings_catalog`, `literature_retrieval` | `findings_catalog.csv`: every FDR-significant network result, and every network measure that changes with age, as a plain-English statement with search terms. `literature_records.csv` if the optional lookup ran. |
| `20_exception_specificity/` | all `build_*` stages | The consensus exception core (connections that are exceptions in more than half of the subjects); exception-specific effects; the ML feature matrix and targets; the final model's AUC and PR-AUC, overall and per class; SHAP, partial-dependence and ICE tables; MMSE-band and CDR results. |
| `21_mediation/` | `mediation` | `mediation_results.csv`: each tested chain X → M → Y with the indirect effect, its bootstrap 95 % interval, and a plain-English statement. |
| `exports/` | the runner | `stage_status.csv`, the ledger. |
| `logs/` | the runner | `run_analysis.lock`, which stops two runs writing the same folder at once. |
| `figures/`, `tables/`, `99_appendix/` | the runner | Created empty; kept for older tools. |

There is no `08_` or `16_` folder in a fresh run. An analysis folder made by
older code may also contain older folders, most of them empty, such as
`03_global_rd/`, `08_centrality/` or `16_functional_placeholder/`. The runner
does not use them.

Figures from the `build` stages go to `docs/figs/` (or `figs_dir`, see above).

## Settings you can change

### Where the data is

The runner takes every path from `sc_config.py`, which reads environment
variables (see [Words](#words-used-in-this-readme) for how to set one). Each
variable can also be set for one run with a runner flag; the flag and the
variable do exactly the same thing.

| Variable | Runner flag | What it points to | Default |
|---|---|---|---|
| `SC_DATA_ROOT` | (none) | The data folder | `<repo>/data` |
| `SC_DERIV_ROOT` | `--deriv-root` | Derivatives (matrices and QC live below it) | `$SC_DATA_ROOT/derivatives` |
| `SC_CONNECTOMES_DIR` | `--connectomes-dir` | The connectome matrices | `$SC_DERIV_ROOT/connectomes` |
| `SC_ANALYSIS_ROOT` | `--analysis-root` | Where results are written | `$SC_DERIV_ROOT/qc/analysis_cohort` |
| `SC_COHORT_DIR` | `--cohort-dir` | The four cohort CSV files | `<repo>/cohort` |
| `SC_EXCLUSIONS` | `--exclusions` | The exclusions YAML file | `<repo>/configs/exclusions.local.yaml` |

For example, to use matrices kept in another folder for one run:

```bash
python -m connectome_analysis.run_analysis --all --connectomes-dir "$HOME/adni_connectomes"
```

The runner does not check that these folders exist. A wrong matrix folder
shows up later as an empty cohort (see
[If something goes wrong](#if-something-goes-wrong)). To see what everything
resolves to before a run:

```bash
python -c "import sc_config; print(sc_config.describe())"
```

It prints each path and marks any that does not exist with
`<- does not exist`. It does not show the exclusions file; the `exclusions`
line of `python sc_doctor.py --analysis` does.

### Parameters: `configs/analysis.yaml`

The runner reads `configs/analysis.yaml`, or another file given with
`--config my.yaml`. For each setting it looks first under
`stages: <stage name>:`, then under `defaults:`, then uses the value built into
the code. If the file is missing, the built-in values are used and the header
says `(absent; using defaults)`.

These settings are read by the runner and change what it does:

| Setting | Where in the file | What it does | Value in the file |
|---|---|---|---|
| `snapshot_mode` | `stages: analysis_snapshot:` | Label written to `00_master/analysis_snapshot.csv`: `provisional` for a working run, `final` for the frozen state behind a report. | `provisional` |
| `strict_tracks_count` | `stages: data_completeness:` | `true`: a subject's tractography counts as done only if `tckinfo` (from MRtrix3) reports at least 3,000,000 streamlines. `false`: the file existing is enough. `true` needs `tckinfo` on your `PATH`; without it the setting has no effect and behaves like `false`. | `false` |
| `edge_perms` | `defaults:` | Permutations for the edge-wise cluster test. More is slower, linearly. | `200` |
| `brain_age_repeats` | `defaults:` | Repeats of 5-fold cross-validation in the brain-age model. | `3` |
| `outcome_search_fast` | `stages: clinical_outcome_search:` | `true`: a quicker search, with fewer feature sets, fewer model types, a smaller grid of model settings, fewer trees and 3 instead of 5 outer cross-validation folds. `false`: the full search, which takes much longer. | `true` |
| `outcome_search_jobs` | `stages: clinical_outcome_search:` | Parallel worker processes for that search. | `4` |
| `literature_mailto` | `stages: literature_retrieval:` | Your contact e-mail for OpenAlex. Required by the `literature_retrieval` stage. | `""` (empty) |
| `figs_dir` | top level (not in the shipped file) | Folder for the `build` stages' figures. | `docs/figs` |

The rest of `configs/analysis.yaml` records the values the code uses, with the
reasons for them: the random seeds, `fdr_alpha`, `holm_alpha`, the length
splits for `lr_sr` and `edr_exceptions`, the EDR exception rule
(`exception_sd`, `min_bin_edges`, `weight_type`, `length_type`), the final
model (`n_estimators`, `min_samples_leaf`, `class_weight`, `n_splits`,
`targets`), the bootstrap settings and `top_k`. **The runner does not pass
these to the code.** Changing them in the file changes nothing; they are
there to be read. To change one of them you have to change the code.

The file also explains why the project has **two different length splits**:
`lr_sr` uses tertiles (33.3 % / 66.7 %) of the pooled CN connection lengths,
and `edr_exceptions` uses quartiles (25 % / 75 %) of the same CN lengths
(short ≤ 78.6856 mm, long > 171.021 mm on the published cohort). They answer
different questions, use different column names (`short_*`/`long_*` versus
`sr_*`/`lr_*`) and neither reads the other.

### Other runner flags

| Flag | What it does | Default |
|---|---|---|
| `--config FILE` | Use this parameter file instead of `configs/analysis.yaml`. | `configs/analysis.yaml` |
| `--python PATH` | Python used to start the `build_*.py` scripts. | The Python running the runner |
| `--lock-timeout-sec N` | Seconds to wait if another run holds the lock on the same analysis root. | `5` |

`python -m connectome_analysis.run_analysis --help` lists every flag.

## The exclusions file

The published results leave out one subject, whose mean diffusivity is about
ten times the normal value (the level of cerebrospinal fluid, not brain
tissue). ADNI subject IDs may not be published, so the ID is not in the
repository. It lives in a local file that git ignores:

```yaml
# configs/exclusions.local.yaml
exclude_subjects:
  - XXX_S_0001          # replace with the real ID; one line per subject
```

**How to find the ID.** With your own ADNI data you can find it yourself. It
is the one subject with the highest whole-brain mean diffusivity, the column
`md_mean_edge_mean` in
`03_global_microstructure_live/live_global_microstructure_subject_table.csv`.
That table is made by the `measures` group of stages and does not depend on
the exclusions file, so run at least
`python -m connectome_analysis.run_analysis --group cohort --group measures`
first. Then:

```bash
python -c "import pandas as pd; t = pd.read_csv('data/derivatives/qc/analysis_cohort/03_global_microstructure_live/live_global_microstructure_subject_table.csv'); top = t.nlargest(1, 'md_mean_edge_mean'); print(top[['subject_id', 'group', 'md_mean_edge_mean']].to_string(index=False)); print('times the median:', round(top['md_mean_edge_mean'].iloc[0] / t['md_mean_edge_mean'].median(), 1))"
```

What you should see (with the real ID in place of `XXX_S_0001`):

```
subject_id group  md_mean_edge_mean
XXX_S_0001    AD           0.010289
times the median: 10.8
```

If you changed the analysis root, put your own folder in place of
`data/derivatives/qc/analysis_cohort`. You can also ask the project author for
the ID privately, under your own ADNI Data Use Agreement (see
[configs/README.md](../configs/README.md)). Never post a participant ID in a
public issue or discussion.

Only write the file once you have the real ID. A placeholder such as
`XXX_S_0001` is counted as "1 subject" but leaves nobody out. The exact format
and a step-by-step way to create the file are in
[configs/README.md](../configs/README.md). Use `--exclusions FILE` or
`SC_EXCLUSIONS=FILE` to keep it somewhere else.

**What changes without it.** The file is read by two build scripts,
`build_exception_specificity` and `build_simple_ml`. Without it they keep that
subject. In `20_exception_specificity/` that changes the per-subject exception
architecture tables (`exception_architecture_subject.csv`,
`exception_architecture_stats.csv`) and every table built from the ML feature
matrix (the final model, SHAP, partial dependence, MMSE bands, CDR results):
they are computed on one more subject and will not match the published
numbers. The consensus core, `exception_tier_delta.csv`, the network tables
from the build stage, and everything the other stages write (`00_master/` to
`19_network_analysis/`, `20_findings/`, `21_mediation/`) do not use the file,
so they are the same either way.

The runner tells you at the start which case you are in:

```
exclusions    : 1 subject(s) from <repo>/configs/exclusions.local.yaml
```

or, without the file:

```
exclusions    : NONE (<repo>/configs/exclusions.local.yaml not found)
                The published ML results exclude one subject. Without the
                exclusions file, the ML stages will not reproduce them.
```

## Repairing region names (`relabel_edr_rois.py`)

Old copies of five files in `17_edr_exceptions/` carry wrong region names,
looked up in the wrong table (`AAL3_labels.csv`, keyed on atlas value). That
gets 132 of the 166 names wrong, from row 35 onward: rows 35 and 36 got
placeholders (`AAL_035`, `AAL_036`), and rows 37 to 166 got the name of a
different region. Only the name columns
are wrong; every number, and every network-level result, is correct. The
current `edr_exceptions` stage writes the right names, so you only need this
tool for result folders made before the fix.

Check first (this changes nothing):

```bash
python -m connectome_analysis.relabel_edr_rois
```

On a folder that needs repair you see, per file, how many name cells would
change and two examples:

```
analysis root : <analysis root>
LUT           : <repo>/atlas/AAL/aal3_node_map_166.csv
mode          : report only

  edr_exception_node_level_fd_sum_len_mean.csv         would fix  69960 of 87980 name cells (87980 rows x 1 col)
        AAL_035 -> Cingulate_Mid_L (row 35)
        AAL_036 -> Cingulate_Mid_R (row 36)
  ...
  edr_exception_repeated_top_regions.csv               would fix    147 of 208 name cells (208 rows x 1 col)
        Paracentral_Lobule_L -> Caudate_L (row 73)
        Paracentral_Lobule_L -> Caudate_L (row 73)

would fix 282850 name cells
re-run with --apply to rewrite them
```

On a folder that is already correct every file says `would fix      0`, and a
file that is not there says `(absent)`.

Then repair:

```bash
python -m connectome_analysis.relabel_edr_rois --apply
```

Each changed file is rewritten, and the original is kept next to it as
`<name>.csv.stale_lut.bak` (only the first time, so a second `--apply` never
overwrites the true original). Add `--analysis-root FOLDER` to work on an
analysis folder other than the default. Because the repaired files get a new
date, a later `--resume` treats the stages that read `17_edr_exceptions/` as
out of date.

## Running the model search on its own

`clinical_outcome_search` can also be started directly, for example to try
other options (`--help` lists them):

```bash
python -m connectome_analysis.analysis_clinical_outcome_search --deriv-root data/derivatives --fast --n-jobs 4
```

These are the same settings the runner uses. Replace `data/derivatives` with
your `$SC_DERIV_ROOT` if you set one. Always give `--deriv-root`: its built-in
default is a folder on the machine the project was developed on. It reads and
writes `<deriv root>/qc/analysis_cohort/18_ml_diagnostics/`, so
`ml_diagnostics` must have run there first. At the end it prints
`wrote <deriv root>/qc/analysis_cohort/18_ml_diagnostics/clinical_outcome_model_search`
and two short tables headed `Top MMSE regression` and
`Top Global CDR classification`. It took about 2 hours 13 minutes when
measured for this README.

## If something goes wrong

**`another run holds the lock`** (exit status 75)
Another run is writing to the same analysis root. Wait for it to finish, or
use a different `--analysis-root`. Use `--lock-timeout-sec 600` to wait up to
ten minutes instead of five seconds.

**`[BLOCK] <stage>: N declared input(s) missing`**
The stage needs files that an earlier stage makes, and they are not there. The
lines below it name missing files and the stage that makes each, but **the
list stops after four**, so it can hide other stages that are needed too. For
`mediation` 6 files are missing, and only the four from `network_analysis` are
shown; the two from `live_brain_age` are not. The reliable fix is to run the
same command again with `--with-deps` added, which runs every stage it needs
first:

```
  [BLOCK] mediation: 6 declared input(s) missing
            19_network_analysis/network_mapping_used.csv  (from stage network_analysis)
            19_network_analysis/functional/network_microstructure_stats.csv  (from stage network_analysis)
            19_network_analysis/functional/network_microstructure_subject.csv  (from stage network_analysis)
            19_network_analysis/functional/network_graph_stats.csv  (from stage network_analysis)
```

When an early stage fails, every stage after it that needs its files is
blocked, so look for the first `[FAIL]` in the output, not the last.

The check only asks whether the files exist. If a stage fails but its files
from an earlier run are still on disk, the later stages are not blocked: they
run on those older files. After fixing a failure, re-run the failed stage and
everything after it, for example with `--from <failed stage>`.

**`[FAIL]  master_cohort: FileNotFoundError: ... cohort/dti.csv`**
The cohort tables are missing. Build them with
`python sc_cohort.py build --exports <folder>`, or point `--cohort-dir` at
the folder that holds them, then check with `python sc_cohort.py check`.

**`master_cohort` is `ok`, but then `data_completeness` fails with
`KeyError: 'group'`, `demographics` with `KeyError: 'bm_p'` and
`connectome_qc` with `KeyError: 'subject_id'`**
The master cohort is empty: no subject had a `count` matrix that could be
found. Check `00_master/qc_snapshot.csv`: `cohort_subjects,0` confirms it.
Check that `$SC_CONNECTOMES_DIR` (or `--connectomes-dir`) points at the
matrices and that the file names follow the pattern in [Inputs](#inputs).

**A Python traceback (many lines starting with `File "...", line ...`) whose
last line is `KeyError: 'unknown stage(s): ...'`** (after `--stage`) or
**`KeyError: "unknown stage '...'. Run with --list to see all 37."`** (after
`--from`)
A stage name is misspelt. Only the last line matters. `--list` shows the exact
names.

**`nothing selected. Use --all, --group, --stage or --from`**
You gave no selection. Add one of those flags.

**`argument --group: invalid choice`**
Groups are `cohort`, `measures`, `inference`, `ml`, `build`.

**`config not found: my.yaml`**
The file given to `--config` does not exist. The path is relative to where you
run the command (the repository root).

**`[FAIL]  literature_retrieval: ValueError: literature_retrieval needs a contact address`**
Set `literature_mailto` under `stages: literature_retrieval:` in your config,
or leave that stage out.

**`[FAIL]  build_...: RuntimeError: build_....py exited 1`**
A `build_*.py` script failed. The last lines of its error output are printed
under the `[FAIL]` line and stored in the `error` column of
`exports/stage_status.csv`.

**`[FAIL]  build_range_restricted_networks` with
`ValueError: invalid literal for int() with base 10: '--analysis-root'`**
In the current code this script reads its first command-line word as a
number, but the runner passes `--analysis-root` there, so the stage fails
every time the runner starts it. In a new analysis root the nine `build_*`
stages after it are then `[BLOCK]`. In a folder with results from an earlier
run they are not blocked: they run on the old network tables (see
[Step 5](#step-5-run-everything)). **In both cases, do the two commands
below.** Until the script is fixed, run it by hand without any arguments,
telling it the folders through environment variables, and then let the runner
redo every stage after it:

```bash
SC_ANALYSIS_ROOT=data/derivatives/qc/analysis_cohort python hcp_analysis/build_range_restricted_networks.py
python -m connectome_analysis.run_analysis --from build_exception_specificity
```

The first command must use the same folders as the runner did. For every
location flag you gave the runner, put the matching variable in front of the
first command, and give the second command the same flags (and the same
`--config`, if you used one):

| Runner flag | Variable for the first command |
|---|---|
| `--analysis-root` | `SC_ANALYSIS_ROOT` (it must match the `analysis root` line the runner prints) |
| `--connectomes-dir` | `SC_CONNECTOMES_DIR` |
| `--deriv-root` | `SC_DERIV_ROOT` |
| `--cohort-dir` | `SC_COHORT_DIR` |
| `--exclusions` | not needed by the first command; give it to the second |

For example, after a run with
`--analysis-root ~/sc_analysis_try --connectomes-dir ~/adni_connectomes`:

```bash
SC_ANALYSIS_ROOT="$HOME/sc_analysis_try" SC_CONNECTOMES_DIR="$HOME/adni_connectomes" python hcp_analysis/build_range_restricted_networks.py
python -m connectome_analysis.run_analysis --from build_exception_specificity --analysis-root ~/sc_analysis_try --connectomes-dir ~/adni_connectomes
```

If you set the `SC_*` variables with `export` instead of using flags, they
are already in place: run the first command as just
`python hcp_analysis/build_range_restricted_networks.py`, with nothing in front
(an `SC_ANALYSIS_ROOT=...` written in front would replace your exported
value), and the second without location flags.

The script counts through the subjects (`...50/530 subjects` and so on),
prints three `wrote network_...csv` lines near the end, and takes about 20
seconds. Its last line starts with `exception edges by class:`.
`build_consensus_core` does not depend on it and runs normally.

**`[WARN]  <stage>: wrote 2/4 declared outputs`**
The stage finished but did not write every file it promises. It is recorded
as `incomplete`, and the run exits with status 1. Later stages that need the
missing files will be blocked.

**`MISS  <package>`** from `sc_doctor.py`
A required Python package is not installed in the active environment. Make
sure the venv is switched on (`source .venv/bin/activate`) and re-run
`pip install -r requirements.txt`.

**`--resume` re-runs far more than you expected**
That is how it works; see [Step 7](#step-7-continue-after-a-stop-or-redo-part-of-the-work).
Use `--from <stage>` or `--stage <stage>` to choose exactly what runs.

**The ML numbers differ from the published ones**
Check the `exclusions` line at the top of the run output (see
[The exclusions file](#the-exclusions-file)).

## The older entry point

`apps/connectome_dashboard/refresh_connectome_dashboard_data.py` still works
for tools that call it. It forwards to this runner: `--mode quick` runs the
`cohort` group and `--mode full` runs `--all`. It passes on `--deriv-root` but
has no `--analysis-root` flag; set `SC_ANALYSIS_ROOT` if you need a different
output folder. For new work, use the runner directly.
