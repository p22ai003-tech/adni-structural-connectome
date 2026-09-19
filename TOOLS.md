# Root tools reference

This page explains the small helper programs that sit at the top of the
repository and whose names start with `sc_`. They answer six questions:

| Question | Tool |
|---|---|
| Where does the project look for its files? | `sc_config.py` |
| Is this computer ready to run the project? | `sc_doctor.py` |
| How do I make the cohort tables from my ADNI download? | `sc_cohort.py` |
| Which subjects are left out on purpose? | `sc_exclusions.py` |
| How do I make a zip of the code that is safe to share? | `sc_package.py` |
| How do I point the imaging settings at a new folder? | `sc_localize.py` |

**Do I need this page?**

- **Yes**, if you are setting up the analysis for the first time. You need
  `sc_config.py`, `sc_doctor.py`, `sc_cohort.py` and, optionally,
  `sc_exclusions.py`.
- **Yes**, if you want to share the code. You need `sc_package.py`.
- **Maybe**, if you run the imaging half on a new computer. You need
  `sc_localize.py` once.
- **No**, if you only want to read the results.

To install Python and the packages, see the top-level [README.md](README.md).
To run the analysis itself, see
[connectome_analysis/README.md](connectome_analysis/README.md).

---

## Contents

- [Before you start](#before-you-start)
- [The usual order for a new user](#the-usual-order-for-a-new-user)
- [sc_config.py: where everything lives](#sc_configpy-where-everything-lives)
- [sc_doctor.py: the preflight check](#sc_doctorpy-the-preflight-check)
- [sc_cohort.py: build and check the cohort tables](#sc_cohortpy-build-and-check-the-cohort-tables)
- [sc_exclusions.py: subjects left out on purpose](#sc_exclusionspy-subjects-left-out-on-purpose)
- [sc_package.py: build the shareable zip](#sc_packagepy-build-the-shareable-zip)
- [sc_localize.py: point the imaging settings at this folder](#sc_localizepy-point-the-imaging-settings-at-this-folder)
- [Imaging tools (not covered here)](#imaging-tools-not-covered-here)

---

## Before you start

### Words used on this page

| Word | Meaning |
|---|---|
| repository root | The top folder of this project, the one that holds `README.md` and this file. Written as `<repo>` below. |
| terminal | The window where you type commands. |
| `cd`, `ls` | `cd <folder>` moves the terminal into a folder. `ls` lists what is in the current folder. |
| `$HOME`, `~` | Your home folder. Both mean the same thing in a command. |
| flag (or option) | A word starting with `--` that you add after a command to change what it does, for example `--check` or `--analysis`. |
| venv | Short for "virtual environment": a private Python installation for this project, in the folder `.venv`. |
| `source` | A terminal command that runs a settings file inside your current terminal, so its settings stay in force. `source .venv/bin/activate` turns on the venv; `source env.sh` loads your saved variables. |
| environment variable | A named setting that the terminal passes to every program it starts. Example: `SC_DATA_ROOT`. |
| `$SC_CONNECTOMES_DIR` (and other `$SC_...` names) | On this page, a short way to write "the folder this variable points to", for example "the connectome folder". These variables are usually **not** set in your terminal, so do not type them into commands. To see the real folder, run `python sc_config.py`. |
| `PATH` | A special environment variable: the list of folders the terminal searches when you type a program's name. "On `PATH`" means the terminal can find the program by name alone. |
| exit status | A number every program returns when it ends. `0` means success; anything else means something went wrong. To see it, type `echo $?` right after the command. |
| CSV | "Comma-separated values": a plain-text table. Each line is a row and commas separate the columns. Spreadsheet programs can open it. |
| git | The program that keeps the history of the code. A **commit** is one saved step in that history. A file is **tracked** when git keeps it; `.gitignore` is the file that lists what git must never track. `git clone` makes a full copy of the code together with its history. |
| ADNI | The Alzheimer's Disease Neuroimaging Initiative, the study the data comes from. |
| ADNI phase | ADNI ran in stages called phases: ADNI 1, ADNI GO, ADNI 2, ADNI 3 and ADNI 4. Each row of an ADNI export says which phase it comes from. |
| IDA | ADNI's Image and Data Archive, the website where ADNI data is downloaded. |
| MRI, T1 scan, DTI | MRI is the brain scanner. A T1 scan is an MRI picture of brain anatomy. DTI (diffusion MRI) is a scan that measures how water moves in the brain; the connections between brain regions are worked out from it. |
| cohort tables | Four CSV files that list the subjects, their diagnosis group and their clinical scores. |
| connectome matrix | A 166 x 166 table of numbers. Each number describes the connection between two brain regions of one subject. |
| CN, MCI, AD | The diagnosis groups: cognitively normal, mild cognitive impairment, Alzheimer's disease. |
| MMSE, CDR | Clinical scores. MMSE is the Mini-Mental State Examination (a memory and thinking test). CDR is the Clinical Dementia Rating. |
| APOE | A gene linked to Alzheimer's disease. The ADNI export lists a person's two copies of it as `APOE A1` and `APOE A2`. |
| ML | Machine learning: programs that learn to predict something (here, diagnosis group or clinical scores) from the data. |
| MRtrix3, FSL, ANTs, dcm2niix | Free brain-imaging programs used by the imaging half. They turn scans into connectome matrices. The analysis half does not use them. |
| pipeline, stage | A pipeline is a chain of programs where each one uses the results of the one before. Each link in the chain is a stage. The analysis has 37 stages, numbered 0 to 36 by `python -m connectome_analysis.run_analysis --list`. `--all` runs all but one: `literature_retrieval` (number 23), which is optional and needs an internet connection. |

### What must already be done

1. Your terminal is in the repository root. Use `cd` to go to the folder you
   cloned or unpacked, for example `cd ~/adni-sc-pipeline` (use your folder's
   name). Then type `ls`: you should see `sc_config.py` and `TOOLS.md` in the
   list.
2. The venv exists and the packages are installed. The top-level
   [README.md](README.md) shows how:
   `python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt`.
   If the terminal answers `python3.12: command not found`, Python 3.12 is not
   installed yet. Install it first (see [README.md](README.md)), then run the
   line again.
3. You have turned on the venv:

   ```bash
   source .venv/bin/activate
   ```

   After this, `python` means the project's Python. Every command on this page
   assumes all three of these steps.

### About the data

ADNI data is covered by a Data Use Agreement. You must get it yourself from
[adni.loni.usc.edu](https://adni.loni.usc.edu/), under your own agreement.
None of it is in this repository, and none of it may ever be committed to git.
That includes images, connectome matrices, the cohort tables, acquisition
manifests and the exclusions file. The `.gitignore` file already keeps the usual
places out of git (`data`, `cohort/`, `configs/*.local.yaml`, `dist/`), but you
are still responsible for what you commit.

In examples on this page, a subject is written as `XXX_S_0001` or `<SUBJECT>`.
Real ADNI subject IDs have the same shape with digits in place of the X's.

---

## The usual order for a new user

This is the shortest path from a fresh copy of the code to "ready to run the
analysis". Each step links to its full explanation below.

| Step | Command | Explained in |
|---|---|---|
| 1. See where the project will look for files | `python sc_config.py` | [sc_config.py](#sc_configpy-where-everything-lives) |
| 2. Tell it where your data is (if not in `data/`) | `export SC_DATA_ROOT=$HOME/adni-data` | [Setting a variable](#how-to-set-a-variable) |
| 3. Check the computer | `python sc_doctor.py --analysis`. At this point `MISS` lines for the four cohort tables (and for `data_root`, if that folder does not exist yet) are expected, so it ends with `N problem(s)`. They go away after steps 4 to 6. Look for other `MISS` lines. | [sc_doctor.py](#sc_doctorpy-the-preflight-check) |
| 4. Build the cohort tables | `python sc_cohort.py build --exports $HOME/adni_exports` | [sc_cohort.py](#sc_cohortpy-build-and-check-the-cohort-tables) |
| 5. Check the cohort tables | `python sc_cohort.py check` | [sc_cohort.py](#sc_cohortpy-build-and-check-the-cohort-tables) |
| 6. Put the connectome matrices in place | copy them into the connectome folder (normally `data/derivatives/connectomes`); the commands are just below this table | [What goes in the connectome folder](#what-goes-in-the-connectome-folder) |
| 7. (Optional) Add the exclusions file | create `configs/exclusions.local.yaml`, only if you need the published ML numbers exactly | [sc_exclusions.py](#sc_exclusionspy-subjects-left-out-on-purpose) |
| 8. Check again | `python sc_doctor.py --analysis` should end with `no problems found.` | [sc_doctor.py](#sc_doctorpy-the-preflight-check) |
| 9. Preview the analysis (dry run) | `python -m connectome_analysis.run_analysis --dry-run --all --analysis-root /tmp/sc-preview` (see the note below) | [connectome_analysis/README.md](connectome_analysis/README.md), which also shows the real run |

**About step 6.** The matrices are made by the imaging half (see the Imaging
section of the top-level [README.md](README.md)). Copy them in with these four
lines. Replace `/path/to/your/matrices` with the folder that holds your
`SC_AAL166_*.csv` files:

```bash
CONN=$(python -c "import sc_config; print(sc_config.paths().connectomes_dir)")
echo "$CONN"
mkdir -p "$CONN"
cp /path/to/your/matrices/SC_AAL166_*.csv "$CONN"/
```

The first line asks `sc_config.py` for the connectome folder and keeps the
answer in a short-lived terminal variable called `CONN`. `echo` shows it, so
you can see where the files will go. `mkdir -p` makes the folder if it is not
there yet. `cp` copies the files.

**Neither the doctor nor the dry run looks at the matrices.** Both can report
success with an empty folder. To check that the files landed, run this. It
prints the folder and how many `count` matrices (one per scan) are in it:

```bash
python -c "import sc_config; d = sc_config.paths().connectomes_dir; print(d, len(list(d.glob('SC_AAL166_*_count.csv'))))"
```

On a fresh copy this prints `<repo>/data/derivatives/connectomes 0`. After the
copy, the number at the end should be the number of scans you have matrices for
(it was `530` for the published analysis).

**About step 9.** A dry run runs no analysis, but it is not read-only: it
creates the analysis folder and writes three small bookkeeping files there
(`exports/stage_status.csv`, `00_master/run_analysis_status.json` and
`logs/run_analysis.lock`). Without `--analysis-root`, that folder is normally
inside your data folder, and the status files of your last real run there are
overwritten.
`--analysis-root /tmp/sc-preview` sends those files to a throw-away folder
instead. Do step 8 first: the dry run does not check the cohort tables or the
matrices, so it can say `dry-run=36` even when nothing is in place. Its last
lines look like this:

```text
  [dry]   build_shap_overall  -> 6 output(s)

dry-run=36   total 0.5s
ledger: /tmp/sc-preview/exports/stage_status.csv
```

---

## sc_config.py: where everything lives

### What is it?

`sc_config.py` decides where the folders and files of the project are. The
other tools and the analysis ask it, instead of having folder names written
into them.

It starts from a few "root" folders and builds all other locations from them.
You can move any location by setting one environment variable. If you set
nothing, everything is found inside the repository.

It never creates folders and never reads your data. It only works out names.

### Do I need it?

You need to **run** it once, to see where the project will look. You only need
to **change** anything if your data is not in `<repo>/data`.

### How to run it

```bash
python sc_config.py
```

**What you should see** on a fresh copy of the code, before any data is in
place (`<repo>` stands for your repository folder):

```text
Resolved configuration
============================================================
  project_root         <repo>
  data_root            <repo>/data  <- does not exist
  raw_images_root      <repo>/data/Images  <- does not exist
  raw_dwi_root         <repo>/data/Images/dti  <- does not exist
  raw_t1_root          <repo>/data/Images/mri  <- does not exist
  deriv_root           <repo>/data/derivatives  <- does not exist
  qc_root              <repo>/data/derivatives/qc  <- does not exist
  analysis_root        <repo>/data/derivatives/qc/analysis_cohort  <- does not exist
  connectomes_dir      <repo>/data/derivatives/connectomes  <- does not exist
  cohort_dir           <repo>/cohort  <- does not exist
  atlas_root           <repo>/atlas
  aal_root             <repo>/atlas/AAL
  run_state_dir        <repo>/data/derivatives/qc/run_state  <- does not exist
  manifest             <repo>/configs/acquisition_manifest.csv  <- does not exist

Toolchains
------------------------------------------------------------
  FSLDIR               (unset; will resolve via PATH)
  MRTRIX_BIN           (unset; will resolve via PATH)
  ANTSPATH             (unset; will resolve via PATH)
```

`<- does not exist` is not an error here. It only tells you that the folder is
not there yet. `sc_doctor.py` (next section) tells you which missing folders
matter.

The same report is available from Python:
`python -c "import sc_config; print(sc_config.describe())"`.

### Every variable

Each row is one environment variable. If you do not set it, the default in the
third column is used. Defaults build on each other: change `SC_DATA_ROOT` and
everything under it moves too.

On this page, `$SC_CONNECTOMES_DIR` means "the connectome folder", and the same
for the other names. The variable is usually not set in your terminal, so do
not type it into commands. Run `python sc_config.py` to see the real folder.

| Variable | What it points to | Default | Who sets it |
|---|---|---|---|
| `SC_PROJECT_ROOT` | The repository | the folder that holds `sc_config.py` | almost never |
| `SC_DATA_ROOT` | The data folder | `$SC_PROJECT_ROOT/data` | **you, if your data is elsewhere** |
| `SC_RAW_IMAGES_ROOT` | Raw ADNI images (imaging half only) | `$SC_DATA_ROOT/Images` | imaging users |
| `SC_DERIV_ROOT` | Derived data ("derivatives": everything computed from the images) | `$SC_DATA_ROOT/derivatives` | sometimes |
| `SC_QC_ROOT` | Quality-control folder | `$SC_DERIV_ROOT/qc` | rarely |
| `SC_ANALYSIS_ROOT` | Where analysis results are written | `$SC_QC_ROOT/analysis_cohort` | sometimes (for a test run) |
| `SC_CONNECTOMES_DIR` | The connectome matrices (the analysis input) | `$SC_DERIV_ROOT/connectomes` | sometimes |
| `SC_COHORT_DIR` | The four cohort tables | `$SC_PROJECT_ROOT/cohort` | sometimes |
| `SC_ATLAS_ROOT` | Atlas files (the brain-region map; shipped with the code) | `$SC_PROJECT_ROOT/atlas` | almost never |
| `SC_RUN_STATE_DIR` | Small bookkeeping files of long runs | `$SC_QC_ROOT/run_state` | almost never |
| `SC_MANIFEST` | Acquisition manifest (imaging half only) | `$SC_PROJECT_ROOT/configs/acquisition_manifest.csv` | imaging users |
| `SC_EXCLUSIONS` | The subject exclusions file (read by `sc_exclusions.py`, not `sc_config.py`) | `$SC_PROJECT_ROOT/configs/exclusions.local.yaml` | sometimes |

The ones most people touch are `SC_DATA_ROOT`, `SC_DERIV_ROOT`,
`SC_CONNECTOMES_DIR`, `SC_ANALYSIS_ROOT`, `SC_COHORT_DIR` and `SC_EXCLUSIONS`.

Some locations are fixed sub-folders and have no variable of their own:

| Name in the report | Always equals |
|---|---|
| `raw_dwi_root` | `$SC_RAW_IMAGES_ROOT/dti` |
| `raw_t1_root` | `$SC_RAW_IMAGES_ROOT/mri` |
| `aal_root` | `$SC_ATLAS_ROOT/AAL` |

Programs also use a few fixed file names inside those folders, for example
`$SC_COHORT_DIR/dti.csv`, `$SC_ATLAS_ROOT/AAL/aal3_node_map_166.csv` (the
166-region look-up table) and `$SC_ANALYSIS_ROOT/00_master/master_cohort.csv`.

### What goes in the connectome folder

The analysis reads one CSV file per subject, scan and measure from
`$SC_CONNECTOMES_DIR`:

```text
SC_AAL166_<SUBJECT>_I<IMAGEID>_<type>.csv
```

Each file is 166 rows by 166 columns of numbers, separated by commas, with no
header row. `<type>` is one of: `count`, `fd_sum`, `len_mean`, `fa_mean`,
`md_mean`, `rd_mean`, `ad_mean`, `count_invnodevol`, `invlen_mean`. How these
files are made belongs to the imaging half: see the Imaging section of the
top-level [README.md](README.md). The commands to copy them into place, and to
count them, are under [the usual order](#the-usual-order-for-a-new-user)
(step 6).

### The three tool variables (imaging half only)

| Variable | What it is | If unset |
|---|---|---|
| `FSLDIR` | The FSL installation folder (the one holding `bin/` and `data/standard/`) | FSL programs are looked up on `PATH` |
| `MRTRIX_BIN` | The MRtrix3 `bin` folder | the folder of `tckgen` on `PATH`, if there is one |
| `ANTSPATH` | The ANTs `bin` folder | ANTs programs are looked up on `PATH` |

The analysis half does not use them.

### How to set a variable

Pick one of three ways.

**For one command only.** Put the setting in front of the command:

```bash
SC_DATA_ROOT=$HOME/adni-data python sc_config.py
```

**For the rest of this terminal session.** Use `export`:

```bash
export SC_DATA_ROOT=$HOME/adni-data
python sc_config.py
```

**Every time, without retyping.** Copy the template, edit it, and load it at the
start of each session:

```bash
cp env.sh.example env.sh      # env.sh is ignored by git, so it stays on your computer
# edit env.sh with any text editor, then:
source env.sh
```

`env.sh.example` sets `SC_PROJECT_ROOT`, `SC_DATA_ROOT` and the three tool
variables, and has commented-out lines for `SC_DERIV_ROOT` and
`SC_RAW_IMAGES_ROOT`.

What to edit in `env.sh`:

- Change the line `export SC_DATA_ROOT="${SC_PROJECT_ROOT}/data"` so it names
  your data folder, for example `export SC_DATA_ROOT="$HOME/adni-data"`.
  Leave `SC_PROJECT_ROOT` as it is.
- Analysis-only users can leave the three tool lines (`MRTRIX_BIN`, `FSLDIR`,
  `ANTSPATH`) as they are.

Careful: `source env.sh` always sets `SC_DATA_ROOT` to whatever that line says.
If you typed `export SC_DATA_ROOT=...` earlier in the same terminal and did not
edit `env.sh`, sourcing it quietly puts `SC_DATA_ROOT` back to `<repo>/data`.

Rules that avoid surprises:

- **Use full paths** (starting with `/` or `$HOME`). A relative path such as
  `my-data` is taken relative to whatever folder you are in when you run the
  command. `sc_config.py` does not convert it for you.
- `~` works (`~/adni-data` becomes your home folder).
- Set variables **before** you start a program. A running program does not see
  later changes.

The analysis runner has flags that do the same job for one run. Each flag sets
the matching variable:

| Flag of `python -m connectome_analysis.run_analysis` | Sets |
|---|---|
| `--deriv-root` | `SC_DERIV_ROOT` |
| `--connectomes-dir` | `SC_CONNECTOMES_DIR` |
| `--analysis-root` | `SC_ANALYSIS_ROOT` |
| `--cohort-dir` | `SC_COHORT_DIR` |
| `--exclusions` | `SC_EXCLUSIONS` |

Unlike the variables, these flags turn relative paths into full paths.

### For programmers

```python
from sc_config import paths, tools

p = paths()
p.connectomes_dir            # a pathlib.Path
p.cohort_dti_csv             # $SC_COHORT_DIR/dti.csv
tools().mrtrix("tckgen")     # full path to tckgen, or FileNotFoundError
```

`paths()` is worked out once per Python process and then remembered. If your
code changes `os.environ` after that, call `sc_config.paths.cache_clear()`.

### If something goes wrong

| What you see | Why | Fix |
|---|---|---|
| `<- does not exist` next to `data_root` | Your data is somewhere else, or the folder was never made | Set `SC_DATA_ROOT` to the folder that holds `derivatives/`, then run `python sc_config.py` again |
| A path in the report that starts with a word, not `/` (for example `my-data`) | You set a relative path | Use a full path: `export SC_DATA_ROOT=$HOME/my-data` |
| `FileNotFoundError: 'tckgen' not found. Set FSLDIR / MRTRIX_BIN / ANTSPATH, or put it on PATH.` (from Python code) | An imaging tool is not installed or not exported | Set the tool variable (see `env.sh.example`) and `source env.sh`. Analysis-only users can ignore this. |

---

## sc_doctor.py: the preflight check

### What is it?

A "preflight check" is a list of checks made before a long job starts, like a
pilot's check before take-off. `sc_doctor.py` checks that the folders, Python
packages, cohort tables and (for the imaging half) the brain-imaging programs
are all in place. It changes nothing.

### Do I need it?

Yes. Run it before your first analysis run, and again whenever something fails
in a strange way.

### How to run it

Pick the mode that matches what you want to run:

| Command | Checks | Use it when |
|---|---|---|
| `python sc_doctor.py --analysis` | Paths, Python packages, cohort tables, exclusions | you run the analysis (most people) |
| `python sc_doctor.py --imaging` | Paths, the 11 imaging programs, the acquisition manifest | you run the imaging half |
| `python sc_doctor.py` | Everything above | you run both halves |

Analysis-only users should use `--analysis`. Without a flag, missing imaging
programs count as problems, so the check fails even though the analysis would
run.

**How to read the output.** Each line starts with one of three marks:

| Mark | Meaning | Counts as a problem? |
|---|---|---|
| `ok` | Found and usable | no |
| `warn` | Missing, but not needed right now (or optional) | no |
| `MISS` | Missing and needed | yes, except the `python` version line (see [Python packages](#what-each-check-means-and-how-to-fix-it) below) |

The last line is either `no problems found.` or `N problem(s). The pipeline will
not complete until these are fixed.` The exit status (a number every program
returns, which scripts can test; type `echo $?` right after the command to see
it) is `0` when nothing is missing and `1` otherwise.

**What you should see on a fresh copy of the code** (paths shortened). The very
first two lines may be a `DeprecationWarning: Accessing jsonschema.__version__
is deprecated ...` message. That is harmless; it is left out below.

```text
$ python sc_doctor.py --analysis
========================================================================
pipeline preflight
========================================================================

Paths
------------------------------------------------------------------------
  ok    project_root               <repo>
  MISS  data_root                  <repo>/data   <- set SC_DATA_ROOT
  warn  raw_images_root            <repo>/data/Images   (absent; created on demand)
  ...
  warn  connectomes_dir            <repo>/data/derivatives/connectomes   (absent; created on demand)
  warn  cohort_dir                 <repo>/cohort   (absent; created on demand)
  ok    atlas_root                 <repo>/atlas
  ok    aal_root                   <repo>/atlas/AAL
  ...

Python packages
------------------------------------------------------------------------
  interpreter: <repo>/.venv/bin/python  (Python 3.12.11)
  ok    numpy                      2.1.3
  ok    pandas                     2.3.2
  ...
  ok    pydicom                    3.0.1

Cohort tables
------------------------------------------------------------------------
  MISS  dti.csv                    <repo>/cohort/dti.csv   <- build with: python sc_cohort.py build --exports <folder>
  MISS  mri.csv                    <repo>/cohort/mri.csv   <- build with: python sc_cohort.py build --exports <folder>
  MISS  dti_master.csv             <repo>/cohort/dti_master.csv   <- build with: python sc_cohort.py build --exports <folder>
  MISS  mri_master.csv             <repo>/cohort/mri_master.csv   <- build with: python sc_cohort.py build --exports <folder>
  warn  exclusions                 <repo>/configs/exclusions.local.yaml absent; the ML stages will not reproduce the published numbers

========================================================================
5 problem(s). The pipeline will not complete until these are fixed.
```

**What you should see when the analysis side is ready** (after setting
`SC_DATA_ROOT`, building the cohort tables and adding the exclusions file):

```text
Paths
  ok    project_root               <repo>
  ok    data_root                  <your data folder>
  ...
  ok    connectomes_dir            <your data folder>/derivatives/connectomes
  ok    cohort_dir                 <repo>/cohort
...
Cohort tables
  ok    dti.csv                    <repo>/cohort/dti.csv
  ok    mri.csv                    <repo>/cohort/mri.csv
  ok    dti_master.csv             <repo>/cohort/dti_master.csv
  ok    mri_master.csv             <repo>/cohort/mri_master.csv
  ok    exclusions                 1 subject(s) from <repo>/configs/exclusions.local.yaml

========================================================================
no problems found.
```

### What each check means and how to fix it

**Paths.** Every location from `sc_config.py`. Only two must exist:
`project_root` and `data_root`. The rest show `warn ... (absent; created on
demand)` when missing, because programs make them when they write output.

| Line | Fix |
|---|---|
| `MISS data_root ... <- set SC_DATA_ROOT` | Make the folder, or point `SC_DATA_ROOT` at the folder that holds your `derivatives/` folder |
| `warn connectomes_dir` | Not counted as a problem, but the analysis needs matrices here. Copy them in as shown under [the usual order](#the-usual-order-for-a-new-user) (step 6), or set `SC_CONNECTOMES_DIR` to where they already are. The doctor does not look inside the matrix files. |
| `warn analysis_root`, `warn qc_root`, `warn run_state_dir` | Nothing to do. They are made on the first run. |
| `warn manifest`, `warn raw_images_root` | Nothing to do for the analysis. They belong to the imaging half. |

**Python packages** (`--analysis`). The first line shows which Python is being
checked. It should be the one inside `.venv`. Then each package:

- Required: numpy, pandas, scipy, sklearn (scikit-learn), matplotlib, seaborn,
  networkx, statsmodels, nibabel, shap, yaml (PyYAML). A missing one is `MISS`
  with the hint `requirements/analysis.txt`.
- Shown as `optional` (a `warn`) if missing: xgboost, tabpfn, pytorch_tabnet,
  jsonschema, pytest, pydicom. Note: `requirements.txt` installs xgboost,
  jsonschema, pytest and pydicom, and xgboost is needed to reproduce the
  published XGBoost results in the ML diagnostics. tabpfn and pytorch_tabnet
  (from `requirements/extras-ml.txt`) are not used by any stage of
  `run_analysis`. A `warn` for them can be ignored.
- Python itself must be version 3.12 or newer. An older one prints
  `MISS  python   3.12 required, found 3.9.21` (with your version number) near
  the top of this section. This line is **not counted** in the final total, so
  check it yourself even when the last line says `no problems found.`

Fix for a missing package: turn on the venv (`source .venv/bin/activate`), then
`pip install -r requirements.txt`. Fix for an old Python: make the venv again
with Python 3.12 (see the top-level [README.md](README.md) if `python3.12` is
not installed). `--clear` empties the old venv first, so nothing from the old
Python is left behind:

```bash
python3.12 -m venv --clear .venv && .venv/bin/pip install -r requirements.txt
source .venv/bin/activate
```

**Cohort tables** (`--analysis`). Checks that the four files exist in
`$SC_COHORT_DIR` and have the columns the analysis reads. Fix: build them with
`sc_cohort.py` ([next section](#sc_cohortpy-build-and-check-the-cohort-tables)),
or set `SC_COHORT_DIR` to where they are. The last line reports the exclusions
file (see [sc_exclusions.py](#sc_exclusionspy-subjects-left-out-on-purpose)). A
missing exclusions file is a `warn`, not a problem.

**Imaging toolchain** (`--imaging`). Looks for 11 programs from MRtrix3
(`mrconvert`, `dwi2fod`, `tckgen`, `tck2connectome`, `dwifslpreproc`), FSL
(`eddy_cpu`, `bet`, `flirt`), ANTs (`antsRegistration`,
`N4BiasFieldCorrection`) and `dcm2niix`, and prints each one's version. A
missing one looks like this:

```text
  MISS  tckgen                     MRtrix3: streamline generation  (set MRTRIX_BIN, or put it on PATH)
  ...
  MRtrix, FSL and ANTs are frequently installed but not exported.
  Copy env.sh.example to env.sh, edit the three paths, then `source env.sh`.
```

Fix: do what the message says. After `source env.sh` on a ready computer:

```text
Imaging toolchain
------------------------------------------------------------------------
  ok    mrconvert                  == mrconvert 3.0.7 ==
  ...
  ok    antsRegistration           ANTs Version: 2.6.5.dev1-g89fa5be
  ok    dcm2niix                   Chris Rorden's dcm2niiX version v1.0.20260416  (JP2:OpenJP
```

For some FSL programs the last column shows a line of the program's help text
instead of a version, for example `Main bet2 options:` for `bet` or
`Copyright(c) 2015, University of Oxford ...` for `eddy_cpu`. That is fine: the
`ok` at the start of the line is what counts.

`dcm2niix` is installed into the venv by `requirements.txt`, so it is usually
`ok` even without `env.sh`.

**Input contract** (`--imaging`). Checks the acquisition manifest (the imaging
half's list of scans). `warn manifest ... (absent; build it with discovery)` is
normal until you run the imaging discovery step. See the Imaging section of the
top-level [README.md](README.md).

### If something goes wrong

| What you see | Why | Fix |
|---|---|---|
| `DeprecationWarning: Accessing jsonschema.__version__ is deprecated ...` at the top | A newer jsonschema prints this when asked its version | Harmless. Ignore it. |
| `interpreter:` shows a Python outside `.venv` | The venv is not turned on | `source .venv/bin/activate`, then run again |
| Many packages `MISS` and `MISS dti.csv ... unreadable: No module named 'pandas'` | Wrong Python (packages not installed there) | Same as above. The cohort lines turn `ok` once pandas is found. |
| `N problem(s)` with only imaging programs `MISS`, and you only run the analysis | You ran it without a flag | Run `python sc_doctor.py --analysis` |
| The doctor stops right after the `Cohort tables` lines with `could not read <repo>/configs/exclusions.local.yaml: ...` and prints no final line (exit status 1) | The exclusions file is badly formed | Fix the file as described under [sc_exclusions.py](#sc_exclusionspy-subjects-left-out-on-purpose) |
| No `exclusions` line at all under `Cohort tables` | The exclusions file exists but is not in the expected shape, for example the `:` after `exclude_subjects` is missing | Run the check command under [sc_exclusions.py](#how-to-check-it) to see the error, then fix the file |

---

## sc_cohort.py: build and check the cohort tables

### What is it?

The analysis needs four CSV files that describe the subjects: their diagnosis
group, age, sex, clinical scores and which scan to use. Together they are called
the cohort tables. `sc_cohort.py` makes them from files you download from ADNI
(`build`), and checks that they are complete (`check`).

### Do I need it?

Yes, once, if you run the analysis and do not already have the four tables.
`check` is useful any time.

### Step A: download three files from ADNI

> Menu names on the ADNI website can change. If one differs from what is written
> here, pick the closest match.
>
> `python sc_cohort.py --help` gives a short summary of these downloads. Where it
> differs from this page, follow this page.

You need three separate exports. They are **not** the same search, and one
cannot stand in for another.

1. Sign in at [adni.loni.usc.edu](https://adni.loni.usc.edu/) with your own
   approved ADNI account and open the Image and Data Archive (IDA).
2. Open **Search**, then **Advanced Image Search**. Choose project **ADNI**.
3. Run the three searches below, one at a time. For each one, tick the result
   columns listed for that file after these steps (in the IDA each search field
   has a box that adds it to the results), then export the result as CSV:

   | Save the export as | Search settings | What the published file held (for comparison) |
   |---|---|---|
   | `dti_master.csv` | Modality = **DTI**, all phases | 8,481 rows, 1,114 subjects. Phases ADNI GO, ADNI 2, ADNI 3 and ADNI 4. Only "Axial DTI" scans and images made from them (16 different descriptions). |
   | `mri_master.csv` | Modality = **MRI**, phases **ADNI 1, ADNI GO and ADNI 2** only | 27,396 rows, 1,232 subjects. Only MPRAGE T1 scans and images made from them: every description is `MPRAGE` (or `mprage`) itself, or ends in `<- MPRAGE` (or `<- mprage`), 46 different descriptions. |
   | `all_mri.csv` (optional) | Modality = **MRI**, **all** phases, image type and description left open, so that every MRI image is listed | 204,459 rows, 3,240 subjects. All five phases, 1,954 different descriptions. These are the T1 scan candidates. |

4. Put the three files in one folder **outside the repository**, for example
   `$HOME/adni_exports/`. Use exactly these file names.
5. Do not edit, re-sort or re-save the files in a spreadsheet program. The build
   keeps the **last** row per subject in the order the file has, and it expects
   `Study Date` written month/day/year (for example `3/14/2012`), as the IDA
   exports it.

Three things to know about these searches:

- **Your `mri_master.csv` will have more rows than 27,396. That is expected.**
  With only the settings in the "Search settings" column, the search lists every
  MRI image of those three phases, not only the MPRAGE ones (in the published
  `all_mri.csv` that is 161,786 rows). The build copies the file as it is, and
  the whole analysis runs with it. Only the ML stages read it (see the last
  point below). To get closer to the published file, also limit the search to
  image descriptions containing `MPRAGE`. That still keeps some related scans,
  such as `MPRAGE Repeat` and `MPRAGE GRAPPA2` (55,026 rows in the published
  `all_mri.csv`).
- **`all_mri.csv` is its own, much larger export.** `mri_master.csv` cannot
  replace it. `mri.csv` picks each subject's T1 scan from `all_mri.csv`. With
  the published files, that gives a T1 scan for all 1,114 DTI subjects. With
  `mri_master.csv` put in its place, only 234 subjects get one, and 154 of those
  get a different scan.
- **`mri_master.csv` affects the ML stages.** Two stages, `ml_diagnostics` and
  `build_simple_ml` (and the stages built on its results), read clinical scores
  (MMSE, CDR) from `dti.csv`, `dti_master.csv` and `mri_master.csv`. For each
  subject they use the score whose date is nearest to the DTI scan. Extra rows
  in `mri_master.csv` can add nearer-dated scores and so change which score is
  used. Use the settings above and compare your row and subject counts with the
  right-hand column.

**Result columns to tick** for `dti_master.csv` and `mri_master.csv`. The
column names in the file must match these exactly.

| Column | How it is used |
|---|---|
| `Subject ID` | required in every table |
| `Research Group` | required: the diagnosis group |
| `Image ID` | required: which DTI scan the analysis uses |
| `Study Date` | required |
| `Phase`, `Sex`, `Age` | required |
| `MMSE Total Score`, `Global CDR` | required in `dti_master.csv` and `mri_master.csv` (clinical scores) |
| `Visit`, `Description`, `Type` | tick these too |
| `APOE A1`, `APOE A2`, `Weight`, `FAQ Total Score`, `GDSCALE Total Score`, `NPI-Q Total Score` | used when present; tick them to reproduce the published tables |

**Result columns to tick** for `all_mri.csv`: `Subject ID` and `Description`
are required (the build chooses the T1 scan with them). `Image ID`, `Type` and
`Phase` are copied into the analysis's master table when present, so tick them
too.

The published exports had these columns, in this order (taken from their
header rows):

| File | Columns |
|---|---|
| `dti_master.csv` (26) | `Subject ID, Phase, Sex, Weight, Research Group, APOE A1, APOE A2, Visit, Study Date, Archive Date, Age, Global CDR, NPI-Q Total Score, MMSE Total Score, GDSCALE Total Score, FAQ Total Score, Modality, Description, Type, Imaging Protocol, Image ID, Structure, Laterality, Image Type, Registration, Tissue` |
| `mri_master.csv` (25) | `Subject ID, Phase, Sex, Weight, Research Group, APOE A1, APOE A2, Visit, Archive Date, Study Date, Age, MMSE Total Score, GDSCALE Total Score, Global CDR, FAQ Total Score, NPI-Q Total Score, Modality, Description, Type, Image ID, Structure, Tissue, Laterality, Image Type, Registration` |
| `all_mri.csv` (9) | `Subject ID, Project, Phase, Sex, Age, Modality, Description, Type, Image ID` |

The build and the analysis find columns by name, so a different column order
or extra columns do no harm.

### Step B: build the tables

```bash
python sc_cohort.py build --exports $HOME/adni_exports
```

By default the tables are written to `$SC_COHORT_DIR` (normally `<repo>/cohort`,
which git ignores). Add `--out <folder>` to write somewhere else.

> **Careful:** `build` overwrites `dti.csv`, `mri.csv`, `dti_master.csv` and
> `mri_master.csv` in the output folder without asking. If you already have
> tables there, build into a new folder first with `--out`.

**If you used `--out`** (say `--out $HOME/new-cohort`), the last line names
that folder, written out in full:

```text
next:  python sc_cohort.py check   (with SC_COHORT_DIR=<your folder>)
```

Nothing else finds that folder on its own. Before you run `check`, the doctor
or the analysis, either set `export SC_COHORT_DIR=$HOME/new-cohort` (see
[How to set a variable](#how-to-set-a-variable)), or pass
`--cohort-dir $HOME/new-cohort` to the analysis runner, or move the four files
into `<repo>/cohort`.

**What you should see** (this example used a tiny made-up export with 3
subjects; with real ADNI exports the numbers are in the hundreds or thousands):

```text
dti.csv              3 subjects  (from 4 DTI rows)
mri.csv              2 subjects  (ranked T1 choice)
                1 DTI subject(s) have no acceptable T1 description
dti_master.csv  copied
mri_master.csv  copied

wrote <repo>/cohort
next:  python sc_cohort.py check
```

The line `N DTI subject(s) have no acceptable T1 description` is information,
not an error. Without `all_mri.csv` you see instead:
`mri.csv         header only (no all_mri.csv given; the analysis reads only Subject ID)`.

### Step C: check the tables

```bash
python sc_cohort.py check
```

This checks `$SC_COHORT_DIR`. To check another folder:
`python sc_cohort.py check --dir <folder>`.

**What you should see:**

```text
cohort folder: <repo>/cohort

  ok       dti.csv            1114 rows
  ok       mri.csv            1114 rows
  ok       dti_master.csv     8481 rows
  ok       mri_master.csv    27396 rows

no problems found.
```

(Those are the row counts of the published tables. Yours depend on your
export.) If some optional columns were not exported, you also see a line such
as `optional columns absent: APOE A1, APOE A2, Weight, ...`. That is not a
problem, but those values are then missing from the analysis.

Exit status: `0` when there are no problems, `1` otherwise.

### Inputs

| File | Where | Needed |
|---|---|---|
| `dti_master.csv` | the folder given to `--exports` | yes |
| `mri_master.csv` | the folder given to `--exports` | yes |
| `all_mri.csv` | the folder given to `--exports` | no |
| `configs/cohort_rules/Ranked_OK_T1.csv` | in the repository | yes, shipped with the code |

### Outputs

All in `$SC_COHORT_DIR` (or `--out`). All are ADNI-restricted: never commit or
share them.

| File | Contents |
|---|---|
| `dti.csv` | One row per subject: the DTI scan the analysis uses. The first column is an unnamed row number; it is part of the published format. `Study Date` is rewritten as year-month-day. |
| `mri.csv` | One row per subject: the chosen T1 scan, with its `rank`. Header only if there was no `all_mri.csv`. |
| `dti_master.csv` | A copy of your export. The ML stages read clinical scores from it. |
| `mri_master.csv` | A copy of your export, for the same reason. |

### How the rows are chosen

- **`dti.csv`**: for each `Subject ID`, the last row in `dti_master.csv`, in
  file order.
- **`mri.csv`**: rows of `all_mri.csv` whose subject is in `dti.csv` and whose
  `Description` appears in `configs/cohort_rules/Ranked_OK_T1.csv`. For each
  subject the lowest rank wins (1 = best). A tie goes to the row that comes
  first in the file. The ranking is a hand-made list of acceptable T1
  descriptions, based on the patterns in
  `configs/cohort_rules/adni_t1_coreg_whitelist.csv`.
- The analysis does not use `mri.csv` to choose subjects. It only copies the
  chosen T1 scan's details (`Image ID`, `Description`, `Type`, `Phase`) into its
  master table when they are there. That is why a header-only `mri.csv` still
  works.

**Grouping rule.** The `Research Group` column gives each subject's diagnosis.
In this project EMCI, LMCI and SMC are grouped with MCI, and only CN, MCI and AD
are analysed. `sc_cohort.py` keeps every row; the grouping is applied later, by
the analysis (`recode_group` in `connectome_analysis/analysis_cohort.py`).

### Settings you can change

| Setting | Where | Default |
|---|---|---|
| Folder holding the exports | `build --exports <folder>` | none; required |
| Output folder | `build --out <folder>` | `$SC_COHORT_DIR` |
| Folder to check | `check --dir <folder>` | `$SC_COHORT_DIR` |
| Acceptable T1 descriptions and their rank | `configs/cohort_rules/Ranked_OK_T1.csv` (columns `description,rank`) | shipped list. Editing it changes `mri.csv`. |

### If something goes wrong

| What you see | Why | Fix |
|---|---|---|
| `missing: .../dti_master.csv` then `(see python sc_cohort.py --help for what to download)` (exit status 2) | A required export is not in the `--exports` folder, or has another name | Check the folder and the exact file names |
| `KeyError: 'Study Date'` (or another column name) at the end of a long error | The export's column names differ, for example `Acq Date` or `Subject` from a different IDA search page | Export again from Advanced Image Search with the columns listed above |
| `ValueError: time data "2012-03-14" doesn't match format "%m/%d/%Y"` | `Study Date` is not month/day/year, usually because the file was re-saved in a spreadsheet program | Download the export again and use it unchanged |
| `MISSING  mri_master.csv` from `check` | The file is not in the checked folder | Run `build`, or point `--dir` / `SC_COHORT_DIR` at the right folder |
| `BAD      dti.csv    4 rows; missing required column(s): Age` | A required column is missing | Export again with that column ticked, then `build` |
| `ok  dti.csv  4 rows; 1 duplicate subject row(s) -- each subject must appear once` and `1 problem(s).` | `dti.csv` or `mri.csv` was edited by hand | Rebuild it with `build` |

---

## sc_exclusions.py: subjects left out on purpose

### What is it?

The published results leave out one subject. The project's outlier audit
flagged it because its mean diffusivity (a measure of how freely water moves in
the tissue) is about ten times normal, the level of the fluid around the brain.
Leaving it out is part of the method.

A subject ID cannot be stored in the repository, because ADNI IDs are covered by
the Data Use Agreement. So the list lives in a small file that git ignores, and
`sc_exclusions.py` reads it. It is a helper module that other programs use; it
has no command of its own.

### Do I need it?

**It is optional.** You need it only if you want the published machine-learning
(ML) numbers exactly. If you do not have the subject ID, skip it: the whole
analysis still runs, the doctor shows a `warn` (not a problem), and the runner
prints a note. The only difference is that the ML stages then include that
subject, so their numbers differ from the published ones.

### The file

Location: `configs/exclusions.local.yaml` (change it with `SC_EXCLUSIONS`, or
with `--exclusions` on the analysis runner). Git ignores every
`*.local.yaml` file, so it stays on your computer.

Format (YAML, a simple text format for settings):

```yaml
exclude_subjects:
  - XXX_S_0001          # reason, for example: CSF-level mean diffusivity, ~10x normal
```

To create it, open a new file in any plain-text editor, for example with
`nano configs/exclusions.local.yaml` (in `nano`, save with Ctrl+O then Enter,
and leave with Ctrl+X). Type or paste the two lines above. Keep the two spaces
before the `-` and the `:` after `exclude_subjects`, and use spaces, not the
Tab key. Save the file.

Put the real subject ID in place of `XXX_S_0001`. The ID is not in this
repository and this page does not give it. If you need it, ask the owners of the
repository you got the code from. Anyone who receives it must hold their own
ADNI Data Use Agreement. If you cannot get it, leave the file out.

The same file may also hold named subject lists under `subject_lists:`. Some
imaging-side code and one test read them. The analysis does not need them, and
a missing list simply counts as empty.

### How to check it

This prints where the file is looked for and how many subjects it lists,
without printing the IDs:

```bash
python -c "import sc_exclusions as e; print(e.path(), len(e.load()))"
```

What you should see:

```text
<repo>/configs/exclusions.local.yaml 1
```

Without the file it prints the same path and `0`.

`python sc_doctor.py --analysis` reports the same thing on its `exclusions`
line, and the analysis runner prints it at the start of every run:

```text
exclusions    : 1 subject(s) from <repo>/configs/exclusions.local.yaml
```

### What happens without it

The analysis runner prints:

```text
exclusions    : NONE (<repo>/configs/exclusions.local.yaml not found)
                The published ML results exclude one subject. Without the
                exclusions file, the ML stages will not reproduce them.
```

The list is applied in two stages, `build_exception_specificity` and
`build_simple_ml`; the stages that use their results inherit the difference.
Those two stages are separate scripts, and they also print a warning of their
own when the list is empty. The runner hides a script's output unless it fails,
so you normally only see it in a failure message or if you run the script
yourself:

```text
NOTE (build_simple_ml.py): no subject exclusions loaded from <repo>/configs/exclusions.local.yaml.
      The published results exclude one subject (CSF-level mean
      diffusivity, ~10x normal). Without that file this run will not
      reproduce them. See sc_exclusions.py.
```

### If something goes wrong

| What you see | Why | Fix |
|---|---|---|
| `could not read <file>: ParserError: ...` or `could not read <file>: ScannerError: ...`, and the program stops | The YAML is badly formed (wrong indentation, a Tab character, a stray bracket) | Fix the file so it looks exactly like the example above. A broken file stops the run on purpose, so it is never mistaken for "no exclusions". The doctor also stops at this point (see [sc_doctor.py](#sc_doctorpy-the-preflight-check)). |
| `AttributeError: 'str' object has no attribute 'get'` at the end of a long error | The `:` after `exclude_subjects` is missing | Add the `:`. The first line must read `exclude_subjects:` |
| The count is `0` but the file exists | The key is misspelled or the list is empty | The key must be `exclude_subjects:` and each ID a line starting with `- ` |

---

## sc_package.py: build the shareable zip

### What is it?

`sc_package.py` makes a zip file of the code that is safe to give to someone
else. It takes only files that git tracks, leaves out a few folders, and then
scans every file for anything that must not leave the computer. If it finds
something, it refuses to write the zip. This scan is called the **safety gate**.

### Do I need it?

Only if you share the code as a zip. You do not need it to run anything.

### Before you start

- The folder must be a git checkout (a copy made with `git clone`, with its
  `.git` folder). A folder unpacked from a zip will not work.
- The working tree must be clean: every change committed and no new files that
  git does not ignore. Check with `git status`.

### How to run it

1. **Check only, write nothing:**

   ```bash
   python sc_package.py --check
   ```

   What you should see:

   ```text
   revision : 8854cc2
   contents : 500 files, 9.6 MB   (469 excluded)

   safety gate
   --------------------------------------------------------------------
     clean: no participant identifiers, credentials or data files.

   --check: nothing written.
   ```

   `revision` is the short name of your current git commit. Your numbers may
   differ.

2. **Build the zip:**

   ```bash
   python sc_package.py
   ```

   What you should see (last line):

   ```text
   wrote <repo>/dist/adni-sc-pipeline-8854cc2.zip  (4.2 MB)
   ```

### Outputs

| What | Where |
|---|---|
| The zip | `dist/adni-sc-pipeline-<revision>.zip` (`dist/` is ignored by git) |
| Inside the zip | one top folder, `adni-sc-pipeline-<revision>/`, holding the code |

### What is left out

| Left out | Why | Keep it with |
|---|---|---|
| Everything git does not track (data, cohort tables, manifests, `env.sh`, the exclusions file) | The zip is made with `git archive`, which only sees tracked files | cannot be included |
| `research_audit/` (except `matrix_data_dictionary.md` and `validate_connectome_v2_contract.py`, which the imaging contract needs) | The audit trail. It is not needed to run anything and it names ADNI participants. | `--include-audit` |
| `scripts/hcp/` | Separate work on a different cohort | `--include-audit` |
| Documents: `.pptx .docx .doc .ppt .pdf .xlsx .numbers .key` | Presentations and reports, not code (about 72 MB) | `--include-documents` |

### The safety gate

After assembling the contents, the tool scans them and refuses to write the zip
if it finds:

- anything shaped like an ADNI participant ID (three digits, `_S_`, four or five
  digits);
- an AWS access key or secret key setting;
- a private key (`-----BEGIN ... PRIVATE KEY-----`);
- a password written into the code (`password = "..."`);
- a data file (`.nii .gz .mif .tck .dcm .mgz .trk .sqlite .sqlite3 .db .pem
  .key`) anywhere outside `atlas/`. The atlas volumes are templates, not
  participant data.

The text patterns are searched in text-like files (`.py .sh .md .txt .yaml .yml
.json .cfg .toml .ini .smk .csv .tsv .ipynb .ts .tsx .js .html` and files
without an extension). The patterns and extension lists are constants at the top
of `sc_package.py`.

This is what a blocked build looks like (here `--include-audit` pulls in the
audit trail, which names participants; IDs replaced by `NNN_S_NNNN`):

```text
$ python sc_package.py --check --include-audit
revision : 8854cc2
contents : 931 files, 24.5 MB   (38 excluded)

safety gate
--------------------------------------------------------------------
   121  ADNI participant identifier

  first 15:
    research_audit/SUPERLIST.md                                ADNI participant identifier: NNN_S_NNNN
    ...

121 problem(s). Nothing written.
Fix them, or re-run with --force if every one is a false positive.
```

The screen output shows the text it found, so do not paste it anywhere public.

### Settings you can change

| Flag | What it does | Default |
|---|---|---|
| `--check` | Run everything, including the gate, but write nothing | off |
| `--out-dir <folder>` | Where the zip is written | `dist/` |
| `--name <text>` | Start of the zip name | `adni-sc-pipeline` |
| `--include-documents` | Keep presentations and reports | off |
| `--include-audit` | Keep `research_audit/` and `scripts/hcp/`. For an internal hand-off only. | off |
| `--force` | Write the zip even if the gate finds something | off |

Only use `--force` after you have looked at every finding and are sure each one
is a false alarm. For a public release the gate should say `clean`.

Exit status: `0` success, `1` the gate found something (nothing written), `2`
the working tree is not clean.

### If something goes wrong

| What you see | Why | Fix |
|---|---|---|
| `working tree is not clean; commit first so the package matches a revision.` | There are uncommitted changes or new files | Commit them (or move them out of the repository), then check with `git status` |
| `subprocess.CalledProcessError: Command '['git', 'status', '--porcelain']' returned non-zero exit status 128.` | The folder is not a git checkout | Run it in a copy made with `git clone` |
| `N problem(s). Nothing written.` | The gate found something | Remove the ID, key or data file from the tracked files, commit, and run again |

---

## sc_localize.py: point the imaging settings at this folder

### What is it?

Almost everything finds its folders through `sc_config.py`. The imaging
workflow is the exception: four settings files record full folder paths on
purpose, so that a run records exactly what it used. When the repository is in
a different folder, or on a different computer, those paths are wrong, and the
imaging workflow fails at its first step. `sc_localize.py` rewrites the old
repository folder to the current one in those four files:

- `configs/connectome_v2.yaml`
- `configs/connectome_v2_retry3.yaml`
- `scforge/workflow/environment_contract.yaml`
- `scforge/configs/scforge.yaml`

It changes only paths that start with the old repository folder. Paths outside
it (for example to the imaging programs, or to a separate data disk) are left
for you to edit; step 4 below shows how to find them.

Because those files are protected by a list of checksums (fingerprints that show
whether a file changed), the tool then refreshes that list by running
`scforge/workflow/lock_source_manifest.py`. That updates
`scforge/workflow/workflow_source_manifest.tsv` and the fingerprint recorded in
`environment_contract.yaml`.

### Do I need it?

Only for the imaging half, once, after you clone or unpack the code into a new
folder. The analysis does not need it.

### How to run it

1. See what would change:

   ```bash
   python sc_localize.py --check
   ```

   On a freshly unpacked copy you should see:

   ```text
   from : <old repository folder>
   to   : <repo>

     configs/connectome_v2.yaml                         22 path(s)
     configs/connectome_v2_retry3.yaml                  22 path(s)
     scforge/workflow/environment_contract.yaml         16 path(s)
     scforge/configs/scforge.yaml                        5 path(s)

   65 path(s) would change. Nothing written.
   ```

   If the files already point here, you see
   `already pointing at this machine; nothing to do.`

2. Make the change:

   ```bash
   python sc_localize.py
   ```

   It prints the same table, then `rewrote 65 path(s) in 4 file(s)`, then
   `re-locking the source manifest` and the output of the lock tool, which
   includes a line like `wrote 29 rows to workflow_source_manifest.tsv` (the
   number may differ). It ends with `Now run: python sc_doctor.py`.

   The lock tool also prints `CONTENT CHANGED (1):` followed by
   `configs/connectome_v2.yaml`. That is expected, not an error: the tool has
   just changed that file, and the lock step records the new version.

3. Run `python sc_localize.py --check` again. It should say
   `already pointing at this machine; nothing to do.`

4. List the full paths it did **not** change (those outside the repository
   folder):

   ```bash
   grep -nE '(: |- )"?/' configs/connectome_v2.yaml configs/connectome_v2_retry3.yaml scforge/workflow/environment_contract.yaml scforge/configs/scforge.yaml | grep -v "$PWD"
   ```

   Each line shows the file, the line number and the path. On a fresh copy,
   after step 2, there are 68 such lines: a system Python (used by MRtrix3's
   helper scripts) and system program folder, and the MRtrix3, MRtrix3Tissue
   and FSL folders (including FSL's MNI brain template) on the computer the
   files came from. For example:

   ```text
   configs/connectome_v2.yaml:152:    executable: "<old home>/fsl/bin/eddy_cpu"
   ```

   Change each one, with a text editor, to where that program is on your
   computer. What these programs are for belongs to the imaging half: see the
   Imaging section of the top-level [README.md](README.md).

5. **After you edit any of these four files by hand**, refresh the checksum
   list. Otherwise the workflow's contract check reports the file as changed:

   ```bash
   python scforge/workflow/lock_source_manifest.py --check
   ```

   If it prints `manifest is out of date (run without --check to rewrite)`, run
   `python scforge/workflow/lock_source_manifest.py`. That prints
   `wrote 29 rows to workflow_source_manifest.tsv` and
   `re-pinned environment_contract.yaml: ...`. Run the `--check` line again; it
   should say `manifest is current, and the contract pin matches.`

These files are tracked by git, so afterwards `git status` shows them as
changed. That is expected on a new computer. (It also means `sc_package.py` will
refuse to run until they are committed or reset.)

### Settings you can change

| Flag | What it does | Default |
|---|---|---|
| `--check` | Report only; write nothing | off |
| `--from-root <path>` | The old repository folder to replace | found automatically from the files |
| `--to-root <path>` | The folder to write instead | the folder holding `sc_localize.py` |
| `--no-relock` | Skip refreshing the checksum list (leaves it out of date) | off |

If the lock step fails, the tool prints
`re-lock failed; run scforge/workflow/lock_source_manifest.py by hand.`
Run `python scforge/workflow/lock_source_manifest.py` yourself.

---

## Imaging tools (not covered here)

These also sit in the repository root, but belong to the imaging half, which is
being rewritten. They are not documented on this page.

| File | Where it is explained |
|---|---|
| `run_imaging.py` | imaging — see [README.md](README.md) |
| `sc_discover.py` | imaging — see [README.md](README.md) |
| `sc_probe_acquisition.py` | imaging — see [README.md](README.md) |
| `sc_manifest.py` | imaging — see [README.md](README.md) |
