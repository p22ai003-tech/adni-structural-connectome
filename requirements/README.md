# requirements/ — the Python packages this project needs

## What is this folder?

This project is written in Python. Python on its own cannot do statistics,
machine learning or read brain images. For that it uses extra add-on libraries
called **packages** (for example `numpy`, `pandas`, `scikit-learn`).

The files in this folder are shopping lists. Each line names one package and
the exact version to install. A program called **pip** (Python's package
installer) reads a list and downloads everything on it.

The imaging tools (MRtrix3, FSL, ANTs) are not Python packages and are not in
these lists. See the Imaging section of the top-level `README.md`.

## Do I need it?

- **Yes**, if you want to run the analysis, the tests, or the dashboard. You
  install from these lists once, at the start.
- You do not need to open or edit these files. The one command in "How to run
  it" below does everything.

## What's inside

| File | What it is for | Do I need it? |
|---|---|---|
| `../requirements.txt` (in the repository root) | The normal install. It just says "install `analysis.txt` and `dev.txt`". | Yes. This is the one to use. |
| `analysis.txt` | Everything the analysis needs: numbers and tables (`numpy`, `pandas`, `scipy`), statistics (`statsmodels`), machine learning (`scikit-learn`, `xgboost`, `shap`), brain networks (`networkx`), reading atlas images (`nibabel`), plots (`matplotlib`, `seaborn`), config files (`PyYAML`), and reading scanner files (`pydicom`, `dcm2niix`). | Yes (pulled in by `requirements.txt`). |
| `dev.txt` | Tools for checking the code: `pytest` runs the tests, `jsonschema` checks file formats, and `fastapi`, `httpx` and `uvicorn` run and test the dashboard's web API. | Yes (pulled in by `requirements.txt`). |
| `dashboard.txt` | The Streamlit dashboard in `apps/connectome_dashboard/`: `streamlit`, `plotly`, `altair`, `scikit-image`. Not needed to run the analysis or reproduce any number. | Only if you want the Streamlit dashboard. |
| `extras-ml.txt` | Two optional neural-network model libraries, `tabpfn` and `pytorch-tabnet`. Only `connectome_analysis/analysis_ml_neural_benchmarks.py` actually runs them, and it is not one of the standard analysis stages. The `ml_diagnostics` stage (and `analysis_ml_goal_search.py`) only write down whether they are installed: a status column says `available_not_wired` if they are, `unavailable_optional_dependency` if not. No numbers change. | Almost never. Read the warning below first. |

## Why every version is fixed ("pinned")

Every line says `==` and an exact version, for example `scikit-learn==1.5.2`.
This is called **pinning**. The versions are the ones installed in the
environment that computed the published results.

It matters because the machine-learning results change when scikit-learn
changes: newer releases pick tree splits differently, and the SHAP interface
also changed between releases. With the pinned versions you get the same
numbers. With "any newer version" you might not.

Only the packages we use directly are pinned. The packages they need in turn
(for example `pillow` for `matplotlib`) are chosen by pip.

## Before you start: get Python 3.12

Use **Python 3.12**. The pinned versions were taken from a Python 3.12
environment, and all tests were run on 3.12. Other versions can fail:
Python 3.9 cannot install `numpy 2.1.3` at all, and for Python 3.14 there is
no ready-made (pre-built) `numpy 2.1.3`.

**Which computers work.** Linux, or a Mac with a recent macOS. Some pinned
packages have no ready-made Mac version for old macOS releases, so on a Mac
you need:

- **macOS 12 (Monterey) or newer** on an Apple Silicon Mac (M1, M2, ...).
  On macOS 11 there is no ready-made `scipy 1.15.3`.
- **macOS 10.15 (Catalina) or newer** on an Intel Mac. On 10.14 there is no
  ready-made `xgboost 2.1.4`.

On an older macOS, pip fails or tries to build packages from source code, and
a different Python does not help. (We checked this by asking pip which
ready-made files exist for each macOS version.) Windows is not covered here.

First open a terminal and go to the **repository root** (the folder that has
`requirements.txt` and `sc_config.py` in it). Some of the options below make
files there, so do this before anything else:

```bash
cd path/to/the/repository
```

Then check whether you already have Python 3.12:

```bash
python3.12 --version
```

If you see `Python 3.12.something`, skip to "How to run it". If you see
`command not found`, install Python 3.12 using **one** of the options below.

**Any Linux or Mac, no admin rights needed: conda** (tested on Linux).
**conda** is a program that installs Python versions (and packages) into
separate "environments" in your home folder. **Miniconda** is the small
free installer for conda; **Anaconda** is the big one. If you have neither,
install Miniconda from https://www.anaconda.com/docs/getting-started/installation
(pick Miniconda, then your system), open a new terminal, and `cd` to the
repository root again. Then:

```bash
conda create -y -n py312 python=3.12
conda run -n py312 python -m venv .venv      # run from the repository root
```

This makes the `.venv` folder for you, so skip step 2 below.

**Any Linux or Mac, no admin rights needed: uv** (tested on Linux). **uv** is a
single program that can download Python versions and make venvs. Install it by
following https://docs.astral.sh/uv/ (the "Installation" section), open a new
terminal, `cd` to the repository root again, and then:

```bash
uv python install 3.12
uv venv --python 3.12 --seed .venv
```

`--seed` puts pip inside the new `.venv`. Without it, step 3 below fails. This
also makes the `.venv` folder, so skip step 2 below. If `uv python install`
warns that a folder "is not on your PATH", you can ignore that for this
project.

**Mac: python.org installer.** Go to https://www.python.org/downloads/macos/
and pick the newest Python 3.12 that has a "macOS 64-bit universal2
installer" (newer 3.12 releases only fix security issues and do not have one).
Run it. You get a `python3.12` command.

**Mac: Homebrew.** **Homebrew** is the usual way to install command-line
programs on a Mac. If you do not have it (`brew: command not found`), install
it from https://brew.sh. Then: `brew install python@3.12`

**Ubuntu 24.04:** `sudo apt install python3.12 python3.12-venv`

We did not run the python.org, Homebrew and apt options while writing this
page. They are the standard installers for those systems.

## How to run it

Every command is typed in a terminal, from the **repository root** (the folder
that has `requirements.txt` and `sc_config.py` in it).

**Step 1. Go to the repository root.**

```bash
cd path/to/the/repository
ls requirements.txt sc_config.py
```

What you should see: both file names printed back, with no error.

**Step 2. Make a venv.** A **venv** ("virtual environment") is a private copy of
Python just for this project. Packages you install into it do not touch the
rest of your computer. It lives in a folder called `.venv`.

```bash
python3.12 -m venv .venv
```

What you should see: nothing. A new `.venv/` folder appears. (Skip this step if
you used conda or uv above. They already made `.venv`.)

**Step 3. Install the packages.**

```bash
.venv/bin/pip install -r requirements.txt
```

What you should see: a lot of `Collecting ...` and `Downloading ...` lines,
then one long line that starts like this:

```
Successfully installed PyYAML-6.0.2 annotated-doc-0.0.5 ... xgboost-2.1.4
```

How long and how big (measured on Linux, fast internet, empty download cache):

- about **40 seconds** (41 s measured);
- about **760 MB** downloaded;
- about **1.6 GB** on disk inside `.venv/`.

About 450 MB of the 1.6 GB is `nvidia-nccl-cu12`, an NVIDIA graphics-card
library that `xgboost` always brings along on ordinary (x86-64) Linux
computers. It is harmless, and you do not need a graphics card. On a Mac it is
not installed, so the Mac download is smaller.

pip may also print `[notice] A new release of pip is available`. You can ignore
it.

**Step 3a. Mac only: install OpenMP.** On a Mac, `xgboost` needs a helper
library called OpenMP. The Mac `xgboost` files look for it in Homebrew's
folder and do not include it. Install it once with Homebrew (see "Mac:
Homebrew" above if you do not have `brew`):

```bash
brew install libomp
```

Do this even if you already had Python 3.12. Without it, Step 5 stops with an
`XGBoostError` (see "If something goes wrong"). On Linux, skip this step.

**Step 4. Switch the venv on.**

```bash
source .venv/bin/activate
```

What you should see: your prompt now starts with `(.venv)`. From now on,
`python` means the project's Python. Check it:

```bash
python --version
```

```
Python 3.12.2
```

(Your last number may be different. The `3.12` is what matters.) You need to
type `source .venv/bin/activate` again every time you open a new terminal.
Every other README in this repository assumes you have done it.

**Step 5. Check that everything is there.**

```bash
python -m pip check
python sc_doctor.py --analysis
```

What you should see: `pip check` prints

```
No broken requirements found.
```

`sc_doctor.py` is the project's **preflight check**: it looks at your computer
before you run anything and lists what is there and what is missing. It prints
three parts: "Paths" (your data folders), "Python packages" and "Cohort
tables". **At this point only the "Python packages" part matters.** It should
list every package as `ok`:

```
Python packages
------------------------------------------------------------------------
  interpreter: .../.venv/bin/python  (Python 3.12.2)
  ok    numpy                      2.1.3
  ok    pandas                     2.3.2
  ok    scipy                      1.15.3
  ok    sklearn                    1.5.2
  ...
  ok    xgboost                    2.1.4
  warn  tabpfn                     optional
  warn  pytorch_tabnet             optional
  ok    jsonschema                 4.25.1
  ok    pytest                     9.1.1
  ok    pydicom                    3.0.1
```

The two `warn ... optional` lines are normal. They are the `extras-ml.txt`
packages, which you did not install.

Three more things you will see, all normal after a fresh install:

- The very first two lines are a warning from the `jsonschema` package. It is
  harmless:

  ```
  .../sc_doctor.py:148: DeprecationWarning: Accessing jsonschema.__version__ is deprecated and will be removed in a future release. Use importlib.metadata directly to query for jsonschema's version.
    _print(OK, mod, getattr(m, "__version__", ""))
  ```

- On a new computer you have no data and no cohort tables yet, so the other
  two parts show `MISS` lines, and the last line counts them as problems:

  ```
  Paths
  ------------------------------------------------------------------------
    ok    project_root               ...
    MISS  data_root                  .../data   <- set SC_DATA_ROOT
  ...
  Cohort tables
  ------------------------------------------------------------------------
    MISS  dti.csv                    .../cohort/dti.csv   <- build with: python sc_cohort.py build --exports <folder>
    MISS  mri.csv                    ...
    MISS  dti_master.csv             ...
    MISS  mri_master.csv             ...

  ========================================================================
  5 problem(s). The pipeline will not complete until these are fixed.
  ```

- Because of those problems, `sc_doctor.py` ends with exit status 1 (the
  code a program hands back to say "not everything is ready").

None of this means the install failed. The install is fine when the "Python
packages" part has no `MISS` line. Getting the data and building the cohort
tables are covered in the top-level `README.md`. Once they are done, the same
command ends with `no problems found.`

That is the whole install. The rest of this page is optional.

### Optional: the Streamlit dashboard

```bash
.venv/bin/pip install -r requirements/dashboard.txt
```

Install it on top of step 3, never instead of it. Measured: 10 to 20 seconds
and about 0.3 to 0.4 GB more. `python -m pip check` should still say
`No broken requirements found.`

### Optional: the neural-network extras (read this first)

`tabpfn` and `pytorch-tabnet` need **PyTorch**. On Linux, pip by default
downloads the PyTorch build for NVIDIA graphics cards, which brings about
**6 GB** of CUDA (NVIDIA graphics-card) libraries. You do not need that. On a
Linux computer without an NVIDIA graphics card, install the small CPU-only
PyTorch **first**, then the extras:

```bash
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -r requirements/extras-ml.txt
```

Measured on Linux: 30 to 40 seconds and about 0.9 GB more. The first command
ends with `Successfully installed ... torch-2.x.x+cpu`. The `+cpu` tells you
that you got the small build. After this, `python sc_doctor.py --analysis`
shows `tabpfn` and `pytorch_tabnet` as `ok`.

The CUDA libraries are only ever downloaded on Linux, so on a Mac you can
leave out the first command. (We did not try the extras on a Mac.)

None of the published headline results need these packages. They use
scikit-learn's ExtraTrees model. If the extras are missing, only the optional
neural-network comparison is unavailable.

## Inputs and outputs

| | What | Where |
|---|---|---|
| Input | The package lists | `requirements.txt` and the four `requirements/*.txt` files |
| Input | An internet connection to https://pypi.org (and https://download.pytorch.org for the CPU PyTorch) | |
| Output | The installed packages | `.venv/` in the repository root |

`.venv/` is listed in `.gitignore`, so git never commits it. To start over,
delete it (`rm -rf .venv`) and repeat steps 2 to 5.

No data is involved in installing. ADNI data (the Alzheimer's Disease
Neuroimaging Initiative) is never downloaded by these commands. It comes only
from https://adni.loni.usc.edu/ under your own Data Use Agreement, and it must
never be committed to this repository.

## Settings you can change

| Setting | Where | What it does | Default |
|---|---|---|---|
| Which lists to install | the `-r` file on the pip command line | `requirements.txt` = analysis + tests. Add `requirements/dashboard.txt` and/or `requirements/extras-ml.txt` if you need them. | `requirements.txt` |
| Package versions | the `==` number on each line | Changing a pin can change the published numbers, especially `scikit-learn`. Do not change pins to reproduce results. | as listed |
| PyTorch build | `--index-url` on the `pip install torch` command | `.../whl/cpu` gives the small CPU-only build. On Linux, leaving it out gives the large GPU build. | CPU build, if you follow this page |
| Where the venv lives | the folder name in `python3.12 -m venv .venv` | Every README assumes `.venv` in the repository root. | `.venv` |

## If something goes wrong

**`python3.12: command not found`**
Python 3.12 is not installed, or it is not on your PATH (the list of folders
your terminal searches for commands). Install it with one of the options in
"Before you start".

**`ERROR: Could not find a version that satisfies the requirement numpy==2.1.3`**
followed by **`ERROR: No matching distribution found for numpy==2.1.3`**
You made the venv with the wrong Python. We got this exact error with
Python 3.9. Check with `.venv/bin/python --version`. If it is not 3.12, delete
the venv (`rm -rf .venv`) and make it again with `python3.12 -m venv .venv`.
The same fix applies if pip starts compiling packages from source code
(`Building wheel for numpy ...`) instead of downloading ready-made ones.

**`ERROR: Could not open requirements file: [Errno 2] No such file or directory: 'requirements.txt'`**
You are not in the repository root. `cd` into the folder that has
`requirements.txt` in it and run the command again.

**`.venv/bin/pip: No such file or directory`**
The venv has no pip. This happens with `uv venv` if you leave out `--seed`.
Delete `.venv` and make it again with `uv venv --python 3.12 --seed .venv`.

**Mac: Step 5 stops with a long error (a "traceback") instead of finishing**
The last lines of it are:

```
xgboost.core.XGBoostError: 
XGBoost Library (libxgboost.dylib) could not be loaded.
Likely causes:
  * OpenMP runtime is not installed
    - vcomp140.dll or libgomp-1.dll for Windows
    - libomp.dylib for Mac OSX
    - libgomp.so for Linux and other UNIX-like OSes
    Mac OSX users: Run `brew install libomp` to install OpenMP runtime.
```

(then a line starting `Error message(s):` with your computer's own details).
`sc_doctor.py` stops in the "Python packages" part and never reaches
"Cohort tables". The OpenMP helper library is missing: you skipped Step 3a.
Run `brew install libomp`, then check with
`python -c "import xgboost; print(xgboost.__version__)"`, which should print
`2.1.4`, and run Step 5 again. Fix this before running the analysis: when
`xgboost` cannot load, the analysis skips the XGBoost models without stopping,
and the published `ml_diagnostics` results include XGBoost rows.

**`python: command not found`**, **`Command 'python' not found`** (Ubuntu) or
**`zsh: command not found: python`** (Mac)
The venv is not switched on, and your computer has no plain `python` command
of its own. Run `source .venv/bin/activate` in the repository root (Step 4)
and try again. Your prompt should start with `(.venv)`. You need to do this in
every new terminal.

**pip downloads several GB of `nvidia-...` packages while installing `extras-ml.txt`**
You installed the extras before the CPU-only PyTorch. Stop pip (Ctrl+C), then
run the two commands in "Optional: the neural-network extras" in order. The
~450 MB `nvidia-nccl-cu12` from step 3 is different: it comes with `xgboost`
and is expected.

**`sc_doctor.py` shows `MISS` next to a package, or `MISS python 3.12 required, found 3.9...`**
You are running a different Python, not the one in `.venv`. The `interpreter:`
line at the top of that section tells you which one. Usually you forgot
`source .venv/bin/activate`. Your prompt should start with `(.venv)`. Activate
and run the check again.
