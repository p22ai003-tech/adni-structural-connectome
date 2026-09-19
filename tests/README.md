# tests/ — automatic checks for the code

## What is this folder?

A **test** is a small program that runs a piece of the project's code and
checks that the answer is what it should be. If the answer is right, the test
**passes**. If not, it **fails** and tells you what it expected and what it got.
You never check anything by hand. You type one command and read the last line.

The tool that finds and runs the tests is called **pytest**.

This folder holds one group of tests, `dashboard_parity/`. It checks the
project's web dashboard. The same command also runs the tests kept in two other
folders, `scforge/tests/` and `research_audit/tests/`. This page covers all of
them.

## Do I need it?

- **Run the tests** after you change any code, to check you broke nothing.
- **You can ignore this folder** if you only want to run the analysis. The
  analysis does not use the tests.

Many of these tests were written for the **original project server**: the one
computer where this project was first run. It holds the data, the analysis
results, the audit records and the imaging tools, each at a fixed folder
location. On any other computer a known set of tests fails. That is expected,
and this page tells you exactly which ones (see "Which result to expect").

## What's inside

This folder:

| File | What it does |
|---|---|
| `dashboard_parity/conftest.py` | Shared set-up for the dashboard tests. (pytest automatically loads a file called `conftest.py` before the tests next to it.) It loads the saved reference numbers, computes the same numbers again with the current code, and starts the dashboard's web API inside the test. No real server or network port is opened. |
| `dashboard_parity/test_core_reference.py` | 6 tests. Checks the dashboard's calculation code against a frozen copy (the **reference**) of what the original dashboard showed. |
| `dashboard_parity/test_api_contracts.py` | 17 tests. Asks the dashboard's web API questions and checks each answer: shape, numbers and safety rules. |

The whole test suite (set up by `pytest.ini` in the repository root):

| Where | Tests | What they cover |
|---|---:|---|
| `tests/` (this folder) | 23 | The dashboard: its calculation code and its web API. |
| `scforge/tests/` | 128 | The imaging workflow package. See the Imaging section of the top-level `README.md`. |
| `research_audit/tests/` | 119 | The project's audit scripts. Not in the shared package (see below). |
| **Total** | **270** | |

`pytest.ini` (in the repository root) sets two things you never have to type:

- `testpaths = tests scforge/tests research_audit/tests` tells pytest where
  the tests are. Without it, pytest would search the whole repository,
  including the very large `data/` folder, and would look stuck.
- `pythonpath = scforge .` lets the tests `import` the project's code.

## Before you start

1. Install everything as described in `requirements/README.md` (or the Install
   section of the top-level `README.md`). The normal install
   (`.venv/bin/pip install -r requirements.txt`) already includes `pytest` and
   the dashboard test packages (`fastapi`, `httpx`).
2. Open a terminal in the **repository root** (the folder with `pytest.ini` in
   it) and switch the venv on:

   ```bash
   source .venv/bin/activate
   ```

   A **venv** is a private Python installation for this project. Your prompt
   now starts with `(.venv)`.

## Which result to expect

How many tests pass depends on which copy of the project you have. Find your
case first. In the repository root, type:

```bash
ls -d research_audit/tests research_audit/outputs
```

(`ls -d` lists the folder names themselves, not what is inside them.)

| What `ls -d` prints | Your case | Command to use in Step 1 |
|---|---|---|
| `ls: cannot access 'research_audit/tests': No such file or directory` (and the same for `outputs`) | **A. The shared package**: the zip made by `sc_package.py`. | the one with the two `--ignore=` options |
| `ls: cannot access 'research_audit/outputs': No such file or directory`, then `research_audit/tests` | **B. A full copy from git** (for example `git clone`). | the plain one |
| `research_audit/outputs` and `research_audit/tests`, no error | **C. The original project server** | the plain one |

Why case A needs two `--ignore=` options: the shared package leaves out the
audit trail. It has no `research_audit/tests/` folder. Only two reference
files are kept in `research_audit/`: `matrix_data_dictionary.md` and
`validate_connectome_v2_contract.py`. pytest skips the missing test folder
without complaint. But two files in `scforge/tests/` import audit scripts that
are not there. Without the two `--ignore=` options, pytest stops before
running anything (see "If something goes wrong").

`research_audit/outputs/` (the saved audit records) is never shared, in any
case. It is excluded from git and from the package, because it names ADNI
participants.

## How to run it

**Step 1. Run everything.**

Case A, the shared package:

```bash
python -m pytest -q \
  --ignore=scforge/tests/test_gpu_transition_gate.py \
  --ignore=scforge/tests/test_prepare_h04a_r1_retry3.py
```

Case B and case C:

```bash
python -m pytest -q
```

`-q` means "quiet": one character per test instead of one line.
`.` = passed, `F` = failed, `s` = skipped, `E` = error. An **error** is
different from a failure: the test could not even start, because something it
needs to prepare first (for example, reading a file) went wrong.

What you should see. It takes about 1 to 3 minutes on the original server,
and a few seconds to a minute elsewhere (most failing tests stop early). The
last line is the one that matters:

| Case | Last line we got |
|---|---|
| A. shared package, on a computer that is not the original server | `33 failed, 100 passed, 2 skipped, 2 warnings, 1 error, 25 subtests passed in 4.67s` |
| B. full git copy, on a computer that is not the original server | `46 failed, 211 passed, 2 skipped, 16 warnings, 11 errors, 33 subtests passed in 12.31s` |
| C. the original project server | `2 failed, 268 passed, 16 warnings, 33 subtests passed in 110.34s (0:01:50)` |

("Subtests" are small checks inside a single test. They are counted
separately.) Your numbers can differ a little. What matters is that **every
failure, error and skip is in the lists in "Tests that only pass on the
original server"** (and, in case C, in "The two tests that fail even on the
original server"). A failure that is not in those lists means something is
wrong.

In case C the output looks like this (the third dot line is shortened):

```
........................................................... [ 21%]
............................................................... [ 45%]
..........................F...........................F........ [ 94%]
...............                                                          [100%]
=================================== FAILURES ===================================
...
=============================== warnings summary ===============================
...
=========================== short test summary info ============================
FAILED research_audit/tests/test_build_h04a_canary_package.py::H04ACanaryPackageTests::test_decision_draft_cannot_authorize_launcher
FAILED research_audit/tests/test_run_fixed_data_feasibility.py::FixedDataFeasibilityTests::test_workflow_contract_detects_required_policy_conflicts
2 failed, 268 passed, 16 warnings, 33 subtests passed in 110.34s (0:01:50)
```

The `warnings summary` block lists `DeprecationWarning` lines from
third-party packages (`starlette`, `anyio`, `pyparsing` via `matplotlib`).
They come from packages that pip chose, not from this project's code, and they
are harmless. One of them says
``Using `httpx` with `starlette.testclient` is deprecated; install `httpx2` instead.``
**Do not install `httpx2`.** The tests use `httpx`, and that warning changes
nothing.

**Step 2. Run only this folder (the dashboard tests).**

```bash
python -m pytest -q tests/dashboard_parity
```

What you should see on the original server (about 15 to 25 seconds):

```
.......................                                                  [100%]
...
23 passed, 2 warnings in 23.41s
```

On any other computer, without the project's results:

```
17 failed, 4 passed, 1 skipped, 2 warnings, 1 error in 1.01s
```

**Step 3. Run one file, or only the tests whose names match a word.**

```bash
python -m pytest -q tests/dashboard_parity/test_core_reference.py
python -m pytest -q tests/dashboard_parity -k "registry or escape or api_is_read_only"
```

What you should see on the original server:

```
6 passed, 2 warnings in 20.47s
```

```
3 passed, 20 deselected, 2 warnings in 0.04s
```

On any other computer the first command gives
`3 failed, 2 passed, 2 warnings, 1 error`. The second gives
`3 passed, 20 deselected, 2 warnings` everywhere, because those three tests
need no data. `-k` picks tests by name. "Deselected" means "not run this time".

**Step 4. Count the tests without running them.**

```bash
python -m pytest --collect-only -q
```

**Collecting** means pytest finding and loading the test files, before any
test runs. The last line should be:

```
270 tests collected in 2.48s
```

In case A, add the two `--ignore=` options from Step 1. The last line is then
`136 tests collected`.

## Tests that only pass on the original server

These tests read files that exist only on the original project server. Some
read the analysis results. Others look a file up at a **fixed absolute path
written in the test code** (the original server's repository folder, or the
folder where MRtrix3 or FSL is installed there). Those paths do not go through
the `$SC_...` settings, so you cannot point them somewhere else. On every other
computer, the tests below fail, error or are skipped. That is expected. (The
only ones that can change are the 14 result-reading dashboard tests described
below, and only once you have made results that match the published ones.)

**`tests/dashboard_parity/`: 17 failed, 1 error, 1 skipped (of 23).**

Four can never pass outside the original server:

| Test | What it needs | Why it cannot pass elsewhere |
|---|---|---|
| `test_api_contracts.py::test_health_and_metadata` | The API's **readiness check** (a web address, `/api/v1/health/ready`, that answers "ready" only when all its input files exist). One of those inputs is the frozen reference. | The frozen reference is read from a fixed path and is not distributed. The API answers `503` (not ready), so the test fails with `assert 503 == 200`. |
| `test_core_reference.py::test_tract_summary_matches_frozen_streamlit` | The frozen reference file `section_11.json` | Fixed path, not distributed. This is the one **error** (`E`): the set-up step that reads the file fails with `FileNotFoundError ... connectome_dashboard_reference_v1/section_11.json`, so the test never starts. |
| `test_api_contracts.py::test_frontend_production_build_is_shadow_scoped` | The built web page, `apps/connectome_web/dist/` | The test reads it at a fixed path. You can build the page yourself (see `apps/connectome_web/README.md`), but this test still looks in the original server's folder, so it fails with `FileNotFoundError ... apps/connectome_web/dist/index.html`. |
| `test_api_contracts.py::test_pipeline_poll_payload_is_compact_and_read_only` | A status file under `research_audit/outputs/` | That folder is never distributed. It fails with `AttributeError: 'NoneType' object has no attribute 'is_file'`. |

The other 14 failures read the analysis results in `$SC_ANALYSIS_ROOT` and
compare them with the published numbers written into the tests (for example
530 subjects). Without results they fail with lines like `assert 404 == 200`,
`KeyError` or `assert 0 == 530`. They can pass only if your results are the
same as the published ones. The skip is
`test_matrix_endpoint_is_allow_listed`: it is skipped when there are no
connectome matrices in `$SC_CONNECTOMES_DIR`.

**`scforge/tests/`: 16 failed, 1 skipped (of 128).** These check the imaging
set-up of the original server itself: its repository folder, its installed
MRtrix3 and FSL programs, and their fingerprints. They fail with
`FileNotFoundError` naming a fixed path, or with an audit message such as
`identity.environment_contract_path`.

| File | Tests that fail |
|---|---:|
| `scforge/tests/test_aal3_label_mapping.py` | 2 |
| `scforge/tests/test_environment_contract.py` | 4 |
| `scforge/tests/test_h04a_retry4_pretract_recovery2_converter.py` | 1 (and 1 skipped: `no processed units available under the recovery run root`) |
| `scforge/tests/test_launcher_terminal_finalization.py` | 3 |
| `scforge/tests/test_response_calibration.py` | 6 |

**`research_audit/tests/` (case B only): 13 failed, 10 errors (of 119).**
They read saved records from `research_audit/outputs/`, which is never
distributed, or audit scripts at the original server's fixed path. They fail
with `FileNotFoundError ... research_audit/outputs/...` or
`RuntimeError: Cannot read ...`. The files are
`test_build_h04a_canary_package.py` (5 errors),
`test_run_fixed_data_feasibility.py` (5 errors),
`test_audit_catalog_sources.py` (4), `test_audit_historical_source_preflight.py` (1),
`test_build_source_metadata_preflight.py` (1),
`test_thesis_grade_reliability_validators.py` (5) and
`test_vfinal_confirmation_executor.py` (2).

Adding it up: case A = 17 + 16 = 33 failed, 1 error, 2 skipped. Case B adds
the 13 failed and 10 errors from `research_audit/tests/`.

## The two tests that fail even on the original server

Both live in `research_audit/tests/`, and both need the saved records in
`research_audit/outputs/`. They were written in July 2026. Files they look at
changed after that, so they no longer match. Nothing is broken.

| Test | What it checks | Why it fails now |
|---|---|---|
| `test_build_h04a_canary_package.py::...::test_decision_draft_cannot_authorize_launcher` | A saved decision record holds the SHA-256 fingerprint (a unique code computed from a file's exact contents) of `scforge/workflow/workflow_source_manifest.tsv`. The test recomputes the fingerprint and compares. | The manifest was updated in September 2026, after the record was saved. The fingerprints differ: `AssertionError: '1e27facd...' != '6053a2f5...'`. |
| `test_run_fixed_data_feasibility.py::...::test_workflow_contract_detects_required_policy_conflicts` | The test checks that the input manifest named in `configs/connectome_v2.yaml` does **not** exist. | The test was written when the config pointed at a manifest that did not exist yet. The config now points at one that does, so the check fails: `AssertionError: True is not false`. |

## What `dashboard_parity` checks

"Parity" means "the same". The project has an older dashboard and a newer one.
The older one is built with **Streamlit** (a Python tool that turns a script
into a web page). The newer one has two parts: a **FastAPI** web API (a Python
program that answers questions sent to web addresses) and a **React** web page
(the part you see in the browser). These tests check that the new one shows
the same things as the old one, and that it is safe.

`test_core_reference.py`:

- the dashboard has 18 sections, with the same names in the same order;
- the tract-length summary for each group matches the frozen reference;
- the long-range / short-range exception thresholds and edge counts match;
- the four requested group comparisons give the same medians, effect sizes
  and corrected p-values;
- the machine-learning feature table has the same number of features and
  subjects in each feature family;
- a file request that tries to climb out of the results folder
  (`../../../../etc/passwd`) is refused.

`test_api_contracts.py` (every question is sent to the API inside the test).
An **endpoint** is one web address of the API, such as
`/api/v1/health/live`:

- the health endpoints answer, with the expected **security headers** (extra
  lines the API adds to every answer to tell the browser to be careful, for
  example `x-content-type-options: nosniff`);
- the API is read-only: any attempt to write (POST, PUT, PATCH, DELETE) gets
  error `405`;
- unknown sections and files get `404`;
- a downloaded results file is byte-for-byte the same as the file on disk;
- the long-range / short-range, demographics, node-ranking, network, coupling,
  machine-learning and findings answers carry the expected numbers and row
  counts;
- a connectome matrix can only be asked for by one of the allowed weight types
  (anything else gets `422`);
- no answer contains `NaN` or `Infinity`, which are not valid JSON;
- the stored machine-learning predictions are served as saved, never re-fitted;
- the built web page (`apps/connectome_web/dist/`) exists, uses the `/next/`
  address, and ships no **source maps** (extra files that let anyone read the
  page's original source code).

The groups in these checks are CN (cognitively normal), MCI and AD. That is
the project's grouping rule: EMCI, LMCI and SMC are grouped with MCI, and only
CN, MCI and AD are analysed.

## Inputs

What the tests read. Nothing here is in git. ADNI data (the Alzheimer's Disease
Neuroimaging Initiative) come only from https://adni.loni.usc.edu/ under your
own Data Use Agreement, and must never be committed.

| What | Where | How it is found | Needed by |
|---|---|---|---|
| Analysis results | `$SC_ANALYSIS_ROOT` (default `$SC_DERIV_ROOT/qc/analysis_cohort`), made by `python -m connectome_analysis.run_analysis --all` | `$SC_...` setting | most `dashboard_parity` tests |
| Connectome matrices, `SC_AAL166_<SUBJECT>_I<IMAGEID>_<type>.csv` | `$SC_CONNECTOMES_DIR` (default `$SC_DERIV_ROOT/connectomes`) | `$SC_...` setting | the matrix endpoint test (**skipped** if there are no matrices) and the readiness check |
| The frozen dashboard reference | `research_audit/outputs/connectome_dashboard_reference_v1/` in the original server's repository folder. **Not distributed.** | fixed path in `tests/dashboard_parity/conftest.py` and `apps/connectome_api/main.py` | `test_tract_summary_matches_frozen_streamlit` and the readiness check |
| The built web page (not in git; built from `apps/connectome_web/`, see `apps/connectome_web/README.md`) | `apps/connectome_web/dist/` in the original server's repository folder | fixed path in `test_api_contracts.py` | `test_frontend_production_build_is_shadow_scoped` |
| Saved audit records. **Not distributed.** | `research_audit/outputs/` | fixed path, or relative to the repository | many `research_audit/tests`, and one dashboard test |
| The imaging tools and the original repository folder | fixed folders on the original server | fixed paths in `scforge/tests/` | 16 `scforge` tests |

The `$SC_...` names are environment variables (settings your terminal passes to
programs). They are resolved in `sc_config.py`. To see where they point on your
machine:

```bash
python -c "import sc_config; print(sc_config.describe())"
```

This shows only the `$SC_...` locations. It does not show the fixed paths in
the table above, because those are written directly into the test and API code.

## Outputs

The tests do not write into the repository, `data/`, `cohort/`, `configs/` or
the analysis results. Tests that need scratch files make them in your system's
temporary folder.

pytest keeps a small memory of the last run in `.pytest_cache/` in the
repository root. It is listed in `.gitignore`. To turn it off, add
`-p no:cacheprovider`.

## Settings you can change

| Setting | Where | What it does | Default |
|---|---|---|---|
| Where the tests are | `testpaths` in `pytest.ini` | Folders pytest searches when you name none. | `tests scforge/tests research_audit/tests` |
| Folders never searched | `norecursedirs` in `pytest.ini` | Keeps pytest out of big or irrelevant folders. | `data dist archive node_modules .git .venv* .envs tools` |
| Import path | `pythonpath` in `pytest.ini` | Lets tests import the project's code. | `scforge .` |
| Results the dashboard tests read | `SC_ANALYSIS_ROOT` (or the dashboard-only override `CONNECTOME_ANALYSIS_ROOT`) | Point the tests at a different results folder. The numbers must match the published ones for the tests to pass. | `$SC_DERIV_ROOT/qc/analysis_cohort` |
| Matrices the dashboard tests read | `SC_CONNECTOMES_DIR` (or `CONNECTOME_MATRICES_DIR`) | Where to find connectome matrices. | `$SC_DERIV_ROOT/connectomes` |
| Stop at the first failure | `-x` on the command line | Handy while fixing one problem. | off |
| Pick tests by name | `-k "word"` | Run only tests whose names contain the word. | all tests |
| Time limit per test | `--timeout=SECONDS` (from `pytest-timeout`) | Fails a test that runs longer than this. | no limit |
| Skip a file | `--ignore=path/to/test_file.py` | Leave one file out. | none skipped |

## If something goes wrong

**`python: command not found`** (Linux), **`Command 'python' not found`**
(Ubuntu) or **`zsh: command not found: python`** (Mac)
The venv is not switched on, and your computer has no plain `python` command.
Run `source .venv/bin/activate` in the repository root and try again. Your
prompt should start with `(.venv)`.

**`/usr/bin/python3: No module named pytest`** (the path at the start can differ)
The venv is not switched on, so `python` is some other Python. Run
`source .venv/bin/activate` and try again.

**`ModuleNotFoundError: No module named 'fastapi'`**
You installed only `requirements/analysis.txt`. Install the full set:
`.venv/bin/pip install -r requirements.txt`. It looks different depending on
the command:

- with `python -m pytest -q` (Step 1):
  `ERROR tests/dashboard_parity - ModuleNotFoundError: No module named 'fastapi'`
  and then `Interrupted: 1 error during collection`;
- with `python -m pytest -q tests/dashboard_parity` (Steps 2 and 3):
  `ImportError while loading conftest '.../tests/dashboard_parity/conftest.py'`
  and then `E   ModuleNotFoundError: No module named 'fastapi'`.

**`Interrupted: 2 errors during collection`** naming
`scforge/tests/test_gpu_transition_gate.py` and
`scforge/tests/test_prepare_h04a_r1_retry3.py`
(with `ImportError: cannot import name 'build_gpu_transition_gate' from 'research_audit'`
or `FileNotFoundError: ... research_audit/prepare_h04a_r1_retry3.py`)
You are running the shared package (case A), which has no audit scripts. Use
the Step 1 command with the two `--ignore=` options.

**Many `dashboard_parity` tests fail with lines like `assert 503 == 200`,
`assert 404 == 200` or `KeyError`**
Expected on any computer that is not the original server. See "Tests that only
pass on the original server". `503` is the API saying "I am not ready: my
input files are missing". If you have made results and expect the
result-reading tests to pass, check where the paths point with
`python -c "import sc_config; print(sc_config.describe())"`. To make the
results, see `connectome_analysis/README.md`. Four of these tests still fail,
because they read files at fixed paths that exist only on the original server.

**`FAILED ... test_decision_draft_cannot_authorize_launcher` and
`FAILED ... test_workflow_contract_detects_required_policy_conflicts`**
Expected on the original server. See "The two tests that fail even on the
original server". In case B you see these two as errors instead, together with
the other `research_audit` errors.

**pytest runs far fewer than 270 tests**
You ran it from inside a sub-folder. pytest then runs only the tests below that
folder (from `tests/` you get 23, from `scforge/` you get 128). `cd` to the
repository root and run the Step 1 command again. (In case A, 136 is correct.)

**pytest seems to hang and never prints a dot**
You probably ran it from a folder *above* the repository, such as your home
folder. pytest looks for `pytest.ini` only in the folder you are in and the
folders above it, so it did not find it. It is now searching everything,
including the very large `data/` folder. Press Ctrl+C, `cd` to the repository
root, and run the Step 1 command again.
