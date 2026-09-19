# hcp_analysis/: the exception-specificity and machine-learning build scripts

## What is this folder?

This folder holds the last part of the analysis: eleven Python scripts whose
names start with `build_`, plus one helper, `sc_paths.py`. The scripts start
from tables that earlier analysis stages have already written, mainly the EDR
exceptions in `17_edr_exceptions/` and the brain-network tables in
`19_network_analysis/`. With them they:

1. describe the **exception** connections network by network, and test which
   differences between CN, MCI and AD show up inside those connections (the
   *exception-specificity* part);
2. build one table of features per participant (the *ML feature matrix*);
3. train models that predict two clinical scores, MMSE and CDR, and measure
   how well they do;
4. explain the models: which features pushed each prediction up or down.

Each script writes straight into the folder the dashboard reads:
`20_exception_specificity/`, or `19_network_analysis/functional/` for the
network tables. Nothing has to be copied by hand.

The analysis runner (`python -m connectome_analysis.run_analysis`) runs these
scripts as its `build` group of stages. You can also run each script by itself.

### Why is it called `hcp_analysis`?

The name is historical. It comes from earlier work in this project. It does
**not** mean that these scripts use Human Connectome Project (HCP) data or an
HCP atlas. Everything here works on the ADNI connectomes built with the
166-region AAL3 atlas. The folder keeps its name because other code finds the
scripts by this path: the runner looks for them in `hcp_analysis/`.

## Do I need it?

- **Yes**, if you want the exception-specificity tables, the ML feature
  matrix, the final model's scores, or the SHAP and partial-dependence tables.
  These feed the "exception-specific signal" section of the dashboard's LR/SR
  page.
- **You rarely run the scripts one at a time.** The runner starts them for you,
  in the right order (see [How to run it](#how-to-run-it)). Read this README
  to learn what each script does and writes, or to run one alone.
- **No**, if you only need the group statistics that `connectome_analysis/`
  writes (every other numbered folder of the analysis root). See
  `connectome_analysis/README.md`.
- **No**, if you want to make connectome matrices from MRI images. See the
  Imaging section of the top-level `README.md`.

## Words used in this README

| Word | Meaning |
|---|---|
| repository root | The top folder of the project, the one holding `README.md` and `sc_config.py`. Run every command from there. |
| venv | A private Python installation for this project, in the folder `.venv`. Switch it on with `source .venv/bin/activate`. |
| analysis root | The folder all analysis results go into: `$SC_ANALYSIS_ROOT`. By default that is `$SC_DERIV_ROOT/qc/analysis_cohort`, which is `data/derivatives/qc/analysis_cohort` inside the repository if you set nothing. |
| stage | One step of the analysis. Each `build_*.py` script here is one stage with the same name. |
| CN, MCI, AD | Cognitively normal, mild cognitive impairment, Alzheimer's disease. The project's grouping rule: EMCI, LMCI and SMC are grouped with MCI, and only CN, MCI and AD are analysed. The scripts read the group from the `group` column of `00_master/master_cohort.csv`, where this rule is already applied. |
| EDR exception | EDR is the "exponential distance rule": longer connections are usually weaker. An exception is a connection whose strength is more than 3 standard deviations above the average for connections of similar length in the same person. The stage `edr_exceptions` finds them. |
| SR, MR, LR | Short-range (tract length up to 78.69 mm), mid-range, and long-range (longer than 171.02 mm) connections. The two cut points are the lower and upper quartiles of tract length in the CN group. |
| network | Each of the 166 brain regions belongs to one of 10 networks: Visual, Somatomotor, DorsalAttention, Salience_VAN, Limbic, Frontoparietal, DMN (default mode network), Subcortical, Cerebellar, Brainstem. The list is `19_network_analysis/network_mapping_used.csv`. |
| FA, MD, RD, AxD | Four diffusion measures of white matter: fractional anisotropy, mean, radial and axial diffusivity. In matrix file names axial diffusivity is `ad_mean`. |
| DIFF | One "diffusivity composite" per network: the average of MD, RD and AxD after each is put on a common scale (median 0, interquartile range 1). |
| MMSE | Mini-Mental State Examination, a 30-point test of memory and thinking. Higher is better. |
| CDR | Clinical Dementia Rating: 0 = no dementia, 0.5 = very mild, 1 or more = mild or worse. |
| MMSE band | MMSE split into three groups of about equal size: Low (impaired), Mid, High (intact). |
| feature | One input column for a model, for example `WB_DMN_DIFF`. |
| ExtraTrees | "Extremely randomised trees", the main model type used here: many decision trees whose votes are averaged. |
| cross-validation, out-of-fold | The participants are split into 5 parts. The model learns on 4 parts and predicts the 5th, five times over, so every prediction is made on people the model has not seen. Those predictions are called out-of-fold. |
| seed | A number that fixes the random choices. The scripts repeat the work with five seeds (11, 23, 37, 51, 73) and average, so one lucky split cannot decide the result. |
| AUC | Area under the ROC curve. 0.5 means guessing, 1.0 means perfect sorting of two classes. |
| PR-AUC | Area under the precision-recall curve. Its "chance" level equals how common the class is. |
| SHAP | A method that says how much each feature pushed one person's prediction up or down. |
| PDP, ICE | Partial dependence plot and individual conditional expectation: how the prediction changes when one feature is moved and everything else is held still, on average (PDP) and per person (ICE). |
| residualise | Remove the part of a feature that age and sex already explain, using a straight-line fit. |
| environment variable | A named setting that your terminal passes to every program it starts, for example `SC_ANALYSIS_ROOT`. Set it for one command by writing it in front: `SC_ANALYSIS_ROOT=<folder> python ...`. Set it for everything you run later in the same terminal window with `export SC_ANALYSIS_ROOT=<folder>`. |
| exit status | A number that a command leaves behind when it ends: 0 means it worked, any other number means it stopped with a problem. Type `echo $?` straight after a command to see it. |
| CSV, JSON | Two plain-text file formats that any text editor can open. CSV ("comma-separated values") is a table, one row per line. JSON holds small labelled records. |
| standard deviation (SD) | How spread out a set of values is around its average. |
| quartile, interquartile range | The lower quartile is the value below which a quarter of the values fall; the upper quartile, three quarters. The interquartile range is the distance between the two, the spread of the middle half. |
| p-value | The chance of seeing a difference at least this large if the groups were really the same. Small (for example below 0.05) means chance is an unlikely explanation. |
| q-value, FDR | A p-value corrected for running many tests at once, so that a few will not look important by luck alone. FDR means "false discovery rate"; the scripts use the Benjamini–Hochberg method. |
| Kruskal–Wallis test | Tests whether three groups (here CN, MCI, AD) differ. It works on ranks (the order of the values), not the values themselves. |
| Brunner–Munzel test | Tests whether two groups differ, working on ranks. It does not assume the two groups are equally spread out. |
| Welch t-test | Tests whether two groups have different averages. It does not assume the two groups are equally spread out. |
| Cliff's delta | An effect size from -1 to 1: how often a value from one group is larger than a value from the other, minus how often it is smaller. 0 means no difference. |
| ADNI phase, ADNI-3 | ADNI collected data in phases, one after another. The phase says in which one a participant's scan was taken. ADNI-3 is the third phase. |
| decision tree | A model that predicts by asking a chain of yes/no questions about the features. |
| HistGradientBoosting | "Gradient boosting": many small decision trees built one after another, each one correcting the mistakes of the trees before it. Named `hist_gb` in the tables. |
| Huber regression, logistic regression | Straight-line models. Huber regression predicts a number and is not thrown off much by a few extreme values. Logistic regression predicts the probability of a class. |
| stratified | The five cross-validation parts are chosen so that each has about the same mix of classes. |
| R² | For a number such as MMSE: the share of the differences between people that the predictions explain. 1 is perfect, 0 is no better than always guessing the average, below 0 is worse than that. |
| ROC curve, precision-recall (PR) curve | Two ways to draw how well a model separates a class from the rest as its decision threshold moves. ROC: share of real cases found against share of non-cases wrongly flagged. PR: precision (share of flagged people who really are cases) against recall (share of real cases found). |
| balanced accuracy | The share of people put in the right class, worked out for each class separately and then averaged, so a large class cannot hide mistakes on a small one. |
| bootstrap | Draw the same number of participants again at random, allowing repeats, and recompute the score; do this many times (here 2000). The middle 95% of those scores is the 95% interval. |
| permutation test | Shuffle the targets at random so they no longer belong to the right people, fit and score the model again, and repeat (here 200 times). The p-value is about the share of shuffles that score at least as well as the real data. |

## What's inside

**The eleven build scripts**, in the order the runner uses:

| File | What it does |
|---|---|
| `build_range_restricted_networks.py` | For every participant and each network, recomputes strength, degree, FA, MD, RD and AxD using only SR connections, only LR connections, and only the SR or LR **exception** connections. Also computes exception burden (share of connections that are exceptions) and exception strength per network. Then compares CN, MCI and AD. Writes to `19_network_analysis/functional/`. |
| `build_consensus_core.py` | Finds the **consensus core**: the connections that are exceptions in more than half of all participants (51 of them). Reports how often each one is present in CN, MCI and AD, and how each participant's exceptions spread over pairs of networks. Has a `--verify` option. |
| `build_exception_specificity.py` | Part A. (A1) For each network, measure, range and pair of groups: is the group difference larger inside the exception connections than inside all connections of the same range? (A2) Sixteen per-participant **exception architecture** features (for example how many exceptions a person has, what share of them join two regions of the same network, what share fall on the consensus core), with group tests. |
| `build_simple_ml.py` | Part B. Builds the ML targets (each person's MMSE and CDR from the visit nearest the scan) and the feature matrix (103 features). Trains three model types on a five-step **feature ladder** with repeated cross-validation, and runs a 200-shuffle permutation test. Every later ML script reads its two main outputs. |
| `build_model_interpretation.py` | Out-of-fold R² (MMSE) and AUC (CDR 0 vs 0.5 or more) for each ladder step; top features by SHAP; per-person SHAP values; ICE and PDP curves. Draws `fig4_shap_mmse.png` and `fig5_ice_mmse.png`. |
| `build_mmse_buckets.py` | Treats MMSE as three bands and reports AUC and PR-AUC per band for every ladder step and model, plus SHAP per band. Draws `fig6_mmse_buckets.png`. |
| `build_final_model_metrics.py` | Scores **the final model** (see [The final model](#the-final-model)) for MMSE bands and three CDR levels: AUC with 95% intervals, PR-AUC and balanced accuracy, overall and within each diagnostic group, ROC and PR curve points, and two cross-check models (logistic regression and gradient boosting). |
| `build_pdp_range_family.py` | Partial dependence for one network measure, the DMN diffusivity composite, in its four versions: SR, LR, SR exceptions, LR exceptions. |
| `build_ml_explain_v2.py` | A dictionary of every feature, a preprocessing audit, a table of the pipeline steps, a feature-selection experiment, SHAP and PDP by class and diagnostic group, and a short JSON summary. Redraws `fig4_shap_mmse.png` with colours based on rank. |
| `build_explain_v3.py` | ROC and PR curve points for CDR (0 vs 0.5 or more), and per-person SHAP values for every class. |
| `build_shap_overall.py` | SHAP averaged over five seeds and computed over all 105 inputs (103 features plus age and sex). Gives each feature's share of the whole model. **Must run last**: it rewrites four tables that `build_ml_explain_v2.py` and `build_explain_v3.py` also write. |

**Helpers and other files**

| File | What it is |
|---|---|
| `sc_paths.py` | Works out every folder the build scripts read and write, from `sc_config.py` and the `SC_*` environment variables. Gives every script the same three options: `--out`, `--analysis-root`, `--figs`. Run it by itself to print the folders. |
| `improve/` | A record of the "improvement sweep": experiments that tried to beat the first model. The runner does not use them. See [The `improve/` folder](#the-improve-folder). |
| `README.md` | This file. |

On a machine where these scripts ran before they learned to write into the
analysis tree, you may also find old result files (`*.csv`, `*.json`) directly
in this folder. They are old copies of participant-level ADNI results. Git
ignores them (see `.gitignore`), and no build script reads them. Only the
`improve/` scripts do.

## Before you start

1. **The Python environment.** Follow the Install section of the top-level
   `README.md`. In short, from the repository root:

   ```bash
   python3.12 -m venv .venv
   .venv/bin/pip install -r requirements.txt
   source .venv/bin/activate
   ```

   Every command below assumes the venv is switched on and you are in the
   repository root. The package versions are pinned on purpose
   (`requirements/analysis.txt`): with another scikit-learn or shap version the
   model numbers move slightly.

2. **ADNI data, under your own agreement.** The connectome matrices and cohort
   tables come from ADNI data, which you must obtain yourself from
   https://adni.loni.usc.edu/ under your own Data Use Agreement. Never commit
   ADNI data, cohort tables, the exclusions file, or any result from this folder
   to git. `.gitignore` already blocks the usual locations.

3. **The cohort tables** in `$SC_COHORT_DIR` (default `cohort/`).
   `build_simple_ml.py` reads the scan date from `dti.csv`, and the MMSE and
   CDR scores of every visit from `dti_master.csv` and `mri_master.csv` (and,
   for people with no dated score there, from `00_master/master_cohort.csv`).
   Check them:

   ```bash
   python sc_cohort.py check
   ```

   What you should see (`<repo>` stands for your repository folder):

   ```
   cohort folder: <repo>/cohort

     ok       dti.csv            1114 rows
     ok       mri.csv            1114 rows
     ok       dti_master.csv     8481 rows
     ok       mri_master.csv    27396 rows

   no problems found.
   ```

   If they are missing, build them from the spreadsheets you exported from
   ADNI:

   ```bash
   python sc_cohort.py build --exports <folder with your ADNI exports>
   ```

   The exports folder must hold `dti_master.csv` and `mri_master.csv`, and
   optionally `all_mri.csv` (`python sc_cohort.py build --help` says the
   same). Without `--exports` the command stops with
   `sc_cohort.py build: error: the following arguments are required: --exports`.
   What goes into those exports is explained under
   [Cohort tables in `connectome_analysis/README.md`](../connectome_analysis/README.md#cohort-tables).
   `configs/README.md` (section `cohort_rules/`) runs the same command on a
   small made-up example.

4. **The connectome matrices** in `$SC_CONNECTOMES_DIR`. Only
   `build_range_restricted_networks.py` reads them. How they are made: see the
   Imaging section of the top-level `README.md`.

5. **The earlier stages must have run.** The build scripts need the outputs of
   the stages `master_cohort`, `network_analysis` and `edr_exceptions`. To
   find out whether your analysis root already has them, do the dry run of
   [Step 2](#step-2-look-at-the-plan)
   (`python -m connectome_analysis.run_analysis --group build --dry-run`). If
   it shows no `[BLOCK]` lines, skip this step. Otherwise make them:

   ```bash
   python -m connectome_analysis.run_analysis --stage edr_exceptions --stage network_analysis --with-deps --resume
   ```

   `--with-deps` also runs the stages that these two need. `--resume` skips
   every stage whose outputs already exist and are newer than its inputs, so a
   stage that has already finished is not run again. From scratch this runs
   six stages and took about 25 minutes on a busy machine (`network_analysis`
   is the slow one). It should end with `ok=6`.

   Add `--dry-run` to see the plan without running it. What you should see
   with `--dry-run` on an analysis root where none of the six has run yet (the
   first lines, which list the folders, are trimmed):

   ```
   plan          : 6 stage(s) [resume] [dry-run]

     [dry]   analysis_snapshot  -> 1 output(s)
     [dry]   master_cohort  -> 1 output(s)
     [dry]   live_node_microstructure  -> 2 output(s)
     [dry]   live_node_graph  -> 2 output(s)
     [dry]   network_analysis  -> 4 output(s)
     [dry]   edr_exceptions  -> 4 output(s)

   dry-run=6   total 0.2s
   ledger: <analysis root>/exports/stage_status.csv
   ```

   Where all six are already done, every line reads `[skip]` instead and the
   summary is `skipped=6`.

6. **The exclusions file (needed to match the published numbers).** The
   published ML results leave out one participant, flagged by the project's
   outlier check (mean diffusivity about ten times the normal value). ADNI
   identifiers may not be published, so the ID lives in a local file that git
   ignores, `configs/exclusions.local.yaml`.

   **The ID is not in this repository or in any README.** Ask the project
   author for it privately, under your own ADNI Data Use Agreement. How to
   ask, and how to create the file step by step, is in
   [`configs/README.md`](../configs/README.md#exclusionslocalyaml-private-never-committed).
   Once you have the real ID, the file looks like this, with the real ID in
   place of `XXX_S_0001`:

   ```yaml
   exclude_subjects:
     - XXX_S_0001
   ```

   **Do not create the file with the placeholder `XXX_S_0001` in it.** A
   placeholder matches nobody, so nobody is left out, yet it hides the
   "no subject exclusions" warning, and `sc_doctor.py` still marks the
   `exclusions` line `ok` with `1 subject(s)`. The real check is the participant
   count: with the right file, `build_exception_specificity.py` prints
   `A2 subject features: 529 subjects x 16 features`. With no file, or with a
   placeholder, it prints `530 subjects`.

   `build_exception_specificity.py` and `build_simple_ml.py` read the file.
   Without it they keep that participant, and the ML tables will not match the
   published ones; everything else still runs. To keep the file somewhere
   else, set `SC_EXCLUSIONS=<file>` (or give the runner `--exclusions <file>`).

7. **Check the machine.**

   ```bash
   python sc_doctor.py --analysis
   ```

   It lists folders, Python packages and cohort tables, each marked `ok`,
   `warn` or `MISS`. The last line should be `no problems found.`

   Two things in this output are harmless and need no action: a
   `DeprecationWarning` about `jsonschema` at the very top (two lines), and
   `warn` lines ending in `(absent; created on demand)`, for `run_state_dir`
   and `manifest`. Only `MISS` lines need fixing.

## How to run it

### Step 1. See which folders the scripts will use

```bash
python hcp_analysis/sc_paths.py
```

What you should see (your repository folder instead of `<repo>`):

```
  out          <repo>/data/derivatives/qc/analysis_cohort/20_exception_specificity
  analysis     <repo>/data/derivatives/qc/analysis_cohort
  func         <repo>/data/derivatives/qc/analysis_cohort/19_network_analysis/functional
  exceptions   <repo>/data/derivatives/qc/analysis_cohort/17_edr_exceptions
  connectomes  <repo>/data/derivatives/connectomes
  figs         <repo>/docs/figs
  cohort       <repo>/cohort
  aal_node_map <repo>/atlas/AAL/aal3_node_map_166.csv
```

`out` is where the ML scripts write. `python hcp_analysis/sc_paths.py --family
network` shows where `build_range_restricted_networks.py` writes (the `func`
folder). A folder that does not exist yet is marked `<- does not exist`. This
command only prints; it creates nothing.

### Step 2. Look at the plan

```bash
python -m connectome_analysis.run_analysis --list
```

The `build` part at the end should look like this (trimmed):

```
--- build -----------------------------------------------------
  26. build_range_restricted_networks  [script]
      Range-restricted and exception network tables.
      needs: edr_exceptions, network_analysis
      makes: 5 file(s), e.g. 19_network_analysis/functional/network_range_restricted_stats.csv
  27. build_consensus_core             [script]
      The consensus exception core, its loss across groups, and network-pair shares.
      needs: edr_exceptions, network_analysis
      makes: 3 file(s), e.g. 20_exception_specificity/consensus_core_edges.csv
  ...
  36. build_shap_overall               [script]
      Seed-averaged SHAP over all features; supersedes the earlier tables.
      needs: build_ml_explain_v2, build_explain_v3
      makes: 6 file(s), e.g. 20_exception_specificity/ml_shap_class_group.csv

37 stages.
```

Then a dry run, which checks the inputs and runs nothing:

```bash
python -m connectome_analysis.run_analysis --group build --dry-run
```

What you should see when the earlier stages are done (the first lines, which
list the folders it will use, are trimmed):

```
plan          : 11 stage(s) [dry-run]

  [dry]   build_range_restricted_networks  -> 5 output(s)
  [dry]   build_consensus_core  -> 3 output(s)
  [dry]   build_exception_specificity  -> 3 output(s)
  [dry]   build_simple_ml  -> 5 output(s)
  [dry]   build_model_interpretation  -> 7 output(s)
  [dry]   build_mmse_buckets  -> 5 output(s)
  [dry]   build_final_model_metrics  -> 4 output(s)
  [dry]   build_pdp_range_family  -> 1 output(s)
  [dry]   build_ml_explain_v2  -> 4 output(s)
  [dry]   build_explain_v3  -> 2 output(s)
  [dry]   build_shap_overall  -> 6 output(s)

dry-run=11   total 0.2s
ledger: <analysis root>/exports/stage_status.csv
```

If you see `[BLOCK]` lines instead, go back to step 5 of
[Before you start](#before-you-start). A dry run changes no results. It does
write two small status files into the analysis root
(`exports/stage_status.csv` and `00_master/run_analysis_status.json`) and
creates a few empty folders there.

### Step 3. Run the first script by hand

At the moment the runner cannot start `build_range_restricted_networks.py`.
The script reads its first command-line word as a number (see
[Settings you can change](#settings-you-can-change)), but the runner's first
word is `--analysis-root`, so the stage fails with
`ValueError: invalid literal for int() with base 10: '--analysis-root'`.
Run it yourself first:

```bash
python hcp_analysis/build_range_restricted_networks.py
```

What you should see (about 25 seconds):

```
CN-referenced ranges: SR <= 78.69 mm | LR > 171.02 mm
subjects: 530 (image-id remapped: 32)
exception edges loaded for 530 subjects
  ...50/530 subjects
  ...
  ...500/530 subjects
node rows: 175628 tract | 175628 exception | subjects missing matrices: 1
wrote network_range_restricted_stats.csv (120 rows) + subject-level (57563)
wrote network_exception_measures_stats.csv (112 rows) + subject-level (45232)
wrote network_exception_range_stats.csv (39 rows)
exception edges by class: {'MR': 64525, 'SR': 32545, 'LR': 29665}
```

Two counts in this output need a word:

- "image-id remapped" counts participants whose image ID in
  `master_cohort.csv` has no matrix file on disk. For them the script uses the
  matrix file that exists for the same participant under another image ID.
- "subjects missing matrices" counts participants with no usable matrix file
  at all. They are skipped.

### Step 4. Run the rest of the build group

```bash
python -m connectome_analysis.run_analysis --group build --resume
```

`--resume` skips stages whose outputs already exist and are newer than their
inputs, so the script from step 3 is not started again. What you should see
(the first lines, which list the folders, are trimmed; your times will
differ):

```
plan          : 11 stage(s) [resume]

  [skip]  build_range_restricted_networks
  [run]   build_consensus_core ...
  [ok]    build_consensus_core  (1.6s)
  [run]   build_exception_specificity ...
  [ok]    build_exception_specificity  (4.3s)
  [run]   build_simple_ml ...
  [ok]    build_simple_ml  (1527.2s)
  [run]   build_model_interpretation ...
  [ok]    build_model_interpretation  (226.0s)
  [run]   build_mmse_buckets ...
  [ok]    build_mmse_buckets  (191.8s)
  [run]   build_final_model_metrics ...
  [ok]    build_final_model_metrics  (122.7s)
  [run]   build_pdp_range_family ...
  [ok]    build_pdp_range_family  (115.3s)
  [run]   build_ml_explain_v2 ...
  [ok]    build_ml_explain_v2  (214.2s)
  [run]   build_explain_v3 ...
  [ok]    build_explain_v3  (23.2s)
  [run]   build_shap_overall ...
  [ok]    build_shap_overall  (675.2s)

ok=10  skipped=1   total 3101.8s
ledger: <analysis root>/exports/stage_status.csv
```

That run took about 50 minutes on a busy 64-core machine.
`build_simple_ml` is the slowest stage because of its 200-shuffle
permutation test. The terminal can stay quiet for many minutes while a stage
works; that is normal.

The runner does not show each script's own messages, only its `[ok]` or
`[FAIL]` line. If a script fails, the last lines of its error are printed
under `[FAIL]` and stored in `exports/stage_status.csv`. The runner's exit
status is 0 when nothing failed or was blocked, and 1 otherwise.

**Figures.** Three stages save PNG pictures into `docs/figs/` inside the
repository: `fig4_shap_mmse.png`, `fig5_ice_mmse.png` and
`fig6_mmse_buckets.png`. Those files are tracked by git. With the same data
and package versions the new pictures are identical to the committed ones;
if anything differs, `git status` shows them as modified. To put the
pictures somewhere else, make a copy of the settings file **outside** the
repository, add a `figs_dir:` line at its end, and give the copy to the
runner with `--config`. Decide this before you run step 4: the last line
below takes the place of step 4's command (once the stages are done,
`--resume` skips them and no pictures are drawn).

```bash
cp configs/analysis.yaml ../my_analysis.yaml
echo "figs_dir: ../my_figures" >> ../my_analysis.yaml
python -m connectome_analysis.run_analysis --group build --resume --config ../my_analysis.yaml
```

`..` means "the folder above the repository", so both the copy and the
figures folder sit next to the repository, not inside it. That keeps them
out of `git status`, and they cannot be committed by accident. The
`figs_dir:` line must start at the very left (not indented, not under
`defaults:` or `stages:`); `echo ... >>` adds it that way. A relative
`figs_dir` is taken from the repository root; the `--config` path is taken
from the folder you are in.

### Step 5. Check the result

This prints the build stages from the runner's record, and the headline
scores of the final model (the "macro" row, which averages the three
classes):

```bash
python - <<'EOF'
import pandas as pd, sc_config
root = sc_config.paths().analysis_root
status = pd.read_csv(root / "exports/stage_status.csv")
status = status[status.group == "build"]
print(status[["stage", "status", "seconds"]].to_string(index=False))
m = pd.read_csv(root / "20_exception_specificity/final_model_metrics.csv")
print(m[(m.scope == "All") & (m.cls == -1)][["target", "auc", "pr_auc"]].to_string(index=False))
EOF
```

The runner's record, `exports/stage_status.csv`, only holds the **most
recent** runner command that ran or planned stages. Every such command
replaces it, including a `--dry-run` and a run of another group (only
`--list`, and a run refused because another run holds the lock, leave it
alone). So run this check straight after step 4, before any other runner
command. If the status column says `dry-run`, or the build stages are
missing, a later command has replaced the record. The scores (the second
table) come from the result files, not from the record, so they are not
affected.

What you should see: every stage `ok` (or `skipped`), then the two targets.
With the same ADNI data, package versions and exclusions file as the
published run, the scores come out as below. When this README was written, a
fresh run on the same inputs matched the existing tables to within rounding
noise (differences below 1e-13).

```
                          stage  status  seconds
build_range_restricted_networks skipped     0.00
           build_consensus_core      ok     1.60
    build_exception_specificity      ok     4.34
                build_simple_ml      ok  1527.21
     build_model_interpretation      ok   225.96
             build_mmse_buckets      ok   191.78
      build_final_model_metrics      ok   122.69
         build_pdp_range_family      ok   115.26
            build_ml_explain_v2      ok   214.24
               build_explain_v3      ok    23.21
             build_shap_overall      ok   675.17
    target      auc   pr_auc
 mmse_band 0.599388 0.432916
cdr_3level 0.657990 0.495043
```

If you used a different analysis root, put `SC_ANALYSIS_ROOT=<folder>` in
front of `python`, so the snippet reads that folder.

### Practising in a separate folder

To practise without touching existing results, point everything at another
analysis root. Connectome matrices and cohort tables are only read, never
changed.

1. Make the earlier stages in that folder, as in step 5 of
   [Before you start](#before-you-start), with `--analysis-root <folder>`
   added.
2. Make a copy of the settings file with its own figures folder, outside the
   repository, as in the **Figures** paragraph of step 4:

   ```bash
   cp configs/analysis.yaml ../my_analysis.yaml
   echo "figs_dir: ../my_figures" >> ../my_analysis.yaml
   ```

3. Run the build group there:

   ```bash
   python hcp_analysis/build_range_restricted_networks.py 0 --analysis-root <folder> --figs ../my_figures
   python -m connectome_analysis.run_analysis --group build --resume --analysis-root <folder> --config ../my_analysis.yaml
   ```

   Choose a `<folder>` outside the repository too (for example
   `../my_analysis_root`), so that none of the practice results can be
   committed by accident.

The `0` in the first line means "all participants" (see
[Settings you can change](#settings-you-can-change)). It must come before the
options.

## Running one script by itself

Every script can run alone. With no options it reads from and writes to the
default analysis root. Run them in this order, because each one reads what the
ones before it wrote:

```bash
python hcp_analysis/build_range_restricted_networks.py   # needs edr_exceptions, network_analysis
python hcp_analysis/build_consensus_core.py              # needs edr_exceptions, network_analysis
python hcp_analysis/build_exception_specificity.py       # needs build_range_restricted_networks
python hcp_analysis/build_simple_ml.py                   # needs build_exception_specificity
python hcp_analysis/build_model_interpretation.py        # this one and the next five need build_simple_ml
python hcp_analysis/build_mmse_buckets.py
python hcp_analysis/build_final_model_metrics.py
python hcp_analysis/build_pdp_range_family.py
python hcp_analysis/build_ml_explain_v2.py
python hcp_analysis/build_explain_v3.py
python hcp_analysis/build_shap_overall.py                # always last
```

Two orders matter:

- **`build_shap_overall.py` last.** It rewrites `ml_shap_class_group.csv`,
  `ml_shap_class_group_dir.csv`, `ml_pdp_class_group.csv` and
  `ml_shap_beeswarm_class.csv` with the five-seed version computed over all 105
  inputs, with CDR as three levels. If `build_ml_explain_v2.py` or
  `build_explain_v3.py` runs after it, those four files go back to the older
  single-seed version (CDR as two classes, no `share_pct` column), and the
  dashboard's percentage shares no longer mean what they say.
- **`build_ml_explain_v2.py` after `build_model_interpretation.py`.** Both
  draw `fig4_shap_mmse.png`; the published picture is the one from
  `build_ml_explain_v2.py`.

**Only `build_consensus_core.py` and `sc_paths.py` understand `--help`.**
The other scripts do not:

- `build_range_restricted_networks.py` stops at once with
  `ValueError: invalid literal for int() with base 10: '--help'`, before it
  does any work. (It reads its first word as a number; see
  [Settings you can change](#settings-you-can-change).)
- The other nine scripts ignore `--help` and start working straight away,
  writing their results into the analysis root (and, for the three that draw
  pictures, into `docs/figs/`). If that happens, press Ctrl+C.

What each script prints last when it finishes well, and how long it took on
a busy 64-core machine:

| Script | Time | Last lines when it worked |
|---|---|---|
| `build_range_restricted_networks.py` | 25 s | `exception edges by class: {'MR': 64525, 'SR': 32545, 'LR': 29665}` (full output in [step 3](#step-3-run-the-first-script-by-hand)) |
| `build_consensus_core.py` | 2 s | `consensus core 51 edges (> 50% of subjects), 37 network pairs`, then `wrote 3 files to <analysis root>/20_exception_specificity` |
| `build_exception_specificity.py` | 5 s | `A2 subject features: 529 subjects x 16 features`, then `A2 stats: 16 features, ...` and one line per feature with a clear group difference |
| `build_simple_ml.py` | 25 min | `wrote ml_ladder_results.csv / ml_predictions.csv / ml_permutation.json` |
| `build_model_interpretation.py` | 4 min | `=== explaining cdr_bin ===`, a line of family shares, then `done` |
| `build_mmse_buckets.py` | 3 min | `wrote figs/fig6_mmse_buckets.png` |
| `build_final_model_metrics.py` | 2 min | two tables, headed `==================== mmse_band` and `==================== cdr_3level` |
| `build_pdp_range_family.py` | 2 min | `missing before imputation:`, then a `range_label` line, then one line per range (four) |
| `build_ml_explain_v2.py` | 3.5 min | `fig4 regenerated with rank colouring`, then `done` |
| `build_explain_v3.py` | 25 s | `total beeswarm rows: 6060`, then `done` |
| `build_shap_overall.py` | 11 min | `beeswarm rows 7272 · pdp rows 1080`, then two summary tables |

`build_simple_ml.py` is quiet for the longest. Its first lines tell you it
found the data:

```
targets: n=202 | mmse=202 | cdr=202 (bin balance 106/96) | ...
dropping 14 features with >50% missing: ['SREXC_Limbic_FA', 'SREXC_Limbic_DIFF', ...]...
features: 202 subjects x 103 columns

=== target mmse: n=202 ===
  F0     elasticnet_huber spearman_mean=0.027 r2_mean=-0.106 mae_mean=2.521
  ...
```

Then one line per ladder step and model, first for `mmse`, then for
`cdr_bin`, each followed by a `permutation (...)` line.

### Checking the consensus core against existing files

`build_consensus_core.py --verify` recomputes the three consensus-core tables
and compares them with the files already in the output folder. It writes no
result files (it only creates the output and figures folders if they are
missing):

```bash
python hcp_analysis/build_consensus_core.py --verify
```

What you should see when they match (tiny differences like `1.42e-14` are
rounding noise from saving numbers as text):

```
edge-level rows 126735, subjects 530, distinct edges 4853
consensus core 51 edges (> 50% of subjects), 37 network pairs

  consensus_core_edges.csv           reproduces (51 rows, max abs diff 1.42e-14 in pct)
  core_edge_loss.csv                 reproduces (51 rows, max abs diff 3.55e-15 in drop_pp)
  exception_networkpair_shares.csv   reproduces (37 rows, max abs diff 1.11e-16 in p)
```

When the folder has no such files yet, each line says
`no published file to compare against`. **Nothing was compared in that
case, but the exit status is still 0**, so do not read a 0 as "it matches":
look for the word `reproduces` on all three lines. A table that does not
match says `DIFFERS`, `ROW COUNT differs` or `only ... rows matched`, and the
exit status is then 1. Use `--out <folder>` to compare against copies kept
somewhere else.

## The feature ladder

`build_simple_ml.py` adds features in five steps ("rungs"), so you can see
what each family adds. Age and sex are always included.

| Step | Adds | Feature names start with | Count |
|---|---|---|---|
| `F0` | age and sex only | `age`, `sex` | 2 |
| `F0+F1` | whole-brain network FA and DIFF, plus global RD | `WB_` | 21 |
| `F0-F2` | the same, from SR and LR connections only | `SR_`, `LR_` | 40 |
| `F0-F3` | the same, from SR and LR **exception** connections only | `SREXC_`, `LREXC_` | 26 |
| `F0-F4` | the 16 exception-architecture features | `ARCH_` | 16 |

That makes 103 connectome features, or 105 inputs with age and sex. Features
missing in more than half of the participants are dropped first (14 of the
exception features). The three model types are a linear model (Huber
regression for MMSE, named `elasticnet_huber` in the tables; logistic
regression for CDR), HistGradientBoosting (`hist_gb`) and ExtraTrees
(`extra_trees`). `ml_feature_dictionary.csv` describes every feature in words.

## The final model

The accepted final model is written down, in words, in
`configs/analysis.yaml` under `stages:` → `build_simple_ml:`. The code that
fits and scores it is `build_final_model_metrics.py`:

1. fill missing values with the median, learned inside each training fold;
2. residualise the 103 connectome features on age and sex, with the fit
   learned inside each training fold only (age and sex themselves stay in,
   unchanged);
3. ExtraTrees classifier: 400 trees, at least 5 participants per leaf,
   `class_weight="balanced"` (rare classes count more);
4. 5-fold stratified cross-validation, repeated for the five seeds, with the
   out-of-fold probabilities averaged over the seeds.

Two targets, on the same 202 participants:

| Target | Classes | Sizes |
|---|---|---|
| `mmse_band` | Low (impaired), Mid, High (intact): MMSE tertiles | 71 / 68 / 63 |
| `cdr_3level` | CDR 0, CDR 0.5, CDR 1 or more | 106 / 67 / 29 |

Sex is coded 1 = female, 0 = male. Keep it that way: with ExtraTrees,
flipping the coding moves the AUCs in the third decimal.

The explanation tables (SHAP, PDP, ICE) come from separate ExtraTrees models
with 600 trees, fitted on all participants at once and without the
age-and-sex residualisation. So the AUCs are out-of-fold scores of the final
model, while the explanations describe those fitted models.

## Inputs

All inputs are read, never changed.

| File | Made by | Read by |
|---|---|---|
| `<analysis root>/00_master/master_cohort.csv` | stage `master_cohort` | range-restricted, exception-specificity, simple ML |
| `<analysis root>/17_edr_exceptions/edr_exception_thresholds.csv` | stage `edr_exceptions` | range-restricted |
| `<analysis root>/17_edr_exceptions/edr_exception_edge_level_fd_sum_len_mean.csv` | stage `edr_exceptions` | range-restricted, consensus core, exception-specificity |
| `<analysis root>/17_edr_exceptions/edr_exception_node_level_fd_sum_len_mean.csv` | stage `edr_exceptions` | range-restricted |
| `<analysis root>/17_edr_exceptions/edr_exception_subject_level_fd_sum_len_mean.csv` | stage `edr_exceptions` | exception-specificity |
| `<analysis root>/19_network_analysis/network_mapping_used.csv` | stage `network_analysis` | range-restricted, consensus core, exception-specificity |
| `<analysis root>/19_network_analysis/functional/network_microstructure_subject.csv` | stage `network_analysis` | simple ML |
| `<analysis root>/19_network_analysis/functional/network_microstructure_stats.csv`, `network_graph_stats.csv` | stage `network_analysis` | exception-specificity |
| `$SC_CONNECTOMES_DIR/SC_AAL166_<SUBJECT>_I<IMAGEID>_<type>.csv` for types `len_mean`, `fd_sum`, `fa_mean`, `md_mean`, `rd_mean`, `ad_mean` | imaging | range-restricted |
| `$SC_COHORT_DIR/dti.csv`, `dti_master.csv`, `mri_master.csv` | `python sc_cohort.py build --exports <folder with your ADNI exports>` (see step 3 of [Before you start](#before-you-start)) | simple ML |
| `atlas/AAL/aal3_node_map_166.csv` | shipped with the repository | consensus core, exception-specificity |
| `configs/exclusions.local.yaml` (or `$SC_EXCLUSIONS`) | you | exception-specificity, simple ML |
| the outputs of the build scripts before it | earlier build scripts | every build script after the first two |

Each connectome matrix is a headerless, comma-separated table of 166 rows and
166 columns, one row and one column per AAL3 region, for example
`SC_AAL166_XXX_S_0001_I123456_fd_sum.csv`.

## Outputs

All outputs are comma-separated tables with a header row, except the two
`.json` files and the three `.png` pictures. Files marked "per person" have
one row (or more) per participant, keyed by the ADNI subject ID. They are
restricted data and must never be shared or committed.

### In `<analysis root>/19_network_analysis/functional/`

Written by `build_range_restricted_networks.py`. The "stats" tables have one
row per network and measure, with the mean for CN, MCI and AD, a
Kruskal–Wallis p-value, and for each pair of groups Cliff's delta (an effect
size from -1 to 1) and a Brunner–Munzel p-value with its FDR-corrected
q-value.

| File | What is in it | Per person? |
|---|---|---|
| `network_range_restricted_subject.csv` | Strength, degree, FA, MD, RD, AxD per network, from SR or LR connections only. Measures are named like `SR-FA`. | yes |
| `network_range_restricted_stats.csv` | Group tests of the table above. | no |
| `network_exception_measures_subject.csv` | The same measures from SR or LR exception connections only (`SRexc-FA`, ...). | yes |
| `network_exception_measures_stats.csv` | Group tests of the table above. | no |
| `network_exception_range_stats.csv` | Group tests of exception burden and exception strength per network (`SRexc-Burden`, `LRexc-Strength`, ...). | no |

### In `<analysis root>/20_exception_specificity/`

| File | Written by | What is in it | Per person? |
|---|---|---|---|
| `consensus_core_edges.csv` | `build_consensus_core.py` | The 51 core connections: region names, % of participants, networks, same network or not, left-right mirror pair or not, median length, usual range class. | no |
| `core_edge_loss.csv` | `build_consensus_core.py` | For each core connection, the share of CN, MCI and AD participants who have it, the CN minus AD drop, and a Welch t-test p and q. | no |
| `exception_networkpair_shares.csv` | `build_consensus_core.py` | For each pair of networks, the average share of a participant's exceptions that falls there, per group, with Cliff's delta (CN vs AD), p and q. | no |
| `exception_tier_delta.csv` | `build_exception_specificity.py` | For each group pair, network, range and measure: effect size inside exception connections vs inside all connections of that range vs whole brain, and whether significance appears (`pop_up`) or disappears (`drop_out`). | no |
| `exception_architecture_subject.csv` | `build_exception_specificity.py` | The 16 exception-architecture features per participant, with group and ADNI phase. | yes |
| `exception_architecture_stats.csv` | `build_exception_specificity.py` | Group tests of those features, plus columns repeating the CN–AD and MCI–AD tests on ADNI-3 participants only. | no |
| `ml_targets.csv` | `build_simple_ml.py` | MMSE and CDR from the visit nearest the scan, days between visit and scan, group, age, sex, phase, and binary CDR (`cdr_bin`: 0 vs 0.5 or more). | yes |
| `ml_feature_matrix.csv` | `build_simple_ml.py` | The 103 features per participant. | yes |
| `ml_missingness.csv` | `build_simple_ml.py` | Share of missing values per feature and group. | no |
| `ml_ladder_results.csv` | `build_simple_ml.py` | Cross-validated scores for every target, ladder step and model (mean and SD over folds and seeds). | no |
| `ml_predictions.csv` | `build_simple_ml.py` | Out-of-fold predictions for the last ladder step (first seed). | yes |
| `ml_permutation.json` | `build_simple_ml.py` | Permutation-test p-value for the best model of the last ladder step, per target. | no |
| `ml_r2_summary.csv` | `build_model_interpretation.py` | Pooled out-of-fold R² (MMSE) or AUC (CDR) per ladder step, ExtraTrees only. | no |
| `ml_shap_values_mmse.csv`, `ml_shap_values_cdr_bin.csv` | `build_model_interpretation.py` | Top 25 features by average SHAP size, with feature family. | no |
| `ml_shap_beeswarm_mmse.csv`, `ml_shap_beeswarm_cdr_bin.csv` | `build_model_interpretation.py` | Per-person SHAP values for the top 12 features. | yes |
| `ml_ice_mmse.csv`, `ml_ice_cdr_bin.csv` | `build_model_interpretation.py` | PDP and ICE curves for the top 6 features. ICE rows are numbered people, not IDs. | numbered rows |
| `mmse_bucket_definition.csv` | `build_mmse_buckets.py` | The three MMSE bands: score range, size, and CN/MCI/AD mix. | no |
| `mmse_bucket_results.csv` | `build_mmse_buckets.py` | Macro and per-band AUC and PR-AUC, balanced accuracy, per ladder step and model. | no |
| `mmse_bucket_classwise.csv`, `mmse_bucket_curves.csv` | `build_mmse_buckets.py` | Per-band scores and ROC/PR curve points for the best model. | no |
| `mmse_bucket_shap_classwise.csv` | `build_mmse_buckets.py` | Top 15 features per band by SHAP. | no |
| `final_model_table.csv` | `build_final_model_metrics.py` | The final model's results as a ready-to-show table (one row per cell: target, row, column, text value). | no |
| `final_model_metrics.csv` | `build_final_model_metrics.py` | The same numbers in tidy form: AUC, PR-AUC, 95% interval and counts, per target, class and scope (All, CN, MCI, AD). `cls = -1` is the macro average. | no |
| `final_model_classwise.csv` | `build_final_model_metrics.py` | Per-class AUC, PR-AUC, chance level and class size. | no |
| `final_model_curves.csv` | `build_final_model_metrics.py` | ROC and PR curve points for both targets. | no |
| `ml_pdp_range_family.csv` | `build_pdp_range_family.py` | Predicted probability of each class as the DMN diffusivity composite moves, for the SR, LR, SR-exception and LR-exception versions, with % missing before filling. | no |
| `ml_feature_dictionary.csv` | `build_ml_explain_v2.py` | Every feature explained in words. | no |
| `ml_preprocessing_audit.csv` | `build_ml_explain_v2.py` | Per feature: % missing, skew, outliers, strongest correlation with another feature. | no |
| `ml_pipeline_steps.csv` | `build_ml_explain_v2.py` | The preprocessing steps in order, in words. Some numbers in this text are written into the script, not recomputed. | no |
| `ml_selection_experiment.csv` | `build_ml_explain_v2.py` | Scores with and without keeping only the 40 best features. | no |
| `ml_inference_summary.json` | `build_ml_explain_v2.py` | Top features per class and each feature family's share, from the class-by-group SHAP table as `build_ml_explain_v2.py` wrote it, plus three fixed text notes. | no |
| `cdr_curves.csv`, `cdr_classwise.csv` | `build_explain_v3.py` | ROC/PR curve points and per-class scores for CDR 0 vs 0.5 or more. | no |
| `ml_shap_overall_beeswarm.csv` | `build_shap_overall.py` | Per-person SHAP for the top 8 features of "impaired vs the rest" (lowest MMSE band; CDR 0.5 or more), with each feature's share of all 105 inputs. | yes |
| `ml_shap_class_group.csv` | `build_shap_overall.py` (first written by `build_ml_explain_v2.py`) | Top 12 features per target, class and group (All, CN, MCI, AD) by average SHAP size, with the share of all 105 inputs (`share_pct`) and its spread over seeds (`share_sd`). | no |
| `ml_shap_class_group_dir.csv` | `build_shap_overall.py` (first written by `build_ml_explain_v2.py`) | The average signed SHAP (direction) for the same cells. | no |
| `ml_shap_family_group.csv` | `build_shap_overall.py` | Each feature family's share per target, class and group. | no |
| `ml_shap_beeswarm_class.csv` | `build_shap_overall.py` (first written by `build_explain_v3.py`) | Per-person SHAP for the top 6 features of every target and class, with group and feature percentile. | yes |
| `ml_pdp_class_group.csv` | `build_shap_overall.py` (first written by `build_ml_explain_v2.py`) | Partial dependence of each class probability for the top 3 features, per group. | no |

In the final tables the MMSE target is `mmse_band` and the CDR target is
`cdr_3level` (three levels). Some earlier tables use `mmse` (the score as a
number) and `cdr_bin` (0 vs 0.5 or more) instead; the `target` column says
which.

### Figures (in `--figs`, default `docs/figs/`)

| File | Written by |
|---|---|
| `fig4_shap_mmse.png` | `build_model_interpretation.py`, then redrawn by `build_ml_explain_v2.py` |
| `fig5_ice_mmse.png` | `build_model_interpretation.py` |
| `fig6_mmse_buckets.png` | `build_mmse_buckets.py` |

### Run records (written by the runner, not by the scripts)

`exports/stage_status.csv` (one row per stage: status, seconds, error) and
`00_master/run_analysis_status.json`, both in the analysis root.

## Settings you can change

### Where things are read and written

| Setting | Where | What it does | Default |
|---|---|---|---|
| `--analysis-root <folder>` | any build script, or the runner | The analysis root to read inputs from and write results to. All sections (`00_`, `17_`, `19_`, `20_`) move together. | `$SC_ANALYSIS_ROOT`, which defaults to `$SC_DERIV_ROOT/qc/analysis_cohort` |
| `--out <folder>` | any build script (not the runner) | The one folder this script writes to. The ML scripts also **read** the earlier ML files (`ml_feature_matrix.csv`, `ml_targets.csv`, `exception_architecture_subject.csv`) from here, so use the same `--out` for every ML script in a chain. | ML scripts: `<analysis root>/20_exception_specificity`. `build_range_restricted_networks.py`: `<analysis root>/19_network_analysis/functional` |
| `--figs <folder>` | any build script | Where the PNG figures go. | `docs/figs` in the repository |
| `figs_dir: <folder>` | top level of the file given to the runner's `--config` | The runner passes this to the scripts as `--figs`. | `docs/figs` |
| `SC_ANALYSIS_ROOT` | environment variable | Same as `--analysis-root`, but for every command in your shell. Resolved in `sc_config.py`. | `$SC_DERIV_ROOT/qc/analysis_cohort` |
| `SC_DERIV_ROOT`, `SC_DATA_ROOT` | environment variables (the runner's `--deriv-root` sets `SC_DERIV_ROOT`) | Move the whole derivatives folder, and with it the analysis root and the connectome folder, unless those have their own variable set. | `$SC_DATA_ROOT/derivatives`; `data/` |
| `SC_CONNECTOMES_DIR` | environment variable, or runner `--connectomes-dir` | Where `build_range_restricted_networks.py` finds the matrices. | `$SC_DERIV_ROOT/connectomes` |
| `SC_COHORT_DIR` | environment variable, or runner `--cohort-dir` | Where `build_simple_ml.py` finds the cohort tables. | `cohort/` |
| `SC_EXCLUSIONS` | environment variable, or runner `--exclusions` | The exclusions file. | `configs/exclusions.local.yaml` |
| first word after `build_range_restricted_networks.py` | command line | A whole number: use only the first N participants of `master_cohort.csv`. `0` means all. It must come before any option. A small number (for example 25) is only a quick "does it run" test: the group tables can come out empty, because each group needs at least 5 participants. | all participants |

A note on `--out`: `build_exception_specificity.py` and `build_simple_ml.py`
read the network tables from `<analysis root>/19_network_analysis/functional/`,
not from `--out`. (`build_simple_ml.py` reads
`network_range_restricted_subject.csv` and
`network_exception_measures_subject.csv` from there.) If you send
`build_range_restricted_networks.py` somewhere else with `--out`, those two
scripts will not see its tables. To move a whole chain, use `--analysis-root`.

The nine scripts from `build_exception_specificity.py` on accept `--out`,
`--analysis-root` and `--figs` in any order and quietly ignore any other
option (which is why `--help` does not work). Two scripts behave differently:
`build_consensus_core.py` knows `--help` and `--verify`, and stops with an
error on an option it does not know; `build_range_restricted_networks.py`
needs a number as its first word whenever you give it options (see the last
row of the table above). Every script creates its output folder and figures
folder if they are missing, even when it then stops with an error.

### Model settings

The numbers that shape the models are written as named values near the top
of each script. To change one, edit the script. Doing so changes the results.

| Script | Name | What it controls | Value |
|---|---|---|---|
| all ML scripts | `SEEDS` | The five random seeds. | `[11, 23, 37, 51, 73]` |
| `build_simple_ml.py` | `N_PERM` | Shuffles in the permutation test. | `200` |
| `build_simple_ml.py` | `NETWORKS` | The ten networks used as features. | see [Words](#words-used-in-this-readme) |
| `build_consensus_core.py` | `CORE_MIN_PCT` | A connection is "core" when it is an exception in more than this percentage of participants. (`build_exception_specificity.py` applies the same 50% rule with its own copy of the number.) | `50.0` |
| `build_consensus_core.py` | `NETPAIR_MIN_PREV` | A network pair is reported when at least this share of participants has an exception in it. | `0.5` |
| `build_mmse_buckets.py` | `N_BUCKETS` | Number of MMSE bands. | `3` |
| `build_final_model_metrics.py` | `BOOT` | Bootstrap repeats for the 95% intervals. | `2000` |
| `build_final_model_metrics.py` | `MIN_PER_SIDE` | Below this many cases (or non-cases) an AUC is not reported. | `10` |
| `build_final_model_metrics.py` | `CURVE_POINTS` | About how many points each saved ROC or PR curve keeps. | `120` |
| `build_pdp_range_family.py` | `GRID`, `RANGES` | Points per partial-dependence curve; the four features compared. | `15`; the four `*_DMN_DIFF` features |
| `build_shap_overall.py` | `OVERALL_TOP`, `BREAKDOWN_TOP` | Features kept per target in the overall table, and per cell in the class-by-group tables. | `8`, `12` |

`configs/analysis.yaml` has blocks named `build_simple_ml`,
`build_final_model_metrics` and `build_shap_overall`. They record the final
model and these reporting values in words, so you can read them in one place.
The scripts do **not** read those blocks: changing them there changes nothing.
The exclusions file is set by `SC_EXCLUSIONS` or `--exclusions`, not by the
`exclude_subjects_file` key. See `configs/README.md`.

## If something goes wrong

**`[FAIL]  build_range_restricted_networks: RuntimeError: build_range_restricted_networks.py exited 1`**,
followed by `ValueError: invalid literal for int() with base 10: '--analysis-root'`

The runner cannot start this one script (see step 3). The stages that need
its tables then show `[BLOCK]` (only `build_consensus_core` still runs). Run
the script by hand, then run the runner again with `--resume`:

```bash
python hcp_analysis/build_range_restricted_networks.py
python -m connectome_analysis.run_analysis --group build --resume
```

If you use a different analysis root, put `0` first:
`python hcp_analysis/build_range_restricted_networks.py 0 --analysis-root <folder>`.
The same error appears whenever you give this script an option without a
number in front of it. `--all` runs the build group too, so it shows the same
failure; the same two commands fix it.

**`[BLOCK] build_...: 8 declared input(s) missing`**, followed by lines such as
`17_edr_exceptions/edr_exception_thresholds.csv  (from stage edr_exceptions)`

The earlier stages have not written their files into this analysis root. Run
them first (step 5 of [Before you start](#before-you-start)), or check that
`--analysis-root` / `SC_ANALYSIS_ROOT` points at the right folder
(`python hcp_analysis/sc_paths.py` shows it). A blocked stage also blocks
every stage after it.

**`FileNotFoundError: [Errno 2] No such file or directory: '.../20_exception_specificity/ml_feature_matrix.csv'`**

You ran an ML script before `build_simple_ml.py`, or with a different `--out`
or `--analysis-root` than `build_simple_ml.py` used. Run the scripts in the
order given in [Running one script by itself](#running-one-script-by-itself),
with the same options. The same kind of error from
`build_exception_specificity.py` naming
`19_network_analysis/functional/network_range_restricted_stats.csv` means
`build_range_restricted_networks.py` has not run in this analysis root.

**`NOTE (build_exception_specificity.py): no subject exclusions loaded from .../exclusions.local.yaml.`**
(also from `build_simple_ml.py`, and as `exclusions    : NONE` from the runner)

The exclusions file is missing, so the excluded participant is kept and the
ML numbers will not match the published ones. The run still works. If you
want the published numbers, get the real ID from the project author and
create the file (step 6 of [Before you start](#before-you-start)). Do not
silence this note with a placeholder ID: that hides the note but leaves
nobody out. With the right file in place, `build_exception_specificity.py`
prints `A2 subject features: 529 subjects x 16 features`; without it, or with
a placeholder, `530 subjects`.

**A script started working when you typed `--help`**

Only `build_consensus_core.py` and `sc_paths.py` have help. Press Ctrl+C. If
the script already wrote files, run the stages again in order.
(`build_range_restricted_networks.py --help` instead stops at once with
`ValueError: invalid literal for int() with base 10: '--help'` and writes no
result files; see the first entry above.)

**`another run holds the lock`** (the runner exits with status 75)

Another runner is already working on the same analysis root. Wait for it to
finish. The lock file is `logs/run_analysis.lock` inside the analysis root.

**The ML stages are very slow on a busy shared machine**

The gradient-boosting models (`hist_gb`) start one worker thread per
processor core. When other programs already keep the machine busy, those
threads wait for each other, and a fit that normally takes a fraction of a
second can take ten seconds or more. Limit them to one thread for the run:

```bash
export OMP_NUM_THREADS=1
python -m connectome_analysis.run_analysis --group build --resume
```

The runner passes the setting on to the scripts. It does not change the
results. On a machine that is not busy you normally do not need it.

**`build_consensus_core.py: error: unrecognized arguments: ...`**

This script checks its options strictly. It only knows `--help`, `--verify`,
`--out`, `--analysis-root` and `--figs`.

**`git status` shows `docs/figs/*.png` as modified**

The build stages redrew the figures (step 4), and your pictures differ from
the committed ones because something in your data, exclusions or package
versions differs. If you did not want the committed pictures replaced, put
them back with `git restore docs/figs/`, and use `--figs` or `figs_dir:` next
time.

**The numbers differ slightly from the published ones**

Check, in this order: the exclusions file (see above); the package versions
(`python sc_doctor.py --analysis` lists them; they must match
`requirements/analysis.txt`); and that `build_shap_overall` ran last.

**`MISS  <package>`** from `sc_doctor.py`

The venv is not switched on, or the install did not finish. Run
`source .venv/bin/activate` and `pip install -r requirements.txt` again.

## The `improve/` folder

`improve/` is a record of the **improvement sweep**: a set of experiments
that tried to beat the first model, an ExtraTrees model on the full feature
ladder. Each experiment is a "lever". Each lever has a matching `verify_*`
script, written separately, that re-runs the lever's key comparisons from
scratch to check the numbers it reported. The final model keeps ExtraTrees
and adds lever 4's residualisation of the features on age and sex inside each
training fold (see [The final model](#the-final-model)).

| File | What it tries |
|---|---|
| `lever1_histgb.py` | Six settings of HistGradientBoosting, compared with ExtraTrees. |
| `lever2_model_classes.py` | Other model types: support-vector machines, k-nearest neighbours, and ridge or logistic regression after a quantile transform. |
| `lever3_ensembling.py` | Combining ExtraTrees, HistGradientBoosting and a linear model: plain average, rank average, and stacking. |
| `lever3b_weighted_blend.py` | Averages that give ExtraTrees a fixed weight of 0.5 or 0.7. Saves every out-of-fold prediction to a file, `lever3_oof.npz` (where: see below). |
| `lever4_representation.py` | Changing the inputs: quantile transform, 20 principal components, and residualising every feature on age and sex. |
| `lever5_target_formulation.py` | Changing the target: MMSE as ranks, MMSE bands predicted as ordered classes, and calibrated CDR probabilities. |
| `verify_gb-tuning.py` | Re-checks lever 1. |
| `verify_new-models.py` | Re-checks lever 2. |
| `verify_ensembling.py` | Re-checks levers 3 and 3b. |
| `verify_feature-transforms.py` | Re-checks lever 4. |
| `verify_target-formulation.py` | Re-checks lever 5. |

They share one protocol: age and sex added as inputs (sex 1 = female), 5-fold
stratified cross-validation repeated for the five seeds, every preprocessing
step learned inside the training fold only, and scores computed on the pooled
out-of-fold predictions of each seed, then averaged. The targets are mainly
MMSE (as a number) and CDR 0 vs 0.5 or more; lever 5 also tries the three
MMSE bands. The scripts only print their results; apart from
`lever3b_weighted_blend.py`, they write no files.

**They are not part of the pipeline.** The runner never starts them, and they
do not use `sc_paths.py`. Each one opens `ml_feature_matrix.csv` and
`ml_targets.csv` from a fixed folder path (a full path, starting with `/`)
written near the top of the file. It points at old copies in `hcp_analysis/`
on the machine where the scripts were written. On any other machine that
path does not exist, and the script stops with `FileNotFoundError`.

The path is written in a different way in different scripts: twice, once per
`pd.read_csv(...)` line, in seven of them; once, in a variable, in the other
four (`BASE` in `lever2_model_classes.py` and `verify_new-models.py`, `D` in
`verify_target-formulation.py`, `OUT` in `lever5_target_formulation.py`).
`lever3b_weighted_blend.py` also **saves** its results to a fixed path
ending in `/hcp_analysis/improve/lever3_oof.npz`, so it would fail at the very
end even after its read paths were fixed.

Rather than edit the scripts in the repository (git would then show them as
changed), make fixed copies outside it. From the repository root, with the
venv switched on:

```bash
mkdir -p ../improve_runs
WORK="$(cd ../improve_runs && pwd)"
ML="$(python -c 'import sc_config; print(sc_config.paths().exception_dir)')"
for f in hcp_analysis/improve/*.py; do
  sed -E -e "s#(['\"])[^'\"]*/hcp_analysis/improve/#\1$WORK/#" \
         -e "s#(['\"])[^'\"]*/hcp_analysis(/ml_|['\"])#\1$ML\2#" \
         "$f" > "$WORK/$(basename "$f")"
done
```

What each line does:

1. `mkdir -p ../improve_runs` makes a folder next to the repository (`..` is
   the folder above it) for the copies.
2. `WORK=...` stores the full path of that folder under the name `WORK`.
3. `ML=...` asks `sc_config.py` for your `<analysis root>/20_exception_specificity`
   folder and stores it under the name `ML`. If you use another analysis
   root, write `SC_ANALYSIS_ROOT=<folder>` in front of `python` on this line.
4. The `for` loop copies each script into `../improve_runs/`. On the way,
   `sed` (a tool that replaces text) swaps the old save folder of
   `lever3b_weighted_blend.py` for `WORK`, and every old read folder for `ML`.
   The scripts in the repository are not changed.

`lever3b_weighted_blend.py` then saves `lever3_oof.npz` in `../improve_runs/`.
`ml_feature_matrix.csv` and `ml_targets.csv` must already exist in the `ML`
folder, so run `build_simple_ml` first (step 4). Several of these scripts
fit gradient-boosting models, so on a busy machine type
`export OMP_NUM_THREADS=1` first (see "The ML stages are very slow" under
[If something goes wrong](#if-something-goes-wrong)). Then run a copy, for
example:

```bash
python ../improve_runs/verify_gb-tuning.py
```

What you should see (about a minute; your times will differ):

```
n=202  n_features=103 (+2 covariates)
VERIFY BASELINE ET MMSE: R2=0.1698 rho=0.3224  per-seed R2=['0.177', '0.160', '0.166', '0.181', '0.166']  [17s]
VERIFY BASELINE ET CDR : AUC=0.6018  per-seed=['0.590', '0.593', '0.616', '0.602', '0.608']  [31s]
VERIFY hgb1 MMSE: R2=0.1604 rho=0.3202  per-seed R2=['0.200', '0.140', '0.139', '0.213', '0.110']  [41s]
VERIFY hgb5 CDR : AUC=0.5502  per-seed=['0.517', '0.557', '0.622', '0.530', '0.524']  [60s]

=== VERDICT NUMBERS ===
MMSE: baseline R2=0.1698/rho=0.3224  vs hgb1 R2=0.1604/rho=0.3202  (claimed 0.1698/0.3224 vs 0.1604/0.3202)
CDR : baseline AUC=0.6018  vs hgb5 AUC=0.5502  (claimed 0.6018 vs 0.5502)
total 60s
```

The "claimed" numbers in brackets are the ones the lever script reported;
the verify script checks that it gets the same.

Four small tables that the dashboard can show have no script in this
repository that writes them: `improvement_mmse.csv` and
`improvement_final_check.csv` (the first model against the age-and-sex
residualised model, from this sweep), `cdr_threeclass.csv` (AUC for each of
the three CDR levels) and `ml_auc_class_diaggroup.csv` (AUC for each class
within each diagnostic group). The runner does not remake them. The dashboard
shows them when they are present in `20_exception_specificity/`.
