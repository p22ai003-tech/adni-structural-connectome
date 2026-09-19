# configs/: settings files

## What is this folder?

This folder holds the **settings files** that the programs read to decide how to
run. Most of them are **YAML** files. YAML is a plain-text format made of
`key: value` lines, where the indentation shows what belongs to what. Indent
with **spaces, never tabs**. The rest are **CSV** files, which are tables saved
as text with commas between the columns.

## Do I need it?

- **Running the analysis:** the analysis runner
  (`python -m connectome_analysis.run_analysis`) reads `analysis.yaml`
  automatically, and you do not have to change anything for a first run. The
  runner works in **stages**. A stage is one step of the analysis, such as
  `demographics`, and `python -m connectome_analysis.run_analysis --list` shows
  them all. To reproduce the published machine-learning numbers exactly, you
  also create one private file,
  [`exclusions.local.yaml`](#exclusionslocalyaml-private-never-committed).
- **Building the cohort tables:** `sc_cohort.py build` reads `cohort_rules/`
  automatically.
- **Imaging:** `connectome_v2.yaml` belongs to the imaging half. See the Imaging
  section of the top-level [README.md](../README.md).

## What's inside

| File | What it is | In git? |
|---|---|---|
| `analysis.yaml` | Statistical settings for the analysis runner (`python -m connectome_analysis.run_analysis`). | yes |
| `cohort_rules/Ranked_OK_T1.csv` | The 299 T1 scan descriptions that count as acceptable, each with a rank from 1 (best) to 6. `sc_cohort.py build` uses it to pick one T1 scan per participant. | yes |
| `cohort_rules/adni_t1_coreg_whitelist.csv` | The 13 human-readable rules (text patterns) that the ranking comes from. No program reads it; it documents where the ranks come from. | yes |
| `connectome_v2.yaml` | Settings and version requirements for the imaging workflow. | yes |
| `connectome_v2_retry3.yaml` | A copy of `connectome_v2.yaml` used for one earlier imaging re-run. It differs only in two file paths under `locked_paths:`: `workflow_source_manifest` (the list of workflow source files) and `environment_contract` (the required software versions). | yes |
| `exclusions.local.yaml` | The list of participants to leave out of the machine-learning stages. It names ADNI participants, so it is private. **You create it yourself.** | **no, never** |
| `acquisition_manifest*.csv` | Scan lists written by the imaging tools. Each row names a participant. | **no, never** |

## Before you start

1. Install the project as the top-level [README.md](../README.md) describes. It
   creates a **venv**, which is a private Python installation just for this
   project.
2. Open a terminal in the repository root (the folder that holds `sc_config.py`)
   and switch the venv on:

   ```bash
   source .venv/bin/activate
   ```

   Every command on this page is run from the repository root. On this page,
   `<repo>` stands for that folder, the one you cloned.
3. Data locations are **not** set in this folder. They come from **environment
   variables** (named settings that programs started from your terminal can
   read). `sc_config.py` reads the first five, and `sc_exclusions.py` reads
   `SC_EXCLUSIONS`:

   | Variable | Default | Runner flag |
   |---|---|---|
   | `SC_DATA_ROOT` | `<repo>/data` | none |
   | `SC_DERIV_ROOT` | `$SC_DATA_ROOT/derivatives` | `--deriv-root` |
   | `SC_CONNECTOMES_DIR` | `$SC_DERIV_ROOT/connectomes` | `--connectomes-dir` |
   | `SC_ANALYSIS_ROOT` | `$SC_DERIV_ROOT/qc/analysis_cohort` | `--analysis-root` |
   | `SC_COHORT_DIR` | `<repo>/cohort` | `--cohort-dir` |
   | `SC_EXCLUSIONS` | `<repo>/configs/exclusions.local.yaml` | `--exclusions` |

   A runner flag sets the matching variable for that one run. To see which
   folders the first five resolve to on your machine, run
   `python -c "import sc_config; print(sc_config.describe())"`. It also prints
   a `Toolchains` block (`FSLDIR`, `MRTRIX_BIN`, `ANTSPATH`). Those say where the
   imaging programs are installed, and you can ignore them for the analysis. It does not show
   `SC_EXCLUSIONS`. To see which exclusions file will be used, look at the
   `exclusions :` line that the analysis runner prints at the start of every
   run, or at the `exclusions` line of `python sc_doctor.py --analysis`.
4. ADNI data (images, cohort tables and anything naming a participant) come
   only from [adni.loni.usc.edu](https://adni.loni.usc.edu/), under your own
   Data Use Agreement. They must never be committed to git.

---

## `analysis.yaml`: settings for the analysis

### How the file is read

- The analysis runner reads `configs/analysis.yaml` every time it runs. To use a
  different file, pass `--config <file>`.
- The file has two sections:
  - `defaults:` holds values that apply to any stage asking for that key.
  - `stages:` holds values for one stage only. The stage names are the ones
    `python -m connectome_analysis.run_analysis --list` prints.
- For each key, the runner uses the first value it finds: first
  `stages.<stage>.<key>`, then `defaults.<key>`, then the value written in the
  code. A file that mentions only the one key you want to change is therefore
  fine.
- For every key the runner reads, the value in the file today is the same as
  the value in the code. As shipped, the file changes nothing.
- A key under a misspelled stage name is **silently ignored**. Copy stage names
  from `--list`.
- No paths go in this file (see "Before you start").

### Keys the runner reads (changing these changes a run)

| Key | Default | What it does |
|---|---|---|
| `defaults.edge_perms` | `200` | How many random shuffles (**permutations**) the edge-by-edge group test in the `edgewise_fd_sum` stage uses. More shuffles take proportionally longer. You can also set it for that stage alone as `stages.edgewise_fd_sum.edge_perms`. |
| `defaults.brain_age_repeats` | `3` | How many times the `live_brain_age` stage repeats its 5-fold cross-validation (cross-validation = train on part of the data, test on the rest, and rotate). You can also set it as `stages.live_brain_age.brain_age_repeats`. |
| `stages.analysis_snapshot.snapshot_mode` | `provisional` | A label written into `00_master/analysis_snapshot.csv`. Use `provisional` for a working run and `final` for the frozen run behind a report. It does not change any calculation. |
| `stages.data_completeness.strict_tracks_count` | `false` | With `false`, a participant's tractogram file (the file of reconstructed fibre tracks) only has to exist to count as present. With `true`, it must also hold at least 3,000,000 **streamlines** (one streamline is one reconstructed track), as counted by `tckinfo` from MRtrix3 (the diffusion-MRI software the imaging half uses). If `tckinfo` is not installed, the count check is skipped. |
| `stages.clinical_outcome_search.outcome_search_fast` | `true` | With `true`, the search runs on a fixed shortlist of feature sets with smaller models. This is what the published search used. With `false`, it runs the full search, which takes longer. |
| `stages.clinical_outcome_search.outcome_search_jobs` | `4` | How many worker processes that search runs at once. |
| `stages.literature_retrieval.literature_mailto` | `""` | Your e-mail address. It is sent to OpenAlex (a free online database of research papers) as a contact, which OpenAlex asks for. Only the optional `literature_retrieval` stage needs it, and `--all` does not run that stage. |
| `figs_dir` (top level; not in the file) | `docs/figs` | The folder where the `build_*` stages save their figures. To change it, add a line such as `figs_dir: my_figures` at the very left of your copy of the file, not indented and not under `defaults:` or `stages:`. A relative path is taken from the repository root. |

### Keys kept as a written record (changing these has no effect today)

These keys document the values the published results used. The runner does
**not** read them: each value is written directly into the analysis code. To
change one, you would have to change the code, and doing so changes published
numbers.

| Key | Value | Meaning | Where the value actually lives |
|---|---|---|---|
| `defaults.seeds` | `[11, 23, 37, 51, 73]` | Starting numbers (**seeds**) for the random-number generator. The machine-learning models are fitted once per seed and the results are averaged. | `SEEDS` in the `hcp_analysis/build_*.py` scripts |
| `defaults.fdr_alpha` | `0.05` | Significance level for the FDR (false discovery rate) correction, which limits false positives when many tests are run. | in the analysis modules |
| `defaults.holm_alpha` | `0.05` | Significance level for the Holm correction, another multiple-testing correction. | in the analysis modules |
| `stages.lr_sr.percentiles` | `[33.333, 66.667]` | Cut points that split connection lengths into short, medium and long thirds. | `connectome_analysis/analysis_length_delay.py` |
| `stages.edr_exceptions.threshold_source_group` | `CN` | The `edr_exceptions` stage looks for EDR exceptions. EDR, the exponential distance rule, says that longer connections are weaker, and an exception is a connection much stronger than its length predicts. Its length cut points are taken from the CN group only. | `connectome_analysis/analysis_edr_exceptions.py` |
| `stages.edr_exceptions.percentiles` | `[25, 75]` | SR (short range) = the shortest quarter of connections, at or below Q1 = 78.6856 mm (Q1, the first quartile, is the length a quarter of connections are shorter than). LR (long range) = the longest quarter, above Q3 = 171.021 mm (Q3, the third quartile, is the length three quarters are shorter than). | same file |
| `stages.edr_exceptions.exception_sd` | `3.0` | An edge is an "exception" when its weight is more than 3 standard deviations above the mean of its length bin, within the same participant. | same file |
| `stages.edr_exceptions.min_bin_edges` | `120` | The smallest number of edges a length bin may hold. | same file |
| `stages.edr_exceptions.weight_type` / `length_type` | `fd_sum` / `len_mean` | Which matrix types give the edge weight and the edge length. | same file |
| `stages.edgewise_fd_sum.connectome_type` | `fd_sum` | The matrix type tested edge by edge. | `connectome_analysis/analysis_edges.py` |
| `stages.build_simple_ml.n_estimators` | `400` | Number of trees in the ExtraTrees model (extremely randomised trees: a machine-learning model made of many decision trees that vote). | `hcp_analysis/build_*.py` |
| `stages.build_simple_ml.min_samples_leaf` | `5` | The smallest number of participants allowed at the end of a tree branch. | same |
| `stages.build_simple_ml.class_weight` | `balanced` | Groups of different sizes count equally during training. | same |
| `stages.build_simple_ml.n_splits` | `5` | 5-fold cross-validation. | same |
| `stages.build_simple_ml.targets` | `[mmse_band, cdr_3level]` | What the models predict: an MMSE band (low / mid / high) and a three-level CDR (0 / 0.5 / ≥1). MMSE (Mini-Mental State Examination) is a 30-point test of memory and thinking. CDR (Clinical Dementia Rating) rates dementia from 0 (none) upward. | same |
| `stages.build_simple_ml.exclude_subjects_file` | `configs/exclusions.local.yaml` | Records where the exclusion list lives. The list is actually found through `SC_EXCLUSIONS` or `--exclusions` ([below](#exclusionslocalyaml-private-never-committed)). | `sc_exclusions.py` |
| `stages.build_final_model_metrics.bootstrap` | `2000` | Number of **resamples** used for the confidence intervals. A resample is a new list of the same size, drawn at random from the cases with repeats allowed (the cases with the outcome and the cases without it are each redrawn); the score is recomputed on each one. The score here is the AUC (area under the ROC curve), where 1.0 means perfect and 0.5 means no better than guessing. A **confidence interval** is the range that holds the middle 95% of those 2000 scores, which shows how precise the score is. | `hcp_analysis/build_final_model_metrics.py` |
| `stages.build_final_model_metrics.min_per_side` | `10` | If there are fewer than this many cases on either side (fewer than 10 people with the outcome, or fewer than 10 without it), neither the score nor an interval is reported. | same |
| `stages.build_final_model_metrics.curve_points` | `120` | Points per plotted curve. | same |
| `stages.build_shap_overall.top_k` | `null` | SHAP shares (how much each feature contributes to a prediction) are computed over **all** features, not just the top few. | `hcp_analysis/build_shap_overall.py` |

The comments in `analysis.yaml` explain that two different length splits exist
on purpose: `lr_sr` uses thirds and `edr_exceptions` uses CN quarters. Neither
reads the other.

### Rules that are not in this file

- **Diagnostic groups.** EMCI, LMCI and SMC are grouped with MCI, and only CN,
  MCI and AD are analysed. This is the project's grouping rule. It is set in the
  code, not here: `recode_group` in `connectome_analysis/analysis_cohort.py`
  does the regrouping, and the same file then keeps only the groups listed in
  `GROUP_ORDER` (`CN`, `MCI`, `AD`), which is defined in
  `connectome_analysis/analysis_config.py`.
- **Paths**, which come from the `SC_*` variables and flags.
- **Which participants to leave out**, which comes from
  `exclusions.local.yaml`.

### How to change a setting, step by step

**Step 1. See which settings file a run will use**, without running anything.
The last part sends the run's bookkeeping files to a throw-away temporary
folder:

```bash
python -m connectome_analysis.run_analysis --all --dry-run --analysis-root "$(mktemp -d)"
```

What you should see on a fresh copy of the repository (your paths will
differ):

```
connectomes   : <repo>/data/derivatives/connectomes
cohort tables : <repo>/cohort
exclusions    : NONE (<repo>/configs/exclusions.local.yaml not found)
                The published ML results exclude one subject. Without the
                exclusions file, the ML stages will not reproduce them.
analysis root : <a temporary folder>
config        : <repo>/configs/analysis.yaml
plan          : 36 stage(s) [dry-run]

  [dry]   analysis_snapshot  -> 1 output(s)
  [dry]   master_cohort  -> 1 output(s)
  ...
dry-run=36   total 0.4s
ledger: <a temporary folder>/exports/stage_status.csv
```

The last line names the **ledger**, a small table the runner keeps of each
stage's status (here, `dry-run` for all 36). The `exclusions : NONE` lines are
normal until you create the private
exclusions file ([below](#exclusionslocalyaml-private-never-committed)). Once
it exists, that line reads `exclusions : 1 subject(s) from ...` instead. The
line to look at here is `config`.

**Step 2. Make your own copy.** Don't edit `analysis.yaml` itself. Copy it to a
name ending in `.local.yaml`, which git ignores. This command copies only if you
do not have a copy yet, so it never overwrites earlier edits:

```bash
if [ -e configs/analysis.local.yaml ]; then echo "you already have a copy - edit that one"; else cp configs/analysis.yaml configs/analysis.local.yaml; fi
```

**Step 3. Edit the copy.** As a harmless test, change
`snapshot_mode: provisional` to `snapshot_mode: final`. One way is **nano**, a
simple text editor that runs inside the terminal:

```bash
nano configs/analysis.local.yaml
```

1. Press Ctrl+W, type `snapshot_mode` and press Enter. The cursor jumps to the
   start of the line `snapshot_mode: provisional`, under `analysis_snapshot:`.
   (Search for `snapshot_mode`, not `provisional`: the word `provisional` also
   appears in the comment on the line above.) nano does not show line numbers,
   but you can press Ctrl+C to check where you are. It prints `line  37/136` at
   the bottom.
2. Press the End key to jump to the end of that line. Press Backspace 11 times
   to delete `provisional`, then type `final`. Do not change the spaces in front
   of `snapshot_mode:`.
3. Press Ctrl+O and then Enter to save, and Ctrl+X to leave nano.

On Linux, this one line does the same change without opening an editor:

```bash
sed -i 's/snapshot_mode: provisional/snapshot_mode: final/' configs/analysis.local.yaml
```

**Step 4. Try the copy on one quick stage, in a practice folder:**

```bash
TRY=$(mktemp -d)
python -m connectome_analysis.run_analysis --stage analysis_snapshot --config configs/analysis.local.yaml --analysis-root "$TRY"
cat "$TRY/00_master/analysis_snapshot.csv"
```

What you should see (the first lines, which list the folders, are left out
here):

```
...
config        : configs/analysis.local.yaml
plan          : 1 stage(s)

  [run]   analysis_snapshot ...
  [ok]    analysis_snapshot  (0.3s)

ok=1   total 0.3s
...
snapshot_mode,created_utc
final,2026-09-18T15:26:42Z
```

This stage needs no data, so it works on a fresh copy of the repository.

**Step 5. Use the copy for real runs** by adding
`--config configs/analysis.local.yaml` to your usual run command. See
[`connectome_analysis/README.md`](../connectome_analysis/README.md).

**`--resume` does not notice a settings change.** With `--resume`, the runner
skips a stage whose output files already exist and are newer than its inputs.
It does not look at the settings file. So after you change a key, re-run the
affected stage **without** `--resume`:

```bash
python -m connectome_analysis.run_analysis --stage <stage> --config configs/analysis.local.yaml
```

Replace `<stage>` with a stage name from `--list`, for example
`edgewise_fd_sum`. To also redo every stage that uses that stage's results, use `--from <stage>`
in place of `--stage <stage>`. Add `--skip literature_retrieval` unless you have
set `literature_mailto` (see the table above), because `--from` also includes
that optional stage.

---

## `cohort_rules/`: how one T1 scan is chosen per participant

A **T1** is the structural brain scan. ADNI often has several T1 scans for each
participant, each with a text **description** such as
`MT1; GradWarp; N3m <- MPRAGE`. In these descriptions, MPRAGE, MP-RAGE and
IR-FSPGR are names of common T1 scan types, and GradWarp and N3 are corrections
ADNI applied to the image (for scanner distortion and for uneven brightness).
A FLAIR is a different kind of scan, not a T1. When `python sc_cohort.py build`
writes `mri.csv`, it chooses one T1 per participant as follows:

1. It keeps rows of `all_mri.csv` whose participant is in `dti.csv` (the list of
   participants with a diffusion scan; DTI, diffusion tensor imaging, is ADNI's
   name for the diffusion scan) and whose `Description` appears **exactly** in
   `Ranked_OK_T1.csv`. Spaces, capital letters and the `<-` arrow must all
   match. For example, `ADNI       MPRAGE` has seven spaces.
2. For each participant, it keeps the row with the lowest rank (1 = best). If
   two rows tie, it keeps the one that comes first in the file.

If you do not supply `all_mri.csv`, `mri.csv` is written with a header only.
The analysis reads only the `Subject ID` column of `mri.csv`, so the results
come out the same.

### The two files

`Ranked_OK_T1.csv` has two columns, `description` and `rank`:

| rank | how many descriptions | kind of scan (from the rules file) |
|---|---|---|
| 1 | 46 | 3-D T1 with GradWarp and N3 correction, masked |
| 2 | 33 | 3-D T1 with N3 correction, masked |
| 3 | 43 | MPRAGE with GradWarp and N3, not masked |
| 4 | 32 | MPRAGE with N3, not masked |
| 5 | 132 | plain MPRAGE / IR-FSPGR / MP-RAGE, as acquired |
| 6 | 13 | site-specific or acquisition-variant MPRAGE names |

The words in the last column are the rules file's own. Ranks 1 and 2 are the
descriptions that start with `MT1;` (for example `MT1; GradWarp; N3m <- MPRAGE`),
which the rules file calls **masked**. For the unmasked ones (ranks 3 and 4) it
notes that the skull has to be removed separately, so *masked* here means that
the parts outside the brain are already cut away.

`adni_t1_coreg_whitelist.csv` has columns `pattern` (a text pattern, written as
a **regular expression**, a compact code for describing text: for example `^`
means "at the start", `.*` means "any text" and `(?i)` means "ignore capital
letters"), `priority` (1 to 8), `examples` and `why_ok`. Each
description's rank is the smallest priority among the patterns it matches. You
can check that the two files agree. Paste the whole block, from `python` down
to the final `EOF`, into the terminal:

```bash
python - <<'EOF'
import re
import pandas as pd

ranked = pd.read_csv("configs/cohort_rules/Ranked_OK_T1.csv")
rules = pd.read_csv("configs/cohort_rules/adni_t1_coreg_whitelist.csv")
patterns = [(re.compile(p), int(pr)) for p, pr in zip(rules["pattern"], rules["priority"])]

bad = [d for d, r in zip(ranked["description"], ranked["rank"])
       if min((pr for p, pr in patterns if p.search(d)), default=None) != r]
print(len(ranked), "descriptions,", len(bad), "disagree with the patterns")
print("descriptions per rank:", ranked["rank"].value_counts().sort_index().to_dict())
EOF
```

What you should see:

```
299 descriptions, 0 disagree with the patterns
descriptions per rank: {1: 46, 2: 33, 3: 43, 4: 32, 5: 132, 6: 13}
```

### Watch the rule work on made-up data

This block uses invented participants (`XXX_S_0001`, `XXX_S_0002`) in a
temporary folder, so it needs no ADNI data. Participant 1 has two T1 scans:
`MPRAGE` (rank 5) and `MT1; GradWarp; N3m <- MPRAGE` (rank 1). Participant 2
has only a FLAIR scan, which is not a T1. Paste all the lines at once:

```bash
DEMO=$(mktemp -d)
printf "%s\n" \
  "Subject ID,Phase,Sex,Research Group,Visit,Study Date,Age,Image ID,Description,Type,MMSE Total Score,Global CDR" \
  "XXX_S_0001,ADNI 2,F,CN,sc,01/15/2012,72,100001,Axial DTI,Original,29,0" \
  "XXX_S_0002,ADNI 3,M,AD,sc,03/10/2018,80,100003,Axial DTI,Original,21,1" > "$DEMO/dti_master.csv"
cp "$DEMO/dti_master.csv" "$DEMO/mri_master.csv"
printf "%s\n" \
  "Subject ID,Study Date,Image ID,Description" \
  "XXX_S_0001,01/15/2012,200001,MPRAGE" \
  "XXX_S_0001,01/15/2012,200002,MT1; GradWarp; N3m <- MPRAGE" \
  "XXX_S_0002,03/10/2018,200003,Axial FLAIR" > "$DEMO/all_mri.csv"
python sc_cohort.py build --exports "$DEMO" --out "$DEMO/cohort"
cut -d, -f2,4,5 "$DEMO/cohort/mri.csv"
```

The last line uses `cut` to show only columns 2, 4 and 5 of `mri.csv`
(`Subject ID`, `Image ID` and `Description`). The file's first column has no
name and holds a row number, and its last column is the `rank`.

What you should see:

```
dti.csv              2 subjects  (from 2 DTI rows)
mri.csv              1 subjects  (ranked T1 choice)
                1 DTI subject(s) have no acceptable T1 description
dti_master.csv  copied
mri_master.csv  copied
...
Subject ID,Image ID,Description
XXX_S_0001,200002,MT1; GradWarp; N3m <- MPRAGE
```

The rank-1 scan wins, and participant 2 is reported as having no acceptable T1.
For building the real cohort tables from your ADNI downloads, run
`python sc_cohort.py --help` and see
[Cohort tables](../connectome_analysis/README.md#cohort-tables) in
`connectome_analysis/README.md`.

### Changing the ranking

Edit `Ranked_OK_T1.csv` by adding, removing or re-ranking lines, then run
`python sc_cohort.py build` again. Keep `adni_t1_coreg_whitelist.csv` in step
with it, and re-run the check above. A different ranking can pick different T1
scans, which moves the published cohort tables.

---

## `exclusions.local.yaml`: private, never committed

### What it is

It lists the participants to leave out of the machine-learning stages
(`build_simple_ml` and `build_exception_specificity`, and every later stage
that reads their outputs). The published results leave out **one** participant,
whom the project's outlier audit flagged because their mean diffusivity (a
measure of how freely water moves in the tissue) is about ten times normal. At
that level the signal comes from cerebrospinal fluid, not brain tissue.

The file **is not in the repository and must never be**. It holds ADNI
participant identifiers, which the ADNI Data Use Agreement restricts. Git
already ignores it through the `.gitignore` rule `*.local.yaml`.

The identifier is not published. Any request for it must be made privately to
the project author, under your own ADNI Data Use Agreement; see the top-level
[README.md](../README.md) for how to contact the author. Never ask for a
participant identifier, or post one, in a public issue or discussion.

### Exact format

```yaml
exclude_subjects:
  - XXX_S_0001          # replace with the real Subject ID; the text after # is a comment
```

- The top key must be spelled exactly `exclude_subjects`.
- Put one participant per line, starting with two spaces, a dash and a space.
  Every `- ` must start in exactly the same column. A dash indented further than
  the line above it is **not** an error: YAML quietly joins the two lines into
  one entry (`XXX_S_0001 - XXX_S_0002`), which matches nobody.
- Use the ADNI **Subject ID** only (the `NNN_S_NNNN` form), with no image ID.
- An optional second section, `subject_lists:`, holds named lists that the
  imaging tools use as default participant sets (`v2_default_subjects`,
  `spatial_contract_panel`, `h04a_r1_expected_units`, and
  `recovery2_diverse_units`, which only a test reads), written as
  `<SUBJECT>_I<IMAGEID>`. The analysis never reads it. If it is missing, the
  imaging tools simply have no default set.

### How to create it

Do this only once you have the real Subject ID. A placeholder such as
`XXX_S_0001` would be counted as an exclusion while leaving nobody out.

1. Open the file in **nano**, a simple text editor that runs inside the
   terminal. If the file already exists, nano opens it with its contents, so
   nothing is lost:

   ```bash
   nano configs/exclusions.local.yaml
   ```

2. If the file is empty, type these two lines, with the real Subject ID in place
   of `XXX_S_0001`. If it already has an `exclude_subjects:` line, add only a new
   `  - ` line under the ones already there.

   ```yaml
   exclude_subjects:
     - XXX_S_0001
   ```

3. Press Ctrl+O and then Enter to save, and Ctrl+X to leave nano.
4. Check the file:

   ```bash
   python sc_doctor.py --analysis | grep exclusions
   git check-ignore -v configs/exclusions.local.yaml
   ```

What you should see: `sc_doctor` prints this line. You may also see two lines
of `DeprecationWarning` about `jsonschema`, which are harmless.

```
  ok    exclusions                 1 subject(s) from <repo>/configs/exclusions.local.yaml
```

The number must equal the number of people you listed. `sc_doctor` only counts
the entries. It does not check that they are real Subject IDs, so a typo or a
placeholder still shows as `ok`.

Without `| grep exclusions`, `sc_doctor` prints many more lines. On a fresh
copy of the repository, lines saying `MISS` for `data_root`, `dti.csv`,
`mri.csv`, `dti_master.csv` and `mri_master.csv` are normal until you have your
data and have built the cohort tables. Because of those five, it also ends with
`5 problem(s). The pipeline will not complete until these are fixed.` That is
expected on a fresh copy. Only the exclusions line matters here.

`git check-ignore` confirms that git will not pick the file up (the line number
may differ):

```
.gitignore:167:*.local.yaml	configs/exclusions.local.yaml
```

To keep the file somewhere else, set `SC_EXCLUSIONS=/path/to/file.yaml`, or pass
`--exclusions /path/to/file.yaml` to the analysis runner.

### What happens without it

Everything still runs. No participant is left out, so the machine-learning
numbers will not match the published ones exactly. The runner says so at the
start of every run:

```
exclusions    : NONE (<repo>/configs/exclusions.local.yaml not found)
                The published ML results exclude one subject. Without the
                exclusions file, the ML stages will not reproduce them.
```

`python sc_doctor.py --analysis` shows it as a warning (`warn`), not as a
problem:

```
  warn  exclusions                 <repo>/configs/exclusions.local.yaml absent; the ML stages will not reproduce the published numbers
```

---

## `connectome_v2.yaml` and `connectome_v2_retry3.yaml`: imaging only

These files hold the settings and version requirements for the imaging workflow,
the half of the project that turns raw scans into matrices. That half is being
rewritten; see the Imaging section of the top-level [README.md](../README.md).
The analysis never reads these files. If you open them, you will find absolute
paths from the machine they were written on. That is expected while the imaging
half is being reworked. Their `atlas:` section records SHA-256 checksums of the
atlas files, which [`atlas/README.md`](../atlas/README.md) shows you how to
check.

## Acquisition manifests: generated, never committed

An **acquisition manifest** is a CSV list of scans, one row per pair of a
diffusion scan and its T1 scan.
Each row names a participant and gives the full path to their images. The
imaging tools write it on the machine that holds the data. Its default location
is `configs/acquisition_manifest.csv`, and the `SC_MANIFEST` variable changes
it. Because it names participants, it is **never committed**. `.gitignore`
blocks `configs/acquisition_manifest*.csv`, `*acquisition_manifest*.csv` and
`*_manifest_v2*.csv`. You can check this:

```bash
git check-ignore -v configs/acquisition_manifest.csv
```

```
.gitignore:53:*acquisition_manifest*.csv	configs/acquisition_manifest.csv
```

The analysis does not use it. If you only run the analysis, check your machine
with `python sc_doctor.py --analysis`. Under "Paths" it prints
`warn  manifest ... (absent; created on demand)`, which is normal.

Plain `python sc_doctor.py`, without `--analysis`, also checks the imaging
tools (MRtrix3, FSL, ANTs and dcm2niix). On a machine without them it prints a
`MISS` line for each one and ends with `N problem(s). The pipeline will not
complete until these are fixed.` That is expected if you only run the analysis.

---

## Inputs and outputs

- **Inputs:** you edit this folder by hand. The pipeline writes into it only
  when the imaging tools write an acquisition manifest.
- **Outputs:** none. The results go to `$SC_ANALYSIS_ROOT` (see
  [Outputs](../connectome_analysis/README.md#outputs) in
  `connectome_analysis/README.md`).

## Settings you can change (summary)

| What | Where | Default |
|---|---|---|
| which analysis settings file to use | `--config <file>` on the runner | `configs/analysis.yaml` |
| permutations, repeats, search size and so on | keys in the [first table above](#keys-the-runner-reads-changing-these-changes-a-run) | as listed |
| where the exclusion list is | `SC_EXCLUSIONS` or `--exclusions <file>` | `configs/exclusions.local.yaml` |
| who is excluded | `exclude_subjects:` in that file | nobody (file absent) |
| acceptable T1 scans and their order | `cohort_rules/Ranked_OK_T1.csv` | 299 descriptions, ranks 1 to 6 |
| where the acquisition manifest is | `SC_MANIFEST` | `configs/acquisition_manifest.csv` |

## If something goes wrong

| What you see | Why | Fix |
|---|---|---|
| `config not found: configs/analysis_typo.yaml` | The `--config` file name is wrong. | Check the spelling. Paths are relative to the repository root. |
| A long error ending in `yaml.parser.ParserError: while parsing a flow sequence` and `expected ',' or ']', but got ...` (the word after `but got` depends on what follows the broken line) | A bracket is not closed, for example `seeds: [11, 23`. | Close the bracket. The first `line ..., column ...` in the message points at the bracket that was opened. |
| `found character '\t' that cannot start any token` | A tab was used for indentation. | Indent with spaces only. |
| You changed a key and nothing changed, and the run printed `[skip]` for that stage | You used `--resume`. It skips a stage whose outputs already exist and are newer than its inputs, and it does not notice a settings change. | Re-run without `--resume`: `python -m connectome_analysis.run_analysis --stage <stage> --config configs/analysis.local.yaml`, or `--from <stage>` to redo everything after it too (see Step 5 above). |
| You changed a key and nothing changed, without `--resume` | Either the key is one of the [record-only keys](#keys-kept-as-a-written-record-changing-these-has-no-effect-today), or the stage name above it is misspelled, in which case it is silently ignored. | Check both tables above, and copy stage names from `python -m connectome_analysis.run_analysis --list`. |
| `[BLOCK] <stage>: N declared input(s) missing`, and the last line starts with `blocked=` (for example `blocked=1`) | That stage needs files that earlier stages write, and they do not exist yet in this analysis root. The lines under `[BLOCK]` name the missing files and the stages that make them. | Run the earlier stages first, or add `--with-deps` so the runner runs them too. |
| `[FAIL]  literature_retrieval: ValueError: literature_retrieval needs a contact address (stages.literature_retrieval.literature_mailto in the config)` | You ran the optional OpenAlex stage without an e-mail address. | Set `literature_mailto: "you@example.org"` under `stages.literature_retrieval` in your `.local.yaml` copy, and pass it with `--config`. |
| `exclusions    : NONE (... not found)` although the file exists | The key is misspelled (for example `exclude_subject:`), so the list is read as empty. | Spell it exactly `exclude_subjects:`. |
| The run stops at once with `could not read <file>: ParserError: while parsing a block mapping` (exit status 1, which means the program stopped with an error) | `exclusions.local.yaml` is not valid YAML, for example a dash is indented less than the one above it. The runner refuses to treat a broken list as "nobody excluded". | Fix the indentation so that all the `- ` lines start in the same column. |
| `exclusions : 1 subject(s)` (or any number) that is smaller than the number of people you listed | A `- ` line is indented further than the one above it, so YAML joined the two lines into one entry that matches nobody. This does not stop the run. | Make every `- ` line start in exactly the same column, then check that the number matches. |
| `1 DTI subject(s) have no acceptable T1 description` from `sc_cohort.py build` | None of that participant's T1 descriptions exactly matches a line in `Ranked_OK_T1.csv`. | This is only information. The analysis does not need `mri.csv` rows. If you want that participant to have a T1, add the exact description to `Ranked_OK_T1.csv`. |
| `git status` lists `exclusions.local.yaml` or a manifest | The file has a name that `.gitignore` does not cover. | Rename it to end in `.local.yaml` (for exclusions) or to contain `acquisition_manifest` (for manifests). **Do not commit it.** |
