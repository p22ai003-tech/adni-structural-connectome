# docs/: thesis material and project notes

## What is this folder?

This folder holds **documents about the project**, not code. The analysis and
the dashboard read nothing from here. The one link is `figs/`: three analysis
stages save their figures into it (see [Figures](#figures-figs)).

The documents are of two kinds:

- **Thesis material.** Drafts of the thesis, earlier reports, slide decks, one
  paper by other authors, and the figures. Most of these are Word (`.docx`),
  PowerPoint (`.pptx`) or PDF files.
- **Operational notes.** Short Markdown notes written while the imaging
  pipeline was being run on the original server: where files were kept, which
  settings were used, and which re-runs were tried. They are kept as a
  historical record.

Do not add participant data to this folder. ADNI data are governed by a Data
Use Agreement: you obtain them from adni.loni.usc.edu under your own agreement,
and they must never be committed to this repository. When you add a note or a
document, write `<SUBJECT>` or `XXX_S_0001` in place of a participant
identifier. The Word, PowerPoint and PDF files here are never checked
automatically (see [Step 2](#step-2-optional-see-the-package-with-the-documents-included)),
so treat them as covered by the same agreement.

## Do I need it?

- **To run the analysis or the dashboard:** no. You can ignore this folder.
  (The analysis writes its three figures into `figs/` by itself.)
- **To read the thesis or see the figures:** yes. Start with
  `iit_thesis/iit_2_complete.docx` (only in the full repository; the shared
  package leaves out Word files, see below) and `figs/`.
- **To learn how the connectome matrices are made:** the operational notes here
  are history, not instructions. For how to run the imaging today, see the
  Imaging section of the top-level [README.md](../README.md).

## Not everything here is in the shared package

The project is shared as a zip file built by `sc_package.py` (in the repository
root). That tool **leaves out binary documents**: every file ending in `.pptx`,
`.docx`, `.doc`, `.ppt`, `.pdf`, `.xlsx`, `.numbers` or `.key`. In this folder
that is 15 files, about 70 MB. Markdown (`.md`), text (`.txt`) and image
(`.png`) files are kept. The last column of the tables below says which is
which. Step 1 of [How to use it](#how-to-use-it) shows how to check this
yourself.

## What's inside

### Thesis material

| Path | What it is | In the shared package? |
|---|---|---|
| `iit_thesis/iit_2_complete.docx` | The latest full thesis document: literature review, dataset and methods, results, discussion and references. Title: *Short-Range DMN Connectivity as a Dominant Marker of Cognitive Severity in Alzheimer's Disease*. | no |
| `iit_thesis/iit_2.docx` | The same thesis before the literature review and the rewritten discussion were added. Kept unchanged so the two can be compared. | no |
| `iit_thesis/iit_2_complete_CHANGES.md` | Every edit made to the existing text of `iit_2.docx` to produce `iit_2_complete.docx`, shown as old text then new text. | yes |
| `iit_thesis/iit_2_complete_QA_NOTES.md` | The checks run on `iit_2_complete.docx` before it was finished: which numbers were recomputed from the analysis outputs and which citations were checked against the source papers. | yes |
| `iit_thesis/SOTA_Report.docx` | An earlier state-of-the-art report (a review of existing research), *Staging of AD using data & model driven estimation of axonal delays*. | no |
| `iit_thesis/Abstract_SOTA.docx` | The abstract of that report. | no |
| `iit_thesis/SOTA_V_5.0.pptx` | The slides for that report (36 slides). | no |
| `iit_thesis/RP_Report.docx` | A later state-of-the-art report plus research proposal, *Characterizing transitions from healthy to pathological ageing*. | no |
| `iit_thesis/RP_6.0.pptx` | The slides for the research proposal (32 slides). | no |
| `iit_thesis.docx` | An earlier draft of the thesis (working title *Limbic System can predict MMSE scores*). | no |
| `iit_thesis_backup_20260824.docx`, `iit_thesis_backup_pre_ml.docx`, `iit_thesis_worklog_version.docx` | Copies of that earlier draft, saved at different points while it was being written. | no |
| `Presentation_new.pptx`, `Presentation_refined.pptx` | Two versions of a slide deck of the findings (16 and 19 slides). | no |
| `literature/7. SR_LR_Compensation.pdf` | A paper by other authors, *Contributions of short and long-range white matter tracts in dynamic compensation with aging*. The thesis cites it. It is their work, so it is not redistributed. | no |
| `literature/SOTA_V_6.0.pptx` | A later version of the state-of-the-art slides (35 slides). | no |
| `figs/` | 13 figures (PNG images) used in the thesis and slides. Listed one by one in [Figures](#figures-figs). | yes |

### Operational notes

These notes describe the original server. The folder names and absolute paths
inside them (for example the data volume and a cloud storage bucket) belong to
that machine. On your machine the same folders are found through environment
variables, which are settings your shell passes to every program. They are
resolved in `sc_config.py`:

| The notes say | On your machine |
|---|---|
| `exp/`, `~/exp/`, or an absolute path ending in `/exp/` (the repository folder on the original server) | `<repo>` |
| the data volume: `data/`, `~/exp/data/`, or an absolute path whose first folder is the volume itself, followed by `derivatives/` or `Images/` | `$SC_DATA_ROOT` (default `<repo>/data`) |
| `.../Images/` (the raw ADNI images) | `$SC_RAW_IMAGES_ROOT` (default `$SC_DATA_ROOT/Images`) |
| `.../derivatives/` | `$SC_DERIV_ROOT` (default `$SC_DATA_ROOT/derivatives`) |
| `.../derivatives/connectomes/` | `$SC_CONNECTOMES_DIR` (default `$SC_DERIV_ROOT/connectomes`) |
| `.../derivatives/qc/analysis_cohort/` | `$SC_ANALYSIS_ROOT` (default `$SC_DERIV_ROOT/qc/analysis_cohort`) |
| `exp/cohort/` | `$SC_COHORT_DIR` (default `<repo>/cohort`) |

`<repo>` means the folder you cloned this repository into. When more than one
row fits a path, use the most specific one. For example,
`~/exp/data/derivatives/connectomes/` is `$SC_CONNECTOMES_DIR`. Code keeps its
place inside the repository: `exp/pipeline/audit/subject_manifest.py` in the
notes is `pipeline/audit/subject_manifest.py` in your copy.

| Path | What it is | In the shared package? |
|---|---|---|
| `DATA_LAYOUT.md` | Which folder each imaging stage wrote its files to, the quality bands the matrices were sorted into, and where backups were kept. | yes |
| `PIPELINE.md` | The imaging stages from raw images to connectome matrices, as they were run at the time. | yes |
| `PARAMETERS.md` | The settings (environment flags and command-line options) of the imaging re-runs. | yes |
| `DECISIONS.md` | Why the imaging pipeline was set up the way it was. | yes |
| `RECOVERY_LANES.md` | The batches of imaging re-runs (called "recovery lanes"), what each one changed and what it produced. | yes |
| `recipe_ledger_original.md` | An earlier version of the lessons that are now in `RECOVERY_LANES.md`. | yes |
| `runbooks/eddy_cli_runbook.md` | How the FSL `eddy` step (which corrects the diffusion images for head motion and scanner distortion) was run from a terminal, using the helpers in `scripts/eddy/`. | yes |
| `recovery/STEP7_RECOVERY_RESUME_POINTER.txt` | A hand-off note from April 2026. The folder it points to is not part of this repository. | yes |
| `s3/synced_ec2_s3_locations.txt` | Which data folders were copied to Amazon S3 (cloud storage). The bucket name is replaced by `<your-bucket>`. A script in `research_audit/` reads this file by its name, so do not rename it. | yes |

### Figures (`figs/`)

Some words used in this table:

- **CN, MCI, AD:** the three diagnostic groups. CN = cognitively normal, MCI =
  mild cognitive impairment, AD = Alzheimer's disease. By the project's
  grouping rule, the ADNI labels EMCI, LMCI and SMC are counted as MCI.
- **Short-range (SR), middle-range (MR), long-range (LR):** connections sorted
  by the length of the fibre tract between the two brain regions.
- **EDR exception:** a connection that is much stronger than expected for its
  length. EDR is the "exponential distance rule", which says connection
  strength falls off with length.
- **Cliff's δ (delta):** an effect size. It measures how far apart two groups
  are, from −1 to +1, where 0 means no difference.
- **FA, MD, RD, AxD:** diffusion measures of the white matter (fractional
  anisotropy, mean, radial and axial diffusivity).
- **SHAP, ICE:** ways to show which inputs a machine-learning model relies on,
  and how its prediction changes when one input changes.
- **MMSE:** the Mini-Mental State Examination, a 30-point test of cognition.

| File | What it shows | Made by |
|---|---|---|
| `fig1_migration.png` | Mean absolute Cliff's δ (the size of the difference, ignoring its direction, from 0 to 1) for short- and long-range connections in the three group comparisons (CN→MCI, CN→AD, MCI→AD), for all connections and for EDR exceptions only. | no script in this repository |
| `fig2_dissociation.png` | The largest significant effect for limbic and frontoparietal EDR exceptions, early (CN→MCI) and late (MCI→AD). | no script in this repository |
| `fig3_fa_late.png` | How many of the 10 functional networks differ significantly in FA and in diffusivity, in each group comparison. | no script in this repository |
| `fig4_shap_mmse.png` | Which features the MMSE prediction model uses (SHAP), overall and per participant. | analysis stage `build_ml_explain_v2` |
| `fig5_ice_mmse.png` | How the predicted MMSE changes as each top feature changes (ICE curves). | analysis stage `build_model_interpretation` |
| `fig6_mmse_buckets.png` | MMSE split into three equal-sized bands: per-band ROC and precision-recall curves, and per-band SHAP. | analysis stage `build_mmse_buckets` |
| `figA_ranges.png` | The tract-length cut-offs used for short-, middle- and long-range connections. | no script in this repository |
| `figA_tract_lengths.png` | The distribution of tract lengths, overall and per group. | no script in this repository |
| `figB_exc_0.png` … `figB_exc_3.png` | Four panels about EDR exceptions: the exception rate per length class and group, the rate per participant, long-range strength of exception and other connections, and the strength ratio of exception to other connections. | no script in this repository |
| `figC_gain_heatmap.png` | For each network and measure, how much the CN–AD effect size changes when only EDR exceptions are used. | no script in this repository |

"No script in this repository" means the figure is a finished image kept for
the thesis. Nothing will overwrite it.

The three figures made by analysis stages are rewritten every time those stages
run. The stage `build_model_interpretation` also writes a first version of
`fig4_shap_mmse.png`, which `build_ml_explain_v2` then replaces.

## Before you start

**To read the documents** you only need ordinary programs:

- `.docx`: Microsoft Word or LibreOffice Writer.
- `.pptx`: Microsoft PowerPoint or LibreOffice Impress.
- `.pdf`: any PDF reader.
- `.png`: any image viewer or web browser.
- `.md` and `.txt`: any text editor. Markdown (`.md`) is plain text with light
  formatting; GitHub and most code editors display it nicely.

**To run the commands below:**

1. Install the project as the top-level [README.md](../README.md) describes.
   That creates a **venv**, which is a private Python installation just for
   this project.
2. Open a terminal in the repository root (the folder that holds
   `sc_config.py`) and switch the venv on:

   ```bash
   source .venv/bin/activate
   ```

   Every command on this page is run from the repository root.
3. Step 3 needs the analysis outputs to exist already. You make them by running
   the analysis once, as described in
   [connectome_analysis/README.md](../connectome_analysis/README.md).

## How to use it

### Step 1: check what goes into the shared package

```bash
python sc_package.py --check
```

`--check` means "look, but write nothing". It builds the package in a temporary
folder, scans it, prints a summary and deletes it. What you should see:

```text
revision : 8854cc2
contents : 500 files, 9.6 MB   (469 excluded)

safety gate
--------------------------------------------------------------------
  clean: no participant identifiers, credentials or data files.

--check: nothing written.
```

The revision (the short name of your latest git commit) and the counts will
differ as the repository changes. The "excluded" count includes the binary
documents from this folder. The "safety gate" is a scan that refuses to build a
package containing an ADNI participant identifier, a password, a key or an
imaging data file. It reads **text files only** (code, Markdown, CSV, YAML,
JSON and similar). It cannot look inside Word, PowerPoint or PDF files.

This only works when every change is committed. See
[If something goes wrong](#if-something-goes-wrong).

### Step 2 (optional): see the package with the documents included

```bash
python sc_package.py --check --include-documents
```

What you should see (same revision as above):

```text
contents : 516 files, 79.6 MB   (453 excluded)
```

The 16 extra files are the 15 binary documents in this folder and one Word
file in the repository root.

**The safety gate still says "clean" here, but it has not checked those 16
files.** It cannot read inside Word, PowerPoint or PDF files, so "clean" covers
only the text files. Treat a package built with `--include-documents` as
restricted: never share it with anyone who is not covered by an ADNI Data Use
Agreement. Leave `--include-documents` off when you build a package to share.
Leave off `--check` only when you really want to write the zip file into
`dist/`.

### Step 3 (optional): remake a figure without touching `docs/figs/`

When you run the analysis normally (`python -m connectome_analysis.run_analysis
--all`), the three analysis-made figures are written straight into
`docs/figs/`. If you only want to see a figure being remade, you can send it to
a scratch folder instead. This example remakes `fig6_mmse_buckets.png` in a
folder called `~/sc-try` (`~` means your home folder):

```bash
mkdir -p ~/sc-try/analysis/20_exception_specificity ~/sc-try/figs
SRC="$(python -c 'import sc_config; print(sc_config.paths().exception_dir)')"
for f in ml_feature_matrix ml_targets ml_ladder_results ml_predictions ml_missingness; do
  cp "$SRC/$f.csv" ~/sc-try/analysis/20_exception_specificity/
done
cp configs/analysis.yaml ~/sc-try/analysis.yaml
echo "figs_dir: $HOME/sc-try/figs" >> ~/sc-try/analysis.yaml
python -m connectome_analysis.run_analysis --stage build_mmse_buckets \
  --analysis-root ~/sc-try/analysis --config ~/sc-try/analysis.yaml
```

What this does, line by line:

1. Makes the scratch folders.
2. Finds where your real analysis outputs are (`$SC_ANALYSIS_ROOT/20_exception_specificity`).
3. Copies the five files this stage needs from there. The runner checks that
   they exist before it starts.
4. Copies the analysis settings file and adds one setting, `figs_dir`, which
   tells the runner where to put figures.
5. Runs only the stage `build_mmse_buckets`, with the scratch folders in place
   of the real ones.

It takes several minutes: six to eight on a quiet 64-core machine, and longer
on a smaller or busy one. What you should see (your folders in place of the
`<...>` parts; the seconds will differ):

```text
connectomes   : <your SC_CONNECTOMES_DIR>
cohort tables : <your SC_COHORT_DIR>
exclusions    : 1 subject(s) from <repo>/configs/exclusions.local.yaml
analysis root : <your home folder>/sc-try/analysis
config        : <your home folder>/sc-try/analysis.yaml
plan          : 1 stage(s)

  [run]   build_mmse_buckets ...
  [ok]    build_mmse_buckets  (371.1s)

ok=1   total 371.4s
ledger: <your home folder>/sc-try/analysis/exports/stage_status.csv
```

The last line names the small table (the "ledger") where the runner records
how each stage went. If the `exclusions` line instead says `NONE (... not
found)` and warns that the ML stages will not reproduce the published results,
you can ignore it for this step: `build_mmse_buckets` does not read the
exclusions file. (The list of excluded participants is applied earlier, when
the copied input files were made.)

Compare the new figure with the one in the repository:

```bash
cmp ~/sc-try/figs/fig6_mmse_buckets.png docs/figs/fig6_mmse_buckets.png && echo identical
```

On the original data this prints `identical`: the figure is reproduced byte
for byte. The copied files contain participant-level data, so delete the
scratch folder when you are done:

```bash
rm -rf ~/sc-try
```

## Inputs and outputs

**Inputs.** Nothing in this folder is an input to the analysis or the
dashboard.

**Outputs.** Three analysis stages write PNG images (200 dots per inch) into
`docs/figs/` by default:

| File | Written by stage | Made from |
|---|---|---|
| `fig4_shap_mmse.png` | `build_model_interpretation`, then replaced by `build_ml_explain_v2` | `$SC_ANALYSIS_ROOT/20_exception_specificity/ml_feature_matrix.csv` and `ml_targets.csv` |
| `fig5_ice_mmse.png` | `build_model_interpretation` | the same two files |
| `fig6_mmse_buckets.png` | `build_mmse_buckets` | the same two files |

## Settings you can change

| Setting | Where | What it does | Default |
|---|---|---|---|
| `figs_dir` | A top-level line in the analysis settings file. `configs/analysis.yaml` does not contain it; add it to a copy and pass the copy with `--config`, as in Step 3. | The folder the three figure stages write into. | `<repo>/docs/figs` |
| `--include-documents` | Flag of `python sc_package.py` | Keeps the binary documents in the package. | off |
| `--check` | Flag of `python sc_package.py` | Scans without writing the zip file. | off (the zip is written to `dist/`) |

## If something goes wrong

**`working tree is not clean; commit first so the package matches a revision.`**
`sc_package.py` only packages committed files, and it refuses to run while
anything is changed or new. Run `git status` to see what. Commit the changes
you want to keep, or move the others out of the repository, then run it again.

**`sc_package.py` prints `N problem(s). Nothing written.`** The safety gate
found something that must not be shared, such as an ADNI participant
identifier in a note you added. The lines above it count the problems by kind,
then, under `first 15:`, list up to 15 of them, each with its file (if N is
larger, fix those and run it again to see the rest). Remove each identifier (write `<SUBJECT>` or `XXX_S_0001` instead),
commit, and run it again. The next line says
`Fix them, or re-run with --force if every one is a false positive.` A false
positive is a match that only looks like an identifier. Do not use `--force`
unless you are sure that none of the matches is a real participant
identifier.

**The figures in `docs/figs/` changed after I ran the analysis.** That is
expected: three stages write there. To see what changed and to put back the
committed versions:

```bash
git diff --stat docs/figs
git restore docs/figs
```

**A path in one of the operational notes does not exist on my machine.** That
is expected. The notes describe the original server. Use the table in
[Operational notes](#operational-notes) to find the same folder through the
`SC_*` variables, or run `python -c "import sc_config; print(sc_config.describe())"`
to print where every folder is on your machine.

**`recovery/STEP7_RECOVERY_RESUME_POINTER.txt` points to a folder that is not
there.** That folder was never part of this repository. The note is kept only
as a record.
