# Cerebral Cortex revision: uniformly rebuilt connectomes and the revision analyses

Code for the revision of the ADNI structural-connectome manuscript (October 2026). The structural connectomes were
rebuilt from the diffusion images of all 530 sessions with one pipeline, and every revision analysis was run on them.

This folder holds **code and output-free notebooks only**. ADNI data, derived matrices, participant-level results, the
participant exclusion list and the manuscript, thesis and report documents are not in the repository (ADNI Data Use
Agreement; the repository's policy of keeping thesis and manuscript documents out). The scripts run from
`docs/iit_dipanjan/tools/` on the analysis machine (that folder is excluded from git); this folder mirrors them, so paths
in the scripts are those of the analysis machine (`/data/derivatives/...`, `/home/ec2-user/exp/...`).

## Layout

| Folder | What it does |
|---|---|
| `tools/rebuild/` | The uniform rebuild. `rb2_subject.py --from-dwi` (one participant): N4, dhollander WM response, single-shell CSD (lmax 8), DTI, 3M iFOD2 streamlines (dynamic seeding, 0.06, 45°, 10–250 mm), SIFT2, tensor sampling along streamlines, AAL3 by FMRIB58→FA SyN registration, assignment within 2 mm, nine matrices; `rb_dk.py` adds the DKT76 parcellation from the same streamlines. `rb3_run_auto.sh` drives all participants. `build_tree.sh` runs the full analysis pipeline (35 stages, ML included) on the connectomes that pass rebuild quality control; `refresh_all.sh` re-runs Phases 1–3 and the notebooks. `rb_act_subject.py` / `rb_act_run.sh`: the ACT re-tracking check (45 participants). `README_rebuild.md`: the rebuild log (participant identifiers replaced by letters). |
| `tools/phase1/` | Confound control: covariates, network features, data audit and quality control (rules R0–R4), 13 conditions (pooled, ADNI-3/4, covariate adjustment, vendor ComBat, propensity matching, atrophy and lesion adjustment, consistent edges, amyloid groups, Siemens only, processing route), tables and verdicts against the earlier estimates. |
| `tools/phase2/` | Length test: edge cache, effect against edge length in ten bins, rank-based slope per doubling of length with bootstrap and permutation, mixed model, alternative short/long boundaries. |
| `tools/phase3/` | Exception-core null models: controls-only core (500 splits), split-half stability, 2.5/3.5 SD thresholds, strength- and length-matched non-core edges, coverage adjusted for total connectivity. Edge weight selectable (`IIT_P3_WEIGHT`). |
| `tools/phase4/` | Incremental prediction: ADNIMERGE2 scores within 180 days of the scan, four feature families, nested ridge models with 20 × 5-fold paired cross-validation, bootstrap intervals, Freedman–Lane permutation, within-diagnosis analyses. |
| `tools/phase5/` | Robustness: DKT network labels by voxel overlap (`p5_dkt_map.py`), Phase 1 features for any parcellation or edge weight (`p5_features.py`, validated against the pipeline to 1e-9), `run_phase5.sh` (DKT, streamline-count and FA weights), `run_cap245.sh` (edges at the 250-mm tracking limit removed), `p5_act.py` (ACT against unconstrained tracking). |
| `tools/phase6/` | Manuscript figures (300 dpi). |
| `tools/phase7/` | `p7_numbers.py`: every number the revised manuscript quotes, from the phase outputs. |
| `tools/iit_chain.py`, `sc_cohort_settings.py` (repository root) | One root variable (`IIT_ROOT`) and one set of settings per chain: density floor, SMC handling, primary sample, length measure, nodes analysed, label hiding. Unset, every script resolves today's paths and settings. Launchers refuse connectomes whose recorded recipe hash differs from the signed recipe, and builders refuse to overwrite a Word file. |
| `tools/narrative_guard.py` | A document builder stops when computed numbers contradict a fixed sentence such as "no crossover" or "at every stage". |
| `tools/run_chain.sh`, `tools/chain_lib.sh`, `tools/test_chain_guards.py` | The whole chain under one root with a ledger of every stage's result; helpers; tests of the guards on invented tables. |
| `notebooks/` | Phase 1–3 notebooks (outputs removed; built by `build_phase*_notebook.py`, kernel with the packages below). |

`tools/docx2txt.py` extracts Word text with insertions and deletions marked. Fix in the pipeline that came with this work:
`hcp_analysis/build_range_restricted_networks.py` no longer stops when a measure is constant in every group.

## Order

1. `tools/rebuild/rb3_run_auto.sh` (all participants), then `tools/rebuild/build_tree.sh` (analysis tree, 35 stages).
2. `tools/rebuild/refresh_all.sh full` (Phases 1–3 and notebooks), `tools/phase4/p4_data.py`, `p4_engine.py 500 20`,
   `p4_tables_figs.py`; `tools/phase5/run_phase5.sh`, `p5_tables_figs.py`, `run_cap245.sh`; `tools/phase6/p6_figures.py`;
   `tools/phase7/p7_numbers.py`.

Environment: Python 3.12 with numpy, pandas, scipy, scikit-learn, statsmodels, neuroCombat, pyreadr, nbformat, matplotlib
and python-docx; the pipeline environment (`.venv_connectome_app`, with nibabel) for the rebuild, the DKT mapping and the
figures; MRtrix3, FSL and ANTs for the rebuild.

## Recipe v4 (November 2026): the connectomes of iteration 4

Iteration 4 of the revision re-processed every session from the raw diffusion DICOMs with one recipe that was fixed and
signed before any result was looked at (`recipe_v4_record/RECIPE.md`). `recipe_v4_record/SIGNED_RECIPE.json` holds the
SHA-256 of the recipe text and of the four run scripts; the committed `tools/recipe_v4/rv4_subject.py`, `rv4_cohort.sh`,
`rv4_t1_ingest.py` and `rv4_clone_start.sh` are those exact files (hashes checked at commit time).

| Step | Script |
|---|---|
| One session, DICOM to connectome: dcm2niix; GE in-plane interpolation undone; dwidenoise; mrdegibbs; Synb0-DisCo + topup from the same-visit T1; eddy with outlier replacement; N4; single-tissue CSD (b = 1000, lmax 8); FastSurfer on the same-visit T1; 5TT; T1 to b = 0 (ANTs rigid or BBR, better white-matter Dice kept); AAL3 through the T1 (SyN); ACT iFOD2, 10 million streamlines with backtracking and GM–WM-interface cropping; SIFT2; matrices of count, SIFT2 weight, mean length and FA/MD/AD/RD sampled along every streamline; QC file | `tools/recipe_v4/rv4_subject.py` |
| Same-visit T1 ingest (IDA download to the stored T1 form) | `rv4_t1_ingest.py` |
| Cohort driver, and the lanes that spread it over several machines | `rv4_cohort.sh`; `rv4_split.py`, `rv4_resplit.py`, `rv4_lanes.sh`, `rv4_clone_start.sh`, `rv4_clone_lanes.sh`, `rv4_pull.sh`, `rv4_overnight.sh` |
| Matrices to the analysis chain's CSVs, and the DKT76 matrices from the same streamlines (refuses unless it reproduces the run's own matrices) | `rv4_derive.py`, `rv4_derive_all.sh` |
| The same-visit FastSurfer outputs the covariates read | `rv4_fs_view.py` |
| Analysis chains (Phases 1–6 under one root, with a ledger): main, and without the flagged sessions | `rv4_chains.sh act10m` and `rv4_chains.sh act10m_noflag`, run with `RV4_SUFFIX=_r2 IIT_FASTSURFER=<root>/derived/fastsurfer_same_visit` for the final roots; `tools/run_chain.sh` |
| Side-by-side summary of the arms | `rv4_findings.py` |

Not in this folder: the iteration-4 document builders (they contain manuscript, thesis and report text), two
operational scripts with machine details (the finish check of the extra machines and the S3 backup), and the run log
(`FREEZE.md`), which names participants; these stay on the analysis machine and its private backup.

Known points about this code, all recorded in the run log:
- `tools/phase5/run_cap245.sh` removes edges by the edge file's `len`. Under recipe v4 that field is the MNI distance between
  region centres, which never reaches 245 mm, so this repeat removes no edges; the documents report the share of present
  edges whose streamlines average 245 mm or more instead (0.07%).
- The Phase 4 permutation test (`tools/phase4/p4_engine.py`) shuffles the residuals of the connectome features after
  regressing them on the smaller model's predictors and adds them back to the fitted part: the Smith procedure in the
  terms of Winkler et al. (2014, NeuroImage 92: 381–397), not the Freedman–Lane test its docstring names. Results unchanged.
- Propensity matching (condition C4) can pick slightly different neighbours on a different CPU (numerical libraries), and
  Phase 4 takes one of two equally near visits for a few participants' composite scores (the sort is not stable).
