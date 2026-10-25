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
