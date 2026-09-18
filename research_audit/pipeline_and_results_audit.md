# Structural-connectome pipeline and results audit

**Audit date:** 2026-07-18 UTC  
**Scope:** `/home/ec2-user/exp`, live local services, current `/data/derivatives` outputs, and the generated dashboard-analysis tree.  
**Mode:** read-only inspection and lightweight integrity/statistical summaries. No production code was changed and no MRtrix, registration, tractography, or analysis refresh was started.

## Executive decision

The project contains a substantial, working structural-connectome pipeline and a live dashboard, but the current 530-subject analysis is **not manuscript-ready and must not be presented as a validated novel discovery**.

The most serious blockers are:

1. **DTI and T1 are paired by subject identity, not visit/date.** In the 530-subject dense cohort, 196/530 (37.0%) pairs come from different ADNI phases. The absolute DTI–T1 age gap is mean 3.20 years, median 2.10 years, 90th percentile 8.41 years, and maximum 13.8 years. Only 197/530 (37.2%) pairs are within 0.5 years. This can invalidate T1-derived anatomy, atlas placement, 5TT, registration, and tractography, and the mismatch differs sharply across diagnostic subgroups.
2. **The current cohort is selected by matrix density alone while carrying a contradictory, older QC gate.** The 530-row master table contains 363 rows marked `sc_matrix_qc_include=False` / `exclude_pending_repair`, yet the cohort builder retains every row with density >=0.6. The old QC table predates the June repairs, so it cannot simply be enforced; it must be regenerated against the current 166-node matrices and then enforced.
3. **Most recovery `count` matrices are not raw streamline-count matrices.** The recovery code unconditionally supplies SIFT2 weights for every output, including `count`. Direct inspection finds 477/530 current `count` matrices exactly equal to `fd_sum`; those same 477 contain fractional nonzero values. Any dashboard or manuscript wording that calls these “tract counts” is incorrect.
4. **Processing recipes are heterogeneous and incompletely recorded.** The dense set mixes at least radial-2 and radial-4 assignment, noACT and legacy ACT routes, 10M and 3M streamline routes, and multiple recovery tags. Radial radius alone has a large association with density (mean 0.753 at radius 2 versus 0.685 at radius 4; Mann–Whitney p=2.32e-11). The current statistical models do not control for these technical recipes.
5. **Diagnosis is strongly confounded with ADNI phase/protocol.** In the dense cohort, AD is 33/44/1 across ADNI 2/3/4, while CN is 3/242/6 and MCI is 1/194/6; chi-square p=5.31e-37. Many large unadjusted network effects disappear after the project’s own age/sex/phase-adjusted permutation test. The headline limbic AD/MD/RD effects have adjusted permutation p=.930/.530/.158, respectively.
6. **The current “structure-function coupling” story is mislabeled.** The measured coupling is between structural graph topology and DTI microstructure (for example nodal efficiency versus FA). No functional MRI measure is present. It may be called *structure–microstructure* or *topology–microstructure* coupling, but not structure–function coupling.
7. **The novelty review is explicitly non-exhaustive.** The generated novelty report uses nine references and acknowledges that its automated retrieval is not exhaustive. That is not enough to prove a literature gap or support a high-confidence novelty claim.

The current dashboard is useful as an **exploratory QA/research-development interface**. It should retain a conspicuous `provisional / not for inference` label until the P0 corrections in this report are completed.

## 1. What exists now

### 1.1 Context and documentation

There is a project-specific context document:

- `structural_connectome_context.md`
- `structural_connectome_context.docx`

Both are dated 2026-05-25. They describe 639 subjects and conclude that most AAL3 outputs should remain excluded; see `structural_connectome_context.md:9`, `:94`, `:157`, and `:173-177`.

The newer June 17 operational documents are:

- `docs/DATA_LAYOUT.md`
- `docs/PIPELINE.md`
- `docs/PARAMETERS.md`
- `docs/DECISIONS.md`
- `docs/RECOVERY_LANES.md`
- `CLAUDE_RESUME_10M.md` (June 12)

The June documentation describes 648 subjects and 530 density-good connectomes (`docs/DATA_LAYOUT.md:3`, `docs/DECISIONS.md:35-37`). Thus the May context and June build documentation conflict. The May document should not be deleted, because it contains important failure evidence, but it should be superseded by one current, reconciled project context that clearly distinguishes historical QC from current QC.

### 1.2 Code and workflow roots

- Production pipeline: `connectome_pipeline/`
- Analysis code: `connectome_analysis/`
- Dashboard: `apps/connectome_dashboard/connectome_app.py`
- Dashboard refresh runner: `apps/connectome_dashboard/refresh_connectome_dashboard_data.py`
- Recovery engine: `scripts/scforge/live/run_sc_route_sota.py`
- Audit/monitoring helpers: `pipeline/audit/`, `pipeline/monitoring/`
- Intended workflow package: `scforge/`
- Main derivatives: `/data/derivatives` through the symlink `data -> /data`

The nominal Snakemake workflow is not executable. `scforge/workflow/Snakefile:1-7` explicitly says it is a placeholder and has an empty `rule all`; every file under `scforge/workflow/rules/` is a one-line placeholder. The successful recovery workflow is therefore a collection of ad hoc Python/shell launchers rather than a fully reproducible DAG.

### 1.3 Live dashboard

Current live state verified on 2026-07-18:

- `connectome-dashboard.service`: active, main PID 149713, active since 2026-06-17 15:05 UTC.
- Streamlit binds to `127.0.0.1:8501` using `.venv_connectome_app`; exact command is in `deploy/connectome-dashboard.service:10-16`.
- nginx is active and listens on ports 80/443. It proxies HTTPS to localhost 8501 and requires basic auth (`deploy/connectome-dashboard.nginx.conf:11-29`).
- Deployment documentation lists `https://<dashboard-host>/` and notes that the certificate is self-signed unless replaced (`deploy/README_connectome_dashboard.md:3-14`).
- `pipeline/monitoring/status_writer.py` is running as PID 9085.
- No active `tckgen`, `tcksift2`, MRtrix batch, or dashboard refresh process was found.

The live status JSON was current on July 18 and reports 530/648 good, mean density .745, median .741, with AD 78/100, MCI 201/241, and CN 251/307. In contrast, the analysis snapshot was last fully refreshed on June 17 and explicitly says `snapshot_mode=provisional`:

- `/data/derivatives/qc/sc_matrix_qc/connectome_status.json`
- `/data/derivatives/qc/analysis_cohort/00_master/dashboard_refresh_status.json`

The monitor status is current, but the inferential analysis is a month-old provisional snapshot. Those are different freshness guarantees and should be displayed separately.

## 2. Reconstructed end-to-end pipeline

The project documentation gives the intended sequence at `docs/PIPELINE.md:8-23`; code inspection confirms the main stages below.

| Stage | Input | Main operation | Output/evidence |
|---|---|---|---|
| Cohort selection | `cohort/dti.csv`, `cohort/mri.csv` | One selected DTI and one selected MRI row per subject | 1,114 unique subject rows in each CSV; pipeline subset 648 |
| DICOM conversion | `/data/Images/dti/<subject>` | dcm2niix / `mrconvert`, gradient/source validation | `/data/derivatives/mif_dwi/<sid>.mif`; `connectome_pipeline/dwi_convert.py` |
| Denoise | DWI MIF | `dwidenoise` | `mif_denoised/<sid>_den.mif` |
| Gibbs removal | denoised MIF | `mrdegibbs` | `mif_unringed/<sid>_den_unr.mif` |
| Eddy/motion | unringed MIF | `dwifslpreproc -rpe_none`, phase encoding from header or fallback, FSL eddy | `eddy/<sid>_preproc.mif`; `connectome_pipeline/dwi_eddy.py:1547-1573` |
| Bias correction | preprocessed DWI | ANTs N4 | `biascorr_1/<sid>_unbiased.mif` |
| T1/b0 setup | selected T1 and DWI b0 | FLIRT 6-DOF / BBR, MRtrix transforms | `t1_anat/`, `t1_fast/`, `dwi_t1_bbr/` |
| 5TT/GMWMI | T1 | `5ttgen`, `5tt2gmwmi` | `fod/<sid>/5tt_b0.mif`, `gmwmi.mif` |
| FOD | preprocessed DWI | Dhollander response, SS3T-CSD, `mtnormalise` | `fod/<sid>/wmfod_final.mif` |
| DTI maps | preprocessed DWI | `dwi2tensor`, `tensor2metric` | `dti/<sid>/{fa,md,ad,rd,dt}.mif` |
| Atlas | AAL3 1 mm MNI | ANTs SyN MNI->T1, T1->b0, nearest-neighbor/generic-label resampling | 166-node AAL3 in DWI space |
| Tractography | FOD and mask/5TT | iFOD2; legacy 3M ACT/GMWMI or recovery 10M noACT dynamic seeding | `.tck` tracks |
| Streamline weighting | tracks + FOD | SIFT2 | `sift_weights.txt` |
| Streamline sampling | tracks + DTI maps | `tcksample -stat_tck mean` | FA/MD/AD/RD `.tsf` |
| Assignment/connectomes | tracks + AAL3 | radial endpoint assignment, `tck2connectome` | nine AAL166 matrices |
| QC/promotion | matrices | density band and recovery promotion | main/mid/low directories |
| Analysis | promoted matrices | graph, DTI, edgewise, coupling, EDR, brain age, ML, network summaries | `/data/derivatives/qc/analysis_cohort` |
| Dashboard | generated analysis tree | read-only Streamlit rendering | localhost 8501 behind nginx |

### BATMAN terminology

The only non-archived code occurrence of “BATMAN” is the docstring in `connectome_pipeline/dwi_eddy.py:5-8`, where it describes a **BATMAN-style single-phase-encoding setup** using `dwifslpreproc -rpe_none`. No BATMAN package, executable, import, independent processing stage, or separate output tree was found. The current pipeline should therefore describe this as the single-PE eddy configuration, not imply that a distinct BATMAN software pipeline is executed.

### Recovery route details

The current recovery route is materially different from the legacy production route:

- AAL3 is kept as 166 contiguous nodes (`scripts/scforge/live/run_sc_route_sota.py:44-50`).
- Default `FORCE_NOACT=1` forces FOD-based noACT tracking for **all** subjects, including isotropic data (`run_sc_route_sota.py:386-400`).
- Tractography requests 10M streamlines with dynamic seeding, cutoff .06, length 10-250 mm, and seed cap 200M (`run_sc_route_sota.py:370-405`).
- SIFT2 is run without `-act` when the route is noACT (`run_sc_route_sota.py:445-451`).
- Cached and fresh registrations are scored by resulting matrix density, and the better density is selected (`run_sc_route_sota.py:589-612`). This is outcome-driven technical selection and must be recorded as part of each subject’s recipe.

The June documentation is partly inconsistent: `docs/DECISIONS.md:9-11` describes noACT as an anisotropic-data fix, while the actual code and `docs/PARAMETERS.md:9` force noACT for all subjects by default.

## 3. Current data/output inventory

### 3.1 Cohort funnel

The current source-of-truth manifest is `/data/derivatives/qc/sc_matrix_qc/subject_manifest.csv`, dated June 17. It has 648 unique series/subjects:

- Full cohort: AD 100, MCI 241, CN 307.
- Density-good main directory: AD 78, MCI 201, CN 251; total 530.
- Mid band: 5.
- Low band: 113.

All 648 manifest rows claim raw DWI/T1, MIF, eddy, T1, BBR, FOD, DTI, parcellation, track, and SIFT presence. Intermediate directories are sparse (37 denoised, 44 unringed, 69 bias-corrected) because intermediates were consumed or cleaned. TSF is present for 638.

Current main AAL166 matrix counts:

| Matrix type | Files |
|---|---:|
| count | 530 |
| count_invnodevol | 530 |
| fd_sum | 530 |
| len_mean | 530 |
| invlen_mean | 530 |
| fa_mean | 529 |
| md_mean | 529 |
| rd_mean | 529 |
| ad_mean | 529 |

Subject `168_S_6938_I1444126` is missing all four diffusivity matrices despite being in the density-good cohort.

The S3 location recorded by current project documentation is `s3://<your-bucket>/exp/` (`docs/DATA_LAYOUT.md:62-65`). This audit did not perform a live S3 object-by-object verification, so local snapshot claims must not be treated as current cloud proof without a new checksum inventory.

### 3.2 Matrix integrity

All 530 `count` matrices inspected are 166x166, finite, symmetric, and zero-diagonal. Count-derived density ranges from .60095 to .95239; median .74133 and mean .74594.

However, density is not sufficient QC:

- 190/530 count matrices have at least one completely zero row; only 340 are zero-row-free. Maximum zero rows is 28.
- Diffusivity matrices have a median of 8 zero rows, and 415/529 have at least one zero row. Their maximum is 164 zero rows.
- `invlen_mean` has a median of 54 zero rows and maximum 164.
- Three FA matrices exceed the physical upper bound 1 (max 1.19158).
- Negative diffusivity values occur in 5 MD subjects, 10 RD subjects, and 2 AD subjects; minima are -0.002719, -0.003133, and -0.002421, respectively.

These files are numerically readable, but not all are physically or anatomically valid. The final QC gate must validate ROI coverage, atlas/mask overlap, endpoint assignment, and physical tensor ranges, not only shape/symmetry/density.

## 4. Release-blocking findings

### P0-A. DTI–T1 visit mismatch

#### Evidence in code/data

- `cohort/dti.csv` contains one selected DTI row per subject and includes DTI phase, study date, visit, age, protocol, and image ID.
- `cohort/mri.csv` contains one selected MRI row per subject, but the chosen MRI is not visit-matched to the DTI.
- `_subject_root_for_native_t1()` strips the DWI image ID to `site_S_subject`, and `_find_native_t1_inputs()` accepts the first matching T1 under that subject root (`connectome_pipeline/connectome_step7.py:3320-3368`).
- The audit manifest explicitly comments that T1 image ID differs and selects `T1_ss_<base>_*.nii.gz` (`pipeline/audit/subject_manifest.py:26-27`, `:93-103`).
- `docs/DATA_LAYOUT.md:28` documents the same base-subject match.

Concrete example:

- `002_S_0413` DTI: ADNI 3, age 87.5, image 863064, 2017-06-21.
- Selected T1: ADNI 1, age 80.4, image 291872.
- Approximate modality interval: 7.1 years.

Derived audit results:

| Cohort | N | Phase mismatch | Mean absolute age gap | Median | 90th pct | Max | <=0.5 y | <=2 y |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Pipeline | 648 | 230 (35.5%) | 3.23 y | 2.10 y | 8.50 y | 16.4 y | 36.4% | 48.8% |
| Dense analysis | 530 | 196 (37.0%) | 3.20 y | 2.10 y | 8.41 y | 13.8 y | 37.2% | 49.8% |

Dense-cohort mean absolute gaps using the original ADNI labels are AD 1.09 y, CN 3.32 y, EMCI 7.42 y, LMCI 6.24 y, MCI 2.26 y, and SMC 5.55 y. Thus mismatch is not random with respect to disease subgroup.

#### Consequence

The selected T1 drives brain anatomy, 5TT/GMWMI, atlas warp, and DWI registration. A multi-year mismatch can introduce true longitudinal atrophy and ventricular/cortical change into what is treated as a same-session structural connectome. This is a biological and technical confound, not merely missing metadata.

#### Required correction

Create a visit-level acquisition manifest before any rerun. Match DTI and T1 by exact study date/visit where possible, then by a pre-specified maximum interval (recommended primary <=90 days; sensitivity <=180 days). Record modality dates, interval, phase, scanner/site, field strength, gradient directions, and image IDs. Exclude subjects outside the primary window or rerun with the closest valid T1. Report a paired-versus-unpaired sensitivity analysis.

### P0-B. Current QC gate is stale and not enforced

`connectome_analysis/analysis_cohort.py:142-165` attaches the SC QC fields, recomputes density, and retains all subjects with density >=.6. It does not filter on `sc_matrix_qc_include`. The merge itself is documented as “without removing rows” at `analysis_cohort.py:180-186`.

The current master table therefore contains:

- `sc_matrix_qc_include=False`: 363
- `sc_matrix_qc_include=True`: 167
- `FAIL_REGISTRATION`: 230
- `WARN_SPARSE_NODE`: 104
- `FAIL_LABEL_COVERAGE`: 85
- `PASS`: 63
- `FAIL_STREAMLINE_ASSIGNMENT`: 48

Important nuance: `/data/derivatives/qc/sc_matrix_qc/sc_matrix_qc_decisions.csv` is dated May 12, while the recovery manifest and current matrices are June 17. These flags are likely stale for repaired outputs. It would be wrong either to ignore them or blindly enforce them. Their presence in the June master table proves that the analysis snapshot has not been reconciled with current post-repair QC.

Required correction:

1. Re-run the full spatial/matrix QC against every current 166-node artifact.
2. Require zero/allowed-zero rows, label survival, registration overlap, assignment fraction, finite values, symmetry, zero diagonal, expected dimensions, and required weight availability.
3. Write a new versioned QC decision table with input/output hashes and timestamps.
4. Build the analysis cohort only from the new `include` decision, not density alone.
5. Produce a participant/series flow diagram showing failure and exclusion reasons by group.

### P0-C. `count` semantics are wrong for recovery outputs

The recovery code declares separate `count` and `fd_sum` outputs at `scripts/scforge/live/run_sc_route_sota.py:52-59`, but `build_connectomes()` always adds `-tck_weights_in <sift>` before it switches on suffix-specific options (`:454-470`). Therefore recovery `count` is SIFT2-weighted, not an integer streamline count.

Direct file audit:

- 477/530 current `count` matrices are exactly array-equal to `fd_sum`.
- Those same 477 contain fractional nonzero values; nearly all their nonzero entries are fractional.
- The remaining 53 are consistent with a different/legacy recipe.

This contaminates:

- any “tract-count density” wording in the dashboard;
- `count` versus `fd_sum` comparisons;
- `count_invnodevol` interpretation;
- topology derived from a supposedly independent raw-count matrix;
- ML features that include both mislabeled duplicate matrices.

Required correction:

- Build true `count` without `-tck_weights_in` and without edge-value scaling.
- Build `fd_sum` with SIFT2 weights.
- Define and document whether `count_invnodevol` is raw count times inverse node volume or SIFT2-weighted strength times inverse node volume; use a non-misleading name for the latter.
- Recompute derived graph, network, edgewise, EDR, coupling, brain-age, and ML features after corrected matrices are written.
- Add an automated assertion: count values must be integer-valued and `count` must not be identical to `fd_sum` for an entire cohort.

### P0-D. Mixed recipes and incomplete provenance

The dense cohort’s manifest records:

- radial 2: 471
- radial 4: 59

The full 648 rows are radial 2: 491 and radial 4: 157. The dense data have mean density .7527 for radial 2 and .6851 for radial 4 (Mann–Whitney p=2.32e-11). Group-by-radius counts are not significantly imbalanced, but the technical effect on topology is large enough to require harmonization or explicit adjustment.

Recipe tags among good subjects are `sota-noact` 339, `cl` 132, and untagged/legacy production 59. Their group distribution is borderline different (chi-square p=.058). Current analysis tables do not carry noACT/ACT, 3M/10M, assignment radius, registration choice, or streamlining cap as model covariates.

Provenance is not sufficient:

- The 648-row master manifest has mostly blank `reg_ncc`, `label_survival`, and `best_recipe` fields.
- There are 141 `SC_AAL_*_generation_provenance.json` files, but they describe the older 170-node, low-density ACT products, not the current `SC_AAL166_*` repaired matrices. Example: `/data/derivatives/connectomes/SC_AAL_305_S_6877_I1478941_generation_provenance.json` records 170 nodes, density .009, 3M ACT, and a “not batch-approved” decision.
- Only nine `result.json` files remain under `/data/derivatives`, while `docs/DECISIONS.md:30-33` records deletion of approximately 2.9 TB of recovery run roots. Promoted matrices survive, but complete run-level reconstruction does not.
- `subject_manifest.py` infers radius from recovery membership rather than authoritative per-output metadata, so it can become stale after reassignment.

Required correction:

- Choose one primary tractography/assignment recipe and rebuild every primary-analysis subject with it.
- If a complete rebuild is temporarily impossible, stratify by recipe and include recipe variables in sensitivity models; do not pool silently.
- Emit one immutable JSON sidecar per matrix set containing exact input hashes, DTI/T1 IDs/dates, commands, tool versions, ACT/noACT, streamline target/actual count, radius, response files, SIFT2 parameters, registration route/score, QC result, timestamp, and output hashes.
- Replace the placeholder Snakemake workflow with an executable Snakemake/Nextflow DAG and a locked environment/container.

## 5. Statistical audit

### 5.1 Cohort confounding

Dense-cohort phase x diagnosis:

| Group | ADNI 2 | ADNI 3 | ADNI 4 |
|---|---:|---:|---:|
| AD | 33 | 44 | 1 |
| CN | 3 | 242 | 6 |
| MCI | 1 | 194 | 6 |

Chi-square p=5.31e-37. Sex also differs across groups (p=.0225). Site, scanner model, acquisition protocol, gradient count, and field strength were not included in the current main network models.

This matters because diffusion metrics and tractography are highly protocol-sensitive. Excluding phase as an ML input does not remove confounding when phase/protocol is encoded in the image-derived features.

Required primary model:

- pre-specified CN-MCI, MCI-AD, and CN-AD contrasts;
- age, sex, phase, site/scanner, field strength, gradient directions, motion/eddy QC, DTI-T1 interval, recipe, assignment radius, streamline count, and density/QC covariates where identifiable;
- ADNI3-only sensitivity, because it is the only phase with reasonable cross-group overlap;
- harmonization such as ComBat only inside training folds for prediction, with diagnosis and biologically important covariates preserved;
- random-effects/multilevel treatment of site where sample size supports it.

### 5.2 Unadjusted claims versus adjusted results

The network tables report Brunner-Munzel/Cliff-delta pairwise results and a permutation group effect adjusting for age, sex, and phase. The latter is implemented by permuting group labels while covariates remain fixed (`connectome_analysis/analysis_stats.py:195-235`). This is not a Freedman-Lane residual permutation and has questionable exchangeability under the extreme phase-group imbalance.

Even with that limitation, the project’s own adjusted results materially contradict the novelty narrative:

| Headline measure | Unadjusted CN-AD q | age/sex/phase permutation p |
|---|---:|---:|
| Functional Limbic AD | 6.28e-10 | .930 |
| Functional Limbic MD | 1.16e-9 | .530 |
| Functional Limbic RD | 4.44e-9 | .158 |
| Functional Limbic FA | 8.76e-5 | .005 |
| Functional DMN FA | 1.20e-10 | .001 |
| Visual nodal efficiency | .025 | .060 |
| Limbic nodal efficiency | .033 | .092 |

For functional network microstructure, 39/40 CN-AD tests have q<.05, but only 10/40 have permutation p<.05 and only 9 satisfy both. For anatomical microstructure, 35/36 have q<.05, but only 11/36 have permutation p<.05 and 10 satisfy both. This pattern is consistent with substantial phase/protocol confounding.

The current novelty report nevertheless calls the limbic result high confidence and describes AD/MD/RD as the primary cross-metric pattern (`/data/derivatives/qc/analysis_cohort/20_findings/novelty_report.md:16-27`). That claim must be withdrawn until a harmonized, covariate-valid rerun confirms it.

### 5.3 Multiplicity and analytic flexibility

- The findings catalog contains 210 “significant” findings, but FDR is applied separately within many metric/network/family tables rather than across a pre-specified primary family.
- Network “affectedness” is a ranking of CN-AD effects, not an independent validation.
- The age catalogue tests pooled cross-sectional correlations and many age-by-group interactions; “no significant interaction” is not proof that an effect is identical across diagnoses.
- Registration-route selection optimizes the same matrix-density property later analyzed, adding technical selection flexibility.

Required correction: define a small primary hypothesis family and correction scope before rerunning. Put all other node/network/edge/EDR results in an explicitly exploratory supplement with hierarchical FDR or permutation max-stat correction.

### 5.4 NBS/edgewise analysis

`connectome_analysis/analysis_edges.py:188-243` implements a component-mass, supra-threshold NBS-like procedure at |t|>=3.1. The refresh runner defaults to only 200 permutations (`apps/connectome_dashboard/refresh_connectome_dashboard_data.py:39`, `:245`). The current output has no component with p<.05; the closest is CN-vs-AD p=.0547, followed by CN-vs-MCI p=.0896.

The output itself correctly calls this NBS-like rather than canonical TFNBS (`analysis_edges.py:307-313`). The manuscript must not claim a significant NBS component. A final run should use at least 5,000-10,000 covariate-aware permutations and a validated NBS/TFNBS implementation.

### 5.5 Brain age

The provisional structural brain-age model has N=529, overall R2=.0095 and MAE=6.70 years. AD-only R2 is -.097. BAG and age-corrected BAG have adjusted permutation p=.10. This is not a clinically useful or validated brain-age model and supplies no current group finding.

### 5.6 Diagnostic ML

Positive implementation details:

- Diagnosis/statistic columns are screened out (`analysis_ml_diagnostics.py:173-202`).
- Imputation and supervised mutual-information feature selection occur inside nested CV pipelines (`:600-720`, `:828-846`).

Remaining issues:

- Outer and inner folds are random diagnosis-stratified folds, not site/phase-held-out folds (`analysis_ml_diagnostics.py:817-831`).
- Full-cohort missingness, sparsity, variance, and correlation filtering occurs before CV (`:497-585`). Although unsupervised, this still uses held-out feature-distribution information and should be folded into the pipeline.
- Technical acquisition/recipe signals remain in structural features, so random CV can exploit phase/site artifacts.
- 1,391 features are modeled in N=530 before supervised selection; no external validation cohort exists.

Current best model is XGBoost: accuracy .549, balanced accuracy .505, macro AUROC .676. Extra Trees balanced accuracy is .503. These are exploratory and only modestly above a balanced three-class baseline. No clinical deployment claim is justified.

Required correction: group/nested CV by site/scanner/phase, fit every preprocessing/harmonization step inside each training fold, report repeated confidence intervals and calibration, and reserve a truly untouched external or temporal test set.

### 5.7 MMSE/CDR prediction

The primary MMSE analysis uses only 62 subjects after outlier removal, up to 1,391 inputs, and a clinical target within a permissive 730-day window. Best primary extra-trees R2=.033, while most models have negative R2. Group-specific R2 values are strongly negative. The all-history sensitivity mixes remote clinical status and cannot be a primary result.

Primary CDR classification has N=65 with four classes and minimum class count 7. Balanced accuracy ranges approximately .32-.40. This is underpowered and not suitable for a paper headline.

Required correction: same-visit or tightly scan-aligned outcomes, no diagnosis-group-based target outlier deletion in the primary analysis, smaller pre-specified feature sets, appropriate ordinal methods for CDR, and external validation.

### 5.8 Mediation

Seven cross-sectional chains were run with 1,000 bootstrap samples; four have bootstrap intervals excluding zero. There is no multiplicity correction. One “proportion mediated” is 115%, indicating inconsistent mediation/suppression rather than a simple mechanistic chain. Diagnostic outcomes use an approximate binary-logit mediation.

These analyses cannot support causal or temporal mediation. They should be labeled exploratory association decompositions, corrected for seven tests, and omitted from the main causal story unless replicated longitudinally.

## 6. Network mapping and coupling terminology

### 6.1 Functional-network approximation

The primary atlas is anatomical AAL3. The functional Yeo-7 assignment is a manual/heuristic crosswalk:

- `atlas/AAL/AAL3_network_mapping_notes.md:23-27` explicitly says the mapping is approximate and parcels can straddle networks.
- Ambiguous assignments are listed at `:29-37`.
- Network sizes are unequal: Dorsal Attention 4 nodes, Frontoparietal 8, Limbic 18, DMN 20, Subcortical 38, etc. (`:45-56`).
- Brainstem and small nuclei have low edge support (`:39-42`).

The “limbic versus DMN” ranking is mapping-dependent because hippocampus/parahippocampus are assigned to DMN, while amygdala/OFC/temporal pole are assigned to Limbic (`:36`, `:49`, `:52`). This is especially important because the current novelty statement is precisely a Limbic-over-DMN ranking.

Required correction: make exact anatomical systems the primary AAL3 grouping, validate the functional crosswalk using parcel-overlap with a standard Yeo atlas, report parcel overlap/uncertainty, and repeat findings with a native functional parcellation such as Schaefer and at least one alternative anatomical atlas.

### 6.2 Coupling mislabeling

The refresh runner calls `run_live_structure_microstructure_coupling()` (`apps/connectome_dashboard/refresh_connectome_dashboard_data.py:227-235`). The novelty report’s measured quantity is “nodal efficiency vs FA” and explicitly admits that it is intra-structural (`novelty_report.md:46-57`). Nevertheless, the section title says “Structure-function coupling”.

Correct terminology:

- **structural topology–microstructure coupling**, or
- **graph–DTI coupling**.

Do not compare its direction as if it were equivalent to SC-FC coupling. A future structure-function claim requires actual functional connectivity data, matched parcellation, explicit SC-FC coupling definition, motion QC, and independent validation.

## 7. What the current results actually support

At present, only a cautious exploratory summary is defensible:

1. The pipeline can generate dense 166-node AAL3 matrices for 530/648 selected subjects.
2. Current unadjusted summaries show widespread DTI group differences, especially FA and diffusivity measures, but most non-FA headline diffusivity effects are highly phase/covariate-sensitive.
3. Functional DMN FA, several FA-based anatomical systems, and frontal topology–microstructure coupling remain candidates because they survive the current age/sex/phase test; they are still not validated against site/scanner, visit mismatch, recipe, density, and refreshed QC.
4. No current NBS component is significant at p<.05.
5. Brain-age, MMSE/CDR prediction, and diagnostic ML are exploratory and not sufficiently accurate/validated for a headline claim.
6. Mediation is exploratory and cross-sectional.
7. Novelty has not been proven. The current report itself states the literature search is not exhaustive (`novelty_report.md:9-14`) and contains only nine cited references (`:113-121`).

## 8. Prioritized corrective-action plan

### P0 — must complete before interpreting biological findings

1. **Freeze claims and snapshot inputs.** Mark dashboard/manuscript V1 as provisional. Create hashes and a read-only manifest of current files before rerun.
2. **Build visit-level multimodal pairing.** Select same-visit/nearest-date T1 for each DTI; pre-specify <=90-day primary and <=180-day sensitivity windows. Include dates, interval, IDs, phase/site/scanner/protocol.
3. **Choose one processing recipe.** Fix the 5TT/ACT alignment or formally choose a justified noACT protocol; standardize streamline target, radial assignment, response estimation, registration, and connectome definitions.
4. **Correct matrix semantics.** Rebuild true count separately from SIFT2 `fd_sum`; define every weight mathematically and add unit/invariant tests.
5. **Regenerate current QC.** Recompute image, registration, atlas, streamline-assignment, zero-row, physical-range, and matrix QC against the rebuilt artifacts. Enforce the new gate.
6. **Lock the analysis cohort.** Produce a flow table by diagnosis and exclusion reason. Do not use density alone.
7. **Rerun primary statistics.** Pre-specify contrasts/outcomes; adjust for acquisition and technical covariates; run ADNI3-only and paired-date sensitivity analyses; use correct multiplicity control and valid residual permutation/mixed models.

### P1 — required for a credible methods/results paper

8. Replace placeholder Snakemake files with an executable workflow; add container/environment lock and integration tests.
9. Emit per-subject/matrix provenance sidecars and preserve run manifests/logs even if large tracks are archived.
10. Validate FA/MD/RD/AD physical ranges and inspect every extreme/negative case at tensor-map and streamline-sampling level.
11. Re-run NBS/TFNBS with a validated package, >=5,000 permutations, and covariate-aware exchangeability.
12. Move every ML preprocessor inside grouped nested CV; hold out site/phase and obtain an external or temporal validation set.
13. Validate the AAL3-to-Yeo crosswalk and repeat network results under alternative mappings/parcellations.
14. Treat missingness/exclusion as a possible selection mechanism; compare included versus excluded subjects and consider inverse-probability/sensitivity analysis.

### P2 — paper consolidation after the corrected run

15. Conduct a protocol-driven systematic/scoping review with reproducible databases, queries, dates, screening log, inclusion/exclusion criteria, PRISMA flow, and claim-to-citation evidence table.
16. Choose one coherent primary story based only on robust corrected results. Candidate story structure:
    - tissue microstructure -> structural topology -> cognition/diagnostic stage;
    - one pre-specified primary network/system;
    - replication/sensitivity across atlas, phase, recipe, and pairing window.
17. Keep ML, EDR, mediation, brain age, and dense edgewise discoveries as secondary/exploratory unless they independently validate.
18. Draft the final manuscript only after the locked cohort and statistics are regenerated. Report null results and failed candidates, not only significant dashboard cards.

## 9. Required rerun deliverables and acceptance criteria

The correction phase should produce at least:

| Deliverable | Minimum acceptance criterion |
|---|---|
| `acquisition_pair_manifest.csv` | One row/series; DTI/T1 dates and IDs; interval; phase/site/scanner/protocol; deterministic selection rule |
| workflow DAG + lock/container | Clean canary rebuild from raw input; resumable; no manual hidden step |
| per-subject provenance JSON | Inputs/outputs hashed; exact commands/tool versions/recipe/QC |
| corrected 9-matrix set | Count integer and distinct from fd_sum; all shapes/symmetry/diagonal/finite tests pass |
| physical metric QC | 0<=FA<=1 or documented tensor correction; nonnegative diffusivity; outliers reviewed |
| spatial QC | Registration/atlas/mask overlays and quantitative thresholds; no unexplained zero nodes |
| cohort flow | Counts by group/phase/site and exclusion reason; missingness comparison |
| primary statistical analysis plan | Hypotheses, outcomes, covariates, contrasts, correction family, sensitivity analyses fixed before results |
| harmonized results package | Main + ADNI3-only + pairing-window + recipe/atlas sensitivity, with effect CIs |
| grouped ML validation | All transforms inside CV; site/phase grouped folds; calibration/CIs; external test if claimed |
| systematic literature evidence table | Search/query/date/screening trail; every gap claim linked to direct evidence |

Suggested go/no-go criteria for manuscript V-final:

- No primary subject outside the pre-specified DTI-T1 interval.
- One primary tractography/connectome recipe, or a validated harmonization design with no material recipe interaction.
- New QC gate applied to current outputs; no stale legacy status in the master table.
- Primary claims remain significant after acquisition/technical covariates and correction for the pre-specified hypothesis family.
- Direction and material effect persist in ADNI3-only and at least one atlas/mapping sensitivity.
- Novelty claim survives the systematic review and is phrased at the exact granularity supported by the search.

## 10. Environment and reproducibility snapshot

Verified installed versions:

- MRtrix3 3.0.7
- FSL FLIRT 6.0
- dcm2niix v1.0.20250505
- Python 3.12.11
- Streamlit 1.57.0
- numpy 2.1.3, pandas 2.3.2, scipy 1.15.3, statsmodels 0.14.5
- scikit-learn 1.5.2, networkx 3.5, nibabel 5.3.2, antspyx 0.6.3
- xgboost 2.1.4, torch 2.12.0, shap 0.51.0, tabpfn 8.0.3

No full project environment lock was found. `scforge/pyproject.toml` declares only numpy, pandas, and pyyaml, which is far short of the actual pipeline/analysis stack. `openpyxl`/`xlsxwriter` were absent in the inspected app environment, explaining CSV fallback for some exports.

## Final verdict

The strongest immediate contribution of this audit is not a new AD biomarker. It is the identification of a **multimodal visit-pairing failure plus a mixed-recipe/QC/provenance problem that can plausibly generate or amplify the current group effects**. Correcting those issues is the necessary next experiment.

After correction, the most promising biological candidates to retest are FA-centered DMN/anatomical effects and frontal structural topology–microstructure coupling, not the currently advertised unadjusted Limbic AD/MD/RD ranking. Under SL-D01, any final paper should let the corrected, phase/site/timing/QC-aware available-data analysis determine the story rather than forcing the existing dashboard cards into one.
