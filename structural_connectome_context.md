# Structural Connectome Project — Current Context

**Status date:** 2026-07-18 10:30 UTC  
**Project root:** `/home/ec2-user/exp`  
**Status:** working exploratory platform; **not manuscript-ready and not a validated biomarker**

## Purpose and source-of-truth rule

This is the concise current implementation context for the structural-connectome project. It supersedes the May 25 root context; the historically important failed 639-subject AAL3 state is retained in the pipeline audit and chat-history reconstruction linked below.

Use this document to orient a reimplementation, then use the linked audits for evidence and exact details:

- [Superlist](research_audit/SUPERLIST.md) — the authoritative execution checklist, status tracker, dependencies, evidence gates, and human decisions.
- [Pipeline and results audit](research_audit/pipeline_and_results_audit.md) — code/data/service audit, release blockers and acceptance criteria.
- [Chat-history reconstruction](research_audit/chat_history_context.md) — Codex/Claude chronology, superseded snapshots and decision provenance.
- [Independent statistics summary](research_audit/outputs/audit_statistics_summary.md) — cohort/acquisition sensitivity findings.
- [Audit cohort manifest](research_audit/outputs/cohort_audit_manifest.csv) — subject-level audit table.
- [Literature evidence map](research_audit/literature_evidence_map.csv) — verified prior-art map; not yet a completed systematic review.
- [Closed-world analysis design v1.1](research_audit/closed_world_analysis_design_v1.md) — SL-H03-approved fixed-data estimand, measured common support, timing/protocol/T1-source/QC sensitivities, validity floor, missingness, exchangeability, and claim limits; not yet the frozen SAP.
- [Immutable local-input lock](research_audit/outputs/available_data_content_lock_v2/available_data_content_lock_validation_v2.json) — PASS content lock for all selected DTI/T1 source files.
- [Fixed-data feasibility memo](research_audit/outputs/fixed_data_feasibility_power_memo_v1.md) — outcome-blind common-support, estimability, design-rank, attrition-stress, power, and workflow-contract audit.
- [SL-H03-C1 approval](research_audit/decisions/sl_h03_c1_approval_20260718.md) — approved DESIGN-ONLY cohort/estimand decision; it authorizes SL-P0-11–14 implementation but no imaging execution.
- [Protocol-composition figure](research_audit/figures/protocol_composition_by_group.png) and [sensitivity heatmap](research_audit/figures/key_metric_sensitivity_heatmap.png).

Important project separation: `/home/ec2-user/sabeesh/context.docx` is the unrelated pulmonary-embolism project context. It was inspected only to prevent project confusion and **was not overwritten**. Structural-connectome context belongs under `/home/ec2-user/exp`.

## Objective

Build reproducible, anatomically valid subject-level diffusion-MRI structural connectomes and determine which effects robustly distinguish CN, MCI and Alzheimer disease after controlling acquisition, registration, processing-recipe and cohort-selection effects. A publication claim is allowed only after the corrected pipeline, locked cohort, prespecified statistics and systematic novelty audit agree.

The intended scientific chain is:

`DTI/T1 acquisition -> tissue microstructure -> structural topology -> diagnostic/cognitive association`

ML, LR/SR, EDR, brain age, mediation and dashboard story cards are secondary unless independently validated.

## Verified current facts

### Workspace and live system

- Live data are reached through `/home/ec2-user/exp/data -> /data`.
- Main derivatives: `/data/derivatives`.
- Production pipeline code: `connectome_pipeline/`.
- Current recovery implementation: `scripts/scforge/live/run_sc_route_sota.py`.
- Analysis package: `connectome_analysis/`.
- Dashboard: `apps/connectome_dashboard/connectome_app.py`.
- Dashboard refresh: `apps/connectome_dashboard/refresh_connectome_dashboard_data.py`.
- The Streamlit dashboard was live on 2026-07-18 behind nginx/auth; no MRtrix or analysis-refresh process was running during the audit.
- Live QC status is newer than the inferential analysis. The dashboard analysis tree was last rebuilt on 2026-06-17 and its refresh metadata still says `snapshot_mode=provisional`.
- The [historical provisional v1 snapshot](research_audit/snapshots/historical_provisional_v1/summary.md) freezes 6,047 canonical files with SHA-256 and records a non-destructive S3 warning: 24 matrix objects across three locally reprocessed subjects are stale or absent in S3.
- The historical candidate `scforge/workflow/Snakefile` resolves a 54-job one-subject PASS-only DAG and remains frozen as `connectome-v2.0.11-canary-candidate`. It is **not executable against the approved SL-D01 cohort contract** and is preserved as historical implementation evidence; no real image-processing run has been authorized or executed.
- The supported real-run entry point is `scforge/workflow/run_connectome_v2.py`. Direct ad hoc execution is not part of the corrected contract.

### Workflow implementation state and reopened contract

- Historical v2.0.11 scientific contract validation: 47/47 PASS, zero warnings.
- Historical v2.0.11 locked host/environment/workflow-source validation: 173/173 PASS, zero warnings.
- Historical v2.0.11 SCForge test discovery: 44/44 PASS, including six provenance/ledger tests.
- Historical schema-only one-subject workflow dry run: PASS, 54 jobs; this proved candidate DAG resolution only, not compatibility with SL-D01, image validity, attrition behavior, or runtime success.
- The workflow explicitly locks CPU Eddy, MRtrix/FSL/ANTs/convert3d/SS3T executables, dependency artifacts, the 24-file workflow source bundle, MNI/AAL3 resources, lmax=6/direction eligibility, and all nine matrix definitions.
- The old Draft 2020-12 provenance/run-ledger schemas are PASS-only: they require every unit and all nine matrices to pass and constrain `fail=0`/`excluded=0`. That conflicts with approved attempt-all, mixed terminal-state, and outcome-specific failure/NA semantics. SL-P0-11–14 are therefore reopened; v2.0.11 must not be used for a canary.
- A provenance audit found and corrected the seed-identity mismatch: the executable RNG seed now derives from `dti_series_uid|recipe_id`, and the actual seed token, step size, and requested/actual streamline counts are recorded.
- No replacement T1 was downloaded; no imaging, S3 write, dashboard refresh, or production-tree mutation occurred during this implementation work.
- The named AWS `aml` principal passed verification at 08:29 UTC. A read-only search found no fresh ADNI/LONI exact-date export, but this is no longer a blocker: SL-D01 fixes the analysis scope to images already available locally.
- Exact-date candidate discovery freezes 4,015 locally known `Type=Original` T1-like IDs and the contingency intake/pairing gate passes 39 tests. That work is preserved as source-audit evidence; external replacement acquisition is not pursued unless SL-D01 is explicitly reopened.
- SL-D01 retains all 530 current local pairs. Their timing, cross-phase, acquisition, and QC limitations become explicit covariates/sensitivity strata; they are not relabeled as visit-matched.
- Diagnosis/DTI identity reconciliation is complete: 530/530 diagnoses resolve at the exact selected Original DTI ID/date; all 32 processed-dashboard versus Original-pipeline ID mismatches are concordant and resolved; harmonized groups are CN 251, MCI 186, AD 78, and SMC 15. The primary CN/MCI/AD ceiling is 515, with SMC retained separately. No local longitudinal/biomarker adjudication table was found, so the source label remains the supplied ADNI image-catalog `Research Group`.
- SL-H01 is approved. The SL-D01 pair manifest is PASS: 530/530 local DTI/T1 series resolve uniquely; intervals are 194 <=90 days, 3 at 91–180 days, and 333 >180 days; 334 are same-phase and 196 cross-phase. Corrected-rerun QC is deliberately `not_run` for 530/530. Source-side T1 QC is unavailable and 225 current T1s are catalogued as processed rather than Original.
- SL-P0-09 is DONE and independently verified. The immutable lock covers 530 pairs, 1,060 series, 451,115 unique files, and 99,066,365,832 bytes (92.262743 GiB), with zero duplicate-content groups, missing/extra files, or source mutations. Locked-manifest SHA-256 is `1e47d263c70a8230ab0b39651b652333d03e31c63308c9f18dfa008f804689f5`; validation SHA-256 is `70619a8c951c4c93416c9280de25b84ab62383119c26ddd39eaff201b89688d2`.
- SL-P0-10 is DONE and independently verified without opening biological outcomes. Aggregate exploratory CN/MCI/AD modeling is feasible only with a parsimonious acquisition representation. CN–MCI has measured common support; AD contrasts have limited cell depth. ADNI 2 is exactly aliased with GE 3 T/41-direction protocol in the observed design. The feasibility validation SHA-256 is `4fd319a3edc2897552a4f9e1415bbf95b06dc42bbfa9b1816850311940e6cdde`.
- SL-H03-C1 is approved, DESIGN ONLY. It fixes all 530 attempts/attrition rows, a 515 CN/MCI/AD outcome-specific contrast ceiling, 15 separate SMC cases, and primary participant-level upper-triangle unweighted `fa_mean` over `count>0` support. It authorizes the SL-P0-11–14 implementation/refreeze but not canary or full imaging execution.

### Current cohort and artifacts

Current manifest: `/data/derivatives/qc/sc_matrix_qc/subject_manifest.csv`.

| Layer | Total | AD | MCI | CN |
|---|---:|---:|---:|---:|
| Selected pipeline cohort | 648 | 100 | 241 | 307 |
| Density-good main directory | 530 | 78 | 201 | 251 |
| Mid band | 5 | — | — | — |
| Low band | 113 | — | — | — |

- All 530 main `count` files are 166x166, finite, symmetric and zero-diagonal.
- Their current density range is 0.60095–0.95239; median 0.74133 and mean 0.74594.
- There are 530 files each for `count`, `count_invnodevol`, `fd_sum`, `len_mean` and `invlen_mean`.
- There are 529 files each for FA, MD, RD and AxD. `168_S_6938_I1444126` is structural-only and lacks all four diffusivity matrices.
- The explicit current filename contract is `SC_AAL166_<series>_<weight>.csv`. The earlier `SC_AAL_...` 169/170-node convention is legacy.

### Current matrix validity limits

Density-good does not mean analysis-valid:

- 190/530 current `count` matrices have at least one fully zero row; maximum 28.
- 415/529 diffusivity matrices have at least one zero row; median 8 and maximum 164.
- `invlen_mean` has median 54 zero rows and maximum 164.
- Three FA matrices exceed the physical bound 1; maximum 1.19158.
- Negative values occur in 5 MD, 10 RD and 2 AxD subjects.
- The June master table carries 363 `sc_matrix_qc_include=False` rows from a May QC table. Those flags predate repair and must be regenerated; they must neither be ignored nor blindly enforced.

### Current pipeline meaning

“BATMAN” is not a separate installed processing stage. Current code uses “BATMAN-style” to describe the single-phase-encoding Eddy configuration in `connectome_pipeline/dwi_eddy.py`. MRtrix supplies the actual preprocessing, tractography and `tck2connectome` operations.

Current operational stages are:

1. DWI selection, DICOM conversion, gradient/shell/volume validation.
2. `dwidenoise` and `mrdegibbs`.
3. `dwifslpreproc -rpe_none` / Eddy and bias correction.
4. T1 selection, T1/B0 registration, 5TT/GMWMI.
5. response estimation, FOD normalization and tensor FA/MD/RD/AxD maps.
6. tractography, SIFT2 and streamline sampling.
7. AAL3 MNI-to-native/DWI transform with contiguous 166-node indexing.
8. endpoint assignment and `tck2connectome` matrix generation.
9. QC/promotion, cohort locking, statistical analysis and dashboard rendering.

The current recovery recipe is heterogeneous: radial-2 and radial-4 assignment, legacy ACT and newer forced-noACT routes, 3M and 10M tractograms, and multiple registration/recovery tags are mixed in the 530-subject set.

### Critical implementation defect: `count` is not raw count

The recovery code adds SIFT2 weights before branching by matrix suffix. Consequently:

- 477/530 current `count` matrices are exactly equal to `fd_sum`.
- The same 477 contain fractional nonzero values.
- Current “tract count” wording is wrong for those files, and analyses treating `count` and `fd_sum` as independent feature families duplicate the same signal.

This defect requires matrix regeneration, not relabeling alone.

### Critical acquisition defect: DTI and T1 are not visit-matched

The pipeline pairs DTI and T1 by subject identity and accepts a matching T1 without enforcing visit/date correspondence.

For the current 530-subject dense cohort:

- 196/530 (37.0%) DTI–T1 pairs come from different ADNI phases.
- Absolute age gap: mean 3.20 years, median 2.10, 90th percentile 8.41, maximum 13.8.
- Only 197/530 (37.2%) are same phase and within 0.5 years.
- Exact current-file dates refine this to 194/530 within 90 days, 197/530 within 180 days, and 333/530 beyond 180 days.
- The 330 provisional replacement IDs have 0/330 coverage in the available exact-date MRI master and 0/330 local raw files. A scientifically clean replacement route would require source-complete date/QC accounting over the 4,015-ID universe, but SL-D01 does not pursue that route.
- The mismatch distribution differs across diagnostic subgroups.

Because the T1 drives anatomy, 5TT/GMWMI and atlas registration, the interval is a biological and technical validity limitation. The corrected run will use the available local pairs, but its results remain exploratory and must be tested across timing/protocol/QC sensitivities; interval adjustment cannot recreate missing contemporaneous anatomy.

### Statistical and acquisition confounding

- Diagnosis is strongly associated with ADNI phase in the reconciled 515-person CN/MCI/AD ceiling: AD 33/44/1, CN 3/242/6, and MCI 0/180/6 across ADNI 2/3/4; Cramér's V=0.410 and p=3.46e-37. ADNI 2 is also exactly identical to the GE 3 T/41-direction protocol indicator in these data, so phase plus protocol is rank-deficient.
- Sex also differs across groups (p=.0225).
- Site, scanner, protocol, gradient directions, DTI–T1 interval, assignment radius and recovery recipe are not fully controlled in current headline models.
- Assignment radius materially affects density: mean 0.753 for radial-2 versus 0.685 for radial-4; Mann–Whitney p=2.32e-11.
- Current random nested CV can learn acquisition/phase effects because it is not site/phase-held-out.

## Candidate findings — not verified discoveries

The July audit invalidates the current “high-confidence Limbic diffusivity” headline as written.

| Candidate | Current evidence | Status |
|---|---|---|
| Limbic AxD/MD/RD elevation | Very small unadjusted q-values, but current age/sex/phase permutation p=.930/.530/.158. | Withdraw as headline pending corrected available-data rerun and timing/protocol/QC sensitivities. |
| Limbic FA reduction | Unadjusted CN–AD q=8.76e-5; current age/sex/phase permutation p=.005. | Retest candidate only; still exposed to site/protocol/recipe/QC confounding and mapping uncertainty. |
| DMN FA reduction | Unadjusted q=1.20e-10; current adjusted permutation p=.001. | Promising retest candidate, not confirmed. |
| Visual/Limbic nodal efficiency | Unadjusted q=.025/.033; adjusted permutation p=.060/.092. | Does not currently survive the project’s adjusted test. |
| Frontal topology–microstructure coupling | A current adjusted candidate; the measure is graph topology versus FA, not functional connectivity. | Retest under corrected matrices/covariates; call topology–microstructure coupling. |

Current null/weak results must be preserved:

- No current NBS-like component reaches p<.05; closest CN–AD p=.0547 with only 200 permutations.
- Brain age is non-predictive: overall R²=.0095, MAE=6.70 years; adjusted BAG p=.10.
- Best current diagnostic model: accuracy .549, balanced accuracy .505, macro-AUROC .676. Exploratory only.
- Primary scan-aligned MMSE: n=62, best R²≈.033. Primary CDR: n=65, best accuracy≈.40.
- Seven cross-sectional indirect-association chains were tested; four bootstrap intervals exclude zero, without multiplicity correction. They do not establish mediation, mechanism or compensation.
- The approximate AAL3-to-Yeo mapping makes any “Limbic above DMN” ranking mapping-dependent; hippocampal/parahippocampal parcels are assigned to DMN while amygdala/OFC/temporal-pole parcels are assigned to Limbic.

## Literature/novelty state

- Novelty is **not established**.
- The previous OpenAlex layer retrieved many records but acknowledged noisy, non-exhaustive coverage; the dashboard report contains only nine cited references.
- The July [literature evidence map](research_audit/literature_evidence_map.csv) already documents extensive prior art for CN/MCI/AD structural topology, limbic/default-mode effects, multiple diffusion weights, classification, mediation, compensation, NBS, rich-club analysis and multimodal prediction.
- A generic claim that DTI/connectome measures differentiate CN/MCI/AD is not novel.
- Any future gap must be stated at exact granularity: acquisition and atlas, metric definition, network, disease stage, covariates, validation design, direction and effect.

## Release gates and fixed constraints

### P0 — biological interpretation is blocked

1. DTI/T1 visit mismatch.
2. SIFT2-weighted matrices mislabeled as raw `count`.
3. Stale post-repair QC and density-only cohort inclusion.
4. Mixed ACT/noACT, 3M/10M, radial-2/radial-4 and registration recipes with incomplete provenance.
5. Diagnosis–phase/protocol/site confounding and invalid/weak exchangeability in current permutation analysis.
6. Physical tensor and zero-node failures.
7. Provisional June analysis snapshot and no locked manuscript release.
8. The v2.0.11 candidate loader/DAG rejects >180-day and SMC rows, assumes normalized inputs that do not yet exist, and cannot publish a mixed PASS/FAIL attrition ledger. SL-P0-11–14 must be versioned and refrozen before canary execution.

### P1 — reproducibility and validation are blocked

1. The corrected DAG/environment/provenance implementation has not yet been exercised on an approved real subject; image validity and resumability remain canary questions.
2. The 530 available DTI–T1 pairs are content-locked with exact intervals and limitation flags. The remaining reproducibility work is a checksum-bound normalized workflow projection and mixed-state execution/attrition contract; contemporaneous replacements will not be acquired under SL-D01.
3. No external or site/phase-held-out validation.
4. Approximate functional-network crosswalk without overlap-based validation or alternate-atlas sensitivity.
5. Literature review not yet protocol-driven/systematic.

## Corrected reimplementation contract

Do not reproduce the current output tree blindly. Reimplement the corrected design below.

### 1. Acquisition manifest

Create one immutable row per chosen DTI series with:

- subject and series/image IDs;
- DTI and T1 dates/visits/phases and interval;
- diagnosis at scan, site, manufacturer/model, field strength, protocol and gradient directions;
- deterministic selection reason and exclusion reason.

The fixed local-source manifest contains the one available selected DTI/T1 pair per subject and no replacement acquisition will be performed. All 530 rows are attempted and retained in technical attrition. Timing is an explicit study limitation and sensitivity dimension, not a source-exclusion rule. Diagnostic contrasts have an outcome-specific pre-QC ceiling of 515 CN/MCI/AD; 15 SMC rows remain separate and descriptive. The full available-data result is exploratory; <=90-day and <=180-day analyses are nearly identical timing restrictions, not independent replications.

### 2. One primary image-processing recipe

- Validate gradients and DWI geometry at every conversion/preprocessing step.
- Lock Eddy options and phase-encoding derivation.
- Register the available subject T1 to B0 and quantify/visualize overlap; retain its acquisition interval in provenance and QC.
- Make ACT/noACT a prespecified, justified protocol rather than an opportunistic mix. If both are necessary, define strata before analysis.
- Lock tractography algorithm, streamline target, cutoff, length range, seed policy, SIFT2 parameters and endpoint-assignment radius.
- Preserve nearest-neighbour/generic-label AAL resampling and stable 1..166 matrix indices.

### 3. Mathematically distinct matrix outputs

Required matrices and invariants:

- `count`: raw integer streamline count; no `-tck_weights_in`.
- `fd_sum`: sum of SIFT2 weights; fractional values allowed.
- `count_invnodevol`: define explicitly from raw count and node volume, or rename if using SIFT2 strength.
- `len_mean`, `invlen_mean`: specify handling of absent/zero edges and units.
- `fa_mean`, `md_mean`, `rd_mean`, `ad_mean`: streamline-sampled tensor metrics with physical-range QC and units.

Automated tests must reject a cohort where `count == fd_sum`, non-integer count values, non-finite/asymmetric matrices, nonzero diagonals, invalid dimensions, FA outside [0,1], negative diffusivity, or unexplained zero nodes.

### 4. Current spatial/matrix QC gate

Regenerate QC from the rebuilt 166-node artifacts. Gate on:

- DTI/T1 and atlas registration quality;
- label survival and parcel volumes;
- mask/5TT/FOD overlap;
- endpoint-assignment fraction;
- dimensions, symmetry, diagonal, finite values and physical ranges;
- zero-row policy by weight;
- required matrix availability and provenance completeness.

The locked cohort must be selected by the new QC decision, not density alone. Report flow and exclusion reasons by diagnosis, phase, site and recipe.

### 5. Prespecified analysis

- Define a small primary hypothesis family before rerun.
- Prespecify CN–MCI, MCI–AD and CN–AD contrasts.
- Use one parsimonious frozen acquisition representation rather than simultaneous redundant fixed effects. The current pre-H03 recommendation uses `protocol_key` plus T1 source class, with outcome-specific rank/condition checks and site/scanner partial pooling only when convergent. Phase, manufacturer, field strength, gradient directions and scanner model are alternative/support sensitivities.
- In the full cohort, model age, sex, nonlinear DTI–T1 interval and diagnosis-by-interval interaction plus the small frozen QC set. Do not fit nonlinear gap or diagnosis-by-gap terms in <=90/<=180 subsets because those windows have only 2/5 distinct gaps and 193 zero-day pairs each.
- Use valid residual permutation or multilevel models; preserve exchangeability blocks.
- Run ADNI3-only, pairing-window, recipe, atlas/network-map and inclusion-threshold sensitivity analyses.
- Apply correction across the declared primary family. Put node/edge/EDR discovery in an exploratory supplement with hierarchical FDR/max-stat control.
- Use ≥5,000–10,000 covariate-aware permutations for final NBS/TFNBS.

### 6. Prediction and dashboard

- Fit imputation, filtering, harmonization and selection entirely within grouped nested CV.
- Hold out site/scanner/phase and reserve an external or temporal test set for any prediction claim.
- Version dashboard snapshots independently from live QC status.
- Keep a visible `provisional / not for inference` banner until go/no-go criteria pass.
- Dashboard language must use AxD for axial diffusivity, topology–microstructure coupling for graph/DTI correlations, and exploratory indirect association for cross-sectional path models.

### 7. Reproducibility outputs

The historical v2.0.11 candidate provides write-once schema-validated PASS sidecars and a PASS-only ledger, but that is insufficient for the approved closed-world design. Before any real canary, a new recipe must accept all 530 technical attempts, create deterministic normalized/staged inputs from locked DICOM or single-NIfTI source classes, never fabricate missing UID/phase-encoding/readout metadata, and emit immutable PASS/PARTIAL/FAIL plus outcome-specific validity/NA records. A 530-row schema/dry run must complete with mixed synthetic terminal states before imaging execution.

## Next actions, in order

Completed prerequisites are tracked in the [superlist](research_audit/SUPERLIST.md): diagnosis/identity reconciliation, SL-H01, the validated and content-locked 530-pair SL-D01 source release, independently verified fixed-data feasibility, and design-only SL-H03-C1 are DONE. SL-P0-11 is active.

1. Refactor and refreeze SL-P0-11–14 under a new recipe: all-530 source normalization, independent failure capture, outcome-specific validity/NA, mixed-state provenance/attrition ledger, tests, locks, and a 530-row dry run.
2. Construct the blinded balanced 12–24-subject canary only after the revised contract passes; imaging execution remains separately gated.
3. Obtain explicit authorization, then execute and review canary spatial overlays, tensor/label/endpoint QC, provenance, runtime, and storage at SL-H04.
4. Scale the corrected workflow and regenerate the QC-locked cohort only after canary approval.
5. Freeze the statistical analysis plan before viewing corrected group effects, then run the prespecified available-data and sensitivity analyses.
6. Use corrected results to retain, qualify, or reject every biological candidate.
7. Complete the protocol-driven review and exact claim-to-prior-art matrix for the surviving claim.
8. Draft manuscript V-final only after the technical, statistical, replication, and novelty gates pass.

## Go/no-go criteria for manuscript V-final

- Every pair has an exact interval and declared timing stratum; no long-gap pair is described as contemporaneous, and claims are calibrated to timing sensitivities.
- One primary processing recipe, or a validated prespecified stratified/harmonized design.
- Raw count is integer and demonstrably distinct from SIFT2 `fd_sum`.
- Current QC is enforced; no stale May decision controls the cohort.
- All primary claims retain direction and material effect after covariates, multiplicity and ADNI3/pairing/recipe/atlas sensitivities.
- Prediction claims use grouped validation and an untouched external/temporal test.
- Novelty survives the systematic review at the exact claimed granularity.
- Null results and failed candidates are reported alongside positive results.

## Current scientific verdict

The July audit’s strongest finding is technical: multimodal visit mismatch, mislabeled matrix semantics, mixed processing recipes and stale QC can plausibly generate or amplify the current group effects. The next research result must come from correcting those failures.

The leading biological retest candidates are FA-centered DMN/anatomical effects and frontal structural topology–microstructure coupling. The advertised unadjusted Limbic AxD/MD/RD ranking, ML, brain age, NBS and causal-compensation stories are not current headline results.
