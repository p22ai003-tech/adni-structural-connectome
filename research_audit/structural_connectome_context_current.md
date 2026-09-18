# Structural Connectome Project — Current Context

**Status date:** 2026-07-19 23:32 UTC  
**Project root:** `/home/ec2-user/exp`  
**Status:** a validated output-only scientific release is available and manuscript drafting is paused at the user's direction; **the claim is still exploratory, not a validated biomarker, externally replicated result, or proven priority claim**

## Purpose and source-of-truth rule

This is the concise current implementation context for the structural-connectome project. It supersedes the May 25 root context; the historically important failed 639-subject AAL3 state is retained in the pipeline audit and chat-history reconstruction linked below.

Use this document to orient a reimplementation, then use the linked audits for evidence and exact details:

- [Superlist](research_audit/SUPERLIST.md) — the authoritative execution checklist, status tracker, dependencies, evidence gates, and human decisions.
- [Pipeline and results audit](research_audit/pipeline_and_results_audit.md) — code/data/service audit, release blockers and acceptance criteria.
- [Chat-history reconstruction](research_audit/chat_history_context.md) — Codex/Claude chronology, superseded snapshots and decision provenance.
- [Independent statistics summary](research_audit/outputs/audit_statistics_summary.md) — cohort/acquisition sensitivity findings.
- [Audit cohort manifest](research_audit/outputs/cohort_audit_manifest.csv) — subject-level audit table.
- [Literature evidence map](research_audit/literature_evidence_map.csv) — verified prior-art map; not yet a completed systematic review.
- [Evidence-complete author manuscript V2](research_audit/manuscript_author_ready_v2_full_FIXED.docx) and [Markdown source](research_audit/manuscript_author_ready_v2_full.md) — the current positive-centred biological-paper draft, with five main figures including a multivariate stage/system/site/transportability centerpiece and four main tables.
- [Full V2 supplement](research_audit/outputs/manuscript_author_ready_v2_full_supplement_FIXED.docx) and [integrated validation](research_audit/outputs/manuscript_author_ready_v2_full_validation.md) — 22 supplementary tables, five supplementary figures, three evidence notes, 38 references, and 33/33 release checks including the targeted novelty update, aggregate privacy and strict Word property-order validation.
- [Explicit V2 requirement trace](research_audit/outputs/manuscript_requirement_trace_v2/validation.md) — independently maps every user-requested tensor, global/nodal graph, coupling, edge-architecture, third-dimension, ML, summary-table, flow-chart and coherent-story requirement to an exact section/figure/table; passes 15/15 and verifies all ten linked figures exist.
- [Refined output-only release](research_audit/outputs/refined_output_release_v1/README.md) — the active user-directed deliverable: four visually inspected figure families in PNG/SVG/PDF, eight aggregate or anonymous source-data tables, an 11-row core result table and the complete 19-claim disposition ledger. A clean rebuild passes 16/16 checks and an independent replay passes 28/28 checks covering the exact builder-source hash, exact upstream statistical equality, 9,999-draw inference scope, independently recomputed stage-centroid geometry and 51 anonymous site deletions, exact core-table reconstruction, claim disposition, figure decoding/dimensions, table/artifact hashes and privacy. The builder invalidates stale independent records before any rebuild. It changes presentation only and contains no manuscript prose.
- [Targeted novelty citation-chain update](research_audit/outputs/two_axis_novelty_audit_v2/citation_chain_update_v2.md) — adds five verified 2024–2026 primary studies to the exact neighborhood (23 total), records the surviving differentiated conjunction, and blocks broad first/progression/pathology/mechanism claims.
- [Frozen V-final confirmation contract](research_audit/outputs/vfinal_confirmation_contract_v1/confirmation_contract.md) — prospectively fixes the corrected-output endpoints, support, model, wild-cluster/Holm family and result classifications; its 11/11 static pass authorizes no Phase B work.
- [Fail-closed V-final executor](research_audit/outputs/vfinal_confirmation_executor_v1/README.md) — implements preflight and hash-authorized aggregate-only execution for the frozen contract, passes eight implementation checks and four synthetic-data tests, emits no participant-level output, and has not been run on corrected study outcomes.
- [Journal-neutral submission readiness](research_audit/outputs/submission_readiness_v2/submission_readiness.md) — separates technical readiness from corrected-confirmation, author, IITJ and ADNI gates and records the conditional Communications Biology versus Network Neuroscience route.
- [GPU transition gate](research_audit/outputs/gpu_transition_gate_v1/gpu_transition_gate.md) — fail-closed current answer to whether Phase A is terminal and EC2 is safe to resize.
- [GPU capacity recommendation](research_audit/outputs/gpu_capacity_recommendation_v1.md) — measured CPU-Eddy runtime strata, local CUDA compatibility gate, practical G6 choices, wave counts, and storage constraints; sizing only until the transition gate passes.
- [Retry4 terminal Phase-A publication](/data/derivatives/scforge_v2/h04a_r1_recovery_20260719_retry4/publication/response_calibration_phase_a_completion.json) — exact 15-unit response-calibration completion with 15/15 PASS outcomes.
- [Authoritative retry4 V2 frozen calibration](/data/derivatives/scforge_v2/h04a_r1_recovery_20260719_retry4/frozen_calibration_retry4_v2/frozen_response_calibration_manifest.json) and [adapter attestation](/data/derivatives/scforge_v2/h04a_r1_recovery_20260719_retry4/frozen_calibration_retry4_v2/retry4_freeze_attestation.json) — diagnosis-blind pooled WM/GM/CSF response evidence for the next gated continuation.
- [Retry4 pre-tractography package V3](research_audit/outputs/h04a_r1_retry4_pretract_package_v3/validation.md) — PASS package for the exact same 15 units. Its 394-job dry run contains 392 bounded anatomical/registration/atlas/FOD/tensor/QC/review jobs plus one manifest freeze and one execution preflight. The user supplied exact approval `Approve SL-H04A-CAL` at 12:11:48 UTC; immutable release validation passed and the bounded live service started at 12:12:03 UTC.
- [Retry4 pre-tractography recovery1 package](research_audit/outputs/h04a_r1_retry4_pretract_recovery1_package_v1/validation.md) — PASS 16/16 package for the immutable original failure. It preserves scientific commands and parameters, changes only SS3T output checks plus native FSL/ANTs output declarations, binds 359 reusable files/23,594,113,929 bytes, and passes an exact 229-job residual dry run with no forbidden or unexpected rule. Exact gate `Approve SL-H04A-CAL-R1` was supplied at 13:30:13 UTC and the bounded service started at 13:30:22 UTC.
- [Retry4 pre-tractography Recovery2 completion](/data/derivatives/scforge_v2/h04a_r1_recovery_20260719_retry4/publication/pre_tractography_canary_recovery2_completion.json) — terminal FAIL after the exactly approved 166-job scope. All 15 fixed FSL-to-ITK conversions and atlas resamples completed, but 15/15 units failed at atlas/5TT validation; no tractography or matrix work ran.
- [Retry4 pre-tractography Recovery3 diagnostics](research_audit/outputs/h04a_r1_retry4_pretract_recovery3_diagnostics_v1/recovery3_diagnostics.json) and [package validation](research_audit/outputs/h04a_r1_retry4_pretract_recovery3_package_v1/validation.json) — PASS diagnosis-blind correction evidence. Linear-plus-[0,1]-clip 5TT resampling passes 15/15 scratch checks; three representative 5TT-masked MNI-to-native-T1 registrations retain all 166 labels with centroid error <=5 mm. The exact 166-job package excludes preprocessing/Eddy, converter rerun, FOD/tensor recomputation, tractography, matrices, statistics, dashboard publication and full-cohort work. It awaits exact gate `Approve SL-H04A-CAL-R3`.
- [Thesis-grade 530 release contract](research_audit/outputs/thesis_grade_530_release_contract_v1/contract.md) and [current readiness V4](research_audit/outputs/thesis_grade_530_current_readiness_v4/evaluation.md) — the strict executable end-state definition. It requires 530/530 case-level GREEN bundles, 4,770 valid matrices, diagnosis-blind tractography stability, blinded human-QC reliability, full SHA-256 replay, statistical reliability and a checksum/restore-verified independent copy. Contract validation passes 21/21; current readiness is correctly NOT_GREEN at 0/530 corrected phase-B bundles. Dedicated validators quantify 3M/5M/10M plus independent-seed stability and blinded double-rating reliability; the final evaluator rejects unbound or status-only PASS JSON. A versioned Phase-B extension now parses and generates the required four-run evidence, but exact dry-run packaging correctly waits for Recovery2 and H04B.
- [Closed-world analysis design v1.1](research_audit/closed_world_analysis_design_v1.md) — SL-H03-approved fixed-data estimand, measured common support, timing/protocol/T1-source/QC sensitivities, validity floor, missingness, exchangeability, and claim limits; not yet the frozen SAP.
- [Immutable local-input lock](research_audit/outputs/available_data_content_lock_v2/available_data_content_lock_validation_v2.json) — PASS content lock for all selected DTI/T1 source files.
- [Fixed-data feasibility memo](research_audit/outputs/fixed_data_feasibility_power_memo_v1.md) — outcome-blind common-support, estimability, design-rank, attrition-stress, power, and workflow-contract audit.
- [SL-H03-C1 approval](research_audit/decisions/sl_h03_c1_approval_20260718.md) — approved DESIGN-ONLY cohort/estimand decision; it authorizes SL-P0-11–14 implementation but no imaging execution.
- [Protocol-composition figure](research_audit/figures/protocol_composition_by_group.png) and [sensitivity heatmap](research_audit/figures/key_metric_sensitivity_heatmap.png).

Important project separation: `/home/ec2-user/sabeesh/context.docx` is the unrelated pulmonary-embolism project context. It was inspected only to prevent project confusion and **was not overwritten**. Structural-connectome context belongs under `/home/ec2-user/exp`.

## Objective

Build reproducible, anatomically valid subject-level diffusion-MRI structural connectomes and determine which effects robustly distinguish CN, MCI and Alzheimer disease after controlling acquisition, registration, processing-recipe and cohort-selection effects. A publication claim is allowed only after the corrected pipeline, locked cohort, prespecified statistics and systematic novelty audit agree.

The supported current scientific chain is:

`DTI/T1 acquisition -> conventional tissue context -> jointly controlled AxD/RD stage axes -> distributed 17-system participation -> paired site robustness -> held-out late-RD coherence -> corrected confirmation`

Graph topology, edge architecture, LR/SR, EDR, brain age, mediation and dashboard story cards are alternative or secondary families unless independently validated. Their exact outcomes remain in the supplement and do not drive the main narrative.

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
- The historical candidate `scforge/workflow/Snakefile` resolves a 54-job one-subject PASS-only DAG and remains frozen as `connectome-v2.0.11-canary-candidate`. It is **not executable against the approved SL-D01 cohort contract** and is preserved as historical implementation evidence. Real execution now uses the later attempt-all recovery workflow described below.
- The supported real-run entry point is `scforge/workflow/run_connectome_v2.py`. Direct ad hoc execution is not part of the corrected contract.

### Workflow implementation state and reopened contract

- Historical v2.0.11 scientific contract validation: 47/47 PASS, zero warnings.
- Historical v2.0.11 locked host/environment/workflow-source validation: 173/173 PASS, zero warnings.
- Historical v2.0.11 SCForge test discovery: 44/44 PASS, including six provenance/ledger tests.
- Historical schema-only one-subject workflow dry run: PASS, 54 jobs; this proved candidate DAG resolution only, not compatibility with SL-D01, image validity, attrition behavior, or runtime success.
- The workflow explicitly locks CPU Eddy, MRtrix/FSL/ANTs/convert3d/SS3T executables, dependency artifacts, the 24-file workflow source bundle, MNI/AAL3 resources, lmax=6/direction eligibility, and all nine matrix definitions.
- The old Draft 2020-12 provenance/run-ledger schemas are PASS-only: they require every unit and all nine matrices to pass and constrain `fail=0`/`excluded=0`. That conflicts with approved attempt-all, mixed terminal-state, and outcome-specific failure/NA semantics. SL-P0-11–14 are therefore reopened; v2.0.11 must not be used for a canary.
- A provenance audit found and corrected the seed-identity mismatch: the executable RNG seed now derives from `dti_series_uid|recipe_id`, and the actual seed token, step size, and requested/actual streamline counts are recorded.
- No replacement T1 was downloaded and no S3 write, dashboard refresh, tractography, matrix generation, production overwrite, or 530-subject run occurred. The 15-unit response-calibration recovery is terminal PASS. Its original pre-tractography attempt, Recovery1 and Recovery2 are immutable terminal FAIL with 15/15 terminal records. Recovery2 completed every corrected converter and atlas-resample job, then exposed bounded atlas-validator, 5TT-resampling and MNI-to-T1 registration defects. Recovery3 diagnosis and packaging are PASS and await exact approval; no imaging process is running.
- The named AWS `aml` principal previously passed verification. At 01:40 UTC its cached SSO token had expired; no AWS mutation was attempted. This is not an imaging blocker and re-login is required only at the explicit safe-resize/availability gate.
- Exact-date candidate discovery freezes 4,015 locally known `Type=Original` T1-like IDs and the contingency intake/pairing gate passes 39 tests. That work is preserved as source-audit evidence; external replacement acquisition is not pursued unless SL-D01 is explicitly reopened.
- SL-D01 retains all 530 current local pairs. Their timing, cross-phase, acquisition, and QC limitations become explicit covariates/sensitivity strata; they are not relabeled as visit-matched.
- Diagnosis/DTI identity reconciliation is complete: 530/530 diagnoses resolve at the exact selected Original DTI ID/date; all 32 processed-dashboard versus Original-pipeline ID mismatches are concordant and resolved; harmonized groups are CN 251, MCI 186, AD 78, and SMC 15. The primary CN/MCI/AD ceiling is 515, with SMC retained separately. No local longitudinal/biomarker adjudication table was found, so the source label remains the supplied ADNI image-catalog `Research Group`.
- SL-H01 is approved. The SL-D01 pair manifest is PASS: 530/530 local DTI/T1 series resolve uniquely; intervals are 194 <=90 days, 3 at 91–180 days, and 333 >180 days; 334 are same-phase and 196 cross-phase. Corrected-rerun QC is deliberately `not_run` for 530/530. Source-side T1 QC is unavailable and 225 current T1s are catalogued as processed rather than Original.
- SL-P0-09 is DONE and independently verified. The immutable lock covers 530 pairs, 1,060 series, 451,115 unique files, and 99,066,365,832 bytes (92.262743 GiB), with zero duplicate-content groups, missing/extra files, or source mutations. Locked-manifest SHA-256 is `1e47d263c70a8230ab0b39651b652333d03e31c63308c9f18dfa008f804689f5`; validation SHA-256 is `70619a8c951c4c93416c9280de25b84ab62383119c26ddd39eaff201b89688d2`.
- SL-P0-10 is DONE and independently verified without opening biological outcomes. Aggregate exploratory CN/MCI/AD modeling is feasible only with a parsimonious acquisition representation. CN–MCI has measured common support; AD contrasts have limited cell depth. ADNI 2 is exactly aliased with GE 3 T/41-direction protocol in the observed design. The feasibility validation SHA-256 is `4fd319a3edc2897552a4f9e1415bbf95b06dc42bbfa9b1816850311940e6cdde`.
- SL-H03-C1 is approved, DESIGN ONLY. It fixes all 530 attempts/attrition rows, a 515 CN/MCI/AD outcome-specific contrast ceiling, 15 separate SMC cases, and primary participant-level upper-triangle unweighted `fa_mean` over `count>0` support. It authorizes the SL-P0-11–14 implementation/refreeze but not canary or full imaging execution.

### Terminal bounded recovery and GPU transition state

- Retry2 and retry3 failure evidence is preserved; retry4 at `/data/derivatives/scforge_v2/h04a_r1_recovery_20260719_retry4` is terminal PASS for SL-P0-16A-R1. Original SL-P0-16B attempt `20260719T121203.754603Z-pre-tractography-cb930ab7` is terminal FAIL: it closed at 12:54:28 UTC after 2,508.6 s with return code 1 and an immutable 15/15 FAIL ledger. All 15 tensor families and all 15 5TT segmentations completed. The terminal log contains 15 SS3T postcondition, 14 BBR image-declaration and three ANTs rule error records; one ANTs job completed to confirm the mismatch and two active duplicate registrations were stopped with the Snakemake child group. The parent launcher published all terminal records normally. No scientific or image-QC rejection appeared.
- Retry2 completed CPU Eddy, bias correction and brain masking for all 15 approved diagnosis-blind units. The final subject completed Eddy in about 10 h 27 min. Retry2 then closed as FAIL at the known pre-MRtrix one-element `Namedlist` shell-selection defect.
- Retry3 corrected that code path but failed closed before response estimation because its seed included `logs/05_select_fod_shells.log`, a continuation-owned file that changed during execution. It was terminally closed and retained; no evidence was deleted or promoted.
- Retry4 uses an upstream-only copy-on-write seed (`00_inputs` and `01_dwi`), full parent/launcher content rehashing, and an exact three-rule allowlist. Its dry run scheduled exactly 15 `select_fod_shells`, 15 `subject_response`, and one `response_calibration_phase_a` job. Live execution completed return code 0 with 15/15 terminal PASS outcomes.
- The response pool exceeds the prespecified minimum 12 and satisfies the exact diversity gate: GE 8 plus Siemens 7; DICOM-series T1 8 plus single-NIfTI T1 7. No diagnosis label was used for response estimation, QC, or pooling.
- The first freeze attempt exposed a schema adapter mismatch: the signed canonical acquisition subset has `subject_id` and `dti_image_id`, whereas the generic loader expected a pre-composed `unit`. The shared locked calibration module was restored byte-for-byte. A retry4-only adapter now derives `subject_id + '_I' + dti_image_id` in memory, rejects disagreement, leaves the signed CSV unchanged, and carries its own hash attestation. Adapter tests pass 3/3, the original calibration tests pass 6/6, and the retry3/retry4 environment/source audit passes 160/160.
- The authoritative frozen calibration is `frozen_calibration_retry4_v2/frozen_response_calibration_manifest.json`, SHA-256 `9348aed8f153f15709f372100d67b0a070fc337960ebe40c224eb3b714462053`; its adapter attestation SHA-256 is `a7dfdace7e8ef17f9dbecf671f3d3fc6d56ae4b820c1ff0b29fa0e4d55c41f9d`. Independent validation confirms expected shapes, finite coefficients, positive isotropic terms, exact input/output hashes and `release_ready_for_phase_b=true`. The earlier unversioned freeze candidate is preserved but explicitly not authorized for continuation; its numerical coefficients are exactly identical, with byte differences only in MRtrix destination-path comments.
- SL-P0-16B-PREP is complete: authoritative package V3 binds the unchanged Phase-A completion, V2 frozen response, exact 15-unit set, 32-core/24-hour/150-GB-total stops, and a pre-tractography-only target. Its dry run passes at exactly 394 jobs with no forbidden rule. SL-H04A-CAL is DONE: exact approval `Approve SL-H04A-CAL` created separate hash-bound response and extension authorities, and release validation passed. The live target may create FOD, tensor, anatomical, registration, atlas and QC/review artifacts only; tractography, SIFT2, connectome matrices, statistics, dashboard refresh and full-cohort processing remain unauthorized.
- The first terminal failure audit identified implementation plumbing, not biological failure. SS3T uses the frozen response and unchanged `lmax=6` command successfully, then failed at an unavailable postcondition `grep`. `epi_reg` natively writes `<prefix>.mat` and `<prefix>.nii.gz`, not the old rule's differently named image. Live and static evidence confirm `antsRegistrationSyN.sh` writes `<prefix>Warped.nii.gz`, not the old declaration. Recovery1 corrected only these declarations/checks and passed its launch-time rehash and exact 229-job dry run.
- Recovery1 then closed terminal FAIL at 14:23:00 UTC after 3,105.3 s and 63/229 completed steps. It produced all 15 WM FODs, all 15 normalized WM FODs, all 15 corrected BBR matrices/images and all 15 MNI-to-T1 affine/warp pairs. For every unit, FSL `convert_xfm` and MRtrix `transformconvert` succeeded, but the locked convert3d 1.4.2 `c3d_affine_tool` segfaulted while serializing the inverse affine to ITK. Consequently no atlas mapping, automated QC or blinded-review bundle completed; tractography and matrices never began.
- Recovery2 changed only that executable. The user supplied exact approval `Approve SL-H04A-CAL-R2`; decision SHA `335172d1d3be3c408060d7cc8dbdaac0bd56b87b096d251f2db8a9f4c76640f8` and execution binding SHA `34663be68437f23f42ac9a0e19064ed3ae10e71e9566f74808166c0125cd58a5` passed. Service `scforge-h04a-r1-pretract-recovery2.service` ended at 23:10:12 UTC with return code 1 after 71.526 s. It completed all 15 conversions and atlas resamples, then all 15 units failed before review publication. Logged errors reduce to two underlying rule families per unit: atlas-contract validation and DWI-space 5TT validation.
- Recovery3 fixes three evidenced implementation/spatial-contract problems without changing the atlas, response functions, FOD/tensor estimation or downstream connectome recipe. First, atlas QC receives a scalar path instead of a one-item Snakemake `Namedlist`. Second, 5TT is resampled linearly and clipped exactly to [0,1] before `5ttcheck`; this passes all 15 scratch units while retaining partial-volume information. Third, nonlinear MNI-to-native-T1 registration uses the MNI brain and a native-T1 brain mask derived from the already completed 5TT. In representative units 003, 009 and 014, label survival is 166/166; atlas/5TT Dice changes 0.778->0.810, 0.417->0.743 and 0.309->0.788, while the two severe centroid errors change 38.82->3.46 mm and 51.67->3.36 mm. Hard label-survival and spatial thresholds are not weakened.
- The separately versioned Recovery3 package passes validation with exactly 166 jobs: 15 each for fixed-image construction, MNI registration, atlas resampling/QC, 5TT/GMWMI resampling, spatial QC, review bundles and automated pre-tractography QC, plus the terminal target. Forbidden and unexpected rule sets are empty. It does not reschedule converter work, preprocessing/Eddy, FOD/tensor recomputation, tractography, SIFT2, matrices, statistics, dashboard publication or full-cohort processing. Its validation SHA-256 is `33da0330f80a99784fa1e8ff4f9b5814392a430df708b16d9a045991f4ca520c`. Exact gate `Approve SL-H04A-CAL-R3` is required before execution.
- The final 530-case objective is now quantified separately from canary success. Case-level GREEN requires locked source identity, complete preprocessing/Eddy QC, quantitative and blinded spatial QC, tissue/FOD/tensor validity, 10M ACT, SIFT2, endpoint-support thresholds, all nine matrix invariants and immutable provenance. The strict target permits zero PARTIAL/FAIL cases for goal completion. Diagnosis-blind phase-B stability now has an executable, hash-bound validator for 3M/5M/10M convergence and independent-seed replication; the human-QC validator requires all 530 primary reviews, every automated warning plus a deterministic 20% audit sample to be double-rated, Cohen kappa >=0.80, and independent adjudication with both original ratings retained. A valid thesis release also requires full SHA-256 evaluation, prespecified inferential reliability and a checksum/restore-verified copy at the documented `s3://sabeesh/exp/` release prefix; no upload has occurred.
- Live EC2 metadata at 12:42 UTC still reports `c6a.16xlarge` (64 vCPU, 123 GiB RAM); `nvidia-smi` cannot communicate with a GPU. A console selection of `g4dn.4xlarge` has therefore not been applied to the running instance. GPU-Eddy equivalence remains a separate scale-up gate and does not block pre-tractography consumption of the already CPU-preprocessed retry4 inputs.
- Execution stages remain distinct: completed `response-calibration-phase-a` produced individual and frozen pooled response functions; the original `pre-tractography-canary`, Recovery1 and Recovery2 are terminal FAIL after preserving valid upstream products; Recovery3 is a packaged but unapproved same-stage spatial-contract correction; launcher `phase-b` is the later tractography/matrix stage and remains behind H04B human review.

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

- Confirmed novelty is **not established**; the current result is a differentiated internal candidate in a bounded review.
- The July evidence base contains a 108-record broad map, a 23-study exact two-axis neighborhood, and a 15-study higher-order matrix. The 19 July update specifically covers longitudinal MCI-to-AD AxD/RD, 730-subject ADNI3 pathology tractometry, longitudinal whole-brain connectomics, advanced diffusion models and multimodal connectome-pathology organization.
- Generic CN/MCI/AD DTI differentiation, structural topology, limbic/default-mode effects, multiple diffusion weights, classification, longitudinal progression, pathology mapping, propagation, mediation, compensation, NBS, rich-club analysis and multimodal prediction are occupied.
- The surviving candidate is the exact jointly corrected, nonredundant system-level AxD MCI-CN plus RD AD-MCI estimand with paired availability, explicit multisite common support, mapping sensitivity and paired site deletion. “Differentiated in this bounded review” is allowed; “first ever” or proof of priority is not.
- Corrected same-cohort processing can test robustness but cannot supply independent replication. A second bibliographic database, dual screening and an untouched compatible cohort remain required for stronger claims.

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

Completed prerequisites are tracked in the [superlist](research_audit/SUPERLIST.md): retry2/retry3 terminal evidence, retry4 response completion, the authoritative V2 calibration, all three pre-tractography terminal publications, the passing Recovery3 diagnosis/package, and the 21/21 thesis-grade 530 release/copy contract are complete. No imaging process is active. Manuscript drafting remains outside the current user-directed scope.

1. Obtain or reject the exact Recovery3 gate `Approve SL-H04A-CAL-R3`. This authorizes only the three evidenced spatial-contract corrections and signed 166-job residual scope for the same 15 diagnosis-blind units.
2. After approval, launch the separate persistent Recovery3 service at no more than 32 CPU cores. It must re-hash all bound inputs, repeat the exact dry run and stop at the 24-hour/150-GB envelope or terminal pre-tractography publication.
3. Validate the completion manifest, terminal ledger, automated-QC records, visual-review indexes and overlays, and verify that no tractography/SIFT2/matrix artifact was created. Only actually passing units proceed to human review.
4. Review automated QC and visual overlays at SL-H04B. Do not start tractography or matrices for any failed/unreviewed unit.
5. In the parallel scale-up stream only, perform the controlled GPU transition and frozen one-unit CPU/GPU Eddy equivalence benchmark before choosing full-cohort capacity. It is not a dependency for this already CPU-preprocessed canary.
6. Authorize tractography/matrices only after H04B; decide whether targeted or larger corrected processing is justified from canary evidence rather than automatically rerunning 530 subjects.
7. Refine and validate scientific figures/tables from corrected outputs; retain, qualify, or reject the current two-axis candidate under the frozen confirmation contract. Manuscript writing remains with the user unless explicitly reassigned.

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

The evidence-complete V2 manuscript now organizes every tested family around one statistically bounded candidate: a distributed, jointly corrected AxD MCI-CN plus RD AD-MCI adjacent-stage phenotype. Full and pair-common-site families survive restricted wild-cluster/Holm inference; the all-three-site family is borderline. Conventional MD/AxD/RD findings supply tissue context, while network selectivity, graph topology, edge architecture, topology–microstructure coupling, brain age, delay, resilience/propagation constructs, and most ML increments do not supply an independently corrected third axis.

This is a differentiated internal result, not a validated biomarker or guaranteed novelty claim. The exact literature search did not locate the same combined estimand and validation bundle, but absence from a bounded search is not proof of priority. The historical matrices remain vulnerable to multimodal visit mismatch, mislabeled matrix semantics, mixed processing recipes, stale QC, and acquisition confounding. Corrected canary evidence must therefore determine whether the two-axis result is retained, qualified, or rejected in V-final.
