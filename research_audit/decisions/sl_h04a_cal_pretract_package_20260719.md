# SL-H04A-CAL pre-tractography package decision record

**Prepared:** 2026-07-19 12:01 UTC  
**Current state:** AWAITING USER DECISION  
**Exact approval text:** `Approve SL-H04A-CAL`

## Recommendation

Approve the exact retry4 V2 frozen response for a bounded 15-unit pre-tractography canary. The technical calibration pool is sufficient for this gate: 15/15 phase-A response outcomes passed, exceeding the prespecified minimum of 12, with GE 8/Siemens 7 and DICOM-series T1 8/single-NIfTI T1 7. This does not establish a CN/MCI/AD effect, biomarker, novelty claim, or final inference.

## Package audit

- Authoritative package: `research_audit/outputs/h04a_r1_retry4_pretract_package_v3/validation.json`
- Validation SHA-256: `95e4f191f54bf9e1238b62b2bcc297d3dfaf58996c356f649198d88c073df6a1`
- Status: PASS; no imaging executed.
- Exact dry-run total: 394 jobs.
- Job composition: 392 anatomical/registration/atlas/FOD/tensor/automated-QC/blinded-review jobs, one manifest-freeze job, and one execution-preflight job.
- Forbidden job count: zero for preprocessing/Eddy, response re-estimation, tractography, SIFT2, connectome matrices, statistics, dashboard publication, and full-cohort processing.
- Resource stop: maximum 32 CPU cores, 24 hours, and 150,000,000,000 total run-root bytes.
- Packaging measurement: 62,792,372,224 effective bytes used, 87,207,627,776 bytes remaining to the stop, and 3,304,015,069,184 filesystem bytes free.
- Terminal boundary: automated QC plus blinded visual-review bundles; next gate is SL-H04B.

## Fail-closed correction

The first package dry run was retained as FAIL because it revealed two necessary contract dependencies—`freeze_manifest` and `execution_preflight`—that were not in the initial proposed rule set. The legacy execution preflight still encoded the completed response-only authorization and could not safely authorize FOD/tensor continuation. A versioned compatibility/preflight layer now preserves the completed response binding and consumes a separate SL-H04A-CAL extension. Package V3 binds all six implementation sources by SHA-256.

Negative tests confirm that wrong approval text is rejected, a dry-run-only binding cannot be elevated to live authority, no approved/live binding is present, no downstream artifact was created, and no imaging process is active.

## What approval would and would not do

Approval would create immutable response-calibration and pre-tractography extension bindings, then permit only the exact packaged 15-unit target. It would not authorize preprocessing/Eddy reruns, response re-estimation, tractography, SIFT2, connectome matrices, statistics, dashboard refresh, biological inference, or any 530-unit run.
