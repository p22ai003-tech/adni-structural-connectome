# SL-H03-C1 recommendation — fixed-data cohort and estimand design

**Gate:** SL-H03  
**Option:** SL-H03-C1  
**Recorded:** 2026-07-18 10:10 UTC  
**Content-lock evidence bound:** 2026-07-18 10:18 UTC  
**Decision state:** `APPROVED`  
**Approval scope:** **DESIGN ONLY**
**Approved:** 2026-07-18 10:28 UTC  
**User response received:** `Approve SL-H03-C1`

## Recommendation

Approve SL-H03-C1 as the cohort and estimand design for the next implementation phase:

- attempt all 530 manifest-selected local DTI–T1 pairs and retain every terminal state in attrition accounting;
- use CN 251, MCI 186, and AD 78 (ceiling n=515 before corrected outcome-specific validity) for the three diagnosis contrasts; retain the 15 SMC separately for processing/QC/attrition and descriptive reporting;
- define the primary participant outcome as the unweighted arithmetic mean of corrected `fa_mean[i,j]` over unique undirected upper-triangle edges with `i < j` and `count[i,j] > 0`; unsupported zero-coded edges and the diagonal do not enter the mean;
- fail the participant-level FA outcome when a supported edge is missing, non-finite, inconsistent with `count`, or outside `[0,1]`, rather than silently dropping or replacing that edge;
- use protocol key plus T1 source class (`Original` versus `Processed`) as the parsimonious fixed acquisition representation, with site/scanner partial pooling only when the outcome-specific model converges and rank/condition diagnostics pass;
- do not enter ADNI phase simultaneously with protocol in the primary fixed design: the ADNI 2 indicator exactly duplicates `GE MEDICAL SYSTEMS|3T|41dir` in these data (36 subjects: CN 3, MCI 0, AD 33);
- reserve nonlinear gap and diagnosis-by-gap terms for the full 515-person analysis. Omit them from <=90-day and <=180-day models because those sets have only 2 and 5 distinct gaps, respectively (193 zero-day pairs in each); all 94 <=90-day Siemens-54 subjects have zero gap;
- control the primary family across CN–MCI, MCI–AD, and CN–AD, reporting effects, confidence intervals, multiplicity-adjusted p-values, exact outcome-valid n, and common-support diagnostics;
- treat CN–MCI as the best-supported aggregate contrast. Label MCI–AD and CN–AD as limited-cell-depth aggregate contrasts and prohibit unsupported cell-specific claims;
- require the declared timing, continuous-gap, Siemens-54, multi-group-site, same-phase, Original-versus-processed T1, QC, attrition, and aggregation sensitivities without selecting the most favorable result.

The primary aggregation sensitivities are count-weighted mean FA, valid SIFT2-`fd_sum`-weighted mean FA, and the unweighted median over the same supported upper-triangle edges. They are sensitivities; they cannot replace the primary unweighted supported-edge mean based on significance.

## Evidence supporting the recommendation

- Processing denominator: 530; diagnostic ceiling: 515; SMC descriptive: 15.
- Full-cohort protocol-plus-T1 linear feasibility proxy: 16 columns, rank 16, residual df 499, column-normalized condition number 5.58.
- Draft phase-plus-protocol cubic proxy: 27 columns, rank 26; site-fixed stress proxy: 78 columns, rank 74.
- CN–MCI: 41 shared site-by-protocol cells, supported n=245/184, eight cells with at least five per group.
- MCI–AD: 18 shared cells, supported n=82/34, no cell with at least five per group.
- CN–AD: 21 shared cells, supported n=129/42, no cell with at least five per group.
- Corrected QC outcomes remain unrun for 530/530; the power/attrition scenarios are planning stress tests, not measured failure rates or biological results.

This recommendation is bound to:

- source pair manifest SHA-256 `a51503e30f0b8e60a2216dc4b70cb69546f302eda56bf5e75469c7bafca8fdd1`;
- SL-P0-09 content-lock validation SHA-256 `70619a8c951c4c93416c9280de25b84ab62383119c26ddd39eaff201b89688d2`, status `PASS`, release `CONTENT_HASH_LOCKED`;
- content-locked pair manifest SHA-256 `1e47d263c70a8230ab0b39651b652333d03e31c63308c9f18dfa008f804689f5`;
- deterministic file-universe SHA-256 `30b5201a9f175877fdb5f93b8c163ae94f668fe03c56dcca013f01ab503a735a`, binding 530 pairs, 1,060 selected series, 451,115 regular unique files, and 99,066,365,832 exact bytes;
- diagnosis reconciliation SHA-256 `0323dc47d905269869074f025f872d8c799a437e27af8daa6acac9d2e54c299a`;
- SL-P0-10 validation SHA-256 `4fd319a3edc2897552a4f9e1415bbf95b06dc42bbfa9b1816850311940e6cdde`, status `PASS`, release `FEASIBILITY_COMPLETE_NO_OUTCOMES_OPENED`;
- SL-P0-10 memo SHA-256 `52a1a695235a360a87fed4682a713b4b70bc0a81f0c4010dd854a96ea7e7b442`;
- design memo v1.1 SHA-256 `03efdd6279edfef759444ef73c21f3c77165a634576f58ffd7d184bcbb584075`.

SL-P0-09 performed byte hashing only: no image processing, transfer, production-path modification, or biological-outcome access occurred. Its PASS release binds the local source universe; it does not authorize source staging, canary execution, or inference.

## Required pre-canary work

Approval does not waive SL-P0-11–14:

- SL-P0-11 must refreeze the input/processing specification against all 530 approved source rows, including long-gap, processed-T1, and SMC attempts, and construct the authoritative metadata/staging projection.
- SL-P0-12 must refactor/refreeze the DAG for the approved source classes, independent per-subject failure, and complete terminal-state capture.
- SL-P0-13 must prove source adaptation, validity-to-NA propagation, mixed PASS/FAIL attrition, and tamper behavior while preserving matrix invariants.
- SL-P0-14 must refreeze mixed-state provenance and outcome-specific attrition ledgers that publish even when failures occur.

## Approval boundary

This approval is **DESIGN ONLY**. It approves the cohort, estimand, primary participant FA definition, support-calibrated contrast roles, and required sensitivity structure as the basis for SL-P0-11–14 and later SAP work.

It explicitly does **not** authorize:

- canary imaging execution;
- full-cohort imaging execution;
- corrected biological group-result unblinding or release;
- a final SAP freeze;
- a novelty or manuscript finding;
- dashboard inference release.

## Approval record

The exact response token `Approve SL-H03-C1` was received from the user on 2026-07-18 at 10:28 UTC. SL-H03-C1 is therefore approved with the DESIGN-ONLY boundary above. SL-P0-11–14 may proceed; canary/full imaging execution and result/dashboard release remain unauthorized.
