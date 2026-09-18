# SL-H03-C1 approval — fixed-data cohort and estimand design

**Gate:** SL-H03  
**Decision:** `APPROVED`  
**Approved:** 2026-07-18 10:28 UTC  
**User response:** `Approve SL-H03-C1`  
**Scope:** **DESIGN ONLY**

## Approved design

- Attempt all 530 checksum-locked local DTI–T1 pairs under one diagnosis-blind corrected recipe and retain every terminal state in technical attrition.
- Use CN 251, MCI 186, and AD 78 as the 515-person pre-QC ceiling for outcome-specific diagnostic contrasts; keep 15 SMC cases separate and descriptive.
- Use the unweighted upper-triangle mean of corrected `fa_mean[i,j]` where `count[i,j] > 0` as the primary participant summary, with three multiplicity-controlled contrasts: CN–MCI, MCI–AD, and CN–AD.
- Treat the full result as an exploratory local available-data association, not a causal effect, progression estimate, same-visit biomarker, clinical classifier, replication, or population-wide ADNI estimate.
- Retain low-quality but technically valid outputs with prespecified quantitative-QC and sensitivity analyses; represent hard-invalid outputs as failure/NA for the affected outcome.
- Use a parsimonious acquisition representation and report non-estimable contrasts as such. Timing, protocol/site/scanner, T1 source, QC, attrition/missingness, influence, and aggregation sensitivities are mandatory.

## Authorized next work

This approval authorizes the design basis and implementation work for reopened SL-P0-11–14: the all-530 source/input contract, independent terminal-state DAG behavior, invariant/integration tests, and mixed PASS/PARTIAL/FAIL plus outcome-specific provenance/attrition ledgers.

## Not authorized

This approval does not authorize canary imaging execution, full-cohort imaging execution, corrected biological result unblinding/release, final SAP freeze, novelty/manuscript findings, or dashboard inference release.

## Bound evidence

- Content-lock validation SHA-256: `70619a8c951c4c93416c9280de25b84ab62383119c26ddd39eaff201b89688d2`.
- Locked 530-row manifest SHA-256: `1e47d263c70a8230ab0b39651b652333d03e31c63308c9f18dfa008f804689f5`.
- Fixed-data feasibility validation SHA-256: `4fd319a3edc2897552a4f9e1415bbf95b06dc42bbfa9b1816850311940e6cdde`.
- Design memo v1.1 SHA-256 at approval: `03efdd6279edfef759444ef73c21f3c77165a634576f58ffd7d184bcbb584075`.

The detailed approved terms remain in `sl_h03_c1_recommendation_20260718.md/.json`.
