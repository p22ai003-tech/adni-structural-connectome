# SL-H04A bounded canary approval

**Decision:** `APPROVED`  
**Approved UTC:** 2026-07-18T13:19:25Z  
**Approver record:** project user via Codex chat  
**Exact user response:** `Approve`

This approval applies only to the hash-bound 24-unit H04A canary package and authorizes response-calibration Phase A followed, if its frozen calibration is valid, by pre-tractography processing and blinded review-bundle generation.

Execution limits are four cores, 72 cumulative hours, 150 GB, and the isolated run root `/data/derivatives/scforge_v2/h04a_canary_20260718_v1`. A valid pooled response requires at least 12 shell-compatible units spanning at least two manufacturer families and both signed T1-source classes.

This approval does not authorize tractography, SIFT2, connectome matrices, biological unblinding, threshold tuning from group separation, dashboard inference release, production overwrite, or full-cohort processing. Those actions remain behind later human gates.

The launcher-compatible signed record is `sl_h04a_canary_start_approval_20260718.json`.
