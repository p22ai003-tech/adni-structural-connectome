# SL-H04A-I1 — original canary interruption-resolution recommendation

**Status:** PENDING USER DECISION  
**Prepared:** 2026-07-18 15:40 UTC  
**Recommendation:** close the exact unended launcher attempt as an interrupted FAIL, preserve every artifact, do not resume in the old run root, and use the separately gated isolated SL-H04A-R1 recovery if the user wants Phase-A evidence to continue.

## What happened

The approved original Phase-A launcher began at `2026-07-18T13:20:19.586744+00:00`. Its Snakemake controller and supported launcher are no longer running, and the attempt has no immutable `.end.json`. The last observed computation completed one CPU-Eddy output at `2026-07-18T15:29:56.101816+00:00`; two process checks at 15:31 and 15:40 UTC found no launcher, Snakemake, or Eddy process.

The aggregate state is:

- T1 normalized: 24/24;
- DWI normalized: 8/24, with 16 explicit normalization failures;
- gradient-contract PASS: 7/24;
- denoised: 4/24;
- Gibbs-ringing correction complete: 3/24;
- CPU Eddy complete: 1/24;
- response outcomes: 0/24;
- Phase-A completion/calibration freeze: 0/24;
- run-root footprint: 8.123 GiB.

The run cannot satisfy its signed minimum of 12 response-compatible units because only seven passed the gradient contract. Restarting the supported launcher automatically is correctly prohibited: the provenance contract requires human resolution of an unclosed matching H04A attempt.

## Exact evidence binding

- Attempt ID: `20260718T132019.562821Z-response-calibration-phase-a-ba0c318b`
- Attempt start: `/data/derivatives/scforge_v2/h04a_canary_20260718_v1/attempts/20260718T132019.562821Z-response-calibration-phase-a-ba0c318b.start.json`
- Attempt-start SHA-256: `1d42f1420b7c9d80a807f126d64a60ccfd80add77d693548cc1b4b0dadb6d706`
- Combined log SHA-256: `1388e9309a6fcf512e7a02d996f0f151ce1b48acae211c59ab508bf48b758277`
- First completed Eddy output SHA-256: `dc7a3ceb92d0f245fd8db1f6c237a6772ea20b8c0c612a75fcdca38c4eed1ade`
- Signed H04A decision SHA-256: `1be4529aa482c73a79f105e6f37ec4fdc9032f0a8894d525f8a53c79dc52ccb4`
- Lower-bound computation duration through the last artifact: 7,776.515 seconds.
- Conservative H04A accounting duration through the no-process observation: 8,398.413 seconds.

## Proposed resolution if approved

1. Write one immutable attempt-end record with status `FAIL`, no fabricated process return code, the conservative 8,398.413-second resource charge, exact start/log/output hashes, and interruption reason `launcher_control_process_absent_before_attempt_end`.
2. Preserve the old run root and all partial outputs read-only for provenance; delete, modify, or overwrite nothing.
3. Mark the original SL-P0-16A attempt interrupted/failed and scientifically insufficient. Do not treat the single Eddy output as response-calibration evidence.
4. Do not restart Phase A in the old run root.
5. If SL-H04A-R1 is also approved, launch the separately isolated 15-unit Siemens/GE recovery with its already signed scope limits: Phase A only, maximum 32 CPU, 24 hours, and 100 GiB; no FOD, tractography, matrices, dashboard update, or full-cohort work.

## Approval boundary

Approval of SL-H04A-I1 authorizes only the immutable interruption closure described above. It does not authorize the recovery run. Approval of SL-H04A-R1 authorizes only the isolated recovery and does not authorize tractography, matrix generation, or full-cohort processing.

To approve both recommended actions in one response, use:

`Approve SL-H04A-I1 and SL-H04A-R1`

To close only the interrupted attempt without starting recovery, use:

`Approve SL-H04A-I1 only`

