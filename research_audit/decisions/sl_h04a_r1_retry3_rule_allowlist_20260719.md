# SL-H04A-R1 retry3 continuation-rule allowlist

**Decision date:** 2026-07-19  
**Scope:** operational safety correction only; no scientific parameter, cohort, resource-cap, or downstream-authorization change

The immutable retry3 V4 dry run completed Snakemake DAG construction and showed
that the full producer chain remained visible. Because run-level manifest and
execution-preflight products were intentionally not copied from retry2, Snakemake
would have scheduled 168 jobs, including 15 `dwi_motion_eddy` jobs, despite the
content-addressed subject seed being complete through brain masking.

No imaging job was executed: V4 used `--dry-run`. The finding is therefore a
prevented duplicate-computation hazard, not a processing failure or loss of
subject outputs.

Retry3 may treat the fully rehashed copy-on-write seed as external input only
when the launcher enforces this exact rule allowlist:

1. `select_fod_shells`
2. `subject_response`
3. `response_calibration_phase_a`

`dwi_motion_eddy`, normalization, denoising, Gibbs correction, bias correction,
masking, FOD reconstruction, tensor fitting, tractography, matrix generation,
statistics, and dashboard publication remain unauthorized. The launcher must
retain content-based `input params` rerun triggers. A new immutable binding and
a passing dry run showing exactly the three rules above are required before any
resize or retry3 execution recommendation.

Dry-run diagnostics are retained rather than overwritten:

- V1: unsupported legacy `--reason` option.
- V2: temporary resolved config outside the contract-bound run root.
- V3: launcher context inputs were not materialized for DAG parsing.
- V4: DAG parsed successfully and detected the genuine upstream-recomputation hazard.
