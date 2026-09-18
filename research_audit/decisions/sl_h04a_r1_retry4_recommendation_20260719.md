# SL-H04A-R1 retry4 recommendation

Retry3 is terminal and preserved. It cleared the original single-output
Namedlist error and produced eight valid shell-extraction products before a
second fail-closed orchestration check stopped the run. The stop was caused by
including continuation-owned log files in the immutable retry seed.

Retry4 is recommended under the existing bounded H04A-R1 authorization. It
uses the same 15 diagnosis-blind units, the same corrected DWI inputs, the same
scientific configuration, and the same 32-core, 24-hour, 100-GB envelope. Its
seed is restricted to immutable upstream areas (`00_inputs` and `01_dwi`).
Logs, response products, attempt evidence, publication records, FODs, tensor
maps, tractography, and matrices are not seeded.

The execution allowlist remains exactly:

- `select_fod_shells`
- `subject_response`
- `response_calibration_phase_a`

No upstream recomputation or downstream continuation is authorized. A fresh
dry run must show exactly 31 jobs before retry4 may start.
