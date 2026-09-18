# SL-H04A-R1 retry4 execution allowlist

Retry4 retains the exact bounded continuation contract proven by the retry3
V5 dry run. The only executable rules are `select_fod_shells`,
`subject_response`, and `response_calibration_phase_a`; rerun triggers are
`input` and `params`.

The retry4-specific correction is limited to the seed boundary. Only immutable
`00_inputs` and `01_dwi` artifacts are copied and hashed. No continuation-owned
log or output can therefore mutate a declared seed file.

Upstream normalization, denoising, Gibbs removal, Eddy, bias correction, and
mask generation remain prohibited. FOD reconstruction, tensor maps,
tractography, SIFT2, connectome matrices, statistics, and dashboard publication
remain prohibited.

A retry4 dry run must schedule exactly 31 jobs: 15 shell selections, 15 subject
responses, and one Phase-A pooled calibration. Any other rule is a stop.
