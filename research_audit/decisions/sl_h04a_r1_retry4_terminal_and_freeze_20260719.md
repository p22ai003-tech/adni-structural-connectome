# SL-H04A-R1 retry4 terminal and frozen-calibration decision

**Decision state:** COMPLETE  
**Verified:** 2026-07-19 11:40 UTC  
**Scope:** exact 15-unit diagnosis-blind response-calibration recovery only

## Outcome

Retry4 completed the exact bounded 31-job DAG: 15 shell selections, 15
subject-level Dhollander WM/GM/CSF response estimates, and one terminal Phase-A
aggregate. The launcher returned zero, the Phase-A completion status is
`PASS`, and all 15 outcomes pass their declared coefficient-shape, finiteness,
positive-isotropic-term and nonzero tissue-voxel checks.

The prespecified freeze gate passes with margin:

- valid response units: 15; required minimum: 12;
- manufacturer families: GE 8 and Siemens 7; required minimum: two;
- T1 source classes: DICOM series 8 and single NIfTI 7; both required classes
  are present;
- diagnosis labels used in response estimation, QC or pooling: no.

Fifteen is therefore sufficient for this technical calibration gate. It is not
sufficient by itself to estimate CN/MCI/AD effects, establish biological
novelty, validate a biomarker, or replace corrected-cohort statistical
analysis.

## Authoritative evidence

- Phase-A completion:
  `/data/derivatives/scforge_v2/h04a_r1_recovery_20260719_retry4/publication/response_calibration_phase_a_completion.json`
- Authoritative frozen manifest:
  `/data/derivatives/scforge_v2/h04a_r1_recovery_20260719_retry4/frozen_calibration_retry4_v2/frozen_response_calibration_manifest.json`
- Frozen-manifest SHA-256:
  `9348aed8f153f15709f372100d67b0a070fc337960ebe40c224eb3b714462053`
- Retry4 freeze attestation:
  `/data/derivatives/scforge_v2/h04a_r1_recovery_20260719_retry4/frozen_calibration_retry4_v2/retry4_freeze_attestation.json`
- Attestation SHA-256:
  `a7dfdace7e8ef17f9dbecf671f3d3fc6d56ae4b820c1ff0b29fa0e4d55c41f9d`

Independent replay verifies the frozen manifest, every implementation/input
file record, all pooled-response hashes and numerical shapes. The WM, GM and
CSF arrays are finite and their isotropic coefficients are positive. The
versioned and earlier diagnostic freeze candidates have exactly identical
numeric arrays; their file hashes differ only because MRtrix writes the output
destination into a comment header.

## Schema-adapter correction

The first freeze attempt failed closed because the signed canonical subset CSV
contains `subject_id` and `dti_image_id`, while the generic loader expected a
pre-composed `unit`. No pooled artifact was written by that failed attempt. A
temporary diagnosis confirmed the deterministic mapping, after which the
shared calibration module was restored to its exact locked SHA-256
`36c3e1c8...90c99`.

The authoritative freeze uses the isolated versioned adapter
`scforge/workflow/freeze_response_calibration_retry4.py`. It derives
`unit = subject_id + "_I" + dti_image_id` in memory, rejects any explicit versus
derived identity disagreement, exposes only manufacturer/T1-source metadata,
does not modify the signed CSV, and writes an implementation attestation.
Validation passes 3/3 adapter tests, 6/6 original response-calibration tests,
and 160/160 retry3/retry4 environment/source checks.

The earlier unversioned manifest under `frozen_calibration/` is preserved for
forensic continuity and marked in the V2 attestation as not authorized for
continuation.

## Boundary and next gate

No FOD reconstruction, tensor map, T1/atlas continuation, tractography, SIFT2,
connectome matrix, statistics, dashboard refresh or full-cohort work occurred.

The next permitted work is preparation and dry-run of a 15-unit
pre-tractography-only package. Actual execution requires a new exact
SL-H04A-CAL approval bound to the V2 manifest/attestation, rule allowlist,
resource limits and stop-at-human-QC boundary. Tractography and matrices remain
outside that gate.
