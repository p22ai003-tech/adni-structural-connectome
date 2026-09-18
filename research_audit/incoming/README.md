# Incoming exact-date metadata handoff

**Current state:** contingency only. SL-D01 fixes the project to locally available images, so no external export is currently requested. Use this directory only if that decision is explicitly reopened.

Place the fresh ADNI/LONI MRI metadata export in this directory. Do not place passwords, session cookies, access tokens, browser device codes, or other credentials here.

Current source instructions are in `../adni_exact_date_export_handoff.md`. The preferred handoff is the untouched IDA Advanced Image Search CSV plus the current `MRIQC` Study File; `MRIMETA`/`MRI3META` may be supplied as a date cross-check. AWS `aml` login does not authenticate the separate IDA/LONI service.

## Requested population and search scope

The frozen locally known request universe is listed in:

- `../outputs/exact_date_t1_candidate_request_v2.csv`
- 4,015 unique Original T1-like Image IDs across all 530 selected subjects
- includes the 330 provisional replacement IDs as a tagged subset, not as the full request
- SHA-256 `4a935b7bbd7d6d5ea5c1c8d80f4cc8fde840abaa88c0d4d2419878591bcc2406`

Run the IDA image search by the 530 subjects and retain every in-scope Original T1-like row. This both resolves the frozen 4,015 IDs and tests whether the older local `all_mri.csv` omitted newer candidates. Join and validate by exact Image ID, not rounded age or filename similarity. Do not discard additional in-scope IDs; flag them as candidate-universe drift for review.

## Required fields

- Subject ID
- Image ID
- Study Date
- Visit
- Phase/project
- Description
- Type, including whether the image is Original
- Series UID if available
- Image-QC status and QC source
- Manufacturer, model, field strength, protocol/acquisition fields if available

CSV is preferred. Additional columns should be retained; they may be needed for acquisition-overlap and tie-break audits.

For the current ADNI source, preserve raw `MRIQC` fields and codes. `StudyDate` is accepted as DICOM `YYYYMMDD`; `LONIImage` is accepted as Image ID; `ParticipantID` is accepted as Subject ID. Do not manually translate series QC or selection fields.

## Acceptance checks

1. Every requested Image ID is present with an exact date and recognized QC disposition; missing, unknown-QC, or unexpected in-scope IDs block pairing until explicitly resolved under the frozen policy.
2. Study Date is an exact calendar date, not age, year, or visit label.
3. Duplicate Image IDs have no conflicting date, subject, series, or QC values.
4. The export has source name, generation date, and an immutable local checksum.
5. Explicit QC failures are never promoted by the deterministic pair builder.

The export will be validated locally before any download manifest is generated. No imaging transfer is authorized by placing metadata here.

The fail-closed validator is `../validate_exact_date_catalog.py`. It refuses a non-empty output directory, pins the 4,015-row roster hash, records source checksums, emits diagnostics on failure, and emits the normalized `exact_date_catalog_v2.csv` only on PASS. Every output row forces `approved_for_download=false`. It consumes one source-preserving joined catalog; the raw image-search plus `MRIQC` join will be finalized and tested against the actual handed-off headers rather than guessed in advance.
