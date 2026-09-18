# ADNI/LONI exact-date export handoff

**Status:** contingency only; SUPERSEDED by approved local-only decision SL-D01  
**Last verified:** 2026-07-18 09:16 UTC  
**Safety:** metadata only; do not add images to an IDA download collection yet.

The current plan does **not** request this export or any replacement images. This handoff is retained only so the source-audited route remains reproducible if the user explicitly reopens external acquisition.

AWS authentication and IDA authentication are separate. The project `aml` AWS principal is now verified, but the missing exact-date source is an ADNI/IDA Study File or image-search export and is not present in the project S3 locations.

## What to export

- [ ] In the IDA, open **Search and Download -> Study Files** and export the current **MAYO ADIR LAB MRI Quality** table (`MRIQC`) for all applicable ADNI phases.
- [ ] In **Advanced Image Search**, run a **subject-based search for all 530 requested subjects** and include every in-scope Original T1-like series, rather than searching only a preselected Image-ID list. Select the required metadata under **Display in result** before using **CSV Download**. The IDA manual states that CSV Download exports the metadata displayed in the results.
- [ ] If available, also export `MRIMETA` and `MRI3META` for an independent scan-date cross-check.
- [ ] Save the raw CSV files unchanged in `research_audit/incoming/`. Do not paste passwords, cookies, access tokens, device codes, or screenshots containing credentials.
- [ ] Record the exact table/export names and the UTC export time in a short text note in the same directory.

The current [IDA User Manual](https://ida.loni.usc.edu/explore/jsp/support/IDA_User_Manual.pdf) documents Advanced Image Search, selectable result fields, and CSV Download. ADNI describes Study Files as the route for tabular data and directs approved users to the IDA for data access in its [data access overview](https://adni.loni.usc.edu/data-samples/) and [FAQ](https://adni.loni.usc.edu/help-faqs/faqs/).

## Fields to retain

From the image-search CSV, retain at least:

- Subject ID / PTID
- Image ID
- Study Date if displayed
- Visit
- ADNI phase/project
- Modality
- Series description
- image `Type`, including whether it is `Original`
- Series UID and protocol/scanner fields if displayed

From `MRIQC`, retain the raw fields that identify the image/series, exact study date, visit, scanner/protocol, and QC. Current official dictionary names include `ParticipantID`, `LONIImage`, `StudyDate`, `VISCODE2`, `SeriesInstanceUID`, `SeriesDescription`, scanner/protocol fields, and series-level QC/selection fields. Preserve the raw QC values; do not recode them manually.

QC schema is source-bound:

- current `MRIQC.SeriesQC`: `1=Pass`, `4=Fail`, `-1=Not assessed`;
- legacy `SERIES_QUALITY`: `1=Excellent`, `2=Good`, `3=Fair`, `4=Unusable`, `-1=Not evaluated`;
- `-1` is unknown/not assessed and ineligible under strict passed-QC analysis, but it is not relabeled as a failure;
- do not apply the legacy grade profile to `SeriesQC`, or the current pass/fail profile to `SERIES_QUALITY`.

The official ADNI data dictionary documents the current [`SeriesQC`](https://adni.loni.usc.edu/data-samples/data-dictionary-search/?q=SeriesQC) and legacy [`SERIES_QUALITY`](https://adni.loni.usc.edu/data-samples/data-dictionary-search/?q=SERIES_QUALITY) fields. Preserve `MRIProtocolPhase` as the MRI acquisition-protocol family; do not substitute it for participant enrollment phase.

For date interpretation:

- use `MRIQC.StudyDate` as the exact study date;
- `MRIMETA`/`MRI3META.EXAMDATE` is the scan date and can be used as a cross-check;
- do **not** use `MMTRNDATE`, which is the IDA upload date;
- do **not** treat `MRI3META.HAS_QC_ERROR` as image QC because ADNI documents it as a case-report-form administrative flag.

These distinctions are documented in ADNI's current [tabular MR data guide](https://adni.loni.usc.edu/quick-start-guide-asset/MRI_tables.html). ADNI also states that scan-specific acquisition details are authoritative in the DICOM headers and describes series QC in its [MRI overview](https://adni.loni.usc.edu/data-samples/adni-data/neuroimaging/mri/) and [MRI quality-control page](https://adni.loni.usc.edu/data-samples/adni-data/neuroimaging/mri/mri-quality-control/).

## Exact request roster

The frozen locally known request roster is:

`research_audit/outputs/exact_date_t1_candidate_request_v2.csv`

- 4,015 rows
- 4,015 unique Original T1-like Image IDs
- 530 unique subjects
- 1–38 candidates per subject
- includes all 530 rounded-age screen IDs and all 330 provisional replacement IDs as tagged subsets
- SHA-256: `4a935b7bbd7d6d5ea5c1c8d80f4cc8fde840abaa88c0d4d2419878591bcc2406`

The source join is by exact Image ID (`LONIImage`), with subject and series identity used as consistency checks. Rounded age and filename similarity are forbidden join keys. The 330 provisional replacements are **not** the complete search universe: nearest-date selection is valid only after every locally known candidate is resolved. Because the local `all_mri.csv` may be stale, the subject-based IDA export must also be checked for additional in-scope Original T1 candidates. Such rows are candidate-universe drift and must be reviewed rather than silently ignored.

## What Codex will do after handoff

1. Hash the untouched source exports.
2. Inspect the real headers/data dictionary, then create and test a source-preserving join of the image-search rows and `MRIQC` by exact Image ID while retaining both raw sources. This multi-source join is intentionally not claimed complete before the actual exports arrive.
3. Compare the subject-based export with the frozen 4,015-ID roster and resolve any candidate-universe drift.
4. Run `validate_exact_date_catalog.py` against the 4,015-ID roster.
5. Emit coverage, rejects, data dictionary, validation JSON, and—only on PASS—the normalized `exact_date_catalog_v2.csv`.
6. Cross-check dates against `MRIMETA`/`MRI3META` or DICOM where available.
7. Run the deterministic <=90-day primary and <=180-day sensitivity pairing builder using its checksum-bound validation attestation.
8. Present the pairs and any QC/absence conflicts at human gate SL-H01 before any image transfer.

The validator always writes `approved_for_download=false`; it does not download or authorize images.

Example after a source-preserving join has been produced:

```bash
/home/ec2-user/exp/.venv_connectome_app/bin/python -m research_audit.validate_exact_date_catalog \
  --catalog research_audit/incoming/adni_exact_date_joined.csv \
  --roster research_audit/outputs/exact_date_t1_candidate_request_v2.csv \
  --source-name "ADNI MRIQC plus IDA Advanced Image Search" \
  --source-generated-at 2026-07-18T00:00:00Z \
  --qc-profile adni_mriqc_current \
  --default-qc-source MRIQC.SeriesQC \
  --out research_audit/cohort_v2/catalog_intake_20260718T000000Z
```

Use the actual export timestamp and filenames; the example names are placeholders, not claims about files already present.
