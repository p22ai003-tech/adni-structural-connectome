# Exact-date catalog source inventory

**Generated:** 2026-07-18 09:16:35 UTC  
**Authenticated AWS evidence:** 2026-07-18T08:33:48Z  
**Purpose:** quantify exact-date coverage, preserve the replacement-source audit, and support the approved local-only analysis design.  
**Safety:** read-only audit; no image download and no production mutation.

## Decisive result

The current selected DTI date is recoverable for all 530 records from dti_master.csv. The current selected T1 acquisition date is also recoverable for all 530 from the raw image-directory date. This repairs the date audit for current pairs, but not for proposed replacements.

Exact-date nearest-neighbour selection cannot be based on the earlier one-candidate-per-subject rounded-age screen. The frozen request universe therefore contains all 4,015 `Type=Original` T1-like Image IDs found in the frozen local all_mri source for all 530 audited subjects, with 1–38 candidates per subject. Its SHA-256 is `4a935b7bbd7d6d5ea5c1c8d80f4cc8fde840abaa88c0d4d2419878591bcc2406`. It contains all 530 rounded-age screen IDs, including the 330 provisional replacement IDs as a nested priority subset; 330 is not the complete export universe.

The 330 provisional replacement IDs all exist in all_mri.csv, which has phase and rounded age but no exact date. None exists in the older exact-date mri_master.csv, and none is present in the local raw MRI tree. More broadly, the old mri_master overlaps only 901/4,015 full-roster IDs across 161/530 subjects and does not provide the complete source-QC accounting required by the intake validator. A fresh authoritative ADNI/LONI export would have to resolve every frozen Image ID with an exact date and recognized QC, expose any candidate drift relative to the frozen local source, and yield at least one eligible T1 for every subject before replacement pairing could run.

## Approved project-scope override

On 2026-07-18 the user fixed the study scope to images already available locally. External replacement acquisition is therefore **not pursued** under the current plan. The active cohort-design path preserves the 530 current pairs and models/reports their real timing, acquisition, and QC limitations. This scope decision does not convert long-gap or low-quality inputs into valid same-visit measurements; it changes them from acquisition blockers into explicit design constraints and sensitivity variables. The signed project record is `decisions/closed_world_data_scope_20260718.md`.

## Verified local coverage

| Measure | Result |
|---|---:|
| Frozen exact-date request IDs | 4,015/4,015 unique |
| Subjects represented in frozen request | 530/530 |
| Candidate range per subject | 1–38 |
| Full-roster IDs represented in frozen local all_mri snapshot | 4,015/4,015 |
| Full-roster IDs represented in old mri_master | 901/4,015 across 161/530 subjects |
| Full-roster IDs present in local raw MRI | 305/4,015 across 305/530 subjects |
| Current selected T1 IDs also in Original-T1 request universe | 305/530 |
| Rounded-age screen IDs nested in full request | 530/530 |
| Provisional replacement IDs nested in full request | 330/330 |
| Selected DTI IDs with exact dti_master date | 530/530 |
| Current T1 IDs with raw-path date | 530/530 |
| Current pairs within 90 days | 194/530 |
| Current pairs within 180 days | 197/530 |
| Current pairs over 180 days | 333/530 |
| Current IDs represented in old mri_master | 57/530 |
| Raw-path dates matching mri_master where available | 57/57 |
| Proposed replacements represented in frozen local all_mri snapshot | 330/330 |
| Proposed replacements represented in exact-date mri_master | 0/330 |
| Proposed replacements present in local raw MRI | 0/330 |

The raw path date is strongly corroborated: it agrees with mri_master Study Date for every current T1 where both are available. It remains a file-provenance date, not a substitute for an authoritative source export for images that are not present locally. Only 305 current selected IDs occur in the frozen roster because that roster intentionally enumerates `Type=Original` T1-like catalog rows; this count must not be misread as loss of current raw-path date evidence, which remains 530/530.

## The three 90-to-180-day current pairs

These were rounded-age “contemporaneous” in the earlier screen but fail the proposed 90-day primary rule while passing the 180-day sensitivity rule.

| Subject | Group | DTI image | Current T1 image | DTI date | T1 path date | Absolute gap days |
|---|---|---:|---:|---|---|---:|
| 003_S_5165 | AD | 388986 | 392426 | 2013-08-16 | 2013-05-16 | 92 |
| 007_S_5196 | AD | 390043 | 377766 | 2013-09-13 | 2013-06-03 | 102 |
| 016_S_4353 | AD | 295021 | 270023 | 2012-03-31 | 2011-11-18 | 134 |

## Source ranking

| Source | Exact date | Full request-universe coverage | Image QC | Decision |
|---|---|---:|---|---|
| Fresh ADNI/LONI image or download export | Expected | Every frozen ID resolved with exact date and recognized QC; candidate drift exposed; at least one eligible T1 for each of 530 subjects | Expected/linked | Scientifically preferable replacement route, but not pursued under approved local-only scope. |
| dti_master.csv | Yes | DTI only; 530/530 selected DTI IDs | Partial acquisition fields | Authoritative for current DTI dates in this audit. |
| Local/S3 raw image-directory path | Yes for locally held image | 305/4,015 full-roster IDs; 0/330 provisional replacements | No catalog QC | Valid provenance evidence for current files; not a complete intake source. |
| mri_master.csv | Yes | 901/4,015 full-roster IDs across 161/530 subjects; 0/330 provisional replacements | Limited, not the required source-QC field | Incomplete legacy source; cannot release pairing. |
| all_mri.csv | No | 4,015/4,015 candidates relative to the frozen local snapshot, including 330/330 provisional replacements | No | Roster-construction source only; not evidence of current IDA completeness and cannot approve pairing. |

## Live S3 inventory

The legacy raw prefixes are s3://<your-bucket>/exp/Images/mri/ and s3://<your-bucket>/exp/Images/dti/, each with 1,114 subject prefixes. The alternative s3://<your-bucket>/exp/data/Images/ prefix is empty. The recorded cloud cohort prefix contains the same 13 local catalog files, last modified 2026-04-20, so it does not supply the missing full-universe dates and source QC.

The named `aml` principal recorded PASS at 2026-07-18T08:29:57Z in account <AWS_ACCOUNT_ID>. Its checksum-bound, read-only catalog search at 2026-07-18T08:33:48Z checked 161 visible bucket names and the documented project-known/plausible locations. It found no fresh authoritative ADNI/LONI exact-date export. The preserved evidence is `aws_aml_principal_verification.json` (SHA-256 `04d33ee533946005a12f756856a1aa14ee0e71ee2f11323fb96c7d36b875dfc1`) and `aws_aml_catalog_search.json` (SHA-256 `9581ca4f7be2af2bd21d0cb1eb6976d71a1d2625fe66389ca111af3a3da148d9`). This report generator reads those snapshots; it does not reauthenticate, search AWS, or transfer an object.

## Active local-only path

1. Lock the 530 current DTI–T1 pairs, exact intervals, local paths/hashes, diagnosis, site/protocol, acquisition, and QC fields.
2. Attempt corrected processing on all available inputs. Preserve technical failures and attrition; do not promote invalid registrations, matrices, or tensor outputs.
3. Use the full available cohort as exploratory evidence and prespecify timing (all, <=180 days, <=90 days), protocol/site, and QC sensitivity analyses.
4. Calibrate the manuscript to available-data evidence and disclose that interval adjustment cannot reconstruct missing contemporaneous anatomy.

## Contingency replacement path — superseded

If the data-scope decision is explicitly reopened, the replacement route would require the following:

1. Export or otherwise recover authoritative ADNI/LONI MRI metadata for the frozen 4,015-ID request universe, including exact Study Date and source-backed image QC/provenance. Preserve the 330 provisional IDs only as a nested priority cross-check.
2. Resolve every frozen Image ID with an exact date and recognized QC, expose any candidate drift relative to the frozen local source, and require at least one eligible T1 for every one of the 530 subjects. Missing or unknown requested IDs block pairing by default.
3. Release the catalog to pairing only when validation records `status=PASS`, `release_status=PAIRING_READY`, and `pairing_ready=true`; join it to the 530 exact DTI dates and apply the frozen <=90-day primary and <=180-day sensitivity rules.
4. Reconcile diagnosis, present the selected pairs at SL-H01, and only then enumerate the exact nonlocal image objects and bytes for separate transfer approval. The final transfer count is determined by exact-date/QC selection and local reconciliation, not capped at 330.

## Reproducibility

Inputs:

- /home/ec2-user/exp/research_audit/outputs/visit_matched_t1_manifest_dryrun.csv — SHA-256 b4b01c5ffb5f1750b15a753f8baa5732fdb1221b3d53a0d3010bc564eafe2f94
- /home/ec2-user/exp/cohort/dti_master.csv — SHA-256 0e8d78b927fe1a7cd731279b37445900b2e39a40fc7ad0fd3551b687c1b90b92
- /home/ec2-user/exp/cohort/mri_master.csv — SHA-256 161577069ec6a7a76f75ae3393d3536573cc271c87af477006a9965b9e24ab38
- /home/ec2-user/exp/cohort/all_mri.csv — SHA-256 1cf186ae47aa7aad9cf6e40a34a8470157b9fc4a06453268855d8971e22f693a
- /home/ec2-user/exp/research_audit/outputs/exact_date_t1_candidate_request_v2.csv — SHA-256 4a935b7bbd7d6d5ea5c1c8d80f4cc8fde840abaa88c0d4d2419878591bcc2406
- /home/ec2-user/exp/research_audit/outputs/exact_date_t1_candidate_request_validation_v2.json — SHA-256 86f5dc9288e85e3e61a96aefece5bef0c2879e43e56b5021e54ad557c071d9a2; recorded status PASS
- /home/ec2-user/exp/research_audit/outputs/aws_aml_principal_verification.json — SHA-256 04d33ee533946005a12f756856a1aa14ee0e71ee2f11323fb96c7d36b875dfc1
- /home/ec2-user/exp/research_audit/outputs/aws_aml_catalog_search.json — SHA-256 9581ca4f7be2af2bd21d0cb1eb6976d71a1d2625fe66389ca111af3a3da148d9

Outputs:

- /home/ec2-user/exp/research_audit/outputs/current_t1_exact_date_evidence_v2.csv
- /home/ec2-user/exp/research_audit/outputs/replacement_t1_catalog_coverage_v2.csv
- /home/ec2-user/exp/research_audit/outputs/catalog_coverage_v2.csv
