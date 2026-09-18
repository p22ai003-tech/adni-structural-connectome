# SL-H04A-R1 — targeted response-calibration recovery recommendation

**Decision state:** AWAITING USER APPROVAL  
**Prior authorization:** SL-H04A remains unchanged; the active four-core run is not modified by this recommendation.

## Proposed authorization

Authorize a new isolated Phase-A response-calibration recovery attempt with:

- the 15 technically eligible units already contained in the approved diagnosis-blind 24-unit canary;
- seven Siemens and eight GE DWI series;
- seven NIfTI-single and eight DICOM-series T1 sources;
- pinned `dcm2niix` v1.0.20260416 full DICOM conversion, executable SHA-256 `353acb3b370faade70b552f284e0a306214a146c076c9f449c5c555b9b148ebd`;
- an isolated new run root and write-once source/workflow/environment manifests;
- maximum 32 CPU cores, 24 wall-clock hours, and 100 GiB additional storage;
- diagnosis/outcome-blind technical eligibility and response QC;
- a hard stop after Phase-A response outcomes and calibration freeze decision.

The attempt must not run FOD reconstruction, tensor maps, tractography, SIFT2, connectome matrices, statistical analyses, or dashboard publication. A pooled response may freeze only if at least 12 units pass and the signed two-manufacturer/two-T1-source diversity contract is satisfied.

## Why a new gate is required

The current SL-H04A approval binds the existing converter/workflow hashes and a four-core cap. Silently changing the converter or compute limit would invalidate that provenance. The recovery attempt therefore requires a new explicit decision while retaining the current attempt as evidence.

## Requested response

Reply **`Approve SL-H04A-R1`** to authorize exactly this bounded recovery attempt. Any other response leaves it unlaunched.

