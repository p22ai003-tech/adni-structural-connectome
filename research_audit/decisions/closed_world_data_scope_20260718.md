# Data-scope decision — locally available images only

**Decision ID:** SL-D01  
**Recorded:** 2026-07-18 09:14 UTC  
**Decision state:** approved by the user  
**Scope:** use only DTI/T1 images already available to the project; do not make external replacement acquisition a dependency.

## Consequences

- Preserve all 530 currently available DTI–T1 pairs and their actual acquisition intervals.
- Do not describe the cohort as visit-matched: 194 pairs are within 90 days, 197 within 180 days, and 333 exceed 180 days.
- Treat interval in days, cross-phase status, site, scanner/protocol, acquisition variables, and quantitative QC as explicit design/sensitivity variables.
- Attempt corrected processing on all locally available inputs, subject to minimum mathematical/technical validity. A failed registration, non-finite matrix, invalid dimensions, or impossible tensor output is retained in the failure/attrition record rather than promoted as a biological measurement.
- Use the full available cohort as exploratory evidence and prespecify timing-, protocol-, and QC-restricted sensitivity analyses.
- Do not transfer or acquire replacement images under the current plan. The 4,015-ID source audit remains provenance and limitation evidence, not a download list.
- Calibrate the manuscript to an available-data observational analysis; the design cannot support an unqualified same-visit multimodal biomarker claim.

This decision changes the execution path, not the measured facts about acquisition mismatch or quality. Those limitations remain visible in all downstream reporting.
