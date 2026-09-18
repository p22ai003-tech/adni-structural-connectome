# Decisions & Root Causes — why the pipeline is the way it is

## The 3 root causes that lifted density 0.09 → ~0.8
1. **Atlas placement.** The old pipeline crushed the AAL3 atlas onto a 2 mm b0 grid with an affine-only
   MNI→T1 warp → atlas mis-placed → density ~0.09. **Fix:** 1 mm AAL3 in DWI world-space + ANTs **SyN**
   nonlinear MNI→T1→b0 → ~0.66.
2. **Underpowered tracking.** 3M streamlines is too few for 166 nodes → ~0.66. **Fix:** **10M** streamlines
   + SIFT2 → ~0.8.
3. **ACT broken on anisotropic data.** ADNI DWI is thick-sliced (≈1.37×1.37×2.7 mm); 5TT under-segments →
   ACT rejects ~all streamlines (0.02% acceptance vs 96% without). **Fix:** **noACT** dynamic seeding
   (auto-detected for anisotropic; cohort median 0.19 ACT → 0.62 noACT).

## Assignment radius — accept the mix (radial 4 prod / 2 recovery)
The dataset is **471 radial-2 (recovery) + 59 radial-4 (production)**. Standardizing fully to radial 4
would need re-tracking ~349 subjects whose 10M tracks were purged (≈1 day on a 128 MB/s disk); standardizing
down to 2 risks regressing production subjects near 0.6. Density is monotonic in radius. **Decision:
accept + document** — keep the dataset as-is, record each subject's radius in the manifest, covary/filter in
analysis. (Re-assigning *up* to 4 is safe-but-expensive and available via `radial4_standardize.py`.)

## Count must come from production files, not result.json
The live count once read recovery `result.json` densities and over-reported (claimed 528, only 447 were
truly written to production; 81 good connectomes were stranded in run-roots). **Decision:** the count and
the dashboard read **only the production `SC_AAL166 count.csv`** (via `subject_manifest.csv`). Stranded
good connectomes were copy-promoted into production → **530**, real-on-disk.

## Sub-standard handling
<0.4 and 0.4–0.6 are **archived, not deleted** — `_lowband_archive/` and `_midband/` (local + S3), best
available kept for investigation. Source inputs (eddy/fod/dti/tracks) are never deleted.

## Disk cleanup
Reclaimed ~2.9 TB by deleting recovery run-roots (10M tracks; connectomes already promoted + S3-backed),
scratch, and failed-stage intermediates — after a comprehensive AAL166 orphan check confirmed 0 good
connectomes would be lost. Ambiguous items archived, not deleted.

## Final dataset
**530/648 good** (AD 78 · MCI 201 · CN 251). EC2↔S3 mirrored (tck/tsf/csv/sift all match). The remaining
118 are genuinely data/registration-limited (mid 5, low 113) and kept in band archives.
