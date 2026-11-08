# Recipe v4, as frozen for the cohort run

What the signature in `SIGNED_RECIPE.json` covers: this text, and the code whose SHA-256 it lists. Chosen by `bench/RULE.md` (fixed Mon 2 Nov 2026 before any result; one change, the GE crop, on the user's written OK of Tue 3 Nov 06:02 IST) and the bench's final comparison (`bench/compare.md`, GO, arm C), with the user's freeze decisions of Tue 3 Nov about 13:20 IST (`FREEZE.md`). No diagnosis is read anywhere in the run.

## Sessions
702 of the 709 sessions with a stored eddy output (`lists/cohort_units.csv`): ADNI-3 631 (Siemens 403, GE 127, Philips 101), ADNI-2 57, ADNI-4 14. Left out, 7: no same-visit T1 (5 none in IDA's catalogue, 1 whose T1 IDA's search did not return); and the one scan IDA labels MPTRONIC, whose headers were rewritten by another tool (manufacturer Philips on a GE DISCOVERY MR750w model) and carry no readout time or echo spacing, so step 1 cannot be applied (found by the canary). ADNI-3 alone is the primary analysis; SMC is held out of MCI. Analyses, not this run, decide that.

## Per session (`rv4_subject.py --arm C --select 10000000 --subset 3000000 --crop-zf`, then `--tag noact --select 3000000`; driver `rv4_cohort.sh`)
1. Raw DWI from DICOM by dcm2niix, so the sidecar describes these voxels: phase-encoding axis and readout time from the sidecar (Philips: axis only, + polarity, readout estimated from the water-fat shift).
2. A scan the scanner interpolated in-plane (ReconMatrixPE above AcquisitionMatrixPE by more than 10%) goes back onto its acquired lines, and its readout to echo spacing x (acquired lines - 1). Then dwidenoise, mrdegibbs.
3. Synb0-DisCo (native CPU port, 5 folds) from the mean b=0 and the same-visit T1; topup on the distorted and synthetic b=0; eddy_cpu `--repol --slm=linear --data_is_shelled` with the topup field.
4. b=1000 shell if multi-shell; N4; mask; dhollander WM response; single-tissue CSD lmax 8; tensor maps.
5. Same-visit T1: stored corrected T1, or the IDA download brought to the same form (dcm2niix, N4 with the stored parameters; `rv4_t1_ingest.py`). FastSurfer `aparc.DKTatlas+aseg.deep` (stored if it read that T1, else CPU `--seg_only`).
6. T1 to b=0: ANTs rigid and BBR, the one with the higher WM-to-FA>0.25 Dice kept. 5TT by MRtrix's own FreeSurfer table (`-sgm_amyg_hipp`; ventral DC is WM). AAL3 from MNI through the T1 (SyN) with the same transform; 166 labels kept, the 12 brainstem nuclei not estimable, analyses on 154 nodes; radial search 2 mm.
7. ACT run: iFOD2 `-act -backtrack -crop_at_gmwmi`, dynamic seeding, cutoff 0.06, angle 45, 10-250 mm, step 1 mm, 10,000,000 streamlines; `tcksift2 -act`, mu saved. Per streamline: endpoints, length, SIFT2 weight, mean FA/MD/AD/RD, node assignment (checked against tck2connectome on the full tractogram before it is deleted).
8. 3M subset: the first 3,000,000 streamlines of the same tractogram (tckgen writes them in random seed order) with their own SIFT2 fit.
9. No-ACT run, the labelled sensitivity arm: the same FOD, atlas, settings and random seed without `-act -backtrack -crop_at_gmwmi` and without `tcksift2 -act`, at 3M streamlines, so that it compares like for like with the ACT run's 3M subset (the user's choice, 3 Nov; also the count of the earlier no-ACT data).
10. Then the FOD, tissue map and MNI-to-T1 warp are deleted to fit 702 sessions on disk; everything an analysis reads is kept.

## Quality
- Every session's whole-brain WM Dice (FastSurfer labels 2 and 41 against FA>0.25) is in its QC file. Under .65 flags the session (`cohort/flags.csv`): it is looked at with diagnosis hidden, and every analysis is repeated without the flagged sessions. Reason: bench session S02, where Synb0's field was wrong (r .18 against the measured field map) and WM Dice fell from .732 to .630.
- A failed run gets one retry; failures are reported, not hidden.

## Not in this recipe
SS3T, FSL 5TT, SyN realignment of the diffusion data, a density floor, `tcksift2 -fd_scale_gm`. No measured field maps (GE has none, so the recipe would not be uniform).

## Known limits
- dwidenoise and mrdegibbs run outside their stated assumptions on partial-Fourier scans; the QC records `partial_fourier`.
- 30 ADNI-2 GE sessions use IDA's processed T1 (gradient-warp and N3 corrected), the only same-visit T1 listed for them.
- Philips polarity is assumed; the bench showed Synb0's correction does not depend on it, and all six Philips field-map comparisons came out positive.
