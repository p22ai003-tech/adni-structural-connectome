# Pipeline — End to End (raw ADNI → AAL166 structural connectome)

Code: `exp/connectome_pipeline/` (production) + `exp/scripts/scforge/live/run_sc_route_sota.py` (recovery
root). Notebooks: `exp/notebooks/structural_connectome_{A,B}.ipynb`. Stage→dir map: `docs/DATA_LAYOUT.md`.

## Stages

| # | Stage | In → Out dir | Tool / script |
|---|---|---|---|
| 1 | DWI → MIF | `Images/dti/<base>` → `mif_dwi/<sid>.mif` | `connectome_pipeline/dwi_convert.py` |
| 2 | Denoise | `mif_dwi` → `mif_denoised/<sid>_den.mif` | `dwidenoise` |
| 3 | De-Gibbs | → `mif_unringed/<sid>_den_unr.mif` | `mrdegibbs` |
| 4 | Eddy/motion | → `eddy/<sid>_preproc.mif` (+ b0 in `dwi_t1_bbr/`) | `dwifslpreproc` (FSL eddy) |
| 5 | Bias correct | → `biascorr_1/<sid>_unbiased.mif` | ANTs N4 |
| 6 | T1→b0 reg | `Images/mri` + b0 → `dwi_t1_bbr/`, `t1_anat/`, `t1_fast/` | FSL FLIRT 6-DOF **BBR** + MRtrix transform |
| 7a | 5TT + GMWMI | → `fod/<sid>/{5tt_b0,gmwmi}.mif` | `5ttgen`, `5tt2gmwmi` |
| 7b | FOD | → `fod/<sid>/wmfod_final.mif` | dwi2response (dhollander) + dwi2fod (SS3T CSD) + mtnormalise |
| 7c | Tractography | → `tracks/<sid>/tracks_final_3000k.tck` | `tckgen` (iFOD2) |
| 7d | SIFT2 | → `tracks/<sid>/sift_weights.txt`, `mu.txt` | `tcksift2` |
| 7e | Parcellation | AAL3 MNI → `parc/<sid>/AAL_b0.nii.gz` | MNI→T1→b0 warp (166 nodes) |
| 7f | DTI metrics | → `dti/<sid>/{fa,md,ad,rd,dt}.mif` | tensor fit |
| 7g | Assignment | → `tracks/<sid>/assignments_aal.csv` | endpoint→node |
| 7h | Connectomes | → `connectomes/SC_AAL166_<sid>_<weight>.csv` (9) | `tcksample` + `tck2connectome` |

## Stage-7 commands (exact)
```bash
# tractography (production 3M; recovery used 10M via run_sc_route_sota.py)
tckgen <wmfod_final.mif> <tracks_final_3000k.tck> -select 3000000 -seed_gmwmi gmwmi.mif -act 5tt_b0.mif
# (recovery/noACT: -seed_dynamic <wmfod> -mask <mask_fod> -select 10000000 -cutoff 0.06, NO -act)

tcksift2 <tracks.tck> <wmfod_final.mif> <sift_weights.txt> -out_mu mu.txt        # (+ -act on production)

tcksample <tracks.tck> <dti/<sid>/{fa,md,ad,rd}.mif> <{m}_mean.tsf> -stat_tck mean    # ×4 diffusivity

tck2connectome <tracks.tck> <AAL_b0.nii.gz|aal3_166_dwi.mif> SC_AAL166_<sid>_<weight>.csv \
   -symmetric -zero_diagonal -tck_weights_in sift_weights.txt \
   -assignment_radial_search <R> [-stat_edge mean] [-scale_invnodevol|-scale_length]
```
**9 weights:** count, count_invnodevol, fd_sum, len_mean, invlen_mean, fa_mean, md_mean, ad_mean, rd_mean.
**Assignment radius R:** production = **4 mm**, recovery = **2 mm** (see `DECISIONS.md`; density is monotonic in R).

## QC gate
Good connectome = AAL166 `count.csv` **density ≥ 0.6** (off-diagonal nonzero fraction). 648 cohort → **530 good**.
Bands: good ≥0.6 (`connectomes/`), mid 0.4–0.6 (`connectomes/_midband/`), low <0.4 (`connectomes/_lowband_archive/`).

## Source of truth + audit
`exp/pipeline/audit/subject_manifest.py` → `subject_manifest.csv` (density from `count.csv`, per-stage
presence). `stage_funnel.py` → funnel. `build_quality_views.py` → `_by_quality/` symlink views.
Live monitor: `exp/pipeline/monitoring/status_writer.py` → `connectome_status.json` → dashboard (8501).
