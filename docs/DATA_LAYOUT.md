# Data Layout — Structural Connectome Project

_Last organized: 2026-06-17. Dataset: 648 ADNI subjects → **530 good** (≥0.6 density) AAL166 connectomes._

`/home/ec2-user/exp/data` is a **symlink to `/data`**. All derivatives live under `/data/derivatives/`.
**Do not move the stage directories** — the pipeline (`connectome_pipeline/`, `scforge/live/`) and the
dashboard hard-code these paths. Quality-band navigation is provided via **symlink views** (see below),
not by relocating data.

## Single source of truth
- **`/data/derivatives/qc/sc_matrix_qc/subject_manifest.csv`** — one row per subject: band, density
  (computed directly from the connectome `count.csv`), per-stage presence+path, EC2/S3 flags.
  Rebuild: `python exp/pipeline/audit/subject_manifest.py`.
- **`…/stage_funnel.csv`** — subjects present at each stage × cohort × band.
  Rebuild: `python exp/pipeline/audit/stage_funnel.py`.

## Pipeline stages → directories (raw → connectome)

| Stage | Dir (`/data/derivatives/…`) | Pattern | Notes |
|---|---|---|---|
| Raw DWI | `Images/dti/<base>/` | ADNI download (base = `site_S_num`, no I-id) | 281 G raw |
| Raw T1 | `Images/mri/<base>/` | ADNI download | |
| S1 mif | `mif_dwi/` | `<sid>.mif`,`.bval`,`.bvec` | sid = `site_S_num_Iimgid` (DWI image-id) |
| S2 denoise | `mif_denoised/` | `<sid>_den.mif` | intermediate (mostly consumed) |
| S3 unring | `mif_unringed/` | `<sid>_den_unr.mif` | intermediate (mostly consumed) |
| S4 eddy | `eddy/` | `<sid>_preproc.mif` | + b0 in `dwi_t1_bbr/` |
| S5 biascorr | `biascorr_1/` | `<sid>_unbiased.mif` / `<sid>_b0.nii.gz` | |
| S6 reg | `dwi_t1_bbr/`,`t1_anat/`,`t1_fast/` | `<sid>_t12b0_bbr_mrtrix.txt`; `T1_ss_<base>_*.nii.gz` | **T1 image-id ≠ DWI image-id** → match by `<base>` |
| S7 FOD/5TT | `fod/<sid>/` | `wmfod*.mif`, `5tt_b0.mif`, `gmwmi.mif`, `mask_*.mif` | |
| S7 DTI | `dti/<sid>/` | `dt.mif`, `fa/md/ad/rd.mif` | |
| S7 parc | `parc/<sid>/` | `AAL_b0.nii.gz` | AAL3 in b0 space (166 nodes) |
| S7 tracks | `tracks/<sid>/` | `tracks_final_3000k.tck`, `sift_weights.txt`, `{fa,md,ad,rd}_mean.tsf`, `assignments_aal.csv`, `mu.txt` | |
| **S7 connectomes** | `connectomes/` | `SC_AAL166_<sid>_<weight>.csv` (9 weights) + `SC_Schaefer200_*` | the dataset output |

**Funnel:** ~648 cohort subjects through mif/eddy/fod/dti/tracks → **530** good connectomes. Intermediate
dirs (`mif_denoised`, `mif_unringed`, `biascorr_1`) are sparse because those steps are consumed downstream.

## Quality bands
- **good (≥0.6):** `connectomes/SC_AAL166_*` (main dir) — 530 subjects.
- **mid (0.4–0.6):** `connectomes/_midband/` — kept for investigation.
- **low (<0.4):** `connectomes/_lowband_archive/` — archived best-available.

## Band-navigable symlink views (browse inputs **and** outputs by band)
`/data/derivatives/_by_quality/{good_ge0.6, mid_0.4_0.6, low_lt0.4, none}/<sid>/` — symlinks to every
stage file for that subject (raw → eddy → fod → dti → tracks → connectome). Zero data duplication.
Rebuild/reset: `python exp/pipeline/audit/build_quality_views.py` (or `rm -rf /data/derivatives/_by_quality`).

## Metadata / clinical / atlas
- Clinical + demographics (group, age, sex, MMSE, CDR, APOE): `exp/cohort/*.csv` (ADNI master lists).
- Group map (cohort → AD/MCI/CN): `…/sc_forge_v1/group_qc/current_connectome_subject_availability_with_group.csv`.
- Atlas: `exp/atlas/AAL/` (AAL3 1mm, 166-node label map), `exp/atlas/Schaefer2018/`.

## QC / audit
`/data/derivatives/qc/sc_matrix_qc/` — `subject_manifest.csv`, `stage_funnel.csv`, `subject_audit.csv`,
`connectome_status.json` (live monitor), `claude_repaired_manifest.csv` (recovery-promoted = radial-2 list),
plus historical QC CSVs.

## Backups / archive
`connectomes/_pre_*_backup/` (pre-recovery connectome snapshots), `connectomes_backup_*` (May/June),
`exp/archive/` (notebooks, recovery handoffs — see `ARCHIVE_INDEX.md`).

## S3 (`s3://sabeesh/exp/`)
- Connectomes: `connectomes/` (+ `_midband/`, `_lowband_archive/`). The 2026-07-18 read-only reconciliation found 4,755/4,779 canonical matrix/cohort files matching, 12 stale-size matrix objects, and 12 missing tensor-matrix objects across three locally reprocessed subjects. S3 is therefore **not an exact current mirror**; local SHA-256 snapshot records are authoritative until a separately approved upload.
- Tracks + tsf + sift: `tracks/<sid>/`.
- Source inputs: `eddy/`, `fod/`, `dti/`, `Images/`, etc.
- Docs + manifest: `_docs/` (`DATA_LAYOUT.md`, `subject_manifest.csv`, `stage_funnel.csv`).
