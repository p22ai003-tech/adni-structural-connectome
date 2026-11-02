# Recipe v4: one frozen from-raw ACT recipe, and its bench

Rule (fixed before any result, never edited here): `/data/derivatives/recipe_v4/bench/RULE.md`. Brief: `/data/derivatives/recipe_v4/BUILD_BRIEF.md`.
Nothing here reads diagnosis, research group, CDR or MMSE. Inputs are `lists/bench_units.csv`, `lists/t1_choice.csv` and each scan's own DICOM sidecar.

| File | What it does |
|---|---|
| `rv4_subject.py` | One session, one arm, resumable stages. `--arm A\|B\|C`, `--select`, `--seed`, `--tag` (variant of an existing arm run), `--out`, `--threads`. `--selftest` needs no participant data. |
| `rv4_run.sh` | Driver: `canary`, `all`, `status`. Lowest priority, 5 x 8 threads (2 x 20 for 10M), never above 40. Waits for memory and disk. One retry per failed run. Writes `bench/DONE` at the end. |
| `rv4_compare.py` | Applies RULE.md mechanically from the QC jsons. Writes `bench/compare.md` and `bench/compare.json`. Sessions appear as S01..S24. |

## Run

```bash
cd /data/derivatives/recipe_v4/scratch      # never launch from inside docs/iit_dipanjan/tools
T=/home/ec2-user/exp/docs/iit_dipanjan/tools/recipe_v4
( setsid nohup nice -n 19 bash $T/rv4_run.sh all >> /data/derivatives/recipe_v4/bench/logs/driver.log 2>&1 & )
bash $T/rv4_run.sh status                   # counts only, no identifiers
touch /data/derivatives/recipe_v4/bench/STOP   # gentle stop: running jobs finish, nothing new starts; delete it to allow a restart
```

`all` runs arms B, C and A of each session together, strata in turn; the second seed on arm B once those have finished; then 10M (and arm C's variants if the rule adopts C); then the final `compare.md` and `bench/DONE`. A job reads `rv4_subject.py` once, at its start: a fix reaches jobs that start later, not one already running. Replace `rv4_run.sh` by rename, never in place, because bash reads a running script by offset.

Restart the same command after any interruption. Finished runs are skipped, finished stages are reused, and a stage whose version in `VERSIONS` was raised is redone along with everything after it. If production needs the machine, `kill -USR2 <pid of the driver's xargs>` lowers the number of parallel jobs by one (`-USR1` raises it again; never above 5 at 8 threads).

## Where things go

- `bench/qc/<arm>_<unit>.json`: everything the rule needs, status, seconds and peak disk per stage. `<arm>` is `A`, `B`, `C`, `B_seed2`, `B_10M`, and so on.
- `bench/work/<arm>/<unit>/`: `endpoints_f32.npy` (n, 2, 3), `lengths_f32.npy`, `weights_f32.npy`, `mu.txt`, `{fa,md,ad,rd}_f32.npy`, `assignments_u8.npy`, `matrices_aal166.npz` (166 labels), eddy's movement and outlier files, `align.json`, `nodes166.nii.gz`, `aparc_dwi1mm.nii.gz`, `fa1mm.nii.gz`, and for arms B and C the FOD and 5TT needed by the variants.
- `bench/work/_shared/<unit>/`: sidecar, denoised series (deleted once B and C have both finished eddy), T1 in NIfTI, CPU FastSurfer if it was needed, MNI-to-T1 warp.
- The tractogram does not survive `s6d_assign`. It is deleted only after assignments rebuilt from the saved float32 endpoints equal `tck2connectome` on the full tractogram (`assign_proof` in every QC json).
- Scratch, `MRTRIX_TMPFILE_DIR` and every `-scratch`: `/data/derivatives/recipe_v4/scratch/`.

## Choices the rule leaves open (all fixed before any bench result)

Written down so that nobody has to read the code to find them, and so that the same choices go into the cohort run.

1. **Raw data** is converted from DICOM by dcm2niix, so the sidecar describes these exact voxels. On Siemens and GE the result equals `mif_dwi` voxel for voxel. On Philips the volumes are the same but in another order (b=0 first here; gradients stay with their volumes; checked on 2 sessions by matching volumes).
2. **Phase encoding** comes from `PhaseEncodingDirection`. Philips sidecars give only `PhaseEncodingAxis` and `EstimatedTotalReadoutTime` (from the water-fat shift). Polarity is not stored, so + is used. This changes nothing in practice: without a field, eddy's result does not depend on polarity or readout scale, and with Synb0 the same acquisition line goes to topup and eddy, so sign and scale cancel in the displacement.
3. **Eddy options**: `--repol --slm=linear --data_is_shelled`, the stored pipeline's options plus the rule's outlier replacement. Mask: `dwi2mask` then dilate, as `dwifslpreproc` does it, arm C after `applytopup`.
4. **Arm C**: the b=0 volumes are aligned to the first (`mcflirt`) before averaging, so the Synb0 input is sharp and sits in eddy's reference frame. Odd axes are padded by repeating the last plane, as `dwifslpreproc` does, and cropped after eddy, so B and C end on one grid. T1 input: `t1_anat/corrected_T1_<subj>_I<t1_iid>`, the same file the stored FastSurfer read.
5. **FastSurfer product**: `aparc.DKTatlas+aseg.deep.mgz` for every session, stored or CPU, because the CPU route runs `--seg_only` and one product keeps the recipe uniform. Stored `aparc+aseg.mgz` (surface-mapped) is not used. Reuse requires the log's segmentation `--t1` to carry the chosen image ID.
6. **Whole-brain WM Dice**: FastSurfer labels 2 and 41 against FA>0.25 on the 1 mm diffusion grid, the `reg_wm_dice` definition of `rb_dk.py` that the .65 bar comes from. In the `.deep` product those labels include the corpus callosum. The 5TT-WM Dice is also in the QC, for information.
7. **Regional mask**: AAL3 labels OFCmed, OFCant, OFCpost, OFClat, Rectus, Frontal_Med_Orb, Frontal_Inf_Orb_2; Temporal_Pole_Sup, Temporal_Pole_Mid; Hippocampus, ParaHippocampal, Amygdala (left and right), carried from MNI through the T1 by each arm's own transform.
8. **Second-seed "whole-brain value r"**: the project's existing definition (`p5_features`, `p5_act`): per network and measure on the 154 nodes, Pearson r across the sessions between the two seeds. Criterion: the lowest per-measure median over networks.
9. **Sessions for the variants**: seeded draw (20261102) within acquisition stratum; 10M one per stratum, second seed 2/1/2/1 (`bench/variants.csv`).
10. **Not used because the rule does not list it**: `tcksift2 -fd_scale_gm`.
11. **BBR**: `epi_reg` with the masked mean b0 and FastSurfer's WM (labels 2, 41, 77) as `--wmseg`. The FLIRT matrix goes to ANTs through `flirt_to_itk` in `rv4_subject.py` (`c3d_affine_tool -fsl2ras` segfaults on this machine). Every run checks the conversion against FLIRT's own resampling (`bbr_flirt_vs_ants_dice`, must be at least .98). If `epi_reg` itself fails on a session, that session keeps rigid and `compare.md` counts it.
12. **Seeds**: `MRTRIX_RNG_SEED` is a hash of session and `--seed`, so arms A, B and C of a session share a seed. `ANTS_RANDOM_SEED` is fixed, because unseeded ANTs rigid moved a session's Dice by .009 between two runs (.759, .750), which matters beside a +.02 bar.
13. **Arm A with no gradients in the stored file** takes the table from `mif_dwi`, as `rb2_subject.py` did, not eddy-rotated (`grad_source` in the QC). A stored eddy output that is not a whole brain fails at the mask check and counts as an arm A failure only.

## Known limits of the frozen recipe

- GE (and any zero-filled) data are interpolated in-plane, and many Siemens scans are partial Fourier, so `dwidenoise` and `mrdegibbs` run outside their stated assumptions. The rule fixes them for everyone; QC records `zero_filled` and `partial_fourier` for each session.
- Stored eddy outputs of at least 3 of the 24 bench sessions (all GE, none with a stored FastSurfer) are not whole brains: their mean b=0 has signal in a few slices only and the DWI mask comes out at 2 to 70 mL, against about 1500 mL from raw. Arm A fails at the mask check for those sessions, by design. Arm A is a reference, not a criterion in the rule, and arms B and C work from raw.
- Attempts spent on code bugs before the full bench was launched are in the `attempt` field of the QC. The rule's "after one retry" is applied to the final status of each run in the full bench, where each failed run gets one retry.
- Arm D (measured field map) is not built. Its series are not on disk; `rv4_compare.py` looks for them every time and says so if they appear.
