# 10M Connectome Rebuild — Resume Note (2026-06-12, updated ~15:55 UTC)

## Instance / data
- c7i.48xlarge, 192 vCPU, 371 GB. Data on persistent EBS root `/` (survives stop/start). EFS at `/mnt/efs`.

## THREE root causes of low density — all FIXED, applied cohort-wide
1. **Atlas placement**: old pipeline crushed AAL3 onto 2mm b0 grid w/ affine-only reg. Fix = ANTs SyN nonlinear (MNI->T1 fullhead) + AAL3 kept at 1mm in DWI world-space. 0.09 -> 0.66.
2. **Underpowered tracking**: 3M streamlines too few for 166 nodes. Fix = 10M streamlines + SIFT2. 0.66 -> ~0.8.
3. **ACT broken on anisotropic data**: thick slices (e.g. 1.37x1.37x2.7mm) -> partial-volume under-segments the 5TT WM -> ACT rejects ~all streamlines (measured 0.02% acceptance vs 96% without ACT, same FOD). Fix = auto-detect anisotropy (sz > 1.4*max(sx,sy)) and route those subjects to FOD tracking WITHOUT ACT (-seed_dynamic + brain -mask); keep full ACT+GMWMI+backtrack for isotropic. **USER DECISION (confirmed): keep this — no-ACT for anisotropic, ACT for isotropic.** Recorded per-subject as `track_mode` in result.json. Affects 128 subjects (AD 56 / MCI 24 / CN 48).

## Runner code state (run_sc_route_sota.py) — gen_tracks_10m
- Per-subject anisotropy routing (above), `track_mode` returned + logged.
- NO-defer default (TCKGEN_BUDGET_S=99999999); 100-min hard cap (TCKGEN_CAP_S=6000); seed cap (TCKGEN_SEED_CAP=200000000) so untrackable brains exit cleanly (no rc=-6 SIGABRT).
- Rejects tckgen rc not in {0,-2,124,130} (killed/corrupt partial -> FAIL, never builds empty connectome).
- process_subject: resume guard (skip status=ok & density>=RESUME_MIN_DENSITY[0.6]) -> safe re-runs.
- promote_sota_to_production.py: manifest MERGES (per-group promotions accumulate).

## CURRENT RUN STATE
- AD: relaunched with ALL 3 fixes. run-root route_sota_AD10M_20260612T104621Z. Launch: /tmp/launch_ad_nodefer.sh.
- **MASTER supervisor** (detached, ppid=1): `/data/derivatives/qc/sc_matrix_qc/cohort_supervisor.sh`. Waits AD -> promote -> retires old orchestrator + kills paused MCI -> runs MCI -> CN -> SMC (all fixed code, +janitor +promote) -> dashboard refresh+restart. Log: cohort_supervisor.log.
- Old orchestrator_10M.sh = RETIRED (superseded). Old paused MCI (SIGSTOP) gets killed+re-run by supervisor.

## Monitors
- tmux `cohort`: live table (TO DO/RUNNING/DONE>=.7/UNTRACK per group). `tmux attach -t cohort`.
- `python3 /data/derivatives/qc/sc_matrix_qc/super_monitor.py` (table) ; adaptive_tick.sh (table+flags).

## If rebooted: relaunch AD via /tmp/launch_ad_nodefer.sh then bash cohort_supervisor.sh (detached). Resume guard skips done.

## Do NOT touch: PE2 (pe_quant_fastapi_v3 :8502, batch_vessel_skeletonization); Codex route21 jobs. Only restart connectome-dashboard service (:8501).
