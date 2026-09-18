# exp/pipeline — consolidated scripts

Canonical home for the pipeline + recovery + monitoring scripts (previously scattered in `/tmp`, at risk
of loss on reboot). Docs in `../docs/`. Production stage code remains in `../connectome_pipeline/` and
`../scripts/scforge/live/run_sc_route_sota.py`.

| Dir | Contents |
|---|---|
| `audit/` | **`subject_manifest.py`** (→ subject_manifest.csv, single source of truth), `stage_funnel.py`, `build_quality_views.py` (→ `_by_quality/` symlink views), `derive_density_check.py`, `density_check.py`, `classify_archive.py` (band quarantine), `rebuild_count.py`, `subject_audit.py` |
| `monitoring/` | **`status_writer.py`** (live → connectome_status.json, read by dashboard), `launch_status.sh`, `simple_mon.py`/`.sh`, `lane_frame.py` |
| `recovery_lanes/` | `radial4_standardize.py`, `batch162.sh`, `fast_1x96.sh`, `fast_crashers.sh`, `freshreg_retry.sh`, `mcicn_recover.sh`, `ad_blitz.sh` (engine: `../scripts/scforge/live/run_sc_route_sota.py`) |
| `sync/` | `s3_push_pass.py`, `sync_connectomes.sh` |
| `cleanup/` | `batch162_diskguard.py`, `launch_diskguard.sh` |
| `preprocess/ registration/ fod_tracto/ connectome/` | (reserved; production code currently in `../connectome_pipeline/`) |
| `archive_oneoffs/` | superseded one-off/diagnostic scripts |

## Routine commands
```bash
PY=/home/ec2-user/exp/.venv_connectome_app/bin/python
$PY pipeline/audit/subject_manifest.py     # rebuild manifest (after any connectome change)
$PY pipeline/audit/stage_funnel.py         # rebuild funnel
$PY pipeline/audit/build_quality_views.py  # rebuild _by_quality symlink views
bash pipeline/monitoring/launch_status.sh  # (re)start the live status writer
```
Docs: `../docs/{PIPELINE,RECOVERY_LANES,PARAMETERS,DECISIONS,DATA_LAYOUT}.md`.
