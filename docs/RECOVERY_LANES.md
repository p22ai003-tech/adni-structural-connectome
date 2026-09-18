# Recovery Lanes — what we tried, what worked, what failed

The base SOTA route produced ~500/648. Recovery lanes pushed it to **530**. All launchers live in
`exp/pipeline/recovery_lanes/`; the engine is `exp/scripts/scforge/live/run_sc_route_sota.py`. See
`PARAMETERS.md` for the env flags.

## The proven sturdy path (the 500)
Existing cached FOD (`fod/<sid>/wmfod_final.mif`, do **not** rebuild) + ANTs **SyN** atlas registration
(`--reg-mode fullhead`) + **10M** streamlines + **noACT** dynamic seeding (`FORCE_NOACT=1`) + promote ≥0.6.

## Lanes (chronological)

| Lane | Recipe | Scope | Outcome |
|---|---|---|---|
| **regen216** | REBUILD_FOD=1, 3M, SyN | 216 deleted-track good subjects | 210/216 reproduced ≥0.6 — works on already-good data |
| **ad_blitz v1** | REBUILD_FOD=1 **+ FORCE_RESPONSE** | hard AD | ❌ FAILED — rebuild collapsed healthy FODs (max→0) → ~0 streamlines |
| **ad_blitz v2** | REBUILD_FOD=0 (cached FOD) | hard AD | ceiling (~71/100) — most AD data-limited |
| **fastcrash** (`fast_crashers.sh`) | **REBUILD_FOD=1** (empty cached FOD) + fresh reg + 10M, 2×48 thr | 9 empty-FOD SIGSEGV crashers | ✅ **8/9** recovered (0.62–0.92) |
| **batch162** (`batch162.sh`) | fastcrash recipe, 16×12 thr, all pending | 162 sub-threshold | ✅ **39** recovered (mean 0.751); rest at data ceiling |
| **freshreg** (`freshreg_retry.sh`) | force fresh reg (`REREG_THRESH=0.99`) | healthy-FOD fails | mostly no gain — needed REBUILD_FOD=1 to generate the grid-matched b0_post |
| **radial4** (`radial4_standardize.py`) | re-assign 10M track at radial 4 (no re-track) | radial-2 recovery subjects w/ surviving track | recovers near-misses; monotonic, never regresses |

## Key lessons (promoted from recipe_ledger)
1. **Don't rebuild a healthy FOD.** `REBUILD_FOD=1 + FORCE_RESPONSE` re-estimates the response from
   scratch; on marginal/atrophied data it picks bad voxels → degenerate FOD → tckgen accepts ~0 streamlines.
   Rebuild *only* when the cached FOD is empty/degenerate (`mrstats wmfod -output max` ≈ 0).
2. **`REBUILD_FOD=1` is required to use the fresh-registration lever** — it generates `b0_post` (in the FOD
   grid) that `fresh_t12b0()` needs; with `REBUILD_FOD=0`, fresh reg silently no-ops.
3. **Near-zero density + healthy FOD = registration/atlas problem,** not FOD/tracking → SyN re-reg / atlas-overlap check, not more tckgen.
4. **A fix that helps subset A can break subset B** → gate every fix on a precondition; never blanket-apply.
5. **Count drift was a measurement bug:** counting from `result.json` (recovery roots) over-counted vs what
   was actually written to production. Fix: count **only** from production `SC_AAL166 count.csv` (the manifest).
6. **Genuine dead-ends:** tckgen SIGSEGV (rc=11) even after FOD rebuild = corrupt/insufficient DWI data.

## Failure-mode → fix map
- tckgen huge ETA / ~0 streamlines → check `mrstats wmfod -output max`; if ~0 → REBUILD_FOD=1 else use cached.
- tracks generate but connectome near-empty (many zero rows) → registration/atlas (SyN re-reg, radius).
- near-miss just under 0.6 on atrophied brain → larger `assignment_radial_search` (monotonic ↑ density).
- few DWI directions / single low shell → data-limited → document exclusion (mid/low band).

## Scaling (this box: 96 physical / 192 vCPU, but **disk-bound at ~128 MB/s**)
tckgen scales near-linearly to 96 threads (4.63× of 16). For real-time per-subject: 1×96 (~6 min/10M).
For throughput on a queue: more workers × fewer threads. **But** re-loading 10M tracks (~5–8 GB) is
disk-bound (~10 min/load) → avoid mass re-assignment; prefer copy-promote of existing run-root connectomes.
