# Connectome Recovery — Recipe Ledger (what creates SUCCESS vs FAILURE)

Principle: **sturdy-first.** Use the proven path before experimenting. Only deviate from it with
evidence, and record the outcome here. Every run logs: recipe (env flags) → outcome (n≥0.6, mean d, failure mode).

## The PROVEN sturdy path (produced the 500 done ≥0.6)
- **Use the EXISTING cached FOD** (`/data/derivatives/fod/<sid>/wmfod_final.mif`) — do NOT rebuild it.
- ANTs **SyN registration** (`--reg-mode fullhead`) — the big atlas-placement lever.
- **10M streamlines** + **noACT** dynamic seeding (`FORCE_NOACT=1`), radial-search assignment.
- Promote only ≥0.6 (`RESUME_MIN_DENSITY=0.6`).

## Runs & outcomes

| Run | Recipe | Cohort/scope | Outcome | Verdict |
|---|---|---|---|---|
| base/SOTA (orig) | existing FOD + SyN + 10M + noACT | all | 500/648 ≥0.6 | ✅ SUCCESS — this is the sturdy path |
| regen216 | **REBUILD_FOD=1**, 3M, SyN | 216 deleted-track GOOD subjects | 59/61 ≥0.6 (mean 0.79) | ✅ works *on already-good data* (3M reproduce) |
| ad_blitz **v1** | **REBUILD_FOD=1 + FORCE_RESPONSE**, 10M | 24 hard AD | 0/5 ≥0.6; FODs collapsed to ~0 → 54-yr ETA → deferred | ❌ FAILURE — rebuild destroyed healthy FODs |
| ad_blitz **v2** | **REBUILD_FOD=0** (use healthy cached FOD), 10M, SyN | 24 hard AD | running (testing the fix) | ⏳ |

## Key lessons
1. **Don't rebuild a FOD that's already healthy.** `REBUILD_FOD=1` + `FORCE_RESPONSE` re-estimates the
   response function from scratch; on marginal/atrophied data it picks bad voxels → **degenerate FOD**
   (max collapses, SH coeffs ~1e-20) → tckgen accepts ~0 streamlines. The original response/FOD was
   estimated carefully once and is good. Rebuild was meant only for the *few* "weakly-scaled" FODs, but
   was applied as a blanket default — it helped good-data subjects (regen216) and **broke** hard ones.
2. **A fix that helps subset A can break subset B.** Gate fixes on a precondition (e.g. only rebuild if
   `mrstats wmfod_final.mif -output max` is below a threshold), never blanket-apply.
3. Near-zero density with a *healthy* FOD (e.g. <SUBJECT>: tracks generated, 157/166 nodes empty) is a
   **registration/atlas** problem, not a FOD or tractography one → needs the atlas-placement check, not more tckgen.

## Failure-mode → fix map
- tckgen ETA huge / ~0 streamlines  → check FOD (`mrstats -output max`); if degenerate, **REBUILD_FOD=0** (use cached).
- tracks generate but connectome near-empty (many zero-rows) → **registration/atlas** issue (SyN re-reg, atlas overlap check).
- DWI few directions / single low shell → genuinely **data-limited** (coarse-atlas salvage or document exclusion).
