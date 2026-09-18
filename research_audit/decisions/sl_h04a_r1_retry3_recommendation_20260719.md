# SL-H04A-R1-RETRY3 corrective continuation recommendation

**Status:** PREPARED; not launched  
**Detected:** 2026-07-19 00:04:38 UTC  
**Affected run:** `/data/derivatives/scforge_v2/h04a_r1_recovery_20260718_retry2`  

## Evidence and diagnosis

Retry2 reached valid corrected-DWI, bias-corrected-DWI, and brain-mask outputs before the first `select_fod_shells` jobs were eligible. Snakemake then raised, before invoking MRtrix:

`TypeError: argument should be a str or an os.PathLike object where __fspath__ returns a str, not 'Namedlist'`

The failure is at `scforge/workflow/rules/05_5tt_fod.smk:143`. `rules.gradient_contract.output` is a one-element Snakemake `Namedlist`; the Python `run:` block passes `input.contract` directly to `Path`. The same latent error exists in `select_tensor_shells` at line 195. The exact fix selects the sole path at rule binding time with `rules.gradient_contract.output[0]` in both rules. It changes no gradient values, shell values, algorithms, commands, unit selection, diagnosis blindness, resource limits, or authorized terminal stage.

## Required continuation design

1. Do not interrupt retry2's remaining CPU-Eddy workers.
2. Let the launcher write immutable retry2 attempt-end and completion evidence.
3. Validate and hash every reusable retry2 artifact. Never treat a filename alone as reusable evidence.
4. Apply the two-line source patch and regenerate the workflow-source manifest.
5. Freeze a retry3 execution decision and extension binding that explicitly declare:
   - authorization, scientific parameters, unit set, and resource caps unchanged;
   - the retry2 TypeError and immutable terminal evidence;
   - the exact old/new source and manifest hashes;
   - a new isolated retry3 run root;
   - hash-validated copy-on-write seeding from retry2;
   - no mtime-based upstream recomputation of seeded artifacts;
   - no FOD reconstruction, tensor maps, tractography, matrices, statistics, dashboard publication, or 530-subject work.
6. Seed the new root only after retry2 is terminal. Use copy-on-write copies, not mutable hard links.
7. Execute only missing Phase-A shell selection and response-calibration work; retain the full 15-unit denominator and explicit terminal outcomes.
8. Freeze pooled responses only if at least 12 valid units span both manufacturer families and both T1-source classes.

## Acceptance conditions

- Retry2 remains immutable and independently auditable.
- No completed Eddy job is recomputed.
- Seed hashes replay exactly before retry3 launch.
- The patched FOD-shell rule passes a real one-unit execution test and all workflow/contract tests.
- Retry3 remains within the cumulative 24-hour and 100-GiB R1 limits.
- The GPU resize gate remains closed until retry3 terminates, completion/manifest hashes validate, and the eligible pooled response is frozen.
