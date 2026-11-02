# Uniform connectome rebuild (Phase 1 option 1) — status 24 Oct 2026, ~21:45 IST

Goal: structure **and** microstructure for all 530 participants from one recipe and one tractogram per person, so the
microstructure-coverage artefact found in Phase 1 (§3.8) disappears. Nothing published is overwritten.

## What the scoping found

1. **Root cause of the coverage artefact.** For the 366 participants of the "current" route the published structural
   matrices came from new tractograms that were not kept, while their FA/MD/AxD/RD matrices came from the *legacy*
   tractograms (`/data/derivatives/tracks/<unit>/tracks_final_3000k.tck` + `*_mean.tsf`). Those legacy tractograms
   cover only part of the brain for about half the cohort (`track_qc.csv`: brain coverage < 0.5 for 261 of 529,
   254 of them current-route; median 0.34 current route vs 0.58–0.60 legacy routes). Legacy track coverage
   correlates ρ = 0.86 with the published microstructure coverage.
2. **v1 (reuse legacy tracks, new atlas) fails**: broken tracks stay broken, and the stored T1-in-DWI images are
   misregistered for some participants (e.g. participant A T1 brain shifted ~24 mm), so a T1-based atlas lands wrong.
3. **v2 (new tractography from the stored FOD, FA-based atlas)** works where the stored FOD is good
   (participant A density 0.80, participant D 0.78, 0 or 3 empty regions, 100% coverage; ~20 min at 12 threads) but fails
   where it is not (participant E density 0.005; participant C 0.02 and its stored FA cannot even be registered).
4. **v3 (`--from-dwi`)** rebuilds bias correction, mask, dhollander response, single-shell CSD (lmax 8) and the tensor
   maps from the eddy-corrected DWI (`/data/derivatives/eddy/<unit>_preproc.mif`, present for all units), then v2.
   Canary running on participant E, participant C (broken) and participant B (good) — see `connectomes_uniform_v3/matrices`.

## Cost (this 64-vCPU machine)

v3 per participant ≈ 25 min at 10–12 threads (≈ 5 CPU-hours) → 530 participants ≈ 2,700 CPU-hours ≈ **42 h** here
(leaving headroom for other services), or ≈ 14 h on a 192-vCPU instance. Disk: intermediates are deleted per
participant; matrices + registration ≈ 40 MB per participant.

## How to run (once the v3 canary is accepted)

```bash
cd /home/ec2-user/exp/docs/iit_dipanjan/tools/rebuild
cut -d, -f1 ../../phase1/qc.csv | tail -n +2 > /data/derivatives/connectomes_uniform_v3/all_sids.txt
( setsid nohup bash -c 'cat /data/derivatives/connectomes_uniform_v3/all_sids.txt | xargs -P 4 -I{} \
    /home/ec2-user/exp/.venv_connectome_app/bin/python -I rb2_subject.py {} --threads 10 --from-dwi \
    --out /data/derivatives/connectomes_uniform_v3 >> /data/derivatives/connectomes_uniform_v3/run.log 2>&1' & )
```
Restartable: finished participants are skipped (their `*_qc.json` exists).

## After the matrices exist

1. Run the analysis pipeline on them: `python -m connectome_analysis.run_analysis --connectomes-dir
   /data/derivatives/connectomes_uniform_v3/matrices --analysis-root <new root>` (exceptions, networks, core).
2. Point Phase 1 (`p1_features.py`, `p1_audit.py`) and Phase 2 (`p2_edges.py uniform_v3`) at the new root/matrices,
   re-run, and refresh both reports with `p1_docx_update.py` / `p2_docx.py`.

## Update 24 Oct 22:00 IST — v3 accepted and launched

v3 canary: participant E density 0.86, participant C 0.83 (both failed before), participant B 0.84; 0 empty regions, 100%
microstructure coverage, FA-template correlation 0.68–0.78, ~23 min each at 10 threads. Full run launched with
`rb3_run_all.sh` (5 workers x 10 threads, nice 19); log `/data/derivatives/connectomes_uniform_v3/run.log`; progress =
number of `*_qc.json` in `matrices/`. Expected finish ≈ Mon 26 Oct afternoon IST. Restart the same script if the
machine reboots (finished participants are skipped).

## Update 24 Oct 23:40 IST — bigger machine, DKT, automatic refresh

- Instance resized by the user to **c7a.48xlarge** (192 cores); `connectome-rebuild` user service resumed the run
  automatically with 18 workers (`rb3_run_auto.sh`, per-participant locks). ~21 min per participant → ~50/h →
  expected finish ≈ Sun 25 Oct 09:30 IST. Resize back to c6a.16xlarge afterwards.
- Second parcellation: every worker now also builds **DKT76** matrices (FastSurfer aparc+aseg, 62 DKT cortical + 14
  subcortical, T1→b0 rigid ANTs) from the same streamlines → `matrices_dkt/` (`rb_dk.py`). The first ~13 participants
  lack DKT; `after_rebuild.sh` sets their AAL3 outputs aside and rebuilds them in full before the refresh.
- All phase scripts take their data locations from `IIT_CONN`, `IIT_AROOT`, `IIT_P1/2/3`, `IIT_SRC`, `IIT_SUFFIX`.
- `refresh_all.sh [dry|full]`: analysis tree (8 stages) → Phase 1 (features, audit, 13 conditions) → Phase 2 → Phase 3 →
  notebooks with suffix `_uniform`. Outputs in `phase1_uniform/ phase2_uniform/ phase3_uniform/`,
  `/data/derivatives/qc/analysis_cohort_uniform`. Word documents are refreshed by hand after reviewing the numbers.
- `after_rebuild.sh` (launched detached) waits for 530, backfills DKT, then runs `refresh_all.sh full`.

## Update 25 Oct 12:30 IST — rebuild finished; quality-controlled analysis tree; documents

- Rebuild complete: 530 AAL3 connectomes (+529 DKT76). The instance is back on c6a.16xlarge; the `connectome-rebuild`
  user service is disabled (it only resumed the rebuild).
- Rebuild quality control (`exclusions_uniform.yaml`, kept out of git): a connectome is excluded when atlas registration
  failed (`reg_fa_corr` < 0.4 in `<unit>_qc.json`) or its density is below 0.6. 10 of 530 fail (8 registration failures,
  max r 0.35 against min 0.48 for every retained participant; 2 with low density). `p1_audit.py` reports them as R0/R4.
- The pipeline takes its cohort from every connectome in the matrix folder, so `build_tree.sh` gives it `matrices_qc/`
  (symlinks to `matrices/` minus the excluded participants) and runs all 35 stages, including the ML stages the thesis
  reports. The repository's `build_range_restricted_networks.py` stops on a measure that is constant in every group (no
  short-range exceptions in the brainstem on the rebuilt connectomes); the guarded copy `patched/` runs in its place.
  The repository is untouched.
- Analysis tree: `/data/derivatives/qc/analysis_cohort_uniform` (520). Earlier trees kept beside it:
  `_n522` (first refresh, pipeline cohort 522) and `_n522_partial`.
- `refresh_after_tree.sh` starts `refresh_all.sh full` (SKIP_ANALYSIS=1) once the tree's network tables exist.
- `docs_rebuilt.sh`: brings Phase1_Confound_Control.docx (in place, user structure kept), Phase2_Length_Test.docx and
  Phase3_Exception_Nulls.docx (regenerated) to the rebuilt connectomes; dated backups in `tools/backups/`. The verdict
  reference stays the PUBLISHED estimate (IIT_P1_COMPARE); the published-connectome versions of the scripts are
  `p1_narrative.py`, `p1_docx_update.py`, `p2_docx_published.py`, `p3_docx_published.py`.
- Phase 1 quality control on the 520 then excludes one more participant (participant F, CN, rule R1: median MD 1.3795 against a
  fence of 1.3794; site 114 has high MD throughout), so Phases 1–3 use 519 while the analysis tree (and the thesis ML) has
  520. Keeping or dropping this participant changes no Phase 1 effect by more than 0.012 in δ (C0 vs C0q). Decision open.

## Update 25 Oct 15:00 IST — Phases 4-6
- Phase 4 `tools/phase4/` (p4_data, p4_engine, p4_tables_figs, p4_docx) → `phase4_uniform/`, Phase4_Incremental_Prediction.docx.
- Phase 5 `tools/phase5/` (p5_dkt_map, p5_features, run_phase5.sh, p5_tables_figs, p5_act, p5_docx) → `phase5_uniform/`,
  Phase5_Robustness.docx. DKT network labels: /data/derivatives/connectomes_uniform_v3/dkt76_network_mapping.csv.
  ACT subset: rb_act_subject.py / rb_act_run.sh → /data/derivatives/connectomes_uniform_v3_act (45 participants);
  tools/phase5/after_act.sh runs the comparison and refreshes the Phase 5 report when the re-tracking ends.
- Phase 6 `tools/phase6/` (p6_figures.py with the pipeline venv, p6_docx.py) → `phase6_uniform/fig_ms*.png`, Phase6_Figures.docx.

## Update 2 Nov 2026 — one root, one set of settings, guards (recipe v4 preparation)
Nothing below changes a default: with `IIT_ROOT` unset every script resolves the paths it always did, and the master cohort, the
Phase 1 features and the audit come out byte-identical on the existing data.
- `IIT_ROOT`: the single root variable (`tools/iit_chain.py`). A chain reads its connectomes from and writes everything under this
  root: `connectomes/`, `analysis_tree/`, `phase1/`..`phase7/`, `docs/`, `notebooks/`, `logs/`. One root per recipe arm.
- Settings (`sc_cohort_settings.py` at the repository root), fixed once per chain in `chain_settings.json`; a later step with another
  value stops: `SC_DENSITY_FLOOR` (0.6; `none` = no floor), `SC_SMC` (`mci`; `hold_out`), `IIT_PRIMARY_SAMPLE` (`adni34`; `adni3`,
  `pooled`), `IIT_LENGTH_MEASURE` (`per_person`; `mni_distance`, `control_median_act`), `SC_N_NODES` (166; 154 leaves out the 12 brainstem
  nuclei), `SC_LABELS` + `SC_LABEL_SEED`, `IIT_R4_FIELD` + `IIT_R4_MIN`. A new chain must state the first five; defaults belong to today's outputs.
- Launchers refuse (exit 3) unless every connectome's `recipe_sha256` (in its `_qc.json`) equals the one in
  `/data/derivatives/recipe_v4/SIGNED_RECIPE.json`; a missing file means refuse. This includes today's default paths: re-running them
  needs a root of its own. `IIT_REHEARSAL=1` / `--rehearsal` bypasses the check, only with its own root; the bypass is written into
  `chain_settings.json`, `REHEARSAL_UNSIGNED.txt` and the first line of every document.
- Labels: under a root, real diagnoses are joined only with `SC_LABELS=real`. `SC_LABELS=shuffled SC_LABEL_SEED=n` permutes diagnosis and
  clinical scores together between participants (recorded in `00_master/label_mode.json`, map in `label_donor_map.csv`).
- Exclusions under a root come from the rule (`iit_chain.py exclusions`): registration below its minimum, or density below the floor.
- Length measure and nodes: a view of the matrices (`connectomes/matrices_view`) with `len_mean` replaced by the fixed per-edge ruler on the
  edges each person has, and nodes beyond `SC_N_NODES` at zero. The no-ACT arm takes the ACT arm's ruler with `IIT_LENGTH_REF`. DKT keeps
  per-person length: MNI centres exist only for AAL3.
- What a recipe run hands over: `SC_AAL166_<unit>_<kind>.csv` and `SC_AAL166_<unit>_qc.json` per session (`iit_chain.py export-matrices`), the
  qc.json carrying `recipe_sha256`; DKT matrices the same, with their own qc.json. `IIT_R4_FIELD`/`IIT_R4_MIN` name the recipe's registration
  measure (`wm_dice` in recipe v4; its exclusion threshold is not decided yet). Under a root, covariates come from each session's `stats/aseg+DKT.stats` in `IIT_FASTSURFER`/`<unit>`: the CPU
  route runs `--seg_only`, so head size is FastSurfer's mask volume, not eTIV.
- Documents: builders take a new version name (`--version=NAME`, `IIT_DOC_VERSION`, or the launcher's argument) and never overwrite;
  in-place editors accept only a working copy (`iit_chain.py work-copy`, then `publish`). `p1_docx.py` refuses to run.
- Narrative guard (`tools/narrative_guard.py`): a document is saved only if its fixed sentences hold on this chain's numbers.
- `tools/run_chain.sh` runs everything under a root with a ledger (`logs/chain_status.tsv`). `tools/test_chain_guards.py` tests the guards.
