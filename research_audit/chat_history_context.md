# Structural-connectome chat-history reconstruction

Last audited: 2026-07-18 UTC

## Scope and provenance

This document reconstructs only the structural-connectome project decisions, commands, results, bugs, and unfinished work from the VS Code Codex/Claude histories requested by the user. Unrelated chat content, credentials, authenticated URLs, and secrets were excluded.

The histories are not independent experiments. Several sidebar titles point to the same underlying session, and many numerical snapshots were superseded by later rebuilds. Session UUIDs and output timestamps are therefore the authoritative identity keys.

### Requested thread map

| User-visible name | Session ID | Durable history | Identification confidence |
|---|---|---|---|
| probable Codex `iit_1` | `019de837-4d86-7db2-8656-895110742d96` | `/home/ec2-user/.codex/sessions/2026/05/02/rollout-2026-05-02T10-23-51-019de837-4d86-7db2-8656-895110742d96.jsonl` | Probable, not title-confirmed. This 2026-05-02 thread begins with the exact Step-7 monitor that the next thread asks to review. No surviving Codex index row carries the literal title `iit_1`. |
| Codex `iit 2`, later renamed `iit_3` | `019e5f55-746f-7322-b76e-7ee4e3d03d4d` | `/home/ec2-user/.codex/sessions/2026/05/25/rollout-2026-05-25T13-31-36-019e5f55-746f-7322-b76e-7ee4e3d03d4d.jsonl` | Confirmed. The first prompt says `iit - 2`; `/home/ec2-user/.codex/session_index.jsonl:26,29` shows the same UUID first as “Review BSH-5.2 chat context” and later as `iit_3`. These are aliases, not two separate runs. |
| Codex `iit_4` | `019e903f-825f-76b3-98e6-dec8ca114c38` | `/home/ec2-user/.codex/sessions/2026/06/04/rollout-2026-06-04T01-29-01-019e903f-825f-76b3-98e6-dec8ca114c38.jsonl` | Confirmed by `/home/ec2-user/.codex/session_index.jsonl:31-32`; first named “Add findings story tab,” then renamed `iit_4`. |
| Claude `iit_1` | `e26ef3c3-fd45-4c91-8237-f60f309e1a06` | `/home/ec2-user/.claude/projects/-home-ec2-user/e26ef3c3-fd45-4c91-8237-f60f309e1a06.jsonl` | Confirmed by the exact custom-title record at line 14434. Its first user message at line 3 imports Codex thread `019e5f55...`. |

Supporting Claude project memory is `/home/ec2-user/.claude/projects/-home-ec2-user/memory/structural-connectome-exp-project.md`. Lines 10-16 describe the final dense-cohort rebuild; lines 18-23 map the Codex iterations and network layer. This memory is a useful index, but claims below were also checked against current source/output files where possible.

## Reconstructed chronology

### 1. Initial Step-7 generation and provisional analysis — probable Codex `iit_1` (2026-05-02 to 2026-05-25)

- The old 285-subject subset run was stopped and Step 7 was relaunched over the 701-subject FOD-ready universe. At that point 525 subjects already had tracks and 176 were missing. The run used ACT/GMWMI where viable, FOD repair, a dynamic-ACT salvage lane, and preserved completed outputs (`019de837...jsonl:855-858`).
- The project assembled the MRtrix pipeline and the analysis layer:
  - DWI conversion and gradient validation.
  - `dwidenoise`, `mrdegibbs`, then `dwifslpreproc`/Eddy.
  - bias correction, B0/T1 registration, 5TT/GMWMI construction.
  - tissue response/FOD estimation, tractography, SIFT2.
  - AAL3 parcellation in diffusion space and `tck2connectome` matrices.
  - FA/MD/RD/AxD, graph, edgewise, LR/SR, delay/EDR, coupling, brain-age and ML analyses.
- `structural_connectome_C.ipynb` and reusable `analysis_*.py` modules were built and executed on a provisional cohort of about 550 subjects. The notebook was changed from CSV-only execution to visible figures/statistics and reportedly rendered 100 figures without execution errors (`019de837...jsonl:4191,4463`). These were explicitly provisional because connectome QC was not locked.
- A password-protected Streamlit dashboard was deployed behind nginx/HTTPS. Access details are intentionally omitted here (`019de837...jsonl:4883`).
- Structural ML was hardened against obvious leakage: the cohort table defined the subject universe and labels only; age, sex, ADNI phase, APOE, weight, population and clinical scores were removed from predictors (`019de837...jsonl:16630`).
- By the end of this thread, early production snapshots ranged from 642 connectomes down to a 639-subject working set. The dashboard exposed global FA/MD/RD/AxD, matrix QC, graph/node measures, LR/SR, EDR, delay, brain age and disease models (`019de837...jsonl:17626,18570`).
- SC-Forge v1 was launched as a scratch, no-regression recovery workflow. Candidate matrices were retained only when they improved and did not worsen key QC; severe collapse was quarantined, and production was not initially overwritten (`019de837...jsonl:58617,59212`).

### 2. Marathon AAL3 rescue, ML rework and route ladder — Codex `iit 2` / `iit_3` (2026-05-25 to 2026-06-12)

- The thread moved from the unrelated PE workspace back to `/home/ec2-user/exp` and audited 639 file-complete matrices. Only one matrix met the then-strict whole-matrix PASS definition; registration, label survival and endpoint assignment were the dominant failures.
- A first SC-Forge promotion pushed 61 improved subjects with backups. A later production push promoted 327 retained subjects / 2,289 matrices with hash verification and retained backups; the dashboard was refreshed and served successfully (`019e5f55...jsonl:2535`).
- The key technical diagnosis was that density was an edge-support problem, not a choice of matrix weight:
  - For a selected AD subject, `count`, `fd_sum`, FA, MD and other matrices shared the same 695 nonzero ROI pairs (`019e5f55...jsonl:15888-15980`, agent events at 05:34 UTC).
  - Reusing the same 3M streamlines with a compact AAL116 scratch parcellation improved density from 0.0484 to 0.1681 and assigned 2,980,155/3,000,000 streamlines. Production AAL3 had retained only 114/166 labels, whereas scratch AAL116 retained 103/116 (`019e5f55...jsonl:15980-16280`, 05:39-05:45 UTC).
  - Conclusion: the primary defect was AAL label survival / spatial registration / streamline-to-ROI assignment. Adding streamlines without fixing that contract would not restore missing nodes.
- The project retained AAL3 rather than silently mixing AAL116 outputs. The chosen repair direction was: stable compact AAL3 indices 1..166, nearest-neighbour label transforms, route-specific registration recovery, radial/forward endpoint-assignment variants, and only then tractography escalation.
- Density was consistently defined as nonzero upper-triangle ROI pairs divided by 13,695 possible pairs for 166 nodes. It is a property of `count` edge support, not the magnitude of SIFT2 `fd_sum`, FA or MD weights.
- The user-required promotion rule was preserved: never replace a production matrix merely because a candidate exists; back up the original and promote only a demonstrably better, non-collapsed candidate.
- The route ladder expanded through R21. A late example, `036_S_6231`, improved count density 0.1769→0.6117 and zero rows 79→4, but remained pending visual review. Immediately before the later Claude rebuild, the working production snapshot was CN 303 / MCI 236 / AD 100, with AD median density still about 0.017. This entire snapshot is superseded.
- Clinical metadata reconciliation increased available historical MMSE/CDR rows, but visit alignment remained a concern. The user correctly required Global CDR to be multiclass classification, not regression; FAQ/GDS were removed from the main modeling goal; age was excluded from predictors; nested CV, preprocessing audits and group-wise target outlier handling were added.
- A then-best no-age MMSE result reached R²≈0.426, while CDR accuracy was ≈0.585. A tuned CDR result fell under nested held-out calibration and was not promoted (`019e5f55...jsonl:14316`). These numbers belong to an older cohort/output generation and are not the current final results.

### 3. Dashboard “Novel Findings” narrative — Codex `iit_4` (2026-06-04)

- The thread consolidated the dashboard’s positive results into a paper-like `Novel Findings` tab with FDR-filtered node/edge tables, model summaries, AAL3 brain-space plots, age-corrected brain-age gap, LR/SR panels, EDR and disease-gradient views.
- It produced synchronized CN/MCI/AD LR/SR panels for exception rate, exception strength and all-edge structural weight. The chat recorded 2,803 exception-rate group-edge rows, 3,803 exception-strength rows and 15,548 all-edge structural rows.
- An early story emphasized a thalamic–limbic/cingulate axis, with thalamic nodes such as `Thal_MDm_L`, `Thal_IL_L`, `Thal_MDl_L`, `Thal_VL_L`, `Thal_VA_L`, and `Thal_PuM_L`, plus cingulate/orbitofrontal/ACC/NAcc/insula/hippocampus/amygdala/neuromodulatory regions. These results predate the dense-530 rebuild and must not be carried into a manuscript without re-running the exact tests on current outputs.
- The thread made two scientifically important interpretation decisions:
  - Raw short-range weights being larger than long-range weights is expected from anatomy/tractography and is not itself a disease result. Clinical inference should use within-range group contrasts and distance-aware EDR residual/exception measures.
  - “Node signal” in the interactive view is the sum of currently displayed incident-edge signal, not an independently tested nodal biomarker.
- Terminology was disambiguated: axial diffusivity should be written AxD or λ1, distinct from Alzheimer disease (AD) and from the EDR decay parameter λ.
- The literature support in this thread was a small anchor set, not the requested exhaustive high-impact review. The story tab is an exploratory narrative prototype, not novelty proof.

### 4. Root-cause audit and final dense rebuild — Claude `iit_1` (2026-06-12 to 2026-06-17)

- Claude’s initial RCA found 648 matrices with median 62/166 valid zero rows; only 149/648 had ≤15 zero rows and 189/648 were effectively empty (<0.05 density). The old pipeline downsampled AAL3 from 1 mm to B0 2 mm, losing small parcels; 24/166 labels were <50 voxels even in healthy cases. It also mixed 169/170-node matrices and counted the four AAL3 label gaps (35, 36, 81, 82) incorrectly in some paths (`e26ef3c3...jsonl:203`).
- Subjects were divided into four recovery lanes: 151 already good, 276 atlas-fixable, 62 scaffold problems and 159 degenerate-track cases. Nonlinear registration was retained only when it improved QC because ANTs SyN could also catastrophically worsen a subject (`e26ef3c3...jsonl:406`).
- Representative canaries established the route logic:
  - a recoverable case improved from 41→8 zero rows, density 0.33→0.56, endpoints 110→158;
  - some cases reached only 2–4 endpoints under every atlas route, implying degenerate tractography;
  - one nonlinear transform worsened a case from 2→103 zero rows, proving the need for keep-best/no-regress gating.
- The final recovery combined compact AAL3 166 indexing, high-resolution/world-space atlas handling, BBR/nonlinear routes where viable, 10M tractography for difficult cases, and a strict preserve-better policy.
- Final chat state (`e26ef3c3...jsonl:13435` and project memory lines 14-16):
  - 530 dense structural connectomes: CN 251, MCI 201, AD 78.
  - Every canonical `count` matrix reported density ≥0.6.
  - Structural/graph analyses use n=530.
  - Diffusivity/network analyses use n=529 because `168_S_6938` was registration-unstable; its good structural matrix (density 0.895) was preserved, but diffusivity was excluded after retries at 0.260/0.328.
  - Connectomes were renamed from `SC_AAL_...` to explicit `SC_AAL166_...` files.
- Two analysis-layer bugs were fixed during the rebuild:
  - modules were globbing obsolete `SC_AAL_` names and silently finding no current matrices;
  - `final_connectome_ready` checked absent `_ALL.csv` and cleaned intermediates, marking everyone unready. Current source resolves subjects from `SC_AAL166_*_count.csv` and defines readiness from canonical count existence (`connectome_analysis/analysis_cohort.py:451-527`).
- The full dashboard analysis layer was regenerated. Current on-disk evidence confirms `master_cohort.csv` has 530 data rows and `group_counts.csv` reports CN 251 / MCI 201 / AD 78.

## MRtrix-to-“BATMAN” pipeline interpretation

The histories and code do not show BATMAN as a separate executable downstream package. “BATMAN” is the reference/tutorial style and legacy derivative naming used to shape the MRtrix workflow. Current code explicitly calls its single-phase-encoding Eddy setup “BATMAN-style” (`connectome_pipeline/dwi_eddy.py:1-9`). The actual connectome generator remains MRtrix `tck2connectome`.

The operational sequence reconstructed from chats and source is:

1. ADNI DICOM/DWI selection and gradient/volume/shell validation.
2. `mrconvert` → `dwidenoise` → `mrdegibbs`.
3. `dwifslpreproc -rpe_none` with phase-encoding direction and Eddy options; QC and rescue lanes.
4. Bias correction and B0 extraction.
5. T1 selection, T1↔B0 registration (BBR/FLIRT plus gated nonlinear rescue), 5TT/GMWMI.
6. Response estimation, FOD construction/normalization.
7. ACT/GMWMI tractography when valid; recovery variants for anisotropic/problem cases; SIFT2.
8. MNI AAL3 → T1/B0 label transform with nearest-neighbour interpolation; compact stable matrix indices 1..166.
9. `tck2connectome` with endpoint-assignment variants and nine matrix families: count, count-inverse-node-volume, SIFT2 `fd_sum`, length/inverse length, FA, MD, AxD and RD.
10. No-regress QC publication, then cohort, microstructure, graph, coupling, LR/SR, delay/EDR, brain-age, edgewise, ML, network and findings layers.

The key BATMAN/tck2connectome lesson from the histories is that streamline count defines edge existence/density. SIFT2, diffusivity and length matrices reweight supported edges; they cannot repair an atlas label that vanished before endpoint assignment.

## Current claims that can be carried forward, with caveats

These are the latest chat claims that also have current output support. They are candidate results, not yet manuscript-valid novelty claims.

| Candidate result | Current evidence | Required wording |
|---|---|---|
| Dense cohort | `/data/derivatives/qc/analysis_cohort/00_master/master_cohort.csv` has 530 rows; `/data/derivatives/qc/analysis_cohort/02_demographics/group_counts.csv` gives CN251/MCI201/AD78. | Report structural n=530 and diffusivity n=529 separately. Include the full attrition path and group-specific exclusions. |
| Limbic functional network has the largest cross-network microstructure effect | `/data/derivatives/qc/analysis_cohort/19_network_analysis/functional/network_affectedness_summary.csv`: AxD δ=-0.554, q=2.88e-10; MD δ=-0.539, q=1.37e-9; RD δ=-0.518, q=9.49e-9; FA δ=+0.359, q=1.23e-4. | “AxD/MD/RD are higher and FA is lower in AD”; do not say FA “rises.” Functional-network assignment is an approximate AAL3→Yeo mapping and is mapping-dependent. |
| Visual and Limbic graph integration is lower in AD | Current `network_inference.md`: Visual nodal efficiency/strength δ≈+0.27, q≈0.025; Limbic nodal efficiency δ≈+0.24, q≈0.033. | Present as structural graph findings, not functional disconnection. Check robustness to matrix normalization, thresholding and parcel size. |
| One frontal intra-structural coupling result | Current `novelty_report.md`: frontal nodal-efficiency↔FA coupling δ≈+0.40, q≈8.1e-5. | Call graph–microstructure coupling. It is not SC–FC coupling and must not be equated with functional decoupling literature. |
| Age-related diffusivity is broad, not diagnosis-specific | The current findings layer reports no significant age×group interaction. | Cross-sectional association only; age and diagnosis are partially confounded. |
| Seven exploratory mediation chains, four bootstrap CIs excluding zero | `/data/derivatives/qc/analysis_cohort/21_mediation/mediation_results.csv`; Claude final at `e26ef3c3...jsonl:14430-14432`. | Hypothesis-generating cross-sectional indirect associations only. Do not use “mechanism,” causal mediation or compensation as established biology. |

## Results that are superseded or unsafe to quote

- All n≈550 provisional notebook results from early `iit_1`.
- The 639-subject file-complete snapshot, its “only one strict PASS” audit, and group median densities from routes 1–21.
- The June-12 CN303/MCI236/AD100 snapshot and AD median density ≈0.017.
- The 633-subject / density≈0.32 dashboard snapshot.
- The older no-age MMSE R²≈0.426 and CDR accuracy≈0.585.
- `iit_4`’s thalamic–limbic/cingulate story, unless the exact same region tests survive the current 530/529 rebuild and the current multiple-testing family.
- Any claim that raw short-range > long-range weight is a disease signature.
- Any claim that an interactive node-signal overlay is an independently significant nodal result.
- Any claim that the OpenAlex pool or four novelty cards prove novelty.

## Discrepancies between chat conclusions and the current workspace

1. **The canonical connectome context document is stale or missing.**
   - `/home/ec2-user/exp/README.md` calls `structural_connectome_context.md/.docx` “current.”
   - Both files are dated 2026-05-25 and describe the pre-rebuild 639-subject failure state.
   - Claude memory also calls them stale.
   - `/home/ec2-user/sabeesh/context.docx` and `/home/ec2-user/sabeesh/context/context.docx` are PE-quantification documents, not structural-connectome context. The Claude memory claim that the latter is the moved project context is incorrect.
   - Action: create/update a dedicated current connectome context; do not overwrite the PE context with connectome material.

2. **The final cohort is called final in chat, while refresh metadata still says provisional.**
   - `/data/derivatives/qc/analysis_cohort/00_master/dashboard_refresh_status.csv` records `snapshot_mode=provisional` at 2026-06-17 11:43 UTC.
   - Network/findings/mediation outputs were written later, through 15:06 UTC.
   - Action: rerun/lock a manuscript snapshot with a versioned cohort manifest, code commit/hash, section status and immutable output directory.

3. **Current ML is weaker than the older chat’s headline numbers.**
   - Current diagnosis ML (`18_ml_diagnostics/ml_model_performance.csv`) peaks at accuracy 0.549, balanced accuracy 0.505 and macro-AUROC 0.676 on n=530.
   - Current primary scan-aligned MMSE has only n=62 after outlier handling and best R²≈0.033; several models have negative R².
   - Current primary CDR has n=65 and best accuracy≈0.40. All-history sensitivity models use n≈195-203 but can mix remote clinical status with the scan.
   - Action: do not make cognition prediction or disease classification the manuscript’s primary novel contribution. Reconcile visit-level labels and add external/site-held-out validation before stronger claims.

4. **The novelty report contains an inaccurate headline.**
   - `20_findings/novelty_report.md` says “diffusivity rises across FA/MD/AD/RD.” FA is not a diffusivity magnitude and the observed FA direction is lower in AD, not higher.
   - It also writes Cliff’s delta as `d`, which can be mistaken for Cohen’s d; use δ.
   - Action: change the heading to “Limbic microstructural alteration across FA, MD, AxD and RD” and state metric-specific directions.

5. **Mediation labels overstate what the implementation establishes.**
   - `analysis_mediation.py` correctly warns that analysis is cross-sectional and non-causal, but output statements still say “significant indirect (mediation)” and the dashboard heading says “Mechanistic & compensation chains.”
   - Seven hand-selected chains were bootstrapped without a visible correction across chains; one result reports 115% mediated, a warning sign for inconsistent/suppression paths rather than a simple mediation narrative.
   - Action: relabel as exploratory indirect-association analysis, correct across the seven chains or preregister them, and avoid compensation/causal wording.

6. **Literature retrieval is not a systematic novelty audit.**
   - Current `novelty_report.md` openly says coverage is non-exhaustive and OpenAlex keyword retrieval can be off-topic.
   - The 1,923-record pool is a retrieval corpus, not 1,923 verified relevant studies; only four web-verified narrative cards were built.
   - Action: conduct a reproducible database search with query strings, dates, screening log, inclusion/exclusion criteria, exact-claim prior-art table and primary-source verification.

7. **The dense ≥0.6 cohort rule may induce selection bias.**
   - Recovery success and registration/tractography quality differed strongly by diagnosis before rebuilding. Restricting analysis to matrices that clear density 0.6 can condition on a technical variable associated with group, atrophy, scanner/site or image quality.
   - Action: report all attempted subjects and group-wise attrition; compare included vs excluded demographics/site/QC; run threshold sensitivity and, if possible, model inclusion probability.

8. **Visual QC and external validation are not yet manuscript-complete.**
   - Chat verification was often programmatic/headless. A few cases remained review-pending during route development, and the final story has no documented independent external cohort.
   - Action: complete blinded visual QC sampling, inter-rater rules, site/scanner sensitivity, one-subject structural-only sensitivity, and independent replication or a clearly labeled internal-only scope.

## Outstanding work before a defensible paper

1. Freeze a current, versioned n=530/n=529 data manifest and regenerate every paper table/figure from that manifest only.
2. Re-audit participant uniqueness, scan selection, site/scanner balance, diagnosis coding and clinical visit alignment. Use participant- and preferably site-grouped validation.
3. Recompute all inferential families with explicit omnibus/post-hoc logic, multiplicity families, effect sizes and confidence intervals. Separate exploratory from confirmatory tests.
4. Quantify the effect of the density≥0.6 inclusion gate and all recovery lanes on group composition and outcomes.
5. Validate AAL3 compact indexing and the approximate AAL3→Yeo network map; run anatomical-network and mapping sensitivity analyses.
6. Reframe mediation as exploratory indirect association or redesign it prospectively/longitudinally. Do not infer compensation from cross-sectional covariance.
7. Keep ML secondary unless current scan-aligned clinical labels and external validation materially improve. Never reuse the old R²/accuracy values.
8. Run the formal literature review before declaring a gap. Build an exact-match table for each candidate claim: atlas, node/network definition, modality, weights, cohort stages, covariates, statistics and reported direction.
9. Re-test the old thalamic/limbic, LR/SR, delay/EDR and brain-age stories on the locked cohort. Drop any tangent that does not survive statistical and biological coherence checks.
10. Build the manuscript around one defensible chain: technical recovery/QC → dense AAL3 network representation → limbic microstructure → graph/coupling context → cautious clinical relevance. ML, mediation, LR/SR and brain age should support that chain only if they pass the locked audit.

## Bottom line from the histories

The strongest current paper direction is not “a classifier distinguishes CN/MCI/AD.” It is a carefully qualified, network-resolved structural-connectome study in which a technically rescued dense AAL3 cohort shows the largest microstructural group effect in the mapped Limbic network, alongside smaller Visual/Limbic graph changes and a limited frontal graph–microstructure coupling result. The result is scientifically interesting but not yet proven novel: current literature work is non-exhaustive, clinical labels are sparse, the dense-cohort gate needs bias analysis, and cross-sectional mediation cannot establish mechanism or compensation.
