# SC-Forge

SC-Forge is a contract-first, QC-gated structural connectome compiler for the
`/home/ec2-user/exp` connectome project.

This package starts with non-destructive validation contracts and dry-run
command planners:

- source DWI / gradient consistency
- AAL3 label mapping and contiguous node contracts
- spatial route scoring scaffolds
- affine transform sanity QC
- matrix integrity QC
- provenance sidecar helpers
- DWI preprocessing command planning
- 5TT/GMWMI command planning
- FOD + DTI metric command planning
- preflight/full tractography command planning
- connectome matrix command planning for `count`, `fd_sum`,
  `count_invnodevol`, `len_mean`, `invlen_mean`, `fa_mean`, `md_mean`,
  `rd_mean`, and `ad_mean`

The first implementation does not overwrite existing derivatives or launch heavy
jobs by default. Heavy registration, tractography, and connectome generation
must be executed only after source, atlas, spatial-contract, and matrix gates
pass on canaries.

## Closed-world v2.1 canary candidate

The active corrective implementation is
`connectome-v2.1.0-closed-world-canary-candidate`. The locked 530-unit manifest
remains the parent denominator, while an immutable, diagnosis-blind,
H04A-approved 12--24-unit full-schema subset is the only executable canary
scope. Full-530 execution is disabled. A technically valid unit produces
exactly nine matrices: `count`, `fd_sum`, `count_invnodevol`,
`len_mean`, `invlen_mean`, `fa_mean`, `md_mean`, `rd_mean`, and `ad_mean`.
Raw DICOM/NIfTI bundle identities remain separate from deterministic staged
input hashes. The current evidence is implementation-only: no real imaging
canary has run.

The v2.0.11 PASS-only schemas and tests remain unchanged as historical
evidence. Version 2.1 adds immutable per-unit `PASS`, `PARTIAL`, or `FAIL`
terminal records. Every non-PASS record requires a failure stage and reason;
each analytical outcome is either `VALID` with a hash-attested artifact or
explicitly `NA`. Invalid matrices or tensor outcomes are never zero-filled,
clipped, or imputed. Matrix validity is outcome-specific: a failed matrix does
not erase independent valid artifacts, while cross-matrix disagreements
invalidate every implicated metric. Density, zero-count nodes, endpoint
assignment fraction, assigned-node count, and endpoint concentration are
quantitative canary QC; they are not automatic exclusions before the blinded
H04 threshold/SAP decision. Biological inference remains locked.

Safe validation commands from `/home/ec2-user/exp`:

```bash
.venv_connectome_workflow/bin/python \
  research_audit/validate_connectome_v2_contract.py \
  --output research_audit/outputs/connectome_v2_contract_validation.json

.venv_connectome_workflow/bin/python \
  scforge/workflow/validate_environment_contract.py \
  --output research_audit/outputs/connectome_v2_environment_validation.json

PYTHONPATH=scforge .venv_connectome_workflow/bin/python \
  -m unittest discover -s scforge/tests -v

.venv_connectome_workflow/bin/python \
  scforge/workflow/smoke_dryrun.py \
  --output research_audit/outputs/workflow_dryrun_v2.json
```

SL-H03-C1 approved design work only; it did not authorize imaging execution.
After the required H04A/H04B decisions exist, the attesting launcher enforces
three separate stages. Every stage uses the same parent manifest, immutable
subset manifest, subset decision, and run root.

The current candidate deliberately keeps `imaging_execution_authorized: false`.
That normative flag is an immutable recipe lock, not the runtime approval
switch. An exact, hash-bound, human-signed H04A decision may authorize only
response-calibration Phase A and the pre-tractography canary, with at most four
cores and cumulative 72-hour/150-GB stops. The launcher rejects recipe mutation,
pending or hash-drifted decisions, any H04A attempt to authorize phase B, and
every authorization/resource failure before creating run state. Phase B still
requires a separate exact H04B continuation decision.

Stage 1 estimates diagnosis-blind subject responses only:

```bash
.venv_connectome_workflow/bin/python scforge/workflow/run_connectome_v2.py \
  --manifest /absolute/path/to/approved_acquisition_pair_manifest_v2.csv \
  --execution-subset-manifest /absolute/path/to/h04a_canary_subset.csv \
  --execution-subset-decision /absolute/path/to/h04a_subset_approval.json \
  --run-root /absolute/path/containing/scforge_v2/canary_run \
  --mode response-calibration-phase-a \
  --cores 4
```

Freeze the terminal phase-A outcomes using the prespecified, human-approved
minimum valid calibration N:

```bash
.venv_connectome_workflow/bin/python \
  scforge/workflow/freeze_response_calibration.py \
  --phase-a-completion /absolute/run/publication/response_calibration_phase_a_completion.json \
  --phase-a-manifest /absolute/run/publication/response_calibration_phase_a_manifest.json \
  --output-dir /absolute/run/frozen_calibration \
  --minimum-valid-subjects APPROVED_MINIMUM_N
```

Stage 2 runs through automated pre-tractography QC and immutable visual-review
bundles, then stops at `AWAITING_HUMAN_QC` without starting tractography:

```bash
.venv_connectome_workflow/bin/python scforge/workflow/run_connectome_v2.py \
  --manifest /absolute/path/to/approved_acquisition_pair_manifest_v2.csv \
  --execution-subset-manifest /absolute/path/to/h04a_canary_subset.csv \
  --execution-subset-decision /absolute/path/to/h04a_subset_approval.json \
  --run-root /absolute/path/containing/scforge_v2/canary_run \
  --mode pre-tractography-canary \
  --response-calibration-manifest /absolute/run/frozen_calibration/frozen_response_calibration_manifest.json \
  --response-calibration-minimum APPROVED_MINIMUM_N \
  --response-calibration-decision /absolute/path/to/calibration_approval.json \
  --cores 4
```

Stage 3 requires the exact blinded human-review manifest and H04B continuation
decision. It schedules tractography only for explicitly approved reviewable
units, while the launcher closes the complete canary denominator in the final
attrition ledger:

```bash
.venv_connectome_workflow/bin/python scforge/workflow/run_connectome_v2.py \
  --manifest /absolute/path/to/approved_acquisition_pair_manifest_v2.csv \
  --execution-subset-manifest /absolute/path/to/h04a_canary_subset.csv \
  --execution-subset-decision /absolute/path/to/h04a_subset_approval.json \
  --run-root /absolute/path/containing/scforge_v2/canary_run \
  --mode phase-b \
  --response-calibration-manifest /absolute/run/frozen_calibration/frozen_response_calibration_manifest.json \
  --response-calibration-minimum APPROVED_MINIMUM_N \
  --response-calibration-decision /absolute/path/to/calibration_approval.json \
  --human-qc-manifest /absolute/path/to/blinded_human_visual_qc.csv \
  --tractography-continuation-decision /absolute/path/to/h04b_continuation_approval.json \
  --cores 32
```

The launcher creates immutable lineage, mode, and attempt identities, runs the
locked Snakemake executable with keep-going and expanded-command printing, and
hashes the combined log. Every mode refreshes the manifest and execution
preflight before its new scientific work begins. Inputs whose exact content is
already attested are timestamp-insensitive at completed scientific consumers,
so a later mode cannot silently recompute approved calibration, preprocessing,
FOD/tensor, or visual-review products. The staged dry-run exercises this exact
phase-A to pre-tractography to phase-B reuse contract. The launcher closes every
canary unit, including early failures, before publishing:

- a complete all-unit attrition ledger,
- a PASS-only analysis-ready manifest (header-only while biological inference
  remains locked), and
- an all-unit, all-outcome validity/NA manifest.

The launcher writes final completion evidence only when expected and terminal
unit counts are equal and every referenced digest revalidates. Direct ad hoc
Snakemake execution is not the provenance-complete route.

For H04A, the launcher reserves the remaining signed storage envelope before
start, counts prior Phase-A and pre-tractography attempt durations (including
failed attempts), and monitors elapsed time and run-root storage while Snakemake
runs. A breach terminates the isolated process group and is recorded as a failed
attempt. If the host or launcher itself is forcibly killed before it can write
the immutable attempt end, the next invocation fails on the unclosed attempt;
a human must resolve that evidence before any restart. Human H04A/H04B approval
and blinded visual review also remain manual gates.

Common smoke commands:

```bash
cd /home/ec2-user/exp/scforge
PYTHONPATH=. /home/ec2-user/fsl/bin/python -m scforge.cli status --config configs/scforge.yaml
PYTHONPATH=. /home/ec2-user/fsl/bin/python -m scforge.cli aal3-summary --config configs/scforge.yaml
PYTHONPATH=. /home/ec2-user/fsl/bin/python -m scforge.cli init --config configs/scforge.yaml
PYTHONPATH=. /home/ec2-user/fsl/bin/python -m unittest discover -s tests
```

Useful dry-run planners:

```bash
PYTHONPATH=. /home/ec2-user/fsl/bin/python -m scforge.cli plan-dwi \
  --subject TEST --nifti dwi.nii.gz --bval dwi.bval --bvec dwi.bvec \
  --out-dir /tmp/scforge_dwi --config configs/scforge.yaml

PYTHONPATH=. /home/ec2-user/fsl/bin/python -m scforge.cli plan-fod \
  --subject TEST --dwi-biascorr dwi.mif --mask mask.mif \
  --out-dir /tmp/scforge_fod --shells 0 1000 --config configs/scforge.yaml

PYTHONPATH=. /home/ec2-user/fsl/bin/python -m scforge.cli plan-preflight \
  --subject TEST --wmfod wmfod.mif --five-tt 5tt.mif --gmwmi gmwmi.mif \
  --out-dir /tmp/scforge_tracks --config configs/scforge.yaml

PYTHONPATH=. /home/ec2-user/fsl/bin/python -m scforge.cli plan-connectome \
  --subject TEST --tracks tracks.tck --nodes nodes.nii.gz \
  --out-dir /tmp/scforge_conn --assignment-variant radial8 \
  --config configs/scforge.yaml
```
