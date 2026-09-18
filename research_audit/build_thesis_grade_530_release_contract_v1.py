#!/usr/bin/env python3
"""Freeze the measurable 530-case thesis release contract without imaging."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml


ROOT = Path("/home/ec2-user/exp")
AUDIT = ROOT / "research_audit"
SCFORGE = ROOT / "scforge"
OUTPUT = AUDIT / "outputs/thesis_grade_530_release_contract_v1"
CONTRACT_JSON = OUTPUT / "contract.json"
CONTRACT_MD = OUTPUT / "contract.md"
VALIDATION_JSON = OUTPUT / "validation.json"
VALIDATION_MD = OUTPUT / "validation.md"

CONFIG = ROOT / "configs/connectome_v2_retry3.yaml"
MATRIX_DICTIONARY = AUDIT / "matrix_data_dictionary.md"
ANALYSIS_DESIGN = AUDIT / "closed_world_analysis_design_v1.md"
INPUT_MANIFEST = AUDIT / "outputs/connectome_v2_input_manifest_v2.csv"
INPUT_SCHEMA = SCFORGE / "workflow/schemas/acquisition_manifest_v2.schema.json"
QC_SOURCE = SCFORGE / "scforge/qc.py"
PROVENANCE_SOURCE = SCFORGE / "scforge/provenance.py"
PREFLIGHT_RULE = SCFORGE / "workflow/rules/06_preflight.smk"
PUBLISH_RULE = SCFORGE / "workflow/rules/08_qc_publish.smk"
RECOVERY2_VALIDATION = (
    AUDIT
    / "outputs/h04a_r1_retry4_pretract_recovery2_package_v1/validation.json"
)
S3_LOCATION_RECORD = ROOT / "docs/s3/synced_ec2_s3_locations.txt"

EXPECTED_MATRICES = [
    "count",
    "fd_sum",
    "count_invnodevol",
    "len_mean",
    "invlen_mean",
    "fa_mean",
    "md_mean",
    "rd_mean",
    "ad_mean",
]
EXPECTED_CASES = 530
EXPECTED_MATRIX_FILES = EXPECTED_CASES * len(EXPECTED_MATRICES)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise ValueError(f"not a regular evidence file: {resolved}")
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def write_immutable(path: Path, data: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    write_immutable(
        path,
        (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite frozen contract: {OUTPUT}")
    sources = [
        CONFIG,
        MATRIX_DICTIONARY,
        ANALYSIS_DESIGN,
        INPUT_MANIFEST,
        INPUT_SCHEMA,
        QC_SOURCE,
        PROVENANCE_SOURCE,
        PREFLIGHT_RULE,
        PUBLISH_RULE,
        RECOVERY2_VALIDATION,
        S3_LOCATION_RECORD,
        Path(__file__),
    ]
    missing = [str(path) for path in sources if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)

    with INPUT_MANIFEST.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    units = [f"{row['subject_id']}_I{row['dti_image_id']}" for row in rows]
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    recovery2 = json.loads(RECOVERY2_VALIDATION.read_text(encoding="utf-8"))
    existing_terminal_records = sorted(
        Path("/data/derivatives/scforge_v2").glob("**/08_qc/terminal_record.json")
    )
    process_text = subprocess.run(
        [
            "pgrep",
            "-af",
            "snakemake|eddy_cpu|eddy_cuda|dwifslpreproc|dwi2response|ss3t_csd_beta1|antsRegistration|c3d_affine_tool|tckgen|tcksift2",
        ],
        check=False,
        capture_output=True,
        text=True,
    ).stdout
    active_imaging_processes = [
        line for line in process_text.splitlines() if "pgrep -af" not in line
    ]
    configured_matrices = [
        item["name"] for item in config["connectome"]["required_matrices"]
    ]
    source_records = {path.name: file_record(path) for path in sources}

    hard_case_gate = {
        "source_identity": [
            "source pair is one of the 530 checksum-locked manifest rows",
            "DWI and T1 normalization records re-hash to the locked sources",
            "gradient count equals DWI volume count",
            "b-vectors and b-values pass the locked gradient contract",
            "phase-encoding direction and total readout time are source-derived; no default or imputation",
        ],
        "preprocessing": [
            "denoise, de-Gibbs, Eddy, bias correction and brain-mask stages complete without fallback",
            "Eddy exits successfully and eddy_quad emits a complete quantitative record",
            "finite nonempty preprocessed DWI and mask outputs",
            "quantitative QC warnings are adjudicated while blind to diagnosis and research outcomes",
        ],
        "anatomy_spatial_model": [
            "T1 bias correction, brain extraction and 5TT complete",
            "5TT-to-DWI-mask Dice >= 0.70",
            "atlas-inside-brain fraction >= 0.80",
            "all 166 contiguous AAL3 labels have positive physical volume",
            "BBR and MNI transforms are finite and nonsingular",
            "WM FOD and normalized tissue compartments are finite and valid",
            "FA is within [0,1], diffusivities within [0,0.01] mm2/s, and tensor identities hold without clipping",
            "all required blinded b0/T1, b0/5TT and b0/atlas overlays pass human review",
        ],
        "tractography_connectome": [
            "exactly 10,000,000 ACT/iFOD2 streamlines complete under the frozen recipe",
            "SIFT2 weight count equals streamline count and mu/iteration records are present",
            "endpoint assignment fraction >= 0.85",
            "all 166 nodes receive at least one endpoint assignment",
            "top-five-node endpoint concentration <= 0.35",
            "all nine 166x166 matrices pass strict algebraic, physical, support and semantic invariants",
            "raw count is integer and demonstrably distinct from SIFT2 fd_sum",
        ],
        "provenance_release": [
            "terminal state is PASS, never PARTIAL/FAIL relabeled as usable",
            "all source, contract, command, environment, QC and output records are hash-bound",
            "all nine matrix file records re-hash and schema/semantic validation passes",
            "no zero padding, value clipping, biological imputation, density-based route selection or silent fallback",
        ],
    }

    contract = {
        "schema_version": "1.0.0",
        "record_type": "thesis_grade_structural_connectome_530_release_contract",
        "status": "FROZEN_ACCEPTANCE_CONTRACT_EXECUTION_INCOMPLETE",
        "generated_utc": utc_now(),
        "objective": "produce 530 genuinely high-quality corrected structural-connectome case bundles, retain a verified independent copy, and release them as thesis-grade only after technical and inferential audits pass",
        "strict_target": {
            "expected_cases": EXPECTED_CASES,
            "required_case_green": EXPECTED_CASES,
            "required_matrix_families_per_case": len(EXPECTED_MATRICES),
            "required_matrix_files": EXPECTED_MATRIX_FILES,
            "required_terminal_records": EXPECTED_CASES,
            "not_attempted_allowed": 0,
            "partial_count_allowed_for_goal_completion": 0,
            "fail_count_allowed_for_goal_completion": 0,
            "fabricated_or_imputed_connectomes_allowed": 0,
            "rule": "530 terminal records are necessary but not sufficient; goal completion requires 530 case-level GREEN bundles",
        },
        "case_green_definition": hard_case_gate,
        "matrix_names": EXPECTED_MATRICES,
        "fixed_thresholds": {
            "five_tt_to_dwi_mask_dice_minimum": 0.70,
            "atlas_inside_brain_fraction_minimum": 0.80,
            "endpoint_assignment_fraction_minimum": 0.85,
            "unique_endpoint_assigned_nodes_minimum": 166,
            "top5_endpoint_fraction_maximum": 0.35,
            "fa_range": [0.0, 1.0],
            "diffusivity_range_mm2_per_s": [0.0, 0.01],
            "tractogram_streamlines": 10_000_000,
            "matrix_shape": [166, 166],
            "matrix_symmetry_absolute_tolerance": 1.0e-8,
            "matrix_zero_diagonal_absolute_tolerance": 1.0e-8,
            "count_integer_absolute_tolerance": 1.0e-6,
        },
        "diagnosis_blind_recipe_stability_gate": {
            "scope": "phase-B canary before any 530-case scale-up",
            "independent_seed_required": True,
            "streamline_convergence_levels": [3_000_000, 5_000_000, 10_000_000],
            "minimum_support_matched_upper_triangle_spearman": 0.95,
            "minimum_tensor_matrix_upper_triangle_pearson": 0.98,
            "maximum_global_metric_relative_difference": 0.05,
            "minimum_node_strength_absolute_agreement_icc": 0.90,
            "maximum_assignment_fraction_absolute_difference": 0.02,
            "selection_may_use_diagnosis_or_group_effect": False,
            "failure_action": "do not scale; repair or issue a new recipe and repeat the blinded canary",
        },
        "human_qc_contract": {
            "diagnosis_and_outcome_blinded": True,
            "all_cases_reviewed": True,
            "second_rater_scope": "all automated warnings plus a deterministic diagnosis-blind 20 percent audit sample",
            "minimum_interrater_kappa": 0.80,
            "disagreements": "adjudicate while blinded and retain both original ratings",
            "density_used_for_review_or_inclusion": False,
        },
        "cohort_release_green": {
            "technical_green_530": "all 530 case bundles satisfy case_green_definition",
            "attrition_green": "530/530 are accounted for; any non-GREEN case prevents the strict 530-output objective from being marked complete",
            "inference_green": [
                "the statistical analysis plan is frozen before corrected group effects are opened",
                "outcome denominators and group/site/protocol missingness are explicit",
                "primary effects pass prespecified multiplicity, acquisition, timing, QC and influence checks",
                "prediction is grouped and leakage-safe; no clinical validation claim without external data",
                "novelty is claimed only at the exact granularity surviving the independent prior-art audit",
            ],
            "thesis_green_rule": "TECHNICAL_GREEN_530 and INFERENCE_GREEN and COPY_GREEN must all pass",
        },
        "copy_contract": {
            "primary_release_root": "/data/releases/structural_connectome_530_v1",
            "independent_copy": "s3://sabeesh/exp/releases/structural_connectome_530_v1/",
            "upload_policy": "no upload until the local candidate passes the full release evaluator",
            "required_local_artifacts": [
                "dataset_description.json",
                "README.md",
                "release_manifest.json",
                "files_sha256.tsv",
                "case_status.csv",
                "analysis_ready_manifest.csv",
                "all_outcome_validity.csv",
            ],
            "remote_integrity": [
                "upload with AWS CLI checksum algorithm SHA256",
                "exact remote key count and byte total equal the local manifest",
                "every object exposes a stored checksum or is covered by a server-side checksum report",
                "remote release-manifest digest equals the local digest",
            ],
            "restore_test": "restore the manifests plus a deterministic 15-case technical-diversity sample into a fresh directory and reproduce every SHA256",
            "copy_green_requires": "local release PASS, upload PASS, remote checksum PASS and restore PASS",
        },
        "current_state": {
            "classification": "NOT_GREEN_CANARY_INCOMPLETE",
            "corrected_full_case_bundles_green": len(existing_terminal_records),
            "recovery2_package_status": recovery2.get("status"),
            "recovery2_residual_jobs": recovery2.get("scheduled_job_counts", {}).get("total"),
            "imaging_running_at_contract_build": bool(active_imaging_processes),
            "next_gate": "SL-H04A-CAL-R2",
            "goal_complete": False,
        },
        "evidence": source_records,
        "scientific_basis": [
            {"doi": "10.1038/s41592-021-01185-5", "role": "integrated reproducible dMRI preprocessing and reconstruction"},
            {"doi": "10.1016/j.neuroimage.2018.09.073", "role": "single-subject and study-wise Eddy QC"},
            {"doi": "10.1016/j.neuroimage.2012.06.005", "role": "anatomically constrained tractography"},
            {"doi": "10.1016/j.neuroimage.2015.06.092", "role": "SIFT2 connectivity weighting"},
            {"url": "https://bids-specification.readthedocs.io/en/stable/derivatives/introduction.html", "role": "derivative provenance and reuse"},
            {"url": "https://docs.aws.amazon.com/AmazonS3/latest/userguide/checking-object-integrity.html", "role": "off-host copy integrity"},
        ],
    }

    checks = {
        "exact_530_manifest_rows": len(rows) == EXPECTED_CASES,
        "exact_530_unique_units": len(set(units)) == EXPECTED_CASES,
        "all_rows_processing_authorized": all(
            row["processing_authorized"].strip().lower() == "true" for row in rows
        ),
        "all_four_diagnosis_labels_retained": set(
            row["diagnosis_at_dti"] for row in rows
        ) == {"CN", "MCI", "AD", "SMC"},
        "exact_nine_matrix_contract": configured_matrices == EXPECTED_MATRICES,
        "strict_166_node_contract": config["qc"]["matrix"]["exact_shape"] == [166, 166],
        "no_padding": config["qc"]["matrix"]["padding_allowed"] is False,
        "no_metric_imputation": config["qc"]["matrix"]["absent_metric_imputation_allowed"] is False,
        "no_density_inclusion": config["qc"]["matrix"]["density_used_as_inclusion_gate"] is False,
        "no_automatic_fallbacks": config["contract"]["silent_fallbacks_allowed"] is False,
        "human_visual_review_required": config["qc"]["human_visual_review_required"] is True,
        "publish_pass_only": config["provenance"]["publish_only_statuses"] == ["PASS"],
        "spatial_thresholds_bound": config["registration"]["quantitative_qc"] == {
            "affine_finite_and_nonsingular": True,
            "minimum_5tt_to_dwi_mask_dice": 0.70,
            "minimum_atlas_label_inside_brain_fraction": 0.80,
        },
        "assignment_thresholds_bound": config["qc"]["assignment"]["candidate_reporting_threshold_minimum_endpoint_assignment_fraction"] == 0.85
        and config["qc"]["assignment"]["candidate_reporting_threshold_minimum_unique_assigned_nodes"] == 166
        and config["qc"]["assignment"]["candidate_reporting_threshold_maximum_top5_endpoint_fraction"] == 0.35,
        "recovery2_preparation_pass": recovery2.get("status") == "PASS"
        and recovery2.get("scheduled_job_counts", {}).get("total") == 166,
        "no_corrected_phase_b_terminal_bundle_exists": len(existing_terminal_records) == 0,
        "no_imaging_process_active": not active_imaging_processes,
        "documented_s3_root_matches_copy_target": "S3 root: s3://sabeesh/exp/"
        in S3_LOCATION_RECORD.read_text(encoding="utf-8"),
        "contract_does_not_claim_current_green": contract["current_state"]["goal_complete"] is False,
        "strict_goal_requires_530_green": contract["strict_target"]["required_case_green"] == EXPECTED_CASES,
        "independent_copy_and_restore_required": "restore" in contract["copy_contract"]["restore_test"],
    }
    if not all(checks.values()):
        raise RuntimeError(
            "contract source validation failed: "
            + ", ".join(name for name, passed in checks.items() if not passed)
        )

    OUTPUT.mkdir(parents=True, exist_ok=False)
    write_json(CONTRACT_JSON, contract)
    markdown = f"""# Thesis-grade 530-case structural-connectome release contract v1

**State:** `{contract['current_state']['classification']}`  
**Strict delivery target:** 530/530 case-level GREEN bundles and {EXPECTED_MATRIX_FILES:,} valid matrix files  
**Current corrected full bundles:** 0/530  
**Next execution gate:** `SL-H04A-CAL-R2`

## What GREEN means

A filename is not an output-quality result. A case is GREEN only when its locked DWI/T1 identity, gradient and Eddy processing, registration, tissue/FOD/tensor products, 10M ACT tractogram, SIFT2 weights, endpoint assignments, all nine matrices, blinded visual QC, provenance and hashes all pass. PARTIAL or FAIL cases remain explicit NA and cannot be relabelled, padded, clipped or imputed.

The strict project objective is met only at **530 case-level GREEN records**. If even one case remains invalid, the pipeline may still yield an attrition-complete scientific dataset, but it is not honestly described as “530 high-quality connectomes,” and this objective remains incomplete.

## Fixed case thresholds

- 5TT-to-DWI mask Dice >= 0.70.
- Atlas-inside-brain fraction >= 0.80.
- Exactly 166 positive-volume AAL3 nodes.
- Exactly 10,000,000 ACT/iFOD2 streamlines and matching SIFT2 weights.
- Endpoint assignment >= 0.85; all 166 nodes represented; top-five endpoint concentration <= 0.35.
- Nine finite, symmetric, nonnegative, zero-diagonal 166x166 matrices with the frozen tensor, support and semantic invariants.
- All blinded spatial overlays PASS and immutable terminal provenance PASS.

## Reliability before scale-up

The diagnosis-blind phase-B canary must pass independent-seed and 3M/5M/10M convergence. Minimum operational reliability is support-matched edge Spearman 0.95, tensor-matrix Pearson 0.98, node-strength absolute-agreement ICC 0.90, no global-metric change above 5%, and no endpoint-assignment change above 0.02. These thresholds are quality controls, not biological effect-selection rules.

## Human QC

All cases receive diagnosis/outcome-blinded review. Every automated warning plus a deterministic 20% audit sample receives a second rating; kappa must be >=0.80, and disagreements are adjudicated while retaining the original ratings.

## Independent copy

The passing local release is staged at `/data/releases/structural_connectome_530_v1`. Its independent off-host copy is `s3://sabeesh/exp/releases/structural_connectome_530_v1/`. Upload remains disabled until local validation passes. COPY_GREEN requires SHA256 upload/checksum evidence, exact remote inventory equality, and a fresh-directory restore of the manifests plus a deterministic 15-case technical-diversity sample.

## Final thesis verdict

`THESIS_GREEN = TECHNICAL_GREEN_530 + INFERENCE_GREEN + COPY_GREEN`.

The current verdict is **not green** because the 15-case pre-tractography qualification is not yet complete and no corrected phase-B case bundle exists. This statement prevents partial preparation from being mistaken for completion.
"""
    write_immutable(CONTRACT_MD, markdown.encode("utf-8"))
    validation = {
        "schema_version": "1.0.0",
        "record_type": "thesis_grade_530_release_contract_validation",
        "status": "PASS",
        "generated_utc": utc_now(),
        "checks": checks,
        "check_count": len(checks),
        "passed_check_count": sum(checks.values()),
        "contract": file_record(CONTRACT_JSON),
        "human_readable_contract": file_record(CONTRACT_MD),
        "imaging_executed": False,
        "copy_executed": False,
        "goal_complete": False,
        "next_gate": "SL-H04A-CAL-R2",
    }
    write_json(VALIDATION_JSON, validation)
    validation_md = "\n".join(
        [
            "# Thesis-grade 530 release-contract validation",
            "",
            "**Contract validation:** PASS",
            "**Current 530-case release:** NOT GREEN / not yet generated",
            "",
            *[f"- [{'x' if value else ' '}] {name}" for name, value in checks.items()],
            "",
            "This validates the acceptance definition and its bindings only. It does not claim that imaging, 530 case bundles, statistical confirmation, upload, or restore has completed.",
        ]
    ) + "\n"
    write_immutable(VALIDATION_MD, validation_md.encode("utf-8"))
    print(
        json.dumps(
            {
                "status": "PASS",
                "checks": len(checks),
                "strict_case_target": EXPECTED_CASES,
                "strict_matrix_target": EXPECTED_MATRIX_FILES,
                "current_state": contract["current_state"]["classification"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
