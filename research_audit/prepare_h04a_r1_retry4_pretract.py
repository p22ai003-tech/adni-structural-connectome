#!/usr/bin/env python3
"""Prepare and dry-run the exact retry4 pre-tractography continuation.

This script is package-only.  It cannot execute imaging: the Snakemake command
is forced to ``--dry-run`` and both decision records remain
``AWAITING_USER_APPROVAL``.  Its purpose is to prove the exact 15-unit lineage,
frozen diagnosis-blind response calibration, bounded rule DAG, and resource
envelope before SL-H04A-CAL is signed.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml


ROOT = Path("/home/ec2-user/exp")
AUDIT = ROOT / "research_audit"
WORKFLOW = ROOT / "scforge/workflow"
RUN_ROOT = Path(
    "/data/derivatives/scforge_v2/h04a_r1_recovery_20260719_retry4"
)
PACKAGE = AUDIT / "outputs/h04a_r1_retry4_pretract_package_v3"
CONFIG = ROOT / "configs/connectome_v2_retry3.yaml"
ENVIRONMENT = WORKFLOW / "environment_contract_retry3.yaml"
SNAKEFILE = WORKFLOW / "Snakefile_h04a_r1_retry4_pretract_v2"
PARENT_MANIFEST = AUDIT / "outputs/connectome_v2_input_manifest_v2.csv"
SUBSET_MANIFEST = (
    AUDIT
    / "outputs/h04a_r1_recovery_package_v1/recovery_execution_manifest_v1.csv"
)
RECOVERY_BINDING = AUDIT / "outputs/h04a_r1_recovery_extension_binding_v7.json"
PHASE_A_COMPLETION = RUN_ROOT / "publication/response_calibration_phase_a_completion.json"
PHASE_A_MANIFEST = RUN_ROOT / "publication/response_calibration_phase_a_manifest.json"
FROZEN_DIR = RUN_ROOT / "frozen_calibration_retry4_v2"
FROZEN_MANIFEST = FROZEN_DIR / "frozen_response_calibration_manifest.json"
FREEZE_ATTESTATION = FROZEN_DIR / "retry4_freeze_attestation.json"

RESPONSE_DECISION = PACKAGE / "response_calibration_decision_proposed.json"
PRETRACT_DECISION = PACKAGE / "pretract_extension_decision_proposed.json"
DRYRUN_BINDING = PACKAGE / "pretract_dryrun_binding.json"
DRYRUN_CONFIG = PACKAGE / "resolved_config_dryrun.yaml"
DRYRUN_LOG = PACKAGE / "snakemake_dry_run.txt"
VALIDATION = PACKAGE / "validation.json"
VALIDATION_MD = PACKAGE / "validation.md"

MODE = "pre-tractography-canary"
TARGET = "pre_tractography_canary"
REQUIRED_USER_RESPONSE = "Approve SL-H04A-CAL"
MAXIMUM_CPU_CORES = 32
MAXIMUM_WALL_CLOCK_HOURS = 24
TOTAL_STORAGE_STOP_GB = 150
TOTAL_STORAGE_STOP_BYTES = TOTAL_STORAGE_STOP_GB * 1_000_000_000

# Only rules at or after mean-b0 extraction and before tractography are allowed.
ALLOWED_RULES = (
    "freeze_manifest",
    "execution_preflight",
    "mean_b0",
    "t1_n4_bias_correct",
    "t1_brain_extract",
    "mean_b0_nifti",
    "b0_to_t1_bbr",
    "invert_bbr_transform",
    "mni_to_t1_nonlinear",
    "b0_one_mm_world_grid",
    "aal3_to_dwi_single_resample",
    "atlas_contract_qc",
    "atlas_native_grid_qc_copy",
    "five_tt_t1",
    "five_tt_wmseg",
    "five_tt_dwi",
    "gmwmi_dwi",
    "select_tensor_shells",
    "pooled_response",
    "fod_shell_compatibility_gate",
    "ss3t_csd",
    "mtnormalise",
    "tensor_fit",
    "tensor_metrics",
    "spatial_qc_images",
    "spatial_quantitative_qc",
    "t1_in_b0_for_visual_qc",
    "visual_review_bundle",
    "automated_pre_tractography_qc",
    TARGET,
)

FORBIDDEN_RULES = frozenset(
    {
        "normalize_dwi_source",
        "normalize_t1_source",
        "input_contract_gate",
        "gradient_contract",
        "dwi_denoise",
        "dwi_degibbs",
        "dwi_motion_eddy",
        "dwi_bias_correct",
        "dwi_brain_mask",
        "select_fod_shells",
        "subject_response",
        "response_calibration_phase_a",
        "tractography_preflight",
        "tractography_10m",
        "sift2_weights",
        "connectome_count",
        "connectome_fd_sum",
        "connectome_len_mean",
        "connectome_invlen_mean",
        "tensor_streamline_samples",
        "tensor_connectome",
        "count_invnodevol",
        "matrix_qc",
        "provenance_sidecar",
        "publish_manifest",
        "phase_b_subject_terminal_records",
        "all",
    }
)

FORBIDDEN_STAGES = (
    "source_normalization",
    "denoise_degibbs_eddy_bias_mask_recomputation",
    "response_reestimation",
    "tractography",
    "sift2",
    "connectome_matrices",
    "statistical_analysis",
    "dashboard_publication",
    "full_cohort_processing",
)

REQUIRED_EXISTING_PRODUCTS = (
    "00_inputs/t1_native.nii",
    "00_inputs/input_contract.json",
    "01_dwi/dwi_preproc.mif",
    "01_dwi/dwi_preproc_biascorr.mif",
    "01_dwi/dwi_brain_mask.mif",
    "01_dwi/gradient_contract.json",
    "05_model/dwi_fod_shells.mif",
    "05_model/fod_shell_selection.json",
    "05_model/response_calibration_outcome.json",
    "05_model/response_wm.txt",
    "05_model/response_gm.txt",
    "05_model/response_csf.txt",
)


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


def verify_file_record(record: Mapping[str, Any], *, label: str) -> Path:
    path = Path(str(record.get("path", ""))).expanduser().resolve()
    if (
        not path.is_file()
        or path.is_symlink()
        or record.get("sha256") != sha256_file(path)
        or record.get("size_bytes") != path.stat().st_size
    ):
        raise ValueError(f"{label} file record differs: {path}")
    return path


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def count_existing_products(units: list[str]) -> dict[str, int]:
    counts = {relative: 0 for relative in REQUIRED_EXISTING_PRODUCTS}
    for unit in units:
        for relative in REQUIRED_EXISTING_PRODUCTS:
            path = RUN_ROOT / "subjects" / unit / relative
            if not path.is_file() or path.is_symlink() or path.stat().st_size < 1:
                raise FileNotFoundError(f"required frozen prerequisite missing: {unit}/{relative}")
            counts[relative] += 1
    if set(counts.values()) != {len(units)}:
        raise ValueError(f"existing-product denominator differs: {counts}")
    return counts


def forbidden_artifacts() -> list[str]:
    patterns = (
        "subjects/*/05_model/wmfod.mif",
        "subjects/*/05_model/wmfod_norm.mif",
        "subjects/*/05_model/tensor.mif",
        "subjects/*/06_preflight/automated_pre_tractography_qc.json",
        "**/*.tck",
        "**/sift2_weights.txt",
        "subjects/*/07_connectome/matrices/*.csv",
        "publication/pre_tractography_canary_manifest.json",
        "publication/pre_tractography_canary_completion.json",
    )
    found: list[str] = []
    for pattern in patterns:
        found.extend(
            str(path.relative_to(RUN_ROOT)) for path in RUN_ROOT.glob(pattern)
        )
    return sorted(set(found))


def build_candidate_response_binding(
    *, completion: Mapping[str, Any], frozen: Mapping[str, Any]
) -> dict[str, Any]:
    pooled = {
        tissue: file_record(Path(str(frozen["pooled_responses"][tissue]["path"])))
        for tissue in ("wm", "gm", "csf")
    }
    return {
        "schema_version": "2.0.0",
        "binding_type": "response_calibration_pretract_dryrun_binding",
        "status": "DRY_RUN_ONLY_NOT_EXECUTION_AUTHORITY",
        "recipe_id": completion["recipe_id"],
        "phase_a_run_id": completion["run_id"],
        "phase_a_completion": file_record(PHASE_A_COMPLETION),
        "phase_a_manifest": file_record(PHASE_A_MANIFEST),
        "response_calibration_manifest": file_record(FROZEN_MANIFEST),
        "minimum_valid_subjects": int(frozen["minimum_valid_subjects"]),
        "valid_units": list(frozen["valid_units"]),
        "valid_pool_technical_diversity": copy.deepcopy(
            frozen["valid_pool_technical_diversity"]
        ),
        "valid_pool_technical_diversity_sha256": frozen[
            "valid_pool_technical_diversity_sha256"
        ],
        "response_calibration_decision_candidate": file_record(RESPONSE_DECISION),
        "pooled_responses": pooled,
        "diagnosis_labels_used": False,
        "phase_b_live_all_units_dependency": False,
        "dummy_response_files_used": False,
    }


def main() -> int:
    if PACKAGE.exists():
        raise FileExistsError(f"refusing to overwrite package: {PACKAGE}")
    required = (
        RUN_ROOT,
        CONFIG,
        ENVIRONMENT,
        SNAKEFILE,
        PARENT_MANIFEST,
        SUBSET_MANIFEST,
        RECOVERY_BINDING,
        PHASE_A_COMPLETION,
        PHASE_A_MANIFEST,
        FROZEN_MANIFEST,
        FREEZE_ATTESTATION,
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"pre-tractography prerequisites missing: {missing}")

    active = subprocess.run(
        [
            "pgrep",
            "-af",
            "snakemake|eddy_cpu|eddy_cuda|dwifslpreproc|dwi2response|ss3t_csd_beta1|tckgen|tcksift2",
        ],
        check=False,
        capture_output=True,
        text=True,
    ).stdout
    active_lines = [line for line in active.splitlines() if "pgrep -af" not in line]
    if active_lines:
        raise RuntimeError("imaging process already active: " + " | ".join(active_lines[:5]))

    completion = load_json(PHASE_A_COMPLETION)
    phase_a = load_json(PHASE_A_MANIFEST)
    frozen = load_json(FROZEN_MANIFEST)
    attestation = load_json(FREEZE_ATTESTATION)
    execution_binding = completion.get("execution_binding")
    if (
        completion.get("record_type") != "response_calibration_phase_a_completion"
        or completion.get("status") != "PASS"
        or completion.get("mode") != "response-calibration-phase-a"
        or completion.get("phase_a_summary")
        != {"expected": 15, "fail": 0, "pass": 15, "terminal": 15}
        or not isinstance(execution_binding, dict)
        or execution_binding.get("approved_unit_count") != 15
        or execution_binding.get("diagnosis_labels_used") is not False
    ):
        raise ValueError("retry4 phase-A completion is not exact 15/15 PASS evidence")
    units = execution_binding.get("units")
    if not isinstance(units, list) or units != sorted(set(units)) or len(units) != 15:
        raise ValueError("retry4 exact unit set differs")
    if completion.get("phase_a_manifest") != file_record(PHASE_A_MANIFEST):
        raise ValueError("phase-A completion no longer binds the exact manifest")
    if (
        phase_a.get("execution_binding") != execution_binding
        or phase_a.get("expected_units") != units
        or phase_a.get("summary") != completion.get("phase_a_summary")
    ):
        raise ValueError("phase-A manifest/completion denominator differs")

    if (
        attestation.get("record_type")
        != "response_calibration_freeze_retry4_attestation"
        or attestation.get("status") != "PASS"
        or attestation.get("authoritative_for_next_gate") is not True
        or attestation.get("diagnosis_labels_used") is not False
        or attestation.get("frozen_manifest") != file_record(FROZEN_MANIFEST)
        or attestation.get("valid_subject_count") != 15
        or attestation.get("invalid_subject_count") != 0
    ):
        raise ValueError("retry4 V2 freeze attestation differs")

    sys.path.insert(0, str(ROOT / "scforge"))
    sys.path.insert(0, str(WORKFLOW))
    from scforge.response_calibration import validate_frozen_response_calibration
    import run_h04a_r1_retry4 as retry4

    pooled_records = {
        tissue: file_record(Path(str(frozen["pooled_responses"][tissue]["path"])))
        for tissue in ("wm", "gm", "csf")
    }
    validate_frozen_response_calibration(
        FROZEN_MANIFEST,
        expected_manifest_sha256=sha256_file(FROZEN_MANIFEST),
        minimum_valid_subjects=12,
        expected_responses=pooled_records,
        expected_technical_diversity=frozen["valid_pool_technical_diversity"],
    )
    if (
        frozen.get("valid_units") != units
        or frozen.get("valid_subject_count") != 15
        or frozen.get("invalid_subject_count") != 0
        or frozen.get("diagnosis_labels_used") is not False
    ):
        raise ValueError("frozen response denominator differs from exact retry4 units")

    product_counts = count_existing_products(units)
    forbidden_before = forbidden_artifacts()
    if forbidden_before:
        raise ValueError(f"downstream artifacts already exist: {forbidden_before[:10]}")

    storage = retry4.base.measure_run_root_storage(RUN_ROOT)
    disk = shutil.disk_usage(RUN_ROOT)
    if storage["effective_bytes"] >= TOTAL_STORAGE_STOP_BYTES:
        raise RuntimeError("pre-tractography total storage cap is already exhausted")
    remaining = TOTAL_STORAGE_STOP_BYTES - int(storage["effective_bytes"])
    if disk.free < remaining:
        raise RuntimeError(
            f"free disk cannot honor storage cap: free={disk.free}, remaining={remaining}"
        )

    PACKAGE.mkdir(parents=True, exist_ok=False)
    diversity = frozen["valid_pool_technical_diversity"]
    diversity_contract = {
        "minimum_valid_manufacturer_families": diversity[
            "minimum_valid_manufacturer_families"
        ],
        "minimum_valid_t1_source_classes": diversity[
            "minimum_valid_t1_source_classes"
        ],
        "required_valid_t1_source_classes": diversity[
            "required_valid_t1_source_classes"
        ],
    }
    response_decision = {
        "schema_version": "2.0.0",
        "decision_type": "response_calibration_phase_b_approval",
        "status": "AWAITING_USER_APPROVAL",
        "decision_gate": "SL-H04A-CAL",
        "required_user_response": REQUIRED_USER_RESPONSE,
        "recipe_id": completion["recipe_id"],
        "phase_a_completion_sha256": sha256_file(PHASE_A_COMPLETION),
        "response_calibration_manifest_sha256": sha256_file(FROZEN_MANIFEST),
        "response_calibration_freeze_attestation": file_record(FREEZE_ATTESTATION),
        "minimum_valid_subjects": 12,
        **diversity_contract,
        "valid_unit_count": 15,
        "valid_manufacturer_counts": diversity["manufacturer_counts"],
        "valid_manufacturer_family_counts": diversity[
            "manufacturer_family_counts"
        ],
        "valid_t1_source_class_counts": diversity["t1_source_class_counts"],
        "valid_pool_technical_diversity_sha256": frozen[
            "valid_pool_technical_diversity_sha256"
        ],
        "diagnosis_labels_used": False,
        "biological_inference_authorized": False,
        "note": "Approval accepts only the frozen response for bounded pre-tractography processing; it is not a CN/MCI/AD or novelty claim.",
    }
    write_json(RESPONSE_DECISION, response_decision)

    pretract_decision = {
        "schema_version": "1.0.0",
        "decision_type": "connectome_retry4_pretract_extension_approval",
        "status": "AWAITING_USER_APPROVAL",
        "decision_gate": "SL-H04A-CAL",
        "required_user_response": REQUIRED_USER_RESPONSE,
        "approval_mode": "H04A_R1_RETRY4_PRETRACT_ONLY",
        "run_root": str(RUN_ROOT),
        "recipe_id": completion["recipe_id"],
        "execution_scope": "exact_15_unit_canary",
        "approved_unit_count": 15,
        "units": units,
        "diagnosis_labels_used": False,
        "phase_a_completion": file_record(PHASE_A_COMPLETION),
        "phase_a_manifest": file_record(PHASE_A_MANIFEST),
        "frozen_response_manifest": file_record(FROZEN_MANIFEST),
        "freeze_attestation": file_record(FREEZE_ATTESTATION),
        "execution_subset_manifest": execution_binding[
            "execution_subset_manifest"
        ],
        "prior_response_recovery_decision": execution_binding[
            "execution_subset_decision"
        ],
        "maximum_cpu_cores": MAXIMUM_CPU_CORES,
        "maximum_wall_clock_hours": MAXIMUM_WALL_CLOCK_HOURS,
        "total_run_root_storage_stop_gb": TOTAL_STORAGE_STOP_GB,
        "total_run_root_storage_stop_bytes": TOTAL_STORAGE_STOP_BYTES,
        "storage_at_packaging": storage,
        "allowed_terminal_stage": "automated_qc_and_blinded_review_bundle",
        "allowed_rules": list(ALLOWED_RULES),
        "rerun_triggers": ["input", "params"],
        "forbidden_rules": sorted(FORBIDDEN_RULES),
        "forbidden_stages": list(FORBIDDEN_STAGES),
        "tractography_authorized": False,
        "matrix_generation_authorized": False,
        "statistical_analysis_authorized": False,
        "dashboard_publication_authorized": False,
        "full_cohort_authorized": False,
        "stop_after_dag": TARGET,
        "next_human_gate": "SL-H04B",
        "dry_run_only_until_approved": True,
        "source_files": {
            "snakefile": file_record(SNAKEFILE),
            "compatibility_validator": file_record(
                ROOT / "scforge/scforge/h04a_r1_retry4_pretract.py"
            ),
            "preflight_rule": file_record(
                WORKFLOW
                / "extensions/h04a_r1_retry4_pretract/00_manifest.smk"
            ),
            "authorization_builder": file_record(
                AUDIT / "authorize_h04a_r1_retry4_pretract.py"
            ),
            "execution_launcher": file_record(
                WORKFLOW / "run_h04a_r1_retry4_pretract.py"
            ),
            "package_builder": file_record(Path(__file__)),
        },
    }
    write_json(PRETRACT_DECISION, pretract_decision)

    candidate_response_binding = build_candidate_response_binding(
        completion=completion, frozen=frozen
    )
    dryrun_binding = {
        "schema_version": "1.0.0",
        "record_type": "retry4_pretract_dryrun_binding",
        "status": "DRY_RUN_ONLY_NOT_EXECUTION_AUTHORITY",
        "mode": MODE,
        "target": TARGET,
        "run_root": str(RUN_ROOT),
        "units": units,
        "execution_binding": execution_binding,
        "response_calibration_binding": candidate_response_binding,
        "response_decision_candidate": file_record(RESPONSE_DECISION),
        "pretract_decision_candidate": file_record(PRETRACT_DECISION),
        "recovery_extension_binding": file_record(RECOVERY_BINDING),
        "imaging_execution_authorized": False,
    }
    write_json(DRYRUN_BINDING, dryrun_binding)

    normative = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    environment = yaml.safe_load(ENVIRONMENT.read_text(encoding="utf-8"))
    recovery_binding = load_json(RECOVERY_BINDING)
    resolved = copy.deepcopy(normative)
    resolved["manifest_path"] = str(PARENT_MANIFEST)
    resolved["run_root"] = str(RUN_ROOT)
    resolved["launcher_mode"] = MODE
    resolved["execution_binding"] = copy.deepcopy(execution_binding)
    resolved["recovery_extension_binding"] = recovery_binding
    resolved["pretract_dryrun_binding"] = dryrun_binding
    resolved["pretract_extension_binding"] = dryrun_binding
    resolved["response_calibration_binding"] = candidate_response_binding
    resolved["execution_manifest_path"] = execution_binding[
        "execution_subset_manifest"
    ]["path"]
    resolved["inputs"]["approved_pair_manifest"]["path"] = str(PARENT_MANIFEST)
    resolved["inputs"]["approved_pair_manifest"]["sha256"] = sha256_file(
        PARENT_MANIFEST
    )
    calibration = resolved["fod"]["response_estimation"]["calibration"]
    calibration["minimum_valid_subjects"] = 12
    calibration["frozen_manifest"] = file_record(FROZEN_MANIFEST)
    calibration["pooled_responses"] = pooled_records

    with tempfile.TemporaryDirectory(
        prefix=".retry4_pretract_dryrun_", dir=RUN_ROOT
    ) as temporary:
        temporary_root = Path(temporary)
        run_context = temporary_root / "run_context.json"
        attempt_context = temporary_root / "attempt_context.json"
        resolved_temp = temporary_root / "resolved_config.yaml"
        context_common = {
            "schema_version": "1.0.0",
            "status": "DRY_RUN_ONLY",
            "launcher_mode": MODE,
            "recipe_id": completion["recipe_id"],
            "run_root": str(RUN_ROOT),
            "execution_binding": execution_binding,
            "response_calibration_binding": candidate_response_binding,
            "recovery_extension_binding": recovery_binding,
            "pretract_extension_binding": dryrun_binding,
            "pretract_dryrun_binding": dryrun_binding,
        }
        run_context.write_text(
            json.dumps(
                {**context_common, "run_id": "retry4-pretract-dryrun"},
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        attempt_context.write_text(
            json.dumps(
                {
                    **context_common,
                    "run_id": "retry4-pretract-dryrun",
                    "attempt_id": "retry4-pretract-dryrun-attempt",
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        resolved["run_context_path"] = str(run_context)
        resolved["attempt_context_path"] = str(attempt_context)
        resolved["resolved_run_config_path"] = str(resolved_temp)
        resolved_temp.write_text(
            yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8"
        )
        DRYRUN_CONFIG.write_text(
            yaml.safe_dump(
                {
                    **resolved,
                    "run_context_path": "DRY_RUN_EPHEMERAL_CONTEXT",
                    "attempt_context_path": "DRY_RUN_EPHEMERAL_CONTEXT",
                    "resolved_run_config_path": str(DRYRUN_CONFIG),
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        command = [
            str(environment["workflow"]["snakemake"]["path"]),
            "--snakefile",
            str(SNAKEFILE),
            "--configfile",
            str(resolved_temp),
            "--cores",
            str(MAXIMUM_CPU_CORES),
            "--rerun-incomplete",
            "--rerun-triggers",
            "input",
            "params",
            "--allowed-rules",
            *ALLOWED_RULES,
            "--keep-going",
            "--printshellcmds",
            "--show-failed-logs",
            "--dry-run",
            TARGET,
        ]
        run = subprocess.run(
            command,
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        output = (run.stdout or "") + ("\n" + run.stderr if run.stderr else "")
        DRYRUN_LOG.write_text(output, encoding="utf-8")
        if run.returncode != 0:
            raise RuntimeError(f"pre-tractography dry run failed: {run.returncode}")

    scheduled_rules = sorted(
        set(re.findall(r"(?m)^rule ([A-Za-z0-9_]+):\s*$", output))
    )
    job_counts: dict[str, int] = {}
    for rule in scheduled_rules:
        match = re.search(rf"(?m)^{re.escape(rule)}\s+(\d+)\s*$", output)
        if match:
            job_counts[rule] = int(match.group(1))
    total = re.search(r"(?m)^total\s+(\d+)\s*$", output)
    job_counts["total"] = int(total.group(1)) if total else -1
    forbidden_scheduled = sorted(set(scheduled_rules) & FORBIDDEN_RULES)
    missing_allowed = sorted(set(ALLOWED_RULES) - set(scheduled_rules))
    unexpected = sorted(set(scheduled_rules) - set(ALLOWED_RULES))
    forbidden_after = forbidden_artifacts()
    checks = {
        "phase_a_exact_15_of_15_pass": True,
        "frozen_response_v2_valid": True,
        "manufacturer_families_ge_2": diversity["manufacturer_family_count"] >= 2,
        "t1_source_classes_ge_2": diversity["t1_source_class_count"] >= 2,
        "diagnosis_labels_unused": frozen["diagnosis_labels_used"] is False,
        "all_15_frozen_prerequisites_present": all(
            value == 15 for value in product_counts.values()
        ),
        "no_downstream_artifacts_before_dryrun": not forbidden_before,
        "snakemake_dryrun_pass": run.returncode == 0,
        "scheduled_rules_exact_allowlist": not missing_allowed and not unexpected,
        "no_forbidden_rules_scheduled": not forbidden_scheduled,
        "no_imaging_artifacts_created_by_dryrun": forbidden_after == forbidden_before,
        "resource_total_storage_below_cap": storage["effective_bytes"]
        < TOTAL_STORAGE_STOP_BYTES,
        "free_space_can_honor_cap": disk.free >= remaining,
        "decisions_remain_unapproved": response_decision["status"]
        == "AWAITING_USER_APPROVAL"
        and pretract_decision["status"] == "AWAITING_USER_APPROVAL",
    }
    report = {
        "schema_version": "1.0.0",
        "record_type": "retry4_pretract_package_validation",
        "generated_utc": utc_now(),
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "run_root": str(RUN_ROOT),
        "unit_count": len(units),
        "units": units,
        "scheduled_rules": scheduled_rules,
        "scheduled_job_counts": job_counts,
        "missing_allowed_rules": missing_allowed,
        "unexpected_rules": unexpected,
        "forbidden_scheduled_rules": forbidden_scheduled,
        "allowed_rules": list(ALLOWED_RULES),
        "forbidden_rules": sorted(FORBIDDEN_RULES),
        "forbidden_stages": list(FORBIDDEN_STAGES),
        "resource_envelope": {
            "maximum_cpu_cores": MAXIMUM_CPU_CORES,
            "maximum_wall_clock_hours": MAXIMUM_WALL_CLOCK_HOURS,
            "total_storage_stop_bytes": TOTAL_STORAGE_STOP_BYTES,
            "storage_at_packaging": storage,
            "storage_remaining_to_stop_bytes": remaining,
            "filesystem_free_bytes": disk.free,
        },
        "technical_calibration_scope": {
            "sufficient_for": [
                "pooled_response_calibration",
                "scanner_and_T1_source_technical_canary",
                "bounded_pretractography_QC",
            ],
            "not_sufficient_for": [
                "CN_MCI_AD_effect_estimation",
                "biomarker_validation",
                "novelty_claim",
                "final_statistical_inference",
            ],
        },
        "evidence": {
            "phase_a_completion": file_record(PHASE_A_COMPLETION),
            "frozen_response_manifest": file_record(FROZEN_MANIFEST),
            "freeze_attestation": file_record(FREEZE_ATTESTATION),
            "response_decision_candidate": file_record(RESPONSE_DECISION),
            "pretract_decision_candidate": file_record(PRETRACT_DECISION),
            "dryrun_binding": file_record(DRYRUN_BINDING),
            "dryrun_log": file_record(DRYRUN_LOG),
            "snakefile": file_record(SNAKEFILE),
        },
        "next_gate": "SL-H04A-CAL",
        "required_user_response": REQUIRED_USER_RESPONSE,
        "imaging_executed": False,
    }
    write_json(VALIDATION, report)
    lines = [
        "# Retry4 bounded pre-tractography package",
        "",
        f"**Generated:** {report['generated_utc']}",
        f"**Verdict:** {report['status']}",
        f"**Units:** {len(units)} (technical calibration only)",
        f"**Dry-run jobs:** {job_counts['total']}",
        f"**Next gate:** {report['next_gate']}",
        "",
        "## Checks",
        "",
    ]
    lines.extend(
        f"- [{'x' if passed else ' '}] {name}" for name, passed in checks.items()
    )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "The package schedules anatomical, registration, atlas, FOD, tensor, automated-QC and blinded-review products only. It does not schedule preprocessing/Eddy, response re-estimation, tractography, SIFT2, matrices, statistics, dashboard publication or full-cohort work.",
            "",
            f"Actual imaging remains disabled until the exact response is: `{REQUIRED_USER_RESPONSE}`.",
        ]
    )
    VALIDATION_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if report["status"] != "PASS":
        raise RuntimeError("pre-tractography package validation failed")
    print(
        json.dumps(
            {
                "status": report["status"],
                "unit_count": report["unit_count"],
                "scheduled_job_counts": job_counts,
                "next_gate": report["next_gate"],
                "imaging_executed": False,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
