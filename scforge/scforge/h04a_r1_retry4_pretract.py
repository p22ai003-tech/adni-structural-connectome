"""Fail-closed compatibility checks for retry4 pre-tractography continuation.

The completed response-recovery execution binding remains immutable.  A
separate extension binding grants (or, for package dry-runs, explicitly does
not grant) authority for the bounded pre-tractography DAG.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Mapping

from . import h04a_r1_retry4 as recovery
from .input_contract import validate_resolved_runtime_config as _validate_base_runtime


EXPECTED_RUN_ROOT = Path(
    "/data/derivatives/scforge_v2/h04a_r1_recovery_20260719_retry4"
).resolve()
REQUIRED_USER_RESPONSE = "Approve SL-H04A-CAL"
ALLOWED_RULES = [
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
    "pre_tractography_canary",
]
FORBIDDEN_STAGES = [
    "source_normalization",
    "denoise_degibbs_eddy_bias_mask_recomputation",
    "response_reestimation",
    "tractography",
    "sift2",
    "connectome_matrices",
    "statistical_analysis",
    "dashboard_publication",
    "full_cohort_processing",
]


def _load_record(record: Any, *, label: str) -> tuple[Path, dict[str, Any]]:
    path = recovery.base._verify_file_record(record, label=label)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{label} JSON root must be an object")
    return path, value


def validate_recovery_extension_binding(
    binding: Any, *, expected_run_root: Path
) -> dict[str, Any]:
    """Retain the exact completed retry4 response-recovery binding."""

    return recovery.validate_recovery_extension_binding(
        binding, expected_run_root=expected_run_root
    )


def validate_pretract_extension_binding(
    binding: Any,
    *,
    expected_run_root: Path,
    execution_binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate either a non-authoritative dry-run or signed live extension."""

    if not isinstance(binding, dict):
        raise ValueError("retry4 pre-tractography extension binding is missing")
    if expected_run_root.expanduser().resolve() != EXPECTED_RUN_ROOT:
        raise ValueError("retry4 pre-tractography run root differs")
    units = execution_binding.get("units")
    expected_common = {
        "schema_version": "1.0.0",
        "run_root": str(EXPECTED_RUN_ROOT),
        "units": units,
    }
    for key, value in expected_common.items():
        if binding.get(key) != value:
            raise ValueError(f"retry4 pre-tractography binding differs at {key}")
    if (
        not isinstance(units, list)
        or units != recovery.base.EXPECTED_UNITS
        or execution_binding.get("approved_unit_count") != 15
        or execution_binding.get("diagnosis_labels_used") is not False
    ):
        raise ValueError("retry4 pre-tractography denominator differs")

    _, pretract_decision = _load_record(
        binding.get("pretract_decision_candidate")
        or binding.get("pretract_decision"),
        label="retry4 pre-tractography decision",
    )
    _, response_decision = _load_record(
        binding.get("response_decision_candidate")
        or binding.get("response_calibration_decision"),
        label="retry4 response-calibration decision",
    )
    common_decision = {
        "decision_type": "connectome_retry4_pretract_extension_approval",
        "decision_gate": "SL-H04A-CAL",
        "required_user_response": REQUIRED_USER_RESPONSE,
        "approval_mode": "H04A_R1_RETRY4_PRETRACT_ONLY",
        "run_root": str(EXPECTED_RUN_ROOT),
        "approved_unit_count": 15,
        "units": units,
        "diagnosis_labels_used": False,
        "maximum_cpu_cores": 32,
        "maximum_wall_clock_hours": 24,
        "total_run_root_storage_stop_gb": 150,
        "total_run_root_storage_stop_bytes": 150_000_000_000,
        "allowed_terminal_stage": "automated_qc_and_blinded_review_bundle",
        "allowed_rules": ALLOWED_RULES,
        "rerun_triggers": ["input", "params"],
        "forbidden_stages": FORBIDDEN_STAGES,
        "tractography_authorized": False,
        "matrix_generation_authorized": False,
        "statistical_analysis_authorized": False,
        "dashboard_publication_authorized": False,
        "full_cohort_authorized": False,
        "stop_after_dag": "pre_tractography_canary",
        "next_human_gate": "SL-H04B",
    }
    for key, value in common_decision.items():
        if pretract_decision.get(key) != value:
            raise ValueError(f"pre-tractography decision differs at {key}")
    if response_decision.get("decision_type") != "response_calibration_phase_b_approval":
        raise ValueError("response-calibration decision type differs")

    status = binding.get("status")
    if status == "DRY_RUN_ONLY_NOT_EXECUTION_AUTHORITY":
        if (
            binding.get("record_type") != "retry4_pretract_dryrun_binding"
            or binding.get("imaging_execution_authorized") is not False
            or pretract_decision.get("status") != "AWAITING_USER_APPROVAL"
            or response_decision.get("status") != "AWAITING_USER_APPROVAL"
        ):
            raise ValueError("dry-run binding improperly grants execution authority")
    elif status == "LOCKED":
        if (
            binding.get("record_type")
            != "retry4_pretract_execution_extension_binding"
            or binding.get("imaging_execution_authorized") is not True
            or pretract_decision.get("status") != "APPROVED"
            or response_decision.get("status") != "APPROVED"
            or pretract_decision.get("user_response") != REQUIRED_USER_RESPONSE
        ):
            raise PermissionError("retry4 pre-tractography approval is not exact")
        for decision, label in (
            (pretract_decision, "pre-tractography"),
            (response_decision, "response-calibration"),
        ):
            for key in ("approved_by", "approved_utc", "user_response"):
                if not isinstance(decision.get(key), str) or not decision[key].strip():
                    raise PermissionError(f"{label} decision lacks {key}")
    else:
        raise ValueError("retry4 pre-tractography binding status differs")
    return copy.deepcopy(binding)


def _generic_authorization(
    authorization: Mapping[str, Any], *, run_root: Path
) -> dict[str, Any]:
    """Adapt resource fields solely for the unchanged scientific comparator."""

    return {
        "approval_mode": "H04A_BOUNDED_CANARY",
        "approved_by": authorization["approved_by"],
        "approved_utc": authorization["approved_utc"],
        "user_response": authorization["user_response"],
        "proposed_run_root": str(run_root),
        "authorized_modes": [
            "response-calibration-phase-a",
            "pre-tractography-canary",
        ],
        "authorized_through": "pre_tractography_review_bundle_only",
        "maximum_cores": 4,
        "minimum_valid_response_calibration_units": 12,
        "minimum_valid_manufacturer_families": 2,
        "minimum_valid_t1_source_classes": 2,
        "required_valid_t1_source_classes": ["dicom_series", "nifti_single"],
        "wall_clock_stop_hours": 72,
        "wall_clock_stop_seconds": 259200,
        "storage_stop_gb": 150,
        "storage_stop_bytes": 150_000_000_000,
        "tractography_authorized": False,
        "matrix_generation_authorized": False,
        "full_cohort_authorized": False,
        "normative_config_sha256": authorization["normative_config_sha256"],
        "workflow_source_manifest_sha256": authorization[
            "workflow_source_manifest_sha256"
        ],
        "environment_contract_sha256": authorization[
            "environment_contract_sha256"
        ],
    }


def validate_h04a_r1_resolved_runtime_config(
    normative_config: Mapping[str, Any],
    resolved_config: Mapping[str, Any],
    run_context: Mapping[str, Any],
) -> None:
    """Prove runtime injections do not alter the locked scientific recipe."""

    resolved = copy.deepcopy(dict(resolved_config))
    context = copy.deepcopy(dict(run_context))
    if resolved.get("launcher_mode") != "pre-tractography-canary":
        raise ValueError("retry4 pre-tractography validator received another mode")
    run_root = Path(str(resolved.get("run_root", ""))).expanduser().resolve()
    recovery_binding = validate_recovery_extension_binding(
        resolved.get("recovery_extension_binding"), expected_run_root=run_root
    )
    if context.get("recovery_extension_binding") != recovery_binding:
        raise ValueError("recovery binding differs across config and run context")
    execution_binding = resolved.get("execution_binding")
    if (
        not isinstance(execution_binding, dict)
        or execution_binding != context.get("execution_binding")
    ):
        raise ValueError("execution binding differs across config and run context")
    pretract_binding = validate_pretract_extension_binding(
        resolved.get("pretract_extension_binding"),
        expected_run_root=run_root,
        execution_binding=execution_binding,
    )
    if context.get("pretract_extension_binding") != pretract_binding:
        raise ValueError("pre-tractography binding differs across immutable contexts")
    response_binding = resolved.get("response_calibration_binding")
    if (
        not isinstance(response_binding, dict)
        or response_binding != context.get("response_calibration_binding")
    ):
        raise ValueError("response-calibration binding differs across contexts")
    if pretract_binding["status"] == "LOCKED":
        if response_binding.get("binding_type") != "response_calibration_phase_b_binding":
            raise PermissionError("live pre-tractography requires an approved response binding")
    elif response_binding.get("status") != "DRY_RUN_ONLY_NOT_EXECUTION_AUTHORITY":
        raise ValueError("dry-run response binding status differs")

    authorization = execution_binding.get("h04a_authorization")
    if not isinstance(authorization, Mapping):
        raise ValueError("response-recovery execution authorization is missing")
    for payload in (resolved, context):
        payload.pop("recovery_extension_binding", None)
        payload.pop("pretract_extension_binding", None)
        payload.pop("pretract_dryrun_binding", None)
        adapted_execution = copy.deepcopy(payload["execution_binding"])
        adapted_execution["h04a_authorization"] = _generic_authorization(
            authorization, run_root=run_root
        )
        payload["execution_binding"] = adapted_execution
        adapted_response = copy.deepcopy(payload["response_calibration_binding"])
        adapted_response["binding_type"] = "response_calibration_phase_b_binding"
        payload["response_calibration_binding"] = adapted_response
    _validate_base_runtime(normative_config, resolved, context)
