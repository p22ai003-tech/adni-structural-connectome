"""Fail-closed compatibility and provenance checks for SL-H04A-R1."""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
import sc_exclusions as _sc_exclusions


import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .input_contract import validate_resolved_runtime_config as _validate_base_runtime


EXPECTED_RUN_ROOT = Path(
    "/data/derivatives/scforge_v2/h04a_r1_recovery_20260718_retry2"
).resolve()
# Loaded from a gitignored local file: these name ADNI participants.
EXPECTED_UNITS = _sc_exclusions.subject_list("h04a_r1_expected_units")
EXPECTED_CONVERTER_SHA256 = (
    "353acb3b370faade70b552f284e0a306214a146c076c9f449c5c555b9b148ebd"
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_file_record(record: Any, *, label: str) -> Path:
    if not isinstance(record, Mapping):
        raise ValueError(f"{label} must be a file record")
    path = Path(str(record.get("path", ""))).expanduser().resolve()
    if (
        not path.is_file()
        or record.get("sha256") != _sha256_file(path)
        or record.get("size_bytes") != path.stat().st_size
    ):
        raise ValueError(f"{label} file identity differs: {path}")
    return path


def validate_recovery_extension_binding(
    binding: Any, *, expected_run_root: Path
) -> dict[str, Any]:
    """Validate every R1-only source and the exact bounded authorization."""

    if not isinstance(binding, dict):
        raise ValueError("R1 resolved config lacks recovery_extension_binding")
    expected = {
        "schema_version": "1.0.0",
        "record_type": "h04a_r1_recovery_extension_binding",
        "status": "LOCKED",
        "run_root": str(EXPECTED_RUN_ROOT),
        "allowed_terminal_stage": "response_calibration_phase_a_freeze_decision",
        "diagnosis_labels_used": False,
        "approved_unit_count": 15,
        "units": EXPECTED_UNITS,
        "maximum_cpu_cores": 32,
        "maximum_wall_clock_hours": 24,
        "maximum_additional_storage_gib": 100,
        "minimum_valid_response_calibration_units": 12,
        "required_manufacturer_families": 2,
        "required_t1_source_classes": 2,
        "forbidden_stages": [
            "fod_reconstruction",
            "tensor_maps",
            "tractography",
            "sift2",
            "connectome_matrices",
            "statistical_analysis",
            "dashboard_publication",
        ],
    }
    for key, value in expected.items():
        if binding.get(key) != value:
            raise ValueError(f"R1 extension binding differs at {key}")
    if expected_run_root.resolve() != EXPECTED_RUN_ROOT:
        raise ValueError("R1 run root differs from the approved isolated root")
    source_files = binding.get("source_files")
    if not isinstance(source_files, Mapping) or set(source_files) != {
        "snakefile",
        "manifest_rule",
        "input_rule",
        "launcher",
        "compatibility_validator",
    }:
        raise ValueError("R1 extension source-file set differs")
    for label, record in source_files.items():
        _verify_file_record(record, label=f"R1 source {label}")
    converter_path = _verify_file_record(binding.get("converter"), label="R1 converter")
    if (
        binding["converter"].get("sha256") != EXPECTED_CONVERTER_SHA256
        or converter_path.name != "dcm2niix"
    ):
        raise ValueError("R1 converter is not the approved pinned dcm2niix")
    for label in (
        "source_approval",
        "source_recommendation",
        "execution_decision",
        "execution_manifest",
        "package_validation",
    ):
        _verify_file_record(binding.get(label), label=f"R1 {label}")
    decision_path = Path(binding["execution_decision"]["path"])
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    if (
        decision.get("decision_type")
        != "connectome_response_recovery_subset_approval"
        or decision.get("status") != "APPROVED"
        or decision.get("approval_mode") != "H04A_R1_BOUNDED_RESPONSE_RECOVERY"
        or decision.get("units") != EXPECTED_UNITS
        or decision.get("converter", {}).get("sha256")
        != EXPECTED_CONVERTER_SHA256
        or decision.get("diagnosis_labels_used") is not False
    ):
        raise ValueError("R1 derived execution decision semantics differ")
    return copy.deepcopy(binding)


def _validated_r1_authorization(execution_binding: Mapping[str, Any]) -> Mapping[str, Any]:
    authorization = execution_binding.get("h04a_authorization")
    if not isinstance(authorization, Mapping):
        raise ValueError("R1 execution binding lacks authorization")
    expected = {
        "approval_mode": "H04A_R1_BOUNDED_RESPONSE_RECOVERY",
        "authorized_modes": ["response-calibration-phase-a"],
        "authorized_through": "response_calibration_phase_a_freeze_decision",
        "maximum_cores": 32,
        "minimum_valid_response_calibration_units": 12,
        "minimum_valid_manufacturer_families": 2,
        "minimum_valid_t1_source_classes": 2,
        "required_valid_t1_source_classes": ["dicom_series", "nifti_single"],
        "wall_clock_stop_hours": 24,
        "wall_clock_stop_seconds": 86400,
        "storage_stop_gb": 100,
        "storage_stop_bytes": 100_000_000_000,
        "tractography_authorized": False,
        "matrix_generation_authorized": False,
        "full_cohort_authorized": False,
        "fod_reconstruction_authorized": False,
        "tensor_maps_authorized": False,
        "statistical_analysis_authorized": False,
        "dashboard_publication_authorized": False,
    }
    for key, value in expected.items():
        if authorization.get(key) != value:
            raise ValueError(f"R1 authorization differs at {key}")
    if authorization.get("proposed_run_root") != str(EXPECTED_RUN_ROOT):
        raise ValueError("R1 authorization run root differs")
    return authorization


def _base_compatibility_authorization(
    authorization: Mapping[str, Any], *, run_root: str
) -> dict[str, Any]:
    """Map only resource-envelope fields for the unchanged base validator.

    The stricter R1 contract is fully checked first.  This adapter then lets the
    unchanged scientific-config comparator prove no imaging parameters changed.
    """

    return {
        "approval_mode": "H04A_BOUNDED_CANARY",
        "approved_by": authorization["approved_by"],
        "approved_utc": authorization["approved_utc"],
        "user_response": authorization["user_response"],
        "proposed_run_root": run_root,
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
    """Validate R1 authority, then prove the scientific config is unchanged."""

    resolved = copy.deepcopy(dict(resolved_config))
    context = copy.deepcopy(dict(run_context))
    if resolved.get("launcher_mode") != "response-calibration-phase-a":
        raise ValueError("R1 can launch only response-calibration Phase A")
    run_root = Path(str(resolved.get("run_root", ""))).expanduser().resolve()
    extension = validate_recovery_extension_binding(
        resolved.get("recovery_extension_binding"), expected_run_root=run_root
    )
    if context.get("recovery_extension_binding") != extension:
        raise ValueError("R1 extension binding differs across config and run context")
    execution_binding = resolved.get("execution_binding")
    if (
        not isinstance(execution_binding, dict)
        or execution_binding.get("units") != EXPECTED_UNITS
        or execution_binding.get("approved_unit_count") != 15
        or execution_binding != context.get("execution_binding")
    ):
        raise ValueError("R1 execution binding or exact unit denominator differs")
    authorization = _validated_r1_authorization(execution_binding)

    resolved.pop("recovery_extension_binding", None)
    context.pop("recovery_extension_binding", None)
    compatibility = _base_compatibility_authorization(
        authorization, run_root=str(run_root)
    )
    resolved_binding = copy.deepcopy(resolved["execution_binding"])
    resolved_binding["h04a_authorization"] = compatibility
    context_binding = copy.deepcopy(context["execution_binding"])
    context_binding["h04a_authorization"] = copy.deepcopy(compatibility)
    resolved["execution_binding"] = resolved_binding
    context["execution_binding"] = context_binding
    _validate_base_runtime(normative_config, resolved, context)
