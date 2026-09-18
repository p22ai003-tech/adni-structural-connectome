"""Fail-closed compatibility checks for the SL-H04A-R1 retry4 continuation."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any, Mapping

from . import h04a_r1 as base
from .retry_seed_retry4 import RETRY4_ID, validate_retry4_seed_manifest


EXPECTED_RUN_ROOT = Path(
    "/data/derivatives/scforge_v2/h04a_r1_recovery_20260719_retry4"
).resolve()
EXPECTED_SOURCE_FILES = {
    "snakefile",
    "base_snakefile",
    "manifest_rule",
    "input_rule",
    "launcher",
    "base_launcher",
    "compatibility_validator",
    "base_compatibility_validator",
    "retry_seed_validator",
    "base_retry_seed_validator",
    "fod_rule",
}
EXPECTED_EVIDENCE_FILES = (
    "source_approval",
    "source_recommendation",
    "execution_decision",
    "execution_manifest",
    "package_validation",
    "source_patch",
    "seed_manifest",
    "workflow_source_manifest",
    "environment_contract",
    "source_attempt_end",
    "source_completion",
    "retry_recommendation",
    "environment_validation",
    "package_builder",
    "launcher_safety_decision",
    "diagnostic_dryrun",
    "supersedes_binding",
    "worker_seed_validation_decision",
)


def validate_recovery_extension_binding(
    binding: Any, *, expected_run_root: Path
) -> dict[str, Any]:
    if not isinstance(binding, dict):
        raise ValueError("R1 retry4 resolved config lacks recovery_extension_binding")
    expected = {
        "schema_version": "1.0.0",
        "record_type": "h04a_r1_recovery_extension_binding",
        "status": "LOCKED",
        "run_root": str(EXPECTED_RUN_ROOT),
        "retry_id": RETRY4_ID,
        "allowed_terminal_stage": "response_calibration_phase_a_freeze_decision",
        "diagnosis_labels_used": False,
        "approved_unit_count": 15,
        "units": base.EXPECTED_UNITS,
        "maximum_cpu_cores": 32,
        "maximum_wall_clock_hours": 24,
        "maximum_additional_storage_gib": 100,
        "minimum_valid_response_calibration_units": 12,
        "required_manufacturer_families": 2,
        "required_t1_source_classes": 2,
        "rerun_triggers": ["input", "params"],
        "allowed_rules": [
            "select_fod_shells",
            "subject_response",
            "response_calibration_phase_a",
        ],
        "upstream_recomputation_authorized": False,
        "seed_excludes_mutable_continuation_paths": True,
        "worker_seed_validation_policy": (
            "full_content_rehash_in_launcher_and_parent_dag;"
            "signed_manifest_path_and_size_validation_in_snakemake_workers"
        ),
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
            raise ValueError(f"R1 retry4 extension binding differs at {key}")
    if expected_run_root.expanduser().resolve() != EXPECTED_RUN_ROOT:
        raise ValueError("R1 retry4 run root differs from the isolated root")

    source_files = binding.get("source_files")
    if not isinstance(source_files, Mapping) or set(source_files) != EXPECTED_SOURCE_FILES:
        raise ValueError("R1 retry4 extension source-file set differs")
    for label, record in source_files.items():
        base._verify_file_record(record, label=f"R1 retry4 source {label}")
    for label in EXPECTED_EVIDENCE_FILES:
        base._verify_file_record(binding.get(label), label=f"R1 retry4 {label}")

    converter_path = base._verify_file_record(
        binding.get("converter"), label="R1 retry4 converter"
    )
    if (
        binding["converter"].get("sha256") != base.EXPECTED_CONVERTER_SHA256
        or converter_path.name != "dcm2niix"
    ):
        raise ValueError("R1 retry4 converter is not the approved pinned dcm2niix")

    decision_path = Path(str(binding["execution_decision"]["path"]))
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    retry = decision.get("execution_retry")
    if (
        decision.get("decision_type")
        != "connectome_response_recovery_subset_approval"
        or decision.get("status") != "APPROVED"
        or decision.get("approval_mode") != "H04A_R1_BOUNDED_RESPONSE_RECOVERY"
        or decision.get("proposed_run_root") != str(EXPECTED_RUN_ROOT)
        or decision.get("units") != base.EXPECTED_UNITS
        or decision.get("converter", {}).get("sha256")
        != base.EXPECTED_CONVERTER_SHA256
        or decision.get("diagnosis_labels_used") is not False
        or not isinstance(retry, Mapping)
        or retry.get("retry_id") != RETRY4_ID
        or retry.get("authorization_change") is not False
        or retry.get("scientific_parameter_change") is not False
        or retry.get("unit_set_change") is not False
        or retry.get("resource_cap_change") is not False
        or retry.get("rerun_triggers") != ["input", "params"]
        or retry.get("seed_manifest") != binding.get("seed_manifest")
        or retry.get("continuation_paths_excluded_from_seed") is not True
    ):
        raise ValueError("R1 retry4 derived execution-decision semantics differ")

    worker_subprocess = (
        "--target-jobs" in sys.argv
        and "--mode" in sys.argv
        and sys.argv[sys.argv.index("--mode") + 1] == "subprocess"
    )
    validate_retry4_seed_manifest(
        binding["seed_manifest"],
        destination_root=EXPECTED_RUN_ROOT,
        approved_units=base.EXPECTED_UNITS,
        rehash_content=not worker_subprocess,
    )
    return copy.deepcopy(binding)


def validate_h04a_r1_resolved_runtime_config(
    normative_config: Mapping[str, Any],
    resolved_config: Mapping[str, Any],
    run_context: Mapping[str, Any],
) -> None:
    """Reuse the unchanged scientific comparator under the retry4 root."""

    old_root = base.EXPECTED_RUN_ROOT
    old_validator = base.validate_recovery_extension_binding
    try:
        base.EXPECTED_RUN_ROOT = EXPECTED_RUN_ROOT
        base.validate_recovery_extension_binding = validate_recovery_extension_binding
        base.validate_h04a_r1_resolved_runtime_config(
            normative_config, resolved_config, run_context
        )
    finally:
        base.EXPECTED_RUN_ROOT = old_root
        base.validate_recovery_extension_binding = old_validator
