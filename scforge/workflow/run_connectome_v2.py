#!/usr/bin/env python3
"""Fail-closed launcher and final attestation writer for connectome-v2.1.

This is the only supported entry point for a real imaging run. It creates an
immutable run identity and attempt record, invokes the locked Snakemake binary
with keep-going and exact shell-command printing, captures the combined
execution log, and closes every attempted unit before writing mixed-state
attrition and completion evidence.  A nonzero workflow exit never turns a
missing or invalid outcome into a numerical value.
"""

from __future__ import annotations

import argparse
import copy
import csv
import datetime as dt
import hashlib
import json
import math
import os
import platform
import re
import shutil
import signal
import stat
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import yaml


WORKFLOW_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = WORKFLOW_DIR.parents[1]
SCFORGE_ROOT = PROJECT_ROOT / "scforge"
sys.path.insert(0, str(SCFORGE_ROOT))

from scforge.provenance import (  # noqa: E402
    MATRIX_OUTCOMES,
    file_record,
    make_failure_terminal_record,
    publish_terminal_ledger,
    sha256_file,
    validate_schema,
    validate_terminal_semantics,
    verify_file_record,
    write_immutable_bytes,
    write_immutable_json,
    write_immutable_jsonl,
)
from scforge.input_contract import load_acquisition_manifest  # noqa: E402
from scforge.response_calibration import (  # noqa: E402
    TISSUES,
    build_valid_pool_technical_diversity,
    load_execution_subset_technical_metadata,
    technical_diversity_contract_from_authorization,
    technical_diversity_sha256,
    validate_response_calibration_outcome as validate_calibration_outcome,
    validate_frozen_response_calibration,
)


NORMATIVE_CONFIG = PROJECT_ROOT / "configs" / "connectome_v2.yaml"
ENVIRONMENT_CONTRACT = WORKFLOW_DIR / "environment_contract.yaml"
SNAKEFILE = WORKFLOW_DIR / "Snakefile"
PROVENANCE_SCHEMA = WORKFLOW_DIR / "schemas" / "provenance_v2.schema.json"
RUN_LEDGER_SCHEMA = WORKFLOW_DIR / "schemas" / "run_ledger_v2.schema.json"
RUN_MODES = (
    "response-calibration-phase-a",
    "pre-tractography-canary",
    "phase-b",
)
CANARY_MIN_UNITS = 12
CANARY_MAX_UNITS = 24
H04A_AUTHORIZED_MODES = (
    "response-calibration-phase-a",
    "pre-tractography-canary",
)
H04A_AUTHORIZED_THROUGH = "pre_tractography_review_bundle_only"
H04A_MAXIMUM_CORES = 4
H04A_MINIMUM_VALID_RESPONSE_UNITS = 12
H04A_MINIMUM_VALID_MANUFACTURER_FAMILIES = 2
H04A_MINIMUM_VALID_T1_SOURCE_CLASSES = 2
H04A_REQUIRED_VALID_T1_SOURCE_CLASSES = ("dicom_series", "nifti_single")
H04A_WALL_CLOCK_STOP_HOURS = 72
H04A_STORAGE_STOP_GB = 150
DECIMAL_GB_BYTES = 1_000_000_000
RESOURCE_MONITOR_INTERVAL_SECONDS = 5.0
RESOURCE_TERMINATION_GRACE_SECONDS = 30.0

RULE_STAGE = {
    "normalize_dwi_source": "source_normalization",
    "normalize_t1_source": "source_normalization",
    "input_contract_gate": "source_normalization",
    "gradient_contract": "dwi_preprocessing",
    "dwi_denoise": "dwi_preprocessing",
    "dwi_degibbs": "dwi_preprocessing",
    "dwi_motion_eddy": "dwi_preprocessing",
    "dwi_bias_correct": "dwi_preprocessing",
    "mean_b0": "dwi_preprocessing",
    "dwi_brain_mask": "dwi_preprocessing",
    "t1_n4_bias_correct": "anatomical_preprocessing",
    "t1_brain_extract": "anatomical_preprocessing",
    "b0_to_t1_bbr": "registration",
    "invert_bbr_transform": "registration",
    "mni_to_t1_nonlinear": "registration",
    "aal3_to_dwi_single_resample": "atlas_warp",
    "five_tt_t1": "fod_tensor",
    "five_tt_wmseg": "fod_tensor",
    "five_tt_dwi": "fod_tensor",
    "gmwmi_dwi": "fod_tensor",
    "select_fod_tensor_shells": "fod_tensor",
    "subject_response": "fod_tensor",
    "pooled_response": "fod_tensor",
    "ss3t_csd": "fod_tensor",
    "mtnormalise": "fod_tensor",
    "tensor_fit": "fod_tensor",
    "tensor_metrics": "fod_tensor",
    "automated_pre_tractography_qc": "fod_tensor",
    "tractography_preflight": "tractography_preflight",
    "tractography_10m": "tractography",
    "sift2_weights": "tractography",
    "connectome_count": "connectome",
    "connectome_fd_sum": "connectome",
    "connectome_len_mean": "connectome",
    "connectome_invlen_mean": "connectome",
    "tensor_streamline_samples": "connectome",
    "tensor_connectome": "connectome",
    "count_invnodevol": "connectome",
    "matrix_qc": "matrix_qc",
    "visual_review_bundle": "visual_qc",
    "provenance_sidecar": "publication",
    "publish_manifest": "publication",
}


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"YAML root must be a mapping: {path}")
    return data


def load_environment() -> dict[str, Any]:
    """The reference contract with this machine's tool sections laid over it,
    ${repo} resolved -- what the workflow itself reads."""
    from scforge.environment import merged_contract

    return merged_contract()


def validate_normative_execution_lock(config: dict[str, Any]) -> None:
    """Require the immutable recipe to remain non-authoritative for imaging.

    H04A authority is carried only by the separately hashed decision binding.
    Toggling this normative flag would mutate the reviewed scientific recipe and
    is therefore rejected rather than treated as an execution approval.
    """

    contract = config.get("contract")
    if not isinstance(contract, dict):
        raise ValueError("config.contract must be a mapping")
    expected = {
        "status": "implementation_candidate_pending_validation",
        "imaging_execution_authorized": False,
        "current_human_gate": "SL-H03-C1",
        "authorization_scope": "design_only_no_imaging",
        "next_execution_gate": "SL-H04A",
    }
    for key, value in expected.items():
        if contract.get(key) != value:
            raise ValueError(
                "normative execution lock differs at "
                f"contract.{key}: {contract.get(key)!r}; the recipe must remain "
                "immutable and H04A authority must come from the signed decision"
            )
    if contract.get("failure_policy") != "fail_closed":
        raise ValueError("normative execution lock must remain fail_closed")
    if contract.get("silent_fallbacks_allowed") is not False:
        raise ValueError(
            "normative execution lock must continue to forbid silent fallbacks"
        )


def _valid_sha256(value: Any) -> bool:
    return bool(re.fullmatch(r"[a-f0-9]{64}", str(value).lower()))


def _source_identity_hashes(row: dict[str, str]) -> dict[str, str]:
    canonical_keys = (
        "dti_raw_bundle_sha256",
        "t1_raw_bundle_sha256",
        "pair_content_bundle_sha256",
    )
    if all(_valid_sha256(row.get(key, "")) for key in canonical_keys):
        return {key: row[key].lower() for key in canonical_keys}
    hashes = {
        key: str(value).lower()
        for key, value in row.items()
        if key.endswith("sha256") and _valid_sha256(value)
    }
    if not hashes:
        raise ValueError(f"no source-identity hashes for attempted unit {row['unit']}")
    return hashes


def _existing_file_records(paths: dict[str, Path]) -> dict[str, dict[str, Any]]:
    return {name: file_record(path) for name, path in paths.items() if path.is_file()}


def _validate_normalization_failure_marker(
    marker: dict[str, Any],
    *,
    unit: str,
    normalizer: str,
    run_id: str,
    attempt_id: str,
    attempt_context: dict[str, Any],
) -> None:
    expected = {
        "schema_version": "2.0.0",
        "record_type": "source_normalization_failure",
        "status": "FAIL",
        "unit": unit,
        "stage": "source_normalization",
        "normalizer": normalizer,
        "run_id": run_id,
        "attempt_id": attempt_id,
        "attempt_context": attempt_context,
    }
    for key, value in expected.items():
        if marker.get(key) != value:
            raise ValueError(
                f"normalization failure marker differs at {key}: {marker.get(key)!r}"
            )
    reason = marker.get("primary_failure_reason")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("normalization failure marker has no primary_failure_reason")
    flags = marker.get("secondary_flags")
    if not isinstance(flags, list) or any(not isinstance(flag, str) for flag in flags):
        raise ValueError("normalization failure marker secondary_flags is invalid")
    hashes = marker.get("source_identity_hashes")
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("normalization failure marker has no source_identity_hashes")
    if any(not _valid_sha256(value) for value in hashes.values()):
        raise ValueError("normalization failure marker contains an invalid source hash")
    generated = marker.get("generated_utc")
    if not isinstance(generated, str) or not generated:
        raise ValueError("normalization failure marker has no generated_utc")
    try:
        parsed_generated = dt.datetime.fromisoformat(generated.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("normalization failure marker generated_utc is invalid") from exc
    if parsed_generated.tzinfo is None:
        raise ValueError("normalization failure marker generated_utc has no timezone")


def _normalization_failure_evidence(
    run_root: Path,
    unit: str,
    expected_source_identity_hashes: dict[str, str],
    *,
    attempt_context_path: Path,
) -> tuple[str, str, list[str], dict[str, dict[str, Any]]] | None:
    attempt_context_path = attempt_context_path.expanduser().resolve()
    attempt = _load_json_mapping(
        attempt_context_path, label="current attempt context"
    )
    attempt_id = str(attempt.get("attempt_id", ""))
    run_id = str(attempt.get("run_id", ""))
    if (
        attempt.get("schema_version") != "1.0.0"
        or attempt.get("status") != "STARTED"
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", attempt_id)
        or not re.fullmatch(r"[a-f0-9]{24}", run_id)
    ):
        raise ValueError("current attempt context identity is invalid")
    attempt_record = file_record(attempt_context_path)
    marker_paths = {
        normalizer: run_root
        / "subjects"
        / unit
        / "00_inputs"
        / "failures"
        / attempt_id
        / f"{normalizer}_normalization_failure.json"
        for normalizer in ("dwi", "t1")
    }
    present = {name: path for name, path in marker_paths.items() if path.is_file()}
    if not present:
        return None
    reasons: list[str] = []
    flags: list[str] = []
    records: dict[str, dict[str, Any]] = {}
    for normalizer, path in sorted(present.items()):
        try:
            marker = json.loads(path.read_text(encoding="utf-8"))
            _validate_normalization_failure_marker(
                marker,
                unit=unit,
                normalizer=normalizer,
                run_id=run_id,
                attempt_id=attempt_id,
                attempt_context=attempt_record,
            )
            observed_hashes = {
                str(key): str(value).lower()
                for key, value in marker["source_identity_hashes"].items()
            }
            required_marker_keys = {
                "dwi": {"dti_raw_bundle_sha256", "pair_content_bundle_sha256"},
                "t1": {"t1_raw_bundle_sha256", "pair_content_bundle_sha256"},
            }[normalizer]
            if (
                set(observed_hashes) != required_marker_keys
                or any(
                    key not in expected_source_identity_hashes
                    or expected_source_identity_hashes[key] != value
                    for key, value in observed_hashes.items()
                )
            ):
                raise ValueError(
                    "normalization failure marker source hashes differ from manifest"
                )
        except Exception as exc:
            return (
                "workflow_orchestration",
                f"malformed_normalization_failure_marker:{normalizer}:{type(exc).__name__}",
                ["structured_failure_stage_untrusted"],
                {f"{normalizer}_normalization_failure": file_record(path)},
            )
        reasons.append(str(marker["primary_failure_reason"]))
        flags.append(f"normalizer:{normalizer}")
        flags.extend(str(flag) for flag in marker["secondary_flags"])
        records[f"{normalizer}_normalization_failure"] = file_record(path)
    primary = reasons[0]
    flags.extend(f"additional_normalization_failure:{reason}" for reason in reasons[1:])
    return "source_normalization", primary, list(dict.fromkeys(flags)), records


def _failed_rule_evidence(
    log_text: str, expected_units: set[str]
) -> dict[str, list[tuple[str, str]]]:
    """Extract only explicit Snakemake rule/unit error blocks; never infer stage."""

    evidence: dict[str, list[tuple[str, str]]] = {}
    matches = list(
        re.finditer(
            r"(?mi)^(?:Error|[A-Za-z]+Exception)\s+in\s+rule\s+([A-Za-z0-9_]+):\s*$",
            log_text,
        )
    )
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(log_text)
        block = log_text[match.end() : min(end, match.end() + 12000)]
        wildcard = re.search(r"(?mi)^\s*wildcards:\s*([^\n]+)$", block)
        if not wildcard:
            continue
        unit_match = re.search(r"(?:^|,\s*)unit=([^,\s]+)", wildcard.group(1))
        if not unit_match:
            continue
        unit = unit_match.group(1).strip()
        if unit not in expected_units:
            continue
        rule = match.group(1)
        stage = RULE_STAGE.get(rule, "workflow_orchestration")
        evidence.setdefault(unit, []).append((stage, rule))
    return evidence


def _rule_failure_for_unit(
    unit: str,
    evidence: dict[str, list[tuple[str, str]]],
    *,
    returncode: int,
) -> tuple[str, str, list[str]]:
    records = evidence.get(unit, [])
    unique = list(dict.fromkeys(records))
    if not unique:
        return (
            "workflow_orchestration",
            f"terminal_record_missing_unknown_stage_after_snakemake_returncode_{returncode}",
            ["no_unambiguous_failed_rule_block_for_unit"],
        )
    stages = {stage for stage, _ in unique}
    rules = [rule for _, rule in unique]
    if len(stages) != 1 or "workflow_orchestration" in stages:
        return (
            "workflow_orchestration",
            "terminal_record_missing_ambiguous_failed_rule_stage",
            [f"failed_rule:{rule}" for rule in rules],
        )
    stage = next(iter(stages))
    return (
        stage,
        f"snakemake_rule_failed:{rules[0]}",
        [f"additional_failed_rule:{rule}" for rule in rules[1:]],
    )


def _processing_contract(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "registration": config.get("registration", {}),
        "atlas": config.get("atlas", {}),
        "fod": config.get("fod", {}),
        "tensor": config.get("tensor", {}),
        "tractography": config.get("tractography", {}),
        "sift2": config.get("sift2", {}),
        "assignment": config.get("connectome", {}).get("assignment", {}),
    }


def build_snakemake_command(
    *,
    snakemake: Path,
    resolved_config_path: Path,
    cores: int,
    mode: str,
) -> list[str]:
    if mode not in RUN_MODES:
        raise ValueError(f"unsupported launcher mode: {mode!r}")
    command = [
        str(snakemake),
        "--snakefile",
        str(SNAKEFILE),
        "--configfile",
        str(resolved_config_path),
        "--cores",
        str(cores),
        "--rerun-incomplete",
        "--rerun-triggers",
        "mtime",
        "--keep-going",
        "--printshellcmds",
        "--show-failed-logs",
    ]
    if mode == "response-calibration-phase-a":
        command.append("response_calibration_phase_a")
    elif mode == "pre-tractography-canary":
        command.append("pre_tractography_canary")
    else:
        command.append("phase_b_subject_terminal_records")
    return command


def prepare_execution_binding(
    *,
    recipe_id: str,
    parent_manifest_path: Path,
    parent_rows: list[dict[str, str]],
    execution_subset_manifest_path: Path,
    execution_subset_decision_path: Path,
    acquisition_schema_path: Path,
    run_root: Path,
    normative_config_path: Path | None = None,
    environment_contract_path: Path | None = None,
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """Validate one immutable, signed H04A 12--24 unit canary authority."""

    subset_path = execution_subset_manifest_path.expanduser().resolve()
    subset_rows = list(
        load_acquisition_manifest(
            subset_path,
            acquisition_schema_path,
            run_root=run_root,
        )
    )
    if not CANARY_MIN_UNITS <= len(subset_rows) <= CANARY_MAX_UNITS:
        raise ValueError(
            "approved canary execution subset must contain "
            f"{CANARY_MIN_UNITS}--{CANARY_MAX_UNITS} units, found {len(subset_rows)}"
        )
    parent_by_unit = {row["unit"]: row for row in parent_rows}
    if len(parent_by_unit) != len(parent_rows):
        raise ValueError("parent acquisition manifest contains duplicate units")
    for row in subset_rows:
        unit = row["unit"]
        if unit not in parent_by_unit:
            raise ValueError(f"canary execution unit is absent from parent manifest: {unit}")
        if row != parent_by_unit[unit]:
            raise ValueError(
                f"canary execution row differs from exact parent-manifest row: {unit}"
            )

    decision_path = execution_subset_decision_path.expanduser().resolve()
    decision = _load_json_mapping(
        decision_path, label="canary execution-subset decision"
    )
    parent_record = file_record(parent_manifest_path)
    subset_record = file_record(subset_path)
    config_path = (
        normative_config_path if normative_config_path is not None else NORMATIVE_CONFIG
    ).expanduser().resolve()
    environment_path = (
        environment_contract_path
        if environment_contract_path is not None
        else ENVIRONMENT_CONTRACT
    ).expanduser().resolve()
    if environment_path == ENVIRONMENT_CONTRACT.resolve():
        environment = load_environment()
    else:
        from scforge.environment import expand

        environment = expand(load_yaml(environment_path), None)
    source_manifest_record = environment.get("workflow", {}).get(
        "source_manifest"
    )
    if not isinstance(source_manifest_record, dict):
        raise ValueError("environment workflow.source_manifest must be a mapping")
    source_manifest_path = Path(
        str(source_manifest_record.get("path", ""))
    ).expanduser().resolve()
    if not source_manifest_path.is_file():
        raise FileNotFoundError(
            f"workflow source manifest missing: {source_manifest_path}"
        )
    actual_source_manifest_sha256 = sha256_file(source_manifest_path)
    if source_manifest_record.get("sha256") != actual_source_manifest_sha256:
        raise ValueError(
            "environment contract workflow-source-manifest SHA-256 is stale"
        )
    decision_expected = {
        "schema_version": "2.0.0",
        "decision_type": "connectome_canary_execution_subset_approval",
        "status": "APPROVED",
        "approval_mode": "H04A_BOUNDED_CANARY",
        "execution_scope": "canary",
        "recipe_id": recipe_id,
        "parent_acquisition_manifest_sha256": parent_record["sha256"],
        "execution_subset_manifest_sha256": subset_record["sha256"],
        "approved_unit_count": len(subset_rows),
        "diagnosis_labels_used": False,
        "selection_locked": True,
        "proposed_run_root": str(run_root.resolve()),
        "authorized_modes": list(H04A_AUTHORIZED_MODES),
        "authorized_through": H04A_AUTHORIZED_THROUGH,
        "maximum_cores": H04A_MAXIMUM_CORES,
        "minimum_valid_response_calibration_units": (
            H04A_MINIMUM_VALID_RESPONSE_UNITS
        ),
        "minimum_valid_manufacturer_families": (
            H04A_MINIMUM_VALID_MANUFACTURER_FAMILIES
        ),
        "minimum_valid_t1_source_classes": (
            H04A_MINIMUM_VALID_T1_SOURCE_CLASSES
        ),
        "required_valid_t1_source_classes": list(
            H04A_REQUIRED_VALID_T1_SOURCE_CLASSES
        ),
        "h04a_wall_clock_stop_hours": H04A_WALL_CLOCK_STOP_HOURS,
        "h04a_storage_stop_gb": H04A_STORAGE_STOP_GB,
        "tractography_authorized": False,
        "matrix_generation_authorized": False,
        "full_cohort_authorized": False,
        "normative_config_sha256": sha256_file(config_path),
        "workflow_source_manifest_sha256": actual_source_manifest_sha256,
        "environment_contract_sha256": sha256_file(environment_path),
    }
    for key, value in decision_expected.items():
        if decision.get(key) != value:
            raise ValueError(
                f"canary execution-subset decision differs at {key}: "
                f"{decision.get(key)!r}"
            )
    for key in ("approved_by", "approved_utc", "user_response"):
        if not isinstance(decision.get(key), str) or not decision[key].strip():
            raise ValueError(
                f"APPROVED canary execution-subset decision requires signed {key}"
            )
    try:
        approval_time = dt.datetime.fromisoformat(
            decision["approved_utc"].strip().replace("Z", "+00:00")
        )
    except ValueError as exc:
        raise ValueError("H04A approved_utc must be an ISO-8601 timestamp") from exc
    if approval_time.tzinfo is None:
        raise ValueError("H04A approved_utc must include a timezone")
    units = sorted(row["unit"] for row in subset_rows)
    h04a_authorization = {
        "approval_mode": decision_expected["approval_mode"],
        "approved_by": decision["approved_by"].strip(),
        "approved_utc": decision["approved_utc"].strip(),
        "user_response": decision["user_response"].strip(),
        "proposed_run_root": decision_expected["proposed_run_root"],
        "authorized_modes": list(H04A_AUTHORIZED_MODES),
        "authorized_through": H04A_AUTHORIZED_THROUGH,
        "maximum_cores": H04A_MAXIMUM_CORES,
        "minimum_valid_response_calibration_units": (
            H04A_MINIMUM_VALID_RESPONSE_UNITS
        ),
        "minimum_valid_manufacturer_families": (
            H04A_MINIMUM_VALID_MANUFACTURER_FAMILIES
        ),
        "minimum_valid_t1_source_classes": (
            H04A_MINIMUM_VALID_T1_SOURCE_CLASSES
        ),
        "required_valid_t1_source_classes": list(
            H04A_REQUIRED_VALID_T1_SOURCE_CLASSES
        ),
        "wall_clock_stop_hours": H04A_WALL_CLOCK_STOP_HOURS,
        "wall_clock_stop_seconds": H04A_WALL_CLOCK_STOP_HOURS * 60 * 60,
        "storage_stop_gb": H04A_STORAGE_STOP_GB,
        "storage_stop_bytes": H04A_STORAGE_STOP_GB * DECIMAL_GB_BYTES,
        "tractography_authorized": False,
        "matrix_generation_authorized": False,
        "full_cohort_authorized": False,
        "normative_config_sha256": decision_expected[
            "normative_config_sha256"
        ],
        "workflow_source_manifest_sha256": decision_expected[
            "workflow_source_manifest_sha256"
        ],
        "environment_contract_sha256": decision_expected[
            "environment_contract_sha256"
        ],
    }
    binding = {
        "schema_version": "2.0.0",
        "binding_type": "connectome_execution_subset_binding",
        "execution_scope": "canary",
        "recipe_id": recipe_id,
        "parent_acquisition_manifest": parent_record,
        "execution_subset_manifest": subset_record,
        "execution_subset_decision": file_record(decision_path),
        "approved_unit_count": len(units),
        "units": units,
        "diagnosis_labels_used": False,
        "selection_locked": True,
        "h04a_authorization": h04a_authorization,
    }
    subset_rows.sort(key=lambda row: row["unit"])
    return binding, subset_rows


def _validated_h04a_authorization(
    execution_binding: dict[str, Any],
) -> dict[str, Any]:
    authorization = execution_binding.get("h04a_authorization")
    if not isinstance(authorization, dict):
        raise PermissionError("execution binding lacks signed H04A authorization")
    expected = {
        "approval_mode": "H04A_BOUNDED_CANARY",
        "authorized_modes": list(H04A_AUTHORIZED_MODES),
        "authorized_through": H04A_AUTHORIZED_THROUGH,
        "maximum_cores": H04A_MAXIMUM_CORES,
        "minimum_valid_response_calibration_units": (
            H04A_MINIMUM_VALID_RESPONSE_UNITS
        ),
        "minimum_valid_manufacturer_families": (
            H04A_MINIMUM_VALID_MANUFACTURER_FAMILIES
        ),
        "minimum_valid_t1_source_classes": (
            H04A_MINIMUM_VALID_T1_SOURCE_CLASSES
        ),
        "required_valid_t1_source_classes": list(
            H04A_REQUIRED_VALID_T1_SOURCE_CLASSES
        ),
        "wall_clock_stop_hours": H04A_WALL_CLOCK_STOP_HOURS,
        "wall_clock_stop_seconds": H04A_WALL_CLOCK_STOP_HOURS * 60 * 60,
        "storage_stop_gb": H04A_STORAGE_STOP_GB,
        "storage_stop_bytes": H04A_STORAGE_STOP_GB * DECIMAL_GB_BYTES,
        "tractography_authorized": False,
        "matrix_generation_authorized": False,
        "full_cohort_authorized": False,
    }
    for key, value in expected.items():
        if authorization.get(key) != value:
            raise PermissionError(
                f"signed H04A authorization differs at {key}: "
                f"{authorization.get(key)!r}"
            )
    for key in (
        "approved_by",
        "approved_utc",
        "user_response",
        "proposed_run_root",
        "normative_config_sha256",
        "workflow_source_manifest_sha256",
        "environment_contract_sha256",
    ):
        if not isinstance(authorization.get(key), str) or not authorization[key]:
            raise PermissionError(f"signed H04A authorization lacks {key}")
    return authorization


def require_h04a_mode_authorization(
    execution_binding: dict[str, Any], *, mode: str, requested_cores: int
) -> dict[str, Any]:
    """Authorize only the two bounded H04A modes and their signed core limit."""

    authorization = _validated_h04a_authorization(execution_binding)
    if mode not in H04A_AUTHORIZED_MODES:
        raise PermissionError(
            f"H04A cannot authorize launcher mode {mode!r}; phase-b requires "
            "a separate exact H04B continuation decision"
        )
    if isinstance(requested_cores, bool) or not isinstance(requested_cores, int):
        raise ValueError("--cores must be a positive integer")
    if requested_cores < 1:
        raise ValueError("--cores must be positive")
    maximum = int(authorization["maximum_cores"])
    if requested_cores > maximum:
        raise PermissionError(
            f"--cores {requested_cores} exceeds signed H04A maximum {maximum}"
        )
    return authorization


def measure_run_root_storage(run_root: Path) -> dict[str, Any]:
    """Measure run-root bytes without following symlinks or double-counting links."""

    root = run_root.expanduser().resolve()
    if not root.exists():
        return {
            "path": str(root),
            "logical_bytes": 0,
            "allocated_bytes": 0,
            "effective_bytes": 0,
            "inode_count": 0,
            "symlink_count": 0,
            "hardlinks_deduplicated": True,
            "symlinks_followed": False,
        }
    if not root.is_dir():
        raise ValueError(f"run root exists but is not a directory: {root}")

    logical_bytes = 0
    allocated_bytes = 0
    symlink_count = 0
    seen_inodes: set[tuple[int, int]] = set()
    pending = [root]
    while pending:
        current = pending.pop()
        try:
            current_stat = current.lstat()
        except FileNotFoundError:
            continue
        current_identity = (current_stat.st_dev, current_stat.st_ino)
        if current_identity not in seen_inodes:
            seen_inodes.add(current_identity)
            logical_bytes += int(current_stat.st_size)
            allocated_bytes += int(getattr(current_stat, "st_blocks", 0)) * 512
        with os.scandir(current) as entries:
            for entry in entries:
                try:
                    entry_stat = entry.stat(follow_symlinks=False)
                except FileNotFoundError:
                    continue
                identity = (entry_stat.st_dev, entry_stat.st_ino)
                if identity in seen_inodes:
                    continue
                seen_inodes.add(identity)
                logical_bytes += int(entry_stat.st_size)
                allocated_bytes += int(getattr(entry_stat, "st_blocks", 0)) * 512
                mode = entry_stat.st_mode
                if stat.S_ISLNK(mode):
                    symlink_count += 1
                elif stat.S_ISDIR(mode):
                    pending.append(Path(entry.path))
    return {
        "path": str(root),
        "logical_bytes": logical_bytes,
        "allocated_bytes": allocated_bytes,
        "effective_bytes": max(logical_bytes, allocated_bytes),
        "inode_count": len(seen_inodes),
        "symlink_count": symlink_count,
        "hardlinks_deduplicated": True,
        "symlinks_followed": False,
    }


def _nearest_existing_ancestor(path: Path) -> Path:
    candidate = path.expanduser().resolve()
    while not candidate.exists():
        if candidate.parent == candidate:
            raise FileNotFoundError(
                f"no existing filesystem ancestor for run root: {path}"
            )
        candidate = candidate.parent
    return candidate


def _h04a_decision_sha256(record: dict[str, Any]) -> str | None:
    value = (
        record.get("execution_binding", {})
        .get("execution_subset_decision", {})
        .get("sha256")
    )
    return str(value) if _valid_sha256(value) else None


def _prior_h04a_attempt_usage(
    attempts_dir: Path, *, decision_sha256: str
) -> tuple[float, list[dict[str, Any]]]:
    if not attempts_dir.exists():
        return 0.0, []
    if not attempts_dir.is_dir():
        raise ValueError(f"attempts path is not a directory: {attempts_dir}")

    duration_seconds = 0.0
    matched: list[dict[str, Any]] = []
    matched_end_paths: set[Path] = set()
    for start_path in sorted(attempts_dir.glob("*.start.json")):
        start = _load_json_mapping(start_path, label="prior H04A attempt start")
        if (
            start.get("launcher_mode") not in H04A_AUTHORIZED_MODES
            or _h04a_decision_sha256(start) != decision_sha256
        ):
            continue
        attempt_id = str(start.get("attempt_id", ""))
        if not attempt_id or start_path.name != f"{attempt_id}.start.json":
            raise ValueError(f"prior H04A attempt-start identity differs: {start_path}")
        end_path = attempts_dir / f"{attempt_id}.end.json"
        if not end_path.is_file():
            raise RuntimeError(
                "unclosed matching H04A attempt requires human resolution before "
                f"restart: {start_path}"
            )
        end = _load_json_mapping(end_path, label="prior H04A attempt end")
        if (
            end.get("attempt_id") != attempt_id
            or end.get("launcher_mode") != start.get("launcher_mode")
            or _h04a_decision_sha256(end) != decision_sha256
            or end.get("status") not in {"PASS", "FAIL"}
        ):
            raise ValueError(f"prior H04A attempt-end binding differs: {end_path}")
        raw_duration = end.get("duration_seconds")
        if isinstance(raw_duration, bool) or not isinstance(raw_duration, (int, float)):
            raise ValueError(f"prior H04A duration is invalid: {end_path}")
        observed_duration = float(raw_duration)
        if not math.isfinite(observed_duration) or observed_duration < 0:
            raise ValueError(f"prior H04A duration is invalid: {end_path}")
        duration_seconds += observed_duration
        matched_end_paths.add(end_path.resolve())
        matched.append(
            {
                "attempt_id": attempt_id,
                "launcher_mode": start["launcher_mode"],
                "status": end["status"],
                "duration_seconds": observed_duration,
                "attempt_start": file_record(start_path),
                "attempt_end": file_record(end_path),
            }
        )
    for end_path in sorted(attempts_dir.glob("*.end.json")):
        end = _load_json_mapping(end_path, label="prior attempt end")
        if (
            end.get("launcher_mode") in H04A_AUTHORIZED_MODES
            and _h04a_decision_sha256(end) == decision_sha256
            and end_path.resolve() not in matched_end_paths
        ):
            raise ValueError(
                f"matching H04A attempt end lacks its immutable start: {end_path}"
            )
    return duration_seconds, matched


def prepare_h04a_resource_preflight(
    *, run_root: Path, execution_binding: dict[str, Any]
) -> dict[str, Any]:
    """Fail closed before launch on cumulative H04A time, storage, or evidence."""

    authorization = _validated_h04a_authorization(execution_binding)
    decision_sha256 = execution_binding.get("execution_subset_decision", {}).get(
        "sha256"
    )
    if not _valid_sha256(decision_sha256):
        raise ValueError("H04A execution decision SHA-256 is invalid")
    prior_duration, prior_attempts = _prior_h04a_attempt_usage(
        run_root / "attempts", decision_sha256=str(decision_sha256)
    )
    wall_limit = float(authorization["wall_clock_stop_seconds"])
    if prior_duration >= wall_limit:
        raise RuntimeError(
            "cumulative H04A wall-clock stop has already been reached: "
            f"{prior_duration:.3f} >= {wall_limit:.3f} seconds"
        )
    storage = measure_run_root_storage(run_root)
    storage_limit = int(authorization["storage_stop_bytes"])
    if storage["effective_bytes"] >= storage_limit:
        raise RuntimeError(
            "H04A storage stop has already been reached: "
            f"{storage['effective_bytes']} >= {storage_limit} bytes"
        )
    filesystem_path = _nearest_existing_ancestor(run_root)
    disk = shutil.disk_usage(filesystem_path)
    storage_remaining = storage_limit - int(storage["effective_bytes"])
    if disk.free < storage_remaining:
        raise RuntimeError(
            "insufficient free space to honor the signed H04A storage envelope: "
            f"free={disk.free}, required_remaining={storage_remaining}"
        )
    return {
        "schema_version": "1.0.0",
        "status": "PASS",
        "checked_utc": utc_now(),
        "decision_sha256": str(decision_sha256),
        "prior_attempts": prior_attempts,
        "prior_duration_seconds": prior_duration,
        "wall_clock_stop_seconds": wall_limit,
        "wall_clock_remaining_seconds": wall_limit - prior_duration,
        "storage_stop_bytes": storage_limit,
        "storage_remaining_bytes": storage_remaining,
        "storage": storage,
        "filesystem": {
            "probe_path": str(filesystem_path),
            "total_bytes": disk.total,
            "used_bytes": disk.used,
            "free_bytes": disk.free,
        },
    }


def _terminate_process_group(
    process: subprocess.Popen[str], *, grace_seconds: float
) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + max(0.0, grace_seconds)
    while process.poll() is None and time.monotonic() < deadline:
        time.sleep(0.1)
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def execute_command_with_h04a_monitor(
    *,
    command: list[str],
    cwd: Path,
    execution_log_path: Path,
    run_root: Path,
    resource_preflight: dict[str, Any] | None,
    monitor_interval_seconds: float = RESOURCE_MONITOR_INTERVAL_SECONDS,
    termination_grace_seconds: float = RESOURCE_TERMINATION_GRACE_SECONDS,
) -> tuple[int, dt.datetime, dt.datetime, dict[str, Any] | None]:
    """Run one command and terminate its process group on an H04A cap breach."""

    if monitor_interval_seconds <= 0:
        raise ValueError("resource monitor interval must be positive")
    violation: dict[str, Any] | None = None
    violation_lock = threading.Lock()
    stop_monitor = threading.Event()
    started = dt.datetime.now(dt.timezone.utc)
    started_monotonic = time.monotonic()

    with execution_log_path.open("x", encoding="utf-8", buffering=1) as log_handle:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )

        def set_violation(record: dict[str, Any]) -> None:
            nonlocal violation
            with violation_lock:
                if violation is None:
                    violation = record

        def monitor() -> None:
            if resource_preflight is None:
                return
            wall_limit = float(resource_preflight["wall_clock_stop_seconds"])
            prior_duration = float(resource_preflight["prior_duration_seconds"])
            storage_limit = int(resource_preflight["storage_stop_bytes"])
            while not stop_monitor.wait(monitor_interval_seconds):
                elapsed = time.monotonic() - started_monotonic
                cumulative = prior_duration + elapsed
                if cumulative >= wall_limit:
                    set_violation(
                        {
                            "type": "wall_clock_stop",
                            "detected_utc": utc_now(),
                            "cumulative_seconds": cumulative,
                            "limit_seconds": wall_limit,
                        }
                    )
                    _terminate_process_group(
                        process, grace_seconds=termination_grace_seconds
                    )
                    return
                try:
                    storage = measure_run_root_storage(run_root)
                except Exception as exc:
                    set_violation(
                        {
                            "type": "storage_measurement_failure",
                            "detected_utc": utc_now(),
                            "error": f"{type(exc).__name__}:{exc}",
                        }
                    )
                    _terminate_process_group(
                        process, grace_seconds=termination_grace_seconds
                    )
                    return
                if storage["effective_bytes"] >= storage_limit:
                    set_violation(
                        {
                            "type": "storage_stop",
                            "detected_utc": utc_now(),
                            "effective_bytes": storage["effective_bytes"],
                            "limit_bytes": storage_limit,
                            "storage": storage,
                        }
                    )
                    _terminate_process_group(
                        process, grace_seconds=termination_grace_seconds
                    )
                    return

        monitor_thread = threading.Thread(
            target=monitor, name="h04a-resource-monitor", daemon=True
        )
        monitor_thread.start()
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            log_handle.write(line)
        process.stdout.close()
        returncode = process.wait()
        stop_monitor.set()
        monitor_thread.join(timeout=max(1.0, termination_grace_seconds + 1.0))
        if monitor_thread.is_alive():
            set_violation(
                {
                    "type": "resource_monitor_shutdown_failure",
                    "detected_utc": utc_now(),
                }
            )
        log_handle.flush()
        os.fsync(log_handle.fileno())
    ended = dt.datetime.now(dt.timezone.utc)

    runtime_record: dict[str, Any] | None = None
    if resource_preflight is not None:
        try:
            post_storage = measure_run_root_storage(run_root)
        except Exception as exc:
            post_storage = None
            set_violation(
                {
                    "type": "postflight_storage_measurement_failure",
                    "detected_utc": utc_now(),
                    "error": f"{type(exc).__name__}:{exc}",
                }
            )
        if (
            post_storage is not None
            and post_storage["effective_bytes"]
            >= int(resource_preflight["storage_stop_bytes"])
        ):
            set_violation(
                {
                    "type": "postflight_storage_stop",
                    "detected_utc": utc_now(),
                    "effective_bytes": post_storage["effective_bytes"],
                    "limit_bytes": resource_preflight["storage_stop_bytes"],
                    "storage": post_storage,
                }
            )
        elapsed = time.monotonic() - started_monotonic
        cumulative = float(resource_preflight["prior_duration_seconds"]) + elapsed
        if cumulative >= float(resource_preflight["wall_clock_stop_seconds"]):
            set_violation(
                {
                    "type": "postflight_wall_clock_stop",
                    "detected_utc": utc_now(),
                    "cumulative_seconds": cumulative,
                    "limit_seconds": resource_preflight["wall_clock_stop_seconds"],
                }
            )
        runtime_record = {
            "schema_version": "1.0.0",
            "monitor_interval_seconds": monitor_interval_seconds,
            "termination_grace_seconds": termination_grace_seconds,
            "elapsed_seconds": elapsed,
            "cumulative_seconds": cumulative,
            "postflight_storage": post_storage,
            "violation": violation,
            "automatic_process_group_termination": True,
        }
        if violation is not None:
            returncode = 124
    return returncode, started, ended, runtime_record


def derive_lineage_id(
    *,
    recipe_id: str,
    acquisition_manifest_sha256: str,
    execution_binding: dict[str, Any],
    run_root: Path,
) -> str:
    """Derive the stable cohort lineage shared by phase A and phase B."""

    if not _valid_sha256(acquisition_manifest_sha256):
        raise ValueError("acquisition-manifest SHA-256 is invalid")
    if execution_binding.get("binding_type") != "connectome_execution_subset_binding":
        raise ValueError("lineage requires an immutable execution-subset binding")
    identity = {
        "recipe_id": recipe_id,
        "parent_acquisition_manifest_sha256": acquisition_manifest_sha256,
        "execution_subset_manifest_sha256": execution_binding[
            "execution_subset_manifest"
        ]["sha256"],
        "execution_subset_decision_sha256": execution_binding[
            "execution_subset_decision"
        ]["sha256"],
        "run_root": str(run_root.resolve()),
    }
    material = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:24]


def derive_mode_run_id(
    *,
    lineage_id: str,
    mode: str,
    phase_b_binding: dict[str, Any] | None = None,
    continuation_binding: dict[str, Any] | None = None,
) -> str:
    """Bind run identity to the immutable evidence consumed by that phase."""

    if not re.fullmatch(r"[a-f0-9]{24}", lineage_id):
        raise ValueError("lineage_id must be 24 lowercase hexadecimal characters")
    if mode not in RUN_MODES:
        raise ValueError(f"unsupported launcher mode: {mode!r}")
    identity: dict[str, Any] = {"lineage_id": lineage_id, "mode": mode}
    if mode in {"pre-tractography-canary", "phase-b"}:
        if phase_b_binding is None:
            raise ValueError(
                f"{mode} run identity requires an immutable calibration binding"
            )
        identity["phase_a_completion_sha256"] = phase_b_binding[
            "phase_a_completion"
        ]["sha256"]
        identity["response_calibration_manifest_sha256"] = phase_b_binding[
            "response_calibration_manifest"
        ]["sha256"]
        identity["minimum_valid_subjects"] = phase_b_binding[
            "minimum_valid_subjects"
        ]
        identity["response_calibration_decision_sha256"] = phase_b_binding[
            "response_calibration_decision"
        ]["sha256"]
    if mode == "phase-b":
        if continuation_binding is None:
            raise ValueError(
                "phase-b run identity requires an immutable tractography continuation binding"
            )
        identity["pre_tractography_completion_sha256"] = continuation_binding[
            "pre_tractography_completion"
        ]["sha256"]
        identity["human_qc_manifest_sha256"] = continuation_binding[
            "human_qc_manifest"
        ]["sha256"]
        identity["continuation_decision_sha256"] = continuation_binding[
            "continuation_decision"
        ]["sha256"]
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def _load_json_mapping(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} missing: {path}")
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid UTF-8 JSON: {path}") from exc
    if not isinstance(record, dict):
        raise TypeError(f"{label} JSON root must be an object: {path}")
    return record


def _verified_nested_file_record(
    record: Any, *, label: str
) -> tuple[Path, dict[str, Any]]:
    if not isinstance(record, dict):
        raise TypeError(f"{label} is not a file record")
    errors = verify_file_record(record)
    if errors:
        raise ValueError(f"{label} is not exact immutable evidence: {'; '.join(errors)}")
    path = Path(str(record["path"])).expanduser().resolve()
    return path, dict(record)


def prepare_phase_b_binding(
    *,
    recipe_id: str,
    phase_a_completion_path: Path,
    response_calibration_manifest_path: Path,
    minimum_valid_subjects: int,
    response_calibration_decision_path: Path,
    execution_binding: dict[str, Any],
) -> dict[str, Any]:
    """Validate and bind the exact diagnosis-blind phase-A-to-B handoff."""

    if (
        not isinstance(minimum_valid_subjects, int)
        or isinstance(minimum_valid_subjects, bool)
        or minimum_valid_subjects < 1
    ):
        raise ValueError("phase-b response-calibration minimum must be a positive integer")
    signed_minimum = execution_binding.get("h04a_authorization", {}).get(
        "minimum_valid_response_calibration_units"
    )
    if minimum_valid_subjects != signed_minimum:
        raise ValueError(
            "phase-b response-calibration minimum differs from signed H04A minimum"
        )

    completion_path = phase_a_completion_path.expanduser().resolve()
    completion = _load_json_mapping(completion_path, label="phase-A completion")
    completion_expected = {
        "schema_version": "2.0.0",
        "record_type": "response_calibration_phase_a_completion",
        "mode": "response-calibration-phase-a",
        "recipe_id": recipe_id,
        "execution_binding": execution_binding,
        "freeze_required_before_phase_b": True,
        "full_run_terminal_publication": False,
    }
    for key, value in completion_expected.items():
        if completion.get(key) != value:
            raise ValueError(
                f"phase-A completion differs at {key}: {completion.get(key)!r}"
            )
    if completion.get("status") not in {"PASS", "PARTIAL"}:
        raise ValueError("phase-A completion must be PASS or PARTIAL before phase B")
    summary = completion.get("phase_a_summary")
    if not isinstance(summary, dict):
        raise ValueError("phase-A completion has no phase_a_summary")
    try:
        expected_count = int(summary["expected"])
        terminal_count = int(summary["terminal"])
        pass_count = int(summary["pass"])
        fail_count = int(summary["fail"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("phase-A completion summary is malformed") from exc
    if (
        expected_count < 1
        or expected_count != terminal_count
        or pass_count + fail_count != terminal_count
    ):
        raise ValueError("phase-A completion does not prove expected=terminal")

    phase_a_manifest_path, phase_a_manifest_record = _verified_nested_file_record(
        completion.get("phase_a_manifest"), label="phase-A manifest"
    )
    phase_a_manifest = _load_json_mapping(
        phase_a_manifest_path, label="phase-A manifest"
    )
    manifest_expected = {
        "schema_version": "2.0.0",
        "record_type": "response_calibration_phase_a_manifest",
        "status": "COMPLETE",
        "mode": "response-calibration-phase-a",
        "run_id": completion.get("run_id"),
        "lineage_id": completion.get("lineage_id"),
        "recipe_id": recipe_id,
        "execution_binding": execution_binding,
        "freeze_required_before_phase_b": True,
        "full_run_terminal_publication": False,
        "dummy_response_files_used": False,
    }
    for key, value in manifest_expected.items():
        if phase_a_manifest.get(key) != value:
            raise ValueError(f"phase-A manifest differs at {key}")
    if phase_a_manifest.get("summary") != summary:
        raise ValueError("phase-A manifest and completion summaries differ")
    phase_a_run_context_path, phase_a_run_context_record = _verified_nested_file_record(
        phase_a_manifest.get("run_context"), label="phase-A run context"
    )
    phase_a_attempt_path, phase_a_attempt_record = _verified_nested_file_record(
        phase_a_manifest.get("attempt_start"), label="phase-A attempt start"
    )
    if completion.get("run_context") != phase_a_run_context_record:
        raise ValueError("phase-A completion and manifest run contexts differ")
    if completion.get("terminal_attempt_start") != phase_a_attempt_record:
        raise ValueError("phase-A completion and manifest attempt starts differ")
    phase_a_attempt = _load_json_mapping(
        phase_a_attempt_path, label="phase-A attempt start"
    )
    if (
        phase_a_attempt.get("run_id") != completion.get("run_id")
        or phase_a_attempt.get("recipe_id") != recipe_id
        or phase_a_attempt.get("launcher_mode") != "response-calibration-phase-a"
        or phase_a_attempt.get("execution_binding") != execution_binding
        or phase_a_attempt.get("run_context") != phase_a_run_context_record
        or not isinstance(phase_a_attempt.get("attempt_id"), str)
        or not phase_a_attempt["attempt_id"]
    ):
        raise ValueError("phase-A attempt identity differs from completion")
    phase_a_run_context = _load_json_mapping(
        phase_a_run_context_path, label="phase-A run context"
    )
    phase_a_context_expected = {
        "schema_version": "1.0.0",
        "status": "LOCKED",
        "run_id": completion.get("run_id"),
        "lineage_id": completion.get("lineage_id"),
        "launcher_mode": "response-calibration-phase-a",
        "recipe_id": recipe_id,
        "normative_config": file_record(NORMATIVE_CONFIG),
        "environment_contract": file_record(ENVIRONMENT_CONTRACT),
    }
    for key, value in phase_a_context_expected.items():
        if phase_a_run_context.get(key) != value:
            raise ValueError(f"phase-A run context differs at {key}")
    if phase_a_run_context.get("execution_binding") != execution_binding:
        raise ValueError("phase-A and current execution-subset bindings differ")
    environment = load_environment()
    if phase_a_run_context.get("workflow_source_manifest") != file_record(
        environment["workflow"]["source_manifest"]["path"]
    ):
        raise ValueError("phase-A workflow source record differs from current recipe")
    if phase_a_attempt.get("lineage_id") != completion.get("lineage_id"):
        raise ValueError("phase-A attempt lineage differs from completion")
    expected_units = phase_a_manifest.get("expected_units")
    outcome_records = phase_a_manifest.get("outcome_records")
    if (
        not isinstance(expected_units, list)
        or any(not isinstance(unit, str) or not unit for unit in expected_units)
        or len(expected_units) != len(set(expected_units))
        or not isinstance(outcome_records, dict)
        or set(outcome_records) != set(expected_units)
    ):
        raise ValueError("phase-A manifest lacks the exact complete unit outcome set")
    if (
        expected_units != execution_binding.get("units")
        or expected_count != execution_binding.get("approved_unit_count")
    ):
        raise ValueError("phase-A outcome denominator differs from approved canary subset")

    observed_status: dict[str, str] = {}
    observed_outcomes: dict[str, dict[str, Any]] = {}
    for unit in expected_units:
        outcome_path, outcome_file_record = _verified_nested_file_record(
            outcome_records[unit], label=f"phase-A outcome {unit}"
        )
        outcome = _load_json_mapping(outcome_path, label=f"phase-A outcome {unit}")
        _validate_response_calibration_outcome(outcome, unit=unit)
        if outcome.get("recipe_id") != recipe_id:
            raise ValueError(f"phase-A outcome recipe differs for {unit}")
        identity = outcome["phase_a_identity"]
        if (
            identity.get("run_id") != completion.get("run_id")
            or identity.get("attempt_id") != phase_a_attempt["attempt_id"]
            or identity.get("run_context") != phase_a_run_context_record
            or identity.get("attempt_context") != phase_a_attempt_record
        ):
            raise ValueError(f"phase-A outcome run/attempt identity differs for {unit}")
        observed_status[unit] = str(outcome["status"])
        observed_outcomes[unit] = {
            "record": outcome_file_record,
            "payload": outcome,
        }
    if sum(status == "PASS" for status in observed_status.values()) != pass_count:
        raise ValueError("phase-A PASS count differs from exact outcome records")
    if sum(status == "FAIL" for status in observed_status.values()) != fail_count:
        raise ValueError("phase-A FAIL count differs from exact outcome records")

    calibration_path = response_calibration_manifest_path.expanduser().resolve()
    calibration_record = _load_json_mapping(
        calibration_path, label="frozen response-calibration manifest"
    )
    calibration_file_record = file_record(calibration_path)
    if calibration_record.get("record_type") != "frozen_response_calibration":
        raise ValueError("frozen response-calibration record type differs")
    if calibration_record.get("status") != "PASS":
        raise ValueError("frozen response calibration is not PASS evidence")
    if calibration_record.get("diagnosis_labels_used") is not False:
        raise ValueError("frozen response calibration is not diagnosis blind")
    if int(calibration_record.get("minimum_valid_subjects", 0)) != minimum_valid_subjects:
        raise ValueError("frozen response minimum differs from phase-B CLI")
    if int(calibration_record.get("total_outcome_count", -1)) != expected_count:
        raise ValueError("frozen response total differs from phase-A denominator")
    if int(calibration_record.get("valid_subject_count", -1)) != pass_count:
        raise ValueError("frozen response valid count differs from phase-A PASS count")
    if int(calibration_record.get("invalid_subject_count", -1)) != fail_count:
        raise ValueError("frozen response invalid count differs from phase-A FAIL count")
    valid_units = calibration_record.get("valid_units")
    invalid_units = calibration_record.get("invalid_units")
    expected_valid = sorted(
        unit for unit, status in observed_status.items() if status == "PASS"
    )
    expected_invalid = sorted(
        unit for unit, status in observed_status.items() if status == "FAIL"
    )
    if valid_units != expected_valid or invalid_units != expected_invalid:
        raise ValueError("frozen response unit membership differs from phase-A outcomes")

    diversity_contract = technical_diversity_contract_from_authorization(
        execution_binding.get("h04a_authorization", {})
    )
    subset_manifest_record, unit_metadata = (
        load_execution_subset_technical_metadata(execution_binding)
    )
    expected_technical_diversity = build_valid_pool_technical_diversity(
        valid_units=expected_valid,
        unit_metadata=unit_metadata,
        execution_subset_manifest=subset_manifest_record,
        **diversity_contract,
    )
    if (
        calibration_record.get("valid_pool_technical_diversity")
        != expected_technical_diversity
        or calibration_record.get("valid_pool_technical_diversity_sha256")
        != technical_diversity_sha256(expected_technical_diversity)
    ):
        raise ValueError(
            "frozen response valid-pool technical diversity differs from exact "
            "phase-A PASS units and bound acquisition metadata"
        )

    phase_a_lineage = calibration_record.get("phase_a_lineage")
    expected_lineage = {
        "run_id": completion.get("run_id"),
        "recipe_id": recipe_id,
        "completion": file_record(completion_path),
        "manifest": phase_a_manifest_record,
        "attempt_start": phase_a_attempt_record,
        "expected_outcome_count": expected_count,
        "terminal_outcome_count": terminal_count,
    }
    if phase_a_lineage != expected_lineage:
        raise ValueError("frozen response calibration does not bind exact phase-A lineage")

    frozen_inputs = calibration_record.get("inputs")
    if not isinstance(frozen_inputs, dict) or set(frozen_inputs) != set(expected_valid):
        raise ValueError("frozen response inputs differ from exact phase-A PASS units")
    for unit in expected_valid:
        phase_a_responses = observed_outcomes[unit]["payload"]["responses"]
        frozen_unit = frozen_inputs.get(unit)
        if not isinstance(frozen_unit, dict) or set(frozen_unit) != set(TISSUES):
            raise ValueError(f"frozen response inputs are incomplete for {unit}")
        for tissue in TISSUES:
            phase_artifact = phase_a_responses[tissue]
            frozen_artifact = frozen_unit[tissue]
            if (
                Path(str(frozen_artifact.get("path", ""))).resolve()
                != Path(str(phase_artifact.get("path", ""))).resolve()
                or frozen_artifact.get("sha256") != phase_artifact.get("sha256")
            ):
                raise ValueError(
                    f"frozen {tissue} input differs from phase-A outcome for {unit}"
                )

    pooled_records = {
        tissue: file_record(
            Path(str(calibration_record.get("pooled_responses", {}).get(tissue, {}).get("path", "")))
        )
        for tissue in TISSUES
    }
    validate_frozen_response_calibration(
        calibration_path,
        expected_manifest_sha256=calibration_file_record["sha256"],
        minimum_valid_subjects=minimum_valid_subjects,
        expected_responses=pooled_records,
        expected_technical_diversity=expected_technical_diversity,
    )

    decision_path = response_calibration_decision_path.expanduser().resolve()
    decision = _load_json_mapping(
        decision_path, label="phase-B response-calibration decision"
    )
    decision_expected = {
        "schema_version": "2.0.0",
        "decision_type": "response_calibration_phase_b_approval",
        "status": "APPROVED",
        "recipe_id": recipe_id,
        "phase_a_completion_sha256": sha256_file(completion_path),
        "response_calibration_manifest_sha256": calibration_file_record["sha256"],
        "minimum_valid_subjects": minimum_valid_subjects,
        **diversity_contract,
        "valid_unit_count": len(expected_valid),
        "valid_manufacturer_counts": expected_technical_diversity[
            "manufacturer_counts"
        ],
        "valid_manufacturer_family_counts": expected_technical_diversity[
            "manufacturer_family_counts"
        ],
        "valid_t1_source_class_counts": expected_technical_diversity[
            "t1_source_class_counts"
        ],
        "valid_pool_technical_diversity_sha256": (
            technical_diversity_sha256(expected_technical_diversity)
        ),
    }
    for key, value in decision_expected.items():
        if decision.get(key) != value:
            raise ValueError(
                f"phase-B response-calibration decision differs at {key}: "
                f"{decision.get(key)!r}"
            )

    return {
        "schema_version": "2.0.0",
        "binding_type": "response_calibration_phase_b_binding",
        "recipe_id": recipe_id,
        "phase_a_run_id": completion["run_id"],
        "phase_a_completion": file_record(completion_path),
        "phase_a_manifest": phase_a_manifest_record,
        "response_calibration_manifest": calibration_file_record,
        "minimum_valid_subjects": minimum_valid_subjects,
        "valid_units": expected_valid,
        "valid_pool_technical_diversity": expected_technical_diversity,
        "valid_pool_technical_diversity_sha256": (
            technical_diversity_sha256(expected_technical_diversity)
        ),
        "response_calibration_decision": file_record(decision_path),
        "pooled_responses": pooled_records,
        "diagnosis_labels_used": False,
        "phase_b_live_all_units_dependency": False,
        "dummy_response_files_used": False,
    }


def inject_phase_b_calibration(
    normative_config: dict[str, Any], phase_b_binding: dict[str, Any]
) -> dict[str, Any]:
    """Return an attempt-local resolved config; never mutate normative config."""

    resolved = copy.deepcopy(normative_config)
    calibration = resolved["fod"]["response_estimation"]["calibration"]
    calibration["minimum_valid_subjects"] = phase_b_binding[
        "minimum_valid_subjects"
    ]
    calibration["frozen_manifest"] = dict(
        phase_b_binding["response_calibration_manifest"]
    )
    calibration["pooled_responses"] = {
        tissue: dict(phase_b_binding["pooled_responses"][tissue])
        for tissue in TISSUES
    }
    resolved["response_calibration_binding"] = copy.deepcopy(phase_b_binding)
    return resolved


def validate_response_calibration_diversity_binding(
    *,
    execution_binding: dict[str, Any],
    response_calibration_binding: dict[str, Any],
) -> dict[str, Any]:
    """Recompute and reverify valid-pool diversity from exact bound metadata."""

    diversity_contract = technical_diversity_contract_from_authorization(
        execution_binding.get("h04a_authorization", {})
    )
    if (
        response_calibration_binding.get("recipe_id")
        != execution_binding.get("recipe_id")
    ):
        raise ValueError("response-calibration and execution recipe bindings differ")
    subset_manifest_record, unit_metadata = (
        load_execution_subset_technical_metadata(execution_binding)
    )
    valid_units = response_calibration_binding.get("valid_units")
    if (
        not isinstance(valid_units, list)
        or not valid_units
        or valid_units != sorted(set(valid_units))
        or not set(valid_units).issubset(set(execution_binding.get("units", [])))
    ):
        raise ValueError("response-calibration binding lacks exact valid units")
    minimum_valid_subjects = response_calibration_binding.get(
        "minimum_valid_subjects"
    )
    if (
        isinstance(minimum_valid_subjects, bool)
        or not isinstance(minimum_valid_subjects, int)
        or minimum_valid_subjects
        != execution_binding.get("h04a_authorization", {}).get(
            "minimum_valid_response_calibration_units"
        )
        or len(valid_units) < minimum_valid_subjects
    ):
        raise ValueError(
            "response-calibration binding differs from signed valid-unit minimum"
        )
    expected_diversity = build_valid_pool_technical_diversity(
        valid_units=valid_units,
        unit_metadata=unit_metadata,
        execution_subset_manifest=subset_manifest_record,
        **diversity_contract,
    )
    expected_digest = technical_diversity_sha256(expected_diversity)
    if (
        response_calibration_binding.get("valid_pool_technical_diversity")
        != expected_diversity
        or response_calibration_binding.get(
            "valid_pool_technical_diversity_sha256"
        )
        != expected_digest
    ):
        raise ValueError(
            "response-calibration binding technical diversity differs from "
            "bound valid-unit acquisition metadata"
        )

    manifest_path, manifest_record = _verified_nested_file_record(
        response_calibration_binding.get("response_calibration_manifest"),
        label="response-calibration manifest",
    )
    manifest = _load_json_mapping(
        manifest_path, label="response-calibration manifest"
    )
    if (
        manifest.get("valid_units") != valid_units
        or manifest.get("valid_pool_technical_diversity") != expected_diversity
        or manifest.get("valid_pool_technical_diversity_sha256")
        != expected_digest
    ):
        raise ValueError("response-calibration manifest diversity differs at continuation")
    pooled_records = response_calibration_binding.get("pooled_responses")
    if not isinstance(pooled_records, dict) or set(pooled_records) != set(TISSUES):
        raise ValueError("response-calibration binding lacks exact pooled responses")
    validate_frozen_response_calibration(
        manifest_path,
        expected_manifest_sha256=manifest_record["sha256"],
        minimum_valid_subjects=int(
            response_calibration_binding.get("minimum_valid_subjects", 0)
        ),
        expected_responses=pooled_records,
        expected_technical_diversity=expected_diversity,
    )

    _, phase_a_completion_record = _verified_nested_file_record(
        response_calibration_binding.get("phase_a_completion"),
        label="phase-A completion",
    )
    decision_path, _ = _verified_nested_file_record(
        response_calibration_binding.get("response_calibration_decision"),
        label="response-calibration approval decision",
    )
    decision = _load_json_mapping(
        decision_path, label="response-calibration approval decision"
    )
    decision_expected = {
        "schema_version": "2.0.0",
        "decision_type": "response_calibration_phase_b_approval",
        "status": "APPROVED",
        "recipe_id": response_calibration_binding.get("recipe_id"),
        "phase_a_completion_sha256": phase_a_completion_record["sha256"],
        "response_calibration_manifest_sha256": manifest_record["sha256"],
        "minimum_valid_subjects": response_calibration_binding.get(
            "minimum_valid_subjects"
        ),
        **diversity_contract,
        "valid_unit_count": len(valid_units),
        "valid_manufacturer_counts": expected_diversity["manufacturer_counts"],
        "valid_manufacturer_family_counts": expected_diversity[
            "manufacturer_family_counts"
        ],
        "valid_t1_source_class_counts": expected_diversity[
            "t1_source_class_counts"
        ],
        "valid_pool_technical_diversity_sha256": expected_digest,
    }
    for key, value in decision_expected.items():
        if decision.get(key) != value:
            raise ValueError(
                f"response-calibration decision diversity differs at {key}"
            )
    return expected_diversity


def prepare_tractography_continuation_binding(
    *,
    recipe_id: str,
    execution_binding: dict[str, Any],
    response_calibration_binding: dict[str, Any],
    pre_tractography_completion_path: Path,
    human_qc_manifest_path: Path,
    continuation_decision_path: Path,
) -> dict[str, Any]:
    """Validate H04B review evidence before any tractography is scheduled."""

    valid_pool_technical_diversity = (
        validate_response_calibration_diversity_binding(
            execution_binding=execution_binding,
            response_calibration_binding=response_calibration_binding,
        )
    )
    diversity_contract = technical_diversity_contract_from_authorization(
        execution_binding.get("h04a_authorization", {})
    )
    diversity_digest = technical_diversity_sha256(
        valid_pool_technical_diversity
    )

    completion_path = pre_tractography_completion_path.expanduser().resolve()
    completion = _load_json_mapping(
        completion_path, label="pre-tractography canary completion"
    )
    completion_expected = {
        "schema_version": "2.0.0",
        "record_type": "pre_tractography_canary_completion",
        "status": "AWAITING_HUMAN_QC",
        "mode": "pre-tractography-canary",
        "recipe_id": recipe_id,
        "execution_binding": execution_binding,
        "response_calibration_binding": response_calibration_binding,
        "full_run_terminal_publication": False,
        "tractography_started": False,
    }
    for key, value in completion_expected.items():
        if completion.get(key) != value:
            raise ValueError(f"pre-tractography completion differs at {key}")
    if completion.get("run_outcome") not in {"PASS", "PARTIAL"}:
        raise ValueError("pre-tractography canary has no reviewable continuation")
    pre_manifest_path, pre_manifest_record = _verified_nested_file_record(
        completion.get("pre_tractography_manifest"),
        label="pre-tractography canary manifest",
    )
    pre_manifest = _load_json_mapping(
        pre_manifest_path, label="pre-tractography canary manifest"
    )
    if (
        pre_manifest.get("schema_version") != "2.0.0"
        or pre_manifest.get("record_type")
        != "pre_tractography_canary_manifest"
        or pre_manifest.get("status") != "COMPLETE"
        or pre_manifest.get("workflow_state") != "AWAITING_HUMAN_QC"
        or pre_manifest.get("recipe_id") != recipe_id
        or pre_manifest.get("execution_binding") != execution_binding
        or pre_manifest.get("summary") != completion.get("pre_tractography_summary")
    ):
        raise ValueError("pre-tractography manifest/completion contract differs")
    expected_units = pre_manifest.get("expected_units")
    ready_units = pre_manifest.get("ready_units")
    review_records = pre_manifest.get("review_bundle_records")
    automated_qc_records = pre_manifest.get(
        "automated_pre_tractography_qc_records"
    )
    if (
        expected_units != execution_binding.get("units")
        or not isinstance(ready_units, list)
        or not ready_units
        or len(ready_units) != len(set(ready_units))
        or not set(ready_units).issubset(set(expected_units))
        or not isinstance(review_records, dict)
        or set(review_records) != set(ready_units)
        or not isinstance(automated_qc_records, dict)
        or set(automated_qc_records) != set(ready_units)
        or completion.get("automated_pre_tractography_qc_records")
        != automated_qc_records
    ):
        raise ValueError("pre-tractography reviewable unit set is not exactly closed")
    for unit in ready_units:
        artifacts = review_records[unit]
        if not isinstance(artifacts, dict) or set(artifacts) != {
            "b0_t1",
            "b0_5tt",
            "b0_atlas",
            "index",
        }:
            raise ValueError(f"pre-tractography review bundle is incomplete for {unit}")
        for name, record in artifacts.items():
            _verified_nested_file_record(
                record, label=f"pre-tractography {unit} {name}"
            )
        automated_path, _ = _verified_nested_file_record(
            automated_qc_records[unit],
            label=f"pre-tractography automated QC for {unit}",
        )
        automated_qc = _load_json_mapping(
            automated_path,
            label=f"pre-tractography automated QC for {unit}",
        )
        if (
            automated_qc.get("schema_version") != "2.0.0"
            or automated_qc.get("record_type")
            != "automated_pre_tractography_qc"
            or automated_qc.get("status") != "PASS"
            or automated_qc.get("unit") != unit
            or automated_qc.get("diagnosis_labels_used") is not False
            or not isinstance(automated_qc.get("image_ranges"), dict)
            or not isinstance(automated_qc.get("wmfod_qc"), dict)
            or not isinstance(automated_qc.get("commands"), list)
            or automated_qc.get("failures") != []
            or {
                "diagnosis",
                "diagnosis_at_dti",
                "group",
                "research_group",
            }.intersection(
                str(key).strip().lower() for key in automated_qc
            )
        ):
            raise ValueError(
                f"pre-tractography automated QC differs for {unit}"
            )

    human_qc_path = human_qc_manifest_path.expanduser().resolve()
    if not human_qc_path.is_file():
        raise FileNotFoundError(f"human QC manifest missing: {human_qc_path}")
    with human_qc_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = tuple(reader.fieldnames or ())
        forbidden = {"diagnosis", "diagnosis_at_dti", "group", "research_group"}
        if forbidden.intersection(str(field).strip().lower() for field in fields):
            raise ValueError("human visual-QC manifest must remain diagnosis blinded")
        required = {"unit", "status", "reviewer", "reviewed_utc"}
        if not required.issubset(fields):
            raise ValueError("human visual-QC manifest lacks required review fields")
        review_rows = [
            {str(key): str(value or "").strip() for key, value in row.items()}
            for row in reader
        ]
    reviewed_units = [row["unit"] for row in review_rows]
    if (
        len(reviewed_units) != len(set(reviewed_units))
        or set(reviewed_units) != set(ready_units)
    ):
        raise ValueError("human visual-QC rows differ from exact reviewable unit set")
    for row in review_rows:
        if row["status"].upper() != "PASS":
            raise ValueError(f"human visual QC is not PASS for {row['unit']}")
        if not row["reviewer"] or not row["reviewed_utc"]:
            raise ValueError(f"human visual-QC identity/time is blank for {row['unit']}")

    human_qc_record = file_record(human_qc_path)
    decision_path = continuation_decision_path.expanduser().resolve()
    decision = _load_json_mapping(
        decision_path, label="tractography continuation decision"
    )
    decision_expected = {
        "schema_version": "2.0.0",
        "decision_type": "connectome_canary_tractography_continuation_approval",
        "status": "APPROVED",
        "execution_scope": "canary",
        "recipe_id": recipe_id,
        "execution_subset_manifest_sha256": execution_binding[
            "execution_subset_manifest"
        ]["sha256"],
        "response_calibration_manifest_sha256": response_calibration_binding[
            "response_calibration_manifest"
        ]["sha256"],
        "pre_tractography_completion_sha256": sha256_file(completion_path),
        "human_qc_manifest_sha256": human_qc_record["sha256"],
        **diversity_contract,
        "valid_unit_count": valid_pool_technical_diversity["valid_unit_count"],
        "valid_manufacturer_counts": valid_pool_technical_diversity[
            "manufacturer_counts"
        ],
        "valid_manufacturer_family_counts": valid_pool_technical_diversity[
            "manufacturer_family_counts"
        ],
        "valid_t1_source_class_counts": valid_pool_technical_diversity[
            "t1_source_class_counts"
        ],
        "valid_pool_technical_diversity_sha256": diversity_digest,
        "approved_unit_count": len(ready_units),
        "diagnosis_labels_used": False,
        "all_reviewed_units_pass": True,
    }
    for key, value in decision_expected.items():
        if decision.get(key) != value:
            raise ValueError(
                f"tractography continuation decision differs at {key}: "
                f"{decision.get(key)!r}"
            )
    return {
        "schema_version": "2.0.0",
        "binding_type": "tractography_continuation_binding",
        "execution_scope": "canary",
        "recipe_id": recipe_id,
        "execution_subset_manifest": dict(
            execution_binding["execution_subset_manifest"]
        ),
        "pre_tractography_completion": file_record(completion_path),
        "pre_tractography_manifest": pre_manifest_record,
        "automated_pre_tractography_qc_records": copy.deepcopy(
            automated_qc_records
        ),
        "human_qc_manifest": human_qc_record,
        "continuation_decision": file_record(decision_path),
        "valid_pool_technical_diversity": valid_pool_technical_diversity,
        "valid_pool_technical_diversity_sha256": (
            diversity_digest
        ),
        "approved_unit_count": len(ready_units),
        "approved_units": sorted(ready_units),
        "diagnosis_labels_used": False,
    }


def validate_phase_b_calibration(config: dict[str, Any]) -> dict[str, Any]:
    expected_technical_diversity = (
        validate_response_calibration_diversity_binding(
            execution_binding=config["execution_binding"],
            response_calibration_binding=config["response_calibration_binding"],
        )
    )
    calibration = config["fod"]["response_estimation"]["calibration"]
    try:
        raw_minimum = calibration["minimum_valid_subjects"]
        if isinstance(raw_minimum, bool):
            raise TypeError
        minimum = int(raw_minimum)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "phase-b requires a prespecified integer response-calibration minimum"
        ) from exc
    if minimum < 1:
        raise ValueError("phase-b response-calibration minimum must be positive")
    response_binding = config["response_calibration_binding"]
    manifest = calibration["frozen_manifest"]
    manifest_path = Path(str(manifest.get("path", ""))).expanduser().resolve()
    manifest_hash = str(manifest.get("sha256", ""))
    if not manifest_path.is_file() or not _valid_sha256(manifest_hash):
        raise ValueError("phase-b requires a frozen response-calibration manifest")
    response_records = calibration["pooled_responses"]
    if (
        minimum != response_binding.get("minimum_valid_subjects")
        or manifest != response_binding.get("response_calibration_manifest")
        or response_records != response_binding.get("pooled_responses")
    ):
        raise ValueError(
            "phase-b calibration injection differs from immutable response binding"
        )
    for tissue in TISSUES:
        expected = response_records.get(tissue, {})
        if not Path(str(expected.get("path", ""))).expanduser().is_file() or not _valid_sha256(
            expected.get("sha256", "")
        ):
            raise ValueError(f"phase-b requires frozen {tissue} response evidence")
    return validate_frozen_response_calibration(
        manifest_path,
        expected_manifest_sha256=manifest_hash,
        minimum_valid_subjects=minimum,
        expected_responses=response_records,
        expected_technical_diversity=expected_technical_diversity,
    )


def _pre_tractography_attrition_by_unit(
    continuation_binding: dict[str, Any],
    *,
    expected_units: tuple[str, ...],
) -> tuple[dict[str, dict[str, Any]], set[str]]:
    """Revalidate pre-tractography attrition for phase-B denominator closure."""

    if (
        continuation_binding.get("schema_version") != "2.0.0"
        or continuation_binding.get("binding_type")
        != "tractography_continuation_binding"
    ):
        raise ValueError("phase-b terminal closure lacks continuation binding")
    approved = continuation_binding.get("approved_units")
    if (
        not isinstance(approved, list)
        or not approved
        or len(approved) != len(set(approved))
        or not set(approved).issubset(set(expected_units))
        or continuation_binding.get("approved_unit_count") != len(approved)
    ):
        raise ValueError("phase-b continuation approved-unit set is invalid")
    manifest_path, _ = _verified_nested_file_record(
        continuation_binding.get("pre_tractography_manifest"),
        label="phase-b pre-tractography manifest",
    )
    manifest = _load_json_mapping(
        manifest_path, label="phase-b pre-tractography manifest"
    )
    if (
        manifest.get("record_type") != "pre_tractography_canary_manifest"
        or manifest.get("status") != "COMPLETE"
        or manifest.get("expected_units") != list(expected_units)
        or manifest.get("ready_units") != approved
    ):
        raise ValueError(
            "phase-b pre-tractography manifest denominator differs"
        )
    ledger_path, _ = _verified_nested_file_record(
        manifest.get("pre_tractography_ledger"),
        label="phase-b pre-tractography ledger",
    )
    entries: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        ledger_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"pre-tractography ledger line {line_number} is invalid JSON"
            ) from exc
        if not isinstance(entry, dict):
            raise ValueError(
                f"pre-tractography ledger line {line_number} is not an object"
            )
        entries.append(entry)
    by_unit = {str(entry.get("unit", "")): entry for entry in entries}
    if (
        len(by_unit) != len(entries)
        or set(by_unit) != set(expected_units)
        or any(
            entry.get("entry_type") != "pre_tractography_canary_outcome"
            for entry in entries
        )
    ):
        raise ValueError("pre-tractography ledger does not close the denominator")
    for unit in expected_units:
        expected_status = "READY_FOR_HUMAN_QC" if unit in approved else "FAIL"
        if by_unit[unit].get("status") != expected_status:
            raise ValueError(
                f"pre-tractography ledger status differs for {unit}"
            )
        if expected_status == "FAIL" and (
            not str(by_unit[unit].get("failure_stage", "")).strip()
            or not str(by_unit[unit].get("primary_failure_reason", "")).strip()
        ):
            raise ValueError(
                f"pre-tractography failed unit lacks attrition reason: {unit}"
            )
    return by_unit, set(approved)


def finalize_terminal_publication(
    *,
    run_root: Path,
    manifest_rows: list[dict[str, str]],
    config: dict[str, Any],
    execution_binding: dict[str, Any] | None = None,
    continuation_binding: dict[str, Any] | None = None,
    run_id: str,
    recipe_id: str,
    run_context_path: Path,
    manifest_path: Path,
    attempt_start_path: Path,
    execution_log_path: Path,
    workflow_returncode: int,
    ended_utc: str,
) -> dict[str, Any]:
    """Close every attempted unit and publish attrition despite DAG failures."""

    expected_units = tuple(row["unit"] for row in manifest_rows)
    if execution_binding is not None and (
        list(expected_units) != execution_binding.get("units")
        or len(expected_units) != execution_binding.get("approved_unit_count")
    ):
        raise ValueError("terminal rows differ from approved canary execution subset")
    expected_set = set(expected_units)
    pre_attrition: dict[str, dict[str, Any]] = {}
    approved_units = set(expected_units)
    if continuation_binding is not None:
        pre_attrition, approved_units = _pre_tractography_attrition_by_unit(
            continuation_binding, expected_units=expected_units
        )
    row_by_unit = {row["unit"]: row for row in manifest_rows}
    log_record = file_record(execution_log_path)
    log_text = execution_log_path.read_text(encoding="utf-8", errors="replace")
    failed_rules = _failed_rule_evidence(log_text, expected_set)
    current_run_context_record = file_record(run_context_path)
    current_attempt_record = file_record(attempt_start_path)
    terminal_paths: list[Path] = []
    for unit in expected_units:
        terminal_path = run_root / "subjects" / unit / "08_qc" / "terminal_record.json"
        if terminal_path.exists():
            if unit not in approved_units:
                raise ValueError(
                    "non-approved pre-tractography unit produced a phase-b "
                    f"terminal record: {unit}"
                )
            existing = _load_json_mapping(
                terminal_path, label=f"existing terminal record {unit}"
            )
            contract_files = existing.get("contract_files", {})
            if (
                existing.get("run_id") != run_id
                or not isinstance(contract_files, dict)
                or contract_files.get("run_context") != current_run_context_record
                or contract_files.get("attempt_context") != current_attempt_record
            ):
                raise ValueError(
                    f"existing terminal record belongs to another attempt: {unit}"
                )
            terminal_paths.append(terminal_path)
            continue

        row = row_by_unit[unit]
        input_dir = run_root / "subjects" / unit / "00_inputs"
        staged_paths = {
            "dwi_raw_mif": input_dir / "dwi_raw.mif",
            "dwi_bvec": input_dir / "dwi.bvec",
            "dwi_bval": input_dir / "dwi.bval",
            "dwi_source_metadata": input_dir / "dwi_source_metadata.json",
            "t1_native_nifti": input_dir / "t1_native.nii",
            "dwi_normalization": input_dir / "dwi_normalization.json",
            "t1_normalization": input_dir / "t1_normalization.json",
            "input_contract": input_dir / "input_contract.json",
        }
        source_files = _existing_file_records(staged_paths)
        staged_input_hashes = {
            key: record["sha256"]
            for key, record in source_files.items()
            if key
            in {
                "dwi_raw_mif",
                "dwi_bvec",
                "dwi_bval",
                "dwi_source_metadata",
                "t1_native_nifti",
            }
        }
        source_identity_hashes = _source_identity_hashes(row)
        intermediate_files: dict[str, dict[str, Any]] = {}
        prior_attrition = (
            pre_attrition.get(unit) if unit not in approved_units else None
        )
        if prior_attrition is not None:
            stage = str(prior_attrition["failure_stage"])
            reason = str(prior_attrition["primary_failure_reason"])
            flags = [
                *(
                    str(value)
                    for value in prior_attrition.get("secondary_flags", [])
                ),
                "tractography_not_authorized_by_h04b_continuation",
            ]
            for prefix, evidence in (
                ("pre_review", prior_attrition.get("review_bundle", {})),
                ("pre_failure", prior_attrition.get("failure_markers", {})),
            ):
                if isinstance(evidence, dict):
                    intermediate_files.update(
                        {
                            f"{prefix}_{name}": dict(record)
                            for name, record in evidence.items()
                            if isinstance(record, dict)
                        }
                    )
            automated = prior_attrition.get("automated_pre_tractography_qc")
            if isinstance(automated, dict):
                intermediate_files[
                    "automated_pre_tractography_qc"
                ] = dict(automated)
        else:
            structured = _normalization_failure_evidence(
                run_root,
                unit,
                source_identity_hashes,
                attempt_context_path=attempt_start_path,
            )
            if structured is not None:
                stage, reason, flags, marker_records = structured
                intermediate_files.update(marker_records)
            else:
                stage, reason, flags = _rule_failure_for_unit(
                    unit, failed_rules, returncode=workflow_returncode
                )
        flags = list(
            dict.fromkeys([*flags, f"snakemake_returncode:{workflow_returncode}"])
        )
        contract_candidates = {
            "manifest": manifest_path,
            "normative_config": NORMATIVE_CONFIG,
            "environment": ENVIRONMENT_CONTRACT,
            "run_context": run_context_path,
            "attempt_context": attempt_start_path,
            "provenance_schema": PROVENANCE_SCHEMA,
        }
        if (input_dir / "input_contract.json").is_file():
            contract_candidates["input_contract"] = input_dir / "input_contract.json"
        contract_files = _existing_file_records(contract_candidates)
        if continuation_binding is not None:
            contract_files.update(
                {
                    "pre_tractography_completion": dict(
                        continuation_binding["pre_tractography_completion"]
                    ),
                    "pre_tractography_manifest": dict(
                        continuation_binding["pre_tractography_manifest"]
                    ),
                    "human_qc_manifest": dict(
                        continuation_binding["human_qc_manifest"]
                    ),
                    "tractography_continuation_decision": dict(
                        continuation_binding["continuation_decision"]
                    ),
                }
            )
        record = make_failure_terminal_record(
            run_id=run_id,
            recipe_id=recipe_id,
            unit=unit,
            terminal_stage=stage,
            primary_failure_reason=reason,
            started_utc=json.loads(
                attempt_start_path.read_text(encoding="utf-8")
            )["started_utc"],
            ended_utc=ended_utc,
            source_identity=row,
            source_identity_hashes=source_identity_hashes,
            staged_input_hashes=staged_input_hashes,
            source_files=source_files,
            contract_files=contract_files,
            intermediate_files=intermediate_files,
            processing_contract=_processing_contract(config),
            command_logs=[log_record],
            secondary_flags=flags,
        )
        schema_errors = validate_schema(record, PROVENANCE_SCHEMA)
        semantic_errors = validate_terminal_semantics(
            record, required_matrices=MATRIX_OUTCOMES
        )
        if schema_errors or semantic_errors:
            raise ValueError(
                f"synthesized terminal record is invalid for {unit}: "
                + "; ".join(schema_errors + semantic_errors)
            )
        terminal_path.parent.mkdir(parents=True, exist_ok=True)
        write_immutable_json(terminal_path, record)
        terminal_paths.append(terminal_path)

    publication_dir = run_root / "publication"
    return publish_terminal_ledger(
        terminal_paths,
        provenance_schema_path=PROVENANCE_SCHEMA,
        ledger_schema_path=RUN_LEDGER_SCHEMA,
        run_context_path=run_context_path,
        expected_units=expected_units,
        required_matrices=MATRIX_OUTCOMES,
        analysis_manifest_path=publication_dir / "analysis_ready_manifest.csv",
        outcome_manifest_path=publication_dir / "outcome_validity_manifest.csv",
        ledger_path=publication_dir / "run_ledger.jsonl",
        ledger_manifest_path=publication_dir / "run_ledger_manifest.json",
        generated_utc=ended_utc,
    )


def _validate_response_calibration_outcome(
    record: dict[str, Any], *, unit: str
) -> None:
    if record.get("schema_version") != "2.0.0":
        raise ValueError(f"response outcome schema differs for {unit}")
    if record.get("record_type") != "response_calibration_outcome":
        raise ValueError(f"response outcome type differs for {unit}")
    if record.get("unit") != unit or record.get("diagnosis_blinded") is not True:
        raise ValueError(f"response outcome identity/blinding differs for {unit}")
    if {"diagnosis", "diagnosis_at_dti", "group", "research_group"}.intersection(record):
        raise ValueError(f"diagnosis/group fields are forbidden in calibration outcome {unit}")
    if not isinstance(record.get("generated_utc"), str) or not record["generated_utc"]:
        raise ValueError(f"response outcome lacks generated_utc for {unit}")
    if not isinstance(record.get("recipe_id"), str) or not record["recipe_id"]:
        raise ValueError(f"response outcome lacks recipe_id for {unit}")
    source_hashes = record.get("source_identity_hashes")
    required_source_hashes = {
        "dti_raw_bundle_sha256",
        "t1_raw_bundle_sha256",
        "pair_content_bundle_sha256",
    }
    if (
        not isinstance(source_hashes, dict)
        or set(source_hashes) != required_source_hashes
        or any(not _valid_sha256(value) for value in source_hashes.values())
    ):
        raise ValueError(f"response outcome source identity hashes differ for {unit}")
    phase_a_identity = record.get("phase_a_identity")
    if not isinstance(phase_a_identity, dict):
        raise ValueError(f"response outcome lacks phase_a_identity for {unit}")
    if not re.fullmatch(r"[a-f0-9]{24}", str(phase_a_identity.get("run_id", ""))):
        raise ValueError(f"response outcome phase-A run_id is invalid for {unit}")
    if not isinstance(phase_a_identity.get("attempt_id"), str) or not phase_a_identity[
        "attempt_id"
    ]:
        raise ValueError(f"response outcome phase-A attempt_id is invalid for {unit}")
    for key in ("run_context", "attempt_context"):
        _, artifact = _verified_nested_file_record(
            phase_a_identity.get(key), label=f"response outcome {unit} {key}"
        )
        phase_a_identity[key] = artifact
    status = record.get("status")
    if status == "PASS":
        responses = record.get("responses")
        if not isinstance(responses, dict) or set(responses) != set(TISSUES):
            raise ValueError(f"PASS response outcome lacks exact tissue responses for {unit}")
        for tissue in TISSUES:
            artifact = responses[tissue]
            if not isinstance(artifact, dict):
                raise ValueError(f"invalid {tissue} response record for {unit}")
            _verified_nested_file_record(
                artifact, label=f"response outcome {unit} {tissue}"
            )
    elif status == "FAIL":
        stage = record.get("failure_stage")
        if not isinstance(stage, str) or not stage.strip():
            raise ValueError(f"FAIL response outcome lacks failure_stage for {unit}")
        reason = record.get("primary_failure_reason")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(f"FAIL response outcome lacks reason for {unit}")
        if record.get("responses") not in ({}, None):
            raise ValueError(f"FAIL response outcome cannot expose responses for {unit}")
    else:
        raise ValueError(f"response outcome has invalid status for {unit}: {status!r}")
    validate_calibration_outcome(record, expected_unit=unit)


def finalize_response_calibration_phase_a(
    *,
    run_root: Path,
    manifest_rows: list[dict[str, str]],
    execution_binding: dict[str, Any],
    run_id: str,
    lineage_id: str,
    recipe_id: str,
    run_context_path: Path,
    attempt_start_path: Path,
    attempt_end_path: Path,
    execution_log_path: Path,
    workflow_returncode: int,
    ended_utc: str,
) -> dict[str, Any]:
    """Close phase-A response outcomes without claiming full-run completion."""

    expected_units = tuple(row["unit"] for row in manifest_rows)
    if (
        list(expected_units) != execution_binding.get("units")
        or len(expected_units) != execution_binding.get("approved_unit_count")
    ):
        raise ValueError("phase-A rows differ from approved canary execution subset")
    expected_set = set(expected_units)
    row_by_unit = {row["unit"]: row for row in manifest_rows}
    log_record = file_record(execution_log_path)
    attempt_start = _load_json_mapping(
        attempt_start_path, label="phase-A attempt start"
    )
    attempt_id = str(attempt_start.get("attempt_id", ""))
    if not attempt_id:
        raise ValueError("phase-A attempt start has no attempt_id")
    failed_rules = _failed_rule_evidence(
        execution_log_path.read_text(encoding="utf-8", errors="replace"),
        expected_set,
    )
    outcome_paths: list[Path] = []
    entries: list[dict[str, Any]] = []
    for unit in expected_units:
        outcome_path = (
            run_root
            / "subjects"
            / unit
            / "05_model"
            / "response_calibration_outcome.json"
        )
        if not outcome_path.exists():
            row = row_by_unit[unit]
            source_hashes = _source_identity_hashes(row)
            structured = _normalization_failure_evidence(
                run_root,
                unit,
                source_hashes,
                attempt_context_path=attempt_start_path,
            )
            if structured is not None:
                stage, reason, flags, marker_records = structured
            else:
                stage, reason, flags = _rule_failure_for_unit(
                    unit, failed_rules, returncode=workflow_returncode
                )
                marker_records = {}
            failure = {
                "schema_version": "2.0.0",
                "record_type": "response_calibration_outcome",
                "status": "FAIL",
                "generated_utc": ended_utc,
                "unit": unit,
                "recipe_id": recipe_id,
                "diagnosis_blinded": True,
                "phase_a_identity": {
                    "run_id": run_id,
                    "attempt_id": attempt_id,
                    "run_context": file_record(run_context_path),
                    "attempt_context": file_record(attempt_start_path),
                },
                "failure_stage": stage,
                "primary_failure_reason": reason,
                "secondary_flags": list(
                    dict.fromkeys(
                        [*flags, f"snakemake_returncode:{workflow_returncode}"]
                    )
                ),
                "source_identity_hashes": source_hashes,
                "responses": {},
                "failure_markers": marker_records,
                "combined_execution_log": log_record,
                "invalid_values_replaced": False,
            }
            _validate_response_calibration_outcome(failure, unit=unit)
            outcome_path.parent.mkdir(parents=True, exist_ok=True)
            write_immutable_json(outcome_path, failure)
        record = json.loads(outcome_path.read_text(encoding="utf-8"))
        _validate_response_calibration_outcome(record, unit=unit)
        expected_source_hashes = _source_identity_hashes(row_by_unit[unit])
        expected_identity = {
            "run_id": run_id,
            "attempt_id": attempt_id,
            "run_context": file_record(run_context_path),
            "attempt_context": file_record(attempt_start_path),
        }
        if (
            record.get("recipe_id") != recipe_id
            or record.get("source_identity_hashes") != expected_source_hashes
            or record.get("phase_a_identity") != expected_identity
        ):
            raise ValueError(
                f"response calibration outcome belongs to another source/attempt: {unit}"
            )
        outcome_record = file_record(outcome_path)
        outcome_paths.append(outcome_path)
        entries.append(
            {
                "schema_version": "2.0.0",
                "entry_type": "response_calibration_phase_a_outcome",
                "run_id": run_id,
                "recipe_id": recipe_id,
                "unit": unit,
                "status": record["status"],
                "failure_stage": record.get("failure_stage"),
                "primary_failure_reason": record.get("primary_failure_reason"),
                "outcome_record": outcome_record,
            }
        )

    publication_dir = run_root / "publication"
    ledger_path = publication_dir / "response_calibration_phase_a_ledger.jsonl"
    manifest_path = publication_dir / "response_calibration_phase_a_manifest.json"
    if ledger_path.exists() or manifest_path.exists():
        raise FileExistsError(
            "phase-A immutable publication already exists without completion"
        )
    write_immutable_jsonl(ledger_path, entries)
    pass_count = sum(entry["status"] == "PASS" for entry in entries)
    fail_count = sum(entry["status"] == "FAIL" for entry in entries)
    run_outcome = (
        "PASS"
        if pass_count == len(entries)
        else ("FAIL" if fail_count == len(entries) else "PARTIAL")
    )
    manifest = {
        "schema_version": "2.0.0",
        "record_type": "response_calibration_phase_a_manifest",
        "status": "COMPLETE",
        "run_outcome": run_outcome,
        "generated_utc": ended_utc,
        "mode": "response-calibration-phase-a",
        "run_id": run_id,
        "lineage_id": lineage_id,
        "recipe_id": recipe_id,
        "execution_binding": copy.deepcopy(execution_binding),
        "run_context": file_record(run_context_path),
        "attempt_start": file_record(attempt_start_path),
        "attempt_end": file_record(attempt_end_path),
        "combined_execution_log": log_record,
        "expected_units": list(expected_units),
        "summary": {
            "expected": len(expected_units),
            "terminal": len(entries),
            "pass": pass_count,
            "fail": fail_count,
        },
        "outcome_records": {
            entry["unit"]: entry["outcome_record"] for entry in entries
        },
        "calibration_freeze_input_outcomes": [
            str(path.resolve()) for path in outcome_paths
        ],
        "phase_a_ledger": file_record(ledger_path),
        "freeze_required_before_phase_b": True,
        "full_run_terminal_publication": False,
        "dummy_response_files_used": False,
    }
    if manifest["summary"]["expected"] != manifest["summary"]["terminal"]:
        raise ValueError("phase-A expected and terminal outcome counts differ")
    write_immutable_json(manifest_path, manifest)
    return manifest


def finalize_pre_tractography_canary(
    *,
    run_root: Path,
    execution_rows: list[dict[str, str]],
    execution_binding: dict[str, Any],
    run_id: str,
    lineage_id: str,
    recipe_id: str,
    run_context_path: Path,
    attempt_start_path: Path,
    attempt_end_path: Path,
    execution_log_path: Path,
    workflow_returncode: int,
    ended_utc: str,
) -> dict[str, Any]:
    """Publish review-ready attrition without claiming terminal analysis output."""

    expected_units = tuple(row["unit"] for row in execution_rows)
    if list(expected_units) != execution_binding.get("units"):
        raise ValueError("pre-tractography execution rows differ from bound canary units")
    expected_set = set(expected_units)
    row_by_unit = {row["unit"]: row for row in execution_rows}
    log_record = file_record(execution_log_path)
    failed_rules = _failed_rule_evidence(
        execution_log_path.read_text(encoding="utf-8", errors="replace"),
        expected_set,
    )
    entries: list[dict[str, Any]] = []
    review_bundle_records: dict[str, dict[str, Any]] = {}
    automated_qc_records: dict[str, dict[str, Any]] = {}
    for unit in expected_units:
        review_dir = run_root / "subjects" / unit / "06_preflight"
        automated_qc_path = review_dir / "automated_pre_tractography_qc.json"
        review_paths = {
            "b0_t1": review_dir / "review_b0_vs_t1.png",
            "b0_5tt": review_dir / "review_b0_vs_5tt.png",
            "b0_atlas": review_dir / "review_b0_vs_aal3.png",
            "index": review_dir / "visual_review_index.csv",
        }
        present = _existing_file_records(review_paths)
        automated_qc_record = (
            file_record(automated_qc_path)
            if automated_qc_path.is_file()
            else None
        )
        local_stage: str | None = None
        local_reason: str | None = None
        local_flags: list[str] = []
        automated_qc_pass = False
        if automated_qc_record is None:
            local_stage = "fod_tensor"
            local_reason = "automated_pre_tractography_qc_missing"
            local_flags = ["automated_qc_not_released_for_human_review"]
        else:
            try:
                automated_qc = _load_json_mapping(
                    automated_qc_path,
                    label=f"automated pre-tractography QC for {unit}",
                )
            except (OSError, TypeError, ValueError) as exc:
                local_stage = "fod_tensor"
                local_reason = (
                    "automated_pre_tractography_qc_unreadable:"
                    f"{type(exc).__name__}"
                )
                local_flags = ["automated_qc_not_released_for_human_review"]
            else:
                automated_qc_pass = (
                    automated_qc.get("schema_version") == "2.0.0"
                    and automated_qc.get("record_type")
                    == "automated_pre_tractography_qc"
                    and automated_qc.get("status") == "PASS"
                    and automated_qc.get("unit") == unit
                    and automated_qc.get("diagnosis_labels_used") is False
                    and isinstance(automated_qc.get("image_ranges"), dict)
                    and isinstance(automated_qc.get("wmfod_qc"), dict)
                    and isinstance(automated_qc.get("commands"), list)
                    and automated_qc.get("failures") == []
                    and not {
                        "diagnosis",
                        "diagnosis_at_dti",
                        "group",
                        "research_group",
                    }.intersection(
                        str(key).strip().lower() for key in automated_qc
                    )
                )
                if not automated_qc_pass:
                    reported_failures = automated_qc.get("failures")
                    first_failure = (
                        str(reported_failures[0]).strip()
                        if isinstance(reported_failures, list)
                        and reported_failures
                        else "contract_or_status_differs"
                    )
                    local_stage = "fod_tensor"
                    local_reason = (
                        "automated_pre_tractography_qc_not_pass:"
                        + first_failure[:1000]
                    )
                    local_flags = [
                        "automated_qc_not_released_for_human_review"
                    ]

        ready = set(present) == set(review_paths) and automated_qc_pass
        stage: str | None = None
        reason: str | None = None
        flags: list[str] = []
        failure_markers: dict[str, dict[str, Any]] = {}
        if ready:
            with review_paths["index"].open(
                newline="", encoding="utf-8-sig"
            ) as handle:
                rows = list(csv.DictReader(handle))
            if (
                len(rows) != 1
                or str(rows[0].get("unit", "")).strip() != unit
                or str(rows[0].get("status", "")).strip().upper() != "PENDING"
            ):
                ready = False
                local_stage = "visual_qc"
                local_reason = "visual_review_index_contract_differs"
                local_flags = ["review_bundle_not_released_for_human_qc"]
        elif local_reason is None:
            local_stage = "visual_qc"
            local_reason = "visual_review_bundle_incomplete"
            local_flags = ["review_bundle_not_released_for_human_qc"]
        if not ready:
            source_hashes = _source_identity_hashes(row_by_unit[unit])
            structured = _normalization_failure_evidence(
                run_root,
                unit,
                source_hashes,
                attempt_context_path=attempt_start_path,
            )
            if structured is not None:
                stage, reason, flags, failure_markers = structured
            elif local_reason is not None:
                stage, reason, flags = local_stage, local_reason, local_flags
            else:
                stage, reason, flags = _rule_failure_for_unit(
                    unit, failed_rules, returncode=workflow_returncode
                )
        entry = {
            "schema_version": "2.0.0",
            "entry_type": "pre_tractography_canary_outcome",
            "run_id": run_id,
            "lineage_id": lineage_id,
            "recipe_id": recipe_id,
            "unit": unit,
            "status": "READY_FOR_HUMAN_QC" if ready else "FAIL",
            "failure_stage": stage,
            "primary_failure_reason": reason,
            "secondary_flags": list(
                dict.fromkeys(
                    [
                        *flags,
                        *(
                            [f"snakemake_returncode:{workflow_returncode}"]
                            if not ready
                            else []
                        ),
                    ]
                )
            ),
            "review_bundle": present,
            "automated_pre_tractography_qc": automated_qc_record,
            "failure_markers": failure_markers,
            "biological_inference_status": "NA",
            "matrix_outcomes_status": "NA",
            "tractography_started": False,
        }
        entries.append(entry)
        if ready:
            review_bundle_records[unit] = present
            assert automated_qc_record is not None
            automated_qc_records[unit] = automated_qc_record

    publication_dir = run_root / "publication"
    ledger_path = publication_dir / "pre_tractography_canary_ledger.jsonl"
    manifest_path = publication_dir / "pre_tractography_canary_manifest.json"
    if ledger_path.exists() or manifest_path.exists():
        raise FileExistsError(
            "pre-tractography immutable publication already exists without completion"
        )
    write_immutable_jsonl(ledger_path, entries)
    ready_units = sorted(
        entry["unit"]
        for entry in entries
        if entry["status"] == "READY_FOR_HUMAN_QC"
    )
    fail_count = len(entries) - len(ready_units)
    run_outcome = (
        "PASS"
        if len(ready_units) == len(entries)
        else ("FAIL" if fail_count == len(entries) else "PARTIAL")
    )
    summary = {
        "expected": len(expected_units),
        "terminal": len(entries),
        "ready_for_human_qc": len(ready_units),
        "fail": fail_count,
    }
    if summary["expected"] != summary["terminal"]:
        raise ValueError("pre-tractography expected and closed counts differ")
    manifest = {
        "schema_version": "2.0.0",
        "record_type": "pre_tractography_canary_manifest",
        "status": "COMPLETE",
        "workflow_state": "AWAITING_HUMAN_QC",
        "run_outcome": run_outcome,
        "generated_utc": ended_utc,
        "mode": "pre-tractography-canary",
        "run_id": run_id,
        "lineage_id": lineage_id,
        "recipe_id": recipe_id,
        "execution_binding": execution_binding,
        "run_context": file_record(run_context_path),
        "attempt_start": file_record(attempt_start_path),
        "attempt_end": file_record(attempt_end_path),
        "combined_execution_log": log_record,
        "expected_units": list(expected_units),
        "ready_units": ready_units,
        "summary": summary,
        "review_bundle_records": review_bundle_records,
        "automated_pre_tractography_qc_records": automated_qc_records,
        "pre_tractography_ledger": file_record(ledger_path),
        "human_qc_required_before_tractography": True,
        "tractography_started": False,
        "full_run_terminal_publication": False,
        "biological_inference_published": False,
        "matrix_outcomes_published": False,
    }
    write_immutable_json(manifest_path, manifest)
    return manifest


def assert_safe_run_root(path: Path) -> None:
    resolved = path.expanduser().resolve()
    if "scforge_v2" not in str(resolved):
        raise ValueError(f"run root requires the safety token scforge_v2: {resolved}")
    if resolved in {Path("/"), PROJECT_ROOT.resolve(), Path.home().resolve()}:
        raise ValueError(f"unsafe run root: {resolved}")
    protected = (
        Path("/data/derivatives/connectomes"),
        Path("/data/derivatives/qc/sc_matrix_qc"),
        PROJECT_ROOT / "research_audit" / "snapshots",
    )
    for candidate in protected:
        try:
            resolved.relative_to(candidate.resolve())
            raise ValueError(f"run root overlaps protected data: {resolved}")
        except ValueError as exc:
            if str(exc).startswith("run root overlaps"):
                raise


def validate_existing_run_context(
    record: dict[str, Any],
    *,
    run_id: str,
    lineage_id: str,
    mode: str,
    recipe_id: str,
    run_root: Path,
    manifest: Path,
    human_qc: Path | None,
    execution_binding: dict[str, Any],
    phase_b_binding: dict[str, Any] | None,
    continuation_binding: dict[str, Any] | None,
) -> None:
    expected = {
        "schema_version": "1.0.0",
        "status": "LOCKED",
        "run_id": run_id,
        "lineage_id": lineage_id,
        "launcher_mode": mode,
        "recipe_id": recipe_id,
        "run_root": str(run_root),
    }
    for key, value in expected.items():
        if record.get(key) != value:
            raise ValueError(f"existing run context differs at {key}: {record.get(key)!r}")
    if record.get("acquisition_manifest") != file_record(manifest):
        raise ValueError("existing run context acquisition-manifest record differs")
    if record.get("execution_binding") != execution_binding:
        raise ValueError("existing run context execution-subset binding differs")
    if human_qc is None:
        if "human_qc_manifest" in record:
            raise ValueError("pre-review run context unexpectedly contains human QC")
    elif record.get("human_qc_manifest") != file_record(human_qc):
        raise ValueError("existing run context human-QC-manifest record differs")
    for name, path in (
        ("normative_config", NORMATIVE_CONFIG),
        ("environment_contract", ENVIRONMENT_CONTRACT),
    ):
        if record.get(name) != file_record(path):
            raise ValueError(f"existing run context {name} record differs")
    workflow_source_manifest = Path(
        load_environment()["workflow"]["source_manifest"]["path"]
    )
    if record.get("workflow_source_manifest") != file_record(
        workflow_source_manifest
    ):
        raise ValueError("existing run context workflow-source record differs")
    if mode in {"pre-tractography-canary", "phase-b"}:
        if record.get("response_calibration_binding") != phase_b_binding:
            raise ValueError("existing run context calibration binding differs")
    elif "response_calibration_binding" in record:
        raise ValueError("phase-A run context unexpectedly contains a phase-B binding")
    if mode == "phase-b":
        if record.get("tractography_continuation_binding") != continuation_binding:
            raise ValueError("existing phase-B continuation binding differs")
    elif "tractography_continuation_binding" in record:
        raise ValueError("pre-continuation run context contains continuation evidence")


def run(args: argparse.Namespace) -> int:
    config = load_yaml(NORMATIVE_CONFIG)
    validate_normative_execution_lock(config)
    normative_config_sha256 = sha256_file(NORMATIVE_CONFIG)
    environment = load_environment()
    recipe_id = str(config["contract"]["recipe_id"])
    mode = str(args.mode)
    if mode not in RUN_MODES:
        raise ValueError(f"unsupported launcher mode: {mode!r}")
    if environment.get("recipe_id") != recipe_id:
        raise ValueError("config and environment recipe IDs differ")

    run_root = args.run_root.expanduser().resolve()
    manifest = args.manifest.expanduser().resolve()
    human_qc_arg = getattr(args, "human_qc_manifest", None)
    human_qc = (
        Path(human_qc_arg).expanduser().resolve()
        if human_qc_arg is not None
        else None
    )
    assert_safe_run_root(run_root)
    if not manifest.is_file():
        raise FileNotFoundError(f"manifest missing: {manifest}")
    if args.cores < 1:
        raise ValueError("--cores must be positive")
    parent_rows = list(
        load_acquisition_manifest(
            manifest,
            Path(config["inputs"]["acquisition_schema"]["path"]),
            run_root=run_root,
            expected_rows=int(
                config["inputs"]["approved_pair_manifest"]["required_row_count"]
            ),
        )
    )
    parent_rows.sort(key=lambda row: row["unit"])

    subset_manifest_arg = getattr(args, "execution_subset_manifest", None)
    subset_decision_arg = getattr(args, "execution_subset_decision", None)
    if subset_manifest_arg is None or subset_decision_arg is None:
        raise ValueError(
            "all canary modes require --execution-subset-manifest and "
            "--execution-subset-decision; full-530 execution is disabled"
        )
    execution_binding, attempt_rows = prepare_execution_binding(
        recipe_id=recipe_id,
        parent_manifest_path=manifest,
        parent_rows=parent_rows,
        execution_subset_manifest_path=Path(subset_manifest_arg),
        execution_subset_decision_path=Path(subset_decision_arg),
        acquisition_schema_path=Path(config["inputs"]["acquisition_schema"]["path"]),
        run_root=run_root,
    )
    h04a_authorization = _validated_h04a_authorization(execution_binding)
    if mode in H04A_AUTHORIZED_MODES:
        require_h04a_mode_authorization(
            execution_binding, mode=mode, requested_cores=args.cores
        )

    manifest_hash = sha256_file(manifest)
    lineage_id = derive_lineage_id(
        recipe_id=recipe_id,
        acquisition_manifest_sha256=manifest_hash,
        execution_binding=execution_binding,
        run_root=run_root,
    )
    contract_dir = run_root / "contract"
    attempts_dir = run_root / "attempts"
    publication_dir = run_root / "publication"

    phase_a_completion_path = (
        publication_dir / "response_calibration_phase_a_completion.json"
    )
    pre_tractography_completion_path = (
        publication_dir / "pre_tractography_canary_completion.json"
    )
    calibration_manifest_arg = getattr(
        args, "response_calibration_manifest", None
    )
    calibration_minimum_arg = getattr(
        args, "response_calibration_minimum", None
    )
    calibration_decision_arg = getattr(args, "response_calibration_decision", None)
    phase_b_binding: dict[str, Any] | None = None
    if mode == "response-calibration-phase-a":
        if any(
            value is not None
            for value in (
                calibration_manifest_arg,
                calibration_minimum_arg,
                calibration_decision_arg,
            )
        ):
            raise ValueError(
                "phase-A mode does not accept phase-B calibration binding arguments"
            )
    else:
        missing_args = [
            flag
            for flag, value in (
                ("--response-calibration-manifest", calibration_manifest_arg),
                ("--response-calibration-minimum", calibration_minimum_arg),
                ("--response-calibration-decision", calibration_decision_arg),
            )
            if value is None
        ]
        if missing_args:
            raise ValueError(
                "phase-b requires immutable calibration arguments: "
                + ", ".join(missing_args)
            )
        try:
            calibration_minimum = int(calibration_minimum_arg)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "--response-calibration-minimum must be a positive integer"
            ) from exc
        signed_calibration_minimum = int(
            h04a_authorization["minimum_valid_response_calibration_units"]
        )
        if calibration_minimum != signed_calibration_minimum:
            raise PermissionError(
                "--response-calibration-minimum differs from the signed H04A "
                f"minimum: {calibration_minimum} != {signed_calibration_minimum}"
            )
        phase_b_binding = prepare_phase_b_binding(
            recipe_id=recipe_id,
            phase_a_completion_path=phase_a_completion_path,
            response_calibration_manifest_path=Path(calibration_manifest_arg),
            minimum_valid_subjects=calibration_minimum,
            response_calibration_decision_path=Path(calibration_decision_arg),
            execution_binding=execution_binding,
        )

    continuation_decision_arg = getattr(
        args, "tractography_continuation_decision", None
    )
    continuation_binding: dict[str, Any] | None = None
    if mode == "phase-b":
        if human_qc is None or continuation_decision_arg is None:
            raise ValueError(
                "phase-b requires --human-qc-manifest and "
                "--tractography-continuation-decision"
            )
        continuation_binding = prepare_tractography_continuation_binding(
            recipe_id=recipe_id,
            execution_binding=execution_binding,
            response_calibration_binding=phase_b_binding,
            pre_tractography_completion_path=pre_tractography_completion_path,
            human_qc_manifest_path=human_qc,
            continuation_decision_path=Path(continuation_decision_arg),
        )
    elif human_qc is not None or continuation_decision_arg is not None:
        raise ValueError(
            f"{mode} must stop before human-QC continuation evidence is supplied"
        )

    run_id = derive_mode_run_id(
        lineage_id=lineage_id,
        mode=mode,
        phase_b_binding=phase_b_binding,
        continuation_binding=continuation_binding,
    )
    completion_path = {
        "response-calibration-phase-a": phase_a_completion_path,
        "pre-tractography-canary": pre_tractography_completion_path,
        "phase-b": publication_dir / "run_completion.json",
    }[mode]
    if completion_path.exists():
        raise FileExistsError(f"launcher mode is already immutably complete: {completion_path}")
    if (
        mode in {"response-calibration-phase-a", "pre-tractography-canary"}
        and (publication_dir / "run_completion.json").exists()
    ):
        raise FileExistsError("terminal phase-B canary is already immutably complete")

    resource_preflight: dict[str, Any] | None = None
    if mode in H04A_AUTHORIZED_MODES:
        resource_preflight = prepare_h04a_resource_preflight(
            run_root=run_root,
            execution_binding=execution_binding,
        )

    # No launcher state is created until every decision, phase binding, and
    # H04A resource preflight above has passed.
    contract_dir.mkdir(parents=True, exist_ok=True)
    attempts_dir.mkdir(parents=True, exist_ok=True)

    run_context_path = contract_dir / {
        "response-calibration-phase-a": "response_calibration_phase_a_run_context.json",
        "pre-tractography-canary": "pre_tractography_canary_run_context.json",
        "phase-b": "phase_b_run_context.json",
    }[mode]
    if run_context_path.exists():
        run_context = _load_json_mapping(run_context_path, label=f"{mode} run context")
        validate_existing_run_context(
            run_context,
            run_id=run_id,
            lineage_id=lineage_id,
            mode=mode,
            recipe_id=recipe_id,
            run_root=run_root,
            manifest=manifest,
            human_qc=human_qc,
            execution_binding=execution_binding,
            phase_b_binding=phase_b_binding,
            continuation_binding=continuation_binding,
        )
    else:
        run_context = {
            "schema_version": "1.0.0",
            "status": "LOCKED",
            "created_utc": utc_now(),
            "run_id": run_id,
            "lineage_id": lineage_id,
            "launcher_mode": mode,
            "recipe_id": recipe_id,
            "run_root": str(run_root),
            "acquisition_manifest": file_record(manifest),
            "execution_binding": copy.deepcopy(execution_binding),
            "normative_config": file_record(NORMATIVE_CONFIG),
            "environment_contract": file_record(ENVIRONMENT_CONTRACT),
            "workflow_source_manifest": file_record(
                environment["workflow"]["source_manifest"]["path"]
            ),
            "host": {
                "node": platform.node(),
                "platform": platform.platform(),
                "architecture": platform.machine(),
                "python": platform.python_version(),
            },
            "safety": {
                "production_overwrite": False,
                "automatic_fallbacks": False,
                "density_route_selection": False,
            },
        }
        if phase_b_binding is not None:
            run_context["response_calibration_binding"] = copy.deepcopy(
                phase_b_binding
            )
        if continuation_binding is not None:
            run_context["tractography_continuation_binding"] = copy.deepcopy(
                continuation_binding
            )
        if human_qc is not None:
            run_context["human_qc_manifest"] = file_record(human_qc)
        write_immutable_json(run_context_path, run_context)

    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    attempt_id = f"{timestamp}-{mode}-{uuid.uuid4().hex[:8]}"
    resolved_config_path = attempts_dir / f"{attempt_id}.resolved_config.yaml"
    attempt_start_path = attempts_dir / f"{attempt_id}.start.json"
    attempt_end_path = attempts_dir / f"{attempt_id}.end.json"
    execution_log_path = attempts_dir / f"{attempt_id}.snakemake.log"

    resolved_config = (
        inject_phase_b_calibration(config, phase_b_binding)
        if phase_b_binding is not None
        else copy.deepcopy(config)
    )
    resolved_config["manifest_path"] = str(manifest)
    resolved_config["run_root"] = str(run_root)
    if human_qc is not None:
        resolved_config["human_qc_manifest"] = str(human_qc)
    resolved_config["run_context_path"] = str(run_context_path)
    resolved_config["attempt_context_path"] = str(attempt_start_path)
    resolved_config["resolved_run_config_path"] = str(resolved_config_path)
    resolved_config["launcher_mode"] = mode
    resolved_config["execution_binding"] = copy.deepcopy(execution_binding)
    resolved_config["execution_manifest_path"] = execution_binding[
        "execution_subset_manifest"
    ]["path"]
    if continuation_binding is not None:
        resolved_config["tractography_continuation_binding"] = copy.deepcopy(
            continuation_binding
        )
    resolved_config["inputs"]["approved_pair_manifest"]["path"] = str(manifest)
    resolved_config["inputs"]["approved_pair_manifest"]["sha256"] = manifest_hash
    if mode in {"pre-tractography-canary", "phase-b"}:
        validate_phase_b_calibration(resolved_config)
    if sha256_file(NORMATIVE_CONFIG) != normative_config_sha256:
        raise RuntimeError("normative config changed while preparing immutable attempt")
    resolved_payload = yaml.safe_dump(resolved_config, sort_keys=False).encode("utf-8")
    write_immutable_bytes(resolved_config_path, resolved_payload)

    snakemake = Path(environment["workflow"]["snakemake"]["path"])
    command = build_snakemake_command(
        snakemake=snakemake,
        resolved_config_path=resolved_config_path,
        cores=args.cores,
        mode=mode,
    )
    attempt_start = {
        "schema_version": "1.0.0",
        "status": "STARTED",
        "started_utc": utc_now(),
        "attempt_id": attempt_id,
        "launcher_mode": mode,
        "run_id": run_id,
        "lineage_id": lineage_id,
        "recipe_id": recipe_id,
        "execution_binding": copy.deepcopy(execution_binding),
        "run_context": file_record(run_context_path),
        "resolved_run_config": file_record(resolved_config_path),
        "snakemake_invocation": command,
        "cwd": str(PROJECT_ROOT),
        "combined_execution_log": str(execution_log_path),
        "printshellcmds": True,
    }
    if resource_preflight is not None:
        attempt_start["h04a_resource_preflight"] = copy.deepcopy(
            resource_preflight
        )
    if phase_b_binding is not None:
        attempt_start["response_calibration_binding"] = copy.deepcopy(
            phase_b_binding
        )
    if continuation_binding is not None:
        attempt_start["tractography_continuation_binding"] = copy.deepcopy(
            continuation_binding
        )
    write_immutable_json(attempt_start_path, attempt_start)

    returncode, started, ended, resource_runtime = (
        execute_command_with_h04a_monitor(
            command=command,
            cwd=PROJECT_ROOT,
            execution_log_path=execution_log_path,
            run_root=run_root,
            resource_preflight=resource_preflight,
        )
    )

    output_paths = (
        {}
        if mode != "phase-b"
        else {
            "analysis_ready_manifest": publication_dir / "analysis_ready_manifest.csv",
            "outcome_validity_manifest": publication_dir / "outcome_validity_manifest.csv",
            "run_ledger": publication_dir / "run_ledger.jsonl",
            "run_ledger_manifest": publication_dir / "run_ledger_manifest.json",
        }
    )
    output_records = {
        name: file_record(path) for name, path in output_paths.items() if path.is_file()
    }
    attempt_end = {
        "schema_version": "1.0.0",
        "status": "PASS" if returncode == 0 else "FAIL",
        "started_utc": started.isoformat(),
        "ended_utc": ended.isoformat(),
        "duration_seconds": (ended - started).total_seconds(),
        "attempt_id": attempt_id,
        "launcher_mode": mode,
        "run_id": run_id,
        "lineage_id": lineage_id,
        "recipe_id": recipe_id,
        "execution_binding": copy.deepcopy(execution_binding),
        "returncode": returncode,
        "attempt_start": file_record(attempt_start_path),
        "combined_execution_log": file_record(execution_log_path),
        "outputs": output_records,
    }
    if resource_preflight is not None:
        attempt_end["h04a_resource_enforcement"] = {
            "preflight": copy.deepcopy(resource_preflight),
            "runtime": copy.deepcopy(resource_runtime),
        }
    write_immutable_json(attempt_end_path, attempt_end)

    if mode == "response-calibration-phase-a":
        phase_a_manifest = finalize_response_calibration_phase_a(
            run_root=run_root,
            manifest_rows=attempt_rows,
            execution_binding=execution_binding,
            run_id=run_id,
            lineage_id=lineage_id,
            recipe_id=recipe_id,
            run_context_path=run_context_path,
            attempt_start_path=attempt_start_path,
            attempt_end_path=attempt_end_path,
            execution_log_path=execution_log_path,
            workflow_returncode=returncode,
            ended_utc=ended.isoformat(),
        )
        phase_a_manifest_path = (
            publication_dir / "response_calibration_phase_a_manifest.json"
        )
        phase_a_ledger_path = (
            publication_dir / "response_calibration_phase_a_ledger.jsonl"
        )
        completion = {
            "schema_version": "2.0.0",
            "record_type": "response_calibration_phase_a_completion",
            "status": phase_a_manifest["run_outcome"],
            "completed_utc": utc_now(),
            "mode": mode,
            "run_id": run_id,
            "lineage_id": lineage_id,
            "recipe_id": recipe_id,
            "execution_binding": copy.deepcopy(execution_binding),
            "run_context": file_record(run_context_path),
            "terminal_attempt_start": file_record(attempt_start_path),
            "terminal_attempt_end": file_record(attempt_end_path),
            "combined_execution_log": file_record(execution_log_path),
            "snakemake_returncode": returncode,
            "phase_a_summary": phase_a_manifest["summary"],
            "phase_a_manifest": file_record(phase_a_manifest_path),
            "phase_a_ledger": file_record(phase_a_ledger_path),
            "freeze_required_before_phase_b": True,
            "full_run_terminal_publication": False,
            "production_overwrite": False,
        }
        write_immutable_json(completion_path, completion)
        return returncode

    if mode == "pre-tractography-canary":
        pre_manifest = finalize_pre_tractography_canary(
            run_root=run_root,
            execution_rows=attempt_rows,
            execution_binding=execution_binding,
            run_id=run_id,
            lineage_id=lineage_id,
            recipe_id=recipe_id,
            run_context_path=run_context_path,
            attempt_start_path=attempt_start_path,
            attempt_end_path=attempt_end_path,
            execution_log_path=execution_log_path,
            workflow_returncode=returncode,
            ended_utc=ended.isoformat(),
        )
        pre_manifest_path = (
            publication_dir / "pre_tractography_canary_manifest.json"
        )
        pre_ledger_path = (
            publication_dir / "pre_tractography_canary_ledger.jsonl"
        )
        completion = {
            "schema_version": "2.0.0",
            "record_type": "pre_tractography_canary_completion",
            "status": "AWAITING_HUMAN_QC",
            "run_outcome": pre_manifest["run_outcome"],
            "completed_utc": utc_now(),
            "mode": mode,
            "run_id": run_id,
            "lineage_id": lineage_id,
            "recipe_id": recipe_id,
            "execution_binding": copy.deepcopy(execution_binding),
            "response_calibration_binding": copy.deepcopy(phase_b_binding),
            "run_context": file_record(run_context_path),
            "terminal_attempt_start": file_record(attempt_start_path),
            "terminal_attempt_end": file_record(attempt_end_path),
            "combined_execution_log": file_record(execution_log_path),
            "snakemake_returncode": returncode,
            "pre_tractography_summary": pre_manifest["summary"],
            "pre_tractography_manifest": file_record(pre_manifest_path),
            "pre_tractography_ledger": file_record(pre_ledger_path),
            "automated_pre_tractography_qc_records": copy.deepcopy(
                pre_manifest["automated_pre_tractography_qc_records"]
            ),
            "human_qc_required_before_tractography": True,
            "tractography_started": False,
            "full_run_terminal_publication": False,
            "production_overwrite": False,
        }
        write_immutable_json(completion_path, completion)
        return returncode

    assert human_qc is not None
    assert continuation_decision_arg is not None
    assert phase_b_binding is not None
    assert continuation_binding is not None
    post_run_continuation_binding = prepare_tractography_continuation_binding(
        recipe_id=recipe_id,
        execution_binding=execution_binding,
        response_calibration_binding=phase_b_binding,
        pre_tractography_completion_path=pre_tractography_completion_path,
        human_qc_manifest_path=human_qc,
        continuation_decision_path=Path(continuation_decision_arg),
    )
    if post_run_continuation_binding != continuation_binding:
        raise ValueError(
            "tractography continuation evidence changed during phase-b execution"
        )

    ledger_manifest = finalize_terminal_publication(
        run_root=run_root,
        manifest_rows=attempt_rows,
        config=resolved_config,
        execution_binding=execution_binding,
        continuation_binding=continuation_binding,
        run_id=run_id,
        recipe_id=recipe_id,
        run_context_path=run_context_path,
        manifest_path=manifest,
        attempt_start_path=attempt_start_path,
        execution_log_path=execution_log_path,
        workflow_returncode=returncode,
        ended_utc=ended.isoformat(),
    )
    if (
        execution_binding.get("execution_scope") == "canary"
        and ledger_manifest.get("summary", {}).get("analysis_ready") != 0
    ):
        raise ValueError(
            "canary publication attempted biological-inference release before H04/SAP"
        )
    output_records = {
        name: file_record(path) for name, path in output_paths.items() if path.is_file()
    }
    missing_outputs = sorted(set(output_paths) - set(output_records))
    if missing_outputs:
        raise FileNotFoundError(
            "Snakemake exited successfully without final evidence: "
            + ", ".join(missing_outputs)
        )
    completion = {
        "schema_version": "2.0.0",
        "status": ledger_manifest["run_outcome"],
        "completed_utc": utc_now(),
        "run_id": run_id,
        "lineage_id": lineage_id,
        "recipe_id": recipe_id,
        "execution_binding": copy.deepcopy(execution_binding),
        "mode": mode,
        "run_context": file_record(run_context_path),
        "terminal_attempt_start": file_record(attempt_start_path),
        "terminal_attempt_end": file_record(attempt_end_path),
        "combined_execution_log": file_record(execution_log_path),
        "snakemake_returncode": returncode,
        "ledger_summary": ledger_manifest["summary"],
        "outputs": output_records,
        "exact_commands_captured": True,
        "biological_inference_release": (
            "LOCKED_PENDING_DIAGNOSIS_BLIND_H04_SAP"
        ),
        "analysis_ready_for_biological_inference": False,
        "production_overwrite": False,
    }
    completion["response_calibration_binding"] = copy.deepcopy(phase_b_binding)
    completion["tractography_continuation_binding"] = copy.deepcopy(
        continuation_binding
    )
    publication_dir.mkdir(parents=True, exist_ok=True)
    write_immutable_json(completion_path, completion)
    return returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--human-qc-manifest", type=Path)
    parser.add_argument("--execution-subset-manifest", type=Path, required=True)
    parser.add_argument("--execution-subset-decision", type=Path, required=True)
    parser.add_argument("--cores", type=int, default=1)
    parser.add_argument("--mode", choices=RUN_MODES, default="phase-b")
    parser.add_argument(
        "--response-calibration-manifest",
        type=Path,
        help="immutable frozen phase-A calibration (pre-tractography and phase B)",
    )
    parser.add_argument(
        "--response-calibration-minimum",
        type=int,
        help="human-approved prespecified minimum valid calibration N",
    )
    parser.add_argument(
        "--response-calibration-decision",
        type=Path,
        help="immutable APPROVED response-calibration decision JSON",
    )
    parser.add_argument(
        "--tractography-continuation-decision",
        type=Path,
        help="immutable H04B approval tied to pre-gate bundles and human QC",
    )
    args = parser.parse_args()
    try:
        return run(args)
    except Exception as exc:
        print(f"FATAL {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
