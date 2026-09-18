"""Diagnosis-blind, non-contagious response-calibration handoff helpers.

The supported release path is :func:`freeze_response_calibration_from_phase_a`.
It consumes an immutable, terminally complete phase-A publication and verifies
every declared outcome before pooling.  :func:`freeze_response_calibration` is
kept as a lower-level helper for focused tests; a manifest made without
``phase_a_lineage`` is deliberately not accepted by the phase-B validator.
"""

from __future__ import annotations

import collections
import csv
import hashlib
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from .input_contract import sha256_file


TISSUES = ("wm", "gm", "csf")
SOURCE_HASH_KEYS = (
    "dti_raw_bundle_sha256",
    "t1_raw_bundle_sha256",
    "pair_content_bundle_sha256",
)
SHA256_RE = re.compile(r"[0-9a-f]{64}")
FORBIDDEN_GROUP_FIELDS = frozenset(
    {"diagnosis", "diagnosis_at_dti", "group", "research_group"}
)
SUPPORTED_T1_SOURCE_CLASSES = ("dicom_series", "nifti_single")


def file_record(path: str | Path) -> dict[str, Any]:
    candidate = Path(path).resolve()
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    return {
        "path": str(candidate),
        "sha256": sha256_file(candidate),
        "size_bytes": candidate.stat().st_size,
    }


def verify_file_record(
    record: Mapping[str, Any],
    *,
    label: str,
    expected_path: str | Path | None = None,
) -> Path:
    if not isinstance(record, Mapping):
        raise ValueError(f"{label} is not a file record")
    path = Path(str(record.get("path", ""))).resolve()
    if expected_path is not None and path != Path(expected_path).resolve():
        raise ValueError(f"{label} path differs from the required path")
    expected_hash = str(record.get("sha256", "")).lower()
    expected_size = record.get("size_bytes")
    if not path.is_file():
        raise FileNotFoundError(f"{label} missing: {path}")
    if not SHA256_RE.fullmatch(expected_hash):
        raise ValueError(f"{label} has an invalid SHA-256")
    if sha256_file(path) != expected_hash:
        raise ValueError(f"{label} SHA-256 mismatch")
    if not isinstance(expected_size, int) or expected_size < 0:
        raise ValueError(f"{label} has invalid size_bytes")
    if path.stat().st_size != expected_size:
        raise ValueError(f"{label} size mismatch")
    return path


def manufacturer_family(value: Any) -> str:
    """Map bound acquisition metadata to the prespecified scanner families."""

    normalized = str(value or "").strip().upper()
    if "SIEMENS" in normalized:
        return "SIEMENS"
    if normalized.startswith("GE"):
        return "GE"
    if "PHILIPS" in normalized:
        return "PHILIPS"
    raise ValueError(f"unsupported manufacturer family in bound canary metadata: {value!r}")


def technical_diversity_contract_from_authorization(
    authorization: Mapping[str, Any],
) -> dict[str, Any]:
    """Extract and type-check the signed H04A valid-pool diversity contract."""

    try:
        minimum_manufacturers = authorization[
            "minimum_valid_manufacturer_families"
        ]
        minimum_t1_classes = authorization["minimum_valid_t1_source_classes"]
        required_t1_classes = authorization["required_valid_t1_source_classes"]
    except KeyError as exc:
        raise ValueError(f"signed H04A diversity field is missing: {exc.args[0]}") from exc
    if (
        isinstance(minimum_manufacturers, bool)
        or not isinstance(minimum_manufacturers, int)
        or minimum_manufacturers < 1
    ):
        raise ValueError("minimum valid manufacturer families must be positive")
    if (
        isinstance(minimum_t1_classes, bool)
        or not isinstance(minimum_t1_classes, int)
        or minimum_t1_classes < 1
    ):
        raise ValueError("minimum valid T1 source classes must be positive")
    if (
        not isinstance(required_t1_classes, list)
        or not required_t1_classes
        or any(not isinstance(value, str) or not value for value in required_t1_classes)
        or required_t1_classes != sorted(set(required_t1_classes))
        or any(value not in SUPPORTED_T1_SOURCE_CLASSES for value in required_t1_classes)
        or minimum_t1_classes != len(required_t1_classes)
    ):
        raise ValueError("required valid T1 source classes are not an exact signed set")
    return {
        "minimum_valid_manufacturer_families": minimum_manufacturers,
        "minimum_valid_t1_source_classes": minimum_t1_classes,
        "required_valid_t1_source_classes": list(required_t1_classes),
    }


def load_execution_subset_technical_metadata(
    execution_binding: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[str, str]]]:
    """Load only unit/manufacturer/T1 class from the exact bound subset CSV."""

    record = execution_binding.get("execution_subset_manifest")
    subset_path = verify_file_record(record, label="bound execution-subset manifest")
    expected_units = execution_binding.get("units")
    if (
        not isinstance(expected_units, list)
        or not expected_units
        or any(not isinstance(unit, str) or not unit for unit in expected_units)
        or expected_units != sorted(set(expected_units))
    ):
        raise ValueError("execution binding lacks an exact sorted unit set")
    with subset_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required_fields = {"unit", "manufacturer", "t1_source_kind"}
        if not required_fields.issubset(reader.fieldnames or ()):
            raise ValueError("execution-subset manifest lacks technical diversity fields")
        rows = list(reader)
    metadata: dict[str, dict[str, str]] = {}
    for row in rows:
        unit = str(row.get("unit", "")).strip()
        if not unit or unit in metadata:
            raise ValueError("execution-subset technical metadata has duplicate/blank unit")
        raw_manufacturer = str(row.get("manufacturer", "")).strip()
        t1_source_class = str(row.get("t1_source_kind", "")).strip()
        if t1_source_class not in SUPPORTED_T1_SOURCE_CLASSES:
            raise ValueError(
                f"unsupported T1 source class in bound canary metadata: {t1_source_class!r}"
            )
        metadata[unit] = {
            "manufacturer": raw_manufacturer,
            "manufacturer_family": manufacturer_family(raw_manufacturer),
            "t1_source_class": t1_source_class,
        }
    if set(metadata) != set(expected_units):
        raise ValueError("bound technical metadata differs from execution unit set")
    return dict(record), metadata


def technical_diversity_sha256(record: Mapping[str, Any]) -> str:
    payload = json.dumps(record, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_valid_pool_technical_diversity(
    *,
    valid_units: Iterable[str],
    unit_metadata: Mapping[str, Mapping[str, Any]],
    execution_subset_manifest: Mapping[str, Any],
    minimum_valid_manufacturer_families: int,
    minimum_valid_t1_source_classes: int,
    required_valid_t1_source_classes: Iterable[str],
) -> dict[str, Any]:
    """Build exact valid-pool composition and fail before pooling if insufficient."""

    contract = technical_diversity_contract_from_authorization(
        {
            "minimum_valid_manufacturer_families": (
                minimum_valid_manufacturer_families
            ),
            "minimum_valid_t1_source_classes": minimum_valid_t1_source_classes,
            "required_valid_t1_source_classes": list(
                required_valid_t1_source_classes
            ),
        }
    )
    units = sorted(valid_units)
    if not units or len(units) != len(set(units)):
        raise ValueError("valid response pool requires unique nonempty units")
    if not isinstance(execution_subset_manifest, Mapping):
        raise ValueError("valid-pool diversity lacks bound subset-manifest evidence")
    missing = sorted(set(units) - set(unit_metadata))
    if missing:
        raise ValueError(
            "valid response units lack bound acquisition metadata: " + ", ".join(missing)
        )
    normalized_metadata: dict[str, dict[str, str]] = {}
    raw_manufacturer_counts: collections.Counter[str] = collections.Counter()
    manufacturer_family_counts: collections.Counter[str] = collections.Counter()
    t1_counts: collections.Counter[str] = collections.Counter()
    for unit in units:
        metadata = unit_metadata[unit]
        raw_manufacturer = str(metadata.get("manufacturer", "")).strip()
        observed_family = str(metadata.get("manufacturer_family", "")).strip()
        expected_family = manufacturer_family(raw_manufacturer)
        t1_source_class = str(metadata.get("t1_source_class", "")).strip()
        if observed_family != expected_family:
            raise ValueError(f"manufacturer-family mapping differs for {unit}")
        if t1_source_class not in SUPPORTED_T1_SOURCE_CLASSES:
            raise ValueError(f"T1 source class differs for {unit}")
        normalized_metadata[unit] = {
            "manufacturer": raw_manufacturer,
            "manufacturer_family": expected_family,
            "t1_source_class": t1_source_class,
        }
        raw_manufacturer_counts[raw_manufacturer] += 1
        manufacturer_family_counts[expected_family] += 1
        t1_counts[t1_source_class] += 1
    required_t1 = set(contract["required_valid_t1_source_classes"])
    if (
        len(manufacturer_family_counts)
        < contract["minimum_valid_manufacturer_families"]
    ):
        raise ValueError(
            "valid response pool has insufficient manufacturer-family diversity: "
            f"{len(manufacturer_family_counts)} < "
            f"{contract['minimum_valid_manufacturer_families']}"
        )
    if set(t1_counts) != required_t1:
        missing_t1 = sorted(required_t1 - set(t1_counts))
        unexpected_t1 = sorted(set(t1_counts) - required_t1)
        raise ValueError(
            "valid response pool does not contain the exact required T1 source "
            f"classes; missing={missing_t1}, unexpected={unexpected_t1}"
        )
    if len(t1_counts) < contract["minimum_valid_t1_source_classes"]:
        raise ValueError(
            "valid response pool has insufficient T1 source-class diversity"
        )
    return {
        "schema_version": "1.0.0",
        "status": "PASS",
        "rule": "PASS_response_units_joined_to_bound_execution_subset_metadata",
        "diagnosis_labels_used": False,
        "execution_subset_manifest": dict(execution_subset_manifest),
        **contract,
        "valid_unit_count": len(units),
        "manufacturer_count": len(raw_manufacturer_counts),
        "manufacturer_counts": dict(sorted(raw_manufacturer_counts.items())),
        "manufacturer_family_count": len(manufacturer_family_counts),
        "manufacturer_family_counts": dict(
            sorted(manufacturer_family_counts.items())
        ),
        "t1_source_class_count": len(t1_counts),
        "t1_source_class_counts": dict(sorted(t1_counts.items())),
        "valid_unit_metadata": normalized_metadata,
    }


def validate_valid_pool_technical_diversity(
    record: Mapping[str, Any],
    *,
    expected_valid_units: Iterable[str],
    expected_execution_subset_manifest: Mapping[str, Any],
    minimum_valid_manufacturer_families: int,
    minimum_valid_t1_source_classes: int,
    required_valid_t1_source_classes: Iterable[str],
) -> dict[str, Any]:
    if not isinstance(record, Mapping):
        raise ValueError("frozen response calibration lacks valid-pool diversity")
    expected = build_valid_pool_technical_diversity(
        valid_units=expected_valid_units,
        unit_metadata=record.get("valid_unit_metadata", {}),
        execution_subset_manifest=expected_execution_subset_manifest,
        minimum_valid_manufacturer_families=minimum_valid_manufacturer_families,
        minimum_valid_t1_source_classes=minimum_valid_t1_source_classes,
        required_valid_t1_source_classes=required_valid_t1_source_classes,
    )
    if dict(record) != expected:
        raise ValueError("frozen valid-pool technical diversity record differs")
    return expected


def _write_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError(
                f"Refusing to replace non-identical frozen calibration artifact: {path}"
            )
        return
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _array_payload(array: np.ndarray) -> bytes:
    lines = [
        " ".join(format(float(value), ".12g") for value in row)
        for row in np.atleast_2d(array)
    ]
    return ("\n".join(lines) + "\n").encode("ascii")


def _required_source_hashes(source_hashes: Mapping[str, Any]) -> dict[str, str]:
    if not isinstance(source_hashes, Mapping) or set(source_hashes) != set(
        SOURCE_HASH_KEYS
    ):
        raise ValueError(
            "response outcome requires the exact DTI, T1 and pair source hashes"
        )
    result = {key: str(source_hashes[key]).lower() for key in SOURCE_HASH_KEYS}
    if any(not SHA256_RE.fullmatch(value) for value in result.values()):
        raise ValueError("response outcome contains an invalid source SHA-256")
    return result


def assess_response_calibration_candidate(
    response_paths: Mapping[str, str | Path],
    *,
    selected_voxel_counts: Mapping[str, int],
    expected_shell_rows: int,
    expected_coefficient_columns: Mapping[str, int],
) -> dict[str, Any]:
    """Apply prespecified, diagnosis-blind response-voxel/coefficient QC."""

    if set(response_paths) != set(TISSUES):
        raise ValueError("response QC requires exact WM, GM and CSF paths")
    if set(selected_voxel_counts) != set(TISSUES):
        raise ValueError("selected voxel counts must cover WM, GM and CSF")
    if expected_shell_rows < 1:
        raise ValueError("expected_shell_rows must be positive")
    if set(expected_coefficient_columns) != set(TISSUES) or any(
        not isinstance(expected_coefficient_columns[tissue], int)
        or expected_coefficient_columns[tissue] < 1
        for tissue in TISSUES
    ):
        raise ValueError("expected coefficient columns must cover all tissues")

    reasons: list[str] = []
    shapes: dict[str, list[int]] = {}
    finite: dict[str, bool] = {}
    positive_isotropic: dict[str, bool] = {}
    for tissue in TISSUES:
        path = Path(response_paths[tissue]).resolve()
        try:
            array = np.loadtxt(path, dtype=float, ndmin=2)
        except Exception as exc:
            shapes[tissue] = []
            finite[tissue] = False
            positive_isotropic[tissue] = False
            reasons.append(
                f"{tissue}_response_unreadable:{type(exc).__name__}"
            )
            continue
        shapes[tissue] = list(array.shape)
        finite[tissue] = bool(array.size and np.isfinite(array).all())
        positive_isotropic[tissue] = bool(
            finite[tissue] and np.all(array[:, 0] > 0.0)
        )
        expected_shape = [
            expected_shell_rows,
            expected_coefficient_columns[tissue],
        ]
        if shapes[tissue] != expected_shape:
            reasons.append(
                f"{tissue}_response_shape:{shapes[tissue]}_expected:{expected_shape}"
            )
        if not finite[tissue]:
            reasons.append(f"{tissue}_response_nonfinite_or_empty")
        if not positive_isotropic[tissue]:
            reasons.append(f"{tissue}_isotropic_coefficient_not_positive")
    for tissue in TISSUES:
        count = selected_voxel_counts[tissue]
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            reasons.append(f"{tissue}_selected_voxel_count_not_positive:{count!r}")
    return {
        "status": "PASS" if not reasons else "FAIL",
        "diagnosis_labels_used": False,
        "selected_voxel_counts": {
            tissue: selected_voxel_counts[tissue] for tissue in TISSUES
        },
        "selected_voxel_count_validity_rule": "each_dhollander_csf_gm_wm_volume_count_must_be_positive",
        "stronger_absolute_minimum_applied": False,
        "expected_shell_rows": expected_shell_rows,
        "expected_coefficient_columns": {
            tissue: expected_coefficient_columns[tissue] for tissue in TISSUES
        },
        "observed_response_shapes": shapes,
        "finite_coefficients": finite,
        "positive_isotropic_coefficients": positive_isotropic,
        "failure_reasons": reasons,
    }


def build_pass_response_calibration_outcome(
    *,
    unit: str,
    recipe_id: str,
    source_identity_hashes: Mapping[str, Any],
    response_paths: Mapping[str, str | Path],
    generated_utc: str,
    phase_a_run_id: str,
    phase_a_attempt_id: str,
    run_context_path: str | Path,
    attempt_context_path: str | Path,
    response_qc: Mapping[str, Any],
    response_voxels_path: str | Path,
    fod_shell_selection_path: str | Path,
) -> dict[str, Any]:
    """Build a fully bound PASS outcome for one phase-A unit."""

    if not unit or not recipe_id or not generated_utc:
        raise ValueError("unit, recipe_id and generated_utc are required")
    if not phase_a_run_id or not phase_a_attempt_id:
        raise ValueError("phase-A run and attempt IDs are required")
    if set(response_paths) != set(TISSUES):
        raise ValueError("PASS outcome requires the exact WM, GM and CSF responses")
    if response_qc.get("status") != "PASS":
        raise ValueError("PASS outcome requires diagnosis-blind PASS response QC")
    shell_selection_path = Path(fod_shell_selection_path).resolve()
    shell_selection = json.loads(
        shell_selection_path.read_text(encoding="utf-8")
    )
    selected_shells = shell_selection.get("selected_shells_s_per_mm2")
    selected_sizes = shell_selection.get("selected_shell_sizes")
    bzero_threshold = shell_selection.get("bzero_threshold_s_per_mm2")
    if (
        shell_selection.get("status") != "PASS"
        or shell_selection.get("unit") != unit
        or not isinstance(selected_shells, list)
        or len(selected_shells) != 2
        or not isinstance(bzero_threshold, (int, float))
        or not 0.0 <= float(selected_shells[0]) <= float(bzero_threshold)
        or float(selected_shells[1]) <= float(bzero_threshold)
        or not isinstance(selected_sizes, list)
        or len(selected_sizes) != 2
        or any(not isinstance(value, int) or value < 1 for value in selected_sizes)
    ):
        raise ValueError("PASS outcome requires a valid two-shell FOD selection")
    record = {
        "schema_version": "2.0.0",
        "record_type": "response_calibration_outcome",
        "status": "PASS",
        "generated_utc": generated_utc,
        "unit": unit,
        "recipe_id": recipe_id,
        "diagnosis_blinded": True,
        "source_identity_hashes": _required_source_hashes(
            source_identity_hashes
        ),
        "phase_a_identity": {
            "run_id": phase_a_run_id,
            "attempt_id": phase_a_attempt_id,
            "run_context": file_record(run_context_path),
            "attempt_context": file_record(attempt_context_path),
        },
        "responses": {
            tissue: file_record(response_paths[tissue]) for tissue in TISSUES
        },
        "response_qc": dict(response_qc),
        "response_voxels": file_record(response_voxels_path),
        "fod_shell_contract": {
            "selection_record": file_record(shell_selection_path),
            "selected_shells_s_per_mm2": [float(value) for value in selected_shells],
            "selected_shell_sizes": selected_sizes,
            "bzero_threshold_s_per_mm2": float(bzero_threshold),
            "selection_uses_diagnosis_labels": False,
        },
        "invalid_values_replaced": False,
    }
    validate_response_calibration_outcome(record)
    return record


def build_failed_response_calibration_outcome(
    *,
    unit: str,
    recipe_id: str,
    source_identity_hashes: Mapping[str, Any],
    generated_utc: str,
    phase_a_run_id: str,
    phase_a_attempt_id: str,
    run_context_path: str | Path,
    attempt_context_path: str | Path,
    primary_failure_reason: str,
    response_qc: Mapping[str, Any] | None = None,
    candidate_response_paths: Mapping[str, str | Path] | None = None,
    response_voxels_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build a terminal FAIL outcome without exposing candidate data as valid."""

    reason = " ".join(str(primary_failure_reason).split())[:2000]
    if not reason:
        raise ValueError("primary_failure_reason must be nonempty")
    candidate_records: dict[str, dict[str, Any]] = {}
    if candidate_response_paths is not None:
        if set(candidate_response_paths) != set(TISSUES):
            raise ValueError("candidate response paths must cover every tissue")
        candidate_records = {
            tissue: file_record(candidate_response_paths[tissue])
            for tissue in TISSUES
        }
    record = {
        "schema_version": "2.0.0",
        "record_type": "response_calibration_outcome",
        "status": "FAIL",
        "generated_utc": generated_utc,
        "unit": unit,
        "recipe_id": recipe_id,
        "diagnosis_blinded": True,
        "source_identity_hashes": _required_source_hashes(
            source_identity_hashes
        ),
        "phase_a_identity": {
            "run_id": phase_a_run_id,
            "attempt_id": phase_a_attempt_id,
            "run_context": file_record(run_context_path),
            "attempt_context": file_record(attempt_context_path),
        },
        "primary_failure_reason": reason,
        "responses": {},
        "response_qc": dict(response_qc or {}),
        "candidate_response_evidence": candidate_records,
        "response_voxels": (
            file_record(response_voxels_path)
            if response_voxels_path is not None
            else None
        ),
        "invalid_values_replaced": False,
    }
    validate_response_calibration_outcome(record)
    return record


def write_failed_response_calibration_outcome(
    path: str | Path, **kwargs: Any
) -> dict[str, Any]:
    record = build_failed_response_calibration_outcome(**kwargs)
    _write_new(
        Path(path).resolve(),
        (json.dumps(record, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    return record


def write_pass_response_calibration_outcome(
    path: str | Path, **kwargs: Any
) -> dict[str, Any]:
    """Create a write-once PASS outcome and return its validated record."""

    record = build_pass_response_calibration_outcome(**kwargs)
    _write_new(
        Path(path).resolve(),
        (json.dumps(record, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    return record


def validate_response_calibration_outcome(
    record: Mapping[str, Any],
    *,
    expected_unit: str | None = None,
    expected_recipe_id: str | None = None,
    expected_run_id: str | None = None,
    expected_attempt_id: str | None = None,
) -> dict[str, Any]:
    """Validate terminal outcome identity plus every referenced file."""

    row = dict(record)
    unit = str(row.get("unit", ""))
    if row.get("schema_version") != "2.0.0":
        raise ValueError(f"response outcome schema differs for {unit or '<unknown>'}")
    if row.get("record_type") != "response_calibration_outcome":
        raise ValueError(f"response outcome type differs for {unit or '<unknown>'}")
    if not unit or (expected_unit is not None and unit != expected_unit):
        raise ValueError("response outcome unit identity differs")
    recipe_id = str(row.get("recipe_id", ""))
    if not recipe_id or (
        expected_recipe_id is not None and recipe_id != expected_recipe_id
    ):
        raise ValueError(f"response outcome recipe identity differs for {unit}")
    if not str(row.get("generated_utc", "")):
        raise ValueError(f"response outcome has no generation time for {unit}")
    if row.get("diagnosis_blinded") is not True:
        raise ValueError(f"response outcome is not diagnosis blinded for {unit}")
    if FORBIDDEN_GROUP_FIELDS.intersection(row):
        raise ValueError(f"Diagnosis/group fields are forbidden in outcome {unit}")
    row["source_identity_hashes"] = _required_source_hashes(
        row.get("source_identity_hashes", {})
    )

    identity = row.get("phase_a_identity")
    if not isinstance(identity, Mapping):
        raise ValueError(f"response outcome lacks phase-A identity for {unit}")
    run_id = str(identity.get("run_id", ""))
    attempt_id = str(identity.get("attempt_id", ""))
    if not run_id or (expected_run_id is not None and run_id != expected_run_id):
        raise ValueError(f"response outcome phase-A run differs for {unit}")
    if not attempt_id or (
        expected_attempt_id is not None and attempt_id != expected_attempt_id
    ):
        raise ValueError(f"response outcome phase-A attempt differs for {unit}")
    run_context_path = verify_file_record(
        identity.get("run_context", {}), label=f"{unit} phase-A run context"
    )
    attempt_context_path = verify_file_record(
        identity.get("attempt_context", {}), label=f"{unit} phase-A attempt context"
    )
    run_context = json.loads(run_context_path.read_text(encoding="utf-8"))
    attempt_context = json.loads(attempt_context_path.read_text(encoding="utf-8"))
    if run_context.get("run_id") != run_id or run_context.get("recipe_id") != recipe_id:
        raise ValueError(f"response outcome run-context identity differs for {unit}")
    if (
        attempt_context.get("attempt_id") != attempt_id
        or attempt_context.get("run_id") != run_id
        or attempt_context.get("recipe_id") != recipe_id
    ):
        raise ValueError(f"response outcome attempt-context identity differs for {unit}")

    status = row.get("status")
    if status == "PASS":
        responses = row.get("responses")
        if not isinstance(responses, Mapping) or set(responses) != set(TISSUES):
            raise ValueError(f"PASS response outcome lacks exact tissues for {unit}")
        for tissue in TISSUES:
            verify_file_record(
                responses[tissue], label=f"{unit} {tissue} response"
            )
        response_qc = row.get("response_qc")
        if (
            not isinstance(response_qc, Mapping)
            or response_qc.get("status") != "PASS"
            or response_qc.get("diagnosis_labels_used") is not False
            or response_qc.get("failure_reasons") != []
        ):
            raise ValueError(f"PASS response outcome lacks PASS response QC for {unit}")
        verify_file_record(
            row.get("response_voxels", {}),
            label=f"{unit} response-voxel selection",
        )
        shell_contract = row.get("fod_shell_contract")
        if not isinstance(shell_contract, Mapping):
            raise ValueError(f"PASS response lacks FOD shell identity for {unit}")
        selection_path = verify_file_record(
            shell_contract.get("selection_record", {}),
            label=f"{unit} FOD shell selection",
        )
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        selected_shells = shell_contract.get("selected_shells_s_per_mm2")
        selected_sizes = shell_contract.get("selected_shell_sizes")
        bzero_threshold = shell_contract.get("bzero_threshold_s_per_mm2")
        try:
            shell_values = [float(value) for value in selected_shells]
            threshold_value = float(bzero_threshold)
        except (TypeError, ValueError):
            shell_values = []
            threshold_value = float("nan")
        if (
            selection.get("status") != "PASS"
            or selection.get("unit") != unit
            or selection.get("selection_uses_diagnosis_labels") is not False
            or shell_contract.get("selection_uses_diagnosis_labels") is not False
            or selected_shells != selection.get("selected_shells_s_per_mm2")
            or selected_sizes != selection.get("selected_shell_sizes")
            or bzero_threshold != selection.get("bzero_threshold_s_per_mm2")
            or isinstance(bzero_threshold, bool)
            or len(shell_values) != 2
            or not np.isfinite(shell_values).all()
            or not np.isfinite(threshold_value)
            or not 0.0 <= shell_values[0] <= threshold_value < shell_values[1]
            or not isinstance(selected_sizes, list)
            or len(selected_sizes) != 2
            or any(not isinstance(value, int) or value < 1 for value in selected_sizes)
        ):
            raise ValueError(f"PASS response FOD shell identity differs for {unit}")
    elif status == "FAIL":
        reason = row.get("primary_failure_reason")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(f"FAIL response outcome lacks reason for {unit}")
        if row.get("responses") not in ({}, None):
            raise ValueError(f"FAIL response outcome cannot expose responses for {unit}")
        candidate_records = row.get("candidate_response_evidence", {})
        if candidate_records not in ({}, None):
            if not isinstance(candidate_records, Mapping) or set(
                candidate_records
            ) != set(TISSUES):
                raise ValueError(
                    f"FAIL response evidence lacks exact tissues for {unit}"
                )
            for tissue in TISSUES:
                verify_file_record(
                    candidate_records[tissue],
                    label=f"{unit} failed-candidate {tissue} response",
                )
        if row.get("response_voxels") is not None:
            verify_file_record(
                row["response_voxels"],
                label=f"{unit} failed response-voxel selection",
            )
    else:
        raise ValueError(f"response outcome has invalid status for {unit}: {status!r}")
    return row


def freeze_response_calibration(
    outcomes: Iterable[Mapping[str, Any]],
    output_dir: str | Path,
    *,
    minimum_valid_subjects: int,
    generated_utc: str,
    responsemean_executable: str | Path,
    expected_responsemean_sha256: str,
    phase_a_lineage: Mapping[str, Any] | None = None,
    unit_acquisition_metadata: Mapping[str, Mapping[str, Any]] | None = None,
    execution_subset_manifest: Mapping[str, Any] | None = None,
    minimum_valid_manufacturer_families: int | None = None,
    minimum_valid_t1_source_classes: int | None = None,
    required_valid_t1_source_classes: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Pool explicit PASS records; intended as a focused lower-level helper."""

    if minimum_valid_subjects < 1:
        raise ValueError("minimum_valid_subjects must be positive")
    rows = [validate_response_calibration_outcome(row) for row in outcomes]
    units = [str(row.get("unit", "")) for row in rows]
    if not rows or len(units) != len(set(units)):
        raise ValueError("Calibration outcomes require unique nonempty units")
    recipe_ids = {str(row["recipe_id"]) for row in rows}
    run_ids = {str(row["phase_a_identity"]["run_id"]) for row in rows}
    attempt_ids = {str(row["phase_a_identity"]["attempt_id"]) for row in rows}
    if len(recipe_ids) != 1 or len(run_ids) != 1 or len(attempt_ids) != 1:
        raise ValueError("Calibration outcomes mix recipe, run or attempt identities")

    valid = [row for row in rows if row.get("status") == "PASS"]
    invalid = [row for row in rows if row.get("status") != "PASS"]
    if len(valid) < minimum_valid_subjects:
        raise ValueError(
            f"Valid response count {len(valid)} is below prespecified minimum "
            f"{minimum_valid_subjects}"
        )
    diversity_arguments = (
        unit_acquisition_metadata,
        execution_subset_manifest,
        minimum_valid_manufacturer_families,
        minimum_valid_t1_source_classes,
        required_valid_t1_source_classes,
    )
    if phase_a_lineage is not None and any(
        value is None for value in diversity_arguments
    ):
        raise ValueError(
            "lineage-bound response freeze requires the signed H04A technical "
            "diversity contract and bound unit metadata"
        )
    valid_pool_diversity: dict[str, Any] | None = None
    if any(value is not None for value in diversity_arguments):
        if any(value is None for value in diversity_arguments):
            raise ValueError("response-freeze technical diversity inputs are incomplete")
        assert unit_acquisition_metadata is not None
        assert execution_subset_manifest is not None
        assert minimum_valid_manufacturer_families is not None
        assert minimum_valid_t1_source_classes is not None
        assert required_valid_t1_source_classes is not None
        valid_pool_diversity = build_valid_pool_technical_diversity(
            valid_units=(str(row["unit"]) for row in valid),
            unit_metadata=unit_acquisition_metadata,
            execution_subset_manifest=execution_subset_manifest,
            minimum_valid_manufacturer_families=(
                minimum_valid_manufacturer_families
            ),
            minimum_valid_t1_source_classes=minimum_valid_t1_source_classes,
            required_valid_t1_source_classes=required_valid_t1_source_classes,
        )
    executable = Path(responsemean_executable).resolve()
    if (
        not executable.is_file()
        or not SHA256_RE.fullmatch(expected_responsemean_sha256)
        or sha256_file(executable) != expected_responsemean_sha256
    ):
        raise ValueError("locked MRtrix responsemean executable differs")
    reference_shells = valid[0]["fod_shell_contract"][
        "selected_shells_s_per_mm2"
    ]
    reference_bzero_threshold = valid[0]["fod_shell_contract"][
        "bzero_threshold_s_per_mm2"
    ]
    if any(
        row["fod_shell_contract"]["selected_shells_s_per_mm2"]
        != reference_shells
        or row["fod_shell_contract"]["bzero_threshold_s_per_mm2"]
        != reference_bzero_threshold
        for row in valid
    ):
        raise ValueError(
            "valid response candidates have incompatible MRtrix shell centroids"
        )

    arrays: dict[str, list[np.ndarray]] = {tissue: [] for tissue in TISSUES}
    input_records: dict[str, dict[str, Any]] = {}
    for row in valid:
        unit_record: dict[str, Any] = {}
        for tissue in TISSUES:
            response_record = dict(row["responses"][tissue])
            path = Path(str(response_record["path"])).resolve()
            array = np.loadtxt(path, dtype=float, ndmin=2)
            if not np.isfinite(array).all():
                raise ValueError(
                    f"Non-finite response coefficients for {row['unit']} {tissue}"
                )
            arrays[tissue].append(array)
            response_record["shape"] = list(array.shape)
            unit_record[tissue] = response_record
        input_records[row["unit"]] = unit_record

    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    pooled_records: dict[str, dict[str, Any]] = {}
    pooling_commands: dict[str, dict[str, Any]] = {}
    for tissue in TISSUES:
        shapes = {array.shape for array in arrays[tissue]}
        if len(shapes) != 1:
            raise ValueError(f"{tissue} response shapes differ: {sorted(shapes)}")
        path = destination / f"pooled_response_{tissue}.txt"
        partial = destination / f"pooled_response_{tissue}.responsemean.partial.txt"
        partial.unlink(missing_ok=True)
        command = [
            str(executable),
            *[
                str(Path(row["responses"][tissue]["path"]).resolve())
                for row in valid
            ],
            str(partial),
        ]
        completed = subprocess.run(
            command, text=True, capture_output=True, check=False
        )
        if completed.returncode != 0 or not partial.is_file():
            raise RuntimeError(
                f"responsemean failed for {tissue} with exit {completed.returncode}: "
                + completed.stderr.strip()
            )
        _write_new(path, partial.read_bytes())
        partial.unlink()
        pooled = np.loadtxt(path, dtype=float, ndmin=2)
        if not np.isfinite(pooled).all():
            raise ValueError(f"responsemean produced non-finite {tissue} coefficients")
        pooled_records[tissue] = {
            **file_record(path),
            "shape": list(pooled.shape),
        }
        pooling_commands[tissue] = {
            "argv": command,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }

    lineage = dict(phase_a_lineage) if phase_a_lineage is not None else None
    manifest = {
        "schema_version": "2.0.0",
        "record_type": "frozen_response_calibration",
        "status": "PASS",
        "generated_utc": generated_utc,
        "method": "mrtrix_responsemean_nonlegacy_scale_compensated",
        "responsemean_executable": file_record(executable),
        "responsemean_legacy_option_used": False,
        "pooling_commands": pooling_commands,
        "diagnosis_labels_used": False,
        "minimum_valid_subjects": minimum_valid_subjects,
        "total_outcome_count": len(rows),
        "valid_subject_count": len(valid),
        "invalid_subject_count": len(invalid),
        "valid_units": sorted(row["unit"] for row in valid),
        "invalid_units": sorted(row["unit"] for row in invalid),
        "inputs": input_records,
        "pooled_responses": pooled_records,
        "fod_shell_compatibility": {
            "rule": "exact_MRtrix_clustered_b0_and_nonzero_centroids",
            "maximum_centroid_difference_s_per_mm2": 0.0,
            "selected_shells_s_per_mm2": reference_shells,
            "bzero_threshold_s_per_mm2": reference_bzero_threshold,
            "candidate_shell_sizes": {
                row["unit"]: row["fod_shell_contract"]["selected_shell_sizes"]
                for row in valid
            },
            "candidate_selection_records": {
                row["unit"]: row["fod_shell_contract"]["selection_record"]
                for row in valid
            },
        },
        "valid_pool_technical_diversity": valid_pool_diversity,
        "valid_pool_technical_diversity_sha256": (
            technical_diversity_sha256(valid_pool_diversity)
            if valid_pool_diversity is not None
            else None
        ),
        "phase_a_lineage": lineage,
        "release_ready_for_phase_b": (
            lineage is not None and valid_pool_diversity is not None
        ),
        "phase_b_live_all_units_dependency": False,
        "dummy_response_files_used": False,
    }
    manifest_path = destination / "frozen_response_calibration_manifest.json"
    _write_new(
        manifest_path,
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    manifest["manifest_path"] = str(manifest_path)
    manifest["manifest_sha256"] = sha256_file(manifest_path)
    return manifest


def freeze_response_calibration_from_phase_a(
    phase_a_completion_path: str | Path,
    phase_a_manifest_path: str | Path,
    output_dir: str | Path,
    *,
    minimum_valid_subjects: int,
    generated_utc: str,
    responsemean_executable: str | Path,
    expected_responsemean_sha256: str,
) -> dict[str, Any]:
    """Freeze exactly one terminally complete immutable phase-A publication."""

    completion_path = Path(phase_a_completion_path).resolve()
    manifest_path = Path(phase_a_manifest_path).resolve()
    completion_record = file_record(completion_path)
    manifest_record = file_record(manifest_path)
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    phase_a = json.loads(manifest_path.read_text(encoding="utf-8"))

    if (
        completion.get("schema_version") != "2.0.0"
        or completion.get("record_type")
        != "response_calibration_phase_a_completion"
        or completion.get("mode") != "response-calibration-phase-a"
    ):
        raise ValueError("phase-A completion contract differs")
    verify_file_record(
        completion.get("phase_a_manifest", {}),
        label="phase-A manifest declared by completion",
        expected_path=manifest_path,
    )
    if (
        phase_a.get("schema_version") != "2.0.0"
        or phase_a.get("record_type")
        != "response_calibration_phase_a_manifest"
        or phase_a.get("status") != "COMPLETE"
        or phase_a.get("mode") != "response-calibration-phase-a"
    ):
        raise ValueError("phase-A manifest is not terminally complete")
    run_id = str(phase_a.get("run_id", ""))
    recipe_id = str(phase_a.get("recipe_id", ""))
    if (
        not run_id
        or not recipe_id
        or completion.get("run_id") != run_id
        or completion.get("recipe_id") != recipe_id
    ):
        raise ValueError("phase-A completion and manifest identities differ")

    summary = phase_a.get("summary", {})
    expected_units = phase_a.get("expected_units")
    outcome_records = phase_a.get("outcome_records")
    if (
        not isinstance(expected_units, list)
        or len(expected_units) != len(set(expected_units))
        or not isinstance(outcome_records, Mapping)
        or set(outcome_records) != set(expected_units)
    ):
        raise ValueError("phase-A manifest outcome set is not exactly closed")
    expected_count = summary.get("expected")
    terminal_count = summary.get("terminal")
    if (
        not isinstance(expected_count, int)
        or not isinstance(terminal_count, int)
        or expected_count != terminal_count
        or expected_count != len(expected_units)
    ):
        raise ValueError("phase-A expected and terminal counts differ")

    attempt_start_path = verify_file_record(
        phase_a.get("attempt_start", {}), label="phase-A attempt start"
    )
    attempt_start = json.loads(attempt_start_path.read_text(encoding="utf-8"))
    attempt_id = str(attempt_start.get("attempt_id", ""))
    if (
        not attempt_id
        or attempt_start.get("run_id") != run_id
        or attempt_start.get("recipe_id") != recipe_id
    ):
        raise ValueError("phase-A attempt-start identity differs")

    execution_binding = phase_a.get("execution_binding")
    if (
        not isinstance(execution_binding, Mapping)
        or completion.get("execution_binding") != execution_binding
        or attempt_start.get("execution_binding") != execution_binding
        or execution_binding.get("units") != expected_units
    ):
        raise ValueError(
            "phase-A freeze inputs lack one exact execution-subset binding"
        )
    run_context_path = verify_file_record(
        phase_a.get("run_context", {}), label="phase-A run context"
    )
    run_context = json.loads(run_context_path.read_text(encoding="utf-8"))
    if (
        run_context.get("run_id") != run_id
        or run_context.get("recipe_id") != recipe_id
        or run_context.get("execution_binding") != execution_binding
        or attempt_start.get("run_context") != file_record(run_context_path)
    ):
        raise ValueError("phase-A run context differs from freeze binding")
    authorization = execution_binding.get("h04a_authorization")
    if not isinstance(authorization, Mapping):
        raise ValueError("phase-A execution binding lacks signed H04A authority")
    signed_minimum = authorization.get(
        "minimum_valid_response_calibration_units"
    )
    if (
        isinstance(signed_minimum, bool)
        or not isinstance(signed_minimum, int)
        or signed_minimum < 1
        or minimum_valid_subjects != signed_minimum
    ):
        raise ValueError(
            "response-freeze minimum differs from signed H04A valid-unit minimum"
        )
    diversity_contract = technical_diversity_contract_from_authorization(
        authorization
    )
    subset_manifest_record, unit_metadata = (
        load_execution_subset_technical_metadata(execution_binding)
    )

    outcomes: list[dict[str, Any]] = []
    outcome_paths: list[str] = []
    for unit in expected_units:
        outcome_path = verify_file_record(
            outcome_records[unit], label=f"phase-A outcome {unit}"
        )
        outcome = json.loads(outcome_path.read_text(encoding="utf-8"))
        outcomes.append(
            validate_response_calibration_outcome(
                outcome,
                expected_unit=unit,
                expected_recipe_id=recipe_id,
                expected_run_id=run_id,
                expected_attempt_id=attempt_id,
            )
        )
        outcome_paths.append(str(outcome_path))
    declared_paths = [
        str(Path(path).resolve())
        for path in phase_a.get("calibration_freeze_input_outcomes", [])
    ]
    if declared_paths != outcome_paths:
        raise ValueError("phase-A declared freeze inputs differ from exact outcomes")
    pass_count = sum(row["status"] == "PASS" for row in outcomes)
    fail_count = sum(row["status"] == "FAIL" for row in outcomes)
    if summary.get("pass") != pass_count or summary.get("fail") != fail_count:
        raise ValueError("phase-A outcome statuses differ from summary")

    lineage = {
        "run_id": run_id,
        "recipe_id": recipe_id,
        "completion": completion_record,
        "manifest": manifest_record,
        "attempt_start": file_record(attempt_start_path),
        "expected_outcome_count": expected_count,
        "terminal_outcome_count": terminal_count,
    }
    return freeze_response_calibration(
        outcomes,
        output_dir,
        minimum_valid_subjects=minimum_valid_subjects,
        generated_utc=generated_utc,
        responsemean_executable=responsemean_executable,
        expected_responsemean_sha256=expected_responsemean_sha256,
        phase_a_lineage=lineage,
        unit_acquisition_metadata=unit_metadata,
        execution_subset_manifest=subset_manifest_record,
        **diversity_contract,
    )


def validate_frozen_response_calibration(
    manifest_path: str | Path,
    *,
    expected_manifest_sha256: str,
    minimum_valid_subjects: int,
    expected_responses: Mapping[str, Mapping[str, Any]],
    expected_technical_diversity: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the exact, lineage-bound phase-B calibration artifact."""

    path = Path(manifest_path).resolve()
    if sha256_file(path) != expected_manifest_sha256:
        raise ValueError("Frozen response-calibration manifest SHA-256 mismatch")
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("status") != "PASS" or record.get("diagnosis_labels_used") is not False:
        raise ValueError("Frozen response calibration is not diagnosis-blind PASS evidence")
    if (
        record.get("method")
        != "mrtrix_responsemean_nonlegacy_scale_compensated"
        or record.get("responsemean_legacy_option_used") is not False
    ):
        raise ValueError("Frozen response pooling method differs from locked responsemean")
    verify_file_record(
        record.get("responsemean_executable", {}),
        label="frozen responsemean executable",
    )
    shell_compatibility = record.get("fod_shell_compatibility")
    if (
        not isinstance(shell_compatibility, Mapping)
        or shell_compatibility.get("rule")
        != "exact_MRtrix_clustered_b0_and_nonzero_centroids"
        or shell_compatibility.get("maximum_centroid_difference_s_per_mm2") != 0.0
        or not isinstance(
            shell_compatibility.get("selected_shells_s_per_mm2"), list
        )
        or len(shell_compatibility["selected_shells_s_per_mm2"]) != 2
        or not isinstance(
            shell_compatibility.get("bzero_threshold_s_per_mm2"), (int, float)
        )
        or not 0.0
        <= float(shell_compatibility["selected_shells_s_per_mm2"][0])
        <= float(shell_compatibility["bzero_threshold_s_per_mm2"])
        or float(shell_compatibility["selected_shells_s_per_mm2"][1])
        <= float(shell_compatibility["bzero_threshold_s_per_mm2"])
    ):
        raise ValueError("Frozen response calibration lacks exact FOD shell compatibility")
    if record.get("release_ready_for_phase_b") is not True:
        raise ValueError("Frozen response calibration lacks complete phase-A lineage")
    lineage = record.get("phase_a_lineage")
    if not isinstance(lineage, Mapping):
        raise ValueError("Frozen response calibration lacks phase-A lineage")
    for key in ("completion", "manifest", "attempt_start"):
        verify_file_record(lineage.get(key, {}), label=f"frozen lineage {key}")
    if (
        lineage.get("expected_outcome_count")
        != lineage.get("terminal_outcome_count")
        or lineage.get("expected_outcome_count") != record.get("total_outcome_count")
    ):
        raise ValueError("Frozen response calibration lineage is not terminally complete")
    if int(record.get("minimum_valid_subjects", 0)) != int(minimum_valid_subjects):
        raise ValueError("Frozen response minimum N differs from recipe")
    if int(record.get("valid_subject_count", 0)) < int(minimum_valid_subjects):
        raise ValueError("Frozen response calibration is below minimum N")
    observed_diversity = record.get("valid_pool_technical_diversity")
    if (
        not isinstance(expected_technical_diversity, Mapping)
        or observed_diversity != expected_technical_diversity
    ):
        raise ValueError("Frozen response technical diversity differs from binding")
    diversity_contract = technical_diversity_contract_from_authorization(
        expected_technical_diversity
    )
    validated_diversity = validate_valid_pool_technical_diversity(
        observed_diversity,
        expected_valid_units=record.get("valid_units", []),
        expected_execution_subset_manifest=expected_technical_diversity.get(
            "execution_subset_manifest", {}
        ),
        **diversity_contract,
    )
    if (
        record.get("valid_pool_technical_diversity_sha256")
        != technical_diversity_sha256(validated_diversity)
    ):
        raise ValueError("Frozen response technical-diversity digest differs")
    if record.get("phase_b_live_all_units_dependency") is not False:
        raise ValueError("Phase B calibration unexpectedly depends on live cohort units")
    if record.get("dummy_response_files_used") is not False:
        raise ValueError("Dummy response files are forbidden")
    for tissue in TISSUES:
        expected = expected_responses[tissue]
        observed = record.get("pooled_responses", {}).get(tissue, {})
        configured_path = Path(str(expected["path"])).resolve()
        if Path(str(observed.get("path", ""))).resolve() != configured_path:
            raise ValueError(f"Frozen {tissue} response path differs from recipe")
        configured_hash = str(expected["sha256"])
        if (
            observed.get("sha256") != configured_hash
            or sha256_file(configured_path) != configured_hash
        ):
            raise ValueError(f"Frozen {tissue} response SHA-256 mismatch")
        if expected.get("size_bytes") is not None and (
            observed.get("size_bytes") != expected.get("size_bytes")
            or configured_path.stat().st_size != expected.get("size_bytes")
        ):
            raise ValueError(f"Frozen {tissue} response size mismatch")
    return record
