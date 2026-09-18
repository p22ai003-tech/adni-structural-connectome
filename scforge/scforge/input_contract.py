"""Closed-world acquisition-manifest and source-normalization helpers.

The acquisition manifest describes immutable *raw* source bundles.  It never
pretends that those bundle digests are hashes of normalized files.  Normalized
MIF/NIfTI/gradient artifacts receive new hashes in per-unit records after
conversion.

Missing phase-encoding direction or total readout time is accepted by the
loader so that every locally available case remains in the 530-case processing
and attrition denominator.  :func:`require_eddy_metadata` is the explicit
fail-closed runtime gate; it provides no inferred or hard-coded default.
"""

from __future__ import annotations

import csv
import copy
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

from jsonschema import FormatChecker

try:  # The locked workflow uses jsonschema 4.x; metadata builders may use host 3.x.
    from jsonschema import Draft202012Validator as _SchemaValidator
except ImportError:  # pragma: no cover - exercised only by the host metadata interpreter
    from jsonschema import Draft7Validator as _SchemaValidator


SHA256_RE = re.compile(r"[0-9a-f]{64}")
SAFE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
SUPPORTED_PE_DIRECTIONS = frozenset({"i", "i-", "j", "j-", "k", "k-"})
NULLABLE_STRING_FIELDS = frozenset(
    {
        "diagnosis_at_dti",
        "dti_dicom_series_uid",
        "dwi_nifti_path",
        "dwi_bvec_path",
        "dwi_bval_path",
        "dwi_json_path",
        "dwi_nifti_sha256",
        "dwi_bvec_sha256",
        "dwi_bval_sha256",
        "dwi_json_sha256",
        "t1_dicom_series_uid",
        "phase",
        "site",
        "manufacturer",
        "scanner_model",
        "protocol",
        "phase_encoding_direction",
        "phase_encoding_source",
        "total_readout_time_source",
    }
)
INTEGER_FIELDS = frozenset(
    {
        "dti_raw_file_count",
        "dti_raw_total_bytes",
        "t1_raw_file_count",
        "t1_raw_total_bytes",
        "abs_pair_gap_days",
    }
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_unit(row: Mapping[str, str]) -> str:
    subject = str(row.get("subject_id", "")).strip()
    image_id = str(row.get("dti_image_id", "")).strip()
    if not SAFE_ID_RE.fullmatch(subject):
        raise ValueError(f"Unsafe subject_id in manifest: {subject!r}")
    if not image_id.isdigit():
        raise ValueError(f"dti_image_id must be numeric for {subject}: {image_id!r}")
    return f"{subject}_I{image_id}"


def validate_resolved_runtime_config(
    normative_config: Mapping[str, Any],
    resolved_config: Mapping[str, Any],
    run_context: Mapping[str, Any],
) -> None:
    """Permit only attested launcher/runtime injection into a resolved config."""

    normative = copy.deepcopy(dict(normative_config))
    resolved = copy.deepcopy(dict(resolved_config))
    runtime_keys = {
        "manifest_path",
        "run_root",
        "human_qc_manifest",
        "run_context_path",
        "attempt_context_path",
        "resolved_run_config_path",
        "launcher_mode",
        "execution_manifest_path",
    }
    mode = str(resolved.get("launcher_mode", ""))
    if mode not in {
        "response-calibration-phase-a",
        "pre-tractography-canary",
        "phase-b",
    }:
        raise ValueError(f"resolved launcher_mode is invalid: {mode!r}")
    if run_context.get("launcher_mode") != mode:
        raise ValueError("resolved launcher_mode differs from locked run context")
    execution_manifest_path = resolved.get("execution_manifest_path")
    resolved_run_root = resolved.get("run_root")
    for key in runtime_keys:
        normative.pop(key, None)
        resolved.pop(key, None)

    normative_inputs = normative.get("inputs", {}).get("approved_pair_manifest")
    resolved_inputs = resolved.get("inputs", {}).get("approved_pair_manifest")
    if not isinstance(normative_inputs, Mapping) or not isinstance(
        resolved_inputs, Mapping
    ):
        raise ValueError("approved_pair_manifest contract is missing")
    resolved["inputs"]["approved_pair_manifest"] = copy.deepcopy(normative_inputs)

    execution_binding = resolved.pop("execution_binding", None)
    locked_execution_binding = run_context.get("execution_binding")
    h04a_authorization = (
        execution_binding.get("h04a_authorization")
        if isinstance(execution_binding, Mapping)
        else None
    )
    h04a_expected = {
        "approval_mode": "H04A_BOUNDED_CANARY",
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
        "wall_clock_stop_seconds": 72 * 60 * 60,
        "storage_stop_gb": 150,
        "storage_stop_bytes": 150 * 1_000_000_000,
        "tractography_authorized": False,
        "matrix_generation_authorized": False,
        "full_cohort_authorized": False,
    }
    h04a_semantics_valid = isinstance(h04a_authorization, Mapping)
    if h04a_semantics_valid:
        h04a_semantics_valid = all(
            h04a_authorization.get(key) == value
            for key, value in h04a_expected.items()
        ) and all(
            isinstance(h04a_authorization.get(key), str)
            and bool(h04a_authorization.get(key))
            for key in (
                "approved_by",
                "approved_utc",
                "user_response",
                "normative_config_sha256",
                "workflow_source_manifest_sha256",
                "environment_contract_sha256",
            )
        )
        h04a_semantics_valid = bool(
            h04a_semantics_valid
            and h04a_authorization.get("proposed_run_root") == resolved_run_root
        )
    if (
        not isinstance(execution_binding, Mapping)
        or execution_binding != locked_execution_binding
        or execution_binding.get("binding_type")
        != "connectome_execution_subset_binding"
        or execution_binding.get("execution_scope") != "canary"
        or execution_binding.get("recipe_id")
        != normative.get("contract", {}).get("recipe_id")
        or execution_binding.get("diagnosis_labels_used") is not False
        or execution_binding.get("selection_locked") is not True
        or execution_manifest_path
        != execution_binding.get("execution_subset_manifest", {}).get("path")
        or execution_binding.get("approved_unit_count")
        != len(execution_binding.get("units", []))
        or not 12 <= execution_binding.get("approved_unit_count", 0) <= 24
        or execution_binding.get("units")
        != sorted(execution_binding.get("units", []))
        or not h04a_semantics_valid
    ):
        raise ValueError("resolved canary execution binding semantics differ")
    if mode in {
        "response-calibration-phase-a",
        "pre-tractography-canary",
    } and mode not in h04a_authorization["authorized_modes"]:
        raise ValueError("resolved H04A launcher mode is not signed")
    if mode == "phase-b" and mode in h04a_authorization["authorized_modes"]:
        raise ValueError("H04A authorization must not include phase-b")

    binding = resolved.pop("response_calibration_binding", None)
    if mode == "response-calibration-phase-a":
        if binding is not None or "response_calibration_binding" in run_context:
            raise ValueError("phase A cannot contain a phase-B calibration binding")
    else:
        locked_binding = run_context.get("response_calibration_binding")
        if not isinstance(binding, Mapping) or binding != locked_binding:
            raise ValueError(
                "resolved calibration binding differs from locked run context"
            )
        if (
            binding.get("binding_type")
            != "response_calibration_phase_b_binding"
            or binding.get("recipe_id")
            != normative.get("contract", {}).get("recipe_id")
            or binding.get("diagnosis_labels_used") is not False
            or binding.get("phase_b_live_all_units_dependency") is not False
            or binding.get("dummy_response_files_used") is not False
        ):
            raise ValueError("response calibration binding semantics differ")
        valid_units = binding.get("valid_units")
        diversity = binding.get("valid_pool_technical_diversity")
        diversity_digest = binding.get("valid_pool_technical_diversity_sha256")
        raw_counts = (
            diversity.get("manufacturer_counts")
            if isinstance(diversity, Mapping)
            else None
        )
        family_counts = (
            diversity.get("manufacturer_family_counts")
            if isinstance(diversity, Mapping)
            else None
        )
        t1_counts = (
            diversity.get("t1_source_class_counts")
            if isinstance(diversity, Mapping)
            else None
        )
        valid_unit_metadata = (
            diversity.get("valid_unit_metadata")
            if isinstance(diversity, Mapping)
            else None
        )
        if (
            not isinstance(valid_units, list)
            or not valid_units
            or valid_units != sorted(set(valid_units))
            or not set(valid_units).issubset(set(execution_binding.get("units", [])))
            or len(valid_units)
            < h04a_authorization["minimum_valid_response_calibration_units"]
            or binding.get("minimum_valid_subjects")
            != h04a_authorization["minimum_valid_response_calibration_units"]
            or not isinstance(diversity, Mapping)
            or diversity.get("status") != "PASS"
            or diversity.get("diagnosis_labels_used") is not False
            or diversity.get("execution_subset_manifest")
            != execution_binding.get("execution_subset_manifest")
            or diversity.get("minimum_valid_manufacturer_families")
            != h04a_authorization["minimum_valid_manufacturer_families"]
            or diversity.get("minimum_valid_t1_source_classes")
            != h04a_authorization["minimum_valid_t1_source_classes"]
            or diversity.get("required_valid_t1_source_classes")
            != h04a_authorization["required_valid_t1_source_classes"]
            or diversity.get("valid_unit_count") != len(valid_units)
            or diversity.get("manufacturer_family_count", 0)
            < h04a_authorization["minimum_valid_manufacturer_families"]
            or not isinstance(raw_counts, Mapping)
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 1
                for value in raw_counts.values()
            )
            or sum(raw_counts.values()) != len(valid_units)
            or not isinstance(family_counts, Mapping)
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 1
                for value in family_counts.values()
            )
            or len(family_counts)
            != diversity.get("manufacturer_family_count")
            or sum(family_counts.values()) != len(valid_units)
            or diversity.get("t1_source_class_count")
            != h04a_authorization["minimum_valid_t1_source_classes"]
            or not isinstance(t1_counts, Mapping)
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 1
                for value in t1_counts.values()
            )
            or sorted(t1_counts)
            != h04a_authorization["required_valid_t1_source_classes"]
            or sum(t1_counts.values()) != len(valid_units)
            or not isinstance(valid_unit_metadata, Mapping)
            or set(valid_unit_metadata) != set(valid_units)
            or diversity_digest
            != hashlib.sha256(
                json.dumps(
                    diversity, sort_keys=True, separators=(",", ":")
                ).encode("utf-8")
            ).hexdigest()
        ):
            raise ValueError(
                "response calibration valid-pool technical diversity differs"
            )
        resolved_calibration = resolved["fod"]["response_estimation"][
            "calibration"
        ]
        normative_calibration = normative["fod"]["response_estimation"][
            "calibration"
        ]
        expected_injections = {
            "minimum_valid_subjects": binding.get("minimum_valid_subjects"),
            "frozen_manifest": binding.get("response_calibration_manifest"),
            "pooled_responses": binding.get("pooled_responses"),
        }
        for key, expected in expected_injections.items():
            if resolved_calibration.get(key) != expected:
                raise ValueError(
                    f"resolved calibration differs from binding at {key}"
                )
            resolved_calibration[key] = copy.deepcopy(normative_calibration[key])
    continuation = resolved.pop("tractography_continuation_binding", None)
    locked_continuation = run_context.get("tractography_continuation_binding")
    if mode == "phase-b":
        approved_units = (
            continuation.get("approved_units", [])
            if isinstance(continuation, Mapping)
            else []
        )
        execution_units = execution_binding.get("units", [])
        if (
            not isinstance(continuation, Mapping)
            or continuation != locked_continuation
            or continuation.get("binding_type")
            != "tractography_continuation_binding"
            or continuation.get("execution_scope") != "canary"
            or continuation.get("recipe_id")
            != normative.get("contract", {}).get("recipe_id")
            or continuation.get("diagnosis_labels_used") is not False
            or continuation.get("execution_subset_manifest")
            != execution_binding.get("execution_subset_manifest")
            or continuation.get("approved_unit_count")
            != len(approved_units)
            or not approved_units
            or any(not isinstance(unit, str) or not unit for unit in approved_units)
            or approved_units != sorted(set(approved_units))
            or not set(approved_units).issubset(execution_units)
            or continuation.get("valid_pool_technical_diversity")
            != binding.get("valid_pool_technical_diversity")
            or continuation.get("valid_pool_technical_diversity_sha256")
            != binding.get("valid_pool_technical_diversity_sha256")
        ):
            raise ValueError("tractography continuation binding semantics differ")
    elif continuation is not None or locked_continuation is not None:
        raise ValueError("pre-tractography modes cannot contain continuation binding")
    if resolved != normative:
        raise ValueError(
            "resolved run config changes scientific fields outside attested runtime overrides"
        )


def _to_schema_record(row: Mapping[str, str]) -> dict[str, Any]:
    record: dict[str, Any] = {}
    for key, value in row.items():
        stripped = str(value or "").strip()
        if key == "processing_authorized":
            if stripped.lower() not in {"true", "1", "yes"}:
                record[key] = False
            else:
                record[key] = True
        elif key in INTEGER_FIELDS:
            try:
                record[key] = int(stripped)
            except ValueError:
                record[key] = stripped
        elif key == "field_strength_t":
            if not stripped:
                record[key] = None
            else:
                try:
                    record[key] = float(stripped)
                except ValueError:
                    record[key] = stripped
        elif key == "total_readout_time":
            if not stripped:
                record[key] = None
            else:
                try:
                    record[key] = float(stripped)
                except ValueError:
                    record[key] = stripped
        elif key in NULLABLE_STRING_FIELDS and not stripped:
            record[key] = None
        else:
            record[key] = stripped
    return record


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def raw_source_paths(row: Mapping[str, str], modality: str) -> tuple[Path, ...]:
    """Return every raw path needed to normalize one modality."""

    if modality == "dwi":
        kind = row["dti_source_kind"]
        if kind == "dicom_series":
            return (Path(row["dti_source_path"]),)
        if kind == "nifti_bundle":
            return tuple(
                Path(row[key])
                for key in (
                    "dwi_nifti_path",
                    "dwi_bvec_path",
                    "dwi_bval_path",
                    "dwi_json_path",
                )
            )
        raise ValueError(f"Unsupported DWI source kind: {kind!r}")
    if modality == "t1":
        if row["t1_source_kind"] not in {"dicom_series", "nifti_single"}:
            raise ValueError(f"Unsupported T1 source kind: {row['t1_source_kind']!r}")
        return (Path(row["t1_source_path"]),)
    raise ValueError(f"Unsupported modality: {modality!r}")


def validate_source_shapes(row: Mapping[str, str], run_root: Path) -> list[str]:
    """Validate path kinds and safety without re-hashing a 92-GiB cohort."""

    failures: list[str] = []
    for modality in ("dwi", "t1"):
        try:
            paths = raw_source_paths(row, modality)
        except Exception as exc:
            failures.append(f"{modality} source declaration: {type(exc).__name__}:{exc}")
            continue
        for path in paths:
            if _is_relative_to(path, run_root):
                failures.append(f"{modality} raw source is inside run_root: {path}")
        primary = paths[0]
        kind = row["dti_source_kind"] if modality == "dwi" else row["t1_source_kind"]
        if kind == "dicom_series":
            if not primary.is_dir():
                failures.append(f"{modality} DICOM series directory missing: {primary}")
        else:
            for path in paths:
                if not path.is_file():
                    failures.append(f"{modality} source file missing: {path}")
    if row.get("source_lock_status") != "COMPLETE_SHA256":
        failures.append("raw source bundle is not content-hash locked")
    return failures


def load_acquisition_manifest(
    path: str | Path,
    schema_path: str | Path,
    *,
    run_root: str | Path,
    expected_rows: int | None = None,
) -> tuple[dict[str, str], ...]:
    """Load and validate the closed-world CSV without diagnosis/gap filtering."""

    manifest_path = Path(path)
    schema_file = Path(schema_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Acquisition manifest does not exist: {manifest_path}")
    schema = json.loads(schema_file.read_text(encoding="utf-8"))
    validator = _SchemaValidator(schema, format_checker=FormatChecker())
    with manifest_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = tuple(reader.fieldnames or ())
        required = tuple(schema.get("required", ()))
        missing = sorted(set(required) - set(fields))
        extras = sorted(set(fields) - set(schema.get("properties", {})))
        if missing:
            raise ValueError(f"Manifest is missing required columns: {missing}")
        if extras:
            raise ValueError(f"Manifest has columns outside acquisition schema v2: {extras}")
        rows: list[dict[str, str]] = []
        for line_number, raw in enumerate(reader, start=2):
            row = {key: str(value or "").strip() for key, value in raw.items()}
            errors = sorted(
                validator.iter_errors(_to_schema_record(row)),
                key=lambda error: tuple(str(item) for item in error.path),
            )
            if errors:
                details = "; ".join(
                    f"{'.'.join(str(item) for item in error.path) or '<row>'}: {error.message}"
                    for error in errors
                )
                raise ValueError(f"Manifest line {line_number} violates acquisition schema v2: {details}")
            row["unit"] = stable_unit(row)
            path_failures = validate_source_shapes(row, Path(run_root))
            if path_failures:
                raise ValueError(
                    f"Manifest line {line_number} has invalid raw sources: "
                    + "; ".join(path_failures)
                )
            rows.append(row)
    if not rows:
        raise ValueError("Acquisition manifest is empty")
    if expected_rows is not None and len(rows) != expected_rows:
        raise ValueError(
            f"Closed-world manifest must contain {expected_rows} rows, found {len(rows)}"
        )
    units = [row["unit"] for row in rows]
    subjects = [row["subject_id"] for row in rows]
    t1_source_ids = [row["t1_source_id"] for row in rows]
    if len(units) != len(set(units)):
        raise ValueError("Manifest contains duplicate subject/DTI units")
    if len(subjects) != len(set(subjects)):
        raise ValueError("Closed-world workflow requires one DTI source per subject")
    if len(t1_source_ids) != len(set(t1_source_ids)):
        raise ValueError("t1_source_id must be stable and unique within the manifest")
    return tuple(rows)


def validate_execution_subset(
    parent_rows: Iterable[Mapping[str, str]],
    execution_rows: Iterable[Mapping[str, str]],
    *,
    minimum_rows: int = 12,
    maximum_rows: int = 24,
) -> tuple[dict[str, str], ...]:
    """Require an exact, bounded row-for-row canary subset of the parent 530."""

    parent = {str(row["unit"]): dict(row) for row in parent_rows}
    subset = tuple(dict(row) for row in execution_rows)
    if not minimum_rows <= len(subset) <= maximum_rows:
        raise ValueError(
            f"canary execution subset must contain {minimum_rows}..{maximum_rows} "
            f"rows, found {len(subset)}"
        )
    units = [str(row["unit"]) for row in subset]
    if len(units) != len(set(units)):
        raise ValueError("canary execution subset contains duplicate units")
    for row in subset:
        unit = str(row["unit"])
        if unit not in parent or row != parent[unit]:
            raise ValueError(
                f"canary execution row is not byte-field-equivalent to parent: {unit}"
            )
    return subset


def require_eddy_metadata(row: Mapping[str, str]) -> tuple[str, float]:
    """Return source-derived eddy metadata, or fail with no fallback."""

    unit = row.get("unit") or stable_unit(row)
    direction = str(row.get("phase_encoding_direction", "")).strip()
    readout_text = str(row.get("total_readout_time", "")).strip()
    missing: list[str] = []
    if not direction:
        missing.append("phase_encoding_direction")
    if not readout_text:
        missing.append("total_readout_time")
    if missing:
        raise ValueError(
            f"{unit} cannot enter eddy: missing source metadata {', '.join(missing)}; no default is allowed"
        )
    if direction not in SUPPORTED_PE_DIRECTIONS:
        raise ValueError(f"{unit} has unsupported phase_encoding_direction: {direction!r}")
    try:
        readout = float(readout_text)
    except ValueError as exc:
        raise ValueError(f"{unit} has invalid total_readout_time: {readout_text!r}") from exc
    if not 0.0 < readout < 1.0:
        raise ValueError(f"{unit} has implausible total_readout_time: {readout}")
    return direction, readout


def source_phase_encoding_token(value: Any) -> str:
    """Accept only an authoritative MRtrix/BIDS image-axis PE token.

    Scanner/world vectors and anatomical abbreviations are not remapped here:
    polarity depends on the converted image orientation, so guessing would be
    scientifically unsafe.
    """

    if not isinstance(value, str):
        raise ValueError(
            f"PhaseEncodingDirection must be an image-axis string, found {type(value).__name__}"
        )
    token = value.strip()
    if token not in SUPPORTED_PE_DIRECTIONS:
        raise ValueError(
            f"Unsupported PhaseEncodingDirection representation {token!r}; no polarity mapping is allowed"
        )
    return token


def select_model_shells(
    shell_centroids: Iterable[float],
    *,
    b0_threshold: float,
    fod_target: int,
    fod_tolerance: int,
    tensor_maximum: int,
) -> dict[str, Any]:
    """Select diagnosis-blind model shells from MRtrix-clustered centroids."""

    values = sorted({float(value) for value in shell_centroids})
    if not values or values[0] > b0_threshold:
        raise ValueError(
            f"MRtrix shell centroids contain no b0 shell at or below {b0_threshold}"
        )
    nonzero_shells = [value for value in values if value > b0_threshold]
    eligible = [
        shell
        for shell in nonzero_shells
        if abs(shell - fod_target) <= fod_tolerance
    ]
    if not eligible:
        raise ValueError(
            f"no nonzero shell is within {fod_tolerance} of b={fod_target}: "
            f"{nonzero_shells}"
        )
    distance = min(abs(shell - fod_target) for shell in eligible)
    nearest = [shell for shell in eligible if abs(shell - fod_target) == distance]
    if len(nearest) != 1:
        raise ValueError(f"FOD target-shell selection is tied: {nearest}")
    tensor_shells = [shell for shell in nonzero_shells if shell <= tensor_maximum]
    if not tensor_shells:
        raise ValueError(
            f"no nonzero shell is at or below tensor maximum b={tensor_maximum}"
        )
    return {
        "observed_nonzero_shells": nonzero_shells,
        "fod_shells": [min(values), nearest[0]],
        "tensor_shells": [min(values), *tensor_shells],
        "selection_uses_diagnosis_labels": False,
    }


def tensor_design_diagnostics(vectors: Iterable[Iterable[float]]) -> dict[str, Any]:
    """Return rank/conditioning of the six-parameter tensor design."""

    import numpy as np

    array = np.asarray(list(vectors), dtype=float)
    if array.ndim != 2 or array.shape[1] != 3 or not np.isfinite(array).all():
        raise ValueError("tensor directions must be a finite Nx3 array")
    norms = np.linalg.norm(array, axis=1)
    if np.any(norms <= 0.0):
        raise ValueError("tensor directions contain a zero vector")
    g = array / norms[:, None]
    design = np.column_stack(
        (
            g[:, 0] ** 2,
            g[:, 1] ** 2,
            g[:, 2] ** 2,
            2.0 * g[:, 0] * g[:, 1],
            2.0 * g[:, 0] * g[:, 2],
            2.0 * g[:, 1] * g[:, 2],
        )
    )
    singular_values = np.linalg.svd(design, compute_uv=False)
    rank = int(np.linalg.matrix_rank(design))
    condition = (
        float(singular_values[0] / singular_values[-1])
        if singular_values.size and singular_values[-1] > 0.0
        else float("inf")
    )
    return {
        "rank": rank,
        "required_rank": 6,
        "condition_number": condition,
        "singular_values": [float(value) for value in singular_values],
        "row_count": int(design.shape[0]),
    }


def assess_atlas_label_support(
    voxel_counts: Mapping[int | str, int],
    *,
    expected_nodes: int,
    hard_minimum_voxels: int = 1,
) -> dict[str, Any]:
    """Validate only source-compatible atlas label survival.

    Parcel size is retained quantitatively for blinded H04 calibration.  It is
    not used as an outcome-dependent inclusion threshold.  The hard contract
    rejects only malformed/missing labels: every predeclared node must retain
    at least one voxel on the workflow grid.
    """

    if expected_nodes < 1 or hard_minimum_voxels != 1:
        raise ValueError(
            "atlas hard support is fixed to one surviving voxel; stronger "
            "thresholds require a separate diagnosis-blind H04 decision"
        )
    expected = list(range(1, expected_nodes + 1))
    normalized: dict[int, int] = {}
    failures: list[str] = []
    for raw_label, raw_count in voxel_counts.items():
        try:
            label = int(raw_label)
        except (TypeError, ValueError):
            failures.append(f"invalid_label_key:{raw_label!r}")
            continue
        if isinstance(raw_count, bool) or not isinstance(raw_count, int):
            failures.append(f"node_{label}:voxel_count_not_integer")
            continue
        normalized[label] = raw_count
    unexpected = sorted(set(normalized).difference(expected))
    missing = sorted(set(expected).difference(normalized))
    if unexpected:
        failures.append("unexpected_nodes:" + ",".join(map(str, unexpected)))
    if missing:
        failures.append("missing_nodes:" + ",".join(map(str, missing)))
    non_surviving = [
        label for label in expected if normalized.get(label, 0) < hard_minimum_voxels
    ]
    if non_surviving:
        failures.append("nodes_without_voxel_support:" + ",".join(map(str, non_surviving)))
    ordered_counts = {str(label): normalized.get(label, 0) for label in expected}
    observed = [normalized[label] for label in expected if label in normalized]
    return {
        "status": "PASS" if not failures else "FAIL",
        "hard_minimum_voxels": hard_minimum_voxels,
        "hard_rule": "exact_contiguous_labels_each_with_at_least_one_voxel",
        "stronger_size_threshold_status": "PENDING_DIAGNOSIS_BLIND_H04_DECISION",
        "voxel_counts": ordered_counts,
        "minimum_observed_voxels": min(observed) if observed else None,
        "maximum_observed_voxels": max(observed) if observed else None,
        "total_labeled_voxels": sum(observed),
        "failures": failures,
    }


def assess_pre_tractography_image_ranges(
    ranges: Mapping[str, Mapping[str, float]],
    *,
    nonnegative_tolerance: float = 1e-12,
    range_tolerance: float = 1e-8,
    fa_minimum: float = 0.0,
    fa_maximum: float = 1.0,
    diffusivity_minimum: float = 0.0,
    diffusivity_maximum: float = 0.01,
) -> list[str]:
    """Return hard technical failures for automated pre-tractography images."""

    if nonnegative_tolerance < 0.0 or range_tolerance < 0.0:
        raise ValueError("image-range tolerances must be nonnegative")
    failures: list[str] = []
    required = ("wmfod_all_sh_coefficients", "wmfod_l0", "gm", "csf", "fa", "md", "rd", "ad")
    normalized: dict[str, tuple[float, float]] = {}
    for name in required:
        record = ranges.get(name)
        if not isinstance(record, Mapping):
            failures.append(f"missing_range:{name}")
            continue
        try:
            minimum = float(record["min"])
            maximum = float(record["max"])
        except (KeyError, TypeError, ValueError):
            failures.append(f"invalid_range:{name}")
            continue
        if not (math.isfinite(minimum) and math.isfinite(maximum)):
            failures.append(f"nonfinite_range:{name}")
            continue
        if minimum > maximum:
            failures.append(f"inverted_range:{name}")
            continue
        normalized[name] = (minimum, maximum)
    if "wmfod_l0" in normalized:
        minimum, maximum = normalized["wmfod_l0"]
        if minimum < -nonnegative_tolerance or maximum <= nonnegative_tolerance:
            failures.append("wmfod_l0:negative_or_all_zero")
    for name in ("gm", "csf"):
        if name in normalized and normalized[name][0] < -nonnegative_tolerance:
            failures.append(f"{name}:negative")
    if "fa" in normalized:
        minimum, maximum = normalized["fa"]
        if (
            minimum < fa_minimum - range_tolerance
            or maximum > fa_maximum + range_tolerance
        ):
            failures.append("fa:outside_0_to_1")
    for name in ("md", "rd", "ad"):
        if name in normalized:
            minimum, maximum = normalized[name]
            if (
                minimum < diffusivity_minimum - range_tolerance
                or maximum > diffusivity_maximum + range_tolerance
            ):
                failures.append(f"{name}:outside_0_to_0.01_mm2_per_s")
    return list(dict.fromkeys(failures))


def staged_hashes(paths: Mapping[str, str | Path]) -> dict[str, dict[str, Any]]:
    """Hash normalized files; these digests are distinct from raw bundles."""

    result: dict[str, dict[str, Any]] = {}
    for name, raw_path in sorted(paths.items()):
        path = Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(f"Staged input missing: {path}")
        result[name] = {
            "path": str(path.resolve()),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return result


def bundle_sha256(records: Iterable[tuple[str, int, str]]) -> str:
    """Hash a deterministic framed list of already-hashed bundle members."""

    digest = hashlib.sha256()
    for relative_name, size_bytes, member_hash in sorted(records):
        if not SHA256_RE.fullmatch(member_hash):
            raise ValueError(f"Invalid member SHA-256 for {relative_name!r}")
        digest.update(relative_name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(int(size_bytes)).encode("ascii"))
        digest.update(b"\0")
        digest.update(member_hash.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def verify_runtime_source_inventory(
    row: Mapping[str, str],
    modality: str,
    inventory_record_path: str | Path,
) -> dict[str, Any]:
    """Verify the exact locked members immediately before/after conversion.

    Matching device/inode/size/mtime/ctime avoids re-reading unchanged image
    payloads.  Any changed or ambiguous identity triggers a full SHA-256 check;
    changed content fails closed.
    """

    record_path = Path(inventory_record_path).resolve()
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if record.get("unit") != row.get("unit") or record.get("modality") != modality:
        raise ValueError("runtime source inventory unit/modality differs")
    bundle_key = (
        "dti_raw_bundle_sha256" if modality == "dwi" else "t1_raw_bundle_sha256"
    )
    expected_bundle = str(row[bundle_key]).lower()
    if record.get("locked_bundle_sha256") != expected_bundle:
        raise ValueError("runtime source inventory bundle differs from manifest")
    members = record.get("members")
    if not isinstance(members, list) or not members:
        raise ValueError("runtime source inventory has no members")
    declared_paths = {Path(str(item.get("absolute_path", ""))).resolve() for item in members}
    source_paths = raw_source_paths(row, modality)
    kind = row["dti_source_kind"] if modality == "dwi" else row["t1_source_kind"]
    if kind == "dicom_series":
        current_paths = {path.resolve() for path in source_paths[0].rglob("*") if path.is_file()}
    else:
        current_paths = {path.resolve() for path in source_paths}
    if current_paths != declared_paths:
        raise ValueError("runtime raw source member set differs from locked inventory")

    observations: list[dict[str, Any]] = []
    bundle_members: list[tuple[str, int, str]] = []
    for item in members:
        path = Path(str(item["absolute_path"])).resolve()
        expected_hash = str(item.get("sha256", "")).lower()
        expected_size = int(item.get("size_bytes", -1))
        if not SHA256_RE.fullmatch(expected_hash) or expected_size < 0:
            raise ValueError(f"invalid locked inventory member: {path}")
        stat = path.stat()
        identity_fields = {
            "st_dev": stat.st_dev,
            "st_ino": stat.st_ino,
            "size_bytes": stat.st_size,
            "st_mtime_ns": stat.st_mtime_ns,
            "st_ctime_ns": stat.st_ctime_ns,
        }
        identity_changed = any(
            int(item.get(key, -1)) != value for key, value in identity_fields.items()
        )
        if identity_changed:
            observed_hash = sha256_file(path)
            if observed_hash != expected_hash:
                raise ValueError(f"runtime source content mutation detected: {path}")
        else:
            observed_hash = expected_hash
        relative_name = str(item.get("relative_path", ""))
        if not relative_name:
            raise ValueError(f"locked inventory member lacks relative_path: {path}")
        bundle_members.append((relative_name, expected_size, expected_hash))
        observations.append(
            {
                "path": str(path),
                "locked_sha256": expected_hash,
                "observed_sha256": observed_hash,
                "content_rehashed": identity_changed,
                "locked_identity": {
                    key: int(item[key]) for key in identity_fields
                },
                "observed_identity": identity_fields,
            }
        )
    observed_bundle = bundle_sha256(bundle_members)
    if observed_bundle != expected_bundle:
        raise ValueError("runtime source bundle digest differs from manifest")
    return {
        "status": "PASS",
        "unit": row["unit"],
        "modality": modality,
        "locked_inventory_record": {
            "path": str(record_path),
            "sha256": sha256_file(record_path),
            "size_bytes": record_path.stat().st_size,
        },
        "observed_bundle_sha256": observed_bundle,
        "member_count": len(observations),
        "members": observations,
        "content_rehash_count": sum(item["content_rehashed"] for item in observations),
    }


def build_runtime_inventory_slices(
    manifest_rows: Iterable[Mapping[str, str]],
    locked_inventory_path: str | Path,
    output_directory: str | Path,
) -> dict[str, Any]:
    """Stream the cohort inventory once into small execution-unit slices."""

    rows = [dict(row) for row in manifest_rows]
    destination = Path(output_directory).resolve()
    if destination.exists():
        raise FileExistsError(f"runtime inventory slice directory exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    handles: dict[str, Any] = {}
    selected: dict[tuple[str, str], tuple[str, str]] = {}
    for row in rows:
        selected[("DTI", str(row["dti_image_id"]))] = (row["unit"], "dwi")
        selected[("T1", str(row["t1_image_id"]))] = (row["unit"], "t1")
    inventory_path = Path(locked_inventory_path).resolve()
    total_inventory_rows = 0
    selected_inventory_rows = 0
    try:
        with inventory_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            required = {
                "modality",
                "image_id",
                "relative_path",
                "absolute_path",
                "size_bytes",
                "sha256",
                "st_dev",
                "st_ino",
                "st_mtime_ns",
                "st_ctime_ns",
            }
            if not required.issubset(reader.fieldnames or []):
                raise ValueError("locked source inventory schema differs")
            for inventory_row in reader:
                total_inventory_rows += 1
                target = selected.get(
                    (str(inventory_row["modality"]), str(inventory_row["image_id"]))
                )
                if target is None:
                    continue
                unit, modality = target
                stream = handles.get(unit)
                if stream is None:
                    stream = (temporary / f".{unit}.jsonl").open(
                        "x", encoding="utf-8"
                    )
                    handles[unit] = stream
                stream.write(
                    json.dumps(
                        {
                            "modality": modality,
                            **{
                                key: inventory_row[key]
                                for key in required - {"modality"}
                            },
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
                selected_inventory_rows += 1
        for stream in handles.values():
            stream.close()
        handles.clear()

        for row in rows:
            staging_path = temporary / f".{row['unit']}.jsonl"
            if not staging_path.is_file():
                raise ValueError(f"no locked inventory rows for {row['unit']}")
            members_by_modality: dict[str, list[dict[str, Any]]] = {
                "dwi": [],
                "t1": [],
            }
            for line in staging_path.read_text(encoding="utf-8").splitlines():
                item = json.loads(line)
                modality = item.pop("modality")
                for key in (
                    "size_bytes",
                    "st_dev",
                    "st_ino",
                    "st_mtime_ns",
                    "st_ctime_ns",
                ):
                    item[key] = int(item[key])
                members_by_modality[modality].append(item)
            staging_path.unlink()
            for modality, bundle_key, count_key, bytes_key in (
                (
                    "dwi",
                    "dti_raw_bundle_sha256",
                    "dti_raw_file_count",
                    "dti_raw_total_bytes",
                ),
                (
                    "t1",
                    "t1_raw_bundle_sha256",
                    "t1_raw_file_count",
                    "t1_raw_total_bytes",
                ),
            ):
                members = sorted(
                    members_by_modality[modality],
                    key=lambda item: item["relative_path"],
                )
                observed_bundle = bundle_sha256(
                    (
                        item["relative_path"],
                        item["size_bytes"],
                        item["sha256"],
                    )
                    for item in members
                )
                if (
                    len(members) != int(row[count_key])
                    or sum(item["size_bytes"] for item in members)
                    != int(row[bytes_key])
                    or observed_bundle != row[bundle_key]
                ):
                    raise ValueError(
                        f"locked inventory slice differs from manifest for "
                        f"{row['unit']} {modality}"
                    )
                payload = {
                    "schema_version": "2.0.0",
                    "unit": row["unit"],
                    "modality": modality,
                    "locked_bundle_sha256": observed_bundle,
                    "members": members,
                }
                (temporary / f"{row['unit']}.{modality}.json").write_text(
                    json.dumps(payload, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
        index = {
            "schema_version": "2.0.0",
            "status": "PASS",
            "locked_inventory": {
                "path": str(inventory_path),
                "sha256": sha256_file(inventory_path),
                "size_bytes": inventory_path.stat().st_size,
                "row_count": total_inventory_rows,
            },
            "execution_unit_count": len(rows),
            "selected_inventory_row_count": selected_inventory_rows,
            "units": sorted(row["unit"] for row in rows),
        }
        (temporary / "index.json").write_text(
            json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.replace(destination)
        return index
    finally:
        for stream in handles.values():
            stream.close()
        if temporary.exists():
            shutil.rmtree(temporary)


def write_normalization_failure_marker(
    path: str | Path,
    *,
    unit: str,
    normalizer: str,
    primary_failure_reason: str,
    source_identity_hashes: Mapping[str, str],
    generated_utc: str,
    run_id: str,
    attempt_id: str,
    attempt_context_path: str | Path,
) -> dict[str, Any]:
    """Atomically create one immutable, attempt-scoped failure marker."""

    if normalizer not in {"dwi", "t1"}:
        raise ValueError(f"Unsupported normalizer: {normalizer!r}")
    reason = " ".join(str(primary_failure_reason).split())[:2000]
    if not reason:
        raise ValueError("primary_failure_reason must be nonempty")
    if not run_id or not attempt_id or not SAFE_ID_RE.fullmatch(attempt_id):
        raise ValueError("run_id and a filesystem-safe attempt_id are required")
    attempt_path = Path(attempt_context_path).resolve()
    if not attempt_path.is_file():
        raise FileNotFoundError(f"attempt context missing: {attempt_path}")
    hashes = {str(key): str(value).lower() for key, value in source_identity_hashes.items()}
    if not hashes or any(not SHA256_RE.fullmatch(value) for value in hashes.values()):
        raise ValueError("source_identity_hashes must contain SHA-256 digests")
    record = {
        "schema_version": "2.0.0",
        "record_type": "source_normalization_failure",
        "status": "FAIL",
        "generated_utc": generated_utc,
        "unit": unit,
        "stage": "source_normalization",
        "normalizer": normalizer,
        "run_id": run_id,
        "attempt_id": attempt_id,
        "attempt_context": {
            "path": str(attempt_path),
            "sha256": sha256_file(attempt_path),
            "size_bytes": attempt_path.stat().st_size,
        },
        "primary_failure_reason": reason,
        "secondary_flags": [
            "no_staged_input_created",
            "biological_inference_invalid",
        ],
        "source_identity_hashes": hashes,
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(record, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if destination.exists():
        existing_payload = destination.read_bytes()
        if existing_payload != payload:
            raise FileExistsError(
                f"Conflicting immutable normalization marker: {destination}"
            )
        return json.loads(existing_payload)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".partial", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError:
            existing_payload = destination.read_bytes()
            if existing_payload != payload:
                raise FileExistsError(
                    f"Conflicting immutable normalization marker: {destination}"
                )
            return json.loads(existing_payload)
    finally:
        temporary.unlink(missing_ok=True)
    return record
