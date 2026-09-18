#!/home/ec2-user/fsl/bin/python
"""Independently verify the immutable balanced HCP379-v2 release.

This implementation deliberately does not import the release assembler.  It
re-hashes and re-opens all 4,770 matrix files, independently recomputes the
530 subject-level density/support/numerical checks, replays route and role
accounting, and writes a separate non-overwriting verification attempt.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


HCP_ROOT = Path("/data/derivatives/hcp379_v2")
RELEASE_ROOT = (
    HCP_ROOT / "balanced_release_v1/integration_release_v1"
)
CURRENT = RELEASE_ROOT / "current_release.json"
VERIFICATION_ROOT = RELEASE_ROOT / "independent_verification_v1"
VERIFICATION_ATTEMPTS = VERIFICATION_ROOT / "attempts"
VERIFICATION_POINTER = VERIFICATION_ROOT / "current_verification.json"
PREFLIGHT_OUTPUT = (
    Path("/home/ec2-user/exp/research_audit/outputs/")
    / "hcp379_balanced_integration_release_verifier_v1/preflight.json"
)
PRODUCTION_RUNNER = Path(
    "/home/ec2-user/exp/scripts/hcp/"
    "run_hcp379_balanced_production_v1.py"
)
ALL_NINE_BUILDER = Path(
    "/home/ec2-user/exp/scripts/hcp/"
    "build_hcp379_hroi_balanced_all_nine_canary_v1.py"
)
EXPECTED_SUBJECTS = 530
EXPECTED_CANARIES = 15
EXPECTED_PRODUCTION = 515
EXPECTED_NODES = 379
EXPECTED_CANARY_SEED_RECORDS = EXPECTED_CANARIES * 2
EXPECTED_SEED_STREAMLINES = 10_000_000
MATRIX_NAMES = (
    "count",
    "fd_sum",
    "count_invnodevol",
    "len_mean",
    "invlen_mean",
    "fa_mean",
    "md_mean",
    "rd_mean",
    "ad_mean",
)
EXPECTED_MATRIX_FILES = EXPECTED_SUBJECTS * len(MATRIX_NAMES)
ALLOWED_TOTALS = (6_000_000, 10_000_000, 15_000_000, 20_000_000)
DENSITY_TARGET = 0.60
MINIMUM_CONNECTED_NODES = 360
MINIMUM_ENDPOINT_ASSIGNMENT_FRACTION = 0.50
MINIMUM_WEIGHTED_SUPPORT_FRACTION = 0.99
HUB_IDS = (361, 365, 366, 370, 374, 375)
EXPECTED_ROUTE_COUNTS = {
    "corrected_core": 227,
    "corrected_legacy_tensor": 216,
    "archive_calibration": 6,
    "corrected_archive_low": 28,
    "corrected_archive_D0": 52,
    "corrected_freesurfer": 1,
}
FORBIDDEN_FIELD_FRAGMENTS = (
    "diagnosis",
    "dx_",
    "outcome",
    "cognitive_status",
    "research_group",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON object required: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise FileNotFoundError(resolved)
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256(resolved),
    }


def verify_record(record: Mapping[str, Any], *, label: str) -> Path:
    path = Path(str(record.get("path", ""))).resolve()
    if (
        not path.is_file()
        or path.is_symlink()
        or dict(record) != file_record(path)
    ):
        raise ValueError(f"{label}: file record differs")
    return path


def tck_header_count(path: Path) -> int:
    """Read the declared streamline count without loading tractogram payload."""
    payload = b""
    with path.open("rb") as handle:
        while b"END\n" not in payload and len(payload) <= 1024 * 1024:
            block = handle.read(4096)
            if not block:
                break
            payload += block
    header, marker, _ = payload.partition(b"END\n")
    if not marker:
        raise ValueError(f"{path}: TCK header terminator absent")
    for raw in header.decode("ascii", errors="strict").splitlines():
        name, separator, value = raw.partition(":")
        if separator and name.strip().lower() == "count":
            return int(value.strip())
    raise ValueError(f"{path}: TCK count field absent")


def verify_canary_production_equivalence(
    record: Mapping[str, Any],
) -> tuple[Path, dict[str, Any]]:
    """Independently replay the complete 15-pair execution-equivalence proof."""
    path = verify_record(
        record, label="canary-production equivalence validation"
    )
    value = load_json(path)
    source_records = value.get("records", {})
    checks = value.get("checks", {})
    metadata = value.get("metadata", [])
    if (
        value.get("record_type")
        != "hcp379_balanced_canary_production_equivalence_validation"
        or value.get("status") != "PASS"
        or value.get("complete_seed_metadata") is not True
        or value.get("seed_metadata_expected_n")
        != EXPECTED_CANARY_SEED_RECORDS
        or value.get("seed_metadata_pass_n")
        != EXPECTED_CANARY_SEED_RECORDS
        or value.get("seed_metadata_waiting_n") != 0
        or value.get("seed_metadata_failure_n") != 0
        or value.get("canary_pair_complete_n") != EXPECTED_CANARIES
        or value.get("production_execution_authorized") is not True
        or value.get("diagnosis_labels_used") is not False
        or value.get("imaging_executed") is not False
        or not isinstance(source_records, Mapping)
        or source_records.get("production_runner")
        != file_record(PRODUCTION_RUNNER)
        or source_records.get("all_nine_builder")
        != file_record(ALL_NINE_BUILDER)
        or not isinstance(checks, Mapping)
        or not checks
        or value.get("passed_checks") != len(checks)
        or value.get("total_checks") != len(checks)
        or any(
            not isinstance(check, Mapping)
            or check.get("status") != "PASS"
            for check in checks.values()
        )
        or not isinstance(metadata, list)
        or len(metadata) != EXPECTED_CANARY_SEED_RECORDS
    ):
        raise ValueError("complete canary-production equivalence differs")
    observed: set[tuple[str, str]] = set()
    canary_units: set[str] = set()
    for row in metadata:
        if not isinstance(row, Mapping):
            raise ValueError("canary seed metadata row is not an object")
        unit = str(row.get("unit", ""))
        seed_class = str(row.get("seed_class", ""))
        key = (unit, seed_class)
        if (
            row.get("status") != "PASS"
            or not unit
            or seed_class not in {"primary", "independent"}
            or key in observed
        ):
            raise ValueError(f"invalid canary seed row: {key}")
        observed.add(key)
        canary_units.add(unit)
        metadata_path = verify_record(
            row.get("metadata", {}),
            label=f"{unit}/{seed_class} tractography metadata",
        )
        parameters = load_json(metadata_path)
        tracks = row.get("tracks", {})
        if not isinstance(tracks, Mapping):
            raise ValueError(f"{unit}/{seed_class}: tracks record absent")
        tracks_path = Path(str(tracks.get("path", ""))).resolve()
        if (
            not tracks_path.is_file()
            or tracks_path.is_symlink()
            or tracks.get("path") != str(tracks_path)
            or tracks.get("size_bytes") != tracks_path.stat().st_size
            or tracks.get("header_streamline_count")
            != EXPECTED_SEED_STREAMLINES
            or tck_header_count(tracks_path) != EXPECTED_SEED_STREAMLINES
            or parameters.get("record_type")
            != "hcp379_stability_tractography_parameters"
            or parameters.get("diagnosis_labels_used") is not False
            or parameters.get("unit") != unit
            or parameters.get("run_id") != f"{seed_class}_10m"
            or parameters.get("seed_class") != seed_class
            or parameters.get("seed_id") != row.get("expected_seed_id")
            or parameters.get("rng_seed") != row.get("expected_rng_seed")
            or parameters.get("algorithm") != "iFOD2"
            or parameters.get("act") is not True
            or parameters.get("backtrack") is not True
            or parameters.get("crop_at_gmwmi") is not True
            or parameters.get("seeding") != "seed_dynamic"
            or parameters.get("requested_streamlines")
            != EXPECTED_SEED_STREAMLINES
            or parameters.get("actual_streamlines")
            != EXPECTED_SEED_STREAMLINES
        ):
            raise ValueError(
                f"{unit}/{seed_class}: tractography identity differs"
            )
    if (
        len(canary_units) != EXPECTED_CANARIES
        or any(
            (unit, seed_class) not in observed
            for unit in canary_units
            for seed_class in ("primary", "independent")
        )
    ):
        raise ValueError("15 complete primary-independent canary pairs absent")
    return path, value


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".partial",
        delete=False,
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def atomic_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    delimiter: str = "\t",
) -> None:
    if not rows:
        raise ValueError(f"empty table: {path}")
    fields = list(rows[0])
    if any(list(row) != fields for row in rows):
        raise ValueError(f"inconsistent columns: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        newline="",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".partial",
        delete=False,
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, delimiter=delimiter
        )
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def read_table(
    path: Path, *, delimiter: str
) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        fields = tuple(reader.fieldnames or ())
        rows = [
            {str(key): str(value or "").strip() for key, value in row.items()}
            for row in reader
        ]
    return fields, rows


def direct_record(path_value: str, digest: str, *, label: str) -> Path:
    path = Path(path_value).resolve()
    if (
        not path.is_file()
        or path.is_symlink()
        or str(path) != path_value
        or len(digest) != 64
        or sha256(path) != digest
    ):
        raise ValueError(f"{label}: path/hash differs")
    return path


def load_matrix(path: Path) -> np.ndarray:
    value = np.loadtxt(path, delimiter=",")
    if value.shape != (EXPECTED_NODES, EXPECTED_NODES):
        raise ValueError(f"{path}: shape is {value.shape}")
    return value


def load_node_volumes(path: Path) -> np.ndarray:
    _, rows = read_table(path, delimiter=",")
    if len(rows) != EXPECTED_NODES:
        raise ValueError("node-volume row count differs")
    rows.sort(key=lambda row: int(row["node_id"]))
    if [int(row["node_id"]) for row in rows] != list(
        range(1, EXPECTED_NODES + 1)
    ):
        raise ValueError("node-volume IDs differ")
    values = np.asarray(
        [float(row["volume_mm3"]) for row in rows], dtype=float
    )
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("node volumes are non-finite or non-positive")
    return values


def technical_qc(
    *,
    unit: str,
    matrices: Mapping[str, np.ndarray],
    node_volumes: np.ndarray,
    endpoint_assignment_fraction: float,
) -> dict[str, Any]:
    failures: list[str] = []
    for name, value in matrices.items():
        if value.shape != (EXPECTED_NODES, EXPECTED_NODES):
            failures.append(f"{name}:shape")
        if not np.isfinite(value).all():
            failures.append(f"{name}:non_finite")
        if np.any(value < -1.0e-12):
            failures.append(f"{name}:negative")
        if not np.allclose(value, value.T, atol=1.0e-8, rtol=1.0e-7):
            failures.append(f"{name}:asymmetric")
        if not np.allclose(np.diag(value), 0.0, atol=1.0e-10, rtol=0):
            failures.append(f"{name}:nonzero_diagonal")
    count = matrices["count"]
    if not np.allclose(count, np.rint(count), atol=1.0e-6, rtol=0):
        failures.append("count:not_integer")
    support = count > 0
    np.fill_diagonal(support, False)
    triangle = np.triu_indices(EXPECTED_NODES, 1)
    supported = int(np.count_nonzero(support[triangle]))
    possible = EXPECTED_NODES * (EXPECTED_NODES - 1) // 2
    density = supported / possible
    connected = int(np.count_nonzero(np.sum(count, axis=1) > 0))
    if density < DENSITY_TARGET:
        failures.append(f"density:{density:.12f}")
    if connected < MINIMUM_CONNECTED_NODES:
        failures.append(f"connected_nodes:{connected}")
    if any(float(np.sum(count[node - 1])) <= 0 for node in HUB_IDS):
        failures.append("locked_hub_disconnected")
    support_fractions = []
    outside_counts = []
    for name in MATRIX_NAMES:
        if name == "count":
            continue
        weighted = matrices[name] > 0
        np.fill_diagonal(weighted, False)
        outside = int(np.count_nonzero((weighted & ~support)[triangle]))
        covered = int(np.count_nonzero((weighted & support)[triangle]))
        fraction = covered / supported if supported else 0.0
        support_fractions.append(fraction)
        outside_counts.append(outside)
        if outside:
            failures.append(f"{name}:support_outside_count:{outside}")
        if fraction < MINIMUM_WEIGHTED_SUPPORT_FRACTION:
            failures.append(f"{name}:support_fraction:{fraction:.12f}")
    inverse = 2.0 * count / (
        node_volumes[:, None] + node_volumes[None, :]
    )
    np.fill_diagonal(inverse, 0.0)
    formula_error = float(
        np.max(np.abs(matrices["count_invnodevol"] - inverse))
    )
    if not np.allclose(
        matrices["count_invnodevol"],
        inverse,
        atol=1.0e-8,
        rtol=1.0e-7,
    ):
        failures.append("count_invnodevol:formula_differs")
    if endpoint_assignment_fraction < MINIMUM_ENDPOINT_ASSIGNMENT_FRACTION:
        failures.append(
            f"endpoint_assignment:{endpoint_assignment_fraction:.12f}"
        )
    return {
        "unit": unit,
        "status": "PASS" if not failures else "FAIL",
        "edge_density": density,
        "supported_edges": supported,
        "connected_nodes": connected,
        "endpoint_assignment_fraction": endpoint_assignment_fraction,
        "minimum_weighted_count_support_fraction": min(
            support_fractions, default=0.0
        ),
        "weighted_edges_outside_count_support": sum(outside_counts),
        "maximum_count_invnodevol_formula_error": formula_error,
        "failure_count": len(failures),
        "failures": "|".join(failures),
    }


def preflight() -> dict[str, Any]:
    if not CURRENT.is_file() or CURRENT.is_symlink():
        return {
            "schema_version": "1.0.0",
            "record_type": (
                "hcp379_balanced_integration_release_verifier_preflight"
            ),
            "generated_utc": utc_now(),
            "status": "AWAITING_ASSEMBLED_RELEASE",
            "release_detected": False,
            "independent_verification_started": False,
            "final_release_authorized": False,
        }
    pointer = load_json(CURRENT)
    manifest_path = verify_record(
        pointer["release_manifest"], label="release manifest"
    )
    manifest = load_json(manifest_path)
    equivalence_path: Path | None = None
    equivalence_error: str | None = None
    try:
        release_records = manifest.get("records", {})
        if not isinstance(release_records, Mapping):
            raise ValueError("release records absent")
        equivalence_path, _ = verify_canary_production_equivalence(
            release_records.get("canary_production_equivalence", {})
        )
    except Exception as exc:
        equivalence_error = f"{type(exc).__name__}:{exc}"
    ready = bool(
        pointer.get("record_type")
        == "diagnosis_blind_hcp379_balanced_integration_release_pointer"
        and pointer.get("status")
        == "PASS_ASSEMBLED_AWAITING_INDEPENDENT_VERIFICATION"
        and pointer.get("non_overwriting") is True
        and manifest.get("record_type")
        == "diagnosis_blind_hcp379_balanced_integration_release"
        and manifest.get("status")
        == "PASS_ASSEMBLED_AWAITING_INDEPENDENT_VERIFICATION"
        and manifest.get("subject_count") == EXPECTED_SUBJECTS
        and manifest.get("matrix_file_count") == EXPECTED_MATRIX_FILES
        and manifest.get("final_release_authorized") is False
        and equivalence_path is not None
    )
    return {
        "schema_version": "1.0.0",
        "record_type": (
            "hcp379_balanced_integration_release_verifier_preflight"
        ),
        "generated_utc": utc_now(),
        "status": "READY_TO_VERIFY" if ready else "INVALID_RELEASE_POINTER",
        "release_detected": True,
        "release_manifest": file_record(manifest_path),
        "canary_production_equivalence": (
            file_record(equivalence_path)
            if equivalence_path is not None
            else None
        ),
        "canary_production_equivalence_verified": (
            equivalence_path is not None
        ),
        "canary_production_equivalence_error": equivalence_error,
        "independent_verification_started": False,
        "final_release_authorized": False,
    }


def validate_source_summary(
    row: Mapping[str, str],
    matrix_records: Mapping[str, Mapping[str, Any]],
    selected_total: int,
    per_seed: int,
) -> None:
    unit = row["unit"]
    summary_path = direct_record(
        row["source_summary_path"],
        row["source_summary_sha256"],
        label=f"{unit}/source summary",
    )
    summary = load_json(summary_path)
    role = row["cohort_role"]
    if role == "CALIBRATION_CANARY_REUSE":
        view = summary.get("views", {}).get("balanced", {})
        valid = bool(
            summary.get("record_type")
            == "diagnosis_blind_hcp379_hroi_balanced_all_nine_canary_summary"
            and summary.get("status")
            == "PASS_HROI_BALANCED_ALL_NINE_CANARY"
            and summary.get("selected_balanced_total_streamlines")
            == selected_total
            and summary.get("selected_per_seed_prefix_streamlines")
            == per_seed
            and summary.get("seed_reliability", {}).get("status") == "PASS"
            and summary.get("combined_density_target_met") is True
            and summary.get("all_views_technical_qc_pass") is True
            and not row["compaction_path"]
            and not row["compaction_sha256"]
        )
    elif role == "BALANCED_PRODUCTION":
        view = summary.get("balanced_view", {})
        compaction_path = direct_record(
            row["compaction_path"],
            row["compaction_sha256"],
            label=f"{unit}/compaction",
        )
        compaction = load_json(compaction_path)
        valid = bool(
            summary.get("record_type")
            == "diagnosis_blind_hcp379_balanced_production_summary"
            and summary.get("status") == "PASS_BALANCED_PRODUCTION"
            and summary.get("selected_balanced_total_streamlines")
            == selected_total
            and summary.get("selected_per_seed_streamlines") == per_seed
            and summary.get("combined_density_target_met") is True
            and summary.get("technical_qc_pass") is True
            and compaction.get("status")
            == "PASS_COMPACTED_GENERATED_TRACKS"
            and compaction.get("retained_scientific_summary")
            == file_record(summary_path)
            and all(
                not Path(str(record.get("path", ""))).exists()
                for record in compaction.get(
                    "generated_tractograms_deleted", {}
                ).values()
            )
        )
    else:
        raise ValueError(f"{unit}: unknown cohort role {role}")
    if (
        not valid
        or summary.get("unit") != unit
        or view.get("technical_status") != "PASS"
        or view.get("sift2", {}).get("weight_count") != selected_total
        or set(view.get("matrices", {})) != set(MATRIX_NAMES)
        or any(
            view["matrices"][name] != matrix_records[name]
            for name in MATRIX_NAMES
        )
        or row["assignments_path"]
        != view.get("assignments", {}).get("path")
        or row["assignments_sha256"]
        != view.get("assignments", {}).get("sha256")
        or row["sift2_weights_path"]
        != view.get("sift2", {}).get("weights", {}).get("path")
        or row["sift2_weights_sha256"]
        != view.get("sift2", {}).get("weights", {}).get("sha256")
    ):
        raise ValueError(f"{unit}: source summary replay differs")
    for label, record in (
        ("assignments", view["assignments"]),
        ("SIFT2 weights", view["sift2"]["weights"]),
    ):
        artifact = Path(str(record.get("path", ""))).resolve()
        if (
            not artifact.is_file()
            or artifact.is_symlink()
            or record.get("path") != str(artifact)
            or record.get("size_bytes") != artifact.stat().st_size
            or len(str(record.get("sha256", ""))) != 64
        ):
            raise ValueError(f"{unit}/{label}: retained artifact differs")


def verify() -> dict[str, Any]:
    ready = preflight()
    if ready["status"] != "READY_TO_VERIFY":
        raise RuntimeError(ready["status"])
    if VERIFICATION_POINTER.exists():
        raise FileExistsError(
            f"verification pointer already exists: {VERIFICATION_POINTER}"
        )
    pointer = load_json(CURRENT)
    manifest_path = verify_record(
        pointer["release_manifest"], label="release manifest"
    )
    manifest = load_json(manifest_path)
    records = manifest.get("records", {})
    equivalence_path, equivalence = verify_canary_production_equivalence(
        records["canary_production_equivalence"]
    )
    analysis_path = verify_record(
        records["analysis_ready_manifest"], label="analysis manifest"
    )
    source_qc_path = verify_record(
        records["subject_qc"], label="source subject QC"
    )
    matrix_index_path = verify_record(
        records["matrix_files"], label="matrix index"
    )
    bound_inputs_path = verify_record(
        records["bound_inputs"], label="bound inputs"
    )
    bound_inputs = load_json(bound_inputs_path)
    if (
        bound_inputs.get("canary_production_equivalence")
        != file_record(equivalence_path)
    ):
        raise ValueError(
            "bound-input canary-production equivalence differs"
        )
    verify_record(records["dataset_description"], label="dataset description")
    verify_record(records["readme"], label="README")
    analysis_fields, subjects = read_table(
        analysis_path, delimiter=","
    )
    qc_fields, source_qc = read_table(source_qc_path, delimiter="\t")
    matrix_fields, matrix_rows = read_table(
        matrix_index_path, delimiter="\t"
    )
    all_fields = {
        field.lower()
        for field in analysis_fields + qc_fields + matrix_fields
    }
    forbidden = sorted(
        field
        for field in all_fields
        if any(fragment in field for fragment in FORBIDDEN_FIELD_FRAGMENTS)
    )
    subject_units = [row["unit"] for row in subjects]
    matrix_by_unit: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in matrix_rows:
        unit = row["unit"]
        name = row["matrix"]
        if name in matrix_by_unit[unit]:
            raise ValueError(f"{unit}/{name}: duplicate matrix index row")
        path = direct_record(
            row["path"], row["sha256"], label=f"{unit}/{name}"
        )
        if int(row["size_bytes"]) != path.stat().st_size:
            raise ValueError(f"{unit}/{name}: indexed size differs")
        matrix_by_unit[unit][name] = {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": row["sha256"],
        }
    selected_values = {
        int(row["selected_balanced_total_streamlines"])
        for row in subjects
    }
    per_seed_values = {
        int(row["selected_per_seed_streamlines"]) for row in subjects
    }
    role_counts = Counter(row["cohort_role"] for row in subjects)
    route_counts = Counter(row["source_recovery_lane"] for row in subjects)
    if (
        forbidden
        or len(subjects) != EXPECTED_SUBJECTS
        or len(set(subject_units)) != EXPECTED_SUBJECTS
        or len(source_qc) != EXPECTED_SUBJECTS
        or {row["unit"] for row in source_qc} != set(subject_units)
        or len(matrix_rows) != EXPECTED_MATRIX_FILES
        or set(matrix_by_unit) != set(subject_units)
        or any(set(rows) != set(MATRIX_NAMES) for rows in matrix_by_unit.values())
        or selected_values != {manifest["selected_balanced_total_streamlines"]}
        or selected_values.pop() not in ALLOWED_TOTALS
        or per_seed_values
        != {manifest["selected_per_seed_streamlines"]}
        or manifest["selected_balanced_total_streamlines"]
        != 2 * manifest["selected_per_seed_streamlines"]
        or role_counts
        != Counter(
            {
                "CALIBRATION_CANARY_REUSE": EXPECTED_CANARIES,
                "BALANCED_PRODUCTION": EXPECTED_PRODUCTION,
            }
        )
        or dict(route_counts) != EXPECTED_ROUTE_COUNTS
        or manifest.get("source_recovery_route_counts")
        != EXPECTED_ROUTE_COUNTS
        or manifest.get("cohort_uniform") is not True
        or manifest.get("per_subject_streamline_tuning_allowed") is not False
        or manifest.get("subject_exclusion_for_low_density_allowed") is not False
        or manifest.get("diagnosis_or_outcome_fields_present") is not False
    ):
        raise ValueError(
            "release identity, blinding, route or uniform-recipe contract differs"
        )
    selected_total = int(manifest["selected_balanced_total_streamlines"])
    per_seed = int(manifest["selected_per_seed_streamlines"])
    source_qc_by_unit = {row["unit"]: row for row in source_qc}
    attempt = (
        VERIFICATION_ATTEMPTS
        / f"{stamp()}-independent-balanced-release-verification"
    )
    attempt.mkdir(parents=True, exist_ok=False)
    verified_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for index, row in enumerate(sorted(subjects, key=lambda item: item["unit"]), start=1):
        unit = row["unit"]
        matrix_records = matrix_by_unit[unit]
        for name in MATRIX_NAMES:
            if (
                row[f"matrix_{name}_path"]
                != matrix_records[name]["path"]
                or row[f"matrix_{name}_sha256"]
                != matrix_records[name]["sha256"]
            ):
                raise ValueError(f"{unit}/{name}: subject index differs")
        validate_source_summary(
            row, matrix_records, selected_total, per_seed
        )
        node_volumes_path = direct_record(
            row["node_volumes_path"],
            row["node_volumes_sha256"],
            label=f"{unit}/node volumes",
        )
        matrices = {
            name: load_matrix(Path(matrix_records[name]["path"]))
            for name in MATRIX_NAMES
        }
        qc = technical_qc(
            unit=unit,
            matrices=matrices,
            node_volumes=load_node_volumes(node_volumes_path),
            endpoint_assignment_fraction=float(
                row["endpoint_assignment_fraction"]
            ),
        )
        source = source_qc_by_unit[unit]
        if (
            source["status"] != "PASS"
            or source["failure_count"] != "0"
            or abs(float(source["edge_density"]) - qc["edge_density"])
            > 5.0e-10
            or int(source["connected_nodes"]) != qc["connected_nodes"]
            or abs(
                float(source["endpoint_assignment_fraction"])
                - qc["endpoint_assignment_fraction"]
            )
            > 5.0e-10
            or abs(float(row["edge_density"]) - qc["edge_density"])
            > 5.0e-10
            or int(row["connected_nodes"]) != qc["connected_nodes"]
        ):
            qc["status"] = "FAIL"
            qc["failure_count"] = int(qc["failure_count"]) + 1
            qc["failures"] = (
                qc["failures"] + "|" if qc["failures"] else ""
            ) + "assembler_qc_replay_differs"
        verified_rows.append(
            {
                "unit": unit,
                "status": qc["status"],
                "cohort_role": row["cohort_role"],
                "source_recovery_lane": row["source_recovery_lane"],
                "edge_density": f"{qc['edge_density']:.12g}",
                "supported_edges": qc["supported_edges"],
                "connected_nodes": qc["connected_nodes"],
                "endpoint_assignment_fraction": (
                    f"{qc['endpoint_assignment_fraction']:.12g}"
                ),
                "minimum_weighted_count_support_fraction": (
                    f"{qc['minimum_weighted_count_support_fraction']:.12g}"
                ),
                "weighted_edges_outside_count_support": qc[
                    "weighted_edges_outside_count_support"
                ],
                "maximum_count_invnodevol_formula_error": (
                    f"{qc['maximum_count_invnodevol_formula_error']:.12g}"
                ),
                "failure_count": qc["failure_count"],
                "failures": qc["failures"],
            }
        )
        if qc["status"] != "PASS":
            failures.append(
                {"unit": unit, "failures": qc["failures"]}
            )
        print(
            f"[{utc_now()}] independent-release-verification "
            f"{index}/{EXPECTED_SUBJECTS} {unit} {qc['status']}",
            flush=True,
        )
    atomic_csv(attempt / "verified_subject_qc.tsv", verified_rows)
    densities = [float(row["edge_density"]) for row in verified_rows]
    result = {
        "schema_version": "1.0.0",
        "record_type": (
            "hcp379_balanced_integration_release_independent_verification"
        ),
        "status": (
            "PASS_VERIFIED_INTEGRATION_RELEASE"
            if not failures
            else "FAIL_INTEGRATION_RELEASE_VERIFICATION"
        ),
        "generated_utc": utc_now(),
        "diagnosis_or_outcome_fields_present": False,
        "independent_of_assembler_implementation": True,
        "release_manifest": file_record(manifest_path),
        "subject_count": len(verified_rows),
        "matrix_file_count": len(matrix_rows),
        "selected_balanced_total_streamlines": selected_total,
        "selected_per_seed_streamlines": per_seed,
        "minimum_observed_edge_density": min(densities),
        "maximum_observed_edge_density": max(densities),
        "all_subjects_density_target_met": all(
            density >= DENSITY_TARGET for density in densities
        ),
        "source_recovery_route_counts": dict(sorted(route_counts.items())),
        "source_recovery_route_sensitivity_required": True,
        "calibration_production_equivalence_verified": True,
        "canary_seed_metadata_replayed_n": (
            equivalence["seed_metadata_pass_n"]
        ),
        "canary_pair_replayed_n": equivalence[
            "canary_pair_complete_n"
        ],
        "canary_production_equivalence": file_record(equivalence_path),
        "failure_subject_n": len(failures),
        "failures": failures,
        "verified_subject_qc": file_record(
            attempt / "verified_subject_qc.tsv"
        ),
        "final_release_authorized": not failures,
    }
    atomic_json(attempt / "verification.json", result)
    if failures:
        return result
    pointer_value = {
        "schema_version": "1.0.0",
        "record_type": (
            "hcp379_balanced_integration_release_verification_pointer"
        ),
        "status": "PASS_VERIFIED_INTEGRATION_RELEASE",
        "created_utc": utc_now(),
        "release_manifest": file_record(manifest_path),
        "verification": file_record(attempt / "verification.json"),
    }
    if VERIFICATION_POINTER.exists():
        raise FileExistsError(VERIFICATION_POINTER)
    atomic_json(VERIFICATION_POINTER, pointer_value)
    return result


def self_test() -> dict[str, Any]:
    with tempfile.TemporaryDirectory(
        prefix="hcp379-balanced-verifier-test."
    ) as temporary:
        root = Path(temporary)
        volume_path = root / "volumes.csv"
        volume_rows = [
            {"node_id": index, "volume_mm3": float(index + 20)}
            for index in range(1, EXPECTED_NODES + 1)
        ]
        atomic_csv(volume_path, volume_rows, delimiter=",")
        volumes = load_node_volumes(volume_path)
        count = np.zeros((EXPECTED_NODES, EXPECTED_NODES), dtype=float)
        target = math.ceil(
            DENSITY_TARGET
            * EXPECTED_NODES
            * (EXPECTED_NODES - 1)
            / 2
        )
        upper = np.triu_indices(EXPECTED_NODES, 1)
        count[upper[0][:target], upper[1][:target]] = 1
        count = count + count.T
        inverse = 2.0 * count / (
            volumes[:, None] + volumes[None, :]
        )
        np.fill_diagonal(inverse, 0.0)
        matrices = {
            name: inverse if name == "count_invnodevol" else count
            for name in MATRIX_NAMES
        }
        passed = technical_qc(
            unit="synthetic",
            matrices=matrices,
            node_volumes=volumes,
            endpoint_assignment_fraction=0.9,
        )
        sparse = dict(matrices)
        sparse_count = np.zeros_like(count)
        for index in range(EXPECTED_NODES - 1):
            sparse_count[index, index + 1] = 1
            sparse_count[index + 1, index] = 1
        sparse["count"] = sparse_count
        failed = technical_qc(
            unit="synthetic_sparse",
            matrices=sparse,
            node_volumes=volumes,
            endpoint_assignment_fraction=0.9,
        )
    checks = {
        "exact_matrix_file_count": EXPECTED_MATRIX_FILES == 4770,
        "route_counts_sum_to_530": (
            sum(EXPECTED_ROUTE_COUNTS.values()) == EXPECTED_SUBJECTS
        ),
        "valid_synthetic_family_passes": passed["status"] == "PASS",
        "sparse_or_inconsistent_family_fails": failed["status"] == "FAIL",
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        value = self_test()
        print(json.dumps(value, indent=2, sort_keys=True))
        return 0 if value["status"] == "PASS" else 1
    if args.verify:
        value = verify()
        print(json.dumps(value, indent=2, sort_keys=True))
        return 0 if value["status"].startswith("PASS_") else 1
    value = preflight()
    atomic_json(PREFLIGHT_OUTPUT, value)
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
