"""Validation for content-addressed H04A corrective-retry seed manifests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence


RETRY3_ID = "SL-H04A-R1-RETRY3"
ALLOWED_SUBJECT_AREAS = frozenset({"00_inputs", "01_dwi", "logs"})


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _valid_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return value == value.lower()


def validate_file_record(record: Mapping[str, Any], *, label: str) -> Path:
    if not isinstance(record, Mapping):
        raise ValueError(f"{label} must be a file-record mapping")
    path_value = record.get("path")
    expected_hash = record.get("sha256")
    expected_size = record.get("size_bytes")
    if not isinstance(path_value, str) or not path_value:
        raise ValueError(f"{label}.path is invalid")
    if not _valid_sha256(expected_hash):
        raise ValueError(f"{label}.sha256 is invalid")
    if isinstance(expected_size, bool) or not isinstance(expected_size, int) or expected_size < 0:
        raise ValueError(f"{label}.size_bytes is invalid")
    path = Path(path_value).expanduser().resolve()
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{label} is not a regular file: {path}")
    if path.stat().st_size != expected_size:
        raise ValueError(f"{label} size differs: {path}")
    if sha256_file(path) != expected_hash:
        raise ValueError(f"{label} SHA-256 differs: {path}")
    return path


def validate_retry_seed_manifest(
    manifest_record: Mapping[str, Any],
    *,
    destination_root: Path,
    approved_units: Sequence[str],
    expected_retry_id: str = RETRY3_ID,
) -> dict[str, Any]:
    """Rehash every seeded file and return a compact validated summary."""

    manifest_path = validate_file_record(manifest_record, label="retry seed manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_scalars = {
        "schema_version": "1.0.0",
        "record_type": "h04a_r1_retry_seed_manifest",
        "status": "PASS",
        "retry_id": expected_retry_id,
        "copy_mode": "reflink_copy_on_write",
        "selection_uses_diagnosis_labels": False,
    }
    for key, expected in expected_scalars.items():
        if manifest.get(key) != expected:
            raise ValueError(f"retry seed manifest differs at {key}")

    destination = destination_root.expanduser().resolve()
    if Path(str(manifest.get("destination_run_root", ""))).expanduser().resolve() != destination:
        raise ValueError("retry seed destination root differs")
    source = Path(str(manifest.get("source_run_root", ""))).expanduser().resolve()
    if source == destination:
        raise ValueError("retry seed source and destination roots must differ")

    validate_file_record(manifest.get("source_attempt_end", {}), label="retry2 attempt end")
    validate_file_record(manifest.get("source_completion", {}), label="retry2 completion")

    allowed_units = tuple(str(unit) for unit in approved_units)
    if not allowed_units or len(set(allowed_units)) != len(allowed_units):
        raise ValueError("approved retry unit list is empty or duplicated")
    if manifest.get("approved_units") != list(allowed_units):
        raise ValueError("retry seed approved-unit list differs")
    allowed_unit_set = set(allowed_units)

    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("retry seed manifest has no files")
    declared_count = manifest.get("file_count")
    declared_bytes = manifest.get("total_bytes")
    if declared_count != len(files):
        raise ValueError("retry seed file count differs")
    if isinstance(declared_bytes, bool) or not isinstance(declared_bytes, int) or declared_bytes < 1:
        raise ValueError("retry seed byte count is invalid")

    seen: set[str] = set()
    observed_bytes = 0
    unit_counts = {unit: 0 for unit in allowed_units}
    for index, row in enumerate(files):
        if not isinstance(row, Mapping):
            raise ValueError(f"retry seed row {index} is not a mapping")
        relative_value = row.get("relative_path")
        if not isinstance(relative_value, str) or not relative_value:
            raise ValueError(f"retry seed row {index} has invalid relative_path")
        relative = PurePosixPath(relative_value)
        parts = relative.parts
        if (
            relative.is_absolute()
            or ".." in parts
            or len(parts) < 4
            or parts[0] != "subjects"
            or parts[1] not in allowed_unit_set
            or parts[2] not in ALLOWED_SUBJECT_AREAS
        ):
            raise ValueError(f"retry seed path is outside the approved subject areas: {relative_value}")
        if relative_value in seen:
            raise ValueError(f"retry seed path is duplicated: {relative_value}")
        seen.add(relative_value)

        expected_size = row.get("size_bytes")
        expected_hash = row.get("sha256")
        if isinstance(expected_size, bool) or not isinstance(expected_size, int) or expected_size < 0:
            raise ValueError(f"retry seed size is invalid: {relative_value}")
        if not _valid_sha256(expected_hash):
            raise ValueError(f"retry seed SHA-256 is invalid: {relative_value}")
        candidate = (destination / Path(*parts)).resolve()
        if not candidate.is_relative_to(destination):
            raise ValueError(f"retry seed path escapes destination: {relative_value}")
        if not candidate.is_file() or candidate.is_symlink():
            raise ValueError(f"retry seed file is missing or not regular: {candidate}")
        if candidate.stat().st_size != expected_size:
            raise ValueError(f"retry seed file size differs: {relative_value}")
        if sha256_file(candidate) != expected_hash:
            raise ValueError(f"retry seed file SHA-256 differs: {relative_value}")
        observed_bytes += expected_size
        unit_counts[parts[1]] += 1

    if observed_bytes != declared_bytes:
        raise ValueError("retry seed total byte count differs")
    if any(count < 1 for count in unit_counts.values()):
        raise ValueError("retry seed does not cover every approved unit")
    return {
        "schema_version": "1.0.0",
        "status": "PASS",
        "retry_id": expected_retry_id,
        "manifest_path": str(manifest_path),
        "manifest_sha256": str(manifest_record["sha256"]),
        "destination_run_root": str(destination),
        "file_count": len(files),
        "total_bytes": observed_bytes,
        "covered_unit_count": len(unit_counts),
        "allowed_subject_areas": sorted(ALLOWED_SUBJECT_AREAS),
    }
