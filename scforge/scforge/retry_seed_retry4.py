"""Immutable-upstream seed validation for the H04A retry4 continuation."""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from .retry_seed import _valid_sha256, validate_file_record
from .retry_seed import validate_retry_seed_manifest as _validate_base_seed


RETRY4_ID = "SL-H04A-R1-RETRY4"
IMMUTABLE_SUBJECT_AREAS = ("00_inputs", "01_dwi")


def validate_retry4_seed_manifest(
    manifest_record: Mapping[str, Any],
    *,
    destination_root: Path,
    approved_units: Sequence[str],
    rehash_content: bool = True,
) -> dict[str, Any]:
    """Require a seed that cannot overlap any retry4 continuation output."""

    manifest_path = validate_file_record(
        manifest_record, label="retry4 seed manifest"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("allowed_subject_areas") != list(IMMUTABLE_SUBJECT_AREAS):
        raise ValueError("retry4 seed must declare only immutable upstream areas")
    if manifest.get("continuation_paths_excluded") is not True:
        raise ValueError("retry4 seed must explicitly exclude continuation paths")

    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("retry4 seed manifest has no files")
    for row in files:
        if not isinstance(row, Mapping):
            raise ValueError("retry4 seed row is not a mapping")
        relative = PurePosixPath(str(row.get("relative_path", "")))
        if len(relative.parts) < 4 or relative.parts[2] not in IMMUTABLE_SUBJECT_AREAS:
            raise ValueError(
                f"retry4 seed contains mutable continuation path: {relative}"
            )

    if rehash_content:
        result = _validate_base_seed(
            manifest_record,
            destination_root=destination_root,
            approved_units=approved_units,
            expected_retry_id=RETRY4_ID,
        )
    else:
        expected_scalars = {
            "schema_version": "1.0.0",
            "record_type": "h04a_r1_retry_seed_manifest",
            "status": "PASS",
            "retry_id": RETRY4_ID,
            "copy_mode": "reflink_copy_on_write",
            "selection_uses_diagnosis_labels": False,
        }
        for key, expected in expected_scalars.items():
            if manifest.get(key) != expected:
                raise ValueError(f"retry4 seed manifest differs at {key}")
        destination = destination_root.expanduser().resolve()
        if Path(str(manifest.get("destination_run_root", ""))).resolve() != destination:
            raise ValueError("retry4 seed destination root differs")
        validate_file_record(
            manifest.get("source_attempt_end", {}), label="retry4 source attempt end"
        )
        validate_file_record(
            manifest.get("source_completion", {}), label="retry4 source completion"
        )
        allowed_units = tuple(str(unit) for unit in approved_units)
        if manifest.get("approved_units") != list(allowed_units):
            raise ValueError("retry4 seed approved-unit list differs")
        allowed_unit_set = set(allowed_units)
        declared_count = manifest.get("file_count")
        declared_bytes = manifest.get("total_bytes")
        if declared_count != len(files):
            raise ValueError("retry4 seed file count differs")
        if isinstance(declared_bytes, bool) or not isinstance(declared_bytes, int):
            raise ValueError("retry4 seed total byte count is invalid")
        seen: set[str] = set()
        covered_units: set[str] = set()
        observed_bytes = 0
        for row in files:
            relative_value = str(row.get("relative_path", ""))
            relative = PurePosixPath(relative_value)
            parts = relative.parts
            expected_size = row.get("size_bytes")
            expected_hash = row.get("sha256")
            if (
                relative.is_absolute()
                or ".." in parts
                or len(parts) < 4
                or parts[0] != "subjects"
                or parts[1] not in allowed_unit_set
                or parts[2] not in IMMUTABLE_SUBJECT_AREAS
            ):
                raise ValueError(f"retry4 seed path is outside immutable areas: {relative}")
            if relative_value in seen:
                raise ValueError(f"retry4 seed path is duplicated: {relative}")
            if (
                isinstance(expected_size, bool)
                or not isinstance(expected_size, int)
                or expected_size < 0
                or not _valid_sha256(expected_hash)
            ):
                raise ValueError(f"retry4 seed row identity is invalid: {relative}")
            candidate = (destination / Path(*parts)).resolve()
            if (
                not candidate.is_relative_to(destination)
                or not candidate.is_file()
                or candidate.is_symlink()
                or candidate.stat().st_size != expected_size
            ):
                raise ValueError(f"retry4 seed worker-visible file differs: {relative}")
            seen.add(relative_value)
            covered_units.add(parts[1])
            observed_bytes += expected_size
        if observed_bytes != declared_bytes or covered_units != allowed_unit_set:
            raise ValueError("retry4 seed worker-visible coverage differs")
        result = {
            "schema_version": "1.0.0",
            "status": "PASS",
            "retry_id": RETRY4_ID,
            "manifest_path": str(manifest_path),
            "manifest_sha256": str(manifest_record["sha256"]),
            "destination_run_root": str(destination),
            "file_count": len(files),
            "total_bytes": observed_bytes,
            "covered_unit_count": len(covered_units),
        }
    result["immutable_subject_areas"] = list(IMMUTABLE_SUBJECT_AREAS)
    result["continuation_paths_excluded"] = True
    result["content_rehashed"] = rehash_content
    return result
