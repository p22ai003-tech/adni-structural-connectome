#!/usr/bin/env python3
"""Compile the 530-row SC-Forge v2.1 manifest from the locked source projection.

This compiler is metadata-only.  The upstream all-file preflight is the sole
source of DICOM/NIfTI identity metadata.  This script never opens an image or
DICOM header, never guesses conversion metadata, and writes each release file
only when absent or byte-identical.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Mapping


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scforge"))

from scforge.input_contract import load_acquisition_manifest, sha256_file  # noqa: E402


DEFAULT_PROJECTION = (
    PROJECT
    / "research_audit"
    / "outputs"
    / "source_metadata_workflow_preflight_v1"
    / "source_metadata_workflow_projection_v1.csv"
)
DEFAULT_PROJECTION_VALIDATION = DEFAULT_PROJECTION.with_name(
    "source_metadata_workflow_projection_validation_v1.json"
)
DEFAULT_SCHEMA = (
    PROJECT
    / "scforge"
    / "workflow"
    / "schemas"
    / "acquisition_manifest_v2.schema.json"
)
DEFAULT_OUTPUT = (
    PROJECT
    / "research_audit"
    / "outputs"
    / "connectome_v2_input_manifest_v2.csv"
)
DEFAULT_VALIDATION = (
    PROJECT
    / "research_audit"
    / "outputs"
    / "connectome_v2_input_manifest_validation_v2.json"
)
EXPECTED_PROJECTION_SHA256 = (
    "1546c065cb60e3c16ba47dc4e586eaf068149c4d50e55ce5935fb3695b1d7757"
)
EXPECTED_PROJECTION_VALIDATION_SHA256 = (
    "060c9cf74b66dd67af60267733217f28fdc85a3be21f385067a9e9c826fe252f"
)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path}")
        return [
            {str(key): str(value or "").strip() for key, value in row.items()}
            for row in reader
        ]


def _truth(value: str, *, label: str) -> bool:
    normalized = value.strip().lower()
    if normalized not in {"true", "false"}:
        raise ValueError(f"{label} must be true or false, found {value!r}")
    return normalized == "true"


def _timing_stratum(abs_gap_days: str) -> str:
    gap = int(abs_gap_days)
    if gap <= 90:
        return "le_90_days"
    if gap <= 180:
        return "days_91_180"
    return "gt_180_days"


def _normalization_readiness(direction: str, readout: str) -> str:
    if direction and readout:
        return "READY"
    if not direction and not readout:
        return "FAIL_MISSING_PHASE_ENCODING_AND_TOTAL_READOUT_TIME"
    if not direction:
        return "FAIL_MISSING_PHASE_ENCODING"
    return "FAIL_MISSING_TOTAL_READOUT_TIME"


def _csv_payload(fieldnames: list[str], rows: list[dict[str, str]]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def _write_new_or_identical(path: Path, payload: bytes) -> None:
    """Atomically create ``path`` without replacing non-identical evidence."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError(f"Refusing to replace non-identical release: {path}")
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
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != payload:
                raise FileExistsError(
                    f"Conflicting release appeared during write: {path}"
                )
    finally:
        temporary.unlink(missing_ok=True)


def _verify_projection_release(
    projection: Path,
    projection_validation: Path,
    *,
    expected_projection_sha256: str,
    expected_validation_sha256: str,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    if sha256_file(projection) != expected_projection_sha256:
        raise ValueError("source-metadata projection SHA-256 differs from H03-C1 release")
    if sha256_file(projection_validation) != expected_validation_sha256:
        raise ValueError("source-metadata validation SHA-256 differs from H03-C1 release")
    validation = json.loads(projection_validation.read_text(encoding="utf-8"))
    projection_record = validation.get("outputs", {}).get("workflow_projection", {})
    locked_inventory = validation.get("inputs", {}).get("locked_file_inventory", {})
    locked_pairs = validation.get("inputs", {}).get("locked_pair_manifest", {})
    if (
        validation.get("status") != "PASS"
        or validation.get("release_status")
        != "SOURCE_METADATA_PREFLIGHT_COMPLETE_CONVERSION_PENDING"
        or validation.get("canary_authorized") is not False
        or projection_record.get("sha256") != expected_projection_sha256
        or projection_record.get("row_count") != 530
        or locked_inventory.get("sha256")
        != "c0f221708a62ccb9acebc9732ac6e86f34c3e6629b63b6d4a3cc0d280e2672b2"
        or locked_inventory.get("rows") != 451115
        or locked_pairs.get("sha256")
        != "1e47d263c70a8230ab0b39651b652333d03e31c63308c9f18dfa008f804689f5"
        or locked_pairs.get("rows") != 530
        or validation.get("counts", {}).get("source_identity_blocked") != 0
        or validation.get("counts", {}).get("conversion_pending") != 530
        or validation.get("counts", {}).get("canary_ready") != 0
    ):
        raise ValueError("source-metadata projection validation semantics differ")
    rows = _read_rows(projection)
    if len(rows) != 530:
        raise ValueError(f"Expected 530 projected source pairs, found {len(rows)}")
    identities = [(row["subject_id"], row["dti_image_id"]) for row in rows]
    if len(identities) != len(set(identities)):
        raise ValueError("source-metadata projection has duplicate DTI identities")
    return rows, validation


def _source_row_to_manifest(
    source: Mapping[str, str], fieldnames: list[str]
) -> dict[str, str]:
    identity = f"{source['subject_id']}_I{source['dti_image_id']}"
    diagnosis = source["diagnosis_at_dti"]
    processing_approved = _truth(
        source["pair_processing_approved"], label=f"{identity} pair_processing_approved"
    )
    primary = _truth(
        source["primary_analysis_eligible"], label=f"{identity} primary eligibility"
    )
    smc = _truth(
        source["smc_retained_separately"], label=f"{identity} SMC retention"
    )
    if not processing_approved:
        raise ValueError(f"Projection unexpectedly rejects processing for {identity}")
    if (diagnosis == "SMC") != smc or primary == smc:
        raise ValueError(f"Projection analysis-role flags differ for {identity}")
    if source["workflow_projection_status"] != "SOURCE_IDENTITY_READY_CONVERSION_PENDING":
        raise ValueError(f"Projection status differs for {identity}")
    if _truth(source["canary_ready"], label=f"{identity} canary_ready"):
        raise ValueError(f"Projection cannot be canary-ready before conversion: {identity}")
    if source["dti_source_class"] != "ORIGINAL_DICOM":
        raise ValueError(f"Unsupported DTI source class for {identity}")
    if source["t1_source_class"] == "ORIGINAL_DICOM":
        t1_kind = "dicom_series"
        t1_path = source["t1_series_dir"]
        t1_uid = source["t1_dicom_series_uid"]
    elif source["t1_source_class"] == "PROCESSED_NIFTI_DERIVED":
        t1_kind = "nifti_single"
        t1_path = source["t1_nifti_source_path"]
        t1_uid = ""
    else:
        raise ValueError(f"Unsupported T1 source class for {identity}")
    direction = source["phase_encoding_direction"]
    readout = source["total_readout_time"]
    if bool(direction) != (source["phase_encoding_direction_status"] == "PASS"):
        if source["phase_encoding_direction_status"] != "PENDING_CONVERSION":
            raise ValueError(f"Phase-encoding projection status differs for {identity}")
    if bool(readout) != (source["total_readout_time_status"] == "PASS"):
        if source["total_readout_time_status"] != "PENDING_CONVERSION":
            raise ValueError(f"Readout-time projection status differs for {identity}")

    output_row = {
        "manifest_schema_version": "2.0.0",
        "data_scope_decision": "SL-D01_local_only",
        "processing_authorized": "true",
        "analysis_role": "smc_retained_separately" if smc else "primary_cn_mci_ad",
        "subject_id": source["subject_id"],
        "diagnosis_at_dti": diagnosis,
        "dti_image_id": source["dti_image_id"],
        "dti_study_date": source["dti_study_date"],
        "dti_source_kind": "dicom_series",
        "dti_source_id": source["dti_source_id"],
        "dti_source_path": source["dti_series_dir"],
        "dti_dicom_series_uid": source["dti_series_instance_uid"],
        "dti_raw_bundle_sha256": source["dti_source_bundle_sha256"],
        "dti_raw_file_count": source["dti_inventory_file_count"],
        "dti_raw_total_bytes": source["dti_inventory_total_bytes"],
        "dwi_nifti_path": "",
        "dwi_bvec_path": "",
        "dwi_bval_path": "",
        "dwi_json_path": "",
        "dwi_nifti_sha256": "",
        "dwi_bvec_sha256": "",
        "dwi_bval_sha256": "",
        "dwi_json_sha256": "",
        "t1_image_id": source["t1_image_id"],
        "t1_study_date": source["t1_study_date"],
        "t1_source_kind": t1_kind,
        "t1_source_id": source["t1_source_id"],
        "t1_source_path": t1_path,
        "t1_dicom_series_uid": t1_uid,
        "t1_raw_bundle_sha256": source["t1_source_bundle_sha256"],
        "t1_raw_file_count": source["t1_inventory_file_count"],
        "t1_raw_total_bytes": source["t1_inventory_total_bytes"],
        "abs_pair_gap_days": source["abs_pair_gap_days"],
        "timing_stratum": _timing_stratum(source["abs_pair_gap_days"]),
        "phase": source["phase"],
        "site": source["site"],
        "manufacturer": source["dti_manufacturer_header"]
        or source["dti_manufacturer_manifest"],
        "scanner_model": source["scanner_model"],
        "field_strength_t": source["dti_field_strength_header_t"]
        or source["dti_field_strength_manifest_t"],
        "protocol": source["protocol"],
        "phase_encoding_direction": direction,
        "phase_encoding_source": "source_metadata_workflow_projection_v1"
        if direction
        else "",
        "total_readout_time": readout,
        "total_readout_time_source": "source_metadata_workflow_projection_v1"
        if readout
        else "",
        "source_lock_status": "COMPLETE_SHA256",
        "pair_content_bundle_sha256": source["pair_source_bundle_sha256"],
        "normalization_readiness": _normalization_readiness(direction, readout),
    }
    missing = set(fieldnames).difference(output_row)
    if missing:
        raise ValueError(f"Builder lacks schema fields: {sorted(missing)}")
    return {key: str(output_row[key]) for key in fieldnames}


def build_manifest(
    projection: Path,
    projection_validation: Path,
    schema: Path,
    output: Path,
    validation: Path,
    *,
    expected_projection_sha256: str = EXPECTED_PROJECTION_SHA256,
    expected_projection_validation_sha256: str = EXPECTED_PROJECTION_VALIDATION_SHA256,
) -> dict[str, Any]:
    source_rows, projection_release = _verify_projection_release(
        projection,
        projection_validation,
        expected_projection_sha256=expected_projection_sha256,
        expected_validation_sha256=expected_projection_validation_sha256,
    )
    schema_record = json.loads(schema.read_text(encoding="utf-8"))
    fieldnames = list(schema_record["required"])
    output_rows = [
        _source_row_to_manifest(source, fieldnames) for source in source_rows
    ]
    payload = _csv_payload(fieldnames, output_rows)

    # Validate the exact bytes before making the release path visible.
    with tempfile.TemporaryDirectory(prefix="connectome_v2_manifest_validation_") as directory:
        candidate = Path(directory) / output.name
        candidate.write_bytes(payload)
        load_acquisition_manifest(
            candidate,
            schema,
            run_root=PROJECT / "scforge_v2_schema_validation",
            expected_rows=530,
        )
    _write_new_or_identical(output, payload)
    validated_rows = load_acquisition_manifest(
        output,
        schema,
        run_root=PROJECT / "scforge_v2_schema_validation",
        expected_rows=530,
    )
    groups = Counter(row["diagnosis_at_dti"] for row in validated_rows)
    t1_kinds = Counter(row["t1_source_kind"] for row in validated_rows)
    readiness = Counter(row["normalization_readiness"] for row in validated_rows)
    result = {
        "schema_version": "2.0.0",
        "status": "PASS",
        "generated_utc": projection_release["generated_utc"],
        "data_scope_decision": "SL-D01_local_only",
        "human_gate": "SL-H03-C1",
        "image_processing_run": False,
        "source_headers_or_voxels_read_by_compiler": False,
        "source_metadata_projection": {
            "path": str(projection.resolve()),
            "sha256": sha256_file(projection),
            "size_bytes": projection.stat().st_size,
            "row_count": len(source_rows),
        },
        "source_metadata_projection_validation": {
            "path": str(projection_validation.resolve()),
            "sha256": sha256_file(projection_validation),
            "size_bytes": projection_validation.stat().st_size,
        },
        "locked_pair_manifest": projection_release["inputs"]["locked_pair_manifest"],
        "locked_file_inventory": projection_release["inputs"]["locked_file_inventory"],
        "content_lock_validation": projection_release["inputs"]["content_lock_validation"],
        "acquisition_schema": str(schema.resolve()),
        "acquisition_schema_sha256": sha256_file(schema),
        "output_manifest": str(output.resolve()),
        "output_manifest_sha256": sha256_file(output),
        "row_count": len(validated_rows),
        "processing_authorized_count": sum(
            row["processing_authorized"].lower() == "true" for row in validated_rows
        ),
        "diagnosis_counts": dict(sorted(groups.items())),
        "t1_source_kind_counts": dict(sorted(t1_kinds.items())),
        "normalization_readiness_counts": dict(sorted(readiness.items())),
        "conversion_pending_count": 530,
        "canary_ready_count": 0,
        "processing_rejection_by_gap": False,
        "processing_rejection_by_diagnosis": False,
        "processing_rejection_by_t1_source_kind": False,
        "phase_encoding_direction_guessed": False,
        "total_readout_time_defaulted": False,
        "release_state": "SOURCE_IDENTITY_READY_CONVERSION_PENDING",
    }
    validation_payload = (
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    _write_new_or_identical(validation, validation_payload)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--projection", type=Path, default=DEFAULT_PROJECTION)
    parser.add_argument(
        "--projection-validation", type=Path, default=DEFAULT_PROJECTION_VALIDATION
    )
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--validation", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument(
        "--expected-projection-sha256", default=EXPECTED_PROJECTION_SHA256
    )
    parser.add_argument(
        "--expected-projection-validation-sha256",
        default=EXPECTED_PROJECTION_VALIDATION_SHA256,
    )
    args = parser.parse_args()
    result = build_manifest(
        args.projection.resolve(),
        args.projection_validation.resolve(),
        args.schema.resolve(),
        args.output.resolve(),
        args.validation.resolve(),
        expected_projection_sha256=args.expected_projection_sha256,
        expected_projection_validation_sha256=(
            args.expected_projection_validation_sha256
        ),
    )
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "status",
                    "row_count",
                    "processing_authorized_count",
                    "t1_source_kind_counts",
                    "normalization_readiness_counts",
                    "output_manifest_sha256",
                    "release_state",
                )
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
