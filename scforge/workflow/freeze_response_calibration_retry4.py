#!/usr/bin/env python3
"""Freeze retry4 Phase-A responses through a versioned subset-schema adapter.

The signed recovery subset is a canonical acquisition manifest with
``subject_id`` and ``dti_image_id`` columns.  The original generic freezer
expects a pre-composed ``unit`` column.  This wrapper derives the exact unit
identity in memory, rejects any disagreement, and leaves the signed CSV and
the locked shared calibration module unchanged.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

import yaml


WORKFLOW_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = WORKFLOW_DIR.parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "connectome_v2_retry3.yaml"
RETRY3_SOURCE_MANIFEST = WORKFLOW_DIR / "workflow_source_manifest_retry3.tsv"
EXPECTED_BASE_CALIBRATION_SHA256 = (
    "36c3e1c8c9b10090514968d13df55fbdb528c340d3ffd174d332120db8a90c99"
)
EXPECTED_RETRY3_SOURCE_MANIFEST_SHA256 = (
    "e86f30b18fd3d86dfda4015c4cde94a89f8a9ab4cffb72866370734cd736cdf4"
)
ADAPTER_ID = "retry4-canonical-subset-unit-adapter-v1"

sys.path.insert(0, str(PROJECT_ROOT / "scforge"))

from scforge import response_calibration as calibration  # noqa: E402


def load_execution_subset_technical_metadata_retry4(
    execution_binding: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[str, str]]]:
    """Load diagnosis-blind technical metadata from either subset schema."""

    record = execution_binding.get("execution_subset_manifest")
    subset_path = calibration.verify_file_record(
        record, label="bound execution-subset manifest"
    )
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
        fieldnames = set(reader.fieldnames or ())
        has_explicit_unit = "unit" in fieldnames
        has_canonical_unit = {"subject_id", "dti_image_id"}.issubset(fieldnames)
        if (
            not {"manufacturer", "t1_source_kind"}.issubset(fieldnames)
            or not (has_explicit_unit or has_canonical_unit)
        ):
            raise ValueError("execution-subset manifest lacks technical diversity fields")
        rows = list(reader)

    metadata: dict[str, dict[str, str]] = {}
    for row in rows:
        explicit_unit = str(row.get("unit", "")).strip()
        subject_id = str(row.get("subject_id", "")).strip()
        dti_image_id = str(row.get("dti_image_id", "")).strip()
        canonical_unit = (
            f"{subject_id}_I{dti_image_id}"
            if subject_id and dti_image_id
            else ""
        )
        if explicit_unit and canonical_unit and explicit_unit != canonical_unit:
            raise ValueError(
                "execution-subset explicit and canonical unit identities differ"
            )
        unit = explicit_unit or canonical_unit
        if not unit or unit in metadata:
            raise ValueError("execution-subset technical metadata has duplicate/blank unit")
        raw_manufacturer = str(row.get("manufacturer", "")).strip()
        t1_source_class = str(row.get("t1_source_kind", "")).strip()
        if t1_source_class not in calibration.SUPPORTED_T1_SOURCE_CLASSES:
            raise ValueError(
                "unsupported T1 source class in bound canary metadata: "
                f"{t1_source_class!r}"
            )
        metadata[unit] = {
            "manufacturer": raw_manufacturer,
            "manufacturer_family": calibration.manufacturer_family(
                raw_manufacturer
            ),
            "t1_source_class": t1_source_class,
        }
    if set(metadata) != set(expected_units):
        raise ValueError("bound technical metadata differs from execution unit set")
    return dict(record), metadata


def _write_new_json(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != encoded:
            raise FileExistsError(f"refusing to replace non-identical attestation: {path}")
        return
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def freeze_with_retry4_adapter(
    *,
    phase_a_completion: Path,
    phase_a_manifest: Path,
    output_dir: Path,
    minimum_valid_subjects: int,
    generated_utc: str,
) -> dict[str, Any]:
    base_module = Path(calibration.__file__).resolve()
    if calibration.sha256_file(base_module) != EXPECTED_BASE_CALIBRATION_SHA256:
        raise ValueError("locked shared response-calibration module differs")
    if (
        calibration.sha256_file(RETRY3_SOURCE_MANIFEST)
        != EXPECTED_RETRY3_SOURCE_MANIFEST_SHA256
    ):
        raise ValueError("locked retry3 workflow source manifest differs")

    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    pooling = config["fod"]["response_estimation"]["calibration"]["pooling"]
    original_loader = calibration.load_execution_subset_technical_metadata
    calibration.load_execution_subset_technical_metadata = (
        load_execution_subset_technical_metadata_retry4
    )
    try:
        result = calibration.freeze_response_calibration_from_phase_a(
            phase_a_completion,
            phase_a_manifest,
            output_dir,
            minimum_valid_subjects=minimum_valid_subjects,
            generated_utc=generated_utc,
            responsemean_executable=pooling["executable"],
            expected_responsemean_sha256=pooling["executable_sha256"],
        )
    finally:
        calibration.load_execution_subset_technical_metadata = original_loader

    superseded = (
        output_dir.parent
        / "frozen_calibration"
        / "frozen_response_calibration_manifest.json"
    )
    attestation = {
        "schema_version": "1.0.0",
        "record_type": "response_calibration_freeze_retry4_attestation",
        "status": "PASS",
        "generated_utc": generated_utc,
        "adapter_id": ADAPTER_ID,
        "adapter_semantics": (
            "derive unit as subject_id + '_I' + dti_image_id from the exact "
            "signed acquisition-subset CSV; reject explicit/canonical identity "
            "disagreement; expose only manufacturer and T1-source metadata"
        ),
        "diagnosis_labels_used": False,
        "signed_subset_modified": False,
        "shared_calibration_module_modified": False,
        "authoritative_for_next_gate": True,
        "implementation": {
            "retry4_adapter": calibration.file_record(Path(__file__)),
            "locked_shared_calibration": calibration.file_record(base_module),
            "locked_retry3_source_manifest": calibration.file_record(
                RETRY3_SOURCE_MANIFEST
            ),
            "retry3_config": calibration.file_record(CONFIG_PATH),
            "responsemean": calibration.file_record(pooling["executable"]),
        },
        "inputs": {
            "phase_a_completion": calibration.file_record(phase_a_completion),
            "phase_a_manifest": calibration.file_record(phase_a_manifest),
        },
        "frozen_manifest": calibration.file_record(result["manifest_path"]),
        "pooled_responses": result["pooled_responses"],
        "valid_subject_count": result["valid_subject_count"],
        "invalid_subject_count": result["invalid_subject_count"],
        "valid_pool_technical_diversity": result[
            "valid_pool_technical_diversity"
        ],
        "superseded_unversioned_candidate": (
            {
                **calibration.file_record(superseded),
                "authorized_for_continuation": False,
                "reason": (
                    "created during fail-closed diagnosis before the adapter was "
                    "isolated and hash-attested"
                ),
            }
            if superseded.is_file()
            else None
        ),
    }
    attestation_path = output_dir / "retry4_freeze_attestation.json"
    _write_new_json(attestation_path, attestation)
    result["retry4_freeze_attestation"] = calibration.file_record(attestation_path)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-a-completion", type=Path, required=True)
    parser.add_argument("--phase-a-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-valid-subjects", type=int, required=True)
    args = parser.parse_args()
    result = freeze_with_retry4_adapter(
        phase_a_completion=args.phase_a_completion.resolve(),
        phase_a_manifest=args.phase_a_manifest.resolve(),
        output_dir=args.output_dir.resolve(),
        minimum_valid_subjects=args.minimum_valid_subjects,
        generated_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "valid_subject_count": result["valid_subject_count"],
                "invalid_subject_count": result["invalid_subject_count"],
                "manifest_path": result["manifest_path"],
                "manifest_sha256": result["manifest_sha256"],
                "valid_pool_technical_diversity": result[
                    "valid_pool_technical_diversity"
                ],
                "retry4_freeze_attestation": result[
                    "retry4_freeze_attestation"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
