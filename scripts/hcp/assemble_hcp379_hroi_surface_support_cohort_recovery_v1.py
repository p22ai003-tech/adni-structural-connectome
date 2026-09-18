#!/usr/bin/env python3
"""Assemble the exact 530-unit HROI cohort after isolated 114 recovery.

The original cohort attempt produced 529 valid uniform-policy candidates and
one input-route failure for the already known isolated FreeSurfer subject
``114_S_6347_I1344943``.  This assembler rehashes all 529 records, verifies
the independently validated isolated recovery, and emits a new non-
overwriting cohort summary with the same HROI policy across all 530 units.

It does not adopt the policy, start pretract, select a streamline count, or
authorize final release.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


EXP = Path("/home/ec2-user/exp")
FAILED_SUMMARY = Path(
    "/data/derivatives/hcp379_v2/source_label_repair_v1/"
    "hroi_surface_support_cohort_v1/attempts/"
    "20260728T062159.667591Z-hroi-cohort-build/summary.json"
)
RECOVERY_SUMMARY = Path(
    "/data/derivatives/hcp379_v2/source_label_repair_v1/"
    "hroi_surface_support_114_recovery_v1/attempts/"
    "20260728T080722.474773Z-hroi-114-recovery/summary.json"
)
RECOVERY_VALIDATION = (
    EXP
    / "research_audit/outputs/"
    "hcp379_hroi_surface_support_114_recovery_v1/validation.json"
)
INVENTORY = Path(
    "/data/derivatives/hcp379_v2/audits/live_cohort_inventory_v1/"
    "hcp379_live_cohort_inventory_530.csv"
)
BASE_BUILDER = (
    EXP / "scripts/hcp/build_hcp379_hroi_surface_support_v1.py"
)
ATTEMPTS = Path(
    "/data/derivatives/hcp379_v2/source_label_repair_v1/"
    "hroi_surface_support_cohort_v1/attempts"
)
UNIT = "114_S_6347_I1344943"
EXPECTED_UNITS = 530


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"not a regular file: {path}")
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def verify(record: Any, label: str) -> Path:
    if not isinstance(record, dict):
        raise ValueError(f"{label}: record missing")
    path = Path(str(record.get("path", ""))).resolve()
    if file_record(path) != record:
        raise ValueError(f"{label}: artifact record differs")
    return path


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def inventory_units() -> list[str]:
    with INVENTORY.open(newline="", encoding="utf-8-sig") as handle:
        values = sorted(
            str(row.get("unit", "")).strip()
            for row in csv.DictReader(handle)
        )
    if (
        len(values) != EXPECTED_UNITS
        or len(set(values)) != EXPECTED_UNITS
        or any(not value for value in values)
    ):
        raise ValueError("live cohort inventory differs")
    return values


def validate_manifest(
    unit: str,
    manifest_path: Path,
    candidate_path: Path,
) -> dict[str, Any]:
    manifest = load_json(manifest_path)
    if (
        manifest.get("record_type")
        != "diagnosis_blind_hcp379_hroi_surface_support_candidate"
        or manifest.get("status") != "PASS_ALL_379_SOURCE_LABELS"
        or manifest.get("diagnosis_labels_used") is not False
        or manifest.get("source_files_modified") is not False
        or manifest.get("unit") != unit
        or manifest.get("present_source_label_n") != 379
        or manifest.get("missing_source_labels") != []
        or manifest.get("policy", {}).get("scope")
        != "bilateral HCP-MMP1 H_ROI only"
        or manifest.get("policy", {}).get("support_tissue")
        != "cerebral white matter in both aseg and source"
        or manifest.get("policy", {}).get(
            "subcortical_voxels_overwritten"
        )
        != 0
        or manifest.get("policy", {}).get(
            "other_cortical_labels_overwritten"
        )
        != 0
        or manifest.get("candidate") != file_record(candidate_path)
    ):
        raise ValueError(f"{unit}: HROI candidate contract differs")
    left = int(
        manifest.get("hemispheres", {})
        .get("lh", {})
        .get("white_surface_support_added_voxels", -1)
    )
    right = int(
        manifest.get("hemispheres", {})
        .get("rh", {})
        .get("white_surface_support_added_voxels", -1)
    )
    changed = int(manifest.get("changed_voxel_n", -1))
    if left <= 0 or right <= 0 or changed != left + right:
        raise ValueError(f"{unit}: changed-voxel accounting differs")
    return {
        "changed_voxel_n": changed,
        "left_added_voxel_n": left,
        "right_added_voxel_n": right,
    }


def bind_inputs() -> dict[str, Any]:
    failed = load_json(FAILED_SUMMARY)
    failed_rows = {
        str(row.get("unit", "")): row
        for row in failed.get("units", [])
    }
    if (
        failed.get("record_type")
        != "diagnosis_blind_hcp379_hroi_surface_support_cohort_summary"
        or failed.get("status")
        != "FAIL_INCOMPLETE_HROI_SUPPORT_COHORT"
        or failed.get("diagnosis_labels_used") is not False
        or failed.get("source_files_modified") is not False
        or failed.get("expected_unit_n") != EXPECTED_UNITS
        or failed.get("valid_unit_n") != EXPECTED_UNITS - 1
        or failed.get("failure_n") != 1
        or set(failed.get("failures", {})) != {UNIT}
        or len(failed_rows) != EXPECTED_UNITS - 1
        or UNIT in failed_rows
    ):
        raise ValueError("failed 529-unit cohort evidence differs")

    retained: dict[str, dict[str, Any]] = {}
    for unit, row in sorted(failed_rows.items()):
        manifest_path = verify(
            row.get("candidate_manifest"),
            f"{unit}:candidate_manifest",
        )
        candidate_path = verify(
            row.get("candidate"),
            f"{unit}:candidate",
        )
        accounting = validate_manifest(
            unit,
            manifest_path,
            candidate_path,
        )
        if (
            int(row.get("changed_voxel_n", -1))
            != accounting["changed_voxel_n"]
            or int(row.get("left_added_voxel_n", -1))
            != accounting["left_added_voxel_n"]
            or int(row.get("right_added_voxel_n", -1))
            != accounting["right_added_voxel_n"]
        ):
            raise ValueError(f"{unit}: cohort row accounting differs")
        retained[unit] = {
            **row,
            "status": "REHASHED_VALID_FROM_FAILED_COHORT",
        }

    recovery = load_json(RECOVERY_SUMMARY)
    validation = load_json(RECOVERY_VALIDATION)
    if (
        recovery.get("record_type")
        != (
            "diagnosis_blind_hcp379_hroi_surface_support_"
            "114_recovery_summary"
        )
        or recovery.get("status")
        != "PASS_114_HROI_RECOVERY_FOR_COHORT_MERGE"
        or recovery.get("unit") != UNIT
        or recovery.get("diagnosis_labels_used") is not False
        or recovery.get("source_files_modified") is not False
        or recovery.get("failed_standard_candidate_modified") is not False
        or recovery.get("cohort_recipe_selected") is not False
        or recovery.get("selected_count_tractography_authorized")
        is not False
        or recovery.get("final_release_authorized") is not False
        or validation.get("status") != "PASS"
        or validation.get("runtime_summary_detected") is not True
        or validation.get("passed_checks") != 6
        or validation.get("total_checks") != 6
    ):
        raise ValueError("isolated 114 recovery evidence differs")
    manifest_path = verify(
        recovery.get("candidate_manifest"),
        "114:candidate_manifest",
    )
    candidate_path = verify(
        recovery.get("candidate"),
        "114:candidate",
    )
    accounting = validate_manifest(
        UNIT,
        manifest_path,
        candidate_path,
    )
    retained[UNIT] = {
        "unit": UNIT,
        "status": "RECOVERED_VALID_ISOLATED_INPUT_ROUTE",
        "candidate_manifest": file_record(manifest_path),
        "candidate": file_record(candidate_path),
        **accounting,
    }

    expected = inventory_units()
    if sorted(retained) != expected:
        raise ValueError("recovered cohort does not match exact inventory")
    return {
        "rows": [retained[unit] for unit in expected],
        "records": {
            "failed_cohort_summary": file_record(FAILED_SUMMARY),
            "isolated_114_recovery_summary": file_record(
                RECOVERY_SUMMARY
            ),
            "isolated_114_recovery_validation": file_record(
                RECOVERY_VALIDATION
            ),
            "inventory": file_record(INVENTORY),
        },
    }


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def execute() -> dict[str, Any]:
    bound = bind_inputs()
    attempt = ATTEMPTS / f"{stamp()}-hroi-cohort-recovery"
    attempt.mkdir(parents=True, exist_ok=False)
    preflight = {
        "schema_version": "1.0.0",
        "record_type": (
            "diagnosis_blind_hcp379_hroi_surface_support_"
            "cohort_recovery_preflight"
        ),
        "generated_utc": utc_now(),
        "status": "READY_TO_ASSEMBLE_EXACT_530",
        "diagnosis_labels_used": False,
        "source_files_modified": False,
        "unit_n": len(bound["rows"]),
        "cohort_recipe_selected": False,
        "selected_count_tractography_authorized": False,
        "final_release_authorized": False,
        "records": bound["records"],
        "implementation": file_record(Path(__file__)),
    }
    atomic_json(attempt / "preflight.json", preflight)
    changed = [
        int(row["changed_voxel_n"]) for row in bound["rows"]
    ]
    summary = {
        "schema_version": "1.0.0",
        "record_type": (
            "diagnosis_blind_hcp379_hroi_surface_support_"
            "cohort_summary"
        ),
        "generated_utc": utc_now(),
        "status": "PASS_COHORT_UNIFORM_HROI_SUPPORT_530",
        "diagnosis_labels_used": False,
        "source_files_modified": False,
        "release_adopted": False,
        "expected_unit_n": EXPECTED_UNITS,
        "valid_unit_n": EXPECTED_UNITS,
        "reused_unit_n": EXPECTED_UNITS - 1,
        "built_unit_n": 1,
        "failure_n": 0,
        "failures": {},
        "changed_voxel_summary": {
            "minimum": min(changed),
            "maximum": max(changed),
            "total": sum(changed),
        },
        "units": bound["rows"],
        "preflight": file_record(attempt / "preflight.json"),
        "builder": file_record(BASE_BUILDER),
        "runner": file_record(Path(__file__)),
        "recovery_records": bound["records"],
    }
    atomic_json(attempt / "summary.json", summary)
    return {
        "attempt": str(attempt),
        "summary": str(attempt / "summary.json"),
        "status": summary["status"],
        "valid_unit_n": summary["valid_unit_n"],
        "failure_n": summary["failure_n"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    bound = bind_inputs()
    if not args.execute:
        value = {
            "status": "READY_TO_ASSEMBLE_EXACT_530",
            "valid_unit_n": len(bound["rows"]),
            "diagnosis_labels_used": False,
            "source_files_modified": False,
            "cohort_recipe_selected": False,
            "selected_count_tractography_authorized": False,
            "final_release_authorized": False,
        }
    else:
        value = execute()
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
