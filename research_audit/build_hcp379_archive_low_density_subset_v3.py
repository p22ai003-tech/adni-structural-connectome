#!/home/ec2-user/fsl/bin/python
"""Build the exact 30-unit archive low-density corrected-recovery subset."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


EXP = Path("/home/ec2-user/exp")
AUDIT_ROOT = (
    EXP
    / "research_audit/outputs/hcp379_archive_corrected_input_audit_v2"
)
ARCHIVE_AUDIT = AUDIT_ROOT / "archive_corrected_input_audit.csv"
ARCHIVE_AUDIT_SUMMARY = (
    AUDIT_ROOT / "archive_corrected_input_audit_summary.json"
)
CALIBRATION_SELECTION = (
    AUDIT_ROOT / "archive_corrected_calibration_selection.json"
)
TOPOLOGY = (
    EXP
    / "research_audit/outputs/hcp379_density_execution_topology_v3/"
    "hcp379_density_execution_subjects.csv"
)
OUTPUT_ROOT = (
    EXP
    / "research_audit/outputs/hcp379_archive_low_density_subset_v3"
)
FULL_CSV = OUTPUT_ROOT / "archive_low_density_30.csv"
EXECUTION_CSV = OUTPUT_ROOT / "archive_low_density_execution_28.csv"
SUMMARY = OUTPUT_ROOT / "archive_low_density_subset_summary.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise ValueError(f"required regular file missing: {resolved}")
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        newline="",
        encoding="utf-8",
        delete=False,
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        encoding="utf-8",
        delete=False,
    ) as handle:
        json.dump(dict(value), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def build() -> dict[str, Any]:
    archive_summary = load_json(ARCHIVE_AUDIT_SUMMARY)
    if (
        archive_summary.get("status") != "PASS"
        or archive_summary.get("audited_n") != 86
        or archive_summary.get("ready_n") != 86
        or archive_summary.get("diagnosis_or_outcome_fields_used") is not False
    ):
        raise ValueError("archive corrected input audit is not exact PASS")
    audit_rows = read_csv(ARCHIVE_AUDIT)
    audit = {row["unit"]: row for row in audit_rows}
    if len(audit) != 86:
        raise ValueError("archive corrected input audit is not unique 86")

    topology_rows = read_csv(TOPOLOGY)
    low_topology = {
        row["unit"]: row
        for row in topology_rows
        if row["execution_lane"] == "corrected_archive_low_30"
    }
    if len(low_topology) != 30 or not set(low_topology).issubset(audit):
        raise ValueError("archive low-density topology is not exact 30")

    calibration = load_json(CALIBRATION_SELECTION)
    calibration_units = {
        str(row["unit"]) for row in calibration.get("units", [])
    }
    if len(calibration_units) != 6:
        raise ValueError("archive calibration selection is not exact six")
    reuse_units = set(low_topology) & calibration_units
    execution_units = set(low_topology) - reuse_units
    if len(reuse_units) != 2 or len(execution_units) != 28:
        raise ValueError("archive low calibration/execution split differs")

    full_rows = []
    execution_rows = []
    for unit in sorted(low_topology):
        row = dict(audit[unit])
        row.update(
            {
                "density_group": low_topology[unit]["density_group"],
                "current_density": low_topology[unit]["current_density"],
                "archive_calibration_primary_reuse": unit in reuse_units,
                "selected_by_diagnosis_or_outcome": False,
            }
        )
        full_rows.append(row)
        if unit in execution_units:
            execution_rows.append(row)
    groups = Counter(row["density_group"] for row in full_rows)
    if groups != {
        "D1_NEAR_THRESHOLD": 28,
        "D2_MODERATE_LOW": 2,
    }:
        raise ValueError(f"archive low-density groups differ: {groups}")
    if any(
        row["route"]
        not in {
            "READY_FOR_CORRECTED_PRETRACT_FROM_EDDY",
            "READY_AFTER_GRADIENT_HEADER_REPAIR",
        }
        for row in full_rows
    ):
        raise ValueError("archive low-density subset contains a nonready input")

    atomic_csv(FULL_CSV, full_rows)
    atomic_csv(EXECUTION_CSV, execution_rows)
    route_counts = Counter(row["route"] for row in execution_rows)
    summary = {
        "schema_version": "1.0.0",
        "record_type": (
            "diagnosis_blind_hcp379_archive_low_density_input_subset"
        ),
        "status": "PASS",
        "generated_utc": utc_now(),
        "diagnosis_labels_used": False,
        "diagnosis_or_outcomes_used": False,
        "lane_n": 30,
        "audited_n": 28,
        "ready_n": 28,
        "archive_calibration_primary_reuse_n": 2,
        "new_execution_n": 28,
        "density_group_counts": dict(sorted(groups.items())),
        "route_counts": dict(sorted(route_counts.items())),
        "reuse_units": sorted(reuse_units),
        "execution_units": sorted(execution_units),
        "historical_nodes_b0_reuse_allowed": False,
        "records": {
            "full_subset": file_record(FULL_CSV),
            "execution_subset": file_record(EXECUTION_CSV),
            "archive_input_audit": file_record(ARCHIVE_AUDIT),
            "archive_input_audit_summary": file_record(
                ARCHIVE_AUDIT_SUMMARY
            ),
            "calibration_selection": file_record(CALIBRATION_SELECTION),
            "density_execution_topology": file_record(TOPOLOGY),
            "builder": file_record(Path(__file__)),
        },
    }
    atomic_json(SUMMARY, summary)
    return summary


if __name__ == "__main__":
    result = build()
    print(json.dumps(result, indent=2, sort_keys=True))
