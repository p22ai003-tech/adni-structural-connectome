#!/home/ec2-user/fsl/bin/python
"""Audit all 530 locked HCP-MMP1 source volumes for the 379 source labels.

The audit is diagnosis blind and read only.  It checks the source-space
``HCPMMP1+aseg.mgz`` files before registration or resampling, so an absent
source parcel cannot be mistaken for a BBR/ANTs failure.  Results are written
to a new attempt directory and no parcellation is modified.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import nibabel as nib
import numpy as np


HCP_ROOT = Path("/data/derivatives/hcp379_v2")
LIVE_INVENTORY = (
    HCP_ROOT
    / "audits/live_cohort_inventory_v1/"
    "hcp379_live_cohort_inventory_530.csv"
)
STANDARD_SOURCE_ROOT = Path("/data/derivatives/parc_hcpmmp1")
FREESURFER_RECOVERY_SOURCE = (
    HCP_ROOT
    / "fastsurfer_repair/114_S_6347_I1344943/hcp/"
    "HCPMMP1+aseg.mgz"
)
RELABEL_LUT = (
    STANDARD_SOURCE_ROOT / "hcpmmp1_subcort_relabel.txt"
)
OUTPUT_ROOT = HCP_ROOT / "audits/source_label_completeness_v1/attempts"
EXPECTED_N = 530
EXPECTED_LABEL_N = 379


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
        raise ValueError(f"not a regular source file: {resolved}")
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
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


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0]) if rows else []
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        newline="",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def read_units() -> list[str]:
    with LIVE_INVENTORY.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    units = [str(row.get("unit", "")).strip() for row in rows]
    if (
        len(units) != EXPECTED_N
        or len(set(units)) != EXPECTED_N
        or not all(units)
    ):
        raise ValueError("live inventory is not the exact 530-unit set")
    return sorted(units)


def read_lut() -> dict[int, int]:
    mapping: dict[int, int] = {}
    with RELABEL_LUT.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            source, node = (int(value) for value in stripped.split())
            if source in mapping:
                raise ValueError(f"duplicate source label: {source}")
            mapping[source] = node
    if (
        len(mapping) != EXPECTED_LABEL_N
        or set(mapping.values()) != set(range(1, EXPECTED_LABEL_N + 1))
    ):
        raise ValueError("relabel LUT is not a one-to-one 379-node mapping")
    return mapping


def source_path(unit: str) -> Path:
    standard = STANDARD_SOURCE_ROOT / unit / "HCPMMP1+aseg.mgz"
    if standard.is_file():
        return standard
    if unit == "114_S_6347_I1344943":
        return FREESURFER_RECOVERY_SOURCE
    raise FileNotFoundError(f"source HCP parcellation missing: {unit}")


def audit_unit(
    unit: str, mapping: Mapping[int, int]
) -> dict[str, Any]:
    path = source_path(unit)
    image = nib.load(str(path))
    data = np.asanyarray(image.dataobj)
    if data.ndim != 3 or not np.issubdtype(data.dtype, np.number):
        raise ValueError(f"{unit}: source is not a numeric 3D label image")
    integer = np.asarray(data, dtype=np.int32)
    if not np.array_equal(data, integer):
        raise ValueError(f"{unit}: source contains non-integer labels")
    maximum = max(int(integer.max()), max(mapping))
    counts = np.bincount(integer.ravel(), minlength=maximum + 1)
    missing_source = [
        source for source in sorted(mapping) if int(counts[source]) == 0
    ]
    missing_nodes = [int(mapping[source]) for source in missing_source]
    present_counts = [
        int(counts[source])
        for source in mapping
        if int(counts[source]) > 0
    ]
    record = file_record(path)
    return {
        "unit": unit,
        "source_path": record["path"],
        "source_size_bytes": record["size_bytes"],
        "source_sha256": record["sha256"],
        "source_shape": "x".join(str(value) for value in integer.shape),
        "expected_source_label_n": EXPECTED_LABEL_N,
        "present_source_label_n": EXPECTED_LABEL_N - len(missing_source),
        "missing_source_label_n": len(missing_source),
        "missing_source_labels": ";".join(
            str(value) for value in missing_source
        ),
        "missing_node_ids": ";".join(str(value) for value in missing_nodes),
        "minimum_present_source_voxels": (
            min(present_counts) if present_counts else 0
        ),
        "label_1120_voxels": int(counts[1120]),
        "label_2120_voxels": int(counts[2120]),
        "status": (
            "PASS_ALL_379_SOURCE_LABELS"
            if not missing_source
            else "FAIL_MISSING_SOURCE_LABELS"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.workers <= min(16, os.cpu_count() or 1):
        parser.error("--workers is outside the 1..16 host contract")

    units = read_units()
    mapping = read_lut()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    attempt = OUTPUT_ROOT / f"{stamp}-source-label-audit"
    attempt.mkdir(parents=True, exist_ok=False)
    rows: dict[str, dict[str, Any]] = {}
    failures: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(audit_unit, unit, mapping): unit
            for unit in units
        }
        for future in as_completed(futures):
            unit = futures[future]
            try:
                rows[unit] = future.result()
            except Exception as exc:
                failures[unit] = f"{type(exc).__name__}:{exc}"

    ordered = [rows[unit] for unit in sorted(rows)]
    csv_path = attempt / "hcp379_source_label_completeness_530.csv"
    atomic_csv(csv_path, ordered)
    missing_frequency: Counter[int] = Counter()
    for row in ordered:
        for value in str(row["missing_node_ids"]).split(";"):
            if value:
                missing_frequency[int(value)] += 1
    failed_rows = [
        row for row in ordered if row["status"] != "PASS_ALL_379_SOURCE_LABELS"
    ]
    status = (
        "PASS_ALL_530_SOURCE_LABELS"
        if not failures and not failed_rows and len(ordered) == EXPECTED_N
        else "HOLD_SOURCE_LABEL_REPAIR_REQUIRED"
    )
    summary = {
        "schema_version": "1.0.0",
        "record_type": (
            "diagnosis_blind_hcp379_source_label_completeness_audit"
        ),
        "status": status,
        "generated_utc": utc_now(),
        "diagnosis_labels_used": False,
        "imaging_executed": False,
        "source_files_modified": False,
        "expected_unit_n": EXPECTED_N,
        "audited_unit_n": len(ordered),
        "read_failure_n": len(failures),
        "all_379_source_labels_n": len(ordered) - len(failed_rows),
        "missing_source_labels_n": len(failed_rows),
        "missing_node_frequency": {
            str(key): value for key, value in sorted(missing_frequency.items())
        },
        "minimum_present_source_voxels": (
            min(
                int(row["minimum_present_source_voxels"])
                for row in ordered
            )
            if ordered
            else 0
        ),
        "failures": failures,
        "audit_csv": file_record(csv_path),
        "live_inventory": file_record(LIVE_INVENTORY),
        "relabel_lut": file_record(RELABEL_LUT),
        "implementation": file_record(Path(__file__)),
    }
    atomic_json(attempt / "summary.json", summary)
    print(json.dumps({**summary, "attempt": str(attempt)}, indent=2))
    return 0 if status == "PASS_ALL_530_SOURCE_LABELS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
