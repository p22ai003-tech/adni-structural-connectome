#!/usr/bin/env python3
"""Audit the 86 archive subjects for a corrected-ACT calibration/rerun.

The recovered archive matrices are technically valid, but their tractograms
were generated without ACT.  They therefore remain a distinct route until a
diagnosis-blind comparison against the locked corrected-ACT recipe is
performed.  This script:

* audits whether every archive subject can reuse its existing Eddy/anatomy;
* selects six technically diverse, outcome-blind calibration subjects; and
* writes immutable-input records for the later corrected-ACT extension.

It performs read-only input inspection.  It does not run image processing,
tractography, statistics, or diagnosis-aware selection.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


EXP = Path("/home/ec2-user/exp")
HCP_ROOT = Path("/data/derivatives/hcp379_v2")
ARCHIVE_UNITS = (
    EXP
    / "research_audit/outputs/hcp_rerun_triage_v1/"
    "archive_recovery_86.txt"
)
ACQUISITION = (
    EXP
    / "research_audit/outputs/connectome_v2_input_manifest_v2.csv"
)
ARCHIVE_SUMMARY = HCP_ROOT / "manifests/archive_restore_summary.json"
AUDIT_SOURCE = EXP / "scripts/hcp/audit_hcp379_scaleup_inputs_v2.py"
OUTPUT_ROOT = (
    EXP
    / "research_audit/outputs/"
    "hcp379_archive_corrected_input_audit_v2"
)
EXPECTED_N = 86
CALIBRATION_N = 6
READY_ROUTES = {
    "READY_FOR_CORRECTED_PRETRACT_FROM_EDDY",
    "READY_AFTER_GRADIENT_HEADER_REPAIR",
}


def load_audit_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "audit_hcp379_scaleup_inputs_v2", AUDIT_SOURCE
    )
    if spec is None or spec.loader is None:
        raise ImportError(AUDIT_SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


AUDIT = load_audit_module()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"not a regular file: {path}")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        encoding="utf-8",
        delete=False,
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


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


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def archive_units() -> list[str]:
    units = [
        value.strip()
        for value in ARCHIVE_UNITS.read_text(encoding="utf-8").splitlines()
        if value.strip()
    ]
    if len(units) != EXPECTED_N or len(set(units)) != EXPECTED_N:
        raise ValueError(
            f"archive identity is {len(units)}/{len(set(units))}, not 86"
        )
    return sorted(units)


def technical_acquisition() -> dict[str, dict[str, str]]:
    output: dict[str, dict[str, str]] = {}
    for row in AUDIT.read_csv(ACQUISITION):
        unit = f"{row['subject_id']}_I{row['dti_image_id']}"
        if unit in output:
            raise ValueError(f"duplicate acquisition identity: {unit}")
        output[unit] = {
            "t1_image_id": str(row["t1_image_id"]),
            "t1_source_id": str(row["t1_source_id"]),
            "dti_source_id": str(row["dti_source_id"]),
            "dti_raw_bundle_sha256": str(
                row["dti_raw_bundle_sha256"]
            ),
            "dwi_bvec_path": str(row.get("dwi_bvec_path", "")),
            "dwi_bval_path": str(row.get("dwi_bval_path", "")),
            "manufacturer": str(row.get("manufacturer", "")),
            "scanner_model": str(row.get("scanner_model", "")),
            "protocol": str(row.get("protocol", "")),
            "phase": str(row.get("phase", "")),
            "t1_source_kind": str(row.get("t1_source_kind", "")),
        }
    if len(output) != 530:
        raise ValueError(f"acquisition manifest has {len(output)} units")
    return output


def archive_metrics() -> dict[str, dict[str, Any]]:
    summary = read_json(ARCHIVE_SUMMARY)
    subjects = summary.get("subjects")
    if (
        not isinstance(subjects, list)
        or len(subjects) != EXPECTED_N
        or summary.get("status_counts") != {"PASS_ALL_NINE": EXPECTED_N}
    ):
        raise ValueError("archive 86/86 PASS summary is absent")
    output: dict[str, dict[str, Any]] = {}
    for row in subjects:
        if not isinstance(row, dict):
            raise TypeError("archive subject record is not a mapping")
        unit = str(row.get("fsid", ""))
        if not unit or unit in output:
            raise ValueError(f"invalid archive subject identity: {unit}")
        output[unit] = row
    return output


def numeric_value(row: Mapping[str, Any], field: str) -> float:
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError):
        return 0.0
    return value if math.isfinite(value) else 0.0


def select_diverse(
    rows: list[dict[str, Any]], count: int
) -> list[str]:
    """Deterministic farthest-point sample using technical features only."""

    if len(rows) < count:
        raise ValueError("not enough technically ready archive subjects")
    numeric_fields = (
        "archive_density",
        "archive_assignment_fraction",
        "archive_mask_volume_mm3",
        "gradient_rows",
        "maximum_b_value",
    )
    categorical_fields = (
        "route",
        "dwi_size",
        "dwi_spacing_mm",
        "shell_sizes",
        "manufacturer",
        "scanner_model",
        "protocol",
        "phase",
        "t1_source_kind",
    )
    ranges: dict[str, tuple[float, float]] = {}
    for field in numeric_fields:
        values = [numeric_value(row, field) for row in rows]
        ranges[field] = (min(values), max(values))

    def distance(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
        score = 0.0
        for field in numeric_fields:
            low, high = ranges[field]
            scale = high - low
            if scale > 0:
                delta = (
                    numeric_value(left, field)
                    - numeric_value(right, field)
                ) / scale
                score += delta * delta
        score += sum(
            str(left.get(field, "")) != str(right.get(field, ""))
            for field in categorical_fields
        ) / len(categorical_fields)
        return score

    by_unit = {str(row["unit"]): row for row in rows}
    ordered = sorted(
        rows,
        key=lambda row: (
            numeric_value(row, "archive_density"),
            str(row["unit"]),
        ),
    )
    selected = [str(ordered[0]["unit"])]
    maximum = max(
        rows,
        key=lambda row: (
            numeric_value(row, "archive_density"),
            str(row["unit"]),
        ),
    )
    maximum_unit = str(maximum["unit"])
    if maximum_unit not in selected:
        selected.append(maximum_unit)
    while len(selected) < count:
        candidates = [
            row for row in rows if str(row["unit"]) not in selected
        ]
        candidate = max(
            candidates,
            key=lambda row: (
                min(
                    distance(row, by_unit[unit])
                    for unit in selected
                ),
                str(row["unit"]),
            ),
        )
        selected.append(str(candidate["unit"]))
    return selected


def self_test() -> None:
    rows = []
    for index in range(10):
        rows.append(
            {
                "unit": f"u{index:02d}",
                "archive_density": index / 10,
                "archive_assignment_fraction": 0.5 + index / 100,
                "archive_mask_volume_mm3": 1000 + index,
                "gradient_rows": 40 + index,
                "maximum_b_value": 1000,
                "route": "a" if index < 5 else "b",
                "dwi_size": "1",
                "dwi_spacing_mm": "2",
                "shell_sizes": "4;36",
                "manufacturer": "m1" if index % 2 else "m2",
                "scanner_model": "s",
                "protocol": "p",
                "phase": "x",
                "t1_source_kind": "n",
            }
        )
    selected = select_diverse(rows, 6)
    if (
        len(selected) != 6
        or len(set(selected)) != 6
        or "u00" not in selected
        or "u09" not in selected
    ):
        raise AssertionError(selected)
    print("SELF_TEST_PASS")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if not 1 <= args.workers <= 32:
        raise ValueError("--workers must be in 1..32")

    units = archive_units()
    acquisition = technical_acquisition()
    archive = archive_metrics()
    if set(units) != set(archive):
        raise ValueError("archive unit list and PASS summary differ")
    missing = sorted(set(units) - set(acquisition))
    if missing:
        raise ValueError(f"acquisition manifest lacks units: {missing}")

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                AUDIT.audit_unit,
                unit,
                recommended_action=(
                    "ARCHIVE_CORRECTED_CALIBRATION_OR_RERUN"
                ),
                acquisition=acquisition[unit],
            ): unit
            for unit in units
        }
        for index, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            unit = str(result["unit"])
            metrics = archive[unit]
            technical = acquisition[unit]
            result.update(
                {
                    "archive_density": metrics.get("density"),
                    "archive_assignment_fraction": metrics.get(
                        "assignment_fraction"
                    ),
                    "archive_connected_nodes": metrics.get(
                        "nodes_present"
                    ),
                    "archive_mask_volume_mm3": metrics.get(
                        "mask_volume_mm3"
                    ),
                    "archive_route": "NO_ACT_RESTORED_TRACK",
                    "manufacturer": technical["manufacturer"],
                    "scanner_model": technical["scanner_model"],
                    "protocol": technical["protocol"],
                    "phase": technical["phase"],
                    "t1_source_kind": technical["t1_source_kind"],
                }
            )
            results.append(result)
            print(
                f"[{utc_now()}] {index}/{EXPECTED_N} {unit} "
                f"{result['route']}",
                flush=True,
            )
    results.sort(key=lambda row: str(row["unit"]))
    route_counts = Counter(str(row["route"]) for row in results)
    ready = [
        row for row in results if str(row["route"]) in READY_ROUTES
    ]
    selected = select_diverse(ready, CALIBRATION_N)
    order = {unit: index + 1 for index, unit in enumerate(selected)}
    for row in results:
        unit = str(row["unit"])
        row["selected_for_corrected_act_calibration"] = unit in order
        row["calibration_selection_order"] = order.get(unit, "")

    selection_rows = [
        next(row for row in results if row["unit"] == unit)
        for unit in selected
    ]
    args.output_root.mkdir(parents=True, exist_ok=True)
    audit_csv = args.output_root / "archive_corrected_input_audit.csv"
    atomic_csv(audit_csv, results)
    calibration_audit_csv = (
        args.output_root / "archive_calibration_input_audit.csv"
    )
    atomic_csv(calibration_audit_csv, selection_rows)
    calibration_audit_summary = {
        "schema_version": "1.0.0",
        "record_type": (
            "diagnosis_blind_hcp379_archive_calibration_input_audit"
        ),
        "status": "PASS",
        "generated_utc": utc_now(),
        "diagnosis_labels_used": False,
        "target_n": CALIBRATION_N,
        "audited_n": len(selection_rows),
        "route_counts": dict(
            sorted(
                Counter(
                    str(row["route"]) for row in selection_rows
                ).items()
            )
        ),
        "parent_audit": file_record(audit_csv),
        "selection_method": (
            "density extremes plus deterministic farthest-point technical "
            "coverage; diagnosis and outcomes unavailable to selection"
        ),
    }
    calibration_audit_summary_path = (
        args.output_root / "archive_calibration_input_audit_summary.json"
    )
    atomic_json(
        calibration_audit_summary_path, calibration_audit_summary
    )
    selection = {
        "schema_version": "1.0.0",
        "record_type": (
            "diagnosis_blind_hcp379_archive_corrected_act_"
            "calibration_selection"
        ),
        "status": "PASS",
        "generated_utc": utc_now(),
        "diagnosis_or_outcome_fields_used": False,
        "selection_method": (
            "density extremes plus deterministic farthest-point coverage "
            "over technical acquisition, geometry, shell, assignment and "
            "mask features"
        ),
        "target_n": CALIBRATION_N,
        "units": [
            {
                "unit": row["unit"],
                "selection_order": order[str(row["unit"])],
                "route": row["route"],
                "archive_density": row["archive_density"],
                "archive_assignment_fraction": row[
                    "archive_assignment_fraction"
                ],
                "archive_mask_volume_mm3": row[
                    "archive_mask_volume_mm3"
                ],
                "dwi_size": row["dwi_size"],
                "dwi_spacing_mm": row["dwi_spacing_mm"],
                "shell_sizes": row["shell_sizes"],
                "manufacturer": row["manufacturer"],
                "scanner_model": row["scanner_model"],
                "protocol": row["protocol"],
                "phase": row["phase"],
                "t1_source_kind": row["t1_source_kind"],
                "dti_source_id": row["dti_source_id"],
                "dti_raw_bundle_sha256": row[
                    "dti_raw_bundle_sha256"
                ],
            }
            for row in selection_rows
        ],
        "source_records": {
            "archive_units": file_record(ARCHIVE_UNITS),
            "acquisition_manifest": file_record(ACQUISITION),
            "archive_summary": file_record(ARCHIVE_SUMMARY),
            "input_audit": file_record(audit_csv),
            "calibration_input_audit": file_record(
                calibration_audit_csv
            ),
            "calibration_input_audit_summary": file_record(
                calibration_audit_summary_path
            ),
            "audit_source": file_record(AUDIT_SOURCE),
        },
    }
    selection_path = (
        args.output_root / "archive_corrected_calibration_selection.json"
    )
    atomic_json(selection_path, selection)
    summary = {
        "schema_version": "1.0.0",
        "record_type": (
            "diagnosis_blind_hcp379_archive_corrected_input_audit"
        ),
        "status": (
            "PASS"
            if len(results) == EXPECTED_N
            and len(ready) == EXPECTED_N
            and len(selected) == CALIBRATION_N
            else "FAIL"
        ),
        "generated_utc": utc_now(),
        "diagnosis_or_outcome_fields_used": False,
        "target_n": EXPECTED_N,
        "audited_n": len(results),
        "ready_n": len(ready),
        "route_counts": dict(sorted(route_counts.items())),
        "calibration_n": len(selected),
        "calibration_units": selected,
        "archive_route_may_enter_uniform_primary_release": False,
        "next_gate": (
            "run the selected six through the locked corrected-ACT recipe, "
            "compare archive versus corrected and independent-seed "
            "differences, then retain or rerun all 86"
        ),
        "records": {
            "input_audit": file_record(audit_csv),
            "selection": file_record(selection_path),
        },
    }
    atomic_json(
        args.output_root / "archive_corrected_input_audit_summary.json",
        summary,
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
