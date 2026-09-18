#!/usr/bin/env python3
"""Build and validate the cohort-uniform H-ROI source support for all 530 units.

Existing valid candidates are rehashed and reused.  Missing candidates are
built in parallel through the versioned single-unit builder.  Invalid or
partial existing directories are never overwritten or deleted.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


ROOT = Path("/home/ec2-user/exp")
BUILDER = ROOT / "scripts/hcp/build_hcp379_hroi_surface_support_v1.py"
PYTHON = Path("/home/ec2-user/fsl/bin/python")
INVENTORY = Path(
    "/data/derivatives/hcp379_v2/audits/live_cohort_inventory_v1/"
    "hcp379_live_cohort_inventory_530.csv"
)
CANDIDATE_ROOT = Path(
    "/data/derivatives/hcp379_v2/source_label_repair_v1/"
    "hroi_surface_support_v1"
)
ATTEMPT_ROOT = Path(
    "/data/derivatives/hcp379_v2/source_label_repair_v1/"
    "hroi_surface_support_cohort_v1/attempts"
)
EXPECTED_UNITS = 530


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
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


def units() -> list[str]:
    with INVENTORY.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    values = sorted(str(row.get("unit", "")).strip() for row in rows)
    if (
        len(values) != EXPECTED_UNITS
        or len(set(values)) != EXPECTED_UNITS
        or any(not value for value in values)
    ):
        raise ValueError("live cohort inventory is not the exact 530-unit set")
    return values


def validate_existing(unit: str) -> dict[str, Any] | None:
    directory = CANDIDATE_ROOT / unit
    manifest_path = directory / "candidate_manifest.json"
    if not directory.exists():
        return None
    if not directory.is_dir() or directory.is_symlink():
        raise ValueError(f"{unit}: candidate path is not a regular directory")
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise ValueError(f"{unit}: partial candidate directory exists")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    candidate = Path(str(manifest.get("candidate", {}).get("path", "")))
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
        or manifest.get("policy", {}).get("subcortical_voxels_overwritten")
        != 0
        or manifest.get("policy", {}).get(
            "other_cortical_labels_overwritten"
        )
        != 0
        or not candidate.is_file()
        or candidate.is_symlink()
        or file_record(candidate) != manifest.get("candidate")
        or manifest.get("implementation") != file_record(BUILDER)
    ):
        raise ValueError(f"{unit}: existing candidate contract differs")
    return {
        "unit": unit,
        "status": "REUSED_VALID",
        "candidate_manifest": file_record(manifest_path),
        "candidate": file_record(candidate),
        "changed_voxel_n": int(manifest["changed_voxel_n"]),
        "left_added_voxel_n": int(
            manifest["hemispheres"]["lh"][
                "white_surface_support_added_voxels"
            ]
        ),
        "right_added_voxel_n": int(
            manifest["hemispheres"]["rh"][
                "white_surface_support_added_voxels"
            ]
        ),
    }


def build_one(unit: str) -> dict[str, Any]:
    result = subprocess.run(
        [
            str(PYTHON),
            str(BUILDER),
            "--unit",
            unit,
            "--output-root",
            str(CANDIDATE_ROOT),
            "--execute",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    stdout_path = CANDIDATE_ROOT / f"{unit}.builder.stdout.json"
    stderr_path = CANDIDATE_ROOT / f"{unit}.builder.stderr.log"
    with stdout_path.open("x", encoding="utf-8") as handle:
        handle.write(result.stdout)
    with stderr_path.open("x", encoding="utf-8") as handle:
        handle.write(result.stderr)
    if result.returncode != 0:
        raise RuntimeError(
            f"{unit}: builder exited {result.returncode}; "
            f"stderr={stderr_path}"
        )
    validated = validate_existing(unit)
    if validated is None:
        raise RuntimeError(f"{unit}: builder did not create a candidate")
    validated["status"] = "BUILT_VALID"
    validated["builder_stdout"] = file_record(stdout_path)
    validated["builder_stderr"] = file_record(stderr_path)
    return validated


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= 16:
        parser.error("--workers must be in 1..16")

    unit_list = units()
    reused: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    invalid: dict[str, str] = {}
    for unit in unit_list:
        try:
            record = validate_existing(unit)
            if record is None:
                missing.append(unit)
            else:
                reused[unit] = record
        except Exception as exc:
            invalid[unit] = f"{type(exc).__name__}:{exc}"
    preflight = {
        "schema_version": "1.0.0",
        "record_type": (
            "diagnosis_blind_hcp379_hroi_surface_support_cohort_preflight"
        ),
        "generated_utc": utc_now(),
        "status": (
            "INVALID_EXISTING_CANDIDATES"
            if invalid
            else "READY_TO_BUILD_MISSING"
            if missing
            else "ALL_530_ALREADY_VALID"
        ),
        "diagnosis_labels_used": False,
        "source_files_modified": False,
        "expected_unit_n": EXPECTED_UNITS,
        "valid_existing_n": len(reused),
        "missing_n": len(missing),
        "invalid_existing_n": len(invalid),
        "missing_units": missing,
        "invalid_existing": invalid,
        "workers": args.workers,
        "builder": file_record(BUILDER),
        "inventory": file_record(INVENTORY),
    }
    if not args.execute:
        print(json.dumps(preflight, sort_keys=True))
        return 0 if not invalid else 2
    if invalid:
        print(json.dumps(preflight, sort_keys=True))
        return 2

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    attempt = ATTEMPT_ROOT / f"{stamp}-hroi-cohort-build"
    attempt.mkdir(parents=True, exist_ok=False)
    atomic_json(attempt / "preflight.json", preflight)
    built: dict[str, dict[str, Any]] = {}
    failures: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_by_unit = {
            executor.submit(build_one, unit): unit for unit in missing
        }
        for future in as_completed(future_by_unit):
            unit = future_by_unit[future]
            try:
                built[unit] = future.result()
            except Exception as exc:
                failures[unit] = f"{type(exc).__name__}:{exc}"

    all_records = {**reused, **built}
    ordered = [all_records[unit] for unit in sorted(all_records)]
    changed = [int(row["changed_voxel_n"]) for row in ordered]
    status = (
        "PASS_COHORT_UNIFORM_HROI_SUPPORT_530"
        if not failures and len(ordered) == EXPECTED_UNITS
        else "FAIL_INCOMPLETE_HROI_SUPPORT_COHORT"
    )
    summary = {
        "schema_version": "1.0.0",
        "record_type": (
            "diagnosis_blind_hcp379_hroi_surface_support_cohort_summary"
        ),
        "generated_utc": utc_now(),
        "status": status,
        "diagnosis_labels_used": False,
        "source_files_modified": False,
        "release_adopted": False,
        "expected_unit_n": EXPECTED_UNITS,
        "valid_unit_n": len(ordered),
        "reused_unit_n": len(reused),
        "built_unit_n": len(built),
        "failure_n": len(failures),
        "failures": failures,
        "changed_voxel_summary": {
            "minimum": min(changed) if changed else None,
            "maximum": max(changed) if changed else None,
            "total": sum(changed),
        },
        "units": ordered,
        "preflight": file_record(attempt / "preflight.json"),
        "builder": file_record(BUILDER),
        "runner": file_record(Path(__file__)),
    }
    atomic_json(attempt / "summary.json", summary)
    print(
        json.dumps(
            {
                "attempt": str(attempt),
                "status": status,
                "valid_unit_n": len(ordered),
                "built_unit_n": len(built),
                "failure_n": len(failures),
            },
            sort_keys=True,
        )
    )
    return 0 if status == "PASS_COHORT_UNIFORM_HROI_SUPPORT_530" else 1


if __name__ == "__main__":
    raise SystemExit(main())
