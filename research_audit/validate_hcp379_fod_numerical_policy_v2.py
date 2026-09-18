#!/usr/bin/env python3
"""Replay the uniform Recovery4 FOD numerical policy over live scalar QC.

The policy is diagnosis blind and does not edit images or prior QC.  It
independently reclassifies every available scalar record using the frozen
soft threshold, hard magnitude limit, and maximum soft-negative fraction.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HCP_ROOT = Path("/data/derivatives/hcp379_v2")
OUTPUT = Path(
    "/home/ec2-user/exp/research_audit/outputs/"
    "hcp379_fod_numerical_policy_v2/validation.json"
)
SCALAR_SOURCE = Path(
    "/home/ec2-user/exp/scripts/hcp/"
    "build_hcp379_pretract_scalar_recovery4_v2.py"
)
SOFT_TOLERANCE = 1.0e-5
HARD_NEGATIVE_TOLERANCE = 1.0e-4
MAXIMUM_SOFT_NEGATIVE_FRACTION = 1.0e-3
RAW_TENSOR_ANOMALY_MAXIMUM = 1.0e-2


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    paths = sorted(
        HCP_ROOT.glob(
            "**/subjects/*/06_preflight/scalar_recovery4_qc.json"
        )
    )
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    for path in paths:
        record = load_json(path)
        unit = str(record.get("unit", ""))
        mask_voxels = int(record.get("mask_voxels") or 0)
        counts = record.get("fod_tissue_numerical_counts") or {}
        ranges = record.get("fod_tissue_ranges") or {}
        if not unit or mask_voxels <= 0:
            errors.append(f"{path}:identity_or_mask")
            continue
        tissue_results: dict[str, Any] = {}
        failures: list[str] = []
        for tissue in ("wmfod_l0", "gm", "csf"):
            soft_count = int(counts.get(f"{tissue}_below_tolerance", 0))
            fraction = soft_count / mask_voxels
            minimum = float((ranges.get(tissue) or {}).get("min", 0.0))
            hard_pass = minimum >= -HARD_NEGATIVE_TOLERANCE
            fraction_pass = fraction <= MAXIMUM_SOFT_NEGATIVE_FRACTION
            if not hard_pass:
                failures.append(f"{tissue}:hard_magnitude")
            if not fraction_pass:
                failures.append(f"{tissue}:soft_fraction")
            tissue_results[tissue] = {
                "minimum": minimum,
                "soft_negative_voxels": soft_count,
                "soft_negative_fraction": fraction,
                "hard_magnitude_pass": hard_pass,
                "soft_fraction_pass": fraction_pass,
            }
        wmfod_max = float(
            (ranges.get("wmfod_l0") or {}).get("max", 0.0)
        )
        if wmfod_max <= SOFT_TOLERANCE:
            failures.append("wmfod_l0:nonzero")
        records.append(
            {
                "unit": unit,
                "source_qc_path": str(path.resolve()),
                "source_qc_sha256": sha256_file(path),
                "status": "PASS" if not failures else "FAIL",
                "failures": failures,
                "mask_voxels": mask_voxels,
                "tissues": tissue_results,
            }
        )

    duplicate_units = len(records) - len(
        {record["unit"] for record in records}
    )
    passing = sum(record["status"] == "PASS" for record in records)
    worst_fraction = max(
        (
            tissue["soft_negative_fraction"]
            for record in records
            for tissue in record["tissues"].values()
        ),
        default=0.0,
    )
    most_negative = min(
        (
            tissue["minimum"]
            for record in records
            for tissue in record["tissues"].values()
        ),
        default=0.0,
    )
    source_text = SCALAR_SOURCE.read_text(encoding="utf-8")
    source_contract_pass = all(
        fragment in source_text
        for fragment in (
            "FOD_TISSUE_NUMERICAL_TOLERANCE = 1.0e-5",
            "FOD_TISSUE_HARD_NEGATIVE_TOLERANCE = 1.0e-4",
            (
                "FOD_TISSUE_BELOW_TOLERANCE_FRACTION_MAXIMUM "
                "= 1.0e-3"
            ),
            "magnitude_and_fraction_guard",
        )
    )
    checks = {
        "all_available_records_replay": {
            "status": (
                "PASS"
                if paths
                and len(records) == len(paths)
                and not errors
                and duplicate_units == 0
                else "FAIL"
            ),
            "evidence": {
                "source_path_n": len(paths),
                "record_n": len(records),
                "duplicate_unit_n": duplicate_units,
                "errors": errors,
            },
        },
        "uniform_policy_is_stricter_than_tensor_allowance": {
            "status": (
                "PASS"
                if MAXIMUM_SOFT_NEGATIVE_FRACTION
                < RAW_TENSOR_ANOMALY_MAXIMUM
                and HARD_NEGATIVE_TOLERANCE > SOFT_TOLERANCE
                else "FAIL"
            ),
            "evidence": {
                "soft_tolerance": SOFT_TOLERANCE,
                "hard_negative_tolerance": HARD_NEGATIVE_TOLERANCE,
                "maximum_soft_negative_fraction": (
                    MAXIMUM_SOFT_NEGATIVE_FRACTION
                ),
                "raw_tensor_anomaly_maximum": (
                    RAW_TENSOR_ANOMALY_MAXIMUM
                ),
            },
        },
        "all_live_records_pass_uniform_policy": {
            "status": "PASS" if passing == len(records) else "FAIL",
            "evidence": {
                "passing_n": passing,
                "record_n": len(records),
                "failing_units": [
                    record["unit"]
                    for record in records
                    if record["status"] != "PASS"
                ],
                "worst_observed_soft_negative_fraction": worst_fraction,
                "most_negative_observed_tissue_value": most_negative,
            },
        },
        "implementation_matches_policy": {
            "status": "PASS" if source_contract_pass else "FAIL",
            "evidence": {
                "source": str(SCALAR_SOURCE.resolve()),
                "source_sha256": sha256_file(SCALAR_SOURCE),
            },
        },
    }
    passed = sum(value["status"] == "PASS" for value in checks.values())
    payload = {
        "schema_version": "2.0.0",
        "record_type": "hcp379_fod_numerical_policy_validation",
        "status": "PASS" if passed == len(checks) else "FAIL",
        "generated_utc": datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "diagnosis_labels_used": False,
        "outcomes_used": False,
        "imaging_executed": False,
        "prior_qc_modified": False,
        "passed_checks": passed,
        "total_checks": len(checks),
        "checks": checks,
        "records": records,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
