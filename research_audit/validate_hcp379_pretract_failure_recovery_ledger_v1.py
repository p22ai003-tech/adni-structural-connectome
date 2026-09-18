#!/usr/bin/env python3
"""Independently validate the exact 515-unit pre-tract failure ledger."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXP = Path("/home/ec2-user/exp")
ROOT = (
    EXP
    / "research_audit/outputs/hcp379_pretract_failure_recovery_ledger_v1"
)
LEDGER = ROOT / "ledger.json"
CSV_PATH = ROOT / "ledger.csv"
TOPOLOGY = (
    EXP
    / "research_audit/outputs/hcp379_balanced_release_topology_v1/"
    "topology.json"
)
OUTPUT = ROOT / "validation.json"
EXPECTED_N = 515


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


def main() -> int:
    checks: dict[str, dict[str, Any]] = {}

    def check(name: str, passed: bool, evidence: Any) -> None:
        checks[name] = {
            "status": "PASS" if passed else "FAIL",
            "evidence": evidence,
        }

    ledger = load_json(LEDGER)
    topology = load_json(TOPOLOGY)
    with CSV_PATH.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    units = [row["unit"] for row in rows]
    topology_units = [str(row["unit"]) for row in topology["production"]]
    status_counts = Counter(row["status"] for row in rows)
    failure_counts = Counter(
        row["failure_class"] for row in rows if row["status"] == "FAILED"
    )

    check(
        "exact_515_identity_partition",
        (
            len(rows) == EXPECTED_N
            and len(set(units)) == EXPECTED_N
            and set(units) == set(topology_units)
            and sum(status_counts.values()) == EXPECTED_N
        ),
        {"row_n": len(rows), "unique_n": len(set(units))},
    )
    check(
        "summary_counts_replay",
        (
            ledger.get("status_counts") == dict(sorted(status_counts.items()))
            and ledger.get("failure_class_counts")
            == dict(sorted(failure_counts.items()))
        ),
        {
            "status_counts": dict(status_counts),
            "failure_class_counts": dict(failure_counts),
        },
    )
    state_errors: list[str] = []
    for row in rows:
        path = Path(row["state_path"])
        if not path.is_file():
            if (
                row["status"] not in {"WAITING"}
                and row["failure_class"] not in {
                    "LEGACY_EDDY_ZERO_VOLUMES_RAW_INTACT",
                    "RAW_ZERO_B0_VOLUME_RECOVERABLE",
                }
            ):
                state_errors.append(f"{row['unit']}:missing_state")
            continue
        state = load_json(path)
        if state.get("unit") != row["unit"]:
            state_errors.append(f"{row['unit']}:identity")
        if row["status"] == "FAILED":
            zero_hold = (
                row["failure_class"] in {
                    "LEGACY_EDDY_ZERO_VOLUMES_RAW_INTACT",
                    "RAW_ZERO_B0_VOLUME_RECOVERABLE",
                }
                and row["error"]
                == "INPUT_QC_HOLD:FAIL_ZERO_EDDY_VOLUMES"
            )
            if not zero_hold and (
                state.get("status") != "FAIL_PRETRACT_RECOVERY4"
                or str(state.get("error") or "") != row["error"]
                or not row["failure_class"]
                or not row["recovery_requirement"]
            ):
                state_errors.append(f"{row['unit']}:failure_binding")
        if row["status"] == "PASS_COMPACTED":
            compaction = (
                load_json(Path(row["compaction_path"]))
                if Path(row["compaction_path"]).is_file()
                else {}
            )
            if (
                state.get("status") != "PASS_PRETRACT_RECOVERY4"
                or not Path(row["compaction_path"]).is_file()
                or compaction.get("unit") != row["unit"]
                or compaction.get("status") != "PASS_COMPACTED"
            ):
                state_errors.append(f"{row['unit']}:pass_binding")
    check(
        "subject_states_and_compaction_bind",
        not state_errors,
        {"errors": state_errors},
    )
    row_by_unit = {row["unit"]: row for row in rows}
    eddy_hold_errors: list[str] = []
    for item in topology["production"]:
        pretract = item.get("pretract", {})
        if pretract.get("eddy_integrity_hold") is not True:
            continue
        unit = str(item["unit"])
        row = row_by_unit[unit]
        if (
            row["status"] != "FAILED"
            or row["failure_class"]
            not in {
                "LEGACY_EDDY_ZERO_VOLUMES_RAW_INTACT",
                "RAW_ZERO_B0_VOLUME_RECOVERABLE",
            }
            or row["error"]
            != "INPUT_QC_HOLD:FAIL_ZERO_EDDY_VOLUMES"
        ):
            eddy_hold_errors.append(unit)
    check(
        "eddy_integrity_holds_outrank_worker_state",
        not eddy_hold_errors,
        {
            "hold_n": sum(
                item.get("pretract", {}).get("eddy_integrity_hold") is True
                for item in topology["production"]
            ),
            "errors": eddy_hold_errors,
        },
    )
    forbidden = {
        "diagnosis",
        "outcome",
        "group",
        "cn",
        "mci",
        "ad",
        "density_result",
    }
    header = set(rows[0]) if rows else set()
    check(
        "ledger_is_diagnosis_blind_and_nonexecuting",
        (
            ledger.get("diagnosis_labels_used") is False
            and ledger.get("outcomes_used") is False
            and ledger.get("imaging_executed") is False
            and ledger.get("qc_thresholds_changed") is False
            and ledger.get("non_overwriting") is True
            and not (forbidden & {value.lower() for value in header})
        ),
        {
            "header": sorted(header),
            "forbidden_present": sorted(
                forbidden & {value.lower() for value in header}
            ),
        },
    )
    allowed_failure_classes = {
        "FOD_L0_HARD_NEGATIVE_BURDEN",
        "FOD_L0_SOFT_NEGATIVE_FRACTION",
        "MTNORMALISE_NONPOSITIVE_TISSUE_BALANCE",
        "SPATIAL_BBR_AND_RIGID_FAIL",
        "SPATIAL_BBR_RIGID_AFFINE_FAIL",
        "LEGACY_EDDY_ZERO_VOLUMES_RAW_INTACT",
        "RAW_ZERO_B0_VOLUME_RECOVERABLE",
        "DICOM_GRADIENT_RECONVERSION_EDDY_PENDING",
        "LOCKED_RAW_VALID_HISTORICAL_EDDY_TENSOR_CORRUPTION",
        "PROVISIONAL_ENGINE_FOD_RANGE_GUARD_MISSING",
        "OTHER_TERMINAL_FAILURE",
    }
    check(
        "failure_classes_are_explicit_and_fail_closed",
        (
            set(failure_counts) <= allowed_failure_classes
            and all(
                "require" in row["recovery_requirement"].lower()
                for row in rows
                if row["status"] == "FAILED"
            )
        ),
        {"failure_classes": sorted(failure_counts)},
    )
    fod_class_errors: list[str] = []
    for row in rows:
        error = row["error"].lower()
        failure_class = row["failure_class"]
        if (
            "wmfod_l0_below_hard_tolerance" in error
            and failure_class != "FOD_L0_HARD_NEGATIVE_BURDEN"
        ):
            fod_class_errors.append(
                f"{row['unit']}:hard_error_as_{failure_class}"
            )
        if (
            failure_class == "FOD_L0_SOFT_NEGATIVE_FRACTION"
            and "wmfod_l0_below_hard_tolerance" in error
        ):
            fod_class_errors.append(
                f"{row['unit']}:soft_class_contains_hard_error"
            )
    check(
        "hard_and_soft_fod_failures_are_distinguished",
        not fod_class_errors,
        {"errors": fod_class_errors},
    )
    spatial_class_errors: list[str] = []
    for row in rows:
        error = row["error"].lower()
        failure_class = row["failure_class"]
        if (
            "neither spatial route passed direct gates" in error
            and "affine=[" in error
            and failure_class != "SPATIAL_BBR_RIGID_AFFINE_FAIL"
        ):
            spatial_class_errors.append(
                f"{row['unit']}:affine_error_as_{failure_class}"
            )
    check(
        "three_route_spatial_failures_are_distinguished",
        not spatial_class_errors,
        {"errors": spatial_class_errors},
    )
    records = ledger.get("records", {})
    check(
        "topology_and_csv_hashes_bind",
        (
            records.get("topology", {}).get("sha256")
            == sha256_file(TOPOLOGY)
            and records.get("ledger_csv", {}).get("sha256")
            == sha256_file(CSV_PATH)
        ),
        {
            "topology_sha256": sha256_file(TOPOLOGY),
            "csv_sha256": sha256_file(CSV_PATH),
        },
    )

    passed = sum(item["status"] == "PASS" for item in checks.values())
    payload = {
        "schema_version": "1.0.0",
        "record_type": "hcp379_pretract_failure_recovery_ledger_validation",
        "status": "PASS" if passed == len(checks) else "FAIL",
        "generated_utc": datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "passed_checks": passed,
        "total_checks": len(checks),
        "checks": checks,
    }
    OUTPUT.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
