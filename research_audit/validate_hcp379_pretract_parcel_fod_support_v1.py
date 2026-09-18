#!/home/ec2-user/fsl/bin/python
"""Independently validate the HCP379 parcel/FOD support audit."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np


EXP = Path("/home/ec2-user/exp")
ROOT = (
    EXP
    / "research_audit/outputs/hcp379_pretract_parcel_fod_support_v1"
)
SUMMARY = ROOT / "summary.json"
SUBJECTS = ROOT / "subject_support_summary.csv"
PARCELS = ROOT / "parcel_support.csv"
SOURCE = (
    EXP
    / "research_audit/audit_hcp379_pretract_parcel_fod_support_v1.py"
)
EXPECTED_TOTAL = 530
EXPECTED_PRODUCTION = 515
EXPECTED_CANARIES = 15
EXPECTED_NODES = 379


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if (
        not resolved.is_file()
        or resolved.is_symlink()
        or resolved.stat().st_size <= 0
    ):
        raise FileNotFoundError(resolved)
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256(resolved),
    }


def check(
    checks: dict[str, Any],
    name: str,
    passed: bool,
    evidence: Mapping[str, Any],
) -> None:
    checks[name] = {
        "status": "PASS" if passed else "FAIL",
        "evidence": dict(evidence),
    }


def main() -> None:
    summary = load_json(SUMMARY)
    subjects = read_csv(SUBJECTS)
    parcels = read_csv(PARCELS)
    checks: dict[str, Any] = {}

    unit_set = {row["unit"] for row in subjects}
    roles = Counter(row["role"] for row in subjects)
    check(
        checks,
        "exact_15_plus_515_identity_partition",
        (
            len(subjects) == EXPECTED_TOTAL
            and len(unit_set) == EXPECTED_TOTAL
            and roles == {
                "PRODUCTION": EXPECTED_PRODUCTION,
                "CANARY_REUSE": EXPECTED_CANARIES,
            }
        ),
        {
            "row_n": len(subjects),
            "unique_n": len(unit_set),
            "role_counts": dict(sorted(roles.items())),
        },
    )

    parcel_by_unit: dict[str, list[dict[str, str]]] = {}
    for row in parcels:
        parcel_by_unit.setdefault(row["unit"], []).append(row)
    malformed: list[str] = []
    for unit, rows in parcel_by_unit.items():
        ids = {int(row["parcel_id"]) for row in rows}
        if (
            len(rows) != EXPECTED_NODES
            or ids != set(range(1, EXPECTED_NODES + 1))
        ):
            malformed.append(unit)
    computed_units = {
        row["unit"]
        for row in subjects
        if row["audit_status"] != "NOT_READY"
        and row["audit_status"] != "AUDIT_ERROR"
    }
    check(
        checks,
        "every_computed_subject_has_exact_379_parcel_rows",
        (
            not malformed
            and set(parcel_by_unit) == computed_units
            and len(parcels)
            == EXPECTED_NODES * len(computed_units)
        ),
        {
            "computed_unit_n": len(computed_units),
            "parcel_unit_n": len(parcel_by_unit),
            "parcel_row_n": len(parcels),
            "malformed_units": malformed,
        },
    )

    replay_errors: list[str] = []
    subject_map = {row["unit"]: row for row in subjects}
    for unit, rows in parcel_by_unit.items():
        direct = np.asarray(
            [float(row["direct_fod_support_fraction"]) for row in rows],
            dtype=float,
        )
        near = np.asarray(
            [float(row["within_4mm_fod_support_fraction"]) for row in rows],
            dtype=float,
        )
        subject = subject_map[unit]
        expected = {
            "zero_direct_fod_support_parcel_n": int(
                np.count_nonzero(direct == 0)
            ),
            "lt_1pct_direct_fod_support_parcel_n": int(
                np.count_nonzero(direct < 0.01)
            ),
            "zero_within_4mm_fod_support_parcel_n": int(
                np.count_nonzero(near == 0)
            ),
        }
        for field, value in expected.items():
            if int(subject[field]) != value:
                replay_errors.append(f"{unit}:{field}")
        numeric = {
            "minimum_direct_fod_support_fraction": float(
                np.min(direct)
            ),
            "p05_direct_fod_support_fraction": float(
                np.quantile(direct, 0.05)
            ),
            "median_direct_fod_support_fraction": float(
                np.median(direct)
            ),
        }
        for field, value in numeric.items():
            if not math.isclose(
                float(subject[field]),
                value,
                rel_tol=1.0e-12,
                abs_tol=1.0e-12,
            ):
                replay_errors.append(f"{unit}:{field}")
    check(
        checks,
        "subject_summaries_replay_from_parcel_rows",
        not replay_errors,
        {"errors": replay_errors[:50], "error_n": len(replay_errors)},
    )

    invalid_distance_rows = [
        f"{row['unit']}:{row['parcel_id']}"
        for row in parcels
        if (
            not math.isfinite(
                float(row["minimum_distance_to_positive_fod_mm"])
            )
            or float(row["minimum_distance_to_positive_fod_mm"]) < 0
        )
    ]
    check(
        checks,
        "parcel_minimum_distances_are_finite_and_nonnegative",
        not invalid_distance_rows,
        {
            "invalid_row_n": len(invalid_distance_rows),
            "invalid_rows": invalid_distance_rows[:50],
        },
    )

    status_counts = dict(
        sorted(Counter(row["audit_status"] for row in subjects).items())
    )
    eligible = [
        row
        for row in subjects
        if row["release_input_eligible"].lower() == "true"
    ]
    replay_release = {
        "evaluated_n": len(eligible),
        "pass_n": sum(
            row["audit_status"] == "PASS_SUPPORT_SCREEN"
            for row in eligible
        ),
        "zero_review_n": sum(
            row["audit_status"]
            == "REVIEW_ZERO_DIRECT_FOD_SUPPORT"
            for row in eligible
        ),
        "low_review_n": sum(
            row["audit_status"]
            == "REVIEW_LT1PCT_DIRECT_FOD_SUPPORT"
            for row in eligible
        ),
        "atlas_hold_n": sum(
            row["audit_status"] == "HOLD_ATLAS_MASK_QC"
            for row in eligible
        ),
        "error_n": sum(
            row["audit_status"] == "AUDIT_ERROR"
            for row in eligible
        ),
    }
    check(
        checks,
        "cohort_counts_replay",
        (
            summary.get("subject_n") == len(subjects)
            and summary.get("parcel_row_n") == len(parcels)
            and summary.get("audit_status_counts") == status_counts
            and summary.get("release_eligible_support_screen")
            == replay_release
        ),
        {
            "status_counts": status_counts,
            "release_eligible": replay_release,
        },
    )

    mechanism = summary["mechanism_evidence"]
    bad = mechanism["adverse_lower_order_reference"]
    good = mechanism["good_lower_order_reference"]
    disconnected = set(
        int(value)
        for value in bad[
            "disconnected_nodes_reproducible_through_8mm"
        ]
    )
    zero = set(int(value) for value in bad["zero_direct_fod_support_parcels"])
    low = set(int(value) for value in bad["lt_1pct_direct_fod_support_parcels"])
    zero_near = set(
        int(value)
        for value in bad["zero_within_4mm_fod_support_parcels"]
    )
    check(
        checks,
        "mechanism_references_are_fail_closed_and_exact",
        (
            len(disconnected) == 29
            and bad["connected_nodes_each_seed"] == [350]
            and good["connected_nodes_each_seed"] == [379]
            and bad["zero_support_disconnected_overlap"]
            == sorted(zero & disconnected)
            and bad["low_support_disconnected_overlap"]
            == sorted(low & disconnected)
            and bad["zero_within_4mm_disconnected_overlap"]
            == sorted(zero_near & disconnected)
            and math.isclose(
                float(
                    bad[
                        "zero_within_4mm_overlap_fraction_of_disconnected"
                    ]
                ),
                len(zero_near & disconnected) / len(disconnected),
                rel_tol=1.0e-12,
                abs_tol=1.0e-12,
            )
        ),
        {
            "bad_connected_nodes": bad["connected_nodes_each_seed"],
            "good_connected_nodes": good["connected_nodes_each_seed"],
            "disconnected_n": len(disconnected),
            "zero_overlap_n": len(zero & disconnected),
            "low_overlap_n": len(low & disconnected),
            "zero_within_4mm_overlap_n": len(
                zero_near & disconnected
            ),
        },
    )

    canaries = [
        row for row in subjects if row["role"] == "CANARY_REUSE"
    ]
    canary_counts = dict(
        sorted(Counter(row["audit_status"] for row in canaries).items())
    )
    canary_not_full = [
        {
            "unit": row["unit"],
            "connected_nodes": int(row["canary_20m_connected_nodes"]),
            "density": float(row["canary_20m_density"]),
            "support_status": row["audit_status"],
            "zero_direct_parcel_n": int(
                row["zero_direct_fod_support_parcel_n"]
            ),
            "zero_within_4mm_parcel_n": int(
                row["zero_within_4mm_fod_support_parcel_n"]
            ),
        }
        for row in canaries
        if int(row["canary_20m_connected_nodes"]) != EXPECTED_NODES
    ]
    canary_direct_zero_but_full = [
        row["unit"]
        for row in canaries
        if int(row["zero_direct_fod_support_parcel_n"]) > 0
        and int(row["canary_20m_connected_nodes"]) == EXPECTED_NODES
    ]
    check(
        checks,
        "canary_calibration_replays",
        (
            len(canaries) == EXPECTED_CANARIES
            and summary["canary_calibration"]["evaluated_n"]
            == EXPECTED_CANARIES
            and summary["canary_calibration"][
                "all_20m_connected_nodes_379"
            ]
            is all(
                int(row["canary_20m_connected_nodes"])
                == EXPECTED_NODES
                for row in canaries
            )
            and summary["canary_calibration"][
                "support_status_counts"
            ]
            == canary_counts
            and summary["canary_calibration"][
                "full_20m_connected_nodes_n"
            ]
            == EXPECTED_CANARIES - len(canary_not_full)
            and summary["canary_calibration"][
                "not_full_20m_connected_nodes"
            ]
            == canary_not_full
            and summary["canary_calibration"][
                "direct_zero_but_full_379_units"
            ]
            == canary_direct_zero_but_full
        ),
        {
            "canary_n": len(canaries),
            "status_counts": canary_counts,
            "connected_nodes": sorted(
                {
                    int(row["canary_20m_connected_nodes"])
                    for row in canaries
                }
            ),
        },
    )

    record_errors: list[str] = []
    for name, record in summary.get("records", {}).items():
        try:
            actual = file_record(Path(str(record["path"])))
            if actual != record:
                record_errors.append(name)
        except Exception as error:
            record_errors.append(f"{name}:{type(error).__name__}")
    check(
        checks,
        "source_and_output_records_are_hash_bound",
        not record_errors,
        {"errors": record_errors},
    )

    compile_result = subprocess.run(
        [
            "/home/ec2-user/fsl/bin/python",
            "-m",
            "py_compile",
            str(SOURCE),
            str(Path(__file__)),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    check(
        checks,
        "implementations_compile",
        compile_result.returncode == 0,
        {
            "returncode": compile_result.returncode,
            "stderr": compile_result.stderr,
        },
    )

    check(
        checks,
        "audit_is_blind_nonexecuting_and_non_authorizing",
        (
            summary.get("diagnosis_labels_used") is False
            and summary.get("outcomes_used") is False
            and summary.get("tractography_executed") is False
            and summary.get("source_files_modified") is False
            and summary.get("non_overwriting") is True
            and summary.get("subject_exclusion_authorized") is False
            and summary.get("production_gate_authorized") is False
            and summary.get("screen_contract", {}).get(
                "zero_support_is_initial_review_not_exclusion"
            )
            is True
        ),
        {
            "diagnosis_labels_used": summary.get(
                "diagnosis_labels_used"
            ),
            "outcomes_used": summary.get("outcomes_used"),
            "tractography_executed": summary.get(
                "tractography_executed"
            ),
            "production_gate_authorized": summary.get(
                "production_gate_authorized"
            ),
        },
    )

    passed = sum(
        value["status"] == "PASS" for value in checks.values()
    )
    validation = {
        "schema_version": "1.0.0",
        "record_type": (
            "hcp379_pretract_parcel_fod_support_audit_validation"
        ),
        "status": "PASS" if passed == len(checks) else "FAIL",
        "generated_utc": utc_now(),
        "passed_checks": passed,
        "total_checks": len(checks),
        "checks": checks,
    }
    output = ROOT / "validation.json"
    temporary = output.with_name(f".{output.name}.partial")
    temporary.write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    print(json.dumps(validation, indent=2, sort_keys=True))
    if validation["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
