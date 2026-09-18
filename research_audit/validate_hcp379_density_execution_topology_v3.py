#!/usr/bin/env python3
"""Hostile validation of the diagnosis-blind 530-subject density topology."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Mapping


EXP = Path("/home/ec2-user/exp")
ROOT = (
    EXP
    / "research_audit/outputs/"
    "hcp379_density_execution_topology_v3"
)
TOPOLOGY = ROOT / "hcp379_density_execution_topology.json"
SUBJECTS = ROOT / "hcp379_density_execution_subjects.csv"
OUTPUT = ROOT / "validation.json"
EXPECTED_N = 530
EXPECTED_DENSITY_GROUPS = {
    "D0_THRESHOLD_QUALIFIED": 58,
    "D1_NEAR_THRESHOLD": 97,
    "D2_MODERATE_LOW": 61,
    "D3_SEVERE_LOW": 68,
    "D4_CRITICAL_LOW": 18,
    "U0_UNGENERATED": 228,
}
EXPECTED_LANES = {
    "corrected_core_227": 227,
    "corrected_legacy_tensor_216": 216,
    "corrected_archive_low_30": 30,
    "conditional_archive_D0_56": 56,
    "corrected_freesurfer_1": 1,
}


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise ValueError(f"not a regular file: {resolved}")
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


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
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def main() -> int:
    topology = load_json(TOPOLOGY)
    rows = read_csv(SUBJECTS)
    checks: dict[str, dict[str, Any]] = {}

    def check(name: str, condition: bool, evidence: Any) -> None:
        checks[name] = {
            "status": "PASS" if condition else "FAIL",
            "evidence": evidence,
        }

    units = [row.get("unit", "") for row in rows]
    check(
        "exact_530_unique_subjects",
        len(rows) == EXPECTED_N
        and len(set(units)) == EXPECTED_N
        and all(units),
        {"row_count": len(rows), "unique_units": len(set(units))},
    )
    density_counts = Counter(row.get("density_group", "") for row in rows)
    check(
        "density_groups_are_exact",
        dict(density_counts) == EXPECTED_DENSITY_GROUPS
        and topology.get("density_group_counts")
        == EXPECTED_DENSITY_GROUPS,
        dict(sorted(density_counts.items())),
    )
    lane_counts = Counter(row.get("execution_lane", "") for row in rows)
    check(
        "execution_lanes_are_disjoint_exact_union",
        dict(lane_counts) == EXPECTED_LANES,
        dict(sorted(lane_counts.items())),
    )
    goal = topology.get("goal_acceptance", {})
    check(
        "hard_density_contract_is_fail_closed",
        goal.get("target_n") == EXPECTED_N
        and goal.get("required_pass_n") == EXPECTED_N
        and goal.get("minimum_density_inclusive") == 0.60
        and goal.get("subject_exclusion_allowed") is False
        and goal.get("edge_imputation_allowed") is False
        and goal.get("per_subject_streamline_count_tuning_allowed")
        is False,
        goal,
    )
    check(
        "routing_is_diagnosis_and_outcome_blind",
        topology.get("diagnosis_or_outcomes_used") is False
        and all(
            row.get("diagnosis_or_outcomes_used") == "False"
            for row in rows
        ),
        {"row_count": len(rows)},
    )
    below = [
        row
        for row in rows
        if row.get("density_group")
        in {
            "D1_NEAR_THRESHOLD",
            "D2_MODERATE_LOW",
            "D3_SEVERE_LOW",
            "D4_CRITICAL_LOW",
        }
    ]
    check(
        "all_244_existing_low_cases_are_corrected",
        len(below) == 244
        and all(
            row.get("planned_final_route") == "CORRECTED_ACT"
            and row.get("tractography_action")
            in {
                "RUN_PHASE_B_SELECTED_UNIFORM_RECIPE",
                "REUSE_PHASE_B_SELECTED_PRIMARY",
                "REUSE_ARCHIVE_CALIBRATION_PRIMARY",
            }
            for row in below
        ),
        {
            "below_threshold_n": len(below),
            "route_counts": dict(
                Counter(row.get("planned_final_route", "") for row in below)
            ),
        },
    )
    archive_low = [
        row
        for row in rows
        if row.get("execution_lane") == "corrected_archive_low_30"
    ]
    check(
        "archive_low_30_are_unconditionally_corrected",
        len(archive_low) == 30
        and Counter(row["density_group"] for row in archive_low)
        == {"D1_NEAR_THRESHOLD": 28, "D2_MODERATE_LOW": 2}
        and all(
            row.get("planned_final_route") == "CORRECTED_ACT"
            for row in archive_low
        ),
        {
            "n": len(archive_low),
            "density_groups": dict(
                Counter(row["density_group"] for row in archive_low)
            ),
        },
    )
    archive_D0 = [
        row
        for row in rows
        if row.get("execution_lane") == "conditional_archive_D0_56"
    ]
    check(
        "only_archive_D0_56_are_concordance_conditional",
        len(archive_D0) == 56
        and all(
            row.get("density_group") == "D0_THRESHOLD_QUALIFIED"
            and row.get("planned_final_route")
            == "ARCHIVE_NO_ACT_OR_CORRECTED_ACT"
            and "if FAIL run corrected Recovery4 for all 56"
            in row.get("route_decision_dependency", "")
            for row in archive_D0
        ),
        {"n": len(archive_D0)},
    )
    legacy = [
        row
        for row in rows
        if row.get("execution_lane") == "corrected_legacy_tensor_216"
    ]
    check(
        "legacy_tensor_216_use_one_corrected_route",
        len(legacy) == 216
        and sum(
            row.get("density_group") != "D0_THRESHOLD_QUALIFIED"
            for row in legacy
        )
        == 214
        and all(
            row.get("planned_final_route") == "CORRECTED_ACT"
            for row in legacy
        ),
        {
            "n": len(legacy),
            "below_threshold_n": sum(
                row.get("density_group") != "D0_THRESHOLD_QUALIFIED"
                for row in legacy
            ),
        },
    )
    reuse_counts = Counter(row.get("reuse_source", "") for row in rows)
    check(
        "reuse_and_new_execution_counts_are_exact",
        reuse_counts
        == {
            "": 457,
            "HISTORICAL_ARCHIVE_NO_ACT": 56,
            "RECOVERY4_PHASE_B_PRIMARY": 15,
            "ARCHIVE_CORRECTED_CALIBRATION_PRIMARY": 2,
        },
        dict(reuse_counts),
    )
    check(
        "both_release_scenarios_cover_530",
        topology.get("release_scenarios")
        == {
            "archive_concordance_fail": {
                "corrected_ACT_n": 530,
                "retained_archive_no_ACT_n": 0,
                "total_n": 530,
            },
            "archive_concordance_pass": {
                "corrected_ACT_n": 474,
                "retained_archive_no_ACT_n": 56,
                "total_n": 530,
            },
        },
        topology.get("release_scenarios"),
    )
    records = topology.get("records", {})
    binding_ok = isinstance(records, Mapping) and bool(records)
    observed_records: dict[str, Any] = {}
    if binding_ok:
        for name, expected in records.items():
            try:
                observed = file_record(Path(str(expected.get("path", ""))))
            except (OSError, ValueError, AttributeError):
                binding_ok = False
                observed = None
            observed_records[str(name)] = observed
            if observed != expected:
                binding_ok = False
    check(
        "all_topology_source_bindings_match",
        binding_ok,
        {
            "record_count": len(records)
            if isinstance(records, Mapping)
            else 0,
            "all_match": binding_ok,
        },
    )
    recovery_record = records.get("density_recovery_plan", {})
    recovery_rows = (
        read_csv(Path(str(recovery_record.get("path", ""))))
        if isinstance(recovery_record, Mapping)
        and Path(str(recovery_record.get("path", ""))).is_file()
        else []
    )
    check(
        "topology_identity_matches_frozen_recovery_manifest",
        len(recovery_rows) == EXPECTED_N
        and {row.get("unit", "") for row in recovery_rows} == set(units),
        {
            "recovery_rows": len(recovery_rows),
            "identity_match": (
                {row.get("unit", "") for row in recovery_rows}
                == set(units)
            ),
        },
    )

    failed = sorted(
        name
        for name, result in checks.items()
        if result["status"] != "PASS"
    )
    report = {
        "schema_version": "1.0.0",
        "record_type": (
            "hcp379_density_execution_topology_hostile_validation"
        ),
        "status": "PASS" if not failed else "FAIL",
        "diagnosis_or_outcome_fields_used": False,
        "imaging_executed_by_validation": False,
        "passed_checks": len(checks) - len(failed),
        "total_checks": len(checks),
        "failed_checks": failed,
        "checks": checks,
        "records": {
            "topology": file_record(TOPOLOGY),
            "subjects": file_record(SUBJECTS),
            "validator": file_record(Path(__file__)),
        },
    }
    atomic_json(OUTPUT, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return int(bool(failed))


if __name__ == "__main__":
    raise SystemExit(main())
