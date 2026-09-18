#!/usr/bin/env python3
"""Decide whether the 86 no-ACT archive cases require corrected-ACT rerun.

Six technically diverse archive subjects are compared across:

1. the recovered no-ACT archive matrices;
2. the locked corrected-ACT primary run; and
3. an independent corrected-ACT seed.

The same frozen matrix-stability thresholds used by Phase B are applied.
Endpoint-assignment differences must also remain within the frozen 0.02
absolute tolerance.  Diagnosis and research outcomes are never read.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np


EXP = Path("/home/ec2-user/exp")
HCP_ROOT = Path("/data/derivatives/hcp379_v2")
KEEP_VALIDATOR_SOURCE = (
    EXP
    / "research_audit/validate_hcp379_keep_route_concordance_v2.py"
)
CONTRACT = (
    EXP
    / "research_audit/outputs/thesis_grade_530_release_contract_v1/"
    "contract.json"
)
SELECTION = (
    EXP
    / "research_audit/outputs/"
    "hcp379_archive_corrected_input_audit_v2/"
    "archive_corrected_calibration_selection.json"
)
DENSITY_EXECUTION_SUBJECTS = (
    EXP
    / "research_audit/outputs/"
    "hcp379_density_execution_topology_v3/"
    "hcp379_density_execution_subjects.csv"
)
ARCHIVE_SUMMARY = HCP_ROOT / "manifests/archive_restore_summary.json"
PRIMARY = (
    HCP_ROOT
    / "archive_corrected_calibration/manifests/"
    "archive_corrected_calibration_manifest.json"
)
REPLICATE = (
    HCP_ROOT
    / "archive_corrected_calibration/manifests/"
    "archive_corrected_replicate_manifest.json"
)
PHASE_VALIDATION = Path(
    "/data/derivatives/scforge_v2/"
    "h04a_r1_recovery_20260719_retry4/publication/"
    "hcp379_recipe_stability_validation.json"
)
OUTPUT = HCP_ROOT / "calibration/archive_route_concordance.json"
MATRIX_NAMES = (
    "count",
    "fd_sum",
    "count_invnodevol",
    "len_mean",
    "invlen_mean",
    "fa_mean",
    "md_mean",
    "rd_mean",
    "ad_mean",
)
EXPECTED_N = 6
EXPECTED_ARCHIVE_N = 86
EXPECTED_ARCHIVE_D0_N = 56
EXPECTED_ARCHIVE_LOW_N = 30
EXPECTED_CALIBRATION_D0_N = 4
EXPECTED_CALIBRATION_LOW_N = 2
D0_GROUP = "D0_THRESHOLD_QUALIFIED"
LOW_GROUPS = {"D1_NEAR_THRESHOLD", "D2_MODERATE_LOW"}


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


KEEP = load_module(
    KEEP_VALIDATOR_SOURCE,
    "validate_hcp379_keep_route_concordance_v2",
)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def archive_matrices(
    unit: str, state: Mapping[str, Any]
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    expected_hashes = state.get("matrix_sha256")
    if not isinstance(expected_hashes, Mapping):
        raise ValueError(f"{unit}:archive matrix hashes absent")
    arrays = {}
    records = {}
    for name in MATRIX_NAMES:
        path = (
            HCP_ROOT
            / "connectomes"
            / f"SC_HCPMMP1_{unit}_{name}.csv"
        )
        record = KEEP.file_record(path)
        if record["sha256"] != expected_hashes.get(name):
            raise ValueError(f"{unit}:{name}:archive state hash differs")
        arrays[name] = KEEP.verified_matrix(record)
        records[name] = record
    return arrays, records


def manifest_runs(
    manifest: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    units = manifest.get("units")
    if (
        manifest.get("status") != "PASS"
        or manifest.get("diagnosis_labels_used") is not False
        or manifest.get("unit_count") != EXPECTED_N
        or not isinstance(units, list)
    ):
        raise ValueError("corrected archive manifest is not 6/6 PASS")
    output = {}
    for record in units:
        if not isinstance(record, Mapping):
            raise TypeError("corrected archive unit is not a mapping")
        unit = str(record.get("unit", ""))
        if not unit or unit in output:
            raise ValueError(f"corrected archive unit identity differs: {unit}")
        output[unit] = record
    return output


def run_matrices(
    record: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    artifacts = record.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("corrected archive artifacts absent")
    arrays = {}
    matrix_records = {}
    for name in MATRIX_NAMES:
        raw = artifacts.get(f"matrix_{name}")
        if not isinstance(raw, Mapping):
            raise ValueError(f"corrected archive matrix absent: {name}")
        matrix_record = dict(raw)
        arrays[name] = KEEP.verified_matrix(matrix_record)
        matrix_records[name] = matrix_record
    return arrays, matrix_records


def archive_states() -> dict[str, dict[str, Any]]:
    summary = load_json(ARCHIVE_SUMMARY)
    rows = summary.get("subjects")
    if (
        summary.get("status_counts") != {"PASS_ALL_NINE": 86}
        or not isinstance(rows, list)
        or len(rows) != 86
    ):
        raise ValueError("archive 86/86 PASS summary differs")
    output = {}
    for row in rows:
        if not isinstance(row, dict):
            raise TypeError("archive state is not a mapping")
        unit = str(row.get("fsid", ""))
        if row.get("status") != "PASS_ALL_NINE" or unit in output:
            raise ValueError(f"archive state differs: {unit}")
        output[unit] = row
    return output


def archive_density_groups() -> dict[str, dict[str, Any]]:
    """Load the frozen diagnosis-blind archive density partition."""

    with DENSITY_EXECUTION_SUBJECTS.open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row.get("inventory_lane") == "archive_86"
        ]
    if (
        len(rows) != EXPECTED_ARCHIVE_N
        or len({row.get("unit", "") for row in rows})
        != EXPECTED_ARCHIVE_N
    ):
        raise ValueError("archive density topology is not exact 86")
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        unit = str(row["unit"])
        group = str(row["density_group"])
        density = float(row["current_density"])
        if (
            group not in {D0_GROUP, *LOW_GROUPS}
            or not math.isfinite(density)
            or (group == D0_GROUP and density < 0.60)
            or (group in LOW_GROUPS and density >= 0.60)
        ):
            raise ValueError(f"{unit}: archive density group differs")
        output[unit] = {
            "density_group": group,
            "current_archive_density": density,
            "route_decision_eligible": group == D0_GROUP,
        }
    d0_n = sum(
        row["density_group"] == D0_GROUP for row in output.values()
    )
    low_n = len(output) - d0_n
    if (
        d0_n != EXPECTED_ARCHIVE_D0_N
        or low_n != EXPECTED_ARCHIVE_LOW_N
    ):
        raise ValueError(
            f"archive density partition differs: D0={d0_n}, low={low_n}"
        )
    return output


def evaluate() -> dict[str, Any]:
    selection = load_json(SELECTION)
    primary = load_json(PRIMARY)
    replicate = load_json(REPLICATE)
    phase = load_json(PHASE_VALIDATION)
    contract = load_json(CONTRACT)
    selection_units = {
        str(row.get("unit", ""))
        for row in selection.get("units", [])
        if isinstance(row, Mapping)
    }
    primary_runs = manifest_runs(primary)
    replicate_runs = manifest_runs(replicate)
    if (
        selection.get("status") != "PASS"
        or selection.get("diagnosis_or_outcome_fields_used") is not False
        or len(selection_units) != EXPECTED_N
        or set(primary_runs) != selection_units
        or set(replicate_runs) != selection_units
        or phase.get("status") != "PASS"
    ):
        raise ValueError("archive calibration identity/gate differs")
    selected = int(
        phase["smallest_density_qualified_scale_up_streamline_count"]
    )
    if (
        int(primary["selected_streamline_count"]) != selected
        or int(replicate["selected_streamline_count"]) != selected
    ):
        raise ValueError("archive calibration selected count differs")
    raw = contract["diagnosis_blind_recipe_stability_gate"]
    thresholds = {
        key: float(raw[key])
        for key in (
            "minimum_support_matched_upper_triangle_spearman",
            "minimum_tensor_matrix_upper_triangle_pearson",
            "minimum_node_strength_absolute_agreement_icc",
            "maximum_global_metric_relative_difference",
        )
    }
    assignment_tolerance = float(
        raw["maximum_assignment_fraction_absolute_difference"]
    )
    states = archive_states()
    density_groups = archive_density_groups()
    if set(states) != set(density_groups):
        raise ValueError("archive state and density topology identities differ")
    calibration_d0_n = sum(
        density_groups[unit]["route_decision_eligible"]
        for unit in selection_units
    )
    if (
        calibration_d0_n != EXPECTED_CALIBRATION_D0_N
        or EXPECTED_N - calibration_d0_n
        != EXPECTED_CALIBRATION_LOW_N
    ):
        raise ValueError(
            "archive calibration is not four D0 plus two corrected-low cases"
        )
    results = []
    for unit in sorted(selection_units):
        archive_state = states[unit]
        archive, archive_records = archive_matrices(
            unit, archive_state
        )
        corrected, corrected_records = run_matrices(
            primary_runs[unit]
        )
        independent, independent_records = run_matrices(
            replicate_runs[unit]
        )
        route = KEEP.compare(archive, corrected, thresholds)
        seed = KEEP.compare(corrected, independent, thresholds)
        archive_assignment = float(
            archive_state["assignment_fraction"]
        )
        corrected_assignment = float(
            primary_runs[unit]["endpoint_assignment_fraction"]
        )
        replicate_assignment = float(
            replicate_runs[unit]["endpoint_assignment_fraction"]
        )
        route_assignment_difference = abs(
            archive_assignment - corrected_assignment
        )
        seed_assignment_difference = abs(
            corrected_assignment - replicate_assignment
        )
        route_pass = (
            route["status"] == "PASS"
            and math.isfinite(route_assignment_difference)
            and route_assignment_difference <= assignment_tolerance
        )
        seed_pass = (
            seed["status"] == "PASS"
            and math.isfinite(seed_assignment_difference)
            and seed_assignment_difference <= assignment_tolerance
        )
        density_record = density_groups[unit]
        route_decision_eligible = bool(
            density_record["route_decision_eligible"]
        )
        decision_status = (
            "PASS"
            if seed_pass and (route_pass or not route_decision_eligible)
            else "FAIL"
        )
        results.append(
            {
                "unit": unit,
                **density_record,
                "archive_matrix_records": archive_records,
                "corrected_primary_matrix_records": corrected_records,
                "corrected_replicate_matrix_records": independent_records,
                "archive_vs_corrected_primary": route,
                "corrected_primary_vs_independent": seed,
                "archive_assignment_fraction": archive_assignment,
                "corrected_primary_assignment_fraction": (
                    corrected_assignment
                ),
                "corrected_replicate_assignment_fraction": (
                    replicate_assignment
                ),
                "route_assignment_fraction_absolute_difference": (
                    route_assignment_difference
                ),
                "seed_assignment_fraction_absolute_difference": (
                    seed_assignment_difference
                ),
                "route_pass": route_pass,
                "seed_pass": seed_pass,
                "status": decision_status,
            }
        )
    all_six_seed_pass = all(row["seed_pass"] for row in results)
    all_D0_routes_pass = all(
        row["route_pass"]
        for row in results
        if row["route_decision_eligible"]
    )
    retain_D0_pass = all_six_seed_pass and all_D0_routes_pass
    return {
        "schema_version": "2.0.0",
        "record_type": (
            "diagnosis_blind_hcp379_archive_route_"
            "concordance_validation"
        ),
        "status": "PASS" if retain_D0_pass else "FAIL",
        "decision": (
            "RETAIN_ARCHIVE_D0_56_AND_USE_CORRECTED_LOW_30_"
            "WITH_ROUTE_SENSITIVITY"
            if retain_D0_pass
            else "RERUN_ARCHIVE_86_WITH_LOCKED_CORRECTED_ACT"
        ),
        "diagnosis_labels_used": False,
        "selected_streamline_count": selected,
        "calibration_n": len(results),
        "archive_subject_count": EXPECTED_ARCHIVE_N,
        "archive_D0_subject_count": EXPECTED_ARCHIVE_D0_N,
        "corrected_archive_low_subject_count": EXPECTED_ARCHIVE_LOW_N,
        "calibration_D0_n": calibration_d0_n,
        "calibration_low_n": EXPECTED_N - calibration_d0_n,
        "all_six_corrected_seed_comparisons_pass": all_six_seed_pass,
        "all_four_D0_archive_routes_pass": all_D0_routes_pass,
        "low_archive_route_is_never_retained": True,
        "thresholds": {
            **thresholds,
            "maximum_assignment_fraction_absolute_difference": (
                assignment_tolerance
            ),
        },
        "selection": KEEP.file_record(SELECTION),
        "density_execution_subjects": KEEP.file_record(
            DENSITY_EXECUTION_SUBJECTS
        ),
        "archive_summary": KEEP.file_record(ARCHIVE_SUMMARY),
        "corrected_primary": KEEP.file_record(PRIMARY),
        "corrected_replicate": KEEP.file_record(REPLICATE),
        "phase_b_validation": KEEP.file_record(PHASE_VALIDATION),
        "contract": KEEP.file_record(CONTRACT),
        "units": results,
        "route_failure_count": sum(
            not row["route_pass"] for row in results
        ),
        "D0_route_failure_count": sum(
            not row["route_pass"]
            for row in results
            if row["route_decision_eligible"]
        ),
        "low_route_failure_count": sum(
            not row["route_pass"]
            for row in results
            if not row["route_decision_eligible"]
        ),
        "seed_failure_count": sum(
            not row["seed_pass"] for row in results
        ),
        "archive_route_allowed_in_route_homogeneous_primary_release": (
            False
        ),
        "route_policy": (
            "All 30 archive cases below density 0.60 use corrected ACT "
            "regardless of concordance. The archive-versus-corrected route "
            "test governs only the 56 already-dense D0 cases, using the four "
            "D0 calibration subjects; all six calibration subjects must pass "
            "corrected primary-versus-independent seed stability. A PASS "
            "retains a route indicator and mandatory route-exclusion "
            "sensitivity; a FAIL reruns all 86 with corrected ACT."
        ),
    }


def write_report(
    path: Path, report: Mapping[str, Any], overwrite: bool
) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"output exists; use --overwrite: {path}")
    with tempfile.NamedTemporaryFile(
        mode="w",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        encoding="utf-8",
        delete=False,
    ) as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def self_test() -> None:
    count = np.zeros((379, 379), dtype=float)
    for index in range(378):
        count[index, index + 1] = count[index + 1, index] = index + 1
    count[0, 2] = count[2, 0] = 379
    matrices = {name: count.copy() for name in MATRIX_NAMES}
    thresholds = {
        "minimum_support_matched_upper_triangle_spearman": 0.95,
        "minimum_tensor_matrix_upper_triangle_pearson": 0.98,
        "minimum_node_strength_absolute_agreement_icc": 0.90,
        "maximum_global_metric_relative_difference": 0.05,
    }
    result = KEEP.compare(matrices, matrices, thresholds)
    if result["status"] != "PASS":
        raise AssertionError(result)
    print(json.dumps({"status": "PASS"}, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--require-pass", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    report = evaluate()
    write_report(args.output, report, args.overwrite)
    print(
        json.dumps(
            {
                "status": report["status"],
                "decision": report["decision"],
                "output": str(args.output.resolve()),
            },
            sort_keys=True,
        )
    )
    return int(args.require_pass and report["status"] != "PASS")


if __name__ == "__main__":
    raise SystemExit(main())
