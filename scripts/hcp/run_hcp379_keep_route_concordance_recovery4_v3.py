#!/home/ec2-user/fsl/bin/python
"""Prepare or execute the Recovery4-bound legacy-ACT retention decision.

Four diagnosis-blind Recovery4 canary subjects overlap the 214-case
high-confidence legacy-ACT lane.  After Phase B, their legacy nine-matrix
bundles are compared with the selected corrected primary recipe and its
independent seed.  Both route and seed comparisons must pass.  The default
mode writes a non-executing plan; ``--execute`` writes the route decision.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any, Mapping


EXP = Path("/home/ec2-user/exp")
HCP_ROOT = Path("/data/derivatives/hcp379_v2")
ENGINE_SOURCE = (
    EXP
    / "research_audit/"
    "validate_hcp379_keep_route_concordance_v2.py"
)
PHASE_BUILDER = (
    EXP / "scforge/workflow/build_hcp379_phase_b_stability_manifest.py"
)
PHASE_VALIDATOR = (
    EXP / "research_audit/validate_hcp379_phase_b_stability.py"
)
RECOVERY4_MASTER = (
    HCP_ROOT / "pretract_recovery4/pretract_recovery4_manifest.json"
)
PLAN = (
    HCP_ROOT
    / "calibration/"
    "keep_route_concordance_recovery4_plan.json"
)
OUTPUT = HCP_ROOT / "calibration/keep_route_concordance.json"
EXPECTED_KEEP_N = 214
EXPECTED_OVERLAP_N = 4


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ENGINE = load_module(
    ENGINE_SOURCE, "hcp379_keep_route_concordance_numeric_engine"
)


def exact_scope() -> tuple[set[str], list[str]]:
    keep = {
        line.strip()
        for line in ENGINE.KEEP_UNITS.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    }
    master = ENGINE.load_json(RECOVERY4_MASTER)
    rows = master.get("units")
    if (
        len(keep) != EXPECTED_KEEP_N
        or master.get("record_type")
        != "diagnosis_blind_hcp379_pretract_recovery4_manifest"
        or master.get("status")
        != "AUTOMATED_PASS_AWAITING_HUMAN_VISUAL_QC"
        or master.get("diagnosis_labels_used") is not False
        or master.get("human_visual_qc_inferred") is not False
        or master.get("unit_count") != 15
        or not isinstance(rows, list)
    ):
        raise ValueError("keep/Recovery4 identity source differs")
    canary = {str(row.get("unit", "")) for row in rows}
    overlap = sorted(keep.intersection(canary))
    if (
        len(canary) != 15
        or "" in canary
        or len(overlap) != EXPECTED_OVERLAP_N
    ):
        raise ValueError("keep/Recovery4 overlap differs")
    return keep, overlap


def legacy_matrix_records(units: list[str]) -> dict[str, Any]:
    records = {}
    for unit in units:
        _, unit_records = ENGINE.legacy_matrices(unit)
        records[unit] = unit_records
    return records


def phase_status() -> dict[str, Any]:
    result = {}
    for name, path in (
        ("manifest", ENGINE.PHASE_MANIFEST),
        ("validation", ENGINE.PHASE_VALIDATION),
    ):
        if path.is_file():
            record = ENGINE.load_json(path)
            result[name] = {
                "present": True,
                "status": record.get("status"),
                "record_type": record.get("record_type"),
                "record": ENGINE.file_record(path),
            }
        else:
            result[name] = {"present": False, "status": "NOT_RUN"}
    return result


def build_plan() -> dict[str, Any]:
    _, overlap = exact_scope()
    matrices = legacy_matrix_records(overlap)
    contract = ENGINE.load_json(ENGINE.CONTRACT)
    raw = contract["diagnosis_blind_recipe_stability_gate"]
    thresholds = {
        key: raw[key]
        for key in (
            "minimum_support_matched_upper_triangle_spearman",
            "minimum_tensor_matrix_upper_triangle_pearson",
            "minimum_node_strength_absolute_agreement_icc",
            "maximum_global_metric_relative_difference",
        )
    }
    plan = {
        "schema_version": "1.0.0",
        "record_type": (
            "diagnosis_blind_hcp379_keep_route_"
            "concordance_recovery4_plan"
        ),
        "status": "PRE_PHASE_DRY_RUN_PASS",
        "diagnosis_labels_used": False,
        "imaging_executed_by_plan": False,
        "execution_authorized": False,
        "keep_subject_count": EXPECTED_KEEP_N,
        "calibration_unit_count": EXPECTED_OVERLAP_N,
        "calibration_units": overlap,
        "remaining_keep_unit_count": 210,
        "failure_rerun_count": 213,
        "matrix_names": list(ENGINE.MATRIX_NAMES),
        "thresholds": thresholds,
        "retention_requires_route_and_seed_pass": True,
        "phase_b_status": phase_status(),
        "selected_streamline_count": None,
        "legacy_matrix_records": matrices,
        "records": {
            "keep_identity_set": ENGINE.file_record(ENGINE.KEEP_UNITS),
            "recovery4_master": ENGINE.file_record(RECOVERY4_MASTER),
            "contract": ENGINE.file_record(ENGINE.CONTRACT),
            "phase_builder": ENGINE.file_record(PHASE_BUILDER),
            "phase_validator": ENGINE.file_record(PHASE_VALIDATOR),
            "numeric_engine": ENGINE.file_record(ENGINE_SOURCE),
            "wrapper": ENGINE.file_record(Path(__file__)),
        },
    }
    ENGINE.write_report(PLAN, plan, overwrite=True)
    return plan


def validate_phase_sources() -> None:
    builder_text = PHASE_BUILDER.read_text(encoding="utf-8")
    validator_text = PHASE_VALIDATOR.read_text(encoding="utf-8")
    if (
        "diagnosis_blind_hcp379_phase_b_recipe_stability_manifest"
        not in builder_text
        or "diagnosis_blind_hcp379_phase_b_recipe_stability_validation"
        not in validator_text
        or "smallest_density_qualified_scale_up_streamline_count"
        not in validator_text
    ):
        raise ValueError("Recovery4 Phase-B source contract differs")


def execute(*, overwrite: bool, require_pass: bool) -> dict[str, Any]:
    validate_phase_sources()
    _, overlap = exact_scope()
    if not ENGINE.PHASE_MANIFEST.is_file() or not ENGINE.PHASE_VALIDATION.is_file():
        raise FileNotFoundError(
            "Recovery4 Phase-B manifest/validation is not complete"
        )
    report = ENGINE.evaluate()
    observed = sorted(
        str(row.get("unit", ""))
        for row in report.get("units", [])
        if isinstance(row, Mapping)
    )
    if (
        report.get("record_type")
        != "diagnosis_blind_hcp379_keep_route_concordance_validation"
        or report.get("diagnosis_labels_used") is not False
        or report.get("calibration_unit_count") != EXPECTED_OVERLAP_N
        or observed != overlap
        or any(
            "seed_pass" not in row or "status" not in row
            for row in report.get("units", [])
        )
    ):
        raise ValueError("keep-route Recovery4 decision differs")
    ENGINE.write_report(OUTPUT, report, overwrite=overwrite)
    if require_pass and report["status"] != "PASS":
        raise RuntimeError(
            "keep-route retention failed; corrected rerun is required"
        )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--pre-phase-dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--require-pass", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    validate_phase_sources()
    if args.self_test:
        ENGINE.self_test()
        _, overlap = exact_scope()
        if len(overlap) != EXPECTED_OVERLAP_N:
            raise AssertionError(overlap)
        print("KEEP_ROUTE_RECOVERY4_SELF_TEST_PASS")
        return 0
    result = (
        execute(
            overwrite=args.overwrite,
            require_pass=args.require_pass,
        )
        if args.execute
        else build_plan()
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
