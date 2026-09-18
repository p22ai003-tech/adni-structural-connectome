#!/home/ec2-user/fsl/bin/python
"""Prepare or run selected-recipe tractography for legacy/tensor recovery.

The final lane has 216 corrected-ACT outputs.  Four selected primary runs are
reused from Phase B and 212 are generated in the non-overwriting
legacy/tensor Recovery4 root.  Every released count matrix must independently
meet density >=0.60; a low result fails recovery rather than excluding the
subject or changing only that subject's recipe.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


EXP = Path("/home/ec2-user/exp")
HCP_ROOT = Path("/data/derivatives/hcp379_v2")
SOURCE = (
    EXP / "scripts/hcp/hcp379_scaleup_tractography_recovery4_v3.py"
)
PRETRACT_WRAPPER_SOURCE = (
    EXP
    / "scripts/hcp/"
    "run_hcp379_legacy_density_pretract_recovery4_v3.py"
)
AUDIT_ROOT = (
    EXP
    / "research_audit/outputs/"
    "hcp379_legacy_density_recovery_input_audit_v3"
)
AUDIT_CSV = AUDIT_ROOT / "legacy_density_recovery_input_audit.csv"
AUDIT_SUMMARY = (
    AUDIT_ROOT / "legacy_density_recovery_input_audit_summary.json"
)
ROOT = HCP_ROOT / "corrected_legacy_tensor_recovery4"
RELEASE_FILENAME = "corrected_legacy_tensor_216_release_manifest.json"
RELEASE_RECORD_TYPE = (
    "diagnosis_blind_hcp379_corrected_legacy_tensor_216_release_manifest"
)
EXPECTED_LANE_N = 216
EXPECTED_EXECUTION_N = 212
EXPECTED_CANARY_OVERLAP_N = 4


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


LEGACY_PRETRACT = load_module(
    PRETRACT_WRAPPER_SOURCE,
    "hcp379_legacy_density_pretract_recovery4_wrapper",
)
TRACT = load_module(
    SOURCE, "hcp379_legacy_density_tractography_recovery4_engine"
)


def specialize(root: Path) -> None:
    LEGACY_PRETRACT.specialize(root)
    pretract = TRACT.PRETRACT
    pretract.AUDIT_CSV = AUDIT_CSV
    pretract.AUDIT_SUMMARY = AUDIT_SUMMARY
    pretract.OUTPUT_ROOT = root
    pretract.DEFAULT_HUMAN_QC = LEGACY_PRETRACT.DEFAULT_HUMAN_QC
    pretract.EXPECTED_AUDIT_N = EXPECTED_LANE_N
    pretract.EXPECTED_CANARY_OVERLAP_N = EXPECTED_CANARY_OVERLAP_N
    pretract.EXPECTED_SCALEUP_N = EXPECTED_EXECUTION_N

    TRACT.ROOT = root
    TRACT.AUDIT_CSV = AUDIT_CSV
    TRACT.AUDIT_SUMMARY = AUDIT_SUMMARY
    TRACT.PRETRACT_SUMMARY = (
        root / "manifests/pretract_recovery4_summary.json"
    )
    TRACT.PRETRACT_PLAN = (
        root / "manifests/pretract_recovery4_plan.json"
    )
    TRACT.TRACT_PLAN = (
        root
        / "manifests/"
        "tractography_recovery4_selected_recipe_plan.json"
    )
    TRACT.EXPECTED_SCALEUP_N = EXPECTED_EXECUTION_N
    TRACT.EXPECTED_CORRECTED_N = EXPECTED_LANE_N
    TRACT.EXPECTED_CANARY_OVERLAP_N = EXPECTED_CANARY_OVERLAP_N

    engine = TRACT.ENGINE
    engine.ROOT = root
    engine.AUDIT_CSV = AUDIT_CSV
    engine.AUDIT_SUMMARY = AUDIT_SUMMARY
    engine.PRETRACT_SUMMARY = TRACT.PRETRACT_SUMMARY
    engine.PRETRACT_GATE = TRACT.PRETRACT_PLAN
    engine.EXPECTED_SCALEUP_N = EXPECTED_EXECUTION_N
    engine.EXPECTED_CORRECTED_N = EXPECTED_LANE_N
    engine.EXPECTED_CANARY_OVERLAP_N = EXPECTED_CANARY_OVERLAP_N
    engine.CORRECTED_RELEASE_RECORD_TYPE = RELEASE_RECORD_TYPE
    engine.CORRECTED_RELEASE_FILENAME = RELEASE_FILENAME


def write_wrapper_plan(
    root: Path, *, compact_tractograms_after_pass: bool
) -> dict[str, Any]:
    contract = LEGACY_PRETRACT.lane_contract()
    shared_path = (
        root
        / "manifests/"
        "tractography_recovery4_selected_recipe_plan.json"
    )
    shared = TRACT.PRETRACT.load_json(shared_path)
    if (
        shared.get("status") != "PRE_PHASE_DRY_RUN_PASS"
        or shared.get("diagnosis_labels_used") is not False
        or shared.get("imaging_executed_by_plan") is not False
        or shared.get("execution_authorized") is not False
        or shared.get("target_noncanary_n") != EXPECTED_EXECUTION_N
        or shared.get("canary_overlap_n")
        != EXPECTED_CANARY_OVERLAP_N
        or shared.get("corrected_release_n") != EXPECTED_LANE_N
        or {row.get("unit") for row in shared.get("units", [])}
        != contract["execution_units"]
        or shared.get("selection_rule")
        != (
            "smallest Phase-B density-qualified stable count; no "
            "subject-level recipe selection"
        )
    ):
        raise ValueError("legacy/tensor tractography pre-Phase plan differs")
    plan = {
        "schema_version": "1.0.0",
        "record_type": (
            "diagnosis_blind_hcp379_legacy_tensor_density_"
            "tractography_recovery4_plan"
        ),
        "status": "PRE_PHASE_DRY_RUN_PASS",
        "generated_utc": TRACT.PRETRACT.utc_now(),
        "diagnosis_labels_used": False,
        "non_overwriting": True,
        "imaging_executed_by_plan": False,
        "execution_authorized": False,
        "compact_tractograms_after_pass": (
            compact_tractograms_after_pass
        ),
        "lane_n": EXPECTED_LANE_N,
        "phase_b_primary_reuse_n": EXPECTED_CANARY_OVERLAP_N,
        "new_tractography_execution_n": EXPECTED_EXECUTION_N,
        "minimum_density_inclusive": 0.60,
        "required_lane_density_pass_n": EXPECTED_LANE_N,
        "density_is_subject_exclusion_gate": False,
        "density_is_whole_cohort_technical_release_gate": True,
        "output_root": str(root.resolve()),
        "canary_overlap_units": sorted(contract["canary_overlap"]),
        "execution_units": sorted(contract["execution_units"]),
        "allowed_streamline_counts": list(TRACT.ALLOWED_COUNTS),
        "records": {
            "shared_plan": TRACT.PRETRACT.file_record(shared_path),
            "input_audit": TRACT.PRETRACT.file_record(AUDIT_CSV),
            "input_audit_summary": TRACT.PRETRACT.file_record(
                AUDIT_SUMMARY
            ),
            "shared_recovery4_runner": TRACT.PRETRACT.file_record(SOURCE),
            "wrapper": TRACT.PRETRACT.file_record(Path(__file__)),
        },
    }
    TRACT.PRETRACT.atomic_json(
        root
        / "manifests/"
        "legacy_tensor_density_tractography_recovery4_plan.json",
        plan,
    )
    return plan


def validate_release(root: Path) -> dict[str, Any]:
    path = root / "manifests" / RELEASE_FILENAME
    manifest = TRACT.PRETRACT.load_json(path)
    units = manifest.get("units")
    if (
        manifest.get("record_type") != RELEASE_RECORD_TYPE
        or manifest.get("status") != "PASS"
        or manifest.get("diagnosis_labels_used") is not False
        or manifest.get("unit_count") != EXPECTED_LANE_N
        or not isinstance(units, list)
        or len(units) != EXPECTED_LANE_N
        or len({str(row.get("unit")) for row in units})
        != EXPECTED_LANE_N
        or manifest.get(
            "density_is_whole_cohort_technical_release_gate"
        )
        is not True
        or manifest.get("minimum_edge_density_inclusive") != 0.60
        or manifest.get("subjects_at_or_above_minimum_density")
        != EXPECTED_LANE_N
        or float(manifest.get("minimum_observed_edge_density", -1.0))
        < 0.60
        or any(float(row.get("edge_density", -1.0)) < 0.60 for row in units)
    ):
        raise ValueError("legacy/tensor corrected release differs")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--pre-phase-dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--nthreads", type=int, default=16)
    parser.add_argument(
        "--compact-tractograms-after-pass", action="store_true"
    )
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    contract = LEGACY_PRETRACT.lane_contract()
    specialize(args.root)
    if args.self_test:
        # Run the mature synthetic contract before installing the adapter,
        # which correctly expects promoted real-subject pretract states.
        TRACT.ENGINE.self_test()
        TRACT.install_recovery4_adapter()
        if (
            len(contract["units"]) != EXPECTED_LANE_N
            or len(contract["execution_units"]) != EXPECTED_EXECUTION_N
            or len(contract["canary_overlap"])
            != EXPECTED_CANARY_OVERLAP_N
        ):
            raise AssertionError(contract)
        print("LEGACY_DENSITY_TRACTOGRAPHY_RECOVERY4_SELF_TEST_PASS")
        return 0

    delegated = [
        str(SOURCE),
        "--execute" if args.execute else "--pre-phase-dry-run",
        "--root",
        str(args.root),
        "--workers",
        str(args.workers),
        "--nthreads",
        str(args.nthreads),
    ]
    if args.compact_tractograms_after_pass:
        delegated.append("--compact-tractograms-after-pass")
    previous = sys.argv
    try:
        sys.argv = delegated
        returncode = int(TRACT.main())
    finally:
        sys.argv = previous
    if returncode:
        return returncode
    result = (
        validate_release(args.root)
        if args.execute
        else write_wrapper_plan(
            args.root,
            compact_tractograms_after_pass=(
                args.compact_tractograms_after_pass
            ),
        )
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
