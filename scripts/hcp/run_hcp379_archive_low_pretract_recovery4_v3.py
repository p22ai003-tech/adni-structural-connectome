#!/home/ec2-user/fsl/bin/python
"""Prepare or run Recovery4 pretract for 28 new archive-low subjects.

The complete below-threshold archive lane contains 30 subjects.  Two are
already in the six-case corrected archive calibration and will reuse those
primary corrected outputs; this wrapper processes only the remaining 28.
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
    EXP / "scripts/hcp/hcp379_scaleup_pretract_recovery4_v3.py"
)
SUBSET_ROOT = (
    EXP
    / "research_audit/outputs/hcp379_archive_low_density_subset_v3"
)
AUDIT_CSV = SUBSET_ROOT / "archive_low_density_execution_28.csv"
AUDIT_SUMMARY = SUBSET_ROOT / "archive_low_density_subset_summary.json"
OUTPUT_ROOT = HCP_ROOT / "corrected_archive_low_recovery4"
DEFAULT_HUMAN_QC = (
    HCP_ROOT
    / "review_recovery4_corrected_overlay/"
    "human_visual_qc_recovery4_corrected_overlay.csv"
)
EXPECTED_LANE_N = 30
EXPECTED_CALIBRATION_REUSE_N = 2
EXPECTED_EXECUTION_N = 28


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PRETRACT = load_module(
    SOURCE, "hcp379_archive_low_pretract_recovery4_engine"
)


def lane_contract() -> dict[str, Any]:
    summary = PRETRACT.load_json(AUDIT_SUMMARY)
    rows = PRETRACT.read_csv(AUDIT_CSV)
    units = {row["unit"] for row in rows}
    reuse = set(str(unit) for unit in summary.get("reuse_units", []))
    if (
        summary.get("record_type")
        != "diagnosis_blind_hcp379_archive_low_density_input_subset"
        or summary.get("status") != "PASS"
        or summary.get("diagnosis_labels_used") is not False
        or summary.get("diagnosis_or_outcomes_used") is not False
        or summary.get("lane_n") != EXPECTED_LANE_N
        or summary.get("audited_n") != EXPECTED_EXECUTION_N
        or summary.get("ready_n") != EXPECTED_EXECUTION_N
        or summary.get("archive_calibration_primary_reuse_n")
        != EXPECTED_CALIBRATION_REUSE_N
        or len(rows) != EXPECTED_EXECUTION_N
        or len(units) != EXPECTED_EXECUTION_N
        or set(summary.get("execution_units", [])) != units
        or len(reuse) != EXPECTED_CALIBRATION_REUSE_N
        or units & reuse
        or any(
            row["route"]
            not in {
                "READY_FOR_CORRECTED_PRETRACT_FROM_EDDY",
                "READY_AFTER_GRADIENT_HEADER_REPAIR",
            }
            for row in rows
        )
    ):
        raise ValueError("archive low-density lane contract differs")
    return {"execution_units": units, "reuse_units": reuse}


def specialize(root: Path) -> None:
    PRETRACT.AUDIT_CSV = AUDIT_CSV
    PRETRACT.AUDIT_SUMMARY = AUDIT_SUMMARY
    PRETRACT.OUTPUT_ROOT = root
    PRETRACT.DEFAULT_HUMAN_QC = DEFAULT_HUMAN_QC
    PRETRACT.EXPECTED_AUDIT_N = EXPECTED_EXECUTION_N
    PRETRACT.EXPECTED_CANARY_OVERLAP_N = 0
    PRETRACT.EXPECTED_SCALEUP_N = EXPECTED_EXECUTION_N


def write_wrapper_plan(
    root: Path,
    *,
    execution_requested: bool,
    compact_reproducible_after_pass: bool,
) -> dict[str, Any]:
    contract = lane_contract()
    shared_path = root / "manifests/pretract_recovery4_plan.json"
    shared = PRETRACT.load_json(shared_path)
    expected_status = (
        "EXECUTION_GATE_PASS"
        if execution_requested
        else str(shared.get("status"))
    )
    if not execution_requested and expected_status not in {
        "PRE_HUMAN_DRY_RUN_PASS",
        "WAITING_FOR_530_HROI_POLICY_ADOPTION",
    }:
        raise ValueError("archive-low shared HROI gate differs")
    if (
        shared.get("status") != expected_status
        or shared.get("execution_requested") is not execution_requested
        or shared.get("imaging_executed_by_plan") is not False
        or shared.get("diagnosis_labels_used") is not False
        or shared.get("target_n") != EXPECTED_EXECUTION_N
        or shared.get("selected_this_invocation_n")
        != EXPECTED_EXECUTION_N
        or shared.get("canary_overlap_excluded_n") != 0
        or {row.get("unit") for row in shared.get("units", [])}
        != contract["execution_units"]
    ):
        raise ValueError("archive low specialized pretract plan differs")
    plan = {
        "schema_version": "1.0.0",
        "record_type": (
            "diagnosis_blind_hcp379_archive_low_density_"
            "pretract_recovery4_plan"
        ),
        "status": expected_status,
        "generated_utc": PRETRACT.utc_now(),
        "diagnosis_labels_used": False,
        "non_overwriting": True,
        "execution_requested": execution_requested,
        "imaging_executed_by_plan": False,
        "streaming_compaction_requested": (
            compact_reproducible_after_pass
        ),
        "lane_n": EXPECTED_LANE_N,
        "archive_calibration_primary_reuse_n": (
            EXPECTED_CALIBRATION_REUSE_N
        ),
        "new_pretract_execution_n": EXPECTED_EXECUTION_N,
        "output_root": str(root.resolve()),
        "reuse_units": sorted(contract["reuse_units"]),
        "execution_units": sorted(contract["execution_units"]),
        "records": {
            "input_subset": PRETRACT.file_record(AUDIT_CSV),
            "input_subset_summary": PRETRACT.file_record(AUDIT_SUMMARY),
            "shared_recovery4_plan": PRETRACT.file_record(shared_path),
            "shared_recovery4_runner": PRETRACT.file_record(SOURCE),
            "wrapper": PRETRACT.file_record(Path(__file__)),
        },
    }
    PRETRACT.atomic_json(
        root
        / "manifests/"
        "archive_low_density_pretract_recovery4_plan.json",
        plan,
    )
    return plan


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--pre-human-dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--human-qc-manifest", type=Path, default=DEFAULT_HUMAN_QC
    )
    parser.add_argument("--root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--nthreads", type=int, default=8)
    parser.add_argument(
        "--compact-reproducible-after-pass", action="store_true"
    )
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    contract = lane_contract()
    specialize(args.root)
    if args.self_test:
        if (
            len(contract["execution_units"]) != EXPECTED_EXECUTION_N
            or len(contract["reuse_units"])
            != EXPECTED_CALIBRATION_REUSE_N
        ):
            raise AssertionError(contract)
        print("ARCHIVE_LOW_PRETRACT_RECOVERY4_SELF_TEST_PASS")
        return 0

    delegated = [
        str(SOURCE),
        "--execute" if args.execute else "--pre-human-dry-run",
        "--human-qc-manifest",
        str(args.human_qc_manifest),
        "--root",
        str(args.root),
        "--workers",
        str(args.workers),
        "--nthreads",
        str(args.nthreads),
    ]
    if args.compact_reproducible_after_pass:
        delegated.append("--compact-reproducible-after-pass")
    previous = sys.argv
    try:
        sys.argv = delegated
        returncode = int(PRETRACT.main())
    finally:
        sys.argv = previous
    if returncode:
        return returncode
    plan = write_wrapper_plan(
        args.root,
        execution_requested=bool(args.execute),
        compact_reproducible_after_pass=(
            args.compact_reproducible_after_pass
        ),
    )
    print(json.dumps(plan, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
