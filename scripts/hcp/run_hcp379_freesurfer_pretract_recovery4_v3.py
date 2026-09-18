#!/home/ec2-user/fsl/bin/python
"""Prepare or run corrected Recovery4 pretract for the recovered FS unit."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


EXP = Path("/home/ec2-user/exp")
HCP_ROOT = Path("/data/derivatives/hcp379_v2")
SOURCE = EXP / "scripts/hcp/hcp379_scaleup_pretract_recovery4_v3.py"
AUDIT_ROOT = (
    EXP
    / "research_audit/outputs/"
    "hcp379_freesurfer_corrected_input_v3"
)
AUDIT_CSV = AUDIT_ROOT / "freesurfer_corrected_input.csv"
AUDIT_SUMMARY = AUDIT_ROOT / "freesurfer_corrected_input_summary.json"
OUTPUT_ROOT = HCP_ROOT / "corrected_freesurfer_recovery4"
DEFAULT_HUMAN_QC = (
    HCP_ROOT
    / "review_recovery4_corrected_overlay/"
    "human_visual_qc_recovery4_corrected_overlay.csv"
)
EXPECTED_N = 1
UNIT = "114_S_6347_I1344943"


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PRETRACT = load_module(
    SOURCE, "hcp379_freesurfer_pretract_recovery4_engine"
)


def lane_contract() -> set[str]:
    summary = PRETRACT.load_json(AUDIT_SUMMARY)
    rows = PRETRACT.read_csv(AUDIT_CSV)
    units = {row["unit"] for row in rows}
    if (
        summary.get("record_type")
        != "diagnosis_blind_hcp379_freesurfer_corrected_input_audit"
        or summary.get("status") != "PASS"
        or summary.get("diagnosis_labels_used") is not False
        or summary.get("diagnosis_or_outcomes_used") is not False
        or summary.get("audited_n") != EXPECTED_N
        or summary.get("ready_n") != EXPECTED_N
        or units != {UNIT}
        or len(rows) != EXPECTED_N
        or rows[0].get("route")
        != "READY_FOR_CORRECTED_PRETRACT_FROM_EDDY"
        or rows[0].get("old_existing_track_reuse_allowed", "").lower()
        != "false"
    ):
        raise ValueError("FreeSurfer corrected-input lane differs")
    return units


def specialize(root: Path) -> None:
    PRETRACT.AUDIT_CSV = AUDIT_CSV
    PRETRACT.AUDIT_SUMMARY = AUDIT_SUMMARY
    PRETRACT.OUTPUT_ROOT = root
    PRETRACT.DEFAULT_HUMAN_QC = DEFAULT_HUMAN_QC
    PRETRACT.EXPECTED_AUDIT_N = EXPECTED_N
    PRETRACT.EXPECTED_CANARY_OVERLAP_N = 0
    PRETRACT.EXPECTED_SCALEUP_N = EXPECTED_N


def write_wrapper_plan(
    root: Path,
    *,
    execution_requested: bool,
    compact_reproducible_after_pass: bool,
) -> dict[str, Any]:
    units = lane_contract()
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
        raise ValueError("FreeSurfer shared HROI gate differs")
    if (
        shared.get("status") != expected_status
        or shared.get("execution_requested") is not execution_requested
        or shared.get("imaging_executed_by_plan") is not False
        or shared.get("diagnosis_labels_used") is not False
        or shared.get("target_n") != EXPECTED_N
        or shared.get("selected_this_invocation_n") != EXPECTED_N
        or shared.get("canary_overlap_excluded_n") != 0
        or {row.get("unit") for row in shared.get("units", [])} != units
    ):
        raise ValueError("FreeSurfer specialized pretract plan differs")
    plan = {
        "schema_version": "1.0.0",
        "record_type": (
            "diagnosis_blind_hcp379_freesurfer_pretract_"
            "recovery4_plan"
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
        "lane_n": EXPECTED_N,
        "new_pretract_execution_n": EXPECTED_N,
        "old_existing_track_reuse_allowed": False,
        "output_root": str(root.resolve()),
        "execution_units": sorted(units),
        "records": {
            "input_audit": PRETRACT.file_record(AUDIT_CSV),
            "input_audit_summary": PRETRACT.file_record(AUDIT_SUMMARY),
            "shared_plan": PRETRACT.file_record(shared_path),
            "shared_runner": PRETRACT.file_record(SOURCE),
            "wrapper": PRETRACT.file_record(Path(__file__)),
        },
    }
    PRETRACT.atomic_json(
        root
        / "manifests/"
        "freesurfer_pretract_recovery4_plan.json",
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
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--nthreads", type=int, default=8)
    parser.add_argument(
        "--compact-reproducible-after-pass", action="store_true"
    )
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    units = lane_contract()
    specialize(args.root)
    if args.self_test:
        if units != {UNIT}:
            raise AssertionError(units)
        print("FREESURFER_PRETRACT_RECOVERY4_SELF_TEST_PASS")
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
