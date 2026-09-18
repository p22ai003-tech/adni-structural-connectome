#!/home/ec2-user/fsl/bin/python
"""Prepare or run Recovery4 pretract for six archive-route calibrators.

The six diagnosis-blind archive calibration subjects are processed in a
separate, non-overwriting root.  This wrapper specializes the validated
Recovery4 scale-up implementation; it never falls back to the retired
Recovery3/v2 promotion gates.  The default is a non-imaging dry run.
Imaging additionally requires ``--execute`` and the genuine complete
Recovery4 human-review CSV.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Mapping


EXP = Path("/home/ec2-user/exp")
HCP_ROOT = Path("/data/derivatives/hcp379_v2")
SOURCE = (
    EXP / "scripts/hcp/hcp379_scaleup_pretract_recovery4_v3.py"
)
AUDIT_ROOT = (
    EXP
    / "research_audit/outputs/"
    "hcp379_archive_corrected_input_audit_v2"
)
AUDIT_CSV = AUDIT_ROOT / "archive_calibration_input_audit.csv"
AUDIT_SUMMARY = (
    AUDIT_ROOT / "archive_calibration_input_audit_summary.json"
)
SELECTION = AUDIT_ROOT / "archive_corrected_calibration_selection.json"
OUTPUT_ROOT = HCP_ROOT / "archive_corrected_calibration_recovery4"
DEFAULT_HUMAN_QC = (
    HCP_ROOT
    / "review_recovery4_corrected_overlay/"
    "human_visual_qc_recovery4_corrected_overlay.csv"
)
WRAPPER_PLAN = (
    OUTPUT_ROOT / "manifests/archive_pretract_recovery4_plan.json"
)
EXPECTED_N = 6


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PRETRACT = load_module(
    SOURCE, "hcp379_archive_pretract_recovery4_engine"
)


def selection_units() -> set[str]:
    selection = PRETRACT.load_json(SELECTION)
    rows = PRETRACT.read_csv(AUDIT_CSV)
    units = {row.get("unit", "") for row in rows}
    selected = {
        str(row.get("unit", ""))
        for row in selection.get("units", [])
        if isinstance(row, Mapping)
    }
    source_records = selection.get("source_records")
    if (
        selection.get("status") != "PASS"
        or selection.get("diagnosis_or_outcome_fields_used") is not False
        or selection.get("target_n") != EXPECTED_N
        or len(rows) != EXPECTED_N
        or len(units) != EXPECTED_N
        or "" in units
        or selected != units
        or not isinstance(source_records, Mapping)
        or source_records.get("calibration_input_audit")
        != PRETRACT.file_record(AUDIT_CSV)
        or source_records.get("calibration_input_audit_summary")
        != PRETRACT.file_record(AUDIT_SUMMARY)
    ):
        raise ValueError("archive calibration selection binding differs")
    summary = PRETRACT.load_json(AUDIT_SUMMARY)
    if (
        summary.get("record_type")
        != "diagnosis_blind_hcp379_archive_calibration_input_audit"
        or summary.get("status") != "PASS"
        or summary.get("diagnosis_labels_used") is not False
        or summary.get("target_n") != EXPECTED_N
        or summary.get("audited_n") != EXPECTED_N
    ):
        raise ValueError("archive calibration input summary differs")
    return units


def specialize(root: Path) -> None:
    """Bind the shared Recovery4 implementation to the six-case lane."""

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
        raise ValueError("archive calibration shared HROI gate differs")
    if (
        shared.get("status") != expected_status
        or shared.get("execution_requested") is not execution_requested
        or shared.get("imaging_executed_by_plan") is not False
        or shared.get("streaming_compaction_requested")
        is not compact_reproducible_after_pass
        or shared.get("diagnosis_labels_used") is not False
        or shared.get("target_n") != EXPECTED_N
        or shared.get("selected_this_invocation_n") != EXPECTED_N
        or shared.get("canary_overlap_excluded_n") != 0
        or {row.get("unit") for row in shared.get("units", [])}
        != selection_units()
    ):
        raise ValueError("specialized Recovery4 pretract plan differs")
    plan = {
        "schema_version": "1.0.0",
        "record_type": (
            "diagnosis_blind_hcp379_archive_pretract_recovery4_plan"
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
        "target_n": EXPECTED_N,
        "output_root": str(root.resolve()),
        "units": sorted(selection_units()),
        "records": {
            "selection": PRETRACT.file_record(SELECTION),
            "input_audit": PRETRACT.file_record(AUDIT_CSV),
            "input_audit_summary": PRETRACT.file_record(AUDIT_SUMMARY),
            "shared_recovery4_plan": PRETRACT.file_record(shared_path),
            "shared_recovery4_runner": PRETRACT.file_record(SOURCE),
            "wrapper": PRETRACT.file_record(Path(__file__)),
        },
    }
    PRETRACT.atomic_json(
        root / "manifests/archive_pretract_recovery4_plan.json", plan
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
        "--compact-reproducible-after-pass",
        action="store_true",
    )
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    units = selection_units()
    specialize(args.root)
    if args.self_test:
        if len(units) != EXPECTED_N:
            raise AssertionError(units)
        print("ARCHIVE_PRETRACT_RECOVERY4_SELF_TEST_PASS")
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
