#!/usr/bin/env python3
"""Run corrected pre-tractography for six archive calibration subjects.

This is a thin, fail-closed specialization of
``hcp379_scaleup_pretract_v2.py``.  It binds the six-row diagnosis-blind
archive calibration audit to a separate non-overwriting output root.  The
underlying scale-up code retains its Recovery3, corrected-atlas, review-pack,
real-human-QC, source-identity, and pooled-response gates.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


EXP = Path("/home/ec2-user/exp")
HCP_ROOT = Path("/data/derivatives/hcp379_v2")
SOURCE = EXP / "scripts/hcp/hcp379_scaleup_pretract_v2.py"
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
OUTPUT_ROOT = HCP_ROOT / "archive_corrected_calibration"
DEFAULT_HUMAN_QC = HCP_ROOT / "review/human_visual_qc.csv"
EXPECTED_N = 6


def load_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "hcp379_scaleup_pretract_v2", SOURCE
    )
    if spec is None or spec.loader is None:
        raise ImportError(SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PRETRACT = load_module()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_record(record: Any, expected: Path) -> None:
    if not isinstance(record, dict):
        raise TypeError(f"file record is not a mapping: {expected}")
    resolved = expected.resolve()
    observed = {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }
    if record != observed:
        raise ValueError(f"file binding differs: {expected}")


def units_from_csv(path: Path) -> set[str]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    units = {str(row.get("unit", "")) for row in rows}
    if (
        len(rows) != EXPECTED_N
        or len(units) != EXPECTED_N
        or "" in units
    ):
        raise ValueError("archive calibration audit identity differs")
    return units


def validate_binding() -> set[str]:
    for path in (AUDIT_CSV, AUDIT_SUMMARY, SELECTION, SOURCE):
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(path)
    summary = PRETRACT.load_json(AUDIT_SUMMARY)
    selection = PRETRACT.load_json(SELECTION)
    units = units_from_csv(AUDIT_CSV)
    selection_units = {
        str(row.get("unit", ""))
        for row in selection.get("units", [])
        if isinstance(row, dict)
    }
    records = selection.get("source_records", {})
    if (
        summary.get("status") != "PASS"
        or summary.get("diagnosis_labels_used") is not False
        or summary.get("target_n") != EXPECTED_N
        or summary.get("audited_n") != EXPECTED_N
        or selection.get("status") != "PASS"
        or selection.get("diagnosis_or_outcome_fields_used") is not False
        or selection.get("target_n") != EXPECTED_N
        or units != selection_units
    ):
        raise ValueError("archive calibration binding differs")
    verify_record(records.get("calibration_input_audit"), AUDIT_CSV)
    verify_record(
        records.get("calibration_input_audit_summary"), AUDIT_SUMMARY
    )
    return units


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--human-qc-manifest", type=Path, default=DEFAULT_HUMAN_QC
    )
    parser.add_argument("--root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--nthreads", type=int, default=8)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    units = validate_binding()
    if args.self_test:
        if len(units) != EXPECTED_N:
            raise AssertionError(units)
        print("SELF_TEST_PASS")
        return 0

    # Specialize only input identity, output root, and expected counts.  All
    # scientific commands and global canary/human-QC gates remain in the
    # shared implementation.
    PRETRACT.AUDIT_CSV = AUDIT_CSV
    PRETRACT.AUDIT_SUMMARY = AUDIT_SUMMARY
    PRETRACT.OUTPUT_ROOT = args.root
    PRETRACT.EXPECTED_TARGET_N = EXPECTED_N
    PRETRACT.EXPECTED_SCALEUP_N = EXPECTED_N
    delegated = [
        str(SOURCE),
        "--human-qc-manifest",
        str(args.human_qc_manifest),
        "--root",
        str(args.root),
        "--workers",
        str(args.workers),
        "--nthreads",
        str(args.nthreads),
    ]
    if args.limit is not None:
        delegated.extend(("--limit", str(args.limit)))
    previous = sys.argv
    try:
        sys.argv = delegated
        return int(PRETRACT.main())
    finally:
        sys.argv = previous


if __name__ == "__main__":
    raise SystemExit(main())
