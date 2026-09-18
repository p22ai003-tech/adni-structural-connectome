#!/usr/bin/env python3
"""Run the locked corrected-ACT recipe for six archive calibration cases.

The shared HCP379 scale-up implementation supplies tractography, SIFT2,
assignments, all nine matrices, and strict subject QC.  This wrapper binds it
to the six diagnosis-blind archive calibration inputs and writes a distinct
calibration manifest instead of a 227-subject release manifest.
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
SOURCE = EXP / "scripts/hcp/hcp379_scaleup_tractography_v2.py"
PRETRACT_WRAPPER_SOURCE = (
    EXP
    / "scripts/hcp/run_hcp379_archive_calibration_pretract_v2.py"
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
ROOT = HCP_ROOT / "archive_corrected_calibration"
PRETRACT_SUMMARY = ROOT / "manifests/pretract_summary.json"
PRETRACT_GATE = ROOT / "manifests/pretract_gate.json"
TRACT_SUMMARY = ROOT / "manifests/tractography_summary.json"
OUTPUT_MANIFEST = (
    ROOT / "manifests/archive_corrected_calibration_manifest.json"
)
EXPECTED_N = 6


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TRACT = load_module(SOURCE, "hcp379_scaleup_tractography_v2")
CAL_PRETRACT = load_module(
    PRETRACT_WRAPPER_SOURCE,
    "run_hcp379_archive_calibration_pretract_v2",
)


def specialize(root: Path) -> None:
    TRACT.ROOT = root
    TRACT.AUDIT_CSV = AUDIT_CSV
    TRACT.AUDIT_SUMMARY = AUDIT_SUMMARY
    TRACT.PRETRACT_SUMMARY = root / "manifests/pretract_summary.json"
    TRACT.PRETRACT_GATE = root / "manifests/pretract_gate.json"
    TRACT.EXPECTED_SCALEUP_N = EXPECTED_N
    TRACT.EXPECTED_CORRECTED_N = EXPECTED_N
    TRACT.EXPECTED_CANARY_OVERLAP_N = 0


def build_manifest(root: Path) -> dict[str, Any]:
    units = CAL_PRETRACT.validate_binding()
    summary_path = root / "manifests/tractography_summary.json"
    summary = TRACT.PRETRACT.load_json(summary_path)
    if (
        summary.get("status") != "PASS"
        or summary.get("target_n") != EXPECTED_N
        or summary.get("status_counts", {}).get("PASS_ALL_NINE")
        != EXPECTED_N
    ):
        raise ValueError("archive calibration tractography is not 6/6 PASS")
    records = []
    selected_counts = set()
    for unit in sorted(units):
        state_path = root / "qc/tractography" / f"{unit}.json"
        state = TRACT.PRETRACT.load_json(state_path)
        if state.get("status") != "PASS_ALL_NINE":
            raise ValueError(f"{unit}:archive calibration state is not PASS")
        artifacts = state.get("artifacts")
        if not isinstance(artifacts, dict) or len(artifacts) < 14:
            raise ValueError(f"{unit}:archive calibration artifacts differ")
        for record in artifacts.values():
            path = Path(str(record.get("path", "")))
            if record != TRACT.PRETRACT.file_record(path):
                raise ValueError(f"{unit}:artifact binding differs: {path}")
        selected_counts.add(int(state["selected_streamline_count"]))
        records.append(
            {
                "unit": unit,
                "status": state["status"],
                "selected_streamline_count": state[
                    "selected_streamline_count"
                ],
                "edge_density": state["edge_density"],
                "endpoint_assignment_fraction": state[
                    "endpoint_assignment_fraction"
                ],
                "matrix_qc": state["matrix_qc"],
                "artifacts": artifacts,
                "subject_state": TRACT.PRETRACT.file_record(state_path),
            }
        )
    if len(selected_counts) != 1:
        raise ValueError("archive calibration streamline count differs")
    manifest = {
        "schema_version": "1.0.0",
        "record_type": (
            "diagnosis_blind_hcp379_archive_corrected_act_"
            "calibration_manifest"
        ),
        "status": "PASS",
        "generated_utc": TRACT.PRETRACT.utc_now(),
        "diagnosis_labels_used": False,
        "unit_count": len(records),
        "selected_streamline_count": next(iter(selected_counts)),
        "density_is_subject_inclusion_gate": False,
        "records": {
            "input_audit": TRACT.PRETRACT.file_record(AUDIT_CSV),
            "input_audit_summary": TRACT.PRETRACT.file_record(
                AUDIT_SUMMARY
            ),
            "pretract_summary": TRACT.PRETRACT.file_record(
                root / "manifests/pretract_summary.json"
            ),
            "pretract_gate": TRACT.PRETRACT.file_record(
                root / "manifests/pretract_gate.json"
            ),
            "tractography_summary": TRACT.PRETRACT.file_record(
                summary_path
            ),
            "phase_b_validation": TRACT.PRETRACT.file_record(
                TRACT.PHASE_VALIDATION
            ),
            "phase_b_manifest": TRACT.PRETRACT.file_record(
                TRACT.PHASE_MANIFEST
            ),
            "shared_tractography_source": TRACT.PRETRACT.file_record(
                SOURCE
            ),
        },
        "units": records,
    }
    TRACT.PRETRACT.atomic_json(
        root / "manifests/archive_corrected_calibration_manifest.json",
        manifest,
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--nthreads", type=int, default=16)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    units = CAL_PRETRACT.validate_binding()
    specialize(args.root)
    if args.self_test:
        TRACT.self_test()
        if len(units) != EXPECTED_N:
            raise AssertionError(units)
        print("ARCHIVE_CALIBRATION_SELF_TEST_PASS")
        return 0

    delegated = [
        str(SOURCE),
        "--root",
        str(args.root),
        "--workers",
        str(args.workers),
        "--nthreads",
        str(args.nthreads),
        "--limit",
        str(EXPECTED_N),
    ]
    previous = sys.argv
    try:
        sys.argv = delegated
        returncode = int(TRACT.main())
    finally:
        sys.argv = previous
    if returncode:
        return returncode
    manifest = build_manifest(args.root)
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    return 0 if manifest["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
