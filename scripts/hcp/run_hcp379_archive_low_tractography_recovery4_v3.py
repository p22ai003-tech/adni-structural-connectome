#!/home/ec2-user/fsl/bin/python
"""Prepare or run corrected tractography for 30 low-density archive units.

Twenty-eight subjects are processed in this root.  Two selected primary
corrected runs are reused from the six-case archive calibration.  The final
composite release is exact 30 and fails closed unless every recomputed count
density is at least 0.60.
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
    EXP / "scripts/hcp/hcp379_scaleup_tractography_recovery4_v3.py"
)
PRETRACT_WRAPPER_SOURCE = (
    EXP
    / "scripts/hcp/run_hcp379_archive_low_pretract_recovery4_v3.py"
)
SUBSET_ROOT = (
    EXP
    / "research_audit/outputs/hcp379_archive_low_density_subset_v3"
)
AUDIT_CSV = SUBSET_ROOT / "archive_low_density_execution_28.csv"
AUDIT_SUMMARY = SUBSET_ROOT / "archive_low_density_subset_summary.json"
ROOT = HCP_ROOT / "corrected_archive_low_recovery4"
CALIBRATION_MANIFEST = (
    HCP_ROOT
    / "archive_corrected_calibration_recovery4/manifests/"
    "archive_corrected_calibration_recovery4_manifest.json"
)
EXECUTION_RELEASE_FILENAME = (
    "corrected_archive_low_execution_28_release_manifest.json"
)
EXECUTION_RELEASE_RECORD_TYPE = (
    "diagnosis_blind_hcp379_corrected_archive_low_execution_28_"
    "release_manifest"
)
COMPOSITE_RELEASE = (
    ROOT / "manifests/corrected_archive_low_30_release_manifest.json"
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


ARCHIVE_LOW_PRETRACT = load_module(
    PRETRACT_WRAPPER_SOURCE,
    "hcp379_archive_low_pretract_recovery4_wrapper",
)
TRACT = load_module(
    SOURCE, "hcp379_archive_low_tractography_recovery4_engine"
)


def specialize(root: Path) -> None:
    ARCHIVE_LOW_PRETRACT.specialize(root)
    pretract = TRACT.PRETRACT
    pretract.AUDIT_CSV = AUDIT_CSV
    pretract.AUDIT_SUMMARY = AUDIT_SUMMARY
    pretract.OUTPUT_ROOT = root
    pretract.DEFAULT_HUMAN_QC = ARCHIVE_LOW_PRETRACT.DEFAULT_HUMAN_QC
    pretract.EXPECTED_AUDIT_N = EXPECTED_EXECUTION_N
    pretract.EXPECTED_CANARY_OVERLAP_N = 0
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
    TRACT.EXPECTED_CORRECTED_N = EXPECTED_EXECUTION_N
    TRACT.EXPECTED_CANARY_OVERLAP_N = 0

    engine = TRACT.ENGINE
    engine.ROOT = root
    engine.AUDIT_CSV = AUDIT_CSV
    engine.AUDIT_SUMMARY = AUDIT_SUMMARY
    engine.PRETRACT_SUMMARY = TRACT.PRETRACT_SUMMARY
    engine.PRETRACT_GATE = TRACT.PRETRACT_PLAN
    engine.EXPECTED_SCALEUP_N = EXPECTED_EXECUTION_N
    engine.EXPECTED_CORRECTED_N = EXPECTED_EXECUTION_N
    engine.EXPECTED_CANARY_OVERLAP_N = 0
    engine.CORRECTED_RELEASE_RECORD_TYPE = (
        EXECUTION_RELEASE_RECORD_TYPE
    )
    engine.CORRECTED_RELEASE_FILENAME = EXECUTION_RELEASE_FILENAME


def write_wrapper_plan(
    root: Path, *, compact_tractograms_after_pass: bool
) -> dict[str, Any]:
    contract = ARCHIVE_LOW_PRETRACT.lane_contract()
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
        or shared.get("canary_overlap_n") != 0
        or shared.get("corrected_release_n") != EXPECTED_EXECUTION_N
        or {row.get("unit") for row in shared.get("units", [])}
        != contract["execution_units"]
    ):
        raise ValueError("archive-low tractography pre-Phase plan differs")
    plan = {
        "schema_version": "1.0.0",
        "record_type": (
            "diagnosis_blind_hcp379_archive_low_density_"
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
        "archive_calibration_primary_reuse_n": (
            EXPECTED_CALIBRATION_REUSE_N
        ),
        "new_tractography_execution_n": EXPECTED_EXECUTION_N,
        "minimum_density_inclusive": 0.60,
        "required_lane_density_pass_n": EXPECTED_LANE_N,
        "density_is_subject_exclusion_gate": False,
        "density_is_whole_cohort_technical_release_gate": True,
        "output_root": str(root.resolve()),
        "reuse_units": sorted(contract["reuse_units"]),
        "execution_units": sorted(contract["execution_units"]),
        "allowed_streamline_counts": list(TRACT.ALLOWED_COUNTS),
        "records": {
            "shared_plan": TRACT.PRETRACT.file_record(shared_path),
            "input_subset": TRACT.PRETRACT.file_record(AUDIT_CSV),
            "input_subset_summary": TRACT.PRETRACT.file_record(
                AUDIT_SUMMARY
            ),
            "archive_calibration_manifest_expected": str(
                CALIBRATION_MANIFEST
            ),
            "shared_recovery4_runner": TRACT.PRETRACT.file_record(SOURCE),
            "wrapper": TRACT.PRETRACT.file_record(Path(__file__)),
        },
    }
    TRACT.PRETRACT.atomic_json(
        root
        / "manifests/"
        "archive_low_density_tractography_recovery4_plan.json",
        plan,
    )
    return plan


def verify_execution_manifest(root: Path) -> dict[str, Any]:
    path = root / "manifests" / EXECUTION_RELEASE_FILENAME
    manifest = TRACT.PRETRACT.load_json(path)
    units = manifest.get("units")
    if (
        manifest.get("record_type") != EXECUTION_RELEASE_RECORD_TYPE
        or manifest.get("status") != "PASS"
        or manifest.get("unit_count") != EXPECTED_EXECUTION_N
        or not isinstance(units, list)
        or len(units) != EXPECTED_EXECUTION_N
        or len({str(row.get("unit")) for row in units})
        != EXPECTED_EXECUTION_N
        or float(manifest.get("minimum_observed_edge_density", -1.0))
        < 0.60
        or any(float(row.get("edge_density", -1.0)) < 0.60 for row in units)
    ):
        raise ValueError("archive-low 28-subject execution release differs")
    return manifest


def build_composite_release(root: Path) -> dict[str, Any]:
    contract = ARCHIVE_LOW_PRETRACT.lane_contract()
    execution_path = root / "manifests" / EXECUTION_RELEASE_FILENAME
    execution = verify_execution_manifest(root)
    calibration = TRACT.PRETRACT.load_json(CALIBRATION_MANIFEST)
    calibration_rows = {
        str(row.get("unit")): row
        for row in calibration.get("units", [])
        if isinstance(row, Mapping)
    }
    if (
        calibration.get("record_type")
        != (
            "diagnosis_blind_hcp379_archive_corrected_act_"
            "calibration_recovery4_manifest"
        )
        or calibration.get("status") != "PASS"
        or calibration.get("diagnosis_labels_used") is not False
        or calibration.get("unit_count") != 6
        or not contract["reuse_units"].issubset(calibration_rows)
        or calibration.get("selected_streamline_count")
        != execution.get("selected_streamline_count")
    ):
        raise ValueError("archive corrected calibration reuse differs")

    records = list(execution["units"])
    for unit in sorted(contract["reuse_units"]):
        source = calibration_rows[unit]
        if (
            source.get("status") != "PASS_ALL_NINE"
            or float(source.get("edge_density", -1.0)) < 0.60
            or not isinstance(source.get("artifacts"), Mapping)
        ):
            raise ValueError(f"{unit}:archive calibration density differs")
        records.append(
            {
                "unit": unit,
                "source_lane": "ARCHIVE_CALIBRATION_PRIMARY_REUSE",
                "selected_streamline_count": calibration[
                    "selected_streamline_count"
                ],
                "endpoint_assignment_fraction": source[
                    "endpoint_assignment_fraction"
                ],
                "edge_density": source["edge_density"],
                "artifacts": source["artifacts"],
                "subject_state": source["subject_state"],
            }
        )
    expected = (
        contract["execution_units"] | contract["reuse_units"]
    )
    if (
        len(records) != EXPECTED_LANE_N
        or {str(row["unit"]) for row in records} != expected
        or any(float(row["edge_density"]) < 0.60 for row in records)
    ):
        raise ValueError("archive-low composite identity/density differs")
    records.sort(key=lambda row: str(row["unit"]))
    manifest = {
        "schema_version": "1.0.0",
        "record_type": (
            "diagnosis_blind_hcp379_corrected_archive_low_30_"
            "release_manifest"
        ),
        "status": "PASS",
        "generated_utc": TRACT.PRETRACT.utc_now(),
        "diagnosis_labels_used": False,
        "unit_count": EXPECTED_LANE_N,
        "selected_streamline_count": execution[
            "selected_streamline_count"
        ],
        "density_is_subject_inclusion_gate": False,
        "density_is_whole_cohort_technical_release_gate": True,
        "minimum_edge_density_inclusive": 0.60,
        "subjects_at_or_above_minimum_density": EXPECTED_LANE_N,
        "minimum_observed_edge_density": min(
            float(row["edge_density"]) for row in records
        ),
        "records": {
            "execution_28": TRACT.PRETRACT.file_record(execution_path),
            "archive_calibration_6": TRACT.PRETRACT.file_record(
                CALIBRATION_MANIFEST
            ),
            "input_subset": TRACT.PRETRACT.file_record(AUDIT_CSV),
            "input_subset_summary": TRACT.PRETRACT.file_record(
                AUDIT_SUMMARY
            ),
        },
        "units": records,
    }
    TRACT.PRETRACT.atomic_json(COMPOSITE_RELEASE, manifest)
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

    contract = ARCHIVE_LOW_PRETRACT.lane_contract()
    specialize(args.root)
    if args.self_test:
        TRACT.ENGINE.self_test()
        TRACT.install_recovery4_adapter()
        if (
            len(contract["execution_units"]) != EXPECTED_EXECUTION_N
            or len(contract["reuse_units"])
            != EXPECTED_CALIBRATION_REUSE_N
        ):
            raise AssertionError(contract)
        print("ARCHIVE_LOW_TRACTOGRAPHY_RECOVERY4_SELF_TEST_PASS")
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
        build_composite_release(args.root)
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
