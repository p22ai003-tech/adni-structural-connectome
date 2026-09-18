#!/home/ec2-user/fsl/bin/python
"""Evaluate archive-route concordance only from Recovery4-bound outputs."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any, Mapping


EXP = Path("/home/ec2-user/exp")
HCP_ROOT = Path("/data/derivatives/hcp379_v2")
LEGACY_VALIDATOR_SOURCE = (
    EXP
    / "research_audit/"
    "validate_hcp379_archive_route_concordance_v2.py"
)
PRIMARY_WRAPPER = (
    EXP
    / "scripts/hcp/"
    "run_hcp379_archive_calibration_tractography_recovery4_v3.py"
)
REPLICATE_WRAPPER = (
    EXP
    / "scripts/hcp/"
    "run_hcp379_archive_calibration_replicate_recovery4_v3.py"
)
ROOT = HCP_ROOT / "archive_corrected_calibration_recovery4"
PRIMARY = (
    ROOT
    / "manifests/"
    "archive_corrected_calibration_recovery4_manifest.json"
)
REPLICATE = (
    ROOT
    / "manifests/"
    "archive_corrected_replicate_recovery4_manifest.json"
)
OUTPUT = HCP_ROOT / "calibration/archive_route_concordance.json"
EXPECTED_N = 6


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


LEGACY = load_module(
    LEGACY_VALIDATOR_SOURCE,
    "hcp379_archive_route_concordance_numeric_engine",
)


def validate_recovery4_manifest(
    path: Path,
    *,
    record_type: str,
    required_record: tuple[str, Path],
) -> dict[str, Any]:
    manifest = LEGACY.load_json(path)
    units = manifest.get("units")
    if (
        manifest.get("record_type") != record_type
        or manifest.get("status") != "PASS"
        or manifest.get("diagnosis_labels_used") is not False
        or manifest.get("unit_count") != EXPECTED_N
        or manifest.get("selected_streamline_count")
        not in (3_000_000, 5_000_000, 10_000_000)
        or not isinstance(units, list)
        or len(units) != EXPECTED_N
        or len({str(row.get("unit", "")) for row in units})
        != EXPECTED_N
        or any(
            not isinstance(row, Mapping)
            or row.get("status") != "PASS_ALL_NINE"
            or not isinstance(row.get("recovery4_input_binding"), Mapping)
            for row in units
        )
    ):
        raise ValueError(f"Recovery4 calibration manifest differs: {path}")
    records = manifest.get("records")
    label, expected = required_record
    if (
        not isinstance(records, Mapping)
        or records.get(label) != LEGACY.KEEP.file_record(expected)
    ):
        raise ValueError(f"Recovery4 source binding differs: {path}")
    return manifest


def evaluate() -> dict[str, Any]:
    primary = validate_recovery4_manifest(
        PRIMARY,
        record_type=(
            "diagnosis_blind_hcp379_archive_corrected_act_"
            "calibration_recovery4_manifest"
        ),
        required_record=("wrapper", PRIMARY_WRAPPER),
    )
    replicate = validate_recovery4_manifest(
        REPLICATE,
        record_type=(
            "diagnosis_blind_hcp379_archive_corrected_act_"
            "replicate_recovery4_manifest"
        ),
        required_record=("wrapper", REPLICATE_WRAPPER),
    )
    if (
        primary["selected_streamline_count"]
        != replicate["selected_streamline_count"]
        or {
            str(row["unit"]) for row in primary["units"]
        }
        != {str(row["unit"]) for row in replicate["units"]}
    ):
        raise ValueError("Recovery4 primary/replicate identity differs")

    LEGACY.PRIMARY = PRIMARY
    LEGACY.REPLICATE = REPLICATE
    report = LEGACY.evaluate()
    if (
        report.get("calibration_n") != EXPECTED_N
        or report.get("diagnosis_labels_used") is not False
        or report.get("archive_D0_subject_count") != 56
        or report.get("corrected_archive_low_subject_count") != 30
        or report.get("calibration_D0_n") != 4
        or report.get("calibration_low_n") != 2
        or report.get("low_archive_route_is_never_retained") is not True
        or report.get("decision")
        not in {
            (
                "RETAIN_ARCHIVE_D0_56_AND_USE_CORRECTED_LOW_30_"
                "WITH_ROUTE_SENSITIVITY"
            ),
            "RERUN_ARCHIVE_86_WITH_LOCKED_CORRECTED_ACT",
        }
        or any(
            row.get("status") not in {"PASS", "FAIL"}
            or "route_pass" not in row
            or "seed_pass" not in row
            or row.get("density_group")
            not in {
                "D0_THRESHOLD_QUALIFIED",
                "D1_NEAR_THRESHOLD",
                "D2_MODERATE_LOW",
            }
            or not isinstance(row.get("route_decision_eligible"), bool)
            for row in report.get("units", [])
        )
    ):
        raise ValueError("archive route numeric decision differs")
    return {
        **report,
        "schema_version": "3.0.0",
        "record_type": (
            "diagnosis_blind_hcp379_archive_route_"
            "concordance_recovery4_validation"
        ),
        "recovery4_inputs_required": True,
        "corrected_primary": LEGACY.KEEP.file_record(PRIMARY),
        "corrected_replicate": LEGACY.KEEP.file_record(REPLICATE),
        "recovery4_sources": {
            "primary_wrapper": LEGACY.KEEP.file_record(PRIMARY_WRAPPER),
            "replicate_wrapper": LEGACY.KEEP.file_record(
                REPLICATE_WRAPPER
            ),
            "validator": LEGACY.KEEP.file_record(Path(__file__)),
            "numeric_engine": LEGACY.KEEP.file_record(
                LEGACY_VALIDATOR_SOURCE
            ),
        },
    }


def self_test() -> None:
    LEGACY.self_test()
    if PRIMARY.parent != REPLICATE.parent or PRIMARY.parent.parent != ROOT:
        raise AssertionError("Recovery4 manifest paths drifted")
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
    LEGACY.write_report(args.output, report, args.overwrite)
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
