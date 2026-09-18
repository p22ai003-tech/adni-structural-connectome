#!/home/ec2-user/fsl/bin/python
"""Assemble an exact corrected archive-86 candidate from Recovery4 lanes.

The merger performs no imaging and never selects the primary archive route.
It combines the corrected archive-low 30 and corrected archive-D0 56 only
when both use the same Phase-B streamline count and independently satisfy
the nine-matrix, identity, provenance, and density contracts.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any, Mapping


EXP = Path("/home/ec2-user/exp")
HCP_ROOT = Path("/data/derivatives/hcp379_v2")
FINAL_SOURCE = EXP / "scripts/hcp/build_hcp379_final_release_v2.py"
LOW_30 = (
    HCP_ROOT
    / "corrected_archive_low_recovery4/manifests/"
    "corrected_archive_low_30_release_manifest.json"
)
D0_56 = (
    HCP_ROOT
    / "corrected_archive_D0_recovery4/manifests/"
    "corrected_archive_D0_56_release_manifest.json"
)
ROOT = HCP_ROOT / "corrected_archive_recovery4"
PLAN = ROOT / "manifests/corrected_archive_86_merge_plan.json"
OUTPUT = ROOT / "manifests/corrected_archive_86_release_manifest.json"
LOW_RECORD_TYPE = (
    "diagnosis_blind_hcp379_corrected_archive_low_30_release_manifest"
)
D0_RECORD_TYPE = (
    "diagnosis_blind_hcp379_corrected_archive_D0_56_release_manifest"
)
OUTPUT_RECORD_TYPE = (
    "diagnosis_blind_hcp379_corrected_archive_86_release_manifest"
)
RERUN_DECISION = "RERUN_ARCHIVE_86_WITH_LOCKED_CORRECTED_ACT"
RETAIN_DECISION = (
    "RETAIN_ARCHIVE_D0_56_AND_USE_CORRECTED_LOW_30_WITH_ROUTE_SENSITIVITY"
)


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FINAL = load_module(
    FINAL_SOURCE, "hcp379_corrected_archive_86_final_contract"
)


def source_state(
    path: Path, *, expected_n: int, expected_record_type: str
) -> dict[str, Any]:
    manifest = FINAL.load_json_optional(path)
    if manifest is None:
        return {
            "path": str(path),
            "present": False,
            "status": "NOT_RUN",
            "expected_n": expected_n,
        }
    rows = manifest.get("units")
    valid_header = bool(
        manifest.get("record_type") == expected_record_type
        and manifest.get("status") == "PASS"
        and manifest.get("diagnosis_labels_used") is False
        and manifest.get("unit_count") == expected_n
        and manifest.get(
            "density_is_whole_cohort_technical_release_gate"
        )
        is True
        and manifest.get("density_is_subject_inclusion_gate") is False
        and manifest.get("minimum_edge_density_inclusive")
        == FINAL.MINIMUM_EDGE_DENSITY
        and manifest.get("subjects_at_or_above_minimum_density")
        == expected_n
        and float(manifest.get("minimum_observed_edge_density", -1.0))
        >= FINAL.MINIMUM_EDGE_DENSITY
        and manifest.get("selected_streamline_count")
        in FINAL.ALLOWED_STREAMLINE_COUNTS
        and isinstance(rows, list)
        and len(rows) == expected_n
    )
    return {
        "path": str(path),
        "present": True,
        "status": manifest.get("status"),
        "record_type": manifest.get("record_type"),
        "unit_count": manifest.get("unit_count"),
        "selected_streamline_count": manifest.get(
            "selected_streamline_count"
        ),
        "minimum_observed_edge_density": manifest.get(
            "minimum_observed_edge_density"
        ),
        "header_contract_pass": valid_header,
        "expected_n": expected_n,
    }


def exact_expected_units() -> tuple[set[str], set[str], set[str]]:
    lanes = FINAL.density_execution_lane_units()
    routes = FINAL.route_identity()
    low = lanes["corrected_archive_low_30"]
    D0 = lanes["conditional_archive_D0_56"]
    archive = routes["archive"]
    if (
        len(low) != 30
        or len(D0) != 56
        or low & D0
        or low | D0 != archive
        or len(archive) != 86
    ):
        raise ValueError("archive-86 execution identity differs")
    return low, D0, archive


def validate_component(
    path: Path,
    *,
    expected_n: int,
    expected_record_type: str,
    expected_units: set[str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = FINAL.load_json(path)
    state = source_state(
        path,
        expected_n=expected_n,
        expected_record_type=expected_record_type,
    )
    if not state["header_contract_pass"]:
        raise ValueError(f"component header contract differs: {path}")
    rows = manifest["units"]
    observed_units: set[str] = set()
    verified_rows: list[dict[str, Any]] = []
    selected_count = manifest["selected_streamline_count"]
    for raw in rows:
        if not isinstance(raw, Mapping):
            raise TypeError(f"component row is not a mapping: {path}")
        row = dict(raw)
        unit = str(row.get("unit", ""))
        density = float(row.get("edge_density", -1.0))
        artifacts = row.get("artifacts")
        row_count = row.get("selected_streamline_count")
        if (
            not unit
            or unit in observed_units
            or density < FINAL.MINIMUM_EDGE_DENSITY
            or not isinstance(artifacts, Mapping)
            or (
                row_count is not None
                and row_count != selected_count
            )
        ):
            raise ValueError(f"{unit}:component unit contract differs")
        matrices = FINAL.artifacts_to_matrices(unit, artifacts)
        if set(matrices) != set(FINAL.MATRIX_NAMES):
            raise ValueError(f"{unit}:nine-matrix contract differs")
        observed_units.add(unit)
        row["source_component_manifest"] = FINAL.file_record(path)
        verified_rows.append(row)
    if observed_units != expected_units:
        raise ValueError(
            f"component identity differs: {path} "
            f"observed={len(observed_units)} expected={len(expected_units)}"
        )
    return manifest, verified_rows


def preflight_payload() -> dict[str, Any]:
    low, D0, archive = exact_expected_units()
    low_state = source_state(
        LOW_30, expected_n=30, expected_record_type=LOW_RECORD_TYPE
    )
    D0_state = source_state(
        D0_56, expected_n=56, expected_record_type=D0_RECORD_TYPE
    )
    sources_ready = bool(
        low_state.get("header_contract_pass")
        and D0_state.get("header_contract_pass")
        and low_state.get("selected_streamline_count")
        == D0_state.get("selected_streamline_count")
    )
    return {
        "schema_version": "1.0.0",
        "record_type": (
            "diagnosis_blind_hcp379_corrected_archive_86_merge_plan"
        ),
        "status": (
            "READY_TO_ASSEMBLE_CORRECTED_ARCHIVE_86"
            if sources_ready
            else "WAITING_FOR_CORRECTED_ARCHIVE_COMPONENTS"
        ),
        "generated_utc": FINAL.utc_now(),
        "diagnosis_labels_used": False,
        "diagnosis_or_outcomes_used": False,
        "non_overwriting": True,
        "imaging_executed_by_plan": False,
        "archive_route_selected_by_plan": False,
        "archive_concordance_still_required": True,
        "target_n": 86,
        "component_counts": {
            "corrected_archive_low": len(low),
            "corrected_archive_D0": len(D0),
            "archive_union": len(archive),
        },
        "minimum_density_inclusive": FINAL.MINIMUM_EDGE_DENSITY,
        "required_density_pass_n": 86,
        "required_matrix_names": list(FINAL.MATRIX_NAMES),
        "allowed_streamline_counts": list(FINAL.ALLOWED_STREAMLINE_COUNTS),
        "sources": {
            "corrected_archive_low_30": low_state,
            "corrected_archive_D0_56": D0_state,
        },
        "output_expected": str(OUTPUT),
        "records": {
            "builder": FINAL.file_record(Path(__file__)),
            "final_release_contract": FINAL.file_record(FINAL_SOURCE),
            "density_execution_topology": FINAL.file_record(
                FINAL.DENSITY_EXECUTION_TOPOLOGY
            ),
            "density_execution_subjects": FINAL.file_record(
                FINAL.DENSITY_EXECUTION_SUBJECTS
            ),
        },
    }


def assemble() -> dict[str, Any]:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite: {OUTPUT}")
    low_units, D0_units, archive_units = exact_expected_units()
    low_manifest, low_rows = validate_component(
        LOW_30,
        expected_n=30,
        expected_record_type=LOW_RECORD_TYPE,
        expected_units=low_units,
    )
    D0_manifest, D0_rows = validate_component(
        D0_56,
        expected_n=56,
        expected_record_type=D0_RECORD_TYPE,
        expected_units=D0_units,
    )
    selected_count = low_manifest["selected_streamline_count"]
    if selected_count != D0_manifest["selected_streamline_count"]:
        raise ValueError("archive components use different Phase-B counts")
    rows = low_rows + D0_rows
    observed = {str(row["unit"]) for row in rows}
    if (
        len(rows) != 86
        or len(observed) != 86
        or observed != archive_units
    ):
        raise ValueError("corrected archive-86 union differs")
    rows.sort(key=lambda row: str(row["unit"]))
    manifest = {
        "schema_version": "1.0.0",
        "record_type": OUTPUT_RECORD_TYPE,
        "status": "PASS",
        "generated_utc": FINAL.utc_now(),
        "diagnosis_labels_used": False,
        "diagnosis_or_outcomes_used": False,
        "non_overwriting": True,
        "unit_count": 86,
        "selected_streamline_count": selected_count,
        "density_is_subject_inclusion_gate": False,
        "density_is_whole_cohort_technical_release_gate": True,
        "minimum_edge_density_inclusive": FINAL.MINIMUM_EDGE_DENSITY,
        "subjects_at_or_above_minimum_density": 86,
        "minimum_observed_edge_density": min(
            float(row["edge_density"]) for row in rows
        ),
        "archive_route_selected_by_manifest": False,
        "archive_concordance_still_required": True,
        "eligible_as_primary_only_if_archive_decision": RERUN_DECISION,
        "eligible_as_corrected_sensitivity_if_archive_decision": (
            RETAIN_DECISION
        ),
        "component_counts": {
            "corrected_archive_low": 30,
            "corrected_archive_D0": 56,
        },
        "records": {
            "corrected_archive_low_30": FINAL.file_record(LOW_30),
            "corrected_archive_D0_56": FINAL.file_record(D0_56),
            "builder": FINAL.file_record(Path(__file__)),
            "final_release_contract": FINAL.file_record(FINAL_SOURCE),
            "density_execution_topology": FINAL.file_record(
                FINAL.DENSITY_EXECUTION_TOPOLOGY
            ),
            "density_execution_subjects": FINAL.file_record(
                FINAL.DENSITY_EXECUTION_SUBJECTS
            ),
        },
        "units": rows,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    FINAL.atomic_json(OUTPUT, manifest)
    return manifest


def self_test() -> None:
    low, D0, archive = exact_expected_units()
    if (
        len(FINAL.MATRIX_NAMES) != 9
        or FINAL.MINIMUM_EDGE_DENSITY != 0.60
        or FINAL.ALLOWED_STREAMLINE_COUNTS
        != (3_000_000, 5_000_000, 10_000_000)
        or len(low) != 30
        or len(D0) != 56
        or len(archive) != 86
        or low & D0
    ):
        raise AssertionError("corrected archive-86 self-test differs")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--assemble", action="store_true")
    mode.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        print("CORRECTED_ARCHIVE_86_RECOVERY4_SELF_TEST_PASS")
        return 0
    payload = assemble() if args.assemble else preflight_payload()
    if not args.assemble:
        PLAN.parent.mkdir(parents=True, exist_ok=True)
        FINAL.atomic_json(PLAN, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
