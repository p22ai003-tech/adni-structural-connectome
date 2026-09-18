#!/home/ec2-user/fsl/bin/python
"""Prepare or run the selected Recovery4 ACT recipe on six archive cases.

The default mode validates an exact non-imaging plan.  ``--execute`` is
accepted by the shared Recovery4 runner only after all six archive pretract
states pass, genuine canary visual QC is bound, and Phase B has selected one
stable 3M, 5M, or 10M recipe.  Outputs stay in the archive-calibration root
and cannot masquerade as the corrected 227-subject release.
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
    / "scripts/hcp/"
    "run_hcp379_archive_calibration_pretract_recovery4_v3.py"
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
ROOT = HCP_ROOT / "archive_corrected_calibration_recovery4"
PRIMARY_MANIFEST = (
    ROOT
    / "manifests/"
    "archive_corrected_calibration_recovery4_manifest.json"
)
EXPECTED_N = 6


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ARCHIVE_PRETRACT = load_module(
    PRETRACT_WRAPPER_SOURCE,
    "hcp379_archive_pretract_recovery4_wrapper",
)
TRACT = load_module(
    SOURCE, "hcp379_archive_tractography_recovery4_engine"
)


def specialize(root: Path) -> None:
    """Specialize both Recovery4 layers and the mature matrix engine."""

    ARCHIVE_PRETRACT.specialize(root)
    pretract = TRACT.PRETRACT
    pretract.AUDIT_CSV = AUDIT_CSV
    pretract.AUDIT_SUMMARY = AUDIT_SUMMARY
    pretract.OUTPUT_ROOT = root
    pretract.DEFAULT_HUMAN_QC = ARCHIVE_PRETRACT.DEFAULT_HUMAN_QC
    pretract.EXPECTED_AUDIT_N = EXPECTED_N
    pretract.EXPECTED_CANARY_OVERLAP_N = 0
    pretract.EXPECTED_SCALEUP_N = EXPECTED_N

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
    TRACT.EXPECTED_SCALEUP_N = EXPECTED_N
    TRACT.EXPECTED_CORRECTED_N = EXPECTED_N
    TRACT.EXPECTED_CANARY_OVERLAP_N = 0

    engine = TRACT.ENGINE
    engine.ROOT = root
    engine.AUDIT_CSV = AUDIT_CSV
    engine.AUDIT_SUMMARY = AUDIT_SUMMARY
    engine.PRETRACT_SUMMARY = TRACT.PRETRACT_SUMMARY
    engine.PRETRACT_GATE = TRACT.PRETRACT_PLAN
    engine.EXPECTED_SCALEUP_N = EXPECTED_N
    engine.EXPECTED_CORRECTED_N = EXPECTED_N
    engine.EXPECTED_CANARY_OVERLAP_N = 0


def build_primary_manifest(root: Path) -> dict[str, Any]:
    units = ARCHIVE_PRETRACT.selection_units()
    summary_path = root / "manifests/tractography_summary.json"
    summary = TRACT.PRETRACT.load_json(summary_path)
    gate_path = (
        root / "manifests/tractography_recovery4_execution_gate.json"
    )
    gate = TRACT.PRETRACT.load_json(gate_path)
    if (
        summary.get("status") != "PASS"
        or summary.get("target_n") != EXPECTED_N
        or summary.get("states_present_n") != EXPECTED_N
        or summary.get("status_counts", {}).get("PASS_ALL_NINE")
        != EXPECTED_N
        or gate.get("status") != "PASS"
        or gate.get("diagnosis_labels_used") is not False
    ):
        raise ValueError("archive primary Recovery4 run is not 6/6 PASS")
    selected = int(gate["selected_streamline_count"])
    records = []
    for unit in sorted(units):
        state_path = (
            root / "qc/tractography" / f"{unit}.json"
        )
        state = TRACT.PRETRACT.load_json(state_path)
        if (
            state.get("record_type")
            != "diagnosis_blind_hcp379_scaleup_tractography_subject"
            or state.get("status") != "PASS_ALL_NINE"
            or state.get("diagnosis_labels_used") is not False
            or state.get("unit") != unit
            or int(state.get("selected_streamline_count", -1))
            != selected
            or state.get("bounded_tensor_maps_sampled") is not True
            or state.get("raw_tensor_maps_sampled") is not False
            or not isinstance(
                state.get("recovery4_input_binding"), Mapping
            )
        ):
            raise ValueError(f"{unit}:primary Recovery4 state differs")
        compaction_path = (
            TRACT.ENGINE.tractogram_compaction_path(root, unit)
        )
        compaction = None
        if compaction_path.is_file():
            compaction = TRACT.ENGINE.validate_tractogram_compaction(
                TRACT.PRETRACT.load_json(compaction_path),
                root=root,
                unit=unit,
            )
            artifacts = TRACT.ENGINE.verified_scaleup_artifacts(
                compaction["retained_artifacts"], unit=unit
            )
        else:
            artifacts = TRACT.ENGINE.verified_scaleup_artifacts(
                state.get("artifacts"), unit=unit
            )
        records.append(
            {
                "unit": unit,
                "status": state["status"],
                "selected_streamline_count": selected,
                "edge_density": state["edge_density"],
                "endpoint_assignment_fraction": state[
                    "endpoint_assignment_fraction"
                ],
                "matrix_qc": state["matrix_qc"],
                "artifacts": artifacts,
                "recovery4_input_binding": state[
                    "recovery4_input_binding"
                ],
                "subject_state": TRACT.PRETRACT.file_record(state_path),
                "tractogram_compaction": (
                    TRACT.PRETRACT.file_record(compaction_path)
                    if compaction is not None
                    else None
                ),
            }
        )
    manifest = {
        "schema_version": "1.0.0",
        "record_type": (
            "diagnosis_blind_hcp379_archive_corrected_act_"
            "calibration_recovery4_manifest"
        ),
        "status": "PASS",
        "generated_utc": TRACT.PRETRACT.utc_now(),
        "diagnosis_labels_used": False,
        "unit_count": len(records),
        "selected_streamline_count": selected,
        "density_is_subject_inclusion_gate": False,
        "records": {
            "selection": TRACT.PRETRACT.file_record(SELECTION),
            "input_audit": TRACT.PRETRACT.file_record(AUDIT_CSV),
            "input_audit_summary": TRACT.PRETRACT.file_record(
                AUDIT_SUMMARY
            ),
            "pretract_summary": TRACT.PRETRACT.file_record(
                root / "manifests/pretract_recovery4_summary.json"
            ),
            "pretract_plan": TRACT.PRETRACT.file_record(
                root / "manifests/pretract_recovery4_plan.json"
            ),
            "tractography_summary": TRACT.PRETRACT.file_record(
                summary_path
            ),
            "execution_gate": TRACT.PRETRACT.file_record(gate_path),
            "phase_b_validation": gate["records"][
                "phase_validation"
            ],
            "phase_b_manifest": gate["records"]["phase_manifest"],
            "shared_recovery4_runner": TRACT.PRETRACT.file_record(SOURCE),
            "wrapper": TRACT.PRETRACT.file_record(Path(__file__)),
        },
        "units": records,
    }
    TRACT.PRETRACT.atomic_json(
        root
        / "manifests/"
        "archive_corrected_calibration_recovery4_manifest.json",
        manifest,
    )
    return manifest


def write_wrapper_plan(
    root: Path, *, compact_tractograms_after_pass: bool
) -> dict[str, Any]:
    shared_path = (
        root
        / "manifests/"
        "tractography_recovery4_selected_recipe_plan.json"
    )
    shared = TRACT.PRETRACT.load_json(shared_path)
    units = ARCHIVE_PRETRACT.selection_units()
    if (
        shared.get("status") != "PRE_PHASE_DRY_RUN_PASS"
        or shared.get("diagnosis_labels_used") is not False
        or shared.get("imaging_executed_by_plan") is not False
        or shared.get("execution_authorized") is not False
        or shared.get("target_noncanary_n") != EXPECTED_N
        or shared.get("canary_overlap_n") != 0
        or shared.get("corrected_release_n") != EXPECTED_N
        or {row.get("unit") for row in shared.get("units", [])}
        != units
    ):
        raise ValueError("archive tractography pre-Phase plan differs")
    plan = {
        "schema_version": "1.0.0",
        "record_type": (
            "diagnosis_blind_hcp379_archive_tractography_"
            "recovery4_plan"
        ),
        "status": "PRE_PHASE_DRY_RUN_PASS",
        "generated_utc": TRACT.PRETRACT.utc_now(),
        "diagnosis_labels_used": False,
        "imaging_executed_by_plan": False,
        "execution_authorized": False,
        "compact_tractograms_after_pass": (
            compact_tractograms_after_pass
        ),
        "target_n": EXPECTED_N,
        "output_root": str(root.resolve()),
        "units": sorted(units),
        "allowed_streamline_counts": list(TRACT.ALLOWED_COUNTS),
        "records": {
            "selection": TRACT.PRETRACT.file_record(SELECTION),
            "shared_plan": TRACT.PRETRACT.file_record(shared_path),
            "shared_recovery4_runner": TRACT.PRETRACT.file_record(SOURCE),
            "wrapper": TRACT.PRETRACT.file_record(Path(__file__)),
        },
    }
    TRACT.PRETRACT.atomic_json(
        root
        / "manifests/"
        "archive_tractography_recovery4_plan.json",
        plan,
    )
    return plan


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

    units = ARCHIVE_PRETRACT.selection_units()
    specialize(args.root)
    if args.self_test:
        # The mature engine's synthetic matrix test creates its own minimal
        # pretract tree and must run before the Recovery4 adapter requires
        # promoted real-subject states.
        TRACT.ENGINE.self_test()
        TRACT.install_recovery4_adapter()
        if len(units) != EXPECTED_N:
            raise AssertionError(units)
        print("ARCHIVE_TRACTOGRAPHY_RECOVERY4_SELF_TEST_PASS")
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
    if args.execute:
        # A limit equal to the complete six-case lane prevents the shared
        # runner from emitting a misleading corrected-227 manifest.
        delegated.extend(("--limit", str(EXPECTED_N)))
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
        build_primary_manifest(args.root)
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
