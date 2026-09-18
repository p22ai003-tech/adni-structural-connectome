#!/home/ec2-user/fsl/bin/python
"""Prepare or run an independent Recovery4 seed for six archive calibrators."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping


EXP = Path("/home/ec2-user/exp")
HCP_ROOT = Path("/data/derivatives/hcp379_v2")
PRIMARY_WRAPPER_SOURCE = (
    EXP
    / "scripts/hcp/"
    "run_hcp379_archive_calibration_tractography_recovery4_v3.py"
)
ROOT = HCP_ROOT / "archive_corrected_calibration_recovery4"
PRIMARY_MANIFEST = (
    ROOT
    / "manifests/"
    "archive_corrected_calibration_recovery4_manifest.json"
)
REPLICATE_MANIFEST = (
    ROOT
    / "manifests/"
    "archive_corrected_replicate_recovery4_manifest.json"
)
EXPECTED_N = 6
SEED_NAMESPACE = "archive_corrected_act_replicate_recovery4_v1"


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PRIMARY = load_module(
    PRIMARY_WRAPPER_SOURCE,
    "hcp379_archive_primary_recovery4_wrapper",
)
TRACT = PRIMARY.TRACT


def replicate_seed(
    row: Mapping[str, str], recipe_id: str
) -> tuple[str, int]:
    token = (
        f"{row['dti_source_id']}|{row['dti_raw_bundle_sha256']}|"
        f"{recipe_id}|{SEED_NAMESPACE}"
    )
    seed = int.from_bytes(
        hashlib.sha256(token.encode("utf-8")).digest()[:4], "big"
    ) & 0x7FFFFFFF
    return token, seed


def replicate_paths(root: Path, unit: str) -> dict[str, Any]:
    pretract = TRACT.PRETRACT.spatial_paths(root, unit)
    subject = pretract["subject"]
    output = subject / "09_archive_calibration_replicate_recovery4"
    return {
        **pretract,
        "tract_state": (
            root / "qc/tractography_replicate" / f"{unit}.json"
        ),
        "tract_lock": (
            root / "locks/tractography_replicate" / f"{unit}.lock"
        ),
        "tract_log": (
            root / "logs/tractography_replicate" / f"{unit}.log"
        ),
        "tracks": output / "tracks_selected.tck",
        "tract_metadata": output / "tractography_parameters.json",
        "weights": output / "sift2_weights.txt",
        "sift2_mu": output / "sift2_mu.txt",
        "sift2_iterations": output / "sift2_iterations.csv",
        "assignments": output / "assignments.csv",
        "matrices": {
            name: output / "connectome" / f"{name}.csv"
            for name in TRACT.ENGINE.MATRIX_NAMES
        },
        "samples": {
            metric: output / "samples" / f"{metric}.csv"
            for metric in ("fa", "md", "rd", "ad")
        },
    }


def validate_primary(
    root: Path, units: set[str], selected: int
) -> dict[str, Any]:
    path = (
        root
        / "manifests/"
        "archive_corrected_calibration_recovery4_manifest.json"
    )
    manifest = TRACT.PRETRACT.load_json(path)
    observed = {
        str(row.get("unit", ""))
        for row in manifest.get("units", [])
        if isinstance(row, Mapping)
    }
    if (
        manifest.get("record_type")
        != (
            "diagnosis_blind_hcp379_archive_corrected_act_"
            "calibration_recovery4_manifest"
        )
        or manifest.get("status") != "PASS"
        or manifest.get("diagnosis_labels_used") is not False
        or manifest.get("unit_count") != EXPECTED_N
        or int(manifest.get("selected_streamline_count", -1))
        != selected
        or observed != units
    ):
        raise ValueError("primary archive Recovery4 manifest differs")
    return manifest


def build_records(root: Path, units: set[str], selected: int) -> list[dict[str, Any]]:
    records = []
    for unit in sorted(units):
        state_path = (
            root / "qc/tractography_replicate" / f"{unit}.json"
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
        ):
            raise ValueError(f"{unit}:replicate Recovery4 state differs")
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
            }
        )
    return records


def write_plan(root: Path) -> dict[str, Any]:
    rows, overlap = TRACT.validate_prephase_scope()
    units = PRIMARY.ARCHIVE_PRETRACT.selection_units()
    if (
        len(rows) != EXPECTED_N
        or overlap
        or {row["unit"] for row in rows} != units
    ):
        raise ValueError("archive replicate pre-Phase scope differs")
    primary_token, primary_seed = TRACT.ENGINE.subject_seed(
        {
            "dti_source_id": "a" * 64,
            "dti_raw_bundle_sha256": "b" * 64,
        },
        "recipe",
    )
    independent_token, independent_seed = replicate_seed(
        {
            "dti_source_id": "a" * 64,
            "dti_raw_bundle_sha256": "b" * 64,
        },
        "recipe",
    )
    if (
        primary_token == independent_token
        or primary_seed == independent_seed
    ):
        raise AssertionError("primary and replicate seeds are not distinct")
    plan = {
        "schema_version": "1.0.0",
        "record_type": (
            "diagnosis_blind_hcp379_archive_replicate_"
            "recovery4_plan"
        ),
        "status": "PRE_PHASE_DRY_RUN_PASS",
        "generated_utc": TRACT.PRETRACT.utc_now(),
        "diagnosis_labels_used": False,
        "imaging_executed_by_plan": False,
        "execution_authorized": False,
        "target_n": EXPECTED_N,
        "output_root": str(root.resolve()),
        "units": sorted(units),
        "seed_namespace": SEED_NAMESPACE,
        "seed_distinctness_self_test": True,
        "phase_b_status": TRACT.phase_status(),
        "allowed_streamline_counts": list(TRACT.ALLOWED_COUNTS),
        "records": {
            "selection": TRACT.PRETRACT.file_record(PRIMARY.SELECTION),
            "shared_recovery4_runner": TRACT.PRETRACT.file_record(
                TRACT.SOURCE if hasattr(TRACT, "SOURCE") else PRIMARY.SOURCE
            ),
            "primary_wrapper": TRACT.PRETRACT.file_record(
                PRIMARY_WRAPPER_SOURCE
            ),
            "wrapper": TRACT.PRETRACT.file_record(Path(__file__)),
        },
    }
    TRACT.PRETRACT.atomic_json(
        root
        / "manifests/"
        "archive_replicate_recovery4_plan.json",
        plan,
    )
    return plan


def execute(root: Path, *, workers: int, nthreads: int) -> dict[str, Any]:
    if shutil.disk_usage(root).free < TRACT.MINIMUM_START_FREE_BYTES:
        raise RuntimeError("storage is below the tractography start floor")
    gate = TRACT.validate_full_gate(root)
    rows = list(gate["rows"])
    units = PRIMARY.ARCHIVE_PRETRACT.selection_units()
    if (
        len(rows) != EXPECTED_N
        or {row["unit"] for row in rows} != units
    ):
        raise ValueError("archive replicate execution scope differs")
    primary = validate_primary(root, units, gate["selected_count"])

    TRACT.ENGINE.tract_paths = replicate_paths
    TRACT.ENGINE.subject_seed = replicate_seed
    for relative in (
        "qc/tractography_replicate",
        "locks/tractography_replicate",
        "logs/tractography_replicate",
        "manifests",
    ):
        (root / relative).mkdir(parents=True, exist_ok=True)

    states: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                TRACT.process_unit,
                row,
                root=root,
                gate=gate,
                nthreads=nthreads,
            ): row["unit"]
            for row in rows
        }
        for index, future in enumerate(as_completed(futures), start=1):
            state = future.result()
            states.append(state)
            print(
                f"[{TRACT.PRETRACT.utc_now()}] "
                f"archive_replicate {index}/{EXPECTED_N} "
                f"{state.get('unit')} {state.get('status')} "
                f"density={state.get('edge_density')} "
                f"error={state.get('error')}",
                flush=True,
            )
    counts: dict[str, int] = {}
    for state in states:
        key = str(state.get("status"))
        counts[key] = counts.get(key, 0) + 1
    summary = {
        "schema_version": "1.0.0",
        "record_type": (
            "diagnosis_blind_hcp379_archive_corrected_"
            "replicate_recovery4_summary"
        ),
        "status": (
            "PASS"
            if len(states) == EXPECTED_N
            and counts.get("PASS_ALL_NINE") == EXPECTED_N
            else "FAIL"
        ),
        "updated_utc": TRACT.PRETRACT.utc_now(),
        "diagnosis_labels_used": False,
        "seed_namespace": SEED_NAMESPACE,
        "target_n": EXPECTED_N,
        "states_present_n": len(states),
        "status_counts": dict(sorted(counts.items())),
    }
    summary_path = (
        root
        / "manifests/"
        "archive_corrected_replicate_recovery4_summary.json"
    )
    TRACT.PRETRACT.atomic_json(summary_path, summary)
    if summary["status"] != "PASS":
        return summary
    records = build_records(root, units, gate["selected_count"])
    manifest = {
        "schema_version": "1.0.0",
        "record_type": (
            "diagnosis_blind_hcp379_archive_corrected_act_"
            "replicate_recovery4_manifest"
        ),
        "status": "PASS",
        "generated_utc": TRACT.PRETRACT.utc_now(),
        "diagnosis_labels_used": False,
        "seed_namespace": SEED_NAMESPACE,
        "unit_count": len(records),
        "selected_streamline_count": gate["selected_count"],
        "density_is_subject_inclusion_gate": False,
        "records": {
            "primary_calibration": TRACT.PRETRACT.file_record(
                root
                / "manifests/"
                "archive_corrected_calibration_recovery4_manifest.json"
            ),
            "replicate_summary": TRACT.PRETRACT.file_record(summary_path),
            "phase_b_validation": gate["records"]["phase_validation"],
            "phase_b_manifest": gate["records"]["phase_manifest"],
            "shared_recovery4_runner": TRACT.PRETRACT.file_record(
                PRIMARY.SOURCE
            ),
            "primary_wrapper": TRACT.PRETRACT.file_record(
                PRIMARY_WRAPPER_SOURCE
            ),
            "wrapper": TRACT.PRETRACT.file_record(Path(__file__)),
        },
        "units": records,
    }
    TRACT.PRETRACT.atomic_json(
        root
        / "manifests/"
        "archive_corrected_replicate_recovery4_manifest.json",
        manifest,
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--pre-phase-dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--nthreads", type=int, default=16)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if (
        not 1 <= args.workers <= 4
        or not 1 <= args.nthreads <= 16
        or args.workers * args.nthreads > (os.cpu_count() or 1)
    ):
        parser.error("worker/thread request exceeds the host contract")

    PRIMARY.specialize(args.root)
    TRACT.install_recovery4_adapter()
    if args.self_test:
        plan = write_plan(args.root)
        if plan["status"] != "PRE_PHASE_DRY_RUN_PASS":
            raise AssertionError(plan)
        print("ARCHIVE_REPLICATE_RECOVERY4_SELF_TEST_PASS")
        return 0
    result = (
        execute(
            args.root,
            workers=args.workers,
            nthreads=args.nthreads,
        )
        if args.execute
        else write_plan(args.root)
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return int(result.get("status") not in {"PASS", "PRE_PHASE_DRY_RUN_PASS"})


if __name__ == "__main__":
    raise SystemExit(main())
