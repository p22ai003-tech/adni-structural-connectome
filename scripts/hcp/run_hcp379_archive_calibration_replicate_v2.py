#!/usr/bin/env python3
"""Generate an independent corrected-ACT seed for six archive calibrators."""

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
CAL_TRACT_SOURCE = (
    EXP
    / "scripts/hcp/"
    "run_hcp379_archive_calibration_tractography_v2.py"
)
ROOT = HCP_ROOT / "archive_corrected_calibration"
PRIMARY_MANIFEST = (
    ROOT / "manifests/archive_corrected_calibration_manifest.json"
)
SUMMARY_PATH = (
    ROOT / "manifests/archive_corrected_replicate_summary.json"
)
OUTPUT_MANIFEST = (
    ROOT / "manifests/archive_corrected_replicate_manifest.json"
)
EXPECTED_N = 6
SEED_NAMESPACE = "archive_corrected_act_replicate_v1"


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CAL = load_module(
    CAL_TRACT_SOURCE,
    "run_hcp379_archive_calibration_tractography_v2",
)
TRACT = CAL.TRACT


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
    pretract = TRACT.PRETRACT.stage_paths(root, unit)
    subject = pretract["subject"]
    output = subject / "09_archive_calibration_replicate"
    matrices = {
        name: output / "connectome" / f"{name}.csv"
        for name in TRACT.MATRIX_NAMES
    }
    samples = {
        metric: output / "samples" / f"{metric}.csv"
        for metric in ("fa", "md", "rd", "ad")
    }
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
        "matrices": matrices,
        "samples": samples,
    }


def validate_primary(root: Path, units: set[str]) -> dict[str, Any]:
    path = root / "manifests/archive_corrected_calibration_manifest.json"
    primary = TRACT.PRETRACT.load_json(path)
    observed = {
        str(row.get("unit", ""))
        for row in primary.get("units", [])
        if isinstance(row, dict)
    }
    if (
        primary.get("status") != "PASS"
        or primary.get("diagnosis_labels_used") is not False
        or primary.get("unit_count") != EXPECTED_N
        or observed != units
    ):
        raise ValueError("primary archive calibration is not 6/6 PASS")
    return primary


def build_records(root: Path, units: set[str]) -> list[dict[str, Any]]:
    records = []
    for unit in sorted(units):
        state_path = (
            root / "qc/tractography_replicate" / f"{unit}.json"
        )
        state = TRACT.PRETRACT.load_json(state_path)
        if state.get("status") != "PASS_ALL_NINE":
            raise ValueError(f"{unit}:replicate state is not PASS")
        artifacts = state.get("artifacts")
        if not isinstance(artifacts, dict) or len(artifacts) < 14:
            raise ValueError(f"{unit}:replicate artifacts differ")
        for record in artifacts.values():
            path = Path(str(record.get("path", "")))
            if record != TRACT.PRETRACT.file_record(path):
                raise ValueError(f"{unit}:replicate binding differs: {path}")
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
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
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
        raise ValueError("worker/thread request exceeds the host contract")
    units = CAL.CAL_PRETRACT.validate_binding()
    CAL.specialize(args.root)
    primary = (
        None if args.self_test else validate_primary(args.root, units)
    )
    primary_token, primary_seed = TRACT.subject_seed(
        {
            "dti_source_id": "a" * 64,
            "dti_raw_bundle_sha256": "b" * 64,
        },
        "recipe",
    )
    replicate_token, replicate_value = replicate_seed(
        {
            "dti_source_id": "a" * 64,
            "dti_raw_bundle_sha256": "b" * 64,
        },
        "recipe",
    )
    if (
        primary_token == replicate_token
        or primary_seed == replicate_value
    ):
        raise AssertionError("primary and replicate seeds are not distinct")
    if args.self_test:
        print("ARCHIVE_REPLICATE_SELF_TEST_PASS")
        return 0
    if shutil.disk_usage(args.root).free < TRACT.MINIMUM_START_FREE_BYTES:
        raise RuntimeError("storage is below the tractography start floor")

    gate = TRACT.load_gate(args.root)
    targets = TRACT.input_rows(gate)
    if (
        len(targets) != EXPECTED_N
        or int(primary["selected_streamline_count"])
        != int(gate["selected_count"])
    ):
        raise ValueError("archive replicate/primary gate differs")
    TRACT.tract_paths = replicate_paths
    TRACT.subject_seed = replicate_seed
    for relative in (
        "qc/tractography_replicate",
        "locks/tractography_replicate",
        "logs/tractography_replicate",
        "manifests",
    ):
        (args.root / relative).mkdir(parents=True, exist_ok=True)
    print(
        f"[{TRACT.PRETRACT.utc_now()}] "
        f"HCP379_ARCHIVE_REPLICATE_START n={len(targets)} "
        f"workers={args.workers} nthreads={args.nthreads} "
        f"selected={gate['selected_count']}",
        flush=True,
    )
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                TRACT.process_unit,
                row,
                root=args.root,
                gate=gate,
                nthreads=args.nthreads,
            ): row["unit"]
            for row in targets
        }
        for index, future in enumerate(as_completed(futures), start=1):
            state = future.result()
            results.append(state)
            print(
                f"[{TRACT.PRETRACT.utc_now()}] {index}/{EXPECTED_N} "
                f"{state['unit']} {state['status']} "
                f"density={state.get('edge_density')} "
                f"error={state.get('error')}",
                flush=True,
            )
    counts: dict[str, int] = {}
    for state in results:
        status = str(state.get("status"))
        counts[status] = counts.get(status, 0) + 1
    summary = {
        "schema_version": "1.0.0",
        "record_type": (
            "diagnosis_blind_hcp379_archive_corrected_replicate_summary"
        ),
        "status": (
            "PASS"
            if len(results) == EXPECTED_N
            and counts.get("PASS_ALL_NINE") == EXPECTED_N
            else "FAIL"
        ),
        "updated_utc": TRACT.PRETRACT.utc_now(),
        "diagnosis_labels_used": False,
        "seed_namespace": SEED_NAMESPACE,
        "target_n": EXPECTED_N,
        "states_present_n": len(results),
        "status_counts": dict(sorted(counts.items())),
    }
    TRACT.PRETRACT.atomic_json(
        args.root
        / "manifests/archive_corrected_replicate_summary.json",
        summary,
    )
    if summary["status"] != "PASS":
        return 1
    records = build_records(args.root, units)
    manifest = {
        "schema_version": "1.0.0",
        "record_type": (
            "diagnosis_blind_hcp379_archive_corrected_act_"
            "replicate_manifest"
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
                args.root
                / "manifests/archive_corrected_calibration_manifest.json"
            ),
            "replicate_summary": TRACT.PRETRACT.file_record(
                args.root
                / "manifests/archive_corrected_replicate_summary.json"
            ),
            "phase_b_validation": TRACT.PRETRACT.file_record(
                TRACT.PHASE_VALIDATION
            ),
            "shared_tractography_source": TRACT.PRETRACT.file_record(
                CAL.SOURCE
            ),
        },
        "units": records,
    }
    TRACT.PRETRACT.atomic_json(
        args.root
        / "manifests/archive_corrected_replicate_manifest.json",
        manifest,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
