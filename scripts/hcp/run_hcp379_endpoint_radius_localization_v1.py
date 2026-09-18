#!/usr/bin/env python3
"""Localize disconnected HCP379 parcels with endpoint-radius diagnostics.

The corrected 009 and selected-BBR 036 HROI constructions contain all 379
labels, yet their balanced 20M count matrices retain two and one disconnected
nodes respectively.  This diagnostic holds the primary 10M tractogram and
atlas fixed, then evaluates radial endpoint assignment at 2, 4, 6 and 8 mm.

The radii above 4 mm are diagnostic only.  They cannot be selected for
production from density or node recovery because a wider radius may introduce
false-positive anatomical assignments.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np


EXP = Path("/home/ec2-user/exp")
HCP_ROOT = Path("/data/derivatives/hcp379_v2")
DENSITY_SOURCE = (
    EXP
    / "scripts/hcp/"
    "build_hcp379_hroi_balanced_seed_density_canary_v1.py"
)
ATTEMPTS = (
    HCP_ROOT / "phase_b_hroi_endpoint_radius_localization_v1/attempts"
)
MRTRIX = Path("/home/ec2-user/mrtrix3/bin")
RADII = (2, 4, 6, 8)
STREAMLINES = 10_000_000
EXPECTED_NODES = 379
CASES = {
    "009_S_4324_I1186579": {
        "summary": Path(
            "/data/derivatives/hcp379_v2/"
            "phase_b_hroi_balanced_density_canary_v1/attempts/"
            "20260729T052406.315883Z-hroi-balanced-density-"
            "009_S_4324_I1186579/summary.json"
        ),
        "expected_disconnected": {77: "L_a47r_ROI", 270: "R_10pp_ROI"},
    },
    "036_S_2380_I1023443": {
        "summary": Path(
            "/data/derivatives/hcp379_v2/"
            "phase_b_hroi_balanced_density_canary_v1/attempts/"
            "20260728T234410.385073Z-hroi-balanced-density-"
            "036_S_2380_I1023443/summary.json"
        ),
        "expected_disconnected": {303: "R_STGa_ROI"},
    },
}


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


DENSITY = load_module(DENSITY_SOURCE, "hcp379_endpoint_radius_density")
BALANCED = DENSITY.BALANCED


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".partial",
        delete=False,
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def disconnected(matrix: np.ndarray) -> list[int]:
    return [
        index + 1
        for index, value in enumerate(np.sum(matrix, axis=1))
        if float(value) <= 0.0
    ]


def environment() -> dict[str, str]:
    value = os.environ.copy()
    value["PATH"] = f"{MRTRIX}:{value.get('PATH', '')}"
    return value


def case_inputs(unit: str) -> dict[str, Any]:
    spec = CASES[unit]
    summary_path = spec["summary"]
    summary = load_json(summary_path)
    atlas_result = load_json(
        Path(summary["hroi_atlas_result"]["path"])
    )
    nodes = Path(
        atlas_result["artifacts"]["hcp379_nodes_b0_1mm"]["path"]
    ).resolve()
    tracks = DENSITY.phase_paths(unit)["primary_tracks"]
    baseline_count = Path(
        summary["seeds"]["primary"]["hroi_count_10m"]["path"]
    ).resolve()
    baseline_assignments = Path(
        summary["seeds"]["primary"]["hroi_assignments_10m"]["path"]
    ).resolve()
    matrix = BALANCED.load_count(baseline_count)
    observed_disconnected = set(disconnected(matrix))
    expected_disconnected = set(spec["expected_disconnected"])
    if (
        summary.get("status")
        != "PASS_HROI_BALANCED_DENSITY_CANARY"
        or summary.get("unit") != unit
        or atlas_result.get("status") != "PASS_HROI_CANARY_ATLAS"
        or atlas_result.get("atlas_qc", {}).get("labels_found")
        != EXPECTED_NODES
        or DENSITY.tck_count(tracks) != STREAMLINES
        or observed_disconnected != expected_disconnected
    ):
        raise ValueError(f"{unit}: endpoint-localization inputs differ")
    for path in (
        summary_path,
        nodes,
        tracks,
        baseline_count,
        baseline_assignments,
        MRTRIX / "tck2connectome",
    ):
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"{unit}: required regular file differs: {path}")
    return {
        "summary_path": summary_path,
        "summary": summary,
        "atlas_result": atlas_result,
        "nodes": nodes,
        "tracks": tracks,
        "baseline_count": baseline_count,
        "baseline_assignments": baseline_assignments,
        "expected_disconnected": spec["expected_disconnected"],
    }


def run_radius(
    unit: str,
    bound: Mapping[str, Any],
    radius: int,
    root: Path,
    threads: int,
) -> dict[str, Any]:
    radius_root = root / f"radius_{radius}mm"
    radius_root.mkdir(parents=True, exist_ok=False)
    if radius == 4:
        count_path = bound["baseline_count"]
        assignments_path = bound["baseline_assignments"]
        reused = True
    else:
        count_path = radius_root / "count.csv"
        assignments_path = radius_root / "assignments.csv"
        partial_count = radius_root / ".count.partial.csv"
        partial_assignments = radius_root / ".assignments.partial.csv"
        command = [
            str(MRTRIX / "tck2connectome"),
            str(bound["tracks"]),
            str(bound["nodes"]),
            str(partial_count),
            "-assignment_radial_search",
            str(radius),
            "-symmetric",
            "-zero_diagonal",
            "-stat_edge",
            "sum",
            "-out_assignments",
            str(partial_assignments),
            "-nthreads",
            str(threads),
            "-force",
        ]
        started = time.monotonic()
        log = radius_root / "tck2connectome.log"
        with log.open("x", encoding="utf-8") as handle:
            handle.write("COMMAND " + " ".join(command) + "\n")
            handle.flush()
            completed = subprocess.run(
                command,
                stdout=handle,
                stderr=subprocess.STDOUT,
                env=environment(),
                check=False,
            )
            handle.write(
                f"\nRETURNCODE {completed.returncode}\n"
                f"ELAPSED_SECONDS {time.monotonic() - started:.3f}\n"
            )
        if completed.returncode != 0:
            raise RuntimeError(f"{unit}/{radius}mm failed: {log}")
        os.replace(partial_count, count_path)
        os.replace(partial_assignments, assignments_path)
        reused = False
    matrix = BALANCED.load_count(count_path)
    assignments = BALANCED.load_assignments(assignments_path)
    replay = BALANCED.count_matrix(assignments, STREAMLINES)
    if not np.array_equal(matrix, replay):
        raise ValueError(f"{unit}/{radius}mm assignment replay differs")
    missing = disconnected(matrix)
    target_nodes = {
        str(node): {
            "name": name,
            "connected": node not in missing,
            "strength": int(np.sum(matrix[node - 1])),
        }
        for node, name in bound["expected_disconnected"].items()
    }
    return {
        "radius_mm": radius,
        "reused_existing_4mm": reused,
        **BALANCED.matrix_qc(matrix),
        "endpoint_assignment_fraction": BALANCED.assignment_fraction(
            assignments, STREAMLINES
        ),
        "disconnected_nodes": missing,
        "target_nodes": target_nodes,
        "count_matrix": BALANCED.file_record(count_path),
        "assignments": BALANCED.file_record(assignments_path),
    }


def classify(levels: Mapping[str, Any], targets: set[int]) -> str:
    connected_by_radius: dict[int, list[int]] = {}
    for radius in RADII:
        row = levels[str(radius)]
        connected_by_radius[radius] = [
            node
            for node in targets
            if row["target_nodes"][str(node)]["connected"]
        ]
    if connected_by_radius[4]:
        return "BASELINE_CONTRACT_UNEXPECTEDLY_CONNECTED"
    if connected_by_radius[6] or connected_by_radius[8]:
        return "ENDPOINTS_NEAR_PARCEL_BUT_OUTSIDE_4MM"
    return "NO_TRACTOGRAPHY_ENDPOINT_SUPPORT_WITHIN_8MM"


def preflight() -> dict[str, Any]:
    bound = {unit: case_inputs(unit) for unit in CASES}
    return {
        "schema_version": "1.0.0",
        "record_type": (
            "diagnosis_blind_hcp379_endpoint_radius_localization_preflight"
        ),
        "generated_utc": utc_now(),
        "status": "READY_ENDPOINT_RADIUS_LOCALIZATION",
        "diagnosis_labels_used": False,
        "outcomes_used": False,
        "units": list(CASES),
        "diagnostic_radii_mm": list(RADII),
        "existing_4mm_reused": True,
        "new_tractography_generated": False,
        "production_assignment_radius_selected": False,
        "scale_up_authorized": False,
        "targets": {
            unit: data["expected_disconnected"] for unit, data in bound.items()
        },
    }


def execute(workers: int, threads: int) -> dict[str, Any]:
    bound = {unit: case_inputs(unit) for unit in CASES}
    attempt = ATTEMPTS / f"{stamp()}-endpoint-radius-localization"
    attempt.mkdir(parents=True, exist_ok=False)
    atomic_json(
        attempt / "preflight.json",
        {
            **preflight(),
            "status": "EXECUTION_INPUTS_BOUND",
            "implementation": BALANCED.file_record(Path(__file__)),
            "inputs": {
                unit: {
                    "summary": BALANCED.file_record(data["summary_path"]),
                    "nodes": BALANCED.file_record(data["nodes"]),
                    "tracks": BALANCED.file_record(data["tracks"]),
                }
                for unit, data in bound.items()
            },
        },
    )
    levels: dict[str, dict[str, Any]] = {unit: {} for unit in CASES}
    tasks = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for unit, data in bound.items():
            for radius in RADII:
                tasks.append(
                    (
                        unit,
                        radius,
                        pool.submit(
                            run_radius,
                            unit,
                            data,
                            radius,
                            attempt / unit,
                            threads,
                        ),
                    )
                )
        for unit, radius, future in tasks:
            levels[unit][str(radius)] = future.result()
    units = {}
    for unit, rows in levels.items():
        localization = classify(
            rows, set(CASES[unit]["expected_disconnected"])
        )
        units[unit] = {
            "status": "PASS_ENDPOINT_RADIUS_LOCALIZATION",
            "localization": localization,
            "levels": rows,
        }
    summary = {
        "schema_version": "1.0.0",
        "record_type": (
            "diagnosis_blind_hcp379_endpoint_radius_localization"
        ),
        "generated_utc": utc_now(),
        "status": "PASS_COMPLETE_ENDPOINT_RADIUS_LOCALIZATION",
        "diagnosis_labels_used": False,
        "outcomes_used": False,
        "non_overwriting": True,
        "source_files_modified": False,
        "new_tractography_generated": False,
        "production_assignment_radius_selected": False,
        "radii_above_4mm_are_diagnostic_only": True,
        "units": units,
        "scale_up_authorized": False,
        "preflight": BALANCED.file_record(attempt / "preflight.json"),
        "implementation": BALANCED.file_record(Path(__file__)),
    }
    atomic_json(attempt / "summary.json", summary)
    return {
        "attempt_root": str(attempt),
        "status": summary["status"],
        "localization": {
            unit: row["localization"] for unit, row in units.items()
        },
        "target_status_by_radius": {
            unit: {
                radius: {
                    node: target
                    for node, target in row["levels"][radius][
                        "target_nodes"
                    ].items()
                }
                for radius in map(str, RADII)
            }
            for unit, row in units.items()
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--threads-per-job", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.workers <= 2:
        parser.error("--workers must be in 1..2")
    if not 1 <= args.threads_per_job <= 16:
        parser.error("--threads-per-job must be in 1..16")
    if args.workers * args.threads_per_job > 16:
        parser.error("worker/thread product must not exceed 16")
    value = (
        execute(args.workers, args.threads_per_job)
        if args.execute
        else preflight()
    )
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
