#!/usr/bin/env python3
"""Build a corrected-HROI balanced two-seed density canary.

This bounded calibration reuses one Phase-B canary's exact primary and
independent 10M tractograms.  It builds that unit's non-overwriting HROI
atlas, reassigns both tractograms to the corrected 379-node image, and
projects balanced totals of 6M, 10M, 15M and 20M from deterministic nested
prefixes.

It does not generate tractography, SIFT2 weights, scalar matrices, biological
statistics, a cohort recipe, or release authorization.  A qualifying density
for this one canary remains directional until the complete Phase-B canary
family and all nine matrices pass.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np


EXP = Path("/home/ec2-user/exp")
HCP_ROOT = Path("/data/derivatives/hcp379_v2")
PHASE_SUBJECT_ROOT = Path(
    "/data/derivatives/scforge_v2/"
    "h04a_r1_recovery_20260719_retry4/subjects"
)
ATLAS_SOURCE = (
    EXP / "scripts/hcp/build_hcp379_hroi_canary_atlas_v1.py"
)
BALANCED_SOURCE = (
    EXP / "scripts/hcp/build_hcp379_balanced_seed_density_projection_v1.py"
)
TCK_COUNTER = EXP / "scripts/hcp/tck_header_count_v1.py"
MRTRIX = Path("/home/ec2-user/mrtrix3/bin")
ATTEMPTS = (
    HCP_ROOT
    / "phase_b_hroi_balanced_density_canary_v1/attempts"
)
DEFAULT_UNIT = "032_S_6804_I1230908"
EXPECTED_STREAMLINES = 10_000_000
EXPECTED_NODES = 379
DENSITY_TARGET = 0.60
BALANCED_TOTALS = {
    6_000_000: 3_000_000,
    10_000_000: 5_000_000,
    15_000_000: 7_500_000,
    20_000_000: 10_000_000,
}


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ATLAS = load_module(ATLAS_SOURCE, "hcp379_hroi_balanced_atlas")
BALANCED = load_module(BALANCED_SOURCE, "hcp379_hroi_balanced_base")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def attempt_stamp() -> str:
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


def atomic_matrix(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial")
    np.savetxt(temporary, value, delimiter=",", fmt="%d")
    os.replace(temporary, path)


def phase_paths(unit: str) -> dict[str, Path]:
    root = PHASE_SUBJECT_ROOT / unit / "09_hcp379_stability"
    return {
        "primary_tracks": root / "primary_10m/tracks.tck",
        "primary_metadata": (
            root / "primary_10m/tractography_parameters.json"
        ),
        "primary_original_count": root / "primary_10m/matrices/count.csv",
        "independent_tracks": root / "independent_10m/tracks.tck",
        "independent_metadata": (
            root / "independent_10m/tractography_parameters.json"
        ),
        "independent_original_count": (
            root / "independent_10m/matrices/count.csv"
        ),
    }


def tck_count(path: Path) -> int:
    result = subprocess.run(
        [
            str(ATLAS.BASE.FSL_PYTHON),
            str(TCK_COUNTER),
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return int(result.stdout.strip())


def validate_phase_pair(unit: str) -> dict[str, Any]:
    paths = phase_paths(unit)
    required = {
        name: path
        for name, path in paths.items()
        if not name.endswith("_original_count")
    }
    missing = [
        name
        for name, path in required.items()
        if not path.is_file() or path.is_symlink()
    ]
    if missing:
        return {
            "ready": False,
            "missing": missing,
            "paths": {name: str(path) for name, path in paths.items()},
        }
    primary_metadata = BALANCED.validate_metadata(
        paths["primary_metadata"], "primary"
    )
    independent_metadata = BALANCED.validate_metadata(
        paths["independent_metadata"], "independent"
    )
    if primary_metadata.get("seed_id") == independent_metadata.get("seed_id"):
        raise ValueError("primary and independent seed identities match")
    counts = {
        "primary": tck_count(paths["primary_tracks"]),
        "independent": tck_count(paths["independent_tracks"]),
    }
    if set(counts.values()) != {EXPECTED_STREAMLINES}:
        raise ValueError(f"tractogram counts differ: {counts}")
    return {
        "ready": True,
        "missing": [],
        "paths": {name: str(path) for name, path in paths.items()},
        "counts": counts,
        "primary_seed_id": primary_metadata["seed_id"],
        "independent_seed_id": independent_metadata["seed_id"],
    }


def environment() -> dict[str, str]:
    value = os.environ.copy()
    value["PATH"] = (
        f"{MRTRIX}:{ATLAS.BASE.FSL}:"
        f"{value.get('PATH', '')}"
    )
    return value


def run_connectome(
    *,
    tracks: Path,
    nodes: Path,
    output: Path,
    assignments: Path,
    log: Path,
    threads: int,
) -> None:
    # MRtrix chooses the text delimiter from the final filename suffix.
    # Keep ".csv" as the terminal suffix while retaining an unmistakably
    # non-final hidden name; "count.csv.partial" would be whitespace-delimited.
    output_partial = output.with_name(
        f".{output.stem}.partial{output.suffix}"
    )
    assignments_partial = assignments.with_name(
        f".{assignments.stem}.partial{assignments.suffix}"
    )
    command = [
        str(MRTRIX / "tck2connectome"),
        str(tracks),
        str(nodes),
        str(output_partial),
        "-assignment_radial_search",
        "4",
        "-symmetric",
        "-zero_diagonal",
        "-stat_edge",
        "sum",
        "-out_assignments",
        str(assignments_partial),
        "-nthreads",
        str(threads),
        "-force",
    ]
    started = time.monotonic()
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
        raise RuntimeError(f"tck2connectome failed: {log}")
    for path in (output_partial, assignments_partial):
        if not path.is_file() or path.stat().st_size <= 0:
            raise RuntimeError(f"tck2connectome output missing: {path}")
    os.replace(output_partial, output)
    os.replace(assignments_partial, assignments)


def support_complementarity(
    primary: np.ndarray, independent: np.ndarray
) -> dict[str, Any]:
    tri = np.triu_indices(EXPECTED_NODES, 1)
    first = primary[tri] > 0
    second = independent[tri] > 0
    union = first | second
    overlap = first & second
    first_only = first & ~second
    second_only = second & ~first
    union_n = int(np.count_nonzero(union))
    return {
        "primary_supported_edges": int(np.count_nonzero(first)),
        "independent_supported_edges": int(np.count_nonzero(second)),
        "overlap_supported_edges": int(np.count_nonzero(overlap)),
        "primary_only_supported_edges": int(np.count_nonzero(first_only)),
        "independent_only_supported_edges": int(np.count_nonzero(second_only)),
        "union_supported_edges": union_n,
        "support_jaccard": (
            int(np.count_nonzero(overlap)) / union_n
            if union_n
            else 0.0
        ),
    }


def original_density(path: Path) -> float | None:
    if not path.is_file() or path.is_symlink():
        return None
    return float(BALANCED.matrix_qc(BALANCED.load_count(path))["edge_density"])


def execute(unit: str, threads: int) -> dict[str, Any]:
    allowed = sorted(ATLAS.BASE.BASE.load_units())
    if unit not in allowed:
        raise ValueError(f"unit is not in the 15-unit Phase-B canary: {unit}")
    candidate = ATLAS.candidate_record(unit)
    pair = validate_phase_pair(unit)
    if not pair["ready"]:
        raise RuntimeError(f"exact Phase-B pair is incomplete: {pair['missing']}")

    attempt_id = (
        f"{attempt_stamp()}-hroi-balanced-density-{unit}"
    )
    attempt = ATTEMPTS / attempt_id
    attempt.mkdir(parents=True, exist_ok=False)
    preflight = {
        "schema_version": "1.0.0",
        "record_type": (
            "diagnosis_blind_hcp379_hroi_balanced_density_canary_preflight"
        ),
        "status": "EXECUTION_INPUTS_BOUND",
        "generated_utc": utc_now(),
        "attempt_id": attempt_id,
        "unit": unit,
        "diagnosis_labels_used": False,
        "tractography_generated": False,
        "pair": pair,
        "candidate": candidate,
        "design": {
            "balanced_total_streamline_counts": list(BALANCED_TOTALS),
            "per_seed_prefix_by_total": {
                str(total): prefix
                for total, prefix in BALANCED_TOTALS.items()
            },
            "density_target": DENSITY_TARGET,
            "assignment_radial_search_mm": 4,
            "cohort_recipe_selected": False,
            "scale_up_authorized": False,
        },
        "implementation": BALANCED.file_record(Path(__file__)),
        "atlas_implementation": BALANCED.file_record(ATLAS_SOURCE),
        "balanced_implementation": BALANCED.file_record(BALANCED_SOURCE),
        "tck_counter": BALANCED.file_record(TCK_COUNTER),
        "tck2connectome": BALANCED.file_record(MRTRIX / "tck2connectome"),
    }
    atomic_json(attempt / "preflight.json", preflight)

    atlas_attempt = attempt / "atlas"
    atlas_result = ATLAS.build_unit(
        unit,
        candidate=candidate,
        expected_t1=ATLAS.BASE.BASE.execution_t1_by_unit()[unit],
        route=ATLAS.registration_route(unit),
        attempt=atlas_attempt,
    )
    if atlas_result.get("status") != "PASS_HROI_CANARY_ATLAS":
        raise RuntimeError(
            f"corrected HROI atlas failed: {atlas_result.get('failure')}"
        )
    nodes = Path(
        str(
            atlas_result["artifacts"][
                "hcp379_nodes_b0_1mm"
            ]["path"]
        )
    ).resolve()
    paths = phase_paths(unit)
    assignment_arrays: dict[str, np.ndarray] = {}
    seed_records: dict[str, Any] = {}
    for seed in ("primary", "independent"):
        seed_root = attempt / seed
        seed_root.mkdir(parents=True, exist_ok=False)
        count_path = seed_root / "hroi_count_10m.csv"
        assignments_path = seed_root / "hroi_assignments_10m.csv"
        run_connectome(
            tracks=paths[f"{seed}_tracks"],
            nodes=nodes,
            output=count_path,
            assignments=assignments_path,
            log=seed_root / "tck2connectome.log",
            threads=threads,
        )
        assignments = BALANCED.load_assignments(assignments_path)
        reconstructed = BALANCED.count_matrix(
            assignments, EXPECTED_STREAMLINES
        )
        recorded = BALANCED.load_count(count_path)
        if not np.array_equal(recorded, reconstructed):
            raise ValueError(f"{seed}: corrected assignment replay differs")
        assignment_arrays[seed] = assignments
        corrected_qc = BALANCED.matrix_qc(recorded)
        seed_records[seed] = {
            "tracks": BALANCED.file_record(paths[f"{seed}_tracks"]),
            "metadata": BALANCED.file_record(paths[f"{seed}_metadata"]),
            "hroi_count_10m": BALANCED.file_record(count_path),
            "hroi_assignments_10m": BALANCED.file_record(assignments_path),
            "assignment_replay_exact": True,
            "corrected_hroi_qc": corrected_qc,
            "original_atlas_density": original_density(
                paths[f"{seed}_original_count"]
            ),
            "endpoint_assignment_fraction": (
                BALANCED.assignment_fraction(
                    assignments, EXPECTED_STREAMLINES
                )
            ),
        }

    levels: dict[str, Any] = {}
    for total, prefix in BALANCED_TOTALS.items():
        primary = BALANCED.count_matrix(
            assignment_arrays["primary"], prefix
        )
        independent = BALANCED.count_matrix(
            assignment_arrays["independent"], prefix
        )
        combined = primary + independent
        matrix_path = (
            attempt / "balanced" / f"balanced_{total // 1_000_000}m_count.csv"
        )
        atomic_matrix(matrix_path, combined)
        qc = BALANCED.matrix_qc(combined)
        levels[str(total)] = {
            "total_streamlines": total,
            "per_seed_prefix_streamlines": prefix,
            "construction": (
                f"first_{prefix}_primary_plus_"
                f"first_{prefix}_independent"
            ),
            "primary_endpoint_assignment_fraction": (
                BALANCED.assignment_fraction(
                    assignment_arrays["primary"], prefix
                )
            ),
            "independent_endpoint_assignment_fraction": (
                BALANCED.assignment_fraction(
                    assignment_arrays["independent"], prefix
                )
            ),
            "support_complementarity": support_complementarity(
                primary, independent
            ),
            "count_matrix": BALANCED.file_record(matrix_path),
            **qc,
            "density_target_0_60_met": (
                float(qc["edge_density"]) >= DENSITY_TARGET
            ),
        }

    qualifying = [
        total
        for total in BALANCED_TOTALS
        if levels[str(total)]["density_target_0_60_met"]
    ]
    summary = {
        "schema_version": "1.0.0",
        "record_type": (
            "diagnosis_blind_hcp379_hroi_balanced_density_canary_summary"
        ),
        "status": "PASS_HROI_BALANCED_DENSITY_CANARY",
        "generated_utc": utc_now(),
        "attempt_id": attempt_id,
        "unit": unit,
        "diagnosis_labels_used": False,
        "non_overwriting": True,
        "source_files_modified": False,
        "tractography_reused_not_regenerated": True,
        "sift2_or_scalar_matrices_generated": False,
        "scale_up_authorized": False,
        "cohort_uniform_recipe_selected": False,
        "density_target": DENSITY_TARGET,
        "density_qualified_balanced_total_counts_for_this_canary": qualifying,
        "smallest_density_qualified_count_for_this_canary": (
            min(qualifying) if qualifying else None
        ),
        "requires_complete_phase_b_canary_family": True,
        "requires_all_nine_matrix_stability": True,
        "requires_uniform_530_subject_construction": True,
        "hroi_atlas_result": BALANCED.file_record(
            atlas_attempt / "subjects" / unit / "result.json"
        ),
        "hroi_nodes": BALANCED.file_record(nodes),
        "seeds": seed_records,
        "levels": levels,
        "preflight": BALANCED.file_record(attempt / "preflight.json"),
        "implementation": BALANCED.file_record(Path(__file__)),
    }
    atomic_json(attempt / "summary.json", summary)
    return {
        "attempt_root": str(attempt),
        "status": summary["status"],
        "unit": unit,
        "smallest_density_qualified_count_for_this_canary": (
            summary["smallest_density_qualified_count_for_this_canary"]
        ),
        "densities": {
            total: levels[total]["edge_density"] for total in levels
        },
    }


def self_test() -> dict[str, Any]:
    first = np.zeros((EXPECTED_NODES, EXPECTED_NODES), dtype=np.int64)
    second = np.zeros_like(first)
    first[0, 1] = first[1, 0] = 2
    first[1, 2] = first[2, 1] = 1
    second[0, 1] = second[1, 0] = 3
    second[2, 3] = second[3, 2] = 1
    value = support_complementarity(first, second)
    checks = {
        "primary_edges": value["primary_supported_edges"] == 2,
        "independent_edges": value["independent_supported_edges"] == 2,
        "overlap_edges": value["overlap_supported_edges"] == 1,
        "primary_only_edges": value["primary_only_supported_edges"] == 1,
        "independent_only_edges": (
            value["independent_only_supported_edges"] == 1
        ),
        "union_edges": value["union_supported_edges"] == 3,
        "jaccard": abs(value["support_jaccard"] - 1 / 3) < 1.0e-12,
        "balanced_totals": list(BALANCED_TOTALS) == [
            6_000_000,
            10_000_000,
            15_000_000,
            20_000_000,
        ],
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unit", default=DEFAULT_UNIT)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.threads <= 16:
        parser.error("--threads must be in 1..16")
    if args.self_test:
        value = self_test()
        print(json.dumps(value, sort_keys=True))
        return 0 if value["status"] == "PASS" else 1

    allowed = sorted(ATLAS.BASE.BASE.load_units())
    if args.unit not in allowed:
        parser.error("--unit must be one of the 15 Phase-B canaries")
    try:
        candidate = ATLAS.candidate_record(args.unit)
        candidate_ready = True
        candidate_error = None
    except FileNotFoundError as exc:
        candidate = None
        candidate_ready = False
        candidate_error = str(exc)
    pair = validate_phase_pair(args.unit)
    preflight = {
        "schema_version": "1.0.0",
        "record_type": (
            "diagnosis_blind_hcp379_hroi_balanced_density_canary_preflight"
        ),
        "status": (
            "READY_HROI_BALANCED_DENSITY_CANARY"
            if candidate_ready and pair["ready"]
            else "WAITING_FOR_HROI_OR_EXACT_10M_PAIR"
        ),
        "generated_utc": utc_now(),
        "unit": args.unit,
        "diagnosis_labels_used": False,
        "candidate_ready": candidate_ready,
        "candidate_error": candidate_error,
        "candidate": candidate,
        "pair": pair,
        "imaging_executed": False,
        "tractography_generated": False,
        "matrix_generation_started": False,
        "scale_up_authorized": False,
        "balanced_total_streamline_counts": list(BALANCED_TOTALS),
    }
    if not args.execute:
        print(json.dumps(preflight, indent=2, sort_keys=True))
        return 0
    if preflight["status"] != "READY_HROI_BALANCED_DENSITY_CANARY":
        print(json.dumps(preflight, indent=2, sort_keys=True))
        return 2
    result = execute(args.unit, args.threads)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
