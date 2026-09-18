#!/usr/bin/env python3
"""Project HCP379 raw density for balanced two-seed 6M/10M/15M/20M ensembles.

This is a diagnosis-blind, count-only calibration.  It does not run
tractography, SIFT2, scalar sampling, biological statistics, or release
assembly.  It reconstructs the exact count matrices from the ordered
endpoint-assignment records of the completed primary and independent 10M
Phase-B runs.  Existing 3M/5M/10M count matrices must replay exactly before
the previously uncomputed balanced 15M level (7.5M from each seed) is used.

The output can identify the smallest balanced total count whose raw density
reaches 0.60 in every completed canary.  It never authorizes scale-up: the
selected construction still requires full nine-matrix generation and
stability validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/home/ec2-user/exp")
DEFAULT_PRETRACT_MANIFEST = Path(
    "/data/derivatives/hcp379_v2/pretract_recovery4/"
    "pretract_recovery4_manifest.json"
)
DEFAULT_PHASE_SUBJECT_ROOT = Path(
    "/data/derivatives/scforge_v2/"
    "h04a_r1_recovery_20260719_retry4/subjects"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/data/derivatives/hcp379_v2/"
    "phase_b_balanced_density_projection_v1/attempts"
)
EXPECTED_NODES = 379
EXPECTED_BASE_STREAMLINES = 10_000_000
DENSITY_TARGET = 0.60
PER_SEED_PREFIXES = (3_000_000, 5_000_000, 7_500_000, 10_000_000)
BALANCED_TOTALS = {
    6_000_000: 3_000_000,
    10_000_000: 5_000_000,
    15_000_000: 7_500_000,
    20_000_000: 10_000_000,
}
BASE_MATRIX_PREFIXES = {
    3_000_000: "3m",
    5_000_000: "5m",
    10_000_000: "10m",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def attempt_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    path = path.resolve()
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root is not an object: {path}")
    return value


def load_units(path: Path) -> list[str]:
    manifest = load_json(path)
    if (
        manifest.get("record_type")
        != "diagnosis_blind_hcp379_pretract_recovery4_manifest"
        or manifest.get("diagnosis_labels_used") is not False
        or manifest.get("unit_count") != 15
    ):
        raise ValueError("Recovery4 pretract manifest contract differs")
    units = [str(row.get("unit", "")) for row in manifest.get("units", [])]
    if len(units) != 15 or len(set(units)) != 15 or any(not unit for unit in units):
        raise ValueError("Recovery4 Phase-B unit set differs")
    return sorted(units)


def unit_inputs(subject_root: Path, unit: str) -> dict[str, Path]:
    root = subject_root / unit / "09_hcp379_stability"
    inputs: dict[str, Path] = {}
    for seed in ("primary", "independent"):
        inputs[f"{seed}_assignments_10m"] = (
            root / f"{seed}_10m" / "assignments.csv"
        )
        inputs[f"{seed}_metadata_10m"] = (
            root / f"{seed}_10m" / "tractography_parameters.json"
        )
        for prefix, suffix in BASE_MATRIX_PREFIXES.items():
            inputs[f"{seed}_count_{prefix}"] = (
                root / f"{seed}_{suffix}" / "matrices" / "count.csv"
            )
    return inputs


def input_readiness(subject_root: Path, unit: str) -> dict[str, Any]:
    paths = unit_inputs(subject_root, unit)
    missing = [
        name
        for name, path in paths.items()
        if not path.is_file() or path.is_symlink()
    ]
    return {
        "unit": unit,
        "ready": not missing,
        "missing": missing,
    }


def load_assignments(path: Path) -> np.ndarray:
    values = np.loadtxt(path, comments="#", dtype=np.int16)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.shape != (EXPECTED_BASE_STREAMLINES, 2):
        raise ValueError(
            f"{path}: assignment shape {values.shape} differs from "
            f"({EXPECTED_BASE_STREAMLINES}, 2)"
        )
    if np.any(values < 0) or np.any(values > EXPECTED_NODES):
        raise ValueError(f"{path}: assignment endpoint lies outside 0..379")
    return values


def count_matrix(assignments: np.ndarray, prefix: int) -> np.ndarray:
    if prefix <= 0 or prefix > assignments.shape[0]:
        raise ValueError(f"invalid prefix {prefix}")
    selected = assignments[:prefix]
    left = selected[:, 0].astype(np.int32, copy=False)
    right = selected[:, 1].astype(np.int32, copy=False)
    valid = (left > 0) & (right > 0) & (left != right)
    left = left[valid] - 1
    right = right[valid] - 1
    low = np.minimum(left, right)
    high = np.maximum(left, right)
    linear = low * EXPECTED_NODES + high
    upper = np.bincount(
        linear, minlength=EXPECTED_NODES * EXPECTED_NODES
    ).reshape(EXPECTED_NODES, EXPECTED_NODES)
    matrix = upper + upper.T
    np.fill_diagonal(matrix, 0)
    return matrix.astype(np.int64, copy=False)


def assignment_fraction(assignments: np.ndarray, prefix: int) -> float:
    selected = assignments[:prefix]
    assigned = (selected[:, 0] > 0) & (selected[:, 1] > 0)
    return float(np.mean(assigned))


def load_count(path: Path) -> np.ndarray:
    value = np.loadtxt(path, delimiter=",")
    if value.shape != (EXPECTED_NODES, EXPECTED_NODES):
        raise ValueError(f"{path}: matrix shape differs")
    if (
        not np.isfinite(value).all()
        or np.any(value < 0)
        or not np.allclose(value, value.T, atol=0.0, rtol=0.0)
        or not np.allclose(np.diag(value), 0.0, atol=0.0, rtol=0.0)
        or not np.allclose(value, np.rint(value), atol=1.0e-6, rtol=0.0)
    ):
        raise ValueError(f"{path}: count-matrix invariants differ")
    return np.rint(value).astype(np.int64)


def validate_metadata(path: Path, seed: str) -> dict[str, Any]:
    value = load_json(path)
    if (
        value.get("record_type")
        != "hcp379_stability_tractography_parameters"
        or value.get("diagnosis_labels_used") is not False
        or value.get("run_id") != f"{seed}_10m"
        or value.get("seed_class") != seed
        or int(value.get("actual_streamlines", -1))
        != EXPECTED_BASE_STREAMLINES
    ):
        raise ValueError(f"{path}: 10M tractography metadata differs")
    return value


def matrix_qc(matrix: np.ndarray) -> dict[str, Any]:
    triangle = matrix[np.triu_indices(EXPECTED_NODES, 1)]
    supported = int(np.count_nonzero(triangle > 0))
    possible = EXPECTED_NODES * (EXPECTED_NODES - 1) // 2
    return {
        "supported_edges": supported,
        "possible_edges": possible,
        "edge_density": supported / possible,
        "connected_nodes": int(np.count_nonzero(np.sum(matrix, axis=1) > 0)),
        "total_assigned_nondiagonal_streamlines": int(np.sum(triangle)),
    }


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.chmod(temporary, 0o444)
    temporary.replace(path)


def atomic_matrix(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    np.savetxt(temporary, value, delimiter=",", fmt="%d")
    os.chmod(temporary, 0o444)
    temporary.replace(path)


def process_unit(
    *,
    subject_root: Path,
    unit: str,
    output_root: Path,
) -> dict[str, Any]:
    paths = unit_inputs(subject_root, unit)
    primary_metadata = validate_metadata(paths["primary_metadata_10m"], "primary")
    independent_metadata = validate_metadata(
        paths["independent_metadata_10m"], "independent"
    )
    if primary_metadata.get("seed_id") == independent_metadata.get("seed_id"):
        raise ValueError(f"{unit}: primary and independent seed ids match")

    primary = load_assignments(paths["primary_assignments_10m"])
    independent = load_assignments(paths["independent_assignments_10m"])
    prefix_matrices: dict[str, dict[int, np.ndarray]] = {
        "primary": {},
        "independent": {},
    }
    replay: dict[str, Any] = {}
    for seed, assignments in (
        ("primary", primary),
        ("independent", independent),
    ):
        for prefix in PER_SEED_PREFIXES:
            prefix_matrices[seed][prefix] = count_matrix(assignments, prefix)
        for prefix, suffix in BASE_MATRIX_PREFIXES.items():
            recorded = load_count(paths[f"{seed}_count_{prefix}"])
            reconstructed = prefix_matrices[seed][prefix]
            exact = np.array_equal(recorded, reconstructed)
            replay[f"{seed}_{suffix}"] = {
                "exact": exact,
                "maximum_absolute_difference": int(
                    np.max(np.abs(recorded - reconstructed))
                ),
            }
            if not exact:
                raise ValueError(
                    f"{unit}:{seed}_{suffix}: assignment replay differs"
                )

    levels: dict[str, Any] = {}
    for total, prefix in BALANCED_TOTALS.items():
        combined = (
            prefix_matrices["primary"][prefix]
            + prefix_matrices["independent"][prefix]
        )
        matrix_path = (
            output_root / unit / f"balanced_{total // 1_000_000}m_count.csv"
        )
        atomic_matrix(matrix_path, combined)
        qc = matrix_qc(combined)
        levels[str(total)] = {
            "total_streamlines": total,
            "per_seed_prefix_streamlines": prefix,
            "construction": (
                f"first_{prefix}_primary_plus_first_{prefix}_independent"
            ),
            "primary_endpoint_assignment_fraction": assignment_fraction(
                primary, prefix
            ),
            "independent_endpoint_assignment_fraction": assignment_fraction(
                independent, prefix
            ),
            "count_matrix": file_record(matrix_path),
            **qc,
            "density_target_0_60_met": qc["edge_density"] >= DENSITY_TARGET,
        }

    unit_record = {
        "schema_version": "1.0.0",
        "record_type": (
            "diagnosis_blind_hcp379_balanced_seed_density_projection_unit"
        ),
        "generated_utc": utc_now(),
        "status": "PASS_DENSITY_PROJECTION",
        "diagnosis_labels_used": False,
        "scale_up_authorized": False,
        "unit": unit,
        "input_evidence": {
            name: file_record(path) for name, path in sorted(paths.items())
        },
        "base_assignment_replay": replay,
        "levels": levels,
    }
    unit_path = output_root / unit / "projection.json"
    atomic_json(unit_path, unit_record)
    return {
        "unit": unit,
        "status": unit_record["status"],
        "record": file_record(unit_path),
        "levels": {
            total: {
                "edge_density": row["edge_density"],
                "density_target_0_60_met": row[
                    "density_target_0_60_met"
                ],
            }
            for total, row in levels.items()
        },
    }


def self_test() -> dict[str, Any]:
    assignments = np.zeros((12, 2), dtype=np.int16)
    assignments[:8] = np.asarray(
        [
            (1, 2),
            (2, 1),
            (1, 1),
            (0, 2),
            (2, 3),
            (3, 2),
            (379, 1),
            (1, 379),
        ],
        dtype=np.int16,
    )
    matrix = count_matrix(assignments, 8)
    checks = {
        "symmetric": np.array_equal(matrix, matrix.T),
        "zero_diagonal": bool(np.all(np.diag(matrix) == 0)),
        "pair_1_2_count_two": int(matrix[0, 1]) == 2,
        "pair_2_3_count_two": int(matrix[1, 2]) == 2,
        "pair_1_379_count_two": int(matrix[0, 378]) == 2,
        "assigned_fraction_includes_same_node": abs(
            assignment_fraction(assignments, 8) - 7 / 8
        )
        < 1.0e-12,
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pretract-manifest",
        type=Path,
        default=DEFAULT_PRETRACT_MANIFEST,
    )
    parser.add_argument(
        "--phase-subject-root",
        type=Path,
        default=DEFAULT_PHASE_SUBJECT_ROOT,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        result = self_test()
        print(json.dumps(result, sort_keys=True))
        return 0 if result["status"] == "PASS" else 1

    pretract_manifest = args.pretract_manifest.resolve()
    phase_subject_root = args.phase_subject_root.resolve()
    units = load_units(pretract_manifest)
    readiness = [
        input_readiness(phase_subject_root, unit) for unit in units
    ]
    ready_units = [row["unit"] for row in readiness if row["ready"]]
    waiting_units = [row["unit"] for row in readiness if not row["ready"]]
    preflight = {
        "schema_version": "1.0.0",
        "record_type": (
            "diagnosis_blind_hcp379_balanced_seed_density_projection_preflight"
        ),
        "generated_utc": utc_now(),
        "status": (
            "READY"
            if len(ready_units) == len(units)
            else "WAITING_FOR_BASE_PHASE_B"
        ),
        "diagnosis_labels_used": False,
        "scale_up_authorized": False,
        "unit_count": len(units),
        "ready_unit_count": len(ready_units),
        "waiting_unit_count": len(waiting_units),
        "ready_units": ready_units,
        "waiting_units": waiting_units,
        "readiness": readiness,
        "design": {
            "balanced_total_streamline_counts": list(BALANCED_TOTALS),
            "per_seed_prefix_by_total": {
                str(total): prefix
                for total, prefix in BALANCED_TOTALS.items()
            },
            "density_target": DENSITY_TARGET,
            "base_3m_5m_10m_assignment_replay_required": True,
            "new_tractography_generated": False,
            "sift2_or_scalar_matrix_generation": False,
            "full_nine_matrix_validation_required_before_scale_up": True,
            "uniform_count_required_across_all_530": True,
            "per_subject_tuning_allowed": False,
        },
        "pretract_manifest": file_record(pretract_manifest),
        "runner_source": file_record(Path(__file__)),
    }
    if not args.execute:
        print(json.dumps(preflight, sort_keys=True))
        return 0
    if waiting_units and not args.allow_partial:
        print(json.dumps(preflight, sort_keys=True))
        return 2
    if not ready_units:
        print(json.dumps(preflight, sort_keys=True))
        return 2

    attempt_root = (
        args.output_root.resolve()
        / f"{attempt_stamp()}-balanced-seed-density-projection"
    )
    attempt_root.mkdir(parents=True, exist_ok=False)
    atomic_json(attempt_root / "preflight.json", preflight)
    unit_results = [
        process_unit(
            subject_root=phase_subject_root,
            unit=unit,
            output_root=attempt_root,
        )
        for unit in ready_units
    ]
    density_qualified_totals = [
        total
        for total in BALANCED_TOTALS
        if len(unit_results) == len(units)
        and all(
            row["levels"][str(total)]["density_target_0_60_met"]
            for row in unit_results
        )
    ]
    selected = (
        min(density_qualified_totals)
        if density_qualified_totals
        else None
    )
    summary = {
        "schema_version": "1.0.0",
        "record_type": (
            "diagnosis_blind_hcp379_balanced_seed_density_projection_summary"
        ),
        "generated_utc": utc_now(),
        "status": (
            "COMPLETE_DENSITY_PROJECTION"
            if len(unit_results) == len(units)
            else "PARTIAL_DENSITY_PROJECTION"
        ),
        "diagnosis_labels_used": False,
        "scale_up_authorized": False,
        "projection_only": True,
        "unit_count_expected": len(units),
        "unit_count_processed": len(unit_results),
        "density_target": DENSITY_TARGET,
        "density_qualified_balanced_total_counts": (
            density_qualified_totals
        ),
        "smallest_density_qualified_balanced_total_count": selected,
        "full_nine_matrix_validation_required_before_scale_up": True,
        "uniform_count_required_across_all_530": True,
        "per_subject_tuning_allowed": False,
        "units": unit_results,
        "preflight": file_record(attempt_root / "preflight.json"),
        "runner_source": file_record(Path(__file__)),
    }
    summary_path = attempt_root / "summary.json"
    atomic_json(summary_path, summary)
    print(
        json.dumps(
            {
                "attempt_root": str(attempt_root),
                "status": summary["status"],
                "unit_count_processed": len(unit_results),
                "smallest_density_qualified_balanced_total_count": selected,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
