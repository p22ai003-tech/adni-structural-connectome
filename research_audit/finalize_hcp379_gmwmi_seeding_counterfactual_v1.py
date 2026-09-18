#!/usr/bin/env python3
"""Finalize the completed HCP379 GMWMI counterfactual without rerunning imaging.

The original bounded runner generated all four 3M tractograms, assignments,
and count matrices, then stopped because it reused a loader whose shape
contract is fixed at 10M assignments.  This recovery finalizer reopens and
replays the already-generated 3M outputs sequentially, verifies their bound
inputs and metadata, builds the two balanced 6M matrices, and writes the
originally intended summary.  It never invokes tckgen or tck2connectome.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np


EXP = Path("/home/ec2-user/exp")
RUNNER_PATH = (
    EXP / "scripts/hcp/run_hcp379_gmwmi_seeding_counterfactual_v1.py"
)


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


RUNNER = load_module(RUNNER_PATH, "hcp379_gmwmi_recovery_runner")
BALANCED = RUNNER.BALANCED
DENSITY = RUNNER.DENSITY


def load_assignments(path: Path) -> np.ndarray:
    values = np.loadtxt(path, comments="#", dtype=np.int16)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if (
        values.shape != (RUNNER.PER_SEED_STREAMLINES, 2)
        or np.any(values < 0)
        or np.any(values > RUNNER.EXPECTED_NODES)
    ):
        raise ValueError(
            f"{path}: assignment shape or endpoint range differs"
        )
    return values


def verify_record(record: Mapping[str, Any], expected: Path) -> None:
    observed = BALANCED.file_record(expected.resolve())
    if dict(record) != observed:
        raise ValueError(f"bound file record differs: {expected}")


def verify_preflight(
    attempt: Path,
    bound: Mapping[str, Mapping[str, Any]],
) -> tuple[Path, dict[str, Any]]:
    preflight_path = attempt / "preflight.json"
    preflight = RUNNER.load_json(preflight_path)
    if (
        preflight.get("record_type")
        != "diagnosis_blind_hcp379_gmwmi_seeding_counterfactual_preflight"
        or preflight.get("status") != "EXECUTION_INPUTS_BOUND"
        or preflight.get("diagnosis_labels_used") is not False
        or preflight.get("outcomes_used") is not False
        or preflight.get("non_overwriting") is not True
        or set(preflight.get("units", [])) != set(RUNNER.UNITS)
    ):
        raise ValueError("execution preflight contract differs")
    verify_record(preflight["implementation"], RUNNER_PATH)
    for unit in RUNNER.UNITS:
        row = preflight["inputs"][unit]
        data = bound[unit]
        expected = {
            "density_summary": data["summary_path"],
            "promoted_pretract_manifest": data["manifest_path"],
            "wmfod": data["wmfod"],
            "five_tt": data["five_tt"],
            "gmwmi": data["gmwmi"],
            "hroi_nodes": data["nodes"],
        }
        for key, path in expected.items():
            verify_record(row[key], Path(path))
        if row.get("seeds") != data["seeds"]:
            raise ValueError(f"{unit}: bound seed metadata differs")
    return preflight_path, preflight


def replay_seed(
    attempt: Path,
    unit: str,
    seed_class: str,
    bound: Mapping[str, Any],
) -> dict[str, Any]:
    output = attempt / unit / seed_class
    tracks = output / "tracks.tck"
    metadata_path = output / "tractography_parameters.json"
    count_path = output / "count_hroi.csv"
    assignments_path = output / "assignments_hroi.csv"
    for path in (tracks, metadata_path, count_path, assignments_path):
        if not path.is_file() or path.is_symlink() or path.stat().st_size <= 0:
            raise ValueError(f"completed regular output absent: {path}")
    if RUNNER.tck_count(tracks) != RUNNER.PER_SEED_STREAMLINES:
        raise ValueError(f"{unit}/{seed_class}: tractogram count differs")
    metadata = RUNNER.load_json(metadata_path)
    expected_seed = bound["seeds"][seed_class]
    if (
        metadata.get("record_type")
        != "hcp379_gmwmi_counterfactual_parameters"
        or metadata.get("unit") != unit
        or metadata.get("role") != RUNNER.ROLE[unit]
        or metadata.get("seed_class") != seed_class
        or metadata.get("seeding") != "seed_gmwmi"
        or metadata.get("algorithm") != "iFOD2"
        or metadata.get("act") is not True
        or metadata.get("backtrack") is not True
        or metadata.get("crop_at_gmwmi") is not True
        or metadata.get("diagnosis_labels_used") is not False
        or int(metadata.get("actual_streamlines", -1))
        != RUNNER.PER_SEED_STREAMLINES
        or int(metadata.get("rng_seed", -1))
        != int(expected_seed["counterfactual_rng_seed"])
        or metadata.get("seed_id")
        != expected_seed["counterfactual_seed_id"]
    ):
        raise ValueError(f"{unit}/{seed_class}: metadata differs")
    assignments = load_assignments(assignments_path)
    replay = BALANCED.count_matrix(
        assignments,
        RUNNER.PER_SEED_STREAMLINES,
    )
    recorded = BALANCED.load_count(count_path)
    if not np.array_equal(replay, recorded):
        raise ValueError(f"{unit}/{seed_class}: assignment replay differs")
    return {
        "unit": unit,
        "seed_class": seed_class,
        "tracks": BALANCED.file_record(tracks),
        "metadata": BALANCED.file_record(metadata_path),
        "count_matrix": BALANCED.file_record(count_path),
        "assignments": BALANCED.file_record(assignments_path),
        "endpoint_assignment_fraction": BALANCED.assignment_fraction(
            assignments,
            RUNNER.PER_SEED_STREAMLINES,
        ),
        "technical_qc": BALANCED.matrix_qc(recorded),
    }


def build_summary(
    attempt: Path,
    bound: Mapping[str, Mapping[str, Any]],
    preflight_path: Path,
    preflight: Mapping[str, Any],
    results: Mapping[tuple[str, str], Mapping[str, Any]],
) -> dict[str, Any]:
    units: dict[str, Any] = {}
    for unit in RUNNER.UNITS:
        matrices = {
            seed: BALANCED.load_count(
                Path(results[(unit, seed)]["count_matrix"]["path"])
            )
            for seed in RUNNER.SEED_CLASSES
        }
        combined = matrices["primary"] + matrices["independent"]
        combined_path = attempt / unit / "balanced_6m_count.csv"
        if combined_path.exists():
            existing = BALANCED.load_count(combined_path)
            if not np.array_equal(existing, combined):
                raise ValueError(f"{unit}: existing balanced matrix differs")
        else:
            DENSITY.atomic_matrix(combined_path, combined)
        qc = BALANCED.matrix_qc(combined)
        baseline = bound[unit]["baseline"]
        baseline_density = float(baseline["edge_density"])
        baseline_assignment = (
            float(baseline["primary_endpoint_assignment_fraction"])
            + float(baseline["independent_endpoint_assignment_fraction"])
        ) / 2.0
        gmwmi_assignment = sum(
            float(results[(unit, seed)]["endpoint_assignment_fraction"])
            for seed in RUNNER.SEED_CLASSES
        ) / 2.0
        units[unit] = {
            "role": RUNNER.ROLE[unit],
            "dynamic_seed_baseline": {
                "edge_density": baseline_density,
                "connected_nodes": int(baseline["connected_nodes"]),
                "possible_edges": int(baseline["possible_edges"]),
                "endpoint_assignment_fraction": baseline_assignment,
                "count_matrix": baseline["count_matrix"],
            },
            "gmwmi_counterfactual": {
                **qc,
                "endpoint_assignment_fraction": gmwmi_assignment,
                "support_complementarity": DENSITY.support_complementarity(
                    matrices["primary"],
                    matrices["independent"],
                ),
                "count_matrix": BALANCED.file_record(combined_path),
                "seeds": {
                    seed: results[(unit, seed)]
                    for seed in RUNNER.SEED_CLASSES
                },
            },
            "difference": {
                "edge_density": float(qc["edge_density"])
                - baseline_density,
                "connected_nodes": int(qc["connected_nodes"])
                - int(baseline["connected_nodes"]),
                "endpoint_assignment_fraction": gmwmi_assignment
                - baseline_assignment,
            },
        }
    return {
        "schema_version": "1.0.0",
        "record_type": (
            "diagnosis_blind_hcp379_gmwmi_seeding_counterfactual"
        ),
        "generated_utc": RUNNER.utc_now(),
        "status": "PASS_COMPLETE_GMWMI_SEEDING_COUNTERFACTUAL",
        "diagnosis_labels_used": False,
        "outcomes_used": False,
        "non_overwriting": True,
        "source_files_modified": False,
        "changed_factor": "seed_dynamic_to_seed_gmwmi_only",
        "new_tractography_generated": True,
        "generated_streamlines": (
            len(RUNNER.UNITS)
            * len(RUNNER.SEED_CLASSES)
            * RUNNER.PER_SEED_STREAMLINES
        ),
        "sift2_or_scalar_matrices_generated": False,
        "units": units,
        "production_recipe_selected": False,
        "scale_up_authorized": False,
        "requires_expansion_to_remaining_low_density_canaries": True,
        "requires_all_nine_and_seed_stability_before_selection": True,
        "preflight": BALANCED.file_record(preflight_path),
        "implementation": dict(preflight["implementation"]),
        "recovery_finalizer": BALANCED.file_record(Path(__file__)),
        "recovery_reason": (
            "Original post-processing reused a 10M-only assignment loader "
            "for complete 3M outputs; no imaging was rerun."
        ),
    }


def finalize(attempt: Path) -> dict[str, Any]:
    attempt = attempt.resolve()
    existing_summary = (
        RUNNER.load_json(attempt / "summary.json")
        if (attempt / "summary.json").exists()
        else {}
    )
    repairable_recovery_summary = bool(
        existing_summary.get("status")
        == "PASS_COMPLETE_GMWMI_SEEDING_COUNTERFACTUAL"
        and existing_summary.get("recovery_reason")
        == (
            "Original post-processing reused a 10M-only assignment loader "
            "for complete 3M outputs; no imaging was rerun."
        )
        and any(
            "possible_edges"
            not in row.get("dynamic_seed_baseline", {})
            for row in existing_summary.get("units", {}).values()
            if isinstance(row, dict)
        )
    )
    if (
        attempt.parent != RUNNER.ATTEMPTS.resolve()
        or not attempt.is_dir()
        or ((attempt / "summary.json").exists() and not repairable_recovery_summary)
    ):
        raise ValueError("exact unfinished GMWMI attempt required")
    bound = {unit: RUNNER.bind_unit(unit) for unit in RUNNER.UNITS}
    preflight_path, preflight = verify_preflight(attempt, bound)
    results = {
        (unit, seed): replay_seed(
            attempt,
            unit,
            seed,
            bound[unit],
        )
        for unit in RUNNER.UNITS
        for seed in RUNNER.SEED_CLASSES
    }
    summary = build_summary(
        attempt,
        bound,
        preflight_path,
        preflight,
        results,
    )
    summary_path = attempt / "summary.json"
    RUNNER.atomic_json(summary_path, summary)
    state_path = attempt / "state.json"
    state = RUNNER.load_json(state_path)
    state.update(
        {
            "status": summary["status"],
            "generated_utc": RUNNER.utc_now(),
            "completed_jobs": [
                f"{unit}/{seed}"
                for unit in RUNNER.UNITS
                for seed in RUNNER.SEED_CLASSES
            ],
            "remaining_jobs": [],
            "summary": BALANCED.file_record(summary_path),
            "recovered_without_imaging_rerun": True,
            "recovery_finalizer": BALANCED.file_record(Path(__file__)),
        }
    )
    RUNNER.atomic_json(state_path, state)
    return {
        "status": summary["status"],
        "attempt": str(attempt),
        "summary": str(summary_path),
        "units": {
            unit: {
                "dynamic_density": row["dynamic_seed_baseline"][
                    "edge_density"
                ],
                "gmwmi_density": row["gmwmi_counterfactual"][
                    "edge_density"
                ],
                "density_difference": row["difference"]["edge_density"],
                "connected_nodes": row["gmwmi_counterfactual"][
                    "connected_nodes"
                ],
            }
            for unit, row in summary["units"].items()
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", type=Path, required=True)
    args = parser.parse_args()
    result = finalize(args.attempt)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
