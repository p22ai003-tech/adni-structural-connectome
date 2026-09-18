#!/usr/bin/env python3
"""Generate one independent 10M superset for the difficult 135 canary.

Balanced 3M, 5M and 6M constructions already fail the fixed 0.60 density
gate on this diagnosis-blind adverse case.  This non-overwriting calibration
therefore generates one isolated independent-seed 10M tractogram and uses
nested two-seed prefixes to evaluate exact 10M, 15M and 20M totals.

It generates count matrices only.  It does not run SIFT2/scalar matrices,
select a cohort recipe, authorize scale-up, or modify the official Phase-B
DAG.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


EXP = Path("/home/ec2-user/exp")
BASE_SOURCE = (
    EXP
    / "scripts/hcp/"
    "run_hcp379_hroi_low_density_135_6m_canary_v1.py"
)
PRIOR_SUMMARY = Path(
    "/data/derivatives/hcp379_v2/"
    "phase_b_hroi_low_density_135_6m_canary_v1/attempts/"
    "20260728T075047.751486Z-low-density-135-balanced-6m/"
    "summary.json"
)
PRIOR_VALIDATION = (
    EXP
    / "research_audit/outputs/"
    "hcp379_hroi_low_density_135_6m_canary_v1/validation.json"
)
ATTEMPTS = Path(
    "/data/derivatives/hcp379_v2/"
    "phase_b_hroi_low_density_135_20m_superset_v1/attempts"
)
INDEPENDENT_STREAMLINES = 10_000_000
LEVELS = {
    10_000_000: 5_000_000,
    15_000_000: 7_500_000,
    20_000_000: 10_000_000,
}
DENSITY_TARGET = 0.60


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = load_module(BASE_SOURCE, "hcp379_low_density_135_6m_base")
BALANCED = BASE.BALANCED


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def bind_inputs() -> dict[str, Any]:
    bound = BASE.inputs()
    prior = load_json(PRIOR_SUMMARY)
    validation = load_json(PRIOR_VALIDATION)
    expected_prior = {
        "3000000": 0.43133559492398543,
        "5000000": 0.4823889098295431,
        "6000000": 0.5018776786586813,
    }
    projected = validation.get(
        "runtime_balanced_density_projection",
        {},
    )
    if (
        prior.get("status")
        != "PASS_LOW_DENSITY_135_6M_CALIBRATION"
        or prior.get("unit") != BASE.UNIT
        or prior.get("diagnosis_labels_used") is not False
        or prior.get("density_target_0_60_met") is not False
        or validation.get("status") != "PASS"
        or validation.get("runtime_summary_detected") is not True
        or validation.get("passed_checks") != 10
        or validation.get("total_checks") != 10
        or set(projected) != set(expected_prior)
        or any(
            abs(
                float(projected[level]["edge_density"])
                - expected
            )
            > 1.0e-15
            or projected[level]["density_target_0_60_met"] is not False
            for level, expected in expected_prior.items()
        )
    ):
        raise ValueError("lower-count adverse calibration differs")
    return {
        **bound,
        "prior_summary": prior,
        "prior_validation": validation,
        "prior_densities": expected_prior,
    }


def preflight() -> dict[str, Any]:
    bound = bind_inputs()
    return {
        "schema_version": "1.0.0",
        "record_type": (
            "diagnosis_blind_hcp379_hroi_low_density_135_"
            "20m_superset_preflight"
        ),
        "generated_utc": utc_now(),
        "status": "READY_LOW_DENSITY_135_10M_SUPERSET",
        "unit": BASE.UNIT,
        "diagnosis_labels_used": False,
        "lower_total_counts_ruled_out": [
            3_000_000,
            5_000_000,
            6_000_000,
        ],
        "lower_count_densities": bound["prior_densities"],
        "independent_streamlines_to_generate": (
            INDEPENDENT_STREAMLINES
        ),
        "balanced_total_counts_to_project": sorted(LEVELS),
        "imaging_executed": False,
        "official_phase_b_modified": False,
        "cohort_recipe_selected": False,
        "scale_up_authorized": False,
        "selected_count_tractography_authorized": False,
        "requires_complete_15_canary_family": True,
        "requires_joint_all_nine_matrix_stability": True,
    }


def execute(threads: int) -> dict[str, Any]:
    bound = bind_inputs()
    attempt = ATTEMPTS / f"{stamp()}-low-density-135-20m-superset"
    attempt.mkdir(parents=True, exist_ok=False)
    bound_preflight = {
        **preflight(),
        "status": "EXECUTION_INPUTS_BOUND",
        "imaging_executed": True,
        "inputs": {
            "primary_tracks": BALANCED.file_record(
                BASE.PRIMARY_TRACKS
            ),
            "primary_metadata": BALANCED.file_record(
                BASE.PRIMARY_METADATA
            ),
            "wmfod": BALANCED.file_record(BASE.WMFOD),
            "five_tt": BALANCED.file_record(BASE.FIVE_TT),
            "hroi_nodes": BALANCED.file_record(bound["nodes"]),
            "primary_hroi_assignments": BALANCED.file_record(
                bound["primary_assignments"]
            ),
            "prior_6m_summary": BALANCED.file_record(PRIOR_SUMMARY),
            "prior_6m_validation": BALANCED.file_record(
                PRIOR_VALIDATION
            ),
        },
        "independent_seed_id": bound["independent_seed_id"],
        "independent_rng_seed": bound["independent_rng_seed"],
        "implementation": BALANCED.file_record(Path(__file__)),
    }
    BASE.atomic_json(attempt / "preflight.json", bound_preflight)

    output = attempt / "independent_10m"
    output.mkdir()
    tracks_partial = output / "tracks.partial.tck"
    tracks = output / "tracks.tck"
    metadata = output / "tractography_parameters.json"
    BASE.run_logged(
        [
            str(BASE.MRTRIX / "tckgen"),
            str(BASE.WMFOD),
            str(tracks_partial),
            "-algorithm",
            "iFOD2",
            "-act",
            str(BASE.FIVE_TT),
            "-backtrack",
            "-crop_at_gmwmi",
            "-seed_dynamic",
            str(BASE.WMFOD),
            "-select",
            str(INDEPENDENT_STREAMLINES),
            "-seeds",
            str(BASE.MAXIMUM_SEEDS),
            "-minlength",
            "10.0",
            "-maxlength",
            "250.0",
            "-cutoff",
            "0.06",
            "-step",
            f"{bound['step_size_mm']:.8f}",
            "-angle",
            "45.0",
            "-nthreads",
            str(threads),
        ],
        log=output / "tckgen.log",
        env=BASE.environment(bound["independent_rng_seed"]),
        outputs=(tracks_partial,),
    )
    if BASE.tck_count(tracks_partial) != INDEPENDENT_STREAMLINES:
        raise ValueError("independent 10M tractogram count differs")
    os.replace(tracks_partial, tracks)
    BASE.atomic_json(
        metadata,
        {
            "schema_version": "1.0.0",
            "record_type": (
                "hcp379_stability_tractography_parameters"
            ),
            "diagnosis_labels_used": False,
            "unit": BASE.UNIT,
            "run_id": (
                "independent_10m_low_density_superset_calibration"
            ),
            "seed_class": "independent",
            "seed_id": bound["independent_seed_id"],
            "rng_seed": bound["independent_rng_seed"],
            "construction": "isolated_adverse_case_superset",
            "algorithm": "iFOD2",
            "act": True,
            "backtrack": True,
            "crop_at_gmwmi": True,
            "seeding": "seed_dynamic",
            "actual_step_size_mm": bound["step_size_mm"],
            "requested_streamlines": INDEPENDENT_STREAMLINES,
            "actual_streamlines": INDEPENDENT_STREAMLINES,
        },
    )

    count_partial = output / "count_hroi.partial.csv"
    assignments_partial = output / "assignments_hroi.partial.csv"
    count_path = output / "count_hroi.csv"
    assignments_path = output / "assignments_hroi.csv"
    BASE.run_logged(
        [
            str(BASE.MRTRIX / "tck2connectome"),
            str(tracks),
            str(bound["nodes"]),
            str(count_partial),
            "-assignment_radial_search",
            "4",
            "-symmetric",
            "-zero_diagonal",
            "-stat_edge",
            "sum",
            "-out_assignments",
            str(assignments_partial),
            "-nthreads",
            str(min(threads, 8)),
            "-force",
        ],
        log=output / "tck2connectome.log",
        env=BASE.environment(bound["independent_rng_seed"]),
        outputs=(count_partial, assignments_partial),
    )
    os.replace(count_partial, count_path)
    os.replace(assignments_partial, assignments_path)

    primary = BALANCED.load_assignments(
        bound["primary_assignments"]
    )
    independent = np.loadtxt(
        assignments_path,
        comments="#",
        dtype=np.int16,
    )
    if independent.shape != (INDEPENDENT_STREAMLINES, 2):
        raise ValueError("independent 10M assignment shape differs")
    independent_full = BALANCED.count_matrix(
        independent,
        INDEPENDENT_STREAMLINES,
    )
    recorded_independent = BALANCED.load_count(count_path)
    if not np.array_equal(independent_full, recorded_independent):
        raise ValueError("independent 10M assignment replay differs")

    level_rows: dict[str, Any] = {}
    for total, prefix in LEVELS.items():
        primary_matrix = BALANCED.count_matrix(primary, prefix)
        independent_matrix = BALANCED.count_matrix(
            independent,
            prefix,
        )
        combined = primary_matrix + independent_matrix
        matrix_path = attempt / f"balanced_{total // 1_000_000}m_count.csv"
        BALANCED.atomic_matrix(matrix_path, combined)
        qc = BALANCED.matrix_qc(combined)
        level_rows[str(total)] = {
            "construction": (
                f"first_{prefix}_primary_plus_"
                f"first_{prefix}_independent"
            ),
            "per_seed_prefix_streamlines": prefix,
            "primary_endpoint_assignment_fraction": (
                BALANCED.assignment_fraction(primary, prefix)
            ),
            "independent_endpoint_assignment_fraction": (
                BALANCED.assignment_fraction(independent, prefix)
            ),
            "support_complementarity": BASE.support(
                primary_matrix,
                independent_matrix,
            ),
            "count_matrix": BALANCED.file_record(matrix_path),
            **qc,
            "density_target_0_60_met": (
                float(qc["edge_density"]) >= DENSITY_TARGET
            ),
        }
    qualified = [
        int(total)
        for total, row in level_rows.items()
        if row["density_target_0_60_met"]
    ]
    summary = {
        "schema_version": "1.0.0",
        "record_type": (
            "diagnosis_blind_hcp379_hroi_low_density_135_"
            "20m_superset_summary"
        ),
        "generated_utc": utc_now(),
        "status": "PASS_LOW_DENSITY_135_20M_SUPERSET_CALIBRATION",
        "unit": BASE.UNIT,
        "diagnosis_labels_used": False,
        "non_overwriting": True,
        "source_files_modified": False,
        "official_phase_b_modified": False,
        "primary_tractography_reused": True,
        "independent_tractography_generated": True,
        "generated_independent_streamlines": (
            INDEPENDENT_STREAMLINES
        ),
        "levels": level_rows,
        "density_qualified_balanced_total_counts_for_this_canary": (
            qualified
        ),
        "smallest_density_qualified_count_for_this_canary": (
            min(qualified) if qualified else None
        ),
        "cohort_recipe_selected": False,
        "scale_up_authorized": False,
        "selected_count_tractography_authorized": False,
        "requires_complete_15_canary_family": True,
        "requires_joint_all_nine_matrix_stability": True,
        "artifacts": {
            "independent_tracks": BALANCED.file_record(tracks),
            "independent_metadata": BALANCED.file_record(metadata),
            "independent_hroi_assignments": BALANCED.file_record(
                assignments_path
            ),
            "independent_hroi_count": BALANCED.file_record(count_path),
            "preflight": BALANCED.file_record(
                attempt / "preflight.json"
            ),
        },
        "implementation": BALANCED.file_record(Path(__file__)),
    }
    BASE.atomic_json(attempt / "summary.json", summary)
    return {
        "attempt": str(attempt),
        "status": summary["status"],
        "densities": {
            total: row["edge_density"]
            for total, row in level_rows.items()
        },
        "smallest_density_qualified_count_for_this_canary": (
            summary["smallest_density_qualified_count_for_this_canary"]
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.threads <= 8:
        parser.error("--threads must be in 1..8")
    value = execute(args.threads) if args.execute else preflight()
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
