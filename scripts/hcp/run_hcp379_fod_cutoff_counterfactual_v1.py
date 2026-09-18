#!/usr/bin/env python3
"""Run a bounded HCP379 FOD-cutoff counterfactual.

The experiment changes only the normalised WM-FOD termination cutoff from
0.06 to 0.05.  It uses the exact source random seeds so that each new run is
paired to its corresponding dynamic-seeding baseline.  Atlas, registration,
ACT/5TT, dynamic seeding, backtracking, cropping, algorithm, step size,
curvature, length limits, streamline count, endpoint assignment and matrix
construction remain fixed.

The screen is diagnosis blind, non-overwriting and limited to two independent
3M runs for low-density full-node canary 007 and passing control 032.  It
creates count evidence only and cannot select a production recipe or authorize
cohort scale-up.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np


EXP = Path("/home/ec2-user/exp")
HCP_ROOT = Path("/data/derivatives/hcp379_v2")
BASE_SOURCE = (
    EXP / "scripts/hcp/run_hcp379_gmwmi_seeding_counterfactual_v1.py"
)
GMWMI_DECISION = (
    EXP
    / "research_audit/outputs/"
    "hcp379_gmwmi_seeding_counterfactual_v1/decision.json"
)
PREDECLARATION = (
    EXP
    / "research_audit/outputs/"
    "hcp379_fod_cutoff_counterfactual_v1/predeclaration.json"
)
ATTEMPTS = (
    HCP_ROOT / "phase_b_hroi_fod_cutoff_counterfactual_v1/attempts"
)
MRTRIX = Path("/home/ec2-user/mrtrix3/bin")
UNITS = ("007_S_5196_I390043", "032_S_6804_I1230908")
ROLE = {
    "007_S_5196_I390043": "LOWEST_DENSITY_FULL_NODE_ADVERSE",
    "032_S_6804_I1230908": "HIGH_DENSITY_PASSING_CONTROL",
}
SEED_CLASSES = ("primary", "independent")
BASELINE_CUTOFF = 0.06
COUNTERFACTUAL_CUTOFF = 0.05
PER_SEED_STREAMLINES = 3_000_000
TOTAL_STREAMLINES = 6_000_000
MAXIMUM_SEEDS = 200_000_000
EXPECTED_NODES = 379
EXPECTED_THREADS_PER_JOB = 16


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = load_module(BASE_SOURCE, "hcp379_fod_cutoff_base")
DENSITY = BASE.DENSITY
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


def run_capture(command: list[str]) -> str:
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def mrstats_count(image: Path, mask: Path) -> int:
    value = run_capture(
        [
            str(MRTRIX / "mrstats"),
            str(image),
            "-mask",
            str(mask),
            "-output",
            "count",
            "-quiet",
        ]
    )
    return int(float(value))


def fod_peak_band_evidence(bound: Mapping[str, Any]) -> dict[str, Any]:
    """Count strongest-FOD peaks around the proposed cutoff in WM voxels."""
    with tempfile.TemporaryDirectory(
        prefix="hcp379_fod_cutoff_predeclare_"
    ) as temporary:
        root = Path(temporary)
        wm_mask = root / "wm_probability_gt_0p5.mif"
        wm_probability = root / "wm_probability.mif"
        peak_vector = root / "peak_vector.mif"
        peak_amplitude = root / "peak_amplitude.mif"
        subprocess.run(
            [
                str(MRTRIX / "mrconvert"),
                str(bound["five_tt"]),
                str(wm_probability),
                "-coord",
                "3",
                "2",
                "-quiet",
            ],
            check=True,
        )
        subprocess.run(
            [
                str(MRTRIX / "mrcalc"),
                str(wm_probability),
                "0.5",
                "-gt",
                str(wm_mask),
                "-quiet",
            ],
            check=True,
        )
        subprocess.run(
            [
                str(MRTRIX / "sh2peaks"),
                str(bound["wmfod"]),
                str(peak_vector),
                "-num",
                "1",
                "-mask",
                str(wm_mask),
                "-nthreads",
                "8",
                "-quiet",
            ],
            check=True,
        )
        subprocess.run(
            [
                str(MRTRIX / "mrmath"),
                str(peak_vector),
                "norm",
                str(peak_amplitude),
                "-axis",
                "3",
                "-quiet",
            ],
            check=True,
        )
        total = mrstats_count(peak_amplitude, wm_mask)
        band_mask = root / "peak_0p05_to_0p06.mif"
        subprocess.run(
            [
                str(MRTRIX / "mrcalc"),
                str(peak_amplitude),
                str(COUNTERFACTUAL_CUTOFF),
                "-ge",
                str(peak_amplitude),
                str(BASELINE_CUTOFF),
                "-lt",
                "-mult",
                str(wm_mask),
                "-mult",
                str(band_mask),
                "-quiet",
            ],
            check=True,
        )
        band_n = mrstats_count(peak_amplitude, band_mask)
    return {
        "wm_probability_rule": "5TT WM channel > 0.5",
        "peak_rule": "largest sh2peaks amplitude per voxel",
        "wm_voxels_n": total,
        "peak_0p05_to_0p06_n": band_n,
        "peak_0p05_to_0p06_fraction": band_n / total,
    }


def bind_units() -> dict[str, dict[str, Any]]:
    decision = load_json(GMWMI_DECISION)
    if (
        decision.get("status")
        != "REJECT_GMWMI_AS_UNIFORM_REMEDY_AT_6M_SCREEN"
        or decision.get("scale_up_authorized") is not False
    ):
        raise ValueError("completed GMWMI rejection is required")
    bound = {unit: BASE.bind_unit(unit) for unit in UNITS}
    for unit, value in bound.items():
        for seed_class in SEED_CLASSES:
            seed = value["seeds"][seed_class]
            if int(seed["source_rng_seed"]) <= 0:
                raise ValueError(f"{unit}/{seed_class}: invalid source RNG")
    return bound


def predeclaration() -> dict[str, Any]:
    bound = bind_units()
    fod_evidence = {
        unit: fod_peak_band_evidence(value)
        for unit, value in bound.items()
    }
    adverse_fraction = fod_evidence[UNITS[0]][
        "peak_0p05_to_0p06_fraction"
    ]
    control_fraction = fod_evidence[UNITS[1]][
        "peak_0p05_to_0p06_fraction"
    ]
    if not adverse_fraction > control_fraction:
        raise ValueError("proposed cutoff lacks adverse/control mechanism")
    return {
        "schema_version": "1.0.0",
        "record_type": (
            "diagnosis_blind_hcp379_fod_cutoff_counterfactual_predeclaration"
        ),
        "generated_utc": utc_now(),
        "status": "PREDECLARED_READY_FOR_BOUNDED_EXECUTION",
        "diagnosis_labels_used": False,
        "outcomes_used": False,
        "non_overwriting": True,
        "units": list(UNITS),
        "roles": ROLE,
        "changed_factor_only": {
            "parameter": "normalised_wmfod_termination_cutoff",
            "baseline": BASELINE_CUTOFF,
            "counterfactual": COUNTERFACTUAL_CUTOFF,
        },
        "held_fixed": {
            "algorithm": "iFOD2",
            "act": True,
            "backtrack": True,
            "crop_at_gmwmi": True,
            "seeding": "seed_dynamic",
            "random_seed": "same exact RNG seed as corresponding baseline",
            "per_seed_streamlines": PER_SEED_STREAMLINES,
            "balanced_total_streamlines": TOTAL_STREAMLINES,
            "maximum_seed_attempts": MAXIMUM_SEEDS,
            "minimum_length_mm": 10.0,
            "maximum_length_mm": 250.0,
            "maximum_angle_degrees": 45.0,
            "step_size": "exact subject-specific baseline value",
            "threads_per_job": EXPECTED_THREADS_PER_JOB,
            "atlas_registration_assignment_and_matrix_construction": (
                "exact final-HROI baseline"
            ),
        },
        "mechanistic_evidence": fod_evidence,
        "baseline_density_6m": {
            unit: float(value["baseline"]["edge_density"])
            for unit, value in bound.items()
        },
        "predeclared_expansion_criteria": {
            "required_connected_nodes_each_unit": EXPECTED_NODES,
            "minimum_endpoint_assignment_fraction_each_unit": 0.85,
            "adverse_minimum_absolute_density_gain": 0.02,
            "adverse_each_seed_supported_edge_gain_must_exceed": (
                "absolute baseline primary-independent supported-edge "
                "difference"
            ),
            "control_maximum_absolute_density_loss": 0.005,
            "control_each_seed_supported_edge_loss_must_not_exceed": (
                "absolute baseline primary-independent supported-edge "
                "difference"
            ),
            "minimum_two_seed_support_jaccard": (
                "corresponding baseline Jaccard minus 0.03"
            ),
            "stable_new_edges_definition": (
                "present in both cutoff-0.05 seeds and absent from the union "
                "of both cutoff-0.06 baseline seeds"
            ),
            "stable_new_edges_required_for_adverse": True,
        },
        "decision_boundary": (
            "Passing this two-unit screen permits only the same cutoff test "
            "on the remaining four corrected failures. It does not select a "
            "production recipe or authorize 530-subject scale-up."
        ),
        "literature_basis": [
            {
                "source": "MRtrix3 tckgen documentation",
                "url": (
                    "https://mrtrix.readthedocs.io/en/latest/reference/"
                    "commands/tckgen.html"
                ),
                "relevance": (
                    "The cutoff is the FOD-amplitude termination threshold."
                ),
            },
            {
                "source": "Tournier et al. iFOD2",
                "url": "https://archive.ismrm.org/2010/1670.html",
                "relevance": (
                    "The fixed tracking algorithm integrates over FODs."
                ),
            },
        ],
        "prior_counterfactual_decision": BALANCED.file_record(GMWMI_DECISION),
        "implementation": BALANCED.file_record(Path(__file__)),
        "imaging_executed": False,
        "production_recipe_selected": False,
        "scale_up_authorized": False,
    }


def validate_stored_predeclaration() -> dict[str, Any]:
    if not PREDECLARATION.is_file() or PREDECLARATION.is_symlink():
        raise FileNotFoundError("stored predeclaration is required")
    value = load_json(PREDECLARATION)
    if (
        value.get("record_type")
        != "diagnosis_blind_hcp379_fod_cutoff_counterfactual_predeclaration"
        or value.get("status")
        != "PREDECLARED_READY_FOR_BOUNDED_EXECUTION"
        or value.get("diagnosis_labels_used") is not False
        or value.get("outcomes_used") is not False
        or value.get("imaging_executed") is not False
        or value.get("scale_up_authorized") is not False
        or value.get("changed_factor_only", {}).get("baseline")
        != BASELINE_CUTOFF
        or value.get("changed_factor_only", {}).get("counterfactual")
        != COUNTERFACTUAL_CUTOFF
        or value.get("implementation")
        != BALANCED.file_record(Path(__file__))
    ):
        raise ValueError("stored predeclaration differs")
    return value


def environment(rng_seed: int) -> dict[str, str]:
    value = os.environ.copy()
    value["MRTRIX_RNG_SEED"] = str(rng_seed)
    value["PATH"] = f"{MRTRIX}:{value.get('PATH', '')}"
    return value


def load_assignments(path: Path, expected_n: int) -> np.ndarray:
    value = np.loadtxt(path, comments="#", dtype=np.int16)
    if value.ndim == 1:
        value = value.reshape(1, -1)
    if value.shape != (expected_n, 2):
        raise ValueError(f"{path}: assignment shape {value.shape} differs")
    if np.any(value < 0) or np.any(value > EXPECTED_NODES):
        raise ValueError(f"{path}: endpoint outside 0..379")
    return value


def prefix_baseline_matrices(
    bound: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    attempt = Path(bound["summary_path"]).parent
    matrices: dict[str, np.ndarray] = {}
    for seed_class in SEED_CLASSES:
        path = attempt / seed_class / "hroi_assignments_10m.csv"
        assignments = BALANCED.load_assignments(path)
        matrices[seed_class] = BALANCED.count_matrix(
            assignments, PER_SEED_STREAMLINES
        )
    return matrices


def stable_support_change(
    baseline: Mapping[str, np.ndarray],
    counterfactual: Mapping[str, np.ndarray],
) -> dict[str, int]:
    triangle = np.triu_indices(EXPECTED_NODES, 1)
    bp = baseline["primary"][triangle] > 0
    bi = baseline["independent"][triangle] > 0
    cp = counterfactual["primary"][triangle] > 0
    ci = counterfactual["independent"][triangle] > 0
    baseline_union = bp | bi
    baseline_overlap = bp & bi
    counterfactual_union = cp | ci
    counterfactual_overlap = cp & ci
    return {
        "stable_new_edges": int(
            np.count_nonzero(counterfactual_overlap & ~baseline_union)
        ),
        "stable_lost_edges": int(
            np.count_nonzero(baseline_overlap & ~counterfactual_union)
        ),
    }


def run_seed(
    unit: str,
    seed_class: str,
    bound: Mapping[str, Any],
    root: Path,
    threads: int,
) -> dict[str, Any]:
    seed = bound["seeds"][seed_class]
    output = root / unit / seed_class
    output.mkdir(parents=True, exist_ok=False)
    tracks_partial = output / "tracks.partial.tck"
    tracks = output / "tracks.tck"
    BASE.run_logged(
        [
            str(MRTRIX / "tckgen"),
            str(bound["wmfod"]),
            str(tracks_partial),
            "-algorithm",
            "iFOD2",
            "-act",
            str(bound["five_tt"]),
            "-backtrack",
            "-crop_at_gmwmi",
            "-seed_dynamic",
            str(bound["wmfod"]),
            "-select",
            str(PER_SEED_STREAMLINES),
            "-seeds",
            str(MAXIMUM_SEEDS),
            "-minlength",
            "10.0",
            "-maxlength",
            "250.0",
            "-cutoff",
            str(COUNTERFACTUAL_CUTOFF),
            "-step",
            f"{seed['step_size_mm']:.8f}",
            "-angle",
            "45.0",
            "-nthreads",
            str(threads),
        ],
        log=output / "tckgen.log",
        env=environment(int(seed["source_rng_seed"])),
        outputs=(tracks_partial,),
    )
    if BASE.tck_count(tracks_partial) != PER_SEED_STREAMLINES:
        raise ValueError(f"{unit}/{seed_class}: tractogram count differs")
    os.replace(tracks_partial, tracks)
    metadata = output / "tractography_parameters.json"
    atomic_json(
        metadata,
        {
            "schema_version": "1.0.0",
            "record_type": "hcp379_fod_cutoff_counterfactual_parameters",
            "generated_utc": utc_now(),
            "unit": unit,
            "role": ROLE[unit],
            "seed_class": seed_class,
            "source_seed_id": seed["source_seed_id"],
            "source_rng_seed": int(seed["source_rng_seed"]),
            "counterfactual_seed_id": (
                f"{seed['source_seed_id']}|fod-cutoff-0.05-counterfactual-v1"
            ),
            "counterfactual_rng_seed": int(seed["source_rng_seed"]),
            "paired_common_random_seed": True,
            "algorithm": "iFOD2",
            "act": True,
            "backtrack": True,
            "crop_at_gmwmi": True,
            "seeding": "seed_dynamic",
            "baseline_cutoff": BASELINE_CUTOFF,
            "counterfactual_cutoff": COUNTERFACTUAL_CUTOFF,
            "actual_step_size_mm": seed["step_size_mm"],
            "maximum_angle_degrees": 45.0,
            "minimum_length_mm": 10.0,
            "maximum_length_mm": 250.0,
            "nthreads": threads,
            "requested_streamlines": PER_SEED_STREAMLINES,
            "actual_streamlines": PER_SEED_STREAMLINES,
            "diagnosis_labels_used": False,
        },
    )
    count = output / "count_hroi.csv"
    assignments = output / "assignments_hroi.csv"
    DENSITY.run_connectome(
        tracks=tracks,
        nodes=bound["nodes"],
        output=count,
        assignments=assignments,
        log=output / "tck2connectome.log",
        threads=min(threads, 8),
    )
    assignment_array = load_assignments(
        assignments, PER_SEED_STREAMLINES
    )
    replay = BALANCED.count_matrix(
        assignment_array, PER_SEED_STREAMLINES
    )
    recorded = BALANCED.load_count(count)
    if not np.array_equal(replay, recorded):
        raise ValueError(f"{unit}/{seed_class}: assignment replay differs")
    return {
        "unit": unit,
        "seed_class": seed_class,
        "tracks": BALANCED.file_record(tracks),
        "metadata": BALANCED.file_record(metadata),
        "count_matrix": BALANCED.file_record(count),
        "assignments": BALANCED.file_record(assignments),
        "endpoint_assignment_fraction": BALANCED.assignment_fraction(
            assignment_array, PER_SEED_STREAMLINES
        ),
        "technical_qc": BALANCED.matrix_qc(recorded),
    }


def execute(workers: int, threads: int) -> dict[str, Any]:
    declaration = validate_stored_predeclaration()
    bound = bind_units()
    attempt = ATTEMPTS / f"{stamp()}-fod-cutoff-counterfactual"
    attempt.mkdir(parents=True, exist_ok=False)
    preflight_path = attempt / "preflight.json"
    atomic_json(
        preflight_path,
        {
            "schema_version": "1.0.0",
            "record_type": (
                "diagnosis_blind_hcp379_fod_cutoff_counterfactual_preflight"
            ),
            "generated_utc": utc_now(),
            "status": "EXECUTION_INPUTS_BOUND",
            "diagnosis_labels_used": False,
            "outcomes_used": False,
            "changed_factor_only": declaration["changed_factor_only"],
            "held_fixed": declaration["held_fixed"],
            "predeclared_expansion_criteria": declaration[
                "predeclared_expansion_criteria"
            ],
            "predeclaration": BALANCED.file_record(PREDECLARATION),
            "implementation": BALANCED.file_record(Path(__file__)),
            "inputs": {
                unit: {
                    "density_summary": BALANCED.file_record(
                        value["summary_path"]
                    ),
                    "promoted_pretract_manifest": BALANCED.file_record(
                        value["manifest_path"]
                    ),
                    "wmfod": BALANCED.file_record(value["wmfod"]),
                    "five_tt": BALANCED.file_record(value["five_tt"]),
                    "hroi_nodes": BALANCED.file_record(value["nodes"]),
                    "seeds": {
                        seed_class: {
                            "source_metadata": value["seeds"][
                                seed_class
                            ]["source_metadata"],
                            "source_seed_id": value["seeds"][
                                seed_class
                            ]["source_seed_id"],
                            "source_rng_seed": value["seeds"][
                                seed_class
                            ]["source_rng_seed"],
                            "step_size_mm": value["seeds"][
                                seed_class
                            ]["step_size_mm"],
                        }
                        for seed_class in SEED_CLASSES
                    },
                }
                for unit, value in bound.items()
            },
            "imaging_executed": True,
            "production_recipe_selected": False,
            "scale_up_authorized": False,
        },
    )
    state_path = attempt / "state.json"
    state: dict[str, Any] = {
        "status": "RUNNING",
        "generated_utc": utc_now(),
        "completed_jobs": [],
        "remaining_jobs": [
            f"{unit}/{seed_class}"
            for unit in UNITS
            for seed_class in SEED_CLASSES
        ],
    }
    atomic_json(state_path, state)
    results: dict[tuple[str, str], dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_map = {
            pool.submit(
                run_seed,
                unit,
                seed_class,
                bound[unit],
                attempt,
                threads,
            ): (unit, seed_class)
            for unit in UNITS
            for seed_class in SEED_CLASSES
        }
        for future in as_completed(future_map):
            key = future_map[future]
            results[key] = future.result()
            label = f"{key[0]}/{key[1]}"
            state["completed_jobs"].append(label)
            state["remaining_jobs"].remove(label)
            state["generated_utc"] = utc_now()
            atomic_json(state_path, state)

    units: dict[str, Any] = {}
    for unit in UNITS:
        baseline_matrices = prefix_baseline_matrices(bound[unit])
        counterfactual_matrices = {
            seed_class: BALANCED.load_count(
                Path(results[(unit, seed_class)]["count_matrix"]["path"])
            )
            for seed_class in SEED_CLASSES
        }
        combined = (
            counterfactual_matrices["primary"]
            + counterfactual_matrices["independent"]
        )
        combined_path = attempt / unit / "balanced_6m_count.csv"
        DENSITY.atomic_matrix(combined_path, combined)
        counterfactual_qc = BALANCED.matrix_qc(combined)
        baseline = bound[unit]["baseline"]
        baseline_complementarity = DENSITY.support_complementarity(
            baseline_matrices["primary"],
            baseline_matrices["independent"],
        )
        counterfactual_complementarity = DENSITY.support_complementarity(
            counterfactual_matrices["primary"],
            counterfactual_matrices["independent"],
        )
        baseline_assignment = (
            float(baseline["primary_endpoint_assignment_fraction"])
            + float(baseline["independent_endpoint_assignment_fraction"])
        ) / 2.0
        counterfactual_assignment = sum(
            float(
                results[(unit, seed_class)][
                    "endpoint_assignment_fraction"
                ]
            )
            for seed_class in SEED_CLASSES
        ) / 2.0
        units[unit] = {
            "role": ROLE[unit],
            "baseline_cutoff_0p06": {
                "edge_density": float(baseline["edge_density"]),
                "connected_nodes": int(baseline["connected_nodes"]),
                "endpoint_assignment_fraction": baseline_assignment,
                "support_complementarity": baseline_complementarity,
                "count_matrix": baseline["count_matrix"],
            },
            "counterfactual_cutoff_0p05": {
                **counterfactual_qc,
                "endpoint_assignment_fraction": counterfactual_assignment,
                "support_complementarity": (
                    counterfactual_complementarity
                ),
                "stable_support_change": stable_support_change(
                    baseline_matrices, counterfactual_matrices
                ),
                "count_matrix": BALANCED.file_record(combined_path),
                "seeds": {
                    seed_class: results[(unit, seed_class)]
                    for seed_class in SEED_CLASSES
                },
            },
            "difference": {
                "edge_density": float(
                    counterfactual_qc["edge_density"]
                )
                - float(baseline["edge_density"]),
                "connected_nodes": int(
                    counterfactual_qc["connected_nodes"]
                )
                - int(baseline["connected_nodes"]),
                "endpoint_assignment_fraction": (
                    counterfactual_assignment - baseline_assignment
                ),
                "seed_supported_edges": {
                    seed_class: int(
                        counterfactual_complementarity[
                            f"{seed_class}_supported_edges"
                        ]
                    )
                    - int(
                        baseline_complementarity[
                            f"{seed_class}_supported_edges"
                        ]
                    )
                    for seed_class in SEED_CLASSES
                },
                "support_jaccard": float(
                    counterfactual_complementarity["support_jaccard"]
                )
                - float(baseline_complementarity["support_jaccard"]),
            },
        }

    summary = {
        "schema_version": "1.0.0",
        "record_type": (
            "diagnosis_blind_hcp379_fod_cutoff_counterfactual"
        ),
        "generated_utc": utc_now(),
        "status": "PASS_COMPLETE_FOD_CUTOFF_COUNTERFACTUAL",
        "diagnosis_labels_used": False,
        "outcomes_used": False,
        "non_overwriting": True,
        "source_files_modified": False,
        "changed_factor_only": declaration["changed_factor_only"],
        "new_tractography_generated": True,
        "generated_streamlines": (
            len(UNITS) * len(SEED_CLASSES) * PER_SEED_STREAMLINES
        ),
        "sift2_or_scalar_matrices_generated": False,
        "units": units,
        "production_recipe_selected": False,
        "scale_up_authorized": False,
        "requires_independent_validation_and_predeclared_decision": True,
        "requires_all_nine_and_seed_stability_before_selection": True,
        "preflight": BALANCED.file_record(preflight_path),
        "predeclaration": BALANCED.file_record(PREDECLARATION),
        "implementation": BALANCED.file_record(Path(__file__)),
    }
    summary_path = attempt / "summary.json"
    atomic_json(summary_path, summary)
    state.update(
        {
            "status": summary["status"],
            "generated_utc": utc_now(),
            "summary": BALANCED.file_record(summary_path),
        }
    )
    atomic_json(state_path, state)
    return {
        "attempt_root": str(attempt),
        "status": summary["status"],
        "units": {
            unit: {
                "role": row["role"],
                "baseline_density": row["baseline_cutoff_0p06"][
                    "edge_density"
                ],
                "counterfactual_density": row[
                    "counterfactual_cutoff_0p05"
                ]["edge_density"],
                "density_difference": row["difference"]["edge_density"],
            }
            for unit, row in units.items()
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--write-predeclaration", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--threads-per-job", type=int, default=EXPECTED_THREADS_PER_JOB
    )
    args = parser.parse_args()
    if not 1 <= args.workers <= 2:
        parser.error("--workers must be in 1..2")
    if args.threads_per_job != EXPECTED_THREADS_PER_JOB:
        parser.error("--threads-per-job must remain exactly 16")
    if args.write_predeclaration:
        value = predeclaration()
        atomic_json(PREDECLARATION, value)
        value = {
            "status": value["status"],
            "predeclaration": str(PREDECLARATION),
            "mechanistic_evidence": value["mechanistic_evidence"],
        }
    elif args.execute:
        value = execute(args.workers, args.threads_per_job)
    else:
        value = predeclaration()
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
