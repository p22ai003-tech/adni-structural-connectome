#!/home/ec2-user/fsl/bin/python
"""Run a bounded two-seed connectome screen for lower-order HCP379 FODs.

The acquisition-aware FOD-order canary established that lmax=2 and lmax=4
produce numerically valid, normalised WM FODs for two prespecified failed
subjects.  This runner asks the next, narrower question: can those FODs drive
the frozen ACT/iFOD2 recipe and produce technically coherent HCP379 count
connectomes?

Each subject receives two independently seeded 3M tractograms.  Their count
matrices are evaluated individually and as a balanced 6M sum.  Six million is
the smallest predeclared Phase-B stability rung.  It is a technical screen,
not a selected production recipe: all-nine matrices, cohort-wide density
qualification, biological analyses, and release authorization remain out of
scope.

All outputs are isolated in a new attempt directory.  No historical,
production, or source artifact is overwritten.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
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
HCP = Path("/data/derivatives/hcp379_v2")
INPUT_ROOT = HCP / "corrected_scaleup_recovery4"
OUTPUT_ROOT = (
    HCP / "acquisition_aware_fod_order_tractography_canary_v1"
)
ATTEMPTS = OUTPUT_ROOT / "attempts"
AUDIT_ROOT = (
    EXP
    / "research_audit/outputs/"
    "hcp379_acquisition_aware_fod_order_tractography_canary_v1"
)
FOD_ROOT = (
    EXP
    / "research_audit/outputs/"
    "hcp379_acquisition_aware_fod_order_canary_v1"
)
FOD_SUMMARY = FOD_ROOT / "summary.json"
FOD_VALIDATION = FOD_ROOT / "validation.json"
RECIPE_PLAN = (
    INPUT_ROOT
    / "manifests/tractography_recovery4_selected_recipe_plan.json"
)
INPUT_AUDIT = (
    EXP
    / "research_audit/outputs/hcp379_scaleup_input_audit_v2/"
    "scaleup_input_audit.csv"
)
HROI_POLICY = (
    HCP
    / "source_label_repair_v1/hroi_surface_support_cohort_v1/"
    "policy_adoption_v1.json"
)
TCK_COUNTER = EXP / "scripts/hcp/tck_header_count_v1.py"
MRTRIX = Path("/home/ec2-user/mrtrix3/bin")
FSL_PYTHON = Path("/home/ec2-user/fsl/bin/python")

TARGETS = {
    "031_S_0618_I1229293": 2,
    "031_S_4721_I1093826": 4,
}
SEED_CLASSES = ("primary", "independent")
RECIPE_ID = "connectome-v2.1.0-closed-world-canary-candidate"
PER_SEED_STREAMLINES = 3_000_000
BALANCED_TOTAL_STREAMLINES = 6_000_000
EXPECTED_NODES = 379
MINIMUM_CONNECTED_NODES = 360
PREFERRED_CONNECTED_NODES = 376
MINIMUM_ENDPOINT_ASSIGNMENT_FRACTION = 0.50
PREFERRED_ENDPOINT_ASSIGNMENT_FRACTION = 0.85
DENSITY_TARGET = 0.60
ASSIGNMENT_RADIUS_MM = 4


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
        mode="w",
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


def atomic_matrix(path: Path, matrix: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.stem}.partial{path.suffix}"
    )
    np.savetxt(temporary, matrix, delimiter=",", fmt="%d")
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if (
        not resolved.is_file()
        or resolved.is_symlink()
        or resolved.stat().st_size <= 0
    ):
        raise FileNotFoundError(resolved)
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256(resolved),
    }


def verify_record(record: Mapping[str, Any]) -> Path:
    path = Path(str(record.get("path", ""))).resolve()
    if file_record(path) != {
        "path": str(path),
        "size_bytes": int(record.get("size_bytes", -1)),
        "sha256": str(record.get("sha256", "")),
    }:
        raise ValueError(f"file record drift: {path}")
    return path


def require_validation(path: Path) -> dict[str, Any]:
    value = load_json(path)
    passed = int(
        value.get("checks_passed", value.get("passed_checks", -1))
    )
    total = int(
        value.get("checks_total", value.get("total_checks", -1))
    )
    if value.get("status") != "PASS" or total <= 0 or passed != total:
        raise ValueError(f"validation differs: {path}")
    return value


def input_rows() -> dict[str, dict[str, str]]:
    with INPUT_AUDIT.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    result = {str(row["unit"]): row for row in rows}
    if len(rows) != 227 or len(result) != 227:
        raise ValueError("scale-up identity audit differs")
    return result


def mrinfo_spacing(path: Path) -> tuple[float, float, float]:
    completed = subprocess.run(
        [str(MRTRIX / "mrinfo"), str(path), "-spacing"],
        check=True,
        capture_output=True,
        text=True,
    )
    values = [float(value) for value in completed.stdout.split()[:3]]
    if len(values) != 3 or any(
        not np.isfinite(value) or value <= 0 for value in values
    ):
        raise ValueError(f"invalid voxel spacing: {path}")
    return tuple(values)  # type: ignore[return-value]


def seed_identity(
    row: Mapping[str, str], seed_class: str
) -> tuple[str, int]:
    base = (
        f"{row['dti_source_id']}|{row['dti_raw_bundle_sha256']}|"
        f"{RECIPE_ID}"
    )
    token = (
        base
        if seed_class == "primary"
        else f"{base}|independent-replicate-1"
    )
    seed = (
        int.from_bytes(
            hashlib.sha256(token.encode("utf-8")).digest()[:4],
            "big",
        )
        & 0x7FFFFFFF
    ) or 1
    return token, seed


def load_assignments(
    path: Path, expected_n: int = PER_SEED_STREAMLINES
) -> np.ndarray:
    values = np.loadtxt(path, comments="#", dtype=np.int16)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.shape != (expected_n, 2):
        raise ValueError(
            f"{path}: assignment shape {values.shape} differs from "
            f"({expected_n}, 2)"
        )
    if np.any(values < 0) or np.any(values > EXPECTED_NODES):
        raise ValueError(f"{path}: endpoint outside 0..{EXPECTED_NODES}")
    return values


def count_matrix(assignments: np.ndarray) -> np.ndarray:
    left = assignments[:, 0].astype(np.int32, copy=False)
    right = assignments[:, 1].astype(np.int32, copy=False)
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


def load_count(path: Path) -> np.ndarray:
    matrix = np.loadtxt(path, delimiter=",")
    if (
        matrix.shape != (EXPECTED_NODES, EXPECTED_NODES)
        or not np.isfinite(matrix).all()
        or np.any(matrix < 0)
        or not np.array_equal(matrix, matrix.T)
        or not np.all(np.diag(matrix) == 0)
        or not np.allclose(
            matrix, np.rint(matrix), atol=1.0e-6, rtol=0.0
        )
    ):
        raise ValueError(f"count matrix invariants differ: {path}")
    return np.rint(matrix).astype(np.int64)


def assignment_fraction(assignments: np.ndarray) -> float:
    assigned = (
        (assignments[:, 0] > 0) & (assignments[:, 1] > 0)
    )
    return float(np.mean(assigned))


def matrix_qc(matrix: np.ndarray) -> dict[str, Any]:
    triangle = matrix[np.triu_indices(EXPECTED_NODES, 1)]
    supported = int(np.count_nonzero(triangle > 0))
    possible = EXPECTED_NODES * (EXPECTED_NODES - 1) // 2
    connected = int(np.count_nonzero(np.sum(matrix, axis=1) > 0))
    return {
        "supported_edges": supported,
        "possible_edges": possible,
        "edge_density": supported / possible,
        "connected_nodes": connected,
        "minimum_connected_nodes_met": (
            connected >= MINIMUM_CONNECTED_NODES
        ),
        "preferred_connected_nodes_met": (
            connected >= PREFERRED_CONNECTED_NODES
        ),
        "all_379_nodes_connected": connected == EXPECTED_NODES,
        "total_assigned_nondiagonal_streamlines": int(
            np.sum(triangle)
        ),
    }


def support_stability(
    primary: np.ndarray, independent: np.ndarray
) -> dict[str, Any]:
    triangle = np.triu_indices(EXPECTED_NODES, 1)
    first = primary[triangle] > 0
    second = independent[triangle] > 0
    union = first | second
    overlap = first & second
    union_n = int(np.count_nonzero(union))
    return {
        "primary_supported_edges": int(np.count_nonzero(first)),
        "independent_supported_edges": int(np.count_nonzero(second)),
        "overlap_supported_edges": int(np.count_nonzero(overlap)),
        "union_supported_edges": union_n,
        "support_jaccard": (
            float(np.count_nonzero(overlap)) / union_n
            if union_n
            else 0.0
        ),
        "primary_only_supported_edges": int(
            np.count_nonzero(first & ~second)
        ),
        "independent_only_supported_edges": int(
            np.count_nonzero(second & ~first)
        ),
    }


def tck_count(path: Path) -> int:
    completed = subprocess.run(
        [str(FSL_PYTHON), str(TCK_COUNTER), str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    return int(completed.stdout.strip())


def frozen_recipe(plan: Mapping[str, Any]) -> dict[str, Any]:
    recipe = plan.get("tractography_recipe")
    if not isinstance(recipe, Mapping):
        raise ValueError("tractography recipe absent")
    expected = {
        "framework": "ACT",
        "algorithm": "iFOD2",
        "act_enabled": True,
        "backtrack": True,
        "crop_at_gmwmi": True,
        "automatic_act_to_noact_switch_allowed": False,
        "automatic_cutoff_ladder_allowed": False,
        "cutoff": 0.06,
        "minimum_length_mm": 10,
        "maximum_length_mm": 250,
        "maximum_angle_degrees": 45,
        "maximum_seed_attempts": 200_000_000,
    }
    if any(recipe.get(key) != value for key, value in expected.items()):
        raise ValueError("frozen tractography recipe differs")
    seeding = recipe.get("seeding")
    if (
        not isinstance(seeding, Mapping)
        or seeding.get("method") != "seed_dynamic"
        or seeding.get("gmwmi_used_as_seed") is not False
    ):
        raise ValueError("frozen seeding recipe differs")
    return dict(recipe)


def preflight() -> dict[str, Any]:
    fod_summary = load_json(FOD_SUMMARY)
    require_validation(FOD_VALIDATION)
    if (
        fod_summary.get("status") != "PASS_CANARY_BOTH_ORDERS"
        or fod_summary.get("target_n") != 2
        or fod_summary.get("passed_n") != 2
        or fod_summary.get("diagnosis_labels_used") is not False
        or fod_summary.get("outcomes_used") is not False
        or fod_summary.get("tractography_generated") is not False
    ):
        raise ValueError("FOD-order canary differs")

    plan = load_json(RECIPE_PLAN)
    recipe = frozen_recipe(plan)
    if (
        plan.get("status") != "PRE_PHASE_DRY_RUN_PASS"
        or plan.get("recipe_id") != RECIPE_ID
        or plan.get("selected_streamline_count") is not None
        or 3_000_000 not in plan.get("allowed_streamline_counts", [])
        or plan.get("selection_rule")
        != (
            "smallest Phase-B density-qualified stable count; "
            "no subject-level recipe selection"
        )
    ):
        raise ValueError("recipe-selection state differs")

    hroi = load_json(HROI_POLICY)
    if (
        hroi.get("status")
        != "PASS_OPERATIONAL_PRETRACT_ADOPTION_FINAL_HUMAN_QC_PENDING"
        or hroi.get("diagnosis_labels_used") is not False
        or hroi.get("unit_n") != 530
    ):
        raise ValueError("HROI policy differs")
    policy_units = hroi.get("units")
    if not isinstance(policy_units, Mapping):
        raise TypeError("HROI unit map absent")

    rows = input_rows()
    fod_units = {
        str(row["unit"]): row
        for row in fod_summary.get("units", [])
        if isinstance(row, Mapping)
    }
    records: list[dict[str, Any]] = []
    for unit, expected_lmax in TARGETS.items():
        row = rows.get(unit)
        fod_row = fod_units.get(unit)
        policy = policy_units.get(unit)
        if (
            not isinstance(row, Mapping)
            or not isinstance(fod_row, Mapping)
            or not isinstance(policy, Mapping)
            or int(fod_row.get("recommended_lmax", -1))
            != expected_lmax
            or fod_row.get("status") != "PASS"
        ):
            raise ValueError(f"target binding differs: {unit}")
        fod_result_path = verify_record(fod_row["result"])
        fod_result = load_json(fod_result_path)
        wmfod = verify_record(fod_result["artifacts"]["wmfod_norm"])
        candidate_manifest = verify_record(policy["candidate_manifest"])
        candidate = verify_record(policy["candidate"])
        candidate_value = load_json(candidate_manifest)
        if (
            candidate_value.get("status")
            != "PASS_ALL_379_SOURCE_LABELS"
            or candidate_value.get("present_source_label_n") != 379
            or candidate_value.get("missing_source_labels") != []
            or candidate_value.get("candidate") != policy["candidate"]
        ):
            raise ValueError(f"HROI candidate differs: {unit}")

        subject = INPUT_ROOT / "subjects" / unit
        five_tt = subject / "05_model/5tt_dwi.mif"
        dwi = subject / "05_model/dwi_fod_shells.mif"
        nodes = subject / "04_hcp/hcp379_nodes_b0_1mm.nii.gz"
        node_volumes = subject / "04_hcp/hcp379_node_volumes.csv"
        atlas_qc_path = subject / "04_hcp/atlas_qc.json"
        hcp_t1 = subject / "04_hcp/HCPMMP1+aseg_t1.nii.gz"
        atlas_qc = load_json(atlas_qc_path)
        if (
            atlas_qc.get("status") != "PASS"
            or atlas_qc.get("labels_found") != EXPECTED_NODES
            or atlas_qc.get("missing_labels") != []
            or atlas_qc.get("unexpected_labels") != []
        ):
            raise ValueError(f"atlas gate differs: {unit}")
        spacing = mrinfo_spacing(dwi)
        step = min(spacing) / 2.0
        seeds = {}
        for seed_class in SEED_CLASSES:
            token, seed = seed_identity(row, seed_class)
            seeds[seed_class] = {
                "token": token,
                "rng_seed": seed,
            }
        if seeds["primary"]["rng_seed"] == seeds["independent"]["rng_seed"]:
            raise ValueError(f"seed collision: {unit}")
        records.append(
            {
                "unit": unit,
                "recommended_lmax": expected_lmax,
                "unique_direction_n": int(
                    fod_row["unique_direction_n"]
                ),
                "dti_source_id": str(row["dti_source_id"]),
                "dti_raw_bundle_sha256": str(
                    row["dti_raw_bundle_sha256"]
                ),
                "wmfod_normalised": file_record(wmfod),
                "five_tt": file_record(five_tt),
                "dwi_for_spacing": file_record(dwi),
                "hcp379_nodes": file_record(nodes),
                "hcp379_node_volumes": file_record(node_volumes),
                "hcp379_atlas_qc": file_record(atlas_qc_path),
                "hcp379_t1_labels": file_record(hcp_t1),
                "hroi_candidate": file_record(candidate),
                "hroi_candidate_manifest": file_record(
                    candidate_manifest
                ),
                "voxel_spacing_mm": list(spacing),
                "step_size_mm": step,
                "seeds": seeds,
            }
        )

    return {
        "schema_version": "1.0.0",
        "record_type": (
            "hcp379_acquisition_aware_fod_order_tractography_"
            "canary_preflight"
        ),
        "status": "READY_BOUNDED_6M_TECHNICAL_SCREEN",
        "generated_utc": utc_now(),
        "diagnosis_labels_used": False,
        "outcomes_used": False,
        "group_effects_used": False,
        "non_overwriting": True,
        "historical_outputs_modified": False,
        "production_recipe_selected": False,
        "scale_up_authorized": False,
        "all_nine_matrix_generation_authorized": False,
        "target_n": 2,
        "per_seed_streamlines": PER_SEED_STREAMLINES,
        "balanced_total_streamlines": BALANCED_TOTAL_STREAMLINES,
        "screen_rationale": (
            "smallest predeclared two-seed Phase-B stability rung; "
            "not a production streamline-count selection"
        ),
        "recipe": recipe,
        "technical_contract": {
            "expected_nodes": EXPECTED_NODES,
            "assignment_radial_search_mm": ASSIGNMENT_RADIUS_MM,
            "minimum_connected_nodes": MINIMUM_CONNECTED_NODES,
            "preferred_connected_nodes": PREFERRED_CONNECTED_NODES,
            "minimum_endpoint_assignment_fraction": (
                MINIMUM_ENDPOINT_ASSIGNMENT_FRACTION
            ),
            "preferred_endpoint_assignment_fraction": (
                PREFERRED_ENDPOINT_ASSIGNMENT_FRACTION
            ),
            "balanced_density_target": DENSITY_TARGET,
            "density_is_subject_exclusion_gate": False,
        },
        "units": records,
        "records": {
            "fod_summary": file_record(FOD_SUMMARY),
            "fod_validation": file_record(FOD_VALIDATION),
            "recipe_plan": file_record(RECIPE_PLAN),
            "input_audit": file_record(INPUT_AUDIT),
            "hroi_policy": file_record(HROI_POLICY),
            "tck_counter": file_record(TCK_COUNTER),
            "tckgen": file_record(MRTRIX / "tckgen"),
            "tck2connectome": file_record(
                MRTRIX / "tck2connectome"
            ),
            "implementation": file_record(Path(__file__)),
        },
    }


def environment(rng_seed: int) -> dict[str, str]:
    value = os.environ.copy()
    value["MRTRIX_RNG_SEED"] = str(rng_seed)
    value["PATH"] = f"{MRTRIX}:{value.get('PATH', '')}"
    return value


def run_logged(
    command: list[str],
    *,
    log_path: Path,
    env: Mapping[str, str] | None = None,
) -> None:
    started = time.monotonic()
    with log_path.open("x", encoding="utf-8") as log:
        log.write("COMMAND " + json.dumps(command) + "\n")
        log.flush()
        completed = subprocess.run(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=dict(env) if env is not None else None,
            check=False,
        )
        log.write(
            f"\nRETURNCODE {completed.returncode}\n"
            f"ELAPSED_SECONDS {time.monotonic() - started:.3f}\n"
        )
    if completed.returncode:
        raise RuntimeError(
            f"command failed rc={completed.returncode}: {command[0]}"
        )


def run_seed(
    unit_spec: Mapping[str, Any],
    seed_class: str,
    *,
    attempt: Path,
    threads: int,
) -> dict[str, Any]:
    unit = str(unit_spec["unit"])
    root = attempt / "subjects" / unit / seed_class
    root.mkdir(parents=True, exist_ok=False)
    tracks_partial = root / "tracks.partial.tck"
    tracks = root / "tracks.tck"
    metadata = root / "tractography_parameters.json"
    count = root / "count.csv"
    assignments = root / "assignments.csv"
    count_partial = root / ".count.partial.csv"
    assignments_partial = root / ".assignments.partial.csv"
    recipe = load_json(attempt / "preflight.json")["recipe"]
    seed = int(unit_spec["seeds"][seed_class]["rng_seed"])
    token = str(unit_spec["seeds"][seed_class]["token"])
    state: dict[str, Any] = {
        "schema_version": "1.0.0",
        "record_type": (
            "hcp379_acquisition_aware_fod_order_tractography_seed"
        ),
        "status": "RUNNING",
        "generated_utc": utc_now(),
        "unit": unit,
        "seed_class": seed_class,
        "diagnosis_labels_used": False,
        "outcomes_used": False,
        "non_overwriting": True,
    }
    started = time.monotonic()
    try:
        run_logged(
            [
                str(MRTRIX / "tckgen"),
                str(unit_spec["wmfod_normalised"]["path"]),
                str(tracks_partial),
                "-algorithm",
                "iFOD2",
                "-act",
                str(unit_spec["five_tt"]["path"]),
                "-backtrack",
                "-crop_at_gmwmi",
                "-seed_dynamic",
                str(unit_spec["wmfod_normalised"]["path"]),
                "-select",
                str(PER_SEED_STREAMLINES),
                "-seeds",
                str(recipe["maximum_seed_attempts"]),
                "-minlength",
                str(recipe["minimum_length_mm"]),
                "-maxlength",
                str(recipe["maximum_length_mm"]),
                "-cutoff",
                str(recipe["cutoff"]),
                "-step",
                f"{float(unit_spec['step_size_mm']):.8f}",
                "-angle",
                str(recipe["maximum_angle_degrees"]),
                "-nthreads",
                str(threads),
            ],
            log_path=root / "tckgen.log",
            env=environment(seed),
        )
        if tck_count(tracks_partial) != PER_SEED_STREAMLINES:
            raise ValueError("tractogram count differs")
        os.replace(tracks_partial, tracks)
        atomic_json(
            metadata,
            {
                "schema_version": "1.0.0",
                "record_type": (
                    "hcp379_acquisition_aware_fod_order_"
                    "tractography_parameters"
                ),
                "status": "PASS",
                "generated_utc": utc_now(),
                "unit": unit,
                "recommended_lmax": int(
                    unit_spec["recommended_lmax"]
                ),
                "seed_class": seed_class,
                "seed_id": token,
                "rng_seed": seed,
                "recipe_id": RECIPE_ID,
                "algorithm": "iFOD2",
                "act": True,
                "backtrack": True,
                "crop_at_gmwmi": True,
                "seeding": "seed_dynamic",
                "cutoff": float(recipe["cutoff"]),
                "actual_step_size_mm": float(
                    unit_spec["step_size_mm"]
                ),
                "minimum_length_mm": float(
                    recipe["minimum_length_mm"]
                ),
                "maximum_length_mm": float(
                    recipe["maximum_length_mm"]
                ),
                "maximum_angle_degrees": float(
                    recipe["maximum_angle_degrees"]
                ),
                "requested_streamlines": PER_SEED_STREAMLINES,
                "actual_streamlines": PER_SEED_STREAMLINES,
                "diagnosis_labels_used": False,
                "outcomes_used": False,
            },
        )
        run_logged(
            [
                str(MRTRIX / "tck2connectome"),
                str(tracks),
                str(unit_spec["hcp379_nodes"]["path"]),
                str(count_partial),
                "-assignment_radial_search",
                str(ASSIGNMENT_RADIUS_MM),
                "-symmetric",
                "-zero_diagonal",
                "-stat_edge",
                "sum",
                "-out_assignments",
                str(assignments_partial),
                "-nthreads",
                str(min(threads, 8)),
            ],
            log_path=root / "tck2connectome.log",
            env=environment(seed),
        )
        if (
            not count_partial.is_file()
            or not assignments_partial.is_file()
        ):
            raise RuntimeError("connectome outputs absent")
        os.replace(count_partial, count)
        os.replace(assignments_partial, assignments)
        assignment_array = load_assignments(assignments)
        recorded = load_count(count)
        replay = count_matrix(assignment_array)
        if not np.array_equal(recorded, replay):
            raise ValueError("assignment replay differs")
        fraction = assignment_fraction(assignment_array)
        qc = matrix_qc(recorded)
        failures: list[str] = []
        if fraction < MINIMUM_ENDPOINT_ASSIGNMENT_FRACTION:
            failures.append(
                "endpoint_assignment_fraction_below_minimum"
            )
        if not qc["minimum_connected_nodes_met"]:
            failures.append("connected_nodes_below_minimum")
        state.update(
            {
                "status": (
                    "PASS_TECHNICAL_SEED"
                    if not failures
                    else "FAIL_TECHNICAL_SEED"
                ),
                "completed_utc": utc_now(),
                "elapsed_seconds": round(
                    time.monotonic() - started, 3
                ),
                "requested_streamlines": PER_SEED_STREAMLINES,
                "actual_streamlines": PER_SEED_STREAMLINES,
                "endpoint_assignment_fraction": fraction,
                "minimum_endpoint_assignment_met": (
                    fraction
                    >= MINIMUM_ENDPOINT_ASSIGNMENT_FRACTION
                ),
                "preferred_endpoint_assignment_met": (
                    fraction
                    >= PREFERRED_ENDPOINT_ASSIGNMENT_FRACTION
                ),
                "assignment_replay_exact": True,
                "technical_qc": qc,
                "failures": failures,
                "artifacts": {
                    "tractogram": file_record(tracks),
                    "tractography_parameters": file_record(metadata),
                    "count_matrix": file_record(count),
                    "assignments": file_record(assignments),
                    "tckgen_log": file_record(root / "tckgen.log"),
                    "tck2connectome_log": file_record(
                        root / "tck2connectome.log"
                    ),
                },
                "error": None,
            }
        )
    except Exception as exc:
        state.update(
            {
                "status": "FAIL_TECHNICAL_SEED",
                "completed_utc": utc_now(),
                "elapsed_seconds": round(
                    time.monotonic() - started, 3
                ),
                "failures": [
                    f"{type(exc).__name__}:{exc}"
                ],
                "error": f"{type(exc).__name__}:{exc}",
            }
        )
    atomic_json(root / "result.json", state)
    return state


def execute(*, workers: int, threads: int) -> dict[str, Any]:
    pre = preflight()
    attempt = (
        ATTEMPTS / f"{stamp()}-fod-order-tractography-canary"
    )
    attempt.mkdir(parents=True, exist_ok=False)
    pre = {
        **pre,
        "status": "EXECUTION_INPUTS_BOUND",
        "attempt_root": str(attempt.resolve()),
        "imaging_executed": True,
        "workers": workers,
        "threads_per_job": threads,
    }
    atomic_json(attempt / "preflight.json", pre)
    state_path = attempt / "state.json"
    jobs = [
        (unit_spec, seed_class)
        for unit_spec in pre["units"]
        for seed_class in SEED_CLASSES
    ]
    state: dict[str, Any] = {
        "status": "RUNNING",
        "generated_utc": utc_now(),
        "attempt_root": str(attempt.resolve()),
        "completed_jobs": [],
        "remaining_jobs": [
            f"{spec['unit']}/{seed_class}"
            for spec, seed_class in jobs
        ],
    }
    atomic_json(state_path, state)
    results: dict[tuple[str, str], dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                run_seed,
                spec,
                seed_class,
                attempt=attempt,
                threads=threads,
            ): (str(spec["unit"]), seed_class)
            for spec, seed_class in jobs
        }
        for index, future in enumerate(as_completed(futures), start=1):
            key = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "unit": key[0],
                    "seed_class": key[1],
                    "status": "FAIL_TECHNICAL_SEED",
                    "error": f"{type(exc).__name__}:{exc}",
                }
            results[key] = result
            label = f"{key[0]}/{key[1]}"
            state["completed_jobs"].append(label)
            state["remaining_jobs"] = [
                value
                for value in state["remaining_jobs"]
                if value != label
            ]
            state["updated_utc"] = utc_now()
            atomic_json(state_path, state)
            print(
                f"[{utc_now()}] lower-order tractography "
                f"{index}/{len(jobs)} {label} "
                f"{result.get('status')}",
                flush=True,
            )

    subject_results: list[dict[str, Any]] = []
    for spec in pre["units"]:
        unit = str(spec["unit"])
        first = results[(unit, "primary")]
        second = results[(unit, "independent")]
        failures: list[str] = []
        if first.get("status") != "PASS_TECHNICAL_SEED":
            failures.append("primary_seed_failed")
        if second.get("status") != "PASS_TECHNICAL_SEED":
            failures.append("independent_seed_failed")
        balanced_record: dict[str, Any] | None = None
        if not failures:
            primary = load_count(
                Path(first["artifacts"]["count_matrix"]["path"])
            )
            independent = load_count(
                Path(second["artifacts"]["count_matrix"]["path"])
            )
            balanced = primary + independent
            balanced_path = (
                attempt
                / "subjects"
                / unit
                / "balanced_6m_count.csv"
            )
            atomic_matrix(balanced_path, balanced)
            qc = matrix_qc(balanced)
            stability = support_stability(primary, independent)
            density_met = float(qc["edge_density"]) >= DENSITY_TARGET
            preferred_assignment_met = all(
                float(row["endpoint_assignment_fraction"])
                >= PREFERRED_ENDPOINT_ASSIGNMENT_FRACTION
                for row in (first, second)
            )
            balanced_record = {
                "total_streamlines": BALANCED_TOTAL_STREAMLINES,
                "construction": (
                    "3M primary plus 3M independent; fixed recipe "
                    "and independently derived RNG seeds"
                ),
                "count_matrix": file_record(balanced_path),
                "technical_qc": qc,
                "support_stability": stability,
                "density_target_0_60_met": density_met,
                "preferred_endpoint_assignment_each_seed_met": (
                    preferred_assignment_met
                ),
                "preferred_connected_nodes_met": qc[
                    "preferred_connected_nodes_met"
                ],
            }
        subject_status = (
            "PASS_TECHNICAL_LOWER_ORDER_CONNECTOME"
            if not failures
            else "FAIL_TECHNICAL_LOWER_ORDER_CONNECTOME"
        )
        subject = {
            "schema_version": "1.0.0",
            "record_type": (
                "hcp379_acquisition_aware_fod_order_"
                "tractography_canary_subject"
            ),
            "status": subject_status,
            "generated_utc": utc_now(),
            "unit": unit,
            "recommended_lmax": int(spec["recommended_lmax"]),
            "unique_direction_n": int(spec["unique_direction_n"]),
            "diagnosis_labels_used": False,
            "outcomes_used": False,
            "non_overwriting": True,
            "failures": failures,
            "seeds": {
                seed_class: file_record(
                    attempt
                    / "subjects"
                    / unit
                    / seed_class
                    / "result.json"
                )
                for seed_class in SEED_CLASSES
            },
            "balanced_6m": balanced_record,
        }
        result_path = (
            attempt / "subjects" / unit / "result.json"
        )
        atomic_json(result_path, subject)
        subject["result"] = file_record(result_path)
        subject_results.append(subject)

    technical_pass_n = sum(
        row["status"] == "PASS_TECHNICAL_LOWER_ORDER_CONNECTOME"
        for row in subject_results
    )
    density_pass_n = sum(
        bool(
            row.get("balanced_6m")
            and row["balanced_6m"]["density_target_0_60_met"]
        )
        for row in subject_results
    )
    if technical_pass_n == len(TARGETS):
        status = (
            "PASS_TECHNICAL_BOTH_ORDERS_DENSITY_SCREEN_MET"
            if density_pass_n == len(TARGETS)
            else "PASS_TECHNICAL_BOTH_ORDERS_DENSITY_SCREEN_NOT_MET"
        )
    else:
        status = "FAIL_LOWER_ORDER_TRACTOGRAPHY_TECHNICAL_CANARY"
    summary = {
        "schema_version": "1.0.0",
        "record_type": (
            "hcp379_acquisition_aware_fod_order_tractography_"
            "canary_summary"
        ),
        "status": status,
        "generated_utc": utc_now(),
        "attempt_root": str(attempt.resolve()),
        "diagnosis_labels_used": False,
        "outcomes_used": False,
        "group_effects_used": False,
        "non_overwriting": True,
        "historical_outputs_modified": False,
        "production_recipe_selected": False,
        "scale_up_authorized": False,
        "all_nine_matrices_generated": False,
        "target_n": len(TARGETS),
        "technical_pass_n": technical_pass_n,
        "balanced_density_target_pass_n": density_pass_n,
        "per_seed_streamlines": PER_SEED_STREAMLINES,
        "balanced_total_streamlines": BALANCED_TOTAL_STREAMLINES,
        "units": subject_results,
        "decision": {
            "technical_question_answered": technical_pass_n
            == len(TARGETS),
            "density_screen_answered": technical_pass_n
            == len(TARGETS),
            "promote_all_ten_lower_order_units": False,
            "select_production_streamline_count": False,
            "authorize_all_nine_canary": (
                technical_pass_n == len(TARGETS)
            ),
            "next_gate": (
                "bounded all-nine lower-order canary plus final "
                "Recovery4 spatial-route promotion"
                if technical_pass_n == len(TARGETS)
                else "mechanism-specific lower-order tractography review"
            ),
        },
        "preflight": file_record(attempt / "preflight.json"),
        "state": file_record(state_path),
        "implementation": file_record(Path(__file__)),
    }
    atomic_json(attempt / "summary.json", summary)
    state.update(
        {
            "status": "COMPLETE",
            "completed_utc": utc_now(),
            "summary_path": str(
                (attempt / "summary.json").resolve()
            ),
        }
    )
    atomic_json(state_path, state)
    summary["state"] = file_record(state_path)
    atomic_json(attempt / "summary.json", summary)
    AUDIT_ROOT.mkdir(parents=True, exist_ok=True)
    atomic_json(AUDIT_ROOT / "summary.json", summary)
    return summary


def self_test() -> dict[str, Any]:
    assignments = np.asarray(
        [[1, 2], [2, 1], [1, 3], [0, 3], [3, 3]],
        dtype=np.int16,
    )
    matrix = count_matrix(assignments)
    primary_token, primary_seed = seed_identity(
        {
            "dti_source_id": "a" * 64,
            "dti_raw_bundle_sha256": "b" * 64,
        },
        "primary",
    )
    independent_token, independent_seed = seed_identity(
        {
            "dti_source_id": "a" * 64,
            "dti_raw_bundle_sha256": "b" * 64,
        },
        "independent",
    )
    checks = {
        "count_replay": bool(
            matrix[0, 1] == 2
            and matrix[1, 0] == 2
            and matrix[0, 2] == 1
            and int(np.sum(matrix) // 2) == 3
        ),
        "assignment_fraction": bool(
            np.isclose(assignment_fraction(assignments), 0.8)
        ),
        "seed_tokens_distinct": primary_token != independent_token,
        "seed_values_distinct": primary_seed != independent_seed,
        "balanced_design_exact": (
            2 * PER_SEED_STREAMLINES
            == BALANCED_TOTAL_STREAMLINES
        ),
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        parser.error("--workers must be in 1..4")
    if not 1 <= args.threads <= 16:
        parser.error("--threads must be in 1..16")
    if args.self_test:
        value = self_test()
    elif args.execute:
        value = execute(workers=args.workers, threads=args.threads)
    else:
        value = preflight()
    print(json.dumps(value, indent=2, sort_keys=True))
    return (
        0
        if not str(value.get("status", "")).startswith("FAIL")
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
