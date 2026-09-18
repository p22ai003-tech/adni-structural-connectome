#!/usr/bin/env python3
"""Compare GMWMI and dynamic seeding on adverse and passing HCP379 canaries.

This bounded counterfactual generates two independent 3M GMWMI-seeded ACT
tractograms for the lowest-density full-node canary (007) and a high-density
passing control (032).  The final HROI atlas, tractography algorithm, ACT/5TT,
step size, curvature, length limits, FOD cutoff, streamline total, endpoint
assignment rule, and matrix construction are held fixed.

The corresponding balanced 6M dynamic-seed result is reused as the baseline.
The experiment is diagnosis-blind and non-overwriting.  It generates count
matrices only and does not select a production recipe or authorize scale-up.
"""

from __future__ import annotations

import argparse
import hashlib
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
PROMOTED_MASTER = (
    HCP_ROOT
    / "pretract_promoted_recovery4_v6/"
    "pretract_promoted_recovery4_v6_manifest.json"
)
DENSITY_ATTEMPTS = (
    HCP_ROOT / "phase_b_hroi_balanced_density_canary_v1/attempts"
)
ATTEMPTS = (
    HCP_ROOT / "phase_b_hroi_gmwmi_seeding_counterfactual_v1/attempts"
)
MRTRIX = Path("/home/ec2-user/mrtrix3/bin")
TCK_COUNTER = EXP / "scripts/hcp/tck_header_count_v1.py"
PYTHON = Path("/home/ec2-user/fsl/bin/python")
UNITS = ("007_S_5196_I390043", "032_S_6804_I1230908")
ROLE = {
    "007_S_5196_I390043": "LOWEST_DENSITY_FULL_NODE_ADVERSE",
    "032_S_6804_I1230908": "HIGH_DENSITY_PASSING_CONTROL",
}
SEED_CLASSES = ("primary", "independent")
PER_SEED_STREAMLINES = 3_000_000
TOTAL_STREAMLINES = 6_000_000
MAXIMUM_SEEDS = 200_000_000
EXPECTED_NODES = 379


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


DENSITY = load_module(DENSITY_SOURCE, "hcp379_gmwmi_cf_density")
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


def latest_density_summary(unit: str) -> tuple[Path, dict[str, Any]]:
    for path in sorted(
        DENSITY_ATTEMPTS.glob(f"*-{unit}/summary.json"), reverse=True
    ):
        try:
            value = load_json(path)
        except Exception:
            continue
        if (
            value.get("record_type")
            == "diagnosis_blind_hcp379_hroi_balanced_density_canary_summary"
            and value.get("status")
            == "PASS_HROI_BALANCED_DENSITY_CANARY"
            and value.get("unit") == unit
            and set(value.get("levels", {}))
            == {"6000000", "10000000", "15000000", "20000000"}
        ):
            return path.resolve(), value
    raise FileNotFoundError(f"{unit}: complete HROI density summary absent")


def promoted_manifest(unit: str) -> tuple[Path, dict[str, Any]]:
    master = load_json(PROMOTED_MASTER)
    rows = [row for row in master.get("units", []) if row.get("unit") == unit]
    if (
        master.get("record_type")
        != "diagnosis_blind_hcp379_pretract_recovery4_manifest"
        or master.get("status")
        != "AUTOMATED_PASS_AWAITING_HUMAN_VISUAL_QC"
        or master.get("diagnosis_labels_used") is not False
        or len(rows) != 1
    ):
        raise ValueError(f"{unit}: promoted pretract master differs")
    path = Path(rows[0]["manifest"]["path"]).resolve()
    if BALANCED.file_record(path) != rows[0]["manifest"]:
        raise ValueError(f"{unit}: promoted manifest record differs")
    value = load_json(path)
    if (
        value.get("record_type")
        != "diagnosis_blind_hcp379_pretract_recovery4_unit_manifest"
        or value.get("status")
        != "AUTOMATED_PASS_AWAITING_HUMAN_VISUAL_QC"
        or value.get("unit") != unit
        or value.get("diagnosis_labels_used") is not False
    ):
        raise ValueError(f"{unit}: promoted unit manifest differs")
    return path, value


def tck_count(path: Path) -> int:
    completed = subprocess.run(
        [str(PYTHON), str(TCK_COUNTER), str(path)],
        capture_output=True,
        text=True,
        check=True,
    )
    return int(completed.stdout.strip())


def counterfactual_seed(source_seed_id: str) -> tuple[str, int]:
    seed_id = f"{source_seed_id}|gmwmi-counterfactual-v1"
    value = (
        int.from_bytes(
            hashlib.sha256(seed_id.encode("utf-8")).digest()[:4],
            "big",
        )
        & 0x7FFFFFFF
    ) or 1
    return seed_id, value


def bind_unit(unit: str) -> dict[str, Any]:
    summary_path, summary = latest_density_summary(unit)
    manifest_path, manifest = promoted_manifest(unit)
    inputs = manifest.get("tractography_inputs", {})
    wmfod = Path(inputs["wmfod_normalised"]["path"]).resolve()
    five_tt = Path(inputs["five_tt"]["path"]).resolve()
    gmwmi = Path(inputs["gmwmi"]["path"]).resolve()
    atlas_result = load_json(Path(summary["hroi_atlas_result"]["path"]))
    nodes = Path(
        atlas_result["artifacts"]["hcp379_nodes_b0_1mm"]["path"]
    ).resolve()
    phase = DENSITY.phase_paths(unit)
    metadata = {
        seed: load_json(phase[f"{seed}_metadata"])
        for seed in SEED_CLASSES
    }
    seeds = {}
    for seed in SEED_CLASSES:
        source = metadata[seed]
        seed_id, rng_seed = counterfactual_seed(str(source["seed_id"]))
        if (
            source.get("record_type")
            != "hcp379_stability_tractography_parameters"
            or source.get("unit") != unit
            or source.get("seed_class") != seed
            or source.get("algorithm") != "iFOD2"
            or source.get("act") is not True
            or source.get("backtrack") is not True
            or source.get("crop_at_gmwmi") is not True
            or source.get("seeding") != "seed_dynamic"
            or source.get("actual_streamlines") != 10_000_000
            or rng_seed == int(source["rng_seed"])
        ):
            raise ValueError(f"{unit}/{seed}: baseline metadata differs")
        seeds[seed] = {
            "source_metadata": BALANCED.file_record(
                phase[f"{seed}_metadata"]
            ),
            "source_seed_id": source["seed_id"],
            "source_rng_seed": int(source["rng_seed"]),
            "counterfactual_seed_id": seed_id,
            "counterfactual_rng_seed": rng_seed,
            "step_size_mm": float(source["actual_step_size_mm"]),
        }
    baseline = summary["levels"]["6000000"]
    if (
        atlas_result.get("status") != "PASS_HROI_CANARY_ATLAS"
        or atlas_result.get("atlas_qc", {}).get("labels_found")
        != EXPECTED_NODES
        or int(baseline["connected_nodes"]) != EXPECTED_NODES
        or int(baseline["total_streamlines"]) != TOTAL_STREAMLINES
    ):
        raise ValueError(f"{unit}: fixed atlas or 6M baseline differs")
    for path in (
        summary_path,
        manifest_path,
        wmfod,
        five_tt,
        gmwmi,
        nodes,
        MRTRIX / "tckgen",
        MRTRIX / "tck2connectome",
        TCK_COUNTER,
    ):
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"{unit}: regular input differs: {path}")
    return {
        "summary_path": summary_path,
        "summary": summary,
        "manifest_path": manifest_path,
        "wmfod": wmfod,
        "five_tt": five_tt,
        "gmwmi": gmwmi,
        "nodes": nodes,
        "seeds": seeds,
        "baseline": baseline,
    }


def environment(rng_seed: int) -> dict[str, str]:
    value = os.environ.copy()
    value["MRTRIX_RNG_SEED"] = str(rng_seed)
    value["PATH"] = f"{MRTRIX}:{value.get('PATH', '')}"
    return value


def run_logged(
    command: list[str],
    *,
    log: Path,
    env: Mapping[str, str],
    outputs: tuple[Path, ...],
) -> None:
    started = time.monotonic()
    with log.open("x", encoding="utf-8") as handle:
        handle.write("COMMAND " + " ".join(command) + "\n")
        handle.flush()
        completed = subprocess.run(
            command,
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=dict(env),
            check=False,
        )
        handle.write(
            f"\nRETURNCODE {completed.returncode}\n"
            f"ELAPSED_SECONDS {time.monotonic() - started:.3f}\n"
        )
    if completed.returncode != 0:
        raise RuntimeError(f"command failed: {log}")
    for path in outputs:
        if not path.is_file() or path.stat().st_size <= 0:
            raise RuntimeError(f"command output missing: {path}")


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
    run_logged(
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
            "-seed_gmwmi",
            str(bound["gmwmi"]),
            "-select",
            str(PER_SEED_STREAMLINES),
            "-seeds",
            str(MAXIMUM_SEEDS),
            "-minlength",
            "10.0",
            "-maxlength",
            "250.0",
            "-cutoff",
            "0.06",
            "-step",
            f"{seed['step_size_mm']:.8f}",
            "-angle",
            "45.0",
            "-nthreads",
            str(threads),
        ],
        log=output / "tckgen.log",
        env=environment(seed["counterfactual_rng_seed"]),
        outputs=(tracks_partial,),
    )
    if tck_count(tracks_partial) != PER_SEED_STREAMLINES:
        raise ValueError(f"{unit}/{seed_class}: tractogram count differs")
    os.replace(tracks_partial, tracks)
    metadata = output / "tractography_parameters.json"
    atomic_json(
        metadata,
        {
            "schema_version": "1.0.0",
            "record_type": "hcp379_gmwmi_counterfactual_parameters",
            "generated_utc": utc_now(),
            "unit": unit,
            "role": ROLE[unit],
            "seed_class": seed_class,
            "seed_id": seed["counterfactual_seed_id"],
            "rng_seed": seed["counterfactual_rng_seed"],
            "source_seed_id": seed["source_seed_id"],
            "source_rng_seed": seed["source_rng_seed"],
            "algorithm": "iFOD2",
            "act": True,
            "backtrack": True,
            "crop_at_gmwmi": True,
            "seeding": "seed_gmwmi",
            "actual_step_size_mm": seed["step_size_mm"],
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
    assignment_array = BALANCED.load_assignments(assignments)
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


def preflight() -> dict[str, Any]:
    bound = {unit: bind_unit(unit) for unit in UNITS}
    return {
        "schema_version": "1.0.0",
        "record_type": (
            "diagnosis_blind_hcp379_gmwmi_seeding_counterfactual_preflight"
        ),
        "generated_utc": utc_now(),
        "status": "READY_GMWMI_SEEDING_COUNTERFACTUAL",
        "diagnosis_labels_used": False,
        "outcomes_used": False,
        "non_overwriting": True,
        "units": list(UNITS),
        "roles": ROLE,
        "changed_factor": "seed_dynamic_to_seed_gmwmi_only",
        "per_seed_streamlines": PER_SEED_STREAMLINES,
        "balanced_total_streamlines": TOTAL_STREAMLINES,
        "dynamic_baseline_density": {
            unit: data["baseline"]["edge_density"]
            for unit, data in bound.items()
        },
        "imaging_executed": False,
        "sift2_or_scalar_matrices_generated": False,
        "production_recipe_selected": False,
        "scale_up_authorized": False,
    }


def execute(workers: int, threads: int) -> dict[str, Any]:
    bound = {unit: bind_unit(unit) for unit in UNITS}
    attempt = ATTEMPTS / f"{stamp()}-gmwmi-seeding-counterfactual"
    attempt.mkdir(parents=True, exist_ok=False)
    preflight_path = attempt / "preflight.json"
    atomic_json(
        preflight_path,
        {
            **preflight(),
            "status": "EXECUTION_INPUTS_BOUND",
            "imaging_executed": True,
            "implementation": BALANCED.file_record(Path(__file__)),
            "inputs": {
                unit: {
                    "density_summary": BALANCED.file_record(
                        data["summary_path"]
                    ),
                    "promoted_pretract_manifest": BALANCED.file_record(
                        data["manifest_path"]
                    ),
                    "wmfod": BALANCED.file_record(data["wmfod"]),
                    "five_tt": BALANCED.file_record(data["five_tt"]),
                    "gmwmi": BALANCED.file_record(data["gmwmi"]),
                    "hroi_nodes": BALANCED.file_record(data["nodes"]),
                    "seeds": data["seeds"],
                }
                for unit, data in bound.items()
            },
        },
    )
    state_path = attempt / "state.json"
    state: dict[str, Any] = {
        "status": "RUNNING",
        "generated_utc": utc_now(),
        "completed_jobs": [],
        "remaining_jobs": [
            f"{unit}/{seed}" for unit in UNITS for seed in SEED_CLASSES
        ],
    }
    atomic_json(state_path, state)
    results: dict[tuple[str, str], dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_map = {
            pool.submit(
                run_seed,
                unit,
                seed,
                bound[unit],
                attempt,
                threads,
            ): (unit, seed)
            for unit in UNITS
            for seed in SEED_CLASSES
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
        matrices = {
            seed: BALANCED.load_count(
                Path(results[(unit, seed)]["count_matrix"]["path"])
            )
            for seed in SEED_CLASSES
        }
        combined = matrices["primary"] + matrices["independent"]
        combined_path = attempt / unit / "balanced_6m_count.csv"
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
            for seed in SEED_CLASSES
        ) / 2.0
        units[unit] = {
            "role": ROLE[unit],
            "dynamic_seed_baseline": {
                "edge_density": baseline_density,
                "connected_nodes": int(baseline["connected_nodes"]),
                "endpoint_assignment_fraction": baseline_assignment,
                "count_matrix": baseline["count_matrix"],
            },
            "gmwmi_counterfactual": {
                **qc,
                "endpoint_assignment_fraction": gmwmi_assignment,
                "support_complementarity": (
                    DENSITY.support_complementarity(
                        matrices["primary"], matrices["independent"]
                    )
                ),
                "count_matrix": BALANCED.file_record(combined_path),
                "seeds": {
                    seed: results[(unit, seed)]
                    for seed in SEED_CLASSES
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
    summary = {
        "schema_version": "1.0.0",
        "record_type": (
            "diagnosis_blind_hcp379_gmwmi_seeding_counterfactual"
        ),
        "generated_utc": utc_now(),
        "status": "PASS_COMPLETE_GMWMI_SEEDING_COUNTERFACTUAL",
        "diagnosis_labels_used": False,
        "outcomes_used": False,
        "non_overwriting": True,
        "source_files_modified": False,
        "changed_factor": "seed_dynamic_to_seed_gmwmi_only",
        "new_tractography_generated": True,
        "generated_streamlines": len(UNITS)
        * len(SEED_CLASSES)
        * PER_SEED_STREAMLINES,
        "sift2_or_scalar_matrices_generated": False,
        "units": units,
        "production_recipe_selected": False,
        "scale_up_authorized": False,
        "requires_expansion_to_remaining_low_density_canaries": True,
        "requires_all_nine_and_seed_stability_before_selection": True,
        "preflight": BALANCED.file_record(preflight_path),
        "implementation": BALANCED.file_record(Path(__file__)),
    }
    atomic_json(attempt / "summary.json", summary)
    state.update(
        {
            "status": summary["status"],
            "generated_utc": utc_now(),
            "summary": BALANCED.file_record(attempt / "summary.json"),
        }
    )
    atomic_json(state_path, state)
    return {
        "attempt_root": str(attempt),
        "status": summary["status"],
        "units": {
            unit: {
                "role": row["role"],
                "dynamic_density": row["dynamic_seed_baseline"][
                    "edge_density"
                ],
                "gmwmi_density": row["gmwmi_counterfactual"][
                    "edge_density"
                ],
                "density_difference": row["difference"]["edge_density"],
                "dynamic_connected_nodes": row[
                    "dynamic_seed_baseline"
                ]["connected_nodes"],
                "gmwmi_connected_nodes": row["gmwmi_counterfactual"][
                    "connected_nodes"
                ],
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
