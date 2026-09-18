#!/usr/bin/env python3
"""Test balanced 6M density on the known low-density 135 canary.

The existing exact primary 10M tractogram and its corrected-HROI endpoint
assignments are reused.  Only one isolated, independent-seed 3M tractogram is
generated, then reassigned to the same corrected HROI atlas.  The first 3M
primary assignments and the new independent 3M assignments are combined to
test the balanced 6M count-density gate.

This targeted adverse-case calibration is non-overwriting and diagnosis
blind.  It does not generate SIFT2/scalar matrices, select a cohort recipe,
or authorize scale-up.
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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np


EXP = Path("/home/ec2-user/exp")
HCP_ROOT = Path("/data/derivatives/hcp379_v2")
UNIT = "135_S_6840_I1263427"
RUN_UNIT = Path(
    "/data/derivatives/scforge_v2/"
    "h04a_r1_recovery_20260719_retry4/subjects"
) / UNIT
PRIMARY_ROOT = RUN_UNIT / "09_hcp379_stability/primary_10m"
PRIMARY_TRACKS = PRIMARY_ROOT / "tracks.tck"
PRIMARY_METADATA = PRIMARY_ROOT / "tractography_parameters.json"
WMFOD = RUN_UNIT / "05_model/wmfod_norm.mif"
FIVE_TT = RUN_UNIT / "05_model/5tt_dwi_recovery3.mif"
HROI_VALIDATION = Path(
    "/data/derivatives/hcp379_v2/source_label_repair_v1/validation/"
    "hroi_surface_support_corrected_act_v1/attempts/"
    "20260728T055524.033287Z-hroi-corrected-act-validation/summary.json"
)
TCK_COUNTER = EXP / "scripts/hcp/tck_header_count_v1.py"
BALANCED_SOURCE = (
    EXP / "scripts/hcp/build_hcp379_balanced_seed_density_projection_v1.py"
)
MRTRIX = Path("/home/ec2-user/mrtrix3/bin")
PYTHON = Path("/home/ec2-user/fsl/bin/python")
ATTEMPTS = (
    HCP_ROOT
    / "phase_b_hroi_low_density_135_6m_canary_v1/attempts"
)
PRIMARY_PREFIX = 3_000_000
INDEPENDENT_STREAMLINES = 3_000_000
TOTAL_STREAMLINES = 6_000_000
MAXIMUM_SEEDS = 200_000_000
DENSITY_TARGET = 0.60


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BALANCED = load_module(BALANCED_SOURCE, "hcp379_low_density_135_base")


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


def tck_count(path: Path) -> int:
    result = subprocess.run(
        [str(PYTHON), str(TCK_COUNTER), str(path)],
        capture_output=True,
        text=True,
        check=True,
    )
    return int(result.stdout.strip())


def independent_seed(primary_seed_id: str) -> tuple[str, int]:
    seed_id = f"{primary_seed_id}|independent-replicate-1"
    seed = (
        int.from_bytes(
            hashlib.sha256(seed_id.encode("utf-8")).digest()[:4],
            "big",
        )
        & 0x7FFFFFFF
    ) or 1
    return seed_id, seed


def inputs() -> dict[str, Any]:
    validation = load_json(HROI_VALIDATION)
    primary = load_json(PRIMARY_METADATA)
    nodes = Path(
        str(validation.get("outputs", {}).get("candidate_nodes", {}).get("path", ""))
    ).resolve()
    assignments = Path(
        str(
            validation.get("outputs", {})
            .get("candidate_assignments", {})
            .get("path", "")
        )
    ).resolve()
    if (
        validation.get("status")
        != "TECHNICAL_PASS_CORRECTED_ACT_SCIENTIFIC_POLICY_REVIEW_PENDING"
        or validation.get("diagnosis_labels_used") is not False
        or validation.get("unit") != UNIT
        or validation.get("streamline_count") != 10_000_000
        or validation.get("atlas", {}).get(
            "all_379_candidate_nodes_present"
        )
        is not True
        or primary.get("record_type")
        != "hcp379_stability_tractography_parameters"
        or primary.get("diagnosis_labels_used") is not False
        or primary.get("unit") != UNIT
        or primary.get("run_id") != "primary_10m"
        or primary.get("seed_class") != "primary"
        or int(primary.get("actual_streamlines", -1)) != 10_000_000
        or tck_count(PRIMARY_TRACKS) != 10_000_000
    ):
        raise ValueError("low-density primary/HROI evidence differs")
    for path in (
        PRIMARY_TRACKS,
        PRIMARY_METADATA,
        WMFOD,
        FIVE_TT,
        nodes,
        assignments,
        MRTRIX / "tckgen",
        MRTRIX / "tck2connectome",
        TCK_COUNTER,
    ):
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"required regular input differs: {path}")
    seed_id, seed = independent_seed(str(primary["seed_id"]))
    if seed == int(primary["rng_seed"]):
        raise ValueError("independent seed equals primary seed")
    return {
        "validation": validation,
        "primary_metadata": primary,
        "nodes": nodes,
        "primary_assignments": assignments,
        "independent_seed_id": seed_id,
        "independent_rng_seed": seed,
        "step_size_mm": float(primary["actual_step_size_mm"]),
    }


def environment(seed: int) -> dict[str, str]:
    value = os.environ.copy()
    value["MRTRIX_RNG_SEED"] = str(seed)
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
        result = subprocess.run(
            command,
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=dict(env),
            check=False,
        )
        handle.write(
            f"\nRETURNCODE {result.returncode}\n"
            f"ELAPSED_SECONDS {time.monotonic() - started:.3f}\n"
        )
    if result.returncode != 0:
        raise RuntimeError(f"command failed: {log}")
    for path in outputs:
        if not path.is_file() or path.stat().st_size <= 0:
            raise RuntimeError(f"command output missing: {path}")


def support(primary: np.ndarray, independent: np.ndarray) -> dict[str, Any]:
    tri = np.triu_indices(379, 1)
    first = primary[tri] > 0
    second = independent[tri] > 0
    union = first | second
    overlap = first & second
    union_n = int(np.count_nonzero(union))
    return {
        "primary_supported_edges": int(np.count_nonzero(first)),
        "independent_supported_edges": int(np.count_nonzero(second)),
        "overlap_supported_edges": int(np.count_nonzero(overlap)),
        "primary_only_supported_edges": int(
            np.count_nonzero(first & ~second)
        ),
        "independent_only_supported_edges": int(
            np.count_nonzero(second & ~first)
        ),
        "union_supported_edges": union_n,
        "support_jaccard": (
            int(np.count_nonzero(overlap)) / union_n if union_n else 0.0
        ),
    }


def execute(threads: int) -> dict[str, Any]:
    bound = inputs()
    attempt = (
        ATTEMPTS / f"{stamp()}-low-density-135-balanced-6m"
    )
    attempt.mkdir(parents=True, exist_ok=False)
    preflight = {
        "schema_version": "1.0.0",
        "record_type": (
            "diagnosis_blind_hcp379_hroi_low_density_135_"
            "balanced_6m_preflight"
        ),
        "status": "EXECUTION_INPUTS_BOUND",
        "generated_utc": utc_now(),
        "unit": UNIT,
        "diagnosis_labels_used": False,
        "known_primary_hroi_density": float(
            bound["validation"]["topology"]["density_candidate"]
        ),
        "primary_prefix_streamlines": PRIMARY_PREFIX,
        "independent_streamlines_to_generate": INDEPENDENT_STREAMLINES,
        "balanced_total_streamlines": TOTAL_STREAMLINES,
        "tractography_scope": "one isolated independent 3M canary only",
        "cohort_recipe_selected": False,
        "scale_up_authorized": False,
        "inputs": {
            "primary_tracks": BALANCED.file_record(PRIMARY_TRACKS),
            "primary_metadata": BALANCED.file_record(PRIMARY_METADATA),
            "wmfod": BALANCED.file_record(WMFOD),
            "five_tt": BALANCED.file_record(FIVE_TT),
            "hroi_nodes": BALANCED.file_record(bound["nodes"]),
            "primary_hroi_assignments": BALANCED.file_record(
                bound["primary_assignments"]
            ),
            "hroi_validation": BALANCED.file_record(HROI_VALIDATION),
        },
        "independent_seed_id": bound["independent_seed_id"],
        "independent_rng_seed": bound["independent_rng_seed"],
        "implementation": BALANCED.file_record(Path(__file__)),
    }
    atomic_json(attempt / "preflight.json", preflight)

    output = attempt / "independent_3m"
    output.mkdir(parents=True, exist_ok=False)
    tracks_partial = output / "tracks.partial.tck"
    tracks = output / "tracks.tck"
    metadata = output / "tractography_parameters.json"
    run_logged(
        [
            str(MRTRIX / "tckgen"),
            str(WMFOD),
            str(tracks_partial),
            "-algorithm",
            "iFOD2",
            "-act",
            str(FIVE_TT),
            "-backtrack",
            "-crop_at_gmwmi",
            "-seed_dynamic",
            str(WMFOD),
            "-select",
            str(INDEPENDENT_STREAMLINES),
            "-seeds",
            str(MAXIMUM_SEEDS),
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
        env=environment(bound["independent_rng_seed"]),
        outputs=(tracks_partial,),
    )
    if tck_count(tracks_partial) != INDEPENDENT_STREAMLINES:
        raise ValueError("independent 3M tractogram count differs")
    os.replace(tracks_partial, tracks)
    atomic_json(
        metadata,
        {
            "schema_version": "1.0.0",
            "record_type": (
                "hcp379_stability_tractography_parameters"
            ),
            "diagnosis_labels_used": False,
            "unit": UNIT,
            "run_id": "independent_3m_low_density_calibration",
            "seed_class": "independent",
            "seed_id": bound["independent_seed_id"],
            "rng_seed": bound["independent_rng_seed"],
            "construction": "isolated_adverse_case_calibration",
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
    run_logged(
        [
            str(MRTRIX / "tck2connectome"),
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
        env=environment(bound["independent_rng_seed"]),
        outputs=(count_partial, assignments_partial),
    )
    os.replace(count_partial, count_path)
    os.replace(assignments_partial, assignments_path)

    primary_assignments = BALANCED.load_assignments(
        bound["primary_assignments"]
    )
    independent_assignments = np.loadtxt(
        assignments_path, comments="#", dtype=np.int16
    )
    if independent_assignments.shape != (INDEPENDENT_STREAMLINES, 2):
        raise ValueError("independent 3M assignment shape differs")
    primary_matrix = BALANCED.count_matrix(
        primary_assignments, PRIMARY_PREFIX
    )
    independent_matrix = BALANCED.count_matrix(
        independent_assignments, INDEPENDENT_STREAMLINES
    )
    recorded_independent = BALANCED.load_count(count_path)
    if not np.array_equal(independent_matrix, recorded_independent):
        raise ValueError("independent assignment replay differs")
    combined = primary_matrix + independent_matrix
    combined_path = attempt / "balanced_6m_count.csv"
    BALANCED.atomic_matrix(combined_path, combined)
    qc = BALANCED.matrix_qc(combined)
    summary = {
        "schema_version": "1.0.0",
        "record_type": (
            "diagnosis_blind_hcp379_hroi_low_density_135_"
            "balanced_6m_summary"
        ),
        "status": "PASS_LOW_DENSITY_135_6M_CALIBRATION",
        "generated_utc": utc_now(),
        "unit": UNIT,
        "diagnosis_labels_used": False,
        "non_overwriting": True,
        "source_files_modified": False,
        "primary_tractography_reused": True,
        "independent_tractography_generated": True,
        "generated_independent_streamlines": INDEPENDENT_STREAMLINES,
        "balanced_total_streamlines": TOTAL_STREAMLINES,
        "primary_endpoint_assignment_fraction": (
            BALANCED.assignment_fraction(
                primary_assignments, PRIMARY_PREFIX
            )
        ),
        "independent_endpoint_assignment_fraction": (
            BALANCED.assignment_fraction(
                independent_assignments, INDEPENDENT_STREAMLINES
            )
        ),
        "support_complementarity": support(
            primary_matrix, independent_matrix
        ),
        "count_matrix": BALANCED.file_record(combined_path),
        **qc,
        "density_target_0_60_met": (
            float(qc["edge_density"]) >= DENSITY_TARGET
        ),
        "cohort_recipe_selected": False,
        "scale_up_authorized": False,
        "requires_complete_15_canary_family": True,
        "requires_joint_all_nine_matrix_stability": True,
        "artifacts": {
            "independent_tracks": BALANCED.file_record(tracks),
            "independent_metadata": BALANCED.file_record(metadata),
            "independent_hroi_assignments": BALANCED.file_record(
                assignments_path
            ),
            "independent_hroi_count": BALANCED.file_record(count_path),
            "preflight": BALANCED.file_record(attempt / "preflight.json"),
        },
        "implementation": BALANCED.file_record(Path(__file__)),
    }
    atomic_json(attempt / "summary.json", summary)
    return {
        "attempt_root": str(attempt),
        "status": summary["status"],
        "edge_density": summary["edge_density"],
        "density_target_0_60_met": summary[
            "density_target_0_60_met"
        ],
    }


def self_test() -> dict[str, Any]:
    seed_id, seed = independent_seed(
        "2a1ef03090b3427d0a6b27f3b8e94b2f58e0026df856ef5dadb57c8081483b99"
        "|77afe0d1aa521d98386a0f7639e2be07377a6adcbb3dcc8e6dbc7105a0f9e093"
        "|connectome-v2.1.0-closed-world-canary-candidate"
    )
    checks = {
        "known_independent_seed": seed == 2106283499,
        "independent_suffix": seed_id.endswith(
            "|independent-replicate-1"
        ),
        "balanced_total": PRIMARY_PREFIX + INDEPENDENT_STREAMLINES
        == TOTAL_STREAMLINES,
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.threads <= 16:
        parser.error("--threads must be in 1..16")
    if args.self_test:
        value = self_test()
        print(json.dumps(value, sort_keys=True))
        return 0 if value["status"] == "PASS" else 1
    bound = inputs()
    preflight = {
        "status": "READY_LOW_DENSITY_135_BALANCED_6M_CANARY",
        "unit": UNIT,
        "diagnosis_labels_used": False,
        "known_primary_hroi_density": float(
            bound["validation"]["topology"]["density_candidate"]
        ),
        "primary_prefix_streamlines": PRIMARY_PREFIX,
        "independent_streamlines_to_generate": INDEPENDENT_STREAMLINES,
        "balanced_total_streamlines": TOTAL_STREAMLINES,
        "imaging_executed": False,
        "cohort_recipe_selected": False,
        "scale_up_authorized": False,
    }
    if not args.execute:
        print(json.dumps(preflight, indent=2, sort_keys=True))
        return 0
    result = execute(args.threads)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
