#!/usr/bin/env python3
"""Build all nine matrices for one selected balanced HROI canary.

The density calibration defines a balanced construction as equal deterministic
prefixes from the exact primary and independent 10M Phase-B tractograms.  This
runner applies that construction to the final corrected HROI atlas and builds
three views:

* the primary half for seed-reliability assessment;
* the independent half for seed-reliability assessment; and
* the concatenated balanced tractogram used for the candidate connectome.

SIFT2 is recomputed separately for every exact view.  It is deliberately not
sliced from a larger run and two independently fitted SIFT2 matrices are not
summed.  No probabilistic tractography is generated, no source artifact is
modified, and a single-canary result cannot authorize cohort scale-up.
"""

from __future__ import annotations

import argparse
import csv
import functools
import hashlib
import importlib.util
import json
import os
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


EXP = Path("/home/ec2-user/exp")
HCP_ROOT = Path("/data/derivatives/hcp379_v2")
PHASE_SUBJECT_ROOT = Path(
    "/data/derivatives/scforge_v2/"
    "h04a_r1_recovery_20260719_retry4/subjects"
)
MRTRIX = Path("/home/ec2-user/mrtrix3/bin")
FSL = Path("/home/ec2-user/fsl/bin")
PYTHON = FSL / "python"
TCK_COUNTER = EXP / "scripts/hcp/tck_header_count_v1.py"
NUMERIC_COUNTER = (
    EXP / "scripts/hcp/count_mrtrix_numeric_values_v1.py"
)
DENSITY_SINGLE_SOURCE = (
    EXP
    / "scripts/hcp/"
    "build_hcp379_hroi_balanced_seed_density_canary_v1.py"
)
TECHNICAL_VALIDATOR_SOURCE = (
    EXP / "research_audit/validate_hcp379_phase_b_stability.py"
)
CONTRACT = (
    EXP
    / "research_audit/outputs/thesis_grade_530_release_contract_v1/"
    "contract.json"
)
HROI_POLICY_RECEIPT = (
    HCP_ROOT
    / "source_label_repair_v1/hroi_surface_support_cohort_v1/"
    "policy_adoption_v1.json"
)
HROI_ATLAS_SUMMARY = (
    HCP_ROOT
    / "source_label_repair_v1/hroi_canary_atlas_v1/attempts/"
    "20260728T081233.507961Z-hroi-canary-atlas/summary.json"
)
PROMOTED_PRETRACT_MASTER = (
    HCP_ROOT
    / "pretract_promoted_recovery4_v6/"
    "pretract_promoted_recovery4_v6_manifest.json"
)
DENSITY_COHORT_ATTEMPTS = (
    HCP_ROOT
    / "phase_b_hroi_balanced_density_canary_cohort_v1/attempts"
)
ATTEMPTS = (
    HCP_ROOT
    / "phase_b_hroi_balanced_all_nine_canary_v1/attempts"
)
EXPECTED_NODES = 379
EXPECTED_CANARIES = 15
EXPECTED_SOURCE_STREAMLINES = 10_000_000
DENSITY_TARGET = 0.60
BALANCED_TOTALS = {
    6_000_000: 3_000_000,
    10_000_000: 5_000_000,
    15_000_000: 7_500_000,
    20_000_000: 10_000_000,
}
MATRIX_NAMES = (
    "count",
    "fd_sum",
    "count_invnodevol",
    "len_mean",
    "invlen_mean",
    "fa_mean",
    "md_mean",
    "rd_mean",
    "ad_mean",
)
SCALAR_METRICS = ("fa", "md", "rd", "ad")
VIEW_NAMES = ("primary", "independent", "balanced")


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


DENSITY_SINGLE = load_module(
    DENSITY_SINGLE_SOURCE, "hcp379_hroi_all_nine_density_single"
)
TECHNICAL = load_module(
    TECHNICAL_VALIDATOR_SOURCE, "hcp379_hroi_all_nine_technical"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def attempt_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root is not an object: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(path)
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def verify_record(record: Mapping[str, Any], *, label: str) -> Path:
    path = Path(str(record.get("path", ""))).resolve()
    if (
        not path.is_file()
        or path.is_symlink()
        or record.get("path") != str(path)
        or record.get("size_bytes") != path.stat().st_size
        or record.get("sha256") != sha256(path)
    ):
        raise ValueError(f"{label}: file record differs")
    return path


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
    temporary = path.with_name(f".{path.stem}.partial{path.suffix}")
    np.savetxt(temporary, value, delimiter=",", fmt="%.12g")
    os.replace(temporary, path)


def environment() -> dict[str, str]:
    value = os.environ.copy()
    value["PATH"] = (
        f"{MRTRIX}:{FSL}:{value.get('PATH', '')}"
    )
    return value


def run_logged(
    command: Sequence[str],
    *,
    log: Path,
    outputs: Iterable[Path],
) -> None:
    output_list = list(outputs)
    log.parent.mkdir(parents=True, exist_ok=True)
    for output in output_list:
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            raise FileExistsError(output)
    started = time.monotonic()
    with log.open("x", encoding="utf-8") as handle:
        handle.write("COMMAND " + " ".join(command) + "\n")
        handle.flush()
        completed = subprocess.run(
            list(command),
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
        raise RuntimeError(f"command failed; see {log}")
    for output in output_list:
        if not output.is_file() or output.stat().st_size <= 0:
            raise RuntimeError(f"expected output is absent: {output}")


def tck_count(path: Path, expected: int | None = None) -> int:
    command = [str(PYTHON), str(TCK_COUNTER), str(path)]
    if expected is not None:
        command.extend(["--expect", str(expected)])
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    return int(completed.stdout.strip())


def numeric_count(path: Path, expected: int) -> int:
    completed = subprocess.run(
        [
            str(PYTHON),
            str(NUMERIC_COUNTER),
            str(path),
            "--expect",
            str(expected),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return int(completed.stdout.strip())


def matrix(path: Path) -> np.ndarray:
    value = np.loadtxt(path, delimiter=",")
    if (
        value.shape != (EXPECTED_NODES, EXPECTED_NODES)
        or not np.isfinite(value).all()
        or np.any(value < 0)
        or not np.allclose(value, value.T, atol=1.0e-8, rtol=1.0e-8)
        or not np.allclose(np.diag(value), 0.0, atol=1.0e-8, rtol=0.0)
    ):
        raise ValueError(f"matrix invariants differ: {path}")
    return value


def assignment_summary(path: Path, expected: int) -> dict[str, Any]:
    observed = 0
    assigned = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            fields = stripped.replace(",", " ").split()
            if len(fields) != 2:
                raise ValueError(
                    f"{path}: assignment width differs at {line_number}"
                )
            left, right = (int(value) for value in fields)
            if not 0 <= left <= EXPECTED_NODES or not 0 <= right <= EXPECTED_NODES:
                raise ValueError(
                    f"{path}: endpoint outside 0..379 at {line_number}"
                )
            observed += 1
            assigned += int(left > 0 and right > 0)
    if observed != expected:
        raise ValueError(
            f"{path}: assignment rows {observed} differ from {expected}"
        )
    return {
        "assignment_row_count": observed,
        "assigned_streamline_count": assigned,
        "endpoint_assignment_fraction": assigned / observed,
    }


def atlas_contract() -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    summary = load_json(HROI_ATLAS_SUMMARY)
    rows = summary.get("units", [])
    by_unit = {
        str(row.get("unit")): row
        for row in rows
        if isinstance(row, Mapping)
    }
    if (
        summary.get("record_type") != "hcp379_hroi_canary_atlas_summary"
        or summary.get("status") != "PASS_HROI_CANARY_ATLAS_15"
        or summary.get("diagnosis_labels_used") is not False
        or summary.get("non_overwriting") is not True
        or len(rows) != EXPECTED_CANARIES
        or len(by_unit) != EXPECTED_CANARIES
        or any(
            row.get("status") != "PASS_HROI_CANARY_ATLAS"
            or row.get("labels_found") != EXPECTED_NODES
            for row in rows
        )
    ):
        raise ValueError("fixed HROI atlas canary summary differs")
    return summary, by_unit


def atlas_inputs(unit: str) -> dict[str, Path]:
    _, by_unit = atlas_contract()
    if unit not in by_unit:
        raise ValueError(f"unit is absent from HROI atlas summary: {unit}")
    result_path = verify_record(
        by_unit[unit]["result"], label=f"{unit}/HROI atlas result"
    )
    result = load_json(result_path)
    if (
        result.get("status") != "PASS_HROI_CANARY_ATLAS"
        or result.get("unit") != unit
        or result.get("diagnosis_labels_used") is not False
        or result.get("matrix_generation_started") is not False
    ):
        raise ValueError(f"{unit}: HROI atlas result differs")
    artifacts = result["artifacts"]
    return {
        "result": result_path,
        "nodes": verify_record(
            artifacts["hcp379_nodes_b0_1mm"],
            label=f"{unit}/HROI nodes",
        ),
        "node_volumes": verify_record(
            artifacts["hcp379_node_volumes"],
            label=f"{unit}/HROI node volumes",
        ),
        "atlas_qc": verify_record(
            artifacts["hcp379_atlas_qc"],
            label=f"{unit}/HROI atlas QC",
        ),
    }


def valid_density_summary(path: Path) -> dict[str, Any]:
    value = load_json(path)
    selected = value.get(
        "smallest_density_qualified_balanced_total_count"
    )
    qualified = value.get(
        "density_qualified_balanced_total_counts", []
    )
    rows = value.get("units", [])
    if (
        value.get("record_type")
        != "diagnosis_blind_hcp379_hroi_balanced_density_canary_cohort_summary"
        or value.get("status")
        != "PASS_COMPLETE_COUNT_ONLY_DENSITY_EVIDENCE"
        or value.get("diagnosis_labels_used") is not False
        or value.get("processed_unit_n") != EXPECTED_CANARIES
        or value.get("failure_n") != 0
        or value.get("count_only_projection") is not True
        or value.get("requires_joint_all_nine_matrix_stability") is not True
        or selected not in BALANCED_TOTALS
        or not isinstance(qualified, list)
        or not qualified
        or qualified != sorted(set(qualified))
        or any(total not in BALANCED_TOTALS for total in qualified)
        or min(qualified) != selected
        or len(rows) != EXPECTED_CANARIES
        or any(
            not row.get("levels", {}).get(str(total), {}).get(
                "density_target_0_60_met", False
            )
            for row in rows
            for total in qualified
        )
    ):
        raise ValueError("HROI density-cohort summary is not selectable")
    return value


def latest_density_summary() -> Path | None:
    for path in sorted(
        DENSITY_COHORT_ATTEMPTS.glob("*/summary.json"), reverse=True
    ):
        try:
            valid_density_summary(path)
        except Exception:
            continue
        return path.resolve()
    return None


@functools.lru_cache(maxsize=1)
def promoted_pretract_by_unit() -> dict[str, dict[str, Any]]:
    master = load_json(PROMOTED_PRETRACT_MASTER)
    rows = master.get("units", [])
    if (
        master.get("record_type")
        != "diagnosis_blind_hcp379_pretract_recovery4_manifest"
        or master.get("status")
        != "AUTOMATED_PASS_AWAITING_HUMAN_VISUAL_QC"
        or master.get("diagnosis_labels_used") is not False
        or master.get("human_visual_qc_inferred") is not False
        or master.get("unit_count") != EXPECTED_CANARIES
        or len(rows) != EXPECTED_CANARIES
    ):
        raise ValueError("promoted Recovery4 pretract master differs")
    by_unit: dict[str, dict[str, Any]] = {}
    for row in rows:
        unit = str(row.get("unit", ""))
        manifest_path = verify_record(
            row.get("manifest", {}),
            label=f"{unit}/promoted Recovery4 manifest",
        )
        manifest = load_json(manifest_path)
        if (
            not unit
            or unit in by_unit
            or manifest.get("record_type")
            != "diagnosis_blind_hcp379_pretract_recovery4_unit_manifest"
            or manifest.get("status")
            != "AUTOMATED_PASS_AWAITING_HUMAN_VISUAL_QC"
            or manifest.get("diagnosis_labels_used") is not False
            or manifest.get("human_visual_qc_inferred") is not False
            or manifest.get("unit") != unit
        ):
            raise ValueError(
                f"{unit}: promoted Recovery4 pretract manifest differs"
            )
        by_unit[unit] = manifest
    if len(by_unit) != EXPECTED_CANARIES:
        raise ValueError("promoted Recovery4 canary unit set differs")
    return by_unit


@functools.lru_cache(maxsize=EXPECTED_CANARIES)
def phase_inputs(unit: str) -> dict[str, Path]:
    root = PHASE_SUBJECT_ROOT / unit
    stability = root / "09_hcp379_stability"
    manifest = promoted_pretract_by_unit()[unit]
    tractography = manifest["tractography_inputs"]
    scalars = manifest["tensor_maps_bounded"]
    return {
        "primary_tracks": stability / "primary_10m/tracks.tck",
        "primary_metadata": (
            stability / "primary_10m/tractography_parameters.json"
        ),
        "independent_tracks": stability / "independent_10m/tracks.tck",
        "independent_metadata": (
            stability / "independent_10m/tractography_parameters.json"
        ),
        "wmfod": verify_record(
            tractography["wmfod_normalised"],
            label=f"{unit}/promoted Recovery4 wmfod",
        ),
        "five_tt": verify_record(
            tractography["five_tt"],
            label=f"{unit}/promoted Recovery4 five_tt",
        ),
        "fa": verify_record(
            scalars["fa"], label=f"{unit}/promoted Recovery4 fa"
        ),
        "md": verify_record(
            scalars["md"], label=f"{unit}/promoted Recovery4 md"
        ),
        "rd": verify_record(
            scalars["rd"], label=f"{unit}/promoted Recovery4 rd"
        ),
        "ad": verify_record(
            scalars["ad"], label=f"{unit}/promoted Recovery4 ad"
        ),
    }


def phase_readiness(unit: str) -> dict[str, Any]:
    paths = phase_inputs(unit)
    missing = [
        name
        for name, path in paths.items()
        if not path.is_file() or path.is_symlink()
    ]
    if missing:
        return {
            "ready": False,
            "missing": missing,
            "paths": {name: str(path) for name, path in paths.items()},
        }
    primary = DENSITY_SINGLE.BALANCED.validate_metadata(
        paths["primary_metadata"], "primary"
    )
    independent = DENSITY_SINGLE.BALANCED.validate_metadata(
        paths["independent_metadata"], "independent"
    )
    if primary.get("seed_id") == independent.get("seed_id"):
        raise ValueError(f"{unit}: seed identities match")
    counts = {
        seed: tck_count(paths[f"{seed}_tracks"])
        for seed in ("primary", "independent")
    }
    if set(counts.values()) != {EXPECTED_SOURCE_STREAMLINES}:
        raise ValueError(f"{unit}: source tractogram counts differ")
    return {
        "ready": True,
        "missing": [],
        "paths": {name: str(path) for name, path in paths.items()},
        "counts": counts,
        "seed_ids": {
            "primary": primary["seed_id"],
            "independent": independent["seed_id"],
        },
    }


def preflight(
    unit: str,
    density_summary: Path | None,
    requested_total: int | None = None,
) -> dict[str, Any]:
    _, by_unit = atlas_contract()
    pair = phase_readiness(unit)
    density_error = None
    density = None
    if density_summary is not None:
        try:
            density = valid_density_summary(density_summary)
        except Exception as exc:
            density_error = f"{type(exc).__name__}:{exc}"
    qualified = (
        list(density.get("density_qualified_balanced_total_counts", []))
        if density is not None
        else []
    )
    selected = (
        requested_total
        if requested_total is not None
        else (
            density.get(
                "smallest_density_qualified_balanced_total_count"
            )
            if density is not None
            else None
        )
    )
    selection_error = None
    if requested_total is not None and requested_total not in qualified:
        selection_error = (
            "requested total is not density-qualified across all 15"
        )
    ready = (
        unit in by_unit
        and pair["ready"]
        and density is not None
        and selected in BALANCED_TOTALS
        and selected in qualified
        and selection_error is None
    )
    return {
        "schema_version": "1.0.0",
        "record_type": (
            "diagnosis_blind_hcp379_hroi_balanced_all_nine_canary_preflight"
        ),
        "generated_utc": utc_now(),
        "status": (
            "READY_HROI_BALANCED_ALL_NINE_CANARY"
            if ready
            else "WAITING_FOR_COMPLETE_DENSITY_OR_EXACT_PAIR"
        ),
        "unit": unit,
        "diagnosis_labels_used": False,
        "non_overwriting": True,
        "source_files_modified": False,
        "tractography_generated": False,
        "only_deterministic_prefix_and_concatenation_planned": True,
        "all_nine_matrix_generation_started": False,
        "density_summary": (
            str(density_summary.resolve())
            if density_summary is not None
            else None
        ),
        "density_summary_error": density_error,
        "density_qualified_balanced_total_counts": qualified,
        "selected_balanced_total_streamlines": selected,
        "selected_candidate_rank": (
            qualified.index(selected) + 1 if selected in qualified else None
        ),
        "selection_mode": (
            "EXPLICIT_UNIFORM_DENSITY_QUALIFIED_CANDIDATE"
            if requested_total is not None
            else "SMALLEST_DENSITY_QUALIFIED_CANDIDATE"
        ),
        "selection_error": selection_error,
        "selected_per_seed_prefix_streamlines": (
            BALANCED_TOTALS.get(selected)
        ),
        "phase_pair": pair,
        "fixed_hroi_atlas_ready": unit in by_unit,
        "sift2_will_be_fit_on_each_exact_view": True,
        "independent_seed_stability_required": True,
        "combined_density_target": DENSITY_TARGET,
        "matrix_names": list(MATRIX_NAMES),
        "cohort_recipe_selected": False,
        "scale_up_authorized": False,
    }


def build_prefix(
    source: Path,
    *,
    output: Path,
    count: int,
    log: Path,
    threads: int,
) -> Path:
    if count == EXPECTED_SOURCE_STREAMLINES:
        return source
    partial = output.with_name(f".{output.stem}.partial{output.suffix}")
    run_logged(
        [
            str(MRTRIX / "tckedit"),
            str(source),
            str(partial),
            "-number",
            str(count),
            "-nthreads",
            str(min(threads, 4)),
        ],
        log=log,
        outputs=(partial,),
    )
    if tck_count(partial, count) != count:
        raise ValueError("deterministic prefix streamline count differs")
    os.replace(partial, output)
    return output


def concatenate(
    primary: Path,
    independent: Path,
    *,
    output: Path,
    expected: int,
    log: Path,
    threads: int,
) -> Path:
    partial = output.with_name(f".{output.stem}.partial{output.suffix}")
    run_logged(
        [
            str(MRTRIX / "tckedit"),
            str(primary),
            str(independent),
            str(partial),
            "-nthreads",
            str(min(threads, 4)),
        ],
        log=log,
        outputs=(partial,),
    )
    if tck_count(partial, expected) != expected:
        raise ValueError("balanced concatenated streamline count differs")
    os.replace(partial, output)
    return output


def build_sift2(
    tracks: Path,
    *,
    fod: Path,
    five_tt: Path,
    root: Path,
    expected: int,
    threads: int,
) -> tuple[Path, dict[str, Any]]:
    weights = root / "sift2_weights.txt"
    mu = root / "sift2_mu.txt"
    iterations = root / "sift2_iterations.csv"
    partial_weights = root / ".sift2_weights.partial.txt"
    partial_mu = root / ".sift2_mu.partial.txt"
    partial_iterations = root / ".sift2_iterations.partial.csv"
    run_logged(
        [
            str(MRTRIX / "tcksift2"),
            str(tracks),
            str(fod),
            str(partial_weights),
            "-act",
            str(five_tt),
            "-out_mu",
            str(partial_mu),
            "-csv",
            str(partial_iterations),
            "-reg_tikhonov",
            "0.0",
            "-reg_tv",
            "0.1",
            "-min_td_frac",
            "0.1",
            "-min_iters",
            "10",
            "-max_iters",
            "1000",
            "-nthreads",
            str(threads),
        ],
        log=root / "sift2.log",
        outputs=(partial_weights, partial_mu, partial_iterations),
    )
    if numeric_count(partial_weights, expected) != expected:
        raise ValueError("SIFT2 weight count differs")
    for partial, final in (
        (partial_weights, weights),
        (partial_mu, mu),
        (partial_iterations, iterations),
    ):
        os.replace(partial, final)
    return weights, {
        "weights": file_record(weights),
        "mu": file_record(mu),
        "iterations": file_record(iterations),
        "log": file_record(root / "sift2.log"),
        "weight_count": expected,
    }


def connectome(
    tracks: Path,
    nodes: Path,
    output: Path,
    *,
    log: Path,
    threads: int,
    extra: Sequence[str],
) -> None:
    partial = output.with_name(f".{output.stem}.partial{output.suffix}")
    run_logged(
        [
            str(MRTRIX / "tck2connectome"),
            str(tracks),
            str(nodes),
            str(partial),
            "-assignment_radial_search",
            "4",
            "-symmetric",
            "-zero_diagonal",
            *extra,
            "-nthreads",
            str(min(threads, 8)),
            "-force",
        ],
        log=log,
        outputs=(partial,),
    )
    os.replace(partial, output)


def count_connectome(
    tracks: Path,
    nodes: Path,
    *,
    root: Path,
    expected: int,
    threads: int,
) -> tuple[Path, Path, dict[str, Any]]:
    output = root / "matrices/count.csv"
    assignments = root / "assignments.csv"
    partial_output = output.with_name(
        f".{output.stem}.partial{output.suffix}"
    )
    partial_assignments = assignments.with_name(
        f".{assignments.stem}.partial{assignments.suffix}"
    )
    run_logged(
        [
            str(MRTRIX / "tck2connectome"),
            str(tracks),
            str(nodes),
            str(partial_output),
            "-assignment_radial_search",
            "4",
            "-symmetric",
            "-zero_diagonal",
            "-stat_edge",
            "sum",
            "-out_assignments",
            str(partial_assignments),
            "-nthreads",
            str(min(threads, 8)),
            "-force",
        ],
        log=root / "count.log",
        outputs=(partial_output, partial_assignments),
    )
    os.replace(partial_output, output)
    os.replace(partial_assignments, assignments)
    summary = assignment_summary(assignments, expected)
    return output, assignments, summary


def build_view(
    *,
    name: str,
    tracks: Path,
    expected: int,
    nodes: Path,
    node_volumes_path: Path,
    model: Mapping[str, Path],
    root: Path,
    threads: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    root.mkdir(parents=True, exist_ok=False)
    weights, sift2 = build_sift2(
        tracks,
        fod=model["wmfod"],
        five_tt=model["five_tt"],
        root=root,
        expected=expected,
        threads=threads,
    )
    count_path, assignments, assignment = count_connectome(
        tracks,
        nodes,
        root=root,
        expected=expected,
        threads=threads,
    )
    fd_path = root / "matrices/fd_sum.csv"
    connectome(
        tracks,
        nodes,
        fd_path,
        log=root / "fd_sum.log",
        threads=threads,
        extra=(
            "-stat_edge",
            "sum",
            "-tck_weights_in",
            str(weights),
        ),
    )
    for metric, scale in (
        ("len", "-scale_length"),
        ("invlen", "-scale_invlength"),
    ):
        connectome(
            tracks,
            nodes,
            root / f"matrices/{metric}_mean.csv",
            log=root / f"{metric}_mean.log",
            threads=threads,
            extra=(scale, "-stat_edge", "mean"),
        )
    for metric in SCALAR_METRICS:
        samples = root / f".{metric}_per_streamline.temporary.csv"
        run_logged(
            [
                str(MRTRIX / "tcksample"),
                str(tracks),
                str(model[metric]),
                str(samples),
                "-stat_tck",
                "mean",
                "-nthreads",
                str(min(threads, 8)),
                "-force",
            ],
            log=root / f"{metric}_sample.log",
            outputs=(samples,),
        )
        if numeric_count(samples, expected) != expected:
            raise ValueError(f"{name}/{metric}: scalar sample count differs")
        connectome(
            tracks,
            nodes,
            root / f"matrices/{metric}_mean.csv",
            log=root / f"{metric}_mean.log",
            threads=threads,
            extra=(
                "-scale_file",
                str(samples),
                "-stat_edge",
                "mean",
            ),
        )
        os.remove(samples)

    loaded = {
        matrix_name: matrix(root / f"matrices/{matrix_name}.csv")
        for matrix_name in MATRIX_NAMES
        if matrix_name != "count_invnodevol"
    }
    node_volumes = TECHNICAL.load_node_volumes(node_volumes_path)
    denominator = node_volumes[:, None] + node_volumes[None, :]
    inverse_volume = 2.0 * loaded["count"] / denominator
    np.fill_diagonal(inverse_volume, 0.0)
    inverse_path = root / "matrices/count_invnodevol.csv"
    atomic_matrix(inverse_path, inverse_volume)
    loaded["count_invnodevol"] = matrix(inverse_path)
    technical, failures = TECHNICAL.matrix_technical_qc(
        loaded, node_volumes=node_volumes
    )
    records = {
        matrix_name: file_record(
            root / f"matrices/{matrix_name}.csv"
        )
        for matrix_name in MATRIX_NAMES
    }
    return loaded, {
        "view": name,
        "streamline_count": expected,
        "tractogram": file_record(tracks),
        "sift2": sift2,
        "assignments": file_record(assignments),
        **assignment,
        "matrices": records,
        "technical_qc": technical,
        "technical_failures": failures,
        "technical_status": "PASS" if not failures else "FAIL",
        "scalar_samples_compacted_after_matrix_generation": True,
    }


def execute(
    unit: str,
    density_summary_path: Path,
    threads: int,
    selected_total: int | None = None,
) -> dict[str, Any]:
    ready = preflight(unit, density_summary_path, selected_total)
    if ready["status"] != "READY_HROI_BALANCED_ALL_NINE_CANARY":
        raise RuntimeError(ready["status"])
    valid_density_summary(density_summary_path)
    selected = int(ready["selected_balanced_total_streamlines"])
    per_seed = BALANCED_TOTALS[selected]
    atlas = atlas_inputs(unit)
    phase = phase_inputs(unit)
    attempt_id = f"{attempt_stamp()}-hroi-all-nine-{unit}-{selected}"
    attempt = ATTEMPTS / attempt_id
    attempt.mkdir(parents=True, exist_ok=False)
    bound = {
        **ready,
        "status": "EXECUTION_INPUTS_BOUND",
        "attempt_id": attempt_id,
        "all_nine_matrix_generation_started": True,
        "density_summary_record": file_record(density_summary_path),
        "hroi_policy_receipt": file_record(HROI_POLICY_RECEIPT),
        "hroi_atlas_summary": file_record(HROI_ATLAS_SUMMARY),
        "hroi_atlas_result": file_record(atlas["result"]),
        "hroi_nodes": file_record(atlas["nodes"]),
        "hroi_node_volumes": file_record(atlas["node_volumes"]),
        "hroi_atlas_qc": file_record(atlas["atlas_qc"]),
        "source_inputs": {
            name: file_record(path) for name, path in phase.items()
        },
        "contract": file_record(CONTRACT),
        "implementation": file_record(Path(__file__)),
        "technical_validator": file_record(TECHNICAL_VALIDATOR_SOURCE),
    }
    atomic_json(attempt / "preflight.json", bound)

    selected_tracks: dict[str, Path] = {}
    for seed in ("primary", "independent"):
        selected_tracks[seed] = build_prefix(
            phase[f"{seed}_tracks"],
            output=attempt / f"{seed}/tracks_{per_seed}.tck",
            count=per_seed,
            log=attempt / f"{seed}_prefix.log",
            threads=threads,
        )
    balanced_tracks = concatenate(
        selected_tracks["primary"],
        selected_tracks["independent"],
        output=attempt / f"balanced/tracks_{selected}.tck",
        expected=selected,
        log=attempt / "balanced_concat.log",
        threads=threads,
    )
    tracks_by_view = {
        "primary": selected_tracks["primary"],
        "independent": selected_tracks["independent"],
        "balanced": balanced_tracks,
    }
    expected_by_view = {
        "primary": per_seed,
        "independent": per_seed,
        "balanced": selected,
    }
    matrices_by_view: dict[str, dict[str, np.ndarray]] = {}
    view_records: dict[str, Any] = {}
    for view in VIEW_NAMES:
        matrices_by_view[view], view_records[view] = build_view(
            name=view,
            tracks=tracks_by_view[view],
            expected=expected_by_view[view],
            nodes=atlas["nodes"],
            node_volumes_path=atlas["node_volumes"],
            model=phase,
            root=attempt / view / "all_nine",
            threads=threads,
        )

    gate = load_json(CONTRACT)["diagnosis_blind_recipe_stability_gate"]
    seed_comparison = TECHNICAL.evaluate_comparison(
        left_id="primary_half",
        right_id="independent_half",
        left=matrices_by_view["primary"],
        right=matrices_by_view["independent"],
        left_assignment=view_records["primary"][
            "endpoint_assignment_fraction"
        ],
        right_assignment=view_records["independent"][
            "endpoint_assignment_fraction"
        ],
        gate=gate,
    )
    combined_density = float(
        view_records["balanced"]["technical_qc"]["edge_density"]
    )
    all_technical_pass = all(
        row["technical_status"] == "PASS"
        for row in view_records.values()
    )
    density_pass = combined_density >= DENSITY_TARGET
    status = (
        "PASS_HROI_BALANCED_ALL_NINE_CANARY"
        if all_technical_pass
        and density_pass
        and seed_comparison["status"] == "PASS"
        else "FAIL_HROI_BALANCED_ALL_NINE_CANARY"
    )
    summary = {
        "schema_version": "1.0.0",
        "record_type": (
            "diagnosis_blind_hcp379_hroi_balanced_all_nine_canary_summary"
        ),
        "generated_utc": utc_now(),
        "status": status,
        "attempt_id": attempt_id,
        "unit": unit,
        "diagnosis_labels_used": False,
        "non_overwriting": True,
        "source_files_modified": False,
        "probabilistic_tractography_generated": False,
        "source_10m_tractograms_reused": True,
        "deterministic_prefix_or_concatenation_only": True,
        "selected_balanced_total_streamlines": selected,
        "selected_candidate_rank": ready["selected_candidate_rank"],
        "selection_mode": ready["selection_mode"],
        "selected_per_seed_prefix_streamlines": per_seed,
        "balanced_construction": (
            f"first_{per_seed}_primary_plus_first_{per_seed}_independent"
        ),
        "sift2_fit_separately_on_each_exact_view": True,
        "independent_sift2_matrices_summed_for_balanced_view": False,
        "matrix_names": list(MATRIX_NAMES),
        "views": view_records,
        "seed_reliability": seed_comparison,
        "combined_density_target": DENSITY_TARGET,
        "combined_edge_density": combined_density,
        "combined_density_target_met": density_pass,
        "all_views_technical_qc_pass": all_technical_pass,
        "cohort_recipe_selected": False,
        "scale_up_authorized": False,
        "requires_complete_15_canary_all_nine_validation": True,
        "requires_hroi_human_release_gate": True,
        "preflight": file_record(attempt / "preflight.json"),
        "density_summary": file_record(density_summary_path),
        "hroi_atlas_result": file_record(atlas["result"]),
        "implementation": file_record(Path(__file__)),
    }
    atomic_json(attempt / "summary.json", summary)
    return {
        "attempt_root": str(attempt),
        "status": status,
        "unit": unit,
        "selected_balanced_total_streamlines": selected,
        "combined_edge_density": combined_density,
        "seed_reliability_status": seed_comparison["status"],
    }


def self_test() -> dict[str, Any]:
    count = np.zeros((EXPECTED_NODES, EXPECTED_NODES), dtype=float)
    count[0, 1] = count[1, 0] = 4.0
    volumes = np.arange(1, EXPECTED_NODES + 1, dtype=float)
    expected = 2.0 * count / (
        volumes[:, None] + volumes[None, :]
    )
    np.fill_diagonal(expected, 0.0)
    checks = {
        "exact_matrix_family": len(MATRIX_NAMES) == 9
        and set(MATRIX_NAMES)
        == {
            "count",
            "fd_sum",
            "count_invnodevol",
            "len_mean",
            "invlen_mean",
            "fa_mean",
            "md_mean",
            "rd_mean",
            "ad_mean",
        },
        "balanced_prefix_map": BALANCED_TOTALS
        == {
            6_000_000: 3_000_000,
            10_000_000: 5_000_000,
            15_000_000: 7_500_000,
            20_000_000: 10_000_000,
        },
        "inverse_volume_formula": bool(
            abs(expected[0, 1] - 8.0 / 3.0) < 1.0e-12
        ),
        "three_exact_views": VIEW_NAMES
        == ("primary", "independent", "balanced"),
        "fixed_density_target": DENSITY_TARGET == 0.60,
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unit", default="032_S_6804_I1230908")
    parser.add_argument("--density-summary", type=Path)
    parser.add_argument(
        "--balanced-total",
        type=int,
        choices=tuple(BALANCED_TOTALS),
        help=(
            "Uniform density-qualified candidate to test. Defaults to the "
            "smallest density-qualified total."
        ),
    )
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
    _, by_unit = atlas_contract()
    if args.unit not in by_unit:
        parser.error("--unit must be one of the fixed 15 HROI canaries")
    density_summary = (
        args.density_summary.resolve()
        if args.density_summary is not None
        else latest_density_summary()
    )
    value = preflight(
        args.unit, density_summary, args.balanced_total
    )
    if not args.execute:
        print(json.dumps(value, indent=2, sort_keys=True))
        return 0
    if density_summary is None or value["status"] != (
        "READY_HROI_BALANCED_ALL_NINE_CANARY"
    ):
        print(json.dumps(value, indent=2, sort_keys=True))
        return 2
    output = execute(
        args.unit,
        density_summary,
        args.threads,
        args.balanced_total,
    )
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0 if output["status"].startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
