#!/usr/bin/env python3
"""Diagnosis-blind HCP379 registration/remapping probe.

This script does not modify the historical HCP parcellations or matrices.  It
tests two independent T1-to-b0 routes against the current mapping:

* cached_bbr: the frozen production FSL BBR transform;
* fresh_normmi: a newly estimated constrained six-DOF FLIRT transform.

Each candidate is resampled with nearest-neighbour interpolation, relabelled to
1..379, assigned against the unchanged tractogram, and audited.  Results are
evidence for choosing a remapping route; the script does not promote a winner.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


EXP = Path("/home/ec2-user/exp")
V2_ROOT = Path("/data/derivatives/hcp379_v2")
TRIAGE_LIST = (
    EXP / "research_audit/outputs/hcp_rerun_triage_v1/hcp_remap_probe_111.txt"
)
MRTRIX = Path("/home/ec2-user/mrtrix3/bin")
FSL = Path("/home/ec2-user/fsl/bin")
FREESURFER = Path("/home/ec2-user/freesurfer")
RELABEL = EXP / "scripts/hcp/relabel_hcp.py"
LUT = Path("/data/derivatives/parc_hcpmmp1/hcpmmp1_subcort_relabel.txt")
N_NODES = 379
HUB_IDS = (365, 374, 361, 370, 366, 375)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def run(command: list[str], log: Any, env: dict[str, str] | None = None) -> None:
    log.write(f"\n[{utc_now()}] COMMAND {json.dumps(command)}\n")
    log.flush()
    result = subprocess.run(
        command,
        stdout=log,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    if result.returncode:
        raise RuntimeError(f"command failed rc={result.returncode}: {command[0]}")


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def read_node_volumes(path: Path) -> tuple[int, float, list[str]]:
    rows = list(csv.DictReader(path.open()))
    voxels = {int(row["node_id"]): int(row["n_voxels"]) for row in rows}
    present = sum(value > 0 for value in voxels.values())
    left = sum(voxels.get(node, 0) for node in range(1, 181))
    right = sum(voxels.get(node, 0) for node in range(181, 361))
    ratio = left / max(right, 1)
    failures: list[str] = []
    if present < 376:
        failures.append(f"atlas_nodes_present={present}<376")
    if not 0.6 <= ratio <= 1.67:
        failures.append(f"cortical_lr_ratio={ratio:.4f}")
    for name, left_id, right_id in (
        ("thal", 361, 370),
        ("hipp", 365, 374),
        ("put", 363, 372),
    ):
        pair_ratio = voxels.get(left_id, 0) / max(voxels.get(right_id, 0), 1)
        if not 0.35 <= pair_ratio <= 2.85:
            failures.append(f"{name}_lr_ratio={pair_ratio:.4f}")
    return present, ratio, failures


def matrix_qc(path: Path, track_count: int) -> dict[str, Any]:
    matrix = np.loadtxt(path, delimiter=",")
    failures: list[str] = []
    if matrix.shape != (N_NODES, N_NODES):
        failures.append(f"shape={matrix.shape}")
    if not np.isfinite(matrix).all():
        failures.append("nonfinite")
    if not np.allclose(matrix, matrix.T, atol=1e-6, rtol=1e-6):
        failures.append("asymmetric")
    if not np.allclose(np.diag(matrix), 0.0, atol=1e-8):
        failures.append("diagonal")
    strength = matrix.sum(axis=1)
    connected = int((strength > 0).sum())
    density = float(np.count_nonzero(matrix) / (N_NODES * (N_NODES - 1)))
    assigned = float(matrix.sum() / 2.0)
    min_hub = float(min(strength[node - 1] for node in HUB_IDS))
    if connected < 360:
        failures.append(f"connected_nodes={connected}<360")
    if assigned / max(track_count, 1) < 0.50:
        failures.append(f"assignment_fraction={assigned / max(track_count, 1):.4f}<0.50")
    if min_hub <= 0:
        failures.append("hub_support=0")
    return {
        "connected_nodes": connected,
        "density": density,
        "assigned_streamlines": assigned,
        "assignment_fraction": assigned / max(track_count, 1),
        "min_hub_strength": min_hub,
        "matrix_failures": failures,
    }


def track_count(path: Path) -> int:
    with path.open("rb") as handle:
        header = handle.read(65536).split(b"END\n", 1)[0].decode(
            "utf-8", "replace"
        )
    for line in header.splitlines():
        if line.startswith("count:"):
            return int(line.split(":", 1)[1].strip())
    raise RuntimeError(f"track count missing from {path}")


def build_variant(
    sid: str,
    variant: str,
    subject_root: Path,
    hcp_t1_grid: Path,
    b0: Path,
    transform: Path,
    track: Path,
    nthreads: int,
    log: Any,
) -> dict[str, Any]:
    output = subject_root / variant
    output.mkdir(parents=True, exist_ok=True)
    raw_b0 = output / "HCPMMP1+aseg_b0.nii.gz"
    nodes = output / "nodes_b0.nii.gz"
    volumes = output / "node_volumes.csv"
    count = output / "count.csv"
    assignments = output / "assignments.csv"

    run(
        [
            str(FSL / "flirt"), "-in", str(hcp_t1_grid), "-ref", str(b0),
            "-applyxfm", "-init", str(transform), "-interp",
            "nearestneighbour", "-datatype", "int", "-out", str(raw_b0),
        ],
        log,
    )
    run(
        [
            str(FSL / "python"), str(RELABEL), str(raw_b0), str(LUT), str(nodes),
            str(volumes),
        ],
        log,
    )
    run(
        [
            str(MRTRIX / "tck2connectome"), str(track), str(nodes), str(count),
            "-symmetric", "-zero_diagonal", "-assignment_radial_search", "4",
            "-stat_edge", "sum", "-out_assignments", str(assignments),
            "-nthreads", str(nthreads), "-force", "-quiet",
        ],
        log,
    )
    atlas_present, cortical_ratio, atlas_failures = read_node_volumes(volumes)
    qc = matrix_qc(count, track_count(track))
    qc.update(
        {
            "variant": variant,
            "atlas_nodes_present": atlas_present,
            "atlas_cortical_lr_ratio": cortical_ratio,
            "atlas_failures": atlas_failures,
            "nodes_path": str(nodes),
            "count_path": str(count),
            "transform_path": str(transform),
        }
    )
    qc["technical_pass"] = not atlas_failures and not qc["matrix_failures"]
    return qc


def probe_subject(sid: str, root: Path, nthreads: int) -> dict[str, Any]:
    subject_root = root / "remap_probes" / sid
    subject_root.mkdir(parents=True, exist_ok=True)
    log_path = subject_root / "probe.log"
    result_path = subject_root / "result.json"
    started = time.monotonic()
    result: dict[str, Any] = {
        "fsid": sid,
        "started_utc": utc_now(),
        "status": "RUNNING",
        "promotion": "NONE_PROBE_ONLY",
    }
    atomic_json(result_path, result)
    try:
        current_work = Path("/data/derivatives/parc_hcpmmp1") / sid
        hcp_mgz = current_work / "HCPMMP1+aseg.mgz"
        current_count = (
            Path("/data/derivatives/connectomes_hcp_fs")
            / f"SC_HCPMMP1_{sid}_count.csv"
        )
        current_volumes = current_work / "node_volumes.csv"
        track = Path("/data/derivatives/tracks") / sid / "tracks_final_3000k.tck"
        bbr_root = Path("/data/derivatives/dwi_t1_bbr")
        t1 = bbr_root / f"{sid}_t1_ras.nii.gz"
        b0 = bbr_root / f"{sid}_b0mean_ras.nii.gz"
        cached_bbr = bbr_root / f"{sid}_t12b0_bbr.mat"
        required = [
            hcp_mgz, current_count, current_volumes, track, t1, b0, cached_bbr
        ]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError("missing remap inputs: " + ",".join(missing))

        n_tracks = track_count(track)
        current_atlas_present, current_lr, current_atlas_failures = (
            read_node_volumes(current_volumes)
        )
        current = matrix_qc(current_count, n_tracks)
        current.update(
            {
                "variant": "current_bbregister",
                "atlas_nodes_present": current_atlas_present,
                "atlas_cortical_lr_ratio": current_lr,
                "atlas_failures": current_atlas_failures,
                "nodes_path": str(current_work / "nodes_b0.nii.gz"),
                "count_path": str(current_count),
                "technical_pass": (
                    not current_atlas_failures and not current["matrix_failures"]
                ),
            }
        )

        fs_env = os.environ.copy()
        fs_env.update(
            {
                "FREESURFER_HOME": str(FREESURFER),
                "SUBJECTS_DIR": "/data/derivatives/fastsurfer",
                "FS_LICENSE": str(FREESURFER / "license.txt"),
                "PATH": f"{FREESURFER / 'bin'}:{fs_env.get('PATH', '')}",
            }
        )
        hcp_t1_grid = subject_root / "HCPMMP1+aseg_t1grid.nii.gz"
        fresh_transform = subject_root / "t1_to_b0_fresh_normmi.mat"
        t1_in_b0 = subject_root / "t1_in_b0_fresh_normmi.nii.gz"
        with log_path.open("a") as log:
            if not hcp_t1_grid.exists():
                run(
                    [
                        str(FREESURFER / "bin/mri_vol2vol"),
                        "--mov", str(hcp_mgz), "--targ", str(t1),
                        "--regheader", "--interp", "nearest",
                        "--keep-precision", "--o", str(hcp_t1_grid),
                    ],
                    log,
                    env=fs_env,
                )
            run(
                [
                    str(FSL / "flirt"), "-in", str(t1), "-ref", str(b0),
                    "-out", str(t1_in_b0), "-omat", str(fresh_transform),
                    "-dof", "6", "-cost", "normmi", "-searchcost", "normmi",
                    "-searchrx", "-40", "40", "-searchry", "-40", "40",
                    "-searchrz", "-40", "40", "-interp", "trilinear",
                ],
                log,
            )
            variants = [current]
            variants.append(
                build_variant(
                    sid, "cached_bbr", subject_root, hcp_t1_grid, b0,
                    cached_bbr, track, nthreads, log,
                )
            )
            variants.append(
                build_variant(
                    sid, "fresh_normmi", subject_root, hcp_t1_grid, b0,
                    fresh_transform, track, nthreads, log,
                )
            )

        result.update(
            status="PASS_PROBE",
            completed_utc=utc_now(),
            elapsed_seconds=round(time.monotonic() - started, 3),
            track_path=str(track),
            track_count=n_tracks,
            variants=variants,
            note=(
                "Probe only. A variant must pass visual registration review and "
                "cohort-level control checks before any promotion."
            ),
        )
    except Exception as exc:
        result.update(
            status="FAIL_PROBE",
            completed_utc=utc_now(),
            elapsed_seconds=round(time.monotonic() - started, 3),
            error=str(exc),
        )
    atomic_json(result_path, result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=V2_ROOT)
    parser.add_argument("--subjects", nargs="*")
    parser.add_argument("--nthreads", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    subjects = args.subjects
    if not subjects:
        subjects = [
            line.strip() for line in TRIAGE_LIST.read_text().splitlines()
            if line.strip()
        ]
    print(f"[{utc_now()}] REMAP_PROBE_START n={len(subjects)}", flush=True)
    results = []
    for index, sid in enumerate(subjects, 1):
        result = probe_subject(sid, args.root, args.nthreads)
        results.append(result)
        variants = result.get("variants", [])
        densities = {
            row["variant"]: round(row["density"], 4) for row in variants
        }
        print(
            f"[{utc_now()}] remap {index}/{len(subjects)} {sid} "
            f"{result['status']} densities={densities}",
            flush=True,
        )
    atomic_json(
        args.root / "manifests/remap_probe_summary.json",
        {
            "updated_utc": utc_now(),
            "n": len(results),
            "status_counts": {
                status: sum(row["status"] == status for row in results)
                for status in sorted({row["status"] for row in results})
            },
            "subjects": results,
        },
    )
    if any(row["status"] != "PASS_PROBE" for row in results):
        raise SystemExit(1)


if __name__ == "__main__":
    import sys

    main()
