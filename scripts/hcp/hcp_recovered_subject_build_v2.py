#!/usr/bin/env python3
"""Build and audit HCP379 products for the recovered FreeSurfer subject."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


FREESURFER = Path("/home/ec2-user/freesurfer")
MRTRIX = Path("/home/ec2-user/mrtrix3/bin")
FSL_PYTHON = Path("/home/ec2-user/fsl/bin/python")
EXP = Path("/home/ec2-user/exp")
ROOT = Path("/data/derivatives/hcp379_v2")
HCP_ATLAS = Path("/home/ec2-user/hcp_atlas")
LUT = Path("/data/derivatives/parc_hcpmmp1/hcpmmp1_subcort_relabel.txt")
DEFAULT_FSID = "114_S_6347_I1344943"
N_NODES = 379
HUB_IDS = (365, 374, 361, 370, 366, 375)
MATRIX_SUFFIXES = (
    "count",
    "fd_sum",
    "len_mean",
    "invlen_mean",
    "fa_mean",
    "md_mean",
    "rd_mean",
    "ad_mean",
    "count_invnodevol",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def run(command: list[str], log: Any, env: dict[str, str]) -> None:
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


def track_count(path: Path) -> int:
    with path.open("rb") as handle:
        header = handle.read(65536).split(b"END\n", 1)[0].decode(
            "utf-8", "replace"
        )
    for line in header.splitlines():
        if line.startswith("count:"):
            return int(line.split(":", 1)[1].strip())
    raise RuntimeError(f"track count missing: {path}")


def validate_matrix(matrix: np.ndarray, name: str) -> list[str]:
    failures = []
    if matrix.shape != (N_NODES, N_NODES):
        return [f"{name}:shape={matrix.shape}"]
    if not np.isfinite(matrix).all():
        failures.append(f"{name}:nonfinite")
    if not np.allclose(matrix, matrix.T, atol=1e-6, rtol=1e-6):
        failures.append(f"{name}:asymmetric")
    if not np.allclose(np.diag(matrix), 0.0, atol=1e-8):
        failures.append(f"{name}:diagonal")
    if float(matrix.min()) < -1e-10:
        failures.append(f"{name}:negative={float(matrix.min())}")
    return failures


def process(fsid: str, root: Path, nthreads: int) -> dict[str, Any]:
    repair_root = root / "fastsurfer_repair" / fsid
    work = repair_root / "hcp"
    work.mkdir(parents=True, exist_ok=True)
    result_path = repair_root / "hcp_build_result.json"
    if result_path.exists():
        prior = json.loads(result_path.read_text())
        if prior.get("status") == "PASS_ALL_NINE_EXISTING_TRACK":
            return prior

    recovery = json.loads(
        (repair_root / "standard_freesurfer_result.json").read_text()
    )
    if recovery.get("status") != "PASS_SURFACE_RECOVERY":
        raise RuntimeError("standard FreeSurfer recovery has not passed")
    subjects_dir = Path(recovery["subjects_dir"])
    recovery_sid = str(recovery["recovery_sid"])
    subject_dir = subjects_dir / recovery_sid
    fsaverage = subjects_dir / "fsaverage"
    if not fsaverage.exists():
        os.symlink(FREESURFER / "subjects/fsaverage", fsaverage)

    track_root = Path("/data/derivatives/tracks") / fsid
    track = track_root / "tracks_final_3000k.tck"
    sift = track_root / "sift_weights.txt"
    b0 = Path("/data/derivatives/dwi_t1_bbr") / f"{fsid}_b0mean_ras.nii.gz"
    required = [
        subject_dir / "surf/lh.sphere.reg",
        subject_dir / "surf/rh.sphere.reg",
        subject_dir / "mri/aseg.mgz",
        track,
        sift,
        b0,
        LUT,
    ]
    required.extend(track_root / f"{metric}_mean.tsf" for metric in ("fa", "md", "rd", "ad"))
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("missing HCP build inputs: " + ",".join(missing))

    log_path = work / "hcp_build.log"
    prefix = root / "connectomes" / f"SC_HCPMMP1_{fsid}"
    prefix.parent.mkdir(parents=True, exist_ok=True)
    assignments = root / "assignments" / fsid / "count_assignments.csv"
    assignments.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    state: dict[str, Any] = {
        "fsid": fsid,
        "status": "RUNNING",
        "started_utc": utc_now(),
        "non_overwriting": True,
        "recovery_sid": recovery_sid,
    }
    atomic_json(result_path, state)
    try:
        env = os.environ.copy()
        env.update(
            {
                "FREESURFER_HOME": str(FREESURFER),
                "FS_LICENSE": str(FREESURFER / "license.txt"),
                "SUBJECTS_DIR": str(subjects_dir),
                "PATH": (
                    f"{FREESURFER / 'bin'}:{MRTRIX}:"
                    f"{env.get('PATH', '')}"
                ),
            }
        )
        annot_paths = {}
        atlas_mgz = work / "HCPMMP1+aseg.mgz"
        b0_atlas = work / "HCPMMP1+aseg_b0.nii.gz"
        nodes = work / "nodes_b0.nii.gz"
        node_volumes = work / "node_volumes.csv"
        register = work / "b0_to_fs.dat"
        with log_path.open("a") as log:
            for hemi in ("lh", "rh"):
                target = subject_dir / "label" / f"{hemi}.HCPMMP1.annot"
                annot_paths[hemi] = target
                run(
                    [
                        str(FREESURFER / "bin/mri_surf2surf"),
                        "--srcsubject",
                        "fsaverage",
                        "--trgsubject",
                        recovery_sid,
                        "--hemi",
                        hemi,
                        "--sval-annot",
                        str(HCP_ATLAS / f"{hemi}.HCPMMP1.annot"),
                        "--tval",
                        str(target),
                    ],
                    log,
                    env,
                )
            run(
                [
                    str(FREESURFER / "bin/mri_aparc2aseg"),
                    "--s",
                    recovery_sid,
                    "--annot",
                    "HCPMMP1",
                    "--o",
                    str(atlas_mgz),
                ],
                log,
                env,
            )
            run(
                [
                    str(FREESURFER / "bin/bbregister"),
                    "--s",
                    recovery_sid,
                    "--mov",
                    str(b0),
                    "--t2",
                    "--init-fsl",
                    "--reg",
                    str(register),
                ],
                log,
                env,
            )
            run(
                [
                    str(FREESURFER / "bin/mri_label2vol"),
                    "--seg",
                    str(atlas_mgz),
                    "--temp",
                    str(b0),
                    "--reg",
                    str(register),
                    "--o",
                    str(b0_atlas),
                ],
                log,
                env,
            )
            run(
                [
                    str(FSL_PYTHON),
                    str(EXP / "scripts/hcp/relabel_hcp.py"),
                    str(b0_atlas),
                    str(LUT),
                    str(nodes),
                    str(node_volumes),
                ],
                log,
                env,
            )
            common = [
                "-symmetric",
                "-zero_diagonal",
                "-assignment_radial_search",
                "4",
                "-force",
                "-quiet",
                "-nthreads",
                str(nthreads),
            ]
            run(
                [
                    str(MRTRIX / "tck2connectome"),
                    str(track),
                    str(nodes),
                    f"{prefix}_count.csv",
                    *common,
                    "-stat_edge",
                    "sum",
                    "-out_assignments",
                    str(assignments),
                ],
                log,
                env,
            )
            run(
                [
                    str(MRTRIX / "tck2connectome"),
                    str(track),
                    str(nodes),
                    f"{prefix}_fd_sum.csv",
                    *common,
                    "-tck_weights_in",
                    str(sift),
                    "-stat_edge",
                    "sum",
                ],
                log,
                env,
            )
            for suffix, scale in (
                ("len_mean", "-scale_length"),
                ("invlen_mean", "-scale_invlength"),
            ):
                run(
                    [
                        str(MRTRIX / "tck2connectome"),
                        str(track),
                        str(nodes),
                        f"{prefix}_{suffix}.csv",
                        *common,
                        scale,
                        "-stat_edge",
                        "mean",
                    ],
                    log,
                    env,
                )
            for metric in ("fa", "md", "rd", "ad"):
                run(
                    [
                        str(MRTRIX / "tck2connectome"),
                        str(track),
                        str(nodes),
                        f"{prefix}_{metric}_mean.csv",
                        *common,
                        "-scale_file",
                        str(track_root / f"{metric}_mean.tsf"),
                        "-stat_edge",
                        "mean",
                    ],
                    log,
                    env,
                )

        count = np.loadtxt(f"{prefix}_count.csv", delimiter=",")
        volumes = {
            int(row["node_id"]): float(row["volume_mm3"])
            for row in csv.DictReader(node_volumes.open())
        }
        vector = np.array([volumes.get(index + 1, 0.0) for index in range(N_NODES)])
        denominator = vector[:, None] + vector[None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            invnode = np.where(denominator > 0, 2.0 * count / denominator, 0.0)
        np.fill_diagonal(invnode, 0.0)
        np.savetxt(
            f"{prefix}_count_invnodevol.csv",
            invnode,
            delimiter=",",
            fmt="%.10g",
        )

        failures: list[str] = []
        hashes = {}
        matrices = {}
        for suffix in MATRIX_SUFFIXES:
            path = Path(f"{prefix}_{suffix}.csv")
            matrix = np.loadtxt(path, delimiter=",")
            matrices[suffix] = matrix
            failures.extend(validate_matrix(matrix, suffix))
            hashes[suffix] = sha256(path)
        strength = count.sum(axis=1)
        connected = int((strength > 0).sum())
        density = float(np.count_nonzero(count) / (N_NODES * (N_NODES - 1)))
        assigned = float(count.sum() / 2.0)
        n_tracks = track_count(track)
        assignment_fraction = assigned / n_tracks
        min_hub_strength = float(min(strength[index - 1] for index in HUB_IDS))
        atlas_nodes_present = sum(value > 0 for value in vector)
        count_support = np.triu(count > 0, 1)
        edge_count = max(int(count_support.sum()), 1)
        weighted_support = {
            suffix: float(
                (
                    np.triu(matrices[suffix] != 0, 1)
                    & count_support
                ).sum()
                / edge_count
            )
            for suffix in ("fd_sum", "fa_mean", "md_mean", "rd_mean", "ad_mean")
        }
        if connected < 360:
            failures.append(f"connected_nodes={connected}<360")
        if density < 0.15:
            failures.append(f"density={density:.6f}<0.15")
        if assignment_fraction < 0.50:
            failures.append(
                f"assignment_fraction={assignment_fraction:.6f}<0.50"
            )
        if min_hub_strength <= 0:
            failures.append("hub_support=0")
        if atlas_nodes_present < 376:
            failures.append(f"atlas_nodes_present={atlas_nodes_present}<376")
        for suffix, support in weighted_support.items():
            if support < 0.95:
                failures.append(f"{suffix}:support={support:.6f}<0.95")

        state.update(
            status=(
                "PASS_ALL_NINE_EXISTING_TRACK"
                if not failures
                else "FAIL_TRACK_REBUILD_REQUIRED"
            ),
            completed_utc=utc_now(),
            elapsed_seconds=round(time.monotonic() - started, 3),
            failures=sorted(set(failures)),
            subjects_dir=str(subjects_dir),
            track_path=str(track),
            track_sha256=sha256(track),
            track_count=n_tracks,
            density=density,
            connected_nodes=connected,
            assigned_streamlines=assigned,
            assignment_fraction=assignment_fraction,
            min_hub_strength=min_hub_strength,
            atlas_nodes_present=atlas_nodes_present,
            weighted_edge_support=weighted_support,
            matrix_sha256=hashes,
            atlas_sha256=sha256(nodes),
            registration_path=str(register),
            annotation_sha256={
                hemi: sha256(path) for hemi, path in annot_paths.items()
            },
        )
    except Exception as exc:
        state.update(
            status="FAIL_HCP_BUILD",
            completed_utc=utc_now(),
            elapsed_seconds=round(time.monotonic() - started, 3),
            error=str(exc),
        )
    atomic_json(result_path, state)
    return state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fsid", default=DEFAULT_FSID)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--nthreads", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = process(args.fsid, args.root, args.nthreads)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    if result["status"] not in {
        "PASS_ALL_NINE_EXISTING_TRACK",
        "FAIL_TRACK_REBUILD_REQUIRED",
    }:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
