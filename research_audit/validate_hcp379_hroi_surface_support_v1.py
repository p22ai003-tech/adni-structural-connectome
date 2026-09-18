#!/home/ec2-user/fsl/bin/python
"""Validate H-ROI surface-support candidates on existing 3M tractograms."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import nibabel as nib
import numpy as np


DEFAULT_UNITS = (
    "002_S_0413_I863064",
    "114_S_6595_I1078732",
    "003_S_6915_I1423373",
    "014_S_6199_I1167004",
)
HCP_ROOT = Path("/data/derivatives/hcp379_v2")
CANDIDATE_ROOT = (
    HCP_ROOT / "source_label_repair_v1/hroi_surface_support_v1"
)
OUTPUT_ROOT = (
    HCP_ROOT
    / "source_label_repair_v1/validation/hroi_surface_support_v1/attempts"
)
FREESURFER = Path("/home/ec2-user/freesurfer")
MRTRIX = Path("/home/ec2-user/mrtrix3")
FSL_PYTHON = Path("/home/ec2-user/fsl/bin/python")
RELABEL = Path("/home/ec2-user/exp/scripts/hcp/relabel_hcp.py")
LUT = Path("/data/derivatives/parc_hcpmmp1/hcpmmp1_subcort_relabel.txt")
POSSIBLE_EDGES = 379 * 378 // 2


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise ValueError(f"not a regular file: {resolved}")
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def run_logged(command: list[str], log: Path, environment: dict[str, str]) -> None:
    with log.open("x", encoding="utf-8") as handle:
        result = subprocess.run(
            command,
            check=False,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            env=environment,
        )
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed with {result.returncode}: {' '.join(command)}"
        )


def validate_unit(unit: str, attempt: Path, nthreads: int) -> dict[str, Any]:
    output = attempt / unit
    output.mkdir(parents=True, exist_ok=False)
    candidate = (
        CANDIDATE_ROOT / unit / "HCPMMP1+aseg_hroi_surface_support.mgz"
    )
    b0 = Path(f"/data/derivatives/dwi_t1_bbr/{unit}_b0mean_ras.nii.gz")
    registration = (
        Path("/data/derivatives/parc_hcpmmp1") / unit / "b0_to_fs.dat"
    )
    archive_tracks = (
        HCP_ROOT / "tracks" / unit / "tracks_regen216_3000k.tck"
    )
    archive_baseline = (
        HCP_ROOT / "connectomes" / f"SC_HCPMMP1_{unit}_count.csv"
    )
    if archive_tracks.is_file() and archive_baseline.is_file():
        tracks = archive_tracks
        baseline = archive_baseline
        track_route = "archive_track_matched_rebuild"
    else:
        tracks = (
            Path("/data/derivatives/tracks")
            / unit
            / "tracks_final_3000k.tck"
        )
        baseline = Path(
            f"/data/derivatives/connectomes_hcp_fs/"
            f"SC_HCPMMP1_{unit}_count.csv"
        )
        track_route = "legacy_hcp_fs"
    for path in (candidate, b0, registration, tracks, baseline, LUT, RELABEL):
        if not path.is_file():
            raise FileNotFoundError(path)
    environment = os.environ.copy()
    environment.update(
        {
            "FREESURFER_HOME": str(FREESURFER),
            "SUBJECTS_DIR": "/data/derivatives/fastsurfer",
            "PATH": (
                f"{MRTRIX / 'bin'}:{FREESURFER / 'bin'}:"
                f"/home/ec2-user/fsl/bin:{environment.get('PATH', '')}"
            ),
        }
    )
    raw_b0 = output / "HCPMMP1+aseg_hroi_surface_support_b0.nii.gz"
    nodes = output / "nodes_hroi_surface_support_b0.nii.gz"
    node_volumes = output / "node_volumes.csv"
    count_path = output / "count.csv"
    run_logged(
        [
            str(FREESURFER / "bin/mri_label2vol"),
            "--seg",
            str(candidate),
            "--temp",
            str(b0),
            "--reg",
            str(registration),
            "--o",
            str(raw_b0),
        ],
        output / "label2vol.log",
        environment,
    )
    run_logged(
        [
            str(FSL_PYTHON),
            str(RELABEL),
            str(raw_b0),
            str(LUT),
            str(nodes),
            str(node_volumes),
        ],
        output / "relabel.log",
        environment,
    )
    run_logged(
        [
            str(MRTRIX / "bin/tck2connectome"),
            str(tracks),
            str(nodes),
            str(count_path),
            "-symmetric",
            "-zero_diagonal",
            "-assignment_radial_search",
            "4",
            "-stat_edge",
            "sum",
            "-force",
            "-quiet",
            "-nthreads",
            str(nthreads),
        ],
        output / "tck2connectome.log",
        environment,
    )
    old = np.loadtxt(baseline, delimiter=",")
    new = np.loadtxt(count_path, delimiter=",")
    node_data = np.asanyarray(nib.load(str(nodes)).dataobj).astype(np.int32)
    labels = set(int(value) for value in np.unique(node_data))
    matrix_contract = (
        old.shape == new.shape == (379, 379)
        and np.isfinite(new).all()
        and (new >= 0).all()
        and np.allclose(new, new.T)
        and np.count_nonzero(np.diag(new)) == 0
    )
    all_nodes = set(range(1, 380)).issubset(labels)
    old_edges = np.triu(old, 1) > 0
    new_edges = np.triu(new, 1) > 0
    h_nodes_connected = (
        np.count_nonzero(new[119]) > 0 and np.count_nonzero(new[299]) > 0
    )
    result = {
        "unit": unit,
        "status": (
            "PASS_TECHNICAL_ASSIGNMENT"
            if matrix_contract
            and all_nodes
            else "FAIL"
        ),
        "diagnosis_labels_used": False,
        "track_streamline_count": 3_000_000,
        "track_route": track_route,
        "matrix_contract_pass": bool(matrix_contract),
        "node_labels_present": len(labels & set(range(1, 380))),
        "node120_voxels_dwi": int((node_data == 120).sum()),
        "node300_voxels_dwi": int((node_data == 300).sum()),
        "old_density": float(old_edges.sum() / POSSIBLE_EDGES),
        "new_density": float(new_edges.sum() / POSSIBLE_EDGES),
        "density_delta": float(
            (new_edges.sum() - old_edges.sum()) / POSSIBLE_EDGES
        ),
        "old_edge_n": int(old_edges.sum()),
        "new_edge_n": int(new_edges.sum()),
        "gained_edge_n": int((new_edges & ~old_edges).sum()),
        "lost_edge_n": int((old_edges & ~new_edges).sum()),
        "node120_old_degree": int(np.count_nonzero(old[119])),
        "node120_new_degree": int(np.count_nonzero(new[119])),
        "node300_old_degree": int(np.count_nonzero(old[299])),
        "node300_new_degree": int(np.count_nonzero(new[299])),
        "h_nodes_connected_in_legacy_3m": bool(h_nodes_connected),
        "assigned_streamline_count_old": float(old.sum() / 2),
        "assigned_streamline_count_new": float(new.sum() / 2),
        "inputs": {
            "candidate": file_record(candidate),
            "tracks": file_record(tracks),
            "baseline_count": file_record(baseline),
        },
        "outputs": {
            "nodes": file_record(nodes),
            "node_volumes": file_record(node_volumes),
            "count": file_record(count_path),
        },
    }
    atomic_json(output / "comparison.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads-per-worker", type=int, default=8)
    parser.add_argument("--units", nargs="+", default=list(DEFAULT_UNITS))
    args = parser.parse_args()
    if not 1 <= args.workers <= 4 or not 1 <= args.threads_per_worker <= 16:
        parser.error("worker/thread values are outside the validation contract")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    attempt = OUTPUT_ROOT / f"{stamp}-hroi-assignment-validation"
    attempt.mkdir(parents=True, exist_ok=False)
    results: dict[str, dict[str, Any]] = {}
    failures: dict[str, str] = {}
    units = tuple(dict.fromkeys(str(unit).strip() for unit in args.units))
    if not units or any(not unit or "/" in unit for unit in units):
        parser.error("--units contains an invalid unit")
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                validate_unit, unit, attempt, args.threads_per_worker
            ): unit
            for unit in units
        }
        for future in as_completed(futures):
            unit = futures[future]
            try:
                results[unit] = future.result()
            except Exception as exc:
                failures[unit] = f"{type(exc).__name__}: {exc}"
    ordered = [results[unit] for unit in units if unit in results]
    technical_pass = (
        not failures
        and len(ordered) == len(units)
        and all(row["status"] == "PASS_TECHNICAL_ASSIGNMENT" for row in ordered)
    )
    summary = {
        "schema_version": "1.0.0",
        "record_type": "hcp379_hroi_surface_support_assignment_validation",
        "status": (
            "TECHNICAL_PASS_SCIENTIFIC_POLICY_REVIEW_PENDING"
            if technical_pass
            else "FAIL"
        ),
        "generated_utc": utc_now(),
        "diagnosis_labels_used": False,
        "source_files_modified": False,
        "unit_n": len(ordered),
        "h_nodes_connected_in_legacy_3m_n": sum(
            bool(row["h_nodes_connected_in_legacy_3m"]) for row in ordered
        ),
        "units": ordered,
        "failures": failures,
        "implementation": file_record(Path(__file__)),
    }
    atomic_json(attempt / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if technical_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
