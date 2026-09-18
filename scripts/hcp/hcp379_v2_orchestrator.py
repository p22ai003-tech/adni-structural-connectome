#!/usr/bin/env python3
"""Non-overwriting recovery orchestrator for the HCP-MMP1 379-node cohort.

The v2 workspace never writes into the historical tracks, parcellations, or
connectome directories.  Its first executable lane restores the 86 distinct
regen216 tractograms from S3 and rebuilds count/geometry matrices against the
already audited HCP node images.  SIFT2 and diffusion-scalar matrices remain
withheld until track-matched sidecars are regenerated.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


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

EXP = Path("/home/ec2-user/exp")
TRIAGE_DIR = EXP / "research_audit/outputs/hcp_rerun_triage_v1"
TRIAGE_CSV = TRIAGE_DIR / "hcp_rerun_triage_manifest.csv"
ARCHIVE_SCREEN_CSV = Path("/tmp/hcp_s3_fullscreen.kk0_n6gw/results.csv")
CURRENT_TRACK_ROOT = Path("/data/derivatives/tracks")
CURRENT_PARC_ROOT = Path("/data/derivatives/parc_hcpmmp1")
CURRENT_MATRIX_ROOT = Path("/data/derivatives/connectomes_hcp_fs")
V2_ROOT = Path("/data/derivatives/hcp379_v2")
S3_TRACK_ROOT = "s3://sabeesh/exp/tracks"
MRTRIX_BIN = Path("/home/ec2-user/mrtrix3/bin")

# A voxel-count threshold is invalid across this cohort because the archived
# DWI grids range from roughly 1.37 x 1.37 x 2.7 mm to 2.67 x 2.67 x 2 mm.
# Gate the whole-brain DWI mask in physical space instead, then require tensor
# maps to support almost all of that subject-specific mask.  The 0.75 L floor
# is deliberately conservative relative to the diagnosis-blind archive
# recovery distribution (currently 1.07--1.69 L).
MIN_BRAIN_MASK_VOLUME_MM3 = 750_000.0
MIN_TENSOR_MASK_SUPPORT_FRACTION = 0.95


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp",
        delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        temp_path = Path(handle.name)
    os.replace(temp_path, path)


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with tempfile.NamedTemporaryFile(
        mode="w", newline="", dir=path.parent, prefix=f".{path.name}.",
        suffix=".tmp", delete=False
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        temp_path = Path(handle.name)
    os.replace(temp_path, path)


def sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tck_header(path: Path) -> str:
    with path.open("rb") as handle:
        raw = handle.read(65536)
    return raw.split(b"END\n", 1)[0].decode("utf-8", "replace")


def header_sha256(path: Path) -> str:
    return hashlib.sha256(tck_header(path).encode()).hexdigest()


def ensure_layout(root: Path) -> None:
    for relative in (
        "tracks",
        "streamline_metrics",
        "connectomes",
        "assignments",
        "qc/subjects",
        "logs/subjects",
        "locks",
        "manifests",
        "provenance",
        "work",
    ):
        (root / relative).mkdir(parents=True, exist_ok=True)


def command_version(command: list[str]) -> str:
    try:
        result = subprocess.run(
            command, check=False, capture_output=True, text=True, timeout=30
        )
        text = (result.stdout or result.stderr).strip()
        return text[:4000]
    except Exception as exc:
        return f"ERROR: {exc}"


def freeze(root: Path) -> None:
    ensure_layout(root)
    triage = read_csv(TRIAGE_CSV)
    started = utc_now()
    rows: list[dict[str, Any]] = []

    for index, subject in enumerate(triage, 1):
        sid = subject["fsid"]
        track = CURRENT_TRACK_ROOT / sid / "tracks_final_3000k.tck"
        nodes = CURRENT_PARC_ROOT / sid / "nodes_b0.nii.gz"
        node_volumes = CURRENT_PARC_ROOT / sid / "node_volumes.csv"
        row: dict[str, Any] = {
            "fsid": sid,
            "group": subject["group"],
            "lane": subject["lane"],
            "recommended_action": subject["recommended_action"],
            "freeze_utc": started,
            "track_path": str(track),
            "track_exists": track.exists(),
            "nodes_path": str(nodes),
            "nodes_exists": nodes.exists(),
            "node_volumes_path": str(node_volumes),
            "node_volumes_exists": node_volumes.exists(),
        }
        if track.exists():
            stat = track.stat()
            row.update(
                track_size_bytes=stat.st_size,
                track_mtime_ns=stat.st_mtime_ns,
                track_header_sha256=header_sha256(track),
            )
        if nodes.exists():
            stat = nodes.stat()
            row.update(nodes_size_bytes=stat.st_size, nodes_mtime_ns=stat.st_mtime_ns)
        if node_volumes.exists():
            row["node_volumes_sha256"] = sha256(node_volumes)

        for suffix in MATRIX_SUFFIXES:
            matrix = CURRENT_MATRIX_ROOT / f"SC_HCPMMP1_{sid}_{suffix}.csv"
            prefix = f"matrix_{suffix}"
            row[f"{prefix}_path"] = str(matrix)
            row[f"{prefix}_exists"] = matrix.exists()
            if matrix.exists():
                stat = matrix.stat()
                row[f"{prefix}_size_bytes"] = stat.st_size
                row[f"{prefix}_mtime_ns"] = stat.st_mtime_ns
                row[f"{prefix}_sha256"] = sha256(matrix)
        rows.append(row)
        if index % 25 == 0:
            print(f"[{utc_now()}] freeze {index}/{len(triage)}", flush=True)

    manifest = root / "manifests/historical_freeze_manifest.csv"
    atomic_csv(manifest, rows)
    manifest_hash = sha256(manifest)
    versions = {
        "python": sys.version,
        "numpy": np.__version__,
        "aws": command_version(["aws", "--version"]),
        "tck2connectome": command_version(
            [str(MRTRIX_BIN / "tck2connectome"), "-version"]
        ),
        "mrinfo": command_version([str(MRTRIX_BIN / "mrinfo"), "-version"]),
    }
    atomic_json(root / "provenance/tool_versions.json", versions)
    atomic_json(
        root / "manifests/historical_freeze_summary.json",
        {
            "schema_version": "1.0.0",
            "started_utc": started,
            "completed_utc": utc_now(),
            "n_subjects": len(rows),
            "manifest": str(manifest),
            "manifest_sha256": manifest_hash,
            "source_triage_manifest": str(TRIAGE_CSV),
            "source_triage_sha256": sha256(TRIAGE_CSV),
            "track_integrity_scope": "size, mtime, and complete TCK header hash",
            "matrix_integrity_scope": "full SHA-256 for every available CSV",
            "node_integrity_scope": "size/mtime for NIfTI; full SHA-256 for node-volume CSV",
            "non_overwriting": True,
        },
    )
    print(f"[{utc_now()}] FREEZE_DONE n={len(rows)} manifest={manifest}", flush=True)


def run_logged(command: list[str], log_handle: Any) -> None:
    log_handle.write(f"\n[{utc_now()}] COMMAND {json.dumps(command)}\n")
    log_handle.flush()
    result = subprocess.run(
        command, stdout=log_handle, stderr=subprocess.STDOUT, text=True
    )
    if result.returncode:
        raise RuntimeError(f"command failed rc={result.returncode}: {command[0]}")


def validate_square(matrix: np.ndarray, suffix: str) -> list[str]:
    failures: list[str] = []
    if matrix.shape != (N_NODES, N_NODES):
        failures.append(f"{suffix}:shape={matrix.shape}")
        return failures
    if not np.isfinite(matrix).all():
        failures.append(f"{suffix}:nonfinite")
    if not np.allclose(matrix, matrix.T, atol=1e-6, rtol=1e-6):
        failures.append(f"{suffix}:asymmetric")
    if not np.allclose(np.diag(matrix), 0.0, atol=1e-8):
        failures.append(f"{suffix}:diagonal")
    if float(np.nanmin(matrix)) < -1e-10:
        failures.append(f"{suffix}:negative")
    return failures


def node_volumes(path: Path) -> np.ndarray:
    values: dict[int, float] = {}
    for row in read_csv(path):
        values[int(row["node_id"])] = float(row["volume_mm3"])
    return np.array([values.get(node, 0.0) for node in range(1, N_NODES + 1)])


def archive_subject(
    sid: str,
    root: Path,
    screen: dict[str, dict[str, str]],
    nthreads: int,
) -> dict[str, Any]:
    ensure_layout(root)
    lock_path = root / "locks" / f"{sid}.lock"
    with lock_path.open("w") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        state_path = root / "qc/subjects" / f"{sid}.json"
        if state_path.exists():
            try:
                prior = json.loads(state_path.read_text())
                if prior.get("status") in {"PASS_COUNT_GEOMETRY", "PASS_ALL_NINE"}:
                    return prior
            except Exception:
                pass

        started = time.monotonic()
        log_path = root / "logs/subjects" / f"{sid}.log"
        track_dir = root / "tracks" / sid
        track_dir.mkdir(parents=True, exist_ok=True)
        final_track = track_dir / "tracks_regen216_3000k.tck"
        nodes = CURRENT_PARC_ROOT / sid / "nodes_b0.nii.gz"
        volumes_csv = CURRENT_PARC_ROOT / sid / "node_volumes.csv"
        prefix = root / "connectomes" / f"SC_HCPMMP1_{sid}"
        assignment_final = root / "assignments" / f"{sid}_radial4.csv"
        state: dict[str, Any] = {
            "fsid": sid,
            "lane": "ARCHIVE_RECOVERY",
            "started_utc": utc_now(),
            "status": "RUNNING",
            "source_s3": f"{S3_TRACK_ROOT}/{sid}/tracks_final_3000k.tck",
            "nodes_source": str(nodes),
            "weighted_status": "WITHHELD_UNMATCHED_SIDECARS",
        }
        try:
            if not nodes.exists() or not volumes_csv.exists():
                raise FileNotFoundError(f"missing nodes or node volumes for {sid}")

            with log_path.open("a") as log:
                if not final_track.exists():
                    part = final_track.with_suffix(".tck.part")
                    if part.exists():
                        part.unlink()
                    run_logged(
                        [
                            "aws", "s3", "cp", "--only-show-errors",
                            state["source_s3"], str(part),
                        ],
                        log,
                    )
                    if part.stat().st_size < 1024:
                        raise RuntimeError("downloaded tractogram is implausibly small")
                    os.replace(part, final_track)

                header = tck_header(final_track)
                count_match = re.search(r"^count:\s*(\d+)", header, re.MULTILINE)
                track_count = int(count_match.group(1)) if count_match else 0
                if track_count != 3_000_000:
                    raise RuntimeError(f"unexpected tractogram count {track_count}")
                if " -act " in header:
                    raise RuntimeError("archive tractogram unexpectedly contains ACT")
                if "seed_dynamic" not in header:
                    raise RuntimeError("archive tractogram lacks dynamic-seeding provenance")

                with tempfile.TemporaryDirectory(
                    prefix=f"{sid}.", dir=root / "work"
                ) as temp_dir_name:
                    temp_dir = Path(temp_dir_name)
                    temp_count = temp_dir / "count.csv"
                    temp_len = temp_dir / "len_mean.csv"
                    temp_invlen = temp_dir / "invlen_mean.csv"
                    temp_invnode = temp_dir / "count_invnodevol.csv"
                    temp_assignments = temp_dir / "assignments.csv"
                    common = [
                        "-symmetric", "-zero_diagonal",
                        "-assignment_radial_search", "4",
                        "-force", "-quiet", "-nthreads", str(nthreads),
                    ]
                    tck2 = str(MRTRIX_BIN / "tck2connectome")
                    run_logged(
                        [
                            tck2, str(final_track), str(nodes), str(temp_count),
                            *common, "-stat_edge", "sum",
                            "-out_assignments", str(temp_assignments),
                        ],
                        log,
                    )
                    run_logged(
                        [
                            tck2, str(final_track), str(nodes), str(temp_len),
                            *common, "-scale_length", "-stat_edge", "mean",
                        ],
                        log,
                    )
                    run_logged(
                        [
                            tck2, str(final_track), str(nodes), str(temp_invlen),
                            *common, "-scale_invlength", "-stat_edge", "mean",
                        ],
                        log,
                    )

                    count = np.loadtxt(temp_count, delimiter=",")
                    length = np.loadtxt(temp_len, delimiter=",")
                    invlength = np.loadtxt(temp_invlen, delimiter=",")
                    failures = []
                    failures.extend(validate_square(count, "count"))
                    failures.extend(validate_square(length, "len_mean"))
                    failures.extend(validate_square(invlength, "invlen_mean"))

                    volumes = node_volumes(volumes_csv)
                    denominator = volumes[:, None] + volumes[None, :]
                    with np.errstate(divide="ignore", invalid="ignore"):
                        invnode = np.where(
                            denominator > 0.0, 2.0 * count / denominator, 0.0
                        )
                    np.fill_diagonal(invnode, 0.0)
                    np.savetxt(temp_invnode, invnode, delimiter=",", fmt="%.10g")
                    failures.extend(validate_square(invnode, "count_invnodevol"))

                    strength = count.sum(axis=1)
                    present = int((strength > 0).sum())
                    density = float(
                        np.count_nonzero(count) / (N_NODES * (N_NODES - 1))
                    )
                    assigned = float(count.sum() / 2.0)
                    min_hub = float(min(strength[node - 1] for node in HUB_IDS))
                    expected = screen[sid]
                    expected_density = float(expected["density"])
                    if present < 360:
                        failures.append(f"nodes_present={present}<360")
                    if min_hub <= 0:
                        failures.append("hub_support=0")
                    if abs(density - expected_density) > 1e-9:
                        failures.append(
                            f"density_drift={density:.12g}!={expected_density:.12g}"
                        )
                    if failures:
                        raise RuntimeError(";".join(failures))

                    output_map = {
                        temp_count: Path(f"{prefix}_count.csv"),
                        temp_len: Path(f"{prefix}_len_mean.csv"),
                        temp_invlen: Path(f"{prefix}_invlen_mean.csv"),
                        temp_invnode: Path(f"{prefix}_count_invnodevol.csv"),
                        temp_assignments: assignment_final,
                    }
                    for source, destination in output_map.items():
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        os.replace(source, destination)

            state.update(
                status="PASS_COUNT_GEOMETRY",
                completed_utc=utc_now(),
                elapsed_seconds=round(time.monotonic() - started, 3),
                track_path=str(final_track),
                track_size_bytes=final_track.stat().st_size,
                track_header_sha256=header_sha256(final_track),
                nodes_present=present,
                density=density,
                assigned_streamlines=assigned,
                assignment_fraction=assigned / track_count,
                min_hub_strength=min_hub,
                generated_matrices=[
                    "count", "len_mean", "invlen_mean", "count_invnodevol"
                ],
                withheld_matrices=[
                    "fd_sum", "fa_mean", "md_mean", "rd_mean", "ad_mean"
                ],
            )
        except Exception as exc:
            state.update(
                status="FAIL",
                completed_utc=utc_now(),
                elapsed_seconds=round(time.monotonic() - started, 3),
                error=str(exc),
            )
        atomic_json(state_path, state)
        return state


def archive_restore(
    root: Path, workers: int, nthreads: int, limit: int | None
) -> None:
    ensure_layout(root)
    triage = read_csv(TRIAGE_CSV)
    subjects = [
        row["fsid"] for row in triage if row["lane"] == "ARCHIVE_RECOVERY"
    ]
    screen_rows = read_csv(ARCHIVE_SCREEN_CSV)
    screen = {row["sid"]: row for row in screen_rows}
    missing = sorted(set(subjects) - set(screen))
    if missing:
        raise RuntimeError(f"archive screen is missing {len(missing)} subjects")
    if limit is not None:
        subjects = subjects[:limit]

    print(
        f"[{utc_now()}] ARCHIVE_START n={len(subjects)} "
        f"workers={workers} nthreads={nthreads}",
        flush=True,
    )
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(archive_subject, sid, root, screen, nthreads): sid
            for sid in subjects
        }
        for index, future in enumerate(as_completed(futures), 1):
            result = future.result()
            results.append(result)
            print(
                f"[{utc_now()}] archive {index}/{len(subjects)} "
                f"{result['fsid']} {result['status']} "
                f"d={result.get('density', 'NA')} "
                f"t={result.get('elapsed_seconds', 'NA')}s",
                flush=True,
            )
            write_archive_summary(root)
    write_archive_summary(root)
    failures = [row for row in results if row["status"] == "FAIL"]
    if failures:
        raise RuntimeError(f"{len(failures)} archive subjects failed")


def mrstats_nonzero_voxels(path: Path) -> int:
    result = subprocess.run(
        [
            str(MRTRIX_BIN / "mrstats"), str(path),
            "-output", "count", "-ignorezero",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        return 0
    try:
        return int(float(result.stdout.strip()))
    except ValueError:
        return 0


def mrinfo_voxel_volume_mm3(path: Path) -> float:
    result = subprocess.run(
        [str(MRTRIX_BIN / "mrinfo"), str(path), "-spacing"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        return 0.0
    try:
        spacing = [float(value) for value in result.stdout.split()[:3]]
    except ValueError:
        return 0.0
    if len(spacing) != 3 or not all(np.isfinite(spacing)) \
            or any(value <= 0 for value in spacing):
        return 0.0
    return float(np.prod(spacing))


def mask_physical_qc(path: Path) -> dict[str, float]:
    nonzero_voxels = mrstats_nonzero_voxels(path)
    voxel_volume_mm3 = mrinfo_voxel_volume_mm3(path)
    return {
        "nonzero_voxels": float(nonzero_voxels),
        "voxel_volume_mm3": voxel_volume_mm3,
        "volume_mm3": float(nonzero_voxels) * voxel_volume_mm3,
    }


def has_diffusion_gradients(path: Path) -> bool:
    result = subprocess.run(
        [str(MRTRIX_BIN / "mrinfo"), str(path), "-dwgrad"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0 and bool(result.stdout.strip())


def archive_weights_subject(
    sid: str,
    root: Path,
    nthreads: int,
) -> dict[str, Any]:
    """Rebuild track-matched SIFT2/scalar sidecars and the five weighted matrices."""
    ensure_layout(root)
    lock_path = root / "locks" / f"{sid}.lock"
    with lock_path.open("w") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        state_path = root / "qc/subjects" / f"{sid}.json"
        if not state_path.exists():
            return {
                "fsid": sid,
                "lane": "ARCHIVE_RECOVERY",
                "status": "FAIL_WEIGHTED",
                "error": "count/geometry state is missing",
            }
        state = json.loads(state_path.read_text())
        if state.get("status") == "PASS_ALL_NINE":
            provenance_changed = False
            completed_mask = (
                root / "streamline_metrics" / sid / "mask_fod.mif"
            )
            if completed_mask.exists() and not state.get("mask_volume_mm3"):
                completed_mask_qc = mask_physical_qc(completed_mask)
                state["mask_voxels"] = int(
                    completed_mask_qc["nonzero_voxels"]
                )
                state["mask_voxel_volume_mm3"] = completed_mask_qc[
                    "voxel_volume_mm3"
                ]
                state["mask_volume_mm3"] = completed_mask_qc["volume_mm3"]
                for stats in state.get("scalar_map_stats", {}).values():
                    stats["mask_support_fraction"] = (
                        float(stats["nonzero_voxels"])
                        / max(float(state["mask_voxels"]), 1.0)
                    )
                provenance_changed = True
            if not state.get("track_sha256"):
                completed_track = (
                    root / "tracks" / sid / "tracks_regen216_3000k.tck"
                )
                if completed_track.exists():
                    state["track_sha256"] = sha256(completed_track)
                    provenance_changed = True
            if not state.get("scalar_sample_sha256"):
                sample_paths = {
                    metric: (
                        root / "streamline_metrics" / sid
                        / f"{metric}_tensor_v2_mean.tsf"
                    )
                    for metric in ("fa", "md", "rd", "ad")
                }
                if all(path.exists() for path in sample_paths.values()):
                    state["scalar_sample_sha256"] = {
                        metric: sha256(path)
                        for metric, path in sample_paths.items()
                    }
                    provenance_changed = True
            if not state.get("scalar_map_sha256"):
                map_paths = {
                    metric: (
                        root / "streamline_metrics" / sid / "tensor_v2"
                        / f"{metric}.mif"
                    )
                    for metric in ("fa", "md", "rd", "ad")
                }
                if all(path.exists() for path in map_paths.values()):
                    state["scalar_map_sha256"] = {
                        metric: sha256(path)
                        for metric, path in map_paths.items()
                    }
                    provenance_changed = True
            if provenance_changed:
                atomic_json(state_path, state)
            return state
        count_prerequisite = (
            Path(root / "connectomes" / f"SC_HCPMMP1_{sid}_count.csv").exists()
            and state.get("nodes_present", 0) >= 360
        )
        if state.get("status") not in {"PASS_COUNT_GEOMETRY", "FAIL_WEIGHTED"} \
                or not count_prerequisite:
            state.update(
                status="FAIL_WEIGHTED",
                error=f"count/geometry prerequisite status={state.get('status')}",
                completed_utc=utc_now(),
            )
            atomic_json(state_path, state)
            return state

        started = time.monotonic()
        log_path = root / "logs/subjects" / f"{sid}_weights.log"
        track = root / "tracks" / sid / "tracks_regen216_3000k.tck"
        nodes = CURRENT_PARC_ROOT / sid / "nodes_b0.nii.gz"
        metric_dir = root / "streamline_metrics" / sid
        metric_dir.mkdir(parents=True, exist_ok=True)
        eddy_original = Path("/data/derivatives/eddy") / f"{sid}_preproc.mif"
        legacy_fod_dir = Path("/data/derivatives/fod") / sid
        response = legacy_fod_dir / "wm_csd.txt"
        ss3t_response = legacy_fod_dir / "wm.txt"
        legacy_ss3t_fod = legacy_fod_dir / "wmfod_final.mif"
        deconv_mode_path = legacy_fod_dir / "deconv_mode.txt"
        deconv_mode = (
            deconv_mode_path.read_text().strip()
            if deconv_mode_path.exists()
            else ""
        )
        if response.exists():
            fod_source_mode = "CSD_REBUILT_FROM_EDDY_AND_WM_CSD_RESPONSE"
            response_provenance_path = response
        elif (
            deconv_mode == "ss3t"
            and ss3t_response.exists()
            and legacy_ss3t_fod.exists()
        ):
            # SS3T does not create the single-shell ``wm_csd.txt`` sidecar.
            # Keep the exact FOD that generated this tractogram: replacing it
            # with a single-tissue CSD fit would make SIFT2 less track-matched.
            fod_source_mode = "EXACT_TRACK_GENERATING_SS3T_FOD_COPY"
            response_provenance_path = ss3t_response
        else:
            fod_source_mode = "UNAVAILABLE"
            response_provenance_path = response
        original_mask = legacy_fod_dir / "mask_fod.mif"
        fod = metric_dir / "wmfod_rebuilt.mif"
        mask = metric_dir / "mask_fod.mif"
        weights = metric_dir / "sift2_weights.txt"
        prefix = root / "connectomes" / f"SC_HCPMMP1_{sid}"
        state["weighted_status"] = "RUNNING_MATCHED_REBUILD"
        state["weighted_started_utc"] = utc_now()
        atomic_json(state_path, state)

        try:
            required = [track, nodes, eddy_original]
            if fod_source_mode == "CSD_REBUILT_FROM_EDDY_AND_WM_CSD_RESPONSE":
                required.append(response)
            elif fod_source_mode == "EXACT_TRACK_GENERATING_SS3T_FOD_COPY":
                required.extend([ss3t_response, legacy_ss3t_fod])
            else:
                required.extend([response, ss3t_response, legacy_ss3t_fod])
            absent = [str(path) for path in required if not path.exists()]
            if absent:
                raise FileNotFoundError(
                    "missing weighted inputs for deconvolution mode "
                    f"{deconv_mode or 'UNKNOWN'}: " + ",".join(absent)
                )

            with log_path.open("a") as log:
                eddy = eddy_original
                if not has_diffusion_gradients(eddy_original):
                    bvec = Path("/data/derivatives/mif_dwi") / f"{sid}.bvec"
                    bval = Path("/data/derivatives/mif_dwi") / f"{sid}.bval"
                    if not bvec.exists() or not bval.exists():
                        raise FileNotFoundError("eddy has no gradients and bvec/bval are absent")
                    eddy = metric_dir / "eddy_gradfix.mif"
                    if not eddy.exists():
                        run_logged(
                            [
                                str(MRTRIX_BIN / "mrconvert"), str(eddy_original),
                                str(eddy), "-fslgrad", str(bvec), str(bval),
                                "-force",
                            ],
                            log,
                        )

                if not mask.exists():
                    run_logged(
                        [
                            str(MRTRIX_BIN / "dwi2mask"), str(eddy), str(mask),
                            "-nthreads", str(nthreads), "-force",
                        ],
                        log,
                    )
                    fresh_mask_qc = mask_physical_qc(mask)
                    if (
                        fresh_mask_qc["volume_mm3"]
                        < MIN_BRAIN_MASK_VOLUME_MM3
                        and original_mask.exists()
                    ):
                        original_mask_qc = mask_physical_qc(original_mask)
                        if original_mask_qc["volume_mm3"] > max(
                            fresh_mask_qc["volume_mm3"] * 2,
                            MIN_BRAIN_MASK_VOLUME_MM3,
                        ):
                            run_logged(
                                [
                                    str(MRTRIX_BIN / "mrgrid"), str(original_mask),
                                    "regrid", "-template", str(eddy),
                                    "-interp", "nearest", "-datatype", "bit",
                                    str(mask), "-force",
                                ],
                                log,
                            )
                            log.write(
                                f"[{utc_now()}] MASK_FALLBACK "
                                f"fresh_mm3={fresh_mask_qc['volume_mm3']:.3f} "
                                f"original_mm3="
                                f"{original_mask_qc['volume_mm3']:.3f}\n"
                            )
                mask_qc = mask_physical_qc(mask)
                mask_voxels = int(mask_qc["nonzero_voxels"])
                if mask_qc["volume_mm3"] < MIN_BRAIN_MASK_VOLUME_MM3:
                    raise RuntimeError(
                        "FOD mask remains implausibly small in physical space: "
                        f"{mask_qc['volume_mm3']:.3f} mm3 "
                        f"({mask_voxels} voxels x "
                        f"{mask_qc['voxel_volume_mm3']:.6g} mm3)"
                    )

                if (
                    not fod.exists()
                    and fod_source_mode
                    == "CSD_REBUILT_FROM_EDDY_AND_WM_CSD_RESPONSE"
                ):
                    run_logged(
                        [
                            str(MRTRIX_BIN / "dwi2fod"), "csd", str(eddy),
                            str(response), str(fod), "-mask", str(mask),
                            "-nthreads", str(nthreads), "-force",
                        ],
                        log,
                    )
                elif (
                    not fod.exists()
                    and fod_source_mode
                    == "EXACT_TRACK_GENERATING_SS3T_FOD_COPY"
                ):
                    run_logged(
                        [
                            str(MRTRIX_BIN / "mrconvert"),
                            str(legacy_ss3t_fod), str(fod), "-force",
                        ],
                        log,
                    )

                tensor_dir = metric_dir / "tensor_v2"
                tensor_dir.mkdir(parents=True, exist_ok=True)
                tensor = tensor_dir / "dt.mif"
                raw_metrics = {
                    metric: tensor_dir / f"{metric}_raw.mif"
                    for metric in ("fa", "md", "rd", "ad")
                }
                bounded_metrics = {
                    metric: tensor_dir / f"{metric}.mif"
                    for metric in ("fa", "md", "rd", "ad")
                }
                if not tensor.exists():
                    run_logged(
                        [
                            str(MRTRIX_BIN / "dwi2tensor"), str(eddy), str(tensor),
                            "-mask", str(mask), "-nthreads", str(nthreads),
                            "-force",
                        ],
                        log,
                    )
                if not all(path.exists() for path in raw_metrics.values()):
                    run_logged(
                        [
                            str(MRTRIX_BIN / "tensor2metric"), str(tensor),
                            "-fa", str(raw_metrics["fa"]),
                            "-adc", str(raw_metrics["md"]),
                            "-rd", str(raw_metrics["rd"]),
                            "-ad", str(raw_metrics["ad"]),
                            "-nthreads", str(nthreads), "-force",
                        ],
                        log,
                    )
                if not bounded_metrics["fa"].exists():
                    run_logged(
                        [
                            str(MRTRIX_BIN / "mrcalc"), str(raw_metrics["fa"]),
                            "0", "-max", "1", "-min",
                            str(bounded_metrics["fa"]), "-force",
                        ],
                        log,
                    )
                for metric in ("md", "rd", "ad"):
                    if not bounded_metrics[metric].exists():
                        run_logged(
                            [
                                str(MRTRIX_BIN / "mrcalc"),
                                str(raw_metrics[metric]), "0", "-max",
                                str(bounded_metrics[metric]), "-force",
                            ],
                            log,
                        )

                scalar_map_stats: dict[str, dict[str, float]] = {}
                for metric, metric_path in bounded_metrics.items():
                    result = subprocess.run(
                        [
                            str(MRTRIX_BIN / "mrstats"), str(metric_path),
                            "-mask", str(mask), "-output", "count",
                            "-output", "mean", "-output", "min", "-output", "max",
                            "-ignorezero",
                        ],
                        check=True,
                        capture_output=True,
                        text=True,
                    )
                    values = [float(value) for value in result.stdout.split()]
                    if len(values) != 4:
                        raise RuntimeError(
                            f"unexpected mrstats output for {metric}: {result.stdout}"
                        )
                    count, mean, minimum, maximum = values
                    scalar_map_stats[metric] = {
                        "nonzero_voxels": count,
                        "mask_support_fraction": (
                            count / max(float(mask_voxels), 1.0)
                        ),
                        "mean": mean,
                        "min": minimum,
                        "max": maximum,
                    }
                    if (
                        scalar_map_stats[metric]["mask_support_fraction"]
                        < MIN_TENSOR_MASK_SUPPORT_FRACTION
                    ):
                        raise RuntimeError(
                            f"{metric} tensor-map mask support is only "
                            f"{scalar_map_stats[metric]['mask_support_fraction']:.6f} "
                            f"(<{MIN_TENSOR_MASK_SUPPORT_FRACTION:.2f}; "
                            f"{count:g}/{mask_voxels:g} voxels)"
                        )
                    if metric == "fa" and not (
                        0.0 <= minimum <= maximum <= 1.000001
                    ):
                        raise RuntimeError(f"FA tensor map is nonphysical: {values}")
                    if metric != "fa" and minimum < -1e-10:
                        raise RuntimeError(
                            f"{metric} tensor map remains negative: {minimum}"
                        )

                if not weights.exists():
                    run_logged(
                        [
                            str(MRTRIX_BIN / "tcksift2"), str(track), str(fod),
                            str(weights), "-nthreads", str(nthreads), "-force",
                        ],
                        log,
                    )

                scalar_samples: dict[str, Path] = {}
                for metric in ("fa", "md", "rd", "ad"):
                    sample = metric_dir / f"{metric}_tensor_v2_mean.tsf"
                    scalar_samples[metric] = sample
                    if not sample.exists():
                        run_logged(
                            [
                                str(MRTRIX_BIN / "tcksample"), str(track),
                                str(bounded_metrics[metric]),
                                str(sample), "-stat_tck", "mean",
                                "-nthreads", str(nthreads), "-force",
                            ],
                            log,
                        )

                with tempfile.TemporaryDirectory(
                    prefix=f"{sid}.weights.", dir=root / "work"
                ) as temp_dir_name:
                    temp_dir = Path(temp_dir_name)
                    specifications = {
                        "fd_sum": [
                            "-tck_weights_in", str(weights), "-stat_edge", "sum"
                        ],
                        "fa_mean": [
                            "-scale_file", str(scalar_samples["fa"]),
                            "-stat_edge", "mean",
                        ],
                        "md_mean": [
                            "-scale_file", str(scalar_samples["md"]),
                            "-stat_edge", "mean",
                        ],
                        "rd_mean": [
                            "-scale_file", str(scalar_samples["rd"]),
                            "-stat_edge", "mean",
                        ],
                        "ad_mean": [
                            "-scale_file", str(scalar_samples["ad"]),
                            "-stat_edge", "mean",
                        ],
                    }
                    temp_outputs: dict[str, Path] = {}
                    common = [
                        "-symmetric", "-zero_diagonal",
                        "-assignment_radial_search", "4",
                        "-force", "-quiet", "-nthreads", str(nthreads),
                    ]
                    for suffix, extra in specifications.items():
                        output = temp_dir / f"{suffix}.csv"
                        temp_outputs[suffix] = output
                        run_logged(
                            [
                                str(MRTRIX_BIN / "tck2connectome"), str(track),
                                str(nodes), str(output), *common, *extra,
                            ],
                            log,
                        )

                    count = np.loadtxt(f"{prefix}_count.csv", delimiter=",")
                    support = np.triu(count > 0, 1)
                    support_denominator = max(int(support.sum()), 1)
                    failures: list[str] = []
                    support_fractions: dict[str, float] = {}
                    ranges: dict[str, dict[str, float]] = {}
                    for suffix, output in temp_outputs.items():
                        matrix = np.loadtxt(output, delimiter=",")
                        failures.extend(validate_square(matrix, suffix))
                        nonzero = matrix[matrix != 0]
                        if nonzero.size:
                            ranges[suffix] = {
                                "min": float(nonzero.min()),
                                "max": float(nonzero.max()),
                            }
                        fraction = float(
                            (np.triu(matrix != 0, 1) & support).sum()
                            / support_denominator
                        )
                        support_fractions[suffix] = fraction
                        if fraction < 0.95:
                            failures.append(f"{suffix}:edge_support={fraction:.6f}<0.95")
                    fa_range = ranges.get("fa_mean")
                    if fa_range and (
                        fa_range["min"] < -1e-6 or fa_range["max"] > 1.000001
                    ):
                        failures.append(f"fa_mean:physical_range={fa_range}")
                    for suffix in ("md_mean", "rd_mean", "ad_mean"):
                        value_range = ranges.get(suffix)
                        if value_range and value_range["min"] < -1e-10:
                            failures.append(f"{suffix}:negative={value_range['min']}")
                    if failures:
                        raise RuntimeError(";".join(sorted(set(failures))))

                    for suffix, output in temp_outputs.items():
                        os.replace(output, Path(f"{prefix}_{suffix}.csv"))

            all_failures: list[str] = []
            matrix_hashes: dict[str, str] = {}
            for suffix in MATRIX_SUFFIXES:
                matrix_path = Path(f"{prefix}_{suffix}.csv")
                if not matrix_path.exists():
                    all_failures.append(f"{suffix}:missing")
                    continue
                matrix = np.loadtxt(matrix_path, delimiter=",")
                all_failures.extend(validate_square(matrix, suffix))
                matrix_hashes[suffix] = sha256(matrix_path)
            if all_failures:
                raise RuntimeError("final nine-matrix gate: " + ";".join(all_failures))

            state.update(
                status="PASS_ALL_NINE",
                weighted_status="PASS_TRACK_MATCHED_REBUILD",
                weighted_completed_utc=utc_now(),
                weighted_elapsed_seconds=round(time.monotonic() - started, 3),
                track_sha256=sha256(track),
                weighted_recipe={
                    "fod": fod_source_mode,
                    "mask": "fresh dwi2mask with audited production-mask fallback",
                    "sift2": "no-ACT tcksift2 against reconstructed FOD",
                    "tensor": (
                        "dwi2tensor and tensor2metric from the same Eddy image "
                        "and tracking mask; FA bounded to [0,1] and negative "
                        "diffusivities set to zero with raw maps retained"
                    ),
                    "scalar_sampling": (
                        "tcksample stat_tck mean from spatially matched "
                        "tensor_v2 FA/MD/RD/AD"
                    ),
                    "archive_auxiliaries_reused": (
                        fod_source_mode
                        == "EXACT_TRACK_GENERATING_SS3T_FOD_COPY"
                    ),
                },
                fod_path=str(fod),
                fod_sha256=sha256(fod),
                fod_source_path=(
                    str(legacy_ss3t_fod)
                    if fod_source_mode
                    == "EXACT_TRACK_GENERATING_SS3T_FOD_COPY"
                    else str(eddy)
                ),
                fod_source_sha256=(
                    sha256(legacy_ss3t_fod)
                    if fod_source_mode
                    == "EXACT_TRACK_GENERATING_SS3T_FOD_COPY"
                    else sha256(eddy)
                ),
                response_path=str(response_provenance_path),
                response_sha256=sha256(response_provenance_path),
                mask_path=str(mask),
                mask_voxels=mask_voxels,
                mask_voxel_volume_mm3=mask_qc["voxel_volume_mm3"],
                mask_volume_mm3=mask_qc["volume_mm3"],
                sift2_weights_path=str(weights),
                sift2_weights_sha256=sha256(weights),
                scalar_sample_paths={
                    metric: str(metric_dir / f"{metric}_tensor_v2_mean.tsf")
                    for metric in ("fa", "md", "rd", "ad")
                },
                scalar_sample_sha256={
                    metric: sha256(metric_dir / f"{metric}_tensor_v2_mean.tsf")
                    for metric in ("fa", "md", "rd", "ad")
                },
                scalar_map_paths={
                    metric: str(bounded_metrics[metric])
                    for metric in ("fa", "md", "rd", "ad")
                },
                scalar_map_sha256={
                    metric: sha256(bounded_metrics[metric])
                    for metric in ("fa", "md", "rd", "ad")
                },
                scalar_map_stats=scalar_map_stats,
                weighted_edge_support=support_fractions,
                weighted_nonzero_ranges=ranges,
                matrix_sha256=matrix_hashes,
                generated_matrices=list(MATRIX_SUFFIXES),
                withheld_matrices=[],
            )
            state.pop("error", None)
        except Exception as exc:
            state.update(
                status="FAIL_WEIGHTED",
                weighted_status="FAIL",
                weighted_completed_utc=utc_now(),
                weighted_elapsed_seconds=round(time.monotonic() - started, 3),
                error=str(exc),
            )
        atomic_json(state_path, state)
        return state


def archive_weights(
    root: Path, workers: int, nthreads: int, limit: int | None
) -> None:
    ensure_layout(root)
    triage = read_csv(TRIAGE_CSV)
    subjects = [
        row["fsid"] for row in triage if row["lane"] == "ARCHIVE_RECOVERY"
    ]
    if limit is not None:
        subjects = subjects[:limit]
    print(
        f"[{utc_now()}] ARCHIVE_WEIGHTS_START n={len(subjects)} "
        f"workers={workers} nthreads={nthreads}",
        flush=True,
    )
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(archive_weights_subject, sid, root, nthreads): sid
            for sid in subjects
        }
        for index, future in enumerate(as_completed(futures), 1):
            result = future.result()
            results.append(result)
            print(
                f"[{utc_now()}] archive-weights {index}/{len(subjects)} "
                f"{result['fsid']} {result['status']} "
                f"t={result.get('weighted_elapsed_seconds', 'NA')}s",
                flush=True,
            )
            write_archive_summary(root)
    write_archive_summary(root)
    failures = [row for row in results if row["status"] != "PASS_ALL_NINE"]
    if failures:
        raise RuntimeError(f"{len(failures)} archive weighted rebuilds failed")


def write_archive_summary(root: Path) -> dict[str, Any]:
    states: list[dict[str, Any]] = []
    for path in sorted((root / "qc/subjects").glob("*.json")):
        try:
            state = json.loads(path.read_text())
        except Exception:
            continue
        if state.get("lane") == "ARCHIVE_RECOVERY":
            states.append(state)
    counts: dict[str, int] = {}
    for state in states:
        status = str(state.get("status", "UNKNOWN"))
        counts[status] = counts.get(status, 0) + 1
    passed = [
        state for state in states
        if state.get("status") in {"PASS_COUNT_GEOMETRY", "PASS_ALL_NINE"}
    ]
    payload = {
        "schema_version": "1.0.0",
        "updated_utc": utc_now(),
        "target_n": 86,
        "states_present": len(states),
        "status_counts": counts,
        "density_summary": {
            "min": min((row["density"] for row in passed), default=None),
            "median": float(np.median([row["density"] for row in passed]))
            if passed else None,
            "max": max((row["density"] for row in passed), default=None),
            "at_least_0_60": sum(row["density"] >= 0.60 for row in passed),
            "at_least_0_70": sum(row["density"] >= 0.70 for row in passed),
        },
        "weighted_products_status": {
            "pass_all_nine": sum(
                state.get("status") == "PASS_ALL_NINE" for state in states
            ),
            "count_geometry_only": sum(
                state.get("status") == "PASS_COUNT_GEOMETRY" for state in states
            ),
            "failed_weighted": sum(
                state.get("status") == "FAIL_WEIGHTED" for state in states
            ),
        },
        "subjects": states,
    }
    atomic_json(root / "manifests/archive_restore_summary.json", payload)
    atomic_csv(root / "manifests/archive_restore_subjects.csv", states)
    return payload


def status(root: Path) -> None:
    freeze_summary = root / "manifests/historical_freeze_summary.json"
    archive_summary = write_archive_summary(root)
    payload = {
        "root": str(root),
        "freeze": json.loads(freeze_summary.read_text())
        if freeze_summary.exists() else {"status": "NOT_STARTED"},
        "archive": archive_summary,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=V2_ROOT)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("freeze")
    archive = subparsers.add_parser("archive-restore")
    archive.add_argument("--workers", type=int, default=2)
    archive.add_argument("--nthreads", type=int, default=2)
    archive.add_argument("--limit", type=int)
    weights = subparsers.add_parser("archive-weights")
    weights.add_argument("--workers", type=int, default=1)
    weights.add_argument("--nthreads", type=int, default=8)
    weights.add_argument("--limit", type=int)
    subparsers.add_parser("status")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "freeze":
        freeze(args.root)
    elif args.command == "archive-restore":
        archive_restore(args.root, args.workers, args.nthreads, args.limit)
    elif args.command == "archive-weights":
        archive_weights(args.root, args.workers, args.nthreads, args.limit)
    elif args.command == "status":
        status(args.root)


if __name__ == "__main__":
    main()
