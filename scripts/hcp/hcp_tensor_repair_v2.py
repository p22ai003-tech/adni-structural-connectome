#!/usr/bin/env python3
"""Non-overwriting repair of physically invalid HCP tensor edge matrices.

The existing tractogram, atlas, SIFT2, count, and geometry matrices are kept.
FA/MD/RD/AxD are refit from the subject's Eddy image in the same grid, bounded
to physical ranges with raw maps retained, sampled in the unchanged streamline
order, and written into HCP379-v2 only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


MRTRIX = Path("/home/ec2-user/mrtrix3/bin")
ROOT = Path("/data/derivatives/hcp379_v2")
SUBJECT_LIST = Path(
    "/home/ec2-user/exp/research_audit/outputs/hcp_rerun_triage_v1/"
    "tensor_weighted_repair_2.txt"
)
N_NODES = 379
MATRIX_SUFFIXES = (
    "count", "fd_sum", "len_mean", "invlen_mean", "fa_mean", "md_mean",
    "rd_mean", "ad_mean", "count_invnodevol",
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
        mode="w", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def run(command: list[str], log: Any) -> None:
    log.write(f"\n[{utc_now()}] COMMAND {json.dumps(command)}\n")
    log.flush()
    result = subprocess.run(
        command, stdout=log, stderr=subprocess.STDOUT, text=True
    )
    if result.returncode:
        raise RuntimeError(f"command failed rc={result.returncode}: {command[0]}")


def mrstats_nonzero(path: Path) -> int:
    result = subprocess.run(
        [
            str(MRTRIX / "mrstats"), str(path), "-output", "count",
            "-ignorezero",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return int(float(result.stdout.strip()))


def mrstats_range(path: Path, mask: Path) -> dict[str, float]:
    result = subprocess.run(
        [
            str(MRTRIX / "mrstats"), str(path), "-mask", str(mask),
            "-output", "min", "-output", "max", "-output", "mean",
            "-quiet",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    values = [float(value) for value in result.stdout.split()]
    if len(values) != 3:
        raise RuntimeError(f"unexpected mrstats output for {path}: {result.stdout!r}")
    return {"min": values[0], "max": values[1], "mean": values[2]}


def condition_count(path: Path, operator: str, threshold: float, mask: Path) -> int:
    """Count mask voxels satisfying an MRtrix mrcalc comparison."""
    calc = subprocess.Popen(
        [
            str(MRTRIX / "mrcalc"), str(path), str(threshold), operator,
            str(mask), "-mult", "-", "-quiet",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if calc.stdout is None:
        raise RuntimeError("failed to open mrcalc stdout")
    stats = subprocess.run(
        [
            str(MRTRIX / "mrstats"), "-", "-output", "count",
            "-ignorezero", "-quiet",
        ],
        stdin=calc.stdout,
        capture_output=True,
        text=True,
    )
    calc.stdout.close()
    _, calc_stderr = calc.communicate()
    if calc.returncode or stats.returncode:
        raise RuntimeError(
            "tensor anomaly count failed: "
            f"mrcalc_rc={calc.returncode} mrstats_rc={stats.returncode} "
            f"mrcalc_stderr={calc_stderr.decode(errors='replace')!r} "
            f"mrstats_stderr={stats.stderr!r}"
        )
    return int(float(stats.stdout.strip()))


def validate_matrix(matrix: np.ndarray, suffix: str) -> list[str]:
    failures: list[str] = []
    if matrix.shape != (N_NODES, N_NODES):
        return [f"{suffix}:shape={matrix.shape}"]
    if not np.isfinite(matrix).all():
        failures.append(f"{suffix}:nonfinite")
    if not np.allclose(matrix, matrix.T, atol=1e-6, rtol=1e-6):
        failures.append(f"{suffix}:asymmetric")
    if not np.allclose(np.diag(matrix), 0.0, atol=1e-8):
        failures.append(f"{suffix}:diagonal")
    if float(matrix.min()) < -1e-10:
        failures.append(f"{suffix}:negative={float(matrix.min())}")
    return failures


def repair_subject(sid: str, root: Path, nthreads: int) -> dict[str, Any]:
    subject_root = root / "tensor_repairs" / sid
    output_dir = subject_root / "connectomes"
    tensor_dir = subject_root / "tensor"
    samples_dir = subject_root / "samples"
    for directory in (output_dir, tensor_dir, samples_dir):
        directory.mkdir(parents=True, exist_ok=True)
    result_path = subject_root / "result.json"
    if result_path.exists():
        prior = json.loads(result_path.read_text())
        if prior.get("status") == "PASS_ALL_NINE":
            return prior

    log_path = subject_root / "repair.log"
    started = time.monotonic()
    state: dict[str, Any] = {
        "fsid": sid,
        "status": "RUNNING",
        "started_utc": utc_now(),
        "non_overwriting": True,
    }
    atomic_json(result_path, state)
    try:
        track = Path("/data/derivatives/tracks") / sid / "tracks_final_3000k.tck"
        nodes = Path("/data/derivatives/parc_hcpmmp1") / sid / "nodes_b0.nii.gz"
        historical_eddy = (
            Path("/data/derivatives/eddy") / f"{sid}_preproc.mif"
        )
        repaired_eddy = subject_root / "eddy_repair" / "eddy_preproc.mif"
        repaired_eddy_result = (
            subject_root / "eddy_repair" / "eddy_repair_result.json"
        )
        eddy = historical_eddy
        eddy_source = "historical"
        if repaired_eddy.exists() and repaired_eddy_result.exists():
            repaired_state = json.loads(repaired_eddy_result.read_text())
            if repaired_state.get("status") == "PASS_EDDY_TENSOR_QC":
                eddy = repaired_eddy
                eddy_source = "hcp379_v2_targeted_eddy_repair"
        current_root = Path("/data/derivatives/connectomes_hcp_fs")
        required = [track, nodes, eddy]
        required.extend(
            current_root / f"SC_HCPMMP1_{sid}_{suffix}.csv"
            for suffix in (
                "count", "fd_sum", "len_mean", "invlen_mean",
                "count_invnodevol",
            )
        )
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError("missing tensor repair inputs: " + ",".join(missing))

        mask = tensor_dir / "mask.mif"
        tensor = tensor_dir / "dt.mif"
        raw = {
            metric: tensor_dir / f"{metric}_raw.mif"
            for metric in ("fa", "md", "rd", "ad")
        }
        bounded = {
            metric: tensor_dir / f"{metric}.mif"
            for metric in ("fa", "md", "rd", "ad")
        }
        with log_path.open("a") as log:
            run(
                [
                    str(MRTRIX / "dwi2mask"), str(eddy), str(mask),
                    "-nthreads", str(nthreads), "-force",
                ],
                log,
            )
            if mrstats_nonzero(mask) < 100_000:
                fallback = Path("/data/derivatives/fod") / sid / "mask_fod.mif"
                if not fallback.exists() or mrstats_nonzero(fallback) < 100_000:
                    raise RuntimeError("fresh and fallback tensor masks are too small")
                run(
                    [
                        str(MRTRIX / "mrgrid"), str(fallback), "regrid",
                        "-template", str(eddy), "-interp", "nearest",
                        "-datatype", "bit", str(mask), "-force",
                    ],
                    log,
                )
            run(
                [
                    str(MRTRIX / "dwi2tensor"), str(eddy), str(tensor),
                    "-mask", str(mask), "-nthreads", str(nthreads), "-force",
                ],
                log,
            )
            run(
                [
                    str(MRTRIX / "tensor2metric"), str(tensor),
                    "-fa", str(raw["fa"]), "-adc", str(raw["md"]),
                    "-rd", str(raw["rd"]), "-ad", str(raw["ad"]),
                    "-nthreads", str(nthreads), "-force",
                ],
                log,
            )

            mask_voxels = mrstats_nonzero(mask)
            raw_map_ranges = {
                metric: mrstats_range(raw[metric], mask)
                for metric in ("fa", "md", "rd", "ad")
            }
            raw_anomaly_counts = {
                "fa_below_zero": condition_count(raw["fa"], "-lt", 0.0, mask),
                "fa_above_one": condition_count(raw["fa"], "-gt", 1.0, mask),
                "md_below_zero": condition_count(raw["md"], "-lt", 0.0, mask),
                "rd_below_zero": condition_count(raw["rd"], "-lt", 0.0, mask),
                "ad_below_zero": condition_count(raw["ad"], "-lt", 0.0, mask),
            }
            raw_anomaly_fractions = {
                name: count / max(mask_voxels, 1)
                for name, count in raw_anomaly_counts.items()
            }
            gross_anomalies = {
                name: fraction
                for name, fraction in raw_anomaly_fractions.items()
                if fraction > 0.01
            }
            if gross_anomalies:
                raise RuntimeError(
                    "gross raw tensor pathology (>1% tracking-mask voxels): "
                    + json.dumps(gross_anomalies, sort_keys=True)
                )

            run(
                [
                    str(MRTRIX / "mrcalc"), str(raw["fa"]), "0", "-max",
                    "1", "-min", str(bounded["fa"]), "-force",
                ],
                log,
            )
            for metric in ("md", "rd", "ad"):
                run(
                    [
                        str(MRTRIX / "mrcalc"), str(raw[metric]), "0",
                        "-max", str(bounded[metric]), "-force",
                    ],
                    log,
                )

            samples: dict[str, Path] = {}
            for metric in ("fa", "md", "rd", "ad"):
                sample = samples_dir / f"{metric}_mean.tsf"
                samples[metric] = sample
                run(
                    [
                        str(MRTRIX / "tcksample"), str(track),
                        str(bounded[metric]), str(sample), "-stat_tck", "mean",
                        "-nthreads", str(nthreads), "-force",
                    ],
                    log,
                )

            prefix = output_dir / f"SC_HCPMMP1_{sid}"
            for suffix in (
                "count", "fd_sum", "len_mean", "invlen_mean",
                "count_invnodevol",
            ):
                shutil.copy2(
                    current_root / f"SC_HCPMMP1_{sid}_{suffix}.csv",
                    Path(f"{prefix}_{suffix}.csv"),
                )
            common = [
                "-symmetric", "-zero_diagonal", "-assignment_radial_search",
                "4", "-stat_edge", "mean", "-nthreads", str(nthreads),
                "-force", "-quiet",
            ]
            for metric in ("fa", "md", "rd", "ad"):
                run(
                    [
                        str(MRTRIX / "tck2connectome"), str(track), str(nodes),
                        f"{prefix}_{metric}_mean.csv", *common,
                        "-scale_file", str(samples[metric]),
                    ],
                    log,
                )

        count = np.loadtxt(f"{prefix}_count.csv", delimiter=",")
        count_support = np.triu(count > 0, 1)
        denominator = max(int(count_support.sum()), 1)
        failures: list[str] = []
        supports: dict[str, float] = {}
        ranges: dict[str, dict[str, float]] = {}
        hashes: dict[str, str] = {}
        for suffix in MATRIX_SUFFIXES:
            path = Path(f"{prefix}_{suffix}.csv")
            matrix = np.loadtxt(path, delimiter=",")
            failures.extend(validate_matrix(matrix, suffix))
            hashes[suffix] = sha256(path)
            if suffix in {"fa_mean", "md_mean", "rd_mean", "ad_mean"}:
                supports[suffix] = float(
                    (np.triu(matrix != 0, 1) & count_support).sum() / denominator
                )
                nonzero = matrix[matrix != 0]
                ranges[suffix] = {
                    "min": float(nonzero.min()),
                    "max": float(nonzero.max()),
                }
                if supports[suffix] < 0.95:
                    failures.append(f"{suffix}:support={supports[suffix]:.6f}")
        if ranges["fa_mean"]["max"] > 1.000001:
            failures.append(f"fa_mean:range={ranges['fa_mean']}")
        if failures:
            raise RuntimeError(";".join(sorted(set(failures))))

        state.update(
            status="PASS_ALL_NINE",
            completed_utc=utc_now(),
            elapsed_seconds=round(time.monotonic() - started, 3),
            output_prefix=str(prefix),
            track_path=str(track),
            track_sha256=sha256(track),
            eddy_path=str(eddy),
            eddy_sha256=sha256(eddy),
            eddy_source=eddy_source,
            tensor_recipe=(
                "dwi2tensor/tensor2metric on current Eddy grid and fresh mask; "
                "FA bounded to [0,1], negative diffusivities set to zero, raw "
                "maps retained"
            ),
            tensor_mask_voxels=mask_voxels,
            raw_map_ranges=raw_map_ranges,
            raw_anomaly_counts=raw_anomaly_counts,
            raw_anomaly_fractions=raw_anomaly_fractions,
            raw_anomaly_gate=(
                "fail if any raw FA outside [0,1] or negative MD/RD/AxD "
                "condition exceeds 1% of the tensor tracking mask"
            ),
            scalar_edge_support=supports,
            scalar_edge_ranges=ranges,
            matrix_sha256=hashes,
            scalar_sample_sha256={
                metric: sha256(samples_dir / f"{metric}_mean.tsf")
                for metric in ("fa", "md", "rd", "ad")
            },
        )
    except Exception as exc:
        state.update(
            status="FAIL",
            completed_utc=utc_now(),
            elapsed_seconds=round(time.monotonic() - started, 3),
            error=str(exc),
        )
    atomic_json(result_path, state)
    return state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--nthreads", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    subjects = [
        line.strip() for line in SUBJECT_LIST.read_text().splitlines()
        if line.strip()
    ]
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(repair_subject, sid, args.root, args.nthreads): sid
            for sid in subjects
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(
                f"[{utc_now()}] tensor-repair {result['fsid']} "
                f"{result['status']} t={result.get('elapsed_seconds', 'NA')}s",
                flush=True,
            )
    atomic_json(
        args.root / "manifests/tensor_repair_summary.json",
        {
            "updated_utc": utc_now(),
            "target_n": len(subjects),
            "pass": sum(row["status"] == "PASS_ALL_NINE" for row in results),
            "fail": sum(row["status"] != "PASS_ALL_NINE" for row in results),
            "subjects": sorted(results, key=lambda row: row["fsid"]),
        },
    )
    if any(row["status"] != "PASS_ALL_NINE" for row in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
