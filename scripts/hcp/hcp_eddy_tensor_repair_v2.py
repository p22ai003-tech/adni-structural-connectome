#!/usr/bin/env python3
"""Targeted non-overwriting Eddy regeneration for tensor-QC failures.

The historical derivative is never modified.  The source DWI is denoised,
de-Gibbsed, passed through a forced-CPU FSL Eddy run, and tensor-audited before
the regenerated Eddy image is made eligible for HCP379-v2 tensor repair.
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
from typing import Any


MRTRIX = Path("/home/ec2-user/mrtrix3/bin")
FSL = Path("/home/ec2-user/fsl")
ROOT = Path("/data/derivatives/hcp379_v2")
DEFAULT_SUBJECT = "016_S_7002_I1493842"


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


def load_tensor_helpers() -> Any:
    path = Path(__file__).with_name("hcp_tensor_repair_v2.py")
    spec = importlib.util.spec_from_file_location("hcp_tensor_repair_v2", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import tensor helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tensor_audit(
    dwi: Path,
    output: Path,
    nthreads: int,
    log: Any,
) -> dict[str, Any]:
    helpers = load_tensor_helpers()
    output.mkdir(parents=True, exist_ok=True)
    mask = output / "mask.mif"
    tensor = output / "dt.mif"
    metrics = {
        metric: output / f"{metric}.mif"
        for metric in ("fa", "md", "rd", "ad")
    }
    run(
        [
            str(MRTRIX / "dwi2mask"),
            str(dwi),
            str(mask),
            "-nthreads",
            str(nthreads),
            "-force",
        ],
        log,
    )
    run(
        [
            str(MRTRIX / "dwi2tensor"),
            str(dwi),
            str(tensor),
            "-mask",
            str(mask),
            "-nthreads",
            str(nthreads),
            "-force",
        ],
        log,
    )
    run(
        [
            str(MRTRIX / "tensor2metric"),
            str(tensor),
            "-fa",
            str(metrics["fa"]),
            "-adc",
            str(metrics["md"]),
            "-rd",
            str(metrics["rd"]),
            "-ad",
            str(metrics["ad"]),
            "-nthreads",
            str(nthreads),
            "-force",
        ],
        log,
    )
    mask_voxels = helpers.mrstats_nonzero(mask)
    counts = {
        "fa_below_zero": helpers.condition_count(
            metrics["fa"], "-lt", 0.0, mask
        ),
        "fa_above_one": helpers.condition_count(
            metrics["fa"], "-gt", 1.0, mask
        ),
        "md_below_zero": helpers.condition_count(
            metrics["md"], "-lt", 0.0, mask
        ),
        "rd_below_zero": helpers.condition_count(
            metrics["rd"], "-lt", 0.0, mask
        ),
        "ad_below_zero": helpers.condition_count(
            metrics["ad"], "-lt", 0.0, mask
        ),
    }
    fractions = {
        name: count / max(mask_voxels, 1)
        for name, count in counts.items()
    }
    ranges = {
        metric: helpers.mrstats_range(path, mask)
        for metric, path in metrics.items()
    }
    failures = [
        f"{name}={fraction:.6f}>0.01"
        for name, fraction in fractions.items()
        if fraction > 0.01
    ]
    return {
        "mask_voxels": mask_voxels,
        "raw_anomaly_counts": counts,
        "raw_anomaly_fractions": fractions,
        "raw_map_ranges": ranges,
        "gate": (
            "PASS" if not failures else "FAIL"
        ),
        "failures": failures,
        "threshold": (
            "each raw FA outside [0,1] or negative MD/RD/AxD condition "
            "must be <=1% of the tensor mask"
        ),
    }


def process_subject(
    sid: str,
    root: Path,
    nthreads: int,
    pe_dir: str,
) -> dict[str, Any]:
    subject_root = root / "tensor_repairs" / sid / "eddy_repair"
    subject_root.mkdir(parents=True, exist_ok=True)
    result_path = subject_root / "eddy_repair_result.json"
    prior = None
    if result_path.exists():
        prior = json.loads(result_path.read_text())
        if prior.get("status") == "PASS_EDDY_TENSOR_QC":
            return prior

    source = Path("/data/derivatives/mif_dwi") / f"{sid}.mif"
    if not source.exists():
        raise FileNotFoundError(f"source DWI missing: {source}")
    denoised = subject_root / "dwi_denoised.mif"
    noise = subject_root / "noise.mif"
    unringed = subject_root / "dwi_denoised_unringed.mif"
    eddy = subject_root / "eddy_preproc.mif"
    scratch = subject_root / "scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    log_path = subject_root / "eddy_repair.log"
    started = time.monotonic()
    state: dict[str, Any] = {
        "fsid": sid,
        "status": "RUNNING",
        "started_utc": utc_now(),
        "non_overwriting": True,
        "historical_eddy_unchanged": (
            f"/data/derivatives/eddy/{sid}_preproc.mif"
        ),
    }
    atomic_json(result_path, state)
    try:
        env = os.environ.copy()
        env.update(
            {
                "DWIFSLPREPROC_FORCE_CPU": "1",
                "FSLDIR": str(FSL),
                "FSLOUTPUTTYPE": "NIFTI_GZ",
                "PATH": (
                    f"{MRTRIX}:{FSL / 'bin'}:{FSL / 'share/fsl/bin'}:"
                    f"{env.get('PATH', '')}"
                ),
            }
        )
        with log_path.open("a") as log:
            run(
                [
                    str(MRTRIX / "dwidenoise"),
                    str(source),
                    str(denoised),
                    "-noise",
                    str(noise),
                    "-nthreads",
                    str(nthreads),
                    "-force",
                ],
                log,
                env,
            )
            run(
                [
                    str(MRTRIX / "mrdegibbs"),
                    str(denoised),
                    str(unringed),
                    "-nthreads",
                    str(nthreads),
                    "-force",
                ],
                log,
                env,
            )
            run(
                [
                    str(MRTRIX / "dwifslpreproc"),
                    str(unringed),
                    str(eddy),
                    "-rpe_none",
                    "-pe_dir",
                    pe_dir,
                    "-eddy_options",
                    (
                        f"--slm=linear --data_is_shelled --repol "
                        f"--cnr_maps --residuals --nthr={nthreads}"
                    ),
                    "-scratch",
                    str(scratch),
                    "-nthreads",
                    str(nthreads),
                    "-force",
                ],
                log,
                env,
            )
            audit = tensor_audit(
                eddy,
                subject_root / "tensor_audit",
                nthreads,
                log,
            )
        if audit["gate"] != "PASS":
            raise RuntimeError(
                "regenerated Eddy tensor failed physical QC: "
                + ";".join(audit["failures"])
            )
        state.update(
            status="PASS_EDDY_TENSOR_QC",
            completed_utc=utc_now(),
            elapsed_seconds=round(time.monotonic() - started, 3),
            source_path=str(source),
            source_sha256=sha256(source),
            denoised_path=str(denoised),
            unringed_path=str(unringed),
            eddy_path=str(eddy),
            eddy_sha256=sha256(eddy),
            pe_dir=pe_dir,
            execution="forced CPU Eddy with explicit thread count",
            tensor_audit=audit,
        )
    except Exception as exc:
        state.update(
            status="FAIL_EDDY_REPAIR",
            completed_utc=utc_now(),
            elapsed_seconds=round(time.monotonic() - started, 3),
            error=str(exc),
        )
    atomic_json(result_path, state)
    return state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", default=DEFAULT_SUBJECT)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--nthreads", type=int, default=16)
    parser.add_argument("--pe-dir", default="j-")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = process_subject(
        args.subject,
        args.root,
        args.nthreads,
        args.pe_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    if result["status"] != "PASS_EDDY_TENSOR_QC":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
