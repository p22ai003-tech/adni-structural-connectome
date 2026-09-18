#!/usr/bin/env python3
"""Non-overwriting standard-FreeSurfer recovery for the missing HCP subject."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


FREESURFER = Path("/home/ec2-user/freesurfer")
ROOT = Path("/data/derivatives/hcp379_v2")
DEFAULT_FSID = "114_S_6347_I1344943"
DEFAULT_T1 = Path(
    "/data/derivatives/t1_anat/corrected_T1_114_S_6347_I1011824.nii.gz"
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


def process(
    fsid: str,
    t1: Path,
    root: Path,
    threads: int,
) -> dict[str, Any]:
    lane_root = root / "fastsurfer_repair"
    subjects_dir = lane_root / "standard_freesurfer"
    recovery_sid = f"{fsid}_FS"
    subject_dir = subjects_dir / recovery_sid
    result_path = lane_root / fsid / "standard_freesurfer_result.json"
    log_path = lane_root / fsid / "standard_freesurfer.log"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    subjects_dir.mkdir(parents=True, exist_ok=True)
    if result_path.exists():
        prior = json.loads(result_path.read_text())
        if prior.get("status") == "PASS_SURFACE_RECOVERY":
            return prior
    if not t1.exists():
        raise FileNotFoundError(t1)

    state: dict[str, Any] = {
        "fsid": fsid,
        "recovery_sid": recovery_sid,
        "status": "RUNNING",
        "started_utc": utc_now(),
        "non_overwriting": True,
        "source_fastsurfer_unchanged": (
            f"/data/derivatives/fastsurfer/{fsid}"
        ),
    }
    atomic_json(result_path, state)
    started = time.monotonic()
    try:
        env = os.environ.copy()
        env.update(
            {
                "FREESURFER_HOME": str(FREESURFER),
                "FS_LICENSE": str(FREESURFER / "license.txt"),
                "SUBJECTS_DIR": str(subjects_dir),
                "PATH": f"{FREESURFER / 'bin'}:{env.get('PATH', '')}",
            }
        )
        command = [
            str(FREESURFER / "bin/recon-all"),
            "-s",
            recovery_sid,
            "-i",
            str(t1),
            "-all",
            "-parallel",
            "-openmp",
            str(threads),
            "-threads",
            str(threads),
            "-no-isrunning",
        ]
        with log_path.open("a") as log:
            log.write(f"\n[{utc_now()}] COMMAND {json.dumps(command)}\n")
            log.flush()
            run = subprocess.run(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
            )
        if run.returncode:
            raise RuntimeError(f"recon-all failed rc={run.returncode}")

        required = {
            "lh_sphere_reg": subject_dir / "surf/lh.sphere.reg",
            "rh_sphere_reg": subject_dir / "surf/rh.sphere.reg",
            "lh_white": subject_dir / "surf/lh.white",
            "rh_white": subject_dir / "surf/rh.white",
            "aseg": subject_dir / "mri/aseg.mgz",
            "wmparc": subject_dir / "mri/wmparc.mgz",
        }
        missing = [name for name, path in required.items() if not path.exists()]
        if missing:
            raise RuntimeError("missing completed surface outputs: " + ",".join(missing))
        state.update(
            status="PASS_SURFACE_RECOVERY",
            completed_utc=utc_now(),
            elapsed_seconds=round(time.monotonic() - started, 3),
            t1_path=str(t1),
            t1_sha256=sha256(t1),
            subjects_dir=str(subjects_dir),
            subject_dir=str(subject_dir),
            required_output_sha256={
                name: sha256(path) for name, path in required.items()
            },
            next_gate=(
                "HCP-MMP1 surface-label transfer and volumetric atlas QC in "
                "the HCP379-v2 tree"
            ),
        )
    except Exception as exc:
        state.update(
            status="FAIL_SURFACE_RECOVERY",
            completed_utc=utc_now(),
            elapsed_seconds=round(time.monotonic() - started, 3),
            error=str(exc),
        )
    atomic_json(result_path, state)
    return state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fsid", default=DEFAULT_FSID)
    parser.add_argument("--t1", type=Path, default=DEFAULT_T1)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--threads", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = process(args.fsid, args.t1, args.root, args.threads)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    if result["status"] != "PASS_SURFACE_RECOVERY":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
