#!/usr/bin/env python3
"""Build or dry-run a high-resolution AAL3-to-DWI atlas contract.

The purpose is to stop crushing small AAL3 labels into the low-resolution b0
grid as the default contract. The output parcellation is kept as a T1-resolution
label image transformed into DWI/world coordinates for tck2connectome.

If ANTs is installed, the script uses nonlinear T1->MNI registration and applies
the inverse transform to AAL3 with GenericLabel interpolation. If ANTs is not
available, it writes and can execute a clearly marked FSL affine fallback.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path("/home/ec2-user/exp")
DEFAULT_AAL3 = ROOT / "data" / "atlas" / "AAL3" / "AAL3v1_1mm.nii.gz"
DEFAULT_MNI = Path("/home/ec2-user/fsl/data/standard/MNI152_T1_1mm_brain.nii.gz")
FALLBACK_MNI = Path("/home/ec2-user/fsl/share/fsl/data/standard/MNI152_T1_1mm_brain.nii.gz")
FLIRT = Path("/home/ec2-user/fsl/share/fsl/bin/flirt")
MRCONVERT = Path("/home/ec2-user/mrtrix3/bin/mrconvert")
MRTRANSFORM = Path("/home/ec2-user/mrtrix3/bin/mrtransform")
TRANSFORMCONVERT = Path("/home/ec2-user/mrtrix3/bin/transformconvert")


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def which(name: str) -> str:
    return shutil.which(name) or ""


def run(cmd: list[str | Path], log_path: Path, env: dict[str, str]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    str_cmd = [str(item) for item in cmd]
    with log_path.open("a", encoding="utf-8", errors="replace") as handle:
        handle.write(f"[{utc()}] $ {' '.join(str_cmd)}\n")
        handle.flush()
        proc = subprocess.run(str_cmd, stdout=handle, stderr=subprocess.STDOUT, text=True, env=env)
        handle.write(f"[{utc()}] rc={proc.returncode}\n\n")
    if proc.returncode != 0:
        raise RuntimeError(f"command failed rc={proc.returncode}: {' '.join(str_cmd)}")


def command_text(commands: list[list[str | Path]]) -> str:
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    for cmd in commands:
        lines.append(" ".join(str(part) for part in cmd))
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sid", required=True)
    parser.add_argument("--t1", type=Path, required=True, help="Native T1 image, preferably 1 mm.")
    parser.add_argument("--t1-brain", type=Path, default=None, help="Skull-stripped native T1 for registration.")
    parser.add_argument("--mean-b0", type=Path, required=True, help="Mean b0 image in DWI space.")
    parser.add_argument("--aal3-mni", type=Path, default=DEFAULT_AAL3)
    parser.add_argument("--mni-template", type=Path, default=DEFAULT_MNI if DEFAULT_MNI.exists() else FALLBACK_MNI)
    parser.add_argument("--out-dir", type=Path, default=Path("/home/ec2-user/exp/reports/aal3_highres_contract"))
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--force-fsl-affine", action="store_true")
    args = parser.parse_args()

    out_dir = args.out_dir / args.sid
    logs = out_dir / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PATH"] = "/home/ec2-user/mrtrix3/bin:/home/ec2-user/fsl/share/fsl/bin:" + env.get("PATH", "")

    ants_syn = which("antsRegistrationSyN.sh")
    ants_apply = which("antsApplyTransforms")
    use_ants = bool(ants_syn and ants_apply and not args.force_fsl_affine)
    t1_ref = args.t1_brain or args.t1

    aal_t1 = out_dir / "AAL3_T1_1mm.nii.gz"
    aal_t1_mif = out_dir / "AAL3_T1_1mm.mif"
    aal_dwi_mif = out_dir / "AAL3_dwi_1mm.mif"
    t1_to_b0_fsl = out_dir / "T1_to_b0_fsl.mat"
    t1_to_b0_mrtrix = out_dir / "T1_to_b0_mrtrix.txt"
    commands: list[list[str | Path]] = []

    if use_ants:
        prefix = out_dir / "T1_to_MNI_"
        commands.append([ants_syn, "-d", "3", "-f", args.mni_template, "-m", t1_ref, "-o", prefix])
        commands.append(
            [
                ants_apply,
                "-d",
                "3",
                "-i",
                args.aal3_mni,
                "-r",
                args.t1,
                "-o",
                aal_t1,
                "-n",
                "GenericLabel",
                "-t",
                f"[{prefix}0GenericAffine.mat,1]",
                "-t",
                f"{prefix}1InverseWarp.nii.gz",
            ]
        )
        registration_policy = "ants_nonlinear_t1_to_mni_inverse_generic_label"
    else:
        mni_to_t1 = out_dir / "mni_to_t1_fsl_affine.mat"
        commands.append([FLIRT, "-in", args.mni_template, "-ref", t1_ref, "-omat", mni_to_t1, "-dof", "12"])
        commands.append(
            [
                FLIRT,
                "-in",
                args.aal3_mni,
                "-ref",
                args.t1,
                "-applyxfm",
                "-init",
                mni_to_t1,
                "-interp",
                "nearestneighbour",
                "-datatype",
                "int",
                "-out",
                aal_t1,
            ]
        )
        registration_policy = "fsl_affine_fallback_mni_to_native_t1"

    commands.extend(
        [
            [FLIRT, "-in", t1_ref, "-ref", args.mean_b0, "-dof", "6", "-omat", t1_to_b0_fsl, "-out", out_dir / "T1_on_mean_b0_check.nii.gz"],
            [TRANSFORMCONVERT, t1_to_b0_fsl, t1_ref, args.mean_b0, "flirt_import", t1_to_b0_mrtrix],
            [MRCONVERT, aal_t1, aal_t1_mif, "-datatype", "uint32", "-force"],
            [MRTRANSFORM, aal_t1_mif, "-linear", t1_to_b0_mrtrix, "-interp", "nearest", "-datatype", "uint32", aal_dwi_mif, "-force"],
        ]
    )

    script_path = out_dir / "run_highres_aal3_contract.sh"
    script_path.write_text(command_text(commands))
    script_path.chmod(0o755)

    provenance: dict[str, Any] = {
        "generated_utc": utc(),
        "sid": args.sid,
        "registration_policy": registration_policy,
        "antsRegistrationSyN": ants_syn,
        "antsApplyTransforms": ants_apply,
        "execute": args.execute,
        "t1": str(args.t1),
        "t1_ref": str(t1_ref),
        "mean_b0": str(args.mean_b0),
        "aal3_mni": str(args.aal3_mni),
        "mni_template": str(args.mni_template),
        "aal3_t1_1mm": str(aal_t1),
        "aal3_dwi_1mm_mif": str(aal_dwi_mif),
        "script": str(script_path),
    }
    (out_dir / "highres_aal3_contract_provenance.json").write_text(json.dumps(provenance, indent=2))

    if args.execute:
        for idx, cmd in enumerate(commands, start=1):
            run(cmd, logs / f"step_{idx:02d}.log", env)

    print(json.dumps(provenance, indent=2))
    if not use_ants:
        print("WARNING: ANTs not available or disabled; using FSL affine fallback. Treat as diagnostic unless overlay passes.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
