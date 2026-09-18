#!/home/ec2-user/fsl/bin/python
"""Select and validate Recovery4 5TT/GMWMI inputs for the 15-unit canary.

Recovery3 FSL BBR products remain the primary route.  For units whose locked
registration policy selected the seeded ANTs rigid failover, the T1-space 5TT
image is resampled once into the native mean-b0 grid as a five-volume time
series, clipped to the valid partial-volume range, checked with ``5ttcheck``,
and used to derive a new GMWMI seed image.  Existing products are never
overwritten.

Every selected 5TT is quantitatively checked against the DWI mask.  The output
summary is diagnosis-blind and records immutable file hashes for downstream
tractography.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import nibabel as nib
import numpy as np


RUN_ROOT = Path(
    "/data/derivatives/scforge_v2/h04a_r1_recovery_20260719_retry4"
)
HCP_ROOT = Path("/data/derivatives/hcp379_v2")
ATLAS_ROOT = HCP_ROOT / "corrected_atlas_canary_recovery4"
ATLAS_SUMMARY = ATLAS_ROOT / "corrected_atlas_canary_recovery4_summary.json"
REGISTRATION_ROOT = HCP_ROOT / "diagnostics/registration_recovery4"
OUTPUT_ROOT = HCP_ROOT / "pretract_spatial_recovery4"
MRTRIX = Path("/home/ec2-user/mrtrix3/bin")
ANTS = Path("/home/ec2-user/exp/.envs/ants-2.6.5/bin")
MINIMUM_5TT_DWI_DICE = 0.70
GEOMETRY_AFFINE_TOLERANCE = 1.0e-5


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


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        encoding="utf-8",
        delete=False,
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def run(
    command: list[str],
    *,
    log: Any,
    env: Mapping[str, str] | None = None,
    outputs: tuple[Path, ...] = (),
) -> subprocess.CompletedProcess[str]:
    log.write(f"\n[{utc_now()}] COMMAND {json.dumps(command)}\n")
    log.flush()
    completed = subprocess.run(
        command,
        stdout=log,
        stderr=subprocess.STDOUT,
        text=True,
        env=dict(env) if env is not None else None,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(
            f"command failed rc={completed.returncode}: {command[0]}"
        )
    missing = [str(path) for path in outputs if not path.is_file()]
    if missing:
        raise RuntimeError("command outputs missing: " + ",".join(missing))
    return completed


def image_range(path: Path) -> dict[str, float]:
    completed = subprocess.run(
        [
            str(MRTRIX / "mrstats"),
            str(path),
            "-output",
            "min",
            "-output",
            "max",
            "-output",
            "mean",
            "-quiet",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    values = [float(value) for value in completed.stdout.split()]
    if len(values) < 3 or len(values) % 3:
        raise RuntimeError(f"unexpected mrstats output for {path}")
    minima = values[0::3]
    maxima = values[1::3]
    means = values[2::3]
    return {
        "min": min(minima),
        "max": max(maxima),
        "mean": float(sum(means) / len(means)),
    }


def atlas_result(unit: str) -> dict[str, Any]:
    path = ATLAS_ROOT / "subjects" / unit / "result.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("status") != "PASS":
        raise ValueError(f"HCP379 atlas result is not PASS: {unit}")
    return record


def selected_paths(
    unit: str,
    route: str,
    output_root: Path,
) -> dict[str, Path]:
    source = RUN_ROOT / "subjects" / unit
    output = output_root / "subjects" / unit
    if route == "seeded_ants_rigid_failover":
        five_tt = output / "05_model/5tt_dwi_recovery4_ants.mif"
        gmwmi = output / "05_model/gmwmi_dwi_recovery4_ants.mif"
        t1_in_b0 = (
            REGISTRATION_ROOT
            / unit
            / "ants_mi_t1_to_b0_rigid_Warped.nii.gz"
        )
    else:
        five_tt = source / "05_model/5tt_dwi_recovery3.mif"
        gmwmi = source / "05_model/gmwmi_dwi_recovery3.mif"
        t1_in_b0 = source / "06_preflight/t1_in_b0_qc.nii.gz"
    return {
        "source": source,
        "output": output,
        "result": output / "06_preflight/spatial_recovery4_qc.json",
        "log": output / "logs/spatial_recovery4.log",
        "five_tt_t1": source / "05_model/5tt_t1_fsl.mif",
        "b0": source / "03_spatial/mean_b0.nii.gz",
        "dwi_mask": source / "01_dwi/dwi_brain_mask.mif",
        "dwi_mask_nifti": source / "06_preflight/dwi_mask_recovery3.nii.gz",
        "five_tt": five_tt,
        "gmwmi": gmwmi,
        "t1_in_b0": t1_in_b0,
        "selected_mask": (
            output / "06_preflight/selected_5tt_mask_recovery4.nii.gz"
        ),
        "selected_sum": (
            output / "06_preflight/selected_5tt_sum_recovery4.mif"
        ),
    }


def existing_pass(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("status") != "PASS":
            return None
        for artifact in record.get("artifacts", {}).values():
            candidate = Path(str(artifact["path"]))
            if (
                not candidate.is_file()
                or candidate.is_symlink()
                or candidate.stat().st_size != int(artifact["size_bytes"])
                or sha256_file(candidate) != artifact["sha256"]
            ):
                return None
        return record
    except Exception:
        return None


def build_failover(
    paths: Mapping[str, Path],
    *,
    transform: Path,
    log: Any,
    threads: int,
) -> None:
    paths["five_tt"].parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(threads)
    with tempfile.TemporaryDirectory(
        dir=paths["output"],
        prefix=".five_tt_recovery4.",
    ) as temporary:
        work = Path(temporary)
        five_tt_t1_nifti = work / "5tt_t1.nii.gz"
        five_tt_b0_nifti = work / "5tt_b0.nii.gz"
        five_tt_b0_raw = work / "5tt_b0_raw.mif"
        run(
            [
                str(MRTRIX / "mrconvert"),
                str(paths["five_tt_t1"]),
                str(five_tt_t1_nifti),
                "-quiet",
            ],
            log=log,
            outputs=(five_tt_t1_nifti,),
        )
        run(
            [
                str(ANTS / "antsApplyTransforms"),
                "-d",
                "3",
                "-e",
                "3",
                "-i",
                str(five_tt_t1_nifti),
                "-r",
                str(paths["b0"]),
                "-o",
                str(five_tt_b0_nifti),
                "-n",
                "Linear",
                "-t",
                str(transform),
                "--float",
                "1",
            ],
            log=log,
            env=env,
            outputs=(five_tt_b0_nifti,),
        )
        run(
            [
                str(MRTRIX / "mrconvert"),
                str(five_tt_b0_nifti),
                str(five_tt_b0_raw),
                "-quiet",
            ],
            log=log,
            outputs=(five_tt_b0_raw,),
        )
        run(
            [
                str(MRTRIX / "mrcalc"),
                str(five_tt_b0_raw),
                "0",
                "-max",
                "1",
                "-min",
                str(paths["five_tt"]),
                "-force",
                "-quiet",
            ],
            log=log,
            outputs=(paths["five_tt"],),
        )
    run(
        [str(MRTRIX / "5ttcheck"), str(paths["five_tt"])],
        log=log,
    )
    run(
        [
            str(MRTRIX / "5tt2gmwmi"),
            str(paths["five_tt"]),
            str(paths["gmwmi"]),
            "-force",
            "-quiet",
        ],
        log=log,
        outputs=(paths["gmwmi"],),
    )


def process_unit(
    unit: str,
    *,
    output_root: Path,
    threads: int,
) -> dict[str, Any]:
    atlas = atlas_result(unit)
    route = str(atlas["registration_route"]["route"])
    paths = selected_paths(unit, route, output_root)
    cached = existing_pass(paths["result"])
    if cached is not None:
        return cached
    paths["result"].parent.mkdir(parents=True, exist_ok=True)
    paths["log"].parent.mkdir(parents=True, exist_ok=True)
    required = [
        paths["five_tt_t1"],
        paths["b0"],
        paths["dwi_mask"],
        paths["dwi_mask_nifti"],
        paths["t1_in_b0"],
    ]
    transform = Path(
        str(atlas["registration_route"]["transform"]["path"])
    )
    required.append(transform)
    if route != "seeded_ants_rigid_failover":
        required.extend([paths["five_tt"], paths["gmwmi"]])
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing spatial inputs: " + ",".join(missing))

    started = time.monotonic()
    failures: list[str] = []
    with paths["log"].open("a", encoding="utf-8") as log:
        log.write(
            f"\n[{utc_now()}] START unit={unit} route={route}\n"
        )
        if route == "seeded_ants_rigid_failover":
            build_failover(
                paths,
                transform=transform,
                log=log,
                threads=threads,
            )
        else:
            run(
                [str(MRTRIX / "5ttcheck"), str(paths["five_tt"])],
                log=log,
            )
        run(
            [
                str(MRTRIX / "mrmath"),
                str(paths["five_tt"]),
                "sum",
                str(paths["selected_sum"]),
                "-axis",
                "3",
                "-nthreads",
                str(threads),
                "-force",
                "-quiet",
            ],
            log=log,
            outputs=(paths["selected_sum"],),
        )
        # Write with the same LAS storage order as the locked DWI-mask NIfTI.
        # MRtrix MIF geometry is stride-independent, but writing the threshold
        # directly from a RAS-strided 5TT to NIfTI can otherwise yield a
        # different array orientation from the DWI reference.
        with tempfile.TemporaryDirectory(
            dir=paths["output"],
            prefix=".five_tt_mask_recovery4.",
        ) as temporary:
            mask_mif = Path(temporary) / "selected_5tt_mask.mif"
            run(
                [
                    str(MRTRIX / "mrthreshold"),
                    str(paths["selected_sum"]),
                    str(mask_mif),
                    "-abs",
                    "0.5",
                    "-force",
                    "-quiet",
                ],
                log=log,
                outputs=(mask_mif,),
            )
            run(
                [
                    str(MRTRIX / "mrconvert"),
                    str(mask_mif),
                    str(paths["selected_mask"]),
                    "-strides",
                    "-1,2,3",
                    "-force",
                    "-quiet",
                ],
                log=log,
                outputs=(paths["selected_mask"],),
            )

    five_tt_range = image_range(paths["five_tt"])
    gmwmi_range = image_range(paths["gmwmi"])
    sum_range = image_range(paths["selected_sum"])
    if five_tt_range["min"] < -1.0e-6 or five_tt_range["max"] > 1.000001:
        failures.append("five_tt_outside_0_1")
    if gmwmi_range["min"] < -1.0e-6 or gmwmi_range["max"] <= 0:
        failures.append("gmwmi_invalid_range")

    five_image = nib.load(str(paths["selected_mask"]))
    five = np.asanyarray(five_image.dataobj) > 0
    dwi_image = nib.load(str(paths["dwi_mask_nifti"]))
    dwi = np.asanyarray(dwi_image.dataobj) > 0
    geometry_ok = (
        five.shape == dwi.shape
        and np.allclose(
            five_image.affine,
            dwi_image.affine,
            atol=GEOMETRY_AFFINE_TOLERANCE,
        )
    )
    if not geometry_ok:
        dice = 0.0
        failures.append("selected_5tt_dwi_geometry_mismatch")
    else:
        denominator = int(five.sum() + dwi.sum())
        dice = (
            2.0 * float(np.logical_and(five, dwi).sum()) / denominator
            if denominator
            else 0.0
        )
    if dice < MINIMUM_5TT_DWI_DICE:
        failures.append(
            f"selected_5tt_dwi_dice={dice:.6f}<"
            f"{MINIMUM_5TT_DWI_DICE:.6f}"
        )

    record = {
        "schema_version": "1.0.0",
        "record_type": (
            "diagnosis_blind_hcp379_pretract_spatial_recovery4_qc"
        ),
        "status": "PASS" if not failures else "FAIL",
        "unit": unit,
        "diagnosis_labels_used": False,
        "non_overwriting": True,
        "completed_utc": utc_now(),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "registration_route": route,
        "selected_5tt_source": (
            "Recovery4 ANTs failover"
            if route == "seeded_ants_rigid_failover"
            else "Recovery3 FSL BBR primary"
        ),
        "selected_5tt_dwi_mask_dice": dice,
        "minimum_selected_5tt_dwi_mask_dice": (
            MINIMUM_5TT_DWI_DICE
        ),
        "geometry_match": geometry_ok,
        "hcp379_atlas_inside_dwi_mask_fraction": atlas["atlas_qc"][
            "atlas_inside_dwi_mask_fraction"
        ],
        "five_tt_range": five_tt_range,
        "five_tt_sum_range": sum_range,
        "gmwmi_range": gmwmi_range,
        "failures": failures,
        "artifacts": {
            "selected_five_tt": file_record(paths["five_tt"]),
            "selected_gmwmi": file_record(paths["gmwmi"]),
            "selected_five_tt_mask": file_record(paths["selected_mask"]),
            "selected_five_tt_sum": file_record(paths["selected_sum"]),
            "selected_t1_in_b0": file_record(paths["t1_in_b0"]),
            "selected_transform": file_record(transform),
        },
    }
    atomic_json(paths["result"], record)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.jobs < 1 or args.threads < 1:
        parser.error("--jobs and --threads must be positive")

    atlas_summary = json.loads(ATLAS_SUMMARY.read_text(encoding="utf-8"))
    if atlas_summary.get("status") != "PASS":
        raise SystemExit("Recovery4 HCP379 atlas summary is not PASS")
    units = [str(row["unit"]) for row in atlas_summary["units"]]
    results: dict[str, dict[str, Any]] = {}
    failures: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = {
            executor.submit(
                process_unit,
                unit,
                output_root=args.output_root,
                threads=args.threads,
            ): unit
            for unit in units
        }
        for future in as_completed(futures):
            unit = futures[future]
            try:
                results[unit] = future.result()
            except Exception as exc:
                failures[unit] = f"{type(exc).__name__}: {exc}"

    rows: list[dict[str, Any]] = []
    for unit in units:
        result = results.get(unit)
        rows.append(
            {
                "unit": unit,
                "status": result.get("status") if result else "FAIL",
                "registration_route": (
                    result.get("registration_route") if result else None
                ),
                "selected_5tt_dwi_mask_dice": (
                    result.get("selected_5tt_dwi_mask_dice")
                    if result
                    else None
                ),
                "failure": failures.get(unit),
                "result": (
                    file_record(
                        args.output_root
                        / "subjects"
                        / unit
                        / "06_preflight/spatial_recovery4_qc.json"
                    )
                    if result is not None
                    else None
                ),
            }
        )
    passed = sum(row["status"] == "PASS" for row in rows)
    summary = {
        "schema_version": "1.0.0",
        "record_type": (
            "diagnosis_blind_hcp379_pretract_spatial_recovery4_summary"
        ),
        "status": (
            "PASS" if passed == len(units) and not failures else "FAIL"
        ),
        "diagnosis_labels_used": False,
        "non_overwriting": True,
        "target_n": len(units),
        "passed_n": passed,
        "failed_n": len(units) - passed,
        "minimum_selected_5tt_dwi_mask_dice": (
            MINIMUM_5TT_DWI_DICE
        ),
        "registration_route_counts": {
            route: sum(
                row["registration_route"] == route for row in rows
            )
            for route in sorted(
                {
                    str(row["registration_route"])
                    for row in rows
                    if row["registration_route"] is not None
                }
            )
        },
        "units": rows,
    }
    atomic_json(
        args.output_root / "spatial_recovery4_summary.json",
        summary,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
