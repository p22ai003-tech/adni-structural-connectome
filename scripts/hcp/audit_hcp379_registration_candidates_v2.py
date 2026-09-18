#!/usr/bin/env python3
"""Benchmark a deterministic T1-to-DWI rigid-registration candidate.

This is a diagnosis-blind, non-overwriting diagnostic for the 15-unit
Recovery3 canary.  It leaves every Recovery3 artifact unchanged, estimates a
seeded ANTs rigid transform from the bound T1 image to mean b=0 space, maps the
existing 5TT brain mask and AAL3 atlas with that transform, and evaluates the
same prospective spatial gates used by Recovery3.
"""

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
from typing import Any

import nibabel as nib
import numpy as np


EXP = Path("/home/ec2-user/exp")
RUN_ROOT = Path(
    "/data/derivatives/scforge_v2/h04a_r1_recovery_20260719_retry4"
)
OUTPUT_ROOT = Path(
    "/data/derivatives/hcp379_v2/diagnostics/"
    "registration_recovery4"
)
BINDING = (
    EXP
    / "research_audit/outputs/"
    "h04a_r1_retry4_pretract_recovery3_package_v1/"
    "recovery3_execution_binding.json"
)
ATLAS = EXP / "atlas/AAL/AAL3v1_1mm_166.nii.gz"
ANTS = EXP / ".envs/ants-2.6.5/bin"
MRTRIX = Path("/home/ec2-user/mrtrix3/bin")
FSL = Path("/home/ec2-user/fsl/bin")
EXPECTED_LABELS = tuple(range(1, 167))
MINIMUM_5TT_DWI_DICE = 0.70
MINIMUM_ATLAS_INSIDE_DWI = 0.80
RANDOM_SEED = 1234


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


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
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


def load_units() -> list[str]:
    record = json.loads(BINDING.read_text(encoding="utf-8"))
    units = record.get("units")
    if (
        record.get("record_type")
        != "retry4_pretract_recovery3_execution_binding"
        or record.get("diagnosis_labels_used") is not False
        or not isinstance(units, list)
        or len(units) != 15
        or len(set(units)) != 15
    ):
        raise ValueError("Recovery3 execution binding differs")
    return sorted(str(unit) for unit in units)


def run(
    command: list[str],
    *,
    log: Any,
    env: dict[str, str],
    outputs: tuple[Path, ...] = (),
) -> None:
    log.write(f"\n[{utc_now()}] COMMAND {json.dumps(command)}\n")
    log.flush()
    completed = subprocess.run(
        command,
        stdout=log,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(
            f"command failed rc={completed.returncode}: {command[0]}"
        )
    missing = [str(path) for path in outputs if not path.is_file()]
    if missing:
        raise RuntimeError("command outputs missing: " + ",".join(missing))


def centroid_mm(image: nib.spatialimages.SpatialImage, mask: np.ndarray) -> np.ndarray:
    indices = np.argwhere(mask)
    if not indices.size:
        raise ValueError("empty mask has no centroid")
    return nib.affines.apply_affine(image.affine, indices.mean(axis=0))


def recovery3_spatial_record(unit: str) -> dict[str, Any]:
    subject = RUN_ROOT / "subjects" / unit
    report = (
        subject
        / "06_preflight/spatial_quantitative_qc_recovery3.json"
    )
    if report.is_file():
        return json.loads(report.read_text(encoding="utf-8"))
    log = subject / "logs/06_spatial_quantitative_qc_recovery3.log"
    text = log.read_text(encoding="utf-8")
    start = text.find("{")
    if start < 0:
        raise ValueError(f"no JSON spatial record in {log}")
    return json.loads(text[start:])


def paths_for(unit: str, output_root: Path) -> dict[str, Path]:
    subject = RUN_ROOT / "subjects" / unit
    output = output_root / unit
    return {
        "subject": subject,
        "output": output,
        "log": output / "registration_candidate.log",
        "result": output / "result.json",
        "b0": subject / "03_spatial/mean_b0.nii.gz",
        "b0_1mm": subject / "04_atlas/b0_1mm_world_grid.nii.gz",
        "dwi_mask": subject / "06_preflight/dwi_mask_recovery3.nii.gz",
        "dwi_mask_1mm": (
            subject / "06_preflight/dwi_mask_1mm_recovery3.nii.gz"
        ),
        "t1_brain": (
            subject / "03_spatial/t1_5tt_brain_recovery3.nii.gz"
        ),
        "t1_mask": (
            subject
            / "03_spatial/mni_registration_5tt_mask_recovery3.nii.gz"
        ),
        "mni_affine": (
            subject
            / "03_spatial/mni5tt_to_t1_recovery3_0GenericAffine.mat"
        ),
        "mni_warp": (
            subject
            / "03_spatial/mni5tt_to_t1_recovery3_1Warp.nii.gz"
        ),
        "prefix": output / "ants_mi_t1_to_b0_rigid_",
        "affine": output / "ants_mi_t1_to_b0_rigid_0GenericAffine.mat",
        "warped_t1": output / "ants_mi_t1_to_b0_rigid_Warped.nii.gz",
        "inverse_warped_b0": (
            output / "ants_mi_t1_to_b0_rigid_InverseWarped.nii.gz"
        ),
        "five_tt_mask_b0": output / "5tt_mask_ants_mi_rigid_b0.nii.gz",
        "atlas_1mm": (
            output / "aal3_nodes_166_dwi_1mm_ants_mi_rigid.nii.gz"
        ),
        "atlas_native": (
            output / "aal3_nodes_166_native_b0_ants_mi_rigid.nii.gz"
        ),
        "review_t1": output / "review_b0_vs_t1_ants_mi_rigid.png",
        "review_atlas": (
            output / "review_b0_vs_aal3_ants_mi_rigid.png"
        ),
    }


def evaluate_unit(
    unit: str,
    *,
    output_root: Path,
    threads: int,
) -> dict[str, Any]:
    paths = paths_for(unit, output_root)
    paths["output"].mkdir(parents=True, exist_ok=True)
    required = (
        paths["b0"],
        paths["b0_1mm"],
        paths["dwi_mask"],
        paths["dwi_mask_1mm"],
        paths["t1_brain"],
        paths["t1_mask"],
        paths["mni_affine"],
        paths["mni_warp"],
        ATLAS,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing candidate inputs: " + ",".join(missing))

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{ANTS}:{MRTRIX}:{FSL}:{env.get('PATH', '')}",
            "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS": str(threads),
            "FSLDIR": str(FSL.parent),
            "FSLOUTPUTTYPE": "NIFTI_GZ",
        }
    )
    with paths["log"].open("a", encoding="utf-8") as log:
        if not (
            paths["affine"].is_file()
            and paths["warped_t1"].is_file()
            and paths["inverse_warped_b0"].is_file()
        ):
            run(
                [
                    str(ANTS / "antsRegistrationSyNQuick.sh"),
                    "-d",
                    "3",
                    "-f",
                    str(paths["b0"]),
                    "-m",
                    str(paths["t1_brain"]),
                    "-x",
                    f"{paths['dwi_mask']},{paths['t1_mask']}",
                    "-t",
                    "r",
                    "-y",
                    "0",
                    "-e",
                    str(RANDOM_SEED),
                    "-n",
                    str(threads),
                    "-o",
                    str(paths["prefix"]),
                ],
                log=log,
                env=env,
                outputs=(
                    paths["affine"],
                    paths["warped_t1"],
                    paths["inverse_warped_b0"],
                ),
            )
        run(
            [
                str(ANTS / "antsApplyTransforms"),
                "-d",
                "3",
                "-i",
                str(paths["t1_mask"]),
                "-r",
                str(paths["b0"]),
                "-o",
                str(paths["five_tt_mask_b0"]),
                "-n",
                "NearestNeighbor",
                "-t",
                str(paths["affine"]),
            ],
            log=log,
            env=env,
            outputs=(paths["five_tt_mask_b0"],),
        )
        run(
            [
                str(ANTS / "antsApplyTransforms"),
                "-d",
                "3",
                "-i",
                str(ATLAS),
                "-r",
                str(paths["b0_1mm"]),
                "-o",
                str(paths["atlas_1mm"]),
                "-n",
                "GenericLabel",
                "-t",
                str(paths["affine"]),
                "-t",
                str(paths["mni_warp"]),
                "-t",
                str(paths["mni_affine"]),
            ],
            log=log,
            env=env,
            outputs=(paths["atlas_1mm"],),
        )
        run(
            [
                str(MRTRIX / "mrtransform"),
                str(paths["atlas_1mm"]),
                str(paths["atlas_native"]),
                "-template",
                str(paths["b0"]),
                "-interp",
                "nearest",
                "-force",
            ],
            log=log,
            env=env,
            outputs=(paths["atlas_native"],),
        )
        run(
            [
                str(FSL / "slices"),
                str(paths["b0"]),
                str(paths["warped_t1"]),
                "-o",
                str(paths["review_t1"]),
            ],
            log=log,
            env=env,
            outputs=(paths["review_t1"],),
        )
        run(
            [
                str(FSL / "slices"),
                str(paths["b0"]),
                str(paths["atlas_native"]),
                "-o",
                str(paths["review_atlas"]),
            ],
            log=log,
            env=env,
            outputs=(paths["review_atlas"],),
        )

    dwi_image = nib.load(str(paths["dwi_mask"]))
    dwi_mask = np.asanyarray(dwi_image.dataobj) > 0
    five_image = nib.load(str(paths["five_tt_mask_b0"]))
    five_mask = np.asanyarray(five_image.dataobj) > 0
    if dwi_mask.shape != five_mask.shape:
        raise ValueError(
            f"native mask shape mismatch: {dwi_mask.shape}:{five_mask.shape}"
        )
    dice_denominator = int(dwi_mask.sum() + five_mask.sum())
    dice = (
        2.0 * float(np.logical_and(dwi_mask, five_mask).sum())
        / dice_denominator
        if dice_denominator
        else 0.0
    )

    atlas_image = nib.load(str(paths["atlas_1mm"]))
    atlas_raw = np.asanyarray(atlas_image.dataobj)
    atlas = np.rint(atlas_raw).astype(np.int32)
    dwi_1mm_image = nib.load(str(paths["dwi_mask_1mm"]))
    dwi_1mm = np.asanyarray(dwi_1mm_image.dataobj) > 0
    if atlas.shape != dwi_1mm.shape:
        raise ValueError(
            f"1-mm atlas/mask shape mismatch: {atlas.shape}:{dwi_1mm.shape}"
        )
    atlas_mask = atlas > 0
    inside = (
        float(np.logical_and(atlas_mask, dwi_1mm).sum())
        / float(atlas_mask.sum())
        if atlas_mask.any()
        else 0.0
    )
    labels = tuple(int(value) for value in np.unique(atlas) if value > 0)
    counts = {
        str(label): int(np.count_nonzero(atlas == label))
        for label in EXPECTED_LABELS
    }
    failures: list[str] = []
    if not np.allclose(atlas_raw, atlas, atol=1.0e-6):
        failures.append("atlas_noninteger")
    if labels != EXPECTED_LABELS:
        failures.append(f"atlas_labels={len(labels)}")
    if counts and min(counts.values()) < 1:
        failures.append("atlas_node_without_voxel_support")
    if dice < MINIMUM_5TT_DWI_DICE:
        failures.append(
            f"five_tt_dwi_dice={dice:.6f}<{MINIMUM_5TT_DWI_DICE:.6f}"
        )
    if inside < MINIMUM_ATLAS_INSIDE_DWI:
        failures.append(
            f"atlas_inside_dwi={inside:.6f}<"
            f"{MINIMUM_ATLAS_INSIDE_DWI:.6f}"
        )

    current = recovery3_spatial_record(unit)
    result = {
        "schema_version": "1.0.0",
        "record_type": (
            "diagnosis_blind_hcp379_registration_candidate_audit"
        ),
        "status": "PASS" if not failures else "FAIL",
        "unit": unit,
        "diagnosis_labels_used": False,
        "completed_utc": utc_now(),
        "candidate": {
            "method": "ANTs seeded rigid T1-to-mean-b0",
            "transform": "rigid",
            "metric": "mutual_information",
            "random_seed": RANDOM_SEED,
            "fixed_mask_used": True,
            "moving_mask_used": True,
            "threads": threads,
        },
        "prospective_thresholds": {
            "minimum_5tt_dwi_mask_dice": MINIMUM_5TT_DWI_DICE,
            "minimum_atlas_inside_dwi_mask_fraction": (
                MINIMUM_ATLAS_INSIDE_DWI
            ),
        },
        "recovery3_bbr": {
            "status": current.get("status"),
            "five_tt_dwi_mask_dice": current.get(
                "five_tt_dwi_mask_dice"
            ),
            "atlas_inside_dwi_mask_fraction": current.get(
                "atlas_inside_brain_fraction"
            ),
        },
        "ants_rigid": {
            "five_tt_dwi_mask_dice": dice,
            "atlas_inside_dwi_mask_fraction": inside,
            "dwi_to_five_tt_centroid_distance_mm": float(
                np.linalg.norm(
                    centroid_mm(dwi_image, dwi_mask)
                    - centroid_mm(five_image, five_mask)
                )
            ),
            "dwi_to_atlas_centroid_distance_mm": float(
                np.linalg.norm(
                    centroid_mm(dwi_1mm_image, dwi_1mm)
                    - centroid_mm(atlas_image, atlas_mask)
                )
            ),
            "labels_found": len(labels),
            "minimum_voxels_per_label": min(counts.values()),
            "maximum_voxels_per_label": max(counts.values()),
        },
        "failures": failures,
        "artifacts": {
            name: file_record(paths[name])
            for name in (
                "affine",
                "warped_t1",
                "five_tt_mask_b0",
                "atlas_1mm",
                "atlas_native",
                "review_t1",
                "review_atlas",
            )
        },
    }
    atomic_json(paths["result"], result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        raise ValueError("--workers must be in 1..4")
    if not 1 <= args.threads <= 16:
        raise ValueError("--threads must be in 1..16")
    if args.workers * args.threads > (os.cpu_count() or 1):
        raise ValueError("requested registration concurrency exceeds host CPUs")

    units = load_units()
    args.output_root.mkdir(parents=True, exist_ok=True)
    print(
        f"[{utc_now()}] REGISTRATION_CANDIDATE_AUDIT_START "
        f"n={len(units)} workers={args.workers} threads={args.threads}",
        flush=True,
    )
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                evaluate_unit,
                unit,
                output_root=args.output_root,
                threads=args.threads,
            ): unit
            for unit in units
        }
        for index, future in enumerate(as_completed(futures), start=1):
            unit = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "schema_version": "1.0.0",
                    "record_type": (
                        "diagnosis_blind_hcp379_registration_candidate_audit"
                    ),
                    "status": "FAIL",
                    "unit": unit,
                    "diagnosis_labels_used": False,
                    "completed_utc": utc_now(),
                    "failures": [f"{type(exc).__name__}:{exc}"],
                }
                atomic_json(
                    args.output_root / unit / "result.json",
                    result,
                )
            results.append(result)
            candidate = result.get("ants_rigid", {})
            print(
                f"[{utc_now()}] registration {index}/{len(units)} "
                f"{unit} {result['status']} "
                f"dice={candidate.get('five_tt_dwi_mask_dice')} "
                f"inside={candidate.get('atlas_inside_dwi_mask_fraction')} "
                f"failures={result.get('failures')}",
                flush=True,
            )

    ordered = sorted(results, key=lambda row: str(row["unit"]))
    passed = sum(row.get("status") == "PASS" for row in ordered)
    summary = {
        "schema_version": "1.0.0",
        "record_type": (
            "diagnosis_blind_hcp379_registration_candidate_audit_summary"
        ),
        "status": "PASS" if passed == len(units) else "FAIL",
        "updated_utc": utc_now(),
        "diagnosis_labels_used": False,
        "target_n": len(units),
        "passed_n": passed,
        "failed_n": len(units) - passed,
        "method": "ANTs seeded rigid T1-to-mean-b0",
        "prospective_thresholds": {
            "minimum_5tt_dwi_mask_dice": MINIMUM_5TT_DWI_DICE,
            "minimum_atlas_inside_dwi_mask_fraction": (
                MINIMUM_ATLAS_INSIDE_DWI
            ),
        },
        "units": [
            {
                "unit": row["unit"],
                "status": row.get("status"),
                "recovery3_bbr": row.get("recovery3_bbr"),
                "ants_rigid": row.get("ants_rigid"),
                "failures": row.get("failures"),
                "result": file_record(
                    args.output_root / str(row["unit"]) / "result.json"
                ),
            }
            for row in ordered
        ],
    }
    summary_path = args.output_root / "summary.json"
    atomic_json(summary_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
