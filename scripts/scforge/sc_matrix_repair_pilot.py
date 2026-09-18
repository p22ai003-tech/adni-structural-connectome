#!/usr/bin/env python3
"""Pilot-first structural connectome repair diagnostics.

This script is intentionally conservative.  The default commands only write
diagnostic artifacts under qc/sc_matrix_qc/pilot_repair/<stamp>; production
parcellations and connectome matrices are not modified here.
"""

from __future__ import annotations

import argparse
import csv
import math
import shutil
import subprocess as sp
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    import nibabel as nib
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"nibabel is required for pilot QC: {exc}") from exc


ROOT = Path("/home/ec2-user/exp")
DERIV = ROOT / "data" / "derivatives"
QC_ROOT = DERIV / "qc" / "sc_matrix_qc"
FAILED_ROOT = QC_ROOT / "failed_repair_outputs" / "20260512T121157Z"
BACKUP_ROOT = QC_ROOT / "repair_backups" / "20260512T103726Z"
LABEL_CSV = ROOT / "atlas" / "AAL" / "AAL3_labels.csv"
PILOT_SIDS = (
    "<SUBJECT>_I<IMAGEID>",
    "<SUBJECT>_I<IMAGEID>",
    "<SUBJECT>_I<IMAGEID>",
)
WEIGHTS = (
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
MRCONVERT = Path("/home/ec2-user/mrtrix3/bin/mrconvert")
FLIRT = Path("/home/ec2-user/fsl/bin/flirt")
CONVERT_XFM = Path("/home/ec2-user/fsl/bin/convert_xfm")
FSL_STD = Path("/home/ec2-user/fsl/data/standard/MNI152_T1_1mm_brain.nii.gz")
AAL_MNI = ROOT / "atlas" / "AAL" / "AAL3v1_1mm.nii.gz"


def _read_valid_nodes() -> set[int]:
    valid: set[int] = set()
    with LABEL_CSV.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                valid.add(int(float(row["node"])))
            except Exception:
                continue
    if not valid:
        raise RuntimeError(f"No AAL nodes read from {LABEL_CSV}")
    return valid


VALID_NODES = _read_valid_nodes()


def _subject_paths(sid: str) -> dict[str, Path]:
    return {
        "b0": DERIV / "dwi_t1_bbr" / f"{sid}_b0mean_ras.nii.gz",
        "t1": DERIV / "dwi_t1_bbr" / f"{sid}_t1_ras.nii.gz",
        "t12b0": DERIV / "dwi_t1_bbr" / f"{sid}_t12b0_bbr.mat",
        "mask_b0": DERIV / "dwi_t1_bbr" / f"{sid}_mask_on_b0_bbr.nii.gz",
        "mask_5tt": DERIV / "fod" / sid / "mask_5tt_brain.mif",
        "parc": DERIV / "parc" / sid,
        "tracks": DERIV / "tracks" / sid,
        "conn": DERIV / "connectomes",
    }


def _variant_sources(sid: str) -> dict[str, Path]:
    return {
        "current_restored": DERIV / "parc" / sid / "AAL_b0.nii.gz",
        "backup_prerepair": BACKUP_ROOT / sid / "parc" / sid / "AAL_b0.nii.gz",
        "failed_direct_repair": FAILED_ROOT / "parc" / sid / "AAL_b0.nii.gz",
    }


def _matrix_sources(sid: str, weight: str) -> dict[str, Path]:
    return {
        "current_restored": DERIV / "connectomes" / f"SC_AAL_{sid}_{weight}.csv",
        "backup_prerepair": BACKUP_ROOT / sid / "connectomes" / f"SC_AAL_{sid}_{weight}.csv",
        "failed_direct_repair": FAILED_ROOT / "connectomes" / f"SC_AAL_{sid}_{weight}.csv",
    }


def _load_img(path: Path) -> tuple[np.ndarray, tuple[int, ...], str]:
    img = nib.load(str(path))
    data = np.asanyarray(img.dataobj)
    return data, tuple(int(v) for v in data.shape[:3]), str(img.affine.round(4).tolist())


def _convert_mif_mask(mask_mif: Path, out_dir: Path) -> Path | None:
    if not mask_mif.exists():
        return None
    out = out_dir / f"{mask_mif.stem}.nii.gz"
    if out.exists():
        return out
    cmd = [str(MRCONVERT), str(mask_mif), str(out), "-quiet", "-force"]
    proc = sp.run(cmd, text=True, stdout=sp.PIPE, stderr=sp.PIPE, check=False)
    if proc.returncode != 0:
        return None
    return out


def _mask_stats(aal_data: np.ndarray, mask_path: Path | None) -> dict[str, float | int | str]:
    prefix = "missing"
    if mask_path is None or not mask_path.exists():
        return {
            "mask_exists": 0,
            "mask_shape": "",
            "aal_label_voxels_in_mask": math.nan,
            "aal_label_overlap_fraction": math.nan,
            "mask_note": prefix,
        }
    try:
        mask_data, mask_shape, _ = _load_img(mask_path)
        if mask_shape != tuple(int(v) for v in aal_data.shape[:3]):
            return {
                "mask_exists": 1,
                "mask_shape": "x".join(str(v) for v in mask_shape),
                "aal_label_voxels_in_mask": math.nan,
                "aal_label_overlap_fraction": math.nan,
                "mask_note": "shape_mismatch",
            }
        aal_nonzero = aal_data > 0
        mask_nonzero = mask_data > 0
        label_voxels = int(aal_nonzero.sum())
        overlap = int((aal_nonzero & mask_nonzero).sum())
        return {
            "mask_exists": 1,
            "mask_shape": "x".join(str(v) for v in mask_shape),
            "aal_label_voxels_in_mask": overlap,
            "aal_label_overlap_fraction": overlap / label_voxels if label_voxels else math.nan,
            "mask_note": "ok",
        }
    except Exception as exc:
        return {
            "mask_exists": int(mask_path.exists()),
            "mask_shape": "",
            "aal_label_voxels_in_mask": math.nan,
            "aal_label_overlap_fraction": math.nan,
            "mask_note": f"{type(exc).__name__}: {exc}",
        }


def aal_metrics(sid: str, variant: str, aal_path: Path, scratch_dir: Path) -> dict[str, object]:
    paths = _subject_paths(sid)
    row: dict[str, object] = {
        "sid": sid,
        "artifact_type": "AAL_b0",
        "variant": variant,
        "path": str(aal_path),
        "exists": int(aal_path.exists()),
        "size_bytes": aal_path.stat().st_size if aal_path.exists() else 0,
        "read_ok": 0,
        "read_error": "",
    }
    if not aal_path.exists() or aal_path.stat().st_size <= 0:
        row["read_error"] = "missing_or_empty"
        return row
    try:
        aal_data, aal_shape, affine = _load_img(aal_path)
        values, counts = np.unique(aal_data.astype(np.int64), return_counts=True)
        label_counts = {
            int(value): int(count)
            for value, count in zip(values, counts)
            if int(value) in VALID_NODES and int(value) > 0
        }
        robust = {label: count for label, count in label_counts.items() if count >= 50}
        tiny = {label: count for label, count in label_counts.items() if 0 < count < 50}
        missing = sorted(VALID_NODES.difference(label_counts))
        row.update(
            {
                "read_ok": 1,
                "shape": "x".join(str(v) for v in aal_shape),
                "affine_rounded": affine,
                "label_voxels": int((aal_data > 0).sum()),
                "valid_labels_present": len(label_counts),
                "labels_ge_50vox": len(robust),
                "tiny_valid_labels": len(tiny),
                "missing_valid_labels": len(missing),
                "missing_valid_label_list": ";".join(str(v) for v in missing[:80]),
                "frontal_oper_l_voxels": label_counts.get(7, 0),
                "frontal_oper_r_voxels": label_counts.get(8, 0),
            }
        )
        tmp = scratch_dir / sid / "_mask_cache"
        tmp.mkdir(parents=True, exist_ok=True)
        mask_5tt = _convert_mif_mask(paths["mask_5tt"], tmp)
        for name, mask_path in (("mask_5tt", mask_5tt), ("mask_b0", paths["mask_b0"])):
            stats = _mask_stats(aal_data, mask_path)
            for key, value in stats.items():
                row[f"{name}_{key}"] = value
    except Exception as exc:
        row["read_error"] = f"{type(exc).__name__}: {exc}"
    return row


def matrix_metrics(sid: str, variant: str, weight: str, matrix_path: Path) -> dict[str, object]:
    row: dict[str, object] = {
        "sid": sid,
        "artifact_type": "matrix",
        "variant": variant,
        "weight": weight,
        "path": str(matrix_path),
        "exists": int(matrix_path.exists()),
        "size_bytes": matrix_path.stat().st_size if matrix_path.exists() else 0,
        "read_ok": 0,
        "read_error": "",
    }
    if not matrix_path.exists() or matrix_path.stat().st_size <= 0:
        row["read_error"] = "missing_or_empty"
        return row
    try:
        matrix = np.genfromtxt(matrix_path, delimiter=",", dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError(f"not square matrix: shape={matrix.shape}")
        n = matrix.shape[0]
        clean = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
        upper = np.triu(np.ones((n, n), dtype=bool), 1)
        row_abs = np.abs(clean).sum(axis=1)
        unexpected_zero = sorted(
            idx + 1 for idx, value in enumerate(row_abs) if value == 0 and (idx + 1) in VALID_NODES
        )
        nonzero_upper = int((np.abs(clean) > 0)[upper].sum())
        possible_upper = int(n * (n - 1) / 2)
        row.update(
            {
                "read_ok": 1,
                "n_rows": n,
                "n_cols": n,
                "finite_fraction": float(np.isfinite(matrix).mean()) if matrix.size else math.nan,
                "symmetry_max_abs": float(np.max(np.abs(clean - clean.T))) if clean.size else math.nan,
                "diagonal_abs_sum": float(np.abs(np.diag(clean)).sum()) if clean.size else math.nan,
                "nonzero_upper_edges": nonzero_upper,
                "density": nonzero_upper / possible_upper if possible_upper else math.nan,
                "unexpected_valid_zero_rows": len(unexpected_zero),
                "unexpected_valid_zero_row_list": ";".join(str(v) for v in unexpected_zero[:80]),
            }
        )
    except Exception as exc:
        row["read_error"] = f"{type(exc).__name__}: {exc}"
    return row


def write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_cmd(cmd: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    proc = sp.run(cmd, text=True, stdout=sp.PIPE, stderr=sp.PIPE, check=False)
    log_path.write_text("$ " + " ".join(cmd) + "\n\n" + proc.stdout + proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"command failed rc={proc.returncode}; see {log_path}")


def build_transform_variants(sid: str, scratch_dir: Path) -> dict[str, Path]:
    paths = _subject_paths(sid)
    required = (FLIRT, CONVERT_XFM, FSL_STD, AAL_MNI, paths["b0"], paths["t1"], paths["t12b0"])
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"{sid}: missing transform inputs: {missing}")

    out: dict[str, Path] = {}
    direct_dir = scratch_dir / sid / "direct_mni_to_b0"
    direct_dir.mkdir(parents=True, exist_ok=True)
    direct_mni2t1 = direct_dir / "mni2t1.mat"
    direct_mni2b0 = direct_dir / "mni2b0.mat"
    direct_aal_b0 = direct_dir / "AAL_b0.nii.gz"
    run_cmd(
        [
            str(FLIRT),
            "-in",
            str(FSL_STD),
            "-ref",
            str(paths["t1"]),
            "-omat",
            str(direct_mni2t1),
            "-dof",
            "12",
        ],
        direct_dir / "flirt_mni2t1.log",
    )
    run_cmd(
        [
            str(CONVERT_XFM),
            "-omat",
            str(direct_mni2b0),
            "-concat",
            str(paths["t12b0"]),
            str(direct_mni2t1),
        ],
        direct_dir / "convert_xfm_mni2b0.log",
    )
    run_cmd(
        [
            str(FLIRT),
            "-in",
            str(AAL_MNI),
            "-ref",
            str(paths["b0"]),
            "-applyxfm",
            "-init",
            str(direct_mni2b0),
            "-interp",
            "nearestneighbour",
            "-out",
            str(direct_aal_b0),
        ],
        direct_dir / "flirt_aal_to_b0_direct.log",
    )
    out["scratch_direct_mni_to_b0"] = direct_aal_b0

    two_step_dir = scratch_dir / sid / "two_step_mni_to_t1_to_b0"
    two_step_dir.mkdir(parents=True, exist_ok=True)
    two_mni2t1 = two_step_dir / "mni2t1.mat"
    two_aal_t1 = two_step_dir / "AAL_t1.nii.gz"
    two_aal_b0 = two_step_dir / "AAL_b0.nii.gz"
    run_cmd(
        [
            str(FLIRT),
            "-in",
            str(FSL_STD),
            "-ref",
            str(paths["t1"]),
            "-omat",
            str(two_mni2t1),
            "-dof",
            "12",
        ],
        two_step_dir / "flirt_mni2t1.log",
    )
    run_cmd(
        [
            str(FLIRT),
            "-in",
            str(AAL_MNI),
            "-ref",
            str(paths["t1"]),
            "-applyxfm",
            "-init",
            str(two_mni2t1),
            "-interp",
            "nearestneighbour",
            "-out",
            str(two_aal_t1),
        ],
        two_step_dir / "flirt_aal_to_t1.log",
    )
    run_cmd(
        [
            str(FLIRT),
            "-in",
            str(two_aal_t1),
            "-ref",
            str(paths["b0"]),
            "-applyxfm",
            "-init",
            str(paths["t12b0"]),
            "-interp",
            "nearestneighbour",
            "-out",
            str(two_aal_b0),
        ],
        two_step_dir / "flirt_aal_to_b0.log",
    )
    out["scratch_two_step_mni_to_t1_to_b0"] = two_aal_b0

    # The restored baseline may have used an older MNI->T1 matrix.  Test that
    # matrix separately so the pilot can distinguish a transform-direction
    # error from a regression in recomputing MNI->T1.
    existing_mni2t1 = paths["parc"] / "mni2t1.mat"
    if existing_mni2t1.exists():
        existing_dir = scratch_dir / sid / "two_step_existing_mni2t1_to_b0"
        existing_dir.mkdir(parents=True, exist_ok=True)
        existing_aal_t1 = existing_dir / "AAL_t1.nii.gz"
        existing_aal_b0 = existing_dir / "AAL_b0.nii.gz"
        run_cmd(
            [
                str(FLIRT),
                "-in",
                str(AAL_MNI),
                "-ref",
                str(paths["t1"]),
                "-applyxfm",
                "-init",
                str(existing_mni2t1),
                "-interp",
                "nearestneighbour",
                "-out",
                str(existing_aal_t1),
            ],
            existing_dir / "flirt_aal_to_t1_existing_mni2t1.log",
        )
        run_cmd(
            [
                str(FLIRT),
                "-in",
                str(existing_aal_t1),
                "-ref",
                str(paths["b0"]),
                "-applyxfm",
                "-init",
                str(paths["t12b0"]),
                "-interp",
                "nearestneighbour",
                "-out",
                str(existing_aal_b0),
            ],
            existing_dir / "flirt_aal_to_b0_existing_mni2t1.log",
        )
        out["scratch_two_step_existing_mni2t1_to_b0"] = existing_aal_b0

        existing_direct_dir = scratch_dir / sid / "direct_existing_mni2t1_to_b0"
        existing_direct_dir.mkdir(parents=True, exist_ok=True)
        existing_mni2b0 = existing_direct_dir / "mni2b0.mat"
        existing_direct_aal_b0 = existing_direct_dir / "AAL_b0.nii.gz"
        run_cmd(
            [
                str(CONVERT_XFM),
                "-omat",
                str(existing_mni2b0),
                "-concat",
                str(paths["t12b0"]),
                str(existing_mni2t1),
            ],
            existing_direct_dir / "convert_xfm_existing_mni2t1_mni2b0.log",
        )
        run_cmd(
            [
                str(FLIRT),
                "-in",
                str(AAL_MNI),
                "-ref",
                str(paths["b0"]),
                "-applyxfm",
                "-init",
                str(existing_mni2b0),
                "-interp",
                "nearestneighbour",
                "-out",
                str(existing_direct_aal_b0),
            ],
            existing_direct_dir / "flirt_aal_to_b0_direct_existing_mni2t1.log",
        )
        out["scratch_direct_existing_mni2t1_to_b0"] = existing_direct_aal_b0

    # This is the closest reproduction of the restored baseline: reuse the
    # already-restored AAL-in-T1 image and only apply the subject's T1->B0 BBR
    # matrix. If this fails, the problem is not simply the MNI->T1 refit.
    restored_aal_t1 = paths["parc"] / "AAL_t1.nii.gz"
    if restored_aal_t1.exists():
        restored_dir = scratch_dir / sid / "reuse_restored_aal_t1_to_b0"
        restored_dir.mkdir(parents=True, exist_ok=True)
        restored_aal_b0 = restored_dir / "AAL_b0.nii.gz"
        run_cmd(
            [
                str(FLIRT),
                "-in",
                str(restored_aal_t1),
                "-ref",
                str(paths["b0"]),
                "-applyxfm",
                "-init",
                str(paths["t12b0"]),
                "-interp",
                "nearestneighbour",
                "-out",
                str(restored_aal_b0),
            ],
            restored_dir / "flirt_restored_aal_t1_to_b0.log",
        )
        out["scratch_reuse_restored_aal_t1_to_b0"] = restored_aal_b0
    return out


def diagnose(sids: list[str], scratch_dir: Path, *, run_transforms: bool) -> int:
    aal_rows: list[dict[str, object]] = []
    matrix_rows: list[dict[str, object]] = []
    decision_rows: list[dict[str, object]] = []

    for sid in sids:
        sources = _variant_sources(sid)
        if run_transforms:
            try:
                sources.update(build_transform_variants(sid, scratch_dir))
            except Exception as exc:
                decision_rows.append(
                    {
                        "sid": sid,
                        "stage": "transform_test",
                        "status": "ERROR",
                        "detail": f"{type(exc).__name__}: {exc}",
                    }
                )
        for variant, aal_path in sources.items():
            aal_rows.append(aal_metrics(sid, variant, aal_path, scratch_dir))
        for weight in WEIGHTS:
            for variant, matrix_path in _matrix_sources(sid, weight).items():
                matrix_rows.append(matrix_metrics(sid, variant, weight, matrix_path))

    write_csv(scratch_dir / "pilot_aal_b0_comparison.csv", aal_rows)
    write_csv(scratch_dir / "pilot_matrix_comparison.csv", matrix_rows)

    # Conservative classification: this is a review aid, not an analysis gate.
    by_sid_variant = {(str(r["sid"]), str(r["variant"])): r for r in aal_rows}
    matrix_by_sid_variant_weight = {
        (str(r["sid"]), str(r["variant"]), str(r.get("weight", ""))): r
        for r in matrix_rows
    }
    for sid in sids:
        current = by_sid_variant.get((sid, "current_restored"), {})
        direct = by_sid_variant.get((sid, "scratch_direct_mni_to_b0"), {})
        two_step = by_sid_variant.get((sid, "scratch_two_step_mni_to_t1_to_b0"), {})
        existing_two_step = by_sid_variant.get((sid, "scratch_two_step_existing_mni2t1_to_b0"), {})
        existing_direct = by_sid_variant.get((sid, "scratch_direct_existing_mni2t1_to_b0"), {})
        restored_aal_t1 = by_sid_variant.get((sid, "scratch_reuse_restored_aal_t1_to_b0"), {})
        failed = by_sid_variant.get((sid, "failed_direct_repair"), {})
        def _num(row: dict[str, object], key: str) -> float:
            try:
                return float(row.get(key, math.nan))
            except Exception:
                return math.nan

        def _matrix_num(variant: str, weight: str, key: str) -> float:
            return _num(matrix_by_sid_variant_weight.get((sid, variant, weight), {}), key)

        def _finite_values(values: Iterable[float]) -> list[float]:
            return [float(v) for v in values if math.isfinite(float(v))]

        current_labels = _num(current, "valid_labels_present")
        direct_labels = _num(direct, "valid_labels_present")
        two_labels = _num(two_step, "valid_labels_present")
        existing_two_labels = _num(existing_two_step, "valid_labels_present")
        existing_direct_labels = _num(existing_direct, "valid_labels_present")
        restored_aal_t1_labels = _num(restored_aal_t1, "valid_labels_present")
        failed_labels = _num(failed, "valid_labels_present")
        current_overlap = _num(current, "mask_5tt_aal_label_overlap_fraction")
        two_overlap = _num(two_step, "mask_5tt_aal_label_overlap_fraction")
        direct_overlap = _num(direct, "mask_5tt_aal_label_overlap_fraction")
        existing_two_overlap = _num(existing_two_step, "mask_5tt_aal_label_overlap_fraction")
        existing_direct_overlap = _num(existing_direct, "mask_5tt_aal_label_overlap_fraction")
        restored_aal_t1_overlap = _num(restored_aal_t1, "mask_5tt_aal_label_overlap_fraction")
        current_density = _matrix_num("current_restored", "fd_sum", "density")
        failed_direct_density = _matrix_num("failed_direct_repair", "fd_sum", "density")
        current_zero_rows = _matrix_num("current_restored", "fd_sum", "unexpected_valid_zero_rows")
        failed_direct_zero_rows = _matrix_num("failed_direct_repair", "fd_sum", "unexpected_valid_zero_rows")
        candidate_label_values = _finite_values(
            (two_labels, existing_two_labels, existing_direct_labels, restored_aal_t1_labels)
        )
        candidate_overlap_values = _finite_values(
            (two_overlap, existing_two_overlap, existing_direct_overlap, restored_aal_t1_overlap)
        )
        best_scratch_labels = max(candidate_label_values) if candidate_label_values else math.nan
        best_scratch_overlap = max(candidate_overlap_values) if candidate_overlap_values else math.nan
        production_pilot_allowed = 0
        suggested_next_lane = "manual_review"
        recommendation = "review"
        review_reason = "insufficient scratch-transform evidence"
        if current_labels < 120 or current_overlap < 0.5:
            recommendation = "registration_or_source_repair_before_post"
            suggested_next_lane = "bbr_registration_pilot"
            review_reason = "restored AAL_b0 has low label survival or poor 5TT overlap"
        elif (
            math.isfinite(failed_direct_density)
            and math.isfinite(current_density)
            and failed_direct_density < current_density
        ) or (math.isfinite(direct_labels) and direct_labels < current_labels) or (
            math.isfinite(direct_overlap) and direct_overlap < current_overlap
        ):
            recommendation = "do_not_use_direct_transform_keep_restored_baseline"
            suggested_next_lane = "assignment_or_tracks_pilot" if current_density < 0.15 else "hold_or_manual_review"
            review_reason = "failed/direct repair worsened labels, overlap, or matrix density"
        elif run_transforms and best_scratch_labels >= current_labels and best_scratch_overlap >= current_overlap:
            recommendation = "candidate_transform_for_one_subject_production_pilot"
            suggested_next_lane = "single_subject_post_pilot"
            production_pilot_allowed = 1
            review_reason = "scratch transform matched or improved restored AAL label survival and 5TT overlap"
        if current_labels < 120 or current_overlap < 0.5:
            production_pilot_allowed = 0
        decision_rows.append(
            {
                "sid": sid,
                "stage": "pilot_review",
                "status": "READY_FOR_REVIEW",
                "current_valid_labels": current_labels,
                "failed_direct_valid_labels": failed_labels,
                "scratch_direct_valid_labels": direct_labels,
                "scratch_two_step_valid_labels": two_labels,
                "scratch_two_step_existing_mni2t1_valid_labels": existing_two_labels,
                "scratch_direct_existing_mni2t1_valid_labels": existing_direct_labels,
                "scratch_reuse_restored_aal_t1_valid_labels": restored_aal_t1_labels,
                "current_5tt_overlap_fraction": current_overlap,
                "scratch_direct_5tt_overlap_fraction": direct_overlap,
                "scratch_two_step_5tt_overlap_fraction": two_overlap,
                "scratch_two_step_existing_mni2t1_5tt_overlap_fraction": existing_two_overlap,
                "scratch_direct_existing_mni2t1_5tt_overlap_fraction": existing_direct_overlap,
                "scratch_reuse_restored_aal_t1_5tt_overlap_fraction": restored_aal_t1_overlap,
                "current_fd_sum_density": current_density,
                "failed_direct_fd_sum_density": failed_direct_density,
                "current_fd_sum_unexpected_zero_rows": current_zero_rows,
                "failed_direct_fd_sum_unexpected_zero_rows": failed_direct_zero_rows,
                "best_scratch_valid_labels": best_scratch_labels,
                "best_scratch_5tt_overlap_fraction": best_scratch_overlap,
                "production_pilot_allowed": production_pilot_allowed,
                "suggested_next_lane": suggested_next_lane,
                "recommendation": recommendation,
                "review_reason": review_reason,
            }
        )

    write_csv(scratch_dir / "pilot_decision_summary.csv", decision_rows)
    print(f"Pilot diagnostics written under: {scratch_dir}")
    print(f"AAL comparison: {scratch_dir / 'pilot_aal_b0_comparison.csv'}")
    print(f"Matrix comparison: {scratch_dir / 'pilot_matrix_comparison.csv'}")
    print(f"Decision summary: {scratch_dir / 'pilot_decision_summary.csv'}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sids", nargs="*", default=list(PILOT_SIDS))
    parser.add_argument("--qc-root", type=Path, default=QC_ROOT)
    parser.add_argument("--stamp", default=time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()))
    parser.add_argument(
        "--run-transform-tests",
        action="store_true",
        help="Generate scratch AAL_b0 variants under pilot_repair/<stamp>; production outputs remain untouched.",
    )
    args = parser.parse_args()

    sids = sorted(dict.fromkeys(args.sids))
    scratch_dir = args.qc_root / "pilot_repair" / args.stamp
    scratch_dir.mkdir(parents=True, exist_ok=True)
    return diagnose(sids, scratch_dir, run_transforms=args.run_transform_tests)


if __name__ == "__main__":
    raise SystemExit(main())
