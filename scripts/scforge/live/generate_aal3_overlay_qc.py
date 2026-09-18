#!/usr/bin/env python3
"""Generate visual QC overlays for AAL3 rescue candidates.

The same script is used for route 1 source-contract candidates and later route
families. Candidate paths are resolved from the route decision row plus the
probe's label/matrix summaries instead of hard-coded route-1 filenames.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


RUN_ROOT = Path("/data/derivatives/qc/sc_matrix_qc/aal3_force_connectome_ad_batch/ad_existing_tracks_20260529T055752Z")


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size <= 0:
        return []
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        if fields:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    tmp.replace(path)


def robust_bg(data: np.ndarray) -> np.ndarray:
    data = np.asarray(data, dtype=float)
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return np.zeros_like(data, dtype=float)
    lo, hi = np.percentile(finite, [1, 99])
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((data - lo) / (hi - lo), 0, 1)


def crop_bounds(mask: np.ndarray, pad: int = 8) -> tuple[slice, slice, slice]:
    pts = np.argwhere(mask)
    if pts.size == 0:
        return tuple(slice(0, s) for s in mask.shape)  # type: ignore[return-value]
    mins = np.maximum(pts.min(axis=0) - pad, 0)
    maxs = np.minimum(pts.max(axis=0) + pad + 1, mask.shape)
    return tuple(slice(int(a), int(b)) for a, b in zip(mins, maxs))  # type: ignore[return-value]


def slice2d(vol: np.ndarray, axis: int, idx: int) -> np.ndarray:
    if axis == 0:
        arr = vol[idx, :, :]
    elif axis == 1:
        arr = vol[:, idx, :]
    else:
        arr = vol[:, :, idx]
    return np.rot90(arr)


def make_overlay(bg: np.ndarray, labels: np.ndarray, axis: int, idx: int) -> np.ndarray:
    b = slice2d(bg, axis, idx)
    lab = slice2d(labels, axis, idx)
    rgb = np.dstack([b, b, b])
    mask = lab > 0
    if mask.any():
        edge = mask ^ (
            np.roll(mask, 1, 0)
            & np.roll(mask, -1, 0)
            & np.roll(mask, 1, 1)
            & np.roll(mask, -1, 1)
        )
        rgb[mask, 0] = np.maximum(rgb[mask, 0], 0.85)
        rgb[mask, 1] *= 0.45
        rgb[mask, 2] *= 0.45
        rgb[edge, :] = [0.0, 1.0, 0.0]
    return rgb


def candidate_rows(run_root: Path) -> list[dict[str, str]]:
    rows = read_csv(run_root / "route_qc_decisions.csv")
    return [
        row
        for row in rows
        if row.get("qc_decision")
        in {"PROMOTE_CANDIDATE_PENDING_VISUAL", "INTERIM_REVIEW_PROMOTION_PENDING_VISUAL"}
    ]


def first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists() and path.stat().st_size > 0:
            return path
    return None


def find_label_path(row: dict[str, str], probe_root: Path) -> Path:
    stage = row.get("tested_route") or row.get("best_candidate") or "AAL3_source_contract_B0"
    label_rows = read_csv(probe_root / "label_survival_summary.csv")
    for label_row in label_rows:
        if label_row.get("stage") == stage and label_row.get("path"):
            path = Path(label_row["path"])
            if path.exists():
                return path
    if stage == "AAL3_source_contract" or row.get("best_candidate") == "AAL3_source_contract":
        path = probe_root / "parc" / "AAL3_source_contract_B0.nii.gz"
        if path.exists():
            return path
    candidates = sorted((probe_root / "parc").rglob("*AAL3*b0*.nii.gz"))
    if candidates:
        preferred = [p for p in candidates if stage in str(p)]
        return preferred[-1] if preferred else candidates[-1]
    raise FileNotFoundError(f"Could not resolve candidate parcellation for {row.get('sid')} under {probe_root}")


def find_background_path(sid: str, probe_root: Path, parc: Path) -> Path:
    paths = [
        probe_root / "parc" / "corrected_T1_on_eddy_vol0.nii.gz",
        probe_root / "b0_candidates" / f"{sid}_eddy_vol0.nii.gz",
        Path("/home/ec2-user/exp/data/derivatives/dwi_t1_bbr") / f"{sid}_b0mean_ras.nii.gz",
        Path("/data/derivatives/dwi_t1_bbr") / f"{sid}_b0mean_ras.nii.gz",
    ]
    paths.extend(sorted((probe_root / "b0_candidates").glob("*eddy_vol0.nii.gz")))
    return first_existing(paths) or parc


def generate_subject(
    row: dict[str, str],
    out_dir: Path,
    run_root: Path,
    previous: dict[str, dict[str, str]],
) -> dict[str, object]:
    sid = row["sid"]
    prior = previous.get(sid, {})
    probe_root = Path(row["probe_root"]) if row.get("probe_root") else None
    if not probe_root or not probe_root.exists():
        # route_qc_decisions written by the scorer has no probe_root column yet; infer from batch results.
        batch_rows = read_csv(run_root / "batch_results.csv")
        match = next((r for r in batch_rows if r.get("sid") == sid), {})
        probe_root = Path(match.get("probe_root", ""))
    parc = find_label_path(row, probe_root)
    bg_path = find_background_path(sid, probe_root, parc)
    labels = np.asanyarray(nib.load(str(parc)).dataobj)
    bg = robust_bg(np.asanyarray(nib.load(str(bg_path)).dataobj))
    mask = labels > 0
    bounds = crop_bounds(mask)
    labels_c = labels[bounds]
    bg_c = bg[bounds]
    pts = np.argwhere(labels_c > 0)
    center = np.median(pts, axis=0).astype(int) if pts.size else np.array(bg_c.shape) // 2
    cuts = [
        ("sagittal", 0, int(center[0])),
        ("coronal", 1, int(center[1])),
        ("axial_low", 2, int(np.percentile(pts[:, 2], 35)) if pts.size else int(center[2])),
        ("axial_mid", 2, int(center[2])),
        ("axial_high", 2, int(np.percentile(pts[:, 2], 65)) if pts.size else int(center[2])),
    ]
    fig, axes = plt.subplots(1, len(cuts), figsize=(18, 4), dpi=140)
    for ax, (name, axis, idx) in zip(axes, cuts):
        ax.imshow(make_overlay(bg_c, labels_c, axis, idx), origin="lower")
        ax.set_title(f"{name} {idx}", fontsize=9)
        ax.axis("off")
    fig.suptitle(
        f"{sid} AAL3 {row.get('best_candidate') or row.get('tested_route')} overlay | labels {row.get('source_labels')}/166 | "
        f"zero {row.get('best_zero_rows')} | density {float(row.get('best_density') or 0):.3f}",
        fontsize=11,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{sid}_aal3_overlay_qc.png"
    fig.tight_layout()
    fig.savefig(png, facecolor="white")
    plt.close(fig)
    return {
        "sid": sid,
        "best_candidate": row.get("best_candidate", ""),
        "best_variant": row.get("best_variant", ""),
        "qc_decision": row.get("qc_decision", ""),
        "overlay_png": str(png),
        "parcellation": str(parc),
        "background": str(bg_path),
        "visual_overlay_status": prior.get("visual_overlay_status") or "generated_pending_review",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    args = parser.parse_args()
    run_root = args.run_root
    rows = candidate_rows(run_root)
    out_dir = run_root / "visual_qc"
    previous = {row.get("sid", ""): row for row in read_csv(run_root / "visual_overlay_qc_manifest.csv")}
    outputs = [generate_subject(row, out_dir, run_root, previous) for row in rows]
    write_csv(run_root / "visual_overlay_qc_manifest.csv", outputs)
    print(f"generated {len(outputs)} overlays")
    for row in outputs:
        print(row["sid"], row["overlay_png"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
    prior = previous.get(sid, {})
