#!/usr/bin/env python3
"""Route 20: zero-row ROI endpoint rescue.

The previous zero-ROI hybrid route seeded inside disconnected AAL3 labels, but
the generated target tracks almost never assigned endpoints back to those
labels. This route changes the testable mechanism: seed ACT tractography from a
mask of the currently zero-row AAL3 labels, merge the targeted tracks with
production tracks, recompute SIFT2, and score with the same AAL3 QC gates.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np


ROOT = Path("/home/ec2-user/exp")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.scforge.live import run_aal3_route7_selective_rescue_batch as route7  # noqa: E402
from scripts.scforge.live.run_aal3_route11b_act_sift2_track_coverage_rerun import (  # noqa: E402
    DEFAULT_ROUTE1_ROOT,
    RUN_PARENT,
    collect_route11_parcellations,
    read_csv,
    refresh_outputs,
)
from scripts.scforge.live.run_sc_aal3_source_contract_probe import (  # noqa: E402
    QC_ROOT,
    aal3_valid_labels,
    discover_inputs,
    write_csv,
    write_json,
)


LANE = "R20"
SCHEMA = "route20_zero_roi_gmwmi_endpoint_v2_per_roi"
SUFFIX = "_route20_zero_roi_gmwmi_endpoint"
PROBE_NAME = "aal3_route20_zero_roi_gmwmi_endpoint_probe"
MRCONVERT = Path("/home/ec2-user/mrtrix3/bin/mrconvert")
TCKEDIT = Path("/home/ec2-user/mrtrix3/bin/tckedit")
TCKGEN_RE = re.compile(
    r"(?P<seeds>\d+)\s+seeds,\s+(?P<streamlines>\d+)\s+streamlines,\s+(?P<selected>\d+)\s+selected"
)


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def update_status(root: Path, **updates: Any) -> None:
    status = read_json(root / "status.json")
    status.update(updates)
    status["updated_utc"] = utc()
    write_json(root / "status.json", status)


def run_cmd(cmd: list[str | Path], log_path: Path, env: dict[str, str], timeout_sec: int) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    str_cmd = [str(item) for item in cmd]
    with log_path.open("a", encoding="utf-8", errors="replace") as handle:
        handle.write(f"[{utc()}] $ {' '.join(str_cmd)}\n")
        handle.write(f"[{utc()}] timeout_sec={timeout_sec}\n")
        handle.flush()
        try:
            proc = subprocess.run(
                str_cmd,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
                timeout=timeout_sec,
            )
        except subprocess.TimeoutExpired as exc:
            handle.write(f"[{utc()}] timeout after {timeout_sec}s\n\n")
            raise TimeoutError(f"command timed out after {timeout_sec}s; see {log_path}") from exc
        handle.write(f"[{utc()}] rc={proc.returncode}\n\n")
    if proc.returncode != 0:
        raise RuntimeError(f"command failed rc={proc.returncode}; see {log_path}")


def zero_row_labels(matrix_path: Path, valid_labels: list[int]) -> list[int]:
    mat = np.loadtxt(matrix_path, delimiter=",")
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError(f"not a square matrix: {matrix_path}")
    valid = set(int(v) for v in valid_labels)
    row_signal = np.abs(mat).sum(axis=0) + np.abs(mat).sum(axis=1)
    return [idx + 1 for idx, value in enumerate(row_signal) if value == 0 and (idx + 1) in valid]


def write_zero_roi_seed_mask(
    *,
    parc_path: Path,
    matrix_path: Path,
    valid_labels: list[int],
    out_path: Path,
    dilation_iters: int,
    constraint_path: Path | None = None,
    constraint_name: str = "",
    labels_override: list[int] | None = None,
) -> dict[str, Any]:
    labels = labels_override if labels_override is not None else zero_row_labels(matrix_path, valid_labels)
    img = nib.load(str(parc_path))
    data = np.asanyarray(img.dataobj).astype(np.int32)
    seed = np.isin(data, labels)
    original_voxels = int(seed.sum())
    if dilation_iters > 0:
        try:
            from scipy import ndimage as ndi

            seed = ndi.binary_dilation(seed, iterations=dilation_iters)
        except Exception:
            dilation_iters = 0
    constraint_voxels = ""
    constraint_overlap_voxels = ""
    if constraint_path is not None:
        constraint_img = nib.load(str(constraint_path))
        constraint = np.asanyarray(constraint_img.dataobj) > 0
        if constraint.shape != seed.shape:
            raise ValueError(f"seed constraint shape mismatch: {constraint_path} has {constraint.shape}, parc has {seed.shape}")
        constraint_voxels = int(constraint.sum())
        seed = seed & constraint
        constraint_overlap_voxels = int(seed.sum())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out = nib.Nifti1Image(seed.astype(np.uint8), img.affine, img.header)
    out.set_data_dtype(np.uint8)
    nib.save(out, str(out_path))
    return {
        "seed_mask": str(out_path),
        "zero_label_count": len(labels),
        "zero_labels": ";".join(str(label) for label in labels),
        "zero_roi_voxels": original_voxels,
        "seed_voxels": int(seed.sum()),
        "dilation_iters": dilation_iters,
        "constraint_name": constraint_name,
        "constraint_path": str(constraint_path or ""),
        "constraint_voxels": constraint_voxels,
        "constraint_overlap_voxels": constraint_overlap_voxels,
    }


def seed_args_for_mask(seed_mode: str, seed_mask: Path, seeds_per_voxel: int) -> list[str | Path]:
    if seed_mode == "gmwmi":
        return ["-crop_at_gmwmi", "-seed_gmwmi", seed_mask]
    if seed_mode == "image":
        return ["-seed_image", seed_mask]
    if seed_mode == "random_per_voxel":
        return ["-seed_random_per_voxel", seed_mask, str(max(1, seeds_per_voxel))]
    if seed_mode == "grid_per_voxel":
        return ["-seed_grid_per_voxel", seed_mask, str(max(1, seeds_per_voxel))]
    raise ValueError(f"unsupported seed mode: {seed_mode}")


def latest_tckgen_progress(log_path: Path) -> dict[str, int]:
    if not log_path.exists():
        return {"tckgen_seeds": 0, "tckgen_streamlines": 0, "tckgen_selected": 0}
    text = log_path.read_text(encoding="utf-8", errors="replace")[-20000:]
    matches = list(TCKGEN_RE.finditer(text))
    if not matches:
        return {"tckgen_seeds": 0, "tckgen_streamlines": 0, "tckgen_selected": 0}
    match = matches[-1]
    return {
        "tckgen_seeds": int(match.group("seeds")),
        "tckgen_streamlines": int(match.group("streamlines")),
        "tckgen_selected": int(match.group("selected")),
    }


def seed_constraint_nifti(
    *,
    sid: str,
    seed_space: str,
    fpaths: dict[str, Path],
    seeds_dir: Path,
    env: dict[str, str],
) -> tuple[Path | None, str]:
    if seed_space == "none":
        return None, ""
    fod_root = fpaths["fod_root"]
    source_by_space = {
        "fod_mask": fod_root / "mask_fod.mif",
        "dwi_mask": fod_root / "mask_dwi.mif",
        "gmwmi_mask": fpaths["gmwmi"],
    }
    source = source_by_space[seed_space]
    if not source.exists():
        raise FileNotFoundError(f"seed-space {seed_space} source missing for {sid}: {source}")
    out = seeds_dir / f"{sid}_{seed_space}.nii.gz"
    if not out.exists() or out.stat().st_size <= 0:
        run_cmd([MRCONVERT, source, out, "-force"], seeds_dir / f"mrconvert_{seed_space}.log", env, timeout_sec=20 * 60)
    return out, seed_space


def parse_zero_labels(seed_info: dict[str, Any]) -> set[int]:
    labels: set[int] = set()
    for item in str(seed_info.get("zero_labels") or "").split(";"):
        item = item.strip()
        if not item:
            continue
        try:
            labels.add(int(float(item)))
        except Exception:
            pass
    return labels


def assignment_zero_roi_summary(
    *,
    assignment_file: Path,
    zero_labels: set[int],
    variant: str,
) -> list[dict[str, Any]]:
    counts = {label: {"incident_streamlines": 0, "endpoint_hits": 0} for label in sorted(zero_labels)}
    total = 0
    if not assignment_file.exists() or assignment_file.stat().st_size <= 0:
        return [
            {
                "variant": variant,
                "label": label,
                "incident_streamlines": 0,
                "endpoint_hits": 0,
                "assignment_rows": 0,
                "assignment_file": str(assignment_file),
            }
            for label in sorted(zero_labels)
        ]
    with assignment_file.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(",", " ").split()
            if len(parts) < 2:
                continue
            try:
                a = int(float(parts[0]))
                b = int(float(parts[1]))
            except Exception:
                continue
            total += 1
            seen = set()
            for label in (a, b):
                if label in counts:
                    counts[label]["endpoint_hits"] += 1
                    seen.add(label)
            for label in seen:
                counts[label]["incident_streamlines"] += 1
    return [
        {
            "variant": variant,
            "label": label,
            "incident_streamlines": values["incident_streamlines"],
            "endpoint_hits": values["endpoint_hits"],
            "assignment_rows": total,
            "assignment_file": str(assignment_file),
        }
        for label, values in counts.items()
    ]


def route20_subject(
    *,
    sid: str,
    route1_root: Path,
    route2_root: Path,
    run_root: Path,
    target_streamlines: int,
    dilation_iters: int,
    variants: list[str],
    mrtrix_threads: int,
    seed_mode: str,
    seed_space: str,
    seeds_per_voxel: int,
    cutoff: float,
    per_roi: bool,
    per_roi_target_streamlines: int,
    per_roi_max_seeds: int,
    max_zero_rois: int,
    force_retry: bool,
) -> dict[str, Any]:
    probe_root = QC_ROOT / PROBE_NAME / f"{run_root.name}_{sid}"
    probe_root.mkdir(parents=True, exist_ok=True)
    if (probe_root / "decision.json").exists() and not force_retry:
        return read_json(probe_root / "decision.json").get("batch_row", {})

    update_status(
        probe_root,
        sid=sid,
        tag=f"{run_root.name}_{sid}",
        phase="starting",
        route20_action=f"zero_roi_{seed_mode}_endpoint",
    )
    inputs = discover_inputs(sid)
    missing = [key for key in ("production_fd_sum", "tracks") if not inputs.get(key)]
    fpaths = route7.fod_paths(sid)
    missing += [key for key, path in fpaths.items() if key != "deconv_mode" and not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing route20 inputs for {sid}: {missing}")

    env = route7.env_for_mrtrix(mrtrix_threads)
    valid_labels = aal3_valid_labels()
    label_rows, matrix_rows, baseline_fd = route7.production_baseline(inputs, valid_labels)
    parc_candidates = collect_route11_parcellations(sid, route1_root, route2_root, valid_labels)
    parc_candidates = [cand for cand in parc_candidates if int(cand.get("labels") or 0) >= 160]
    if not parc_candidates:
        batch_row = route7.row_from_best(
            sid,
            "ok",
            probe_root,
            {},
            baseline_fd,
            label_rows,
            {"route20_reason": "no high-label parcellation for zero-ROI GMWMI endpoint rescue"},
        )
        write_csv(probe_root / "label_survival_summary.csv", label_rows)
        write_csv(probe_root / "matrix_candidate_summary.csv", matrix_rows)
        write_json(probe_root / "decision.json", {"decision": "AAL3_ROUTE20_NO_HIGH_LABEL_PARCELLATION", "best": {}, "batch_row": batch_row})
        update_status(probe_root, phase="complete", completed_utc=utc(), decision="AAL3_ROUTE20_NO_HIGH_LABEL_PARCELLATION")
        return batch_row

    parc_candidate = parc_candidates[0]
    base_stage = str(parc_candidate["candidate"])
    label_rows.append({"stage": base_stage, **parc_candidate["label_stats"]})
    write_csv(probe_root / "label_survival_summary.csv", label_rows)
    write_csv(probe_root / "matrix_candidate_summary.csv", matrix_rows)

    track_root = probe_root / "tracks"
    logs = probe_root / "logs"
    seeds = probe_root / "seeds"
    track_root.mkdir(exist_ok=True)
    logs.mkdir(exist_ok=True)
    seeds.mkdir(exist_ok=True)

    constraint_path, constraint_name = seed_constraint_nifti(
        sid=sid,
        seed_space=seed_space,
        fpaths=fpaths,
        seeds_dir=seeds,
        env=env,
    )
    zero_labels_all = zero_row_labels(Path(inputs["production_fd_sum"]), valid_labels)
    zero_labels_run = list(zero_labels_all)
    if max_zero_rois > 0:
        zero_labels_run = zero_labels_run[:max_zero_rois]
    cutoff_tag = str(cutoff).replace(".", "p")
    target_tracks_for_merge: list[Path] = []
    seed_rows: list[dict[str, Any]] = []

    if per_roi:
        target_suffix = f"{per_roi_target_streamlines // 1000}kperroi"
        mode_tag = f"perroi_{seed_mode}_dil{dilation_iters}_cut{cutoff_tag}"
        if seed_space != "none":
            mode_tag = f"{mode_tag}_{seed_space}"
        for label in zero_labels_run:
            seed_mask = seeds / f"{sid}_zero_roi_label{label}_{seed_space}_dil{dilation_iters}_seed.nii.gz"
            seed_row = write_zero_roi_seed_mask(
                parc_path=Path(parc_candidate["path"]),
                matrix_path=Path(inputs["production_fd_sum"]),
                valid_labels=valid_labels,
                out_path=seed_mask,
                dilation_iters=dilation_iters,
                constraint_path=constraint_path,
                constraint_name=constraint_name,
                labels_override=[label],
            )
            seed_row["label"] = label
            seed_row["seed_policy"] = "per_roi_fixed_target"
            seed_rows.append(seed_row)
            if int(seed_row["seed_voxels"]) <= 0:
                continue
            target_tracks = track_root / f"{sid}_zero_roi_label{label}_{mode_tag}_{target_suffix}.tck"
            log_path = logs / f"tckgen_zero_roi_label{label}_{mode_tag}_{target_suffix}.log"
            if not target_tracks.exists() or target_tracks.stat().st_size <= 0:
                update_status(
                    probe_root,
                    phase=f"tckgen_zero_roi_{seed_mode}_label_{label}",
                    zero_label_count=len(zero_labels_all),
                    zero_labels_attempted=len(zero_labels_run),
                    active_zero_label=label,
                    seed_voxels=seed_row["seed_voxels"],
                    seed_mode=seed_mode,
                    seed_space=seed_space,
                    seeds_per_voxel=seeds_per_voxel,
                    cutoff=cutoff,
                    select_streamlines=per_roi_target_streamlines,
                )
                try:
                    route7.run_track_cmd(
                        [
                            route7.TCKGEN,
                            fpaths["wmfod"],
                            target_tracks,
                            "-act",
                            fpaths["act"],
                            "-backtrack",
                            *seed_args_for_mask(seed_mode, seed_mask, seeds_per_voxel),
                            "-cutoff",
                            str(cutoff),
                            "-maxlength",
                            "250",
                            "-minlength",
                            "10",
                            "-select",
                            str(per_roi_target_streamlines),
                            "-seeds",
                            str(max(1, per_roi_max_seeds)),
                            "-nthreads",
                            str(mrtrix_threads),
                            "-force",
                        ],
                        log_path,
                        env,
                        "tckgen",
                    )
                except Exception as exc:
                    seed_row["tckgen_error"] = f"{type(exc).__name__}: {exc}"
                    write_csv(probe_root / "zero_roi_seed_summary.csv", seed_rows)
                seed_row.update(latest_tckgen_progress(log_path))
                write_csv(probe_root / "zero_roi_seed_summary.csv", seed_rows)
            if (
                target_tracks.exists()
                and target_tracks.stat().st_size > 0
                and int(seed_row.get("tckgen_selected") or 0) > 0
            ):
                target_tracks_for_merge.append(target_tracks)
        seed_info = {
            "seed_mask": "per_roi",
            "zero_label_count": len(zero_labels_all),
            "zero_labels": ";".join(str(label) for label in zero_labels_all),
            "zero_labels_attempted": len(zero_labels_run),
            "zero_roi_voxels": sum(int(row.get("zero_roi_voxels") or 0) for row in seed_rows),
            "seed_voxels": sum(int(row.get("seed_voxels") or 0) for row in seed_rows),
            "dilation_iters": dilation_iters,
            "constraint_name": constraint_name,
            "constraint_path": str(constraint_path or ""),
            "per_roi": 1,
            "per_roi_target_streamlines": per_roi_target_streamlines,
            "per_roi_max_seeds": per_roi_max_seeds,
            "max_zero_rois": max_zero_rois,
            "target_track_count": len(target_tracks_for_merge),
        }
        write_csv(probe_root / "zero_roi_seed_summary.csv", seed_rows + [dict(seed_info, label="ALL", seed_policy="per_roi_aggregate")])
    else:
        target_suffix = f"{target_streamlines // 1000}k"
        mode_tag = f"{seed_mode}_dil{dilation_iters}_cut{cutoff_tag}"
        if seed_space != "none":
            mode_tag = f"{mode_tag}_{seed_space}"
        seed_mask = seeds / f"{sid}_zero_roi_{seed_space}_dil{dilation_iters}_seed.nii.gz"
        seed_info = write_zero_roi_seed_mask(
            parc_path=Path(parc_candidate["path"]),
            matrix_path=Path(inputs["production_fd_sum"]),
            valid_labels=valid_labels,
            out_path=seed_mask,
            dilation_iters=dilation_iters,
            constraint_path=constraint_path,
            constraint_name=constraint_name,
        )
        seed_info["per_roi"] = 0
        write_csv(probe_root / "zero_roi_seed_summary.csv", [seed_info])
        target_tracks = track_root / f"{sid}_zero_roi_{mode_tag}_{target_suffix}.tck"
        if int(seed_info["zero_label_count"]) > 0 and int(seed_info["seed_voxels"]) > 0:
            if not target_tracks.exists() or target_tracks.stat().st_size <= 0:
                update_status(
                    probe_root,
                    phase=f"tckgen_zero_roi_{seed_mode}",
                    zero_label_count=seed_info["zero_label_count"],
                    seed_voxels=seed_info["seed_voxels"],
                    seed_mode=seed_mode,
                    seed_space=seed_space,
                    seeds_per_voxel=seeds_per_voxel,
                    cutoff=cutoff,
                    select_streamlines=target_streamlines,
                )
                route7.run_track_cmd(
                    [
                        route7.TCKGEN,
                        fpaths["wmfod"],
                        target_tracks,
                        "-act",
                        fpaths["act"],
                        "-backtrack",
                        *seed_args_for_mask(seed_mode, seed_mask, seeds_per_voxel),
                        "-cutoff",
                        str(cutoff),
                        "-maxlength",
                        "250",
                        "-minlength",
                        "10",
                        "-select",
                        str(target_streamlines),
                        "-nthreads",
                        str(mrtrix_threads),
                        "-force",
                    ],
                    logs / f"tckgen_zero_roi_{mode_tag}_{target_suffix}.log",
                    env,
                    "tckgen",
                )
            if target_tracks.exists() and target_tracks.stat().st_size > 0:
                target_tracks_for_merge.append(target_tracks)

    if int(seed_info["zero_label_count"]) == 0 or int(seed_info["seed_voxels"]) == 0 or not target_tracks_for_merge:
        batch_row = route7.row_from_best(
            sid,
            "ok",
            probe_root,
            {},
            baseline_fd,
            label_rows,
            {
                "route20_action": f"zero_roi_{seed_mode}_endpoint",
                "route20_reason": "production matrix has no valid zero-row labels or no seed voxels",
                "route11_schema": SCHEMA,
                "route11_lane": LANE,
                "target_lane": LANE,
            },
        )
        write_json(probe_root / "decision.json", {"decision": "AAL3_ROUTE20_NO_ZERO_ROI_SEED", "best": {}, "batch_row": batch_row})
        update_status(probe_root, phase="complete", completed_utc=utc(), decision="AAL3_ROUTE20_NO_ZERO_ROI_SEED", latest=batch_row)
        return batch_row

    combined_tracks = track_root / f"{sid}_prod3m_plus_zero_roi_{mode_tag}_{target_suffix}.tck"
    weights = track_root / f"{sid}_prod3m_plus_zero_roi_{mode_tag}_{target_suffix}_sift2_weights.txt"
    mu = track_root / f"{sid}_prod3m_plus_zero_roi_{mode_tag}_{target_suffix}_mu.txt"
    label_stage = f"route20_prod3m_plus_zero_roi_{mode_tag}_{target_suffix}_{base_stage}"

    if not combined_tracks.exists() or combined_tracks.stat().st_size <= 0:
        update_status(probe_root, phase=f"tckedit_merge_production_zero_roi_{seed_mode}")
        run_cmd(
            [TCKEDIT, inputs["tracks"], *target_tracks_for_merge, combined_tracks, "-nthreads", str(mrtrix_threads), "-force"],
            logs / f"tckedit_prod3m_plus_zero_roi_{mode_tag}_{target_suffix}.log",
            env,
            timeout_sec=2 * 60 * 60,
        )

    preflight = route7.endpoint_preflight_for_tracks(
        sid=sid,
        probe_root=probe_root,
        label_stage=label_stage,
        parc_path=Path(parc_candidate["path"]),
        tracks=combined_tracks,
        env=env,
    )
    if int(route7.fnum(preflight.get("endpoint_preflight_pass"), 0)) != 1:
        batch_row = route7.row_from_best(
            sid,
            "ok",
            probe_root,
            {},
            baseline_fd,
            label_rows,
            {
                "route20_action": f"zero_roi_{seed_mode}_endpoint",
                "route20_reason": (
                    "merged production+zero-ROI tracks failed endpoint preflight before SIFT2/connectome"
                ),
                "route11_schema": SCHEMA,
                "route11_lane": LANE,
                "target_lane": LANE,
                "target_streamlines": target_streamlines,
                "dilation_iters": seed_info["dilation_iters"],
                "seed_mode": seed_mode,
                "seed_space": seed_space,
                "zero_label_count": seed_info["zero_label_count"],
                "zero_labels_attempted": seed_info.get("zero_labels_attempted", seed_info["zero_label_count"]),
                "target_track_count": len(target_tracks_for_merge),
                "combined_tracks": str(combined_tracks),
                "endpoint_preflight_decision": preflight.get("endpoint_preflight_decision", ""),
                "endpoint_preflight_failure_class": preflight.get("failure_class", ""),
                "endpoint_preflight_unique_labels": preflight.get("unique_endpoint_labels", ""),
                "endpoint_preflight_top3_fraction": preflight.get("top3_endpoint_label_fraction", ""),
                "endpoint_preflight_on_label_fraction": preflight.get("endpoint_on_label_fraction", ""),
            },
        )
        write_json(
            probe_root / "decision.json",
            {
                "decision": "AAL3_ROUTE20_ENDPOINT_PREFLIGHT_FAILED",
                "endpoint_preflight": preflight,
                "best": {},
                "batch_row": batch_row,
            },
        )
        update_status(
            probe_root,
            phase="complete",
            completed_utc=utc(),
            decision="AAL3_ROUTE20_ENDPOINT_PREFLIGHT_FAILED",
            latest=batch_row,
        )
        return batch_row

    if not weights.exists() or weights.stat().st_size <= 0:
        update_status(probe_root, phase=f"tcksift2_prod3m_plus_zero_roi_{seed_mode}")
        fd_scale_gm_default = route7.deconv_mode(fpaths["deconv_mode"]) == "single_shell"
        last_error: Exception | None = None
        for idx, fd_scale_gm in enumerate([fd_scale_gm_default, not fd_scale_gm_default], start=1):
            cmd = [
                route7.TCKSIFT2,
                combined_tracks,
                fpaths["wmfod"],
                weights,
                "-act",
                fpaths["act"],
                "-out_mu",
                mu,
                "-nthreads",
                str(max(1, mrtrix_threads)),
            ]
            if fd_scale_gm:
                cmd.append("-fd_scale_gm")
            try:
                route7.run_track_cmd(cmd, logs / f"tcksift2_prod3m_plus_zero_roi_{mode_tag}_attempt{idx}.log", env, "tcksift2")
                last_error = None
                break
            except Exception as exc:
                last_error = exc
                if weights.exists():
                    weights.unlink()
        if last_error is not None:
            raise last_error

    label_rows.append({"stage": label_stage, **parc_candidate["label_stats"]})
    write_csv(probe_root / "label_survival_summary.csv", label_rows)
    route7.run_connectomes(
        sid=sid,
        probe_root=probe_root,
        label_stage=label_stage,
        parc_path=Path(parc_candidate["path"]),
        tracks=combined_tracks,
        weights=weights,
        variants=variants,
        label_rows=label_rows,
        matrix_rows=matrix_rows,
        valid_labels=valid_labels,
        env=env,
    )

    zero_labels = parse_zero_labels(seed_info)
    zero_assignment_rows: list[dict[str, Any]] = []
    for row in matrix_rows:
        if row.get("candidate") != label_stage or row.get("metric") != "fd_sum":
            continue
        assignment_file = Path(str(row.get("assignment_file") or ""))
        zero_assignment_rows.extend(
            assignment_zero_roi_summary(
                assignment_file=assignment_file,
                zero_labels=zero_labels,
                variant=str(row.get("variant") or ""),
            )
        )
    write_csv(probe_root / "zero_roi_assignment_summary.csv", zero_assignment_rows)

    best = route7.choose_best_matrix(matrix_rows, label_rows, baseline_fd)
    batch_row = route7.row_from_best(
        sid,
        "ok",
        probe_root,
        best,
        baseline_fd,
        label_rows,
        {
            "route20_action": f"zero_roi_{seed_mode}_endpoint",
            "route11_schema": SCHEMA,
            "route11_lane": LANE,
            "target_lane": LANE,
            "target_streamlines": target_streamlines,
            "dilation_iters": seed_info["dilation_iters"],
            "seed_mode": seed_mode,
            "seed_space": seed_space,
            "seeds_per_voxel": seeds_per_voxel,
            "cutoff": cutoff,
            "zero_label_count": seed_info["zero_label_count"],
            "zero_labels_attempted": seed_info.get("zero_labels_attempted", seed_info["zero_label_count"]),
            "seed_voxels": seed_info["seed_voxels"],
            "per_roi": int(bool(per_roi)),
            "per_roi_target_streamlines": per_roi_target_streamlines,
            "target_track_count": len(target_tracks_for_merge),
            "target_tracks": ";".join(str(path) for path in target_tracks_for_merge),
            "combined_tracks": str(combined_tracks),
        },
    )
    write_json(probe_root / "decision.json", {"decision": "AAL3_ROUTE20_ZERO_ROI_ENDPOINT", "best": best, "batch_row": batch_row})
    update_status(probe_root, phase="complete", completed_utc=utc(), decision="AAL3_ROUTE20_ZERO_ROI_ENDPOINT", latest=batch_row)
    return batch_row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route1-root", type=Path, default=DEFAULT_ROUTE1_ROOT)
    parser.add_argument("--route2-root", type=Path, default=None)
    parser.add_argument("--subjects", nargs="+", required=True)
    parser.add_argument("--target-streamlines", type=int, default=1_000_000)
    parser.add_argument("--per-roi", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--per-roi-target-streamlines", type=int, default=50_000)
    parser.add_argument("--per-roi-max-seeds", type=int, default=30_000_000)
    parser.add_argument("--max-zero-rois", type=int, default=0, help="0 means attempt every current zero-row ROI.")
    parser.add_argument("--dilation-iters", type=int, default=2)
    parser.add_argument("--seed-mode", choices=["gmwmi", "image", "random_per_voxel", "grid_per_voxel"], default="image")
    parser.add_argument("--seed-space", choices=["none", "fod_mask", "dwi_mask", "gmwmi_mask"], default="gmwmi_mask")
    parser.add_argument("--seeds-per-voxel", type=int, default=20)
    parser.add_argument("--cutoff", type=float, default=0.05)
    parser.add_argument("--mrtrix-threads", type=int, default=8)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--variants", nargs="*", default=["radial4", "radial6", "reverse5", "reverse10"])
    parser.add_argument("--retry", action="store_true")
    args = parser.parse_args()

    route1_root = args.route1_root
    route2_root = args.route2_root or route1_root.parent / f"{route1_root.name}_route2"
    run_root = route1_root.parent / f"{route1_root.name}{SUFFIX}"
    run_root.mkdir(parents=True, exist_ok=True)
    subjects = sorted(set(args.subjects))
    existing = read_csv(run_root / "batch_results.csv")
    if args.retry:
        results = [row for row in existing if row.get("sid") not in set(subjects)]
        pending = subjects
    else:
        finished = {row.get("sid") for row in existing if row.get("status") in {"ok", "failed"}}
        results = list(existing)
        pending = [sid for sid in subjects if sid not in finished]

    write_json(
        run_root / "run_manifest.json",
        {
            "mode": SCHEMA,
            "route1_root": str(route1_root),
            "route2_root": str(route2_root),
            "root": str(run_root),
            "probe_parent": str(QC_ROOT / PROBE_NAME),
            "subjects": subjects,
            "workers": args.workers,
            "target_streamlines": args.target_streamlines,
            "per_roi": args.per_roi,
            "per_roi_target_streamlines": args.per_roi_target_streamlines,
            "per_roi_max_seeds": args.per_roi_max_seeds,
            "max_zero_rois": args.max_zero_rois,
            "dilation_iters": args.dilation_iters,
            "seed_mode": args.seed_mode,
            "seed_space": args.seed_space,
            "seeds_per_voxel": args.seeds_per_voxel,
            "cutoff": args.cutoff,
            "mrtrix_threads": args.mrtrix_threads,
            "variants": args.variants,
            "started_utc": utc(),
        },
    )
    update_status(
        run_root,
        route11_schema=SCHEMA,
        phase="running",
        total=len(subjects),
        completed=sum(1 for row in results if row.get("status") == "ok"),
        failed=sum(1 for row in results if row.get("status") == "failed"),
        active=[],
    )

    active: set[str] = set()

    def wrapped(sid: str) -> dict[str, Any]:
        active.add(sid)
        update_status(run_root, active=sorted(active))
        try:
            return route20_subject(
                sid=sid,
                route1_root=route1_root,
                route2_root=route2_root,
                run_root=run_root,
                target_streamlines=args.target_streamlines,
                dilation_iters=args.dilation_iters,
                variants=args.variants,
                mrtrix_threads=max(1, args.mrtrix_threads),
                seed_mode=args.seed_mode,
                seed_space=args.seed_space,
                seeds_per_voxel=args.seeds_per_voxel,
                cutoff=args.cutoff,
                per_roi=args.per_roi,
                per_roi_target_streamlines=args.per_roi_target_streamlines,
                per_roi_max_seeds=args.per_roi_max_seeds,
                max_zero_rois=args.max_zero_rois,
                force_retry=args.retry,
            )
        finally:
            active.discard(sid)
            update_status(run_root, active=sorted(active))

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(wrapped, sid): sid for sid in pending}
        for future in as_completed(futures):
            sid = futures[future]
            try:
                row = future.result()
            except Exception as exc:
                row = {
                    "sid": sid,
                    "status": "failed",
                    "route11_schema": SCHEMA,
                    "route11_lane": LANE,
                    "target_lane": LANE,
                    "error": f"{type(exc).__name__}: {exc}",
                    "finished_utc": utc(),
                }
            results = [old for old in results if old.get("sid") != sid] + [row]
            write_csv(run_root / "batch_results.csv", results)
            update_status(
                run_root,
                completed=sum(1 for item in results if item.get("status") == "ok"),
                failed=sum(1 for item in results if item.get("status") == "failed"),
                latest=row,
                active=sorted(active),
            )
            refresh_outputs(route1_root, run_root)

    update_status(run_root, phase="complete", completed_utc=utc(), active=[])
    refresh_outputs(route1_root, run_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
