#!/usr/bin/env python3
"""Scratch-only AAL3 label-preserving dilation rescue.

This route expands existing AAL3 labels locally into nearby unlabeled voxels
before tck2connectome. It does not invent missing labels and never overwrites
production outputs. The intended target is high-label-survival subjects with
persistent zero rows after ordinary assignment variants.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
from scipy import ndimage

from scripts.scforge.live.run_sc_aal3_source_contract_probe import (
    DERIV,
    MRTRIX,
    TCK2CONNECTOME,
    aal3_valid_labels,
    discover_inputs,
    label_stats,
    matrix_stats,
    mrtrix_thread_args,
    run,
    write_csv,
    write_json,
)

OUT_PARENT = DERIV / "qc" / "sc_matrix_qc" / "aal3_route4_label_rescue_probe"


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def connectome_timeout_sec() -> int:
    for key in ("AAL3_CONNECTOME_TIMEOUT_SEC", "ROUTE4_CONNECTOME_TIMEOUT_SEC"):
        value = os.environ.get(key, "").strip()
        if value:
            try:
                return max(60, int(float(value)))
            except ValueError:
                pass
    return 45 * 60


def run_connectome_cmd(cmd: list[str | Path], log_path: Path, env: dict[str, str]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    str_cmd = [str(x) for x in cmd]
    timeout_sec = connectome_timeout_sec()
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


def matrix_meets_early_stop(stats: dict[str, Any], matrix_rows: list[dict[str, Any]]) -> bool:
    prod = next((row for row in matrix_rows if row.get("candidate") == "production_AAL3"), {})
    baseline_density = fnum(prod.get("valid_density"), 0.0)
    baseline_zero = int(fnum(prod.get("valid_zero_rows"), 9999))
    return (
        int(fnum(stats.get("read_ok"), 0)) == 1
        and fnum(stats.get("finite_fraction"), 0.0) >= 1.0
        and fnum(stats.get("symmetry_max_abs"), 1.0) == 0.0
        and fnum(stats.get("diagonal_abs_sum"), 1.0) == 0.0
        and fnum(stats.get("valid_density"), 0.0) >= 0.10
        and fnum(stats.get("valid_density"), 0.0) >= baseline_density
        and int(fnum(stats.get("valid_zero_rows"), 9999)) <= 15
        and int(fnum(stats.get("valid_zero_rows"), 9999)) <= baseline_zero
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size <= 0:
        return []
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def update_status(root: Path, **updates: Any) -> None:
    status = read_json(root / "status.json")
    status.update(updates)
    status["last_update_utc"] = utc()
    write_json(root / "status.json", status)


def fnum(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except Exception:
        return default


def choose_base_parcellation(sid: str, route3_root: Path) -> tuple[str, Path]:
    for row in read_csv(route3_root / "batch_results.csv"):
        if row.get("sid") != sid:
            continue
        probe = Path(row.get("probe_root") or "")
        decision = read_json(probe / "decision.json")
        best = decision.get("best") or {}
        matrix = Path(best.get("matrix") or "")
        label_stage = best.get("label_stage") or row.get("best_label_stage") or ""
        if matrix.exists() and label_stage:
            for label_row in read_csv(probe / "label_survival_summary.csv"):
                if label_row.get("stage") == label_stage:
                    # The exact image path is recorded in the label row.
                    path = Path(label_row.get("path") or "")
                    if path.exists():
                        return f"route4_from_{label_stage}", path
        if probe.exists():
            # Fallback: search for the named label-stage image under likely QC dirs.
            for path in sorted(probe.glob("**/*AAL3_b0.nii.gz")):
                return "route4_from_route3_probe", path

    inputs = discover_inputs(sid)
    if inputs.get("production_aal_b0"):
        return "route4_from_production", inputs["production_aal_b0"]
    raise FileNotFoundError(f"{sid}: no base parcellation found for route4")


def b0_mask_for_subject(sid: str) -> Path | None:
    path = DERIV / "dwi_t1_bbr" / f"{sid}_mask_on_b0_bbr.nii.gz"
    return path if path.exists() else None


def dilate_labels(base_path: Path, out_path: Path, radius_mm: float, mask_path: Path | None) -> Path:
    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path
    img = nib.load(str(base_path))
    data = np.rint(np.asanyarray(img.dataobj)).astype(np.int32)
    labels = data > 0
    zero = ~labels
    zooms = img.header.get_zooms()[:3]
    dist, nearest = ndimage.distance_transform_edt(zero, sampling=zooms, return_indices=True)
    fill = zero & np.isfinite(dist) & (dist <= float(radius_mm))
    if mask_path and mask_path.exists():
        try:
            mask = np.asanyarray(nib.load(str(mask_path)).dataobj) > 0
            if mask.shape == data.shape:
                fill &= mask
        except Exception:
            pass
    out = data.copy()
    out[fill] = data[tuple(axis[fill] for axis in nearest)]
    out_img = nib.Nifti1Image(out.astype(np.int16), img.affine, img.header)
    out_img.header.set_data_dtype(np.int16)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(out_img, str(out_path))
    return out_path


def assignment_args(name: str) -> list[str]:
    table = {
        "forward80": ["-assignment_forward_search", "80"],
        "forward120": ["-assignment_forward_search", "120"],
        "radial8": ["-assignment_radial_search", "8"],
    }
    if name not in table:
        raise ValueError(f"Unknown assignment variant: {name}")
    return table[name]


def best_candidate(matrix_rows: list[dict[str, Any]], label_rows: list[dict[str, Any]], baseline_fd: dict[str, Any]) -> dict[str, Any]:
    labels = {r.get("stage"): fnum(r.get("n_labels"), 0) for r in label_rows}
    base_density = fnum(baseline_fd.get("valid_density"), 0)
    base_zero = fnum(baseline_fd.get("valid_zero_rows"), 9999)
    candidates = [
        row for row in matrix_rows
        if row.get("candidate") != "production_AAL3"
        and row.get("metric") == "fd_sum"
        and int(fnum(row.get("read_ok"), 0)) == 1
    ]

    def key(row: dict[str, Any]) -> tuple[float, float, float, float]:
        n_labels = labels.get(row.get("label_stage"), 0)
        density = fnum(row.get("valid_density"), -1)
        zero = fnum(row.get("valid_zero_rows"), 9999)
        safe = 1.0 if density >= base_density and zero <= base_zero else 0.0
        return (safe, n_labels, -zero, density)

    candidates.sort(key=key, reverse=True)
    return candidates[0] if candidates else {}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sid", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--route3-root", type=Path, required=True)
    parser.add_argument("--radii-mm", nargs="*", type=float, default=[2.0, 4.0, 6.0])
    parser.add_argument("--variants", nargs="*", default=["forward80", "forward120", "radial8"])
    args = parser.parse_args()

    env = os.environ.copy()
    env["PATH"] = f"{MRTRIX}:" + env.get("PATH", "")
    run_root = OUT_PARENT / args.tag
    run_root.mkdir(parents=True, exist_ok=True)
    (OUT_PARENT / "latest_tag.txt").write_text(args.tag, encoding="utf-8")
    if (run_root / "decision.json").exists():
        print(run_root)
        return 0
    write_json(run_root / "status.json", {"sid": args.sid, "tag": args.tag, "phase": "starting", "started_utc": utc(), "variants": args.variants})

    inputs = discover_inputs(args.sid)
    missing = [key for key in ("tracks", "weights", "production_fd_sum") if not inputs.get(key)]
    if missing:
        raise FileNotFoundError(f"Missing required inputs: {missing}")
    valid_labels = aal3_valid_labels()
    base_name, base_path = choose_base_parcellation(args.sid, args.route3_root)
    mask_path = b0_mask_for_subject(args.sid)

    label_rows: list[dict[str, Any]] = []
    matrix_rows: list[dict[str, Any]] = [{"candidate": "production_AAL3", "variant": "production", "metric": "fd_sum", **matrix_stats(inputs["production_fd_sum"], valid_labels)}]
    if inputs.get("production_aal_b0"):
        label_rows.append({"stage": "production_AAL_b0", **label_stats(inputs["production_aal_b0"], valid_labels)})
    label_rows.append({"stage": base_name, **label_stats(base_path, valid_labels)})

    parc = run_root / "parc"
    conn = run_root / "connectomes"
    logs = run_root / "logs"
    conn.mkdir(parents=True, exist_ok=True)
    dilated: list[tuple[str, Path]] = []
    for radius in args.radii_mm:
        stage = f"route4_dilate_{str(radius).replace('.', 'p')}mm"
        out_path = parc / f"{stage}_AAL3_b0.nii.gz"
        update_status(run_root, phase="dilate_labels", active_radius_mm=radius)
        dilated_path = dilate_labels(base_path, out_path, radius, mask_path)
        label_rows.append({"stage": stage, **label_stats(dilated_path, valid_labels)})
        dilated.append((stage, dilated_path))
    write_csv(run_root / "label_survival_summary.csv", label_rows)
    write_csv(run_root / "matrix_candidate_summary.csv", matrix_rows)

    total_jobs = len(dilated) * len(args.variants)
    done = 0
    for stage, aal_b0 in dilated:
        for variant in args.variants:
            update_status(run_root, phase="connectome_candidates", active_parcellation=stage, active_variant=variant, connectome_jobs_done=done, connectome_jobs_total=total_jobs)
            out_csv = conn / f"SC_AAL3_{args.sid}_{stage}__{variant}__fd_sum.csv"
            assign_csv = conn / f"assignments_{stage}__{variant}__fd_sum.csv"
            stats = matrix_stats(out_csv, valid_labels)
            if int(fnum(stats.get("read_ok"), 0)) != 1 or not assign_csv.exists() or assign_csv.stat().st_size <= 0:
                cmd: list[str | Path] = [
                    TCK2CONNECTOME,
                    inputs["tracks"],
                    aal_b0,
                    out_csv,
                    "-symmetric",
                    "-zero_diagonal",
                    *mrtrix_thread_args(env),
                    *assignment_args(variant),
                    "-out_assignments",
                    assign_csv,
                    "-stat_edge",
                    "sum",
                    "-tck_weights_in",
                    str(inputs["weights"]),
                ]
                try:
                    run_connectome_cmd(cmd, logs / f"tck2connectome_{stage}__{variant}__fd_sum.log", env)
                    stats = matrix_stats(out_csv, valid_labels)
                except Exception as exc:
                    stats = matrix_stats(out_csv, valid_labels)
                    stats["error"] = f"{type(exc).__name__}: {exc}"
            done += 1
            row = {"candidate": stage, "label_stage": stage, "variant": variant, "metric": "fd_sum", "assignment_file": str(assign_csv), **stats}
            matrix_rows.append(row)
            write_csv(run_root / "matrix_candidate_summary.csv", matrix_rows)
            if matrix_meets_early_stop(row, matrix_rows):
                update_status(run_root, decision=f"early_stop_numeric_candidate_{variant}")
                break

    best = best_candidate(matrix_rows, label_rows, matrix_rows[0])
    label_stage = str(best.get("label_stage") or "")
    best_label = next((r for r in label_rows if r.get("stage") == label_stage), {})
    baseline_density = fnum(matrix_rows[0].get("valid_density"), 0)
    best_density = fnum(best.get("valid_density"), 0)
    baseline_zero = int(fnum(matrix_rows[0].get("valid_zero_rows"), 9999))
    best_zero = int(fnum(best.get("valid_zero_rows"), 9999))
    decision = {
        "decision": "AAL3_ROUTE4_LABEL_RESCUE_CANDIDATE" if best else "AAL3_ROUTE4_LABEL_RESCUE_NO_CANDIDATE",
        "best": best,
        "baseline_fd_sum": matrix_rows[0],
        "base_parcellation": str(base_path),
        "source_label_stage": label_stage,
        "source_labels": best_label.get("n_labels", ""),
        "source_label_survival": best_label.get("expected_label_survival_fraction", ""),
        "baseline_density": baseline_density,
        "best_density": best_density,
        "density_delta": best_density - baseline_density,
        "baseline_zero_rows": baseline_zero,
        "best_zero_rows": best_zero,
        "zero_rows_delta": best_zero - baseline_zero,
    }
    write_json(run_root / "decision.json", decision)
    update_status(run_root, phase="complete", completed_utc=utc(), decision=decision["decision"], connectome_jobs_done=total_jobs, connectome_jobs_total=total_jobs)
    print(run_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
