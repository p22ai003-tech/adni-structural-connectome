#!/usr/bin/env python3
"""QC-only pilot for the reference AAL->T1->B0 parcellation route.

This script does not touch production parcellations or connectomes.  It writes
candidate AAL_b0 images and count/fd_sum connectomes under:

  data/derivatives/qc/sc_matrix_qc/registration_parc_pilot/<tag>/<sid>/

The goal is to test whether the supervisor/reference two-step route
(`AAL/MNI -> native T1 -> B0`, nearest-neighbour labels) improves matrix
density and unexpected zero-row burden before any batch repair is attempted.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import nibabel as nib
import numpy as np

ROOT = Path("/home/ec2-user/exp")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from connectome_pipeline.bbr_bridge import _find_t1_brain, _find_t1_for_registration  # noqa: E402
from connectome_pipeline.connectome_step7 import Step7Config, _base_env, _subject_paths, _tool  # noqa: E402
from connectome_pipeline.pipeline_paths import resolve_pipeline_paths  # noqa: E402


DEFAULT_SIDS = (
    "002_S_5230_I1042952",
    "002_S_4213_I888016",
    "003_S_6067_I1511200",
)
MATRIX_METRICS = ("count", "fd_sum")
GROUP_REQUIRED = ("TRACKS_FINAL", "WEIGHTS", "B0MEAN", "T12B0", "AAL_B0")


@dataclass(frozen=True)
class Route:
    name: str
    aal_b0: Path
    generated: bool


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def run_cmd(cmd: list[str], *, env: dict[str, str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n[{now_utc()}] $ {' '.join(map(str, cmd))}\n")
        proc = subprocess.run(cmd, stdout=handle, stderr=handle, text=True, env=env, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed rc={proc.returncode}: {' '.join(map(str, cmd))}; see {log_path}")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        fields: list[str] = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    else:
        fields = ["empty"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def update_status(run_root: Path, payload: dict[str, object]) -> None:
    status = dict(payload)
    status["updated_utc"] = now_utc()
    tmp = run_root / ".reference_parc_pilot_status.json.tmp"
    out = run_root / "reference_parc_pilot_status.json"
    tmp.write_text(json.dumps(status, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(out)


def load_valid_labels(label_csv: Path) -> set[int]:
    labels: set[int] = set()
    with label_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            raw = row.get("node") or row.get("label") or row.get("index") or ""
            try:
                labels.add(int(float(raw)))
            except Exception:
                continue
    if not labels:
        raise RuntimeError(f"No valid atlas labels read from {label_csv}")
    return labels


def image_shape(path: Path) -> str:
    try:
        img = nib.load(str(path))
        return "x".join(str(v) for v in img.shape[:3])
    except Exception:
        return ""


def label_metrics(path: Path, valid_labels: set[int], mask_paths: dict[str, Path]) -> dict[str, object]:
    out: dict[str, object] = {
        "aal_b0": str(path),
        "aal_exists": int(path.exists()),
        "aal_shape": "",
        "labeled_voxels": math.nan,
        "valid_labels_present": math.nan,
        "labels_ge_50_vox": math.nan,
        "missing_valid_labels": math.nan,
        "AAL_007_voxels": math.nan,
        "AAL_008_voxels": math.nan,
        "mask_b0_overlap_fraction": math.nan,
        "mask_5tt_overlap_fraction": math.nan,
        "read_error": "",
    }
    if not path.exists():
        out["read_error"] = "missing"
        return out
    try:
        img = nib.load(str(path))
        data = np.asanyarray(img.dataobj)
        rounded = np.rint(data).astype(np.int32, copy=False)
        out["aal_shape"] = "x".join(str(v) for v in rounded.shape[:3])
        nonzero = rounded > 0
        out["labeled_voxels"] = int(nonzero.sum())
        labels, counts = np.unique(rounded[nonzero], return_counts=True)
        count_map = {int(k): int(v) for k, v in zip(labels, counts)}
        present = set(count_map).intersection(valid_labels)
        ge50 = {label for label in present if count_map.get(label, 0) >= 50}
        out["valid_labels_present"] = len(present)
        out["labels_ge_50_vox"] = len(ge50)
        out["missing_valid_labels"] = len(valid_labels.difference(present))
        out["AAL_007_voxels"] = count_map.get(7, 0)
        out["AAL_008_voxels"] = count_map.get(8, 0)
        for mask_name, mask_path in mask_paths.items():
            if not mask_path.exists():
                continue
            try:
                mask = np.asanyarray(nib.load(str(mask_path)).dataobj) > 0
                if mask.shape[:3] == nonzero.shape[:3] and nonzero.sum() > 0:
                    out[f"{mask_name}_overlap_fraction"] = float((nonzero & mask).sum() / nonzero.sum())
            except Exception:
                continue
    except Exception as exc:
        out["read_error"] = f"{type(exc).__name__}: {exc}"
    return out


def read_matrix(path: Path) -> np.ndarray:
    if not path.exists() or path.stat().st_size <= 0:
        raise FileNotFoundError(path)
    try:
        arr = np.genfromtxt(path, delimiter=",", dtype=float)
        if arr.ndim == 2 and arr.shape[0] > 1:
            return arr
    except Exception:
        pass
    arr = np.genfromtxt(path, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"matrix ndim={arr.ndim}, expected 2")
    return arr


def matrix_metrics(path: Path, valid_labels: set[int]) -> dict[str, object]:
    out: dict[str, object] = {
        "matrix_path": str(path),
        "matrix_exists": int(path.exists()),
        "matrix_size_bytes": path.stat().st_size if path.exists() else 0,
        "read_ok": 0,
        "n_rows": math.nan,
        "n_cols": math.nan,
        "finite_fraction": math.nan,
        "symmetry_max_abs": math.nan,
        "diagonal_abs_sum": math.nan,
        "nonzero_upper_edges": math.nan,
        "density": math.nan,
        "unexpected_valid_zero_rows": math.nan,
        "unexpected_valid_zero_row_labels": "",
        "read_error": "",
    }
    try:
        matrix = read_matrix(path)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError(f"shape={matrix.shape}, expected square")
        n = int(matrix.shape[0])
        clean = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
        upper = np.triu(np.ones((n, n), dtype=bool), 1)
        nonzero_upper = int((np.abs(clean) > 0)[upper].sum())
        possible_upper = int(n * (n - 1) / 2)
        row_abs = np.abs(clean).sum(axis=1)
        zero_rows = {idx + 1 for idx, value in enumerate(row_abs) if value == 0}
        unexpected = sorted(label for label in zero_rows if label in valid_labels)
        out.update(
            {
                "read_ok": 1,
                "n_rows": n,
                "n_cols": n,
                "finite_fraction": float(np.isfinite(matrix).mean()) if matrix.size else 0.0,
                "symmetry_max_abs": float(np.max(np.abs(clean - clean.T))) if clean.size else math.nan,
                "diagonal_abs_sum": float(np.abs(np.diag(clean)).sum()) if clean.size else math.nan,
                "nonzero_upper_edges": nonzero_upper,
                "density": nonzero_upper / possible_upper if possible_upper else math.nan,
                "unexpected_valid_zero_rows": len(unexpected),
                "unexpected_valid_zero_row_labels": ";".join(map(str, unexpected)),
            }
        )
    except Exception as exc:
        out["read_error"] = f"{type(exc).__name__}: {exc}"
    return out


def find_or_convert_mask(cfg: Step7Config, sid_dir: Path, mask_path: Path, name: str) -> Path:
    if not mask_path.exists():
        return mask_path
    if mask_path.suffix == ".gz" or mask_path.suffix == ".nii":
        return mask_path
    out = sid_dir / f"{name}.nii.gz"
    if out.exists():
        return out
    run_cmd([_tool(cfg, "mrconvert"), str(mask_path), str(out), "-force"], env=_base_env(cfg), log_path=sid_dir / "logs" / "mrconvert_masks.log")
    return out


def build_routes(cfg: Step7Config, sid: str, paths: dict[str, Path], sid_dir: Path) -> list[Route]:
    env = _base_env(cfg)
    logs = sid_dir / "logs"
    flirt = shutil.which("flirt", path=env["PATH"])
    convert_xfm = shutil.which("convert_xfm", path=env["PATH"])
    if not flirt or not convert_xfm:
        raise RuntimeError("FSL flirt/convert_xfm not found")
    t1_anat = cfg.deriv_root / "t1_anat"
    native_t1 = _find_t1_for_registration(t1_anat, sid)
    native_t1_brain = _find_t1_brain(t1_anat, sid) or native_t1
    mni_brain = cfg.fsl_dir / "data" / "standard" / "MNI152_T1_1mm_brain.nii.gz"
    if not mni_brain.exists():
        raise FileNotFoundError(mni_brain)

    routes = [Route("production_current", paths["AAL_B0"], generated=False)]

    # Current Step 7 direct composed route, reproduced in QC output.
    current_mni2t1 = sid_dir / "current_step7_mni2t1.mat"
    current_mni2b0 = sid_dir / "current_step7_mni2b0.mat"
    current_aal = sid_dir / "AAL_b0_current_step7_direct.nii.gz"
    run_cmd(
        [flirt, "-in", str(mni_brain), "-ref", str(paths["T1_RAS"]), "-omat", str(current_mni2t1), "-dof", "12"],
        env=env,
        log_path=logs / "current_step7_direct.log",
    )
    run_cmd(
        [convert_xfm, "-omat", str(current_mni2b0), "-concat", str(paths["T12B0"]), str(current_mni2t1)],
        env=env,
        log_path=logs / "current_step7_direct.log",
    )
    run_cmd(
        [
            flirt,
            "-in",
            str(cfg.aal_mni),
            "-ref",
            str(paths["B0MEAN"]),
            "-applyxfm",
            "-init",
            str(current_mni2b0),
            "-interp",
            "nearestneighbour",
            "-datatype",
            "int",
            "-out",
            str(current_aal),
        ],
        env=env,
        log_path=logs / "current_step7_direct.log",
    )
    routes.append(Route("current_step7_direct_reproduced", current_aal, generated=True))

    # Reference-style route using native skull-stripped T1, then existing BBR T1->B0.
    native_mni2t1 = sid_dir / "native_t1brain_mni2t1.mat"
    native_aal_t1 = sid_dir / "AAL_t1_native_t1brain.nii.gz"
    native_aal_b0 = sid_dir / "AAL_b0_native_t1brain_twostep_bbr.nii.gz"
    run_cmd(
        [
            flirt,
            "-in",
            str(cfg.aal_mni),
            "-ref",
            str(native_t1_brain),
            "-omat",
            str(native_mni2t1),
            "-interp",
            "nearestneighbour",
            "-datatype",
            "int",
            "-out",
            str(native_aal_t1),
            "-dof",
            "12",
        ],
        env=env,
        log_path=logs / "native_t1brain_twostep_bbr.log",
    )
    run_cmd(
        [
            flirt,
            "-in",
            str(native_aal_t1),
            "-ref",
            str(paths["B0MEAN"]),
            "-applyxfm",
            "-init",
            str(paths["T12B0"]),
            "-interp",
            "nearestneighbour",
            "-datatype",
            "int",
            "-out",
            str(native_aal_b0),
        ],
        env=env,
        log_path=logs / "native_t1brain_twostep_bbr.log",
    )
    routes.append(Route("native_t1brain_twostep_existing_bbr", native_aal_b0, generated=True))

    # Reference-shell style T1->B0 route: corrected native T1 to B0 using 6 DOF,
    # then apply that matrix to the native-T1 label image.
    ref_t12b0 = sid_dir / "reference_flirt6_t12b0.mat"
    ref_t1_on_b0 = sid_dir / "reference_flirt6_t1_on_b0.nii.gz"
    ref_aal_b0 = sid_dir / "AAL_b0_reference_flirt6_twostep.nii.gz"
    run_cmd(
        [flirt, "-in", str(native_t1), "-ref", str(paths["B0MEAN"]), "-omat", str(ref_t12b0), "-out", str(ref_t1_on_b0), "-dof", "6"],
        env=env,
        log_path=logs / "reference_flirt6_twostep.log",
    )
    run_cmd(
        [
            flirt,
            "-in",
            str(native_aal_t1),
            "-ref",
            str(paths["B0MEAN"]),
            "-applyxfm",
            "-init",
            str(ref_t12b0),
            "-interp",
            "nearestneighbour",
            "-datatype",
            "int",
            "-out",
            str(ref_aal_b0),
        ],
        env=env,
        log_path=logs / "reference_flirt6_twostep.log",
    )
    routes.append(Route("reference_flirt6_native_t1_twostep", ref_aal_b0, generated=True))
    return routes


def tck2connectome_for_route(
    cfg: Step7Config,
    sid: str,
    paths: dict[str, Path],
    route: Route,
    out_dir: Path,
    metric: str,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"SC_AAL_{sid}_{route.name}_{metric}.csv"
    cmd = [
        _tool(cfg, "tck2connectome"),
        str(paths["TRACKS_FINAL"]),
        str(route.aal_b0),
        str(out),
        "-symmetric",
        "-zero_diagonal",
        "-assignment_radial_search",
        str(cfg.assignment_radial_search or 4),
    ]
    if metric == "count":
        cmd += ["-out_assignments", str(out_dir / f"assignments_{route.name}.csv"), "-stat_edge", "sum"]
    elif metric == "fd_sum":
        cmd += ["-tck_weights_in", str(paths["WEIGHTS"]), "-stat_edge", "sum"]
    else:
        raise ValueError(metric)
    run_cmd(cmd, env=_base_env(cfg), log_path=out_dir / "logs" / f"tck2connectome_{route.name}_{metric}.log")
    return out


def compare_to_baseline(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    baseline_by_sid_metric: dict[tuple[str, str], dict[str, object]] = {}
    for row in rows:
        if row.get("route") == "production_current":
            baseline_by_sid_metric[(str(row["sid"]), str(row["metric"]))] = row
    out: list[dict[str, object]] = []
    for row in rows:
        base = baseline_by_sid_metric.get((str(row["sid"]), str(row["metric"])))
        merged = dict(row)
        if base:
            for key in ("density", "unexpected_valid_zero_rows", "nonzero_upper_edges"):
                try:
                    merged[f"{key}_baseline"] = float(base.get(key, math.nan))
                    merged[f"{key}_delta_vs_baseline"] = float(row.get(key, math.nan)) - float(base.get(key, math.nan))
                except Exception:
                    merged[f"{key}_baseline"] = math.nan
                    merged[f"{key}_delta_vs_baseline"] = math.nan
        out.append(merged)
    return out


def decision_rows(label_rows: list[dict[str, object]], matrix_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    label_by_sid_route = {(str(r["sid"]), str(r["route"])): r for r in label_rows}
    fd_rows = [r for r in matrix_rows if r.get("metric") == "fd_sum"]
    by_sid: dict[str, list[dict[str, object]]] = {}
    for row in fd_rows:
        by_sid.setdefault(str(row["sid"]), []).append(row)
    for sid, rows in by_sid.items():
        baseline = next((r for r in rows if r.get("route") == "production_current"), None)
        candidates = [r for r in rows if r.get("route") != "production_current" and r.get("read_ok") == 1]
        best = None
        for cand in candidates:
            if best is None:
                best = cand
                continue
            key_cand = (
                float(cand.get("unexpected_valid_zero_rows", 1e9)) * -1,
                float(cand.get("density", -1)),
                float(cand.get("nonzero_upper_edges", -1)),
            )
            key_best = (
                float(best.get("unexpected_valid_zero_rows", 1e9)) * -1,
                float(best.get("density", -1)),
                float(best.get("nonzero_upper_edges", -1)),
            )
            if key_cand > key_best:
                best = cand
        decision = "NO_CANDIDATE"
        reason = "no candidate fd_sum matrix was generated"
        if baseline and best:
            density_delta = float(best.get("density_delta_vs_baseline", math.nan))
            zero_delta = float(best.get("unexpected_valid_zero_rows_delta_vs_baseline", math.nan))
            if math.isfinite(density_delta) and math.isfinite(zero_delta) and density_delta >= 0 and zero_delta <= 0:
                decision = "CANDIDATE_NOT_WORSE"
                reason = "best candidate does not reduce density or increase unexpected valid zero rows"
                if density_delta > 0 or zero_delta < 0:
                    decision = "CANDIDATE_IMPROVES"
                    reason = "best candidate improves density and/or unexpected valid zero rows"
            else:
                decision = "KEEP_BASELINE"
                reason = "best candidate worsens density or unexpected valid zero rows"
        label_best = label_by_sid_route.get((sid, str(best.get("route")))) if best else {}
        label_base = label_by_sid_route.get((sid, "production_current"), {})
        out.append(
            {
                "sid": sid,
                "baseline_density": baseline.get("density") if baseline else math.nan,
                "baseline_unexpected_valid_zero_rows": baseline.get("unexpected_valid_zero_rows") if baseline else math.nan,
                "baseline_valid_labels_present": label_base.get("valid_labels_present", math.nan),
                "best_route": best.get("route") if best else "",
                "best_density": best.get("density") if best else math.nan,
                "best_unexpected_valid_zero_rows": best.get("unexpected_valid_zero_rows") if best else math.nan,
                "best_valid_labels_present": label_best.get("valid_labels_present", math.nan),
                "decision": decision,
                "reason": reason,
            }
        )
    return out


def process_subject(cfg: Step7Config, run_root: Path, sid: str, valid_labels: set[int], status: dict[str, object]) -> None:
    paths = _subject_paths(cfg, sid, create_dirs=False)
    sid_dir = run_root / sid
    sid_dir.mkdir(parents=True, exist_ok=True)
    missing = [key for key in GROUP_REQUIRED if not paths[key].exists()]
    if missing:
        raise RuntimeError(f"{sid}: missing required inputs: {', '.join(missing)}")
    mask_paths: dict[str, Path] = {}
    if paths["MASK_B0"].exists():
        mask_paths["mask_b0"] = paths["MASK_B0"]
    if paths["MASK_5TT_BRAIN"].exists():
        mask_paths["mask_5tt"] = find_or_convert_mask(cfg, sid_dir, paths["MASK_5TT_BRAIN"], "mask_5tt_brain")

    status.update({"current_sid": sid, "current_stage": "building parcellation routes"})
    update_status(run_root, status)
    routes = build_routes(cfg, sid, paths, sid_dir)

    label_rows: list[dict[str, object]] = []
    matrix_rows: list[dict[str, object]] = []
    for route in routes:
        status.update({"current_sid": sid, "current_stage": f"label QC: {route.name}"})
        update_status(run_root, status)
        row = {"sid": sid, "route": route.name, "generated": int(route.generated)}
        row.update(label_metrics(route.aal_b0, valid_labels, mask_paths))
        label_rows.append(row)

        for metric in MATRIX_METRICS:
            status.update({"current_sid": sid, "current_stage": f"tck2connectome {metric}: {route.name}"})
            update_status(run_root, status)
            if route.name == "production_current":
                source = paths["SC_COUNT"] if metric == "count" else paths["SC_FD_SUM"]
            else:
                source = tck2connectome_for_route(cfg, sid, paths, route, sid_dir / "matrices", metric)
            mrow = {"sid": sid, "route": route.name, "metric": metric}
            mrow.update(matrix_metrics(source, valid_labels))
            matrix_rows.append(mrow)

    write_csv(sid_dir / "parcellation_route_metrics.csv", label_rows)
    write_csv(sid_dir / "matrix_route_metrics.csv", compare_to_baseline(matrix_rows))
    write_csv(sid_dir / "pilot_decision_summary.csv", decision_rows(label_rows, compare_to_baseline(matrix_rows)))


def collect_run_outputs(run_root: Path) -> None:
    label_rows: list[dict[str, object]] = []
    matrix_rows: list[dict[str, object]] = []
    decision: list[dict[str, object]] = []
    for sid_dir in sorted(p for p in run_root.iterdir() if p.is_dir()):
        for name, bucket in (
            ("parcellation_route_metrics.csv", label_rows),
            ("matrix_route_metrics.csv", matrix_rows),
            ("pilot_decision_summary.csv", decision),
        ):
            path = sid_dir / name
            if not path.exists():
                continue
            with path.open(newline="", encoding="utf-8") as handle:
                bucket.extend(csv.DictReader(handle))
    write_csv(run_root / "reference_parcellation_pilot_labels.csv", label_rows)
    write_csv(run_root / "reference_parcellation_pilot_matrices.csv", matrix_rows)
    write_csv(run_root / "reference_parcellation_pilot_summary.csv", decision)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sid", action="append", dest="sids", help="Pilot subject ID. Repeatable.")
    parser.add_argument("--tag", default=time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()))
    parser.add_argument(
        "--qc-root",
        type=Path,
        default=ROOT / "data/derivatives/qc/sc_matrix_qc/registration_parc_pilot",
    )
    parser.add_argument("--project-root", type=Path, default=ROOT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = resolve_pipeline_paths(args.project_root, create_layout=True)
    cfg = Step7Config(deriv_root=paths.deriv_root, aal_mni=paths.aal_mni)
    valid_labels = load_valid_labels(paths.aal_root / "AAL3_labels.csv")
    sids = args.sids or list(DEFAULT_SIDS)
    run_root = args.qc_root / args.tag
    run_root.mkdir(parents=True, exist_ok=True)
    status: dict[str, object] = {
        "tag": args.tag,
        "run_root": str(run_root),
        "total_subjects": len(sids),
        "completed_subjects": 0,
        "failed_subjects": 0,
        "current_sid": "",
        "current_stage": "starting",
        "started_utc": now_utc(),
        "done": False,
    }
    update_status(run_root, status)
    failures: list[dict[str, object]] = []
    for idx, sid in enumerate(sids, start=1):
        status.update({"current_sid": sid, "current_stage": "starting subject", "subject_index": idx})
        update_status(run_root, status)
        try:
            process_subject(cfg, run_root, sid, valid_labels, status)
            status["completed_subjects"] = int(status["completed_subjects"]) + 1
        except Exception as exc:
            failures.append({"sid": sid, "error": f"{type(exc).__name__}: {exc}"})
            status["failed_subjects"] = int(status["failed_subjects"]) + 1
        finally:
            write_csv(run_root / "reference_parcellation_pilot_failures.csv", failures)
            collect_run_outputs(run_root)
            update_status(run_root, status)
    status.update({"current_sid": "", "current_stage": "complete", "done": True, "completed_utc": now_utc()})
    collect_run_outputs(run_root)
    update_status(run_root, status)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
