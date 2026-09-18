#!/usr/bin/env python3
"""Scratch-only AAL3 source-contract probe for sparse SC matrices.

This replays the source contract that produced a healthy AAL116 scratch
connectome, but uses the thesis AAL3 atlas:

    AAL3/MNI -> MNI brain -> true native T1 -> eddy volume-0 B0

The script writes only under qc/sc_matrix_qc/aal3_source_contract_probe and
never overwrites production derivatives.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np


ROOT = Path("/home/ec2-user/exp")
DERIV = ROOT / "data" / "derivatives"
QC_ROOT = DERIV / "qc" / "sc_matrix_qc"
OUT_PARENT = QC_ROOT / "aal3_source_contract_probe"

AAL3 = ROOT / "atlas" / "AAL" / "AAL3v1_1mm.nii.gz"
AAL3_LABELS = ROOT / "atlas" / "AAL" / "AAL3_labels.csv"

FSLDIR = Path(os.environ.get("FSLDIR", "/home/ec2-user/fsl"))
FLIRT = FSLDIR / "bin" / "flirt"
MNI_BRAIN = FSLDIR / "data" / "standard" / "MNI152_T1_1mm_brain.nii.gz"
MRTRIX = Path("/home/ec2-user/mrtrix3/bin")
MRCONVERT = MRTRIX / "mrconvert"
TCK2CONNECTOME = MRTRIX / "tck2connectome"


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def stamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def update_status(root: Path, **updates: Any) -> None:
    status = read_json(root / "status.json")
    status.update(updates)
    status["last_update_utc"] = utc()
    write_json(root / "status.json", status)


def command_timeout_sec(cmd: list[str], env: dict[str, str]) -> int | None:
    name = Path(cmd[0]).name if cmd else ""
    if name == "tck2connectome":
        raw = str(env.get("AAL3_CONNECTOME_TIMEOUT_SEC", "")).strip()
        if not raw:
            raw = str(env.get("SCFORGE_CONNECTOME_TIMEOUT_SEC", "")).strip()
        default = 0
    elif name == "fnirt":
        raw = str(env.get("AAL3_FNIRT_TIMEOUT_SEC", "")).strip()
        if not raw:
            raw = str(env.get("SCFORGE_FNIRT_TIMEOUT_SEC", "")).strip()
        default = 2 * 60 * 60
    else:
        return None
    try:
        timeout = int(float(raw)) if raw else default
    except Exception:
        timeout = 0
    return timeout if timeout > 0 else None


def run(cmd: list[str | Path], log_path: Path, env: dict[str, str]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    str_cmd = [str(x) for x in cmd]
    timeout = command_timeout_sec(str_cmd, env)
    with log_path.open("a", encoding="utf-8", errors="replace") as handle:
        handle.write(f"[{utc()}] $ {' '.join(str_cmd)}\n")
        handle.flush()
        try:
            proc = subprocess.run(
                str_cmd,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            handle.write(f"[{utc()}] timeout_after_sec={timeout}\n\n")
            raise RuntimeError(f"command timed out after {timeout}s; see {log_path}") from exc
        handle.write(f"[{utc()}] rc={proc.returncode}\n\n")
    if proc.returncode != 0:
        raise RuntimeError(f"command failed rc={proc.returncode}; see {log_path}")


def mrtrix_thread_args(env: dict[str, str]) -> list[str]:
    value = str(env.get("MRTRIX_NTHREADS", "")).strip()
    if value and value != "0":
        return ["-nthreads", value]
    return []


def first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists() and path.stat().st_size > 0:
            return path
    return None


def subject_short(sid: str) -> str:
    return sid.split("_I", 1)[0]


def discover_inputs(sid: str) -> dict[str, Any]:
    subj = subject_short(sid)
    return {
        "eddy": first_existing([DERIV / "eddy" / f"{sid}_preproc.mif", Path("/data/derivatives/eddy") / f"{sid}_preproc.mif"]),
        "t1_from_dicom": sorted((DERIV / "t1_anat").glob(f"t1_from_dicom_{subj}_*.nii.gz")),
        "corrected_t1": sorted((DERIV / "t1_anat").glob(f"corrected_T1_{subj}_*.nii.gz")),
        "tracks": first_existing([DERIV / "tracks" / sid / "tracks_final_3000k.tck", Path("/data/derivatives/tracks") / sid / "tracks_final_3000k.tck"]),
        "weights": first_existing([DERIV / "tracks" / sid / "sift_weights.txt", Path("/data/derivatives/tracks") / sid / "sift_weights.txt"]),
        "production_aal_b0": first_existing([DERIV / "parc" / sid / "AAL_b0.nii.gz", Path("/data/derivatives/parc") / sid / "AAL_b0.nii.gz"]),
        "production_count": first_existing([DERIV / "connectomes" / f"SC_AAL_{sid}_count.csv", Path("/data/derivatives/connectomes") / f"SC_AAL_{sid}_count.csv"]),
        "production_fd_sum": first_existing([DERIV / "connectomes" / f"SC_AAL_{sid}_fd_sum.csv", Path("/data/derivatives/connectomes") / f"SC_AAL_{sid}_fd_sum.csv"]),
    }


def aal3_valid_labels() -> list[int]:
    with AAL3_LABELS.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    labels = sorted({int(float(row["atlas_value"])) for row in rows if row.get("atlas_value")})
    if len(labels) < 150:
        raise RuntimeError(f"Too few AAL3 labels read from {AAL3_LABELS}: {len(labels)}")
    return labels


def labels_in_image(path: Path) -> list[int]:
    data = np.asanyarray(nib.load(str(path)).dataobj)
    vals = np.unique(np.rint(data[np.isfinite(data)]).astype(np.int64))
    return sorted(int(v) for v in vals if int(v) > 0)


def label_stats(path: Path, expected_labels: list[int]) -> dict[str, Any]:
    out: dict[str, Any] = {"path": str(path), "exists": int(path.exists()), "read_ok": 0}
    if not path.exists() or path.stat().st_size <= 0:
        out["error"] = "missing_or_empty"
        return out
    try:
        img = nib.load(str(path))
        data = np.rint(np.asanyarray(img.dataobj)).astype(np.int64)
        labels, counts = np.unique(data, return_counts=True)
        label_counts = {int(v): int(c) for v, c in zip(labels, counts) if int(v) > 0}
        present = set(label_counts)
        missing = [label for label in expected_labels if label not in present]
        out.update(
            {
                "read_ok": 1,
                "shape": "x".join(str(x) for x in img.shape[:3]),
                "nonzero_voxels": int((data > 0).sum()),
                "n_labels": len(present),
                "expected_label_count": len(expected_labels),
                "missing_expected_count": len(missing),
                "expected_label_survival_fraction": float((len(expected_labels) - len(missing)) / max(1, len(expected_labels))),
                "tiny_labels_lt10vox": sum(1 for label in expected_labels if 0 < label_counts.get(label, 0) < 10),
                "tiny_labels_lt50vox": sum(1 for label in expected_labels if 0 < label_counts.get(label, 0) < 50),
                "aal_007_voxels": int(label_counts.get(7, 0)),
                "aal_008_voxels": int(label_counts.get(8, 0)),
                "missing_expected_first20": " ".join(str(x) for x in missing[:20]),
            }
        )
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def matrix_array(path: Path) -> np.ndarray:
    try:
        arr = np.genfromtxt(path, delimiter=",", dtype=float)
        if arr.ndim == 2 and arr.shape[0] > 1 and arr.shape[1] > 1:
            return arr
    except Exception:
        pass
    return np.loadtxt(path, dtype=float)


def matrix_stats(path: Path, valid_labels: list[int], expected_n: int = 170) -> dict[str, Any]:
    out: dict[str, Any] = {
        "matrix": str(path),
        "exists": int(path.exists()),
        "read_ok": 0,
        "n": math.nan,
        "density": math.nan,
        "valid_density": math.nan,
        "zero_rows": math.nan,
        "valid_zero_rows": math.nan,
        "nonzero_upper_edges": math.nan,
        "valid_nonzero_upper_edges": math.nan,
        "finite_fraction": math.nan,
        "symmetry_max_abs": math.nan,
        "diagonal_abs_sum": math.nan,
        "error": "",
    }
    if not path.exists() or path.stat().st_size <= 0:
        out["error"] = "missing_or_empty"
        return out
    try:
        mat = matrix_array(path)
        if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
            raise ValueError(f"not square: {mat.shape}")
        n = int(mat.shape[0])
        clean = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)
        upper = np.triu(np.ones((n, n), dtype=bool), 1)
        valid_idx = np.array([label - 1 for label in valid_labels if 1 <= label <= n], dtype=int)
        valid = clean[np.ix_(valid_idx, valid_idx)] if valid_idx.size else np.empty((0, 0))
        valid_upper = np.triu(np.ones(valid.shape, dtype=bool), 1) if valid.size else np.empty(valid.shape, dtype=bool)
        row_abs = np.abs(clean).sum(axis=1)
        valid_row_abs = np.abs(valid).sum(axis=1) if valid.size else np.array([])
        out.update(
            {
                "read_ok": 1,
                "n": n,
                "density": float((np.abs(clean) > 0)[upper].sum() / max(1, int(upper.sum()))),
                "valid_density": float((np.abs(valid) > 0)[valid_upper].sum() / max(1, int(valid_upper.sum()))) if valid.size else math.nan,
                "zero_rows": int((row_abs == 0).sum()),
                "valid_zero_rows": int((valid_row_abs == 0).sum()) if valid.size else math.nan,
                "nonzero_upper_edges": int((np.abs(clean) > 0)[upper].sum()),
                "valid_nonzero_upper_edges": int((np.abs(valid) > 0)[valid_upper].sum()) if valid.size else math.nan,
                "finite_fraction": float(np.isfinite(mat).mean()),
                "symmetry_max_abs": float(np.max(np.abs(clean - clean.T))) if clean.size else math.nan,
                "diagonal_abs_sum": float(np.abs(np.diag(clean)).sum()) if clean.size else math.nan,
            }
        )
        if n != expected_n:
            out["error"] = f"unexpected_shape_expected_{expected_n}"
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def generate_eddy_vol0(sid: str, run_root: Path, eddy: Path, env: dict[str, str]) -> Path:
    out = run_root / "b0_candidates" / f"{sid}_eddy_vol0.nii.gz"
    out.parent.mkdir(parents=True, exist_ok=True)
    if not out.exists():
        run([MRCONVERT, eddy, out, "-coord", "3", "0", "-quiet", "-force"], run_root / "logs" / "eddy_vol0.log", env)
    return out


def classify(best: dict[str, Any], baseline: dict[str, Any]) -> str:
    if not best or int(float(best.get("read_ok", 0) or 0)) != 1:
        return "AAL3_ROUTE_FAILED"
    valid_density = float(best.get("valid_density", 0) or 0)
    valid_zero_rows = int(float(best.get("valid_zero_rows", 9999) or 9999))
    base_density = float(baseline.get("valid_density", 0) or 0) if baseline else 0.0
    base_zero = int(float(baseline.get("valid_zero_rows", 9999) or 9999)) if baseline else 9999
    if valid_density >= 0.60 and valid_zero_rows <= 10:
        return "AAL3_SUPERVISOR_CONTRACT_STRONG"
    if valid_density > base_density and valid_zero_rows < base_zero:
        return "AAL3_SUPERVISOR_CONTRACT_IMPROVES"
    if valid_density > base_density:
        return "AAL3_SUPERVISOR_CONTRACT_DENSITY_ONLY"
    return "AAL3_ROUTE_NOT_BETTER_THAN_BASELINE"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sid", default="003_S_0908_I1249292")
    parser.add_argument("--tag", default=f"aal3_source_contract_{stamp()}")
    parser.add_argument("--variants", nargs="*", default=["default", "forward80"])
    parser.add_argument(
        "--metrics",
        nargs="*",
        default=["count", "fd_sum", "fd_invnodevol"],
        choices=["count", "fd_sum", "fd_invnodevol"],
        help="Connectome metrics to generate for each assignment variant.",
    )
    parser.add_argument(
        "--force-connectomes-on-low-survival",
        action="store_true",
        help="Diagnostic mode: run assignment/connectome variants even when the AAL3 B0 label-survival screen is poor.",
    )
    args = parser.parse_args()

    for tool in (FLIRT, MRCONVERT, TCK2CONNECTOME, AAL3, AAL3_LABELS, MNI_BRAIN):
        if not Path(tool).exists():
            raise FileNotFoundError(tool)

    env = os.environ.copy()
    env["FSLDIR"] = str(FSLDIR)
    env["PATH"] = f"{MRTRIX}:{FSLDIR / 'bin'}:" + env.get("PATH", "")

    run_root = OUT_PARENT / args.tag
    run_root.mkdir(parents=True, exist_ok=True)
    (OUT_PARENT / "latest_tag.txt").write_text(args.tag, encoding="utf-8")
    existing_decision = run_root / "decision.json"
    if existing_decision.exists() and existing_decision.stat().st_size > 0:
        print(run_root)
        return 0

    write_json(
        run_root / "status.json",
        {
            "tag": args.tag,
            "sid": args.sid,
            "root": str(run_root),
            "phase": "starting",
            "started_utc": utc(),
            "variants": args.variants,
        },
    )

    inputs = discover_inputs(args.sid)
    input_report = {k: ([str(x) for x in v] if isinstance(v, list) else str(v) if v else "") for k, v in inputs.items()}
    write_json(run_root / "input_report.json", input_report)
    required = ["eddy", "tracks", "weights"]
    missing = [key for key in required if not inputs.get(key)]
    if missing:
        raise FileNotFoundError(f"Missing required inputs: {missing}")
    if not inputs.get("t1_from_dicom") or not inputs.get("corrected_t1"):
        raise FileNotFoundError("Need t1_from_dicom and corrected_T1 for the replayed source contract.")

    valid_labels = aal3_valid_labels()
    image_labels = labels_in_image(AAL3)
    if set(valid_labels) != set(image_labels):
        raise RuntimeError(f"AAL3 image/CSV label mismatch: csv={len(valid_labels)} image={len(image_labels)}")

    update_status(run_root, phase="baseline")
    label_rows: list[dict[str, Any]] = [{"stage": "source_AAL3", **label_stats(AAL3, valid_labels)}]
    if inputs.get("production_aal_b0"):
        label_rows.append({"stage": "production_AAL_b0", **label_stats(inputs["production_aal_b0"], valid_labels)})
    matrix_rows: list[dict[str, Any]] = []
    baseline_fd = {}
    for metric, path in (("production_count", inputs.get("production_count")), ("production_fd_sum", inputs.get("production_fd_sum"))):
        if path:
            stats = {"candidate": "production_AAL3", "variant": "production", "metric": metric.replace("production_", ""), **matrix_stats(path, valid_labels)}
            matrix_rows.append(stats)
            if metric == "production_fd_sum":
                baseline_fd = stats
    write_csv(run_root / "label_survival_summary.csv", label_rows)
    write_csv(run_root / "matrix_candidate_summary.csv", matrix_rows)

    logs = run_root / "logs"
    parc = run_root / "parc"
    conn = run_root / "connectomes"
    parc.mkdir(exist_ok=True)
    conn.mkdir(exist_ok=True)

    update_status(run_root, phase="b0_candidate")
    eddy_vol0 = generate_eddy_vol0(args.sid, run_root, inputs["eddy"], env)

    update_status(run_root, phase="aal_to_mni")
    aal2mni = parc / "AAL3_AAL2MNI.nii.gz"
    aal2mni_mat = parc / "AAL3_AAL2MNI.mat"
    run(
        [
            FLIRT,
            "-in",
            AAL3,
            "-ref",
            MNI_BRAIN,
            "-dof",
            "6",
            "-omat",
            aal2mni_mat,
            "-interp",
            "nearestneighbour",
            "-datatype",
            "int",
            "-out",
            aal2mni,
        ],
        logs / "aal3_aal2mni.log",
        env,
    )
    label_rows.append({"stage": "AAL3_AAL2MNI", **label_stats(aal2mni, valid_labels)})
    write_csv(run_root / "label_survival_summary.csv", label_rows)

    t1_ref = Path(inputs["t1_from_dicom"][0])
    t1_in = Path(inputs["corrected_t1"][0])
    update_status(run_root, phase="aal_to_true_native_t1", t1_ref=str(t1_ref), t1_input=str(t1_in), b0=str(eddy_vol0))
    aal_t1 = parc / "AAL3_MNI_to_t1_from_dicom.nii.gz"
    aal_t1_mat = parc / "AAL3_MNI_to_t1_from_dicom.mat"
    run(
        [
            FLIRT,
            "-in",
            aal2mni,
            "-ref",
            t1_ref,
            "-omat",
            aal_t1_mat,
            "-interp",
            "nearestneighbour",
            "-datatype",
            "int",
            "-out",
            aal_t1,
            "-dof",
            "12",
        ],
        logs / "aal3_mni_to_t1_from_dicom.log",
        env,
    )
    label_rows.append({"stage": "AAL3_true_native_T1", **label_stats(aal_t1, valid_labels)})
    write_csv(run_root / "label_survival_summary.csv", label_rows)

    update_status(run_root, phase="t1_to_b0")
    t12b0 = parc / "corrected_T1_to_eddy_vol0.mat"
    t1_on_b0 = parc / "corrected_T1_on_eddy_vol0.nii.gz"
    run(
        [FLIRT, "-in", t1_in, "-ref", eddy_vol0, "-omat", t12b0, "-out", t1_on_b0, "-dof", "6"],
        logs / "corrected_t1_to_eddy_vol0.log",
        env,
    )

    update_status(run_root, phase="aal_t1_to_b0")
    aal_b0 = parc / "AAL3_source_contract_B0.nii.gz"
    run(
        [
            FLIRT,
            "-in",
            aal_t1,
            "-ref",
            eddy_vol0,
            "-applyxfm",
            "-init",
            t12b0,
            "-interp",
            "nearestneighbour",
            "-datatype",
            "int",
            "-out",
            aal_b0,
        ],
        logs / "aal3_t1_to_eddy_vol0.log",
        env,
    )
    b0_label_stats = label_stats(aal_b0, valid_labels)
    label_rows.append({"stage": "AAL3_source_contract_B0", **b0_label_stats})
    write_csv(run_root / "label_survival_summary.csv", label_rows)

    b0_survival = float(b0_label_stats.get("expected_label_survival_fraction", 0) or 0)
    aal007 = int(float(b0_label_stats.get("aal_007_voxels", 0) or 0))
    aal008 = int(float(b0_label_stats.get("aal_008_voxels", 0) or 0))
    if (b0_survival < 0.50 or aal007 == 0 or aal008 == 0) and not args.force_connectomes_on_low_survival:
        decision = "AAL3_ROUTE_LABEL_SURVIVAL_FAILED"
        lines = [
            "# AAL3 Source Contract Probe Decision",
            "",
            f"Generated: {utc()}",
            "",
            "Scratch-only: no production outputs were overwritten.",
            "",
            "## Tested contract",
            "",
            "- Atlas: `AAL3v1_1mm.nii.gz`",
            "- Route: `AAL3/MNI -> MNI brain -> t1_from_dicom -> eddy_vol0`",
            "- T1-to-B0 moving image: `corrected_T1`",
            "- Label interpolation: `nearestneighbour`, integer labels",
            "",
            "## Early stop reason",
            "",
            "The replayed AAL3 parcellation failed before connectome assignment.",
            f"- B0 label survival fraction: `{b0_survival:.4f}`",
            f"- AAL_007 voxels in B0: `{aal007}`",
            f"- AAL_008 voxels in B0: `{aal008}`",
            "",
            "Assignment/radial-search variants were skipped because missing B0 labels cannot be repaired by `tck2connectome`.",
            "",
            "## Decision",
            "",
            f"`{decision}`",
            "",
        ]
        (run_root / "decision.md").write_text("\n".join(lines), encoding="utf-8")
        write_json(
            run_root / "decision.json",
            {
                "decision": decision,
                "best": {},
                "baseline_fd_sum": baseline_fd,
                "b0_label_survival": b0_label_stats,
            },
        )
        update_status(
            run_root,
            phase="complete",
            completed_utc=utc(),
            active_variant="",
            active_metric="",
            connectome_jobs_done=0,
            connectome_jobs_total=0,
            decision=decision,
        )
        print(run_root)
        return 0

    assignment_args = {
        "default": [],
        "radial4": ["-assignment_radial_search", "4"],
        "radial8": ["-assignment_radial_search", "8"],
        "forward40": ["-assignment_forward_search", "40"],
        "forward80": ["-assignment_forward_search", "80"],
    }
    all_metrics = {
        "count": [],
        "fd_sum": ["-tck_weights_in", str(inputs["weights"])],
        "fd_invnodevol": ["-scale_invnodevol", "-tck_weights_in", str(inputs["weights"])],
    }
    metrics = {name: all_metrics[name] for name in args.metrics}
    total_jobs = len(args.variants) * len(metrics)
    done = 0
    for variant in args.variants:
        extra = assignment_args.get(variant)
        if extra is None:
            raise ValueError(f"Unknown assignment variant: {variant}")
        for metric, scale_args in metrics.items():
            update_status(
                run_root,
                phase="connectome_candidates",
                active_variant=variant,
                active_metric=metric,
                connectome_jobs_done=done,
                connectome_jobs_total=total_jobs,
            )
            out_csv = conn / f"SC_AAL3_{args.sid}_source_contract__{variant}__{metric}.csv"
            assign_csv = conn / f"assignments_source_contract__{variant}__{metric}.csv"
            cached_stats = matrix_stats(out_csv, valid_labels)
            if (
                int(float(cached_stats.get("read_ok", 0) or 0)) == 1
                and assign_csv.exists()
                and assign_csv.stat().st_size > 0
            ):
                done += 1
                matrix_rows.append(
                    {
                        "candidate": "AAL3_source_contract",
                        "variant": variant,
                        "metric": metric,
                        "t1_ref": "t1_from_dicom",
                        "t1_input": "corrected_t1",
                        "b0": "eddy_vol0",
                        **cached_stats,
                    }
                )
                write_csv(run_root / "matrix_candidate_summary.csv", matrix_rows)
                continue

            cmd: list[str | Path] = [
                TCK2CONNECTOME,
                inputs["tracks"],
                aal_b0,
                out_csv,
                "-symmetric",
                "-zero_diagonal",
                *mrtrix_thread_args(env),
                *extra,
                "-out_assignments",
                assign_csv,
                "-stat_edge",
                "sum",
                *scale_args,
            ]
            run(cmd, logs / f"tck2connectome_source_contract__{variant}__{metric}.log", env)
            done += 1
            matrix_rows.append(
                {
                    "candidate": "AAL3_source_contract",
                    "variant": variant,
                    "metric": metric,
                    "t1_ref": "t1_from_dicom",
                    "t1_input": "corrected_t1",
                    "b0": "eddy_vol0",
                    **matrix_stats(out_csv, valid_labels),
                }
            )
            write_csv(run_root / "matrix_candidate_summary.csv", matrix_rows)

    candidates = [
        row
        for row in matrix_rows
        if row.get("candidate") == "AAL3_source_contract"
        and row.get("metric") in {"fd_sum", "fd_invnodevol"}
        and int(float(row.get("read_ok", 0) or 0)) == 1
    ]
    candidates.sort(key=lambda row: (-float(row.get("valid_density", -1) or -1), int(float(row.get("valid_zero_rows", 9999) or 9999))))
    best = candidates[0] if candidates else {}
    decision = classify(best, baseline_fd)

    lines = [
        "# AAL3 Source Contract Probe Decision",
        "",
        f"Generated: {utc()}",
        "",
        "Scratch-only: no production outputs were overwritten.",
        "",
        "## Tested contract",
        "",
        "- Atlas: `AAL3v1_1mm.nii.gz`",
        "- Route: `AAL3/MNI -> MNI brain -> t1_from_dicom -> eddy_vol0`",
        "- T1-to-B0 moving image: `corrected_T1`",
        "- Label interpolation: `nearestneighbour`, integer labels",
        "",
        "## Production baseline",
        "",
    ]
    if baseline_fd:
        lines.extend(
            [
                f"- Matrix: `{baseline_fd.get('matrix')}`",
                f"- Valid-node density: `{float(baseline_fd.get('valid_density')):.4f}`",
                f"- Valid zero rows: `{int(float(baseline_fd.get('valid_zero_rows'))):d}`",
                "",
            ]
        )
    else:
        lines.append("Production `fd_sum` baseline was not readable.\n")
    lines.extend(["## Best candidate", ""])
    if best:
        lines.extend(
            [
                f"- Variant: `{best.get('variant')}`",
                f"- Metric: `{best.get('metric')}`",
                f"- Shape: `{int(float(best.get('n'))):d} x {int(float(best.get('n'))):d}`",
                f"- Valid-node density: `{float(best.get('valid_density')):.4f}`",
                f"- Valid zero rows: `{int(float(best.get('valid_zero_rows'))):d}`",
                f"- All-row density: `{float(best.get('density')):.4f}`",
                f"- All zero rows: `{int(float(best.get('zero_rows'))):d}`",
                "",
            ]
        )
    else:
        lines.append("No readable AAL3 source-contract candidate was generated.\n")
    lines.extend(["## Decision", "", f"`{decision}`", ""])
    (run_root / "decision.md").write_text("\n".join(lines), encoding="utf-8")
    write_json(run_root / "decision.json", {"decision": decision, "best": best, "baseline_fd_sum": baseline_fd})
    update_status(
        run_root,
        phase="complete",
        completed_utc=utc(),
        active_variant="",
        active_metric="",
        connectome_jobs_done=total_jobs,
        connectome_jobs_total=total_jobs,
        decision=decision,
    )
    print(run_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
