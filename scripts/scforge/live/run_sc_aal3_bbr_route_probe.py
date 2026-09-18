#!/usr/bin/env python3
"""Scratch-only AAL3 BBR/native-route probe for sparse AD connectomes.

This is route 2 of the AAL3 rescue ladder. It reuses existing 3M streamlines
and SIFT2 weights, but tries BBR/native T1 parcellation routes before
`tck2connectome`. It never overwrites production files.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path("/home/ec2-user/exp")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.scforge.live.run_sc_aal3_source_contract_probe import (  # noqa: E402
    AAL3,
    AAL3_LABELS,
    DERIV,
    FLIRT,
    FSLDIR,
    MNI_BRAIN,
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

CONVERT_XFM = FSLDIR / "bin" / "convert_xfm"
OUT_PARENT = DERIV / "qc" / "sc_matrix_qc" / "aal3_route2_bbr_probe"


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def stamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


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


def bbr_inputs(sid: str) -> dict[str, Path]:
    bbr = DERIV / "dwi_t1_bbr"
    return {
        "b0": bbr / f"{sid}_b0mean_ras.nii.gz",
        "t1": bbr / f"{sid}_t1_ras.nii.gz",
        "t1brain": bbr / f"{sid}_t1brain_ras.nii.gz",
        "t12b0": bbr / f"{sid}_t12b0_bbr.mat",
    }


def route_aal_to_t1_to_b0(
    *,
    route_name: str,
    t1_ref: Path,
    b0_ref: Path,
    t12b0: Path,
    out_dir: Path,
    logs: Path,
    env: dict[str, str],
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    aal_t1 = out_dir / f"{route_name}_AAL3_t1.nii.gz"
    mni2t1 = out_dir / f"{route_name}_mni2t1.mat"
    aal_b0 = out_dir / f"{route_name}_AAL3_b0.nii.gz"
    if not aal_b0.exists() or aal_b0.stat().st_size <= 0:
        run(
            [
                FLIRT,
                "-in",
                AAL3,
                "-ref",
                t1_ref,
                "-omat",
                mni2t1,
                "-interp",
                "nearestneighbour",
                "-datatype",
                "int",
                "-out",
                aal_t1,
                "-dof",
                "12",
            ],
            logs / f"{route_name}_aal3_to_t1.log",
            env,
        )
        run(
            [
                FLIRT,
                "-in",
                aal_t1,
                "-ref",
                b0_ref,
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
            logs / f"{route_name}_aal3_t1_to_b0.log",
            env,
        )
    return aal_b0


def route_direct_composed(*, route_name: str, t1_ref: Path, b0_ref: Path, t12b0: Path, out_dir: Path, logs: Path, env: dict[str, str]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    mni2t1 = out_dir / f"{route_name}_mni2t1.mat"
    mni2b0 = out_dir / f"{route_name}_mni2b0.mat"
    aal_b0 = out_dir / f"{route_name}_AAL3_b0.nii.gz"
    if not aal_b0.exists() or aal_b0.stat().st_size <= 0:
        run([FLIRT, "-in", MNI_BRAIN, "-ref", t1_ref, "-omat", mni2t1, "-dof", "12"], logs / f"{route_name}_mni2t1.log", env)
        run([CONVERT_XFM, "-omat", mni2b0, "-concat", t12b0, mni2t1], logs / f"{route_name}_mni2b0.log", env)
        run(
            [
                FLIRT,
                "-in",
                AAL3,
                "-ref",
                b0_ref,
                "-applyxfm",
                "-init",
                mni2b0,
                "-interp",
                "nearestneighbour",
                "-datatype",
                "int",
                "-out",
                aal_b0,
            ],
            logs / f"{route_name}_aal3_direct_b0.log",
            env,
        )
    return aal_b0


def route_reuse_existing_aal_t1(*, sid: str, b0_ref: Path, t12b0: Path, out_dir: Path, logs: Path, env: dict[str, str]) -> Path | None:
    aal_t1 = DERIV / "parc" / sid / "AAL_t1.nii.gz"
    if not aal_t1.exists():
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    aal_b0 = out_dir / "reuse_existing_AAL_t1_AAL3_b0.nii.gz"
    if not aal_b0.exists() or aal_b0.stat().st_size <= 0:
        run(
            [
                FLIRT,
                "-in",
                aal_t1,
                "-ref",
                b0_ref,
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
            logs / "reuse_existing_aal_t1_to_b0.log",
            env,
        )
    return aal_b0


def assignment_args(name: str) -> list[str]:
    table = {
        "radial4": ["-assignment_radial_search", "4"],
        "forward40": ["-assignment_forward_search", "40"],
        "forward80": ["-assignment_forward_search", "80"],
    }
    if name not in table:
        raise ValueError(f"Unknown assignment variant: {name}")
    return table[name]


def finite_float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except Exception:
        return default


def best_candidate(matrix_rows: list[dict[str, Any]], label_rows: list[dict[str, Any]], baseline_fd: dict[str, Any]) -> dict[str, Any]:
    labels_by_stage = {row.get("stage"): row for row in label_rows}
    base_density = finite_float(baseline_fd.get("valid_density"), 0.0)
    base_zero = finite_float(baseline_fd.get("valid_zero_rows"), 9999.0)
    candidates = [
        row
        for row in matrix_rows
        if row.get("candidate") != "production_AAL3"
        and row.get("metric") == "fd_sum"
        and int(finite_float(row.get("read_ok"), 0)) == 1
    ]
    def key(row: dict[str, Any]) -> tuple[float, float, float, float]:
        label_stage = row.get("label_stage", "")
        labels = finite_float(labels_by_stage.get(label_stage, {}).get("n_labels"), 0)
        density = finite_float(row.get("valid_density"), -1)
        zero = finite_float(row.get("valid_zero_rows"), 9999)
        safe = 1.0 if density >= base_density and zero <= base_zero else 0.0
        return (safe, labels, -zero, density)
    candidates.sort(key=key, reverse=True)
    return candidates[0] if candidates else {}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sid", required=True)
    parser.add_argument("--tag", default=f"aal3_route2_bbr_{stamp()}")
    parser.add_argument("--variants", nargs="*", default=["forward40", "forward80"])
    parser.add_argument(
        "--min-labels-for-connectome",
        type=int,
        default=160,
        help="Skip tck2connectome for route candidates below this label-survival count.",
    )
    args = parser.parse_args()

    for tool in (FLIRT, CONVERT_XFM, TCK2CONNECTOME, AAL3, AAL3_LABELS, MNI_BRAIN):
        if not Path(tool).exists():
            raise FileNotFoundError(tool)

    env = os.environ.copy()
    env["FSLDIR"] = str(FSLDIR)
    env["PATH"] = f"{MRTRIX}:{FSLDIR / 'bin'}:" + env.get("PATH", "")

    run_root = OUT_PARENT / args.tag
    run_root.mkdir(parents=True, exist_ok=True)
    (OUT_PARENT / "latest_tag.txt").write_text(args.tag, encoding="utf-8")
    if (run_root / "decision.json").exists():
        print(run_root)
        return 0

    write_json(run_root / "status.json", {"sid": args.sid, "tag": args.tag, "phase": "starting", "started_utc": utc(), "variants": args.variants})

    inputs = discover_inputs(args.sid)
    bbr = bbr_inputs(args.sid)
    input_report = {
        **{k: ([str(x) for x in v] if isinstance(v, list) else str(v) if v else "") for k, v in inputs.items()},
        **{f"bbr_{k}": str(v) for k, v in bbr.items()},
    }
    write_json(run_root / "input_report.json", input_report)
    missing = [key for key in ("tracks", "weights") if not inputs.get(key)]
    missing += [f"bbr_{key}" for key, path in bbr.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required inputs: {missing}")

    valid_labels = aal3_valid_labels()
    logs = run_root / "logs"
    parc = run_root / "parc"
    conn = run_root / "connectomes"
    parc.mkdir(exist_ok=True)
    conn.mkdir(exist_ok=True)

    update_status(run_root, phase="baseline")
    label_rows: list[dict[str, Any]] = [{"stage": "source_AAL3", **label_stats(AAL3, valid_labels)}]
    if inputs.get("production_aal_b0"):
        label_rows.append({"stage": "production_AAL_b0", **label_stats(inputs["production_aal_b0"], valid_labels)})
    matrix_rows: list[dict[str, Any]] = []
    baseline_fd: dict[str, Any] = {}
    if inputs.get("production_fd_sum"):
        baseline_fd = {"candidate": "production_AAL3", "variant": "production", "metric": "fd_sum", **matrix_stats(inputs["production_fd_sum"], valid_labels)}
        matrix_rows.append(baseline_fd)
    write_csv(run_root / "label_survival_summary.csv", label_rows)
    write_csv(run_root / "matrix_candidate_summary.csv", matrix_rows)

    update_status(run_root, phase="build_routes")
    routes: dict[str, Path] = {
        "route2_bbr_t1brain_twostep": route_aal_to_t1_to_b0(
            route_name="route2_bbr_t1brain_twostep",
            t1_ref=bbr["t1brain"],
            b0_ref=bbr["b0"],
            t12b0=bbr["t12b0"],
            out_dir=parc / "route2_bbr_t1brain_twostep",
            logs=logs,
            env=env,
        ),
        "route2_bbr_t1_twostep": route_aal_to_t1_to_b0(
            route_name="route2_bbr_t1_twostep",
            t1_ref=bbr["t1"],
            b0_ref=bbr["b0"],
            t12b0=bbr["t12b0"],
            out_dir=parc / "route2_bbr_t1_twostep",
            logs=logs,
            env=env,
        ),
        "route2_bbr_direct_composed": route_direct_composed(
            route_name="route2_bbr_direct_composed",
            t1_ref=bbr["t1"],
            b0_ref=bbr["b0"],
            t12b0=bbr["t12b0"],
            out_dir=parc / "route2_bbr_direct_composed",
            logs=logs,
            env=env,
        ),
    }
    reused = route_reuse_existing_aal_t1(sid=args.sid, b0_ref=bbr["b0"], t12b0=bbr["t12b0"], out_dir=parc / "route2_reuse_existing_aal_t1", logs=logs, env=env)
    if reused:
        routes["route2_reuse_existing_aal_t1"] = reused

    for route_name, aal_b0 in routes.items():
        label_rows.append({"stage": route_name, **label_stats(aal_b0, valid_labels)})
    write_csv(run_root / "label_survival_summary.csv", label_rows)
    labels_by_route = {str(row.get("stage")): int(finite_float(row.get("n_labels"), 0)) for row in label_rows}

    total_jobs = len(routes) * len(args.variants)
    done = 0
    for route_name, aal_b0 in routes.items():
        for variant in args.variants:
            update_status(run_root, phase="connectome_candidates", active_route=route_name, active_variant=variant, connectome_jobs_done=done, connectome_jobs_total=total_jobs)
            out_csv = conn / f"SC_AAL3_{args.sid}_{route_name}__{variant}__fd_sum.csv"
            assign_csv = conn / f"assignments_{route_name}__{variant}__fd_sum.csv"
            route_labels = labels_by_route.get(route_name, 0)
            if route_labels < args.min_labels_for_connectome:
                done += 1
                matrix_rows.append(
                    {
                        "candidate": route_name,
                        "label_stage": route_name,
                        "variant": variant,
                        "metric": "fd_sum",
                        "assignment_file": "",
                        "exists": 0,
                        "read_ok": 0,
                        "valid_density": "",
                        "valid_zero_rows": "",
                        "error": f"skipped_low_label_survival_{route_labels}_lt_{args.min_labels_for_connectome}",
                    }
                )
                write_csv(run_root / "matrix_candidate_summary.csv", matrix_rows)
                continue

            stats = matrix_stats(out_csv, valid_labels)
            if int(finite_float(stats.get("read_ok"), 0)) != 1 or not assign_csv.exists() or assign_csv.stat().st_size <= 0:
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
                run(cmd, logs / f"tck2connectome_{route_name}__{variant}__fd_sum.log", env)
                stats = matrix_stats(out_csv, valid_labels)
            done += 1
            matrix_rows.append(
                {
                    "candidate": route_name,
                    "label_stage": route_name,
                    "variant": variant,
                    "metric": "fd_sum",
                    "assignment_file": str(assign_csv),
                    **stats,
                }
            )
            write_csv(run_root / "matrix_candidate_summary.csv", matrix_rows)

    best = best_candidate(matrix_rows, label_rows, baseline_fd)
    baseline_density = finite_float(baseline_fd.get("valid_density"), 0.0)
    baseline_zero = int(finite_float(baseline_fd.get("valid_zero_rows"), 9999))
    best_density = finite_float(best.get("valid_density"), 0.0)
    best_zero = int(finite_float(best.get("valid_zero_rows"), 9999))
    label_stage = str(best.get("label_stage") or "")
    best_label = next((r for r in label_rows if r.get("stage") == label_stage), {})
    if not best:
        route_labels = [r for r in label_rows if str(r.get("stage", "")).startswith("route2_")]
        route_labels.sort(key=lambda r: finite_float(r.get("n_labels"), 0), reverse=True)
        best_label = route_labels[0] if route_labels else {}
        label_stage = str(best_label.get("stage") or "")
    decision = {
        "decision": "AAL3_ROUTE2_BBR_CANDIDATE" if best else "AAL3_ROUTE2_BBR_NO_CANDIDATE",
        "best": best,
        "baseline_fd_sum": baseline_fd,
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
