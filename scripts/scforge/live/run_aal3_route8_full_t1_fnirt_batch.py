#!/usr/bin/env python3
"""Run route-8 full-T1 FNIRT AAL3 rescue for label-limited AD failures.

Route 7's skull-stripped T1 FNIRT path can fail badly when the brain mask or
contrast differs from the MNI brain template. Route 8 keeps the same production
tracks/SIFT2 weights, but estimates the nonlinear MNI transform with full T1
against the full MNI152 T1 image, then maps AAL3 to B0 through the existing BBR
T1-to-B0 transform.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path("/home/ec2-user/exp")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.scforge.live.run_aal3_route7_selective_rescue_batch import (  # noqa: E402
    APPLYWARP,
    FNIRT,
    INVWARP,
    PROBE_PARENT as ROUTE7_PROBE_PARENT,
    SCORER,
    TCK2CONNECTOME,
    assignment_args,
    choose_best_matrix,
    env_for_mrtrix,
    fnum,
    matrix_stats,
    mrtrix_thread_args,
    production_baseline,
    row_from_best,
)
from scripts.scforge.live.run_sc_aal3_bbr_route_probe import bbr_inputs  # noqa: E402
from scripts.scforge.live.run_sc_aal3_source_contract_probe import (  # noqa: E402
    AAL3,
    FLIRT,
    FSLDIR,
    aal3_valid_labels,
    discover_inputs,
    label_stats,
    run,
    write_csv,
    write_json,
)

RUN_PARENT = Path("/data/derivatives/qc/sc_matrix_qc/aal3_force_connectome_ad_batch")
PROBE_PARENT = ROUTE7_PROBE_PARENT.parent / "aal3_route8_full_t1_fnirt_probe"
MNI_FULL = FSLDIR / "data" / "standard" / "MNI152_T1_1mm.nii.gz"
MIN_PROMOTABLE_LABELS = 160
CONVERT_XFM = FSLDIR / "bin" / "convert_xfm"
ROUTE8_SCHEMA = "full_t1_inverse_bbr_v3"
DEFAULT_CONNECTOME_TIMEOUT_SEC = 45 * 60


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


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


def same_nifti_grid(first: Path, second: Path) -> bool:
    """Return true when two NIfTI files can be used on the same voxel grid."""
    try:
        import nibabel as nib
        import numpy as np

        img1 = nib.load(str(first))
        img2 = nib.load(str(second))
        return img1.shape[:3] == img2.shape[:3] and bool(np.allclose(img1.affine, img2.affine, atol=1e-3))
    except Exception:
        return False


def update_status(root: Path, **updates: Any) -> None:
    status = read_json(root / "status.json")
    status.update(updates)
    status["updated_utc"] = utc()
    write_json(root / "status.json", status)


def run_with_timeout(cmd: list[str | Path], log_path: Path, env: dict[str, str], timeout_sec: int) -> None:
    """Run a command with an audit log and a hard timeout."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    str_cmd = [str(x) for x in cmd]
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
                timeout=timeout_sec if timeout_sec > 0 else None,
            )
        except subprocess.TimeoutExpired as exc:
            handle.write(f"[{utc()}] timeout_after_sec={timeout_sec}\n\n")
            raise TimeoutError(f"command timed out after {timeout_sec}s; see {log_path}") from exc
        handle.write(f"[{utc()}] rc={proc.returncode}\n\n")
    if proc.returncode != 0:
        raise RuntimeError(f"command failed rc={proc.returncode}; see {log_path}")


def empty_failed_matrix_row(
    *,
    label_stage: str,
    variant: str,
    assign_csv: Path,
    error: Exception,
) -> dict[str, Any]:
    """Record a failed assignment variant without failing the whole subject."""
    return {
        "candidate": label_stage,
        "label_stage": label_stage,
        "variant": variant,
        "metric": "fd_sum",
        "assignment_file": str(assign_csv),
        "read_ok": 0,
        "valid_density": -1,
        "valid_zero_rows": 9999,
        "route8_variant_status": "failed_continue_next_variant",
        "route8_variant_error": f"{type(error).__name__}: {error}",
        "route8_variant_failed_utc": utc(),
    }


def run_connectomes_route8(
    *,
    sid: str,
    probe_root: Path,
    label_stage: str,
    parc_path: Path,
    tracks: Path,
    weights: Path,
    variants: list[str],
    label_rows: list[dict[str, Any]],
    matrix_rows: list[dict[str, Any]],
    valid_labels: list[int],
    env: dict[str, str],
    timeout_sec: int,
) -> list[dict[str, Any]]:
    conn = probe_root / "connectomes"
    logs = probe_root / "logs"
    conn.mkdir(parents=True, exist_ok=True)
    total = len(variants)
    for idx, variant in enumerate(variants, start=1):
        update_status(
            probe_root,
            phase="connectome_candidates",
            active_variant=variant,
            connectome_jobs_done=idx - 1,
            connectome_jobs_total=total,
            connectome_timeout_sec=timeout_sec,
        )
        out_csv = conn / f"SC_AAL3_{sid}_{label_stage}__{variant}__fd_sum.csv"
        assign_csv = conn / f"assignments_{label_stage}__{variant}__fd_sum.csv"
        stats = matrix_stats(out_csv, valid_labels)
        if int(fnum(stats.get("read_ok"), 0)) != 1 or not assign_csv.exists() or assign_csv.stat().st_size <= 0:
            cmd: list[str | Path] = [
                TCK2CONNECTOME,
                tracks,
                parc_path,
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
                weights,
            ]
            try:
                run_with_timeout(cmd, logs / f"tck2connectome_{label_stage}__{variant}__fd_sum.log", env, timeout_sec)
                stats = matrix_stats(out_csv, valid_labels)
            except Exception as exc:
                failed_row = empty_failed_matrix_row(
                    label_stage=label_stage,
                    variant=variant,
                    assign_csv=assign_csv,
                    error=exc,
                )
                matrix_rows.append(failed_row)
                write_csv(probe_root / "matrix_candidate_summary.csv", matrix_rows)
                update_status(
                    probe_root,
                    active_variant=variant,
                    latest_variant_status="failed_continue_next_variant",
                    latest_variant_error=f"{type(exc).__name__}: {exc}",
                )
                continue
        matrix_rows.append(
            {
                "candidate": label_stage,
                "label_stage": label_stage,
                "variant": variant,
                "metric": "fd_sum",
                "assignment_file": str(assign_csv),
                **stats,
            }
        )
        write_csv(probe_root / "matrix_candidate_summary.csv", matrix_rows)
    update_status(probe_root, connectome_jobs_done=total, connectome_jobs_total=total)
    return matrix_rows


def finished_sids(run_root: Path, retry_failed: bool) -> set[str]:
    finished: set[str] = set()
    for row in read_csv(run_root / "batch_results.csv"):
        sid = row.get("sid", "")
        if not sid:
            continue
        if row.get("status") == "ok" and row.get("route8_schema") == ROUTE8_SCHEMA:
            finished.add(sid)
        elif row.get("status") == "failed" and not retry_failed:
            finished.add(sid)
    return finished


def queue_subjects(route6_root: Path) -> list[dict[str, str]]:
    rows = read_csv(route6_root / "route_qc_decisions.csv")
    out: dict[str, dict[str, str]] = {}
    for row in rows:
        if not row.get("sid"):
            continue
        if row.get("qc_decision") not in {"TRY_ANOTHER_ROUTE", "REVIEW_ROUTE"}:
            continue
        if row.get("diagnostic_bucket") != "label_limited_diagnostic":
            continue
        out[row["sid"]] = {**row, "route8_action": "full_t1_fnirt_label_rescue"}
    return [out[sid] for sid in sorted(out)]


def run_one(
    row: dict[str, str],
    tag: str,
    route1_root: Path,
    variants: list[str],
    mrtrix_threads: int,
    connectome_timeout_sec: int,
) -> dict[str, Any]:
    sid = row["sid"]
    env = env_for_mrtrix(mrtrix_threads)
    probe_root = PROBE_PARENT / f"{tag}_{sid}"
    probe_root.mkdir(parents=True, exist_ok=True)
    cached = read_json(probe_root / "decision.json")
    if cached.get("route8_schema") == ROUTE8_SCHEMA:
        return cached.get("batch_row", {})
    update_status(probe_root, sid=sid, tag=f"{tag}_{sid}", phase="starting", route8_action=row.get("route8_action"))

    inputs = discover_inputs(sid)
    missing = [key for key in ("tracks", "weights", "production_fd_sum") if not inputs.get(key)]
    bbr = bbr_inputs(sid)
    missing += [f"bbr_{key}" for key, path in bbr.items() if not path.exists()]
    if not MNI_FULL.exists():
        missing.append("mni_full")
    if missing:
        raise FileNotFoundError(f"Missing route8 full-T1 FNIRT inputs for {sid}: {missing}")

    valid_labels = aal3_valid_labels()
    label_rows, matrix_rows, baseline_fd = production_baseline(inputs, valid_labels)
    logs = probe_root / "logs"
    parc = probe_root / "parc" / "route8_fnirt_full_t1"
    parc.mkdir(parents=True, exist_ok=True)

    t1_to_mni = parc / "route8_t1full2mni_affine.mat"
    t1_to_mni_warp = parc / "route8_t1full2mni_warpcoef.nii.gz"
    mni_to_t1_warp = parc / "route8_mni2t1full_warpcoef.nii.gz"
    aal_t1 = parc / "route8_fnirt_AAL3_t1full.nii.gz"
    aal_b0_bbr = parc / "route8_fnirt_AAL3_b0_bbr.nii.gz"
    t12b0_inv = parc / "route8_t12b0_bbr_inverse.mat"
    aal_b0_inv = parc / "route8_fnirt_AAL3_b0_bbr_inverse.nii.gz"
    if not aal_t1.exists() or aal_t1.stat().st_size <= 0:
        update_status(probe_root, phase="full_t1_affine")
        run([FLIRT, "-in", bbr["t1"], "-ref", MNI_FULL, "-omat", t1_to_mni, "-dof", "12"], logs / "t1full_to_mni_affine.log", env)
        update_status(probe_root, phase="full_t1_fnirt")
        run([FNIRT, f"--in={bbr['t1']}", f"--ref={MNI_FULL}", f"--aff={t1_to_mni}", f"--cout={t1_to_mni_warp}"], logs / "t1full_to_mni_fnirt.log", env)
        update_status(probe_root, phase="invert_warp")
        run([INVWARP, f"--warp={t1_to_mni_warp}", f"--ref={bbr['t1']}", f"--out={mni_to_t1_warp}"], logs / "mni_to_t1full_invwarp.log", env)
        update_status(probe_root, phase="apply_aal3_warp")
        run([APPLYWARP, f"--in={AAL3}", f"--ref={bbr['t1']}", f"--warp={mni_to_t1_warp}", f"--out={aal_t1}", "--interp=nn"], logs / "aal3_mni_to_t1full_fnirt.log", env)

    if same_nifti_grid(aal_t1, bbr["b0"]):
        if not t12b0_inv.exists() or t12b0_inv.stat().st_size <= 0:
            update_status(probe_root, phase="invert_t1_to_b0_bbr_matrix")
            run([CONVERT_XFM, "-omat", t12b0_inv, "-inverse", bbr["t12b0"]], logs / "invert_t12b0_bbr_matrix.log", env)
        if not aal_b0_inv.exists() or aal_b0_inv.stat().st_size <= 0:
            update_status(probe_root, phase="apply_inverse_bbr_to_b0")
            run(
                [
                    FLIRT,
                    "-in",
                    aal_t1,
                    "-ref",
                    bbr["b0"],
                    "-applyxfm",
                    "-init",
                    t12b0_inv,
                    "-interp",
                    "nearestneighbour",
                    "-datatype",
                    "int",
                    "-out",
                    aal_b0_inv,
                ],
                logs / "aal3_t1full_to_b0_inverse_bbr.log",
                env,
            )

        inverse_stats = label_stats(aal_b0_inv, valid_labels)
        label_rows.append({"stage": "route8_full_t1_inverse_bbr_parcellation", **inverse_stats})
        if int(float(inverse_stats.get("n_labels") or 0)) >= MIN_PROMOTABLE_LABELS:
            label_stage = "route8_full_t1_inverse_bbr_parcellation"
            parc_path = aal_b0_inv
            full_t1_label_stats = inverse_stats
            route8_grid_policy = "applied_inverse_bbr_matrix_after_full_t1_fnirt"
        else:
            label_stage = "route8_full_t1_samegrid_parcellation"
            parc_path = aal_t1
            full_t1_label_stats = label_stats(parc_path, valid_labels)
            label_rows.append({"stage": label_stage, **full_t1_label_stats})
            route8_grid_policy = "inverse_bbr_low_label_survival_used_full_t1_aal_directly_because_t1_and_b0_grids_match"
    else:
        label_stage = "route8_full_t1_bbr_parcellation"
        parc_path = aal_b0_bbr
        route8_grid_policy = "applied_t1_to_b0_bbr_because_t1_and_b0_grids_differ"
        if not aal_b0_bbr.exists() or aal_b0_bbr.stat().st_size <= 0:
            update_status(probe_root, phase="apply_t1_to_b0_bbr")
            run(
                [
                    FLIRT,
                    "-in",
                    aal_t1,
                    "-ref",
                    bbr["b0"],
                    "-applyxfm",
                    "-init",
                    bbr["t12b0"],
                    "-interp",
                    "nearestneighbour",
                    "-datatype",
                    "int",
                    "-out",
                    aal_b0_bbr,
                ],
                logs / "aal3_t1full_to_b0_bbr.log",
                env,
            )
        full_t1_label_stats = label_stats(parc_path, valid_labels)
        label_rows.append({"stage": label_stage, **full_t1_label_stats})

    old_bbr = parc / "route8_fnirt_AAL3_b0.nii.gz"
    if old_bbr.exists() and old_bbr != parc_path:
        label_rows.append({"stage": "route8_legacy_bbr_parcellation", **label_stats(old_bbr, valid_labels)})

    write_csv(probe_root / "label_survival_summary.csv", label_rows)
    write_csv(probe_root / "matrix_candidate_summary.csv", matrix_rows)
    n_labels = int(float(full_t1_label_stats.get("n_labels") or 0))
    if n_labels < MIN_PROMOTABLE_LABELS:
        batch_row = row_from_best(
            sid,
            "ok",
            probe_root,
            {},
            baseline_fd,
            label_rows,
            {
                "route8_action": row.get("route8_action", ""),
                "route8_reason": f"full-T1 FNIRT label survival {n_labels}/166 below promotable gate {MIN_PROMOTABLE_LABELS}/166",
                "route8_schema": ROUTE8_SCHEMA,
                "route8_grid_policy": route8_grid_policy,
            },
        )
        write_json(
            probe_root / "decision.json",
            {
                "route8_schema": ROUTE8_SCHEMA,
                "decision": "AAL3_ROUTE8_FULL_T1_FNIRT_LOW_LABEL_SURVIVAL",
                "best": {},
                "batch_row": batch_row,
            },
        )
        update_status(probe_root, phase="complete", completed_utc=utc(), decision="AAL3_ROUTE8_FULL_T1_FNIRT_LOW_LABEL_SURVIVAL")
        return batch_row

    run_connectomes_route8(
        sid=sid,
        probe_root=probe_root,
        label_stage=label_stage,
        parc_path=parc_path,
        tracks=inputs["tracks"],
        weights=inputs["weights"],
        variants=variants,
        label_rows=label_rows,
        matrix_rows=matrix_rows,
        valid_labels=valid_labels,
        env=env,
        timeout_sec=connectome_timeout_sec,
    )
    best = choose_best_matrix(matrix_rows, label_rows, baseline_fd)
    batch_row = row_from_best(
        sid,
        "ok",
        probe_root,
        best,
        baseline_fd,
        label_rows,
        {"route8_action": row.get("route8_action", ""), "route8_schema": ROUTE8_SCHEMA, "route8_grid_policy": route8_grid_policy},
    )
    write_json(
        probe_root / "decision.json",
        {"route8_schema": ROUTE8_SCHEMA, "decision": "AAL3_ROUTE8_FULL_T1_FNIRT_LABEL_RESCUE", "best": best, "batch_row": batch_row},
    )
    update_status(probe_root, phase="complete", completed_utc=utc(), decision="AAL3_ROUTE8_FULL_T1_FNIRT_LABEL_RESCUE")
    return batch_row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route1-root", type=Path, required=True)
    parser.add_argument("--route6-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--mrtrix-threads", type=int, default=4)
    parser.add_argument(
        "--variants",
        nargs="*",
        default=["endvox", "radial8", "radial16", "forward40", "forward80", "forward120"],
        help="Assignment variants to try. Cheap endpoint/radial variants run first by default.",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--connectome-timeout-sec", type=int, default=DEFAULT_CONNECTOME_TIMEOUT_SEC)
    parser.add_argument("--retry-failed", action="store_true", help="Retry subjects already marked failed in batch_results.csv.")
    args = parser.parse_args()

    run_root = RUN_PARENT / f"{args.route1_root.name}_route8_full_t1_fnirt"
    run_root.mkdir(parents=True, exist_ok=True)
    tag = run_root.name
    subjects = queue_subjects(args.route6_root)
    if args.limit > 0:
        subjects = subjects[: args.limit]
    results: list[dict[str, Any]] = list(read_csv(run_root / "batch_results.csv"))
    done = finished_sids(run_root, retry_failed=args.retry_failed)
    pending = [row for row in subjects if row["sid"] not in done]
    initial_ok = sum(1 for r in results if r.get("status") == "ok")
    initial_failed = sum(1 for r in results if r.get("status") == "failed")
    write_json(
        run_root / "run_manifest.json",
        {
            "mode": "route8_full_t1_fnirt_label_rescue",
            "route1_root": str(args.route1_root),
            "route6_root": str(args.route6_root),
            "root": str(run_root),
            "tag": tag,
            "n_subjects": len(subjects),
            "workers": args.workers,
            "mrtrix_threads": args.mrtrix_threads,
            "variants": args.variants,
            "connectome_timeout_sec": args.connectome_timeout_sec,
            "retry_failed": bool(args.retry_failed),
            "started_utc": utc(),
        },
    )
    update_status(run_root, phase="running", total=len(subjects), completed=initial_ok, failed=initial_failed, active=[])

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {
            pool.submit(
                run_one,
                row,
                tag,
                args.route1_root,
                args.variants,
                args.mrtrix_threads,
                args.connectome_timeout_sec,
            ): row
            for row in pending
        }
        update_status(run_root, active=[row["sid"] for row in pending[: args.workers]])
        for fut in as_completed(futures):
            subject_row = futures[fut]
            sid = subject_row["sid"]
            try:
                row = fut.result()
            except Exception as exc:
                row = {
                    "sid": sid,
                    "status": "failed",
                    "route8_action": subject_row.get("route8_action", ""),
                    "route8_schema": ROUTE8_SCHEMA,
                    "error": f"{type(exc).__name__}: {exc}",
                    "finished_utc": utc(),
                }
            results = [r for r in results if r.get("sid") != sid] + [row]
            write_csv(run_root / "batch_results.csv", results)
            completed = sum(1 for r in results if r.get("status") == "ok")
            failed = sum(1 for r in results if r.get("status") == "failed")
            remaining = [r for r in pending if r["sid"] not in {x.get("sid") for x in results}]
            update_status(run_root, completed=completed, failed=failed, active=[r["sid"] for r in remaining[: args.workers]], latest=row)
            subprocess.run(
                ["/home/ec2-user/exp/.venv_connectome_app/bin/python", str(SCORER), "--run-root", str(run_root)],
                cwd="/home/ec2-user/exp",
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )

    update_status(run_root, phase="complete", completed=sum(1 for r in results if r.get("status") == "ok"), failed=sum(1 for r in results if r.get("status") == "failed"), active=[], completed_utc=utc())
    subprocess.run(
        ["/home/ec2-user/exp/.venv_connectome_app/bin/python", str(SCORER), "--run-root", str(run_root)],
        cwd="/home/ec2-user/exp",
        stdout=(run_root / "score_stdout.log").open("w"),
        stderr=(run_root / "score_stderr.log").open("w"),
        check=False,
    )
    print(run_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
