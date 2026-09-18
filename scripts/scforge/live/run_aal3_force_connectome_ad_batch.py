#!/usr/bin/env python3
"""Scratch-only forced AAL3 source-contract connectome probes by diagnosis group.

This is a diagnostic batch: it does not overwrite production matrices.  It
selects subjects that already have production connectomes, reuses existing
tracks/SIFT2 weights, and forces AAL3 source-contract tck2connectome variants
even when label survival is poor.  The purpose is to learn whether better AAL3
assignment can rescue sparse matrices before considering tractography reruns.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any


ROOT = Path("/home/ec2-user/exp")
DERIV = Path("/data/derivatives")
QC_ROOT = DERIV / "qc" / "sc_matrix_qc"
OUT_PARENT = QC_ROOT / "aal3_force_connectome_ad_batch"
PROBE_PARENT = ROOT / "data" / "derivatives" / "qc" / "sc_matrix_qc" / "aal3_source_contract_probe"
PYTHON = ROOT / ".venv_connectome_app" / "bin" / "python"
PROBE_SCRIPT = ROOT / "scripts" / "scforge" / "live" / "run_sc_aal3_source_contract_probe.py"
MASTER = DERIV / "qc" / "analysis_cohort" / "00_master" / "master_cohort.csv"


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


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size <= 0:
        return []
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        return list(csv.DictReader(handle))


def append_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    fields = list(row)
    if exists:
        with path.open(newline="", encoding="utf-8", errors="replace") as handle:
            reader = csv.reader(handle)
            try:
                existing_fields = next(reader)
            except StopIteration:
                existing_fields = []
        if existing_fields:
            fields = list(existing_fields)
            for key in row:
                if key not in fields:
                    fields.append(key)
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def sid_from_master_row(row: dict[str, str]) -> str:
    subject = str(row.get("subject_id") or row.get("Subject ID") or "").strip()
    image = str(row.get("Image ID") or "").strip()
    if image.endswith(".0"):
        image = image[:-2]
    return f"{subject}_I{image}" if subject and image else ""


def has_file(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def scan_inputs_for_master_row(row: dict[str, str]) -> dict[str, Any]:
    subject = str(row.get("subject_id") or row.get("Subject ID") or "").strip()
    master_image = str(row.get("Image ID") or "").strip()
    if master_image.endswith(".0"):
        master_image = master_image[:-2]

    candidate_sids: list[tuple[str, str]] = []
    exact_sid = sid_from_master_row(row)
    if exact_sid:
        candidate_sids.append((exact_sid, "master_image_id"))

    if subject:
        for fd_path in sorted((DERIV / "connectomes").glob(f"SC_AAL_{subject}_I*_fd_sum.csv")):
            sid = fd_path.name.removeprefix("SC_AAL_").removesuffix("_fd_sum.csv")
            if sid and sid != exact_sid:
                candidate_sids.append((sid, "existing_step7_scan"))

    seen: set[str] = set()
    for sid, image_source in candidate_sids:
        if sid in seen:
            continue
        seen.add(sid)
        fd = DERIV / "connectomes" / f"SC_AAL_{sid}_fd_sum.csv"
        tracks = DERIV / "tracks" / sid / "tracks_final_3000k.tck"
        weights = DERIV / "tracks" / sid / "sift_weights.txt"
        if has_file(fd) and has_file(tracks) and has_file(weights):
            image_id = sid.rsplit("_I", 1)[-1] if "_I" in sid else ""
            return {
                "sid": sid,
                "image_id": image_id,
                "master_image_id": master_image,
                "image_id_source": image_source,
                "production_fd_sum": fd,
                "tracks": tracks,
                "weights": weights,
            }
    return {}


def eligible_subjects(groups: set[str], limit: int) -> list[dict[str, Any]]:
    rows = read_csv(MASTER)
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        group = str(row.get("group") or row.get("Research Group") or "").strip().upper()
        if group not in groups:
            continue
        scan_inputs = scan_inputs_for_master_row(row)
        sid = str(scan_inputs.get("sid") or "")
        if not sid or sid in seen:
            continue
        seen.add(sid)
        out.append(
            {
                "sid": sid,
                "subject_id": row.get("subject_id") or row.get("Subject ID") or "",
                "image_id": scan_inputs.get("image_id") or row.get("Image ID") or "",
                "master_image_id": scan_inputs.get("master_image_id") or row.get("Image ID") or "",
                "image_id_source": scan_inputs.get("image_id_source") or "master_image_id",
                "group": group,
                "production_fd_sum": str(scan_inputs["production_fd_sum"]),
                "tracks": str(scan_inputs["tracks"]),
                "weights": str(scan_inputs["weights"]),
            }
        )
        if limit and len(out) >= limit:
            break
    return out


def fnum(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except Exception:
        return default


def inum(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return default


def best_fd_candidate(probe_root: Path) -> dict[str, Any]:
    rows = [
        row
        for row in read_csv(probe_root / "matrix_candidate_summary.csv")
        if row.get("candidate") == "AAL3_source_contract" and row.get("metric") == "fd_sum"
    ]
    rows.sort(key=lambda row: (-fnum(row.get("valid_density"), -1), inum(row.get("valid_zero_rows"), 9999)))
    return rows[0] if rows else {}


def label_row(probe_root: Path, stage: str) -> dict[str, Any]:
    rows = [row for row in read_csv(probe_root / "label_survival_summary.csv") if row.get("stage") == stage]
    return rows[-1] if rows else {}


def production_fd_row(probe_root: Path) -> dict[str, Any]:
    rows = [
        row
        for row in read_csv(probe_root / "matrix_candidate_summary.csv")
        if row.get("candidate") == "production_AAL3" and row.get("metric") == "fd_sum"
    ]
    return rows[-1] if rows else {}


def run_one(subject: dict[str, Any], args: argparse.Namespace, run_root: Path) -> dict[str, Any]:
    sid = subject["sid"]
    tag = f"{args.tag}_{sid}"
    probe_root = PROBE_PARENT / tag
    done_marker = run_root / "subjects" / sid / "done.json"
    if done_marker.exists() and probe_root.exists():
        result = read_json(done_marker)
        result["resumed"] = 1
        return result

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["MRTRIX_NTHREADS"] = str(args.mrtrix_threads)
    env["FSLDIR"] = env.get("FSLDIR", "/home/ec2-user/fsl")
    if args.connectome_timeout_sec:
        env["AAL3_CONNECTOME_TIMEOUT_SEC"] = str(args.connectome_timeout_sec)

    cmd = [
        str(PYTHON),
        str(PROBE_SCRIPT),
        "--sid",
        sid,
        "--tag",
        tag,
        "--variants",
        *args.variants,
        "--metrics",
        "fd_sum",
        "--force-connectomes-on-low-survival",
    ]
    log_path = run_root / "logs" / f"{sid}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = utc()
    with log_path.open("a", encoding="utf-8", errors="replace") as handle:
        handle.write(f"[{started}] $ {' '.join(cmd)}\n")
        handle.flush()
        proc = subprocess.run(cmd, stdout=handle, stderr=subprocess.STDOUT, text=True, env=env, cwd=str(ROOT))
        handle.write(f"[{utc()}] rc={proc.returncode}\n")

    best = best_fd_candidate(probe_root)
    baseline = production_fd_row(probe_root)
    label_prod = label_row(probe_root, "production_AAL_b0")
    label_source = label_row(probe_root, "AAL3_source_contract_B0")
    result = {
        "sid": sid,
        "group": subject.get("group", ""),
        "status": "ok" if proc.returncode == 0 and best else "failed_or_no_candidate",
        "rc": proc.returncode,
        "probe_root": str(probe_root),
        "log": str(log_path),
        "production_label_survival": label_prod.get("expected_label_survival_fraction", ""),
        "source_contract_label_survival": label_source.get("expected_label_survival_fraction", ""),
        "baseline_density": baseline.get("valid_density", ""),
        "baseline_zero_rows": baseline.get("valid_zero_rows", ""),
        "best_variant": best.get("variant", ""),
        "best_density": best.get("valid_density", ""),
        "best_zero_rows": best.get("valid_zero_rows", ""),
        "density_delta": fnum(best.get("valid_density"), float("nan")) - fnum(baseline.get("valid_density"), float("nan")),
        "zero_rows_delta": inum(best.get("valid_zero_rows"), 9999) - inum(baseline.get("valid_zero_rows"), 9999),
        "started_utc": started,
        "finished_utc": utc(),
        "resumed": 0,
    }
    done_marker.parent.mkdir(parents=True, exist_ok=True)
    write_json(done_marker, result)
    return result


def update_status(run_root: Path, total: int, completed: int, failed: int, active: list[str], latest: dict[str, Any] | None = None) -> None:
    write_json(
        run_root / "status.json",
        {
            "tag": run_root.name.replace("aal3_force_connectome_ad_batch_", ""),
            "root": str(run_root),
            "phase": "running" if completed < total else "complete",
            "total": total,
            "completed": completed,
            "failed": failed,
            "active": active,
            "latest": latest or {},
            "updated_utc": utc(),
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="")
    parser.add_argument("--groups", nargs="*", default=["AD"], help="Diagnosis groups to include, for example AD CN MCI.")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--mrtrix-threads", type=int, default=4)
    parser.add_argument(
        "--connectome-timeout-sec",
        type=int,
        default=int(os.environ.get("AAL3_CONNECTOME_TIMEOUT_SEC", "2700")),
        help="Timeout for each tck2connectome call inside the source-contract probe.",
    )
    parser.add_argument("--variants", nargs="*", default=["radial4", "forward40", "forward80"])
    parser.add_argument("--no-update-latest", action="store_true", help="Do not update the parent latest_tag.txt pointer.")
    args = parser.parse_args()
    groups = {str(group).strip().upper() for group in args.groups if str(group).strip()}
    if not groups:
        raise ValueError("At least one group is required.")
    if not args.tag:
        group_slug = "_".join(sorted(groups)).lower()
        args.tag = f"{group_slug}_existing_tracks_{stamp()}"

    run_root = OUT_PARENT / args.tag
    run_root.mkdir(parents=True, exist_ok=True)
    if not args.no_update_latest:
        (OUT_PARENT / "latest_tag.txt").write_text(args.tag, encoding="utf-8")

    subjects = eligible_subjects(groups, args.limit)
    write_csv(run_root / "subjects.csv", subjects)
    write_json(
        run_root / "run_manifest.json",
        {
            "tag": args.tag,
            "root": str(run_root),
            "started_utc": utc(),
            "limit": args.limit,
            "groups": sorted(groups),
            "workers": args.workers,
            "mrtrix_threads": args.mrtrix_threads,
            "connectome_timeout_sec": args.connectome_timeout_sec,
            "variants": args.variants,
            "n_subjects": len(subjects),
            "mode": "scratch_only_existing_tracks_forced_aal3_source_contract",
        },
    )
    if not subjects:
        update_status(run_root, 0, 0, 0, [])
        raise RuntimeError(f"No eligible {sorted(groups)} subjects with fd_sum connectome, tracks, and SIFT2 weights were found.")

    completed = 0
    failed = 0
    results: list[dict[str, Any]] = []
    update_status(run_root, len(subjects), completed, failed, [])
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        pending_subjects = list(subjects)
        future_to_subject = {}

        def submit_until_full() -> None:
            while pending_subjects and len(future_to_subject) < max(1, args.workers):
                subject = pending_subjects.pop(0)
                future_to_subject[pool.submit(run_one, subject, args, run_root)] = subject

        def active_sids() -> list[str]:
            return sorted(subject["sid"] for subject in future_to_subject.values())

        submit_until_full()
        update_status(run_root, len(subjects), completed, failed, active_sids())
        while future_to_subject:
            done_futures, _ = wait(future_to_subject, return_when=FIRST_COMPLETED)
            latest_row: dict[str, Any] | None = None
            for future in done_futures:
                subject = future_to_subject.pop(future)
                sid = subject["sid"]
                try:
                    row = future.result()
                except Exception as exc:
                    failed += 1
                    row = {
                        "sid": sid,
                        "group": subject.get("group", ""),
                        "status": "exception",
                        "error": f"{type(exc).__name__}: {exc}",
                        "finished_utc": utc(),
                    }
                completed += 1
                if row.get("status") != "ok":
                    failed += 1 if row.get("status") != "exception" else 0
                results.append(row)
                latest_row = row
                append_csv(run_root / "batch_results_incremental.csv", row)
                write_csv(run_root / "batch_results.csv", results)
            submit_until_full()
            update_status(run_root, len(subjects), completed, failed, active_sids(), latest_row)

    update_status(run_root, len(subjects), completed, failed, [], results[-1] if results else {})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
