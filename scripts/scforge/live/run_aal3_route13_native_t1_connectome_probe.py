#!/usr/bin/env python3
"""Route 13 connectome probe from corrected native-T1 AAL registration.

This is the downstream, scratch-only half of Route 13.  The Step 7 canary
rebuilds native-T1 5TT/GMWMI/FOD inputs.  This probe tests the corrected AAL3
registration contract against existing 3M tractography and SIFT2 weights
without overwriting production connectomes.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


ROOT = Path("/home/ec2-user/exp")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from connectome_pipeline.connectome_step7 import (  # noqa: E402
    Step7Config,
    _base_env,
    _find_native_t1_inputs,
    _subject_paths,
)
from scripts.scforge.live.run_aal3_route7_selective_rescue_batch import (  # noqa: E402
    choose_best_matrix,
    env_for_mrtrix,
    production_baseline,
    row_from_best,
)
from scripts.scforge.live.run_aal3_route8_full_t1_fnirt_batch import (  # noqa: E402
    CONVERT_XFM,
    run_connectomes_route8,
    run_with_timeout,
)
from scripts.scforge.live.run_sc_aal3_source_contract_probe import (  # noqa: E402
    AAL3,
    FLIRT,
    FSLDIR,
    QC_ROOT,
    aal3_valid_labels,
    discover_inputs,
    label_stats,
    write_csv,
    write_json,
)


RUN_PARENT = Path("/data/derivatives/qc/sc_matrix_qc/aal3_force_connectome_ad_batch")
DEFAULT_ROUTE1_ROOT = RUN_PARENT / "ad_existing_tracks_20260529T055752Z"
DEFAULT_ROUTE13_ROOT = DEFAULT_ROUTE1_ROOT.parent / f"{DEFAULT_ROUTE1_ROOT.name}_route13_step7_preflight_rebuild"
PROBE_PARENT = QC_ROOT / "aal3_route13_native_t1_connectome_probe"
ROUTE13_SCHEMA = "native_t1_step7_parc_existing_tracks_v1"
LABEL_STAGE = "route13_native_t1_twostep"


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


def update_status(root: Path, **updates: Any) -> None:
    status = read_json(root / "status.json")
    status.update(updates)
    status["updated_utc"] = utc()
    write_json(root / "status.json", status)


def append_or_replace_csv(path: Path, row: dict[str, Any], key: str = "sid") -> None:
    rows = [existing for existing in read_csv(path) if existing.get(key) != str(row.get(key, ""))]
    rows.append(row)
    write_csv(path, rows)


def step7_ok_sids(route13_root: Path, stage: str) -> set[str]:
    ok: set[str] = set()
    for row in read_csv(route13_root / "step7_summary.csv"):
        if row.get("stage") == stage and row.get("status") == "ok":
            sid = row.get("sid") or ""
            if sid:
                ok.add(sid)
    return ok


def target_rows(route13_root: Path, subjects: set[str], require_fod: bool) -> list[dict[str, str]]:
    rows = read_csv(route13_root / "subjects.csv")
    prep_ok = step7_ok_sids(route13_root, "prep")
    fod_ok = step7_ok_sids(route13_root, "fod")
    allowed = fod_ok if require_fod else prep_ok
    out: list[dict[str, str]] = []
    for row in rows:
        sid = row.get("sid") or row.get("subject") or row.get("subject_id") or ""
        if not sid:
            continue
        if subjects and sid not in subjects:
            continue
        if sid not in allowed:
            continue
        out.append({"sid": sid, "group": (row.get("group") or "AD").strip().upper() or "AD"})
    return out


def finished_sids(run_root: Path, retry_failed: bool) -> set[str]:
    done: set[str] = set()
    for row in read_csv(run_root / "batch_results.csv"):
        sid = row.get("sid", "")
        if not sid:
            continue
        if row.get("status") == "ok" and row.get("route13_schema") == ROUTE13_SCHEMA:
            done.add(sid)
        elif row.get("status") == "failed" and not retry_failed:
            done.add(sid)
    for marker in (run_root / "subjects").glob("*/done.json"):
        payload = read_json(marker)
        sid = str(payload.get("sid") or marker.parent.name)
        if payload.get("route13_schema") == ROUTE13_SCHEMA and sid:
            if payload.get("status") == "ok" or not retry_failed:
                done.add(sid)
    return done


def run_fsl(cmd: list[str | Path], log_path: Path, timeout_sec: int) -> None:
    env = _base_env(Step7Config())
    run_with_timeout(cmd, log_path, env, timeout_sec)


def build_native_t1_aal_b0(sid: str, probe_root: Path, timeout_sec: int) -> Path:
    cfg = Step7Config()
    paths = _subject_paths(cfg, sid, create_dirs=False)
    native_t1, native_t1_brain, native_source = _find_native_t1_inputs(cfg, sid)
    native_ref = native_t1_brain or native_t1
    if not paths["B0MEAN"].exists():
        raise RuntimeError(f"missing B0 mean: {paths['B0MEAN']}")
    if not paths["T12B0"].exists():
        raise RuntimeError(f"missing T1->B0 matrix: {paths['T12B0']}")

    parc_dir = probe_root / "parc" / LABEL_STAGE
    logs = probe_root / "logs"
    parc_dir.mkdir(parents=True, exist_ok=True)
    mni2t1 = parc_dir / "mni2t1_native.mat"
    aal_t1 = parc_dir / "AAL3_native_t1.nii.gz"
    aal_b0 = parc_dir / "AAL3_native_t1_to_b0.nii.gz"
    provenance = {
        "sid": sid,
        "route13_schema": ROUTE13_SCHEMA,
        "label_stage": LABEL_STAGE,
        "native_t1": str(native_t1),
        "native_t1_brain": str(native_t1_brain or ""),
        "native_t1_source": native_source,
        "b0mean": str(paths["B0MEAN"]),
        "t12b0": str(paths["T12B0"]),
        "aal_mni": str(AAL3),
        "created_utc": utc(),
    }

    update_status(probe_root, phase="aal_to_native_t1", active_variant="", route13_schema=ROUTE13_SCHEMA)
    run_fsl(
        [
            FLIRT,
            "-in",
            FSLDIR / "data" / "standard" / "MNI152_T1_1mm_brain.nii.gz",
            "-ref",
            native_ref,
            "-omat",
            mni2t1,
            "-dof",
            "12",
        ],
        logs / "flirt_mni2t1_native.log",
        timeout_sec,
    )
    update_status(probe_root, phase="aal_to_native_t1_labels")
    run_fsl(
        [
            FLIRT,
            "-in",
            AAL3,
            "-ref",
            native_ref,
            "-applyxfm",
            "-init",
            mni2t1,
            "-interp",
            "nearestneighbour",
            "-datatype",
            "int",
            "-out",
            aal_t1,
        ],
        logs / "flirt_aal_to_native_t1.log",
        timeout_sec,
    )
    update_status(probe_root, phase="aal_native_t1_to_b0")
    run_fsl(
        [
            FLIRT,
            "-in",
            aal_t1,
            "-ref",
            paths["B0MEAN"],
            "-applyxfm",
            "-init",
            paths["T12B0"],
            "-interp",
            "nearestneighbour",
            "-datatype",
            "int",
            "-out",
            aal_b0,
        ],
        logs / "flirt_aal_native_t1_to_b0.log",
        timeout_sec,
    )
    write_json(probe_root / "parc" / "route13_parcellation_provenance.json", provenance)
    return aal_b0


def run_one(
    *,
    row: dict[str, str],
    tag: str,
    run_root: Path,
    route13_root: Path,
    variants: list[str],
    mrtrix_threads: int,
    transform_timeout_sec: int,
    connectome_timeout_sec: int,
) -> dict[str, Any]:
    sid = row["sid"]
    group = row.get("group") or "AD"
    probe_root = PROBE_PARENT / f"{tag}_route13_native_t1_connectome_{sid}"
    probe_root.mkdir(parents=True, exist_ok=True)
    update_status(probe_root, phase="starting", sid=sid, group=group, route13_schema=ROUTE13_SCHEMA)
    valid_labels = aal3_valid_labels()
    inputs = discover_inputs(sid)
    label_rows, matrix_rows, baseline_fd = production_baseline(inputs, valid_labels)
    try:
        if not inputs.get("tracks"):
            raise RuntimeError("missing existing 3M tracks")
        if not inputs.get("weights"):
            raise RuntimeError("missing existing SIFT2 weights")
        if not inputs.get("production_fd_sum"):
            raise RuntimeError("missing production fd_sum baseline")

        aal_b0 = build_native_t1_aal_b0(sid, probe_root, transform_timeout_sec)
        label_row = {"stage": LABEL_STAGE, **label_stats(aal_b0, valid_labels)}
        label_rows.append(label_row)
        write_csv(probe_root / "label_survival_summary.csv", label_rows)
        write_csv(probe_root / "matrix_candidate_summary.csv", matrix_rows)

        matrix_rows = run_connectomes_route8(
            sid=sid,
            probe_root=probe_root,
            label_stage=LABEL_STAGE,
            parc_path=aal_b0,
            tracks=inputs["tracks"],
            weights=inputs["weights"],
            variants=variants,
            label_rows=label_rows,
            matrix_rows=matrix_rows,
            valid_labels=valid_labels,
            env=env_for_mrtrix(mrtrix_threads),
            timeout_sec=connectome_timeout_sec,
        )
        best = choose_best_matrix(matrix_rows, label_rows, baseline_fd)
        status = "ok" if best else "failed"
        result = row_from_best(
            sid,
            status,
            probe_root,
            best,
            baseline_fd,
            label_rows,
            extra={
                "group": group,
                "route13_schema": ROUTE13_SCHEMA,
                "route13_root": str(route13_root),
                "candidate_parc_path": str(aal_b0),
                "candidate_policy": "native T1 AAL3 two-step to B0; existing 3M tracks and existing SIFT2 weights",
            },
        )
        if not best:
            result["route13_error"] = "no readable fd_sum candidate"
    except Exception as exc:
        result = row_from_best(
            sid,
            "failed",
            probe_root,
            {},
            baseline_fd,
            label_rows,
            extra={
                "group": group,
                "route13_schema": ROUTE13_SCHEMA,
                "route13_root": str(route13_root),
                "route13_error": f"{type(exc).__name__}: {exc}",
            },
        )
    write_csv(probe_root / "label_survival_summary.csv", label_rows)
    write_csv(probe_root / "matrix_candidate_summary.csv", matrix_rows)
    write_json(probe_root / "decision.json", {"route13_schema": ROUTE13_SCHEMA, "batch_row": result})
    subject_dir = run_root / "subjects" / sid
    subject_dir.mkdir(parents=True, exist_ok=True)
    write_json(subject_dir / "done.json", result)
    update_status(probe_root, phase="complete" if result.get("status") == "ok" else "failed")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route1-root", type=Path, default=DEFAULT_ROUTE1_ROOT)
    parser.add_argument("--route13-root", type=Path, default=DEFAULT_ROUTE13_ROOT)
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--subjects", nargs="*", default=[])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--mrtrix-threads", type=int, default=2)
    parser.add_argument("--variants", nargs="+", default=["forward40", "forward80", "forward120"])
    parser.add_argument("--require-fod", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--transform-timeout-sec", type=int, default=900)
    parser.add_argument("--connectome-timeout-sec", type=int, default=2700)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    run_root = args.run_root or args.route13_root
    run_root.mkdir(parents=True, exist_ok=True)
    subjects = {sid.strip() for sid in args.subjects if sid.strip()}
    rows = target_rows(args.route13_root, subjects, require_fod=args.require_fod)
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]
    done = finished_sids(run_root, retry_failed=args.retry_failed)
    rows = [row for row in rows if row["sid"] not in done]
    tag = args.route1_root.name
    write_csv(run_root / "route13_connectome_queue.csv", rows)
    print(f"AAL3 Route 13 native-T1 connectome probe | {utc()}", flush=True)
    print(f"run root: {run_root}", flush=True)
    print(f"eligible pending: {len(rows)} | require_fod={args.require_fod} | execute={args.execute}", flush=True)
    if rows:
        print("preview: " + ", ".join(row["sid"] for row in rows[:10]), flush=True)
    if not args.execute or not rows:
        return 0

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        futures = [
            pool.submit(
                run_one,
                row=row,
                tag=tag,
                run_root=run_root,
                route13_root=args.route13_root,
                variants=args.variants,
                mrtrix_threads=args.mrtrix_threads,
                transform_timeout_sec=args.transform_timeout_sec,
                connectome_timeout_sec=args.connectome_timeout_sec,
            )
            for row in rows
        ]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            append_or_replace_csv(run_root / "batch_results.csv", result)
            print(
                f"{result.get('sid')} {result.get('status')} "
                f"dens {result.get('baseline_density')}->{result.get('best_density')} "
                f"zero {result.get('baseline_zero_rows')}->{result.get('best_zero_rows')}",
                flush=True,
            )

    write_csv(run_root / "batch_results_incremental.csv", results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
