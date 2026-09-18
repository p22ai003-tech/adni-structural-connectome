#!/usr/bin/env python3
"""Route 10 diagnostic triage for AAL3 rescue subjects after Route 9.

This route does not promote or overwrite production. It generates the evidence
needed to choose the next corrective lane:

* registration / affine / space-contract repair
* ACT/SIFT2 tractography coverage rerun
* targeted zero-row coverage rescue

The key added signal is an endpoint-density map from the existing tracks in the
same grid as the tested AAL3 parcellation. If labels survive but endpoints miss
the labels, the failure is not solved by another connectome assignment flag.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


ROOT = Path("/home/ec2-user/exp")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.scforge.live.generate_aal3_overlay_qc import (  # noqa: E402
    crop_bounds,
    find_background_path,
    find_label_path,
    generate_subject,
    robust_bg,
    slice2d,
)
from scripts.scforge.live.run_sc_aal3_source_contract_probe import (  # noqa: E402
    discover_inputs,
    write_csv,
    write_json,
)


RUN_PARENT = Path("/data/derivatives/qc/sc_matrix_qc/aal3_force_connectome_ad_batch")
DEFAULT_ROUTE1_ROOT = RUN_PARENT / "ad_existing_tracks_20260529T055752Z"
DEFAULT_ROUTE9_ROOT = DEFAULT_ROUTE1_ROOT.parent / f"{DEFAULT_ROUTE1_ROOT.name}_route9_route8_assignment_retry"
DEFAULT_REPORT_AUDIT = Path("/home/ec2-user/exp/reports/aal3_route_failure_audit/route_failure_investigation.csv")
TCKMAP = Path("/home/ec2-user/mrtrix3/bin/tckmap")
ROUTE10_SUFFIX = "_route10_endpoint_overlay_triage"
STATUS_LOCK = threading.Lock()


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


def fnum(value: Any, default: float = float("nan")) -> float:
    try:
        if value in {"", None}:
            return default
        return float(value)
    except Exception:
        return default


def inum(value: Any, default: int = 0) -> int:
    try:
        if value in {"", None}:
            return default
        return int(float(value))
    except Exception:
        return default


def safe_ratio(num: float, den: float) -> float:
    if not math.isfinite(num) or not math.isfinite(den) or den <= 0:
        return float("nan")
    return float(num / den)


def update_status(root: Path, **updates: Any) -> None:
    with STATUS_LOCK:
        status = read_json(root / "status.json")
        status.update(updates)
        status["updated_utc"] = utc()
        write_json(root / "status.json", status)


def subject_groups(route1_root: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for row in read_csv(route1_root / "subjects.csv"):
        sid = row.get("sid") or row.get("subject") or row.get("subject_id") or ""
        if sid:
            out[sid] = (row.get("group") or "").strip().upper()
    return out


def failure_audit_rows(path: Path, route1_root: Path) -> dict[str, dict[str, str]]:
    candidates = [
        path,
        route1_root / "route_failure_investigation.csv",
        DEFAULT_REPORT_AUDIT,
    ]
    for candidate in candidates:
        rows = read_csv(candidate)
        if rows:
            return {row.get("sid", ""): row for row in rows if row.get("sid")}
    return {}


def route10_subject_rows(route9_rows: list[dict[str, str]], groups: dict[str, str]) -> list[dict[str, Any]]:
    return [
        {
            "sid": row.get("sid", ""),
            "group": groups.get(row.get("sid", ""), ""),
            "route9_decision": row.get("qc_decision", ""),
        }
        for row in route9_rows
        if row.get("sid")
    ]


def queue_rows(route9_root: Path, audit: dict[str, dict[str, str]], requested: set[str]) -> list[dict[str, str]]:
    rows = [row for row in read_csv(route9_root / "route_qc_decisions.csv") if row.get("sid")]
    if requested:
        rows = [row for row in rows if row["sid"] in requested]
    # Route 10 is for completed/current Route 9 subjects. It includes current
    # keep-production rows because they can still have assignment-space defects.
    rows.sort(key=lambda row: row.get("sid", ""))
    for row in rows:
        row.update({f"audit_{k}": v for k, v in audit.get(row["sid"], {}).items()})
    return rows


def endpoint_density_map(
    *,
    tracks: Path,
    weights: Path | None,
    template: Path,
    out_path: Path,
    threads: int,
    force: bool,
) -> tuple[bool, str]:
    if out_path.exists() and out_path.stat().st_size > 0 and not force:
        return True, "cached"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd: list[str] = [
        str(TCKMAP),
        str(tracks),
        str(out_path),
        "-template",
        str(template),
        "-ends_only",
        "-nthreads",
        str(threads),
        "-quiet",
        "-force",
    ]
    if weights and weights.exists():
        cmd.extend(["-tck_weights_in", str(weights)])
    try:
        proc = subprocess.run(cmd, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    if proc.returncode != 0:
        return False, (proc.stderr or proc.stdout or f"tckmap exited {proc.returncode}")[-800:]
    return True, "ok"


def endpoint_metrics(endpoint_map: Path, parc: Path) -> dict[str, Any]:
    out: dict[str, Any] = {
        "endpoint_map": str(endpoint_map),
        "endpoint_map_read_ok": 0,
        "endpoint_weight_total": "",
        "endpoint_weight_inside_labels": "",
        "endpoint_inside_label_fraction": "",
        "endpoint_positive_voxels": "",
        "endpoint_positive_voxels_inside_labels": "",
    }
    if not endpoint_map.exists() or endpoint_map.stat().st_size <= 0:
        out["endpoint_error"] = "missing_endpoint_map"
        return out
    try:
        endpoints = np.nan_to_num(np.asanyarray(nib.load(str(endpoint_map)).dataobj), nan=0.0, posinf=0.0, neginf=0.0)
        labels = np.rint(np.asanyarray(nib.load(str(parc)).dataobj)).astype(np.int64)
        if endpoints.shape != labels.shape:
            raise ValueError(f"shape mismatch: endpoints {endpoints.shape}, labels {labels.shape}")
        label_mask = labels > 0
        total = float(np.abs(endpoints).sum())
        inside = float(np.abs(endpoints[label_mask]).sum())
        positive = endpoints > 0
        out.update(
            {
                "endpoint_map_read_ok": 1,
                "endpoint_weight_total": total,
                "endpoint_weight_inside_labels": inside,
                "endpoint_inside_label_fraction": safe_ratio(inside, total),
                "endpoint_positive_voxels": int(positive.sum()),
                "endpoint_positive_voxels_inside_labels": int((positive & label_mask).sum()),
            }
        )
    except Exception as exc:
        out["endpoint_error"] = f"{type(exc).__name__}: {exc}"
    return out


def endpoint_overlay(bg: np.ndarray, labels: np.ndarray, endpoints: np.ndarray, axis: int, idx: int) -> np.ndarray:
    b = slice2d(bg, axis, idx)
    lab = slice2d(labels, axis, idx)
    ep = slice2d(endpoints, axis, idx)
    rgb = np.dstack([b, b, b])

    lab_mask = lab > 0
    if lab_mask.any():
        edge = lab_mask ^ (
            np.roll(lab_mask, 1, 0)
            & np.roll(lab_mask, -1, 0)
            & np.roll(lab_mask, 1, 1)
            & np.roll(lab_mask, -1, 1)
        )
        rgb[edge, :] = [0.0, 0.95, 0.2]

    finite = ep[np.isfinite(ep) & (ep > 0)]
    if finite.size:
        hi = float(np.percentile(finite, 99))
        if hi <= 0:
            hi = float(finite.max())
        norm = np.clip(ep / max(hi, 1e-12), 0, 1)
        rgb[..., 0] = np.maximum(rgb[..., 0], norm)
        rgb[..., 1] = np.maximum(rgb[..., 1], 0.25 * norm)
        rgb[..., 2] *= 1 - 0.65 * norm
    return rgb


def generate_endpoint_overlay_png(sid: str, parc: Path, bg_path: Path, endpoint_map: Path, out_dir: Path) -> Path:
    labels = np.rint(np.asanyarray(nib.load(str(parc)).dataobj)).astype(np.int64)
    bg = robust_bg(np.asanyarray(nib.load(str(bg_path)).dataobj))
    endpoints = np.nan_to_num(np.asanyarray(nib.load(str(endpoint_map)).dataobj), nan=0.0, posinf=0.0, neginf=0.0)
    mask = (labels > 0) | (endpoints > 0)
    bounds = crop_bounds(mask)
    labels_c = labels[bounds]
    bg_c = bg[bounds]
    ep_c = endpoints[bounds]
    pts = np.argwhere(mask[bounds])
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
        ax.imshow(endpoint_overlay(bg_c, labels_c, ep_c, axis, idx), origin="lower")
        ax.set_title(f"{name} {idx}", fontsize=9)
        ax.axis("off")
    fig.suptitle(f"{sid} endpoint cloud vs AAL3 labels | orange=endpoints | green=label edge", fontsize=11)
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{sid}_route10_endpoint_overlay.png"
    fig.tight_layout()
    fig.savefig(png, facecolor="white")
    plt.close(fig)
    return png


def next_lane(row: dict[str, str], metrics: dict[str, Any]) -> tuple[str, str]:
    issue = row.get("audit_issue_class") or row.get("issue_class") or ""
    labels = inum(row.get("source_labels"), 0)
    zero = inum(row.get("best_zero_rows"), 9999)
    both = fnum(row.get("both_assigned_fraction"))
    neither = fnum(row.get("both_unassigned_fraction"))
    endpoint_inside = fnum(metrics.get("endpoint_inside_label_fraction"))

    if issue == "space_contract_break" or (math.isfinite(both) and both <= 0.05 and math.isfinite(neither) and neither >= 0.90):
        return (
            "R11A_space_affine_registration_repair",
            "labels survive but essentially all endpoints are unassigned; repair grid/affine/transform contract before rerunning tractography",
        )
    if labels < 160:
        return (
            "R11A_registration_label_rebuild",
            "label survival is below target; fix atlas-to-B0 registration and label interpolation first",
        )
    if issue == "endpoint_assignment_space_mismatch":
        if math.isfinite(endpoint_inside) and endpoint_inside < 0.35:
            return (
                "R11A_space_affine_registration_repair",
                "endpoint-density cloud mostly misses AAL3 labels, so route needs space/affine overlay repair",
            )
        return (
            "R11B_ACT_SIFT2_endpoint_termination_rerun",
            "labels survive but endpoint assignment remains poor; if visual overlay is anatomically good, rerun ACT/SIFT2 with validated 5TT/GMWMI",
        )
    if issue == "local_zero_row_problem":
        return (
            "R11C_targeted_zero_row_coverage_rescue",
            "matrix is plausible but specific AAL3 rows remain empty; prioritize zero-row ROI coverage rescue",
        )
    if issue == "track_coverage_limited" or (math.isfinite(both) and both >= 0.85 and zero > 15):
        return (
            "R11B_ACT_SIFT2_track_coverage_rerun",
            "assignment is acceptable but too many AAL3 ROIs have no streamlines; rerun tractography for coverage rather than another assignment route",
        )
    if issue == "candidate_not_better":
        return (
            "keep_current_review_only",
            "tested route is worse than current production; keep production and only continue if visual/assignment QC remains open",
        )
    return (
        "R11_review_then_select_registration_or_tractography",
        "mixed evidence; use generated overlays plus endpoint fraction to choose registration repair or tractography rerun",
    )


def run_one(row: dict[str, str], run_root: Path, route9_root: Path, out_root: Path, threads: int, force: bool) -> dict[str, Any]:
    sid = row["sid"]
    subject_dir = out_root / "subjects" / sid
    subject_dir.mkdir(parents=True, exist_ok=True)
    status_file = subject_dir / "status.json"
    if status_file.exists() and not force:
        cached = read_json(status_file)
        if cached.get("phase") == "complete":
            return cached.get("result", {})

    update_status(subject_dir, sid=sid, phase="starting")
    inputs = discover_inputs(sid)
    probe_root = Path(row.get("probe_root") or "")
    if not probe_root.exists():
        raise FileNotFoundError(f"Route9 probe_root missing for {sid}: {probe_root}")

    update_status(subject_dir, sid=sid, phase="resolve_parcellation")
    parc = find_label_path(row, probe_root)
    bg_path = find_background_path(sid, probe_root, parc)

    update_status(subject_dir, sid=sid, phase="atlas_overlay")
    overlay_manifest_row: dict[str, Any] = {}
    overlay_error = ""
    try:
        overlay_manifest_row = generate_subject(row, subject_dir, route9_root, {})
    except Exception as exc:
        overlay_error = f"{type(exc).__name__}: {exc}"

    tracks = Path(inputs.get("tracks") or "")
    weights = Path(inputs.get("weights") or "") if inputs.get("weights") else None
    if not tracks.exists():
        raise FileNotFoundError(f"tracks missing for {sid}: {tracks}")

    endpoint_map = subject_dir / f"{sid}_route10_endpoint_density.nii.gz"
    update_status(subject_dir, sid=sid, phase="endpoint_tckmap")
    ok, message = endpoint_density_map(
        tracks=tracks,
        weights=weights,
        template=parc,
        out_path=endpoint_map,
        threads=threads,
        force=force,
    )
    if not ok:
        raise RuntimeError(message)

    update_status(subject_dir, sid=sid, phase="endpoint_metrics")
    metrics = endpoint_metrics(endpoint_map, parc)
    endpoint_png = ""
    endpoint_overlay_error = ""
    if int(metrics.get("endpoint_map_read_ok") or 0) == 1:
        update_status(subject_dir, sid=sid, phase="endpoint_overlay")
        try:
            endpoint_png = str(generate_endpoint_overlay_png(sid, parc, bg_path, endpoint_map, subject_dir))
        except Exception as exc:
            endpoint_overlay_error = f"{type(exc).__name__}: {exc}"

    lane, lane_reason = next_lane(row, metrics)
    result: dict[str, Any] = {
        "sid": sid,
        "status": "ok",
        "route10_decision": "DIAGNOSTIC_COMPLETE",
        "next_lane": lane,
        "next_lane_reason": lane_reason,
        "route9_decision": row.get("qc_decision", ""),
        "issue_class": row.get("audit_issue_class") or row.get("issue_class") or "",
        "source_labels": row.get("source_labels", ""),
        "best_density": row.get("best_density", ""),
        "density_delta": row.get("density_delta", ""),
        "best_zero_rows": row.get("best_zero_rows", ""),
        "zero_rows_delta": row.get("zero_rows_delta", ""),
        "both_assigned_fraction": row.get("both_assigned_fraction", ""),
        "one_unassigned_fraction": row.get("one_unassigned_fraction", ""),
        "both_unassigned_fraction": row.get("both_unassigned_fraction", ""),
        "parcellation": str(parc),
        "background": str(bg_path),
        "endpoint_map": str(endpoint_map),
        "endpoint_overlay_png": endpoint_png,
        "atlas_overlay_png": overlay_manifest_row.get("overlay_png", ""),
        "atlas_overlay_error": overlay_error,
        "endpoint_overlay_error": endpoint_overlay_error,
        "tckmap_status": message,
        "tracks": str(tracks),
        "weights": str(weights) if weights else "",
        **metrics,
    }
    write_json(subject_dir / "route10_triage_subject.json", result)
    write_json(status_file, {"sid": sid, "phase": "complete", "updated_utc": utc(), "result": result})
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route1-root", type=Path, default=DEFAULT_ROUTE1_ROOT)
    parser.add_argument("--route9-root", type=Path, default=DEFAULT_ROUTE9_ROOT)
    parser.add_argument("--out-root", type=Path, default=None)
    parser.add_argument("--failure-audit", type=Path, default=DEFAULT_REPORT_AUDIT)
    parser.add_argument("--subjects", nargs="*", default=[])
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--mrtrix-threads", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    route1_root = args.route1_root
    route9_root = args.route9_root
    out_root = args.out_root or route1_root.parent / f"{route1_root.name}{ROUTE10_SUFFIX}"
    out_root.mkdir(parents=True, exist_ok=True)

    groups = subject_groups(route1_root)
    audit = failure_audit_rows(args.failure_audit, route1_root)
    rows = queue_rows(route9_root, audit, set(args.subjects))
    write_csv(out_root / "subjects.csv", route10_subject_rows(rows, groups))
    update_status(
        out_root,
        route10_schema="endpoint_overlay_triage_v1",
        phase="running",
        route1_root=str(route1_root),
        route9_root=str(route9_root),
        total=len(rows),
        completed=0,
        failed=0,
        active=[],
        next_route_plan={
            "Route 10": "endpoint-density map plus atlas/B0 overlay triage; no production overwrite",
            "Route 11A": "space/affine/registration repair for high-label but unassigned endpoint failures",
            "Route 11B": "ACT/SIFT2 tractography coverage or endpoint-termination rerun after overlay passes",
            "Route 11C": "targeted zero-row ROI coverage rescue for otherwise plausible matrices",
        },
    )

    completed: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    active: set[str] = set()

    def wrapped(row: dict[str, str]) -> dict[str, Any]:
        sid = row["sid"]
        active.add(sid)
        update_status(out_root, active=sorted(active))
        try:
            result = run_one(row, route1_root, route9_root, out_root, max(1, args.mrtrix_threads), args.force)
            return result
        finally:
            active.discard(sid)
            update_status(out_root, active=sorted(active))

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        future_map = {pool.submit(wrapped, row): row for row in rows}
        for future in as_completed(future_map):
            row = future_map[future]
            try:
                result = future.result()
                completed.append(result)
                manifest.append(
                    {
                        "sid": result.get("sid", ""),
                        "route10_status": result.get("status", ""),
                        "atlas_overlay_png": result.get("atlas_overlay_png", ""),
                        "endpoint_overlay_png": result.get("endpoint_overlay_png", ""),
                        "visual_overlay_status": "generated_pending_review",
                        "next_lane": result.get("next_lane", ""),
                    }
                )
            except Exception as exc:
                err = {
                    "sid": row.get("sid", ""),
                    "status": "failed",
                    "route10_decision": "DIAGNOSTIC_FAILED",
                    "error": f"{type(exc).__name__}: {exc}",
                    "route9_decision": row.get("qc_decision", ""),
                    "issue_class": row.get("audit_issue_class") or row.get("issue_class") or "",
                }
                failed.append(err)
            all_rows = completed + failed
            write_csv(out_root / "batch_results.csv", all_rows)
            write_csv(out_root / "route10_triage.csv", all_rows)
            write_csv(out_root / "visual_overlay_qc_manifest.csv", manifest)
            update_status(out_root, completed=len(completed), failed=len(failed), active=sorted(active))

    lane_counts: dict[str, int] = {}
    issue_counts: dict[str, int] = {}
    for result in completed + failed:
        lane = str(result.get("next_lane") or result.get("route10_decision") or "")
        issue = str(result.get("issue_class") or "unknown")
        lane_counts[lane] = lane_counts.get(lane, 0) + 1
        issue_counts[issue] = issue_counts.get(issue, 0) + 1
    summary = {
        "generated_utc": utc(),
        "route10_schema": "endpoint_overlay_triage_v1",
        "route1_root": str(route1_root),
        "route9_root": str(route9_root),
        "out_root": str(out_root),
        "total": len(rows),
        "completed": len(completed),
        "failed": len(failed),
        "lane_counts": lane_counts,
        "issue_counts": issue_counts,
    }
    write_json(out_root / "route10_summary.json", summary)
    update_status(out_root, phase="complete", completed=len(completed), failed=len(failed), active=[], summary=summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"wrote {out_root / 'route10_triage.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
