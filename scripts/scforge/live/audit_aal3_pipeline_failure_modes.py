#!/usr/bin/env python3
"""Audit unresolved AAL3 rescue cases by pipeline failure mode.

This is intentionally diagnostic. It does not promote, delete, or rerun data.
It combines the route-failure ledger with MRtrix input checks and tckgen logs so
the next route can be selected from the actual bottleneck rather than from the
route number alone.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any


RUN_PARENT = Path("/data/derivatives/qc/sc_matrix_qc/aal3_force_connectome_ad_batch")
DEFAULT_R1 = RUN_PARENT / "ad_existing_tracks_20260529T055752Z"
FOD_ROOT = Path("/data/derivatives/fod")
QC_ROOT = Path("/data/derivatives/qc/sc_matrix_qc")
MRTRIX_BIN = Path(os.environ.get("MRTRIX_BIN", "/home/ec2-user/mrtrix3/bin"))

TCKGEN_RE = re.compile(
    r"(?P<seeds>\d+)\s+seeds,\s+(?P<streamlines>\d+)\s+streamlines,\s+(?P<selected>\d+)\s+selected"
)


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size <= 0:
        return []
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        return list(csv.DictReader(handle))


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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def fnum(value: Any, default: float = math.nan) -> float:
    try:
        if value in {"", None}:
            return default
        return float(value)
    except Exception:
        return default


def run_text(cmd: list[str], timeout: int = 45) -> tuple[int, str]:
    try:
        proc = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return 124, ""
    except Exception:
        return 127, ""
    return proc.returncode, (proc.stdout or "").strip()


def mrstats(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False}
    cmd = [
        str(MRTRIX_BIN / "mrstats"),
        str(path),
        "-output",
        "mean",
        "-output",
        "min",
        "-output",
        "max",
        "-output",
        "count",
    ]
    rc, out = run_text(cmd)
    parts = out.split()
    row: dict[str, Any] = {"exists": True, "mrstats_rc": rc}
    if rc == 0 and len(parts) >= 4:
        row.update(
            {
                "mean": fnum(parts[0]),
                "min": fnum(parts[1]),
                "max": fnum(parts[2]),
                "count": fnum(parts[3]),
            }
        )
    return row


def mrcalc_nonzero_fraction(path: Path) -> float:
    if not path.exists():
        return math.nan
    cmd = (
        f"{MRTRIX_BIN / 'mrcalc'} {path} 0 -gt - | "
        f"{MRTRIX_BIN / 'mrstats'} - -output mean"
    )
    try:
        proc = subprocess.run(["/bin/bash", "-lc", cmd], text=True, capture_output=True, timeout=60)
    except Exception:
        return math.nan
    if proc.returncode != 0:
        return math.nan
    return fnum((proc.stdout or "").split()[0] if proc.stdout.split() else "")


def tckgen_logs_for_sid(sid: str) -> list[Path]:
    roots = [
        QC_ROOT / "aal3_route11b_act_sift2_track_coverage_rerun_probe",
        QC_ROOT / "aal3_route11c_zero_row_coverage_rescue_probe",
        QC_ROOT / "aal3_route12a_tckgen_fallback_rescue_probe",
        QC_ROOT / "aal3_route12b_zero_row_targeted_rerun_probe",
        QC_ROOT / "aal3_route14_step7_rebuilt_track_coverage_probe",
        QC_ROOT / "aal3_route15_step7_dense_track_coverage_probe",
        QC_ROOT / "aal3_route18_gmwmi_dense_track_coverage_probe",
        QC_ROOT / "aal3_route20_zero_roi_gmwmi_endpoint_probe",
    ]
    logs: list[Path] = []
    for root in roots:
        if root.exists():
            logs.extend(sorted(root.glob(f"*{sid}*/logs/tckgen*.log")))
    return logs


def parse_tckgen_logs(sid: str) -> dict[str, Any]:
    logs = tckgen_logs_for_sid(sid)
    max_seeds = 0
    max_streamlines = 0
    max_selected = 0
    killed = 0
    zero_selected_killed = 0
    modes: set[str] = set()
    latest: dict[str, Any] = {}
    for path in logs:
        text = path.read_text(encoding="utf-8", errors="replace")[-20000:]
        if "SIGTERM" in text or "Terminated by kill command" in text:
            killed += 1
        mode_match = re.search(r"tckgen_([A-Za-z0-9]+)_", path.name)
        if mode_match:
            modes.add(mode_match.group(1))
        matches = list(TCKGEN_RE.finditer(text))
        if not matches:
            continue
        match = matches[-1]
        seeds = int(match.group("seeds"))
        streamlines = int(match.group("streamlines"))
        selected = int(match.group("selected"))
        if selected == 0 and ("SIGTERM" in text or "Terminated by kill command" in text):
            zero_selected_killed += 1
        if selected >= max_selected:
            latest = {
                "latest_tckgen_log": str(path),
                "latest_tckgen_seeds": seeds,
                "latest_tckgen_streamlines": streamlines,
                "latest_tckgen_selected": selected,
            }
        max_seeds = max(max_seeds, seeds)
        max_streamlines = max(max_streamlines, streamlines)
        max_selected = max(max_selected, selected)
    selected_per_million_seeds = math.nan
    if max_seeds > 0:
        selected_per_million_seeds = max_selected / max_seeds * 1_000_000.0
    out: dict[str, Any] = {
        "tckgen_logs_seen": len(logs),
        "tckgen_killed_logs": killed,
        "tckgen_zero_selected_killed_logs": zero_selected_killed,
        "tckgen_modes_seen": ",".join(sorted(modes)),
        "tckgen_max_seeds": max_seeds,
        "tckgen_max_streamlines": max_streamlines,
        "tckgen_max_selected": max_selected,
        "tckgen_selected_per_million_seeds": selected_per_million_seeds,
    }
    out.update(latest)
    return out


def subject_input_stats(sid: str) -> dict[str, Any]:
    fod_dir = FOD_ROOT / sid
    gmwmi = mrstats(fod_dir / "gmwmi.mif")
    five_tt = mrstats(fod_dir / "5tt_b0.mif")
    # wmfod.mif can be large; existence is enough for this routing audit. The
    # actionable signal here is GMWMI/5TT seed quality plus tckgen efficiency.
    wmfod_path = fod_dir / "wmfod.mif"
    return {
        "fod_dir_exists": fod_dir.exists(),
        "gmwmi_exists": gmwmi.get("exists", False),
        "gmwmi_mean": gmwmi.get("mean", math.nan),
        "gmwmi_max": gmwmi.get("max", math.nan),
        "gmwmi_voxel_count": gmwmi.get("count", math.nan),
        "gmwmi_nonzero_fraction": mrcalc_nonzero_fraction(fod_dir / "gmwmi.mif"),
        "five_tt_exists": five_tt.get("exists", False),
        "five_tt_min": five_tt.get("min", math.nan),
        "five_tt_max": five_tt.get("max", math.nan),
        "five_tt_mean": five_tt.get("mean", math.nan),
        "wmfod_exists": wmfod_path.exists(),
        "mask_mif_exists": (fod_dir / "mask.mif").exists(),
    }


def recommend(row: dict[str, Any]) -> tuple[str, str]:
    issue = str(row.get("issue_class", ""))
    labels = fnum(row.get("current_source_labels"))
    density = fnum(row.get("current_best_density"), 0.0)
    zero_rows = fnum(row.get("current_best_zero_rows"), 9999.0)
    both = fnum(row.get("both_assigned_fraction"), 0.0)
    gmwmi_mean = fnum(row.get("gmwmi_mean"), 0.0)
    gmwmi_nonzero = fnum(row.get("gmwmi_nonzero_fraction"), 0.0)
    selected = fnum(row.get("tckgen_max_selected"), 0.0)
    zero_selected_killed = fnum(row.get("tckgen_zero_selected_killed_logs"), 0.0)
    killed = fnum(row.get("tckgen_killed_logs"), 0.0)

    if issue in {"space_contract_break", "endpoint_assignment_space_mismatch"} or (
        labels >= 160 and both < 0.50
    ):
        return (
            "registration_space_contract_repair",
            "Labels survive but streamlines do not assign to labels; inspect affine/grid/endpoint overlay before more tractography.",
        )
    if not row.get("gmwmi_exists") or not row.get("five_tt_exists") or not row.get("wmfod_exists"):
        return (
            "rebuild_missing_fod_5tt_gmwmi",
            "Required ACT/FOD inputs are missing; rerunning tckgen routes cannot be reliable.",
        )
    if zero_selected_killed > 0 or (killed > 0 and selected < 1000):
        return (
            "rebuild_5tt_gmwmi_or_seed_mask",
            "tckgen is spending huge seeds with near-zero selected streamlines; fix GMWMI/5TT seeding before another route.",
        )
    if gmwmi_mean < 0.01 or gmwmi_nonzero < 0.02:
        return (
            "rebuild_5tt_gmwmi_preflight",
            "GMWMI is very thin/sparse; regenerate 5TT/GMWMI and validate seed mask coverage first.",
        )
    if labels >= 160 and both >= 0.85 and (density < 0.10 or zero_rows > 50):
        return (
            "true_high_density_act_sift2_rerun",
            "Atlas and assignment are mostly valid, but streamline coverage is low; run a real higher-density ACT/SIFT2 rerun, not 1.5M fallback.",
        )
    if labels >= 160 and both >= 0.85 and zero_rows > 15:
        return (
            "targeted_zero_row_roi_coverage",
            "Most matrix contract checks pass; target the specific zero-row ROIs and keep review open.",
        )
    if issue == "probe_execution_failure":
        return (
            "isolate_probe_failure",
            "Probe failed before usable metrics; rerun one subject/variant with detailed logs before queueing more routes.",
        )
    return (
        "manual_overlay_then_route_specific_repair",
        "Mixed evidence; inspect overlay, zero-row map, and endpoint cloud before selecting the next route.",
    )


def make_report(route1_root: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    lines = [
        "# AAL3 Pipeline Failure-Mode Audit",
        "",
        f"Generated UTC: {summary['generated_utc']}",
        f"Run root: `{route1_root}`",
        "",
        "## Main Finding",
        "",
        (
            "The unresolved cases are no longer mostly an AAL3 label-survival problem. "
            "Many subjects have high label survival and good endpoint assignment, but still "
            "show sparse matrices or many zero rows. That points to tract coverage and/or "
            "GMWMI/5TT seeding quality rather than another atlas-transform tweak."
        ),
        "",
        "## Recommended Next-Step Counts",
    ]
    for key, count in summary["recommendation_counts"].items():
        lines.append(f"- {key}: {count}")
    lines.extend(["", "## Important Automation Change", ""])
    lines.append(
        "Do not route all `track_coverage_limited`, `sparse_matrix_persistent`, and "
        "`probe_execution_failure` cases into the same 1.5M Route 12A fallback. "
        "Preflight GMWMI/5TT and tckgen seed efficiency first, then choose either "
        "GMWMI rebuild, true high-density ACT/SIFT2 rerun, registration repair, or "
        "targeted zero-row rescue."
    )
    lines.extend(["", "## Example Rows", ""])
    lines.append(
        "| Subject | Issue | Labels | Density | Zero rows | Both assigned | GMWMI mean | GMWMI nz frac | Tckgen selected | Recommendation |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for row in rows[:40]:
        lines.append(
            "| {sid} | {issue} | {labels} | {density:.4f} | {zero:.0f} | {both:.3f} | "
            "{gmwmi_mean:.5f} | {gmwmi_nz:.4f} | {selected:.0f} | {rec} |".format(
                sid=row.get("sid", ""),
                issue=row.get("issue_class", ""),
                labels=fnum(row.get("current_source_labels"), 0.0),
                density=fnum(row.get("current_best_density"), 0.0),
                zero=fnum(row.get("current_best_zero_rows"), 9999.0),
                both=fnum(row.get("both_assigned_fraction"), 0.0),
                gmwmi_mean=fnum(row.get("gmwmi_mean"), 0.0),
                gmwmi_nz=fnum(row.get("gmwmi_nonzero_fraction"), 0.0),
                selected=fnum(row.get("tckgen_max_selected"), 0.0),
                rec=row.get("recommended_next_step", ""),
            )
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route1-root", type=Path, default=DEFAULT_R1)
    parser.add_argument("--include-solved", action="store_true")
    args = parser.parse_args()

    source_rows = read_csv(args.route1_root / "route_failure_investigation.csv")
    if not source_rows:
        raise SystemExit(f"missing investigation CSV: {args.route1_root / 'route_failure_investigation.csv'}")

    out_rows: list[dict[str, Any]] = []
    for src in source_rows:
        if not args.include_solved and src.get("state") == "solved_full_qc_production":
            continue
        sid = src.get("sid", "")
        if not sid:
            continue
        row: dict[str, Any] = dict(src)
        row.update(subject_input_stats(sid))
        row.update(parse_tckgen_logs(sid))
        rec, rationale = recommend(row)
        row["recommended_next_step"] = rec
        row["recommendation_rationale"] = rationale
        out_rows.append(row)

    out_rows.sort(
        key=lambda row: (
            row.get("recommended_next_step", ""),
            -fnum(row.get("current_best_zero_rows"), 0.0),
            fnum(row.get("current_best_density"), 0.0),
            row.get("sid", ""),
        )
    )

    summary = {
        "generated_utc": utc(),
        "route1_root": str(args.route1_root),
        "subjects_audited": len(out_rows),
        "issue_counts": dict(Counter(str(row.get("issue_class", "")) for row in out_rows)),
        "recommendation_counts": dict(Counter(str(row.get("recommended_next_step", "")) for row in out_rows)),
    }

    write_csv(args.route1_root / "pipeline_failure_mode_audit.csv", out_rows)
    write_json(args.route1_root / "pipeline_failure_mode_audit_summary.json", summary)
    report = make_report(args.route1_root, out_rows, summary)
    (args.route1_root / "pipeline_failure_mode_audit.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
