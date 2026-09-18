#!/usr/bin/env python3
"""Build lane-specific SC matrix repair canary manifests.

This is read-only with respect to production parcellations and connectomes.
It chooses small, evidence-backed canary sets from the existing whole-matrix QC
CSV files so we can test lane-specific repairs before any batch repair.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path


ROOT = Path("/home/ec2-user/exp")
QC_ROOT = ROOT / "data/derivatives/qc/sc_matrix_qc"
LANE_ROOT = QC_ROOT / "lane_canaries"

REGISTRATION_PRIORITY = (
    "<SUBJECT>_I<IMAGEID>",
    "<SUBJECT>_I<IMAGEID>",
)
ASSIGNMENT_PRIORITY = (
    "<SUBJECT>_I<IMAGEID>",
)


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    if not fields:
        fields = ["empty"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_txt(path: Path, sids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(sids) + ("\n" if sids else ""), encoding="utf-8")


def num(row: dict[str, str], key: str, default: float = 0.0) -> float:
    raw = row.get(key, "")
    try:
        val = float(raw)
    except Exception:
        return default
    if math.isnan(val):
        return default
    return val


def sid_key(row: dict[str, str]) -> str:
    return row.get("sid") or row.get("subject_id") or ""


def dedupe_by_sid(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[str] = set()
    out: list[dict[str, str]] = []
    for row in rows:
        sid = sid_key(row)
        if not sid or sid in seen:
            continue
        seen.add(sid)
        out.append(row)
    return out


def registration_score(row: dict[str, str]) -> tuple[float, float, float, float]:
    """Higher score means a better severe registration/parcellation canary."""
    return (
        num(row, "n_label_tiny_or_absent_zero_nodes") + num(row, "n_poor_overlap_zero_nodes"),
        num(row, "n_unexpected_zero_rows"),
        -num(row, "density", 1.0),
        -num(row, "largest_component_frac", 1.0),
    )


def assignment_score(row: dict[str, str]) -> tuple[float, float, float]:
    return (
        num(row, "n_coverage_ok_zero_nodes") + num(row, "n_zero_assignment_zero_nodes"),
        num(row, "n_unexpected_zero_rows"),
        -num(row, "density", 1.0),
    )


def pick_with_priority(
    rows: list[dict[str, str]],
    *,
    limit: int,
    priority: tuple[str, ...],
    score_key,
) -> list[dict[str, str]]:
    by_sid = {sid_key(row): row for row in rows if sid_key(row)}
    picked: list[dict[str, str]] = []
    used: set[str] = set()
    for sid in priority:
        if sid in by_sid and sid not in used:
            picked.append(by_sid[sid])
            used.add(sid)
    for row in sorted(rows, key=score_key, reverse=True):
        sid = sid_key(row)
        if not sid or sid in used:
            continue
        picked.append(row)
        used.add(sid)
        if len(picked) >= limit:
            break
    return picked[:limit]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default=time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()))
    parser.add_argument("--registration-limit", type=int, default=5)
    parser.add_argument("--assignment-limit", type=int, default=5)
    parser.add_argument("--hold-limit", type=int, default=20)
    parser.add_argument(
        "--exclude-manifest",
        action="append",
        type=Path,
        default=[],
        help="Previous lane_canary_manifest.csv to exclude from the new pick. Repeatable.",
    )
    args = parser.parse_args()

    root = LANE_ROOT / args.tag
    root.mkdir(parents=True, exist_ok=True)

    post_rows = read_rows(QC_ROOT / "post_parcellation_repair_targets.csv")
    bbr_rows = read_rows(QC_ROOT / "bbr_registration_repair_targets.csv")
    assignment_rows = read_rows(QC_ROOT / "assignment_or_tracks_repair_targets.csv")
    decisions = read_rows(QC_ROOT / "sc_matrix_qc_decisions.csv")

    exclude_sids: set[str] = set()
    for manifest_path in args.exclude_manifest:
        for row in read_rows(manifest_path):
            sid = sid_key(row)
            if sid and "canary" in str(row.get("lane", "")):
                exclude_sids.add(sid)

    registration_pool = [row for row in dedupe_by_sid(bbr_rows + post_rows) if sid_key(row) not in exclude_sids]
    registration_pick = pick_with_priority(
        registration_pool,
        limit=args.registration_limit,
        priority=REGISTRATION_PRIORITY,
        score_key=registration_score,
    )
    assignment_pick = pick_with_priority(
        [row for row in dedupe_by_sid(assignment_rows) if sid_key(row) not in exclude_sids],
        limit=args.assignment_limit,
        priority=ASSIGNMENT_PRIORITY,
        score_key=assignment_score,
    )

    hold_rows = [
        row
        for row in decisions
        if str(row.get("analysis_gate", "")).lower() == "include"
        and str(row.get("sc_matrix_qc_status", "")).upper() in {"PASS", "WARN_SPARSE_NODE", "WARN"}
    ][: args.hold_limit]

    manifest: list[dict[str, object]] = []
    for lane, rows in (
        ("registration_parcellation_canary", registration_pick),
        ("assignment_tracks_canary", assignment_pick),
        ("baseline_hold_do_not_touch", hold_rows),
    ):
        for rank, row in enumerate(rows, start=1):
            manifest.append(
                {
                    "lane": lane,
                    "rank": rank,
                    "sid": sid_key(row),
                    "group": row.get("group", ""),
                    "qc_status": row.get("whole_matrix_qc_status") or row.get("sc_matrix_qc_status", ""),
                    "analysis_gate": row.get("analysis_gate", ""),
                    "density": row.get("density", ""),
                    "n_unexpected_zero_rows": row.get("n_unexpected_zero_rows", ""),
                    "n_label_tiny_or_absent_zero_nodes": row.get("n_label_tiny_or_absent_zero_nodes", ""),
                    "n_poor_overlap_zero_nodes": row.get("n_poor_overlap_zero_nodes", ""),
                    "n_coverage_ok_zero_nodes": row.get("n_coverage_ok_zero_nodes", ""),
                    "n_zero_assignment_zero_nodes": row.get("n_zero_assignment_zero_nodes", ""),
                    "touch_policy": "QC_ONLY_DO_NOT_OVERWRITE" if "canary" in lane else "DO_NOT_TOUCH",
                }
            )

    write_csv(root / "lane_canary_manifest.csv", manifest)
    write_txt(root / "registration_parcellation_canary_sids.txt", [sid_key(r) for r in registration_pick])
    write_txt(root / "assignment_tracks_canary_sids.txt", [sid_key(r) for r in assignment_pick])
    write_txt(root / "baseline_hold_do_not_touch_sids.txt", [sid_key(r) for r in hold_rows])
    status = {
        "tag": args.tag,
        "run_root": str(root),
        "created_utc": now_utc(),
        "current_lane": "none",
        "current_sid": "",
        "stage": "manifest_built",
        "registration_total": len(registration_pick),
        "registration_done": 0,
        "assignment_total": len(assignment_pick),
        "assignment_done": 0,
        "runner_done": False,
        "rule": "lane-specific canary only; no production overwrite",
    }
    (root / "lane_canary_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    (LANE_ROOT / "latest_tag.txt").write_text(args.tag + "\n", encoding="utf-8")

    print(f"Lane canary manifest: {root / 'lane_canary_manifest.csv'}")
    print(f"Registration/parcellation canaries: {len(registration_pick)}")
    print(f"Assignment/tracks canaries: {len(assignment_pick)}")
    print(f"Baseline hold subjects: {len(hold_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
