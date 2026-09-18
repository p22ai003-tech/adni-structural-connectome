#!/usr/bin/env python3
"""Create explicit Route 11B/11C queue roots from Route 10 and Route 11A.

Route 11B and 11C are downstream corrective lanes, not generic assignment
variants. This script makes those lanes visible to the overview monitor by
writing subjects.csv, route_queue.csv, and status.json for each route.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any


RUN_PARENT = Path("/data/derivatives/qc/sc_matrix_qc/aal3_force_connectome_ad_batch")
DEFAULT_R1 = RUN_PARENT / "ad_existing_tracks_20260529T055752Z"
R11B_SUFFIX = "_route11b_act_sift2_track_coverage_rerun"
R11C_SUFFIX = "_route11c_zero_row_coverage_rescue"


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


def groups(route1_root: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for row in read_csv(route1_root / "subjects.csv"):
        sid = row.get("sid") or row.get("subject") or row.get("subject_id") or ""
        if sid:
            out[sid] = (row.get("group") or "AD").strip().upper() or "AD"
    return out


def row_base(row: dict[str, str], group_map: dict[str, str], source: str, target_lane: str, reason: str) -> dict[str, Any]:
    sid = row.get("sid", "")
    return {
        "sid": sid,
        "group": group_map.get(sid, "AD"),
        "target_lane": target_lane,
        "source": source,
        "reason": reason,
        "issue_class": row.get("issue_class", ""),
        "source_labels": row.get("source_labels", ""),
        "best_density": row.get("best_density", ""),
        "best_zero_rows": row.get("best_zero_rows", ""),
        "both_assigned_fraction": row.get("both_assigned_fraction", ""),
        "one_unassigned_fraction": row.get("one_unassigned_fraction", ""),
        "both_unassigned_fraction": row.get("both_unassigned_fraction", ""),
        "next_lane": row.get("next_lane", ""),
    }


def merge_nonempty(base: dict[str, str], update: dict[str, str]) -> dict[str, str]:
    merged = dict(base)
    for key, value in update.items():
        if value not in {"", None}:
            merged[key] = value
    return merged


def build_queues(
    route1_root: Path,
    route10_root: Path,
    route11a_root: Path,
    route11b_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    group_map = groups(route1_root)
    r10_rows = [row for row in read_csv(route10_root / "route10_triage.csv") if row.get("sid")]
    r11a_rows = [row for row in read_csv(route11a_root / "route_qc_decisions.csv") if row.get("sid")]
    r11b_queue = {row["sid"]: row for row in read_csv(route11b_root / "route_queue.csv") if row.get("sid")}
    r11b_rows = [row for row in read_csv(route11b_root / "route_qc_decisions.csv") if row.get("sid")]
    r11b_results = [row for row in read_csv(route11b_root / "batch_results.csv") if row.get("sid")]

    r11b: dict[str, dict[str, Any]] = {}
    r11c: dict[str, dict[str, Any]] = {}

    for row in r10_rows:
        lane = row.get("next_lane", "")
        issue = row.get("issue_class", "")
        if lane.startswith("R11B"):
            r11b[row["sid"]] = row_base(row, group_map, "route10", "R11B", row.get("next_lane_reason", lane))
        elif lane.startswith("R11C") or issue == "local_zero_row_problem":
            r11c[row["sid"]] = row_base(row, group_map, "route10", "R11C", row.get("next_lane_reason", lane))
        elif lane == "R11_review_then_select_registration_or_tractography":
            r11b[row["sid"]] = row_base(
                row,
                group_map,
                "route10_review",
                "R11B",
                "review-lane subject: choose registration repair versus ACT/SIFT2 rerun after overlay review",
            )

    for row in r11a_rows:
        if row.get("qc_decision") != "TRY_ANOTHER_ROUTE":
            continue
        labels = inum(row.get("source_labels"), 0)
        zero = inum(row.get("best_zero_rows"), 9999)
        both = fnum(row.get("both_assigned_fraction"), float("nan"))
        sid = row["sid"]
        if labels >= 160 and both == both and both >= 0.85 and zero > 15:
            r11c[sid] = row_base(
                row,
                group_map,
                "route11a_failed",
                "R11C",
                "R11A repaired assignment but zero rows remain high; queue targeted zero-row coverage rescue",
            )
            r11b.pop(sid, None)
        else:
            r11b[sid] = row_base(
                row,
                group_map,
                "route11a_failed",
                "R11B",
                "R11A failed to find a production-safe registration candidate; queue review/tractography-repair lane",
            )

    for row in r11b_rows:
        if row.get("qc_decision") != "TRY_ANOTHER_ROUTE":
            continue
        sid = row["sid"]
        enriched = merge_nonempty(r11b_queue.get(sid, {}), row)
        r11c[sid] = row_base(
            enriched,
            group_map,
            "route11b_failed_or_weak",
            "R11C",
            "R11B failed final QC; queue targeted zero-row/coverage rescue as the next corrective lane",
        )

    for row in r11b_results:
        if row.get("status") not in {"failed", "timeout"}:
            continue
        sid = row["sid"]
        enriched = merge_nonempty(r11b_queue.get(sid, {}), row)
        r11c[sid] = row_base(
            enriched,
            group_map,
            "route11b_failed_or_timeout",
            "R11C",
            "R11B did not complete successfully; queue targeted zero-row/coverage rescue as the next corrective lane",
        )

    return sorted(r11b.values(), key=lambda r: r["sid"]), sorted(r11c.values(), key=lambda r: r["sid"])


def write_queue(root: Path, lane: str, rows: list[dict[str, Any]], route1_root: Path, upstream_roots: dict[str, Path]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    write_csv(root / "route_queue.csv", rows)
    write_csv(root / "subjects.csv", [{"sid": row["sid"], "group": row.get("group", "AD")} for row in rows])
    status_path = root / "status.json"
    existing = read_json(status_path)
    if existing.get("phase") in {"running", "complete"}:
        existing.update(
            {
                "lane": lane,
                "route1_root": str(route1_root),
                "upstream_roots": {key: str(value) for key, value in upstream_roots.items()},
                "queued_total": len(rows),
                "last_queue_materialized_utc": utc(),
                "updated_utc": utc(),
            }
        )
        write_json(status_path, existing)
    else:
        write_json(
            status_path,
            {
                "phase": "queued",
                "lane": lane,
                "route1_root": str(route1_root),
                "upstream_roots": {key: str(value) for key, value in upstream_roots.items()},
                "total": len(rows),
                "completed": 0,
                "failed": 0,
                "active": [],
                "updated_utc": utc(),
            },
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route1-root", type=Path, default=DEFAULT_R1)
    parser.add_argument("--route10-root", type=Path, default=None)
    parser.add_argument("--route11a-root", type=Path, default=None)
    args = parser.parse_args()

    route1 = args.route1_root
    route10 = args.route10_root or route1.parent / f"{route1.name}_route10_endpoint_overlay_triage"
    route11a = args.route11a_root or route1.parent / f"{route1.name}_route11a_space_affine_registration_repair"
    r11b_root = route1.parent / f"{route1.name}{R11B_SUFFIX}"
    r11c_root = route1.parent / f"{route1.name}{R11C_SUFFIX}"

    r11b, r11c = build_queues(route1, route10, route11a, r11b_root)
    upstream = {"route10": route10, "route11a": route11a, "route11b": r11b_root}
    write_queue(r11b_root, "R11B_ACT_SIFT2_or_review_repair", r11b, route1, upstream)
    write_queue(r11c_root, "R11C_targeted_zero_row_coverage_rescue", r11c, route1, upstream)
    summary = {
        "generated_utc": utc(),
        "route1_root": str(route1),
        "r11b_root": str(r11b_root),
        "r11c_root": str(r11c_root),
        "r11b_queued": len(r11b),
        "r11c_queued": len(r11c),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
