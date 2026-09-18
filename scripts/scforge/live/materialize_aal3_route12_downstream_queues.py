#!/usr/bin/env python3
"""Create Route 12 queues from the latest unresolved AAL3 investigation.

Route 12 is the explicit continuation after R11C. It keeps weak review-pushed
subjects open and routes them by the dominant failure mechanism.

If ``pipeline_failure_mode_audit.csv`` is present, it is used as the stronger
routing signal. Subjects whose current evidence says "rebuild 5TT/GMWMI first"
are intentionally held out of Route 12 instead of being sent through another
low-yield tckgen retry.

* R12A: true high-density ACT/SIFT2 tract-coverage rerun.
* R12B: targeted zero-row coverage rerun.
* R12C: registration/space-contract repair using endpoint-overlap evidence.
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
INVESTIGATOR = Path("/home/ec2-user/exp/scripts/scforge/live/investigate_aal3_route_failures.py")
R12_SUFFIXES = {
    "R12A": "_route12a_tckgen_fallback_rescue",
    "R12B": "_route12b_zero_row_targeted_rerun",
    "R12C": "_route12c_registration_contract_repair",
}
HOLD_FILES = {
    "preflight_rebuild": "route12_preflight_rebuild_hold.csv",
    "probe_isolation": "route12_probe_isolation_hold.csv",
    "manual_review": "route12_manual_review_hold.csv",
}


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


def group_map(route1_root: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for row in read_csv(route1_root / "subjects.csv"):
        sid = row.get("sid") or row.get("subject") or row.get("subject_id") or ""
        if sid:
            out[sid] = (row.get("group") or "AD").strip().upper() or "AD"
    return out


def route10_rows(route1_root: Path) -> dict[str, dict[str, str]]:
    route10 = route1_root.parent / f"{route1_root.name}_route10_endpoint_overlay_triage"
    return {row["sid"]: row for row in read_csv(route10 / "route10_triage.csv") if row.get("sid")}


def latest_investigation(route1_root: Path) -> list[dict[str, str]]:
    path = route1_root / "route_failure_investigation.csv"
    rows = read_csv(path)
    if rows:
        return rows
    # Keep this import-free for shell supervisor use. If the investigation does
    # not exist yet, the supervisor should run the investigator first.
    return []


def latest_pipeline_audit(route1_root: Path) -> dict[str, dict[str, str]]:
    path = route1_root / "pipeline_failure_mode_audit.csv"
    return {row["sid"]: row for row in read_csv(path) if row.get("sid")}


def is_solved(row: dict[str, str]) -> bool:
    return row.get("state") == "solved_full_qc_production" or row.get("final_qc_status") == "solved"


def choose_lane(row: dict[str, str], r10: dict[str, str]) -> tuple[str, str]:
    issue = row.get("issue_class", "")
    if issue in {"probe_execution_failure", "track_coverage_limited", "sparse_matrix_persistent"}:
        return "R12A", "coverage or execution failure after R11; run GMWMI-first tractography fallback"
    if issue == "local_zero_row_problem":
        return "R12B", "specific AAL3 rows remain empty; run targeted zero-row coverage fallback"
    if issue in {"space_contract_break", "endpoint_assignment_space_mismatch", "registration_label_survival"}:
        if r10.get("endpoint_map"):
            return "R12C", "endpoint/label space contract remains suspect; retry registration candidate repair"
        return "R12A", "space-contract issue lacks Route10 endpoint map; run fallback track lane while preserving review state"
    if issue == "candidate_not_better":
        return "R12B", "production was better but zero-row/numeric QC remains open; try targeted coverage only"
    if r10.get("endpoint_map") and issue in {"mixed_unresolved", "pending"}:
        return "R12C", "mixed unresolved case with endpoint evidence available; retry registration contract repair"
    return "R12A", "mixed unresolved case; run conservative fallback coverage lane"


def choose_lane_from_audit(audit: dict[str, str], r10: dict[str, str]) -> tuple[str, str, str | None]:
    rec = audit.get("recommended_next_step", "")
    if rec == "true_high_density_act_sift2_rerun":
        return (
            "R12A",
            "audit recommends true high-density ACT/SIFT2 rerun; run dynamic-first coverage lane",
            None,
        )
    if rec == "targeted_zero_row_roi_coverage":
        return (
            "R12B",
            "audit recommends targeted zero-row ROI coverage rescue",
            None,
        )
    if rec == "registration_space_contract_repair":
        return (
            "R12C",
            "audit recommends registration/space-contract repair",
            None,
        )
    if rec in {
        "rebuild_5tt_gmwmi_preflight",
        "rebuild_5tt_gmwmi_or_seed_mask",
        "rebuild_missing_fod_5tt_gmwmi",
    }:
        return (
            "",
            audit.get("recommendation_rationale", "requires Step-7 prep/FOD/5TT/GMWMI rebuild before more tractography"),
            "preflight_rebuild",
        )
    if rec == "isolate_probe_failure":
        return (
            "",
            audit.get("recommendation_rationale", "probe failure needs isolated rerun/log inspection before queueing another route"),
            "probe_isolation",
        )
    if rec == "manual_overlay_then_route_specific_repair":
        if r10.get("endpoint_map"):
            return (
                "R12C",
                "manual/mixed audit row with endpoint evidence; use registration contract repair while preserving review state",
                None,
            )
        return (
            "",
            audit.get("recommendation_rationale", "mixed evidence needs overlay/endpoint review before another automated route"),
            "manual_review",
        )
    return "", "audit recommendation is not route-12 actionable yet", "manual_review"


def queue_row(row: dict[str, str], group_for_sid: dict[str, str], lane: str, reason: str, r10: dict[str, str]) -> dict[str, Any]:
    sid = row.get("sid", "")
    out: dict[str, Any] = {
        "sid": sid,
        "group": group_for_sid.get(sid, "AD"),
        "target_lane": lane,
        "source": "route_failure_investigation",
        "reason": reason,
        "issue_class": row.get("issue_class", ""),
        "state": row.get("state", ""),
        "latest_route": row.get("latest_route", ""),
        "latest_decision": row.get("latest_decision", ""),
        "source_labels": row.get("current_source_labels", ""),
        "best_density": row.get("current_best_density", ""),
        "best_zero_rows": row.get("current_best_zero_rows", ""),
        "both_assigned_fraction": row.get("both_assigned_fraction", ""),
        "one_unassigned_fraction": row.get("one_unassigned_fraction", ""),
        "both_unassigned_fraction": row.get("both_unassigned_fraction", ""),
        "routes_tested": row.get("routes_tested", ""),
        "endpoint_map": r10.get("endpoint_map", ""),
        "endpoint_inside_label_fraction": r10.get("endpoint_inside_label_fraction", ""),
        "next_lane": r10.get("next_lane", ""),
    }
    return out


def build_queues(route1_root: Path) -> dict[str, list[dict[str, Any]]]:
    groups = group_map(route1_root)
    r10_by_sid = route10_rows(route1_root)
    audit_by_sid = latest_pipeline_audit(route1_root)
    queues = {lane: [] for lane in R12_SUFFIXES}
    holds: dict[str, list[dict[str, Any]]] = {key: [] for key in HOLD_FILES}
    for row in latest_investigation(route1_root):
        sid = row.get("sid", "")
        if not sid or is_solved(row):
            continue
        r10 = r10_by_sid.get(sid, {})
        audit = audit_by_sid.get(sid, {})
        hold_key: str | None = None
        if audit:
            lane, reason, hold_key = choose_lane_from_audit(audit, r10)
        else:
            lane, reason = choose_lane(row, r10)
        qrow = queue_row(row, groups, lane or hold_key or "HOLD", reason, r10)
        if audit:
            qrow["recommended_next_step"] = audit.get("recommended_next_step", "")
            qrow["recommendation_rationale"] = audit.get("recommendation_rationale", "")
            qrow["gmwmi_mean"] = audit.get("gmwmi_mean", "")
            qrow["gmwmi_nonzero_fraction"] = audit.get("gmwmi_nonzero_fraction", "")
            qrow["tckgen_max_selected"] = audit.get("tckgen_max_selected", "")
            qrow["tckgen_selected_per_million_seeds"] = audit.get("tckgen_selected_per_million_seeds", "")
        if hold_key:
            holds.setdefault(hold_key, []).append(qrow)
        else:
            queues[lane].append(qrow)
    for lane in queues:
        queues[lane].sort(key=lambda item: item["sid"])
    for key in holds:
        holds[key].sort(key=lambda item: item["sid"])
    queues["_holds"] = holds  # type: ignore[assignment]
    return queues


def write_queue(root: Path, lane: str, rows: list[dict[str, Any]], route1_root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    write_csv(root / "route_queue.csv", rows)
    write_csv(root / "subjects.csv", [{"sid": row["sid"], "group": row.get("group", "AD")} for row in rows])
    status = read_json(root / "status.json")
    queued_sids = {str(row.get("sid", "")) for row in rows if row.get("sid")}
    done_sids: set[str] = set()
    failed_sids: set[str] = set()
    for name in ("batch_results_incremental.csv", "batch_results.csv", "route_qc_decisions.csv"):
        for done_row in read_csv(root / name):
            sid = done_row.get("sid") or done_row.get("subject_id") or done_row.get("subject") or ""
            if sid in queued_sids:
                done_sids.add(sid)
                if done_row.get("status") == "failed":
                    failed_sids.add(sid)
    status.update(
        {
            "phase": "queued",
            "lane": lane,
            "route1_root": str(route1_root),
            "total": len(rows),
            "queued_total": len(rows),
            "completed": len(done_sids),
            "failed": len(failed_sids),
            "active": [],
            "last_queue_materialized_utc": utc(),
            "updated_utc": utc(),
        }
    )
    write_json(root / "status.json", status)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route1-root", type=Path, default=DEFAULT_R1)
    args = parser.parse_args()

    route1 = args.route1_root
    queues = build_queues(route1)
    holds = queues.pop("_holds", {})  # type: ignore[assignment]
    summary: dict[str, Any] = {
        "generated_utc": utc(),
        "route1_root": str(route1),
        "used_pipeline_failure_mode_audit": bool((route1 / "pipeline_failure_mode_audit.csv").exists()),
    }
    for lane, suffix in R12_SUFFIXES.items():
        root = route1.parent / f"{route1.name}{suffix}"
        write_queue(root, lane, queues[lane], route1)
        summary[f"{lane.lower()}_root"] = str(root)
        summary[f"{lane.lower()}_queued"] = len(queues[lane])
    for hold_key, file_name in HOLD_FILES.items():
        rows = holds.get(hold_key, []) if isinstance(holds, dict) else []
        write_csv(route1 / file_name, rows)
        summary[f"hold_{hold_key}"] = len(rows)
    write_json(route1 / "route12_queue_materialize_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
