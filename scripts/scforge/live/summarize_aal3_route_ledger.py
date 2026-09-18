#!/usr/bin/env python3
"""Build a consolidated AD AAL3 rescue ledger across all tested routes.

The per-route QC files are intentionally separate, but during a live rescue run
it is easy to lose track of where each subject currently sits. This script reads
the route QC decisions, visual/promotion manifests, and batch status files, then
writes a subject-level ledger and a compact JSON summary into the Route 1 run
root. It is non-destructive and never promotes or edits production outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any


RUN_PARENT = Path("/data/derivatives/qc/sc_matrix_qc/aal3_force_connectome_ad_batch")


ROUTE_SUFFIXES = [
    ("R1", ""),
    ("R2", "_route2"),
    ("R3", "_route3_assignment"),
    ("R4", "_route4_label_rescue"),
    ("R5", "_route5_deep_assignment"),
    ("R6", "_route6_all_voxels_diagnostic"),
    ("R7", "_route7_selective_track_rescue"),
    ("R8", "_route8_full_t1_fnirt"),
    ("R9", "_route9_route8_assignment_retry"),
    ("R10", "_route10_endpoint_overlay_triage"),
    ("R11A", "_route11a_space_affine_registration_repair"),
    ("R11B", "_route11b_act_sift2_track_coverage_rerun"),
    ("R11C", "_route11c_zero_row_coverage_rescue"),
    ("R12A", "_route12a_tckgen_fallback_rescue"),
    ("R12B", "_route12b_zero_row_targeted_rerun"),
    ("R12C", "_route12c_registration_contract_repair"),
    ("R13", "_route13_step7_preflight_rebuild"),
    ("R14", "_route14_step7_rebuilt_track_coverage"),
    ("R14B", "_route14b_step7_assignment_retry"),
    ("R15", "_route15_step7_dense_track_coverage"),
    ("R16", "_route16_zero_roi_hybrid"),
    ("R17", "_route17_step7_dense_unpushed"),
    ("R18", "_route18_gmwmi_dense_track_coverage"),
    ("R19", "_route19_gmwmi_3m_retry"),
    ("R20", "_route20_zero_roi_gmwmi_endpoint"),
    ("R21", "_route21_endpoint_shell_assignment_map"),
]


DECISION_RANK = {
    "PROMOTE_CANDIDATE_PENDING_VISUAL": 5,
    "INTERIM_REVIEW_PROMOTION_PENDING_VISUAL": 4,
    "KEEP_PRODUCTION_NUMERIC_OK": 3,
    "REVIEW_ROUTE": 2,
    "TRY_ANOTHER_ROUTE": 1,
    "": 0,
}


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size <= 0:
        return []
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists() or path.stat().st_size <= 0:
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def completed_subject_count(route_root: Path, status: dict[str, Any]) -> int:
    sids = {
        row.get("sid", "")
        for row in read_csv(route_root / "batch_results.csv")
        if row.get("sid")
    }
    subjects_dir = route_root / "subjects"
    if subjects_dir.exists():
        for marker in subjects_dir.glob("*/done.json"):
            payload = read_json(marker)
            sid = str(payload.get("sid") or marker.parent.name).strip()
            if sid:
                sids.add(sid)
    return max(len(sids), int(status.get("completed") or 0))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def latest_run_root() -> Path:
    latest = RUN_PARENT / "latest_tag.txt"
    if latest.exists():
        tag = latest.read_text(encoding="utf-8").strip()
        if tag:
            return RUN_PARENT / tag
    runs = sorted((p for p in RUN_PARENT.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime, reverse=True)
    if not runs:
        raise SystemExit(f"No AAL3 force-connectome run roots under {RUN_PARENT}")
    return runs[0]


def route_roots(run_root: Path) -> list[tuple[str, Path]]:
    return [(rid, run_root.parent / f"{run_root.name}{suffix}") for rid, suffix in ROUTE_SUFFIXES]


def preferred_decision(rows: list[dict[str, str]]) -> dict[str, str]:
    if not rows:
        return {}
    return sorted(
        rows,
        key=lambda r: (
            int(r.get("_route_order", 0)),
            DECISION_RANK.get(r.get("qc_decision", ""), 0),
        ),
    )[-1]


def promotion_rank(row: dict[str, str]) -> int:
    if row.get("promotion_class") == "final_qc_pass":
        return 3
    if row.get("promotion_class") in {"", "solved"}:
        return 3
    if row.get("promotion_class") == "numeric_pass_pending_visual_not_solved":
        return 2
    if row.get("promotion_class") == "interim_review_not_solved":
        return 1
    return 1


def promotion_sort_key(row: dict[str, str]) -> tuple[str, int]:
    """Order promotion rows by the actual production overwrite time.

    A later route can be superseded by reapplying a better earlier-route
    candidate. The ledger should describe the current production overwrite, not
    whichever route has the highest route id or promotion class.
    """
    return (row.get("promoted_utc", ""), int(row.get("_route_order", 0)))


def promotion_is_solved(row: dict[str, str]) -> bool:
    return row.get("promotion_class") in {"final_qc_pass", "solved"} or row.get("final_qc_status") == "solved"


def visual_statuses(route_root: Path) -> dict[str, str]:
    statuses: dict[str, str] = {}
    for row in read_csv(route_root / "visual_overlay_qc_manifest.csv"):
        sid = row.get("sid", "")
        if sid:
            statuses[sid] = row.get("visual_overlay_status", "")
    return statuses


def summarize(run_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    subjects = {row.get("sid", ""): row for row in read_csv(run_root / "subjects.csv") if row.get("sid")}
    route_decisions_by_sid: dict[str, list[dict[str, str]]] = {}
    promoted_by_sid: dict[str, dict[str, str]] = {}
    route_summaries: dict[str, Any] = {}

    for route_order, (route_id, route_root) in enumerate(route_roots(run_root), start=1):
        status = read_json(route_root / "status.json")
        route_visuals = visual_statuses(route_root)
        decisions = read_csv(route_root / "route_qc_decisions.csv")
        decision_counts: dict[str, int] = {}
        for row in decisions:
            row = dict(row)
            sid = row.get("sid", "")
            if not sid:
                continue
            row["_route"] = route_id
            row["_route_order"] = str(route_order)
            if sid in route_visuals:
                row["visual_overlay_status"] = route_visuals[sid]
            route_decisions_by_sid.setdefault(sid, []).append(row)
            decision = row.get("qc_decision", "")
            decision_counts[decision] = decision_counts.get(decision, 0) + 1

        promoted_rows = [
            dict(row, _route=route_id, _route_order=str(route_order))
            for row in read_csv(route_root / "production_promotion_manifest.csv")
            if row.get("status") == "promoted" and row.get("sid")
        ]
        promoted_sids = {row["sid"] for row in promoted_rows}
        for row in promoted_rows:
            sid = row["sid"]
            if sid not in promoted_by_sid or promotion_sort_key(row) >= promotion_sort_key(promoted_by_sid[sid]):
                promoted_by_sid[sid] = row

        route_summaries[route_id] = {
            "root": str(route_root),
            "exists": route_root.exists(),
            "phase": status.get("phase", "not_started" if not route_root.exists() else ""),
            "completed": completed_subject_count(route_root, status),
            "total": int(status.get("total") or 0),
            "decision_rows": len(decisions),
            "decisions": decision_counts,
            "promoted_rows": len(promoted_sids),
        }

    for sid in route_decisions_by_sid:
        subjects.setdefault(sid, {"sid": sid, "group": "AD"})

    ledger: list[dict[str, Any]] = []
    for sid in sorted(subjects):
        subj = subjects[sid]
        decisions = route_decisions_by_sid.get(sid, [])
        latest = sorted(decisions, key=lambda r: int(r.get("_route_order", 0)))[-1] if decisions else {}
        best = preferred_decision(decisions)
        promoted = promoted_by_sid.get(sid, {})

        if promoted and promotion_is_solved(promoted):
            state = "solved_full_qc_production"
            next_action = "none"
        elif promoted and promoted.get("promotion_class") == "numeric_pass_pending_visual_not_solved":
            state = "numeric_production_pending_visual_keep_trying"
            next_action = "review overlay; keep alternate-route search open until visual QC passes"
        elif promoted and promoted.get("promotion_class") == "interim_review_not_solved":
            state = "interim_production_keep_trying"
            next_action = "continue alternate routes until full QC"
        elif best.get("qc_decision") in {
            "PROMOTE_CANDIDATE_PENDING_VISUAL",
            "INTERIM_REVIEW_PROMOTION_PENDING_VISUAL",
        } and best.get("visual_overlay_status", "").startswith(("failed", "rejected")):
            state = "visual_failed_keep_trying"
            next_action = "do not promote this visual-failed candidate; continue alternate routes"
        elif best.get("qc_decision") == "PROMOTE_CANDIDATE_PENDING_VISUAL":
            state = "numeric_pass_pending_visual"
            next_action = "generate/review overlay, then promote if visual passes"
        elif best.get("qc_decision") == "INTERIM_REVIEW_PROMOTION_PENDING_VISUAL":
            state = "interim_candidate_pending_visual"
            next_action = "visual review allowed for interim, but continue routes for full QC"
        elif latest.get("qc_decision") == "REVIEW_ROUTE":
            state = "review_route"
            next_action = "inspect reasons and continue next route"
        elif best.get("qc_decision") == "KEEP_PRODUCTION_NUMERIC_OK":
            state = "production_current_numeric_ok"
            next_action = "no overwrite; keep production unless a better full-QC candidate appears"
        elif latest.get("qc_decision") == "TRY_ANOTHER_ROUTE":
            state = "needs_next_route"
            next_action = "continue queued rescue route"
        elif sid in subjects and not decisions:
            state = "not_scored_yet"
            next_action = "wait for current route output"
        else:
            state = "unknown"
            next_action = "inspect route artifacts"

        ledger.append(
            {
                "sid": sid,
                "group": subj.get("group", "AD"),
                "state": state,
                "next_action": next_action,
                "latest_route": latest.get("_route", ""),
                "latest_decision": latest.get("qc_decision", ""),
                "best_route": best.get("_route", ""),
                "best_decision": best.get("qc_decision", ""),
                "best_candidate": best.get("best_candidate", ""),
                "best_variant": best.get("best_variant", ""),
                "source_labels": best.get("source_labels", ""),
                "baseline_density": best.get("baseline_density", ""),
                "best_density": best.get("best_density", ""),
                "density_delta": best.get("density_delta", ""),
                "baseline_zero_rows": best.get("baseline_zero_rows", ""),
                "best_zero_rows": best.get("best_zero_rows", ""),
                "zero_rows_delta": best.get("zero_rows_delta", ""),
                "both_assigned_fraction": best.get("both_assigned_fraction", ""),
                "one_unassigned_fraction": best.get("one_unassigned_fraction", ""),
                "both_unassigned_fraction": best.get("both_unassigned_fraction", ""),
                "visual_overlay_status": promoted.get("visual_overlay_status", best.get("visual_overlay_status", "")),
                "promoted_route": promoted.get("_route", ""),
                "promotion_class": promoted.get("promotion_class", ""),
                "final_qc_status": promoted.get("final_qc_status", best.get("final_qc_status", "")),
                "routes_tested": ",".join(sorted({row.get("_route", "") for row in decisions if row.get("_route")})),
                "qc_reasons": latest.get("qc_reasons", ""),
            }
        )

    state_counts: dict[str, int] = {}
    for row in ledger:
        state_counts[row["state"]] = state_counts.get(row["state"], 0) + 1

    summary = {
        "generated_utc": utc(),
        "run_root": str(run_root),
        "total_subjects": len(ledger),
        "state_counts": state_counts,
        "routes": route_summaries,
        "production_unique_subjects": len(promoted_by_sid),
        "production_full_qc_solved": sum(
            1 for row in promoted_by_sid.values() if promotion_is_solved(row)
        ),
        "production_interim_review_not_solved": sum(
            1 for row in promoted_by_sid.values() if row.get("promotion_class") == "interim_review_not_solved"
        ),
        "production_numeric_pending_visual_not_solved": sum(
            1
            for row in promoted_by_sid.values()
            if row.get("promotion_class") == "numeric_pass_pending_visual_not_solved"
        ),
        "production_not_solved_keep_trying": sum(
            1 for row in promoted_by_sid.values() if not promotion_is_solved(row)
        ),
    }
    return ledger, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=None)
    args = parser.parse_args()
    run_root = args.run_root or latest_run_root()
    ledger, summary = summarize(run_root)
    write_csv(run_root / "route_subject_ledger.csv", ledger)
    write_json(run_root / "route_subject_ledger_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"wrote {run_root / 'route_subject_ledger.csv'}")
    print(f"wrote {run_root / 'route_subject_ledger_summary.json'}")


if __name__ == "__main__":
    main()
