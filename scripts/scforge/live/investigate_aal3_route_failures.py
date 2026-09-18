#!/usr/bin/env python3
"""Investigate unresolved AAL3 rescue subjects across all active rescue routes.

The live monitor is intentionally compact. This report is the opposite: it
keeps a per-subject route matrix and a short corrective recommendation so we
can decide whether the next step is another registration route, assignment
route, or a true tractography/Step-7 rerun.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any


RUN_PARENT = Path("/data/derivatives/qc/sc_matrix_qc/aal3_force_connectome_ad_batch")

ROUTES: list[tuple[str, str, str]] = [
    ("R1", "", "source-contract AAL3 to B0 with existing tracks"),
    ("R2", "_route2", "BBR/native registration variants"),
    ("R3", "_route3_assignment", "assignment-policy variants on existing parcellations"),
    ("R4", "_route4_label_rescue", "label-preserving dilation rescue"),
    ("R5", "_route5_deep_assignment", "deep endpoint-assignment search"),
    ("R6", "_route6_all_voxels_diagnostic", "diagnostic all-voxels traversal"),
    ("R7", "_route7_selective_track_rescue", "selective FNIRT or dynamic-track rescue"),
    ("R8", "_route8_full_t1_fnirt", "full-T1 FNIRT label rescue"),
    ("R9", "_route9_route8_assignment_retry", "assignment retry on Route-8 parcellations"),
    ("R10", "_route10_endpoint_overlay_triage", "endpoint-overlay triage and next-lane selection"),
    ("R11A", "_route11a_space_affine_registration_repair", "endpoint-overlap space/affine repair"),
    ("R11B", "_route11b_act_sift2_track_coverage_rerun", "ACT/SIFT2 track-coverage rerun"),
    ("R11C", "_route11c_zero_row_coverage_rescue", "targeted zero-row coverage rescue"),
    ("R12A", "_route12a_tckgen_fallback_rescue", "GMWMI-first lower-target tractography fallback"),
    ("R12B", "_route12b_zero_row_targeted_rerun", "zero-row targeted tractography fallback"),
    ("R12C", "_route12c_registration_contract_repair", "registration/space-contract repair"),
    ("R13", "_route13_step7_preflight_rebuild", "Step-7 preflight rebuild of native T1/ACT prerequisites"),
    ("R14", "_route14_step7_rebuilt_track_coverage", "rebuilt Step-7 track-coverage test"),
    ("R14B", "_route14b_step7_assignment_retry", "assignment retry on rebuilt Step-7 tracks"),
    ("R15", "_route15_step7_dense_track_coverage", "denser rebuilt Step-7 track-coverage test"),
    ("R16", "_route16_zero_roi_hybrid", "hybrid zero-row ROI coverage rescue"),
    ("R17", "_route17_step7_dense_unpushed", "dense rebuilt Step-7 route for unpushed cases"),
    ("R18", "_route18_gmwmi_dense_track_coverage", "GMWMI dense track-coverage rerun"),
    ("R19", "_route19_gmwmi_3m_retry", "GMWMI 3M retry for Route-18 failures"),
    ("R20", "_route20_zero_roi_gmwmi_endpoint", "per-ROI zero-row endpoint coverage canary"),
]

PASSISH_DECISIONS = {
    "PROMOTE_CANDIDATE_PENDING_VISUAL",
    "INTERIM_REVIEW_PROMOTION_PENDING_VISUAL",
    "KEEP_PRODUCTION_NUMERIC_OK",
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


def fmt(value: Any, digits: int = 4) -> str:
    val = fnum(value)
    if val != val:
        return ""
    return f"{val:.{digits}f}"


def route_root(run_root: Path, suffix: str) -> Path:
    return run_root.parent / f"{run_root.name}{suffix}"


def route_decisions(run_root: Path) -> dict[str, dict[str, dict[str, str]]]:
    out: dict[str, dict[str, dict[str, str]]] = {}
    for route_id, suffix, _desc in ROUTES:
        root = route_root(run_root, suffix)
        rows = [row for row in read_csv(root / "route_qc_decisions.csv") if row.get("sid")]
        out[route_id] = {row["sid"]: row for row in rows}
    return out


def subject_set(run_root: Path, decisions: dict[str, dict[str, dict[str, str]]]) -> set[str]:
    sids = {row.get("sid", "") for row in read_csv(run_root / "subjects.csv") if row.get("sid")}
    for rows in decisions.values():
        sids.update(rows.keys())
    return {sid for sid in sids if sid}


def current_route(rows_by_route: dict[str, dict[str, str]]) -> tuple[str, dict[str, str]]:
    for route_id, _suffix, _desc in reversed(ROUTES):
        row = rows_by_route.get(route_id, {})
        if row:
            return route_id, row
    return "", {}


def route9_status(row: dict[str, str]) -> str:
    if not row:
        return "not_tested"
    decision = row.get("qc_decision", "")
    if decision == "TRY_ANOTHER_ROUTE":
        return "failed_route9"
    if decision == "KEEP_PRODUCTION_NUMERIC_OK":
        return "route9_not_better_keep_current"
    if decision in {"PROMOTE_CANDIDATE_PENDING_VISUAL", "INTERIM_REVIEW_PROMOTION_PENDING_VISUAL"}:
        return "route9_candidate"
    return decision or "unknown"


def classify(row: dict[str, str], route6: dict[str, str]) -> tuple[str, str, str]:
    if not row:
        return ("pending", "No scored route row yet.", "Wait for Route 1 or active route scoring.")

    status = row.get("status", "")
    labels = inum(row.get("source_labels"), 0)
    zero = inum(row.get("best_zero_rows") or row.get("diagnostic_zero_rows"), 9999)
    density = fnum(row.get("best_density") or row.get("diagnostic_density"), -1)
    delta = fnum(row.get("density_delta"), 0)
    both = fnum(row.get("both_assigned_fraction"), float("nan"))
    one = fnum(row.get("one_unassigned_fraction"), float("nan"))
    neither = fnum(row.get("both_unassigned_fraction"), float("nan"))
    decision = row.get("qc_decision", "")
    bucket = route6.get("diagnostic_bucket", "")

    if status and status != "ok":
        return (
            "probe_execution_failure",
            "The route did not finish cleanly.",
            "Inspect the subject probe logs, rerun that route with per-variant failure isolation, then score again.",
        )

    if labels and labels < 160:
        return (
            "registration_label_survival",
            f"AAL3 label survival is {labels}/166, below the 160/166 target.",
            "Do not keep searching assignment parameters first. Rebuild the atlas-to-B0 registration chain for this subject: inspect T1/B0 alignment, BBR matrix direction, FNIRT warp direction, nearest-neighbor label interpolation, and overlay against B0.",
        )

    if both == both and both <= 0.05 and neither >= 0.90:
        return (
            "space_contract_break",
            "Almost every streamline is unassigned despite high label survival.",
            "Treat this as a parcellation/streamline-space mismatch. Verify NIfTI grid, affine, label integer survival, and B0 overlay; rerun the registration/composed-transform route before any tractography rerun.",
        )

    if both == both and both < 0.65 and neither > 0.20:
        return (
            "endpoint_assignment_space_mismatch",
            "Labels survived, but many streamline endpoints are outside the labels.",
            "Inspect overlay plus endpoint cloud. If overlay is shifted, correct registration. If overlay is good but endpoints terminate away from gray matter, rerun ACT tractography/SIFT2 with validated 5TT and GMWMI seeding.",
        )

    if bucket == "track_coverage_limited" or (both == both and both >= 0.85 and zero > 50):
        return (
            "track_coverage_limited",
            "The atlas is usable, but existing streamlines do not cover enough AAL3 ROIs.",
            "This is the strongest case for rerunning Step-7 tractography: validate 5TT/FOD/GMWMI, generate a scratch ACT track set with denser GM-WM seeding, then rebuild connectomes.",
        )

    if zero > 15 and density >= 0.10:
        return (
            "local_zero_row_problem",
            "Density is plausible, but too many AAL3 rows have no connections.",
            "Keep the best production/interim matrix, then run a targeted coverage rescue for the zero-row ROIs; if repeated routes keep the same zero rows, rerun tractography for coverage.",
        )

    if density < 0.10 or zero > 15:
        return (
            "sparse_matrix_persistent",
            "The candidate remains sparse or zero-row heavy.",
            "Use Route-6 bucket plus overlay to choose between registration rebuild and tractography rerun; do not solve this by accepting density alone.",
        )

    if delta < 0 and decision == "KEEP_PRODUCTION_NUMERIC_OK":
        return (
            "candidate_not_better",
            "The tested route is worse than current production.",
            "Keep the current production matrix and continue only if visual/assignment QC is still open.",
        )

    return (
        "mixed_unresolved",
        "No single numeric failure dominates.",
        "Review the overlay, assignment fractions, zero-row map, and Route-6 diagnostic bucket before choosing the next route.",
    )


def build_reports(run_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    decisions = route_decisions(run_root)
    ledger_rows = {row.get("sid", ""): row for row in read_csv(run_root / "route_subject_ledger.csv") if row.get("sid")}
    sids = sorted(subject_set(run_root, decisions))
    investigation: list[dict[str, Any]] = []
    matrix: list[dict[str, Any]] = []

    for sid in sids:
        rows_by_route = {route_id: decisions.get(route_id, {}).get(sid, {}) for route_id, _suffix, _desc in ROUTES}
        latest_route, latest = current_route(rows_by_route)
        r6 = rows_by_route.get("R6", {})
        issue, interpretation, remedial = classify(latest, r6)
        ledger = ledger_rows.get(sid, {})

        investigation.append(
            {
                "sid": sid,
                "state": ledger.get("state", ""),
                "latest_route": latest_route,
                "latest_decision": latest.get("qc_decision", ""),
                "route9_status": route9_status(rows_by_route.get("R9", {})),
                "route6_bucket": r6.get("diagnostic_bucket", ""),
                "issue_class": issue,
                "interpretation": interpretation,
                "recommended_corrective_measure": remedial,
                "current_best_route": ledger.get("best_route", ""),
                "current_best_variant": ledger.get("best_variant", latest.get("best_variant", "")),
                "current_source_labels": ledger.get("source_labels", latest.get("source_labels", "")),
                "current_best_density": ledger.get("best_density", latest.get("best_density", "")),
                "current_density_delta": ledger.get("density_delta", latest.get("density_delta", "")),
                "current_best_zero_rows": ledger.get("best_zero_rows", latest.get("best_zero_rows", "")),
                "current_zero_rows_delta": ledger.get("zero_rows_delta", latest.get("zero_rows_delta", "")),
                "both_assigned_fraction": ledger.get("both_assigned_fraction", latest.get("both_assigned_fraction", "")),
                "one_unassigned_fraction": ledger.get("one_unassigned_fraction", latest.get("one_unassigned_fraction", "")),
                "both_unassigned_fraction": ledger.get("both_unassigned_fraction", latest.get("both_unassigned_fraction", "")),
                "routes_tested": ledger.get("routes_tested", ",".join(r for r, row in rows_by_route.items() if row)),
                "latest_qc_reasons": latest.get("qc_reasons", ledger.get("qc_reasons", "")),
                "latest_final_qc_reasons": latest.get("final_qc_reasons", ""),
            }
        )

        row: dict[str, Any] = {"sid": sid, "state": ledger.get("state", "")}
        for route_id, _suffix, _desc in ROUTES:
            rr = rows_by_route.get(route_id, {})
            row[f"{route_id}_decision"] = rr.get("qc_decision", "")
            row[f"{route_id}_variant"] = rr.get("best_variant", "")
            row[f"{route_id}_labels"] = rr.get("source_labels", "")
            row[f"{route_id}_density"] = rr.get("best_density") or rr.get("diagnostic_density", "")
            row[f"{route_id}_density_delta"] = rr.get("density_delta", "")
            row[f"{route_id}_zero_rows"] = rr.get("best_zero_rows") or rr.get("diagnostic_zero_rows", "")
            row[f"{route_id}_zero_delta"] = rr.get("zero_rows_delta", "")
            row[f"{route_id}_both_assigned"] = rr.get("both_assigned_fraction", "")
            row[f"{route_id}_bucket"] = rr.get("diagnostic_bucket", "")
        matrix.append(row)

    counts: dict[str, int] = {}
    route9_counts: dict[str, int] = {}
    for row in investigation:
        counts[row["issue_class"]] = counts.get(row["issue_class"], 0) + 1
        route9_counts[row["route9_status"]] = route9_counts.get(row["route9_status"], 0) + 1

    summary = {
        "generated_utc": utc(),
        "run_root": str(run_root),
        "subjects": len(investigation),
        "issue_counts": counts,
        "route9_status_counts": route9_counts,
        "route_descriptions": {rid: desc for rid, _suffix, desc in ROUTES},
    }
    return investigation, matrix, summary


def write_markdown(path: Path, investigation: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    route9_failed = [row for row in investigation if row["route9_status"] == "failed_route9"]
    route9_current = [row for row in investigation if row["route9_status"] != "not_tested"]
    lines = [
        "# AAL3 Route Failure Investigation",
        "",
        f"Generated UTC: {summary['generated_utc']}",
        f"Run root: `{summary['run_root']}`",
        "",
        "## Counts",
        "",
        "### Route 9 status",
    ]
    for key, value in sorted(summary["route9_status_counts"].items()):
        lines.append(f"- {key}: {value}")
    lines.extend(["", "### Issue classes"])
    for key, value in sorted(summary["issue_counts"].items()):
        lines.append(f"- {key}: {value}")
    lines.extend(
        [
            "",
            "## Route 9 Completed Non-Pass Subjects",
            "",
            "| Subject | Latest decision | Labels | Density | Zero rows | Both assigned | Issue | Corrective measure |",
            "|---|---:|---:|---:|---:|---:|---|---|",
        ]
    )
    for row in route9_current:
        lines.append(
            "| {sid} | {decision} | {labels} | {density} | {zero} | {both} | {issue} | {fix} |".format(
                sid=row["sid"],
                decision=row["latest_decision"],
                labels=row["current_source_labels"],
                density=fmt(row["current_best_density"]),
                zero=row["current_best_zero_rows"],
                both=fmt(row["both_assigned_fraction"], 3),
                issue=row["issue_class"],
                fix=row["recommended_corrective_measure"],
            )
        )
    if not route9_failed:
        lines.append("| none |  |  |  |  |  |  |  |")
    lines.extend(
        [
            "",
            "## Route Definitions",
            "",
        ]
    )
    for route_id, desc in summary["route_descriptions"].items():
        lines.append(f"- {route_id}: {desc}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=RUN_PARENT / "ad_existing_tracks_20260529T055752Z")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory for report outputs. Defaults to the run root.",
    )
    args = parser.parse_args()

    investigation, matrix, summary = build_reports(args.run_root)
    out_dir = args.out_dir or args.run_root
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "route_failure_investigation.csv", investigation)
    write_csv(out_dir / "route_failure_metric_matrix.csv", matrix)
    write_json(out_dir / "route_failure_investigation_summary.json", summary)
    write_markdown(out_dir / "route_failure_investigation.md", investigation, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"wrote {out_dir / 'route_failure_investigation.csv'}")
    print(f"wrote {out_dir / 'route_failure_metric_matrix.csv'}")
    print(f"wrote {out_dir / 'route_failure_investigation.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
