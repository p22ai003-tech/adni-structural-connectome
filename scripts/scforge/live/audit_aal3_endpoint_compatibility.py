#!/usr/bin/env python3
"""Audit AAL3 endpoint/node compatibility for current AD route state.

This does not create or promote connectomes. It reads the route ledger and
assignment files, then asks which surviving AAL3 labels receive no endpoints or
no nonzero matrix edges. This is meant to prevent blind route proliferation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path("/home/ec2-user/exp")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.scforge.live.run_sc_aal3_source_contract_probe import aal3_valid_labels, matrix_array  # noqa: E402


DEFAULT_RUN_ROOT = Path(
    "/data/derivatives/qc/sc_matrix_qc/aal3_force_connectome_ad_batch/"
    "ad_existing_tracks_20260529T055752Z"
)
DEFAULT_REPORT_DIR = Path("/home/ec2-user/exp/reports/aal3_endpoint_compatibility")
DEFAULT_NODE_QC = Path("/home/ec2-user/exp/reports/aal3_node_qc/aal3_node_qc_latest.csv")


ROUTE_SUFFIX = {
    "R1": "",
    "R2": "_route2",
    "R3": "_route3_assignment",
    "R4": "_route4_label_rescue",
    "R5": "_route5_deep_assignment",
    "R6": "_route6_all_voxels_diagnostic",
    "R7": "_route7_selective_track_rescue",
    "R8": "_route8_full_t1_fnirt",
    "R9": "_route9_route8_assignment_retry",
    "R10": "_route10_endpoint_overlay_triage",
    "R11A": "_route11a_space_affine_registration_repair",
    "R11B": "_route11b_act_sift2_track_coverage_rerun",
    "R11C": "_route11c_zero_row_coverage_rescue",
    "R12A": "_route12a_tckgen_fallback_rescue",
    "R12B": "_route12b_zero_row_targeted_rerun",
    "R12C": "_route12c_registration_contract_repair",
    "R13": "_route13_step7_preflight_rebuild",
    "R14": "_route14_step7_rebuilt_track_coverage",
    "R14B": "_route14b_step7_assignment_retry",
    "R15": "_route15_step7_dense_track_coverage",
    "R16": "_route16_zero_roi_hybrid",
    "R17": "_route17_step7_dense_unpushed",
    "R18": "_route18_gmwmi_dense_track_coverage",
    "R19": "_route19_gmwmi_3m_retry",
    "R20": "_route20_zero_roi_gmwmi_endpoint",
    "R21": "_route21_endpoint_shell_assignment_map",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size <= 0:
        return []
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fnum(value: Any, default: float = math.nan) -> float:
    try:
        text = str(value).strip()
        if not text or text.lower() in {"nan", "none"}:
            return default
        out = float(text)
        return out if math.isfinite(out) else default
    except Exception:
        return default


def route_root(run_root: Path, route: str) -> Path:
    return Path(str(run_root) + ROUTE_SUFFIX.get(route, ""))


def load_route_decisions(run_root: Path) -> dict[tuple[str, str], dict[str, str]]:
    out: dict[tuple[str, str], dict[str, str]] = {}
    for route in ROUTE_SUFFIX:
        root = route_root(run_root, route)
        for row in read_csv(root / "route_qc_decisions.csv"):
            sid = row.get("sid", "")
            if not sid:
                continue
            out[(sid, route)] = row
    return out


def matrix_from_assignment(assignment_file: Path, sid: str) -> Path | None:
    name = assignment_file.name
    if not name.startswith("assignments_"):
        return None
    suffix = name.removeprefix("assignments_")
    return assignment_file.parent / f"SC_AAL3_{sid}_{suffix}"


def parse_assignments(path: Path, max_rows: int = 250_000) -> tuple[Counter[int], Counter[int], int, int, int, int, int, int]:
    """Return endpoint counts, incident streamline counts, rows and assignment buckets."""
    endpoint_counts: Counter[int] = Counter()
    incident_counts: Counter[int] = Counter()
    rows = both = one_unassigned = both_unassigned = malformed = skipped_after_cap = 0
    if not path.exists() or path.stat().st_size <= 0:
        return endpoint_counts, incident_counts, rows, both, one_unassigned, both_unassigned, malformed, skipped_after_cap
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if max_rows > 0 and rows >= max_rows:
                skipped_after_cap = -1
                break
            parts = re.split(r"[\s,]+", line)
            if len(parts) < 2:
                malformed += 1
                continue
            try:
                a = int(float(parts[0]))
                b = int(float(parts[1]))
            except ValueError:
                malformed += 1
                continue
            rows += 1
            positive = [x for x in (a, b) if x > 0]
            if len(positive) == 2:
                both += 1
            elif len(positive) == 1:
                one_unassigned += 1
            else:
                both_unassigned += 1
            for label in positive:
                endpoint_counts[label] += 1
            for label in set(positive):
                incident_counts[label] += 1
    return endpoint_counts, incident_counts, rows, both, one_unassigned, both_unassigned, malformed, skipped_after_cap


def matrix_zero_labels(path: Path, valid_labels: list[int]) -> tuple[set[int], float, bool]:
    if not path or not path.exists() or path.stat().st_size <= 0:
        return set(valid_labels), math.nan, False
    arr = matrix_array(path)
    if arr.ndim != 2:
        return set(valid_labels), math.nan, False
    n = arr.shape[0]
    zero: set[int] = set()
    nonzero_edges = 0
    possible = 0
    for label in valid_labels:
        idx = label - 1
        if idx < 0 or idx >= n:
            zero.add(label)
            continue
        row = np.asarray(arr[idx, :], dtype=float)
        row = np.where(np.isfinite(row), row, 0.0)
        row[idx] = 0.0
        if not np.any(row > 0):
            zero.add(label)
    valid_idx = [label - 1 for label in valid_labels if 0 <= label - 1 < n]
    sub = np.asarray(arr[np.ix_(valid_idx, valid_idx)], dtype=float)
    sub = np.where(np.isfinite(sub), sub, 0.0)
    np.fill_diagonal(sub, 0.0)
    triu = sub[np.triu_indices_from(sub, k=1)]
    nonzero_edges = int(np.count_nonzero(triu > 0))
    possible = int(len(triu))
    density = nonzero_edges / possible if possible else math.nan
    return zero, density, True


def node_lookup(path: Path) -> dict[int, dict[str, str]]:
    rows = read_csv(path)
    out: dict[int, dict[str, str]] = {}
    for row in rows:
        try:
            label = int(float(row.get("aal3_label", "")))
        except ValueError:
            continue
        out[label] = row
    return out


def selected_decision_for_subject(
    ledger_row: dict[str, str],
    route_rows: dict[tuple[str, str], dict[str, str]],
) -> tuple[str, dict[str, str] | None]:
    sid = ledger_row.get("sid", "")
    route_order = [
        ledger_row.get("promoted_route", ""),
        ledger_row.get("best_route", ""),
        ledger_row.get("latest_route", ""),
    ]
    for route in route_order:
        if route and (sid, route) in route_rows:
            return route, route_rows[(sid, route)]
    return "", None


def classify_label(
    *,
    has_endpoint: bool,
    is_zero: bool,
    node_row: dict[str, str],
) -> str:
    survival = fnum(node_row.get("label_survival_frequency"), 0)
    zero_freq = fnum(node_row.get("zero_row_frequency"), 0)
    if is_zero and not has_endpoint and survival >= 0.9:
        return "survives_but_no_endpoint"
    if is_zero and has_endpoint:
        return "endpoint_present_but_no_weighted_edge"
    if is_zero and survival < 0.9:
        return "label_survival_or_tiny_label"
    if zero_freq >= 0.5:
        return "historically_recurrent_zero"
    return "endpoint_supported"


def subject_corrective_class(
    *,
    audit_status: str,
    zero_count: int,
    endpoint_zero_count: int,
    endpoint_present_zero_count: int,
    both_assigned_fraction: float,
    both_unassigned_fraction: float,
) -> str:
    if audit_status != "ok":
        return "missing_artifacts_rebuild_assignment_and_matrix"
    if zero_count <= 15:
        return "numeric_zero_rows_within_qc_gate"
    if both_assigned_fraction < 0.85 or both_unassigned_fraction > 0.05:
        return "space_or_endpoint_assignment_contract_failure"
    if endpoint_zero_count >= max(3, endpoint_present_zero_count):
        return "roi_endpoint_coverage_failure"
    if endpoint_present_zero_count > 0:
        return "weighted_edge_or_sift2_edge_survival_failure"
    return "mixed_zero_row_failure"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--node-qc", type=Path, default=DEFAULT_NODE_QC)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--max-assignment-rows",
        type=int,
        default=250_000,
        help="Rows parsed per assignment file. Use 0 for full exact parsing.",
    )
    args = parser.parse_args()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = args.out_dir
    valid_labels = aal3_valid_labels()
    nodes = node_lookup(args.node_qc)
    ledger = [r for r in read_csv(args.run_root / "route_subject_ledger.csv") if r.get("group") == "AD"]
    if args.limit > 0:
        ledger = ledger[: args.limit]
    route_rows = load_route_decisions(args.run_root)

    subject_rows: list[dict[str, Any]] = []
    label_counter: dict[int, Counter[str]] = defaultdict(Counter)
    label_endpoint_sum: Counter[int] = Counter()
    label_incident_sum: Counter[int] = Counter()
    unresolved_subjects = 0

    for led in ledger:
        sid = led.get("sid", "")
        route, decision = selected_decision_for_subject(led, route_rows)
        if decision is None:
            subject_rows.append(
                {
                    "sid": sid,
                    "state": led.get("state", ""),
                    "selected_route": route,
                    "audit_status": "missing_decision_row",
                }
            )
            continue
        assignment_text = str(decision.get("assignment_file", "")).strip()
        assignment = Path(assignment_text) if assignment_text else Path()
        matrix = matrix_from_assignment(assignment, sid) if assignment_text else None
        if assignment_text:
            endpoint_counts, incident_counts, rows, both, one, neither, malformed, skipped_after_cap = parse_assignments(
                assignment,
                max_rows=args.max_assignment_rows,
            )
        else:
            endpoint_counts, incident_counts, rows, both, one, neither, malformed, skipped_after_cap = (
                Counter(),
                Counter(),
                0,
                0,
                0,
                0,
                0,
                0,
            )
        zero_labels, density, matrix_ok = matrix_zero_labels(matrix, valid_labels) if matrix else (set(valid_labels), math.nan, False)
        endpoint_zero_labels = [label for label in sorted(zero_labels) if endpoint_counts.get(label, 0) == 0]
        endpoint_present_zero_labels = [label for label in sorted(zero_labels) if endpoint_counts.get(label, 0) > 0]
        both_fraction = both / rows if rows else math.nan
        both_unassigned_fraction = neither / rows if rows else math.nan
        audit_status = "ok" if assignment_text and assignment.exists() and matrix_ok else "missing_assignment_or_matrix"
        unresolved_subjects += int(len(zero_labels) > 15)
        for label in valid_labels:
            node_row = nodes.get(label, {})
            has_endpoint = endpoint_counts.get(label, 0) > 0
            is_zero = label in zero_labels
            cls = classify_label(has_endpoint=has_endpoint, is_zero=is_zero, node_row=node_row)
            label_counter[label][cls] += 1
            if is_zero:
                label_counter[label]["zero_subjects_in_selected_matrix"] += 1
            if has_endpoint:
                label_counter[label]["subjects_with_endpoint"] += 1
            label_endpoint_sum[label] += endpoint_counts.get(label, 0)
            label_incident_sum[label] += incident_counts.get(label, 0)
        subject_rows.append(
            {
                "sid": sid,
                "state": led.get("state", ""),
                "selected_route": route,
                "selected_decision": decision.get("qc_decision", ""),
                "final_qc_status": led.get("final_qc_status", ""),
                "assignment_file": str(assignment),
                "matrix_file": str(matrix) if matrix else "",
                "audit_status": audit_status,
                "corrective_class": subject_corrective_class(
                    audit_status=audit_status,
                    zero_count=len(zero_labels),
                    endpoint_zero_count=len(endpoint_zero_labels),
                    endpoint_present_zero_count=len(endpoint_present_zero_labels),
                    both_assigned_fraction=both_fraction,
                    both_unassigned_fraction=both_unassigned_fraction,
                ),
                "assignment_rows": rows,
                "both_assigned_fraction": both_fraction if rows else "",
                "one_unassigned_fraction": one / rows if rows else "",
                "both_unassigned_fraction": both_unassigned_fraction if rows else "",
                "malformed_assignment_rows": malformed,
                "assignment_rows_skipped_after_cap": skipped_after_cap,
                "assignment_parse_cap": args.max_assignment_rows,
                "matrix_density_recomputed": density,
                "zero_row_count_recomputed": len(zero_labels),
                "zero_rows_without_endpoint_count": len(endpoint_zero_labels),
                "zero_rows_with_endpoint_count": len(endpoint_present_zero_labels),
                "zero_rows_without_endpoint_first30": " ".join(str(x) for x in endpoint_zero_labels[:30]),
                "zero_rows_with_endpoint_first30": " ".join(str(x) for x in endpoint_present_zero_labels[:30]),
                "ledger_best_density": led.get("best_density", ""),
                "ledger_best_zero_rows": led.get("best_zero_rows", ""),
                "qc_reasons": led.get("qc_reasons", ""),
            }
        )

    label_rows: list[dict[str, Any]] = []
    for label in valid_labels:
        node = nodes.get(label, {})
        counts = label_counter[label]
        label_rows.append(
            {
                "aal3_label": label,
                "aal3_name": node.get("aal3_name", ""),
                "node_qc_recommendation": node.get("aal3cc_action_recommendation", ""),
                "node_qc_reason": node.get("aal3cc_reason", ""),
                "node_qc_zero_row_frequency": node.get("zero_row_frequency", ""),
                "node_qc_label_survival_frequency": node.get("label_survival_frequency", ""),
                "selected_subjects_zero_row": counts.get("zero_subjects_in_selected_matrix", 0),
                "selected_subjects_with_endpoint": counts.get("subjects_with_endpoint", 0),
                "survives_but_no_endpoint": counts.get("survives_but_no_endpoint", 0),
                "endpoint_present_but_no_weighted_edge": counts.get("endpoint_present_but_no_weighted_edge", 0),
                "label_survival_or_tiny_label": counts.get("label_survival_or_tiny_label", 0),
                "historically_recurrent_zero": counts.get("historically_recurrent_zero", 0),
                "endpoint_supported": counts.get("endpoint_supported", 0),
                "total_endpoint_hits": label_endpoint_sum.get(label, 0),
                "total_incident_streamlines": label_incident_sum.get(label, 0),
            }
        )
    label_rows.sort(
        key=lambda r: (
            int(r["selected_subjects_zero_row"]),
            int(r["survives_but_no_endpoint"]),
            fnum(r["node_qc_zero_row_frequency"], 0),
        ),
        reverse=True,
    )

    subject_path = out_dir / f"aal3_endpoint_subject_audit_{stamp}.csv"
    label_path = out_dir / f"aal3_endpoint_label_audit_{stamp}.csv"
    summary_path = out_dir / f"aal3_endpoint_audit_summary_{stamp}.json"
    write_csv(subject_path, subject_rows)
    write_csv(label_path, label_rows)
    latest_subject = out_dir / "aal3_endpoint_subject_audit_latest.csv"
    latest_label = out_dir / "aal3_endpoint_label_audit_latest.csv"
    latest_summary = out_dir / "aal3_endpoint_audit_summary_latest.json"
    shutil.copy2(subject_path, latest_subject)
    shutil.copy2(label_path, latest_label)
    summary = {
        "generated_utc": stamp,
        "run_root": str(args.run_root),
        "subjects_audited": len(subject_rows),
        "subjects_with_unresolved_zero_rows_gt15": unresolved_subjects,
        "audit_status_counts": dict(Counter(r.get("audit_status", "") for r in subject_rows)),
        "corrective_class_counts": dict(Counter(r.get("corrective_class", "") for r in subject_rows)),
        "top_zero_row_labels": label_rows[:20],
        "subject_csv": str(subject_path),
        "label_csv": str(label_path),
        "latest_subject_csv": str(latest_subject),
        "latest_label_csv": str(latest_label),
    }
    summary_text = json.dumps(summary, indent=2, default=str)
    summary_path.write_text(summary_text)
    latest_summary.write_text(summary_text)
    print(summary_text)


if __name__ == "__main__":
    main()
