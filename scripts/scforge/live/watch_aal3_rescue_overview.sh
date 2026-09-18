#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-/data/derivatives/qc/sc_matrix_qc/aal3_force_connectome_ad_batch}"

clear 2>/dev/null || true
date -u +"AAL3 rescue overview | %Y-%m-%dT%H:%M:%SZ"
echo "root: $ROOT"
echo ""

python - "$ROOT" <<'PY'
from __future__ import annotations

import csv
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(sys.argv[1])
LOCAL_QC = Path("/home/ec2-user/exp/data/derivatives/qc/sc_matrix_qc")
GROUP_ORDER = ["AD", "CN", "MCI"]
ROUTES = [
    ("Route 1", ""),
    ("Route 2", "_route2"),
    ("Route 3", "_route3_assignment"),
    ("Route 4", "_route4_label_rescue"),
    ("Route 5", "_route5_deep_assignment"),
    ("Route 6", "_route6_all_voxels_diagnostic"),
    ("Route 7", "_route7_selective_track_rescue"),
    ("Route 8", "_route8_full_t1_fnirt"),
    ("Route 9", "_route9_route8_assignment_retry"),
    ("Route 10", "_route10_endpoint_overlay_triage"),
    ("Route 11A", "_route11a_space_affine_registration_repair"),
    ("Route 11B", "_route11b_act_sift2_track_coverage_rerun"),
    ("Route 11C", "_route11c_zero_row_coverage_rescue"),
    ("Route 12A", "_route12a_tckgen_fallback_rescue"),
    ("Route 12B", "_route12b_zero_row_targeted_rerun"),
    ("Route 12C", "_route12c_registration_contract_repair"),
    ("Route 13", "_route13_step7_preflight_rebuild"),
    ("Route 14", "_route14_step7_rebuilt_track_coverage"),
    ("Route 14B", "_route14b_step7_assignment_retry"),
    ("Route 15", "_route15_step7_dense_track_coverage"),
    ("Route 16", "_route16_zero_roi_hybrid"),
    ("Route 17", "_route17_step7_dense_unpushed"),
    ("Route 18", "_route18_gmwmi_dense_track_coverage"),
    ("Route 19", "_route19_gmwmi_3m_retry"),
    ("Route 20", "_route20_zero_roi_gmwmi_endpoint"),
    ("Route 21", "_route21_endpoint_shell_assignment_map"),
]

PASS_DECISIONS = {
    "PROMOTE_CANDIDATE_PENDING_VISUAL",
    "INTERIM_REVIEW_PROMOTION_PENDING_VISUAL",
    "KEEP_PRODUCTION_NUMERIC_OK",
}
FAIL_DECISIONS = {"TRY_ANOTHER_ROUTE", "REVIEW_ROUTE"}


def route_display(route_name: str) -> str:
    return route_name.replace("Route ", "R")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists() or path.stat().st_size <= 0:
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size <= 0:
        return []
    try:
        with path.open(newline="", encoding="utf-8", errors="replace") as handle:
            return list(csv.DictReader(handle))
    except Exception:
        return []


def manifest_subject_filter(route_root: Path) -> set[str] | None:
    manifest = read_json(route_root / "run_manifest.json")
    subjects: list[str] = []
    for key in ("subject_subset", "subjects", "selected_subjects", "run_subjects"):
        raw = manifest.get(key)
        if isinstance(raw, list):
            subjects.extend(str(value) for value in raw if str(value).strip())
    if not subjects:
        return None
    return {subject.strip() for subject in subjects if subject.strip()}


def filter_rows_to_manifest_subjects(route_root: Path, rows: list[dict[str, str]]) -> list[dict[str, str]]:
    allowed = manifest_subject_filter(route_root)
    if not allowed:
        return rows
    filtered: list[dict[str, str]] = []
    for row in rows:
        sid = row.get("sid") or row.get("subject") or row.get("subject_id") or ""
        if sid in allowed:
            filtered.append(row)
    return filtered


def route_dir_names() -> set[str]:
    names: set[str] = set()
    for path in ROOT.iterdir() if ROOT.exists() else []:
        if not path.is_dir():
            continue
        for _, suffix in ROUTES[1:]:
            if suffix and suffix in path.name:
                names.add(path.name)
                break
    return names


def discover_base_roots() -> list[Path]:
    if not ROOT.exists():
        return []
    excluded = route_dir_names()
    roots: list[Path] = []
    for path in ROOT.iterdir():
        if not path.is_dir() or path.name in excluded:
            continue
        if (path / "subjects.csv").exists() or (path / "run_manifest.json").exists() or (path / "status.json").exists():
            roots.append(path)
    return sorted(roots, key=lambda p: p.name)


def normalize_group(value: str) -> str:
    group = (value or "").strip().upper()
    if group in {"AD", "CN", "MCI"}:
        return group
    return "UNKNOWN"


def subject_group_map(base_root: Path, route_root: Path) -> dict[str, str]:
    rows = read_csv(route_root / "subjects.csv") or read_csv(base_root / "subjects.csv")
    groups: dict[str, str] = {}
    for row in rows:
        sid = row.get("sid") or row.get("subject") or row.get("subject_id") or ""
        if not sid:
            continue
        groups[sid] = normalize_group(row.get("group", ""))
    return groups


def result_rows(route_root: Path) -> list[dict[str, str]]:
    for name in ["batch_results_incremental.csv", "batch_results.csv", "route_qc_decisions.csv"]:
        rows = read_csv(route_root / name)
        if rows:
            return filter_rows_to_manifest_subjects(route_root, rows)
    return []


def done_marker_rows(route_root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    subjects_dir = route_root / "subjects"
    if not subjects_dir.exists():
        return rows
    for marker in subjects_dir.glob("*/done.json"):
        payload = read_json(marker)
        sid = str(payload.get("sid") or marker.parent.name)
        rows.append({"sid": sid, "group": str(payload.get("group") or "")})
    return filter_rows_to_manifest_subjects(route_root, rows)


def route_progress(base_root: Path, route_root: Path) -> dict[str, tuple[int, int, str]]:
    status = read_json(route_root / "status.json")
    if not route_root.exists() and not status:
        return {}

    base_subject_groups = subject_group_map(base_root, base_root)
    route_subject_rows = read_csv(route_root / "subjects.csv")
    subject_groups = subject_group_map(base_root, route_root)
    total_by_group = {group: 0 for group in GROUP_ORDER}
    status_total = int(status.get("total") or 0)
    status_done = int(status.get("completed") or 0)
    phase = str(status.get("phase") or ("not started" if not route_root.exists() else ""))

    if route_subject_rows:
        for group in subject_groups.values():
            if group in total_by_group:
                total_by_group[group] += 1
    elif route_root == base_root:
        for group in base_subject_groups.values():
            if group in total_by_group:
                total_by_group[group] += 1
    else:
        if status_total:
            active_groups: set[str] = set()
            active_sids = status.get("active") or []
            if isinstance(active_sids, list):
                for sid_value in active_sids:
                    sid = str(sid_value).split(":", 1)[0]
                    group = normalize_group(subject_groups.get(sid) or base_subject_groups.get(sid, ""))
                    if group in total_by_group:
                        active_groups.add(group)
            if len(active_groups) == 1:
                total_by_group[next(iter(active_groups))] = status_total
        base_groups = sorted({group for group in base_subject_groups.values() if group in total_by_group})
        if status_total and len(base_groups) == 1:
            total_by_group[base_groups[0]] = status_total

    rows = result_rows(route_root) + done_marker_rows(route_root)
    done_sids_by_group: dict[str, set[str]] = {group: set() for group in GROUP_ORDER}
    for row in rows:
        sid = row.get("sid") or row.get("subject") or ""
        if not sid:
            continue
        group = normalize_group(row.get("group") or subject_groups.get(sid, ""))
        if group in done_sids_by_group:
            done_sids_by_group[group].add(sid)

    done_by_group = {group: len(done_sids_by_group[group]) for group in GROUP_ORDER}
    if "_route13_step7_preflight_rebuild" in route_root.name:
        stages = [stage.strip() for stage in str(status.get("stages") or "").split(",") if stage.strip()]
        target_stage = str(status.get("active_stage") or "")
        if not target_stage:
            target_stage = str(status.get("last_completed_stage") or "")
        if not target_stage and str(status.get("phase") or "").startswith("complete") and stages:
            target_stage = stages[-1]
        if target_stage:
            done_sids_by_group = {group: set() for group in GROUP_ORDER}
            for row in read_csv(route_root / "step7_summary.csv"):
                if (row.get("stage") or "") != target_stage:
                    continue
                if (row.get("status") or "") not in {"ok", "error"}:
                    continue
                sid = row.get("sid") or ""
                if not sid:
                    continue
                group = normalize_group(row.get("group") or subject_groups.get(sid, ""))
                if group in done_sids_by_group:
                    done_sids_by_group[group].add(sid)
            done_by_group = {group: len(done_sids_by_group[group]) for group in GROUP_ORDER}
            if phase == "running":
                phase = f"running:{target_stage}"
            elif phase:
                phase = f"{phase}:{target_stage}"
    if (
        status_done
        and not manifest_subject_filter(route_root)
        and len([g for g, total in total_by_group.items() if total > 0]) == 1
    ):
        group = [g for g, total in total_by_group.items() if total > 0][0]
        done_by_group[group] = max(done_by_group[group], status_done)
    if phase.startswith("complete") and any(
        total_by_group[group] and done_by_group[group] < total_by_group[group] for group in GROUP_ORDER
    ):
        phase = "incomplete"

    return {
        group: (done_by_group[group], total_by_group[group], phase)
        for group in GROUP_ORDER
        if total_by_group[group] or done_by_group[group]
    }


def promotion_subjects(base_root: Path) -> dict[str, set[str]]:
    pushed: dict[str, set[str]] = {group: set() for group in GROUP_ORDER}
    base_groups = subject_group_map(base_root, base_root)
    for _, suffix in ROUTES:
        route_root = base_root.parent / f"{base_root.name}{suffix}"
        route_groups = subject_group_map(base_root, route_root)
        for row in read_csv(route_root / "production_promotion_manifest.csv"):
            if row.get("status") != "promoted":
                continue
            sid = row.get("sid") or ""
            if not sid:
                continue
            group = normalize_group(row.get("group") or route_groups.get(sid) or base_groups.get(sid, ""))
            if group in pushed:
                pushed[group].add(sid)
    return pushed


def promotion_counts(base_root: Path) -> dict[str, int]:
    return {group: len(values) for group, values in promotion_subjects(base_root).items()}


def promotion_qc_counts(base_root: Path) -> dict[str, dict[str, int]]:
    counts = {group: {"full_qc": 0, "review_push": 0} for group in GROUP_ORDER}
    ledger_rows = read_csv(base_root / "route_subject_ledger.csv")
    if ledger_rows:
        for row in ledger_rows:
            sid = row.get("sid", "")
            group = normalize_group(row.get("group", ""))
            if not sid or group not in counts:
                continue
            if not row.get("promoted_route") and not row.get("promotion_class"):
                continue
            if row.get("state") == "solved_full_qc_production" or row.get("final_qc_status") == "solved":
                counts[group]["full_qc"] += 1
            else:
                counts[group]["review_push"] += 1
        return counts

    base_groups = subject_group_map(base_root, base_root)
    latest: dict[str, dict[str, str]] = {}
    for _, suffix in ROUTES:
        route_root = base_root.parent / f"{base_root.name}{suffix}"
        route_groups = subject_group_map(base_root, route_root)
        for row in read_csv(route_root / "production_promotion_manifest.csv"):
            if row.get("status") != "promoted":
                continue
            sid = row.get("sid") or ""
            if not sid:
                continue
            row = dict(row)
            row["_group"] = normalize_group(row.get("group") or route_groups.get(sid) or base_groups.get(sid, ""))
            previous = latest.get(sid)
            if previous is None or row.get("promoted_utc", "") >= previous.get("promoted_utc", ""):
                latest[sid] = row
    for row in latest.values():
        group = normalize_group(row.get("_group", ""))
        if group not in counts:
            continue
        if row.get("final_qc_status") == "solved" or row.get("promotion_class") == "final_qc_pass":
            counts[group]["full_qc"] += 1
        else:
            counts[group]["review_push"] += 1
    return counts


def route_decision_status_counts(base_root: Path) -> dict[str, dict[str, int]]:
    counts = {route_name: {"pass": 0, "fail": 0, "rows": 0} for route_name, _ in ROUTES}
    for route_name, suffix in ROUTES:
        route_root = base_root.parent / f"{base_root.name}{suffix}"
        for row in filter_rows_to_manifest_subjects(route_root, read_csv(route_root / "route_qc_decisions.csv")):
            decision = row.get("qc_decision", "")
            if not decision:
                continue
            counts[route_name]["rows"] += 1
            if decision in PASS_DECISIONS:
                counts[route_name]["pass"] += 1
            elif decision in FAIL_DECISIONS:
                counts[route_name]["fail"] += 1
            else:
                counts[route_name]["fail"] += 1
    return counts


def active_process_summary() -> dict[str, list[str]]:
    try:
        proc = subprocess.check_output(["ps", "-eww", "-o", "pid,etime,args"], text=True)
    except Exception:
        return {}
    active: dict[str, list[str]] = {}
    for line in proc.splitlines():
        if (
            "tckgen" not in line
            and "tcksift2" not in line
            and "tck2connectome" not in line
            and "tckmap" not in line
            and "run_aal3_" not in line
        ):
            continue
        if "aal3_force_connectome_ad_batch" not in line and "aal3_source_contract_probe" not in line:
            continue
        label = "connectome"
        if "route9" in line:
            label = "AD Route 9"
        elif "route10" in line:
            label = "AD Route 10"
        elif "route8" in line:
            label = "AD Route 8"
        elif "cn_mci_existing_tracks" in line:
            label = "CN/MCI Route 1"
        elif "ad_existing_tracks" in line:
            label = "AD Route 1"
        parts = line.strip().split(maxsplit=2)
        if len(parts) == 3:
            active.setdefault(label, []).append(f"{parts[0]} {parts[1]}")
    return active


def running_subject_counts(subject_groups: dict[str, str]) -> dict[str, int]:
    rows = running_subject_rows(subject_groups, [])
    counts = {group: 0 for group in GROUP_ORDER}
    for row in rows:
        counts[row["group"]] += 1
    return counts


def running_subject_rows(subject_groups: dict[str, str], base_roots: list[Path]) -> list[dict[str, str]]:
    try:
        proc = subprocess.check_output(["ps", "-eww", "-o", "pid,etime,args"], text=True)
    except Exception:
        return []

    active: dict[tuple[str, str, str], dict[str, str]] = {}
    # FNIRT inputs append suffixes such as "_t1_ras" immediately after the
    # image id, so a trailing word-boundary misses active Route 8 subjects.
    sid_pattern = re.compile(r"(?<![A-Za-z0-9])\d{3}_S_\d{4}_I\d+(?=[^0-9]|$)")
    compute_terms = (
        "tck2connectome",
        "tckgen",
        "tcksift2",
        "tckedit",
        "tckmap",
        "fnirt",
        "run_sc_aal3_source_contract_probe.py",
        "run_aal3_route2_bbr_batch.py",
        "run_aal3_route3_assignment_batch.py",
        "run_aal3_route4_label_rescue_batch.py",
        "run_aal3_route5_deep_assignment_batch.py",
        "run_aal3_route6_all_voxels_diagnostic_batch.py",
        "run_aal3_route7_selective_rescue_batch.py",
        "run_aal3_route8_full_t1_fnirt_batch.py",
        "run_aal3_route9_route8_assignment_retry_batch.py",
        "run_aal3_route10_endpoint_overlay_triage.py",
        "run_aal3_route11b_act_sift2_track_coverage_rerun.py",
        "run_aal3_route11c_zero_row_coverage_rescue.py",
        "run_aal3_route12a_tckgen_fallback_rescue.py",
        "run_aal3_route12b_zero_row_targeted_rerun.py",
        "run_aal3_route12c_registration_contract_repair.py",
        "run_aal3_route13_step7_preflight_rebuild.py",
        "run_aal3_route14_step7_rebuilt_track_coverage.py",
        "run_aal3_route14b_assignment_retry.py",
        "run_aal3_route15_step7_dense_track_coverage.py",
        "run_aal3_route16_zero_roi_hybrid.py",
        "run_aal3_route17_step7_dense_unpushed.py",
        "run_aal3_route18_gmwmi_dense_track_coverage.py",
        "run_aal3_route19_gmwmi_3m_retry.py",
        "run_aal3_route20_zero_roi_gmwmi_endpoint.py",
        "run_aal3_route21_endpoint_shell_assignment_map.py",
        "run_aal3_force_connectome_ad_batch.py",
    )
    scope_terms = (
        "aal3_force_connectome_ad_batch",
        "aal3_source_contract_probe",
        "aal3_route2_bbr_probe",
        "aal3_route3_assignment_probe",
        "aal3_route4_label_rescue_probe",
        "aal3_route5_deep_assignment_probe",
        "aal3_route6_all_voxels_diagnostic_probe",
        "aal3_route7_selective_rescue_probe",
        "aal3_route8_full_t1_fnirt_probe",
        "aal3_route9_route8_assignment_retry_probe",
        "aal3_route11b_act_sift2_track_coverage_rerun_probe",
        "aal3_route11c_zero_row_coverage_rescue_probe",
        "aal3_route12a_tckgen_fallback_rescue_probe",
        "aal3_route12b_zero_row_targeted_rerun_probe",
        "aal3_route12c_registration_contract_repair_probe",
        "aal3_route14_step7_rebuilt_track_coverage_probe",
        "aal3_route14b_step7_assignment_retry_probe",
        "aal3_route15_step7_dense_track_coverage_probe",
        "aal3_route16_zero_roi_hybrid_probe",
        "aal3_route17_step7_dense_unpushed_probe",
        "aal3_route18_gmwmi_dense_track_coverage_probe",
        "aal3_route19_gmwmi_3m_retry_probe",
        "aal3_route20_zero_roi_gmwmi_endpoint_probe",
        "aal3_route21_endpoint_shell_assignment_map_probe",
        "route13_step7_preflight_rebuild",
        "route10_endpoint_overlay_triage",
    )

    for line in proc.splitlines():
        # Wrapper commands can include multiple queued --subjects even though
        # only one worker is active. Status files and MRtrix children provide
        # the actual active subject rows for these lanes.
        if "run_aal3_route11b_act_sift2_track_coverage_rerun.py" in line:
            continue
        if "run_aal3_route11c_zero_row_coverage_rescue.py" in line:
            continue
        if "run_aal3_route12a_tckgen_fallback_rescue.py" in line:
            continue
        if "run_aal3_route12b_zero_row_targeted_rerun.py" in line:
            continue
        if "run_aal3_route12c_registration_contract_repair.py" in line:
            continue
        if "run_aal3_route14_step7_rebuilt_track_coverage.py" in line:
            continue
        if "run_aal3_route14b_assignment_retry.py" in line:
            continue
        if "run_aal3_route15_step7_dense_track_coverage.py" in line:
            continue
        if "run_aal3_route16_zero_roi_hybrid.py" in line:
            continue
        if "run_aal3_route17_step7_dense_unpushed.py" in line:
            continue
        if "run_aal3_route18_gmwmi_dense_track_coverage.py" in line:
            continue
        if "run_aal3_route19_gmwmi_3m_retry.py" in line:
            continue
        if "run_aal3_route20_zero_roi_gmwmi_endpoint.py" in line:
            continue
        if "run_aal3_route21_endpoint_shell_assignment_map.py" in line:
            continue
        if not any(term in line for term in compute_terms):
            continue
        if not any(term in line for term in scope_terms):
            continue
        route = "Route 1"
        if "route21" in line:
            route = "Route 21"
        elif "route20" in line:
            route = "Route 20"
        elif "route19" in line:
            route = "Route 19"
        elif "route18" in line:
            route = "Route 18"
        elif "route17" in line:
            route = "Route 17"
        elif "route16" in line:
            route = "Route 16"
        elif "route15" in line:
            route = "Route 15"
        elif "route14b" in line:
            route = "Route 14B"
        elif "route14" in line:
            route = "Route 14"
        elif "route13" in line:
            route = "Route 13"
        elif "route12c" in line:
            route = "Route 12C"
        elif "route12b" in line:
            route = "Route 12B"
        elif "route12a" in line:
            route = "Route 12A"
        elif "route10" in line:
            route = "Route 10"
        elif "route11c" in line:
            route = "Route 11C"
        elif "route11b" in line:
            route = "Route 11B"
        elif "route9" in line:
            route = "Route 9"
        elif "route8" in line:
            route = "Route 8"
        elif "route7" in line:
            route = "Route 7"
        elif "route6" in line:
            route = "Route 6"
        elif "route5" in line:
            route = "Route 5"
        elif "route4" in line:
            route = "Route 4"
        elif "route3" in line:
            route = "Route 3"
        elif "route2" in line:
            route = "Route 2"
        for sid in sid_pattern.findall(line):
            group = normalize_group(subject_groups.get(sid, ""))
            if group not in GROUP_ORDER:
                continue
            status = status_for_active_subject(base_roots, route, sid)
            phase, pct = progress_from_status(status)
            if phase.startswith("complete"):
                phase = infer_process_stage(line)
                pct = 10
            active[(group, route, sid)] = {
                "group": group,
                "route": route,
                "sid": sid,
                "phase": phase,
                "pct": str(pct),
            }
    return sorted(active.values(), key=lambda row: (GROUP_ORDER.index(row["group"]), row["route"], row["sid"]))


def infer_process_stage(line: str) -> str:
    if "tckgen" in line:
        return "tckgen"
    if "tcksift2" in line:
        return "tcksift2"
    if "tck2connectome" in line:
        if "assignment_forward_search 120" in line:
            return "connectome_candidates/forward120"
        if "assignment_forward_search 80" in line:
            return "connectome_candidates/forward80"
        if "assignment_forward_search 40" in line:
            return "connectome_candidates/forward40"
        return "connectome_candidates"
    if "fnirt" in line:
        return "fnirt"
    return "running"


def process_has_route_subject(proc_text: str, route_name: str, sid: str) -> bool:
    route_token = route_name.lower().replace("route ", "route").replace(" ", "")
    for line in proc_text.splitlines():
        lower = line.lower()
        if sid not in line:
            continue
        if route_token in lower:
            return True
        if route_name == "Route 1" and (
            "run_aal3_force_connectome" in lower or "aal3_source_contract" in lower
        ):
            return True
    return False


def status_active_subject_rows(subject_groups: dict[str, str], base_roots: list[Path]) -> list[dict[str, str]]:
    active: dict[tuple[str, str, str], dict[str, str]] = {}
    try:
        proc_text = subprocess.check_output(["ps", "-eww", "-o", "args"], text=True)
    except Exception:
        proc_text = ""
    for base_root in base_roots:
        for route_name, suffix in ROUTES:
            route_root = base_root.parent / f"{base_root.name}{suffix}"
            status = read_json(route_root / "status.json")
            active_sids = status.get("active") or []
            if route_name == "Route 13" and not active_sids and str(status.get("phase") or "") == "running":
                stage = str(status.get("active_stage") or "running")
                sid_pattern = re.compile(r"(?<![A-Za-z0-9])\d{3}_S_\d{4}_I\d+(?=[^0-9]|$)")
                live_sids: set[str] = set()
                route13_terms = (
                    "5ttgen",
                    "dwi2response",
                    "dwi2fod",
                    "mtnormalise",
                    "mrconvert",
                    "run_aal3_route13_step7_preflight_rebuild.py",
                )
                for line in proc_text.splitlines():
                    if not any(term in line for term in route13_terms):
                        continue
                    for sid in sid_pattern.findall(line):
                        live_sids.add(sid)
                active_sids = [f"{sid}:{stage}" for sid in sorted(live_sids)]
            if not isinstance(active_sids, list):
                continue
            for sid_value in active_sids:
                sid_raw = str(sid_value)
                sid = sid_raw.split(":", 1)[0]
                if not sid:
                    continue
                group = normalize_group(subject_groups.get(sid, ""))
                if group not in GROUP_ORDER:
                    continue
                subject_status = status_for_active_subject(base_roots, route_name, sid)
                status_mtime = float(subject_status.get("_status_mtime") or 0)
                # Long MRtrix calls do not update status.json until they finish.
                # Keep them visible longer than the 45-minute connectome cap, but
                # still drop genuinely stale rows left behind by killed wrappers.
                if status_mtime and time.time() - status_mtime > 75 * 60:
                    continue
                if proc_text and not process_has_route_subject(proc_text, route_name, sid):
                    if route_name != "Route 13":
                        continue
                    route13_live = "run_aal3_route13_step7_preflight_rebuild.py" in proc_text
                    sid_live = sid in proc_text
                    if not route13_live and not sid_live:
                        continue
                phase, pct = progress_from_status(subject_status or status)
                if ":" in sid_raw and not subject_status.get("active_metric") and not subject_status.get("active_variant"):
                    stage = sid_raw.split(":", 1)[1].strip()
                    if stage:
                        phase = stage
                if phase.startswith("complete"):
                    continue
                active[(group, route_name, sid)] = {
                    "group": group,
                    "route": route_name,
                    "sid": sid,
                    "phase": phase,
                    "pct": str(pct),
                }
    return sorted(active.values(), key=lambda row: (GROUP_ORDER.index(row["group"]), row["route"], row["sid"]))


def status_for_active_subject(base_roots: list[Path], route: str, sid: str) -> dict[str, Any]:
    candidates: list[Path] = []
    for base_root in base_roots:
        name = base_root.name
        if route == "Route 2":
            candidates.append(LOCAL_QC / "aal3_route2_bbr_probe" / f"{name}_route2_{sid}" / "status.json")
        elif route == "Route 3":
            candidates.append(LOCAL_QC / "aal3_route3_assignment_probe" / f"{name}_route3_assignment_{sid}" / "status.json")
        elif route == "Route 4":
            candidates.append(LOCAL_QC / "aal3_route4_label_rescue_probe" / f"{name}_route4_label_rescue_{sid}" / "status.json")
        elif route == "Route 5":
            candidates.append(LOCAL_QC / "aal3_route5_deep_assignment_probe" / f"{name}_route5_deep_assignment_{sid}" / "status.json")
        elif route == "Route 6":
            candidates.append(LOCAL_QC / "aal3_route6_all_voxels_diagnostic_probe" / f"{name}_route6_all_voxels_diagnostic_{sid}" / "status.json")
        elif route == "Route 7":
            candidates.append(LOCAL_QC / "aal3_route7_selective_rescue_probe" / f"{name}_route7_selective_track_rescue_{sid}" / "status.json")
        elif route == "Route 8":
            candidates.append(LOCAL_QC / "aal3_route8_full_t1_fnirt_probe" / f"{name}_route8_full_t1_fnirt_{sid}" / "status.json")
        elif route == "Route 9":
            candidates.append(
                LOCAL_QC
                / "aal3_route9_route8_assignment_retry_probe"
                / f"{name}_route9_route8_assignment_retry_{sid}"
                / "status.json"
            )
        elif route == "Route 10":
            candidates.append(
                base_root.parent
                / f"{name}_route10_endpoint_overlay_triage"
                / "subjects"
                / sid
                / "status.json"
            )
        elif route == "Route 11A":
            candidates.append(
                LOCAL_QC
                / "aal3_route11a_space_affine_registration_repair_probe"
                / f"{name}_route11a_space_affine_registration_repair_{sid}"
                / "status.json"
            )
        elif route == "Route 11B":
            candidates.append(
                LOCAL_QC
                / "aal3_route11b_act_sift2_track_coverage_rerun_probe"
                / f"{name}_route11b_act_sift2_track_coverage_rerun_{sid}"
                / "status.json"
            )
        elif route == "Route 11C":
            candidates.append(
                LOCAL_QC
                / "aal3_route11c_zero_row_coverage_rescue_probe"
                / f"{name}_route11c_zero_row_coverage_rescue_{sid}"
                / "status.json"
            )
        elif route == "Route 12A":
            candidates.append(
                LOCAL_QC
                / "aal3_route12a_tckgen_fallback_rescue_probe"
                / f"{name}_route12a_tckgen_fallback_rescue_{sid}"
                / "status.json"
            )
        elif route == "Route 12B":
            candidates.append(
                LOCAL_QC
                / "aal3_route12b_zero_row_targeted_rerun_probe"
                / f"{name}_route12b_zero_row_targeted_rerun_{sid}"
                / "status.json"
            )
        elif route == "Route 12C":
            candidates.append(
                LOCAL_QC
                / "aal3_route12c_registration_contract_repair_probe"
                / f"{name}_route12c_registration_contract_repair_{sid}"
                / "status.json"
            )
        elif route == "Route 13":
            candidates.append(base_root.parent / f"{name}_route13_step7_preflight_rebuild" / "status.json")
        elif route == "Route 14":
            candidates.append(
                LOCAL_QC
                / "aal3_route14_step7_rebuilt_track_coverage_probe"
                / f"{name}_route14_step7_rebuilt_track_coverage_{sid}"
                / "status.json"
            )
        elif route == "Route 14B":
            candidates.append(
                LOCAL_QC
                / "aal3_route14b_step7_assignment_retry_probe"
                / f"{name}_route14b_step7_assignment_retry_{sid}"
                / "status.json"
            )
        elif route == "Route 15":
            candidates.append(
                LOCAL_QC
                / "aal3_route15_step7_dense_track_coverage_probe"
                / f"{name}_route15_step7_dense_track_coverage_{sid}"
                / "status.json"
            )
        elif route == "Route 16":
            candidates.append(
                LOCAL_QC
                / "aal3_route16_zero_roi_hybrid_probe"
                / f"{name}_route16_zero_roi_hybrid_{sid}"
                / "status.json"
            )
        elif route == "Route 17":
            candidates.append(
                LOCAL_QC
                / "aal3_route17_step7_dense_unpushed_probe"
                / f"{name}_route17_step7_dense_unpushed_{sid}"
                / "status.json"
            )
        elif route == "Route 18":
            candidates.append(
                LOCAL_QC
                / "aal3_route18_gmwmi_dense_track_coverage_probe"
                / f"{name}_route18_gmwmi_dense_track_coverage_{sid}"
                / "status.json"
            )
        elif route == "Route 19":
            candidates.append(
                LOCAL_QC
                / "aal3_route19_gmwmi_3m_retry_probe"
                / f"{name}_route19_gmwmi_3m_retry_{sid}"
                / "status.json"
            )
        elif route == "Route 20":
            candidates.append(
                LOCAL_QC
                / "aal3_route20_zero_roi_gmwmi_endpoint_probe"
                / f"{name}_route20_zero_roi_gmwmi_endpoint_{sid}"
                / "status.json"
            )
        elif route == "Route 21":
            candidates.append(
                LOCAL_QC
                / "aal3_route21_endpoint_shell_assignment_map_probe"
                / f"{name}_route21_endpoint_shell_assignment_map_{sid}"
                / "status.json"
            )
        else:
            candidates.append(LOCAL_QC / "aal3_source_contract_probe" / f"{name}_{sid}" / "status.json")

    existing = [path for path in candidates if path.exists() and path.stat().st_size > 0]
    if not existing:
        for parent in [
            LOCAL_QC / "aal3_source_contract_probe",
            LOCAL_QC / "aal3_route2_bbr_probe",
            LOCAL_QC / "aal3_route3_assignment_probe",
            LOCAL_QC / "aal3_route4_label_rescue_probe",
            LOCAL_QC / "aal3_route5_deep_assignment_probe",
            LOCAL_QC / "aal3_route6_all_voxels_diagnostic_probe",
            LOCAL_QC / "aal3_route7_selective_rescue_probe",
            LOCAL_QC / "aal3_route8_full_t1_fnirt_probe",
            LOCAL_QC / "aal3_route9_route8_assignment_retry_probe",
            LOCAL_QC / "aal3_route12a_tckgen_fallback_rescue_probe",
            LOCAL_QC / "aal3_route12b_zero_row_targeted_rerun_probe",
            LOCAL_QC / "aal3_route12c_registration_contract_repair_probe",
            LOCAL_QC / "aal3_route14_step7_rebuilt_track_coverage_probe",
            LOCAL_QC / "aal3_route14b_step7_assignment_retry_probe",
            LOCAL_QC / "aal3_route15_step7_dense_track_coverage_probe",
            LOCAL_QC / "aal3_route16_zero_roi_hybrid_probe",
            LOCAL_QC / "aal3_route17_step7_dense_unpushed_probe",
            LOCAL_QC / "aal3_route18_gmwmi_dense_track_coverage_probe",
            LOCAL_QC / "aal3_route19_gmwmi_3m_retry_probe",
            LOCAL_QC / "aal3_route20_zero_roi_gmwmi_endpoint_probe",
            LOCAL_QC / "aal3_route21_endpoint_shell_assignment_map_probe",
            ROOT,
        ]:
            existing.extend(parent.glob(f"*_{sid}/status.json"))
    if not existing:
        return {}
    latest = max(existing, key=lambda path: path.stat().st_mtime)
    payload = read_json(latest)
    try:
        payload["_status_mtime"] = latest.stat().st_mtime
        payload["_status_path"] = str(latest)
    except Exception:
        pass
    return payload


def tckgen_log_progress(status: dict[str, Any]) -> int | None:
    try:
        status_path = Path(str(status.get("_status_path") or ""))
    except Exception:
        return None
    if not status_path:
        return None
    requested = int(float(status.get("select_streamlines") or 0))
    if requested <= 0:
        return None
    logs_dir = status_path.parent / "logs"

    def latest_selected(log_path: Path) -> int | None:
        try:
            lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception:
            return None
        for line in reversed(lines[-120:]):
            match = re.search(r"(\d+)\s+selected\b", line)
            if match:
                return int(match.group(1))
        return None

    try:
        shard_count = int(status.get("target_shards") or 0)
    except Exception:
        shard_count = 0
    if shard_count > 1:
        selected_total = 0
        found = False
        for log_path in sorted(logs_dir.glob("tckgen_*_shard*.log")):
            selected = latest_selected(log_path)
            if selected is None:
                continue
            found = True
            selected_total += selected
        if found:
            pct = int(round(100 * selected_total / requested))
            if selected_total > 0 and pct == 0:
                pct = 1
            return max(0, min(99, pct))

    mode = str(status.get("active_track_mode") or "dynamic")
    suffix = f"{requested // 1000}k"
    log_path = logs_dir / f"tckgen_{mode}_{suffix}.log"
    if not log_path.exists():
        candidates = sorted(logs_dir.glob("tckgen_*.log"), key=lambda path: path.stat().st_mtime, reverse=True)
        if not candidates:
            return None
        log_path = candidates[0]
    selected = latest_selected(log_path)
    if selected is not None:
        pct = int(round(100 * selected / requested))
        if selected > 0 and pct == 0:
            pct = 1
        return max(0, min(99, pct))
    return None


def progress_from_status(status: dict[str, Any]) -> tuple[str, int]:
    phase = str(status.get("phase") or "running")
    active_variant = str(status.get("active_variant") or "")
    active_metric = str(status.get("active_metric") or "")
    detail = phase
    if active_variant:
        detail = f"{detail}/{active_variant}"
    if active_metric:
        detail = f"{detail}/{active_metric}"

    try:
        done = float(status.get("connectome_jobs_done") or 0)
        total = float(status.get("connectome_jobs_total") or 0)
    except Exception:
        done, total = 0.0, 0.0
    if phase == "complete":
        return detail, 100
    if phase.startswith("tckgen"):
        log_pct = tckgen_log_progress(status)
        if log_pct is not None:
            return detail, log_pct
    if total > 0:
        return detail, max(0, min(99, int(round(60 + 40 * (done / total)))))

    phase_pct = {
        "starting": 2,
        "baseline": 6,
        "b0_candidate": 12,
        "aal_to_mni": 22,
        "aal_to_true_native_t1": 35,
        "t1_to_b0": 45,
        "aal_t1_to_b0": 55,
        "fnirt_label_rescue": 35,
        "fnirt_nonlinear": 35,
        "full_t1_fnirt": 35,
        "route8_full_t1_fnirt": 35,
        "route8_label_to_b0": 50,
        "route8_label_survival": 55,
        "running": 10,
    }
    return detail, phase_pct.get(phase, 10)


def progress_bar(pct_text: str, width: int = 24) -> str:
    try:
        pct = max(0, min(100, int(float(pct_text))))
    except Exception:
        pct = 0
    filled = int(round(width * pct / 100))
    return "[" + "#" * filled + "." * (width - filled) + f"] {pct:3d}%"


base_roots = discover_base_roots()
if not base_roots:
    print("No AAL3 rescue roots found.")
    raise SystemExit(0)

cells: dict[str, dict[str, list[tuple[int, int, str]]]] = {
    group: {route_name: [] for route_name, _ in ROUTES} for group in GROUP_ORDER
}
pushed_by_group = {group: 0 for group in GROUP_ORDER}
pushed_sids_by_group: dict[str, set[str]] = {group: set() for group in GROUP_ORDER}
full_qc_by_group = {group: 0 for group in GROUP_ORDER}
review_push_by_group = {group: 0 for group in GROUP_ORDER}
route_decision_totals = {route_name: {"pass": 0, "fail": 0, "rows": 0} for route_name, _ in ROUTES}
root_labels: list[str] = []
all_subject_groups: dict[str, str] = {}

for base_root in base_roots:
    root_labels.append(base_root.name)
    all_subject_groups.update(subject_group_map(base_root, base_root))
    pushed = promotion_subjects(base_root)
    for group, sids in pushed.items():
        pushed_sids_by_group[group].update(sids)
        pushed_by_group[group] = len(pushed_sids_by_group[group])
    qc_counts = promotion_qc_counts(base_root)
    for group, values in qc_counts.items():
        full_qc_by_group[group] += values["full_qc"]
        review_push_by_group[group] += values["review_push"]
    for route_name, suffix in ROUTES:
        route_root = base_root.parent / f"{base_root.name}{suffix}"
        progress = route_progress(base_root, route_root)
        for group, item in progress.items():
            cells[group][route_name].append(item)
    status_counts = route_decision_status_counts(base_root)
    for route_name, values in status_counts.items():
        route_decision_totals[route_name]["pass"] += values["pass"]
        route_decision_totals[route_name]["fail"] += values["fail"]
        route_decision_totals[route_name]["rows"] += values["rows"]


def render_cell(items: list[tuple[int, int, str]]) -> str:
    if not items:
        return "-"
    done = sum(item[0] for item in items)
    total = sum(item[1] for item in items)
    display_total = max(done, total)
    phases = {item[2] for item in items if item[2]}
    suffix = "*" if done < total or any("running" in phase for phase in phases) else ""
    return f"{done}/{display_total}{suffix}" if display_total else f"{done}/0{suffix}"


running_by_key: dict[tuple[str, str, str], dict[str, str]] = {}
for row in status_active_subject_rows(all_subject_groups, base_roots) + running_subject_rows(all_subject_groups, base_roots):
    running_by_key[(row["group"], row["route"], row["sid"])] = row
running_rows = sorted(running_by_key.values(), key=lambda row: (GROUP_ORDER.index(row["group"]), row["route"], row["sid"]))
route_order = {route_name: idx for idx, (route_name, _) in enumerate(ROUTES)}
running_jobs_by_group: dict[str, int] = {group: 0 for group in GROUP_ORDER}
running_sids_by_group: dict[str, set[str]] = {group: set() for group in GROUP_ORDER}
running_sids_by_group_route: dict[tuple[str, str], set[str]] = {}
running_sids_by_route: dict[str, set[str]] = {route_name: set() for route_name, _ in ROUTES}
for row in running_rows:
    running_jobs_by_group[row["group"]] += 1
    running_sids_by_group[row["group"]].add(row["sid"])
    running_sids_by_group_route.setdefault((row["group"], row["route"]), set()).add(row["sid"])
for row in running_rows:
    running_sids_by_route.setdefault(row["route"], set()).add(f"{row['group']}:{row['sid']}")
running_by_group = running_jobs_by_group

def route_done_total(group: str, route_name: str) -> tuple[int, int, bool]:
    done = 0
    total = 0
    running = False
    for d, t, phase in cells[group][route_name]:
        done += d
        total += t
        running = running or "running" in phase
        running = running or (phase == "incomplete" and d < t)
    return done, total, running


def render_route_cell(group: str, route_name: str, accounted_override: int | None = None) -> str:
    if not cells[group][route_name]:
        return "-"
    done, total, _running_phase = route_done_total(group, route_name)
    if accounted_override is not None:
        done = max(done, accounted_override)
    display_total = max(done, total)
    running = bool(running_sids_by_group_route.get((group, route_name)))
    suffix = "*" if running or done < total else ""
    return f"{done}/{display_total}{suffix}" if display_total else f"{done}/0{suffix}"


accounted_by_group: dict[str, int] = {}
queued_by_group: dict[str, int] = {}
for group in GROUP_ORDER:
    r1_done, r1_total, _ = route_done_total(group, "Route 1")
    active_unpushed = running_sids_by_group[group] - pushed_sids_by_group[group]
    accounted_by_group[group] = r1_done
    queued_by_group[group] = max(r1_total - pushed_by_group[group] - len(active_unpushed), 0)


headers = ["Group"] + [route_display(route_name) for route_name, _ in ROUTES] + [
    "Unpushed",
    "Running",
    "Pushed",
    "Full QC",
    "Review push",
    "Needs QC",
]
group_rows: list[list[str]] = []
for group in GROUP_ORDER:
    _r1_done, r1_total, _ = route_done_total(group, "Route 1")
    needs_qc = max(r1_total - full_qc_by_group[group], 0)
    group_rows.append(
        [group]
        + [
            render_route_cell(group, route_name)
            for route_name, _ in ROUTES
        ]
        + [str(queued_by_group[group]) if queued_by_group[group] else "-"]
        + [str(running_by_group.get(group, 0)) if running_by_group.get(group, 0) else "-"]
        + [str(pushed_by_group[group])]
        + [str(full_qc_by_group[group]) if full_qc_by_group[group] else "-"]
        + [str(review_push_by_group[group]) if review_push_by_group[group] else "-"]
        + [str(needs_qc) if needs_qc else "-"]
    )

total_row = ["TOTAL"]
for route_name, _ in ROUTES:
    done = 0
    total = 0
    for group in GROUP_ORDER:
        for d, t, phase in cells[group][route_name]:
            done += d
            total += t
    running = any(running_sids_by_group_route.get((group, route_name)) for group in GROUP_ORDER)
    display_total = max(done, total)
    suffix = "*" if running or done < total else ""
    total_row.append(f"{done}/{display_total}{suffix}" if display_total else "-")
total_queued = sum(queued_by_group.values())
total_running = sum(running_by_group.values())
total_row.append(str(total_queued) if total_queued else "-")
total_row.append(str(total_running) if total_running else "-")
total_row.append(str(sum(pushed_by_group.values())))
total_row.append(str(sum(full_qc_by_group.values())) if sum(full_qc_by_group.values()) else "-")
total_row.append(str(sum(review_push_by_group.values())) if sum(review_push_by_group.values()) else "-")
total_needs_qc = sum(max(route_done_total(group, "Route 1")[1] - full_qc_by_group[group], 0) for group in GROUP_ORDER)
total_row.append(str(total_needs_qc) if total_needs_qc else "-")

pass_row = ["PASS"]
fail_row = ["FAIL"]
running_route_row = ["RUNNING"]
for route_name, _ in ROUTES:
    count = len(running_sids_by_route.get(route_name, set()))
    running_route_row.append(str(count) if count else "-")
running_route_row += ["-", str(sum(running_by_group.values())) if sum(running_by_group.values()) else "-", "-", "-", "-", "-"]

for route_name, _ in ROUTES:
    values = route_decision_totals[route_name]
    if values["rows"]:
        pass_row.append(str(values["pass"]))
        fail_row.append(str(values["fail"]))
    else:
        pass_row.append("-")
        fail_row.append("-")
pass_row += ["-", "-", "-", "-", "-", "-"]
fail_row += ["-", "-", "-", "-", "-", "-"]

rows_for_width = group_rows + [running_route_row, pass_row, fail_row, total_row]

widths = [len(header) for header in headers]
for row in rows_for_width:
    for idx, value in enumerate(row):
        widths[idx] = max(widths[idx], len(value))

def print_row(values: list[str]) -> None:
    green = "\033[1;32m"
    reset = "\033[0m"
    pushed_col = headers.index("Pushed")
    cells: list[str] = []
    for idx, value in enumerate(values):
        cell = value.ljust(widths[idx])
        if idx == pushed_col:
            cell = f"{green}{cell}{reset}"
        cells.append(cell)
    print(" | ".join(cells))

def print_separator() -> None:
    print("-+-".join("-" * width for width in widths))

print_row(headers)
print_separator()
for row in group_rows:
    print_row(row)
print_separator()
print_row(running_route_row)
print_separator()
print_row(pass_row)
print_row(fail_row)
print_separator()
print_row(total_row)

if running_rows:
    print("")
    print("Running subjects")
    run_headers = ["Group", "Route", "Subject", "Progress", "Stage"]
    run_rows = [
        [row["group"], route_display(row["route"]), row["sid"], progress_bar(row["pct"]), row["phase"]]
        for row in running_rows
    ]
    run_widths = [len(header) for header in run_headers]
    for row in run_rows:
        for idx, value in enumerate(row):
            run_widths[idx] = max(run_widths[idx], len(value))
    print(" | ".join(value.ljust(run_widths[idx]) for idx, value in enumerate(run_headers)))
    print("-+-".join("-" * width for width in run_widths))
    for row in run_rows:
        print(" | ".join(value.ljust(run_widths[idx]) for idx, value in enumerate(row)))
PY
