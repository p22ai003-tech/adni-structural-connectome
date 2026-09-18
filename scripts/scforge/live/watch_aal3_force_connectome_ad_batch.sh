#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/derivatives/qc/sc_matrix_qc/aal3_force_connectome_ad_batch"
LATEST_FILE="$ROOT/latest_tag.txt"
if [[ $# -gt 0 ]]; then
  RUN_ROOT="$1"
elif [[ -s "$LATEST_FILE" ]]; then
  TAG="$(cat "$LATEST_FILE")"
  RUN_ROOT="$ROOT/$TAG"
else
  RUN_ROOT="$(ls -td "$ROOT"/* 2>/dev/null | head -n 1 || true)"
fi

clear 2>/dev/null || true
date -u +"AAL3 forced AD rescue monitor | %Y-%m-%dT%H:%M:%SZ"
if [[ -z "${RUN_ROOT:-}" || ! -d "$RUN_ROOT" ]]; then
  echo "No run root found."
  exit 0
fi
echo "run root: $RUN_ROOT"

python - "$RUN_ROOT" <<'PY'
from pathlib import Path
import csv, json, subprocess, sys

root = Path(sys.argv[1])

def read_json(path: Path) -> dict:
    if not path.exists() or path.stat().st_size <= 0:
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}

def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size <= 0:
        return []
    try:
        return list(csv.DictReader(path.open()))
    except Exception:
        return []

def bar(count: int, total: int, width: int = 22) -> str:
    fill = int(width * count / total) if total else 0
    return "[" + "#" * fill + "." * (width - fill) + f"] {count}/{total}"

def route_status(route_root: Path) -> tuple[int, int, str]:
    status = read_json(route_root / "status.json")
    completed = int(status.get("completed") or 0)
    failed = int(status.get("failed") or 0)
    total = int(status.get("total") or 0)
    phase = str(status.get("phase") or "not started")
    if phase == "complete" and total and completed < total and completed + failed >= total:
        completed = total
    return completed, total, phase

def has_process(fragment: str) -> bool:
    try:
        subprocess.check_output(["pgrep", "-af", fragment], text=True)
        return True
    except Exception:
        return False

def promotion_rows(route_root: Path) -> list[dict[str, str]]:
    rows = [r for r in read_csv(route_root / "production_promotion_manifest.csv") if r.get("status") == "promoted"]
    for row in rows:
        row["_route"] = route_root.name
    return rows

status = read_json(root / "status.json")
total = int(status.get("total") or 0)
done = int(status.get("completed") or 0)
failed = int(status.get("failed") or 0)
scored = read_csv(root / "route_qc_decisions.csv")

decision_counts: dict[str, int] = {}
for row in scored:
    decision = row.get("qc_decision") or ""
    decision_counts[decision] = decision_counts.get(decision, 0) + 1

route2 = root.parent / f"{root.name}_route2"
route3 = root.parent / f"{root.name}_route3_assignment"
route4 = root.parent / f"{root.name}_route4_label_rescue"
route5 = root.parent / f"{root.name}_route5_deep_assignment"
route6 = root.parent / f"{root.name}_route6_all_voxels_diagnostic"
route7 = root.parent / f"{root.name}_route7_selective_track_rescue"
route8 = root.parent / f"{root.name}_route8_full_t1_fnirt"
route9 = root.parent / f"{root.name}_route9_route8_assignment_retry"
route_roots = [root, route2, route3, route4, route5, route6, route7, route8, route9]
route_ids = ["R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9"]
visual_status_by_route_sid: dict[tuple[str, str], str] = {}
for route_id, route_root in zip(route_ids, route_roots):
    for row in read_csv(route_root / "visual_overlay_qc_manifest.csv"):
        sid = row.get("sid") or ""
        if sid:
            visual_status_by_route_sid[(route_id, sid)] = row.get("visual_overlay_status", "")

priority = {"final_qc_pass": 3, "": 3, "interim_review_not_solved": 2}
unique_promoted: dict[str, dict[str, str]] = {}
event_count = 0
for route_root in route_roots:
    rows = promotion_rows(route_root)
    event_count += len(rows)
    for row in rows:
        sid = row.get("sid") or ""
        if not sid:
            continue
        current = unique_promoted.get(sid)
        current_priority = priority.get(current.get("promotion_class", ""), 1) if current else -1
        row_priority = priority.get(row.get("promotion_class", ""), 1)
        if current is None or row_priority > current_priority:
            unique_promoted[sid] = row

unique_final = sum(1 for row in unique_promoted.values() if row.get("promotion_class") in {"", "final_qc_pass"})
unique_interim = sum(1 for row in unique_promoted.values() if row.get("promotion_class") == "interim_review_not_solved")

decision_rank = {
    "PROMOTE_CANDIDATE_PENDING_VISUAL": 5,
    "INTERIM_REVIEW_PROMOTION_PENDING_VISUAL": 4,
    "KEEP_PRODUCTION_NUMERIC_OK": 3,
    "REVIEW_ROUTE": 2,
    "TRY_ANOTHER_ROUTE": 1,
    "": 0,
}
subjects = {r.get("sid", "") for r in read_csv(root / "subjects.csv") if r.get("sid")}
route_decisions_by_sid: dict[str, list[dict[str, str]]] = {}
for route_order, (route_id, route_root) in enumerate(zip(route_ids, route_roots), start=1):
    for row in read_csv(route_root / "route_qc_decisions.csv"):
        sid = row.get("sid") or ""
        if not sid:
            continue
        subjects.add(sid)
        row = dict(row)
        row["_route"] = route_id
        row["_route_order"] = str(route_order)
        if (route_id, sid) in visual_status_by_route_sid:
            row["visual_overlay_status"] = visual_status_by_route_sid[(route_id, sid)]
        route_decisions_by_sid.setdefault(sid, []).append(row)

live_states: dict[str, int] = {}
for sid in subjects:
    decisions_for_sid = route_decisions_by_sid.get(sid, [])
    latest_decision = (
        sorted(decisions_for_sid, key=lambda r: int(r.get("_route_order", 0)))[-1]
        if decisions_for_sid
        else {}
    )
    best_decision = (
        sorted(
            decisions_for_sid,
            key=lambda r: (decision_rank.get(r.get("qc_decision", ""), 0), int(r.get("_route_order", 0))),
        )[-1]
        if decisions_for_sid
        else {}
    )
    promoted = unique_promoted.get(sid)
    if promoted and promoted.get("promotion_class") != "interim_review_not_solved":
        state = "solved_full_qc_production"
    elif promoted and promoted.get("promotion_class") == "interim_review_not_solved":
        state = "interim_production_keep_trying"
    elif best_decision.get("qc_decision") in {
        "PROMOTE_CANDIDATE_PENDING_VISUAL",
        "INTERIM_REVIEW_PROMOTION_PENDING_VISUAL",
    } and best_decision.get("visual_overlay_status", "").startswith(("failed", "rejected")):
        state = "visual_failed_keep_trying"
    elif best_decision.get("qc_decision") == "PROMOTE_CANDIDATE_PENDING_VISUAL":
        state = "numeric_pass_pending_visual"
    elif best_decision.get("qc_decision") == "INTERIM_REVIEW_PROMOTION_PENDING_VISUAL":
        state = "interim_candidate_pending_visual"
    elif best_decision.get("qc_decision") == "KEEP_PRODUCTION_NUMERIC_OK":
        state = "production_current_numeric_ok"
    elif latest_decision.get("qc_decision") == "REVIEW_ROUTE":
        state = "review_route"
    elif latest_decision.get("qc_decision") == "TRY_ANOTHER_ROUTE":
        state = "needs_next_route"
    elif decisions_for_sid:
        state = "unknown_route_state"
    else:
        state = "not_scored_yet"
    live_states[state] = live_states.get(state, 0) + 1

full_goal_open = (
    live_states.get("interim_production_keep_trying", 0)
    + live_states.get("interim_candidate_pending_visual", 0)
    + live_states.get("numeric_pass_pending_visual", 0)
    + live_states.get("visual_failed_keep_trying", 0)
    + live_states.get("review_route", 0)
    + live_states.get("needs_next_route", 0)
    + live_states.get("not_scored_yet", 0)
    + live_states.get("unknown_route_state", 0)
)
current_ok = decision_counts.get("KEEP_PRODUCTION_NUMERIC_OK", 0)
first_route_fail = decision_counts.get("TRY_ANOTHER_ROUTE", 0)
first_route_review = decision_counts.get("REVIEW_ROUTE", 0)
first_route_interim = decision_counts.get("INTERIM_REVIEW_PROMOTION_PENDING_VISUAL", 0)
first_route_final = decision_counts.get("PROMOTE_CANDIDATE_PENDING_VISUAL", 0)

r2_done, r2_total, r2_phase = route_status(route2)
r3_done, r3_total, r3_phase = route_status(route3)
r4_done, r4_total, r4_phase = route_status(route4)
r5_done, r5_total, r5_phase = route_status(route5)
r6_done, r6_total, r6_phase = route_status(route6)
r7_done, r7_total, r7_phase = route_status(route7)
r8_done, r8_total, r8_phase = route_status(route8)
r9_done, r9_total, r9_phase = route_status(route9)
r7_active_status = read_json(route7 / "status.json").get("active") if route7.exists() else []
if r7_phase == "running" and not r7_active_status and not has_process("route7"):
    r7_phase = "queued/stale status; no live route7 process"

print("")
print("PLAIN STATUS")
print(f"AD eligible batch:        {bar(done, total)} first route complete | failed jobs={failed}")
print(f"Production pushed:        {len(unique_promoted)} unique subjects | full_QC_solved={unique_final} | temp_review_not_solved={unique_interim} | events={event_count}")
print(
    "Live subject state:      "
    f"full={live_states.get('solved_full_qc_production', 0)} | "
    f"interim_keep_trying={live_states.get('interim_production_keep_trying', 0)} | "
    f"pending_visual={live_states.get('numeric_pass_pending_visual', 0)} | "
    f"visual_failed_keep_trying={live_states.get('visual_failed_keep_trying', 0)} | "
    f"interim_candidate={live_states.get('interim_candidate_pending_visual', 0)} | "
    f"current_ok={live_states.get('production_current_numeric_ok', 0)} | "
    f"needs_next_route={live_states.get('needs_next_route', 0)} | "
    f"not_scored={live_states.get('not_scored_yet', 0)}"
)
print(f"Full-goal still open:    {full_goal_open} subject(s) still need visual QC, full QC promotion, or another route")
print(f"Already OK in production: {current_ok} subject(s), no overwrite needed")
print(f"First-route scoring:      scored={len(scored)} | full_candidate={first_route_final} | temp_review={first_route_interim} | current_ok={current_ok} | failed_next_route={first_route_fail} | review={first_route_review}")
print(f"Still not first-route done: {max(total - done, 0)} subject(s)")
print("Route progress:")
print(f"  R1 source-contract:      {done}/{total} {status.get('phase') or ''}")
print(f"  R2 BBR/reuse:            {r2_done}/{r2_total} {r2_phase}")
print(f"  R3 assignment rescue:    {r3_done}/{r3_total} {r3_phase}")
print(f"  R4 label rescue:         {r4_done}/{r4_total} {r4_phase}")
print(f"  R5 deep assignment:      {r5_done}/{r5_total} {r5_phase}")
print(f"  R6 all-vox diagnostic:   {r6_done}/{r6_total} {r6_phase}")
print(f"  R7 selective rescue:     {r7_done}/{r7_total} {r7_phase}")
print(f"  R8 full-T1 FNIRT:        {r8_done}/{r8_total} {r8_phase}")
print(f"  R9 R8 assignment retry:  {r9_done}/{r9_total} {r9_phase}")
print("")
print("DETAILS")
PY

python - "$RUN_ROOT" <<'PY'
from pathlib import Path
import csv, json, re, subprocess, sys
from datetime import datetime, timezone

def active_mrtrix_jobs() -> list[str]:
    try:
        proc = subprocess.check_output(["ps", "-eo", "pid,etimes,pcpu,cmd"], text=True)
    except Exception:
        return []
    jobs: list[str] = []
    for line in proc.splitlines():
        if "tck2connectome" not in line or "aal3_" not in line:
            continue
        parts = line.strip().split(maxsplit=3)
        if len(parts) < 4:
            continue
        _pid, etimes, pcpu, cmd = parts
        sid_match = re.search(r"/tracks/([^/]+)/tracks_final", cmd)
        sid = sid_match.group(1) if sid_match else "unknown"
        route = "route?"
        if "aal3_route2_bbr_probe" in cmd:
            route = "R2"
        elif "aal3_route7_selective_rescue_probe" in cmd:
            route = "R7"
        elif "aal3_route3_assignment_probe" in cmd:
            route = "R3/R5/R6"
        variant_match = re.search(r"__(forward\d+|radial\d+|reverse\d+|endvox|allvoxels)__fd_sum", cmd)
        variant = variant_match.group(1) if variant_match else ""
        try:
            minutes = int(etimes) / 60.0
            elapsed = f"{minutes:.0f}m" if minutes >= 10 else f"{minutes:.1f}m"
        except Exception:
            elapsed = f"{etimes}s"
        jobs.append(f"{route}:{sid} {variant} {elapsed} {pcpu}%CPU".strip())
    return jobs

def active_process_sids(fragment: str) -> set[str]:
    try:
        proc = subprocess.check_output(["pgrep", "-af", fragment], text=True)
    except Exception:
        return set()
    sids: set[str] = set()
    for line in proc.splitlines():
        sid_match = (
            re.search(r"--sid\s+(\S+)", line)
            or re.search(r"/tracks/([^/]+)/tracks_final", line)
            or re.search(r"route7_selective_track_rescue_([0-9]{3}_S_[^/\s]+_I[0-9]+)", line)
            or re.search(r"route7_selective_rescue_probe/[^/\s]*_([0-9]{3}_S_[^/\s]+_I[0-9]+)", line)
        )
        if sid_match:
            sids.add(sid_match.group(1))
    return sids

def fmt_elapsed(seconds: str) -> str:
    try:
        minutes = int(float(seconds)) / 60.0
    except Exception:
        return f"{seconds}s"
    if minutes >= 60:
        return f"{minutes / 60.0:.1f}h"
    if minutes >= 10:
        return f"{minutes:.0f}m"
    return f"{minutes:.1f}m"

def active_process_health_by_sid() -> dict[str, list[str]]:
    try:
        proc = subprocess.check_output(["ps", "-eo", "pid,etimes,pcpu,cmd"], text=True)
    except Exception:
        return {}
    tools = ("fnirt", "flirt", "applywarp", "invwarp", "tckgen", "tcksift2", "tck2connectome")
    out: dict[str, list[str]] = {}
    for line in proc.splitlines():
        if not any(tool in line for tool in tools):
            continue
        parts = line.strip().split(maxsplit=3)
        if len(parts) < 4:
            continue
        _pid, etimes, pcpu, cmd = parts
        sid_matches = [
            m.group(1)
            for m in (
                re.search(r"/tracks/([^/]+)/tracks_final", cmd),
                re.search(r"/fod/([^/]+)/", cmd),
                re.search(r"route7_selective_track_rescue_([0-9]{3}_S_[^/\s]+_I[0-9]+)", cmd),
                re.search(r"route8_full_t1_fnirt_([0-9]{3}_S_[^/\s]+_I[0-9]+)", cmd),
                re.search(r"route9_route8_assignment_retry_([0-9]{3}_S_[^/\s]+_I[0-9]+)", cmd),
            )
            if m
        ]
        if not sid_matches:
            continue
        tool = Path(cmd.split()[0]).name
        summary = f"{tool} {fmt_elapsed(etimes)} {pcpu}%CPU"
        for sid in sorted(set(sid_matches)):
            out.setdefault(sid, []).append(summary)
    return out

root = Path(sys.argv[1])
status_path = root / "status.json"
status = json.loads(status_path.read_text()) if status_path.exists() else {}
manifest_path = root / "run_manifest.json"
manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
process_health = active_process_health_by_sid()
total = int(status.get("total") or 0)
done = int(status.get("completed") or 0)
failed = int(status.get("failed") or 0)
phase = status.get("phase", "")
latest = status.get("latest") or {}
width = 34
filled = int(width * done / total) if total else 0
bar = "#" * filled + "." * (width - filled)
print(f"phase={phase} | completed={done}/{total} | failed={failed}")
print(f"progress [{bar}] {done}/{total}")
started = manifest.get("started_utc")
if started and done:
    try:
        start_dt = datetime.fromisoformat(started.replace("Z", "+00:00"))
        hours = max((datetime.now(timezone.utc) - start_dt).total_seconds() / 3600.0, 1e-6)
        rate = done / hours
        remaining = max(total - done, 0)
        eta = remaining / rate if rate > 0 else 0
        print(f"rate={rate:.1f}/hr | ETA={eta:.1f} hr")
    except Exception:
        pass
try:
    proc = subprocess.check_output(["pgrep", "-af", "run_sc_aal3_source_contract_probe.py"], text=True)
    active_sids = []
    for line in proc.splitlines():
        m = re.search(r"--sid\s+(\S+)", line)
        if m:
            active_sids.append(m.group(1))
    if active_sids:
        print("active probes: " + ", ".join(active_sids[:8]) + (" ..." if len(active_sids) > 8 else ""))
except Exception:
    pass
if latest:
    print(
        "latest: "
        f"{latest.get('sid','')} | status={latest.get('status','')} | "
        f"variant={latest.get('best_variant','')} | "
        f"density={latest.get('baseline_density','')}->{latest.get('best_density','')} | "
        f"zero={latest.get('baseline_zero_rows','')}->{latest.get('best_zero_rows','')}"
    )

results_path = root / "batch_results.csv"
if results_path.exists() and results_path.stat().st_size:
    rows = list(csv.DictReader(results_path.open()))
    ok = [r for r in rows if r.get("status") == "ok"]
    improved = [
        r for r in ok
        if float(r.get("density_delta") or 0) > 0 and float(r.get("zero_rows_delta") or 9999) <= 0
    ]
    variants = {}
    for r in ok:
        variants[r.get("best_variant") or ""] = variants.get(r.get("best_variant") or "", 0) + 1
    print(f"ok={len(ok)} | density+zero improved={len(improved)} | best variants={variants}")
    if rows:
        r = rows[-1]
        print(
            f"last completed: {r.get('sid','')} {r.get('status','')} "
            f"{r.get('best_variant','')} | "
            f"dens {r.get('baseline_density','')}->{r.get('best_density','')} | "
            f"zero {r.get('baseline_zero_rows','')}->{r.get('best_zero_rows','')}"
        )
qc_summary = root / "route_qc_summary.json"
if qc_summary.exists() and qc_summary.stat().st_size:
    try:
        qc = json.loads(qc_summary.read_text())
        decisions = qc.get("decisions") or {}
        print(
            "QC decisions: "
            f"promote_pending_visual={decisions.get('PROMOTE_CANDIDATE_PENDING_VISUAL', 0)} | "
            f"interim_review_pending_visual={decisions.get('INTERIM_REVIEW_PROMOTION_PENDING_VISUAL', 0)} | "
            f"review={decisions.get('REVIEW_ROUTE', 0)} | "
            f"try_another_route={decisions.get('TRY_ANOTHER_ROUTE', 0)}"
        )
    except Exception:
        pass
promotion_manifest = root / "production_promotion_manifest.csv"
if promotion_manifest.exists() and promotion_manifest.stat().st_size:
    try:
        promoted_rows = [r for r in csv.DictReader(promotion_manifest.open()) if r.get("status") == "promoted"]
        final_rows = [r for r in promoted_rows if r.get("promotion_class") in {"", "final_qc_pass"}]
        interim_rows = [r for r in promoted_rows if r.get("promotion_class") == "interim_review_not_solved"]
        print(f"route1 production promoted: {len(promoted_rows)} | final={len(final_rows)} | interim_review={len(interim_rows)}")
    except Exception:
        pass

def stage_bar(label, count, denom, width=24):
    count = int(count or 0)
    denom = int(denom or 0)
    fill = int(width * count / denom) if denom else 0
    print(f"{label:<30} [{'#' * fill}{'.' * (width - fill)}] {count}/{denom}")

print("")
print("stage flow")
stage_bar("route 1 completed", done, total)
scored_rows = []
if (root / "route_qc_decisions.csv").exists() and (root / "route_qc_decisions.csv").stat().st_size:
    scored_rows = list(csv.DictReader((root / "route_qc_decisions.csv").open()))
stage_bar("route 1 scored", len(scored_rows), total)
if scored_rows:
    decisions = {}
    for r in scored_rows:
        decisions[r.get("qc_decision", "")] = decisions.get(r.get("qc_decision", ""), 0) + 1
    final_n = decisions.get("PROMOTE_CANDIDATE_PENDING_VISUAL", 0)
    interim_n = decisions.get("INTERIM_REVIEW_PROMOTION_PENDING_VISUAL", 0)
    current_ok_n = decisions.get("KEEP_PRODUCTION_NUMERIC_OK", 0)
    try_n = decisions.get("TRY_ANOTHER_ROUTE", 0)
    review_n = decisions.get("REVIEW_ROUTE", 0)
    print(f"  numeric buckets: final={final_n} | interim_review={interim_n} | current_ok={current_ok_n} | review={review_n} | route_fail={try_n}")
visual_rows = []
if (root / "visual_overlay_qc_manifest.csv").exists() and (root / "visual_overlay_qc_manifest.csv").stat().st_size:
    visual_rows = list(csv.DictReader((root / "visual_overlay_qc_manifest.csv").open()))
visual_pass = [r for r in visual_rows if "passed" in r.get("visual_overlay_status", "")]
stage_bar("visual QC passed", len(visual_pass), max(1, len(visual_rows)) if visual_rows else 0)
promoted_rows = []
if promotion_manifest.exists() and promotion_manifest.stat().st_size:
    promoted_rows = [r for r in csv.DictReader(promotion_manifest.open()) if r.get("status") == "promoted"]
stage_bar("route1 production promoted", len(promoted_rows), total)
final_promoted = [r for r in promoted_rows if r.get("promotion_class") in {"", "final_qc_pass"}]
interim_promoted = [r for r in promoted_rows if r.get("promotion_class") == "interim_review_not_solved"]
if promoted_rows:
    print(f"  promoted buckets: final_ok={len(final_promoted)} | interim_review={len(interim_promoted)}")
route2_root = root.parent / (root.name + "_route2")
route3_root = root.parent / (root.name + "_route3_assignment")
route4_root = root.parent / (root.name + "_route4_label_rescue")
route5_root = root.parent / (root.name + "_route5_deep_assignment")
route6_root = root.parent / (root.name + "_route6_all_voxels_diagnostic")
route7_root = root.parent / (root.name + "_route7_selective_track_rescue")
route8_root = root.parent / (root.name + "_route8_full_t1_fnirt")
route9_root = root.parent / (root.name + "_route9_route8_assignment_retry")
if route2_root.exists():
    route2_status = json.loads((route2_root / "status.json").read_text()) if (route2_root / "status.json").exists() else {}
    stage_bar("route 2 completed", route2_status.get("completed", 0), route2_status.get("total", 0))
else:
    route2_queue = 0
    if scored_rows:
        route2_queue = sum(1 for r in scored_rows if r.get("qc_decision") in {"TRY_ANOTHER_ROUTE", "REVIEW_ROUTE"})
    print(f"route 2 alternative search     not started | queued_candidates={route2_queue}")

print("")
print("route plan and queues")
route1_final = 0
route1_interim = 0
route1_current_ok = 0
route1_review = 0
route1_fail = 0
route1_unscored = max(total - len(scored_rows), 0) if total else 0
if scored_rows:
    route1_final = sum(1 for r in scored_rows if r.get("qc_decision") == "PROMOTE_CANDIDATE_PENDING_VISUAL")
    route1_interim = sum(1 for r in scored_rows if r.get("qc_decision") == "INTERIM_REVIEW_PROMOTION_PENDING_VISUAL")
    route1_current_ok = sum(1 for r in scored_rows if r.get("qc_decision") == "KEEP_PRODUCTION_NUMERIC_OK")
    route1_review = sum(1 for r in scored_rows if r.get("qc_decision") == "REVIEW_ROUTE")
    route1_fail = sum(1 for r in scored_rows if r.get("qc_decision") == "TRY_ANOTHER_ROUTE")
route1_promoted = len(promoted_rows)
route1_unsolved_queue = route1_interim + route1_review + route1_fail
route2_completed = 0
route2_total = 0
route2_final = 0
route2_interim = 0
route2_fail = 0
route2_promoted = 0
route2_promoted_final = 0
route2_promoted_interim = 0
route2_active_lines = []
route3_completed = 0
route3_total = 0
route3_final = 0
route3_interim = 0
route3_fail = 0
route3_promoted = 0
route3_promoted_final = 0
route3_promoted_interim = 0
route3_active_lines = []
route4_completed = 0
route4_failed_jobs = 0
route4_phase = "not started"
route4_total = 0
route4_final = 0
route4_interim = 0
route4_fail = 0
route4_promoted = 0
route4_promoted_final = 0
route4_promoted_interim = 0
route4_active_lines = []
route5_completed = 0
route5_total = 0
route5_final = 0
route5_interim = 0
route5_fail = 0
route5_promoted = 0
route5_promoted_final = 0
route5_promoted_interim = 0
route5_active_lines = []
route6_completed = 0
route6_total = 0
route6_final = 0
route6_interim = 0
route6_fail = 0
route6_promoted = 0
route6_promoted_final = 0
route6_promoted_interim = 0
route6_active_lines = []
route6_diagnostic_buckets = {}
route7_completed = 0
route7_total = 0
route7_final = 0
route7_interim = 0
route7_fail = 0
route7_promoted = 0
route7_promoted_final = 0
route7_promoted_interim = 0
route7_active_lines = []
actual_route7_sids = active_process_sids("route7")
route7_live = bool(actual_route7_sids)
route8_completed = 0
route8_total = 0
route8_final = 0
route8_interim = 0
route8_fail = 0
route8_promoted = 0
route8_promoted_final = 0
route8_promoted_interim = 0
route8_active_lines = []
route9_completed = 0
route9_total = 0
route9_final = 0
route9_interim = 0
route9_fail = 0
route9_promoted = 0
route9_promoted_final = 0
route9_promoted_interim = 0
route9_active_lines = []
if route2_root.exists():
    route2_status_path = route2_root / "status.json"
    if route2_status_path.exists():
        try:
            route2_status = json.loads(route2_status_path.read_text())
            route2_completed = int(route2_status.get("completed") or 0)
            route2_total = int(route2_status.get("total") or 0)
            for sid in route2_status.get("active") or []:
                probe_status = root.parents[3] / "sc_matrix_qc" / "aal3_route2_bbr_probe" / f"{route2_root.name}_{sid}" / "status.json"
                # root.parents[3] is usually /data/derivatives/qc; if the
                # probe lives under the repo mirror instead, fall back below.
                alt_probe_status = Path("/home/ec2-user/exp/data/derivatives/qc/sc_matrix_qc/aal3_route2_bbr_probe") / f"{route2_root.name}_{sid}" / "status.json"
                probe = {}
                for status_candidate in (probe_status, alt_probe_status):
                    if status_candidate.exists():
                        try:
                            probe = json.loads(status_candidate.read_text())
                            break
                        except Exception:
                            pass
                if probe:
                    active_route = probe.get("active_route") or probe.get("phase") or ""
                    active_variant = probe.get("active_variant") or ""
                    jobs_done = probe.get("connectome_jobs_done")
                    jobs_total = probe.get("connectome_jobs_total")
                    jobs = f" {jobs_done}/{jobs_total}" if jobs_total not in {None, "", 0} else ""
                    route2_active_lines.append(f"{sid}: {active_route} {active_variant}{jobs}".strip())
                else:
                    route2_active_lines.append(str(sid))
        except Exception:
            pass
if route3_root.exists():
    route3_status_path = route3_root / "status.json"
    if route3_status_path.exists():
        try:
            route3_status = json.loads(route3_status_path.read_text())
            route3_completed = int(route3_status.get("completed") or 0)
            route3_total = int(route3_status.get("total") or 0)
            for sid in route3_status.get("active") or []:
                probe_status = Path("/home/ec2-user/exp/data/derivatives/qc/sc_matrix_qc/aal3_route3_assignment_probe") / f"{route3_root.name}_{sid}" / "status.json"
                probe = {}
                if probe_status.exists():
                    try:
                        probe = json.loads(probe_status.read_text())
                    except Exception:
                        probe = {}
                if probe:
                    active_route = probe.get("active_parcellation") or probe.get("phase") or ""
                    active_variant = probe.get("active_variant") or ""
                    jobs_done = probe.get("connectome_jobs_done")
                    jobs_total = probe.get("connectome_jobs_total")
                    jobs = f" {jobs_done}/{jobs_total}" if jobs_total not in {None, "", 0} else ""
                    route3_active_lines.append(f"{sid}: {active_route} {active_variant}{jobs}".strip())
                else:
                    route3_active_lines.append(str(sid))
        except Exception:
            pass
    route3_decisions = route3_root / "route_qc_decisions.csv"
    if route3_decisions.exists() and route3_decisions.stat().st_size:
        try:
            route3_rows = list(csv.DictReader(route3_decisions.open()))
            route3_final = sum(1 for r in route3_rows if r.get("qc_decision") == "PROMOTE_CANDIDATE_PENDING_VISUAL")
            route3_interim = sum(1 for r in route3_rows if r.get("qc_decision") == "INTERIM_REVIEW_PROMOTION_PENDING_VISUAL")
            route3_fail = sum(1 for r in route3_rows if r.get("qc_decision") == "TRY_ANOTHER_ROUTE")
        except Exception:
            pass
if route4_root.exists():
    route4_status_path = route4_root / "status.json"
    if route4_status_path.exists():
        try:
            route4_status = json.loads(route4_status_path.read_text())
            route4_completed = int(route4_status.get("completed") or 0)
            route4_failed_jobs = int(route4_status.get("failed") or 0)
            route4_phase = str(route4_status.get("phase") or "not started")
            route4_total = int(route4_status.get("total") or 0)
            for sid in route4_status.get("active") or []:
                probe_status = Path("/home/ec2-user/exp/data/derivatives/qc/sc_matrix_qc/aal3_route4_label_rescue_probe") / f"{route4_root.name}_{sid}" / "status.json"
                probe = {}
                if probe_status.exists():
                    try:
                        probe = json.loads(probe_status.read_text())
                    except Exception:
                        probe = {}
                if probe:
                    active_route = probe.get("active_parcellation") or probe.get("phase") or ""
                    active_variant = probe.get("active_variant") or ""
                    jobs_done = probe.get("connectome_jobs_done")
                    jobs_total = probe.get("connectome_jobs_total")
                    jobs = f" {jobs_done}/{jobs_total}" if jobs_total not in {None, "", 0} else ""
                    route4_active_lines.append(f"{sid}: {active_route} {active_variant}{jobs}".strip())
                else:
                    route4_active_lines.append(str(sid))
        except Exception:
            pass
    route4_decisions = route4_root / "route_qc_decisions.csv"
    if route4_decisions.exists() and route4_decisions.stat().st_size:
        try:
            route4_rows = list(csv.DictReader(route4_decisions.open()))
            route4_final = sum(1 for r in route4_rows if r.get("qc_decision") == "PROMOTE_CANDIDATE_PENDING_VISUAL")
            route4_interim = sum(1 for r in route4_rows if r.get("qc_decision") == "INTERIM_REVIEW_PROMOTION_PENDING_VISUAL")
            route4_fail = sum(1 for r in route4_rows if r.get("qc_decision") == "TRY_ANOTHER_ROUTE")
        except Exception:
            pass
route2_decisions = route2_root / "route_qc_decisions.csv"
if route2_decisions.exists() and route2_decisions.stat().st_size:
    try:
        route2_rows = list(csv.DictReader(route2_decisions.open()))
        route2_final = sum(1 for r in route2_rows if r.get("qc_decision") == "PROMOTE_CANDIDATE_PENDING_VISUAL")
        route2_interim = sum(1 for r in route2_rows if r.get("qc_decision") == "INTERIM_REVIEW_PROMOTION_PENDING_VISUAL")
        route2_fail = sum(1 for r in route2_rows if r.get("qc_decision") == "TRY_ANOTHER_ROUTE")
    except Exception:
        pass
if route5_root.exists():
    route5_status_path = route5_root / "status.json"
    if route5_status_path.exists():
        try:
            route5_status = json.loads(route5_status_path.read_text())
            route5_completed = int(route5_status.get("completed") or 0)
            route5_total = int(route5_status.get("total") or 0)
            for sid in route5_status.get("active") or []:
                probe_status = Path("/home/ec2-user/exp/data/derivatives/qc/sc_matrix_qc/aal3_route3_assignment_probe") / f"{route5_root.name}_{sid}" / "status.json"
                probe = {}
                if probe_status.exists():
                    try:
                        probe = json.loads(probe_status.read_text())
                    except Exception:
                        probe = {}
                if probe:
                    active_route = probe.get("active_parcellation") or probe.get("phase") or ""
                    active_variant = probe.get("active_variant") or ""
                    jobs_done = probe.get("connectome_jobs_done")
                    jobs_total = probe.get("connectome_jobs_total")
                    jobs = f" {jobs_done}/{jobs_total}" if jobs_total not in {None, "", 0} else ""
                    route5_active_lines.append(f"{sid}: {active_route} {active_variant}{jobs}".strip())
                else:
                    route5_active_lines.append(str(sid))
        except Exception:
            pass
    route5_decisions = route5_root / "route_qc_decisions.csv"
    if route5_decisions.exists() and route5_decisions.stat().st_size:
        try:
            route5_rows = list(csv.DictReader(route5_decisions.open()))
            route5_final = sum(1 for r in route5_rows if r.get("qc_decision") == "PROMOTE_CANDIDATE_PENDING_VISUAL")
            route5_interim = sum(1 for r in route5_rows if r.get("qc_decision") == "INTERIM_REVIEW_PROMOTION_PENDING_VISUAL")
            route5_fail = sum(1 for r in route5_rows if r.get("qc_decision") == "TRY_ANOTHER_ROUTE")
        except Exception:
            pass
if route6_root.exists():
    route6_status_path = route6_root / "status.json"
    if route6_status_path.exists():
        try:
            route6_status = json.loads(route6_status_path.read_text())
            route6_completed = int(route6_status.get("completed") or 0)
            route6_total = int(route6_status.get("total") or 0)
            for sid in route6_status.get("active") or []:
                route6_active_lines.append(str(sid))
        except Exception:
            pass
    route6_decisions = route6_root / "route_qc_decisions.csv"
    if route6_decisions.exists() and route6_decisions.stat().st_size:
        try:
            route6_rows = list(csv.DictReader(route6_decisions.open()))
            route6_final = sum(1 for r in route6_rows if r.get("qc_decision") == "PROMOTE_CANDIDATE_PENDING_VISUAL")
            route6_interim = sum(1 for r in route6_rows if r.get("qc_decision") == "INTERIM_REVIEW_PROMOTION_PENDING_VISUAL")
            route6_fail = sum(1 for r in route6_rows if r.get("qc_decision") == "TRY_ANOTHER_ROUTE")
            for r in route6_rows:
                bucket = r.get("diagnostic_bucket", "")
                if bucket:
                    route6_diagnostic_buckets[bucket] = route6_diagnostic_buckets.get(bucket, 0) + 1
        except Exception:
            pass
if route7_root.exists():
    route7_status_path = route7_root / "status.json"
    if route7_status_path.exists():
        try:
            route7_status = json.loads(route7_status_path.read_text())
            route7_completed = int(route7_status.get("completed") or 0)
            route7_total = int(route7_status.get("total") or 0)
            if route7_status.get("phase") == "running" and route7_status.get("active"):
                route7_live = True
            for active_entry in route7_status.get("active") or []:
                active_entry = str(active_entry)
                sid = active_entry.split(":", 1)[0]
                if actual_route7_sids and sid not in actual_route7_sids:
                    continue
                action_hint = active_entry.split(":", 1)[1] if ":" in active_entry else ""
                probe_status = Path("/home/ec2-user/exp/data/derivatives/qc/sc_matrix_qc/aal3_route7_selective_rescue_probe") / f"{route7_root.name}_{sid}" / "status.json"
                probe = {}
                if probe_status.exists():
                    try:
                        probe = json.loads(probe_status.read_text())
                    except Exception:
                        probe = {}
                if probe:
                    action = probe.get("route7_action") or action_hint or probe.get("phase") or ""
                    phase_label = probe.get("phase") or ""
                    active_variant = probe.get("active_variant") or ""
                    active_track_mode = probe.get("active_track_mode") or ""
                    jobs_done = probe.get("connectome_jobs_done")
                    jobs_total = probe.get("connectome_jobs_total")
                    jobs = f" {jobs_done}/{jobs_total}" if jobs_total not in {None, "", 0} else ""
                    health = "; ".join(process_health.get(sid, [])[:2])
                    health_text = f" [{health}]" if health else ""
                    route7_active_lines.append(
                        f"{sid}: {action} {phase_label} {active_track_mode} {active_variant}{jobs}{health_text}".strip()
                    )
                else:
                    route7_active_lines.append(active_entry)
        except Exception:
            pass
    route7_decisions = route7_root / "route_qc_decisions.csv"
    if route7_decisions.exists() and route7_decisions.stat().st_size:
        try:
            route7_rows = list(csv.DictReader(route7_decisions.open()))
            route7_final = sum(1 for r in route7_rows if r.get("qc_decision") == "PROMOTE_CANDIDATE_PENDING_VISUAL")
            route7_interim = sum(1 for r in route7_rows if r.get("qc_decision") == "INTERIM_REVIEW_PROMOTION_PENDING_VISUAL")
            route7_fail = sum(1 for r in route7_rows if r.get("qc_decision") == "TRY_ANOTHER_ROUTE")
        except Exception:
            pass
if route8_root.exists():
    route8_status_path = route8_root / "status.json"
    if route8_status_path.exists():
        try:
            route8_status = json.loads(route8_status_path.read_text())
            route8_completed = int(route8_status.get("completed") or 0)
            route8_total = int(route8_status.get("total") or 0)
            for sid in route8_status.get("active") or []:
                probe_status = Path("/home/ec2-user/exp/data/derivatives/qc/sc_matrix_qc/aal3_route8_full_t1_fnirt_probe") / f"{route8_root.name}_{sid}" / "status.json"
                probe = {}
                if probe_status.exists():
                    try:
                        probe = json.loads(probe_status.read_text())
                    except Exception:
                        probe = {}
                if probe:
                    phase_label = probe.get("phase") or ""
                    active_variant = probe.get("active_variant") or ""
                    jobs_done = probe.get("connectome_jobs_done")
                    jobs_total = probe.get("connectome_jobs_total")
                    jobs = f" {jobs_done}/{jobs_total}" if jobs_total not in {None, "", 0} else ""
                    health = "; ".join(process_health.get(str(sid), [])[:2])
                    health_text = f" [{health}]" if health else ""
                    route8_active_lines.append(f"{sid}: {phase_label} {active_variant}{jobs}{health_text}".strip())
                else:
                    route8_active_lines.append(str(sid))
        except Exception:
            pass
    route8_decisions = route8_root / "route_qc_decisions.csv"
    if route8_decisions.exists() and route8_decisions.stat().st_size:
        try:
            route8_rows = list(csv.DictReader(route8_decisions.open()))
            route8_final = sum(1 for r in route8_rows if r.get("qc_decision") == "PROMOTE_CANDIDATE_PENDING_VISUAL")
            route8_interim = sum(1 for r in route8_rows if r.get("qc_decision") == "INTERIM_REVIEW_PROMOTION_PENDING_VISUAL")
            route8_fail = sum(1 for r in route8_rows if r.get("qc_decision") == "TRY_ANOTHER_ROUTE")
        except Exception:
            pass
if route9_root.exists():
    route9_status_path = route9_root / "status.json"
    if route9_status_path.exists():
        try:
            route9_status = json.loads(route9_status_path.read_text())
            route9_completed = int(route9_status.get("completed") or 0)
            route9_total = int(route9_status.get("total") or 0)
            for sid in route9_status.get("active") or []:
                probe_status = Path("/home/ec2-user/exp/data/derivatives/qc/sc_matrix_qc/aal3_route9_route8_assignment_retry_probe") / f"{route9_root.name}_{sid}" / "status.json"
                probe = {}
                if probe_status.exists():
                    try:
                        probe = json.loads(probe_status.read_text())
                    except Exception:
                        probe = {}
                if probe:
                    phase_label = probe.get("phase") or ""
                    active_variant = probe.get("active_variant") or ""
                    active_parc = probe.get("active_parcellation") or ""
                    jobs_done = probe.get("connectome_jobs_done")
                    jobs_total = probe.get("connectome_jobs_total")
                    jobs = f" {jobs_done}/{jobs_total}" if jobs_total not in {None, "", 0} else ""
                    health = "; ".join(process_health.get(str(sid), [])[:2])
                    health_text = f" [{health}]" if health else ""
                    route9_active_lines.append(f"{sid}: {phase_label} {active_parc} {active_variant}{jobs}{health_text}".strip())
                else:
                    route9_active_lines.append(str(sid))
        except Exception:
            pass
    route9_decisions = route9_root / "route_qc_decisions.csv"
    if route9_decisions.exists() and route9_decisions.stat().st_size:
        try:
            route9_rows = list(csv.DictReader(route9_decisions.open()))
            route9_final = sum(1 for r in route9_rows if r.get("qc_decision") == "PROMOTE_CANDIDATE_PENDING_VISUAL")
            route9_interim = sum(1 for r in route9_rows if r.get("qc_decision") == "INTERIM_REVIEW_PROMOTION_PENDING_VISUAL")
            route9_fail = sum(1 for r in route9_rows if r.get("qc_decision") == "TRY_ANOTHER_ROUTE")
        except Exception:
            pass

def promotion_counts(route_root: Path) -> tuple[int, int, int]:
    manifest = route_root / "production_promotion_manifest.csv"
    if not manifest.exists() or manifest.stat().st_size <= 0:
        return 0, 0, 0
    try:
        rows = [r for r in csv.DictReader(manifest.open()) if r.get("status") == "promoted"]
    except Exception:
        return 0, 0, 0
    final = sum(1 for r in rows if r.get("promotion_class") in {"", "final_qc_pass"})
    interim = sum(1 for r in rows if r.get("promotion_class") == "interim_review_not_solved")
    return len(rows), final, interim

def promotion_manifest_rows(route_root: Path) -> list[dict[str, str]]:
    manifest = route_root / "production_promotion_manifest.csv"
    if not manifest.exists() or manifest.stat().st_size <= 0:
        return []
    try:
        rows = [dict(r) for r in csv.DictReader(manifest.open()) if r.get("status") == "promoted"]
    except Exception:
        return []
    for row in rows:
        row["_route_root"] = route_root.name
    return rows

route2_promoted, route2_promoted_final, route2_promoted_interim = promotion_counts(route2_root)
route3_promoted, route3_promoted_final, route3_promoted_interim = promotion_counts(route3_root)
route4_promoted, route4_promoted_final, route4_promoted_interim = promotion_counts(route4_root)
route5_promoted, route5_promoted_final, route5_promoted_interim = promotion_counts(route5_root)
route6_promoted, route6_promoted_final, route6_promoted_interim = promotion_counts(route6_root)
route7_promoted, route7_promoted_final, route7_promoted_interim = promotion_counts(route7_root)
route8_promoted, route8_promoted_final, route8_promoted_interim = promotion_counts(route8_root)
route9_promoted, route9_promoted_final, route9_promoted_interim = promotion_counts(route9_root)
total_promoted_all_routes = route1_promoted + route2_promoted + route3_promoted + route4_promoted + route5_promoted + route6_promoted + route7_promoted + route8_promoted + route9_promoted
total_promoted_final = len(final_promoted) + route2_promoted_final + route3_promoted_final + route4_promoted_final + route5_promoted_final + route6_promoted_final + route7_promoted_final + route8_promoted_final + route9_promoted_final
total_promoted_interim = len(interim_promoted) + route2_promoted_interim + route3_promoted_interim + route4_promoted_interim + route5_promoted_interim + route6_promoted_interim + route7_promoted_interim + route8_promoted_interim + route9_promoted_interim
all_promotion_rows: list[dict[str, str]] = []
for promotion_root in (root, route2_root, route3_root, route4_root, route5_root, route6_root, route7_root, route8_root, route9_root):
    all_promotion_rows.extend(promotion_manifest_rows(promotion_root))
promotion_priority = {"final_qc_pass": 3, "": 3, "interim_review_not_solved": 2}
unique_promoted: dict[str, dict[str, str]] = {}
for row in all_promotion_rows:
    sid = row.get("sid") or ""
    if not sid:
        continue
    current = unique_promoted.get(sid)
    row_priority = promotion_priority.get(row.get("promotion_class", ""), 1)
    current_priority = promotion_priority.get(current.get("promotion_class", ""), 1) if current else -1
    if current is None or row_priority > current_priority:
        unique_promoted[sid] = row
unique_promoted_final = sum(1 for row in unique_promoted.values() if row.get("promotion_class") in {"", "final_qc_pass"})
unique_promoted_interim = sum(1 for row in unique_promoted.values() if row.get("promotion_class") == "interim_review_not_solved")

route_plan = [
    {
        "id": "R1",
        "name": "source-contract corrected-T1 to B0",
        "what": "AAL3 MNI -> MNI brain -> t1_from_dicom/native T1 -> eddy B0, with corrected_T1 used for T1-to-B0 registration",
        "short": "AAL3 MNI -> corrected/native T1 -> eddy B0",
        "tests": [
            "parcellation: AAL3/MNI -> MNI brain -> t1_from_dicom/native T1 -> eddy volume-0 B0",
            "assignment variants: radial4, forward40, forward80",
            "output: scratch fd_sum matrices first, then visual QC before production copy",
        ],
        "status": "running" if phase == "running" else (phase or "unknown"),
        "done": done,
        "total": total,
        "final": route1_final,
        "interim": route1_interim,
        "review": route1_review,
        "fail": route1_fail,
        "current_ok": route1_current_ok,
        "promoted": route1_promoted,
        "promoted_final": len(final_promoted),
        "promoted_interim": len(interim_promoted),
        "queue_next": route1_unsolved_queue,
    },
    {
        "id": "R2",
        "name": "BBR native/direct/reuse route",
        "what": "AAL3 MNI -> BBR T1/T1-brain/direct B0, plus existing AAL_t1 reuse; same 3M tracks",
        "short": "BBR T1/T1-brain/direct B0 + AAL_t1 reuse",
        "tests": [
            "route2_bbr_t1brain_twostep: AAL3/MNI -> skull-stripped T1 -> B0",
            "route2_bbr_t1_twostep: AAL3/MNI -> full native T1 -> B0",
            "route2_bbr_direct_composed: composed MNI -> B0 transform",
            "route2_reuse_existing_aal_t1: existing AAL_t1 -> B0 with BBR transform",
            "assignment variants: forward40, forward80; skip connectome if label survival <160",
        ],
        "status": "running" if route2_root.exists() and route2_completed < route2_total else ("not started" if not route2_root.exists() else "scoring/promote"),
        "done": route2_completed,
        "total": route2_total or route1_unsolved_queue,
        "final": route2_final,
        "interim": route2_interim,
        "review": 0,
        "fail": route2_fail,
        "current_ok": 0,
        "promoted": route2_promoted,
        "promoted_final": route2_promoted_final,
        "promoted_interim": route2_promoted_interim,
        "queue_next": route2_fail,
    },
    {
        "id": "R3",
        "name": "assignment-policy rescue",
        "what": "reuse best AAL3-B0 candidate, then test reverse/radial/long-forward streamline assignment",
        "short": "reverse/radial/long-forward assignment on best AAL3-B0",
        "tests": [
            "candidate parcellations: production AAL_b0, R1 source-contract, and surviving R2 parcellations; skip connectome if label survival <160",
            "assignment variants: reverse0, reverse80, radial8, forward120",
            "purpose: separate assignment-policy failures from registration/label-survival failures",
        ],
        "status": "running" if route3_root.exists() and route3_completed < route3_total else ("not started" if not route3_root.exists() else "scoring/promote"),
        "done": route3_completed,
        "total": route3_total or max((route2_total or route1_unsolved_queue) - route2_final - route2_interim, 0),
        "final": route3_final,
        "interim": route3_interim,
        "review": 0,
        "fail": route3_fail,
        "current_ok": 0,
        "promoted": route3_promoted,
        "promoted_final": route3_promoted_final,
        "promoted_interim": route3_promoted_interim,
        "queue_next": route3_fail,
    },
    {
        "id": "R4",
        "name": "label-preserving rescue",
        "what": "nearest-neighbour resampling plus controlled label cleanup/dilation before tck2connectome",
        "short": "label cleanup/dilation before tck2connectome",
        "tests": [
            "base: best high-label AAL3-B0 from R3 or production when no route candidate is usable",
            "label rescue: controlled nearest-label fill at 2mm, 4mm, 6mm inside B0 brain mask when available",
            "assignment variants: forward80, forward120, radial8",
            "purpose: rescue tiny label gaps without changing atlas identity or rerunning tractography",
        ],
        "status": (
            "running"
            if route4_root.exists() and route4_phase == "running"
            else ("not started" if not route4_root.exists() else "scoring/promote")
        ),
        "done": (
            route4_total
            if route4_phase == "complete"
            and route4_total
            and route4_completed < route4_total
            and route4_completed + route4_failed_jobs >= route4_total
            else route4_completed
        ),
        "total": route4_total,
        "final": route4_final,
        "interim": route4_interim,
        "review": 0,
        "fail": route4_fail,
        "current_ok": 0,
        "promoted": route4_promoted,
        "promoted_final": route4_promoted_final,
        "promoted_interim": route4_promoted_interim,
        "queue_next": route4_fail,
    },
    {
        "id": "R5",
        "name": "deep endpoint-assignment fallback",
        "what": "endpoint-compatible fallback for persistent failures before manual registration/track review",
        "short": "endvox/radial16/radial24/forward200, then manual review if still sparse",
        "tests": [
            "assignment variants: endvox, radial16, radial24, forward200",
            "final QC still requires visual overlay; interim density improvements may be copied to production but stay open",
            "if still failing: debug transform chain/cost-function and only then consider rerunning tracks",
            "paper fallback: document unresolved subject rather than blindly accepting sparse AAL3",
        ],
        "status": "running" if route5_root.exists() and route5_completed < route5_total else ("not started" if not route5_root.exists() else "scoring/promote"),
        "done": route5_completed,
        "total": route5_total or route4_fail,
        "final": route5_final,
        "interim": route5_interim,
        "review": 0,
        "fail": route5_fail,
        "current_ok": 0,
        "promoted": route5_promoted,
        "promoted_final": route5_promoted_final,
        "promoted_interim": route5_promoted_interim,
        "queue_next": route5_fail,
    },
    {
        "id": "R6",
        "name": "all-voxels traversal diagnostic",
        "what": "diagnostic-only test: do existing streamlines pass through missing ROIs even when endpoints do not land there?",
        "short": "all_voxels diagnostic only; never auto-promote",
        "tests": [
            "assignment variant: -assignment_all_voxels on best surviving AAL3-B0 parcellation",
            "if all_voxels fills zero-row ROIs: endpoint assignment/termination is the bottleneck",
            "if all_voxels stays sparse: existing 3M streamlines do not cover those AAL3 ROIs",
            "production rule: diagnostic output is not promoted as the endpoint connectome matrix",
        ],
        "status": "running" if route6_root.exists() and route6_completed < route6_total else ("not started" if not route6_root.exists() else "diagnostic complete"),
        "done": route6_completed,
        "total": route6_total or route5_fail,
        "final": route6_final,
        "interim": route6_interim,
        "review": 0,
        "fail": route6_fail,
        "current_ok": 0,
        "promoted": route6_promoted,
        "promoted_final": route6_promoted_final,
        "promoted_interim": route6_promoted_interim,
        "queue_next": route6_fail or route5_fail,
    },
    {
        "id": "R7",
        "name": "selective registration/track rescue",
        "what": "last resort for unresolved subjects: manual overlay/transform review, then selective rerun of tractography if labels are good but tracks do not cover ROIs",
        "short": "manual overlay + selective tractography rerun only for true track-coverage failures",
        "tests": [
            "manual check: AAL3-B0 overlay against B0/T1 for every unresolved subject",
            "registration rescue: re-estimate cost-function/transform only if overlay is visibly wrong",
            "tractography rescue: rerun tracks only if overlay/labels pass but endpoint/all_voxels evidence says existing tracks lack coverage",
            "promotion rule: final QC needs numeric+visual pass; interim density-improved outputs can be pushed but remain open",
        ],
        "status": (
            "running"
            if route7_live
            else (
                "queued/stale status; no live route7 process"
                if route7_root.exists() and route7_completed < route7_total
                else ("not started" if not route7_root.exists() else "scoring/promote")
            )
        ),
        "done": route7_completed,
        "total": route7_total or route6_fail or route5_fail,
        "final": route7_final,
        "interim": route7_interim,
        "review": 0,
        "fail": route7_fail,
        "current_ok": 0,
        "promoted": route7_promoted,
        "promoted_final": route7_promoted_final,
        "promoted_interim": route7_promoted_interim,
        "queue_next": route7_fail,
    },
    {
        "id": "R8",
        "name": "full-T1 FNIRT + B0-frame rescue",
        "what": "label-limited fallback: estimate nonlinear MNI transform with full T1 against full MNI152, then test inverse-BBR or same-grid B0-frame AAL3 candidates",
        "short": "full-T1 nonlinear AAL3 with inverse-BBR/same-grid B0 candidates",
        "tests": [
            "parcellation: AAL3/MNI -> full native T1 with inverse FNIRT; if T1/B0 grids match, test inverse-BBR first and same-grid fallback",
            "assignment variants: forward80, forward120, radial8 using existing production tracks and SIFT2 weights",
            "purpose: fix label survival failures seen with skull-stripped FNIRT before accepting unresolved status",
            "promotion rule: final QC needs numeric+visual pass; interim density-improved outputs can be pushed but remain open",
        ],
        "status": "running" if route8_root.exists() and route8_completed < route8_total else ("not started" if not route8_root.exists() else "scoring/promote"),
        "done": route8_completed,
        "total": route8_total or route6_diagnostic_buckets.get("label_limited_diagnostic", 0),
        "final": route8_final,
        "interim": route8_interim,
        "review": 0,
        "fail": route8_fail,
        "current_ok": 0,
        "promoted": route8_promoted,
        "promoted_final": route8_promoted_final,
        "promoted_interim": route8_promoted_interim,
        "queue_next": route8_fail,
    },
    {
        "id": "R9",
        "name": "Route8 parcellation assignment retry",
        "what": "reuse high-label Route8 AAL3-B0 parcellations and retry endpoint/radial/short-forward assignment policies without rerunning FNIRT or tractography",
        "short": "cheap assignment retry on Route8 parcellations",
        "tests": [
            "candidate parcellations: Route8 inverse-BBR, BBR, and same-grid AAL3 outputs with label survival >=160/166",
            "assignment variants: endvox, radial8, radial16, forward40 by default; longer variants only if explicitly requested",
            "purpose: rescue Route8 subjects that failed because a single long forward-search assignment stalled or produced a sparse matrix",
            "promotion rule: final QC needs numeric+visual pass; interim density-improved outputs can be pushed but remain open",
        ],
        "status": "running" if route9_root.exists() and route9_completed < route9_total else ("not started" if not route9_root.exists() else "scoring/promote"),
        "done": route9_completed,
        "total": route9_total or route8_fail + route8_interim,
        "final": route9_final,
        "interim": route9_interim,
        "review": 0,
        "fail": route9_fail,
        "current_ok": 0,
        "promoted": route9_promoted,
        "promoted_final": route9_promoted_final,
        "promoted_interim": route9_promoted_interim,
        "queue_next": route9_fail,
    },
]

print("QC gates: full solve requires labels>=160/166, zero_rows<=15, density>=0.10, assignment QC, and visual pass; interim production overwrite can happen for density improvement but stays open")
print("Flow: unresolved R1 -> R2 -> R3 -> R4 -> R5 -> R6 diagnostic -> R7 selective rescue + R8 full-T1 FNIRT -> R9 Route8 assignment retry")
print(
    "Production promoted: "
    f"unique_subjects={len(unique_promoted)} | unique_final_ok={unique_promoted_final} | "
    f"unique_interim_review={unique_promoted_interim} | promotion_events={total_promoted_all_routes}"
)
for route in route_plan:
    denom = int(route["total"] or 0)
    complete = int(route["done"] or 0)
    fill = int(18 * complete / denom) if denom else 0
    bar = "#" * fill + "." * (18 - fill)
    if denom:
        progress = f"[{bar}] {complete}/{denom}"
    else:
        progress = "[..................] 0/0"
    print(
        f"{route['id']} {route['name']:<36} {progress} | "
        f"{route['status']} | final={route['final']} interim={route['interim']} "
        f"current_ok={route['current_ok']} review={route['review']} fail={route['fail']} promoted={route['promoted']} "
        f"next_queue={route['queue_next']} | {route['short']}"
    )
    if route["promoted"]:
        print(f"   promoted breakdown: final_ok={route.get('promoted_final', 0)} | interim_review={route.get('promoted_interim', 0)}")
    if route["id"] == "R6" and route6_diagnostic_buckets:
        print("   diagnostic buckets: " + " | ".join(f"{k}={v}" for k, v in sorted(route6_diagnostic_buckets.items())))
    print(f"   goal: {route['what']}")
    for test in route["tests"]:
        print(f"   - {test}")
if route1_unscored:
    print(f"pending in active top progress: {route1_unscored} subjects have not reached route-1 scoring yet")
def active_summary(lines, limit=8):
    shown = lines[:limit]
    extra = len(lines) - len(shown)
    suffix = f" | +{extra} more active" if extra > 0 else ""
    return " | ".join(shown) + suffix

if route2_active_lines:
    print("route 2 active: " + active_summary(route2_active_lines))
if route3_active_lines:
    print("route 3 active: " + active_summary(route3_active_lines))
if route4_active_lines:
    print("route 4 active: " + active_summary(route4_active_lines))
if route5_active_lines:
    print("route 5 active: " + active_summary(route5_active_lines))
if route6_active_lines:
    print("route 6 active: " + active_summary(route6_active_lines))
if route7_active_lines:
    print("route 7 active: " + active_summary(route7_active_lines))
if route8_active_lines:
    print("route 8 active: " + active_summary(route8_active_lines))
if route9_active_lines:
    print("route 9 active: " + active_summary(route9_active_lines))
mrtrix_jobs = active_mrtrix_jobs()
if mrtrix_jobs:
    print("active MRtrix: " + " | ".join(mrtrix_jobs[:6]))
PY

echo
attach_line() {
  local label="$1"
  shift
  local session
  for session in "$@"; do
    # tmux target matching accepts prefixes, so force exact session matching.
    if tmux ls 2>/dev/null | sed 's/:.*//' | grep -Fxq "$session"; then
      printf "%-15s tmux attach -t %s\n" "$label:" "$session"
      return 0
    fi
    if [[ "$session" == "aal3_ad_rescue_monitor" ]] && pgrep -af "watch_aal3_force_connectome_ad_batch.sh" >/dev/null 2>&1; then
      printf "%-15s tmux attach -t %s (watch loop active)\n" "$label:" "$session"
      return 0
    fi
  done
  printf "%-15s not active\n" "$label:"
}

attach_line "attach run" aal3_ad_rescue
attach_line "attach monitor" aal3_ad_rescue_monitor
attach_line "attach route2" aal3_ad_route2_followup aal3_ad_route2_bbr
attach_line "attach route3" aal3_ad_route3_assignment
attach_line "attach route4" aal3_ad_route4_label_rescue
attach_line "attach route5" aal3_ad_route5_deep_assignment
attach_line "attach route6" aal3_ad_route6_all_voxels
attach_line "attach route7" aal3_ad_route7_track_rescue
attach_line "attach route9" aal3_ad_route9_assignment_retry
attach_line "attach chain" aal3_ad_route_followup_chain
echo "one-shot:       /home/ec2-user/exp/scripts/scforge/live/watch_aal3_force_connectome_ad_batch.sh"
