#!/usr/bin/env python3
"""Build an evidence-based next-action queue for AD AAL3 rescue.

This is a routing/triage report, not a processing route. It combines:

* the consolidated AD route ledger,
* the endpoint-space action audit,
* production tractogram header audit, and
* currently running R18/R19 subjects where available.

The output separates subjects that need upstream tract/FOD/ACT repair from
subjects where parcellation/assignment repair is still the right lever.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path("/home/ec2-user/exp")
RUN_ROOT = Path(
    "/data/derivatives/qc/sc_matrix_qc/aal3_force_connectome_ad_batch/"
    "ad_existing_tracks_20260529T055752Z"
)
REPORT_DIR = ROOT / "reports" / "aal3_next_actions"
ENDPOINT_ACTION_CSV = (
    ROOT
    / "reports"
    / "aal3_endpoint_space_contract_ad_production_5k"
    / "aal3_ad_endpoint_space_action_latest.csv"
)
TRACK_HEADER_CSV = ROOT / "reports" / "aal3_track_header_audit" / "aal3_ad_track_header_audit_latest.csv"


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def stamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


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


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
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


def fnum(value: Any, default: float = float("nan")) -> float:
    try:
        if value in {"", None}:
            return default
        out = float(value)
        return out
    except Exception:
        return default


def running_subjects() -> dict[str, str]:
    """Return best-effort map sid -> active route from process args."""
    active: dict[str, str] = {}
    for route, suffix in (
        ("R18", "_route18_gmwmi_dense_track_coverage"),
        ("R19", "_route19_gmwmi_3m_retry"),
    ):
        status = read_json(Path(f"{RUN_ROOT}{suffix}") / "status.json")
        for sid in status.get("active", []) or []:
            if isinstance(sid, str) and sid:
                active[sid] = route
    try:
        proc = subprocess.run(
            ["ps", "-eo", "args="],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return active
    route_hint = ""
    for line in proc.stdout.splitlines():
        if "run_aal3_route18_gmwmi_dense_track_coverage.py" in line:
            route_hint = "R18"
        elif "run_aal3_route19_gmwmi_3m_retry" in line:
            route_hint = "R19"
        elif "aal3_route18_gmwmi_dense_track_coverage_probe" in line:
            route_hint = "R18"
        elif "aal3_route19_gmwmi_3m_retry_probe" in line:
            route_hint = "R19"
        else:
            continue
        for part in line.replace("/", " ").split():
            if "_S_" in part and "_I" in part:
                sid = part.strip(" ,'\"")
                sid = sid.split("_route", 1)[0]
                active[sid] = route_hint
    return active


def classify(row: dict[str, Any]) -> tuple[str, str]:
    state = str(row.get("state") or "")
    action = str(row.get("recommended_action") or "")
    active_route = str(row.get("active_route") or "")
    labels = fnum(row.get("parc_label_count"), 0)
    unique = fnum(row.get("unique_endpoint_labels"), 0)
    top3 = fnum(row.get("top3_endpoint_label_fraction"), 1)
    best_zero = fnum(row.get("best_zero_rows"), 9999)
    best_density = fnum(row.get("best_density"), 0)
    source = str(row.get("source_basename") or "")
    max_tracks = str(row.get("max_num_tracks") or "")
    chunked = str(row.get("command_history_has_chunks") or "")

    if state == "solved_full_qc_production" or action == "no_action_full_qc_solved":
        return "solved_full_qc", "Already full-QC solved; do not rerun."
    if active_route:
        return "wait_active_rerun", f"{active_route} is currently running; score endpoint-space after it finishes."
    if action == "upstream_track_fod_act_repair":
        return (
            "upstream_endpoint_preflight",
            (
                "Production endpoints collapse into very few labels; run a small "
                "single-pass wmfod.mif GMWMI/dynamic endpoint-space preflight before "
                "SIFT2/connectome."
            ),
        )
    if labels < 160:
        return "highres_label_contract_rebuild", "AAL3 label survival is below 160/166; repair atlas-to-B0 label contract first."
    if unique < 30 or top3 > 0.60:
        return (
            "endpoint_distribution_review",
            "Endpoint cloud is plausible but concentrated; verify overlay/endpoints before another dense connectome route.",
        )
    if best_density >= 0.10 and best_zero <= 35:
        return (
            "near_qc_assignment_refine",
            "Close to QC; prioritize assignment variants/visual overlay rather than upstream tractography.",
        )
    if chunked == "1" and source == "wmfod_final.mif" and max_tracks == "200000":
        return (
            "parcellation_assignment_with_track_risk",
            "Chunked wmfod_final production tracks are risky, but endpoint cloud is broad enough; keep parcellation/assignment routes active.",
        )
    return "parcellation_assignment_repair", "Endpoint cloud is broad enough; continue parcellation/assignment/label-contract repair."


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    parser.add_argument("--endpoint-action-csv", type=Path, default=ENDPOINT_ACTION_CSV)
    parser.add_argument("--track-header-csv", type=Path, default=TRACK_HEADER_CSV)
    parser.add_argument("--out-dir", type=Path, default=REPORT_DIR)
    args = parser.parse_args()

    endpoint = {row["sid"]: row for row in read_csv(args.endpoint_action_csv) if row.get("sid")}
    headers = {row["sid"]: row for row in read_csv(args.track_header_csv) if row.get("sid")}
    ledger = {row["sid"]: row for row in read_csv(args.run_root / "route_subject_ledger.csv") if row.get("sid")}
    active = running_subjects()

    sids = sorted(set(endpoint) | set(headers) | set(ledger))
    rows: list[dict[str, Any]] = []
    for sid in sids:
        row: dict[str, Any] = {"sid": sid}
        # Endpoint/header audits are diagnostic and can lag behind promotion
        # state. Merge them first, then let the consolidated route ledger remain
        # authoritative for state/promotion/final-QC fields.
        row.update(endpoint.get(sid, {}))
        row.update(headers.get(sid, {}))
        row.update(ledger.get(sid, {}))
        row["active_route"] = active.get(sid, "")
        next_action, reason = classify(row)
        row["next_action"] = next_action
        row["next_action_reason"] = reason
        rows.append(row)

    counts = Counter(row["next_action"] for row in rows)
    payload = {
        "generated_utc": utc(),
        "rows": len(rows),
        "counts": dict(counts),
        "csv": "",
    }
    now = stamp()
    out_csv = args.out_dir / f"aal3_ad_next_action_queue_{now}.csv"
    out_json = args.out_dir / f"aal3_ad_next_action_queue_{now}.json"
    latest_csv = args.out_dir / "aal3_ad_next_action_queue_latest.csv"
    latest_json = args.out_dir / "aal3_ad_next_action_queue_latest.json"
    payload["csv"] = str(out_csv)
    write_csv(out_csv, rows)
    write_csv(latest_csv, rows)
    write_json(out_json, payload)
    write_json(latest_json, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
