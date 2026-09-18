#!/usr/bin/env python3
"""Route 12C: registration/space-contract repair for unresolved AAL3 cases.

This reuses the endpoint-overlap candidate search from Route 11A, but writes to
Route 12C-specific roots so the supervisor and monitor can track it as the next
lane after R11C.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


ROOT = Path("/home/ec2-user/exp")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.scforge.live import run_aal3_route11a_space_affine_registration_repair as route11a  # noqa: E402
from scripts.scforge.live.run_sc_aal3_source_contract_probe import QC_ROOT, write_csv, write_json  # noqa: E402


RUN_PARENT = Path("/data/derivatives/qc/sc_matrix_qc/aal3_force_connectome_ad_batch")
DEFAULT_ROUTE1_ROOT = RUN_PARENT / "ad_existing_tracks_20260529T055752Z"
DEFAULT_ROUTE10_ROOT = DEFAULT_ROUTE1_ROOT.parent / f"{DEFAULT_ROUTE1_ROOT.name}_route10_endpoint_overlay_triage"
ROUTE12C_SUFFIX = "_route12c_registration_contract_repair"
ROUTE12C_SCHEMA = "route12c_registration_contract_repair_v1"
PROBE_PARENT = QC_ROOT / "aal3_route12c_registration_contract_repair_probe"
STATUS_LOCK = threading.Lock()


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


def update_status(root: Path, **updates: Any) -> None:
    with STATUS_LOCK:
        status = read_json(root / "status.json")
        status.update(updates)
        status["updated_utc"] = utc()
        write_json(root / "status.json", status)


def finished_sids(run_root: Path, retry_failed: bool) -> set[str]:
    done: set[str] = set()
    for row in read_csv(run_root / "batch_results.csv"):
        sid = row.get("sid", "")
        if not sid:
            continue
        if row.get("status") == "ok" and row.get("route11a_schema") == ROUTE12C_SCHEMA:
            done.add(sid)
        elif row.get("status") == "failed" and not retry_failed:
            done.add(sid)
    return done


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route1-root", type=Path, default=DEFAULT_ROUTE1_ROOT)
    parser.add_argument("--route10-root", type=Path, default=DEFAULT_ROUTE10_ROOT)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--mrtrix-threads", type=int, default=2)
    parser.add_argument("--variants", nargs="*", default=["forward40", "forward80", "radial16", "endvox"])
    parser.add_argument("--max-candidates", type=int, default=3)
    parser.add_argument("--connectome-timeout-sec", type=int, default=45 * 60)
    parser.add_argument("--subjects", nargs="*", default=[])
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()

    route1_root = args.route1_root
    route10_root = args.route10_root
    out_root = route1_root.parent / f"{route1_root.name}{ROUTE12C_SUFFIX}"
    out_root.mkdir(parents=True, exist_ok=True)
    PROBE_PARENT.mkdir(parents=True, exist_ok=True)

    # Reuse Route 11A's tested implementation but redirect its schema/probe root.
    route11a.PROBE_PARENT = PROBE_PARENT
    route11a.ROUTE11A_SCHEMA = ROUTE12C_SCHEMA

    rows = [row for row in read_csv(out_root / "route_queue.csv") if row.get("sid")]
    if args.subjects:
        wanted = set(args.subjects)
        rows = [row for row in rows if row.get("sid") in wanted]
    done = finished_sids(out_root, args.retry_failed)
    rows = [row for row in rows if row.get("sid") not in done]
    tag = out_root.name

    update_status(
        out_root,
        route12c_schema=ROUTE12C_SCHEMA,
        phase="running",
        route1_root=str(route1_root),
        route10_root=str(route10_root),
        total=len(done) + len(rows),
        completed=len(done),
        failed=0,
        active=[],
    )

    existing = read_csv(out_root / "batch_results.csv")
    results: list[dict[str, Any]] = [row for row in existing if row.get("sid") in done]
    active: set[str] = set()

    def wrapped(input_row: dict[str, str]) -> dict[str, Any]:
        sid = input_row["sid"]
        active.add(sid)
        update_status(out_root, active=sorted(active))
        try:
            return route11a.run_one(
                row=input_row,
                tag=tag,
                route10_root=route10_root,
                variants=args.variants,
                max_candidates=max(1, args.max_candidates),
                mrtrix_threads=max(1, args.mrtrix_threads),
                timeout_sec=args.connectome_timeout_sec,
            )
        finally:
            active.discard(sid)
            update_status(out_root, active=sorted(active))

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(wrapped, row): row for row in rows}
        for future in as_completed(futures):
            input_row = futures[future]
            sid = input_row.get("sid", "")
            try:
                row = future.result()
            except Exception as exc:
                row = {
                    "sid": sid,
                    "status": "failed",
                    "route12c_schema": ROUTE12C_SCHEMA,
                    "error": f"{type(exc).__name__}: {exc}",
                    "finished_utc": utc(),
                }
            results = [r for r in results if r.get("sid") != sid] + [row]
            write_csv(out_root / "batch_results.csv", results)
            update_status(
                out_root,
                completed=sum(1 for r in results if r.get("status") == "ok"),
                failed=sum(1 for r in results if r.get("status") == "failed"),
                active=sorted(active),
                latest=row,
            )

    summary = {
        "generated_utc": utc(),
        "route12c_schema": ROUTE12C_SCHEMA,
        "route1_root": str(route1_root),
        "route10_root": str(route10_root),
        "out_root": str(out_root),
        "total": len(results),
        "completed": sum(1 for row in results if row.get("status") == "ok"),
        "failed": sum(1 for row in results if row.get("status") == "failed"),
    }
    write_json(out_root / "route12c_summary.json", summary)
    update_status(out_root, phase="complete", completed=summary["completed"], failed=summary["failed"], active=[], summary=summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"wrote {out_root / 'batch_results.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
