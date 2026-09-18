#!/usr/bin/env python3
"""Print a compact status screen for the final SC spatial-contract closeout."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

QC_ROOT = Path("/home/ec2-user/exp/data/derivatives/qc/sc_matrix_qc")
LATEST = QC_ROOT / "latest_final_spatial_contract_closeout.txt"


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size <= 0:
        return []
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        return list(csv.DictReader(handle))


def bar(done: int, total: int, width: int = 26) -> str:
    if total <= 0:
        return "[" + "." * width + "]"
    n = min(width, max(0, round(width * done / total)))
    return "[" + "#" * n + "." * (width - n) + "]"


def main() -> int:
    if LATEST.exists():
        root = Path(LATEST.read_text(encoding="utf-8").strip())
    else:
        roots = sorted(QC_ROOT.glob("final_spatial_contract_closeout_*"), key=lambda p: p.stat().st_mtime)
        root = roots[-1] if roots else None
    if not root or not root.exists():
        print("SC spatial-contract closeout monitor")
        print("No closeout run found yet.")
        return 0

    status = read_json(root / "status.json")
    decisions = read_csv(root / "subject_route_decisions.csv")
    summary = read_csv(root / "canary_validation_summary.csv")
    total = int(status.get("total_subjects") or len(decisions) or 0)
    done = int(status.get("completed_subjects") or len([r for r in decisions if r.get("probe_status") == "done"]))
    failed = int(status.get("failed_subjects") or len([r for r in decisions if str(r.get("probe_status", "")).startswith("failed")]))
    strict = int(status.get("strict_pass_subjects") or sum(1 for r in decisions if str(r.get("strict_pass")) == "1"))

    print(f"SC spatial-contract closeout | {status.get('last_update_utc', '')}")
    print(f"root   {root}")
    print(f"phase  {status.get('phase', 'unknown')}")
    print(f"sid    {status.get('current_sid', '-') } | role={status.get('current_role', '-') } | variant={status.get('current_variant', '-')}")
    print()
    pct = (100.0 * done / total) if total else 0.0
    print(f"probe      {bar(done, total)} {pct:5.1f}% done={done}/{total} fail={failed}")
    print(f"strict     {bar(strict, max(total, 1))} strict_full_matrix_pass={strict}/{total}")
    print()
    print(f"latest: {status.get('latest_finding', '-')}")
    print(f"next  : {status.get('next_allowed_action', 'wait for closeout decision')}")
    if summary:
        row = summary[-1]
        print()
        print(
            "summary "
            f"completed={row.get('completed_subjects')} strict={row.get('strict_pass_subjects')} "
            f"improves={row.get('improves_but_fails_strict_gate')} recommendation={row.get('final_recommendation')}"
        )
    if decisions:
        print()
        print("recent decisions")
        for row in decisions[-6:]:
            print(
                f"  {row.get('sid',''):<24} {row.get('route_decision',''):<38} "
                f"{row.get('decision','')} dens={row.get('candidate_fd_valid_density','')}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
