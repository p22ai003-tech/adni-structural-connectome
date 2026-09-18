#!/usr/bin/env python3
"""Strict post-hoc review for scratch SC matrix rectification runs.

This script does not modify production outputs. It reads a rectification run's
validation_summary.csv / promotion_manifest.csv and writes stricter review files
that distinguish partial numerical improvement from analysis-eligible repair.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from pathlib import Path
from typing import Any


MIN_DENSITY = 0.15
WARN_ZERO_ROWS = 20


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fnum(value: Any, default: float = math.nan) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def inum(value: Any, default: int = 999999) -> int:
    try:
        return int(float(value))
    except Exception:
        return default


def review_row(row: dict[str, str]) -> dict[str, Any]:
    lane = row.get("lane", "")
    density = fnum(row.get("candidate_density") or row.get("best_density"))
    baseline_density = fnum(row.get("baseline_density"))
    zero = inum(row.get("candidate_zero_rows") or row.get("best_zero_rows"))
    baseline_zero = inum(row.get("baseline_zero_rows"))
    density_delta = fnum(row.get("density_delta"), density - baseline_density)
    zero_delta = inum(row.get("zero_delta"), zero - baseline_zero)

    strict = "NO_NUMERIC_REVIEW"
    reason = "row has no candidate density/zero-row fields"
    batch_allowed = 0

    if lane == "MNI_TO_T1_LABEL_SURVIVAL":
        strict = "LABEL_ROUTE_DIAGNOSTIC_ONLY"
        reason = "label survival diagnostics do not produce complete SC matrices"
    elif math.isfinite(density):
        if density_delta < 0 or zero_delta > 0:
            strict = "RESTORE_OR_RETRY_DIFFERENT_LANE"
            reason = "candidate worsens density or zero-row burden versus baseline"
        elif density < MIN_DENSITY:
            strict = "PARTIAL_IMPROVEMENT_NOT_BATCH_READY"
            reason = f"candidate density {density:.4g} is below validation floor {MIN_DENSITY}"
        elif zero > WARN_ZERO_ROWS:
            strict = "DENSITY_PASS_BUT_ZERO_ROWS_REMAIN"
            reason = f"candidate density passes, but {zero} valid zero rows remain above review threshold {WARN_ZERO_ROWS}"
        else:
            strict = "BATCH_READY_CANDIDATE"
            reason = "candidate passes density floor and zero-row warning threshold"
            batch_allowed = 1

    return {
        **row,
        "strict_decision": strict,
        "strict_reason": reason,
        "strict_batch_allowed": batch_allowed,
        "strict_min_density": MIN_DENSITY,
        "strict_warn_zero_rows": WARN_ZERO_ROWS,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    args = parser.parse_args()

    run_root = args.run_root
    rows = [review_row(row) for row in read_csv(run_root / "validation_summary.csv")]
    write_csv(run_root / "strict_validation_summary.csv", rows)

    promo_rows = [review_row(row) for row in read_csv(run_root / "promotion_manifest.csv")]
    write_csv(run_root / "strict_promotion_manifest.csv", promo_rows)

    counts = Counter(row.get("strict_decision", "UNKNOWN") for row in rows)
    lines = ["# Strict SC Matrix Rectification Review", ""]
    lines.append(f"- Minimum density gate: {MIN_DENSITY}")
    lines.append(f"- Zero-row warning threshold: {WARN_ZERO_ROWS}")
    lines.append("")
    for key, value in sorted(counts.items()):
        lines.append(f"- {key}: {value}")
    lines.append("")
    lines.append("Production outputs remain untouched. Batch repair is allowed only for rows marked `BATCH_READY_CANDIDATE`, and only after manual review of the full scratch artifacts.")
    (run_root / "strict_findings_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(run_root / "strict_validation_summary.csv")
    print(run_root / "strict_findings_summary.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
