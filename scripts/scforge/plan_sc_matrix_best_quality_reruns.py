#!/usr/bin/env python3
"""Build a selective rerun plan for SC matrix quality recovery.

This is intentionally read-only with respect to production derivatives. It reads
the current SC matrix QC decision table and writes a per-subject action plan that
answers which subjects should be kept and which earliest pipeline stage should
be rerun if we want the best available SC matrices without blindly reprocessing
every subject from raw DWI.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_QC_DIR = Path("/home/ec2-user/exp/data/derivatives/qc/sc_matrix_qc")
DEFAULT_DECISIONS = DEFAULT_QC_DIR / "sc_matrix_qc_decisions.csv"
DEFAULT_OUT = DEFAULT_QC_DIR / "sc_matrix_best_quality_rerun_plan.csv"
DEFAULT_SUMMARY = DEFAULT_QC_DIR / "sc_matrix_best_quality_rerun_summary.csv"


STATUS_PLAN = {
    "PASS": {
        "action": "KEEP",
        "earliest_stage": "none",
        "rerun_scope": "none",
        "canary_required": "no",
        "production_policy": "use_current_output",
        "reason": "SC matrix passes current whole-matrix QC gate.",
    },
    "WARN_SPARSE_NODE": {
        "action": "KEEP_WITH_REVIEW",
        "earliest_stage": "none",
        "rerun_scope": "none",
        "canary_required": "no",
        "production_policy": "include_only_if_warn_review_accepts_sparse_nodes",
        "reason": "Sparse-node warning; do not rerun automatically unless manual review rejects it.",
    },
    "FAIL_LABEL_COVERAGE": {
        "action": "REPAIR_CANARY_FIRST",
        "earliest_stage": "step7_post_parcellation",
        "rerun_scope": "AAL_b0 plus tck2connectome outputs; reuse existing tracks if valid",
        "canary_required": "yes",
        "production_policy": "backup_validate_promote_only_if_density_and_zero_rows_improve",
        "reason": "AAL label coverage is absent or too small in B0-space parcellation.",
    },
    "FAIL_REGISTRATION": {
        "action": "REPAIR_CANARY_FIRST",
        "earliest_stage": "bbr_t1_to_b0_then_step7_post",
        "rerun_scope": "T1-to-B0 bridge, AAL_b0, post matrices; rerun prep/FOD/tracks only if geometry changes invalidate them",
        "canary_required": "yes",
        "production_policy": "backup_validate_promote_only_if spatial overlap and matrix QC improve",
        "reason": "AAL labels exist but overlap poorly with B0/5TT/FOD/DWI mask space.",
    },
    "FAIL_STREAMLINE_ASSIGNMENT": {
        "action": "REPAIR_CANARY_FIRST",
        "earliest_stage": "step7_post_assignment",
        "rerun_scope": "tck2connectome assignment variants; reuse existing parcellation and tracks if overlay is acceptable",
        "canary_required": "yes",
        "production_policy": "backup_validate_promote_only_if assignments improve and matrices remain complete",
        "reason": "Anatomy is less likely to be the blocker; streamline-to-parcel assignment or track endpoints need diagnostics.",
    },
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--decisions", type=Path, default=DEFAULT_DECISIONS)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    args = ap.parse_args()

    rows = read_rows(args.decisions)
    if not rows:
        raise SystemExit(f"No rows found in {args.decisions}")

    planned: list[dict[str, str]] = []
    for r in rows:
        status = r.get("sc_matrix_qc_status", "").strip()
        plan = STATUS_PLAN.get(
            status,
            {
                "action": "REVIEW",
                "earliest_stage": "unknown",
                "rerun_scope": "manual_review",
                "canary_required": "yes",
                "production_policy": "do_not_promote_without_manual_decision",
                "reason": f"Unmapped QC status: {status}",
            },
        )
        planned.append(
            {
                "sid": r.get("sid", ""),
                "subject_id": r.get("subject_id", ""),
                "group": r.get("group", ""),
                "sc_matrix_qc_status": status,
                "analysis_gate": r.get("analysis_gate", ""),
                "density": r.get("density", ""),
                "n_unexpected_zero_rows": r.get("n_unexpected_zero_rows", ""),
                "frontal_oper_l_zero": r.get("frontal_oper_l_zero", ""),
                "frontal_oper_r_zero": r.get("frontal_oper_r_zero", ""),
                "action": plan["action"],
                "earliest_stage": plan["earliest_stage"],
                "rerun_scope": plan["rerun_scope"],
                "canary_required": plan["canary_required"],
                "production_policy": plan["production_policy"],
                "reason": plan["reason"],
                "source_decision_reasons": r.get("decision_reasons", ""),
            }
        )

    fields = list(planned[0])
    write_rows(args.out, planned, fields)

    summary_rows: list[dict[str, str | int]] = []
    counts = Counter((r["group"], r["sc_matrix_qc_status"], r["action"], r["earliest_stage"]) for r in planned)
    for (group, status, action, earliest_stage), n in sorted(counts.items()):
        summary_rows.append(
            {
                "group": group,
                "sc_matrix_qc_status": status,
                "action": action,
                "earliest_stage": earliest_stage,
                "n_subjects": n,
            }
        )
    write_rows(args.summary, summary_rows, ["group", "sc_matrix_qc_status", "action", "earliest_stage", "n_subjects"])

    action_counts = Counter(r["action"] for r in planned)
    stage_counts = Counter(r["earliest_stage"] for r in planned)
    print(f"Wrote {args.out}")
    print(f"Wrote {args.summary}")
    print("actions:", dict(action_counts))
    print("earliest_stages:", dict(stage_counts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
