#!/usr/bin/env python3
"""Generate group-wise QC report for completed EC2 connectomes.

The report is read-only. It audits completed SC matrices with the existing
whole-matrix QC workflow and writes concise group summaries for CN/MCI/AD.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path("/home/ec2-user/exp")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from connectome_analysis.analysis_config import get_analysis_paths  # noqa: E402
from connectome_analysis.analysis_sc_matrix_qc import run_sc_matrix_qc  # noqa: E402
from connectome_pipeline.pipeline_status import _load_subject_group_map, collect_all_stage_group_status  # noqa: E402

SUBJECT_RE = re.compile(r"^(\d{3}_S_\d{4})")
REQUIRED_POST_SUFFIXES = (
    "ALL",
    "count",
    "fd_sum",
    "len_mean",
    "invlen_mean",
    "fa_mean",
    "md_mean",
    "rd_mean",
    "ad_mean",
    "count_invnodevol",
)


def subject_id(sid: str) -> str:
    match = SUBJECT_RE.match(sid)
    return match.group(1) if match else sid.split("_I", 1)[0]


def connectome_sids(conn_dir: Path, suffix: str) -> set[str]:
    prefix = "SC_AAL_"
    postfix = f"_{suffix}.csv"
    if not conn_dir.exists():
        return set()
    return {
        path.name[len(prefix) : -len(postfix)]
        for path in conn_dir.glob(f"{prefix}*{postfix}")
        if path.name.startswith(prefix) and path.name.endswith(postfix)
    }


def completed_sids(conn_dir: Path) -> set[str]:
    all_candidates: set[str] = set()
    for suffix in REQUIRED_POST_SUFFIXES:
        all_candidates.update(connectome_sids(conn_dir, suffix))
    return {
        sid
        for sid in all_candidates
        if all((conn_dir / f"SC_AAL_{sid}_{suffix}.csv").exists() for suffix in REQUIRED_POST_SUFFIXES)
    }


def summarize_status(df: pd.DataFrame) -> pd.DataFrame:
    groups = ["CN", "MCI", "AD"]
    statuses = [
        "PASS",
        "WARN_SPARSE_NODE",
        "FAIL_REGISTRATION",
        "FAIL_LABEL_COVERAGE",
        "FAIL_STREAMLINE_ASSIGNMENT",
        "FAIL_SOURCE",
        "NOT_AUDITED",
    ]
    rows = []
    for group in groups:
        sub = df[df["group"] == group].copy()
        row = {"group": group, "completed_connectomes": int(len(sub))}
        for status in statuses:
            row[status] = int((sub["whole_matrix_qc_status"] == status).sum())
        row["analysis_ready_PASS_or_WARN"] = int(
            sub["whole_matrix_qc_status"].isin(["PASS", "WARN_SPARSE_NODE"]).sum()
        )
        row["exclude_pending_repair"] = int(
            sub["whole_matrix_qc_status"].str.startswith("FAIL_", na=False).sum()
            + (sub["whole_matrix_qc_status"] == "NOT_AUDITED").sum()
        )
        row["median_density"] = float(pd.to_numeric(sub["density"], errors="coerce").median()) if len(sub) else np.nan
        row["median_unexpected_zero_rows"] = (
            float(pd.to_numeric(sub["n_unexpected_zero_rows"], errors="coerce").median()) if len(sub) else np.nan
        )
        rows.append(row)
    total = pd.DataFrame(rows)
    if not total.empty:
        numeric = total.drop(columns=["group"]).apply(pd.to_numeric, errors="coerce")
        total_row = {"group": "Total"}
        median_source = {
            "median_density": "density",
            "median_unexpected_zero_rows": "n_unexpected_zero_rows",
        }
        for col in numeric.columns:
            if col.startswith("median_"):
                source_col = median_source.get(col)
                if source_col and source_col in df.columns:
                    total_row[col] = float(pd.to_numeric(df[source_col], errors="coerce").median())
                else:
                    total_row[col] = np.nan
            else:
                total_row[col] = int(numeric[col].fillna(0).sum())
        total = pd.concat([total, pd.DataFrame([total_row])], ignore_index=True)
    return total


def summarize_weights(matrix_audit: pd.DataFrame) -> pd.DataFrame:
    if matrix_audit.empty:
        return pd.DataFrame()
    rows = []
    for (group, weight), sub in matrix_audit.groupby(["group", "weight"], dropna=False):
        density = pd.to_numeric(sub["density"], errors="coerce")
        zero_rows = pd.to_numeric(sub["n_unexpected_zero_rows"], errors="coerce")
        rows.append(
            {
                "group": group,
                "weight": weight,
                "n_matrices": int(len(sub)),
                "read_errors": int(sub["read_error"].astype(str).str.len().gt(0).sum()),
                "finite_ok": int(pd.to_numeric(sub["finite_fraction"], errors="coerce").ge(1.0).sum()),
                "symmetric_ok": int(pd.to_numeric(sub["symmetry_max_abs"], errors="coerce").le(1e-6).sum()),
                "zero_diagonal_ok": int(pd.to_numeric(sub["diagonal_abs_sum"], errors="coerce").le(1e-6).sum()),
                "density_median": float(density.median()),
                "density_q25": float(density.quantile(0.25)),
                "density_q75": float(density.quantile(0.75)),
                "unexpected_zero_rows_median": float(zero_rows.median()),
            }
        )
    return pd.DataFrame(rows).sort_values(["group", "weight"])


def write_report(
    out_dir: Path,
    group_summary: pd.DataFrame,
    weight_summary: pd.DataFrame,
    stage_counts: pd.DataFrame,
) -> Path:
    def table_block(df: pd.DataFrame) -> str:
        if df.empty:
            return "No rows."
        return "```text\n" + df.to_string(index=False) + "\n```"

    lines = [
        "# Completed Structural Connectome QC Report",
        "",
        f"Generated UTC: `{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}`",
        "",
        "## Criteria",
        "",
        "- A completed connectome means all required AAL matrices exist: `ALL`, `count`, `fd_sum`, `len_mean`, `invlen_mean`, `fa_mean`, `md_mean`, `rd_mean`, `ad_mean`, and `count_invnodevol`.",
        "- Matrix integrity checks require square/readable matrices, finite values, symmetry, and zero diagonal.",
        "- Whole-matrix QC uses `fd_sum` as the primary structural matrix.",
        "- `PASS` is analysis-ready. `WARN_SPARSE_NODE` can be included only with explicit review. `FAIL_*` and `NOT_AUDITED` remain excluded pending repair.",
        "- Expected AAL3 label gaps are not counted as biological/registration failures; valid AAL3 zero rows are counted.",
        "",
        "## Stage Counts",
        "",
        table_block(stage_counts),
        "",
        "## Group-Wise Completed Connectome QC",
        "",
        table_block(group_summary),
        "",
        "## Matrix-Weight Integrity Summary",
        "",
        table_block(weight_summary) if not weight_summary.empty else "No matrix audit rows.",
        "",
        "## Output Files",
        "",
        f"- `connectome_qc_group_summary.csv`",
        f"- `connectome_qc_weight_summary.csv`",
        f"- Full SC QC CSVs in `{out_dir}`",
    ]
    report_path = out_dir / "completed_connectome_qc_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Reuse SC QC CSVs already present in output-dir instead of rerunning the heavy audit.",
    )
    args = parser.parse_args()

    paths = get_analysis_paths(
        notebook_dir=Path("/home/ec2-user/exp"),
        deriv_root=Path("/home/ec2-user/exp/data/derivatives"),
        cohort_dti_csv=Path("/home/ec2-user/exp/cohort/dti.csv"),
        cohort_mri_csv=Path("/home/ec2-user/exp/cohort/mri.csv"),
    )
    out_dir = args.output_dir or (
        paths.deriv_root / "qc" / "sc_matrix_qc" / f"completed_connectome_qc_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.reuse_existing and (out_dir / "sc_matrix_qc_repair_plan.csv").exists():
        outputs = {}
        for path in out_dir.glob("*.csv"):
            outputs[path.stem] = pd.read_csv(path)
    else:
        outputs = run_sc_matrix_qc(paths=paths, output_dir=out_dir)
    repair_plan = outputs.get("sc_matrix_qc_repair_plan", pd.DataFrame()).copy()
    matrix_audit = outputs.get("sc_matrix_integrity_subjects", pd.DataFrame()).copy()

    group_map = _load_subject_group_map(paths.cohort_dti_csv)
    complete = sorted(completed_sids(paths.connectomes_dir))
    rows = []
    repair_lookup = {
        str(row.sid): row._asdict()
        for row in repair_plan.itertuples(index=False)
    } if not repair_plan.empty else {}
    for sid in complete:
        rec = repair_lookup.get(sid, {})
        group = rec.get("group") or group_map.get(subject_id(sid), "")
        rows.append(
            {
                "sid": sid,
                "subject_id": subject_id(sid),
                "group": group,
                "whole_matrix_qc_status": rec.get("whole_matrix_qc_status", "NOT_AUDITED"),
                "analysis_gate": rec.get("analysis_gate", "exclude_pending_repair"),
                "repair_lane": rec.get("repair_lane", ""),
                "density": rec.get("density", np.nan),
                "n_unexpected_zero_rows": rec.get("n_unexpected_zero_rows", np.nan),
                "whole_matrix_reason": rec.get("whole_matrix_reason", ""),
            }
        )
    completed_df = pd.DataFrame(rows)
    group_summary = summarize_status(completed_df)
    weight_summary = summarize_weights(matrix_audit)

    status = collect_all_stage_group_status(location="ec2")
    stage_rows = []
    for row in status["rows"] + [status["total_row"]]:
        stage_rows.append(
            {
                "group": row["group"],
                "FOD": row["stages"]["FOD"]["done"],
                "Tracks": row["stages"]["Tracks"]["done"],
                "Parc": row["stages"]["Parc"]["done"],
                "DTI": row["stages"]["DTI"]["done"],
                "Connectomes": row["stages"]["Connectomes"]["done"],
            }
        )
    stage_counts = pd.DataFrame(stage_rows)

    completed_df.to_csv(out_dir / "connectome_qc_subjects.csv", index=False)
    group_summary.to_csv(out_dir / "connectome_qc_group_summary.csv", index=False)
    weight_summary.to_csv(out_dir / "connectome_qc_weight_summary.csv", index=False)
    stage_counts.to_csv(out_dir / "connectome_stage_counts.csv", index=False)
    report_path = write_report(out_dir, group_summary, weight_summary, stage_counts)
    (out_dir / "run_metadata.json").write_text(
        json.dumps({"report_path": str(report_path), "output_dir": str(out_dir)}, indent=2),
        encoding="utf-8",
    )
    print(f"QC report complete: {report_path}")
    print(group_summary.to_string(index=False))


if __name__ == "__main__":
    main()
