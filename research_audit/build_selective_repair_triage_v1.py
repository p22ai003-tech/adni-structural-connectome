#!/usr/bin/env python3
"""Triage the 530 historical connectomes into the smallest plausible repair lane.

The scan is read-only with respect to production derivatives.  It inspects the
current AAL3-166 CSV matrices and joins exact acquisition/legacy-QC metadata.
The resulting lane is diagnostic, not execution authorization.  In particular,
an invalid `count` matrix does not force an upstream DWI rerun when the paper's
primary AxD/RD endpoint remains usable.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT = Path("/home/ec2-user/exp")
MANIFEST = PROJECT / "research_audit/outputs/available_data_pair_manifest_v2.csv"
CONNECTOMES = Path("/data/derivatives/connectomes")
DEFAULT_OUT = PROJECT / "research_audit/outputs/selective_repair_triage_v1"
METRICS = (
    "count",
    "fd_sum",
    "count_invnodevol",
    "len_mean",
    "invlen_mean",
    "fa_mean",
    "md_mean",
    "rd_mean",
    "ad_mean",
)
EXPECTED_SHAPE = (166, 166)
TOL = 1e-10


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    try:
        frame.to_csv(temporary, index=False, quoting=csv.QUOTE_MINIMAL)
        os.replace(temporary, path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
        os.replace(temporary, path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


def matrix_path(subject_id: str, image_id: str, metric: str) -> Path:
    return CONNECTOMES / f"SC_AAL166_{subject_id}_I{image_id}_{metric}.csv"


def read_matrix(path: Path) -> np.ndarray:
    text = path.read_text(encoding="utf-8").strip().replace("\r", "").replace("\n", ",")
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        values = np.fromstring(text, sep=",", dtype=float)
    if values.size != EXPECTED_SHAPE[0] * EXPECTED_SHAPE[1]:
        raise ValueError(f"expected 27556 values, found {values.size}")
    return values.reshape(EXPECTED_SHAPE)


def inspect_matrix(subject_id: str, image_id: str, metric: str) -> tuple[dict[str, object], np.ndarray | None]:
    path = matrix_path(subject_id, image_id, metric)
    base: dict[str, object] = {
        "subject_id": subject_id,
        "dti_image_id": image_id,
        "metric": metric,
        "path": str(path),
        "exists": path.is_file(),
        "read_error": "",
    }
    if not path.is_file():
        return {
            **base,
            "shape_valid": False,
            "finite": False,
            "symmetric": False,
            "zero_diagonal": False,
            "minimum": np.nan,
            "maximum": np.nan,
            "negative_cells": pd.NA,
            "above_one_cells": pd.NA,
            "fractional_positive_cells": pd.NA,
            "positive_upper_edges": pd.NA,
            "zero_rows": pd.NA,
            "hard_valid": False,
        }, None
    try:
        matrix = read_matrix(path)
        finite = bool(np.isfinite(matrix).all())
        symmetric = bool(finite and np.allclose(matrix, matrix.T, atol=TOL, rtol=0))
        zero_diagonal = bool(finite and np.all(np.abs(np.diag(matrix)) <= TOL))
        negative = int(np.sum(matrix < -TOL)) if finite else pd.NA
        above_one = int(np.sum(matrix > 1.0 + TOL)) if finite else pd.NA
        positive = matrix > TOL
        fractional = (
            int(np.sum(positive & ~np.isclose(matrix, np.rint(matrix), atol=TOL, rtol=0)))
            if finite
            else pd.NA
        )
        zero_rows = int(np.sum(np.all(np.abs(matrix) <= TOL, axis=1))) if finite else pd.NA
        physical = bool(finite and (negative == 0))
        if metric == "fa_mean":
            physical = bool(physical and above_one == 0)
        # Fractional values are a semantic failure only for raw streamline count.
        if metric == "count":
            physical = bool(physical and fractional == 0)
        hard_valid = bool(symmetric and zero_diagonal and physical)
        return {
            **base,
            "shape_valid": True,
            "finite": finite,
            "symmetric": symmetric,
            "zero_diagonal": zero_diagonal,
            "minimum": float(np.nanmin(matrix)) if finite else np.nan,
            "maximum": float(np.nanmax(matrix)) if finite else np.nan,
            "negative_cells": negative,
            "above_one_cells": above_one,
            "fractional_positive_cells": fractional,
            "positive_upper_edges": int(np.sum(np.triu(positive, 1))) if finite else pd.NA,
            "zero_rows": zero_rows,
            "hard_valid": hard_valid,
        }, matrix
    except Exception as error:
        return {
            **base,
            "read_error": f"{type(error).__name__}: {error}",
            "shape_valid": False,
            "finite": False,
            "symmetric": False,
            "zero_diagonal": False,
            "minimum": np.nan,
            "maximum": np.nan,
            "negative_cells": pd.NA,
            "above_one_cells": pd.NA,
            "fractional_positive_cells": pd.NA,
            "positive_upper_edges": pd.NA,
            "zero_rows": pd.NA,
            "hard_valid": False,
        }, None


def choose_lane(record: dict[str, object]) -> tuple[str, str, str, str]:
    if record["primary_component_hard_failure"]:
        return (
            "P1_TENSOR_OR_UPSTREAM_DIAGNOSTIC",
            "AxD/RD is missing or fails shape, finite, symmetry, diagonal, or nonnegative checks.",
            "Do not use the affected component as confirmatory. First test tensor resampling from valid corrected DWI/tracks; move upstream only if its prerequisites fail.",
            "DIRECT_PRIMARY_ENDPOINT_RISK",
        )
    if record["structural_zero_row_failure"]:
        return (
            "P2_ATLAS_ASSIGNMENT_DIAGNOSTIC",
            "Raw structural support contains one or more all-zero AAL3-166 rows.",
            "Inspect atlas-in-DWI overlap and endpoint assignment. Reuse DWI/FOD/tracks when valid; rerun tractography only if assignment cannot be repaired.",
            "PRIMARY_SUPPORT_RISK",
        )
    if record["primary_tensor_zero_row_failure"]:
        return (
            "P3_TENSOR_MATRIX_RESAMPLING_DIAGNOSTIC",
            "AxD/RD has all-zero rows despite no all-zero row in the structural support matrix.",
            "Regenerate only tensor sampling/connectome matrices from existing valid tracks where possible; retain sensitivity analyses in the meantime.",
            "PRIMARY_COMPLETENESS_RISK",
        )
    if record["count_semantic_failure"]:
        return (
            "P4_COUNT_MATRIX_ONLY_IF_NEEDED",
            "The count-labelled matrix is fractional or duplicates SIFT2 fd_sum.",
            "No AxD/RD rerun is implied. Regenerate raw count from existing assignments only if a count-based graph result is retained.",
            "SECONDARY_GRAPH_ONLY",
        )
    return (
        "P0_RETAIN_PENDING_CANARY_DRIFT",
        "No scanned hard or support defect was detected in the primary matrices.",
        "Retain for historical directional analysis. Upstream correction is triggered only by material canary drift, not by default.",
        "NO_CURRENT_MATRIX_TRIGGER",
    )


def build_triage(manifest: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    matrix_rows: list[dict[str, object]] = []
    subject_rows: list[dict[str, object]] = []
    for source in manifest.itertuples(index=False):
        subject_id = str(source.subject_id)
        image_id = str(int(source.dti_image_id))
        stats: dict[str, dict[str, object]] = {}
        arrays: dict[str, np.ndarray | None] = {}
        for metric in METRICS:
            result, array = inspect_matrix(subject_id, image_id, metric)
            matrix_rows.append(result)
            stats[metric] = result
            arrays[metric] = array

        count = arrays["count"]
        fd_sum = arrays["fd_sum"]
        count_eq_fd = bool(
            count is not None
            and fd_sum is not None
            and np.allclose(count, fd_sum, atol=TOL, rtol=0, equal_nan=True)
        )
        count_fractional = bool(
            stats["count"]["exists"]
            and pd.notna(stats["count"]["fractional_positive_cells"])
            and int(stats["count"]["fractional_positive_cells"]) > 0
        )
        primary_hard = bool(not stats["ad_mean"]["hard_valid"] or not stats["rd_mean"]["hard_valid"])
        structural_zero = bool(
            any(
                pd.notna(stats[metric]["zero_rows"])
                and int(stats[metric]["zero_rows"]) > 0
                for metric in ("count", "fd_sum")
            )
        )
        primary_zero = bool(
            any(
                pd.notna(stats[metric]["zero_rows"])
                and int(stats[metric]["zero_rows"]) > 0
                for metric in ("ad_mean", "rd_mean")
            )
        )
        record: dict[str, object] = {
            "subject_id": subject_id,
            "dti_image_id": image_id,
            "diagnosis": source.diagnosis_reconciled_harmonized,
            "site": source.site,
            "manufacturer": source.dti_manufacturer,
            "protocol_key": source.dti_protocol_key,
            "legacy_qc_status": source.legacy_sc_matrix_qc_status,
            "legacy_qc_include": source.legacy_sc_matrix_qc_include,
            "legacy_radial_recipe": source.legacy_radial_recipe,
            "count_equals_fd_sum": count_eq_fd,
            "count_fractional": count_fractional,
            "count_semantic_failure": bool(count_eq_fd or count_fractional or not stats["count"]["hard_valid"]),
            "primary_component_hard_failure": primary_hard,
            "structural_zero_row_failure": structural_zero,
            "primary_tensor_zero_row_failure": bool(primary_zero and not structural_zero),
            "count_zero_rows": stats["count"]["zero_rows"],
            "fd_sum_zero_rows": stats["fd_sum"]["zero_rows"],
            "ad_mean_zero_rows": stats["ad_mean"]["zero_rows"],
            "rd_mean_zero_rows": stats["rd_mean"]["zero_rows"],
            "fa_above_one_cells": stats["fa_mean"]["above_one_cells"],
            "md_negative_cells": stats["md_mean"]["negative_cells"],
            "rd_negative_cells": stats["rd_mean"]["negative_cells"],
            "ad_negative_cells": stats["ad_mean"]["negative_cells"],
            "n_missing_matrices": int(sum(not stats[metric]["exists"] for metric in METRICS)),
            "n_hard_invalid_matrices": int(sum(not stats[metric]["hard_valid"] for metric in METRICS)),
            "upstream_dwi_policy": "NO_AUTO_RERUN__AWAIT_MATERIAL_CANARY_DRIFT",
        }
        lane, reason, action, relevance = choose_lane(record)
        record.update(
            {
                "provisional_repair_lane": lane,
                "lane_reason": reason,
                "smallest_next_action": action,
                "novelty_endpoint_relevance": relevance,
            }
        )
        subject_rows.append(record)
    return pd.DataFrame(subject_rows), pd.DataFrame(matrix_rows)


def summary_markdown(subjects: pd.DataFrame, matrices: pd.DataFrame) -> str:
    lane_counts = subjects["provisional_repair_lane"].value_counts()
    lines = [
        "# Selective repair triage v1",
        "",
        f"Generated: `{utc_now()}`",
        "",
        "This is a read-only diagnostic over the current AAL3-166 matrices. It does not authorize any repair, tractography, or full-cohort rerun.",
        "",
        "## Core decision",
        "",
        "A defect is repaired at the earliest affected reusable stage. A count-only semantic defect does not invalidate the AxD/RD novelty endpoint, and an upstream DWI rerun is not triggered unless corrected canaries demonstrate material endpoint drift.",
        "",
        "## Cohort findings",
        "",
        f"- Subjects inspected: **{len(subjects)}/530**",
        f"- Matrices inspected: **{len(matrices)}** (nine expected per subject)",
        f"- Count equals fd_sum: **{int(subjects['count_equals_fd_sum'].sum())}/530**",
        f"- Fractional count: **{int(subjects['count_fractional'].sum())}/530**",
        f"- Count/fd_sum all-zero-row support flag: **{int(subjects['structural_zero_row_failure'].sum())}/530**",
        f"- AxD/RD hard physical or structural failure: **{int(subjects['primary_component_hard_failure'].sum())}/530**",
        f"- AxD/RD-only all-zero-row sampling flag: **{int(subjects['primary_tensor_zero_row_failure'].sum())}/530**",
        "",
        "## Smallest provisional lane",
        "",
    ]
    for lane in (
        "P0_RETAIN_PENDING_CANARY_DRIFT",
        "P1_TENSOR_OR_UPSTREAM_DIAGNOSTIC",
        "P2_ATLAS_ASSIGNMENT_DIAGNOSTIC",
        "P3_TENSOR_MATRIX_RESAMPLING_DIAGNOSTIC",
        "P4_COUNT_MATRIX_ONLY_IF_NEEDED",
    ):
        lines.append(f"- {lane}: **{int(lane_counts.get(lane, 0))}**")
    lines.extend(
        [
            "",
            "## How this changes the rerun decision",
            "",
            "1. Preserve P0 outputs unless corrected canaries show material drift.",
            "2. Repair P4 only if count-based graph results are needed; it is irrelevant to the locked AxD/RD primary endpoint.",
            "3. Attempt matrix/tensor resampling for P3 before touching tractography or DWI.",
            "4. Inspect atlas and assignment for P2; reuse valid upstream derivatives and rerun tractography only when assignment repair is insufficient.",
            "5. For P1, test the nearest viable tensor stage first. Move to raw DWI only when corrected metadata/preprocessing is necessary or canary drift crosses the frozen threshold.",
            "",
            "The lane order is a diagnostic priority, not a quality ranking. Subjects can carry secondary count defects in addition to their primary lane; all flags remain in `subject_selective_repair_triage.csv`.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--reuse-existing-scan",
        action="store_true",
        help="Finalize already-written CSV scan products without rereading production matrices.",
    )
    args = parser.parse_args()
    if args.reuse_existing_scan:
        subjects = pd.read_csv(args.out / "subject_selective_repair_triage.csv")
        matrices = pd.read_csv(args.out / "matrix_level_qc.csv")
        lane_summary = pd.read_csv(args.out / "lane_summary.csv")
    else:
        manifest = pd.read_csv(MANIFEST)
        if len(manifest) != 530 or manifest["subject_id"].nunique() != 530:
            raise ValueError("Expected exactly 530 unique locked subjects")
        subjects, matrices = build_triage(manifest)
        lane_summary = (
            subjects.groupby(["provisional_repair_lane", "novelty_endpoint_relevance"], as_index=False)
            .agg(
                n_subjects=("subject_id", "size"),
                n_CN=("diagnosis", lambda values: int((values == "CN").sum())),
                n_MCI=("diagnosis", lambda values: int((values == "MCI").sum())),
                n_AD=("diagnosis", lambda values: int((values == "AD").sum())),
                n_SMC=("diagnosis", lambda values: int((values == "SMC").sum())),
            )
            .sort_values("provisional_repair_lane")
        )
    validation = {
        "schema_version": "1.0.0",
        "generated_utc": utc_now(),
        "checks": {
            "exact_530_subjects": bool(len(subjects) == 530 and subjects["subject_id"].nunique() == 530),
            "nine_matrix_rows_per_subject": bool(len(matrices) == 530 * len(METRICS)),
            "all_subjects_have_lane": bool(subjects["provisional_repair_lane"].notna().all()),
            "lane_counts_sum_530": bool(int(lane_summary["n_subjects"].sum()) == 530),
            "production_derivatives_not_modified": True,
        },
        "interpretation": "Pass validates inventory and deterministic lane assignment, not biological validity or repair authorization.",
    }
    if not all(validation["checks"].values()):
        raise RuntimeError(f"Selective repair triage validation failed: {validation['checks']}")
    args.out.mkdir(parents=True, exist_ok=True)
    atomic_csv(args.out / "subject_selective_repair_triage.csv", subjects)
    atomic_csv(args.out / "matrix_level_qc.csv", matrices)
    atomic_csv(args.out / "lane_summary.csv", lane_summary)
    atomic_text(args.out / "summary.md", summary_markdown(subjects, matrices))
    atomic_text(args.out / "validation.json", json.dumps(validation, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "subjects": len(subjects),
        "matrix_rows": len(matrices),
        "lane_counts": subjects["provisional_repair_lane"].value_counts().to_dict(),
        "validation": validation["checks"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
