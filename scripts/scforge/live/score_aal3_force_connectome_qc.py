#!/usr/bin/env python3
"""Score AAL3 forced-connectome rescue results against concrete QC goals.

This script is intentionally non-destructive. It reads a scratch batch run and
writes route-level decisions that can be used to decide whether the tested route
is worth visual QC / promotion, or whether the subject needs another AAL3 route.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import subprocess
import time
from pathlib import Path
from typing import Any


RUN_PARENT = Path("/data/derivatives/qc/sc_matrix_qc/aal3_force_connectome_ad_batch")
ROOT = Path("/home/ec2-user/exp")
DERIV = ROOT / "data" / "derivatives"
AAL3_LABELS = ROOT / "atlas" / "AAL" / "AAL3_labels.csv"

EXPECTED_AAL3_LABELS = 166

PROMOTE = {
    "min_labels": 160,
    "max_zero_rows": 15,
    "min_density": 0.10,
    "min_density_delta": 0.0,
    "max_zero_delta": 0,
    "min_both_assigned_fraction": 0.85,
    "max_one_unassigned_fraction": 0.15,
    "max_both_unassigned_fraction": 0.05,
}

REVIEW = {
    "min_labels": 150,
    "max_zero_rows": 25,
    "min_density": 0.05,
    "min_density_delta": -0.02,
    "min_both_assigned_fraction": 0.75,
    "max_one_unassigned_fraction": 0.25,
    "max_both_unassigned_fraction": 0.10,
}

# Interim production-review is intentionally looser than full QC. It supports
# user-requested production refreshes whenever density improves over production,
# while keeping the subject open for better routes because label survival,
# zero-row, assignment, and visual QC may not be fully solved.
INTERIM_REVIEW_PROMOTION = {
    "min_density_delta": 0.0,
}


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def acquire_run_lock(run_root: Path):
    """Prevent concurrent scorers from rewriting one route's QC artifacts."""
    key = hashlib.sha1(str(run_root.resolve()).encode("utf-8")).hexdigest()[:16]
    lock_path = Path("/tmp") / f"aal3_score_{key}.lock"
    handle = lock_path.open("w")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        return None
    handle.write(f"{utc()} {run_root}\n")
    handle.flush()
    return handle


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


def result_rows(run_root: Path) -> list[dict[str, Any]]:
    """Load the best available per-subject rows for a run.

    Resumed batches can temporarily have a partial batch_results.csv while the
    durable per-subject done.json markers still represent completed subjects.
    Merge both sources by sid so scoring is not accidentally narrowed during a
    resume/fill-in run.
    """
    by_sid: dict[str, dict[str, Any]] = {}
    for row in read_csv(run_root / "batch_results.csv"):
        sid = str(row.get("sid") or "").strip()
        if sid:
            by_sid[sid] = dict(row)
    subjects_dir = run_root / "subjects"
    if subjects_dir.exists():
        for marker in sorted(subjects_dir.glob("*/done.json")):
            payload = read_json(marker)
            sid = str(payload.get("sid") or marker.parent.name).strip()
            if sid:
                by_sid[sid] = payload
    return list(by_sid.values())


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
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
        return float(value)
    except Exception:
        return default


def inum(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return default


def aal3_valid_labels() -> list[int]:
    try:
        rows = read_csv(AAL3_LABELS)
        labels = sorted({int(float(row["atlas_value"])) for row in rows if row.get("atlas_value")})
        if len(labels) >= 150:
            return labels
    except Exception:
        pass
    return list(range(1, EXPECTED_AAL3_LABELS + 1))


def latest_run_root() -> Path:
    latest = RUN_PARENT / "latest_tag.txt"
    if latest.exists() and latest.read_text(encoding="utf-8").strip():
        return RUN_PARENT / latest.read_text(encoding="utf-8").strip()
    runs = sorted((p for p in RUN_PARENT.glob("*") if p.is_dir()), key=lambda p: p.stat().st_mtime, reverse=True)
    if not runs:
        raise SystemExit(f"No run roots found under {RUN_PARENT}")
    return runs[0]


def label_stage(probe_root: Path, stage: str) -> dict[str, str]:
    rows = [row for row in read_csv(probe_root / "label_survival_summary.csv") if row.get("stage") == stage]
    return rows[-1] if rows else {}


def matrix_row(probe_root: Path, candidate: str, variant: str, metric: str = "fd_sum") -> dict[str, str]:
    rows = [
        row
        for row in read_csv(probe_root / "matrix_candidate_summary.csv")
        if row.get("candidate") == candidate and row.get("variant") == variant and row.get("metric") == metric
    ]
    return rows[-1] if rows else {}


def assignment_path(probe_root: Path, variant: str) -> Path:
    return probe_root / "connectomes" / f"assignments_source_contract__{variant}__fd_sum.csv"


def count_matrix_metrics(path: Path, valid_labels: list[int]) -> dict[str, Any]:
    """Count-density from a raw streamline-count connectome matrix.

    Density here means ROI pairs with at least one assigned streamline divided
    by possible valid AAL3 ROI pairs. This is separate from fd_sum/SIFT2
    weights, which remain the production edge weight.
    """
    if not path.is_file() or path.stat().st_size <= 0:
        return {}
    try:
        import numpy as np

        mat = np.loadtxt(path, delimiter=",")
        if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
            return {}
        present = [(out_idx, label - 1) for out_idx, label in enumerate(valid_labels) if 0 <= label - 1 < mat.shape[0]]
        if len(valid_labels) < 2:
            return {}
        sub = np.zeros((len(valid_labels), len(valid_labels)), dtype=float)
        clean = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)
        for out_i, src_i in present:
            for out_j, src_j in present:
                sub[out_i, out_j] = clean[src_i, src_j]
        upper = np.triu(np.ones_like(sub, dtype=bool), 1)
        edge_values = sub[upper]
        nonzero = int((np.abs(edge_values) > 0).sum())
        possible = int(upper.sum())
        row_signal = np.abs(sub).sum(axis=0) + np.abs(sub).sum(axis=1)
        return {
            "tract_count_density": nonzero / possible if possible else "",
            "tract_count_zero_rows": int((row_signal == 0).sum()),
            "tract_count_nonzero_edges": nonzero,
            "tract_count_possible_edges": possible,
            "tract_count_total_streamlines": float(edge_values[edge_values > 0].sum()) if edge_values.size else 0.0,
            "tract_count_metric_source": str(path),
        }
    except Exception:
        return {}


def assignment_count_metrics(path: Path, valid_labels: list[int]) -> dict[str, Any]:
    """Count-density directly from tck2connectome endpoint assignments."""
    if not path.exists() or path.stat().st_size <= 0:
        return {}
    valid = set(int(label) for label in valid_labels)
    edges: set[tuple[int, int]] = set()
    incident: set[int] = set()
    valid_pair_rows = 0
    total_assignment_rows = 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(",", " ").split()
            if len(parts) < 2:
                continue
            try:
                a = int(float(parts[0]))
                b = int(float(parts[1]))
            except Exception:
                continue
            total_assignment_rows += 1
            if a in valid and b in valid and a != b:
                lo, hi = sorted((a, b))
                edges.add((lo, hi))
                incident.add(lo)
                incident.add(hi)
                valid_pair_rows += 1
    possible = len(valid_labels) * (len(valid_labels) - 1) // 2
    return {
        "tract_count_density": len(edges) / possible if possible else "",
        "tract_count_zero_rows": len(valid_labels) - len(incident),
        "tract_count_nonzero_edges": len(edges),
        "tract_count_possible_edges": possible,
        "tract_count_total_streamlines": valid_pair_rows,
        "tract_count_assignment_rows": total_assignment_rows,
        "tract_count_metric_source": str(path),
    }


def first_finite(*values: Any, default: float = float("nan")) -> float:
    for value in values:
        out = fnum(value, float("nan"))
        if math.isfinite(out):
            return out
    return default


def first_int(*values: Any, default: int = 9999) -> int:
    for value in values:
        try:
            if value in {"", None}:
                continue
            return int(float(value))
        except Exception:
            continue
    return default


def assignment_qc(path: Path) -> dict[str, Any]:
    total = 0
    both_assigned = 0
    one_unassigned = 0
    both_unassigned = 0
    malformed = 0
    if not path.exists() or path.stat().st_size <= 0:
        return {
            "assignment_file": str(path),
            "assignment_rows": 0,
            "both_assigned": 0,
            "one_unassigned": 0,
            "both_unassigned": 0,
            "malformed_assignment_rows": 0,
            "both_assigned_fraction": "",
            "one_unassigned_fraction": "",
            "both_unassigned_fraction": "",
        }
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(",", " ").split()
            if len(parts) < 2:
                malformed += 1
                continue
            try:
                a = int(float(parts[0]))
                b = int(float(parts[1]))
            except Exception:
                malformed += 1
                continue
            total += 1
            if a > 0 and b > 0:
                both_assigned += 1
            elif a <= 0 and b <= 0:
                both_unassigned += 1
            else:
                one_unassigned += 1
    frac = lambda x: x / total if total else ""
    return {
        "assignment_file": str(path),
        "assignment_rows": total,
        "both_assigned": both_assigned,
        "one_unassigned": one_unassigned,
        "both_unassigned": both_unassigned,
        "malformed_assignment_rows": malformed,
        "both_assigned_fraction": frac(both_assigned),
        "one_unassigned_fraction": frac(one_unassigned),
        "both_unassigned_fraction": frac(both_unassigned),
    }


def threshold_reasons(row: dict[str, Any], thresholds: dict[str, float], strict_delta: bool) -> list[str]:
    reasons: list[str] = []
    density = first_finite(row.get("qc_density"), row.get("best_density"), default=0.0)
    density_delta = first_finite(row.get("qc_density_delta"), row.get("density_delta"), default=0.0)
    zero_rows = first_int(row.get("qc_zero_rows"), row.get("best_zero_rows"), default=9999)
    zero_rows_delta = first_int(row.get("qc_zero_rows_delta"), row.get("zero_rows_delta"), default=9999)
    basis = str(row.get("qc_density_basis") or "fd_sum weighted nonzero edges")
    if row["source_labels"] < thresholds["min_labels"]:
        reasons.append(f"labels {row['source_labels']}/{EXPECTED_AAL3_LABELS} < {thresholds['min_labels']}")
    if zero_rows > thresholds["max_zero_rows"]:
        reasons.append(f"{basis} zero rows {zero_rows} > {thresholds['max_zero_rows']}")
    if density < thresholds["min_density"]:
        reasons.append(f"{basis} density {density:.4f} < {thresholds['min_density']:.2f}")
    if density_delta < thresholds["min_density_delta"]:
        reasons.append(f"{basis} density delta {density_delta:.4f} < {thresholds['min_density_delta']:.2f}")
    if strict_delta and zero_rows_delta > thresholds["max_zero_delta"]:
        reasons.append(f"{basis} zero rows worsened by {zero_rows_delta}")
    both = row.get("both_assigned_fraction")
    one = row.get("one_unassigned_fraction")
    neither = row.get("both_unassigned_fraction")
    if both in {"", None}:
        reasons.append("assignment QC missing")
    elif both < thresholds["min_both_assigned_fraction"]:
        reasons.append(f"both-assigned {both:.3f} < {thresholds['min_both_assigned_fraction']:.2f}")
    if one not in {"", None} and one > thresholds["max_one_unassigned_fraction"]:
        reasons.append(f"one-unassigned {one:.3f} > {thresholds['max_one_unassigned_fraction']:.2f}")
    if neither not in {"", None} and neither > thresholds["max_both_unassigned_fraction"]:
        reasons.append(f"both-unassigned {neither:.3f} > {thresholds['max_both_unassigned_fraction']:.2f}")
    return reasons


def interim_review_reasons(row: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    density_delta = first_finite(row.get("qc_density_delta"), row.get("density_delta"), default=0.0)
    basis = str(row.get("qc_density_basis") or "fd_sum weighted nonzero edges")
    if density_delta <= INTERIM_REVIEW_PROMOTION["min_density_delta"]:
        reasons.append(
            f"{basis} density delta {density_delta:.4f} <= {INTERIM_REVIEW_PROMOTION['min_density_delta']:.2f}"
        )
    return reasons


def interim_review_flags(row: dict[str, Any]) -> list[str]:
    flags: list[str] = []
    density = first_finite(row.get("qc_density"), row.get("best_density"), default=0.0)
    zero_rows = first_int(row.get("qc_zero_rows"), row.get("best_zero_rows"), default=9999)
    zero_rows_delta = first_int(row.get("qc_zero_rows_delta"), row.get("zero_rows_delta"), default=9999)
    basis = str(row.get("qc_density_basis") or "fd_sum weighted nonzero edges")
    if row["source_labels"] < PROMOTE["min_labels"]:
        flags.append(f"labels {row['source_labels']}/{EXPECTED_AAL3_LABELS} < {PROMOTE['min_labels']}")
    if density < PROMOTE["min_density"]:
        flags.append(f"{basis} density {density:.4f} < {PROMOTE['min_density']:.2f}")
    if zero_rows > PROMOTE["max_zero_rows"]:
        flags.append(f"{basis} zero rows {zero_rows} > {PROMOTE['max_zero_rows']}")
    if zero_rows_delta > 0:
        flags.append(f"{basis} zero rows worsened by {zero_rows_delta}")
    assignment_rows = inum(row.get("assignment_rows"), 0)
    both = fnum(row.get("both_assigned_fraction"), float("nan"))
    one = fnum(row.get("one_unassigned_fraction"), float("nan"))
    neither = fnum(row.get("both_unassigned_fraction"), float("nan"))
    if assignment_rows <= 0 or not math.isfinite(both):
        flags.append("assignment QC missing")
    elif both < PROMOTE["min_both_assigned_fraction"]:
        flags.append(
            f"both-endpoint assignment {both:.3f} < {PROMOTE['min_both_assigned_fraction']:.2f}"
        )
    if math.isfinite(one) and one > PROMOTE["max_one_unassigned_fraction"]:
        flags.append(
            f"one-end-unassigned {one:.3f} > {PROMOTE['max_one_unassigned_fraction']:.2f}"
        )
    if math.isfinite(neither) and neither > PROMOTE["max_both_unassigned_fraction"]:
        flags.append(
            f"both-unassigned {neither:.3f} > {PROMOTE['max_both_unassigned_fraction']:.2f}"
        )
    return flags


def production_numeric_ok(row: dict[str, Any]) -> bool:
    baseline_density = first_finite(row.get("baseline_tract_count_density"), row.get("baseline_density"), default=0.0)
    baseline_zero_rows = first_int(row.get("baseline_tract_count_zero_rows"), row.get("baseline_zero_rows"), default=9999)
    return (
        row["production_labels"] >= PROMOTE["min_labels"]
        and baseline_zero_rows <= PROMOTE["max_zero_rows"]
        and baseline_density >= PROMOTE["min_density"]
    )


def score_route10_diagnostic_row(result: dict[str, str]) -> dict[str, Any]:
    """Represent endpoint-overlay triage as a next-route decision.

    Route 10 is diagnostic only: it creates endpoint/atlas overlays and assigns
    each subject to the next repair lane. It does not create a replacement
    connectome/parcellation pair, so it must not be counted as a production
    promotion candidate by the generic density-improvement policy.
    """
    density = fnum(result.get("best_density"), 0.0)
    density_delta = fnum(result.get("density_delta"), 0.0)
    zero_rows = inum(result.get("best_zero_rows"), 9999)
    zero_delta = inum(result.get("zero_rows_delta"), 9999)
    next_lane = result.get("next_lane", "")
    next_reason = result.get("next_lane_reason", "")
    route10_decision = result.get("route10_decision", "")
    issue_class = result.get("issue_class", "")
    details = [part for part in [route10_decision, next_lane, issue_class, next_reason] if part]
    return {
        "sid": result.get("sid", ""),
        "status": result.get("status", ""),
        "probe_root": "",
        "tested_route": "Route 10 endpoint-overlay triage",
        "best_candidate": result.get("parcellation", ""),
        "best_variant": "",
        "production_labels": 0,
        "source_labels": inum(result.get("source_labels"), 0),
        "source_label_survival": "",
        "missing_source_labels": "",
        "baseline_density": density - density_delta,
        "best_density": density,
        "density_delta": density_delta,
        "baseline_zero_rows": zero_rows - zero_delta,
        "best_zero_rows": zero_rows,
        "zero_rows_delta": zero_delta,
        "qc_density_basis": "fd_sum weighted nonzero edges",
        "qc_density": density,
        "qc_density_delta": density_delta,
        "qc_zero_rows": zero_rows,
        "qc_zero_rows_delta": zero_delta,
        "visual_overlay_status": "diagnostic_overlay_generated" if result.get("atlas_overlay_png") else "pending_manual",
        "assignment_file": "",
        "assignment_rows": "",
        "both_assigned": "",
        "one_unassigned": "",
        "both_unassigned": "",
        "malformed_assignment_rows": "",
        "both_assigned_fraction": fnum(result.get("both_assigned_fraction"), float("nan")),
        "one_unassigned_fraction": fnum(result.get("one_unassigned_fraction"), float("nan")),
        "both_unassigned_fraction": fnum(result.get("both_unassigned_fraction"), float("nan")),
        "route10_decision": route10_decision,
        "next_lane": next_lane,
        "issue_class": issue_class,
        "endpoint_inside_label_fraction": fnum(result.get("endpoint_inside_label_fraction"), float("nan")),
        "atlas_overlay_png": result.get("atlas_overlay_png", ""),
        "endpoint_overlay_png": result.get("endpoint_overlay_png", ""),
        "candidate_parc_path": result.get("parcellation", ""),
        "candidate_matrix_path": "",
        "qc_decision": "TRY_ANOTHER_ROUTE",
        "qc_reasons": "diagnostic-only route; " + "; ".join(details),
        "final_qc_reasons": "route did not generate a production replacement connectome",
        "review_qc_reasons": "",
        "final_qc_status": "not_solved_try_another_route",
    }


def score_row(result: dict[str, str], compute_assignments: bool) -> dict[str, Any]:
    if "route10_decision" in result or "next_lane" in result:
        return score_route10_diagnostic_row(result)

    probe_root = Path(result.get("probe_root") or "")
    variant = result.get("best_variant") or ""
    candidate = result.get("best_candidate", "AAL3_source_contract")
    source_stage = result.get("best_label_stage") or result.get("source_label_stage") or (
        "AAL3_source_contract_B0" if "best_candidate" not in result else ""
    )
    assignment_file = Path(result.get("assignment_file") or "") if result.get("assignment_file") else None
    prod_label = label_stage(probe_root, "production_AAL_b0")
    source_label = label_stage(probe_root, source_stage)
    prod_matrix = matrix_row(probe_root, "production_AAL3", "production")
    prod_count_matrix = matrix_row(probe_root, "production_AAL3", "production", "count")
    best_matrix = matrix_row(probe_root, candidate, variant) if variant else {}
    valid_labels = aal3_valid_labels()
    if compute_assignments and variant:
        assign = assignment_qc(assignment_file if assignment_file else assignment_path(probe_root, variant))
    else:
        assign = {}
    baseline_count_path = Path(prod_count_matrix.get("matrix") or "")
    if not baseline_count_path.is_file():
        baseline_count_path = DERIV / "connectomes" / f"SC_AAL_{result.get('sid', '')}_count.csv"
    baseline_count = count_matrix_metrics(baseline_count_path, valid_labels)
    candidate_count: dict[str, Any] = {}
    if compute_assignments and variant:
        candidate_count = assignment_count_metrics(assignment_file if assignment_file else assignment_path(probe_root, variant), valid_labels)
    if not candidate_count:
        best_count_matrix = matrix_row(probe_root, candidate, variant, "count") if variant else {}
        best_count_path = Path(best_count_matrix.get("matrix") or "")
        candidate_count = count_matrix_metrics(best_count_path, valid_labels) if best_count_path.is_file() else {}

    baseline_fd_density = fnum(prod_matrix.get("valid_density"), fnum(result.get("baseline_density"), 0.0))
    best_fd_density = fnum(best_matrix.get("valid_density"), fnum(result.get("best_density"), 0.0))
    baseline_fd_zero_rows = inum(prod_matrix.get("valid_zero_rows"), inum(result.get("baseline_zero_rows"), 9999))
    best_fd_zero_rows = inum(best_matrix.get("valid_zero_rows"), inum(result.get("best_zero_rows"), 9999))
    baseline_count_density = first_finite(baseline_count.get("tract_count_density"), default=float("nan"))
    best_count_density = first_finite(candidate_count.get("tract_count_density"), default=float("nan"))
    baseline_count_zero_rows = first_int(baseline_count.get("tract_count_zero_rows"), default=9999)
    best_count_zero_rows = first_int(candidate_count.get("tract_count_zero_rows"), default=9999)
    has_count_qc = math.isfinite(baseline_count_density) and math.isfinite(best_count_density)
    row: dict[str, Any] = {
        "sid": result.get("sid", ""),
        "status": result.get("status", ""),
        "probe_root": str(probe_root),
        "tested_route": result.get("tested_route") or source_stage,
        "best_candidate": candidate,
        "best_variant": variant,
        "production_labels": inum(prod_label.get("n_labels"), 0),
        "source_labels": inum(source_label.get("n_labels"), inum(result.get("source_labels"), 0)),
        "source_label_survival": fnum(source_label.get("expected_label_survival_fraction"), fnum(result.get("source_label_survival"), 0.0)),
        "missing_source_labels": inum(source_label.get("missing_expected_count"), EXPECTED_AAL3_LABELS),
        "baseline_density": baseline_fd_density,
        "best_density": best_fd_density,
        "density_delta": fnum(result.get("density_delta"), 0.0),
        "baseline_zero_rows": baseline_fd_zero_rows,
        "best_zero_rows": best_fd_zero_rows,
        "zero_rows_delta": inum(result.get("zero_rows_delta"), 9999),
        "baseline_tract_count_density": baseline_count.get("tract_count_density", ""),
        "best_tract_count_density": candidate_count.get("tract_count_density", ""),
        "tract_count_density_delta": (best_count_density - baseline_count_density) if has_count_qc else "",
        "baseline_tract_count_zero_rows": baseline_count.get("tract_count_zero_rows", ""),
        "best_tract_count_zero_rows": candidate_count.get("tract_count_zero_rows", ""),
        "tract_count_zero_rows_delta": (best_count_zero_rows - baseline_count_zero_rows) if has_count_qc else "",
        "baseline_tract_count_nonzero_edges": baseline_count.get("tract_count_nonzero_edges", ""),
        "best_tract_count_nonzero_edges": candidate_count.get("tract_count_nonzero_edges", ""),
        "baseline_tract_count_total_streamlines": baseline_count.get("tract_count_total_streamlines", ""),
        "best_tract_count_total_streamlines": candidate_count.get("tract_count_total_streamlines", ""),
        "tract_count_possible_edges": candidate_count.get("tract_count_possible_edges", baseline_count.get("tract_count_possible_edges", "")),
        "tract_count_metric_source": candidate_count.get("tract_count_metric_source", ""),
        "qc_density_basis": "tract-count nonzero ROI pairs" if has_count_qc else "fd_sum weighted nonzero edges",
        "qc_density": best_count_density if has_count_qc else best_fd_density,
        "qc_density_delta": (best_count_density - baseline_count_density) if has_count_qc else fnum(result.get("density_delta"), 0.0),
        "qc_zero_rows": best_count_zero_rows if has_count_qc else best_fd_zero_rows,
        "qc_zero_rows_delta": (best_count_zero_rows - baseline_count_zero_rows) if has_count_qc else inum(result.get("zero_rows_delta"), 9999),
        "visual_overlay_status": "pending_manual",
    }
    row.update(assign)
    row["interim_review_flags"] = ""
    if result.get("status") != "ok":
        row["qc_decision"] = "TRY_ANOTHER_ROUTE"
        row["qc_reasons"] = "probe did not finish with an ok candidate"
        row["final_qc_reasons"] = "probe did not finish with an ok candidate"
        row["review_qc_reasons"] = ""
        return row

    promote_reasons = threshold_reasons(row, PROMOTE, strict_delta=True)
    row["final_qc_reasons"] = "; ".join(promote_reasons)
    if not promote_reasons:
        row["qc_decision"] = "PROMOTE_CANDIDATE_PENDING_VISUAL"
        row["qc_reasons"] = "numeric QC passed; visual overlay still required"
        row["final_qc_reasons"] = ""
        row["review_qc_reasons"] = ""
        return row

    interim_reasons = interim_review_reasons(row)
    if not interim_reasons:
        flags = interim_review_flags(row)
        row["qc_decision"] = "INTERIM_REVIEW_PROMOTION_PENDING_VISUAL"
        row["qc_reasons"] = (
            "density-improved interim production-refresh candidate; "
            "not final QC solved; eligible for explicit interim production overwrite while routes continue"
        )
        row["review_qc_reasons"] = ""
        row["final_qc_status"] = "not_solved_keep_trying_routes"
        row["interim_review_flags"] = "; ".join(flags)
        return row

    if production_numeric_ok(row):
        row["qc_decision"] = "KEEP_PRODUCTION_NUMERIC_OK"
        row["qc_reasons"] = "current production already passes label/density/zero-row numeric QC; tested candidate is not better"
        row["review_qc_reasons"] = "; ".join(interim_reasons)
        row["final_qc_status"] = "current_production_numeric_ok_assignment_visual_not_reaudited"
        return row

    review_reasons = threshold_reasons(row, REVIEW, strict_delta=False)
    row["review_qc_reasons"] = "; ".join(review_reasons)
    if not review_reasons:
        row["qc_decision"] = "REVIEW_ROUTE"
        row["qc_reasons"] = "final QC failed: " + "; ".join(promote_reasons)
        row["final_qc_status"] = "not_solved_keep_trying_routes"
        return row

    row["qc_decision"] = "TRY_ANOTHER_ROUTE"
    row["qc_reasons"] = "final QC failed: " + "; ".join(promote_reasons)
    row["final_qc_status"] = "not_solved_try_another_route"
    return row


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    decisions: dict[str, int] = {}
    variants: dict[str, int] = {}
    for row in rows:
        decisions[row.get("qc_decision", "")] = decisions.get(row.get("qc_decision", ""), 0) + 1
        variants[row.get("best_variant", "")] = variants.get(row.get("best_variant", ""), 0) + 1
    return {
        "generated_utc": utc(),
        "rows": len(rows),
        "decisions": decisions,
        "best_variants": variants,
        "thresholds": {"promote": PROMOTE, "review": REVIEW, "expected_aal3_labels": EXPECTED_AAL3_LABELS},
        "density_policy": {
            "qc_density_basis": "tract-count nonzero ROI pairs when assignment/count data are available; fd_sum weighted nonzero edges fallback otherwise",
            "production_edge_weight": "fd_sum/SIFT2 remains the promoted structural-connectome edge weight",
            "tract_count_denominator": EXPECTED_AAL3_LABELS * (EXPECTED_AAL3_LABELS - 1) // 2,
        },
        "interim_review_promotion_policy": INTERIM_REVIEW_PROMOTION,
    }


def maybe_refresh_monitor_qc(run_root: Path) -> None:
    try:
        subprocess.run(
            ["/home/ec2-user/exp/scripts/scforge/live/watch_aal3_force_connectome_ad_batch.sh"],
            cwd="/home/ec2-user/exp",
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except Exception:
        pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--no-assignment-qc", action="store_true", help="Skip reading 3M-row assignment files.")
    args = parser.parse_args()

    run_root = args.run_root or latest_run_root()
    lock_handle = acquire_run_lock(run_root)
    if lock_handle is None:
        print(f"scorer already running for {run_root}; skipping duplicate launch")
        return 0

    results = result_rows(run_root)
    if not results:
        raise SystemExit(f"No batch results found in {run_root}")
    rows = [score_row(row, compute_assignments=not args.no_assignment_qc) for row in results]
    write_csv(run_root / "route_qc_decisions.csv", rows)
    payload = summarize(rows)
    payload["run_root"] = str(run_root)
    write_json(run_root / "route_qc_summary.json", payload)
    maybe_refresh_monitor_qc(run_root)
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"wrote {run_root / 'route_qc_decisions.csv'}")
    print(f"wrote {run_root / 'route_qc_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
