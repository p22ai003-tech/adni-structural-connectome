#!/usr/bin/env python3
"""Read-only validation gate for repaired structural connectome matrices.

This script does not delete, move, or regenerate anything.  It checks whether
the outputs from a repair lane are complete enough to be considered for
downstream analysis.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    import nibabel as nib
except Exception:  # pragma: no cover - reported in output if unavailable
    nib = None


ROOT = Path("/home/ec2-user/exp")
DERIV = ROOT / "data" / "derivatives"
QC_ROOT = DERIV / "qc" / "sc_matrix_qc"
LABEL_CSV = ROOT / "atlas" / "AAL" / "AAL3_labels.csv"
WEIGHTS = (
    "count",
    "fd_sum",
    "len_mean",
    "invlen_mean",
    "fa_mean",
    "md_mean",
    "ad_mean",
    "rd_mean",
    "count_invnodevol",
)
PRIMARY_ZERO_ROW_WEIGHTS = ("count", "fd_sum")
ALL_METRICS = set(WEIGHTS)


@dataclass(frozen=True)
class Atlas:
    valid_nodes: set[int]
    expected_gaps: set[int]
    max_node: int


def read_sid_list(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(path)
    sids = [line.strip() for line in path.read_text(encoding="ascii").splitlines() if line.strip()]
    return sorted(dict.fromkeys(sids))


def load_atlas(path: Path) -> Atlas:
    valid: set[int] = set()
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw = row.get("node") or row.get("label") or ""
            try:
                valid.add(int(float(raw)))
            except ValueError:
                continue
    if not valid:
        raise ValueError(f"No AAL labels read from {path}")
    max_node = max(valid)
    return Atlas(valid_nodes=valid, expected_gaps=set(range(1, max_node + 1)).difference(valid), max_node=max_node)


def _safe_float(value: object) -> float:
    try:
        out = float(value)
    except Exception:
        return math.nan
    return out


def read_matrix(path: Path) -> np.ndarray:
    return np.genfromtxt(path, delimiter=",", dtype=float)


def matrix_checks(path: Path, atlas: Atlas) -> dict[str, object]:
    row: dict[str, object] = {
        "exists": int(path.exists()),
        "size_bytes": path.stat().st_size if path.exists() else 0,
        "read_ok": 0,
        "read_error": "",
        "n_rows": math.nan,
        "n_cols": math.nan,
        "finite_fraction": math.nan,
        "symmetry_max_abs": math.nan,
        "diagonal_abs_sum": math.nan,
        "nonzero_upper_edges": math.nan,
        "density": math.nan,
        "n_unexpected_zero_rows": math.nan,
        "unexpected_zero_rows": "",
    }
    if not path.exists() or path.stat().st_size <= 0:
        row["read_error"] = "missing_or_empty"
        return row
    try:
        matrix = read_matrix(path)
        if matrix.ndim != 2:
            raise ValueError(f"matrix ndim={matrix.ndim}, expected 2")
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError(f"matrix shape={matrix.shape}, expected square")
        n = int(matrix.shape[0])
        clean = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
        valid_nodes = sorted(node for node in atlas.valid_nodes if 1 <= node <= n)
        row_abs = np.abs(clean).sum(axis=1)
        zero_rows = {idx + 1 for idx, value in enumerate(row_abs) if value == 0}
        unexpected = sorted(zero_rows.intersection(valid_nodes))
        upper = np.triu(np.ones((n, n), dtype=bool), 1)
        nonzero_upper = int((np.abs(clean) > 0)[upper].sum())
        possible_upper = int(n * (n - 1) / 2)
        row.update(
            {
                "read_ok": 1,
                "n_rows": n,
                "n_cols": n,
                "finite_fraction": float(np.isfinite(matrix).mean()) if matrix.size else 0.0,
                "symmetry_max_abs": float(np.max(np.abs(clean - clean.T))) if clean.size else math.nan,
                "diagonal_abs_sum": float(np.abs(np.diag(clean)).sum()) if clean.size else math.nan,
                "nonzero_upper_edges": nonzero_upper,
                "density": nonzero_upper / possible_upper if possible_upper else math.nan,
                "n_unexpected_zero_rows": len(unexpected),
                "unexpected_zero_rows": ";".join(str(x) for x in unexpected),
            }
        )
    except Exception as exc:
        row["read_error"] = f"{type(exc).__name__}: {exc}"
    return row


def validate_all_csv(path: Path) -> dict[str, object]:
    out: dict[str, object] = {
        "sc_all_exists": int(path.exists()),
        "sc_all_size_bytes": path.stat().st_size if path.exists() else 0,
        "sc_all_read_ok": 0,
        "sc_all_rows": 0,
        "sc_all_metrics": "",
        "sc_all_missing_metrics": "",
        "sc_all_finite_fraction": math.nan,
        "sc_all_nonzero_values": 0,
        "sc_all_read_error": "",
    }
    if not path.exists() or path.stat().st_size <= 0:
        out["sc_all_read_error"] = "missing_or_empty"
        return out
    try:
        metrics: set[str] = set()
        rows = 0
        finite = 0
        nonzero = 0
        with path.open(newline="", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            required = {"i", "j", "metric", "value"}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(f"missing required columns; got {reader.fieldnames}")
            for rec in reader:
                rows += 1
                metrics.add(str(rec.get("metric", "")).strip())
                value = _safe_float(rec.get("value"))
                if math.isfinite(value):
                    finite += 1
                if math.isfinite(value) and value != 0:
                    nonzero += 1
        missing = sorted(ALL_METRICS.difference(metrics))
        out.update(
            {
                "sc_all_read_ok": 1,
                "sc_all_rows": rows,
                "sc_all_metrics": ";".join(sorted(metrics)),
                "sc_all_missing_metrics": ";".join(missing),
                "sc_all_finite_fraction": finite / rows if rows else 0.0,
                "sc_all_nonzero_values": nonzero,
            }
        )
    except Exception as exc:
        out["sc_all_read_error"] = f"{type(exc).__name__}: {exc}"
    return out


def aal_label_count(path: Path, atlas: Atlas) -> dict[str, object]:
    out = {
        "aal_exists": int(path.exists()),
        "aal_size_bytes": path.stat().st_size if path.exists() else 0,
        "aal_read_ok": 0,
        "aal_shape": "",
        "aal_valid_labels_present": math.nan,
        "aal_read_error": "",
    }
    if not path.exists() or path.stat().st_size <= 0:
        out["aal_read_error"] = "missing_or_empty"
        return out
    if nib is None:
        out["aal_read_error"] = "nibabel_not_available"
        return out
    try:
        img = nib.load(str(path))
        data = np.asanyarray(img.dataobj)
        present = {int(x) for x in np.unique(data) if int(x) in atlas.valid_nodes}
        out.update(
            {
                "aal_read_ok": 1,
                "aal_shape": "x".join(str(v) for v in data.shape),
                "aal_valid_labels_present": len(present),
            }
        )
    except Exception as exc:
        out["aal_read_error"] = f"{type(exc).__name__}: {exc}"
    return out


def validate_subject(sid: str, deriv_root: Path, atlas: Atlas, min_density: float, min_labels: int) -> dict[str, object]:
    conn_dir = deriv_root / "connectomes"
    parc_dir = deriv_root / "parc" / sid
    tracks_dir = deriv_root / "tracks" / sid
    row: dict[str, object] = {
        "sid": sid,
        "validation_status": "PASS",
        "analysis_gate": "include",
        "reasons": "",
        "backup_required": "no",
        "assignments_exists": int((tracks_dir / "assignments_aal.csv").exists()),
        "assignments_size_bytes": (tracks_dir / "assignments_aal.csv").stat().st_size
        if (tracks_dir / "assignments_aal.csv").exists()
        else 0,
    }

    row.update(aal_label_count(parc_dir / "AAL_b0.nii.gz", atlas))
    row.update(validate_all_csv(conn_dir / f"SC_AAL_{sid}_ALL.csv"))

    reasons: list[str] = []
    if row["assignments_exists"] != 1 or int(row["assignments_size_bytes"]) <= 0:
        reasons.append("missing_or_empty_assignments")
    if row["aal_read_ok"] != 1:
        reasons.append(f"AAL_b0_unreadable:{row['aal_read_error']}")
    elif int(row["aal_valid_labels_present"]) < min_labels:
        reasons.append(f"AAL_b0_low_label_survival:{row['aal_valid_labels_present']}<{min_labels}")
    if row["sc_all_read_ok"] != 1:
        reasons.append(f"SC_ALL_unreadable:{row['sc_all_read_error']}")
    elif str(row["sc_all_missing_metrics"]):
        reasons.append(f"SC_ALL_missing_metrics:{row['sc_all_missing_metrics']}")

    primary_density: list[float] = []
    for weight in WEIGHTS:
        checks = matrix_checks(conn_dir / f"SC_AAL_{sid}_{weight}.csv", atlas)
        prefix = f"{weight}_"
        for key, value in checks.items():
            row[f"{prefix}{key}"] = value
        if checks["read_ok"] != 1:
            reasons.append(f"{weight}_unreadable:{checks['read_error']}")
            continue
        if float(checks["finite_fraction"]) < 1.0:
            reasons.append(f"{weight}_nonfinite_values")
        if float(checks["symmetry_max_abs"]) > 1e-5:
            reasons.append(f"{weight}_not_symmetric")
        if float(checks["diagonal_abs_sum"]) > 1e-5:
            reasons.append(f"{weight}_nonzero_diagonal")
        if weight in PRIMARY_ZERO_ROW_WEIGHTS:
            density = float(checks["density"])
            primary_density.append(density)
            if int(checks["n_unexpected_zero_rows"]) > 0:
                reasons.append(f"{weight}_unexpected_valid_zero_rows:{checks['n_unexpected_zero_rows']}")
            if not math.isfinite(density) or density < min_density:
                reasons.append(f"{weight}_low_density:{density:.4g}<{min_density}")
            if int(checks["nonzero_upper_edges"]) <= 0:
                reasons.append(f"{weight}_no_nonzero_edges")

    row["primary_min_density"] = min(primary_density) if primary_density else math.nan
    if reasons:
        row["validation_status"] = "FAIL_VALIDATION"
        row["analysis_gate"] = "exclude_pending_review"
        row["backup_required"] = "yes_old_outputs_available_in_repair_backups"
    row["reasons"] = " | ".join(reasons) if reasons else "all required repaired SC outputs passed validation"
    return row


def write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include", type=Path, default=QC_ROOT / "repair_include_post_parcellation.txt")
    parser.add_argument("--deriv-root", type=Path, default=DERIV)
    parser.add_argument("--qc-root", type=Path, default=QC_ROOT)
    parser.add_argument("--label-csv", type=Path, default=LABEL_CSV)
    parser.add_argument("--min-density", type=float, default=0.15)
    parser.add_argument("--min-aal-labels", type=int, default=150)
    parser.add_argument("--tag", default="post_parcellation_repair")
    args = parser.parse_args()

    atlas = load_atlas(args.label_csv)
    sids = read_sid_list(args.include)
    rows = [validate_subject(sid, args.deriv_root, atlas, args.min_density, args.min_aal_labels) for sid in sids]
    subject_csv = args.qc_root / f"{args.tag}_validation_subjects.csv"
    write_csv(subject_csv, rows)

    counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("validation_status", "UNKNOWN"))
        counts[status] = counts.get(status, 0) + 1
    gate_counts: dict[str, int] = {}
    for row in rows:
        gate = str(row.get("analysis_gate", "UNKNOWN"))
        gate_counts[gate] = gate_counts.get(gate, 0) + 1
    summary_rows = [
        {"metric": "subjects_checked", "value": len(rows)},
        *({"metric": f"status_{k}", "value": v} for k, v in sorted(counts.items())),
        *({"metric": f"gate_{k}", "value": v} for k, v in sorted(gate_counts.items())),
        {"metric": "subject_csv", "value": str(subject_csv)},
    ]
    summary_csv = args.qc_root / f"{args.tag}_validation_summary.csv"
    write_csv(summary_csv, summary_rows)

    print(f"validated subjects: {len(rows)}")
    print("status:", ", ".join(f"{k}={v}" for k, v in sorted(counts.items())) or "none")
    print("gate:", ", ".join(f"{k}={v}" for k, v in sorted(gate_counts.items())) or "none")
    print(f"subject report: {subject_csv}")
    print(f"summary report: {summary_csv}")
    return 0 if counts.get("FAIL_VALIDATION", 0) == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
