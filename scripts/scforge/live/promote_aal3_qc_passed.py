#!/usr/bin/env python3
"""Promote QC-passed or improved-review AAL3 fd_sum matrices into production.

The script is deliberately narrow: it only promotes subjects that passed numeric
QC (final pass) or explicit improved-review policy. By default every promotion
requires an explicit visual overlay pass in visual_overlay_qc_manifest.csv. With
the user-requested overrides, numeric-pass or density-improved candidates can be
copied to production even when visual QC is still unresolved; those rows remain
open for review/alternate routes rather than being marked solved.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import time
from pathlib import Path
from typing import Any


DERIV = Path("/data/derivatives")
COUNT_MATRIX_WIDTH = 170


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def stamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size <= 0:
        return []
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        return list(csv.DictReader(handle))


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


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def same_file(a: Path, b: Path) -> bool:
    try:
        return a.samefile(b)
    except FileNotFoundError:
        return False


def copy2_unless_same(src: Path, dst: Path) -> bool:
    if same_file(src, dst):
        return False
    shutil.copy2(src, dst)
    return True


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


def stored_full_qc_numeric_pass(row: dict[str, str]) -> bool:
    """Recognize older promoted rows that predate promotion_class bookkeeping."""
    return (
        inum(row.get("source_labels"), 0) >= 160
        and inum(row.get("best_zero_rows"), 9999) <= 15
        and fnum(row.get("best_density"), 0.0) >= 0.10
        and fnum(row.get("both_assigned_fraction"), 0.0) >= 0.85
        and fnum(row.get("one_unassigned_fraction"), 0.0) <= 0.15
        and fnum(row.get("both_unassigned_fraction"), 0.0) <= 0.05
    )


def matrix_qc(path: Path) -> dict[str, Any]:
    """Compute density/zero rows directly from a square connectome CSV."""
    if not path.exists() or path.stat().st_size <= 0:
        return {"density": float("nan"), "zero_rows": None, "n": 0}
    matrix: list[list[float]] = []
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        for raw in csv.reader(handle):
            values: list[float] = []
            for value in raw:
                try:
                    values.append(float(value))
                except Exception:
                    values = []
                    break
            if values:
                matrix.append(values)
    n = len(matrix)
    if n == 0:
        return {"density": float("nan"), "zero_rows": None, "n": 0}
    width = min([n] + [len(row) for row in matrix])
    if width <= 1:
        return {"density": float("nan"), "zero_rows": n, "n": n}
    nonzero_edges = 0
    zero_rows = 0
    for i in range(width):
        row_has_edge = False
        for j in range(width):
            if i == j:
                continue
            value = matrix[i][j]
            if value > 0:
                row_has_edge = True
                if j > i:
                    nonzero_edges += 1
        if not row_has_edge:
            zero_rows += 1
    possible = width * (width - 1) / 2
    return {"density": nonzero_edges / possible if possible else float("nan"), "zero_rows": zero_rows, "n": width}


def assignment_count_matrix(path: Path, width: int = COUNT_MATRIX_WIDTH) -> list[list[int]]:
    """Build a symmetric tract-count matrix from tck2connectome assignments.

    The production edge-weight matrix remains fd_sum/SIFT2. This count matrix
    exists for QC and the SC viewer density chart, where density means at least
    one tract between two different ROI labels.
    """
    matrix = [[0 for _ in range(width)] for _ in range(width)]
    if not path.exists() or path.stat().st_size <= 0:
        return matrix
    with path.open(encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            raw = raw.strip()
            if not raw or raw.startswith("#"):
                continue
            parts = raw.replace(",", " ").split()
            if len(parts) < 2:
                continue
            try:
                a = int(float(parts[0]))
                b = int(float(parts[1]))
            except Exception:
                continue
            if a <= 0 or b <= 0 or a > width or b > width or a == b:
                continue
            i = a - 1
            j = b - 1
            matrix[i][j] += 1
            matrix[j][i] += 1
    return matrix


def write_count_matrix(path: Path, matrix: list[list[int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerows(matrix)
    tmp.replace(path)


def visual_pass_map(run_root: Path) -> dict[str, dict[str, str]]:
    rows = read_csv(run_root / "visual_overlay_qc_manifest.csv")
    return {row.get("sid", ""): row for row in rows if row.get("visual_overlay_status", "").startswith("passed")}


def visual_status_map(run_root: Path) -> dict[str, dict[str, str]]:
    rows = read_csv(run_root / "visual_overlay_qc_manifest.csv")
    return {row.get("sid", ""): row for row in rows if row.get("sid")}


def matching_matrix_row(row: dict[str, str], probe_root: Path) -> dict[str, str]:
    candidate = row.get("best_candidate") or row.get("tested_route") or "AAL3_source_contract"
    variant = row.get("best_variant") or ""
    rows = [
        candidate_row
        for candidate_row in read_csv(probe_root / "matrix_candidate_summary.csv")
        if candidate_row.get("candidate") == candidate
        and candidate_row.get("variant") == variant
        and candidate_row.get("metric", "fd_sum") == "fd_sum"
    ]
    return rows[-1] if rows else {}


def matching_label_path(row: dict[str, str], probe_root: Path) -> Path:
    stage = row.get("tested_route") or row.get("best_candidate") or "AAL3_source_contract_B0"
    for label_row in read_csv(probe_root / "label_survival_summary.csv"):
        if label_row.get("stage") == stage and label_row.get("path"):
            path = Path(label_row["path"])
            if path.exists():
                return path
    if row.get("best_candidate") == "AAL3_source_contract":
        path = probe_root / "parc" / "AAL3_source_contract_B0.nii.gz"
        if path.exists():
            return path
    candidates = sorted((probe_root / "parc").rglob("*AAL3*b0*.nii.gz"))
    if candidates:
        preferred = [path for path in candidates if stage in str(path)]
        return preferred[-1] if preferred else candidates[-1]
    return Path("__missing_candidate_parcellation__")


def promote_row(row: dict[str, str], visual: dict[str, str], backup_root: Path, dry_run: bool) -> dict[str, Any]:
    sid = row["sid"]
    variant = row.get("best_variant", "")
    probe_root = Path(row.get("probe_root") or "")
    if row.get("candidate_matrix"):
        cand_matrix = Path(row["candidate_matrix"])
    else:
        matrix_row = matching_matrix_row(row, probe_root)
        cand_matrix = Path(matrix_row.get("matrix") or "__missing_candidate_matrix__")
    if row.get("candidate_parc"):
        cand_parc = Path(row["candidate_parc"])
    else:
        cand_parc = matching_label_path(row, probe_root)
    prod_matrix = DERIV / "connectomes" / f"SC_AAL_{sid}_fd_sum.csv"
    prod_count_matrix = DERIV / "connectomes" / f"SC_AAL_{sid}_count.csv"
    prod_parc = DERIV / "parc" / sid / "AAL_b0.nii.gz"
    use_count_qc = str(row.get("qc_density_basis", "")).startswith("tract-count")
    assignment_file = Path(row.get("assignment_file") or "__missing_assignment_file__")
    missing = [str(p) for p in [cand_matrix, cand_parc, prod_matrix, prod_parc] if not p.exists()]
    if use_count_qc and not assignment_file.exists():
        missing.append(str(assignment_file))
    decision = row.get("qc_decision", "")
    visual_status = visual.get("visual_overlay_status", "")
    solved = decision == "PROMOTE_CANDIDATE_PENDING_VISUAL" and visual_status.startswith("passed")
    numeric_override = decision == "PROMOTE_CANDIDATE_PENDING_VISUAL" and not solved
    out: dict[str, Any] = {
        "sid": sid,
        "promoted_utc": utc(),
        "dry_run": int(dry_run),
        "status": "missing_inputs" if missing else ("dry_run_ready" if dry_run else "promoted"),
        "missing_inputs": ";".join(missing),
        "best_variant": variant,
        "best_candidate": row.get("best_candidate", ""),
        "tested_route": row.get("tested_route", ""),
        "qc_decision": row.get("qc_decision", ""),
        "promotion_class": (
            "final_qc_pass"
            if solved
            else ("numeric_pass_pending_visual_not_solved" if numeric_override else "interim_review_not_solved")
        ),
        "final_qc_status": "solved" if solved else "not_solved_keep_trying_routes",
        "source_labels": row.get("source_labels", ""),
        "best_density": row.get("best_density", ""),
        "best_zero_rows": row.get("best_zero_rows", ""),
        "both_assigned_fraction": row.get("both_assigned_fraction", ""),
        "interim_review_flags": row.get("interim_review_flags", ""),
        "visual_overlay_status": visual.get("visual_overlay_status", ""),
        "visual_overlay_png": visual.get("overlay_png", ""),
        "promotion_note": "final QC solved"
        if solved
        else "production overwrite; subject remains open for review/alternate route",
        "candidate_matrix": str(cand_matrix),
        "candidate_parc": str(cand_parc),
        "production_matrix": str(prod_matrix),
        "production_count_matrix": str(prod_count_matrix),
        "production_parc": str(prod_parc),
        "backup_matrix": "",
        "backup_count_matrix": "",
        "backup_parc": "",
        "qc_density_basis": row.get("qc_density_basis", ""),
        "candidate_matrix_sha256": sha256(cand_matrix) if cand_matrix.exists() else "",
        "candidate_parc_sha256": sha256(cand_parc) if cand_parc.exists() else "",
        "assignment_file": str(assignment_file) if use_count_qc else "",
    }
    if missing:
        return out
    current_qc = matrix_qc(prod_matrix)
    candidate_qc = matrix_qc(cand_matrix)
    current_density = fnum(current_qc.get("density"))
    candidate_density = fnum(row.get("best_density"), fnum(candidate_qc.get("density")))
    current_zero_rows = current_qc.get("zero_rows")
    candidate_zero_rows = inum(row.get("best_zero_rows"), inum(candidate_qc.get("zero_rows"), 9999))
    current_count_density = fnum(row.get("baseline_tract_count_density"))
    candidate_count_density = fnum(row.get("best_tract_count_density"))
    current_count_zero_rows = (
        inum(row.get("baseline_tract_count_zero_rows"), 9999)
        if str(row.get("baseline_tract_count_zero_rows", "")).strip()
        else None
    )
    candidate_count_zero_rows = (
        inum(row.get("best_tract_count_zero_rows"), 9999)
        if str(row.get("best_tract_count_zero_rows", "")).strip()
        else None
    )
    compare_current_density = current_count_density if use_count_qc and math.isfinite(current_count_density) else current_density
    compare_candidate_density = (
        candidate_count_density if use_count_qc and math.isfinite(candidate_count_density) else candidate_density
    )
    compare_current_zero_rows = current_count_zero_rows if use_count_qc and current_count_zero_rows is not None else current_zero_rows
    compare_candidate_zero_rows = (
        candidate_count_zero_rows if use_count_qc and candidate_count_zero_rows is not None else candidate_zero_rows
    )
    out["current_production_density"] = current_density if math.isfinite(current_density) else ""
    out["current_production_zero_rows"] = current_zero_rows if current_zero_rows is not None else ""
    out["candidate_computed_density"] = fnum(candidate_qc.get("density")) if math.isfinite(fnum(candidate_qc.get("density"))) else ""
    out["candidate_computed_zero_rows"] = candidate_qc.get("zero_rows") if candidate_qc.get("zero_rows") is not None else ""
    out["current_tract_count_density"] = current_count_density if math.isfinite(current_count_density) else ""
    out["candidate_tract_count_density"] = candidate_count_density if math.isfinite(candidate_count_density) else ""
    out["current_tract_count_zero_rows"] = current_count_zero_rows if current_count_zero_rows is not None else ""
    out["candidate_tract_count_zero_rows"] = candidate_count_zero_rows if candidate_count_zero_rows is not None else ""
    out["comparison_density"] = compare_candidate_density if math.isfinite(compare_candidate_density) else ""
    out["comparison_current_density"] = compare_current_density if math.isfinite(compare_current_density) else ""
    out["comparison_zero_rows"] = compare_candidate_zero_rows if compare_candidate_zero_rows is not None else ""
    out["comparison_current_zero_rows"] = compare_current_zero_rows if compare_current_zero_rows is not None else ""
    sid_backup = backup_root / sid
    backup_matrix = sid_backup / prod_matrix.name
    backup_count_matrix = sid_backup / prod_count_matrix.name
    backup_parc = sid_backup / prod_parc.name
    out["backup_matrix"] = str(backup_matrix)
    out["backup_count_matrix"] = str(backup_count_matrix) if use_count_qc else ""
    out["backup_parc"] = str(backup_parc)
    out["production_matrix_sha256_before"] = sha256(prod_matrix)
    out["production_count_matrix_sha256_before"] = sha256(prod_count_matrix) if prod_count_matrix.exists() else ""
    out["production_parc_sha256_before"] = sha256(prod_parc)
    if (
        not use_count_qc
        and
        out["candidate_matrix_sha256"] == out["production_matrix_sha256_before"]
        and out["candidate_parc_sha256"] == out["production_parc_sha256_before"]
    ):
        out["status"] = "promoted"
        out["promotion_note"] = "current production already matches candidate; recorded manifest without copy"
        out["production_matrix_sha256_after"] = out["production_matrix_sha256_before"]
        out["production_count_matrix_sha256_after"] = out["production_count_matrix_sha256_before"]
        out["production_parc_sha256_after"] = out["production_parc_sha256_before"]
        return out
    worse_density = (
        math.isfinite(compare_current_density)
        and math.isfinite(compare_candidate_density)
        and compare_candidate_density < compare_current_density - 1e-12
    )
    no_density_gain = (
        math.isfinite(compare_current_density)
        and math.isfinite(compare_candidate_density)
        and compare_candidate_density <= compare_current_density + 1e-12
    )
    worse_zero = compare_current_zero_rows is not None and compare_candidate_zero_rows > int(compare_current_zero_rows)
    no_zero_gain = compare_current_zero_rows is not None and compare_candidate_zero_rows >= int(compare_current_zero_rows)
    if worse_density or worse_zero or (no_density_gain and no_zero_gain):
        out["status"] = "skipped_not_better_than_current_production"
        reasons = []
        if worse_density:
            reasons.append(f"candidate density {compare_candidate_density:.6g} < current {compare_current_density:.6g}")
        elif no_density_gain:
            reasons.append(f"candidate density {compare_candidate_density:.6g} <= current {compare_current_density:.6g}")
        if worse_zero:
            reasons.append(f"candidate zero rows {compare_candidate_zero_rows} > current {compare_current_zero_rows}")
        elif no_zero_gain:
            reasons.append(f"candidate zero rows {compare_candidate_zero_rows} >= current {compare_current_zero_rows}")
        out["promotion_note"] = "skipped: " + "; ".join(reasons)
        return out
    if not dry_run:
        sid_backup.mkdir(parents=True, exist_ok=True)
        shutil.copy2(prod_matrix, backup_matrix)
        if use_count_qc and prod_count_matrix.exists():
            shutil.copy2(prod_count_matrix, backup_count_matrix)
        shutil.copy2(prod_parc, backup_parc)
        copy2_unless_same(cand_matrix, prod_matrix)
        if use_count_qc:
            write_count_matrix(prod_count_matrix, assignment_count_matrix(assignment_file))
        prod_parc.parent.mkdir(parents=True, exist_ok=True)
        copy2_unless_same(cand_parc, prod_parc)
        out["production_matrix_sha256_after"] = sha256(prod_matrix)
        out["production_count_matrix_sha256_after"] = sha256(prod_count_matrix) if prod_count_matrix.exists() else ""
        out["production_parc_sha256_after"] = sha256(prod_parc)
    return out


def upgrade_existing_visual_passes(existing_rows: list[dict[str, str]], passed_visuals: dict[str, dict[str, str]]) -> int:
    """Mark already-promoted numeric-pass rows solved once visual QC passes."""
    upgraded = 0
    for row in existing_rows:
        sid = row.get("sid", "")
        visual = passed_visuals.get(sid)
        numeric_pass_decision = row.get("qc_decision") == "PROMOTE_CANDIDATE_PENDING_VISUAL"
        legacy_numeric_pass = not row.get("qc_decision") and stored_full_qc_numeric_pass(row)
        if (
            row.get("status") == "promoted"
            and (numeric_pass_decision or legacy_numeric_pass)
            and row.get("final_qc_status") != "solved"
            and visual is not None
        ):
            if legacy_numeric_pass:
                row["qc_decision"] = "PROMOTE_CANDIDATE_PENDING_VISUAL"
            row["promotion_class"] = "final_qc_pass"
            row["final_qc_status"] = "solved"
            row["visual_overlay_status"] = visual.get("visual_overlay_status", "")
            row["visual_overlay_png"] = visual.get("overlay_png", row.get("visual_overlay_png", ""))
            row["promotion_note"] = "final QC solved after visual review"
            upgraded += 1
    return upgraded


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--execute", action="store_true", help="Actually copy files. Without this, only dry-run.")
    parser.add_argument(
        "--allow-interim-without-visual-pass",
        action="store_true",
        help=(
            "User override: promote INTERIM_REVIEW rows even if the visual overlay "
            "status is not a pass. Final-QC rows still require visual pass."
        ),
    )
    parser.add_argument(
        "--allow-final-without-visual-pass",
        action="store_true",
        help=(
            "User override: promote numeric-pass rows even if the visual overlay "
            "status is not a pass. These rows are not marked solved."
        ),
    )
    parser.add_argument(
        "--reapply-existing-if-better",
        action="store_true",
        help=(
            "Reconsider already-promoted rows in this route manifest and copy them "
            "again only if they improve on the current production matrix. This is "
            "used to restore a better earlier route if a later route overwrote it."
        ),
    )
    args = parser.parse_args()

    run_root = args.run_root
    decisions = read_csv(run_root / "route_qc_decisions.csv")
    passed_visuals = visual_pass_map(run_root)
    all_visuals = visual_status_map(run_root)
    executed_manifest = run_root / "production_promotion_manifest.csv"
    manifest = executed_manifest if args.execute else run_root / "production_promotion_dry_run.csv"
    existing_rows = read_csv(executed_manifest)
    upgraded_existing = upgrade_existing_visual_passes(existing_rows, passed_visuals) if args.execute else 0
    already_promoted = {row.get("sid", "") for row in existing_rows if row.get("status") == "promoted"}
    latest_promoted_by_sid: dict[str, dict[str, str]] = {}
    for row in existing_rows:
        sid = row.get("sid", "")
        if row.get("status") != "promoted" or not sid:
            continue
        if sid not in latest_promoted_by_sid or row.get("promoted_utc", "") >= latest_promoted_by_sid[sid].get(
            "promoted_utc", ""
        ):
            latest_promoted_by_sid[sid] = row
    candidates: list[dict[str, str]] = []
    visuals: dict[str, dict[str, str]] = {}
    for row in decisions:
        sid = row.get("sid", "")
        decision = row.get("qc_decision", "")
        if not sid or sid in already_promoted:
            continue
        if decision == "PROMOTE_CANDIDATE_PENDING_VISUAL":
            visual = passed_visuals.get(sid)
            if visual is None and args.allow_final_without_visual_pass:
                visual = all_visuals.get(sid, {"sid": sid, "visual_overlay_status": "override_no_visual_manifest"})
            if visual is not None:
                candidates.append(row)
                visuals[sid] = visual
        elif decision == "INTERIM_REVIEW_PROMOTION_PENDING_VISUAL":
            visual = passed_visuals.get(sid)
            if visual is None and args.allow_interim_without_visual_pass:
                visual = all_visuals.get(sid, {"sid": sid, "visual_overlay_status": "override_no_visual_manifest"})
            if visual is not None:
                candidates.append(row)
                visuals[sid] = visual
    if args.execute and args.reapply_existing_if_better:
        for row in latest_promoted_by_sid.values():
            sid = row.get("sid", "")
            if not sid:
                continue
            candidates.append(row)
            visuals[sid] = {
                "sid": sid,
                "visual_overlay_status": row.get("visual_overlay_status", "reapply_existing_if_better"),
                "overlay_png": row.get("visual_overlay_png", ""),
            }
    backup_root = run_root / "production_backups" / stamp()
    new_rows = [promote_row(row, visuals[row["sid"]], backup_root, dry_run=not args.execute) for row in candidates]
    rows_to_append = [row for row in new_rows if row.get("status") != "skipped_not_better_than_current_production"]
    rows = existing_rows + rows_to_append if args.execute else new_rows
    write_csv(manifest, rows)
    summary = {
        "generated_utc": utc(),
        "run_root": str(run_root),
        "execute": bool(args.execute),
        "allow_interim_without_visual_pass": bool(args.allow_interim_without_visual_pass),
        "allow_final_without_visual_pass": bool(args.allow_final_without_visual_pass),
        "reapply_existing_if_better": bool(args.reapply_existing_if_better),
        "already_promoted": len(already_promoted),
        "upgraded_existing_visual_pass": upgraded_existing,
        "eligible_visual_pass_new": len(candidates),
        "promoted_new": sum(1 for row in new_rows if row.get("status") == "promoted"),
        "promoted_total": sum(1 for row in rows if row.get("status") == "promoted"),
        "dry_run_ready": sum(1 for row in new_rows if row.get("status") == "dry_run_ready"),
        "missing_inputs": sum(1 for row in new_rows if row.get("status") == "missing_inputs"),
        "skipped_not_better_than_current_production": sum(
            1 for row in new_rows if row.get("status") == "skipped_not_better_than_current_production"
        ),
        "backup_root": str(backup_root),
        "manifest": str(manifest),
    }
    write_json(run_root / ("production_promotion_summary.json" if args.execute else "production_promotion_dry_run_summary.json"), summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
