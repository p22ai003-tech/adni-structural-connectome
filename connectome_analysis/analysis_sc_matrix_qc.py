from __future__ import annotations

import math
import re
import subprocess
import tempfile
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import nibabel as nib
import numpy as np
import pandas as pd

from connectome_analysis.analysis_config import AnalysisPaths, get_analysis_paths
from connectome_analysis.analysis_cohort import recode_group


CONNECTOME_RE = re.compile(r"^SC_AAL166_(?P<sid>\d{3}_S_\d+(?:_I\d+)?)_(?P<weight>.+)\.csv$")
MATRIX_WEIGHTS = (
    "count",
    "fd_sum",
    "len_mean",
    "invlen_mean",
    "fa_mean",
    "md_mean",
    "rd_mean",
    "ad_mean",
)
PRIMARY_WEIGHT = "fd_sum"
FRONTAL_OPERCULAR_NODES = (7, 8)
MIN_LABEL_VOXELS = 50
MIN_MASK_OVERLAP_FRAC = 0.01
WARN_VALID_ZERO_ROWS = 25
REPAIR_LANE_ACTIONS = {
    "analysis_include": "include in analysis after standard QC review",
    "manual_review_sparse": "manual heatmap/overlay review before final inference",
    "post_parcellation": "exclude from analysis; downstream post-parcellation batch repair is blocked until a raw upstream supervisor-route canary passes",
    "bbr_registration": "exclude from analysis; rerun from upstream T1/B0 registration contract, not by blind Step 7 post repair",
    "assignment_or_tracks": "exclude from analysis; run scratch assignment/track-contract replay only after anatomy passes QC",
    "source_failure": "exclude from analysis; restore or rebuild missing/corrupt upstream source artifacts before any rerun",
}
FINAL_RECTIFICATION_DECISION_NAME = "FINAL_SC_MATRIX_RECTIFICATION_DECISION_20260518.md"
FINAL_RECTIFICATION_GUIDANCE = (
    "Downstream SC repair from current artifacts is not batch-approved. "
    "Use PASS/WARN subjects only for analysis; FAIL subjects require upstream "
    "source-contract replay or full supervisor-pipeline rerun before promotion."
)


@dataclass(frozen=True)
class AtlasInfo:
    labels: pd.DataFrame
    label_lookup: dict[int, str]
    valid_nodes: set[int]
    max_node: int
    expected_gaps: set[int]


def sid_to_subject_id(sid: str) -> str:
    return "_".join(str(sid).split("_")[:3])


def _default_label_path(paths: AnalysisPaths) -> Path:
    """Labels keyed on the MATRIX ROW of the 166-node SC_AAL166 matrices.

    This used AAL3_labels.csv, which is keyed on the original atlas value
    (1..170 with 35, 36, 81, 82 unused). Applied to 166-row matrices that made
    rows 35, 36, 81 and 82 look like "expected gaps" and misnamed every row
    from 35 onward.
    """
    return paths.notebook_dir / "atlas" / "AAL" / "aal3_labels_166.csv"


def load_aal3_labels(paths: AnalysisPaths) -> AtlasInfo:
    label_path = _default_label_path(paths)
    labels = pd.read_csv(label_path)
    if "node" not in labels.columns or "atlas_label" not in labels.columns:
        raise ValueError(f"AAL3 label file is missing required columns: {label_path}")
    labels = labels.copy()
    labels["node"] = pd.to_numeric(labels["node"], errors="coerce").astype("Int64")
    labels = labels.dropna(subset=["node"]).copy()
    labels["node"] = labels["node"].astype(int)
    label_lookup = dict(zip(labels["node"], labels["atlas_label"].astype(str)))
    valid_nodes = set(label_lookup)
    max_node = int(labels["node"].max())
    expected_gaps = set(range(1, max_node + 1)).difference(valid_nodes)
    return AtlasInfo(
        labels=labels,
        label_lookup=label_lookup,
        valid_nodes=valid_nodes,
        max_node=max_node,
        expected_gaps=expected_gaps,
    )


def _cohort_group_lookup(paths: AnalysisPaths) -> dict[str, str]:
    if not paths.cohort_dti_csv.exists():
        return {}
    cohort = pd.read_csv(paths.cohort_dti_csv)
    if "Subject ID" in cohort.columns:
        cohort["subject_id"] = cohort["Subject ID"].astype(str)
    if "Research Group" in cohort.columns:
        cohort["group"] = cohort["Research Group"].map(recode_group)
    elif "group" in cohort.columns:
        cohort["group"] = cohort["group"].map(recode_group)
    else:
        return {}
    cohort = cohort.dropna(subset=["subject_id", "group"]).drop_duplicates("subject_id")
    return dict(zip(cohort["subject_id"], cohort["group"]))


def _connectome_files(paths: AnalysisPaths, weights: Iterable[str]) -> list[Path]:
    files: list[Path] = []
    weight_set = set(weights)
    if not paths.connectomes_dir.exists():
        return files
    for path in sorted(paths.connectomes_dir.glob("SC_AAL166_*.csv")):
        match = CONNECTOME_RE.match(path.name)
        if not match:
            continue
        if match.group("weight") in weight_set:
            files.append(path)
    return files


def _parse_connectome_file(path: Path) -> tuple[str, str] | None:
    match = CONNECTOME_RE.match(path.name)
    if not match:
        return None
    return match.group("sid"), match.group("weight")


def _node_label(node: int, atlas: AtlasInfo) -> str:
    return atlas.label_lookup.get(int(node), f"EXPECTED_LABEL_GAP_{int(node):03d}")


def _read_matrix(path: Path) -> np.ndarray:
    matrix = pd.read_csv(path, header=None).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"Not a 2D matrix: {path}")
    return matrix


def _list_to_text(values: Iterable[int]) -> str:
    vals = [int(v) for v in values]
    return ";".join(str(v) for v in sorted(vals))


def audit_sc_matrices(
    paths: AnalysisPaths,
    atlas: AtlasInfo,
    weights: Iterable[str] = MATRIX_WEIGHTS,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, set[int]]]:
    group_lookup = _cohort_group_lookup(paths)
    rows: list[dict] = []
    zero_counts: Counter[tuple[str, int]] = Counter()
    total_counts: Counter[str] = Counter()
    target_nodes_by_sid: dict[str, set[int]] = defaultdict(lambda: set(FRONTAL_OPERCULAR_NODES))

    for path in _connectome_files(paths, weights):
        parsed = _parse_connectome_file(path)
        if parsed is None:
            continue
        sid, weight = parsed
        subject_id = sid_to_subject_id(sid)
        total_counts[weight] += 1
        try:
            matrix = _read_matrix(path)
            finite_mask = np.isfinite(matrix)
            finite_fraction = float(finite_mask.mean()) if matrix.size else 0.0
            clean = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
            n_rows, n_cols = clean.shape
            n = min(n_rows, n_cols)
            row_abs_sum = np.abs(clean).sum(axis=1)
            col_abs_sum = np.abs(clean).sum(axis=0)
            zero_rows = {idx + 1 for idx, val in enumerate(row_abs_sum[:n]) if val == 0}
            zero_cols = {idx + 1 for idx, val in enumerate(col_abs_sum[:n]) if val == 0}
            expected_gap_zero_rows = zero_rows.intersection(atlas.expected_gaps)
            valid_zero_rows = zero_rows.intersection(atlas.valid_nodes)
            unexpected_zero_rows = zero_rows.difference(atlas.expected_gaps)
            if weight in {PRIMARY_WEIGHT, "count", "len_mean"}:
                target_nodes_by_sid[sid].update(valid_zero_rows)
            for node in zero_rows:
                zero_counts[(weight, node)] += 1
            if n > 1:
                upper_mask = np.triu(np.ones((n, n), dtype=bool), 1)
                nonzero_upper = int((np.abs(clean[:n, :n]) > 0)[upper_mask].sum())
                possible_upper = int(n * (n - 1) / 2)
                density = nonzero_upper / possible_upper if possible_upper else np.nan
                symmetry = float(np.max(np.abs(clean[:n, :n] - clean[:n, :n].T)))
            else:
                nonzero_upper = 0
                density = np.nan
                symmetry = np.nan
            rows.append(
                {
                    "sid": sid,
                    "subject_id": subject_id,
                    "group": group_lookup.get(subject_id, np.nan),
                    "weight": weight,
                    "matrix_path": str(path),
                    "n_rows": n_rows,
                    "n_cols": n_cols,
                    "finite_fraction": finite_fraction,
                    "symmetry_max_abs": symmetry,
                    "diagonal_abs_sum": float(np.abs(np.diag(clean[:n, :n])).sum()) if n else np.nan,
                    "nonzero_upper_edges": nonzero_upper,
                    "density": density,
                    "n_zero_rows": len(zero_rows),
                    "n_zero_cols": len(zero_cols),
                    "n_expected_gap_zero_rows": len(expected_gap_zero_rows),
                    "n_valid_zero_rows": len(valid_zero_rows),
                    "n_unexpected_zero_rows": len(unexpected_zero_rows),
                    "zero_rows": _list_to_text(zero_rows),
                    "zero_cols": _list_to_text(zero_cols),
                    "expected_gap_zero_rows": _list_to_text(expected_gap_zero_rows),
                    "valid_zero_rows": _list_to_text(valid_zero_rows),
                    "unexpected_zero_rows": _list_to_text(unexpected_zero_rows),
                    "frontal_oper_l_zero": int(7 in zero_rows),
                    "frontal_oper_r_zero": int(8 in zero_rows),
                    "read_error": "",
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "sid": sid,
                    "subject_id": subject_id,
                    "group": group_lookup.get(subject_id, np.nan),
                    "weight": weight,
                    "matrix_path": str(path),
                    "read_error": f"{type(exc).__name__}: {exc}",
                }
            )

    audit = pd.DataFrame(rows)
    summary_rows: list[dict] = []
    for weight in sorted(total_counts):
        total = int(total_counts[weight])
        max_node = max(atlas.max_node, int(audit.loc[audit["weight"] == weight, "n_rows"].dropna().max() or 0))
        for node in range(1, max_node + 1):
            n_zero = int(zero_counts[(weight, node)])
            summary_rows.append(
                {
                    "weight": weight,
                    "node": node,
                    "atlas_label": _node_label(node, atlas),
                    "expected_label_gap": int(node in atlas.expected_gaps),
                    "zero_subjects": n_zero,
                    "total_subjects": total,
                    "zero_subject_pct": n_zero / total if total else np.nan,
                }
            )
    zero_summary = pd.DataFrame(summary_rows)
    return audit, zero_summary, {sid: set(nodes) for sid, nodes in target_nodes_by_sid.items()}


def _load_label_image(path: Path) -> np.ndarray:
    return np.asanyarray(nib.load(str(path)).dataobj)


def _convert_mif_mask_to_array(mask_path: Path, mrtrix_bin: Path) -> np.ndarray | None:
    if not mask_path.exists():
        return None
    with tempfile.TemporaryDirectory(prefix="scqc_mask_") as tmpdir:
        out_path = Path(tmpdir) / "mask.nii.gz"
        cmd = [str(mrtrix_bin / "mrconvert"), str(mask_path), str(out_path), "-force", "-quiet"]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return np.asanyarray(nib.load(str(out_path)).dataobj) > 0


def audit_parcellation_coverage(
    paths: AnalysisPaths,
    atlas: AtlasInfo,
    target_nodes_by_sid: dict[str, set[int]],
    max_subjects: int | None = None,
) -> pd.DataFrame:
    rows: list[dict] = []
    mrtrix_bin = Path("/home/ec2-user/mrtrix3/bin")
    sids = sorted(target_nodes_by_sid)
    if max_subjects is not None:
        sids = sids[:max_subjects]
    for sid in sids:
        parc_path = paths.deriv_root / "parc" / sid / "AAL_b0.nii.gz"
        mask_path = paths.deriv_root / "fod" / sid / "mask_5tt_brain.mif"
        nodes = sorted(n for n in target_nodes_by_sid[sid] if n in atlas.valid_nodes)
        if not nodes:
            continue
        subject_id = sid_to_subject_id(sid)
        if not parc_path.exists():
            for node in nodes:
                rows.append(
                    {
                        "sid": sid,
                        "subject_id": subject_id,
                        "node": node,
                        "atlas_label": _node_label(node, atlas),
                        "AAL_b0_path": str(parc_path),
                        "mask_5tt_path": str(mask_path),
                        "source_status": "missing_AAL_b0",
                    }
                )
            continue
        try:
            label_img = _load_label_image(parc_path)
            mask_img = None
            mask_error = ""
            try:
                mask_img = _convert_mif_mask_to_array(mask_path, mrtrix_bin)
            except Exception as exc:
                mask_error = f"{type(exc).__name__}: {exc}"
            shape_match = bool(mask_img is not None and tuple(label_img.shape) == tuple(mask_img.shape))
            for node in nodes:
                node_mask = label_img == node
                label_voxels = int(node_mask.sum())
                if mask_img is not None and shape_match and label_voxels:
                    overlap_voxels = int((node_mask & mask_img).sum())
                    overlap_frac = overlap_voxels / label_voxels
                else:
                    overlap_voxels = np.nan
                    overlap_frac = np.nan
                if label_voxels < MIN_LABEL_VOXELS:
                    source_status = "label_tiny_or_absent"
                elif mask_img is None:
                    source_status = "missing_or_unreadable_5tt_mask"
                elif not shape_match:
                    source_status = "geometry_mismatch"
                elif not np.isfinite(overlap_frac) or overlap_frac < MIN_MASK_OVERLAP_FRAC:
                    source_status = "poor_5tt_overlap"
                else:
                    source_status = "coverage_ok"
                rows.append(
                    {
                        "sid": sid,
                        "subject_id": subject_id,
                        "node": node,
                        "atlas_label": _node_label(node, atlas),
                        "AAL_b0_path": str(parc_path),
                        "mask_5tt_path": str(mask_path),
                        "AAL_b0_shape": "x".join(map(str, label_img.shape)),
                        "mask_5tt_shape": "x".join(map(str, mask_img.shape)) if mask_img is not None else "",
                        "geometry_match": int(shape_match),
                        "label_voxels": label_voxels,
                        "mask_overlap_voxels": overlap_voxels,
                        "mask_overlap_frac": overlap_frac,
                        "source_status": source_status,
                        "mask_read_error": mask_error,
                    }
                )
        except Exception as exc:
            for node in nodes:
                rows.append(
                    {
                        "sid": sid,
                        "subject_id": subject_id,
                        "node": node,
                        "atlas_label": _node_label(node, atlas),
                        "AAL_b0_path": str(parc_path),
                        "mask_5tt_path": str(mask_path),
                        "source_status": "read_error_AAL_b0",
                        "mask_read_error": f"{type(exc).__name__}: {exc}",
                    }
                )
    return pd.DataFrame(rows)


def _read_assignment_counts(path: Path) -> tuple[Counter[int], int]:
    try:
        data = np.loadtxt(path, comments="#", dtype=np.int32, ndmin=2)
        if data.size == 0:
            return Counter(), 0
        labels = data[:, :2].reshape(-1)
        labels = labels[np.isfinite(labels)]
        labels = labels.astype(np.int32, copy=False)
        bincount = np.bincount(labels[labels >= 0])
        counts = Counter({idx: int(val) for idx, val in enumerate(bincount) if val})
        return counts, int(data.shape[0])
    except Exception:
        counts: Counter[int] = Counter()
        n_lines = 0
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for raw in handle:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                parts = re.split(r"[,\s]+", line)
                labels: list[int] = []
                for part in parts[:2]:
                    try:
                        labels.append(int(float(part)))
                    except ValueError:
                        continue
                if not labels:
                    continue
                n_lines += 1
                counts.update(labels)
        return counts, n_lines


def audit_streamline_assignments(
    paths: AnalysisPaths,
    atlas: AtlasInfo,
    target_nodes_by_sid: dict[str, set[int]],
    max_subjects: int | None = None,
) -> pd.DataFrame:
    rows: list[dict] = []
    sids = sorted(target_nodes_by_sid)
    if max_subjects is not None:
        sids = sids[:max_subjects]
    for sid in sids:
        subject_id = sid_to_subject_id(sid)
        assign_path = paths.deriv_root / "tracks" / sid / "assignments_aal.csv"
        nodes = sorted(n for n in target_nodes_by_sid[sid] if n in atlas.valid_nodes)
        if not nodes:
            continue
        if not assign_path.exists():
            for node in nodes:
                rows.append(
                    {
                        "sid": sid,
                        "subject_id": subject_id,
                        "node": node,
                        "atlas_label": _node_label(node, atlas),
                        "assignments_path": str(assign_path),
                        "assignment_status": "missing_assignments",
                    }
                )
            continue
        try:
            counts, n_streamlines = _read_assignment_counts(assign_path)
            endpoint_total = max(2 * n_streamlines, 1)
            unassigned_endpoint_frac = counts.get(0, 0) / endpoint_total
            for node in nodes:
                endpoint_count = int(counts.get(node, 0))
                rows.append(
                    {
                        "sid": sid,
                        "subject_id": subject_id,
                        "node": node,
                        "atlas_label": _node_label(node, atlas),
                        "assignments_path": str(assign_path),
                        "n_assignment_rows": n_streamlines,
                        "unassigned_endpoint_count": int(counts.get(0, 0)),
                        "unassigned_endpoint_frac": unassigned_endpoint_frac,
                        "node_assignment_endpoint_count": endpoint_count,
                        "assignment_status": "zero_node_assignment" if endpoint_count == 0 else "assignment_ok",
                    }
                )
        except Exception as exc:
            for node in nodes:
                rows.append(
                    {
                        "sid": sid,
                        "subject_id": subject_id,
                        "node": node,
                        "atlas_label": _node_label(node, atlas),
                        "assignments_path": str(assign_path),
                        "assignment_status": "read_error_assignments",
                        "assignment_error": f"{type(exc).__name__}: {exc}",
                    }
                )
    return pd.DataFrame(rows)


def build_frontal_opercular_qc(
    matrix_audit: pd.DataFrame,
    coverage: pd.DataFrame,
    assignments: pd.DataFrame,
    atlas: AtlasInfo,
) -> pd.DataFrame:
    rows: list[dict] = []
    primary = matrix_audit[matrix_audit["weight"].isin([PRIMARY_WEIGHT, "count", "len_mean"])].copy()
    if primary.empty:
        return pd.DataFrame()
    by_key = {(r.sid, r.weight): r for r in primary.itertuples(index=False)}
    cov = coverage[coverage["node"].isin(FRONTAL_OPERCULAR_NODES)].copy()
    asg = assignments[assignments["node"].isin(FRONTAL_OPERCULAR_NODES)].copy()
    cov_lookup = {(r.sid, int(r.node)): r for r in cov.itertuples(index=False)}
    asg_lookup = {(r.sid, int(r.node)): r for r in asg.itertuples(index=False)}
    for sid in sorted(set(primary["sid"])):
        for node in FRONTAL_OPERCULAR_NODES:
            row = {
                "sid": sid,
                "subject_id": sid_to_subject_id(sid),
                "node": node,
                "atlas_label": _node_label(node, atlas),
            }
            for weight in [PRIMARY_WEIGHT, "count", "len_mean"]:
                rec = by_key.get((sid, weight))
                zero_rows = set()
                if rec is not None and isinstance(rec.zero_rows, str):
                    zero_rows = {int(x) for x in rec.zero_rows.split(";") if x.strip().isdigit()}
                    row["group"] = getattr(rec, "group", np.nan)
                row[f"{weight}_row_zero"] = int(node in zero_rows)
            cov_rec = cov_lookup.get((sid, node))
            if cov_rec is not None:
                for col in [
                    "label_voxels",
                    "mask_overlap_voxels",
                    "mask_overlap_frac",
                    "source_status",
                    "geometry_match",
                ]:
                    row[col] = getattr(cov_rec, col, np.nan)
            asg_rec = asg_lookup.get((sid, node))
            if asg_rec is not None:
                row["node_assignment_endpoint_count"] = getattr(asg_rec, "node_assignment_endpoint_count", np.nan)
                row["unassigned_endpoint_frac"] = getattr(asg_rec, "unassigned_endpoint_frac", np.nan)
                row["assignment_status"] = getattr(asg_rec, "assignment_status", "")
            if row.get(f"{PRIMARY_WEIGHT}_row_zero", 0) == 0:
                cause = "not_zero_in_primary_matrix"
            elif row.get("source_status") in {"label_tiny_or_absent"}:
                cause = "FAIL_LABEL_COVERAGE"
            elif row.get("source_status") in {"poor_5tt_overlap", "geometry_mismatch"}:
                cause = "FAIL_REGISTRATION"
            elif row.get("assignment_status") == "zero_node_assignment":
                cause = "FAIL_STREAMLINE_ASSIGNMENT"
            elif row.get("assignment_status") == "missing_assignments":
                cause = "FAIL_SOURCE"
            else:
                cause = "WARN_SPARSE_NODE"
            row["suspected_cause"] = cause
            rows.append(row)
    return pd.DataFrame(rows)


def classify_subjects(
    matrix_audit: pd.DataFrame,
    coverage: pd.DataFrame,
    assignments: pd.DataFrame,
    frontal_qc: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict] = []
    primary = matrix_audit[matrix_audit["weight"] == PRIMARY_WEIGHT].copy()
    if primary.empty:
        return pd.DataFrame()
    cov_by_sid = {sid: df for sid, df in coverage.groupby("sid")} if not coverage.empty else {}
    asg_by_sid = {sid: df for sid, df in assignments.groupby("sid")} if not assignments.empty else {}
    frontal_by_sid = {sid: df for sid, df in frontal_qc.groupby("sid")} if not frontal_qc.empty else {}
    for rec in primary.itertuples(index=False):
        sid = rec.sid
        reasons: list[str] = []
        status = "PASS"
        read_error = getattr(rec, "read_error", "")
        has_read_error = isinstance(read_error, str) and bool(read_error.strip())
        if has_read_error:
            status = "FAIL_SOURCE"
            reasons.append(f"primary matrix read error: {read_error}")
        fr = frontal_by_sid.get(sid, pd.DataFrame())
        if status == "PASS" and not fr.empty:
            frontal_zero = fr[fr[f"{PRIMARY_WEIGHT}_row_zero"] == 1].copy()
            if (frontal_zero["suspected_cause"] == "FAIL_REGISTRATION").any():
                status = "FAIL_REGISTRATION"
                labels = frontal_zero.loc[
                    frontal_zero["suspected_cause"] == "FAIL_REGISTRATION", "atlas_label"
                ].astype(str).tolist()
                reasons.append(f"frontal opercular labels are zero and poorly overlap 5TT mask: {', '.join(labels)}")
            elif (frontal_zero["suspected_cause"] == "FAIL_LABEL_COVERAGE").any():
                status = "FAIL_LABEL_COVERAGE"
                labels = frontal_zero.loc[
                    frontal_zero["suspected_cause"] == "FAIL_LABEL_COVERAGE", "atlas_label"
                ].astype(str).tolist()
                reasons.append(f"frontal opercular labels are zero and have implausible label coverage: {', '.join(labels)}")
            elif (frontal_zero["suspected_cause"] == "FAIL_SOURCE").any():
                status = "FAIL_SOURCE"
                labels = frontal_zero.loc[
                    frontal_zero["suspected_cause"] == "FAIL_SOURCE", "atlas_label"
                ].astype(str).tolist()
                reasons.append(f"frontal opercular assignment/source evidence is missing: {', '.join(labels)}")
            elif (frontal_zero["suspected_cause"] == "FAIL_STREAMLINE_ASSIGNMENT").any():
                status = "FAIL_STREAMLINE_ASSIGNMENT"
                labels = frontal_zero.loc[
                    frontal_zero["suspected_cause"] == "FAIL_STREAMLINE_ASSIGNMENT", "atlas_label"
                ].astype(str).tolist()
                reasons.append(f"frontal opercular labels have coverage but zero assignments: {', '.join(labels)}")
        if status == "PASS" and getattr(rec, "n_unexpected_zero_rows", 0) >= WARN_VALID_ZERO_ROWS:
            status = "WARN_SPARSE_NODE"
            reasons.append(f"{int(rec.n_unexpected_zero_rows)} non-gap zero rows in {PRIMARY_WEIGHT} matrix")
        if not reasons:
            reasons.append("primary SC matrix passes structural integrity and targeted frontal opercular QC")
        action = {
            "PASS": "include in analysis",
            "WARN_SPARSE_NODE": "include provisionally; manually review sparse nodes before final inference",
            "FAIL_SOURCE": "exclude; restore or rebuild missing/corrupt upstream DWI/T1/FOD/tracks artifacts before rerun",
            "FAIL_LABEL_COVERAGE": (
                "exclude; do not run blind downstream post repair. First test raw/reference AAL/MNI->native-T1->B0 "
                "nearest-neighbor replay and promote only after canary validation"
            ),
            "FAIL_REGISTRATION": (
                "exclude; rerun or replay the upstream T1-to-B0/BBR source contract in scratch before any Step 7 repair"
            ),
            "FAIL_STREAMLINE_ASSIGNMENT": (
                "exclude; run scratch assignment/track-contract replay only after parcellation and mask overlap pass QC"
            ),
        }.get(status, "manual review")
        rows.append(
            {
                "sid": sid,
                "subject_id": rec.subject_id,
                "group": getattr(rec, "group", np.nan),
                "sc_matrix_qc_status": status,
                "analysis_gate": "include" if status in {"PASS", "WARN_SPARSE_NODE"} else "exclude_pending_repair",
                "recommended_repair_stage": action,
                "n_unexpected_zero_rows": getattr(rec, "n_unexpected_zero_rows", np.nan),
                "density": getattr(rec, "density", np.nan),
                "frontal_oper_l_zero": getattr(rec, "frontal_oper_l_zero", np.nan),
                "frontal_oper_r_zero": getattr(rec, "frontal_oper_r_zero", np.nan),
                "decision_reasons": " | ".join(reasons),
            }
        )
    return pd.DataFrame(rows)


def _parse_node_text(value: object) -> set[int]:
    if not isinstance(value, str) or not value.strip():
        return set()
    out: set[int] = set()
    for item in value.split(";"):
        item = item.strip()
        if not item:
            continue
        try:
            out.add(int(float(item)))
        except ValueError:
            continue
    return out


def _component_stats(matrix_path: Path, atlas: AtlasInfo) -> dict[str, float | int | str]:
    try:
        matrix = _read_matrix(matrix_path)
        clean = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    except Exception as exc:
        return {
            "component_read_error": f"{type(exc).__name__}: {exc}",
            "n_components": np.nan,
            "largest_component_nodes": np.nan,
            "largest_component_frac": np.nan,
            "n_isolated_valid_nodes": np.nan,
        }
    n = min(clean.shape)
    valid_nodes = sorted(node for node in atlas.valid_nodes if 1 <= node <= n)
    valid_set = set(valid_nodes)
    adjacency: dict[int, set[int]] = {node: set() for node in valid_nodes}
    for i in valid_nodes:
        row = clean[i - 1, :n]
        nz = np.flatnonzero(np.abs(row) > 0)
        for j0 in nz:
            j = int(j0 + 1)
            if j != i and j in valid_set:
                adjacency[i].add(j)
                adjacency[j].add(i)
    isolated = [node for node, neighbors in adjacency.items() if not neighbors]
    seen: set[int] = set()
    component_sizes: list[int] = []
    for node in valid_nodes:
        if node in seen:
            continue
        queue: deque[int] = deque([node])
        seen.add(node)
        size = 0
        while queue:
            current = queue.popleft()
            size += 1
            for neighbor in adjacency[current]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append(neighbor)
        component_sizes.append(size)
    largest = max(component_sizes) if component_sizes else 0
    return {
        "component_read_error": "",
        "n_components": len(component_sizes),
        "largest_component_nodes": largest,
        "largest_component_frac": largest / len(valid_nodes) if valid_nodes else np.nan,
        "n_isolated_valid_nodes": len(isolated),
    }


def _lane_from_evidence(
    *,
    read_error: str,
    n_zero_nodes: int,
    density: float,
    label_fail_count: int,
    registration_fail_count: int,
    assignment_fail_count: int,
) -> tuple[str, str, str]:
    if read_error:
        return "FAIL_SOURCE", "source_failure", f"primary matrix read error: {read_error}"
    if n_zero_nodes <= 0 and (not np.isfinite(density) or density >= 0.15):
        return "PASS", "analysis_include", "whole-matrix structure passes density and zero-node checks"
    min_registration = max(2, math.ceil(0.10 * max(n_zero_nodes, 1)))
    min_label = max(3, math.ceil(0.15 * max(n_zero_nodes, 1)))
    min_assignment = max(3, math.ceil(0.15 * max(n_zero_nodes, 1)))
    if registration_fail_count >= min_registration:
        return (
            "FAIL_REGISTRATION",
            "bbr_registration",
            f"{registration_fail_count}/{n_zero_nodes} zero valid nodes have poor AAL_b0-to-5TT overlap",
        )
    if label_fail_count >= min_label:
        return (
            "FAIL_LABEL_COVERAGE",
            "post_parcellation",
            f"{label_fail_count}/{n_zero_nodes} zero valid nodes are absent or implausibly tiny in AAL_b0",
        )
    if assignment_fail_count >= min_assignment:
        return (
            "FAIL_STREAMLINE_ASSIGNMENT",
            "assignment_or_tracks",
            f"{assignment_fail_count}/{n_zero_nodes} zero valid nodes have coverage but no streamline assignments",
        )
    if n_zero_nodes >= WARN_VALID_ZERO_ROWS or (np.isfinite(density) and density < 0.15):
        return (
            "WARN_SPARSE_NODE",
            "manual_review_sparse",
            f"{n_zero_nodes} non-gap zero nodes or low density require visual review",
        )
    return "PASS", "analysis_include", "whole-matrix structure passes repair-lane checks"


def build_whole_matrix_repair_plan(
    matrix_audit: pd.DataFrame,
    coverage: pd.DataFrame,
    assignments: pd.DataFrame,
    decisions: pd.DataFrame,
    atlas: AtlasInfo,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    primary = matrix_audit[matrix_audit["weight"] == PRIMARY_WEIGHT].copy()
    if primary.empty:
        empty = pd.DataFrame()
        return empty, empty, empty
    coverage_by_sid = {sid: df for sid, df in coverage.groupby("sid")} if not coverage.empty else {}
    assignments_by_sid = {sid: df for sid, df in assignments.groupby("sid")} if not assignments.empty else {}
    decision_lookup = {
        r.sid: r
        for r in decisions.itertuples(index=False)
    } if not decisions.empty else {}

    rows: list[dict] = []
    for rec in primary.itertuples(index=False):
        sid = str(rec.sid)
        zero_nodes = sorted(_parse_node_text(getattr(rec, "unexpected_zero_rows", "")))
        zero_node_set = set(zero_nodes)
        cov = coverage_by_sid.get(sid, pd.DataFrame())
        asg = assignments_by_sid.get(sid, pd.DataFrame())
        if not cov.empty and zero_node_set:
            cov = cov[cov["node"].isin(zero_node_set)].copy()
        if not asg.empty and zero_node_set:
            asg = asg[asg["node"].isin(zero_node_set)].copy()

        source_counts = cov["source_status"].value_counts(dropna=False).to_dict() if not cov.empty else {}
        label_fail_count = int(source_counts.get("label_tiny_or_absent", 0))
        registration_fail_count = int(
            source_counts.get("poor_5tt_overlap", 0)
            + source_counts.get("geometry_mismatch", 0)
            + source_counts.get("missing_or_unreadable_5tt_mask", 0)
        )
        assignment_fail_count = int((asg.get("assignment_status", pd.Series(dtype=str)) == "zero_node_assignment").sum()) if not asg.empty else 0
        coverage_ok_count = int(source_counts.get("coverage_ok", 0))
        read_error = str(getattr(rec, "read_error", "") or "").strip()
        density = float(getattr(rec, "density", np.nan))
        status, lane, reason = _lane_from_evidence(
            read_error=read_error,
            n_zero_nodes=len(zero_nodes),
            density=density,
            label_fail_count=label_fail_count,
            registration_fail_count=registration_fail_count,
            assignment_fail_count=assignment_fail_count,
        )
        component = _component_stats(Path(getattr(rec, "matrix_path")), atlas)
        decision = decision_lookup.get(sid)
        severity_score = (
            len(zero_nodes)
            + 100.0 * max(0.0, 0.20 - density if np.isfinite(density) else 0.20)
            + 0.25 * float(component.get("n_isolated_valid_nodes", 0) or 0)
        )
        rows.append(
            {
                "sid": sid,
                "subject_id": getattr(rec, "subject_id", sid_to_subject_id(sid)),
                "group": getattr(rec, "group", np.nan),
                "whole_matrix_qc_status": status,
                "repair_lane": lane,
                "analysis_gate": "include" if status in {"PASS", "WARN_SPARSE_NODE"} else "exclude_pending_repair",
                "include_in_analysis": int(status in {"PASS", "WARN_SPARSE_NODE"}),
                "repair_priority": round(float(severity_score), 3),
                "recommended_action": REPAIR_LANE_ACTIONS.get(lane, "manual review"),
                "whole_matrix_reason": reason,
                "previous_targeted_status": getattr(decision, "sc_matrix_qc_status", ""),
                "previous_targeted_reason": getattr(decision, "decision_reasons", ""),
                "density": density,
                "nonzero_upper_edges": getattr(rec, "nonzero_upper_edges", np.nan),
                "n_unexpected_zero_rows": len(zero_nodes),
                "unexpected_zero_nodes": _list_to_text(zero_nodes),
                "unexpected_zero_node_labels": ";".join(_node_label(node, atlas) for node in zero_nodes),
                "n_label_tiny_or_absent_zero_nodes": label_fail_count,
                "n_poor_overlap_zero_nodes": registration_fail_count,
                "n_coverage_ok_zero_nodes": coverage_ok_count,
                "n_zero_assignment_zero_nodes": assignment_fail_count,
                "median_mask_overlap_zero_nodes": float(cov["mask_overlap_frac"].median()) if "mask_overlap_frac" in cov and len(cov) else np.nan,
                "median_assignment_endpoints_zero_nodes": float(asg["node_assignment_endpoint_count"].median()) if "node_assignment_endpoint_count" in asg and len(asg) else np.nan,
                "unassigned_endpoint_frac": float(asg["unassigned_endpoint_frac"].median()) if "unassigned_endpoint_frac" in asg and len(asg) else np.nan,
                **component,
            }
        )

    repair_plan = pd.DataFrame(rows).sort_values(
        ["include_in_analysis", "repair_lane", "repair_priority", "sid"],
        ascending=[True, True, False, True],
    )
    repair_targets = repair_plan[repair_plan["analysis_gate"] == "exclude_pending_repair"].copy()
    gate = repair_plan[
        [
            "sid",
            "subject_id",
            "group",
            "whole_matrix_qc_status",
            "repair_lane",
            "analysis_gate",
            "include_in_analysis",
            "density",
            "n_unexpected_zero_rows",
            "whole_matrix_reason",
        ]
    ].sort_values(["include_in_analysis", "group", "sid"], ascending=[False, True, True])
    return repair_plan, repair_targets, gate


def build_zero_pattern_clusters(repair_plan: pd.DataFrame, max_nodes: int = 20) -> pd.DataFrame:
    if repair_plan.empty:
        return pd.DataFrame()
    rows: list[dict] = []
    for rec in repair_plan.itertuples(index=False):
        nodes = _parse_node_text(getattr(rec, "unexpected_zero_nodes", ""))
        signature = ";".join(str(node) for node in sorted(nodes)[:max_nodes])
        labels = str(getattr(rec, "unexpected_zero_node_labels", "") or "").split(";")[:max_nodes]
        rows.append(
            {
                "sid": rec.sid,
                "group": rec.group,
                "whole_matrix_qc_status": rec.whole_matrix_qc_status,
                "repair_lane": rec.repair_lane,
                "zero_signature": signature,
                "zero_signature_labels": ";".join(labels),
            }
        )
    df = pd.DataFrame(rows)
    grouped = (
        df.groupby(["zero_signature", "zero_signature_labels", "repair_lane"], dropna=False)
        .agg(
            n_subjects=("sid", "nunique"),
            groups=("group", lambda s: ";".join(sorted(set(str(x) for x in s if pd.notna(x))))),
            statuses=("whole_matrix_qc_status", lambda s: ";".join(sorted(set(map(str, s))))),
            example_subjects=("sid", lambda s: ";".join(list(map(str, s))[:10])),
        )
        .reset_index()
        .sort_values(["n_subjects", "repair_lane"], ascending=[False, True])
    )
    return grouped


def _decision_file_text(out_dir: Path) -> str:
    decision_path = out_dir / FINAL_RECTIFICATION_DECISION_NAME
    if decision_path.exists():
        return f"Final rectification decision file: `{decision_path}`."
    qc_root_decision = out_dir.parent / FINAL_RECTIFICATION_DECISION_NAME
    if qc_root_decision.exists():
        return f"Final rectification decision file: `{qc_root_decision}`."
    return "Final rectification decision file not found in the current QC output tree."


def write_final_qc_recommendations(
    out_dir: Path,
    decisions: pd.DataFrame,
    repair_plan: pd.DataFrame,
    analysis_gate: pd.DataFrame,
) -> Path:
    status_counts = (
        decisions["sc_matrix_qc_status"].value_counts(dropna=False).to_dict()
        if not decisions.empty and "sc_matrix_qc_status" in decisions
        else {}
    )
    lane_counts = (
        repair_plan["repair_lane"].value_counts(dropna=False).to_dict()
        if not repair_plan.empty and "repair_lane" in repair_plan
        else {}
    )
    gate_counts = (
        analysis_gate["analysis_gate"].value_counts(dropna=False).to_dict()
        if not analysis_gate.empty and "analysis_gate" in analysis_gate
        else {}
    )

    def _count_lines(title: str, counts: dict) -> list[str]:
        lines = [f"## {title}", ""]
        if not counts:
            lines.append("- no rows")
        else:
            for key, value in sorted(counts.items(), key=lambda item: str(item[0])):
                lines.append(f"- {key}: {value}")
        lines.append("")
        return lines

    lines = [
        "# SC Matrix QC Final Recommendations",
        "",
        FINAL_RECTIFICATION_GUIDANCE,
        "",
        _decision_file_text(out_dir),
        "",
        "## Operational decision",
        "",
        "- Do not run a full downstream post-parcellation / registration / assignment batch repair from current Step 7 artifacts.",
        "- Include only `PASS` and explicitly reviewed `WARN_SPARSE_NODE` subjects in downstream analysis.",
        "- Treat `FAIL_*` subjects as excluded until a raw upstream supervisor-route replay or full preprocessing rerun passes canary validation.",
        "- If a future replay succeeds, validate every promoted subject against density, valid zero rows, label survival, required matrices, symmetry, finite values, and zero diagonal.",
        "",
        "## What changed from earlier repair manifests",
        "",
        "Earlier manifests classify likely failure lanes, but they are not authorization to batch rerun those lanes. Recent scratch canaries showed that current downstream artifacts do not generalize into a safe repair route.",
        "",
    ]
    lines += _count_lines("Targeted QC Status Counts", status_counts)
    lines += _count_lines("Whole-Matrix Repair Lane Counts", lane_counts)
    lines += _count_lines("Analysis Gate Counts", gate_counts)
    lines += [
        "## Next Valid Salvage Path",
        "",
        "Run a scratch-only raw upstream replay that follows the supervisor contract for a small matched panel: raw DWI/T1 preprocessing, T1-to-B0 registration, AAL/MNI to native T1 to B0 with nearest-neighbour interpolation, tractography, SIFT2, connectome generation, and strict matrix validation. Promote nothing unless the canary passes.",
        "",
    ]

    path = out_dir / "sc_matrix_qc_final_recommendations.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def run_sc_matrix_qc(
    paths: AnalysisPaths | None = None,
    output_dir: str | Path | None = None,
    weights: Iterable[str] = MATRIX_WEIGHTS,
    max_subjects: int | None = None,
) -> dict[str, pd.DataFrame]:
    paths = paths or get_analysis_paths(
        notebook_dir=Path("/home/ec2-user/exp"),
        deriv_root=Path("/home/ec2-user/exp/data/derivatives"),
        cohort_dti_csv=Path("/home/ec2-user/exp/cohort/dti.csv"),
        cohort_mri_csv=Path("/home/ec2-user/exp/cohort/mri.csv"),
    )
    out_dir = Path(output_dir) if output_dir else paths.qc_dir / "sc_matrix_qc"
    out_dir.mkdir(parents=True, exist_ok=True)

    atlas = load_aal3_labels(paths)
    matrix_audit, zero_summary, target_nodes_by_sid = audit_sc_matrices(paths, atlas, weights=weights)
    if max_subjects is not None:
        keep_sids = sorted(target_nodes_by_sid)[:max_subjects]
        target_nodes_by_sid = {sid: target_nodes_by_sid[sid] for sid in keep_sids}

    coverage = audit_parcellation_coverage(paths, atlas, target_nodes_by_sid, max_subjects=max_subjects)
    assignments = audit_streamline_assignments(paths, atlas, target_nodes_by_sid, max_subjects=max_subjects)
    frontal_qc = build_frontal_opercular_qc(matrix_audit, coverage, assignments, atlas)
    decisions = classify_subjects(matrix_audit, coverage, assignments, frontal_qc)
    repair_targets = decisions[decisions["analysis_gate"] == "exclude_pending_repair"].copy()
    whole_repair_plan, whole_repair_targets, analysis_gate = build_whole_matrix_repair_plan(
        matrix_audit,
        coverage,
        assignments,
        decisions,
        atlas,
    )
    zero_clusters = build_zero_pattern_clusters(whole_repair_plan)

    outputs = {
        "sc_matrix_integrity_subjects": matrix_audit,
        "sc_zero_node_summary": zero_summary,
        "parc_label_coverage_qc": coverage,
        "streamline_assignment_qc": assignments,
        "sc_frontal_opercular_qc": frontal_qc,
        "sc_matrix_qc_decisions": decisions,
        "sc_matrix_qc_repair_targets": repair_targets,
        "sc_matrix_qc_repair_plan": whole_repair_plan,
        "sc_matrix_qc_whole_matrix_repair_targets": whole_repair_targets,
        "sc_matrix_qc_analysis_gate": analysis_gate,
        "sc_zero_pattern_clusters": zero_clusters,
        "post_parcellation_repair_targets": whole_repair_plan[whole_repair_plan["repair_lane"] == "post_parcellation"].copy(),
        "bbr_registration_repair_targets": whole_repair_plan[whole_repair_plan["repair_lane"] == "bbr_registration"].copy(),
        "assignment_or_tracks_repair_targets": whole_repair_plan[whole_repair_plan["repair_lane"] == "assignment_or_tracks"].copy(),
        "source_failure_repair_targets": whole_repair_plan[whole_repair_plan["repair_lane"] == "source_failure"].copy(),
    }
    for name, df in outputs.items():
        df.to_csv(out_dir / f"{name}.csv", index=False)

    manifest = pd.DataFrame(
        [
            {
                "output": name,
                "path": str(out_dir / f"{name}.csv"),
                "rows": int(len(df)),
                "columns": int(len(df.columns)),
            }
            for name, df in outputs.items()
        ]
    )
    manifest.to_csv(out_dir / "sc_matrix_qc_manifest.csv", index=False)
    final_recommendations_path = write_final_qc_recommendations(out_dir, decisions, whole_repair_plan, analysis_gate)
    manifest_extra = pd.DataFrame(
        [
            {
                "output": "sc_matrix_qc_final_recommendations",
                "path": str(final_recommendations_path),
                "rows": 0,
                "columns": 0,
            }
        ]
    )
    manifest = pd.concat([manifest, manifest_extra], ignore_index=True)
    manifest.to_csv(out_dir / "sc_matrix_qc_manifest.csv", index=False)
    outputs["manifest"] = manifest
    return outputs


if __name__ == "__main__":
    result = run_sc_matrix_qc()
    decisions = result["sc_matrix_qc_decisions"]
    print("SC matrix QC complete")
    print(f"subjects={decisions['sid'].nunique() if not decisions.empty else 0}")
    if not decisions.empty:
        print(decisions["sc_matrix_qc_status"].value_counts(dropna=False).to_string())
    repair_plan = result.get("sc_matrix_qc_repair_plan", pd.DataFrame())
    if not repair_plan.empty:
        print("\nwhole-matrix repair gate")
        print(repair_plan["whole_matrix_qc_status"].value_counts(dropna=False).to_string())
        print("\nrepair lanes")
        print(repair_plan["repair_lane"].value_counts(dropna=False).to_string())
