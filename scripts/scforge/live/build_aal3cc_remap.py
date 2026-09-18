#!/usr/bin/env python3
"""Apply an explicit AAL3 connectome-compatible remapping.

This is the implementation layer for the AAL3-CC recommendation. The node-QC
builder writes a merge template; this script applies a reviewed TSV map to a
parcellation and, optionally, to an existing full AAL3 matrix.

Required TSV columns:
  old_id, old_name, new_id, new_name, action

Rows with blank/zero new_id are excluded. This is intentional: recurrent
zero-row labels should not be silently forced into production without review.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

ROOT = Path("/home/ec2-user/exp")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.scforge.live.run_sc_aal3_source_contract_probe import matrix_array  # noqa: E402


DEFAULT_MAP = Path("/home/ec2-user/exp/reports/aal3_node_qc/aal3_connectome_compatible_merge_template_latest.tsv")
DEFAULT_OUT = Path("/home/ec2-user/exp/reports/aal3cc")


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_rows(path: Path, rows: list[dict[str, Any]], delimiter: str = ",") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter=delimiter)
        writer.writeheader()
        writer.writerows(rows)


def int_or_zero(value: Any) -> int:
    text = str(value or "").strip()
    if not text:
        return 0
    try:
        val = float(text)
    except ValueError:
        return 0
    if not math.isfinite(val):
        return 0
    return int(val)


def load_map(path: Path, compact_ids: bool) -> tuple[dict[int, int], dict[int, str], list[dict[str, Any]]]:
    rows = read_rows(path)
    raw_pairs: list[tuple[int, int, str]] = []
    audit: list[dict[str, Any]] = []
    for row in rows:
        old_id = int_or_zero(row.get("old_id"))
        raw_new_id = int_or_zero(row.get("new_id"))
        old_name = row.get("old_name") or f"AAL3_{old_id:03d}"
        new_name = row.get("new_name") or old_name
        action = (row.get("action") or "").strip().lower()
        if old_id <= 0:
            continue
        if raw_new_id <= 0 or action in {"exclude", "review"}:
            audit.append(
                {
                    "old_id": old_id,
                    "old_name": old_name,
                    "raw_new_id": raw_new_id,
                    "new_id": 0,
                    "new_name": "",
                    "action_applied": "exclude",
                    "input_action": row.get("action", ""),
                    "reason": row.get("reason", ""),
                }
            )
            continue
        raw_pairs.append((old_id, raw_new_id, new_name))

    if compact_ids:
        raw_to_compact = {raw_id: idx + 1 for idx, raw_id in enumerate(sorted({new for _, new, _ in raw_pairs}))}
    else:
        raw_to_compact = {raw_id: raw_id for _, raw_id, _ in raw_pairs}

    mapping: dict[int, int] = {}
    names: dict[int, str] = {}
    for old_id, raw_new_id, new_name in raw_pairs:
        new_id = raw_to_compact[raw_new_id]
        mapping[old_id] = new_id
        names.setdefault(new_id, new_name)
        audit.append(
            {
                "old_id": old_id,
                "old_name": "",
                "raw_new_id": raw_new_id,
                "new_id": new_id,
                "new_name": names[new_id],
                "action_applied": "keep_or_merge",
            }
        )
    audit.sort(key=lambda r: int(r["old_id"]))
    return mapping, names, audit


def remap_parcellation(in_path: Path, out_path: Path, mapping: dict[int, int]) -> dict[str, Any]:
    img = nib.load(str(in_path))
    data = np.rint(np.asanyarray(img.dataobj)).astype(np.int32)
    out = np.zeros(data.shape, dtype=np.int32)
    for old_id, new_id in mapping.items():
        out[data == old_id] = int(new_id)
    out_img = nib.Nifti1Image(out, img.affine, img.header)
    out_img.set_data_dtype(np.int32)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(out_img, str(out_path))
    labels, counts = np.unique(out, return_counts=True)
    nonzero = {int(label): int(count) for label, count in zip(labels, counts) if int(label) > 0}
    return {
        "parcellation_in": str(in_path),
        "parcellation_out": str(out_path),
        "output_labels": len(nonzero),
        "output_nonzero_voxels": int(sum(nonzero.values())),
    }


def remap_matrix(in_path: Path, out_path: Path, mapping: dict[int, int], names: dict[int, str]) -> dict[str, Any]:
    mat = np.asarray(matrix_array(in_path), dtype=float)
    mat = np.where(np.isfinite(mat), mat, 0.0)
    n_new = max(names) if names else max(mapping.values(), default=0)
    out = np.zeros((n_new, n_new), dtype=float)
    grouped: dict[int, list[int]] = defaultdict(list)
    for old_id, new_id in mapping.items():
        old_idx = old_id - 1
        if 0 <= old_idx < mat.shape[0]:
            grouped[new_id].append(old_idx)
    for new_i, old_i_list in grouped.items():
        for new_j, old_j_list in grouped.items():
            block = mat[np.ix_(old_i_list, old_j_list)]
            out[new_i - 1, new_j - 1] = float(np.nansum(block))
    out = (out + out.T) / 2.0
    np.fill_diagonal(out, 0.0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(out_path, out, delimiter=",", fmt="%.10g")
    triu = out[np.triu_indices_from(out, k=1)]
    return {
        "matrix_in": str(in_path),
        "matrix_out": str(out_path),
        "matrix_nodes": int(out.shape[0]),
        "matrix_nonzero_edges": int(np.count_nonzero(triu > 0)),
        "matrix_density": float(np.count_nonzero(triu > 0) / len(triu)) if len(triu) else 0.0,
    }


def write_lut(path: Path, names: dict[int, str]) -> None:
    rows = [{"new_id": new_id, "new_name": names[new_id]} for new_id in sorted(names)]
    write_rows(path, rows, delimiter="\t")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", type=Path, default=DEFAULT_MAP)
    parser.add_argument("--parcellation", type=Path, default=None)
    parser.add_argument("--matrix", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--prefix", default="AAL3CC")
    parser.add_argument("--compact-ids", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    stamp = utc().replace(":", "").replace("-", "")
    out_dir = args.out_dir
    mapping, names, audit = load_map(args.map, compact_ids=args.compact_ids)
    if not mapping:
        raise RuntimeError(f"no active keep/merge rows found in {args.map}")

    outputs: dict[str, Any] = {
        "generated_utc": utc(),
        "map": str(args.map),
        "active_old_labels": len(mapping),
        "output_nodes": len(set(mapping.values())),
        "compact_ids": args.compact_ids,
    }

    write_lut(out_dir / f"{args.prefix}_lut_{stamp}.tsv", names)
    write_lut(out_dir / f"{args.prefix}_lut_latest.tsv", names)
    write_rows(out_dir / f"{args.prefix}_remap_audit_{stamp}.csv", audit)
    write_rows(out_dir / f"{args.prefix}_remap_audit_latest.csv", audit)

    if args.parcellation:
        outputs.update(remap_parcellation(args.parcellation, out_dir / f"{args.prefix}_parcellation_{stamp}.nii.gz", mapping))
    if args.matrix:
        outputs.update(remap_matrix(args.matrix, out_dir / f"{args.prefix}_matrix_{stamp}.csv", mapping, names))

    summary_path = out_dir / f"{args.prefix}_remap_summary_{stamp}.json"
    latest_summary = out_dir / f"{args.prefix}_remap_summary_latest.json"
    text = json.dumps(outputs, indent=2)
    summary_path.write_text(text)
    latest_summary.write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
