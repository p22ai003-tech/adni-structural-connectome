#!/usr/bin/env python
"""Repair ROI names in the 17_edr_exceptions artifacts.

The artifacts in that section were written with ROI names looked up from
``atlas/AAL/AAL3_labels.csv``. That table is keyed on the original AAL3 atlas
value, and AAL3v1 leaves values 35, 36, 81 and 82 unused. The 166-node
connectome compacts the surviving regions, so matrix row 37 is atlas value 39.
Names taken from that table are therefore shifted from row 37 onward, and 132
of the 166 rows carry a wrong name: what is printed as ``Cingulate_Post_L`` is
in fact ``Hippocampus_L``.

Only the name columns are wrong. Every node index, count, rate and p-value in
these files is correct, and every network-level result is correct as well,
because ``AAL3_network_mapping.csv`` is keyed on the matrix index and every
consumer joins on the index rather than the name. So this is a relabel, not a
recomputation: it is exactly what re-running the stage would produce for these
columns, at none of the cost.

``analysis_edr_exceptions.py`` has been fixed, so a future run is correct
without this tool. Use it to repair artifacts already on disk.

    python -m connectome_analysis.relabel_edr_rois            # report only
    python -m connectome_analysis.relabel_edr_rois --apply    # rewrite, with backups
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import sc_config  # noqa: E402

# file -> [(index column, name column), ...]
TARGETS: dict[str, list[tuple[str, str]]] = {
    "edr_exception_node_level_fd_sum_len_mean.csv": [("node", "atlas_label")],
    "edr_exception_edge_level_fd_sum_len_mean.csv": [("i", "roi_a"), ("j", "roi_b")],
    "edr_exception_edge_group_map_fd_sum_len_mean.csv": [("i", "roi_a"), ("j", "roi_b")],
    "edr_exception_pairwise_roi_rankings.csv": [("node", "atlas_label")],
    "edr_exception_repeated_top_regions.csv": [("node", "atlas_label")],
}


def load_lut() -> dict[int, str]:
    p = sc_config.paths().aal_node_map
    if not p.is_file():
        raise SystemExit(f"correct LUT not found: {p}")
    m = pd.read_csv(p)
    return dict(zip(m["new_id"].astype(int), m["name"].astype(str)))


def _edge_index_base(df: pd.DataFrame, cols: list[tuple[str, str]]) -> int:
    """Edge tables store i/j as 0-based positions; node tables are 1-based.

    Guessing wrong would shift every name by one, so infer it from the data
    rather than assume: a 0-based table contains a 0, a 1-based one reaches n.
    """
    idx_col = cols[0][0]
    lo = pd.to_numeric(df[idx_col], errors="coerce").min()
    return 0 if lo == 0 else 1


def process(root: Path, lut: dict[int, str], apply: bool) -> int:
    changed_total = 0
    for fname, cols in TARGETS.items():
        path = root / "17_edr_exceptions" / fname
        if not path.is_file():
            print(f"  {fname:<52} (absent)")
            continue
        df = pd.read_csv(path)
        base = _edge_index_base(df, cols)
        changed = 0
        examples: list[str] = []
        for idx_col, name_col in cols:
            if idx_col not in df.columns or name_col not in df.columns:
                print(f"  {fname:<52} missing {idx_col}/{name_col}; skipped")
                continue
            node = pd.to_numeric(df[idx_col], errors="coerce") + (1 - base)
            correct = node.map(lut)
            differs = correct.notna() & (df[name_col].astype(str) != correct)
            changed += int(differs.sum())
            for _, row in df.loc[differs, [idx_col, name_col]].head(2).iterrows():
                n = int(row[idx_col]) + (1 - base)
                examples.append(f"{row[name_col]} -> {lut[n]} (row {n})")
            if apply:
                df.loc[differs, name_col] = correct[differs]
        changed_total += changed
        state = "would fix" if not apply else "fixed"
        cells = len(df) * len(cols)
        print(f"  {fname:<52} {state} {changed:>6} of {cells} name cells "
              f"({len(df)} rows x {len(cols)} col)")
        for e in examples[:2]:
            print(f"        {e}")
        if apply and changed:
            backup = path.with_suffix(path.suffix + ".stale_lut.bak")
            if not backup.exists():
                shutil.copy2(path, backup)
            df.to_csv(path, index=False)
    return changed_total


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--analysis-root", type=Path, default=None)
    ap.add_argument("--apply", action="store_true",
                    help="rewrite the files (originals kept as *.stale_lut.bak)")
    args = ap.parse_args(argv)

    root = args.analysis_root or sc_config.paths().analysis_root
    lut = load_lut()
    print(f"analysis root : {root}")
    print(f"LUT           : {sc_config.paths().aal_node_map}")
    print(f"mode          : {'APPLY' if args.apply else 'report only'}\n")
    total = process(Path(root), lut, args.apply)
    print(f"\n{'fixed' if args.apply else 'would fix'} {total} name cells")
    if not args.apply and total:
        print("re-run with --apply to rewrite them")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
