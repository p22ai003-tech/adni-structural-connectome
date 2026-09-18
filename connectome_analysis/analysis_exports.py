from __future__ import annotations

from pathlib import Path
from typing import Mapping

import pandas as pd


def _safe_sheet_name(name: str) -> str:
    return (
        str(name)
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
        .replace("*", "_")
        .replace("?", "_")
        .replace("[", "_")
        .replace("]", "_")
    )[:31]


def write_excel_bundle(
    tables: Mapping[str, pd.DataFrame],
    out_path: str | Path,
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with pd.ExcelWriter(out_path) as writer:
            for sheet, df in tables.items():
                df.to_excel(writer, sheet_name=_safe_sheet_name(sheet), index=False)
        return out_path
    except ModuleNotFoundError as exc:
        if exc.name not in {"openpyxl", "xlsxwriter"}:
            raise
    except ImportError as exc:
        if "openpyxl" not in str(exc) and "xlsxwriter" not in str(exc):
            raise

    fallback_dir = out_path.with_suffix("")
    fallback_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for sheet, df in tables.items():
        safe_sheet = _safe_sheet_name(sheet)
        csv_path = fallback_dir / f"{safe_sheet}.csv"
        df.to_csv(csv_path, index=False)
        rows.append({"sheet": sheet, "csv": str(csv_path), "rows": len(df), "columns": len(df.columns)})
    manifest_path = out_path.with_suffix(".csv")
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    readme = out_path.with_suffix(".txt")
    export_text(
        [
            "Excel workbook fallback",
            "The current Python environment lacks openpyxl/xlsxwriter, so the workbook sheets were exported as CSV files.",
            f"CSV sheet directory: {fallback_dir}",
            f"Manifest: {manifest_path}",
        ],
        readme,
    )
    return manifest_path


def export_manifest(entries: list[dict], out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(entries).to_csv(out_path, index=False)
    return out_path


def export_text(lines: list[str], out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    return out_path


def collect_csv_exports(root: str | Path) -> pd.DataFrame:
    root = Path(root)
    rows = []
    for path in sorted(root.rglob("*.csv")):
        rows.append({"relative_path": str(path.relative_to(root)), "bytes": path.stat().st_size})
    return pd.DataFrame(rows)
