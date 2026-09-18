from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def write_records_csv(records: list[dict], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(path, index=False)
    return path


def write_json(data: object, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def append_markdown_section(path: str | Path, title: str, lines: list[str]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n## {title}\n\n")
        for line in lines:
            handle.write(f"{line}\n")
    return path


def write_manifest_pair(
    *,
    ready_records: list[dict],
    quarantine_records: list[dict],
    out_dir: str | Path,
) -> dict[str, str]:
    out_dir = Path(out_dir)
    ready = write_records_csv(ready_records, out_dir / "analysis_ready_manifest.csv")
    quarantine = write_records_csv(quarantine_records, out_dir / "quarantine_manifest.csv")
    return {"analysis_ready_manifest": str(ready), "quarantine_manifest": str(quarantine)}
