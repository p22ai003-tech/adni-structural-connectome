from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .serialization import json_safe, records
from .settings import Settings


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _file_descriptor(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {
            "path": str(path),
            "available": False,
        }
    stat = path.stat()
    return {
        "path": str(path),
        "available": True,
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "modified_utc": datetime.fromtimestamp(
            stat.st_mtime,
            tz=timezone.utc,
        ).isoformat(),
    }


def _present(path: Path | None) -> bool:
    """True when an optional source is configured and exists.

    The three HCP379 sources are None on any machine without the imaging
    server's run records (settings._opt). Calling .is_file() or .parent on None
    raised AttributeError and turned these pages into HTTP 500s; they now report
    the source as unavailable instead.
    """
    return path is not None and path.is_file()


def pipeline_sources(settings: Settings) -> list[dict[str, Any]]:
    return [
        _file_descriptor(path)
        for path in (
            settings.hcp379_live_summary,
            settings.hcp379_live_ledger,
            settings.hcp379_release_topology,
            settings.pipeline_status,
            settings.pipeline_density,
            settings.pipeline_manifest,
        )
        if path is not None
    ]


def release_status(settings: Settings) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "summary": None,
        "group_rows": [],
        "subject_rows": [],
        "ledger_summary": [],
        "topology": None,
        "warnings": [],
    }
    if _present(settings.hcp379_live_summary):
        summary = _read_json(settings.hcp379_live_summary)
        payload["summary"] = json_safe(summary)
        payload["group_rows"] = json_safe(summary.get("rows", []))
        subject_path = settings.hcp379_live_summary.parent / (
            "subject_status_density.csv"
        )
        if subject_path.is_file():
            subject = pd.read_csv(subject_path)
            payload["subject_rows"] = records(subject)
    else:
        payload["warnings"].append("HCP379 live summary is unavailable")

    if _present(settings.hcp379_live_ledger):
        ledger = pd.read_csv(settings.hcp379_live_ledger)
        group_columns = [
            column
            for column in (
                "lane",
                "status",
                "subject_state_status",
                "failure_class",
                "recovery_requirement",
            )
            if column in ledger.columns
        ]
        if group_columns:
            summary = (
                ledger.groupby(group_columns, dropna=False)
                .size()
                .rename("subjects")
                .reset_index()
                .sort_values(
                    ["lane", "subjects"],
                    ascending=[True, False],
                )
            )
            payload["ledger_summary"] = records(summary)
        payload["ledger_subject_n"] = int(len(ledger))
    else:
        payload["warnings"].append("HCP379 recovery ledger is unavailable")

    if _present(settings.hcp379_release_topology):
        topology = _read_json(settings.hcp379_release_topology)
        payload["topology"] = json_safe(topology)
    else:
        payload["warnings"].append("HCP379 release topology is unavailable")
    payload["sources"] = pipeline_sources(settings)
    return payload


def legacy_pipeline_status(settings: Settings) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "status": None,
        "density": None,
        "manifest_summary": None,
        "warnings": [],
    }
    if settings.pipeline_status.is_file():
        payload["status"] = json_safe(_read_json(settings.pipeline_status))
    else:
        payload["warnings"].append("Legacy pipeline status is unavailable")
    if settings.pipeline_density.is_file():
        payload["density"] = json_safe(_read_json(settings.pipeline_density))
    else:
        payload["warnings"].append("Legacy density status is unavailable")
    if settings.pipeline_manifest.is_file():
        manifest = pd.read_csv(settings.pipeline_manifest)
        result: dict[str, Any] = {
            "subject_n": int(len(manifest)),
            "columns": [str(column) for column in manifest.columns],
        }
        if "group" in manifest.columns:
            result["group_counts"] = {
                str(key): int(value)
                for key, value in manifest["group"]
                .astype(str)
                .value_counts()
                .to_dict()
                .items()
            }
        if "band" in manifest.columns:
            result["density_band_counts"] = {
                str(key): int(value)
                for key, value in manifest["band"]
                .fillna("missing")
                .astype(str)
                .value_counts()
                .to_dict()
                .items()
            }
        if "density" in manifest.columns:
            density = pd.to_numeric(
                manifest["density"],
                errors="coerce",
            )
            result["density"] = {
                "available_n": int(density.notna().sum()),
                "median": float(density.median()),
                "mean": float(density.mean()),
                "ge_0_60_n": int(density.ge(0.60).sum()),
            }
        payload["manifest_summary"] = result
    else:
        payload["warnings"].append("Legacy pipeline manifest is unavailable")
    payload["sources"] = pipeline_sources(settings)
    return payload


def pipeline_subjects(
    settings: Settings,
    *,
    source: str,
    offset: int,
    limit: int,
) -> dict[str, Any]:
    if offset < 0 or limit < 1 or limit > settings.table_max_limit:
        raise ValueError("invalid page")
    paths = {
        "hcp379": (
            settings.hcp379_live_summary.parent / "subject_status_density.csv"
            if settings.hcp379_live_summary is not None
            else None
        ),
        "legacy": settings.pipeline_manifest,
        "recovery-ledger": settings.hcp379_live_ledger,
    }
    if source not in paths:
        raise KeyError(f"unknown pipeline source: {source}")
    path = paths[source]
    if path is None or not path.is_file():
        raise FileNotFoundError(path if path is not None else f"{source} source is not available on this machine")
    frame = pd.read_csv(path)
    page = frame.iloc[offset : offset + limit]
    return {
        "source": source,
        "path": str(path),
        "offset": offset,
        "limit": limit,
        "total_rows": int(len(frame)),
        "columns": [str(column) for column in frame.columns],
        "rows": records(page),
    }
