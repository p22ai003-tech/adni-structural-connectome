from __future__ import annotations

import hashlib
import json
import mimetypes
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from .sections import Section
from .serialization import json_safe, records
from .settings import Settings


ALLOWED_ARTIFACT_SUFFIXES = {
    ".csv",
    ".json",
    ".md",
    ".png",
    ".svg",
    ".pdf",
}


@dataclass(frozen=True)
class Artifact:
    id: str
    relative_path: str
    name: str
    suffix: str
    size_bytes: int
    mtime_ns: int
    mime_type: str
    kind: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "relative_path": self.relative_path,
            "name": self.name,
            "suffix": self.suffix,
            "size_bytes": self.size_bytes,
            "mtime_ns": self.mtime_ns,
            "mime_type": self.mime_type,
            "kind": self.kind,
        }


def _artifact_id(relative_path: str) -> str:
    return hashlib.sha256(relative_path.encode("utf-8")).hexdigest()[:24]


def _artifact_kind(path: Path) -> str:
    if path.suffix.lower() == ".csv":
        return "table"
    if path.suffix.lower() == ".json":
        return "json"
    if path.suffix.lower() == ".md":
        return "note"
    return "figure"


@lru_cache(maxsize=8)
def _inventory_cached(
    root_string: str,
    root_mtime_ns: int,
) -> tuple[Artifact, ...]:
    del root_mtime_ns
    root = Path(root_string)
    result: list[Artifact] = []
    for path in sorted(root.rglob("*")):
        if (
            not path.is_file()
            or path.is_symlink()
            or path.suffix.lower() not in ALLOWED_ARTIFACT_SUFFIXES
        ):
            continue
        relative = path.relative_to(root).as_posix()
        stat = path.stat()
        result.append(
            Artifact(
                id=_artifact_id(relative),
                relative_path=relative,
                name=path.name,
                suffix=path.suffix.lower(),
                size_bytes=stat.st_size,
                mtime_ns=stat.st_mtime_ns,
                mime_type=(
                    mimetypes.guess_type(path.name)[0]
                    or "application/octet-stream"
                ),
                kind=_artifact_kind(path),
            )
        )
    return tuple(result)


def artifact_inventory(settings: Settings) -> tuple[Artifact, ...]:
    root = settings.analysis_root
    if not root.is_dir():
        return ()
    return _inventory_cached(str(root), root.stat().st_mtime_ns)


def section_artifacts(
    settings: Settings,
    section: Section,
) -> list[Artifact]:
    prefixes = tuple(f"{folder}/" for folder in section.folders)
    return [
        artifact
        for artifact in artifact_inventory(settings)
        if artifact.relative_path.startswith(prefixes)
    ]


def artifact_by_id(settings: Settings, artifact_id: str) -> Artifact:
    matches = [
        artifact
        for artifact in artifact_inventory(settings)
        if artifact.id == artifact_id
    ]
    if len(matches) != 1:
        raise KeyError(f"unknown artifact id: {artifact_id}")
    return matches[0]


def artifact_path(settings: Settings, artifact: Artifact) -> Path:
    candidate = (settings.analysis_root / artifact.relative_path).resolve()
    candidate.relative_to(settings.analysis_root)
    if (
        not candidate.is_file()
        or candidate.is_symlink()
        or candidate.suffix.lower() not in ALLOWED_ARTIFACT_SUFFIXES
    ):
        raise FileNotFoundError(candidate)
    return candidate


def table_payload(
    settings: Settings,
    artifact: Artifact,
    offset: int,
    limit: int,
) -> dict[str, Any]:
    if artifact.suffix != ".csv":
        raise ValueError("artifact is not a CSV table")
    if offset < 0 or limit < 1 or limit > settings.table_max_limit:
        raise ValueError("invalid table page")
    path = artifact_path(settings, artifact)
    header = pd.read_csv(path, nrows=0)
    page = pd.read_csv(path, skiprows=range(1, offset + 1), nrows=limit)
    with path.open("rb") as handle:
        total_rows = max(0, sum(1 for _ in handle) - 1)
    return {
        "artifact": artifact.to_dict(),
        "offset": offset,
        "limit": limit,
        "total_rows": total_rows,
        "columns": [str(column) for column in header.columns],
        "rows": records(page),
    }


def artifact_content(
    settings: Settings,
    artifact: Artifact,
) -> dict[str, Any]:
    path = artifact_path(settings, artifact)
    if artifact.suffix == ".md":
        return {
            "artifact": artifact.to_dict(),
            "content": path.read_text(encoding="utf-8"),
        }
    if artifact.suffix == ".json":
        return {
            "artifact": artifact.to_dict(),
            "content": json_safe(
                json.loads(path.read_text(encoding="utf-8"))
            ),
        }
    raise ValueError("artifact has no JSON/text content endpoint")


def subject_table_path(
    settings: Settings,
    section: Section,
) -> Path | None:
    if not section.subject_table:
        return None
    candidates = [
        settings.analysis_root / folder / section.subject_table
        for folder in section.folders
    ]
    return next((path for path in candidates if path.is_file()), None)


def subject_metric_catalog(
    settings: Settings,
    section: Section,
) -> dict[str, Any]:
    path = subject_table_path(settings, section)
    if path is None:
        return {"available": False, "metrics": []}
    frame = pd.read_csv(path)
    excluded = {
        "subject_id",
        "group",
        "age",
        "sex",
        "phase",
        "Image ID",
    }
    metrics = [
        str(column)
        for column in frame.columns
        if column not in excluded
        and pd.api.types.is_numeric_dtype(frame[column])
    ]
    return {
        "available": True,
        "source": str(path.relative_to(settings.analysis_root)),
        "subject_n": int(len(frame)),
        "metrics": metrics,
    }


def subject_metric_values(
    settings: Settings,
    section: Section,
    metric: str,
) -> dict[str, Any]:
    path = subject_table_path(settings, section)
    if path is None:
        raise FileNotFoundError("subject table unavailable")
    frame = pd.read_csv(path)
    if metric not in frame.columns:
        raise KeyError(f"unknown metric: {metric}")
    columns = [
        column
        for column in ("subject_id", "group", "age", "sex", "phase", metric)
        if column in frame.columns
    ]
    result = frame[columns].copy()
    result[metric] = pd.to_numeric(result[metric], errors="coerce")
    return {
        "source": str(path.relative_to(settings.analysis_root)),
        "metric": metric,
        "rows": records(result),
    }
