from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd

from .lr_sr import aal3_connectome_matrix_path
from .serialization import json_safe, records
from .settings import Settings


ALLOWED_MATRIX_WEIGHTS = (
    "fd_sum",
    "count",
    "count_invnodevol",
    "invlen_mean",
    "len_mean",
    "fa_mean",
    "md_mean",
    "ad_mean",
    "rd_mean",
)


def _master(settings: Settings) -> pd.DataFrame:
    path = settings.analysis_root / "00_master/master_cohort.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def connectome_subjects(settings: Settings) -> pd.DataFrame:
    master_path = settings.analysis_root / "00_master/master_cohort.csv"
    signature = (
        master_path.stat().st_mtime_ns,
        settings.connectomes_root.stat().st_mtime_ns,
    )
    return _connectome_subjects_cached(settings, signature).copy()


@lru_cache(maxsize=4)
def _connectome_subjects_cached(
    settings: Settings,
    signature: tuple[int, int],
) -> pd.DataFrame:
    del signature
    master = _master(settings)
    columns = [
        column
        for column in (
            "subject_id",
            "group",
            "Image ID",
            "sex",
            "age",
            "phase",
            "density",
            "is_dense",
            "n_connectome_types",
        )
        if column in master.columns
    ]
    result = master[columns].copy()
    weight_pattern = "|".join(
        re.escape(weight)
        for weight in sorted(
            ALLOWED_MATRIX_WEIGHTS,
            key=len,
            reverse=True,
        )
    )
    filename_pattern = re.compile(
        rf"^SC_AAL166_(?P<subject>.+)_I[^_]+_"
        rf"(?P<weight>{weight_pattern})\.csv$"
    )
    available_by_subject: dict[str, set[str]] = {}
    for path in settings.connectomes_root.iterdir():
        if not path.is_file():
            continue
        match = filename_pattern.match(path.name)
        if match is None:
            continue
        available_by_subject.setdefault(
            match.group("subject"),
            set(),
        ).add(match.group("weight"))
    result["available_weights"] = [
        [
            weight
            for weight in ALLOWED_MATRIX_WEIGHTS
            if weight
            in available_by_subject.get(
                str(subject_id),
                set(),
            )
        ]
        for subject_id in result["subject_id"]
    ]
    return result


def _subject_row(
    settings: Settings,
    subject_id: str,
) -> pd.Series:
    master = _master(settings)
    selected = master[
        master["subject_id"].astype(str).eq(str(subject_id))
    ]
    if selected.empty:
        raise KeyError(f"unknown subject: {subject_id}")
    return selected.iloc[0]


def load_connectome_matrix(
    settings: Settings,
    subject_id: str,
    weight: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    if weight not in ALLOWED_MATRIX_WEIGHTS:
        raise ValueError(f"unsupported matrix weight: {weight}")
    row = _subject_row(settings, subject_id)
    path = aal3_connectome_matrix_path(
        settings.connectomes_root,
        str(subject_id),
        row.get("Image ID"),
        weight,
    )
    if path is None:
        raise FileNotFoundError(
            f"{subject_id} has no {weight} matrix"
        )
    try:
        matrix = np.loadtxt(path, delimiter=",")
    except (OSError, ValueError) as error:
        raise ValueError(f"matrix is unreadable: {path}") from error
    if (
        matrix.ndim != 2
        or matrix.shape[0] != matrix.shape[1]
        or matrix.shape[0] > 1_000
    ):
        raise ValueError(f"invalid square matrix: {path}")
    try:
        source = str(path.relative_to(settings.project_root))
    except ValueError:
        source = str(path)
    metadata = {
        "subject_id": str(subject_id),
        "group": str(row.get("group", "")),
        "image_id": json_safe(row.get("Image ID")),
        "weight": weight,
        "source": source,
        "shape": list(matrix.shape),
        "mtime_ns": path.stat().st_mtime_ns,
    }
    return matrix, metadata


def atlas_labels(
    settings: Settings,
    matrix_size: int,
) -> list[dict[str, Any]]:
    if not settings.aal_labels.is_file():
        return [
            {
                "matrix_idx": index + 1,
                "label": f"Node {index + 1}",
            }
            for index in range(matrix_size)
        ]
    labels = pd.read_csv(settings.aal_labels).iloc[:matrix_size].copy()
    return [
        {
            "matrix_idx": index + 1,
            "node_name": str(row.get("node_name", f"Node {index + 1}")),
            "label": str(
                row.get(
                    "atlas_label",
                    row.get("node_name", f"Node {index + 1}"),
                )
            ),
            "atlas_value": json_safe(row.get("atlas_value")),
        }
        for index, (_, row) in enumerate(labels.iterrows())
    ]


def matrix_summary(
    settings: Settings,
    subject_id: str,
    weight: str,
) -> dict[str, Any]:
    matrix, metadata = load_connectome_matrix(
        settings,
        subject_id,
        weight,
    )
    upper = matrix[np.triu_indices_from(matrix, k=1)]
    finite = upper[np.isfinite(upper)]
    positive = finite[finite > 0]
    possible_edges = int(matrix.shape[0] * (matrix.shape[0] - 1) // 2)
    summary = {
        "possible_edges": possible_edges,
        "finite_edges": int(finite.size),
        "positive_edges": int(positive.size),
        "density": (
            float(positive.size / possible_edges)
            if possible_edges
            else None
        ),
        "sum": float(np.sum(positive)) if positive.size else 0.0,
        "mean": float(np.mean(positive)) if positive.size else None,
        "median": (
            float(np.median(positive)) if positive.size else None
        ),
        "sd": (
            float(np.std(positive, ddof=1))
            if positive.size > 1
            else None
        ),
        "min": float(np.min(positive)) if positive.size else None,
        "max": float(np.max(positive)) if positive.size else None,
        "symmetric": bool(
            np.allclose(matrix, matrix.T, equal_nan=True)
        ),
        "finite_fraction": float(np.isfinite(matrix).mean()),
    }
    return {
        "metadata": metadata,
        "summary": summary,
    }


def matrix_payload(
    settings: Settings,
    subject_id: str,
    weight: str,
) -> dict[str, Any]:
    matrix, metadata = load_connectome_matrix(
        settings,
        subject_id,
        weight,
    )
    return {
        "metadata": metadata,
        "labels": atlas_labels(settings, matrix.shape[0]),
        "matrix": json_safe(matrix.tolist()),
    }


def matrix_edges(
    settings: Settings,
    subject_id: str,
    weight: str,
    *,
    positive_only: bool = True,
    offset: int = 0,
    limit: int = 2_000,
) -> dict[str, Any]:
    if offset < 0 or limit < 1 or limit > settings.table_max_limit:
        raise ValueError("invalid edge page")
    matrix, metadata = load_connectome_matrix(
        settings,
        subject_id,
        weight,
    )
    upper = np.triu_indices_from(matrix, k=1)
    frame = pd.DataFrame(
        {
            "source_index": upper[0] + 1,
            "target_index": upper[1] + 1,
            "value": matrix[upper],
        }
    )
    frame = frame[np.isfinite(frame["value"])].copy()
    if positive_only:
        frame = frame[frame["value"] > 0].copy()
    labels = atlas_labels(settings, matrix.shape[0])
    label_by_index = {
        int(item["matrix_idx"]): item["label"] for item in labels
    }
    frame["source_label"] = frame["source_index"].map(label_by_index)
    frame["target_label"] = frame["target_index"].map(label_by_index)
    frame = frame.sort_values(
        "value",
        ascending=False,
        kind="mergesort",
    ).reset_index(drop=True)
    page = frame.iloc[offset : offset + limit]
    return {
        "metadata": metadata,
        "positive_only": positive_only,
        "offset": offset,
        "limit": limit,
        "total_rows": int(len(frame)),
        "rows": records(page),
    }
