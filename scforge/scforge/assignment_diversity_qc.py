from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class AssignmentDiversityQC:
    path: str
    readable: bool
    total_streamlines: int
    assigned_streamlines: int
    unique_assigned_nodes: int
    unique_assignment_pairs: int
    self_loop_fraction: float
    zero_pair_fraction: float
    top5_endpoint_fraction: float
    collapse_class: str
    status: str
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _empty_result(path: Path, reason: str) -> AssignmentDiversityQC:
    return AssignmentDiversityQC(
        path=str(path),
        readable=False,
        total_streamlines=0,
        assigned_streamlines=0,
        unique_assigned_nodes=0,
        unique_assignment_pairs=0,
        self_loop_fraction=1.0,
        zero_pair_fraction=1.0,
        top5_endpoint_fraction=1.0,
        collapse_class="UNREADABLE_ASSIGNMENTS",
        status="FAIL",
        reason=reason,
    )


def read_assignment_pairs(path: str | Path) -> np.ndarray:
    path = Path(path)
    rows: list[list[int]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(",", " ").split()
            if len(parts) < 2:
                continue
            try:
                rows.append([int(float(parts[0])), int(float(parts[1]))])
            except ValueError:
                continue
    if not rows:
        return np.empty((0, 2), dtype=np.int32)
    return np.asarray(rows, dtype=np.int32)


def classify_assignment_pairs(pairs: np.ndarray, *, path: str | Path = "") -> AssignmentDiversityQC:
    path = Path(path) if path else Path("")
    if pairs.ndim != 2 or pairs.shape[1] != 2 or pairs.shape[0] == 0:
        return _empty_result(path, "no_two_column_assignment_pairs")

    total = int(pairs.shape[0])
    assigned_mask = (pairs[:, 0] > 0) & (pairs[:, 1] > 0)
    assigned_streamlines = int(assigned_mask.sum())
    endpoints = pairs.reshape(-1)
    positive = endpoints[endpoints > 0]
    unique_assigned_nodes = int(np.unique(positive).size) if positive.size else 0
    unique_assignment_pairs = int(np.unique(np.sort(pairs[assigned_mask], axis=1), axis=0).shape[0]) if assigned_streamlines else 0

    self_loop_fraction = float((pairs[:, 0] == pairs[:, 1]).sum() / total)
    zero_pair_fraction = float(((pairs[:, 0] == 0) | (pairs[:, 1] == 0)).sum() / total)
    if endpoints.size:
        _, endpoint_counts = np.unique(endpoints, return_counts=True)
        top5_endpoint_fraction = float(np.sort(endpoint_counts)[-5:].sum() / endpoints.size)
    else:
        top5_endpoint_fraction = 1.0

    if unique_assigned_nodes < 20 or top5_endpoint_fraction > 0.90:
        collapse = "CATASTROPHIC_COLLAPSE"
    elif unique_assigned_nodes < 80 or top5_endpoint_fraction > 0.70:
        collapse = "SEVERE_COLLAPSE"
    elif unique_assigned_nodes < 120 or top5_endpoint_fraction > 0.50:
        collapse = "WARN_ASSIGNMENT_CONCENTRATION"
    else:
        collapse = "PASS_ASSIGNMENT_DIVERSITY"

    return AssignmentDiversityQC(
        path=str(path),
        readable=True,
        total_streamlines=total,
        assigned_streamlines=assigned_streamlines,
        unique_assigned_nodes=unique_assigned_nodes,
        unique_assignment_pairs=unique_assignment_pairs,
        self_loop_fraction=self_loop_fraction,
        zero_pair_fraction=zero_pair_fraction,
        top5_endpoint_fraction=top5_endpoint_fraction,
        collapse_class=collapse,
        status="PASS" if collapse == "PASS_ASSIGNMENT_DIVERSITY" else "WARN" if collapse.startswith("WARN") else "FAIL",
    )


def classify_assignment(path: str | Path) -> AssignmentDiversityQC:
    path = Path(path)
    try:
        pairs = read_assignment_pairs(path)
    except Exception as exc:
        return _empty_result(path, f"read_error:{type(exc).__name__}:{exc}")
    return classify_assignment_pairs(pairs, path=path)
