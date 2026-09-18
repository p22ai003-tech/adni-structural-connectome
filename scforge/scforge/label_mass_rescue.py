from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class LabelMassRescueResult:
    labels_rescued: tuple[int, ...]
    labels_still_missing: tuple[int, ...]
    voxels_added: int
    method: str

    def to_dict(self) -> dict:
        return {
            "labels_rescued": ";".join(str(x) for x in self.labels_rescued),
            "labels_still_missing": ";".join(str(x) for x in self.labels_still_missing),
            "voxels_added": int(self.voxels_added),
            "method": self.method,
        }


def rescue_missing_labels_from_probability_maps(
    label_image: np.ndarray,
    probability_maps: Mapping[int, np.ndarray],
    *,
    valid_labels: tuple[int, ...],
    min_voxels: int = 50,
    allowed_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, LabelMassRescueResult]:
    """Recover labels that vanish after label-safe transforms.

    This is intentionally conservative: it only adds voxels from a transformed
    per-label probability/mass map, never by blind dilation. If no probability
    map exists for a missing label, the label remains missing.
    """

    out = np.asarray(label_image).copy()
    if allowed_mask is None:
        allowed = np.ones(out.shape, dtype=bool)
    else:
        allowed = np.asarray(allowed_mask).astype(bool)

    rescued: list[int] = []
    still_missing: list[int] = []
    voxels_added = 0
    occupied = out > 0

    for label in valid_labels:
        label = int(label)
        current = int((out == label).sum())
        if current >= min_voxels:
            continue
        prob = probability_maps.get(label)
        if prob is None:
            still_missing.append(label)
            continue
        prob_arr = np.asarray(prob, dtype=float)
        if prob_arr.shape != out.shape:
            still_missing.append(label)
            continue
        needed = int(max(0, min_voxels - current))
        candidates = allowed & ~occupied & np.isfinite(prob_arr) & (prob_arr > 0)
        if not candidates.any():
            still_missing.append(label)
            continue
        scores = prob_arr[candidates]
        coords = np.argwhere(candidates)
        order = np.argsort(scores)[::-1][:needed]
        chosen = coords[order]
        out[tuple(chosen.T)] = label
        occupied[tuple(chosen.T)] = True
        added = int(chosen.shape[0])
        voxels_added += added
        if int((out == label).sum()) >= min_voxels:
            rescued.append(label)
        else:
            still_missing.append(label)

    return out, LabelMassRescueResult(
        labels_rescued=tuple(rescued),
        labels_still_missing=tuple(still_missing),
        voxels_added=voxels_added,
        method="probability_mass_top_voxels",
    )
