"""Unified network x measure ranked table for the LR/SR analysis page.

Combines the precomputed per-network group statistics (graph topology, microstructure,
structure-function coupling) with a freshly aggregated per-network EDR LR-exception
summary (burden + strength), then ranks every network within each measure by the
CN-AD effect size. This is the "holistic" table: which measure separates the cohort,
in which network, and how strongly.

All statistics are CN-vs-AD Cliff's delta + Brunner-Munzel p, BH-FDR corrected within
each measure across networks (mirrors the pipeline's per-family correction scope).
The EDR-exception measures are exploratory (they do not survive correction at the
per-network level; the exception signal lives in the whole-brain edge-class x diagnosis
interaction, not in any single network).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as _sps

from .settings import Settings

_FUNCTIONAL_NETWORKS = (
    "Visual", "Somatomotor", "DorsalAttention", "Salience_VAN", "Limbic",
    "Frontoparietal", "DMN", "Subcortical", "Cerebellar", "Brainstem",
)

# (family, source metric column, display label) — order defines column order in the table
_MEASURES = (
    ("graph", "nodal_eff", "Nodal-eff"),
    ("graph", "strength", "Strength"),
    ("microstructure", "fa_mean", "FA"),
    ("microstructure", "md_mean", "MD"),
    ("microstructure", "rd_mean", "RD"),
    ("microstructure", "ad_mean", "AxD"),
    ("coupling", "strength__md_mean", "Coupling"),
    ("exception", "exc_burden", "Exc-burden"),
    ("exception", "exc_strength", "Exc-strength"),
)

_CONTRAST_GROUPS = {
    "cn_mci": ("CN", "MCI"),
    "cn_ad": ("CN", "AD"),
    "mci_ad": ("MCI", "AD"),
}

_LOCK = Lock()


def _mtime(path: Path) -> int:
    try:
        return path.stat().st_mtime_ns
    except OSError:
        return -1


def _cliffs_delta(first: np.ndarray, second: np.ndarray) -> float:
    a = np.asarray(first, dtype=float)
    b = np.asarray(second, dtype=float)
    if a.size == 0 or b.size == 0:
        return float("nan")
    greater = int((a[:, None] > b[None, :]).sum())
    less = int((a[:, None] < b[None, :]).sum())
    return (greater - less) / (a.size * b.size)


def _bh_fdr(pvalues: list[float]) -> list[float]:
    values = np.asarray(pvalues, dtype=float)
    n = values.size
    if n == 0:
        return []
    order = np.argsort(values)
    adjusted = np.empty(n, dtype=float)
    running = 1.0
    for step, idx in enumerate(order[::-1]):
        rank = n - step
        running = min(running, values[idx] * n / rank)
        adjusted[idx] = running
    return adjusted.tolist()


def _exception_network_stats(settings: Settings) -> dict[tuple[str, str], dict[str, float]]:
    """Aggregate node-level LR-exception counts/strengths to functional networks and
    test CN vs AD per network. Returns {(network, 'exc_burden'|'exc_strength'): stat}."""
    mapping_path = settings.analysis_root / "19_network_analysis/network_mapping_used.csv"
    node_path = (
        settings.analysis_root
        / "17_edr_exceptions/edr_exception_node_level_fd_sum_len_mean.csv"
    )
    if not mapping_path.is_file() or not node_path.is_file():
        return {}
    mapping = pd.read_csv(mapping_path)
    # join by matrix index, NOT atlas_label: the section-17 node-level artifact carries
    # shifted ROI names for rows >= 35 (stale upstream AAL3_labels.csv).
    node2net = dict(zip(mapping["matrix_idx"].astype(int), mapping["functional_network"]))

    node = pd.read_csv(node_path)
    node["network"] = pd.to_numeric(node["node"], errors="coerce").map(node2net)
    node = node[node["network"].isin(_FUNCTIONAL_NETWORKS)].copy()

    # per subject x network: burden = sum(exc)/sum(edges); strength = count-weighted mean
    node["_ec"] = pd.to_numeric(node["lr_exception_count"], errors="coerce").fillna(0.0)
    node["_tc"] = pd.to_numeric(node["lr_edge_count"], errors="coerce").fillna(0.0)
    node["_st"] = pd.to_numeric(node["lr_exception_strength"], errors="coerce")
    node["_sw"] = node["_st"].notna().astype(float) * node["_ec"]
    node["_sn"] = node["_st"].fillna(0.0) * node["_ec"]

    grouped = node.groupby(["subject_id", "group", "network"], dropna=False).agg(
        ec=("_ec", "sum"), tc=("_tc", "sum"), sn=("_sn", "sum"), sw=("_sw", "sum")
    ).reset_index()
    grouped["burden"] = np.where(grouped["tc"] > 0, grouped["ec"] / grouped["tc"], np.nan)
    grouped["strength"] = np.where(grouped["sw"] > 0, grouped["sn"] / grouped["sw"], np.nan)

    out: dict[tuple[str, str], dict[str, float]] = {}
    for metric_key, column in (("exc_burden", "burden"), ("exc_strength", "strength")):
        recs: list[dict[str, Any]] = []
        for network in _FUNCTIONAL_NETWORKS:
            sub = grouped[grouped["network"] == network]
            arrays = {
                g: sub.loc[sub["group"] == g, column].dropna().to_numpy(dtype=float)
                for g in ("CN", "MCI", "AD")
            }
            if min(len(arrays["CN"]), len(arrays["MCI"]), len(arrays["AD"])) < 5:
                continue
            rec = dict(
                network=network,
                mean_CN=float(np.mean(arrays["CN"])),
                mean_MCI=float(np.mean(arrays["MCI"])),
                mean_AD=float(np.mean(arrays["AD"])),
                kw_p=float(_sps.kruskal(arrays["CN"], arrays["MCI"], arrays["AD"]).pvalue),
            )
            for prefix, (first, second) in _CONTRAST_GROUPS.items():
                try:
                    bm_p = float(
                        _sps.brunnermunzel(
                            arrays[first], arrays[second], alternative="two-sided"
                        ).pvalue
                    )
                except ValueError:
                    bm_p = float("nan")
                rec[f"{prefix}_cliffs_delta"] = _cliffs_delta(arrays[first], arrays[second])
                rec[f"{prefix}_bm_p"] = bm_p
            recs.append(rec)
        for prefix in _CONTRAST_GROUPS:
            qs = _bh_fdr([r[f"{prefix}_bm_p"] for r in recs])
            for rec, q in zip(recs, qs):
                rec[f"{prefix}_bm_q"] = q
        for rec in recs:
            out[(rec["network"], metric_key)] = rec
    return out


def _load_precomputed(settings: Settings) -> dict[tuple[str, str], dict[str, float]]:
    base = settings.analysis_root / "19_network_analysis/functional"
    out: dict[tuple[str, str], dict[str, float]] = {}
    for name in (
        "network_graph_stats.csv",
        "network_microstructure_stats.csv",
        "network_coupling_stats.csv",
    ):
        path = base / name
        if not path.is_file():
            continue
        frame = pd.read_csv(path)
        for _, row in frame.iterrows():
            out[(str(row["network"]), str(row["metric"]))] = row.to_dict()
    return out


def _signature(settings: Settings) -> tuple[int, ...]:
    base = settings.analysis_root / "19_network_analysis"
    paths = [
        base / "functional/network_graph_stats.csv",
        base / "functional/network_microstructure_stats.csv",
        base / "functional/network_coupling_stats.csv",
        base / "functional/network_range_restricted_stats.csv",
        base / "functional/network_exception_range_stats.csv",
        base / "functional/network_exception_measures_stats.csv",
        base / "network_mapping_used.csv",
        settings.analysis_root
        / "17_edr_exceptions/edr_exception_node_level_fd_sum_len_mean.csv",
    ]
    return tuple(_mtime(p) for p in paths)


# Range-restricted (Group B) and exception-by-range (Group C) measure column order.
_RANGE_MEASURES = tuple(
    f"{rng}-{measure}"
    for rng in ("SR", "LR")
    for measure in ("Strength", "Degree", "FA", "MD", "RD", "AxD")
)
_EXC_RANGE_MEASURES = tuple(
    f"{rng}exc-{measure}"
    for rng in ("SR", "LR")
    for measure in ("Burden", "Strength")
)
# same measure set as the tract group, but computed on EDR-exception edges only
_EXC_MEASURE_COLUMNS = tuple(
    f"{rng}exc-{measure}"
    for rng in ("SR", "LR")
    for measure in ("Strength", "Degree", "FA", "MD", "RD", "AxD")
)


def _load_long_stats(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    """Load a long-format network x measure stats CSV keyed by (network, measure)."""
    if not path.is_file():
        return {}
    frame = pd.read_csv(path)
    return {
        (str(row["network"]), str(row["measure"])): row.to_dict()
        for _, row in frame.iterrows()
    }


def _rank_and_cells(
    stats: dict[tuple[str, str], dict[str, Any]],
    measures: tuple[str, ...],
    family: str,
    contrast: str = "cn_ad",
) -> list[dict[str, Any]]:
    """Rank networks within each measure by |Cliff's delta| for `contrast`, emit cells."""
    delta_col = f"{contrast}_cliffs_delta"
    q_col = f"{contrast}_bm_q"
    ranks: dict[tuple[str, str], int] = {}
    for measure in measures:
        present = [
            (net, abs(_fnum(stats[(net, measure)][delta_col])))
            for net in _FUNCTIONAL_NETWORKS
            if (net, measure) in stats
            and delta_col in stats[(net, measure)]
            and not np.isnan(_fnum(stats[(net, measure)][delta_col]))
        ]
        present.sort(key=lambda item: -item[1])
        for position, (net, _) in enumerate(present, start=1):
            ranks[(net, measure)] = position

    cells: list[dict[str, Any]] = []
    for net in _FUNCTIONAL_NETWORKS:
        for measure in measures:
            row = stats.get((net, measure))
            if row is None:
                cells.append(
                    dict(network=net, family=family, measure=measure, available=False)
                )
                continue
            q = _fnum(row.get(q_col))
            first, second = _CONTRAST_GROUPS[contrast]
            mean_first = _fnum(row.get(f"mean_{first}"))
            mean_second = _fnum(row.get(f"mean_{second}"))
            delta = _fnum(row.get(delta_col))
            cells.append(
                dict(
                    network=net,
                    family=family,
                    measure=measure,
                    available=True,
                    rank=ranks.get((net, measure)),
                    mean_CN=_fnum(row.get("mean_CN")),
                    mean_MCI=_fnum(row.get("mean_MCI")),
                    mean_AD=_fnum(row.get("mean_AD")),
                    cliffs_delta=delta,
                    kw_p=_fnum(row.get("kw_p")),
                    bm_q=q,
                    direction=_direction_from(delta, mean_first, mean_second),
                    significant=bool(q < 0.05) if q == q else False,
                )
            )
    return cells


def _direction_from(delta: float, mean_first: float, mean_second: float) -> str:
    """Direction from the first group to the second in the contrast.

    "up" = higher in the second (later-stage) group, "down" = lower in it. Derived from
    Cliff's delta (rank-based, negative = second group ranks higher) so the arrow agrees
    with the rank and the Brunner-Munzel q, which are also rank-based. Comparing means
    instead can disagree under skew. Falls back to means if delta is unavailable.
    """
    if delta == delta:  # not NaN
        return "up" if delta < 0 else "down"
    return "down" if mean_first > mean_second else "up"


@lru_cache(maxsize=8)
def _network_measure_ranked_cached(
    settings: Settings, signature: tuple[int, ...], contrast: str
) -> dict[str, Any]:
    del signature  # only present to bust the cache when source artifacts change
    return _build(settings, contrast)


def _fnum(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result


def _build(settings: Settings, contrast: str = "cn_ad") -> dict[str, Any]:
    precomputed = _load_precomputed(settings)
    exceptions = _exception_network_stats(settings)
    combined = {**precomputed, **exceptions}
    if not combined:
        return {"status": "missing", "measures": [], "networks": [], "cells": []}

    # re-key the whole-connectome stats by DISPLAY label so group A can share the same
    # ranking/cell code path as the other groups
    label_by_metric = {metric: label for _family, metric, label in _MEASURES}
    all_stats = {
        (net, label_by_metric[metric]): row
        for (net, metric), row in combined.items()
        if metric in label_by_metric
    }
    all_measures = tuple(label for _family, _metric, label in _MEASURES)
    cells = _rank_and_cells(all_stats, all_measures, "all", contrast)

    functional = settings.analysis_root / "19_network_analysis/functional"
    range_stats = _load_long_stats(functional / "network_range_restricted_stats.csv")
    exc_stats = _load_long_stats(functional / "network_exception_range_stats.csv")

    groups = [
        {
            "key": "all",
            "label": "A · Whole connectome (all edges)",
            "description": (
                "Every positive edge. Graph topology, microstructure and coupling per "
                "functional network."
            ),
            "measures": [label for _family, _metric, label in _MEASURES],
            "cells": cells,
        }
    ]
    if range_stats:
        groups.append(
            {
                "key": "range",
                "label": "B · SR / LR tracts only",
                "description": (
                    "The same node-level measures recomputed using ONLY short-range "
                    "(<= CN Q1) or long-range (> CN Q3) edges, then aggregated to "
                    "networks the same way, so the columns are directly comparable "
                    "with group A."
                ),
                "measures": list(_RANGE_MEASURES),
                "cells": _rank_and_cells(range_stats, _RANGE_MEASURES, "range", contrast),
            }
        )
    exc_measure_stats = _load_long_stats(
        functional / "network_exception_measures_stats.csv"
    )
    if exc_measure_stats:
        groups.append(
            {
                "key": "exception_measures",
                "label": "C · SR / LR EDR exceptions only",
                "description": (
                    "The same node-level measures as group B, but computed using ONLY "
                    "the EDR-exception edges (weight > mean + 3 SD for its own distance "
                    "bin), split by the exception's SR or LR range. Directly comparable "
                    "column-for-column with groups A and B. Note that exception edges "
                    "are sparse, so a network/range with few exceptions has a smaller "
                    "effective n than the full cohort."
                ),
                "measures": list(_EXC_MEASURE_COLUMNS),
                "cells": _rank_and_cells(
                    exc_measure_stats, _EXC_MEASURE_COLUMNS, "exception", contrast
                ),
            }
        )
    if exc_stats:
        groups.append(
            {
                "key": "exception",
                "label": "D · SR / LR exception burden and strength",
                "description": (
                    "How many exceptions a network carries (burden = share of its edges "
                    "in that range that are exceptions) and how strong they are."
                ),
                "measures": list(_EXC_RANGE_MEASURES),
                "cells": _rank_and_cells(exc_stats, _EXC_RANGE_MEASURES, "exception", contrast),
            }
        )

    return {
        "status": "ok",
        "n_subjects": 530,
        "contrast": contrast,
        "contrast_label": "-".join(_CONTRAST_GROUPS[contrast]),
        "contrasts": [
            {"key": key, "label": "-".join(pair)}
            for key, pair in _CONTRAST_GROUPS.items()
        ],
        # retained for backwards compatibility with the original single-table payload
        "measures": [
            {"key": metric, "label": label, "family": family}
            for family, metric, label in _MEASURES
        ],
        "networks": list(_FUNCTIONAL_NETWORKS),
        "cells": cells,
        "groups": groups,
        "note": (
            "Cell rank is within-measure by CN-AD |Cliff's delta| (rank 1 = largest "
            "effect). Direction is CN->AD (down = falls, up = rises). Significance is "
            "Brunner-Munzel CN-AD, BH-FDR corrected within each measure across the 10 "
            "functional networks. Group A microstructure/graph/coupling values are the "
            "precomputed pipeline statistics; groups B and C are computed from the "
            "connectome matrices and the EDR exception artifacts using the same "
            "aggregation and the same tests. All EDR-exception measures are "
            "EXPLORATORY: no single network survives correction in either range, so "
            "the exception effect is global rather than network-localizable."
        ),
    }


def network_measure_ranked_table(
    settings: Settings, contrast: str = "cn_ad"
) -> dict[str, Any]:
    """Public reader: unified network x measure ranked table (cached by artifact mtime).

    `contrast` selects the pairwise group comparison driving rank, direction and q:
    "cn_mci", "cn_ad" (default) or "mci_ad".
    """
    if contrast not in _CONTRAST_GROUPS:
        raise ValueError(f"unsupported contrast: {contrast}")
    with _LOCK:
        return _network_measure_ranked_cached(settings, _signature(settings), contrast)
