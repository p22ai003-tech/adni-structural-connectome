from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from .sections import Section
from .serialization import records
from .settings import Settings
from .statistics import GROUP_ORDER


GROUP_PAIRS = (("CN", "MCI"), ("CN", "AD"), ("MCI", "AD"))
NON_METRIC_COLUMNS = {
    "subject_id",
    "group",
    "age",
    "sex",
    "phase",
    "Image ID",
}


@dataclass(frozen=True)
class NodeSource:
    folder: str
    filename: str
    layout: str
    preferred_metrics: tuple[str, ...]


NODE_SOURCES = {
    "local-roi-dti": NodeSource(
        "04_node_microstructure_live",
        "live_node_microstructure_long.csv",
        "metric-value",
        ("fa_mean", "md_mean", "ad_mean", "rd_mean"),
    ),
    "node-metrics": NodeSource(
        "07_node_graph_live",
        "live_node_graph_long.csv",
        "wide",
        ("strength", "degree", "nodal_eff"),
    ),
}


PREFERRED_SUBJECT_METRICS: dict[str, tuple[str, ...]] = {
    "global-dti": (
        "fa_mean_edge_mean",
        "fa_mean_edge_median",
        "md_mean_edge_mean",
        "md_mean_edge_median",
        "ad_mean_edge_mean",
        "ad_mean_edge_median",
        "rd_mean_edge_mean",
        "rd_mean_edge_median",
    ),
    "global-graph": (
        "mean_strength",
        "total_strength",
        "density",
        "global_efficiency",
        "charpath_len",
    ),
    "coupling": (
        "strength_fa_mean_rho",
        "strength_md_mean_rho",
        "strength_ad_mean_rho",
        "strength_rd_mean_rho",
        "degree_fa_mean_rho",
        "degree_md_mean_rho",
        "degree_ad_mean_rho",
        "degree_rd_mean_rho",
        "nodal_eff_fa_mean_rho",
        "nodal_eff_md_mean_rho",
        "nodal_eff_ad_mean_rho",
        "nodal_eff_rd_mean_rho",
    ),
    "brain-age": (
        "BAG",
        "BAG_age_corrected",
        "brain_age_pred",
    ),
    "lr-sr": (
        "short_w_mean",
        "long_w_mean",
        "sr_lr_w_diff",
        "sr_lr_w_ratio",
        "short_global_eff",
        "long_global_eff",
        "intra_w_mean",
        "inter_w_mean",
    ),
    "edr-exceptions": (
        "edr_lambda",
        "edr_slope",
        "sr_exception_pct",
        "mr_exception_pct",
        "lr_exception_pct",
        "sr_exception_strength",
        "lr_exception_strength",
        "sr_lr_exception_pct_diff",
        "sr_lr_exception_strength_ratio",
    ),
    "delay": (
        "mean_delay_ms",
        "median_delay_ms",
        "p90_delay_ms",
        "delay_burden_ms",
        "long_delay_fraction",
        "delay_weighted_strength",
        "delay_path_ms",
    ),
    "advanced": (
        "edr_residual_mean",
        "edr_residual_short",
        "edr_residual_long",
        "structural_compensation_index",
        "long_range_vulnerability",
        "short_range_preservation",
    ),
}


METRIC_DESCRIPTIONS = {
    "fa_mean_edge_mean": "Mean edge-wise fractional anisotropy.",
    "md_mean_edge_mean": "Mean edge-wise mean diffusivity.",
    "ad_mean_edge_mean": "Mean edge-wise axial diffusivity.",
    "rd_mean_edge_mean": "Mean edge-wise radial diffusivity.",
    "mean_strength": "Mean weighted structural-connectome strength.",
    "total_strength": "Sum of weighted structural connections.",
    "density": "Fraction of possible undirected edges that are present.",
    "global_efficiency": "Whole-network inverse shortest-path efficiency.",
    "charpath_len": "Characteristic weighted shortest-path length.",
    "BAG": "Predicted structural brain age minus chronological age.",
    "BAG_age_corrected": "Brain-age gap after fitted age-bias correction.",
    "brain_age_pred": "Cross-validated predicted structural brain age.",
    "edr_lambda": "Distance-decay parameter from the fitted edge-distance relationship.",
    "edr_slope": "Fitted log-strength versus tract-length slope.",
    "sr_lr_w_diff": "Short-range minus long-range mean connection strength.",
    "sr_lr_w_ratio": "Short-range divided by long-range mean connection strength.",
    "mean_delay_ms": "Mean length-derived propagation-delay proxy.",
    "delay_burden_ms": "Aggregate length-derived delay burden.",
    "structural_compensation_index": "Short-versus-long residual structural compensation index.",
    "long_range_vulnerability": "Relative long-range structural vulnerability summary.",
}


def _relative(settings: Settings, path: Path) -> str:
    return path.relative_to(settings.analysis_root).as_posix()


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _numeric_metric_columns(frame: pd.DataFrame) -> list[str]:
    metrics: list[str] = []
    for column in frame.columns:
        name = str(column)
        if name in NON_METRIC_COLUMNS:
            continue
        if (
            name.startswith("n_")
            or name.endswith("_n_edges")
            or name.endswith("_edge_count")
            or name.endswith("_count")
        ):
            continue
        numeric = pd.to_numeric(frame[column], errors="coerce")
        if int(numeric.notna().sum()) >= 3:
            metrics.append(name)
    return metrics


def _subject_path(settings: Settings, section: Section) -> Path:
    if not section.subject_table:
        raise FileNotFoundError(
            f"subject table unavailable for {section.id}"
        )
    for folder in section.folders:
        candidate = settings.analysis_root / folder / section.subject_table
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(section.subject_table)


@lru_cache(maxsize=24)
def _subject_frame_cached(path_string: str, mtime_ns: int) -> pd.DataFrame:
    del mtime_ns
    path = Path(path_string)
    frame = pd.read_csv(path)
    if path.name == "live_subject_level_coupling.csv":
        required = {
            "subject_id",
            "group",
            "nodal_metric",
            "micro_metric",
            "rho",
        }
        if not required.issubset(frame.columns):
            raise ValueError("coupling subject table has an invalid schema")
        work = frame.copy()
        work["metric"] = (
            work["nodal_metric"].astype(str)
            + "_"
            + work["micro_metric"].astype(str)
            + "_rho"
        )
        index = [
            column
            for column in ("subject_id", "group", "age", "sex", "phase")
            if column in work.columns
        ]
        frame = (
            work.pivot_table(
                index=index,
                columns="metric",
                values="rho",
                aggfunc="first",
            )
            .reset_index()
            .rename_axis(None, axis=1)
        )
    return frame


def subject_frame(settings: Settings, section: Section) -> tuple[pd.DataFrame, Path]:
    path = _subject_path(settings, section)
    return _subject_frame_cached(str(path), path.stat().st_mtime_ns).copy(), path


def subject_metric_profile(
    settings: Settings,
    section: Section,
) -> dict[str, Any]:
    try:
        frame, path = subject_frame(settings, section)
    except FileNotFoundError:
        return {
            "available": False,
            "source": None,
            "subject_n": 0,
            "metrics": [],
        }
    metrics = _numeric_metric_columns(frame)
    preferred = PREFERRED_SUBJECT_METRICS.get(section.id, ())
    ordered = [metric for metric in preferred if metric in metrics]
    ordered.extend(metric for metric in metrics if metric not in ordered)
    return {
        "available": bool(ordered),
        "source": _relative(settings, path),
        "subject_n": int(frame["subject_id"].nunique())
        if "subject_id" in frame.columns
        else int(len(frame)),
        "metrics": [
            {
                "id": metric,
                "label": metric.replace("_", " "),
                "description": METRIC_DESCRIPTIONS.get(
                    metric,
                    "Recorded numeric measure from the authoritative analysis table.",
                ),
            }
            for metric in ordered
        ],
    }


def _descriptives(values: pd.DataFrame, value_column: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group in GROUP_ORDER:
        group_values = pd.to_numeric(
            values.loc[values["group"].astype(str).eq(group), value_column],
            errors="coerce",
        ).dropna()
        rows.append(
            {
                "group": group,
                "n": int(len(group_values)),
                "mean": float(group_values.mean())
                if len(group_values)
                else math.nan,
                "sd": float(group_values.std(ddof=1))
                if len(group_values) > 1
                else math.nan,
                "median": float(group_values.median())
                if len(group_values)
                else math.nan,
                "q1": float(group_values.quantile(0.25))
                if len(group_values)
                else math.nan,
                "q3": float(group_values.quantile(0.75))
                if len(group_values)
                else math.nan,
                "iqr": float(
                    group_values.quantile(0.75)
                    - group_values.quantile(0.25)
                )
                if len(group_values)
                else math.nan,
                "min": float(group_values.min())
                if len(group_values)
                else math.nan,
                "max": float(group_values.max())
                if len(group_values)
                else math.nan,
            }
        )
    return pd.DataFrame(rows)


def _stored_metric_table(
    settings: Settings,
    section: Section,
    metric: str,
    suffix: str,
) -> tuple[pd.DataFrame | None, Path | None]:
    filename = f"{metric}{suffix}"
    for folder in section.folders:
        path = settings.analysis_root / folder / filename
        if path.is_file():
            return pd.read_csv(path), path
    return None, None


def _iqr_display_rows(
    frame: pd.DataFrame,
    metric: str,
) -> tuple[pd.DataFrame, dict[str, int]]:
    pieces: list[pd.DataFrame] = []
    removed: dict[str, int] = {}
    for group in GROUP_ORDER:
        subset = frame[frame["group"].astype(str).eq(group)].copy()
        values = pd.to_numeric(subset[metric], errors="coerce")
        finite = subset[values.notna()].copy()
        values = pd.to_numeric(finite[metric], errors="coerce")
        if values.empty:
            removed[group] = 0
            continue
        q1 = float(values.quantile(0.25))
        q3 = float(values.quantile(0.75))
        iqr = q3 - q1
        keep = values.between(q1 - 1.5 * iqr, q3 + 1.5 * iqr)
        removed[group] = int((~keep).sum())
        pieces.append(finite.loc[keep].copy())
    if not pieces:
        return frame.iloc[0:0].copy(), removed
    return pd.concat(pieces, ignore_index=True), removed


def subject_metric_analysis(
    settings: Settings,
    section: Section,
    metric: str,
) -> dict[str, Any]:
    frame, source = subject_frame(settings, section)
    if metric not in frame.columns:
        raise KeyError(f"unknown metric: {metric}")
    keep = [
        column
        for column in ("subject_id", "group", "age", "sex", "phase", metric)
        if column in frame.columns
    ]
    values = frame[keep].copy()
    values[metric] = pd.to_numeric(values[metric], errors="coerce")
    values = values[
        values["group"].astype(str).isin(GROUP_ORDER)
        & values[metric].notna()
    ].copy()

    stored_desc, desc_path = _stored_metric_table(
        settings,
        section,
        metric,
        "_descriptives.csv",
    )
    if stored_desc is None:
        stored_desc = _descriptives(values, metric)
    stored_pairwise, pairwise_path = _stored_metric_table(
        settings,
        section,
        metric,
        "_pairwise.csv",
    )
    display, removed = _iqr_display_rows(values, metric)
    sources = [source]
    if desc_path:
        sources.append(desc_path)
    if pairwise_path:
        sources.append(pairwise_path)
    return {
        "metric": metric,
        "description": METRIC_DESCRIPTIONS.get(metric),
        "source": _relative(settings, source),
        "statistic_source": (
            "stored analysis outputs"
            if desc_path or pairwise_path
            else "computed descriptively from the source subject table"
        ),
        "rows": records(values),
        "display_rows": records(display),
        "display_removed": removed,
        "descriptives": records(stored_desc),
        "pairwise": records(stored_pairwise)
        if stored_pairwise is not None
        else [],
        "sources": [_relative(settings, path) for path in sources],
        "display_filter": (
            "Per-group 1.5-IQR filtering is visual only. All stored "
            "statistics and downloadable rows retain every finite observation."
        ),
    }


def node_metric_profile(
    settings: Settings,
    section: Section,
) -> dict[str, Any]:
    config = NODE_SOURCES.get(section.id)
    if config is None:
        return {
            "available": False,
            "source": None,
            "metrics": [],
            "nodes": 0,
        }
    path = settings.analysis_root / config.folder / config.filename
    if not path.is_file():
        return {
            "available": False,
            "source": _relative(settings, path),
            "metrics": [],
            "nodes": 0,
        }
    header = pd.read_csv(path, nrows=5000)
    if config.layout == "metric-value":
        available = set(header.get("metric", pd.Series(dtype=str)).astype(str))
        metrics = [
            metric for metric in config.preferred_metrics if metric in available
        ]
    else:
        metrics = [
            metric
            for metric in config.preferred_metrics
            if metric in header.columns
        ]
    nodes = (
        int(pd.to_numeric(header["node"], errors="coerce").nunique())
        if "node" in header.columns
        else 0
    )
    return {
        "available": bool(metrics),
        "source": _relative(settings, path),
        "layout": config.layout,
        "metrics": [
            {
                "id": metric,
                "label": metric.replace("_", " "),
                "description": METRIC_DESCRIPTIONS.get(
                    metric,
                    "Node-level AAL3 measure.",
                ),
            }
            for metric in metrics
        ],
        "nodes": nodes,
    }


@lru_cache(maxsize=8)
def _node_frame_cached(path_string: str, mtime_ns: int) -> pd.DataFrame:
    del mtime_ns
    return pd.read_csv(path_string)


def _node_frame(
    settings: Settings,
    section: Section,
    metric: str,
) -> tuple[pd.DataFrame, Path, str]:
    config = NODE_SOURCES.get(section.id)
    if config is None:
        raise FileNotFoundError(
            f"node table unavailable for {section.id}"
        )
    path = settings.analysis_root / config.folder / config.filename
    frame = _node_frame_cached(
        str(path),
        path.stat().st_mtime_ns,
    ).copy()
    required = {"subject_id", "group", "node", "node_name"}
    if not required.issubset(frame.columns):
        raise ValueError("node table has an invalid schema")
    if config.layout == "metric-value":
        if not {"metric", "value"}.issubset(frame.columns):
            raise ValueError("long node table has an invalid schema")
        frame = frame[frame["metric"].astype(str).eq(metric)].copy()
        value_column = "value"
    else:
        if metric not in frame.columns:
            raise KeyError(f"unknown node metric: {metric}")
        value_column = metric
    frame[value_column] = pd.to_numeric(
        frame[value_column],
        errors="coerce",
    )
    frame["node"] = pd.to_numeric(frame["node"], errors="coerce")
    frame = frame[
        frame["group"].astype(str).isin(GROUP_ORDER)
        & frame[value_column].notna()
        & frame["node"].notna()
    ].copy()
    return frame, path, value_column


def _aal_labels(settings: Settings) -> pd.DataFrame:
    labels = _read_csv(settings.aal_labels)
    labels["node"] = pd.to_numeric(labels["node"], errors="coerce")
    return labels


def bh_fdr(values: list[float]) -> list[float]:
    indexed = sorted(
        [
            (index, float(value))
            for index, value in enumerate(values)
            if pd.notna(value) and np.isfinite(float(value))
        ],
        key=lambda item: item[1],
    )
    adjusted = [math.nan] * len(values)
    total = len(indexed)
    running = 1.0
    for rank_from_end, (index, p_value) in enumerate(
        reversed(indexed),
        start=1,
    ):
        rank = total - rank_from_end + 1
        running = min(running, p_value * total / rank)
        adjusted[index] = min(running, 1.0)
    return adjusted


def node_catalog(
    settings: Settings,
    section: Section,
    metric: str,
) -> dict[str, Any]:
    frame, path, value_column = _node_frame(
        settings,
        section,
        metric,
    )
    nodes = (
        frame.groupby(["node", "node_name"], dropna=False)
        .agg(
            observations=(value_column, "count"),
            subjects=("subject_id", "nunique"),
        )
        .reset_index()
    )
    labels = _aal_labels(settings)
    keep = [
        column
        for column in (
            "node",
            "atlas_label",
            "atlas_value",
            "atlas_version",
        )
        if column in labels.columns
    ]
    nodes = nodes.merge(labels[keep], on="node", how="left")
    nodes["atlas_label"] = nodes["atlas_label"].fillna(nodes["node_name"])
    tests_path = path.parent / f"{metric}_node_tests.csv"
    if tests_path.is_file():
        tests = pd.read_csv(tests_path)
        test_columns = [
            column
            for column in ("node", "kw_stat", "kw_p", "kw_q", "n")
            if column in tests.columns
        ]
        if "node" in test_columns:
            tests["node"] = pd.to_numeric(tests["node"], errors="coerce")
            nodes = nodes.merge(
                tests[test_columns],
                on="node",
                how="left",
            )
    nodes = nodes.sort_values(["node", "node_name"]).reset_index(drop=True)
    return {
        "metric": metric,
        "source": _relative(settings, path),
        "nodes": records(nodes),
    }


@lru_cache(maxsize=48)
def _node_pairwise_cached(
    path_string: str,
    mtime_ns: int,
    label_path_string: str,
    label_mtime_ns: int,
    layout: str,
    metric: str,
    group_a: str,
    group_b: str,
) -> pd.DataFrame:
    del mtime_ns, label_mtime_ns
    frame = pd.read_csv(path_string)
    if layout == "metric-value":
        frame = frame[frame["metric"].astype(str).eq(metric)].copy()
        value_column = "value"
    else:
        value_column = metric
    frame[value_column] = pd.to_numeric(
        frame[value_column],
        errors="coerce",
    )
    frame = frame[
        frame["group"].astype(str).isin((group_a, group_b))
        & frame[value_column].notna()
    ].copy()
    rows: list[dict[str, Any]] = []
    for (node, node_name), subset in frame.groupby(
        ["node", "node_name"],
        dropna=False,
    ):
        first = subset.loc[
            subset["group"].astype(str).eq(group_a),
            value_column,
        ].dropna()
        second = subset.loc[
            subset["group"].astype(str).eq(group_b),
            value_column,
        ].dropna()
        if len(first) < 5 or len(second) < 5:
            continue
        test = stats.mannwhitneyu(
            first,
            second,
            alternative="two-sided",
        )
        first_median = float(first.median())
        second_median = float(second.median())
        rows.append(
            {
                "node": int(node),
                "node_name": str(node_name),
                "metric": metric,
                "comparison": f"{group_a} vs {group_b}",
                f"n_{group_a}": int(len(first)),
                f"n_{group_b}": int(len(second)),
                f"median_{group_a}": first_median,
                f"median_{group_b}": second_median,
                "median_diff_a_minus_b": (
                    first_median - second_median
                ),
                "higher_group": (
                    group_a
                    if first_median > second_median
                    else group_b
                    if second_median > first_median
                    else "tie"
                ),
                "p_value": float(test.pvalue),
                "cliffs_delta": (
                    2.0
                    * float(test.statistic)
                    / float(len(first) * len(second))
                    - 1.0
                ),
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["q_value"] = bh_fdr(result["p_value"].tolist())
    result = result.sort_values(
        ["p_value", "q_value", "node"],
        na_position="last",
    ).reset_index(drop=True)
    result.insert(0, "rank", range(1, len(result) + 1))
    labels = pd.read_csv(label_path_string)
    labels["node"] = pd.to_numeric(labels["node"], errors="coerce")
    keep = [
        column
        for column in (
            "node",
            "atlas_label",
            "atlas_value",
            "atlas_version",
        )
        if column in labels.columns
    ]
    result = result.merge(labels[keep], on="node", how="left")
    result["atlas_label"] = result["atlas_label"].fillna(
        result["node_name"]
    )
    return result


def node_pairwise_ranking(
    settings: Settings,
    section: Section,
    metric: str,
    group_a: str,
    group_b: str,
) -> dict[str, Any]:
    if (group_a, group_b) not in GROUP_PAIRS:
        raise ValueError("group pair must be CN-MCI, CN-AD, or MCI-AD")
    config = NODE_SOURCES.get(section.id)
    if config is None:
        raise FileNotFoundError(
            f"node table unavailable for {section.id}"
        )
    path = settings.analysis_root / config.folder / config.filename
    labels = settings.aal_labels
    result = _node_pairwise_cached(
        str(path),
        path.stat().st_mtime_ns,
        str(labels),
        labels.stat().st_mtime_ns,
        config.layout,
        metric,
        group_a,
        group_b,
    ).copy()
    return {
        "metric": metric,
        "group_a": group_a,
        "group_b": group_b,
        "source": _relative(settings, path),
        "method": (
            "Two-sided Mann-Whitney U per AAL3 node; Cliff delta is "
            "first group minus second group; Benjamini-Hochberg correction "
            "is applied across all tested nodes for this contrast."
        ),
        "rows": records(result),
    }


def node_cross_comparison(
    settings: Settings,
    section: Section,
    metric: str,
    top_n: int,
) -> dict[str, Any]:
    if top_n < 1 or top_n > 166:
        raise ValueError("top_n must be between 1 and 166")
    frames: list[pd.DataFrame] = []
    for group_a, group_b in GROUP_PAIRS:
        payload = node_pairwise_ranking(
            settings,
            section,
            metric,
            group_a,
            group_b,
        )
        ranking = pd.DataFrame(payload["rows"]).head(top_n).copy()
        if ranking.empty:
            continue
        ranking["direction"] = ranking["higher_group"].map(
            lambda value: (
                f"{group_a}>{group_b}"
                if value == group_a
                else f"{group_a}<{group_b}"
                if value == group_b
                else f"{group_a}={group_b}"
            )
        )
        frames.append(ranking)
    if not frames:
        return {
            "metric": metric,
            "top_n": top_n,
            "rows": [],
        }
    combined = pd.concat(frames, ignore_index=True)
    combined["p_value"] = pd.to_numeric(
        combined["p_value"],
        errors="coerce",
    )
    combined["q_value"] = pd.to_numeric(
        combined["q_value"],
        errors="coerce",
    )
    combined["nominal"] = combined["p_value"].lt(0.05)
    combined["fdr"] = combined["q_value"].lt(0.05)
    selected = combined[combined["nominal"]].copy()
    if selected.empty:
        selected = combined.copy()
    summary = (
        selected.groupby(
            ["node", "node_name", "atlas_label"],
            dropna=False,
        )
        .agg(
            top_n_occurrences=("comparison", "nunique"),
            fdr_significant_comparisons=("fdr", "sum"),
            best_p=("p_value", "min"),
            best_q=("q_value", "min"),
            comparisons=(
                "comparison",
                lambda values: "; ".join(
                    dict.fromkeys(map(str, values))
                ),
            ),
            directions=(
                "direction",
                lambda values: "; ".join(
                    dict.fromkeys(map(str, values))
                ),
            ),
        )
        .reset_index()
        .sort_values(
            [
                "fdr_significant_comparisons",
                "top_n_occurrences",
                "best_p",
                "best_q",
            ],
            ascending=[False, False, True, True],
        )
        .reset_index(drop=True)
    )
    summary.insert(0, "rank", range(1, len(summary) + 1))
    return {
        "metric": metric,
        "top_n": top_n,
        "method": (
            "Nodes repeated within the selected Top-N across CN-MCI, "
            "CN-AD, and MCI-AD rankings. Nominal p<0.05 rows are summarized "
            "when present; FDR-significant contrast counts remain explicit."
        ),
        "rows": records(summary),
    }


def node_values(
    settings: Settings,
    section: Section,
    metric: str,
    node: int,
) -> dict[str, Any]:
    frame, path, value_column = _node_frame(
        settings,
        section,
        metric,
    )
    subset = frame[frame["node"].astype(int).eq(int(node))].copy()
    if subset.empty:
        raise KeyError(f"unknown node: {node}")
    keep = [
        column
        for column in (
            "subject_id",
            "group",
            "node",
            "node_name",
            "age",
            "sex",
            "phase",
            value_column,
        )
        if column in subset.columns
    ]
    values = subset[keep].rename(columns={value_column: "value"})
    display, removed = _iqr_display_rows(
        values.rename(columns={"value": metric}),
        metric,
    )
    display = display.rename(columns={metric: "value"})
    desc_source = values.rename(columns={"value": metric})
    return {
        "metric": metric,
        "node": int(node),
        "node_name": str(values["node_name"].iloc[0]),
        "source": _relative(settings, path),
        "rows": records(values),
        "display_rows": records(display),
        "display_removed": removed,
        "descriptives": records(_descriptives(desc_source, metric)),
        "display_filter": (
            "Per-group 1.5-IQR filtering is visual only. Inferential "
            "rankings use every finite subject observation."
        ),
    }


def demographics_summary(settings: Settings) -> dict[str, Any]:
    root = settings.analysis_root
    master_path = root / "00_master/master_cohort.csv"
    group_path = root / "02_demographics/group_counts.csv"
    completeness_path = root / "01_qc/data_completeness.csv"
    age_desc_path = root / "02_demographics/age_descriptives.csv"
    age_pairwise_path = root / "02_demographics/age_pairwise.csv"
    clinical_path = root / "02_demographics/clinical_summary.csv"
    master = _read_csv(master_path)
    group = _read_csv(group_path)
    completeness = _read_csv(completeness_path)
    keep_completeness = [
        column
        for column in (
            "group",
            "n_step7_tracks_done",
            "n_step7_sift2_done",
            "n_step7_parc_done",
            "n_step7_dti_done",
            "n_with_final_connectomes",
        )
        if column in completeness.columns
    ]
    cohort = group.merge(
        completeness[keep_completeness],
        on="group",
        how="left",
    )
    age_column = "age" if "age" in master.columns else "Age"
    sex_column = "sex" if "sex" in master.columns else "Sex"
    subject_column = (
        "subject_id" if "subject_id" in master.columns else "Subject ID"
    )
    age_rows = master[
        [subject_column, "group", age_column, sex_column]
    ].rename(
        columns={
            subject_column: "subject_id",
            age_column: "age",
            sex_column: "sex",
        }
    )
    age_rows["age"] = pd.to_numeric(age_rows["age"], errors="coerce")
    age_rows = age_rows[
        age_rows["group"].astype(str).isin(GROUP_ORDER)
        & age_rows["age"].notna()
    ]
    clinical_metrics: list[dict[str, Any]] = []
    if clinical_path.is_file():
        clinical = pd.read_csv(clinical_path)
        for metric in clinical.get("metric", pd.Series(dtype=str)).dropna():
            slug = {
                "MMSE Total Score": "mmse_total_score",
                "Global CDR": "global_cdr",
                "NPI-Q Total Score": "npi-q_total_score",
                "GDSCALE Total Score": "gdscale_total_score",
                "FAQ Total Score": "faq_total_score",
            }.get(
                str(metric),
                str(metric).lower().replace(" ", "_"),
            )
            desc_path = root / "02_demographics" / f"{slug}_descriptives.csv"
            pair_path = root / "02_demographics" / f"{slug}_pairwise.csv"
            clinical_metrics.append(
                {
                    "metric": str(metric),
                    "descriptives": records(pd.read_csv(desc_path))
                    if desc_path.is_file()
                    and desc_path.stat().st_size > 1
                    else [],
                    "pairwise": records(pd.read_csv(pair_path))
                    if pair_path.is_file()
                    and pair_path.stat().st_size > 1
                    else [],
                }
            )
    return {
        "cohort": records(cohort),
        "age_rows": records(age_rows),
        "age_descriptives": records(_read_csv(age_desc_path)),
        "age_pairwise": records(_read_csv(age_pairwise_path)),
        "clinical": clinical_metrics,
        "definitions": {
            "analysis_cohort": (
                "Subjects in the authoritative master cohort used by the "
                "current AAL3 analysis outputs."
            ),
            "diffusivity_subset": (
                "Subjects with the required FA/MD/AxD/RD matrices; graph-only "
                "analyses can retain structurally complete cases without all "
                "diffusion-weighted matrices."
            ),
            "pairwise": (
                "Stored Brunner-Munzel and Mann-Whitney results with their "
                "recorded multiplicity-adjusted q values."
            ),
        },
        "sources": [
            _relative(settings, path)
            for path in (
                master_path,
                group_path,
                completeness_path,
                age_desc_path,
                age_pairwise_path,
                clinical_path,
            )
            if path.is_file()
        ],
    }
