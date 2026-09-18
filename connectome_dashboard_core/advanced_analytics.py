from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import stats

from .section_analytics import bh_fdr
from .serialization import records
from .settings import Settings


ML_TASKS = {
    "diagnosis": {
        "label": "CN / MCI / AD classification",
        "kind": "classification",
        "performance": "ml_model_performance.csv",
        "status": "ml_model_status.csv",
        "detail": "ml_class_metrics.csv",
        "confusion": "ml_confusion_matrix.csv",
        "predictions": "ml_cross_validated_predictions.csv",
        "importance": "ml_feature_importance.csv",
        "roc": "ml_roc_curve.csv",
        "shap": "ml_shap_contributions.csv",
    },
    "cdr": {
        "label": "Global CDR classification",
        "kind": "classification",
        "performance": "cdr_classification_model_performance.csv",
        "status": "cdr_classification_model_status.csv",
        "detail": "cdr_classification_class_metrics.csv",
        "confusion": "cdr_classification_confusion_matrix.csv",
        "predictions": "cdr_classification_cv_predictions.csv",
        "importance": "cdr_classification_feature_importance.csv",
    },
    "score": {
        "label": "MMSE score prediction",
        "kind": "regression",
        "performance": "score_prediction_model_performance.csv",
        "status": "score_prediction_model_status.csv",
        "detail": "score_prediction_group_performance.csv",
        "predictions": "score_prediction_cv_predictions.csv",
        "importance": "score_prediction_feature_importance.csv",
        "stage": "score_prediction_mci_stage_summary.csv",
    },
}

NETWORK_FAMILIES = {
    "microstructure": (
        "Microstructure (FA / MD / AxD / RD)",
        "network_microstructure_subject.csv",
        "network_microstructure_stats.csv",
    ),
    "graph": (
        "Graph topology",
        "network_graph_subject.csv",
        "network_graph_stats.csv",
    ),
    "coupling": (
        "Topology-microstructure coupling",
        "network_coupling_subject.csv",
        "network_coupling_stats.csv",
    ),
}


def _analysis_path(settings: Settings, *parts: str) -> Path:
    return settings.analysis_root.joinpath(*parts)


def _relative(settings: Settings, path: Path) -> str:
    return path.relative_to(settings.analysis_root).as_posix()


def _optional_csv(path: Path) -> pd.DataFrame:
    if not path.is_file() or path.stat().st_size <= 1:
        return pd.DataFrame()
    return pd.read_csv(path)


def ml_task_catalog(settings: Settings) -> dict[str, Any]:
    root = _analysis_path(settings, "18_ml_diagnostics")
    tasks: list[dict[str, Any]] = []
    for task_id, spec in ML_TASKS.items():
        performance = _optional_csv(root / spec["performance"])
        tasks.append(
            {
                "id": task_id,
                "label": spec["label"],
                "kind": spec["kind"],
                "available": not performance.empty,
                "models": sorted(
                    performance.get(
                        "model",
                        pd.Series(dtype=str),
                    )
                    .dropna()
                    .astype(str)
                    .unique()
                    .tolist()
                ),
                "targets": sorted(
                    performance.get(
                        "target",
                        pd.Series(dtype=str),
                    )
                    .dropna()
                    .astype(str)
                    .unique()
                    .tolist()
                ),
                "roles": sorted(
                    performance.get(
                        "analysis_role",
                        pd.Series(dtype=str),
                    )
                    .dropna()
                    .astype(str)
                    .unique()
                    .tolist()
                ),
            }
        )
    return {
        "tasks": tasks,
        "contract": {
            "validation": (
                "Reported performance comes from recorded cross-validated "
                "predictions; the dashboard does not refit models."
            ),
            "interpretation": (
                "Feature importance and SHAP are model-attribution summaries, "
                "not causal effects or independent confirmatory tests."
            ),
            "clinical_scope": (
                "Clinical-target tasks use only the recorded target-matching "
                "cohort and must not be conflated with the full 530-subject "
                "diagnostic cohort."
            ),
        },
    }


def ml_task_data(
    settings: Settings,
    task: str,
) -> dict[str, Any]:
    if task not in ML_TASKS:
        raise KeyError(f"unknown ML task: {task}")
    spec = ML_TASKS[task]
    root = _analysis_path(settings, "18_ml_diagnostics")
    frames: dict[str, pd.DataFrame] = {}
    sources: list[str] = []
    for key, filename in spec.items():
        if key in {"label", "kind"}:
            continue
        path = root / filename
        frame = _optional_csv(path)
        frames[key] = frame
        if path.is_file():
            sources.append(_relative(settings, path))
    performance = frames.get("performance", pd.DataFrame())
    if performance.empty:
        raise FileNotFoundError(spec["performance"])
    return {
        "id": task,
        "label": spec["label"],
        "kind": spec["kind"],
        "performance": records(performance),
        "status": records(frames.get("status", pd.DataFrame())),
        "detail": records(frames.get("detail", pd.DataFrame())),
        "confusion": records(frames.get("confusion", pd.DataFrame())),
        "predictions": records(frames.get("predictions", pd.DataFrame())),
        "importance": records(frames.get("importance", pd.DataFrame())),
        "roc": records(frames.get("roc", pd.DataFrame())),
        "shap": records(frames.get("shap", pd.DataFrame())),
        "stage": records(frames.get("stage", pd.DataFrame())),
        "sources": sources,
    }


def network_catalog(settings: Settings) -> dict[str, Any]:
    result: dict[str, Any] = {"schemes": []}
    for scheme in ("functional", "anatomical"):
        root = _analysis_path(settings, "19_network_analysis", scheme)
        families: list[dict[str, Any]] = []
        for family_id, (label, subject_name, stats_name) in (
            NETWORK_FAMILIES.items()
        ):
            subject_path = root / subject_name
            frame = _optional_csv(subject_path)
            metrics = sorted(
                frame.get("metric", pd.Series(dtype=str))
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )
            families.append(
                {
                    "id": family_id,
                    "label": label,
                    "available": bool(metrics),
                    "metrics": metrics,
                    "subject_source": _relative(
                        settings,
                        subject_path,
                    ),
                    "stats_source": _relative(
                        settings,
                        root / stats_name,
                    ),
                }
            )
        result["schemes"].append(
            {
                "id": scheme,
                "label": (
                    "Functional networks"
                    if scheme == "functional"
                    else "Anatomical systems"
                ),
                "families": families,
            }
        )
    result["mapping_contract"] = {
        "functional": (
            "Documented Yeo-7-inspired approximation on AAL3 anatomy; "
            "mapping-sensitive and not native voxelwise overlap."
        ),
        "anatomical": (
            "Deterministic anatomical-system grouping from the recorded "
            "AAL3 crosswalk."
        ),
        "modality": (
            "All displayed connectivity is structural; network labels do "
            "not convert tractography into functional connectivity."
        ),
    }
    return result


def _network_root(settings: Settings, scheme: str) -> Path:
    if scheme not in {"functional", "anatomical"}:
        raise ValueError("scheme must be functional or anatomical")
    return _analysis_path(settings, "19_network_analysis", scheme)


def network_distribution(
    settings: Settings,
    scheme: str,
    family: str,
    metric: str,
) -> dict[str, Any]:
    if family not in NETWORK_FAMILIES:
        raise KeyError(f"unknown network family: {family}")
    label, subject_name, stats_name = NETWORK_FAMILIES[family]
    root = _network_root(settings, scheme)
    subject_path = root / subject_name
    stats_path = root / stats_name
    subject = _optional_csv(subject_path)
    stats_frame = _optional_csv(stats_path)
    if subject.empty or metric not in set(subject["metric"].astype(str)):
        raise KeyError(f"unknown network metric: {metric}")
    values = subject[subject["metric"].astype(str).eq(metric)].copy()
    stats_rows = (
        stats_frame[stats_frame["metric"].astype(str).eq(metric)].copy()
        if not stats_frame.empty and "metric" in stats_frame.columns
        else pd.DataFrame()
    )
    return {
        "scheme": scheme,
        "family": family,
        "family_label": label,
        "metric": metric,
        "rows": records(values),
        "statistics": records(stats_rows),
        "sources": [
            _relative(settings, subject_path),
            _relative(settings, stats_path),
        ],
    }


def network_affectedness(
    settings: Settings,
    scheme: str,
    family: str | None = None,
) -> dict[str, Any]:
    root = _network_root(settings, scheme)
    path = root / "network_affectedness_summary.csv"
    frame = _optional_csv(path)
    if family:
        frame = frame[
            frame["feature_family"].astype(str).eq(family)
        ].copy()
    if "cn_ad_cliffs_delta" in frame.columns:
        frame["_absolute_effect"] = pd.to_numeric(
            frame["cn_ad_cliffs_delta"],
            errors="coerce",
        ).abs()
        frame = frame.sort_values(
            "_absolute_effect",
            ascending=False,
            na_position="last",
        ).drop(columns="_absolute_effect")
    return {
        "scheme": scheme,
        "families": sorted(
            frame.get("feature_family", pd.Series(dtype=str))
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        ),
        "rows": records(frame),
        "source": _relative(settings, path),
    }


def network_blocks(
    settings: Settings,
    scheme: str,
    metric: Literal["density", "mean_weight"],
) -> dict[str, Any]:
    root = _network_root(settings, scheme)
    filename = (
        "block_groupmean_density.csv"
        if metric == "density"
        else "block_groupmean_mean_weight.csv"
    )
    path = root / filename
    stats_path = root / "block_stats.csv"
    frame = _optional_csv(path)
    block_stats = _optional_csv(stats_path)
    networks = sorted(
        set(frame.get("net_a", pd.Series(dtype=str)).dropna().astype(str))
        | set(frame.get("net_b", pd.Series(dtype=str)).dropna().astype(str))
    )
    return {
        "scheme": scheme,
        "metric": metric,
        "networks": networks,
        "rows": records(frame),
        "statistics": records(block_stats),
        "sources": [
            _relative(settings, path),
            _relative(settings, stats_path),
        ],
    }


@lru_cache(maxsize=16)
def _coupling_aal_long_cached(
    graph_path_string: str,
    graph_mtime_ns: int,
    micro_path_string: str,
    micro_mtime_ns: int,
    label_path_string: str,
    label_mtime_ns: int,
    topology_metric: str,
    micro_metric: str,
) -> pd.DataFrame:
    del graph_mtime_ns, micro_mtime_ns, label_mtime_ns
    graph_path = Path(graph_path_string)
    micro_path = Path(micro_path_string)
    graph_needed = {
        "subject_id",
        "group",
        "node",
        "node_name",
        topology_metric,
    }
    micro_needed = {
        "subject_id",
        "group",
        "node",
        "node_name",
        "metric",
        "value",
    }
    graph = pd.read_csv(
        graph_path,
        usecols=lambda column: column in graph_needed,
    )
    micro = pd.read_csv(
        micro_path,
        usecols=lambda column: column in micro_needed,
    )
    if (
        not graph_needed.issubset(graph.columns)
        or not micro_needed.issubset(micro.columns)
    ):
        raise ValueError("coupling-AAL source schema is incomplete")
    graph = graph.rename(
        columns={
            "group": "group_graph",
            topology_metric: "topology_value",
        }
    )
    micro = micro[micro["metric"].astype(str).eq(micro_metric)].copy()
    micro = micro.rename(
        columns={"group": "group_micro", "value": "micro_value"}
    )
    merged = graph.merge(
        micro,
        on=["subject_id", "node", "node_name"],
        how="inner",
    )
    merged["group"] = (
        merged["group_graph"]
        .fillna(merged["group_micro"])
        .astype(str)
    )
    for column in ("topology_value", "micro_value"):
        merged[column] = pd.to_numeric(merged[column], errors="coerce")
    merged = merged.dropna(
        subset=["topology_value", "micro_value"]
    ).copy()
    topology_mean = merged.groupby("subject_id")[
        "topology_value"
    ].transform("mean")
    micro_mean = merged.groupby("subject_id")[
        "micro_value"
    ].transform("mean")
    topology_sd = merged.groupby("subject_id")[
        "topology_value"
    ].transform(lambda values: values.std(ddof=0))
    micro_sd = merged.groupby("subject_id")[
        "micro_value"
    ].transform(lambda values: values.std(ddof=0))
    merged["topology_z"] = np.where(
        topology_sd > 0,
        (merged["topology_value"] - topology_mean) / topology_sd,
        np.nan,
    )
    merged["micro_z"] = np.where(
        micro_sd > 0,
        (merged["micro_value"] - micro_mean) / micro_sd,
        np.nan,
    )
    merged["coupling_index"] = (
        merged["topology_z"] * merged["micro_z"]
    )
    merged["micro_topology_ratio"] = np.where(
        merged["topology_value"].abs() > 0,
        merged["micro_value"] / merged["topology_value"],
        np.nan,
    )
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
    merged = merged.merge(labels[keep], on="node", how="left")
    merged["atlas_label"] = merged["atlas_label"].fillna(
        merged["node_name"]
    )
    return merged


def coupling_aal_catalog(settings: Settings) -> dict[str, Any]:
    return {
        "topology_metrics": ["strength", "degree", "nodal_eff"],
        "microstructure_metrics": [
            "fa_mean",
            "md_mean",
            "ad_mean",
            "rd_mean",
        ],
        "value_metrics": [
            "coupling_index",
            "micro_topology_ratio",
        ],
        "definition": {
            "coupling_index": (
                "Within each subject, z-score the chosen topology and "
                "microstructure measures across matched AAL3 nodes, then "
                "multiply the two z-scores at each node."
            ),
            "micro_topology_ratio": (
                "Microstructure value divided by topology value at the "
                "same subject and AAL3 node, where topology is non-zero."
            ),
            "inference": (
                "Two-sided Mann-Whitney U per node with Cliff delta and "
                "Benjamini-Hochberg correction across AAL3 nodes."
            ),
        },
    }


def _coupling_aal_sources(
    settings: Settings,
) -> tuple[Path, Path, Path]:
    return (
        _analysis_path(
            settings,
            "07_node_graph_live",
            "live_node_graph_long.csv",
        ),
        _analysis_path(
            settings,
            "04_node_microstructure_live",
            "live_node_microstructure_long.csv",
        ),
        settings.aal_labels,
    )


def coupling_aal_ranking(
    settings: Settings,
    topology_metric: str,
    micro_metric: str,
    value_metric: str,
    group_a: str,
    group_b: str,
) -> dict[str, Any]:
    catalog = coupling_aal_catalog(settings)
    if topology_metric not in catalog["topology_metrics"]:
        raise ValueError("unknown topology metric")
    if micro_metric not in catalog["microstructure_metrics"]:
        raise ValueError("unknown microstructure metric")
    if value_metric not in catalog["value_metrics"]:
        raise ValueError("unknown coupling-AAL value metric")
    if (group_a, group_b) not in (
        ("CN", "MCI"),
        ("CN", "AD"),
        ("MCI", "AD"),
    ):
        raise ValueError("invalid diagnostic contrast")
    graph_path, micro_path, label_path = _coupling_aal_sources(settings)
    frame = _coupling_aal_long_cached(
        str(graph_path),
        graph_path.stat().st_mtime_ns,
        str(micro_path),
        micro_path.stat().st_mtime_ns,
        str(label_path),
        label_path.stat().st_mtime_ns,
        topology_metric,
        micro_metric,
    )
    frame = frame[
        frame["group"].astype(str).isin((group_a, group_b))
    ].copy()
    frame[value_metric] = pd.to_numeric(
        frame[value_metric],
        errors="coerce",
    )
    frame = frame.dropna(subset=[value_metric])
    rows: list[dict[str, Any]] = []
    for (node, node_name, atlas_label), subset in frame.groupby(
        ["node", "node_name", "atlas_label"],
        dropna=False,
    ):
        first = subset.loc[
            subset["group"].astype(str).eq(group_a),
            value_metric,
        ].dropna()
        second = subset.loc[
            subset["group"].astype(str).eq(group_b),
            value_metric,
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
                "atlas_label": str(atlas_label),
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
    if not result.empty:
        result["q_value"] = bh_fdr(result["p_value"].tolist())
        result = result.sort_values(
            ["p_value", "q_value", "node"],
            na_position="last",
        ).reset_index(drop=True)
        result.insert(0, "rank", range(1, len(result) + 1))
    return {
        "topology_metric": topology_metric,
        "microstructure_metric": micro_metric,
        "value_metric": value_metric,
        "group_a": group_a,
        "group_b": group_b,
        "rows": records(result),
        "sources": [
            _relative(settings, graph_path),
            _relative(settings, micro_path),
            str(label_path),
        ],
    }


def novel_findings_summary(settings: Settings) -> dict[str, Any]:
    root = _analysis_path(settings, "20_findings")
    cards_path = root / "novelty_cards.json"
    catalog_path = root / "findings_catalog.csv"
    report_path = root / "novelty_report.md"
    cards: list[dict[str, Any]] = []
    if cards_path.is_file():
        parsed = json.loads(cards_path.read_text(encoding="utf-8"))
        if isinstance(parsed, list):
            cards = [
                item for item in parsed if isinstance(item, dict)
            ]
    catalog = _optional_csv(catalog_path)
    if not catalog.empty:
        catalog["q"] = pd.to_numeric(catalog["q"], errors="coerce")
        catalog = catalog.sort_values(
            ["q", "feature_family", "network", "metric"],
            na_position="last",
        ).reset_index(drop=True)
    family_counts: list[dict[str, Any]] = []
    if not catalog.empty:
        family_counts = records(
            catalog.groupby(
                ["scheme", "feature_family"],
                dropna=False,
            )
            .size()
            .reset_index(name="fdr_supported_findings")
            .sort_values(
                ["fdr_supported_findings", "scheme", "feature_family"],
                ascending=[False, True, True],
            )
        )
    confidence_counts: dict[str, int] = {}
    for card in cards:
        confidence = str(card.get("confidence", "unspecified"))
        confidence_counts[confidence] = confidence_counts.get(
            confidence,
            0,
        ) + 1
    return {
        "cards": cards,
        "catalog": records(catalog),
        "family_counts": family_counts,
        "confidence_counts": confidence_counts,
        "report_available": report_path.is_file(),
        "contract": {
            "claim_scope": (
                "These are literature-grounded exploratory hypotheses "
                "derived from existing FDR-controlled or recorded "
                "cross-validated outputs; they are not independent "
                "replication or causal claims."
            ),
            "functional_boundary": (
                "All connectomes are structural. Functional-network "
                "labels use the documented AAL3 mapping approximation."
            ),
            "literature_boundary": (
                "The attached citations support context, but retrieval "
                "is not an exhaustive systematic review."
            ),
        },
        "sources": [
            _relative(settings, path)
            for path in (
                cards_path,
                catalog_path,
                report_path,
            )
            if path.is_file()
        ],
    }
