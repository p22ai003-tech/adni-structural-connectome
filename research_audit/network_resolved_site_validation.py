#!/usr/bin/env python3
"""Fold-safe site-held-out validation of network-resolved two-axis features.

This is a tertiary coherence analysis for the historical discovery matrices.
It compares demographic/acquisition baselines, scalar network composites, and
network-resolved AxD/RD feature sets under nested site-grouped validation.  It
does not establish an independently validated biomarker and never writes
participant-level predictions or identifiers.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from stage_specific_diffusivity_analysis import PRIMARY_NETWORKS, eligible_cohort
from submission_sprint_analysis import INPUTS, collect_metrics, load_exact_cohort, sha256


PROJECT = Path("/home/ec2-user/exp")
DEFAULT_OUT = PROJECT / "research_audit" / "outputs" / "network_resolved_site_validation_v2"
BOOTSTRAP_ITERATIONS = 2000
SEED = 20260718
ENDPOINTS = (
    ("MCI_vs_CN", "MCI", "CN", "ad_mean", "axial"),
    ("AD_vs_MCI", "AD", "MCI", "rd_mean", "radial"),
)


def build_feature_frame(cohort: pd.DataFrame, metrics: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    exact = eligible_cohort(cohort).copy()
    micro = metrics[
        metrics["mapping"].isin(["anatomical", "functional"])
        & metrics["feature_family"].eq("microstructure")
        & metrics["metric"].isin(["ad_mean", "rd_mean"])
    ].copy()
    micro["network"] = micro["mapping"].astype(str) + "::" + micro["network"].astype(str)
    micro = micro[micro["network"].isin(PRIMARY_NETWORKS)].copy()
    micro["feature"] = micro["metric"].astype(str) + "::" + micro["network"].astype(str)
    wide = micro.pivot(index="subject_id", columns="feature", values="value")
    wide.columns = [str(column) for column in wide.columns]
    wide = wide.reset_index()
    frame = exact[
        ["subject_id", "group", "site", "age", "log_gap_days", "sex", "t1_source"]
    ].merge(wide, on="subject_id", how="inner", validate="1:1")
    feature_sets: dict[str, list[str]] = {}
    for metric, component in (("ad_mean", "axial"), ("rd_mean", "radial")):
        columns = [f"{metric}::{network}" for network in PRIMARY_NETWORKS]
        missing = sorted(set(columns) - set(frame.columns))
        if missing:
            raise ValueError(f"Missing network features: {missing}")
        feature_sets[component] = columns
        frame[f"{component}_composite"] = frame[columns].mean(axis=1, skipna=True)
    return frame, feature_sets


def _support(frame: pd.DataFrame, left: str, right: str) -> pd.DataFrame:
    use = frame[frame["group"].isin([left, right])].copy()
    common = use.groupby("site")["group"].nunique()
    use = use[use["site"].isin(common[common.eq(2)].index)].copy()
    if use["site"].nunique() < 8:
        raise ValueError("Insufficient pair-common sites")
    return use


def _pipeline(numeric: list[str], categorical: list[str]) -> Pipeline:
    transformer = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
    )
    return Pipeline(
        [
            ("preprocess", transformer),
            (
                "model",
                LogisticRegression(
                    penalty="l2",
                    solver="liblinear",
                    class_weight="balanced",
                    max_iter=3000,
                    random_state=SEED,
                ),
            ),
        ]
    )


def _nested_site_predictions(
    use: pd.DataFrame,
    positive: str,
    numeric: list[str],
    categorical: list[str],
    network_features: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[pd.DataFrame] = []
    coefficient_rows: list[dict[str, object]] = []
    y_all = use["group"].eq(positive).astype(int)
    for held_site in sorted(use["site"].astype(str).unique()):
        test_mask = use["site"].astype(str).eq(held_site)
        train = use.loc[~test_mask].copy()
        test = use.loc[test_mask].copy()
        y_train = y_all.loc[train.index]
        if y_train.nunique() != 2 or y_all.loc[test.index].nunique() != 2:
            raise ValueError("Every outer site must contain both classes")
        inner_splits = min(5, int(train["site"].nunique()))
        search = GridSearchCV(
            estimator=_pipeline(numeric, categorical),
            param_grid={"model__C": [0.01, 0.1, 1.0, 10.0]},
            scoring="roc_auc",
            cv=GroupKFold(n_splits=inner_splits),
            refit=True,
            n_jobs=4,
            error_score="raise",
        )
        features = numeric + categorical
        search.fit(train[features], y_train, groups=train["site"])
        probability = search.predict_proba(test[features])[:, 1]
        rows.append(
            pd.DataFrame(
                {
                    "site": test["site"].astype(str).to_numpy(),
                    "truth": y_all.loc[test.index].to_numpy(dtype=int),
                    "probability": probability,
                }
            )
        )
        if network_features:
            best = search.best_estimator_
            names = best.named_steps["preprocess"].get_feature_names_out()
            coefficients = best.named_steps["model"].coef_.reshape(-1)
            for name, coefficient in zip(names, coefficients):
                raw = str(name).replace("numeric__", "")
                if raw in network_features:
                    coefficient_rows.append(
                        {
                            "feature": raw,
                            "coefficient": float(coefficient),
                            "held_out_fold": held_site,
                        }
                    )
    return pd.concat(rows, ignore_index=True), pd.DataFrame(coefficient_rows)


def _metrics(predictions: pd.DataFrame) -> dict[str, float]:
    truth = predictions["truth"].to_numpy(dtype=int)
    probability = predictions["probability"].to_numpy(dtype=float)
    predicted = (probability >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(truth, predicted, labels=[0, 1]).ravel()
    return {
        "roc_auc": float(roc_auc_score(truth, probability)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predicted)),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else np.nan,
        "specificity": float(tn / (tn + fp)) if tn + fp else np.nan,
    }


def _bootstrap_deltas(predictions: dict[str, pd.DataFrame], contrast: str) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    anchor = predictions["baseline"].reset_index(drop=True)
    sites = np.array(sorted(anchor["site"].unique()))
    site_index = {site: index for index, site in enumerate(sites)}
    site_codes = anchor["site"].map(site_index).to_numpy(dtype=int)
    truth = anchor["truth"].to_numpy(dtype=int)
    rows = []
    comparisons = [
        (model, "baseline") for model in predictions if model != "baseline"
    ] + [
        ("endpoint_networks", "endpoint_composite"),
        ("two_axis_networks", "two_axis_composites"),
    ]
    for comparator, reference in comparisons:
        comparison = predictions[comparator].reset_index(drop=True)
        reference_frame = predictions[reference].reset_index(drop=True)
        if not (
            np.array_equal(comparison["site"].to_numpy(), anchor["site"].to_numpy())
            and np.array_equal(comparison["truth"].to_numpy(), truth)
            and np.array_equal(reference_frame["site"].to_numpy(), anchor["site"].to_numpy())
            and np.array_equal(reference_frame["truth"].to_numpy(), truth)
        ):
            raise ValueError("Prediction rows are not aligned across model specifications")
        comparison_probability = comparison["probability"].to_numpy(dtype=float)
        reference_probability = reference_frame["probability"].to_numpy(dtype=float)
        deltas = []
        for _ in range(BOOTSTRAP_ITERATIONS):
            site_counts = rng.multinomial(
                len(sites), np.full(len(sites), 1.0 / len(sites), dtype=float)
            )
            weights = site_counts[site_codes].astype(float)
            if weights[truth == 0].sum() == 0 or weights[truth == 1].sum() == 0:
                continue
            deltas.append(
                roc_auc_score(truth, comparison_probability, sample_weight=weights)
                - roc_auc_score(truth, reference_probability, sample_weight=weights)
            )
        values = np.asarray(deltas, dtype=float)
        rows.append(
            {
                "contrast": contrast,
                "comparator": comparator,
                "reference": reference,
                "iterations": len(values),
                "delta_auc_median": float(np.median(values)),
                "delta_auc_ci95_low": float(np.quantile(values, 0.025)),
                "delta_auc_ci95_high": float(np.quantile(values, 0.975)),
                "positive_delta_fraction": float(np.mean(values > 0)),
                "bootstrap_unit": "site",
                "evidence_status": "post-selection internal validation",
            }
        )
    return pd.DataFrame(rows)


def run_models(frame: pd.DataFrame, feature_sets: dict[str, list[str]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    delta_frames = []
    coefficient_frames = []
    baseline_numeric = ["age", "log_gap_days"]
    categorical = ["sex", "t1_source"]
    for contrast, positive, reference, endpoint_metric, component in ENDPOINTS:
        use = _support(frame, positive, reference)
        endpoint_features = feature_sets[component]
        all_networks = feature_sets["axial"] + feature_sets["radial"]
        specifications = {
            "baseline": baseline_numeric,
            "endpoint_composite": baseline_numeric + [f"{component}_composite"],
            "endpoint_networks": baseline_numeric + endpoint_features,
            "two_axis_composites": baseline_numeric + ["axial_composite", "radial_composite"],
            "two_axis_networks": baseline_numeric + all_networks,
        }
        prediction_sets: dict[str, pd.DataFrame] = {}
        for model, numeric in specifications.items():
            networks = set(numeric) & set(all_networks)
            predictions, coefficients = _nested_site_predictions(
                use,
                positive,
                numeric,
                categorical,
                networks,
            )
            prediction_sets[model] = predictions
            performance = _metrics(predictions)
            summary_rows.append(
                {
                    "contrast": contrast,
                    "positive_group": positive,
                    "reference_group": reference,
                    "model": model,
                    "n": int(len(predictions)),
                    "n_sites": int(predictions["site"].nunique()),
                    **performance,
                    "validation": "nested leave-one-site-out on pair-common sites",
                    "evidence_status": "post-selection internal validation",
                }
            )
            if not coefficients.empty:
                coefficients["contrast"] = contrast
                coefficients["model"] = model
                coefficient_frames.append(coefficients)
        delta_frames.append(_bootstrap_deltas(prediction_sets, contrast))
    coefficients = pd.concat(coefficient_frames, ignore_index=True)
    coefficient_summary = (
        coefficients.groupby(["contrast", "model", "feature"], as_index=False)
        .agg(
            outer_folds=("coefficient", "size"),
            mean_coefficient=("coefficient", "mean"),
            mean_abs_coefficient=("coefficient", lambda values: float(np.mean(np.abs(values)))),
            positive_sign_fraction=("coefficient", lambda values: float(np.mean(np.asarray(values) > 0))),
        )
    )
    coefficient_summary["component"] = coefficient_summary["feature"].str.split("::").str[0]
    coefficient_summary["network"] = coefficient_summary["feature"].str.split("::", n=1).str[1]
    coefficient_summary = coefficient_summary.sort_values(
        ["contrast", "model", "mean_abs_coefficient"], ascending=[True, True, False]
    )
    return pd.DataFrame(summary_rows), pd.concat(delta_frames, ignore_index=True), coefficient_summary


def make_figure(summary: pd.DataFrame, out: Path) -> None:
    figures = out / "figures"
    figures.mkdir(parents=True, exist_ok=False)
    order = ["baseline", "endpoint_composite", "endpoint_networks", "two_axis_composites", "two_axis_networks"]
    labels = ["Baseline", "Endpoint\ncomposite", "Endpoint\nnetworks", "Two-axis\ncomposites", "Two-axis\nnetworks"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, contrast in zip(axes, ["MCI_vs_CN", "AD_vs_MCI"]):
        frame = summary[summary["contrast"].eq(contrast)].set_index("model").loc[order]
        ax.bar(np.arange(len(order)), frame["roc_auc"], color=["#777777", "#31688e", "#35b779", "#b63679", "#f1605d"])
        ax.axhline(0.5, color="black", linewidth=0.8)
        ax.set_xticks(np.arange(len(order)), labels)
        ax.set_ylim(0.35, max(0.75, float(frame["roc_auc"].max()) + 0.06))
        ax.set_title(contrast.replace("_vs_", " vs "))
        ax.set_ylabel("Site-held-out ROC AUC")
    fig.suptitle("Nested site-held-out network-resolution analysis")
    fig.tight_layout()
    fig.savefig(figures / "network_resolved_site_held_out_auc.png", dpi=260, bbox_inches="tight")
    plt.close(fig)


def _table(frame: pd.DataFrame, columns: list[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in frame[columns].itertuples(index=False, name=None):
        values = [f"{float(value):.4g}" if isinstance(value, (float, np.floating)) else str(value) for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_summary(out: Path, summary: pd.DataFrame, deltas: pd.DataFrame) -> None:
    text = f"""# Network-resolved site-held-out validation

**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}  
**Status:** aggregate-only, nested site-held-out, post-selection internal validation.

## Performance

{_table(summary, ['contrast', 'model', 'n', 'n_sites', 'roc_auc', 'balanced_accuracy', 'sensitivity', 'specificity'])}

## Pairwise AUC increments

{_table(deltas, ['contrast', 'comparator', 'delta_auc_median', 'delta_auc_ci95_low', 'delta_auc_ci95_high', 'positive_delta_fraction'])}

All imputation, scaling, regularization selection, and model fitting occur inside the outer training data. Sites, rather than participants, define both outer and inner validation groups. Network resolution is considered supportive only if it improves over the matching scalar composite with a stable site-bootstrap interval. This analysis is not independent because the two biological axes were selected in the same historical cohort.
"""
    (out / "results_summary.md").write_text(text, encoding="utf-8")


def validate(out: Path) -> dict[str, object]:
    required = (
        "classification_summary.csv",
        "classification_increment_bootstrap.csv",
        "network_coefficient_stability.csv",
        "results_summary.md",
        "figures/network_resolved_site_held_out_auc.png",
        "run_manifest.json",
    )
    missing = [name for name in required if not (out / name).exists()]
    csvs = [pd.read_csv(out / name) for name in required if name.endswith(".csv") and (out / name).exists()]
    prohibited = {"subject_id", "participant_id", "image_id", "series_id", "site"}
    no_ids = all(not (prohibited & {column.lower() for column in frame.columns}) for frame in csvs)
    report = {
        "pass": not missing and no_ids,
        "missing_required_files": missing,
        "no_identifier_or_fold_columns": no_ids,
        "classification_rows": int(len(csvs[0])) if csvs else 0,
        "files": {},
    }
    for name in required:
        path = out / name
        if path.exists():
            report["files"][name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    return report


def run(out: Path) -> None:
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite existing release: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{out.name}.", dir=out.parent))
    try:
        cohort = load_exact_cohort()
        metrics = collect_metrics()
        frame, feature_sets = build_feature_frame(cohort, metrics)
        summary, deltas, coefficients = run_models(frame, feature_sets)
        summary.to_csv(temporary / "classification_summary.csv", index=False)
        deltas.to_csv(temporary / "classification_increment_bootstrap.csv", index=False)
        coefficients.to_csv(temporary / "network_coefficient_stability.csv", index=False)
        make_figure(summary, temporary)
        write_summary(temporary, summary, deltas)
        manifest = {
        "release": "network_resolved_site_validation_v2",
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "evidence_status": "post-selection internal validation",
            "writes_participant_data": False,
            "validation": "nested leave-one-site-out on pair-common sites",
            "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
            "primary_network_count": len(PRIMARY_NETWORKS),
            "software": {"sklearn": sklearn.__version__, "numpy": np.__version__, "pandas": pd.__version__},
            "inputs": {
                key: {"path": str(path), "sha256": sha256(path)}
                for key, path in INPUTS.items()
                if key in {"exact_manifest", "functional_micro", "anatomical_micro"}
            },
        }
        (temporary / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        report = validate(temporary)
        (temporary / "release_validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        if not report["pass"]:
            raise RuntimeError(f"Release validation failed: {report}")
        os.replace(temporary, out)
    except Exception:
        failed = out.parent / f"{temporary.name}.failed"
        os.replace(temporary, failed)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> None:
    run(parse_args().out)


if __name__ == "__main__":
    main()
