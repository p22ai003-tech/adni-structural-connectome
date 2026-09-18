#!/usr/bin/env python3
"""Aggregate-only extensions for the stage-specific two-axis phenotype.

This analysis answers two bounded questions without emitting participant-level
records:

1. Do the adjacent-stage effects vary across the 17 nonredundant systems?
2. Does the discovered two-axis phenotype add site-held-out discrimination
   beyond a demographic/acquisition baseline?

The classification analysis is explicitly post-selection internal validation;
it is not an independent diagnostic-performance estimate.
"""

from __future__ import annotations

import argparse
import hashlib
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
import statsmodels
import statsmodels.formula.api as smf
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from stage_specific_diffusivity_analysis import (
    PRIMARY_NETWORKS,
    _support_frame,
    build_composites,
    eligible_cohort,
)
from submission_sprint_analysis import (
    INPUTS,
    _standardize,
    collect_metrics,
    load_exact_cohort,
    normal_scores,
    sha256,
)


PROJECT = Path("/home/ec2-user/exp")
DEFAULT_OUT = PROJECT / "research_audit" / "outputs" / "two_axis_extensions_v1"
BOOTSTRAP_ITERATIONS = 2000
BOOTSTRAP_SEED = 20260718


def _pair_long_frame(
    cohort: pd.DataFrame,
    metrics: pd.DataFrame,
    metric: str,
    left: str,
    right: str,
) -> pd.DataFrame:
    cohort = eligible_cohort(cohort)
    meta = cohort[
        ["subject_id", "group", "age", "sex", "site", "log_gap_days", "t1_source"]
    ]
    micro = metrics[
        metrics["mapping"].isin(["anatomical", "functional"])
        & metrics["feature_family"].eq("microstructure")
        & metrics["metric"].eq(metric)
    ].copy()
    micro["network"] = micro["mapping"].astype(str) + "::" + micro["network"].astype(str)
    micro = micro[micro["network"].isin(PRIMARY_NETWORKS)].merge(
        meta, on="subject_id", how="inner", validate="m:1"
    )
    micro = micro[np.isfinite(micro["value"])].copy()
    micro["value_z"] = micro.groupby("network", sort=False)["value"].transform(
        normal_scores
    )
    micro = micro[micro["group"].isin([left, right])].copy()
    support = micro.groupby("site")["group"].nunique()
    micro = micro[micro["site"].isin(support[support.eq(2)].index)].copy()
    micro["age_z"] = _standardize(micro["age"])
    micro["gap_z"] = _standardize(micro["log_gap_days"])
    return micro


def network_heterogeneity_tests(
    cohort: pd.DataFrame, metrics: pd.DataFrame
) -> pd.DataFrame:
    specifications = (
        ("axial_MCI_vs_CN", "ad_mean", "MCI", "CN"),
        ("radial_AD_vs_MCI", "rd_mean", "AD", "MCI"),
    )
    rows: list[dict[str, object]] = []
    reference_network = PRIMARY_NETWORKS[0]
    for endpoint, metric, left, right in specifications:
        frame = _pair_long_frame(cohort, metrics, metric, left, right)
        formula = (
            f'value_z ~ C(group, Treatment(reference="{right}")) * '
            f'C(network, Treatment(reference="{reference_network}")) + '
            "age_z + C(sex) + gap_z + C(t1_source) + C(site)"
        )
        base = smf.ols(formula, data=frame).fit()
        fit = base.get_robustcov_results(
            cov_type="cluster",
            groups=frame["site"].to_numpy(),
            use_correction=True,
        )
        names = base.model.exog_names
        interactions = [
            index
            for index, name in enumerate(names)
            if name.startswith("C(group") and ":C(network" in name
        ]
        if len(interactions) != len(PRIMARY_NETWORKS) - 1:
            raise ValueError(
                f"Expected {len(PRIMARY_NETWORKS) - 1} interactions, got {len(interactions)}"
            )
        restriction = np.zeros((len(interactions), len(names)), dtype=float)
        for row, column in enumerate(interactions):
            restriction[row, column] = 1.0
        test = fit.f_test(restriction)
        rows.append(
            {
                "endpoint": endpoint,
                "contrast": f"{left}_vs_{right}",
                "metric": metric,
                "n_subjects": int(frame["subject_id"].nunique()),
                "n_sites": int(frame["site"].nunique()),
                "n_network_observations": int(len(frame)),
                "n_nonredundant_networks": len(PRIMARY_NETWORKS),
                "interaction_df": len(interactions),
                "f_statistic": float(np.asarray(test.fvalue).reshape(-1)[0]),
                "p_value": float(np.asarray(test.pvalue).reshape(-1)[0]),
                "covariance": "site-clustered with site fixed effects",
                "interpretation": "network-varying adjacent-stage effect",
            }
        )
    return pd.DataFrame(rows)


def _classifier(numeric: list[str], categorical: list[str]) -> Pipeline:
    numeric_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    transform = ColumnTransformer(
        [
            ("numeric", numeric_pipe, numeric),
            ("categorical", categorical_pipe, categorical),
        ]
    )
    return Pipeline(
        [
            ("transform", transform),
            (
                "model",
                LogisticRegression(
                    C=1.0,
                    class_weight="balanced",
                    max_iter=3000,
                    solver="liblinear",
                    random_state=BOOTSTRAP_SEED,
                ),
            ),
        ]
    )


def _binary_metrics(y: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    prediction = (probability >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    return {
        "roc_auc": float(roc_auc_score(y, probability)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else np.nan,
        "specificity": float(tn / (tn + fp)) if tn + fp else np.nan,
    }


def _site_loso_predictions(
    frame: pd.DataFrame,
    left: str,
    right: str,
    models: dict[str, tuple[list[str], list[str]]],
) -> pd.DataFrame:
    use = _support_frame(
        frame, "pair_common_sites", left=left, right=right
    ).copy()
    use = use[use["group"].isin([left, right])].copy()
    use["target"] = use["group"].eq(left).astype(int)
    sites = sorted(use["site"].astype(str).unique())
    rows: list[pd.DataFrame] = []
    for site in sites:
        train = use[~use["site"].astype(str).eq(site)].copy()
        test = use[use["site"].astype(str).eq(site)].copy()
        if train["target"].nunique() != 2 or test["target"].nunique() != 2:
            raise ValueError(f"Site {site} does not preserve both diagnostic groups")
        fold = pd.DataFrame(
            {
                "site": np.repeat(site, len(test)),
                "target": test["target"].to_numpy(dtype=int),
            }
        )
        for name, (numeric, categorical) in models.items():
            model = _classifier(numeric, categorical)
            columns = numeric + categorical
            model.fit(train[columns], train["target"])
            fold[name] = model.predict_proba(test[columns])[:, 1]
        rows.append(fold)
    return pd.concat(rows, ignore_index=True)


def _site_bootstrap_differences(
    predictions: pd.DataFrame,
    comparator: str,
    reference: str = "baseline",
    iterations: int = BOOTSTRAP_ITERATIONS,
) -> dict[str, float]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    sites = predictions["site"].drop_duplicates().to_numpy()
    deltas: list[float] = []
    for _ in range(iterations):
        sampled = rng.choice(sites, size=len(sites), replace=True)
        pieces = [predictions[predictions["site"].eq(site)] for site in sampled]
        boot = pd.concat(pieces, ignore_index=True)
        if boot["target"].nunique() != 2:
            continue
        baseline_auc = roc_auc_score(boot["target"], boot[reference])
        comparator_auc = roc_auc_score(boot["target"], boot[comparator])
        deltas.append(float(comparator_auc - baseline_auc))
    values = np.asarray(deltas, dtype=float)
    if len(values) < int(iterations * 0.95):
        raise RuntimeError("Too few valid site-bootstrap iterations")
    return {
        "iterations": int(len(values)),
        "delta_auc_median": float(np.median(values)),
        "delta_auc_ci95_low": float(np.quantile(values, 0.025)),
        "delta_auc_ci95_high": float(np.quantile(values, 0.975)),
        "positive_delta_fraction": float(np.mean(values > 0)),
    }


def site_held_out_classification(
    composites: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline_numeric = ["age", "log_gap_days"]
    baseline_categorical = ["sex", "t1_source"]
    specifications = (
        ("MCI_vs_CN", "MCI", "CN", "axial_burden"),
        ("AD_vs_MCI", "AD", "MCI", "radial_burden"),
    )
    summary_rows: list[dict[str, object]] = []
    delta_rows: list[dict[str, object]] = []
    for contrast, left, right, endpoint_feature in specifications:
        models = {
            "baseline": (baseline_numeric, baseline_categorical),
            "endpoint_axis": (
                baseline_numeric + [endpoint_feature],
                baseline_categorical,
            ),
            "two_axis": (
                baseline_numeric + ["axial_burden", "radial_burden"],
                baseline_categorical,
            ),
        }
        predictions = _site_loso_predictions(composites, left, right, models)
        for model in models:
            metrics = _binary_metrics(
                predictions["target"].to_numpy(dtype=int),
                predictions[model].to_numpy(dtype=float),
            )
            summary_rows.append(
                {
                    "contrast": contrast,
                    "positive_group": left,
                    "reference_group": right,
                    "model": model,
                    "n": int(len(predictions)),
                    "n_sites": int(predictions["site"].nunique()),
                    **metrics,
                    "validation": "leave-one-site-out on pair-common sites",
                    "evidence_status": "post-selection internal validation",
                }
            )
        for comparator in ("endpoint_axis", "two_axis"):
            delta_rows.append(
                {
                    "contrast": contrast,
                    "comparator": comparator,
                    "reference": "baseline",
                    **_site_bootstrap_differences(predictions, comparator),
                    "bootstrap_unit": "site",
                    "evidence_status": "descriptive post-selection uncertainty",
                }
            )
    return pd.DataFrame(summary_rows), pd.DataFrame(delta_rows)


def _make_figure(summary: pd.DataFrame, deltas: pd.DataFrame, out: Path) -> None:
    figures = out / "figures"
    figures.mkdir(parents=True, exist_ok=False)
    colors = {
        "baseline": "#9e9e9e",
        "endpoint_axis": "#31688e",
        "two_axis": "#b63679",
    }
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8), sharey=True)
    for ax, contrast in zip(axes, ("MCI_vs_CN", "AD_vs_MCI")):
        frame = summary[summary["contrast"].eq(contrast)]
        order = ["baseline", "endpoint_axis", "two_axis"]
        values = [float(frame.loc[frame["model"].eq(name), "roc_auc"].iloc[0]) for name in order]
        ax.bar(range(len(order)), values, color=[colors[name] for name in order])
        ax.axhline(0.5, color="black", linewidth=0.8, linestyle="--")
        ax.set_xticks(range(len(order)), ["Baseline", "Endpoint\naxis", "Two axes"])
        ax.set_ylim(0.45, 1.0)
        ax.set_title(contrast.replace("_", " "))
        ax.set_ylabel("Leave-one-site-out ROC AUC")
        for index, value in enumerate(values):
            ax.text(index, value + 0.015, f"{value:.3f}", ha="center", fontsize=9)
    fig.suptitle("Internal site-held-out discrimination of the two-axis phenotype")
    fig.tight_layout()
    fig.savefig(figures / "site_held_out_auc.png", dpi=260)
    plt.close(fig)


def _write_summary(
    out: Path,
    heterogeneity: pd.DataFrame,
    classification: pd.DataFrame,
    deltas: pd.DataFrame,
) -> None:
    def table(frame: pd.DataFrame, columns: list[str]) -> str:
        view = frame[columns].copy()
        for column in view.select_dtypes(include="float").columns:
            view[column] = view[column].map(lambda value: f"{value:.4g}")
        header = "| " + " | ".join(columns) + " |"
        rule = "| " + " | ".join("---" for _ in columns) + " |"
        rows = [
            "| " + " | ".join(map(str, row)) + " |"
            for row in view.itertuples(index=False, name=None)
        ]
        return "\n".join([header, rule, *rows])

    text = f"""# Two-axis extension analysis

**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}  
**Status:** aggregate-only, post-selection internal validation.

## Network heterogeneity

{table(heterogeneity, ['endpoint', 'n_subjects', 'n_sites', 'n_network_observations', 'interaction_df', 'f_statistic', 'p_value'])}

The joint interaction tests show that both adjacent-stage effects vary in magnitude across the 17 nonredundant systems. This permits the term *spatially heterogeneous*, but it does not make a nonsignificant individual system evidence of sparing.

## Site-held-out discrimination

{table(classification, ['contrast', 'model', 'n', 'n_sites', 'roc_auc', 'balanced_accuracy', 'sensitivity', 'specificity'])}

## Increment beyond the baseline

{table(deltas, ['contrast', 'comparator', 'iterations', 'delta_auc_median', 'delta_auc_ci95_low', 'delta_auc_ci95_high', 'positive_delta_fraction'])}

All preprocessing and scaling occur inside each training fold, and each test fold is a site containing both relevant diagnostic groups. These estimates remain optimistic for a novel biomarker claim because the biological axes were discovered in the same data. They are used only to test whether ML coheres with the biological result.
"""
    (out / "results_summary.md").write_text(text, encoding="utf-8")


def _manifest(out: Path) -> None:
    payload = {
        "schema_version": "1.0",
        "release": "two_axis_extensions_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_status": "POST_SELECTION_INTERNAL_VALIDATION",
        "participant_level_outputs_written": False,
        "primary_network_count": len(PRIMARY_NETWORKS),
        "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "inputs": {
            key: {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for key, path in INPUTS.items()
        },
        "packages": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "sklearn": sklearn.__version__,
            "statsmodels": statsmodels.__version__,
        },
    }
    (out / "run_manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _validate(out: Path) -> dict[str, object]:
    required = [
        "network_heterogeneity_tests.csv",
        "classification_summary.csv",
        "classification_increment_bootstrap.csv",
        "results_summary.md",
        "run_manifest.json",
        "figures/site_held_out_auc.png",
    ]
    missing = [name for name in required if not (out / name).is_file()]
    heterogeneity = pd.read_csv(out / "network_heterogeneity_tests.csv")
    classification = pd.read_csv(out / "classification_summary.csv")
    deltas = pd.read_csv(out / "classification_increment_bootstrap.csv")
    csvs = [heterogeneity, classification, deltas]
    forbidden = {"subject_id", "participant_id", "pair_id", "image_id"}
    checks: dict[str, object] = {
        "required_files_present": not missing,
        "missing_required_files": missing,
        "two_heterogeneity_tests": len(heterogeneity) == 2,
        "six_classification_rows": len(classification) == 6,
        "four_increment_rows": len(deltas) == 4,
        "bounded_probabilities": bool(
            classification[
                ["roc_auc", "balanced_accuracy", "sensitivity", "specificity"]
            ].apply(lambda column: column.between(0, 1).all()).all()
        ),
        "no_identifier_columns": all(forbidden.isdisjoint(frame.columns) for frame in csvs),
    }
    checks["pass"] = bool(
        all(value for key, value in checks.items() if key != "missing_required_files")
    )
    checks["files"] = {
        str(path.relative_to(out)): {
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in sorted(out.rglob("*"))
        if path.is_file() and path.name != "release_validation.json"
    }
    return checks


def run_release(out: Path) -> None:
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite existing release: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{out.name}.", dir=out.parent))
    try:
        cohort = load_exact_cohort()
        metrics = collect_metrics()
        composites, _ = build_composites(cohort, metrics, PRIMARY_NETWORKS)
        heterogeneity = network_heterogeneity_tests(cohort, metrics)
        classification, deltas = site_held_out_classification(composites)
        heterogeneity.to_csv(temporary / "network_heterogeneity_tests.csv", index=False)
        classification.to_csv(temporary / "classification_summary.csv", index=False)
        deltas.to_csv(
            temporary / "classification_increment_bootstrap.csv", index=False
        )
        _make_figure(classification, deltas, temporary)
        _write_summary(temporary, heterogeneity, classification, deltas)
        _manifest(temporary)
        validation = _validate(temporary)
        (temporary / "release_validation.json").write_text(
            json.dumps(validation, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if not validation["pass"]:
            raise RuntimeError(f"Release validation failed: {validation}")
        os.rename(temporary, out)
    except Exception:
        failed = temporary.with_name(temporary.name + ".failed")
        if temporary.exists() and not failed.exists():
            os.rename(temporary, failed)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_release(args.out.resolve())
    print(f"PASS: aggregate-only extension release written to {args.out.resolve()}")


if __name__ == "__main__":
    main()
