from __future__ import annotations

import argparse
import importlib.util
import math
import re
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning
from scipy import stats
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    explained_variance_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    r2_score,
    roc_auc_score,
)
from sklearn.inspection import permutation_importance
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVR

try:
    from xgboost import XGBClassifier, XGBRegressor

    HAS_XGBOOST = True
except Exception:  # pragma: no cover - optional local dependency
    XGBClassifier = None
    XGBRegressor = None
    HAS_XGBOOST = False


GROUP_ORDER = ("CN", "MCI", "AD")
SCORE_TARGETS = ("MMSE Total Score",)
CDR_CLASSIFICATION_TARGET = "Global CDR"
CLINICAL_SCORE_COLUMNS = (*SCORE_TARGETS, CDR_CLASSIFICATION_TARGET, "NPI-Q Total Score")
PRIMARY_MODE = "primary_scan_aligned_730d"
HISTORY_MODE = "sensitivity_all_available_history"
RANDOM_STATE = 42
STRUCTURAL_SCORE_FEATURE_POLICY = "structural_connectome_only"
MIN_SCORE_SUBJECTS = 60
MIN_SCORE_GROUPS = 2

warnings.filterwarnings("ignore", category=PerformanceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn.feature_selection")


@dataclass(frozen=True)
class FeatureSet:
    name: str
    columns: tuple[str, ...]
    policy: str


def _sanitize(value: object) -> str:
    text = str(value).strip()
    text = re.sub(r"[^0-9A-Za-z]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text.lower() or "value"


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def _diagnostics_root(exp_root: Path) -> Path:
    return exp_root / "data" / "derivatives" / "qc" / "analysis_cohort" / "18_ml_diagnostics"


def _cohort_root(exp_root: Path) -> Path:
    return exp_root / "data" / "derivatives" / "qc" / "analysis_cohort" / "00_master"


def _numeric_frame(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return df.loc[:, columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).copy()


def _drop_all_empty_features(x: pd.DataFrame, feature_names: list[str]) -> tuple[pd.DataFrame, list[str]]:
    kept = [c for c in feature_names if c in x.columns and x[c].notna().any()]
    return x.loc[:, kept].copy(), kept


def _metadata_features(master: pd.DataFrame) -> pd.DataFrame:
    keep = ["subject_id"]
    work = master.drop_duplicates("subject_id").copy()
    out = work[keep].copy()
    numeric_cols = [
        "age",
        "Age",
        "weight",
        "Weight",
        "apoe_e4_count",
        "apoe_e4_carrier",
        "APOE A1",
        "APOE A2",
        "n_connectome_types",
        "n_sc_qc_series",
        "n_sc_qc_failed_series",
    ]
    for col in numeric_cols:
        if col in work.columns:
            out[f"metadata__{_sanitize(col)}"] = pd.to_numeric(work[col], errors="coerce")
    categorical_cols = ["sex", "Sex", "phase", "Phase", "sc_matrix_qc_status", "sc_matrix_qc_gate"]
    for col in categorical_cols:
        if col in work.columns:
            dummies = pd.get_dummies(work[col].astype(str).replace({"nan": np.nan}), prefix=f"metadata__{_sanitize(col)}", dummy_na=True)
            out = pd.concat([out, dummies.astype(float)], axis=1)
    out = out.loc[:, ~out.columns.duplicated()].copy()
    return out


def _master_clinical_features(master: pd.DataFrame) -> pd.DataFrame:
    work = master.drop_duplicates("subject_id").copy()
    out = work[["subject_id"]].copy()
    for col in CLINICAL_SCORE_COLUMNS:
        if col in work.columns:
            out[f"clinical_master__{_sanitize(col)}"] = pd.to_numeric(work[col], errors="coerce")
    return out


def _matched_clinical_features(matches: pd.DataFrame, mode: str) -> pd.DataFrame:
    cols = ["subject_id"]
    if matches.empty:
        return pd.DataFrame(columns=cols)
    sub = matches[matches["target_mode"].astype(str).eq(mode)].copy()
    if sub.empty:
        return pd.DataFrame(columns=cols)
    sub["target_value"] = pd.to_numeric(sub["target_value"], errors="coerce")
    values = sub.pivot_table(index="subject_id", columns="target", values="target_value", aggfunc="first")
    values.columns = [f"clinical_{_sanitize(mode)}__{_sanitize(c)}" for c in values.columns]
    deltas = sub.pivot_table(index="subject_id", columns="target", values="abs_day_delta", aggfunc="first")
    deltas.columns = [f"clinical_{_sanitize(mode)}__{_sanitize(c)}__abs_day_delta" for c in deltas.columns]
    out = pd.concat([values, deltas], axis=1).reset_index()
    return out


def _clinical_feature_name_tokens(target: str) -> tuple[str, ...]:
    clean = _sanitize(target)
    return (
        f"__{clean}",
        f"__{clean}__",
        f"__{clean}__abs_day_delta",
    )


def _drop_target_leakage(columns: list[str], target: str) -> list[str]:
    tokens = _clinical_feature_name_tokens(target)
    kept: list[str] = []
    for col in columns:
        if any(token in col for token in tokens) and col.startswith("clinical_"):
            continue
        kept.append(col)
    return kept


def _score_feature_sets(feature_sets: list[FeatureSet]) -> list[FeatureSet]:
    return [fs for fs in feature_sets if fs.policy == STRUCTURAL_SCORE_FEATURE_POLICY]


def _score_outlier_filter(task: pd.DataFrame, target: str, target_mode: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    work = task.copy()
    work["_score_outlier"] = False
    if "group" not in work.columns or "target_value" not in work.columns:
        return work.drop(columns=["_score_outlier"], errors="ignore"), pd.DataFrame()
    for group, sub in work.groupby("group", dropna=False):
        values = pd.to_numeric(sub["target_value"], errors="coerce").dropna()
        if values.empty:
            q1 = q3 = iqr = lower = upper = np.nan
        else:
            q1 = float(values.quantile(0.25))
            q3 = float(values.quantile(0.75))
            iqr = float(q3 - q1)
            lower = float(q1 - 1.5 * iqr)
            upper = float(q3 + 1.5 * iqr)
        for idx in sub.index:
            value = pd.to_numeric(pd.Series([work.loc[idx, "target_value"]]), errors="coerce").iloc[0]
            removed = bool(pd.notna(value) and pd.notna(lower) and pd.notna(upper) and (value < lower or value > upper))
            work.loc[idx, "_score_outlier"] = removed
            rows.append(
                {
                    "target": target,
                    "target_mode": target_mode,
                    "subject_id": work.loc[idx, "subject_id"],
                    "group": group,
                    "target_value": float(value) if pd.notna(value) else np.nan,
                    "q1": q1,
                    "q3": q3,
                    "iqr": iqr,
                    "lower_bound": lower,
                    "upper_bound": upper,
                    "removed_as_outlier": removed,
                }
            )
    clean = work[~work["_score_outlier"].astype(bool)].drop(columns=["_score_outlier"], errors="ignore").reset_index(drop=True)
    return clean, pd.DataFrame(rows)


def _build_model_table(root: Path, exp_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    matrix = _read_csv(root / "ml_subject_feature_matrix.csv")
    catalog = _read_csv(root / "ml_feature_catalog.csv")
    matches = _read_csv(root / "score_prediction_target_matches.csv")
    master = _read_csv(_cohort_root(exp_root) / "master_cohort.csv")
    if matrix.empty or catalog.empty:
        raise FileNotFoundError(f"Missing ML diagnostics inputs in {root}")

    table = matrix.copy()
    for extra in (
        _metadata_features(master),
        _master_clinical_features(master),
        _matched_clinical_features(matches, PRIMARY_MODE),
        _matched_clinical_features(matches, HISTORY_MODE),
    ):
        if not extra.empty and "subject_id" in extra.columns:
            table = table.merge(extra, on="subject_id", how="left")
    table = table.loc[:, ~table.columns.duplicated()].copy()
    return table, catalog, matches


def _feature_sets(
    table: pd.DataFrame,
    catalog: pd.DataFrame,
    max_sets: int | None = None,
    only: set[str] | None = None,
) -> list[FeatureSet]:
    structural_all = [
        c
        for c in catalog.loc[~catalog["final_status"].astype(str).str.startswith("dropped"), "feature"].astype(str)
        if c in table.columns
    ]
    structural_selected = [
        c
        for c in catalog.loc[catalog["final_status"].astype(str).eq("selected_in_cv"), "feature"].astype(str)
        if c in table.columns
    ]
    compact_groups = {
        "Global DTI",
        "Global graph",
        "Brain age",
        "LR/SR and delay",
        "Delay",
        "Advanced structural",
        "EDR exceptions",
        "Raw matrix summaries",
        "Coupling",
    }
    structural_compact = [
        row.feature
        for row in catalog[["feature", "feature_group", "final_status"]].itertuples(index=False)
        if str(row.feature_group) in compact_groups
        and not str(row.final_status).startswith("dropped")
        and str(row.feature) in table.columns
    ]
    metadata_cols = [c for c in table.columns if c.startswith("metadata__")]
    metadata_no_age = [c for c in metadata_cols if "age" not in c]
    clinical_master = [c for c in table.columns if c.startswith("clinical_master__")]
    clinical_primary = [c for c in table.columns if c.startswith(f"clinical_{_sanitize(PRIMARY_MODE)}__")]
    clinical_history = [c for c in table.columns if c.startswith(f"clinical_{_sanitize(HISTORY_MODE)}__")]

    candidates = [
        FeatureSet("structural_prefiltered", tuple(structural_all), "structural_connectome_only"),
        FeatureSet("structural_selected_cv", tuple(structural_selected), "structural_connectome_only"),
        FeatureSet("structural_compact", tuple(structural_compact), "structural_connectome_only"),
        FeatureSet("metadata_no_age", tuple(metadata_no_age), "metadata_no_age"),
        FeatureSet("metadata_with_age", tuple(metadata_cols), "metadata_with_age"),
        FeatureSet("clinical_master", tuple(clinical_master), "clinical_assisted"),
        FeatureSet("clinical_scan_aligned_730d", tuple(clinical_primary), "clinical_assisted"),
        FeatureSet("clinical_all_history", tuple(clinical_history), "clinical_assisted"),
        FeatureSet("structural_plus_metadata_no_age", tuple([*structural_all, *metadata_no_age]), "structural_plus_metadata_no_age"),
        FeatureSet("structural_plus_metadata_with_age", tuple([*structural_all, *metadata_cols]), "structural_plus_metadata_with_age"),
        FeatureSet("all_scan_aligned_no_age", tuple([*structural_all, *metadata_no_age, *clinical_master, *clinical_primary]), "clinical_assisted_no_age"),
        FeatureSet("all_scan_aligned_with_age", tuple([*structural_all, *metadata_cols, *clinical_master, *clinical_primary]), "clinical_assisted_with_age"),
        FeatureSet("all_history_no_age", tuple([*structural_all, *metadata_no_age, *clinical_master, *clinical_history]), "clinical_assisted_no_age"),
        FeatureSet("all_history_with_age", tuple([*structural_all, *metadata_cols, *clinical_master, *clinical_history]), "clinical_assisted_with_age"),
    ]
    cleaned: list[FeatureSet] = []
    for fs in candidates:
        if only is not None and fs.name not in only:
            continue
        cols = tuple(dict.fromkeys(c for c in fs.columns if c in table.columns and c not in {"subject_id", "group"}))
        if cols:
            cleaned.append(FeatureSet(fs.name, cols, fs.policy))
    return cleaned if max_sets is None else cleaned[:max_sets]


def _select_k(n_features: int, n_subjects: int, task: str) -> int | str:
    if n_features <= 2:
        return "all"
    cap = 180 if task == "classification" else 80
    if n_features <= cap:
        return "all"
    upper = max(2, min(cap, n_subjects - 2))
    return int(upper)


def _class_model_specs(n_features: int, n_subjects: int, n_classes: int, fast: bool = False) -> dict[str, Pipeline]:
    k = _select_k(n_features, n_subjects, "classification")
    specs: dict[str, Pipeline] = {
        "logistic_l2_balanced": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("select", SelectKBest(score_func=f_classif, k=k)),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=0.8,
                        class_weight="balanced",
                        max_iter=1500,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "extra_trees_500_leaf1": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("select", SelectKBest(score_func=mutual_info_classif, k=k)),
                (
                    "model",
                    ExtraTreesClassifier(
                        n_estimators=500,
                        min_samples_leaf=1,
                        class_weight="balanced",
                        max_features="sqrt",
                        n_jobs=4,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "random_forest_500_leaf1": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("select", SelectKBest(score_func=mutual_info_classif, k=k)),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=500,
                        min_samples_leaf=1,
                        class_weight="balanced_subsample",
                        max_features="sqrt",
                        n_jobs=4,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "hist_gradient_boosting": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        learning_rate=0.04,
                        max_leaf_nodes=15,
                        l2_regularization=0.05,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "mlp_deep_classifier": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("select", SelectKBest(score_func=f_classif, k=k)),
                ("scaler", StandardScaler()),
                (
                    "model",
                    MLPClassifier(
                        hidden_layer_sizes=(192, 64, 16),
                        activation="relu",
                        alpha=0.02,
                        batch_size=32,
                        early_stopping=True,
                        max_iter=260,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }
    if HAS_XGBOOST and XGBClassifier is not None:
        specs["xgboost_depth2"] = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("select", SelectKBest(score_func=mutual_info_classif, k=k)),
                (
                    "model",
                    XGBClassifier(
                        objective="multi:softprob",
                        eval_metric="mlogloss",
                        num_class=n_classes,
                        n_estimators=220,
                        max_depth=2,
                        learning_rate=0.035,
                        subsample=0.90,
                        colsample_bytree=0.90,
                        min_child_weight=1,
                        reg_lambda=1.5,
                        tree_method="hist",
                        n_jobs=4,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )
        specs["xgboost_depth4"] = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("select", SelectKBest(score_func=mutual_info_classif, k=k)),
                (
                    "model",
                    XGBClassifier(
                        objective="multi:softprob",
                        eval_metric="mlogloss",
                        num_class=n_classes,
                        n_estimators=160,
                        max_depth=4,
                        learning_rate=0.04,
                        subsample=0.85,
                        colsample_bytree=0.75,
                        min_child_weight=2,
                        reg_lambda=2.0,
                        tree_method="hist",
                        n_jobs=4,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )
    if fast:
        keep = {"logistic_l2_balanced", "extra_trees_500_leaf1", "hist_gradient_boosting", "xgboost_depth2"}
        specs = {name: pipe for name, pipe in specs.items() if name in keep}
    return specs


def _reg_model_specs(n_features: int, n_subjects: int, fast: bool = False) -> dict[str, Pipeline]:
    k = _select_k(n_features, n_subjects, "regression")
    pca_components = max(2, min(32, n_subjects - 4, n_features))
    specs: dict[str, Pipeline] = {
        "ridge_select": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("select", SelectKBest(score_func=mutual_info_regression, k=k)),
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=20.0)),
            ]
        ),
        "elastic_net_select": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("select", SelectKBest(score_func=mutual_info_regression, k=k)),
                ("scaler", StandardScaler()),
                ("model", ElasticNet(alpha=0.08, l1_ratio=0.35, max_iter=10000, random_state=RANDOM_STATE)),
            ]
        ),
        "pca_ridge": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=pca_components, random_state=RANDOM_STATE)),
                ("model", Ridge(alpha=10.0)),
            ]
        ),
        "extra_trees_500_leaf1": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("select", SelectKBest(score_func=mutual_info_regression, k=k)),
                (
                    "model",
                    ExtraTreesRegressor(
                        n_estimators=500,
                        min_samples_leaf=1,
                        max_features="sqrt",
                        n_jobs=4,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "random_forest_500_leaf1": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("select", SelectKBest(score_func=mutual_info_regression, k=k)),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=500,
                        min_samples_leaf=1,
                        max_features="sqrt",
                        n_jobs=4,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "hist_gradient_boosting": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                (
                    "model",
                    HistGradientBoostingRegressor(
                        learning_rate=0.035,
                        max_leaf_nodes=15,
                        l2_regularization=0.05,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "svr_rbf": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("select", SelectKBest(score_func=mutual_info_regression, k=k)),
                ("scaler", StandardScaler()),
                ("model", SVR(C=8.0, epsilon=0.15, gamma="scale")),
            ]
        ),
        "mlp_deep_regression": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("select", SelectKBest(score_func=mutual_info_regression, k=k)),
                ("scaler", StandardScaler()),
                (
                    "model",
                    MLPRegressor(
                        hidden_layer_sizes=(192, 64, 16),
                        activation="relu",
                        alpha=0.05,
                        batch_size=16,
                        early_stopping=True,
                        max_iter=320,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }
    if HAS_XGBOOST and XGBRegressor is not None:
        specs["xgboost_depth2"] = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("select", SelectKBest(score_func=mutual_info_regression, k=k)),
                (
                    "model",
                    XGBRegressor(
                        objective="reg:squarederror",
                        eval_metric="rmse",
                        n_estimators=240,
                        max_depth=2,
                        learning_rate=0.035,
                        subsample=0.90,
                        colsample_bytree=0.90,
                        min_child_weight=1,
                        reg_lambda=1.5,
                        tree_method="hist",
                        n_jobs=4,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )
        specs["xgboost_depth4"] = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("select", SelectKBest(score_func=mutual_info_regression, k=k)),
                (
                    "model",
                    XGBRegressor(
                        objective="reg:squarederror",
                        eval_metric="rmse",
                        n_estimators=180,
                        max_depth=4,
                        learning_rate=0.04,
                        subsample=0.85,
                        colsample_bytree=0.75,
                        min_child_weight=2,
                        reg_lambda=2.0,
                        tree_method="hist",
                        n_jobs=4,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )
    if fast:
        keep = {
            "ridge_select",
            "elastic_net_select",
            "pca_ridge",
            "extra_trees_500_leaf1",
            "hist_gradient_boosting",
            "xgboost_depth2",
        }
        specs = {name: pipe for name, pipe in specs.items() if name in keep}
    return specs


def _selected_names(pipe: Pipeline, feature_names: list[str]) -> list[str]:
    selector = pipe.named_steps.get("select")
    if selector is None or selector == "passthrough" or not hasattr(selector, "get_support"):
        return feature_names
    support = selector.get_support()
    n = min(len(feature_names), len(support))
    return [feature_names[i] for i in range(n) if support[i]]


def _importance(pipe: Pipeline, feature_names: list[str]) -> dict[str, float]:
    model = pipe.named_steps.get("model")
    if model is None:
        return {}
    selected = _selected_names(pipe, feature_names)
    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        values = np.asarray(model.coef_, dtype=float)
        if values.ndim > 1:
            values = np.nanmean(np.abs(values), axis=0)
        else:
            values = np.abs(values)
    else:
        return {}
    if len(values) != len(selected):
        return {}
    return {feature: float(value) for feature, value in zip(selected, values)}


def _permutation_importance(
    model: Pipeline,
    x_test: pd.DataFrame,
    y_test: np.ndarray,
    feature_names: list[str],
    scoring: str,
) -> dict[str, float]:
    if len(feature_names) > 60:
        return {}
    try:
        result = permutation_importance(
            model,
            x_test,
            y_test,
            scoring=scoring,
            n_repeats=3,
            random_state=RANDOM_STATE,
            n_jobs=1,
        )
    except Exception:
        return {}
    values = np.asarray(result.importances_mean, dtype=float)
    return {feature: float(max(value, 0.0)) for feature, value in zip(feature_names, values)}


def _safe_probs(model: Pipeline, x: pd.DataFrame, n_classes: int) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        try:
            probs = np.asarray(model.predict_proba(x), dtype=float)
            row_sum = np.nansum(probs, axis=1, keepdims=True)
            valid = np.isfinite(row_sum) & (row_sum > 0)
            probs = np.where(valid, probs / row_sum, probs)
            return probs
        except Exception:
            pass
    return np.full((len(x), n_classes), np.nan)


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    out = {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else np.nan,
        "explained_variance": float(explained_variance_score(y_true, y_pred)) if len(y_true) >= 2 else np.nan,
        "bias_mean_error": float(np.nanmean(y_pred - y_true)),
        "spearman_r": np.nan,
        "pearson_r": np.nan,
    }
    if len(y_true) >= 3 and len(np.unique(y_true)) > 1 and len(np.unique(y_pred)) > 1:
        out["spearman_r"] = float(stats.spearmanr(y_true, y_pred).statistic)
        out["pearson_r"] = float(stats.pearsonr(y_true, y_pred).statistic)
    return out


def _evaluate_classification(
    table: pd.DataFrame,
    feature_sets: list[FeatureSet],
    *,
    fast: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    task = table[table["group"].isin(GROUP_ORDER)].copy().reset_index(drop=True)
    encoder = LabelEncoder().fit(list(GROUP_ORDER))
    y = encoder.transform(task["group"].astype(str))
    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    perf_rows: list[dict] = []
    pred_rows: list[dict] = []
    class_rows: list[dict] = []
    importance_rows: list[dict] = []

    for fs in feature_sets:
        feature_names = list(fs.columns)
        x = _numeric_frame(task, feature_names)
        x, feature_names = _drop_all_empty_features(x, feature_names)
        if len(feature_names) < 2:
            perf_rows.append(
                {
                    "task": "disease_prediction",
                    "target": "CN_MCI_AD",
                    "feature_set": fs.name,
                    "feature_policy": fs.policy,
                    "model": "not_run",
                    "status": "skipped_no_observed_features",
                    "n_subjects": int(len(task)),
                    "n_features": int(len(feature_names)),
                    "accuracy": np.nan,
                    "balanced_accuracy": np.nan,
                    "macro_f1": np.nan,
                    "weighted_f1": np.nan,
                    "macro_auroc_ovr": np.nan,
                    "log_loss": np.nan,
                }
            )
            continue
        model_specs = _class_model_specs(len(feature_names), len(task), len(encoder.classes_), fast=fast)
        for model_name, pipe in model_specs.items():
            fold_importance: Counter = Counter()
            try:
                for fold, (train_idx, test_idx) in enumerate(outer.split(x, y), start=1):
                    model = clone(pipe)
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", category=ConvergenceWarning)
                        warnings.filterwarnings("ignore", category=UserWarning)
                        warnings.filterwarnings("ignore", category=FutureWarning)
                        warnings.filterwarnings("ignore", category=RuntimeWarning)
                        model.fit(x.iloc[train_idx], y[train_idx])
                    pred = model.predict(x.iloc[test_idx])
                    probs = _safe_probs(model, x.iloc[test_idx], len(encoder.classes_))
                    importances = _importance(model, feature_names)
                    if not importances:
                        importances = _permutation_importance(model, x.iloc[test_idx], y[test_idx], feature_names, "balanced_accuracy")
                    fold_importance.update(importances)
                    for row_pos, idx in enumerate(test_idx):
                        row = {
                            "task": "disease_prediction",
                            "feature_set": fs.name,
                            "feature_policy": fs.policy,
                            "model": model_name,
                            "fold": fold,
                            "subject_id": task.iloc[idx]["subject_id"],
                            "true_group": encoder.inverse_transform([y[idx]])[0],
                            "predicted_group": encoder.inverse_transform([int(pred[row_pos])])[0],
                        }
                        for class_idx, class_name in enumerate(encoder.classes_):
                            row[f"prob_{class_name}"] = float(probs[row_pos, class_idx])
                        pred_rows.append(row)
            except Exception as exc:
                perf_rows.append(
                    {
                        "task": "disease_prediction",
                        "target": "CN_MCI_AD",
                        "feature_set": fs.name,
                        "feature_policy": fs.policy,
                        "model": model_name,
                        "status": f"failed: {exc}",
                        "n_subjects": int(len(task)),
                        "n_features": int(len(feature_names)),
                        "accuracy": np.nan,
                        "balanced_accuracy": np.nan,
                        "macro_f1": np.nan,
                        "weighted_f1": np.nan,
                        "macro_auroc_ovr": np.nan,
                        "log_loss": np.nan,
                    }
                )
                continue
            pred_df = pd.DataFrame([r for r in pred_rows if r["feature_set"] == fs.name and r["model"] == model_name])
            y_true = encoder.transform(pred_df["true_group"].astype(str))
            y_pred = encoder.transform(pred_df["predicted_group"].astype(str))
            prob_cols = [f"prob_{g}" for g in encoder.classes_]
            probs_all = pred_df[prob_cols].to_numpy(dtype=float)
            macro_auroc = np.nan
            model_log_loss = np.nan
            if np.isfinite(probs_all).all():
                try:
                    macro_auroc = float(roc_auc_score(y_true, probs_all, multi_class="ovr", average="macro"))
                    model_log_loss = float(log_loss(y_true, probs_all, labels=list(range(len(encoder.classes_)))))
                except Exception:
                    pass
            perf_rows.append(
                {
                    "task": "disease_prediction",
                    "target": "CN_MCI_AD",
                    "feature_set": fs.name,
                    "feature_policy": fs.policy,
                    "model": model_name,
                    "status": "ok",
                    "n_subjects": int(len(task)),
                    "n_features": int(len(feature_names)),
                    "accuracy": float(accuracy_score(y_true, y_pred)),
                    "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
                    "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
                    "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
                    "macro_auroc_ovr": macro_auroc,
                    "log_loss": model_log_loss,
                }
            )
            precision, recall, f1_vals, support = precision_recall_fscore_support(
                y_true,
                y_pred,
                labels=list(range(len(encoder.classes_))),
                zero_division=0,
            )
            for idx, class_name in enumerate(encoder.classes_):
                class_rows.append(
                    {
                        "task": "disease_prediction",
                        "feature_set": fs.name,
                        "feature_policy": fs.policy,
                        "model": model_name,
                        "class": class_name,
                        "support": int(support[idx]),
                        "precision": float(precision[idx]),
                        "recall": float(recall[idx]),
                        "f1": float(f1_vals[idx]),
                    }
                )
            for rank, (feature, value) in enumerate(fold_importance.most_common(80), start=1):
                importance_rows.append(
                    {
                        "task": "disease_prediction",
                        "target": "CN_MCI_AD",
                        "target_mode": "",
                        "feature_set": fs.name,
                        "feature_policy": fs.policy,
                        "model": model_name,
                        "rank": rank,
                        "feature": feature,
                        "importance": float(value / outer.n_splits),
                    }
                )
    return pd.DataFrame(perf_rows), pd.DataFrame(pred_rows), pd.DataFrame(class_rows), pd.DataFrame(importance_rows)


def _score_tasks(
    matches: pd.DataFrame,
    table: pd.DataFrame,
    *,
    targets: set[str] | None = None,
    modes: set[str] | None = None,
) -> tuple[list[pd.DataFrame], pd.DataFrame]:
    tasks: list[pd.DataFrame] = []
    outlier_frames: list[pd.DataFrame] = []
    for target in SCORE_TARGETS:
        if targets is not None and target not in targets:
            continue
        for mode, role in ((PRIMARY_MODE, "primary"), (HISTORY_MODE, "sensitivity")):
            if modes is not None and mode not in modes:
                continue
            sub = matches[matches["target"].astype(str).eq(target) & matches["target_mode"].astype(str).eq(mode)].copy()
            if sub.empty:
                continue
            sub["target_value"] = pd.to_numeric(sub["target_value"], errors="coerce")
            sub = sub.dropna(subset=["target_value"])
            task = sub.merge(table.drop(columns=["group"], errors="ignore"), on="subject_id", how="inner")
            task["target"] = target
            task["target_mode"] = mode
            task["analysis_role"] = role
            raw_n = int(len(task))
            task, outlier_audit = _score_outlier_filter(task, target, mode)
            if not outlier_audit.empty:
                outlier_audit["analysis_role"] = role
                outlier_frames.append(outlier_audit)
            task["n_subjects_before_outlier_filter"] = raw_n
            task["n_outliers_removed"] = int(raw_n - len(task))
            group_count = int(task["group"].astype(str).nunique()) if "group" in task.columns else 0
            if len(task) >= MIN_SCORE_SUBJECTS and group_count >= MIN_SCORE_GROUPS:
                tasks.append(task.reset_index(drop=True))
    audit = pd.concat(outlier_frames, ignore_index=True) if outlier_frames else pd.DataFrame()
    return tasks, audit


def _evaluate_regression(
    table: pd.DataFrame,
    matches: pd.DataFrame,
    feature_sets: list[FeatureSet],
    *,
    fast: bool = False,
    targets: set[str] | None = None,
    modes: set[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tasks, outlier_audit = _score_tasks(matches, table, targets=targets, modes=modes)
    feature_sets = _score_feature_sets(feature_sets)
    perf_rows: list[dict] = []
    pred_rows: list[dict] = []
    importance_rows: list[dict] = []

    for task in tasks:
        target = str(task["target"].iloc[0])
        target_mode = str(task["target_mode"].iloc[0])
        role = str(task["analysis_role"].iloc[0])
        n_before_outliers = int(task.get("n_subjects_before_outlier_filter", pd.Series([len(task)])).iloc[0])
        n_outliers_removed = int(task.get("n_outliers_removed", pd.Series([0])).iloc[0])
        y = task["target_value"].to_numpy(dtype=float)
        splits = 5 if len(task) >= 80 else 3
        outer = KFold(n_splits=splits, shuffle=True, random_state=RANDOM_STATE)
        for fs in feature_sets:
            feature_names = _drop_target_leakage(list(fs.columns), target)
            feature_names = [c for c in feature_names if c in task.columns]
            if len(feature_names) < 2:
                continue
            x = _numeric_frame(task, feature_names)
            x, feature_names = _drop_all_empty_features(x, feature_names)
            if len(feature_names) < 2:
                continue
            model_specs = _reg_model_specs(len(feature_names), len(task), fast=fast)
            for model_name, pipe in model_specs.items():
                fold_importance: Counter = Counter()
                try:
                    for fold, (train_idx, test_idx) in enumerate(outer.split(x), start=1):
                        model = clone(pipe)
                        with warnings.catch_warnings():
                            warnings.filterwarnings("ignore", category=ConvergenceWarning)
                            warnings.filterwarnings("ignore", category=UserWarning)
                            warnings.filterwarnings("ignore", category=FutureWarning)
                            warnings.filterwarnings("ignore", category=RuntimeWarning)
                            model.fit(x.iloc[train_idx], y[train_idx])
                        pred = np.ravel(model.predict(x.iloc[test_idx]))
                        importances = _importance(model, feature_names)
                        if not importances:
                            importances = _permutation_importance(model, x.iloc[test_idx], y[test_idx], feature_names, "r2")
                        fold_importance.update(importances)
                        for row_pos, idx in enumerate(test_idx):
                            pred_rows.append(
                                {
                                    "task": "score_prediction",
                                    "target": target,
                                    "target_mode": target_mode,
                                    "analysis_role": role,
                                    "feature_set": fs.name,
                                    "feature_policy": fs.policy,
                                    "model": model_name,
                                    "fold": fold,
                                    "subject_id": task.iloc[idx]["subject_id"],
                                    "group": task.iloc[idx].get("group", ""),
                                    "true_score": float(y[idx]),
                                    "predicted_score": float(pred[row_pos]),
                                    "residual": float(y[idx] - pred[row_pos]),
                                }
                            )
                except Exception as exc:
                    perf_rows.append(
                        {
                            "task": "score_prediction",
                            "target": target,
                            "target_mode": target_mode,
                            "analysis_role": role,
                            "feature_set": fs.name,
                            "feature_policy": fs.policy,
                            "model": model_name,
                            "status": f"failed: {exc}",
                            "n_subjects": int(len(task)),
                            "n_subjects_before_outlier_filter": n_before_outliers,
                            "n_outliers_removed": n_outliers_removed,
                            "n_features": int(len(feature_names)),
                            "n_folds": int(splits),
                            "mae": np.nan,
                            "rmse": np.nan,
                            "r2": np.nan,
                            "explained_variance": np.nan,
                            "bias_mean_error": np.nan,
                            "spearman_r": np.nan,
                            "pearson_r": np.nan,
                        }
                    )
                    continue
                pred_df = pd.DataFrame(
                    [
                        r
                        for r in pred_rows
                        if r["target"] == target
                        and r["target_mode"] == target_mode
                        and r["feature_set"] == fs.name
                        and r["model"] == model_name
                    ]
                )
                metrics = _regression_metrics(pred_df["true_score"].to_numpy(dtype=float), pred_df["predicted_score"].to_numpy(dtype=float))
                perf_rows.append(
                    {
                        "task": "score_prediction",
                        "target": target,
                        "target_mode": target_mode,
                        "analysis_role": role,
                        "feature_set": fs.name,
                        "feature_policy": fs.policy,
                        "model": model_name,
                        "status": "ok",
                        "n_subjects": int(len(task)),
                        "n_subjects_before_outlier_filter": n_before_outliers,
                        "n_outliers_removed": n_outliers_removed,
                        "n_features": int(len(feature_names)),
                        "n_folds": int(splits),
                        **metrics,
                    }
                )
                for rank, (feature, value) in enumerate(fold_importance.most_common(80), start=1):
                    importance_rows.append(
                        {
                            "task": "score_prediction",
                            "target": target,
                            "target_mode": target_mode,
                            "feature_set": fs.name,
                            "feature_policy": fs.policy,
                            "model": model_name,
                            "rank": rank,
                            "feature": feature,
                            "importance": float(value / splits),
                        }
                    )
    return pd.DataFrame(perf_rows), pd.DataFrame(pred_rows), pd.DataFrame(importance_rows), outlier_audit


def _dependency_status() -> pd.DataFrame:
    modules = {
        "xgboost": HAS_XGBOOST,
        "torch": importlib.util.find_spec("torch") is not None,
        "tabpfn": importlib.util.find_spec("tabpfn") is not None,
        "pytorch_tabnet": importlib.util.find_spec("pytorch_tabnet") is not None,
        "rtdl": importlib.util.find_spec("rtdl") is not None,
        "tabm": importlib.util.find_spec("tabm") is not None,
        "shap": importlib.util.find_spec("shap") is not None,
    }
    return pd.DataFrame(
        [
            {
                "dependency": name,
                "available": bool(available),
                "status": "available" if available else "missing",
            }
            for name, available in modules.items()
        ]
    )


def _write_summary(out_dir: Path, disease_perf: pd.DataFrame, score_perf: pd.DataFrame) -> None:
    disease_best = pd.DataFrame()
    if not disease_perf.empty:
        disease_best = disease_perf.sort_values(["accuracy", "macro_f1"], ascending=False).head(10)
    score_best = pd.DataFrame()
    if not score_perf.empty:
        score_best = score_perf.sort_values(["r2", "spearman_r"], ascending=False).head(15)
    lines = [
        "# Enhanced ML Goal Search",
        "",
        "This file is experimental output for the explicit R2/accuracy goal. Score-prediction rows are restricted to structural-connectome feature sets and exclude age, demographics, diagnosis labels, APOE, phase, and clinical scores as predictors.",
        "",
        "## Best Disease Models",
        "",
        disease_best.to_string(index=False) if not disease_best.empty else "No disease models ran.",
        "",
        "## Best Score Models",
        "",
        score_best.to_string(index=False) if not score_best.empty else "No score models ran.",
        "",
    ]
    (out_dir / "enhanced_goal_search_summary.md").write_text("\n".join(lines), encoding="utf-8")


def run_goal_search(
    exp_root: Path,
    max_feature_sets: int | None = None,
    only_feature_sets: set[str] | None = None,
    out_dir: Path | None = None,
    task: str = "both",
    fast: bool = False,
    score_targets: set[str] | None = None,
    score_modes: set[str] | None = None,
) -> Path:
    root = _diagnostics_root(exp_root)
    out_dir = out_dir or (exp_root / "outputs" / "enhanced_goal_search")
    out_dir.mkdir(parents=True, exist_ok=True)
    table, catalog, matches = _build_model_table(root, exp_root)
    feature_sets = _feature_sets(table, catalog, max_sets=max_feature_sets, only=only_feature_sets)
    feature_set_rows = [
        {
            "feature_set": fs.name,
            "feature_policy": fs.policy,
            "n_features": len(fs.columns),
            "example_features": "; ".join(fs.columns[:12]),
        }
        for fs in feature_sets
    ]
    pd.DataFrame(feature_set_rows).to_csv(out_dir / "enhanced_feature_sets.csv", index=False)
    _dependency_status().to_csv(out_dir / "enhanced_dependency_status.csv", index=False)

    disease_perf = disease_pred = class_metrics = disease_importance = pd.DataFrame()
    if task in {"both", "disease"}:
        disease_perf, disease_pred, class_metrics, disease_importance = _evaluate_classification(table, feature_sets, fast=fast)
        disease_perf.to_csv(out_dir / "enhanced_disease_model_performance.csv", index=False)
        disease_pred.to_csv(out_dir / "enhanced_disease_cv_predictions.csv", index=False)
        class_metrics.to_csv(out_dir / "enhanced_disease_class_metrics.csv", index=False)
        disease_importance.to_csv(out_dir / "enhanced_disease_feature_importance.csv", index=False)

    score_perf = score_pred = score_importance = score_outliers = pd.DataFrame()
    if task in {"both", "score"}:
        score_perf, score_pred, score_importance, score_outliers = _evaluate_regression(
            table,
            matches,
            feature_sets,
            fast=fast,
            targets=score_targets,
            modes=score_modes,
        )
        score_perf.to_csv(out_dir / "enhanced_score_model_performance.csv", index=False)
        score_pred.to_csv(out_dir / "enhanced_score_cv_predictions.csv", index=False)
        score_importance.to_csv(out_dir / "enhanced_score_feature_importance.csv", index=False)
        score_outliers.to_csv(out_dir / "enhanced_score_outlier_audit.csv", index=False)
    importance = pd.concat([disease_importance, score_importance], ignore_index=True)
    importance.to_csv(out_dir / "enhanced_feature_importance.csv", index=False)
    _write_summary(out_dir, disease_perf, score_perf)
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run enhanced ML feature/model goal search for connectome diagnostics.")
    parser.add_argument("--exp-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Writable output directory. Defaults to <repo>/outputs/enhanced_goal_search.",
    )
    parser.add_argument("--max-feature-sets", type=int, default=None)
    parser.add_argument("--feature-set", action="append", default=None, help="Run only this named feature set. May be repeated.")
    parser.add_argument("--task", choices=("both", "disease", "score"), default="both")
    parser.add_argument("--fast", action="store_true", help="Use a reduced model tier for faster feature-set sweeps.")
    parser.add_argument("--score-target", action="append", default=None, help="Limit score regression to a target. May be repeated.")
    parser.add_argument("--score-mode", action="append", default=None, help="Limit score regression to a target mode. May be repeated.")
    args = parser.parse_args()
    only = set(args.feature_set) if args.feature_set else None
    out_dir = run_goal_search(
        args.exp_root,
        max_feature_sets=args.max_feature_sets,
        only_feature_sets=only,
        out_dir=args.out_dir,
        task=args.task,
        fast=args.fast,
        score_targets=set(args.score_target) if args.score_target else None,
        score_modes=set(args.score_mode) if args.score_mode else None,
    )
    print(out_dir)


if __name__ == "__main__":
    main()
