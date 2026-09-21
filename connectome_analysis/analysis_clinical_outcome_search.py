from __future__ import annotations

import argparse
import math
import os
import warnings
from collections import Counter
from pathlib import Path

for _thread_var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_thread_var, "1")

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import SelectKBest, mutual_info_classif, mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.kernel_ridge import KernelRidge
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import BayesianRidge, ElasticNet, HuberRegressor, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    explained_variance_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.compose import TransformedTargetRegressor

from connectome_analysis.analysis_config import GROUP_ORDER, get_analysis_paths
from connectome_analysis.analysis_ml_diagnostics import (
    CDR_CLASSIFICATION_TARGET,
    CDR_CLASS_ORDER,
    PRIMARY_SCORE_WINDOW_DAYS,
    SCORE_REGRESSION_TARGETS,
    _cdr_class,
    _score_outlier_filter,
    _specificity_by_class,
)

try:
    from xgboost import XGBClassifier, XGBRegressor

    HAS_XGBOOST = True
except Exception:  # pragma: no cover
    XGBClassifier = None
    XGBRegressor = None
    HAS_XGBOOST = False


SEARCH_TARGET_R2 = 0.70
SEARCH_TARGET_ACCURACY = 0.70
PRIMARY_MODE = f"primary_scan_aligned_{PRIMARY_SCORE_WINDOW_DAYS}d"
HISTORY_MODE = "sensitivity_all_available_history"
OUT_DIR_NAME = "clinical_outcome_model_search"
MMSE_SCORE_MAX = 30.0


class QuantileClipper(BaseEstimator, TransformerMixin):
    def __init__(self, lower: float = 0.01, upper: float = 0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, x, y=None):
        arr = np.asarray(x, dtype=float)
        self.lower_bounds_ = np.nanquantile(arr, self.lower, axis=0)
        self.upper_bounds_ = np.nanquantile(arr, self.upper, axis=0)
        return self

    def transform(self, x):
        arr = np.asarray(x, dtype=float)
        return np.clip(arr, self.lower_bounds_, self.upper_bounds_)


class CumulativeLogitClassifier(BaseEstimator, ClassifierMixin):
    """Ordinal multiclass classifier using one cumulative logistic model per threshold."""

    def __init__(self, C: float = 1.0, max_iter: int = 1500, class_weight: str | None = "balanced"):
        self.C = C
        self.max_iter = max_iter
        self.class_weight = class_weight

    def fit(self, x, y):
        y_arr = np.asarray(y, dtype=int)
        self.classes_ = np.sort(np.unique(y_arr))
        self.n_classes_ = int(len(self.classes_))
        self.models_: list[LogisticRegression | None] = []
        self.constants_: list[float | None] = []
        for threshold_idx in range(self.n_classes_ - 1):
            binary = (y_arr > threshold_idx).astype(int)
            if len(np.unique(binary)) < 2:
                self.models_.append(None)
                self.constants_.append(float(binary[0]))
                continue
            model = LogisticRegression(
                C=self.C,
                penalty="l2",
                solver="liblinear",
                class_weight=self.class_weight,
                max_iter=self.max_iter,
                random_state=42,
            )
            model.fit(x, binary)
            self.models_.append(model)
            self.constants_.append(None)
        return self

    def predict_proba(self, x):
        if self.n_classes_ <= 1:
            return np.ones((len(x), 1), dtype=float)
        cumulative = []
        for model, constant in zip(self.models_, self.constants_):
            if model is None:
                prob = np.full(len(x), float(constant), dtype=float)
            else:
                prob = model.predict_proba(x)[:, 1]
            cumulative.append(prob)
        gt = np.vstack(cumulative).T
        gt = np.minimum.accumulate(gt, axis=1)
        probs = np.zeros((len(x), self.n_classes_), dtype=float)
        probs[:, 0] = 1.0 - gt[:, 0]
        for idx in range(1, self.n_classes_ - 1):
            probs[:, idx] = gt[:, idx - 1] - gt[:, idx]
        probs[:, -1] = gt[:, -1]
        probs = np.clip(probs, 0.0, 1.0)
        row_sum = probs.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        return probs / row_sum

    def predict(self, x):
        probs = self.predict_proba(x)
        return self.classes_[np.argmax(probs, axis=1)]


class RoundedOrdinalRegressorClassifier(BaseEstimator, ClassifierMixin):
    """Treat ordinal labels as ordered numbers and round predictions back to classes."""

    def __init__(self, regressor=None):
        self.regressor = regressor

    def fit(self, x, y):
        self.classes_ = np.sort(np.unique(np.asarray(y, dtype=int)))
        base = self.regressor if self.regressor is not None else ExtraTreesRegressor(n_estimators=240, random_state=42, n_jobs=1)
        self.regressor_ = clone(base)
        self.regressor_.fit(x, np.asarray(y, dtype=float))
        return self

    def predict(self, x):
        raw = np.asarray(self.regressor_.predict(x), dtype=float)
        rounded = np.rint(raw).astype(int)
        rounded = np.clip(rounded, int(self.classes_[0]), int(self.classes_[-1]))
        return rounded

    def predict_proba(self, x):
        raw = np.asarray(self.regressor_.predict(x), dtype=float)
        distances = np.abs(raw[:, None] - self.classes_[None, :])
        logits = -1.8 * distances
        logits -= logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        return probs / probs.sum(axis=1, keepdims=True)


def _mmse_deficit_transform(y):
    arr = np.asarray(y, dtype=float)
    return np.log1p(np.clip(MMSE_SCORE_MAX - arr, 0.0, None))


def _mmse_deficit_inverse(z):
    arr = np.asarray(z, dtype=float)
    return MMSE_SCORE_MAX - np.expm1(arr)


def _mi_regression_fixed(x, y):
    return mutual_info_regression(x, y, random_state=42)


def _mi_classification_fixed(x, y):
    return mutual_info_classif(x, y, random_state=42)


def _analysis_dir(deriv_root: str | Path | None = None) -> Path:
    # With no explicit root the locations come from sc_config, like every other
    # stage. Falling back to a fixed /data/derivatives here used to send this
    # stage's reads and writes to the published tree whatever SC_ANALYSIS_ROOT
    # said -- including during a run meant to start from an empty one.
    if deriv_root:
        paths = get_analysis_paths(deriv_root=deriv_root)
    else:
        paths = get_analysis_paths()
    return paths.section_dir("18", "ml_diagnostics")


def _read_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _allowed_feature(row: pd.Series) -> bool:
    feature = str(row.get("feature", ""))
    group = str(row.get("feature_group", ""))
    lower = f"{group} {feature}".lower()
    if "brain age" in lower or "brain_age" in lower:
        return False
    forbidden = ("clinical", "diagnosis", "group", "apoe", "phase", "sex", "gender", "weight")
    if any(token in lower for token in forbidden):
        return False
    # Avoid explicit age predictors and obvious age-derived names; structural delay/path-length fields are kept.
    if "age" in lower and "advantage" not in lower:
        return False
    return True


def _feature_policies(matrix: pd.DataFrame, catalog: pd.DataFrame) -> dict[str, list[str]]:
    ready = catalog[catalog["final_status"].astype(str).isin(["retained_for_model", "retained_not_selected", "selected_in_cv"])].copy()
    ready = ready[ready.apply(_allowed_feature, axis=1)].copy()
    ready = ready[ready["feature"].astype(str).isin(matrix.columns)].copy()
    ready["selected_fold_count"] = pd.to_numeric(ready.get("selected_fold_count", 0), errors="coerce").fillna(0)
    ready["missingness"] = pd.to_numeric(ready.get("missingness", np.nan), errors="coerce").fillna(1.0)
    low_missing = ready["missingness"].le(0.10)
    very_low_missing = ready["missingness"].le(0.02)
    policies: dict[str, list[str]] = {}
    policies["all_structural_no_age"] = ready["feature"].astype(str).drop_duplicates().tolist()
    policies["selected_cv_no_age"] = ready.loc[ready["final_status"].astype(str).eq("selected_in_cv"), "feature"].astype(str).drop_duplicates().tolist()
    policies["compact_no_raw_edges_no_age"] = ready.loc[
        ~ready["feature_group"].astype(str).isin(["Raw high-variance edges", "Brain age"]),
        "feature",
    ].astype(str).drop_duplicates().tolist()
    policies["selected_no_raw_edges_no_age"] = ready.loc[
        ready["final_status"].astype(str).eq("selected_in_cv")
        & ~ready["feature_group"].astype(str).isin(["Raw high-variance edges", "Brain age"]),
        "feature",
    ].astype(str).drop_duplicates().tolist()
    policies["selected_repeated_cv_no_age"] = ready.loc[
        ready["selected_fold_count"].ge(2)
        & ~ready["feature_group"].astype(str).isin(["Raw high-variance edges", "Brain age"]),
        "feature",
    ].astype(str).drop_duplicates().tolist()
    policies["high_stability_no_age"] = ready.loc[
        ready["selected_fold_count"].ge(10)
        & ~ready["feature_group"].astype(str).isin(["Raw high-variance edges", "Brain age"]),
        "feature",
    ].astype(str).drop_duplicates().tolist()
    policies["top_stability_no_age"] = ready.loc[
        ready["selected_fold_count"].ge(15)
        & ~ready["feature_group"].astype(str).isin(["Raw high-variance edges", "Brain age"]),
        "feature",
    ].astype(str).drop_duplicates().tolist()
    stable_groups = {
        "Global DTI",
        "Local ROI DTI",
        "Node metrics",
        "Raw matrix summaries",
        "LR/SR and delay",
        "EDR exceptions",
        "AAL EDR exceptions",
        "Coupling",
        "Delay",
        "Global graph",
        "Advanced structural",
    }
    policies["stable_summary_no_age"] = ready.loc[ready["feature_group"].astype(str).isin(stable_groups), "feature"].astype(str).drop_duplicates().tolist()
    policies["global_network_summary_no_age"] = ready.loc[
        ready["feature_group"].astype(str).isin(
            [
                "Global DTI",
                "Global graph",
                "Raw matrix summaries",
                "LR/SR and delay",
                "Delay",
                "EDR exceptions",
                "Advanced structural",
                "Coupling",
            ]
        ),
        "feature",
    ].astype(str).drop_duplicates().tolist()
    policies["regional_topology_no_age"] = ready.loc[
        ready["feature_group"].astype(str).isin(["Local ROI DTI", "Node metrics", "Coupling"]),
        "feature",
    ].astype(str).drop_duplicates().tolist()
    policies["edr_delay_no_age"] = ready.loc[
        ready["feature_group"].astype(str).isin(["EDR exceptions", "AAL EDR exceptions", "LR/SR and delay", "Delay"]),
        "feature",
    ].astype(str).drop_duplicates().tolist()
    policies["edr_global_stability_no_age"] = ready.loc[
        ready["selected_fold_count"].ge(5)
        & ready["feature_group"].astype(str).isin(
            [
                "Global DTI",
                "Raw matrix summaries",
                "LR/SR and delay",
                "Delay",
                "EDR exceptions",
                "AAL EDR exceptions",
                "Coupling",
                "Advanced structural",
            ]
        ),
        "feature",
    ].astype(str).drop_duplicates().tolist()
    policies["microstructure_topology_stability_no_age"] = ready.loc[
        ready["selected_fold_count"].ge(10)
        & ready["feature_group"].astype(str).isin(["Global DTI", "Local ROI DTI", "Node metrics", "Raw matrix summaries", "Coupling"]),
        "feature",
    ].astype(str).drop_duplicates().tolist()
    policies["low_missing_structural_no_age"] = ready.loc[
        low_missing
        & ~ready["feature_group"].astype(str).isin(["Raw high-variance edges", "Brain age"]),
        "feature",
    ].astype(str).drop_duplicates().tolist()
    policies["very_low_missing_structural_no_age"] = ready.loc[
        very_low_missing
        & ~ready["feature_group"].astype(str).isin(["Raw high-variance edges", "Brain age"]),
        "feature",
    ].astype(str).drop_duplicates().tolist()
    policies["low_missing_selected_no_age"] = ready.loc[
        low_missing
        & ready["final_status"].astype(str).eq("selected_in_cv")
        & ~ready["feature_group"].astype(str).isin(["Raw high-variance edges", "Brain age"]),
        "feature",
    ].astype(str).drop_duplicates().tolist()
    policies["low_missing_stable_no_age"] = ready.loc[
        low_missing
        & ready["selected_fold_count"].ge(5)
        & ~ready["feature_group"].astype(str).isin(["Raw high-variance edges", "Brain age"]),
        "feature",
    ].astype(str).drop_duplicates().tolist()
    policies["low_missing_global_topology_no_age"] = ready.loc[
        low_missing
        & ready["feature_group"].astype(str).isin(
            [
                "Global DTI",
                "Global graph",
                "Raw matrix summaries",
                "Node metrics",
                "LR/SR and delay",
                "Delay",
                "EDR exceptions",
                "Coupling",
                "Advanced structural",
            ]
        ),
        "feature",
    ].astype(str).drop_duplicates().tolist()
    policies["no_matrix_summaries_no_age"] = ready.loc[
        ~ready["feature_group"].astype(str).isin(["Raw matrix summaries", "Raw high-variance edges", "Brain age"]),
        "feature",
    ].astype(str).drop_duplicates().tolist()
    return {name: features for name, features in policies.items() if len(features) >= 5}


def _gini(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    arr = arr[arr >= 0]
    if arr.size == 0 or float(arr.sum()) == 0.0:
        return np.nan
    arr = np.sort(arr)
    n = arr.size
    return float((2.0 * np.arange(1, n + 1).dot(arr) / (n * arr.sum())) - ((n + 1.0) / n))


def _subject_from_count_path(path: Path) -> str:
    name = path.name
    prefix = "SC_AAL166_"
    suffix = "_count_invnodevol.csv"
    if name.startswith(prefix) and name.endswith(suffix):
        subject_scan = name[len(prefix) : -len(suffix)]
        parts = subject_scan.split("_")
        if len(parts) >= 3:
            return "_".join(parts[:3])
        return subject_scan
    return path.stem


def _graph_topology_from_matrix(subject_id: str, mat: np.ndarray) -> dict:
    try:
        import networkx as nx
        from networkx.algorithms import community
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("networkx is required for graph topology features") from exc

    arr = np.asarray(mat, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"{subject_id}: expected square connectome matrix, got {arr.shape}")
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = np.maximum(arr, arr.T)
    np.fill_diagonal(arr, 0.0)
    positive = arr[arr > 0]
    n_nodes = arr.shape[0]
    possible_edges = n_nodes * (n_nodes - 1) / 2.0
    binary = arr > 0
    n_edges = int(np.triu(binary, 1).sum())
    density = float(n_edges / possible_edges) if possible_edges else np.nan
    strength = arr.sum(axis=1)
    degree = binary.sum(axis=1).astype(float)
    top_n = max(1, int(math.ceil(0.10 * n_nodes)))
    top_strength = np.sort(strength)[-top_n:]
    rows = {
        "subject_id": subject_id,
        "graph_topology__count_density": density,
        "graph_topology__count_edges": float(n_edges),
        "graph_topology__strength_gini": _gini(strength),
        "graph_topology__degree_gini": _gini(degree),
        "graph_topology__edge_weight_gini": _gini(positive),
        "graph_topology__top10_strength_share": float(top_strength.sum() / strength.sum()) if strength.sum() > 0 else np.nan,
        "graph_topology__degree_mean": float(np.nanmean(degree)),
        "graph_topology__degree_sd": float(np.nanstd(degree)),
        "graph_topology__strength_mean": float(np.nanmean(strength)),
        "graph_topology__strength_sd": float(np.nanstd(strength)),
    }
    if n_edges == 0:
        for key in (
            "graph_topology__transitivity",
            "graph_topology__weighted_clustering",
            "graph_topology__degree_assortativity",
            "graph_topology__core_max",
            "graph_topology__core_mean",
            "graph_topology__weighted_modularity",
            "graph_topology__n_communities",
            "graph_topology__participation_mean",
            "graph_topology__participation_sd",
            "graph_topology__hub_edge_density_top10",
            "graph_topology__largest_component_fraction",
        ):
            rows[key] = np.nan
        return rows

    graph = nx.from_numpy_array(arr)
    graph.remove_edges_from(nx.selfloop_edges(graph))
    rows["graph_topology__transitivity"] = float(nx.transitivity(graph))
    try:
        rows["graph_topology__weighted_clustering"] = float(nx.average_clustering(graph, weight="weight"))
    except Exception:
        rows["graph_topology__weighted_clustering"] = np.nan
    try:
        rows["graph_topology__degree_assortativity"] = float(nx.degree_assortativity_coefficient(graph))
    except Exception:
        rows["graph_topology__degree_assortativity"] = np.nan
    try:
        core = np.asarray(list(nx.core_number(graph).values()), dtype=float)
        rows["graph_topology__core_max"] = float(np.nanmax(core))
        rows["graph_topology__core_mean"] = float(np.nanmean(core))
    except Exception:
        rows["graph_topology__core_max"] = np.nan
        rows["graph_topology__core_mean"] = np.nan
    try:
        components = [len(c) for c in nx.connected_components(graph)]
        rows["graph_topology__largest_component_fraction"] = float(max(components) / n_nodes) if components else np.nan
    except Exception:
        rows["graph_topology__largest_component_fraction"] = np.nan

    hub_idx = np.argsort(strength)[-top_n:]
    if top_n > 1:
        hub_edges = int(np.triu(binary[np.ix_(hub_idx, hub_idx)], 1).sum())
        rows["graph_topology__hub_edge_density_top10"] = float(hub_edges / (top_n * (top_n - 1) / 2.0))
    else:
        rows["graph_topology__hub_edge_density_top10"] = np.nan

    try:
        communities = list(community.greedy_modularity_communities(graph, weight="weight"))
        rows["graph_topology__weighted_modularity"] = float(community.modularity(graph, communities, weight="weight"))
        rows["graph_topology__n_communities"] = float(len(communities))
        node_to_comm = {}
        for comm_idx, comm_nodes in enumerate(communities):
            for node in comm_nodes:
                node_to_comm[int(node)] = comm_idx
        participation = []
        for node in range(n_nodes):
            total = float(strength[node])
            if total <= 0:
                continue
            by_comm: dict[int, float] = {}
            neighbors = np.flatnonzero(arr[node] > 0)
            for neighbor in neighbors:
                comm_idx = node_to_comm.get(int(neighbor), -1)
                by_comm[comm_idx] = by_comm.get(comm_idx, 0.0) + float(arr[node, neighbor])
            participation.append(1.0 - sum((value / total) ** 2 for value in by_comm.values()))
        rows["graph_topology__participation_mean"] = float(np.nanmean(participation)) if participation else np.nan
        rows["graph_topology__participation_sd"] = float(np.nanstd(participation)) if participation else np.nan
    except Exception:
        rows["graph_topology__weighted_modularity"] = np.nan
        rows["graph_topology__n_communities"] = np.nan
        rows["graph_topology__participation_mean"] = np.nan
        rows["graph_topology__participation_sd"] = np.nan
    return rows


def _load_or_build_graph_topology_features(ml_dir: Path, matrix: pd.DataFrame) -> pd.DataFrame:
    out_dir = ml_dir / OUT_DIR_NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / "graph_topology_subject_features.csv"
    expected_subjects = set(matrix["subject_id"].astype(str))
    if cache_path.exists():
        try:
            cached = pd.read_csv(cache_path)
            if "subject_id" in cached.columns and expected_subjects.issubset(set(cached["subject_id"].astype(str))):
                return cached
        except Exception:
            pass
    connectome_dir = ml_dir.parents[2] / "connectomes"
    rows = []
    for path in sorted(connectome_dir.glob("SC_AAL166_*_count_invnodevol.csv")):
        subject_id = _subject_from_count_path(path)
        if subject_id not in expected_subjects:
            continue
        try:
            mat = pd.read_csv(path, header=None).to_numpy(dtype=float)
            rows.append(_graph_topology_from_matrix(subject_id, mat))
        except Exception as exc:
            rows.append({"subject_id": subject_id, "graph_topology__error": str(exc)})
    features = pd.DataFrame(rows)
    features.to_csv(cache_path, index=False)
    return features


def _add_graph_topology_features(
    ml_dir: Path, matrix: pd.DataFrame, policies: dict[str, list[str]]
) -> tuple[pd.DataFrame, dict[str, list[str]], pd.DataFrame]:
    features = _load_or_build_graph_topology_features(ml_dir, matrix)
    if features.empty or "subject_id" not in features.columns:
        return matrix, policies, pd.DataFrame(
            [
                {
                    "meta_feature_family": "graph_topology",
                    "source_model": "count_invnodevol_connectome",
                    "n_subjects": 0,
                    "note": "Graph topology feature derivation skipped because no connectome matrices were found.",
                }
            ]
        )
    feature_cols = [col for col in features.columns if col.startswith("graph_topology__") and not col.endswith("__error")]
    out = matrix.drop(columns=feature_cols, errors="ignore").merge(features[["subject_id", *feature_cols]], on="subject_id", how="left")
    policies = dict(policies)
    policies["graph_topology_no_age"] = feature_cols
    for base_name in (
        "top_stability_no_age",
        "low_missing_global_topology_no_age",
        "no_matrix_summaries_no_age",
        "compact_no_raw_edges_no_age",
    ):
        if base_name in policies:
            policies[f"{base_name}_plus_graph_topology_no_age"] = [*policies[base_name], *feature_cols]
    audit = pd.DataFrame(
        [
            {
                "meta_feature_family": "graph_topology",
                "source_model": "count_invnodevol_connectome",
                "n_subjects": int(features["subject_id"].nunique()),
                "note": "No-age structural graph features derived from AAL count connectomes: modularity, community participation, core, hub concentration, clustering, assortativity, and inequality summaries.",
            }
        ]
    )
    return out, policies, audit


def _add_disease_meta_features(ml_dir: Path, matrix: pd.DataFrame, policies: dict[str, list[str]]) -> tuple[pd.DataFrame, dict[str, list[str]], pd.DataFrame]:
    pred_path = ml_dir / "ml_cross_validated_predictions.csv"
    if not pred_path.exists():
        return matrix, policies, pd.DataFrame()
    pred = pd.read_csv(pred_path)
    needed = {"subject_id", "model", "prob_AD", "prob_CN", "prob_MCI"}
    if pred.empty or not needed.issubset(pred.columns):
        return matrix, policies, pd.DataFrame()
    frames: list[pd.DataFrame] = []
    audit_rows: list[dict] = []
    prob_cols = ["prob_CN", "prob_MCI", "prob_AD"]
    for model, sub in pred.groupby("model", dropna=False):
        sub = sub[["subject_id", *prob_cols]].copy()
        for col in prob_cols:
            sub[col] = pd.to_numeric(sub[col], errors="coerce")
        agg = sub.groupby("subject_id", as_index=False)[prob_cols].mean()
        rename = {col: f"disease_meta__{model}__{col.replace('prob_', 'prob_')}" for col in prob_cols}
        agg = agg.rename(columns=rename)
        frames.append(agg)
        audit_rows.append(
            {
                "meta_feature_family": "disease_oof_probability",
                "source_model": model,
                "n_subjects": int(agg["subject_id"].nunique()),
                "note": "Out-of-fold disease-classifier probabilities from structural features; diagnosis labels are not joined as direct predictors.",
            }
        )
    ensemble = pred[["subject_id", *prob_cols]].copy()
    for col in prob_cols:
        ensemble[col] = pd.to_numeric(ensemble[col], errors="coerce")
    ensemble = ensemble.groupby("subject_id", as_index=False)[prob_cols].mean()
    ensemble = ensemble.rename(columns={col: f"disease_meta__ensemble__{col.replace('prob_', 'prob_')}" for col in prob_cols})
    frames.append(ensemble)
    audit_rows.append(
        {
            "meta_feature_family": "disease_oof_probability",
            "source_model": "ensemble_mean",
            "n_subjects": int(ensemble["subject_id"].nunique()),
            "note": "Mean out-of-fold disease-classifier probabilities across structural disease models.",
        }
    )
    meta = frames[0]
    for frame in frames[1:]:
        meta = meta.merge(frame, on="subject_id", how="outer")
    out = matrix.merge(meta, on="subject_id", how="left")
    meta_features = [col for col in meta.columns if col != "subject_id"]
    if meta_features:
        policies = dict(policies)
        policies["disease_meta_only_experimental"] = meta_features
        for base_name in (
            "selected_cv_no_age",
            "top_stability_no_age",
            "global_network_summary_no_age",
            "low_missing_global_topology_no_age",
            "compact_no_raw_edges_no_age",
            "no_matrix_summaries_no_age",
        ):
            if base_name in policies:
                policies[f"{base_name}_plus_disease_meta_experimental"] = [*policies[base_name], *meta_features]
    return out, policies, pd.DataFrame(audit_rows)


def _add_age_feature(ml_dir: Path, matrix: pd.DataFrame, policies: dict[str, list[str]]) -> tuple[pd.DataFrame, dict[str, list[str]], pd.DataFrame]:
    master_path = ml_dir.parent / "00_master" / "master_cohort.csv"
    if not master_path.exists():
        return matrix, policies, pd.DataFrame(
            [
                {
                    "meta_feature_family": "chronological_age",
                    "source_model": "master_cohort",
                    "n_subjects": 0,
                    "note": f"Requested age experiment skipped because {master_path} was not found.",
                }
            ]
        )
    master = pd.read_csv(master_path)
    age_col = "age" if "age" in master.columns else ("Age" if "Age" in master.columns else None)
    if age_col is None or "subject_id" not in master.columns:
        return matrix, policies, pd.DataFrame(
            [
                {
                    "meta_feature_family": "chronological_age",
                    "source_model": "master_cohort",
                    "n_subjects": 0,
                    "note": "Requested age experiment skipped because master_cohort.csv lacks subject_id and age/Age columns.",
                }
            ]
        )
    age_feature = "age__chronological_years"
    age = master[["subject_id", age_col]].copy()
    age[age_feature] = pd.to_numeric(age[age_col], errors="coerce")
    age = age[["subject_id", age_feature]].drop_duplicates("subject_id")
    out = matrix.drop(columns=[age_feature], errors="ignore").merge(age, on="subject_id", how="left")
    age_features = [age_feature]
    policies = dict(policies)
    policies["age_only_experimental"] = age_features
    for base_name in (
        "selected_cv_no_age",
        "selected_repeated_cv_no_age",
        "high_stability_no_age",
        "top_stability_no_age",
        "edr_delay_no_age",
        "global_network_summary_no_age",
        "low_missing_stable_no_age",
        "low_missing_global_topology_no_age",
        "compact_no_raw_edges_no_age",
    ):
        if base_name in policies:
            policies[f"{base_name}_plus_age_experimental"] = [*policies[base_name], *age_features]
    audit = pd.DataFrame(
        [
            {
                "meta_feature_family": "chronological_age",
                "source_model": "master_cohort",
                "n_subjects": int(age[age_feature].notna().sum()),
                "note": "Explicit user-requested sensitivity experiment: chronological age from master_cohort.csv is included as a predictor.",
            }
        ]
    )
    return out, policies, audit


def _add_non_age_metadata_features(
    ml_dir: Path, matrix: pd.DataFrame, policies: dict[str, list[str]]
) -> tuple[pd.DataFrame, dict[str, list[str]], pd.DataFrame]:
    master_path = ml_dir.parent / "00_master" / "master_cohort.csv"
    family = "non_age_clinical_metadata"
    if not master_path.exists():
        return matrix, policies, pd.DataFrame(
            [
                {
                    "meta_feature_family": family,
                    "source_model": "master_cohort",
                    "n_subjects": 0,
                    "note": f"Requested non-age metadata experiment skipped because {master_path} was not found.",
                }
            ]
        )
    master = pd.read_csv(master_path)
    if "subject_id" not in master.columns:
        return matrix, policies, pd.DataFrame(
            [
                {
                    "meta_feature_family": family,
                    "source_model": "master_cohort",
                    "n_subjects": 0,
                    "note": "Requested non-age metadata experiment skipped because master_cohort.csv lacks subject_id.",
                }
            ]
        )
    meta = master[["subject_id"]].copy()
    sex_col = "sex" if "sex" in master.columns else ("Sex" if "Sex" in master.columns else None)
    if sex_col is not None:
        sex = master[sex_col].astype(str).str.strip().str.upper()
        meta["metadata__sex_female"] = sex.str.startswith("F").astype(float)
        meta["metadata__sex_male"] = sex.str.startswith("M").astype(float)
        meta.loc[~sex.str.startswith(("F", "M")), ["metadata__sex_female", "metadata__sex_male"]] = np.nan
    for source_col, feature_col in (
        ("apoe_e4_count", "metadata__apoe_e4_count"),
        ("apoe_e4_carrier", "metadata__apoe_e4_carrier"),
    ):
        if source_col in master.columns:
            meta[feature_col] = pd.to_numeric(master[source_col], errors="coerce")
    meta_features = [col for col in meta.columns if col != "subject_id"]
    if not meta_features:
        return matrix, policies, pd.DataFrame(
            [
                {
                    "meta_feature_family": family,
                    "source_model": "master_cohort",
                    "n_subjects": 0,
                    "note": "Requested non-age metadata experiment skipped because sex/APOE features were unavailable.",
                }
            ]
        )
    meta = meta[["subject_id", *meta_features]].drop_duplicates("subject_id")
    out = matrix.drop(columns=meta_features, errors="ignore").merge(meta, on="subject_id", how="left")
    policies = dict(policies)
    policies["non_age_metadata_only_experimental"] = meta_features
    for base_name in (
        "selected_cv_no_age",
        "top_stability_no_age",
        "global_network_summary_no_age",
        "low_missing_global_topology_no_age",
        "compact_no_raw_edges_no_age",
        "no_matrix_summaries_no_age",
    ):
        if base_name in policies:
            policies[f"{base_name}_plus_non_age_metadata_experimental"] = [*policies[base_name], *meta_features]
    audit = pd.DataFrame(
        [
            {
                "meta_feature_family": family,
                "source_model": "master_cohort",
                "n_subjects": int(meta[meta_features].notna().any(axis=1).sum()),
                "note": "Explicit no-age sensitivity experiment: sex and APOE e4 features from master_cohort.csv are included; age, diagnosis labels, and clinical scores are excluded.",
            }
        ]
    )
    return out, policies, audit


def _is_explicit_age_policy(series: pd.Series) -> pd.Series:
    text = series.astype(str)
    return text.eq("age_only_experimental") | text.str.endswith("_plus_age_experimental") | text.str.contains(
        "plus_age_experimental", regex=False
    )


def _is_non_age_metadata_policy(series: pd.Series) -> pd.Series:
    text = series.astype(str)
    return text.eq("non_age_metadata_only_experimental") | text.str.contains(
        "plus_non_age_metadata_experimental", regex=False
    )


def _is_disease_meta_policy(series: pd.Series) -> pd.Series:
    text = series.astype(str)
    return text.eq("disease_meta_only_experimental") | text.str.contains("plus_disease_meta_experimental", regex=False)


def _k_values(n_features: int, n_subjects: int, fast: bool) -> list[int]:
    candidates = [5, 10, 20, 40, 80, 140] if not fast else [10, 20, 60]
    upper = max(2, min(n_features, n_subjects - 2))
    return sorted({min(upper, k) for k in candidates if min(upper, k) >= 2})


def _mmse_deficit_regressor(pipe: Pipeline) -> TransformedTargetRegressor:
    return TransformedTargetRegressor(
        regressor=pipe,
        func=_mmse_deficit_transform,
        inverse_func=_mmse_deficit_inverse,
        check_inverse=False,
    )


def _regression_specs(n_features: int, n_subjects: int, fast: bool) -> dict[str, tuple[Pipeline, dict]]:
    k_values = _k_values(n_features, n_subjects, fast)
    mi = _mi_regression_fixed
    specs: dict[str, tuple[Pipeline, dict]] = {
        "search_ridge_select": (
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("clip", QuantileClipper()),
                    ("select", SelectKBest(score_func=mi, k=k_values[0])),
                    ("scaler", StandardScaler()),
                    ("model", Ridge(random_state=42)),
                ]
            ),
            {"select__k": k_values, "model__alpha": [0.5, 2.0, 8.0] if not fast else [2.0]},
        ),
        "search_ridge_indicator_select": (
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                    ("clip", QuantileClipper()),
                    ("select", SelectKBest(score_func=mi, k=k_values[0])),
                    ("scaler", StandardScaler()),
                    ("model", Ridge(random_state=42)),
                ]
            ),
            {"select__k": k_values, "model__alpha": [0.5, 2.0, 8.0] if not fast else [2.0]},
        ),
        "search_elastic_net_select": (
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("clip", QuantileClipper()),
                    ("select", SelectKBest(score_func=mi, k=k_values[0])),
                    ("scaler", StandardScaler()),
                    ("model", ElasticNet(max_iter=12000, random_state=42)),
                ]
            ),
            {"select__k": k_values, "model__alpha": [0.05, 0.2, 0.8] if not fast else [0.2], "model__l1_ratio": [0.2, 0.5, 0.8] if not fast else [0.5]},
        ),
        "search_bayesian_ridge_select": (
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("clip", QuantileClipper()),
                    ("select", SelectKBest(score_func=mi, k=k_values[0])),
                    ("scaler", StandardScaler()),
                    ("model", BayesianRidge()),
                ]
            ),
            {"select__k": k_values, "model__alpha_1": [1e-6, 1e-4] if not fast else [1e-6], "model__lambda_1": [1e-6, 1e-4] if not fast else [1e-6]},
        ),
        "search_huber_select": (
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("clip", QuantileClipper()),
                    ("select", SelectKBest(score_func=mi, k=k_values[0])),
                    ("scaler", StandardScaler()),
                    ("model", HuberRegressor(max_iter=1000)),
                ]
            ),
            {"select__k": k_values, "model__alpha": [1e-4, 1e-2, 1e-1] if not fast else [1e-2], "model__epsilon": [1.2, 1.35, 1.8] if not fast else [1.35]},
        ),
        "search_extra_trees_regression": (
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("clip", QuantileClipper()),
                    ("select", SelectKBest(score_func=mi, k=k_values[0])),
                    ("model", ExtraTreesRegressor(n_estimators=260 if fast else 360, random_state=42, n_jobs=1, max_features="sqrt")),
                ]
            ),
            {"select__k": k_values, "model__min_samples_leaf": [1, 2, 4] if not fast else [2], "model__max_depth": [None, 8] if not fast else [None]},
        ),
        "search_extra_trees_indicator_regression": (
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                    ("clip", QuantileClipper()),
                    ("select", SelectKBest(score_func=mi, k=k_values[0])),
                    ("model", ExtraTreesRegressor(n_estimators=260 if fast else 360, random_state=42, n_jobs=1, max_features="sqrt")),
                ]
            ),
            {"select__k": k_values, "model__min_samples_leaf": [1, 2, 4] if not fast else [2], "model__max_depth": [None, 8] if not fast else [None]},
        ),
        "search_extra_trees_conservative_regression": (
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("clip", QuantileClipper()),
                    ("select", SelectKBest(score_func=mi, k=k_values[0])),
                    (
                        "model",
                        ExtraTreesRegressor(
                            n_estimators=320 if fast else 460,
                            random_state=43,
                            n_jobs=1,
                            max_features=0.45,
                            min_samples_leaf=3,
                        ),
                    ),
                ]
            ),
            {"select__k": k_values, "model__max_depth": [None, 8] if not fast else [None]},
        ),
        "search_random_forest_regression": (
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("clip", QuantileClipper()),
                    ("select", SelectKBest(score_func=mi, k=k_values[0])),
                    ("model", RandomForestRegressor(n_estimators=240 if fast else 360, random_state=42, n_jobs=1, max_features="sqrt")),
                ]
            ),
            {"select__k": k_values, "model__min_samples_leaf": [2, 4] if not fast else [2], "model__max_depth": [None, 8] if not fast else [None]},
        ),
        "search_gradient_boosting_regression": (
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("clip", QuantileClipper()),
                    ("select", SelectKBest(score_func=mi, k=k_values[0])),
                    ("model", GradientBoostingRegressor(random_state=42)),
                ]
            ),
            {
                "select__k": k_values,
                "model__n_estimators": [80, 140] if not fast else [100],
                "model__learning_rate": [0.025, 0.05] if not fast else [0.04],
                "model__max_depth": [1, 2] if not fast else [1],
            },
        ),
        "search_extra_trees_mmse_deficit_regression": (
            _mmse_deficit_regressor(
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("clip", QuantileClipper()),
                        ("select", SelectKBest(score_func=mi, k=k_values[0])),
                        ("model", ExtraTreesRegressor(n_estimators=260 if fast else 360, random_state=42, n_jobs=1, max_features="sqrt")),
                    ]
                )
            ),
            {
                "regressor__select__k": k_values,
                "regressor__model__min_samples_leaf": [1, 2, 4] if not fast else [2],
                "regressor__model__max_depth": [None, 8] if not fast else [None],
            },
        ),
        "search_hist_gradient_boosting_regression": (
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("clip", QuantileClipper()),
                    ("select", SelectKBest(score_func=mi, k=k_values[0])),
                    ("model", HistGradientBoostingRegressor(random_state=42)),
                ]
            ),
            {
                "select__k": k_values,
                "model__learning_rate": [0.025, 0.05] if not fast else [0.035],
                "model__max_leaf_nodes": [7, 15] if not fast else [15],
                "model__l2_regularization": [0.05, 0.3] if not fast else [0.05],
            },
        ),
        "search_svr_rbf": (
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("clip", QuantileClipper()),
                    ("select", SelectKBest(score_func=mi, k=k_values[0])),
                    ("scaler", StandardScaler()),
                    ("model", SVR(kernel="rbf")),
                ]
            ),
            {"select__k": k_values, "model__C": [1.0, 5.0, 20.0] if not fast else [5.0], "model__epsilon": [0.2, 0.8] if not fast else [0.5]},
        ),
        "search_svr_rbf_indicator": (
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                    ("clip", QuantileClipper()),
                    ("select", SelectKBest(score_func=mi, k=k_values[0])),
                    ("scaler", StandardScaler()),
                    ("model", SVR(kernel="rbf")),
                ]
            ),
            {"select__k": k_values, "model__C": [1.0, 5.0, 20.0] if not fast else [5.0], "model__epsilon": [0.2, 0.8] if not fast else [0.5]},
        ),
        "search_kernel_ridge_rbf": (
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("clip", QuantileClipper()),
                    ("select", SelectKBest(score_func=mi, k=k_values[0])),
                    ("scaler", StandardScaler()),
                    ("model", KernelRidge(kernel="rbf")),
                ]
            ),
            {
                "select__k": k_values,
                "model__alpha": [0.2, 1.0, 5.0] if not fast else [1.0],
                "model__gamma": [0.003, 0.01, 0.03] if not fast else [0.01],
            },
        ),
        "search_pca_ridge": (
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("clip", QuantileClipper()),
                    ("scaler", StandardScaler()),
                    ("pca", PCA(random_state=42)),
                    ("model", Ridge(random_state=42)),
                ]
            ),
            {"pca__n_components": [4, 8, 12] if not fast else [8], "model__alpha": [1.0, 8.0] if not fast else [4.0]},
        ),
        "search_knn_regression": (
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("clip", QuantileClipper()),
                    ("select", SelectKBest(score_func=mi, k=k_values[0])),
                    ("scaler", StandardScaler()),
                    ("model", KNeighborsRegressor()),
                ]
            ),
            {"select__k": k_values, "model__n_neighbors": [5, 9, 15] if not fast else [9], "model__weights": ["distance", "uniform"] if not fast else ["distance"]},
        ),
        "search_pls_regression": (
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("clip", QuantileClipper()),
                    ("scaler", StandardScaler()),
                    ("model", PLSRegression()),
                ]
            ),
            {"model__n_components": [2, 4, 8] if not fast else [4]},
        ),
    }
    if n_subjects >= 80:
        specs["search_mlp_regression"] = (
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("clip", QuantileClipper()),
                    ("select", SelectKBest(score_func=mi, k=k_values[0])),
                    ("scaler", StandardScaler()),
                    ("model", MLPRegressor(hidden_layer_sizes=(64, 16), early_stopping=True, max_iter=350, random_state=42)),
                ]
            ),
            {"select__k": k_values, "model__alpha": [1e-2, 5e-2] if not fast else [1e-2]},
        )
    if HAS_XGBOOST and XGBRegressor is not None:
        specs["search_xgboost_regression"] = (
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("clip", QuantileClipper()),
                    ("select", SelectKBest(score_func=mi, k=k_values[0])),
                    (
                        "model",
                        XGBRegressor(
                            objective="reg:squarederror",
                            eval_metric="rmse",
                            tree_method="hist",
                            n_estimators=220,
                            random_state=42,
                            n_jobs=1,
                        ),
                    ),
                ]
            ),
            {"select__k": k_values, "model__max_depth": [2, 3] if not fast else [2], "model__learning_rate": [0.025, 0.05] if not fast else [0.04]},
        )
        specs["search_xgboost_mmse_deficit_regression"] = (
            _mmse_deficit_regressor(
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("clip", QuantileClipper()),
                        ("select", SelectKBest(score_func=mi, k=k_values[0])),
                        (
                            "model",
                            XGBRegressor(
                                objective="reg:squarederror",
                                eval_metric="rmse",
                                tree_method="hist",
                                n_estimators=260,
                                random_state=42,
                                n_jobs=1,
                            ),
                        ),
                    ]
                )
            ),
            {
                "regressor__select__k": k_values,
                "regressor__model__max_depth": [2, 3] if not fast else [2],
                "regressor__model__learning_rate": [0.02, 0.04] if not fast else [0.035],
            },
        )
    if fast:
        keep = {
            "search_ridge_select",
            "search_ridge_indicator_select",
            "search_bayesian_ridge_select",
            "search_huber_select",
            "search_extra_trees_regression",
            "search_extra_trees_indicator_regression",
            "search_extra_trees_conservative_regression",
            "search_extra_trees_mmse_deficit_regression",
            "search_random_forest_regression",
            "search_gradient_boosting_regression",
            "search_hist_gradient_boosting_regression",
            "search_svr_rbf",
            "search_svr_rbf_indicator",
            "search_kernel_ridge_rbf",
            "search_knn_regression",
            "search_xgboost_regression",
            "search_xgboost_mmse_deficit_regression",
        }
        specs = {name: spec for name, spec in specs.items() if name in keep}
    return specs


def _classification_specs(n_features: int, n_subjects: int, n_classes: int, fast: bool) -> dict[str, tuple[Pipeline, dict]]:
    k_values = _k_values(n_features, n_subjects, fast)
    mi = _mi_classification_fixed
    specs: dict[str, tuple[Pipeline, dict]] = {
        "search_multinomial_elastic_net": (
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("clip", QuantileClipper()),
                    ("select", SelectKBest(score_func=mi, k=k_values[0])),
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            penalty="elasticnet",
                            solver="saga",
                            class_weight="balanced",
                            max_iter=1500,
                            random_state=42,
                        ),
                    ),
                ]
            ),
            {"select__k": k_values, "model__C": [0.4, 1.0, 3.0] if not fast else [1.0], "model__l1_ratio": [0.2, 0.5, 0.8] if not fast else [0.5]},
        ),
        "search_multinomial_elastic_net_accuracy": (
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("clip", QuantileClipper()),
                    ("select", SelectKBest(score_func=mi, k=k_values[0])),
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            penalty="elasticnet",
                            solver="saga",
                            class_weight=None,
                            max_iter=2000,
                            random_state=42,
                        ),
                    ),
                ]
            ),
            {"select__k": k_values, "model__C": [0.4, 1.0, 3.0] if not fast else [1.0], "model__l1_ratio": [0.2, 0.5, 0.8] if not fast else [0.5]},
        ),
        "search_multinomial_elastic_net_accuracy_indicator": (
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                    ("clip", QuantileClipper()),
                    ("select", SelectKBest(score_func=mi, k=k_values[0])),
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            penalty="elasticnet",
                            solver="saga",
                            class_weight=None,
                            max_iter=2000,
                            random_state=42,
                        ),
                    ),
                ]
            ),
            {"select__k": k_values, "model__C": [0.4, 1.0, 3.0] if not fast else [1.0], "model__l1_ratio": [0.2, 0.5, 0.8] if not fast else [0.5]},
        ),
        "search_extra_trees_multiclass": (
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("clip", QuantileClipper()),
                    ("select", SelectKBest(score_func=mi, k=k_values[0])),
                    ("model", ExtraTreesClassifier(n_estimators=260 if fast else 360, class_weight="balanced", random_state=42, n_jobs=1, max_features="sqrt")),
                ]
            ),
            {"select__k": k_values, "model__min_samples_leaf": [1, 2, 4] if not fast else [2], "model__max_depth": [None, 8] if not fast else [None]},
        ),
        "search_extra_trees_multiclass_accuracy": (
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("clip", QuantileClipper()),
                    ("select", SelectKBest(score_func=mi, k=k_values[0])),
                    ("model", ExtraTreesClassifier(n_estimators=260 if fast else 360, class_weight=None, random_state=42, n_jobs=1, max_features="sqrt")),
                ]
            ),
            {"select__k": k_values, "model__min_samples_leaf": [1, 2, 4] if not fast else [2], "model__max_depth": [None, 8] if not fast else [None]},
        ),
        "search_extra_trees_multiclass_accuracy_indicator": (
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                    ("clip", QuantileClipper()),
                    ("select", SelectKBest(score_func=mi, k=k_values[0])),
                    ("model", ExtraTreesClassifier(n_estimators=260 if fast else 360, class_weight=None, random_state=42, n_jobs=1, max_features="sqrt")),
                ]
            ),
            {"select__k": k_values, "model__min_samples_leaf": [1, 2, 4] if not fast else [2], "model__max_depth": [None, 8] if not fast else [None]},
        ),
        "search_extra_trees_conservative_multiclass": (
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("clip", QuantileClipper()),
                    ("select", SelectKBest(score_func=mi, k=k_values[0])),
                    (
                        "model",
                        ExtraTreesClassifier(
                            n_estimators=320 if fast else 460,
                            class_weight="balanced",
                            random_state=43,
                            n_jobs=1,
                            max_features=0.45,
                            min_samples_leaf=3,
                        ),
                    ),
                ]
            ),
            {"select__k": k_values, "model__max_depth": [None, 8] if not fast else [None]},
        ),
        "search_random_forest_multiclass": (
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("clip", QuantileClipper()),
                    ("select", SelectKBest(score_func=mi, k=k_values[0])),
                    ("model", RandomForestClassifier(n_estimators=240 if fast else 360, class_weight="balanced_subsample", random_state=42, n_jobs=1, max_features="sqrt")),
                ]
            ),
            {"select__k": k_values, "model__min_samples_leaf": [2, 4] if not fast else [2], "model__max_depth": [None, 8] if not fast else [None]},
        ),
        "search_random_forest_multiclass_accuracy": (
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("clip", QuantileClipper()),
                    ("select", SelectKBest(score_func=mi, k=k_values[0])),
                    ("model", RandomForestClassifier(n_estimators=240 if fast else 360, class_weight=None, random_state=42, n_jobs=1, max_features="sqrt")),
                ]
            ),
            {"select__k": k_values, "model__min_samples_leaf": [2, 4] if not fast else [2], "model__max_depth": [None, 8] if not fast else [None]},
        ),
        "search_gradient_boosting_multiclass": (
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("clip", QuantileClipper()),
                    ("select", SelectKBest(score_func=mi, k=k_values[0])),
                    ("model", GradientBoostingClassifier(random_state=42)),
                ]
            ),
            {
                "select__k": k_values,
                "model__n_estimators": [80, 140] if not fast else [100],
                "model__learning_rate": [0.025, 0.05] if not fast else [0.04],
                "model__max_depth": [1, 2] if not fast else [1],
            },
        ),
        "search_hist_gradient_boosting_multiclass": (
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("clip", QuantileClipper()),
                    ("select", SelectKBest(score_func=mi, k=k_values[0])),
                    ("model", HistGradientBoostingClassifier(random_state=42)),
                ]
            ),
            {"select__k": k_values, "model__learning_rate": [0.025, 0.05] if not fast else [0.04], "model__max_leaf_nodes": [7, 15] if not fast else [15]},
        ),
        "search_lda_shrinkage": (
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("clip", QuantileClipper()),
                    ("select", SelectKBest(score_func=mi, k=k_values[0])),
                    ("scaler", StandardScaler()),
                    ("model", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
                ]
            ),
            {"select__k": k_values},
        ),
        "search_gaussian_nb": (
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("clip", QuantileClipper()),
                    ("select", SelectKBest(score_func=mi, k=k_values[0])),
                    ("model", GaussianNB()),
                ]
            ),
            {"select__k": k_values, "model__var_smoothing": [1e-9, 1e-7, 1e-5] if not fast else [1e-7]},
        ),
        "search_knn_multiclass": (
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("clip", QuantileClipper()),
                    ("select", SelectKBest(score_func=mi, k=k_values[0])),
                    ("scaler", StandardScaler()),
                    ("model", KNeighborsClassifier()),
                ]
            ),
            {"select__k": k_values, "model__n_neighbors": [5, 9, 15] if not fast else [9], "model__weights": ["distance", "uniform"] if not fast else ["distance"]},
        ),
        "search_svc_rbf": (
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("clip", QuantileClipper()),
                    ("select", SelectKBest(score_func=mi, k=k_values[0])),
                    ("scaler", StandardScaler()),
                    ("model", SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=42)),
                ]
            ),
            {"select__k": k_values, "model__C": [0.5, 2.0, 8.0] if not fast else [2.0], "model__gamma": ["scale", "auto"] if not fast else ["scale"]},
        ),
        "search_svc_rbf_accuracy": (
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("clip", QuantileClipper()),
                    ("select", SelectKBest(score_func=mi, k=k_values[0])),
                    ("scaler", StandardScaler()),
                    ("model", SVC(kernel="rbf", class_weight=None, probability=True, random_state=42)),
                ]
            ),
            {"select__k": k_values, "model__C": [0.5, 2.0, 8.0] if not fast else [2.0], "model__gamma": ["scale", "auto"] if not fast else ["scale"]},
        ),
        "search_cumulative_logit_ordinal": (
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("clip", QuantileClipper()),
                    ("select", SelectKBest(score_func=mi, k=k_values[0])),
                    ("scaler", StandardScaler()),
                    ("model", CumulativeLogitClassifier()),
                ]
            ),
            {"select__k": k_values, "model__C": [0.4, 1.0, 3.0] if not fast else [1.0]},
        ),
        "search_cumulative_logit_accuracy_ordinal": (
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("clip", QuantileClipper()),
                    ("select", SelectKBest(score_func=mi, k=k_values[0])),
                    ("scaler", StandardScaler()),
                    ("model", CumulativeLogitClassifier(class_weight=None)),
                ]
            ),
            {"select__k": k_values, "model__C": [0.4, 1.0, 3.0] if not fast else [1.0]},
        ),
        "search_extra_trees_ordinal_regressor": (
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("clip", QuantileClipper()),
                    ("select", SelectKBest(score_func=mi, k=k_values[0])),
                    (
                        "model",
                        RoundedOrdinalRegressorClassifier(
                            ExtraTreesRegressor(n_estimators=260 if fast else 360, random_state=42, n_jobs=1, max_features="sqrt")
                        ),
                    ),
                ]
            ),
            {
                "select__k": k_values,
                "model__regressor__min_samples_leaf": [1, 2, 4] if not fast else [2],
                "model__regressor__max_depth": [None, 8] if not fast else [None],
            },
        ),
        "search_random_forest_ordinal_regressor": (
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("clip", QuantileClipper()),
                    ("select", SelectKBest(score_func=mi, k=k_values[0])),
                    (
                        "model",
                        RoundedOrdinalRegressorClassifier(
                            RandomForestRegressor(n_estimators=240 if fast else 360, random_state=42, n_jobs=1, max_features="sqrt")
                        ),
                    ),
                ]
            ),
            {
                "select__k": k_values,
                "model__regressor__min_samples_leaf": [2, 4] if not fast else [2],
                "model__regressor__max_depth": [None, 8] if not fast else [None],
            },
        ),
    }
    if n_subjects >= 120:
        specs["search_mlp_multiclass"] = (
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("clip", QuantileClipper()),
                    ("select", SelectKBest(score_func=mi, k=k_values[0])),
                    ("scaler", StandardScaler()),
                    ("model", MLPClassifier(hidden_layer_sizes=(64, 16), early_stopping=True, max_iter=350, random_state=42)),
                ]
            ),
            {"select__k": k_values, "model__alpha": [1e-2, 5e-2] if not fast else [1e-2]},
        )
    if HAS_XGBOOST and XGBClassifier is not None:
        specs["search_xgboost_multiclass"] = (
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("clip", QuantileClipper()),
                    ("select", SelectKBest(score_func=mi, k=k_values[0])),
                    (
                        "model",
                        XGBClassifier(
                            objective="multi:softprob",
                            eval_metric="mlogloss",
                            num_class=n_classes,
                            tree_method="hist",
                            n_estimators=220,
                            random_state=42,
                            n_jobs=1,
                        ),
                    ),
                ]
            ),
            {"select__k": k_values, "model__max_depth": [2, 3] if not fast else [2], "model__learning_rate": [0.025, 0.05] if not fast else [0.04]},
        )
    if fast:
        keep = {
            "search_multinomial_elastic_net",
            "search_multinomial_elastic_net_accuracy",
            "search_multinomial_elastic_net_accuracy_indicator",
            "search_extra_trees_multiclass",
            "search_extra_trees_multiclass_accuracy",
            "search_extra_trees_multiclass_accuracy_indicator",
            "search_extra_trees_conservative_multiclass",
            "search_random_forest_multiclass",
            "search_random_forest_multiclass_accuracy",
            "search_gradient_boosting_multiclass",
            "search_hist_gradient_boosting_multiclass",
            "search_lda_shrinkage",
            "search_gaussian_nb",
            "search_knn_multiclass",
            "search_svc_rbf",
            "search_svc_rbf_accuracy",
            "search_cumulative_logit_ordinal",
            "search_cumulative_logit_accuracy_ordinal",
            "search_extra_trees_ordinal_regressor",
            "search_random_forest_ordinal_regressor",
            "search_xgboost_multiclass",
        }
        specs = {name: spec for name, spec in specs.items() if name in keep}
    return specs


def _reg_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    out = {
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else np.nan,
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "explained_variance": float(explained_variance_score(y_true, y_pred)) if len(y_true) >= 2 else np.nan,
        "spearman_r": np.nan,
        "pearson_r": np.nan,
    }
    if len(y_true) >= 3 and len(np.unique(y_true)) > 1 and len(np.unique(y_pred)) > 1:
        out["spearman_r"] = float(stats.spearmanr(y_true, y_pred).statistic)
        out["pearson_r"] = float(stats.pearsonr(y_true, y_pred).statistic)
    return out


def _run_grid(pipe, grid, scoring: str, inner, x_train, y_train, n_jobs: int) -> GridSearchCV:
    search = GridSearchCV(
        pipe,
        grid,
        scoring=scoring,
        cv=inner,
        n_jobs=n_jobs,
        pre_dispatch=n_jobs,
        refit=True,
        error_score=np.nan,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        search.fit(x_train, y_train)
    return search


def _progress(enabled: bool, message: str) -> None:
    if enabled:
        print(message, flush=True)


def _first_fold_label(values: pd.Series) -> str:
    values = values.astype(str)
    mode = values.mode(dropna=True)
    if not mode.empty:
        return str(mode.iloc[0])
    return str(values.iloc[0]) if len(values) else "ensemble"


def _run_regression_search(
    matrix: pd.DataFrame,
    matches: pd.DataFrame,
    policies: dict[str, list[str]],
    fast: bool,
    n_jobs: int,
    include_history: bool,
    model_filter: set[str] | None = None,
    progress: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    pred_rows: list[dict] = []
    modes = {PRIMARY_MODE: "primary", HISTORY_MODE: "sensitivity"}
    if not include_history:
        modes = {PRIMARY_MODE: "primary"}
    target = SCORE_REGRESSION_TARGETS[0]
    for mode, role in modes.items():
        base = matches[matches["target"].astype(str).eq(target) & matches["target_mode"].astype(str).eq(mode)].copy()
        base["target_value"] = pd.to_numeric(base["target_value"], errors="coerce")
        base = base.dropna(subset=["target_value"])
        task = base.merge(matrix, on="subject_id", how="inner", suffixes=("", "_matrix"))
        raw_n = len(task)
        task, outlier_audit = _score_outlier_filter(task, target, mode)
        if len(task) < 40:
            continue
        split_labels = task["group"].astype(str)
        min_group = int(split_labels.value_counts().min())
        outer_splits = min(3 if fast else 5, min_group) if min_group >= 3 else 3
        outer = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=777)
        for policy_name, features in policies.items():
            x = task[features].apply(pd.to_numeric, errors="coerce")
            y = task["target_value"].to_numpy(dtype=float)
            specs = _regression_specs(len(features), len(task), fast)
            if model_filter:
                specs = {name: spec for name, spec in specs.items() if name in model_filter}
                if not specs:
                    continue
            inner_splits = min(3, max(2, outer_splits - 1))
            for model_name, (pipe, grid) in specs.items():
                _progress(
                    progress,
                    f"[regression] mode={mode} policy={policy_name} model={model_name} "
                    f"n={len(task)} features={len(features)} outer={outer_splits} inner={inner_splits}",
                )
                model_pred_rows: list[dict] = []
                best_scores = []
                params = []
                for fold, (train_idx, test_idx) in enumerate(outer.split(x, split_labels), start=1):
                    inner = KFold(n_splits=inner_splits, shuffle=True, random_state=900 + fold)
                    search = _run_grid(pipe, grid, "r2", inner, x.iloc[train_idx], y[train_idx], n_jobs)
                    pred = search.best_estimator_.predict(x.iloc[test_idx])
                    best_scores.append(float(search.best_score_) if np.isfinite(search.best_score_) else np.nan)
                    params.append(str(search.best_params_))
                    for pos, idx in enumerate(test_idx):
                        row = {
                            "task_type": "regression",
                            "target": target,
                            "target_mode": mode,
                            "analysis_role": role,
                            "feature_policy": policy_name,
                            "model": model_name,
                            "fold": fold,
                            "subject_id": task.iloc[idx]["subject_id"],
                            "group": task.iloc[idx]["group"],
                            "true_score": float(y[idx]),
                            "predicted_score": float(pred[pos]),
                            "residual": float(y[idx] - pred[pos]),
                        }
                        model_pred_rows.append(row)
                        pred_rows.append(row)
                pred_df = pd.DataFrame(model_pred_rows)
                metrics = _reg_metrics(pred_df["true_score"].to_numpy(dtype=float), pred_df["predicted_score"].to_numpy(dtype=float))
                _progress(
                    progress,
                    f"[regression done] mode={mode} policy={policy_name} model={model_name} "
                    f"r2={metrics['r2']:.4f} mae={metrics['mae']:.4f} rmse={metrics['rmse']:.4f}",
                )
                rows.append(
                    {
                        "task_type": "regression",
                        "target": target,
                        "target_mode": mode,
                        "analysis_role": role,
                        "feature_policy": policy_name,
                        "model": model_name,
                        "status": "ok",
                        "n_subjects": int(len(pred_df)),
                        "n_subjects_before_outlier_filter": int(raw_n),
                        "n_outliers_removed": int(raw_n - len(task)),
                        "n_features": int(len(features)),
                        "n_outer_folds": int(outer_splits),
                        "mean_inner_r2": float(np.nanmean(best_scores)),
                        "best_params_by_fold": " | ".join(params),
                        "goal_r2_0p70_met": bool(metrics["r2"] >= SEARCH_TARGET_R2),
                        **metrics,
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(pred_rows)


def _run_cdr_search(
    matrix: pd.DataFrame,
    matches: pd.DataFrame,
    policies: dict[str, list[str]],
    fast: bool,
    n_jobs: int,
    include_history: bool,
    model_filter: set[str] | None = None,
    progress: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    pred_rows: list[dict] = []
    modes = {PRIMARY_MODE: "primary", HISTORY_MODE: "sensitivity"}
    if not include_history:
        modes = {PRIMARY_MODE: "primary"}
    for mode, role in modes.items():
        base = matches[matches["target"].astype(str).eq(CDR_CLASSIFICATION_TARGET) & matches["target_mode"].astype(str).eq(mode)].copy()
        base["target_value"] = pd.to_numeric(base["target_value"], errors="coerce")
        base["cdr_class"] = base["target_value"].map(_cdr_class)
        base = base.dropna(subset=["cdr_class"])
        task = base.merge(matrix, on="subject_id", how="inner", suffixes=("", "_matrix"))
        if len(task) < 40:
            continue
        labels = [label for label in CDR_CLASS_ORDER if label in set(task["cdr_class"].astype(str))]
        label_to_int = {label: idx for idx, label in enumerate(labels)}
        y_all = task["cdr_class"].map(label_to_int).to_numpy(dtype=int)
        min_class = int(pd.Series(y_all).value_counts().min())
        outer_splits = min(3 if fast else 5, min_class) if min_class >= 3 else 3
        outer = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=888)
        for policy_name, features in policies.items():
            x = task[features].apply(pd.to_numeric, errors="coerce")
            specs = _classification_specs(len(features), len(task), len(labels), fast)
            if model_filter:
                specs = {name: spec for name, spec in specs.items() if name in model_filter}
                if not specs:
                    continue
            inner_splits = min(3, max(2, min_class - 1))
            for model_name, (pipe, grid) in specs.items():
                _progress(
                    progress,
                    f"[classification] mode={mode} policy={policy_name} model={model_name} "
                    f"n={len(task)} features={len(features)} classes={len(labels)} outer={outer_splits} inner={inner_splits}",
                )
                model_pred_rows: list[dict] = []
                best_scores = []
                params = []
                prob_rows = []
                for fold, (train_idx, test_idx) in enumerate(outer.split(x, y_all), start=1):
                    inner = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=910 + fold)
                    search = _run_grid(pipe, grid, "accuracy", inner, x.iloc[train_idx], y_all[train_idx], n_jobs)
                    best = search.best_estimator_
                    pred = best.predict(x.iloc[test_idx])
                    best_scores.append(float(search.best_score_) if np.isfinite(search.best_score_) else np.nan)
                    params.append(str(search.best_params_))
                    prob = np.full((len(test_idx), len(labels)), np.nan)
                    if hasattr(best, "predict_proba"):
                        try:
                            raw_prob = np.asarray(best.predict_proba(x.iloc[test_idx]), dtype=float)
                            classes = getattr(best.named_steps.get("model"), "classes_", list(range(len(labels))))
                            for col_idx, class_idx in enumerate(classes):
                                if int(class_idx) < len(labels):
                                    prob[:, int(class_idx)] = raw_prob[:, col_idx]
                        except Exception:
                            pass
                    for pos, idx in enumerate(test_idx):
                        row = {
                            "task_type": "classification",
                            "target": CDR_CLASSIFICATION_TARGET,
                            "target_mode": mode,
                            "analysis_role": role,
                            "feature_policy": policy_name,
                            "model": model_name,
                            "fold": fold,
                            "subject_id": task.iloc[idx]["subject_id"],
                            "group": task.iloc[idx]["group"],
                            "true_class": labels[int(y_all[idx])],
                            "predicted_class": labels[int(pred[pos])],
                            "true_cdr_value": float(task.iloc[idx]["target_value"]),
                        }
                        for class_idx, label in enumerate(labels):
                            row[f"prob_{label}"] = float(prob[pos, class_idx])
                        model_pred_rows.append(row)
                        pred_rows.append(row)
                pred_df = pd.DataFrame(model_pred_rows)
                y_true = pred_df["true_class"].map(label_to_int).to_numpy(dtype=int)
                y_pred = pred_df["predicted_class"].map(label_to_int).to_numpy(dtype=int)
                specificity = _specificity_by_class(y_true, y_pred, list(range(len(labels))))
                macro_specificity = float(np.nanmean(list(specificity.values())))
                prob_cols = [f"prob_{label}" for label in labels]
                auroc = np.nan
                if set(prob_cols).issubset(pred_df.columns):
                    prob = pred_df[prob_cols].to_numpy(dtype=float)
                    if np.isfinite(prob).all():
                        try:
                            auroc = float(roc_auc_score(y_true, prob, multi_class="ovr", average="macro"))
                        except Exception:
                            pass
                accuracy = float(accuracy_score(y_true, y_pred))
                _progress(
                    progress,
                    f"[classification done] mode={mode} policy={policy_name} model={model_name} "
                    f"accuracy={accuracy:.4f} balanced_accuracy={balanced_accuracy_score(y_true, y_pred):.4f} "
                    f"macro_f1={f1_score(y_true, y_pred, average='macro'):.4f}",
                )
                rows.append(
                    {
                        "task_type": "classification",
                        "target": CDR_CLASSIFICATION_TARGET,
                        "target_mode": mode,
                        "analysis_role": role,
                        "feature_policy": policy_name,
                        "model": model_name,
                        "status": "ok",
                        "n_subjects": int(len(pred_df)),
                        "n_classes": int(len(labels)),
                        "n_features": int(len(features)),
                        "n_outer_folds": int(outer_splits),
                        "mean_inner_accuracy": float(np.nanmean(best_scores)),
                        "best_params_by_fold": " | ".join(params),
                        "accuracy": accuracy,
                        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
                        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
                        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
                        "macro_recall_sensitivity": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
                        "macro_specificity": macro_specificity,
                        "macro_auroc_ovr": auroc,
                        "goal_accuracy_0p70_met": bool(accuracy >= SEARCH_TARGET_ACCURACY),
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(pred_rows)


def _append_oof_ensembles(perf: pd.DataFrame, pred: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if perf.empty or pred.empty:
        return perf, pred
    posthoc_model = perf["model"].astype(str).str.startswith("ensemble_") | perf["model"].astype(str).str.contains("__fold_prior_calibrated", regex=False)
    pred_model = pred["model"].astype(str).str.startswith("ensemble_") | pred["model"].astype(str).str.contains("__fold_prior_calibrated", regex=False)
    existing_derived_perf = perf[posthoc_model].copy()
    existing_derived_pred = pred[pred_model].copy()
    perf = perf[~posthoc_model].copy()
    pred = pred[~pred_model].copy()
    perf["_dedupe_source"] = "base"
    pred["_dedupe_source"] = "base"
    if not existing_derived_perf.empty:
        existing_derived_perf["_dedupe_source"] = "existing_derived"
    if not existing_derived_pred.empty:
        existing_derived_pred["_dedupe_source"] = "existing_derived"
    perf_rows: list[dict] = []
    pred_rows: list[dict] = []

    def _dedupe_best_rows(perf_df: pd.DataFrame, pred_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        if perf_df.empty:
            return perf_df.drop(columns=["_dedupe_source"], errors="ignore"), pred_df.drop(columns=["_dedupe_source"], errors="ignore")
        key_cols = [col for col in ("task_type", "target", "target_mode", "feature_policy", "model") if col in perf_df.columns]
        if not key_cols or "_dedupe_source" not in perf_df.columns:
            return perf_df.drop(columns=["_dedupe_source"], errors="ignore"), pred_df.drop(columns=["_dedupe_source"], errors="ignore")
        chunks = []
        for _, sub in perf_df.groupby(key_cols, dropna=False):
            task = str(sub["task_type"].iloc[0])
            metric = "r2" if task == "regression" else "accuracy"
            sub = sub.copy()
            sub[metric] = pd.to_numeric(sub.get(metric), errors="coerce")
            chunks.append(sub.sort_values(metric, ascending=False, na_position="last").iloc[[0]])
        selected = pd.concat(chunks, ignore_index=True, sort=False) if chunks else perf_df
        if pred_df.empty or "_dedupe_source" not in pred_df.columns:
            return selected.drop(columns=["_dedupe_source"], errors="ignore"), pred_df.drop(columns=["_dedupe_source"], errors="ignore")
        choice_cols = [*key_cols, "_dedupe_source"]
        choice = selected[choice_cols].drop_duplicates()
        pred_key_cols = [col for col in key_cols if col in pred_df.columns]
        merged_pred = pred_df.merge(choice, on=[*pred_key_cols, "_dedupe_source"], how="inner")
        pred_subject_keys = [col for col in ("task_type", "target", "target_mode", "feature_policy", "model", "subject_id") if col in merged_pred.columns]
        if pred_subject_keys:
            merged_pred = merged_pred.drop_duplicates(pred_subject_keys, keep="last")
        return selected.drop(columns=["_dedupe_source"], errors="ignore"), merged_pred.drop(columns=["_dedupe_source"], errors="ignore")

    # Age sensitivity rows are kept as explicit base-model artifacts, but they
    # must not be blended into the primary posthoc no-age ensembles.
    ensemble_perf = perf[~_is_explicit_age_policy(perf["feature_policy"])].copy() if "feature_policy" in perf.columns else perf.copy()
    ensemble_pred = pred[~_is_explicit_age_policy(pred["feature_policy"])].copy() if "feature_policy" in pred.columns else pred.copy()

    # Metadata sensitivity rows have not improved MMSE in current CV; keep
    # regression posthoc ensembles structural-only so they cannot dilute them.
    reg_ensemble_perf = (
        ensemble_perf[
            ~_is_non_age_metadata_policy(ensemble_perf["feature_policy"])
            & ~_is_disease_meta_policy(ensemble_perf["feature_policy"])
        ].copy()
        if "feature_policy" in ensemble_perf.columns
        else ensemble_perf.copy()
    )
    reg_ensemble_pred = (
        ensemble_pred[
            ~_is_non_age_metadata_policy(ensemble_pred["feature_policy"])
            & ~_is_disease_meta_policy(ensemble_pred["feature_policy"])
        ].copy()
        if "feature_policy" in ensemble_pred.columns
        else ensemble_pred.copy()
    )
    reg_perf = reg_ensemble_perf[reg_ensemble_perf["task_type"].astype(str).eq("regression")].copy()
    reg_pred = reg_ensemble_pred[reg_ensemble_pred["task_type"].astype(str).eq("regression")].copy()
    for mode, mode_perf in reg_perf.groupby("target_mode", dropna=False):
        mode_perf = mode_perf.sort_values("r2", ascending=False, na_position="last")
        for ensemble_name, n_take in (
            ("ensemble_oof_top2_mean_regression", 2),
            ("ensemble_oof_top3_mean_regression", 3),
            ("ensemble_oof_top5_mean_regression", 5),
            ("ensemble_oof_all_mean_regression", 999),
        ):
            selected = mode_perf.head(n_take)
            keys = set(zip(selected["feature_policy"].astype(str), selected["model"].astype(str)))
            sub = reg_pred[reg_pred["target_mode"].astype(str).eq(str(mode))].copy()
            sub = sub[[((str(row["feature_policy"]), str(row["model"])) in keys) for _, row in sub.iterrows()]]
            if sub.empty:
                continue
            if "fold" not in sub.columns:
                sub["fold"] = "ensemble"
            fold_index = ["subject_id", "group", "true_score"]
            fold_map = sub.groupby(fold_index, dropna=False)["fold"].agg(_first_fold_label)
            pivot = sub.pivot_table(
                index=fold_index,
                columns=["feature_policy", "model"],
                values="predicted_score",
                aggfunc="mean",
            )
            if pivot.empty:
                continue
            index_rows = pivot.index.to_frame(index=False)
            index_rows["fold"] = [fold_map.get(key, "ensemble") for key in pivot.index]
            true_values = pd.to_numeric(index_rows["true_score"], errors="coerce").to_numpy(dtype=float)
            raw_mean = pivot.mean(axis=1).to_numpy(dtype=float)
            raw_median = pivot.median(axis=1).to_numpy(dtype=float)
            variants: list[tuple[str, np.ndarray, str]] = [
                (ensemble_name, raw_mean, f"Fixed mean of top {min(n_take, len(mode_perf))} OOF base models by R2"),
                (
                    ensemble_name.replace("_mean_regression", "_median_regression"),
                    raw_median,
                    f"Fixed median of top {min(n_take, len(mode_perf))} OOF base models by R2",
                ),
            ]
            folds = index_rows["fold"].astype(str).to_numpy()
            if len(np.unique(folds)) >= 2 and len(index_rows) >= 20:
                calibrated = np.empty(len(index_rows), dtype=float)
                calibration_notes: list[str] = []
                for fold in np.unique(folds):
                    test_mask = folds == fold
                    train_mask = ~test_mask
                    if train_mask.sum() < 8 or len(np.unique(raw_mean[train_mask])) < 2:
                        calibrated[test_mask] = raw_mean[test_mask]
                        calibration_notes.append(f"fold={fold}:identity")
                        continue
                    calibrator = Ridge(alpha=1.0)
                    calibrator.fit(raw_mean[train_mask].reshape(-1, 1), true_values[train_mask])
                    calibrated[test_mask] = calibrator.predict(raw_mean[test_mask].reshape(-1, 1))
                    calibration_notes.append(
                        f"fold={fold}:ridge_slope={float(calibrator.coef_[0]):.3f},intercept={float(calibrator.intercept_):.3f}"
                    )
                variants.append(
                    (
                        ensemble_name.replace("_mean_regression", "_calibrated_mean_regression"),
                        calibrated,
                        f"Fold-safe ridge calibration of top {min(n_take, len(mode_perf))} OOF mean ensemble | {' | '.join(calibration_notes)}",
                    )
                )
                if pivot.shape[1] >= 2:
                    stacked = np.empty(len(index_rows), dtype=float)
                    stack_notes: list[str] = []
                    design = pivot.to_numpy(dtype=float)
                    for fold in np.unique(folds):
                        test_mask = folds == fold
                        train_mask = ~test_mask
                        if train_mask.sum() < max(8, pivot.shape[1] + 2):
                            stacked[test_mask] = raw_mean[test_mask]
                            stack_notes.append(f"fold={fold}:mean_fallback")
                            continue
                        stacker = Ridge(alpha=1.0)
                        stacker.fit(design[train_mask], true_values[train_mask])
                        stacked[test_mask] = stacker.predict(design[test_mask])
                        stack_notes.append(
                            f"fold={fold}:ridge_stack_n={pivot.shape[1]},intercept={float(stacker.intercept_):.3f}"
                        )
                    variants.append(
                        (
                            ensemble_name.replace("_mean_regression", "_stacked_ridge_regression"),
                            stacked,
                            f"Fold-safe ridge stack of top {min(n_take, len(mode_perf))} OOF base models | {' | '.join(stack_notes)}",
                        )
                    )

            for variant_name, pred_values, params_note in variants:
                pred_values = np.clip(np.asarray(pred_values, dtype=float), 0.0, MMSE_SCORE_MAX)
                metrics = _reg_metrics(true_values, pred_values)
                for row_idx, (_, row) in enumerate(index_rows.iterrows()):
                    pred_rows.append(
                        {
                            "task_type": "regression",
                            "target": SCORE_REGRESSION_TARGETS[0],
                            "target_mode": mode,
                            "analysis_role": "primary" if mode == PRIMARY_MODE else "sensitivity",
                            "feature_policy": "posthoc_oof_ensemble_exploratory",
                            "model": variant_name,
                            "fold": row["fold"],
                            "subject_id": row["subject_id"],
                            "group": row["group"],
                            "true_score": float(true_values[row_idx]),
                            "predicted_score": float(pred_values[row_idx]),
                            "residual": float(true_values[row_idx] - pred_values[row_idx]),
                        }
                    )
                perf_rows.append(
                    {
                        "task_type": "regression",
                        "target": SCORE_REGRESSION_TARGETS[0],
                        "target_mode": mode,
                        "analysis_role": "primary" if mode == PRIMARY_MODE else "sensitivity",
                        "feature_policy": "posthoc_oof_ensemble_exploratory",
                        "model": variant_name,
                        "status": "ok",
                        "n_subjects": int(len(index_rows)),
                        "n_subjects_before_outlier_filter": np.nan,
                        "n_outliers_removed": np.nan,
                        "n_features": int(len(keys)),
                        "n_outer_folds": np.nan,
                        "mean_inner_r2": np.nan,
                        "best_params_by_fold": params_note,
                        "goal_r2_0p70_met": bool(metrics["r2"] >= SEARCH_TARGET_R2),
                        **metrics,
                    }
                )

    cls_perf = ensemble_perf[ensemble_perf["task_type"].astype(str).eq("classification")].copy()
    cls_pred = ensemble_pred[ensemble_pred["task_type"].astype(str).eq("classification")].copy()
    prob_cols = [f"prob_{label}" for label in CDR_CLASS_ORDER]

    def _calibrated_class_predictions(agg: pd.DataFrame, labels: list[str]) -> tuple[list[str], np.ndarray, str]:
        available_prob_cols = [f"prob_{label}" for label in labels]
        raw_prob = agg[available_prob_cols].to_numpy(dtype=float)
        folds = agg["fold"].astype(str).to_numpy() if "fold" in agg.columns else np.full(len(agg), "all", dtype=object)
        true_labels = agg["true_class"].astype(str).to_numpy()
        pred_labels = np.empty(len(agg), dtype=object)
        bias_grid = [-1.2, -0.8, -0.4, 0.0, 0.4, 0.8, 1.2]
        candidates = np.asarray(np.meshgrid(*([bias_grid] * len(labels)))).T.reshape(-1, len(labels))
        candidate_notes: list[str] = []
        for fold in np.unique(folds):
            test_mask = folds == fold
            train_mask = ~test_mask
            if train_mask.sum() < max(8, len(labels) * 2):
                bias = np.zeros(len(labels), dtype=float)
            else:
                train_prob = raw_prob[train_mask]
                train_true = true_labels[train_mask]
                log_prob = np.log(np.clip(train_prob, 1e-8, 1.0))
                best_bias = np.zeros(len(labels), dtype=float)
                best_score = (-np.inf, -np.inf)
                for candidate in candidates:
                    train_pred = np.asarray(labels, dtype=object)[np.argmax(log_prob + candidate, axis=1)]
                    accuracy = float(np.mean(train_true == train_pred))
                    if accuracy < best_score[0]:
                        continue
                    balanced = float(balanced_accuracy_score(train_true, train_pred))
                    score = (accuracy, balanced)
                    if score > best_score:
                        best_score = score
                        best_bias = candidate
                bias = best_bias
            pred_labels[test_mask] = np.asarray(labels, dtype=object)[np.argmax(np.log(np.clip(raw_prob[test_mask], 1e-8, 1.0)) + bias, axis=1)]
            candidate_notes.append(f"fold={fold}:bias={','.join(f'{value:.1f}' for value in bias)}")
        return pred_labels.tolist(), raw_prob, " | ".join(candidate_notes)

    base_calibrated_rows: list[dict] = []
    base_calibrated_pred_rows: list[dict] = []
    base_groups = [c for c in ("target_mode", "feature_policy", "model") if c in cls_pred.columns]
    base_calibration_keys: set[tuple[str, str, str]] = set()
    if not cls_perf.empty and {"target_mode", "feature_policy", "model", "accuracy"}.issubset(cls_perf.columns):
        ranked_base = cls_perf[
            ~cls_perf["model"].astype(str).str.startswith("ensemble_")
            & ~cls_perf["model"].astype(str).str.contains("__fold_prior_calibrated", regex=False)
        ].copy()
        ranked_base["accuracy"] = pd.to_numeric(ranked_base["accuracy"], errors="coerce")
        for _, mode_perf in ranked_base.sort_values("accuracy", ascending=False, na_position="last").groupby("target_mode", dropna=False):
            for _, row in mode_perf.head(12).iterrows():
                base_calibration_keys.add((str(row["target_mode"]), str(row["feature_policy"]), str(row["model"])))
    if base_groups:
        for (mode, feature_policy, model_name), sub in cls_pred.groupby(base_groups, dropna=False):
            if str(model_name).startswith("ensemble_") or "__fold_prior_calibrated" in str(model_name):
                continue
            if (str(mode), str(feature_policy), str(model_name)) not in base_calibration_keys:
                continue
            labels = [label for label in CDR_CLASS_ORDER if f"prob_{label}" in sub.columns]
            if len(labels) < 2:
                continue
            available_prob_cols = [f"prob_{label}" for label in labels]
            prob_values = sub[available_prob_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
            if sub.empty or not np.isfinite(prob_values).all() or "true_class" not in sub.columns:
                continue
            agg_cols = ["subject_id", "group", "true_class", "true_cdr_value", "fold", *available_prob_cols]
            agg = sub[[c for c in agg_cols if c in sub.columns]].copy()
            if "fold" not in agg.columns:
                agg["fold"] = "all"
            pred_labels, prob, calibration_note = _calibrated_class_predictions(agg, labels)
            label_to_int = {label: idx for idx, label in enumerate(labels)}
            y_true = agg["true_class"].astype(str).map(label_to_int).to_numpy(dtype=int)
            y_pred = np.asarray([label_to_int[label] for label in pred_labels], dtype=int)
            specificity = _specificity_by_class(y_true, y_pred, list(range(len(labels))))
            macro_specificity = float(np.nanmean(list(specificity.values())))
            auroc = np.nan
            if np.isfinite(prob).all():
                try:
                    auroc = float(roc_auc_score(y_true, prob, multi_class="ovr", average="macro"))
                except Exception:
                    pass
            calibrated_model_name = f"{model_name}__fold_prior_calibrated"
            for row_idx, (_, row) in enumerate(agg.iterrows()):
                out = {
                    "task_type": "classification",
                    "target": CDR_CLASSIFICATION_TARGET,
                    "target_mode": mode,
                    "analysis_role": "primary" if mode == PRIMARY_MODE else "sensitivity",
                    "feature_policy": feature_policy,
                    "model": calibrated_model_name,
                    "fold": row["fold"],
                    "subject_id": row["subject_id"],
                    "group": row["group"],
                    "true_class": row["true_class"],
                    "predicted_class": pred_labels[row_idx],
                    "true_cdr_value": float(row["true_cdr_value"]),
                }
                for label in labels:
                    out[f"prob_{label}"] = float(row[f"prob_{label}"])
                base_calibrated_pred_rows.append(out)
            base_calibrated_rows.append(
                {
                    "task_type": "classification",
                    "target": CDR_CLASSIFICATION_TARGET,
                    "target_mode": mode,
                    "analysis_role": "primary" if mode == PRIMARY_MODE else "sensitivity",
                    "feature_policy": feature_policy,
                    "model": calibrated_model_name,
                    "status": "ok",
                    "n_subjects": int(len(agg)),
                    "n_classes": int(len(labels)),
                    "n_features": np.nan,
                    "n_outer_folds": np.nan,
                    "mean_inner_accuracy": np.nan,
                    "best_params_by_fold": f"Fold-safe class-prior calibration of base OOF model | {calibration_note}",
                    "accuracy": float(accuracy_score(y_true, y_pred)),
                    "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
                    "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
                    "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
                    "macro_recall_sensitivity": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
                    "macro_specificity": macro_specificity,
                    "macro_auroc_ovr": auroc,
                    "goal_accuracy_0p70_met": bool(float(accuracy_score(y_true, y_pred)) >= SEARCH_TARGET_ACCURACY),
                }
            )

    for mode, mode_perf in cls_perf.groupby("target_mode", dropna=False):
        labels = [label for label in CDR_CLASS_ORDER if f"prob_{label}" in cls_pred.columns]
        label_to_int = {label: idx for idx, label in enumerate(labels)}
        rank_sets: list[tuple[str, str, pd.DataFrame]] = []
        for rank_metric, name_fragment in (("accuracy", ""), ("balanced_accuracy", "balanced_"), ("macro_f1", "f1_")):
            if rank_metric in mode_perf.columns:
                rank_sets.append((rank_metric, name_fragment, mode_perf.sort_values(rank_metric, ascending=False, na_position="last")))
        for rank_metric, name_fragment, ranked_perf in rank_sets:
            for base_ensemble_name, n_take in (
                ("ensemble_oof_top2_prob_classification", 2),
                ("ensemble_oof_top3_prob_classification", 3),
                ("ensemble_oof_top5_prob_classification", 5),
                ("ensemble_oof_all_prob_classification", 999),
            ):
                ensemble_name = (
                    base_ensemble_name
                    if not name_fragment
                    else base_ensemble_name.replace("ensemble_oof_", f"ensemble_oof_{name_fragment}")
                )
                selected = ranked_perf.head(n_take)
                keys = set(zip(selected["feature_policy"].astype(str), selected["model"].astype(str)))
                sub = cls_pred[cls_pred["target_mode"].astype(str).eq(str(mode))].copy()
                sub = sub[[((str(row["feature_policy"]), str(row["model"])) in keys) for _, row in sub.iterrows()]]
                if sub.empty or not labels:
                    continue
                group_cols = ["subject_id", "group", "true_class", "true_cdr_value", "fold"]
                if "fold" not in sub.columns:
                    sub["fold"] = "all"
                base_group_cols = ["subject_id", "group", "true_class", "true_cdr_value"]
                fold_map = sub.groupby(base_group_cols, dropna=False)["fold"].agg(_first_fold_label)
                agg = sub.groupby(base_group_cols, as_index=False)[[f"prob_{label}" for label in labels]].mean()
                if not agg.empty:
                    agg["fold"] = [
                        fold_map.get((row["subject_id"], row["group"], row["true_class"], row["true_cdr_value"]), "all")
                        for _, row in agg.iterrows()
                    ]
                if agg.empty:
                    continue
                for calibrated in (False, True):
                    if calibrated:
                        pred_labels, prob, calibration_note = _calibrated_class_predictions(agg, labels)
                        model_name = ensemble_name.replace("_prob_classification", "_calibrated_prob_classification")
                        params_note = f"Fold-safe class-prior calibration of top {min(n_take, len(ranked_perf))} OOF base models ranked by {rank_metric} | {calibration_note}"
                    else:
                        prob = agg[[f"prob_{label}" for label in labels]].to_numpy(dtype=float)
                        pred_idx = np.nanargmax(prob, axis=1)
                        pred_labels = [labels[int(idx)] for idx in pred_idx]
                        model_name = ensemble_name
                        params_note = f"Fixed probability mean of top {min(n_take, len(ranked_perf))} OOF base models ranked by {rank_metric}"
                    y_true = agg["true_class"].astype(str).map(label_to_int).to_numpy(dtype=int)
                    y_pred = np.asarray([label_to_int[label] for label in pred_labels], dtype=int)
                    specificity = _specificity_by_class(y_true, y_pred, list(range(len(labels))))
                    macro_specificity = float(np.nanmean(list(specificity.values())))
                    auroc = np.nan
                    if np.isfinite(prob).all():
                        try:
                            auroc = float(roc_auc_score(y_true, prob, multi_class="ovr", average="macro"))
                        except Exception:
                            pass
                    accuracy = float(accuracy_score(y_true, y_pred))
                    for row_idx, (_, row) in enumerate(agg.iterrows()):
                        out = {
                            "task_type": "classification",
                            "target": CDR_CLASSIFICATION_TARGET,
                            "target_mode": mode,
                            "analysis_role": "primary" if mode == PRIMARY_MODE else "sensitivity",
                            "feature_policy": "posthoc_oof_ensemble_exploratory",
                            "model": model_name,
                            "fold": row["fold"],
                            "subject_id": row["subject_id"],
                            "group": row["group"],
                            "true_class": row["true_class"],
                            "predicted_class": pred_labels[row_idx],
                            "true_cdr_value": float(row["true_cdr_value"]),
                        }
                        for label in labels:
                            out[f"prob_{label}"] = float(row[f"prob_{label}"])
                        pred_rows.append(out)
                    perf_rows.append(
                        {
                            "task_type": "classification",
                            "target": CDR_CLASSIFICATION_TARGET,
                            "target_mode": mode,
                            "analysis_role": "primary" if mode == PRIMARY_MODE else "sensitivity",
                            "feature_policy": "posthoc_oof_ensemble_exploratory",
                            "model": model_name,
                            "status": "ok",
                            "n_subjects": int(len(agg)),
                            "n_classes": int(len(labels)),
                            "n_features": int(len(keys)),
                            "n_outer_folds": np.nan,
                            "mean_inner_accuracy": np.nan,
                            "best_params_by_fold": params_note,
                            "accuracy": accuracy,
                            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
                            "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
                            "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
                            "macro_recall_sensitivity": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
                            "macro_specificity": macro_specificity,
                            "macro_auroc_ovr": auroc,
                            "goal_accuracy_0p70_met": bool(accuracy >= SEARCH_TARGET_ACCURACY),
                        }
                    )

    derived_perf_frames = []
    derived_pred_frames = []
    if perf_rows:
        new_perf = pd.DataFrame(perf_rows)
        new_perf["_dedupe_source"] = "new_derived"
        derived_perf_frames.append(new_perf)
    if pred_rows:
        new_pred = pd.DataFrame(pred_rows)
        new_pred["_dedupe_source"] = "new_derived"
        derived_pred_frames.append(new_pred)
    if base_calibrated_rows:
        new_base_perf = pd.DataFrame(base_calibrated_rows)
        new_base_perf["_dedupe_source"] = "new_derived"
        derived_perf_frames.append(new_base_perf)
    if base_calibrated_pred_rows:
        new_base_pred = pd.DataFrame(base_calibrated_pred_rows)
        new_base_pred["_dedupe_source"] = "new_derived"
        derived_pred_frames.append(new_base_pred)
    perf = pd.concat([df for df in (perf, existing_derived_perf, *derived_perf_frames) if df is not None and not df.empty], ignore_index=True, sort=False)
    pred = pd.concat([df for df in (pred, existing_derived_pred, *derived_pred_frames) if df is not None and not df.empty], ignore_index=True, sort=False)
    return _dedupe_best_rows(perf, pred)


def _write_ceiling_audit(out_dir: Path, matches: pd.DataFrame, perf: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    active = matches[matches["target"].astype(str).isin([SCORE_REGRESSION_TARGETS[0], CDR_CLASSIFICATION_TARGET])].copy()
    active["target_value_num"] = pd.to_numeric(active["target_value"], errors="coerce")

    for target, target_df in active.groupby("target", dropna=False):
        for mode, mode_df in target_df.groupby("target_mode", dropna=False):
            vals = pd.to_numeric(mode_df["target_value"], errors="coerce").dropna()
            if vals.empty:
                continue
            row = {
                "target": target,
                "target_mode": mode,
                "audit_type": "target_distribution_and_null_baseline",
                "n_subjects": int(mode_df["subject_id"].nunique()),
                "group_counts": ";".join(f"{group}:{int(n)}" for group, n in mode_df.groupby("group")["subject_id"].nunique().items()),
            }
            if target == SCORE_REGRESSION_TARGETS[0]:
                median = float(vals.median())
                mean = float(vals.mean())
                group_medians = mode_df.groupby("group")["target_value_num"].median()
                group_pred = mode_df["group"].map(group_medians).to_numpy(dtype=float)
                y = mode_df["target_value_num"].to_numpy(dtype=float)
                row.update(
                    {
                        "target_type": "regression",
                        "target_min": float(vals.min()),
                        "target_median": median,
                        "target_mean": mean,
                        "target_max": float(vals.max()),
                        "target_sd": float(vals.std()),
                        "median_baseline_r2": float(r2_score(vals, np.full(len(vals), median))) if len(vals) >= 2 else np.nan,
                        "median_baseline_mae": float(mean_absolute_error(vals, np.full(len(vals), median))),
                        "mean_baseline_r2": float(r2_score(vals, np.full(len(vals), mean))) if len(vals) >= 2 else np.nan,
                        "mean_baseline_mae": float(mean_absolute_error(vals, np.full(len(vals), mean))),
                        "diagnosis_group_apparent_r2_not_model": float(r2_score(y, group_pred)) if len(y) >= 2 else np.nan,
                        "diagnosis_group_apparent_mae_not_model": float(mean_absolute_error(y, group_pred)),
                    }
                )
            else:
                classes = vals.map(_cdr_class).dropna().astype(str)
                counts = classes.value_counts().sort_index()
                majority = str(counts.idxmax())
                y_majority = np.full(len(classes), majority, dtype=object)
                group_majority = mode_df.assign(cdr_class=mode_df["target_value_num"].map(_cdr_class)).dropna(subset=["cdr_class"])
                group_map = group_majority.groupby("group")["cdr_class"].agg(lambda s: str(s.value_counts().idxmax()))
                y_group_true = group_majority["cdr_class"].astype(str).to_numpy()
                y_group_pred = group_majority["group"].map(group_map).astype(str).to_numpy()
                row.update(
                    {
                        "target_type": "classification",
                        "class_counts": ";".join(f"{key}:{int(value)}" for key, value in counts.items()),
                        "majority_class": majority,
                        "majority_accuracy": float(accuracy_score(classes, y_majority)),
                        "majority_balanced_accuracy": float(balanced_accuracy_score(classes, y_majority)),
                        "majority_macro_f1": float(f1_score(classes, y_majority, average="macro", zero_division=0)),
                        "diagnosis_group_apparent_accuracy_not_model": float(accuracy_score(y_group_true, y_group_pred)),
                        "diagnosis_group_apparent_balanced_accuracy_not_model": float(balanced_accuracy_score(y_group_true, y_group_pred)),
                        "diagnosis_group_apparent_macro_f1_not_model": float(f1_score(y_group_true, y_group_pred, average="macro", zero_division=0)),
                    }
                )
            rows.append(row)

    if not perf.empty:
        reg_perf = perf[perf["task_type"].astype(str).eq("regression")].copy()
        if not reg_perf.empty:
            reg_perf["r2"] = pd.to_numeric(reg_perf["r2"], errors="coerce")
            for mode, mode_perf in reg_perf.groupby("target_mode", dropna=False):
                best = mode_perf.sort_values("r2", ascending=False, na_position="last").iloc[0]
                rows.append(
                    {
                        "target": SCORE_REGRESSION_TARGETS[0],
                        "target_mode": mode,
                        "audit_type": "best_current_model",
                        "target_type": "regression",
                        "model": best.get("model"),
                        "feature_policy": best.get("feature_policy"),
                        "n_subjects": best.get("n_subjects"),
                        "best_r2": best.get("r2"),
                        "best_mae": best.get("mae"),
                        "best_rmse": best.get("rmse"),
                        "goal_met": bool(pd.notna(best.get("r2")) and best.get("r2") >= SEARCH_TARGET_R2),
                    }
                )
        cdr_perf = perf[perf["task_type"].astype(str).eq("classification")].copy()
        if not cdr_perf.empty:
            cdr_perf["accuracy"] = pd.to_numeric(cdr_perf["accuracy"], errors="coerce")
            for mode, mode_perf in cdr_perf.groupby("target_mode", dropna=False):
                best = mode_perf.sort_values("accuracy", ascending=False, na_position="last").iloc[0]
                rows.append(
                    {
                        "target": CDR_CLASSIFICATION_TARGET,
                        "target_mode": mode,
                        "audit_type": "best_current_model",
                        "target_type": "classification",
                        "model": best.get("model"),
                        "feature_policy": best.get("feature_policy"),
                        "n_subjects": best.get("n_subjects"),
                        "best_accuracy": best.get("accuracy"),
                        "best_balanced_accuracy": best.get("balanced_accuracy"),
                        "best_macro_f1": best.get("macro_f1"),
                        "goal_met": bool(pd.notna(best.get("accuracy")) and best.get("accuracy") >= SEARCH_TARGET_ACCURACY),
                    }
                )

    for target, target_df in active.groupby("target", dropna=False):
        pivot = target_df.pivot_table(index="subject_id", columns="target_mode", values="target_value_num", aggfunc="first")
        if {PRIMARY_MODE, HISTORY_MODE}.issubset(pivot.columns):
            both = pivot.dropna(subset=[PRIMARY_MODE, HISTORY_MODE]).copy()
            if both.empty:
                continue
            row = {
                "target": target,
                "target_mode": "primary_vs_history",
                "audit_type": "target_stability",
                "n_subjects": int(len(both)),
                "mean_abs_delta": float((both[PRIMARY_MODE] - both[HISTORY_MODE]).abs().mean()),
            }
            if target == CDR_CLASSIFICATION_TARGET:
                primary_class = both[PRIMARY_MODE].map(_cdr_class).astype(str)
                history_class = both[HISTORY_MODE].map(_cdr_class).astype(str)
                row["exact_class_agreement"] = float((primary_class == history_class).mean())
            else:
                row["pearson_r"] = float(both[[PRIMARY_MODE, HISTORY_MODE]].corr().iloc[0, 1])
            rows.append(row)

    audit = pd.DataFrame(rows)
    audit.to_csv(out_dir / "clinical_outcome_model_ceiling_audit.csv", index=False)
    return audit


def _timing_bin(days: object) -> str:
    value = pd.to_numeric(pd.Series([days]), errors="coerce").iloc[0]
    if pd.isna(value):
        return "unknown"
    if value == 0:
        return "exact_0d"
    if value <= 30:
        return "within_30d"
    if value <= 90:
        return "within_90d"
    if value <= 180:
        return "within_180d"
    if value <= 365:
        return "within_365d"
    if value <= 730:
        return "within_730d"
    return "over_730d"


def _compact_counts(series: pd.Series) -> str:
    counts = series.astype(str).value_counts(dropna=False).sort_index()
    return ";".join(f"{key}:{int(value)}" for key, value in counts.items())


def _eta_squared_by_class(x: pd.Series, y: pd.Series) -> float:
    work = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "y": y.astype(str)}).dropna(subset=["x", "y"])
    if work.empty or work["y"].nunique() < 2:
        return np.nan
    grand = float(work["x"].mean())
    ss_between = 0.0
    ss_total = float(((work["x"] - grand) ** 2).sum())
    if ss_total <= 0:
        return np.nan
    for _, sub in work.groupby("y"):
        ss_between += float(len(sub) * ((sub["x"].mean() - grand) ** 2))
    return float(ss_between / ss_total)


def _write_signal_audit(
    out_dir: Path,
    matrix: pd.DataFrame,
    catalog: pd.DataFrame,
    matches: pd.DataFrame,
    perf: pd.DataFrame,
    pred: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Write diagnostics that explain target/feature limits without changing modeling."""
    out_dir.mkdir(parents=True, exist_ok=True)
    policies = _feature_policies(matrix, catalog)
    allowed_features = list(dict.fromkeys(policies.get("all_structural_no_age", [])))
    catalog_idx = catalog.drop_duplicates("feature").set_index("feature") if "feature" in catalog.columns else pd.DataFrame()

    target_rows: list[dict] = []
    active_targets = [SCORE_REGRESSION_TARGETS[0], CDR_CLASSIFICATION_TARGET]
    active = matches[matches["target"].astype(str).isin(active_targets)].copy()
    active["target_value_num"] = pd.to_numeric(active["target_value"], errors="coerce")
    active["timing_bin"] = active["abs_day_delta"].map(_timing_bin) if "abs_day_delta" in active.columns else "unknown"
    for (target, mode), sub in active.groupby(["target", "target_mode"], dropna=False):
        values = pd.to_numeric(sub["target_value"], errors="coerce").dropna()
        row = {
            "target": target,
            "target_mode": mode,
            "audit_scope": "target_distribution",
            "n_subjects": int(sub["subject_id"].nunique()),
            "group_counts": _compact_counts(sub["group"]),
            "timing_counts": _compact_counts(sub["timing_bin"]),
            "mean_abs_day_delta": float(pd.to_numeric(sub.get("abs_day_delta"), errors="coerce").mean()) if "abs_day_delta" in sub.columns else np.nan,
            "median_abs_day_delta": float(pd.to_numeric(sub.get("abs_day_delta"), errors="coerce").median()) if "abs_day_delta" in sub.columns else np.nan,
        }
        if target == SCORE_REGRESSION_TARGETS[0] and not values.empty:
            row.update(
                {
                    "target_type": "regression",
                    "target_min": float(values.min()),
                    "target_median": float(values.median()),
                    "target_mean": float(values.mean()),
                    "target_max": float(values.max()),
                    "target_sd": float(values.std()),
                    "target_unique_values": int(values.nunique()),
                    "ceiling_warning": "MMSE has a strong upper-bound pileup near 30; scan timing and ceiling effects can limit R2.",
                }
            )
        elif target == CDR_CLASSIFICATION_TARGET and not values.empty:
            classes = values.map(_cdr_class).dropna().astype(str)
            row.update(
                {
                    "target_type": "classification",
                    "class_counts": _compact_counts(classes),
                    "majority_class": str(classes.value_counts().idxmax()) if not classes.empty else "",
                    "majority_accuracy": float(classes.value_counts(normalize=True).max()) if not classes.empty else np.nan,
                    "ceiling_warning": "CDR is imbalanced and ordinal; exact multiclass accuracy is capped by rare class support unless class signal is strong.",
                }
            )
        target_rows.append(row)

        for timing_bin, timing_sub in sub.groupby("timing_bin", dropna=False):
            vals = pd.to_numeric(timing_sub["target_value"], errors="coerce").dropna()
            if vals.empty:
                continue
            timing_row = {
                "target": target,
                "target_mode": mode,
                "audit_scope": "target_by_timing_bin",
                "timing_bin": timing_bin,
                "n_subjects": int(timing_sub["subject_id"].nunique()),
                "group_counts": _compact_counts(timing_sub["group"]),
            }
            if target == SCORE_REGRESSION_TARGETS[0]:
                timing_row.update({"target_mean": float(vals.mean()), "target_sd": float(vals.std()), "target_median": float(vals.median())})
            else:
                timing_row.update({"class_counts": _compact_counts(vals.map(_cdr_class).dropna().astype(str))})
            target_rows.append(timing_row)

    target_audit = pd.DataFrame(target_rows)
    target_audit.to_csv(out_dir / "clinical_outcome_signal_target_audit.csv", index=False)

    feature_rows: list[dict] = []
    for target in active_targets:
        for mode in [PRIMARY_MODE, HISTORY_MODE]:
            base = matches[matches["target"].astype(str).eq(target) & matches["target_mode"].astype(str).eq(mode)].copy()
            if base.empty:
                continue
            base["target_value"] = pd.to_numeric(base["target_value"], errors="coerce")
            base = base.dropna(subset=["target_value"])
            task = base.merge(matrix, on="subject_id", how="inner", suffixes=("", "_matrix"))
            if target == SCORE_REGRESSION_TARGETS[0]:
                task, _ = _score_outlier_filter(task, target, mode)
                y_numeric = pd.to_numeric(task["target_value"], errors="coerce")
                y_label = y_numeric.astype(str)
            else:
                task["cdr_class"] = task["target_value"].map(_cdr_class)
                task = task.dropna(subset=["cdr_class"])
                cdr_order = {label: idx for idx, label in enumerate(CDR_CLASS_ORDER)}
                y_numeric = task["cdr_class"].map(cdr_order).astype(float)
                y_label = task["cdr_class"].astype(str)
            if len(task) < 20:
                continue
            for feature in allowed_features:
                if feature not in task.columns:
                    continue
                x = pd.to_numeric(task[feature], errors="coerce")
                valid = x.notna() & y_numeric.notna()
                if int(valid.sum()) < 20 or x[valid].nunique() < 2 or y_numeric[valid].nunique() < 2:
                    continue
                try:
                    spearman = float(stats.spearmanr(x[valid], y_numeric[valid]).statistic)
                except Exception:
                    spearman = np.nan
                row = {
                    "target": target,
                    "target_mode": mode,
                    "feature": feature,
                    "feature_group": catalog_idx.at[feature, "feature_group"] if not catalog_idx.empty and feature in catalog_idx.index else "",
                    "n_subjects": int(valid.sum()),
                    "feature_missingness_in_target_cohort": float(1.0 - valid.mean()),
                    "spearman_r": spearman,
                    "abs_spearman_r": abs(spearman) if pd.notna(spearman) else np.nan,
                    "selected_fold_count": catalog_idx.at[feature, "selected_fold_count"] if not catalog_idx.empty and feature in catalog_idx.index and "selected_fold_count" in catalog_idx.columns else np.nan,
                }
                if target == CDR_CLASSIFICATION_TARGET:
                    row["eta_squared_by_cdr_class"] = _eta_squared_by_class(x[valid], y_label[valid])
                feature_rows.append(row)
    feature_audit = pd.DataFrame(feature_rows)
    if not feature_audit.empty:
        sort_cols = ["target", "target_mode", "abs_spearman_r"]
        feature_audit = feature_audit.sort_values(sort_cols, ascending=[True, True, False], na_position="last")
        feature_audit = feature_audit.groupby(["target", "target_mode"], group_keys=False).head(100)
    feature_audit.to_csv(out_dir / "clinical_outcome_signal_feature_audit.csv", index=False)

    prediction_rows: list[dict] = []
    if perf is not None and not perf.empty and pred is not None and not pred.empty:
        for task_type, metric in (("regression", "r2"), ("classification", "accuracy")):
            task_perf = perf[perf["task_type"].astype(str).eq(task_type)].copy()
            if task_perf.empty or metric not in task_perf.columns:
                continue
            task_perf[metric] = pd.to_numeric(task_perf[metric], errors="coerce")
            for _, best in task_perf.sort_values(metric, ascending=False, na_position="last").groupby("target", dropna=False).head(1).iterrows():
                target = str(best.get("target"))
                mode = str(best.get("target_mode"))
                feature_policy = str(best.get("feature_policy"))
                model = str(best.get("model"))
                sub = pred[
                    pred["target"].astype(str).eq(target)
                    & pred["target_mode"].astype(str).eq(mode)
                    & pred["feature_policy"].astype(str).eq(feature_policy)
                    & pred["model"].astype(str).eq(model)
                ].copy()
                if sub.empty:
                    continue
                match_cols = ["subject_id", "target", "target_mode", "abs_day_delta", "clinical_source"]
                sub = sub.merge(matches[[c for c in match_cols if c in matches.columns]].drop_duplicates(), on=["subject_id", "target", "target_mode"], how="left")
                sub["timing_bin"] = sub["abs_day_delta"].map(_timing_bin) if "abs_day_delta" in sub.columns else "unknown"
                for scope, group_cols in {
                    "overall": [],
                    "by_diagnosis_group": ["group"],
                    "by_timing_bin": ["timing_bin"],
                    "by_group_and_timing_bin": ["group", "timing_bin"],
                }.items():
                    grouped = [((), sub)] if not group_cols else sub.groupby(group_cols, dropna=False)
                    for key, part in grouped:
                        if len(part) < 3:
                            continue
                        row = {
                            "target": target,
                            "target_mode": mode,
                            "feature_policy": feature_policy,
                            "model": model,
                            "task_type": task_type,
                            "audit_scope": scope,
                            "n_subjects": int(part["subject_id"].nunique()),
                        }
                        if group_cols:
                            key_values = key if isinstance(key, tuple) else (key,)
                            for col, value in zip(group_cols, key_values):
                                row[col] = value
                        if task_type == "regression" and {"true_score", "predicted_score"}.issubset(part.columns):
                            y_true = pd.to_numeric(part["true_score"], errors="coerce").to_numpy(dtype=float)
                            y_pred = pd.to_numeric(part["predicted_score"], errors="coerce").to_numpy(dtype=float)
                            if len(y_true) >= 2 and len(np.unique(y_true)) > 1:
                                row["r2"] = float(r2_score(y_true, y_pred))
                            row["mae"] = float(mean_absolute_error(y_true, y_pred))
                            row["rmse"] = float(math.sqrt(mean_squared_error(y_true, y_pred)))
                        elif task_type == "classification" and {"true_class", "predicted_class"}.issubset(part.columns):
                            y_true = part["true_class"].astype(str).to_numpy()
                            y_pred = part["predicted_class"].astype(str).to_numpy()
                            row["accuracy"] = float(accuracy_score(y_true, y_pred))
                            row["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
                            row["macro_f1"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
                            row["true_class_counts"] = _compact_counts(part["true_class"])
                            row["predicted_class_counts"] = _compact_counts(part["predicted_class"])
                        prediction_rows.append(row)
    prediction_audit = pd.DataFrame(prediction_rows)
    prediction_audit.to_csv(out_dir / "clinical_outcome_signal_prediction_audit.csv", index=False)

    notes = [
        "# Clinical outcome signal audit",
        "",
        "Generated from current no-age structural model artifacts.",
        "",
        "- `clinical_outcome_signal_target_audit.csv`: target timing, class balance, and distribution warnings.",
        "- `clinical_outcome_signal_feature_audit.csv`: top no-age structural features by univariate target association.",
        "- `clinical_outcome_signal_prediction_audit.csv`: best-model performance broken down by timing and diagnostic group.",
        "",
        "Interpretation: these audits are diagnostic only. They do not change targets, dashboard layout, predictors, or model promotion.",
    ]
    (out_dir / "clinical_outcome_signal_audit.md").write_text("\n".join(notes) + "\n")
    return {"target_audit": target_audit, "feature_audit": feature_audit, "prediction_audit": prediction_audit}


def _safe_read_optional_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _merge_existing_outputs(
    out_dir: Path,
    perf: pd.DataFrame,
    pred: pd.DataFrame,
    policy_rows: pd.DataFrame,
    meta_audit: pd.DataFrame,
    allowed_policies: set[str],
    include_age: bool,
    include_disease_meta: bool,
    include_non_age_metadata: bool,
    *,
    progress: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    existing_perf = _safe_read_optional_csv(out_dir / "clinical_outcome_model_search_performance.csv")
    existing_pred = _safe_read_optional_csv(out_dir / "clinical_outcome_model_search_cv_predictions.csv")
    existing_policies = _safe_read_optional_csv(out_dir / "clinical_outcome_model_search_feature_policies.csv")
    existing_meta = _safe_read_optional_csv(out_dir / "clinical_outcome_model_search_meta_feature_audit.csv")
    allowed_policies = {str(policy) for policy in allowed_policies}

    def _drop_derived_rows(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        out = df.copy()
        if "model" in out.columns:
            out = out[~out["model"].astype(str).str.startswith("ensemble_")].copy()
            out = out[~out["model"].astype(str).str.contains("__fold_prior_calibrated", regex=False)].copy()
        return out

    def _scope_new_base_rows(df: pd.DataFrame) -> pd.DataFrame:
        out = _drop_derived_rows(df)
        if "feature_policy" in out.columns:
            out = out[out["feature_policy"].astype(str).isin(allowed_policies)].copy()
        return out

    # Keep all existing base rows. A focused sensitivity run may pass only a
    # few policies, but it should replace only matching rows, not erase prior
    # no-age/meta searches that are needed for posthoc ensemble recomputation.
    existing_perf = _drop_derived_rows(existing_perf)
    existing_pred = _drop_derived_rows(existing_pred)
    perf = _scope_new_base_rows(perf)
    pred = _scope_new_base_rows(pred)

    if existing_perf.empty:
        return perf, pred, policy_rows, meta_audit

    perf_frames = []
    existing_perf = existing_perf.copy()
    existing_perf["_source"] = "existing"
    perf_frames.append(existing_perf)
    if not perf.empty:
        new_perf = perf.copy()
        new_perf["_source"] = "new"
        perf_frames.append(new_perf)
    all_perf = pd.concat(perf_frames, ignore_index=True, sort=False)
    perf_keys = [c for c in ("task_type", "target", "target_mode", "feature_policy", "model") if c in all_perf.columns]
    chosen_chunks = []
    for _, sub in all_perf.groupby(perf_keys, dropna=False):
        task = str(sub["task_type"].iloc[0])
        metric = "r2" if task == "regression" else "accuracy"
        sub = sub.copy()
        sub[metric] = pd.to_numeric(sub.get(metric), errors="coerce")
        chosen_chunks.append(sub.sort_values(metric, ascending=False, na_position="last").iloc[[0]])
    merged_perf_with_source = pd.concat(chosen_chunks, ignore_index=True, sort=False) if chosen_chunks else pd.DataFrame()
    choice = merged_perf_with_source[[*perf_keys, "_source"]].drop_duplicates() if not merged_perf_with_source.empty else pd.DataFrame()
    merged_perf = merged_perf_with_source.drop(columns=["_source"], errors="ignore")

    pred_frames = []
    if not existing_pred.empty:
        existing_pred = existing_pred.copy()
        existing_pred["_source"] = "existing"
        pred_frames.append(existing_pred)
    if not pred.empty:
        new_pred = pred.copy()
        new_pred["_source"] = "new"
        pred_frames.append(new_pred)
    if pred_frames and not choice.empty:
        all_pred = pd.concat(pred_frames, ignore_index=True, sort=False)
        pred_merge_keys = [c for c in perf_keys if c in all_pred.columns]
        merged_pred = all_pred.merge(choice, on=pred_merge_keys, how="inner")
        merged_pred = merged_pred[merged_pred["_source_x"].astype(str).eq(merged_pred["_source_y"].astype(str))].copy()
        merged_pred = merged_pred.drop(columns=["_source_x", "_source_y"], errors="ignore")
        # One OOF prediction per subject/model is required. Older merged outputs can
        # contain the same subject under different fold labels from prior reruns.
        pred_keys = [c for c in ("task_type", "target", "target_mode", "feature_policy", "model", "subject_id") if c in merged_pred.columns]
        if pred_keys:
            merged_pred = merged_pred.drop_duplicates(pred_keys, keep="last")
    else:
        merged_pred = pred

    policy_frames = [df for df in (existing_policies, policy_rows) if df is not None and not df.empty]
    if policy_frames:
        merged_policies = pd.concat(policy_frames, ignore_index=True, sort=False)
        if "feature_policy" in merged_policies.columns:
            merged_policies = merged_policies.drop_duplicates(["feature_policy"], keep="last")
    else:
        merged_policies = policy_rows

    meta_frames = [df for df in (existing_meta, meta_audit) if df is not None and not df.empty]
    if meta_frames:
        merged_meta = pd.concat(meta_frames, ignore_index=True, sort=False)
        keys = [c for c in ("meta_feature_family", "source_model") if c in merged_meta.columns]
        if keys:
            merged_meta = merged_meta.drop_duplicates(keys, keep="last")
    else:
        merged_meta = meta_audit

    _progress(
        progress,
        f"[merge-existing] scoped_base_performance_rows={len(merged_perf)} scoped_base_prediction_rows={len(merged_pred)} "
        f"new_rows={len(perf)} existing_rows={len(existing_perf)}",
    )
    return merged_perf, merged_pred, merged_policies, merged_meta


def run_search(
    deriv_root: str | Path | None = None,
    *,
    fast: bool = False,
    n_jobs: int = 2,
    include_history: bool = False,
    include_disease_meta: bool = False,
    include_age: bool = False,
    include_non_age_metadata: bool = False,
    include_graph_topology: bool = False,
    policy_filter: set[str] | None = None,
    model_filter: set[str] | None = None,
    merge_existing: bool = False,
    progress: bool = False,
) -> dict[str, pd.DataFrame]:
    ml_dir = _analysis_dir(deriv_root)
    matrix = _read_required(ml_dir / "ml_subject_feature_matrix.csv")
    catalog = _read_required(ml_dir / "ml_feature_catalog.csv")
    matches = _read_required(ml_dir / "score_prediction_target_matches.csv")
    policies = _feature_policies(matrix, catalog)
    meta_audit = pd.DataFrame()
    if include_disease_meta:
        matrix, policies, meta_audit = _add_disease_meta_features(ml_dir, matrix, policies)
    if include_non_age_metadata:
        matrix, policies, non_age_meta_audit = _add_non_age_metadata_features(ml_dir, matrix, policies)
        meta_audit = pd.concat([df for df in (meta_audit, non_age_meta_audit) if df is not None and not df.empty], ignore_index=True, sort=False)
    if include_graph_topology:
        matrix, policies, graph_topology_audit = _add_graph_topology_features(ml_dir, matrix, policies)
        meta_audit = pd.concat([df for df in (meta_audit, graph_topology_audit) if df is not None and not df.empty], ignore_index=True, sort=False)
    if include_age:
        matrix, policies, age_audit = _add_age_feature(ml_dir, matrix, policies)
        meta_audit = pd.concat([df for df in (meta_audit, age_audit) if df is not None and not df.empty], ignore_index=True, sort=False)
    if fast:
        fast_order = [
            "all_structural_no_age",
            "selected_cv_no_age",
            "selected_no_raw_edges_no_age",
            "selected_repeated_cv_no_age",
            "high_stability_no_age",
            "top_stability_no_age",
            "compact_no_raw_edges_no_age",
            "stable_summary_no_age",
            "global_network_summary_no_age",
            "regional_topology_no_age",
            "edr_delay_no_age",
            "edr_global_stability_no_age",
            "microstructure_topology_stability_no_age",
            "low_missing_structural_no_age",
            "very_low_missing_structural_no_age",
            "low_missing_selected_no_age",
            "low_missing_stable_no_age",
            "low_missing_global_topology_no_age",
            "no_matrix_summaries_no_age",
            "graph_topology_no_age",
            "top_stability_no_age_plus_graph_topology_no_age",
            "low_missing_global_topology_no_age_plus_graph_topology_no_age",
            "no_matrix_summaries_no_age_plus_graph_topology_no_age",
            "compact_no_raw_edges_no_age_plus_graph_topology_no_age",
            "disease_meta_only_experimental",
            "selected_cv_no_age_plus_disease_meta_experimental",
            "top_stability_no_age_plus_disease_meta_experimental",
            "global_network_summary_no_age_plus_disease_meta_experimental",
            "low_missing_global_topology_no_age_plus_disease_meta_experimental",
            "compact_no_raw_edges_no_age_plus_disease_meta_experimental",
            "no_matrix_summaries_no_age_plus_disease_meta_experimental",
            "non_age_metadata_only_experimental",
            "selected_cv_no_age_plus_non_age_metadata_experimental",
            "top_stability_no_age_plus_non_age_metadata_experimental",
            "global_network_summary_no_age_plus_non_age_metadata_experimental",
            "low_missing_global_topology_no_age_plus_non_age_metadata_experimental",
            "compact_no_raw_edges_no_age_plus_non_age_metadata_experimental",
            "no_matrix_summaries_no_age_plus_non_age_metadata_experimental",
            "age_only_experimental",
            "selected_cv_no_age_plus_age_experimental",
            "selected_repeated_cv_no_age_plus_age_experimental",
            "high_stability_no_age_plus_age_experimental",
            "top_stability_no_age_plus_age_experimental",
            "edr_delay_no_age_plus_age_experimental",
            "global_network_summary_no_age_plus_age_experimental",
            "low_missing_stable_no_age_plus_age_experimental",
            "low_missing_global_topology_no_age_plus_age_experimental",
            "compact_no_raw_edges_no_age_plus_age_experimental",
        ]
        policies = {name: policies[name] for name in fast_order if name in policies}
    merge_scope_policies = set(policies)
    if policy_filter:
        policies = {name: features for name, features in policies.items() if name in policy_filter}
        if not policies:
            raise ValueError(f"No requested feature policies are available: {sorted(policy_filter)}")
    _progress(
        progress,
        f"[search] fast={fast} include_history={include_history} policies={len(policies)} "
        f"include_age={include_age} include_non_age_metadata={include_non_age_metadata} "
        f"include_graph_topology={include_graph_topology} "
        f"model_filter={sorted(model_filter) if model_filter else 'all'} merge_existing={merge_existing}",
    )
    reg_perf, reg_pred = _run_regression_search(matrix, matches, policies, fast, n_jobs, include_history, model_filter, progress)
    cdr_perf, cdr_pred = _run_cdr_search(matrix, matches, policies, fast, n_jobs, include_history, model_filter, progress)
    perf = pd.concat([reg_perf, cdr_perf], ignore_index=True, sort=False)
    pred = pd.concat([reg_pred, cdr_pred], ignore_index=True, sort=False)
    out_dir = ml_dir / OUT_DIR_NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    policy_rows = pd.DataFrame(
        [{"feature_policy": name, "n_features": len(features), "example_features": "; ".join(features[:12])} for name, features in policies.items()]
    )
    if merge_existing:
        perf, pred, policy_rows, meta_audit = _merge_existing_outputs(
            out_dir,
            perf,
            pred,
            policy_rows,
            meta_audit,
            merge_scope_policies,
            include_age,
            include_disease_meta,
            include_non_age_metadata,
            progress=progress,
        )
    perf, pred = _append_oof_ensembles(perf, pred)
    perf.to_csv(out_dir / "clinical_outcome_model_search_performance.csv", index=False)
    pred.to_csv(out_dir / "clinical_outcome_model_search_cv_predictions.csv", index=False)
    policy_rows.to_csv(out_dir / "clinical_outcome_model_search_feature_policies.csv", index=False)
    meta_audit.to_csv(out_dir / "clinical_outcome_model_search_meta_feature_audit.csv", index=False)
    summary_rows = []
    # The active paper/search goal is explicitly no-age. Keep any requested age
    # sensitivity rows in the detailed artifacts, but never let them satisfy or
    # redefine the goal summary.
    goal_perf = perf[~_is_explicit_age_policy(perf["feature_policy"])].copy() if "feature_policy" in perf.columns else perf.copy()
    final_reg_perf = goal_perf[goal_perf["task_type"].astype(str).eq("regression")].copy()
    final_cdr_perf = goal_perf[goal_perf["task_type"].astype(str).eq("classification")].copy()
    if not final_reg_perf.empty:
        best = final_reg_perf.sort_values("r2", ascending=False, na_position="last").iloc[0]
        summary_rows.append({"task": "MMSE regression", "best_metric": "r2", "best_value": best.get("r2"), "goal": SEARCH_TARGET_R2, "goal_met": bool(best.get("r2", np.nan) >= SEARCH_TARGET_R2), "model": best.get("model"), "feature_policy": best.get("feature_policy"), "target_mode": best.get("target_mode")})
    if not final_cdr_perf.empty:
        best = final_cdr_perf.sort_values("accuracy", ascending=False, na_position="last").iloc[0]
        summary_rows.append({"task": "Global CDR classification", "best_metric": "accuracy", "best_value": best.get("accuracy"), "goal": SEARCH_TARGET_ACCURACY, "goal_met": bool(best.get("accuracy", np.nan) >= SEARCH_TARGET_ACCURACY), "model": best.get("model"), "feature_policy": best.get("feature_policy"), "target_mode": best.get("target_mode")})
    pd.DataFrame(summary_rows).to_csv(out_dir / "clinical_outcome_model_search_goal_summary.csv", index=False)
    ceiling_audit = _write_ceiling_audit(out_dir, matches, perf)
    signal_audit = _write_signal_audit(out_dir, matrix, catalog, matches, perf, pred)
    return {"performance": perf, "predictions": pred, "feature_policies": policy_rows, "goal_summary": pd.DataFrame(summary_rows), "ceiling_audit": ceiling_audit, "out_dir": pd.DataFrame([{"out_dir": str(out_dir)}])}


def write_current_signal_audit(deriv_root: str | Path | None = None) -> dict[str, pd.DataFrame]:
    ml_dir = _analysis_dir(deriv_root)
    out_dir = ml_dir / OUT_DIR_NAME
    matrix = _read_required(ml_dir / "ml_subject_feature_matrix.csv")
    catalog = _read_required(ml_dir / "ml_feature_catalog.csv")
    matches = _read_required(ml_dir / "score_prediction_target_matches.csv")
    perf = _safe_read_optional_csv(out_dir / "clinical_outcome_model_search_performance.csv")
    pred = _safe_read_optional_csv(out_dir / "clinical_outcome_model_search_cv_predictions.csv")
    return _write_signal_audit(out_dir, matrix, catalog, matches, perf, pred)


def main() -> int:
    parser = argparse.ArgumentParser(description="Leakage-safe stronger clinical outcome model search.")
    parser.add_argument("--deriv-root", default=None,
                        help="derivatives root (default: from sc_config / the SC_* variables)")
    parser.add_argument("--fast", action="store_true", help="Use a smaller grid for quick iteration.")
    parser.add_argument("--include-history", action="store_true", help="Also run all-history sensitivity mode.")
    parser.add_argument("--include-disease-meta", action="store_true", help="Add experimental out-of-fold structural disease-probability meta-features.")
    parser.add_argument("--include-age", action="store_true", help="Add explicit user-requested chronological-age experimental feature policies.")
    parser.add_argument("--include-non-age-metadata", action="store_true", help="Add explicit no-age sex/APOE metadata sensitivity feature policies.")
    parser.add_argument("--include-graph-topology", action="store_true", help="Add no-age graph topology features derived from structural count connectomes.")
    parser.add_argument("--policy", action="append", default=None, help="Limit to a feature policy. May be repeated.")
    parser.add_argument("--model", action="append", default=None, help="Limit to a model name. May be repeated.")
    parser.add_argument("--merge-existing", action="store_true", help="Merge new search rows with existing search outputs, keeping the best duplicate by task metric.")
    parser.add_argument("--progress", action="store_true", help="Print progress for each policy/model branch.")
    parser.add_argument("--audit-only", action="store_true", help="Write signal/timing/error audit artifacts from existing outputs without running model search.")
    parser.add_argument("--n-jobs", type=int, default=2)
    args = parser.parse_args()
    if args.audit_only:
        outputs = write_current_signal_audit(args.deriv_root)
        print("wrote signal audit artifacts")
        for name, frame in outputs.items():
            print(f"{name}: {len(frame)} rows")
        return 0
    outputs = run_search(
        args.deriv_root,
        fast=args.fast,
        n_jobs=args.n_jobs,
        include_history=args.include_history,
        include_disease_meta=args.include_disease_meta,
        include_age=args.include_age,
        include_non_age_metadata=args.include_non_age_metadata,
        include_graph_topology=args.include_graph_topology,
        policy_filter=set(args.policy) if args.policy else None,
        model_filter=set(args.model) if args.model else None,
        merge_existing=args.merge_existing,
        progress=args.progress,
    )
    perf = outputs["performance"]
    print(f"wrote {outputs['out_dir'].iloc[0]['out_dir']}")
    if not perf.empty:
        reg = perf[perf["task_type"].eq("regression")].sort_values("r2", ascending=False, na_position="last").head(10)
        cdr = perf[perf["task_type"].eq("classification")].sort_values("accuracy", ascending=False, na_position="last").head(10)
        print("\nTop MMSE regression")
        print(reg[["target_mode", "feature_policy", "model", "n_subjects", "r2", "mae", "rmse", "spearman_r"]].to_string(index=False))
        print("\nTop Global CDR classification")
        print(cdr[["target_mode", "feature_policy", "model", "n_subjects", "accuracy", "balanced_accuracy", "macro_f1", "macro_specificity"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
