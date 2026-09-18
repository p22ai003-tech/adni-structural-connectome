from __future__ import annotations

import math
import importlib.util
import re
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import SelectKBest, mutual_info_classif, mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    explained_variance_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    median_absolute_error,
    mean_squared_error,
    precision_score,
    precision_recall_fscore_support,
    recall_score,
    r2_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from connectome_analysis.analysis_config import GROUP_ORDER, AnalysisPaths
from connectome_analysis.analysis_edges import load_connectome_stack
from connectome_analysis.analysis_plots import save_dataframe, write_inference_markdown

try:
    from xgboost import DMatrix, XGBClassifier, XGBRegressor

    HAS_XGBOOST = True
except Exception:  # pragma: no cover - fallback path depends on local optional package
    DMatrix = None
    XGBClassifier = None
    XGBRegressor = None
    HAS_XGBOOST = False

try:
    import statsmodels.formula.api as smf

    HAS_STATSMODELS = True
except Exception:  # pragma: no cover - optional dependency
    smf = None
    HAS_STATSMODELS = False


TARGET_NAME = "multiclass_CN_MCI_AD"
FEATURE_SET_NAME = "comprehensive_structural"
LEAKAGE_CLINICAL_COLUMNS = (
    "Global CDR",
    "MMSE Total Score",
    "NPI-Q Total Score",
    "GDSCALE Total Score",
    "FAQ Total Score",
)
SCORE_REGRESSION_TARGETS = ("MMSE Total Score",)
CDR_CLASSIFICATION_TARGET = "Global CDR"
CDR_CLASS_ORDER = ("0", "0.5", "1", "2+")
ACTIVE_CLINICAL_TARGETS = (*SCORE_REGRESSION_TARGETS, CDR_CLASSIFICATION_TARGET)
AUDIT_SCORE_TARGETS = (*ACTIVE_CLINICAL_TARGETS, "NPI-Q Total Score")
CLINICAL_TARGET_WINDOWS_DAYS = (0, 30, 90, 180, 365, 730)
PRIMARY_SCORE_WINDOW_DAYS = 730
MIN_SCORE_TARGET_SUBJECTS = 60
MIN_SCORE_TARGET_GROUPS = 2
MIN_NEURAL_SCORE_SUBJECTS = 60
MIN_NEURAL_CLASSIFIER_SUBJECTS = 120
MISSINGNESS_DROP_THRESHOLD = 0.70
SPARSE_PRESENCE_THRESHOLD = 0.30
NEAR_ZERO_VARIANCE_THRESHOLD = 1e-8
CORRELATION_DROP_THRESHOLD = 0.95


@dataclass(frozen=True)
class FeatureSpec:
    feature: str
    feature_group: str
    source_section: str
    source_file: str
    source_column: str


class NoiseAugmentedMLPRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        hidden_layer_sizes: tuple[int, ...] = (48, 12),
        alpha: float = 1e-2,
        learning_rate_init: float = 1e-3,
        noise_sd: float = 0.03,
        augmentation_repeats: int = 3,
        max_iter: int = 220,
        random_state: int = 42,
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.alpha = alpha
        self.learning_rate_init = learning_rate_init
        self.noise_sd = noise_sd
        self.augmentation_repeats = augmentation_repeats
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, x: np.ndarray, y: np.ndarray) -> "NoiseAugmentedMLPRegressor":
        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        rng = np.random.default_rng(self.random_state)
        x_parts = [x_arr]
        y_parts = [y_arr]
        for _ in range(max(0, int(self.augmentation_repeats))):
            x_parts.append(x_arr + rng.normal(0.0, float(self.noise_sd), size=x_arr.shape))
            y_parts.append(y_arr)
        self.model_ = MLPRegressor(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation="relu",
            alpha=self.alpha,
            learning_rate_init=self.learning_rate_init,
            early_stopping=True,
            max_iter=self.max_iter,
            random_state=self.random_state,
        )
        self.model_.fit(np.vstack(x_parts), np.concatenate(y_parts))
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.model_.predict(np.asarray(x, dtype=float))


def _section_path(paths: AnalysisPaths, rel: str) -> Path:
    return paths.output_root / rel


def _safe_read_csv(paths: AnalysisPaths, rel: str) -> pd.DataFrame:
    path = _section_path(paths, rel)
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _sanitize(value: object) -> str:
    text = str(value).strip()
    text = re.sub(r"[^0-9A-Za-z]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text.lower() or "value"


def _is_statistic_leakage_column(column: object) -> bool:
    lower = str(column).strip().lower()
    if "welch" in lower:
        return True
    exact = {
        "p",
        "q",
        "p_value",
        "q_value",
        "pvalue",
        "qvalue",
        "kw_p",
        "kw_q",
        "bm_p",
        "bm_q",
        "mwu_p",
        "mwu_q",
        "perm_p",
        "perm_q",
        "bh_q",
        "fdr_q",
    }
    if lower in exact:
        return True
    return bool(
        re.search(
            r"(^|_)(p|q|pval|qval|pvalue|qvalue|kw_p|bm_p|mwu_p|perm_p|bh_q|fdr_q)$",
            lower,
        )
    )


def _numeric_series(df: pd.DataFrame, candidates: tuple[str, ...]) -> pd.Series | None:
    for col in candidates:
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce")
    return None


def _subject_universe(paths: AnalysisPaths, master: pd.DataFrame) -> pd.DataFrame:
    base = master.drop_duplicates("subject_id").copy()
    base = base[base["group"].isin(GROUP_ORDER)].copy()
    completeness = _safe_read_csv(paths, "01_qc/data_completeness_subject_level.csv")
    if not completeness.empty and {"subject_id", "final_connectome_ready"}.issubset(completeness.columns):
        complete_ids = set(
            completeness.loc[
                pd.to_numeric(completeness["final_connectome_ready"], errors="coerce").fillna(0).astype(int).eq(1),
                "subject_id",
            ].astype(str)
        )
        filtered = base[base["subject_id"].astype(str).isin(complete_ids)].copy()
        # master is already the dense analysis cohort; only narrow if the completeness flag is meaningful
        # (never collapse to empty — that previously zeroed the ML universe).
        if not filtered.empty:
            base = filtered
    return base[["subject_id", "group"]].reset_index(drop=True)


def _add_feature_specs(
    specs: list[FeatureSpec],
    columns: list[str],
    feature_group: str,
    source_section: str,
    source_file: str,
    source_columns: dict[str, str] | None = None,
) -> None:
    source_columns = source_columns or {}
    for col in columns:
        specs.append(
            FeatureSpec(
                feature=col,
                feature_group=feature_group,
                source_section=source_section,
                source_file=source_file,
                source_column=source_columns.get(col, col),
            )
        )


def _merge_feature_frame(
    matrix: pd.DataFrame,
    feature_df: pd.DataFrame,
    specs: list[FeatureSpec],
    *,
    prefix: str,
    feature_group: str,
    source_section: str,
    source_file: str,
    exclude_cols: set[str] | None = None,
) -> pd.DataFrame:
    exclude_cols = exclude_cols or set()
    if feature_df.empty or "subject_id" not in feature_df.columns:
        return matrix
    work = feature_df.copy()
    rename: dict[str, str] = {}
    source_columns: dict[str, str] = {}
    for col in work.columns:
        if col == "subject_id" or col in exclude_cols:
            continue
        if _is_statistic_leakage_column(col):
            continue
        if not pd.api.types.is_numeric_dtype(work[col]):
            continue
        new_col = f"{prefix}__{_sanitize(col)}"
        if new_col in matrix.columns or new_col in rename.values():
            suffix = 2
            while f"{new_col}_{suffix}" in matrix.columns or f"{new_col}_{suffix}" in rename.values():
                suffix += 1
            new_col = f"{new_col}_{suffix}"
        rename[col] = new_col
        source_columns[new_col] = col
    if not rename:
        return matrix
    keep = ["subject_id", *rename.keys()]
    work = work[keep].rename(columns=rename)
    matrix = matrix.merge(work, on="subject_id", how="left")
    _add_feature_specs(specs, list(rename.values()), feature_group, source_section, source_file, source_columns)
    return matrix


def _pivot_node_micro(paths: AnalysisPaths) -> tuple[pd.DataFrame, list[FeatureSpec]]:
    rel = "04_node_microstructure_live/live_node_microstructure_long.csv"
    df = _safe_read_csv(paths, rel)
    specs: list[FeatureSpec] = []
    if df.empty or not {"subject_id", "metric", "node_name", "value"}.issubset(df.columns):
        return pd.DataFrame(), specs
    df = df.copy()
    df["feature"] = "local_roi_dti__" + df["metric"].astype(str).map(_sanitize) + "__" + df["node_name"].astype(str).map(_sanitize)
    wide = df.pivot_table(index="subject_id", columns="feature", values="value", aggfunc="mean").reset_index()
    _add_feature_specs(specs, [c for c in wide.columns if c != "subject_id"], "Local ROI DTI", "04_node_microstructure_live", rel, {c: "value" for c in wide.columns if c != "subject_id"})
    return wide, specs


def _pivot_node_graph(paths: AnalysisPaths) -> tuple[pd.DataFrame, list[FeatureSpec]]:
    rel = "07_node_graph_live/live_node_graph_long.csv"
    df = _safe_read_csv(paths, rel)
    specs: list[FeatureSpec] = []
    metrics = [c for c in ("strength", "degree", "nodal_eff") if c in df.columns]
    if df.empty or not {"subject_id", "node_name"}.issubset(df.columns) or not metrics:
        return pd.DataFrame(), specs
    pivots = []
    for metric in metrics:
        tmp = df.pivot_table(index="subject_id", columns="node_name", values=metric, aggfunc="mean")
        tmp.columns = [f"node_metrics__{_sanitize(metric)}__{_sanitize(c)}" for c in tmp.columns]
        pivots.append(tmp)
    wide = pd.concat(pivots, axis=1).reset_index()
    _add_feature_specs(specs, [c for c in wide.columns if c != "subject_id"], "Node metrics", "07_node_graph_live", rel, {c: c.split("__", 1)[-1] for c in wide.columns if c != "subject_id"})
    return wide, specs


def _pivot_coupling(paths: AnalysisPaths) -> tuple[pd.DataFrame, list[FeatureSpec]]:
    rel = "09_coupling_live/live_subject_level_coupling.csv"
    df = _safe_read_csv(paths, rel)
    specs: list[FeatureSpec] = []
    if df.empty or not {"subject_id", "nodal_metric", "micro_metric", "rho"}.issubset(df.columns):
        return pd.DataFrame(), specs
    df = df.copy()
    df["pair"] = df["nodal_metric"].astype(str).map(_sanitize) + "__" + df["micro_metric"].astype(str).map(_sanitize)
    rho = df.pivot_table(index="subject_id", columns="pair", values="rho", aggfunc="mean")
    rho.columns = [f"coupling__rho__{c}" for c in rho.columns]
    n_nodes = df.pivot_table(index="subject_id", columns="pair", values="n_nodes", aggfunc="mean") if "n_nodes" in df.columns else pd.DataFrame(index=rho.index)
    if not n_nodes.empty:
        n_nodes.columns = [f"coupling__n_nodes__{c}" for c in n_nodes.columns]
        wide = pd.concat([rho, n_nodes], axis=1).reset_index()
    else:
        wide = rho.reset_index()
    _add_feature_specs(specs, [c for c in wide.columns if c != "subject_id"], "Coupling", "09_coupling_live", rel)
    return wide, specs


def _pivot_edr_nodes(paths: AnalysisPaths) -> tuple[pd.DataFrame, list[FeatureSpec]]:
    rel = "17_edr_exceptions/edr_exception_node_level_fd_sum_len_mean.csv"
    df = _safe_read_csv(paths, rel)
    specs: list[FeatureSpec] = []
    metrics = [
        c
        for c in (
            "sr_exception_rate",
            "lr_exception_rate",
            "sr_exception_strength",
            "lr_exception_strength",
            "sr_lr_exception_rate_diff",
        )
        if c in df.columns
    ]
    if df.empty or not {"subject_id", "node_name"}.issubset(df.columns) or not metrics:
        return pd.DataFrame(), specs
    pivots = []
    for metric in metrics:
        tmp = df.pivot_table(index="subject_id", columns="node_name", values=metric, aggfunc="mean")
        tmp.columns = [f"edr_aal__{_sanitize(metric)}__{_sanitize(c)}" for c in tmp.columns]
        pivots.append(tmp)
    wide = pd.concat(pivots, axis=1).reset_index()
    _add_feature_specs(specs, [c for c in wide.columns if c != "subject_id"], "AAL EDR exceptions", "17_edr_exceptions", rel)
    return wide, specs


def _matrix_summary_and_edges(
    paths: AnalysisPaths,
    master: pd.DataFrame,
    base_subjects: set[str],
    connectome_type: str,
    max_edge_features: int = 15,
) -> tuple[pd.DataFrame, list[FeatureSpec]]:
    specs: list[FeatureSpec] = []
    try:
        stack = load_connectome_stack(paths, master, connectome_type=connectome_type)
    except Exception:
        return pd.DataFrame(), specs
    subject_ids = np.asarray(stack.subject_ids, dtype=object)
    keep_subject = np.array([sid in base_subjects for sid in subject_ids], dtype=bool)
    if not keep_subject.any():
        return pd.DataFrame(), specs
    mats = stack.matrices[keep_subject]
    subject_ids = subject_ids[keep_subject]
    iu = np.triu_indices(stack.nodes, 1)
    vals = mats[:, iu[0], iu[1]].astype(float)
    finite = np.isfinite(vals)
    positive = finite & (vals > 0)
    possible = vals.shape[1]
    rows = []
    summary_cols = [
        "positive_edge_count",
        "positive_density",
        "edge_mean",
        "edge_median",
        "edge_sd",
        "edge_iqr",
        "edge_q10",
        "edge_q90",
        "edge_sum",
        "edge_min",
        "edge_max",
    ]
    for sid, row_vals, pos_mask in zip(subject_ids, vals, positive):
        valid_vals = row_vals[pos_mask]
        if valid_vals.size:
            q10, q25, q75, q90 = np.nanpercentile(valid_vals, [10, 25, 75, 90])
            rows.append(
                {
                    "subject_id": sid,
                    f"matrix_summary__{connectome_type}__positive_edge_count": int(pos_mask.sum()),
                    f"matrix_summary__{connectome_type}__positive_density": float(pos_mask.sum() / possible),
                    f"matrix_summary__{connectome_type}__edge_mean": float(np.nanmean(valid_vals)),
                    f"matrix_summary__{connectome_type}__edge_median": float(np.nanmedian(valid_vals)),
                    f"matrix_summary__{connectome_type}__edge_sd": float(np.nanstd(valid_vals, ddof=1)) if valid_vals.size > 1 else np.nan,
                    f"matrix_summary__{connectome_type}__edge_iqr": float(q75 - q25),
                    f"matrix_summary__{connectome_type}__edge_q10": float(q10),
                    f"matrix_summary__{connectome_type}__edge_q90": float(q90),
                    f"matrix_summary__{connectome_type}__edge_sum": float(np.nansum(valid_vals)),
                    f"matrix_summary__{connectome_type}__edge_min": float(np.nanmin(valid_vals)),
                    f"matrix_summary__{connectome_type}__edge_max": float(np.nanmax(valid_vals)),
                }
            )
        else:
            rows.append({"subject_id": sid, **{f"matrix_summary__{connectome_type}__{col}": np.nan for col in summary_cols}})
    out = pd.DataFrame(rows)
    summary_features = [c for c in out.columns if c != "subject_id"]
    _add_feature_specs(specs, summary_features, "Raw matrix summaries", "connectomes", f"SC_AAL166_*_{connectome_type}.csv")
    edge_presence = positive.sum(axis=0) / max(len(subject_ids), 1)
    edge_var = np.nanvar(np.where(finite, vals, np.nan), axis=0)
    eligible = np.where((edge_presence >= SPARSE_PRESENCE_THRESHOLD) & np.isfinite(edge_var) & (edge_var > 0))[0]
    if eligible.size:
        selected = eligible[np.argsort(edge_var[eligible])[::-1][:max_edge_features]]
        edge_df = pd.DataFrame({"subject_id": subject_ids})
        for edge_idx in selected:
            i = int(iu[0][edge_idx] + 1)
            j = int(iu[1][edge_idx] + 1)
            col = f"raw_edge__{connectome_type}__n{i:03d}_n{j:03d}"
            edge_df[col] = vals[:, edge_idx]
        out = out.merge(edge_df, on="subject_id", how="left")
        _add_feature_specs(
            specs,
            [c for c in edge_df.columns if c != "subject_id"],
            "Raw high-variance edges",
            "connectomes",
            f"SC_AAL166_*_{connectome_type}.csv",
        )
    return out, specs


def _assemble_feature_matrix(paths: AnalysisPaths, master: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = _subject_universe(paths, master)
    matrix = base.copy()
    specs: list[FeatureSpec] = []

    subject_tables = (
        ("03_global_microstructure_live/live_global_microstructure_subject_table.csv", "Global DTI", "03_global_microstructure_live", "global_dti"),
        ("06_global_graph_live/live_global_graph_subject_table.csv", "Global graph", "06_global_graph_live", "global_graph"),
        ("11_brain_age_live/live_brain_age_predictions.csv", "Brain age", "11_brain_age_live", "brain_age"),
        ("12_length_delay/lr_sr_subject_level_fd_sum_len_mean.csv", "LR/SR and delay", "12_length_delay", "lr_sr"),
        ("13_delay/delay_subject_level_fd_sum_len_mean.csv", "Delay", "13_delay", "delay"),
        ("15_advanced_structural/edr_subject_level_fd_sum_len_mean.csv", "Advanced structural", "15_advanced_structural", "advanced"),
        ("17_edr_exceptions/edr_exception_subject_level_fd_sum_len_mean.csv", "EDR exceptions", "17_edr_exceptions", "edr"),
    )
    for rel, group, section, prefix in subject_tables:
        df = _safe_read_csv(paths, rel)
        matrix = _merge_feature_frame(
            matrix,
            df,
            specs,
            prefix=prefix,
            feature_group=group,
            source_section=section,
            source_file=rel,
            exclude_cols={"group", "age", "sex", "phase"},
        )

    for wide, wide_specs in (_pivot_node_micro(paths), _pivot_node_graph(paths), _pivot_coupling(paths), _pivot_edr_nodes(paths)):
        if not wide.empty:
            matrix = matrix.merge(wide, on="subject_id", how="left")
            specs.extend(wide_specs)

    base_subjects = set(base["subject_id"].astype(str))
    for connectome_type in ("fd_sum", "count", "len_mean", "fa_mean", "md_mean", "rd_mean", "ad_mean"):
        wide, wide_specs = _matrix_summary_and_edges(paths, master, base_subjects, connectome_type)
        if not wide.empty:
            matrix = matrix.merge(wide, on="subject_id", how="left")
            specs.extend(wide_specs)

    catalog = pd.DataFrame([spec.__dict__ for spec in specs])
    return matrix, catalog


def _prefilter_features(matrix: pd.DataFrame, catalog: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    feature_cols = [c for c in matrix.columns if c not in {"subject_id", "group"}]
    numeric = matrix[feature_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    n_subjects = len(numeric)
    catalog = catalog.drop_duplicates("feature").copy()
    catalog = catalog.set_index("feature", drop=False)
    rows = []
    retained = []
    for feature in feature_cols:
        series = numeric[feature]
        finite = np.isfinite(series.to_numpy(dtype=float, copy=False))
        valid_values = series[finite]
        missingness = 1.0 - float(finite.sum() / max(n_subjects, 1))
        n_unique = int(valid_values.nunique(dropna=True))
        variance = float(valid_values.var(ddof=1)) if len(valid_values) > 1 else 0.0
        group = str(catalog.loc[feature, "feature_group"]) if feature in catalog.index else "Unknown"
        sparse_sensitive = any(token in group.lower() for token in ("aal", "edge", "node"))
        present = float((np.abs(valid_values) > 0).sum() / max(n_subjects, 1)) if sparse_sensitive else float(finite.sum() / max(n_subjects, 1))
        final_status = "prefilter_pass"
        drop_reason = ""
        if missingness > MISSINGNESS_DROP_THRESHOLD:
            final_status = "dropped_missingness"
            drop_reason = f">{int(MISSINGNESS_DROP_THRESHOLD * 100)}% missing"
        elif n_unique <= 1 or not np.isfinite(variance) or variance < NEAR_ZERO_VARIANCE_THRESHOLD:
            final_status = "dropped_low_variance"
            drop_reason = f"near-zero variance or <=1 unique value"
        elif sparse_sensitive and present < SPARSE_PRESENCE_THRESHOLD:
            final_status = "dropped_sparse"
            drop_reason = f"present in <{int(SPARSE_PRESENCE_THRESHOLD * 100)}% of subjects"
        else:
            retained.append(feature)
        rows.append(
            {
                "feature": feature,
                "missingness": missingness,
                "variance": variance,
                "n_unique": n_unique,
                "sparse_presence": present,
                "correlation_cluster": "",
                "selected_fold_count": 0,
                "elastic_net_nonzero_fold_count": 0,
                "final_status": final_status,
                "drop_reason": drop_reason,
            }
        )
    metrics = pd.DataFrame(rows).set_index("feature")
    retained_after_corr = []
    corr_drop: dict[str, tuple[str, str]] = {}
    if retained:
        work = numeric[retained]
        if len(retained) > 1:
            corr = work.corr(method="spearman", min_periods=max(10, int(0.2 * n_subjects))).abs()
            order_df = metrics.loc[retained, ["missingness", "variance"]].copy()
            order_df["priority"] = [
                0
                if "summary" in f or "global" in f
                else 2
                if "raw_edge" not in f
                else 3
                for f in order_df.index
            ]
            order_df = order_df.sort_values(["missingness", "priority", "variance"], ascending=[True, True, False])
            assigned: set[str] = set()
            cluster_id = 0
            for feature in order_df.index:
                if feature in assigned:
                    continue
                cluster_id += 1
                retained_after_corr.append(feature)
                metrics.loc[feature, "correlation_cluster"] = f"cluster_{cluster_id}"
                assigned.add(feature)
                high = corr.index[(corr[feature] >= CORRELATION_DROP_THRESHOLD).fillna(False)].tolist()
                for other in high:
                    if other == feature or other in assigned:
                        continue
                    assigned.add(other)
                    corr_drop[other] = (feature, f"correlated |Spearman r|>={CORRELATION_DROP_THRESHOLD:.2f} with {feature}")
                    metrics.loc[other, "correlation_cluster"] = f"cluster_{cluster_id}"
        else:
            retained_after_corr = retained
    for feature, (keeper, reason) in corr_drop.items():
        metrics.loc[feature, "final_status"] = "dropped_correlated"
        metrics.loc[feature, "drop_reason"] = reason
    for feature in retained_after_corr:
        if metrics.loc[feature, "final_status"] == "prefilter_pass":
            metrics.loc[feature, "final_status"] = "retained_for_model"
            metrics.loc[feature, "drop_reason"] = ""
    catalog = catalog.join(metrics, how="outer", rsuffix="_metric").reset_index(drop=True)
    return numeric, catalog, retained_after_corr


def _k_options(n_features: int) -> list[int]:
    values = [min(n_features, k) for k in (120,)]
    return sorted(set(v for v in values if v >= 2))


def _pipeline_specs(n_features: int, n_subjects: int | None = None, n_classes: int | None = None) -> dict[str, tuple[Pipeline, dict]]:
    score_func = partial(mutual_info_classif, random_state=42)
    k_values = _k_options(n_features)
    class_count = int(n_classes or len(GROUP_ORDER))
    specs: dict[str, tuple[Pipeline, dict]] = {
        "multinomial_elastic_net": (
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("select", SelectKBest(score_func=score_func, k=k_values[0])),
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            penalty="elasticnet",
                            solver="saga",
                            class_weight="balanced",
                            max_iter=900,
                            tol=1e-2,
                            random_state=42,
                        ),
                    ),
                ]
            ),
            {
                "select__k": k_values,
                "model__C": [1.0],
                "model__l1_ratio": [0.5],
            },
        )
    }
    if HAS_XGBOOST and XGBClassifier is not None:
        specs["xgboost_multiclass"] = (
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("select", SelectKBest(score_func=score_func, k=k_values[0])),
                    (
                        "model",
                        XGBClassifier(
                            objective="multi:softprob",
                            eval_metric="mlogloss",
                            num_class=class_count,
                            tree_method="hist",
                            n_estimators=90,
                            learning_rate=0.05,
                            subsample=0.85,
                            colsample_bytree=0.85,
                            random_state=42,
                            n_jobs=2,
                        ),
                    ),
                ]
            ),
            {
                "select__k": k_values,
                "model__max_depth": [3],
                "model__min_child_weight": [1],
            },
        )
    specs["extra_trees_multiclass"] = (
        Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("select", SelectKBest(score_func=score_func, k=k_values[0])),
                (
                    "model",
                    ExtraTreesClassifier(
                        n_estimators=120,
                        class_weight="balanced",
                        max_features="sqrt",
                        random_state=42,
                        n_jobs=2,
                    ),
                ),
            ]
        ),
        {
            "select__k": k_values,
            "model__min_samples_leaf": [2],
        },
    )
    specs["random_forest_multiclass"] = (
        Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("select", SelectKBest(score_func=score_func, k=k_values[0])),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=120,
                        class_weight="balanced_subsample",
                        max_features="sqrt",
                        random_state=42,
                        n_jobs=2,
                    ),
                ),
            ]
        ),
        {
            "select__k": k_values,
            "model__min_samples_leaf": [2],
        },
    )
    if n_subjects is None or n_subjects >= MIN_NEURAL_CLASSIFIER_SUBJECTS:
        specs["mlp_multiclass"] = (
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("select", SelectKBest(score_func=score_func, k=k_values[0])),
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        MLPClassifier(
                            hidden_layer_sizes=(64, 16),
                            activation="relu",
                            early_stopping=True,
                            max_iter=250,
                            random_state=42,
                        ),
                    ),
                ]
            ),
            {
                "select__k": k_values,
                "model__alpha": [1e-2],
                "model__learning_rate_init": [1e-3],
            },
        )
    return specs


def _selected_features(pipeline: Pipeline, feature_names: list[str]) -> list[str]:
    selector = pipeline.named_steps.get("select")
    if selector is None or not hasattr(selector, "get_support"):
        return feature_names
    support = selector.get_support()
    return [name for name, keep in zip(feature_names, support) if keep]


def _model_importance(pipeline: Pipeline, selected_names: list[str], model_name: str) -> dict[str, float]:
    model = pipeline.named_steps.get("model")
    if model is None:
        return {}
    if model_name == "multinomial_elastic_net" and hasattr(model, "coef_"):
        values = np.nanmean(np.abs(np.asarray(model.coef_, dtype=float)), axis=0)
    elif hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float)
    else:
        return {}
    return {feature: float(value) for feature, value in zip(selected_names, values)}


def _logistic_nonzero_features(pipeline: Pipeline, selected_names: list[str]) -> set[str]:
    model = pipeline.named_steps.get("model")
    if model is None or not hasattr(model, "coef_"):
        return set()
    coefs = np.nanmean(np.abs(np.asarray(model.coef_, dtype=float)), axis=0)
    return {feature for feature, value in zip(selected_names, coefs) if np.isfinite(value) and abs(value) > 1e-8}


def _xgb_shap_fold(pipeline: Pipeline, x_test: pd.DataFrame, selected_names: list[str]) -> dict[str, float]:
    if not HAS_XGBOOST or DMatrix is None or "xgboost" not in str(type(pipeline.named_steps.get("model"))).lower():
        return {}
    try:
        imputed = pipeline.named_steps["imputer"].transform(x_test)
        selected = pipeline.named_steps["select"].transform(imputed)
        booster = pipeline.named_steps["model"].get_booster()
        contrib = booster.predict(DMatrix(selected), pred_contribs=True)
    except Exception:
        return {}
    arr = np.asarray(contrib, dtype=float)
    if arr.ndim == 3:
        arr = arr[:, :, :-1]
        values = np.nanmean(np.abs(arr), axis=(0, 1))
    elif arr.ndim == 2:
        values = np.nanmean(np.abs(arr[:, :-1]), axis=0)
    else:
        return {}
    return {feature: float(value) for feature, value in zip(selected_names, values)}


def _specificity_by_class(y_true: np.ndarray, y_pred: np.ndarray, labels: list[int]) -> dict[int, float]:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    out = {}
    total = cm.sum()
    for idx, label in enumerate(labels):
        tp = cm[idx, idx]
        fp = cm[:, idx].sum() - tp
        fn = cm[idx, :].sum() - tp
        tn = total - tp - fp - fn
        out[label] = float(tn / (tn + fp)) if (tn + fp) else np.nan
    return out


def _run_multiclass_models(
    matrix: pd.DataFrame,
    numeric: pd.DataFrame,
    feature_names: list[str],
    catalog: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    task = matrix[["subject_id", "group"]].copy()
    valid_group = task["group"].isin(GROUP_ORDER)
    task = task[valid_group].reset_index(drop=True)
    x = numeric.loc[valid_group.to_numpy(), feature_names].reset_index(drop=True)
    encoder = LabelEncoder()
    encoder.fit(GROUP_ORDER)
    y = encoder.transform(task["group"].astype(str))
    counts = pd.Series(task["group"]).value_counts()
    min_class = int(counts.min())
    if min_class < 3 or len(feature_names) < 2:
        empty = pd.DataFrame()
        return {
            "performance": empty,
            "predictions": empty,
            "confusion": empty,
            "importance": empty,
            "shap": empty,
            "roc": empty,
            "class_metrics": empty,
            "fold_params": empty,
            "model_status": _disease_model_status_rows(len(task)),
            "catalog": catalog,
        }
    outer_splits = min(5, min_class)
    inner_splits = min(3, max(2, min_class - 1))
    outer = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=42)
    model_specs = _pipeline_specs(len(feature_names), len(task))
    prediction_rows: list[dict] = []
    importance_accumulator: dict[str, Counter] = defaultdict(Counter)
    shap_accumulator: dict[str, Counter] = defaultdict(Counter)
    selected_counts = Counter()
    nonzero_counts = Counter()
    best_param_rows: list[dict] = []

    for model_name, (pipe, grid) in model_specs.items():
        for fold, (train_idx, test_idx) in enumerate(outer.split(x, y), start=1):
            inner = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=420 + fold)
            search = GridSearchCV(pipe, grid, scoring="balanced_accuracy", cv=inner, n_jobs=4, pre_dispatch=4, refit=True)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                warnings.filterwarnings("ignore", category=UserWarning)
                search.fit(x.iloc[train_idx], y[train_idx])
            best = search.best_estimator_
            selected = _selected_features(best, feature_names)
            selected_counts.update(selected)
            if model_name == "multinomial_elastic_net":
                nonzero_counts.update(_logistic_nonzero_features(best, selected))
            importances = _model_importance(best, selected, model_name)
            importance_accumulator[model_name].update(importances)
            shap_values = _xgb_shap_fold(best, x.iloc[test_idx], selected)
            shap_accumulator[model_name].update(shap_values)
            y_pred = best.predict(x.iloc[test_idx])
            y_prob = best.predict_proba(x.iloc[test_idx]) if hasattr(best, "predict_proba") else np.full((len(test_idx), len(GROUP_ORDER)), np.nan)
            best_param_rows.append(
                {
                    "target": TARGET_NAME,
                    "feature_set": FEATURE_SET_NAME,
                    "model": model_name,
                    "fold": fold,
                    "best_inner_balanced_accuracy": float(search.best_score_),
                    "best_params": str(search.best_params_),
                    "n_selected_features": len(selected),
                }
            )
            for row_pos, idx in enumerate(test_idx):
                row = {
                    "target": TARGET_NAME,
                    "feature_set": FEATURE_SET_NAME,
                    "model": model_name,
                    "fold": fold,
                    "subject_id": task.iloc[idx]["subject_id"],
                    "true_group": encoder.inverse_transform([y[idx]])[0],
                    "predicted_group": encoder.inverse_transform([int(y_pred[row_pos])])[0],
                }
                for class_idx, class_name in enumerate(encoder.classes_):
                    row[f"prob_{class_name}"] = float(y_prob[row_pos, class_idx])
                prediction_rows.append(row)

    predictions = pd.DataFrame(prediction_rows)
    performance_rows = []
    class_metric_rows = []
    confusion_rows = []
    roc_rows = []
    labels_int = list(range(len(encoder.classes_)))
    for model_name in sorted(predictions["model"].dropna().unique()) if not predictions.empty else []:
        pred = predictions[predictions["model"] == model_name].copy()
        y_true = encoder.transform(pred["true_group"].astype(str))
        y_pred = encoder.transform(pred["predicted_group"].astype(str))
        prob = pred[[f"prob_{g}" for g in encoder.classes_]].to_numpy(dtype=float)
        macro_auroc = np.nan
        try:
            macro_auroc = float(roc_auc_score(y_true, prob, multi_class="ovr", average="macro"))
        except Exception:
            pass
        neg_log_loss = np.nan
        try:
            neg_log_loss = float(log_loss(y_true, prob, labels=labels_int))
        except Exception:
            pass
        performance_rows.append(
            {
                "target": TARGET_NAME,
                "feature_set": FEATURE_SET_NAME,
                "model": model_name,
                "status": "ok",
                "n_subjects": int(len(pred)),
                "n_features_input": int(len(feature_names)),
                "n_outer_folds": int(outer_splits),
                "n_inner_folds": int(inner_splits),
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
                "macro_auroc_ovr": macro_auroc,
                "log_loss": neg_log_loss,
                "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
                "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
                "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
                "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
            }
        )
        precision, recall, f1_vals, support = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=labels_int,
            zero_division=0,
        )
        specificity = _specificity_by_class(y_true, y_pred, labels_int)
        for idx, class_name in enumerate(encoder.classes_):
            class_metric_rows.append(
                {
                    "target": TARGET_NAME,
                    "feature_set": FEATURE_SET_NAME,
                    "model": model_name,
                    "class": class_name,
                    "support": int(support[idx]),
                    "precision": float(precision[idx]),
                    "recall_sensitivity": float(recall[idx]),
                    "specificity": float(specificity.get(idx, np.nan)),
                    "f1": float(f1_vals[idx]),
                }
            )
        cm = confusion_matrix(y_true, y_pred, labels=labels_int)
        for i, true_class in enumerate(encoder.classes_):
            for j, pred_class in enumerate(encoder.classes_):
                confusion_rows.append(
                    {
                        "target": TARGET_NAME,
                        "feature_set": FEATURE_SET_NAME,
                        "model": model_name,
                        "true_group": true_class,
                        "predicted_group": pred_class,
                        "n": int(cm[i, j]),
                    }
                )
        for class_idx, class_name in enumerate(encoder.classes_):
            try:
                fpr, tpr, thresholds = roc_curve((y_true == class_idx).astype(int), prob[:, class_idx])
                for point_idx, (fpr_v, tpr_v, thr) in enumerate(zip(fpr, tpr, thresholds)):
                    roc_rows.append(
                        {
                            "target": TARGET_NAME,
                            "feature_set": FEATURE_SET_NAME,
                            "model": model_name,
                            "class": class_name,
                            "point": point_idx,
                            "fpr": float(fpr_v),
                            "tpr": float(tpr_v),
                            "threshold": float(thr) if np.isfinite(thr) else np.nan,
                        }
                    )
            except Exception:
                continue

    importance_rows = []
    for model_name, counter in importance_accumulator.items():
        for rank, (feature, value) in enumerate(counter.most_common(), start=1):
            importance_rows.append(
                {
                    "target": TARGET_NAME,
                    "feature_set": FEATURE_SET_NAME,
                    "model": model_name,
                    "rank": rank,
                    "feature": feature,
                    "feature_group": _feature_group_for(catalog, feature),
                    "importance": float(value / max(outer_splits, 1)),
                    "selected_fold_count": int(selected_counts[feature]),
                    "importance_method": "abs_elastic_net_coef" if model_name == "multinomial_elastic_net" else "tree_feature_importance",
                }
            )
    shap_rows = []
    for model_name, counter in shap_accumulator.items():
        for rank, (feature, value) in enumerate(counter.most_common(), start=1):
            shap_rows.append(
                {
                    "target": TARGET_NAME,
                    "feature_set": FEATURE_SET_NAME,
                    "model": model_name,
                    "rank": rank,
                    "feature": feature,
                    "feature_group": _feature_group_for(catalog, feature),
                    "mean_abs_contribution": float(value / max(outer_splits, 1)),
                    "explanation_method": "TreeSHAP pred_contribs",
                }
            )

    catalog = catalog.copy()
    catalog["selected_fold_count"] = catalog["feature"].map(lambda f: int(selected_counts[str(f)])).fillna(0).astype(int)
    catalog["elastic_net_nonzero_fold_count"] = catalog["feature"].map(lambda f: int(nonzero_counts[str(f)])).fillna(0).astype(int)
    retained = catalog["final_status"].astype(str).eq("retained_for_model")
    catalog.loc[retained & catalog["selected_fold_count"].gt(0), "final_status"] = "selected_in_cv"
    catalog.loc[retained & catalog["selected_fold_count"].eq(0), "final_status"] = "retained_not_selected"

    return {
        "performance": pd.DataFrame(performance_rows),
        "predictions": predictions,
        "confusion": pd.DataFrame(confusion_rows),
        "importance": pd.DataFrame(importance_rows),
        "shap": pd.DataFrame(shap_rows),
        "roc": pd.DataFrame(roc_rows),
        "class_metrics": pd.DataFrame(class_metric_rows),
        "fold_params": pd.DataFrame(best_param_rows),
        "model_status": _disease_model_status_rows(len(task)),
        "catalog": catalog,
    }


def _read_csv_columns(path: Path, columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        header = pd.read_csv(path, nrows=0)
    except Exception:
        return pd.DataFrame()
    usecols = [col for col in columns if col in header.columns]
    if not usecols:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, usecols=usecols, low_memory=False)
    except Exception:
        return pd.DataFrame()


def _subject_id_from_clinical_table(df: pd.DataFrame) -> pd.Series:
    if "Subject ID" in df.columns:
        return df["Subject ID"].astype(str)
    if "subject_id" in df.columns:
        return df["subject_id"].astype(str)
    return pd.Series(index=df.index, dtype=object)


def _dti_anchor_table(paths: AnalysisPaths, feature_matrix: pd.DataFrame) -> pd.DataFrame:
    anchors = feature_matrix[["subject_id", "group"]].drop_duplicates("subject_id").copy()
    dti = _read_csv_columns(
        paths.cohort_dti_csv,
        ["Subject ID", "subject_id", "Study Date", "Visit", "Image ID"],
    )
    if dti.empty:
        anchors["anchor_date"] = pd.NaT
        anchors["dti_visit"] = np.nan
        anchors["dti_image_id"] = np.nan
        return anchors
    dti = dti.copy()
    dti["subject_id"] = _subject_id_from_clinical_table(dti)
    dti["anchor_date"] = pd.to_datetime(dti.get("Study Date"), errors="coerce")
    dti["dti_visit"] = dti.get("Visit", pd.Series(index=dti.index, dtype=object))
    dti["dti_image_id"] = dti.get("Image ID", pd.Series(index=dti.index, dtype=object))
    dti = dti.sort_values(["subject_id", "anchor_date"], na_position="last").drop_duplicates("subject_id")
    return anchors.merge(dti[["subject_id", "anchor_date", "dti_visit", "dti_image_id"]], on="subject_id", how="left")


def _clinical_score_sources(paths: AnalysisPaths) -> tuple[Path, ...]:
    cohort_dir = paths.cohort_dti_csv.parent
    candidates = [
        paths.cohort_dti_csv,
        cohort_dir / "dti_master.csv",
        cohort_dir / "mri_master.csv",
    ]
    out: list[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        path = Path(path)
        if path not in seen and path.exists():
            out.append(path)
            seen.add(path)
    return tuple(out)


def _read_clinical_score_rows(paths: AnalysisPaths) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    base_cols = ["Subject ID", "subject_id", "Study Date", "Visit", "Image ID"]
    for path in _clinical_score_sources(paths):
        df = _read_csv_columns(path, [*base_cols, *AUDIT_SCORE_TARGETS])
        if df.empty:
            continue
        df = df.copy()
        df["subject_id"] = _subject_id_from_clinical_table(df)
        df["clinical_date"] = pd.to_datetime(df.get("Study Date"), errors="coerce")
        df["clinical_visit"] = df.get("Visit", pd.Series(index=df.index, dtype=object))
        df["clinical_image_id"] = df.get("Image ID", pd.Series(index=df.index, dtype=object))
        df["clinical_source"] = path.name
        for target in AUDIT_SCORE_TARGETS:
            if target not in df.columns:
                df[target] = np.nan
        keep = [
            "subject_id",
            "clinical_date",
            "clinical_visit",
            "clinical_image_id",
            "clinical_source",
            *AUDIT_SCORE_TARGETS,
        ]
        rows.append(df[keep])
    if not rows:
        return pd.DataFrame()
    clinical = pd.concat(rows, ignore_index=True)
    clinical = clinical.dropna(subset=["subject_id"]).copy()
    clinical["subject_id"] = clinical["subject_id"].astype(str)
    dedupe_cols = ["subject_id", "clinical_date", "clinical_source", *AUDIT_SCORE_TARGETS]
    return clinical.drop_duplicates(dedupe_cols).reset_index(drop=True)


def _source_priority(source: object) -> int:
    order = {"dti.csv": 0, "dti_master.csv": 1, "mri_master.csv": 2}
    return order.get(str(source), 9)


def _nearest_score_matches(
    anchors: pd.DataFrame,
    clinical: pd.DataFrame,
    target: str,
    *,
    max_day_delta: int | None,
    target_mode: str,
) -> pd.DataFrame:
    columns = [
        "subject_id",
        "group",
        "target",
        "target_mode",
        "target_value",
        "anchor_date",
        "clinical_date",
        "abs_day_delta",
        "clinical_source",
        "clinical_visit",
        "clinical_image_id",
    ]
    if clinical.empty or target not in clinical.columns:
        return pd.DataFrame(columns=columns)
    scores = clinical[["subject_id", "clinical_date", "clinical_visit", "clinical_image_id", "clinical_source", target]].copy()
    scores["target_value"] = pd.to_numeric(scores[target], errors="coerce")
    scores = scores.dropna(subset=["target_value"])
    if scores.empty:
        return pd.DataFrame(columns=columns)
    merged = anchors.merge(scores.drop(columns=[target]), on="subject_id", how="inner")
    if merged.empty:
        return pd.DataFrame(columns=columns)
    merged["day_delta"] = (merged["clinical_date"] - merged["anchor_date"]).dt.days
    merged["abs_day_delta"] = merged["day_delta"].abs()
    if max_day_delta is not None:
        merged = merged[merged["abs_day_delta"].le(max_day_delta)].copy()
    if merged.empty:
        return pd.DataFrame(columns=columns)
    merged["_source_priority"] = merged["clinical_source"].map(_source_priority)
    merged = merged.sort_values(
        ["subject_id", "abs_day_delta", "_source_priority", "clinical_date"],
        na_position="last",
    )
    matched = merged.drop_duplicates("subject_id").copy()
    matched["target"] = target
    matched["target_mode"] = target_mode
    return matched[columns].reset_index(drop=True)


def _group_counts_for_matches(matches: pd.DataFrame) -> dict[str, int]:
    counts = matches["group"].astype(str).value_counts() if not matches.empty and "group" in matches.columns else pd.Series(dtype=int)
    return {f"n_{group}": int(counts.get(group, 0)) for group in GROUP_ORDER}


def _eligible_score_target(matches: pd.DataFrame) -> bool:
    if matches.empty or len(matches) < MIN_SCORE_TARGET_SUBJECTS:
        return False
    group_counts = matches["group"].astype(str).value_counts()
    return int((group_counts > 0).sum()) >= MIN_SCORE_TARGET_GROUPS


def _build_score_target_tables(paths: AnalysisPaths, feature_matrix: pd.DataFrame) -> dict[str, pd.DataFrame]:
    anchors = _dti_anchor_table(paths, feature_matrix)
    clinical = _read_clinical_score_rows(paths)
    audit_rows: list[dict] = []
    match_frames: list[pd.DataFrame] = []

    for target in AUDIT_SCORE_TARGETS:
        for window in CLINICAL_TARGET_WINDOWS_DAYS:
            matches = _nearest_score_matches(
                anchors,
                clinical,
                target,
                max_day_delta=window,
                target_mode=f"scan_aligned_{window}d",
            )
            audit_rows.append(
                {
                    "target": target,
                    "target_mode": f"scan_aligned_{window}d",
                    "max_day_delta": int(window),
                    "n_subjects": int(len(matches)),
                    "n_groups": int((matches["group"].astype(str).value_counts() > 0).sum()) if not matches.empty else 0,
                    "target_model_type": "regression" if target in SCORE_REGRESSION_TARGETS else ("classification" if target == CDR_CLASSIFICATION_TARGET else "audit_only"),
                    "eligible_for_primary_model": bool(target in ACTIVE_CLINICAL_TARGETS and window == PRIMARY_SCORE_WINDOW_DAYS and _eligible_score_target(matches)),
                    "eligible_for_sensitivity_model": False,
                    **_group_counts_for_matches(matches),
                }
            )
        primary = _nearest_score_matches(
            anchors,
            clinical,
            target,
            max_day_delta=PRIMARY_SCORE_WINDOW_DAYS,
            target_mode=f"primary_scan_aligned_{PRIMARY_SCORE_WINDOW_DAYS}d",
        )
        if not primary.empty:
            match_frames.append(primary)
        historical = _nearest_score_matches(
            anchors,
            clinical,
            target,
            max_day_delta=None,
            target_mode="sensitivity_all_available_history",
        )
        if not historical.empty:
            match_frames.append(historical)
        audit_rows.append(
            {
                "target": target,
                "target_mode": "sensitivity_all_available_history",
                "max_day_delta": np.nan,
                "n_subjects": int(len(historical)),
                "n_groups": int((historical["group"].astype(str).value_counts() > 0).sum()) if not historical.empty else 0,
                "eligible_for_primary_model": False,
                "target_model_type": "regression" if target in SCORE_REGRESSION_TARGETS else ("classification" if target == CDR_CLASSIFICATION_TARGET else "audit_only"),
                "eligible_for_sensitivity_model": bool(target in ACTIVE_CLINICAL_TARGETS and _eligible_score_target(historical)),
                **_group_counts_for_matches(historical),
            }
        )

    matches_all = pd.concat(match_frames, ignore_index=True) if match_frames else pd.DataFrame()
    audit = pd.DataFrame(audit_rows)
    return {"anchors": anchors, "clinical_rows": clinical, "target_audit": audit, "target_matches": matches_all}


def _score_k_options(n_features: int, n_subjects: int) -> list[int]:
    upper = max(2, min(n_features, n_subjects - 2))
    values = [min(upper, 60)]
    values = sorted(set(v for v in values if v >= 2))
    return values or [min(n_features, 2)]


def _score_regression_pipeline_specs(n_features: int, n_subjects: int) -> dict[str, tuple[Pipeline, dict]]:
    score_func = partial(mutual_info_regression, random_state=42)
    k_values = _score_k_options(n_features, n_subjects)
    specs: dict[str, tuple[Pipeline, dict]] = {
        "elastic_net_regression": (
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("select", SelectKBest(score_func=score_func, k=k_values[0])),
                    ("scaler", StandardScaler()),
                    ("model", ElasticNet(max_iter=8000, tol=1e-3, random_state=42)),
                ]
            ),
            {
                "select__k": k_values,
                "model__alpha": [0.2],
                "model__l1_ratio": [0.5],
            },
        )
    }
    if HAS_XGBOOST and XGBRegressor is not None:
        specs["xgboost_regression"] = (
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("select", SelectKBest(score_func=score_func, k=k_values[0])),
                    (
                        "model",
                        XGBRegressor(
                            objective="reg:squarederror",
                            eval_metric="rmse",
                            tree_method="hist",
                            n_estimators=120,
                            learning_rate=0.045,
                            subsample=0.85,
                            colsample_bytree=0.85,
                            random_state=42,
                            n_jobs=2,
                        ),
                    ),
                ]
            ),
            {
                "select__k": k_values,
                "model__max_depth": [3],
                "model__min_child_weight": [1],
            },
        )
    specs["extra_trees_regression"] = (
        Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("select", SelectKBest(score_func=score_func, k=k_values[0])),
                (
                    "model",
                    ExtraTreesRegressor(
                        n_estimators=120,
                        max_features="sqrt",
                        random_state=42,
                        n_jobs=2,
                    ),
                ),
            ]
        ),
        {
            "select__k": k_values,
            "model__min_samples_leaf": [2],
        },
    )
    specs["random_forest_regression"] = (
        Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("select", SelectKBest(score_func=score_func, k=k_values[0])),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=120,
                        max_features="sqrt",
                        random_state=42,
                        n_jobs=2,
                    ),
                ),
            ]
        ),
        {
            "select__k": k_values,
            "model__min_samples_leaf": [2],
        },
    )
    specs["hist_gradient_boosting_regression"] = (
        Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("select", SelectKBest(score_func=score_func, k=k_values[0])),
                (
                    "model",
                    HistGradientBoostingRegressor(
                        learning_rate=0.035,
                        max_leaf_nodes=15,
                        l2_regularization=0.05,
                        random_state=42,
                    ),
                ),
            ]
        ),
        {
            "select__k": k_values,
        },
    )
    if n_subjects >= MIN_NEURAL_SCORE_SUBJECTS:
        specs["mlp_regression"] = (
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("select", SelectKBest(score_func=score_func, k=k_values[0])),
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        MLPRegressor(
                            hidden_layer_sizes=(48, 12),
                            activation="relu",
                            early_stopping=True,
                            max_iter=220,
                            random_state=42,
                        ),
                    ),
                ]
            ),
            {
                "select__k": k_values,
                "model__alpha": [1e-2],
                "model__learning_rate_init": [1e-3],
            },
        )
        specs["mlp_augmented_regression"] = (
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("select", SelectKBest(score_func=score_func, k=k_values[0])),
                    ("scaler", StandardScaler()),
                    ("model", NoiseAugmentedMLPRegressor(random_state=42)),
                ]
            ),
            {
                "select__k": k_values,
                "model__alpha": [2e-2],
                "model__learning_rate_init": [1e-3],
                "model__noise_sd": [0.03],
            },
        )
    return specs


def _selected_regression_features(pipeline: Pipeline, feature_names: list[str]) -> list[str]:
    return _selected_features(pipeline, feature_names)


def _regression_model_importance(pipeline: Pipeline, selected_names: list[str], model_name: str) -> dict[str, float]:
    model = pipeline.named_steps.get("model")
    if model is None:
        return {}
    if model_name == "elastic_net_regression" and hasattr(model, "coef_"):
        values = np.abs(np.asarray(model.coef_, dtype=float))
    elif hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float)
    else:
        return {}
    return {feature: float(value) for feature, value in zip(selected_names, values)}


def _xgb_regression_shap_fold(pipeline: Pipeline, x_test: pd.DataFrame, selected_names: list[str]) -> dict[str, float]:
    if not HAS_XGBOOST or DMatrix is None or "xgboost" not in str(type(pipeline.named_steps.get("model"))).lower():
        return {}
    try:
        imputed = pipeline.named_steps["imputer"].transform(x_test)
        selected = pipeline.named_steps["select"].transform(imputed)
        booster = pipeline.named_steps["model"].get_booster()
        contrib = booster.predict(DMatrix(selected), pred_contribs=True)
    except Exception:
        return {}
    arr = np.asarray(contrib, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 2:
        return {}
    values = np.nanmean(np.abs(arr[:, :-1]), axis=0)
    return {feature: float(value) for feature, value in zip(selected_names, values)}


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    out = {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "median_ae": float(median_absolute_error(y_true, y_pred)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else np.nan,
        "explained_variance": float(explained_variance_score(y_true, y_pred)) if len(y_true) >= 2 else np.nan,
        "bias_mean_error": float(np.mean(y_pred - y_true)),
        "residual_sd": float(np.std(y_true - y_pred, ddof=1)) if len(y_true) >= 2 else np.nan,
        "spearman_r": np.nan,
        "spearman_p": np.nan,
        "pearson_r": np.nan,
        "pearson_p": np.nan,
        "calibration_slope": np.nan,
        "calibration_intercept": np.nan,
    }
    if len(y_true) >= 3 and len(np.unique(y_true)) > 1 and len(np.unique(y_pred)) > 1:
        rho, pvalue = stats.spearmanr(y_true, y_pred)
        out["spearman_r"] = float(rho)
        out["spearman_p"] = float(pvalue)
        pearson_r, pearson_p = stats.pearsonr(y_true, y_pred)
        out["pearson_r"] = float(pearson_r)
        out["pearson_p"] = float(pearson_p)
        slope, intercept = np.polyfit(y_pred, y_true, 1)
        out["calibration_slope"] = float(slope)
        out["calibration_intercept"] = float(intercept)
    return out


def _score_model_status_rows() -> pd.DataFrame:
    rows = []
    rows.append({"model": "elastic_net_regression", "status": "run", "reason": "sklearn regularized linear regression baseline"})
    rows.append(
        {
            "model": "xgboost_regression",
            "status": "run" if HAS_XGBOOST and XGBRegressor is not None else "unavailable_optional_dependency",
            "reason": "xgboost installed" if HAS_XGBOOST and XGBRegressor is not None else "xgboost is not installed in the app environment",
        }
    )
    rows.append({"model": "extra_trees_regression", "status": "run", "reason": "sklearn tree ensemble regression baseline"})
    rows.append({"model": "random_forest_regression", "status": "run", "reason": "sklearn bagged tree regression baseline"})
    rows.append({"model": "hist_gradient_boosting_regression", "status": "run", "reason": "sklearn histogram gradient boosting regression"})
    rows.append(
        {
            "model": "mlp_regression",
            "status": "available_conditionally",
            "reason": f"runs only when a target mode has at least {MIN_NEURAL_SCORE_SUBJECTS} labeled subjects",
        }
    )
    rows.append(
        {
            "model": "mlp_augmented_regression",
            "status": "available_conditionally",
            "reason": "training-fold-only Gaussian feature augmentation around a sklearn MLP regressor",
        }
    )
    optional_modules = {
        "ft_transformer": ("rtdl", "requires a tabular transformer package such as rtdl/rtdl-num-embeddings"),
        "tabm": ("tabm", "requires a TabM implementation package"),
        "tabpfn_regressor": ("tabpfn", "requires the TabPFN package and local model availability"),
        "tabnet_regressor": ("pytorch_tabnet", "requires pytorch-tabnet and torch in the app environment"),
    }
    for model, (module, reason) in optional_modules.items():
        available = importlib.util.find_spec(module) is not None
        rows.append(
            {
                "model": model,
                "status": "available_not_wired" if available else "unavailable_optional_dependency",
                "reason": "optional dependency is installed but not enabled in this lightweight dashboard runner" if available else reason,
            }
        )
    return pd.DataFrame(rows)


def _disease_model_status_rows(n_subjects: int) -> pd.DataFrame:
    rows = [
        {"model": "multinomial_elastic_net", "status": "run", "reason": "sklearn linear probability baseline"},
        {
            "model": "xgboost_multiclass",
            "status": "run" if HAS_XGBOOST and XGBClassifier is not None else "unavailable_optional_dependency",
            "reason": "xgboost installed" if HAS_XGBOOST and XGBClassifier is not None else "xgboost is not installed in the app environment",
        },
        {"model": "extra_trees_multiclass", "status": "run", "reason": "sklearn tree ensemble baseline"},
        {"model": "random_forest_multiclass", "status": "run", "reason": "sklearn bagged tree ensemble baseline"},
        {
            "model": "mlp_multiclass",
            "status": "run" if n_subjects >= MIN_NEURAL_CLASSIFIER_SUBJECTS else "skipped_insufficient_subjects",
            "reason": f"requires at least {MIN_NEURAL_CLASSIFIER_SUBJECTS} subjects for neural classifier benchmark",
        },
    ]
    optional_modules = {
        "ft_transformer_classifier": ("rtdl", "requires a tabular transformer package such as rtdl/rtdl-num-embeddings"),
        "tabm_classifier": ("tabm", "requires a TabM implementation package"),
        "tabpfn_classifier": ("tabpfn", "requires the TabPFN package and local model availability"),
        "tabnet_classifier": ("pytorch_tabnet", "requires pytorch-tabnet and torch in the app environment"),
    }
    for model, (module, reason) in optional_modules.items():
        available = importlib.util.find_spec(module) is not None
        rows.append(
            {
                "model": model,
                "status": "available_not_wired" if available else "unavailable_optional_dependency",
                "reason": "optional dependency is installed but not enabled in this lightweight dashboard runner" if available else reason,
            }
        )
    return pd.DataFrame(rows)


def _score_outlier_filter(task: pd.DataFrame, target: str, target_mode: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    work = task.copy()
    work["_score_outlier"] = False
    if "group" not in work.columns or "target_value" not in work.columns:
        return work.drop(columns=["_score_outlier"], errors="ignore"), pd.DataFrame()
    for group, sub in work.groupby("group", dropna=False):
        values = pd.to_numeric(sub["target_value"], errors="coerce").dropna()
        if values.empty:
            lower = upper = q1 = q3 = iqr = np.nan
        else:
            q1 = float(values.quantile(0.25))
            q3 = float(values.quantile(0.75))
            iqr = float(q3 - q1)
            lower = float(q1 - 1.5 * iqr)
            upper = float(q3 + 1.5 * iqr)
        for idx in sub.index:
            value = pd.to_numeric(pd.Series([work.loc[idx, "target_value"]]), errors="coerce").iloc[0]
            is_outlier = bool(pd.notna(value) and pd.notna(lower) and pd.notna(upper) and (value < lower or value > upper))
            work.loc[idx, "_score_outlier"] = is_outlier
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
                    "removed_as_outlier": is_outlier,
                }
            )
    clean = work[~work["_score_outlier"].astype(bool)].drop(columns=["_score_outlier"], errors="ignore").reset_index(drop=True)
    return clean, pd.DataFrame(rows)


def _run_score_prediction_models(
    matrix: pd.DataFrame,
    numeric: pd.DataFrame,
    feature_names: list[str],
    catalog: pd.DataFrame,
    target_matches: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    empty = pd.DataFrame()
    if target_matches.empty:
        return {
            "score_performance": empty,
            "score_predictions": empty,
            "score_importance": empty,
            "score_residuals": empty,
            "score_group_performance": empty,
            "score_fold_params": empty,
            "score_mixed_models": empty,
            "score_outlier_audit": empty,
            "score_model_status": _score_model_status_rows(),
        }

    numeric_with_subject = pd.concat(
        [matrix[["subject_id", "group"]].reset_index(drop=True), numeric[feature_names].reset_index(drop=True)],
        axis=1,
    )
    performance_rows: list[dict] = []
    prediction_rows: list[dict] = []
    fold_param_rows: list[dict] = []
    importance_rows: list[dict] = []
    outlier_frames: list[pd.DataFrame] = []

    train_modes = {
        f"primary_scan_aligned_{PRIMARY_SCORE_WINDOW_DAYS}d": "primary",
        "sensitivity_all_available_history": "sensitivity",
    }
    for target in SCORE_REGRESSION_TARGETS:
        for target_mode, analysis_role in train_modes.items():
            matches = target_matches[
                target_matches["target"].astype(str).eq(target)
                & target_matches["target_mode"].astype(str).eq(target_mode)
            ].copy()
            matches["target_value"] = pd.to_numeric(matches.get("target_value"), errors="coerce")
            matches = matches.dropna(subset=["target_value"])
            task = matches.merge(numeric_with_subject.drop(columns=["group"]), on="subject_id", how="inner")
            task = task.dropna(subset=["target_value"]).reset_index(drop=True)
            raw_n_subjects = int(len(task))
            task, outlier_audit = _score_outlier_filter(task, target, target_mode)
            if not outlier_audit.empty:
                outlier_audit["analysis_role"] = analysis_role
                outlier_frames.append(outlier_audit)
            n_outliers_removed = int(raw_n_subjects - len(task))
            group_counts = task["group"].astype(str).value_counts() if "group" in task.columns else pd.Series(dtype=int)
            eligible = len(task) >= MIN_SCORE_TARGET_SUBJECTS and int((group_counts > 0).sum()) >= MIN_SCORE_TARGET_GROUPS
            if not eligible:
                performance_rows.append(
                    {
                        "target": target,
                        "target_mode": target_mode,
                        "analysis_role": analysis_role,
                        "model": "not_run",
                        "status": "skipped_insufficient_labels",
                        "n_subjects": int(len(task)),
                        "n_subjects_before_outlier_filter": raw_n_subjects,
                        "n_outliers_removed": n_outliers_removed,
                        "n_groups": int((group_counts > 0).sum()),
                        "n_features_input": int(len(feature_names)),
                        "mae": np.nan,
                        "rmse": np.nan,
                        "r2": np.nan,
                        "spearman_r": np.nan,
                        "spearman_p": np.nan,
                    }
                )
                continue

            x = task[feature_names]
            y = task["target_value"].to_numpy(dtype=float)
            outer_splits = min(3, max(3, len(task) // 40))
            inner_splits = min(3, max(2, outer_splits - 1))
            outer = KFold(n_splits=outer_splits, shuffle=True, random_state=42)
            model_specs = _score_regression_pipeline_specs(len(feature_names), len(task))
            target_importance: dict[tuple[str, str], Counter] = defaultdict(Counter)
            selected_counts: dict[str, Counter] = defaultdict(Counter)

            for model_name, (pipe, grid) in model_specs.items():
                for fold, (train_idx, test_idx) in enumerate(outer.split(x, y), start=1):
                    inner = KFold(n_splits=inner_splits, shuffle=True, random_state=420 + fold)
                    search = GridSearchCV(
                        pipe,
                        grid,
                        scoring="neg_mean_absolute_error",
                        cv=inner,
                        n_jobs=2,
                        pre_dispatch=2,
                        refit=True,
                    )
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", category=ConvergenceWarning)
                        warnings.filterwarnings("ignore", category=UserWarning)
                        search.fit(x.iloc[train_idx], y[train_idx])
                    best = search.best_estimator_
                    selected = _selected_regression_features(best, feature_names)
                    selected_counts[model_name].update(selected)
                    importances = _regression_model_importance(best, selected, model_name)
                    method = "abs_elastic_net_coef" if model_name == "elastic_net_regression" else "tree_feature_importance"
                    target_importance[(model_name, method)].update(importances)
                    shap_values = _xgb_regression_shap_fold(best, x.iloc[test_idx], selected)
                    target_importance[(model_name, "xgboost_treeshap_abs_contribution")].update(shap_values)
                    pred = best.predict(x.iloc[test_idx])
                    fold_param_rows.append(
                        {
                            "target": target,
                            "target_mode": target_mode,
                            "analysis_role": analysis_role,
                            "model": model_name,
                            "fold": fold,
                            "best_inner_neg_mae": float(search.best_score_),
                            "best_params": str(search.best_params_),
                            "n_selected_features": int(len(selected)),
                        }
                    )
                    for row_pos, idx in enumerate(test_idx):
                        prediction_rows.append(
                            {
                                "target": target,
                                "target_mode": target_mode,
                                "analysis_role": analysis_role,
                                "model": model_name,
                                "fold": fold,
                                "subject_id": task.iloc[idx]["subject_id"],
                                "group": task.iloc[idx]["group"],
                                "true_score": float(y[idx]),
                                "predicted_score": float(pred[row_pos]),
                                "residual": float(y[idx] - pred[row_pos]),
                                "abs_day_delta": task.iloc[idx].get("abs_day_delta", np.nan),
                                "clinical_source": task.iloc[idx].get("clinical_source", ""),
                            }
                        )

                model_pred = pd.DataFrame(prediction_rows)
                model_pred = model_pred[
                    model_pred["target"].astype(str).eq(target)
                    & model_pred["target_mode"].astype(str).eq(target_mode)
                    & model_pred["model"].astype(str).eq(model_name)
                ].copy()
                metrics = _regression_metrics(
                    model_pred["true_score"].to_numpy(dtype=float),
                    model_pred["predicted_score"].to_numpy(dtype=float),
                )
                performance_rows.append(
                    {
                        "target": target,
                        "target_mode": target_mode,
                        "analysis_role": analysis_role,
                        "model": model_name,
                        "status": "ok",
                        "n_subjects": int(len(model_pred)),
                        "n_subjects_before_outlier_filter": raw_n_subjects,
                        "n_outliers_removed": n_outliers_removed,
                        "n_groups": int((model_pred["group"].astype(str).value_counts() > 0).sum()),
                        "n_features_input": int(len(feature_names)),
                        "n_outer_folds": int(outer_splits),
                        "n_inner_folds": int(inner_splits),
                        **metrics,
                    }
                )

            for (model_name, method), counter in target_importance.items():
                for rank, (feature, value) in enumerate(counter.most_common(), start=1):
                    importance_rows.append(
                        {
                            "target": target,
                            "target_mode": target_mode,
                            "analysis_role": analysis_role,
                            "model": model_name,
                            "importance_method": method,
                            "rank": rank,
                            "feature": feature,
                            "feature_group": _feature_group_for(catalog, feature),
                            "importance": float(value / max(outer_splits, 1)),
                            "selected_fold_count": int(selected_counts[model_name][feature]),
                        }
                    )

    predictions = pd.DataFrame(prediction_rows)
    residuals = predictions.copy()
    group_rows: list[dict] = []
    if not predictions.empty:
        for (target, target_mode, model, group), sub in predictions.groupby(["target", "target_mode", "model", "group"], dropna=False):
            metrics = _regression_metrics(sub["true_score"].to_numpy(dtype=float), sub["predicted_score"].to_numpy(dtype=float))
            group_rows.append(
                {
                    "target": target,
                    "target_mode": target_mode,
                    "model": model,
                    "group": group,
                    "n_subjects": int(len(sub)),
                    **metrics,
                }
            )
    mixed_models = _run_score_prediction_mixed_models(predictions)
    return {
        "score_performance": pd.DataFrame(performance_rows),
        "score_predictions": predictions,
        "score_importance": pd.DataFrame(importance_rows),
        "score_residuals": residuals,
        "score_group_performance": pd.DataFrame(group_rows),
        "score_fold_params": pd.DataFrame(fold_param_rows),
        "score_mixed_models": mixed_models,
        "score_outlier_audit": pd.concat(outlier_frames, ignore_index=True) if outlier_frames else pd.DataFrame(),
        "score_model_status": _score_model_status_rows(),
    }


def _cdr_class(value: object) -> str | None:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return None
    if numeric <= 0:
        return "0"
    if numeric <= 0.5:
        return "0.5"
    if numeric <= 1:
        return "1"
    return "2+"


def _cdr_classification_model_status_rows(n_subjects: int) -> pd.DataFrame:
    status = _disease_model_status_rows(n_subjects).copy()
    if not status.empty:
        status["target"] = CDR_CLASSIFICATION_TARGET
        status["task_type"] = "ordinal_multiclass_classification"
    return status


def _run_cdr_classification_models(
    matrix: pd.DataFrame,
    numeric: pd.DataFrame,
    feature_names: list[str],
    catalog: pd.DataFrame,
    target_matches: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    empty = pd.DataFrame()
    if target_matches.empty:
        return {
            "cdr_performance": empty,
            "cdr_predictions": empty,
            "cdr_confusion": empty,
            "cdr_class_metrics": empty,
            "cdr_importance": empty,
            "cdr_fold_params": empty,
            "cdr_model_status": _cdr_classification_model_status_rows(0),
        }

    numeric_with_subject = pd.concat(
        [matrix[["subject_id", "group"]].reset_index(drop=True), numeric[feature_names].reset_index(drop=True)],
        axis=1,
    )
    train_modes = {
        f"primary_scan_aligned_{PRIMARY_SCORE_WINDOW_DAYS}d": "primary",
        "sensitivity_all_available_history": "sensitivity",
    }
    performance_rows: list[dict] = []
    prediction_rows: list[dict] = []
    fold_param_rows: list[dict] = []
    importance_rows: list[dict] = []

    for target_mode, analysis_role in train_modes.items():
        matches = target_matches[
            target_matches["target"].astype(str).eq(CDR_CLASSIFICATION_TARGET)
            & target_matches["target_mode"].astype(str).eq(target_mode)
        ].copy()
        matches["target_value"] = pd.to_numeric(matches.get("target_value"), errors="coerce")
        matches = matches.dropna(subset=["target_value"])
        matches["cdr_class"] = matches["target_value"].map(_cdr_class)
        matches = matches.dropna(subset=["cdr_class"])
        task = matches.merge(numeric_with_subject.drop(columns=["group"]), on="subject_id", how="inner")
        task = task.dropna(subset=["cdr_class"]).reset_index(drop=True)
        group_counts = task["group"].astype(str).value_counts() if "group" in task.columns else pd.Series(dtype=int)
        class_counts = task["cdr_class"].astype(str).value_counts()
        class_labels = [label for label in CDR_CLASS_ORDER if int(class_counts.get(label, 0)) > 0]
        min_class = int(class_counts.min()) if not class_counts.empty else 0
        eligible = (
            len(task) >= MIN_SCORE_TARGET_SUBJECTS
            and int((group_counts > 0).sum()) >= MIN_SCORE_TARGET_GROUPS
            and len(class_labels) >= 2
            and min_class >= 3
        )
        if not eligible:
            performance_rows.append(
                {
                    "target": CDR_CLASSIFICATION_TARGET,
                    "target_mode": target_mode,
                    "analysis_role": analysis_role,
                    "model": "not_run",
                    "status": "skipped_insufficient_classes_or_labels",
                    "n_subjects": int(len(task)),
                    "n_groups": int((group_counts > 0).sum()),
                    "n_classes": int(len(class_labels)),
                    "min_class_count": min_class,
                    "n_features_input": int(len(feature_names)),
                    "accuracy": np.nan,
                    "balanced_accuracy": np.nan,
                    "macro_f1": np.nan,
                    "macro_precision": np.nan,
                    "macro_recall_sensitivity": np.nan,
                    "macro_specificity": np.nan,
                    "macro_auroc_ovr": np.nan,
                }
            )
            continue

        class_to_int = {label: idx for idx, label in enumerate(class_labels)}
        int_to_class = {idx: label for label, idx in class_to_int.items()}
        x = task[feature_names]
        y = task["cdr_class"].map(class_to_int).to_numpy(dtype=int)
        labels_int = list(range(len(class_labels)))
        outer_splits = min(3, min_class)
        inner_splits = min(3, max(2, min_class - 1))
        outer = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=42)
        model_specs = _pipeline_specs(len(feature_names), len(task), n_classes=len(class_labels))
        target_importance: dict[tuple[str, str], Counter] = defaultdict(Counter)
        selected_counts: dict[str, Counter] = defaultdict(Counter)

        for model_name, (pipe, grid) in model_specs.items():
            for fold, (train_idx, test_idx) in enumerate(outer.split(x, y), start=1):
                inner = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=420 + fold)
                search = GridSearchCV(
                    pipe,
                    grid,
                    scoring="balanced_accuracy",
                    cv=inner,
                    n_jobs=2,
                    pre_dispatch=2,
                    refit=True,
                )
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=ConvergenceWarning)
                    warnings.filterwarnings("ignore", category=UserWarning)
                    search.fit(x.iloc[train_idx], y[train_idx])
                best = search.best_estimator_
                selected = _selected_features(best, feature_names)
                selected_counts[model_name].update(selected)
                importances = _model_importance(best, selected, model_name)
                method = "abs_elastic_net_coef" if model_name == "multinomial_elastic_net" else "tree_feature_importance"
                target_importance[(model_name, method)].update(importances)
                shap_values = _xgb_shap_fold(best, x.iloc[test_idx], selected)
                target_importance[(model_name, "xgboost_treeshap_abs_contribution")].update(shap_values)
                y_pred = best.predict(x.iloc[test_idx])
                prob = np.full((len(test_idx), len(class_labels)), np.nan)
                if hasattr(best, "predict_proba"):
                    try:
                        raw_prob = np.asarray(best.predict_proba(x.iloc[test_idx]), dtype=float)
                        model_classes = getattr(best.named_steps.get("model"), "classes_", labels_int)
                        for col_idx, class_idx in enumerate(model_classes):
                            if int(class_idx) in labels_int and col_idx < raw_prob.shape[1]:
                                prob[:, int(class_idx)] = raw_prob[:, col_idx]
                    except Exception:
                        pass
                fold_param_rows.append(
                    {
                        "target": CDR_CLASSIFICATION_TARGET,
                        "target_mode": target_mode,
                        "analysis_role": analysis_role,
                        "model": model_name,
                        "fold": fold,
                        "best_inner_balanced_accuracy": float(search.best_score_),
                        "best_params": str(search.best_params_),
                        "n_selected_features": int(len(selected)),
                    }
                )
                for row_pos, idx in enumerate(test_idx):
                    row = {
                        "target": CDR_CLASSIFICATION_TARGET,
                        "target_mode": target_mode,
                        "analysis_role": analysis_role,
                        "model": model_name,
                        "fold": fold,
                        "subject_id": task.iloc[idx]["subject_id"],
                        "group": task.iloc[idx]["group"],
                        "true_class": int_to_class[int(y[idx])],
                        "predicted_class": int_to_class[int(y_pred[row_pos])],
                        "true_cdr_value": float(task.iloc[idx]["target_value"]),
                        "abs_day_delta": task.iloc[idx].get("abs_day_delta", np.nan),
                        "clinical_source": task.iloc[idx].get("clinical_source", ""),
                    }
                    for class_idx, class_name in int_to_class.items():
                        row[f"prob_{class_name}"] = float(prob[row_pos, class_idx])
                    prediction_rows.append(row)

            model_pred = pd.DataFrame(prediction_rows)
            model_pred = model_pred[
                model_pred["target"].astype(str).eq(CDR_CLASSIFICATION_TARGET)
                & model_pred["target_mode"].astype(str).eq(target_mode)
                & model_pred["model"].astype(str).eq(model_name)
            ].copy()
            if model_pred.empty:
                continue
            y_true = model_pred["true_class"].map(class_to_int).to_numpy(dtype=int)
            y_hat = model_pred["predicted_class"].map(class_to_int).to_numpy(dtype=int)
            prob_cols = [f"prob_{label}" for label in class_labels]
            prob = model_pred[prob_cols].to_numpy(dtype=float) if set(prob_cols).issubset(model_pred.columns) else np.full((len(model_pred), len(class_labels)), np.nan)
            macro_auroc = np.nan
            if np.isfinite(prob).all():
                try:
                    macro_auroc = float(roc_auc_score(y_true, prob, multi_class="ovr", average="macro", labels=labels_int))
                except Exception:
                    pass
            loss = np.nan
            if np.isfinite(prob).all():
                try:
                    loss = float(log_loss(y_true, prob, labels=labels_int))
                except Exception:
                    pass
            specificity = _specificity_by_class(y_true, y_hat, labels_int)
            macro_specificity = float(np.nanmean([specificity.get(label, np.nan) for label in labels_int]))
            performance_rows.append(
                {
                    "target": CDR_CLASSIFICATION_TARGET,
                    "target_mode": target_mode,
                    "analysis_role": analysis_role,
                    "model": model_name,
                    "status": "ok",
                    "n_subjects": int(len(model_pred)),
                    "n_groups": int((model_pred["group"].astype(str).value_counts() > 0).sum()),
                    "n_classes": int(len(class_labels)),
                    "min_class_count": min_class,
                    "n_features_input": int(len(feature_names)),
                    "n_outer_folds": int(outer_splits),
                    "n_inner_folds": int(inner_splits),
                    "accuracy": float(accuracy_score(y_true, y_hat)),
                    "balanced_accuracy": float(balanced_accuracy_score(y_true, y_hat)),
                    "macro_auroc_ovr": macro_auroc,
                    "log_loss": loss,
                    "macro_f1": float(f1_score(y_true, y_hat, average="macro")),
                    "weighted_f1": float(f1_score(y_true, y_hat, average="weighted")),
                    "macro_precision": float(precision_score(y_true, y_hat, average="macro", zero_division=0)),
                    "macro_recall_sensitivity": float(recall_score(y_true, y_hat, average="macro", zero_division=0)),
                    "macro_specificity": macro_specificity,
                }
            )

        for (model_name, method), counter in target_importance.items():
            for rank, (feature, value) in enumerate(counter.most_common(), start=1):
                importance_rows.append(
                    {
                        "target": CDR_CLASSIFICATION_TARGET,
                        "target_mode": target_mode,
                        "analysis_role": analysis_role,
                        "model": model_name,
                        "importance_method": method,
                        "rank": rank,
                        "feature": feature,
                        "feature_group": _feature_group_for(catalog, feature),
                        "importance": float(value / max(outer_splits, 1)),
                        "selected_fold_count": int(selected_counts[model_name][feature]),
                    }
                )

    predictions = pd.DataFrame(prediction_rows)
    class_metric_rows: list[dict] = []
    confusion_rows: list[dict] = []
    if not predictions.empty:
        for (target_mode, model), sub in predictions.groupby(["target_mode", "model"], dropna=False):
            present_labels = [label for label in CDR_CLASS_ORDER if label in set(sub["true_class"].astype(str)) or label in set(sub["predicted_class"].astype(str))]
            class_to_int = {label: idx for idx, label in enumerate(present_labels)}
            y_true = sub["true_class"].map(class_to_int).to_numpy(dtype=int)
            y_hat = sub["predicted_class"].map(class_to_int).to_numpy(dtype=int)
            labels_int = list(range(len(present_labels)))
            precision, recall, f1_vals, support = precision_recall_fscore_support(y_true, y_hat, labels=labels_int, zero_division=0)
            specificity = _specificity_by_class(y_true, y_hat, labels_int)
            for idx, label in enumerate(present_labels):
                class_metric_rows.append(
                    {
                        "target": CDR_CLASSIFICATION_TARGET,
                        "target_mode": target_mode,
                        "model": model,
                        "class": label,
                        "support": int(support[idx]),
                        "precision": float(precision[idx]),
                        "recall_sensitivity": float(recall[idx]),
                        "specificity": float(specificity.get(idx, np.nan)),
                        "f1": float(f1_vals[idx]),
                    }
                )
            cm = confusion_matrix(y_true, y_hat, labels=labels_int)
            for i, true_class in enumerate(present_labels):
                for j, pred_class in enumerate(present_labels):
                    confusion_rows.append(
                        {
                            "target": CDR_CLASSIFICATION_TARGET,
                            "target_mode": target_mode,
                            "model": model,
                            "true_class": true_class,
                            "predicted_class": pred_class,
                            "n": int(cm[i, j]),
                        }
                    )

    return {
        "cdr_performance": pd.DataFrame(performance_rows),
        "cdr_predictions": predictions,
        "cdr_confusion": pd.DataFrame(confusion_rows),
        "cdr_class_metrics": pd.DataFrame(class_metric_rows),
        "cdr_importance": pd.DataFrame(importance_rows),
        "cdr_fold_params": pd.DataFrame(fold_param_rows),
        "cdr_model_status": _cdr_classification_model_status_rows(len(target_matches)),
    }


def _run_score_prediction_mixed_models(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()
    rows: list[dict] = []
    if not HAS_STATSMODELS or smf is None:
        return pd.DataFrame(
            [
                {
                    "target": "",
                    "target_mode": "",
                    "model": "",
                    "status": "skipped_statsmodels_unavailable",
                    "term": "",
                    "coef": np.nan,
                    "p_value": np.nan,
                }
            ]
        )
    for (target, target_mode, model), sub in predictions.groupby(["target", "target_mode", "model"], dropna=False):
        work = sub[["subject_id", "true_score", "predicted_score"]].copy()
        work["true_score"] = pd.to_numeric(work["true_score"], errors="coerce")
        work["predicted_score"] = pd.to_numeric(work["predicted_score"], errors="coerce")
        work = work.replace([np.inf, -np.inf], np.nan).dropna()
        if len(work) < 10 or work["predicted_score"].nunique() < 2:
            rows.append(
                {
                    "target": target,
                    "target_mode": target_mode,
                    "model": model,
                    "status": "skipped_insufficient_variation",
                    "method": "",
                    "term": "",
                    "coef": np.nan,
                    "p_value": np.nan,
                }
            )
            continue
        repeated = work["subject_id"].duplicated().any()
        try:
            if repeated:
                fit = smf.mixedlm("true_score ~ predicted_score", work, groups=work["subject_id"]).fit(reml=False, method="lbfgs", disp=False)
                method = "mixedlm_subject_random_intercept"
            else:
                fit = smf.ols("true_score ~ predicted_score", data=work).fit(cov_type="HC3")
                method = "robust_ols_no_repeated_subjects"
            for term, coef in fit.params.items():
                rows.append(
                    {
                        "target": target,
                        "target_mode": target_mode,
                        "model": model,
                        "status": "ok",
                        "method": method,
                        "term": term,
                        "coef": float(coef),
                        "p_value": float(fit.pvalues.get(term, np.nan)),
                    }
                )
        except Exception as exc:
            rows.append(
                {
                    "target": target,
                    "target_mode": target_mode,
                    "model": model,
                    "status": f"failed: {exc}",
                    "method": "",
                    "term": "",
                    "coef": np.nan,
                    "p_value": np.nan,
                }
            )
    return pd.DataFrame(rows)


def _mci_stage(value_mmse: object, value_cdr: object) -> str:
    mmse = pd.to_numeric(pd.Series([value_mmse]), errors="coerce").iloc[0]
    cdr = pd.to_numeric(pd.Series([value_cdr]), errors="coerce").iloc[0]
    if pd.notna(cdr) and cdr >= 1:
        return "MCI high severity"
    if pd.notna(mmse) and mmse < 24:
        return "MCI high severity"
    if pd.notna(mmse) and mmse >= 27 and (pd.isna(cdr) or cdr <= 0.5):
        return "MCI low severity"
    if (pd.notna(mmse) and 24 <= mmse <= 26) or (pd.notna(cdr) and cdr == 0.5):
        return "MCI intermediate severity"
    return "MCI unstaged"


def _score_mci_stage_tables(target_matches: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if target_matches.empty:
        return pd.DataFrame(), pd.DataFrame()
    primary_mode = f"primary_scan_aligned_{PRIMARY_SCORE_WINDOW_DAYS}d"
    sub = target_matches[target_matches["target_mode"].astype(str).eq(primary_mode)].copy()
    sub = sub[sub["target"].isin(["MMSE Total Score", "Global CDR"])]
    if sub.empty:
        return pd.DataFrame(), pd.DataFrame()
    wide = sub.pivot_table(index=["subject_id", "group"], columns="target", values="target_value", aggfunc="first").reset_index()
    wide = wide[wide["group"].astype(str).eq("MCI")].copy()
    if wide.empty:
        return pd.DataFrame(), pd.DataFrame()
    wide["mci_stage"] = [_mci_stage(row.get("MMSE Total Score"), row.get("Global CDR")) for _, row in wide.iterrows()]
    summary = (
        wide.groupby("mci_stage", dropna=False)
        .agg(
            n_subjects=("subject_id", "nunique"),
            median_mmse=("MMSE Total Score", "median"),
            median_global_cdr=("Global CDR", "median"),
        )
        .reset_index()
    )
    return wide, summary


def _feature_group_for(catalog: pd.DataFrame, feature: str) -> str:
    match = catalog.loc[catalog["feature"].astype(str).eq(str(feature)), "feature_group"]
    return str(match.iloc[0]) if not match.empty else "Unknown"


def _feature_group_summary(catalog: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group, sub in catalog.groupby("feature_group", dropna=False):
        statuses = sub["final_status"].fillna("unknown").astype(str)
        reasons = sub["drop_reason"].fillna("").astype(str)
        selected = int(statuses.eq("selected_in_cv").sum())
        dropped = int(statuses.str.startswith("dropped").sum())
        reason_counts = Counter(r for r in reasons if r)
        reason_text = "; ".join(f"{reason} ({count})" for reason, count in reason_counts.most_common(4))
        if not reason_text and selected:
            reason_text = "selected repeatedly inside cross-validation"
        elif not reason_text:
            reason_text = "retained as candidate or not selected in CV"
        rows.append(
            {
                "feature_group": group,
                "features": int(len(sub)),
                "selected": selected,
                "dropped": dropped,
                "reason": reason_text,
            }
        )
    return pd.DataFrame(rows).sort_values(["selected", "features"], ascending=[False, False]).reset_index(drop=True)


def _preprocessing_audit(catalog: pd.DataFrame) -> pd.DataFrame:
    statuses = catalog.get("final_status", pd.Series(dtype=str)).fillna("unknown").astype(str)
    rows = [
        {
            "step": "feature_scope",
            "policy": "structural_connectome_only",
            "rule": "Use structural-connectome features only; exclude age, sex, phase, APOE, demographics, diagnosis labels, clinical scores, and group inference statistics as predictors.",
            "threshold": "",
            "n_features": int(len(catalog)),
        },
        {
            "step": "missingness_filter",
            "policy": "unsupervised_prefilter",
            "rule": "Drop features with excessive missingness before supervised modeling.",
            "threshold": f"missingness <= {MISSINGNESS_DROP_THRESHOLD:.2f}",
            "n_features": int((~statuses.eq("dropped_missingness")).sum()),
        },
        {
            "step": "sparse_presence_filter",
            "policy": "unsupervised_prefilter",
            "rule": "For sparse AAL/node/edge features, require enough non-zero presence across subjects.",
            "threshold": f"presence >= {SPARSE_PRESENCE_THRESHOLD:.2f}",
            "n_features": int((~statuses.eq("dropped_sparse")).sum()),
        },
        {
            "step": "variance_filter",
            "policy": "unsupervised_prefilter",
            "rule": "Drop constant and near-zero variance features.",
            "threshold": f"variance >= {NEAR_ZERO_VARIANCE_THRESHOLD:g} and unique values > 1",
            "n_features": int((~statuses.eq("dropped_low_variance")).sum()),
        },
        {
            "step": "correlation_filter",
            "policy": "unsupervised_prefilter",
            "rule": "Within highly correlated feature clusters, keep the lower-missingness/higher-priority representative.",
            "threshold": f"Spearman |r| < {CORRELATION_DROP_THRESHOLD:.2f} between retained representatives",
            "n_features": int((~statuses.eq("dropped_correlated")).sum()),
        },
        {
            "step": "imputation",
            "policy": "inside_cv_pipeline",
            "rule": "Median imputation is fit on each training fold and applied to the held-out fold.",
            "threshold": "median",
            "n_features": int(statuses.eq("retained_for_model").sum() + statuses.eq("selected_in_cv").sum()),
        },
        {
            "step": "supervised_selection",
            "policy": "inside_nested_cv",
            "rule": "Mutual-information feature selection is fit inside the training folds only.",
            "threshold": "model-specific k",
            "n_features": int(statuses.eq("selected_in_cv").sum()),
        },
        {
            "step": "normalization",
            "policy": "inside_cv_pipeline",
            "rule": "StandardScaler normalization is fit on training folds for linear, SVM/PCA, and neural/MLP models; tree and boosting models use the same filtered numeric features without requiring scaling.",
            "threshold": "z-score for scale-sensitive estimators",
            "n_features": int(statuses.eq("retained_for_model").sum() + statuses.eq("selected_in_cv").sum()),
        },
    ]
    return pd.DataFrame(rows)


def run_ml_diagnostics(paths: AnalysisPaths, master: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out_dir = paths.section_dir("18", "ml_diagnostics")
    out_dir.mkdir(parents=True, exist_ok=True)
    feature_matrix, catalog_seed = _assemble_feature_matrix(paths, master)
    numeric, catalog, retained_features = _prefilter_features(feature_matrix, catalog_seed)
    if not retained_features:
        raise ValueError("No ML features survived leakage-safe unsupervised filtering")
    model_outputs = _run_multiclass_models(feature_matrix, numeric, retained_features, catalog)
    score_targets = _build_score_target_tables(paths, feature_matrix)
    score_outputs = _run_score_prediction_models(
        feature_matrix,
        numeric,
        retained_features,
        catalog,
        score_targets["target_matches"],
    )
    cdr_outputs = _run_cdr_classification_models(
        feature_matrix,
        numeric,
        retained_features,
        catalog,
        score_targets["target_matches"],
    )
    mci_stage_subjects, mci_stage_summary = _score_mci_stage_tables(score_targets["target_matches"])
    catalog = model_outputs["catalog"]
    group_summary = _feature_group_summary(catalog)
    preprocessing_audit = _preprocessing_audit(catalog)

    save_dataframe(feature_matrix, out_dir / "ml_subject_feature_matrix.csv")
    save_dataframe(catalog, out_dir / "ml_feature_catalog.csv")
    save_dataframe(group_summary, out_dir / "ml_feature_group_summary.csv")
    save_dataframe(preprocessing_audit, out_dir / "ml_preprocessing_audit.csv")
    save_dataframe(model_outputs["performance"], out_dir / "ml_model_performance.csv")
    save_dataframe(model_outputs["predictions"], out_dir / "ml_cross_validated_predictions.csv")
    save_dataframe(model_outputs["confusion"], out_dir / "ml_confusion_matrix.csv")
    save_dataframe(model_outputs["importance"], out_dir / "ml_feature_importance.csv")
    save_dataframe(model_outputs["shap"], out_dir / "ml_shap_contributions.csv")
    save_dataframe(model_outputs["roc"], out_dir / "ml_roc_curve.csv")
    save_dataframe(model_outputs["class_metrics"], out_dir / "ml_class_metrics.csv")
    save_dataframe(model_outputs["fold_params"], out_dir / "ml_cv_fold_best_params.csv")
    save_dataframe(model_outputs["model_status"], out_dir / "ml_model_status.csv")
    save_dataframe(score_targets["target_audit"], out_dir / "clinical_score_target_audit.csv")
    save_dataframe(score_targets["target_matches"], out_dir / "score_prediction_target_matches.csv")
    save_dataframe(score_outputs["score_performance"], out_dir / "score_prediction_model_performance.csv")
    save_dataframe(score_outputs["score_predictions"], out_dir / "score_prediction_cv_predictions.csv")
    save_dataframe(score_outputs["score_importance"], out_dir / "score_prediction_feature_importance.csv")
    save_dataframe(score_outputs["score_residuals"], out_dir / "score_prediction_residuals.csv")
    save_dataframe(score_outputs["score_group_performance"], out_dir / "score_prediction_group_performance.csv")
    save_dataframe(score_outputs["score_fold_params"], out_dir / "score_prediction_cv_fold_best_params.csv")
    save_dataframe(score_outputs["score_mixed_models"], out_dir / "score_prediction_mixed_model.csv")
    save_dataframe(score_outputs["score_outlier_audit"], out_dir / "score_prediction_outlier_audit.csv")
    save_dataframe(score_outputs["score_model_status"], out_dir / "score_prediction_model_status.csv")
    save_dataframe(cdr_outputs["cdr_performance"], out_dir / "cdr_classification_model_performance.csv")
    save_dataframe(cdr_outputs["cdr_predictions"], out_dir / "cdr_classification_cv_predictions.csv")
    save_dataframe(cdr_outputs["cdr_confusion"], out_dir / "cdr_classification_confusion_matrix.csv")
    save_dataframe(cdr_outputs["cdr_class_metrics"], out_dir / "cdr_classification_class_metrics.csv")
    save_dataframe(cdr_outputs["cdr_importance"], out_dir / "cdr_classification_feature_importance.csv")
    save_dataframe(cdr_outputs["cdr_fold_params"], out_dir / "cdr_classification_cv_fold_best_params.csv")
    save_dataframe(cdr_outputs["cdr_model_status"], out_dir / "cdr_classification_model_status.csv")
    save_dataframe(mci_stage_subjects, out_dir / "score_prediction_mci_stage_subjects.csv")
    save_dataframe(mci_stage_summary, out_dir / "score_prediction_mci_stage_summary.csv")
    write_inference_markdown(
        [
            "ML Diagnostics predicts multiclass diagnostic group CN/MCI/AD only.",
            "The feature matrix uses structural connectome feature classes, then applies leakage-safe missingness, variance, sparsity, and correlation filtering before supervised selection inside nested cross-validation.",
            "The master cohort CSV is used only to define the subject universe and true diagnostic labels; it does not contribute predictors such as ADNI phase, population columns, APOE, age, sex, weight, or clinical scores.",
            "Group-derived p-values, q-values, Welch statistics, and pairwise inference outputs are not used as predictors.",
        ],
        out_dir / "ml_diagnostics_inference.md",
    )
    write_inference_markdown(
        [
            f"Clinical Outcome Prediction uses the same structural feature matrix but trains MMSE Total Score as a continuous regression target and Global CDR as an ordinal multiclass classification target with classes {', '.join(CDR_CLASS_ORDER)}; age, sex, phase, APOE, diagnosis labels, and clinical score columns remain excluded from model inputs.",
            f"The primary score target mode uses the nearest clinical score within {PRIMARY_SCORE_WINDOW_DAYS} days of the selected DTI scan and trains only when at least {MIN_SCORE_TARGET_SUBJECTS} labeled subjects across at least {MIN_SCORE_TARGET_GROUPS} groups are available.",
            "Before MMSE regression model training, target-value outliers are removed separately within each diagnosis group using a 1.5x IQR rule and written to score_prediction_outlier_audit.csv.",
            "Global CDR is not target-outlier filtered because high CDR values are clinically meaningful severity classes; classification metrics are written separately to cdr_classification_*.csv.",
            "Preprocessing includes median imputation, unsupervised missingness/sparsity/near-zero-variance/correlation filters, nested-CV feature selection, and scaling for linear and MLP models; tree and boosting models use the same filtered features but do not require normalization. The explicit preprocessing audit is written to ml_preprocessing_audit.csv.",
            "The all-available-history mode is written as sensitivity output because it can mix remote clinical status with the current connectome scan.",
            "Elastic-net, XGBoost, tree ensembles, histogram boosting, MLP, and augmented MLP are benchmarked; FT-Transformer, TabM, TabNet, and TabPFN remain dependency-gated candidates.",
            "Mixed-model output uses a subject random intercept only when repeated prediction rows exist; otherwise it falls back to robust OLS without age.",
        ],
        out_dir / "score_prediction_inference.md",
    )
    return {
        "feature_matrix": feature_matrix,
        "feature_catalog": catalog,
        "feature_group_summary": group_summary,
        "preprocessing_audit": preprocessing_audit,
        **model_outputs,
        **score_targets,
        **score_outputs,
        **cdr_outputs,
        "mci_stage_subjects": mci_stage_subjects,
        "mci_stage_summary": mci_stage_summary,
        "out_dir": out_dir,
    }
