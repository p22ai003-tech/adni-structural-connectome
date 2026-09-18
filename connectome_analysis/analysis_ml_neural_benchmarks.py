from __future__ import annotations

import argparse
import importlib.util
import math
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    explained_variance_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler

from connectome_analysis.analysis_ml_goal_search import (
    GROUP_ORDER,
    PRIMARY_MODE,
    SCORE_TARGETS,
    STRUCTURAL_SCORE_FEATURE_POLICY,
    _build_model_table,
    _diagnostics_root,
    _drop_target_leakage,
    _feature_sets,
    _numeric_frame,
    _score_tasks,
)

RANDOM_STATE = 42


def _availability() -> pd.DataFrame:
    modules = ("torch", "pytorch_tabnet", "tabpfn", "shap")
    return pd.DataFrame(
        [
            {
                "dependency": module,
                "available": importlib.util.find_spec(module) is not None,
            }
            for module in modules
        ]
    )


def _preprocess_train_test(x_train: pd.DataFrame, x_test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    observed = [c for c in x_train.columns if x_train[c].notna().any()]
    x_train = x_train.loc[:, observed]
    x_test = x_test.loc[:, observed]
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    train = scaler.fit_transform(imputer.fit_transform(x_train))
    test = scaler.transform(imputer.transform(x_test))
    return train.astype(np.float32), test.astype(np.float32)


def _augment(x: np.ndarray, y: np.ndarray, *, reps: int = 3, noise_sd: float = 0.03) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(RANDOM_STATE)
    xs = [x]
    ys = [y]
    for _ in range(reps):
        xs.append(x + rng.normal(0.0, noise_sd, size=x.shape).astype(np.float32))
        ys.append(y)
    return np.vstack(xs), np.concatenate(ys)


def _reg_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    out = {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "explained_variance": float(explained_variance_score(y_true, y_pred)),
        "spearman_r": np.nan,
        "pearson_r": np.nan,
    }
    if len(y_true) >= 3 and len(np.unique(y_true)) > 1 and len(np.unique(y_pred)) > 1:
        out["spearman_r"] = float(stats.spearmanr(y_true, y_pred).statistic)
        out["pearson_r"] = float(stats.pearsonr(y_true, y_pred).statistic)
    return out


def _tabnet_classifier():
    from pytorch_tabnet.tab_model import TabNetClassifier

    return TabNetClassifier(
        n_d=12,
        n_a=12,
        n_steps=3,
        seed=RANDOM_STATE,
        verbose=0,
        device_name="cpu",
    )


def _tabnet_regressor():
    from pytorch_tabnet.tab_model import TabNetRegressor

    return TabNetRegressor(
        n_d=12,
        n_a=12,
        n_steps=3,
        seed=RANDOM_STATE,
        verbose=0,
        device_name="cpu",
    )


def _tabpfn_classifier():
    from tabpfn import TabPFNClassifier

    return TabPFNClassifier(
        n_estimators=4,
        device="cpu",
        ignore_pretraining_limits=True,
        random_state=RANDOM_STATE,
        show_progress_bar=False,
    )


def _tabpfn_regressor():
    from tabpfn import TabPFNRegressor

    return TabPFNRegressor(
        n_estimators=4,
        device="cpu",
        ignore_pretraining_limits=True,
        random_state=RANDOM_STATE,
        show_progress_bar=False,
    )


def _classification_model_names(feature_count: int) -> list[str]:
    names = ["mlp_deep", "mlp_deep_augmented", "tabnet_classifier", "tabpfn_classifier"]
    if feature_count > 100:
        return ["mlp_deep", "mlp_deep_augmented", "tabnet_classifier"]
    return names


def _regression_model_names(feature_count: int) -> list[str]:
    names = ["mlp_deep", "mlp_deep_augmented", "tabnet_regressor", "tabpfn_regressor"]
    if feature_count > 100:
        return ["mlp_deep", "mlp_deep_augmented", "tabnet_regressor"]
    return names


def _class_feature_importance(model, x_test: pd.DataFrame, y_test: np.ndarray, features: list[str]) -> dict[str, float]:
    try:
        result = permutation_importance(
            model,
            x_test,
            y_test,
            scoring="balanced_accuracy",
            n_repeats=3,
            random_state=RANDOM_STATE,
            n_jobs=1,
        )
        return {f: float(max(v, 0.0)) for f, v in zip(features, result.importances_mean)}
    except Exception:
        return {}


def _run_disease(table: pd.DataFrame, feature_sets, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    task = table[table["group"].isin(GROUP_ORDER)].copy().reset_index(drop=True)
    encoder = LabelEncoder().fit(list(GROUP_ORDER))
    y = encoder.transform(task["group"].astype(str))
    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    perf_rows = []
    pred_rows = []
    class_rows = []
    importance_rows = []
    for fs in feature_sets:
        features = [c for c in fs.columns if c in task.columns]
        x = _numeric_frame(task, features)
        features = [c for c in features if x[c].notna().any()]
        x = x.loc[:, features]
        for model_name in _classification_model_names(len(features)):
            fold_importance = Counter()
            try:
                for fold, (train_idx, test_idx) in enumerate(outer.split(x, y), start=1):
                    x_train, x_test = _preprocess_train_test(x.iloc[train_idx], x.iloc[test_idx])
                    y_train = y[train_idx]
                    if model_name == "mlp_deep":
                        model = MLPClassifier(
                            hidden_layer_sizes=(192, 64, 16),
                            alpha=0.02,
                            early_stopping=True,
                            max_iter=300,
                            random_state=RANDOM_STATE,
                        )
                        model.fit(x_train, y_train)
                        pred = model.predict(x_test)
                        prob = model.predict_proba(x_test)
                    elif model_name == "mlp_deep_augmented":
                        model = MLPClassifier(
                            hidden_layer_sizes=(192, 64, 16),
                            alpha=0.03,
                            early_stopping=True,
                            max_iter=300,
                            random_state=RANDOM_STATE,
                        )
                        x_aug, y_aug = _augment(x_train, y_train)
                        model.fit(x_aug, y_aug)
                        pred = model.predict(x_test)
                        prob = model.predict_proba(x_test)
                    elif model_name == "tabnet_classifier":
                        model = _tabnet_classifier()
                        model.fit(x_train, y_train, max_epochs=50, patience=8, batch_size=128, virtual_batch_size=32)
                        pred = model.predict(x_test).astype(int)
                        prob = model.predict_proba(x_test)
                    else:
                        model = _tabpfn_classifier()
                        model.fit(x_train, y_train)
                        pred = model.predict(x_test).astype(int)
                        prob = model.predict_proba(x_test)
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
                            row[f"prob_{class_name}"] = float(prob[row_pos, class_idx])
                        pred_rows.append(row)
                    if len(features) <= 60:
                        wrapped = ExtraTreesClassifier(n_estimators=80, random_state=RANDOM_STATE).fit(x_train, y_train)
                        fold_importance.update(_class_feature_importance(wrapped, pd.DataFrame(x_test, columns=features), y[test_idx], features))
            except Exception as exc:
                perf_rows.append(
                    {
                        "task": "disease_prediction",
                        "feature_set": fs.name,
                        "feature_policy": fs.policy,
                        "model": model_name,
                        "status": f"failed: {exc}",
                        "n_subjects": len(task),
                        "n_features": len(features),
                        "accuracy": np.nan,
                        "balanced_accuracy": np.nan,
                        "macro_f1": np.nan,
                        "weighted_f1": np.nan,
                        "macro_auroc_ovr": np.nan,
                    }
                )
                continue
            sub = pd.DataFrame([r for r in pred_rows if r["feature_set"] == fs.name and r["model"] == model_name])
            y_true = encoder.transform(sub["true_group"].astype(str))
            y_pred = encoder.transform(sub["predicted_group"].astype(str))
            prob = sub[[f"prob_{g}" for g in encoder.classes_]].to_numpy(dtype=float)
            macro_auroc = np.nan
            try:
                macro_auroc = float(roc_auc_score(y_true, prob, multi_class="ovr", average="macro"))
            except Exception:
                pass
            perf_rows.append(
                {
                    "task": "disease_prediction",
                    "feature_set": fs.name,
                    "feature_policy": fs.policy,
                    "model": model_name,
                    "status": "ok",
                    "n_subjects": len(task),
                    "n_features": len(features),
                    "accuracy": float(accuracy_score(y_true, y_pred)),
                    "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
                    "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
                    "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
                    "macro_auroc_ovr": macro_auroc,
                }
            )
            precision, recall, f1_vals, support = precision_recall_fscore_support(y_true, y_pred, labels=list(range(len(encoder.classes_))), zero_division=0)
            for idx, class_name in enumerate(encoder.classes_):
                class_rows.append(
                    {
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
            for rank, (feature, value) in enumerate(fold_importance.most_common(50), start=1):
                importance_rows.append(
                    {
                        "task": "disease_prediction",
                        "target": "CN_MCI_AD",
                        "feature_set": fs.name,
                        "feature_policy": fs.policy,
                        "model": model_name,
                        "rank": rank,
                        "feature": feature,
                        "importance": float(value / outer.n_splits),
                    }
                )
    return pd.DataFrame(perf_rows), pd.DataFrame(pred_rows), pd.DataFrame(class_rows), pd.DataFrame(importance_rows)


def _run_scores(table: pd.DataFrame, matches: pd.DataFrame, feature_sets) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    perf_rows = []
    pred_rows = []
    importance_rows = []
    tasks, outlier_audit = _score_tasks(matches, table, modes={PRIMARY_MODE})
    feature_sets = [fs for fs in feature_sets if fs.policy == STRUCTURAL_SCORE_FEATURE_POLICY]
    for task in tasks:
        target = str(task["target"].iloc[0])
        target_mode = str(task["target_mode"].iloc[0])
        n_before_outliers = int(task.get("n_subjects_before_outlier_filter", pd.Series([len(task)])).iloc[0])
        n_outliers_removed = int(task.get("n_outliers_removed", pd.Series([0])).iloc[0])
        y = task["target_value"].to_numpy(dtype=np.float32)
        outer = KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        for fs in feature_sets:
            features = _drop_target_leakage([c for c in fs.columns if c in task.columns], target)
            x = _numeric_frame(task, features)
            features = [c for c in features if x[c].notna().any()]
            x = x.loc[:, features]
            for model_name in _regression_model_names(len(features)):
                fold_importance = Counter()
                try:
                    for fold, (train_idx, test_idx) in enumerate(outer.split(x), start=1):
                        x_train, x_test = _preprocess_train_test(x.iloc[train_idx], x.iloc[test_idx])
                        y_train = y[train_idx]
                        if model_name == "mlp_deep":
                            model = MLPRegressor(
                                hidden_layer_sizes=(192, 64, 16),
                                alpha=0.05,
                                early_stopping=True,
                                max_iter=350,
                                random_state=RANDOM_STATE,
                            )
                            model.fit(x_train, y_train)
                            pred = model.predict(x_test)
                        elif model_name == "mlp_deep_augmented":
                            model = MLPRegressor(
                                hidden_layer_sizes=(192, 64, 16),
                                alpha=0.06,
                                early_stopping=True,
                                max_iter=350,
                                random_state=RANDOM_STATE,
                            )
                            x_aug, y_aug = _augment(x_train, y_train)
                            model.fit(x_aug, y_aug)
                            pred = model.predict(x_test)
                        elif model_name == "tabnet_regressor":
                            model = _tabnet_regressor()
                            model.fit(x_train, y_train.reshape(-1, 1), max_epochs=50, patience=8, batch_size=64, virtual_batch_size=16)
                            pred = np.ravel(model.predict(x_test))
                        else:
                            model = _tabpfn_regressor()
                            model.fit(x_train, y_train)
                            pred = np.ravel(model.predict(x_test))
                        for row_pos, idx in enumerate(test_idx):
                            pred_rows.append(
                                {
                                    "task": "score_prediction",
                                    "target": target,
                                    "target_mode": target_mode,
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
                        if len(features) <= 60:
                            wrapped = ExtraTreesRegressor(n_estimators=80, random_state=RANDOM_STATE).fit(x_train, y_train)
                            try:
                                result = permutation_importance(
                                    wrapped,
                                    pd.DataFrame(x_test, columns=features),
                                    y[test_idx],
                                    scoring="r2",
                                    n_repeats=3,
                                    random_state=RANDOM_STATE,
                                )
                                fold_importance.update({f: float(max(v, 0.0)) for f, v in zip(features, result.importances_mean)})
                            except Exception:
                                pass
                except Exception as exc:
                    perf_rows.append(
                        {
                            "task": "score_prediction",
                            "target": target,
                            "target_mode": target_mode,
                            "feature_set": fs.name,
                            "feature_policy": fs.policy,
                            "model": model_name,
                            "status": f"failed: {exc}",
                            "n_subjects": len(task),
                            "n_subjects_before_outlier_filter": n_before_outliers,
                            "n_outliers_removed": n_outliers_removed,
                            "n_features": len(features),
                            "mae": np.nan,
                            "rmse": np.nan,
                            "r2": np.nan,
                            "explained_variance": np.nan,
                            "spearman_r": np.nan,
                            "pearson_r": np.nan,
                        }
                    )
                    continue
                pred_df = pd.DataFrame(
                    [r for r in pred_rows if r["target"] == target and r["feature_set"] == fs.name and r["model"] == model_name]
                )
                metrics = _reg_metrics(pred_df["true_score"].to_numpy(dtype=float), pred_df["predicted_score"].to_numpy(dtype=float))
                perf_rows.append(
                    {
                        "task": "score_prediction",
                        "target": target,
                        "target_mode": target_mode,
                        "feature_set": fs.name,
                        "feature_policy": fs.policy,
                        "model": model_name,
                        "status": "ok",
                        "n_subjects": len(task),
                        "n_subjects_before_outlier_filter": n_before_outliers,
                        "n_outliers_removed": n_outliers_removed,
                        "n_features": len(features),
                        **metrics,
                    }
                )
                for rank, (feature, value) in enumerate(fold_importance.most_common(50), start=1):
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
                            "importance": float(value / outer.n_splits),
                        }
                    )
    return pd.DataFrame(perf_rows), pd.DataFrame(pred_rows), pd.DataFrame(importance_rows), outlier_audit


def run_neural_benchmarks(exp_root: Path, out_dir: Path, only_feature_sets: set[str] | None = None, task: str = "both") -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    table, catalog, matches = _build_model_table(_diagnostics_root(exp_root), exp_root)
    feature_sets = _feature_sets(table, catalog, only=only_feature_sets)
    _availability().to_csv(out_dir / "neural_dependency_status.csv", index=False)
    pd.DataFrame(
        [{"feature_set": fs.name, "feature_policy": fs.policy, "n_features": len(fs.columns)} for fs in feature_sets]
    ).to_csv(out_dir / "neural_feature_sets.csv", index=False)
    importance = []
    if task in {"both", "disease"}:
        disease_perf, disease_pred, class_metrics, disease_imp = _run_disease(table, feature_sets, out_dir)
        disease_perf.to_csv(out_dir / "neural_disease_model_performance.csv", index=False)
        disease_pred.to_csv(out_dir / "neural_disease_cv_predictions.csv", index=False)
        class_metrics.to_csv(out_dir / "neural_disease_class_metrics.csv", index=False)
        disease_imp.to_csv(out_dir / "neural_disease_feature_importance.csv", index=False)
        importance.append(disease_imp)
    if task in {"both", "score"}:
        score_perf, score_pred, score_imp, score_outliers = _run_scores(table, matches, feature_sets)
        score_perf.to_csv(out_dir / "neural_score_model_performance.csv", index=False)
        score_pred.to_csv(out_dir / "neural_score_cv_predictions.csv", index=False)
        score_imp.to_csv(out_dir / "neural_score_feature_importance.csv", index=False)
        score_outliers.to_csv(out_dir / "neural_score_outlier_audit.csv", index=False)
        importance.append(score_imp)
    if importance:
        pd.concat(importance, ignore_index=True).to_csv(out_dir / "neural_feature_importance.csv", index=False)
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run optional neural benchmarks for connectome ML goal search.")
    parser.add_argument("--exp-root", type=Path, default=Path("/home/ec2-user/exp"))
    parser.add_argument("--out-dir", type=Path, default=Path("/home/ec2-user/exp/outputs/enhanced_neural_benchmarks"))
    parser.add_argument("--feature-set", action="append", default=None)
    parser.add_argument("--task", choices=("both", "disease", "score"), default="both")
    args = parser.parse_args()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        out = run_neural_benchmarks(
            args.exp_root,
            args.out_dir,
            only_feature_sets=set(args.feature_set) if args.feature_set else None,
            task=args.task,
        )
    print(out)


if __name__ == "__main__":
    main()
