"""Readers for the exception-specificity analysis (Part A) and the simple
MMSE/CDR models (Part B) shown in the LR/SR page's "exception-specific signal"
section. All artifacts live under analysis_root/20_exception_specificity/ and
are precomputed; nothing is fitted at request time.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Any

import pandas as pd

from .settings import Settings

_LOCK = Lock()


def _mtime(path: Path) -> int:
    try:
        return path.stat().st_mtime_ns
    except OSError:
        return -1


def _root(settings: Settings) -> Path:
    return settings.analysis_root / "20_exception_specificity"


def _signature(settings: Settings) -> tuple[int, ...]:
    root = _root(settings)
    names = (
        "exception_tier_delta.csv",
        "exception_architecture_stats.csv",
        "ml_ladder_results.csv",
        "ml_predictions.csv",
        "ml_permutation.json",
        "ml_targets.csv",
        "ml_r2_summary.csv",
        "ml_shap_values_mmse.csv",
        "ml_ice_mmse.csv",
        "mmse_bucket_results.csv",
        "ml_shap_class_group.csv",
        "ml_pipeline_steps.csv",
    )
    return tuple(_mtime(root / n) for n in names)


@lru_cache(maxsize=4)
def _payload_cached(settings: Settings, signature: tuple[int, ...]) -> dict[str, Any]:
    del signature
    root = _root(settings)
    out: dict[str, Any] = {"status": "ok"}

    delta_path = root / "exception_tier_delta.csv"
    if delta_path.is_file():
        delta = pd.read_csv(delta_path)
        out["tier_delta"] = delta.to_dict(orient="records")
        out["tier_delta_summary"] = [
            dict(contrast=str(ck),
                 mean_gain_vs_range=float(sub.gain_vs_range.mean()),
                 pop_ups=int(sub.pop_up.sum()),
                 drop_outs=int(sub.drop_out.sum()),
                 cells=int(len(sub)))
            for ck, sub in delta.groupby("contrast")
        ]
    else:
        out["status"] = "partial"

    arch_path = root / "exception_architecture_stats.csv"
    if arch_path.is_file():
        out["architecture"] = pd.read_csv(arch_path).to_dict(orient="records")
    else:
        out["status"] = "partial"

    ladder_path = root / "ml_ladder_results.csv"
    if ladder_path.is_file():
        out["ml_ladder"] = pd.read_csv(ladder_path).to_dict(orient="records")
        preds = pd.read_csv(root / "ml_predictions.csv")
        out["ml_predictions"] = preds.to_dict(orient="records")
        out["ml_permutation"] = json.load(open(root / "ml_permutation.json"))
        targets = pd.read_csv(root / "ml_targets.csv")
        levels = (targets.dropna(subset=["cdr"])
                  .groupby(["cdr", "group"]).size().unstack(fill_value=0))
        out["cdr_level_counts"] = [
            dict(level=float(lvl),
                 CN=int(row.get("CN", 0)), MCI=int(row.get("MCI", 0)), AD=int(row.get("AD", 0)),
                 total=int(row.sum()))
            for lvl, row in levels.iterrows()
        ]
        out["ml_cohort"] = dict(
            n=int(len(targets)),
            mmse_n=int(targets["mmse"].notna().sum()),
            cdr_n=int(targets["cdr"].notna().sum()),
            cdr_balance=[int((targets["cdr_bin"] == 0).sum()), int((targets["cdr_bin"] == 1).sum())],
            median_lag_days=float(targets["mmse_lag_days"].median()),
        )
    else:
        out["ml_status"] = "pending"

    r2_path = root / "ml_r2_summary.csv"
    if r2_path.is_file():
        out["ml_r2"] = pd.read_csv(r2_path).to_dict(orient="records")
    shap_frames, ice_frames = [], []
    for target in ("mmse", "cdr_bin"):
        sp = root / f"ml_shap_values_{target}.csv"
        ip = root / f"ml_ice_{target}.csv"
        if sp.is_file():
            shap_frames.append(pd.read_csv(sp))
        if ip.is_file():
            ice_frames.append(pd.read_csv(ip))
    if shap_frames:
        allshap = pd.concat(shap_frames, ignore_index=True)
        out["ml_shap"] = allshap.to_dict(orient="records")
        share = (allshap.groupby(["target", "family"]).mean_abs_shap.sum().reset_index())
        total = share.groupby("target").mean_abs_shap.transform("sum")
        share["pct"] = (share.mean_abs_shap / total * 100).round(1)
        out["ml_shap_family_share"] = share.to_dict(orient="records")
    bucket_res = root / "mmse_bucket_results.csv"
    if bucket_res.is_file():
        out["mmse_bucket_results"] = pd.read_csv(bucket_res).to_dict(orient="records")
        for name, key in (("mmse_bucket_definition", "mmse_bucket_definition"),
                          ("mmse_bucket_classwise", "mmse_bucket_classwise"),
                          ("mmse_bucket_curves", "mmse_bucket_curves"),
                          ("mmse_bucket_shap_classwise", "mmse_bucket_shap")):
            path = root / f"{name}.csv"
            if path.is_file():
                out[key] = pd.read_csv(path).to_dict(orient="records")

    if ice_frames:
        ice = pd.concat(ice_frames, ignore_index=True)
        out["ml_ice"] = ice[ice.pdp.notna()].to_dict(orient="records")

    # explainability v2: preprocessing transparency + class x diagnostic-group attribution
    for name, key in (("ml_feature_dictionary", "feature_dictionary"),
                      ("ml_pipeline_steps", "pipeline_steps"),
                      ("ml_preprocessing_audit", "preprocessing_audit"),
                      ("ml_selection_experiment", "selection_experiment"),
                      ("ml_shap_class_group", "shap_class_group"),
                      ("ml_shap_overall_beeswarm", "shap_overall"),
                      ("ml_shap_family_group", "shap_family_group"),
                      ("ml_shap_class_group_dir", "shap_class_group_dir"),
                      ("ml_pdp_class_group", "pdp_class_group"),
                      ("ml_pdp_range_family", "pdp_range_family"),
                      ("cdr_curves", "cdr_curves"),
                      ("cdr_classwise", "cdr_classwise"),
                      ("ml_shap_beeswarm_class", "shap_beeswarm_class"),
                      ("cdr_threeclass", "cdr_threeclass"),
                      ("improvement_final_check", "improvement_check"),
                      ("improvement_mmse", "improvement_mmse"),
                      ("ml_auc_class_diaggroup", "auc_class_diaggroup"),
                      ("final_model_table", "final_model_table"),
                      ("final_model_metrics", "final_model_metrics"),
                      ("final_model_classwise", "final_model_classwise"),
                      ("final_model_curves", "final_model_curves")):
        path = root / f"{name}.csv"
        if path.is_file():
            out[key] = pd.read_csv(path).to_dict(orient="records")
    inf = root / "ml_inference_summary.json"
    if inf.is_file():
        out["inference_summary"] = json.load(open(inf))

    return out


def exception_specificity_payload(settings: Settings) -> dict[str, Any]:
    """Precomputed Part A + Part B artifacts for the dashboard (mtime-cached)."""
    with _LOCK:
        return _payload_cached(settings, _signature(settings))
