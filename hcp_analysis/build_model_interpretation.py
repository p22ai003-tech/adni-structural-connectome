#!/usr/bin/env python
"""Model interpretation for the simple MMSE/CDR models: R2, SHAP, ICE/PDP.

Uses the same feature matrix, targets, folds and models as build_simple_ml.py so
the numbers agree with the ladder table. Everything is computed out-of-fold where
it is a performance number, and on a refit-on-all model where it is an explanation
(SHAP / ICE), which is the standard split.

Outputs (hcp_analysis/):
  ml_r2_summary.csv        out-of-fold R2 / MAE / Spearman (MMSE) and AUC (CDR) per rung x model
  ml_shap_values.csv       mean |SHAP| per feature (top 25) for the best model, per target
  ml_shap_beeswarm.csv     per-subject SHAP for the top 12 features (for the beeswarm plot)
  ml_ice.csv               ICE + PDP curves for the top 6 features, per target
  figs/fig4_shap_mmse.png  beeswarm + bar
  figs/fig5_ice_mmse.png   ICE/PDP small multiples
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import partial_dependence
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

# Paths resolve through sc_paths: project-relative, overridable with --out, and
# defaulting to the analysis-tree section the dashboard actually reads. Writing
# straight there is what removes the old manual copy step.
try:
    import sc_paths
except ModuleNotFoundError:  # loose script run from outside hcp_analysis/
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent))
    import sc_paths

_P = sc_paths.resolve_for_script("ml")

OUT = _P.out
FIGS = _P.figs
SEEDS = [11, 23, 37, 51, 73]
BLUE, ORANGE = "#2a78d6", "#eb6834"
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3e0"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "figure.facecolor": SURF, "axes.facecolor": SURF})


def deframe(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def load():
    feats = pd.read_csv(OUT / "ml_feature_matrix.csv", index_col=0)
    targets = pd.read_csv(OUT / "ml_targets.csv", index_col=0)
    covar = pd.DataFrame({"age": targets.age,
                          "sex": (targets.sex.astype(str).str.upper().str[0] == "F").astype(float)},
                         index=targets.index)
    return feats, targets, covar


def rung_columns(feats, rung):
    f1 = [c for c in feats.columns if c.startswith("WB_")]
    f2 = [c for c in feats.columns if c.startswith(("SR_", "LR_"))]
    f3 = [c for c in feats.columns if c.startswith(("SREXC_", "LREXC_"))]
    f4 = [c for c in feats.columns if c.startswith("ARCH_")]
    return {"F0": [], "F0+F1": f1, "F0-F2": f1 + f2, "F0-F3": f1 + f2 + f3,
            "F0-F4": f1 + f2 + f3 + f4}[rung]


def oof_r2(feats, targets, covar):
    """Out-of-fold R2 pooled across all held-out predictions (not fold-mean)."""
    rows = []
    for target, kind in (("mmse", "reg"), ("cdr_bin", "clf")):
        y_all = targets[target].dropna()
        idx = y_all.index.intersection(feats.index)
        y = y_all.loc[idx]
        strat = ((targets.loc[idx, "group"].astype(str) + "|" + (y > y.median()).astype(str))
                 if kind == "reg" else y.astype(int).astype(str))
        for rung in ("F0", "F0+F1", "F0-F2", "F0-F3", "F0-F4"):
            X = pd.concat([covar.loc[idx], feats.loc[idx, rung_columns(feats, rung)]], axis=1)
            pooled = []
            for seed in SEEDS:
                est = (ExtraTreesRegressor(n_estimators=400, min_samples_leaf=5, random_state=seed, n_jobs=8)
                       if kind == "reg" else
                       ExtraTreesClassifier(n_estimators=400, min_samples_leaf=5, class_weight="balanced",
                                            random_state=seed, n_jobs=8))
                imp = SimpleImputer(strategy="median")
                oof = np.full(len(y), np.nan)
                for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, strat):
                    Xtr = imp.fit_transform(X.iloc[tr]); Xte = imp.transform(X.iloc[te])
                    est.fit(Xtr, y.iloc[tr])
                    oof[te] = est.predict(Xte) if kind == "reg" else est.predict_proba(Xte)[:, 1]
                pooled.append(r2_score(y, oof) if kind == "reg" else roc_auc_score(y, oof))
            rows.append(dict(target=target, rung=rung, model="extra_trees",
                             metric="pooled_oof_r2" if kind == "reg" else "pooled_oof_auc",
                             value=float(np.mean(pooled)), sd=float(np.std(pooled)), n=len(y),
                             n_features=X.shape[1]))
            print(f"  {target:8} {rung:6} {rows[-1]['metric']}={rows[-1]['value']:+.3f} ± {rows[-1]['sd']:.3f}", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "ml_r2_summary.csv", index=False)
    return frame


def explain(feats, targets, covar, target="mmse", kind="reg", rung="F0-F4", top_n=25):
    y_all = targets[target].dropna()
    idx = y_all.index.intersection(feats.index)
    y = y_all.loc[idx]
    X = pd.concat([covar.loc[idx], feats.loc[idx, rung_columns(feats, rung)]], axis=1)
    imp = SimpleImputer(strategy="median")
    Xi = pd.DataFrame(imp.fit_transform(X), columns=X.columns, index=X.index)
    est = (ExtraTreesRegressor(n_estimators=600, min_samples_leaf=5, random_state=11, n_jobs=8)
           if kind == "reg" else
           ExtraTreesClassifier(n_estimators=600, min_samples_leaf=5, class_weight="balanced",
                                random_state=11, n_jobs=8))
    est.fit(Xi, y)

    sv = shap.TreeExplainer(est).shap_values(Xi)
    if isinstance(sv, list):
        sv = sv[1]
    elif sv.ndim == 3:
        sv = sv[:, :, 1]
    mean_abs = pd.Series(np.abs(sv).mean(axis=0), index=Xi.columns).sort_values(ascending=False)
    fam = lambda c: ("covariate" if c in ("age", "sex") else
                     "F1 whole-brain" if c.startswith("WB_") else
                     "F2 SR/LR" if c.startswith(("SR_", "LR_")) else
                     "F3 exceptions" if c.startswith(("SREXC_", "LREXC_")) else "F4 architecture")
    out = pd.DataFrame({"feature": mean_abs.index[:top_n], "mean_abs_shap": mean_abs.values[:top_n]})
    out["family"] = out.feature.map(fam)
    out["target"] = target
    out.to_csv(OUT / f"ml_shap_values_{target}.csv", index=False)
    top12 = list(mean_abs.index[:12])
    bee = pd.DataFrame(sv, columns=Xi.columns, index=Xi.index)[top12].reset_index().melt(
        id_vars=Xi.index.name or "index", var_name="feature", value_name="shap")
    bee["feature_value"] = [Xi.loc[r[Xi.index.name or "index"], r["feature"]] for _, r in bee.iterrows()]
    bee["target"] = target
    bee.to_csv(OUT / f"ml_shap_beeswarm_{target}.csv", index=False)

    # family-level share of total attribution
    share = out.groupby("family").mean_abs_shap.sum()
    share = (share / share.sum() * 100).round(1)
    print(f"  {target} SHAP family share of top-{top_n}: {share.to_dict()}", flush=True)

    # ICE + PDP for top 6
    ice_rows = []
    for feat in mean_abs.index[:6]:
        pd_res = partial_dependence(est, Xi, [Xi.columns.get_loc(feat)], kind="both", grid_resolution=25)
        gv = pd_res["grid_values"][0]
        ind = pd_res["individual"][0]
        avg = pd_res["average"][0]
        for j, g in enumerate(gv):
            ice_rows.append(dict(target=target, feature=feat, grid=float(g), pdp=float(avg[j]),
                                 ice_lo=float(np.percentile(ind[:, j], 10)),
                                 ice_hi=float(np.percentile(ind[:, j], 90))))
        for i in range(0, ind.shape[0], max(1, ind.shape[0] // 40)):
            for j, g in enumerate(gv):
                ice_rows.append(dict(target=target, feature=feat, grid=float(g), pdp=np.nan,
                                     ice_lo=np.nan, ice_hi=np.nan, ice_curve=float(ind[i, j]), subject=i))
    pd.DataFrame(ice_rows).to_csv(OUT / f"ml_ice_{target}.csv", index=False)
    return est, Xi, y, sv, mean_abs


def figures(target, Xi, sv, mean_abs, r2_frame):
    top = list(mean_abs.index[:12])[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2), gridspec_kw={"width_ratios": [1, 1.25]})
    ax = axes[0]
    ax.barh(range(len(top)), [mean_abs[f] for f in top], color=BLUE, zorder=3)
    ax.set_yticks(range(len(top))); ax.set_yticklabels([t.replace("_", " ") for t in top], fontsize=8.5)
    ax.set_xlabel("mean |SHAP| (MMSE points)")
    ax.set_title("Feature importance", loc="left", fontsize=11)
    deframe(ax)
    ax = axes[1]
    rng = np.random.default_rng(3)
    for i, f in enumerate(top):
        v = sv[:, Xi.columns.get_loc(f)]
        x = Xi[f].to_numpy(float)
        norm = (x - np.nanmin(x)) / (np.ptp(x) if np.ptp(x) else 1)
        ax.scatter(v, i + rng.uniform(-0.16, 0.16, len(v)), c=norm, cmap="coolwarm",
                   s=8, alpha=0.65, linewidths=0)
    ax.set_yticks(range(len(top))); ax.set_yticklabels([])
    ax.axvline(0, color=INK2, lw=1)
    ax.set_xlabel("SHAP value (impact on predicted MMSE)")
    ax.set_title("Per-subject effect (colour = feature value, blue low → red high)", loc="left", fontsize=11)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)
    r2row = r2_frame[(r2_frame.target == target) & (r2_frame.rung == "F0-F4")].iloc[0]
    fig.text(0.012, 0.965, f"What the {target.upper()} model uses", fontsize=13, fontweight="bold", va="top")
    fig.text(0.012, 0.915,
             f"ExtraTrees on the full F0–F4 ladder (n={int(r2row.n)}, {int(r2row.n_features)} features). "
             f"Pooled out-of-fold R² = {r2row.value:+.3f} ± {r2row.sd:.3f}.",
             fontsize=9, color=INK2, va="top")
    fig.tight_layout(rect=[0, 0, 1, 0.87])
    fig.savefig(FIGS / f"fig4_shap_{target}.png", dpi=200)
    plt.close(fig)

    ice = pd.read_csv(OUT / f"ml_ice_{target}.csv")
    feats6 = ice.feature.unique()[:6]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.4))
    for ax, f in zip(axes.ravel(), feats6):
        sub = ice[ice.feature == f]
        curves = sub.dropna(subset=["ice_curve"])
        for _, g in curves.groupby("subject"):
            ax.plot(g.grid, g.ice_curve, color=BLUE, alpha=0.13, lw=0.8)
        band = sub.dropna(subset=["pdp"])
        ax.fill_between(band.grid, band.ice_lo, band.ice_hi, color=BLUE, alpha=0.13, lw=0)
        ax.plot(band.grid, band.pdp, color=ORANGE, lw=2.5)
        ax.set_title(f.replace("_", " "), fontsize=9.5, loc="left")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(color=GRID, lw=0.7); ax.set_axisbelow(True)
        ax.tick_params(labelsize=8)
    fig.text(0.012, 0.965, f"How each feature moves the prediction — ICE + partial dependence ({target.upper()})",
             fontsize=13, fontweight="bold", va="top")
    fig.text(0.012, 0.918,
             "Thin blue lines = individual subjects (ICE); orange = average effect (PDP); band = 10–90th percentile.",
             fontsize=9, color=INK2, va="top")
    fig.supylabel(f"predicted {target.upper()}", fontsize=9.5)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(FIGS / f"fig5_ice_{target}.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    feats, targets, covar = load()
    print("=== pooled out-of-fold performance ===", flush=True)
    r2 = oof_r2(feats, targets, covar)
    for target, kind in (("mmse", "reg"), ("cdr_bin", "clf")):
        print(f"=== explaining {target} ===", flush=True)
        est, Xi, y, sv, mean_abs = explain(feats, targets, covar, target, kind)
        if target == "mmse":
            figures(target, Xi, sv, mean_abs, r2)
    print("done", flush=True)
