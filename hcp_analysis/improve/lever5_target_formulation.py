#!/usr/bin/env python
"""LEVER 5 - target formulation.

(a) MMSE: rank-transform y in-fold (fit ExtraTrees on train ranks, predict,
    Spearman on raw y; R2 not comparable on the rank scale -- also report a
    supplementary back-mapped variant via the train inverse-CDF, which IS
    R2-comparable, using the same fitted models).
(b) MMSE ordinal 3-band (Low<=27 / Mid(27,29] / High(29,30], qcut as baseline):
    two cumulative binary classifiers P(band>=1), P(band>=2) (Frank & Hall
    reconstruction) vs direct 3-class, macro AUC. Baselines re-run here:
    ExtraTrees and HistGB at the F0+F1 rung (the rung the quoted 0.627/0.660
    numbers come from), exact replication of build_mmse_buckets.py protocol
    (StratifiedKFold on band, seeds [11,23,37,51,73], seed-averaged proba).
(c) CDR: in-fold CalibratedClassifierCV (sigmoid, isotonic) around the baseline
    ExtraTrees pipeline -- AUC change + Brier score.

Fixed protocol: covariates age + sex(F=1) prepended; repeated stratified 5-fold
CV seeds [11,23,37,51,73]; MMSE stratified on group + (mmse>median); CDR on
cdr_bin; all fitted preprocessing inside the training fold; pooled OOF metrics
per seed, averaged over seeds.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import scipy.stats as sps
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (ExtraTreesClassifier, ExtraTreesRegressor,
                              HistGradientBoostingClassifier)
from sklearn.impute import SimpleImputer
from sklearn.metrics import brier_score_loss, r2_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

OUT = "/home/ec2-user/exp/hcp_analysis"
SEEDS = [11, 23, 37, 51, 73]

feats = pd.read_csv(f"{OUT}/ml_feature_matrix.csv", index_col=0)
targets = pd.read_csv(f"{OUT}/ml_targets.csv", index_col=0)
idx = targets.index.intersection(feats.index)
targets, feats = targets.loc[idx], feats.loc[idx]
covar = pd.DataFrame({"age": targets.age,
                      "sex": (targets.sex.astype(str).str.upper().str[0] == "F").astype(float)},
                     index=idx)
X_full = pd.concat([covar, feats], axis=1)                      # 105 cols
wb = [c for c in feats.columns if c.startswith("WB_")]
X_f01 = pd.concat([covar, feats[wb]], axis=1)                   # F0+F1 rung

y_mmse = targets.mmse.astype(float)
strat_mmse = targets.group.astype(str) + "|" + (y_mmse > y_mmse.median()).astype(str)
y_cdr = targets.cdr_bin.astype(int)

def et_reg(seed):
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("m", ExtraTreesRegressor(n_estimators=400, min_samples_leaf=5,
                                               random_state=seed, n_jobs=8))])

def et_clf(seed):
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("m", ExtraTreesClassifier(n_estimators=400, min_samples_leaf=5,
                                                class_weight="balanced",
                                                random_state=seed, n_jobs=8))])

def hgb_clf(seed):
    return HistGradientBoostingClassifier(max_depth=3, max_iter=250, learning_rate=0.05,
                                          l2_regularization=1.0, class_weight="balanced",
                                          random_state=seed)

# ---------------------------------------------------------------- (a) MMSE rank
print("=" * 72)
print("(a) MMSE regression: raw-y baseline vs in-fold rank-transformed y")
rows = {k: {"rho": [], "r2": []} for k in ("baseline_raw_y", "rank_y", "rank_y_backmap")}
for seed in SEEDS:
    oof_base = np.zeros(len(idx)); oof_rank = np.zeros(len(idx)); oof_back = np.zeros(len(idx))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X_full, strat_mmse):
        ytr = y_mmse.iloc[tr]
        est = et_reg(seed); est.fit(X_full.iloc[tr], ytr)
        oof_base[te] = est.predict(X_full.iloc[te])
        r_tr = sps.rankdata(ytr)                                  # train-fold ranks only
        est_r = et_reg(seed); est_r.fit(X_full.iloc[tr], r_tr)
        pr = est_r.predict(X_full.iloc[te])
        oof_rank[te] = pr
        # supplementary: map predicted ranks back through the train inverse-CDF
        ys = np.sort(ytr.values)
        oof_back[te] = np.interp(pr, np.sort(r_tr), ys)
    rows["baseline_raw_y"]["rho"].append(sps.spearmanr(y_mmse, oof_base).statistic)
    rows["baseline_raw_y"]["r2"].append(r2_score(y_mmse, oof_base))
    rows["rank_y"]["rho"].append(sps.spearmanr(y_mmse, oof_rank).statistic)
    rows["rank_y"]["r2"].append(np.nan)                           # rank scale: R2 not comparable
    rows["rank_y_backmap"]["rho"].append(sps.spearmanr(y_mmse, oof_back).statistic)
    rows["rank_y_backmap"]["r2"].append(r2_score(y_mmse, oof_back))
for k, v in rows.items():
    print(f"  {k:18} pooled-OOF Spearman={np.mean(v['rho']):.4f} (sd {np.std(v['rho']):.4f})"
          f"  R2={np.mean(v['r2']):.4f}" if not np.isnan(v["r2"]).any() else
          f"  {k:18} pooled-OOF Spearman={np.mean(v['rho']):.4f} (sd {np.std(v['rho']):.4f})"
          f"  R2=n/a (predictions on rank scale)", flush=True)

# ------------------------------------------------------- (b) ordinal 3-band
print("=" * 72)
print("(b) MMSE 3-band: cumulative-binary ordinal vs direct 3-class (macro AUC)")
band = pd.qcut(y_mmse, 3, labels=False, duplicates="drop").astype(int)  # 0 Low 1 Mid 2 High
print("  band counts:", band.value_counts().sort_index().to_dict())
onehot = np.eye(3)[band]

def run_direct(make, X):
    acc = np.zeros((len(idx), 3)); per_seed = []
    for seed in SEEDS:
        oof = np.zeros((len(idx), 3))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, band):
            est = make(seed); est.fit(X.iloc[tr], band.iloc[tr])
            oof[te] = est.predict_proba(X.iloc[te])
        acc += oof / len(SEEDS)
        per_seed.append(roc_auc_score(band, oof, multi_class="ovr", average="macro"))
    return roc_auc_score(band, acc, multi_class="ovr", average="macro"), np.mean(per_seed), np.std(per_seed)

def run_cumulative(make, X):
    acc = np.zeros((len(idx), 3)); per_seed = []
    for seed in SEEDS:
        oof = np.zeros((len(idx), 3))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, band):
            p = np.zeros((len(te), 2))
            for j, thr in enumerate((1, 2)):                      # P(band>=1), P(band>=2)
                est = make(seed); est.fit(X.iloc[tr], (band.iloc[tr] >= thr).astype(int))
                p[:, j] = est.predict_proba(X.iloc[te])[:, 1]
            p[:, 1] = np.minimum(p[:, 0], p[:, 1])                # enforce monotone cumulative
            pr = np.column_stack([1 - p[:, 0], p[:, 0] - p[:, 1], p[:, 1]])
            oof[te] = np.clip(pr, 0, None) / np.clip(pr, 0, None).sum(1, keepdims=True)
        acc += oof / len(SEEDS)
        per_seed.append(roc_auc_score(band, oof, multi_class="ovr", average="macro"))
    return roc_auc_score(band, acc, multi_class="ovr", average="macro"), np.mean(per_seed), np.std(per_seed)

for name, fn, make, X in [
        ("direct 3-class ExtraTrees  @F0+F1", run_direct, et_clf, X_f01),
        ("direct 3-class HistGB      @F0+F1", run_direct, hgb_clf, X_f01),
        ("cumulative ExtraTrees      @F0+F1", run_cumulative, et_clf, X_f01),
        ("cumulative HistGB          @F0+F1", run_cumulative, hgb_clf, X_f01),
        ("cumulative ExtraTrees      @full ", run_cumulative, et_clf, X_full)]:
    a_avg, a_ps, a_sd = fn(make, X)
    print(f"  {name}: macroAUC(seed-avg proba)={a_avg:.4f}"
          f" | mean per-seed macroAUC={a_ps:.4f} (sd {a_sd:.4f})", flush=True)

# ------------------------------------------------------------ (c) CDR calib
print("=" * 72)
print("(c) CDR: in-fold probability calibration (pooled OOF AUC + Brier)")
res = {k: {"auc": [], "brier": []} for k in ("baseline", "sigmoid", "isotonic")}
for seed in SEEDS:
    oof = {k: np.zeros(len(idx)) for k in res}
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X_full, y_cdr):
        base = et_clf(seed); base.fit(X_full.iloc[tr], y_cdr.iloc[tr])
        oof["baseline"][te] = base.predict_proba(X_full.iloc[te])[:, 1]
        for meth in ("sigmoid", "isotonic"):
            cal = CalibratedClassifierCV(et_clf(seed), method=meth, cv=5)
            cal.fit(X_full.iloc[tr], y_cdr.iloc[tr])
            oof[meth][te] = cal.predict_proba(X_full.iloc[te])[:, 1]
    for k in res:
        res[k]["auc"].append(roc_auc_score(y_cdr, oof[k]))
        res[k]["brier"].append(brier_score_loss(y_cdr, oof[k]))
for k, v in res.items():
    print(f"  {k:9} pooled-OOF AUC={np.mean(v['auc']):.4f} (sd {np.std(v['auc']):.4f})"
          f"  Brier={np.mean(v['brier']):.4f} (sd {np.std(v['brier']):.4f})", flush=True)
print("done")
