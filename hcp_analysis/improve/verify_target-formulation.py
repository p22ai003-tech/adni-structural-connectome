#!/usr/bin/env python
"""Adversarial verification of LEVER 5 (target formulation) report.

Independent re-implementation (not a copy of lever5_target_formulation.py).
Fixed protocol: covariates age + sex(F=1) prepended; repeated stratified 5-fold
CV, seeds [11,23,37,51,73]; MMSE regression stratified on group + (mmse>median);
CDR on cdr_bin; band task stratified on band (matches the scheme that produced
the quoted 0.627/0.660 baselines in build_mmse_buckets.py); all fitted
preprocessing inside training folds; pooled-OOF metrics per seed, averaged.

A) FULL RE-RUN (claimed WIN): MMSE raw-y ET baseline vs in-fold rank-y ET,
   plus the back-mapped supplementary. Per-seed Spearman printed.
B) SPOT-CHECK (claimed NO WIN): direct 3-class ET & HistGB @F0+F1 vs
   cumulative (Frank-Hall) HistGB @F0+F1, macro AUC.
C) SPOT-CHECK (claimed NO WIN): CDR baseline ET vs in-fold sigmoid/isotonic
   CalibratedClassifierCV; AUC + Brier. Baseline run with and without
   class_weight='balanced' to pin down which matches the quoted 0.602.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (ExtraTreesClassifier, ExtraTreesRegressor,
                              HistGradientBoostingClassifier)
from sklearn.impute import SimpleImputer
from sklearn.metrics import brier_score_loss, r2_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
import time

D = "/home/ec2-user/exp/hcp_analysis"
SEEDS = [11, 23, 37, 51, 73]
t0 = time.time()

feats = pd.read_csv(f"{D}/ml_feature_matrix.csv", index_col=0)
targ = pd.read_csv(f"{D}/ml_targets.csv", index_col=0)
idx = targ.index.intersection(feats.index)
feats, targ = feats.loc[idx], targ.loc[idx]
cov = pd.DataFrame({"age": targ.age.astype(float),
                    "sex": (targ.sex.astype(str).str.upper().str[0] == "F").astype(float)}, index=idx)
Xf = pd.concat([cov, feats], axis=1)
Xwb = pd.concat([cov, feats[[c for c in feats.columns if c.startswith("WB_")]]], axis=1)
y = targ.mmse.astype(float)
strat = targ.group.astype(str) + "_" + (y > y.median()).astype(str)
ycdr = targ.cdr_bin.astype(int)
band = pd.qcut(y, 3, labels=False, duplicates="drop").astype(int)
print(f"n={len(idx)} full={Xf.shape[1]} F0+F1={Xwb.shape[1]} bands={band.value_counts().sort_index().to_dict()}")

def reg(s):
    return Pipeline([("i", SimpleImputer(strategy="median")),
                     ("m", ExtraTreesRegressor(n_estimators=400, min_samples_leaf=5, random_state=s, n_jobs=8))])

def clf(s, bal="balanced"):
    return Pipeline([("i", SimpleImputer(strategy="median")),
                     ("m", ExtraTreesClassifier(n_estimators=400, min_samples_leaf=5, class_weight=bal,
                                                random_state=s, n_jobs=8))])

def hgb(s):
    return HistGradientBoostingClassifier(max_depth=3, max_iter=250, learning_rate=0.05,
                                          l2_regularization=1.0, class_weight="balanced", random_state=s)

# ---- A) MMSE rank transform: FULL RE-RUN --------------------------------
print("\n[A] MMSE raw-y vs rank-y (full re-run)")
per = {k: {"rho": [], "r2": []} for k in ("raw", "rank", "backmap")}
for s in SEEDS:
    p_raw, p_rank, p_back = (np.empty(len(idx)) for _ in range(3))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=s).split(Xf, strat):
        ytr = y.iloc[tr]
        m = reg(s).fit(Xf.iloc[tr], ytr)
        p_raw[te] = m.predict(Xf.iloc[te])
        rtr = rankdata(ytr)                       # fitted on TRAIN fold only
        mr = reg(s).fit(Xf.iloc[tr], rtr)
        pr = mr.predict(Xf.iloc[te])
        p_rank[te] = pr
        p_back[te] = np.interp(pr, np.sort(rtr), np.sort(ytr.values))
    per["raw"]["rho"].append(spearmanr(y, p_raw).statistic);   per["raw"]["r2"].append(r2_score(y, p_raw))
    per["rank"]["rho"].append(spearmanr(y, p_rank).statistic); per["rank"]["r2"].append(np.nan)
    per["backmap"]["rho"].append(spearmanr(y, p_back).statistic); per["backmap"]["r2"].append(r2_score(y, p_back))
for k, v in per.items():
    r2 = "n/a" if np.isnan(v["r2"]).any() else f"{np.mean(v['r2']):.4f}"
    print(f"  {k:8} Spearman={np.mean(v['rho']):.4f} (sd {np.std(v['rho']):.4f})  R2={r2}"
          f"  per-seed rho={[round(x,4) for x in v['rho']]}")
wins = sum(a > b for a, b in zip(per["rank"]["rho"], per["raw"]["rho"]))
print(f"  rank beats raw in {wins}/5 seeds; delta={np.mean(per['rank']['rho'])-np.mean(per['raw']['rho']):+.4f}")

# ---- B) 3-band spot-check ------------------------------------------------
print("\n[B] MMSE 3-band spot-check (macro AUC; seed-avg proba | mean per-seed)")
def run_band(make, X, cumulative):
    avg = np.zeros((len(idx), 3)); ps = []
    for s in SEEDS:
        oof = np.zeros((len(idx), 3))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=s).split(X, band):
            if cumulative:
                q = np.zeros((len(te), 2))
                for j, thr in enumerate((1, 2)):
                    e = make(s).fit(X.iloc[tr], (band.iloc[tr] >= thr).astype(int))
                    q[:, j] = e.predict_proba(X.iloc[te])[:, 1]
                q[:, 1] = np.minimum(q[:, 0], q[:, 1])
                pr = np.column_stack([1 - q[:, 0], q[:, 0] - q[:, 1], q[:, 1]])
                pr = np.clip(pr, 0, None); oof[te] = pr / pr.sum(1, keepdims=True)
            else:
                oof[te] = make(s).fit(X.iloc[tr], band.iloc[tr]).predict_proba(X.iloc[te])
        avg += oof / len(SEEDS)
        ps.append(roc_auc_score(band, oof, multi_class="ovr", average="macro"))
    return roc_auc_score(band, avg, multi_class="ovr", average="macro"), np.mean(ps)
for name, make, X, cum in [("direct ET     @F0+F1", clf, Xwb, False),
                           ("direct HistGB @F0+F1", hgb, Xwb, False),
                           ("cumul  HistGB @F0+F1", hgb, Xwb, True)]:
    a, p = run_band(make, X, cum)
    print(f"  {name}: {a:.4f} | {p:.4f}")

# ---- C) CDR calibration spot-check --------------------------------------
print("\n[C] CDR calibration spot-check (pooled OOF AUC / Brier)")
res = {k: {"auc": [], "br": []} for k in ("base_bal", "base_nobal", "sigmoid", "isotonic")}
for s in SEEDS:
    oof = {k: np.empty(len(idx)) for k in res}
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=s).split(Xf, ycdr):
        oof["base_bal"][te] = clf(s).fit(Xf.iloc[tr], ycdr.iloc[tr]).predict_proba(Xf.iloc[te])[:, 1]
        oof["base_nobal"][te] = clf(s, None).fit(Xf.iloc[tr], ycdr.iloc[tr]).predict_proba(Xf.iloc[te])[:, 1]
        for meth in ("sigmoid", "isotonic"):
            c = CalibratedClassifierCV(clf(s), method=meth, cv=5).fit(Xf.iloc[tr], ycdr.iloc[tr])
            oof[meth][te] = c.predict_proba(Xf.iloc[te])[:, 1]
    for k in res:
        res[k]["auc"].append(roc_auc_score(ycdr, oof[k]))
        res[k]["br"].append(brier_score_loss(ycdr, oof[k]))
for k, v in res.items():
    print(f"  {k:10} AUC={np.mean(v['auc']):.4f} (sd {np.std(v['auc']):.4f})  Brier={np.mean(v['br']):.4f} (sd {np.std(v['br']):.4f})")
print(f"\nruntime {time.time()-t0:.0f}s")
