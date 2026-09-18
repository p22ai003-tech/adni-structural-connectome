#!/usr/bin/env python
"""Explain v3: CDR ROC/PR curves (so both targets have a curve pair), and
per-subject SHAP beeswarm data per class, with diagnostic group attached.

Outputs (hcp_analysis/):
  cdr_curves.csv            ROC + PR curve points for the CDR binary model (pooled OOF)
  cdr_classwise.csv         per-class AUC / PR-AUC / support for CDR
  ml_shap_beeswarm_class.csv per-subject SHAP for the top-6 features of every
                             target x class, with diagnostic group + feature percentile
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold

OUT = Path("/home/ec2-user/exp/hcp_analysis")
SEEDS = [11, 23, 37, 51, 73]


def load():
    feats = pd.read_csv(OUT / "ml_feature_matrix.csv", index_col=0)
    targets = pd.read_csv(OUT / "ml_targets.csv", index_col=0)
    covar = pd.DataFrame({"age": targets.age,
                          "sex": (targets.sex.astype(str).str.upper().str[0] == "F").astype(float)},
                         index=targets.index)
    return feats, targets, covar


def cdr_curves(feats, targets, covar):
    y = targets["cdr_bin"].dropna()
    idx = y.index.intersection(feats.index)
    y = y.loc[idx].astype(int)
    X = pd.concat([covar.loc[idx], feats.loc[idx]], axis=1)
    proba = np.zeros(len(y))
    for seed in SEEDS:
        est = ExtraTreesClassifier(n_estimators=400, min_samples_leaf=5, class_weight="balanced",
                                   random_state=seed, n_jobs=8)
        oof = np.zeros(len(y))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
            imp = SimpleImputer(strategy="median")
            est.fit(imp.fit_transform(X.iloc[tr]), y.iloc[tr])
            oof[te] = est.predict_proba(imp.transform(X.iloc[te]))[:, 1]
        proba += oof / len(SEEDS)
    rows, cls_rows = [], []
    labels = {0: "CDR 0", 1: "CDR >= 0.5"}
    onehot = np.stack([1 - y.values, y.values], axis=1)
    probs = np.stack([1 - proba, proba], axis=1)
    for c in (0, 1):
        fpr, tpr, _ = roc_curve(onehot[:, c], probs[:, c])
        for i in range(0, len(fpr), max(1, len(fpr) // 120)):
            rows.append(dict(kind="roc", cls=c, label=labels[c], x=float(fpr[i]), y=float(tpr[i])))
        prec, rec, _ = precision_recall_curve(onehot[:, c], probs[:, c])
        for i in range(0, len(prec), max(1, len(prec) // 120)):
            rows.append(dict(kind="pr", cls=c, label=labels[c], x=float(rec[i]), y=float(prec[i])))
        cls_rows.append(dict(cls=c, label=labels[c], support=int(onehot[:, c].sum()),
                             auc=roc_auc_score(onehot[:, c], probs[:, c]),
                             pr_auc=average_precision_score(onehot[:, c], probs[:, c]),
                             baseline_pr=float(onehot[:, c].mean())))
    pd.DataFrame(rows).to_csv(OUT / "cdr_curves.csv", index=False)
    cw = pd.DataFrame(cls_rows)
    cw.to_csv(OUT / "cdr_classwise.csv", index=False)
    print(cw.round(3).to_string(index=False), flush=True)


def beeswarm_class(feats, targets, covar):
    out = []
    for target in ("mmse_band", "cdr_bin"):
        if target == "mmse_band":
            y0 = targets["mmse"].dropna()
            idx = y0.index.intersection(feats.index)
            y = pd.qcut(y0.loc[idx], 3, labels=False, duplicates="drop").astype(int)
            labels = {0: "Low (impaired)", 1: "Mid", 2: "High (intact)"}
        else:
            y0 = targets["cdr_bin"].dropna()
            idx = y0.index.intersection(feats.index)
            y = y0.loc[idx].astype(int)
            labels = {0: "CDR 0", 1: "CDR >= 0.5"}
        X = pd.concat([covar.loc[idx], feats.loc[idx]], axis=1)
        Xi = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(X),
                          columns=X.columns, index=X.index)
        est = ExtraTreesClassifier(n_estimators=600, min_samples_leaf=5, class_weight="balanced",
                                   random_state=11, n_jobs=8).fit(Xi, y)
        sv = shap.TreeExplainer(est).shap_values(Xi)
        per_class = sv if isinstance(sv, list) else [sv[:, :, c] for c in range(sv.shape[2])]
        grp = targets.loc[idx, "group"]
        pctl = Xi.rank(pct=True)
        for c, mat in enumerate(per_class):
            m = pd.DataFrame(mat, columns=Xi.columns, index=Xi.index)
            top6 = m.abs().mean().nlargest(6).index
            for f in top6:
                for sid in Xi.index:
                    out.append(dict(target=target, cls=c, cls_label=labels[c], feature=f,
                                    subject=sid, group=grp.loc[sid],
                                    shap=float(m.loc[sid, f]),
                                    feature_pctl=float(pctl.loc[sid, f])))
        print(f"beeswarm rows for {target}: done", flush=True)
    pd.DataFrame(out).to_csv(OUT / "ml_shap_beeswarm_class.csv", index=False)
    print(f"total beeswarm rows: {len(out)}", flush=True)


if __name__ == "__main__":
    feats, targets, covar = load()
    cdr_curves(feats, targets, covar)
    beeswarm_class(feats, targets, covar)
    print("done", flush=True)
