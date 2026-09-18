#!/usr/bin/env python
"""MMSE as a 3-class quantile-bucketed target (the ceiling effect makes 4 buckets
impossible - too many ties at 29/30 collapse the top two bins).

Reports, per rung and model: macro/weighted AUC and PR-AUC (average precision),
balanced accuracy, and PER-CLASS one-vs-rest AUC + PR-AUC. Then per-class SHAP
so we can see which features drive each severity band.

Outputs (hcp_analysis/):
  mmse_bucket_definition.csv     bin edges, counts, group mix
  mmse_bucket_results.csv        rung x model x metric (incl. per-class)
  mmse_bucket_classwise.csv      per-class AUC / PR-AUC / support for the best model
  mmse_bucket_shap_classwise.csv per-class mean |SHAP| per feature
  mmse_bucket_curves.csv         ROC + PR curve points per class (for the dashboard)
  figs/fig6_mmse_buckets.png     per-class ROC + PR + class-wise SHAP
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             precision_recall_curve, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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
N_BUCKETS = 3
LABELS = ["Low (impaired)", "Mid", "High (intact)"]
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
CLASS_COLORS = [ORANGE, "#eda100", BLUE]
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3e0"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "figure.facecolor": SURF, "axes.facecolor": SURF})


def deframe(ax, axis="both"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def rung_columns(feats, rung):
    f1 = [c for c in feats.columns if c.startswith("WB_")]
    f2 = [c for c in feats.columns if c.startswith(("SR_", "LR_"))]
    f3 = [c for c in feats.columns if c.startswith(("SREXC_", "LREXC_"))]
    f4 = [c for c in feats.columns if c.startswith("ARCH_")]
    return {"F0": [], "F0+F1": f1, "F0-F2": f1 + f2, "F0-F3": f1 + f2 + f3,
            "F0-F4": f1 + f2 + f3 + f4}[rung]


def models(seed):
    return {
        "logreg": Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
                            ("m", LogisticRegressionCV(Cs=5, class_weight="balanced", max_iter=4000,
                                                       random_state=seed))]),
        "hist_gb": HistGradientBoostingClassifier(max_depth=3, max_iter=250, learning_rate=0.05,
                                                  l2_regularization=1.0, class_weight="balanced",
                                                  random_state=seed),
        "extra_trees": Pipeline([("imp", SimpleImputer(strategy="median")),
                                 ("m", ExtraTreesClassifier(n_estimators=400, min_samples_leaf=5,
                                                            class_weight="balanced", random_state=seed,
                                                            n_jobs=8))]),
    }


def main() -> None:
    feats = pd.read_csv(OUT / "ml_feature_matrix.csv", index_col=0)
    targets = pd.read_csv(OUT / "ml_targets.csv", index_col=0)
    covar = pd.DataFrame({"age": targets.age,
                          "sex": (targets.sex.astype(str).str.upper().str[0] == "F").astype(float)},
                         index=targets.index)
    m = targets.dropna(subset=["mmse"])
    idx = m.index.intersection(feats.index)
    m = m.loc[idx]
    bucket = pd.qcut(m.mmse, N_BUCKETS, labels=False, duplicates="drop")
    n_classes = int(bucket.nunique())
    edges = pd.qcut(m.mmse, N_BUCKETS, duplicates="drop").cat.categories
    definition = pd.DataFrame({
        "class": range(n_classes), "label": LABELS[:n_classes],
        "range": [str(e) for e in edges],
        "n": [int((bucket == c).sum()) for c in range(n_classes)],
        **{g: [int(((bucket == c) & (m.group == g)).sum()) for c in range(n_classes)]
           for g in ("CN", "MCI", "AD")},
    })
    definition.to_csv(OUT / "mmse_bucket_definition.csv", index=False)
    print(definition.to_string(index=False), flush=True)

    y = bucket.astype(int)
    rows, classwise_rows = [], []
    best = None
    for rung in ("F0", "F0+F1", "F0-F2", "F0-F3", "F0-F4"):
        X = pd.concat([covar.loc[idx], feats.loc[idx, rung_columns(feats, rung)]], axis=1)
        for mname in models(0):
            proba_acc = np.zeros((len(y), n_classes))
            bal = []
            for seed in SEEDS:
                est = models(seed)[mname]
                oof = np.zeros((len(y), n_classes))
                for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
                    est.fit(X.iloc[tr], y.iloc[tr])
                    oof[te] = est.predict_proba(X.iloc[te])
                proba_acc += oof / len(SEEDS)
                bal.append(balanced_accuracy_score(y, oof.argmax(1)))
            macro_auc = roc_auc_score(y, proba_acc, multi_class="ovr", average="macro")
            weighted_auc = roc_auc_score(y, proba_acc, multi_class="ovr", average="weighted")
            onehot = np.eye(n_classes)[y]
            macro_ap = average_precision_score(onehot, proba_acc, average="macro")
            rec = dict(rung=rung, model=mname, n=len(y), n_features=X.shape[1],
                       macro_auc=macro_auc, weighted_auc=weighted_auc, macro_pr_auc=macro_ap,
                       bal_acc=float(np.mean(bal)), bal_acc_sd=float(np.std(bal)))
            for c in range(n_classes):
                rec[f"auc_class{c}"] = roc_auc_score(onehot[:, c], proba_acc[:, c])
                rec[f"pr_auc_class{c}"] = average_precision_score(onehot[:, c], proba_acc[:, c])
            rows.append(rec)
            print(f"  {rung:6} {mname:12} macroAUC={macro_auc:.3f} macroPR={macro_ap:.3f} "
                  f"bal={rec['bal_acc']:.3f} | per-class AUC "
                  + " ".join(f"{rec[f'auc_class{c}']:.3f}" for c in range(n_classes)), flush=True)
            if best is None or macro_auc > best[0]:
                best = (macro_auc, rung, mname, proba_acc.copy(), X.copy())

    pd.DataFrame(rows).to_csv(OUT / "mmse_bucket_results.csv", index=False)
    _, brung, bmodel, bproba, bX = best
    print(f"\nbest: {bmodel} @ {brung} macroAUC={best[0]:.3f}", flush=True)

    onehot = np.eye(n_classes)[y]
    curves = []
    for c in range(n_classes):
        fpr, tpr, _ = roc_curve(onehot[:, c], bproba[:, c])
        step = max(1, len(fpr) // 120)
        for i in range(0, len(fpr), step):
            curves.append(dict(kind="roc", cls=c, label=LABELS[c], x=float(fpr[i]), y=float(tpr[i])))
        prec, rec_, _ = precision_recall_curve(onehot[:, c], bproba[:, c])
        step = max(1, len(prec) // 120)
        for i in range(0, len(prec), step):
            curves.append(dict(kind="pr", cls=c, label=LABELS[c], x=float(rec_[i]), y=float(prec[i])))
        classwise_rows.append(dict(
            cls=c, label=LABELS[c], support=int(onehot[:, c].sum()),
            auc=roc_auc_score(onehot[:, c], bproba[:, c]),
            pr_auc=average_precision_score(onehot[:, c], bproba[:, c]),
            baseline_pr=float(onehot[:, c].mean()), model=bmodel, rung=brung))
    pd.DataFrame(curves).to_csv(OUT / "mmse_bucket_curves.csv", index=False)
    cw = pd.DataFrame(classwise_rows)
    cw.to_csv(OUT / "mmse_bucket_classwise.csv", index=False)
    print(cw.round(3).to_string(index=False), flush=True)

    # per-class SHAP on a refit model
    imp = SimpleImputer(strategy="median")
    Xi = pd.DataFrame(imp.fit_transform(bX), columns=bX.columns, index=bX.index)
    est = ExtraTreesClassifier(n_estimators=600, min_samples_leaf=5, class_weight="balanced",
                               random_state=11, n_jobs=8).fit(Xi, y)
    sv = shap.TreeExplainer(est).shap_values(Xi)
    if isinstance(sv, list):
        per_class = sv
    else:
        per_class = [sv[:, :, c] for c in range(sv.shape[2])]
    fam = lambda c: ("covariate" if c in ("age", "sex") else
                     "F1 whole-brain" if c.startswith("WB_") else
                     "F2 SR/LR" if c.startswith(("SR_", "LR_")) else
                     "F3 exceptions" if c.startswith(("SREXC_", "LREXC_")) else "F4 architecture")
    shap_rows = []
    for c in range(n_classes):
        s = pd.Series(np.abs(per_class[c]).mean(axis=0), index=Xi.columns).sort_values(ascending=False)
        for f, v in s.head(15).items():
            shap_rows.append(dict(cls=c, label=LABELS[c], feature=f, mean_abs_shap=float(v), family=fam(f)))
        print(f"\n  class {c} ({LABELS[c]}) top drivers: " + ", ".join(s.head(5).index), flush=True)
    shap_df = pd.DataFrame(shap_rows)
    shap_df.to_csv(OUT / "mmse_bucket_shap_classwise.csv", index=False)

    # figure
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6))
    cur = pd.DataFrame(curves)
    ax = axes[0]
    for c in range(n_classes):
        d = cur[(cur.kind == "roc") & (cur.cls == c)]
        ax.plot(d.x, d.y, color=CLASS_COLORS[c], lw=2,
                label=f"{LABELS[c]} (AUC {cw.loc[c,'auc']:.2f})")
    ax.plot([0, 1], [0, 1], color=INK2, lw=1, ls="--")
    ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate")
    ax.set_title("ROC, one-vs-rest", loc="left", fontsize=11); ax.legend(frameon=False, fontsize=8.5)
    deframe(ax)
    ax = axes[1]
    for c in range(n_classes):
        d = cur[(cur.kind == "pr") & (cur.cls == c)]
        ax.plot(d.x, d.y, color=CLASS_COLORS[c], lw=2,
                label=f"{LABELS[c]} (AP {cw.loc[c,'pr_auc']:.2f}, base {cw.loc[c,'baseline_pr']:.2f})")
        ax.axhline(cw.loc[c, "baseline_pr"], color=CLASS_COLORS[c], lw=1, ls=":", alpha=0.6)
    ax.set_xlabel("recall"); ax.set_ylabel("precision")
    ax.set_title("Precision–recall (dotted = chance)", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8.5); deframe(ax)
    ax = axes[2]
    top = shap_df[shap_df.cls == 0].head(8).iloc[::-1]
    ax.barh(range(len(top)), top.mean_abs_shap, color=CLASS_COLORS[0], zorder=3)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels([f.replace("_", " ") for f in top.feature], fontsize=8)
    ax.set_xlabel("mean |SHAP|")
    ax.set_title(f"Drivers of the {LABELS[0]} class", loc="left", fontsize=11); deframe(ax)
    fig.text(0.012, 0.965, "MMSE as 3 equal-sized severity bands", fontsize=13, fontweight="bold", va="top")
    fig.text(0.012, 0.915,
             f"{bmodel} @ {brung}, n={len(y)}; macro AUC {best[0]:.3f}, macro PR-AUC "
             f"{pd.DataFrame(rows).query('rung==@brung and model==@bmodel').macro_pr_auc.iloc[0]:.3f}. "
             "Quantile bands make the classes equal-sized, so PR chance = 0.33 for each.",
             fontsize=9, color=INK2, va="top")
    fig.tight_layout(rect=[0, 0, 1, 0.87])
    fig.savefig(FIGS / "fig6_mmse_buckets.png", dpi=200)
    plt.close(fig)
    print("\nwrote figs/fig6_mmse_buckets.png", flush=True)


if __name__ == "__main__":
    main()
