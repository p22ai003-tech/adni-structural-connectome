#!/usr/bin/env python
"""Final-model discrimination: one accepted model per target, no rung ladder.

Model (the configuration that survived the improvement sweep):
    in-fold median imputation -> in-fold residualisation of the 103 connectome
    features on [1, age, sex] (beta fit on the training fold only; age and sex
    stay in the design matrix, raw) -> ExtraTreesClassifier(400, leaf 5,
    class_weight="balanced"), repeated 5x5 StratifiedKFold over seeds
    11/23/37/51/73, out-of-fold probabilities averaged across seeds.

Sex is coded 1 = female, 0 = male. With ExtraTrees the direction is not
cosmetic: flipping it mirrors the random split thresholds and moves the
reported AUCs in the third decimal.

Targets, both three-class so the two dashboard columns mirror exactly:
    mmse_band   Low (impaired) / Mid / High (intact)   MMSE tertiles, 71/68/63
    cdr_3level  CDR 0 / CDR 0.5 / CDR >=1              106/67/29

Outputs (hcp_analysis/):
    final_model_table.csv     long-format render table (target,row,col,value)
    final_model_metrics.csv   tidy numeric metrics incl. per diagnostic group
    final_model_classwise.csv per-class auc / pr_auc / chance / support
    final_model_curves.csv    ROC + PR curve points for both targets
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

H = Path("/home/ec2-user/exp/hcp_analysis")
SEEDS = [11, 23, 37, 51, 73]
GROUPS = ["CN", "MCI", "AD"]
MIN_PER_SIDE = 10  # below this the bootstrap interval spans most of the range
BOOT = 2000
CURVE_POINTS = 120


def models(seed):
    """Same three estimators as the published ladder, so the cross-check rows compare like with like."""
    return {
        "extra_trees": ExtraTreesClassifier(n_estimators=400, min_samples_leaf=5,
                                            class_weight="balanced", random_state=seed, n_jobs=8),
        "logreg": Pipeline([("sc", StandardScaler()),
                            ("m", LogisticRegressionCV(Cs=5, class_weight="balanced",
                                                       max_iter=4000, random_state=seed))]),
        "hist_gb": HistGradientBoostingClassifier(max_depth=3, max_iter=250, learning_rate=0.05,
                                                  l2_regularization=1.0, class_weight="balanced",
                                                  random_state=seed),
    }


def load():
    feats = pd.read_csv(H / "ml_feature_matrix.csv", index_col=0)
    t = pd.read_csv(H / "ml_targets.csv", index_col=0)
    cov = pd.DataFrame({"age": t.age,
                        "sex": (t.sex.astype(str).str.upper().str[0] == "F").astype(float)},
                       index=t.index)
    return feats, t, cov


def build_targets(feats, t):
    """Both targets on the identical subject index, so the two columns are comparable."""
    idx = t.index.intersection(feats.index)
    band = pd.qcut(t.loc[idx, "mmse"], 3, labels=False, duplicates="drop").astype(int).values
    cdr = t.loc[idx, "cdr"]
    lvl = np.where(cdr == 0, 0, np.where(cdr == 0.5, 1, 2))
    return idx, {
        "mmse_band": dict(y=band, labels=["Low (impaired)", "Mid", "High (intact)"],
                          title="MMSE severity bands"),
        "cdr_3level": dict(y=lvl, labels=["CDR 0", "CDR 0.5", "CDR ≥ 1"],
                           title="CDR three levels"),
    }


def oof_proba(X, y, model_key):
    """Pooled out-of-fold probabilities: impute in fold, residualise in fold, average over seeds."""
    proba = np.zeros((len(y), 3))
    bal = []
    for seed in SEEDS:
        est = models(seed)[model_key]
        oof = np.zeros_like(proba)
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
            imp = SimpleImputer(strategy="median")
            Xtr = imp.fit_transform(X.iloc[tr])
            Xte = imp.transform(X.iloc[te])
            # residualise the connectome features only; age/sex pass through raw
            A = np.column_stack([np.ones(len(tr)), Xtr[:, 0], Xtr[:, 1]])
            B = np.column_stack([np.ones(len(te)), Xte[:, 0], Xte[:, 1]])
            coef, _, _, _ = np.linalg.lstsq(A, Xtr[:, 2:], rcond=None)
            Xtr = np.column_stack([Xtr[:, :2], Xtr[:, 2:] - A @ coef])
            Xte = np.column_stack([Xte[:, :2], Xte[:, 2:] - B @ coef])
            est.fit(Xtr, y[tr])
            oof[te] = est.predict_proba(Xte)
        proba += oof / len(SEEDS)
        bal.append(float(balanced_accuracy_score(y, oof.argmax(1))))
    return proba, bal


def scored(y_true_bin, score, ci=False, rng=None):
    """AUC, average precision and (optionally) a bootstrap interval.

    Returns reason "absent" when the class simply does not occur here and
    "small" when it occurs but too rarely to carry information — two very
    different kinds of blank that must not be printed the same way.
    """
    pos, neg = int(y_true_bin.sum()), int((1 - y_true_bin).sum())
    if pos == 0 or neg == 0:
        return dict(auc=None, pr_auc=None, lo=None, hi=None, n_pos=pos, n_neg=neg, reason="absent")
    if pos < MIN_PER_SIDE or neg < MIN_PER_SIDE:
        return dict(auc=None, pr_auc=None, lo=None, hi=None, n_pos=pos, n_neg=neg, reason="small")
    out = dict(auc=float(roc_auc_score(y_true_bin, score)),
               pr_auc=float(average_precision_score(y_true_bin, score)),
               lo=None, hi=None, n_pos=pos, n_neg=neg, reason="ok")
    if ci:
        ip, ineg = np.flatnonzero(y_true_bin == 1), np.flatnonzero(y_true_bin == 0)
        draws = []
        for _ in range(BOOT):
            b = np.concatenate([rng.choice(ip, len(ip)), rng.choice(ineg, len(ineg))])
            draws.append(roc_auc_score(y_true_bin[b], score[b]))
        out["lo"], out["hi"] = (float(np.percentile(draws, 2.5)),
                                float(np.percentile(draws, 97.5)))
    return out


def oriented(y_true_bin, score):
    """AUC of a fixed external score, read in whichever direction predicts the class."""
    pos, neg = int(y_true_bin.sum()), int((1 - y_true_bin).sum())
    if pos < MIN_PER_SIDE or neg < MIN_PER_SIDE:
        return None
    a = float(roc_auc_score(y_true_bin, score))
    return max(a, 1.0 - a)


def thin(xs, ys, kind, cls, label):
    step = max(1, len(xs) // CURVE_POINTS)
    keep = sorted(set(range(0, len(xs), step)) | {len(xs) - 1})  # never lose the endpoint
    return [dict(kind=kind, cls=cls, label=label, x=float(xs[i]), y=float(ys[i]))
            for i in keep]


def main() -> None:
    feats, t, cov = load()
    idx, targets = build_targets(feats, t)
    X = pd.concat([cov.loc[idx], feats.loc[idx]], axis=1)
    grp = t.loc[idx, "group"].astype(str).values
    group_n = {g: int((grp == g).sum()) for g in GROUPS}
    severity = np.select([grp == "CN", grp == "MCI", grp == "AD"], [0.0, 1.0, 2.0])
    is_ad = (grp == "AD").astype(int)
    rng = np.random.default_rng(0)

    metrics, classwise, curves, table = [], [], [], []

    for tkey, spec in targets.items():
        y, labels = spec["y"], spec["labels"]
        onehot = np.eye(3)[y]
        proba = {m: oof_proba(X, y, m) for m in ("extra_trees", "logreg", "hist_gb")}
        p, bal = proba["extra_trees"]
        print(f"[{tkey}] pooled OOF done", flush=True)

        res = {}  # (scope, class) -> scored() dict
        for c in range(3):
            res[("All", c)] = scored(onehot[:, c], p[:, c], ci=True, rng=rng)
            classwise.append(dict(target=tkey, cls=c, label=labels[c],
                                  auc=res[("All", c)]["auc"], pr_auc=res[("All", c)]["pr_auc"],
                                  baseline_pr=float(onehot[:, c].mean()),
                                  support=int(onehot[:, c].sum())))
            fpr, tpr, _ = roc_curve(onehot[:, c], p[:, c])
            prec, rec, _ = precision_recall_curve(onehot[:, c], p[:, c])
            curves += [dict(target=tkey, **r) for r in thin(fpr, tpr, "roc", c, labels[c])]
            curves += [dict(target=tkey, **r) for r in thin(rec, prec, "pr", c, labels[c])]
            for g in GROUPS:
                m = grp == g
                res[(g, c)] = scored(onehot[m, c], p[m, c], ci=True, rng=rng)

        macro_auc = float(roc_auc_score(y, p, multi_class="ovr", average="macro"))
        macro_ap = float(np.mean([average_precision_score(onehot[:, c], p[:, c]) for c in range(3)]))

        for scope in ["All"] + GROUPS:
            for c in range(3):
                r = res[(scope, c)]
                metrics.append(dict(target=tkey, scope=scope, cls=c, cls_label=labels[c],
                                    auc=r["auc"], pr_auc=r["pr_auc"], ci_lo=r["lo"], ci_hi=r["hi"],
                                    n_pos=r["n_pos"], n_neg=r["n_neg"], reason=r["reason"]))
            aucs = [res[(scope, c)]["auc"] for c in range(3)]
            metrics.append(dict(target=tkey, scope=scope, cls=-1, cls_label="macro",
                                auc=macro_auc if scope == "All" else
                                (float(np.mean(aucs)) if all(a is not None for a in aucs) else None),
                                pr_auc=macro_ap if scope == "All" else None,
                                ci_lo=None, ci_hi=None,
                                n_pos=len(y) if scope == "All" else group_n[scope], n_neg=0,
                                reason="ok"))

        # ---------- render table ----------
        cols = [("Macro", -1)] + [(labels[c], c) for c in range(3)]

        def add(order, row_label, getter):
            for ci_, (col_label, c) in enumerate(cols):
                table.append(dict(target=tkey, row_order=order, row_label=row_label,
                                  col_order=ci_, col_label=col_label, value=getter(c)))

        def auc_cell(scope, c, with_ci=True):
            r = res[(scope, c)]
            if r["reason"] == "absent":
                return "no cases"
            if r["reason"] == "small":
                return f"— ({min(r['n_pos'], r['n_neg'])})"
            return f"{r['auc']:.3f}"

        def ci_cell(scope, c):
            r = res[(scope, c)]
            return "—" if r["lo"] is None else f"{r['lo']:.2f}–{r['hi']:.2f}"

        def macro_cell(scope):
            aucs = [res[(scope, c)]["auc"] for c in range(3)]
            if scope == "All":
                return f"{macro_auc:.3f}"
            return f"{np.mean(aucs):.3f}" if all(a is not None for a in aucs) else "—"

        scopes = [("All", "AUC · all 202")] + [
            (g, f"AUC · within {g} ({group_n[g]})") for g in GROUPS]
        for i, (scope, lbl) in enumerate(scopes):
            add(2 * i, lbl,
                lambda c, scope=scope: macro_cell(scope) if c < 0 else auc_cell(scope, c))
            add(2 * i + 1, "      95% interval",
                lambda c, scope=scope: "—" if c < 0 else ci_cell(scope, c))
        add(8, "PR-AUC · all 202",
            lambda c: f"{macro_ap:.3f}" if c < 0 else f"{res[('All', c)]['pr_auc']:.3f}")
        add(9, "PR-AUC chance = prevalence",
            lambda c: "—" if c < 0 else f"{onehot[:, c].mean():.3f}")
        add(10, "balanced accuracy",
            lambda c: f"{np.mean(bal):.3f}" if c < 0 else "—")
        add(11, "      across-seed sd",
            lambda c: f"{np.std(bal):.3f}" if c < 0 else "—")
        add(12, "class size",
            lambda c: "202" if c < 0 else str(int(onehot[:, c].sum())))

        # ---------- what the discrimination is actually made of ----------
        nonad = grp != "AD"
        def cell(v):
            return "—" if v is None else f"{v:.3f}"

        add(13, "AUC · diagnosis label only",
            lambda c: "—" if c < 0 else cell(oriented(onehot[:, c], severity)))
        add(14, "AUC · non-AD subjects only",
            lambda c: "—" if c < 0 else cell(scored(onehot[nonad, c], p[nonad, c])["auc"]))
        add(15, "AUC · same score detects AD",
            lambda c: "—" if c < 0 else cell(oriented(is_ad, p[:, c])))

        for j, (m, nice) in enumerate((("logreg", "logistic regression"),
                                       ("hist_gb", "gradient boosting"))):
            q = proba[m][0]
            add(16 + j, f"AUC · {nice}",
                lambda c, q=q: f"{roc_auc_score(y, q, multi_class='ovr', average='macro'):.3f}"
                if c < 0 else f"{roc_auc_score(onehot[:, c], q[:, c]):.3f}")

    pd.DataFrame(table).to_csv(H / "final_model_table.csv", index=False)
    pd.DataFrame(metrics).to_csv(H / "final_model_metrics.csv", index=False)
    pd.DataFrame(classwise).to_csv(H / "final_model_classwise.csv", index=False)
    pd.DataFrame(curves).to_csv(H / "final_model_curves.csv", index=False)
    tab = pd.DataFrame(table)
    for tk in tab.target.unique():
        w = tab[tab.target == tk].pivot(index=["row_order", "row_label"],
                                        columns="col_label", values="value")
        order = tab[tab.target == tk].sort_values("col_order").col_label.unique()
        print("=" * 20, tk, flush=True)
        print(w[list(order)].to_string(), flush=True)


if __name__ == "__main__":
    main()
