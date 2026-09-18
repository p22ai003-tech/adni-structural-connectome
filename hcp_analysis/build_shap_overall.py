#!/usr/bin/env python
"""Overall SHAP view + honest importance shares.

Two jobs:

1. OVERALL. One signed beeswarm per target, on a binary "impairment" contrast —
   MMSE lowest tertile vs the rest, and CDR >= 0.5 vs the rest (the locked
   binary target). Both encode impaired as 1, so a positive SHAP value means
   "pushes this subject toward impairment" on both sides and the two panels
   mirror each other honestly. For a binary fit the class-0 SHAP matrix is the
   exact negation of class-1, so only class 1 is emitted.

2. SHARES. The published class x group SHAP file kept only the top twelve
   features, so a percentage computed from it would be a share of twelve, not
   of the model. Here mean|SHAP| is computed over all 105 features and the
   share is taken against that full total before the file is truncated for
   display, which is the only way the bracketed percentages on the dashboard
   mean what they say.

Estimator matches the other explanation panels exactly (ExtraTrees 600 trees,
leaf 5, balanced, random_state 11, single in-sample fit on the median-imputed
matrix). Attribution therefore describes the fitted model; the AUCs elsewhere
on the page are out-of-fold and come from a different, 400-tree model.

Outputs (hcp_analysis/):
    ml_shap_overall_beeswarm.csv   per-subject signed SHAP, impairment contrast
    ml_shap_class_group.csv        regenerated, now carrying share_pct
    ml_shap_class_group_dir.csv    regenerated companion (unchanged content)
    ml_shap_family_group.csv       family shares per target x class x group
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer

H = Path("/home/ec2-user/exp/hcp_analysis")
GROUPS = ("CN", "MCI", "AD")
OVERALL_TOP = 8
SEEDS = [11, 23, 37, 51, 73]  # attribution is averaged over seeds: a single fit's ranking is a lottery
BREAKDOWN_TOP = 12

FAMILY = lambda c: ("covariate" if c in ("age", "sex") else
                    "F1 whole-brain" if c.startswith("WB_") else
                    "F2 SR/LR" if c.startswith(("SR_", "LR_")) else
                    "F3 exceptions" if c.startswith(("SREXC_", "LREXC_")) else "F4 architecture")


def load():
    feats = pd.read_csv(H / "ml_feature_matrix.csv", index_col=0)
    targets = pd.read_csv(H / "ml_targets.csv", index_col=0)
    covar = pd.DataFrame({"age": targets.age,
                          "sex": (targets.sex.astype(str).str.upper().str[0] == "F").astype(float)},
                         index=targets.index)
    return feats, targets, covar


def design(feats, targets, covar, idx):
    X = pd.concat([covar.loc[idx], feats.loc[idx]], axis=1)
    return pd.DataFrame(SimpleImputer(strategy="median").fit_transform(X),
                        columns=X.columns, index=X.index)


def fit_shap(Xi, y, return_est=False):
    """Mean SHAP over five seeds.

    SR_DMN_DIFF and WB_DMN_DIFF correlate at r = 0.98, so a single forest splits
    the credit between them essentially at random and the published ordering
    reproduced in only 28% of 200 seeds. Averaging removes that lottery; the
    per-seed spread is reported alongside so the remaining instability is visible.
    """
    ests, stacks = [], []
    for seed in SEEDS:
        est = ExtraTreesClassifier(n_estimators=600, min_samples_leaf=5, class_weight="balanced",
                                   random_state=seed, n_jobs=8).fit(Xi, y)
        sv = shap.TreeExplainer(est).shap_values(Xi)
        stacks.append(sv if isinstance(sv, list) else [sv[:, :, c] for c in range(sv.shape[2])])
        ests.append(est)
    per_class = [np.mean([st[c] for st in stacks], axis=0) for c in range(len(stacks[0]))]
    per_seed = [[st[c] for st in stacks] for c in range(len(stacks[0]))]
    return (ests, per_class, per_seed) if return_est else per_class


def overall(feats, targets, covar):
    """Signed per-subject SHAP for the impairment class of each target."""
    rows = []
    specs = {
        "mmse_low": ("MMSE lowest tertile (impaired)", "MMSE"),
        "cdr_impaired": ("CDR ≥ 0.5 (impaired)", "CDR"),
    }
    for target, (pos_label, short) in specs.items():
        if target == "mmse_low":
            y0 = targets["mmse"].dropna()
            idx = y0.index.intersection(feats.index)
            band = pd.qcut(y0.loc[idx], 3, labels=False, duplicates="drop").astype(int)
            y = (band == 0).astype(int)
        else:
            y0 = targets["cdr_bin"].dropna()
            idx = y0.index.intersection(feats.index)
            y = y0.loc[idx].astype(int)
        Xi = design(feats, targets, covar, idx)
        m = pd.DataFrame(fit_shap(Xi, y)[1], columns=Xi.columns, index=Xi.index)
        imp = m.abs().mean()
        share = 100.0 * imp / imp.sum()          # denominator = all 105 features
        pctl = Xi.rank(pct=True)                 # colour by rank, never by raw value
        grp = targets.loc[idx, "group"]
        for f in imp.sort_values(ascending=False).index[:OVERALL_TOP]:
            pos = float(m[f].clip(lower=0).sum())
            neg = float(-m[f].clip(upper=0).sum())
            tot = pos + neg
            for sid in Xi.index:
                rows.append(dict(target=target, target_label=pos_label, short=short,
                                 feature=f, family=FAMILY(f),
                                 share_pct=float(share[f]),
                                 pos_pct=100.0 * pos / tot if tot else np.nan,
                                 n_pos=int(y.sum()), n_neg=int((1 - y).sum()),
                                 subject=sid, group=grp.loc[sid],
                                 shap=float(m.loc[sid, f]),
                                 feature_pctl=float(pctl.loc[sid, f])))
        print(f"[overall] {target} done ({int(y.sum())} impaired / {int((1 - y).sum())})", flush=True)
    return pd.DataFrame(rows)


def class_group(feats, targets, covar):
    """Regenerate the class x group tables, carrying a share of the FULL feature set."""
    out_abs, out_dir, out_fam, bee, pdp = [], [], [], [], []
    for target in ("mmse_band", "cdr_3level"):
        if target == "mmse_band":
            y0 = targets["mmse"].dropna()
            idx = y0.index.intersection(feats.index)
            y = pd.qcut(y0.loc[idx], 3, labels=False, duplicates="drop").astype(int)
            labels = {0: "Low (impaired)", 1: "Mid", 2: "High (intact)"}
        else:
            cdr = targets["cdr"].dropna()
            idx = cdr.index.intersection(feats.index)
            y = pd.Series(np.where(cdr.loc[idx] == 0, 0, np.where(cdr.loc[idx] == 0.5, 1, 2)),
                          index=idx)
            labels = {0: "CDR 0", 1: "CDR 0.5", 2: "CDR ≥ 1"}
        Xi = design(feats, targets, covar, idx)
        ests, per_class, per_seed = fit_shap(Xi, y, return_est=True)
        grp = targets.loc[idx, "group"]
        pctl = Xi.rank(pct=True)
        for c, mat in enumerate(per_class):
            m = pd.DataFrame(mat, columns=Xi.columns, index=Xi.index)
            for g, sub in [("All", m)] + [(g, m.loc[grp == g]) for g in GROUPS]:
                if len(sub) < 5:
                    continue
                imp = sub.abs().mean().sort_values(ascending=False)
                share = 100.0 * imp / imp.sum()
                rows_idx = Xi.index.get_indexer(sub.index)
                seed_shares = []
                for sm in per_seed[c]:
                    si = np.abs(sm[rows_idx]).mean(axis=0)
                    seed_shares.append(pd.Series(100.0 * si / si.sum(), index=Xi.columns))
                share_sd = pd.concat(seed_shares, axis=1).std(axis=1)
                sgn = sub.mean()
                fam = pd.Series({f: FAMILY(f) for f in imp.index})
                fam_share = share.groupby(fam).sum().sort_values(ascending=False)
                for f in fam_share.index:
                    out_fam.append(dict(target=target, cls=c, cls_label=labels[c], group=g,
                                        family=f, share_pct=float(fam_share[f])))
                for f in imp.index[:BREAKDOWN_TOP]:
                    out_abs.append(dict(target=target, cls=c, cls_label=labels[c], group=g,
                                        n=len(sub), feature=f, family=FAMILY(f),
                                        mean_abs_shap=float(imp[f]),
                                        share_pct=float(share[f]),
                                        share_sd=float(share_sd[f])))
                    out_dir.append(dict(target=target, cls=c, cls_label=labels[c], group=g,
                                        feature=f, mean_signed_shap=float(sgn[f])))
            top6 = m.abs().mean().nlargest(6).index
            for f in top6:
                for sid in Xi.index:
                    bee.append(dict(target=target, cls=c, cls_label=labels[c], feature=f,
                                    subject=sid, group=grp.loc[sid],
                                    shap=float(m.loc[sid, f]),
                                    feature_pctl=float(pctl.loc[sid, f])))
        overall_imp = pd.Series(np.abs(np.stack(per_class)).mean(axis=(0, 1)),
                                index=Xi.columns).sort_values(ascending=False)
        for f in list(overall_imp.index[:3]):
            grid = np.quantile(Xi[f], np.linspace(0.05, 0.95, 15))
            for c in range(len(per_class)):
                for g in ("All",) + GROUPS:
                    rows_g = Xi if g == "All" else Xi.loc[grp == g]
                    if len(rows_g) < 5:
                        continue
                    base = rows_g.copy()
                    for gv in grid:
                        base[f] = gv
                        pdp.append(dict(target=target, feature=f, cls=c, cls_label=labels[c],
                                        group=g, grid=float(gv),
                                        prob=float(np.mean([e.predict_proba(base)[:, c].mean()
                                                            for e in ests]))))
        print(f"[class_group] {target} done", flush=True)
    return (pd.DataFrame(out_abs), pd.DataFrame(out_dir), pd.DataFrame(out_fam),
            pd.DataFrame(bee), pd.DataFrame(pdp))


def main() -> None:
    feats, targets, covar = load()
    ov = overall(feats, targets, covar)
    ov.to_csv(H / "ml_shap_overall_beeswarm.csv", index=False)
    abs_, dir_, fam_, bee_, pdp_ = class_group(feats, targets, covar)
    abs_.to_csv(H / "ml_shap_class_group.csv", index=False)
    dir_.to_csv(H / "ml_shap_class_group_dir.csv", index=False)
    fam_.to_csv(H / "ml_shap_family_group.csv", index=False)
    bee_.to_csv(H / "ml_shap_beeswarm_class.csv", index=False)
    pdp_.to_csv(H / "ml_pdp_class_group.csv", index=False)
    print(f"beeswarm rows {len(bee_)} · pdp rows {len(pdp_)}", flush=True)

    head = ov.drop_duplicates(["target", "feature"])[
        ["target", "feature", "family", "share_pct", "pos_pct"]]
    print("\n=== overall drivers (share of all 105 features; pos_pct = share of |SHAP| pushing toward impairment)")
    print(head.round(2).to_string(index=False), flush=True)
    print("\n=== top-3 per cell, All scope")
    top = (abs_[abs_.group == "All"].sort_values(["target", "cls", "share_pct"], ascending=[1, 1, 0])
           .groupby(["target", "cls_label"]).head(3))
    print(top[["target", "cls_label", "feature", "share_pct"]].round(2).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
