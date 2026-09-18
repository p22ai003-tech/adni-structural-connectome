#!/usr/bin/env python
"""Part B — simple, honest MMSE/CDR models from the tiered network features.

Design (user-locked): CDR binary 0 vs >=0.5; per-network FA + DIFF composite
(mean robust-z of MD/RD/AxD); models = ElasticNet(+Huber) / HistGradientBoosting /
ExtraTrees; repeated 5x5 CV; incremental ladder F0 -> +F1 -> +F2 -> +F3 -> +F4.

Hygiene: outlier 003_S_4373 excluded; composites use target-independent robust
scaling (median/IQR, documented); model-side imputation+scaling inside each fold;
nearest-visit targets with lag recorded and a <=365d sensitivity; permutation test
on the final rung.

Outputs (hcp_analysis/):
  ml_targets.csv            per-subject MMSE/CDR nearest-visit + lag + phase
  ml_feature_matrix.csv     per-subject F1..F4 features (+age/sex/phase)
  ml_ladder_results.csv     per target x model x rung: repeated-CV performance
  ml_predictions.csv        out-of-fold predictions (final rung, best model)
  ml_permutation.json       permutation p for the final rung per target
  ml_missingness.csv        feature missingness audit by group
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV, HuberRegressor, LogisticRegressionCV
from sklearn.metrics import balanced_accuracy_score, mean_absolute_error, r2_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ANALYSIS = Path("/home/ec2-user/exp/data/derivatives/qc/analysis_cohort")
OUT = Path("/home/ec2-user/exp/hcp_analysis")
FUNC = ANALYSIS / "19_network_analysis/functional"
NETWORKS = ["Visual", "Somatomotor", "DorsalAttention", "Salience_VAN", "Limbic",
            "Frontoparietal", "DMN", "Subcortical", "Cerebellar", "Brainstem"]
OUTLIER = {"003_S_4373"}
SEEDS = [11, 23, 37, 51, 73]
N_PERM = 200


def robust_z(s: pd.Series) -> pd.Series:
    med = s.median(); iqr = s.quantile(0.75) - s.quantile(0.25)
    return (s - med) / (iqr if iqr > 0 else 1.0)


def build_targets() -> pd.DataFrame:
    master = pd.read_csv(ANALYSIS / "00_master/master_cohort.csv")
    phase_col = "phase" if "phase" in master.columns else "Phase"
    base = master.set_index("subject_id")[["group", "age", "sex", phase_col]].rename(columns={phase_col: "phase"})

    scan = pd.read_csv("/home/ec2-user/exp/cohort/dti.csv", low_memory=False)
    sc = [c for c in scan.columns if "subj" in c.lower()][0]
    scan_date = (scan.assign(_d=pd.to_datetime(scan["Study Date"], errors="coerce"))
                 .dropna(subset=["_d"]).groupby(sc)["_d"].min())

    visits = []
    for f in ("/home/ec2-user/exp/cohort/mri_master.csv", "/home/ec2-user/exp/cohort/dti_master.csv"):
        d = pd.read_csv(f, low_memory=False)
        c = [x for x in d.columns if "subj" in x.lower()][0]
        d = d.rename(columns={c: "sid"})
        d["_d"] = pd.to_datetime(d.get("Study Date"), errors="coerce")
        visits.append(d[["sid", "_d", "MMSE Total Score", "Global CDR"]])
    mc = master.rename(columns={"subject_id": "sid"})
    mc["_d"] = pd.NaT
    visits.append(mc[["sid", "_d", "MMSE Total Score", "Global CDR"]])
    vis = pd.concat(visits, ignore_index=True)
    vis = vis[vis.sid.isin(base.index)]

    rows = []
    for sid, sub in vis.groupby("sid"):
        sdate = scan_date.get(sid, pd.NaT)
        rec = {"subject_id": sid}
        for col, name in (("MMSE Total Score", "mmse"), ("Global CDR", "cdr")):
            hh = sub.dropna(subset=[col])
            if hh.empty:
                continue
            if pd.notna(sdate) and hh["_d"].notna().any():
                hh = hh.assign(lag=(hh["_d"] - sdate).abs().dt.days)
                hh = hh.sort_values("lag", na_position="last")
                rec[name] = float(hh.iloc[0][col])
                rec[f"{name}_lag_days"] = float(hh.iloc[0]["lag"]) if pd.notna(hh.iloc[0]["lag"]) else np.nan
            else:
                rec[name] = float(hh[col].median())
                rec[f"{name}_lag_days"] = np.nan
        if len(rec) > 1:
            rows.append(rec)
    t = pd.DataFrame(rows).set_index("subject_id").join(base, how="inner")
    t = t[~t.index.isin(OUTLIER)]
    t["cdr_bin"] = np.where(t.cdr.notna(), (t.cdr > 0).astype(float), np.nan)
    t.to_csv(OUT / "ml_targets.csv")
    print(f"targets: n={len(t)} | mmse={t.mmse.notna().sum()} | cdr={t.cdr.notna().sum()} "
          f"(bin balance {int((t.cdr_bin == 0).sum())}/{int((t.cdr_bin == 1).sum())}) | "
          f"median lag mmse={t.mmse_lag_days.median():.0f}d", flush=True)
    return t


def composites(long: pd.DataFrame, prefix: str, measure_map: dict[str, str]) -> pd.DataFrame:
    """subject x network long frame -> FA + DIFF composite wide columns."""
    wide = long.pivot_table(index="subject_id", columns=["network", "measure"], values="value")
    out = {}
    for net in NETWORKS:
        fa_col = (net, measure_map["FA"])
        if fa_col in wide.columns:
            out[f"{prefix}_{net}_FA"] = wide[fa_col]
        zs = []
        for m in ("MD", "RD", "AxD"):
            col = (net, measure_map[m])
            if col in wide.columns:
                zs.append(robust_z(wide[col]))
        if zs:
            out[f"{prefix}_{net}_DIFF"] = pd.concat(zs, axis=1).mean(axis=1)
    return pd.DataFrame(out)


def build_features(targets: pd.DataFrame) -> pd.DataFrame:
    mic = pd.read_csv(FUNC / "network_microstructure_subject.csv")
    mic["measure"] = mic.metric.map({"fa_mean": "FA", "md_mean": "MD", "rd_mean": "RD", "ad_mean": "AxD"})
    f1 = composites(mic.dropna(subset=["measure"]), "WB", {m: m for m in ("FA", "MD", "RD", "AxD")})
    glob_rd = (mic[mic.measure == "RD"].groupby("subject_id")["value"].mean().rename("WB_global_RD"))

    rng = pd.read_csv(FUNC / "network_range_restricted_subject.csv")
    f2 = pd.concat([
        composites(rng[rng.measure.str.startswith(f"{r}-")].assign(measure=lambda d: d.measure.str[3:]),
                   r, {m: m for m in ("FA", "MD", "RD", "AxD")})
        for r in ("SR", "LR")], axis=1)

    exc = pd.read_csv(FUNC / "network_exception_measures_subject.csv")
    f3 = pd.concat([
        composites(exc[exc.measure.str.startswith(f"{r}exc-")].assign(measure=lambda d: d.measure.str.split("-").str[1]),
                   f"{r}EXC", {m: m for m in ("FA", "MD", "RD", "AxD")})
        for r in ("SR", "LR")], axis=1)

    arch = pd.read_csv(OUT / "exception_architecture_subject.csv").set_index("subject_id")
    f4 = arch.drop(columns=[c for c in ("group", "phase") if c in arch.columns]).add_prefix("ARCH_")

    feats = pd.concat([f1, glob_rd, f2, f3, f4], axis=1)
    feats = feats.loc[feats.index.isin(targets.index)]
    feats = feats[~feats.index.isin(OUTLIER)]

    miss = feats.join(targets["group"]).groupby("group").apply(lambda d: d.isna().mean(), include_groups=False)
    miss.T.to_csv(OUT / "ml_missingness.csv")
    high = (feats.isna().mean() > 0.5)
    if high.any():
        print(f"dropping {int(high.sum())} features with >50% missing: {list(feats.columns[high])[:6]}...", flush=True)
        feats = feats.loc[:, ~high]
    feats.to_csv(OUT / "ml_feature_matrix.csv")
    print(f"features: {feats.shape[0]} subjects x {feats.shape[1]} columns", flush=True)
    return feats


LADDER = ["F0", "F0+F1", "F0-F2", "F0-F3", "F0-F4"]


def rung_columns(feats: pd.DataFrame, rung: str) -> list[str]:
    f1 = [c for c in feats.columns if c.startswith("WB_")]
    f2 = [c for c in feats.columns if c.startswith(("SR_", "LR_"))]
    f3 = [c for c in feats.columns if c.startswith(("SREXC_", "LREXC_"))]
    f4 = [c for c in feats.columns if c.startswith("ARCH_")]
    return {"F0": [], "F0+F1": f1, "F0-F2": f1 + f2, "F0-F3": f1 + f2 + f3, "F0-F4": f1 + f2 + f3 + f4}[rung]


def make_models(kind: str, seed: int):
    if kind == "reg":
        return {
            "elasticnet_huber": Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
                                          ("m", HuberRegressor(alpha=1.0, max_iter=2000))]),
            "hist_gb": HistGradientBoostingRegressor(max_depth=3, max_iter=250, learning_rate=0.05,
                                                     l2_regularization=1.0, random_state=seed),
            "extra_trees": Pipeline([("imp", SimpleImputer(strategy="median")),
                                     ("m", ExtraTreesRegressor(n_estimators=400, min_samples_leaf=5,
                                                               random_state=seed, n_jobs=8))]),
        }
    return {
        "logreg": Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
                            ("m", LogisticRegressionCV(Cs=5, class_weight="balanced", max_iter=4000,
                                                       random_state=seed))]),
        "hist_gb": HistGradientBoostingClassifier(max_depth=3, max_iter=250, learning_rate=0.05,
                                                  l2_regularization=1.0, class_weight="balanced",
                                                  random_state=seed),
        "extra_trees": Pipeline([("imp", SimpleImputer(strategy="median")),
                                 ("m", ExtraTreesClassifier(n_estimators=400, min_samples_leaf=5,
                                                            class_weight="balanced", random_state=seed, n_jobs=8))]),
    }


def evaluate(feats: pd.DataFrame, targets: pd.DataFrame) -> None:
    results, preds_out = [], []
    perm = {}
    covar = pd.DataFrame({
        "age": targets.age, "sex": (targets.sex.astype(str).str.upper().str[0] == "F").astype(float)
    }, index=targets.index)

    for target, kind in (("mmse", "reg"), ("cdr_bin", "clf")):
        y_all = targets[target].dropna()
        idx = y_all.index.intersection(feats.index)
        y = y_all.loc[idx]
        strat = (targets.loc[idx, "group"].astype(str) + "|" + (y > y.median()).astype(str)) if kind == "reg" \
            else y.astype(int).astype(str)
        print(f"\n=== target {target}: n={len(y)} ===", flush=True)
        for rung in LADDER:
            X = pd.concat([covar.loc[idx], feats.loc[idx, rung_columns(feats, rung)]], axis=1)
            for mname in make_models(kind, 0):
                scores = []
                oof = np.full(len(y), np.nan)
                for seed in SEEDS:
                    model_set = make_models(kind, seed)
                    est = model_set[mname]
                    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
                    for tr, te in cv.split(X, strat):
                        est.fit(X.iloc[tr], y.iloc[tr])
                        if kind == "reg":
                            p = est.predict(X.iloc[te])
                            scores.append(dict(spearman=sps.spearmanr(y.iloc[te], p).statistic,
                                               r2=r2_score(y.iloc[te], p),
                                               mae=mean_absolute_error(y.iloc[te], p)))
                        else:
                            p = est.predict_proba(X.iloc[te])[:, 1] if hasattr(est, "predict_proba") else est.decision_function(X.iloc[te])
                            lab = est.predict(X.iloc[te])
                            scores.append(dict(auc=roc_auc_score(y.iloc[te], p),
                                               bal_acc=balanced_accuracy_score(y.iloc[te], lab)))
                        if seed == SEEDS[0]:
                            oof[te] = p
                sc = pd.DataFrame(scores)
                rec = dict(target=target, rung=rung, model=mname, n=len(y),
                           n_features=X.shape[1],
                           **{f"{k}_mean": sc[k].mean() for k in sc.columns},
                           **{f"{k}_sd": sc[k].std() for k in sc.columns})
                results.append(rec)
                key = "spearman_mean" if kind == "reg" else "auc_mean"
                print(f"  {rung:6} {mname:16} " + " ".join(f"{k}={v:.3f}" for k, v in rec.items()
                      if k.endswith("_mean")), flush=True)
                if rung == LADDER[-1]:
                    preds_out.append(pd.DataFrame({"subject_id": idx, "target": target, "model": mname,
                                                   "y_true": y.values, "y_oof": oof}))
        # permutation on final rung, best model by primary metric
        res = pd.DataFrame(results)
        fin = res[(res.target == target) & (res.rung == LADDER[-1])]
        key = "spearman_mean" if kind == "reg" else "auc_mean"
        best = fin.sort_values(key, ascending=False).iloc[0]
        Xf = pd.concat([covar.loc[idx], feats.loc[idx, rung_columns(feats, LADDER[-1])]], axis=1)
        rng_ = np.random.default_rng(7)
        null = []
        for _ in range(N_PERM):
            yp = pd.Series(rng_.permutation(y.values), index=y.index)
            est = make_models(kind, 3)[best.model]
            cv = StratifiedKFold(5, shuffle=True, random_state=3)
            vals = []
            for tr, te in cv.split(Xf, strat):
                est.fit(Xf.iloc[tr], yp.iloc[tr])
                if kind == "reg":
                    vals.append(sps.spearmanr(yp.iloc[te], est.predict(Xf.iloc[te])).statistic)
                else:
                    pp = est.predict_proba(Xf.iloc[te])[:, 1]
                    vals.append(roc_auc_score(yp.iloc[te], pp))
            null.append(float(np.mean(vals)))
        obs = float(best[key])
        pval = (1 + sum(1 for v in null if v >= obs)) / (1 + N_PERM)
        perm[target] = dict(model=str(best.model), metric=key, observed=obs, perm_p=pval,
                            null_mean=float(np.mean(null)), n_perm=N_PERM)
        print(f"  permutation ({best.model}): obs {obs:.3f} vs null {np.mean(null):.3f} -> p={pval:.4f}", flush=True)

    pd.DataFrame(results).to_csv(OUT / "ml_ladder_results.csv", index=False)
    pd.concat(preds_out, ignore_index=True).to_csv(OUT / "ml_predictions.csv", index=False)
    json.dump(perm, open(OUT / "ml_permutation.json", "w"), indent=1)
    print("\nwrote ml_ladder_results.csv / ml_predictions.csv / ml_permutation.json", flush=True)


if __name__ == "__main__":
    t = build_targets()
    f = build_features(t)
    evaluate(f, t)
