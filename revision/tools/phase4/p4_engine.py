"""Phase 4 step 2 — nested incremental prediction with paired resampling.

Usage: python p4_engine.py [P=500] [REPEATS=20]

Design (fixed before results were examined):
  targets     continuous: CDR-SB, ADAS-Cog 13, PHC memory, PHC executive function (primary); MMSE, FAQ (secondary);
              binary: global CDR >= 0.5 (AUC). Scores within 180 days of the diffusion scan.
  samples     primary = ADNI-3/4 after quality control; sensitivity = pooled after quality control.
  blocks      C = age, sex, education; Dx = diagnosis (MCI, AD indicators, CN reference);
              F1 whole brain, F2 tract length, F3 exceptions, F4 architecture (p4_data.py).
  models      connectome ladder  C | C+F1 | C+F1-2 | C+F1-3 | C+F1-4      (the manuscript's ladder)
              beyond diagnosis   C+Dx | C+Dx+F1 | ... | C+Dx+F1-4          (the key comparison: last against first)
              single families    C+Dx+Fk for k = 1..4                       (order-independent contribution)
              within diagnosis   C | C+F1-4 inside CN, MCI and AD separately
  learner     ridge regression (ridge classifier for the binary target) with the penalty chosen by internal
              leave-one-out cross-validation, median imputation and standardisation inside each training fold.
  resampling  5-fold cross-validation stratified by diagnosis, repeated 20 times with the same folds for every model,
              so model comparisons are paired. Out-of-sample R^2 (AUC) per repeat; mean and 2.5-97.5% range over repeats.
              Difference between two models: mean paired difference over repeats, 95% interval from 2,000 bootstrap
              resamples of participants of the repeat-averaged out-of-fold predictions.
  inference   the paired bootstrap interval of the gain is the main criterion. In addition, a Freedman-Lane permutation
              test for each 'adds the connectome' comparison: every connectome feature is split into the part explained by
              the smaller model's predictors (covariates, and diagnosis when present) and a residual; the residuals are
              shuffled across participants (within diagnosis groups when diagnosis is in the model) P times and added back,
              so the shuffled features keep their relation to the covariates and their overfitting cost but lose any
              further relation to the score. Each shuffle is evaluated with 5 repeats of the same cross-validation and
              compared with the observed gain from the same 5 repeats. (A plain shuffle of the features is too lenient:
              uninformative features lower out-of-sample R^2 by about 0.05, so almost any real gain beats them.)
  association partial Spearman correlations within each diagnosis group between each score and seven pre-specified
              connectome measures (limbic AxD and MD, limbic long-range MD, limbic short-range AxD, DMN short- and long-range
              MD, core coverage), adjusted for age, sex and education; BH within each group.
Outputs (IIT_P4): results/models.csv, results/comparisons.csv, results/association.csv, results/oof_*.npz, notes.json
"""
import sys as _sys; from pathlib import Path as _Path; _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import iit_chain as chain; chain.boot()      # IIT_ROOT and the chain settings; no effect when IIT_ROOT is unset
import os
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import json, sys, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV, RidgeClassifierCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
O = Path(os.environ.get("IIT_P4", "/home/ec2-user/exp/docs/iit_dipanjan/phase4_uniform")); R = O / "results"; R.mkdir(exist_ok=True)
P1 = Path(os.environ.get("IIT_P1", "/home/ec2-user/exp/docs/iit_dipanjan/phase1_uniform"))
NPERM = int(sys.argv[1]) if len(sys.argv) > 1 else 500
REPEATS = int(sys.argv[2]) if len(sys.argv) > 2 else 20
PERM_REPEATS, BOOT, K = 5, 2000, 5
ALPHAS = np.logspace(-3, 5, 33)
D = pd.read_csv(O / "p4_data.csv", index_col=0)
fam = pd.read_csv(O / "feature_families.csv", index_col=0).family
FAM = {k: list(fam.index[fam == k]) for k in ["F1", "F2", "F3", "F4"]}
D["female"] = D.female.astype(float)
D["dx_mci"], D["dx_ad"] = (D.group == "MCI").astype(float), (D.group == "AD").astype(float)
C, DX = ["age", "female", "education"], ["dx_mci", "dx_ad"]
TARGETS = {"cdrsb": "reg", "adas13": "reg", "phc_mem": "reg", "phc_exf": "reg", "mmse": "reg", "faq": "reg", "cdr_impaired": "clf"}
FF = lambda ks: [c for k in ks for c in FAM[k]]
MODELS = {"C": C, "C+F1": C + FF(["F1"]), "C+F1-2": C + FF(["F1", "F2"]), "C+F1-3": C + FF(["F1", "F2", "F3"]),
          "C+F1-4": C + FF(["F1", "F2", "F3", "F4"]),
          "C+Dx": C + DX, "C+Dx+F1": C + DX + FF(["F1"]), "C+Dx+F1-2": C + DX + FF(["F1", "F2"]),
          "C+Dx+F1-3": C + DX + FF(["F1", "F2", "F3"]), "C+Dx+F1-4": C + DX + FF(["F1", "F2", "F3", "F4"]),
          "C+Dx+F2": C + DX + FF(["F2"]), "C+Dx+F3": C + DX + FF(["F3"]), "C+Dx+F4": C + DX + FF(["F4"])}
COMPARE = [("C+F1", "C"), ("C+F1-2", "C+F1"), ("C+F1-3", "C+F1-2"), ("C+F1-4", "C+F1-3"), ("C+F1-4", "C"),
           ("C+Dx", "C"), ("C+Dx+F1", "C+Dx"), ("C+Dx+F1-2", "C+Dx+F1"), ("C+Dx+F1-3", "C+Dx+F1-2"), ("C+Dx+F1-4", "C+Dx+F1-3"),
           ("C+Dx+F1-4", "C+Dx"), ("C+Dx+F2", "C+Dx"), ("C+Dx+F3", "C+Dx"), ("C+Dx+F4", "C+Dx"), ("C+Dx+F1-4", "C+F1-4")]
PERMUTE = {("C+F1-4", "C"): False, ("C+Dx+F1-4", "C+Dx"): True}       # value: shuffle within diagnosis groups


def learner(kind):
    est = RidgeCV(alphas=ALPHAS, gcv_mode="svd") if kind == "reg" else RidgeClassifierCV(alphas=ALPHAS, class_weight="balanced")
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), est)


def score(kind, y, p):
    if kind == "clf": return roc_auc_score(y, p)
    return 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)


def folds(strata, repeats, seed0=0):
    return [list(StratifiedKFold(K, shuffle=True, random_state=seed0 + r).split(np.zeros(len(strata)), strata)) for r in range(repeats)]


def oof(X, y, kind, fl):
    """out-of-fold predictions, one row per repeat"""
    P = np.zeros((len(fl), len(y)))
    for r, splits in enumerate(fl):
        for tr, te in splits:
            m = learner(kind).fit(X[tr], y[tr])
            P[r, te] = m.decision_function(X[te]) if kind == "clf" else m.predict(X[te])
    return P


def boot_diff(kind, y, pa, pb, strata, rng):
    """paired bootstrap of participants on repeat-averaged out-of-fold predictions"""
    out = np.empty(BOOT); groups = [np.flatnonzero(strata == g) for g in np.unique(strata)]
    for b in range(BOOT):
        i = np.concatenate([rng.choice(g, len(g)) for g in groups])
        if kind == "clf" and len(np.unique(y[i])) < 2: out[b] = np.nan; continue
        out[b] = score(kind, y[i], pa[i]) - score(kind, y[i], pb[i])
    return np.nanpercentile(out, [2.5, 97.5])


def run(sample, target, group=None):
    kind = TARGETS[target]; rng = np.random.default_rng(7)
    d = D[(chain.primary(D) == 1) if sample == "primary" else D.index.notnull()]
    if group: d = d[d.group == group]
    d = d[d[target].notna()]
    y = d[target].values.astype(float)
    strata = (d.group.values if group is None else (y > np.median(y)).astype(int).astype(str)) if kind == "reg" else \
             (d.group.values + "|" + y.astype(int).astype(str) if group is None else y.astype(int).astype(str))
    cls, cnt = np.unique(strata, return_counts=True)
    if group and (len(d) < 30 or len(cls) < 2 or cnt.min() < K): return [], [], {}
    names = ["C", "C+F1-4"] if group else list(MODELS)
    fl = folds(strata, REPEATS); preds, rows = {}, []
    for nm in names:
        P = oof(d[MODELS[nm]].values.astype(float), y, kind, fl); preds[nm] = P
        s = np.array([score(kind, y, P[r]) for r in range(REPEATS)])
        rows.append(dict(sample=sample, target=target, group=group or "all", model=nm, n=len(y), n_features=len(MODELS[nm]),
                         metric="AUC" if kind == "clf" else "R2", mean=s.mean(), lo=np.percentile(s, 2.5), hi=np.percentile(s, 97.5)))
    comps = [("C+F1-4", "C")] if group else COMPARE
    crow = []
    for a, b in comps:
        diff = np.array([score(kind, y, preds[a][r]) - score(kind, y, preds[b][r]) for r in range(REPEATS)])
        lo, hi = boot_diff(kind, y, preds[a].mean(0), preds[b].mean(0), strata, rng)
        rec = dict(sample=sample, target=target, group=group or "all", model=a, versus=b, n=len(y), diff=diff.mean(),
                   diff_lo=lo, diff_hi=hi, share_repeats_positive=float((diff > 0).mean()), perm_p=np.nan)
        if (a, b) in PERMUTE or group:
            within = PERMUTE.get((a, b), False); fblock = [c for c in MODELS[a] if c not in MODELS[b]]
            flp = fl[:PERM_REPEATS]; Xb = d[MODELS[b]].values.astype(float); Fb = d[fblock].values.astype(float)
            Fi = np.where(np.isfinite(Fb), Fb, np.nanmedian(Fb, 0)); Xi = np.where(np.isfinite(Xb), Xb, np.nanmedian(Xb, 0))
            Z = np.c_[np.ones(len(y)), Xi]; fitted = Z @ np.linalg.lstsq(Z, Fi, rcond=None)[0]; resid = Fi - fitted
            obs = np.mean([score(kind, y, p) for p in oof(np.c_[Xb, Fi], y, kind, flp)]) - np.mean([score(kind, y, p) for p in preds[b][:PERM_REPEATS]])
            base = np.mean([score(kind, y, p) for p in preds[b][:PERM_REPEATS]]); null = np.empty(NPERM)
            gidx = [np.flatnonzero(d.group.values == g) for g in np.unique(d.group.values)] if within else [np.arange(len(y))]
            for k in range(NPERM):
                perm = np.arange(len(y))
                for g in gidx: perm[g] = rng.permutation(g)
                null[k] = np.mean([score(kind, y, p) for p in oof(np.c_[Xb, fitted + resid[perm]], y, kind, flp)]) - base
            rec.update(perm_p=(1 + np.sum(null >= obs)) / (1 + NPERM), perm_obs=obs, perm_null_mean=null.mean(),
                       perm_null_hi=np.percentile(null, 95))
        crow.append(rec)
    keep = {nm: preds[nm].mean(0) for nm in names}
    return rows, crow, {f"{sample}|{target}|{group or 'all'}|{nm}": v for nm, v in keep.items()} | {f"{sample}|{target}|{group or 'all'}|y": y,
                                                                                                    f"{sample}|{target}|{group or 'all'}|ids": d.index.values}


t0 = time.time()
tasks = [(s, t, g) for s in ("primary", "pooled") for t in TARGETS for g in (None, "CN", "MCI", "AD")]
res = Parallel(n_jobs=min(len(tasks), 60), verbose=0, batch_size=1)(delayed(run)(*x) for x in sorted(tasks, key=lambda x: x[2] is not None))
models = pd.DataFrame([r for x in res for r in x[0]]); comps = pd.DataFrame([r for x in res for r in x[1]])
models.to_csv(R / "models.csv", index=False); comps.to_csv(R / "comparisons.csv", index=False)
np.savez_compressed(R / "oof_predictions.npz", **{k: v for x in res for k, v in x[2].items()})

# ---------------- within-diagnosis associations
FQ = pd.read_csv(P1 / "features_qc.csv", index_col=0)
ASSOC = {"Limbic AxD": "WB|Limbic|AxD", "Limbic MD": "WB|Limbic|MD", "Limbic long-range MD": "TR|Limbic|LR-MD",
         "Limbic short-range AxD": "TR|Limbic|SR-AxD", "DMN short-range MD": "TR|DMN|SR-MD", "DMN long-range MD": "TR|DMN|LR-MD",
         "Core coverage": "COV|core|coverage"}
rows = []
for sample in ("primary", "pooled"):
    d0 = D[(chain.primary(D) == 1) if sample == "primary" else D.index.notnull()].join(FQ[list(ASSOC.values())])
    for g in ("CN", "MCI", "AD"):
        dg = d0[d0.group == g]
        for t in [t for t in TARGETS if TARGETS[t] == "reg"]:
            for lab, col in ASSOC.items():
                x = dg[[t, col] + C].dropna()
                if len(x) < 15: continue
                rk = x.rank(); Z = np.c_[np.ones(len(x)), rk[C].values]
                res_ = lambda v: v - Z @ np.linalg.lstsq(Z, v, rcond=None)[0]
                r_, p_ = stats.pearsonr(res_(rk[t].values), res_(rk[col].values))
                rows.append(dict(sample=sample, group=g, target=t, feature=lab, n=len(x), rho=r_, p=p_))
A = pd.DataFrame(rows, columns=["sample", "group", "target", "feature", "n", "rho", "p"])      # may be empty in a small sample
A["q"] = np.nan
for (s, g), idx in A.groupby(["sample", "group"]).groups.items():
    p = A.loc[idx, "p"].values; o = np.argsort(p); q = np.empty(len(p)); run_ = 1.0
    for k, i in enumerate(o[::-1]): run_ = min(run_, p[i] * len(p) / (len(p) - k)); q[i] = run_
    A.loc[idx, "q"] = q
A.to_csv(R / "association.csv", index=False)
n = {s: {t: {g: int(((chain.primary(D) == 1) if s == "primary" else D.index.notnull())[D[t].notna() & (D.group == g)].sum()) for g in ("CN", "MCI", "AD")}
         for t in TARGETS} for s in ("primary", "pooled")}
json.dump(dict(n=n, repeats=REPEATS, perms=NPERM, perm_repeats=PERM_REPEATS, boot=BOOT, folds=K,
               lag_median={t: float(D.loc[D[t].notna(), f"{t if t != 'cdr_impaired' else 'cdr_global'}_lag"].median()) for t in TARGETS},
               seconds=round(time.time() - t0)), open(O / "notes.json", "w"), indent=1)
print(f"done in {time.time() - t0:.0f} s")
key = comps[(comps.group == "all") & (comps.model == "C+Dx+F1-4") & (comps.versus == "C+Dx")]
print(key[["sample", "target", "n", "diff", "diff_lo", "diff_hi", "perm_p"]].round(4).to_string(index=False))
