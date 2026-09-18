"""Lever 4 - representation changes.
(a) in-fold QuantileTransformer(normal) -> ExtraTrees AND Ridge/LogReg
(b) in-fold impute+scale+PCA(20) on features (covariates kept raw, prepended) -> ExtraTrees
(c) in-fold age+sex residualisation of every feature -> ExtraTrees
Baseline re-run in this script: ExtraTrees n=400 min_samples_leaf=5, median imputation.
Protocol: covariates age+sex(F=1) prepended; repeated stratified 5-fold CV seeds [11,23,37,51,73];
all fitted preprocessing inside training fold; pooled OOF metrics per seed, averaged over seeds.
"""
import numpy as np, pandas as pd, time
from scipy.stats import spearmanr
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier
from sklearn.linear_model import RidgeCV, LogisticRegressionCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, roc_auc_score

SEEDS = [11, 23, 37, 51, 73]
NJOBS = 8

X = pd.read_csv('/home/ec2-user/exp/hcp_analysis/ml_feature_matrix.csv', index_col=0)
T = pd.read_csv('/home/ec2-user/exp/hcp_analysis/ml_targets.csv', index_col=0)
assert X.index.equals(T.index)

age = T['age'].values.astype(float)
sex = (T['sex'] == 'F').astype(float).values
cov = np.column_stack([age, sex])                       # covariates, prepended
F = X.values.astype(float)                              # 103 features (with NaNs)
Xfull = np.column_stack([cov, F])                       # cov prepended, as baseline
mmse = T['mmse'].values.astype(float)
cdr = T['cdr_bin'].values.astype(int)

strat_mmse = (T['group'].astype(str) + '_' + (mmse > np.median(mmse)).astype(int).astype(str)).values
strat_cdr = cdr

def et_reg(seed):  return ExtraTreesRegressor(n_estimators=400, min_samples_leaf=5, random_state=seed, n_jobs=NJOBS)
def et_clf(seed):  return ExtraTreesClassifier(n_estimators=400, min_samples_leaf=5, random_state=seed, n_jobs=NJOBS)

# ---- representation builders: fit on train, apply to train+test. Input = full prepended matrix.
def rep_baseline(Xtr, Xte, *_):
    imp = SimpleImputer(strategy='median').fit(Xtr)
    return imp.transform(Xtr), imp.transform(Xte)

def rep_qt(Xtr, Xte, *_):
    imp = SimpleImputer(strategy='median').fit(Xtr)
    Xtr, Xte = imp.transform(Xtr), imp.transform(Xte)
    qt = QuantileTransformer(output_distribution='normal',
                             n_quantiles=min(100, Xtr.shape[0]), random_state=0).fit(Xtr)
    return qt.transform(Xtr), qt.transform(Xte)

def rep_pca(Xtr, Xte, *_):
    # PCA(20) on the 103 features; covariates kept raw and prepended (as in baseline)
    ctr, cte = Xtr[:, :2], Xte[:, :2]
    ftr, fte = Xtr[:, 2:], Xte[:, 2:]
    imp = SimpleImputer(strategy='median').fit(ftr)
    ftr, fte = imp.transform(ftr), imp.transform(fte)
    sc = StandardScaler().fit(ftr)
    ftr, fte = sc.transform(ftr), sc.transform(fte)
    p = PCA(n_components=20, random_state=0).fit(ftr)
    return np.column_stack([ctr, p.transform(ftr)]), np.column_stack([cte, p.transform(fte)])

def rep_resid(Xtr, Xte, *_):
    # residualise every feature on age+sex (linear model fit on train); covariates stay prepended
    ctr, cte = Xtr[:, :2], Xte[:, :2]
    ftr, fte = Xtr[:, 2:], Xte[:, 2:]
    imp = SimpleImputer(strategy='median').fit(ftr)
    ftr, fte = imp.transform(ftr), imp.transform(fte)
    Dtr = np.column_stack([np.ones(len(ctr)), ctr])     # intercept + age + sex
    Dte = np.column_stack([np.ones(len(cte)), cte])
    beta, *_ = np.linalg.lstsq(Dtr, ftr, rcond=None)
    return np.column_stack([ctr, ftr - Dtr @ beta]), np.column_stack([cte, fte - Dte @ beta])

def ridge():   return RidgeCV(alphas=np.logspace(-2, 4, 13))
def logreg():  return LogisticRegressionCV(Cs=np.logspace(-3, 2, 11), cv=5, scoring='roc_auc',
                                           max_iter=5000, n_jobs=NJOBS)

CONFIGS = [   # name, representation, mmse model factory(seed), cdr model factory(seed)
    ('baseline_ET',      rep_baseline, et_reg,            et_clf),
    ('a_QT_ET',          rep_qt,       et_reg,            et_clf),
    ('a_QT_linear',      rep_qt,       lambda s: ridge(), lambda s: logreg()),
    ('b_PCA20_ET',       rep_pca,      et_reg,            et_clf),
    ('c_resid_ET',       rep_resid,    et_reg,            et_clf),
]

results = {}
for name, rep, mk_reg, mk_clf in CONFIGS:
    t0 = time.time()
    rhos, r2s, aucs = [], [], []
    for seed in SEEDS:
        # ---- MMSE
        oof = np.full(len(mmse), np.nan)
        skf = StratifiedKFold(5, shuffle=True, random_state=seed)
        for tr, te in skf.split(Xfull, strat_mmse):
            Xtr, Xte = rep(Xfull[tr], Xfull[te])
            m = mk_reg(seed).fit(Xtr, mmse[tr])
            oof[te] = m.predict(Xte)
        rhos.append(spearmanr(mmse, oof).statistic)
        r2s.append(r2_score(mmse, oof))
        # ---- CDR
        oof = np.full(len(cdr), np.nan)
        skf = StratifiedKFold(5, shuffle=True, random_state=seed)
        for tr, te in skf.split(Xfull, strat_cdr):
            Xtr, Xte = rep(Xfull[tr], Xfull[te])
            c = mk_clf(seed).fit(Xtr, cdr[tr])
            oof[te] = c.predict_proba(Xte)[:, 1]
        aucs.append(roc_auc_score(cdr, oof))
    results[name] = (np.mean(rhos), np.std(rhos), np.mean(r2s), np.std(r2s),
                     np.mean(aucs), np.std(aucs))
    print(f"{name:14s}  MMSE rho={np.mean(rhos):.4f}+-{np.std(rhos):.4f}  "
          f"R2={np.mean(r2s):.4f}+-{np.std(r2s):.4f}  "
          f"CDR AUC={np.mean(aucs):.4f}+-{np.std(aucs):.4f}   [{time.time()-t0:.0f}s]", flush=True)

print("\n=== SUMMARY (mean over 5 seeds of pooled-OOF metrics) ===")
print(f"{'config':14s} {'MMSE_rho':>9s} {'MMSE_R2':>9s} {'CDR_AUC':>9s}")
for name, (r, _, q, _, a, _) in results.items():
    print(f"{name:14s} {r:9.4f} {q:9.4f} {a:9.4f}")
