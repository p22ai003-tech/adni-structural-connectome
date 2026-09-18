"""Lever 3 - ensembling: blend {ExtraTrees, HistGB, ElasticNet/LogReg} OOF predictions
(simple mean; rank-average for MMSE) and sklearn Stacking with Ridge/LogReg final.
Fixed protocol: age+sex covariates prepended, repeated stratified 5-fold CV,
seeds [11,23,37,51,73], all fitted preprocessing inside training folds only.
"""
import time
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (ExtraTreesRegressor, ExtraTreesClassifier,
                              HistGradientBoostingRegressor, HistGradientBoostingClassifier,
                              StackingRegressor, StackingClassifier)
from sklearn.linear_model import ElasticNetCV, LogisticRegression, Ridge
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import r2_score, roc_auc_score

SEEDS = [11, 23, 37, 51, 73]
NJ = 8
t0 = time.time()

X = pd.read_csv('/home/ec2-user/exp/hcp_analysis/ml_feature_matrix.csv', index_col=0)
Y = pd.read_csv('/home/ec2-user/exp/hcp_analysis/ml_targets.csv', index_col=0)
assert X.index.equals(Y.index)

# Covariates age + sex(F=1) prepended
cov = pd.DataFrame({'age': Y['age'].values,
                    'sex': (Y['sex'] == 'F').astype(float).values}, index=X.index)
Xf = pd.concat([cov, X], axis=1).values.astype(float)

mmse = Y['mmse'].values.astype(float)
cdr = Y['cdr_bin'].values.astype(int)
strat_mmse = (Y['group'] + '_' + (Y['mmse'] > Y['mmse'].median()).astype(str)).values
strat_cdr = cdr

# ---------------- model builders (fresh per fold; all preprocessing in-pipeline) --------
def et_reg(seed):
    return Pipeline([('imp', SimpleImputer(strategy='median')),
                     ('m', ExtraTreesRegressor(n_estimators=400, min_samples_leaf=5,
                                               random_state=seed, n_jobs=NJ))])

def et_clf(seed):
    return Pipeline([('imp', SimpleImputer(strategy='median')),
                     ('m', ExtraTreesClassifier(n_estimators=400, min_samples_leaf=5,
                                                random_state=seed, n_jobs=NJ))])

def hgb_reg(seed):
    return HistGradientBoostingRegressor(max_iter=250, learning_rate=0.05,
                                         max_leaf_nodes=15, min_samples_leaf=10,
                                         l2_regularization=1.0, early_stopping=False,
                                         random_state=seed)

def hgb_clf(seed):
    return HistGradientBoostingClassifier(max_iter=250, learning_rate=0.05,
                                          max_leaf_nodes=15, min_samples_leaf=10,
                                          l2_regularization=1.0, early_stopping=False,
                                          random_state=seed)

def enet_reg(seed):
    return Pipeline([('imp', SimpleImputer(strategy='median')),
                     ('sc', StandardScaler()),
                     ('m', ElasticNetCV(l1_ratio=[0.2, 0.5, 0.9], n_alphas=30,
                                        cv=5, max_iter=5000, random_state=seed, n_jobs=NJ))])

def logreg_clf(seed):
    return Pipeline([('imp', SimpleImputer(strategy='median')),
                     ('sc', StandardScaler()),
                     ('m', LogisticRegression(C=1.0, penalty='l2', max_iter=5000,
                                              random_state=seed))])

# ---------------- MMSE ----------------
mmse_res = {k: {'rho': [], 'r2': []} for k in
            ['baseline_ET', 'HistGB', 'ElasticNet', 'mean_blend', 'rank_blend', 'stacking']}

for seed in SEEDS:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    n = len(mmse)
    oof = {k: np.full(n, np.nan) for k in ['baseline_ET', 'HistGB', 'ElasticNet', 'stacking']}
    for tr, te in skf.split(Xf, strat_mmse):
        for name, builder in [('baseline_ET', et_reg), ('HistGB', hgb_reg), ('ElasticNet', enet_reg)]:
            mdl = builder(seed)
            mdl.fit(Xf[tr], mmse[tr])
            oof[name][te] = mdl.predict(Xf[te])
        stack = StackingRegressor(
            estimators=[('et', et_reg(seed)), ('hgb', hgb_reg(seed)), ('enet', enet_reg(seed))],
            final_estimator=Ridge(alpha=1.0),
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=seed + 1)
                .split(Xf[tr], strat_mmse[tr]),
            n_jobs=1)
        stack.fit(Xf[tr], mmse[tr])
        oof['stacking'][te] = stack.predict(Xf[te])
    oof['mean_blend'] = (oof['baseline_ET'] + oof['HistGB'] + oof['ElasticNet']) / 3.0
    oof['rank_blend'] = (rankdata(oof['baseline_ET']) + rankdata(oof['HistGB'])
                         + rankdata(oof['ElasticNet'])) / 3.0
    for k, p in oof.items():
        assert not np.isnan(p).any()
        mmse_res[k]['rho'].append(spearmanr(mmse, p).statistic)
        if k != 'rank_blend':  # ranks are not in MMSE units; R2 undefined
            mmse_res[k]['r2'].append(r2_score(mmse, p))
    print(f'[MMSE seed {seed}] done  {time.time()-t0:.0f}s', flush=True)

# ---------------- CDR ----------------
cdr_res = {k: [] for k in ['baseline_ET', 'HistGB', 'LogReg', 'mean_blend', 'stacking']}

for seed in SEEDS:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    n = len(cdr)
    oof = {k: np.full(n, np.nan) for k in ['baseline_ET', 'HistGB', 'LogReg', 'stacking']}
    for tr, te in skf.split(Xf, strat_cdr):
        for name, builder in [('baseline_ET', et_clf), ('HistGB', hgb_clf), ('LogReg', logreg_clf)]:
            mdl = builder(seed)
            mdl.fit(Xf[tr], cdr[tr])
            oof[name][te] = mdl.predict_proba(Xf[te])[:, 1]
        stack = StackingClassifier(
            estimators=[('et', et_clf(seed)), ('hgb', hgb_clf(seed)), ('lr', logreg_clf(seed))],
            final_estimator=LogisticRegression(max_iter=5000, random_state=seed),
            stack_method='predict_proba',
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=seed + 1),
            n_jobs=1)
        stack.fit(Xf[tr], cdr[tr])
        oof['stacking'][te] = stack.predict_proba(Xf[te])[:, 1]
    oof['mean_blend'] = (oof['baseline_ET'] + oof['HistGB'] + oof['LogReg']) / 3.0
    for k, p in oof.items():
        assert not np.isnan(p).any()
        cdr_res[k].append(roc_auc_score(cdr, p))
    print(f'[CDR  seed {seed}] done  {time.time()-t0:.0f}s', flush=True)

# ---------------- report ----------------
print('\n================ LEVER 3: ENSEMBLING — RESULTS ================')
print('MMSE (pooled OOF per seed, mean +/- sd over seeds):')
for k in ['baseline_ET', 'HistGB', 'ElasticNet', 'mean_blend', 'rank_blend', 'stacking']:
    rho = np.array(mmse_res[k]['rho'])
    if mmse_res[k]['r2']:
        r2 = np.array(mmse_res[k]['r2'])
        print(f'  {k:12s}  Spearman={rho.mean():.4f} (+/-{rho.std():.4f})   '
              f'R2={r2.mean():.4f} (+/-{r2.std():.4f})')
    else:
        print(f'  {k:12s}  Spearman={rho.mean():.4f} (+/-{rho.std():.4f})   R2=n/a (rank units)')
print('CDR (pooled OOF AUC, mean +/- sd over seeds):')
for k in ['baseline_ET', 'HistGB', 'LogReg', 'mean_blend', 'stacking']:
    a = np.array(cdr_res[k])
    print(f'  {k:12s}  AUC={a.mean():.4f} (+/-{a.std():.4f})')
print(f'\nTotal runtime: {time.time()-t0:.0f}s')
