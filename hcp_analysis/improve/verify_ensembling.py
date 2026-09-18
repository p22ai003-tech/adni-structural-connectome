"""Adversarial verification of Lever 3 (ensembling) report.
Fresh from-scratch re-run: baseline ExtraTrees vs ensemble variants
(mean blend, rank blend, weighted 0.5/0.7 ET blends, stacking) on MMSE + CDR.
Fixed protocol: age+sex(F=1) prepended; repeated stratified 5-fold CV,
seeds [11,23,37,51,73]; MMSE strat = group + (mmse>median), CDR strat = cdr_bin;
all fitted preprocessing inside training folds; pooled OOF metrics per seed,
averaged over seeds.
"""
import time, warnings
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
warnings.filterwarnings('ignore')

SEEDS = [11, 23, 37, 51, 73]
NJ = 8
t0 = time.time()

X = pd.read_csv('/home/ec2-user/exp/hcp_analysis/ml_feature_matrix.csv', index_col=0)
Y = pd.read_csv('/home/ec2-user/exp/hcp_analysis/ml_targets.csv', index_col=0)
assert X.index.equals(Y.index)

cov = pd.DataFrame({'age': Y['age'].values,
                    'sex': (Y['sex'] == 'F').astype(float).values}, index=X.index)
Xf = pd.concat([cov, X], axis=1).values.astype(float)
mmse = Y['mmse'].values.astype(float)
cdr = Y['cdr_bin'].values.astype(int)
strat_mmse = (Y['group'] + '_' + (Y['mmse'] > Y['mmse'].median()).astype(str)).values
strat_cdr = cdr

def et_reg(s):
    return Pipeline([('i', SimpleImputer(strategy='median')),
                     ('m', ExtraTreesRegressor(n_estimators=400, min_samples_leaf=5,
                                               random_state=s, n_jobs=NJ))])
def et_clf(s):
    return Pipeline([('i', SimpleImputer(strategy='median')),
                     ('m', ExtraTreesClassifier(n_estimators=400, min_samples_leaf=5,
                                                random_state=s, n_jobs=NJ))])
def hgb_reg(s):
    return HistGradientBoostingRegressor(max_iter=250, learning_rate=0.05,
                                         max_leaf_nodes=15, min_samples_leaf=10,
                                         l2_regularization=1.0, early_stopping=False,
                                         random_state=s)
def hgb_clf(s):
    return HistGradientBoostingClassifier(max_iter=250, learning_rate=0.05,
                                          max_leaf_nodes=15, min_samples_leaf=10,
                                          l2_regularization=1.0, early_stopping=False,
                                          random_state=s)
def enet_reg(s):
    return Pipeline([('i', SimpleImputer(strategy='median')),
                     ('s', StandardScaler()),
                     ('m', ElasticNetCV(l1_ratio=[0.2, 0.5, 0.9], n_alphas=30, cv=5,
                                        max_iter=5000, random_state=s, n_jobs=NJ))])
def lr_clf(s):
    return Pipeline([('i', SimpleImputer(strategy='median')),
                     ('s', StandardScaler()),
                     ('m', LogisticRegression(C=1.0, penalty='l2', max_iter=5000,
                                              random_state=s))])

# ---------------- MMSE ----------------
mm = {k: {'rho': [], 'r2': []} for k in
      ['baseline_ET', 'HistGB', 'ElasticNet', 'mean_blend', 'rank_blend',
       'w50_blend', 'w70_blend', 'stacking']}
for seed in SEEDS:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    o = {k: np.full(len(mmse), np.nan) for k in ['et', 'hgb', 'enet', 'stack']}
    for tr, te in skf.split(Xf, strat_mmse):
        for k, b in [('et', et_reg), ('hgb', hgb_reg), ('enet', enet_reg)]:
            m = b(seed); m.fit(Xf[tr], mmse[tr]); o[k][te] = m.predict(Xf[te])
        st = StackingRegressor(
            estimators=[('et', et_reg(seed)), ('hgb', hgb_reg(seed)), ('enet', enet_reg(seed))],
            final_estimator=Ridge(alpha=1.0),
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=seed + 1)
                .split(Xf[tr], strat_mmse[tr]),
            n_jobs=1)
        st.fit(Xf[tr], mmse[tr]); o['stack'][te] = st.predict(Xf[te])
    preds = {
        'baseline_ET': o['et'], 'HistGB': o['hgb'], 'ElasticNet': o['enet'],
        'mean_blend': (o['et'] + o['hgb'] + o['enet']) / 3.0,
        'rank_blend': (rankdata(o['et']) + rankdata(o['hgb']) + rankdata(o['enet'])) / 3.0,
        'w50_blend': 0.5 * o['et'] + 0.25 * o['hgb'] + 0.25 * o['enet'],
        'w70_blend': 0.7 * o['et'] + 0.15 * o['hgb'] + 0.15 * o['enet'],
        'stacking': o['stack'],
    }
    for k, p in preds.items():
        assert not np.isnan(p).any()
        mm[k]['rho'].append(spearmanr(mmse, p).statistic)
        if k != 'rank_blend':
            mm[k]['r2'].append(r2_score(mmse, p))
    print(f'[MMSE seed {seed}] {time.time()-t0:.0f}s', flush=True)

# ---------------- CDR ----------------
cc = {k: [] for k in ['baseline_ET', 'HistGB', 'LogReg', 'mean_blend', 'rank_blend',
                      'w50_blend', 'w70_blend', 'stacking']}
for seed in SEEDS:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    o = {k: np.full(len(cdr), np.nan) for k in ['et', 'hgb', 'lr', 'stack']}
    for tr, te in skf.split(Xf, strat_cdr):
        for k, b in [('et', et_clf), ('hgb', hgb_clf), ('lr', lr_clf)]:
            m = b(seed); m.fit(Xf[tr], cdr[tr]); o[k][te] = m.predict_proba(Xf[te])[:, 1]
        st = StackingClassifier(
            estimators=[('et', et_clf(seed)), ('hgb', hgb_clf(seed)), ('lr', lr_clf(seed))],
            final_estimator=LogisticRegression(max_iter=5000, random_state=seed),
            stack_method='predict_proba',
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=seed + 1),
            n_jobs=1)
        st.fit(Xf[tr], cdr[tr]); o['stack'][te] = st.predict_proba(Xf[te])[:, 1]
    preds = {
        'baseline_ET': o['et'], 'HistGB': o['hgb'], 'LogReg': o['lr'],
        'mean_blend': (o['et'] + o['hgb'] + o['lr']) / 3.0,
        'rank_blend': (rankdata(o['et']) + rankdata(o['hgb']) + rankdata(o['lr'])) / 3.0,
        'w50_blend': 0.5 * o['et'] + 0.25 * o['hgb'] + 0.25 * o['lr'],
        'w70_blend': 0.7 * o['et'] + 0.15 * o['hgb'] + 0.15 * o['lr'],
        'stacking': o['stack'],
    }
    for k, p in preds.items():
        assert not np.isnan(p).any()
        cc[k].append(roc_auc_score(cdr, p))
    print(f'[CDR  seed {seed}] {time.time()-t0:.0f}s', flush=True)

print('\n========== VERIFICATION RESULTS ==========')
print('MMSE (pooled OOF per seed, mean +/- sd over 5 seeds):')
for k in mm:
    rho = np.array(mm[k]['rho'])
    if mm[k]['r2']:
        r2 = np.array(mm[k]['r2'])
        print(f'  {k:12s} Spearman={rho.mean():.4f} (+/-{rho.std():.4f})  R2={r2.mean():.4f} (+/-{r2.std():.4f})')
    else:
        print(f'  {k:12s} Spearman={rho.mean():.4f} (+/-{rho.std():.4f})  R2=n/a')
print('CDR (pooled OOF AUC, mean +/- sd over 5 seeds):')
for k in cc:
    a = np.array(cc[k])
    print(f'  {k:12s} AUC={a.mean():.4f} (+/-{a.std():.4f})')
print(f'\nruntime {time.time()-t0:.0f}s')
