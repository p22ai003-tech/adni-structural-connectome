"""Lever 3 follow-up: fixed a-priori ET-weighted blends (w_ET in {0.5,0.7}, rest split
equally) + rank-blend for CDR. Same fixed protocol; base models identical to lever3.
Saves per-seed OOF predictions to lever3_oof.npz."""
import time, warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (ExtraTreesRegressor, ExtraTreesClassifier,
                              HistGradientBoostingRegressor, HistGradientBoostingClassifier)
from sklearn.linear_model import ElasticNetCV, LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import r2_score, roc_auc_score
warnings.filterwarnings('ignore')

SEEDS = [11, 23, 37, 51, 73]; NJ = 8; t0 = time.time()
X = pd.read_csv('/home/ec2-user/exp/hcp_analysis/ml_feature_matrix.csv', index_col=0)
Y = pd.read_csv('/home/ec2-user/exp/hcp_analysis/ml_targets.csv', index_col=0)
cov = pd.DataFrame({'age': Y['age'].values, 'sex': (Y['sex']=='F').astype(float).values}, index=X.index)
Xf = pd.concat([cov, X], axis=1).values.astype(float)
mmse = Y['mmse'].values.astype(float); cdr = Y['cdr_bin'].values.astype(int)
strat_mmse = (Y['group'] + '_' + (Y['mmse'] > Y['mmse'].median()).astype(str)).values

def et_reg(s):  return Pipeline([('i',SimpleImputer(strategy='median')),('m',ExtraTreesRegressor(n_estimators=400,min_samples_leaf=5,random_state=s,n_jobs=NJ))])
def et_clf(s):  return Pipeline([('i',SimpleImputer(strategy='median')),('m',ExtraTreesClassifier(n_estimators=400,min_samples_leaf=5,random_state=s,n_jobs=NJ))])
def hgb_reg(s): return HistGradientBoostingRegressor(max_iter=250,learning_rate=0.05,max_leaf_nodes=15,min_samples_leaf=10,l2_regularization=1.0,early_stopping=False,random_state=s)
def hgb_clf(s): return HistGradientBoostingClassifier(max_iter=250,learning_rate=0.05,max_leaf_nodes=15,min_samples_leaf=10,l2_regularization=1.0,early_stopping=False,random_state=s)
def enet_reg(s):return Pipeline([('i',SimpleImputer(strategy='median')),('s',StandardScaler()),('m',ElasticNetCV(l1_ratio=[0.2,0.5,0.9],n_alphas=30,cv=5,max_iter=5000,random_state=s,n_jobs=NJ))])
def lr_clf(s):  return Pipeline([('i',SimpleImputer(strategy='median')),('s',StandardScaler()),('m',LogisticRegression(C=1.0,max_iter=5000,random_state=s))])

save = {}
res_m = {k: {'rho': [], 'r2': []} for k in ['baseline_ET','w50_blend','w70_blend']}
res_c = {k: [] for k in ['baseline_ET','w50_blend','w70_blend','rank_blend']}

for seed in SEEDS:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    o = {k: np.full(len(mmse), np.nan) for k in ['et','hgb','enet']}
    for tr, te in skf.split(Xf, strat_mmse):
        for k, b in [('et',et_reg),('hgb',hgb_reg),('enet',enet_reg)]:
            m = b(seed); m.fit(Xf[tr], mmse[tr]); o[k][te] = m.predict(Xf[te])
    for k in o: save[f'mmse_{k}_{seed}'] = o[k]
    for name, w in [('w50_blend',(0.5,0.25,0.25)), ('w70_blend',(0.7,0.15,0.15))]:
        p = w[0]*o['et'] + w[1]*o['hgb'] + w[2]*o['enet']
        res_m[name]['rho'].append(spearmanr(mmse,p).statistic); res_m[name]['r2'].append(r2_score(mmse,p))
    res_m['baseline_ET']['rho'].append(spearmanr(mmse,o['et']).statistic)
    res_m['baseline_ET']['r2'].append(r2_score(mmse,o['et']))
    print(f'[MMSE seed {seed}] {time.time()-t0:.0f}s', flush=True)

for seed in SEEDS:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    o = {k: np.full(len(cdr), np.nan) for k in ['et','hgb','lr']}
    for tr, te in skf.split(Xf, cdr):
        for k, b in [('et',et_clf),('hgb',hgb_clf),('lr',lr_clf)]:
            m = b(seed); m.fit(Xf[tr], cdr[tr]); o[k][te] = m.predict_proba(Xf[te])[:,1]
    for k in o: save[f'cdr_{k}_{seed}'] = o[k]
    for name, w in [('w50_blend',(0.5,0.25,0.25)), ('w70_blend',(0.7,0.15,0.15))]:
        res_c[name].append(roc_auc_score(cdr, w[0]*o['et']+w[1]*o['hgb']+w[2]*o['lr']))
    res_c['rank_blend'].append(roc_auc_score(cdr,(rankdata(o['et'])+rankdata(o['hgb'])+rankdata(o['lr']))/3))
    res_c['baseline_ET'].append(roc_auc_score(cdr, o['et']))
    print(f'[CDR  seed {seed}] {time.time()-t0:.0f}s', flush=True)

np.savez('/home/ec2-user/exp/hcp_analysis/improve/lever3_oof.npz', **save)
print('\n===== LEVER 3b: WEIGHTED BLENDS =====')
print('MMSE:')
for k, v in res_m.items():
    print(f'  {k:12s} Spearman={np.mean(v["rho"]):.4f} (+/-{np.std(v["rho"]):.4f})  R2={np.mean(v["r2"]):.4f} (+/-{np.std(v["r2"]):.4f})')
print('CDR:')
for k, v in res_c.items():
    print(f'  {k:12s} AUC={np.mean(v):.4f} (+/-{np.std(v):.4f})')
print(f'runtime {time.time()-t0:.0f}s')
