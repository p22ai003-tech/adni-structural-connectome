"""Adversarial verification of lever2 (new model classes) NEGATIVE claim.

Independently re-implements the fixed protocol from scratch and re-runs:
  - BASELINE ExtraTrees (both targets)
  - MMSE spot-checks: SVR-RBF C=10 (best lever reg), KNN k=20
  - CDR  spot-checks: KNN k=20, KNN k=10 (best lever clf), SVC-RBF C=1
Protocol: age+sex(F=1) prepended; StratifiedKFold(5, shuffle, seed) for seeds
[11,23,37,51,73]; MMSE strat = group x (mmse>median), CDR strat = cdr_bin;
all fitted preprocessing inside training folds (Pipelines); pooled OOF metrics
per seed, averaged over seeds.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.metrics import r2_score, roc_auc_score

SEEDS = [11, 23, 37, 51, 73]
BASE = '/home/ec2-user/exp/hcp_analysis'

X = pd.read_csv(f'{BASE}/ml_feature_matrix.csv', index_col=0)
y = pd.read_csv(f'{BASE}/ml_targets.csv', index_col=0).loc[X.index]
print(f'n subjects = {len(X)}, n feature cols = {X.shape[1]}')

cov = pd.DataFrame({'age': y['age'], 'sex_F': (y['sex'] == 'F').astype(float)}, index=X.index)
Xf = pd.concat([cov, X], axis=1).values.astype(float)
print(f'design matrix = {Xf.shape}, NaN frac = {np.isnan(Xf).mean():.3f}')

mmse = y['mmse'].values.astype(float)
cdr = y['cdr_bin'].values.astype(int)
strat_mmse = (y['group'].astype(str) + '_' + (y['mmse'] > y['mmse'].median()).astype(str)).values

def cv_reg(make, strat):
    r2s, rhos = [], []
    for seed in SEEDS:
        skf = StratifiedKFold(5, shuffle=True, random_state=seed)
        oof = np.full(len(mmse), np.nan)
        for tr, te in skf.split(Xf, strat):
            m = make(seed); m.fit(Xf[tr], mmse[tr]); oof[te] = m.predict(Xf[te])
        assert not np.isnan(oof).any()
        r2s.append(r2_score(mmse, oof)); rhos.append(spearmanr(mmse, oof).statistic)
    return np.mean(r2s), np.mean(rhos)

def cv_clf(make):
    aucs = []
    for seed in SEEDS:
        skf = StratifiedKFold(5, shuffle=True, random_state=seed)
        oof = np.full(len(cdr), np.nan)
        for tr, te in skf.split(Xf, cdr):
            m = make(seed); m.fit(Xf[tr], cdr[tr]); oof[te] = m.predict_proba(Xf[te])[:, 1]
        assert not np.isnan(oof).any()
        aucs.append(roc_auc_score(cdr, oof))
    return np.mean(aucs)

imp = lambda: SimpleImputer(strategy='median')

print('\n=== MMSE (pooled OOF, mean over seeds) ===')
r2, rho = cv_reg(lambda s: Pipeline([('i', imp()),
    ('m', ExtraTreesRegressor(n_estimators=400, min_samples_leaf=5, random_state=s, n_jobs=8))]), strat_mmse)
print(f'BASELINE ExtraTrees   R2={r2:.4f}  rho={rho:.4f}   (report: 0.1698 / 0.3224)')
r2, rho = cv_reg(lambda s: Pipeline([('i', imp()), ('sc', StandardScaler()),
    ('m', SVR(kernel='rbf', C=10.0, gamma='scale', epsilon=0.1))]), strat_mmse)
print(f'SVR-RBF C=10          R2={r2:.4f}  rho={rho:.4f}   (report: 0.1444 / 0.3216)')
r2, rho = cv_reg(lambda s: Pipeline([('i', imp()), ('sc', StandardScaler()),
    ('m', KNeighborsRegressor(n_neighbors=20, weights='distance'))]), strat_mmse)
print(f'KNN k=20 (dist)       R2={r2:.4f}  rho={rho:.4f}   (report: 0.0857 / 0.2740)')

print('\n=== CDR (pooled OOF AUC, mean over seeds) ===')
auc = cv_clf(lambda s: Pipeline([('i', imp()),
    ('m', ExtraTreesClassifier(n_estimators=400, min_samples_leaf=5, random_state=s, n_jobs=8))]))
print(f'BASELINE ExtraTrees   AUC={auc:.4f}   (report: 0.6018)')
auc = cv_clf(lambda s: Pipeline([('i', imp()), ('sc', StandardScaler()),
    ('m', KNeighborsClassifier(n_neighbors=20, weights='distance'))]))
print(f'KNN k=20 (dist)       AUC={auc:.4f}   (report: 0.5472)')
auc = cv_clf(lambda s: Pipeline([('i', imp()), ('sc', StandardScaler()),
    ('m', KNeighborsClassifier(n_neighbors=10, weights='distance'))]))
print(f'KNN k=10 (dist)       AUC={auc:.4f}   (report: 0.5533)')
auc = cv_clf(lambda s: Pipeline([('i', imp()), ('sc', StandardScaler()),
    ('m', SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, class_weight='balanced', random_state=s))]))
print(f'SVC-RBF C=1 bal       AUC={auc:.4f}   (report: 0.5004)')
