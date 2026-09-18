"""Lever 2: new model classes vs ExtraTrees baseline, fixed protocol.

Protocol: age+sex(F=1) prepended; repeated stratified 5-fold CV, seeds [11,23,37,51,73];
MMSE stratified on group + (mmse>median), CDR on cdr_bin; all fitted preprocessing
inside the training fold (sklearn Pipelines); pooled OOF metrics per seed, mean over seeds.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import r2_score, roc_auc_score

SEEDS = [11, 23, 37, 51, 73]
BASE = '/home/ec2-user/exp/hcp_analysis'

X = pd.read_csv(f'{BASE}/ml_feature_matrix.csv', index_col=0)
y = pd.read_csv(f'{BASE}/ml_targets.csv', index_col=0).loc[X.index]

# Covariates prepended: age + sex (F=1)
cov = pd.DataFrame({'age': y['age'], 'sex_F': (y['sex'] == 'F').astype(float)}, index=X.index)
Xfull = pd.concat([cov, X], axis=1).values.astype(float)

mmse = y['mmse'].values.astype(float)
cdr = y['cdr_bin'].values.astype(int)
# MMSE stratification label: group x (mmse > median)
strat_mmse = (y['group'].astype(str) + '_' + (y['mmse'] > y['mmse'].median()).astype(str)).values


def run_cv_reg(make_model, Xa, ya, strat):
    r2s, rhos = [], []
    for seed in SEEDS:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        oof = np.full(len(ya), np.nan)
        for tr, te in skf.split(Xa, strat):
            m = make_model(seed)
            m.fit(Xa[tr], ya[tr])
            oof[te] = m.predict(Xa[te])
        r2s.append(r2_score(ya, oof))
        rhos.append(spearmanr(ya, oof).statistic)
    return float(np.mean(r2s)), float(np.mean(rhos))


def run_cv_clf(make_model, Xa, ya):
    aucs = []
    for seed in SEEDS:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        oof = np.full(len(ya), np.nan)
        for tr, te in skf.split(Xa, ya):
            m = make_model(seed)
            m.fit(Xa[tr], ya[tr])
            oof[te] = m.predict_proba(Xa[te])[:, 1]
        aucs.append(roc_auc_score(ya, oof))
    return float(np.mean(aucs))


imp = lambda: SimpleImputer(strategy='median')
qt = lambda seed: QuantileTransformer(output_distribution='normal',
                                      n_quantiles=100, random_state=seed)

# ---------- MMSE ----------
reg_configs = {
    'SVR-RBF C=1':  lambda s: Pipeline([('i', imp()), ('sc', StandardScaler()),
                                        ('m', SVR(kernel='rbf', C=1.0, gamma='scale', epsilon=0.1))]),
    'SVR-RBF C=10': lambda s: Pipeline([('i', imp()), ('sc', StandardScaler()),
                                        ('m', SVR(kernel='rbf', C=10.0, gamma='scale', epsilon=0.1))]),
    'KNN k=10 (dist)': lambda s: Pipeline([('i', imp()), ('sc', StandardScaler()),
                                           ('m', KNeighborsRegressor(n_neighbors=10, weights='distance'))]),
    'KNN k=20 (dist)': lambda s: Pipeline([('i', imp()), ('sc', StandardScaler()),
                                           ('m', KNeighborsRegressor(n_neighbors=20, weights='distance'))]),
    'Ridge a=10 +QT':  lambda s: Pipeline([('i', imp()), ('q', qt(s)), ('m', Ridge(alpha=10.0))]),
    'Ridge a=100 +QT': lambda s: Pipeline([('i', imp()), ('q', qt(s)), ('m', Ridge(alpha=100.0))]),
}

base_reg = lambda s: Pipeline([('i', imp()),
    ('m', ExtraTreesRegressor(n_estimators=400, min_samples_leaf=5, random_state=s, n_jobs=8))])

print('=== MMSE (pooled OOF, mean over 5 seeds) ===')
b_r2, b_rho = run_cv_reg(base_reg, Xfull, mmse, strat_mmse)
print(f'BASELINE ExtraTrees      R2={b_r2:.4f}  rho={b_rho:.4f}')
mmse_results = {}
for name, mk in reg_configs.items():
    r2v, rho = run_cv_reg(mk, Xfull, mmse, strat_mmse)
    mmse_results[name] = (r2v, rho)
    print(f'{name:24s} R2={r2v:.4f}  rho={rho:.4f}')

# ---------- CDR ----------
clf_configs = {
    'SVC-RBF C=1 bal':  lambda s: Pipeline([('i', imp()), ('sc', StandardScaler()),
        ('m', SVC(kernel='rbf', C=1.0, gamma='scale', probability=True,
                  class_weight='balanced', random_state=s))]),
    'SVC-RBF C=10 bal': lambda s: Pipeline([('i', imp()), ('sc', StandardScaler()),
        ('m', SVC(kernel='rbf', C=10.0, gamma='scale', probability=True,
                  class_weight='balanced', random_state=s))]),
    'KNN k=10 (dist)': lambda s: Pipeline([('i', imp()), ('sc', StandardScaler()),
        ('m', KNeighborsClassifier(n_neighbors=10, weights='distance'))]),
    'KNN k=20 (dist)': lambda s: Pipeline([('i', imp()), ('sc', StandardScaler()),
        ('m', KNeighborsClassifier(n_neighbors=20, weights='distance'))]),
    'LogReg C=0.1 bal +QT': lambda s: Pipeline([('i', imp()), ('q', qt(s)),
        ('m', LogisticRegression(C=0.1, class_weight='balanced', max_iter=5000, random_state=s))]),
    'LogReg C=1 bal +QT':   lambda s: Pipeline([('i', imp()), ('q', qt(s)),
        ('m', LogisticRegression(C=1.0, class_weight='balanced', max_iter=5000, random_state=s))]),
}

base_clf = lambda s: Pipeline([('i', imp()),
    ('m', ExtraTreesClassifier(n_estimators=400, min_samples_leaf=5, random_state=s, n_jobs=8))])

print()
print('=== CDR (pooled OOF AUC, mean over 5 seeds) ===')
b_auc = run_cv_clf(base_clf, Xfull, cdr)
print(f'BASELINE ExtraTrees      AUC={b_auc:.4f}')
for name, mk in clf_configs.items():
    auc = run_cv_clf(mk, Xfull, cdr)
    print(f'{name:24s} AUC={auc:.4f}')
