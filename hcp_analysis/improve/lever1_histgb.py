"""Lever 1: HistGradientBoosting tuning vs ExtraTrees baseline (fixed protocol)."""
import time
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.metrics import r2_score, roc_auc_score

SEEDS = [11, 23, 37, 51, 73]
NJ = 8

X = pd.read_csv('/home/ec2-user/exp/hcp_analysis/ml_feature_matrix.csv', index_col=0)
T = pd.read_csv('/home/ec2-user/exp/hcp_analysis/ml_targets.csv', index_col=0)
assert X.index.equals(T.index)

# Covariates age + sex(F=1) prepended
cov = pd.DataFrame({'age': T['age'], 'sex': (T['sex'] == 'F').astype(float)}, index=T.index)
Xf = pd.concat([cov, X], axis=1).values.astype(float)

y_mmse = T['mmse'].values.astype(float)
y_cdr = T['cdr_bin'].values.astype(int)
strat_mmse = (T['group'].astype(str) + '_' + (T['mmse'] > T['mmse'].median()).astype(int).astype(str)).values
strat_cdr = y_cdr

HGB_CONFIGS = [
    ('hgb1_d2_lr.03_600_l2-1',   dict(max_depth=2,    learning_rate=0.03, max_iter=600, l2_regularization=1.0)),
    ('hgb2_d2_lr.1_300_l2-.1',   dict(max_depth=2,    learning_rate=0.1,  max_iter=300, l2_regularization=0.1)),
    ('hgb3_d3_lr.03_600_l2-1',   dict(max_depth=3,    learning_rate=0.03, max_iter=600, l2_regularization=1.0)),
    ('hgb4_d3_lr.1_300_l2-1',    dict(max_depth=3,    learning_rate=0.1,  max_iter=300, l2_regularization=1.0)),
    ('hgb5_dN_leaf15_lr.03_600', dict(max_depth=None, max_leaf_nodes=15, learning_rate=0.03, max_iter=600, l2_regularization=0.1)),
    ('hgb6_dN_leaf31_lr.03_300', dict(max_depth=None, max_leaf_nodes=31, learning_rate=0.03, max_iter=300, l2_regularization=1.0)),
]

def run_cv(make_model, task, impute):
    """Pooled OOF metrics per seed, averaged over seeds. task: 'reg' or 'clf'."""
    strat = strat_mmse if task == 'reg' else strat_cdr
    y = y_mmse if task == 'reg' else y_cdr
    per_seed = []
    for seed in SEEDS:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        oof = np.full(len(y), np.nan)
        for tr, te in skf.split(Xf, strat):
            Xtr, Xte = Xf[tr], Xf[te]
            if impute:  # fitted inside training fold only
                imp = SimpleImputer(strategy='median')
                Xtr = imp.fit_transform(Xtr)
                Xte = imp.transform(Xte)
            m = make_model(seed)
            m.fit(Xtr, y[tr])
            oof[te] = m.predict_proba(Xte)[:, 1] if task == 'clf' else m.predict(Xte)
        assert not np.isnan(oof).any()
        if task == 'reg':
            per_seed.append((r2_score(y, oof), spearmanr(y, oof)[0]))
        else:
            per_seed.append((roc_auc_score(y, oof),))
    return tuple(np.mean([s[i] for s in per_seed]) for i in range(len(per_seed[0])))

results = {}
t0 = time.time()

# --- Baseline: ExtraTrees n=400, min_samples_leaf=5, median imputation ---
r2, rho = run_cv(lambda s: ExtraTreesRegressor(n_estimators=400, min_samples_leaf=5, random_state=s, n_jobs=NJ), 'reg', impute=True)
results['BASELINE_ET_mmse'] = (r2, rho)
print(f"BASELINE ExtraTrees  MMSE: R2={r2:.4f}  Spearman={rho:.4f}   [{time.time()-t0:.0f}s]", flush=True)
auc, = run_cv(lambda s: ExtraTreesClassifier(n_estimators=400, min_samples_leaf=5, random_state=s, n_jobs=NJ), 'clf', impute=True)
results['BASELINE_ET_cdr'] = (auc,)
print(f"BASELINE ExtraTrees  CDR : AUC={auc:.4f}   [{time.time()-t0:.0f}s]", flush=True)

# --- Lever: HistGB configs (native NaN handling, no imputation) ---
for name, kw in HGB_CONFIGS:
    r2, rho = run_cv(lambda s, kw=kw: HistGradientBoostingRegressor(random_state=s, **kw), 'reg', impute=False)
    results[name + '_mmse'] = (r2, rho)
    print(f"{name:28s} MMSE: R2={r2:.4f}  Spearman={rho:.4f}   [{time.time()-t0:.0f}s]", flush=True)
for name, kw in HGB_CONFIGS:
    auc, = run_cv(lambda s, kw=kw: HistGradientBoostingClassifier(random_state=s, **kw), 'clf', impute=False)
    results[name + '_cdr'] = (auc,)
    print(f"{name:28s} CDR : AUC={auc:.4f}   [{time.time()-t0:.0f}s]", flush=True)

best_m = max((k for k in results if k.endswith('_mmse') and k.startswith('hgb')), key=lambda k: results[k][0])
best_c = max((k for k in results if k.endswith('_cdr') and k.startswith('hgb')), key=lambda k: results[k][0])
print("\n=== SUMMARY ===")
print(f"MMSE baseline: R2={results['BASELINE_ET_mmse'][0]:.4f} rho={results['BASELINE_ET_mmse'][1]:.4f}")
print(f"MMSE best HGB: {best_m}  R2={results[best_m][0]:.4f} rho={results[best_m][1]:.4f}")
print(f"CDR  baseline: AUC={results['BASELINE_ET_cdr'][0]:.4f}")
print(f"CDR  best HGB: {best_c}  AUC={results[best_c][0]:.4f}")
print(f"total {time.time()-t0:.0f}s")
