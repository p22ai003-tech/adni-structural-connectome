"""Adversarial verification of lever1_histgb report: independent re-run of
baseline ExtraTrees + best-claimed HGB configs (hgb1 MMSE, hgb5 CDR), fixed protocol."""
import time
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.ensemble import (ExtraTreesRegressor, ExtraTreesClassifier,
                              HistGradientBoostingRegressor, HistGradientBoostingClassifier)
from sklearn.metrics import r2_score, roc_auc_score

SEEDS = [11, 23, 37, 51, 73]
NJ = 8

X = pd.read_csv('/home/ec2-user/exp/hcp_analysis/ml_feature_matrix.csv', index_col=0)
T = pd.read_csv('/home/ec2-user/exp/hcp_analysis/ml_targets.csv', index_col=0)
assert X.index.equals(T.index)
print(f"n={len(X)}  n_features={X.shape[1]} (+2 covariates)")

cov = pd.DataFrame({'age': T['age'], 'sex': (T['sex'] == 'F').astype(float)}, index=T.index)
Xf = pd.concat([cov, X], axis=1).values.astype(float)
y_mmse = T['mmse'].values.astype(float)
y_cdr = T['cdr_bin'].values.astype(int)
strat_mmse = (T['group'].astype(str) + '_' + (T['mmse'] > T['mmse'].median()).astype(int).astype(str)).values
strat_cdr = y_cdr

def run_cv(make_model, task, impute):
    strat = strat_mmse if task == 'reg' else strat_cdr
    y = y_mmse if task == 'reg' else y_cdr
    per_seed = []
    for seed in SEEDS:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        oof = np.full(len(y), np.nan)
        for tr, te in skf.split(Xf, strat):
            Xtr, Xte = Xf[tr], Xf[te]
            if impute:
                imp = SimpleImputer(strategy='median')
                Xtr = imp.fit_transform(Xtr)
                Xte = imp.transform(Xte)
            m = make_model(seed)
            m.fit(Xtr, y[tr])
            oof[te] = m.predict_proba(Xte)[:, 1] if task == 'clf' else m.predict(Xte)
        assert not np.isnan(oof).any()
        per_seed.append((r2_score(y, oof), spearmanr(y, oof)[0]) if task == 'reg'
                        else (roc_auc_score(y, oof),))
    return tuple(np.mean([s[i] for s in per_seed]) for i in range(len(per_seed[0]))), per_seed

t0 = time.time()
(r2, rho), ps = run_cv(lambda s: ExtraTreesRegressor(n_estimators=400, min_samples_leaf=5,
                                                     random_state=s, n_jobs=NJ), 'reg', True)
print(f"VERIFY BASELINE ET MMSE: R2={r2:.4f} rho={rho:.4f}  per-seed R2={[f'{a:.3f}' for a,_ in ps]}  [{time.time()-t0:.0f}s]", flush=True)
(auc,), ps = run_cv(lambda s: ExtraTreesClassifier(n_estimators=400, min_samples_leaf=5,
                                                   random_state=s, n_jobs=NJ), 'clf', True)
print(f"VERIFY BASELINE ET CDR : AUC={auc:.4f}  per-seed={[f'{a:.3f}' for a, in ps]}  [{time.time()-t0:.0f}s]", flush=True)

hgb1 = dict(max_depth=2, learning_rate=0.03, max_iter=600, l2_regularization=1.0)
(r2h, rhoh), ps = run_cv(lambda s: HistGradientBoostingRegressor(random_state=s, **hgb1), 'reg', False)
print(f"VERIFY hgb1 MMSE: R2={r2h:.4f} rho={rhoh:.4f}  per-seed R2={[f'{a:.3f}' for a,_ in ps]}  [{time.time()-t0:.0f}s]", flush=True)

hgb5 = dict(max_depth=None, max_leaf_nodes=15, learning_rate=0.03, max_iter=600, l2_regularization=0.1)
(auch,), ps = run_cv(lambda s: HistGradientBoostingClassifier(random_state=s, **hgb5), 'clf', False)
print(f"VERIFY hgb5 CDR : AUC={auch:.4f}  per-seed={[f'{a:.3f}' for a, in ps]}  [{time.time()-t0:.0f}s]", flush=True)

print("\n=== VERDICT NUMBERS ===")
print(f"MMSE: baseline R2={r2:.4f}/rho={rho:.4f}  vs hgb1 R2={r2h:.4f}/rho={rhoh:.4f}  (claimed 0.1698/0.3224 vs 0.1604/0.3202)")
print(f"CDR : baseline AUC={auc:.4f}  vs hgb5 AUC={auch:.4f}  (claimed 0.6018 vs 0.5502)")
print(f"total {time.time()-t0:.0f}s")
