"""Adversarial verification of lever4 (representation changes).
Independent re-implementation: baseline_ET, c_resid_ET (claimed MMSE winner),
a_QT_ET (claimed small R2 gain). Fixed protocol:
- age + sex(F=1) prepended; repeated stratified 5-fold CV, seeds [11,23,37,51,73]
- MMSE stratified on group + (mmse>median); CDR on cdr_bin
- ALL fitted preprocessing inside training fold only
- pooled OOF metrics per seed, averaged over seeds
"""
import numpy as np, pandas as pd, time
from scipy.stats import spearmanr
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import r2_score, roc_auc_score

SEEDS = [11, 23, 37, 51, 73]

X = pd.read_csv('/home/ec2-user/exp/hcp_analysis/ml_feature_matrix.csv', index_col=0)
T = pd.read_csv('/home/ec2-user/exp/hcp_analysis/ml_targets.csv', index_col=0)
assert X.index.equals(T.index), "index mismatch"

age = T['age'].astype(float).values
sex = (T['sex'] == 'F').astype(float).values
feats = X.values.astype(float)
Z = np.column_stack([age, sex, feats])          # prepended covariates + 103 features
mmse = T['mmse'].astype(float).values
cdr = T['cdr_bin'].astype(int).values
y_strat_mmse = np.char.add(T['group'].astype(str).values, (mmse > np.median(mmse)).astype(str))

def prep_baseline(Ztr, Zte):
    im = SimpleImputer(strategy='median').fit(Ztr)
    return im.transform(Ztr), im.transform(Zte)

def prep_resid(Ztr, Zte):
    """median impute (in-fold) then residualise each feature on intercept+age+sex fit on train."""
    im = SimpleImputer(strategy='median').fit(Ztr[:, 2:])
    Ftr, Fte = im.transform(Ztr[:, 2:]), im.transform(Zte[:, 2:])
    Dtr = np.column_stack([np.ones(Ztr.shape[0]), Ztr[:, :2]])
    Dte = np.column_stack([np.ones(Zte.shape[0]), Zte[:, :2]])
    B, *_ = np.linalg.lstsq(Dtr, Ftr, rcond=None)
    return (np.column_stack([Ztr[:, :2], Ftr - Dtr @ B]),
            np.column_stack([Zte[:, :2], Fte - Dte @ B]))

def prep_qt(Ztr, Zte):
    im = SimpleImputer(strategy='median').fit(Ztr)
    A, Bm = im.transform(Ztr), im.transform(Zte)
    qt = QuantileTransformer(output_distribution='normal',
                             n_quantiles=min(100, A.shape[0]), random_state=0).fit(A)
    return qt.transform(A), qt.transform(Bm)

PREPS = {'baseline_ET': prep_baseline, 'c_resid_ET': prep_resid, 'a_QT_ET': prep_qt}

for name, prep in PREPS.items():
    t0 = time.time(); rhos, r2s, aucs = [], [], []
    for seed in SEEDS:
        # MMSE
        oof = np.empty(len(mmse)); oof[:] = np.nan
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(Z, y_strat_mmse):
            A, B = prep(Z[tr], Z[te])
            r = ExtraTreesRegressor(n_estimators=400, min_samples_leaf=5,
                                    random_state=seed, n_jobs=8).fit(A, mmse[tr])
            oof[te] = r.predict(B)
        assert not np.isnan(oof).any()
        rhos.append(spearmanr(mmse, oof).statistic); r2s.append(r2_score(mmse, oof))
        # CDR
        oof = np.empty(len(cdr)); oof[:] = np.nan
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(Z, cdr):
            A, B = prep(Z[tr], Z[te])
            c = ExtraTreesClassifier(n_estimators=400, min_samples_leaf=5,
                                     random_state=seed, n_jobs=8).fit(A, cdr[tr])
            oof[te] = c.predict_proba(B)[:, 1]
        assert not np.isnan(oof).any()
        aucs.append(roc_auc_score(cdr, oof))
    print(f"{name:12s} rho={np.mean(rhos):.4f}+-{np.std(rhos):.4f} "
          f"R2={np.mean(r2s):.4f}+-{np.std(r2s):.4f} "
          f"AUC={np.mean(aucs):.4f}+-{np.std(aucs):.4f} "
          f"per-seed rho={np.round(rhos,4).tolist()} [{time.time()-t0:.0f}s]", flush=True)
