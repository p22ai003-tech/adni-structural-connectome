#!/usr/bin/env python
"""Partial dependence for the same network measure across the four edge ranges.

The page already shows partial dependence for whichever feature leads a class.
This asks a different and more pointed question: taking one network (the DMN,
which leads almost every class) and one measure (the diffusivity composite),
does the relationship with cognition live in short-range edges, long-range
edges, or specifically in the exception edges that this thesis defines?

Same four curves are produced for both targets, pooled over all 202 subjects
(no diagnostic-group split), so the four ranges can be read side by side.

Estimator and protocol match the explanation panels: ExtraTrees 600 / leaf 5 /
balanced, averaged over the five seeds, fitted in sample on the median-imputed
matrix. No SHAP is needed here, only the fitted models.

Output: hcp_analysis/ml_pdp_range_family.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer

# Paths resolve through sc_paths: project-relative, overridable with --out, and
# defaulting to the analysis-tree section the dashboard actually reads. Writing
# straight there is what removes the old manual copy step.
try:
    import sc_paths
except ModuleNotFoundError:  # loose script run from outside hcp_analysis/
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent))
    import sc_paths

_P = sc_paths.resolve_for_script("ml")

H = _P.out
SEEDS = [11, 23, 37, 51, 73]
GRID = 15

RANGES = [
    ("SR_DMN_DIFF", "Short-range", 1),
    ("LR_DMN_DIFF", "Long-range", 2),
    ("SREXC_DMN_DIFF", "Short-range exceptions", 3),
    ("LREXC_DMN_DIFF", "Long-range exceptions", 4),
]


def main() -> None:
    feats = pd.read_csv(H / "ml_feature_matrix.csv", index_col=0)
    t = pd.read_csv(H / "ml_targets.csv", index_col=0)
    cov = pd.DataFrame({"age": t.age,
                        "sex": (t.sex.astype(str).str.upper().str[0] == "F").astype(float)},
                       index=t.index)

    rows = []
    for target in ("mmse_band", "cdr_3level"):
        if target == "mmse_band":
            y0 = t["mmse"].dropna()
            idx = y0.index.intersection(feats.index)
            y = pd.qcut(y0.loc[idx], 3, labels=False, duplicates="drop").astype(int)
            labels = {0: "Low (impaired)", 1: "Mid", 2: "High (intact)"}
        else:
            cdr = t["cdr"].dropna()
            idx = cdr.index.intersection(feats.index)
            y = pd.Series(np.where(cdr.loc[idx] == 0, 0, np.where(cdr.loc[idx] == 0.5, 1, 2)),
                          index=idx)
            labels = {0: "CDR 0", 1: "CDR 0.5", 2: "CDR ≥ 1"}

        X = pd.concat([cov.loc[idx], feats.loc[idx]], axis=1)
        Xi = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(X),
                          columns=X.columns, index=X.index)
        ests = [ExtraTreesClassifier(n_estimators=600, min_samples_leaf=5,
                                     class_weight="balanced", random_state=s, n_jobs=8).fit(Xi, y)
                for s in SEEDS]

        for feat, range_label, order in RANGES:
            # how much of this curve rests on imputed values, not measurements
            missing_pct = float(100.0 * X[feat].isna().mean())
            grid = np.quantile(Xi[feat], np.linspace(0.05, 0.95, GRID))
            base = Xi.copy()
            for gv in grid:
                base[feat] = gv
                probs = np.mean([e.predict_proba(base) for e in ests], axis=0).mean(axis=0)
                for c in range(len(labels)):
                    rows.append(dict(target=target, feature=feat, range_label=range_label,
                                     range_order=order, cls=c, cls_label=labels[c],
                                     grid=float(gv), prob=float(probs[c]),
                                     missing_pct=missing_pct))
        print(f"[{target}] range-family PDP done", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(H / "ml_pdp_range_family.csv", index=False)

    swing = (out.groupby(["target", "range_label", "cls_label"])
             .prob.agg(lambda v: v.max() - v.min()).mul(100).round(2))
    print("\nprobability swing (percentage points) across each feature's 5th-95th percentile:")
    print(swing.to_string())
    print("\nmissing before imputation:")
    print(out.groupby(["range_label"]).missing_pct.first().round(1).to_string())


if __name__ == "__main__":
    main()
