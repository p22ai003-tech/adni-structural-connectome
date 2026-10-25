"""Build notebooks/phase2_length_test.ipynb (then execute with nbconvert)."""
import nbformat as nbf
from pathlib import Path

import os
NB = Path(f"/home/ec2-user/exp/docs/iit_dipanjan/notebooks/phase2_length_test{os.environ.get('IIT_SUFFIX', '')}.ipynb")
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
# Phase 2 — A direct test of the length claim
**Cerebral Cortex revision · ADNI structural connectome (AAL3)**

The manuscript claims that early (CN–MCI) effects are larger on long-range connections and AD-stage effects larger on
short-range ones, based on a paired Wilcoxon test over 40 network × measure cells with quartile cut-offs. Phase 1
showed that test treats dependent cells as independent. This notebook tests length dependence directly:

1. **effect against length** — the group effect in ten edge-length bins;
2. **length dependence** — the slope of that effect on log2 length (rank-based, permutation p, bootstrap interval),
   and a linear mixed model with a diagnosis × length interaction;
3. **alternative cut-offs** — the published 40-cell statistic with terciles, quintiles and fixed 70/140 mm.

Each method is run here step by step on one example (whole-brain MD, CN versus MCI, primary analysis); the full set
of results comes from `tools/phase2/p2_engine.py` (same functions, 2,000 resamples) and is loaded at the end.

**Data.** Option 2 of 24 October: the published connectomes, the quality-controlled primary sample of Phase 1, and
microstructure coverage handled by two checks (P10 coverage-adjusted, P11 coverage ≥ 25%).
""")

code(r"""
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd, scipy, statsmodels
from scipy.stats import rankdata
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from IPython.display import Image, display
import warnings; warnings.filterwarnings("ignore")
T0 = time.time(); rng = np.random.default_rng(1)
import os
O = Path(os.environ.get("IIT_P2", "/home/ec2-user/exp/docs/iit_dipanjan/phase2")); P1 = Path(os.environ.get("IIT_P1", "/home/ec2-user/exp/docs/iit_dipanjan/phase1"))
AR = Path(os.environ.get("IIT_AROOT", "/home/ec2-user/exp/data/derivatives/qc/analysis_cohort")); SRC = os.environ.get("IIT_SRC", "published")
for lib in [np, pd, scipy, statsmodels]: print(f"{lib.__name__:12} {lib.__version__}")
""")

md(r"""
## 1. Edge-level data and the primary sample
`edges_published.npz` holds, for every participant and every one of the 13,695 region pairs, the mean streamline
length, the SIFT2 weight and the mean FA, MD, RD and AxD of the published matrices (written by `p2_edges.py`).
The primary sample and the quality control are those of Phase 1.
""")

code(r"""
E = np.load(O / f"edges_{SRC}.npz", allow_pickle=True)
cov = pd.read_csv(P1 / "covariates.csv").set_index("subject_id").loc[E["sid"]]
qc = pd.read_csv(P1 / "qc.csv", index_col=0).reindex(cov.index)
L = E["len"].astype(float)
print("participants x edges:", L.shape, "| structural edges present:", f"{np.isfinite(L).mean():.3f}",
      "| of those with an MD value:", f"{np.isfinite(E['md']).sum() / np.isfinite(L).sum():.3f}")
prim = ((cov.adni34 == 1) & (qc.qc_fail == 0)).values
rows = np.flatnonzero(prim); g = cov.group.values[rows]
print("primary sample:", dict(pd.Series(g).value_counts()))
MD = E["md"][rows].astype(float); MD[MD > 3e-3] = np.nan                     # quality control: impossible single values
LEN = L[rows]
""")

md(r"""
## 2. Length bins
Ten bins at the deciles of edge length among CN structural edges of the primary sample. For each participant the bin
value is the mean MD over the edges in that bin that carry an MD value (fewer than three edges → missing).
""")

code(r"""
cn = g == "CN"; pool = LEN[cn][np.isfinite(LEN[cn])]
edges_q = np.percentile(pool, np.linspace(0, 100, 11)); edges_q[0], edges_q[-1] = 0, np.inf
centres = np.array([np.median(pool[(pool > edges_q[k]) & (pool <= edges_q[k + 1])]) for k in range(10)])
display(pd.DataFrame({"bin": range(1, 11), "from (mm)": edges_q[:-1].round(1), "to (mm)": edges_q[1:].round(1), "median (mm)": centres.round(1)}))
X = np.full((len(rows), 10), np.nan)
for k in range(10):
    v = np.where((LEN > edges_q[k]) & (LEN <= edges_q[k + 1]), MD, np.nan); n = np.isfinite(v).sum(1)
    X[:, k] = np.where(n >= 3, np.nansum(v, 1) / np.maximum(n, 1), np.nan)
print("participants with a value, per bin:", np.isfinite(X).sum(0))

# covariate adjustment (as in Phase 1): value ~ group + age + sex + education, subtract the covariate part
Z = cov.iloc[rows][["age", "female", "education"]].values.astype(float); Zc = Z - Z.mean(0)
G = pd.get_dummies(pd.Series(g), drop_first=True).values.astype(float)
for k in range(10):
    ok = np.isfinite(X[:, k]); D = np.column_stack([np.ones(ok.sum()), G[ok], Zc[ok]])
    beta = np.linalg.lstsq(D, X[ok, k], rcond=None)[0]; X[ok, k] -= Zc[ok] @ beta[1 + G.shape[1]:]
""")

md(r"""
## 3. The effect against length (example: MD, CN versus MCI)
Cliff's δ in each bin, signed so that > 0 means higher MD in MCI ("abnormal"). The band is a subject-level bootstrap
(participants resampled within groups); 500 resamples here, 2,000 in the engine.
""")

code(r"""
def cliff_cols(A, B):
    out = np.full(A.shape[1], np.nan)
    for c in range(A.shape[1]):
        a, b = A[:, c], B[:, c]; a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        if len(a) < 5 or len(b) < 5: continue
        r = rankdata(np.concatenate([a, b])); u = r[:len(a)].sum() - len(a) * (len(a) + 1) / 2
        out[c] = 2 * u / (len(a) * len(b)) - 1
    return out
ia, ib = np.flatnonzero(g == "CN"), np.flatnonzero(g == "MCI")
e = -cliff_cols(X[ia], X[ib])                                   # > 0 = higher MD in MCI
boots = np.array([-cliff_cols(X[rng.choice(ia, len(ia))], X[rng.choice(ib, len(ib))]) for _ in range(500)])
lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
fig, ax = plt.subplots(figsize=(7, 3.6))
ax.fill_between(centres, lo, hi, color="#2563eb", alpha=0.15); ax.plot(centres, e, "-o", color="#2563eb", ms=4)
ax.set_xscale("log"); ax.set_xticks([30, 60, 120, 240]); ax.set_xticklabels(["30", "60", "120", "240"]); ax.axhline(0, color="#9ca3af", lw=0.8)
ax.set_xlabel("edge length (mm, log scale)"); ax.set_ylabel("effect (Cliff's δ, > 0 = higher in MCI)"); ax.set_title("Whole-brain MD, CN vs MCI, primary analysis", fontsize=10)
plt.tight_layout(); plt.show()
print("effect per bin:", e.round(3))
""")

md(r"""
## 4. Is the length dependence real? Rank-based slope with permutation test
The statistic is the least-squares slope of the effect on log2 length across the ten bins — the change in effect per
**doubling** of edge length. The bootstrap gives its interval. The permutation test shuffles diagnosis labels
between the two groups (so length can have no group-specific effect), recomputes the slope, and asks how often a
slope at least this large arises by chance.
""")

code(r"""
lx = np.log2(centres); lx -= lx.mean()
slope = lambda ev: np.polyfit(lx[np.isfinite(ev)], ev[np.isfinite(ev)], 1)[0]
s_obs = slope(e); s_boot = np.array([slope(b) for b in boots])
both = np.concatenate([ia, ib]); s_perm = []
for _ in range(500):
    p = rng.permutation(both); s_perm.append(slope(-cliff_cols(X[p[:len(ia)]], X[p[len(ia):]])))
s_perm = np.array(s_perm)
print(f"slope = {s_obs:+.3f} per doubling of length, 95% CI [{np.percentile(s_boot, 2.5):+.3f}, {np.percentile(s_boot, 97.5):+.3f}], "
      f"permutation p = {(np.sum(np.abs(s_perm) >= abs(s_obs)) + 1) / (len(s_perm) + 1):.3f}  (500 resamples)")
eng = pd.read_csv(O / f"results/interaction_{SRC}_P2.csv").query("region == 'Whole brain' and measure == 'MD' and contrast == 'CN-MCI'").iloc[0]
print(f"engine (2,000 resamples): slope {eng.I_delta:+.3f}, 95% CI [{eng.I_delta_lo:+.3f}, {eng.I_delta_hi:+.3f}], p = {eng.I_delta_p:.3f}")
fig, ax = plt.subplots(figsize=(6, 3)); ax.hist(s_perm, bins=40, color="#9ca3af"); ax.axvline(s_obs, color="#2563eb", lw=2)
ax.set_title("slopes under shuffled labels (grey) and observed (blue)", fontsize=9); plt.tight_layout(); plt.show()
""")

md(r"""
## 5. The mixed model with a diagnosis × length interaction
Bin values are standardised to CN (z), and z is modelled on group, log2 length and their interaction, with a random
intercept and a random slope for each participant. The interaction coefficient is the difference between groups in
how z changes per doubling of length. This model works on means rather than ranks, so it complements Section 4.
""")

code(r"""
mu, sd = np.nanmean(X[ia], 0), np.nanstd(X[ia], 0, ddof=1); Zs = (X - mu) / sd
sub = np.concatenate([ia, ib])
df = pd.DataFrame({"z": Zs[sub].ravel(), "lx": np.tile(lx, len(sub)), "mci": np.repeat((g[sub] == "MCI").astype(float), 10),
                   "pid": np.repeat(sub, 10)}).dropna()
fit = smf.mixedlm("z ~ mci * lx", df, groups=df.pid, re_formula="~lx").fit(method="lbfgs")
print(fit.summary().tables[1].loc[["mci", "lx", "mci:lx"]])
print(f"\ninteraction: {fit.params['mci:lx']:+.3f} SD per doubling of length (p = {fit.pvalues['mci:lx']:.3f}); engine: {eng.lmm_coef:+.3f} (p = {eng.lmm_p:.3f})")
""")

md(r"""
## 6. Alternative cut-offs: validation first
The 40-cell statistic needs network values on short and long edges. The engine recomputes them from the edge data
with the published recipe (node mean over valid edges, then node-weighted network mean). With the published
quartile cut-offs in the pooled sample it reproduces the published values — checked here for limbic short-range MD.
""")

code(r"""
th = pd.read_csv(AR / "17_edr_exceptions/edr_exception_thresholds.csv").iloc[0]
mp = pd.read_csv(AR / "19_network_analysis/network_mapping_used.csv")
inc = np.zeros((len(E["i"]), 166)); inc[np.arange(len(E["i"])), E["i"]] = 1; inc[np.arange(len(E["i"])), E["j"]] = 1
lim = mp.functional_network.values == "Limbic"
v = E["md"].astype(float); ok = np.isfinite(v) & (L <= float(th.short_max_mm))
S, C = np.where(ok, v, 0) @ inc, ok.astype(float) @ inc
mine = np.where(C[:, lim].sum(1) > 0, S[:, lim].sum(1) / np.maximum(C[:, lim].sum(1), 1), np.nan)
pub = pd.read_csv(P1 / "features_published.csv", index_col=0).reindex(cov.index)["TR|Limbic|SR-MD"].values
print(f"short-range cut-off {th.short_max_mm:.2f} mm; largest |difference| from the published limbic SR-MD values: {np.nanmax(np.abs(mine - pub)):.1e}")
print("engine check over all 40 short-range features:", json.load(open(O / "results/validation_P0.json")))
""")

md(r"""
## 7. All results (2,000 resamples, from the engine)
""")

code(r"""
for f in ["fig_p2_curves_wholebrain", "fig_p2_length_slopes", "fig_p2_cutoffs", "fig_p2_curves_limbic"]:
    display(Image(filename=str(O / f"{f}.png"), width=1000))
""")

code(r"""
pd.set_option("display.max_colwidth", 60)
display(pd.read_csv(O / f"tables/length_slopes_{SRC}.csv"))
display(pd.read_csv(O / f"tables/cutoffs_table_{SRC}.csv"))
print(f"run time {time.time() - T0:.0f} s")
""")

nb = nbf.v4.new_notebook(); nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "iit-phase1", "display_name": "IIT Phase 1 (confound control)", "language": "python"}
nbf.write(nb, NB); print("wrote", NB, "with", len(cells), "cells")
