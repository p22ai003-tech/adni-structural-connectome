"""Build notebooks/phase3_exception_nulls<suffix>.ipynb (then execute with nbconvert)."""
import os
import nbformat as nbf
from pathlib import Path

NB = Path(f"/home/ec2-user/exp/docs/iit_dipanjan/notebooks/phase3_exception_nulls{os.environ.get('IIT_SUFFIX', '')}.ipynb")
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
# Phase 3 — Null models for the exceptions and the shared core
**Cerebral Cortex revision · ADNI structural connectome (AAL3)**

An **EDR exception** is an edge much stronger than other edges of similar length in the same person's connectome
(strength above the bin mean + 3 SD, in up to 30 equal-count length bins). The manuscript's **core** is the set of
edges that are exceptions in more than half of all participants (51 edges), and **coverage** is the share of the core
a participant expresses. This notebook tests the core step by step:

1. recompute the exceptions and check them against the analysis tree;
2. **circularity** — define the core in half of the controls only and compare the other half with MCI and AD;
3. **stability** — define the core in two random halves and measure their overlap;
4. **thresholds** — 2.5, 3 and 3.5 SD;
5. **specificity** — core edges against ordinary edges matched on strength and length;
6. **total connectivity** — coverage adjusted for density and total weight.

The full results (500 splits, 1,000 matched sets) come from `tools/phase3/p3_engine.py` and are loaded at the end.
""")

code(r"""
import json, os, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
from scipy.stats import rankdata
import matplotlib.pyplot as plt
from IPython.display import Image, display
T0 = time.time(); rng = np.random.default_rng(7)
P1 = Path(os.environ.get("IIT_P1", "/home/ec2-user/exp/docs/iit_dipanjan/phase1")); P2 = Path(os.environ.get("IIT_P2", "/home/ec2-user/exp/docs/iit_dipanjan/phase2"))
P3 = Path(os.environ.get("IIT_P3", "/home/ec2-user/exp/docs/iit_dipanjan/phase3")); SRC = os.environ.get("IIT_SRC", "published")
AR = Path(os.environ.get("IIT_AROOT", "/home/ec2-user/exp/data/derivatives/qc/analysis_cohort"))
E = np.load(P2 / f"edges_{SRC}.npz", allow_pickle=True)
cov = pd.read_csv(P1 / "covariates.csv").set_index("subject_id").loc[E["sid"]]; qc = pd.read_csv(P1 / "qc.csv", index_col=0).reindex(cov.index)
L, W = E["len"].astype(float), E["fd"].astype(float)
print("data:", SRC, "| participants x edges:", L.shape)
""")

md(r"""
## 1. Exceptions, step by step (one participant)
For one participant: the valid edges are split into equal-count length bins (about 120 edges each, at most 30
bins); within each bin an edge is an exception if its SIFT2 weight exceeds the bin mean + 3 SD.
""")

code(r"""
def exceptions_one(l, w, k_sd=3.0, min_bin_edges=120, max_bins=30):
    v = np.isfinite(l) & (l > 0) & np.isfinite(w) & (w > 0); x = np.zeros(len(l), bool); cut = np.full(len(l), np.nan)
    nb = int(np.clip(v.sum() // min_bin_edges, 3, max_bins)); edges = np.unique(np.nanquantile(l[v], np.linspace(0, 1, nb + 1)))
    ids = np.full(len(l), -1); ids[v] = np.searchsorted(edges[1:-1], l[v], side="right")
    for b in np.unique(ids[v]):
        m = v & (ids == b)
        if m.sum() < 20: continue
        c = w[m].mean() + k_sd * w[m].std(ddof=1); cut[m] = c; x[m] = w[m] > c
    return x, cut, v
p = int(np.flatnonzero(cov.group.values == "CN")[0]); x, cut, v = exceptions_one(L[p], W[p])
print(f"{cov.index[p]}: {v.sum()} valid edges, {x.sum()} exceptions")
fig, ax = plt.subplots(figsize=(8, 4))
ax.scatter(L[p][v & ~x], W[p][v & ~x], s=3, color="#9ca3af", label="ordinary edge"); ax.scatter(L[p][x], W[p][x], s=8, color="#dc2626", label="exception")
o = np.argsort(L[p][v]); ax.plot(L[p][v][o], cut[v][o], color="#111827", lw=1, label="bin mean + 3 SD")
ax.set_yscale("log"); ax.set_xlabel("edge length (mm)"); ax.set_ylabel("SIFT2 weight (log)"); ax.legend(fontsize=8); plt.tight_layout(); plt.show()
""")

code(r"""
X3 = np.array([exceptions_one(L[q], W[q])[0] for q in range(len(cov))])
s = json.load(open(P3 / f"results/summary_{SRC}.json"))
print("exceptions per participant, median:", int(np.median(X3.sum(1))), "| engine:", s["validation"])
""")

md(r"""
## 2. Circularity: a core defined in controls only
The published core used everyone, including the people whose groups are then compared. Here the core is defined in
a random half of CN participants (primary sample: ADNI-3/4 after quality control), and coverage is compared between
the other half of CN and the MCI and AD participants. One split in detail, then 100 splits.
""")

code(r"""
def cliff(a, b):
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]; r = rankdata(np.concatenate([a, b])); u = r[:len(a)].sum() - len(a) * (len(a) + 1) / 2
    return 2 * u / (len(a) * len(b)) - 1
prim = np.flatnonzero(((cov.adni34 == 1) & (qc.qc_fail == 0)).values); g = cov.group.values[prim]
cn = np.flatnonzero(g == "CN")
def one_split():
    perm = rng.permutation(cn); disc = perm[: len(cn) // 2]; core = X3[prim[disc]].mean(0) > 0.5
    rest = np.setdiff1d(np.arange(len(prim)), disc); cv = X3[prim[rest]][:, core].mean(1); gg = g[rest]
    return core.sum(), {f"{a}-{b}": cliff(cv[gg == a], cv[gg == b]) for a, b in [("CN", "MCI"), ("CN", "AD"), ("MCI", "AD")]}
n, d = one_split(); print(f"one split: core of {n} edges; coverage δ (unadjusted):", {k: round(v_, 3) for k, v_ in d.items()})
res = pd.DataFrame([one_split()[1] for _ in range(100)])
print("100 splits, median δ:", res.median().round(3).to_dict())
""")

md(r"""
## 3. Stability: two random halves
""")

code(r"""
dice = []
for _ in range(100):
    h1 = np.concatenate([rng.permutation(np.flatnonzero(g == k))[: (g == k).sum() // 2] for k in ("CN", "MCI", "AD")]); h2 = np.setdiff1d(np.arange(len(prim)), h1)
    c1, c2 = X3[prim[h1]].mean(0) > 0.5, X3[prim[h2]].mean(0) > 0.5; dice.append(2 * (c1 & c2).sum() / (c1.sum() + c2.sum()))
print(f"Dice overlap of the two half-cores: median {np.median(dice):.2f} (100 splits)")
""")

md(r"""
## 4–6. Thresholds, specificity and total connectivity — full results
""")

code(r"""
for f in ["fig_p3_frequency_thresholds", "fig_p3_stability", "fig_p3_controls_only", "fig_p3_matched_null"]:
    display(Image(filename=str(P3 / f"{f}.png"), width=1000))
pd.set_option("display.max_colwidth", 90)
for t in ["coverage_variants", "stability", "thresholds", "matched_null"]:
    display(pd.read_csv(P3 / f"tables/{t}_{SRC}.csv"))
print(f"run time {time.time() - T0:.0f} s")
""")

nb = nbf.v4.new_notebook(); nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "iit-phase1", "display_name": "IIT Phase 1 (confound control)", "language": "python"}
nbf.write(nb, NB); print("wrote", NB, "with", len(cells), "cells")
