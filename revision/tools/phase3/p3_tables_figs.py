"""Phase 3 step 2 — report tables and figures from the engine outputs.

Usage: python p3_tables_figs.py [source]
"""
import sys as _sys; from pathlib import Path as _Path; _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import iit_chain as chain; chain.boot()      # IIT_ROOT and the chain settings; no effect when IIT_ROOT is unset
import os
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = sys.argv[1] if len(sys.argv) > 1 else "published"
O = Path(os.environ.get("IIT_P3", "/home/ec2-user/exp/docs/iit_dipanjan/phase3")); R = O / "results"; T = O / "tables"; T.mkdir(exist_ok=True)
rd = lambda n: pd.read_csv(R / f"{n}_{SRC}.csv")
A, Bs, C, Dt, E, F = rd("A_core_coverage"), rd("B_controls_only_split"), rd("C_split_half"), rd("D_thresholds"), rd("E_matched_null"), rd("F_connectivity_adjusted")
core = rd("core_edges"); freq = rd("edge_frequency"); S = json.load(open(R / f"summary_{SRC}.json"))
DIST = np.load(R / f"distributions_{SRC}.npz")
CT = ["CN-MCI", "CN-AD", "MCI-AD"]
f3 = lambda x: f"{x:+.3f}".replace("-", "−")
fp = lambda p: ("< 0.001" if p < 0.001 else f"{p:.3f}") if np.isfinite(p) else ""

# ---------------- T1: coverage differences under every definition
CORE_LAB = "Published core (all 530)" if SRC == "published" else "Pooled-sample core (manuscript definition, all participants)"
rows = []
for c in ["P2", "P0"]:
    add = lambda label, df: rows.append({"Condition": c, "Core / adjustment": label,
                                         **{ct.replace("-", "–"): f"{f3(df[df.contrast == ct].delta.iloc[0])} ({fp(df[df.contrast == ct].p.iloc[0])})" for ct in CT}})
    a = A[A.condition == c]
    for lab in a.core.unique():
        sub = a[a.core == lab]; add(f"{CORE_LAB if lab.startswith('published') else 'Core recomputed in this sample'}, {int(sub.core_size.iloc[0])} edges", sub)
    b = Bs[Bs.condition == c]
    rows.append({"Condition": c, "Core / adjustment": f"Controls-only core, held-out CN (median of {int(b.splits.iloc[0])} splits; core ≈ {int(b.core_size_median.iloc[0])} edges)",
                 **{ct.replace("-", "–"): f"{f3(b[b.contrast == ct].delta_median.iloc[0])} [{f3(b[b.contrast == ct].delta_lo.iloc[0])}, {f3(b[b.contrast == ct].delta_hi.iloc[0])}]; "
                                           f"p < 0.05 in {100 * b[b.contrast == ct].share_p_below_05.iloc[0]:.0f}%" for ct in CT}})
    for sd in [2.5, 3.5]:
        d = Dt[(Dt.condition == c) & (Dt.sd == sd) & (Dt.core_defined_in == "CN only")]
        add(f"Exceptions at {sd} SD, CN-defined core ({int(d.core_size.iloc[0])} edges)", d)
    for lab in F.adjustment.unique():
        add(f"{'Published-style' if SRC == 'published' else 'Pooled-sample'} core {lab}", F[(F.condition == c) & (F.adjustment == lab)])
pd.DataFrame(rows).to_csv(T / f"coverage_variants_{SRC}.csv", index=False)

# ---------------- T2: stability and thresholds
rows = []
for c in ["P2", "P0"]:
    for _, x in C[C.condition == c].iterrows():
        rows.append({"Condition": c, "Split": x["sample"], "Dice of the two half-cores, median [95% range]": f"{x.dice_median:.2f} [{x.dice_lo:.2f}, {x.dice_hi:.2f}]",
                     "Correlation of edge exception frequencies": f"{x.freq_corr_median:.3f}"})
pd.DataFrame(rows).to_csv(T / f"stability_{SRC}.csv", index=False)
rows = []
for c in ["P2", "P0"]:
    for sd in [2.5, 3.0, 3.5]:
        for lab in ["all participants", "CN only"]:
            d = Dt[(Dt.condition == c) & (Dt.sd == sd) & (Dt.core_defined_in == lab)].iloc[0]
            rows.append({"Condition": c, "Threshold (SD)": sd, "Core defined in": lab, "Exceptions per participant (median)": int(d.exceptions_per_participant),
                         "Core edges": int(d.core_size), "Overlap with the 3 SD core (Jaccard)": f"{d.jaccard_with_3sd:.2f}"})
pd.DataFrame(rows).to_csv(T / f"thresholds_{SRC}.csv", index=False)

# ---------------- T3: matched null
rows = []
for c in ["P2", "P0"]:
    for _, x in E[E.condition == c].iterrows():
        rows.append({"Condition": c, "Contrast": x.contrast.replace("-", "–"), "Core edges: mean |δ| of strength": f"{x.core_mean_abs_delta:.3f}",
                     "Matched ordinary edges: mean |δ| (null mean)": f"{x.null_abs_mean:.3f}", "p (core more affected)": fp(x.p_abs_greater),
                     "Core: mean signed δ": f3(x.core_mean_delta), "Null: mean signed δ [95% range]": f"{f3(x.null_mean)} [{f3(x.null_lo)}, {f3(x.null_hi)}]"})
pd.DataFrame(rows).to_csv(T / f"matched_null_{SRC}.csv", index=False)

# ---------------- Figure P3-1: exception frequency and core size against threshold
fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
fa = freq.freq_all[freq.freq_all > 0.02]
axes[0].hist(fa, bins=60, color="#9ca3af"); axes[0].axvline(0.5, color="#dc2626", lw=1.5)
axes[0].set_yscale("log"); axes[0].set_xlabel("share of participants in whom the edge is an exception"); axes[0].set_ylabel("edges (log scale)")
axes[0].set_title(f"Exception frequency of every edge; right of the red line = core ({int((freq.freq_all > 0.5).sum())} edges)", fontsize=9.5)
for k, (c, col) in enumerate([("P2", "#2563eb"), ("P0", "#111827")]):
    d = Dt[(Dt.condition == c) & (Dt.core_defined_in == "all participants")].drop_duplicates("sd")
    axes[1].bar(np.arange(3) + (k - 0.5) * 0.35, d.core_size, width=0.35, color=col, label={"P2": "primary sample", "P0": "pooled 530"}[c])
axes[1].set_xticks(range(3)); axes[1].set_xticklabels(["2.5 SD", "3.0 SD (published)", "3.5 SD"]); axes[1].set_ylabel("core edges")
axes[1].set_title("Core size against the exception threshold", fontsize=9.5); axes[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig(O / "fig_p3_frequency_thresholds.png", dpi=250); plt.close()

# ---------------- Figure P3-2: controls-only core, held-out comparison across splits
fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
for ax, ct in zip(axes, CT):
    d = DIST[f"B__P2__{ct}"]; d = d[np.isfinite(d)]
    if d.size: ax.hist(d, bins=30, color="#93c5fd")
    pub = A[(A.condition == "P2") & (A.core.str.startswith("published")) & (A.contrast == ct)].delta.iloc[0]
    ax.axvline(pub, color="#111827", ls="--", lw=1.2, label=f"{'published' if SRC == 'published' else 'pooled-sample'} core: {pub:+.3f}"); ax.axvline(0, color="#dc2626", lw=1)
    ax.set_title(f"{ct.replace('-', '–')}: coverage δ with a controls-only core", fontsize=9.5); ax.set_xlabel("Cliff's δ (> 0 = lower coverage in the later group)")
    ax.legend(fontsize=7.5)
axes[0].set_ylabel("splits")
plt.tight_layout(); plt.savefig(O / "fig_p3_controls_only.png", dpi=250); plt.close()

# ---------------- Figure P3-3: split-half stability
fig, ax = plt.subplots(figsize=(6.5, 3.4))
for lab, col in [("whole_sample", "#2563eb"), ("CN_only", "#16a34a")]:
    d = DIST[f"C__P2__{lab}"]; d = d[np.isfinite(d)]
    if d.size: ax.hist(d, bins=30, alpha=0.6, color=col, label=f"{lab.replace('_', ' ')}: median {np.median(d):.2f}")
ax.set_xlabel("Dice overlap of the cores defined in two random halves"); ax.set_ylabel("splits"); ax.legend(fontsize=8)
ax.set_title("Split-half stability of the core (primary sample)", fontsize=9.5)
plt.tight_layout(); plt.savefig(O / "fig_p3_stability.png", dpi=250); plt.close()

# ---------------- Figure P3-4: matched null
fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
for ax, ct in zip(axes, CT):
    d = DIST[f"E__P2__{ct}"]; d = d[np.isfinite(d)]; obs = float(DIST[f"Eobs__P2__{ct}"][0])
    if d.size: ax.hist(d, bins=30, color="#d1d5db")
    ax.axvline(obs, color="#2563eb", lw=2, label=f"core edges: {obs:.3f}")
    p = E[(E.condition == "P2") & (E.contrast == ct)].p_abs_greater.iloc[0]
    ax.set_title(f"{ct.replace('-', '–')}: p " + (fp(p) if fp(p).startswith("<") else f"= {fp(p)}"), fontsize=9.5); ax.set_xlabel("mean |δ| of edge strength over the core-sized set"); ax.legend(fontsize=7.5)
axes[0].set_ylabel("matched sets (of 1,000)")
fig.suptitle("Core edges against strength- and length-matched ordinary edges (primary sample)", fontsize=10)
plt.tight_layout(); plt.savefig(O / "fig_p3_matched_null.png", dpi=250); plt.close()
print("phase 3 tables and figures written for", SRC)
