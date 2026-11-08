"""Phase 4 step 3 — report tables and figures from results/ (p4_engine.py)."""
import sys as _sys; from pathlib import Path as _Path; _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import iit_chain as chain; chain.boot()      # IIT_ROOT and the chain settings; no effect when IIT_ROOT is unset
import os
import json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

O = Path(os.environ.get("IIT_P4", "/home/ec2-user/exp/docs/iit_dipanjan/phase4_uniform")); R = O / "results"; T = O / "tables"; T.mkdir(exist_ok=True)
M = pd.read_csv(R / "models.csv"); Cm = pd.read_csv(R / "comparisons.csv"); A = pd.read_csv(R / "association.csv")
D = pd.read_csv(O / "p4_data.csv", index_col=0); notes = json.load(open(O / "notes.json"))
TN = {"cdrsb": "CDR-SB", "adas13": "ADAS-Cog 13", "phc_mem": "Memory (PHC)", "phc_exf": "Executive (PHC)", "mmse": "MMSE",
      "faq": "FAQ", "cdr_impaired": "CDR ≥ 0.5 (AUC)"}
TARGETS = list(TN)
f3 = lambda x: "" if not np.isfinite(x) else f"{x:+.3f}".replace("-", "−")
f2 = lambda x: "" if not np.isfinite(x) else f"{x:.3f}"
fp = lambda p: "" if not np.isfinite(p) else ("< 0.002" if p < 0.002 else f"{p:.3f}")

# ---------------- T1 score availability
rows = []
for t in TARGETS:
    lagc = f"{t}_lag" if t != "cdr_impaired" else "cdr_global_lag"
    for s, m in (("primary", chain.primary(D) == 1), ("pooled", D.index.notnull())):
        x = D[m & D[t].notna()]
        rows.append({"Score": TN[t].replace(" (AUC)", ""), "Sample": chain.sample_label() if s == "primary" else "pooled", "n": len(x),
                     **{g: int((x.group == g).sum()) for g in ("CN", "MCI", "AD")}, "Median days from scan": f"{x[lagc].median():.0f}"})
pd.DataFrame(rows).to_csv(T / "scores.csv", index=False)

# ---------------- T2 model performance, primary sample
ORDER = ["C", "C+F1", "C+F1-2", "C+F1-3", "C+F1-4", "C+Dx", "C+Dx+F1", "C+Dx+F1-2", "C+Dx+F1-3", "C+Dx+F1-4", "C+Dx+F2", "C+Dx+F3", "C+Dx+F4"]
LAB = {"C": "Covariates (age, sex, education)", "C+F1": "+ whole brain (F1)", "C+F1-2": "+ tract length (F2)", "C+F1-3": "+ exceptions (F3)",
       "C+F1-4": "+ architecture (F4) = covariates + connectome", "C+Dx": "Covariates + diagnosis", "C+Dx+F1": "+ whole brain (F1)",
       "C+Dx+F1-2": "+ tract length (F2)", "C+Dx+F1-3": "+ exceptions (F3)", "C+Dx+F1-4": "+ architecture (F4) = diagnosis + connectome",
       "C+Dx+F2": "Diagnosis + tract length only", "C+Dx+F3": "Diagnosis + exceptions only", "C+Dx+F4": "Diagnosis + architecture only"}
for s in ("primary", "pooled"):
    m = M[(M["sample"] == s) & (M.group == "all")]
    tab = m.assign(cell=m.apply(lambda r: f"{r['mean']:.3f} [{r.lo:.3f}, {r.hi:.3f}]", axis=1)).pivot_table(index="model", columns="target", values="cell", aggfunc="first")
    tab = tab.reindex(index=ORDER, columns=TARGETS)
    tab.index = [LAB[i] for i in tab.index]; tab.columns = [TN[c] for c in tab.columns]
    tab.rename_axis("Model").reset_index().to_csv(T / f"models_{s}.csv", index=False)

# ---------------- T3 comparisons
KEYC = [("C+F1-4", "C", "Connectome over covariates"), ("C+Dx", "C", "Diagnosis over covariates"),
        ("C+Dx+F1-4", "C+Dx", "Connectome over diagnosis (key)"), ("C+Dx+F1", "C+Dx", "Whole brain over diagnosis"),
        ("C+Dx+F2", "C+Dx", "Tract length over diagnosis"), ("C+Dx+F3", "C+Dx", "Exceptions over diagnosis"),
        ("C+Dx+F4", "C+Dx", "Architecture over diagnosis"), ("C+Dx+F1-2", "C+Dx+F1", "Ladder: + tract length after whole brain"),
        ("C+Dx+F1-3", "C+Dx+F1-2", "Ladder: + exceptions after F1–F2"), ("C+Dx+F1-4", "C+Dx+F1-3", "Ladder: + architecture after F1–F3")]
for s in ("primary", "pooled"):
    rows = []
    for a, b, lab in KEYC:
        r = {"Comparison": lab}
        for t in TARGETS:
            x = Cm[(Cm["sample"] == s) & (Cm.group == "all") & (Cm.target == t) & (Cm.model == a) & (Cm.versus == b)].iloc[0]
            r[TN[t]] = f"{f3(x['diff'])} [{f3(x.diff_lo)}, {f3(x.diff_hi)}]" + (f"; p {fp(x.perm_p)}" if np.isfinite(x.perm_p) else "")
        rows.append(r)
    pd.DataFrame(rows).to_csv(T / f"comparisons_{s}.csv", index=False)

# ---------------- T4 within diagnosis
rows = []
for s in ("primary", "pooled"):
    for g in ("CN", "MCI", "AD"):
        r = {"Sample": chain.sample_label() if s == "primary" else "pooled", "Group": g}
        for t in TARGETS:
            x = Cm[(Cm["sample"] == s) & (Cm.group == g) & (Cm.target == t)]
            r[TN[t]] = "" if not len(x) else f"{f3(x['diff'].iloc[0])} [{f3(x.diff_lo.iloc[0])}, {f3(x.diff_hi.iloc[0])}]; p {fp(x.perm_p.iloc[0])}"
        rows.append(r)
pd.DataFrame(rows).to_csv(T / "within_diagnosis.csv", index=False)

# ---------------- T5 associations (primary)
a = A[A["sample"] == "primary"]
tab = a.assign(cell=a.apply(lambda r: f"{r.rho:+.2f}{'*' if r.q < 0.05 else ''}".replace("-", "−"), axis=1))
tab = tab.pivot_table(index=["group", "feature"], columns="target", values="cell", aggfunc="first")
tab = tab.reindex(columns=[t for t in TARGETS if t in tab.columns]).rename(columns=TN)
tab = tab.reindex(pd.MultiIndex.from_product([["CN", "MCI", "AD"], list(dict.fromkeys(a.feature))]))
tab.rename_axis(["Group", "Measure"]).reset_index().to_csv(T / "association_primary.csv", index=False)

# ---------------- Figure P4-1: performance by feature family, with diagnosis-only and diagnosis + connectome lines
COL = {"conn": "#2563eb", "dx": "#111827", "dxc": "#16a34a"}
fig, axes = plt.subplots(2, 4, figsize=(15, 7.4)); axes = axes.ravel()
LAD = ["C", "C+F1", "C+F1-2", "C+F1-3", "C+F1-4"]; XL = ["covariates", "+F1 whole\nbrain", "+F2 tract\nlength", "+F3\nexceptions", "+F4 archi-\ntecture"]
for ax, t in zip(axes, TARGETS):
    m = M[(M["sample"] == "primary") & (M.group == "all") & (M.target == t)].set_index("model")
    x = np.arange(len(LAD)); v = m.loc[LAD]
    ax.errorbar(x, v["mean"], yerr=[v["mean"] - v.lo, v.hi - v["mean"]], color=COL["conn"], marker="o", ms=5, lw=1.6, capsize=3,
                label="covariates + connectome, adding one family at a time")
    vd = m.loc[["C+Dx+F1", "C+Dx+F1-2", "C+Dx+F1-3", "C+Dx+F1-4"]]
    ax.errorbar(x[1:] + 0.08, vd["mean"], yerr=[vd["mean"] - vd.lo, vd.hi - vd["mean"]], color=COL["dxc"], marker="s", ms=4, lw=1.2,
                capsize=2, label="diagnosis + covariates + connectome")
    ax.axhline(m.loc["C+Dx", "mean"], color=COL["dx"], ls="--", lw=1.2, label="diagnosis + covariates (no connectome)")
    ax.set_xticks(x); ax.set_xticklabels(XL, fontsize=7.5); ax.set_title(TN[t], fontsize=10)
    ax.set_ylabel("AUC" if t == "cdr_impaired" else "out-of-sample R²", fontsize=8.5)
axes[-1].axis("off")
h, l = axes[0].get_legend_handles_labels(); axes[-1].legend(h, l, loc="center", fontsize=9, frameon=False,
                                                           title=f"{chain.sample_label()} after quality control;\nbars: 2.5–97.5% over 20 repeats of 5-fold CV",
                                                           title_fontsize=8.5)
plt.tight_layout(); plt.savefig(O / "fig_p4_ladder.png", dpi=300); plt.close()

# ---------------- Figure P4-2: what the connectome adds beyond diagnosis, and within each diagnosis
fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), sharey=True)
for ax, (title, sel) in zip(axes, [("Beyond diagnosis (all participants)", lambda t: [("all", "C+Dx+F1-4", "C+Dx")]),
                                    ("Within each diagnosis (covariates + connectome over covariates)", lambda t: [(g, "C+F1-4", "C") for g in ("CN", "MCI", "AD")])]):
    cols = {"all": "#16a34a", "CN": "#38bdf8", "MCI": "#84cc16", "AD": "#fb7185"}
    for yi, t in enumerate(TARGETS):
        y0 = len(TARGETS) - 1 - yi
        if yi % 2 == 0: ax.axhspan(y0 - 0.5, y0 + 0.5, color="#f3f4f6", zorder=0)
        items = sel(t)
        for k, (g, a, b) in enumerate(items):
            x = Cm[(Cm["sample"] == "primary") & (Cm.group == g) & (Cm.target == t) & (Cm.model == a) & (Cm.versus == b)]
            if not len(x): continue
            x = x.iloc[0]; yy = y0 + (len(items) - 1) / 2 * 0.22 - k * 0.22
            ax.plot([x.diff_lo, x.diff_hi], [yy, yy], color=cols[g], lw=1.5)
            ax.scatter([x["diff"]], [yy], s=34, color=cols[g] if (x.diff_lo > 0 or x.diff_hi < 0) else "white", edgecolors=cols[g], linewidths=1.4, zorder=3,
                       label=(g if g != "all" else "diagnosis + connectome over diagnosis") if yi == 0 else None)
    ax.axvline(0, color="#6b7280", ls="--", lw=1); ax.set_title(title, fontsize=10)
    ax.set_xlabel("gain in out-of-sample R² (AUC for CDR ≥ 0.5), 95% CI"); ax.legend(fontsize=8, loc="lower right")
axes[0].set_yticks(range(len(TARGETS))[::-1]); axes[0].set_yticklabels([TN[t] for t in TARGETS], fontsize=9)
fig.text(0.5, 0.005, "filled: 95% interval excludes zero; open: it does not", ha="center", fontsize=8.5)
plt.tight_layout(rect=(0, 0.03, 1, 1)); plt.savefig(O / "fig_p4_gain.png", dpi=300); plt.close()

# ---------------- Figure P4-3: within-diagnosis partial correlations
feats = list(dict.fromkeys(A.feature)); tg = [t for t in TARGETS if t in set(A.target)]
fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), sharey=True)
for ax, g in zip(axes, ("CN", "MCI", "AD")):
    a = A[(A["sample"] == "primary") & (A.group == g)]
    Mx = a.pivot_table(index="feature", columns="target", values="rho").reindex(index=feats, columns=tg)
    Q = a.pivot_table(index="feature", columns="target", values="q").reindex(index=feats, columns=tg)
    im = ax.imshow(Mx.values, cmap="RdBu_r", vmin=-0.4, vmax=0.4, aspect="auto")
    for i in range(len(feats)):
        for j in range(len(tg)):
            v = Mx.values[i, j]
            if np.isfinite(v): ax.text(j, i, f"{v:+.2f}{'*' if Q.values[i, j] < 0.05 else ''}".replace("-", "−"), ha="center", va="center", fontsize=7.5)
    n = int(a.n.median()) if len(a) else 0
    ax.set_title(f"{g} (n ≈ {n})", fontsize=10); ax.set_xticks(range(len(tg))); ax.set_xticklabels([TN[t] for t in tg], rotation=35, ha="right", fontsize=8)
axes[0].set_yticks(range(len(feats))); axes[0].set_yticklabels(feats, fontsize=8.5)
cb = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.01); cb.set_label("partial Spearman ρ (age, sex, education)", fontsize=8.5)
plt.savefig(O / "fig_p4_association.png", dpi=300, bbox_inches="tight"); plt.close()
print("phase 4 tables and figures written")
