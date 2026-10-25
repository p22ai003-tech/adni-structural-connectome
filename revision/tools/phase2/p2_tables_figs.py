"""Phase 2 step 3 — merge the engine outputs, write the report tables and draw the figures.

Usage: python p2_tables_figs.py [source]      (default: published)

For the uniformly rebuilt connectomes (source uniform_v3) the tables and figures show the rebuilt P2 (primary) and P0
beside the same two conditions on the published connectomes (phase2/tables/*_published.csv, suffix "pub"); P10 and P11
are left out because every rebuilt connectome carries diffusion values on all connected edges (they equal P2).
"""
import os
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = sys.argv[1] if len(sys.argv) > 1 else "published"
O = Path(os.environ.get("IIT_P2", "/home/ec2-user/exp/docs/iit_dipanjan/phase2")); R = O / "results"; T = O / "tables"; T.mkdir(exist_ok=True)
CONDS = ["P0", "P2", "P10", "P11"]
LABEL = {"P0": "P0 pooled 530, published, unadjusted", "P2": "P2 primary: ADNI-3/4 after QC, adjusted",
         "P10": "P10 primary + microstructure coverage", "P11": "P11 primary, coverage ≥ 25%"}
COL = {"P0": "#111827", "P2": "#2563eb", "P10": "#ea580c", "P11": "#16a34a"}
CT = ["CN-MCI", "CN-AD", "MCI-AD"]; MEAS = ["MD", "AxD", "RD", "FA"]
cur = pd.concat([pd.read_csv(R / f"curves_{SRC}_{c}.csv") for c in CONDS], ignore_index=True)
itx = pd.concat([pd.read_csv(R / f"interaction_{SRC}_{c}.csv") for c in CONDS], ignore_index=True)
cut = pd.concat([pd.read_csv(R / f"cutoffs_{SRC}_{c}.csv") for c in CONDS], ignore_index=True)
notes = {c: json.load(open(R / f"notes_{SRC}_{c}.json")) for c in CONDS}
for name, df in [("curves", cur), ("interaction", itx), ("cutoffs", cut)]:
    df.to_csv(T / f"{name}_{SRC}.csv", index=False)
TAB = FIG = CONDS; HEAD = {c: c for c in CONDS}
if SRC != "published":                       # add the published-connectome P0 and P2 for comparison
    PT = Path(os.environ.get("IIT_P2_COMPARE", "/home/ec2-user/exp/docs/iit_dipanjan/phase2")) / "tables"
    add = lambda df, name: pd.concat([df, pd.read_csv(PT / f"{name}_published.csv").query("condition in ['P0', 'P2']")
                                      .assign(condition=lambda d: d.condition + "pub")], ignore_index=True)
    cur, itx, cut = add(cur, "curves"), add(itx, "interaction"), add(cut, "cutoffs")
    TAB, FIG = ["P2", "P0", "P2pub", "P0pub"], ["P0pub", "P2pub", "P0", "P2"]
    LABEL.update({"P0": "P0 pooled, rebuilt connectomes, unadjusted", "P2": "P2 primary (ADNI-3/4 after QC, adjusted), rebuilt connectomes",
                  "P0pub": "published connectomes: pooled 530, unadjusted (as published)", "P2pub": "published connectomes: primary"})
    COL.update({"P0pub": "#111827", "P2pub": "#7dd3fc", "P0": "#6b7280"})
    HEAD = {"P2": "Rebuilt: primary (P2)", "P0": "Rebuilt: pooled (P0)", "P2pub": "Published: primary (P2)", "P0pub": "Published: pooled (P0)"}
f3 = lambda x: f"{x:+.3f}".replace("-", "−")
fp = lambda p: ("< 0.001" if p < 0.001 else f"{p:.3f}") if np.isfinite(p) else ""

# ---------------- tables for the report
rows = []
for reg in ["Whole brain", "Limbic"]:
    for ct in CT:
        for m in MEAS:
            r = {"Region": reg, "Contrast": ct.replace("-", "–"), "Measure": m}
            for c in TAB:
                x = itx[(itx.condition == c) & (itx.region == reg) & (itx.contrast == ct) & (itx.measure == m)]
                if len(x):
                    x = x.iloc[0]; r[HEAD[c]] = f"{f3(x.I_delta)} [{f3(x.I_delta_lo)}, {f3(x.I_delta_hi)}] p {fp(x.I_delta_p)}"
            rows.append(r)
pd.DataFrame(rows).to_csv(T / f"length_slopes_{SRC}.csv", index=False)
rows = []
for reg in ["Whole brain", "Limbic"]:
    for ct in CT:
        for m in MEAS:
            x = itx[(itx.condition == "P2") & (itx.region == reg) & (itx.contrast == ct) & (itx.measure == m)].iloc[0]
            rows.append({"Region": reg, "Contrast": ct.replace("-", "–"), "Measure": m, "Effect, shortest bin": f"{x.delta_short:+.3f}",
                         "Effect, longest bin": f"{x.delta_long:+.3f}", "Rank-based slope [95% CI]": f"{f3(x.I_delta)} [{f3(x.I_delta_lo)}, {f3(x.I_delta_hi)}]",
                         "p (permutation)": fp(x.I_delta_p), "Mixed model: interaction (p)": f"{f3(x.lmm_coef)} ({fp(x.lmm_p)})" if np.isfinite(x.lmm_coef) else "",
                         "Two-stage interaction [95% CI], p": f"{f3(x.I)} [{f3(x.lo)}, {f3(x.hi)}], {fp(x.p_perm)}"})
pd.DataFrame(rows).to_csv(T / f"primary_detail_{SRC}.csv", index=False)
rows = []
for sch in cut.scheme.unique():
    for ct in CT:
        r = {"Short / long definition": sch, "Contrast": ct.replace("-", "–")}
        x0 = cut[(cut.scheme == sch) & (cut.condition == "P0")].iloc[0]
        r["Cut-offs (mm)"] = f"≤ {x0.short_max_mm:.0f} / > {x0.long_min_mm:.0f}"
        if "P0pub" in TAB:
            x1 = cut[(cut.scheme == sch) & (cut.condition == "P0pub")].iloc[0]
            r["Cut-offs, published connectomes (mm)"] = f"≤ {x1.short_max_mm:.0f} / > {x1.long_min_mm:.0f}"
        for c in TAB:
            x = cut[(cut.scheme == sch) & (cut.condition == c) & (cut.contrast == ct)]
            if len(x):
                x = x.iloc[0]; r[HEAD[c]] = f"{f3(x['diff'])} [{f3(x.lo)}, {f3(x.hi)}] {int(x.sr_gt_lr)}/{int(x.n_cells)}"
        rows.append(r)
pd.DataFrame(rows).to_csv(T / f"cutoffs_table_{SRC}.csv", index=False)

# ---------------- Figure P2-1/P2-2: effect against length
for reg, fname in [("Whole brain", "fig_p2_curves_wholebrain"), ("Limbic", "fig_p2_curves_limbic")]:
    fig, axes = plt.subplots(4, 3, figsize=(13, 12), sharex=True)
    for i, m in enumerate(MEAS):
        for j, ct in enumerate(CT):
            ax = axes[i, j]
            for c in FIG:
                d = cur[(cur.condition == c) & (cur.region == reg) & (cur.measure == m) & (cur.contrast == ct)].sort_values("bin")
                if not len(d): continue
                sgn = -1 if m != "FA" else 1                     # plot the effect: > 0 = abnormal in the later group
                e, lo, hi = sgn * d.delta, sgn * d.lo, sgn * d.hi
                if sgn < 0: lo, hi = hi, lo
                ax.plot(d.centre_mm, e, marker="o", color=COL[c], ms=3, lw=2 if c == "P2" else 1.1, ls="--" if c == "P0" else (":" if c.endswith("pub") else "-"),
                        label=LABEL[c] if (i == 0 and j == 0) else None, zorder=3 if c == "P2" else 2)
                if c == "P2": ax.fill_between(d.centre_mm, lo, hi, color=COL[c], alpha=0.15, lw=0)
            ax.axhline(0, color="#9ca3af", lw=0.8); ax.set_xscale("log"); ax.set_xticks([30, 60, 120, 240]); ax.set_xticklabels(["30", "60", "120", "240"])
            ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            ax.set_title(f"{m}, {ct.replace('-', '–')}", fontsize=10)
            if i == 3: ax.set_xlabel("edge length (mm, log scale)")
    fig.supylabel("effect size: Cliff's δ, signed so that > 0 = abnormal in the later group", fontsize=10)
    fig.legend(loc="lower center", ncol=2, fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(f"{reg}: group effect in ten edge-length bins (band: 95% bootstrap interval, primary analysis)", fontsize=11)
    plt.tight_layout(rect=(0, 0.04, 1, 0.97)); plt.savefig(O / f"{fname}.png", dpi=250); plt.close()

# ---------------- Figure P2-3: length slopes
fig, axes = plt.subplots(1, 2, figsize=(13, 6.5), sharey=True)
ylab = [f"{m}, {ct.replace('-', '–')}" for ct in CT for m in MEAS]
for ax, reg in zip(axes, ["Whole brain", "Limbic"]):
    for yi, (ct, m) in enumerate([(ct, m) for ct in CT for m in MEAS]):
        y0 = len(ylab) - 1 - yi
        if (yi // 4) % 2 == 0: ax.axhspan(y0 - 0.5, y0 + 0.5, color="#f3f4f6", zorder=0)
        for k, c in enumerate(TAB):
            x = itx[(itx.condition == c) & (itx.region == reg) & (itx.contrast == ct) & (itx.measure == m)]
            if not len(x): continue
            x = x.iloc[0]; yy = y0 + (1.5 - k) * 0.18
            ax.plot([x.I_delta_lo, x.I_delta_hi], [yy, yy], color=COL[c], lw=1.3)
            ax.scatter([x.I_delta], [yy], color=COL[c] if x.I_delta_p < 0.05 else "white", edgecolors=COL[c], s=26, zorder=3,
                       label=LABEL[c] if (yi == 0 and reg == "Whole brain") else None)
    ax.axvline(0, color="#6b7280", ls="--", lw=1); ax.set_title(reg, fontsize=11)
    ax.set_xlabel("change in effect per doubling of edge length\n(> 0: long edges more affected; < 0: short edges more affected)")
axes[0].set_yticks(range(len(ylab))[::-1]); axes[0].set_yticklabels(ylab, fontsize=8.5)
NCOL = 4 if SRC == "published" else 2                    # rebuilt: four long labels, two per row
fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=NCOL, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.01),
           title="filled circle: permutation p < 0.05; open: not significant", title_fontsize=8)
plt.tight_layout(rect=(0, 0.08 if NCOL == 4 else 0.12, 1, 1)); plt.savefig(O / "fig_p2_length_slopes.png", dpi=250); plt.close()

# ---------------- Figure P2-4: alternative cut-offs
schemes = list(cut.scheme.unique())
fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), sharey=True)
for ax, ct in zip(axes, CT):
    for si, sch in enumerate(schemes):
        y0 = len(schemes) - 1 - si
        if si % 2 == 0: ax.axhspan(y0 - 0.5, y0 + 0.5, color="#f3f4f6", zorder=0)
        for k, c in enumerate(TAB):
            x = cut[(cut.scheme == sch) & (cut.condition == c) & (cut.contrast == ct)]
            if not len(x): continue
            x = x.iloc[0]; yy = y0 + (1.5 - k) * 0.18
            ax.plot([x.lo, x.hi], [yy, yy], color=COL[c], lw=1.3)
            ax.scatter([x["diff"]], [yy], color=COL[c], s=24, zorder=3, label=LABEL[c] if (si == 0 and ct == CT[0]) else None)
    ax.axvline(0, color="#6b7280", ls="--", lw=1); ax.set_title(ct.replace("-", "–"), fontsize=10)
    ax.set_xlabel("mean |δ| short-range − long-range (95% CI)")
axes[0].set_yticks(range(len(schemes))[::-1]); axes[0].set_yticklabels(schemes, fontsize=9)
fig.legend(loc="lower center", ncol=NCOL, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.02))
plt.tight_layout(rect=(0, 0.07 if NCOL == 4 else 0.13, 1, 1)); plt.savefig(O / "fig_p2_cutoffs.png", dpi=250); plt.close()
json.dump(notes, open(T / f"notes_{SRC}.json", "w"), indent=1)
print("phase 2 tables and figures written for", SRC)
