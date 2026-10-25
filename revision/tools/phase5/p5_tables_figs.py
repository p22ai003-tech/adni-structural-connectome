"""Phase 5 step 3 — robustness tables and figures: the rebuilt AAL3 results against the second parcellation (DKT76) and
other edge weights for the exception rule (streamline count, FA), and, once available, ACT tractography (p5_act.py).
Reads the Phase 1-3 outputs of each variant (run_phase5.sh) and of the main rebuilt run."""
import os
import json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

U = Path("/home/ec2-user/exp/docs/iit_dipanjan"); P5 = Path(os.environ.get("IIT_P5", U / "phase5_uniform")); T = P5 / "tables"; T.mkdir(exist_ok=True)
VAR = {"aal3": ("Rebuilt, AAL3 (main)", U / "phase1_uniform", U / "phase2_uniform", "uniform_v3", U / "phase3_uniform", "uniform_v3"),
       "dkt": ("DKT parcellation", P5 / "dkt/phase1", P5 / "dkt/phase2", "uniform_v3_dkt", P5 / "dkt/phase3", "uniform_v3_dkt"),
       "count": ("Streamline-count weights", P5 / "aal3_count/phase1", None, None, P5 / "aal3_count/phase3", "uniform_v3"),
       "fa": ("FA weights", P5 / "aal3_fa/phase1", None, None, P5 / "aal3_fa/phase3", "uniform_v3")}
KEY = [("Limbic AxD, CN–MCI", "WB|Limbic|AxD", "CN-MCI"), ("Salience AxD, CN–MCI", "WB|Salience_VAN|AxD", "CN-MCI"),
       ("Limbic AxD, CN–AD", "WB|Limbic|AxD", "CN-AD"), ("DMN FA, MCI–AD", "WB|DMN|FA", "MCI-AD"),
       ("Limbic LR-MD, CN–MCI", "TR|Limbic|LR-MD", "CN-MCI"), ("Limbic SR-AxD, CN–AD", "TR|Limbic|SR-AxD", "CN-AD"),
       ("Limbic SR-AxD, MCI–AD", "TR|Limbic|SR-AxD", "MCI-AD"), ("Limbic LRexc-MD, CN–MCI", "EX|Limbic|LRexc-MD", "CN-MCI"),
       ("DMN SRexc-RD, CN–AD", "EX|DMN|SRexc-RD", "CN-AD"), ("Frontoparietal SRexc-AxD, MCI–AD", "EX|Frontoparietal|SRexc-AxD", "MCI-AD"),
       ("Core coverage, CN–MCI", "COV|core|coverage", "CN-MCI"), ("Core coverage, CN–AD", "COV|core|coverage", "CN-AD"),
       ("Core coverage, MCI–AD", "COV|core|coverage", "MCI-AD")]
CT = ["CN-MCI", "CN-AD", "MCI-AD"]; PLANNED = ["C1", "C2", "C3", "C4", "C5", "C6", "C8"]
fq = lambda q: "" if not np.isfinite(q) else (f"{q:.3f}" if q >= 0.001 else f"{q:.0e}")
cell = lambda d, q: "" if not np.isfinite(d) else f"{d:+.3f} ({fq(q)})".replace("-", "−")
pub = pd.read_csv(U / "phase1/effects_long.csv"); pub = pub[pub.condition.str.startswith("C0 ")]
EFF = {k: pd.read_csv(v[1] / "effects_long.csv").assign(c=lambda d: d.condition.str.split().str[0]) for k, v in VAR.items()}
SRL = {k: pd.read_csv(v[1] / "sr_lr_summary.csv").assign(c=lambda d: d.condition.str.split().str[0]) for k, v in VAR.items()}


def get(df, c, f, ct):
    r = df[(df.c == c) & (df.feature == f) & (df.contrast == ct)]
    return (float(r.delta.iloc[0]), float(r.q.iloc[0])) if len(r) else (np.nan, np.nan)


# ---------------- T1 key effects, primary analysis (C2), and how many planned controls keep each effect
rows = []
for lab, f, ct in KEY:
    p = pub[(pub.feature == f) & (pub.contrast == ct)]; d0 = float(p.delta.iloc[0])
    r = {"Effect": lab, "Published": cell(d0, float(p.q.iloc[0]))}
    for k, v in VAR.items():
        d, q = get(EFF[k], "C2", f, ct)
        same = k in ("count", "fa") and not f.startswith(("EX", "COV"))
        held = sum(1 for c in PLANNED if (lambda x: np.isfinite(x[0]) and x[1] < 0.05 and np.sign(x[0]) == np.sign(d0))(get(EFF[k], c, f, ct)))
        r[v[0]] = "as main" if same else (cell(d, q) + (f"; holds {held}/{len(PLANNED)}" if np.isfinite(d) else "not estimable"))
    rows.append(r)
pd.DataFrame(rows).to_csv(T / "key_effects_variants.csv", index=False)

# ---------------- T2 network counts (C2) and T3 short - long (C2)
rows = []
for k, v in VAR.items():
    if k in ("count", "fa"): continue
    e = EFF[k][(EFF[k].c == "C2") & (EFF[k].kind == "WB")]
    nnet = e.feature.str.split("|").str[1].nunique()
    r = {"Variant": f"{v[0]} ({nnet} networks)"}
    for m in ["AxD", "MD", "RD", "FA"]:
        r[m] = " / ".join(str(int(((e.measure == m) & (e.contrast == ct) & (e.q < 0.05)).sum())) for ct in CT)
    rows.append(r)
pd.DataFrame(rows).to_csv(T / "network_counts_variants.csv", index=False)
rows = []
for k, v in VAR.items():
    if k in ("count", "fa"): continue
    for kind, nm in (("TR", "all edges"), ("EX", "exception edges")):
        r = {"Variant": v[0], "Edges": nm}
        for ct in CT:
            x = SRL[k][(SRL[k].c == "C2") & (SRL[k].kind == kind) & (SRL[k].contrast == ct)]
            r[ct.replace("-", "–")] = "" if not len(x) else f"{x['diff'].iloc[0]:+.3f} [{x.ci_lo.iloc[0]:+.3f}, {x.ci_hi.iloc[0]:+.3f}]".replace("-", "−")
        rows.append(r)
pd.DataFrame(rows).to_csv(T / "sr_lr_variants.csv", index=False)

# ---------------- T4 Phase 2 length slopes, primary analysis: AAL3 against DKT
ia = pd.read_csv(U / "phase2_uniform/tables/interaction_uniform_v3.csv"); ia = ia[ia.condition == "P2"]
idk = pd.read_csv(P5 / "dkt/phase2/results/interaction_uniform_v3_dkt_P2.csv")
f3 = lambda x: f"{x:+.3f}".replace("-", "−"); fp = lambda p: "< 0.001" if p < 0.001 else f"{p:.3f}"
rows = []
for reg in ["Whole brain", "Limbic"]:
    for ct in CT:
        for m in ["MD", "AxD", "RD", "FA"]:
            r = {"Region": reg, "Contrast": ct.replace("-", "–"), "Measure": m}
            for nm, df in (("AAL3 (main)", ia), ("DKT", idk)):
                x = df[(df.region == reg) & (df.contrast == ct) & (df.measure == m)]
                r[nm] = "" if not len(x) else f"{f3(x.I_delta.iloc[0])} [{f3(x.I_delta_lo.iloc[0])}, {f3(x.I_delta_hi.iloc[0])}] p {fp(x.I_delta_p.iloc[0])}"
                r[nm + ": mixed model p"] = "" if not len(x) else fp(x.lmm_p.iloc[0])
            rows.append(r)
pd.DataFrame(rows).to_csv(T / "length_slopes_dkt.csv", index=False)

# ---------------- T5 Phase 3 null models across parcellation and weights
rows = []
for k, v in VAR.items():
    R = v[4] / "results"; s = v[5]
    if not (R / f"A_core_coverage_{s}.csv").is_file(): continue
    A = pd.read_csv(R / f"A_core_coverage_{s}.csv"); B = pd.read_csv(R / f"B_controls_only_split_{s}.csv"); C = pd.read_csv(R / f"C_split_half_{s}.csv")
    E = pd.read_csv(R / f"E_matched_null_{s}.csv"); F = pd.read_csv(R / f"F_connectivity_adjusted_{s}.csv")
    core = int(A[(A.condition == "P0") & (A.core == "all participants")].core_size.iloc[0])
    r = {"Variant": v[0], "Core edges (pooled)": core}
    if core == 0:
        rows.append(r | {"Split-half Dice (P2)": "no core"}); continue
    cw = C[(C.condition == "P2") & (C["sample"] == "whole sample")]
    r["Split-half Dice (P2)"] = f"{cw.dice_median.iloc[0]:.2f}" if len(cw) else ""
    for ct in CT:
        b = B[(B.condition == "P2") & (B.contrast == ct)]
        r[f"Controls-only {ct.replace('-', '–')}: δ (% splits p<0.05)"] = f"{b.delta_median.iloc[0]:+.3f} ({100 * b.share_p_below_05.iloc[0]:.0f}%)".replace("-", "−") if len(b) else ""
    for ct in ["CN-MCI", "CN-AD"]:
        e = E[(E.condition == "P2") & (E.contrast == ct)]
        r[f"Core more affected than matched, {ct.replace('-', '–')}: p"] = fp(e.p_abs_greater.iloc[0]) if len(e) else ""
    rows.append(r)
pd.DataFrame(rows).to_csv(T / "nulls_variants.csv", index=False)

# ---------------- Figure P5-1: key effects (primary analysis) under each variant
COL = {"aal3": "#2563eb", "dkt": "#ea580c", "count": "#16a34a", "fa": "#9333ea"}
fig, ax = plt.subplots(figsize=(10, 7.5)); y = np.arange(len(KEY))[::-1]
for yi, (lab, f, ct) in zip(y, KEY):
    if yi % 2 == 0: ax.axhspan(yi - 0.5, yi + 0.5, color="#f3f4f6", zorder=0)
    p = pub[(pub.feature == f) & (pub.contrast == ct)]
    ax.scatter([p.delta.iloc[0]], [yi + 0.3], marker="D", s=30, color="#111827", zorder=3, label="published" if yi == y[0] else None)
    for kk, k in enumerate(VAR):
        if k in ("count", "fa") and not f.startswith(("EX", "COV")): continue
        d, q = get(EFF[k], "C2", f, ct)
        if not np.isfinite(d): continue
        yy = yi + 0.15 - kk * 0.15
        ax.scatter([d], [yy], s=34, color=COL[k] if q < 0.05 else "white", edgecolors=COL[k], linewidths=1.4, zorder=3,
                   label=VAR[k][0] + " (primary)" if yi == y[0] else None)
ax.axvline(0, color="#9ca3af", lw=0.8); ax.set_yticks(y); ax.set_yticklabels([k[0] for k in KEY], fontsize=9)
ax.set_xlabel("Cliff's δ (negative = higher in the later-stage group; coverage: positive = lower in the later group)")
ax.legend(fontsize=8, loc="lower left", title="filled = q < 0.05", title_fontsize=8)
plt.tight_layout(); plt.savefig(P5 / "fig_p5_key_effects.png", dpi=300); plt.close()

# ---------------- Figure P5-2: length slopes AAL3 vs DKT
fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharey=True)
lab = [f"{m}, {ct.replace('-', '–')}" for ct in CT for m in ["MD", "AxD", "RD", "FA"]]
for ax, reg in zip(axes, ["Whole brain", "Limbic"]):
    for yi, (ct, m) in enumerate([(ct, m) for ct in CT for m in ["MD", "AxD", "RD", "FA"]]):
        y0 = len(lab) - 1 - yi
        if (yi // 4) % 2 == 0: ax.axhspan(y0 - 0.5, y0 + 0.5, color="#f3f4f6", zorder=0)
        for kk, (nm, df, col) in enumerate((("AAL3 (main)", ia, "#2563eb"), ("DKT", idk, "#ea580c"))):
            x = df[(df.region == reg) & (df.contrast == ct) & (df.measure == m)]
            if not len(x): continue
            x = x.iloc[0]; yy = y0 + 0.12 - kk * 0.24
            ax.plot([x.I_delta_lo, x.I_delta_hi], [yy, yy], color=col, lw=1.3)
            ax.scatter([x.I_delta], [yy], s=26, color=col if x.I_delta_p < 0.05 else "white", edgecolors=col, zorder=3,
                       label=nm if (yi == 0 and reg == "Whole brain") else None)
    ax.axvline(0, color="#6b7280", ls="--", lw=1); ax.set_title(reg, fontsize=10)
    ax.set_xlabel("change in effect per doubling of edge length\n(< 0: short edges more affected)")
axes[0].set_yticks(range(len(lab))[::-1]); axes[0].set_yticklabels(lab, fontsize=8.5); axes[0].legend(fontsize=8, loc="lower left")
plt.tight_layout(); plt.savefig(P5 / "fig_p5_slopes_dkt.png", dpi=300); plt.close()
print("phase 5 tables and figures written")
