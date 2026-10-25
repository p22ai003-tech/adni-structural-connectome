"""Phase 6 — manuscript figures at 300 dpi from the rebuilt-connectome results (Phases 1-5).

  fig_ms1_design.png     study design and sample flow
  fig_ms2_heatmap.png    effect size by network, measure and edge class, three contrasts (primary analysis)
  fig_ms3_brain.png      the exception core on a glass brain, coloured by its coverage loss in AD; network effects by stage
  fig_ms4_length.png     group effect against edge length with 95% bands (primary analysis)
  fig_ms5_srlr.png       short- minus long-range difference across analyses (forest plot)
  fig_ms6_prediction.png out-of-sample R^2 by feature family with the diagnosis-only reference
Run with the pipeline environment (nibabel): /home/ec2-user/exp/.venv_connectome_app/bin/python -I p6_figures.py
"""
import json
from pathlib import Path
import numpy as np, pandas as pd, nibabel as nib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib import cm, colors

U = Path("/home/ec2-user/exp/docs/iit_dipanjan"); O = U / "phase6_uniform"; O.mkdir(exist_ok=True)
P1, P2, P3, P4, P5 = (U / f"phase{k}_uniform" for k in range(1, 6))
A = Path("/data/derivatives/qc/analysis_cohort_uniform")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
GCOL = {"CN": "#38bdf8", "MCI": "#84cc16", "AD": "#fb7185"}
NETS = ["Visual", "Somatomotor", "DorsalAttention", "Salience_VAN", "Limbic", "Frontoparietal", "DMN", "Subcortical", "Cerebellar", "Brainstem"]
NICE = {"DorsalAttention": "Dorsal attention", "Salience_VAN": "Salience/VAN", "Frontoparietal": "Frontoparietal"}
CT = ["CN-MCI", "CN-AD", "MCI-AD"]; ctn = lambda c: c.replace("-", "–")
e = pd.read_csv(P1 / "effects_long.csv"); e["c"] = e.condition.str.split().str[0]
notes = json.load(open(P1 / "condition_notes.json")); n2 = next(v for k, v in notes.items() if k.startswith("C2 ") and k.endswith(" n"))
n0 = next(v for k, v in notes.items() if k.startswith("C0 ") and k.endswith(" n"))

# ---------------- Figure 1: design
fig, ax = plt.subplots(figsize=(11, 6.2)); ax.axis("off"); ax.set_xlim(0, 11); ax.set_ylim(0, 6.2)


def box(x, y, w, h, text, fc="#eff6ff", ec="#2563eb", fs=8.5, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.08", fc=fc, ec=ec, lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else None, wrap=True)


def arrow(x0, y0, x1, y1): ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", color="#374151", lw=1.1))


box(0.2, 5.0, 3.0, 0.95, "ADNI diffusion MRI\n530 sessions (CN 251, MCI 201, AD 78)\nADNI-2, ADNI-3, ADNI-4", bold=False)
box(3.8, 5.0, 3.4, 0.95, "Uniform reconstruction (one recipe)\nCSD, 3M iFOD2 streamlines, SIFT2\nFA, MD, AxD, RD along every streamline")
box(7.8, 5.0, 3.0, 0.95, "Two parcellations, same streamlines\nAAL3 (166 regions, 10 networks)\nDKT (76 regions, 8 networks)")
arrow(3.2, 5.47, 3.8, 5.47); arrow(7.2, 5.47, 7.8, 5.47)
box(3.6, 3.6, 3.8, 1.0, "Quality control\n10 failed reconstructions (registration, density)\n+ 1 diffusivity outlier", fc="#fef2f2", ec="#dc2626")
arrow(5.5, 5.0, 5.5, 4.6)
box(3.2, 2.3, 4.6, 1.0, f"{sum(n0.values()) - 1} participants analysed\nprimary sample ADNI-3/4: n = {sum(n2.values())}\n"
                         f"(CN {n2['CN']}, MCI {n2['MCI']}, AD {n2['AD']})", fc="#f0fdf4", ec="#16a34a", bold=True)
arrow(5.5, 3.6, 5.5, 3.3)
ANA = [("Network contrasts\n10 networks × 4 measures\nall / short / long / exception\nedges; 9 confound controls"),
       ("Length dependence\neffect per doubling of\nlength; mixed model;\nalternative cut-offs"),
       ("Exception core\ncontrols-only definition,\nsplit-half stability,\nmatched-edge null"),
       ("Cognition\nnested prediction of\nCDR-SB, ADAS-Cog and\ncomposite scores\n(≤ 180 days from scan)"),
       ("Robustness\nDKT parcellation,\nstreamline-count / FA\nweights, ACT tracking")]
for k, t in enumerate(ANA):
    x = 0.25 + k * 2.15; box(x, 0.35, 1.95, 1.45, t, fc="#f9fafb", ec="#6b7280", fs=7.6); arrow(5.5, 2.3, x + 0.97, 1.8)
plt.savefig(O / "fig_ms1_design.png", dpi=300, bbox_inches="tight"); plt.close()

# ---------------- Figure 2: effect-size heatmap (primary analysis C2)
CLS = [("WB", "{m}", "all"), ("TR", "SR-{m}", "short"), ("TR", "LR-{m}", "long"), ("EX", "SRexc-{m}", "short exc."), ("EX", "LRexc-{m}", "long exc.")]
MEAS = ["MD", "AxD", "RD", "FA"]
fig, axes = plt.subplots(1, 3, figsize=(15, 5.8), sharey=True)
norm = colors.TwoSlopeNorm(vmin=-0.6, vcenter=0, vmax=0.6)
for ax, ct in zip(axes, CT):
    x2 = e[(e.c == "C2") & (e.contrast == ct)].set_index("feature")
    M = np.full((len(NETS), len(MEAS) * len(CLS)), np.nan); Q = np.full_like(M, np.nan); xl = []
    for mi, m in enumerate(MEAS):
        for ci, (kind, pat, lab) in enumerate(CLS):
            col = mi * len(CLS) + ci; xl.append(f"{m} {lab}")
            for ni, net in enumerate(NETS):
                f = f"{kind}|{net}|{pat.format(m=m)}"
                if f in x2.index:
                    d = x2.loc[f, "delta"]; M[ni, col] = -d if m != "FA" else d; Q[ni, col] = x2.loc[f, "q"]
    im = ax.imshow(M, cmap="RdBu_r", norm=norm, aspect="auto")
    for (i, j), q in np.ndenumerate(Q):
        if np.isfinite(q) and q < 0.05: ax.text(j, i, "•", ha="center", va="center", fontsize=7, color="black")
    for b in range(1, len(MEAS)): ax.axvline(b * len(CLS) - 0.5, color="white", lw=2)
    ax.set_xticks(range(len(xl))); ax.set_xticklabels(xl, rotation=90, fontsize=7); ax.set_title(ctn(ct), fontsize=10)
axes[0].set_yticks(range(len(NETS))); axes[0].set_yticklabels([NICE.get(n, n) for n in NETS])
cb = fig.colorbar(im, ax=axes, fraction=0.015, pad=0.01); cb.set_label("Cliff's δ, signed so that > 0 = abnormal in the later group")
fig.suptitle(f"Primary analysis (ADNI-3/4, adjusted for age, sex and education); • q < 0.05", fontsize=10, y=0.99)
plt.savefig(O / "fig_ms2_heatmap.png", dpi=300, bbox_inches="tight"); plt.close()

# ---------------- Figure 3: core on a glass brain and network effects by stage
atlas = nib.load("/home/ec2-user/exp/atlas/AAL/AAL3v1_1mm_166.nii.gz"); lab = np.asarray(atlas.dataobj).astype(int)
idx = np.argwhere(lab > 0); vals = lab[lab > 0]; xyz = nib.affines.apply_affine(atlas.affine, idx)
cen = np.array([xyz[vals == k].mean(0) for k in range(1, 167)])
mask = np.asarray(nib.load("/home/ec2-user/fsl/data/standard/MNI152_T1_1mm_brain_mask.nii.gz").dataobj) > 0
maff = nib.load("/home/ec2-user/fsl/data/standard/MNI152_T1_1mm_brain_mask.nii.gz").affine
E = dict(np.load(P2 / "edges_uniform_v3.npz", allow_pickle=True)); L, W = E["len"].astype(float), E["fd"].astype(float)
cov = pd.read_csv(P1 / "covariates.csv").set_index("subject_id").loc[E["sid"]]; qc = pd.read_csv(P1 / "qc.csv", index_col=0).reindex(cov.index)


def exceptions(k_sd=3.0, min_bin_edges=120, max_bins=30):
    X = np.zeros(L.shape, bool)
    for p in range(L.shape[0]):
        l, w = L[p], W[p]; v = np.isfinite(l) & (l > 0) & np.isfinite(w) & (w > 0); nv = int(v.sum())
        if nv < 120: continue
        nb = int(np.clip(nv // min_bin_edges, 3, max_bins)); ed = np.unique(np.nanquantile(l[v], np.linspace(0, 1, nb + 1)))
        ids = np.full(L.shape[1], -1); ids[v] = np.searchsorted(ed[1:-1], l[v], side="right")
        for b in np.unique(ids[v]):
            m = v & (ids == b); vv = w[m]
            if vv.size >= 20: X[p, m] = w[m] > vv.mean() + k_sd * vv.std(ddof=1)
    return X


X = exceptions(); coh = (qc.in_cohort == 1).values; core = X[coh].mean(0) > 0.5
prim = coh & (cov.adni34 == 1).values & (qc.qc_fail == 0).values
f_cn = X[prim & (cov.group == "CN").values].mean(0); f_ad = X[prim & (cov.group == "AD").values].mean(0)
loss = (f_cn - f_ad)[core]; ci_, cj_ = E["i"][core], E["j"][core]
fig = plt.figure(figsize=(14, 8.6)); gs = fig.add_gridspec(2, 3, height_ratios=[1.1, 1])


def outline(ax, view):
    proj = {"axial": mask.any(2), "sagittal": mask.any(0), "coronal": mask.any(1)}[view]
    pts = np.argwhere(proj)
    if view == "axial": xs, ys = maff[0, 0] * pts[:, 0] + maff[0, 3], maff[1, 1] * pts[:, 1] + maff[1, 3]
    elif view == "sagittal": xs, ys = maff[1, 1] * pts[:, 0] + maff[1, 3], maff[2, 2] * pts[:, 1] + maff[2, 3]
    else: xs, ys = maff[0, 0] * pts[:, 0] + maff[0, 3], maff[2, 2] * pts[:, 1] + maff[2, 3]
    ax.scatter(xs, ys, s=0.2, color="#e5e7eb", zorder=0, rasterized=True); ax.set_aspect("equal"); ax.axis("off")


cmap = matplotlib.colormaps["magma_r"]; nl = colors.Normalize(vmin=0, vmax=max(0.3, float(np.nanmax(loss))))
top = []
for k, view in enumerate(["axial", "sagittal", "coronal"]):
    ax = fig.add_subplot(gs[0, k]); outline(ax, view); top.append(ax)
    a_, b_ = {"axial": (0, 1), "sagittal": (1, 2), "coronal": (0, 2)}[view]
    order = np.argsort(loss)
    for t in order:
        p, q = cen[ci_[t]], cen[cj_[t]]
        ax.plot([p[a_], q[a_]], [p[b_], q[b_]], color=cmap(nl(loss[t])), lw=0.6 + 2.0 * nl(max(loss[t], 0)), alpha=0.85)
    nodes = np.unique(np.r_[ci_, cj_]); ax.scatter(cen[nodes, a_], cen[nodes, b_], s=6, color="#111827", zorder=3)
    ax.set_title(f"{view} view", fontsize=10)
top[0].text(0, 1.12, f"Exception core ({int(core.sum())} edges)", transform=top[0].transAxes, fontsize=11, fontweight="bold")
sm = cm.ScalarMappable(norm=nl, cmap=cmap); cb = fig.colorbar(sm, ax=top, fraction=0.02, pad=0.01)
cb.set_label("exception frequency CN − AD\n(coverage loss in AD)", fontsize=9)
for k, ct in enumerate(CT):
    ax = fig.add_subplot(gs[1, k]); outline(ax, "axial")
    x2 = e[(e.c == "C2") & (e.contrast == ct) & (e.kind == "WB") & (e.measure == "AxD")].set_index("feature")
    mp = pd.read_csv(A / "19_network_analysis/network_mapping_used.csv").set_index("matrix_idx").functional_network
    dn = np.array([-x2.loc[f"WB|{mp[k_ + 1]}|AxD", "delta"] if f"WB|{mp[k_ + 1]}|AxD" in x2.index else np.nan for k_ in range(166)])
    sc = ax.scatter(cen[:, 0], cen[:, 1], c=dn, cmap="Reds", vmin=0, vmax=0.6, s=28, edgecolors="#374151", linewidths=0.3, zorder=3)
    ax.set_title(f"Network AxD effect, {ctn(ct)}", fontsize=10)
fig.colorbar(sc, ax=fig.axes[-3:], fraction=0.02, pad=0.01).set_label("Cliff's δ (> 0 = higher AxD in the later group)")
plt.savefig(O / "fig_ms3_brain.png", dpi=300, bbox_inches="tight"); plt.close()

# ---------------- Figure 4: effect against length (primary analysis, whole brain)
cur = pd.read_csv(P2 / "tables/curves_uniform_v3.csv")
fig, axes = plt.subplots(len(MEAS), 3, figsize=(11, 10), sharex=True, sharey="row")
for i, m in enumerate(MEAS):
    for j, ct in enumerate(CT):
        ax = axes[i, j]
        for c, col, ls in (("P0", "#6b7280", "--"), ("P2", "#2563eb", "-")):
            d = cur[(cur.condition == c) & (cur.region == "Whole brain") & (cur.measure == m) & (cur.contrast == ct)].sort_values("bin")
            sg = 1 if m == "FA" else -1; y, lo, hi = sg * d.delta, sg * d.lo, sg * d.hi
            if sg < 0: lo, hi = hi, lo
            ax.plot(d.centre_mm, y, color=col, ls=ls, marker="o", ms=3, lw=1.8 if c == "P2" else 1.1,
                    label={"P0": "pooled, unadjusted", "P2": "primary (ADNI-3/4, adjusted)"}[c] if (i == 0 and j == 0) else None)
            if c == "P2": ax.fill_between(d.centre_mm, lo, hi, color=col, alpha=0.15, lw=0)
        ax.axhline(0, color="#9ca3af", lw=0.8); ax.set_xscale("log"); ax.set_xticks([50, 100, 200]); ax.set_xticklabels(["50", "100", "200"])
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        if i == 0: ax.set_title(ctn(ct), fontsize=10)
        if j == 0: ax.set_ylabel(f"{m}\nCliff's δ (> 0 = abnormal)")
        if i == len(MEAS) - 1: ax.set_xlabel("edge length (mm, log scale)")
axes[0, 0].legend(fontsize=8, frameon=False)
plt.tight_layout(); plt.savefig(O / "fig_ms4_length.png", dpi=300); plt.close()

# ---------------- Figure 5: short - long forest across analyses
s1 = pd.read_csv(P1 / "sr_lr_summary.csv").assign(c=lambda d: d.condition.str.split().str[0])
sp = pd.read_csv(U / "phase1/sr_lr_summary.csv"); sp = sp[sp.condition.str.startswith("C0 ")].assign(c="Published")
sd = pd.read_csv(P5 / "dkt/phase1/sr_lr_summary.csv").assign(c=lambda d: "DKT " + d.condition.str.split().str[0])
S = pd.concat([sp, s1, sd]); S = S[S.kind == "TR"]
ROWS = [("Published", "Published connectomes, pooled 530"), ("C0", "Rebuilt, pooled"), ("C1", "Rebuilt, ADNI-3/4"),
        ("C2", "Rebuilt, primary (ADNI-3/4 adjusted)"), ("C3", "+ vendor ComBat"), ("C4", "Propensity-matched"),
        ("C5", "+ atrophy and lesions"), ("C8", "Siemens only"), ("DKT C2", "DKT parcellation, primary")]
COL3 = {"CN-MCI": "#2563eb", "CN-AD": "#ea580c", "MCI-AD": "#16a34a"}
fig, ax = plt.subplots(figsize=(8.5, 6))
for k, (c, labx) in enumerate(ROWS):
    y0 = len(ROWS) - 1 - k
    if k % 2 == 0: ax.axhspan(y0 - 0.5, y0 + 0.5, color="#f3f4f6", zorder=0)
    for j, ct in enumerate(CT):
        r = S[(S.c == c) & (S.contrast == ct)]
        if not len(r): continue
        r = r.iloc[0]; yy = y0 + (1 - j) * 0.25
        ax.plot([r.ci_lo, r.ci_hi], [yy, yy], color=COL3[ct], lw=1.4)
        ax.scatter([r["diff"]], [yy], color=COL3[ct], s=26, zorder=3, label=ctn(ct) if k == 0 else None)
ax.axvline(0, color="#6b7280", ls="--", lw=1); ax.set_yticks(range(len(ROWS))[::-1]); ax.set_yticklabels([r[1] for r in ROWS])
ax.set_xlabel("mean |δ| short-range − long-range cells (95% subject-level bootstrap CI)\n> 0: larger effects on short edges")
ax.legend(fontsize=8, loc="lower right", title="contrast", title_fontsize=8)
plt.tight_layout(); plt.savefig(O / "fig_ms5_srlr.png", dpi=300); plt.close()

# ---------------- Figure 6: prediction by feature family (primary scores)
Mo = pd.read_csv(P4 / "results/models.csv"); Mo = Mo[(Mo["sample"] == "primary") & (Mo.group == "all")]
TGT = [("cdrsb", "CDR-SB"), ("adas13", "ADAS-Cog 13"), ("phc_mem", "Memory composite"), ("phc_exf", "Executive composite")]
LAD = ["C", "C+F1", "C+F1-2", "C+F1-3", "C+F1-4"]; XL = ["covariates", "+ whole\nbrain", "+ tract\nlength", "+ excep-\ntions", "+ archi-\ntecture"]
fig, axes = plt.subplots(1, 4, figsize=(14, 3.8))
for ax, (t, nm) in zip(axes, TGT):
    m = Mo[Mo.target == t].set_index("model"); x = np.arange(len(LAD))
    v = m.loc[LAD]; ax.errorbar(x, v["mean"], yerr=[v["mean"] - v.lo, v.hi - v["mean"]], color="#2563eb", marker="o", ms=4, capsize=3, label="covariates + connectome")
    vd = m.loc[["C+Dx+F1", "C+Dx+F1-2", "C+Dx+F1-3", "C+Dx+F1-4"]]
    ax.errorbar(x[1:] + 0.08, vd["mean"], yerr=[vd["mean"] - vd.lo, vd.hi - vd["mean"]], color="#16a34a", marker="s", ms=3.5, capsize=2, label="diagnosis + connectome")
    ax.axhline(m.loc["C+Dx", "mean"], color="#111827", ls="--", lw=1.1, label="diagnosis only")
    ax.set_xticks(x); ax.set_xticklabels(XL, fontsize=7.5); ax.set_title(nm, fontsize=10); ax.set_ylabel("out-of-sample R²")
axes[0].legend(fontsize=7.5, frameon=False, loc="lower right")
plt.tight_layout(); plt.savefig(O / "fig_ms6_prediction.png", dpi=300); plt.close()
json.dump({"core_edges": int(core.sum()), "core_loss_median": float(np.median(loss)), "core_loss_max": float(loss.max())},
          open(O / "figure_notes.json", "w"), indent=1)
print("manuscript figures written to", O)
