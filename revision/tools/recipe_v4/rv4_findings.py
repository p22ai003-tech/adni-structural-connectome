"""Recipe v4 findings, side by side: the earlier no-ACT result (quick v6, 530 old connectomes) against the four recipe v4 chains
(ACT 10M = main; no-ACT 3M; ACT 10M without flagged sessions; ACT 3M, the paired arm). Reads result tables only, computes nothing
beyond counts, flags and core overlaps. No participant identifier is printed. Writes <chains>/FINDINGS.md.
Usage: python -I rv4_findings.py"""
import json
from pathlib import Path
import numpy as np, pandas as pd

R4 = Path("/data/derivatives/recipe_v4"); C = R4 / "chains"
ROOTS = {"v6 (old no-ACT)": R4 / "quick_v6", "ACT 10M (main)": C / "act10m", "no-ACT 3M": C / "noact3m",
         "ACT 10M, flagged out": C / "act10m_noflag", "ACT 3M": C / "act3m"}
K = list(ROOTS); MAIN = "ACT 10M (main)"; OLD = "v6 (old no-ACT)"
CT = ["CN-MCI", "CN-AD", "MCI-AD"]; CTN = {"CN-MCI": "CN–MCI", "CN-AD": "CN–AD", "MCI-AD": "MCI–AD"}
NETS = ["Visual", "Somatomotor", "DorsalAttention", "Salience_VAN", "Limbic", "Frontoparietal", "DMN", "Subcortical", "Cerebellar", "Brainstem"]
OUT = []
w = lambda *a: OUT.extend(a)
def f3(x, sign=True):
    if x is None or not np.isfinite(x): return "n/a"
    return (f"{x:+.3f}" if sign else f"{x:.3f}").replace("-", "−")
def fq(q):
    if q is None or not np.isfinite(q): return "n/a"
    return f"{q:.0e}".replace("e-0", "e−").replace("e-", "e−") if q < 0.001 else f"{q:.3f}"
def ci(v, lo, hi): return f"{f3(v)} [{f3(lo)}, {f3(hi)}]"
def star(q): return "" if not np.isfinite(q) else (" *" if q < 0.05 else "")
def table(head, rows):
    w("| " + " | ".join(head) + " |", "|" + "---|" * len(head))
    for r in rows: w("| " + " | ".join(str(c) for c in r) + " |")
    w("")
def cond(df):
    df = df.copy(); df["c"] = df.condition.astype(str).str.split().str[0]; return df
def rd(r, p):
    p = ROOTS[r] / p
    return pd.read_csv(p) if p.is_file() else None

E = {r: cond(rd(r, "phase1/effects_long.csv")) for r in K}
S = {r: cond(rd(r, "phase1/sr_lr_summary.csv")) for r in K}
def eff(r, c, f, ct):
    x = E[r][(E[r].c == c) & (E[r].feature == f) & (E[r].contrast == ct)]
    if not len(x): return (np.nan, np.nan, 0, 0)
    x = x.iloc[0]; return (float(x.delta), float(x.q), int(x.n_first), int(x.n_second))
def ecell(r, c, f, ct):
    d, q, *_ = eff(r, c, f, ct); return f"{f3(d)} ({fq(q)}){star(q)}"
def counts(r, c, m, ct):
    e = E[r]; return int(((e.c == c) & (e.kind == "WB") & (e.measure == m) & (e.contrast == ct) & (e.q < 0.05)).sum())

w("# Recipe v4 results beside the earlier no-ACT results", "")
w("δ is Cliff's delta (negative = higher in the later group, for diffusivity); q is Benjamini–Hochberg within the engine's family; * marks q < .05. "
  "Columns: **v6** = the earlier result on the 530 old connectomes (no ACT, no distortion correction, mixed eddy); **ACT 10M** = recipe v4 main arm; "
  "**no-ACT 3M** = same sessions, unconstrained tracking; **ACT 10M, flagged out** = main arm without the sessions flagged at freeze (WM Dice < .65 or < 30 diffusion directions); "
  "**ACT 3M** = the first 3M streamlines of the ACT run, the arm paired with no-ACT 3M at equal streamline count. Primary sample = ADNI-3, SMC held out, adjusted for age, sex, education (C2) unless a row says otherwise.", "")
rows = []
for r in K:
    s = json.load(open(ROOTS[r] / "chain_settings.json"))
    st = s.get("settings", {})
    rows.append([r, st.get("density_floor"), st.get("r4_field"), st.get("r4_min"), st.get("n_nodes"), st.get("length_measure"), s.get("signed_recipe_check", "")[:70]])
table(["Arm", "density floor", "R4 field", "R4 min", "nodes", "ruler", "recipe check"], rows)

# ---------------------------------------------------------------- samples
w("## 1. Samples", "")
rows = []
for r in K:
    q = pd.read_csv(ROOTS[r] / "phase1/qc.csv", index_col=0); cv = pd.read_csv(ROOTS[r] / "phase1/covariates.csv").set_index("subject_id")
    keep = q.index[q.qc_fail == 0]; g = cv.reindex(keep).group.value_counts()
    n2 = lambda ct: "/".join(map(str, eff(r, "C2", "WB|Limbic|AxD", ct)[2:]))
    rows.append([r, len(q), int((q.qc_fail == 1).sum()), f"{g.get('CN', 0)}/{g.get('MCI', 0)}/{g.get('AD', 0)}", n2("CN-MCI"), n2("CN-AD"), n2("MCI-AD")])
table(["Arm", "sessions in", "excluded by QC", "pooled after QC CN/MCI/AD", "primary CN–MCI n", "primary CN–AD n", "primary MCI–AD n"], rows)
rows = []
for r in K:
    q = pd.read_csv(ROOTS[r] / "phase1/qc.csv", index_col=0)
    vc = q[q.qc_fail == 1].qc_reason.value_counts()
    rows.append([r, "; ".join(f"{k}: {int(v)}" for k, v in vc.items()) or "none"])
table(["Arm", "QC exclusions by reason"], rows)

# ---------------------------------------------------------------- diffusivity
w("## 2. Diffusion measures by network (Table 2): networks with q < .05", "")
for c, lab in (("C2", "C2 primary, adjusted"), ("C0", "C0 pooled, unadjusted"), ("C3", "C3 + vendor ComBat"), ("C5", "C5 + MTL volume + WMH"), ("C7", "C7 amyloid-defined"), ("C8", "C8 Siemens only")):
    rows = []
    for m in ("AxD", "MD", "RD", "FA"):
        rows.append([m] + [" / ".join(str(counts(r, c, m, ct)) for ct in CT) for r in K])
    nn = lambda r: f"{sum(1 for f in E[r][(E[r].c == c) & (E[r].kind == 'WB') & (E[r].measure == 'AxD')].feature.unique())}"
    w(f"**{lab}** (CN–MCI / CN–AD / MCI–AD; networks tested: " + ", ".join(f"{r} {nn(r)}" for r in K) + ")", "")
    table(["Measure"] + K, rows)
KEYF = [("Limbic AxD", "WB|Limbic|AxD"), ("Subcortical AxD", "WB|Subcortical|AxD"), ("Salience/VAN AxD", "WB|Salience_VAN|AxD"), ("DMN AxD", "WB|DMN|AxD"),
        ("Limbic MD", "WB|Limbic|MD"), ("Limbic RD", "WB|Limbic|RD"), ("Limbic FA", "WB|Limbic|FA"), ("DMN FA", "WB|DMN|FA"), ("Limbic long MD", "TR|Limbic|LR-MD"),
        ("Limbic long AxD", "TR|Limbic|LR-AxD"), ("Limbic short AxD", "TR|Limbic|SR-AxD"), ("Limbic long-exception MD", "EX|Limbic|LRexc-MD"),
        ("Limbic long-exception AxD", "EX|Limbic|LRexc-AxD"), ("DMN short-exception RD", "EX|DMN|SRexc-RD"), ("Frontoparietal short-exception AxD", "EX|Frontoparietal|SRexc-AxD"),
        ("Core coverage", "COV|core|coverage")]
rows = []
for lab, f in KEYF:
    for ct in CT:
        rows.append([lab, CTN[ct]] + [ecell(r, "C2", f, ct) for r in K])
w("**Named effects quoted in the text, C2: δ (q)**", ""); table(["Effect", "Contrast"] + K, rows)
rows = []
for m in ("AxD", "MD", "RD", "FA"):
    for ct in CT:
        rows.append([m, CTN[ct]] + [", ".join(f"{n[:4]} {f3(eff(r, 'C2', f'WB|{n}|{m}', ct)[0])}{star(eff(r, 'C2', f'WB|{n}|{m}', ct)[1])}" for n in NETS if np.isfinite(eff(r, 'C2', f'WB|{n}|{m}', ct)[0])) for r in (OLD, MAIN)])
w("**Every network, C2, v6 against ACT 10M** (first four letters of the network)", ""); table(["Measure", "Contrast", OLD, MAIN], rows)
rows = []
for c, lab in (("C0", "C0 pooled, unadjusted"), ("C0q", "C0q pooled after QC"), ("C1", "C1 primary, unadjusted"), ("C2", "C2 primary, adjusted"), ("C3", "C3 + vendor ComBat"),
               ("C4", "C4 propensity-matched"), ("C5", "C5 + MTL volume + WMH"), ("C6", "C6 consistent edges"), ("C7", "C7 amyloid-defined"), ("C8", "C8 Siemens only")):
    for ct in CT:
        rows.append([lab, CTN[ct]] + [f"{ecell(r, c, 'WB|Limbic|AxD', ct)} n {'/'.join(map(str, eff(r, c, 'WB|Limbic|AxD', ct)[2:]))}" for r in K])
w("**Limbic AxD under each confound control: δ (q) n**", ""); table(["Condition", "Contrast"] + K, rows)
rows = []
for c, lab in (("C2", "C2"), ("C5", "C5 + MTL + WMH"), ("C8", "C8 Siemens")):
    for f, fl in (("COV|core|coverage", "Core coverage"), ("WB|Limbic|MD", "Limbic MD"), ("WB|DMN|FA", "DMN FA")):
        for ct in CT:
            rows.append([lab, fl, CTN[ct]] + [ecell(r, c, f, ct) for r in K])
w("**Other quoted effects under atrophy adjustment and Siemens only**", ""); table(["Condition", "Effect", "Contrast"] + K, rows)

# ---------------------------------------------------------------- length
w("## 3. Length", "")
IT = {r: rd(r, "phase2/tables/interaction_uniform_v3.csv") for r in K}
for r in K: IT[r] = IT[r][IT[r].condition == "P2"]
for reg in ("Whole brain", "Limbic"):
    rows = []
    for ct in CT:
        for m in ("MD", "RD", "AxD", "FA"):
            row = [CTN[ct], m]
            for r in K:
                x = IT[r][(IT[r].region == reg) & (IT[r].contrast == ct) & (IT[r].measure == m)].iloc[0]
                ex = "" if (x.I_delta_lo <= 0 <= x.I_delta_hi) else " †"
                row.append(f"{ci(x.I_delta, x.I_delta_lo, x.I_delta_hi)}{ex}; mixed {f3(x.lmm_coef)} p {fq(x.lmm_p)}")
            rows.append(row)
    w(f"**{reg}: slope of δ per doubling of MNI distance (bootstrap 95% CI; † = CI excludes 0) and mixed-model diagnosis × log2 length.** Negative = short edges more affected.", "")
    table(["Contrast", "Measure"] + K, rows)
rows = []
for c, lab in (("C2", "C2 primary, adjusted (Table 3)"), ("C0", "C0 pooled"), ("C3", "C3 + ComBat"), ("C5", "C5 + MTL + WMH"), ("C8", "C8 Siemens")):
    for kind in ("TR", "EX"):
        for ct in CT:
            row = [lab, "all edges" if kind == "TR" else "exceptions", CTN[ct]]
            for r in K:
                x = S[r][(S[r].c == c) & (S[r].kind == kind) & (S[r].contrast == ct)]
                if not len(x): row.append("n/a"); continue
                x = x.iloc[0]; ex = "" if (x.ci_lo <= 0 <= x.ci_hi) else " †"
                row.append(f"{ci(x['diff'], x.ci_lo, x.ci_hi)}{ex} ({int(x.sr_gt_lr)}/{int(x.n_cells)})")
            rows.append(row)
w("**Short minus long, mean |δ| over network × measure cells (bootstrap 95% CI; cells with short > long).** Positive = short connections more affected.", "")
table(["Condition", "Edges", "Contrast"] + K, rows)
CU = {r: rd(r, "phase2/tables/cutoffs_uniform_v3.csv") for r in K}
rows = []
for sch in ("Quartiles (published)", "Terciles", "Quintiles"):
    for ct in CT:
        row = [sch, CTN[ct]]
        for r in K:
            x = CU[r][(CU[r].condition == "P2") & (CU[r].scheme == sch) & (CU[r].contrast == ct)].iloc[0]
            ex = "" if (x.lo <= 0 <= x.hi) else " †"
            row.append(f"{ci(x['diff'], x.lo, x.hi)}{ex}")
        rows.append(row)
w("**Cut-off choice (Phase 2 engine, P2)**", ""); table(["Scheme", "Contrast"] + K, rows)

# ---------------------------------------------------------------- core
w("## 4. Exceptions and core", "")
key = lambda d: set(zip(d.i.astype(int), d.j.astype(int)))
dice = lambda a, b: 2 * len(a & b) / max(len(a) + len(b), 1)
cores = {r: key(rd(r, "phase3/results/core_edges_uniform_v3.csv")) for r in K}
table(["Dice of cores"] + K, [[a, len(cores[a])] [0:1] + [f"{dice(cores[a], cores[b]):.2f}" for b in K] for a in K])
w("Core sizes: " + "; ".join(f"{r} {len(cores[r])}" for r in K) + ".", "")
P3 = {r: {s: rd(r, f"phase3/results/{s}_uniform_v3.csv") for s in ("A_core_coverage", "B_controls_only_split", "C_split_half", "E_matched_null", "F_connectivity_adjusted")} for r in K}
rows = []
for ct in CT:
    row = ["controls-only core, 500 splits", CTN[ct]]
    for r in K:
        b = P3[r]["B_controls_only_split"]; b = b[(b.condition == "P2") & (b.contrast == ct)].iloc[0]
        row.append(f"{f3(b.delta_median)} [{f3(b.delta_lo)}, {f3(b.delta_hi)}], {100 * b.share_p_below_05:.0f}% p<.05")
    rows.append(row)
for ct in CT:
    row = ["core vs 1,000 matched non-core sets", CTN[ct]]
    for r in K:
        e_ = P3[r]["E_matched_null"]; e_ = e_[(e_.condition == "P2") & (e_.contrast == ct)].iloc[0]
        row.append(f"{e_.core_mean_abs_delta:.3f} vs {e_.null_abs_mean:.3f}, p {fq(e_.p_abs_greater)}")
    rows.append(row)
F = {r: P3[r]["F_connectivity_adjusted"] for r in K}
for adj in F[MAIN][F[MAIN].condition == "P2"].adjustment.unique():
    if "route" in adj: continue                 # one recipe for every v4 session: the route row adjusts for nothing there
    for ct in CT:
        row = [f"coverage {adj}", CTN[ct]]
        for r in K:
            x = F[r][(F[r].condition == "P2") & (F[r].adjustment == adj) & (F[r].contrast == ct)]
            row.append(f"{f3(x.delta.iloc[0])} (p {fq(x.p.iloc[0])})" if len(x) else "n/a")
        rows.append(row)
row = ["split-half Dice, whole sample / CN only", ""]
for r in K:
    c_ = P3[r]["C_split_half"]; c_ = c_[c_.condition == "P2"].set_index("sample")
    row.append(f"{c_.loc['whole sample', 'dice_median']:.2f} / {c_.loc['CN only', 'dice_median']:.2f}")
rows.append(row)
w("**Phase 3 checks on the core (primary)**: positive coverage δ = lower coverage in the later group.", ""); table(["Check", "Contrast"] + K, rows)

# ---------------------------------------------------------------- Phase 4
w("## 5. Prediction of cognitive scores (Phase 4, primary)", "")
TN = {"cdrsb": "CDR-SB", "adas13": "ADAS-Cog 13", "phc_mem": "Memory", "phc_exf": "Executive", "mmse": "MMSE", "faq": "FAQ"}
C4 = {r: rd(r, "phase4/results/comparisons.csv") for r in K}; M4 = {r: rd(r, "phase4/results/models.csv") for r in K}
def c4(r, t, a, b, g="all"):
    x = C4[r][(C4[r]["sample"] == "primary") & (C4[r].target == t) & (C4[r].model == a) & (C4[r].versus == b) & (C4[r].group == g)]
    return x.iloc[0] if len(x) else None
rows = []
for t, lab in TN.items():
    for what, a, b, g in (("gain over diagnosis", "C+Dx+F1-4", "C+Dx", "all"), ("gain within MCI", "C+F1-4", "C", "MCI")):
        row = [lab, what]
        for r in K:
            x = c4(r, t, a, b, g)
            if x is None: row.append("n/a"); continue
            ex = " †" if (x.diff_lo > 0 or x.diff_hi < 0) else ""
            row.append(f"{ci(x['diff'], x.diff_lo, x.diff_hi)}{ex} n {int(x.n)}")
        rows.append(row)
w("**Out-of-sample R² gain from connectome features F1–F4 (paired resampling, 95% CI; † = CI excludes 0)**", ""); table(["Score", "Comparison"] + K, rows)
rows = []
for t, lab in list(TN.items()) + [("cdr_impaired", "CDR ≥ 0.5 (AUC)")]:
    row = [lab]
    for r in K:
        m = M4[r][(M4[r]["sample"] == "primary") & (M4[r].target == t) & (M4[r].group == "all")].set_index("model")
        row.append(f"Dx {m.loc['C+Dx', 'mean']:.3f}, conn {m.loc['C+F1-4', 'mean']:.3f}, both {m.loc['C+Dx+F1-4', 'mean']:.3f}")
    rows.append(row)
w("**Model ladder: covariates + diagnosis / + connectome / + both (mean out-of-sample R² or AUC)**", ""); table(["Score"] + K, rows)
rows = []
for r in K:
    d = rd(r, "phase4/p4_data.csv"); fam = rd(r, "phase4/feature_families.csv")
    if d is None or fam is None: rows.append([r, "n/a", "", ""]); continue
    feats = [f for f in fam.feature if f in d]
    iqr = (d[feats].quantile(.75) - d[feats].quantile(.25)).replace(0, np.nan); z = ((d[feats] - d[feats].median()) / iqr).abs()
    worst = z.max().sort_values(ascending=False).head(3)
    rows.append([r, int((z > 50).any(axis=1).sum()), ", ".join(f"{f} ({fam.set_index('feature').family[f]}) {v:.0f}" for f, v in worst.items()),
                 int((z > 50).any(axis=0).sum())])
w("**Data check: sessions with any connectome feature more than 50 IQRs from the median** (ridge models are not robust to these; a model ladder whose F4 rungs collapse points here)", "")
table(["Arm", "sessions", "largest |value − median| / IQR (family)", "features affected"], rows)

# ---------------------------------------------------------------- Phase 5
w("## 6. Robustness (Phase 5)", "")
for r in K:
    k5 = rd(r, "phase5/tables/key_effects_variants.csv")
    w(f"**Key effects across variants, {r}**", ""); table(list(k5.columns), k5.fillna("").values.tolist())
for r in K:
    s5 = rd(r, "phase5/tables/sr_lr_variants.csv")
    w(f"**Short minus long across variants, {r}**", ""); table(list(s5.columns), s5.fillna("").values.tolist())
txt = ROOTS["ACT 3M"] / "phase5/tables/act_summary.txt"
if txt.is_file():
    w("**Paired ACT 3M against no-ACT 3M (same sessions, same streamline count; `chains/act3m/phase5/tables/`)**", "", txt.read_text().strip(), "")
    ke = pd.read_csv(ROOTS["ACT 3M"] / "phase5/tables/act_key_effects.csv"); table(list(ke.columns), ke.values.tolist())
    ag = pd.read_csv(ROOTS["ACT 3M"] / "phase5/tables/act_agreement.csv"); table(list(ag.columns), ag.values.tolist())

# ---------------------------------------------------------------- tree ML
w("## 7. Thesis analysis tree: diagnosis classifier (18_ml_diagnostics)", "")
rows = []
for r in K:
    p = rd(r, "analysis_tree/18_ml_diagnostics/ml_model_performance.csv")
    if p is None: rows.append([r, "missing"]); continue
    p = p[p.status.astype(str).str.startswith("ok")] if "status" in p and p.status.astype(str).str.startswith("ok").any() else p
    b = p.sort_values("macro_auroc_ovr", ascending=False).iloc[0]
    rows.append([r, int(b.n_subjects), b.model, f"{b.macro_auroc_ovr:.3f}", f"{b.balanced_accuracy:.3f}", ", ".join(f"{m.split('_')[0]} {a:.3f}" for m, a in zip(p.model, p.macro_auroc_ovr))])
table(["Arm", "n", "best model", "macro AUROC", "balanced accuracy", "all models AUROC"], rows)
(C / "FINDINGS.md").write_text("\n".join(OUT) + "\n")
print(f"wrote {C / 'FINDINGS.md'} ({len(OUT)} lines)")
