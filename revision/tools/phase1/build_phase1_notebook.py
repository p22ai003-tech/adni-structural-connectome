"""Build notebooks/phase1_confound_control.ipynb (then execute with nbconvert)."""
import sys as _sys; from pathlib import Path as _Path; _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import iit_chain as chain; chain.boot()      # IIT_ROOT and the chain settings; no effect when IIT_ROOT is unset
import os
import nbformat as nbf
from pathlib import Path

import os
NB = Path(os.environ.get("IIT_NOTEBOOKS", "/home/ec2-user/exp/docs/iit_dipanjan/notebooks")) / f"phase1_confound_control{os.environ.get('IIT_SUFFIX', '')}.ipynb"
NB.parent.mkdir(exist_ok=True)
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
# Phase 1 — Confound control
**Cerebral Cortex revision · ADNI structural connectome (AAL3, 530 participants)**

This notebook re-runs every test behind the Phase 1 report (`Phase1_Confound_Control.docx`) step by step:
where the data come from, how each statistical test works, how each confound control is applied, and how the
tables and figures are produced. Every number in the report can be traced to a cell here.

**Question.** Do the manuscript's group findings survive control of acquisition protocol, demographics, scanner
vendor, medial temporal atrophy, white-matter hyperintensities and amyloid status?

**How to run.** Kernel: *IIT Phase 1 (confound control)*. A full run takes about 7 minutes; most of it is the
subject-level bootstrap in Section 6.

**Second version (24 Oct, evening).** Every condition except C0 now runs on quality-controlled data (Section 2.3:
11 participants with failed processing excluded, physically impossible single values set to missing), and three
processing checks were added (Section 2.4 and 4.6: C9 processing route, C10 microstructure coverage, C11 coverage at
least 25%). The first-version results are kept in `phase1/archive_v1_firstreport/` and compared in Figure P1-3.

**Data protection.** The notebook reads ADNI data from local paths only (ADNI Data Use Agreement). The folder
`docs/iit_dipanjan/` is excluded from git; never copy outputs into the public repository.
""")

code(r"""
import glob, json, os, re, sys, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
import scipy, statsmodels, sklearn, matplotlib
from scipy import stats
from scipy.stats import rankdata
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from sklearn.linear_model import LogisticRegression
from neuroCombat import neuroCombat
import pyreadr
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)
T0 = time.time()

EXP = Path("/home/ec2-user/exp")
A = Path(os.environ.get("IIT_AROOT", "/home/ec2-user/exp/data/derivatives/qc/analysis_cohort"))           # analysis tree
CONN = Path(os.environ.get("IIT_CONN", "/data/derivatives/connectomes"))               # 166 x 166 edge matrices
FS = Path(os.environ.get("IIT_FASTSURFER", "/data/derivatives/fastsurfer"))   # FastSurfer segmentations
AM2 = EXP / "cohort/adnimerge/extracted/ADNIMERGE2/data"   # ADNIMERGE2 tables (.rda)
OUT = Path(os.environ.get("IIT_P1", "/home/ec2-user/exp/docs/iit_dipanjan/phase1"))
B_BOOT = int(os.environ.get("IIT_B", 2000))                # bootstrap resamples (2000 as in the manuscript)
PUBLISHED = str(CONN) == "/data/derivatives/connectomes"   # the published matrices (validation checks apply)
print("data:", CONN, "| analysis root:", A, "| outputs:", OUT)
rng = np.random.default_rng(20261024)
for lib in [np, pd, scipy, statsmodels, sklearn, matplotlib, pyreadr]:
    print(f"{lib.__name__:12} {lib.__version__}")
print("python", sys.version.split()[0])
""")

md(r"""
## 1. Participants and covariates
One row per participant (n = 530). Sources:
* **Demographics, diagnosis, phase** — the published master cohort.
* **Scan date and scanner vendor** — from the image ID of the session actually analysed (taken from the connectome
  file names; the master cohort's own image ID is wrong for 32 AD participants).
* **Education** — ADNIMERGE2 `PTDEMOG`.
* **Medial temporal volume** — FastSurfer: hippocampus + amygdala + entorhinal + parahippocampal cortex, as a
  proportion of estimated intracranial volume (eTIV).
* **White-matter hyperintensities** — ADNIMERGE2 `UCD_WMH` (FLAIR), visit nearest the diffusion scan within 180 days.
* **Amyloid** — ADNIMERGE2 `UCBERKELEY_AMY_6MM` (status and centiloids), nearest PET within 180 days.
""")

code(r"""
def am2(t):
    r = pyreadr.read_r(str(AM2 / f"{t}.rda")); return r[list(r)[0]]

m = pd.read_csv(A / "00_master/master_cohort.csv", low_memory=False)
cov = m[["subject_id", "group", "age", "sex", "phase"]].copy()
cov["female"] = cov.sex.astype(str).str.upper().str[0].eq("F").astype(int)
cov["RID"] = cov.subject_id.str[-4:].astype(int)
img = {re.search(r"SC_AAL166_(\d{3}_S_\d{4})_I(\d+)_count", f).group(1): re.search(r"_I(\d+)_count", f).group(1)
       for f in glob.glob(str(CONN / "SC_AAL166_*_count.csv"))}
cov["image_id"] = cov.subject_id.map(img)
dti = pd.read_csv(EXP / "cohort/dti_master.csv", low_memory=False)
dti["Image ID"] = dti["Image ID"].astype(str).str.replace(r"\.0$", "", regex=True)
dti = dti.drop_duplicates("Image ID").set_index("Image ID")
cov["scan_date"] = pd.to_datetime(cov.image_id.map(dti["Study Date"]), errors="coerce")
cov["vendor"] = (cov.image_id.map(dti["Imaging Protocol"]).astype(str).str.extract(r"Manufacturer=([A-Za-z]+)")[0]
                 .str.upper().map({"SIEMENS": "Siemens", "PHILIPS": "Philips", "GE": "GE"}).fillna("unrecorded"))
cov["phase"] = cov.phase.astype(str).str.replace("ADNI ", "ADNI-")
cov["adni34"] = cov.phase.isin(["ADNI-3", "ADNI-4"]).astype(int)
sys.path.insert(0, "/home/ec2-user/exp/docs/iit_dipanjan/tools")
import iit_chain as chain
chain.boot(); PL = chain.sample_label()
cov["adni34"] = chain.primary(cov)                         # the primary-sample flag (IIT_PRIMARY_SAMPLE; default ADNI-3/4)

ed = am2("PTDEMOG")[["RID", "PTEDUCAT"]]; ed = ed[pd.to_numeric(ed.PTEDUCAT, errors="coerce") > 0]
cov["education"] = cov.RID.map(ed.groupby("RID").PTEDUCAT.median())

def aseg(path):
    v, etiv = {}, np.nan
    for line in open(path):
        if line.startswith("# Measure EstimatedTotalIntraCranialVol"): etiv = float(line.split(",")[-2])
        elif not line.startswith("#"):
            p = line.split()
            if len(p) >= 5: v[p[4]] = float(p[3])
    return etiv, v
def dkt(path):
    return {p[0]: float(p[3]) for p in (l.split() for l in open(path) if not l.startswith("#")) if p and p[0] in ("entorhinal", "parahippocampal")}
vols = []
for sid, iid in zip(cov.subject_id, cov.image_id):
    cands = [c for c in sorted(FS.glob(f"{sid}_I*")) if (c / "stats/aseg.stats").is_file()]
    if not cands: vols.append({}); continue
    d = ([c for c in cands if c.name == f"{sid}_I{iid}"] or cands)[0]
    etiv, v = aseg(d / "stats/aseg.stats")
    lh, rh = dkt(d / "stats/lh.aparc.DKTatlas.mapped.stats"), dkt(d / "stats/rh.aparc.DKTatlas.mapped.stats")
    hip = v["Left-Hippocampus"] + v["Right-Hippocampus"]; amy = v["Left-Amygdala"] + v["Right-Amygdala"]
    vols.append(dict(etiv=etiv, mtl=hip + amy + sum(lh[k] + rh[k] for k in lh)))
cov = pd.concat([cov, pd.DataFrame(vols)], axis=1)
cov["mtl_ratio"] = cov.mtl / cov.etiv * 1e3

def nearest(t, col, dcol, nonneg=True, window=180):
    df = am2(t)[["RID", col, dcol]].copy(); df[col] = pd.to_numeric(df[col], errors="coerce")
    if nonneg: df.loc[df[col] < 0, col] = np.nan
    df[dcol] = pd.to_datetime(df[dcol], errors="coerce"); df = df.dropna()
    x = cov[["subject_id", "RID", "scan_date"]].merge(df, on="RID", how="left")
    x["lag"] = (x[dcol] - x.scan_date).dt.days.abs()
    x = x.sort_values("lag").drop_duplicates("subject_id").set_index("subject_id")
    return cov.subject_id.map(x[col].where(x.lag <= window))
cov["wmh_cm3"] = nearest("UCD_WMH", "TOTAL_WMH", "EXAMDATE")
cov["amyloid_pos"] = nearest("UCBERKELEY_AMY_6MM", "AMYLOID_STATUS", "SCANDATE")
cov["wmh_log"] = np.log(cov.wmh_cm3 / (cov.etiv / 1e3) * 100 + 0.01)
cov = cov.set_index("subject_id")
if chain.root() is not None:      # a chain's covariates are one file: the engines and this notebook read the same values
    _c = pd.read_csv(OUT / "covariates.csv").set_index("subject_id").loc[cov.index]
    cov = _c[cov.columns.intersection(_c.columns)].copy(); cov["adni34"] = chain.primary(_c)

print("Participants by phase, vendor and diagnosis")
display(pd.crosstab([cov.phase, cov.vendor], cov.group))
print("Covariates by diagnosis (mean; n available)")
display(cov.groupby("group")[["age", "female", "education", "mtl_ratio", "wmh_cm3"]].agg(["mean", "count"]).round(2).T)
""")

md(r"""
## 2. Connectome features
The published per-participant network values are used unchanged:
* **WB** — whole-brain network FA, MD, RD and AxD (40 values per participant);
* **TR** — the same measures on short-range (≤ CN Q1 = 78.7 mm) and long-range (> CN Q3 = 171.0 mm) edges (80);
* **EX** — the same measures on short- and long-range EDR-exception edges (up to 80);
* **coverage** of the 51-edge consensus core (structural, so all 530 participants).

The cell after this one recomputes WB and TR values from the raw edge matrices with the published recipe (node
mean over valid edges, then edge-count-weighted network mean) and checks it against the published values. The same
recipe restricted to *consistent* edges (present in ≥ 90% of ADNI-3/4 participants) gives condition C6.
""")

code(r"""
M = {"fa_mean": "FA", "md_mean": "MD", "rd_mean": "RD", "ad_mean": "AxD"}
wb = pd.read_csv(A / "19_network_analysis/functional/network_microstructure_subject.csv"); wb["feat"] = "WB|" + wb.network + "|" + wb.metric.map(M)
tr = pd.read_csv(A / "19_network_analysis/functional/network_range_restricted_subject.csv"); tr = tr[tr.measure.str.contains("FA|MD|RD|AxD")]; tr["feat"] = "TR|" + tr.network + "|" + tr.measure
ex = pd.read_csv(A / "19_network_analysis/functional/network_exception_measures_subject.csv"); ex = ex[ex.measure.str.contains("FA|MD|RD|AxD")]; ex["feat"] = "EX|" + ex.network + "|" + ex.measure
pub = pd.concat([d[["subject_id", "feat", "value"]] for d in (wb, tr, ex)]).pivot_table(index="subject_id", columns="feat", values="value").reindex(cov.index)
e = pd.read_csv(A / "17_edr_exceptions/edr_exception_edge_level_fd_sum_len_mean.csv", usecols=["subject_id", "i", "j", "exception_flag"]); e = e[e.exception_flag == 1]
freq = e.groupby(["i", "j"]).subject_id.nunique() / len(cov); core = set(freq[freq > 0.5].index)
pub["COV|core|coverage"] = pub.index.map(e[e.set_index(["i", "j"]).index.isin(core)].groupby("subject_id").size() / len(core)).fillna(0.0)
print("feature table:", pub.shape, "| core edges:", len(core))
pub.filter(like="Limbic").iloc[:3, :6]
""")

code(r"""
mp = pd.read_csv(A / "19_network_analysis/network_mapping_used.csv"); idx2net = dict(zip(mp.matrix_idx.astype(int), mp.functional_network))
NETS = ["Visual", "Somatomotor", "DorsalAttention", "Salience_VAN", "Limbic", "Frontoparietal", "DMN", "Subcortical", "Cerebellar", "Brainstem"]
th = pd.read_csv(A / "17_edr_exceptions/edr_exception_thresholds.csv").iloc[0]; SR_MAX, LR_MIN = float(th.short_max_mm), float(th.long_min_mm)
net_of = np.array([idx2net.get(k + 1) for k in range(166)], dtype=object)
fsid = {}
for sid, iid in zip(m.subject_id, m["Image ID"].astype(str).str.replace(r"\.0$", "", regex=True)):
    f = CONN / f"SC_AAL166_{sid}_I{iid}_len_mean.csv"
    fsid[sid] = f"{sid}_I{iid}" if f.is_file() else sorted(CONN.glob(f"SC_AAL166_{sid}_I*_len_mean.csv"))[0].name[10:-13]

def networks(L, W, micro, keep=None):
    out = {}; base = (L > 0) & (W > 0); np.fill_diagonal(base, False)
    for tag, rmask in {"WB": None, "SR": base & (L <= SR_MAX), "LR": base & (L > LR_MIN)}.items():
        for key, lbl in M.items():
            arr = micro[key]; v = np.isfinite(arr) & (arr > 0)
            if tag == "WB": np.fill_diagonal(v, False)
            else: v &= rmask
            if keep is not None: v &= keep
            cnt = v.sum(1).astype(float); node = np.where(cnt > 0, np.where(v, arr, 0).sum(1) / np.maximum(cnt, 1), np.nan)
            for net in NETS:
                sel = (net_of == net) & np.isfinite(node); w = cnt[sel]
                out[f"WB|{net}|{lbl}" if tag == "WB" else f"TR|{net}|{tag}-{lbl}"] = (node[sel] * w).sum() / w.sum() if w.sum() else np.nan
    return out

mats, pres, n34 = {}, np.zeros((166, 166)), 0
for sid in cov.index:
    rd = lambda k: np.loadtxt(CONN / f"SC_AAL166_{fsid[sid]}_{k}.csv", delimiter=",")
    L, W = rd("len_mean"), rd("fd_sum")
    micro = {k: rd(k) for k in M} if (CONN / f"SC_AAL166_{fsid[sid]}_fa_mean.csv").is_file() else None
    mats[sid] = (L, W, micro)
    if cov.loc[sid, "adni34"] == 1: pres += (L > 0) & (W > 0); n34 += 1
keep = pres / n34 >= 0.90; np.fill_diagonal(keep, False)
rep = pd.DataFrame({s: networks(*mats[s]) for s in cov.index if mats[s][2] is not None}).T
con = pd.DataFrame({s: networks(*mats[s], keep=keep) for s in cov.index if mats[s][2] is not None}).T
diff = (rep - pub.loc[rep.index, rep.columns]).abs()
print(f"recomputed vs published: max |difference| = {np.nanmax(diff.values):.1e} over {diff.notna().sum().sum()} values")
print(f"consistent edges: {int(keep.sum() // 2)} of {int((pres > 0).sum() // 2)} ever reconstructed")
""")

md(r"""
### 2.3 Data audit and quality control
**How this came about.** While writing up the ComBat step (Section 4.2) the harmonised values looked wrong, and the
cause was a few physically impossible diffusion values. Every participant was then audited (script `p1_audit.py`;
the same rules are re-run here and checked against its output):

* **diffusion values** against physical limits (nothing in tissue diffuses faster than free water, about
  3.0 × 10⁻³ mm²/s at 37 °C) and against identities a diffusion tensor must satisfy — FA ≤ 1, AxD ≥ RD and
  MD = (AxD + 2 RD) / 3;
* **the structural connectome**: edge density and regions without any connection;
* **the covariates**: age, education, intracranial volume, medial temporal volume, hyperintensity volume, amyloid
  status against centiloids (all clean).

**Rules.** A participant is excluded when **R1** the median of their ten whole-brain network MD values lies above
Q3 + 3 × IQR of the cohort (Tukey's "far-out" fence), **R2** two or more whole-brain networks break the tensor
identities, or **R3** edge density lies below Q1 − 3 × IQR (and below 0.6). **R0** marks participants outside the
analysis pipeline's cohort (density below 0.6, or in the run's exclusions file), and for the uniformly rebuilt
connectomes **R4** marks a failed atlas registration (the participant's FA map correlates below 0.4 with the
FMRIB58 template). Separately, any single network value whose
FA/MD/AxD/RD quartet breaks a limit (FA > 1 or < 0.05, AxD < RD, a diffusivity above free water) is set to missing.
The rules were written after the problem was seen; they are physical and mathematical limits, not tuned to results.
""")

code(r"""
FREE_WATER = 3.0e-3
def far_out(x):
    q1, q3 = np.nanpercentile(x, [25, 75]); return q1 - 3 * (q3 - q1), q3 + 3 * (q3 - q1)
def tensor_bad(X, stem, tol):
    FA, MD, AD, RD = (X.get(stem + k) for k in ("FA", "MD", "AxD", "RD"))
    bad = (FA > 1) | (AD < RD) | (MD > FREE_WATER) | (AD > FREE_WATER) | (RD > FREE_WATER)
    if tol is not None: bad |= (MD - (AD + 2 * RD) / 3).abs() / MD > tol
    return bad.fillna(False)
import sc_cohort_settings as cohort_settings
K = cohort_settings.n_nodes(); FLOOR = cohort_settings.density_floor(); iu = np.triu_indices(K, 1)
qc = pd.DataFrame(index=cov.index)
qc["wb_md_median"] = pub.filter(regex=r"^WB\|.*\|MD$").median(axis=1) * 1e3
qc["n_wb_tensor_fail"] = sum(tensor_bad(pub, f"WB|{n}|", 0.01).astype(int) for n in NETS)
pres = {s_: (mats[s_][0] > 0) & (mats[s_][1] > 0) if cohort_settings.get("length_measure") == "per_person" else mats[s_][1] > 0 for s_ in cov.index}
qc["density"] = [pres[s_][iu].mean() for s_ in cov.index]
qc["zero_nodes"] = [int((pres[s_][:K, :K].sum(1) == 0).sum()) for s_ in cov.index]
lo_md, hi_md = far_out(qc.wb_md_median); lo_den, _ = far_out(qc.density); lo_den = min(lo_den, FLOOR)
qc["R0"] = ~cov.index.isin(m.subject_id)          # outside the analysis tree's cohort (pipeline density floor 0.6)
qc["R1"], qc["R2"], qc["R3"] = qc.wb_md_median > hi_md, qc.n_wb_tensor_fail >= 2, qc.density < lo_den
reg = {s_: json.loads(q.read_text()).get(chain.r4_field()) for s_ in cov.index if chain.r4_field() and (q := CONN / f"SC_AAL166_{fsid[s_]}_qc.json").is_file()}
qc["R4"] = cov.index.map(reg).astype(float) < chain.r4_min()     # rebuilt data only: atlas registration failed
qc["qc_fail"] = (qc.R0 | qc.R1 | qc.R2 | qc.R3 | qc.R4).astype(int)
print(f"R1 fence: median whole-brain MD > {hi_md:.3f} × 10⁻³ mm²/s   R3 fence: density < {lo_den:.3f}")
print("whole-brain networks breaking the tensor identities, participants per count:", qc.n_wb_tensor_fail.value_counts().sort_index().to_dict())
display(cov[["group", "phase", "vendor"]].join(qc).loc[qc.qc_fail == 1, ["group", "phase", "vendor", "wb_md_median", "n_wb_tensor_fail", "density", "zero_nodes", "R0", "R1", "R2", "R3", "R4"]]
        .sort_values("wb_md_median", ascending=False).round(3))
print("excluded by group:", cov[qc.qc_fail == 1].group.value_counts().to_dict(), "| of them in ADNI-3/4:", cov[(qc.qc_fail == 1) & (cov.adni34 == 1)].group.value_counts().to_dict())

def mask_values(X):
    X = X.copy()
    for st in sorted({c[:-2] for c in X.columns if c.endswith("FA")}):
        quart = [st + k for k in ("FA", "MD", "AxD", "RD") if st + k in X]
        if len(quart) < 4: continue
        bad = tensor_bad(X, st, None) | (X[st + "FA"] < 0.05); X.loc[bad, quart] = np.nan
    return X
pubq, conq = mask_values(pub), mask_values(con)
keep = qc.qc_fail == 0
print(f"single values set to missing among the {int(keep.sum())} participants kept: {int(pubq[keep].isna().sum().sum() - pub[keep].isna().sum().sum())}")

ref_qc = pd.read_csv(OUT / "qc.csv", index_col=0).reindex(cov.index)
ref_fq = pd.read_csv(OUT / "features_qc.csv", index_col=0).reindex(index=cov.index, columns=pub.columns)
assert (ref_qc.qc_fail.values == qc.qc_fail.values).all() and np.allclose(ref_fq.values, pubq.values, equal_nan=True, rtol=1e-9)
print("matches p1_audit.py (qc.csv, features_qc.csv)")

fig, ax = plt.subplots(figsize=(11, 3.2))
for grp, col in [("CN", "#38bdf8"), ("MCI", "#84cc16"), ("AD", "#fb7185")]:
    v = qc.loc[cov.group == grp, "wb_md_median"]; ax.scatter(v, np.random.default_rng(1).uniform(-0.3, 0.3, len(v)) + ["CN", "MCI", "AD"].index(grp), s=9, color=col, alpha=0.7)
ax.axvline(hi_md, color="#dc2626", lw=1.2); ax.text(hi_md, 2.45, f"  R1 fence {hi_md:.2f}", color="#dc2626", fontsize=8, va="top")
ax.axvline(3.0, color="#6b7280", ls=":", lw=1); ax.text(3.0, -0.45, "  free water, 37 °C", color="#6b7280", fontsize=8)
ax.set_xscale("log"); ax.set_yticks([0, 1, 2]); ax.set_yticklabels(["CN", "MCI", "AD"])
ax.set_xlabel("participant median of the ten whole-brain network MD values (×10⁻³ mm²/s, log scale)")
ax.set_title("One dot per participant: the eight right of the red line meet R1; R2 and R3 add three more", fontsize=9)
plt.tight_layout(); plt.savefig(OUT / "fig_nb_qc_md.png", dpi=200); plt.show()
""")

md(r"""
**Reading this.** Eight participants meet R1 and R2 together — their diffusivities are two to ten times normal. Two
more meet R2 only: one has FA above 1 in every network (impossible: FA is bounded by 1), the other MD of
0.03–0.39 × 10⁻³ with FA of 0.73 everywhere. One meets R3: 81 of its 166 regions have no connection at all. Six of
the eleven are in the ADNI-3/4 primary sample.
""")

md(r"""
### 2.4 Microstructure coverage and processing route
The published connectomes come from three processing routes (Supplementary Table S24). The audit found that in the
current route the diffusion values (FA, MD, AxD, RD) were sampled on only part of each participant's connectome:
many edges that exist structurally carry no diffusion value. **Microstructure coverage** = the share of a
participant's connected edges that carry a diffusion value. The cell computes it from the edge matrices, plots it
by group, and shows that it predicts the values themselves.
""")

code(r"""
qc["md_coverage"] = [((mats[s_][2]["md_mean"][iu] > 0) & pres[s_][iu]).sum() / max(pres[s_][iu].sum(), 1) if mats[s_][2] is not None else np.nan
                     for s_ in cov.index]
fin = pd.read_csv(EXP / "research_audit/route_attribution_20260919/final.csv")
fin = fin.assign(subject_id=fin.sid.str.extract(r"^(\d{3}_S_\d{4})")[0]).drop_duplicates("subject_id").set_index("subject_id")
qc["route"] = fin.s4.reindex(cov.index).str[0].map({"A": "legacy_step7", "B": "legacy_r4", "C": "new", "D": "new", "E": "new", "F": "new"})
assert np.allclose(qc.md_coverage, ref_qc.md_coverage, equal_nan=True) and (qc.route == ref_qc.route).all()
for c in ["qc_fail", "md_coverage", "route"]: cov[c] = qc[c]
cov["route_r4"], cov["route_step7"] = (cov.route == "legacy_r4").astype(float), (cov.route == "legacy_step7").astype(float)
cov["log_coverage"] = np.log(cov.md_coverage + 0.01)
a34q = cov[(cov.adni34 == 1) & (cov.qc_fail == 0)].copy()            # the primary sample
wbmd, wbfa = qc.wb_md_median, pub.filter(regex=r"^WB\|.*\|FA$").median(axis=1)

print("primary sample: microstructure coverage and processing route by group")
display(a34q.groupby("group").agg(n=("md_coverage", "size"), coverage_median=("md_coverage", "median"),
                                  below_5pct=("md_coverage", lambda v: (v < 0.05).mean()), below_25pct=("md_coverage", lambda v: (v < 0.25).mean()),
                                  current_route=("route", lambda v: (v == "new").mean())).reindex(["CN", "MCI", "AD"]).round(3))
print("median whole-brain FA and MD by route (primary sample):")
display(pd.DataFrame({"FA": wbfa[a34q.index].groupby(a34q.route).median(), "MD (×10⁻³)": wbmd[a34q.index].groupby(a34q.route).median(),
                      "n": a34q.route.value_counts()}).round(3))

nw = a34q[a34q.route == "new"]
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
ax = axes[0]
for i, grp in enumerate(["CN", "MCI", "AD"]):
    v = a34q.loc[a34q.group == grp, "md_coverage"].dropna() * 100
    ax.boxplot([v], positions=[i], widths=0.5, showfliers=False, patch_artist=True, boxprops=dict(facecolor={"CN": "#38bdf8", "MCI": "#84cc16", "AD": "#fb7185"}[grp], alpha=0.5))
    ax.scatter(np.random.default_rng(2).uniform(-0.18, 0.18, len(v)) + i, v, s=7, color="#374151", alpha=0.5)
ax.set_xticks([0, 1, 2]); ax.set_xticklabels([f"{g}\nn = {(a34q.group == g).sum()}" for g in ["CN", "MCI", "AD"]])
ax.set_ylabel("connected edges carrying diffusion values (%)"); ax.set_title("Microstructure coverage by group (primary sample)", fontsize=9.5)
for ax, val, nice in [(axes[1], wbmd, "median whole-brain MD (×10⁻³ mm²/s)"), (axes[2], wbfa, "median whole-brain FA")]:
    for grp, col in [("CN", "#38bdf8"), ("MCI", "#84cc16"), ("AD", "#fb7185")]:
        d_ = nw[nw.group == grp]; ax.scatter(d_.md_coverage * 100, val[d_.index], s=10, alpha=0.6, color=col, label=grp)
    r_ = {g: stats.spearmanr(nw[nw.group == g].md_coverage, val[nw[nw.group == g].index], nan_policy="omit").statistic for g in ["CN", "MCI"]}
    ax.set_xlabel("microstructure coverage (%)"); ax.set_ylabel(nice)
    ax.set_title(f"Current route: {nice.split(' (')[0].replace('median whole-brain ', 'whole-brain ')} against coverage\n"
                 f"Spearman ρ: CN {r_['CN']:+.2f}, MCI {r_['MCI']:+.2f}", fontsize=9.5)
axes[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig(OUT / "fig_p1_coverage.png", dpi=300); plt.show()
""")

md(r"""
**Reading this.** Left: AD participants have much lower coverage (median about a fifth of their edges, against
about half for CN and MCI); a third of AD participants have diffusion values on fewer than 5% of their edges.
Middle and right: within the current route, low coverage goes with high MD and low FA, in CN and MCI alike — no
disease is involved there, so this is a property of the processing. Because AD participants have lower coverage,
part of their higher diffusivity and lower FA could come from processing rather than disease. Section 4.6 explains
the three conditions that test this (C9–C11).
""")

md(r"""
## 3. The statistical tests, step by step
Example used throughout: **limbic axial diffusivity (whole brain), CN versus MCI**, pooled sample.

### 3.1 Cliff's δ
δ(a, b) = P(a > b) − P(a < b): the probability that a randomly chosen CN value exceeds a randomly chosen MCI value,
minus the reverse. It runs from −1 to 1; **negative means higher in the later-stage group** (manuscript convention).
It is computed here from ranks (ties count half), which is equivalent to counting all pairs.
""")

code(r"""
def cliff(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float); a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if len(a) < 3 or len(b) < 3: return np.nan
    r = rankdata(np.concatenate([a, b])); u = r[:len(a)].sum() - len(a) * (len(a) + 1) / 2
    return 2 * u / (len(a) * len(b)) - 1

x = pub["WB|Limbic|AxD"]; cn, mci = x[cov.group == "CN"].dropna().values, x[cov.group == "MCI"].dropna().values
brute = ((cn[:, None] > mci[None, :]).sum() - (cn[:, None] < mci[None, :]).sum()) / (len(cn) * len(mci))
print(f"Cliff's δ (ranks)       = {cliff(cn, mci):+.6f}")
print(f"Cliff's δ (all {len(cn)*len(mci):,} pairs) = {brute:+.6f}")
print(f"median AxD: CN {np.median(cn):.3e}, MCI {np.median(mci):.3e}  -> negative δ = higher in MCI")
""")

md(r"""
### 3.2 Brunner–Munzel test
A rank-based test of P(a > b) = 0.5 that, unlike the Mann–Whitney test, does not assume equal variances or shapes
in the two groups — appropriate here because group sizes and spreads differ (CN 251 vs AD 78).
""")

code(r"""
bm = stats.brunnermunzel(cn, mci, alternative="two-sided")
mw = stats.mannwhitneyu(cn, mci, alternative="two-sided")
print(f"Brunner–Munzel: statistic {bm.statistic:.3f}, p = {bm.pvalue:.2e}")
print(f"(Mann–Whitney for comparison: p = {mw.pvalue:.2e})")
""")

md(r"""
### 3.3 Benjamini–Hochberg false discovery rate
The same families as the manuscript: whole-brain = the three contrasts of each network × measure; tract-length and
exception = the ten networks of each measure within a contrast; coverage = its three contrasts. *q* is the corrected
value. The implementation below is checked against `statsmodels`.
""")

code(r"""
def bh(p):
    p = np.asarray(p, float); q = np.full_like(p, np.nan); ok = np.isfinite(p)
    if ok.sum():
        pv = p[ok]; o = np.argsort(pv); n = len(pv); r = pv[o] * n / np.arange(1, n + 1)
        r = np.minimum.accumulate(r[::-1])[::-1]; out = np.empty(n); out[o] = np.minimum(r, 1); q[ok] = out
    return q
demo = np.array([0.001, 0.02, 0.03, 0.2, 0.04, np.nan])
print("ours       ", np.round(bh(demo), 4))
print("statsmodels", np.round(multipletests(demo[~np.isnan(demo)], method="fdr_bh")[1], 4))
""")

md(r"""
### 3.4 Inclusion rule
As in the manuscript, a network measure is analysed only if **every** group has at least five values; sparse
exception cells (for example Brainstem exception edges in AD) are dropped.
""")

code(r"""
CONTRASTS = [("CN", "MCI"), ("CN", "AD"), ("MCI", "AD")]
def eligible(X, g):
    return [c for c in X.columns if min(X.loc[g == k, c].notna().sum() for k in ("CN", "MCI", "AD")) >= 5]
print("eligible features (pooled):", len(eligible(pub, cov.group)), "of", pub.shape[1])

def effects(X, g, tag):
    rows = []
    for a, b in CONTRASTS:
        for c in X.columns:
            xa, xb = X.loc[g == a, c].values.astype(float), X.loc[g == b, c].values.astype(float)
            fa, fb = xa[np.isfinite(xa)], xb[np.isfinite(xb)]
            p = stats.brunnermunzel(fa, fb).pvalue if min(len(fa), len(fb)) >= 3 else np.nan
            rows.append(dict(condition=tag, feature=c, contrast=f"{a}-{b}", delta=cliff(xa, xb), p=p, n_first=len(fa), n_second=len(fb)))
    df = pd.DataFrame(rows); df[["kind", "network", "measure"]] = df.feature.str.split("|", expand=True); df["q"] = np.nan
    for _, idx in df[df.kind == "WB"].groupby("feature").groups.items(): df.loc[idx, "q"] = bh(df.loc[idx, "p"])
    for _, idx in df[df.kind.isin(["TR", "EX"])].groupby(["kind", "measure", "contrast"]).groups.items(): df.loc[idx, "q"] = bh(df.loc[idx, "p"])
    cv = df.kind == "COV"; df.loc[cv, "q"] = bh(df.loc[cv, "p"])
    return df
""")

md(r"""
## 4. The confound controls, one at a time
Two kinds of control are used, and it matters which is which:

* **Statistical methods** keep every participant and remove an unwanted source of variation from the *values*
  (regression adjustment, ComBat harmonisation).
* **Sample restrictions** remove *participants* so that the remaining groups no longer differ on the confound
  (protocol restriction, single-vendor restriction, matching, amyloid-defined groups).

| Condition | Control | Type | What it does | Participants removed |
|---|---|---|---|---|
| C1 | ADNI-3/4 only | sample restriction | drops the ADNI-2 protocol, which is almost all AD | 37 ADNI-2 scans |
| C2 | age, sex, education | statistical method | regression: removes the covariate part of each value | none beyond C1 |
| C3 | vendor ComBat | statistical method | removes each vendor's mean and spread shift per measure | none beyond C1 |
| C4 | propensity matching | sample restriction | keeps CN/MCI participants who resemble the comparison group | unmatched participants |
| C5 | MTL volume + WMH | statistical method | regression on atrophy and lesion burden | 20 without both measures |
| C6 | consistent edges | different measurement | recomputes network values from edges present in ≥ 90% | none beyond C1 |
| C7 | amyloid groups | sample restriction | amyloid-negative CN vs amyloid-positive MCI/AD | those without PET or with the other status |
| C8 | Siemens only | sample restriction | removes vendor differences by using one vendor | Philips and GE scans |
| all but C0 | quality control (§2.3) | sample restriction + value masking | removes failed processing; sets impossible single values to missing | 11 (6 in ADNI-3/4) |
| C9 | processing route | statistical method | regression on route (two legacy routes vs current) | none beyond C2 |
| C10 | microstructure coverage | statistical method | regression on log coverage (diffusion features only) | 1 AD without diffusion matrices |
| C11 | coverage ≥ 25% | sample restriction | keeps participants with diffusion values on at least a quarter of their edges | 140 |

The next cell prints the exact numbers remaining under each.
""")

code(r"""
rem = []
A_ = (cov.adni34 == 1) & (cov.qc_fail == 0)
for lab, idx in [("C0q pooled after QC", cov.index[cov.qc_fail == 0]), (f"C1 {PL} after QC", cov.index[A_]),
                 ("C5 with MTL + WMH", cov.index[A_ & cov.mtl_ratio.notna() & cov.wmh_log.notna()]),
                 ("C7 amyloid-defined", cov.index[A_ & (((cov.group == "CN") & (cov.amyloid_pos == 0)) | (cov.group.isin(["MCI", "AD"]) & (cov.amyloid_pos == 1)))]),
                 ("C8 Siemens only", cov.index[A_ & (cov.vendor == "Siemens")]),
                 ("C11 coverage >= 25%", cov.index[A_ & (cov.md_coverage >= 0.25)])]:
    rem.append({"condition": lab, **cov.loc[idx].group.value_counts().reindex(["CN", "MCI", "AD"]).to_dict(), "total": len(idx)})
rem.insert(0, {"condition": "C0 pooled (published)", **cov.group.value_counts().reindex(["CN", "MCI", "AD"]).to_dict(), "total": len(cov)})
pd.DataFrame(rem)
""")

md(r"""
### 4.1 Covariate adjustment (conditions C2, C5–C8)
Each measure is fitted on diagnosis **and** the covariates; only the covariate part is subtracted. This removes the
influence of age, sex and education while leaving the group difference intact (fitting on covariates alone would
also remove part of the group effect, because age and sex differ between groups). The regression for the example
feature is shown first.
""")

code(r"""
a34 = cov[cov.adni34 == 1].copy()                                  # before quality control (used in 4.2 only)
d = a34q.join(pubq[["WB|Limbic|AxD"]].rename(columns={"WB|Limbic|AxD": "limbic_axd"})).dropna(subset=["limbic_axd"])
d["limbic_axd_e3"] = d.limbic_axd * 1e3
fit = smf.ols("limbic_axd_e3 ~ C(group, Treatment('CN')) + age + female + education", data=d).fit()
print(fit.summary().tables[1])

def adjust(X, g, Z):
    Zc = Z - Z.mean(); G = pd.get_dummies(g, drop_first=True).astype(float); out = X.copy()
    for c in X.columns:
        ok = X[c].notna() & Zc.notna().all(1)
        if ok.sum() < 20: out[c] = np.nan; continue
        D = np.column_stack([np.ones(ok.sum()), G[ok].values, Zc[ok].values])
        beta = np.linalg.lstsq(D, X.loc[ok, c].values, rcond=None)[0]
        out[c] = np.where(ok, X[c].values - Zc.fillna(0).values @ beta[1 + G.shape[1]:], np.nan)
    return out
COVS = ["age", "female", "education"]
adj = adjust(pubq.loc[a34q.index, ["WB|Limbic|AxD"]], a34q.group, a34q[COVS])
g = a34q.group
print(f"\nprimary sample, limbic AxD CN–MCI: δ unadjusted {cliff(pubq.loc[a34q.index, 'WB|Limbic|AxD'][g=='CN'], pubq.loc[a34q.index, 'WB|Limbic|AxD'][g=='MCI']):+.3f}"
      f" -> adjusted {cliff(adj['WB|Limbic|AxD'][g=='CN'], adj['WB|Limbic|AxD'][g=='MCI']):+.3f}")
""")

md(r"""
### 4.2 Scanner-vendor harmonisation with ComBat (C3) and the Siemens-only check (C8)
**What ComBat is.** ComBat (Johnson et al. 2007; adapted to diffusion MRI by Fortin et al. 2017) is a statistical
harmonisation method. **No participant is removed.** For every measure *v* (for example limbic AxD) and every
participant *j* scanned on vendor *i*, it assumes

  value(*i,j,v*) = α(*v*) + X(*j*)·β(*v*) + γ(*i,v*) + δ(*i,v*)·ε(*i,j,v*)

where α is the overall mean, X·β the effects we want to **keep** (diagnosis, age, sex, education), γ an additive
vendor shift and δ a multiplicative vendor scaling of the residual spread.

**Steps.**
1. Fit the model and standardise each measure.
2. Estimate γ and δ for each vendor. Instead of trusting each measure's estimate alone, ComBat shrinks them towards
   the average over all measures (empirical Bayes) — this is what keeps small batches such as GE (15 participants)
   from being over-corrected.
3. Remove the vendor shift and scaling, then add back the mean and the protected effects:
   harmonised = (value − α − X·β − γ*) / δ* + α + X·β.

**Where it is applied.** Only within ADNI-3/4 (after quality control: Siemens 375, Philips 98, GE 14). It cannot be applied to the ADNI-2
scans, because they form a batch that is almost entirely AD: the batch effect and the disease effect cannot be
separated, and harmonising would remove the AD difference itself. That is why ADNI-2 is handled by restriction (C1).

**Implementation.** `neuroCombat` (the Fortin et al. Python implementation), parametric empirical Bayes, with
diagnosis and sex as categorical and age and education as continuous protected covariates. Missing values (sparse
exception cells) are filled with the measure's median for the fit and set back to missing afterwards.

**The Siemens-only check.** Because AD is unevenly spread across vendors (a third of GE participants against a tenth
overall), ComBat's model could also absorb part of the disease effect. C8 therefore repeats the primary analysis on
Siemens scans only — a sample restriction that needs no model.
""")

code(r"""
def combat(X, cv):
    batches = cv.vendor
    usable = [c for c in X.columns if all(X.loc[batches == b, c].notna().sum() >= 3 and X.loc[batches == b, c].std() > 0 for b in batches.unique())]
    Xi = X[usable].apply(lambda s: s.fillna(s.median()))
    res = neuroCombat(dat=Xi.T.values, covars=cv[["vendor", "group", "female", "age", "education"]],
                      batch_col="vendor", categorical_cols=["group", "female"], continuous_cols=["age", "education"])["data"].T
    H = X.copy(); H[usable] = np.where(X[usable].notna(), res, np.nan); return H, len(usable)
Xa34 = pub.loc[a34.index, eligible(pub.loc[a34.index], a34.group)]
H, nh = combat(Xa34, a34)
def vendor_table(X, H, cv, f="WB|Limbic|AxD"):
    by = [cv.vendor, cv.group]
    t = pd.DataFrame({"n": cv.groupby(["vendor", "group"]).size(),
                      "median before": X[f].groupby(by).median() * 1e3, "median after": H[f].groupby(by).median() * 1e3,
                      "mean before": X[f].groupby(by).mean() * 1e3, "mean after": H[f].groupby(by).mean() * 1e3,
                      "SD before": X[f].groupby(by).std() * 1e3, "SD after": H[f].groupby(by).std() * 1e3})
    return t.reindex(["Siemens", "Philips", "GE"], level=0).reindex(["CN", "MCI", "AD"], level=1).round(3)
print(f"ComBat harmonised {nh} of {Xa34.shape[1]} features (participants before = after = {len(H)})")
print("Limbic AxD (×10⁻³ mm²/s) by vendor and diagnosis, before and after ComBat:")
display(vendor_table(Xa34, H, a34))
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
for ax, data, title in [(axes[0], Xa34, "before ComBat"), (axes[1], H, "after ComBat")]:
    groups = [(v, grp) for v in ["Siemens", "Philips", "GE"] for grp in ["CN", "MCI", "AD"]]
    vals = [data.loc[(a34.vendor == v) & (a34.group == grp), "WB|Limbic|AxD"].dropna() * 1e3 for v, grp in groups]
    bp = ax.boxplot(vals, widths=0.6, patch_artist=True, showfliers=False)
    for patch, (v, grp) in zip(bp["boxes"], groups): patch.set_facecolor({"CN": "#38bdf8", "MCI": "#84cc16", "AD": "#fb7185"}[grp])
    ax.set_xticks(range(1, 10)); ax.set_xticklabels([f"{v}\n{grp}" for v, grp in groups], fontsize=7)
    ax.set_title(f"Limbic AxD by vendor and diagnosis — {title}", fontsize=9)
axes[0].set_ylabel("AxD (×10⁻³ mm²/s)"); plt.tight_layout(); plt.savefig(OUT / "fig_nb_combat_before_qc.png", dpi=200); plt.show()
""")

md(r"""
**Reading the plot and the table — this is where a data problem showed up.** If ComBat were only removing vendor
offsets, the boxes would move up or down but keep their height, and the CN < MCI < AD ordering inside each vendor
would stay. That is *not* what happens: after ComBat every vendor's spread widens (the SD of Siemens CN goes from
0.29 to 0.47, Philips CN from 0.10 to 0.31 × 10⁻³ mm²/s) and Philips AD moves up.

The table explains why. The GE CN *mean* (3.13 × 10⁻³) is twice its median (1.62) and its SD is 3.45 — one GE
participant has limbic AxD of 10.8 × 10⁻³ mm²/s, more than three times the diffusivity of free water at body
temperature (about 3.0 × 10⁻³). A Siemens CN participant at 4.6 also inflates the Siemens SD. ComBat estimates each vendor's shift and spread from **means and variances**,
so a single impossible value in a batch of 15 distorts the correction applied to every participant.

The rank-based statistics (Cliff's δ, Brunner–Munzel) hardly notice one extreme value, but the model-based steps
(ComBat, the least-squares covariate adjustment and the atrophy adjustment) do. This is what led to the audit and
quality control of Section 2.3. The cell below repeats ComBat on the quality-controlled primary sample: the Siemens
spread now stays as it was, Philips widens a little and the GE level is pulled towards the other two — which is what
harmonisation is supposed to do.
""")

code(r"""
Xq = pubq.loc[a34q.index, eligible(pubq.loc[a34q.index], a34q.group)]
Hq, _ = combat(Xq, a34q)
display(vendor_table(Xq, Hq, a34q))
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
for ax, data, title in [(axes[0], Xq, "before ComBat"), (axes[1], Hq, "after ComBat")]:
    groups = [(v, grp) for v in ["Siemens", "Philips", "GE"] for grp in ["CN", "MCI", "AD"]]
    vals = [data.loc[(a34q.vendor == v) & (a34q.group == grp), "WB|Limbic|AxD"].dropna() * 1e3 for v, grp in groups]
    bp = ax.boxplot(vals, widths=0.6, patch_artist=True, showfliers=False)
    for patch, (v, grp) in zip(bp["boxes"], groups): patch.set_facecolor({"CN": "#38bdf8", "MCI": "#84cc16", "AD": "#fb7185"}[grp])
    ax.set_xticks(range(1, 10)); ax.set_xticklabels([f"{v}\n{grp}" for v, grp in groups], fontsize=7)
    ax.set_title(f"Limbic AxD after quality control — {title}", fontsize=9)
axes[0].set_ylabel("AxD (×10⁻³ mm²/s)"); plt.tight_layout(); plt.savefig(OUT / "fig_nb_combat_after_qc.png", dpi=200); plt.show()
""")

md(r"""
### 4.3 Propensity matching (C4)
For each contrast, a logistic regression predicts group membership from age, sex, education and vendor; each
later-stage participant is matched to the nearest earlier-stage participant(s) on the logit, without replacement,
within 0.2 standard deviations (1:1 for CN–MCI, 2:1 for the AD contrasts). Balance is judged by standardised mean
differences (|SMD| < 0.1 is the usual target).
""")

code(r"""
def smd(d, a, b, c):
    x1, x0 = d.loc[d.group == b, c], d.loc[d.group == a, c]; return (x1.mean() - x0.mean()) / np.sqrt((x1.var() + x0.var()) / 2)
def match(cv, a, b, ratio):
    d = cv[cv.group.isin([a, b])].dropna(subset=COVS).copy()
    Xp = pd.concat([d[COVS], pd.get_dummies(d.vendor, drop_first=True).astype(float)], axis=1)
    d["logit"] = LogisticRegression(C=1e6, max_iter=5000).fit(Xp, (d.group == b).astype(int)).decision_function(Xp)
    cal = 0.2 * d.logit.std(); pool = d[d.group == a].copy(); ka, kb = [], []
    for sid, row in d[d.group == b].sample(frac=1, random_state=1).iterrows():
        dist = (pool.logit - row.logit).abs(); cand = dist[dist <= cal].nsmallest(ratio)
        if len(cand) == ratio: kb.append(sid); ka += list(cand.index); pool = pool.drop(cand.index)
    return d, d.loc[ka + kb]
bal, matched = [], {}
fig, axes = plt.subplots(1, 3, figsize=(12, 3))
for ax, ((a, b), ratio) in zip(axes, zip(CONTRASTS, [1, 2, 2])):
    full, mm = match(a34q, a, b, ratio); matched[(a, b)] = mm
    for c in COVS: bal.append(dict(contrast=f"{a}–{b}", covariate=c, SMD_before=smd(full, a, b, c), SMD_after=smd(mm, a, b, c), n_after=f"{(mm.group==a).sum()}/{(mm.group==b).sum()}"))
    for grp, col in [(a, "#2563eb"), (b, "#dc2626")]:
        ax.hist(full.loc[full.group == grp, "logit"], bins=25, alpha=0.25, color=col, density=True)
        ax.hist(mm.loc[mm.group == grp, "logit"], bins=25, histtype="step", color=col, density=True, lw=1.5, label=f"{grp} (matched)")
    ax.set_title(f"{a}–{b}: propensity (logit)", fontsize=9); ax.legend(fontsize=7)
plt.tight_layout(); plt.savefig(OUT / "fig_nb_propensity.png", dpi=200); plt.show()
pd.DataFrame(bal).round(3)
""")

md(r"""
### 4.4 Medial temporal atrophy and white-matter hyperintensities (C5)
C5 adds medial temporal volume and log FLAIR hyperintensity volume to the covariates. These are themselves markers of
disease, so this is a deliberately strict test: an effect that survives it is not explained by atrophy, lesions or
the partial-volume contamination they cause.

**This is the condition quality control changed most.** In the first version of this report, C5 removed nearly all
diffusivity differences. The second cell shows why: with the corrupt participants included, the fitted slope of
diffusivity on medial temporal volume is several times steeper, so subtracting "the atrophy part" also subtracted the
group difference. After quality control the slope is small and the group differences survive.
""")

code(r"""
d = a34q.join(pubq[["TR|Limbic|SR-AxD", "TR|Limbic|LR-MD"]]).dropna(subset=["mtl_ratio"])
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, f, nice in zip(axes, ["TR|Limbic|SR-AxD", "TR|Limbic|LR-MD"], ["limbic short-range AxD", "limbic long-range MD"]):
    for grp, col in [("CN", "#38bdf8"), ("MCI", "#84cc16"), ("AD", "#fb7185")]:
        s_ = d[d.group == grp]; ax.scatter(s_.mtl_ratio, s_[f] * 1e3, s=10, alpha=0.6, color=col, label=grp)
    r = stats.spearmanr(d.mtl_ratio, d[f], nan_policy="omit").statistic
    ax.set_xlabel("medial temporal volume (‰ of eTIV)"); ax.set_ylabel(f"{nice} (×10⁻³ mm²/s)"); ax.set_title(f"{nice}: Spearman ρ with MTL volume = {r:.2f}", fontsize=9)
axes[0].legend(fontsize=8); plt.tight_layout(); plt.savefig(OUT / "fig_nb_mtl_scatter.png", dpi=200); plt.show()
""")

code(r"""
C5COV = COVS + ["mtl_ratio", "wmh_log"]
rows = []
for lab, cv_, X in [("before quality control", a34, pub), ("after quality control", a34q, pubq)]:
    c_ = cv_.dropna(subset=C5COV); x = X.loc[c_.index, "WB|Limbic|AxD"] * 1e3; ok = x.notna()
    fit = smf.ols("y ~ C(group) + age + female + education + mtl_ratio + wmh_log", data=c_[ok].assign(y=x[ok])).fit()
    xa = adjust(X.loc[c_.index, ["WB|Limbic|AxD"]], c_.group, c_[C5COV])["WB|Limbic|AxD"]; g_ = c_.group
    rows.append({"sample": lab, "n": int(ok.sum()), "slope on MTL volume (×10⁻³ per ‰)": fit.params["mtl_ratio"], "p": fit.pvalues["mtl_ratio"],
                 "δ CN–MCI after C5": cliff(xa[g_ == "CN"], xa[g_ == "MCI"]), "δ CN–AD after C5": cliff(xa[g_ == "CN"], xa[g_ == "AD"])})
pd.DataFrame(rows).round(4)
""")

md(r"""
### 4.5 Amyloid-defined groups (C7)
Amyloid-negative CN versus amyloid-positive MCI and AD (PET within 180 days), within ADNI-3/4.
""")

code(r"""
pd.crosstab(a34q.group, a34q.amyloid_pos.map({0.0: "amyloid −", 1.0: "amyloid +"}).fillna("no PET ≤180 d"))
""")

md(r"""
### 4.6 Processing checks: route and microstructure coverage (C9–C11)
Section 2.4 showed that processing route and microstructure coverage differ between groups and shift the values.
Three conditions test whether the findings depend on them:

* **C9** adds processing route (two indicator variables) to the age/sex/education regression;
* **C10** adds log microstructure coverage to that regression, for the diffusion features (core coverage is a
  structural measure and keeps the C2 adjustment);
* **C11** keeps only participants whose coverage is at least 25% — a restriction that needs no model.

C10 assumes a straight-line relation between log coverage and each measure; C11 makes no such assumption but keeps
only 20 AD participants. They are checks, not corrections: neither can recover diffusion values that were never
sampled. The definitive fix is to recompute FA, MD, AxD and RD on every connected edge with one recipe.
""")

md(r"""
## 5. All conditions — effect sizes and q values
| | Condition |
|---|---|
| C0 | pooled 530, unadjusted (published; the only condition without quality control) |
| C0q | pooled after quality control (519), unadjusted |
| C1 | ADNI-3/4 after quality control, unadjusted |
| **C2** | **ADNI-3/4 after quality control, adjusted for age, sex and education — primary** |
| C3 | C2 + vendor ComBat |
| C4 | ADNI-3/4, propensity-matched per contrast |
| C5 | C2 + medial temporal volume + WMH |
| C6 | C2 on consistent edges |
| C7 | C2 in amyloid-defined groups |
| C8 | C2 restricted to Siemens |
| C9 | C2 + processing route |
| C10 | C2 + microstructure coverage |
| C11 | C2 restricted to coverage ≥ 25% |
""")

code(r"""
X6 = pubq.copy(); X6.update(conq)
P = a34q.index
C0TAG = "C0 Pooled 530 (published)" if PUBLISHED else "C0 Pooled, rebuilt connectomes"
spec = [(C0TAG, cov.index, pub, None, None),
        ("C0q Pooled after QC", cov.index[cov.qc_fail == 0], pubq, None, None),
        (f"C1 {PL}", P, pubq, None, None),
        (f"C2 {PL} + age, sex, education", P, pubq, COVS, None),
        ("C3 + vendor ComBat", P, pubq, COVS, "combat"),
        ("C4 Propensity-matched", P, pubq, None, "match"),
        ("C5 + MTL volume + WMH", P, pubq, COVS + ["mtl_ratio", "wmh_log"], None),
        ("C6 Consistent edges", P, X6, COVS, None),
        ("C7 Amyloid-defined groups", P, pubq, COVS, "amyloid"),
        ("C8 Siemens only", P[a34q.vendor == "Siemens"], pubq, COVS, None),
        ("C9 + processing route", P, pubq, COVS + ["route_r4", "route_step7"], None),
        ("C10 + microstructure coverage", P, pubq, COVS, "coverage"),
        ("C11 Coverage at least 25%", P[a34q.md_coverage >= 0.25], pubq, COVS, None)]
NAMES = [c for c, *_ in spec]
MAIN = [c for c in NAMES if c.split()[0] in ("C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8")]   # report's main tables
SENS = [c for c in NAMES if c not in MAIN]                                                             # C0q, C9-C11
DATA, EFF = {}, []
for tag, idx, X, covs, special in spec:
    cv = cov.loc[idx].copy(); Xc = X.reindex(idx)
    if special == "amyloid":
        k = ((cv.group == "CN") & (cv.amyloid_pos == 0)) | (cv.group.isin(["MCI", "AD"]) & (cv.amyloid_pos == 1)); cv, Xc = cv[k], Xc[k]
    if covs: ok = cv[covs].notna().all(1); cv, Xc = cv[ok], Xc[ok]
    Xc = Xc[eligible(Xc, cv.group)]
    if special == "combat": Xc, _ = combat(Xc, cv)
    Xa = adjust(Xc, cv.group, cv[covs]) if covs else Xc
    if special == "coverage":
        micro = [c for c in Xa.columns if not c.startswith("COV|")]; ok = cv.log_coverage.notna()
        cv, Xa = cv[ok], Xa.loc[ok].copy(); Xa[micro] = adjust(Xc.loc[ok, micro], cv.group, cv[covs + ["log_coverage"]])
    if special == "match":
        parts = []
        for (a, b), mm in matched.items():
            ef = effects(Xa.loc[mm.index], mm.group, tag); parts.append(ef[ef.contrast == f"{a}-{b}"])
        EFF.append(pd.concat(parts)); DATA[tag] = (Xa, cv.group, "match")
    else:
        EFF.append(effects(Xa, cv.group, tag)); DATA[tag] = (Xa, cv.group, None)
    print(f"{tag:36} n = {dict(cv.group.value_counts())}")
eff = pd.concat(EFF, ignore_index=True)

ref = pd.read_csv(OUT / "effects_long.csv")
chk = eff.merge(ref, on=["condition", "feature", "contrast"], suffixes=("", "_report"))
print(f"\nagreement with the report's engine: max |Δδ| = {np.nanmax((chk.delta - chk.delta_report).abs()):.1e}, "
      f"max |Δq| = {np.nanmax((chk.q - chk.q_report).abs()):.1e} over {len(chk)} estimates")
""")

md(r"""
### 5.1 Diffusivity before anisotropy (F1)
Each cell is the **number of networks (out of ten)** in which that measure differs significantly (q < 0.05) between
the two groups under that condition. Read along a row to see one condition; read down a column to see how one
contrast changes across conditions. The verdict tests the *pattern* (diffusivity widespread at CN–MCI, FA mainly in
AD contrasts), not whether each count is unchanged — counts can fall while the pattern holds. Rows below the red
line are the sensitivity conditions (C0q, C9–C11). Note the FA columns for CN–AD: 6–7 networks in most conditions,
but 0 on Siemens alone (C8) and with coverage controlled (C10, C11).
""")

code(r"""
CT = ["CN-MCI", "CN-AD", "MCI-AD"]
f1 = pd.DataFrame([{**{"condition": c.split()[0]}, **{f"{m} {ct}": int(((eff.condition == c) & (eff.kind == "WB") & (eff.measure == m) & (eff.contrast == ct) & (eff.q < 0.05)).sum())
                    for m in ["AxD", "MD", "RD", "FA"] for ct in CT}} for c in MAIN + SENS]).set_index("condition")
fig, ax = plt.subplots(figsize=(12, 5.6))
im = ax.imshow(f1.values, cmap="Blues", vmin=0, vmax=10, aspect="auto")  # one row per condition
ax.set_xticks(range(f1.shape[1])); ax.set_xticklabels(f1.columns, rotation=40, ha="right", fontsize=8)
ax.set_yticks(range(f1.shape[0])); ax.set_yticklabels(f1.index)
for i in range(f1.shape[0]):
    for j in range(f1.shape[1]): ax.text(j, i, f1.values[i, j], ha="center", va="center", fontsize=8, color="white" if f1.values[i, j] > 6 else "black")
for x_ in [2.5, 5.5, 8.5]: ax.axvline(x_, color="white", lw=3)
ax.axhline(len(MAIN) - 0.5, color="#dc2626", lw=1.5)          # below the red line: sensitivity conditions
ax.set_title("Networks (of 10) separating groups, q < 0.05"); plt.colorbar(im, ax=ax, fraction=0.02); plt.tight_layout()
plt.savefig(OUT / "fig_p1_f1_heatmap.png", dpi=300); plt.show()
""")

md(r"""
## 6. Short- versus long-range effects (F3) with a subject-level bootstrap
**The statistic.** For a contrast, take |δ| in each of the 40 network × measure cells for short-range edges and for
long-range edges, average each, and subtract: **SR − LR**. Positive = effects are larger on short-range edges;
negative = larger on long-range edges.

**Why a new interval.** The manuscript compared the 40 short-range and 40 long-range cells with a paired Wilcoxon
test, which treats the cells as 40 independent observations. They are not: all 40 come from the same participants,
and the four measures in a network are strongly correlated. The bootstrap below resamples **participants** (within
each group), recomputes every cell and the SR − LR difference 2,000 times, and takes the middle 95% of those values.
If that interval excludes zero, the difference is supported; if it includes zero, it is not.
""")

code(r"""
MEAS = ["FA", "MD", "RD", "AxD"]
def cells(X, g, a, b, kind="TR"):
    pre = ("SR-", "LR-") if kind == "TR" else ("SRexc-", "LRexc-"); A_, B_ = (g == a).values, (g == b).values; ds, dl = [], []
    for net in NETS:
        for mm in MEAS:
            s_, l_ = f"{kind}|{net}|{pre[0]}{mm}", f"{kind}|{net}|{pre[1]}{mm}"
            if s_ in X and l_ in X:
                v1, v2 = cliff(X[s_].values[A_], X[s_].values[B_]), cliff(X[l_].values[A_], X[l_].values[B_])
                if np.isfinite(v1) and np.isfinite(v2): ds.append(abs(v1)); dl.append(abs(v2))
    return np.array(ds), np.array(dl)

X0, g0, _ = DATA[C0TAG]
print("Published approach (paired Wilcoxon across cells) vs the cell counts:")
for a, b in CONTRASTS:
    s_, l_ = cells(X0, g0, a, b)
    print(f"  {a}–{b}: SR {s_.mean():.3f}  LR {l_.mean():.3f}  SR>LR in {(s_>l_).sum()}/{len(s_)} cells  Wilcoxon p = {stats.wilcoxon(s_, l_).pvalue:.1e}")

def boot(X, g, a, b, kind, B=B_BOOT):
    ia, ib = np.flatnonzero((g == a).values), np.flatnonzero((g == b).values)
    sub, gg = X.iloc[np.concatenate([ia, ib])].reset_index(drop=True), g.iloc[np.concatenate([ia, ib])].reset_index(drop=True)
    ia, ib = np.flatnonzero((gg == a).values), np.flatnonzero((gg == b).values); out = []
    for _ in range(B):
        pick = np.concatenate([rng.choice(ia, len(ia)), rng.choice(ib, len(ib))])
        s_, l_ = cells(sub.iloc[pick].reset_index(drop=True), gg.iloc[pick].reset_index(drop=True), a, b, kind); out.append(s_.mean() - l_.mean())
    return np.array(out)
""")

code(r"""
t1 = time.time(); live, dists = [], {}
for tag in [C0TAG, f"C2 {PL} + age, sex, education"]:
    X, g, _ = DATA[tag]
    for kind in ["TR", "EX"]:
        for a, b in CONTRASTS:
            s_, l_ = cells(X, g, a, b, kind); bs = boot(X, g, a, b, kind); dists[(tag[:2], kind, f"{a}-{b}")] = bs
            lo, hi = np.percentile(bs, [2.5, 97.5])
            live.append(dict(condition=tag[:2], edges=kind, contrast=f"{a}–{b}", SR=round(s_.mean(), 3), LR=round(l_.mean(), 3),
                             SR_minus_LR=round(s_.mean() - l_.mean(), 3), CI_low=round(lo, 3), CI_high=round(hi, 3),
                             supported="yes" if lo > 0 or hi < 0 else "no (includes 0)", cells_SR_larger=f"{(s_>l_).sum()}/{len(s_)}"))
print(f"bootstrap ({B_BOOT} resamples × 12 comparisons) took {time.time()-t1:.0f} s")
pd.DataFrame(live)
""")

code(r"""
fig, axes = plt.subplots(1, 3, figsize=(13, 3.2), sharey=True)
for ax, ct in zip(axes, CT):
    for tag, col in [("C0", "#111827"), ("C2", "#2563eb")]:
        bs = dists[(tag, "TR", ct)]; ax.hist(bs, bins=40, alpha=0.45, color=col, label=f"{tag}: 95% CI [{np.percentile(bs,2.5):+.3f}, {np.percentile(bs,97.5):+.3f}]")
    ax.axvline(0, color="#dc2626", lw=1.2); ax.set_title(f"{ct.replace('-', '–')}: bootstrap SR − LR (all edges)", fontsize=9); ax.legend(fontsize=7)
    ax.set_xlabel("mean |δ| short-range − long-range")
plt.tight_layout(); plt.savefig(OUT / "fig_nb_bootstrap.png", dpi=200); plt.show()
""")

md(r"""
**Reading the histograms.** Each histogram is 2,000 bootstrap values of SR − LR. If the red zero line falls inside
the central 95% of a histogram, the data cannot rule out that short- and long-range effects are equal. In the pooled
data (black) zero lies outside the CN–AD and MCI–AD histograms but inside the CN–MCI one; in the primary analysis
(blue) zero lies inside all three.

The remaining conditions use the identical code; their bootstrap intervals are loaded from the report's run
(`sr_lr_summary.csv`). Set `RUN_ALL = True` to recompute them here (about 35 extra minutes).
""")

code(r"""
RUN_ALL = False
srl = pd.read_csv(OUT / "sr_lr_summary.csv")
if RUN_ALL:
    rows = []
    for tag, (X, g, sp) in DATA.items():
        for kind in ["TR", "EX"]:
            for a, b in CONTRASTS:
                XX, gg = (X.loc[matched[(a, b)].index], matched[(a, b)].group) if sp == "match" else (X, g)
                s_, l_ = cells(XX, gg, a, b, kind); bs = boot(XX, gg, a, b, kind)
                rows.append(dict(condition=tag, kind=kind, contrast=f"{a}-{b}", n_cells=len(s_), sr_mean=s_.mean(), lr_mean=l_.mean(),
                                 diff=s_.mean() - l_.mean(), ci_lo=np.percentile(bs, 2.5), ci_hi=np.percentile(bs, 97.5), sr_gt_lr=int((s_ > l_).sum())))
    srl = pd.DataFrame(rows)
chk = pd.DataFrame(live).query("edges == 'TR'")
print("live bootstrap vs report (C0, C2, all edges):")
for _, r in chk.iterrows():
    rr = srl[(srl.condition.str.split().str[0] == r.condition) & (srl.kind == "TR") & (srl.contrast == r.contrast.replace("–", "-"))].iloc[0]
    print(f"  {r.condition} {r.contrast}: live [{r.CI_low:+.3f}, {r.CI_high:+.3f}]  report [{rr.ci_lo:+.3f}, {rr.ci_hi:+.3f}]  (same point estimate {r.SR_minus_LR:+.3f} = {rr['diff']:+.3f})")
""")

md(r"""
## 7. Figures used in the report
### Figure P1-1 — leading effects under each condition
One panel per effect. Inside a panel, each row is one condition (C0 at the top, C8 at the bottom); the bar runs from
zero to the signed Cliff's δ. **Filled circle = significant (q < 0.05); open circle = not significant.** The
dashed line marks the published value (pooled 530, published connectomes) so changes are easy to see, and C2 (primary)
is drawn in blue. On the rebuilt connectomes C0 is the pooled rebuilt analysis, so the gap between the dashed line and
C0 is the effect of the rebuild alone.
""")

code(r"""
KEY = [("Limbic AxD, CN–MCI", "WB|Limbic|AxD", "CN-MCI"), ("Salience AxD, CN–MCI", "WB|Salience_VAN|AxD", "CN-MCI"),
       ("Limbic AxD, CN–AD", "WB|Limbic|AxD", "CN-AD"), ("DMN FA, MCI–AD", "WB|DMN|FA", "MCI-AD"),
       ("Limbic long-range MD, CN–MCI", "TR|Limbic|LR-MD", "CN-MCI"), ("Limbic short-range AxD, CN–AD", "TR|Limbic|SR-AxD", "CN-AD"),
       ("Limbic short-range AxD, MCI–AD", "TR|Limbic|SR-AxD", "MCI-AD"), ("Limbic LR-exception MD, CN–MCI", "EX|Limbic|LRexc-MD", "CN-MCI"),
       ("DMN SR-exception RD, CN–AD", "EX|DMN|SRexc-RD", "CN-AD"), ("Frontoparietal SR-exc. AxD, MCI–AD", "EX|Frontoparietal|SRexc-AxD", "MCI-AD"),
       ("Core coverage, CN–MCI", "COV|core|coverage", "CN-MCI"), ("Core coverage, CN–AD", "COV|core|coverage", "CN-AD"),
       ("Core coverage, MCI–AD", "COV|core|coverage", "MCI-AD")]
conds = MAIN; labels = [f"{c.split()[0]} {c.split(' ', 1)[1]}" for c in conds]
CMP = Path(os.environ.get("IIT_P1_COMPARE", str(OUT / "archive_v1_firstreport")))
PUBC0 = (eff if PUBLISHED else pd.read_csv(CMP / "effects_long.csv")).pipe(lambda d: d[d.condition.str.split().str[0] == "C0"])
fig, axes = plt.subplots(4, 4, figsize=(15, 13), sharex=True); axes = axes.ravel()
for ax, (lab, feat, ct) in zip(axes, KEY):
    vals = [eff[(eff.condition == c) & (eff.feature == feat) & (eff.contrast == ct)] for c in conds]
    y = np.arange(len(conds))[::-1]
    for yi, c, v in zip(y, conds, vals):
        if not len(v): continue
        dlt, q = v.delta.iloc[0], v.q.iloc[0]; col = "#2563eb" if c.startswith("C2") else ("#111827" if c.startswith("C0") else "#6b7280")
        ax.plot([0, dlt], [yi, yi], color=col, lw=2 if c[:2] in ("C0", "C2") else 1.2, alpha=0.8)
        ax.scatter([dlt], [yi], s=46, color=col if q < 0.05 else "white", edgecolors=col, linewidths=1.4, zorder=3)
    pv = PUBC0[(PUBC0.feature == feat) & (PUBC0.contrast == ct)]
    if len(pv): ax.axvline(pv.delta.iloc[0], color="#111827", ls="--", lw=0.8)
    ax.axvline(0, color="#9ca3af", lw=0.8)
    for yi in y[::2]: ax.axhspan(yi - 0.5, yi + 0.5, color="#f3f4f6", zorder=0)
    ax.set_yticks(y); ax.set_yticklabels([c.split()[0] for c in conds], fontsize=8); ax.set_ylim(-0.6, len(conds) - 0.4)
    ax.set_title(lab, fontsize=9.5); ax.set_xlim(-0.65, 0.65)
for ax in axes[len(KEY):]: ax.axis("off")
axes[len(KEY)].text(0, 0.95, "Conditions\n" + "\n".join(labels) + "\n\nFilled = q < 0.05; open = not significant\nDashed line = published value (pooled 530)\nNegative δ = higher in the later-stage group",
                    va="top", fontsize=9, transform=axes[len(KEY)].transAxes)
for ax in axes[:len(KEY)]: ax.tick_params(labelbottom=True, labelsize=8)   # sharex hides them above empty panels
for ax in axes[len(KEY) - 4:len(KEY)]: ax.set_xlabel("Cliff's δ")
plt.tight_layout(); plt.savefig(OUT / "fig_p1_key_effects.png", dpi=300); plt.show()
""")

md(r"""
### Figure P1-2 — short- minus long-range difference under each condition
Each grey band is one condition; inside it, the three dots are the three contrasts. The horizontal line through a dot
is its 95% bootstrap interval. **If a line crosses the dashed zero line, that difference is not statistically
supported.** Right of zero = larger effects on short-range edges; left = larger on long-range edges.
""")

code(r"""
colors = {"CN-MCI": "#2563eb", "CN-AD": "#ea580c", "MCI-AD": "#16a34a"}
fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
for ax, kind, title in zip(axes, ["TR", "EX"], ["All short- and long-range edges", "EDR-exception edges"]):
    for i, c in enumerate(conds):
        yc = len(conds) - 1 - i
        if i % 2 == 0: ax.axhspan(yc - 0.5, yc + 0.5, color="#f3f4f6", zorder=0)
        for j, ct in enumerate(CT):
            r = srl[(srl.condition == c) & (srl.kind == kind) & (srl.contrast == ct)]
            if not len(r) or not np.isfinite(r.ci_lo.iloc[0]): continue
            r = r.iloc[0]; yy = yc + (1 - j) * 0.25
            ax.plot([r.ci_lo, r.ci_hi], [yy, yy], color=colors[ct], lw=1.4)
            ax.scatter([r["diff"]], [yy], color=colors[ct], s=28, zorder=3, label=ct.replace("-", "–") if i == 0 else None)
    ax.axvline(0, color="#6b7280", ls="--", lw=1); ax.set_title(title, fontsize=10)
    ax.set_xlabel("mean |δ| short-range − long-range (95% bootstrap CI)")
axes[0].set_yticks(range(len(conds))[::-1]); axes[0].set_yticklabels(labels, fontsize=8.5)
axes[1].legend(title="Contrast (top to bottom in each band)", fontsize=8, title_fontsize=8, loc="lower right")
plt.tight_layout(); plt.savefig(OUT / "fig_p1_sr_lr_forest.png", dpi=300); plt.show()
""")

md(r"""
### Figure P1-3 — what quality control changed
Three panels: the primary analysis (C2), atrophy and lesion adjustment (C5) and ComBat (C3). In each row the grey dot
is the first version of this report (no quality control) and the blue dot the current run; filled = q < 0.05, open =
not significant. Short lines mean little changed. The middle panel is the one that matters: without the corrupt
participants, atrophy and lesion adjustment no longer removes the diffusivity effects. On the rebuilt connectomes the
panels are the pooled analysis after quality control (C0q), the primary analysis (C2) and the atrophy adjustment (C5),
with the grey dot from the published connectomes (after quality control) and the blue dot from the rebuilt ones.
""")

code(r"""
OLD = pd.concat([pd.read_csv(f) for f in [CMP / "effects_long.csv", CMP / "effects_long_C8.csv"] if f.is_file()]).drop_duplicates(["condition", "feature", "contrast"])
OLD["c"] = OLD.condition.str.split().str[0]; E = eff.assign(c=eff.condition.str.split().str[0])
pairs = ([("C2", "Primary analysis (C2)"), ("C5", "Atrophy and lesion adjustment (C5)"), ("C3", "Vendor ComBat (C3)")] if PUBLISHED else
         [("C0q", "Pooled after QC (C0q)"), ("C2", "Primary analysis (C2)"), ("C5", "Atrophy and lesion adjustment (C5)")])
OLD_LAB, NEW_LAB = ("first version (no QC)", "after quality control") if PUBLISHED else ("published connectomes", "rebuilt connectomes")
fig, axes = plt.subplots(1, 3, figsize=(15, 6), sharey=True)
y = np.arange(len(KEY))[::-1]
for ax, (c, title) in zip(axes, pairs):
    for yi, (lab, feat, ct) in zip(y, KEY):
        ra = OLD[(OLD.c == c) & (OLD.feature == feat) & (OLD.contrast == ct)].iloc[0]; rb = E[(E.c == c) & (E.feature == feat) & (E.contrast == ct)].iloc[0]
        ax.plot([ra.delta, rb.delta], [yi, yi], color="#9ca3af", lw=1.2)
        ax.scatter([ra.delta], [yi], s=40, color="#6b7280" if ra.q < 0.05 else "white", edgecolors="#6b7280", zorder=3)
        ax.scatter([rb.delta], [yi], s=46, color="#2563eb" if rb.q < 0.05 else "white", edgecolors="#2563eb", linewidths=1.4, zorder=4)
    ax.axvline(0, color="#9ca3af", lw=0.8); ax.set_xlim(-0.65, 0.65); ax.set_title(title, fontsize=10); ax.set_xlabel("Cliff's δ")
    for yi in y[::2]: ax.axhspan(yi - 0.5, yi + 0.5, color="#f3f4f6", zorder=0)
axes[0].set_yticks(y); axes[0].set_yticklabels([k[0] for k in KEY], fontsize=8.5)
axes[2].scatter([], [], color="#6b7280", label=OLD_LAB); axes[2].scatter([], [], color="#2563eb", label=NEW_LAB)
axes[2].legend(fontsize=8, loc="lower right", title="filled = q < 0.05", title_fontsize=8)
plt.tight_layout(); plt.savefig(OUT / "fig_p1_qc_pairs.png", dpi=300); plt.show()
""")

md(r"""
### Processing checks at a glance (C2 against C9, C10, C11)
""")

code(r"""
tab = {}
for c in [f"C2 {PL} + age, sex, education"] + [x for x in SENS if not x.startswith("C0q")]:
    col = {}
    for lab, feat, ct in KEY[:10]:
        r = eff[(eff.condition == c) & (eff.feature == feat) & (eff.contrast == ct)].iloc[0]; col[lab] = f"{r.delta:+.3f}{'*' if r.q < 0.05 else ''}"
    col["FA networks, CN–AD"] = int(((eff.condition == c) & (eff.kind == "WB") & (eff.measure == "FA") & (eff.contrast == "CN-AD") & (eff.q < 0.05)).sum())
    for ct in CT:
        r = srl[(srl.condition == c) & (srl.kind == "TR") & (srl.contrast == ct)].iloc[0]; col[f"SR − LR {ct}"] = f"{r['diff']:+.3f} [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}]"
    tab[c.split()[0]] = col
print("* = q < 0.05")
pd.DataFrame(tab)
""")

md(r"""
## 8. Validation
The pooled, unadjusted condition must reproduce the manuscript exactly; otherwise differences in the other conditions
could come from the code rather than the controls.
""")

code(r"""
C0 = eff[eff.condition.str.split().str[0] == "C0"]
if not PUBLISHED:
    print("rebuilt matrices: the checks against the published numbers do not apply;")
    print("quality control matches p1_audit.py (checked in Section 2.3); primary sample:", dict(DATA[f"C2 {PL} + age, sex, education"][1].value_counts()))
else:
    v = lambda f, ct: C0[(C0.feature == f) & (C0.contrast == ct)].iloc[0]
    assert [int(((C0.kind == "WB") & (C0.measure == "FA") & (C0.contrast == ct) & (C0.q < 0.05)).sum()) for ct in CT] == [4, 9, 9]
    assert round(v("WB|Limbic|AxD", "CN-AD").delta, 3) == -0.554 and round(v("TR|Limbic|SR-AxD", "CN-AD").delta, 3) == -0.583
    assert round(v("COV|core|coverage", "CN-AD").delta, 3) == 0.393
    s_, l_ = cells(X0, g0, "CN", "AD"); assert (round(s_.mean(), 3), round(l_.mean(), 3), int((s_ > l_).sum())) == (0.434, 0.340, 37)
    n_of = lambda tag: dict(DATA[tag][1].value_counts())
    assert n_of(f"C2 {PL} + age, sex, education") == {"CN": 244, "MCI": 199, "AD": 44} and n_of("C0q Pooled after QC") == {"CN": 246, "MCI": 200, "AD": 73}
    assert cov.qc_fail.sum() == 11 and cov[(cov.qc_fail == 1) & (cov.adni34 == 1)].group.value_counts().to_dict() == {"CN": 4, "MCI": 1, "AD": 1}
    print("All validation checks passed: the pooled analysis reproduces the published Table 2, Table 3, Table 4 and Table 6 values;")
    print("quality control matches p1_audit.py and leaves the primary sample at CN 244, MCI 199, AD 44.")
print(f"Total run time {time.time()-T0:.0f} s")
""")

nb = nbf.v4.new_notebook(); nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "iit-phase1", "display_name": "IIT Phase 1 (confound control)", "language": "python"}
nbf.write(nb, NB); print("wrote", NB, "with", len(cells), "cells")
