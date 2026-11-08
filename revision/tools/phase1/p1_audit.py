"""Phase 1 step 1b — data-quality audit of everything the Phase 1 analysis reads.

Run after p1_covariates.py and p1_features.py. Checks, per participant:
  A. network-level diffusion values (features_published.csv): physical bounds, MD = (AxD + 2 RD) / 3 identity,
     AxD >= MD >= RD ordering, robust participant-level outliers in FA and MD;
  B. the raw edge matrices: density, nodes without edges, total SIFT2 weight, edge length, edge-level bounds;
  C. processing route and registration quality (research_audit/route_attribution_20260919/final.csv);
  D. covariates: age, education, intracranial volume, medial temporal volume, WMH, amyloid status vs centiloid.
Writes phase1/audit/audit_participants.csv (one row per participant, all metrics + flags) and audit_summary.json.

It also owns the quality-control rules used by every post-QC condition (decided 2026-10-24, after the ComBat diagnostic):
  participant excluded (phase1/qc.csv, qc_fail = 1) when
    R1  median whole-brain MD lies above Q3 + 3 IQR of the cohort (implausible diffusivity level);
    R2  two or more whole-brain networks hold values impossible for a diffusion tensor (FA > 1, AxD < RD,
        MD differing from (AxD + 2 RD) / 3 by more than 1 %, or a diffusivity above free water);
    R3  structural density below Q1 - 3 IQR and below 0.6, the analysis pipeline's floor (failed reconstruction);
    R0  outside the analysis pipeline's cohort (density below 0.6, or listed in the run's exclusions file);
    R4  rebuilt data only: atlas registration failed (FA map vs FMRIB58 template correlation < 0.4, from <unit>_qc.json);
  single network value set to missing (phase1/features_qc.csv, features_consistent_edges_qc.csv) when the
  FA/MD/AxD/RD quartet of that network and edge set has FA > 1 or < 0.05, AxD < RD, or a diffusivity above free water.
"""
import sys as _sys; from pathlib import Path as _Path; _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import iit_chain as chain; chain.boot()      # IIT_ROOT and the chain settings; no effect when IIT_ROOT is unset
import os
import json, re, glob
from pathlib import Path
import numpy as np, pandas as pd

import sc_cohort_settings as cohort_settings
FLOOR = cohort_settings.density_floor()    # the analysis pipeline's density floor (SC_DENSITY_FLOOR, default 0.6; 0 = no floor)
EXP = Path("/home/ec2-user/exp")
A = Path(os.environ.get("IIT_AROOT", "/home/ec2-user/exp/data/derivatives/qc/analysis_cohort"))
CONN = Path(os.environ.get("IIT_CONN", "/data/derivatives/connectomes"))
O = Path(os.environ.get("IIT_P1", "/home/ec2-user/exp/docs/iit_dipanjan/phase1")); AU = O / "audit"; AU.mkdir(exist_ok=True)
cov = pd.read_csv(O / "covariates.csv").set_index("subject_id")
pub = pd.read_csv(O / "features_published.csv", index_col=0).reindex(cov.index)
FREE_WATER = 3.0e-3                       # diffusivity of free water at 37 C, mm2/s


def robust_z(x):
    med = np.nanmedian(x); mad = np.nanmedian(np.abs(x - med)) * 1.4826
    return (x - med) / mad


def far_out(x):                            # Tukey far-out fences, Q1 - 3 IQR and Q3 + 3 IQR
    q1, q3 = np.nanpercentile(x, [25, 75]); return q1 - 3 * (q3 - q1), q3 + 3 * (q3 - q1)


R = pd.DataFrame(index=cov.index)
R[["group", "phase", "vendor"]] = cov[["group", "phase", "vendor"]]

# ---------------- A. network-level diffusion values
dif = pub.columns[pub.columns.str.contains(r"(?:MD|AxD|RD)$")]; fa = pub.columns[pub.columns.str.endswith("FA")]
R["n_dif_over_free_water"] = (pub[dif] > FREE_WATER).sum(1)
R["n_fa_over_1"] = (pub[fa] > 1).sum(1)
R["n_fa_under_0.05"] = (pub[fa] < 0.05).sum(1)
ident, order = [], []
for c in pub.columns[pub.columns.str.endswith("MD")]:
    stem = c[:-2]
    if stem + "AxD" in pub and stem + "RD" in pub:
        pred = (pub[stem + "AxD"] + 2 * pub[stem + "RD"]) / 3
        ident.append(((pub[c] - pred).abs() / pub[c]).rename(c))
        order.append(((pub[stem + "AxD"] < pub[c] * 0.999) | (pub[c] < pub[stem + "RD"] * 0.999)).rename(c))
ident = pd.concat(ident, axis=1); order = pd.concat(order, axis=1)
R["md_identity_max_relerr"] = ident.max(1)
R["n_md_identity_fail"] = (ident > 0.01).sum(1)            # MD differs from (AxD + 2 RD) / 3 by > 1 %
R["n_order_fail"] = order.sum(1)
R["wb_md_median"] = pub.filter(regex=r"^WB\|.*\|MD$").median(1) * 1e3
R["wb_fa_median"] = pub.filter(regex=r"^WB\|.*\|FA$").median(1)
for m in ["wb_md_median", "wb_fa_median"]:
    lo, hi = far_out(R[m]); R[m + "_far_out"] = ((R[m] < lo) | (R[m] > hi)).astype(int); R[m + "_rz"] = robust_z(R[m])

# ---------------- B. raw edge matrices
m = pd.read_csv(A / "00_master/master_cohort.csv", low_memory=False)
fsid = {}
for sid, iid in zip(m.subject_id, m["Image ID"].astype(str).str.replace(r"\.0$", "", regex=True)):
    f = CONN / f"SC_AAL166_{sid}_I{iid}_len_mean.csv"
    fsid[sid] = f"{sid}_I{iid}" if f.is_file() else sorted(CONN.glob(f"SC_AAL166_{sid}_I*_len_mean.csv"))[0].name[10:-13]
for sid in cov.index:                                 # participants outside the cohort: audit their matrices too
    if sid not in fsid and (alts := sorted(CONN.glob(f"SC_AAL166_{sid}_I*_len_mean.csv"))):
        fsid[sid] = alts[0].name[10:-13]
K = cohort_settings.n_nodes(); iu = np.triu_indices(K, 1); rows = {}     # 166 unless SC_N_NODES leaves out the last ones
R4_FIELD = chain.r4_field(); n_qc_json = n_qc_lacking = 0      # IIT_R4_FIELD=none states that this chain has no registration check
for sid in cov.index:
    if sid not in fsid: continue                      # no matrices for this participant (partial runs only)
    rd = lambda k: np.loadtxt(CONN / f"SC_AAL166_{fsid[sid]}_{k}.csv", delimiter=",")
    L, W = rd("len_mean"), rd("fd_sum")
    # With a fixed length measure len_mean is 0 on edges that measure does not define, so presence comes from the weights alone.
    present = (L > 0) & (W > 0) if cohort_settings.get("length_measure") == "per_person" else W > 0
    r = dict(density=present[iu].mean(), zero_nodes=int((present[:K, :K].sum(1) == 0).sum()), fd_total=W[iu].sum(),
             len_median=np.median(L[iu][present[iu] & (L[iu] > 0)]) if (present[iu] & (L[iu] > 0)).any() else np.nan)
    if (CONN / f"SC_AAL166_{fsid[sid]}_md_mean.csv").is_file():
        md, fam, ad_, rd_ = rd("md_mean"), rd("fa_mean"), rd("ad_mean"), rd("rd_mean"); p = present[iu]
        r.update(edge_md_over_free_water=float((md[iu][p] > FREE_WATER).mean()), edge_fa_over_1=float((fam[iu][p] > 1).mean()),
                 edge_md_median=float(np.median(md[iu][p])) * 1e3,
                 md_coverage=float(((md[iu] > 0) & p).sum() / max(p.sum(), 1)))   # structural edges carrying microstructure
    if (q := CONN / f"SC_AAL166_{fsid[sid]}_qc.json").is_file():          # the uniform rebuild records its registration
        n_qc_json += 1
        if R4_FIELD:
            r["reg_fa_corr_rebuild"] = json.loads(q.read_text()).get(R4_FIELD)
            r["reg_fa_corr_rebuild"] = np.nan if r["reg_fa_corr_rebuild"] is None else r["reg_fa_corr_rebuild"]
            n_qc_lacking += int(not np.isfinite(float(r["reg_fa_corr_rebuild"])))
    rows[sid] = r
if n_qc_lacking:                                                # rule R4 must not go quiet: NaN < 0.4 is False for everybody
    raise SystemExit(f"R4 GUARD: {n_qc_lacking} of {n_qc_json} qc.json files lack {R4_FIELD}, so rule R4 (atlas registration) cannot run. "
                     f"Record the field in the recipe, or set IIT_R4_FIELD=none to state that this chain has no such check.")
R = R.join(pd.DataFrame(rows).T.apply(pd.to_numeric))
for mtr in ["density", "fd_total", "len_median"]:
    lo, hi = far_out(R[mtr]); R[mtr + "_far_out"] = ((R[mtr] < lo) | (R[mtr] > hi)).astype(int); R[mtr + "_rz"] = robust_z(R[mtr])

# ---------------- C. processing route and registration
if chain.root() is None:
    fin = pd.read_csv(EXP / "research_audit/route_attribution_20260919/final.csv")
    fin["subject_id"] = fin.sid.str.extract(r"^(\d{3}_S_\d{4})")[0]
    fin = fin.drop_duplicates("subject_id").set_index("subject_id")
    R["route"] = fin.s4.reindex(R.index).str[0].map({"A": "legacy_step7", "B": "legacy_r4", "C": "new", "D": "new", "E": "new", "F": "new"})
    R["route_detail"] = fin.s4.reindex(R.index)
    R["reg_ncc"] = fin.reg_ncc.reindex(R.index)
else:                                  # a chain is one recipe from raw data: the earlier routes do not describe its connectomes
    R["route"] = R["route_detail"] = "one recipe"; R["reg_ncc"] = np.nan

# ---------------- D. covariates
for c in ["age", "education", "etiv", "mtl_ratio", "wmh_log", "centiloid"]:
    R[c] = cov[c]
R["age_implausible"] = ((R.age < 50) | (R.age > 100)).astype(int)
R["education_implausible"] = ((R.education < 4) | (R.education > 25)).astype(int)
for c in ["etiv", "mtl_ratio", "wmh_log"]:
    R[c + "_rz"] = robust_z(R[c]); lo, hi = far_out(R[c]); R[c + "_far_out"] = ((R[c] < lo) | (R[c] > hi)).astype(int)
R["amyloid_status_vs_centiloid"] = np.where(cov.amyloid_pos.notna() & cov.centiloid.notna(),
                                            ((cov.amyloid_pos == 1) != (cov.centiloid >= 20)).astype(float), np.nan)
# ---------------- E. quality-control decision
NETS = sorted({c.split("|")[1] for c in pub.columns if c.startswith("WB|")})
def tensor_bad(X, stem, tol):
    FA, MD, AD, RD = (X.get(stem + k) for k in ("FA", "MD", "AxD", "RD"))
    bad = (FA > 1) | (AD < RD) | (MD > FREE_WATER) | (AD > FREE_WATER) | (RD > FREE_WATER)
    if tol is not None: bad |= (MD - (AD + 2 * RD) / 3).abs() / MD > tol
    return bad.fillna(False)
R["n_wb_tensor_fail"] = sum(tensor_bad(pub, f"WB|{n}|", 0.01).astype(int) for n in NETS)
lo_md, hi_md = far_out(R.wb_md_median); lo_den, _ = far_out(R.density)
lo_den = min(lo_den, FLOOR)        # failed = below the far-out fence AND below the analysis pipeline's density floor (default 0.6; no floor: R3 off)
R1, R2, R3 = R.wb_md_median > hi_md, R.n_wb_tensor_fail >= 2, R.density < lo_den
R["in_cohort"] = R.index.isin(m.subject_id).astype(int)      # the analysis pipeline keeps connectomes with density >= 0.6
R0 = R.in_cohort == 0
if "reg_fa_corr_rebuild" not in R: R["reg_fa_corr_rebuild"] = np.nan
R4 = R.reg_fa_corr_rebuild < chain.r4_min()
r0_label = np.where(R.density < FLOOR, "R0 below the pipeline density floor", "R0 excluded before the analysis")
R["qc_reason"] = [", ".join(k for k, f in ((l0, z), ("R1 diffusivity level", a), ("R2 tensor values", b), ("R3 reconstruction", c),
                                           ("R4 atlas registration", e)) if f)
                  for l0, z, a, b, c, e in zip(r0_label, R0, R1, R2, R3, R4)]
R["qc_fail"] = (R0 | R1 | R2 | R3 | R4).astype(int)
R.to_csv(AU / "audit_participants.csv")
R[["qc_fail", "qc_reason", "in_cohort", "wb_md_median", "n_wb_tensor_fail", "density", "zero_nodes", "md_coverage", "reg_fa_corr_rebuild",
   "route"]].to_csv(O / "qc.csv")

def mask_values(X):
    X = X.copy(); n = 0
    stems = sorted({c[:-2] for c in X.columns if c.endswith("FA")})
    for st in stems:
        quart = [st + k for k in ("FA", "MD", "AxD", "RD") if st + k in X]
        if len(quart) < 4: continue
        bad = tensor_bad(X, st, None) | (X[st + "FA"] < 0.05)
        n += int(X.loc[bad, quart].notna().sum().sum()); X.loc[bad, quart] = np.nan
    return X, n
FQ, n_masked = mask_values(pub); FQ.to_csv(O / "features_qc.csv")
CE, n_masked_ce = mask_values(pd.read_csv(O / "features_consistent_edges.csv", index_col=0)); CE.to_csv(O / "features_consistent_edges_qc.csv")

# ---------------- summary
a34 = chain.primary(cov).reindex(R.index) == 1 if chain.sample() != "adni34" else R.phase.isin(["ADNI-3", "ADNI-4"])
S = {"n": int(len(R)), "qc_fail": int(R.qc_fail.sum()), "qc_by_reason": R[R.qc_fail == 1].qc_reason.value_counts().to_dict(),
     "qc_fail_adni34": R[(R.qc_fail == 1) & a34].group.value_counts().to_dict(),
     "values_masked_among_kept": int(FQ[R.qc_fail == 0].isna().sum().sum() - pub[R.qc_fail == 0].isna().sum().sum()),
     "fences": dict(md_upper=round(float(hi_md), 3), density_lower=round(float(lo_den), 3))}
cvg = R[a34 & (R.qc_fail == 0)]
S["md_coverage_adni34_kept"] = {g: dict(median=round(float(d.md_coverage.median()), 3), share_below_5pct=round(float((d.md_coverage < 0.05).mean()), 3))
                                for g, d in cvg.groupby("group")}
for c in ["n_dif_over_free_water", "n_fa_over_1", "n_fa_under_0.05", "n_md_identity_fail", "n_order_fail",
          "wb_md_median_far_out", "wb_fa_median_far_out", "density_far_out", "fd_total_far_out", "len_median_far_out",
          "age_implausible", "education_implausible", "etiv_far_out", "mtl_ratio_far_out", "wmh_log_far_out"]:
    flag = R[c] > 0
    S[c] = dict(participants=int(flag.sum()), of_which_qc_fail=int((flag & (R.qc_fail == 1)).sum()),
                in_adni34_passing_qc=int((flag & a34 & (R.qc_fail == 0)).sum()))
S["amyloid_status_vs_centiloid_mismatch"] = int(np.nansum(R.amyloid_status_vs_centiloid))
S["route_by_group_adni34_qc"] = pd.crosstab(R.loc[a34 & (R.qc_fail == 0), "route"], R.loc[a34 & (R.qc_fail == 0), "group"]).to_dict()
S["route_by_group_all"] = pd.crosstab(R.route, R.group).to_dict()
if chain.root() is not None:                                     # a chain records the rules it ran under
    S["rules"] = dict(density_floor=FLOOR, r3=("off (no floor)" if FLOOR == 0 else "on"),
                      r4=(f"{R4_FIELD} < {chain.r4_min():g}" if R4_FIELD else "off: this chain has no registration check (IIT_R4_FIELD=none)"),
                      qc_json_files=n_qc_json, primary_sample=chain.sample())
json.dump(S, open(AU / "audit_summary.json", "w"), indent=1)
print(json.dumps(S, indent=1))
