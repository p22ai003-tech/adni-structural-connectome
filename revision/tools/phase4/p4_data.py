"""Phase 4 step 1 — one row per participant: covariates, diagnosis, cognitive scores at the diffusion scan, and the four
connectome feature families of the manuscript's prediction ladder, computed on the uniformly rebuilt connectomes.

Scores (ADNIMERGE2, visit nearest the diffusion scan, within 180 days; negative codes = missing):
  CDR-SB (CDR.CDRSB), global CDR (CDR.CDGLOBAL), ADAS-Cog 13 (ADAS.TOTAL13), memory and executive-function composites
  (ADSP Phenotype Harmonization Consortium, ADSP_PHC_COGN.PHC_MEM / PHC_EXF; the older UWNPSYCHSUM ADNI-Mem / ADNI-EF
  cover only 296 participants and 1 with AD), MMSE (MMSE.MMSCORE), FAQ (FAQ.FAQTOTAL).
Feature families (the recipe of hcp_analysis/build_simple_ml.py, re-implemented here so that every participant gets
values, not only those with the old image-search scores):
  F1 whole brain   per network FA and a diffusivity composite DIFF (mean robust z of MD, RD and AxD), + global RD
  F2 tract length  the same within short-range (SR) and long-range (LR) edges
  F3 exceptions    the same within SR and LR exception edges
  F4 architecture  exception architecture (pipeline table exception_architecture_subject.csv) + core coverage
Robust z uses the median and IQR over the analysed participants (target-independent). Participants failing the Phase 1
quality control (qc.csv) are dropped.

Env: IIT_AROOT (analysis tree), IIT_P1 (Phase 1 run: covariates, qc, features), IIT_P4 (output), IIT_AM2 (ADNIMERGE2 data).
"""
import sys as _sys; from pathlib import Path as _Path; _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import iit_chain as chain; chain.boot()      # IIT_ROOT and the chain settings; no effect when IIT_ROOT is unset
import os
import warnings
from pathlib import Path
import numpy as np, pandas as pd, pyreadr
import sc_cohort_settings as cohort_settings

warnings.filterwarnings("ignore")
A = Path(os.environ.get("IIT_AROOT", "/data/derivatives/qc/analysis_cohort_uniform"))
P1 = Path(os.environ.get("IIT_P1", "/home/ec2-user/exp/docs/iit_dipanjan/phase1_uniform"))
O = Path(os.environ.get("IIT_P4", "/home/ec2-user/exp/docs/iit_dipanjan/phase4_uniform")); O.mkdir(exist_ok=True)
AM2 = Path(os.environ.get("IIT_AM2", "/home/ec2-user/exp/cohort/adnimerge/extracted/ADNIMERGE2/data"))
FUNC = A / "19_network_analysis/functional"
NETWORKS = ["Visual", "Somatomotor", "DorsalAttention", "Salience_VAN", "Limbic", "Frontoparietal", "DMN", "Subcortical",
            "Cerebellar", "Brainstem"]
WINDOW = 180

cov = pd.read_csv(P1 / "covariates.csv")
qc = pd.read_csv(P1 / "qc.csv", index_col=0)
cov = cov[cov.subject_id.map(qc.qc_fail).eq(0)].copy()
cov["scan_date"] = pd.to_datetime(cov.scan_date)


def am2(t):
    r = pyreadr.read_r(str(AM2 / f"{t}.rda")); return r[list(r)[0]]


def nearest(table, col, dcol):
    df = am2(table)[["RID", col, dcol]].copy(); df[col] = pd.to_numeric(df[col], errors="coerce")
    if table not in ("ADSP_PHC_COGN",): df.loc[df[col] < 0, col] = np.nan      # negative = missing code (PHC z-scores can be < 0)
    df[dcol] = pd.to_datetime(df[dcol], errors="coerce"); df = df.dropna()
    x = cov[["subject_id", "RID", "scan_date"]].merge(df, on="RID", how="left")
    x["lag"] = (x[dcol] - x.scan_date).dt.days.abs()
    x = x.sort_values("lag").drop_duplicates("subject_id").set_index("subject_id")
    ok = x.lag <= WINDOW
    return x[col].where(ok), x.lag.where(ok)


SCORES = {"cdrsb": ("CDR", "CDRSB", "VISDATE"), "cdr_global": ("CDR", "CDGLOBAL", "VISDATE"), "adas13": ("ADAS", "TOTAL13", "VISDATE"),
          "phc_mem": ("ADSP_PHC_COGN", "PHC_MEM", "EXAMDATE"), "phc_exf": ("ADSP_PHC_COGN", "PHC_EXF", "EXAMDATE"),
          "mmse": ("MMSE", "MMSCORE", "VISDATE"), "faq": ("FAQ", "FAQTOTAL", "VISDATE")}
D = cov.set_index("subject_id")[["group", "phase", "adni34", "vendor", "age", "female", "education"]].copy()
for name, (t, col, dcol) in SCORES.items():
    v, lag = nearest(t, col, dcol); D[name] = v; D[f"{name}_lag"] = lag
donors = cohort_settings.donor_map(A)       # labels shuffled: the scores move with the diagnosis, each person takes their donor's
if donors is not None:
    scores = [c for name in SCORES for c in (name, f"{name}_lag")]
    D[scores] = D[scores].reindex([donors.get(s, s) for s in D.index]).values
D["cdr_impaired"] = np.where(D.cdr_global.notna(), (D.cdr_global >= 0.5).astype(float), np.nan)


def robust_z(x):
    med = np.nanmedian(x); iqr = np.nanpercentile(x, 75) - np.nanpercentile(x, 25)
    return (x - med) / (iqr if iqr > 0 else 1.0)


def composites(long, prefix):
    """subject x network long frame with measure in FA/MD/RD/AxD -> FA and DIFF composite columns."""
    long = long[long.subject_id.isin(D.index)]
    wide = long.pivot_table(index="subject_id", columns=["network", "measure"], values="value")
    out = {}
    for net in NETWORKS:
        if (net, "FA") in wide.columns: out[f"{prefix}_{net}_FA"] = wide[(net, "FA")]
        zs = [robust_z(wide[(net, m)]) for m in ("MD", "RD", "AxD") if (net, m) in wide.columns]
        if zs: out[f"{prefix}_{net}_DIFF"] = pd.concat(zs, axis=1).mean(axis=1)
    return pd.DataFrame(out)


mic = pd.read_csv(FUNC / "network_microstructure_subject.csv")
mic["measure"] = mic.metric.map({"fa_mean": "FA", "md_mean": "MD", "rd_mean": "RD", "ad_mean": "AxD"})
f1 = composites(mic.dropna(subset=["measure"]), "WB")
f1["WB_global_RD"] = mic[mic.measure == "RD"].groupby("subject_id").value.mean()
rng = pd.read_csv(FUNC / "network_range_restricted_subject.csv")
f2 = pd.concat([composites(rng[rng.measure.str.startswith(f"{r}-")].assign(measure=lambda d: d.measure.str[3:]), r) for r in ("SR", "LR")], axis=1)
exc = pd.read_csv(FUNC / "network_exception_measures_subject.csv")
f3 = pd.concat([composites(exc[exc.measure.str.startswith(f"{r}exc-")].assign(measure=lambda d: d.measure.str.split("-").str[1]),
                           f"{r}EXC") for r in ("SR", "LR")], axis=1)
arch = pd.read_csv(A / "20_exception_specificity/exception_architecture_subject.csv").set_index("subject_id")
f4 = arch.drop(columns=[c for c in ("group", "phase") if c in arch.columns]).add_prefix("ARCH_")
f4["ARCH_core_coverage"] = pd.read_csv(P1 / "features_qc.csv", index_col=0)["COV|core|coverage"]
F = pd.concat([f1, f2, f3, f4], axis=1).reindex(D.index)
high = F.isna().mean() > 0.5
F = F.loc[:, ~high]
fam = {c: ("F1" if c.startswith("WB_") else "F2" if c.startswith(("SR_", "LR_")) else "F3" if c.startswith(("SREXC_", "LREXC_")) else "F4")
       for c in F.columns}
pd.Series(fam, name="family").rename_axis("feature").to_csv(O / "feature_families.csv")
D.join(F).to_csv(O / "p4_data.csv")

prim = D[chain.primary(D) == 1]
print(f"participants after Phase 1 QC: {len(D)} ({chain.sample_label()} {len(prim)}); features {F.shape[1]} "
      f"(F1 {sum(v == 'F1' for v in fam.values())}, F2 {sum(v == 'F2' for v in fam.values())}, F3 {sum(v == 'F3' for v in fam.values())}, "
      f"F4 {sum(v == 'F4' for v in fam.values())}); dropped for >50% missing: {list(high[high].index)}")
for s in list(SCORES) + ["cdr_impaired"]:
    lag = D.get(f"{s}_lag", D["cdr_global_lag"])
    print(f"  {s:12s} pooled n = {D[s].notna().sum():3d}  {chain.sample_label()} n = {prim[s].notna().sum():3d} "
          f"(CN {prim[prim.group == 'CN'][s].notna().sum()}, MCI {prim[prim.group == 'MCI'][s].notna().sum()}, AD {prim[prim.group == 'AD'][s].notna().sum()})"
          f"  median lag {lag[D[s].notna()].median():.0f} d")
