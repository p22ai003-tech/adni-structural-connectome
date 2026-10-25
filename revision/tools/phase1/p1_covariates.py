"""Phase 1 step 1 — one covariate row per participant (n = 530).

age, sex (1 = female), education (PTDEMOG), phase, scanner vendor, DWI scan date;
FastSurfer eTIV, hippocampus, medial-temporal volume (amygdala + entorhinal + parahippocampal);
FLAIR WMH (UCD_WMH) and amyloid PET (UC Berkeley, centiloids + status) from the visit nearest the
diffusion scan (within 180 days). ADNIMERGE2 tables are read from the extracted .rda files.
"""
import glob, re, sys
from pathlib import Path
import numpy as np, pandas as pd, pyreadr

EXP = Path("/home/ec2-user/exp"); OUT = EXP / "docs/iit_dipanjan/phase1"
AM2 = Path(sys.argv[1])          # .../adnimerge2_extract/ADNIMERGE2/data
FS = Path("/data/derivatives/fastsurfer")

def am2(t):
    r = pyreadr.read_r(str(AM2 / f"{t}.rda")); return r[list(r)[0]]

m = pd.read_csv(EXP / "data/derivatives/qc/analysis_cohort/00_master/master_cohort.csv", low_memory=False)
m = m[["subject_id", "group", "age", "sex", "phase"]].copy()
m["female"] = m.sex.astype(str).str.upper().str[0].eq("F").astype(int)
m["RID"] = m.subject_id.str[-4:].astype(int)

# true image id of the analysed session (master's Image ID is wrong for 32 AD) -> scan date + vendor
img = {}
for f in glob.glob("/data/derivatives/connectomes/SC_AAL166_*_count.csv"):
    s = re.search(r"SC_AAL166_(\d{3}_S_\d{4})_I(\d+)_count", f)
    if s: img[s.group(1)] = s.group(2)
m["image_id"] = m.subject_id.map(img)
dti = pd.read_csv(EXP / "cohort/dti_master.csv", low_memory=False)
dti["Image ID"] = dti["Image ID"].astype(str).str.replace(r"\.0$", "", regex=True)
dti = dti.drop_duplicates("Image ID").set_index("Image ID")
m["scan_date"] = pd.to_datetime(m.image_id.map(dti["Study Date"]), errors="coerce")
vend = m.image_id.map(dti["Imaging Protocol"]).astype(str).str.extract(r"Manufacturer=([A-Za-z]+)")[0]
m["vendor"] = vend.str.upper().map({"SIEMENS": "Siemens", "PHILIPS": "Philips", "GE": "GE"}).fillna("unrecorded")
m["phase"] = m.phase.astype(str).str.replace("ADNI ", "ADNI-")
m["adni34"] = m.phase.isin(["ADNI-3", "ADNI-4"]).astype(int)

# education
ed = am2("PTDEMOG")[["RID", "PTEDUCAT"]].copy()
ed["PTEDUCAT"] = pd.to_numeric(ed.PTEDUCAT, errors="coerce"); ed = ed[ed.PTEDUCAT > 0]
m["education"] = m.RID.map(ed.groupby("RID").PTEDUCAT.median())

# FastSurfer volumes
def aseg(path):
    v, etiv = {}, np.nan
    for line in open(path):
        if line.startswith("# Measure EstimatedTotalIntraCranialVol"): etiv = float(line.split(",")[-2])
        if not line.startswith("#"):
            p = line.split()
            if len(p) >= 5: v[p[4]] = float(p[3])
    return etiv, v
def dkt(path, rois=("entorhinal", "parahippocampal")):
    out = {}
    for line in open(path):
        p = line.split()
        if p and not line.startswith("#") and p[0] in rois: out[p[0]] = float(p[3])   # GrayVol
    return out
rows = []
for sid, iid in zip(m.subject_id, m.image_id):
    cands = [c for c in sorted(FS.glob(f"{sid}_I*")) if (c / "stats/aseg.stats").is_file()]
    if not cands: rows.append({}); continue
    pref = [c for c in cands if c.name == f"{sid}_I{iid}"]   # the analysed session first
    d = (pref or cands)[0]; etiv, v = aseg(d / "stats/aseg.stats")
    lh = dkt(d / "stats/lh.aparc.DKTatlas.mapped.stats"); rh = dkt(d / "stats/rh.aparc.DKTatlas.mapped.stats")
    hip = v.get("Left-Hippocampus", np.nan) + v.get("Right-Hippocampus", np.nan)
    amy = v.get("Left-Amygdala", np.nan) + v.get("Right-Amygdala", np.nan)
    ctx = sum(lh.get(k, np.nan) + rh.get(k, np.nan) for k in ("entorhinal", "parahippocampal"))
    rows.append(dict(etiv=etiv, hippocampus=hip, mtl=hip + amy + ctx))
m = pd.concat([m.reset_index(drop=True), pd.DataFrame(rows)], axis=1)

# nearest-visit ADNIMERGE2 values
def nearest(t, col, dcol, window=180, nonneg=True):
    df = am2(t)[["RID", col, dcol]].copy(); df[col] = pd.to_numeric(df[col], errors="coerce")
    if nonneg: df.loc[df[col] < 0, col] = np.nan
    df[dcol] = pd.to_datetime(df[dcol], errors="coerce"); df = df.dropna()
    x = m[["subject_id", "RID", "scan_date"]].merge(df, on="RID", how="left")
    x["lag"] = (x[dcol] - x.scan_date).dt.days.abs()
    x = x.sort_values("lag").drop_duplicates("subject_id").set_index("subject_id")
    return x[col].where(x.lag <= window), x.lag.where(x.lag <= window)
m["wmh_cm3"], m["wmh_lag"] = [m.subject_id.map(s) for s in nearest("UCD_WMH", "TOTAL_WMH", "EXAMDATE")]
m["centiloid"], m["amy_lag"] = [m.subject_id.map(s) for s in nearest("UCBERKELEY_AMY_6MM", "CENTILOIDS", "SCANDATE", nonneg=False)]
st, _ = nearest("UCBERKELEY_AMY_6MM", "AMYLOID_STATUS", "SCANDATE")
m["amyloid_pos"] = m.subject_id.map(st)

# derived covariates
m["hippo_ratio"] = m.hippocampus / m.etiv * 1e3          # per mille of eTIV
m["mtl_ratio"] = m.mtl / m.etiv * 1e3
m["wmh_log"] = np.log(m.wmh_cm3 / (m.etiv / 1e3) * 100 + 0.01)   # log % of eTIV
m.to_csv(OUT / "covariates.csv", index=False)

print("n =", len(m))
print(m.groupby("group")[["age", "female", "education", "hippo_ratio", "mtl_ratio", "wmh_cm3", "centiloid"]].agg(["mean", "count"]).round(2).T.to_string())
print("\nvendor x phase:\n", pd.crosstab([m.phase, m.vendor], m.group).to_string())
print("\namyloid status within 180 d, ADNI-3/4:\n", pd.crosstab(m[m.adni34 == 1].group, m[m.adni34 == 1].amyloid_pos, margins=True).to_string())
