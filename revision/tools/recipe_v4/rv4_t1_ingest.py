"""Recipe v4: bring the same-visit T1s downloaded from IDA (3 Nov 2026) to the form the stored ones have, so every session's T1 went through one route.

Stored route (connectome_pipeline/t1_pipeline.py, April 2026): dcm2niix, mrconvert to t1_from_dicom_<subj>_I<id>.nii.gz, then
N4BiasFieldCorrection -d 3 -s 2 -b [200] -c [200x200x100x50,1e-6] to corrected_T1_<subj>_I<id>.nii.gz. Same here, written to
recipe_v4/t1_anat/ (the stored /data/derivatives/t1_anat is only read). IDA's processed ADNI-2 T1s (MT1 GradWarp N3m) arrive as NIfTI and skip dcm2niix.
No diagnosis is read: inputs are lists/t1_choice.csv and the downloaded images.

Usage: rv4_t1_ingest.py [--jobs 8] [--units FILE]   writes recipe_v4/t1_anat/ingest.csv (ingest_<FILE stem>.csv with --units); safe to re-run (finished images are skipped)
--units: only the sessions listed in FILE, one per line (the cohort driver calls it for its own slice before starting, so a machine never waits on T1s it does not need)
"""
import argparse, json, os, subprocess, tempfile, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np, pandas as pd, nibabel as nib

ap = argparse.ArgumentParser(); ap.add_argument("--jobs", type=int, default=8); ap.add_argument("--units", default=""); a = ap.parse_args()
os.nice(19)
R4 = Path("/data/derivatives/recipe_v4"); OUT = R4 / "t1_anat"; X = Path("/data/adni_downloads/t1_same_visit/extracted")
DCM = "/home/ec2-user/exp/.envs/dcm2niix-20260724/bin/dcm2niix"; N4 = "/home/ec2-user/exp/.envs/ants-2.6.5/bin/N4BiasFieldCorrection"; MR = "/home/ec2-user/mrtrix3/bin"
ENV = dict(os.environ, ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS="1", MRTRIX_TMPFILE_DIR=str(R4 / "scratch"))


def one(row):
    subj, iid = row["subj"], row["iid"]; base = f"{subj}_I{iid}"; raw, cor = OUT / f"t1_from_dicom_{base}.nii.gz", OUT / f"corrected_T1_{base}.nii.gz"
    info = dict(unit=row["unit"], subj=subj, t1_iid=iid, t1_type=row["t1_type"])
    try:
        src = row["src"]
        if len(src) != 1 or f"/{subj}/" not in str(src[0]): raise RuntimeError(f"image folder not found once under the right person: {len(src)}")
        if not cor.exists():
            with tempfile.TemporaryDirectory(dir=R4 / "scratch") as td:
                nii = sorted(src[0].glob("*.nii")) + sorted(src[0].glob("*.nii.gz"))
                if not nii:
                    subprocess.run([DCM, "-z", "y", "-b", "y", "-o", td, str(src[0])], check=True, capture_output=True, env=ENV)
                    nii = sorted(Path(td).glob("*.nii.gz"))
                if len(nii) != 1: raise RuntimeError(f"expected one image, got {len(nii)}: {[p.name for p in nii]}")
                subprocess.run([f"{MR}/mrconvert", str(nii[0]), str(raw), "-quiet", "-force"], check=True, capture_output=True, env=ENV)
                tmp = Path(td) / "cor.nii.gz"
                subprocess.run([N4, "-d", "3", "-i", str(raw), "-o", f"[{tmp},{Path(td) / 'bias.nii.gz'}]", "-s", "2", "-b", "[200]", "-c", "[200x200x100x50,1e-6]"],
                               check=True, capture_output=True, env=ENV)
                os.replace(tmp, cor)                    # appears only when complete
        im = nib.load(str(cor)); info.update(status="ok", shape=list(im.shape), vox=[round(float(v), 3) for v in im.header.get_zooms()[:3]])
        if im.ndim != 3 or min(im.shape) < 100 or max(info["vox"]) > 2.0: info["status"] = "check"   # a T1 should be one 3D volume at about 1 mm
    except Exception as e:
        info.update(status="failed", error=f"{type(e).__name__}: {str(e)[:300]}")
    return info


t = pd.read_csv(R4 / "lists/t1_choice.csv"); t = t[(t.on_disk == False) & t.t1_iid.notna()].copy(); t["iid"] = t.t1_iid.astype(int).astype(str)
if a.units: t = t[t.unit.isin(open(a.units).read().split())]
idx = {}
for p in X.rglob("I*"):
    if p.is_dir(): idx.setdefault(p.name[1:], []).append(p)
t["src"] = [idx.get(i, []) for i in t.iid]
first = set(open(R4 / "lists/canary_units.txt").read().split()) if (R4 / "lists/canary_units.txt").is_file() else set()
t = t.assign(k=~t.unit.isin(first)).sort_values("k", kind="stable").drop(columns="k")   # the canary's T1s first
OUT.mkdir(exist_ok=True); (R4 / "scratch").mkdir(exist_ok=True); t0 = time.time()
with ProcessPoolExecutor(a.jobs) as ex: res = list(ex.map(one, t.to_dict("records")))
d = pd.DataFrame(res); d.to_csv(OUT / (f"ingest_{Path(a.units).stem}.csv" if a.units else "ingest.csv"), index=False)
print(time.strftime("%F %T"), f"{len(d)} T1s in {round((time.time() - t0) / 60, 1)} min:", d.status.value_counts().to_dict(), flush=True)
