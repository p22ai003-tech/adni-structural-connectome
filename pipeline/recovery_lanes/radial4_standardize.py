#!/usr/bin/env python
"""Phase 2: re-assign radial-2 recovery subjects (that still have a 10M track) at radial 4.
This (a) RECOVERS the 81 stranded good connectomes into production, and (b) standardizes those subjects
to radial 4. Monotonic in radius => never regresses. Backs up each old connectome first. Structural
4 weights (count/fd_sum/count_invnodevol/len_mean) — the ones that define density/count, matching how
recovery connectomes were built. Production (already radial 4) is untouched.
"""
import csv, glob, os, json, shutil, subprocess
import numpy as np
import concurrent.futures as cf

MR = "/home/ec2-user/mrtrix3/bin"
PROD = "/data/derivatives/connectomes"
BKP = "/data/derivatives/connectomes/_pre_radial4_backup"; os.makedirs(BKP, exist_ok=True)
AUDIT = "/data/derivatives/qc/sc_matrix_qc/subject_audit.csv"
SPECS = [("count", None, None), ("fd_sum", "sum", None),
         ("count_invnodevol", None, "invnodevol"), ("len_mean", "mean", "length")]

def dens(p):
    try:
        m = np.loadtxt(p, delimiter=","); n = m.shape[0]; o = m[~np.eye(n, dtype=bool)]
        return float((o > 0).sum() / o.size)
    except Exception:
        return None

def pick_run(sid):
    """run-root dir (highest result density) that has track + atlas + sift."""
    best = None; bd = -1
    for d in glob.glob(f"/data/derivatives/qc/sc_matrix_qc/*/{sid}/"):
        if not os.path.exists(d + "tracks_10M.tck"): continue
        atl = (d + "aal3_166_dwi_fresh.mif") if os.path.exists(d + "aal3_166_dwi_fresh.mif") \
            else ((d + "aal3_166_dwi_cached.mif") if os.path.exists(d + "aal3_166_dwi_cached.mif") else None)
        if not atl or not os.path.exists(d + "sift2_10M.txt"): continue
        rd = 0
        try: rd = (json.load(open(d + "result.json")).get("aal_qc") or {}).get("density") or 0
        except Exception: pass
        if rd >= bd: bd = rd; best = (d, atl)
    return best

# subjects to process: radial-2 with a surviving 10M track
rows = list(csv.DictReader(open(AUDIT)))
targets = []
for r in rows:
    if r["radial"] != "2": continue
    run = pick_run(r["sid"])
    if run: targets.append((r["sid"], run, float(r["density"] or 0)))
print(f"radial-2 subjects with usable track+atlas+sift: {len(targets)}")

def work(item):
    sid, (drr, atlas), old_d = item
    tck = drr + "tracks_10M.tck"; sift = drr + "sift2_10M.txt"
    # build to a temp count first; only commit if radial-4 >= current (never regress)
    tmp = f"/tmp/r4_{sid}_count.csv"
    subprocess.run([f"{MR}/tck2connectome", tck, atlas, tmp, "-symmetric", "-zero_diagonal",
                    "-assignment_radial_search", "4", "-tck_weights_in", sift, "-force"],
                   capture_output=True)
    nd = dens(tmp)
    if nd is None or nd < max(old_d, 0.0) - 1e-9:
        return (sid, old_d, nd, "skip(noregress)")
    # commit all 4 structural weights at radial 4
    for suf, stat, scale in SPECS:
        out = f"{PROD}/SC_AAL166_{sid}_{suf}.csv"
        if os.path.exists(out):
            b = f"{BKP}/{os.path.basename(out)}"
            if not os.path.exists(b): shutil.copy2(out, b)
        cmd = [f"{MR}/tck2connectome", tck, atlas, out, "-symmetric", "-zero_diagonal",
               "-assignment_radial_search", "4", "-tck_weights_in", sift, "-force"]
        if stat: cmd += ["-stat_edge", stat]
        if scale == "invnodevol": cmd += ["-scale_invnodevol"]
        elif scale == "length": cmd += ["-scale_length"]
        subprocess.run(cmd, capture_output=True)
    return (sid, old_d, nd, "PASS" if nd >= 0.6 else "still<0.6")

res = []
with cf.ThreadPoolExecutor(max_workers=10) as ex:
    for r in ex.map(work, targets): res.append(r)

newgood = sum(1 for _, o, n, t in res if (n or 0) >= 0.6 and o < 0.6)
stand = sum(1 for _, o, n, t in res if t == "PASS")
skip = sum(1 for _, o, n, t in res if t.startswith("skip"))
print(f"\nprocessed {len(res)} | newly-good (recovered, was <0.6 in prod): {newgood} | now>=0.6: {stand} | skipped(noregress): {skip}")
print("recovered sample:")
for sid, o, n, t in sorted(res, key=lambda x: -((x[2] or 0) - x[1]))[:8]:
    print(f"  {sid.split('_I')[0]:<14} prod {o:.3f} -> radial4 {(n or 0):.3f}  {t}")
