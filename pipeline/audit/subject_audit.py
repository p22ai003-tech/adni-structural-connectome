#!/usr/bin/env python
"""Single-source connectome audit -> subject_audit.csv.
Density is computed DIRECTLY from SC_AAL166_<sid>_count.csv (ground truth) — this replaces the
result.json-based count that drifted. One row per subject. Run again after radial-4 standardization.
"""
import glob, os, re, csv, json, subprocess
import numpy as np
import concurrent.futures as cf

DERIV = "/data/derivatives"
CONN = DERIV + "/connectomes"
MR = "/home/ec2-user/mrtrix3/bin"
OUT = DERIV + "/qc/sc_matrix_qc/subject_audit.csv"
MANIFEST = DERIV + "/qc/sc_matrix_qc/claude_repaired_manifest.csv"
GROUPCSV = DERIV + "/sc_forge_v1/group_qc/current_connectome_subject_availability_with_group.csv"

G2 = {}
for r in csv.DictReader(open(GROUPCSV)):
    g = r.get("group", "?"); G2[r["sid"]] = "MCI" if g == "SMC" else g
RECOVERED = set()  # recovery-promoted == built at radial 2
if os.path.exists(MANIFEST):
    RECOVERED = {r["sid"] for r in csv.DictReader(open(MANIFEST))}

def band(d):
    if d is None or d <= 0: return "none"
    if d >= 0.6: return "good"
    if d >= 0.4: return "mid"
    return "low"

def density(p):
    try:
        m = np.loadtxt(p, delimiter=","); n = m.shape[0]
        off = m[~np.eye(n, dtype=bool)]; return round(float((off > 0).sum() / off.size), 4)
    except Exception:
        return None

def mrcount(p):
    try:
        return int(float(subprocess.run([f"{MR}/mrstats", p, "-output", "count", "-mask", p],
                   capture_output=True, text=True, timeout=60).stdout.strip()))
    except Exception:
        try:
            return int(float(subprocess.run([f"{MR}/mrstats", p, "-output", "count"],
                       capture_output=True, text=True, timeout=60).stdout.strip()))
        except Exception:
            return None

# ---- S3 inventory (one listing each) -> per-subject file-type sets ----
def s3_sets():
    has = {}
    for pref in ("tracks/", "connectomes/"):
        out = subprocess.run(["aws", "s3", "ls", f"s3://sabeesh/exp/{pref}", "--recursive"],
                             capture_output=True, text=True).stdout
        for ln in out.splitlines():
            key = ln.split()[-1] if ln.split() else ""
            m = re.search(r"[0-9]{3}_S_[0-9]+_I[0-9]+", key)
            if not m: continue
            sid = m.group(0); fn = key.split("/")[-1]; s = has.setdefault(sid, set())
            if fn.endswith(".tck"): s.add("tck")
            elif fn.endswith(".tsf"): s.add("tsf")
            elif "sift_weights" in fn or "sift2" in fn: s.add("sift")
            elif fn.endswith(".csv") and "SC_AAL166_" in fn: s.add("csv")
    return has

print("listing S3 ...")
S3 = s3_sets()

def reco_track(sid):
    g = glob.glob(f"{DERIV}/qc/sc_matrix_qc/*/{sid}/tracks_10M.tck")
    return g[0] if g else ""

def best_result(sid):
    cands = glob.glob(f"{DERIV}/qc/sc_matrix_qc/*/{sid}/result.json")
    best = None; bd = -1
    for rj in cands:
        try: r = json.load(open(rj))
        except Exception: continue
        d = (r.get("aal_qc") or {}).get("density") or 0
        if d > bd: bd = d; best = r
    return best

def find_conn(sid):
    """count.csv lives in main (good) or _midband/_lowband_archive (quarantined)."""
    for sub in ("", "/_midband", "/_lowband_archive"):
        p = f"{CONN}{sub}/SC_AAL166_{sid}_count.csv"
        if os.path.exists(p): return p, sub
    return None, None

def audit_one(sid):
    g = G2.get(sid, "?")
    cnt, sub = find_conn(sid)
    d = density(cnt) if cnt else None
    base = f"{CONN}{sub}" if sub is not None else CONN
    nw = len([f for f in glob.glob(f"{base}/SC_AAL166_{sid}_*.csv")])
    prod_tck = f"{DERIV}/tracks/{sid}/tracks_final_3000k.tck"
    rtck = reco_track(sid)
    has_prod = os.path.exists(prod_tck)
    track_kind = "prod3M" if has_prod else ("reco10M" if rtck else "none")
    track_path = prod_tck if has_prod else rtck
    sift = (os.path.exists(f"{DERIV}/tracks/{sid}/sift_weights.txt") or
            bool(glob.glob(f"{DERIV}/qc/sc_matrix_qc/*/{sid}/sift2_10M.txt")))
    fod = bool(glob.glob(f"{DERIV}/fod/{sid}/wmfod*.mif"))
    # mask quality = mask_fod voxels / mask_5tt_brain voxels
    mfod = f"{DERIV}/fod/{sid}/mask_fod.mif"; m5tt = f"{DERIV}/fod/{sid}/mask_5tt_brain.mif"
    mv = mrcount(mfod) if os.path.exists(mfod) else None
    bv = mrcount(m5tt) if os.path.exists(m5tt) else None
    mq = round(mv / bv, 3) if (mv and bv) else ""
    radial = 2 if sid in RECOVERED else (4 if has_prod else "?")
    res = best_result(sid)
    ncc = ""; surv = ""; recipe = ""
    if res:
        ncc = res.get("reg_ncc", ""); surv = res.get("aal_label_survival", "")
        rc = res.get("reg_choice") or {}; reg = "fresh" if rc.get("fresh") else "cached"
        recipe = f"fod={res.get('fod','?')};reg={reg}"
    # EC2 file presence
    e_tck = has_prod or bool(rtck)
    e_tsf = bool(glob.glob(f"{DERIV}/tracks/{sid}/*.tsf"))
    e_csv = cnt is not None
    e_sift = sift
    s = S3.get(sid, set())
    insync = all((t in s) == ec for t, ec in
                 (("tck", e_tck), ("tsf", e_tsf), ("csv", e_csv), ("sift", e_sift)))
    return {
        "sid": sid, "group": g, "density": d if d is not None else "",
        "band": band(d), "radial": radial, "track": track_kind, "track_path": track_path,
        "n_weights": nw, "sift": int(e_sift), "fod": int(fod),
        "mask_voxels": mv if mv else "", "mask_quality": mq,
        "reg_ncc": ncc, "label_survival": surv, "best_recipe": recipe,
        "ec2_tck": int(e_tck), "ec2_tsf": int(e_tsf), "ec2_csv": int(e_csv), "ec2_sift": int(e_sift),
        "s3_tck": int("tck" in s), "s3_tsf": int("tsf" in s), "s3_csv": int("csv" in s), "s3_sift": int("sift" in s),
        "in_sync": int(insync),
    }

sids = sorted(G2)
print(f"auditing {len(sids)} subjects (mask mrstats in parallel)...")
rows = []
with cf.ThreadPoolExecutor(max_workers=16) as ex:
    for r in ex.map(audit_one, sids): rows.append(r)

cols = ["sid", "group", "density", "band", "radial", "track", "track_path", "n_weights",
        "sift", "fod", "mask_voxels", "mask_quality", "reg_ncc", "label_survival", "best_recipe",
        "ec2_tck", "ec2_tsf", "ec2_csv", "ec2_sift", "s3_tck", "s3_tsf", "s3_csv", "s3_sift", "in_sync"]
with open(OUT, "w", newline="") as h:
    w = csv.DictWriter(h, fieldnames=cols); w.writeheader()
    for r in sorted(rows, key=lambda x: (x["group"], -(x["density"] or 0) if isinstance(x["density"], float) else 0)):
        w.writerow(r)

# summary
from collections import Counter
bands = Counter((r["group"], r["band"]) for r in rows)
print(f"\nwrote {OUT}  ({len(rows)} subjects)")
for g in ["AD", "MCI", "CN"]:
    good = bands[(g, "good")]; mid = bands[(g, "mid")]; low = bands[(g, "low")]; non = bands[(g, "none")]
    print(f"  {g:<4} good={good:<4} mid={mid:<3} low={low:<3} none={non}")
tot_good = sum(1 for r in rows if r["band"] == "good")
print(f"  TOTAL good (>=0.6) = {tot_good}")
ns = sum(1 for r in rows if not r["in_sync"])
print(f"  out-of-sync EC2/S3: {ns}")
