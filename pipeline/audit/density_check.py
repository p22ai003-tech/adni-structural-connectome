#!/usr/bin/env python
# Live connectome density (EC2 vs S3) + file inventory (tck/tsf/csv EC2 vs S3) per cohort.
# Usage: density_check.py [all|overall|AD|MCI|CN]  -> merges into /tmp/density_check.json
import json, glob, os, subprocess, tempfile, statistics, time, sys, csv, re
from collections import defaultdict
import numpy as np

TARGET = sys.argv[1] if len(sys.argv) > 1 else "all"
OUT = "/tmp/density_check.json"
G2 = {}
for r in csv.DictReader(open("/data/derivatives/sc_forge_v1/group_qc/current_connectome_subject_availability_with_group.csv")):
    g = r.get("group", "?"); G2[r["sid"]] = "MCI" if g == "SMC" else g
# best (current) density per subject -> inventory counts only GOOD (>=0.6) subjects' files
BEST = {}
for _f in ["/tmp/ad_good_density.txt", "/tmp/mci_good_density.txt", "/tmp/cn_good_density.txt"]:
    if os.path.exists(_f):
        for _ln in open(_f):
            _p = _ln.split()
            if len(_p) >= 2:
                try: BEST[_p[0]] = max(BEST.get(_p[0], 0), float(_p[1]))
                except Exception: pass
for _rf in glob.glob("/tmp/lane[0-9]_run_root.txt") + glob.glob("/tmp/lane[0-9]_mcicn_run_root.txt"):
    try: _rt = open(_rf).read().strip()
    except Exception: continue
    for _rj in glob.glob(_rt + "/*/result.json"):
        _sid = os.path.basename(os.path.dirname(_rj))
        try: _d = (json.load(open(_rj)).get("aal_qc") or {}).get("density") or 0
        except Exception: continue
        BEST[_sid] = max(BEST.get(_sid, 0), _d)

def sid_of(fn):
    m = re.search(r"[0-9]{3}_S_[0-9]+_I[0-9]+", fn); return m.group(0) if m else None

def density(path):
    try:
        m = np.loadtxt(path, delimiter=",")
        if m.ndim != 2 or m.shape[0] < 2: return None
        n = m.shape[0]; off = m[~np.eye(n, dtype=bool)]; return float((off > 0).sum() / off.size)
    except Exception:
        return None

def st_(ds):
    ds = [d for d in ds if d is not None]
    return {"n": len(ds), "mean": round(statistics.mean(ds), 3) if ds else 0,
            "median": round(statistics.median(ds), 3) if ds else 0}

def want(g):
    return TARGET in ("all", "overall", "regen") or g == TARGET

REGEN = set(l.strip() for l in open("/tmp/regen216_subs.txt") if l.strip()) if os.path.exists("/tmp/regen216_subs.txt") else set()
open("/tmp/density_check.status", "w").write("running")
out = subprocess.run(["aws","s3","ls","s3://sabeesh/exp/tracks/","--recursive"], capture_output=True, text=True).stdout

# ---- density EC2 (local) + S3 (download count.csv) ----
ec2 = defaultdict(list); s3 = defaultdict(list); tmp = tempfile.mkdtemp()
for f in glob.glob("/data/derivatives/connectomes/*count.csv"):
    if "_invnodevol" in f: continue
    _sid = sid_of(os.path.basename(f)); g = G2.get(_sid)
    if not g or not want(g): continue
    d = density(f)
    if d is not None:
        ec2[g].append(d); ec2["overall"].append(d)
        if _sid in REGEN: ec2["regen"].append(d)
for l in out.splitlines():
    key = l.split()[-1] if l.split() else ""
    if not (key.endswith("count.csv") and "_invnodevol" not in key): continue
    _sidk = sid_of(os.path.basename(key)); g = G2.get(_sidk)
    if not g or not want(g): continue
    lp = os.path.join(tmp, os.path.basename(key))
    if subprocess.run(["aws","s3","cp",f"s3://sabeesh/{key}",lp], capture_output=True).returncode == 0:
        d = density(lp)
        if d is not None:
            s3[g].append(d); s3["overall"].append(d)
            if _sidk in REGEN: s3["regen"].append(d)
        try: os.remove(lp)
        except: pass

# ---- file inventory: subjects with tck/tsf/csv in EC2 vs S3 (always full) ----
s3has = defaultdict(set)
for l in out.splitlines():
    key = l.split()[-1] if l.split() else ""; sid = sid_of(key)
    if not sid: continue
    fn = key.split("/")[-1]
    if fn.endswith(".tck"): s3has[sid].add("tck")
    elif fn.endswith(".tsf"): s3has[sid].add("tsf")
    elif fn.endswith(".csv"): s3has[sid].add("csv")
    elif "sift_weights" in fn: s3has[sid].add("sift")
REGEN = set(l.strip() for l in open("/tmp/regen216_subs.txt") if l.strip()) if os.path.exists("/tmp/regen216_subs.txt") else set()
inv = defaultdict(lambda: defaultdict(lambda: {"ec2": 0, "s3": 0}))
for sid, g in G2.items():
    if BEST.get(sid, 0) < 0.6: continue   # GOOD files only (not the existing April baseline)
    e = {"tck": bool(glob.glob(f"/data/derivatives/tracks/{sid}/*.tck")) or bool(glob.glob(f"/data/derivatives/qc/sc_matrix_qc/*/{sid}/*.tck")),
         "tsf": bool(glob.glob(f"/data/derivatives/tracks/{sid}/*.tsf")),
         "csv": bool(glob.glob(f"/data/derivatives/connectomes/*{sid}*.csv")),
         "sift": bool(glob.glob(f"/data/derivatives/tracks/{sid}/sift_weights.txt"))}
    groups = [g, "overall"] + (["regen"] if sid in REGEN else [])
    for typ in ("tck", "tsf", "csv", "sift"):
        if e[typ]:
            for gg in groups: inv[gg][typ]["ec2"] += 1
        if typ in s3has.get(sid, ()):
            for gg in groups: inv[gg][typ]["s3"] += 1

res = {}
try: res = json.load(open(OUT))
except Exception: pass
res["ts"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
for g in ["overall", "AD", "MCI", "CN"]:
    row = res.get(g, {})
    if want(g):
        row["ec2"] = st_(ec2[g]); row["s3"] = st_(s3[g])
    row["files"] = {t: dict(inv[g][t]) for t in ("tck", "tsf", "csv", "sift")}
    res[g] = row
res["regen"] = {"ec2": st_(ec2["regen"]), "s3": st_(s3["regen"]),
                "files": {t: dict(inv["regen"][t]) for t in ("tck", "tsf", "csv", "sift")}}
json.dump(res, open(OUT, "w"), indent=2)
open("/tmp/density_check.status", "w").write("done")
print(json.dumps(res))
