#!/usr/bin/env python
# Simple LIVE monitor: total + per-group counts + a per-subject progress bar for everything running.
# Each running subject completes when its 10M tckgen hits the cap -> bar = elapsed/cap (predicts the count climb).
import json, time, subprocess, re, os, glob
S = "/data/derivatives/qc/sc_matrix_qc/connectome_status.json"
import csv
G2 = {}
for r in csv.DictReader(open("/data/derivatives/sc_forge_v1/group_qc/current_connectome_subject_availability_with_group.csv")):
    g = r.get("group", "?"); G2[r["sid"]] = "MCI" if g == "SMC" else g
s = json.load(open(S)); o = s["overall"]; pct = 100 * o["done"] // o["total"]
age = int(time.time() - time.mktime(time.strptime(s["ts"], "%Y-%m-%dT%H:%M:%SZ")))
bar = "#" * (pct * 30 // 100) + "." * (30 - pct * 30 // 100)
print("=" * 60)
print(f"  CONNECTOMES DONE: {o['done']}/{o['total']}  ({pct}%)   mean d {o['mean_d']}")
print(f"  [{bar}]")
print("=" * 60)
print(f"  {'group':<6}{'done':>9}{'running':>9}{'pending':>9}")
print("  " + "-" * 33)
for c in ["AD", "MCI", "CN"]:
    x = s["cohort"][c]
    print(f"  {c:<6}{str(x['done'])+'/'+str(x['total']):>9}{x['running']:>9}{x['total']-x['done']:>9}")
print("  " + "-" * 33)

# ---- per-subject progress for everything in flight ----
CAPS = {"fast1x96": 4200, "batch162": 4200, "freshreg": 4200, "fastcrash": 4200, "mcicn_recover2": 4200, "mcicn_recover": 6000, "ad_blitz": 6000, "donorresp": 6000, "regen216": 99999999, "adL7b": 6000, "adborder": 10800}
def capfor(root):
    for k, v in CAPS.items():
        if k in root: return v
    return 6000
# gather running tckgen: pid -> (etimes, root, sid)
rows = []
ps = subprocess.run(["ps", "-eo", "etimes=,args="], capture_output=True, text=True).stdout
for ln in ps.splitlines():
    ln = ln.strip()
    if "/tckgen " not in ln and not ln.split()[1:2] == ["tckgen"] and "tckgen" not in ln.split("/")[-1][:7]:
        if "bin/tckgen" not in ln: continue
    try: et = int(ln.split()[0])
    except: continue
    sid = re.search(r"[0-9]{3}_S_[0-9]+_I[0-9]+", ln)
    rt = re.search(r"sc_matrix_qc/([A-Za-z0-9_]+?_\d{4}T\d{6,}Z)", ln)
    if not (sid and rt): continue
    root = rt.group(1); cap = capfor(root)
    p = min(100, int(100 * et / cap))
    tck = glob.glob(f"/data/derivatives/qc/sc_matrix_qc/{root}/{sid.group(0)}/tracks*.tck")
    gb = (os.path.getsize(tck[0]) / 1e9) if tck else 0
    rows.append((p, et, sid.group(0), G2.get(sid.group(0), "?"), gb, root.split("_")[0]))
rows.sort(reverse=True)
print(f"  IN PROGRESS — {len(rows)} subjects tracking (bar = elapsed/cap; full bar -> finishes -> count climbs)")
for p, et, sid, coh, gb, run in rows[:30]:
    b = "#" * (p * 20 // 100) + "." * (20 - p * 20 // 100)
    short = sid.split("_I")[0]
    print(f"   {coh:<3} {short:<12} [{b}] {p:3d}%  {et//60:>2}min  {gb:4.1f}GB  {run}")
print("  " + "-" * 33)

# ---- HISTORY: recently completed subjects, wall time, and PASS/FAIL vs the 0.6 threshold ----
# getmtime is cheap (no read); only json.load the newest ~15 so the 15s refresh stays light.
allrj = glob.glob("/data/derivatives/qc/sc_matrix_qc/*/*/result.json")
allrj.sort(key=os.path.getmtime, reverse=True)
hrows = []
for rj in allrj[:15]:
    try:
        r = json.load(open(rj)); sid = os.path.basename(os.path.dirname(rj))
        sec = int(r.get("seconds") or 0); d = (r.get("aal_qc") or {}).get("density") or 0
        run = os.path.basename(os.path.dirname(os.path.dirname(rj))).split("_")[0]
    except Exception:
        continue
    hrows.append((sid, G2.get(sid, "?"), sec, d, run))
npass = sum(1 for _, _, _, d, _ in hrows if d >= 0.6)
print(f"  HISTORY — last {len(hrows)} completed (newest first)  |  PASS = density >= 0.6  |  {npass} PASS / {len(hrows)-npass} FAIL")
for i, (sid, coh, sec, d, run) in enumerate(hrows, 1):
    mark = "PASS ✓" if d >= 0.6 else "FAIL ✗"
    short = sid.split("_I")[0]; tm = f"{sec//60}m{sec%60:02d}s"
    print(f"   {i:>2}. {coh:<3} {short:<12} {tm:>8}  d={d:.3f}  {mark:<7} {run}")
print("  " + "-" * 33)
print(f"  load {s['host']['load']}/{s['host']['ncpu']} | data age {age}s {'LIVE' if age<60 else 'STALE'} | refresh 15s, Ctrl-C exit")
