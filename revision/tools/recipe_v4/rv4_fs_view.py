"""The recipe's own same-visit FastSurfer output for every session, as one folder of links (FREEZE.md, 6 Nov).
The chains' Phase 1 covariates read IIT_FASTSURFER/<unit>/stats/aseg+DKT.stats. /data/derivatives/fastsurfer holds the OLD runs, whose
T1 is another visit for 338 sessions and absent for 174: the recipe re-ran those on CPU into cohort/work/_shared/<unit>/fs/<unit>.
Here each unit links to the folder the recipe itself used, read from its QC record (stage s4a_fastsurfer):
  'stored, same T1'              -> /data/derivatives/fastsurfer/<unit>
  'run here on CPU, --seg_only'  -> cohort/work/_shared/<unit>/fs/<unit>
Refuses unless every ok session gets exactly one folder that has stats/aseg+DKT.stats. Prints counts only.
Usage: python -I rv4_fs_view.py <out dir (new)>"""
import glob, json, os, sys
from collections import Counter
from pathlib import Path

R4 = Path("/data/derivatives/recipe_v4"); B = R4 / "cohort"; OLD = Path("/data/derivatives/fastsurfer")
out = Path(sys.argv[1])
if out.exists(): raise SystemExit(f"{out} exists: make a new view")
links, n, bad = {}, Counter(), []
for f in sorted(glob.glob(str(B / "qc/C_*.json"))):
    if "/C_noact_" in f: continue
    d = json.load(open(f))
    if d.get("status") != "ok": n["not ok, skipped"] += 1; continue
    u = d["unit"]; s = (d.get("stages") or {}).get("s4a_fastsurfer") or {}; how = s.get("fastsurfer")
    src = OLD / u if how == "stored, same T1" else B / f"work/_shared/{u}/fs/{u}" if how == "run here on CPU, --seg_only" else None
    if src is None or not (src / "stats/aseg+DKT.stats").is_file(): bad.append(how); continue
    links[u] = src; n[how] += 1
if bad: raise SystemExit(f"{len(bad)} ok sessions without a usable FastSurfer folder ({Counter(bad)}); nothing written")
out.mkdir(parents=True)
for u, src in links.items(): os.symlink(src, out / u)
print(f"{len(links)} links in {out}: {dict(n)}")
