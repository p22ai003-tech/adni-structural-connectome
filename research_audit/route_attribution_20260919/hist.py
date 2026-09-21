import re, subprocess, sys, json
from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
D=Path('/data/derivatives'); MR='/home/ec2-user/mrtrix3/bin/mrinfo'
conn=sorted({re.match(r'SC_AAL166_(.*)_count\.csv',p.name).group(1) for p in (D/'connectomes').glob('SC_AAL166_*_count.csv')})
def hist(p):
    if not p.exists(): return None
    r=subprocess.run([MR,'-quiet','-property','command_history',str(p)],capture_output=True,text=True)
    return r.stdout.splitlines()
def norm(s):
    s=re.sub(r"\d{3}_S_\d{4,5}_I\d+","<SID>",s); s=re.sub(r"\d{3}_S_\d{4,5}","<SUBJ>",s); s=re.sub(r"I\d{5,}","I<ID>",s)
    s=re.sub(r"'?/[^ ']*/([^/ ']*)'?",r"\1",s); s=s.replace("Seagate Hub",""); s=re.sub(r"tmp[a-z0-9_]{8}","",s); s=s.replace("'","")
    return s
def one(sid):
    out={}
    e=hist(D/'eddy'/f'{sid}_preproc.mif') or []
    out['eddy_hist']=e
    dt=hist(D/'dti'/sid/'dt.mif')
    out['dt_hist']=dt
    out['fa_hist']=hist(D/'dti'/sid/'fa.mif')
    out['5tt_t1']=hist(D/'fod'/sid/'5tt_t1.mif')
    out['5tt_b0']=hist(D/'fod'/sid/'5tt_b0.mif')
    out['gmwmi']=hist(D/'fod'/sid/'gmwmi.mif')
    out['wmfod_final']=hist(D/'fod'/sid/'wmfod_final.mif')
    return sid,out
res={}
with ThreadPoolExecutor(24) as ex:
    for sid,o in ex.map(one,conn): res[sid]=o
json.dump(res,open(sys.argv[1],'w'))
print(len(res))
