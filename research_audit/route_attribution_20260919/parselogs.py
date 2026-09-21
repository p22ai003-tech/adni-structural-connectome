import re,glob,os,json
rows=[]
for f in glob.glob('/data/derivatives/qc/sc_matrix_qc/*.log'):
    root=None
    for ln in open(f,errors='replace'):
        m=re.search(r'SOTA route start \| mode=(\S+) \| n=(\d+) \| workers=(\d+) x (\d+) threads \| run_root=(\S+)',ln)
        if m: root=m.group(5); mode=m.group(1); continue
        m=re.match(r'^\[(\S+)\] (\d{3}_S_\d{4}_I\d+)\s+(\S+)\s+ncc=(\S+) surv=(\S+)/166 cand\[z=(\S+) d=(\S+)\] base\[z=(\S+) d=(\S+)\] -> (\S+) ?(.*)$',ln.rstrip())
        if m:
            ts,sid,st,ncc,surv,cz,cd,bz,bd,flag,rest=m.groups()
            rows.append(dict(log=os.path.basename(f),root=os.path.basename(root) if root else None,mode=mode if root else None,ts=ts,sid=sid,status=st,ncc=ncc,surv=surv,cz=cz,cd=cd,bz=bz,bd=bd,flag=flag,rest=rest[:200]))
json.dump(rows,open('logrows.json','w'))
print(len(rows))
