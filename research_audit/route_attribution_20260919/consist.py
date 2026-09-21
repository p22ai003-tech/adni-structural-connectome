import glob, os, re, json, numpy as np, datetime as dt
from concurrent.futures import ProcessPoolExecutor
C='/data/derivatives/connectomes'
SIDS=sorted(set(re.match(r'SC_AAL166_(\d{3}_S_\d{4}_I\d+)_count\.csv',os.path.basename(p)).group(1) for p in glob.glob(C+'/SC_AAL166_*_count.csv')))
T=['count','fd_sum','count_invnodevol','len_mean','invlen_mean','fa_mean','md_mean','rd_mean','ad_mean']
def one(sid):
    r={'sid':sid}
    nz={}
    for t in T:
        p=f'{C}/SC_AAL166_{sid}_{t}.csv'
        if not os.path.exists(p): continue
        m=np.loadtxt(p,delimiter=','); np.fill_diagonal(m,0)
        nz[t]=m>0
        r['mt_'+t]=dt.datetime.fromtimestamp(os.path.getmtime(p),dt.UTC).strftime('%m-%dT%H:%M')
        if t=='count':
            r['density']=float(nz[t][~np.eye(166,dtype=bool)].mean()); r['total']=float(m.sum()/2)
            r['is_int']=bool(np.allclose(m,np.round(m)))
            r['sym']=bool(np.allclose(m,m.T))
        if t=='fd_sum': r['fd_total']=float(m.sum()/2)
        if t=='len_mean': 
            r['len_med']=float(np.median(m[m>0])) if (m>0).any() else 0
    for t in T[1:]:
        if t in nz: r['jac_'+t]=float((nz[t]&nz['count']).sum()/max(1,(nz[t]|nz['count']).sum()))
    return r
with ProcessPoolExecutor(32) as ex: res=list(ex.map(one,SIDS,chunksize=4))
json.dump(res,open('/tmp/claude-1000/-home-ec2-user/e26ef3c3-fd45-4c91-8237-f60f309e1a06/scratchpad/legacy/consist.json','w'))
print(len(res))
