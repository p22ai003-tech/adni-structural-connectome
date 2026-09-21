import json,pandas as pd,numpy as np,os
from concurrent.futures import ProcessPoolExecutor
C='/data/derivatives/connectomes'; B='/data/derivatives/connectomes_backup_sota_20260612T082420Z'
GAP=[34,35,80,81]; keep=[i for i in range(170) if i not in GAP]
d=pd.read_csv('attrib.csv')
def dens(m):
    n=m.shape[0]; o=m[~np.eye(n,dtype=bool)]; return float((o>0).sum()/o.size)
def one(s):
    r={'sid':s}
    p=np.loadtxt(f'{C}/SC_AAL166_{s}_count.csv',delimiter=',')
    r['dprod']=round(dens(p),4)
    for t in ['count','invlen_mean','fa_mean']:
        b=f'{B}/SC_AAL_{s}_{t}.csv'
        if os.path.exists(b):
            m=np.loadtxt(b,delimiter=',')
            if m.shape==(170,170):
                m=m[np.ix_(keep,keep)]
                q=np.loadtxt(f'{C}/SC_AAL166_{s}_{t}.csv',delimiter=',')
                r['bk_'+t]='same' if np.allclose(m,q,rtol=1e-4,atol=1e-6) else 'diff'
                if t=='count': r['dbk']=round(dens(m),4)
            else: r['bk_'+t]='shape'+str(m.shape)
        else: r['bk_'+t]='nobk'
    return r
with ProcessPoolExecutor(32) as ex: res=list(ex.map(one,d.sid))
d=d.merge(pd.DataFrame(res),on='sid')
d.to_csv('attrib2.csv',index=False)
