import pandas as pd,numpy as np,os
from concurrent.futures import ProcessPoolExecutor
C='/data/derivatives/connectomes'
BK={'sf0525':'/data/derivatives/connectomes_backup_scforge_v1_20260525T021232Z','sf0527':'/data/derivatives/connectomes_backup_scforge_v1_20260527T024931Z','sota0612':'/data/derivatives/connectomes_backup_sota_20260612T082420Z'}
GAP=[34,35,80,81]; keep=[i for i in range(170) if i not in GAP]
d=pd.read_csv('attrib5.csv')
def one(s):
    r={'sid':s}
    for t in ['count','fd_sum','invlen_mean','fa_mean']:
        qp=f'{C}/SC_AAL166_{s}_{t}.csv'
        if not os.path.exists(qp): r['missing_'+t]=1; continue
        q=np.loadtxt(qp,delimiter=',')
        for k,b in BK.items():
            p=f'{b}/SC_AAL_{s}_{t}.csv'
            if not os.path.exists(p): r[f'{k}_{t}']='-'; continue
            m=np.loadtxt(p,delimiter=',')
            if m.shape!=(170,170): r[f'{k}_{t}']='shape'; continue
            m=m[np.ix_(keep,keep)]
            r[f'{k}_{t}']='same' if np.allclose(m,q,rtol=1e-4,atol=1e-6) else 'diff'
    return r
with ProcessPoolExecutor(32) as ex: res=list(ex.map(one,d.sid))
d=d.merge(pd.DataFrame(res),on='sid')
d.to_csv('attrib6.csv',index=False)
