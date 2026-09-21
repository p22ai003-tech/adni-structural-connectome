import numpy as np,os,datetime as dt,collections
GAP=[34,35,80,81]; keep=[i for i in range(170) if i not in GAP]
C=collections.Counter()
for s in open('leg53.txt').read().split():
    pm=dt.datetime.fromtimestamp(os.path.getmtime(f'/data/derivatives/parc/{s}/AAL_b0.nii.gz'),dt.UTC).strftime('%m-%d')
    p=f'repro53/{s}_f40.csv'
    if not os.path.exists(p) or os.path.getsize(p)==0:
        p2=f'repro/{s}_f40.csv'
        p=p2 if os.path.exists(p2) and os.path.getsize(p2)>0 else None
    import time
    if p is None or time.time()-os.path.getmtime(p)<30: C[(pm,'pending')]+=1; continue
    try: r=np.loadtxt(p,delimiter=',')
    except Exception: C[(pm,'pending')]+=1; continue
    if r.shape==(170,170): r=r[np.ix_(keep,keep)]
    q=np.loadtxt(f'/data/derivatives/connectomes/SC_AAL166_{s}_count.csv',delimiter=',')
    C[(pm,'SAME' if r.shape==q.shape and np.allclose(r,q) else 'diff')]+=1
for k in sorted(C): print(k,C[k])
