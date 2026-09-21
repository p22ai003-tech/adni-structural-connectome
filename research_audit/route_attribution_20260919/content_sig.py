import glob,os,re,csv,numpy as np,pandas as pd
import concurrent.futures as cf
C='/data/derivatives/connectomes'
d=pd.read_csv('per_subject_530.csv')
def sig(s):
    c=np.loadtxt(f'{C}/SC_AAL166_{s}_count.csv',delimiter=',')
    f=np.loadtxt(f'{C}/SC_AAL166_{s}_fd_sum.csv',delimiter=',')
    integer=bool(np.allclose(c,np.round(c),atol=1e-9))
    eqfd=bool(np.allclose(c,f,rtol=1e-4,atol=1e-9))
    nzc=(c>0); nzf=(f>0)
    samenz=bool((nzc==nzf).all())
    # total streamline sum in count (for integer counts ~ assigned streamlines)
    tot=float(np.triu(c,1).sum())
    tot_fd=float(np.triu(f,1).sum())
    return s,integer,eqfd,samenz,tot,tot_fd
res={}
with cf.ThreadPoolExecutor(24) as ex:
    for r in ex.map(sig,d.sid): res[r[0]]=r[1:]
d['count_integer']=d.sid.map(lambda s:res[s][0])
d['count_eq_fdsum']=d.sid.map(lambda s:res[s][1])
d['count_fd_same_nz']=d.sid.map(lambda s:res[s][2])
d['count_total']=d.sid.map(lambda s:res[s][3])
d['fd_total']=d.sid.map(lambda s:res[s][4])
d.to_csv('per_subject_530.csv',index=False)
d['tagx']=d.tag.fillna('none')
print(pd.crosstab(d.tagx,[d.count_integer,d.count_eq_fdsum]))
print(pd.crosstab(d.tagx,d.count_fd_same_nz))
