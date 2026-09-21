import pandas as pd,numpy as np,os
from concurrent.futures import ProcessPoolExecutor
d=pd.read_csv('attrib6.csv')
GAP=[34,35,80,81]; keep=[i for i in range(170) if i not in GAP]
def one(s):
    a=f'/data/derivatives/tracks/{s}/assignments_aal.csv'
    if not os.path.exists(a): return {'sid':s,'asg':'noasg'}
    x=np.loadtxt(a,comments='#',dtype=np.int64)
    if x.ndim!=2 or x.shape[1]!=2: return {'sid':s,'asg':'badshape'}
    n=170
    M=np.zeros((n+1,n+1))
    np.add.at(M,(x[:,0],x[:,1]),1)
    M=M+M.T; M=M[1:,1:]; np.fill_diagonal(M,0)
    if M.shape!=(170,170): return {'sid':s,'asg':'dim'}
    M=M[np.ix_(keep,keep)]
    q=np.loadtxt(f'/data/derivatives/connectomes/SC_AAL166_{s}_count.csv',delimiter=',')
    # also compare nonzero pattern
    same=np.allclose(M,q,rtol=1e-5,atol=1e-6)
    pat=((M>0)==(q>0)).all()
    return {'sid':s,'asg':'same' if same else ('pattern' if pat else 'diff'),'maxlab':int(x.max())}
with ProcessPoolExecutor(24) as ex: res=list(ex.map(one,d.sid))
d=d.merge(pd.DataFrame(res),on='sid')
d.to_csv('attrib7.csv',index=False)
print(pd.crosstab([d.cday,d.tag,d.fam.where(d.fam=='NO_MATCH','matched')],d.asg))
