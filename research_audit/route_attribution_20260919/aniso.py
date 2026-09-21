import pandas as pd,subprocess,os
from concurrent.futures import ThreadPoolExecutor
d=pd.read_csv('attrib7.csv')
MR='/home/ec2-user/mrtrix3/bin/mrinfo'
def sp(p):
    if not os.path.exists(p): return None
    o=subprocess.run([MR,p,'-spacing'],capture_output=True,text=True).stdout.split()
    try: return [float(x) for x in o[:3]]
    except: return None
def one(s):
    e=sp(f'/data/derivatives/eddy/{s}_preproc.mif'); f=sp(f'/data/derivatives/fod/{s}/wmfod_final.mif')
    an=lambda v: None if v is None else bool(v[2]>1.4*max(v[0],v[1]))
    return {'sid':s,'eddy_sp':e,'fod_sp':f,'aniso_eddy':an(e),'aniso_fod':an(f)}
with ThreadPoolExecutor(16) as ex: res=list(ex.map(one,d.sid))
r=pd.DataFrame(res); d=d.merge(r,on='sid'); d.to_csv('attrib8.csv',index=False)
print(pd.crosstab(d.aniso_eddy,d.aniso_fod,dropna=False))
