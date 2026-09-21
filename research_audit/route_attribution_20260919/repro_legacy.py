import pandas as pd, numpy as np, subprocess, os, sys, concurrent.futures as cf, csv
x=pd.read_csv('legacy53.csv')
MR='/home/ec2-user/mrtrix3/bin'
GAP=[34,35,80,81]; keep=[k for k in range(170) if k not in GAP]
VARIANTS=[('forward40',['-assignment_forward_search','40']),('forward80',['-assignment_forward_search','80']),('forward120',['-assignment_forward_search','120']),('radial4',['-assignment_radial_search','4']),('radial8',['-assignment_radial_search','8']),('default',[])]
out=open('repro_legacy_results.csv','w'); w=csv.writer(out); w.writerow(['idx','parc_mtime','variant','equal','corr','dens_pub','dens_rep']); out.flush()
def run(i,s):
    tck=f'/data/derivatives/tracks/{s}/tracks_final_3000k.tck'; parc=f'/data/derivatives/parc/{s}/AAL_b0.nii.gz'
    pub=np.loadtxt(f'/data/derivatives/connectomes/SC_AAL166_{s}_count.csv',delimiter=',')
    iu=np.triu_indices(166,1); res=[]
    for name,opt in VARIANTS:
        f=f'repro/L{i}_{name}.csv'
        subprocess.run([f'{MR}/tck2connectome',tck,parc,f,'-symmetric','-zero_diagonal',*opt,'-stat_edge','sum','-nthreads','6','-force','-quiet'],capture_output=True)
        try:
            m=np.loadtxt(f,delimiter=',')
            m=np.pad(m,((0,max(0,170-m.shape[0])),(0,max(0,170-m.shape[0]))))[:170,:170][np.ix_(keep,keep)]
        except Exception: res.append((name,'ERR',None,None,None)); continue
        eq=bool(np.array_equal(m,pub)); c=float(np.corrcoef(m[iu],pub[iu])[0,1])
        dn=lambda a:(a[~np.eye(166,dtype=bool)]>0).mean()
        res.append((name,eq,round(c,5),round(dn(pub),4),round(dn(m),4)))
        os.remove(f)
        if eq: break
    return i,res
with cf.ThreadPoolExecutor(8) as ex:
    futs=[ex.submit(run,i,s) for i,s in enumerate(x.sid)]
    for fu in cf.as_completed(futs):
        i,res=fu.result()
        for r in res: w.writerow([i,x.parc_mtime[i],*r])
        out.flush()
print('DONE')
