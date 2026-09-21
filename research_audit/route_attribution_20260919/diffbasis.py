import pandas as pd, numpy as np
d=pd.read_csv('per_subject_530.csv'); d['tagx']=d.tag.fillna('none')
def dbasis(r):
    if pd.isna(r.fa_mtime) or r.fa_mtime=='' : return 'NO_fa/md/ad/rd'
    if str(r.fa_mtime).startswith('2026-06-17T10'): return 'reprocess4: same 10M track+atlas, r4, SIFT-weighted'
    if not r.count_eq_fdsum: return 'legacy route (same basis as legacy count)'
    # fa written by convert_166 at 06-15 13:00 from SC_AAL_ fa
    if r.bk_sota_fa_mean=='diff': return 'cl-promote generated: 3M prod ACT track + SyN atlas, r4, unweighted'
    if r.bk_sota_fa_mean=='absent' and r.tagx=='none' and r.count_eq_fdsum and r.jacc_count_fa==1.0: return 'cl-promote generated (manifest row lost): 3M prod track + SyN atlas, r4'
    return 'legacy pre-SOTA fa (old atlas/route)'
d['diff_basis']=d.apply(dbasis,axis=1)
d['count_fa_consistent']=d.jacc_count_fa==1.0
d.to_csv('per_subject_530.csv',index=False)
print(pd.crosstab(d.diff_basis,d.count_fa_consistent,margins=True).to_string())
print(d.groupby('diff_basis').jacc_count_fa.describe().to_string())
# invlen consistency
d['count_invlen_consistent']=d.jacc_count_invlen==1.0
print(pd.crosstab(d.recipe.str[:30],d.count_invlen_consistent,margins=True).to_string())
