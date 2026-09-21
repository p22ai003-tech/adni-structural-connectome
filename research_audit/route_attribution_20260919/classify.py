import pandas as pd, re
d=pd.read_csv('per_subject_530.csv'); d['tagx']=d.tag.fillna('none')
def fam(x):
    if not isinstance(x,str) or not x: return ''
    return ';'.join(sorted({re.sub(r'_?(\d{8}T\d{6}Z|\d{4}T\d{6}Z|\d{6}Z)$','',p) for p in x.split(';') if p}))
d['fam']=d.log_match.map(fam)
def writer(r):
    cm=r.count_mtime
    if cm.startswith('2026-06-17T10'): return 'reprocess4_build(06-17)'
    if r.in_pre_ad3_backup: return 'ad3_build(06-17)'
    if '2026-06-17T04:07'<=cm<='2026-06-17T04:23' and r.in_pre_radial4_backup: return 'radial4_standardize(06-17)'
    if r.in_pre_recover_backup: return 'fastcopy_runroot(06-17)'
    if cm.startswith('2026-06-15T13:0'): return 'convert_166(06-15)'
    return 'OTHER'
d['final_writer']=d.apply(writer,axis=1)
# source recipe for structural weights
RECIPE={
 'route_sota_laneB_batch':'E3M_r4',  # existing 3M ACT prod track, SyN fullhead, cached BBR, radial4
 'route_sota_laneCD':'E3M_r4',
 'lane1_AD_keepbest':'T3M_r2_actiso_rebuild',
 'lane1_AD_fullhead3M':'T3M_r2_actiso_cachedfod',
 'lane5_AD_10Mmid':'T10M_r2_actiso_rebuild',
 'lane1_MCICN':'T3M_r2_rebuild',
 'base_all2':'T3M_r2_noact_rebuild','base_all2;regen216':'T3M_r2_noact_rebuild','regen216':'T3M_r2_noact_rebuild','gap11':'T3M_r2_noact_rebuild',
 'ad_L5':'T10M_r2_noact_rebuild','ad_L7':'T3M_r2_noact_rebuild_forceresp','recover':'T3M_r2_noact_rebuild_forceresp',
 'recover3_mid':'T10M_r2_noact_rebuild_forceresp',
 'mcicn_recover':'T10M_r2_noact_cachedfod','mcicn_recover2':'T10M_r2_noact_cachedfod_cap4200',
 'fastcrash':'T10M_r2_noact_rebuild_cap4200','batch162':'T10M_r2_noact_rebuild_rereg099_cap4200',
 'reprocess4':'T10M_r4_noact_rebuild_rereg099_cap5400',
}
def recipe(r):
    if not r.count_eq_fdsum: return 'LEGACY_preSOTA(3M ACT, unweighted count)'
    w=r.final_writer
    if w=='radial4_standardize(06-17)': return 'T10M_reassigned_r4(10M track from recovery root)'
    if w=='ad3_build(06-17)': return 'T10M_r2_(ad_blitz root track, re-assigned)'
    if w=='reprocess4_build(06-17)': return 'T10M_r4_noact_rebuild_rereg099_cap5400(fresh atlas)'
    if r.fam: return RECIPE.get(r.fam,'?'+r.fam)
    if r.tagx=='cl' and str(r.promoted_utc).startswith('20260612'): return 'T10M_r4_AD10M(ACTiso/noACTaniso) [no log; inferred]'
    return '?'
d['recipe']=d.apply(recipe,axis=1)
d.to_csv('per_subject_530.csv',index=False)
print(pd.crosstab(d.recipe,d.final_writer,margins=True).to_string())
