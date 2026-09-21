import glob,os,re,csv,json,datetime,numpy as np
import concurrent.futures as cf
D='/data/derivatives'; C=D+'/connectomes'
BK=D+'/connectomes_backup_sota_20260612T082420Z'
GAP=[34,35,80,81]; keep=[i for i in range(170) if i not in GAP]
man={r['sid']:r for r in csv.DictReader(open(D+'/qc/sc_matrix_qc/claude_repaired_manifest.csv'))}
grp={}
for r in csv.DictReader(open(D+'/sc_forge_v1/group_qc/current_connectome_subject_availability_with_group.csv')):
    grp[r['sid']]=r.get('group','?')
subs=sorted({re.match(r'SC_AAL166_(.+_I\d+)_count\.csv',os.path.basename(f)).group(1) for f in glob.glob(C+'/SC_AAL166_*_count.csv')})
def ld(p):
    try: return np.loadtxt(p,delimiter=',')
    except Exception: return None
def dens(m):
    n=m.shape[0]; o=m[~np.eye(n,dtype=bool)]; return float((o>0).sum()/o.size)
def ts(p): return datetime.datetime.fromtimestamp(os.path.getmtime(p),datetime.UTC).strftime('%Y-%m-%dT%H:%M:%S')
def row(s):
    r={'sid':s,'group':grp.get(s,'?')}
    m=man.get(s); r['tag']=m['tag'] if m else ''; r['man_new_density']=m['new_density'] if m else ''; r['promoted_utc']=m['promoted_utc'] if m else ''
    r['weights_written']=m['weights_written'] if m else ''
    cnt=ld(f'{C}/SC_AAL166_{s}_count.csv'); r['density']=round(dens(cnt),4)
    r['count_mtime']=ts(f'{C}/SC_AAL166_{s}_count.csv')
    fp=f'{C}/SC_AAL166_{s}_fa_mean.csv'
    if os.path.exists(fp):
        fa=ld(fp); r['fa_mtime']=ts(fp); r['fa_density']=round(dens(fa),4)
        a=cnt>0; b=fa>0; np.fill_diagonal(a,False); np.fill_diagonal(b,False)
        r['jacc_count_fa']=round(float((a&b).sum()/max((a|b).sum(),1)),4)
    else: r['fa_mtime']='';r['fa_density']='';r['jacc_count_fa']=''
    ip=f'{C}/SC_AAL166_{s}_invlen_mean.csv'
    if os.path.exists(ip):
        iv=ld(ip); a=cnt>0; b=iv>0; np.fill_diagonal(a,False); np.fill_diagonal(b,False)
        r['jacc_count_invlen']=round(float((a&b).sum()/max((a|b).sum(),1)),4); r['invlen_mtime']=ts(ip)
    # len_mean vs count
    lp=f'{C}/SC_AAL166_{s}_len_mean.csv'; lm=ld(lp); a=cnt>0; b=lm>0; np.fill_diagonal(a,False); np.fill_diagonal(b,False)
    r['jacc_count_len']=round(float((a&b).sum()/max((a|b).sum(),1)),4)
    # backups
    for tag,base in [('bk_sota',BK)]:
        for w in ['count','fa_mean']:
            p=f'{base}/SC_AAL_{s}_{w}.csv'
            if os.path.exists(p):
                bm=ld(p)
                if bm is not None and bm.shape==(170,170):
                    bm=bm[np.ix_(keep,keep)]
                    cur=cnt if w=='count' else (ld(fp) if os.path.exists(fp) else None)
                    r[f'{tag}_{w}']='same' if cur is not None and np.allclose(bm,cur,rtol=1e-5,atol=1e-9) else 'diff'
                    r[f'{tag}_{w}_mtime']=ts(p)
                    if w=='count': r['bk_count_density']=round(dens(bm),4)
                else: r[f'{tag}_{w}']='badshape'
            else: r[f'{tag}_{w}']='absent'
    for bd in ['_pre_recover_backup','_pre_radial4_backup','_pre_ad3_backup','_pre_reprocess4_backup','_lowband_archive','_midband']:
        r['in'+bd]=int(os.path.exists(f'{C}/{bd}/SC_AAL166_{s}_count.csv'))
    r['n_types']=len(glob.glob(f'{C}/SC_AAL166_{s}_*.csv'))
    return r
rows=[]
with cf.ThreadPoolExecutor(24) as ex:
    for r in ex.map(row,subs): rows.append(r)
keys=[]
for r in rows:
    for k in r:
        if k not in keys: keys.append(k)
w=csv.DictWriter(open('per_subject_530.csv','w'),fieldnames=keys); w.writeheader()
for r in rows: w.writerow(r)
print(len(rows))
