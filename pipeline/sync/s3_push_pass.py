import json,glob,os,re,sys,subprocess
running=set(re.findall(r'[0-9]{3}_S_[0-9]+_I[0-9]+', sys.argv[1] if len(sys.argv)>1 else ''))
S3='s3://sabeesh/exp/tracks'
# best density per subject
best={}
for f in ['/tmp/ad_good_density.txt','/tmp/mci_good_density.txt','/tmp/cn_good_density.txt']:
    if os.path.exists(f):
        for ln in open(f):
            p=ln.split()
            if len(p)>=2:
                try: best[p[0]]=max(best.get(p[0],0),float(p[1]))
                except: pass
for rf in glob.glob('/tmp/lane[0-9]_run_root.txt')+glob.glob('/tmp/lane[0-9]_mcicn_run_root.txt'):
    for rj in glob.glob(open(rf).read().strip()+'/*/result.json'):
        sid=os.path.basename(os.path.dirname(rj))
        try: d=(json.load(open(rj)).get('aal_qc') or {}).get('density') or 0
        except: continue
        best[sid]=max(best.get(sid,0),d)
# regen targets
regen={}
if os.path.exists('/tmp/regen216_target.tsv'):
    for ln in open('/tmp/regen216_target.tsv'):
        a=ln.rstrip().split('\t')
        if len(a)>=2: regen[a[0]]=float(a[1])
rgroot=open('/tmp/regen216_root.txt').read().strip() if os.path.exists('/tmp/regen216_root.txt') else ''
pushed=set(l.strip() for l in open('/tmp/s3_pushed.list')) if os.path.exists('/tmp/s3_pushed.list') else set()
failed=set(l.split('\t')[0] for l in open('/tmp/regen216_failed.tsv')) if os.path.exists('/tmp/regen216_failed.tsv') else set()

def cp(tck,sid,name):
    r=subprocess.run(['aws','s3','cp',tck,f'{S3}/{sid}/{name}'],capture_output=True)
    return r.returncode==0

def bundle_aux(sid):
    # make the S3 subject folder self-contained: sift + tsf samples + connectome CSVs
    tdir=f'/data/derivatives/tracks/{sid}'
    for f in ['sift_weights.txt','fa_mean.tsf','md_mean.tsf','rd_mean.tsf','ad_mean.tsf','assignments_aal.csv','mu.txt']:
        p=f'{tdir}/{f}'
        if os.path.exists(p): subprocess.run(['aws','s3','cp',p,f'{S3}/{sid}/{f}'],capture_output=True)
    for c in glob.glob(f'/data/derivatives/connectomes/*{sid}*.csv'):
        subprocess.run(['aws','s3','cp',c,f'{S3}/{sid}/connectomes/{os.path.basename(c)}'],capture_output=True)

newpush=0
for sid,d in best.items():
    if d<0.6 or sid in pushed or sid in running: continue
    if sid in regen:
        # regenerated track must reproduce its density before archiving
        rj=f'{rgroot}/{sid}/result.json'
        if not os.path.exists(rj): continue
        try: nd=(json.load(open(rj)).get('aal_qc') or {}).get('density') or 0
        except: continue
        if nd<0.6 or nd<regen[sid]-0.07:
            if sid not in failed: open('/tmp/regen216_failed.tsv','a').write(f'{sid}\t{nd}\t{regen[sid]}\tneeds-recipe\n')
            continue
        tcks=glob.glob(f'{rgroot}/{sid}/tracks*.tck')
        if not tcks: continue
        if cp(tcks[0],sid,'tracks_final_3000k.tck'):   # REPLACE stale S3 track with the good regenerated one
            os.remove(tcks[0]); bundle_aux(sid); pushed.add(sid); newpush+=1
            open('/tmp/s3_pushed.list','w').write('\n'.join(sorted(pushed))+'\n'); open('/tmp/s3_backup_counter.txt','w').write(str(len(pushed)))
    else:
        # mappable / recovered: push production track (or run-root if present)
        tck=f'/data/derivatives/tracks/{sid}/tracks_final_3000k.tck'
        if not os.path.exists(tck):
            cands=glob.glob(f'/data/derivatives/qc/sc_matrix_qc/*/{sid}/tracks*.tck')
            if not cands: continue
            tck=cands[0]
        if cp(tck,sid,os.path.basename(tck)):
            bundle_aux(sid); pushed.add(sid); newpush+=1
            open('/tmp/s3_pushed.list','w').write('\n'.join(sorted(pushed))+'\n'); open('/tmp/s3_backup_counter.txt','w').write(str(len(pushed)))
open('/tmp/s3_pushed.list','w').write('\n'.join(sorted(pushed))+'\n')
open('/tmp/s3_backup_counter.txt','w').write(f'{len(pushed)}')
print(f'pushed_total={len(pushed)} new_this_pass={newpush}')
