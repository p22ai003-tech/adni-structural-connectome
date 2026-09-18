#!/usr/bin/env python
# Single source of truth: computes full pipeline state + health -> /tmp/connectome_status.json
import json,glob,os,re,csv,time,subprocess,statistics
STATUS='/data/derivatives/qc/sc_matrix_qc/connectome_status.json'  # persistent (web app reads this)
TOTALS={'AD':100,'MCI':241,'CN':307}   # SMC (24) folded into MCI -> 217+24=241; total 648
GOODF={'AD':'/tmp/ad_good_density.txt','MCI':'/tmp/mci_good_density.txt','CN':'/tmp/cn_good_density.txt'}
G2={}
try:
    for r in csv.DictReader(open('/data/derivatives/sc_forge_v1/group_qc/current_connectome_subject_availability_with_group.csv')):
        g=r.get('group','?'); G2[r['sid']]='MCI' if g=='SMC' else g   # SMC counts as MCI
except Exception: pass

def best_density():
    # SINGLE SOURCE OF TRUTH: density from subject_manifest.csv (computed directly from the production
    # SC_AAL166 count.csv files). No result.json -> the count can't drift from what's on disk.
    # Re-run exp/pipeline/audit/subject_manifest.py whenever connectomes change.
    best={}
    af='/data/derivatives/qc/sc_matrix_qc/subject_manifest.csv'
    try:
        for r in csv.DictReader(open(af)):
            try: best[r['sid']]=float(r['density'] or 0)
            except: best[r['sid']]=0.0
    except Exception: pass
    return best

def alive(pat):
    return int(subprocess.run(['pgrep','-fc',pat],capture_output=True,text=True).stdout.strip() or 0)

def runalive(root):
    try: return len(subprocess.run(['pgrep','-f','run-root '+root],capture_output=True,text=True).stdout.split())
    except: return 0

while True:
    best=best_density()
    done=[s for s,d in best.items() if d>=0.6]
    ds=[best[s] for s in done]
    _ps=subprocess.run(['ps','-o','args=','-C','tckgen,dwi2fod,dwi2response,tcksift2,tck2connectome,mrtransform,5ttgen,mrgrid,dwi2mask'],capture_output=True,text=True).stdout
    running_all=set(re.findall(r'[0-9]{3}_S_[0-9]+_I[0-9]+',_ps))
    cohort={}
    for g,tot in TOTALS.items():
        gv=[(s,best[s]) for s in best if G2.get(s)==g]
        gd=[d for s,d in gv if d>=0.6]
        mid=len([d for s,d in gv if 0.4<=d<0.6])
        low=len([d for s,d in gv if 0<d<0.4])
        rung=len([s for s,d in gv if s in running_all])
        cohort[g]={'total':tot,'done':len(gd),
                   'mean_d':round(statistics.mean(gd),3) if gd else 0,
                   'median_d':round(statistics.median(gd),3) if gd else 0,
                   'mid':mid,'low':low,'running':rung,
                   'todo':max(tot-len(gd)-mid-low-rung,0)}
    # run roots
    runs={}
    for tag in ['regen216','recover2','gap11','ad_blitz','donorresp']:
        root=sorted((d for d in glob.glob(f'/data/derivatives/qc/sc_matrix_qc/{tag}*') if os.path.isdir(d)), key=os.path.getmtime)
        root=root[-1] if root else ''   # most-recent dir (not alphabetical: ad_blitz2_ vs ad_blitz_)
        rds=[]; n=0
        for rj in (glob.glob(root+'/*/result.json') if root else []):
            n+=1
            try: dd=(json.load(open(rj)).get('aal_qc') or {}).get('density') or 0
            except: continue
            if dd>=0.6: rds.append(dd)
        runs[tag]={'root':os.path.basename(root),'done':n,'alive':runalive(root) if root else 0,
                   'mean_d':round(statistics.mean(rds),3) if rds else 0,
                   'median_d':round(statistics.median(rds),3) if rds else 0,'ge06':len(rds)}
    loops={k:alive(v) for k,v in {'s3_backup':'s3_backup_all.sh'}.items()}  # promote_loop retired (writes deprecated 170)
    try: s3n=int(open('/tmp/s3_backup_counter.txt').read().strip())
    except: s3n=0
    la=float(open('/proc/loadavg').read().split()[0]); ncpu=os.cpu_count() or 1
    mem=subprocess.run(['free','-g'],capture_output=True,text=True).stdout.splitlines()
    ramfree=int(mem[1].split()[6]); ramtot=int(mem[1].split()[1])
    swapused=0
    try: swapused=int(mem[2].split()[2])
    except Exception: pass
    disk=subprocess.run(['df','-BG','/data'],capture_output=True,text=True).stdout.splitlines()[1].split()
    diskfree=int(disk[3].rstrip('G'))
    tmpused=0
    try: tmpused=int(subprocess.run(['df','-BG','/tmp'],capture_output=True,text=True).stdout.splitlines()[1].split()[2].rstrip('G'))
    except Exception: pass
    # ALERTS — project is COMPLETE (530/648, accepted). Only flag real host-health issues now;
    # no "compute stopped"/"loop down" noise (recovery runs + promote/s3_backup loops are intentionally retired).
    alerts=[]
    if diskfree<80: alerts.append(f'LOW DISK: {diskfree}G')
    if ramfree<25: alerts.append(f'LOW RAM: {ramfree}G')
    if tmpused>150: alerts.append(f'TMPFS HIGH {tmpused}G (RAM/OOM risk)')
    if swapused>20: alerts.append(f'SWAP HIGH {swapused}G')
    st={
      'ts':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
      'overall':{'done':len(done),'total':sum(TOTALS.values()),'mean_d':round(statistics.mean(ds),3) if ds else 0,'median_d':round(statistics.median(ds),3) if ds else 0},
      'cohort':cohort,'runs':runs,'loops':loops,
      's3_pushed':s3n,'s3_total':sum(TOTALS.values()),
      'host':{'load':round(la,1),'ncpu':ncpu,'load_ratio':round(la/ncpu,2),'ram_free':ramfree,'ram_total':ramtot,'disk_free_g':diskfree,'tckgen':alive('tckgen'),'tmp_used_g':tmpused,'swap_used_g':swapused},
      'alerts':alerts,
    }
    try:
        tmp=STATUS+'.tmp'; open(tmp,'w').write(json.dumps(st,indent=2)); os.replace(tmp,STATUS)
    except Exception as e:
        pass
    # Priority policy: AD blitz keeps its launch nice (0) so it gets the box; every OTHER compute
    # proc is pushed to nice 19 (below the PE app on 8502/8503, which stays responsive).
    try:
        pids=subprocess.run(['pgrep','-f','tckgen|dwi2fod|dwi2response|tcksift2|tck2connectome|mrtransform|5ttgen|mrgrid|dwi2mask|antsReg|run_sc_route'],capture_output=True,text=True).stdout.split()
        for p in pids:
            try: cl=open(f'/proc/{p}/cmdline','rb').read().decode('utf-8','ignore')
            except Exception: continue
            if 'ad_blitz' in cl: continue   # AD priority -> leave at nice 0
            subprocess.run(['renice','-n','19','-p',p],capture_output=True)
    except Exception: pass
    time.sleep(20)
