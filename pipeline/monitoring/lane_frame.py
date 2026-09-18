import json, glob, os, time, csv, re
import statistics as ST

GCSV="/data/derivatives/sc_forge_v1/group_qc/current_connectome_subject_availability_with_group.csv"
g2={}
try:
    for r in csv.DictReader(open(GCSV)): g2[r['sid']]=('MCI' if r['group']=='SMC' else r['group'])  # SMC counts as MCI
except: pass
def grp(s): return g2.get(s,'?')

# Accurate "running" detection: a subject is RUNNING iff a live mrtrix process is operating on it
# (tckgen buffers its file writes, so mtime-based checks miss subjects deep in a long tckgen).
import subprocess as _sp
def _running_by_root():
    # map each RUN-ROOT -> set of sids actively being processed IN THAT ROOT.
    # (a subject is "running in lane X" only if its live process operates inside lane X's root,
    #  so the same sid running in ad_L5 does NOT falsely show as running in the base root)
    try:
        out=_sp.run(["ps","-o","args=","-C","tckgen,dwi2fod,dwi2response,tcksift2,tck2connectome,mrtransform,5ttgen,mrgrid,dwi2mask"],
                    capture_output=True,text=True,timeout=10).stdout
    except Exception:
        out=""
    m={}
    for path,sid in re.findall(r'(\S*/([0-9]{3}_S_[0-9]+_I[0-9]+)/\S*)', out):
        i=path.find('/'+sid+'/')
        if i>0: m.setdefault(path[:i],set()).add(sid)
    return m
_RUNMAP=_running_by_root()

TOTALS  ={'AD':100,'MCI':241,'CN':307}
GOODFILE={'AD':'/tmp/ad_good_density.txt','MCI':'/tmp/mci_good_density.txt','CN':'/tmp/cn_good_density.txt'}
NEEDFILE={'AD':'/tmp/ad_needswork.txt','MCI':'/tmp/mci_needswork.txt','CN':'/tmp/cn_needswork.txt'}
ROOTPATS={'AD' :['/tmp/lane[0-9]_run_root.txt','/tmp/ad_blitz_root.txt','/tmp/donorresp_root.txt'],
          'MCI':['/tmp/lane[0-9]_mcicn_run_root.txt','/tmp/mci_lane[0-9]_run_root.txt','/tmp/donorresp_root.txt'],
          'CN' :['/tmp/lane[0-9]_mcicn_run_root.txt','/tmp/cn_lane[0-9]_run_root.txt','/tmp/donorresp_root.txt']}
SHORT={0:'exist-chk',1:'BASE-build',3:'donor-resp',4:'reg>base',5:'10M LAST',6:'mask>base',7:'FOD-regen',8:'exclude',9:'mfix>base'}
FOLDED={4,6,9}   # these fixes are now applied inside the BASE build -> not independent lanes
# order: existing, base, the live recovery lanes (FOD-regen -> 10M -> exclude), then folded-in
LANE_ORDER=[0,1,7,3,5,8,4,6,9]
GR="\033[1;32m"; BL="\033[1;34m"; CY="\033[1;36m"; DM="\033[90m"; YEL="\033[1;33m"; RS="\033[0m"
W=48                       # visible width of each table column
SEP=DM+" │ "+RS       # clean vertical separator

# ---- per-root index (read each subject dir ONCE per frame) ----
_CACHE={}
def index_root(root):
    if root in _CACHE: return _CACHE[root]
    rows=[]
    if root and os.path.isdir(root):
        for d in glob.glob(root+'/*'):
            if not os.path.isdir(d): continue
            sid=os.path.basename(d); rj=d+'/result.json'
            # Live mrtrix process is the AUTHORITY on "running": a subject being reprocessed in place
            # (RESUME_MIN_DENSITY) still has a STALE result.json until the new one overwrites it, so
            # "has result.json" != done. If a process is on it, it's RUNNING (and its stale result is
            # excluded from the bands until it finishes).
            if sid in _RUNMAP.get(root,set()):
                rows.append((sid,None,None,True)); continue
            if os.path.exists(rj):
                try:
                    r=json.load(open(rj)); st=r.get('status'); dd=(r.get('aal_qc') or {}).get('density')
                except: st='err'; dd=None
                rows.append((sid,st,dd,False))
            else:
                act=False
                for p in (d, d+'/tracks_10M.tck', d+'/sota.log', d+'/wmfod_rebuilt.mif'):
                    try:
                        if os.path.exists(p) and time.time()-os.path.getmtime(p)<180: act=True; break
                    except: pass
                rows.append((sid,None,None,act))
    _CACHE[root]=rows
    return rows

def roots_for(group):
    roots={}
    for pat in ROOTPATS[group]:
        for f in glob.glob(pat):
            bn=os.path.basename(f); m=re.search(r'lane(\d+)',bn)
            if m: n=int(m.group(1))
            elif 'blitz' in bn: n=5          # AD blitz = 10M run -> show under L5 (10M LAST)
            elif 'donorresp' in bn: n=3      # donor-response salvage lane -> L3
            else: continue
            v=open(f).read().strip()
            if v and os.path.isdir(v): roots[n]=v
    return roots

def gdens(group):
    gf=GOODFILE.get(group); out=[]
    if gf and os.path.exists(gf):
        for l in open(gf):
            p=l.split()
            if len(p)==2 and grp(p[0])==group:
                try: out.append((p[0],float(p[1])))
                except: pass
    return out

def scan(root,group):
    o=dict(run=0,d60=0,mid=0,brk=0,err=0,defer=0,crash=0,seen=0)
    for sid,st,dd,act in index_root(root):
        if grp(sid)!=group: continue
        o['seen']+=1
        if act: o['run']+=1; continue
        if st is None: continue
        if st=='ok' and dd is not None:
            if dd>=0.6: o['d60']+=1
            elif dd>=0.4: o['mid']+=1
            else: o['brk']+=1
        else:
            o['err']+=1
            if st=='deferred_slow': o['defer']+=1
            else: o['crash']+=1
    return o

def queue_count(n,group):
    cands=[f'/tmp/lane{n}_queue.txt'] if group=='AD' else [f'/tmp/{group.lower()}_lane{n}_queue.txt']
    for qf in cands:
        if os.path.exists(qf):
            raw=open(qf).read().split()
            if len(raw)==1 and raw[0].isdigit(): return int(raw[0])
            if raw: return len(raw)
    return None

def render_lines(group):
    total=TOTALS[group]; roots=roots_for(group); gds=gdens(group)
    bd={s:d for s,d in gds}
    for n,root in roots.items():
        for sid,st,dd,act in index_root(root):
            if grp(sid)==group and dd is not None: bd[sid]=max(bd.get(sid,0.0),dd)
    act=set(); stt={}
    for n,root in roots.items():
        for sid,st,dd,a in index_root(root):
            if grp(sid)!=group: continue
            if a: act.add(sid)
            if st is not None and (sid not in stt or st=='ok'): stt[sid]=st
    subs=set()
    nf=NEEDFILE.get(group)
    if nf and os.path.exists(nf): subs|=set(l.strip() for l in open(nf) if l.strip())
    subs|=set(s for s,_ in gds)
    if not subs: subs=set(bd)
    need=sum(1 for _ in open(nf)) if (nf and os.path.exists(nf)) else max(total-len(gds),0)
    done=sum(1 for d in bd.values() if d>=0.6); running=len(act); pending=max(total-done-running,0)
    PROG[group]=(done,total,[d for d in bd.values() if d>=0.6])
    dn=rn=mdd=lo=df=todo=0
    for s in subs:
        b=bd.get(s)
        if b is not None and b>=0.6: dn+=1
        elif s in act: rn+=1
        elif b is not None and 0.4<=b<0.6: mdd+=1
        elif b is not None and b<0.4: lo+=1
        elif stt.get(s)=='deferred_slow': df+=1
        else: todo+=1
    L=[]
    L.append(f"{CY}{group} — LANE CASCADE   {time.strftime('%H:%M:%S')}{RS}")
    L.append("="*W)
    L.append(f"{GR}OVERALL {group}:{total} DONE:{done} RUN:{running} PEND:{pending}{RS}")
    L.append(f"disp D:{dn} R:{rn} | mid:{mdd} <.4:{lo} def:{df} todo:{todo}")
    L.append("="*W)
    L.append(f"  {'lane':<13}{'pnd':>4}{'Run':>4}{'Don':>5}{'gp':>5}{'lp':>5}{'<.4':>5}{'er':>4}")
    L.append(f"{BL}{'-'*W}{RS}")
    routel1=''
    for n in LANE_ORDER:
        lab=SHORT[n]
        if n==0:
            ds=[d for _,d in gds]
            g60=sum(d>=0.6 for d in ds); g46=sum(0.4<=d<0.6 for d in ds); g4=sum(d<0.4 for d in ds)
            pend='-'; rnv='-'; done_=len(ds); err=0; mid=g46; brk=g4; gp=g60
        else:
            root=roots.get(n)
            if root is not None:
                sc=scan(root,group); done_=sc['d60']+sc['mid']+sc['brk']+sc['err']
                q=queue_count(n,group); base=q if q is not None else (need if n==1 else sc['seen'])
                pend=max(base-done_-sc['run'],0); rnv=sc['run']
                gp=sc['d60']; mid=sc['mid']; brk=sc['brk']; err=sc['err']
                if n==1: routel1=f"{sc['mid']+sc['brk']}>L5(10M-last) {sc['err']}>L7(FOD-regen)"
            else:
                q=queue_count(n,group)
                pend=(q if q is not None else '-'); rnv='-'; done_=gp=mid=brk=err=0
        _row=f"  {('L'+str(n)+' '+lab):<13}{pend:>4}{rnv:>4}{done_:>5}{gp:>5}{mid:>5}{brk:>5}{err:>4}"
        if n in FOLDED:
            L.append(f"{DM}{_row}{RS}")
        elif isinstance(rnv, int) and rnv > 0:
            L.append(f"\033[1;33m{_row}\033[0m")   # running lane row -> yellow
        else:
            L.append(_row)
    L.append(f"{BL}{'-'*W}{RS}")
    allds=[d for _,d in gds]
    for n,root in roots.items():
        for sid,st,dd,a in index_root(root):
            if grp(sid)==group and dd is not None: allds.append(dd)
    q=[d for d in allds if d<0.6]; gp=[d for d in allds if d>=0.6]
    md=[d for d in allds if 0.4<=d<0.6]; lod=[d for d in allds if d<0.4]
    mm=lambda xs,fn: f"{fn(xs):.2f}" if xs else "-"
    L.append(f"  {'mean d':<13}{mm(q,ST.mean):>4}{'-':>4}{mm(allds,ST.mean):>5}{mm(gp,ST.mean):>5}{mm(md,ST.mean):>5}{mm(lod,ST.mean):>5}{'':>4}")
    L.append(f"  {'median d':<13}{mm(q,ST.median):>4}{'-':>4}{mm(allds,ST.median):>5}{mm(gp,ST.median):>5}{mm(md,ST.median):>5}{mm(lod,ST.median):>5}{'':>4}")
    L.append(f"  {DM}route: {routel1}{RS}" if routel1 else "")
    L.append(f"{YEL}  Running: {running}{RS}")
    return L

def vlen(s): return len(re.sub(r'\033\[[0-9;]*m','',s))
def padv(s,w): return s+' '*max(0,w-vlen(s))

PROG={}
cols=[render_lines(g) for g in ('AD','MCI','CN')]
H=max(len(c) for c in cols)
for c in cols: c.extend(['']*(H-len(c)))
# ---- bold-yellow OVERALL banner FROM CANONICAL STATUS JSON (single source of truth) ----
import json as _json
_FW=W*3+6
_S=None
try:
    _S=_json.load(open('/data/derivatives/qc/sc_matrix_qc/connectome_status.json'))
    _td=_S['overall']['done']; _tt=_S['overall']['total']; _mean=_S['overall']['mean_d']; _med=_S['overall']['median_d']
    _coh=_S['cohort']
except Exception:
    _td=sum(p[0] for p in PROG.values()); _tt=sum(p[1] for p in PROG.values())
    _alld=sorted(d for p in PROG.values() for d in p[2]); _n=len(_alld)
    _mean=round(sum(_alld)/_n,3) if _n else 0; _med=round(_alld[_n//2],3) if _n else 0
    _coh={k:{'done':PROG[k][0],'total':PROG[k][1]} for k in PROG}
_pct=(100*_td/_tt) if _tt else 0
_fill=int(_pct/100*40); _bar='█'*_fill+'░'*(40-_fill)
_cohstr='  ·  '.join(f"{k} {v['done']}/{v['total']}" for k,v in _coh.items())
print(f"{YEL}{'='*_FW}{RS}")
print(f"{YEL}  ★ CONNECTOMES COMPLETE: {_td}/{_tt}  ({_pct:.0f}%)    {_cohstr}{RS}")
print(f"{YEL}  [{_bar}]   mean d {_mean:.3f}  ·  median d {_med:.3f}   (over {_td} done >=0.6){RS}")
print(f"{YEL}{'='*_FW}{RS}")
for i in range(H):
    print(SEP.join(padv(cols[k][i],W) for k in range(3)))
import subprocess
nt=subprocess.run("pgrep -c tckgen",shell=True,capture_output=True,text=True).stdout.strip()
load=open('/proc/loadavg').read().split()[0]
ncpu=os.cpu_count() or 1
ratio=float(load)/ncpu
sat=GR if ratio<1.0 else (YEL if ratio<1.3 else "\033[1;31m")
print(f"{DM}{'='*(W*3+6)}{RS}")
# --- regen + S3 backup + alerts (canonical), BELOW the tables ---
try:
    _rgdone=_S['runs']['regen216']['done']; _s3=_S['s3_pushed']; _s3t=_S['s3_total']; _al=_S.get('alerts',[])
    _s3fill=int((_s3/_s3t if _s3t else 0)*40); _s3bar='█'*_s3fill+'░'*(40-_s3fill)
    print(f"{YEL}  [{_s3bar}]  REGEN {_rgdone}/216 rebuilt  ·  S3-BACKUP {_s3}/{_s3t} pushed{RS}")
    if _al: print(f"\033[1;31m  !! ALERTS: {' | '.join(_al)}{RS}")
except Exception:
    pass
print(f"  active tckgen:{nt}  load(1m):{load}/{ncpu} ({sat}{ratio:.2f}x{RS})    (refresh 15s, Ctrl-C exit)")