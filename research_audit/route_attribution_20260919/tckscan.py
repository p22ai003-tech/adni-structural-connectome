import glob, os, re, subprocess, json, datetime as dt
from concurrent.futures import ThreadPoolExecutor
PROD = sorted(set(re.match(r'SC_AAL166_(\d{3}_S_\d{4}_I\d+)_count\.csv', os.path.basename(p)).group(1)
                  for p in glob.glob('/data/derivatives/connectomes/SC_AAL166_*_count.csv')))
def info(sid):
    t = f'/data/derivatives/tracks/{sid}/tracks_final_3000k.tck'
    r = {'sid': sid, 'exists': os.path.exists(t)}
    if not r['exists']: return r
    st = os.stat(t); r['size'] = st.st_size; r['mtime'] = dt.datetime.fromtimestamp(st.st_mtime, dt.UTC).strftime('%Y-%m-%dT%H:%M')
    out = subprocess.run(['/home/ec2-user/mrtrix3/bin/tckinfo', t], capture_output=True, text=True, timeout=120).stdout
    kv = {}
    for ln in out.splitlines():
        m = re.match(r'\s{4}(\S[^:]*):\s+(.*)$', ln)
        if m: kv.setdefault(m.group(1).strip(), m.group(2).strip())
    r.update({k: kv.get(k) for k in ('count','max_num_tracks','method','act','backtrack','crop_at_gmwmi','source','mrtrix_version','threshold','command_history','ROI','timestamp','total_count','max_num_seeds')})
    return r
with ThreadPoolExecutor(16) as ex: res = list(ex.map(info, PROD))
json.dump(res, open('/tmp/claude-1000/-home-ec2-user/e26ef3c3-fd45-4c91-8237-f60f309e1a06/scratchpad/legacy/tckscan.json','w'), indent=1)
print(len(res))
