import json,re,sys,collections
sids=set(open('leg53.txt').read().split())
f=sys.argv[1]
hits=collections.defaultdict(list)
pat=re.compile(r'(\d{3}_S_\d{4}_I\d+)')
with open(f,errors='replace') as h:
    for ln in h:
        if 'promot' not in ln.lower(): continue
        try: d=json.loads(ln)
        except: continue
        p=d.get('payload',{})
        if p.get('type')!='function_call_output': continue
        o=p.get('output','')
        for m in pat.finditer(o):
            s=m.group(1)
            if s in sids:
                seg=o[max(0,m.start()-200):m.end()+300]
                if 'promot' in seg.lower() or 'route' in seg.lower():
                    hits[s].append((d.get('timestamp','')[:16], re.sub(r'\d{3}_S_\d{4}','<S>',seg).replace('\n',' | ')[:500]))
print(len(hits))
json.dump(hits,open('codexhits.json','w'))
