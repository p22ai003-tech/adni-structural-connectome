import json,re,sys
a,b=sys.argv[1],sys.argv[2]; lim=int(sys.argv[3]) if len(sys.argv)>3 else 4000
def m(s): 
    s=s if isinstance(s,str) else json.dumps(s)
    return re.sub(r'\d{3}_S_\d{4}','<S>',s)
for ln in open('events.jsonl'):
    d=json.loads(ln)
    if not (a<=d['ts']<=b): continue
    if d['kind']=='use':
        inp=d['input']; body=inp.get('command') if isinstance(inp,dict) and 'command' in inp else inp
        print(f"=== {d['ts'][:19]} USE {d['name']} :: {m(inp.get('description','') if isinstance(inp,dict) else '')}\n{m(body)[:lim]}")
    elif d['kind']=='result':
        print(f"--- {d['ts'][:19]} RESULT\n{m(d['text'] or '')[:lim]}")
    else:
        print(f"### {d['ts'][:19]} {d['role']} TEXT\n{m(d['text'] or '')[:lim]}")
