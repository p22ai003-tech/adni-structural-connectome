import re, json
from collections import Counter
R=json.load(open('/tmp/claude-1000/-home-ec2-user/e26ef3c3-fd45-4c91-8237-f60f309e1a06/scratchpad/legacy/hist530.json'))
C=Counter(); E=Counter(); V=Counter(); DN=Counter(); GB=Counter()
for sid,o in R.items():
    e=o['eddy_hist'] or []
    # take last chain: find last dwifslpreproc and everything before since last raw mrconvert
    idx=[i for i,x in enumerate(e) if re.match(r"(\S*/)?mrconvert ",x) and ('-export_grad_fsl' in x) ]
    start=idx[-1] if idx else 0
    ch=e[start:]
    has=lambda t: any(re.match(rf"(\S*/)?{t} ",x) for x in ch)
    if any('dwi_post_eddy' in x for x in e):
        cls='salvage_mrconvert_dwi_post_eddy(no chain recorded, no grad table)'
    else:
        cls='conv' + ('+denoise' if has('dwidenoise') else '') + ('+degibbs' if has('mrdegibbs') else '') + ('+dwifslpreproc' if has('dwifslpreproc') else '')
        # detect eddy input stage
        d=[x for x in ch if re.match(r"(\S*/)?dwifslpreproc ",x)]
        if d:
            x=d[-1]
            pe=re.search(r'-pe_dir (\S+)',x).group(1)
            opts=re.search(r"eddy_options[= ]+'?(--[^']*?)'?(?= -nthreads| -scratch| -force| -eddy_mask| \(version)",x)
            o_=re.sub(r' ?--nthr=\d','',opts.group(1)) if opts else '?'
            E[(pe,o_,'eddy_mask' if '-eddy_mask' in x else '', 'eddyqc' if '-eddyqc_all' in x else '')]+=1
            V[re.search(r'version=([^)]*)',x).group(1)]+=1
        conv=[x for x in ch if '-export_grad_fsl' in x]
        if conv: V['conv@'+re.search(r'version=([^)]*)',conv[-1]).group(1)]+=1
        dn=[x for x in ch if re.match(r"(\S*/)?dwidenoise ",x)]
        if dn: DN[re.sub(r"\S+\.mif'?|'[^']*'",'X',dn[-1]).split('(')[0].strip()+' @'+re.search(r'version=([^)]*)',dn[-1]).group(1)]+=1
        gb=[x for x in ch if re.match(r"(\S*/)?mrdegibbs ",x)]
        if gb: GB[re.sub(r"\S+\.mif'?|'[^']*'",'X',gb[-1]).split('(')[0].strip()+' @'+re.search(r'version=([^)]*)',gb[-1]).group(1)]+=1
    C[cls]+=1
for n,c in [('chain',C),('eddy(pe_dir,options,mask,qc)',E),('versions',V),('denoise',DN),('degibbs',GB)]:
    print('=====',n)
    for k,v in c.most_common(): print(v,k)
