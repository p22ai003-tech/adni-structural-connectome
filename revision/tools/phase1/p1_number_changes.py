"""Old (pooled 530, as published) -> new (ADNI-3/4 adjusted, primary) for every headline number."""
import sys as _sys; from pathlib import Path as _Path; _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import iit_chain as chain; chain.boot()      # IIT_ROOT and the chain settings; no effect when IIT_ROOT is unset
from pathlib import Path
import os
import numpy as np, pandas as pd
O = Path(os.environ.get("IIT_P1", "/home/ec2-user/exp/docs/iit_dipanjan/phase1"))
e = pd.read_csv(O / "effects_long.csv"); s = pd.read_csv(O / "sr_lr_summary.csv")
C0, C2 = "C0 Pooled 530 (published)", f"C2 {chain.sample_label()} + age, sex, education"
if os.environ.get("IIT_SRC", "published") != "published":      # rebuilt connectomes: the old column is the published C0
    P0 = Path(os.environ.get("IIT_P1_COMPARE", "/home/ec2-user/exp/docs/iit_dipanjan/phase1"))
    e0, s0 = pd.read_csv(P0 / "effects_long.csv"), pd.read_csv(P0 / "sr_lr_summary.csv")
    e = pd.concat([e0[e0.condition == C0], e[~e.condition.str.startswith("C0 ")]])
    s = pd.concat([s0[s0.condition == C0], s[~s.condition.str.startswith("C0 ")]])
fq = lambda q: (f"{q:.3f}" if q >= 0.001 else f"{q:.1e}") if np.isfinite(q) else ""
def cell(c, f, ct):
    r = e[(e.condition == c) & (e.feature == f) & (e.contrast == ct)].iloc[0]; return f"{r.delta:+.3f} (q {fq(r.q)})"
NICE = {"Salience_VAN": "Salience", "DorsalAttention": "Dorsal attention"}
def leader(kind, ct):                      # largest |δ| among significant effects of the primary analysis
    x = e[(e.condition == C2) & (e.kind == kind) & (e.contrast == ct) & (e.q < 0.05)]
    if x.empty: x = e[(e.condition == C2) & (e.kind == kind) & (e.contrast == ct) & e.delta.notna()]   # nothing significant
    r = x.loc[x.delta.abs().idxmax()]; net, meas = r.feature.split("|")[1:]
    return f"{NICE.get(net, net)} {meas}, {ct.replace('-', '–')} (new leader)", r.feature
rows = []
for m in ["AxD", "MD", "RD", "FA"]:
    cnt = lambda c: " / ".join(str(int(((e.condition == c) & (e.kind == "WB") & (e.measure == m) & (e.contrast == ct) & (e.q < 0.05)).sum())) for ct in ["CN-MCI", "CN-AD", "MCI-AD"])
    rows.append(("Table 2", f"{m}: networks separating groups (CN–MCI / CN–AD / MCI–AD)", cnt(C0), cnt(C2)))
for loc, lab, f, ct in [("Results §2, Table 4", "Limbic AxD, CN–MCI", "WB|Limbic|AxD", "CN-MCI"),
                        ("Results §2, Table 4", "Limbic AxD, CN–AD", "WB|Limbic|AxD", "CN-AD"),
                        ("Results §2, Table 4", "DMN FA, MCI–AD (published leader)", "WB|DMN|FA", "MCI-AD"),
                        ("Results §2, Table 4", *leader("WB", "MCI-AD"), "MCI-AD"),
                        ("Results §3, Table 4", "Limbic LR-MD, CN–MCI", "TR|Limbic|LR-MD", "CN-MCI"),
                        ("Abstract, Results §3, Table 4", "Limbic SR-AxD, CN–AD", "TR|Limbic|SR-AxD", "CN-AD"),
                        ("Results §3, Table 4", "Limbic SR-AxD, MCI–AD", "TR|Limbic|SR-AxD", "MCI-AD"),
                        ("Results §3", *leader("TR", "MCI-AD"), "MCI-AD"),
                        ("Results §4, Table 4", "Limbic LRexc-MD, CN–MCI", "EX|Limbic|LRexc-MD", "CN-MCI"),
                        ("Results §4, Table 4", "DMN SRexc-RD, CN–AD (published leader)", "EX|DMN|SRexc-RD", "CN-AD"),
                        ("Results §4", *leader("EX", "CN-AD"), "CN-AD"),
                        ("Results §4, Table 4", "Frontoparietal SRexc-AxD, MCI–AD", "EX|Frontoparietal|SRexc-AxD", "MCI-AD"),
                        ("Results §5, Table 6", "Core coverage, CN–MCI", "COV|core|coverage", "CN-MCI"),
                        ("Results §5, Table 6", "Core coverage, CN–AD", "COV|core|coverage", "CN-AD"),
                        ("Results §5, Table 6", "Core coverage, MCI–AD", "COV|core|coverage", "MCI-AD")]:
    rows.append((loc, lab, cell(C0, f, ct), cell(C2, f, ct)))
for kind, nm in [("TR", "all edges"), ("EX", "exception edges")]:
    for ct in ["CN-MCI", "CN-AD", "MCI-AD"]:
        g = lambda c: s[(s.condition == c) & (s.kind == kind) & (s.contrast == ct)].iloc[0]
        f = lambda r: f"SR {r.sr_mean:.3f} / LR {r.lr_mean:.3f}; SR−LR {r['diff']:+.3f} [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}]; SR>LR {int(r.sr_gt_lr)}/{int(r.n_cells)}"
        rows.append(("Table 3, Fig. 4", f"Short vs long range, {nm}, {ct.replace('-', '–')}", f(g(C0)), f(g(C2))))
pd.DataFrame(rows, columns=["Where", "Quantity", "Published (pooled 530)", "Primary (ADNI-3/4 after QC, adjusted)"
                           + (", rebuilt connectomes" if os.environ.get("IIT_SRC", "published") != "published" else "")]).to_csv(O / "tables/number_changes.csv", index=False)
print(pd.read_csv(O / "tables/number_changes.csv").to_string(index=False))
