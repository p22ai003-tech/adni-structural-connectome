"""Phase 5 step 5 — tractography settings: unconstrained (main) against anatomically constrained (ACT) tracking for the
45-participant subset re-tracked by rb_act_subject.py.

Builds an edge cache for each setting (the same 45 participants), runs p5_features.py on both (network values,
exceptions and core with the main recipe, cut-offs from each setting's own CN edge lengths) and compares:
  agreement   per network value (whole brain, short and long range): Pearson r across participants and ICC(3,1)
  core        Dice overlap of the two exception cores (edges exceptional in > 50% of the 45)
  effects     Cliff's delta, CN versus MCI and CN versus AD, for the leading network values, under both settings
Writes phase5_uniform/tables/act_agreement.csv, act_key_effects.csv, act_summary.txt.
"""
import os, subprocess, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import rankdata

U = Path("/home/ec2-user/exp/docs/iit_dipanjan"); P5 = U / "phase5_uniform"; T = P5 / "tables"; Q = P5 / "act"; Q.mkdir(exist_ok=True)
ACT = Path("/data/derivatives/connectomes_uniform_v3_act"); units = [u for u in open(ACT / "units.txt").read().split() if u]
units = [u for u in units if (ACT / f"matrices/SC_AAL166_{u}_qc.json").is_file()]
sids = [u[:10] for u in units]
E = dict(np.load(U / "phase2_uniform/edges_uniform_v3.npz", allow_pickle=True))
rows = [list(E["sid"]).index(s) for s in sids]; iu = (E["i"], E["j"])
main = {k: (v[rows] if k in ("len", "fd", "count", "fa", "md", "rd", "ad") else v) for k, v in E.items()}
main["sid"] = np.array(sids, dtype=object); np.savez_compressed(Q / "edges_main45.npz", **main)
KINDS = {"len": "len_mean", "fd": "fd_sum", "count": "count", "fa": "fa_mean", "md": "md_mean", "rd": "rd_mean", "ad": "ad_mean"}
act = {k: np.full((len(units), len(iu[0])), np.nan, np.float32) for k in KINDS}
for r, u in enumerate(units):
    for k, kind in KINDS.items():
        v = np.loadtxt(ACT / f"matrices/SC_AAL166_{u}_{kind}.csv", delimiter=",")[iu]; act[k][r] = np.where(v > 0, v, np.nan)
pres = np.isfinite(act["len"]) & np.isfinite(act["fd"])
for k in ["fa", "md", "rd", "ad"]: act[k][~pres] = np.nan
act.update(sid=np.array(sids, dtype=object), i=E["i"], j=E["j"], net_i=E["net_i"], net_j=E["net_j"])
np.savez_compressed(Q / "edges_act45.npz", **act)
PY = U / "tools/.venv/bin/python"; env = dict(os.environ, IIT_P1=str(U / "phase1_uniform"))
for name in ("main45", "act45"):
    subprocess.run([str(PY), "-I", str(U / "tools/phase5/p5_features.py"), str(Q / f"edges_{name}.npz"), str(P5 / "aal3_network_mapping.csv"),
                    str(Q / name), "fd"], check=True, env=env)
fm = pd.read_csv(Q / "main45/features_published.csv", index_col=0).loc[sids]; fa = pd.read_csv(Q / "act45/features_published.csv", index_col=0).loc[sids]


def icc31(x, y):
    Y = np.c_[x, y]; n, k = Y.shape; gm = Y.mean()
    msr = k * ((Y.mean(1) - gm) ** 2).sum() / (n - 1); mse = (((Y - Y.mean(1, keepdims=True) - Y.mean(0) + gm) ** 2).sum()) / ((n - 1) * (k - 1))
    return (msr - mse) / (msr + (k - 1) * mse)


rows = []
for fam, pref in (("whole brain", "WB|"), ("short range", "TR|"), ("long range", "TR|")):
    cols = [c for c in fm.columns if c.startswith(pref) and (fam == "whole brain" or (("|SR-" in c) if fam == "short range" else ("|LR-" in c)))]
    for meas in ["FA", "MD", "AxD", "RD"]:
        cs = [c for c in cols if c.endswith(f"|{meas}") or c.endswith(f"-{meas}")]
        rs, ic = [], []
        for c in cs:
            ok = fm[c].notna() & fa[c].notna()
            if ok.sum() >= 10: rs.append(np.corrcoef(fm.loc[ok, c], fa.loc[ok, c])[0, 1]); ic.append(icc31(fm.loc[ok, c].values, fa.loc[ok, c].values))
        rows.append({"Edges": fam, "Measure": meas, "Network values": len(rs), "Pearson r, median [range]": f"{np.median(rs):.2f} [{min(rs):.2f}, {max(rs):.2f}]",
                     "ICC(3,1), median [range]": f"{np.median(ic):.2f} [{min(ic):.2f}, {max(ic):.2f}]"})
pd.DataFrame(rows).to_csv(T / "act_agreement.csv", index=False)
cm_ = pd.read_csv(Q / "main45/core_edges.csv"); ca = pd.read_csv(Q / "act45/core_edges.csv")
km, ka = set(zip(cm_.i, cm_.j)), set(zip(ca.i, ca.j)); dice = 2 * len(km & ka) / max(len(km) + len(ka), 1)
grp = pd.read_csv(U / "phase1_uniform/covariates.csv").set_index("subject_id").loc[sids, "group"].values


def cliff(a, b):
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]; r = rankdata(np.r_[a, b]); u = r[:len(a)].sum() - len(a) * (len(a) + 1) / 2
    return 2 * u / (len(a) * len(b)) - 1


KEY = [("Limbic AxD", "WB|Limbic|AxD"), ("Limbic MD", "WB|Limbic|MD"), ("Limbic long-range MD", "TR|Limbic|LR-MD"),
       ("Limbic short-range AxD", "TR|Limbic|SR-AxD"), ("DMN short-range MD", "TR|DMN|SR-MD"), ("Whole-brain-average FA (DMN)", "WB|DMN|FA"),
       ("Core coverage", "COV|core|coverage")]
rows = []
for lab, f in KEY:
    r = {"Measure": lab}
    for nm, F in (("unconstrained", fm), ("ACT", fa)):
        v = F[f].values.astype(float)
        for a_, b_ in (("CN", "MCI"), ("CN", "AD")):
            r[f"{nm}: {a_}–{b_}"] = f"{cliff(v[grp == a_], v[grp == b_]):+.2f}".replace("-", "−")
    rows.append(r)
pd.DataFrame(rows).to_csv(T / "act_key_effects.csv", index=False)
import json
tm, ta = json.load(open(Q / "main45/thresholds.json")), json.load(open(Q / "act45/thresholds.json"))
dens = lambda Ed: np.nanmean(np.isfinite(Ed["len"]).mean(1))
lenm = lambda Ed: np.nanmedian(np.nanmedian(Ed["len"], 1))
ag = pd.read_csv(T / "act_agreement.csv")
wb_r = [float(x.split(" ")[0]) for x in ag[ag.Edges == "whole brain"]["Pearson r, median [range]"]]
word = "closely" if np.median(wb_r) >= 0.8 else ("moderately" if np.median(wb_r) >= 0.6 else "only weakly")
ke = pd.read_csv(T / "act_key_effects.csv"); num = lambda x: float(str(x).replace("−", "-"))
same = sum(np.sign(num(r[f"unconstrained: {c}"])) == np.sign(num(r[f"ACT: {c}"])) for _, r in ke.iterrows() for c in ("CN–MCI", "CN–AD"))
close = sum(abs(num(r[f"unconstrained: {c}"]) - num(r[f"ACT: {c}"])) < 0.15 for _, r in ke.iterrows() for c in ("CN–MCI", "CN–AD"))
txt = (f"For the {len(units)} re-tracked participants (CN {int((grp == 'CN').sum())}, MCI {int((grp == 'MCI').sum())}, AD {int((grp == 'AD').sum())}), "
       f"ACT changes the connectome's geometry: edge density {dens(main):.2f} → {dens(act):.2f} and median edge length {lenm(main):.0f} → "
       f"{lenm(act):.0f} mm (short/long cut-offs {tm['short_max_mm']:.0f}/{tm['long_min_mm']:.0f} → {ta['short_max_mm']:.0f}/{ta['long_min_mm']:.0f} mm). "
       f"Whole-brain network diffusion values agree {word} between the two settings (median Pearson r by measure "
       f"{', '.join(f'{m} {v:.2f}' for m, v in zip(ag[ag.Edges == 'whole brain'].Measure, wb_r))}; short- and long-range values in the "
       f"table). The exception cores overlap with Dice {dice:.2f} ({len(km)} and {len(ka)} edges). Of the {2 * len(ke)} group effects of "
       f"the leading measures (second table), {same} keep their sign under ACT and {close} change by less than 0.15 in δ; with 15 "
       "participants per group only the direction and rough size can be compared.")
open(T / "act_summary.txt", "w").write(txt); print(txt)
