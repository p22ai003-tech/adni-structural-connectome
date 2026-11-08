"""Phase 1 step 4 — summary tables and verdicts from the engine outputs (figures: see the notebook, Section 7).

Verdict rules (fixed before looking at results):
  single effect (F2, F4, F5): holds = same sign as the published (C0) value and q < 0.05;
                              weakened = same sign, q >= 0.05; reversed = opposite sign and q < 0.05.
  F3 (SR - LR difference):    holds = 95% bootstrap interval excludes 0 in the published direction;
                              weakened = same direction, interval includes 0; reversed = opposite direction,
                              interval excludes 0.
  F1 (diffusivity before FA): holds = MD or AxD separates CN and MCI in >= 5 networks AND FA separates groups
                              in more networks in CN-AD than in CN-MCI.

On the uniformly rebuilt connectomes (IIT_SRC=uniform_v3) the reference is still the PUBLISHED estimate (C0 of the
published-data run in IIT_P1_COMPARE): every rebuilt condition, C0 included, is judged against what the manuscript
reported, and the published values appear as an extra first column/row labelled "Published".
"""
import sys as _sys; from pathlib import Path as _Path; _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import iit_chain as chain; chain.boot()      # IIT_ROOT and the chain settings; no effect when IIT_ROOT is unset
import os
import json
from pathlib import Path
import numpy as np, pandas as pd

O = Path(os.environ.get("IIT_P1", "/home/ec2-user/exp/docs/iit_dipanjan/phase1")); T = O / "tables"; T.mkdir(exist_ok=True)
e = pd.read_csv(O / "effects_long.csv"); s = pd.read_csv(O / "sr_lr_summary.csv")
notes = json.load(open(O / "condition_notes.json"))
MAIN = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"]          # main tables; C0q and C9-C11 go to tables/sensitivity_*
REBUILT = os.environ.get("IIT_SRC", "published") != "published"
if REBUILT:                                    # published-data run: its C0 is the reference
    P0 = Path(os.environ.get("IIT_P1_COMPARE", "/home/ec2-user/exp/docs/iit_dipanjan/phase1"))
    ep = pd.read_csv(P0 / "effects_long.csv"); sp = pd.read_csv(P0 / "sr_lr_summary.csv")
    ep = ep[ep.condition.str.split().str[0] == "C0"].assign(condition="Published")
    sp = sp[sp.condition.str.split().str[0] == "C0"].assign(condition="Published")
    e = pd.concat([ep, e], ignore_index=True); s = pd.concat([sp, s], ignore_index=True)
    MAIN = ["Published"] + MAIN
ALL = list(dict.fromkeys(e.condition))
CONDS = [c for c in ALL if c.split(" ", 1)[0] in MAIN]
SHORT = {c: c.split(" ", 1)[0] for c in ALL}
LABEL = {c: (c.split(" ", 1) + ["published manuscript"])[1] for c in ALL}
CT = ["CN-MCI", "CN-AD", "MCI-AD"]

KEY = [  # (finding, label, feature, contrast)
    ("F2", "Limbic AxD, CN–MCI", "WB|Limbic|AxD", "CN-MCI"),
    ("F2", "Salience AxD, CN–MCI", "WB|Salience_VAN|AxD", "CN-MCI"),
    ("F2", "Limbic AxD, CN–AD", "WB|Limbic|AxD", "CN-AD"),
    ("F2", "DMN FA, MCI–AD", "WB|DMN|FA", "MCI-AD"),
    ("F2", "Limbic LR-MD, CN–MCI", "TR|Limbic|LR-MD", "CN-MCI"),
    ("F2", "Limbic SR-AxD, CN–AD", "TR|Limbic|SR-AxD", "CN-AD"),
    ("F2", "Limbic SR-AxD, MCI–AD", "TR|Limbic|SR-AxD", "MCI-AD"),
    ("F4", "Limbic LRexc-MD, CN–MCI", "EX|Limbic|LRexc-MD", "CN-MCI"),
    ("F4", "DMN SRexc-RD, CN–AD", "EX|DMN|SRexc-RD", "CN-AD"),
    ("F4", "Frontoparietal SRexc-AxD, MCI–AD", "EX|Frontoparietal|SRexc-AxD", "MCI-AD"),
    ("F5", "Core coverage, CN–MCI", "COV|core|coverage", "CN-MCI"),
    ("F5", "Core coverage, CN–AD", "COV|core|coverage", "CN-AD"),
    ("F5", "Core coverage, MCI–AD", "COV|core|coverage", "MCI-AD"),
]


def verdict(d0, d, q):
    if not np.isfinite(d) or not np.isfinite(q):
        return "n/a"
    if np.sign(d) == np.sign(d0):
        return "holds" if q < 0.05 else "weakened"
    return "reversed" if q < 0.05 else "weakened"


# ---------------- key effects
rows = []
for f, lab, feat, ct in KEY:
    base = e[(e.condition == CONDS[0]) & (e.feature == feat) & (e.contrast == ct)]
    d0 = base.delta.iloc[0] if len(base) else np.nan
    for c in ALL:
        r = e[(e.condition == c) & (e.feature == feat) & (e.contrast == ct)]
        d, q = (r.delta.iloc[0], r.q.iloc[0]) if len(r) else (np.nan, np.nan)
        rows.append(dict(finding=f, effect=lab, condition=SHORT[c], delta=d, q=q,
                         verdict="reference" if c == CONDS[0] else verdict(d0, d, q),
                         n=f"{int(r.n_first.iloc[0])}/{int(r.n_second.iloc[0])}" if len(r) else ""))
key = pd.DataFrame(rows); key.to_csv(T / "key_effects.csv", index=False)

# ---------------- F3
s["verdict"] = ""
for kind in ["TR", "EX"]:
    for ct in CT:
        b = s[(s.condition == CONDS[0]) & (s.kind == kind) & (s.contrast == ct)].iloc[0]
        for i in s[(s.kind == kind) & (s.contrast == ct)].index:
            r = s.loc[i]
            if r.condition == CONDS[0]:
                s.loc[i, "verdict"] = "reference"; continue
            same = np.sign(r["diff"]) == np.sign(b["diff"])
            excl = (r.ci_lo > 0) or (r.ci_hi < 0)
            s.loc[i, "verdict"] = ("holds" if excl else "weakened") if same else ("reversed" if excl else "weakened")
s["condition_short"] = s.condition.map(SHORT)
s.to_csv(T / "sr_lr_verdicts.csv", index=False)

# ---------------- F1
f1 = []
for c in ALL:
    x = e[(e.condition == c) & (e.kind == "WB")]
    rec = {"condition": SHORT[c]}
    for m in ["AxD", "MD", "RD", "FA"]:
        for ct in CT:
            rec[f"{m} {ct}"] = int(((x.measure == m) & (x.contrast == ct) & (x.q < 0.05)).sum())
    ok = max(rec["MD CN-MCI"], rec["AxD CN-MCI"]) >= 5 and rec["FA CN-AD"] > rec["FA CN-MCI"]
    rec["verdict"] = "reference" if c == CONDS[0] else ("holds" if ok else "weakened")
    f1.append(rec)
f1 = pd.DataFrame(f1); f1[f1.condition.isin(MAIN)].to_csv(T / "f1_network_counts.csv", index=False)

# ---------------- verdict matrix (finding x condition)
vm = {}
for c in CONDS[1:]:
    sc = SHORT[c]; col = {}
    col["F1 Diffusivity before FA"] = f1.loc[f1.condition == sc, "verdict"].iloc[0]
    for lab in [k[1] for k in KEY]:
        col[lab] = key[(key.effect == lab) & (key.condition == sc)].verdict.iloc[0]
    for kind, nm in [("TR", "F3 SR−LR, tract"), ("EX", "F3 SR−LR, exceptions")]:
        for ct in CT:
            r = s[(s.condition == c) & (s.kind == kind) & (s.contrast == ct)]
            col[f"{nm}, {ct.replace('-', '–')}"] = r.verdict.iloc[0] if len(r) else "n/a"
    vm[sc] = col
vm = pd.DataFrame(vm); vm.to_csv(T / "verdict_matrix.csv")
print(vm.to_string())

# ---------------- figures: Figure P1-1 and P1-2 are drawn by notebooks/phase1_confound_control.ipynb (Section 7),
# which writes fig_p1_key_effects.png and fig_p1_sr_lr_forest.png into phase1/

# ---------------- extra tables for the document
DESC = {
    "C0": "All 530 participants, unadjusted (as published)",
    "C0q": "All participants passing quality control, unadjusted",
    "C1": "ADNI-3 and ADNI-4 participants passing quality control, unadjusted",
    "C2": "C1 adjusted for age, sex and education — PRIMARY",
    "C3": "C2 after ComBat harmonisation of scanner vendor (Siemens, Philips, GE)",
    "C4": "C1 propensity-matched on age, sex, education and vendor, separately for each contrast",
    "C5": "C2 additionally adjusted for medial temporal volume and FLAIR white-matter hyperintensity volume",
    "C6": "C2 using only edges reconstructed in at least 90% of ADNI-3/4 participants",
    "C7": "C2 in amyloid-defined groups: amyloid-negative CN, amyloid-positive MCI and AD",
    "C8": "C2 restricted to Siemens scanners",
    "C9": "C2 additionally adjusted for processing route",
    "C10": "C2 additionally adjusted for microstructure coverage (share of edges carrying diffusion values)",
    "C11": "C2 restricted to participants with microstructure coverage of at least 25%",
}
SHORT = {c: c.split(" ", 1)[0] for c in ALL}
if REBUILT:
    DESC.update({
        "Published": "Published estimate (530 participants, published connectomes): the reference for every verdict",
        "C0": "All participants passing rebuild quality control, rebuilt connectomes, unadjusted",
        "C0q": "C0 without the participants meeting the diffusion-value rules R1–R3 (Section 3.7)",
        "C9": "C2 additionally adjusted for the published processing route (does the old route label still carry signal?)",
        "C10": "Not applicable: every rebuilt connectome carries diffusion values on all connected edges",
        "C11": "Not applicable (as C10)"})
crow = []
for c in ALL:
    n = notes.get(c + " n", {})
    if SHORT[c] == "Published":
        pn = json.load(open(P0 / "condition_notes.json")); n = next(v for k, v in pn.items() if k.startswith("C0 ") and k.endswith(" n"))
    if SHORT[c] == "C4":
        n = {"CN": "by contrast", "MCI": "(Appendix A)", "AD": ""}
    crow.append({"Condition": SHORT[c], "Description": DESC[SHORT[c]], "CN": n.get("CN", ""), "MCI": n.get("MCI", ""), "AD": n.get("AD", "")})
pd.DataFrame(crow).to_csv(T / "conditions.csv", index=False)

fq = lambda q: "" if not np.isfinite(q) else (f"{q:.3f}" if q >= 0.001 else f"{q:.0e}")
kw = key.assign(cell=lambda d: d.apply(lambda r: "" if not np.isfinite(r.delta) else f"{r.delta:+.3f} ({fq(r.q)})", axis=1))
kw = kw.pivot_table(index="effect", columns="condition", values="cell", aggfunc="first").reindex([k[1] for k in KEY])
kw = kw[[SHORT[c] for c in CONDS]].reset_index().rename(columns={"effect": "Effect: δ (q)"})
kw.to_csv(T / "key_effects_wide.csv", index=False)

srt = s.assign(Edges=s.kind.map({"TR": "All", "EX": "Exceptions"}),
               SR=s.sr_mean.round(3), LR=s.lr_mean.round(3),
               diff_ci=s.apply(lambda r: f"{r['diff']:+.3f} [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}]", axis=1),
               cells=s.apply(lambda r: f"{int(r.sr_gt_lr)}/{int(r.n_cells)}", axis=1))
srt = srt[["condition_short", "Edges", "contrast", "SR", "LR", "diff_ci", "cells", "verdict"]]
srt.columns = ["Condition", "Edges", "Contrast", "SR mean |δ|", "LR mean |δ|", "SR − LR [95% CI]", "SR > LR cells", "Verdict"]
srt[srt.Condition.isin(MAIN)].to_csv(T / "sr_lr_table.csv", index=False)

# ---------------- sensitivity conditions (after QC; processing checks), same layout as the main tables
SENS = ["C2", "C0q", "C9", "C10", "C11"]
ks = kw0 = key.assign(cell=lambda d: d.apply(lambda r: "" if not np.isfinite(r.delta) else f"{r.delta:+.3f} ({fq(r.q)})", axis=1))
ks = ks.pivot_table(index="effect", columns="condition", values="cell", aggfunc="first").reindex([k[1] for k in KEY]).reindex(columns=SENS)
SENS = [c for c in SENS if c in set(f1.condition)]
f1s = f1.set_index("condition").loc[SENS]
rows = [{"Finding": k, **ks.loc[k].to_dict()} for k in ks.index]
for m in ["AxD", "MD", "RD", "FA"]:
    rows.append({"Finding": f"{m}: networks with q < 0.05 (CN–MCI / CN–AD / MCI–AD)",
                 **{c: " / ".join(str(int(f1s.loc[c, f"{m} {ct}"])) for ct in CT) for c in SENS}})
rows.append({"Finding": "F1 verdict", **{c: f1s.loc[c, "verdict"] for c in SENS}})
for kind, nm in [("TR", "all edges"), ("EX", "exception edges")]:
    for ct in CT:
        r = {"Finding": f"SR − LR, {nm}, {ct.replace('-', '–')} [95% CI]"}
        for c in SENS:
            x = s[(s.condition_short == c) & (s.kind == kind) & (s.contrast == ct)]
            r[c] = f"{x['diff'].iloc[0]:+.3f} [{x.ci_lo.iloc[0]:+.3f}, {x.ci_hi.iloc[0]:+.3f}]" if len(x) else ""
        rows.append(r)
pd.DataFrame(rows).to_csv(T / "sensitivity_processing.csv", index=False)

for c in CONDS:
    if SHORT[c] == "C4":
        rows = [{"Contrast": ct, "n (first/second)": "/".join(str(v) for v in b["n"].values()),
                 "SMD age": b["smd"]["age"], "SMD female": b["smd"]["female"], "SMD education": b["smd"]["education"]}
                for ct, b in notes[c].items()]
        pd.DataFrame(rows).to_csv(T / "matching_balance.csv", index=False)
print("document tables written")
