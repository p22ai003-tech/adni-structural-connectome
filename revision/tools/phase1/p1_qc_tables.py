"""Phase 1 step 4b — quality-control and processing tables for the report (Sections 3.7 and 3.8).

  tables/qc_exclusions.csv     the participants excluded by p1_audit.py, with the rule that excluded them
  tables/what_changed.csv      first report (no QC; archive_v1_firstreport/) against the current QC'd run, for the
                               primary analysis (C2), atrophy adjustment (C5) and ComBat (C3)
  tables/coverage_by_group.csv microstructure coverage and processing route by diagnosis, primary sample
  qc_summary.json              numbers quoted in the narrative

On the uniformly rebuilt connectomes (IIT_SRC != published): the exclusions carry the rebuild rules (R0 / R4), the
comparison is published connectomes (IIT_P1_COMPARE, after quality control) against rebuilt connectomes for the pooled
(C0q), primary (C2) and atrophy-adjusted (C5) analyses, and tables/rebuild_comparison.csv replaces coverage_by_group.csv.
"""
import os
import json
from pathlib import Path
import numpy as np, pandas as pd

O = Path(os.environ.get("IIT_P1", "/home/ec2-user/exp/docs/iit_dipanjan/phase1")); T = O / "tables"; A1 = Path(os.environ.get("IIT_P1_COMPARE", str(O / "archive_v1_firstreport")))
cov = pd.read_csv(O / "covariates.csv").set_index("subject_id")
qc = pd.read_csv(O / "qc.csv", index_col=0).reindex(cov.index)
au = pd.read_csv(O / "audit/audit_participants.csv", index_col=0).reindex(cov.index)
AUS = json.load(open(O / "audit/audit_summary.json"))
CT = ["CN-MCI", "CN-AD", "MCI-AD"]
REBUILT = os.environ.get("IIT_SRC", "published") != "published"
fq = lambda q: (f"{q:.3f}" if q >= 0.001 else f"{q:.0e}") if np.isfinite(q) else ""


def load(folder):
    e = pd.read_csv(folder / "effects_long.csv"); s = pd.read_csv(folder / "sr_lr_summary.csv")
    for extra in ["C8"]:                                   # the first report ran C8 separately
        if (folder / f"effects_long_{extra}.csv").is_file():
            e = pd.concat([e, pd.read_csv(folder / f"effects_long_{extra}.csv")]); s = pd.concat([s, pd.read_csv(folder / f"sr_lr_summary_{extra}.csv")])
    e["c"] = e.condition.str.split().str[0]; s["c"] = s.condition.str.split().str[0]
    return e, s


e1, s1 = load(A1); e2, s2 = load(O)

# ---------------- exclusions
x = cov[qc.qc_fail == 1].join(qc[["qc_reason", "wb_md_median", "density", "zero_nodes"] + (["reg_fa_corr_rebuild"] if REBUILT else [])])
x = x.sort_values(["qc_reason", "wb_md_median"], ascending=[True, False])
if REBUILT:
    def plain(r):
        parts = (["failed atlas registration"] if "R4" in r else []) + (["edge density below 0.6"] if ("density floor" in r or "R3" in r) else [])
        parts += (["median MD above the far-out fence"] if "R1" in r else []) + (["impossible tensor values"] if "R2" in r else [])
        return " and ".join(parts) or "excluded before the analysis"
    x = x.sort_values("reg_fa_corr_rebuild")
    pd.DataFrame({"Participant": x.index, "Group": x.group.values, "Phase": x.phase.values, "Vendor": x.vendor.values,
                  "Reason": x.qc_reason.map(plain).values, "Registration (r with template)": x.reg_fa_corr_rebuild.round(2).values,
                  "Edge density": x.density.round(2).values}).to_csv(T / "qc_exclusions.csv", index=False)
REASON = {"R1 diffusivity level, R2 tensor values": "implausible diffusivity and impossible tensor values",
          "R2 tensor values": "impossible tensor values", "R3 reconstruction": "failed reconstruction"}
if not REBUILT:
    pd.DataFrame({"Participant": x.index, "Group": x.group.values, "Phase": x.phase.values, "Vendor": x.vendor.values,
                  "Reason": x.qc_reason.map(REASON).values, "Median WB MD (×10⁻³ mm²/s)": x.wb_md_median.round(2).values,
                  "Edge density": x.density.round(2).values}).to_csv(T / "qc_exclusions.csv", index=False)

# ---------------- first report vs now
KEY = [("Limbic AxD, CN–MCI", "WB|Limbic|AxD", "CN-MCI"), ("Limbic LR-MD, CN–MCI", "TR|Limbic|LR-MD", "CN-MCI"),
       ("Limbic LRexc-MD, CN–MCI", "EX|Limbic|LRexc-MD", "CN-MCI"), ("Limbic AxD, CN–AD", "WB|Limbic|AxD", "CN-AD"),
       ("Limbic SR-AxD, CN–AD", "TR|Limbic|SR-AxD", "CN-AD"), ("Limbic SR-AxD, MCI–AD", "TR|Limbic|SR-AxD", "MCI-AD"),
       ("DMN FA, MCI–AD", "WB|DMN|FA", "MCI-AD"), ("DMN SRexc-RD, CN–AD", "EX|DMN|SRexc-RD", "CN-AD"),
       ("Frontoparietal SRexc-AxD, MCI–AD", "EX|Frontoparietal|SRexc-AxD", "MCI-AD"), ("Core coverage, CN–AD", "COV|core|coverage", "CN-AD")]
PAIRS = [("C2", "Primary (C2)"), ("C5", "Atrophy + lesions (C5)"), ("C3", "ComBat (C3)")]
TAGS = ("first report", "now")
if REBUILT:
    PAIRS = [("C0q", "Pooled after QC (C0q)"), ("C2", "Primary (C2)"), ("C5", "Atrophy + lesions (C5)")]
    TAGS = ("published connectomes", "rebuilt")


def dq(e, c, f, ct):
    r = e[(e.c == c) & (e.feature == f) & (e.contrast == ct)]
    return (float(r.delta.iloc[0]), float(r.q.iloc[0])) if len(r) else (np.nan, np.nan)


def f1n(e, c, m, ct):
    return int(((e.c == c) & (e.kind == "WB") & (e.measure == m) & (e.contrast == ct) & (e.q < 0.05)).sum())


def sri(s, c, ct, kind="TR"):
    r = s[(s.c == c) & (s.kind == kind) & (s.contrast == ct)].iloc[0]
    return dict(diff=float(r["diff"]), lo=float(r.ci_lo), hi=float(r.ci_hi), cells=int(r.sr_gt_lr), n=int(r.n_cells))


rows = []
for lab, f, ct in KEY:
    r = {"Finding": lab}
    for c, nm in PAIRS:
        for tag, e in zip(TAGS, (e1, e2)):
            d, q = dq(e, c, f, ct); r[f"{nm}: {tag}"] = f"{d:+.3f} ({fq(q)})"
    rows.append(r)
for m in ["AxD", "MD", "RD", "FA"]:
    r = {"Finding": f"{m}: networks with q < 0.05 (CN–MCI / CN–AD / MCI–AD)"}
    for c, nm in PAIRS:
        for tag, e in zip(TAGS, (e1, e2)):
            r[f"{nm}: {tag}"] = " / ".join(str(f1n(e, c, m, ct)) for ct in CT)
    rows.append(r)
for ct in CT:
    r = {"Finding": f"SR − LR, all edges, {ct.replace('-', '–')} [95% CI]"}
    for c, nm in PAIRS:
        for tag, s in zip(TAGS, (s1, s2)):
            v = sri(s, c, ct); r[f"{nm}: {tag}"] = f"{v['diff']:+.3f} [{v['lo']:+.3f}, {v['hi']:+.3f}]"
    rows.append(r)
pd.DataFrame(rows).to_csv(T / "what_changed.csv", index=False)

if REBUILT:
    # ---------------- published against rebuilt connectomes, primary sample of the rebuilt run
    au1 = pd.read_csv(A1 / "audit/audit_participants.csv", index_col=0).reindex(cov.index)
    P = cov[(cov.adni34 == 1) & (qc.qc_fail == 0)]
    med = lambda d, c, g: float(d.loc[P.index[P.group == g], c].median())
    rows = []
    for lab, c, fmt in [("Edges carrying diffusion values (median)", "md_coverage", "{:.0%}"), ("Edge density (median)", "density", "{:.2f}"),
                        ("Median edge length, mm", "len_median", "{:.0f}"), ("Whole-brain FA (median)", "wb_fa_median", "{:.3f}"),
                        ("Whole-brain MD, ×10⁻³ mm²/s (median)", "wb_md_median", "{:.3f}")]:
        r = {"Measure": lab}
        for g in ["CN", "MCI", "AD"]:
            r[f"{g}: published"] = fmt.format(med(au1, c, g)); r[f"{g}: rebuilt"] = fmt.format(med(au, c, g))
        rows.append(r)
    pd.DataFrame(rows).to_csv(T / "rebuild_comparison.csv", index=False)
    from scipy import stats
    both = P.index[au1.loc[P.index, "wb_md_median"].notna() & au.loc[P.index, "wb_md_median"].notna()]
    route = qc.loc[P.index, "route"]
    S = dict(n_excluded=int(qc.qc_fail.sum()), excluded_adni34=AUS["qc_fail_adni34"], by_reason=AUS["qc_by_reason"],
             values_masked=AUS["values_masked_among_kept"], fences=AUS["fences"],
             reg_retained_min=float(qc.loc[qc.qc_fail == 0, "reg_fa_corr_rebuild"].min()),
             reg_excluded_max=float(qc.loc[qc.reg_fa_corr_rebuild < 0.4, "reg_fa_corr_rebuild"].max()),
             cov_retained_min=float(qc.loc[qc.qc_fail == 0, "md_coverage"].min()),
             cov_published_by_group={g: med(au1, "md_coverage", g) for g in ["CN", "MCI", "AD"]},
             len_by_route_published={k: float(v) for k, v in au1.loc[P.index].groupby(route).len_median.median().items()},
             len_by_route_rebuilt={k: float(v) for k, v in au.loc[P.index].groupby(route).len_median.median().items()},
             fa_by_route_published={k: float(v) for k, v in au1.loc[P.index].groupby(route).wb_fa_median.median().items()},
             fa_by_route_rebuilt={k: float(v) for k, v in au.loc[P.index].groupby(route).wb_fa_median.median().items()},
             rho_md_pub_vs_rebuilt=float(stats.spearmanr(au1.loc[both, "wb_md_median"], au.loc[both, "wb_md_median"]).statistic),
             rho_fa_pub_vs_rebuilt=float(stats.spearmanr(au1.loc[both, "wb_fa_median"], au.loc[both, "wb_fa_median"]).statistic),
             first={c: {lab: dq(e1, c, f, ct) for lab, f, ct in KEY} for c, _ in PAIRS},
             now={c: {lab: dq(e2, c, f, ct) for lab, f, ct in KEY} for c, _ in PAIRS},
             f1_first={c: {f"{m} {ct}": f1n(e1, c, m, ct) for m in ["AxD", "MD", "RD", "FA"] for ct in CT} for c, _ in PAIRS},
             f1_now={c: {f"{m} {ct}": f1n(e2, c, m, ct) for m in ["AxD", "MD", "RD", "FA"] for ct in CT} for c, _ in PAIRS},
             srlr_first={c: {ct: sri(s1, c, ct) for ct in CT} for c, _ in PAIRS},
             srlr_now={c: {ct: sri(s2, c, ct) for ct in CT} for c, _ in PAIRS})
    json.dump(S, open(O / "qc_summary.json", "w"), indent=1)
    print(pd.read_csv(T / "qc_exclusions.csv").to_string(index=False)); print()
    print(pd.read_csv(T / "rebuild_comparison.csv").to_string(index=False)); print()
    print(pd.read_csv(T / "what_changed.csv").to_string(index=False)); print()
    print(json.dumps({k: S[k] for k in S if k.startswith(("reg_", "cov_", "len_", "fa_", "rho_"))}, indent=1))
    raise SystemExit
# ---------------- coverage and route by group (primary sample)
P = cov[(cov.adni34 == 1) & (qc.qc_fail == 0)].join(qc[["md_coverage", "route"]])
cg = P.groupby("group").agg(n=("md_coverage", "size"), median=("md_coverage", "median"),
                            q1=("md_coverage", lambda v: v.quantile(0.25)), q3=("md_coverage", lambda v: v.quantile(0.75)),
                            below5=("md_coverage", lambda v: 100 * (v < 0.05).mean()), below25=("md_coverage", lambda v: 100 * (v < 0.25).mean()),
                            new_route=("route", lambda v: 100 * (v == "new").mean())).reindex(["CN", "MCI", "AD"])
pd.DataFrame({"Group": cg.index, "n": cg.n.values,
              "Edges carrying diffusion values, median [IQR]": [f"{m:.0%} [{a:.0%}–{b:.0%}]" for m, a, b in zip(cg["median"], cg.q1, cg.q3)],
              "Below 5%": [f"{v:.0f}%" for v in cg.below5], "Below 25%": [f"{v:.0f}%" for v in cg.below25],
              "Current processing route": [f"{v:.0f}%" for v in cg.new_route]}).to_csv(T / "coverage_by_group.csv", index=False)

# ---------------- numbers for the text
from scipy import stats
nw = P[(P.route == "new")].join(au[["wb_md_median", "wb_fa_median"]])
rho = {g: dict(md=float(stats.spearmanr(d.md_coverage, d.wb_md_median, nan_policy="omit").statistic),
               fa=float(stats.spearmanr(d.md_coverage, d.wb_fa_median, nan_policy="omit").statistic))
       for g, d in nw[nw.group.isin(["CN", "MCI"])].groupby("group")}
cn = nw[nw.group == "CN"]
lohi = dict(md_low=float(cn[cn.md_coverage < 0.05].wb_md_median.median()), md_high=float(cn[cn.md_coverage > 0.3].wb_md_median.median()),
            fa_low=float(cn[cn.md_coverage < 0.05].wb_fa_median.median()), fa_high=float(cn[cn.md_coverage > 0.3].wb_fa_median.median()),
            n_low=int((cn.md_coverage < 0.05).sum()), n_high=int((cn.md_coverage > 0.3).sum()))
rt = P.join(au[["wb_fa_median", "wb_md_median"]])
route_fa = rt.groupby("route").wb_fa_median.median().to_dict(); route_md = rt.groupby("route").wb_md_median.median().to_dict()
S = dict(n_excluded=int(qc.qc_fail.sum()), excluded_adni34=AUS["qc_fail_adni34"], by_reason=AUS["qc_by_reason"],
         values_masked=AUS["values_masked_among_kept"], fences=AUS["fences"],
         coverage={g: dict(median=float(cg.loc[g, "median"]), below5=float(cg.loc[g, "below5"]), new_route=float(cg.loc[g, "new_route"])) for g in cg.index},
         rho_new_route=rho, cn_low_vs_high=lohi, route_fa=route_fa, route_md=route_md,
         first={c: {lab: dq(e1, c, f, ct) for lab, f, ct in KEY} for c, _ in PAIRS},
         now={c: {lab: dq(e2, c, f, ct) for lab, f, ct in KEY} for c, _ in PAIRS},
         f1_first={c: {f"{m} {ct}": f1n(e1, c, m, ct) for m in ["AxD", "MD", "RD", "FA"] for ct in CT} for c, _ in PAIRS},
         f1_now={c: {f"{m} {ct}": f1n(e2, c, m, ct) for m in ["AxD", "MD", "RD", "FA"] for ct in CT} for c, _ in PAIRS},
         srlr_first={c: {ct: sri(s1, c, ct) for ct in CT} for c, _ in PAIRS},
         srlr_now={c: {ct: sri(s2, c, ct) for ct in CT} for c, _ in PAIRS})
json.dump(S, open(O / "qc_summary.json", "w"), indent=1)
print(pd.read_csv(T / "qc_exclusions.csv").to_string(index=False)); print()
print(pd.read_csv(T / "coverage_by_group.csv").to_string(index=False)); print()
print(pd.read_csv(T / "what_changed.csv").to_string(index=False)); print()
print(json.dumps({k: S[k] for k in ["rho_new_route", "cn_low_vs_high", "route_fa", "route_md"]}, indent=1))
