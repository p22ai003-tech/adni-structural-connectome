#!/usr/bin/env python
"""Part A — what is exception-specific?

A1  Tier-delta matrix: for every network x base-measure x contrast, the like-for-like
    difference |delta_exception| - |delta_all-edges-same-range|, plus vs whole-brain.
A2  Subject-level exception-ARCHITECTURE features (no conventional counterpart):
    within-network fraction, hub participation, homotopic share, consensus-core share,
    per-range exception counts/rates/strengths, exc/non-exc log ratios, EDR lambda.
    Group stats on all of it (3 contrasts, BH-FDR) + an ADNI-3-only sensitivity column
    so phase-vulnerable rows are visible inline.

Outputs (hcp_analysis/):
  exception_tier_delta.csv           A1 long table
  exception_architecture_subject.csv A2 per-subject features (input to Part B / F4)
  exception_architecture_stats.csv   A2 group stats incl. adni3-only sensitivity
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

# Paths resolve through sc_paths: project-relative, overridable with --out, and
# defaulting to the analysis-tree section the dashboard actually reads. Writing
# straight there is what removes the old manual copy step.
try:
    import sc_paths
except ModuleNotFoundError:  # loose script run from outside hcp_analysis/
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent))
    import sc_paths

try:
    import sc_exclusions
except ModuleNotFoundError:
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import sc_exclusions

_P = sc_paths.resolve_for_script("ml")

ANALYSIS = _P.analysis
OUT = _P.out
FUNC = _P.func
NETWORKS = ["Visual", "Somatomotor", "DorsalAttention", "Salience_VAN", "Limbic",
            "Frontoparietal", "DMN", "Subcortical", "Cerebellar", "Brainstem"]
MEASURES = ["Strength", "Degree", "FA", "MD", "RD", "AxD"]
CONTRASTS = (("cn_mci", "CN", "MCI"), ("cn_ad", "CN", "AD"), ("mci_ad", "MCI", "AD"))
# Loaded from a gitignored local file: the identifier is ADNI-restricted.
OUTLIER = sc_exclusions.warn_if_empty(__file__.rsplit("/", 1)[-1])


def cliffs(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if a.size == 0 or b.size == 0:
        return float("nan")
    return float(((a[:, None] > b[None, :]).sum() - (a[:, None] < b[None, :]).sum()) / (a.size * b.size))


def bh(p):
    v = np.asarray(p, float); n = v.size
    if n == 0:
        return []
    order = np.argsort(v); out = np.empty(n); run = 1.0
    for k, i in enumerate(order[::-1]):
        run = min(run, v[i] * n / (n - k)); out[i] = run
    return out.tolist()


def a1_tier_delta() -> None:
    stats_b = pd.read_csv(FUNC / "network_range_restricted_stats.csv")
    stats_c = pd.read_csv(FUNC / "network_exception_measures_stats.csv")
    mic = pd.read_csv(FUNC / "network_microstructure_stats.csv")
    mic["measure"] = mic.metric.map({"fa_mean": "FA", "md_mean": "MD", "rd_mean": "RD", "ad_mean": "AxD"})
    gr = pd.read_csv(FUNC / "network_graph_stats.csv")
    gr["measure"] = gr.metric.map({"strength": "Strength", "degree": "Degree"})
    wb = pd.concat([mic, gr]).dropna(subset=["measure"])

    def get(df, net, meas):
        row = df[(df.network == net) & (df.measure == meas)]
        return row.iloc[0] if len(row) else None

    rows = []
    for ck, _f, _s in CONTRASTS:
        d, q = f"{ck}_cliffs_delta", f"{ck}_bm_q"
        for net in NETWORKS:
            for rng in ("SR", "LR"):
                for meas in MEASURES:
                    exc = get(stats_c, net, f"{rng}exc-{meas}")
                    base = get(stats_b, net, f"{rng}-{meas}")
                    whole = get(wb, net, meas)
                    if exc is None or base is None:
                        continue
                    rows.append(dict(
                        contrast=ck, network=net, range=rng, measure=meas,
                        delta_exc=abs(exc[d]), q_exc=exc[q],
                        delta_range=abs(base[d]), q_range=base[q],
                        delta_whole=abs(whole[d]) if whole is not None else np.nan,
                        q_whole=whole[q] if whole is not None else np.nan,
                        gain_vs_range=abs(exc[d]) - abs(base[d]),
                        gain_vs_whole=(abs(exc[d]) - abs(whole[d])) if whole is not None else np.nan,
                        pop_up=bool(exc[q] < 0.05 and base[q] >= 0.05),
                        drop_out=bool(exc[q] >= 0.05 and base[q] < 0.05),
                    ))
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "exception_tier_delta.csv", index=False)
    for ck, _f, _s in CONTRASTS:
        sub = frame[frame.contrast == ck]
        print(f"A1 {ck}: mean gain vs range {sub.gain_vs_range.mean():+.3f} | "
              f"pop-ups {int(sub.pop_up.sum())} | drop-outs {int(sub.drop_out.sum())}", flush=True)


def a2_architecture() -> None:
    master = pd.read_csv(ANALYSIS / "00_master/master_cohort.csv")
    grp = master.set_index("subject_id")["group"]
    phase_col = "phase" if "phase" in master.columns else "Phase"
    phase = master.set_index("subject_id")[phase_col].astype(str)

    lut = pd.read_csv(_P.aal_node_map)
    true_name = dict(zip(lut["new_id"].astype(int), lut["name"]))
    mapping = pd.read_csv(ANALYSIS / "19_network_analysis/network_mapping_used.csv")
    idx2net = dict(zip(mapping["matrix_idx"].astype(int), mapping["functional_network"]))

    def homotopic(i, j):
        a, b = true_name.get(i, ""), true_name.get(j, "")
        return (a[:-2] == b[:-2]) and {a[-2:], b[-2:]} == {"_L", "_R"}

    edges = pd.read_csv(ANALYSIS / "17_edr_exceptions/edr_exception_edge_level_fd_sum_len_mean.csv")
    edges = edges[edges.exception_flag == 1].copy()
    edges["pair"] = edges.apply(lambda r: (min(int(r.i), int(r.j)), max(int(r.i), int(r.j))), axis=1)
    # consensus core: pairs present in > 50% of all subjects
    recur = edges.groupby("pair")["subject_id"].nunique()
    core = set(recur[recur > 0.5 * edges.subject_id.nunique()].index)
    print(f"A2 consensus core edges (>50% of subjects): {len(core)}", flush=True)

    edges["same_net"] = [idx2net.get(int(i)) == idx2net.get(int(j)) for i, j in zip(edges.i, edges.j)]
    edges["homo"] = [homotopic(int(i), int(j)) for i, j in zip(edges.i, edges.j)]
    edges["is_core"] = edges["pair"].isin(core)

    per = edges.groupby("subject_id").agg(
        n_exc=("pair", "size"),
        within_net_frac=("same_net", "mean"),
        homotopic_frac=("homo", "mean"),
        core_frac=("is_core", "mean"),
    )
    part = edges.melt(id_vars=["subject_id"], value_vars=["i", "j"], value_name="node")
    per["participation_frac"] = part.groupby("subject_id")["node"].nunique() / 166.0

    subj = pd.read_csv(ANALYSIS / "17_edr_exceptions/edr_exception_subject_level_fd_sum_len_mean.csv")
    keep = [c for c in subj.columns if c in (
        "subject_id", "edr_lambda", "edr_r",
        "sr_exception_rate", "lr_exception_rate", "mr_exception_rate",
        "sr_exception_pct", "lr_exception_pct", "mr_exception_pct",
        "sr_exception_strength", "lr_exception_strength",
        "sr_w_mean", "lr_w_mean",
    )]
    subj = subj[keep].set_index("subject_id")
    for rng in ("sr", "lr"):
        s, w = f"{rng}_exception_strength", f"{rng}_w_mean"
        if s in subj.columns and w in subj.columns:
            with np.errstate(divide="ignore", invalid="ignore"):
                subj[f"{rng}_exc_log_ratio"] = np.log(subj[s]) - np.log(subj[w])

    feats = per.join(subj, how="outer").join(grp, how="inner").join(phase.rename("phase"))
    feats = feats[~feats.index.isin(OUTLIER)]
    feats.to_csv(OUT / "exception_architecture_subject.csv")
    print(f"A2 subject features: {feats.shape[0]} subjects x {feats.shape[1] - 2} features", flush=True)

    adni3 = feats[feats.phase.str.contains("3", na=False)]
    rows = []
    feature_cols = [c for c in feats.columns if c not in ("group", "phase")]
    for col in feature_cols:
        arrays = {g: feats.loc[feats.group == g, col].dropna().to_numpy(float) for g in ("CN", "MCI", "AD")}
        if min(map(len, arrays.values())) < 5:
            continue
        rec = dict(feature=col,
                   mean_CN=np.mean(arrays["CN"]), mean_MCI=np.mean(arrays["MCI"]), mean_AD=np.mean(arrays["AD"]),
                   n_CN=len(arrays["CN"]), n_MCI=len(arrays["MCI"]), n_AD=len(arrays["AD"]))
        for ck, f, s in CONTRASTS:
            try:
                bm = float(sps.brunnermunzel(arrays[f], arrays[s], alternative="two-sided").pvalue)
            except ValueError:
                bm = float("nan")
            rec[f"{ck}_cliffs_delta"] = cliffs(arrays[f], arrays[s])
            rec[f"{ck}_bm_p"] = bm
        a3 = {g: adni3.loc[adni3.group == g, col].dropna().to_numpy(float) for g in ("CN", "MCI", "AD")}
        for ck, f, s in (("cn_ad", "CN", "AD"), ("mci_ad", "MCI", "AD")):
            if min(len(a3[f]), len(a3[s])) >= 5:
                try:
                    rec[f"adni3_{ck}_p"] = float(sps.brunnermunzel(a3[f], a3[s], alternative="two-sided").pvalue)
                except ValueError:
                    rec[f"adni3_{ck}_p"] = float("nan")
                rec[f"adni3_{ck}_delta"] = cliffs(a3[f], a3[s])
        rows.append(rec)
    stats = pd.DataFrame(rows)
    for ck, _f, _s in CONTRASTS:
        stats[f"{ck}_bm_q"] = bh(stats[f"{ck}_bm_p"].tolist())
    stats.to_csv(OUT / "exception_architecture_stats.csv", index=False)
    sig = stats[(stats[[f"{c}_bm_q" for c, _, _ in CONTRASTS]] < 0.05).any(axis=1)]
    print(f"A2 stats: {len(stats)} features, {len(sig)} with >=1 significant contrast", flush=True)
    for _, r in sig.iterrows():
        best = min((("cn_mci", r.cn_mci_bm_q), ("cn_ad", r.cn_ad_bm_q), ("mci_ad", r.mci_ad_bm_q)), key=lambda t: t[1])
        a3p = r.get(f"adni3_{best[0]}_p", np.nan)
        print(f"    {r.feature:24} best={best[0]} q={best[1]:.4f} | adni3 p={a3p if a3p == a3p else float('nan'):.3f}"
              if a3p == a3p else f"    {r.feature:24} best={best[0]} q={best[1]:.4f} | adni3 n/a", flush=True)


if __name__ == "__main__":
    a1_tier_delta()
    a2_architecture()
