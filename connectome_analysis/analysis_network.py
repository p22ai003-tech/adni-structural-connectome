"""Network-/system-level analysis: aggregate the node-level feature families (DTI microstructure, graph
metrics, structure-microstructure coupling) and the structural connectome into canonical brain networks
(functional, Yeo-7 + subcortical/cerebellar/brainstem) and anatomical systems, then ask which network is
most affected CN -> MCI -> AD.

Keyed on the connectome MATRIX INDEX (1..166) via atlas/AAL/AAL3_network_mapping.csv. The node-level long
tables produced earlier in the refresh carry the same matrix index in their `node` column.

Outputs under ANALYSIS_ROOT/19_network_analysis/<scheme>/:
  network_microstructure_subject.csv / _stats.csv      (per subject x network FA/MD/AD/RD)
  network_graph_subject.csv          / _stats.csv      (per subject x network strength/degree/nodal_eff)
  network_coupling_subject.csv       / _stats.csv      (per subject x network within-network Spearman rho)
  block_within_between_subject.csv                      (per subject x network-pair mean_weight + density)
  block_groupmean_mean_weight.csv / block_groupmean_density.csv   (network x network, per group)
  block_cn_minus_ad_mean_weight.csv                     (contrast matrix)
  block_stats.csv                                       (per network-pair group stats)
  network_affectedness_summary.csv                      (ranked: which network is most affected, per family)
"""
from __future__ import annotations

from itertools import combinations_with_replacement
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from connectome_analysis.analysis_config import GROUP_ORDER, AnalysisPaths
from connectome_analysis.analysis_edges import load_connectome_stack
from connectome_analysis.analysis_plots import save_dataframe, write_inference_markdown
from connectome_analysis.analysis_stats import (
    art_interaction,
    bh_fdr,
    friedman_systemwise,
    group_descriptives,
    kruskal_summary,
    pairwise_robust_tests,
    permutation_group_effect,
    wilcoxon_pairwise,
)

MAPPING_CSV = Path(__file__).resolve().parents[1] / "atlas" / "AAL" / "AAL3_network_mapping.csv"
SCHEMES = {"functional": "functional_network", "anatomical": "anatomical_system"}
MICRO_METRICS = ["fa_mean", "md_mean", "ad_mean", "rd_mean"]
GRAPH_METRICS = ["strength", "degree", "nodal_eff"]
COUPLING_PAIRS = [("strength", "fa_mean"), ("strength", "md_mean"),
                  ("nodal_eff", "fa_mean"), ("nodal_eff", "md_mean")]
MIN_NODES_COUPLING = 5  # within-network Spearman needs a few nodes to be meaningful


# --------------------------------------------------------------------------------------------------- IO
def _load_mapping() -> pd.DataFrame:
    m = pd.read_csv(MAPPING_CSV)
    m["matrix_idx"] = pd.to_numeric(m["matrix_idx"], errors="coerce").astype(int)
    return m


def _node_network(mapping: pd.DataFrame, scheme_col: str) -> pd.Series:
    """matrix_idx -> network label for a scheme."""
    return mapping.set_index("matrix_idx")[scheme_col]


def _read_long(path: Path, what: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"network analysis needs {what} at {path} (run the node-level sections first)."
        )
    return pd.read_csv(path)


# ------------------------------------------------------------------------------------------------- STATS
def _metric_group_stats(sub: pd.DataFrame, value_col: str) -> dict:
    """Group stats for one network x metric column. Returns means per group + KW/perm + CN-AD effect."""
    sub = sub.dropna(subset=[value_col])
    row: dict = {"n": int(len(sub))}
    pairs = [("CN", "MCI"), ("CN", "AD"), ("MCI", "AD")]
    if sub.empty or "group" not in sub.columns:
        for g in GROUP_ORDER:
            row[f"mean_{g}"] = np.nan
        row.update({"kw_p": np.nan, "perm_p": np.nan})
        for a, b in pairs:
            tag = f"{a.lower()}_{b.lower()}"
            row.update({f"{tag}_cliffs_delta": np.nan, f"{tag}_bm_p": np.nan, f"{tag}_bm_q": np.nan})
        row["cn_ad_cliffs_delta"] = np.nan  # back-compat alias
        return row
    desc = group_descriptives(sub, value_col)
    means = dict(zip(desc["group"], desc["mean"])) if "group" in desc.columns else {}
    for g in GROUP_ORDER:
        row[f"mean_{g}"] = means.get(g, np.nan)
    omni = kruskal_summary(sub, value_col)
    perm = permutation_group_effect(sub, value_col)
    row["kw_p"] = omni.pvalue
    row["perm_p"] = perm["p_perm"]
    pw = pairwise_robust_tests(sub, value_col)  # rows ordered CN-MCI, CN-AD, MCI-AD; δ>0 => group_a>group_b
    for a, b in pairs:
        tag = f"{a.lower()}_{b.lower()}"
        r = pw[(pw["group_a"] == a) & (pw["group_b"] == b)]
        if not r.empty:
            r = r.iloc[0]
            row[f"{tag}_cliffs_delta"] = float(r["cliffs_delta"])
            row[f"{tag}_bm_p"] = float(r["bm_p"])
            row[f"{tag}_bm_q"] = float(r["bm_q"]) if "bm_q" in r else np.nan
        else:
            row[f"{tag}_cliffs_delta"] = row[f"{tag}_bm_p"] = row[f"{tag}_bm_q"] = np.nan
    row["cn_ad_cliffs_delta"] = row.get("cn_ad_cliffs_delta", np.nan)  # back-compat alias used by affectedness
    return row


def _stats_over(long_df: pd.DataFrame, group_keys: list[str], value_col: str,
                covars: pd.DataFrame) -> pd.DataFrame:
    """Run _metric_group_stats for every combination of `group_keys` (e.g. network, metric)."""
    rows = []
    merged = long_df.merge(covars, on="subject_id", how="left", suffixes=("", "_cov"))
    if "group" not in merged.columns and "group_cov" in merged.columns:
        merged["group"] = merged["group_cov"]
    for keys, sub in merged.groupby(group_keys):
        keys = keys if isinstance(keys, tuple) else (keys,)
        need = ["subject_id", "group", "age", "sex", "phase", value_col]
        s = sub[[c for c in need if c in sub.columns]].copy()
        rec = dict(zip(group_keys, keys))
        rec.update(_metric_group_stats(s, value_col))
        rows.append(rec)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------- FEATURE BUILDERS
def _network_microstructure(node_micro: pd.DataFrame, net: pd.Series) -> pd.DataFrame:
    """Per subject x network x metric n_edges-weighted mean of node FA/MD/AD/RD."""
    df = node_micro[node_micro["metric"].isin(MICRO_METRICS)].copy()
    df["network"] = df["node"].map(net)
    df = df.dropna(subset=["network", "value"])
    w = df["n_edges"].clip(lower=0).fillna(0) if "n_edges" in df.columns else 1.0
    df["_wv"] = df["value"] * w
    df["_w"] = w
    g = df.groupby(["subject_id", "group", "network", "metric"], as_index=False).agg(
        _wv=("_wv", "sum"), _w=("_w", "sum"), n_nodes=("node", "nunique"))
    g["value"] = np.where(g["_w"] > 0, g["_wv"] / g["_w"], np.nan)
    return g[["subject_id", "group", "network", "metric", "value", "n_nodes"]]


def _network_graph(node_graph: pd.DataFrame, net: pd.Series) -> pd.DataFrame:
    """Per subject x network mean of node strength/degree/nodal_eff -> long over metric."""
    df = node_graph.copy()
    df["network"] = df["node"].map(net)
    df = df.dropna(subset=["network"])
    g = df.groupby(["subject_id", "group", "network"], as_index=False).agg(
        {**{m: "mean" for m in GRAPH_METRICS}, "node": "nunique"}).rename(columns={"node": "n_nodes"})
    long = g.melt(id_vars=["subject_id", "group", "network", "n_nodes"],
                  value_vars=GRAPH_METRICS, var_name="metric", value_name="value")
    return long


def _network_coupling(node_graph: pd.DataFrame, node_micro: pd.DataFrame, net: pd.Series) -> pd.DataFrame:
    """Per subject x network within-network Spearman rho(node graph metric, node DTI metric)."""
    micro_wide = node_micro[node_micro["metric"].isin(MICRO_METRICS)].pivot_table(
        index=["subject_id", "node"], columns="metric", values="value").reset_index()
    base = node_graph[["subject_id", "group", "node", *GRAPH_METRICS]].merge(
        micro_wide, on=["subject_id", "node"], how="inner")
    base["network"] = base["node"].map(net)
    base = base.dropna(subset=["network"])
    rows = []
    for (sid, grp, network), sub in base.groupby(["subject_id", "group", "network"]):
        for nodal_metric, micro_metric in COUPLING_PAIRS:
            if micro_metric not in sub.columns:
                continue
            v = sub[[nodal_metric, micro_metric]].replace([np.inf, -np.inf], np.nan).dropna()
            rho = np.nan
            if len(v) >= MIN_NODES_COUPLING and v[nodal_metric].nunique() > 1 and v[micro_metric].nunique() > 1:
                rho = float(stats.spearmanr(v[nodal_metric], v[micro_metric]).correlation)
            rows.append({"subject_id": sid, "group": grp, "network": network,
                         "metric": f"{nodal_metric}__{micro_metric}", "value": rho,
                         "n_nodes": int(len(v))})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------- BLOCK CONNECTIVITY
def _block_connectivity(paths: AnalysisPaths, master: pd.DataFrame, net: pd.Series,
                        networks: list[str]) -> pd.DataFrame:
    """Per subject x network-pair: mean_weight (fd_sum over existing edges) + density (count nonzero frac)."""
    fd = load_connectome_stack(paths, master, connectome_type="fd_sum")
    ct = load_connectome_stack(paths, master, connectome_type="count")
    ct_by_sid = {sid: m for sid, m in zip(ct.subject_ids, ct.matrices)}
    # network membership as matrix-index arrays (matrix idx is 1-based -> 0-based positions)
    members = {nw: np.array([i - 1 for i in net.index[net == nw]], dtype=int) for nw in networks}
    rows = []
    for sid, grp, W in zip(fd.subject_ids, fd.groups, fd.matrices):
        C = ct_by_sid.get(sid)
        n = W.shape[0]
        for a, b in combinations_with_replacement(networks, 2):
            ia, ib = members[a], members[b]
            ia = ia[ia < n]; ib = ib[ib < n]
            if ia.size == 0 or ib.size == 0:
                continue
            wsub = W[np.ix_(ia, ib)]; csub = C[np.ix_(ia, ib)] if C is not None else wsub
            if a == b:  # within: upper triangle, exclude diagonal
                iu = np.triu_indices(ia.size, 1)
                wvals = wsub[iu]; cvals = csub[iu]
            else:
                wvals = wsub.ravel(); cvals = csub.ravel()
            if wvals.size == 0:
                continue
            pos = np.isfinite(wvals) & (wvals > 0)
            rows.append({
                "subject_id": sid, "group": grp, "net_a": a, "net_b": b,
                "kind": "within" if a == b else "between",
                "mean_weight": float(wvals[pos].mean()) if pos.any() else 0.0,
                "density": float((np.isfinite(cvals) & (cvals > 0)).sum() / cvals.size),
                "n_pairs": int(cvals.size),
            })
    return pd.DataFrame(rows)


def _affectedness(stats_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Combine per-family stats into one ranked 'which network is most affected' table (FDR per family)."""
    parts = []
    for family, df in stats_frames.items():
        if df is None or df.empty:
            continue
        d = df.copy()
        d["feature_family"] = family
        d["q_kw"] = bh_fdr(d["kw_p"].to_numpy())
        d["abs_cn_ad_delta"] = d["cn_ad_cliffs_delta"].abs()
        parts.append(d)
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    keep = ["feature_family", "network", "metric", "n", "mean_CN", "mean_MCI", "mean_AD",
            "kw_p", "q_kw", "perm_p",
            "cn_mci_cliffs_delta", "cn_mci_bm_p", "cn_mci_bm_q",
            "cn_ad_cliffs_delta", "cn_ad_bm_p", "cn_ad_bm_q",
            "mci_ad_cliffs_delta", "mci_ad_bm_p", "mci_ad_bm_q",
            "abs_cn_ad_delta"]
    out = out[[c for c in keep if c in out.columns]]
    return out.sort_values(["feature_family", "abs_cn_ad_delta"], ascending=[True, False]).reset_index(drop=True)


def _systemwise(subject_df: pd.DataFrame, family: str, excl: float = 0.30):
    """System-wise (network-vs-network) stats on a per-subject network frame (WITHIN-SUBJECT paired factor).
    Per metric x group{CN,MCI,AD,Pooled}: structural-exclude networks >excl NaN, Friedman omnibus +
    pairwise Wilcoxon (r_rb, BH-FDR). Per metric: ART group x network interaction.
    Returns (pairwise_df, omnibus_df, interaction_df)."""
    need = {"subject_id", "group", "network", "metric", "value"}
    if subject_df is None or subject_df.empty or not need.issubset(subject_df.columns):
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    pair_parts, omni_rows, inter_rows = [], [], []
    for metric, dm in subject_df.groupby("metric"):
        allwide = dm.pivot_table(index="subject_id", columns="network", values="value")
        keep_global = [c for c in allwide.columns if allwide[c].isna().mean() <= excl]
        it = art_interaction(dm[dm["network"].isin(keep_global)][["value", "group", "network"]])
        inter_rows.append({"family": family, "metric": metric, "art_F": it["art_F"],
                           "art_p": it["art_p"], "n": it["n"]})
        for grp in ["CN", "MCI", "AD", "Pooled"]:
            sub = dm if grp == "Pooled" else dm[dm["group"] == grp]
            if sub.empty:
                continue
            wide = sub.pivot_table(index="subject_id", columns="network", values="value")
            keep = [c for c in wide.columns if wide[c].isna().mean() <= excl]
            excluded = [c for c in wide.columns if c not in keep]
            wide = wide[keep]
            fr = friedman_systemwise(wide)
            omni_rows.append({"family": family, "metric": metric, "group": grp, **fr,
                              "excluded_networks": ";".join(sorted(excluded))})
            pw = wilcoxon_pairwise(wide)
            if not pw.empty:
                pw.insert(0, "group", grp)
                pw.insert(0, "metric", metric)
                pw.insert(0, "family", family)
                pair_parts.append(pw)
    pairwise = pd.concat(pair_parts, ignore_index=True) if pair_parts else pd.DataFrame()
    return pairwise, pd.DataFrame(omni_rows), pd.DataFrame(inter_rows)


# -------------------------------------------------------------------------------------------- ENTRYPOINT
def run_network_analysis(paths: AnalysisPaths, master: pd.DataFrame) -> dict:
    out_root = paths.section_dir("19", "network_analysis")
    out_root.mkdir(parents=True, exist_ok=True)
    mapping = _load_mapping()
    save_dataframe(mapping, out_root / "network_mapping_used.csv")

    node_micro = _read_long(
        paths.section_dir("04", "node_microstructure_live") / "live_node_microstructure_long.csv",
        "node microstructure long table")
    node_graph = _read_long(
        paths.section_dir("07", "node_graph_live") / "live_node_graph_long.csv",
        "node graph long table")
    covars = master[[c for c in ["subject_id", "group", "age", "sex", "phase"] if c in master.columns]] \
        .drop_duplicates("subject_id")

    summary: dict = {"schemes": {}}
    for scheme, scheme_col in SCHEMES.items():
        sdir = out_root / scheme
        sdir.mkdir(parents=True, exist_ok=True)
        net = _node_network(mapping, scheme_col)
        networks = [n for n in mapping[scheme_col].drop_duplicates().tolist()]

        micro = _network_microstructure(node_micro, net)
        graph = _network_graph(node_graph, net)
        coupling = _network_coupling(node_graph, node_micro, net)
        save_dataframe(micro, sdir / "network_microstructure_subject.csv")
        save_dataframe(graph, sdir / "network_graph_subject.csv")
        save_dataframe(coupling, sdir / "network_coupling_subject.csv")

        micro_stats = _stats_over(micro, ["network", "metric"], "value", covars)
        graph_stats = _stats_over(graph, ["network", "metric"], "value", covars)
        coup_stats = _stats_over(coupling, ["network", "metric"], "value", covars)
        save_dataframe(micro_stats, sdir / "network_microstructure_stats.csv")
        save_dataframe(graph_stats, sdir / "network_graph_stats.csv")
        save_dataframe(coup_stats, sdir / "network_coupling_stats.csv")

        # system-wise (network-vs-network, within-subject paired) stats for each feature family
        sw_pair, sw_omni, sw_inter = [], [], []
        for fam, fdf in (("microstructure", micro), ("graph", graph), ("coupling", coupling)):
            pw, om, it = _systemwise(fdf, fam)
            if not pw.empty:
                sw_pair.append(pw)
            if not om.empty:
                sw_omni.append(om)
            if not it.empty:
                sw_inter.append(it)
        sw_pair_df = pd.concat(sw_pair, ignore_index=True) if sw_pair else pd.DataFrame()
        sw_omni_df = pd.concat(sw_omni, ignore_index=True) if sw_omni else pd.DataFrame()
        sw_inter_df = pd.concat(sw_inter, ignore_index=True) if sw_inter else pd.DataFrame()
        if not sw_inter_df.empty:
            sw_inter_df["art_q"] = bh_fdr(sw_inter_df["art_p"].to_numpy())
        save_dataframe(sw_pair_df, sdir / "network_systemwise_pairwise.csv")
        save_dataframe(sw_omni_df, sdir / "network_systemwise_omnibus.csv")
        save_dataframe(sw_inter_df, sdir / "network_systemwise_interaction.csv")

        # block connectivity
        block = _block_connectivity(paths, master, net, networks)
        save_dataframe(block, sdir / "block_within_between_subject.csv")
        # group-mean matrices + CN-AD contrast (mean_weight) + per-block stats
        for value_col, fname in [("mean_weight", "block_groupmean_mean_weight.csv"),
                                 ("density", "block_groupmean_density.csv")]:
            gm = block.groupby(["group", "net_a", "net_b"], as_index=False)[value_col].mean()
            save_dataframe(gm, sdir / fname)
        block["pair"] = block["net_a"] + " | " + block["net_b"]
        block_stats = _stats_over(block, ["net_a", "net_b", "kind"], "mean_weight", covars)
        save_dataframe(block_stats, sdir / "block_stats.csv")

        affected = _affectedness({
            "microstructure": micro_stats, "graph": graph_stats,
            "coupling": coup_stats,
            "connectivity": block_stats.assign(network=block_stats["net_a"] + " | " + block_stats["net_b"],
                                               metric=block_stats["kind"]),
        })
        save_dataframe(affected, sdir / "network_affectedness_summary.csv")
        summary["schemes"][scheme] = {
            "networks": networks,
            "n_subjects_micro": int(micro["subject_id"].nunique()),
            "n_subjects_block": int(block["subject_id"].nunique()),
            "top_affected": affected.head(8).to_dict("records") if not affected.empty else [],
        }

    # inference note (functional scheme headline); helper bulletises each line.
    fa = pd.read_csv(out_root / "functional" / "network_affectedness_summary.csv")
    lines = ["Network-level analysis on the dense cohort. 'Affectedness' ranks networks by the CN-vs-AD "
             "Cliff's delta within each feature family (BH-FDR across networks)."]
    for fam in fa["feature_family"].drop_duplicates():
        top = fa[fa["feature_family"] == fam].head(3)
        bits = ", ".join(f"{r.network}/{r.metric} (d={r.cn_ad_cliffs_delta:+.2f}, q={r.q_kw:.2g})"
                         for r in top.itertuples())
        lines.append(f"{fam}: {bits}")
    write_inference_markdown(lines, out_root / "network_inference.md")
    return summary
