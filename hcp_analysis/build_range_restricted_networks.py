#!/usr/bin/env python
"""Per-network measures restricted to SR and LR tract ranges, plus SR/LR EDR exceptions.

Produces the data behind two extra table-groups on the LR/SR page:
  Group B  "SR/LR tracts"      - node-level strength/degree/FA/MD/RD/AxD computed using ONLY the
                                 edges whose tract length falls in the SR (<=CN Q1) or LR (>CN Q3)
                                 range, then aggregated to functional networks.
  Group C  "SR/LR exceptions"  - per-network EDR exception burden + strength within SR and LR,
                                 aggregated from the edge-level exception artifact.

Aggregation matches connectome_analysis/analysis_network.py so the three groups are comparable:
graph metrics = plain mean over the network's nodes; microstructure = edge-count-weighted mean.

Outputs (long format, one row per network x measure):
  hcp_analysis/network_range_restricted_stats.csv
  hcp_analysis/network_exception_range_stats.csv
"""
from __future__ import annotations

import sys
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

_P = sc_paths.resolve_for_script("network")

ANALYSIS = _P.analysis
CONN = _P.connectomes
OUT = _P.out
NETWORKS = [
    "Visual", "Somatomotor", "DorsalAttention", "Salience_VAN", "Limbic",
    "Frontoparietal", "DMN", "Subcortical", "Cerebellar", "Brainstem",
]
MICRO = {"fa_mean": "FA", "md_mean": "MD", "rd_mean": "RD", "ad_mean": "AxD"}
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 0


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, float); b = np.asarray(b, float)
    if a.size == 0 or b.size == 0:
        return float("nan")
    return float(((a[:, None] > b[None, :]).sum() - (a[:, None] < b[None, :]).sum()) / (a.size * b.size))


def bh_fdr(p: list[float]) -> list[float]:
    v = np.asarray(p, float); n = v.size
    if n == 0:
        return []
    order = np.argsort(v); out = np.empty(n); run = 1.0
    for k, i in enumerate(order[::-1]):
        run = min(run, v[i] * n / (n - k)); out[i] = run
    return out.tolist()


# the three pairwise contrasts, matching the precomputed section-19 column names
CONTRASTS = (("cn_mci", "CN", "MCI"), ("cn_ad", "CN", "AD"), ("mci_ad", "MCI", "AD"))


def group_stats(frame: pd.DataFrame, value: str, keys: list[str]) -> pd.DataFrame:
    """CN/MCI/AD stats per key combination, BH-FDR corrected within each measure.

    Emits all three pairwise contrasts (CN-MCI, CN-AD, MCI-AD) so the dashboard can
    switch between them, matching the column names section 19 already uses.
    """
    rows = []
    for key, sub in frame.groupby(keys):
        arrays = {g: sub.loc[sub["group"] == g, value].dropna().to_numpy(float) for g in ("CN", "MCI", "AD")}
        if min(len(arrays["CN"]), len(arrays["MCI"]), len(arrays["AD"])) < 5:
            continue
        rec = dict(zip(keys, key if isinstance(key, tuple) else (key,)))
        rec.update(
            mean_CN=float(np.mean(arrays["CN"])), mean_MCI=float(np.mean(arrays["MCI"])),
            mean_AD=float(np.mean(arrays["AD"])),
            kw_p=float(sps.kruskal(arrays["CN"], arrays["MCI"], arrays["AD"]).pvalue),
            n_CN=len(arrays["CN"]), n_MCI=len(arrays["MCI"]), n_AD=len(arrays["AD"]),
        )
        for prefix, first, second in CONTRASTS:
            try:
                bm = float(sps.brunnermunzel(arrays[first], arrays[second], alternative="two-sided").pvalue)
            except ValueError:
                bm = float("nan")
            rec[f"{prefix}_cliffs_delta"] = cliffs_delta(arrays[first], arrays[second])
            rec[f"{prefix}_bm_p"] = bm
        rows.append(rec)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    # correct within each measure across networks, per contrast (section-19 scope)
    for prefix, _first, _second in CONTRASTS:
        out[f"{prefix}_bm_q"] = np.nan
        for _measure, idx in out.groupby("measure").groups.items():
            out.loc[idx, f"{prefix}_bm_q"] = bh_fdr(out.loc[idx, f"{prefix}_bm_p"].tolist())
    return out


def main() -> None:
    mapping = pd.read_csv(ANALYSIS / "19_network_analysis/network_mapping_used.csv")
    idx2net = dict(zip(mapping["matrix_idx"].astype(int), mapping["functional_network"]))
    # join by matrix index, NOT atlas_label: the 17_edr artifacts carry shifted ROI names
    # for rows >= 35 (stale AAL3_labels.csv upstream); the node column is the true index.
    label2net = dict(zip(mapping["matrix_idx"].astype(int), mapping["functional_network"]))

    thresholds = pd.read_csv(ANALYSIS / "17_edr_exceptions/edr_exception_thresholds.csv").iloc[0]
    sr_max = float(thresholds["short_max_mm"]); lr_min = float(thresholds["long_min_mm"])
    print(f"CN-referenced ranges: SR <= {sr_max:.2f} mm | LR > {lr_min:.2f} mm", flush=True)

    master = pd.read_csv(ANALYSIS / "00_master/master_cohort.csv")
    subjects = []
    remapped = 0
    for _, row in master.iterrows():
        sid = str(row["subject_id"])
        fsid = f"{sid}_I{row['Image ID']}"
        if not (CONN / f"SC_AAL166_{fsid}_len_mean.csv").is_file():
            # master_cohort's "Image ID" is wrong for 32 AD subjects; the processed
            # session is on disk under a different image id. Resolve it rather than
            # dropping them (dropping would cut AD from 78 to 46 and bias the table).
            alts = sorted(CONN.glob(f"SC_AAL166_{sid}_I*_len_mean.csv"))
            if alts:
                fsid = alts[0].name[len("SC_AAL166_"):-len("_len_mean.csv")]
                remapped += 1
        subjects.append((sid, str(row["group"]), fsid))
    print(f"subjects: {len(subjects)} (image-id remapped: {remapped})", flush=True)
    if LIMIT:
        subjects = subjects[:LIMIT]

    # exception edge list per participant, for the exception-restricted measures
    edge_exc = pd.read_csv(
        ANALYSIS / "17_edr_exceptions/edr_exception_edge_level_fd_sum_len_mean.csv"
    )
    edge_exc = edge_exc[edge_exc["exception_flag"] == 1]
    exc_by_subject: dict[str, dict[str, list[tuple[int, int]]]] = {}
    for (sid_key, cls), sub in edge_exc.groupby(["subject_id", "edr_class"]):
        exc_by_subject.setdefault(str(sid_key), {})[str(cls)] = list(
            zip(sub["i"].astype(int), sub["j"].astype(int))
        )
    print(f"exception edges loaded for {len(exc_by_subject)} subjects", flush=True)

    # ---------- Group B: range-restricted tract measures ----------
    records = []
    exc_records = []
    missing = 0
    for n_done, (sid, grp, fsid) in enumerate(subjects, 1):
        try:
            length = pd.read_csv(CONN / f"SC_AAL166_{fsid}_len_mean.csv", header=None).to_numpy(float)
            weight = pd.read_csv(CONN / f"SC_AAL166_{fsid}_fd_sum.csv", header=None).to_numpy(float)
            micro = {
                key: pd.read_csv(CONN / f"SC_AAL166_{fsid}_{key}.csv", header=None).to_numpy(float)
                for key in MICRO
            }
        except (FileNotFoundError, ValueError):
            missing += 1
            continue
        n = length.shape[0]
        valid = (length > 0) & (weight > 0)
        np.fill_diagonal(valid, False)
        masks = {"SR": valid & (length <= sr_max), "LR": valid & (length > lr_min)}

        # exception-edge masks: only the EDR exceptions, split by their own range class
        exc_masks: dict[str, np.ndarray] = {}
        for rng in ("SR", "LR"):
            em = np.zeros_like(valid, dtype=bool)
            for i, j in exc_by_subject.get(sid, {}).get(rng, []):
                if 1 <= i <= n and 1 <= j <= n:
                    em[i - 1, j - 1] = True
                    em[j - 1, i - 1] = True
            exc_masks[rng] = em & valid

        for rng, mask in {**masks, **{f"{k}exc": v for k, v in exc_masks.items()}}.items():
            deg = mask.sum(1).astype(float)
            strength = np.where(mask, weight, 0.0).sum(1)
            # Average each DTI metric ONLY over edges where that metric is actually
            # defined. The fa/md/rd/ad matrices carry 0 where the metric is undefined
            # even when fd_sum > 0; dividing by the full edge count dilutes the mean,
            # and because AD connectomes are sparser the dilution is group-dependent
            # and can flip the direction of the effect.
            micro_node: dict[str, np.ndarray] = {}
            micro_count: dict[str, np.ndarray] = {}
            for key, lbl in MICRO.items():
                arr = micro[key]
                vmask = mask & np.isfinite(arr) & (arr > 0)
                vdeg = vmask.sum(1).astype(float)
                with np.errstate(invalid="ignore", divide="ignore"):
                    micro_node[lbl] = np.where(
                        vdeg > 0,
                        np.where(vmask, arr, 0.0).sum(1) / np.maximum(vdeg, 1.0),
                        np.nan,
                    )
                micro_count[lbl] = vdeg
            for node in range(n):
                net = idx2net.get(node + 1)
                if net not in NETWORKS:
                    continue
                record = dict(
                    subject_id=sid, group=grp, network=net, node=node + 1, rng=rng,
                    degree=deg[node], strength=strength[node], n_edges=deg[node],
                )
                for lbl in MICRO.values():
                    record[lbl] = micro_node[lbl][node]
                    record[f"n_{lbl}"] = micro_count[lbl][node]
                (exc_records if rng.endswith("exc") else records).append(record)
        if n_done % 50 == 0:
            print(f"  ...{n_done}/{len(subjects)} subjects", flush=True)

    def aggregate_to_networks(node_rows: list[dict]) -> pd.DataFrame:
        """node-level -> per subject x network x measure long frame (matches section 19)."""
        nodes = pd.DataFrame(node_rows)
        keys = ["subject_id", "group", "network", "rng"]
        graph_net = (
            nodes.groupby(keys, as_index=False)
            .agg(Strength=("strength", "mean"), Degree=("degree", "mean"))
            .melt(id_vars=keys, var_name="measure", value_name="value")
        )
        micro_rows = []
        for lbl in MICRO.values():
            sub = nodes.dropna(subset=[lbl]).copy()
            # weight each node's metric by the number of edges that metric was defined on
            sub["_wsum"] = sub[lbl] * sub[f"n_{lbl}"]
            agg = sub.groupby(keys, as_index=False).agg(
                _num=("_wsum", "sum"), _den=(f"n_{lbl}", "sum"))
            agg["value"] = np.where(agg["_den"] > 0, agg["_num"] / agg["_den"], np.nan)
            agg["measure"] = lbl
            micro_rows.append(agg[keys + ["measure", "value"]])
        out = pd.concat([graph_net] + micro_rows, ignore_index=True)
        out["measure"] = out["rng"] + "-" + out["measure"]
        return out

    print(f"node rows: {len(records)} tract | {len(exc_records)} exception "
          f"| subjects missing matrices: {missing}", flush=True)

    subj_b = aggregate_to_networks(records)
    subj_b.to_csv(OUT / "network_range_restricted_subject.csv", index=False)
    stats_b = group_stats(subj_b, "value", ["network", "measure"])
    stats_b.to_csv(OUT / "network_range_restricted_stats.csv", index=False)
    print(f"wrote network_range_restricted_stats.csv ({len(stats_b)} rows) + subject-level ({len(subj_b)})", flush=True)

    subj_c = aggregate_to_networks(exc_records)
    subj_c.to_csv(OUT / "network_exception_measures_subject.csv", index=False)
    stats_cm = group_stats(subj_c, "value", ["network", "measure"])
    stats_cm.to_csv(OUT / "network_exception_measures_stats.csv", index=False)
    print(f"wrote network_exception_measures_stats.csv ({len(stats_cm)} rows) + subject-level ({len(subj_c)})", flush=True)

    # ---------- Group C: SR/LR exceptions per network ----------
    edges = pd.read_csv(ANALYSIS / "17_edr_exceptions/edr_exception_edge_level_fd_sum_len_mean.csv")
    node_lvl = pd.read_csv(ANALYSIS / "17_edr_exceptions/edr_exception_node_level_fd_sum_len_mean.csv")
    node_lvl["network"] = pd.to_numeric(node_lvl["node"], errors="coerce").map(label2net)
    node_lvl = node_lvl[node_lvl["network"].isin(NETWORKS)]

    exc_rows = []
    for rng, prefix in (("SR", "sr"), ("LR", "lr")):
        frame = node_lvl.copy()
        frame["_ec"] = pd.to_numeric(frame[f"{prefix}_exception_count"], errors="coerce").fillna(0.0)
        frame["_tc"] = pd.to_numeric(frame[f"{prefix}_edge_count"], errors="coerce").fillna(0.0)
        frame["_st"] = pd.to_numeric(frame[f"{prefix}_exception_strength"], errors="coerce")
        frame["_sw"] = frame["_st"].notna().astype(float) * frame["_ec"]
        frame["_sn"] = frame["_st"].fillna(0.0) * frame["_ec"]
        agg = frame.groupby(["subject_id", "group", "network"], as_index=False).agg(
            ec=("_ec", "sum"), tc=("_tc", "sum"), sn=("_sn", "sum"), sw=("_sw", "sum"))
        agg["Burden"] = np.where(agg["tc"] > 0, agg["ec"] / agg["tc"], np.nan)
        agg["Strength"] = np.where(agg["sw"] > 0, agg["sn"] / agg["sw"], np.nan)
        long = agg.melt(id_vars=["subject_id", "group", "network"], value_vars=["Burden", "Strength"],
                        var_name="measure", value_name="value")
        long["measure"] = f"{rng}exc-" + long["measure"]
        exc_rows.append(long)
    stats_c = group_stats(pd.concat(exc_rows, ignore_index=True), "value", ["network", "measure"])
    stats_c.to_csv(OUT / "network_exception_range_stats.csv", index=False)
    print(f"wrote network_exception_range_stats.csv ({len(stats_c)} rows)", flush=True)
    print(f"exception edges by class: {edges['edr_class'].value_counts().to_dict()}", flush=True)


if __name__ == "__main__":
    main()
