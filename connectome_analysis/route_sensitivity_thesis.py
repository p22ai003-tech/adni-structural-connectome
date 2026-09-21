"""Route sensitivity for the thesis's range-split and exception tables.

route_sensitivity.py covers the whole-connectome findings, LR/SR summary
measures, the exception interaction and the severity model. The thesis also
rests on three things that are computed separately and so needed their own
check:

* Tables B and C: FA/MD/RD/AxD, degree and strength per functional network,
  recomputed on short- or long-range edges only (B) and on EDR exception edges
  only (C), tested for each pairwise contrast with BH across the ten networks
  within each measure.
* The "disease signal moves from long-range to short-range" result: for the
  forty network x diffusion-measure combinations, whether the group effect is
  larger on short- or on long-range edges.
* The exception-architecture measures (consensus-core share and the rest),
  tested per contrast with BH across the sixteen measures.

Each is recomputed exactly as published, then on the ``new`` route family
alone, then on everyone with the route offset removed -- the same two checks
as route_sensitivity.py, from which the route handling is imported.

    python -m connectome_analysis.route_sensitivity_thesis
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for extra in (PROJECT_ROOT, PROJECT_ROOT / "hcp_analysis"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from connectome_analysis.analysis_config import get_analysis_paths  # noqa: E402
from connectome_analysis.route_sensitivity import (  # noqa: E402
    REFERENCE, _read, load_routes, remove_route_shift,
)

CONTRASTS = (("cn_mci", "CN", "MCI"), ("cn_ad", "CN", "AD"), ("mci_ad", "MCI", "AD"))


# Copied from hcp_analysis/build_range_restricted_networks.py, which reads
# sys.argv at import time and so cannot be imported from another entry point.
# Identical arithmetic; the reproduction check against the published tables
# (max q difference 0) is what confirms it.
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
DIFFUSION = ("AxD", "FA", "MD", "RD")
Q = 0.05
VARIANTS = ("orig", "same_route", "route_adjusted")


def pairwise(frame: pd.DataFrame, keys: list[str], correct_within: str | None) -> pd.DataFrame:
    """Brunner-Munzel and Cliff's delta per key and contrast, as the builders do."""
    rows = []
    for key, sub in frame.groupby(keys):
        arrays = {g: sub.loc[sub["group"] == g, "value"].dropna().to_numpy(float)
                  for g in ("CN", "MCI", "AD")}
        if min(map(len, arrays.values())) < 5:
            continue
        rec = dict(zip(keys, key if isinstance(key, tuple) else (key,)))
        for ck, first, second in CONTRASTS:
            try:
                p = float(sps.brunnermunzel(arrays[first], arrays[second],
                                            alternative="two-sided").pvalue)
            except ValueError:
                p = float("nan")
            rec[f"{ck}_delta"] = cliffs_delta(arrays[first], arrays[second])
            rec[f"{ck}_p"] = p
        rows.append(rec)
    out = pd.DataFrame(rows)
    for ck, _, _ in CONTRASTS:
        if correct_within is None:
            out[f"{ck}_q"] = bh_fdr(out[f"{ck}_p"].tolist())
        else:
            out[f"{ck}_q"] = np.nan
            for _, index in out.groupby(correct_within).groups.items():
                out.loc[index, f"{ck}_q"] = bh_fdr(out.loc[index, f"{ck}_p"].tolist())
    return out


def variants(frame: pd.DataFrame, keys: list[str]) -> dict[str, pd.DataFrame]:
    """The published frame, the same-route subset, and the route-adjusted frame."""
    adjusted = []
    for _, sub in frame.groupby(keys):
        sub = sub.dropna(subset=["value", "group", "route_family"]).copy()
        if sub.empty:
            continue
        sub["value"] = remove_route_shift(sub)
        adjusted.append(sub)
    return {
        "orig": frame,
        "same_route": frame[frame["route_family"] == REFERENCE],
        "route_adjusted": pd.concat(adjusted, ignore_index=True),
    }


def run_three(frame, keys, correct_within) -> pd.DataFrame:
    parts = []
    for name, data in variants(frame, keys).items():
        table = pairwise(data, keys, correct_within)
        table.insert(0, "variant", name)
        parts.append(table)
    return pd.concat(parts, ignore_index=True)


def holds(results: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """For each published significant cell, whether each check keeps it."""
    orig = results[results["variant"] == "orig"].set_index(keys)
    rows = []
    for ck, _, _ in CONTRASTS:
        sig = orig[orig[f"{ck}_q"] < Q]
        for name in ("same_route", "route_adjusted"):
            other = results[results["variant"] == name].set_index(keys).reindex(sig.index)
            same_sign = np.sign(other[f"{ck}_delta"]) == np.sign(sig[f"{ck}_delta"])
            kept = (other[f"{ck}_q"] < Q) & same_sign
            rows.append({"contrast": ck, "check": name, "published_significant": int(len(sig)),
                         "held": int(kept.sum()), "sign_flipped": int((~same_sign).sum())})
    return pd.DataFrame(rows)


def reversal(results: pd.DataFrame, short: str, long: str) -> pd.DataFrame:
    """Is the group effect larger on short- or on long-range edges?

    Over the forty network x diffusion-measure combinations, per contrast:
    how many have the larger absolute effect on short-range edges, and the
    mean absolute effect in each range.
    """
    rows = []
    for name in VARIANTS:
        data = results[results["variant"] == name].copy()
        data["base"] = data["measure"].str.split("-", n=1).str[1]
        data["range"] = data["measure"].str.split("-", n=1).str[0]
        data = data[data["base"].isin(DIFFUSION)]
        for ck, _, _ in CONTRASTS:
            wide = data.pivot_table(index=["network", "base"], columns="range",
                                    values=f"{ck}_delta").dropna()
            s, l = wide[short].abs(), wide[long].abs()
            rows.append({
                "variant": name, "contrast": ck, "combinations": int(len(wide)),
                "short_range_larger": int((s > l).sum()),
                "mean_abs_delta_short": round(float(s.mean()), 4),
                "mean_abs_delta_long": round(float(l.mean()), 4),
                "wilcoxon_p": float(sps.wilcoxon(s, l).pvalue) if len(wide) > 5 else float("nan"),
            })
    return pd.DataFrame(rows)


def named(results: pd.DataFrame, cells: list[tuple[str, str, str]]) -> pd.DataFrame:
    """The specific cells the thesis text names."""
    rows = []
    for network, measure, ck in cells:
        for name in VARIANTS:
            r = results[(results["variant"] == name) & (results["network"] == network)
                        & (results["measure"] == measure)]
            if r.empty:
                continue
            r = r.iloc[0]
            rows.append({"network": network, "measure": measure, "contrast": ck, "variant": name,
                         "delta": round(float(r[f"{ck}_delta"]), 3), "q": float(r[f"{ck}_q"])})
    return pd.DataFrame(rows)


def table_a(net_dir: Path, routes: pd.Series) -> pd.DataFrame:
    """Whole-connectome microstructure per network (Table A).

    Unlike Tables B and C, its published q-values correct across the three
    pairwise contrasts within each network and measure -- that is what
    pairwise_robust_tests does -- so it is recomputed with that function
    rather than with pairwise() above.
    """
    from connectome_analysis.analysis_stats import pairwise_robust_tests

    frame = _read(net_dir / "network_microstructure_subject.csv").rename(columns={"metric": "measure"})
    frame["route_family"] = frame["subject_id"].map(routes)
    rows = []
    for name, data in variants(frame, ["network", "measure"]).items():
        for (net, meas), sub in data.groupby(["network", "measure"]):
            for r in pairwise_robust_tests(sub.dropna(subset=["value"]), "value").itertuples():
                rows.append({"variant": name, "network": net, "measure": meas,
                             "contrast": f"{r.group_a.lower()}_{r.group_b.lower()}",
                             "delta": r.cliffs_delta, "q": r.bm_q})
    return pd.DataFrame(rows)


def main() -> int:
    import sc_config

    paths = get_analysis_paths(notebook_dir=PROJECT_ROOT, ensure=True)
    routes = load_routes(sc_config.paths().cohort_dir / "processing_route.csv")
    root = paths.output_root
    out = paths.section_dir("22", "route_sensitivity")
    out.mkdir(parents=True, exist_ok=True)
    summary: dict = {}

    net_dir = root / "19_network_analysis" / "functional"
    for label, subject_csv, stats_csv, short, long in (
        ("table_B_range", "network_range_restricted_subject.csv",
         "network_range_restricted_stats.csv", "SR", "LR"),
        ("table_C_exceptions", "network_exception_measures_subject.csv",
         "network_exception_measures_stats.csv", "SRexc", "LRexc"),
    ):
        frame = _read(net_dir / subject_csv)
        frame["route_family"] = frame["subject_id"].map(routes)
        results = run_three(frame, ["network", "measure"], "measure")
        results.to_csv(out / f"{label}_three_ways.csv", index=False)

        published = _read(net_dir / stats_csv)
        mine = results[results["variant"] == "orig"].merge(published, on=["network", "measure"])
        worst = max(float((mine[f"{ck}_q"] - mine[f"{ck}_bm_q"]).abs().max()) for ck, _, _ in CONTRASTS)

        kept = holds(results, ["network", "measure"])
        rev = reversal(results, short, long)
        kept.to_csv(out / f"{label}_holds.csv", index=False)
        rev.to_csv(out / f"{label}_reversal.csv", index=False)
        summary[label] = {"reproduces_published_max_q_diff": worst,
                          "holds": kept.to_dict("records"), "reversal": rev.to_dict("records")}
        print(f"\n== {label} (max q difference from published: {worst:.2g})")
        print(kept.to_string(index=False))
        print(rev.to_string(index=False))

        if label == "table_C_exceptions":
            cells = named(results, [
                ("Limbic", "LRexc-MD", "cn_mci"), ("Limbic", "LRexc-RD", "cn_mci"),
                ("Frontoparietal", "SRexc-AxD", "mci_ad"), ("Frontoparietal", "SRexc-MD", "mci_ad"),
                ("Frontoparietal", "SRexc-RD", "mci_ad"),
            ])
            cells.to_csv(out / "thesis_named_exception_cells.csv", index=False)
            print(cells.to_string(index=False))

    a = table_a(net_dir, routes)
    a.to_csv(out / "table_A_microstructure_three_ways.csv", index=False)
    counts = (a.assign(sig=a["q"] < Q).groupby(["measure", "contrast", "variant"])["sig"].sum()
              .unstack("variant")[list(VARIANTS)].astype(int))
    counts.to_csv(out / "table_A_networks_separating.csv")
    summary["table_A_networks_separating"] = counts.reset_index().to_dict("records")
    print("\n== table A: networks separating the groups (q < 0.05)")
    print(counts.to_string())

    arch = _read(root / "20_exception_specificity" / "exception_architecture_subject.csv")
    features = [c for c in arch.columns if c not in ("subject_id", "group", "phase")]
    long_arch = arch.melt(id_vars=["subject_id", "group"], value_vars=features,
                          var_name="measure", value_name="value")
    long_arch["route_family"] = long_arch["subject_id"].map(routes)
    results = run_three(long_arch, ["measure"], None)
    results.to_csv(out / "architecture_three_ways.csv", index=False)
    published = _read(root / "20_exception_specificity" / "exception_architecture_stats.csv")
    mine = results[results["variant"] == "orig"].merge(published, left_on="measure", right_on="feature")
    worst = max(float((mine[f"{ck}_q"] - mine[f"{ck}_bm_q"]).abs().max()) for ck, _, _ in CONTRASTS)
    kept = holds(results, ["measure"])
    kept.to_csv(out / "architecture_holds.csv", index=False)
    wide = results.pivot_table(index="measure", columns="variant", values=["cn_ad_delta", "cn_ad_q"])
    wide.to_csv(out / "architecture_cn_ad.csv")
    summary["architecture"] = {"reproduces_published_max_q_diff": worst, "holds": kept.to_dict("records")}
    print(f"\n== exception architecture (max q difference from published: {worst:.2g})")
    print(kept.to_string(index=False))
    print(wide.round(4).to_string())

    (out / "thesis_tables_summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    return 0


if __name__ == "__main__" and "--core" not in sys.argv:
    raise SystemExit(main())


def core_and_pairs(routes: pd.Series) -> dict:
    """Core-edge losses (Welch t on 0/1 presence) and network-pair shares,
    reusing build_consensus_core's own loader and recipes."""
    from scipy import stats as st

    import build_consensus_core as bcc

    edges, names, nets = bcc.load()
    core = bcc.consensus_core(edges, names, nets)
    subj = edges[["subject_id", "group"]].drop_duplicates().set_index("subject_id")["group"]
    route = subj.index.to_series().map(routes)

    # core edges: presence indicator per subject, CN vs AD
    loss_rows = []
    for _, c in core.iterrows():
        present = set(edges[(edges["i"] == c["i"]) & (edges["j"] == c["j"])]["subject_id"])
        frame = pd.DataFrame({"group": subj, "route_family": route,
                              "value": subj.index.to_series().isin(present).astype(float)})
        for name, data in variants(frame.assign(key=1), ["key"]).items():
            by = {g: data.loc[data["group"] == g, "value"].to_numpy() for g in ("CN", "AD")}
            loss_rows.append({"edge": c["edge"], "variant": name,
                              "drop_pp": 100 * (by["CN"].mean() - by["AD"].mean()),
                              "p": float(st.ttest_ind(by["CN"], by["AD"], equal_var=False).pvalue)})
    loss = pd.DataFrame(loss_rows)
    loss["q"] = np.nan
    for name, index in loss.groupby("variant").groups.items():
        loss.loc[index, "q"] = bh_fdr(loss.loc[index, "p"].tolist())
    published = set(loss[(loss.variant == "orig") & (loss.q < Q)]["edge"])
    kept = {}
    for name in ("same_route", "route_adjusted"):
        x = loss[(loss.variant == name) & loss.edge.isin(published)]
        kept[name] = int(((x.q < Q) & (x.drop_pp > 0)).sum())

    # network-pair shares: the one pair that survived correction
    e = edges.copy()
    e["netpair"] = [" – ".join(sorted([nets.get(a, "?"), nets.get(b, "?")])) for a, b in zip(e["i"], e["j"])]
    share = (e.groupby(["subject_id", "netpair"]).size() / e.groupby("subject_id").size()).unstack(fill_value=0.0)
    share = share.reindex(subj.index).fillna(0.0)
    cereb = pd.DataFrame({"group": subj, "route_family": route,
                          "value": share.get("Cerebellar – Cerebellar", 0.0)})
    pair = pairwise(cereb.assign(measure="within_cerebellar"), ["measure"], None)
    pair_rows = []
    for name, data in variants(cereb.assign(measure="within_cerebellar"), ["measure"]).items():
        r = pairwise(data, ["measure"], None).iloc[0]
        pair_rows.append({"variant": name, "cn_ad_delta": round(float(r["cn_ad_delta"]), 3),
                          "cn_ad_p": float(r["cn_ad_p"])})
    return {"core_edges_lost_in_AD": {"published": len(published), **kept},
            "within_cerebellar_share": pair_rows,
            "loss_table": loss}


if __name__ == "__main__" and "--core" in sys.argv:
    import sc_config

    paths = get_analysis_paths(notebook_dir=PROJECT_ROOT, ensure=True)
    routes = load_routes(sc_config.paths().cohort_dir / "processing_route.csv")
    result = core_and_pairs(routes)
    out = paths.section_dir("22", "route_sensitivity")
    result.pop("loss_table").to_csv(out / "core_edge_loss_three_ways.csv", index=False)
    (out / "core_and_pairs_summary.json").write_text(json.dumps(result, indent=2, default=float) + "\n")
    print(json.dumps(result, indent=2, default=float))
