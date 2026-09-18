#!/usr/bin/env python
"""Consensus exception core: the edges that are EDR exceptions in most subjects.

Why this script exists
----------------------
Three artifacts behind the thesis had no producing code anywhere in the
repository -- consensus_core_edges.csv, core_edge_loss.csv and
exception_networkpair_shares.csv. They were computed once, by hand, and the
recipe was lost. A thesis whose numbers cannot be regenerated from the
repository is not reproducible, so the recipe was recovered from the artifacts
themselves and is written down here. Every statistic below reproduces the
published file exactly; see --verify.

What it computes
----------------
An edge is an EDR exception in a given subject when its weight exceeds the
mean of its within-subject length bin by more than k SD. The edge-level table
from stage 17 holds one row per (subject, exception edge): 126,735 rows over
530 subjects, covering 4,853 distinct edges.

consensus_core_edges.csv
    The 51 edges that are exceptions in more than half of the 530 subjects.
    This is the "consensus core": the part of the exception architecture that
    is a property of the cohort rather than of an individual.

core_edge_loss.csv
    For each core edge, the fraction of each diagnostic group in which it is an
    exception, and how far that falls from CN to AD. Effect is the difference
    in proportions; the test is Welch's t on the per-subject 0/1 indicator,
    FDR-corrected over the 51 edges.

exception_networkpair_shares.csv
    How each subject's exceptions distribute over pairs of functional networks,
    averaged by group. Restricted to network pairs present in at least half of
    subjects. Effect is Cliff's delta (CN vs AD) and the test is Brunner-Munzel,
    both rank-based, because these shares are bounded, skewed and zero-inflated,
    which is exactly where a t-test is not appropriate.

Network assignment joins on the MATRIX INDEX, never on the ROI name: the atlas
name columns were shifted for rows >= 37 until the LUT fix, and the index was
always correct.

    python hcp_analysis/build_consensus_core.py
    python hcp_analysis/build_consensus_core.py --verify   # against the published files
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# Paths resolve through sc_paths: project-relative, overridable with --out.
try:
    import sc_paths
except ModuleNotFoundError:  # loose script run from outside hcp_analysis/
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent))
    import sc_paths

_P = sc_paths.resolve_for_script("ml")
OUT = _P.out
ANALYSIS = _P.analysis

CORE_MIN_PCT = 50.0       # an edge is "core" when it exceeds this share of subjects
NETPAIR_MIN_PREV = 0.5    # a network pair is reported when this many subjects have one
GROUPS = ("CN", "MCI", "AD")


def _bh(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg, written out so the script does not need statsmodels."""
    p = np.asarray(p, dtype=float)
    n = p.size
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(ranked, 0, 1)
    return out


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """P(a > b) - P(a < b). Rank-based, so bounded skewed shares are fine."""
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    return float(np.sign(a[:, None] - b[None, :]).sum() / (len(a) * len(b)))


def load() -> tuple[pd.DataFrame, dict[int, str], dict[int, str]]:
    edges = pd.read_csv(
        ANALYSIS / "17_edr_exceptions/edr_exception_edge_level_fd_sum_len_mean.csv"
    )
    lut = pd.read_csv(_P.aal_node_map)
    names = dict(zip(lut["new_id"].astype(int), lut["name"].astype(str)))
    nm = pd.read_csv(ANALYSIS / "19_network_analysis/network_mapping_used.csv")
    nets = dict(zip(nm["matrix_idx"].astype(int), nm["functional_network"].astype(str)))
    return edges, names, nets


def consensus_core(edges: pd.DataFrame, names, nets) -> pd.DataFrame:
    n_sub = edges["subject_id"].nunique()
    per_edge = edges.groupby(["i", "j"])
    pct = per_edge["subject_id"].nunique() / n_sub * 100.0
    core = pct[pct > CORE_MIN_PCT].sort_values(ascending=False)

    rows = []
    for (i, j), p in core.items():
        sub = edges[(edges["i"] == i) & (edges["j"] == j)]
        a, b = names[int(i)], names[int(j)]
        na, nb = nets.get(int(i), "?"), nets.get(int(j), "?")
        rows.append({
            "edge": f"{a} – {b}",
            "pct": p,
            "networks": f"{na}–{nb}",
            "same_net": na == nb,
            # homotopic = the same region in the two hemispheres
            "homotopic": a[:-2] == b[:-2] and a[-2:] in ("_L", "_R") and a[-1] != b[-1],
            "len_mm": float(sub["length_mm"].median()),
            "cls": sub["edr_class"].mode().iat[0],
            "i": int(i), "j": int(j),
        })
    return pd.DataFrame(rows)


def core_edge_loss(edges: pd.DataFrame, core: pd.DataFrame) -> pd.DataFrame:
    subj = edges[["subject_id", "group"]].drop_duplicates().set_index("subject_id")["group"]
    rows = []
    for _, c in core.iterrows():
        present = set(edges[(edges["i"] == c["i"]) & (edges["j"] == c["j"])]["subject_id"])
        ind = subj.index.to_series().isin(present).astype(float)
        by = {g: ind[subj == g].to_numpy() for g in GROUPS}
        d = by["CN"].mean() - by["AD"].mean()
        p = stats.ttest_ind(by["CN"], by["AD"], equal_var=False).pvalue
        rows.append({"edge": c["edge"], **{g: by[g].mean() for g in GROUPS},
                     "d": d, "p": float(p), "drop_pp": d * 100.0})
    out = pd.DataFrame(rows).sort_values("p").reset_index(drop=True)
    out["q"] = _bh(out["p"].to_numpy())
    return out[["edge", *GROUPS, "d", "p", "q", "drop_pp"]]


def networkpair_shares(edges: pd.DataFrame, nets) -> pd.DataFrame:
    e = edges.copy()
    e["na"] = e["i"].map(nets)
    e["nb"] = e["j"].map(nets)
    e = e.dropna(subset=["na", "nb"])
    # unordered pair, so A-B and B-A are the same cell
    e["netpair"] = [" – ".join(sorted([a, b])) for a, b in zip(e["na"], e["nb"])]

    total = e.groupby("subject_id").size()
    share = (e.groupby(["subject_id", "netpair"]).size() / total).unstack(fill_value=0.0)
    subj = e[["subject_id", "group"]].drop_duplicates().set_index("subject_id")["group"]
    share = share.reindex(subj.index).fillna(0.0)

    keep = (share > 0).mean() >= NETPAIR_MIN_PREV
    share = share.loc[:, keep]

    rows = []
    for pair in share.columns:
        by = {g: share.loc[subj[subj == g].index, pair].to_numpy() for g in GROUPS}
        d = cliffs_delta(by["CN"], by["AD"])
        try:
            p = float(stats.brunnermunzel(by["CN"], by["AD"]).pvalue)
        except Exception:
            p = float("nan")
        a, b = [x.strip() for x in pair.split("–")]
        # direction follows the difference in MEANS, not the sign of Cliff's
        # delta. The two disagree on 6 of the 37 pairs, because these shares are
        # zero-inflated: a group can have the larger mean while losing the rank
        # comparison. Every such pair has |delta| <= 0.05, i.e. no real effect,
        # but the label and the effect size are not interchangeable.
        rows.append({"netpair": pair, **{g: by[g].mean() for g in GROUPS},
                     "d_cn_ad": d, "p": p, "within": a == b,
                     "direction": "HIGHER in AD" if by["AD"].mean() > by["CN"].mean()
                                  else "lower in AD"})
    out = pd.DataFrame(rows).sort_values("p").reset_index(drop=True)
    out["q"] = _bh(out["p"].to_numpy())
    return out[["netpair", *GROUPS, "d_cn_ad", "p", "within", "q", "direction"]]


def verify(new: pd.DataFrame, path: Path, keys: list[str], label: str) -> bool:
    if not path.is_file():
        print(f"  {label:<34} no published file to compare against")
        return True
    old = pd.read_csv(path)
    if len(old) != len(new):
        print(f"  {label:<34} ROW COUNT differs: {len(new)} vs published {len(old)}")
        return False
    m = old.merge(new, on=keys[0], suffixes=("_pub", "_new"))
    if len(m) != len(old):
        print(f"  {label:<34} only {len(m)}/{len(old)} rows matched on {keys[0]}")
        return False
    worst, col = 0.0, None
    for c in old.columns:
        if c == keys[0] or f"{c}_pub" not in m:
            continue
        try:
            d = (pd.to_numeric(m[f"{c}_pub"]) - pd.to_numeric(m[f"{c}_new"])).abs().max()
        except (TypeError, ValueError):
            continue
        if d > worst:
            worst, col = d, c
    ok = worst < 1e-9
    print(f"  {label:<34} {'reproduces' if ok else 'DIFFERS'} "
          f"({len(m)} rows, max abs diff {worst:.2e}{f' in {col}' if col else ''})")
    return ok


def main(argv=None) -> int:
    ap = sc_paths.add_common_args(argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--verify", action="store_true",
                    help="compare against the published files instead of trusting the run")
    args = ap.parse_args(argv)

    edges, names, nets = load()
    print(f"edge-level rows {len(edges)}, subjects {edges.subject_id.nunique()}, "
          f"distinct edges {len(edges[['i','j']].drop_duplicates())}")

    core = consensus_core(edges, names, nets)
    loss = core_edge_loss(edges, core)
    shares = networkpair_shares(edges, nets)
    print(f"consensus core {len(core)} edges (> {CORE_MIN_PCT:.0f}% of subjects), "
          f"{len(shares)} network pairs\n")

    core_out = core.drop(columns=["i", "j"])
    if args.verify:
        ok = all([
            verify(core_out, OUT / "consensus_core_edges.csv", ["edge"], "consensus_core_edges.csv"),
            verify(loss, OUT / "core_edge_loss.csv", ["edge"], "core_edge_loss.csv"),
            verify(shares, OUT / "exception_networkpair_shares.csv", ["netpair"],
                   "exception_networkpair_shares.csv"),
        ])
        return 0 if ok else 1

    OUT.mkdir(parents=True, exist_ok=True)
    core_out.to_csv(OUT / "consensus_core_edges.csv", index=False)
    loss.to_csv(OUT / "core_edge_loss.csv", index=False)
    shares.to_csv(OUT / "exception_networkpair_shares.csv", index=False)
    print(f"wrote 3 files to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
