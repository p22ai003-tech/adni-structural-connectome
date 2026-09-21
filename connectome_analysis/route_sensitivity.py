"""Do the published group findings survive the processing-route confound?

Background
----------
The 530 published matrices were not produced by one pipeline. Forensic
attribution (research_audit/route_attribution_20260919/) found three families:

    new           3M/10M tracks, SyN full-head atlas, mostly radius 2     366
    legacy_r4     legacy 3M ACT tracks re-assigned at radius 4            111
    legacy_step7  pre-SOTA step 7, affine 170-label atlas, unweighted      53

and the family is unevenly split by diagnosis: the new family is 65% of CN,
67% of MCI and 87% of AD. The family on its own moves the measures -- within
CN alone, legacy_r4 density is far below new -- so part of a CN-AD difference
could be a processing difference.

What this does
--------------
Every group finding is re-tested twice, with the published test unchanged:

``same_route``
    only the ``new`` family. No route difference can exist inside it, so this
    is the cleanest check; the cost is fewer subjects and so less power.
``route_adjusted``
    all subjects, with the route shift removed first: each measure is fitted
    as ``value ~ group + route`` and the route terms are subtracted, which
    takes out the route offset while leaving the group difference in place.

A finding counts as holding up when it is still significant after the same
FDR correction the original used, in the same direction.

    python -m connectome_analysis.route_sensitivity
    python -m connectome_analysis.route_sensitivity --from-attribution \\
        research_audit/route_attribution_20260919/final.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from connectome_analysis.analysis_config import get_analysis_paths  # noqa: E402
from connectome_analysis.analysis_stats import (  # noqa: E402
    bh_fdr,
    kruskal_summary,
    pairwise_robust_tests,
)

Q_THRESH = 0.05
FAMILIES = ("new", "legacy_r4", "legacy_step7")
REFERENCE = "new"


# --------------------------------------------------------------------------- routes

def family_of(route: str) -> str:
    """Map the fine-grained ``s4`` route label onto its family."""
    text = str(route)
    if text.startswith("A_"):
        return "legacy_step7"
    if text.startswith("B_"):
        return "legacy_r4"
    if text[:2] in {"C_", "D_", "E_", "F_"}:
        return "new"
    raise ValueError(f"unrecognised route label: {route!r}")


def routes_from_attribution(path: Path) -> pd.DataFrame:
    """Build the per-subject route table from the forensic attribution."""
    raw = pd.read_csv(path, usecols=["sid", "s4"])
    table = pd.DataFrame({
        "subject_id": raw["sid"].astype(str).str.replace(r"_I\d+$", "", regex=True),
        "image_id": raw["sid"].astype(str).str.extract(r"_I(\d+)$")[0],
        "route": raw["s4"],
    })
    table["route_family"] = table["route"].map(family_of)
    if table["subject_id"].duplicated().any():
        raise ValueError("attribution has more than one row for a subject")
    return table


def load_routes(path: Path) -> pd.Series:
    table = pd.read_csv(path)
    return table.set_index("subject_id")["route_family"]


# --------------------------------------------------------------------------- tests

def group_test(sub: pd.DataFrame, value_col: str = "value") -> dict:
    """The published test: Kruskal-Wallis omnibus plus CN-AD Cliff's delta.

    Same functions the network analysis calls, so an unadjusted run over all
    subjects reproduces the published numbers exactly.
    """
    sub = sub.dropna(subset=[value_col, "group"])
    out = {"n": int(len(sub)), "kw_p": np.nan, "cn_ad_delta": np.nan, "cn_ad_bm_p": np.nan}
    if sub["group"].nunique() < 2:
        return out
    out["kw_p"] = kruskal_summary(sub, value_col).pvalue
    pw = pairwise_robust_tests(sub, value_col)
    row = pw[(pw["group_a"] == "CN") & (pw["group_b"] == "AD")]
    if not row.empty:
        out["cn_ad_delta"] = float(row.iloc[0]["cliffs_delta"])
        out["cn_ad_bm_p"] = float(row.iloc[0]["bm_p"])
    return out


def remove_route_shift(sub: pd.DataFrame, value_col: str = "value") -> pd.Series:
    """Subtract the route offsets estimated alongside the group effect.

    Fitting route on its own would also absorb part of the group difference,
    because route and group are correlated. Fitting both together attributes
    each part to the right term, and only the route part is removed.
    """
    data = sub.dropna(subset=[value_col, "group", "route_family"])
    groups = [g for g in ("MCI", "AD") if (data["group"] == g).any()]
    routes = [r for r in FAMILIES if r != REFERENCE and (data["route_family"] == r).any()]
    columns = [np.ones(len(data))]
    columns += [(data["group"] == g).to_numpy(float) for g in groups]
    columns += [(data["route_family"] == r).to_numpy(float) for r in routes]
    design = np.column_stack(columns)
    y = data[value_col].to_numpy(float)
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    shift = np.zeros(len(data))
    for index, route in enumerate(routes):
        shift += beta[1 + len(groups) + index] * (data["route_family"] == route).to_numpy(float)
    return pd.Series(y - shift, index=data.index)


def three_ways(sub: pd.DataFrame) -> dict:
    """The published test, then the two checks, for one measure."""
    rec = {}
    for tag, frame in (
        ("orig", sub),
        ("same_route", sub[sub["route_family"] == REFERENCE]),
    ):
        for key, value in group_test(frame).items():
            rec[f"{tag}_{key}"] = value
    adjusted = sub.dropna(subset=["value", "group", "route_family"]).copy()
    adjusted["value"] = remove_route_shift(adjusted)
    for key, value in group_test(adjusted).items():
        rec[f"route_adjusted_{key}"] = value
    return rec


# --------------------------------------------------------------------------- network findings

def _read(path: Path) -> pd.DataFrame:
    """Read a published table exactly as it was written.

    pandas' default CSV float parser can be off by one unit in the last place.
    That is harmless for a mean and not for a rank test: the coupling measures
    are correlations with many exact ties, and a one-ULP difference breaks a
    tie. With the default parser 27 coupling p-values differed from the
    published ones by up to 7e-4; with round_trip they match to 1e-16.
    """
    return pd.read_csv(path, float_precision="round_trip")


def _family_frames(scheme_dir: Path) -> dict[str, tuple[pd.DataFrame, list[str]]]:
    """Per-subject long tables for each feature family, keyed as published."""
    micro = _read(scheme_dir / "network_microstructure_subject.csv")
    graph = _read(scheme_dir / "network_graph_subject.csv")
    coupling = _read(scheme_dir / "network_coupling_subject.csv")
    block = _read(scheme_dir / "block_within_between_subject.csv")
    block = block.assign(
        network=block["net_a"] + " | " + block["net_b"],
        metric=block["kind"],
        value=block["mean_weight"],
    )
    return {
        "microstructure": (micro, ["network", "metric"]),
        "graph": (graph, ["network", "metric"]),
        "coupling": (coupling, ["network", "metric"]),
        "connectivity": (block, ["network", "metric"]),
    }


def network_findings(analysis_root: Path, routes: pd.Series) -> pd.DataFrame:
    rows = []
    for scheme in ("functional", "anatomical"):
        scheme_dir = analysis_root / "19_network_analysis" / scheme
        if not scheme_dir.is_dir():
            continue
        for family, (frame, keys) in _family_frames(scheme_dir).items():
            frame = frame.copy()
            frame["route_family"] = frame["subject_id"].map(routes)
            for key_values, sub in frame.groupby(keys):
                rec = {"scheme": scheme, "feature_family": family,
                       **dict(zip(keys, key_values))}
                rec.update(three_ways(sub))
                rows.append(rec)
    out = pd.DataFrame(rows)
    # FDR exactly as published: within scheme x feature family, on the KW p.
    for tag in ("orig", "same_route", "route_adjusted"):
        out[f"{tag}_q"] = np.nan
        for _, index in out.groupby(["scheme", "feature_family"]).groups.items():
            out.loc[index, f"{tag}_q"] = bh_fdr(out.loc[index, f"{tag}_kw_p"].to_numpy())
    return out


def verify_against_published(results: pd.DataFrame, analysis_root: Path) -> dict:
    """The unadjusted recomputation must reproduce the published summary."""
    worst = 0.0
    compared = 0
    for scheme in ("functional", "anatomical"):
        f = analysis_root / "19_network_analysis" / scheme / "network_affectedness_summary.csv"
        if not f.is_file():
            continue
        published = _read(f)
        mine = results[results["scheme"] == scheme]
        merged = published.merge(mine, on=["feature_family", "network", "metric"], how="inner")
        compared += len(merged)
        for a, b in (("kw_p", "orig_kw_p"), ("q_kw", "orig_q"), ("cn_ad_cliffs_delta", "orig_cn_ad_delta")):
            diff = (merged[a] - merged[b]).abs().max()
            worst = max(worst, float(0.0 if pd.isna(diff) else diff))
    return {"rows_compared": compared, "max_abs_difference": worst}


def verdicts(results: pd.DataFrame) -> pd.DataFrame:
    """Keep the published findings and say whether each held up."""
    found = results[(results["orig_q"] < Q_THRESH) & results["orig_cn_ad_delta"].notna()].copy()
    for tag in ("same_route", "route_adjusted"):
        same_sign = np.sign(found[f"{tag}_cn_ad_delta"]) == np.sign(found["orig_cn_ad_delta"])
        found[f"{tag}_holds"] = (found[f"{tag}_q"] < Q_THRESH) & same_sign
        found[f"{tag}_delta_ratio"] = found[f"{tag}_cn_ad_delta"] / found["orig_cn_ad_delta"]
    return found


# --------------------------------------------------------------------------- global density

def global_density(connectomes_dir: Path, master: pd.DataFrame, routes: pd.Series) -> pd.DataFrame:
    """The headline connectivity measure, tested the same three ways."""
    records = []
    for sid, group in master[["subject_id", "group"]].itertuples(index=False):
        matches = sorted(connectomes_dir.glob(f"SC_AAL166_{sid}_I*_count.csv"))
        if not matches:
            continue
        m = np.loadtxt(matches[0], delimiter=",")
        off = m[~np.eye(m.shape[0], dtype=bool)]
        records.append({"subject_id": sid, "group": group, "value": float((off > 0).mean())})
    frame = pd.DataFrame(records)
    frame["route_family"] = frame["subject_id"].map(routes)
    rec = {"measure": "global_density", **three_ways(frame)}
    within_cn = frame[frame["group"] == "CN"]
    for family in FAMILIES:
        rec[f"CN_median_{family}"] = float(within_cn.loc[within_cn["route_family"] == family, "value"].median())
    return pd.DataFrame([rec])


# --------------------------------------------------------------------------- LR / SR

LR_SR_TABLES = (
    ("edr_exceptions", "17_edr_exceptions/edr_exception_subject_level_fd_sum_len_mean.csv",
     "17_edr_exceptions/edr_exception_summary.csv"),
    ("lr_sr", "12_length_delay/lr_sr_subject_level_fd_sum_len_mean.csv",
     "12_length_delay/lr_sr_summary.csv"),
)


def lr_sr_measures(analysis_root: Path, routes: pd.Series) -> pd.DataFrame:
    """The long-/short-range and EDR exception measures, the same three ways."""
    rows = []
    for block_name, subject_csv, summary_csv in LR_SR_TABLES:
        subject = _read(analysis_root / subject_csv)
        subject["route_family"] = subject["subject_id"].map(routes)
        published = _read(analysis_root / summary_csv).set_index("metric")
        for metric in published.index:
            sub = subject[["subject_id", "group", "route_family", metric]].rename(
                columns={metric: "value"})
            rec = {"block": block_name, "metric": metric,
                   "published_kw_p": float(published.loc[metric, "kw_p"])}
            rec.update(three_ways(sub))
            rows.append(rec)
    out = pd.DataFrame(rows)
    out["reproduces"] = (out["published_kw_p"] - out["orig_kw_p"]).abs() < 1e-9
    return out


def exception_interaction(routes: pd.Series) -> pd.DataFrame:
    """The edge-class x diagnosis interaction, exactly as the dashboard runs it.

    The outcome is each subject's log(exception strength) - log(non-exception
    strength), so the route adjustment is applied to the two log strengths and
    the contrast is recomputed from them.
    """
    from connectome_dashboard_core.lr_sr import EDR_MEASURES, edr_exception_subject_table
    from connectome_dashboard_core.settings import get_settings
    from connectome_dashboard_core.statistics import edr_lr_measure_inference

    columns = ("lr_exception_measure_median", "lr_nonexception_measure_median")
    settings = get_settings()
    rows = []
    for measure in EDR_MEASURES:
        subject, _ = edr_exception_subject_table(settings, measure)
        subject = subject.copy()
        subject["route_family"] = subject["subject_id"].astype(str).map(routes)

        adjusted = subject.copy()
        for column in columns:
            logged = np.log(pd.to_numeric(adjusted[column], errors="coerce").where(lambda v: v > 0))
            frame = pd.DataFrame({"value": logged, "group": adjusted["group"].astype(str),
                                  "route_family": adjusted["route_family"]})
            fitted = remove_route_shift(frame)
            adjusted[column] = np.nan
            adjusted.loc[fitted.index, column] = np.exp(fitted)

        variants = {
            "orig": subject,
            "same_route": subject[subject["route_family"] == REFERENCE],
            "route_adjusted": adjusted,
        }
        for name, frame in variants.items():
            _, tests, _ = edr_lr_measure_inference(frame)
            inter = tests[tests["Question"].str.contains("interaction", case=False)]
            for r in inter.itertuples():
                rows.append({
                    "measure": measure, "status": EDR_MEASURES[measure]["status"],
                    "variant": name, "contrast": r.Contrast, "n": r.N,
                    "p": float(r.p), "holm_p": float(inter.loc[r.Index, "Adjusted p"]),
                })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- the final ML model

def ml_sensitivity(routes: pd.Series) -> pd.DataFrame:
    """The published severity model, re-fitted three ways plus a route-only floor.

    Reuses build_final_model_metrics' own loader, targets, seeds, estimator and
    fold scheme, so ``published`` reproduces the reported macro AUC and the
    other rows differ from it in one thing each:

    ``route_adjusted``  route indicators join age and sex in the in-fold
                        residualisation of the connectome features
    ``same_route``      only subjects from the ``new`` family
    ``route_only``      route indicators alone -- how much of the target the
                        processing route could explain by itself
    """
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    sys.path.insert(0, str(PROJECT_ROOT / "hcp_analysis"))
    import build_final_model_metrics as fm

    feats, targets_table, cov = fm.load()
    idx, targets = fm.build_targets(feats, targets_table)
    route = routes.reindex(idx)
    if route.isna().any():
        raise ValueError(f"{int(route.isna().sum())} ML subject(s) have no route label")
    dummies = pd.DataFrame(
        {f"route_{r}": (route == r).astype(float).to_numpy()
         for r in FAMILIES if r != REFERENCE},
        index=idx,
    )

    def oof(X: pd.DataFrame, y: np.ndarray, n_covariates: int) -> np.ndarray:
        """fm.oof_proba with the number of pass-through covariates as a parameter."""
        proba = np.zeros((len(y), 3))
        for seed in fm.SEEDS:
            out = np.zeros_like(proba)
            for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
                imp = SimpleImputer(strategy="median")
                Xtr = imp.fit_transform(X.iloc[tr])
                Xte = imp.transform(X.iloc[te])
                k = n_covariates
                if Xtr.shape[1] > k:
                    A = np.column_stack([np.ones(len(tr)), Xtr[:, :k]])
                    B = np.column_stack([np.ones(len(te)), Xte[:, :k]])
                    coef, *_ = np.linalg.lstsq(A, Xtr[:, k:], rcond=None)
                    Xtr = np.column_stack([Xtr[:, :k], Xtr[:, k:] - A @ coef])
                    Xte = np.column_stack([Xte[:, :k], Xte[:, k:] - B @ coef])
                est = fm.models(seed)["extra_trees"]
                est.fit(Xtr, y[tr])
                out[te] = est.predict_proba(Xte)
            proba += out / len(fm.SEEDS)
        return proba

    base = pd.concat([cov.loc[idx], feats.loc[idx]], axis=1)
    with_route = pd.concat([cov.loc[idx], dummies, feats.loc[idx]], axis=1)
    keep = (route == REFERENCE).to_numpy()

    rows = []
    for tkey, spec in targets.items():
        y = np.asarray(spec["y"])
        variants = {
            "published": (base, y, 2),
            "route_adjusted": (with_route, y, 2 + dummies.shape[1]),
            "same_route": (base.loc[keep], y[keep], 2),
            "route_only": (dummies, y, dummies.shape[1]),
        }
        for name, (X, yy, k) in variants.items():
            p = oof(X, yy, k)
            rows.append({
                "target": tkey, "variant": name, "n": int(len(yy)),
                "macro_auc": float(roc_auc_score(yy, p, multi_class="ovr", average="macro")),
            })
            print(f"  ml {tkey:<11} {name:<15} n={len(yy):<4} macro AUC {rows[-1]['macro_auc']:.3f}",
                  flush=True)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- driver

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--routes", type=Path, default=None,
                        help="CSV of subject_id,route_family (default: <cohort>/processing_route.csv)")
    parser.add_argument("--from-attribution", type=Path, default=None,
                        help="build the routes CSV from the forensic attribution first")
    args = parser.parse_args(argv)

    import sc_config

    paths = get_analysis_paths(notebook_dir=PROJECT_ROOT, ensure=True)
    route_csv = args.routes or (sc_config.paths().cohort_dir / "processing_route.csv")
    if args.from_attribution:
        table = routes_from_attribution(args.from_attribution)
        route_csv.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(route_csv, index=False)
        print(f"routes   : wrote {route_csv} ({len(table)} subjects)")
    if not route_csv.is_file():
        print(f"no route table at {route_csv}; pass --from-attribution", file=sys.stderr)
        return 2
    routes = load_routes(route_csv)

    master = pd.read_csv(paths.master_dir / "master_cohort.csv")
    master["route_family"] = master["subject_id"].map(routes)
    missing = int(master["route_family"].isna().sum())
    if missing:
        print(f"{missing} cohort subject(s) have no route label", file=sys.stderr)
        return 1

    out_dir = paths.section_dir("22", "route_sensitivity")
    out_dir.mkdir(parents=True, exist_ok=True)

    crosstab = pd.crosstab(master["route_family"], master["group"])[["CN", "MCI", "AD"]]
    crosstab.to_csv(out_dir / "route_by_group.csv")
    print("route x group\n" + crosstab.to_string() + "\n")

    density = global_density(paths.connectomes_dir, master, routes)
    density.to_csv(out_dir / "global_density.csv", index=False)

    results = network_findings(paths.output_root, routes)
    results.to_csv(out_dir / "network_all_measures.csv", index=False)
    check = verify_against_published(results, paths.output_root)
    found = verdicts(results)
    found.to_csv(out_dir / "network_findings_verdicts.csv", index=False)

    summary = {
        "subjects": int(len(master)),
        "route_by_group": crosstab.to_dict(),
        "reproduces_published": check,
        "network_findings": int(len(found)),
        "same_route_hold": int(found["same_route_holds"].sum()),
        "route_adjusted_hold": int(found["route_adjusted_holds"].sum()),
        "same_route_median_delta_ratio": float(found["same_route_delta_ratio"].median()),
        "route_adjusted_median_delta_ratio": float(found["route_adjusted_delta_ratio"].median()),
        "by_family": {
            family: {
                "findings": int(len(g)),
                "same_route_hold": int(g["same_route_holds"].sum()),
                "route_adjusted_hold": int(g["route_adjusted_holds"].sum()),
            }
            for family, g in found.groupby("feature_family")
        },
        "global_density": density.iloc[0].to_dict(),
    }
    lr = lr_sr_measures(paths.output_root, routes)
    lr.to_csv(out_dir / "lr_sr_measures.csv", index=False)
    published_sig = lr[lr["published_kw_p"] < Q_THRESH]
    summary["lr_sr"] = {
        "reproduces_published": bool(lr["reproduces"].all()),
        "measures_significant_as_published": int(len(published_sig)),
        "same_route_still_p_lt_05": int((published_sig["same_route_kw_p"] < Q_THRESH).sum()),
        "route_adjusted_still_p_lt_05": int((published_sig["route_adjusted_kw_p"] < Q_THRESH).sum()),
    }
    print("lr/sr measures:", summary["lr_sr"], flush=True)

    inter = exception_interaction(routes)
    inter.to_csv(out_dir / "lr_exception_interaction.csv", index=False)
    summary["lr_exception_interaction_cn_ad_holm"] = {
        f"{r.measure}:{r.variant}": float(f"{r.holm_p:.3g}")
        for r in inter[inter["contrast"] == "CN vs AD"].itertuples()
    }
    print("interaction CN vs AD (Holm):", summary["lr_exception_interaction_cn_ad_holm"], flush=True)

    # The within-family mix: a same-route check is only as clean as the family
    # is homogeneous, and this family is not -- most of its AD subjects came
    # through AD-specific recovery lanes. Written out so the caveat has numbers.
    within = master[master["route_family"] == REFERENCE].merge(
        pd.read_csv(route_csv)[["subject_id", "route"]], on="subject_id")
    pd.crosstab(within["route"], within["group"])[["CN", "MCI", "AD"]].to_csv(
        out_dir / "new_family_lanes_by_group.csv")

    ml = ml_sensitivity(routes)
    ml.to_csv(out_dir / "ml_model.csv", index=False)
    summary["ml_macro_auc"] = {
        f"{r.target}:{r.variant}": round(r.macro_auc, 4) for r in ml.itertuples()
    }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    print(json.dumps(summary, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
