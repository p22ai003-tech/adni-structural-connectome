#!/usr/bin/env python3
"""Falsify support-induced frontal topology--FA coupling interpretations.

The historical coupling statistic is one within-subject Spearman correlation
across 32 anatomical frontal AAL3 nodes.  This script reproduces that statistic
from existing participant-level node tables, then recomputes partial Spearman
correlations after rank-residualising node topology and node FA for node degree
and FA-edge support.  It reads no image arrays and changes no production output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from submission_sprint_analysis import (
    GROUPS,
    Stratum,
    adjust_p,
    build_strata,
    fit_feature_contrasts,
    load_exact_cohort,
    markdown_table,
    sha256,
)


PROJECT = Path("/home/ec2-user/exp")
ANALYSIS = Path("/data/derivatives/qc/analysis_cohort")
DEFAULT_OUT = (
    PROJECT / "research_audit" / "outputs" / "coupling_support_falsification_v1"
)
INPUTS = {
    "exact_manifest": PROJECT
    / "research_audit"
    / "outputs"
    / "available_data_pair_manifest_v2.csv",
    "mapping": PROJECT / "atlas" / "AAL" / "AAL3_network_mapping.csv",
    "node_micro": ANALYSIS
    / "04_node_microstructure_live"
    / "live_node_microstructure_long.csv",
    "node_graph": ANALYSIS
    / "07_node_graph_live"
    / "live_node_graph_long.csv",
    "published_network_coupling": ANALYSIS
    / "19_network_analysis"
    / "anatomical"
    / "network_coupling_subject.csv",
}

VARIANTS = (
    "raw_nodal_eff_fa",
    "partial_degree_nodal_eff_fa",
    "partial_degree_fa_edges_nodal_eff_fa",
    "raw_strength_fa",
    "partial_degree_strength_fa",
    "partial_degree_fa_edges_strength_fa",
)


def _finite_frame(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    use = frame.loc[:, list(columns)].replace([np.inf, -np.inf], np.nan).dropna()
    return use


def partial_spearman(
    frame: pd.DataFrame,
    left: str,
    right: str,
    controls: Sequence[str],
    minimum_nodes: int = 8,
) -> float:
    """Partial Spearman rho via least-squares residuals of rank variables."""

    columns = [left, right, *controls]
    use = _finite_frame(frame, columns)
    if len(use) < minimum_nodes:
        return math.nan
    for column in columns:
        if use[column].nunique() < 2:
            return math.nan
    ranks = np.column_stack([rankdata(use[column]) for column in columns])
    design = np.column_stack([np.ones(len(use)), ranks[:, 2:]])
    left_residual = ranks[:, 0] - design @ np.linalg.lstsq(
        design, ranks[:, 0], rcond=None
    )[0]
    right_residual = ranks[:, 1] - design @ np.linalg.lstsq(
        design, ranks[:, 1], rcond=None
    )[0]
    if np.std(left_residual) == 0 or np.std(right_residual) == 0:
        return math.nan
    return float(np.corrcoef(left_residual, right_residual)[0, 1])


def raw_spearman(
    frame: pd.DataFrame, left: str, right: str, minimum_nodes: int = 5
) -> float:
    use = _finite_frame(frame, [left, right])
    if len(use) < minimum_nodes or use[left].nunique() < 2 or use[right].nunique() < 2:
        return math.nan
    return float(spearmanr(use[left], use[right]).statistic)


def build_subject_variants(
    mapping_path: Path,
    graph_path: Path,
    micro_path: Path,
) -> pd.DataFrame:
    mapping = pd.read_csv(mapping_path)
    frontal_nodes = set(
        pd.to_numeric(
            mapping.loc[mapping["anatomical_system"].eq("Frontal"), "matrix_idx"],
            errors="raise",
        ).astype(int)
    )
    if len(frontal_nodes) != 32:
        raise ValueError(f"Expected 32 frontal nodes, found {len(frontal_nodes)}")

    graph = pd.read_csv(graph_path)
    micro = pd.read_csv(micro_path)
    graph = graph[graph["node"].isin(frontal_nodes)][
        ["subject_id", "node", "strength", "degree", "nodal_eff"]
    ]
    micro = micro[
        micro["node"].isin(frontal_nodes) & micro["metric"].eq("fa_mean")
    ][["subject_id", "node", "value", "n_edges"]].rename(
        columns={"value": "fa", "n_edges": "fa_edges"}
    )
    nodes = graph.merge(micro, on=["subject_id", "node"], how="inner", validate="1:1")
    rows = []
    for subject_id, frame in nodes.groupby("subject_id", sort=True):
        rows.append(
            {
                "subject_id": subject_id,
                "n_frontal_rows": len(frame),
                "raw_nodal_eff_fa": raw_spearman(frame, "nodal_eff", "fa"),
                "partial_degree_nodal_eff_fa": partial_spearman(
                    frame, "nodal_eff", "fa", ["degree"]
                ),
                "partial_degree_fa_edges_nodal_eff_fa": partial_spearman(
                    frame, "nodal_eff", "fa", ["degree", "fa_edges"]
                ),
                "raw_strength_fa": raw_spearman(frame, "strength", "fa"),
                "partial_degree_strength_fa": partial_spearman(
                    frame, "strength", "fa", ["degree"]
                ),
                "partial_degree_fa_edges_strength_fa": partial_spearman(
                    frame, "strength", "fa", ["degree", "fa_edges"]
                ),
            }
        )
    subjects = pd.DataFrame(rows)
    # Historical node-microstructure products do not represent every locked
    # source pair.  Preserve that attrition rather than fabricating rows.
    if subjects["subject_id"].nunique() != 529:
        raise ValueError(
            "Expected the current historical node-table intersection to contain 529 subjects"
        )
    return subjects


def validate_raw_reproduction(
    subjects: pd.DataFrame, published_path: Path
) -> dict[str, object]:
    published = pd.read_csv(published_path)
    published = published[
        published["network"].eq("Frontal")
        & published["metric"].eq("nodal_eff__fa_mean")
    ][["subject_id", "value"]].rename(columns={"value": "published_raw"})
    merged = subjects.merge(
        published, on="subject_id", how="outer", validate="1:1", indicator=True
    )
    matched = merged["_merge"].eq("both")
    both_finite = matched & merged[["raw_nodal_eff_fa", "published_raw"]].notna().all(axis=1)
    nan_match = merged.loc[matched, "raw_nodal_eff_fa"].isna().eq(
        merged.loc[matched, "published_raw"].isna()
    )
    max_difference = float(
        np.max(
            np.abs(
                merged.loc[both_finite, "raw_nodal_eff_fa"]
                - merged.loc[both_finite, "published_raw"]
            )
        )
    )
    return {
        "current_node_table_subjects": int(subjects["subject_id"].nunique()),
        "published_coupling_subjects": int(published["subject_id"].nunique()),
        "matched_subjects": int(matched.sum()),
        "current_node_subjects_absent_from_published_coupling": int(
            merged["_merge"].eq("left_only").sum()
        ),
        "published_subjects_absent_from_current_nodes": int(
            merged["_merge"].eq("right_only").sum()
        ),
        "finite_pairs": int(both_finite.sum()),
        "nan_pattern_matches": bool(nan_match.all()),
        "max_absolute_difference": max_difference,
        "pass": bool(
            subjects["subject_id"].nunique() == 529
            and published["subject_id"].nunique() == 514
            and matched.sum() == 514
            and merged["_merge"].eq("left_only").sum() == 15
            and merged["_merge"].eq("right_only").sum() == 0
            and nan_match.all()
            and max_difference <= 1e-12
        ),
    }


def model_variants(
    subjects: pd.DataFrame,
    cohort: pd.DataFrame,
    strata: Sequence[Stratum],
) -> pd.DataFrame:
    metadata = cohort[
        [
            "subject_id",
            "group",
            "age",
            "sex",
            "site",
            "log_gap_days",
            "t1_source",
            "protocol_bucket",
        ]
    ]
    rows: list[dict[str, object]] = []
    for variant in VARIANTS:
        frame = subjects[["subject_id", variant]].rename(columns={variant: "value"})
        frame = frame.merge(metadata, on="subject_id", how="left", validate="1:1")
        frame["feature_id"] = variant
        frame["mapping"] = "anatomical"
        frame["feature_family"] = "frontal_topology_fa_support_falsification"
        frame["network"] = "Frontal"
        frame["metric"] = variant
        for stratum in strata:
            rows.extend(fit_feature_contrasts(frame, stratum))
    results = pd.DataFrame(rows)
    results["holm_p_three_contrasts_per_variant_stratum"] = results.groupby(
        ["stratum", "feature_id"], sort=False
    )["p_value"].transform(lambda values: adjust_p(values, "holm"))
    results["holm_p_all_variants_contrasts_per_stratum"] = results.groupby(
        "stratum", sort=False
    )["p_value"].transform(lambda values: adjust_p(values, "holm"))
    return results


def falsification_decision(results: pd.DataFrame) -> dict[str, object]:
    key_strata = [
        "full_exact",
        "timing_le90",
        "siemens54",
        "timing_le90_siemens54",
        "timing_le90_siemens54_overlap_sites",
        "original_t1",
        "legacy_qc_include",
    ]
    partials = [
        "partial_degree_nodal_eff_fa",
        "partial_degree_fa_edges_nodal_eff_fa",
        "partial_degree_strength_fa",
        "partial_degree_fa_edges_strength_fa",
    ]
    view = results[
        results["feature_id"].isin(partials) & results["stratum"].isin(key_strata)
    ]
    mci = view[view["contrast"].eq("MCI_vs_CN")]
    ad = view[view["contrast"].eq("AD_vs_MCI")]
    mci_positive_fraction = float((mci["rank_normal_beta"] > 0).mean())
    ad_negative_fraction = float((ad["rank_normal_beta"] < 0).mean())
    all_mci_positive = bool((mci["rank_normal_beta"] > 0).all())
    all_ad_negative = bool((ad["rank_normal_beta"] < 0).all())
    inverted_u_survives = bool(all_mci_positive and all_ad_negative)
    return {
        "candidate": "MCI_peak_in_frontal_topology_FA_coupling",
        "support_controls": partials,
        "key_strata": key_strata,
        "mci_vs_cn_positive_fraction_after_support_control": mci_positive_fraction,
        "ad_vs_mci_negative_fraction_after_support_control": ad_negative_fraction,
        "all_mci_vs_cn_positive_after_support_control": all_mci_positive,
        "all_ad_vs_mci_negative_after_support_control": all_ad_negative,
        "inverted_u_survives_falsification": inverted_u_survives,
        "decision": "REJECT_BIOLOGICAL_INVERTED_U_HEADLINE"
        if not inverted_u_survives
        else "RETAIN_FOR_FURTHER_TESTING",
        "reason": "MCI elevation reverses or attenuates after controlling node degree and FA-edge support"
        if not inverted_u_survives
        else "Both stage directions remained consistent after support controls",
    }


def make_figure(subjects: pd.DataFrame, cohort: pd.DataFrame, out: Path) -> None:
    plot = subjects.merge(cohort[["subject_id", "group"]], on="subject_id", how="left")
    variants = [
        "raw_nodal_eff_fa",
        "partial_degree_nodal_eff_fa",
        "partial_degree_fa_edges_nodal_eff_fa",
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), sharey=True)
    rng = np.random.default_rng(20260718)
    for axis, variant in zip(axes, variants):
        for position, group in enumerate(GROUPS):
            values = plot.loc[plot["group"].eq(group), variant].dropna().to_numpy()
            jitter = rng.normal(position, 0.045, len(values))
            axis.scatter(jitter, values, s=8, alpha=0.22)
            if len(values):
                axis.plot(
                    [position - 0.22, position + 0.22],
                    [np.median(values), np.median(values)],
                    color="black",
                    linewidth=2,
                )
        axis.set_xticks(range(3), GROUPS)
        axis.set_title(variant.replace("_", "\n"), fontsize=8)
        axis.axhline(0, color="grey", linewidth=0.7)
    axes[0].set_ylabel("Within-subject frontal topology--FA Spearman rho")
    fig.suptitle("Apparent stage pattern before and after edge-support control")
    fig.tight_layout()
    fig.savefig(out / "coupling_support_falsification.png", dpi=240)
    plt.close(fig)


def write_summary(
    out: Path,
    reproduction: dict[str, object],
    decision: dict[str, object],
    results: pd.DataFrame,
) -> None:
    strict = results[
        results["stratum"].isin(
            [
                "full_exact",
                "timing_le90_siemens54",
                "timing_le90_siemens54_overlap_sites",
                "legacy_qc_include",
            ]
        )
        & results["feature_id"].isin(
            [
                "raw_nodal_eff_fa",
                "partial_degree_nodal_eff_fa",
                "partial_degree_fa_edges_nodal_eff_fa",
            ]
        )
    ].copy()
    text = f"""# Frontal topology--FA coupling support falsification

**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}  
**Decision:** {decision['decision']}  
**Scope:** existing node tables only; no image arrays or processing.

## Construct and exact reproduction

The historical value is one participant-level Spearman correlation across 32 anatomical frontal AAL3 nodes between SIFT2/fd_sum-derived nodal topology and node-wise FA means. Current node tables permit regeneration for {reproduction['current_node_table_subjects']} subjects, whereas the published network-coupling table contains {reproduction['published_coupling_subjects']}. All {reproduction['matched_subjects']} shared subjects reproduce exactly: {reproduction['finite_pairs']} finite pairs agree to maximum absolute difference {reproduction['max_absolute_difference']:.3g}, and missing-value pattern agreement is {reproduction['nan_pattern_matches']}. The 15 additional current-node subjects are retained in this reanalysis and recorded as historical table attrition, not silently discarded.

Although topology and FA use different edge weights, they share tractography support. We therefore rank-residualised topology and FA for node degree, then for node degree plus the number of positive FA edges, before recomputing within-subject correlation.

## Falsification result

The original MCI-above-CN and AD-below-MCI shape does not survive support control. Across the four partial-correlation variants and seven declared strata, the fraction of MCI--CN effects remaining positive is {decision['mci_vs_cn_positive_fraction_after_support_control']:.3f}; all positive is {decision['all_mci_vs_cn_positive_after_support_control']}. The AD--MCI negative fraction is {decision['ad_vs_mci_negative_fraction_after_support_control']:.3f}; all negative is {decision['all_ad_vs_mci_negative_after_support_control']}.

This rules out the attractive interpretation that the raw pattern demonstrates MCI compensation or a biological inverted-U. It may instead reflect how edge support and nodal connectedness jointly enter the two maps. Residualized AD-related reductions remain directionally interesting in some strata, but they are not stable enough for a biological headline.

## Selected model evidence

{markdown_table(strict, ['feature_id', 'stratum', 'contrast', 'n_CN', 'n_MCI', 'n_AD', 'rank_normal_beta', 'ci95_low', 'ci95_high', 'p_value', 'holm_p_three_contrasts_per_variant_stratum', 'holm_p_all_variants_contrasts_per_stratum'], 60)}

## Claim disposition

- **Reject:** “Frontal topology--FA coupling rises in MCI as compensation and collapses in AD.”
- **Retain as a methods result:** shared edge support can produce a compelling stage pattern in topology--microstructure coupling; degree/support residualization is a necessary sensitivity.
- **Do not infer:** compensation, mechanism, causality, preserved cognition, or clinical utility.
"""
    (out / "coupling_support_falsification_summary.md").write_text(text, encoding="utf-8")


def validate_release(
    out: Path,
    reproduction: dict[str, object],
    decision: dict[str, object],
) -> dict[str, object]:
    results = pd.read_csv(out / "coupling_support_contrasts.csv")
    subjects = pd.read_csv(out / "coupling_support_subjects.csv")
    checks = {
        "raw_reproduction_pass": bool(reproduction["pass"]),
        "current_node_subject_intersection_529": len(subjects) == 529
        and subjects["subject_id"].nunique() == 529,
        "six_variants_present": set(results["feature_id"]) == set(VARIANTS),
        "three_contrasts_present": set(results["contrast"])
        == {"MCI_vs_CN", "AD_vs_CN", "AD_vs_MCI"},
        "p_values_bounded": bool(results["p_value"].dropna().between(0, 1).all()),
        "inverted_u_rejected": decision["decision"]
        == "REJECT_BIOLOGICAL_INVERTED_U_HEADLINE",
        "required_files_present": all(
            (out / relative).is_file()
            for relative in [
                "coupling_support_subjects.csv",
                "coupling_support_contrasts.csv",
                "raw_reproduction_validation.json",
                "falsification_decision.json",
                "coupling_support_falsification_summary.md",
                "coupling_support_falsification.png",
                "run_manifest.json",
            ]
        ),
    }
    checks["pass"] = bool(all(checks.values()))
    checks["files"] = {
        str(path.relative_to(out)): {"bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in sorted(out.rglob("*"))
        if path.is_file() and path.name != "release_validation.json"
    }
    return checks


def run_release(out: Path, inputs: dict[str, Path] = INPUTS) -> None:
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite existing release: {out}")
    for key, path in inputs.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing {key}: {path}")
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{out.name}.", dir=out.parent))
    try:
        cohort = load_exact_cohort(inputs["exact_manifest"])
        subjects = build_subject_variants(
            inputs["mapping"], inputs["node_graph"], inputs["node_micro"]
        )
        reproduction = validate_raw_reproduction(
            subjects, inputs["published_network_coupling"]
        )
        if not reproduction["pass"]:
            raise RuntimeError(f"Raw coupling reproduction failed: {reproduction}")
        strata = build_strata(cohort)
        results = model_variants(subjects, cohort, strata)
        decision = falsification_decision(results)

        subjects.to_csv(temporary / "coupling_support_subjects.csv", index=False)
        results.to_csv(temporary / "coupling_support_contrasts.csv", index=False)
        (temporary / "raw_reproduction_validation.json").write_text(
            json.dumps(reproduction, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (temporary / "falsification_decision.json").write_text(
            json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        make_figure(subjects, cohort, temporary)
        write_summary(temporary, reproduction, decision, results)
        run_manifest = {
            "schema_version": "1.0",
            "release": "coupling_support_falsification_v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "scientific_status": "EXPLORATORY_CANDIDATE_FALSIFICATION",
            "image_or_voxel_data_read": False,
            "image_processing_run": False,
            "production_outputs_modified": False,
            "inputs": {
                key: {
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
                for key, path in inputs.items()
            },
            "variants": list(VARIANTS),
        }
        (temporary / "run_manifest.json").write_text(
            json.dumps(run_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        validation = validate_release(temporary, reproduction, decision)
        (temporary / "release_validation.json").write_text(
            json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if not validation["pass"]:
            raise RuntimeError(f"Release validation failed: {validation}")
        os.rename(temporary, out)
    except Exception:
        failed = temporary.with_name(temporary.name + ".failed")
        if temporary.exists() and not failed.exists():
            os.rename(temporary, failed)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_release(args.out.resolve())
    print(f"PASS: coupling support falsification written to {args.out.resolve()}")


if __name__ == "__main__":
    main()
