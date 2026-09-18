#!/usr/bin/env python3
"""Bounded aggregate-only screen of additional connectome dimensions.

This screen asks whether four predeclared candidate layers can add a defensible
biological dimension to the locked AxD MCI-CN / RD AD-MCI paper:

1. global structural topology;
2. fixed-velocity tract-length delay proxies;
3. legacy connectome brain-age gap; and
4. within-subject spatial dispersion across the complete nonredundant
   17-system inventory.

The screen uses the exact diagnosis-at-DTI cohort and the same site-fixed,
site-clustered covariate model as the locked phenotype.  It writes aggregate
results only.  It is a falsification screen on historical outputs, not a new
confirmatory family and not a license to promote a statistically significant
but invalid construct.
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

import numpy as np
import pandas as pd

from stage_specific_diffusivity_analysis import (
    COMPONENT_METRICS,
    PRIMARY_NETWORKS,
    adjust,
    eligible_cohort,
    fit_group_contrast,
)
from submission_sprint_analysis import collect_metrics, load_exact_cohort, normal_scores


PROJECT = Path("/home/ec2-user/exp")
DEFAULT_OUT = PROJECT / "research_audit" / "outputs" / "expanded_dimension_screen_v1"
TOPOLOGY = Path(
    "/data/derivatives/qc/analysis_cohort/06_global_graph_live/"
    "live_global_graph_subject_table.csv"
)
DELAY = Path(
    "/data/derivatives/qc/analysis_cohort/13_delay/"
    "delay_subject_level_fd_sum_len_mean.csv"
)
BRAIN_AGE = Path(
    "/data/derivatives/qc/analysis_cohort/11_brain_age_live/"
    "live_brain_age_predictions.csv"
)
BRAIN_AGE_PERFORMANCE = Path(
    "/data/derivatives/qc/analysis_cohort/11_brain_age_live/"
    "live_brain_age_model_performance.csv"
)

CONTRASTS = (
    ("MCI-CN", "MCI", "CN"),
    ("AD-MCI", "AD", "MCI"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def merge_aggregate_source(
    cohort: pd.DataFrame, path: Path, columns: list[str], drop: list[str]
) -> pd.DataFrame:
    source = pd.read_csv(path, usecols=["subject_id", *drop, *columns])
    source["subject_id"] = source["subject_id"].astype(str)
    source = source.drop(columns=drop)
    if source["subject_id"].duplicated().any():
        raise ValueError(f"Duplicate participant keys in {path}")
    return cohort.merge(source, on="subject_id", how="inner", validate="1:1")


def fit_domain(
    frame: pd.DataFrame,
    domain: str,
    metrics: list[str],
    construct_valid: bool,
    historical_output_safe: bool,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for metric in metrics:
        for contrast, left, right in CONTRASTS:
            result = fit_group_contrast(
                frame,
                metric,
                {left: 1.0, right: -1.0},
                reference="CN",
                scope="full_site_fe",
            )
            rows.append(
                {
                    "domain": domain,
                    "metric": metric,
                    "contrast": contrast,
                    "n": result["n"],
                    "n_sites": result["n_sites"],
                    "beta": result["beta"],
                    "ci95_low": result["ci95_low"],
                    "ci95_high": result["ci95_high"],
                    "p_value": result["p_value"],
                    "construct_valid": construct_valid,
                    "historical_output_safe": historical_output_safe,
                    "model": "rank-normalized outcome; age, sex, gap, T1 source, and site fixed effects; site-clustered covariance",
                }
            )
    output = pd.DataFrame(rows)
    output["holm_p_within_domain"] = adjust(output["p_value"], "holm")
    output["statistically_supported"] = output["holm_p_within_domain"].lt(0.05)
    output["headline_eligible"] = (
        output["statistically_supported"]
        & output["construct_valid"]
        & output["historical_output_safe"]
    )
    return output


def build_profile_dispersion(cohort: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = collect_metrics()
    micro = metrics[
        metrics["mapping"].isin(["anatomical", "functional"])
        & metrics["feature_family"].eq("microstructure")
        & metrics["metric"].isin(COMPONENT_METRICS)
    ].copy()
    micro["network_id"] = (
        micro["mapping"].astype(str) + "::" + micro["network"].astype(str)
    )
    micro = micro[micro["network_id"].isin(PRIMARY_NETWORKS)].copy()
    micro["component"] = micro["metric"].map(COMPONENT_METRICS)
    micro = micro.merge(
        cohort[["subject_id"]], on="subject_id", how="inner", validate="m:1"
    )
    micro = micro[np.isfinite(micro["value"])].copy()
    micro["feature"] = micro["network_id"] + "::" + micro["component"]
    micro["feature_z"] = micro.groupby("feature", sort=False)["value"].transform(
        normal_scores
    )

    summary = (
        micro.groupby(["subject_id", "component"], as_index=False)
        .agg(
            n_systems=("feature_z", "size"),
            profile_sd=("feature_z", "std"),
            profile_mad=(
                "feature_z",
                lambda values: float(
                    np.median(np.abs(values.to_numpy() - np.median(values.to_numpy())))
                ),
            ),
        )
    )
    complete = summary[summary["n_systems"].eq(len(PRIMARY_NETWORKS))].copy()
    wide = complete.pivot(
        index="subject_id", columns="component", values=["profile_sd", "profile_mad"]
    )
    wide.columns = [f"{stat}_{component}" for stat, component in wide.columns]
    wide = wide.reset_index().merge(cohort, on="subject_id", how="inner", validate="1:1")

    coverage_rows: list[dict[str, object]] = []
    for component in ("axial", "radial"):
        component_rows = complete[complete["component"].eq(component)].merge(
            cohort[["subject_id", "group"]], on="subject_id", validate="1:1"
        )
        counts = component_rows["group"].value_counts()
        coverage_rows.append(
            {
                "component": component,
                "required_systems": len(PRIMARY_NETWORKS),
                "n_complete": int(len(component_rows)),
                "n_CN": int(counts.get("CN", 0)),
                "n_MCI": int(counts.get("MCI", 0)),
                "n_AD": int(counts.get("AD", 0)),
            }
        )
    return wide, pd.DataFrame(coverage_rows)


def construct_register(overall_brain_age_r2: float) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "domain": "global_topology",
                "decision": "REJECT_HEADLINE",
                "reason": "Exact site-aware adjacent contrasts do not survive within-domain correction; legacy topology is density and tractography-recipe sensitive.",
                "key_validity_value": "historical count/fd_sum topology only",
            },
            {
                "domain": "fixed_velocity_delay",
                "decision": "REJECT_CONSTRUCT",
                "reason": "The dashboard applies one fixed 6 mm/ms velocity, so delay is a rescaled tract-length proxy rather than measured conduction physiology; exact contrasts do not survive correction.",
                "key_validity_value": "velocity fixed at 6 mm/ms",
            },
            {
                "domain": "legacy_brain_age",
                "decision": "REJECT_CONSTRUCT",
                "reason": "A corrected group contrast cannot rescue a brain-age model with negligible out-of-sample age prediction.",
                "key_validity_value": f"overall R2={overall_brain_age_r2:.6f}",
            },
            {
                "domain": "complete_system_profile_dispersion",
                "decision": "SUPPLEMENT_ONLY_IF_SUPPORTED",
                "reason": "This is a computable individual spatial-unevenness construct, but it is post-selection, coverage-restricted, and not independent confirmation of the primary phenotype.",
                "key_validity_value": "requires all 17 nonredundant systems",
            },
            {
                "domain": "epicenter_or_network_diffusion",
                "decision": "NOT_ESTIMABLE",
                "reason": "The current join lacks contemporaneous regional atrophy or amyloid/tau maps needed to seed and validate a disease-spread model.",
                "key_validity_value": "no scan-aligned pathology target",
            },
            {
                "domain": "spectral_resilience_or_communicability",
                "decision": "DEFER_CORRECTED_MATRICES",
                "reason": "These measures are mathematically computable but prior AD structural-connectome work already covers graph spectra, algebraic connectivity, k-core, and robustness; historical density/QC variation makes a new result unsafe.",
                "key_validity_value": "prior art occupied and recipe sensitive",
            },
        ]
    )


def write_summary(
    out: Path,
    tests: pd.DataFrame,
    constructs: pd.DataFrame,
    coverage: pd.DataFrame,
    overall_brain_age_r2: float,
) -> None:
    supported = tests[tests["statistically_supported"]]
    eligible = tests[tests["headline_eligible"]]
    lines = [
        "# Expanded data-dimension screen",
        "",
        f"**Generated:** {utc_now()}  ",
        "**Scope:** historical outputs, exact diagnosis-at-DTI cohort, aggregate release only  ",
        f"**Tests:** {len(tests)} across four prespecified construct families  ",
        f"**Within-family corrected positives:** {len(supported)}  ",
        f"**Additional headline-eligible dimensions:** {len(eligible)}",
        "",
        "## Decision",
        "",
        "No additional headline dimension is retained. Global topology and fixed-velocity delay do not survive exact site-aware correction. The legacy brain-age construct has overall R2 "
        f"{overall_brain_age_r2:.4f}, so any diagnostic contrast is not interpretable as accelerated ageing. Complete-system spatial dispersion is retained only as a falsification screen and does not independently confirm the two-axis phenotype.",
        "",
        "## Corrected positive rows",
        "",
    ]
    if supported.empty:
        lines.append("None.")
    else:
        lines.append("| Domain | Metric | Contrast | beta | p | Holm p | Headline eligible |")
        lines.append("|---|---|---:|---:|---:|---:|---|")
        for row in supported.itertuples(index=False):
            lines.append(
                f"| {row.domain} | {row.metric} | {row.contrast} | {row.beta:.4f} | "
                f"{row.p_value:.4g} | {row.holm_p_within_domain:.4g} | {bool(row.headline_eligible)} |"
            )
    lines.extend(
        [
            "",
            "## Complete-system dispersion coverage",
            "",
            "| Component | Complete N | CN | MCI | AD |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in coverage.itertuples(index=False):
        lines.append(
            f"| {row.component} | {row.n_complete} | {row.n_CN} | {row.n_MCI} | {row.n_AD} |"
        )
    lines.extend(
        [
            "",
            "## Construct decisions",
            "",
            "| Domain | Decision | Reason |",
            "|---|---|---|",
        ]
    )
    for row in constructs.itertuples(index=False):
        lines.append(f"| {row.domain} | {row.decision} | {row.reason} |")
    lines.extend(
        [
            "",
            "This screen does not weaken the locked AxD/RD result. It prevents unrelated dashboard tangents from being used to manufacture a third biological layer.",
        ]
    )
    (out / "screen_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(out: Path) -> None:
    if out.exists():
        raise FileExistsError(f"Versioned output already exists: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    cohort = eligible_cohort(load_exact_cohort())

    topology = merge_aggregate_source(
        cohort,
        TOPOLOGY,
        ["mean_strength", "density", "global_efficiency", "charpath_len"],
        ["group", "age", "sex", "phase"],
    )
    delay = merge_aggregate_source(
        cohort,
        DELAY,
        [
            "mean_delay_ms",
            "median_delay_ms",
            "p90_delay_ms",
            "delay_burden_ms",
            "long_delay_fraction",
            "delay_path_ms",
        ],
        ["group"],
    )
    brain_age = merge_aggregate_source(
        cohort,
        BRAIN_AGE,
        ["BAG", "BAG_age_corrected"],
        [],
    )
    performance = pd.read_csv(BRAIN_AGE_PERFORMANCE)
    overall = performance[performance["group"].eq("Overall")]
    if len(overall) != 1:
        raise ValueError("Expected exactly one overall brain-age performance row")
    overall_r2 = float(overall.iloc[0]["r2"])

    profile, coverage = build_profile_dispersion(cohort)
    domains = [
        fit_domain(
            topology,
            "global_topology",
            ["mean_strength", "density", "global_efficiency", "charpath_len"],
            construct_valid=True,
            historical_output_safe=False,
        ),
        fit_domain(
            delay,
            "fixed_velocity_delay",
            [
                "mean_delay_ms",
                "median_delay_ms",
                "p90_delay_ms",
                "delay_burden_ms",
                "long_delay_fraction",
                "delay_path_ms",
            ],
            construct_valid=False,
            historical_output_safe=False,
        ),
        fit_domain(
            brain_age,
            "legacy_brain_age",
            ["BAG", "BAG_age_corrected"],
            construct_valid=False,
            historical_output_safe=False,
        ),
        fit_domain(
            profile,
            "complete_system_profile_dispersion",
            [
                "profile_sd_axial",
                "profile_mad_axial",
                "profile_sd_radial",
                "profile_mad_radial",
            ],
            construct_valid=True,
            historical_output_safe=False,
        ),
    ]
    tests = pd.concat(domains, ignore_index=True)
    constructs = construct_register(overall_r2)

    with tempfile.TemporaryDirectory(prefix=f".{out.name}.", dir=out.parent) as temp_dir:
        temporary = Path(temp_dir)
        tests.to_csv(temporary / "adjusted_candidate_tests.csv", index=False)
        constructs.to_csv(temporary / "construct_decisions.csv", index=False)
        coverage.to_csv(temporary / "profile_dispersion_coverage.csv", index=False)
        write_summary(temporary, tests, constructs, coverage, overall_r2)

        forbidden = {"subject_id", "participant_id", "pair_id", "image_id"}
        no_ids = all(
            forbidden.isdisjoint({column.lower() for column in frame.columns})
            for frame in (tests, constructs, coverage)
        )
        expected_counts = {
            "global_topology": 8,
            "fixed_velocity_delay": 12,
            "legacy_brain_age": 4,
            "complete_system_profile_dispersion": 8,
        }
        observed_counts = tests["domain"].value_counts().to_dict()
        validation_checks = {
            "expected_test_count_32": len(tests) == 32,
            "expected_domain_counts": observed_counts == expected_counts,
            "finite_effects_and_p_values": bool(
                np.isfinite(tests[["beta", "p_value", "holm_p_within_domain"]]).all().all()
            ),
            "p_values_in_unit_interval": bool(
                tests[["p_value", "holm_p_within_domain"]].ge(0).all().all()
                and tests[["p_value", "holm_p_within_domain"]].le(1).all().all()
            ),
            "no_identifier_columns": no_ids,
            "exact_eligible_n_515": len(cohort) == 515,
            "brain_age_performance_captured": math.isclose(
                overall_r2, 0.009548509083225243, rel_tol=0, abs_tol=1e-12
            ),
            "no_headline_eligible_addition": not tests["headline_eligible"].any(),
        }
        validation = {
            "schema_version": "1.0.0",
            "generated_utc": utc_now(),
            "status": "PASS" if all(validation_checks.values()) else "FAIL",
            "checks": validation_checks,
            "test_count": int(len(tests)),
            "corrected_positive_count": int(tests["statistically_supported"].sum()),
            "headline_eligible_count": int(tests["headline_eligible"].sum()),
        }
        (temporary / "validation.json").write_text(
            json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if validation["status"] != "PASS":
            raise RuntimeError(f"Expanded-dimension validation failed: {validation_checks}")

        inputs = [TOPOLOGY, DELAY, BRAIN_AGE, BRAIN_AGE_PERFORMANCE]
        release = {
            "schema_version": "1.0.0",
            "generated_utc": utc_now(),
            "analysis": "expanded_dimension_screen_v1",
            "status": "PASS",
            "privacy": "aggregate_only_no_participant_identifiers",
            "input_records": [
                {"path": str(path), "sha256": sha256(path), "size_bytes": path.stat().st_size}
                for path in inputs
            ],
            "outputs": {},
        }
        for path in sorted(temporary.iterdir()):
            if path.name == "release_manifest.json":
                continue
            release["outputs"][path.name] = {
                "sha256": sha256(path),
                "size_bytes": path.stat().st_size,
            }
        (temporary / "release_manifest.json").write_text(
            json.dumps(release, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.rename(temporary, out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    run(args.out.expanduser().resolve())
    print(args.out.expanduser().resolve())


if __name__ == "__main__":
    main()
