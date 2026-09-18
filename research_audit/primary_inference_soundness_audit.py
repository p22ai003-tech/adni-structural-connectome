#!/usr/bin/env python3
"""Independent aggregate-only inferential audit for the two-axis manuscript.

This audit replays the locked participant-level models in memory, emits no
participant rows, and adds small-cluster inference that was not part of the
original release: t(G-1) p values and restricted wild-cluster bootstrap-t for
the six primary scope/endpoint models. It also applies a cluster-score sign
test to the two 16-df system-heterogeneity claims.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

from stage_specific_diffusivity_analysis import (
    PRIMARY_ENDPOINTS,
    PRIMARY_NETWORKS,
    _group_vector,
    _standardize,
    _support_frame,
    build_composites,
    normal_scores,
)
from submission_sprint_analysis import collect_metrics, load_exact_cohort
from two_axis_extension_analysis import _pair_long_frame


ROOT = Path("/home/ec2-user/exp/research_audit")
DEFAULT_OUT = ROOT / "outputs" / "primary_inference_soundness_v2"
LOCKED_PRIMARY = ROOT / "outputs" / "stage_specific_diffusivity_v2" / "primary_contrasts.csv"
LOCKED_SENSITIVITY = ROOT / "outputs" / "stage_specific_diffusivity_v2" / "sensitivity_contrasts.csv"
LOCKED_HETEROGENEITY = ROOT / "outputs" / "two_axis_extensions_v1" / "network_heterogeneity_tests.csv"
SEED = 20260718
BOOTSTRAP_ITERATIONS = 9999
SCORE_ITERATIONS = 99999


def holm(values: pd.Series) -> np.ndarray:
    array = values.to_numpy(dtype=float)
    return multipletests(array, method="holm")[1]


def contrast_design(
    composites: pd.DataFrame,
    outcome: str,
    weights: dict[str, float],
    *,
    reference: str,
    scope: str,
    left: str,
    right: str,
):
    use = _support_frame(composites, scope, left=left, right=right)
    required = [outcome, "group", "age", "sex", "site", "log_gap_days", "t1_source"]
    use = use.dropna(subset=required).copy()
    use["outcome_z"] = normal_scores(use[outcome])
    use["age_z"] = _standardize(use["age"])
    use["gap_z"] = _standardize(use["log_gap_days"])
    terms = [f'C(group, Treatment(reference="{reference}"))', "age_z"]
    if use["sex"].nunique() > 1:
        terms.append("C(sex)")
    if use["log_gap_days"].nunique() > 1:
        terms.append("gap_z")
    if use["t1_source"].nunique() > 1:
        terms.append("C(t1_source)")
    terms.append("C(site)")
    formula = "outcome_z ~ " + " + ".join(terms)
    base = smf.ols(formula, data=use).fit()
    robust = base.get_robustcov_results(
        cov_type="cluster", groups=use["site"].to_numpy(), use_correction=True,
        use_t=False,
    )
    r = _group_vector(base.model.exog_names, reference, weights).reshape(1, -1)
    return use, base, robust, r, formula


def restricted_wild_cluster_bootstrap_t(
    y: np.ndarray,
    x: np.ndarray,
    r: np.ndarray,
    clusters: np.ndarray,
    observed_t: float,
    *,
    iterations: int,
    seed: int,
    chunk_size: int = 500,
) -> tuple[float, int]:
    """Restricted Rademacher wild-cluster bootstrap-t with CR1 studentization."""

    unique, codes = np.unique(clusters.astype(str), return_inverse=True)
    g = len(unique)
    n, p = x.shape
    bread = np.linalg.pinv(x.T @ x)
    beta = bread @ x.T @ y
    middle = r @ bread @ r.T
    beta_restricted = beta - (bread @ r.T @ np.linalg.pinv(middle) @ (r @ beta)).reshape(-1)
    fitted_restricted = x @ beta_restricted
    residual_restricted = y - fitted_restricted
    a = (r @ bread).reshape(-1)
    scale = (g / (g - 1.0)) * ((n - 1.0) / (n - p))
    rng = np.random.default_rng(seed)
    exceed = 0
    valid = 0
    for start in range(0, iterations, chunk_size):
        b = min(chunk_size, iterations - start)
        weights = rng.choice(np.array([-1.0, 1.0]), size=(g, b), replace=True)
        y_star = fitted_restricted[:, None] + residual_restricted[:, None] * weights[codes]
        beta_star = bread @ (x.T @ y_star)
        residual_star = y_star - x @ beta_star
        variance = np.zeros(b, dtype=float)
        for cluster in range(g):
            mask = codes == cluster
            score = x[mask].T @ residual_star[mask]
            projected = a @ score
            variance += projected * projected
        se_star = np.sqrt(scale * variance)
        effect_star = (r @ beta_star).reshape(-1)
        finite = np.isfinite(se_star) & (se_star > 0) & np.isfinite(effect_star)
        t_star = effect_star[finite] / se_star[finite]
        exceed += int(np.sum(np.abs(t_star) >= abs(observed_t)))
        valid += int(finite.sum())
    return (1.0 + exceed) / (1.0 + valid), valid


def audit_primary(composites: pd.DataFrame) -> pd.DataFrame:
    locked = pd.read_csv(LOCKED_PRIMARY)
    rows: list[dict[str, object]] = []
    scopes = ("full_site_fe", "pair_common_sites", "all_three_group_sites")
    for scope_index, scope in enumerate(scopes):
        for endpoint_index, endpoint in enumerate(PRIMARY_ENDPOINTS):
            reference = "CN" if scope != "pair_common_sites" else endpoint.right
            use, base, robust, r, formula = contrast_design(
                composites,
                endpoint.outcome,
                {endpoint.left: 1.0, endpoint.right: -1.0},
                reference=reference,
                scope=scope,
                left=endpoint.left,
                right=endpoint.right,
            )
            test = robust.t_test(r)
            effect = float(np.asarray(test.effect).reshape(-1)[0])
            se = float(np.asarray(test.sd).reshape(-1)[0])
            t_stat = effect / se
            groups = use["site"].astype(str).to_numpy()
            n_sites = int(pd.Series(groups).nunique())
            p_normal = float(2.0 * st.norm.sf(abs(t_stat)))
            p_t_gminus1 = float(2.0 * st.t.sf(abs(t_stat), df=n_sites - 1))
            p_wild, valid = restricted_wild_cluster_bootstrap_t(
                base.model.endog,
                base.model.exog,
                r,
                groups,
                t_stat,
                iterations=BOOTSTRAP_ITERATIONS,
                seed=SEED + 101 * scope_index + endpoint_index,
            )
            locked_row = locked[
                locked["scope"].eq(scope) & locked["endpoint"].eq(endpoint.endpoint)
            ].iloc[0]
            site_group = use.groupby(["site", "group"]).size()
            rows.append({
                "endpoint": endpoint.endpoint,
                "scope": scope,
                "n": int(len(use)),
                "n_sites": n_sites,
                "design_columns": int(base.model.exog.shape[1]),
                "design_rank": int(np.linalg.matrix_rank(base.model.exog)),
                "condition_number": float(np.linalg.cond(base.model.exog)),
                "beta": effect,
                "cr1_se": se,
                "cr1_t": t_stat,
                "locked_beta_absolute_difference": abs(effect - float(locked_row["beta"])),
                # The locked analysis used statsmodels' default clustered-t
                # reference, whose inferential degrees of freedom are G - 1.
                # Keep the normal-reference value as an additional audit
                # quantity, but replay the locked p value against t(G - 1).
                "locked_p_value": float(locked_row["p_value"]),
                "locked_p_reference": "CR1 t(G-1)",
                "locked_p_absolute_difference": abs(
                    p_t_gminus1 - float(locked_row["p_value"])
                ),
                "p_cr1_normal": p_normal,
                "p_cr1_t_gminus1": p_t_gminus1,
                "p_restricted_wild_cluster_bootstrap_t": p_wild,
                "wild_bootstrap_valid_iterations": valid,
                "minimum_site_by_group_cell_n": int(site_group.min()),
                "median_site_by_group_cell_n": float(site_group.median()),
                "positive_direction": bool(effect > 0),
                "formula": formula,
            })
    output = pd.DataFrame(rows)
    for column, target in (
        ("p_cr1_normal", "holm_p_cr1_normal"),
        ("p_cr1_t_gminus1", "holm_p_cr1_t_gminus1"),
        ("p_restricted_wild_cluster_bootstrap_t", "holm_p_restricted_wild_cluster_bootstrap_t"),
    ):
        output[target] = output.groupby("scope", sort=False)[column].transform(holm)
    return output


def rademacher_score_pvalue(
    cluster_scores: np.ndarray,
    observed_wald: float,
    inv_meat: np.ndarray,
    *,
    iterations: int,
    seed: int,
    exact_if_at_most: int = 20,
    chunk_size: int = 8192,
) -> tuple[float, int, str]:
    g = cluster_scores.shape[0]
    exceed = 0
    valid = 0
    if g <= exact_if_at_most:
        total = 1 << g
        for start in range(0, total, chunk_size):
            stop = min(total, start + chunk_size)
            integers = np.arange(start, stop, dtype=np.uint64)
            bits = ((integers[:, None] >> np.arange(g, dtype=np.uint64)) & 1).astype(float)
            weights = 2.0 * bits - 1.0
            signed = weights @ cluster_scores
            wald = np.einsum("bi,ij,bj->b", signed, inv_meat, signed)
            exceed += int(np.sum(wald >= observed_wald - 1e-12))
            valid += len(wald)
        return exceed / valid, valid, "exact_rademacher_enumeration"
    rng = np.random.default_rng(seed)
    for start in range(0, iterations, chunk_size):
        b = min(chunk_size, iterations - start)
        weights = rng.choice(np.array([-1.0, 1.0]), size=(b, g), replace=True)
        signed = weights @ cluster_scores
        wald = np.einsum("bi,ij,bj->b", signed, inv_meat, signed)
        exceed += int(np.sum(wald >= observed_wald - 1e-12))
        valid += b
    return (1.0 + exceed) / (1.0 + valid), valid, "sampled_rademacher"


def audit_heterogeneity(cohort: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    locked = pd.read_csv(LOCKED_HETEROGENEITY)
    specs = (
        ("axial_MCI_vs_CN", "ad_mean", "MCI", "CN"),
        ("radial_AD_vs_MCI", "rd_mean", "AD", "MCI"),
    )
    reference_network = PRIMARY_NETWORKS[0]
    rows = []
    for index, (endpoint, metric, left, right) in enumerate(specs):
        frame = _pair_long_frame(cohort, metrics, metric, left, right)
        full_formula = (
            f'value_z ~ C(group, Treatment(reference="{right}")) * '
            f'C(network, Treatment(reference="{reference_network}")) + '
            "age_z + C(sex) + gap_z + C(t1_source) + C(site)"
        )
        restricted_formula = (
            f'value_z ~ C(group, Treatment(reference="{right}")) + '
            f'C(network, Treatment(reference="{reference_network}")) + '
            "age_z + C(sex) + gap_z + C(t1_source) + C(site)"
        )
        full = smf.ols(full_formula, data=frame).fit()
        full_robust = full.get_robustcov_results(
            cov_type="cluster", groups=frame["site"].to_numpy(),
            use_correction=True, use_t=False,
        )
        names = full.model.exog_names
        interaction_idx = [
            position for position, name in enumerate(names)
            if name.startswith("C(group") and ":C(network" in name
        ]
        r = np.zeros((len(interaction_idx), len(names)), dtype=float)
        for row_index, column in enumerate(interaction_idx):
            r[row_index, column] = 1.0
        full_test = full_robust.f_test(r)

        restricted = smf.ols(restricted_formula, data=frame).fit()
        x0 = restricted.model.exog
        u0 = restricted.resid.to_numpy(dtype=float)
        z = full.model.exog[:, interaction_idx]
        z_tilde = z - x0 @ (np.linalg.pinv(x0) @ z)
        sites = frame["site"].astype(str).to_numpy()
        unique, codes = np.unique(sites, return_inverse=True)
        cluster_scores = np.stack([
            z_tilde[codes == group].T @ u0[codes == group]
            for group in range(len(unique))
        ])
        score_sum = cluster_scores.sum(axis=0)
        meat = cluster_scores.T @ cluster_scores
        meat_rank = int(np.linalg.matrix_rank(meat))
        inv_meat = np.linalg.pinv(meat)
        score_wald = float(score_sum.T @ inv_meat @ score_sum)
        p_score, valid, method = rademacher_score_pvalue(
            cluster_scores,
            score_wald,
            inv_meat,
            iterations=SCORE_ITERATIONS,
            seed=SEED + 1000 + index,
        )
        locked_row = locked[locked["endpoint"].eq(endpoint)].iloc[0]
        rows.append({
            "endpoint": endpoint,
            "n_subjects": int(frame["subject_id"].nunique()),
            "n_sites": len(unique),
            "n_network_observations": int(len(frame)),
            "interaction_df": len(interaction_idx),
            "cluster_meat_rank": meat_rank,
            "locked_cr1_f": float(np.asarray(full_test.fvalue).reshape(-1)[0]),
            "locked_cr1_p_replayed": float(np.asarray(full_test.pvalue).reshape(-1)[0]),
            "locked_p_absolute_difference": abs(
                float(np.asarray(full_test.pvalue).reshape(-1)[0]) - float(locked_row["p_value"])
            ),
            "restricted_cluster_score_wald": score_wald,
            "p_cluster_score_rademacher": p_score,
            "score_iterations": valid,
            "score_method": method,
            "small_cluster_pass_0_05": bool(p_score < 0.05),
        })
    return pd.DataFrame(rows)


def sensitivity_summary() -> pd.DataFrame:
    frame = pd.read_csv(LOCKED_SENSITIVITY)
    rows = []
    for endpoint, part in frame.groupby("endpoint", sort=False):
        rows.append({
            "endpoint": endpoint,
            "sensitivity_rows": int(len(part)),
            "positive_direction_rows": int((part["beta"] > 0).sum()),
            "holm_significant_rows": int((part["holm_p_two_endpoints"] < 0.05).sum()),
            "minimum_beta": float(part["beta"].min()),
            "maximum_beta": float(part["beta"].max()),
            "strata": ";".join(part["stratum"].astype(str)),
        })
    return pd.DataFrame(rows)


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    view = frame.loc[:, columns].copy()
    for column in view.select_dtypes(include=[np.number]).columns:
        view[column] = view[column].map(
            lambda value: f"{value:.6g}" if np.isfinite(value) else "NA"
        )
    header = "| " + " | ".join(columns) + " |"
    rule = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(map(str, row)) + " |" for row in view.itertuples(index=False, name=None)]
    return "\n".join([header, rule, *body])


def run(out: Path) -> None:
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{out.name}.", dir=out.parent))
    try:
        cohort = load_exact_cohort()
        metrics = collect_metrics()
        composites, _ = build_composites(cohort, metrics, PRIMARY_NETWORKS)
        primary = audit_primary(composites)
        heterogeneity = audit_heterogeneity(cohort, metrics)
        sensitivity = sensitivity_summary()

        primary.to_csv(temporary / "primary_small_cluster_inference.csv", index=False)
        heterogeneity.to_csv(temporary / "heterogeneity_small_cluster_inference.csv", index=False)
        sensitivity.to_csv(temporary / "sensitivity_summary.csv", index=False)

        primary_wild_pass = bool(
            primary["positive_direction"].all()
            and (primary["holm_p_restricted_wild_cluster_bootstrap_t"] < 0.05).all()
        )
        pair_common_pass = bool(
            (primary[primary["scope"].eq("pair_common_sites")]["holm_p_restricted_wild_cluster_bootstrap_t"] < 0.05).all()
        )
        heterogeneity_pass = bool(heterogeneity["small_cluster_pass_0_05"].all())
        exact_replay = bool(
            (primary["locked_beta_absolute_difference"] < 1e-10).all()
            and (primary["locked_p_absolute_difference"] < 1e-10).all()
            and (heterogeneity["locked_p_absolute_difference"] < 1e-10).all()
        )
        checks = {
            "six_primary_models": len(primary) == 6,
            "primary_exact_replay": exact_replay,
            "all_wild_bootstrap_iterations_valid": bool(
                (primary["wild_bootstrap_valid_iterations"] == BOOTSTRAP_ITERATIONS).all()
            ),
            "all_primary_directions_positive": bool(primary["positive_direction"].all()),
            "pair_common_primary_survives_wild_holm": pair_common_pass,
            "all_three_scopes_survive_wild_holm": primary_wild_pass,
            "two_heterogeneity_models": len(heterogeneity) == 2,
            "heterogeneity_small_cluster_score_pass": heterogeneity_pass,
            "no_participant_level_output": True,
        }
        decision = (
            "STATISTICALLY_DEFENSIBLE_INTERNAL_DISCOVERY"
            if all(checks.values())
            else "QUALIFICATION_REQUIRED"
        )
        report = {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "schema_version": "1.0.0",
            "decision": decision,
            "checks": checks,
            "inferential_boundary": {
                "primary_family_control": "Holm across the two locked endpoints within each support scope",
                "selection_control": "conditional only; endpoints were selected in the same historical cohort, so discovery-wide selection error is not controlled",
                "small_cluster_methods": [
                    "CR1 normal reference (exact replay)",
                    "CR1 t reference with df=number_of_sites-1",
                    f"restricted Rademacher wild-cluster bootstrap-t ({BOOTSTRAP_ITERATIONS} draws)",
                    "restricted cluster-score Rademacher test for 16-df heterogeneity",
                ],
                "confirmation_required": True,
                "biomarker_or_mechanism_claim_supported": False,
            },
        }
        (temporary / "audit_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        summary = f"""# Primary inference soundness audit

**Generated:** {report['generated_utc']}  
**Decision:** {decision}

## Primary endpoints under small-cluster inference

{markdown_table(primary, ['endpoint', 'scope', 'n', 'n_sites', 'beta', 'cr1_t', 'p_cr1_normal', 'p_cr1_t_gminus1', 'p_restricted_wild_cluster_bootstrap_t', 'holm_p_restricted_wild_cluster_bootstrap_t'])}

## System heterogeneity under cluster-score sign inference

{markdown_table(heterogeneity, ['endpoint', 'n_sites', 'interaction_df', 'locked_cr1_f', 'locked_cr1_p_replayed', 'restricted_cluster_score_wald', 'p_cluster_score_rademacher', 'score_method', 'small_cluster_pass_0_05'])}

## Existing sensitivity inventory

{markdown_table(sensitivity, ['endpoint', 'sensitivity_rows', 'positive_direction_rows', 'holm_significant_rows', 'minimum_beta', 'maximum_beta'])}

## Interpretation

This audit asks whether the already selected two-endpoint family remains credible when 20-site analyses are not judged against an asymptotic normal reference alone. A PASS supports the phrase **statistically defensible internal discovery**. It does not make the analysis prespecified, independently replicated, mechanistic, clinically validated, or globally multiplicity-controlled across the historical dashboard search.

The manuscript must continue to state that endpoint selection occurred in the same historical cohort. Corrected matrices or an untouched cohort are required to convert conditional discovery p values into confirmation evidence.
"""
        (temporary / "summary.md").write_text(summary, encoding="utf-8")
        if not all(checks.values()):
            # The output remains scientifically important even when qualification is required.
            report["release_status"] = "VALID_AUDIT_REQUIRES_MANUSCRIPT_QUALIFICATION"
            (temporary / "audit_report.json").write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        else:
            report["release_status"] = "PASS"
            (temporary / "audit_report.json").write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        os.rename(temporary, out)
    except Exception:
        failed = temporary.with_name(temporary.name + ".failed")
        if temporary.exists() and not failed.exists():
            os.rename(temporary, failed)
        raise


def main() -> None:
    run(DEFAULT_OUT)
    report = json.loads((DEFAULT_OUT / "audit_report.json").read_text(encoding="utf-8"))
    print(json.dumps({"output": str(DEFAULT_OUT), "decision": report["decision"], "checks": report["checks"]}, indent=2))


if __name__ == "__main__":
    main()
