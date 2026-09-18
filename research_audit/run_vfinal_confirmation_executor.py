#!/usr/bin/env python3
"""Fail-closed executor for the frozen corrected-processing confirmation.

The script intentionally does not create tractography, matrices, network
features, or an authorization record.  It consumes an outcome-blindly frozen
feature bundle only after a separate technical/QC gate has authorized the
locked analysis.  Participant-level values remain in memory; all written
artifacts are aggregate only.

Required cohort columns
-----------------------
subject_id, group, age, sex, site, gap_days, t1_source, qc_pass,
exclusion_reason

Required feature columns
------------------------
subject_id, system_id, ad_mean, rd_mean

Required authorization JSON fields
----------------------------------
contract_id, status (AUTHORIZED_FOR_LOCKED_ANALYSIS), qc_exclusions_frozen
(true), outcome_blind_freeze (true), cohort_sha256, features_sha256,
corrected_matrix_manifest_sha256, phase_b_terminal_record,
technical_gate_passed (true), uniform_recipe (true), atlas_nodes (166),
assignment (uniform direct radial-search radius 4 mm)

Preflight mode validates the locked schema and support without estimating a
diagnosis effect. Execute mode additionally verifies the authorization hashes,
fits the two pair-common-site models, runs the prespecified restricted
Rademacher wild-cluster bootstrap-t, applies Holm correction across exactly the
two endpoints, and emits the frozen classification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests


ROOT = Path("/home/ec2-user/exp/research_audit")
DEFAULT_CONTRACT = ROOT / "outputs" / "vfinal_confirmation_contract_v1" / "confirmation_contract.json"
EXPECTED_CONTRACT_ID = "SC-VFINAL-CONFIRM-20260719-V1"
EXPECTED_SYSTEMS = (
    "anatomical::BasalGanglia",
    "anatomical::Brainstem",
    "anatomical::Cerebellum",
    "anatomical::Frontal",
    "anatomical::Limbic",
    "anatomical::Occipital",
    "anatomical::Parietal",
    "anatomical::Temporal",
    "anatomical::Thalamus",
    "functional::DMN",
    "functional::DorsalAttention",
    "functional::Frontoparietal",
    "functional::Limbic",
    "functional::Salience_VAN",
    "functional::Somatomotor",
    "functional::Subcortical",
    "functional::Visual",
)
COHORT_COLUMNS = (
    "subject_id",
    "group",
    "age",
    "sex",
    "site",
    "gap_days",
    "t1_source",
    "qc_pass",
    "exclusion_reason",
)
FEATURE_COLUMNS = ("subject_id", "system_id", "ad_mean", "rd_mean")
AUTH_STATUS = "AUTHORIZED_FOR_LOCKED_ANALYSIS"
ASSIGNMENT = "uniform direct radial-search radius 4 mm"


@dataclass(frozen=True)
class Endpoint:
    endpoint_id: str
    outcome: str
    left: str
    right: str


ENDPOINTS = (
    Endpoint("AxD_MCI_minus_CN", "axd_burden", "MCI", "CN"),
    Endpoint("RD_AD_minus_MCI", "rd_burden", "AD", "MCI"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    except Exception:
        try:
            os.unlink(name)
        except FileNotFoundError:
            pass
        raise


def normal_scores(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    out = pd.Series(np.nan, index=values.index, dtype=float)
    keep = numeric.notna() & np.isfinite(numeric)
    if not keep.any():
        return out
    ranks = numeric.loc[keep].rank(method="average")
    probability = (ranks - 0.5) / len(ranks)
    out.loc[keep] = st.norm.ppf(probability)
    return out


def standardize(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    scale = float(numeric.std(ddof=0))
    if not np.isfinite(scale) or scale == 0:
        return pd.Series(0.0, index=values.index)
    return (numeric - float(numeric.mean())) / scale


def parse_bool(values: pd.Series, field: str) -> pd.Series:
    mapping = {
        True: True,
        False: False,
        1: True,
        0: False,
        "1": True,
        "0": False,
        "true": True,
        "false": False,
        "yes": True,
        "no": False,
    }
    parsed = values.map(lambda value: mapping.get(value if isinstance(value, bool) else str(value).strip().lower()))
    if parsed.isna().any():
        bad = sorted(values.loc[parsed.isna()].astype(str).unique())
        raise ValueError(f"{field} contains non-boolean values: {bad}")
    return parsed.astype(bool)


def load_contract(path: Path) -> dict[str, object]:
    contract = json.loads(path.read_text(encoding="utf-8"))
    if contract.get("contract_id") != EXPECTED_CONTRACT_ID:
        raise ValueError(f"Unexpected contract_id: {contract.get('contract_id')!r}")
    if contract.get("status") != "FROZEN_BEFORE_CORRECTED_CONNECTOME_RESULTS":
        raise ValueError("Confirmation contract is not in the frozen pre-results state")
    if tuple(contract.get("primary_system_inventory", ())) != EXPECTED_SYSTEMS:
        raise ValueError("Contract system inventory differs from the frozen 17-system inventory")
    model = contract.get("primary_support_and_model", {})
    expected = {
        "minimum_sites_per_endpoint_for_confirmatory_classification": 15,
        "minimum_participants_per_contrast_group": 50,
        "bootstrap_iterations": 9999,
        "bootstrap_seed": 20260719,
        "alpha_two_sided": 0.05,
    }
    for key, value in expected.items():
        if model.get(key) != value:
            raise ValueError(f"Contract field {key!r} changed: {model.get(key)!r}")
    return contract


def load_inputs(cohort_path: Path, feature_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    cohort = pd.read_csv(cohort_path, dtype={"subject_id": str, "site": str})
    features = pd.read_csv(feature_path, dtype={"subject_id": str, "system_id": str})
    missing_cohort = sorted(set(COHORT_COLUMNS) - set(cohort.columns))
    missing_features = sorted(set(FEATURE_COLUMNS) - set(features.columns))
    if missing_cohort:
        raise ValueError(f"Cohort input missing columns: {missing_cohort}")
    if missing_features:
        raise ValueError(f"Feature input missing columns: {missing_features}")
    cohort = cohort.loc[:, COHORT_COLUMNS].copy()
    features = features.loc[:, FEATURE_COLUMNS].copy()
    cohort["subject_id"] = cohort["subject_id"].astype(str)
    features["subject_id"] = features["subject_id"].astype(str)
    features["system_id"] = features["system_id"].astype(str)
    cohort["site"] = cohort["site"].astype(str)
    if cohort["subject_id"].duplicated().any():
        raise ValueError("Cohort input contains duplicate subject_id rows")
    if features.duplicated(["subject_id", "system_id"]).any():
        raise ValueError("Feature input contains duplicate subject_id/system_id rows")
    cohort["qc_pass"] = parse_bool(cohort["qc_pass"], "qc_pass")
    cohort["group"] = cohort["group"].astype(str).str.upper()
    invalid_groups = sorted(set(cohort["group"]) - {"CN", "MCI", "AD", "SMC", "UNKNOWN"})
    if invalid_groups:
        raise ValueError(f"Unrecognized diagnosis groups: {invalid_groups}")
    for column in ("age", "gap_days"):
        cohort[column] = pd.to_numeric(cohort[column], errors="coerce")
    for column in ("ad_mean", "rd_mean"):
        features[column] = pd.to_numeric(features[column], errors="coerce")
        finite = features[column].dropna()
        if not np.isfinite(finite).all() or (finite <= 0).any() or (finite >= 0.01).any():
            raise ValueError(f"{column} contains nonfinite or physically implausible values")
    unknown_systems = sorted(set(features["system_id"]) - set(EXPECTED_SYSTEMS))
    if unknown_systems:
        raise ValueError(f"Feature bundle contains systems outside the frozen inventory: {unknown_systems}")
    eligible = cohort[cohort["qc_pass"] & cohort["group"].isin(["CN", "MCI", "AD"])].copy()
    if eligible.empty:
        raise ValueError("No QC-passed CN/MCI/AD participants")
    if eligible[["age", "sex", "site", "gap_days", "t1_source"]].isna().any().any():
        raise ValueError("QC-passed participants have missing locked covariates")
    feature_ids = set(features["subject_id"])
    unknown_ids = sorted(feature_ids - set(cohort["subject_id"]))
    if unknown_ids:
        raise ValueError(f"Feature bundle contains {len(unknown_ids)} subject IDs absent from cohort")
    excluded_with_features = set(cohort.loc[~cohort["qc_pass"], "subject_id"]) & feature_ids
    if excluded_with_features:
        raise ValueError("QC-excluded participants remain in the frozen feature bundle")
    features = features[features["subject_id"].isin(eligible["subject_id"])].copy()
    observed = set(features["system_id"])
    missing_inventory = sorted(set(EXPECTED_SYSTEMS) - observed)
    if missing_inventory:
        raise ValueError(f"Frozen feature bundle has no values for systems: {missing_inventory}")
    metadata = {
        "cohort_rows": int(len(cohort)),
        "qc_passed_cn_mci_ad": int(len(eligible)),
        "feature_rows": int(len(features)),
        "excluded_rows": int((~cohort["qc_pass"]).sum()),
        "group_counts": {str(k): int(v) for k, v in eligible["group"].value_counts().sort_index().items()},
        "site_count": int(eligible["site"].nunique()),
    }
    return eligible, features, metadata


def build_composites(cohort: pd.DataFrame, features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    long = features.melt(
        id_vars=["subject_id", "system_id"],
        value_vars=["ad_mean", "rd_mean"],
        var_name="component",
        value_name="value",
    )
    long["feature"] = long["system_id"] + "::" + long["component"]
    long["feature_z"] = long.groupby("feature", sort=False)["value"].transform(normal_scores)
    wide = long.pivot(index="subject_id", columns=["system_id", "component"], values="feature_z")
    rows: list[dict[str, object]] = []
    for subject_id, values in wide.iterrows():
        paired = [
            system
            for system in EXPECTED_SYSTEMS
            if np.isfinite(values.get((system, "ad_mean"), np.nan))
            and np.isfinite(values.get((system, "rd_mean"), np.nan))
        ]
        if not paired:
            continue
        rows.append(
            {
                "subject_id": str(subject_id),
                "n_paired_systems": len(paired),
                "axd_burden": float(np.mean([values[(system, "ad_mean")] for system in paired])),
                "rd_burden": float(np.mean([values[(system, "rd_mean")] for system in paired])),
            }
        )
    composite = cohort.merge(pd.DataFrame(rows), on="subject_id", how="inner", validate="1:1")
    coverage_rows: list[dict[str, object]] = []
    availability = features.assign(
        paired=lambda frame: frame["ad_mean"].notna() & frame["rd_mean"].notna()
    )
    availability = availability.merge(cohort[["subject_id", "group"]], on="subject_id", how="inner", validate="m:1")
    eligible_counts = cohort["group"].value_counts()
    for system in EXPECTED_SYSTEMS:
        part = availability[availability["system_id"].eq(system)]
        for group in ("CN", "MCI", "AD"):
            available = int(part.loc[part["group"].eq(group), "paired"].sum())
            total = int(eligible_counts.get(group, 0))
            coverage_rows.append(
                {
                    "system_id": system,
                    "group": group,
                    "available": available,
                    "eligible": total,
                    "available_pct": 100.0 * available / total if total else math.nan,
                }
            )
    return composite, pd.DataFrame(coverage_rows)


def contrast_vector(names: Sequence[str], left: str, right: str) -> np.ndarray:
    reference = right
    coefficient = f'C(group, Treatment(reference="{reference}"))[T.{left}]'
    if coefficient not in names:
        raise ValueError(f"Required diagnosis coefficient absent: {coefficient}")
    vector = np.zeros(len(names), dtype=float)
    vector[list(names).index(coefficient)] = 1.0
    return vector


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
    unique, codes = np.unique(clusters.astype(str), return_inverse=True)
    g = len(unique)
    n, p = x.shape
    bread = np.linalg.pinv(x.T @ x)
    beta = bread @ x.T @ y
    r2 = r.reshape(1, -1)
    middle = r2 @ bread @ r2.T
    beta_restricted = beta - (bread @ r2.T @ np.linalg.pinv(middle) @ (r2 @ beta)).reshape(-1)
    fitted_restricted = x @ beta_restricted
    residual_restricted = y - fitted_restricted
    projection = (r2 @ bread).reshape(-1)
    scale = (g / (g - 1.0)) * ((n - 1.0) / (n - p))
    rng = np.random.default_rng(seed)
    exceed = 0
    valid = 0
    for start in range(0, iterations, chunk_size):
        count = min(chunk_size, iterations - start)
        weights = rng.choice(np.array([-1.0, 1.0]), size=(g, count), replace=True)
        y_star = fitted_restricted[:, None] + residual_restricted[:, None] * weights[codes]
        beta_star = bread @ (x.T @ y_star)
        residual_star = y_star - x @ beta_star
        variance = np.zeros(count, dtype=float)
        for cluster in range(g):
            mask = codes == cluster
            score = x[mask].T @ residual_star[mask]
            projected = projection @ score
            variance += projected * projected
        se_star = np.sqrt(scale * variance)
        effect_star = (r2 @ beta_star).reshape(-1)
        finite = np.isfinite(se_star) & (se_star > 0) & np.isfinite(effect_star)
        t_star = effect_star[finite] / se_star[finite]
        exceed += int(np.sum(np.abs(t_star) >= abs(observed_t)))
        valid += int(finite.sum())
    return (1.0 + exceed) / (1.0 + valid), valid


def endpoint_support(composite: pd.DataFrame, endpoint: Endpoint) -> pd.DataFrame:
    use = composite[composite["group"].isin([endpoint.left, endpoint.right])].copy()
    site_groups = use.groupby("site")["group"].nunique()
    common_sites = site_groups[site_groups.eq(2)].index
    return use[use["site"].isin(common_sites)].copy()


def fit_endpoint(composite: pd.DataFrame, endpoint: Endpoint, contract: dict[str, object]) -> dict[str, object]:
    use = endpoint_support(composite, endpoint)
    required = [endpoint.outcome, "group", "age", "sex", "site", "gap_days", "t1_source"]
    use = use.dropna(subset=required).copy()
    counts = use["group"].value_counts()
    n_sites = int(use["site"].nunique())
    model_spec = contract["primary_support_and_model"]
    if n_sites < int(model_spec["minimum_sites_per_endpoint_for_confirmatory_classification"]):
        raise ValueError(f"{endpoint.endpoint_id}: only {n_sites} pair-common sites")
    for group in (endpoint.left, endpoint.right):
        if int(counts.get(group, 0)) < int(model_spec["minimum_participants_per_contrast_group"]):
            raise ValueError(f"{endpoint.endpoint_id}: only {int(counts.get(group, 0))} participants in {group}")
    use["outcome_z"] = normal_scores(use[endpoint.outcome])
    use["age_z"] = standardize(use["age"])
    use["gap_z"] = standardize(np.log1p(use["gap_days"].clip(lower=0)))
    terms = [f'C(group, Treatment(reference="{endpoint.right}"))', "age_z"]
    if use["sex"].nunique() > 1:
        terms.append("C(sex)")
    if use["gap_z"].nunique() > 1:
        terms.append("gap_z")
    if use["t1_source"].nunique() > 1:
        terms.append("C(t1_source)")
    terms.append("C(site)")
    formula = "outcome_z ~ " + " + ".join(terms)
    base = smf.ols(formula, data=use).fit()
    if np.linalg.matrix_rank(base.model.exog) != base.model.exog.shape[1]:
        raise ValueError(f"{endpoint.endpoint_id}: rank-deficient design matrix")
    robust = base.get_robustcov_results(
        cov_type="cluster",
        groups=use["site"].astype(str).to_numpy(),
        use_correction=True,
        use_t=False,
    )
    r = contrast_vector(base.model.exog_names, endpoint.left, endpoint.right)
    test = robust.t_test(r)
    beta = float(np.asarray(test.effect).reshape(-1)[0])
    se = float(np.asarray(test.sd).reshape(-1)[0])
    t_value = beta / se
    df = n_sites - 1
    critical = float(st.t.ppf(0.975, df=df))
    raw_p = float(2.0 * st.t.sf(abs(t_value), df=df))
    wild_p, valid = restricted_wild_cluster_bootstrap_t(
        base.model.endog,
        base.model.exog,
        r,
        use["site"].astype(str).to_numpy(),
        t_value,
        iterations=int(model_spec["bootstrap_iterations"]),
        seed=int(model_spec["bootstrap_seed"]),
    )
    return {
        "endpoint": endpoint.endpoint_id,
        "contrast": f"{endpoint.left} - {endpoint.right}",
        "support_scope": "pair-common sites",
        "n": int(len(use)),
        f"n_{endpoint.left}": int(counts.get(endpoint.left, 0)),
        f"n_{endpoint.right}": int(counts.get(endpoint.right, 0)),
        "n_sites": n_sites,
        "beta": beta,
        "cr1_se": se,
        "ci95_low": beta - critical * se,
        "ci95_high": beta + critical * se,
        "cr1_t": t_value,
        "cr1_t_gminus1_p": raw_p,
        "restricted_wild_cluster_bootstrap_t_p": wild_p,
        "wild_bootstrap_valid_iterations": valid,
        "positive_direction": bool(beta > 0),
        "formula": formula,
        "design_rank": int(np.linalg.matrix_rank(base.model.exog)),
        "design_columns": int(base.model.exog.shape[1]),
        "condition_number": float(np.linalg.cond(base.model.exog)),
    }


def verify_authorization(path: Path, cohort_path: Path, feature_path: Path, contract: dict[str, object]) -> dict[str, object]:
    auth = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "contract_id": EXPECTED_CONTRACT_ID,
        "status": AUTH_STATUS,
        "qc_exclusions_frozen": True,
        "outcome_blind_freeze": True,
        "technical_gate_passed": True,
        "uniform_recipe": True,
        "atlas_nodes": 166,
        "assignment": ASSIGNMENT,
    }
    for key, expected in required.items():
        if auth.get(key) != expected:
            raise PermissionError(f"Authorization field {key!r} must equal {expected!r}")
    if auth.get("cohort_sha256") != sha256(cohort_path):
        raise PermissionError("Authorization cohort_sha256 does not match the input")
    if auth.get("features_sha256") != sha256(feature_path):
        raise PermissionError("Authorization features_sha256 does not match the input")
    for key in ("corrected_matrix_manifest_sha256", "phase_b_terminal_record"):
        if not isinstance(auth.get(key), str) or not auth[key].strip():
            raise PermissionError(f"Authorization field {key!r} is required")
    if contract.get("execution_authorized") is not False:
        raise ValueError("Frozen contract provenance changed unexpectedly")
    return auth


def classify(results: pd.DataFrame) -> str:
    if not bool(results["positive_direction"].all()):
        return "NOT_REPLICATED_UNDER_CORRECTED_PROCESSING"
    passed = results["positive_direction"] & results["holm_wild_p"].le(0.05)
    if bool(passed.all()):
        return "PROCESSING_CONFIRMED_INTERNAL"
    if bool(results["positive_direction"].all()) and not bool(passed.any()):
        return "DIRECTIONAL_ONLY_NOT_CONFIRMED"
    if bool(passed.any()):
        return "PARTIAL_REPLICATION_QUALIFY_CLAIM"
    return "DIRECTIONAL_ONLY_NOT_CONFIRMED"


def render_markdown(results: pd.DataFrame, classification: str, metadata: dict[str, object]) -> str:
    lines = [
        "# V-final corrected-processing confirmation",
        "",
        f"**Generated:** {datetime.now(timezone.utc).isoformat()}",
        f"**Frozen classification:** `{classification}`",
        "",
        "This is same-cohort processing confirmation, not independent or external replication.",
        "",
        "| Endpoint | N | Sites | Beta | 95% CI | CR1 t(G-1) p | Wild p | Holm wild p |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in results.itertuples(index=False):
        lines.append(
            f"| {row.endpoint} | {row.n} | {row.n_sites} | {row.beta:.4f} | "
            f"{row.ci95_low:.4f} to {row.ci95_high:.4f} | {row.cr1_t_gminus1_p:.4g} | "
            f"{row.restricted_wild_cluster_bootstrap_t_p:.4g} | {row.holm_wild_p:.4g} |"
        )
    lines.extend(
        [
            "",
            "## Input/QC summary",
            "",
            f"- QC-passed CN/MCI/AD participants: {metadata['qc_passed_cn_mci_ad']}",
            f"- Sites: {metadata['site_count']}",
            f"- Group counts: {json.dumps(metadata['group_counts'], sort_keys=True)}",
            f"- Aggregate technical exclusions: {metadata['excluded_rows']}",
            "",
            "The locked primary family contains exactly two endpoints. Graph, edge, network-selectivity, machine-learning and mechanistic analyses cannot change this classification.",
            "",
        ]
    )
    return "\n".join(lines)


def preflight_report(contract_path: Path, cohort_path: Path, feature_path: Path, metadata: dict[str, object], composite: pd.DataFrame) -> dict[str, object]:
    support = {}
    minimum_sites = 15
    minimum_group_n = 50
    for endpoint in ENDPOINTS:
        use = endpoint_support(composite, endpoint)
        counts = use["group"].value_counts()
        n_sites = int(use["site"].nunique())
        if n_sites < minimum_sites:
            raise ValueError(f"{endpoint.endpoint_id}: preflight has only {n_sites} pair-common sites")
        for group in (endpoint.left, endpoint.right):
            if int(counts.get(group, 0)) < minimum_group_n:
                raise ValueError(
                    f"{endpoint.endpoint_id}: preflight has only {int(counts.get(group, 0))} participants in {group}"
                )
        support[endpoint.endpoint_id] = {
            "n": int(len(use)),
            "n_sites": n_sites,
            "group_counts": {str(k): int(v) for k, v in counts.sort_index().items()},
        }
    return {
        "status": "PASS_PREFLIGHT_NO_EFFECT_ESTIMATION",
        "contract": str(contract_path),
        "contract_sha256": sha256(contract_path),
        "cohort": str(cohort_path),
        "cohort_sha256": sha256(cohort_path),
        "features": str(feature_path),
        "features_sha256": sha256(feature_path),
        "metadata": metadata,
        "pair_common_support": support,
        "execution_performed": False,
    }


def run(args: argparse.Namespace) -> None:
    contract_path = Path(args.contract).resolve()
    cohort_path = Path(args.cohort).resolve()
    feature_path = Path(args.features).resolve()
    out = Path(args.out).resolve()
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite existing release: {out}")
    contract = load_contract(contract_path)
    cohort, features, metadata = load_inputs(cohort_path, feature_path)
    composite, coverage = build_composites(cohort, features)
    if args.mode == "preflight":
        report = preflight_report(contract_path, cohort_path, feature_path, metadata, composite)
        out.mkdir(parents=True, exist_ok=False)
        atomic_text(out / "preflight.json", json.dumps(report, indent=2, sort_keys=True) + "\n")
        return
    if not args.authorization:
        raise PermissionError("Execute mode requires --authorization")
    authorization_path = Path(args.authorization).resolve()
    authorization = verify_authorization(authorization_path, cohort_path, feature_path, contract)
    rows = [fit_endpoint(composite, endpoint, contract) for endpoint in ENDPOINTS]
    results = pd.DataFrame(rows)
    results["holm_wild_p"] = multipletests(
        results["restricted_wild_cluster_bootstrap_t_p"].to_numpy(dtype=float),
        method="holm",
    )[1]
    classification = classify(results)
    out.mkdir(parents=True, exist_ok=False)
    results.to_csv(out / "primary_confirmation_results.csv", index=False)
    coverage.to_csv(out / "system_coverage_aggregate.csv", index=False)
    summary = {
        "status": "PASS_LOCKED_EXECUTION",
        "contract_id": EXPECTED_CONTRACT_ID,
        "contract_sha256": sha256(contract_path),
        "authorization": str(authorization_path),
        "authorization_sha256": sha256(authorization_path),
        "input_hashes": {
            "cohort_sha256": sha256(cohort_path),
            "features_sha256": sha256(feature_path),
        },
        "authorization_evidence": authorization,
        "classification": classification,
        "same_cohort_processing_confirmation_not_independent_replication": True,
        "metadata": metadata,
    }
    atomic_text(out / "confirmation_summary.json", json.dumps(summary, indent=2, sort_keys=True) + "\n")
    atomic_text(out / "confirmation_summary.md", render_markdown(results, classification, metadata))


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--contract", default=str(DEFAULT_CONTRACT))
    command.add_argument("--cohort", required=True)
    command.add_argument("--features", required=True)
    command.add_argument("--out", required=True)
    command.add_argument("--mode", choices=("preflight", "execute"), default="preflight")
    command.add_argument("--authorization")
    return command


if __name__ == "__main__":
    run(parser().parse_args())
