#!/usr/bin/env python3
"""SL-P0-10 fixed-data feasibility audit for the approved local cohort.

This program reads only the locked tabular pair manifest, its validation file,
the diagnosis reconciliation, and the SL-H01 decision.  It does not open image
paths, connectomes, or biological outcome tables.  The outputs describe cohort
composition, observed acquisition overlap, fixed-design numerical feasibility,
hypothetical attrition stress tests, and standardized detectable-effect bounds.

The power quantities are planning diagnostics, not biological results.  They
assume independent homoscedastic Gaussian residuals and are optimistic when
site clustering, QC attrition, or model uncertainty is material.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


PROJECT = Path("/home/ec2-user/exp")
AUDIT = PROJECT / "research_audit"
DEFAULT_OUTPUT = AUDIT / "outputs"
DEFAULT_FIGURES = AUDIT / "figures"
DEFAULT_MANIFEST = DEFAULT_OUTPUT / "available_data_pair_manifest_v2.csv"
DEFAULT_MANIFEST_VALIDATION = (
    DEFAULT_OUTPUT / "available_data_pair_manifest_validation_v2.json"
)
DEFAULT_DIAGNOSIS = DEFAULT_OUTPUT / "diagnosis_reconciliation_v2.csv"
DEFAULT_DECISION = AUDIT / "decisions" / "sl_h01_available_data_policy_20260718.json"
DEFAULT_WORKFLOW_CONFIG = PROJECT / "configs" / "connectome_v2.yaml"
DEFAULT_HISTORICAL_PROXY_VALIDATION = (
    DEFAULT_OUTPUT / "historical_source_preflight_proxy_validation_v1.json"
)

GROUPS = ("CN", "MCI", "AD")
ALL_GROUPS = ("CN", "MCI", "AD", "SMC")
CONTRASTS = (("CN", "MCI"), ("MCI", "AD"), ("CN", "AD"))
RANDOM_SEED = 20260718
SIMULATIONS = 10_000
ALPHA_FAMILY = 0.05
N_PRIMARY_CONTRASTS = 3
TARGET_POWER = 0.80

OUTPUT_FEASIBILITY = "cohort_v2_feasibility.csv"
OUTPUT_POPULATIONS = "cohort_v2_population_summary.csv"
OUTPUT_SUPPORT = "cohort_v2_common_support.csv"
OUTPUT_DESIGN = "cohort_v2_design_diagnostics.csv"
OUTPUT_ESTIMABILITY = "cohort_v2_estimability.csv"
OUTPUT_POWER = "cohort_v2_power_attrition.csv"
OUTPUT_WORKFLOW_CONTRACT = "cohort_v2_workflow_contract_feasibility.csv"
OUTPUT_VALIDATION = "cohort_v2_feasibility_validation.json"
OUTPUT_MEMO = "fixed_data_feasibility_power_memo_v1.md"
FIG_SUPPORT = "cohort_v2_common_support.png"
FIG_POWER = "cohort_v2_mde_attrition.png"


@dataclass(frozen=True)
class Source:
    path: Path
    frame: pd.DataFrame
    sha256: str
    size_bytes: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_bool(value: Any, label: str) -> bool:
    token = str(value).strip().lower()
    if token == "true":
        return True
    if token == "false":
        return False
    raise ValueError(f"{label} must be true/false, found {value!r}")


def read_csv_source(path: Path, label: str) -> Source:
    if path.is_symlink():
        raise ValueError(f"Refusing symlinked {label}: {path}")
    data = path.read_bytes()
    text = data.decode("utf-8-sig")
    reader = csv.reader(text.splitlines())
    try:
        header = next(reader)
    except StopIteration as exc:
        raise ValueError(f"{label} is empty") from exc
    header = [item.strip() for item in header]
    if any(not item for item in header) or len(header) != len(set(header)):
        raise ValueError(f"{label} has blank or duplicate headers")
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    if list(frame.columns) != header:
        raise ValueError(f"{label} header changed during parsing")
    return Source(path, frame, hashlib.sha256(data).hexdigest(), len(data))


def require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}")


def load_json(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink():
        raise ValueError(f"Refusing symlinked {label}: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return value


def validate_inputs(
    manifest_path: Path,
    manifest_validation_path: Path,
    diagnosis_path: Path,
    decision_path: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    manifest = read_csv_source(manifest_path, "available-data manifest")
    diagnosis = read_csv_source(diagnosis_path, "diagnosis reconciliation")
    validation = load_json(manifest_validation_path, "manifest validation")
    decision = load_json(decision_path, "SL-H01 decision")

    if validation.get("status") != "PASS":
        raise ValueError("available-data manifest validation is not PASS")
    bound_manifest = validation.get("outputs", {}).get("pair_manifest", {})
    if bound_manifest.get("sha256") != manifest.sha256:
        raise ValueError("manifest checksum does not match its validation")
    if int(bound_manifest.get("row_count", -1)) != len(manifest.frame):
        raise ValueError("manifest row count does not match its validation")
    attachment = validation.get("diagnosis_attachment", {})
    if attachment.get("sha256") != diagnosis.sha256:
        raise ValueError("diagnosis checksum does not match manifest attachment")
    if decision.get("gate") != "SL-H01" or decision.get("decision") != "APPROVED":
        raise ValueError("SL-H01 is not approved")
    if "SL-P0-10_FIXED_DATA_FEASIBILITY" not in decision.get("authorizes", []):
        raise ValueError("SL-H01 does not authorize SL-P0-10")

    required_manifest = {
        "subject_id",
        "dti_image_id",
        "current_t1_image_id",
        "diagnosis_reconciled_harmonized",
        "diagnosis_reconciled_primary_analysis_eligible",
        "diagnosis_reconciled_smc_retained_separately",
        "dti_t1_gap_days_abs",
        "timing_stratum",
        "dti_phase",
        "dti_age_at_scan",
        "dti_sex",
        "dti_manufacturer",
        "dti_field_strength_t",
        "dti_gradient_directions",
        "dti_protocol_key",
        "site",
        "current_t1_type",
        "current_t1_is_original",
        "same_adni_phase",
        "corrected_rerun_qc_status",
        "available_for_corrected_rerun",
        "diagnosis_reconciliation_source_sha256",
    }
    required_diagnosis = {
        "subject_id",
        "selected_dti_image_id",
        "diagnosis_harmonized",
        "primary_analysis_eligible",
        "smc_retained_separately",
    }
    require_columns(manifest.frame, required_manifest, "available-data manifest")
    require_columns(diagnosis.frame, required_diagnosis, "diagnosis reconciliation")

    frame = manifest.frame.copy()
    if len(frame) != int(decision.get("processing_denominator", -1)):
        raise ValueError("manifest denominator differs from approved SL-H01 denominator")
    for column in ("subject_id", "dti_image_id", "current_t1_image_id"):
        if frame[column].duplicated().any():
            raise ValueError(f"manifest contains duplicate {column}")
    if set(frame["diagnosis_reconciliation_source_sha256"]) != {diagnosis.sha256}:
        raise ValueError("manifest rows are not uniformly bound to diagnosis checksum")

    diag_keyed = diagnosis.frame.set_index(
        ["subject_id", "selected_dti_image_id"], drop=False
    )
    if not diag_keyed.index.is_unique:
        raise ValueError("diagnosis reconciliation key is not unique")
    for row in frame.itertuples(index=False):
        key = (row.subject_id, row.dti_image_id)
        if key not in diag_keyed.index:
            raise ValueError(f"diagnosis key missing for {key}")
        drow = diag_keyed.loc[key]
        if row.diagnosis_reconciled_harmonized != drow["diagnosis_harmonized"]:
            raise ValueError(f"diagnosis mismatch for {key}")
        if strict_bool(
            row.diagnosis_reconciled_primary_analysis_eligible,
            f"manifest primary eligibility {key}",
        ) != strict_bool(drow["primary_analysis_eligible"], f"diagnosis eligibility {key}"):
            raise ValueError(f"diagnosis eligibility mismatch for {key}")

    frame["group"] = frame["diagnosis_reconciled_harmonized"].str.strip()
    unknown = sorted(set(frame["group"]) - set(ALL_GROUPS))
    if unknown:
        raise ValueError(f"unexpected diagnosis groups: {unknown}")
    frame["primary_eligible"] = frame[
        "diagnosis_reconciled_primary_analysis_eligible"
    ].map(lambda value: strict_bool(value, "primary eligibility"))
    frame["smc_separate"] = frame[
        "diagnosis_reconciled_smc_retained_separately"
    ].map(lambda value: strict_bool(value, "SMC flag"))
    frame["same_phase"] = frame["same_adni_phase"].map(
        lambda value: strict_bool(value, "same-phase flag")
    )
    frame["t1_original"] = frame["current_t1_is_original"].map(
        lambda value: strict_bool(value, "T1 Original flag")
    )
    frame["available"] = frame["available_for_corrected_rerun"].map(
        lambda value: strict_bool(value, "corrected-rerun availability")
    )
    for column in ("dti_t1_gap_days_abs", "dti_age_at_scan", "dti_field_strength_t"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    if (frame["dti_t1_gap_days_abs"] < 0).any():
        raise ValueError("absolute DTI-T1 gap cannot be negative")
    if not frame["available"].all():
        raise ValueError("approved denominator includes a pair unavailable for rerun")
    if set(frame["corrected_rerun_qc_status"]) != {"not_run"}:
        raise ValueError("corrected-rerun QC is not uniformly outcome-blind/not_run")

    counts = frame["group"].value_counts().to_dict()
    expected = {"CN": 251, "MCI": 186, "AD": 78, "SMC": 15}
    if counts != expected:
        raise ValueError(f"approved diagnosis counts changed: {counts}")
    if int(frame["primary_eligible"].sum()) != int(
        decision.get("primary_cn_mci_ad_ceiling", -1)
    ):
        raise ValueError("primary ceiling differs from SL-H01")

    evidence = {
        "manifest": {
            "path": str(manifest.path.resolve()),
            "sha256": manifest.sha256,
            "size_bytes": manifest.size_bytes,
            "rows": len(manifest.frame),
        },
        "manifest_validation": {
            "path": str(manifest_validation_path.resolve()),
            "sha256": sha256_file(manifest_validation_path),
        },
        "diagnosis": {
            "path": str(diagnosis.path.resolve()),
            "sha256": diagnosis.sha256,
            "size_bytes": diagnosis.size_bytes,
            "rows": len(diagnosis.frame),
        },
        "decision": {
            "path": str(decision_path.resolve()),
            "sha256": sha256_file(decision_path),
            "recorded_utc": decision.get("recorded_utc"),
        },
    }
    return frame, evidence


def population_masks(frame: pd.DataFrame) -> dict[str, tuple[pd.Series, str]]:
    primary = frame["primary_eligible"]
    siemens54 = frame["dti_protocol_key"].eq("SIEMENS|3T|54dir")
    le90 = frame["dti_t1_gap_days_abs"].le(90)
    le180 = frame["dti_t1_gap_days_abs"].le(180)
    le90_siemens = primary & le90 & siemens54
    candidate = frame.loc[le90_siemens]
    site_groups = candidate.groupby("site")["group"].nunique()
    overlap_sites = set(site_groups[site_groups >= 2].index.astype(str))
    masks: dict[str, tuple[pd.Series, str]] = {
        "all_available": (pd.Series(True, index=frame.index), "processing_attrition"),
        "primary_full": (primary, "exploratory_primary"),
        "timing_le_90": (primary & le90, "required_timing_sensitivity"),
        "timing_le_180": (primary & le180, "required_timing_sensitivity"),
        "protocol_siemens_3t_54": (
            primary & siemens54,
            "protocol_sensitivity",
        ),
        "timing_le90_siemens_3t_54": (
            le90_siemens,
            "timing_protocol_sensitivity",
        ),
        "timing_le90_siemens_3t_54_sites_ge2groups": (
            le90_siemens & frame["site"].isin(overlap_sites),
            "site_overlap_stress_test",
        ),
        "t1_original": (
            primary & frame["t1_original"],
            "t1_type_sensitivity",
        ),
        "t1_processed": (
            primary & ~frame["t1_original"],
            "t1_type_sensitivity",
        ),
        "same_adni_phase": (primary & frame["same_phase"], "phase_sensitivity"),
    }
    return masks


def population_summary(
    frame: pd.DataFrame, masks: Mapping[str, tuple[pd.Series, str]]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for name, (mask, role) in masks.items():
        data = frame.loc[mask]
        counts = data["group"].value_counts().reindex(ALL_GROUPS, fill_value=0)
        rows.append(
            {
                "population": name,
                "role": role,
                "n_total": len(data),
                "n_cn": int(counts["CN"]),
                "n_mci": int(counts["MCI"]),
                "n_ad": int(counts["AD"]),
                "n_smc": int(counts["SMC"]),
                "n_sites": data["site"].nunique(),
                "n_protocols": data["dti_protocol_key"].nunique(),
                "n_phases": data["dti_phase"].nunique(),
                "n_t1_original": int(data["t1_original"].sum()),
                "n_t1_processed": int((~data["t1_original"]).sum()),
                "gap_days_median": float(data["dti_t1_gap_days_abs"].median()),
                "gap_days_min": float(data["dti_t1_gap_days_abs"].min()),
                "gap_days_max": float(data["dti_t1_gap_days_abs"].max()),
                "gap_days_unique_values": int(data["dti_t1_gap_days_abs"].nunique()),
                "gap_days_zero_count": int(data["dti_t1_gap_days_abs"].eq(0).sum()),
            }
        )
    return pd.DataFrame(rows)


CELL_FAMILIES: dict[str, tuple[str, ...]] = {
    "timing": ("timing_stratum",),
    "phase": ("dti_phase",),
    "site": ("site",),
    "protocol": ("dti_protocol_key",),
    "t1_type": ("current_t1_type",),
    "timing_t1_type": ("timing_stratum", "current_t1_type"),
    "site_protocol": ("site", "dti_protocol_key"),
    "phase_protocol": ("dti_phase", "dti_protocol_key"),
    "timing_site_protocol_t1_type": (
        "timing_stratum",
        "site",
        "dti_protocol_key",
        "current_t1_type",
    ),
}


def cell_key(columns: Sequence[str], values: Sequence[Any]) -> str:
    return json.dumps(
        {column: str(value) for column, value in zip(columns, values)},
        sort_keys=True,
        separators=(",", ":"),
    )


def feasibility_cells(
    frame: pd.DataFrame, masks: Mapping[str, tuple[pd.Series, str]]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for population, (mask, role) in masks.items():
        data = frame.loc[mask].copy()
        group_order = ALL_GROUPS if population == "all_available" else GROUPS
        for family, columns in CELL_FAMILIES.items():
            counts = (
                data.groupby([*columns, "group"], dropna=False)
                .size()
                .unstack("group", fill_value=0)
                .reindex(columns=group_order, fill_value=0)
            )
            for index, count_row in counts.iterrows():
                values = index if isinstance(index, tuple) else (index,)
                primary_counts = count_row.reindex(GROUPS, fill_value=0)
                total = int(count_row.sum())
                groups_present = int((primary_counts > 0).sum())
                for group in group_order:
                    rows.append(
                        {
                            "population": population,
                            "population_role": role,
                            "cell_family": family,
                            "cell_key": cell_key(columns, values),
                            "group": group,
                            "n": int(count_row[group]),
                            "cell_total": total,
                            "primary_groups_present": groups_present,
                            "all_three_present": bool((primary_counts > 0).all()),
                            "all_three_min_5": bool((primary_counts >= 5).all()),
                            "all_three_min_10": bool((primary_counts >= 10).all()),
                        }
                    )
    return pd.DataFrame(rows)


def common_support(
    frame: pd.DataFrame, masks: Mapping[str, tuple[pd.Series, str]]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    scopes = [("ALL3", GROUPS), *[(f"{a}_vs_{b}", (a, b)) for a, b in CONTRASTS]]
    for population, (mask, _role) in masks.items():
        if population == "all_available":
            continue
        data = frame.loc[mask & frame["primary_eligible"]]
        for family, columns in CELL_FAMILIES.items():
            counts = (
                data.groupby([*columns, "group"], dropna=False)
                .size()
                .unstack("group", fill_value=0)
                .reindex(columns=GROUPS, fill_value=0)
            )
            for scope, groups in scopes:
                group_list = list(groups)
                total_by_group = data["group"].value_counts().reindex(group_list, fill_value=0)
                record: dict[str, Any] = {
                    "population": population,
                    "cell_family": family,
                    "support_scope": scope,
                    "levels_total": len(counts),
                }
                for threshold in (1, 5, 10):
                    supported = (counts[group_list] >= threshold).all(axis=1)
                    supported_counts = counts.loc[supported, group_list].sum(axis=0)
                    record[f"levels_min_{threshold}"] = int(supported.sum())
                    for group in GROUPS:
                        if group in group_list:
                            n = int(supported_counts.get(group, 0))
                            denom = int(total_by_group.get(group, 0))
                            record[f"n_{group.lower()}_min_{threshold}"] = n
                            record[f"fraction_{group.lower()}_min_{threshold}"] = (
                                n / denom if denom else math.nan
                            )
                        else:
                            record[f"n_{group.lower()}_min_{threshold}"] = ""
                            record[f"fraction_{group.lower()}_min_{threshold}"] = ""
                rows.append(record)
    return pd.DataFrame(rows)


def _scaled_power_basis(values: np.ndarray) -> tuple[np.ndarray, list[str]]:
    values = values.astype(float)
    mean = float(values.mean())
    std = float(values.std(ddof=0))
    z = (values - mean) / std if std > 0 else np.zeros_like(values)
    raw = np.column_stack((z, z**2, z**3))
    raw -= raw.mean(axis=0)
    scales = np.sqrt(np.mean(raw**2, axis=0))
    scales[scales == 0] = 1.0
    return raw / scales, [f"center={mean:.12g};scale={std:.12g}"]


def _category_dummies(
    series: pd.Series, prefix: str, reference: str | None = None
) -> tuple[np.ndarray, list[str], str]:
    values = series.astype(str)
    levels = sorted(values.unique().tolist())
    if not levels:
        return np.empty((len(series), 0)), [], ""
    ref = reference if reference in levels else levels[0]
    kept = [level for level in levels if level != ref]
    matrix = np.column_stack([(values == level).astype(float) for level in kept]) if kept else np.empty((len(series), 0))
    return matrix, [f"{prefix}[{level}]" for level in kept], ref


def build_design(
    data: pd.DataFrame, specification: str
) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    """Build outcome-blind proxy designs for rank/support diagnostics.

    Parsimonious designs use linear age and log-gap terms. Cubic designs use a
    three-column centered/scaled basis and diagnosis-by-gap interactions to
    stress-test the dimensionality proposed in the draft analysis design.
    """

    n = len(data)
    cubic = specification.startswith("cubic_")
    arrays: list[np.ndarray] = [np.ones((n, 1), dtype=float)]
    names = ["Intercept"]
    group_mci = data["group"].eq("MCI").astype(float).to_numpy()
    group_ad = data["group"].eq("AD").astype(float).to_numpy()
    arrays.append(np.column_stack((group_mci, group_ad)))
    names.extend(("group[MCI]", "group[AD]"))

    age_basis, age_state = _scaled_power_basis(
        data["dti_age_at_scan"].to_numpy(float)
    )
    arrays.append(age_basis if cubic else age_basis[:, :1])
    names.extend(("age_b1", "age_b2", "age_b3") if cubic else ("age_b1",))
    sex, sex_names, sex_ref = _category_dummies(data["dti_sex"], "sex", "F")
    arrays.append(sex)
    names.extend(sex_names)

    log_gap = np.log1p(data["dti_t1_gap_days_abs"].to_numpy(float))
    gap_basis, gap_state = _scaled_power_basis(log_gap)
    if cubic:
        arrays.append(gap_basis)
        names.extend(("log_gap_b1", "log_gap_b2", "log_gap_b3"))
        interactions = np.column_stack(
            [group_mci * gap_basis[:, index] for index in range(3)]
            + [group_ad * gap_basis[:, index] for index in range(3)]
        )
        arrays.append(interactions)
        names.extend([f"group[MCI]:log_gap_b{i}" for i in range(1, 4)])
        names.extend([f"group[AD]:log_gap_b{i}" for i in range(1, 4)])
    else:
        arrays.append(gap_basis[:, :1])
        names.append("log_gap_b1")

    include_phase = specification in {
        "parsimonious_phase",
        "cubic_phase_protocol",
        "cubic_phase_protocol_t1",
        "cubic_site_fixed_stress",
    }
    include_protocol = specification in {
        "parsimonious_protocol",
        "parsimonious_protocol_t1",
        "cubic_phase_protocol",
        "cubic_phase_protocol_t1",
        "cubic_site_fixed_stress",
    }
    refs: dict[str, str] = {"sex": sex_ref}
    if include_phase:
        phase, phase_names, phase_ref = _category_dummies(
            data["dti_phase"], "phase", "ADNI 3"
        )
        arrays.append(phase)
        names.extend(phase_names)
        refs["phase"] = phase_ref
    if include_protocol:
        protocol, protocol_names, protocol_ref = _category_dummies(
            data["dti_protocol_key"], "protocol", "SIEMENS|3T|54dir"
        )
        arrays.append(protocol)
        names.extend(protocol_names)
        refs["protocol"] = protocol_ref
    if specification in {
        "parsimonious_protocol_t1",
        "cubic_phase_protocol_t1",
        "cubic_site_fixed_stress",
    }:
        t1 = (~data["t1_original"]).astype(float).to_numpy()[:, None]
        arrays.append(t1)
        names.append("t1[Processed]")
    if specification == "cubic_site_fixed_stress":
        site, site_names, site_ref = _category_dummies(data["site"], "site")
        arrays.append(site)
        names.extend(site_names)
        refs["site"] = site_ref

    matrix = np.column_stack(arrays)
    metadata = {
        "age_basis_state": age_state[0],
        "log_gap_basis_state": gap_state[0],
        "references": refs,
        "basis_note": (
            "centered/scaled cubic interaction feasibility proxy; final SAP spline not frozen here"
            if cubic
            else "parsimonious centered/scaled linear feasibility proxy"
        ),
    }
    return matrix, names, metadata


def design_diagnostics(
    frame: pd.DataFrame, masks: Mapping[str, tuple[pd.Series, str]]
) -> tuple[pd.DataFrame, dict[tuple[str, str, str], dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    contrast_lookup: dict[tuple[str, str, str], dict[str, Any]] = {}
    specs = (
        "parsimonious_core",
        "parsimonious_phase",
        "parsimonious_protocol",
        "parsimonious_protocol_t1",
        "cubic_phase_protocol",
        "cubic_phase_protocol_t1",
        "cubic_site_fixed_stress",
    )
    q = NormalDist().inv_cdf(1 - ALPHA_FAMILY / N_PRIMARY_CONTRASTS / 2) + NormalDist().inv_cdf(TARGET_POWER)
    for population, (mask, _role) in masks.items():
        if population == "all_available":
            continue
        data = frame.loc[mask & frame["primary_eligible"]].reset_index(drop=True)
        for spec in specs:
            matrix, names, metadata = build_design(data, spec)
            singular = np.linalg.svd(matrix, compute_uv=False)
            tolerance = max(matrix.shape) * np.finfo(float).eps * singular[0]
            rank = int((singular > tolerance).sum())
            column_norms = np.linalg.norm(matrix, axis=0)
            normalized = matrix[:, column_norms > 0] / column_norms[column_norms > 0]
            normalized_singular = np.linalg.svd(normalized, compute_uv=False)
            norm_tolerance = max(normalized.shape) * np.finfo(float).eps * normalized_singular[0]
            norm_rank = int((normalized_singular > norm_tolerance).sum())
            condition = (
                float(normalized_singular[0] / normalized_singular[-1])
                if norm_rank == normalized.shape[1]
                else math.inf
            )
            zero_columns = [
                names[index]
                for index, norm in enumerate(column_norms)
                if norm <= 1e-12
            ]
            duplicate_columns: list[list[str]] = []
            for left_index in range(matrix.shape[1]):
                for right_index in range(left_index + 1, matrix.shape[1]):
                    if np.allclose(
                        matrix[:, left_index],
                        matrix[:, right_index],
                        rtol=0,
                        atol=1e-12,
                    ):
                        duplicate_columns.append(
                            [names[left_index], names[right_index]]
                        )
            row = {
                "population": population,
                "specification": spec,
                "n": len(data),
                "columns": matrix.shape[1],
                "rank": rank,
                "rank_deficiency": matrix.shape[1] - rank,
                "residual_df": len(data) - rank,
                "normalized_condition_number": condition,
                "full_column_rank": rank == matrix.shape[1],
                "near_singular_condition_gt_1000": condition > 1000,
                "zero_columns_json": json.dumps(zero_columns),
                "exact_duplicate_columns_json": json.dumps(duplicate_columns),
                "basis_note": metadata["basis_note"],
                "basis_state_json": json.dumps(metadata, sort_keys=True),
                "column_names_json": json.dumps(names),
            }
            rows.append(row)

            u, s, vt = np.linalg.svd(matrix, full_matrices=False)
            rowspace = vt[:rank, :]
            xtx_pinv = np.linalg.pinv(matrix.T @ matrix, rcond=1e-12)
            group_indices = {name: names.index(name) for name in ("group[MCI]", "group[AD]")}
            for left, right in CONTRASTS:
                contrast = np.zeros(matrix.shape[1])
                if right == "MCI":
                    contrast[group_indices["group[MCI]"]] += 1
                if right == "AD":
                    contrast[group_indices["group[AD]"]] += 1
                if left == "MCI":
                    contrast[group_indices["group[MCI]"]] -= 1
                if left == "AD":
                    contrast[group_indices["group[AD]"]] -= 1
                projected = rowspace.T @ (rowspace @ contrast)
                error = float(np.linalg.norm(contrast - projected))
                estimable = error <= 1e-8 * max(1.0, np.linalg.norm(contrast))
                se_factor = (
                    float(math.sqrt(max(0.0, contrast @ xtx_pinv @ contrast)))
                    if estimable
                    else math.nan
                )
                contrast_lookup[(population, spec, f"{left}_vs_{right}")] = {
                    "algebraically_estimable": estimable,
                    "rowspace_error": error,
                    "standard_error_per_residual_sd": se_factor,
                    "adjusted_standardized_mde_80pct": q * se_factor if estimable else math.nan,
                }
    return pd.DataFrame(rows), contrast_lookup


def estimability_table(
    frame: pd.DataFrame,
    masks: Mapping[str, tuple[pd.Series, str]],
    support: pd.DataFrame,
    contrast_lookup: Mapping[tuple[str, str, str], Mapping[str, Any]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    support_index = support.set_index(["population", "cell_family", "support_scope"])
    for population, (mask, _role) in masks.items():
        if population == "all_available":
            continue
        data = frame.loc[mask & frame["primary_eligible"]]
        counts = data["group"].value_counts().reindex(GROUPS, fill_value=0)
        for left, right in CONTRASTS:
            contrast_name = f"{left}_vs_{right}"
            design = contrast_lookup[
                (population, "parsimonious_protocol", contrast_name)
            ]
            sp = support_index.loc[(population, "site_protocol", contrast_name)]
            common_cells = int(sp["levels_min_1"])
            common_left = int(sp[f"n_{left.lower()}_min_1"])
            common_right = int(sp[f"n_{right.lower()}_min_1"])
            robust_cells = int(sp["levels_min_5"])
            if not design["algebraically_estimable"]:
                status = "NOT_ESTIMABLE_DESIGN_ALIAS"
            elif common_cells == 0:
                status = "NOT_ESTIMABLE_NO_COMMON_SITE_PROTOCOL_CELL"
            elif min(common_left, common_right) < 10:
                status = "ALGEBRAICALLY_ESTIMABLE_SUPPORT_FRAGILE"
            elif robust_cells == 0:
                status = "ESTIMABLE_WITH_LIMITED_CELL_DEPTH"
            else:
                status = "ESTIMABLE_WITH_MEASURED_COMMON_SUPPORT"
            rows.append(
                {
                    "record_type": "population_contrast",
                    "population": population,
                    "scope_family": "aggregate",
                    "scope_key": "ALL",
                    "contrast": contrast_name,
                    "n_left": int(counts[left]),
                    "n_right": int(counts[right]),
                    "common_site_protocol_cells": common_cells,
                    "common_supported_n_left": common_left,
                    "common_supported_n_right": common_right,
                    "site_protocol_cells_min_5_each": robust_cells,
                    "algebraically_estimable": bool(design["algebraically_estimable"]),
                    "status": status,
                    "rule": "aggregate phase+protocol cubic-basis proxy plus observed site-protocol support",
                }
            )

    primary = frame.loc[frame["primary_eligible"]]
    for family, columns in CELL_FAMILIES.items():
        counts = (
            primary.groupby([*columns, "group"], dropna=False)
            .size()
            .unstack("group", fill_value=0)
            .reindex(columns=GROUPS, fill_value=0)
        )
        for index, count_row in counts.iterrows():
            values = index if isinstance(index, tuple) else (index,)
            key = cell_key(columns, values)
            for left, right in CONTRASTS:
                n_left, n_right = int(count_row[left]), int(count_row[right])
                if n_left == 0 or n_right == 0:
                    status = "NOT_ESTIMABLE_MISSING_GROUP"
                elif min(n_left, n_right) < 5:
                    status = "PRESENT_BUT_DESCRIPTIVE_ONLY_VERY_SPARSE"
                elif min(n_left, n_right) < 10:
                    status = "WITHIN_CELL_ESTIMABLE_BUT_SPARSE"
                else:
                    status = "WITHIN_CELL_ESTIMABLE"
                rows.append(
                    {
                        "record_type": "exact_cell_contrast",
                        "population": "primary_full",
                        "scope_family": family,
                        "scope_key": key,
                        "contrast": f"{left}_vs_{right}",
                        "n_left": n_left,
                        "n_right": n_right,
                        "common_site_protocol_cells": "",
                        "common_supported_n_left": "",
                        "common_supported_n_right": "",
                        "site_protocol_cells_min_5_each": "",
                        "algebraically_estimable": n_left > 0 and n_right > 0,
                        "status": status,
                        "rule": "cell presence; >=5/group sparse threshold; >=10/group stable descriptive threshold",
                    }
                )
    return pd.DataFrame(rows)


def naive_mde(n_left: np.ndarray | float, n_right: np.ndarray | float) -> np.ndarray:
    n_left_array = np.asarray(n_left, dtype=float)
    n_right_array = np.asarray(n_right, dtype=float)
    q = NormalDist().inv_cdf(1 - ALPHA_FAMILY / N_PRIMARY_CONTRASTS / 2) + NormalDist().inv_cdf(TARGET_POWER)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = q * np.sqrt(1 / n_left_array + 1 / n_right_array)
    result[(n_left_array <= 1) | (n_right_array <= 1)] = np.nan
    return result


def power_attrition(
    frame: pd.DataFrame,
    masks: Mapping[str, tuple[pd.Series, str]],
    contrast_lookup: Mapping[tuple[str, str, str], Mapping[str, Any]],
) -> pd.DataFrame:
    scenarios = {
        "none": lambda d: np.zeros(len(d)),
        "uniform_10pct": lambda d: np.full(len(d), 0.10),
        "uniform_20pct": lambda d: np.full(len(d), 0.20),
        "uniform_30pct": lambda d: np.full(len(d), 0.30),
        "differential_ad_25pct_others_10pct": lambda d: np.where(
            d["group"].eq("AD"), 0.25, 0.10
        ),
        "metadata_stress_processed_plus_long_gap": lambda d: np.clip(
            0.05
            + np.where(d["t1_original"], 0.0, 0.15)
            + np.where(d["dti_t1_gap_days_abs"].gt(180), 0.10, 0.0),
            0,
            0.60,
        ),
    }
    rng = np.random.default_rng(RANDOM_SEED)
    rows: list[dict[str, Any]] = []
    for population, (mask, _role) in masks.items():
        if population == "all_available":
            continue
        data = frame.loc[mask & frame["primary_eligible"]].reset_index(drop=True)
        group_arrays = {group: data["group"].eq(group).to_numpy() for group in GROUPS}
        baseline_counts = {group: int(values.sum()) for group, values in group_arrays.items()}
        for scenario, probability_function in scenarios.items():
            failure_probability = np.asarray(probability_function(data), dtype=float)
            if scenario == "none":
                retained_counts = {
                    group: np.full(SIMULATIONS, baseline_counts[group], dtype=int)
                    for group in GROUPS
                }
            else:
                retained = rng.random((SIMULATIONS, len(data))) >= failure_probability[None, :]
                retained_counts = {
                    group: retained[:, values].sum(axis=1)
                    for group, values in group_arrays.items()
                }
            for group in GROUPS:
                values = retained_counts[group]
                rows.append(
                    {
                        "record_type": "group_retention",
                        "population": population,
                        "scenario": scenario,
                        "contrast_or_group": group,
                        "baseline_n_left": baseline_counts[group],
                        "baseline_n_right": "",
                        "expected_failure_probability_mean": float(
                            failure_probability[group_arrays[group]].mean()
                        ),
                        "retained_n_mean": float(values.mean()),
                        "retained_n_p2_5": float(np.quantile(values, 0.025)),
                        "retained_n_median": float(np.quantile(values, 0.5)),
                        "retained_n_p97_5": float(np.quantile(values, 0.975)),
                        "probability_either_group_below_20": float((values < 20).mean()),
                        "naive_standardized_mde_median": "",
                        "naive_standardized_mde_p97_5": "",
                        "adjusted_complete_case_mde_parsimonious_protocol_proxy": "",
                    }
                )
            for left, right in CONTRASTS:
                left_counts, right_counts = retained_counts[left], retained_counts[right]
                mdes = naive_mde(left_counts, right_counts)
                finite = mdes[np.isfinite(mdes)]
                contrast_name = f"{left}_vs_{right}"
                adjusted = contrast_lookup[
                    (population, "parsimonious_protocol", contrast_name)
                ][
                    "adjusted_standardized_mde_80pct"
                ]
                rows.append(
                    {
                        "record_type": "contrast_mde",
                        "population": population,
                        "scenario": scenario,
                        "contrast_or_group": contrast_name,
                        "baseline_n_left": baseline_counts[left],
                        "baseline_n_right": baseline_counts[right],
                        "expected_failure_probability_mean": float(failure_probability.mean()),
                        "retained_n_mean": "",
                        "retained_n_p2_5": "",
                        "retained_n_median": "",
                        "retained_n_p97_5": "",
                        "probability_either_group_below_20": float(
                            ((left_counts < 20) | (right_counts < 20)).mean()
                        ),
                        "naive_standardized_mde_median": float(np.median(finite)) if len(finite) else math.nan,
                        "naive_standardized_mde_p97_5": float(np.quantile(finite, 0.975)) if len(finite) else math.nan,
                        "adjusted_complete_case_mde_parsimonious_protocol_proxy": (
                            adjusted if scenario == "none" else ""
                        ),
                    }
                )
    return pd.DataFrame(rows)


def load_historical_proxy_validation(
    path: Path, expected_manifest_sha256: str
) -> dict[str, Any]:
    proxy = load_json(path, "historical source-preflight validation")
    if proxy.get("status") != "PASS":
        raise ValueError("historical source-preflight proxy is not PASS")
    if proxy.get("release_status") != "HISTORICAL_PROXY_ONLY_RAW_SOURCE_VALIDATION_REQUIRED":
        raise ValueError("historical source-preflight release role is invalid")
    if proxy.get("input_manifest", {}).get("sha256") != expected_manifest_sha256:
        raise ValueError("historical source-preflight proxy is bound to another manifest")
    for name, evidence in proxy.get("outputs", {}).items():
        output_path = path.parent / name
        if not output_path.is_file() or sha256_file(output_path) != evidence.get("sha256"):
            raise ValueError(f"historical source-preflight output checksum mismatch: {name}")
    expected_counts = {
        "rows": 530,
        "pe_and_readout_complete": 380,
        "pe_or_readout_missing": 150,
        "unique_directions_lt_30": 10,
        "multiple_nonzero_shells": 6,
        "b0_vector_norm_issue": 51,
        "processed_t1_headers": 225,
        "processed_t1_uid_text_found": 0,
    }
    if proxy.get("counts") != expected_counts:
        raise ValueError("historical source-preflight counts changed")
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "release_status": proxy["release_status"],
        "counts": proxy["counts"],
        "safety": proxy.get("safety", {}),
        "limitations": proxy.get("limitations", []),
    }


def workflow_contract_feasibility(
    frame: pd.DataFrame,
    config_path: Path,
    historical_proxy: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compare the SL-D01 manifest and SL-H01 policy with the v2 workflow contract.

    This is a tabular/schema audit only.  Header-derived fields are classified
    as potentially derivable but are not claimed present until a rowwise
    extraction and validation step proves coverage.
    """

    if config_path.is_symlink():
        raise ValueError(f"Refusing symlinked workflow config: {config_path}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("workflow configuration must parse as a mapping")
    canonical_fields = list(config["inputs"]["source_identity_required"])
    workflow_fields = [
        "dwi_nifti",
        "dwi_bvec",
        "dwi_bval",
        "dwi_json",
        "t1_nifti",
        "dwi_nifti_sha256",
        "dwi_bvec_sha256",
        "dwi_bval_sha256",
        "dwi_json_sha256",
        "t1_nifti_sha256",
        "phase_encoding_direction",
        "total_readout_time",
        "pair_approved",
    ]
    direct_map = {
        "subject_id": "subject_id",
        "diagnosis_at_dti": "diagnosis_reconciled_harmonized",
        "dti_image_id": "dti_image_id",
        "dti_study_date": "dti_study_date",
        "t1_image_id": "current_t1_image_id",
        "t1_study_date": "current_t1_study_date",
        "abs_pair_gap_days": "dti_t1_gap_days_abs",
        "phase": "dti_phase",
        "site": "site",
        "manufacturer": "dti_manufacturer",
        "field_strength_t": "dti_field_strength_t",
        "protocol": "dti_protocol_key",
    }
    header_fields = {
        "dti_series_uid": (
            "DICOM SeriesInstanceUID",
            "Extract from each resolved local DTI series; any row without an authoritative UID fails closed.",
        ),
        "scanner_model": (
            "DICOM ManufacturerModelName",
            "Extract and normalize from local DTI headers; publish missingness before canary selection.",
        ),
        "phase_encoding_direction": (
            "DICOM/header-derived BIDS PhaseEncodingDirection",
            "Derive with dcm2niix/mrinfo and verify against the staged JSON; no hardcoded default.",
        ),
        "total_readout_time": (
            "DICOM/header-derived BIDS TotalReadoutTime",
            "Derive and cross-check rowwise; rows lacking sufficient echo-spacing metadata cannot enter eddy.",
        ),
    }
    staging_fields = {
        "source_file_sha256": (
            "exact local source-file or source-bundle content hashes",
            "Bind the SL-P0-09 content-hash inventory and define whether this field is a file or bundle digest.",
        ),
        "dwi_nifti": (
            "staged mrconvert/dcm2niix output",
            "Create a write-once canary staging manifest from the resolved local DTI directory.",
        ),
        "dwi_bvec": (
            "staged gradient export",
            "Export from the same source conversion and validate count, norm, and rotation.",
        ),
        "dwi_bval": (
            "staged gradient export",
            "Export from the same source conversion and validate against DWI volumes.",
        ),
        "dwi_json": (
            "staged JSON sidecar",
            "Generate from the same source conversion; retain tool/version and source hashes.",
        ),
        "t1_nifti": (
            "existing native NIfTI or write-once source conversion",
            "Resolve one native local T1 input per approved pair and hash exact bytes.",
        ),
        "dwi_nifti_sha256": ("staged dwi_nifti bytes", "Hash exact staged bytes."),
        "dwi_bvec_sha256": ("staged dwi_bvec bytes", "Hash exact staged bytes."),
        "dwi_bval_sha256": ("staged dwi_bval bytes", "Hash exact staged bytes."),
        "dwi_json_sha256": ("staged dwi_json bytes", "Hash exact staged bytes."),
        "t1_nifti_sha256": ("staged t1_nifti bytes", "Hash exact staged bytes."),
    }

    rows: list[dict[str, Any]] = []
    for section, fields in (
        ("source_identity_required", canonical_fields),
        ("workflow_required_fields", workflow_fields),
    ):
        for field in fields:
            if field in direct_map:
                column = direct_map[field]
                present = int(frame[column].astype(str).str.strip().ne("").sum())
                classification = "present"
                blocker = present != len(frame)
                action = "Rename into the workflow schema and preserve source-column provenance."
                evidence = f"manifest column {column}"
            elif field == "t1_series_uid":
                column = ""
                present = 0
                classification = "unavailable"
                blocker = True
                evidence = (
                    "305 Original T1 sources may carry a DICOM SeriesInstanceUID; historical "
                    "header-only proxy found no UID-like text in 225/225 Processed NIfTI inputs"
                )
                action = (
                    "Revise the contract to nullable t1_dicom_series_uid plus mandatory stable "
                    "t1_source_id = ADNI Image ID + locked source-bundle hash; never fabricate a UID."
                )
            elif field in header_fields:
                evidence, action = header_fields[field]
                if field in {"phase_encoding_direction", "total_readout_time"}:
                    evidence += (
                        f"; historical selected-ID derivative proxy complete for "
                        f"{historical_proxy['counts']['pe_and_readout_complete']}/530 and missing "
                        f"{historical_proxy['counts']['pe_or_readout_missing']}/530"
                    )
                column = ""
                present = 0
                classification = "derivable_from_local_headers"
                blocker = True
            elif field in staging_fields:
                evidence, action = staging_fields[field]
                column = ""
                present = 0
                classification = "derivable_from_local_files"
                blocker = True
            elif field == "pair_approved":
                column = ""
                present = 0
                classification = "policy_conflicting"
                blocker = True
                evidence = "SL-H01 approves all 530 for processing/attrition, 515 for primary contrasts, and 15 SMC separately; current workflow rejects SMC and >180-day pairs"
                action = "Add explicit processing_approved and primary_analysis_eligible fields, and remove the hard rejection of policy-approved descriptive/long-gap attempts."
            else:
                column = ""
                present = 0
                classification = "unavailable"
                blocker = True
                evidence = "no manifest mapping or declared local derivation"
                action = "Resolve the field source or revise the contract before canary execution."
            rows.append(
                {
                    "record_type": "required_field",
                    "contract_section": section,
                    "required_field_or_policy": field,
                    "classification": classification,
                    "manifest_column": column,
                    "rows_present_in_manifest": present,
                    "rows_required": len(frame),
                    "blocking_before_canary": blocker,
                    "evidence": evidence,
                    "required_action": action,
                }
            )

    manifest_path = Path(config["inputs"]["approved_pair_manifest"]["path"])
    policy_rows = [
        (
            "approved_pair_manifest.path",
            "policy_conflicting",
            f"configured={manifest_path}; exists={manifest_path.is_file()}; approved manifest is {DEFAULT_MANIFEST}",
            "Point a new recipe/config version at the checksum-locked SL-D01 manifest or its validated workflow projection.",
        ),
        (
            "approved_pair_manifest.sha256",
            "unavailable",
            f"configured={config['inputs']['approved_pair_manifest'].get('sha256')}",
            "Freeze the exact workflow-projection manifest SHA-256 after required metadata is populated.",
        ),
        (
            "pairing.primary_max_abs_days=90",
            "policy_conflicting",
            "SL-H01 defines full available-data CN/MCI/AD as exploratory primary; <=90 days is a required sensitivity, not the only processable cohort.",
            "Revise the recipe population policy and preserve <=90 days as a named sensitivity.",
        ),
        (
            "pairing.sensitivity_max_abs_days=180",
            "policy_conflicting",
            f"current workflow rejects >180 days; approved manifest contains {int(frame['dti_t1_gap_days_abs'].gt(180).sum())} such pairs",
            "Allow all 530 processing attempts and carry continuous gap/long-gap flags into analysis rather than rejecting source rows.",
        ),
        (
            "pairing.require_original_t1=true",
            "policy_conflicting",
            f"approved fixed cohort contains {int((~frame['t1_original']).sum())} processed T1 inputs",
            "Allow the available T1 type, record it, and run Original-versus-processed sensitivity; do not relabel processed as native Original.",
        ),
        (
            "t1_and_tissue.t1_source=approved_visit_matched_native_original",
            "policy_conflicting",
            f"{int(frame['dti_t1_gap_days_abs'].gt(180).sum())} pairs are >180 days and {int((~frame['t1_original']).sum())} T1 inputs are processed",
            "Replace visit-matched/native-Original wording with available local T1 plus explicit timing/type covariates and limitations.",
        ),
        (
            "workflow hard-rejects SMC",
            "policy_conflicting",
            f"SL-H01 retains {int(frame['smc_separate'].sum())} SMC in processing/QC/attrition but outside primary contrasts",
            "Process SMC through the same technical recipe and exclude it only at the primary inference selector.",
        ),
    ]
    for policy, classification, evidence, action in policy_rows:
        rows.append(
            {
                "record_type": "policy",
                "contract_section": "workflow_policy",
                "required_field_or_policy": policy,
                "classification": classification,
                "manifest_column": "",
                "rows_present_in_manifest": "",
                "rows_required": "",
                "blocking_before_canary": True,
                "evidence": evidence,
                "required_action": action,
            }
        )
    contract = pd.DataFrame(rows)
    evidence = {
        "path": str(config_path.resolve()),
        "sha256": sha256_file(config_path),
        "recipe_id": config.get("contract", {}).get("recipe_id"),
        "configured_manifest_path": str(manifest_path),
        "configured_manifest_exists": manifest_path.is_file(),
        "classification_counts": contract["classification"]
        .value_counts()
        .sort_index()
        .astype(int)
        .to_dict(),
        "blocking_rows": int(contract["blocking_before_canary"].astype(bool).sum()),
        "historical_source_preflight_proxy": dict(historical_proxy),
    }
    return contract, evidence


def save_plots(
    population_table: pd.DataFrame,
    support: pd.DataFrame,
    power: pd.DataFrame,
    figure_dir: Path,
) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    display_populations = [
        "primary_full",
        "timing_le_90",
        "timing_le_180",
        "protocol_siemens_3t_54",
        "timing_le90_siemens_3t_54",
        "timing_le90_siemens_3t_54_sites_ge2groups",
        "t1_original",
        "t1_processed",
    ]
    p = population_table.set_index("population").loc[display_populations]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    x = np.arange(len(p))
    width = 0.25
    colors = {"CN": "#2c7fb8", "MCI": "#fdae61", "AD": "#d7191c"}
    for offset, group in enumerate(GROUPS):
        axes[0].bar(
            x + (offset - 1) * width,
            p[f"n_{group.lower()}"],
            width,
            label=group,
            color=colors[group],
        )
    axes[0].set_xticks(x, [item.replace("_", "\n") for item in display_populations], rotation=35, ha="right")
    axes[0].set_ylabel("Subjects")
    axes[0].set_title("Available subjects by prespecified population")
    axes[0].legend(frameon=False)

    s = support[
        (support["cell_family"] == "site_protocol")
        & (support["support_scope"] == "ALL3")
        & support["population"].isin(display_populations)
    ].set_index("population").reindex(display_populations)
    for offset, group in enumerate(GROUPS):
        axes[1].bar(
            x + (offset - 1) * width,
            100 * pd.to_numeric(s[f"fraction_{group.lower()}_min_1"]),
            width,
            label=group,
            color=colors[group],
        )
    axes[1].set_xticks(x, [item.replace("_", "\n") for item in display_populations], rotation=35, ha="right")
    axes[1].set_ylabel("% in site×protocol cells containing all 3 groups")
    axes[1].set_title("Measured three-group common support")
    axes[1].set_ylim(0, 105)
    axes[1].legend(frameon=False)
    fig.suptitle("SL-P0-10 fixed-data overlap audit (no biological outcomes)")
    fig.savefig(figure_dir / FIG_SUPPORT, dpi=180)
    plt.close(fig)

    scenarios = [
        "none",
        "uniform_10pct",
        "uniform_20pct",
        "uniform_30pct",
        "differential_ad_25pct_others_10pct",
        "metadata_stress_processed_plus_long_gap",
    ]
    q = power[
        (power["record_type"] == "contrast_mde")
        & (power["population"] == "primary_full")
    ].copy()
    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    for contrast, group_data in q.groupby("contrast_or_group"):
        values = group_data.set_index("scenario").reindex(scenarios)
        ax.plot(
            range(len(scenarios)),
            pd.to_numeric(values["naive_standardized_mde_median"]),
            marker="o",
            linewidth=2,
            label=contrast.replace("_vs_", " vs "),
        )
    ax.set_xticks(range(len(scenarios)), [item.replace("_", "\n") for item in scenarios], rotation=20, ha="right")
    ax.set_ylabel("Approximate standardized MDE (80% power)")
    ax.set_title("Hypothetical attrition raises the detectable-effect floor\nHolm worst-case α=0.05/3; independent Gaussian planning approximation")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(figure_dir / FIG_POWER, dpi=180)
    plt.close(fig)


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "NA"
    if isinstance(value, (float, np.floating)):
        return f"{value:.{digits}f}"
    return str(value)


def build_memo(
    generated_utc: str,
    population_table: pd.DataFrame,
    feasibility: pd.DataFrame,
    support: pd.DataFrame,
    design: pd.DataFrame,
    estimability: pd.DataFrame,
    power: pd.DataFrame,
    workflow_contract: pd.DataFrame,
    workflow_evidence: Mapping[str, Any],
    evidence: Mapping[str, Any],
) -> str:
    pop = population_table.set_index("population")
    full = pop.loc["primary_full"]
    exact = support[
        (support["population"] == "primary_full")
        & (support["cell_family"] == "timing_site_protocol_t1_type")
        & (support["support_scope"] == "ALL3")
    ].iloc[0]
    site = support[
        (support["population"] == "primary_full")
        & (support["cell_family"] == "site")
        & (support["support_scope"] == "ALL3")
    ].iloc[0]
    site_protocol = support[
        (support["population"] == "primary_full")
        & (support["cell_family"] == "site_protocol")
        & (support["support_scope"] == "ALL3")
    ].iloc[0]
    aggregate = estimability[estimability["record_type"] == "population_contrast"]
    full_estimability = aggregate[aggregate["population"] == "primary_full"]
    parsimonious_design = design[
        (design["population"] == "primary_full")
        & (design["specification"] == "parsimonious_protocol")
    ].iloc[0]
    cubic_design = design[
        (design["population"] == "primary_full")
        & (design["specification"] == "cubic_phase_protocol")
    ].iloc[0]
    site_design = design[
        (design["population"] == "primary_full")
        & (design["specification"] == "cubic_site_fixed_stress")
    ].iloc[0]
    baseline_power = power[
        (power["population"] == "primary_full")
        & (power["scenario"] == "none")
        & (power["record_type"] == "contrast_mde")
    ].set_index("contrast_or_group")
    stress_power = power[
        (power["population"] == "primary_full")
        & (power["scenario"] == "metadata_stress_processed_plus_long_gap")
        & (power["record_type"] == "contrast_mde")
    ].set_index("contrast_or_group")
    contract_counts = (
        workflow_contract["classification"].value_counts().sort_index().to_dict()
    )
    policy_conflicts = workflow_contract[
        (workflow_contract["record_type"] == "policy")
        & (workflow_contract["classification"] == "policy_conflicting")
    ]
    timing_t1_counts: dict[tuple[str, str, str], int] = {}
    for row in feasibility[
        (feasibility["population"] == "primary_full")
        & (feasibility["cell_family"] == "timing_t1_type")
    ].itertuples(index=False):
        key = json.loads(row.cell_key)
        timing_t1_counts[
            (key["timing_stratum"], key["current_t1_type"], row.group)
        ] = int(row.n)

    def timing_t1(timing: str, t1_type: str, group: str) -> int:
        return timing_t1_counts.get((timing, t1_type, group), 0)

    lines = [
        "# SL-P0-10 fixed-data feasibility, overlap, attrition-risk, and power memo",
        "",
        f"**Generated:** {generated_utc}",
        "**Scope:** the core feasibility analysis read tabular inputs only. No voxel arrays, connectome values, biological outcomes, or group-result inference were read; the separately checksum-bound historical proxy used read-only access to 530 MIF headers and 225 processed-T1 NIfTI headers with zero voxel-array loads.",
        "**Decision boundary:** all 530 local pairs remain in processing/attrition; CN/MCI/AD inference has a pre-QC ceiling of 515; SMC remains descriptive.",
        "",
        "## Executive finding",
        "",
        "The fixed cohort can support an exploratory aggregate CN/MCI/AD association model, but it does not support an acquisition-saturated or broadly phase/site-specific disease claim. CN–MCI has the strongest observed overlap. Any contrast involving AD is substantially more support-limited, especially inside timing-restricted or processed-T1 strata. The corrected study must therefore use a parsimonious prespecified nuisance representation, publish support diagnostics, and label unsupported cell-specific contrasts non-estimable.",
        "",
        "## Locked cohort and required populations",
        "",
        f"- Full primary ceiling: CN {int(full.n_cn)}, MCI {int(full.n_mci)}, AD {int(full.n_ad)}; total {int(full.n_total)}.",
        f"- All-input denominator: {int(pop.loc['all_available'].n_total)}, including {int(pop.loc['all_available'].n_smc)} SMC retained separately.",
        f"- <=90 days: CN {int(pop.loc['timing_le_90'].n_cn)}, MCI {int(pop.loc['timing_le_90'].n_mci)}, AD {int(pop.loc['timing_le_90'].n_ad)}.",
        f"- <=180 days: CN {int(pop.loc['timing_le_180'].n_cn)}, MCI {int(pop.loc['timing_le_180'].n_mci)}, AD {int(pop.loc['timing_le_180'].n_ad)}. This adds only three AD subjects beyond <=90 days.",
        f"- Gap-support warning: <=90 days has only {int(pop.loc['timing_le_90'].gap_days_unique_values)} distinct gap values ({int(pop.loc['timing_le_90'].gap_days_zero_count)}/{int(pop.loc['timing_le_90'].n_total)} are zero); <=180 days has only {int(pop.loc['timing_le_180'].gap_days_unique_values)} distinct values. The <=90-day Siemens-54 subset is uniformly zero-gap. Nonlinear gap or diagnosis-by-gap terms are therefore not estimable inside those restricted subsets and must be reduced by an outcome-blind rule; continuous-gap interaction belongs to the full-cohort analysis.",
        f"- Siemens 3 T / 54 directions: CN {int(pop.loc['protocol_siemens_3t_54'].n_cn)}, MCI {int(pop.loc['protocol_siemens_3t_54'].n_mci)}, AD {int(pop.loc['protocol_siemens_3t_54'].n_ad)}.",
        f"- <=90-day Siemens-54: CN {int(pop.loc['timing_le90_siemens_3t_54'].n_cn)}, MCI {int(pop.loc['timing_le90_siemens_3t_54'].n_mci)}, AD {int(pop.loc['timing_le90_siemens_3t_54'].n_ad)}.",
        f"- Original T1: CN {int(pop.loc['t1_original'].n_cn)}, MCI {int(pop.loc['t1_original'].n_mci)}, AD {int(pop.loc['t1_original'].n_ad)}; processed T1: CN {int(pop.loc['t1_processed'].n_cn)}, MCI {int(pop.loc['t1_processed'].n_mci)}, AD {int(pop.loc['t1_processed'].n_ad)}.",
        f"- Timing×T1-type cells (CN/MCI/AD): <=90 Original {timing_t1('le_90_days', 'Original', 'CN')}/{timing_t1('le_90_days', 'Original', 'MCI')}/{timing_t1('le_90_days', 'Original', 'AD')}; <=90 Processed {timing_t1('le_90_days', 'Processed', 'CN')}/{timing_t1('le_90_days', 'Processed', 'MCI')}/{timing_t1('le_90_days', 'Processed', 'AD')}; 91–180 Processed {timing_t1('days_91_180', 'Processed', 'CN')}/{timing_t1('days_91_180', 'Processed', 'MCI')}/{timing_t1('days_91_180', 'Processed', 'AD')}; >180 Original {timing_t1('gt_180_days', 'Original', 'CN')}/{timing_t1('gt_180_days', 'Original', 'MCI')}/{timing_t1('gt_180_days', 'Original', 'AD')}; >180 Processed {timing_t1('gt_180_days', 'Processed', 'CN')}/{timing_t1('gt_180_days', 'Processed', 'MCI')}/{timing_t1('gt_180_days', 'Processed', 'AD')}.",
        "",
        "## Common-support findings",
        "",
        f"- {int(site.levels_min_1)}/{int(site.levels_total)} sites contain all three primary groups; only {int(site.levels_min_5)} sites contain at least five per group and {int(site.levels_min_10)} contains at least ten per group.",
        f"- {int(site_protocol.levels_min_1)}/{int(site_protocol.levels_total)} site×protocol cells contain all three groups; **none** contains at least five per group.",
        f"- The exact timing×site×protocol×T1-type partition has {int(exact.levels_min_1)} all-three cells out of {int(exact.levels_total)}, containing CN {int(exact.n_cn_min_1)}, MCI {int(exact.n_mci_min_1)}, AD {int(exact.n_ad_min_1)}; **none** reaches five per group.",
        "- The 91–180-day cell contains three AD and no CN or MCI. No within-cell primary diagnosis contrast is estimable there.",
        "- ADNI 2 contains AD 33, CN 3, and MCI 0; MCI contrasts within ADNI 2 are not estimable. ADNI 4 contains only one AD, so AD contrasts there are descriptive only.",
        "",
        "## Design rank and estimability",
        "",
        f"The parsimonious protocol-adjusted feasibility proxy has {int(parsimonious_design['columns'])} columns, rank {int(parsimonious_design['rank'])}, residual df {int(parsimonious_design['residual_df'])}, and column-normalized condition number {_fmt(parsimonious_design['normalized_condition_number'], 2)}. The draft-sized cubic phase+protocol proxy has {int(cubic_design['columns'])} columns, rank {int(cubic_design['rank'])}; the cubic site-fixed stress design has {int(site_design['columns'])} columns and rank {int(site_design['rank'])}. Rank deficiency in the latter designs proves that a saturated implementation cannot be used unchanged. These are outcome-blind proxy diagnostics; they do not freeze the final natural-spline basis or replace a mixed model.",
        "- Exact rank cause: the ADNI 2 indicator is identical to the `GE MEDICAL SYSTEMS|3T|41dir` protocol indicator (36 subjects: CN 3, MCI 0, AD 33). Phase plus protocol therefore contains a deterministic duplicate column in the full design.",
        "",
    ]
    for row in full_estimability.itertuples(index=False):
        lines.append(
            f"- {row.contrast.replace('_vs_', ' vs ')}: **{row.status}**; "
            f"{row.common_site_protocol_cells} shared site×protocol cells, with "
            f"{row.common_supported_n_left}/{row.common_supported_n_right} supported subjects and "
            f"{row.site_protocol_cells_min_5_each} cells reaching >=5 per compared group."
        )
    lines.extend(
        [
            "",
            "A returned coefficient is not sufficient for a cell-specific claim. Rows marked `NOT_ESTIMABLE_*` in the estimability CSV lack one group or lack algebraic/common-cell support. `SUPPORT_FRAGILE` and `LIMITED_CELL_DEPTH` contrasts may be computed only as qualified aggregate stress tests; they are not stand-alone acquisition-stratum discoveries.",
            "",
            "## Detectable-effect floor and attrition stress",
            "",
            "The MDE values below are standardized residual-scale planning bounds for 80% power using the conservative three-contrast threshold alpha=0.05/3. They are not expected effects. The naive values ignore covariates and clustering; the adjusted values use the complete-case parsimonious protocol proxy. Both are optimistic if corrected QC failures are differential or site clustering is strong.",
            "",
            "| Contrast | Naive complete-case MDE | Parsimonious protocol proxy MDE | Median MDE under metadata stress |",
            "|---|---:|---:|---:|",
        ]
    )
    for contrast in ("CN_vs_MCI", "MCI_vs_AD", "CN_vs_AD"):
        baseline = baseline_power.loc[contrast]
        stress = stress_power.loc[contrast]
        lines.append(
            f"| {contrast.replace('_vs_', ' vs ')} | {_fmt(float(baseline.naive_standardized_mde_median))} | "
            f"{_fmt(float(baseline.adjusted_complete_case_mde_parsimonious_protocol_proxy))} | "
            f"{_fmt(float(stress.naive_standardized_mde_median))} |"
        )
    lines.extend(
        [
            "",
            "Actual corrected-run attrition is currently unknown: corrected QC is `not_run` for 530/530. The simulation scenarios are hypothetical stress tests, not predicted failure rates. Legacy QC is deliberately excluded from the prospective gate because it predates the corrected recipe.",
            "",
            "## Fixed decisions recommended for SL-H03/SAP",
            "",
            "- Keep the full 515-person CN/MCI/AD analysis exploratory and the 530-person denominator visible in every attrition table.",
            "- Use a parsimonious acquisition representation; do not enter phase, protocol, manufacturer, gradient count, and site as mutually redundant saturated fixed effects.",
            "- Retain site/scanner partial pooling only when the fitted outcome model converges and publish the fixed-design rank/condition checks alongside it.",
            "- Treat CN–MCI as the best-supported contrast. Calibrate AD contrasts to their limited common support and wider detectable-effect floor.",
            "- Run <=90, <=180, continuous-gap, Siemens-54, site-overlap, Original-T1, processed-T1, and QC sensitivities exactly as declared. Do not present <=90 and <=180 as independent replications.",
            "- Mark cell-specific comparisons with an absent diagnosis as non-estimable. Do not drop nuisance terms or merge acquisition cells after looking at biological results merely to obtain significance.",
            "- Freeze outcome definitions, hard-validity rules, quantitative QC, contrast coding, spline knots, random seeds, and multiplicity before corrected group outcomes are opened.",
            "",
            "## Manifest-to-workflow contract feasibility",
            "",
            f"The current `{workflow_evidence['recipe_id']}` configuration is **not executable against the approved 530-row manifest as written**. The contract audit found {contract_counts.get('present', 0)} directly mapped required fields, {contract_counts.get('derivable_from_local_headers', 0)} header-derived fields still requiring extraction, {contract_counts.get('derivable_from_local_files', 0)} staging/hash fields still requiring construction, {contract_counts.get('unavailable', 0)} unavailable freeze values, and {contract_counts.get('policy_conflicting', 0)} policy-conflicting field/policy rows ({len(policy_conflicts)} are explicit policy clauses).",
            "",
            "Material conflicts are: the configured acquisition-manifest path does not point to the approved SL-D01 release; the workflow treats <=90 days as the only primary window and rejects >180-day pairs; it requires Original T1 despite 225 available processed T1 inputs; its T1 wording assumes visit-matched native Original anatomy; and it hard-rejects the 15 SMC subjects that SL-H01 requires in technical processing/attrition. These are pre-canary contract blockers, not reasons to discard local data.",
            "",
            "DTI SeriesInstanceUID, scanner model, phase-encoding direction, and total readout time are absent from the current manifest and may be extractable from authoritative local headers, but coverage is unproven until a rowwise, fail-closed metadata pass runs. Original T1 DICOM sources may expose a UID; processed T1 NIfTI sources do not have an authoritative DICOM UID carrier in the historical header proxy. Total readout time must not be guessed. Staged NIfTI/gradient/JSON paths and their exact hashes also require a write-once source-staging manifest. The workflow contract CSV records each field, source, status, and corrective action.",
            "",
            f"A separately checksum-bound **historical derivative proxy** (not locked raw-source evidence) found PE direction plus TotalReadoutTime in {workflow_evidence['historical_source_preflight_proxy']['counts']['pe_and_readout_complete']}/530 selected-ID MIF headers, with {workflow_evidence['historical_source_preflight_proxy']['counts']['pe_or_readout_missing']} missing; {workflow_evidence['historical_source_preflight_proxy']['counts']['unique_directions_lt_30']} gradient sets had fewer than 30 antipodally canonicalized unique directions, {workflow_evidence['historical_source_preflight_proxy']['counts']['multiple_nonzero_shells']} were multi-shell, and {workflow_evidence['historical_source_preflight_proxy']['counts']['b0_vector_norm_issue']} had a nonzero b<=50 vector under the strict proxy check. It also found no UID-like text in 225/225 processed-T1 NIfTI headers. These are preflight warnings only: raw-source extraction and conversion-time gradient checks can confirm, resolve, or supersede them.",
            "",
            "Because the header-only proxy found no UID-like text in the processed T1 NIfTI inputs, the contract must not assume that a DICOM UID is recoverable for them. Use nullable `t1_dicom_series_uid` plus a mandatory stable `t1_source_id` formed from the ADNI Image ID and locked source-bundle hash. A UID cannot be fabricated, and lack of a UID is not a reason to discard an otherwise approved local input.",
            "",
            "Required action before canary: issue a new recipe/config version aligned with SL-H01, create a checksum-bound workflow projection of all 530 processing attempts, populate/validate rowwise source metadata and staged-file hashes, and use analysis selectors—not source rejection—to define the 515-person primary set and timing/T1-type sensitivities.",
            "",
            "## Provenance and safety",
            "",
            f"- Pair manifest SHA-256: `{evidence['manifest']['sha256']}`.",
            f"- Diagnosis reconciliation SHA-256: `{evidence['diagnosis']['sha256']}`.",
            f"- SL-H01 decision SHA-256: `{evidence['decision']['sha256']}`.",
            f"- Workflow config SHA-256: `{workflow_evidence['sha256']}`.",
            f"- Historical source-preflight proxy validation SHA-256: `{workflow_evidence['historical_source_preflight_proxy']['sha256']}`; release role `{workflow_evidence['historical_source_preflight_proxy']['release_status']}`.",
            "- Core feasibility run image paths read: 0. Its upstream historical proxy read 530 MIF headers and 225 processed-T1 NIfTI headers, but loaded 0 voxel arrays.",
            "- Image files processed, transferred, modified, or deleted: 0.",
            "- Biological outcomes inspected: 0. This memo makes no claim about CN/MCI/AD connectome differences.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_dataframe(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, lineterminator="\n")


def run_analysis(
    manifest_path: Path,
    manifest_validation_path: Path,
    diagnosis_path: Path,
    decision_path: Path,
    workflow_config_path: Path,
    historical_proxy_validation_path: Path,
    output_dir: Path,
    figure_dir: Path,
    generated_utc: str | None = None,
) -> dict[str, Any]:
    generated = generated_utc or datetime.now(timezone.utc).isoformat()
    frame, evidence = validate_inputs(
        manifest_path, manifest_validation_path, diagnosis_path, decision_path
    )
    historical_proxy = load_historical_proxy_validation(
        historical_proxy_validation_path, evidence["manifest"]["sha256"]
    )
    masks = population_masks(frame)
    populations = population_summary(frame, masks)
    feasibility = feasibility_cells(frame, masks)
    support = common_support(frame, masks)
    design, contrast_lookup = design_diagnostics(frame, masks)
    estimability = estimability_table(frame, masks, support, contrast_lookup)
    power = power_attrition(frame, masks, contrast_lookup)
    workflow_contract, workflow_evidence = workflow_contract_feasibility(
        frame, workflow_config_path, historical_proxy
    )

    required_counts = {
        row.population: int(row.n_total) for row in populations.itertuples(index=False)
    }
    expected_counts = {
        "all_available": 530,
        "primary_full": 515,
        "timing_le_90": 194,
        "timing_le_180": 197,
        "protocol_siemens_3t_54": 260,
        "timing_le90_siemens_3t_54": 94,
        "timing_le90_siemens_3t_54_sites_ge2groups": 75,
        "t1_original": 305,
        "t1_processed": 210,
        "same_adni_phase": 333,
    }
    if required_counts != expected_counts:
        raise ValueError(f"prespecified population counts changed: {required_counts}")

    exact_support = support[
        (support["population"] == "primary_full")
        & (support["cell_family"] == "timing_site_protocol_t1_type")
        & (support["support_scope"] == "ALL3")
    ].iloc[0]
    if int(exact_support["levels_min_1"]) != 13 or int(exact_support["levels_min_5"]) != 0:
        raise ValueError("exact joint common-support invariant changed")
    full_cubic = design[
        (design["population"] == "primary_full")
        & (design["specification"] == "cubic_phase_protocol")
    ].iloc[0]
    duplicate_pairs = json.loads(full_cubic["exact_duplicate_columns_json"])
    expected_alias = [
        "phase[ADNI 2]",
        "protocol[GE MEDICAL SYSTEMS|3T|41dir]",
    ]
    if expected_alias not in duplicate_pairs:
        raise ValueError("expected ADNI2/GE41 exact design alias is absent")
    timing_t1_expected = {
        ("le_90_days", "Original"): [102, 58, 21],
        ("le_90_days", "Processed"): [4, 2, 7],
        ("days_91_180", "Processed"): [0, 0, 3],
        ("gt_180_days", "Original"): [55, 47, 22],
        ("gt_180_days", "Processed"): [90, 79, 25],
    }
    timing_rows = feasibility[
        (feasibility["population"] == "primary_full")
        & (feasibility["cell_family"] == "timing_t1_type")
    ]
    observed_timing_t1: dict[tuple[str, str], list[int]] = {}
    for key, group_rows in timing_rows.groupby("cell_key"):
        values = json.loads(key)
        observed_timing_t1[
            (values["timing_stratum"], values["current_t1_type"])
        ] = [
            int(
                group_rows.loc[group_rows["group"] == group, "n"].iloc[0]
            )
            for group in GROUPS
        ]
    if observed_timing_t1 != timing_t1_expected:
        raise ValueError(f"timing-by-T1 overlap changed: {observed_timing_t1}")

    if output_dir.exists() and any(output_dir.glob("cohort_v2_*")):
        raise FileExistsError(f"refusing to overwrite existing feasibility release in {output_dir}")
    if (output_dir / OUTPUT_MEMO).exists():
        raise FileExistsError(f"refusing to overwrite {output_dir / OUTPUT_MEMO}")
    if any((figure_dir / name).exists() for name in (FIG_SUPPORT, FIG_POWER)):
        raise FileExistsError(f"refusing to overwrite feasibility figures in {figure_dir}")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    figure_dir.parent.mkdir(parents=True, exist_ok=True)
    stage_root = Path(tempfile.mkdtemp(prefix=".fixed-data-feasibility-", dir=output_dir.parent))
    stage_output = stage_root / "outputs"
    stage_figures = stage_root / "figures"
    stage_output.mkdir()
    stage_figures.mkdir()
    try:
        tables = {
            OUTPUT_FEASIBILITY: feasibility,
            OUTPUT_POPULATIONS: populations,
            OUTPUT_SUPPORT: support,
            OUTPUT_DESIGN: design,
            OUTPUT_ESTIMABILITY: estimability,
            OUTPUT_POWER: power,
            OUTPUT_WORKFLOW_CONTRACT: workflow_contract,
        }
        for name, table in tables.items():
            _write_dataframe(table, stage_output / name)
        save_plots(populations, support, power, stage_figures)
        memo = build_memo(
            generated,
            populations,
            feasibility,
            support,
            design,
            estimability,
            power,
            workflow_contract,
            workflow_evidence,
            evidence,
        )
        (stage_output / OUTPUT_MEMO).write_text(memo, encoding="utf-8")

        outputs: dict[str, Any] = {}
        for name in [*tables, OUTPUT_MEMO]:
            path = stage_output / name
            outputs[name] = {
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
                "rows": len(tables[name]) if name in tables else None,
            }
        figures: dict[str, Any] = {}
        for name in (FIG_SUPPORT, FIG_POWER):
            path = stage_figures / name
            figures[name] = {
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        status_counts = (
            estimability["status"].value_counts().sort_index().astype(int).to_dict()
        )
        validation = {
            "schema_version": "1.0",
            "status": "PASS",
            "release_status": "FEASIBILITY_COMPLETE_NO_OUTCOMES_OPENED",
            "generated_utc": generated,
            "random_seed": RANDOM_SEED,
            "simulation_replicates": SIMULATIONS,
            "power_assumptions": {
                "target_power": TARGET_POWER,
                "family_alpha": ALPHA_FAMILY,
                "primary_contrasts": N_PRIMARY_CONTRASTS,
                "threshold": "two-sided alpha=0.05/3 planning bound",
                "limitations": "independent homoscedastic Gaussian residual approximation; no clustering/design uncertainty",
            },
            "inputs": evidence,
            "counts": {
                "processing_denominator": 530,
                "primary_ceiling": 515,
                "smc_descriptive": 15,
                "population_rows": len(populations),
                "feasibility_cell_rows": len(feasibility),
                "support_rows": len(support),
                "design_rows": len(design),
                "estimability_rows": len(estimability),
                "power_attrition_rows": len(power),
                "workflow_contract_rows": len(workflow_contract),
                "workflow_contract_blocking_rows": int(
                    workflow_contract["blocking_before_canary"].astype(bool).sum()
                ),
                "exact_joint_all_three_cells": int(exact_support["levels_min_1"]),
                "exact_joint_all_three_cells_min_5": int(exact_support["levels_min_5"]),
                "timing_t1_type_cells": len(observed_timing_t1),
                "estimability_status_counts": status_counts,
            },
            "checks": {
                "sl_h01_approved": True,
                "manifest_checksum_bound": True,
                "diagnosis_checksum_bound": True,
                "population_counts_match_prespecified": True,
                "all_530_available_for_attempt": True,
                "corrected_qc_outcome_blind_not_run": True,
                "exact_group_timing_site_protocol_t1_cells_emitted": True,
                "common_support_thresholds_emitted": True,
                "rank_and_condition_diagnostics_emitted": True,
                "estimable_and_nonestimable_contrasts_emitted": True,
                "attrition_stress_is_hypothetical": True,
                "workflow_contract_feasibility_emitted": True,
                "workflow_config_policy_conflicts_detected": bool(
                    (
                        workflow_contract["classification"]
                        == "policy_conflicting"
                    ).any()
                ),
                "phase_protocol_exact_alias_detected": True,
                "historical_proxy_checksum_bound": True,
                "processed_t1_uid_unavailable_not_fabricated": True,
            },
            "safety": {
                "core_run_image_files_read": 0,
                "connectome_files_read": 0,
                "biological_outcomes_read": 0,
                "upstream_historical_mif_headers_read": historical_proxy["safety"].get(
                    "historical_mif_headers_read", 0
                ),
                "upstream_processed_t1_headers_read": historical_proxy["safety"].get(
                    "processed_t1_headers_read", 0
                ),
                "upstream_voxel_arrays_read": historical_proxy["safety"].get(
                    "raw_source_voxel_arrays_read", 0
                )
                + historical_proxy["safety"].get(
                    "historical_derivative_voxel_arrays_read", 0
                ),
                "images_processed": 0,
                "images_transferred": 0,
                "images_modified_or_deleted": 0,
                "production_paths_modified": 0,
            },
            "outputs": outputs,
            "figures": figures,
            "workflow_contract": workflow_evidence,
        }
        validation_path = stage_output / OUTPUT_VALIDATION
        validation_path.write_text(
            json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        output_dir.mkdir(parents=True, exist_ok=True)
        figure_dir.mkdir(parents=True, exist_ok=True)
        for path in stage_output.iterdir():
            os.replace(path, output_dir / path.name)
        for path in stage_figures.iterdir():
            os.replace(path, figure_dir / path.name)
        return validation
    finally:
        shutil.rmtree(stage_root, ignore_errors=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--manifest-validation", type=Path, default=DEFAULT_MANIFEST_VALIDATION
    )
    parser.add_argument("--diagnosis", type=Path, default=DEFAULT_DIAGNOSIS)
    parser.add_argument("--decision", type=Path, default=DEFAULT_DECISION)
    parser.add_argument(
        "--workflow-config", type=Path, default=DEFAULT_WORKFLOW_CONFIG
    )
    parser.add_argument(
        "--historical-proxy-validation",
        type=Path,
        default=DEFAULT_HISTORICAL_PROXY_VALIDATION,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURES)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    validation = run_analysis(
        args.manifest,
        args.manifest_validation,
        args.diagnosis,
        args.decision,
        args.workflow_config,
        args.historical_proxy_validation,
        args.output_dir,
        args.figure_dir,
    )
    print(
        json.dumps(
            {
                "status": validation["status"],
                "release_status": validation["release_status"],
                "validation": str((args.output_dir / OUTPUT_VALIDATION).resolve()),
                "memo": str((args.output_dir / OUTPUT_MEMO).resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
