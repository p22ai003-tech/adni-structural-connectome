#!/usr/bin/env python3
"""Reconcile diagnosis and DTI-series identity for the fixed local cohort.

The selected series in the connectome manifest is the analysis identity.  The
dashboard metadata row is retained only as a cross-check because, for some
subjects, it names a processed derivative rather than the Original DTI series
that was actually processed.  Diagnosis is resolved at the selected DTI date
from local catalog metadata and is never inferred from the dashboard's broad
CN/MCI/AD label alone.

This builder is intentionally closed-world and non-destructive.  It reads only
the supplied local CSV files, writes new outputs atomically, and refuses to
overwrite an existing result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import numbers
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


SCHEMA_VERSION = "2.0.0"
SUBJECT_RE = re.compile(r"^\d{3}_S_\d{4}$")
KNOWN_DIAGNOSES = {"CN", "MCI", "EMCI", "LMCI", "AD", "SMC"}
MCI_SUBTYPES = {"MCI", "EMCI", "LMCI"}

REQUIRED_COHORT_COLUMNS = {
    "subject_id",
    "actual_dti_image_id",
    "Image ID",
    "group_dashboard",
    "dti_study_date",
    "Description",
    "Type",
}
REQUIRED_DTI_COLUMNS = {
    "Subject ID",
    "Phase",
    "Sex",
    "Research Group",
    "Visit",
    "Study Date",
    "Age",
    "Description",
    "Type",
    "Image ID",
}
REQUIRED_MRI_COLUMNS = {"Subject ID", "Research Group", "Study Date"}

OUTPUT_COLUMNS = [
    "subject_id",
    "selected_dti_image_id",
    "dashboard_dti_image_id",
    "dti_study_date",
    "dti_visit",
    "dti_phase",
    "dti_age",
    "dti_sex",
    "dti_description",
    "dti_type",
    "diagnosis_raw_at_dti",
    "diagnosis_harmonized",
    "diagnosis_harmonization_rule",
    "diagnosis_resolution_status",
    "diagnosis_resolution_method",
    "diagnosis_source_catalog",
    "diagnosis_source_catalog_sha256",
    "diagnosis_source_data_row",
    "diagnosis_source_image_id",
    "diagnosis_source_date",
    "diagnosis_day_offset",
    "diagnosis_is_exact_dti_date",
    "primary_analysis_group",
    "primary_analysis_eligible",
    "smc_retained_separately",
    "same_date_evidence_labels",
    "same_date_evidence_sources",
    "same_date_evidence_rows",
    "same_date_diagnosis_conflict",
    "dashboard_group",
    "dashboard_group_agrees_with_harmonized_diagnosis",
    "dti_identity_mismatch",
    "dti_identity_resolution_status",
    "identity_subject_match",
    "identity_date_match",
    "identity_visit_match",
    "identity_age_match",
    "identity_diagnosis_match",
    "dashboard_dti_description",
    "dashboard_dti_type",
    "selected_dti_is_original_axial_dti",
    "dashboard_row_is_processed_dti_derivative",
    "cohort_manifest_source",
    "cohort_manifest_sha256",
    "dti_catalog_source",
    "dti_catalog_sha256",
    "mri_crosscheck_source",
    "mri_crosscheck_sha256",
]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_positive_integer(value: Any) -> int | None:
    """Normalize IDs without accepting scientific notation or lossy floats."""

    if pd.isna(value) or isinstance(value, bool):
        return None
    if isinstance(value, numbers.Integral):
        number = int(value)
        return number if number > 0 else None
    if isinstance(value, numbers.Real):
        number = float(value)
        if (
            not math.isfinite(number)
            or not number.is_integer()
            or number <= 0
            or abs(number) > 2**53
        ):
            return None
        return int(number)
    match = re.fullmatch(r"[Ii]?(\d+)", str(value).strip())
    if not match:
        return None
    number = int(match.group(1))
    return number if number > 0 else None


def positive_ids(values: pd.Series, label: str) -> pd.Series:
    parsed = values.map(normalize_positive_integer)
    bad = parsed.isna()
    if bad.any():
        raise ValueError(
            f"{label} contains invalid positive integer IDs: "
            f"{values.loc[bad].head(10).tolist()}"
        )
    return parsed.astype(int)


def normalize_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def normalize_diagnosis(value: Any) -> str | None:
    text = normalize_text(value).upper()
    return text if text in KNOWN_DIAGNOSES else None


def harmonize_diagnosis(value: str | None) -> tuple[str, str, str, bool, bool]:
    """Return harmonized label, rule, primary label, eligibility, SMC flag."""

    if value in {"CN", "AD"}:
        return value, "identity", value, True, False
    if value in MCI_SUBTYPES:
        rule = "identity" if value == "MCI" else f"explicit_{value}_to_MCI"
        return "MCI", rule, "MCI", True, False
    if value == "SMC":
        return "SMC", "retain_SMC_distinct_from_MCI", "", False, True
    return "UNKNOWN", "no_supported_local_diagnosis", "", False, False


def parse_dates(values: pd.Series) -> pd.Series:
    # format="mixed" is required by current pandas for the historical mixture
    # of ISO and M/D/YYYY values in the local ADNI exports.
    return pd.to_datetime(values, format="mixed", errors="coerce").dt.normalize()


def json_labels(values: Iterable[str]) -> str:
    return json.dumps(sorted(set(values)), separators=(",", ":"))


def _source_records(
    dti: pd.DataFrame,
    mri: pd.DataFrame,
) -> pd.DataFrame:
    frames = []
    for frame, name, priority in (
        (dti, "cohort/dti_master.csv", 0),
        (mri, "cohort/mri_master.csv", 1),
    ):
        part = pd.DataFrame(
            {
                "subject_id": frame["Subject ID"].map(normalize_text),
                "date": frame["_study_date"],
                "diagnosis": frame["Research Group"].map(normalize_diagnosis),
                "source_catalog": name,
                "source_priority": priority,
                "source_data_row": frame["_source_data_row"],
                "source_image_id": frame.get("_image_id", pd.Series(pd.NA, index=frame.index)),
            }
        )
        frames.append(part)
    records = pd.concat(frames, ignore_index=True)
    return records[
        records["subject_id"].str.fullmatch(SUBJECT_RE)
        & records["date"].notna()
        & records["diagnosis"].notna()
    ].copy()


def _choose_nearest_record(
    records: pd.DataFrame,
    subject_id: str,
    dti_date: pd.Timestamp,
) -> tuple[pd.Series | None, list[str], int | None]:
    candidates = records[records["subject_id"].eq(subject_id)].copy()
    if candidates.empty or pd.isna(dti_date):
        return None, [], None
    candidates["day_offset_signed"] = (candidates["date"] - dti_date).dt.days
    candidates["day_offset_abs"] = candidates["day_offset_signed"].abs()
    minimum = int(candidates["day_offset_abs"].min())
    nearest = candidates[candidates["day_offset_abs"].eq(minimum)].copy()
    labels = sorted(nearest["diagnosis"].dropna().unique())
    if len(labels) != 1:
        return None, labels, minimum
    chosen = nearest[nearest["diagnosis"].eq(labels[0])].sort_values(
        ["source_priority", "source_data_row"], kind="mergesort"
    ).iloc[0]
    return chosen, labels, minimum


def build_reconciliation(
    cohort: pd.DataFrame,
    dti_master: pd.DataFrame,
    mri_master: pd.DataFrame,
    *,
    expected_subjects: int | None = 530,
    source_paths: dict[str, str] | None = None,
    source_hashes: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build the 1:1 diagnosis/identity reconciliation and validation record."""

    missing_cohort = sorted(REQUIRED_COHORT_COLUMNS - set(cohort.columns))
    missing_dti = sorted(REQUIRED_DTI_COLUMNS - set(dti_master.columns))
    missing_mri = sorted(REQUIRED_MRI_COLUMNS - set(mri_master.columns))
    if missing_cohort or missing_dti or missing_mri:
        raise ValueError(
            "Missing required columns: "
            f"cohort={missing_cohort}, dti_master={missing_dti}, "
            f"mri_master={missing_mri}"
        )

    paths = {
        "cohort": "research_audit/outputs/cohort_audit_manifest.csv",
        "dti": "cohort/dti_master.csv",
        "mri": "cohort/mri_master.csv",
    }
    if source_paths:
        paths.update(source_paths)
    hashes = {"cohort": "", "dti": "", "mri": ""}
    if source_hashes:
        hashes.update(source_hashes)

    selected = cohort.copy().reset_index(drop=True)
    selected["subject_id"] = selected["subject_id"].map(normalize_text)
    bad_subjects = ~selected["subject_id"].str.fullmatch(SUBJECT_RE)
    if bad_subjects.any():
        raise ValueError(
            f"Invalid cohort subject IDs: {selected.loc[bad_subjects, 'subject_id'].head(10).tolist()}"
        )
    if selected["subject_id"].duplicated().any():
        raise ValueError("Cohort contains duplicate subject_id rows")
    selected["_selected_id"] = positive_ids(
        selected["actual_dti_image_id"], "actual_dti_image_id"
    )
    selected["_dashboard_id"] = positive_ids(selected["Image ID"], "Image ID")
    if selected["_selected_id"].duplicated().any():
        raise ValueError("Cohort contains duplicate selected DTI Image IDs")
    if expected_subjects is not None and len(selected) != expected_subjects:
        raise ValueError(
            f"Expected {expected_subjects} unique subjects, found {len(selected)}"
        )

    dti = dti_master.copy().reset_index(drop=True)
    dti["_source_data_row"] = dti.index + 1
    dti["_image_id"] = positive_ids(dti["Image ID"], "dti_master Image ID")
    dti["_study_date"] = parse_dates(dti["Study Date"])
    mri = mri_master.copy().reset_index(drop=True)
    mri["_source_data_row"] = mri.index + 1
    mri["_study_date"] = parse_dates(mri["Study Date"])
    if "Image ID" in mri:
        mri["_image_id"] = mri["Image ID"].map(normalize_positive_integer)
    else:
        mri["_image_id"] = pd.NA

    dti_by_id = {key: group for key, group in dti.groupby("_image_id", sort=False)}
    evidence = _source_records(dti, mri)
    evidence_by_subject_date = {
        key: group
        for key, group in evidence.groupby(["subject_id", "date"], sort=False)
    }

    rows: list[dict[str, Any]] = []
    for _, cohort_row in selected.sort_values("subject_id", kind="mergesort").iterrows():
        subject_id = cohort_row["subject_id"]
        selected_id = int(cohort_row["_selected_id"])
        dashboard_id = int(cohort_row["_dashboard_id"])
        selected_rows = dti_by_id.get(selected_id, pd.DataFrame())
        dashboard_rows = dti_by_id.get(dashboard_id, pd.DataFrame())

        direct: pd.Series | None = None
        direct_ambiguous = False
        if len(selected_rows) == 1:
            direct = selected_rows.iloc[0]
        elif len(selected_rows) > 1:
            comparison = [
                "Subject ID",
                "Study Date",
                "Visit",
                "Research Group",
                "Phase",
                "Age",
                "Sex",
                "Description",
                "Type",
            ]
            direct_ambiguous = any(
                selected_rows[column].map(normalize_text).nunique(dropna=False) > 1
                for column in comparison
            )
            if not direct_ambiguous:
                direct = selected_rows.sort_values("_source_data_row").iloc[0]

        cohort_date = parse_dates(pd.Series([cohort_row["dti_study_date"]])).iloc[0]
        direct_date = direct["_study_date"] if direct is not None else pd.NaT
        date_conflict = (
            pd.notna(cohort_date)
            and pd.notna(direct_date)
            and cohort_date != direct_date
        )
        dti_date = direct_date if pd.notna(direct_date) else cohort_date
        direct_subject = normalize_text(direct["Subject ID"]) if direct is not None else ""
        subject_conflict = bool(direct is not None and direct_subject != subject_id)

        same_date = evidence_by_subject_date.get((subject_id, dti_date), pd.DataFrame())
        same_labels = (
            sorted(same_date["diagnosis"].dropna().unique())
            if not same_date.empty
            else []
        )
        same_sources = (
            sorted(same_date["source_catalog"].dropna().unique())
            if not same_date.empty
            else []
        )
        same_date_conflict = len(same_labels) > 1

        diagnosis_raw: str | None = None
        diagnosis_status = "UNKNOWN_NO_LOCAL_DIAGNOSIS"
        diagnosis_method = "none"
        source_catalog = ""
        source_row: int | str = ""
        source_image_id: int | str = ""
        source_date = pd.NaT
        day_offset: int | str = ""

        direct_diagnosis = (
            normalize_diagnosis(direct["Research Group"]) if direct is not None else None
        )
        if direct_ambiguous:
            diagnosis_status = "UNKNOWN_AMBIGUOUS_SELECTED_IMAGE_ID"
        elif subject_conflict:
            diagnosis_status = "UNKNOWN_SELECTED_IMAGE_SUBJECT_CONFLICT"
        elif date_conflict:
            diagnosis_status = "UNKNOWN_SELECTED_IMAGE_DATE_CONFLICT"
        elif same_date_conflict:
            diagnosis_status = "UNKNOWN_SAME_DATE_DIAGNOSIS_CONFLICT"
        elif direct is not None and direct_diagnosis is not None and pd.notna(direct_date):
            diagnosis_raw = direct_diagnosis
            diagnosis_status = "RESOLVED_EXACT_DTI_IMAGE_ID_DATE"
            diagnosis_method = "selected_dti_image_id_exact_date"
            source_catalog = paths["dti"]
            source_row = int(direct["_source_data_row"])
            source_image_id = selected_id
            source_date = direct_date
            day_offset = 0
        elif len(same_labels) == 1:
            chosen = same_date.sort_values(
                ["source_priority", "source_data_row"], kind="mergesort"
            ).iloc[0]
            diagnosis_raw = same_labels[0]
            diagnosis_status = "RESOLVED_EXACT_DTI_DATE_CONSENSUS"
            diagnosis_method = "same_subject_same_date_local_catalog_consensus"
            source_catalog = paths["dti"] if chosen["source_priority"] == 0 else paths["mri"]
            source_row = int(chosen["source_data_row"])
            source_image_id = (
                int(chosen["source_image_id"])
                if pd.notna(chosen["source_image_id"])
                else ""
            )
            source_date = chosen["date"]
            day_offset = 0
        else:
            chosen, nearest_labels, nearest_gap = _choose_nearest_record(
                evidence, subject_id, dti_date
            )
            if chosen is not None:
                diagnosis_raw = str(chosen["diagnosis"])
                diagnosis_status = "RESOLVED_NEAREST_LOCAL_DIAGNOSIS"
                diagnosis_method = "nearest_subject_dated_local_catalog_consensus"
                source_catalog = paths["dti"] if chosen["source_priority"] == 0 else paths["mri"]
                source_row = int(chosen["source_data_row"])
                source_image_id = (
                    int(chosen["source_image_id"])
                    if pd.notna(chosen["source_image_id"])
                    else ""
                )
                source_date = chosen["date"]
                day_offset = int(chosen["day_offset_signed"])
            elif nearest_gap is not None and len(nearest_labels) > 1:
                diagnosis_status = "UNKNOWN_NEAREST_DATE_DIAGNOSIS_CONFLICT"

        harmonized, harmonization_rule, primary_group, primary_ok, smc_flag = (
            harmonize_diagnosis(diagnosis_raw)
        )

        display: pd.Series | None = None
        display_ambiguous = False
        if len(dashboard_rows) == 1:
            display = dashboard_rows.iloc[0]
        elif len(dashboard_rows) > 1:
            display_ambiguous = True
        identity_mismatch = dashboard_id != selected_id
        actual_original = bool(
            direct is not None
            and normalize_text(direct["Type"]).lower() == "original"
            and normalize_text(direct["Description"]).lower() == "axial dti"
        )
        display_processed = bool(
            display is not None
            and normalize_text(display["Type"]).lower() == "processed"
            and "dti" in normalize_text(display["Description"]).lower()
            and "<-" in normalize_text(display["Description"])
        )

        def equal_text(column: str) -> bool:
            return bool(
                direct is not None
                and display is not None
                and normalize_text(direct[column]) == normalize_text(display[column])
            )

        subject_match = equal_text("Subject ID")
        visit_match = equal_text("Visit")
        age_match = equal_text("Age")
        dx_match = bool(
            direct is not None
            and display is not None
            and normalize_diagnosis(direct["Research Group"])
            == normalize_diagnosis(display["Research Group"])
            and normalize_diagnosis(direct["Research Group"]) is not None
        )
        date_match = bool(
            direct is not None
            and display is not None
            and pd.notna(direct["_study_date"])
            and pd.notna(display["_study_date"])
            and direct["_study_date"] == display["_study_date"]
        )
        if not identity_mismatch and direct is not None and not direct_ambiguous:
            identity_status = "CONSISTENT_SELECTED_IMAGE_ID"
        elif (
            identity_mismatch
            and not display_ambiguous
            and actual_original
            and display_processed
            and subject_match
            and date_match
            and visit_match
            and age_match
            and dx_match
        ):
            identity_status = "RESOLVED_PROCESSED_DERIVATIVE_TO_ORIGINAL_DTI"
        else:
            identity_status = "UNRESOLVED_DTI_IDENTITY_MISMATCH"

        dashboard_group = normalize_text(cohort_row["group_dashboard"]).upper()
        dashboard_agrees = bool(harmonized != "UNKNOWN" and dashboard_group == harmonized)
        source_hash = hashes["dti"] if source_catalog == paths["dti"] else (
            hashes["mri"] if source_catalog == paths["mri"] else ""
        )
        row = {
            "subject_id": subject_id,
            "selected_dti_image_id": selected_id,
            "dashboard_dti_image_id": dashboard_id,
            "dti_study_date": dti_date.date().isoformat() if pd.notna(dti_date) else "",
            "dti_visit": normalize_text(direct["Visit"]) if direct is not None else "",
            "dti_phase": normalize_text(direct["Phase"]) if direct is not None else "",
            "dti_age": direct["Age"] if direct is not None else "",
            "dti_sex": normalize_text(direct["Sex"]) if direct is not None else "",
            "dti_description": normalize_text(direct["Description"]) if direct is not None else "",
            "dti_type": normalize_text(direct["Type"]) if direct is not None else "",
            "diagnosis_raw_at_dti": diagnosis_raw or "UNKNOWN",
            "diagnosis_harmonized": harmonized,
            "diagnosis_harmonization_rule": harmonization_rule,
            "diagnosis_resolution_status": diagnosis_status,
            "diagnosis_resolution_method": diagnosis_method,
            "diagnosis_source_catalog": source_catalog,
            "diagnosis_source_catalog_sha256": source_hash,
            "diagnosis_source_data_row": source_row,
            "diagnosis_source_image_id": source_image_id,
            "diagnosis_source_date": source_date.date().isoformat() if pd.notna(source_date) else "",
            "diagnosis_day_offset": day_offset,
            "diagnosis_is_exact_dti_date": bool(day_offset == 0),
            "primary_analysis_group": primary_group,
            "primary_analysis_eligible": primary_ok,
            "smc_retained_separately": smc_flag,
            "same_date_evidence_labels": json_labels(same_labels),
            "same_date_evidence_sources": json_labels(same_sources),
            "same_date_evidence_rows": len(same_date),
            "same_date_diagnosis_conflict": same_date_conflict,
            "dashboard_group": dashboard_group,
            "dashboard_group_agrees_with_harmonized_diagnosis": dashboard_agrees,
            "dti_identity_mismatch": identity_mismatch,
            "dti_identity_resolution_status": identity_status,
            "identity_subject_match": subject_match,
            "identity_date_match": date_match,
            "identity_visit_match": visit_match,
            "identity_age_match": age_match,
            "identity_diagnosis_match": dx_match,
            "dashboard_dti_description": normalize_text(display["Description"]) if display is not None else normalize_text(cohort_row["Description"]),
            "dashboard_dti_type": normalize_text(display["Type"]) if display is not None else normalize_text(cohort_row["Type"]),
            "selected_dti_is_original_axial_dti": actual_original,
            "dashboard_row_is_processed_dti_derivative": display_processed,
            "cohort_manifest_source": paths["cohort"],
            "cohort_manifest_sha256": hashes["cohort"],
            "dti_catalog_source": paths["dti"],
            "dti_catalog_sha256": hashes["dti"],
            "mri_crosscheck_source": paths["mri"],
            "mri_crosscheck_sha256": hashes["mri"],
        }
        rows.append(row)

    result = pd.DataFrame(rows, columns=OUTPUT_COLUMNS).sort_values(
        ["subject_id", "selected_dti_image_id"], kind="mergesort", ignore_index=True
    )
    unresolved_identity = result["dti_identity_resolution_status"].str.startswith(
        "UNRESOLVED"
    )
    unknown_diagnosis = result["diagnosis_harmonized"].eq("UNKNOWN")
    mismatch = result["dti_identity_mismatch"]
    status = "PASS" if not unresolved_identity.any() else "PASS_WITH_EXPLICIT_UNRESOLVED_IDENTITY"
    release_status = (
        "DIAGNOSIS_RECONCILED_FOR_AVAILABLE_DATA_MANIFEST"
        if not unresolved_identity.any() and not unknown_diagnosis.any()
        else "RECONCILIATION_REQUIRES_EXCLUSION_OR_SENSITIVITY"
    )
    validation = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "release_status": release_status,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "key": ["subject_id", "selected_dti_image_id"],
        "schema_columns": OUTPUT_COLUMNS,
        "counts": {
            "rows": len(result),
            "unique_subjects": int(result["subject_id"].nunique()),
            "unique_selected_dti_image_ids": int(result["selected_dti_image_id"].nunique()),
            "diagnosis_exact_date": int(result["diagnosis_is_exact_dti_date"].sum()),
            "diagnosis_nearest_date": int(result["diagnosis_resolution_status"].eq("RESOLVED_NEAREST_LOCAL_DIAGNOSIS").sum()),
            "diagnosis_unknown": int(unknown_diagnosis.sum()),
            "primary_analysis_eligible": int(result["primary_analysis_eligible"].sum()),
            "smc_retained_separately": int(result["smc_retained_separately"].sum()),
            "dti_identity_mismatches": int(mismatch.sum()),
            "dti_identity_mismatches_resolved": int((mismatch & ~unresolved_identity).sum()),
            "dti_identity_mismatches_unresolved": int((mismatch & unresolved_identity).sum()),
            "dashboard_group_disagreements": int((~result["dashboard_group_agrees_with_harmonized_diagnosis"]).sum()),
        },
        "diagnosis_raw_counts": {
            str(key): int(value)
            for key, value in result["diagnosis_raw_at_dti"].value_counts(dropna=False).sort_index().items()
        },
        "diagnosis_harmonized_counts": {
            str(key): int(value)
            for key, value in result["diagnosis_harmonized"].value_counts(dropna=False).sort_index().items()
        },
        "checks": {
            "one_row_per_subject_and_selected_dti": not result.duplicated(["subject_id", "selected_dti_image_id"]).any(),
            "all_selected_dti_ids_unique": not result["selected_dti_image_id"].duplicated().any(),
            "all_unknown_states_explicit": bool(
                result.loc[unknown_diagnosis, "diagnosis_resolution_status"].str.startswith("UNKNOWN").all()
            ),
            "smc_not_mapped_to_mci": bool(
                result.loc[result["diagnosis_raw_at_dti"].eq("SMC"), "diagnosis_harmonized"].eq("SMC").all()
            ),
            "all_identity_mismatches_explicit": bool(
                result.loc[mismatch, "dti_identity_resolution_status"].ne("").all()
            ),
            "all_primary_rows_have_cn_mci_ad": bool(
                result.loc[result["primary_analysis_eligible"], "primary_analysis_group"].isin(["CN", "MCI", "AD"]).all()
            ),
        },
        "policy": {
            "identity_key": "selected Original DTI Image ID from the connectome cohort manifest",
            "diagnosis_priority": [
                "selected DTI Image ID exact-date Research Group",
                "same-subject same-date local DTI/MRI catalog consensus",
                "nearest dated local DTI/MRI catalog consensus",
                "explicit UNKNOWN state",
            ],
            "mci_harmonization": "MCI, EMCI, and LMCI map explicitly to MCI",
            "smc_harmonization": "SMC remains SMC and is excluded from the CN/MCI/AD primary group",
            "dashboard_group_role": "cross-check only; never the sole diagnosis source",
            "nearest_date_rule": "no arbitrary time cutoff; signed day offset is retained for sensitivity/exclusion",
        },
        "sources": {
            key: {"path": paths[key], "sha256": hashes[key]}
            for key in ("cohort", "dti", "mri")
        },
    }
    return result, validation


def _write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", dir=path.parent, delete=False
    )
    temporary = Path(handle.name)
    try:
        with handle:
            frame.to_csv(handle, index=False, lineterminator="\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _write_json_atomic(payload: dict[str, Any], path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    )
    temporary = Path(handle.name)
    try:
        with handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def write_outputs(
    result: pd.DataFrame,
    validation: dict[str, Any],
    output_csv: Path,
    mismatch_csv: Path,
    validation_json: Path,
) -> None:
    for path in (output_csv, mismatch_csv, validation_json):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing output: {path}")
    _write_csv_atomic(result, output_csv)
    mismatches = result[result["dti_identity_mismatch"]].copy()
    _write_csv_atomic(mismatches, mismatch_csv)
    validation["outputs"] = {
        "diagnosis_reconciliation": {
            "path": str(output_csv),
            "sha256": file_sha256(output_csv),
            "rows": len(result),
        },
        "dti_identity_mismatch_resolution": {
            "path": str(mismatch_csv),
            "sha256": file_sha256(mismatch_csv),
            "rows": len(mismatches),
        },
    }
    _write_json_atomic(validation, validation_json)


def main() -> int:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cohort",
        type=Path,
        default=project / "research_audit/outputs/cohort_audit_manifest.csv",
    )
    parser.add_argument(
        "--dti-master", type=Path, default=project / "cohort/dti_master.csv"
    )
    parser.add_argument(
        "--mri-master", type=Path, default=project / "cohort/mri_master.csv"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project / "research_audit/outputs/diagnosis_reconciliation_v2.csv",
    )
    parser.add_argument(
        "--mismatch-output",
        type=Path,
        default=project
        / "research_audit/outputs/dti_identity_mismatch_resolution_v2.csv",
    )
    parser.add_argument(
        "--validation",
        type=Path,
        default=project
        / "research_audit/outputs/diagnosis_reconciliation_validation_v2.json",
    )
    args = parser.parse_args()

    inputs = {"cohort": args.cohort, "dti": args.dti_master, "mri": args.mri_master}
    for name, path in inputs.items():
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"{name} input must be an existing non-symlink file: {path}")
    paths = {
        key: str(path.relative_to(project)) if path.is_relative_to(project) else str(path)
        for key, path in inputs.items()
    }
    hashes = {key: file_sha256(path) for key, path in inputs.items()}
    result, validation = build_reconciliation(
        pd.read_csv(args.cohort),
        pd.read_csv(args.dti_master),
        pd.read_csv(args.mri_master),
        expected_subjects=530,
        source_paths=paths,
        source_hashes=hashes,
    )
    write_outputs(result, validation, args.output, args.mismatch_output, args.validation)
    print(
        json.dumps(
            {
                "status": validation["status"],
                "release_status": validation["release_status"],
                "rows": len(result),
                "output": str(args.output),
                "validation": str(args.validation),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
