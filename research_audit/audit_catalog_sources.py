#!/usr/bin/env python3
"""Audit exact-date coverage for the current and proposed DTI-T1 pairs.

This is a read-only source audit. It does not download images or alter the
production pipeline. Outputs are confined to research_audit/outputs plus the
catalog source report in research_audit.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT = Path("/home/ec2-user/exp")
AUDIT = PROJECT / "research_audit"
OUTPUT = AUDIT / "outputs"
MRI_ROOT = Path("/data/Images/mri")

DRY_RUN = OUTPUT / "visit_matched_t1_manifest_dryrun.csv"
DTI_MASTER = PROJECT / "cohort" / "dti_master.csv"
MRI_MASTER = PROJECT / "cohort" / "mri_master.csv"
ALL_MRI = PROJECT / "cohort" / "all_mri.csv"
REQUEST_ROSTER = OUTPUT / "exact_date_t1_candidate_request_v2.csv"
REQUEST_ROSTER_VALIDATION = (
    OUTPUT / "exact_date_t1_candidate_request_validation_v2.json"
)
AWS_PRINCIPAL_VERIFICATION = OUTPUT / "aws_aml_principal_verification.json"
AWS_CATALOG_SEARCH = OUTPUT / "aws_aml_catalog_search.json"

FROZEN_REQUEST_ROSTER_SHA256 = (
    "4a935b7bbd7d6d5ea5c1c8d80f4cc8fde840abaa88c0d4d2419878591bcc2406"
)
FROZEN_REQUEST_COUNTS = {
    "selected_subjects": 530,
    "request_rows": 4015,
    "unique_t1_image_ids": 4015,
    "subjects_with_candidates": 530,
    "rounded_age_screen_ids": 530,
    "rounded_age_screen_ids_in_request": 530,
    "provisional_replacement_ids": 330,
    "provisional_replacement_ids_in_request": 330,
    "current_t1_ids_in_request": 305,
    "minimum_candidates_per_subject": 1,
    "maximum_candidates_per_subject": 38,
}

IMAGE_DIR_RE = re.compile(r"^I(?P<image_id>\d+)$")
DATE_DIR_RE = re.compile(
    r"^(?P<date>\d{4}-\d{2}-\d{2})(?:_(?P<hour>\d{2})_(?P<minute>\d{2})_(?P<second>\d{2}(?:\.\d+)?))?$"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_image_id(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype("Int64")


def _strict_boolean(series: pd.Series, label: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    mapped = series.astype(str).str.strip().str.lower().map(
        {"true": True, "false": False}
    )
    if mapped.isna().any():
        raise RuntimeError(f"Request roster has invalid {label} values")
    return mapped.astype(bool)


def load_frozen_request_roster(
    roster_path: Path = REQUEST_ROSTER,
    validation_path: Path = REQUEST_ROSTER_VALIDATION,
    all_mri_path: Path = ALL_MRI,
    dryrun_path: Path = DRY_RUN,
    *,
    expected_sha256: str = FROZEN_REQUEST_ROSTER_SHA256,
    expected_counts: dict[str, int] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load and independently verify the exact-date request universe.

    The roster is a scientific input, not a convenience list.  Bind it to its
    immutable hash, validation record, and the two source snapshots before a
    catalog-coverage report can be regenerated.
    """

    expected_counts = expected_counts or FROZEN_REQUEST_COUNTS
    try:
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Cannot read request-roster validation: {validation_path}"
        ) from exc
    if validation.get("status") != "PASS":
        raise RuntimeError("Request-roster validation status is not PASS")

    actual_sha256 = sha256(roster_path)
    recorded_sha256 = str(validation.get("output", {}).get("sha256", ""))
    if actual_sha256 != expected_sha256 or recorded_sha256 != actual_sha256:
        raise RuntimeError("Frozen request-roster SHA-256 does not match")

    recorded_inputs = validation.get("inputs", {})
    for name, path in (("all_mri", all_mri_path), ("dryrun", dryrun_path)):
        recorded = str(recorded_inputs.get(name, {}).get("sha256", ""))
        if not recorded or recorded != sha256(path):
            raise RuntimeError(f"Request-roster source hash mismatch for {name}")

    roster = pd.read_csv(roster_path, low_memory=False)
    required = {
        "subject_id",
        "group",
        "dti_image_id",
        "t1_image_id",
        "is_current_t1_image",
        "is_rounded_age_screen_candidate",
        "is_provisional_replacement_candidate",
    }
    missing = sorted(required - set(roster.columns))
    if missing:
        raise RuntimeError(f"Request roster is missing columns: {missing}")
    roster = roster.copy()
    roster["subject_id"] = roster["subject_id"].astype(str).str.strip()
    for column in ("dti_image_id", "t1_image_id"):
        roster[column] = normalize_image_id(roster[column])
        if roster[column].isna().any() or roster[column].le(0).any():
            raise RuntimeError(f"Request roster has invalid {column} values")
        roster[column] = roster[column].astype(int)
    for column in (
        "is_current_t1_image",
        "is_rounded_age_screen_candidate",
        "is_provisional_replacement_candidate",
    ):
        roster[column] = _strict_boolean(roster[column], column)

    per_subject = roster.groupby("subject_id", sort=False)
    computed_counts = {
        "selected_subjects": roster["subject_id"].nunique(),
        "request_rows": len(roster),
        "unique_t1_image_ids": roster["t1_image_id"].nunique(),
        "subjects_with_candidates": roster["subject_id"].nunique(),
        "rounded_age_screen_ids": int(
            roster["is_rounded_age_screen_candidate"].sum()
        ),
        "rounded_age_screen_ids_in_request": int(
            roster["is_rounded_age_screen_candidate"].sum()
        ),
        "provisional_replacement_ids": int(
            roster["is_provisional_replacement_candidate"].sum()
        ),
        "provisional_replacement_ids_in_request": int(
            roster["is_provisional_replacement_candidate"].sum()
        ),
        "current_t1_ids_in_request": int(roster["is_current_t1_image"].sum()),
        "minimum_candidates_per_subject": int(per_subject.size().min()),
        "maximum_candidates_per_subject": int(per_subject.size().max()),
    }
    recorded_counts = validation.get("counts", {})
    for key, expected in expected_counts.items():
        if computed_counts.get(key) != expected or recorded_counts.get(key) != expected:
            raise RuntimeError(
                f"Frozen request-roster count mismatch for {key}: "
                f"computed={computed_counts.get(key)}, "
                f"recorded={recorded_counts.get(key)}, expected={expected}"
            )
    if roster["t1_image_id"].duplicated().any():
        raise RuntimeError("Request roster contains duplicate T1 Image IDs")
    conflicts = per_subject.agg(
        dti_ids=("dti_image_id", "nunique"), groups=("group", "nunique")
    )
    if conflicts["dti_ids"].ne(1).any() or conflicts["groups"].ne(1).any():
        raise RuntimeError("Request roster has conflicting subject mappings")
    return roster, validation


def load_authenticated_aws_evidence(
    principal_path: Path = AWS_PRINCIPAL_VERIFICATION,
    search_path: Path = AWS_CATALOG_SEARCH,
) -> dict[str, Any]:
    """Load the preserved read-only AWS evidence without making a live call."""

    try:
        principal = json.loads(principal_path.read_text(encoding="utf-8"))
        search = json.loads(search_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("Cannot read authenticated AWS catalog evidence") from exc
    principal_sha256 = sha256(principal_path)
    linked = search.get("principal_verification", {})
    safety = search.get("safety", {})
    if principal.get("status") != "PASS" or linked.get("status") != "PASS":
        raise RuntimeError("Authenticated AWS principal evidence is not PASS")
    if linked.get("sha256") != principal_sha256:
        raise RuntimeError("AWS catalog search is not bound to the principal evidence")
    if principal.get("profile") != "aml" or search.get("profile") != "aml":
        raise RuntimeError("AWS evidence is not for the named aml profile")
    if principal.get("account") != search.get("account"):
        raise RuntimeError("AWS principal/search account mismatch")
    if (
        safety.get("read_only") is not True
        or safety.get("objects_downloaded") != 0
        or safety.get("objects_written") != 0
        or safety.get("credentials_recorded") is not False
        or principal.get("secrets_recorded") is not False
    ):
        raise RuntimeError("AWS evidence does not attest a credential-safe read-only search")
    return {
        "principal": principal,
        "search": search,
        "principal_sha256": principal_sha256,
        "search_sha256": sha256(search_path),
    }


def format_authenticated_aws_evidence(evidence: dict[str, Any]) -> str:
    principal = evidence["principal"]
    search = evidence["search"]
    results = search.get("results", {})
    scope = search.get("scope", {})
    found = results.get("fresh_authoritative_exact_date_export_found")
    if found is not False:
        raise RuntimeError("AWS evidence no longer supports the recorded negative result")
    return (
        f"The named `{principal['profile']}` principal recorded PASS at "
        f"{principal['verified_utc']} in account {principal['account']}. Its "
        f"checksum-bound, read-only catalog search at {search['searched_utc']} "
        f"checked {scope.get('visible_account_bucket_count', 'the recorded')} visible "
        f"bucket names and the documented project-known/plausible locations. It found "
        f"no fresh authoritative ADNI/LONI exact-date export. The preserved evidence "
        f"is `aws_aml_principal_verification.json` (SHA-256 "
        f"`{evidence['principal_sha256']}`) and `aws_aml_catalog_search.json` "
        f"(SHA-256 `{evidence['search_sha256']}`). This report generator reads those "
        f"snapshots; it does not reauthenticate, search AWS, or transfer an object."
    )


def summarize_request_coverage(
    roster: pd.DataFrame,
    mri_master_ids: set[int],
    local_mri_ids: set[int],
) -> dict[str, int]:
    request_ids = set(roster["t1_image_id"].astype(int))
    mri_overlap = request_ids & mri_master_ids
    local_overlap = request_ids & local_mri_ids
    per_subject = roster.groupby("subject_id").size()
    return {
        "request_roster_rows": len(roster),
        "request_roster_unique_t1_image_ids": len(request_ids),
        "request_roster_subjects": roster["subject_id"].nunique(),
        "request_roster_min_candidates_per_subject": int(per_subject.min()),
        "request_roster_max_candidates_per_subject": int(per_subject.max()),
        "request_roster_current_t1_ids": int(roster["is_current_t1_image"].sum()),
        "request_roster_rounded_age_screen_ids": int(
            roster["is_rounded_age_screen_candidate"].sum()
        ),
        "request_roster_provisional_replacement_ids": int(
            roster["is_provisional_replacement_candidate"].sum()
        ),
        "request_roster_ids_in_mri_master": len(mri_overlap),
        "request_roster_subjects_in_mri_master": roster.loc[
            roster["t1_image_id"].isin(mri_overlap), "subject_id"
        ].nunique(),
        "request_roster_ids_in_local_raw_mri": len(local_overlap),
        "request_roster_subjects_in_local_raw_mri": roster.loc[
            roster["t1_image_id"].isin(local_overlap), "subject_id"
        ].nunique(),
    }


def parse_path_datetime(date_dir: str) -> tuple[str | None, str | None]:
    match = DATE_DIR_RE.match(date_dir)
    if not match:
        return None, None
    date_text = match.group("date")
    if match.group("hour") is None:
        return date_text, None
    second = float(match.group("second"))
    second_int = int(second)
    microsecond = int(round((second - second_int) * 1_000_000))
    parsed = datetime(
        *map(int, date_text.split("-")),
        int(match.group("hour")),
        int(match.group("minute")),
        second_int,
        microsecond,
    )
    return date_text, parsed.isoformat()


def index_local_mri() -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for image_dir in sorted(MRI_ROOT.glob("*/*/*/I*")):
        if not image_dir.is_dir():
            continue
        image_match = IMAGE_DIR_RE.match(image_dir.name)
        if not image_match:
            continue
        date_text, datetime_text = parse_path_datetime(image_dir.parent.name)
        files = [path for path in image_dir.iterdir() if path.is_file()]
        records.append(
            {
                "subject_id": image_dir.parents[2].name,
                "image_id": int(image_match.group("image_id")),
                "path_date": date_text,
                "path_datetime": datetime_text,
                "image_dir": str(image_dir),
                "file_count": len(files),
                "total_bytes": sum(path.stat().st_size for path in files),
            }
        )
    result = pd.DataFrame.from_records(records)
    if result.empty:
        raise RuntimeError(f"No local MRI image directories found below {MRI_ROOT}")
    duplicate = result[result.duplicated("image_id", keep=False)]
    if not duplicate.empty:
        conflicting = duplicate.groupby("image_id")["path_date"].nunique()
        if (conflicting > 1).any():
            raise RuntimeError("A local MRI image ID resolves to conflicting path dates")
        result = result.sort_values("image_dir").drop_duplicates("image_id", keep="first")
    return result


def unique_source_rows(
    frame: pd.DataFrame, image_ids: set[int], source_name: str
) -> pd.DataFrame:
    selected = frame[frame["image_id"].isin(image_ids)].copy()
    date_counts = selected.groupby("image_id")["study_date"].nunique(dropna=True)
    if (date_counts > 1).any():
        bad = date_counts[date_counts > 1].index.tolist()
        raise RuntimeError(f"{source_name} has conflicting study dates for IDs {bad[:10]}")
    return (
        selected.sort_values(["image_id", "study_date"])
        .drop_duplicates("image_id", keep="last")
        .copy()
    )


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)

    request_roster, request_validation = load_frozen_request_roster()
    aws_evidence = load_authenticated_aws_evidence()
    aws_evidence_text = format_authenticated_aws_evidence(aws_evidence)
    dry = pd.read_csv(DRY_RUN)
    dti_master = pd.read_csv(DTI_MASTER, low_memory=False)
    mri_master = pd.read_csv(MRI_MASTER, low_memory=False)
    all_mri = pd.read_csv(ALL_MRI, low_memory=False)
    local_mri = index_local_mri()

    for column in ("dti_image_id", "current_t1_image_id", "candidate_t1_image_id"):
        dry[column] = normalize_image_id(dry[column])
    for frame in (dti_master, mri_master, all_mri):
        frame["image_id"] = normalize_image_id(frame["Image ID"])
    dti_master["study_date"] = pd.to_datetime(
        dti_master["Study Date"], errors="coerce"
    ).dt.normalize()
    mri_master["study_date"] = pd.to_datetime(
        mri_master["Study Date"], errors="coerce"
    ).dt.normalize()
    local_mri["path_date_parsed"] = pd.to_datetime(
        local_mri["path_date"], errors="coerce"
    ).dt.normalize()

    selected_dti_ids = set(dry["dti_image_id"].dropna().astype(int))
    dti_selected = unique_source_rows(dti_master, selected_dti_ids, "dti_master")
    dti_selected = dti_selected[
        ["image_id", "Subject ID", "study_date", "Visit", "Phase", "Description"]
    ].rename(
        columns={
            "image_id": "dti_image_id",
            "Subject ID": "dti_master_subject_id",
            "study_date": "dti_study_date",
            "Visit": "dti_visit",
            "Phase": "dti_master_phase",
            "Description": "dti_description",
        }
    )

    current = dry.merge(dti_selected, on="dti_image_id", how="left", validate="m:1")
    local_current = local_mri.rename(
        columns={
            "subject_id": "current_t1_local_subject_id",
            "image_id": "current_t1_image_id",
            "path_date": "current_t1_path_date",
            "path_datetime": "current_t1_path_datetime",
            "path_date_parsed": "current_t1_path_date_parsed",
            "image_dir": "current_t1_image_dir",
            "file_count": "current_t1_file_count",
            "total_bytes": "current_t1_total_bytes",
        }
    )
    current = current.merge(
        local_current,
        on="current_t1_image_id",
        how="left",
        validate="m:1",
    )
    current["current_t1_gap_days_signed"] = (
        current["current_t1_path_date_parsed"] - current["dti_study_date"]
    ).dt.days
    current["current_t1_gap_days_abs"] = current[
        "current_t1_gap_days_signed"
    ].abs()
    current["current_within_90_days"] = current["current_t1_gap_days_abs"].le(90)
    current["current_within_180_days"] = current["current_t1_gap_days_abs"].le(180)
    current["current_exact_date_source"] = "local_raw_path_date"

    mri_dates = unique_source_rows(
        mri_master,
        set(current["current_t1_image_id"].dropna().astype(int)),
        "mri_master",
    )[["image_id", "study_date"]].rename(
        columns={
            "image_id": "current_t1_image_id",
            "study_date": "mri_master_study_date",
        }
    )
    current = current.merge(
        mri_dates,
        on="current_t1_image_id",
        how="left",
        validate="m:1",
    )
    current["path_matches_mri_master_date"] = (
        current["mri_master_study_date"].notna()
        & current["current_t1_path_date_parsed"].eq(current["mri_master_study_date"])
    )

    current_columns = [
        "subject_id",
        "group",
        "dti_image_id",
        "dti_study_date",
        "dti_visit",
        "dti_master_phase",
        "current_t1_image_id",
        "current_t1_local_subject_id",
        "current_t1_path_date",
        "current_t1_path_datetime",
        "current_t1_gap_days_signed",
        "current_t1_gap_days_abs",
        "current_within_90_days",
        "current_within_180_days",
        "current_t1_image_dir",
        "current_t1_file_count",
        "current_t1_total_bytes",
        "mri_master_study_date",
        "path_matches_mri_master_date",
        "current_exact_date_source",
        "recommended_action",
    ]
    current_output = current[current_columns].copy()
    for column in ("dti_study_date", "mri_master_study_date"):
        current_output[column] = current_output[column].dt.strftime("%Y-%m-%d")
    current_path = OUTPUT / "current_t1_exact_date_evidence_v2.csv"
    current_output.to_csv(current_path, index=False)

    replacements = dry[
        dry["recommended_action"].eq("replace_after_exact_date_and_qc")
    ].copy()
    local_candidate_ids = set(local_mri["image_id"].dropna().astype(int))
    mri_master_ids = set(mri_master["image_id"].dropna().astype(int))
    all_mri_ids = set(all_mri["image_id"].dropna().astype(int))
    request_ids = set(request_roster["t1_image_id"].astype(int))
    if not request_ids.issubset(all_mri_ids):
        missing = sorted(request_ids - all_mri_ids)
        raise RuntimeError(
            f"Frozen request IDs are no longer present in all_mri: {missing[:10]}"
        )

    expected_screen_pairs = set(
        zip(dry["subject_id"], dry["candidate_t1_image_id"].astype(int))
    )
    roster_screen = request_roster[
        request_roster["is_rounded_age_screen_candidate"]
    ]
    actual_screen_pairs = set(
        zip(roster_screen["subject_id"], roster_screen["t1_image_id"].astype(int))
    )
    if expected_screen_pairs != actual_screen_pairs:
        raise RuntimeError("Frozen request roster does not contain the exact dry-run screen pairs")

    expected_provisional_pairs = set(
        zip(
            replacements["subject_id"],
            replacements["candidate_t1_image_id"].astype(int),
        )
    )
    roster_provisional = request_roster[
        request_roster["is_provisional_replacement_candidate"]
    ]
    actual_provisional_pairs = set(
        zip(
            roster_provisional["subject_id"],
            roster_provisional["t1_image_id"].astype(int),
        )
    )
    if expected_provisional_pairs != actual_provisional_pairs:
        raise RuntimeError(
            "Frozen request roster does not contain the exact provisional replacement subset"
        )

    request_coverage = summarize_request_coverage(
        request_roster, mri_master_ids, local_candidate_ids
    )
    replacements["candidate_in_all_mri"] = replacements[
        "candidate_t1_image_id"
    ].isin(all_mri_ids)
    replacements["candidate_in_exact_date_mri_master"] = replacements[
        "candidate_t1_image_id"
    ].isin(mri_master_ids)
    replacements["candidate_in_local_raw_mri"] = replacements[
        "candidate_t1_image_id"
    ].isin(local_candidate_ids)
    replacements["exact_candidate_date_available"] = False
    replacements["catalog_resolution"] = (
        "fresh_authoritative_ADNI_LONI_exact_day_export_required"
    )
    replacement_columns = [
        "subject_id",
        "group",
        "dti_image_id",
        "candidate_t1_image_id",
        "candidate_t1_phase",
        "candidate_t1_age",
        "candidate_description",
        "candidate_in_all_mri",
        "candidate_in_exact_date_mri_master",
        "candidate_in_local_raw_mri",
        "exact_candidate_date_available",
        "catalog_resolution",
    ]
    replacement_path = OUTPUT / "replacement_t1_catalog_coverage_v2.csv"
    replacements[replacement_columns].to_csv(replacement_path, index=False)

    current_mri_master_covered = int(current["mri_master_study_date"].notna().sum())
    current_mri_master_matches = int(current["path_matches_mri_master_date"].sum())
    coverage_records = [
        ("request_roster_rows", request_coverage["request_roster_rows"]),
        (
            "request_roster_unique_t1_image_ids",
            request_coverage["request_roster_unique_t1_image_ids"],
        ),
        ("request_roster_subjects", request_coverage["request_roster_subjects"]),
        (
            "request_roster_min_candidates_per_subject",
            request_coverage["request_roster_min_candidates_per_subject"],
        ),
        (
            "request_roster_max_candidates_per_subject",
            request_coverage["request_roster_max_candidates_per_subject"],
        ),
        (
            "request_roster_ids_in_all_mri",
            len(request_ids & all_mri_ids),
        ),
        (
            "request_roster_ids_in_exact_date_mri_master",
            request_coverage["request_roster_ids_in_mri_master"],
        ),
        (
            "request_roster_subjects_in_exact_date_mri_master",
            request_coverage["request_roster_subjects_in_mri_master"],
        ),
        (
            "request_roster_ids_in_local_raw_mri",
            request_coverage["request_roster_ids_in_local_raw_mri"],
        ),
        (
            "request_roster_subjects_in_local_raw_mri",
            request_coverage["request_roster_subjects_in_local_raw_mri"],
        ),
        (
            "request_roster_current_t1_ids",
            request_coverage["request_roster_current_t1_ids"],
        ),
        (
            "request_roster_rounded_age_screen_ids",
            request_coverage["request_roster_rounded_age_screen_ids"],
        ),
        (
            "request_roster_provisional_replacement_ids",
            request_coverage["request_roster_provisional_replacement_ids"],
        ),
        ("selected_dti_rows", len(current)),
        ("selected_dti_exact_dates_in_dti_master", int(current["dti_study_date"].notna().sum())),
        ("current_t1_path_dates", int(current["current_t1_path_date_parsed"].notna().sum())),
        ("current_t1_within_90_days", int(current["current_within_90_days"].sum())),
        ("current_t1_within_180_days", int(current["current_within_180_days"].sum())),
        ("current_t1_over_180_days", int((current["current_t1_gap_days_abs"] > 180).sum())),
        ("current_t1_ids_in_mri_master", current_mri_master_covered),
        ("current_t1_path_dates_matching_mri_master", current_mri_master_matches),
        ("replacement_candidates", len(replacements)),
        ("replacement_candidates_in_all_mri", int(replacements["candidate_in_all_mri"].sum())),
        (
            "replacement_candidates_in_exact_date_mri_master",
            int(replacements["candidate_in_exact_date_mri_master"].sum()),
        ),
        (
            "replacement_candidates_in_local_raw_mri",
            int(replacements["candidate_in_local_raw_mri"].sum()),
        ),
        ("local_raw_mri_subjects", int(local_mri["subject_id"].nunique())),
        ("local_raw_mri_image_ids", int(local_mri["image_id"].nunique())),
    ]
    coverage = pd.DataFrame(coverage_records, columns=["measure", "value"])
    coverage_path = OUTPUT / "catalog_coverage_v2.csv"
    coverage.to_csv(coverage_path, index=False)

    path_only = current[
        current["current_within_180_days"] & ~current["current_within_90_days"]
    ][
        [
            "subject_id",
            "group",
            "dti_image_id",
            "current_t1_image_id",
            "dti_study_date",
            "current_t1_path_date",
            "current_t1_gap_days_abs",
        ]
    ].sort_values("current_t1_gap_days_abs")
    path_only_rows = "\n".join(
        "| "
        + " | ".join(
            [
                str(row.subject_id),
                str(row.group),
                str(int(row.dti_image_id)),
                str(int(row.current_t1_image_id)),
                row.dti_study_date.strftime("%Y-%m-%d"),
                str(row.current_t1_path_date),
                str(int(row.current_t1_gap_days_abs)),
            ]
        )
        + " |"
        for row in path_only.itertuples(index=False)
    )

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    report = f"""# Exact-date catalog source inventory

**Generated:** {generated}  
**Authenticated AWS evidence:** {aws_evidence["search"]["searched_utc"]}  
**Purpose:** quantify exact-date coverage, preserve the replacement-source audit, and support the approved local-only analysis design.  
**Safety:** read-only audit; no image download and no production mutation.

## Decisive result

The current selected DTI date is recoverable for all 530 records from dti_master.csv. The current selected T1 acquisition date is also recoverable for all 530 from the raw image-directory date. This repairs the date audit for current pairs, but not for proposed replacements.

Exact-date nearest-neighbour selection cannot be based on the earlier one-candidate-per-subject rounded-age screen. The frozen request universe therefore contains all {request_coverage["request_roster_rows"]:,} `Type=Original` T1-like Image IDs found in the frozen local all_mri source for all {request_coverage["request_roster_subjects"]} audited subjects, with {request_coverage["request_roster_min_candidates_per_subject"]}–{request_coverage["request_roster_max_candidates_per_subject"]} candidates per subject. Its SHA-256 is `{sha256(REQUEST_ROSTER)}`. It contains all 530 rounded-age screen IDs, including the 330 provisional replacement IDs as a nested priority subset; 330 is not the complete export universe.

The 330 provisional replacement IDs all exist in all_mri.csv, which has phase and rounded age but no exact date. None exists in the older exact-date mri_master.csv, and none is present in the local raw MRI tree. More broadly, the old mri_master overlaps only {request_coverage["request_roster_ids_in_mri_master"]:,}/{request_coverage["request_roster_rows"]:,} full-roster IDs across {request_coverage["request_roster_subjects_in_mri_master"]}/{request_coverage["request_roster_subjects"]} subjects and does not provide the complete source-QC accounting required by the intake validator. A fresh authoritative ADNI/LONI export would have to resolve every frozen Image ID with an exact date and recognized QC, expose any candidate drift relative to the frozen local source, and yield at least one eligible T1 for every subject before replacement pairing could run.

## Approved project-scope override

On 2026-07-18 the user fixed the study scope to images already available locally. External replacement acquisition is therefore **not pursued** under the current plan. The active cohort-design path preserves the 530 current pairs and models/reports their real timing, acquisition, and QC limitations. This scope decision does not convert long-gap or low-quality inputs into valid same-visit measurements; it changes them from acquisition blockers into explicit design constraints and sensitivity variables. The signed project record is `decisions/closed_world_data_scope_20260718.md`.

## Verified local coverage

| Measure | Result |
|---|---:|
| Frozen exact-date request IDs | {request_coverage["request_roster_unique_t1_image_ids"]:,}/{request_coverage["request_roster_rows"]:,} unique |
| Subjects represented in frozen request | {request_coverage["request_roster_subjects"]}/530 |
| Candidate range per subject | {request_coverage["request_roster_min_candidates_per_subject"]}–{request_coverage["request_roster_max_candidates_per_subject"]} |
| Full-roster IDs represented in frozen local all_mri snapshot | {len(request_ids & all_mri_ids):,}/{request_coverage["request_roster_rows"]:,} |
| Full-roster IDs represented in old mri_master | {request_coverage["request_roster_ids_in_mri_master"]:,}/{request_coverage["request_roster_rows"]:,} across {request_coverage["request_roster_subjects_in_mri_master"]}/530 subjects |
| Full-roster IDs present in local raw MRI | {request_coverage["request_roster_ids_in_local_raw_mri"]:,}/{request_coverage["request_roster_rows"]:,} across {request_coverage["request_roster_subjects_in_local_raw_mri"]}/530 subjects |
| Current selected T1 IDs also in Original-T1 request universe | {request_coverage["request_roster_current_t1_ids"]}/530 |
| Rounded-age screen IDs nested in full request | {request_coverage["request_roster_rounded_age_screen_ids"]}/530 |
| Provisional replacement IDs nested in full request | {request_coverage["request_roster_provisional_replacement_ids"]}/330 |
| Selected DTI IDs with exact dti_master date | {int(current["dti_study_date"].notna().sum())}/530 |
| Current T1 IDs with raw-path date | {int(current["current_t1_path_date_parsed"].notna().sum())}/530 |
| Current pairs within 90 days | {int(current["current_within_90_days"].sum())}/530 |
| Current pairs within 180 days | {int(current["current_within_180_days"].sum())}/530 |
| Current pairs over 180 days | {int((current["current_t1_gap_days_abs"] > 180).sum())}/530 |
| Current IDs represented in old mri_master | {current_mri_master_covered}/530 |
| Raw-path dates matching mri_master where available | {current_mri_master_matches}/{current_mri_master_covered} |
| Proposed replacements represented in frozen local all_mri snapshot | {int(replacements["candidate_in_all_mri"].sum())}/330 |
| Proposed replacements represented in exact-date mri_master | {int(replacements["candidate_in_exact_date_mri_master"].sum())}/330 |
| Proposed replacements present in local raw MRI | {int(replacements["candidate_in_local_raw_mri"].sum())}/330 |

The raw path date is strongly corroborated: it agrees with mri_master Study Date for every current T1 where both are available. It remains a file-provenance date, not a substitute for an authoritative source export for images that are not present locally. Only 305 current selected IDs occur in the frozen roster because that roster intentionally enumerates `Type=Original` T1-like catalog rows; this count must not be misread as loss of current raw-path date evidence, which remains 530/530.

## The three 90-to-180-day current pairs

These were rounded-age “contemporaneous” in the earlier screen but fail the proposed 90-day primary rule while passing the 180-day sensitivity rule.

| Subject | Group | DTI image | Current T1 image | DTI date | T1 path date | Absolute gap days |
|---|---|---:|---:|---|---|---:|
{path_only_rows}

## Source ranking

| Source | Exact date | Full request-universe coverage | Image QC | Decision |
|---|---|---:|---|---|
| Fresh ADNI/LONI image or download export | Expected | Every frozen ID resolved with exact date and recognized QC; candidate drift exposed; at least one eligible T1 for each of 530 subjects | Expected/linked | Scientifically preferable replacement route, but not pursued under approved local-only scope. |
| dti_master.csv | Yes | DTI only; 530/530 selected DTI IDs | Partial acquisition fields | Authoritative for current DTI dates in this audit. |
| Local/S3 raw image-directory path | Yes for locally held image | {request_coverage["request_roster_ids_in_local_raw_mri"]:,}/{request_coverage["request_roster_rows"]:,} full-roster IDs; 0/330 provisional replacements | No catalog QC | Valid provenance evidence for current files; not a complete intake source. |
| mri_master.csv | Yes | {request_coverage["request_roster_ids_in_mri_master"]:,}/{request_coverage["request_roster_rows"]:,} full-roster IDs across {request_coverage["request_roster_subjects_in_mri_master"]}/530 subjects; 0/330 provisional replacements | Limited, not the required source-QC field | Incomplete legacy source; cannot release pairing. |
| all_mri.csv | No | {len(request_ids & all_mri_ids):,}/{request_coverage["request_roster_rows"]:,} candidates relative to the frozen local snapshot, including 330/330 provisional replacements | No | Roster-construction source only; not evidence of current IDA completeness and cannot approve pairing. |

## Live S3 inventory

The legacy raw prefixes are s3://sabeesh/exp/Images/mri/ and s3://sabeesh/exp/Images/dti/, each with 1,114 subject prefixes. The alternative s3://sabeesh/exp/data/Images/ prefix is empty. The recorded cloud cohort prefix contains the same 13 local catalog files, last modified 2026-04-20, so it does not supply the missing full-universe dates and source QC.

{aws_evidence_text}

## Active local-only path

1. Lock the 530 current DTI–T1 pairs, exact intervals, local paths/hashes, diagnosis, site/protocol, acquisition, and QC fields.
2. Attempt corrected processing on all available inputs. Preserve technical failures and attrition; do not promote invalid registrations, matrices, or tensor outputs.
3. Use the full available cohort as exploratory evidence and prespecify timing (all, <=180 days, <=90 days), protocol/site, and QC sensitivity analyses.
4. Calibrate the manuscript to available-data evidence and disclose that interval adjustment cannot reconstruct missing contemporaneous anatomy.

## Contingency replacement path — superseded

If the data-scope decision is explicitly reopened, the replacement route would require the following:

1. Export or otherwise recover authoritative ADNI/LONI MRI metadata for the frozen {request_coverage["request_roster_rows"]:,}-ID request universe, including exact Study Date and source-backed image QC/provenance. Preserve the 330 provisional IDs only as a nested priority cross-check.
2. Resolve every frozen Image ID with an exact date and recognized QC, expose any candidate drift relative to the frozen local source, and require at least one eligible T1 for every one of the 530 subjects. Missing or unknown requested IDs block pairing by default.
3. Release the catalog to pairing only when validation records `status=PASS`, `release_status=PAIRING_READY`, and `pairing_ready=true`; join it to the 530 exact DTI dates and apply the frozen <=90-day primary and <=180-day sensitivity rules.
4. Reconcile diagnosis, present the selected pairs at SL-H01, and only then enumerate the exact nonlocal image objects and bytes for separate transfer approval. The final transfer count is determined by exact-date/QC selection and local reconciliation, not capped at 330.

## Reproducibility

Inputs:

- {DRY_RUN} — SHA-256 {sha256(DRY_RUN)}
- {DTI_MASTER} — SHA-256 {sha256(DTI_MASTER)}
- {MRI_MASTER} — SHA-256 {sha256(MRI_MASTER)}
- {ALL_MRI} — SHA-256 {sha256(ALL_MRI)}
- {REQUEST_ROSTER} — SHA-256 {sha256(REQUEST_ROSTER)}
- {REQUEST_ROSTER_VALIDATION} — SHA-256 {sha256(REQUEST_ROSTER_VALIDATION)}; recorded status {request_validation["status"]}
- {AWS_PRINCIPAL_VERIFICATION} — SHA-256 {aws_evidence["principal_sha256"]}
- {AWS_CATALOG_SEARCH} — SHA-256 {aws_evidence["search_sha256"]}

Outputs:

- {current_path}
- {replacement_path}
- {coverage_path}
"""
    report_path = AUDIT / "catalog_source_inventory.md"
    report_path.write_text(report, encoding="utf-8")

    print(report_path)
    print(coverage.to_string(index=False))


if __name__ == "__main__":
    main()
