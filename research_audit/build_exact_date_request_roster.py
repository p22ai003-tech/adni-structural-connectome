#!/usr/bin/env python3
"""Build the complete T1 exact-date metadata request universe.

The earlier 330-row file contains one rounded-age replacement candidate per
affected subject. It is not sufficient for exact-date nearest-neighbour
selection. This builder enumerates every Original T1-like image for all 530
subjects in the audited dense cohort so the exact-date stage cannot simply
confirm a candidate selected with rounded age.
"""

from __future__ import annotations

import argparse
import json
import math
import numbers
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from research_audit.build_visit_pair_manifest import T1_RE, file_sha256


REQUIRED_DRYRUN_COLUMNS = {
    "subject_id",
    "group",
    "dti_image_id",
    "current_t1_image_id",
    "candidate_t1_image_id",
    "recommended_action",
}
REQUIRED_MRI_COLUMNS = {
    "Subject ID",
    "Phase",
    "Age",
    "Modality",
    "Description",
    "Type",
    "Image ID",
}
SUBJECT_RE = re.compile(r"^\d{3}_S_\d{4}$")


def normalize_positive_integer(value: Any) -> int | None:
    """Normalize an ID without accepting scientific notation or lossy floats."""

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
    numeric = values.map(normalize_positive_integer)
    invalid = numeric.isna()
    if invalid.any():
        examples = values[invalid].head(10).tolist()
        raise ValueError(f"{label} contains invalid positive integer IDs: {examples}")
    return numeric.astype(int)


def build_request_roster(
    all_mri: pd.DataFrame,
    dryrun: pd.DataFrame,
    *,
    expected_subjects: int = 530,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    missing_dryrun = sorted(REQUIRED_DRYRUN_COLUMNS - set(dryrun.columns))
    missing_mri = sorted(REQUIRED_MRI_COLUMNS - set(all_mri.columns))
    if missing_dryrun or missing_mri:
        raise ValueError(
            f"Missing columns: dryrun={missing_dryrun}, all_mri={missing_mri}"
        )
    selected = dryrun[list(dryrun.columns)].copy()
    selected["subject_id"] = selected["subject_id"].astype(str).str.strip()
    invalid_subjects = selected.loc[
        ~selected["subject_id"].str.fullmatch(SUBJECT_RE), "subject_id"
    ].tolist()
    if invalid_subjects:
        raise ValueError(f"Dry-run contains invalid subject IDs: {invalid_subjects[:10]}")
    selected["dti_image_id"] = positive_ids(selected["dti_image_id"], "dti_image_id")
    selected["current_t1_image_id"] = positive_ids(
        selected["current_t1_image_id"], "current_t1_image_id"
    )
    selected["candidate_t1_image_id"] = positive_ids(
        selected["candidate_t1_image_id"], "candidate_t1_image_id"
    )
    if len(selected) != expected_subjects or selected["subject_id"].nunique() != expected_subjects:
        raise ValueError(
            f"Expected {expected_subjects} unique selected subjects; found "
            f"{len(selected)} rows/{selected['subject_id'].nunique()} subjects"
        )
    if selected["dti_image_id"].duplicated().any():
        raise ValueError("Dry-run contains duplicate selected DTI Image IDs")

    source = all_mri[all_mri["Subject ID"].astype(str).isin(set(selected["subject_id"]))].copy()
    source["is_t1_description"] = source["Description"].astype("string").str.contains(T1_RE, na=False)
    source["is_original"] = source["Type"].astype("string").str.strip().str.lower().eq("original")
    source["is_mri"] = source["Modality"].astype("string").str.strip().str.lower().eq("mri")
    source = source[
        source["is_t1_description"] & source["is_original"] & source["is_mri"]
    ].copy()
    source["t1_image_id"] = positive_ids(source["Image ID"], "all_mri Image ID")
    if source["t1_image_id"].duplicated().any():
        duplicates = source.loc[source["t1_image_id"].duplicated(keep=False), "t1_image_id"].unique()
        raise ValueError(f"Original T1 Image IDs are duplicated: {list(duplicates[:10])}")

    selected_fields = selected[
        [
            "subject_id",
            "group",
            "dti_image_id",
            "current_t1_image_id",
            "candidate_t1_image_id",
            "recommended_action",
        ]
    ]
    result = source.merge(
        selected_fields,
        left_on="Subject ID",
        right_on="subject_id",
        how="left",
        validate="m:1",
    )
    result = pd.DataFrame(
        {
            "subject_id": result["subject_id"],
            "group": result["group"],
            "dti_image_id": result["dti_image_id"].astype(int),
            "t1_image_id": result["t1_image_id"].astype(int),
            "catalog_phase": result["Phase"].astype("string"),
            "catalog_age_rounded": pd.to_numeric(result["Age"], errors="coerce"),
            "catalog_modality": result["Modality"].astype("string"),
            "catalog_description": result["Description"].astype("string"),
            "catalog_type": result["Type"].astype("string"),
            "is_current_t1_image": result["t1_image_id"].eq(result["current_t1_image_id"]),
            "is_rounded_age_screen_candidate": result["t1_image_id"].eq(
                result["candidate_t1_image_id"]
            ),
            "is_provisional_replacement_candidate": result["t1_image_id"].eq(
                result["candidate_t1_image_id"]
            )
            & result["recommended_action"].eq("replace_after_exact_date_and_qc"),
            "source_catalog": "cohort/all_mri.csv",
        }
    ).sort_values(["subject_id", "t1_image_id"], kind="mergesort", ignore_index=True)

    missing_subjects = sorted(set(selected["subject_id"]) - set(result["subject_id"]))
    screen_ids = set(selected["candidate_t1_image_id"].astype(int))
    provisional_ids = set(
        selected.loc[
            selected["recommended_action"].eq("replace_after_exact_date_and_qc"),
            "candidate_t1_image_id",
        ].astype(int)
    )
    request_ids = set(result["t1_image_id"].astype(int))
    missing_screen_ids = sorted(screen_ids - request_ids)
    if missing_subjects or missing_screen_ids:
        raise ValueError(
            f"Request universe incomplete: missing_subjects={missing_subjects[:10]}, "
            f"missing_screen_ids={missing_screen_ids[:10]}"
        )
    validation = {
        "schema_version": "1.0",
        "status": "PASS",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "counts": {
            "selected_subjects": len(selected),
            "request_rows": len(result),
            "unique_t1_image_ids": result["t1_image_id"].nunique(),
            "subjects_with_candidates": result["subject_id"].nunique(),
            "rounded_age_screen_ids": len(screen_ids),
            "rounded_age_screen_ids_in_request": len(screen_ids & request_ids),
            "provisional_replacement_ids": len(provisional_ids),
            "provisional_replacement_ids_in_request": len(provisional_ids & request_ids),
            "current_t1_ids_in_request": int(result["is_current_t1_image"].sum()),
            "minimum_candidates_per_subject": int(result.groupby("subject_id").size().min()),
            "maximum_candidates_per_subject": int(result.groupby("subject_id").size().max()),
        },
        "rules": {
            "subject_universe": "all subjects in visit_matched_t1_manifest_dryrun.csv",
            "candidate_definition": "Modality=MRI, Type=Original, and description matches frozen T1_RE",
            "rounded_age_used_for_selection": False,
            "request_join_key": "exact T1 Image ID",
            "download_approval": False,
        },
    }
    return result, validation


def write_outputs(
    result: pd.DataFrame,
    validation: dict[str, Any],
    output_csv: Path,
    validation_json: Path,
    inputs: dict[str, Path],
) -> None:
    for path in (output_csv, validation_json):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing output: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv, index=False)
    payload = dict(validation)
    payload["inputs"] = {
        name: {
            "path": str(path.resolve()),
            "size_bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for name, path in inputs.items()
    }
    payload["output"] = {
        "path": str(output_csv.resolve()),
        "size_bytes": output_csv.stat().st_size,
        "sha256": file_sha256(output_csv),
    }
    validation_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all-mri", type=Path, required=True)
    parser.add_argument("--dryrun", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--expected-subjects", type=int, default=530)
    args = parser.parse_args()
    result, validation = build_request_roster(
        pd.read_csv(args.all_mri, low_memory=False),
        pd.read_csv(args.dryrun, low_memory=False),
        expected_subjects=args.expected_subjects,
    )
    write_outputs(
        result,
        validation,
        args.out,
        args.validation,
        {"all_mri": args.all_mri, "dryrun": args.dryrun},
    )
    print(json.dumps(validation["counts"], indent=2, sort_keys=True))
    print(args.out.resolve())


if __name__ == "__main__":
    main()
