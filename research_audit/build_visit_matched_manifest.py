#!/usr/bin/env python3
"""Build a dry-run T1 replacement manifest for the current DTI cohort.

No data are downloaded and no production paths are changed.  The available
``all_mri.csv`` catalog contains phase and age (rounded to 0.1 year), but not
study dates.  Therefore this script identifies likely same-visit T1 candidates
and explicitly marks every replacement as requiring exact-date verification
before a rerun.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT = Path("/home/ec2-user/exp")
AUDIT = PROJECT / "research_audit"
OUT = AUDIT / "outputs"
COHORT = OUT / "cohort_audit_manifest.csv"
MRI_CATALOG = PROJECT / "cohort" / "all_mri.csv"
LOCAL_MRI = Path("/data/Images/mri")

# Original high-resolution T1 acquisitions suitable for a clean reprocessing
# route.  Derived masks, FreeSurfer products, normalized images and other
# post-processed files are deliberately excluded.
T1_PATTERN = r"MPRAGE|MP-RAGE|IR-SPGR|IR-FSPGR|BRAVO"


def local_image_path(subject_id: str, image_id: int) -> str | None:
    root = LOCAL_MRI / subject_id
    if not root.exists():
        return None
    matches = list(root.rglob(f"I{image_id}"))
    return str(matches[0]) if matches else None


def main() -> None:
    cohort = pd.read_csv(COHORT)
    catalog = pd.read_csv(MRI_CATALOG)
    candidates = catalog[
        catalog["Type"].eq("Original")
        & catalog["Description"].str.contains(
            T1_PATTERN, case=False, regex=True, na=False
        )
    ].copy()
    candidates["candidate_is_repeat"] = candidates["Description"].str.contains(
        "repeat", case=False, na=False
    )
    candidates["Age"] = pd.to_numeric(candidates["Age"], errors="coerce")
    candidates["Image ID"] = pd.to_numeric(
        candidates["Image ID"], errors="coerce"
    ).astype("Int64")

    rows = []
    for record in cohort.itertuples(index=False):
        subject = record.subject_id
        options = candidates[candidates["Subject ID"].eq(subject)].copy()
        if options.empty:
            best = None
        else:
            options["candidate_age_gap_years"] = (
                options["Age"] - float(record.dti_age)
            ).abs()
            options["candidate_phase_match"] = options["Phase"].eq(record.dti_phase)
            # Timing dominates selection.  Within the same rounded-age visit,
            # prefer same phase and a non-repeat series.  Image ID provides a
            # deterministic final tie-break only; exact date and acquisition QC
            # must decide the final series.
            options = options.sort_values(
                [
                    "candidate_age_gap_years",
                    "candidate_phase_match",
                    "candidate_is_repeat",
                    "Image ID",
                ],
                ascending=[True, False, True, True],
            )
            best = options.iloc[0]

        current_ok = bool(record.contemporaneous_t1)
        if best is None:
            action = "exclude_or_source_missing_t1"
            candidate_id = pd.NA
            candidate_gap = np.nan
            candidate_phase_match = False
            candidate_description = pd.NA
            candidate_age = np.nan
            candidate_phase = pd.NA
            n_candidates = 0
            local_path = None
        else:
            candidate_id = int(best["Image ID"])
            candidate_gap = float(best["candidate_age_gap_years"])
            candidate_phase_match = bool(best["candidate_phase_match"])
            candidate_description = best["Description"]
            candidate_age = float(best["Age"])
            candidate_phase = best["Phase"]
            n_candidates = len(options)
            local_path = local_image_path(subject, candidate_id)
            if current_ok:
                action = "retain_current_t1"
            elif candidate_gap <= 0.5 and candidate_phase_match:
                action = "replace_after_exact_date_and_qc"
            else:
                action = "exclude_or_source_closer_t1"

        rows.append(
            {
                "subject_id": subject,
                "group": record.group_corrected,
                "dti_image_id": record.actual_dti_image_id,
                "dti_phase": record.dti_phase,
                "dti_age": record.dti_age,
                "current_t1_image_id": record.actual_t1_image_id,
                "current_t1_phase": record.t1_phase,
                "current_t1_age": record.t1_age,
                "current_gap_years": record.dti_t1_age_gap_years,
                "current_is_contemporaneous": current_ok,
                "candidate_t1_image_id": candidate_id,
                "candidate_t1_phase": candidate_phase,
                "candidate_t1_age": candidate_age,
                "candidate_age_gap_years": candidate_gap,
                "candidate_phase_match": candidate_phase_match,
                "candidate_description": candidate_description,
                "candidate_is_already_local": local_path is not None,
                "candidate_local_path": local_path,
                "n_original_t1_candidates": n_candidates,
                "recommended_action": action,
                "exact_study_date_required": action
                == "replace_after_exact_date_and_qc",
                "candidate_selection_status": "metadata_screen_only_not_approved",
            }
        )

    result = pd.DataFrame(rows)
    result.to_csv(OUT / "visit_matched_t1_manifest_dryrun.csv", index=False)

    summary = (
        result.groupby(["recommended_action", "group"], dropna=False)
        .size()
        .rename("n")
        .reset_index()
    )
    summary.to_csv(OUT / "visit_matched_t1_manifest_summary.csv", index=False)
    action_counts = result["recommended_action"].value_counts()
    replacement = result[result["recommended_action"].eq("replace_after_exact_date_and_qc")]
    text = [
        "# Visit-matched T1 dry-run manifest",
        "",
        "This is a metadata screen, not an approved acquisition manifest. `all_mri.csv` has age rounded to 0.1 year and phase but lacks exact study dates. Every replacement must be verified against exact DTI/T1 dates and image QC before downloading or rerunning.",
        "",
        f"- Retain current contemporaneous T1: **{int(action_counts.get('retain_current_t1', 0))}**",
        f"- Candidate replacement after exact-date/QC check: **{int(action_counts.get('replace_after_exact_date_and_qc', 0))}**",
        f"- Exclude or source a closer T1: **{int(action_counts.get('exclude_or_source_closer_t1', 0))}**",
        f"- Replacement candidates already local: **{int(replacement['candidate_is_already_local'].sum())} / {len(replacement)}**",
        "",
        "Recommended primary rule: same visit and <=90 days. A <=180-day sensitivity cohort can be prespecified. The exact-day catalog or ADNI download manifest is required to apply either rule.",
        "",
    ]
    (OUT / "visit_matched_t1_manifest_summary.md").write_text(
        "\n".join(text), encoding="utf-8"
    )
    print("\n".join(text))


if __name__ == "__main__":
    main()
