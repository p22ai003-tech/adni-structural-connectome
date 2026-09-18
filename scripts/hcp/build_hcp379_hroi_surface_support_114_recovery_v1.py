#!/usr/bin/env python3
"""Recover the uniform HROI candidate for the isolated 114 FreeSurfer route.

The cohort builder expects every FreeSurfer subject under the standard
``/data/derivatives/fastsurfer/<unit>`` layout.  Unit
``114_S_6347_I1344943`` instead has a separately validated full FreeSurfer
reconstruction under the HCP379 recovery root.  This adapter exposes that
immutable reconstruction through a new attempt-local SUBJECTS_DIR and then
executes the unchanged cohort-uniform HROI builder.

No source, failed cohort directory, or historical derivative is modified.
The adapter is selected only by the pre-existing input route, never by
diagnosis, density, or any scientific outcome.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXP = Path("/home/ec2-user/exp")
BASE_SOURCE = (
    EXP / "scripts/hcp/build_hcp379_hroi_surface_support_v1.py"
)
UNIT = "114_S_6347_I1344943"
RECOVERED_SUBJECT = Path(
    "/data/derivatives/hcp379_v2/fastsurfer_repair/"
    "standard_freesurfer/114_S_6347_I1344943_FS"
)
RECOVERED_HCP_SOURCE = Path(
    "/data/derivatives/hcp379_v2/fastsurfer_repair/"
    "114_S_6347_I1344943/hcp/HCPMMP1+aseg.mgz"
)
ATTEMPTS = Path(
    "/data/derivatives/hcp379_v2/source_label_repair_v1/"
    "hroi_surface_support_114_recovery_v1/attempts"
)


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = load_module(BASE_SOURCE, "hcp379_hroi_surface_support_base_v1")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")


def required_inputs() -> tuple[Path, ...]:
    return (
        BASE_SOURCE,
        RECOVERED_HCP_SOURCE,
        RECOVERED_SUBJECT / "mri/orig.mgz",
        RECOVERED_SUBJECT / "mri/aseg.mgz",
        RECOVERED_SUBJECT / "label/lh.HCPMMP1.annot",
        RECOVERED_SUBJECT / "label/rh.HCPMMP1.annot",
        BASE.RELABEL_LUT,
        BASE.LABEL2VOL,
    )


def preflight() -> dict[str, Any]:
    missing = [
        str(path)
        for path in required_inputs()
        if not path.is_file() or path.is_symlink()
    ]
    return {
        "schema_version": "1.0.0",
        "record_type": (
            "diagnosis_blind_hcp379_hroi_surface_support_"
            "114_recovery_preflight"
        ),
        "generated_utc": utc_now(),
        "status": (
            "READY_114_HROI_RECOVERY"
            if not missing
            else "MISSING_114_RECOVERY_INPUTS"
        ),
        "unit": UNIT,
        "diagnosis_labels_used": False,
        "route_basis": "pre-existing isolated FreeSurfer recovery path",
        "source_files_modified": False,
        "missing_inputs": missing,
        "recovered_subject": str(RECOVERED_SUBJECT),
        "recovered_hcp_source": str(RECOVERED_HCP_SOURCE),
        "base_builder": BASE.file_record(BASE_SOURCE),
        "cohort_recipe_selected": False,
        "selected_count_tractography_authorized": False,
        "final_release_authorized": False,
    }


def execute() -> dict[str, Any]:
    bound = preflight()
    if bound["status"] != "READY_114_HROI_RECOVERY":
        raise ValueError("isolated 114 recovery inputs differ")
    attempt = ATTEMPTS / f"{stamp()}-hroi-114-recovery"
    attempt.mkdir(parents=True, exist_ok=False)
    BASE.atomic_json(attempt / "preflight.json", bound)

    subjects_dir = attempt / "subjects_dir_adapter"
    subjects_dir.mkdir()
    subject_link = subjects_dir / UNIT
    os.symlink(RECOVERED_SUBJECT, subject_link, target_is_directory=True)
    if subject_link.resolve() != RECOVERED_SUBJECT.resolve():
        raise ValueError("attempt-local subject adapter differs")

    candidate_root = attempt / "candidates"
    BASE.SUBJECTS_DIR = subjects_dir
    BASE.RECOVERED_SOURCE = RECOVERED_HCP_SOURCE
    manifest = BASE.build(UNIT, candidate_root)
    manifest_path = candidate_root / UNIT / "candidate_manifest.json"
    if (
        manifest.get("status") != "PASS_ALL_379_SOURCE_LABELS"
        or manifest.get("diagnosis_labels_used") is not False
        or manifest.get("source_files_modified") is not False
        or manifest.get("unit") != UNIT
        or manifest.get("present_source_label_n") != 379
        or manifest.get("missing_source_labels") != []
        or manifest.get("policy", {}).get(
            "subcortical_voxels_overwritten"
        )
        != 0
        or manifest.get("policy", {}).get(
            "other_cortical_labels_overwritten"
        )
        != 0
        or manifest.get("original_source")
        != BASE.file_record(RECOVERED_HCP_SOURCE)
        or manifest.get("implementation") != BASE.file_record(BASE_SOURCE)
    ):
        raise ValueError("isolated 114 HROI candidate contract differs")

    summary = {
        "schema_version": "1.0.0",
        "record_type": (
            "diagnosis_blind_hcp379_hroi_surface_support_"
            "114_recovery_summary"
        ),
        "generated_utc": utc_now(),
        "status": "PASS_114_HROI_RECOVERY_FOR_COHORT_MERGE",
        "unit": UNIT,
        "diagnosis_labels_used": False,
        "route_basis": "pre-existing isolated FreeSurfer recovery path",
        "source_files_modified": False,
        "failed_standard_candidate_modified": False,
        "cohort_recipe_selected": False,
        "selected_count_tractography_authorized": False,
        "final_release_authorized": False,
        "subject_adapter": {
            "path": str(subject_link.absolute()),
            "target": str(RECOVERED_SUBJECT.resolve()),
            "preserved_with_attempt": True,
        },
        "recovered_inputs": {
            "hcp_source": BASE.file_record(RECOVERED_HCP_SOURCE),
            "orig": BASE.file_record(
                RECOVERED_SUBJECT / "mri/orig.mgz"
            ),
            "aseg": BASE.file_record(
                RECOVERED_SUBJECT / "mri/aseg.mgz"
            ),
            "lh_annotation": BASE.file_record(
                RECOVERED_SUBJECT / "label/lh.HCPMMP1.annot"
            ),
            "rh_annotation": BASE.file_record(
                RECOVERED_SUBJECT / "label/rh.HCPMMP1.annot"
            ),
        },
        "candidate_manifest": BASE.file_record(manifest_path),
        "candidate": manifest["candidate"],
        "base_builder": BASE.file_record(BASE_SOURCE),
        "implementation": BASE.file_record(Path(__file__)),
        "preflight": BASE.file_record(attempt / "preflight.json"),
    }
    BASE.atomic_json(attempt / "summary.json", summary)
    return {
        "attempt": str(attempt),
        "status": summary["status"],
        "candidate_manifest": str(manifest_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    value = execute() if args.execute else preflight()
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0 if not value.get("missing_inputs") else 1


if __name__ == "__main__":
    raise SystemExit(main())
