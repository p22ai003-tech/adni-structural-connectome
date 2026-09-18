#!/usr/bin/env python3
"""Adopt the cohort-uniform H-ROI support for bounded pretract execution.

The receipt requires all 530 candidates, exact source/implementation hashes,
one genuinely affected corrected-ACT canary, one complete-source exact-10M
reuse test, and explicit agent visual triage.  It authorizes only operational
pretract use.  It does not satisfy final human visual QC or release gates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


ROOT = Path("/home/ec2-user/exp")
COHORT_ATTEMPTS = Path(
    "/data/derivatives/hcp379_v2/source_label_repair_v1/"
    "hroi_surface_support_cohort_v1/attempts"
)
SOURCE_AUDIT = Path(
    "/data/derivatives/hcp379_v2/audits/"
    "source_label_completeness_v1/attempts/"
    "20260728T045255.695073Z-source-label-audit/summary.json"
)
AFFECTED_VALIDATION = Path(
    "/data/derivatives/hcp379_v2/source_label_repair_v1/validation/"
    "hroi_affected_corrected_act_v1/attempts/"
    "20260728T060900.544977Z-affected-corrected-act-canary/summary.json"
)
COMPLETE_VALIDATION = Path(
    "/data/derivatives/hcp379_v2/source_label_repair_v1/validation/"
    "hroi_surface_support_corrected_act_v1/attempts/"
    "20260728T055524.033287Z-hroi-corrected-act-validation/summary.json"
)
AGENT_VISUAL_TRIAGE = Path(
    "/data/derivatives/hcp379_v2/source_label_repair_v1/review/"
    "hroi_corrected_act_v1/attempts/"
    "20260728T062728.490070Z-hroi-corrected-act-review/"
    "agent_visual_triage.json"
)
OUTPUT = Path(
    "/data/derivatives/hcp379_v2/source_label_repair_v1/"
    "hroi_surface_support_cohort_v1/policy_adoption_v1.json"
)
EXPECTED_UNITS = 530


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"not a regular file: {path}")
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def verify(record: Mapping[str, Any], label: str) -> Path:
    path = Path(str(record.get("path", ""))).resolve()
    if file_record(path) != dict(record):
        raise ValueError(f"{label}: artifact binding differs")
    return path


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def latest_cohort_summary() -> Path | None:
    candidates = sorted(COHORT_ATTEMPTS.glob("*/summary.json"))
    return candidates[-1].resolve() if candidates else None


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    os.chmod(temporary, 0o444)
    os.replace(temporary, path)


def validate_evidence(cohort_summary_path: Path) -> dict[str, Any]:
    cohort = load_json(cohort_summary_path)
    if (
        cohort.get("record_type")
        != "diagnosis_blind_hcp379_hroi_surface_support_cohort_summary"
        or cohort.get("status")
        != "PASS_COHORT_UNIFORM_HROI_SUPPORT_530"
        or cohort.get("diagnosis_labels_used") is not False
        or cohort.get("source_files_modified") is not False
        or cohort.get("release_adopted") is not False
        or cohort.get("expected_unit_n") != EXPECTED_UNITS
        or cohort.get("valid_unit_n") != EXPECTED_UNITS
        or cohort.get("failure_n") != 0
    ):
        raise ValueError("cohort-uniform candidate summary differs")
    unit_rows = {
        str(row.get("unit")): row for row in cohort.get("units", [])
    }
    if len(unit_rows) != EXPECTED_UNITS or "" in unit_rows:
        raise ValueError("cohort candidate unit set differs")
    adopted_units: dict[str, Any] = {}
    for unit, row in sorted(unit_rows.items()):
        manifest_path = verify(
            row["candidate_manifest"], f"{unit}:candidate_manifest"
        )
        candidate_path = verify(row["candidate"], f"{unit}:candidate")
        manifest = load_json(manifest_path)
        if (
            manifest.get("record_type")
            != "diagnosis_blind_hcp379_hroi_surface_support_candidate"
            or manifest.get("status") != "PASS_ALL_379_SOURCE_LABELS"
            or manifest.get("diagnosis_labels_used") is not False
            or manifest.get("source_files_modified") is not False
            or manifest.get("unit") != unit
            or manifest.get("present_source_label_n") != 379
            or manifest.get("missing_source_labels") != []
            or manifest.get("candidate") != row["candidate"]
            or manifest.get("policy", {}).get(
                "subcortical_voxels_overwritten"
            )
            != 0
            or manifest.get("policy", {}).get(
                "other_cortical_labels_overwritten"
            )
            != 0
        ):
            raise ValueError(f"{unit}: candidate manifest differs")
        adopted_units[unit] = {
            "candidate_manifest": row["candidate_manifest"],
            "candidate": row["candidate"],
            "original_source": manifest["original_source"],
            "changed_voxel_n": int(manifest["changed_voxel_n"]),
        }
        if candidate_path != Path(
            str(manifest["candidate"]["path"])
        ).resolve():
            raise ValueError(f"{unit}: candidate path differs")

    audit = load_json(SOURCE_AUDIT)
    if (
        audit.get("record_type")
        != "diagnosis_blind_hcp379_source_label_completeness_audit"
        or audit.get("status") != "HOLD_SOURCE_LABEL_REPAIR_REQUIRED"
        or audit.get("diagnosis_labels_used") is not False
        or audit.get("audited_unit_n") != EXPECTED_UNITS
        or audit.get("read_failure_n") != 0
        or audit.get("all_379_source_labels_n") != 495
        or audit.get("missing_source_labels_n") != 35
        or audit.get("missing_node_frequency") != {"120": 34, "300": 3}
    ):
        raise ValueError("source completeness audit differs")

    affected = load_json(AFFECTED_VALIDATION)
    affected_neighbours = affected.get("anatomical_neighbours", {})
    left_names = {
        str(row.get("name"))
        for row in affected_neighbours.get("node120_top15", [])
    }
    right_names = {
        str(row.get("name"))
        for row in affected_neighbours.get("node300_top15", [])
    }
    if (
        affected.get("status")
        != "TECHNICAL_PASS_AFFECTED_CORRECTED_ACT_POLICY_REVIEW_PENDING"
        or affected.get("diagnosis_labels_used") is not False
        or affected.get("unit") != "002_S_0413_I863064"
        or affected.get("atlas", {}).get(
            "all_379_candidate_nodes_present"
        )
        is not True
        or affected.get("atlas", {}).get(
            "changed_only_zero_to_h_nodes"
        )
        is not True
        or affected.get("assignment", {}).get(
            "endpoint_assignment_fraction", 0
        )
        < 0.85
        or affected.get("topology", {}).get("node120_degree", 0) <= 0
        or affected.get("topology", {}).get("node300_degree", 0) <= 0
        or "Left-Hippocampus" not in left_names
        or "Right-Hippocampus" not in right_names
    ):
        raise ValueError("affected corrected-ACT policy canary differs")

    complete = load_json(COMPLETE_VALIDATION)
    if (
        complete.get("status")
        != "TECHNICAL_PASS_CORRECTED_ACT_SCIENTIFIC_POLICY_REVIEW_PENDING"
        or complete.get("diagnosis_labels_used") is not False
        or complete.get("unit") != "135_S_6840_I1263427"
        or complete.get("atlas", {}).get(
            "all_379_candidate_nodes_present"
        )
        is not True
        or complete.get("atlas", {}).get(
            "changed_only_zero_to_h_nodes"
        )
        is not True
        or abs(
            float(complete.get("assignment", {}).get("fraction_delta", 1))
        )
        > 1.0e-12
        or abs(
            float(complete.get("topology", {}).get("density_delta", 1))
        )
        > 0.005
        or complete.get("topology", {}).get(
            "node120_degree_candidate", 0
        )
        <= complete.get("topology", {}).get("node120_degree_original", 0)
        or complete.get("topology", {}).get(
            "node300_degree_candidate", 0
        )
        <= complete.get("topology", {}).get("node300_degree_original", 0)
    ):
        raise ValueError("complete-source corrected-ACT reuse test differs")

    visual = load_json(AGENT_VISUAL_TRIAGE)
    if (
        visual.get("status")
        != "PASS_AGENT_VISUAL_TRIAGE_NOT_HUMAN_QC"
        or visual.get("diagnosis_labels_used") is not False
        or visual.get("human_visual_qc_inferred") is not False
        or visual.get("operational_pretract_adoption_supported") is not True
        or visual.get("final_release_human_qc_satisfied") is not False
        or {
            str(row.get("unit")) for row in visual.get("decisions", [])
        }
        != {"002_S_0413_I863064", "135_S_6840_I1263427"}
        or any(
            row.get("decision") != "PASS_AGENT_VISUAL_TRIAGE"
            for row in visual.get("decisions", [])
        )
    ):
        raise ValueError("agent visual triage differs")

    return {
        "units": adopted_units,
        "records": {
            "cohort_summary": file_record(cohort_summary_path),
            "source_label_audit": file_record(SOURCE_AUDIT),
            "affected_corrected_act_validation": file_record(
                AFFECTED_VALIDATION
            ),
            "complete_source_corrected_act_validation": file_record(
                COMPLETE_VALIDATION
            ),
            "agent_visual_triage": file_record(AGENT_VISUAL_TRIAGE),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort-summary", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    cohort_summary = (
        args.cohort_summary.resolve()
        if args.cohort_summary
        else latest_cohort_summary()
    )
    if cohort_summary is None or not cohort_summary.is_file():
        print(
            json.dumps(
                {
                    "status": "WAITING_FOR_530_COHORT_CANDIDATE_SUMMARY",
                    "pretract_use_authorized": False,
                },
                sort_keys=True,
            )
        )
        return 0
    evidence = validate_evidence(cohort_summary)
    preflight = {
        "status": "READY_TO_ADOPT_FOR_PRETRACT_ONLY",
        "unit_n": len(evidence["units"]),
        "pretract_use_authorized": False,
        "final_release_authorized": False,
    }
    if not args.execute:
        print(json.dumps(preflight, sort_keys=True))
        return 0
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    receipt = {
        "schema_version": "1.0.0",
        "record_type": (
            "diagnosis_blind_hcp379_hroi_surface_support_policy_adoption"
        ),
        "generated_utc": utc_now(),
        "status": (
            "PASS_OPERATIONAL_PRETRACT_ADOPTION_"
            "FINAL_HUMAN_QC_PENDING"
        ),
        "diagnosis_labels_used": False,
        "source_files_modified": False,
        "cohort_uniform": True,
        "unit_n": EXPECTED_UNITS,
        "pretract_use_authorized": True,
        "selected_count_tractography_authorized": False,
        "final_release_authorized": False,
        "human_visual_qc_inferred": False,
        "final_human_visual_qc_pending": True,
        "per_subject_policy_tuning_allowed": False,
        "policy": {
            "scope": "bilateral HCP-MMP1 H_ROI only",
            "support_tissue": (
                "cerebral white matter in both aseg and original source"
            ),
            "application": "same diagnosis-blind construction for all 530",
            "other_cortical_labels_overwritten": 0,
            "subcortical_voxels_overwritten": 0,
        },
        "units": evidence["units"],
        "records": evidence["records"],
        "implementation": file_record(Path(__file__)),
    }
    atomic_json(OUTPUT, receipt)
    print(
        json.dumps(
            {
                "output": str(OUTPUT),
                "status": receipt["status"],
                "unit_n": receipt["unit_n"],
                "pretract_use_authorized": True,
                "final_release_authorized": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
