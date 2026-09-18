#!/home/ec2-user/fsl/bin/python
"""Build the exact corrected-input row for the recovered FreeSurfer unit."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any


EXP = Path("/home/ec2-user/exp")
HCP_ROOT = Path("/data/derivatives/hcp379_v2")
UNIT = "114_S_6347_I1344943"
SOURCE = EXP / "scripts/hcp/audit_hcp379_scaleup_inputs_v2.py"
ATTESTATION = (
    HCP_ROOT
    / "fastsurfer_repair"
    / UNIT
    / "hcp_source_recovery_attestation_v3.json"
)
OUTPUT_ROOT = (
    EXP
    / "research_audit/outputs/"
    "hcp379_freesurfer_corrected_input_v3"
)
OUTPUT_CSV = OUTPUT_ROOT / "freesurfer_corrected_input.csv"
OUTPUT_SUMMARY = OUTPUT_ROOT / "freesurfer_corrected_input_summary.json"


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


AUDIT = load_module(SOURCE, "hcp379_freesurfer_input_audit_engine")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise ValueError(f"not a regular file: {resolved}")
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    attestation = load_json(ATTESTATION)
    if (
        attestation.get("record_type")
        != (
            "diagnosis_blind_hcp379_freesurfer_source_"
            "recovery_attestation"
        )
        or attestation.get("status")
        != "PASS_HCP_SOURCE_READY_FOR_CORRECTED_ROUTE"
        or attestation.get("unit") != UNIT
        or attestation.get("diagnosis_or_outcome_fields_used") is not False
        or attestation.get("atlas_nodes_present") != 379
        or attestation.get("old_existing_track_is_final_release_candidate")
        is not False
    ):
        raise ValueError("FreeSurfer source attestation differs")
    acquisition = AUDIT.acquisition_by_unit()
    row = AUDIT.audit_unit(
        UNIT,
        recommended_action="CORRECTED_RECOVERY4_AFTER_FS_SOURCE_REPAIR",
        acquisition=acquisition[UNIT],
    )
    dwi_failures = [
        item
        for item in str(row["failures"]).split(";")
        if item
        and (
            item.startswith("eddy_")
            or item.startswith("gradient_")
            or item == "gradient_table_physical_qc_failed"
        )
    ]
    if (
        dwi_failures
        or int(row["gradient_rows"]) != int(row["dwi_size"].split("x")[-1])
        or int(row["b0_rows"]) < 1
        or int(row["diffusion_rows"]) < 6
        or int(row["diffusion_gradient_norm_failures"]) != 0
        or row["fastsurfer_t1_identity_match"] is not True
    ):
        raise ValueError(
            f"{UNIT}: recovered corrected-input DWI differs: {dwi_failures}"
        )
    recovered_hcp = Path(
        str(
            attestation["records"]["recovered_hcp_source"][
                "hcp379_source_mgz"
            ]["path"]
        )
    )
    if (
        file_record(recovered_hcp)
        != attestation["records"]["recovered_hcp_source"][
            "hcp379_source_mgz"
        ]
    ):
        raise ValueError("recovered HCP source binding differs")
    standard = load_json(
        Path(
            str(
                attestation["records"]["standard_freesurfer_result"][
                    "path"
                ]
            )
        )
    )
    row.update(
        {
            "route": "READY_FOR_CORRECTED_PRETRACT_FROM_EDDY",
            "failures": "",
            "repairable_actions": "",
            "hcp_source_parcellation_path": str(recovered_hcp.resolve()),
            "hcp_source_parcellation_present": True,
            "historical_nodes_b0_reuse_allowed": False,
            "full_dwi_preprocessing_planned": False,
            "recovered_freesurfer_subject_dir": str(
                Path(str(standard["subject_dir"])).resolve()
            ),
            "source_recovery_attestation_path": str(
                ATTESTATION.resolve()
            ),
            "source_recovery_attestation_sha256": sha256_file(
                ATTESTATION
            ),
            "old_existing_track_reuse_allowed": False,
        }
    )
    summary = {
        "schema_version": "1.0.0",
        "record_type": (
            "diagnosis_blind_hcp379_freesurfer_corrected_input_audit"
        ),
        "status": "PASS",
        "diagnosis_labels_used": False,
        "diagnosis_or_outcomes_used": False,
        "target_n": 1,
        "audited_n": 1,
        "ready_n": 1,
        "route_counts": {
            "READY_FOR_CORRECTED_PRETRACT_FROM_EDDY": 1
        },
        "historical_nodes_b0_reuse_allowed": False,
        "old_existing_track_reuse_allowed": False,
        "records": {
            "source_attestation": file_record(ATTESTATION),
            "shared_audit_engine": file_record(SOURCE),
            "builder": file_record(Path(__file__)),
        },
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    AUDIT.atomic_csv(args.output_root / OUTPUT_CSV.name, [row])
    AUDIT.atomic_json(args.output_root / OUTPUT_SUMMARY.name, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
