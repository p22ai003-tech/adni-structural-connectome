#!/home/ec2-user/fsl/bin/python
"""Build Recovery4-safe HCP379 pre-tractography inputs for 216 subjects.

This is the non-overwriting successor to ``hcp379_scaleup_pretract_v2.py``.
It reuses the audited Eddy and exact-T1 inputs, while treating the older
runner as a provisional image-construction engine only.  Promotion requires
the Recovery4 contracts that were validated on the 15-subject canary:

* genuine diagnosis-blind human review of the immutable Recovery4 pack;
* BBR as the primary T1-to-DWI registration route;
* a deterministic seeded ANTs rigid failover only when direct 5TT or HCP379
  QC rejects BBR;
* all 379 HCP nodes, direct atlas-inside-mask and 5TT/DWI Dice gates;
* unchanged raw tensor maps plus separate physically bounded maps;
* the Recovery4 1% raw tensor anomaly-fraction and FOD numerical gates; and
* per-subject SHA-256 provenance for every promoted tractography input.

The default mode is a pre-human dry run.  Imaging requires ``--execute`` and
a complete real human-QC CSV.  Neither mode reads diagnosis or outcomes.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import importlib.util
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import nibabel as nib
import numpy as np
import yaml


EXP = Path("/home/ec2-user/exp")
DERIV = Path("/data/derivatives")
HCP_ROOT = DERIV / "hcp379_v2"
OUTPUT_ROOT = HCP_ROOT / "corrected_scaleup_recovery4"
V2_SOURCE = EXP / "scripts/hcp/hcp379_scaleup_pretract_v2.py"
SCALAR_SOURCE = (
    EXP / "scripts/hcp/build_hcp379_pretract_scalar_recovery4_v2.py"
)
COMPACTOR_SOURCE = (
    EXP / "scripts/hcp/compact_hcp379_pretract_recovery4_v3.py"
)
HROI_POLICY_ADOPTION_SOURCE = (
    EXP / "scripts/hcp/adopt_hcp379_hroi_surface_support_policy_v1.py"
)
HROI_POLICY_ADOPTION = (
    HCP_ROOT
    / "source_label_repair_v1/"
    "hroi_surface_support_cohort_v1/"
    "policy_adoption_v1.json"
)
PHASE_B_GATE_SOURCE = (
    EXP / "scripts/hcp/run_hcp379_phase_b_stability_v2.py"
)
AUDIT_CSV = (
    EXP
    / "research_audit/outputs/hcp379_scaleup_input_audit_v2/"
    "scaleup_input_audit.csv"
)
AUDIT_SUMMARY = AUDIT_CSV.with_name("scaleup_input_audit_summary.json")
RECOVERY4_MASTER = (
    HCP_ROOT / "pretract_recovery4/pretract_recovery4_manifest.json"
)
RECOVERY4_REVIEW = (
    HCP_ROOT
    / "review_recovery4_corrected_overlay/"
    "hcp379_visual_review_pack_recovery4_corrected_overlay_manifest.json"
)
CORRECTED_OVERLAY_QUANTITATIVE_AUDIT = (
    HCP_ROOT
    / "review_recovery4_corrected_overlay/"
    "hcp379_corrected_overlay_quantitative_audit.json"
)
DEFAULT_HUMAN_QC = (
    HCP_ROOT
    / "review_recovery4_corrected_overlay/"
    "human_visual_qc_recovery4_corrected_overlay.csv"
)
FROZEN_RESPONSE_MANIFEST = (
    DERIV
    / "scforge_v2/h04a_r1_recovery_20260719_retry4/"
    "frozen_calibration_retry4_v2/frozen_response_calibration_manifest.json"
)
CONFIG = EXP / "configs/connectome_v2_retry3.yaml"
FOD_ORDER_AUDIT_ROOT = (
    EXP
    / "research_audit/outputs/hcp379_fod_order_compatibility_v1"
)
FOD_ORDER_AUDIT = FOD_ORDER_AUDIT_ROOT / "audit.json"
FOD_ORDER_TABLE = FOD_ORDER_AUDIT_ROOT / "fod_order_compatibility.csv"
FOD_ORDER_VALIDATION = FOD_ORDER_AUDIT_ROOT / "validation.json"
FAILURE_LEDGER_ROOT = (
    EXP
    / "research_audit/outputs/"
    "hcp379_pretract_failure_recovery_ledger_v1"
)
FAILURE_LEDGER = FAILURE_LEDGER_ROOT / "ledger.csv"
FAILURE_LEDGER_VALIDATION = FAILURE_LEDGER_ROOT / "validation.json"
MRTRIX = Path("/home/ec2-user/mrtrix3/bin")
FSL = Path("/home/ec2-user/fsl/bin")
ANTS = EXP / ".envs/ants-2.6.5/bin"
FREESURFER = Path("/home/ec2-user/freesurfer")
FSL_PYTHON = FSL / "python"
RELABEL = EXP / "scripts/hcp/relabel_hcp.py"
RELABEL_LUT = (
    DERIV / "parc_hcpmmp1/hcpmmp1_subcort_relabel.txt"
)
EXPECTED_AUDIT_N = 227
EXPECTED_CANARY_N = 15
EXPECTED_CANARY_OVERLAP_N = 11
EXPECTED_SCALEUP_N = 216
EXPECTED_HROI_POLICY_N = 530
EXPECTED_FOD_ORDER_HOLD_N = 10
MINIMUM_5TT_DWI_DICE = 0.70
MINIMUM_CORTICAL_HEMISPHERE_ATLAS_INSIDE_DWI_MASK = 0.80
ANTS_RANDOM_SEED = 1234
GEOMETRY_AFFINE_TOLERANCE = 1.0e-5


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


V2 = load_module("hcp379_scaleup_pretract_v2_engine", V2_SOURCE)
SCALAR = load_module("hcp379_pretract_scalar_recovery4", SCALAR_SOURCE)
PHASE_B_GATE = load_module(
    "hcp379_phase_b_recovery4_gate",
    PHASE_B_GATE_SOURCE,
)
ORIGINAL_ATLAS_QC = V2.ATLAS_QC_MODULE.atlas_qc


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


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
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def verify_file_record(record: Mapping[str, Any], *, label: str) -> Path:
    path = Path(str(record.get("path", ""))).expanduser().resolve()
    if (
        not path.is_file()
        or path.is_symlink()
        or path.stat().st_size != int(record.get("size_bytes", -1))
        or sha256_file(path) != record.get("sha256")
    ):
        raise ValueError(f"immutable file record differs: {label}={path}")
    return path


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root is not an object: {path}")
    return value


def validate_hroi_policy_adoption(
    *, required: bool
) -> dict[str, Any]:
    """Validate the cohort-uniform source policy without weakening dry runs."""

    if not HROI_POLICY_ADOPTION.is_file():
        if required:
            raise FileNotFoundError(
                "530-subject HROI pretract-adoption receipt is required: "
                f"{HROI_POLICY_ADOPTION}"
            )
        return {
            "status": "WAITING_FOR_530_HROI_POLICY_ADOPTION",
            "pretract_use_authorized": False,
            "receipt": None,
            "units": {},
        }
    receipt = load_json(HROI_POLICY_ADOPTION)
    units = receipt.get("units")
    if (
        receipt.get("record_type")
        != (
            "diagnosis_blind_hcp379_hroi_surface_support_"
            "policy_adoption"
        )
        or receipt.get("status")
        != (
            "PASS_OPERATIONAL_PRETRACT_ADOPTION_"
            "FINAL_HUMAN_QC_PENDING"
        )
        or receipt.get("diagnosis_labels_used") is not False
        or receipt.get("source_files_modified") is not False
        or receipt.get("cohort_uniform") is not True
        or receipt.get("unit_n") != EXPECTED_HROI_POLICY_N
        or receipt.get("pretract_use_authorized") is not True
        or receipt.get("selected_count_tractography_authorized")
        is not False
        or receipt.get("final_release_authorized") is not False
        or receipt.get("human_visual_qc_inferred") is not False
        or receipt.get("final_human_visual_qc_pending") is not True
        or receipt.get("per_subject_policy_tuning_allowed") is not False
        or not isinstance(units, Mapping)
        or len(units) != EXPECTED_HROI_POLICY_N
        or "" in units
    ):
        raise ValueError("HROI pretract-adoption receipt differs")
    implementation = verify_file_record(
        receipt["implementation"],
        label="HROI policy-adoption implementation",
    )
    if implementation != HROI_POLICY_ADOPTION_SOURCE.resolve():
        raise ValueError("HROI adoption implementation path differs")
    records = receipt.get("records")
    if not isinstance(records, Mapping):
        raise ValueError("HROI adoption evidence records are missing")
    required_records = {
        "cohort_summary",
        "source_label_audit",
        "affected_corrected_act_validation",
        "complete_source_corrected_act_validation",
        "agent_visual_triage",
    }
    if not required_records.issubset(records):
        raise ValueError("HROI adoption evidence set differs")
    for name in sorted(required_records):
        verify_file_record(
            records[name], label=f"HROI adoption evidence {name}"
        )
    return {
        "status": "PASS_OPERATIONAL_PRETRACT_ADOPTION",
        "pretract_use_authorized": True,
        "receipt": file_record(HROI_POLICY_ADOPTION),
        "units": dict(units),
    }


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return [
            {str(key): str(value or "").strip() for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        encoding="utf-8",
        delete=False,
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        newline="",
        encoding="utf-8",
        delete=False,
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def environment(nthreads: int) -> dict[str, str]:
    env = V2.environment(nthreads)
    env["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(nthreads)
    return env


def run(
    command: list[str],
    *,
    log: Any,
    env: Mapping[str, str],
    outputs: tuple[Path, ...] = (),
) -> None:
    if outputs and all(
        path.is_file() and path.stat().st_size > 0 for path in outputs
    ):
        return
    for path in outputs:
        if path.is_file() or path.is_symlink():
            path.unlink()
        path.parent.mkdir(parents=True, exist_ok=True)
    log.write(f"\n[{utc_now()}] COMMAND {json.dumps(command)}\n")
    log.flush()
    completed = subprocess.run(
        command,
        stdout=log,
        stderr=subprocess.STDOUT,
        text=True,
        env=dict(env),
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(
            f"command failed rc={completed.returncode}: {command[0]}"
        )
    missing = [
        str(path)
        for path in outputs
        if not path.is_file() or path.stat().st_size <= 0
    ]
    if missing:
        raise RuntimeError("command outputs missing: " + ",".join(missing))


def validate_human_qc(path: Path, units: set[str]) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            f"genuine human-QC CSV is required for --execute: {path}"
        )
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = tuple(reader.fieldnames or ())
        forbidden = {
            "diagnosis",
            "diagnosis_at_dti",
            "group",
            "research_group",
            "outcome",
        }
        lowered = {str(field).strip().lower() for field in fields}
        if forbidden.intersection(lowered):
            raise ValueError("human-QC CSV is not diagnosis blind")
        required = {"unit", "status", "reviewer", "reviewed_utc"}
        if not required.issubset(fields):
            raise ValueError("human-QC CSV lacks required fields")
        rows = [
            {str(key): str(value or "").strip() for key, value in row.items()}
            for row in reader
        ]
    if len(rows) != len(units) or {row["unit"] for row in rows} != units:
        raise ValueError("human-QC rows differ from the exact Recovery4 set")
    for row in rows:
        if (
            row["status"].upper() != "PASS"
            or not row["reviewer"]
            or not row["reviewed_utc"]
        ):
            raise ValueError(f"human QC is incomplete for {row['unit']}")
        text = row["reviewed_utc"]
        reviewed_at = datetime.fromisoformat(
            text[:-1] + "+00:00" if text.endswith("Z") else text
        )
        if reviewed_at.tzinfo is None:
            raise ValueError(
                f"human-QC timestamp lacks timezone for {row['unit']}"
            )
        if reviewed_at > datetime.now(timezone.utc):
            raise ValueError(
                f"human-QC timestamp is in the future for {row['unit']}"
            )
    return file_record(path)


def validate_recovery4_gate(
    *,
    human_qc: Path,
    execute: bool,
) -> dict[str, Any]:
    for path in (
        AUDIT_CSV,
        AUDIT_SUMMARY,
        RECOVERY4_MASTER,
        RECOVERY4_REVIEW,
        CORRECTED_OVERLAY_QUANTITATIVE_AUDIT,
        FROZEN_RESPONSE_MANIFEST,
        CONFIG,
        V2_SOURCE,
        SCALAR_SOURCE,
        HROI_POLICY_ADOPTION_SOURCE,
        RELABEL,
        RELABEL_LUT,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    audit = load_json(AUDIT_SUMMARY)
    if (
        audit.get("status") != "PASS"
        or audit.get("audited_n") != EXPECTED_AUDIT_N
        or audit.get("diagnosis_labels_used") not in (False, None)
    ):
        raise ValueError("227-subject scale-up input audit differs")

    master = load_json(RECOVERY4_MASTER)
    master_units = {
        str(row["unit"]) for row in master.get("units", [])
    }
    if (
        master.get("record_type")
        != "diagnosis_blind_hcp379_pretract_recovery4_manifest"
        or master.get("status")
        != "AUTOMATED_PASS_AWAITING_HUMAN_VISUAL_QC"
        or master.get("diagnosis_labels_used") is not False
        or master.get("human_visual_qc_inferred") is not False
        or master.get("tractography_started") is not False
        or master.get("matrix_generation_started") is not False
        or len(master_units) != EXPECTED_CANARY_N
    ):
        raise ValueError("immutable Recovery4 pretract master differs")
    for name, record in master.get("automated_qc", {}).items():
        verify_file_record(record, label=f"Recovery4 {name} summary")
        summary = load_json(Path(str(record["path"])))
        if summary.get("status") != "PASS":
            raise ValueError(f"Recovery4 {name} summary is not PASS")
    for row in master["units"]:
        verify_file_record(
            row["manifest"], label=f"Recovery4 unit {row['unit']} manifest"
        )
        unit_manifest = load_json(Path(str(row["manifest"]["path"])))
        if (
            unit_manifest.get("unit") != row["unit"]
            or unit_manifest.get("status")
            != "AUTOMATED_PASS_AWAITING_HUMAN_VISUAL_QC"
            or unit_manifest.get("diagnosis_labels_used") is not False
        ):
            raise ValueError(
                f"Recovery4 unit manifest differs: {row['unit']}"
            )

    review = load_json(RECOVERY4_REVIEW)
    if (
        review.get("record_type")
        != (
            "diagnosis_blind_hcp379_visual_review_pack_"
            "recovery4_corrected_overlay"
        )
        or review.get("visualization_correction", {}).get(
            "human_hcp379_column_requires_re_review"
        )
        is not True
        or review.get("visualization_correction", {}).get(
            "imaging_derivatives_changed"
        )
        is not False
        or
        review.get("status") != "READY_FOR_HUMAN_REVIEW"
        or review.get("diagnosis_labels_used") is not False
        or review.get("human_visual_qc_inferred") is not False
        or review.get("unit_count") != EXPECTED_CANARY_N
        or {str(row["unit"]) for row in review.get("units", [])}
        != master_units
    ):
        raise ValueError("Recovery4 visual-review manifest differs")
    for name in ("review_pdf", "blank_human_qc_template", "pretract_manifest"):
        verify_file_record(review[name], label=f"review {name}")
    for row in review["units"]:
        verify_file_record(
            row["review_page"],
            label=f"corrected review page {row['unit']}",
        )
        verify_file_record(
            row["categorical_label_edges"],
            label=f"categorical HCP379 edges {row['unit']}",
        )
        for name, record in row["source_panels"].items():
            verify_file_record(
                record,
                label=f"corrected review panel {row['unit']}:{name}",
            )

    corrected_overlay_audit = load_json(
        CORRECTED_OVERLAY_QUANTITATIVE_AUDIT
    )
    if (
        corrected_overlay_audit.get("record_type")
        != (
            "diagnosis_blind_hcp379_recovery4_corrected_overlay_"
            "quantitative_audit"
        )
        or corrected_overlay_audit.get("diagnosis_labels_used") is not False
        or corrected_overlay_audit.get("unit_count") != EXPECTED_CANARY_N
        or corrected_overlay_audit.get("all_units_have_379_labels")
        is not True
    ):
        raise ValueError("corrected-overlay quantitative audit differs")
    raw_hard_holds = corrected_overlay_audit.get("hard_hold_units")
    if not isinstance(raw_hard_holds, list):
        raise TypeError("corrected-overlay hard-hold set is malformed")

    # Resolve a quantitative hard hold only through the exact same
    # hash-bound V6 promotion contract used by Phase B. This prevents the
    # scale-up workers from accepting the canonical human-QC CSV alone while
    # also preventing a valid promoted 009 route from being rejected by the
    # superseded pre-promotion audit.
    active_sources = PHASE_B_GATE.active_sources()
    active_master_path = Path(
        active_sources["pretract_manifest"]
    ).resolve()
    active_review_path = Path(active_sources["review_manifest"]).resolve()
    active_units = PHASE_B_GATE.units_from_pretract(active_master_path)
    if set(active_units) != master_units:
        raise ValueError("active Recovery4 gate unit set differs")
    active_gate = PHASE_B_GATE.validate_upstream(
        active_units,
        sources=active_sources,
    )
    if (
        active_gate.get("corrected_overlay_quantitative_audit")
        != file_record(CORRECTED_OVERLAY_QUANTITATIVE_AUDIT)
        or active_gate.get(
            "corrected_overlay_raw_hard_hold_units"
        )
        != raw_hard_holds
    ):
        raise ValueError("active Recovery4 quantitative gate differs")
    effective_hard_holds = active_gate.get(
        "corrected_overlay_hard_hold_units"
    )
    if not isinstance(effective_hard_holds, list):
        raise TypeError("active Recovery4 hard-hold set is malformed")
    if execute and effective_hard_holds:
        raise ValueError(
            "corrected-overlay quantitative hard holds remain: "
            + ",".join(str(unit) for unit in effective_hard_holds)
        )

    response = load_json(FROZEN_RESPONSE_MANIFEST)
    if (
        response.get("status") != "PASS"
        or response.get("diagnosis_labels_used") is not False
        or response.get("dummy_response_files_used") is not False
    ):
        raise ValueError("frozen response calibration differs")
    response_paths: dict[str, Path] = {}
    for tissue in ("wm", "gm", "csf"):
        record = response.get("pooled_responses", {}).get(tissue)
        if not isinstance(record, dict):
            raise ValueError(f"missing frozen {tissue} response")
        response_paths[tissue] = verify_file_record(
            record, label=f"pooled {tissue} response"
        )
    shell = response.get("fod_shell_compatibility", {})
    if shell.get("selected_shells_s_per_mm2") != [0.0, 1000.0]:
        raise ValueError("frozen FOD shell selection differs")

    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    lmax = int(config["fod"]["lmax"])
    bzero_threshold = float(
        config["dwi_preprocessing"]["registration_reference"][
            "b0_threshold_s_per_mm2"
        ]
    )
    if lmax != 6 or bzero_threshold != float(
        shell["bzero_threshold_s_per_mm2"]
    ):
        raise ValueError("recipe and frozen response shell contract differ")

    fod_order_audit = load_json(FOD_ORDER_AUDIT)
    fod_order_validation = load_json(FOD_ORDER_VALIDATION)
    fod_order_rows = read_csv(FOD_ORDER_TABLE)
    fod_order_by_unit = {
        str(row["unit"]): row for row in fod_order_rows
    }
    fod_order_holds = {
        unit
        for unit, row in fod_order_by_unit.items()
        if row.get("status")
        == "HOLD_ACQUISITION_AWARE_LMAX_REQUIRED"
    }
    if (
        fod_order_audit.get("record_type")
        != "hcp379_fod_order_compatibility_audit"
        or fod_order_audit.get("status")
        != "PASS_AUDIT_COMPLETE_WITH_ACQUISITION_AWARE_HOLDS"
        or fod_order_audit.get("diagnosis_labels_used") is not False
        or fod_order_audit.get("outcomes_used") is not False
        or fod_order_audit.get("connectome_density_used") is not False
        or fod_order_audit.get("observed_unit_n")
        != EXPECTED_HROI_POLICY_N
        or fod_order_audit.get("fixed_lmax6_compatible_n")
        != EXPECTED_HROI_POLICY_N - EXPECTED_FOD_ORDER_HOLD_N
        or fod_order_audit.get("acquisition_aware_hold_n")
        != EXPECTED_FOD_ORDER_HOLD_N
        or fod_order_validation.get("status") != "PASS"
        or fod_order_validation.get("checks_passed")
        != fod_order_validation.get("checks_total")
        or len(fod_order_rows) != EXPECTED_HROI_POLICY_N
        or len(fod_order_by_unit) != EXPECTED_HROI_POLICY_N
        or len(fod_order_holds) != EXPECTED_FOD_ORDER_HOLD_N
        or any(
            row.get("diagnosis_labels_used", "").lower() != "false"
            or row.get("outcomes_used", "").lower() != "false"
            or row.get("current_lmax") != "6"
            or row.get("required_even_sh_coefficient_n") != "28"
            for row in fod_order_rows
        )
    ):
        raise ValueError("FOD-order compatibility guard differs")

    failure_ledger_rows = read_csv(FAILURE_LEDGER)
    failure_ledger_validation = load_json(FAILURE_LEDGER_VALIDATION)
    failure_ledger_by_unit = {
        str(row["unit"]): row for row in failure_ledger_rows
    }
    mechanism_holds = {
        unit: row
        for unit, row in failure_ledger_by_unit.items()
        if row.get("status") == "FAILED"
    }
    if (
        len(failure_ledger_rows) != 515
        or len(failure_ledger_by_unit) != 515
        or failure_ledger_validation.get("status") != "PASS"
        or failure_ledger_validation.get("checks_passed")
        != failure_ledger_validation.get("checks_total")
        or any(
            not row.get("failure_class")
            or not row.get("recovery_requirement")
            for row in mechanism_holds.values()
        )
    ):
        raise ValueError("mechanism-specific failure ledger differs")

    human_record = (
        validate_human_qc(human_qc, master_units) if execute else None
    )
    hroi_policy = validate_hroi_policy_adoption(required=execute)
    return {
        "canary_units": sorted(master_units),
        "human_visual_qc": human_record,
        "hroi_source_policy_status": hroi_policy["status"],
        "hroi_source_policy_receipt": hroi_policy["receipt"],
        "hroi_source_policy_units": hroi_policy["units"],
        "recovery4_master": file_record(RECOVERY4_MASTER),
        "review_manifest": file_record(RECOVERY4_REVIEW),
        "corrected_overlay_quantitative_audit": file_record(
            CORRECTED_OVERLAY_QUANTITATIVE_AUDIT
        ),
        "corrected_overlay_raw_hard_hold_units": list(raw_hard_holds),
        "corrected_overlay_hard_hold_units": list(
            effective_hard_holds
        ),
        "active_recovery4_source_mode": active_gate.get("source_mode"),
        "active_recovery4_master": file_record(active_master_path),
        "active_review_manifest": file_record(active_review_path),
        "promotion_receipt": active_gate.get("promotion_receipt"),
        "frozen_response_manifest": file_record(
            FROZEN_RESPONSE_MANIFEST
        ),
        "pooled_responses": {
            tissue: file_record(path)
            for tissue, path in response_paths.items()
        },
        "response_paths": response_paths,
        "lmax": lmax,
        "bzero_threshold": bzero_threshold,
        "fod_order_audit": file_record(FOD_ORDER_AUDIT),
        "fod_order_table": file_record(FOD_ORDER_TABLE),
        "fod_order_validation": file_record(FOD_ORDER_VALIDATION),
        "fod_order_by_unit": fod_order_by_unit,
        "fod_order_hold_units": sorted(fod_order_holds),
        "failure_ledger": file_record(FAILURE_LEDGER),
        "failure_ledger_validation": file_record(
            FAILURE_LEDGER_VALIDATION
        ),
        "mechanism_hold_by_unit": mechanism_holds,
    }


def validate_audit_rows(
    gate: Mapping[str, Any],
) -> tuple[list[dict[str, str]], list[str]]:
    rows = read_csv(AUDIT_CSV)
    if (
        len(rows) != EXPECTED_AUDIT_N
        or len({row["unit"] for row in rows}) != EXPECTED_AUDIT_N
    ):
        raise ValueError("scale-up audit CSV does not contain exact 227 units")
    canary = set(str(unit) for unit in gate["canary_units"])
    overlap = sorted({row["unit"] for row in rows}.intersection(canary))
    if len(overlap) != EXPECTED_CANARY_OVERLAP_N:
        raise ValueError(
            f"corrected/canary overlap differs: {len(overlap)}"
        )
    for row in rows:
        unit = row["unit"]
        if (
            row.get("diagnosis_labels_used", "").lower() != "false"
            or not re.fullmatch(r"[0-9a-f]{64}", row.get("dti_source_id", ""))
            or not re.fullmatch(
                r"[0-9a-f]{64}", row.get("dti_raw_bundle_sha256", "")
            )
            or row.get("fastsurfer_t1_identity_match", "").lower()
            != "true"
            or row.get("hcp_source_parcellation_present", "").lower()
            != "true"
            or row.get("historical_nodes_b0_reuse_allowed", "").lower()
            != "false"
            or row.get("full_dwi_preprocessing_planned", "").lower()
            != "false"
            or row.get("expected_t1_image_id")
            != row.get("fastsurfer_t1_image_id")
            or row.get("route")
            not in {
                "READY_FOR_CORRECTED_PRETRACT_FROM_EDDY",
                "READY_AFTER_GRADIENT_HEADER_REPAIR",
            }
        ):
            raise ValueError(f"input-audit identity contract differs: {unit}")
        for field in (
            "eddy_path",
            "fastsurfer_t1_path",
            "hcp_source_parcellation_path",
        ):
            source = Path(row[field]).resolve()
            if not source.is_file() or source.is_symlink():
                raise ValueError(
                    f"audited source is not a regular file: {unit}:{field}"
                )
        eddy_size = int(row.get("eddy_size_bytes", "0"))
        if Path(row["eddy_path"]).stat().st_size != eddy_size:
            raise ValueError(f"audited Eddy size differs: {unit}")
        if row["route"] == "READY_AFTER_GRADIENT_HEADER_REPAIR":
            for suffix in ("bvec", "bval"):
                sidecar = DERIV / "mif_dwi" / f"{unit}.{suffix}"
                if not sidecar.is_file() or sidecar.is_symlink():
                    raise ValueError(
                        f"gradient-repair sidecar missing: {unit}.{suffix}"
                    )
    policy_units = gate.get("hroi_source_policy_units")
    policy_status = gate.get("hroi_source_policy_status")
    if policy_status == "PASS_OPERATIONAL_PRETRACT_ADOPTION":
        if (
            not isinstance(policy_units, Mapping)
            or len(policy_units) != EXPECTED_HROI_POLICY_N
        ):
            raise ValueError("HROI policy unit map differs")
        rebound: list[dict[str, str]] = []
        for audited in rows:
            row = dict(audited)
            unit = row["unit"]
            policy = policy_units.get(unit)
            if not isinstance(policy, Mapping):
                raise ValueError(f"HROI policy lacks audited unit: {unit}")
            original_path = Path(
                row["hcp_source_parcellation_path"]
            ).resolve()
            original_record = file_record(original_path)
            if policy.get("original_source") != original_record:
                raise ValueError(
                    f"HROI original-source binding differs: {unit}"
                )
            candidate_manifest_path = verify_file_record(
                policy["candidate_manifest"],
                label=f"HROI candidate manifest {unit}",
            )
            candidate_path = verify_file_record(
                policy["candidate"],
                label=f"HROI candidate source {unit}",
            )
            candidate_manifest = load_json(candidate_manifest_path)
            if (
                candidate_manifest.get("record_type")
                != (
                    "diagnosis_blind_hcp379_hroi_surface_support_"
                    "candidate"
                )
                or candidate_manifest.get("status")
                != "PASS_ALL_379_SOURCE_LABELS"
                or candidate_manifest.get("diagnosis_labels_used")
                is not False
                or candidate_manifest.get("source_files_modified")
                is not False
                or candidate_manifest.get("unit") != unit
                or candidate_manifest.get("present_source_label_n") != 379
                or candidate_manifest.get("missing_source_labels") != []
                or candidate_manifest.get("candidate")
                != policy["candidate"]
                or candidate_manifest.get("original_source")
                != original_record
                or candidate_manifest.get("changed_voxel_n")
                != policy.get("changed_voxel_n")
                or candidate_manifest.get("policy", {}).get(
                    "subcortical_voxels_overwritten"
                )
                != 0
                or candidate_manifest.get("policy", {}).get(
                    "other_cortical_labels_overwritten"
                )
                != 0
            ):
                raise ValueError(f"HROI candidate contract differs: {unit}")
            row.update(
                {
                    "hcp_source_parcellation_original_path": str(
                        original_path
                    ),
                    "hcp_source_parcellation_original_sha256": str(
                        original_record["sha256"]
                    ),
                    "hcp_source_parcellation_path": str(candidate_path),
                    "hcp_source_parcellation_candidate_sha256": str(
                        policy["candidate"]["sha256"]
                    ),
                    "hcp_source_parcellation_policy_manifest_path": str(
                        candidate_manifest_path
                    ),
                    "hcp_source_parcellation_policy_manifest_sha256": str(
                        policy["candidate_manifest"]["sha256"]
                    ),
                    "hcp_source_parcellation_policy": (
                        "HROI_SURFACE_SUPPORT_V1_COHORT_UNIFORM"
                    ),
                }
            )
            rebound.append(row)
        rows = rebound
    elif policy_status != "WAITING_FOR_530_HROI_POLICY_ADOPTION":
        raise ValueError("HROI source-policy status differs")
    selected = [row for row in rows if row["unit"] not in canary]
    if len(selected) != EXPECTED_SCALEUP_N:
        raise ValueError(f"non-canary scale-up differs: {len(selected)}")
    return sorted(selected, key=lambda row: row["unit"]), overlap


def tool_contract() -> dict[str, dict[str, Any]]:
    tools = {
        "mrconvert": MRTRIX / "mrconvert",
        "dwibiascorrect": MRTRIX / "dwibiascorrect",
        "dwiextract": MRTRIX / "dwiextract",
        "dwi2mask": MRTRIX / "dwi2mask",
        "mrmath": MRTRIX / "mrmath",
        "mrthreshold": MRTRIX / "mrthreshold",
        "mrcalc": MRTRIX / "mrcalc",
        "mrtransform": MRTRIX / "mrtransform",
        "mrgrid": MRTRIX / "mrgrid",
        "transformconvert": MRTRIX / "transformconvert",
        "dwi2tensor": MRTRIX / "dwi2tensor",
        "tensor2metric": MRTRIX / "tensor2metric",
        "mtnormalise": MRTRIX / "mtnormalise",
        "5ttgen": MRTRIX / "5ttgen",
        "5ttcheck": MRTRIX / "5ttcheck",
        "5tt2gmwmi": MRTRIX / "5tt2gmwmi",
        "bet": FSL / "bet",
        "epi_reg": FSL / "epi_reg",
        "convert_xfm": FSL / "convert_xfm",
        "flirt": FSL / "flirt",
        "antsRegistrationSyNQuick": ANTS / "antsRegistrationSyNQuick.sh",
        "antsApplyTransforms": ANTS / "antsApplyTransforms",
        "N4BiasFieldCorrection": ANTS / "N4BiasFieldCorrection",
        "mri_vol2vol": FREESURFER / "bin/mri_vol2vol",
        "slices": FSL / "slices",
        "ss3t_csd_beta1": V2.SS3T_SCRIPT,
        "ss3t_python": V2.SS3T_PYTHON,
        "fsl_python": FSL_PYTHON,
        "relabel_hcp": RELABEL,
    }
    return {name: file_record(path) for name, path in tools.items()}


def build_plan(
    *,
    root: Path,
    gate: Mapping[str, Any],
    rows: list[dict[str, str]],
    overlap: list[str],
    execute: bool,
    limit: int | None,
    compact_reproducible_after_pass: bool,
) -> dict[str, Any]:
    selected = rows[:limit] if limit is not None else rows
    hroi_policy_ready = (
        gate.get("hroi_source_policy_status")
        == "PASS_OPERATIONAL_PRETRACT_ADOPTION"
    )
    if execute and not hroi_policy_ready:
        raise ValueError(
            "pretract execution requires the adopted 530-subject HROI policy"
        )
    plan = {
        "schema_version": "1.0.0",
        "record_type": (
            "diagnosis_blind_hcp379_recovery4_scaleup_pretract_plan"
        ),
        "status": (
            "EXECUTION_GATE_PASS"
            if execute
            else (
                "PRE_HUMAN_DRY_RUN_PASS"
                if hroi_policy_ready
                else "WAITING_FOR_530_HROI_POLICY_ADOPTION"
            )
        ),
        "generated_utc": utc_now(),
        "diagnosis_labels_used": False,
        "non_overwriting": True,
        "execution_requested": execute,
        "imaging_executed_by_plan": False,
        "hroi_source_policy_ready": hroi_policy_ready,
        "pretract_execution_authorized": bool(
            execute and hroi_policy_ready
        ),
        "streaming_compaction_requested": (
            compact_reproducible_after_pass
        ),
        "streaming_compaction_policy": {
            "write_ahead_sha256_plan_required": True,
            "raw_tensor_maps_preserved": True,
            "bounded_tensor_maps_preserved": True,
            "selected_5tt_gmwmi_hcp379_wmfod_preserved": True,
            "unclassified_artifacts_preserved": True,
            "only_classified_reproducible_working_images_pruned": True,
        },
        "output_root": str(root.resolve()),
        "target_n": EXPECTED_SCALEUP_N,
        "selected_this_invocation_n": len(selected),
        "canary_overlap_excluded_n": len(overlap),
        "canary_overlap_excluded": overlap,
        "route_counts": dict(
            sorted(Counter(row["route"] for row in rows).items())
        ),
        "registration_policy": {
            "primary": "FSL BBR",
            "failover": "ANTs seeded rigid",
            "failover_random_seed": ANTS_RANDOM_SEED,
            "failover_trigger": (
                "direct HCP379 QC failure, either cortical hemisphere "
                "inside-DWI fraction below 0.80, or 5TT/DWI Dice below 0.70"
            ),
            "silent_fallback": False,
            "select_route_by_connectome_density": False,
        },
        "tensor_policy": {
            "raw_maps_preserved": True,
            "bounded_maps_separate": True,
            "maximum_fraction_per_raw_anomaly_condition": (
                SCALAR.RAW_ANOMALY_FRACTION_MAXIMUM
            ),
        },
        "gate_records": {
            key: value
            for key, value in gate.items()
            if key
            in {
                "human_visual_qc",
                "recovery4_master",
                "review_manifest",
                "corrected_overlay_quantitative_audit",
                "corrected_overlay_raw_hard_hold_units",
                "corrected_overlay_hard_hold_units",
                "active_recovery4_source_mode",
                "active_recovery4_master",
                "active_review_manifest",
                "promotion_receipt",
                "frozen_response_manifest",
                "pooled_responses",
                "fod_order_audit",
                "fod_order_table",
                "fod_order_validation",
                "fod_order_hold_units",
                "failure_ledger",
                "failure_ledger_validation",
                "hroi_source_policy_status",
                "hroi_source_policy_receipt",
            }
        },
        "tool_contract": tool_contract(),
        "builder": file_record(Path(__file__)),
        "provisional_engine": file_record(V2_SOURCE),
        "scalar_qc_implementation": file_record(SCALAR_SOURCE),
        "compactor_implementation": file_record(COMPACTOR_SOURCE),
        "hroi_policy_adoption_implementation": file_record(
            HROI_POLICY_ADOPTION_SOURCE
        ),
        "units": [
            {
                "unit": row["unit"],
                "input_route": row["route"],
                "dti_source_id": row["dti_source_id"],
                "dti_raw_bundle_sha256": row["dti_raw_bundle_sha256"],
                "expected_t1_image_id": row["expected_t1_image_id"],
                "hcp_source_parcellation_policy": row.get(
                    "hcp_source_parcellation_policy",
                    "WAITING_FOR_530_HROI_POLICY_ADOPTION",
                ),
                **(
                    {
                        "hcp_source_parcellation_original": dict(
                            gate["hroi_source_policy_units"][row["unit"]][
                                "original_source"
                            ]
                        ),
                        "hcp_source_parcellation_selected": dict(
                            gate["hroi_source_policy_units"][row["unit"]][
                                "candidate"
                            ]
                        ),
                        "hcp_source_policy_candidate_manifest": dict(
                            gate["hroi_source_policy_units"][row["unit"]][
                                "candidate_manifest"
                            ]
                        ),
                    }
                    if hroi_policy_ready
                    else {}
                ),
            }
            for row in selected
        ],
    }
    stamp = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        + "-pretract-recovery4-plan.json"
    )
    atomic_json(root / "manifests/attempts" / stamp, plan)
    atomic_json(root / "manifests/pretract_recovery4_plan.json", plan)
    return plan


def install_provisional_engine_guards() -> None:
    """Allow the engine to finish, while withholding Recovery4 promotion."""

    def provisional_atlas_qc(*args: Any, **kwargs: Any) -> dict[str, Any]:
        original = ORIGINAL_ATLAS_QC(*args, **kwargs)
        value = dict(original)
        value["provisional_engine_only"] = True
        value["original_status"] = original.get("status")
        value["original_failures"] = list(original.get("failures", []))
        # Direct Recovery4 QC is rerun after the engine and is authoritative.
        value["status"] = "PASS"
        value["failures"] = []
        return value

    V2.ATLAS_QC_MODULE.atlas_qc = provisional_atlas_qc
    scforge_path = str(EXP / "scforge")
    if scforge_path not in sys.path:
        sys.path.insert(0, scforge_path)
    import scforge.input_contract as input_contract

    # The engine's any-voxel extrema gate is intentionally superseded.
    # Recovery4 computes condition counts and applies the locked 1% fraction
    # gate before promotion.
    input_contract.assess_pre_tractography_image_ranges = lambda ranges: []

    def provisional_mrstats_range(
        path: Path, mask: Path | None = None
    ) -> dict[str, float]:
        """Aggregate extrema across scalar or multi-volume MRtrix images."""

        value = all_volume_range(path, mask)
        return {
            "min": float(value["min"]),
            "max": float(value["max"]),
        }

    # The provisional v2 engine assumed that mrstats always emitted one
    # min/max pair.  FOD spherical-harmonic images emit one pair per
    # coefficient volume.  Recovery4 must aggregate all finite volumes before
    # the authoritative fraction-based scalar/FOD QC is rerun below.
    V2.mrstats_range = provisional_mrstats_range


def all_volume_range(path: Path, mask: Path | None = None) -> dict[str, Any]:
    command = [
        str(MRTRIX / "mrstats"),
        str(path),
    ]
    if mask is not None:
        command.extend(("-mask", str(mask)))
    command.extend(("-output", "min", "-output", "max", "-quiet"))
    result = subprocess.run(
        command, check=True, capture_output=True, text=True
    )
    values = [float(value) for value in result.stdout.split()]
    if not values or len(values) % 2:
        raise ValueError(f"unexpected mrstats output: {path}")
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"non-finite image values: {path}")
    minima = values[0::2]
    maxima = values[1::2]
    return {
        "volume_count": len(minima),
        "min": min(minima),
        "max": max(maxima),
        "all_finite": True,
    }


def archive_prior_subject_state(
    path: Path,
    *,
    root: Path,
    unit: str,
) -> dict[str, Any] | None:
    """Preserve a failed or interrupted state before a subject-scoped retry."""

    if not path.is_file():
        return None
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    prior = load_json(path)
    archive = (
        root
        / "qc/subjects/attempts"
        / unit
        / f"prior-state-{digest}.json"
    )
    record = {
        "schema_version": "1.0.0",
        "record_type": "hcp379_pretract_recovery4_prior_state_archive",
        "unit": unit,
        "archived_utc": utc_now(),
        "source_path": str(path.resolve()),
        "source_size_bytes": len(raw),
        "source_sha256": digest,
        "prior_state": prior,
    }
    if archive.is_file():
        existing = load_json(archive)
        if (
            existing.get("record_type")
            != record["record_type"]
            or existing.get("unit") != unit
            or existing.get("source_size_bytes") != len(raw)
            or existing.get("source_sha256") != digest
            or existing.get("prior_state") != prior
        ):
            raise ValueError(f"prior-state archive collision: {archive}")
    else:
        atomic_json(archive, record)
    return file_record(archive)


def validate_reusable_pass_engine_state(
    path: Path,
    *,
    unit: str,
) -> dict[str, Any]:
    """Replay a completed provisional engine before a subject-scoped retry.

    This permits a later Recovery4 QC/spatial recovery to reuse expensive,
    already-completed image construction without silently trusting file
    presence.  The original engine record is not rewritten.  Every declared
    artifact is reopened, checked against its recorded path and byte size, and
    hashed for the recovery state.
    """

    state = load_json(path)
    artifacts = state.get("artifacts")
    if (
        state.get("record_type")
        != "diagnosis_blind_hcp379_scaleup_pretract"
        or state.get("status") != "PASS_PRETRACT"
        or state.get("unit") != unit
        or state.get("diagnosis_labels_used") is not False
        or not isinstance(artifacts, Mapping)
    ):
        raise ValueError(f"provisional engine is not reusable: {unit}")
    replay: dict[str, dict[str, Any]] = {}
    for name, record in artifacts.items():
        if (
            not isinstance(record, Mapping)
            or not isinstance(record.get("path"), str)
            or not isinstance(record.get("size_bytes"), int)
        ):
            raise ValueError(f"engine artifact record differs: {unit}:{name}")
        artifact = Path(str(record["path"])).resolve()
        current = file_record(artifact)
        if (
            current["path"] != str(artifact)
            or current["size_bytes"] != record["size_bytes"]
        ):
            raise ValueError(f"engine artifact drift: {unit}:{name}")
        replay[str(name)] = current
    required = {
        "dwi",
        "mask",
        "b0",
        "t1",
        "five_tt",
        "gmwmi",
        "hcp_nodes",
        "hcp_volumes",
        "wmfod_norm",
        "fa",
        "md",
        "rd",
        "ad",
        "atlas_qc",
        "automated_qc",
    }
    missing = sorted(required - set(replay))
    if missing:
        raise ValueError(
            f"reusable engine artifacts missing: {unit}:{missing}"
        )
    return {
        "state": state,
        "state_record": file_record(path),
        "artifact_hash_replay": replay,
    }


def image_mask_dice(
    first: Path,
    second: Path,
) -> tuple[bool, float]:
    first_image = nib.load(str(first))
    second_image = nib.load(str(second))
    geometry_ok = (
        first_image.shape == second_image.shape
        and np.allclose(
            first_image.affine,
            second_image.affine,
            atol=GEOMETRY_AFFINE_TOLERANCE,
        )
    )
    if not geometry_ok:
        return False, 0.0
    first_mask = np.asanyarray(first_image.dataobj) > 0
    second_mask = np.asanyarray(second_image.dataobj) > 0
    denominator = int(first_mask.sum() + second_mask.sum())
    dice = (
        2.0 * float(np.logical_and(first_mask, second_mask).sum())
        / denominator
        if denominator
        else 0.0
    )
    return True, dice


def spatial_paths(root: Path, unit: str) -> dict[str, Path]:
    base = V2.stage_paths(root, unit)
    subject = base["subject"]
    ants = subject / "03_spatial/recovery4_ants"
    qc = subject / "06_preflight"
    base.update(
        {
            "recovery4_lock": root / "locks_recovery4" / f"{unit}.lock",
            "engine_state": qc / "provisional_v2_engine_state.json",
            "route_qc": qc / "spatial_route_recovery4_qc.json",
            "dwi_mask_nifti": qc / "dwi_mask_recovery4.nii.gz",
            "bbr_five_tt_sum": qc / "bbr_5tt_sum_recovery4.mif",
            "bbr_five_tt_mask": qc / "bbr_5tt_mask_recovery4.nii.gz",
            "bbr_atlas_qc": qc / "bbr_hcp379_atlas_qc_recovery4.json",
            "bbr_t1_in_b0": (
                subject / "03_spatial/t1_in_b0_bbr_recovery4.nii.gz"
            ),
            "bbr_t1_overlay": qc / "review_b0_vs_t1_bbr_recovery4.png",
            "bbr_five_tt_overlay": (
                qc / "review_b0_vs_5tt_bbr_recovery4.png"
            ),
            "ants_dir": ants,
            "ants_log": subject / "logs/subjects/ants_recovery4.log",
            "ants_t1_mask_sum": ants / "5tt_t1_sum.mif",
            "ants_t1_mask": ants / "5tt_t1_mask.nii.gz",
            "ants_t1_brain": ants / "t1_5tt_brain.nii.gz",
            "ants_prefix": ants / "t1_to_b0_rigid_",
            "ants_affine": ants / "t1_to_b0_rigid_0GenericAffine.mat",
            "ants_warped_t1": ants / "t1_to_b0_rigid_Warped.nii.gz",
            "ants_inverse_warped_b0": (
                ants / "t1_to_b0_rigid_InverseWarped.nii.gz"
            ),
            "ants_five_tt_t1_nifti": ants / "5tt_t1.nii.gz",
            "ants_five_tt_raw_nifti": ants / "5tt_dwi_raw.nii.gz",
            "ants_five_tt": ants / "5tt_dwi_recovery4.mif",
            "ants_gmwmi": ants / "gmwmi_dwi_recovery4.mif",
            "ants_five_tt_sum": qc / "ants_5tt_sum_recovery4.mif",
            "ants_five_tt_mask": qc / "ants_5tt_mask_recovery4.nii.gz",
            "ants_hcp_b0_raw": ants / "HCPMMP1+aseg_b0_1mm_raw.nii.gz",
            "ants_hcp_nodes": ants / "hcp379_nodes_b0_1mm.nii.gz",
            "ants_hcp_volumes": ants / "hcp379_node_volumes.csv",
            "ants_atlas_qc": qc / "ants_hcp379_atlas_qc_recovery4.json",
            "ants_atlas_native": ants / "hcp379_nodes_b0_native_qc.nii.gz",
            "ants_atlas_overlay": ants / "review_b0_vs_hcp379.png",
            "ants_t1_overlay": ants / "review_b0_vs_t1.png",
            "ants_five_tt_overlay": ants / "review_b0_vs_5tt.png",
        }
    )
    return base


def build_five_tt_mask_qc(
    *,
    five_tt: Path,
    gmwmi: Path,
    sum_path: Path,
    mask_path: Path,
    dwi_mask: Path,
    dwi_mask_nifti: Path,
    log: Any,
    env: Mapping[str, str],
    nthreads: int,
) -> dict[str, Any]:
    temporary_mask = mask_path.with_name(mask_path.name + ".partial.mif")
    run(
        [
            str(MRTRIX / "mrmath"),
            str(five_tt),
            "sum",
            str(sum_path),
            "-axis",
            "3",
            "-nthreads",
            str(nthreads),
            "-force",
            "-quiet",
        ],
        log=log,
        env=env,
        outputs=(sum_path,),
    )
    run(
        [
            str(MRTRIX / "mrthreshold"),
            str(sum_path),
            str(temporary_mask),
            "-abs",
            "0.5",
            "-force",
            "-quiet",
        ],
        log=log,
        env=env,
        outputs=(temporary_mask,),
    )
    run(
        [
            str(MRTRIX / "mrconvert"),
            str(temporary_mask),
            str(mask_path),
            "-strides",
            "-1,2,3",
            "-force",
            "-quiet",
        ],
        log=log,
        env=env,
        outputs=(mask_path,),
    )
    run(
        [
            str(MRTRIX / "mrconvert"),
            str(dwi_mask),
            str(dwi_mask_nifti),
            "-strides",
            "-1,2,3",
            "-datatype",
            "bit",
            "-force",
            "-quiet",
        ],
        log=log,
        env=env,
        outputs=(dwi_mask_nifti,),
    )
    geometry_ok, dice = image_mask_dice(mask_path, dwi_mask_nifti)
    five_range = all_volume_range(five_tt)
    gmwmi_range = all_volume_range(gmwmi)
    failures: list[str] = []
    if not geometry_ok:
        failures.append("selected_5tt_dwi_geometry_mismatch")
    if dice < MINIMUM_5TT_DWI_DICE:
        failures.append(
            f"selected_5tt_dwi_dice={dice:.6f}<"
            f"{MINIMUM_5TT_DWI_DICE:.6f}"
        )
    if five_range["min"] < -1.0e-6 or five_range["max"] > 1.000001:
        failures.append("five_tt_outside_0_1")
    if gmwmi_range["min"] < -1.0e-6 or gmwmi_range["max"] <= 0:
        failures.append("gmwmi_invalid_range")
    return {
        "status": "PASS" if not failures else "FAIL",
        "geometry_match": geometry_ok,
        "five_tt_dwi_mask_dice": dice,
        "minimum_five_tt_dwi_mask_dice": MINIMUM_5TT_DWI_DICE,
        "five_tt_range": five_range,
        "gmwmi_range": gmwmi_range,
        "failures": failures,
        "artifacts": {
            "five_tt": file_record(five_tt),
            "gmwmi": file_record(gmwmi),
            "five_tt_sum": file_record(sum_path),
            "five_tt_mask": file_record(mask_path),
            "dwi_mask_nifti": file_record(dwi_mask_nifti),
        },
    }


def direct_atlas_qc(
    *,
    unit: str,
    nodes: Path,
    volumes: Path,
    paths: Mapping[str, Path],
    output: Path,
) -> dict[str, Any]:
    qc = ORIGINAL_ATLAS_QC(
        unit=unit,
        nodes_path=nodes,
        reference_path=paths["b0_1mm"],
        mask_path=paths["mask_1mm"],
        node_volumes_path=volumes,
        source_node_volumes_path=paths["hcp_t1_volumes"],
    )
    nodes_image = nib.load(str(nodes))
    mask_image = nib.load(str(paths["mask_1mm"]))
    node_data = np.rint(
        np.asanyarray(nodes_image.dataobj)
    ).astype(np.int32)
    mask_data = np.asanyarray(mask_image.dataobj) > 0
    failures = list(qc.get("failures", []))
    hemisphere_inside: dict[str, float | None] = {
        "left": None,
        "right": None,
    }
    if (
        node_data.shape == mask_data.shape
        and np.allclose(
            nodes_image.affine,
            mask_image.affine,
            atol=1.0e-5,
        )
    ):
        for side, lower, upper in (
            ("left", 1, 180),
            ("right", 181, 360),
        ):
            cortical = (node_data >= lower) & (node_data <= upper)
            denominator = int(np.count_nonzero(cortical))
            fraction = (
                float(np.count_nonzero(cortical & mask_data))
                / denominator
                if denominator
                else 0.0
            )
            hemisphere_inside[side] = fraction
            if (
                fraction
                < MINIMUM_CORTICAL_HEMISPHERE_ATLAS_INSIDE_DWI_MASK
            ):
                failures.append(
                    f"{side}_cortical_atlas_inside_dwi_mask="
                    f"{fraction:.6f}<"
                    f"{MINIMUM_CORTICAL_HEMISPHERE_ATLAS_INSIDE_DWI_MASK:.6f}"
                )
    else:
        failures.append(
            "cortical_hemisphere_atlas_inside_dwi_mask_unavailable"
        )
    qc["cortical_hemisphere_inside_dwi_mask_fraction"] = (
        hemisphere_inside
    )
    qc[
        "minimum_cortical_hemisphere_atlas_inside_dwi_mask_fraction"
    ] = MINIMUM_CORTICAL_HEMISPHERE_ATLAS_INSIDE_DWI_MASK
    qc["failures"] = failures
    qc["status"] = "PASS" if not failures else "FAIL"
    atomic_json(output, qc)
    return qc


def assess_bbr(
    *,
    unit: str,
    paths: Mapping[str, Path],
    log: Any,
    env: Mapping[str, str],
    nthreads: int,
) -> dict[str, Any]:
    tissue = build_five_tt_mask_qc(
        five_tt=paths["five_tt"],
        gmwmi=paths["gmwmi"],
        sum_path=paths["bbr_five_tt_sum"],
        mask_path=paths["bbr_five_tt_mask"],
        dwi_mask=paths["mask"],
        dwi_mask_nifti=paths["dwi_mask_nifti"],
        log=log,
        env=env,
        nthreads=nthreads,
    )
    atlas = direct_atlas_qc(
        unit=unit,
        nodes=paths["hcp_nodes"],
        volumes=paths["hcp_volumes"],
        paths=paths,
        output=paths["bbr_atlas_qc"],
    )
    run(
        [
            str(FSL / "flirt"),
            "-in",
            str(paths["t1"]),
            "-ref",
            str(paths["b0"]),
            "-applyxfm",
            "-init",
            str(paths["t1_to_b0"]),
            "-interp",
            "trilinear",
            "-out",
            str(paths["bbr_t1_in_b0"]),
        ],
        log=log,
        env=env,
        outputs=(paths["bbr_t1_in_b0"],),
    )
    run(
        [
            str(FSL / "slices"),
            str(paths["b0"]),
            str(paths["bbr_t1_in_b0"]),
            "-o",
            str(paths["bbr_t1_overlay"]),
        ],
        log=log,
        env=env,
        outputs=(paths["bbr_t1_overlay"],),
    )
    run(
        [
            str(FSL / "slices"),
            str(paths["b0"]),
            str(paths["bbr_five_tt_mask"]),
            "-o",
            str(paths["bbr_five_tt_overlay"]),
        ],
        log=log,
        env=env,
        outputs=(paths["bbr_five_tt_overlay"],),
    )
    failures = list(tissue["failures"])
    if atlas.get("status") != "PASS":
        failures.extend(f"hcp379:{value}" for value in atlas["failures"])
    return {
        "status": "PASS" if not failures else "FAIL",
        "route": "fsl_bbr_primary",
        "transform_format": "fsl",
        "transform": file_record(paths["t1_to_b0"]),
        "tissue_qc": tissue,
        "atlas_qc": atlas,
        "failures": failures,
        "selected": {
            "five_tt": paths["five_tt"],
            "gmwmi": paths["gmwmi"],
            "hcp_nodes": paths["hcp_nodes"],
            "hcp_volumes": paths["hcp_volumes"],
            "atlas_qc": paths["bbr_atlas_qc"],
            "atlas_overlay": paths["atlas_overlay"],
            "t1_in_b0": paths["bbr_t1_in_b0"],
            "t1_overlay": paths["bbr_t1_overlay"],
            "five_tt_overlay": paths["bbr_five_tt_overlay"],
        },
    }


def build_ants_failover(
    *,
    unit: str,
    paths: Mapping[str, Path],
    env: Mapping[str, str],
    nthreads: int,
    transform_type: str = "r",
    route_name: str = "seeded_ants_rigid_failover",
) -> dict[str, Any]:
    if transform_type not in {"r", "a", "s"}:
        raise ValueError(f"unsupported ANTs failover type: {transform_type}")
    if route_name not in {
        "seeded_ants_rigid_failover",
        "seeded_ants_affine_recovery",
        "seeded_ants_syn_recovery",
    }:
        raise ValueError(f"unsupported ANTs failover route: {route_name}")
    paths["ants_dir"].mkdir(parents=True, exist_ok=True)
    paths["ants_log"].parent.mkdir(parents=True, exist_ok=True)
    with paths["ants_log"].open("a", encoding="utf-8") as log:
        t1_mask_mif = paths["ants_t1_mask"].with_name(
            "5tt_t1_mask.partial.mif"
        )
        registration_outputs = [
            paths["ants_affine"],
            paths["ants_warped_t1"],
            paths["ants_inverse_warped_b0"],
        ]
        if transform_type == "s":
            registration_outputs.extend(
                [paths["ants_warp"], paths["ants_inverse_warp"]]
            )
        run(
            [
                str(MRTRIX / "mrmath"),
                str(paths["five_tt_t1"]),
                "sum",
                str(paths["ants_t1_mask_sum"]),
                "-axis",
                "3",
                "-nthreads",
                str(nthreads),
                "-force",
                "-quiet",
            ],
            log=log,
            env=env,
            outputs=(paths["ants_t1_mask_sum"],),
        )
        run(
            [
                str(MRTRIX / "mrthreshold"),
                str(paths["ants_t1_mask_sum"]),
                str(t1_mask_mif),
                "-abs",
                "0.5",
                "-force",
                "-quiet",
            ],
            log=log,
            env=env,
            outputs=(t1_mask_mif,),
        )
        run(
            [
                str(MRTRIX / "mrconvert"),
                str(t1_mask_mif),
                str(paths["ants_t1_mask"]),
                "-strides",
                "-1,2,3",
                "-datatype",
                "bit",
                "-force",
                "-quiet",
            ],
            log=log,
            env=env,
            outputs=(paths["ants_t1_mask"],),
        )
        run(
            [
                str(MRTRIX / "mrcalc"),
                str(paths["t1"]),
                str(paths["ants_t1_mask"]),
                "-mult",
                str(paths["ants_t1_brain"]),
                "-force",
                "-quiet",
            ],
            log=log,
            env=env,
            outputs=(paths["ants_t1_brain"],),
        )
        run(
            [
                str(ANTS / "antsRegistrationSyNQuick.sh"),
                "-d",
                "3",
                "-f",
                str(paths["b0"]),
                "-m",
                str(paths["ants_t1_brain"]),
                "-x",
                f"{paths['dwi_mask_nifti']},{paths['ants_t1_mask']}",
                "-t",
                transform_type,
                "-y",
                "0",
                "-e",
                str(ANTS_RANDOM_SEED),
                "-n",
                str(nthreads),
                "-o",
                str(paths["ants_prefix"]),
            ],
            log=log,
            env=env,
            outputs=tuple(registration_outputs),
        )
        transform_arguments = (
            [
                "-t",
                str(paths["ants_warp"]),
                "-t",
                str(paths["ants_affine"]),
            ]
            if transform_type == "s"
            else ["-t", str(paths["ants_affine"])]
        )
        run(
            [
                str(MRTRIX / "mrconvert"),
                str(paths["five_tt_t1"]),
                str(paths["ants_five_tt_t1_nifti"]),
                "-force",
                "-quiet",
            ],
            log=log,
            env=env,
            outputs=(paths["ants_five_tt_t1_nifti"],),
        )
        run(
            [
                str(ANTS / "antsApplyTransforms"),
                "-d",
                "3",
                "-e",
                "3",
                "-i",
                str(paths["ants_five_tt_t1_nifti"]),
                "-r",
                str(paths["b0"]),
                "-o",
                str(paths["ants_five_tt_raw_nifti"]),
                "-n",
                "Linear",
                *transform_arguments,
                "--float",
                "1",
            ],
            log=log,
            env=env,
            outputs=(paths["ants_five_tt_raw_nifti"],),
        )
        run(
            [
                str(MRTRIX / "mrcalc"),
                str(paths["ants_five_tt_raw_nifti"]),
                "0",
                "-max",
                "1",
                "-min",
                str(paths["ants_five_tt"]),
                "-force",
                "-quiet",
            ],
            log=log,
            env=env,
            outputs=(paths["ants_five_tt"],),
        )
        run(
            [str(MRTRIX / "5ttcheck"), str(paths["ants_five_tt"])],
            log=log,
            env=env,
        )
        run(
            [
                str(MRTRIX / "5tt2gmwmi"),
                str(paths["ants_five_tt"]),
                str(paths["ants_gmwmi"]),
                "-force",
                "-quiet",
            ],
            log=log,
            env=env,
            outputs=(paths["ants_gmwmi"],),
        )
        run(
            [
                str(ANTS / "antsApplyTransforms"),
                "-d",
                "3",
                "-i",
                str(paths["hcp_t1_raw"]),
                "-r",
                str(paths["b0_1mm"]),
                "-o",
                str(paths["ants_hcp_b0_raw"]),
                "-n",
                "GenericLabel",
                *transform_arguments,
            ],
            log=log,
            env=env,
            outputs=(paths["ants_hcp_b0_raw"],),
        )
        run(
            [
                str(FSL_PYTHON),
                str(RELABEL),
                str(paths["ants_hcp_b0_raw"]),
                str(RELABEL_LUT),
                str(paths["ants_hcp_nodes"]),
                str(paths["ants_hcp_volumes"]),
            ],
            log=log,
            env=env,
            outputs=(
                paths["ants_hcp_nodes"],
                paths["ants_hcp_volumes"],
            ),
        )

        deformation_qc: dict[str, Any] | None = None
        if transform_type == "s":
            run(
                [
                    str(ANTS / "CreateJacobianDeterminantImage"),
                    "3",
                    str(paths["ants_warp"]),
                    str(paths["ants_jacobian"]),
                    "0",
                    "1",
                ],
                log=log,
                env=env,
                outputs=(paths["ants_jacobian"],),
            )
            jacobian_range = V2.mrstats_range(paths["ants_jacobian"])
            deformation_qc = {
                "status": (
                    "PASS"
                    if jacobian_range["min"] > 0.0
                    else "FAIL"
                ),
                "jacobian_determinant_range": jacobian_range,
                "all_jacobians_positive": jacobian_range["min"] > 0.0,
                "jacobian": file_record(paths["ants_jacobian"]),
            }

        tissue = build_five_tt_mask_qc(
            five_tt=paths["ants_five_tt"],
            gmwmi=paths["ants_gmwmi"],
            sum_path=paths["ants_five_tt_sum"],
            mask_path=paths["ants_five_tt_mask"],
            dwi_mask=paths["mask"],
            dwi_mask_nifti=paths["dwi_mask_nifti"],
            log=log,
            env=env,
            nthreads=nthreads,
        )
        atlas = direct_atlas_qc(
            unit=unit,
            nodes=paths["ants_hcp_nodes"],
            volumes=paths["ants_hcp_volumes"],
            paths=paths,
            output=paths["ants_atlas_qc"],
        )
        run(
            [
                str(MRTRIX / "mrtransform"),
                str(paths["ants_hcp_nodes"]),
                str(paths["ants_atlas_native"]),
                "-template",
                str(paths["b0"]),
                "-interp",
                "nearest",
                "-force",
                "-quiet",
            ],
            log=log,
            env=env,
            outputs=(paths["ants_atlas_native"],),
        )
        run(
            [
                str(FSL / "slices"),
                str(paths["b0"]),
                str(paths["ants_atlas_native"]),
                "-o",
                str(paths["ants_atlas_overlay"]),
            ],
            log=log,
            env=env,
            outputs=(paths["ants_atlas_overlay"],),
        )
        run(
            [
                str(FSL / "slices"),
                str(paths["b0"]),
                str(paths["ants_warped_t1"]),
                "-o",
                str(paths["ants_t1_overlay"]),
            ],
            log=log,
            env=env,
            outputs=(paths["ants_t1_overlay"],),
        )
        run(
            [
                str(FSL / "slices"),
                str(paths["b0"]),
                str(paths["ants_five_tt_mask"]),
                "-o",
                str(paths["ants_five_tt_overlay"]),
            ],
            log=log,
            env=env,
            outputs=(paths["ants_five_tt_overlay"],),
        )

    failures = list(tissue["failures"])
    if atlas.get("status") != "PASS":
        failures.extend(f"hcp379:{value}" for value in atlas["failures"])
    if (
        deformation_qc is not None
        and deformation_qc["status"] != "PASS"
    ):
        failures.append("nonpositive_syn_jacobian")
    transform_records = (
        [
            file_record(paths["ants_warp"]),
            file_record(paths["ants_affine"]),
        ]
        if transform_type == "s"
        else [file_record(paths["ants_affine"])]
    )
    return {
        "status": "PASS" if not failures else "FAIL",
        "route": route_name,
        "transform_format": "ants",
        "ants_transform_type": transform_type,
        "transform": transform_records[0],
        "transforms_in_application_order": transform_records,
        "random_seed": ANTS_RANDOM_SEED,
        "deformation_qc": deformation_qc,
        "tissue_qc": tissue,
        "atlas_qc": atlas,
        "failures": failures,
        "selected": {
            "five_tt": paths["ants_five_tt"],
            "gmwmi": paths["ants_gmwmi"],
            "hcp_nodes": paths["ants_hcp_nodes"],
            "hcp_volumes": paths["ants_hcp_volumes"],
            "atlas_qc": paths["ants_atlas_qc"],
            "atlas_overlay": paths["ants_atlas_overlay"],
            "t1_in_b0": paths["ants_warped_t1"],
            "t1_overlay": paths["ants_t1_overlay"],
            "five_tt_overlay": paths["ants_five_tt_overlay"],
        },
    }


def ants_affine_variant_paths(
    paths: Mapping[str, Path],
) -> dict[str, Path]:
    """Return a disjoint path namespace for the affine recovery candidate."""

    variant = dict(paths)
    subject = paths["subject"]
    directory = subject / "03_spatial/recovery4_ants_affine"
    qc = subject / "06_preflight"
    variant.update(
        {
            "ants_dir": directory,
            "ants_log": (
                subject / "logs/subjects/ants_affine_recovery4.log"
            ),
            "ants_t1_mask_sum": directory / "5tt_t1_sum.mif",
            "ants_t1_mask": directory / "5tt_t1_mask.nii.gz",
            "ants_t1_brain": directory / "t1_5tt_brain.nii.gz",
            "ants_prefix": directory / "t1_to_b0_affine_",
            "ants_affine": (
                directory / "t1_to_b0_affine_0GenericAffine.mat"
            ),
            "ants_warped_t1": (
                directory / "t1_to_b0_affine_Warped.nii.gz"
            ),
            "ants_inverse_warped_b0": (
                directory / "t1_to_b0_affine_InverseWarped.nii.gz"
            ),
            "ants_five_tt_t1_nifti": directory / "5tt_t1.nii.gz",
            "ants_five_tt_raw_nifti": (
                directory / "5tt_dwi_raw.nii.gz"
            ),
            "ants_five_tt": directory / "5tt_dwi_recovery4.mif",
            "ants_gmwmi": directory / "gmwmi_dwi_recovery4.mif",
            "ants_five_tt_sum": qc / "ants_affine_5tt_sum_recovery4.mif",
            "ants_five_tt_mask": (
                qc / "ants_affine_5tt_mask_recovery4.nii.gz"
            ),
            "ants_hcp_b0_raw": (
                directory / "HCPMMP1+aseg_b0_1mm_raw.nii.gz"
            ),
            "ants_hcp_nodes": directory / "hcp379_nodes_b0_1mm.nii.gz",
            "ants_hcp_volumes": (
                directory / "hcp379_node_volumes.csv"
            ),
            "ants_atlas_qc": (
                qc / "ants_affine_hcp379_atlas_qc_recovery4.json"
            ),
            "ants_atlas_native": (
                directory / "hcp379_nodes_b0_native_qc.nii.gz"
            ),
            "ants_atlas_overlay": (
                directory / "review_b0_vs_hcp379.png"
            ),
            "ants_t1_overlay": directory / "review_b0_vs_t1.png",
            "ants_five_tt_overlay": directory / "review_b0_vs_5tt.png",
        }
    )
    return variant


def ants_syn_variant_paths(
    paths: Mapping[str, Path],
) -> dict[str, Path]:
    """Return a disjoint path namespace for the SyN recovery candidate."""

    variant = dict(paths)
    subject = paths["subject"]
    directory = subject / "03_spatial/recovery4_ants_syn"
    qc = subject / "06_preflight"
    variant.update(
        {
            "ants_dir": directory,
            "ants_log": subject / "logs/subjects/ants_syn_recovery4.log",
            "ants_t1_mask_sum": directory / "5tt_t1_sum.mif",
            "ants_t1_mask": directory / "5tt_t1_mask.nii.gz",
            "ants_t1_brain": directory / "t1_5tt_brain.nii.gz",
            "ants_prefix": directory / "t1_to_b0_syn_",
            "ants_affine": directory / "t1_to_b0_syn_0GenericAffine.mat",
            "ants_warp": directory / "t1_to_b0_syn_1Warp.nii.gz",
            "ants_inverse_warp": (
                directory / "t1_to_b0_syn_1InverseWarp.nii.gz"
            ),
            "ants_warped_t1": directory / "t1_to_b0_syn_Warped.nii.gz",
            "ants_inverse_warped_b0": (
                directory / "t1_to_b0_syn_InverseWarped.nii.gz"
            ),
            "ants_jacobian": (
                directory / "t1_to_b0_syn_jacobian_determinant.nii.gz"
            ),
            "ants_five_tt_t1_nifti": directory / "5tt_t1.nii.gz",
            "ants_five_tt_raw_nifti": directory / "5tt_dwi_raw.nii.gz",
            "ants_five_tt": directory / "5tt_dwi_recovery4.mif",
            "ants_gmwmi": directory / "gmwmi_dwi_recovery4.mif",
            "ants_five_tt_sum": qc / "ants_syn_5tt_sum_recovery4.mif",
            "ants_five_tt_mask": qc / "ants_syn_5tt_mask_recovery4.nii.gz",
            "ants_hcp_b0_raw": (
                directory / "HCPMMP1+aseg_b0_1mm_raw.nii.gz"
            ),
            "ants_hcp_nodes": directory / "hcp379_nodes_b0_1mm.nii.gz",
            "ants_hcp_volumes": directory / "hcp379_node_volumes.csv",
            "ants_atlas_qc": (
                qc / "ants_syn_hcp379_atlas_qc_recovery4.json"
            ),
            "ants_atlas_native": (
                directory / "hcp379_nodes_b0_native_qc.nii.gz"
            ),
            "ants_atlas_overlay": (
                directory / "review_b0_vs_hcp379.png"
            ),
            "ants_t1_overlay": directory / "review_b0_vs_t1.png",
            "ants_five_tt_overlay": directory / "review_b0_vs_5tt.png",
        }
    )
    return variant


def existing_final_pass(
    path: Path,
    *,
    root: Path,
    unit: str,
    row: Mapping[str, str],
    gate: Mapping[str, Any],
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        state = load_json(path)
        if state.get("status") != "PASS_PRETRACT_RECOVERY4":
            return None
        source_identity = state.get("source_identity")
        if not isinstance(source_identity, Mapping):
            return None
        selected_source = file_record(
            Path(row["hcp_source_parcellation_path"])
        )
        original_source = file_record(
            Path(row["hcp_source_parcellation_original_path"])
        )
        candidate_manifest = file_record(
            Path(row["hcp_source_parcellation_policy_manifest_path"])
        )
        if (
            source_identity.get("hcp_source_parcellation")
            != selected_source
            or source_identity.get(
                "hcp_source_parcellation_original"
            )
            != original_source
            or source_identity.get(
                "hcp_source_policy_candidate_manifest"
            )
            != candidate_manifest
            or source_identity.get(
                "hcp_source_policy_adoption_receipt"
            )
            != gate.get("hroi_source_policy_receipt")
        ):
            return None
        try:
            for name, record in state.get("artifacts", {}).items():
                verify_file_record(record, label=f"cached {name}")
        except Exception:
            compaction_path = (
                root / "qc/pretract_compaction" / f"{unit}.json"
            )
            compaction = load_json(compaction_path)
            retained = compaction.get("retained_artifacts")
            deleted = compaction.get("deleted_artifacts")
            required = {
                "selected_five_tt",
                "selected_gmwmi",
                "selected_hcp379_nodes",
                "selected_hcp379_node_volumes",
                "selected_hcp379_atlas_qc",
                "wmfod_normalised",
                "fa_raw",
                "md_raw",
                "rd_raw",
                "ad_raw",
                "fa_bounded",
                "md_bounded",
                "rd_bounded",
                "ad_bounded",
                "spatial_route_qc",
                "scalar_recovery4_qc",
            }
            if (
                compaction.get("record_type")
                != "diagnosis_blind_hcp379_pretract_compaction"
                or compaction.get("status") != "PASS_COMPACTED"
                or compaction.get("diagnosis_labels_used") is not False
                or compaction.get("recovery4_inputs") is not True
                or compaction.get("unit") != unit
                or Path(str(compaction.get("root", ""))).resolve()
                != root.resolve()
                or compaction.get("source_pretract_state")
                != file_record(path)
                or compaction.get("raw_tensor_maps_preserved") is not True
                or not isinstance(retained, Mapping)
                or not required.issubset(retained)
                or not isinstance(deleted, Mapping)
            ):
                return None
            for name in required:
                verify_file_record(
                    retained[name], label=f"cached compacted {name}"
                )
            dwi = deleted.get("dwi")
            if (
                not isinstance(dwi, Mapping)
                or dwi.get("deleted") is not True
                or Path(str(dwi.get("path", ""))).exists()
            ):
                return None
        return state
    except Exception:
        return None


def process_unit(
    row: Mapping[str, str],
    *,
    root: Path,
    gate: Mapping[str, Any],
    nthreads: int,
    reuse_valid_engine: bool = False,
    allow_affine_failover: bool = False,
    allow_syn_failover: bool = False,
) -> dict[str, Any]:
    unit = str(row["unit"])
    paths = spatial_paths(root, unit)
    paths["recovery4_lock"].parent.mkdir(parents=True, exist_ok=True)
    with paths["recovery4_lock"].open("w") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        fod_order = gate.get("fod_order_by_unit", {}).get(unit)
        if not isinstance(fod_order, Mapping):
            raise ValueError(f"FOD-order guard missing unit: {unit}")
        if (
            fod_order.get("status")
            == "HOLD_ACQUISITION_AWARE_LMAX_REQUIRED"
        ):
            try:
                existing_hold = load_json(paths["state"])
            except Exception:
                existing_hold = {}
            if (
                existing_hold.get("status")
                == "HOLD_FOD_ORDER_RECOVERY_REQUIRED"
                and existing_hold.get("unit") == unit
                and existing_hold.get("fod_order_guard")
                == gate.get("fod_order_audit")
            ):
                return existing_hold
            prior_state_archive = archive_prior_subject_state(
                paths["state"],
                root=root,
                unit=unit,
            )
            hold = {
                "schema_version": "1.0.0",
                "record_type": (
                    "diagnosis_blind_hcp379_scaleup_pretract_recovery4"
                ),
                "status": "HOLD_FOD_ORDER_RECOVERY_REQUIRED",
                "unit": unit,
                "diagnosis_labels_used": False,
                "outcomes_used": False,
                "connectome_density_used": False,
                "non_overwriting": True,
                "completed_utc": utc_now(),
                "input_route": row["route"],
                "current_frozen_lmax": int(
                    fod_order["current_lmax"]
                ),
                "required_even_sh_coefficient_n": int(
                    fod_order["required_even_sh_coefficient_n"]
                ),
                "selected_shell_unique_antipodal_direction_n": int(
                    fod_order[
                        "selected_shell_unique_antipodal_direction_n"
                    ]
                ),
                "recommended_lmax": int(
                    fod_order["recommended_lmax"]
                ),
                "recovery_requirement": (
                    "Validated non-overwriting acquisition-aware FOD-order "
                    "recovery; fixed lmax=6 promotion is prohibited."
                ),
                "fod_order_guard": gate["fod_order_audit"],
                "fod_order_table": gate["fod_order_table"],
                "fod_order_validation": gate["fod_order_validation"],
                "prior_state_archive": prior_state_archive,
                "imaging_executed_by_hold": False,
                "error": (
                    "Fixed lmax=6 requires 28 unique directions, but this "
                    f"unit has {fod_order['selected_shell_unique_antipodal_direction_n']}."
                ),
            }
            atomic_json(paths["state"], hold)
            return hold
        mechanism_hold = gate.get("mechanism_hold_by_unit", {}).get(unit)
        if isinstance(mechanism_hold, Mapping):
            try:
                existing_hold = load_json(paths["state"])
            except Exception:
                existing_hold = {}
            if (
                existing_hold.get("status")
                == "HOLD_MECHANISM_SPECIFIC_RECOVERY_REQUIRED"
                and existing_hold.get("unit") == unit
                and existing_hold.get("failure_ledger")
                == gate.get("failure_ledger")
            ):
                return existing_hold
            prior_state_archive = archive_prior_subject_state(
                paths["state"],
                root=root,
                unit=unit,
            )
            hold = {
                "schema_version": "1.0.0",
                "record_type": (
                    "diagnosis_blind_hcp379_scaleup_pretract_recovery4"
                ),
                "status": (
                    "HOLD_MECHANISM_SPECIFIC_RECOVERY_REQUIRED"
                ),
                "unit": unit,
                "diagnosis_labels_used": False,
                "outcomes_used": False,
                "connectome_density_used": False,
                "non_overwriting": True,
                "completed_utc": utc_now(),
                "input_route": row["route"],
                "failure_class": mechanism_hold["failure_class"],
                "recovery_requirement": mechanism_hold[
                    "recovery_requirement"
                ],
                "failure_ledger": gate["failure_ledger"],
                "failure_ledger_validation": gate[
                    "failure_ledger_validation"
                ],
                "prior_state_archive": prior_state_archive,
                "imaging_executed_by_hold": False,
                "error": (
                    "Fail-closed mechanism-specific hold: "
                    f"{mechanism_hold['failure_class']}"
                ),
            }
            atomic_json(paths["state"], hold)
            return hold
        cached = existing_final_pass(
            paths["state"],
            root=root,
            unit=unit,
            row=row,
            gate=gate,
        )
        if cached is not None:
            return cached
        prior_state_archive = archive_prior_subject_state(
            paths["state"],
            root=root,
            unit=unit,
        )
        started = time.monotonic()
        final: dict[str, Any] = {
            "schema_version": "1.0.0",
            "record_type": (
                "diagnosis_blind_hcp379_scaleup_pretract_recovery4"
            ),
            "status": "RUNNING_RECOVERY4",
            "unit": unit,
            "diagnosis_labels_used": False,
            "non_overwriting": True,
            "started_utc": utc_now(),
            "input_route": row["route"],
            "hcp_source_parcellation_policy": row.get(
                "hcp_source_parcellation_policy"
            ),
            "prior_state_archive": prior_state_archive,
            "provisional_engine_reuse_requested": reuse_valid_engine,
            "affine_failover_authorized": allow_affine_failover,
            "syn_failover_authorized": allow_syn_failover,
        }
        atomic_json(paths["state"], final)
        try:
            engine_gate = {
                "response_paths": gate["response_paths"],
                "pooled_responses": gate["pooled_responses"],
                "lmax": gate["lmax"],
                "bzero_threshold": gate["bzero_threshold"],
            }
            if reuse_valid_engine:
                engine_replay = validate_reusable_pass_engine_state(
                    paths["engine_state"],
                    unit=unit,
                )
                engine = engine_replay["state"]
                final["provisional_engine_reuse"] = {
                    "status": "PASS_HASH_REPLAY",
                    "engine_state": engine_replay["state_record"],
                    "artifact_hash_replay": engine_replay[
                        "artifact_hash_replay"
                    ],
                }
            else:
                engine = V2.process_unit(
                    row,
                    root=root,
                    gate=engine_gate,
                    nthreads=nthreads,
                )
                atomic_json(paths["engine_state"], engine)
            if engine.get("status") != "PASS_PRETRACT":
                raise RuntimeError(
                    "provisional image-construction engine failed: "
                    + str(engine.get("error"))
                )

            # Rebind the scalar helper to this versioned subject tree.
            SCALAR.RUN_ROOT = root
            scalar = SCALAR.process_unit(unit, output_root=root)
            if scalar.get("status") != "PASS":
                raise ValueError(
                    "Recovery4 scalar/FOD QC failed: "
                    + ";".join(scalar.get("failures", []))
                )

            env = environment(nthreads)
            with paths["log"].open("a", encoding="utf-8") as log:
                bbr = assess_bbr(
                    unit=unit,
                    paths=paths,
                    log=log,
                    env=env,
                    nthreads=nthreads,
                )
            candidates: dict[str, Any] = {"fsl_bbr_primary": bbr}
            if bbr["status"] == "PASS":
                selected = bbr
            else:
                ants = build_ants_failover(
                    unit=unit,
                    paths=paths,
                    env=env,
                    nthreads=nthreads,
                )
                candidates["seeded_ants_rigid_failover"] = ants
                if (
                    ants["status"] != "PASS"
                    and allow_affine_failover
                ):
                    affine = build_ants_failover(
                        unit=unit,
                        paths=ants_affine_variant_paths(paths),
                        env=env,
                        nthreads=nthreads,
                        transform_type="a",
                        route_name="seeded_ants_affine_recovery",
                    )
                    candidates["seeded_ants_affine_recovery"] = affine
                else:
                    affine = None
                if (
                    ants["status"] != "PASS"
                    and (
                        affine is None
                        or affine["status"] != "PASS"
                    )
                    and allow_syn_failover
                ):
                    syn = build_ants_failover(
                        unit=unit,
                        paths=ants_syn_variant_paths(paths),
                        env=env,
                        nthreads=nthreads,
                        transform_type="s",
                        route_name="seeded_ants_syn_recovery",
                    )
                    candidates["seeded_ants_syn_recovery"] = syn
                else:
                    syn = None
                if (
                    ants["status"] != "PASS"
                    and (
                        affine is None
                        or affine["status"] != "PASS"
                    )
                    and (
                        syn is None
                        or syn["status"] != "PASS"
                    )
                ):
                    raise ValueError(
                        "neither spatial route passed direct gates; "
                        f"bbr={bbr['failures']};"
                        f"ants={ants['failures']};"
                        f"affine={None if affine is None else affine['failures']};"
                        f"syn={None if syn is None else syn['failures']}"
                    )
                selected = (
                    syn
                    if syn is not None and syn["status"] == "PASS"
                    else (
                        affine
                        if (
                            affine is not None
                            and affine["status"] == "PASS"
                        )
                        else ants
                    )
                )

            route_record = {
                "schema_version": "1.0.0",
                "record_type": (
                    "diagnosis_blind_hcp379_scaleup_spatial_route_recovery4"
                ),
                "status": "PASS",
                "unit": unit,
                "diagnosis_labels_used": False,
                "completed_utc": utc_now(),
                "selection_uses_connectome_density": False,
                "primary_route": "fsl_bbr_primary",
                "failover_only_after_direct_qc_failure": True,
                "selected_route": selected["route"],
                "selected_transform_format": selected["transform_format"],
                "selected_transform": selected["transform"],
                "candidates": {
                    name: {
                        key: value
                        for key, value in candidate.items()
                        if key != "selected"
                    }
                    for name, candidate in candidates.items()
                },
            }
            atomic_json(paths["route_qc"], route_record)

            chosen = selected["selected"]
            artifacts = {
                "dwi_bias_corrected": file_record(paths["dwi"]),
                "mean_b0": file_record(paths["b0"]),
                "dwi_mask": file_record(paths["mask"]),
                "t1_n4": file_record(paths["t1"]),
                "selected_five_tt": file_record(chosen["five_tt"]),
                "selected_gmwmi": file_record(chosen["gmwmi"]),
                "selected_hcp379_nodes": file_record(chosen["hcp_nodes"]),
                "selected_hcp379_node_volumes": file_record(
                    chosen["hcp_volumes"]
                ),
                "selected_hcp379_atlas_qc": file_record(
                    chosen["atlas_qc"]
                ),
                "selected_hcp379_overlay": file_record(
                    chosen["atlas_overlay"]
                ),
                "selected_t1_in_b0": file_record(chosen["t1_in_b0"]),
                "selected_t1_overlay": file_record(
                    chosen["t1_overlay"]
                ),
                "selected_five_tt_overlay": file_record(
                    chosen["five_tt_overlay"]
                ),
                "selected_transform": selected["transform"],
                "spatial_route_qc": file_record(paths["route_qc"]),
                "wmfod_normalised": file_record(paths["wmfod_norm"]),
                "gm_normalised": file_record(paths["gm_norm"]),
                "csf_normalised": file_record(paths["csf_norm"]),
                "scalar_recovery4_qc": file_record(
                    Path(str(scalar["artifacts"]["wmfod_l0"]["path"])).parents[
                        1
                    ]
                    / "06_preflight/scalar_recovery4_qc.json"
                ),
                "wmfod_l0": scalar["artifacts"]["wmfod_l0"],
            }
            for metric in ("fa", "md", "rd", "ad"):
                artifacts[f"{metric}_raw"] = file_record(paths[metric])
                artifacts[f"{metric}_bounded"] = scalar["artifacts"][
                    f"{metric}_bounded"
                ]

            final.update(
                {
                    "status": "PASS_PRETRACT_RECOVERY4",
                    "completed_utc": utc_now(),
                    "elapsed_seconds": round(
                        time.monotonic() - started, 3
                    ),
                    "registration_route": selected["route"],
                    "selected_5tt_dwi_mask_dice": selected["tissue_qc"][
                        "five_tt_dwi_mask_dice"
                    ],
                    "hcp379_atlas_inside_dwi_mask_fraction": selected[
                        "atlas_qc"
                    ]["atlas_inside_dwi_mask_fraction"],
                    "maximum_raw_tensor_anomaly_fraction": max(
                        scalar["raw_anomaly_fractions"].values(),
                        default=0.0,
                    ),
                    "raw_tensor_maps_preserved": True,
                    "bounded_tensor_maps_separate": True,
                    "provisional_engine_promoted_directly": False,
                    "artifacts": artifacts,
                    "source_identity": {
                        "dti_source_id": row["dti_source_id"],
                        "dti_raw_bundle_sha256": row[
                            "dti_raw_bundle_sha256"
                        ],
                        "expected_t1_image_id": row[
                            "expected_t1_image_id"
                        ],
                        "eddy_source": file_record(Path(row["eddy_path"])),
                        "fastsurfer_t1_source": file_record(
                            Path(row["fastsurfer_t1_path"])
                        ),
                        "hcp_source_parcellation": file_record(
                            Path(row["hcp_source_parcellation_path"])
                        ),
                        "hcp_source_parcellation_original": file_record(
                            Path(
                                row[
                                    "hcp_source_parcellation_original_path"
                                ]
                            )
                        ),
                        "hcp_source_policy_candidate_manifest": file_record(
                            Path(
                                row[
                                    "hcp_source_parcellation_policy_"
                                    "manifest_path"
                                ]
                            )
                        ),
                        "hcp_source_policy_adoption_receipt": gate[
                            "hroi_source_policy_receipt"
                        ],
                    },
                    "error": None,
                }
            )
        except Exception as exc:
            final.update(
                {
                    "status": "FAIL_PRETRACT_RECOVERY4",
                    "completed_utc": utc_now(),
                    "elapsed_seconds": round(
                        time.monotonic() - started, 3
                    ),
                    "error": f"{type(exc).__name__}:{exc}",
                }
            )
        atomic_json(paths["state"], final)
        return final


def write_summary(root: Path) -> dict[str, Any]:
    states: list[dict[str, Any]] = []
    for path in sorted((root / "qc/subjects").glob("*.json")):
        try:
            state = load_json(path)
        except Exception:
            continue
        if state.get("record_type") == (
            "diagnosis_blind_hcp379_scaleup_pretract_recovery4"
        ):
            states.append(state)
    counts = Counter(str(state.get("status", "UNKNOWN")) for state in states)
    route_counts = Counter(
        str(state["registration_route"])
        for state in states
        if state.get("status") == "PASS_PRETRACT_RECOVERY4"
    )
    rows = [
        {
            "unit": state.get("unit"),
            "status": state.get("status"),
            "registration_route": state.get("registration_route"),
            "selected_5tt_dwi_mask_dice": state.get(
                "selected_5tt_dwi_mask_dice"
            ),
            "hcp379_atlas_inside_dwi_mask_fraction": state.get(
                "hcp379_atlas_inside_dwi_mask_fraction"
            ),
            "maximum_raw_tensor_anomaly_fraction": state.get(
                "maximum_raw_tensor_anomaly_fraction"
            ),
            "elapsed_seconds": state.get("elapsed_seconds"),
            "error": state.get("error"),
        }
        for state in sorted(states, key=lambda value: str(value.get("unit")))
    ]
    summary = {
        "schema_version": "1.0.0",
        "record_type": (
            "diagnosis_blind_hcp379_scaleup_pretract_recovery4_summary"
        ),
        "status": (
            "PASS"
            if len(states) == EXPECTED_SCALEUP_N
            and counts.get("PASS_PRETRACT_RECOVERY4")
            == EXPECTED_SCALEUP_N
            else "IN_PROGRESS"
        ),
        "updated_utc": utc_now(),
        "diagnosis_labels_used": False,
        "non_overwriting": True,
        "target_n": EXPECTED_SCALEUP_N,
        "states_present_n": len(states),
        "status_counts": dict(sorted(counts.items())),
        "registration_route_counts": dict(sorted(route_counts.items())),
    }
    atomic_csv(root / "manifests/pretract_recovery4_subjects.csv", rows)
    atomic_json(root / "manifests/pretract_recovery4_summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--pre-human-dry-run",
        action="store_true",
        help="validate and write the exact 216-unit plan without imaging",
    )
    mode.add_argument(
        "--execute",
        action="store_true",
        help="run imaging only after a complete genuine human-QC CSV",
    )
    parser.add_argument(
        "--human-qc-manifest", type=Path, default=DEFAULT_HUMAN_QC
    )
    parser.add_argument("--root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--nthreads", type=int, default=8)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--compact-reproducible-after-pass",
        action="store_true",
        help=(
            "after each subject PASS, run the hash-audited Recovery4 "
            "compactor to retain scientific inputs/QC and reclaim only "
            "classified reproducible working images"
        ),
    )
    args = parser.parse_args()
    execute = bool(args.execute)
    if args.compact_reproducible_after_pass and not execute:
        parser.error(
            "--compact-reproducible-after-pass requires --execute"
        )
    if args.limit is not None and not 1 <= args.limit <= EXPECTED_SCALEUP_N:
        parser.error(f"--limit must be in 1..{EXPECTED_SCALEUP_N}")
    if (
        not 1 <= args.workers <= 8
        or not 1 <= args.nthreads <= 16
        or args.workers * args.nthreads > (os.cpu_count() or 1)
    ):
        parser.error("worker/thread request exceeds the host contract")

    gate = validate_recovery4_gate(
        human_qc=args.human_qc_manifest,
        execute=execute,
    )
    rows, overlap = validate_audit_rows(gate)
    plan = build_plan(
        root=args.root,
        gate=gate,
        rows=rows,
        overlap=overlap,
        execute=execute,
        limit=args.limit,
        compact_reproducible_after_pass=(
            args.compact_reproducible_after_pass
        ),
    )
    if not execute:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0

    install_provisional_engine_guards()
    compactor = (
        load_module(
            "hcp379_pretract_recovery4_streaming_compactor",
            COMPACTOR_SOURCE,
        )
        if args.compact_reproducible_after_pass
        else None
    )
    selected = rows[: args.limit] if args.limit is not None else rows
    print(
        f"[{utc_now()}] RECOVERY4_SCALEUP_START "
        f"n={len(selected)}/{EXPECTED_SCALEUP_N} "
        f"workers={args.workers} threads={args.nthreads}",
        flush=True,
    )
    results: dict[str, dict[str, Any]] = {}
    compaction_results: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                process_unit,
                row,
                root=args.root,
                gate=gate,
                nthreads=args.nthreads,
            ): row["unit"]
            for row in selected
        }
        for index, future in enumerate(as_completed(futures), start=1):
            unit = futures[future]
            try:
                state = future.result()
            except Exception as exc:
                state = {
                    "unit": unit,
                    "status": "FAIL_PRETRACT_RECOVERY4",
                    "error": f"{type(exc).__name__}:{exc}",
                }
            results[unit] = state
            if (
                compactor is not None
                and state.get("status") == "PASS_PRETRACT_RECOVERY4"
            ):
                try:
                    compacted = compactor.compact_unit(
                        root=args.root.resolve(),
                        unit=unit,
                        execute=True,
                    )
                except Exception as exc:
                    compacted = {
                        "unit": unit,
                        "status": "FAIL_COMPACTION_RECOVERY4",
                        "error": f"{type(exc).__name__}:{exc}",
                    }
                compaction_results[unit] = compacted
            print(
                f"[{utc_now()}] recovery4_pretract "
                f"{index}/{len(selected)} {unit} "
                f"{state.get('status')} "
                f"route={state.get('registration_route')} "
                f"compaction={compaction_results.get(unit, {}).get('status')} "
                f"error={state.get('error') or compaction_results.get(unit, {}).get('error')}",
                flush=True,
            )
    summary = write_summary(args.root)
    if compactor is not None:
        compaction_summary = {
            "schema_version": "2.0.0",
            "record_type": (
                "diagnosis_blind_hcp379_pretract_compaction_"
                "recovery4_streaming_summary"
            ),
            "status": (
                "PASS_COMPACTED"
                if len(compaction_results) == len(selected)
                and all(
                    row.get("status") == "PASS_COMPACTED"
                    for row in compaction_results.values()
                )
                else "INCOMPLETE"
            ),
            "updated_utc": utc_now(),
            "diagnosis_labels_used": False,
            "root": str(args.root.resolve()),
            "selected_this_invocation_n": len(selected),
            "compacted_n": sum(
                row.get("status") == "PASS_COMPACTED"
                for row in compaction_results.values()
            ),
            "reclaimed_bytes": sum(
                int(row.get("reclaimed_bytes", 0))
                for row in compaction_results.values()
            ),
            "compactor": file_record(COMPACTOR_SOURCE),
            "results": [
                compaction_results[unit]
                for unit in sorted(compaction_results)
            ],
        }
        atomic_json(
            args.root
            / "manifests/pretract_compaction_recovery4_streaming_summary.json",
            compaction_summary,
        )
    invocation_pass = all(
        state.get("status") == "PASS_PRETRACT_RECOVERY4"
        for state in results.values()
    ) and (
        compactor is None
        or (
            len(compaction_results) == len(selected)
            and all(
                row.get("status") == "PASS_COMPACTED"
                for row in compaction_results.values()
            )
        )
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if invocation_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
