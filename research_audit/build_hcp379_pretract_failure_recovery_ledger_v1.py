#!/usr/bin/env python3
"""Build an exact diagnosis-blind Recovery4 pre-tract status/recovery ledger.

This is an audit artifact only.  It does not run imaging, rewrite subject
states, relax a QC threshold, or authorize tractography.  Every one of the 515
production-execution units is classified from the current topology plus its
subject state as PASS_COMPACTED, RUNNING, FAILED, or WAITING.  Failed units are
assigned a mechanism and a fail-closed recovery requirement.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


EXP = Path("/home/ec2-user/exp")
TOPOLOGY = (
    EXP
    / "research_audit/outputs/hcp379_balanced_release_topology_v1/"
    "topology.json"
)
OUTPUT = (
    EXP
    / "research_audit/outputs/hcp379_pretract_failure_recovery_ledger_v1"
)
EXPECTED_N = 515
EDDY_INTEGRITY_ROWS = (
    Path("/data/derivatives/hcp379_v2/audits/")
    / "eddy_volume_integrity_v1/rows.csv"
)
EDDY_INTEGRITY_SUMMARY = EDDY_INTEGRITY_ROWS.with_name("summary.json")
DICOM_RECONVERTED_EDDY_UNIT = "005_S_0610_I906100"
DICOM_RECONVERTED_EDDY_VALIDATION = (
    EXP
    / "research_audit/outputs/hcp379_eddy_extension_005_S_0610_v1/"
    "validation.json"
)
LOCKED_RAW_EDDY_UNIT = "005_S_6084_I915209"
LOCKED_RAW_EDDY_VALIDATION = (
    EXP
    / "research_audit/outputs/hcp379_eddy_extension_005_S_6084_v1/"
    "validation.json"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "unit",
        "lane",
        "status",
        "subject_state_status",
        "failure_class",
        "recovery_requirement",
        "state_path",
        "compaction_path",
        "error",
    ]
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def classify_failure(unit: str, error: str) -> tuple[str, str]:
    if unit == DICOM_RECONVERTED_EDDY_UNIT:
        if not DICOM_RECONVERTED_EDDY_VALIDATION.is_file():
            raise ValueError(
                "fresh-DICOM Eddy extension validation is missing for "
                f"{unit}"
            )
        validation = load_json(DICOM_RECONVERTED_EDDY_VALIDATION)
        if validation.get("status") != "PASS":
            raise ValueError(
                "fresh-DICOM Eddy extension validation does not pass for "
                f"{unit}"
            )
        return (
            "DICOM_GRADIENT_RECONVERSION_EDDY_PENDING",
            (
                "Require the validated diagnosis-blind Eddy extension from "
                "the freshly reconverted DICOM source, then replay unchanged "
                "Recovery4 image, tensor, FOD and spatial QC."
            ),
        )
    if unit == LOCKED_RAW_EDDY_UNIT:
        if not LOCKED_RAW_EDDY_VALIDATION.is_file():
            raise ValueError(
                "locked-raw Eddy extension validation is missing for "
                f"{unit}"
            )
        validation = load_json(LOCKED_RAW_EDDY_VALIDATION)
        if validation.get("status") != "PASS":
            raise ValueError(
                "locked-raw Eddy extension validation does not pass for "
                f"{unit}"
            )
        return (
            "LOCKED_RAW_VALID_HISTORICAL_EDDY_TENSOR_CORRUPTION",
            (
                "Require the validated diagnosis-blind GPU Eddy extension "
                "from the tensor-valid locked raw DWI, then replay unchanged "
                "Recovery4 image, tensor, FOD and spatial QC."
            ),
        )
    lowered = error.lower()
    if (
        "invalid mrstats range" in lowered
        and "wmfod_norm.mif" in lowered
    ):
        return (
            "PROVISIONAL_ENGINE_FOD_RANGE_GUARD_MISSING",
            (
                "Require subject-scoped replay of the unchanged provisional "
                "engine after installing the already validated aggregate "
                "multi-volume FOD range guard; then apply unchanged "
                "Recovery4 scalar/FOD and spatial QC."
            ),
        )
    # Hard-negative WM-FOD l=0 coefficients are a model/input failure, not
    # numerical round-off.  Test this before the softer substring below:
    # current engine errors also report
    # ``wmfod_l0_below_tolerance_fraction``, which otherwise causes a hard
    # failure to be misclassified as a soft-negative case.
    if "wmfod_l0_below_hard_tolerance" in lowered:
        return (
            "FOD_L0_HARD_NEGATIVE_BURDEN",
            (
                "Require a diagnosis-blind subject-scoped FOD reconstruction "
                "and response/model audit under the unchanged cohort policy; "
                "do not relax the hard-negative tolerance, reuse the current "
                "FOD, or tune a threshold per subject."
            ),
        )
    if "wmfod_l0_below_tolerance" in lowered:
        return (
            "FOD_L0_SOFT_NEGATIVE_FRACTION",
            (
                "Require a cohort-uniform magnitude-plus-fraction policy "
                "validated independently; do not relabel or tune per subject."
            ),
        )
    if "mtnormalise" in lowered:
        return (
            "MTNORMALISE_NONPOSITIVE_TISSUE_BALANCE",
            (
                "Require an isolated normalization recovery with preserved "
                "inputs and equivalent downstream WM-FOD/scalar QC."
            ),
        )
    if (
        "neither spatial route passed direct gates" in lowered
        and "affine=[" in lowered
    ):
        return (
            "SPATIAL_BBR_RIGID_AFFINE_FAIL",
            (
                "Require a diagnosis-blind source-geometry audit and a "
                "separately versioned nonlinear spatial failover, evaluated "
                "against the unchanged tissue, atlas and visual-QC gates; do "
                "not relax overlap thresholds or tune registration per "
                "subject."
            ),
        )
    if "neither spatial route passed direct gates" in lowered:
        return (
            "SPATIAL_BBR_AND_RIGID_FAIL",
            (
                "Require a separately versioned affine failover evaluated "
                "against the unchanged tissue, atlas and visual-QC gates."
            ),
        )
    return (
        "OTHER_TERMINAL_FAILURE",
        "Require mechanism-specific non-overwriting diagnosis-blind recovery.",
    )


def eddy_integrity_holds() -> dict[str, dict[str, str]]:
    if not EDDY_INTEGRITY_ROWS.is_file():
        return {}
    with EDDY_INTEGRITY_ROWS.open(
        newline="", encoding="utf-8-sig"
    ) as handle:
        rows = list(csv.DictReader(handle))
    return {
        str(row["unit"]): {str(k): str(v or "") for k, v in row.items()}
        for row in rows
        if row.get("status") == "FAIL_ZERO_EDDY_VOLUMES"
        and int(row.get("eddy_zero_volume_n", "0")) > 0
    }


def build() -> dict[str, Any]:
    topology = load_json(TOPOLOGY)
    production = topology.get("production")
    if (
        topology.get("production_execution_n") != EXPECTED_N
        or not isinstance(production, list)
        or len(production) != EXPECTED_N
    ):
        raise ValueError("exact 515-unit production topology required")

    zero_eddy = eddy_integrity_holds()
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in production:
        unit = str(item.get("unit", ""))
        lane = str(item.get("lane", ""))
        pretract = item.get("pretract")
        if not unit or unit in seen or not isinstance(pretract, dict):
            raise ValueError(f"invalid or duplicate production row: {unit}")
        seen.add(unit)
        state_path = Path(str(pretract.get("state_path", "")))
        compaction_path = Path(str(pretract.get("compaction_path", "")))
        state: dict[str, Any] = {}
        if state_path.is_file():
            state = load_json(state_path)
            if state.get("unit") != unit:
                raise ValueError(f"state identity differs: {unit}")
        state_status = str(state.get("status", "MISSING"))
        error = str(state.get("error") or "")
        compaction: dict[str, Any] = {}
        if compaction_path.is_file():
            compaction = load_json(compaction_path)
            if compaction.get("unit") != unit:
                raise ValueError(f"compaction identity differs: {unit}")
        compaction_status = str(compaction.get("status", "MISSING"))

        # Input-integrity holds outrank transient or terminal worker state.
        # A worker may have selected a unit from an older queue snapshot; that
        # must never convert an independently established Eddy hold into
        # RUNNING or PASS_COMPACTED release evidence.
        if pretract.get("eddy_integrity_hold") is True:
            status = "FAILED"
            eddy_row = zero_eddy.get(unit, {})
            if int(eddy_row.get("raw_zero_volume_n", "0") or 0) == 0:
                failure_class = "LEGACY_EDDY_ZERO_VOLUMES_RAW_INTACT"
                recovery = (
                    "Require regeneration of only this diagnosis-blind Eddy "
                    "derivative from its intact pre-Eddy source, then replay "
                    "unchanged Recovery4 image and scalar QC."
                )
            else:
                failure_class = "RAW_ZERO_B0_VOLUME_RECOVERABLE"
                recovery = (
                    "Require removal of only the exact zero-valued raw b0 "
                    "volume under the predeclared minimum-b0/diffusion gate, "
                    "then regenerate Eddy and replay unchanged Recovery4 QC."
                )
            error = "INPUT_QC_HOLD:FAIL_ZERO_EDDY_VOLUMES"
        # The topology is an immutable routing snapshot.  Recovery workers can
        # legitimately advance a subject after that snapshot, so release
        # readiness must bind the current state and current compaction record
        # rather than the stale embedded status fields.
        elif (
            state_status == "PASS_PRETRACT_RECOVERY4"
            and compaction_status == "PASS_COMPACTED"
            and state_path.is_file()
            and compaction_path.is_file()
        ):
            status = "PASS_COMPACTED"
            failure_class = ""
            recovery = ""
        elif state_status == "RUNNING":
            status = "RUNNING"
            failure_class = ""
            recovery = ""
        elif state_status == "FAIL_PRETRACT_RECOVERY4":
            status = "FAILED"
            if unit in zero_eddy:
                failure_class = "LEGACY_EDDY_ZERO_VOLUMES_RAW_INTACT"
                recovery = (
                    "Require regeneration of only this diagnosis-blind Eddy "
                    "derivative "
                    "from its intact pre-Eddy source, then replay unchanged "
                    "Recovery4 image and scalar QC."
                )
            else:
                failure_class, recovery = classify_failure(unit, error)
        else:
            status = "WAITING"
            failure_class = ""
            recovery = ""

        rows.append(
            {
                "unit": unit,
                "lane": lane,
                "status": status,
                "subject_state_status": state_status,
                "failure_class": failure_class,
                "recovery_requirement": recovery,
                "state_path": str(state_path),
                "compaction_path": str(compaction_path),
                "error": error,
            }
        )

    rows.sort(key=lambda row: row["unit"])
    statuses = Counter(row["status"] for row in rows)
    failures = Counter(
        row["failure_class"] for row in rows if row["status"] == "FAILED"
    )
    lanes = {
        lane: dict(
            sorted(
                Counter(
                    row["status"] for row in rows if row["lane"] == lane
                ).items()
            )
        )
        for lane in sorted({row["lane"] for row in rows})
    }
    payload = {
        "schema_version": "1.0.0",
        "record_type": "hcp379_pretract_failure_recovery_ledger",
        "status": (
            "PASS_ALL_PRETRACT_READY"
            if statuses.get("PASS_COMPACTED") == EXPECTED_N
            else "RECOVERY_AND_PROCESSING_IN_PROGRESS"
        ),
        "generated_utc": utc_now(),
        "diagnosis_labels_used": False,
        "outcomes_used": False,
        "imaging_executed": False,
        "qc_thresholds_changed": False,
        "non_overwriting": True,
        "expected_unit_n": EXPECTED_N,
        "observed_unit_n": len(rows),
        "status_counts": dict(sorted(statuses.items())),
        "failure_class_counts": dict(sorted(failures.items())),
        "lane_status_counts": lanes,
        "failed_units": [
            {
                key: row[key]
                for key in (
                    "unit",
                    "lane",
                    "failure_class",
                    "recovery_requirement",
                    "state_path",
                    "error",
                )
            }
            for row in rows
            if row["status"] == "FAILED"
        ],
        "records": {
            "topology": file_record(TOPOLOGY),
            "builder": file_record(Path(__file__)),
            **(
                {
                    "eddy_integrity_rows": file_record(
                        EDDY_INTEGRITY_ROWS
                    ),
                    "eddy_integrity_summary": file_record(
                        EDDY_INTEGRITY_SUMMARY
                    ),
                }
                if EDDY_INTEGRITY_ROWS.is_file()
                and EDDY_INTEGRITY_SUMMARY.is_file()
                else {}
            ),
            **(
                {
                    "dicom_reconverted_eddy_validation": file_record(
                        DICOM_RECONVERTED_EDDY_VALIDATION
                    )
                }
                if DICOM_RECONVERTED_EDDY_VALIDATION.is_file()
                else {}
            ),
            **(
                {
                    "locked_raw_eddy_validation": file_record(
                        LOCKED_RAW_EDDY_VALIDATION
                    )
                }
                if LOCKED_RAW_EDDY_VALIDATION.is_file()
                else {}
            ),
        },
    }
    atomic_csv(OUTPUT / "ledger.csv", rows)
    payload["records"]["ledger_csv"] = file_record(OUTPUT / "ledger.csv")
    atomic_json(OUTPUT / "ledger.json", payload)
    return payload


def main() -> int:
    payload = build()
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
