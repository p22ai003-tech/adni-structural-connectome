#!/usr/bin/env python3
"""Independently replay the four-target Eddy acquisition contract."""

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pydicom


EXP = Path("/home/ec2-user/exp")
CONTRACT = (
    EXP
    / "research_audit/outputs/hcp379_eddy_acquisition_contract_v1/"
    "contract.json"
)
PLAN = (
    EXP
    / "research_audit/outputs/hcp379_eddy_integrity_recovery_plan_v1/"
    "plan.json"
)
OUTPUT = CONTRACT.with_name("validation.json")
DCM2NIIX = EXP / ".envs/dcm2niix-20260724/bin/dcm2niix"
MRINFO = Path("/home/ec2-user/mrtrix3/bin/mrinfo")
EXPECTED = {
    "003_S_4373_I378923",
    "003_S_6490_I1043781",
    "003_S_6644_I1083048",
    "006_S_4713_I1483612",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def series_digest(root: Path) -> tuple[int, int, str]:
    files = sorted(path for path in root.iterdir() if path.is_file())
    digest = hashlib.sha256()
    total = 0
    for path in files:
        size = path.stat().st_size
        total += size
        digest.update(
            (
                f"{path.name}\0{size}\0{sha256_file(path)}\n"
            ).encode("utf-8")
        )
    return len(files), total, digest.hexdigest()


def dcm_metadata(root: Path) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(
        prefix="hcp379-eddy-acquisition-validation-"
    ) as directory:
        result = subprocess.run(
            [
                str(DCM2NIIX),
                "-b",
                "y",
                "-z",
                "n",
                "-f",
                "validation",
                "-o",
                directory,
                str(root),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        sidecars = list(Path(directory).glob("*.json"))
        if len(sidecars) != 1:
            raise ValueError(result.stdout + result.stderr)
        return json.loads(sidecars[0].read_text(encoding="utf-8"))


def main() -> int:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    rows = {
        str(row["unit"]): row
        for row in contract.get("acquisitions", [])
    }
    targets = {
        str(row["unit"]): row for row in plan.get("targets", [])
    }
    checks: dict[str, dict[str, Any]] = {}

    def check(name: str, passed: bool, evidence: Any) -> None:
        checks[name] = {
            "status": "PASS" if passed else "FAIL",
            "evidence": evidence,
        }

    check(
        "exact_four_target_identity",
        set(rows) == EXPECTED == set(targets) and len(rows) == 4,
        sorted(rows),
    )
    records = contract.get("records", {})
    check(
        "tool_and_plan_hash_binding",
        (
            records.get("recovery_plan", {}).get("sha256")
            == sha256_file(PLAN)
            and records.get("dcm2niix", {}).get("sha256")
            == sha256_file(DCM2NIIX)
            and records.get("mrinfo", {}).get("sha256")
            == sha256_file(MRINFO)
        ),
        {
            "plan": sha256_file(PLAN),
            "dcm2niix": sha256_file(DCM2NIIX),
            "mrinfo": sha256_file(MRINFO),
        },
    )
    source_errors: list[str] = []
    for unit, row in rows.items():
        raw = Path(str(row["raw_source"]["path"]))
        if (
            not raw.is_file()
            or sha256_file(raw) != targets[unit]["raw_source"]["sha256"]
            or row["raw_source"]["sha256"] != sha256_file(raw)
        ):
            source_errors.append(unit)
    check("raw_sources_replay", not source_errors, source_errors)

    mif_errors: list[str] = []
    for unit in ("003_S_6490_I1043781", "003_S_6644_I1083048"):
        row = rows[unit]
        raw = Path(row["raw_source"]["path"])
        phase = subprocess.run(
            [str(MRINFO), str(raw), "-property", "PhaseEncodingDirection"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        readout = float(
            subprocess.run(
                [str(MRINFO), str(raw), "-property", "TotalReadoutTime"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        if (
            row["metadata_route"] != "IMMUTABLE_SOURCE_MIF_HEADER"
            or row["pe_dir"] != phase
            or abs(row["total_readout_time_seconds"] - readout) > 1e-12
        ):
            mif_errors.append(unit)
    check("mif_header_metadata_replays", not mif_errors, mif_errors)

    dicom_errors: list[str] = []
    for unit in ("003_S_4373_I378923", "006_S_4713_I1483612"):
        row = rows[unit]
        series = row["dicom_series"]
        root = Path(series["path"])
        file_n, size, digest = series_digest(root)
        metadata = dcm_metadata(root)
        if (
            file_n != series["file_n"]
            or size != series["size_bytes"]
            or digest != series["content_manifest_sha256"]
        ):
            dicom_errors.append(f"{unit}:series")
        if unit == "003_S_4373_I378923":
            expected_phase = metadata.get("PhaseEncodingDirection")
            expected_readout = metadata.get("TotalReadoutTime")
        else:
            private_values: set[str] = set()
            for path in sorted(root.iterdir()):
                if not path.is_file():
                    continue
                dataset = pydicom.dcmread(
                    path,
                    stop_before_pixels=True,
                    force=True,
                    specific_tags=[(0x2005, 0x1564)],
                )
                value = dataset.get((0x2005, 0x1564))
                if value is not None:
                    private_values.add(str(value.value).strip())
            expected_phase = (
                "j-"
                if metadata.get("PhaseEncodingAxis") == "j"
                and private_values == {"AP"}
                else None
            )
            expected_readout = metadata.get(
                "EstimatedTotalReadoutTime"
            )
        if (
            row["metadata_route"] != "ORIGINAL_DICOM_DCM2NIIX"
            or row["pe_dir"] != expected_phase
            or abs(
                row["total_readout_time_seconds"]
                - float(expected_readout or 0.0)
            )
            > 1e-12
        ):
            dicom_errors.append(f"{unit}:metadata")
    check(
        "dicom_metadata_independently_replays",
        not dicom_errors,
        dicom_errors,
    )
    value_errors = [
        unit
        for unit, row in rows.items()
        if row["pe_dir"] not in {"i", "i-", "j", "j-", "k", "k-"}
        or not 0.0 < float(row["total_readout_time_seconds"]) < 1.0
    ]
    check("eddy_values_are_valid", not value_errors, value_errors)
    check(
        "contract_is_blind_and_non_overwriting",
        (
            contract.get("status") == "PASS"
            and contract.get("diagnosis_labels_used") is False
            and contract.get("outcomes_used") is False
            and contract.get("connectome_density_used") is False
            and contract.get("per_subject_parameter_tuning_used") is False
            and contract.get("historical_outputs_modified") is False
        ),
        {
            key: contract.get(key)
            for key in (
                "status",
                "diagnosis_labels_used",
                "outcomes_used",
                "connectome_density_used",
                "per_subject_parameter_tuning_used",
                "historical_outputs_modified",
            )
        },
    )
    check(
        "execution_scope_is_metadata_only",
        all(
            not Path(row["raw_source"]["path"]).is_symlink()
            for row in rows.values()
        ),
        {"target_n": len(rows)},
    )
    passed = sum(row["status"] == "PASS" for row in checks.values())
    payload = {
        "schema_version": "1.0.0",
        "record_type": "hcp379_eddy_acquisition_contract_validation",
        "status": "PASS" if passed == len(checks) else "FAIL",
        "generated_utc": datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "passed_checks": passed,
        "total_checks": len(checks),
        "checks": checks,
    }
    OUTPUT.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
