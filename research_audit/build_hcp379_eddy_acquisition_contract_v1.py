#!/usr/bin/env python3
"""Build acquisition metadata for the four diagnosis-blind Eddy repairs.

Two immutable source MIFs carry usable phase/readout metadata.  For the two
older sources that do not, the exact original DICOM series are converted in a
temporary directory with a hash-bound dcm2niix binary.  Only acquisition
fields needed by ``dwifslpreproc`` are retained; converted images are deleted
with the temporary directory and no historical derivative is modified.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import pydicom


EXP = Path("/home/ec2-user/exp")
PLAN = (
    EXP
    / "research_audit/outputs/hcp379_eddy_integrity_recovery_plan_v1/"
    "plan.json"
)
OUTPUT = (
    EXP
    / "research_audit/outputs/hcp379_eddy_acquisition_contract_v1/"
    "contract.json"
)
DCM2NIIX = (
    EXP / ".envs/dcm2niix-20260724/bin/dcm2niix"
)
MRINFO = Path("/home/ec2-user/mrtrix3/bin/mrinfo")
EXPECTED_UNITS = (
    "003_S_4373_I378923",
    "003_S_6490_I1043781",
    "003_S_6644_I1083048",
    "006_S_4713_I1483612",
)
DICOM_ROOTS = {
    "003_S_4373_I378923": Path(
        "/data/Images/dti/003_S_4373/Axial_DTI/"
        "2012-07-17_12_14_59.0/I378923"
    ),
    "006_S_4713_I1483612": Path(
        "/data/Images/dti/006_S_4713/Axial_DTI/"
        "2021-08-19_13_33_57.0/I1483612"
    ),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def series_record(root: Path) -> dict[str, Any]:
    files = sorted(path for path in root.iterdir() if path.is_file())
    if not files:
        raise ValueError(f"empty DICOM series: {root}")
    digest = hashlib.sha256()
    total = 0
    for path in files:
        content_sha = sha256_file(path)
        size = path.stat().st_size
        total += size
        digest.update(
            f"{path.name}\0{size}\0{content_sha}\n".encode("utf-8")
        )
    return {
        "path": str(root.resolve()),
        "file_n": len(files),
        "size_bytes": total,
        "content_manifest_sha256": digest.hexdigest(),
    }


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


def mif_property(path: Path, name: str) -> str:
    result = subprocess.run(
        [str(MRINFO), str(path), "-property", name],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def dcm2niix_metadata(root: Path) -> tuple[dict[str, Any], str]:
    with tempfile.TemporaryDirectory(
        prefix="hcp379-eddy-acquisition-"
    ) as directory:
        output = Path(directory)
        result = subprocess.run(
            [
                str(DCM2NIIX),
                "-b",
                "y",
                "-z",
                "n",
                "-f",
                "acquisition",
                "-o",
                str(output),
                str(root),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        sidecars = list(output.glob("*.json"))
        if len(sidecars) != 1:
            raise ValueError(
                f"expected one dcm2niix sidecar for {root}, got {sidecars}"
            )
        metadata = json.loads(sidecars[0].read_text(encoding="utf-8"))
        return metadata, result.stdout + result.stderr


def private_phase_direction(root: Path) -> dict[str, Any]:
    values: set[str] = set()
    dicom_direction: set[str] = set()
    for path in sorted(root.iterdir()):
        if not path.is_file():
            continue
        dataset = pydicom.dcmread(
            path,
            stop_before_pixels=True,
            force=True,
            specific_tags=[(0x0018, 0x1312), (0x2005, 0x1564)],
        )
        standard = dataset.get((0x0018, 0x1312))
        private = dataset.get((0x2005, 0x1564))
        if standard is not None:
            dicom_direction.add(str(standard.value).strip())
        if private is not None:
            values.add(str(private.value).strip())
    return {
        "in_plane_phase_encoding_direction_dicom": sorted(dicom_direction),
        "philips_private_phase_direction": sorted(values),
    }


def build() -> dict[str, Any]:
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    targets = plan.get("targets")
    if (
        plan.get("status") != "PASS"
        or plan.get("target_n") != 4
        or not isinstance(targets, list)
        or tuple(sorted(str(row["unit"]) for row in targets))
        != EXPECTED_UNITS
    ):
        raise ValueError("exact PASS four-target recovery plan required")
    if not DCM2NIIX.is_file() or not MRINFO.is_file():
        raise FileNotFoundError("metadata tools are unavailable")

    acquisitions: list[dict[str, Any]] = []
    for target in sorted(targets, key=lambda row: str(row["unit"])):
        unit = str(target["unit"])
        raw = Path(str(target["raw_source"]["path"])).resolve()
        if (
            not raw.is_file()
            or sha256_file(raw) != target["raw_source"]["sha256"]
        ):
            raise ValueError(f"{unit}: raw source binding differs")
        if unit not in DICOM_ROOTS:
            phase = mif_property(raw, "PhaseEncodingDirection")
            readout_text = mif_property(raw, "TotalReadoutTime")
            readout = float(readout_text)
            if phase not in {"i", "i-", "j", "j-", "k", "k-"}:
                raise ValueError(f"{unit}: invalid MIF phase direction")
            if not 0.0 < readout < 1.0:
                raise ValueError(f"{unit}: invalid MIF readout")
            acquisitions.append(
                {
                    "unit": unit,
                    "pe_dir": phase,
                    "total_readout_time_seconds": readout,
                    "metadata_route": "IMMUTABLE_SOURCE_MIF_HEADER",
                    "phase_direction_confidence": "DIRECT",
                    "readout_confidence": "DIRECT",
                    "raw_source": file_record(raw),
                }
            )
            continue

        dicom_root = DICOM_ROOTS[unit]
        metadata, conversion_log = dcm2niix_metadata(dicom_root)
        series = series_record(dicom_root)
        selected: dict[str, Any] = {
            key: metadata.get(key)
            for key in (
                "Manufacturer",
                "PhaseEncodingDirection",
                "PhaseEncodingAxis",
                "TotalReadoutTime",
                "EstimatedTotalReadoutTime",
                "EffectiveEchoSpacing",
                "EstimatedEffectiveEchoSpacing",
                "WaterFatShift",
                "EchoTrainLength",
                "AcquisitionMatrixPE",
                "ReconMatrixPE",
                "InPlanePhaseEncodingDirectionDICOM",
            )
            if metadata.get(key) is not None
        }
        if unit == "003_S_4373_I378923":
            phase = str(metadata.get("PhaseEncodingDirection", ""))
            readout = float(metadata.get("TotalReadoutTime", 0.0))
            phase_confidence = "DIRECT_DCM2NIIX"
            readout_confidence = "DIRECT_DCM2NIIX"
            private = {}
        else:
            private = private_phase_direction(dicom_root)
            axis = str(metadata.get("PhaseEncodingAxis", ""))
            private_values = private[
                "philips_private_phase_direction"
            ]
            if axis != "j" or private_values != ["AP"]:
                raise ValueError(
                    f"{unit}: Philips phase-axis/sign evidence differs"
                )
            phase = "j-"
            readout = float(
                metadata.get("EstimatedTotalReadoutTime", 0.0)
            )
            phase_confidence = (
                "DICOM_AXIS_PLUS_PHILIPS_AP_SIGN_MAPPING"
            )
            readout_confidence = "DCM2NIIX_VENDOR_ESTIMATE"
        if phase not in {"i", "i-", "j", "j-", "k", "k-"}:
            raise ValueError(f"{unit}: missing DICOM phase direction")
        if not 0.0 < readout < 1.0:
            raise ValueError(f"{unit}: missing DICOM readout")
        acquisitions.append(
            {
                "unit": unit,
                "pe_dir": phase,
                "total_readout_time_seconds": readout,
                "metadata_route": "ORIGINAL_DICOM_DCM2NIIX",
                "phase_direction_confidence": phase_confidence,
                "readout_confidence": readout_confidence,
                "raw_source": file_record(raw),
                "dicom_series": series,
                "selected_dcm2niix_metadata": selected,
                "private_phase_evidence": private,
                "conversion_log_sha256": hashlib.sha256(
                    conversion_log.encode("utf-8")
                ).hexdigest(),
            }
        )

    payload = {
        "schema_version": "1.0.0",
        "record_type": "hcp379_eddy_acquisition_contract",
        "status": "PASS",
        "generated_utc": utc_now(),
        "target_n": len(acquisitions),
        "diagnosis_labels_used": False,
        "outcomes_used": False,
        "connectome_density_used": False,
        "per_subject_parameter_tuning_used": False,
        "historical_outputs_modified": False,
        "philips_ap_mapping": (
            "For a LAS-oriented NIfTI with phase axis j, DICOM AP is "
            "encoded as FSL/MRtrix j-."
        ),
        "acquisitions": acquisitions,
        "records": {
            "recovery_plan": file_record(PLAN),
            "dcm2niix": file_record(DCM2NIIX),
            "mrinfo": file_record(MRINFO),
            "builder": file_record(Path(__file__)),
        },
    }
    atomic_json(OUTPUT, payload)
    return payload


def main() -> int:
    payload = build()
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
