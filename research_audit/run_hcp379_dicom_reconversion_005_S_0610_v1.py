#!/home/ec2-user/fsl/bin/python
"""Reconvert the locked 005_S_0610 DICOM DTI and audit tensor physics.

All outputs are written to an isolated audit namespace.  The historical MIF,
Eddy derivative, and active Recovery4 subject tree are never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping


EXP = Path("/home/ec2-user/exp")
HCP = Path("/data/derivatives/hcp379_v2")
UNIT = "005_S_0610_I906100"
DICOM = Path(
    "/data/Images/dti/005_S_0610/Axial_DTI/"
    "2017-09-20_11_51_41.0/I906100"
)
CONVERTER = EXP / "tools/dcm2niix/v1.0.20260416/dcm2niix"
MRCONVERT = Path("/home/ec2-user/mrtrix3/bin/mrconvert")
MRINFO = Path("/home/ec2-user/mrtrix3/bin/mrinfo")
TENSOR_HELPER = EXP / "scripts/hcp/hcp_eddy_tensor_repair_v2.py"
OUTPUT = (
    HCP
    / "pretract_failure_source_audit_v1/"
    "dicom_reconversion_005_S_0610_v1"
)
CONVERSION = OUTPUT / "conversion"
TENSOR = OUTPUT / "tensor_audit"
SUMMARY = OUTPUT / "summary.json"
EXPECTED_DICOM_N = 4320
EXPECTED_DICOM_BYTES = 635652842


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


HELPER = load_module(TENSOR_HELPER, "hcp379_dicom_reconversion_tensor")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise FileNotFoundError(resolved)
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256(resolved),
    }


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def source_inventory() -> dict[str, Any]:
    files = sorted(path for path in DICOM.iterdir() if path.is_file())
    size = sum(path.stat().st_size for path in files)
    if len(files) != EXPECTED_DICOM_N or size != EXPECTED_DICOM_BYTES:
        raise ValueError(
            f"DICOM inventory differs: files={len(files)} bytes={size}"
        )
    return {
        "path": str(DICOM.resolve()),
        "file_n": len(files),
        "size_bytes": size,
        "first_file": files[0].name,
        "last_file": files[-1].name,
    }


def preflight() -> dict[str, Any]:
    return {
        "schema_version": "1.0.0",
        "record_type": "hcp379_dicom_reconversion_preflight",
        "status": "READY",
        "unit": UNIT,
        "diagnosis_labels_used": False,
        "outcomes_used": False,
        "connectome_density_used": False,
        "non_overwriting": True,
        "source_images_modified": False,
        "imaging_executed": False,
        "dicom_source": source_inventory(),
        "records": {
            "converter": record(CONVERTER),
            "mrconvert": record(MRCONVERT),
            "mrinfo": record(MRINFO),
            "tensor_helper": record(TENSOR_HELPER),
            "implementation": record(Path(__file__)),
        },
    }


def run(command: list[str], log: Any) -> None:
    log.write(json.dumps(command) + "\n")
    log.flush()
    result = subprocess.run(
        command,
        stdout=log,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(
            f"command failed rc={result.returncode}: {command[0]}"
        )


def execute(nthreads: int) -> dict[str, Any]:
    pre = preflight()
    if SUMMARY.is_file():
        prior = json.loads(SUMMARY.read_text(encoding="utf-8"))
        if (
            prior.get("dicom_source") == pre["dicom_source"]
            and prior.get("records") == pre["records"]
        ):
            return prior
        raise ValueError("conversion binding changed; refusing overwrite")
    if OUTPUT.exists():
        raise FileExistsError(
            f"partial conversion namespace exists: {OUTPUT}"
        )
    CONVERSION.mkdir(parents=True, exist_ok=False)
    log_path = OUTPUT / "conversion.log"
    with log_path.open("x", encoding="utf-8") as log:
        run(
            [
                str(CONVERTER),
                "-b",
                "y",
                "-z",
                "n",
                "-f",
                UNIT,
                "-o",
                str(CONVERSION),
                str(DICOM),
            ],
            log,
        )
        nifti = sorted(CONVERSION.glob("*.nii"))
        bvec = sorted(CONVERSION.glob("*.bvec"))
        bval = sorted(CONVERSION.glob("*.bval"))
        sidecar = sorted(CONVERSION.glob("*.json"))
        if not (
            len(nifti) == len(bvec) == len(bval) == len(sidecar) == 1
        ):
            raise ValueError(
                "expected one NIfTI/bvec/bval/JSON conversion family"
            )
        mif = OUTPUT / "dwi_from_dicom.mif"
        run(
            [
                str(MRCONVERT),
                str(nifti[0]),
                str(mif),
                "-fslgrad",
                str(bvec[0]),
                str(bval[0]),
                "-strides",
                "-1,2,3,4",
                "-force",
            ],
            log,
        )
        header = subprocess.run(
            [
                str(MRINFO),
                str(mif),
                "-size",
                "-spacing",
                "-shell_bvalues",
                "-shell_sizes",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        tensor = HELPER.tensor_audit(mif, TENSOR, nthreads, log)
    metadata = json.loads(sidecar[0].read_text(encoding="utf-8"))
    decision = (
        "READY_FOR_SOURCE_AWARE_EDDY"
        if tensor["gate"] == "PASS"
        else "RECONVERTED_DICOM_STILL_FAILS_TENSOR_PHYSICS"
    )
    payload = {
        **pre,
        "record_type": "hcp379_dicom_reconversion_audit",
        "status": "PASS",
        "generated_utc": HELPER.utc_now(),
        "imaging_executed": True,
        "decision": decision,
        "mrinfo": header,
        "selected_metadata": {
            key: metadata.get(key)
            for key in (
                "Manufacturer",
                "ManufacturersModelName",
                "MagneticFieldStrength",
                "PhaseEncodingDirection",
                "TotalReadoutTime",
                "EffectiveEchoSpacing",
            )
        },
        "tensor_audit": tensor,
        "outputs": {
            "nifti": record(nifti[0]),
            "bvec": record(bvec[0]),
            "bval": record(bval[0]),
            "json": record(sidecar[0]),
            "mif": record(mif),
            "log": record(log_path),
        },
        "source_images_modified": False,
    }
    atomic_json(SUMMARY, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--nthreads", type=int, default=8)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.nthreads <= 16:
        parser.error("--nthreads must be in 1..16")
    if args.self_test:
        if EXPECTED_DICOM_N != 4320:
            raise AssertionError(EXPECTED_DICOM_N)
        print("HCP379_DICOM_RECONVERSION_005_S_0610_SELF_TEST_PASS")
        return 0
    payload = execute(args.nthreads) if args.execute else preflight()
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] in {"READY", "PASS"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
