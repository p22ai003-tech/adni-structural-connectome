#!/usr/bin/env python3
"""Build corrected pre-tractography HCP379 inputs for the 216 non-canary cases.

The route reuses only the audited Eddy derivative (repairing its gradient
header for 28 subjects), then rebuilds bias correction, mask, exact-T1 5TT,
BBR, HCP379 labels, pooled-response SS3T-CSD and tensor maps.  It never reads
diagnosis and never reuses historical ``nodes_b0``.

Execution is gated on the 15-subject Recovery3/atlas canary and its real human
visual-QC manifest.  The 11 canary subjects that belong to the 227-subject
corrected-track queue are excluded because their corrected pretract products
already exist in the validated canary run.
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
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml


EXP = Path("/home/ec2-user/exp")
DERIV = Path("/data/derivatives")
HCP_ROOT = DERIV / "hcp379_v2"
OUTPUT_ROOT = HCP_ROOT / "corrected_scaleup"
CANARY_RUN_ROOT = (
    DERIV / "scforge_v2/h04a_r1_recovery_20260719_retry4"
)
AUDIT_CSV = (
    EXP
    / "research_audit/outputs/hcp379_scaleup_input_audit_v2/"
    "scaleup_input_audit.csv"
)
AUDIT_SUMMARY = AUDIT_CSV.with_name("scaleup_input_audit_summary.json")
RECOVERY3_BINDING = (
    EXP
    / "research_audit/outputs/"
    "h04a_r1_retry4_pretract_recovery3_package_v1/"
    "recovery3_execution_binding.json"
)
RECOVERY3_COMPLETION = (
    CANARY_RUN_ROOT
    / "publication/pre_tractography_canary_recovery3_completion.json"
)
ATLAS_SUMMARY = (
    HCP_ROOT
    / "corrected_atlas_canary/corrected_atlas_canary_summary.json"
)
REVIEW_MANIFEST = (
    HCP_ROOT / "review/hcp379_visual_review_pack_manifest.json"
)
DEFAULT_HUMAN_QC = HCP_ROOT / "review/human_visual_qc.csv"
CONFIG = EXP / "configs/connectome_v2_retry3.yaml"
ENVIRONMENT = EXP / "scforge/workflow/environment_contract_retry3.yaml"
ATLAS_QC_SOURCE = (
    EXP / "scripts/hcp/hcp379_corrected_atlas_canary_v2.py"
)
RELABEL = EXP / "scripts/hcp/relabel_hcp.py"
RELABEL_LUT = (
    DERIV / "parc_hcpmmp1/hcpmmp1_subcort_relabel.txt"
)
MRTRIX = Path("/home/ec2-user/mrtrix3/bin")
FSL = Path("/home/ec2-user/fsl/bin")
FREESURFER = Path("/home/ec2-user/freesurfer")
ANTS = EXP / ".envs/ants-2.6.5/bin"
SS3T_PYTHON = Path("/usr/bin/python3.9")
SS3T_SCRIPT = Path("/home/ec2-user/MRtrix3Tissue/bin/ss3t_csd_beta1")
FSL_PYTHON = FSL / "python"
MINIMUM_MASK_VOLUME_MM3 = 750_000.0
MAXIMUM_MASK_VOLUME_MM3 = 2_500_000.0
EXPECTED_TARGET_N = 227
EXPECTED_SCALEUP_N = 216


def load_atlas_qc_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "hcp379_atlas_qc_v2", ATLAS_QC_SOURCE
    )
    if spec is None or spec.loader is None:
        raise ImportError(ATLAS_QC_SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ATLAS_QC_MODULE = load_atlas_qc_module()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root is not an object: {path}")
    return value


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


def size_record(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise ValueError(f"not a regular file: {resolved}")
    return {"path": str(resolved), "size_bytes": resolved.stat().st_size}


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
    value = os.environ.copy()
    value.update(
        {
            "FSLDIR": str(FSL.parent),
            "FSLOUTPUTTYPE": "NIFTI_GZ",
            "FREESURFER_HOME": str(FREESURFER),
            "FS_LICENSE": str(FREESURFER / "license.txt"),
            "SUBJECTS_DIR": str(DERIV / "fastsurfer"),
            "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS": str(nthreads),
            "OMP_NUM_THREADS": str(nthreads),
            "MRTRIX_NTHREADS": str(nthreads),
            "LC_ALL": "C",
            "PATH": ":".join(
                (
                    str(FREESURFER / "bin"),
                    str(ANTS),
                    str(FSL),
                    str(MRTRIX),
                    str(SS3T_SCRIPT.parent),
                    value.get("PATH", ""),
                )
            ),
        }
    )
    return value


def run(
    command: list[str],
    *,
    log: Any,
    env: Mapping[str, str],
    outputs: tuple[Path, ...] = (),
) -> None:
    if outputs and all(path.is_file() and path.stat().st_size > 0 for path in outputs):
        return
    for path in outputs:
        if path.is_file() or path.is_symlink():
            path.unlink()
    log.write(f"\n[{utc_now()}] COMMAND {json.dumps(command)}\n")
    log.flush()
    result = subprocess.run(
        command,
        check=False,
        stdout=log,
        stderr=subprocess.STDOUT,
        text=True,
        env=dict(env),
    )
    if result.returncode:
        raise RuntimeError(
            f"command failed rc={result.returncode}: {command[0]}"
        )
    absent = [
        str(path)
        for path in outputs
        if not path.is_file() or path.stat().st_size <= 0
    ]
    if absent:
        raise RuntimeError("command outputs absent: " + ",".join(absent))


def mrstats_range(path: Path, mask: Path | None = None) -> dict[str, float]:
    command = [
        str(MRTRIX / "mrstats"),
        str(path),
        "-output",
        "min",
        "-output",
        "max",
    ]
    if mask is not None:
        command.extend(("-mask", str(mask)))
    result = subprocess.run(
        command, check=True, capture_output=True, text=True
    )
    values = [float(value) for value in result.stdout.split()]
    if len(values) != 2 or not all(math.isfinite(value) for value in values):
        raise ValueError(f"invalid mrstats range for {path}: {values}")
    return {"min": values[0], "max": values[1]}


def mask_physical_qc(path: Path) -> dict[str, float]:
    count = subprocess.run(
        [
            str(MRTRIX / "mrstats"),
            str(path),
            "-output",
            "count",
            "-ignorezero",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    spacing = subprocess.run(
        [str(MRTRIX / "mrinfo"), str(path), "-spacing"],
        check=True,
        capture_output=True,
        text=True,
    )
    nonzero = int(float(count.stdout.strip()))
    values = [float(value) for value in spacing.stdout.split()[:3]]
    if (
        len(values) != 3
        or not all(math.isfinite(value) and value > 0 for value in values)
    ):
        raise ValueError(f"invalid mask spacing: {values}")
    voxel = float(np.prod(values))
    return {
        "nonzero_voxels": nonzero,
        "voxel_volume_mm3": voxel,
        "volume_mm3": nonzero * voxel,
    }


def validate_human_qc(path: Path, units: set[str]) -> dict[str, Any]:
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
        if forbidden.intersection(
            str(field).strip().lower() for field in fields
        ):
            raise ValueError("human-QC manifest is not diagnosis blind")
        if not {"unit", "status", "reviewer", "reviewed_utc"}.issubset(fields):
            raise ValueError("human-QC manifest lacks required fields")
        rows = [
            {str(key): str(value or "").strip() for key, value in row.items()}
            for row in reader
        ]
    if {row["unit"] for row in rows} != units or len(rows) != len(units):
        raise ValueError("human-QC rows differ from the exact canary set")
    for row in rows:
        if (
            row["status"].upper() != "PASS"
            or not row["reviewer"]
            or not row["reviewed_utc"]
        ):
            raise ValueError(f"human QC is incomplete for {row['unit']}")
        timestamp_text = row["reviewed_utc"]
        try:
            reviewed_at = datetime.fromisoformat(
                timestamp_text[:-1] + "+00:00"
                if timestamp_text.endswith("Z")
                else timestamp_text
            )
        except ValueError as exc:
            raise ValueError(
                f"human-QC timestamp is invalid for {row['unit']}: "
                f"{timestamp_text}"
            ) from exc
        if reviewed_at.tzinfo is None:
            raise ValueError(
                f"human-QC timestamp lacks timezone for {row['unit']}"
            )
        if reviewed_at > datetime.now(timezone.utc):
            raise ValueError(
                f"human-QC timestamp is in the future for {row['unit']}"
            )
    return file_record(path)


def validate_global_gate(human_qc: Path) -> dict[str, Any]:
    audit = load_json(AUDIT_SUMMARY)
    binding = load_json(RECOVERY3_BINDING)
    completion = load_json(RECOVERY3_COMPLETION)
    atlas = load_json(ATLAS_SUMMARY)
    review = load_json(REVIEW_MANIFEST)
    units = set(str(unit) for unit in binding.get("units", []))
    if (
        audit.get("status") != "PASS"
        or audit.get("audited_n") != EXPECTED_TARGET_N
        or binding.get("diagnosis_labels_used") is not False
        or len(units) != 15
        or completion.get("status") != "PASS"
        or completion.get("pre_tractography_summary", {}).get(
            "ready_for_human_qc"
        )
        != 15
        or atlas.get("status") != "PASS"
        or atlas.get("passed_unit_count") != 15
        or review.get("status") != "READY_FOR_HUMAN_REVIEW"
        or review.get("human_visual_qc_inferred") is not False
    ):
        raise ValueError("corrected pretract scale-up gate is not PASS")
    human_record = validate_human_qc(human_qc, units)
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    calibration = config["fod"]["response_estimation"]["calibration"]
    responses = {}
    for tissue, record in calibration["pooled_responses"].items():
        path = Path(record["path"]).resolve()
        if (
            not path.is_file()
            or path.stat().st_size != int(record["size_bytes"])
            or sha256_file(path) != record["sha256"]
        ):
            raise ValueError(f"frozen pooled {tissue} response differs")
        responses[tissue] = path
    return {
        "canary_units": sorted(units),
        "human_visual_qc": human_record,
        "scaleup_input_audit_csv": file_record(AUDIT_CSV),
        "scaleup_input_audit": file_record(AUDIT_SUMMARY),
        "recovery3_completion": file_record(RECOVERY3_COMPLETION),
        "corrected_atlas_summary": file_record(ATLAS_SUMMARY),
        "visual_review_pack": file_record(REVIEW_MANIFEST),
        "pooled_responses": {
            tissue: file_record(path) for tissue, path in responses.items()
        },
        "response_paths": responses,
        "lmax": int(config["fod"]["lmax"]),
        "bzero_threshold": float(
            config["dwi_preprocessing"]["registration_reference"][
                "b0_threshold_s_per_mm2"
            ]
        ),
    }


def stage_paths(root: Path, unit: str) -> dict[str, Path]:
    subject = root / "subjects" / unit
    return {
        "subject": subject,
        "state": root / "qc/subjects" / f"{unit}.json",
        "lock": root / "locks" / f"{unit}.lock",
        "log": root / "logs/subjects" / f"{unit}.log",
        "eddy_gradfix": subject / "00_input/eddy_gradfix.mif",
        "dwi": subject / "01_dwi/dwi_biascorr.mif",
        "b0_series": subject / "01_dwi/b0_series.mif",
        "b0_mif": subject / "01_dwi/mean_b0.mif",
        "b0": subject / "01_dwi/mean_b0.nii.gz",
        "mask": subject / "01_dwi/dwi_brain_mask.mif",
        "t1": subject / "02_anat/t1_n4.nii.gz",
        "t1_bias": subject / "02_anat/t1_n4_bias.nii.gz",
        "t1_brain": subject / "02_anat/t1_brain.nii.gz",
        "t1_mask": subject / "02_anat/t1_brain_mask.nii.gz",
        "five_tt_t1": subject / "05_model/5tt_t1_fsl.mif",
        "wmseg": subject / "05_model/5tt_wmseg.nii.gz",
        "b0_to_t1": subject / "03_spatial/b0_to_t1_bbr.mat",
        "b0_to_t1_image": subject / "03_spatial/b0_to_t1_bbr.nii.gz",
        "t1_to_b0": subject / "03_spatial/t1_to_b0_bbr.mat",
        "t1_to_b0_mrtrix": subject / "03_spatial/t1_to_b0_bbr_mrtrix.txt",
        "five_tt": subject / "05_model/5tt_dwi.mif",
        "gmwmi": subject / "05_model/gmwmi_dwi.mif",
        "b0_1mm": subject / "04_hcp/b0_1mm_world_grid.nii.gz",
        "hcp_t1_raw": subject / "04_hcp/HCPMMP1+aseg_t1.nii.gz",
        "hcp_t1_nodes": subject / "04_hcp/hcp379_nodes_t1.nii.gz",
        "hcp_t1_volumes": subject / "04_hcp/hcp379_node_volumes_t1.csv",
        "hcp_b0_raw": subject / "04_hcp/HCPMMP1+aseg_b0_1mm_raw.nii.gz",
        "hcp_nodes": subject / "04_hcp/hcp379_nodes_b0_1mm.nii.gz",
        "hcp_volumes": subject / "04_hcp/hcp379_node_volumes.csv",
        "mask_1mm": subject / "04_hcp/dwi_mask_1mm.nii.gz",
        "atlas_qc": subject / "04_hcp/atlas_qc.json",
        "atlas_native": subject / "04_hcp/hcp379_nodes_b0_native_qc.nii.gz",
        "atlas_overlay": subject / "04_hcp/review_b0_vs_hcp379.png",
        "fod_dwi": subject / "05_model/dwi_fod_shells.mif",
        "wmfod": subject / "05_model/wmfod.mif",
        "gm": subject / "05_model/gm.mif",
        "csf": subject / "05_model/csf.mif",
        "wmfod_norm": subject / "05_model/wmfod_norm.mif",
        "gm_norm": subject / "05_model/gm_norm.mif",
        "csf_norm": subject / "05_model/csf_norm.mif",
        "tensor": subject / "05_model/tensor.mif",
        "fa": subject / "05_model/fa.mif",
        "md": subject / "05_model/md.mif",
        "rd": subject / "05_model/rd.mif",
        "ad": subject / "05_model/ad.mif",
        "wmfod_l0": subject / "05_model/wmfod_l0.mif",
        "automated_qc": subject / "06_preflight/automated_qc.json",
    }


def process_unit(
    row: Mapping[str, str],
    *,
    root: Path,
    gate: Mapping[str, Any],
    nthreads: int,
) -> dict[str, Any]:
    unit = str(row["unit"])
    paths = stage_paths(root, unit)
    for path in paths.values():
        if path.suffix or path.name.endswith(".nii.gz"):
            path.parent.mkdir(parents=True, exist_ok=True)
    with paths["lock"].open("w") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        prior = (
            load_json(paths["state"]) if paths["state"].is_file() else None
        )
        if prior and prior.get("status") == "PASS_PRETRACT":
            required = [
                Path(record["path"])
                for record in prior.get("artifacts", {}).values()
            ]
            if required and all(
                path.is_file() and path.stat().st_size > 0
                for path in required
            ):
                return prior
        if prior and prior.get("status") != "PASS_PRETRACT":
            # A crashed or failed attempt is not a provenance-valid cache.
            # Clear only this script's new, subject-scoped output tree and
            # rebuild it from the audited immutable sources.
            subject_root = paths["subject"].resolve()
            expected_parent = (root / "subjects").resolve()
            if subject_root.parent != expected_parent:
                raise ValueError(
                    f"refusing unsafe subject cleanup: {subject_root}"
                )
            if subject_root.is_dir():
                for candidate in sorted(
                    subject_root.rglob("*"),
                    key=lambda value: len(value.parts),
                    reverse=True,
                ):
                    if candidate.is_file() or candidate.is_symlink():
                        candidate.unlink()
                    elif candidate.is_dir():
                        candidate.rmdir()
                subject_root.rmdir()
            for path in paths.values():
                if path.suffix or path.name.endswith(".nii.gz"):
                    path.parent.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        state: dict[str, Any] = {
            "schema_version": "1.0.0",
            "record_type": "diagnosis_blind_hcp379_scaleup_pretract",
            "status": "RUNNING",
            "unit": unit,
            "diagnosis_labels_used": False,
            "started_utc": utc_now(),
            "input_route": row["route"],
            "historical_nodes_b0_reused": False,
        }
        atomic_json(paths["state"], state)
        env = environment(nthreads)
        try:
            eddy_source = Path(row["eddy_path"])
            dwi_source = eddy_source
            with paths["log"].open("a", encoding="utf-8") as log:
                if row["route"] == "READY_AFTER_GRADIENT_HEADER_REPAIR":
                    run(
                        [
                            str(MRTRIX / "mrconvert"),
                            str(eddy_source),
                            str(paths["eddy_gradfix"]),
                            "-fslgrad",
                            str(DERIV / "mif_dwi" / f"{unit}.bvec"),
                            str(DERIV / "mif_dwi" / f"{unit}.bval"),
                            "-force",
                        ],
                        log=log,
                        env=env,
                        outputs=(paths["eddy_gradfix"],),
                    )
                    dwi_source = paths["eddy_gradfix"]
                run(
                    [
                        str(MRTRIX / "dwibiascorrect"),
                        "ants",
                        str(dwi_source),
                        str(paths["dwi"]),
                        "-nthreads",
                        str(nthreads),
                    ],
                    log=log,
                    env=env,
                    outputs=(paths["dwi"],),
                )
                run(
                    [
                        str(MRTRIX / "dwiextract"),
                        str(paths["dwi"]),
                        str(paths["b0_series"]),
                        "-bzero",
                        "-config",
                        "BZeroThreshold",
                        str(gate["bzero_threshold"]),
                        "-nthreads",
                        str(nthreads),
                    ],
                    log=log,
                    env=env,
                    outputs=(paths["b0_series"],),
                )
                run(
                    [
                        str(MRTRIX / "mrmath"),
                        str(paths["b0_series"]),
                        "mean",
                        str(paths["b0_mif"]),
                        "-axis",
                        "3",
                        "-nthreads",
                        str(nthreads),
                        "-force",
                    ],
                    log=log,
                    env=env,
                    outputs=(paths["b0_mif"],),
                )
                run(
                    [
                        str(MRTRIX / "mrconvert"),
                        str(paths["b0_mif"]),
                        str(paths["b0"]),
                        "-force",
                    ],
                    log=log,
                    env=env,
                    outputs=(paths["b0"],),
                )
                run(
                    [
                        str(MRTRIX / "dwi2mask"),
                        str(paths["dwi"]),
                        str(paths["mask"]),
                        "-config",
                        "BZeroThreshold",
                        str(gate["bzero_threshold"]),
                        "-nthreads",
                        str(nthreads),
                    ],
                    log=log,
                    env=env,
                    outputs=(paths["mask"],),
                )
                mask_qc = mask_physical_qc(paths["mask"])
                mask_method = "dwi2mask"
                if not (
                    MINIMUM_MASK_VOLUME_MM3
                    <= mask_qc["volume_mm3"]
                    <= MAXIMUM_MASK_VOLUME_MM3
                ):
                    b0_brain = paths["b0"].with_name("b0_brain.nii.gz")
                    b0_mask = paths["b0"].with_name(
                        "b0_brain_mask.nii.gz"
                    )
                    run(
                        [
                            str(FSL / "bet"),
                            str(paths["b0"]),
                            str(b0_brain),
                            "-m",
                            "-R",
                            "-f",
                            "0.20",
                        ],
                        log=log,
                        env=env,
                        outputs=(b0_brain, b0_mask),
                    )
                    paths["mask"].unlink(missing_ok=True)
                    run(
                        [
                            str(MRTRIX / "mrconvert"),
                            str(b0_mask),
                            str(paths["mask"]),
                            "-datatype",
                            "bit",
                            "-force",
                        ],
                        log=log,
                        env=env,
                        outputs=(paths["mask"],),
                    )
                    mask_qc = mask_physical_qc(paths["mask"])
                    mask_method = "bet_b0_fallback"
                if not (
                    MINIMUM_MASK_VOLUME_MM3
                    <= mask_qc["volume_mm3"]
                    <= MAXIMUM_MASK_VOLUME_MM3
                ):
                    raise ValueError(f"brain mask physical QC failed: {mask_qc}")

                t1_source = Path(row["fastsurfer_t1_path"])
                run(
                    [
                        str(ANTS / "N4BiasFieldCorrection"),
                        "-d",
                        "3",
                        "-i",
                        str(t1_source),
                        "-o",
                        f"[{paths['t1']},{paths['t1_bias']}]",
                    ],
                    log=log,
                    env=env,
                    outputs=(paths["t1"], paths["t1_bias"]),
                )
                run(
                    [
                        str(FSL / "bet"),
                        str(paths["t1"]),
                        str(paths["t1_brain"]),
                        "-m",
                        "-R",
                        "-f",
                        "0.30",
                    ],
                    log=log,
                    env=env,
                    outputs=(paths["t1_brain"], paths["t1_mask"]),
                )
                run(
                    [
                        str(MRTRIX / "5ttgen"),
                        "fsl",
                        str(paths["t1"]),
                        str(paths["five_tt_t1"]),
                        "-nocrop",
                        "-nthreads",
                        str(nthreads),
                        "-force",
                    ],
                    log=log,
                    env=env,
                    outputs=(paths["five_tt_t1"],),
                )
                run(
                    [str(MRTRIX / "5ttcheck"), str(paths["five_tt_t1"])],
                    log=log,
                    env=env,
                )
                wm_tissue = paths["wmseg"].with_suffix(".wm.mif")
                run(
                    [
                        str(MRTRIX / "mrconvert"),
                        str(paths["five_tt_t1"]),
                        str(wm_tissue),
                        "-coord",
                        "3",
                        "2",
                        "-force",
                    ],
                    log=log,
                    env=env,
                    outputs=(wm_tissue,),
                )
                run(
                    [
                        str(MRTRIX / "mrthreshold"),
                        str(wm_tissue),
                        str(paths["wmseg"]),
                        "-abs",
                        "0.5",
                        "-force",
                    ],
                    log=log,
                    env=env,
                    outputs=(paths["wmseg"],),
                )
                bbr_prefix = paths["b0_to_t1_image"].with_suffix("").with_suffix("")
                if not (
                    paths["b0_to_t1"].is_file()
                    and paths["b0_to_t1_image"].is_file()
                ):
                    run(
                        [
                            str(FSL / "epi_reg"),
                            f"--epi={paths['b0']}",
                            f"--t1={paths['t1']}",
                            f"--t1brain={paths['t1_brain']}",
                            f"--wmseg={paths['wmseg']}",
                            f"--out={bbr_prefix}",
                        ],
                        log=log,
                        env=env,
                        outputs=(
                            paths["b0_to_t1"],
                            paths["b0_to_t1_image"],
                        ),
                    )
                run(
                    [
                        str(FSL / "convert_xfm"),
                        "-omat",
                        str(paths["t1_to_b0"]),
                        "-inverse",
                        str(paths["b0_to_t1"]),
                    ],
                    log=log,
                    env=env,
                    outputs=(paths["t1_to_b0"],),
                )
                run(
                    [
                        str(MRTRIX / "transformconvert"),
                        str(paths["t1_to_b0"]),
                        str(paths["t1"]),
                        str(paths["b0"]),
                        "flirt_import",
                        str(paths["t1_to_b0_mrtrix"]),
                        "-force",
                    ],
                    log=log,
                    env=env,
                    outputs=(paths["t1_to_b0_mrtrix"],),
                )
                five_tt_raw = paths["five_tt"].with_name(
                    "5tt_dwi_unclipped.mif"
                )
                run(
                    [
                        str(MRTRIX / "mrtransform"),
                        str(paths["five_tt_t1"]),
                        str(five_tt_raw),
                        "-linear",
                        str(paths["t1_to_b0_mrtrix"]),
                        "-template",
                        str(paths["b0_mif"]),
                        "-interp",
                        "linear",
                        "-nthreads",
                        str(nthreads),
                        "-force",
                    ],
                    log=log,
                    env=env,
                    outputs=(five_tt_raw,),
                )
                run(
                    [
                        str(MRTRIX / "mrcalc"),
                        str(five_tt_raw),
                        "0",
                        "-max",
                        "1",
                        "-min",
                        str(paths["five_tt"]),
                        "-force",
                    ],
                    log=log,
                    env=env,
                    outputs=(paths["five_tt"],),
                )
                run(
                    [str(MRTRIX / "5ttcheck"), str(paths["five_tt"])],
                    log=log,
                    env=env,
                )
                run(
                    [
                        str(MRTRIX / "5tt2gmwmi"),
                        str(paths["five_tt"]),
                        str(paths["gmwmi"]),
                        "-force",
                    ],
                    log=log,
                    env=env,
                    outputs=(paths["gmwmi"],),
                )

                run(
                    [
                        str(MRTRIX / "mrgrid"),
                        str(paths["b0"]),
                        "regrid",
                        "-voxel",
                        "1.0",
                        str(paths["b0_1mm"]),
                        "-force",
                    ],
                    log=log,
                    env=env,
                    outputs=(paths["b0_1mm"],),
                )
                hcp_source = Path(row["hcp_source_parcellation_path"])
                run(
                    [
                        str(FREESURFER / "bin/mri_vol2vol"),
                        "--mov",
                        str(hcp_source),
                        "--targ",
                        str(paths["t1"]),
                        "--regheader",
                        "--interp",
                        "nearest",
                        "--keep-precision",
                        "--o",
                        str(paths["hcp_t1_raw"]),
                    ],
                    log=log,
                    env=env,
                    outputs=(paths["hcp_t1_raw"],),
                )
                run(
                    [
                        str(FSL_PYTHON),
                        str(RELABEL),
                        str(paths["hcp_t1_raw"]),
                        str(RELABEL_LUT),
                        str(paths["hcp_t1_nodes"]),
                        str(paths["hcp_t1_volumes"]),
                    ],
                    log=log,
                    env=env,
                    outputs=(
                        paths["hcp_t1_nodes"],
                        paths["hcp_t1_volumes"],
                    ),
                )
                run(
                    [
                        str(FSL / "flirt"),
                        "-in",
                        str(paths["hcp_t1_raw"]),
                        "-ref",
                        str(paths["b0_1mm"]),
                        "-applyxfm",
                        "-init",
                        str(paths["t1_to_b0"]),
                        "-interp",
                        "nearestneighbour",
                        "-datatype",
                        "int",
                        "-out",
                        str(paths["hcp_b0_raw"]),
                    ],
                    log=log,
                    env=env,
                    outputs=(paths["hcp_b0_raw"],),
                )
                run(
                    [
                        str(FSL_PYTHON),
                        str(RELABEL),
                        str(paths["hcp_b0_raw"]),
                        str(RELABEL_LUT),
                        str(paths["hcp_nodes"]),
                        str(paths["hcp_volumes"]),
                    ],
                    log=log,
                    env=env,
                    outputs=(paths["hcp_nodes"], paths["hcp_volumes"]),
                )
                run(
                    [
                        str(MRTRIX / "mrtransform"),
                        str(paths["mask"]),
                        str(paths["mask_1mm"]),
                        "-template",
                        str(paths["b0_1mm"]),
                        "-interp",
                        "nearest",
                        "-datatype",
                        "bit",
                        "-force",
                    ],
                    log=log,
                    env=env,
                    outputs=(paths["mask_1mm"],),
                )
                atlas_qc = ATLAS_QC_MODULE.atlas_qc(
                    unit=unit,
                    nodes_path=paths["hcp_nodes"],
                    reference_path=paths["b0_1mm"],
                    mask_path=paths["mask_1mm"],
                    node_volumes_path=paths["hcp_volumes"],
                    source_node_volumes_path=paths["hcp_t1_volumes"],
                )
                atomic_json(paths["atlas_qc"], atlas_qc)
                if atlas_qc["status"] != "PASS":
                    raise ValueError(
                        "corrected HCP379 atlas QC failed: "
                        + ";".join(atlas_qc["failures"])
                    )
                run(
                    [
                        str(MRTRIX / "mrtransform"),
                        str(paths["hcp_nodes"]),
                        str(paths["atlas_native"]),
                        "-template",
                        str(paths["b0"]),
                        "-interp",
                        "nearest",
                        "-force",
                    ],
                    log=log,
                    env=env,
                    outputs=(paths["atlas_native"],),
                )
                run(
                    [
                        str(FSL / "slices"),
                        str(paths["b0"]),
                        str(paths["atlas_native"]),
                        "-o",
                        str(paths["atlas_overlay"]),
                    ],
                    log=log,
                    env=env,
                    outputs=(paths["atlas_overlay"],),
                )

                run(
                    [
                        str(MRTRIX / "dwiextract"),
                        str(paths["dwi"]),
                        str(paths["fod_dwi"]),
                        "-shells",
                        "0,1000",
                        "-config",
                        "BZeroThreshold",
                        str(gate["bzero_threshold"]),
                        "-nthreads",
                        str(nthreads),
                        "-force",
                    ],
                    log=log,
                    env=env,
                    outputs=(paths["fod_dwi"],),
                )
                responses = gate["response_paths"]
                run(
                    [
                        str(SS3T_PYTHON),
                        str(SS3T_SCRIPT),
                        str(paths["fod_dwi"]),
                        str(responses["wm"]),
                        str(paths["wmfod"]),
                        str(responses["gm"]),
                        str(paths["gm"]),
                        str(responses["csf"]),
                        str(paths["csf"]),
                        "-mask",
                        str(paths["mask"]),
                        "-lmax",
                        str(gate["lmax"]),
                        "-config",
                        "BZeroThreshold",
                        str(gate["bzero_threshold"]),
                        "-nthreads",
                        str(nthreads),
                    ],
                    log=log,
                    env=env,
                    outputs=(paths["wmfod"], paths["gm"], paths["csf"]),
                )
                run(
                    [
                        str(MRTRIX / "mtnormalise"),
                        str(paths["wmfod"]),
                        str(paths["wmfod_norm"]),
                        str(paths["gm"]),
                        str(paths["gm_norm"]),
                        str(paths["csf"]),
                        str(paths["csf_norm"]),
                        "-mask",
                        str(paths["mask"]),
                        "-nthreads",
                        str(nthreads),
                        "-force",
                    ],
                    log=log,
                    env=env,
                    outputs=(
                        paths["wmfod_norm"],
                        paths["gm_norm"],
                        paths["csf_norm"],
                    ),
                )
                run(
                    [
                        str(MRTRIX / "dwi2tensor"),
                        str(paths["fod_dwi"]),
                        str(paths["tensor"]),
                        "-mask",
                        str(paths["mask"]),
                        "-iter",
                        "2",
                        "-config",
                        "BZeroThreshold",
                        str(gate["bzero_threshold"]),
                        "-nthreads",
                        str(nthreads),
                        "-force",
                    ],
                    log=log,
                    env=env,
                    outputs=(paths["tensor"],),
                )
                run(
                    [
                        str(MRTRIX / "tensor2metric"),
                        str(paths["tensor"]),
                        "-fa",
                        str(paths["fa"]),
                        "-adc",
                        str(paths["md"]),
                        "-rd",
                        str(paths["rd"]),
                        "-ad",
                        str(paths["ad"]),
                        "-nthreads",
                        str(nthreads),
                        "-force",
                    ],
                    log=log,
                    env=env,
                    outputs=(
                        paths["fa"],
                        paths["md"],
                        paths["rd"],
                        paths["ad"],
                    ),
                )
                run(
                    [
                        str(MRTRIX / "mrconvert"),
                        str(paths["wmfod_norm"]),
                        str(paths["wmfod_l0"]),
                        "-coord",
                        "3",
                        "0",
                        "-force",
                    ],
                    log=log,
                    env=env,
                    outputs=(paths["wmfod_l0"],),
                )

            ranges = {
                "wmfod_all_sh_coefficients": mrstats_range(
                    paths["wmfod_norm"], paths["mask"]
                ),
                "wmfod_l0": mrstats_range(
                    paths["wmfod_l0"], paths["mask"]
                ),
                "gm": mrstats_range(paths["gm_norm"], paths["mask"]),
                "csf": mrstats_range(paths["csf_norm"], paths["mask"]),
                "fa": mrstats_range(paths["fa"], paths["mask"]),
                "md": mrstats_range(paths["md"], paths["mask"]),
                "rd": mrstats_range(paths["rd"], paths["mask"]),
                "ad": mrstats_range(paths["ad"], paths["mask"]),
            }
            from scforge.input_contract import (
                assess_pre_tractography_image_ranges,
            )

            range_failures = assess_pre_tractography_image_ranges(ranges)
            automated_qc = {
                "schema_version": "1.0.0",
                "record_type": (
                    "diagnosis_blind_hcp379_scaleup_automated_pretract_qc"
                ),
                "status": "PASS" if not range_failures else "FAIL",
                "unit": unit,
                "diagnosis_labels_used": False,
                "mask_method": mask_method,
                "mask_qc": mask_qc,
                "atlas_qc": atlas_qc,
                "image_ranges": ranges,
                "range_failures": range_failures,
            }
            atomic_json(paths["automated_qc"], automated_qc)
            if range_failures:
                raise ValueError(
                    "pretract image-range QC failed: "
                    + ";".join(range_failures)
                )
            artifacts = {
                name: size_record(paths[name])
                for name in (
                    "dwi",
                    "b0",
                    "mask",
                    "t1",
                    "five_tt",
                    "gmwmi",
                    "t1_to_b0",
                    "hcp_nodes",
                    "hcp_volumes",
                    "atlas_qc",
                    "atlas_overlay",
                    "wmfod_norm",
                    "fa",
                    "md",
                    "rd",
                    "ad",
                    "automated_qc",
                )
            }
            state.update(
                {
                    "status": "PASS_PRETRACT",
                    "completed_utc": utc_now(),
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                    "mask_method": mask_method,
                    "mask_qc": mask_qc,
                    "atlas_qc": atlas_qc,
                    "image_ranges": ranges,
                    "pooled_response_records": gate["pooled_responses"],
                    "artifacts": artifacts,
                    "error": None,
                }
            )
        except Exception as exc:
            state.update(
                {
                    "status": "FAIL_PRETRACT",
                    "completed_utc": utc_now(),
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                    "error": f"{type(exc).__name__}:{exc}",
                }
            )
        atomic_json(paths["state"], state)
        return state


def write_summary(root: Path) -> dict[str, Any]:
    rows = []
    for path in sorted((root / "qc/subjects").glob("*.json")):
        try:
            state = load_json(path)
        except Exception:
            continue
        rows.append(
            {
                "unit": state.get("unit"),
                "status": state.get("status"),
                "elapsed_seconds": state.get("elapsed_seconds"),
                "mask_volume_mm3": state.get("mask_qc", {}).get(
                    "volume_mm3"
                ),
                "atlas_inside_mask_fraction": state.get(
                    "atlas_qc", {}
                ).get("atlas_inside_dwi_mask_fraction"),
                "error": state.get("error"),
            }
        )
    counts: dict[str, int] = {}
    for row in rows:
        status = str(row["status"])
        counts[status] = counts.get(status, 0) + 1
    summary = {
        "schema_version": "1.0.0",
        "record_type": "diagnosis_blind_hcp379_scaleup_pretract_summary",
        "status": (
            "PASS"
            if len(rows) == EXPECTED_SCALEUP_N
            and counts.get("PASS_PRETRACT") == EXPECTED_SCALEUP_N
            else "IN_PROGRESS"
        ),
        "updated_utc": utc_now(),
        "diagnosis_labels_used": False,
        "target_n": EXPECTED_SCALEUP_N,
        "states_present_n": len(rows),
        "status_counts": dict(sorted(counts.items())),
    }
    atomic_csv(root / "manifests/pretract_subjects.csv", rows)
    atomic_json(root / "manifests/pretract_summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--human-qc-manifest", type=Path, default=DEFAULT_HUMAN_QC
    )
    parser.add_argument("--root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--nthreads", type=int, default=8)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if (
        not 1 <= args.workers <= 8
        or not 1 <= args.nthreads <= 16
        or args.workers * args.nthreads > (os.cpu_count() or 1)
    ):
        raise ValueError("worker/thread request exceeds the host contract")
    sys_path = str(EXP / "scforge")
    if sys_path not in os.sys.path:
        os.sys.path.insert(0, sys_path)
    for path in (
        AUDIT_CSV,
        AUDIT_SUMMARY,
        RECOVERY3_BINDING,
        RECOVERY3_COMPLETION,
        ATLAS_SUMMARY,
        REVIEW_MANIFEST,
        CONFIG,
        ENVIRONMENT,
        ATLAS_QC_SOURCE,
        RELABEL,
        RELABEL_LUT,
        SS3T_PYTHON,
        SS3T_SCRIPT,
        FSL_PYTHON,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    gate = validate_global_gate(args.human_qc_manifest)
    canary = set(gate["canary_units"])
    audit_rows = read_csv(AUDIT_CSV)
    for row in audit_rows:
        unit = row["unit"]
        if (
            row.get("diagnosis_labels_used", "").lower() != "false"
            or not re.fullmatch(r"[0-9a-f]{64}", row.get("dti_source_id", ""))
            or not re.fullmatch(
                r"[0-9a-f]{64}", row.get("dti_raw_bundle_sha256", "")
            )
            or row.get("fastsurfer_t1_identity_match", "").lower() != "true"
            or row.get("hcp_source_parcellation_present", "").lower()
            != "true"
            or row.get("historical_nodes_b0_reuse_allowed", "").lower()
            != "false"
            or row.get("full_dwi_preprocessing_planned", "").lower()
            != "false"
            or row.get("expected_t1_image_id")
            != row.get("fastsurfer_t1_image_id")
        ):
            raise ValueError(f"input-audit identity contract differs for {unit}")
        for field in (
            "eddy_path",
            "fastsurfer_t1_path",
            "hcp_source_parcellation_path",
        ):
            source = Path(row[field]).resolve()
            if not source.is_file() or source.is_symlink():
                raise ValueError(
                    f"input-audit source is not a regular file for "
                    f"{unit}: {field}={source}"
                )
        if row["route"] == "READY_AFTER_GRADIENT_HEADER_REPAIR":
            for suffix in ("bvec", "bval"):
                sidecar = DERIV / "mif_dwi" / f"{unit}.{suffix}"
                if not sidecar.is_file() or sidecar.is_symlink():
                    raise ValueError(
                        f"gradient-repair sidecar differs for {unit}: "
                        f"{sidecar}"
                    )
    targets = [
        row
        for row in audit_rows
        if row["unit"] not in canary
        and row["route"]
        in {
            "READY_FOR_CORRECTED_PRETRACT_FROM_EDDY",
            "READY_AFTER_GRADIENT_HEADER_REPAIR",
        }
    ]
    if len(targets) != EXPECTED_SCALEUP_N:
        raise ValueError(f"non-canary scale-up set is {len(targets)}, not 216")
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be positive")
        targets = targets[: args.limit]
    for relative in (
        "subjects",
        "qc/subjects",
        "locks",
        "logs/subjects",
        "manifests",
    ):
        (args.root / relative).mkdir(parents=True, exist_ok=True)
    gate_public = {key: value for key, value in gate.items() if key != "response_paths"}
    atomic_json(args.root / "manifests/pretract_gate.json", gate_public)
    print(
        f"[{utc_now()}] HCP379_SCALEUP_PRETRACT_START "
        f"n={len(targets)} workers={args.workers} "
        f"nthreads={args.nthreads}",
        flush=True,
    )
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                process_unit,
                row,
                root=args.root,
                gate=gate,
                nthreads=args.nthreads,
            ): row["unit"]
            for row in targets
        }
        for index, future in enumerate(as_completed(futures), start=1):
            state = future.result()
            results.append(state)
            summary = write_summary(args.root)
            print(
                f"[{utc_now()}] {index}/{len(targets)} {state['unit']} "
                f"{state['status']} error={state.get('error')} "
                f"cohort_pass={summary['status_counts'].get('PASS_PRETRACT', 0)}",
                flush=True,
            )
    summary = write_summary(args.root)
    if args.limit is not None:
        return int(any(row["status"] != "PASS_PRETRACT" for row in results))
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
