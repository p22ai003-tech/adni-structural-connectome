#!/usr/bin/env python3
"""Build the Recovery4 HCP379 atlas canary with locked registration failover.

Recovery3 FSL BBR remains the primary route.  A deterministic, seeded ANTs
rigid transform is used only when the prospective Recovery3 spatial gate
failed and the independently generated ANTs candidate passed both locked
spatial thresholds.  If neither proxy route passes, the primary BBR transform
is retained for direct subject-specific HCP379 adjudication; this currently
applies only to the borderline AAL3 proxy case and does not waive HCP379 QC.

All outputs are written beneath a new Recovery4 root.  No Recovery3, FastSurfer
or historical derivative is modified, and no diagnosis or outcome field is
read.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


EXP = Path("/home/ec2-user/exp")
RUN_ROOT = Path(
    "/data/derivatives/scforge_v2/h04a_r1_recovery_20260719_retry4"
)
HCP_V2_ROOT = Path("/data/derivatives/hcp379_v2")
OUTPUT_ROOT = HCP_V2_ROOT / "corrected_atlas_canary_recovery4"
REGISTRATION_AUDIT_ROOT = (
    HCP_V2_ROOT / "diagnostics/registration_recovery4"
)
BASE_ATLAS_SCRIPT = (
    EXP / "scripts/hcp/hcp379_corrected_atlas_canary_v2.py"
)
HCP_SOURCE_ROOT = Path("/data/derivatives/parc_hcpmmp1")
FASTSURFER_ROOT = Path("/data/derivatives/fastsurfer")
FREESURFER = Path("/home/ec2-user/freesurfer")
FSL = Path("/home/ec2-user/fsl/bin")
MRTRIX = Path("/home/ec2-user/mrtrix3/bin")
ANTS = EXP / ".envs/ants-2.6.5/bin"
RELABEL = EXP / "scripts/hcp/relabel_hcp.py"
RELABEL_LUT = HCP_SOURCE_ROOT / "hcpmmp1_subcort_relabel.txt"
FSL_PYTHON = Path("/home/ec2-user/fsl/bin/python")


def load_base_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "hcp379_corrected_atlas_canary_v2",
        BASE_ATLAS_SCRIPT,
    )
    if spec is None or spec.loader is None:
        raise ImportError(BASE_ATLAS_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = load_base_module()


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
    if not resolved.is_file() or resolved.is_symlink():
        raise ValueError(f"not a regular file: {resolved}")
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


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


def run(
    command: list[str],
    *,
    log: Any,
    env: Mapping[str, str],
    outputs: tuple[Path, ...] = (),
) -> None:
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
    missing = [str(path) for path in outputs if not path.is_file()]
    if missing:
        raise RuntimeError("command outputs missing: " + ",".join(missing))


def spatial_record(unit: str) -> dict[str, Any]:
    subject = RUN_ROOT / "subjects" / unit
    report = (
        subject
        / "06_preflight/spatial_quantitative_qc_recovery3.json"
    )
    if report.is_file():
        return json.loads(report.read_text(encoding="utf-8"))
    log = subject / "logs/06_spatial_quantitative_qc_recovery3.log"
    text = log.read_text(encoding="utf-8")
    start = text.find("{")
    if start < 0:
        raise ValueError(f"no Recovery3 spatial record in {log}")
    return json.loads(text[start:])


def registration_route(unit: str) -> dict[str, Any]:
    primary = spatial_record(unit)
    candidate_path = REGISTRATION_AUDIT_ROOT / unit / "result.json"
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    if primary.get("status") == "PASS":
        route = "recovery3_fsl_bbr_primary"
        proxy_status = "PASS"
        transform = (
            RUN_ROOT
            / "subjects"
            / unit
            / "03_spatial/t1_to_b0_bbr.mat"
        )
        transform_format = "fsl"
        reason = "primary BBR passed both prospective Recovery3 spatial gates"
    elif candidate.get("status") == "PASS":
        route = "seeded_ants_rigid_failover"
        proxy_status = "PASS"
        transform = Path(candidate["artifacts"]["affine"]["path"])
        transform_format = "ants"
        reason = (
            "primary BBR failed a prospective spatial gate and the locked "
            "seeded ANTs rigid candidate passed both gates"
        )
    else:
        route = "recovery3_fsl_bbr_direct_hcp_adjudication"
        proxy_status = "FAIL_PENDING_DIRECT_HCP379_QC"
        transform = (
            RUN_ROOT
            / "subjects"
            / unit
            / "03_spatial/t1_to_b0_bbr.mat"
        )
        transform_format = "fsl"
        reason = (
            "neither AAL3 proxy route passed; retain the primary BBR transform "
            "without lowering thresholds and require direct HCP379 QC"
        )
    if not transform.is_file():
        raise FileNotFoundError(transform)
    return {
        "unit": unit,
        "route": route,
        "proxy_status": proxy_status,
        "transform_format": transform_format,
        "transform": file_record(transform),
        "reason": reason,
        "recovery3_bbr": {
            "status": primary.get("status"),
            "five_tt_dwi_mask_dice": primary.get(
                "five_tt_dwi_mask_dice"
            ),
            "atlas_inside_dwi_mask_fraction": primary.get(
                "atlas_inside_brain_fraction"
            ),
        },
        "seeded_ants_candidate": {
            "status": candidate.get("status"),
            "five_tt_dwi_mask_dice": candidate.get(
                "ants_rigid", {}
            ).get("five_tt_dwi_mask_dice"),
            "atlas_inside_dwi_mask_fraction": candidate.get(
                "ants_rigid", {}
            ).get("atlas_inside_dwi_mask_fraction"),
        },
    }


def existing_pass(result_path: Path) -> dict[str, Any] | None:
    if not result_path.is_file():
        return None
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("status") != "PASS":
            return None
        for record in result.get("artifacts", {}).values():
            path = Path(str(record["path"]))
            if (
                not path.is_file()
                or path.is_symlink()
                or path.stat().st_size != int(record["size_bytes"])
                or sha256_file(path) != record["sha256"]
            ):
                return None
        return result
    except Exception:
        return None


def build_unit(
    unit: str,
    *,
    expected_t1: Mapping[str, str],
    route: Mapping[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    subject_output = output_root / "subjects" / unit
    result_path = subject_output / "result.json"
    cached = existing_pass(result_path)
    if cached is not None:
        return cached
    subject_output.mkdir(parents=True, exist_ok=True)

    log_path = subject_output / "build.log"
    hcp_mgz = HCP_SOURCE_ROOT / unit / "HCPMMP1+aseg.mgz"
    fastsurfer_t1 = BASE.fastsurfer_t1_path(unit)
    corrected_t1 = RUN_ROOT / "subjects" / unit / "02_anat/t1_n4.nii.gz"
    b0 = RUN_ROOT / "subjects" / unit / "03_spatial/mean_b0.nii.gz"
    b0_1mm = (
        RUN_ROOT
        / "subjects"
        / unit
        / "04_atlas/b0_1mm_world_grid.nii.gz"
    )
    dwi_mask_1mm = (
        RUN_ROOT
        / "subjects"
        / unit
        / "06_preflight/dwi_mask_1mm_recovery3.nii.gz"
    )
    transform = Path(str(route["transform"]["path"]))
    hcp_t1 = subject_output / "HCPMMP1+aseg_t1_native.nii.gz"
    nodes_t1 = subject_output / "hcp379_nodes_t1_native.nii.gz"
    node_volumes_t1 = (
        subject_output / "hcp379_node_volumes_t1_native.csv"
    )
    hcp_b0_raw = subject_output / "HCPMMP1+aseg_b0_1mm_raw.nii.gz"
    nodes = subject_output / "hcp379_nodes_b0_1mm.nii.gz"
    node_volumes = subject_output / "hcp379_node_volumes.csv"
    nodes_native = subject_output / "hcp379_nodes_b0_native_qc.nii.gz"
    overlay = subject_output / "review_b0_vs_hcp379.png"
    qc_path = subject_output / "atlas_qc.json"

    required = (
        hcp_mgz,
        fastsurfer_t1,
        corrected_t1,
        b0,
        b0_1mm,
        dwi_mask_1mm,
        transform,
        RELABEL,
        RELABEL_LUT,
        FSL_PYTHON,
        FREESURFER / "bin/mri_vol2vol",
        FSL / "flirt",
        FSL / "slices",
        MRTRIX / "mrtransform",
        ANTS / "antsApplyTransforms",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "missing Recovery4 atlas inputs: " + ",".join(missing)
        )
    observed_t1_id = BASE.t1_image_id(fastsurfer_t1)
    if observed_t1_id != expected_t1["t1_image_id"]:
        raise ValueError(
            f"FastSurfer T1 differs for {unit}: "
            f"{observed_t1_id}!={expected_t1['t1_image_id']}"
        )

    state: dict[str, Any] = {
        "schema_version": "1.0.0",
        "record_type": (
            "diagnosis_blind_hcp379_corrected_atlas_recovery4_build"
        ),
        "status": "RUNNING",
        "unit": unit,
        "diagnosis_labels_used": False,
        "started_utc": utc_now(),
        "non_overwriting": True,
        "registration_route": dict(route),
    }
    atomic_json(result_path, state)
    started = time.monotonic()
    env = os.environ.copy()
    env.update(
        {
            "FREESURFER_HOME": str(FREESURFER),
            "FS_LICENSE": str(FREESURFER / "license.txt"),
            "SUBJECTS_DIR": str(FASTSURFER_ROOT),
            "FSLDIR": str(FSL.parent),
            "FSLOUTPUTTYPE": "NIFTI_GZ",
            "PATH": (
                f"{FREESURFER / 'bin'}:{FSL}:{MRTRIX}:{ANTS}:"
                f"{env.get('PATH', '')}"
            ),
        }
    )
    try:
        with log_path.open("a", encoding="utf-8") as log:
            run(
                [
                    str(FREESURFER / "bin/mri_vol2vol"),
                    "--mov",
                    str(hcp_mgz),
                    "--targ",
                    str(corrected_t1),
                    "--regheader",
                    "--interp",
                    "nearest",
                    "--keep-precision",
                    "--o",
                    str(hcp_t1),
                ],
                log=log,
                env=env,
                outputs=(hcp_t1,),
            )
            run(
                [
                    str(FSL_PYTHON),
                    str(RELABEL),
                    str(hcp_t1),
                    str(RELABEL_LUT),
                    str(nodes_t1),
                    str(node_volumes_t1),
                ],
                log=log,
                env=env,
                outputs=(nodes_t1, node_volumes_t1),
            )
            if route["transform_format"] == "fsl":
                command = [
                    str(FSL / "flirt"),
                    "-in",
                    str(hcp_t1),
                    "-ref",
                    str(b0_1mm),
                    "-applyxfm",
                    "-init",
                    str(transform),
                    "-interp",
                    "nearestneighbour",
                    "-datatype",
                    "int",
                    "-out",
                    str(hcp_b0_raw),
                ]
            elif route["transform_format"] == "ants":
                command = [
                    str(ANTS / "antsApplyTransforms"),
                    "-d",
                    "3",
                    "-i",
                    str(hcp_t1),
                    "-r",
                    str(b0_1mm),
                    "-o",
                    str(hcp_b0_raw),
                    "-n",
                    "GenericLabel",
                    "-t",
                    str(transform),
                ]
            else:
                raise ValueError(
                    f"unknown transform format: {route['transform_format']}"
                )
            run(
                command,
                log=log,
                env=env,
                outputs=(hcp_b0_raw,),
            )
            run(
                [
                    str(FSL_PYTHON),
                    str(RELABEL),
                    str(hcp_b0_raw),
                    str(RELABEL_LUT),
                    str(nodes),
                    str(node_volumes),
                ],
                log=log,
                env=env,
                outputs=(nodes, node_volumes),
            )

        qc = BASE.atlas_qc(
            unit=unit,
            nodes_path=nodes,
            reference_path=b0_1mm,
            mask_path=dwi_mask_1mm,
            node_volumes_path=node_volumes,
            source_node_volumes_path=node_volumes_t1,
        )
        atomic_json(qc_path, qc)
        with log_path.open("a", encoding="utf-8") as log:
            run(
                [
                    str(MRTRIX / "mrtransform"),
                    str(nodes),
                    str(nodes_native),
                    "-template",
                    str(b0),
                    "-interp",
                    "nearest",
                    "-force",
                ],
                log=log,
                env=env,
                outputs=(nodes_native,),
            )
            run(
                [
                    str(FSL / "slices"),
                    str(b0),
                    str(nodes_native),
                    "-o",
                    str(overlay),
                ],
                log=log,
                env=env,
                outputs=(overlay,),
            )

        artifacts = {
            "hcp_source_parcellation": file_record(hcp_mgz),
            "fastsurfer_t1": file_record(fastsurfer_t1),
            "corrected_t1": file_record(corrected_t1),
            "selected_t1_to_b0_transform": file_record(transform),
            "corrected_b0_1mm_reference": file_record(b0_1mm),
            "corrected_dwi_mask_1mm": file_record(dwi_mask_1mm),
            "hcp_t1_native": file_record(hcp_t1),
            "hcp379_nodes_t1_native": file_record(nodes_t1),
            "hcp379_node_volumes_t1_native": file_record(
                node_volumes_t1
            ),
            "hcp_b0_raw": file_record(hcp_b0_raw),
            "hcp379_nodes_b0_1mm": file_record(nodes),
            "hcp379_node_volumes": file_record(node_volumes),
            "hcp379_atlas_qc": file_record(qc_path),
            "hcp379_nodes_b0_native_qc": file_record(nodes_native),
            "hcp379_visual_overlay": file_record(overlay),
        }
        state.update(
            {
                "status": qc["status"],
                "completed_utc": utc_now(),
                "elapsed_seconds": round(
                    time.monotonic() - started, 3
                ),
                "t1_identity": {
                    "execution_manifest_t1_image_id": expected_t1[
                        "t1_image_id"
                    ],
                    "fastsurfer_t1_image_id": observed_t1_id,
                    "match": True,
                },
                "atlas_qc": qc,
                "artifacts": artifacts,
                "failure": (
                    None if qc["status"] == "PASS" else "atlas_qc_failed"
                ),
            }
        )
    except Exception as exc:
        state.update(
            {
                "status": "FAIL",
                "completed_utc": utc_now(),
                "elapsed_seconds": round(
                    time.monotonic() - started, 3
                ),
                "failure": f"{type(exc).__name__}:{exc}",
            }
        )
    atomic_json(result_path, state)
    return state


def write_summary(
    output_root: Path,
    results: list[dict[str, Any]],
    routes: Mapping[str, Mapping[str, Any]],
) -> Path:
    ordered = sorted(results, key=lambda row: str(row["unit"]))
    passed = sum(row.get("status") == "PASS" for row in ordered)
    route_counts: dict[str, int] = {}
    for route in routes.values():
        name = str(route["route"])
        route_counts[name] = route_counts.get(name, 0) + 1
    summary = {
        "schema_version": "1.0.0",
        "record_type": (
            "diagnosis_blind_hcp379_corrected_atlas_recovery4_summary"
        ),
        "status": "PASS" if passed == 15 else "FAIL",
        "updated_utc": utc_now(),
        "diagnosis_labels_used": False,
        "unit_count": len(ordered),
        "passed_unit_count": passed,
        "failed_unit_count": len(ordered) - passed,
        "registration_policy": {
            "primary": "Recovery3 FSL BBR",
            "failover": "seeded ANTs rigid only after prospective proxy failure",
            "thresholds_lowered": False,
            "direct_hcp379_qc_required_for_proxy_exception": True,
            "route_counts": dict(sorted(route_counts.items())),
        },
        "units": [
            {
                "unit": row["unit"],
                "status": row.get("status"),
                "registration_route": routes[row["unit"]]["route"],
                "proxy_status": routes[row["unit"]]["proxy_status"],
                "labels_found": row.get("atlas_qc", {}).get(
                    "labels_found"
                ),
                "atlas_inside_dwi_mask_fraction": row.get(
                    "atlas_qc", {}
                ).get("atlas_inside_dwi_mask_fraction"),
                "minimum_voxels_per_node": row.get(
                    "atlas_qc", {}
                ).get("minimum_voxels_per_node"),
                "failure": row.get("failure"),
                "result": file_record(
                    output_root
                    / "subjects"
                    / str(row["unit"])
                    / "result.json"
                ),
            }
            for row in ordered
        ],
    }
    path = output_root / "corrected_atlas_canary_recovery4_summary.json"
    atomic_json(path, summary)
    csv_path = output_root / "corrected_atlas_canary_recovery4_summary.csv"
    fields = (
        "unit",
        "status",
        "registration_route",
        "proxy_status",
        "labels_found",
        "atlas_inside_dwi_mask_fraction",
        "minimum_voxels_per_node",
        "failure",
    )
    with tempfile.NamedTemporaryFile(
        mode="w",
        dir=csv_path.parent,
        prefix=f".{csv_path.name}.",
        suffix=".tmp",
        newline="",
        encoding="utf-8",
        delete=False,
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in summary["units"]:
            writer.writerow({key: row.get(key) for key in fields})
        temporary = Path(handle.name)
    os.replace(temporary, csv_path)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        raise ValueError("--workers must be in 1..8")
    units = BASE.load_units()
    expected_t1 = BASE.execution_t1_by_unit()
    if not set(units).issubset(expected_t1):
        raise ValueError("execution manifest lacks a bound canary T1")
    routes = {unit: registration_route(unit) for unit in units}
    args.output_root.mkdir(parents=True, exist_ok=True)
    route_manifest = {
        "schema_version": "1.0.0",
        "record_type": (
            "diagnosis_blind_hcp379_recovery4_registration_routes"
        ),
        "created_utc": utc_now(),
        "diagnosis_labels_used": False,
        "selection_rule": (
            "BBR primary; seeded ANTs rigid only when BBR proxy failed and "
            "ANTs passed; otherwise retain BBR and require direct HCP379 QC"
        ),
        "thresholds_lowered": False,
        "routes": [routes[unit] for unit in sorted(routes)],
    }
    atomic_json(
        args.output_root / "registration_route_manifest.json",
        route_manifest,
    )

    print(
        f"[{utc_now()}] HCP379_RECOVERY4_ATLAS_START "
        f"n={len(units)} workers={args.workers}",
        flush=True,
    )
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                build_unit,
                unit,
                expected_t1=expected_t1[unit],
                route=routes[unit],
                output_root=args.output_root,
            ): unit
            for unit in units
        }
        for index, future in enumerate(as_completed(futures), start=1):
            unit = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "schema_version": "1.0.0",
                    "record_type": (
                        "diagnosis_blind_hcp379_corrected_atlas_"
                        "recovery4_build"
                    ),
                    "status": "FAIL",
                    "unit": unit,
                    "diagnosis_labels_used": False,
                    "completed_utc": utc_now(),
                    "failure": f"{type(exc).__name__}:{exc}",
                }
                atomic_json(
                    args.output_root
                    / "subjects"
                    / unit
                    / "result.json",
                    result,
                )
            results.append(result)
            print(
                f"[{utc_now()}] recovery4-atlas {index}/{len(units)} "
                f"{unit} {result.get('status')} "
                f"route={routes[unit]['route']} "
                f"inside={result.get('atlas_qc', {}).get('atlas_inside_dwi_mask_fraction')} "
                f"failure={result.get('failure')}",
                flush=True,
            )
    summary_path = write_summary(args.output_root, results, routes)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
