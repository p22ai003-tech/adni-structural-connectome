#!/usr/bin/env python3
"""Build non-overwriting HROI-corrected atlases for the 15 Phase-B canaries.

The HROI source correction changes only the source parcellation. Existing
FODs, 5TT images, transforms and tractograms are reused unchanged. This
builder reapplies each canary's already selected Recovery4 transform to the
hash-bound cohort-uniform HROI candidate and reruns direct 379-node QC.

Every execution writes a new attempt. It does not replace the original
Recovery4 atlas, authorize a streamline count, run tractography, or satisfy
final human visual QC.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


EXP = Path("/home/ec2-user/exp")
HCP_ROOT = Path("/data/derivatives/hcp379_v2")
BASE_SOURCE = (
    EXP
    / "scripts/hcp/hcp379_corrected_atlas_canary_recovery4_v2.py"
)
CANDIDATE_ROOT = (
    HCP_ROOT
    / "source_label_repair_v1/hroi_surface_support_v1"
)
POLICY_ADOPTION = (
    HCP_ROOT
    / "source_label_repair_v1/hroi_surface_support_cohort_v1/"
    "policy_adoption_v1.json"
)
PROMOTED_PRETRACT_MASTER = (
    HCP_ROOT
    / "pretract_promoted_recovery4_v6/"
    "pretract_promoted_recovery4_v6_manifest.json"
)
ATTEMPTS = (
    HCP_ROOT
    / "source_label_repair_v1/hroi_canary_atlas_v1/attempts"
)
EXPECTED_N = 15
TARGETED_SYN_UNIT = "009_S_4324_I1186579"


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = load_module(BASE_SOURCE, "hcp379_hroi_canary_atlas_base")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


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
    os.replace(temporary, path)


def candidate_record(unit: str) -> dict[str, Any]:
    manifest_path = CANDIDATE_ROOT / unit / "candidate_manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise FileNotFoundError(manifest_path)
    manifest = load_json(manifest_path)
    candidate = Path(
        str(manifest.get("candidate", {}).get("path", ""))
    ).resolve()
    original_result = (
        BASE.OUTPUT_ROOT / "subjects" / unit / "result.json"
    )
    original = load_json(original_result)
    if (
        manifest.get("record_type")
        != "diagnosis_blind_hcp379_hroi_surface_support_candidate"
        or manifest.get("status") != "PASS_ALL_379_SOURCE_LABELS"
        or manifest.get("diagnosis_labels_used") is not False
        or manifest.get("source_files_modified") is not False
        or manifest.get("unit") != unit
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
        or BASE.file_record(candidate) != manifest.get("candidate")
        or original.get("status") != "PASS"
        or original.get("artifacts", {}).get(
            "hcp_source_parcellation"
        )
        != manifest.get("original_source")
    ):
        raise ValueError(f"HROI canary source contract differs: {unit}")
    return {
        "candidate_manifest": BASE.file_record(manifest_path),
        "candidate": manifest["candidate"],
        "original_source": manifest["original_source"],
        "changed_voxel_n": int(manifest["changed_voxel_n"]),
    }


def required_candidates() -> tuple[list[str], dict[str, Any], list[str]]:
    units = sorted(BASE.BASE.load_units())
    if len(units) != EXPECTED_N or len(set(units)) != EXPECTED_N:
        raise ValueError("Phase-B canary unit set differs")
    candidates: dict[str, Any] = {}
    missing: list[str] = []
    for unit in units:
        try:
            candidates[unit] = candidate_record(unit)
        except FileNotFoundError:
            missing.append(unit)
    return units, candidates, missing


def registration_route(unit: str) -> dict[str, Any]:
    """Return the exact promoted registration route used by tractography.

    Recovery4 originally selected routes from the prospective BBR/rigid
    proxy.  The 009 canary was subsequently human-promoted to a conservative
    BBR-initialized SyN chain.  Re-entering the older route here would map a
    corrected tractogram to a different atlas geometry, so the HROI build must
    bind that promoted chain explicitly.
    """

    if unit != TARGETED_SYN_UNIT:
        return BASE.registration_route(unit)
    master = load_json(PROMOTED_PRETRACT_MASTER)
    rows = [
        row
        for row in master.get("units", [])
        if row.get("unit") == unit
    ]
    if (
        master.get("record_type")
        != "diagnosis_blind_hcp379_pretract_recovery4_manifest"
        or master.get("status")
        != "AUTOMATED_PASS_AWAITING_HUMAN_VISUAL_QC"
        or master.get("diagnosis_labels_used") is not False
        or len(rows) != 1
        or rows[0].get("promotion_status")
        != "HUMAN_VISUAL_QC_PASS_PROMOTED"
        or rows[0].get("registration_route")
        != "bbr_initialized_syn_conservative"
    ):
        raise ValueError("promoted 009 registration contract differs")
    manifest_path = Path(rows[0]["manifest"]["path"]).resolve()
    if BASE.file_record(manifest_path) != rows[0]["manifest"]:
        raise ValueError("promoted 009 manifest file record differs")
    manifest = load_json(manifest_path)
    registration = manifest.get("registration", {})
    chain = registration.get("transform_chain", {})
    warp_record = chain.get("warp", {})
    affine_record = chain.get("affine", {})
    warp = Path(str(warp_record.get("path", ""))).resolve()
    affine = Path(str(affine_record.get("path", ""))).resolve()
    if (
        manifest.get("record_type")
        != "diagnosis_blind_hcp379_pretract_recovery4_unit_manifest"
        or manifest.get("status")
        != "AUTOMATED_PASS_AWAITING_HUMAN_VISUAL_QC"
        or manifest.get("unit") != unit
        or manifest.get("promotion", {}).get("status")
        != "HUMAN_VISUAL_QC_PASS_ROUTE_PROMOTED"
        or registration.get("route")
        != "bbr_initialized_syn_conservative"
        or BASE.file_record(warp) != warp_record
        or BASE.file_record(affine) != affine_record
    ):
        raise ValueError("promoted 009 nonlinear route differs")
    original = BASE.registration_route(unit)
    return {
        "unit": unit,
        "route": "bbr_initialized_syn_conservative",
        "proxy_status": "PASS_PROMOTED_AFTER_DIRECT_HCP379_QC",
        "transform_format": "ants_chain",
        # Preserve the singular transform field for existing validators while
        # recording and applying the complete forward chain below.
        "transform": warp_record,
        "transforms": [warp_record, affine_record],
        "reason": (
            "the older direct-BBR route was superseded by the exact "
            "human-promoted BBR-initialized conservative SyN route used by "
            "Phase-B tractography"
        ),
        "recovery3_bbr": original["recovery3_bbr"],
        "seeded_ants_candidate": original["seeded_ants_candidate"],
        "promoted_candidate": {
            "status": "PASS",
            "five_tt_dwi_mask_dice": registration[
                "selected_5tt_dwi_mask_dice"
            ],
            "atlas_inside_dwi_mask_fraction": registration[
                "hcp379_atlas_inside_dwi_mask_fraction"
            ],
            "manifest": BASE.file_record(manifest_path),
        },
    }


def environment() -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "FREESURFER_HOME": str(BASE.FREESURFER),
            "FS_LICENSE": str(BASE.FREESURFER / "license.txt"),
            "SUBJECTS_DIR": str(BASE.FASTSURFER_ROOT),
            "FSLDIR": str(BASE.FSL.parent),
            "FSLOUTPUTTYPE": "NIFTI_GZ",
            "PATH": (
                f"{BASE.FREESURFER / 'bin'}:{BASE.FSL}:"
                f"{BASE.MRTRIX}:{BASE.ANTS}:{env.get('PATH', '')}"
            ),
        }
    )
    return env


def build_unit(
    unit: str,
    *,
    candidate: Mapping[str, Any],
    expected_t1: Mapping[str, str],
    route: Mapping[str, Any],
    attempt: Path,
) -> dict[str, Any]:
    output = attempt / "subjects" / unit
    output.mkdir(parents=True, exist_ok=False)
    result_path = output / "result.json"
    log_path = output / "build.log"
    hcp_source = Path(str(candidate["candidate"]["path"])).resolve()
    fastsurfer_t1 = BASE.BASE.fastsurfer_t1_path(unit)
    corrected_t1 = (
        BASE.RUN_ROOT / "subjects" / unit / "02_anat/t1_n4.nii.gz"
    )
    b0 = BASE.RUN_ROOT / "subjects" / unit / "03_spatial/mean_b0.nii.gz"
    b0_1mm = (
        BASE.RUN_ROOT
        / "subjects"
        / unit
        / "04_atlas/b0_1mm_world_grid.nii.gz"
    )
    dwi_mask_1mm = (
        BASE.RUN_ROOT
        / "subjects"
        / unit
        / "06_preflight/dwi_mask_1mm_recovery3.nii.gz"
    )
    transform_records = route.get("transforms", [route["transform"]])
    transforms = [
        Path(str(record["path"])).resolve()
        for record in transform_records
    ]
    if any(
        BASE.file_record(path) != record
        for path, record in zip(transforms, transform_records)
    ):
        raise ValueError("selected registration transform chain differs")
    transform = transforms[0]
    hcp_t1 = output / "HCPMMP1+aseg_hroi_t1_native.nii.gz"
    nodes_t1 = output / "hcp379_nodes_hroi_t1_native.nii.gz"
    node_volumes_t1 = output / "hcp379_node_volumes_hroi_t1_native.csv"
    hcp_b0_raw = output / "HCPMMP1+aseg_hroi_b0_1mm_raw.nii.gz"
    nodes = output / "hcp379_nodes_hroi_b0_1mm.nii.gz"
    node_volumes = output / "hcp379_node_volumes_hroi.csv"
    nodes_native = output / "hcp379_nodes_hroi_b0_native_qc.nii.gz"
    overlay = output / "review_b0_vs_hcp379_hroi.png"
    qc_path = output / "atlas_qc.json"
    observed_t1_id = BASE.BASE.t1_image_id(fastsurfer_t1)
    if observed_t1_id != expected_t1["t1_image_id"]:
        raise ValueError(f"FastSurfer T1 identity differs: {unit}")
    state: dict[str, Any] = {
        "schema_version": "1.0.0",
        "record_type": (
            "diagnosis_blind_hcp379_hroi_canary_atlas_build"
        ),
        "status": "RUNNING",
        "unit": unit,
        "diagnosis_labels_used": False,
        "source_files_modified": False,
        "non_overwriting": True,
        "tractography_started": False,
        "matrix_generation_started": False,
        "started_utc": utc_now(),
        "registration_route": dict(route),
        "hroi_candidate": dict(candidate),
    }
    atomic_json(result_path, state)
    started = time.monotonic()
    env = environment()
    try:
        with log_path.open("x", encoding="utf-8") as log:
            BASE.run(
                [
                    str(BASE.FREESURFER / "bin/mri_vol2vol"),
                    "--mov",
                    str(hcp_source),
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
            BASE.run(
                [
                    str(BASE.FSL_PYTHON),
                    str(BASE.RELABEL),
                    str(hcp_t1),
                    str(BASE.RELABEL_LUT),
                    str(nodes_t1),
                    str(node_volumes_t1),
                ],
                log=log,
                env=env,
                outputs=(nodes_t1, node_volumes_t1),
            )
            if route["transform_format"] == "fsl":
                transform_command = [
                    str(BASE.FSL / "flirt"),
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
                transform_command = [
                    str(BASE.ANTS / "antsApplyTransforms"),
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
            elif route["transform_format"] == "ants_chain":
                transform_command = [
                    str(BASE.ANTS / "antsApplyTransforms"),
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
                ]
                for chain_transform in transforms:
                    transform_command.extend(["-t", str(chain_transform)])
            else:
                raise ValueError("unknown Recovery4 transform format")
            BASE.run(
                transform_command,
                log=log,
                env=env,
                outputs=(hcp_b0_raw,),
            )
            BASE.run(
                [
                    str(BASE.FSL_PYTHON),
                    str(BASE.RELABEL),
                    str(hcp_b0_raw),
                    str(BASE.RELABEL_LUT),
                    str(nodes),
                    str(node_volumes),
                ],
                log=log,
                env=env,
                outputs=(nodes, node_volumes),
            )
        qc = BASE.BASE.atlas_qc(
            unit=unit,
            nodes_path=nodes,
            reference_path=b0_1mm,
            mask_path=dwi_mask_1mm,
            node_volumes_path=node_volumes,
            source_node_volumes_path=node_volumes_t1,
        )
        atomic_json(qc_path, qc)
        with log_path.open("a", encoding="utf-8") as log:
            BASE.run(
                [
                    str(BASE.MRTRIX / "mrtransform"),
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
            BASE.run(
                [
                    str(BASE.FSL / "slices"),
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
            "hcp_source_parcellation_original": candidate[
                "original_source"
            ],
            "hcp_source_parcellation_hroi": BASE.file_record(hcp_source),
            "hcp_source_candidate_manifest": candidate[
                "candidate_manifest"
            ],
            "fastsurfer_t1": BASE.file_record(fastsurfer_t1),
            "corrected_t1": BASE.file_record(corrected_t1),
            "selected_t1_to_b0_transform": BASE.file_record(transform),
            "selected_t1_to_b0_transform_chain": [
                BASE.file_record(path) for path in transforms
            ],
            "corrected_b0_1mm_reference": BASE.file_record(b0_1mm),
            "corrected_dwi_mask_1mm": BASE.file_record(dwi_mask_1mm),
            "hcp_t1_native": BASE.file_record(hcp_t1),
            "hcp379_nodes_t1_native": BASE.file_record(nodes_t1),
            "hcp379_node_volumes_t1_native": BASE.file_record(
                node_volumes_t1
            ),
            "hcp_b0_raw": BASE.file_record(hcp_b0_raw),
            "hcp379_nodes_b0_1mm": BASE.file_record(nodes),
            "hcp379_node_volumes": BASE.file_record(node_volumes),
            "hcp379_atlas_qc": BASE.file_record(qc_path),
            "hcp379_nodes_b0_native_qc": BASE.file_record(nodes_native),
            "hcp379_visual_overlay": BASE.file_record(overlay),
        }
        state.update(
            {
                "status": (
                    "PASS_HROI_CANARY_ATLAS"
                    if qc.get("status") == "PASS"
                    else "FAIL_HROI_CANARY_ATLAS_QC"
                ),
                "completed_utc": utc_now(),
                "elapsed_seconds": round(
                    time.monotonic() - started, 3
                ),
                "atlas_qc": qc,
                "artifacts": artifacts,
                "failure": (
                    None
                    if qc.get("status") == "PASS"
                    else "atlas_qc_failed"
                ),
            }
        )
    except Exception as exc:
        state.update(
            {
                "status": "FAIL_HROI_CANARY_ATLAS",
                "completed_utc": utc_now(),
                "elapsed_seconds": round(
                    time.monotonic() - started, 3
                ),
                "failure": f"{type(exc).__name__}:{exc}",
            }
        )
    atomic_json(result_path, state)
    return state


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        parser.error("--workers must be in 1..4")
    units, candidates, missing = required_candidates()
    preflight = {
        "schema_version": "1.0.0",
        "record_type": "hcp379_hroi_canary_atlas_preflight",
        "status": (
            "READY_HROI_CANARY_ATLAS_15"
            if not missing
            else "WAITING_FOR_HROI_CANARY_SOURCES"
        ),
        "generated_utc": utc_now(),
        "diagnosis_labels_used": False,
        "expected_unit_n": EXPECTED_N,
        "ready_unit_n": len(candidates),
        "missing_units": missing,
        "imaging_executed": False,
        "tractography_started": False,
        "matrix_generation_started": False,
    }
    if not args.execute or missing:
        print(json.dumps(preflight, indent=2, sort_keys=True))
        return 0
    attempt_id = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        + "-hroi-canary-atlas"
    )
    attempt = ATTEMPTS / attempt_id
    attempt.mkdir(parents=True, exist_ok=False)
    expected_t1 = BASE.BASE.execution_t1_by_unit()
    routes = {unit: registration_route(unit) for unit in units}
    atomic_json(
        attempt / "input_manifest.json",
        {
            **preflight,
            "status": "EXECUTION_INPUTS_BOUND",
            "attempt_id": attempt_id,
            "candidates": candidates,
            "routes": routes,
            "policy_adoption": (
                BASE.file_record(POLICY_ADOPTION)
                if POLICY_ADOPTION.is_file()
                else None
            ),
            "implementation": BASE.file_record(Path(__file__)),
            "base_implementation": BASE.file_record(BASE_SOURCE),
        },
    )
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                build_unit,
                unit,
                candidate=candidates[unit],
                expected_t1=expected_t1[unit],
                route=routes[unit],
                attempt=attempt,
            ): unit
            for unit in units
        }
        for index, future in enumerate(as_completed(futures), start=1):
            unit = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "unit": unit,
                    "status": "FAIL_HROI_CANARY_ATLAS",
                    "failure": f"{type(exc).__name__}:{exc}",
                }
            results.append(result)
            print(
                f"[{utc_now()}] hroi-canary-atlas "
                f"{index}/{EXPECTED_N} {unit} {result.get('status')}",
                flush=True,
            )
    passed = sum(
        row.get("status") == "PASS_HROI_CANARY_ATLAS"
        for row in results
    )
    summary = {
        "schema_version": "1.0.0",
        "record_type": "hcp379_hroi_canary_atlas_summary",
        "status": (
            "PASS_HROI_CANARY_ATLAS_15"
            if passed == EXPECTED_N
            else "FAIL_HROI_CANARY_ATLAS"
        ),
        "completed_utc": utc_now(),
        "attempt_id": attempt_id,
        "diagnosis_labels_used": False,
        "source_files_modified": False,
        "non_overwriting": True,
        "unit_n": EXPECTED_N,
        "passed_unit_n": passed,
        "failed_unit_n": EXPECTED_N - passed,
        "tractography_reused_not_regenerated": True,
        "matrix_generation_started": False,
        "policy_adoption": (
            BASE.file_record(POLICY_ADOPTION)
            if POLICY_ADOPTION.is_file()
            else None
        ),
        "units": [
            {
                "unit": row["unit"],
                "status": row.get("status"),
                "atlas_inside_dwi_mask_fraction": row.get(
                    "atlas_qc", {}
                ).get("atlas_inside_dwi_mask_fraction"),
                "labels_found": row.get("atlas_qc", {}).get(
                    "labels_found"
                ),
                "result": (
                    BASE.file_record(
                        attempt
                        / "subjects"
                        / row["unit"]
                        / "result.json"
                    )
                    if (
                        attempt
                        / "subjects"
                        / row["unit"]
                        / "result.json"
                    ).is_file()
                    else None
                ),
                "failure": row.get("failure"),
            }
            for row in sorted(results, key=lambda value: value["unit"])
        ],
        "implementation": BASE.file_record(Path(__file__)),
    }
    atomic_json(attempt / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["status"] == "PASS_HROI_CANARY_ATLAS_15" else 1


if __name__ == "__main__":
    raise SystemExit(main())
