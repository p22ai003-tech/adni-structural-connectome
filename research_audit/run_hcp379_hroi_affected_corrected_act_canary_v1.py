#!/home/ec2-user/fsl/bin/python
"""Run one affected-case corrected-ACT canary for the H-ROI repair policy."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import yaml


UNIT = "002_S_0413_I863064"
STREAMLINES = 3_000_000
EXP = Path("/home/ec2-user/exp")
HCP_ROOT = Path("/data/derivatives/hcp379_v2")
SUBJECT = HCP_ROOT / "corrected_scaleup_recovery4/subjects" / UNIT
CANDIDATE_MANIFEST = (
    HCP_ROOT
    / "source_label_repair_v1/hroi_surface_support_v1"
    / UNIT
    / "candidate_manifest.json"
)
OUTPUT_ROOT = (
    HCP_ROOT
    / "source_label_repair_v1/validation/"
    "hroi_affected_corrected_act_v1/attempts"
)
AUDIT_CSV = (
    EXP
    / "research_audit/outputs/hcp379_scaleup_input_audit_v2/"
    "scaleup_input_audit.csv"
)
PLAN = (
    HCP_ROOT
    / "corrected_scaleup_recovery4/manifests/"
    "tractography_recovery4_selected_recipe_plan.json"
)
CONFIG = EXP / "configs/connectome_v2_retry3.yaml"
SUPPORT_SOURCE = (
    EXP / "research_audit/validate_hcp379_hroi_corrected_act_v1.py"
)
MRTRIX = Path("/home/ec2-user/mrtrix3/bin")
FREESURFER = Path("/home/ec2-user/freesurfer")
FSL = Path("/home/ec2-user/fsl/bin")
FSL_PYTHON = FSL / "python"
RELABEL = EXP / "scripts/hcp/relabel_hcp.py"
RELABEL_LUT = (
    Path("/data/derivatives/parc_hcpmmp1")
    / "hcpmmp1_subcort_relabel.txt"
)
TCK_HEADER_COUNT = EXP / "scripts/hcp/tck_header_count_v1.py"
POSSIBLE_EDGES = 379 * 378 // 2


def load_support() -> Any:
    spec = importlib.util.spec_from_file_location(
        "hcp379_hroi_corrected_act_support", SUPPORT_SOURCE
    )
    if spec is None or spec.loader is None:
        raise ImportError(SUPPORT_SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SUPPORT = load_support()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def audit_row() -> dict[str, str]:
    with AUDIT_CSV.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["unit"] == UNIT]
    if len(rows) != 1:
        raise ValueError("affected canary audit identity differs")
    row = rows[0]
    for field in ("dti_source_id", "dti_raw_bundle_sha256"):
        value = row[field]
        if len(value) != 64 or any(
            character not in "0123456789abcdef" for character in value
        ):
            raise ValueError(f"audit {field} differs")
    return row


def seed_record() -> tuple[str, int, str]:
    row = audit_row()
    plan = SUPPORT.load_json(PLAN)
    recipe_id = str(plan["recipe_id"])
    token = (
        f"{row['dti_source_id']}|{row['dti_raw_bundle_sha256']}|"
        f"{recipe_id}|hroi-affected-corrected-act-policy-canary-v1"
    )
    seed = int.from_bytes(
        hashlib.sha256(token.encode("utf-8")).digest()[:4], "big"
    ) & 0x7FFFFFFF
    return token, seed, recipe_id


def validate(attempt: Path, threads: int) -> dict[str, Any]:
    candidate_manifest = SUPPORT.load_json(CANDIDATE_MANIFEST)
    if (
        candidate_manifest.get("status") != "PASS_ALL_379_SOURCE_LABELS"
        or candidate_manifest.get("unit") != UNIT
        or candidate_manifest.get("diagnosis_labels_used") is not False
    ):
        raise ValueError("affected candidate manifest differs")
    scalar_qc_path = SUBJECT / "06_preflight/scalar_recovery4_qc.json"
    scalar_qc = SUPPORT.load_json(scalar_qc_path)
    bbr_qc_path = SUBJECT / "06_preflight/bbr_hcp379_atlas_qc_recovery4.json"
    bbr_qc = SUPPORT.load_json(bbr_qc_path)
    if (
        scalar_qc.get("status") != "PASS"
        or scalar_qc.get("diagnosis_labels_used") is not False
        or bbr_qc.get("failures")
        != [
            "label_set_differs:found=378:missing=[120]:unexpected=[]",
            "nodes_below_voxel_minimum:[120]",
            "source_or_mapped_node_volumes_nonpositive",
        ]
    ):
        raise ValueError("affected pretract failure is not H-only")

    candidate_source = SUPPORT.verified_path(
        candidate_manifest["candidate"], "candidate source"
    )
    corrected_t1 = SUBJECT / "02_anat/t1_n4.nii.gz"
    b0_1mm = SUBJECT / "04_hcp/b0_1mm_world_grid.nii.gz"
    transform = SUBJECT / "03_spatial/t1_to_b0_bbr.mat"
    original_nodes = SUBJECT / "04_hcp/hcp379_nodes_b0_1mm.nii.gz"
    dwi = SUBJECT / "01_dwi/dwi_biascorr.mif"
    wmfod = SUBJECT / "05_model/wmfod_norm.mif"
    five_tt = SUBJECT / "05_model/5tt_dwi.mif"
    for path in (
        corrected_t1,
        b0_1mm,
        transform,
        original_nodes,
        dwi,
        wmfod,
        five_tt,
        RELABEL,
        RELABEL_LUT,
        FREESURFER / "bin/mri_vol2vol",
        FSL / "flirt",
        MRTRIX / "mrinfo",
        MRTRIX / "tckgen",
        MRTRIX / "tck2connectome",
        TCK_HEADER_COUNT,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    environment = os.environ.copy()
    environment.update(
        {
            "FREESURFER_HOME": str(FREESURFER),
            "FS_LICENSE": str(FREESURFER / "license.txt"),
            "SUBJECTS_DIR": "/data/derivatives/fastsurfer",
            "FSLDIR": str(FSL.parent),
            "FSLOUTPUTTYPE": "NIFTI_GZ",
            "PATH": (
                f"{FREESURFER / 'bin'}:{FSL}:{MRTRIX}:"
                f"{environment.get('PATH', '')}"
            ),
        }
    )
    candidate_t1_raw = attempt / "HCPMMP1+aseg_hroi_t1.nii.gz"
    candidate_t1_nodes = attempt / "hcp379_hroi_nodes_t1.nii.gz"
    candidate_t1_volumes = attempt / "hcp379_hroi_volumes_t1.csv"
    candidate_b0_raw = attempt / "HCPMMP1+aseg_hroi_b0_1mm_raw.nii.gz"
    candidate_nodes = attempt / "hcp379_hroi_nodes_b0_1mm.nii.gz"
    candidate_volumes = attempt / "hcp379_hroi_volumes_b0_1mm.csv"
    tracks_partial = attempt / "tracks_3m.partial.tck"
    tracks = attempt / "tracks_3m.tck"
    count_path = attempt / "count.csv"
    assignments = attempt / "assignments.csv"

    SUPPORT.run_logged(
        [
            str(FREESURFER / "bin/mri_vol2vol"),
            "--mov",
            str(candidate_source),
            "--targ",
            str(corrected_t1),
            "--regheader",
            "--interp",
            "nearest",
            "--keep-precision",
            "--o",
            str(candidate_t1_raw),
        ],
        log=attempt / "01_candidate_to_t1.log",
        environment=environment,
        outputs=(candidate_t1_raw,),
    )
    SUPPORT.run_logged(
        [
            str(FSL_PYTHON),
            str(RELABEL),
            str(candidate_t1_raw),
            str(RELABEL_LUT),
            str(candidate_t1_nodes),
            str(candidate_t1_volumes),
        ],
        log=attempt / "02_candidate_relabel_t1.log",
        environment=environment,
        outputs=(candidate_t1_nodes, candidate_t1_volumes),
    )
    SUPPORT.run_logged(
        [
            str(FSL / "flirt"),
            "-in",
            str(candidate_t1_raw),
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
            str(candidate_b0_raw),
        ],
        log=attempt / "03_candidate_to_b0.log",
        environment=environment,
        outputs=(candidate_b0_raw,),
    )
    SUPPORT.run_logged(
        [
            str(FSL_PYTHON),
            str(RELABEL),
            str(candidate_b0_raw),
            str(RELABEL_LUT),
            str(candidate_nodes),
            str(candidate_volumes),
        ],
        log=attempt / "04_candidate_relabel_b0.log",
        environment=environment,
        outputs=(candidate_nodes, candidate_volumes),
    )

    spacing_text = subprocess.run(
        [str(MRTRIX / "mrinfo"), str(dwi), "-spacing"],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    ).stdout
    spacing = [float(value) for value in spacing_text.split()[:3]]
    step = min(spacing) / 2.0
    token, seed, recipe_id = seed_record()
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    recipe = config["tractography"]
    environment["MRTRIX_RNG_SEED"] = str(seed)
    SUPPORT.run_logged(
        [
            str(MRTRIX / "tckgen"),
            str(wmfod),
            str(tracks_partial),
            "-algorithm",
            "iFOD2",
            "-act",
            str(five_tt),
            "-backtrack",
            "-crop_at_gmwmi",
            "-seed_dynamic",
            str(wmfod),
            "-select",
            str(STREAMLINES),
            "-seeds",
            str(recipe["maximum_seed_attempts"]),
            "-minlength",
            str(recipe["minimum_length_mm"]),
            "-maxlength",
            str(recipe["maximum_length_mm"]),
            "-cutoff",
            str(recipe["cutoff"]),
            "-step",
            f"{step:.8f}",
            "-angle",
            str(recipe["maximum_angle_degrees"]),
            "-nthreads",
            str(threads),
        ],
        log=attempt / "05_tckgen.log",
        environment=environment,
        outputs=(tracks_partial,),
    )
    observed = int(
        subprocess.run(
            [
                str(FSL_PYTHON),
                str(TCK_HEADER_COUNT),
                str(tracks_partial),
                "--expect",
                str(STREAMLINES),
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    os.replace(tracks_partial, tracks)
    SUPPORT.run_logged(
        [
            str(MRTRIX / "tck2connectome"),
            str(tracks),
            str(candidate_nodes),
            str(count_path),
            "-assignment_radial_search",
            "4",
            "-symmetric",
            "-zero_diagonal",
            "-stat_edge",
            "sum",
            "-out_assignments",
            str(assignments),
            "-nthreads",
            str(threads),
        ],
        log=attempt / "06_tck2connectome.log",
        environment=environment,
        outputs=(count_path, assignments),
    )

    original_image, original = SUPPORT.integer_data(original_nodes)
    candidate_image, candidate = SUPPORT.integer_data(candidate_nodes)
    geometry_pass = bool(
        original_image.shape == candidate_image.shape
        and np.allclose(
            original_image.affine, candidate_image.affine, atol=1e-5, rtol=0
        )
    )
    changed = original != candidate
    allowed = (original == 0) & np.isin(
        candidate, np.asarray([120, 300], dtype=np.int32)
    )
    changed_only_as_intended = bool(
        np.count_nonzero(changed & ~allowed) == 0
    )
    labels = set(int(value) for value in np.unique(candidate))
    all_nodes = set(range(1, 380)).issubset(labels)
    matrix = np.loadtxt(count_path, delimiter=",")
    assignment = SUPPORT.assignment_summary(assignments)
    edges = np.triu(matrix, 1) > 0
    names = SUPPORT.node_names(UNIT)
    technical_pass = bool(
        observed == STREAMLINES
        and geometry_pass
        and changed_only_as_intended
        and all_nodes
        and SUPPORT.matrix_contract(matrix)
        and assignment["row_count"] == STREAMLINES
        and assignment["malformed_row_count"] == 0
        and assignment["endpoint_assignment_fraction"] >= 0.50
        and np.count_nonzero(matrix[119]) > 0
        and np.count_nonzero(matrix[299]) > 0
    )
    result = {
        "schema_version": "1.0.0",
        "record_type": (
            "hcp379_hroi_affected_corrected_act_policy_canary"
        ),
        "status": (
            "TECHNICAL_PASS_AFFECTED_CORRECTED_ACT_POLICY_REVIEW_PENDING"
            if technical_pass
            else "FAIL"
        ),
        "generated_utc": utc_now(),
        "diagnosis_labels_used": False,
        "source_files_modified": False,
        "unit": UNIT,
        "purpose": (
            "corrected-ACT validation of the H-ROI support policy in a "
            "source atlas genuinely missing node 120"
        ),
        "streamline_count": STREAMLINES,
        "tractography": {
            "algorithm": "iFOD2",
            "act": True,
            "backtrack": True,
            "crop_at_gmwmi": True,
            "seeding": "seed_dynamic",
            "rng_seed": seed,
            "seed_token": token,
            "recipe_id": recipe_id,
            "step_size_mm": step,
        },
        "atlas": {
            "original_missing_nodes": [120],
            "geometry_pass": geometry_pass,
            "changed_only_zero_to_h_nodes": changed_only_as_intended,
            "unexpected_changed_voxel_n": int(
                np.count_nonzero(changed & ~allowed)
            ),
            "all_379_candidate_nodes_present": all_nodes,
            "node120_voxels_original": int((original == 120).sum()),
            "node120_voxels_candidate": int((candidate == 120).sum()),
            "node300_voxels_original": int((original == 300).sum()),
            "node300_voxels_candidate": int((candidate == 300).sum()),
        },
        "assignment": assignment,
        "topology": {
            "matrix_contract_pass": SUPPORT.matrix_contract(matrix),
            "density": float(edges.sum() / POSSIBLE_EDGES),
            "edge_n": int(edges.sum()),
            "node120_degree": int(np.count_nonzero(matrix[119])),
            "node300_degree": int(np.count_nonzero(matrix[299])),
            "node120_strength": float(matrix[119].sum()),
            "node300_strength": float(matrix[299].sum()),
        },
        "anatomical_neighbours": {
            "node120_name": names[120],
            "node120_top15": SUPPORT.top_neighbours(
                matrix, 120, names
            ),
            "node300_name": names[300],
            "node300_top15": SUPPORT.top_neighbours(
                matrix, 300, names
            ),
        },
        "inputs": {
            "candidate_manifest": SUPPORT.file_record(
                CANDIDATE_MANIFEST
            ),
            "scalar_qc": SUPPORT.file_record(scalar_qc_path),
            "bbr_atlas_qc": SUPPORT.file_record(bbr_qc_path),
            "wmfod": SUPPORT.file_record(wmfod),
            "five_tt": SUPPORT.file_record(five_tt),
            "transform": SUPPORT.file_record(transform),
        },
        "outputs": {
            "candidate_nodes": SUPPORT.file_record(candidate_nodes),
            "tractogram": SUPPORT.file_record(
                tracks, hash_content=False
            ),
            "count": SUPPORT.file_record(count_path),
            "assignments": SUPPORT.file_record(
                assignments, hash_content=False
            ),
        },
        "implementation": SUPPORT.file_record(Path(__file__)),
    }
    SUPPORT.atomic_json(attempt / "summary.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.threads <= 24:
        parser.error("--threads must be in 1..24")
    for path in (
        CANDIDATE_MANIFEST,
        AUDIT_CSV,
        PLAN,
        CONFIG,
        SUPPORT_SOURCE,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not args.execute:
        print(
            json.dumps(
                {
                    "status": "DRY_RUN_PASS",
                    "unit": UNIT,
                    "streamlines": STREAMLINES,
                    "output_root": str(OUTPUT_ROOT),
                },
                sort_keys=True,
            )
        )
        return 0
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    attempt = OUTPUT_ROOT / f"{stamp}-affected-corrected-act-canary"
    attempt.mkdir(parents=True, exist_ok=False)
    try:
        result = validate(attempt, args.threads)
    except Exception as exc:
        result = {
            "schema_version": "1.0.0",
            "record_type": (
                "hcp379_hroi_affected_corrected_act_policy_canary"
            ),
            "status": "FAIL",
            "generated_utc": utc_now(),
            "diagnosis_labels_used": False,
            "source_files_modified": False,
            "unit": UNIT,
            "failure": f"{type(exc).__name__}: {exc}",
            "implementation": SUPPORT.file_record(Path(__file__)),
        }
        SUPPORT.atomic_json(attempt / "summary.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"].startswith("TECHNICAL_PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
