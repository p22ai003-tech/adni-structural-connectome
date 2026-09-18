#!/home/ec2-user/fsl/bin/python
"""Independently validate the non-promoted 009 candidate and V5 review pack."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import nibabel as nib
import numpy as np
from PIL import Image


EXP = Path("/home/ec2-user/exp")
ROOT = Path("/data/derivatives/hcp379_v2")
UNIT = "009_S_4324_I1186579"
ROUTE = "bbr_initialized_syn_conservative"
ORIGINAL = ROOT / "pretract_recovery4/pretract_recovery4_manifest.json"
CANDIDATE_ROOT = ROOT / "pretract_candidate_recovery4_v4"
CANDIDATE = CANDIDATE_ROOT / "pretract_candidate_recovery4_v4_manifest.json"
REVIEW_ROOT = ROOT / "review_recovery4_candidate_overlay_v5"
REVIEW = (
    REVIEW_ROOT
    / "hcp379_visual_review_pack_recovery4_candidate_v5_manifest.json"
)
CANONICAL_HUMAN_QC = (
    ROOT
    / "review_recovery4_corrected_overlay/"
    "human_visual_qc_recovery4_corrected_overlay.csv"
)
OUTPUT_ROOT = (
    EXP
    / "research_audit/outputs/hcp379_009_candidate_pretract_v4"
)
OUTPUT = OUTPUT_ROOT / "validation.json"
MRINFO = Path("/home/ec2-user/mrtrix3/bin/mrinfo")
FIVE_TT_CHECK = Path("/home/ec2-user/mrtrix3/bin/5ttcheck")
BUILDER = (
    EXP / "scripts/hcp/prepare_hcp379_009_candidate_pretract_v4.py"
)
REVIEW_BUILDER = (
    EXP
    / "scripts/hcp/build_hcp379_visual_review_pack_recovery4_v5.py"
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
    if not resolved.is_file() or resolved.is_symlink():
        raise ValueError(f"not a regular file: {resolved}")
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root is not an object: {path}")
    return value


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        encoding="utf-8",
        delete=False,
    ) as handle:
        json.dump(
            dict(value),
            handle,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def file_records(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, Mapping):
        if {"path", "size_bytes", "sha256"}.issubset(value):
            yield dict(value)
            return
        for child in value.values():
            yield from file_records(child)
    elif isinstance(value, list):
        for child in value:
            yield from file_records(child)


def mrinfo_size(path: Path) -> list[int]:
    completed = subprocess.run(
        [str(MRINFO), str(path), "-size"],
        check=True,
        capture_output=True,
        text=True,
    )
    return [int(item) for item in completed.stdout.split()]


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(f"validation already exists: {OUTPUT}")
    checks: dict[str, dict[str, Any]] = {}

    def check(name: str, passed: bool, evidence: Any) -> None:
        checks[name] = {
            "status": "PASS" if passed else "FAIL",
            "evidence": evidence,
        }

    original = load_json(ORIGINAL)
    candidate = load_json(CANDIDATE)
    review = load_json(REVIEW)
    original_by_unit = {
        str(row["unit"]): row for row in original["units"]
    }
    candidate_by_unit = {
        str(row["unit"]): row for row in candidate["units"]
    }
    check(
        "exact_15_unit_candidate_master",
        candidate.get("record_type")
        == (
            "diagnosis_blind_hcp379_pretract_candidate_recovery4_v4_"
            "manifest"
        )
        and candidate.get("status")
        == "PROVISIONAL_AUTOMATED_PASS_AWAITING_HUMAN_VISUAL_QC"
        and candidate.get("unit_count") == 15
        and candidate.get("diagnosis_labels_used") is False
        and candidate.get("human_visual_qc_inferred") is False
        and candidate.get("tractography_started") is False
        and candidate.get("matrix_generation_started") is False
        and set(candidate_by_unit) == set(original_by_unit)
        and len(candidate_by_unit) == 15,
        {
            "status": candidate.get("status"),
            "unit_count": candidate.get("unit_count"),
            "units_equal_original": (
                set(candidate_by_unit) == set(original_by_unit)
            ),
        },
    )
    unchanged = [
        unit
        for unit in original_by_unit
        if unit != UNIT
        and candidate_by_unit.get(unit) == original_by_unit[unit]
    ]
    check(
        "fourteen_original_routes_are_byte_bound_unchanged",
        len(unchanged) == 14,
        {"unchanged_n": len(unchanged), "units": sorted(unchanged)},
    )
    substitution = candidate.get("route_substitution", {})
    check(
        "009_route_is_provisional_not_promoted",
        substitution.get("unit") == UNIT
        and substitution.get("candidate_route") == ROUTE
        and substitution.get("promotion_status") == "NOT_PROMOTED"
        and substitution.get("human_review_required") is True,
        substitution,
    )

    target_path = Path(
        candidate_by_unit[UNIT]["manifest"]["path"]
    )
    target = load_json(target_path)
    qc_path = Path(
        target["automated_qc"]["candidate_pretract"]["path"]
    )
    qc = load_json(qc_path)
    atlas = qc.get("atlas_qc", {})
    five_tt = qc.get("five_tt_qc", {})
    jacobian = qc.get("jacobian_in_dwi_mask", {})
    check(
        "009_quantitative_candidate_contract",
        target.get("status")
        == "PROVISIONAL_AUTOMATED_PASS_AWAITING_HUMAN_VISUAL_QC"
        and target.get("registration", {}).get("route") == ROUTE
        and five_tt.get("five_tt_dwi_mask_dice", 0) >= 0.70
        and atlas.get("labels_present") == 379
        and atlas.get("missing_labels") == []
        and atlas.get("minimum_node_voxels_1mm", 0) >= 1
        and atlas.get("atlas_inside_dwi_mask_fraction", 0) >= 0.80
        and atlas.get(
            "left_cortical_inside_dwi_mask_fraction", 0
        )
        >= 0.80
        and atlas.get(
            "right_cortical_inside_dwi_mask_fraction", 0
        )
        >= 0.80
        and jacobian.get("nonpositive_fraction") == 0,
        {
            "five_tt_dwi_mask_dice": five_tt.get(
                "five_tt_dwi_mask_dice"
            ),
            "labels_present": atlas.get("labels_present"),
            "minimum_node_voxels": atlas.get(
                "minimum_node_voxels_1mm"
            ),
            "atlas_inside_dwi": atlas.get(
                "atlas_inside_dwi_mask_fraction"
            ),
            "left_inside_dwi": atlas.get(
                "left_cortical_inside_dwi_mask_fraction"
            ),
            "right_inside_dwi": atlas.get(
                "right_cortical_inside_dwi_mask_fraction"
            ),
            "jacobian_nonpositive_fraction": jacobian.get(
                "nonpositive_fraction"
            ),
        },
    )

    five_tt_path = Path(
        target["tractography_inputs"]["five_tt"]["path"]
    )
    gmwmi_path = Path(
        target["tractography_inputs"]["gmwmi"]["path"]
    )
    check_run = subprocess.run(
        [str(FIVE_TT_CHECK), str(five_tt_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    five_tt_nifti = Path(qc["artifacts"]["five_tt_nifti"]["path"])
    values = nib.load(five_tt_nifti).get_fdata()
    check(
        "finalized_five_tt_and_gmwmi_are_tractography_ready",
        check_run.returncode == 0
        and mrinfo_size(five_tt_path) == [256, 256, 80, 5]
        and mrinfo_size(gmwmi_path) == [256, 256, 80]
        and values.shape == (256, 256, 80, 5)
        and np.all(np.isfinite(values))
        and values.min() >= -1.0e-8
        and values.max() <= 1.0 + 1.0e-6,
        {
            "5ttcheck_returncode": check_run.returncode,
            "five_tt_size": mrinfo_size(five_tt_path),
            "gmwmi_size": mrinfo_size(gmwmi_path),
            "five_tt_minimum": float(values.min()),
            "five_tt_maximum": float(values.max()),
        },
    )

    nodes_path = Path(
        target["hcp379"]["nodes_b0_1mm"]["path"]
    )
    nodes = np.rint(nib.load(nodes_path).get_fdata()).astype(np.int32)
    labels = sorted(
        int(value) for value in np.unique(nodes) if value > 0
    )
    check(
        "finalized_hcp379_has_exact_labels",
        labels == list(range(1, 380)),
        {
            "label_count": len(labels),
            "minimum_label": min(labels),
            "maximum_label": max(labels),
        },
    )

    records = list(
        file_records(
            {
                "candidate": candidate,
                "target": target,
                "qc": qc,
                "review": review,
            }
        )
    )
    mismatched = []
    for record in records:
        try:
            current = file_record(Path(record["path"]))
        except Exception as exc:
            mismatched.append(
                f"{record.get('path')}:{type(exc).__name__}:{exc}"
            )
            continue
        if current != record:
            mismatched.append(str(record["path"]))
    check(
        "all_candidate_and_review_file_records_are_current",
        not mismatched,
        {
            "record_count": len(records),
            "mismatched": mismatched,
        },
    )

    review_by_unit = {
        str(row["unit"]): row for row in review.get("units", [])
    }
    target_review = review_by_unit.get(UNIT, {})
    target_exact = target_review.get("exact_candidate_inputs", {})
    review_pages = []
    image_failures = []
    for row in review.get("units", []):
        page = Path(row["review_page"]["path"])
        review_pages.append(str(page))
        try:
            with Image.open(page) as image:
                image.verify()
        except Exception as exc:
            image_failures.append(
                f"{page}:{type(exc).__name__}:{exc}"
            )
    check(
        "v5_review_pack_is_exact_input_bound",
        review.get("status") == "READY_FOR_EXACT_INPUT_HUMAN_REVIEW"
        and review.get("unit_count") == 15
        and len(review_by_unit) == 15
        and review.get("candidate_pretract_master")
        == file_record(CANDIDATE)
        and target_review.get("registration_route") == ROUTE
        and target_exact.get("selected_t1")
        == target["registration"]["selected_t1_in_b0"]
        and target_exact.get("five_tt")
        == target["tractography_inputs"]["five_tt"]
        and target_exact.get("hcp379_nodes_native")
        == target["hcp379"]["nodes_b0_native_qc"]
        and not image_failures,
        {
            "status": review.get("status"),
            "unit_count": review.get("unit_count"),
            "review_pages": len(review_pages),
            "image_failures": image_failures,
        },
    )

    prohibited = [
        str(path)
        for path in CANDIDATE_ROOT.rglob("*")
        if path.is_file()
        and (
            path.suffix == ".tck"
            or path.name.startswith("SC_HCPMMP1_")
            or "tractography_summary" in path.name
        )
    ]
    check(
        "candidate_has_not_started_tractography_or_matrices",
        not prohibited
        and candidate.get("tractography_started") is False
        and candidate.get("matrix_generation_started") is False,
        {"prohibited_artifacts": prohibited},
    )
    check(
        "canonical_human_gate_remains_absent",
        not CANONICAL_HUMAN_QC.exists(),
        {"path": str(CANONICAL_HUMAN_QC)},
    )

    compile_runs = {}
    for source in (BUILDER, REVIEW_BUILDER):
        completed = subprocess.run(
            [
                "/home/ec2-user/fsl/bin/python",
                "-m",
                "py_compile",
                str(source),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        compile_runs[source.name] = {
            "returncode": completed.returncode,
            "stderr": completed.stderr,
        }
    check(
        "builders_compile",
        all(row["returncode"] == 0 for row in compile_runs.values()),
        compile_runs,
    )

    passed = sum(row["status"] == "PASS" for row in checks.values())
    report = {
        "schema_version": "1.0.0",
        "record_type": (
            "hcp379_009_candidate_pretract_recovery4_v4_validation"
        ),
        "status": "PASS" if passed == len(checks) else "FAIL",
        "generated_utc": utc_now(),
        "diagnosis_labels_used": False,
        "imaging_executed_by_validation": False,
        "tractography_started": False,
        "matrix_generation_started": False,
        "passed_checks": passed,
        "total_checks": len(checks),
        "checks": checks,
        "records": {
            "candidate_master": file_record(CANDIDATE),
            "candidate_unit": file_record(target_path),
            "candidate_qc": file_record(qc_path),
            "review_manifest": file_record(REVIEW),
            "builder": file_record(BUILDER),
            "review_builder": file_record(REVIEW_BUILDER),
        },
    }
    atomic_json(OUTPUT, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
