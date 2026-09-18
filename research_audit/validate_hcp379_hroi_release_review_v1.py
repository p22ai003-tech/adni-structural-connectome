#!/home/ec2-user/fsl/bin/python
"""Independently validate the 16-page HROI human-review package."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
from PIL import Image
from PyPDF2 import PdfReader


ATTEMPT_ROOT = Path(
    "/data/derivatives/hcp379_v2/source_label_repair_v1/review/"
    "hroi_release_v1/attempts"
)
POLICY_RECEIPT = Path(
    "/data/derivatives/hcp379_v2/source_label_repair_v1/"
    "hroi_surface_support_cohort_v1/policy_adoption_v1.json"
)
CANARY_SUMMARY = Path(
    "/data/derivatives/hcp379_v2/source_label_repair_v1/"
    "hroi_canary_atlas_v1/attempts/"
    "20260728T081233.507961Z-hroi-canary-atlas/summary.json"
)
RECOVERED_UNIT = "114_S_6347_I1344943"
OUTPUT = Path(
    "/home/ec2-user/exp/research_audit/outputs/"
    "hcp379_hroi_release_review_v1/validation.json"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record_matches(record: dict[str, Any]) -> bool:
    path = Path(record["path"]).resolve()
    return bool(
        path.is_file()
        and not path.is_symlink()
        and path.stat().st_size == int(record["size_bytes"])
        and sha256_file(path) == str(record["sha256"])
    )


def canonical_data(path: Path) -> tuple[np.ndarray, np.ndarray]:
    image = nib.as_closest_canonical(nib.load(str(path)))
    data = np.asanyarray(image.dataobj)
    if data.ndim == 4 and data.shape[-1] == 1:
        data = data[..., 0]
    if data.ndim != 3:
        raise ValueError(f"image is not 3D: {path}")
    return data, image.affine


def latest_manifest() -> Path:
    candidates = sorted(ATTEMPT_ROOT.glob("*/review_manifest.json"))
    if not candidates:
        raise FileNotFoundError("no complete HROI review manifest exists")
    return candidates[-1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=None)
    args = parser.parse_args()
    manifest_path = (
        args.manifest.resolve() if args.manifest else latest_manifest()
    )
    manifest = load_json(manifest_path)
    policy = load_json(POLICY_RECEIPT)
    canary = load_json(CANARY_SUMMARY)
    failures: list[str] = []
    checks: list[dict[str, Any]] = []

    def check(name: str, condition: bool, detail: Any) -> None:
        checks.append(
            {"name": name, "pass": bool(condition), "detail": detail}
        )
        if not condition:
            failures.append(name)

    check(
        "manifest_contract",
        bool(
            manifest.get("record_type")
            == "diagnosis_blind_hcp379_hroi_release_visual_review"
            and manifest.get("status") == "READY_FOR_HUMAN_VISUAL_QC"
            and manifest.get("unit_n") == 16
            and manifest.get("diagnosis_labels_used") is False
            and manifest.get("human_visual_qc_inferred") is False
            and manifest.get("final_release_authorized") is False
            and manifest.get("selected_count_tractography_authorized")
            is False
        ),
        manifest.get("status"),
    )

    expected_units = [
        str(row["unit"]) for row in canary.get("units", [])
    ] + [RECOVERED_UNIT]
    observed_units = [
        str(row["unit"]) for row in manifest.get("units", [])
    ]
    check(
        "exact_unit_scope",
        bool(
            canary.get("status") == "PASS_HROI_CANARY_ATLAS_15"
            and canary.get("unit_n") == 15
            and len(expected_units) == 16
            and len(set(expected_units)) == 16
            and observed_units == expected_units
        ),
        observed_units,
    )

    pdf_record = manifest.get("pdf", {})
    pdf_ok = isinstance(pdf_record, dict) and record_matches(pdf_record)
    pdf_pages = (
        len(PdfReader(str(Path(pdf_record["path"]))).pages)
        if pdf_ok
        else 0
    )
    check(
        "pdf_integrity_and_page_count",
        bool(pdf_ok and pdf_pages == 16),
        {"record_matches": pdf_ok, "pages": pdf_pages},
    )

    template_record = manifest.get("review_template", {})
    template_ok = (
        isinstance(template_record, dict) and record_matches(template_record)
    )
    template_rows: list[dict[str, str]] = []
    if template_ok:
        with Path(template_record["path"]).open(
            encoding="utf-8", newline=""
        ) as handle:
            template_rows = list(csv.DictReader(handle))
    review_fields = (
        "source_support_status",
        "dwi_atlas_status",
        "overall_status",
        "reviewer",
        "reviewed_utc",
        "note",
    )
    check(
        "blank_review_template",
        bool(
            template_ok
            and [row.get("unit") for row in template_rows]
            == expected_units
            and all(
                all((row.get(field) or "") == "" for field in review_fields)
                for row in template_rows
            )
        ),
        {"record_matches": template_ok, "row_n": len(template_rows)},
    )

    receipt_units = policy.get("units", {})
    source_failures: list[str] = []
    page_failures: list[str] = []
    evidence_failures: list[str] = []
    for row in manifest.get("units", []):
        unit = str(row.get("unit", ""))
        receipt = receipt_units.get(unit, {})
        for name in (
            "original_source",
            "candidate",
            "candidate_manifest",
            "t1",
            "dwi_overlay",
            "page_png",
        ):
            record = row.get(name, {})
            if not isinstance(record, dict) or not record_matches(record):
                page_failures.append(f"{unit}:{name}:record")
        page_path = Path(row["page_png"]["path"])
        try:
            with Image.open(page_path) as image:
                if image.format != "PNG" or image.size != (2250, 1530):
                    page_failures.append(f"{unit}:page_png:format")
        except Exception as exc:
            page_failures.append(
                f"{unit}:page_png:{type(exc).__name__}"
            )
        if (
            row.get("original_source", {}).get("sha256")
            != receipt.get("original_source", {}).get("sha256")
            or row.get("candidate", {}).get("sha256")
            != receipt.get("candidate", {}).get("sha256")
            or row.get("candidate_manifest", {}).get("sha256")
            != receipt.get("candidate_manifest", {}).get("sha256")
        ):
            source_failures.append(f"{unit}:receipt_binding")
            continue
        original, original_affine = canonical_data(
            Path(row["original_source"]["path"])
        )
        candidate, candidate_affine = canonical_data(
            Path(row["candidate"]["path"])
        )
        changed = candidate != original
        left = int((changed & (candidate == 1120)).sum())
        right = int((changed & (candidate == 2120)).sum())
        if (
            original.shape != candidate.shape
            or not np.allclose(
                original_affine, candidate_affine, atol=1e-5
            )
            or int(changed.sum()) != left + right
            or left != int(row.get("left_added_source_voxels", -1))
            or right != int(row.get("right_added_source_voxels", -1))
            or left + right != int(receipt.get("changed_voxel_n", -1))
        ):
            source_failures.append(f"{unit}:independent_source_replay")
        if (
            not 0.0
            < float(row.get("atlas_inside_dwi_mask_fraction", 0.0))
            <= 1.0
            or not str(row.get("registration_route", "")).strip()
        ):
            evidence_failures.append(f"{unit}:dwi_metrics")

    check(
        "all_page_artifacts",
        not page_failures,
        page_failures,
    )
    check(
        "independent_source_policy_replay",
        not source_failures,
        source_failures,
    )
    check(
        "dwi_evidence_metrics",
        not evidence_failures,
        evidence_failures,
    )
    check(
        "immutable_upstream_bindings",
        bool(
            record_matches(manifest["policy_receipt"])
            and record_matches(manifest["canary_summary"])
            and manifest["policy_receipt"]["sha256"]
            == sha256_file(POLICY_RECEIPT)
            and manifest["canary_summary"]["sha256"]
            == sha256_file(CANARY_SUMMARY)
        ),
        {
            "policy": manifest.get("policy_receipt", {}).get("sha256"),
            "canary": manifest.get("canary_summary", {}).get("sha256"),
        },
    )
    check(
        "implementation_bindings",
        bool(
            record_matches(manifest["implementation"])
            and record_matches(manifest["implementation_dependency"])
        ),
        {
            "implementation": manifest.get("implementation", {}).get(
                "sha256"
            ),
            "dependency": manifest.get(
                "implementation_dependency", {}
            ).get("sha256"),
        },
    )

    validation = {
        "schema_version": "1.0.0",
        "record_type": "hcp379_hroi_release_review_validation",
        "status": "PASS" if not failures else "FAIL",
        "generated_utc": utc_now(),
        "manifest": str(manifest_path),
        "passed": sum(int(row["pass"]) for row in checks),
        "total": len(checks),
        "failures": failures,
        "checks": checks,
        "human_visual_qc_completed": False,
        "final_release_authorized": False,
        "selected_count_tractography_authorized": False,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(validation, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
