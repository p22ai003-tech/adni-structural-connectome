#!/home/ec2-user/fsl/bin/python
"""Record explicit human approval of the exact 16-page final-HROI pack.

The helper never infers approval from an earlier review or chat statement.  It
requires the exact attestation text, verifies the immutable manifest, PDF,
template, and all 16 PNG identities, then writes the completed CSV atomically
without replacing an existing decision.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(
    "/data/derivatives/hcp379_v2/source_label_repair_v1/"
    "review/hroi_release_v1/attempts/"
    "20260728T083614.440650Z-hroi-release-review"
)
MANIFEST = ROOT / "review_manifest.json"
PDF = ROOT / "hcp379_hroi_release_review.pdf"
TEMPLATE = ROOT / "human_visual_qc_hroi_release_template.csv"
COMPLETED = ROOT / "human_visual_qc_hroi_release_completed.csv"
ATTESTATION_RECORD = (
    ROOT / "human_visual_qc_hroi_release_attestation.json"
)
PREFLIGHT_OUTPUT = Path(
    "/home/ec2-user/exp/research_audit/outputs/"
    "hcp379_hroi_release_human_attestation_helper_v1/preflight.json"
)
EXACT_ATTESTATION = "Final HROI 16/16 PASS"
EXPECTED_PDF_SHA256 = (
    "90e36df6f817e05a23272a20e02fd11690ca137e33bfc253a86892ccb1c9e209"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


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


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return [
            {
                str(key): str(value or "").strip()
                for key, value in row.items()
            }
            for row in csv.DictReader(handle)
        ]


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
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


def atomic_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fields = [
        "unit",
        "source_support_status",
        "dwi_atlas_status",
        "overall_status",
        "reviewer",
        "reviewed_utc",
        "note",
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


def preflight() -> tuple[dict[str, Any], list[str]]:
    manifest = load_json(MANIFEST)
    units = manifest.get("units", [])
    manifest_units = [
        str(row.get("unit", ""))
        for row in units
        if isinstance(row, Mapping)
    ]
    template = read_csv(TEMPLATE)
    template_units = [row.get("unit", "") for row in template]
    png_records: list[dict[str, Any]] = []
    for unit in manifest_units:
        png = ROOT / f"{unit}_hroi_release_review.png"
        png_records.append(record(png))
    if (
        manifest.get("record_type")
        != "diagnosis_blind_hcp379_hroi_release_visual_review"
        or manifest.get("status") != "READY_FOR_HUMAN_VISUAL_QC"
        or manifest.get("diagnosis_labels_used") is not False
        or manifest.get("human_visual_qc_inferred") is not False
        or manifest.get("unit_n") != 16
        or len(manifest_units) != 16
        or len(set(manifest_units)) != 16
        or len(template_units) != 16
        or set(template_units) != set(manifest_units)
        or sha256(PDF) != EXPECTED_PDF_SHA256
    ):
        raise ValueError("exact final-HROI review package differs")
    payload = {
        "schema_version": "1.0.0",
        "record_type": "hcp379_hroi_release_human_attestation_preflight",
        "status": (
            "ALREADY_RECORDED"
            if COMPLETED.is_file()
            else "READY_FOR_EXPLICIT_HUMAN_ATTESTATION"
        ),
        "generated_utc": utc_now(),
        "required_attestation": EXACT_ATTESTATION,
        "unit_n": len(manifest_units),
        "units": manifest_units,
        "diagnosis_labels_used": False,
        "human_visual_qc_inferred": False,
        "imaging_executed": False,
        "records": {
            "manifest": record(MANIFEST),
            "pdf": record(PDF),
            "template": record(TEMPLATE),
            "pngs": png_records,
            "implementation": record(Path(__file__)),
        },
    }
    return payload, manifest_units


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--reviewer", default="")
    parser.add_argument("--attestation", default="")
    args = parser.parse_args()
    pre, units = preflight()
    if not args.execute:
        PREFLIGHT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(PREFLIGHT_OUTPUT, pre)
        print(json.dumps(pre, indent=2, sort_keys=True))
        return 0
    if COMPLETED.exists() or ATTESTATION_RECORD.exists():
        raise FileExistsError("final-HROI attestation already exists")
    reviewer = args.reviewer.strip()
    attestation = args.attestation.strip()
    if not reviewer:
        raise ValueError("a non-empty human reviewer identity is required")
    if attestation != EXACT_ATTESTATION:
        raise ValueError(
            f"exact attestation required: {EXACT_ATTESTATION!r}"
        )
    reviewed = utc_now()
    rows = [
        {
            "unit": unit,
            "source_support_status": "PASS",
            "dwi_atlas_status": "PASS",
            "overall_status": "PASS",
            "reviewer": reviewer,
            "reviewed_utc": reviewed,
            "note": (
                f"Explicit attestation: {EXACT_ATTESTATION}; "
                f"PDF sha256={EXPECTED_PDF_SHA256}"
            ),
        }
        for unit in units
    ]
    atomic_csv(COMPLETED, rows)
    payload = {
        "schema_version": "1.0.0",
        "record_type": "hcp379_hroi_release_human_attestation",
        "status": "PASS_EXPLICIT_HUMAN_ATTESTATION_RECORDED",
        "generated_utc": utc_now(),
        "reviewer": reviewer,
        "reviewed_utc": reviewed,
        "attestation": attestation,
        "unit_n": len(rows),
        "diagnosis_labels_used": False,
        "human_visual_qc_inferred": False,
        "imaging_executed": False,
        "records": {
            **pre["records"],
            "completed_review": record(COMPLETED),
        },
    }
    atomic_json(ATTESTATION_RECORD, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
