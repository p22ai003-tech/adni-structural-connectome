#!/usr/bin/env python3
"""Record the exact agent visual triage of the corrected-ACT H-ROI pages.

This is deliberately not represented as human visual QC and cannot satisfy a
final-release human gate.  It may support bounded operational pretract use
when combined with the quantitative affected- and complete-source canaries.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


REVIEW_MANIFEST = Path(
    "/data/derivatives/hcp379_v2/source_label_repair_v1/review/"
    "hroi_corrected_act_v1/attempts/"
    "20260728T062728.490070Z-hroi-corrected-act-review/"
    "review_manifest.json"
)
OUTPUT = REVIEW_MANIFEST.parent / "agent_visual_triage.json"
DECISIONS = {
    "002_S_0413_I863064": {
        "decision": "PASS_AGENT_VISUAL_TRIAGE",
        "observations": [
            "bilateral H support is confined to the medial temporal region",
            "no gross extracranial, contralateral, or distant cortical island is visible",
            "the repaired left parcel remains spatially coherent across sagittal, coronal, and axial centroid views",
        ],
    },
    "135_S_6840_I1263427": {
        "decision": "PASS_AGENT_VISUAL_TRIAGE",
        "observations": [
            "bilateral H support is confined to the medial temporal region",
            "no gross extracranial, contralateral, or distant cortical island is visible",
            "the broader left DWI support is contiguous on the displayed views and agrees with the larger left node voxel count",
        ],
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"not a regular file: {path}")
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def verify(record: Mapping[str, Any]) -> None:
    path = Path(str(record.get("path", ""))).resolve()
    if file_record(path) != dict(record):
        raise ValueError(f"review artifact binding differs: {path}")


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
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
    os.chmod(temporary, 0o444)
    os.replace(temporary, path)


def main() -> int:
    review = json.loads(REVIEW_MANIFEST.read_text(encoding="utf-8"))
    units = {str(row.get("unit")): row for row in review.get("units", [])}
    if (
        review.get("record_type")
        != "diagnosis_blind_hcp379_hroi_corrected_act_visual_review"
        or review.get("status") != "READY_FOR_VISUAL_QC"
        or review.get("diagnosis_labels_used") is not False
        or set(units) != set(DECISIONS)
    ):
        raise ValueError("corrected-ACT review manifest differs")
    verify(review["pdf"])
    for row in units.values():
        verify(row["png"])
        verify(row["nodes"])
        verify(row["validation"])

    record = {
        "schema_version": "1.0.0",
        "record_type": (
            "diagnosis_blind_hcp379_hroi_agent_visual_triage"
        ),
        "generated_utc": utc_now(),
        "status": "PASS_AGENT_VISUAL_TRIAGE_NOT_HUMAN_QC",
        "diagnosis_labels_used": False,
        "reviewer_type": "codex_agent_visual_inspection",
        "human_visual_qc_inferred": False,
        "final_release_human_qc_satisfied": False,
        "operational_pretract_adoption_supported": True,
        "scope": (
            "gross spatial plausibility of the H-ROI support on one genuinely "
            "affected and one complete-source corrected-ACT case"
        ),
        "limitations": [
            "centroid orthogonal views are not full volumetric manual segmentation review",
            "this record cannot satisfy a final-release human visual-QC gate",
        ],
        "decisions": [
            {"unit": unit, **DECISIONS[unit]} for unit in sorted(DECISIONS)
        ],
        "review_manifest": file_record(REVIEW_MANIFEST),
        "review_pdf": review["pdf"],
        "implementation": file_record(Path(__file__)),
    }
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    atomic_json(OUTPUT, record)
    print(
        json.dumps(
            {
                "output": str(OUTPUT),
                "status": record["status"],
                "final_release_human_qc_satisfied": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
