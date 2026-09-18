#!/usr/bin/env python3
"""Independently validate the readable exact-input HCP379 V6 review pack."""

from __future__ import annotations

import csv
import hashlib
import json
import py_compile
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from PIL import Image


ROOT = Path("/data/derivatives/hcp379_v2")
EXP = Path("/home/ec2-user/exp")
TARGET = "009_S_4324_I1186579"
ROUTE = "bbr_initialized_syn_conservative"
V5_MANIFEST = (
    ROOT
    / "review_recovery4_candidate_overlay_v5/"
    "hcp379_visual_review_pack_recovery4_candidate_v5_manifest.json"
)
V6_ROOT = ROOT / "review_recovery4_candidate_overlay_v6"
V6_MANIFEST = (
    V6_ROOT
    / "hcp379_visual_review_pack_recovery4_candidate_v6_manifest.json"
)
TEMPLATE = (
    V6_ROOT / "human_visual_qc_recovery4_candidate_v6_template.csv"
)
PDF = (
    V6_ROOT / "hcp379_recovery4_v6_exact_readable_blinded_review.pdf"
)
INSTRUCTIONS = V6_ROOT / "REVIEW_INSTRUCTIONS.md"
BUILDER = (
    EXP
    / "scripts/hcp/"
    "build_hcp379_visual_review_pack_recovery4_v6.py"
)
OUTPUT = (
    EXP
    / "research_audit/outputs/hcp379_visual_review_recovery4_v6/"
    "validation.json"
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
        "sha256": sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root is not an object: {path}")
    return value


def verify_record(record: Mapping[str, Any]) -> bool:
    try:
        return file_record(Path(str(record.get("path", "")))) == dict(record)
    except Exception:
        return False


def role_records(row: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for label, record in row.get("source_panels", {}).items():
        lowered = str(label).lower()
        if "hcp379" in lowered:
            role = "hcp"
        elif "5tt" in lowered:
            role = "5tt"
        elif "t1" in lowered:
            role = "t1"
        else:
            raise ValueError(f"unrecognized panel label: {label}")
        if role in result:
            raise ValueError(f"duplicate panel role: {role}")
        result[role] = dict(record)
    return result


def main() -> int:
    v5 = load_json(V5_MANIFEST)
    v6 = load_json(V6_MANIFEST)
    checks: dict[str, dict[str, Any]] = {}

    def check(name: str, passed: bool, evidence: Any) -> None:
        checks[name] = {
            "status": "PASS" if passed else "FAIL",
            "evidence": evidence,
        }

    v5_by_unit = {
        str(row["unit"]): row for row in v5.get("units", [])
    }
    v6_by_unit = {
        str(row["unit"]): row for row in v6.get("units", [])
    }
    check(
        "v6_manifest_contract",
        v6.get("record_type")
        == (
            "diagnosis_blind_hcp379_visual_review_pack_"
            "recovery4_candidate_v6"
        )
        and v6.get("status") == "READY_FOR_EXACT_INPUT_HUMAN_REVIEW"
        and v6.get("diagnosis_labels_used") is False
        and v6.get("human_visual_qc_inferred") is False
        and v6.get("tractography_started") is False
        and v6.get("matrix_generation_started") is False
        and v6.get("unit_count") == 15
        and v6.get("route_substitution", {}).get("unit") == TARGET
        and v6.get("route_substitution", {}).get("candidate_route")
        == ROUTE
        and v6.get("route_substitution", {}).get("promotion_status")
        == "NOT_PROMOTED",
        {
            "record_type": v6.get("record_type"),
            "status": v6.get("status"),
            "unit_count": v6.get("unit_count"),
            "route_substitution": v6.get("route_substitution"),
        },
    )
    check(
        "exact_v5_unit_and_pretract_binding",
        set(v6_by_unit) == set(v5_by_unit)
        and len(v6_by_unit) == 15
        and all(
            v6_by_unit[unit].get("pretract_manifest")
            == v5_by_unit[unit].get("pretract_manifest")
            for unit in v5_by_unit
        ),
        {
            "v5_units": len(v5_by_unit),
            "v6_units": len(v6_by_unit),
            "sets_equal": set(v6_by_unit) == set(v5_by_unit),
        },
    )
    panel_failures: list[str] = []
    for unit in sorted(v5_by_unit):
        try:
            if role_records(v5_by_unit[unit]) != role_records(
                v6_by_unit[unit]
            ):
                panel_failures.append(unit)
        except Exception as exc:
            panel_failures.append(f"{unit}:{exc}")
    check(
        "all_source_panels_are_byte_identical_to_v5",
        not panel_failures,
        {"failures": panel_failures, "units_checked": len(v5_by_unit)},
    )

    stale_records: list[str] = []
    for unit, row in v6_by_unit.items():
        if not verify_record(row.get("pretract_manifest", {})):
            stale_records.append(f"{unit}:pretract_manifest")
        if not verify_record(row.get("review_page", {})):
            stale_records.append(f"{unit}:review_page")
        for role, record in role_records(row).items():
            if not verify_record(record):
                stale_records.append(f"{unit}:source_panel:{role}")
    for name in (
        "candidate_pretract_master",
        "supersedes_review_pack",
        "round1_transcription",
        "review_pdf",
        "human_qc_template",
        "review_instructions",
        "builder",
    ):
        if not verify_record(v6.get(name, {})):
            stale_records.append(name)
    check(
        "all_v6_file_records_are_current",
        not stale_records,
        {"stale_records": stale_records},
    )

    page_failures: list[str] = []
    dimensions: set[tuple[int, int]] = set()
    for unit, row in v6_by_unit.items():
        try:
            path = Path(str(row["review_page"]["path"]))
            with Image.open(path) as image:
                dimensions.add(image.size)
                image.verify()
        except Exception as exc:
            page_failures.append(f"{unit}:{exc}")
    try:
        pdf_payload = PDF.read_bytes()
        if not (
            pdf_payload.startswith(b"%PDF-")
            and b"%%EOF" in pdf_payload[-4096:]
        ):
            raise ValueError("PDF header/trailer differs")
        pdf_pages = len(
            re.findall(rb"/Type\s*/Page\b", pdf_payload)
        )
    except Exception as exc:
        pdf_pages = -1
        page_failures.append(f"PDF:{exc}")
    check(
        "fifteen_readable_pages_and_pdf",
        not page_failures
        and pdf_pages == 15
        and dimensions == {(2280, 834)},
        {
            "pdf_pages": pdf_pages,
            "page_dimensions": sorted(dimensions),
            "failures": page_failures,
        },
    )

    with TEMPLATE.open(newline="", encoding="utf-8-sig") as handle:
        template_rows = list(csv.DictReader(handle))
    template_by_unit = {
        str(row.get("unit")): row for row in template_rows
    }
    unchanged = set(v6_by_unit) - {TARGET}
    check(
        "blank_fail_closed_panelwise_template",
        set(template_by_unit) == set(v6_by_unit)
        and len(template_rows) == 15
        and all(
            template_by_unit[unit].get("B0 vsT1") == "PASS"
            and template_by_unit[unit].get("B0 vs 5TT") == "PASS"
            and not template_by_unit[unit].get("B0 vs HCP379")
            and not template_by_unit[unit].get("reviewer")
            and not template_by_unit[unit].get("reviewed_utc")
            for unit in unchanged
        )
        and all(
            not template_by_unit[TARGET].get(field)
            for field in (
                "B0 vsT1",
                "B0 vs 5TT",
                "B0 vs HCP379",
                "reviewer",
                "reviewed_utc",
            )
        ),
        {
            "rows": len(template_rows),
            "unchanged_units_with_carried_t1_5tt": len(unchanged),
            "target_all_three_blank": TARGET in template_by_unit,
        },
    )

    instructions = INSTRUCTIONS.read_text(encoding="utf-8")
    layout = v6.get("visual_layout_correction", {})
    check(
        "layout_guidance_is_explicit_and_non_imaging",
        layout.get("short_nonoverlapping_panel_titles") is True
        and layout.get(
            "montage_orientation_and_slice_guidance_added"
        )
        is True
        and layout.get("source_panel_artifacts_changed") is False
        and layout.get("imaging_derivatives_changed") is False
        and layout.get("prior_hcp379_status_carried") is False
        and "not left/right hemisphere columns" in instructions
        and "REVIEW remains unresolved" in instructions,
        {"layout": layout},
    )

    compile_error = None
    try:
        py_compile.compile(str(BUILDER), doraise=True)
    except Exception as exc:
        compile_error = str(exc)
    check(
        "builder_compiles",
        compile_error is None,
        {"error": compile_error},
    )

    target = v6_by_unit.get(TARGET, {})
    check(
        "009_exact_candidate_route_is_bound",
        target.get("registration_route") == ROUTE
        and target.get("exact_candidate_inputs")
        == v5_by_unit.get(TARGET, {}).get("exact_candidate_inputs")
        and target.get("diagnosis_labels_used") is False
        and target.get("human_visual_qc_inferred") is False,
        {
            "route": target.get("registration_route"),
            "exact_candidate_inputs_present": bool(
                target.get("exact_candidate_inputs")
            ),
        },
    )

    passed = sum(
        row["status"] == "PASS" for row in checks.values()
    )
    result = {
        "schema_version": "1.0.0",
        "record_type": (
            "hcp379_visual_review_recovery4_v6_validation"
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
            "v5_manifest": file_record(V5_MANIFEST),
            "v6_manifest": file_record(V6_MANIFEST),
            "builder": file_record(BUILDER),
            "validator": file_record(Path(__file__)),
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
