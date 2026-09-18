#!/usr/bin/env python3
"""Statically validate the Phase-B run-record single-path binding fix."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


SOURCE = Path(
    "/home/ec2-user/exp/scforge/workflow/extensions/"
    "h04a_r1_retry4_phase_b_hcp379/07_hcp379_stability.smk"
)
OUTPUT = Path(
    "/home/ec2-user/exp/research_audit/outputs/"
    "hcp379_phase_b_run_record_binding_v1/validation.json"
)
EXPECTED_NAMES = {
    "tracks",
    "metadata",
    "weights",
    "assignments",
    "count_matrix",
    "fd_sum",
    "count_invnodevol",
    "len_mean",
    "invlen_mean",
    "fa_mean",
    "md_mean",
    "rd_mean",
    "ad_mean",
    "source_parcellation",
    "corrected_nodes",
    "node_volumes",
    "source_node_volumes",
    "atlas_qc",
    "visual_overlay",
    "atlas_result",
    "corrected_atlas_gate",
    "t1_to_b0",
    "b0_reference",
}
MATRIX_INPUT_NAMES = {
    "count_matrix",
    "fd_sum",
    "count_invnodevol",
    "len_mean",
    "invlen_mean",
    "fa_mean",
    "md_mean",
    "rd_mean",
    "ad_mean",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def single_path(value: object) -> Path:
    while not isinstance(value, (str, Path)):
        values = list(value)  # type: ignore[arg-type]
        if len(values) != 1:
            raise ValueError(len(values))
        value = values[0]
    return Path(str(value)).resolve()


def rejects(value: object) -> bool:
    try:
        single_path(value)
    except ValueError:
        return True
    return False


def main() -> int:
    text = SOURCE.read_text(encoding="utf-8")
    start = text.index("rule hcp379_stability_run_record:")
    try:
        end = text.index("\n\nrule ", start + 1)
    except ValueError:
        end = len(text)
    block = text[start:end]
    used_names = set()
    marker = '_single_input_path("'
    offset = 0
    while True:
        found = block.find(marker, offset)
        if found < 0:
            break
        name_start = found + len(marker)
        name_end = block.index('"', name_start)
        used_names.add(block[name_start:name_end])
        offset = name_end + 1
    expected = Path("/tmp/hcp379-binding-probe").resolve()
    checks = {
        "run_record_rule_present": bool(block),
        "single_path_helper_present": (
            "def _single_input_path(name):" in block
        ),
        "recursive_singleton_unwrap_present": (
            "while not isinstance(value, (str, Path)):" in block
        ),
        "no_direct_path_input_conversion_in_rule": (
            "Path(input." not in block
        ),
        "all_named_inputs_use_helper": (
            (EXPECTED_NAMES - MATRIX_INPUT_NAMES) <= used_names
            and '"count_matrix" if name == "count" else name' in block
            and "for name in HCP379_MATRIX_NAMES" in block
        ),
        "string_resolves": (
            single_path("/tmp/hcp379-binding-probe") == expected
        ),
        "one_level_singleton_resolves": (
            single_path(["/tmp/hcp379-binding-probe"]) == expected
        ),
        "nested_singleton_resolves": (
            single_path([[["/tmp/hcp379-binding-probe"]]]) == expected
        ),
        "empty_rejected": rejects([]),
        "multiple_rejected": rejects(["a", "b"]),
    }
    result = {
        "schema_version": "1.0.0",
        "record_type": "hcp379_phase_b_run_record_binding_validation",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "generated_utc": utc_now(),
        "checks": checks,
        "used_named_inputs": sorted(used_names),
        "missing_direct_named_inputs": sorted(
            (EXPECTED_NAMES - MATRIX_INPUT_NAMES) - used_names
        ),
        "matrix_named_inputs_bound_dynamically": sorted(
            MATRIX_INPUT_NAMES
        ),
        "source": {
            "path": str(SOURCE.resolve()),
            "size_bytes": SOURCE.stat().st_size,
            "sha256": sha256(SOURCE),
        },
        "imaging_executed": False,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(".json.partial")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(OUTPUT)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
