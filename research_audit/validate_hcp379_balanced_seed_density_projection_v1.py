#!/usr/bin/env python3
"""Independently validate the balanced-seed HCP379 density projection."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("/home/ec2-user/exp")
RUNNER = (
    ROOT
    / "scripts/hcp/build_hcp379_balanced_seed_density_projection_v1.py"
)
DEFAULT_OUTPUT = (
    ROOT
    / "research_audit/outputs/"
    "hcp379_balanced_seed_density_projection_v1/validation.json"
)
PYTHON = ROOT / ".venv_connectome_workflow/bin/python"


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
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def load_runner() -> Any:
    specification = importlib.util.spec_from_file_location(
        "hcp379_balanced_density_projection_v1", RUNNER
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("runner import specification is unavailable")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def call_runner(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(PYTHON), str(RUNNER), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    module = load_runner()
    source = RUNNER.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_modules = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported_from_modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    call_names = {
        (
            f"{node.func.value.id}.{node.func.attr}"
            if isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            else node.func.id
            if isinstance(node.func, ast.Name)
            else ""
        )
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
    }

    self_test = module.self_test()
    dry_run = call_runner()
    try:
        dry_value = json.loads(dry_run.stdout)
    except Exception:
        dry_value = {}
    with tempfile.TemporaryDirectory(
        prefix="hcp379-balanced-density-validator-"
    ) as temporary:
        empty_subject_root = Path(temporary) / "empty-subjects"
        empty_subject_root.mkdir()
        refusal = call_runner(
            "--execute",
            "--phase-subject-root",
            str(empty_subject_root),
            "--output-root",
            str(Path(temporary) / "attempts"),
        )
        refusal_created_output = (Path(temporary) / "attempts").exists()

    checks = {
        "runner_exists_and_is_regular": RUNNER.is_file()
        and not RUNNER.is_symlink(),
        "locked_node_and_density_contract": (
            module.EXPECTED_NODES == 379
            and module.DENSITY_TARGET == 0.60
        ),
        "balanced_design_is_exact": module.BALANCED_TOTALS
        == {
            6_000_000: 3_000_000,
            10_000_000: 5_000_000,
            15_000_000: 7_500_000,
            20_000_000: 10_000_000,
        },
        "matrix_reconstruction_self_test_passes": (
            self_test.get("status") == "PASS"
            and all(self_test.get("checks", {}).values())
        ),
        "runner_has_no_external_command_executor": (
            "subprocess" not in imported_modules
            and "subprocess" not in imported_from_modules
            and "os.system" not in call_names
            and "os.popen" not in call_names
        ),
        "dry_run_is_diagnosis_blind_and_non_authorizing": (
            dry_run.returncode == 0
            and dry_value.get("diagnosis_labels_used") is False
            and dry_value.get("scale_up_authorized") is False
            and dry_value.get("unit_count") == 15
            and dry_value.get("design", {}).get(
                "new_tractography_generated"
            )
            is False
            and dry_value.get("design", {}).get(
                "sift2_or_scalar_matrix_generation"
            )
            is False
            and dry_value.get("design", {}).get(
                "per_subject_tuning_allowed"
            )
            is False
        ),
        "incomplete_live_inputs_fail_closed_without_attempt": (
            refusal.returncode == 2 and not refusal_created_output
        ),
        "full_nine_matrix_gate_is_explicit": (
            dry_value.get("design", {}).get(
                "full_nine_matrix_validation_required_before_scale_up"
            )
            is True
            and dry_value.get("design", {}).get(
                "uniform_count_required_across_all_530"
            )
            is True
        ),
    }
    report = {
        "schema_version": "1.0.0",
        "record_type": (
            "hcp379_balanced_seed_density_projection_static_validation"
        ),
        "generated_utc": utc_now(),
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "check_count": len(checks),
        "runner": file_record(RUNNER),
        "validator": file_record(Path(__file__)),
        "live_preflight_status": dry_value.get("status"),
        "live_ready_unit_count": dry_value.get("ready_unit_count"),
        "live_waiting_unit_count": dry_value.get("waiting_unit_count"),
        "self_test": self_test,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_name(output.name + ".partial")
    temporary_output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.chmod(temporary_output, 0o444)
    temporary_output.replace(output)
    print(
        json.dumps(
            {
                "output": str(output),
                "status": report["status"],
                "checks_passed": sum(checks.values()),
                "checks_total": len(checks),
                "live_preflight_status": report["live_preflight_status"],
            },
            sort_keys=True,
        )
    )
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
