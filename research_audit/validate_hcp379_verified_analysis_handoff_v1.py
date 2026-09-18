#!/home/ec2-user/fsl/bin/python
"""Validate the HCP379-v2 verified analysis-handoff package without imaging."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


EXP = Path("/home/ec2-user/exp")
BUILDER = (
    EXP
    / "scripts/hcp/prepare_hcp379_verified_analysis_handoff_v1.py"
)
LOADER = EXP / "connectome_analysis/hcp379_verified_handoff.py"
DASHBOARD = EXP / "apps/connectome_dashboard/connectome_app.py"
REFRESH = (
    EXP
    / "apps/connectome_dashboard/refresh_connectome_dashboard_data.py"
)
ANALYSIS_CONFIG = EXP / "connectome_analysis/analysis_config.py"
PREFLIGHT = (
    EXP
    / "research_audit/outputs/hcp379_verified_analysis_handoff_v1/"
    "preflight.json"
)
OUTPUT = (
    EXP
    / "research_audit/outputs/hcp379_verified_analysis_handoff_v1/"
    "validation.json"
)
REAL_HANDOFF_ROOT = (
    Path("/data/derivatives/hcp379_v2") / "analysis_handoff_v1"
)
PYTHON = Path("/home/ec2-user/fsl/bin/python")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256(resolved),
    }


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
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
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def parser_flags(tree: ast.AST) -> set[str]:
    flags: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
        ):
            for argument in node.args:
                if (
                    isinstance(argument, ast.Constant)
                    and isinstance(argument.value, str)
                    and argument.value.startswith("--")
                ):
                    flags.add(argument.value)
    return flags


def mutation_calls(tree: ast.AST) -> set[str]:
    forbidden = {"copy", "copy2", "copyfile", "symlink", "link"}
    return {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in forbidden
    }


def run() -> dict[str, Any]:
    watched = [DASHBOARD, REFRESH, ANALYSIS_CONFIG]
    before = {str(path): sha256(path) for path in watched}
    compile_result = subprocess.run(
        [
            str(PYTHON),
            "-m",
            "py_compile",
            str(BUILDER),
            str(LOADER),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    selftest_result = subprocess.run(
        [str(PYTHON), str(BUILDER), "--self-test"],
        text=True,
        capture_output=True,
        check=False,
    )
    try:
        selftest = json.loads(selftest_result.stdout)
    except json.JSONDecodeError:
        selftest = {}
    preflight_result = subprocess.run(
        [str(PYTHON), str(BUILDER)],
        text=True,
        capture_output=True,
        check=False,
    )
    preflight = load_json(PREFLIGHT) if PREFLIGHT.is_file() else {}
    after = {str(path): sha256(path) for path in watched}
    builder_tree = ast.parse(BUILDER.read_text(encoding="utf-8"))
    loader_tree = ast.parse(LOADER.read_text(encoding="utf-8"))
    flags = parser_flags(builder_tree)
    mutators = mutation_calls(builder_tree) | mutation_calls(loader_tree)
    real_handoffs = (
        sorted(
            path.name
            for path in REAL_HANDOFF_ROOT.iterdir()
            if path.is_dir() and not path.name.startswith(".")
        )
        if REAL_HANDOFF_ROOT.is_dir()
        else []
    )
    dashboard_source = DASHBOARD.read_text(encoding="utf-8")
    refresh_source = REFRESH.read_text(encoding="utf-8")
    checks = {
        "sources_compile": compile_result.returncode == 0,
        "full_530_by_9_selftest_passes": (
            selftest_result.returncode == 0
            and selftest.get("status") == "PASS"
            and selftest.get("passed") == 8
            and selftest.get("total") == 8
        ),
        "terminal_receipt_is_release_bound": (
            selftest.get("checks", {}).get(
                "terminal_receipt_is_release_bound"
            )
            is True
        ),
        "post_verification_matrix_tamper_is_rejected": (
            selftest.get("checks", {}).get(
                "post_verification_matrix_tamper_fails"
            )
            is True
        ),
        "diagnosis_field_is_rejected": (
            selftest.get("checks", {}).get("diagnosis_field_fails")
            is True
        ),
        "existing_immutable_handoff_is_not_overwritten": (
            selftest.get("checks", {}).get("existing_handoff_refused")
            is True
            and "--overwrite" not in flags
        ),
        "consumer_loader_replays_full_index": (
            selftest.get("checks", {}).get("consumer_loader_replays")
            is True
        ),
        "no_copy_link_or_relabel_primitive": not mutators,
        "no_activation_cli": "--activate" not in flags,
        "default_preflight_is_fail_closed": (
            preflight_result.returncode == 0
            and preflight.get("status") == "AWAITING_VERIFIED_RELEASE"
            and preflight.get("handoff_created") is False
            and preflight.get("matrix_rehash_executed") is False
        ),
        "no_real_handoff_claimed_before_release": not real_handoffs,
        "legacy_loader_mismatch_is_evidence_backed": (
            "SC_AAL166_" in dashboard_source
            and "SC_AAL166_" in refresh_source
            and preflight.get("compatibility", {}).get(
                "legacy_dashboard_compatible"
            )
            is False
            and preflight.get("compatibility", {}).get(
                "live_dashboard_activation_authorized"
            )
            is False
        ),
        "live_dashboard_sources_unchanged": before == after,
    }
    passed = sum(checks.values())
    result = {
        "schema_version": "1.0.0",
        "record_type": "hcp379_v2_verified_analysis_handoff_validation",
        "status": "PASS" if passed == len(checks) else "FAIL",
        "generated_utc": utc_now(),
        "passed": passed,
        "total": len(checks),
        "checks": checks,
        "imaging_executed": False,
        "handoff_created": False,
        "live_dashboard_modified": False,
        "real_handoff_directories": real_handoffs,
        "selftest": selftest,
        "preflight": preflight,
        "records": {
            "builder": file_record(BUILDER),
            "loader": file_record(LOADER),
            "preflight": file_record(PREFLIGHT),
            "dashboard": file_record(DASHBOARD),
            "dashboard_refresh": file_record(REFRESH),
            "analysis_config": file_record(ANALYSIS_CONFIG),
            "validator": file_record(Path(__file__)),
        },
    }
    atomic_json(OUTPUT, result)
    return result


def main() -> int:
    result = run()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
