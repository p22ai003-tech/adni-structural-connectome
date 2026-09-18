#!/usr/bin/env python3
"""Build an aggregate, fail-closed Phase-A terminal and GPU-resize gate record."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT = Path("/home/ec2-user/exp")
sys.path.insert(0, str(PROJECT / "scforge"))

from scforge.response_calibration import (  # noqa: E402
    validate_frozen_response_calibration,
    validate_response_calibration_outcome,
    verify_file_record,
)


DEFAULT_RUN_ROOT = Path(
    "/data/derivatives/scforge_v2/h04a_r1_recovery_20260718_retry2"
)
DEFAULT_DECISION = Path(
    "/home/ec2-user/exp/research_audit/outputs/h04a_r1_recovery_package_v1/"
    "recovery_execution_decision_retry2.json"
)
DEFAULT_GPU_BUILD = Path(
    "/home/ec2-user/exp/research_audit/outputs/"
    "gpu_eddy_equivalence_package_v1_build_record.json"
)
DEFAULT_OUTPUT_VALIDATION = Path(
    "/home/ec2-user/exp/research_audit/outputs/refined_output_release_v1/"
    "independent_validation.json"
)
DEFAULT_RETRY3_PACKAGE_VALIDATION = Path(
    "/home/ec2-user/exp/research_audit/outputs/"
    "h04a_r1_retry3_package_v2/retry3_package_validation.json"
)
DEFAULT_RETRY3_DRYRUN_VALIDATION = Path(
    "/home/ec2-user/exp/research_audit/outputs/"
    "h04a_r1_retry3_dryrun_v5/validation.json"
)
DEFAULT_OUTPUT_DIR = Path(
    "/home/ec2-user/exp/research_audit/outputs/gpu_transition_gate_v1"
)
KNOWN_TERMINAL_SERVICE_STATES = {"inactive", "failed"}
EXPECTED_RETRY3_CONTINUATION_RULES = {
    "response_calibration_phase_a",
    "select_fod_shells",
    "subject_response",
}
EXPECTED_RESIZE_PREREQUISITE_COUNT = 13
IMAGING_COMMANDS = {
    "eddy_cpu",
    "eddy_cuda11.0",
    "dwifslpreproc",
    "snakemake",
    "dwi2response",
    "dwibiascorrect",
    "dwi2mask",
    "dwiextract",
    "mrmath",
    "responsemean",
}
IMAGING_ARGUMENT_TOKENS = {
    "run_h04a_r1_recovery.py",
    "run_connectome_v2.py",
    "Snakefile_h04a_r1",
    "snakemake",
    "dwifslpreproc",
    "eddy_cpu",
    "eddy_cuda11.0",
    "dwi2response",
    "dwibiascorrect",
    "dwi2mask",
    "dwiextract",
    "responsemean",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def service_state(service: str) -> str:
    result = subprocess.run(
        ["systemctl", "--user", "is-active", service],
        text=True,
        capture_output=True,
        check=False,
    )
    return (result.stdout or result.stderr).strip() or "unknown"


def active_imaging_processes(run_root: Path) -> list[dict[str, object]]:
    result = subprocess.run(
        ["ps", "-eo", "pid=,comm=,args="],
        text=True,
        capture_output=True,
        check=True,
    )
    found: list[dict[str, object]] = []
    run_token = str(run_root)
    for line in result.stdout.splitlines():
        fields = line.strip().split(None, 2)
        if len(fields) < 2:
            continue
        pid_text, command = fields[:2]
        arguments = fields[2] if len(fields) == 3 else ""
        command_name = Path(command).name
        direct_imaging_command = command_name in IMAGING_COMMANDS
        scoped_wrapper = (
            run_token in arguments
            and any(token in arguments for token in IMAGING_ARGUMENT_TOKENS)
        )
        if command_name.startswith("eddy_") or direct_imaging_command or scoped_wrapper:
            found.append({
                "pid": int(pid_text),
                "command": command_name,
                "scoped_wrapper": scoped_wrapper,
            })
    return found


def add(
    checks: list[dict[str, object]],
    name: str,
    passed: bool,
    detail: object,
    *,
    category: str = "resize_prerequisite",
) -> None:
    checks.append(
        {
            "name": name,
            "pass": bool(passed),
            "detail": detail,
            "category": category,
        }
    )


def downstream_artifacts(run_root: Path) -> list[str]:
    patterns = (
        "subjects/*/05_model/wmfod.mif",
        "subjects/*/05_model/wmfod_norm.mif",
        "subjects/*/07_tractography/*.tck",
        "subjects/*/07_tractography/sift2_weights.txt",
        "subjects/*/07_connectome/matrices/*.csv",
        "publication/analysis_ready_manifest.csv",
    )
    hits: list[str] = []
    for pattern in patterns:
        hits.extend(str(path.relative_to(run_root)) for path in run_root.glob(pattern))
    return sorted(hits)


def build_record(
    *,
    run_root: Path,
    decision_path: Path,
    gpu_build_path: Path,
    output_validation_path: Path,
    retry3_package_validation_path: Path,
    retry3_dryrun_validation_path: Path,
    service: str,
) -> dict[str, object]:
    checks: list[dict[str, object]] = []
    decision = load_json(decision_path)
    expected_units = decision.get("units", [])
    minimum_valid = int(decision["minimum_valid_response_calibration_units"])
    add(checks, "decision_is_exact_15_unit_phase_a_scope",
        isinstance(expected_units, list) and len(expected_units) == 15 and len(set(expected_units)) == 15,
        {"expected_units": len(expected_units), "minimum_valid": minimum_valid})
    add(checks, "decision_prohibits_downstream_execution",
        decision.get("fod_reconstruction_authorized") is False
        and decision.get("tractography_authorized") is False
        and decision.get("matrix_generation_authorized") is False
        and decision.get("full_cohort_authorized") is False,
        "FOD, tractography, matrices and full cohort all false")

    state = service_state(service)
    processes = active_imaging_processes(run_root)
    service_terminal = state in KNOWN_TERMINAL_SERVICE_STATES
    add(checks, "phase_a_service_is_known_terminal", service_terminal, state)
    add(checks, "no_active_imaging_processes", not processes,
        {"active_count": len(processes), "commands": sorted({row["command"] for row in processes})})

    completion_path = run_root / "publication" / "response_calibration_phase_a_completion.json"
    manifest_path = run_root / "publication" / "response_calibration_phase_a_manifest.json"
    completion_valid = False
    terminal_exact = False
    outcome_validation_pass = False
    exact_attempt_closure_valid = False
    pass_count = 0
    fail_count = 0
    completion_status = "MISSING"
    terminal_error: str | None = None
    if completion_path.is_file() and manifest_path.is_file():
        try:
            completion = load_json(completion_path)
            manifest = load_json(manifest_path)
            verify_file_record(
                completion.get("phase_a_manifest", {}),
                label="phase-A manifest",
                expected_path=manifest_path,
            )
            summary = completion.get("phase_a_summary", {})
            completion_status = str(completion.get("status", ""))
            pass_count = int(summary.get("pass", -1))
            fail_count = int(summary.get("fail", -1))
            terminal_count = int(summary.get("terminal", -1))
            expected_count = int(summary.get("expected", -1))
            completion_valid = (
                completion.get("record_type") == "response_calibration_phase_a_completion"
                and completion.get("mode") == "response-calibration-phase-a"
                and completion_status in {"PASS", "PARTIAL", "FAIL"}
                and completion.get("production_overwrite") is False
                and completion.get("full_run_terminal_publication") is False
            )
            terminal_exact = (
                expected_count == terminal_count == len(expected_units)
                and pass_count + fail_count == terminal_count
                and manifest.get("expected_units") == expected_units
                and manifest.get("summary") == summary
            )
            outcomes = manifest.get("outcome_records", {})
            if set(outcomes) != set(expected_units):
                raise ValueError("terminal outcome set differs from approved units")
            for unit in expected_units:
                outcome_path = verify_file_record(
                    outcomes[unit], label="phase-A outcome"
                )
                outcome = load_json(outcome_path)
                validate_response_calibration_outcome(
                    outcome,
                    expected_unit=unit,
                    expected_recipe_id=str(completion["recipe_id"]),
                    expected_run_id=str(completion["run_id"]),
                )
            attempt_start_paths = sorted((run_root / "attempts").glob("*.start.json"))
            attempt_end_paths = sorted((run_root / "attempts").glob("*.end.json"))
            if len(attempt_start_paths) != 1 or len(attempt_end_paths) != 1:
                raise ValueError(
                    "Phase A requires exactly one immutable start/end pair; "
                    f"found {len(attempt_start_paths)} starts and {len(attempt_end_paths)} ends"
                )
            attempt_start_path = verify_file_record(
                completion.get("terminal_attempt_start", {}),
                label="terminal attempt start",
                expected_path=attempt_start_paths[0],
            )
            attempt_end_path = verify_file_record(
                completion.get("terminal_attempt_end", {}),
                label="terminal attempt end",
                expected_path=attempt_end_paths[0],
            )
            attempt_start = load_json(attempt_start_path)
            attempt_end = load_json(attempt_end_path)
            attempt_id = str(attempt_start.get("attempt_id", ""))
            returncode = attempt_end.get("returncode")
            if (
                not attempt_id
                or attempt_end.get("attempt_id") != attempt_id
                or attempt_end.get("launcher_mode") != "response-calibration-phase-a"
                or attempt_end.get("attempt_start") != completion.get("terminal_attempt_start")
                or attempt_end.get("combined_execution_log")
                != completion.get("combined_execution_log")
                or returncode != completion.get("snakemake_returncode")
                or attempt_end.get("status")
                != ("PASS" if returncode == 0 else "FAIL")
            ):
                raise ValueError("terminal start, end and completion records are inconsistent")
            exact_attempt_closure_valid = True
            outcome_validation_pass = True
        except Exception as exc:  # fail-closed evidence record
            terminal_error = f"{type(exc).__name__}: {exc}"
    add(checks, "phase_a_completion_and_manifest_exist",
        completion_path.is_file() and manifest_path.is_file(),
        {"completion": completion_path.is_file(), "manifest": manifest_path.is_file()})
    add(checks, "phase_a_completion_contract_valid", completion_valid,
        {"status": completion_status, "error": terminal_error})
    add(checks, "all_15_units_have_exact_terminal_outcomes", terminal_exact,
        {"pass": pass_count, "fail": fail_count, "expected": len(expected_units)})
    add(checks, "terminal_outcome_records_and_hashes_validate", outcome_validation_pass,
        terminal_error or (
            "all outcome records replay"
            if outcome_validation_pass
            else "terminal publication not yet available"
        ))
    add(
        checks,
        "exact_single_attempt_start_end_completion_closure",
        exact_attempt_closure_valid,
        terminal_error
        or (
            "one immutable start/end pair is bound by the completion record"
            if exact_attempt_closure_valid
            else "terminal attempt closure is not yet available"
        ),
    )

    frozen_path = run_root / "frozen_calibration" / "frozen_response_calibration_manifest.json"
    frozen_valid = False
    frozen_error: str | None = None
    frozen_valid_count = 0
    frozen_diversity: dict[str, Any] = {}
    if frozen_path.is_file():
        try:
            frozen = load_json(frozen_path)
            frozen_valid_count = int(frozen.get("valid_subject_count", 0))
            frozen_diversity = dict(frozen.get("valid_pool_technical_diversity", {}))
            validate_frozen_response_calibration(
                frozen_path,
                expected_manifest_sha256=sha256(frozen_path),
                minimum_valid_subjects=minimum_valid,
                expected_responses=frozen.get("pooled_responses", {}),
                expected_technical_diversity=frozen_diversity,
            )
            frozen_valid = True
        except Exception as exc:
            frozen_error = f"{type(exc).__name__}: {exc}"
    add(
        checks,
        "frozen_response_calibration_valid",
        frozen_valid,
        {"exists": frozen_path.is_file(), "valid_units": frozen_valid_count, "error": frozen_error},
        category="scientific_readiness",
    )

    downstream = downstream_artifacts(run_root)
    add(checks, "no_fod_tractography_matrix_or_full_publication_artifact", not downstream, downstream)

    gpu_build = load_json(gpu_build_path)
    gpu_checks = gpu_build.get("runner_static_validation", {}).get("checks", {})
    gpu_package_pass = (
        gpu_build.get("status") == "PASS"
        and gpu_build.get("execution_performed") is False
        and gpu_build.get("resize_performed") is False
        and isinstance(gpu_checks, dict)
        and len(gpu_checks) == 11
        and all(gpu_checks.values())
    )
    add(checks, "gpu_equivalence_package_static_validation_passes", gpu_package_pass,
        {"checks": sum(bool(value) for value in gpu_checks.values()), "total": len(gpu_checks)})

    output_validation = load_json(output_validation_path)
    output_checks = output_validation.get("checks", {})
    output_builder_validation_path = output_validation_path.parent / "validation.json"
    output_validator_path = PROJECT / "research_audit" / "validate_refined_output_release_v1.py"
    builder_validation_hash_matches = (
        output_builder_validation_path.is_file()
        and output_validation.get("builder_validation_sha256")
        == sha256(output_builder_validation_path)
    )
    validator_hash_matches = (
        output_validator_path.is_file()
        and output_validation.get("validator_sha256") == sha256(output_validator_path)
    )
    output_release_pass = (
        output_validation.get("pass") is True
        and isinstance(output_checks, dict)
        and len(output_checks) == 28
        and all(value is True for value in output_checks.values())
        and builder_validation_hash_matches
        and validator_hash_matches
    )
    add(
        checks,
        "refined_output_release_independent_validation_passes",
        output_release_pass,
        {
            "checks": sum(value is True for value in output_checks.values())
            if isinstance(output_checks, dict)
            else 0,
            "total": len(output_checks) if isinstance(output_checks, dict) else 0,
            "builder_validation_hash_matches": builder_validation_hash_matches,
            "validator_hash_matches": validator_hash_matches,
        },
        category="output_readiness",
    )

    retry3_package_pass = False
    retry3_package_detail: object = "missing"
    if retry3_package_validation_path.is_file():
        try:
            retry3_package = load_json(retry3_package_validation_path)
            package_checks = retry3_package.get("checks", {})
            retry3_package_pass = (
                retry3_package.get("status") == "PASS"
                and isinstance(package_checks, dict)
                and len(package_checks) == 15
                and all(value is True for value in package_checks.values())
                and retry3_package.get("summary") == {"passed": 15, "total": 15}
            )
            retry3_package_detail = {
                "status": retry3_package.get("status"),
                "checks": sum(value is True for value in package_checks.values())
                if isinstance(package_checks, dict)
                else 0,
                "total": len(package_checks) if isinstance(package_checks, dict) else 0,
            }
        except Exception as exc:
            retry3_package_detail = f"{type(exc).__name__}: {exc}"
    add(
        checks,
        "retry3_copy_on_write_package_validation_passes",
        retry3_package_pass,
        retry3_package_detail,
    )

    retry3_dryrun_pass = False
    retry3_dryrun_detail: object = "missing"
    if retry3_dryrun_validation_path.is_file():
        try:
            retry3_dryrun = load_json(retry3_dryrun_validation_path)
            dryrun_checks = retry3_dryrun.get("checks", {})
            scheduled_rules = set(retry3_dryrun.get("scheduled_rules", []))
            retry3_dryrun_pass = (
                retry3_dryrun.get("status") == "PASS"
                and isinstance(dryrun_checks, dict)
                and len(dryrun_checks) == 11
                and all(value is True for value in dryrun_checks.values())
                and scheduled_rules == EXPECTED_RETRY3_CONTINUATION_RULES
                and set(retry3_dryrun.get("allowed_rules", []))
                == EXPECTED_RETRY3_CONTINUATION_RULES
                and retry3_dryrun.get("unexpected_upstream_rules") == []
                and retry3_dryrun.get("forbidden_downstream_rules") == []
                and retry3_dryrun.get("missing_expected_continuation_rules") == []
            )
            retry3_dryrun_detail = {
                "status": retry3_dryrun.get("status"),
                "checks": sum(value is True for value in dryrun_checks.values())
                if isinstance(dryrun_checks, dict)
                else 0,
                "total": len(dryrun_checks) if isinstance(dryrun_checks, dict) else 0,
                "scheduled_rules": sorted(scheduled_rules),
            }
        except Exception as exc:
            retry3_dryrun_detail = f"{type(exc).__name__}: {exc}"
    add(
        checks,
        "retry3_continuation_dryrun_is_minimal_and_passes",
        retry3_dryrun_pass,
        retry3_dryrun_detail,
    )

    resize_checks = [
        row for row in checks if row["category"] == "resize_prerequisite"
    ]
    safe_to_resize = (
        len(resize_checks) == EXPECTED_RESIZE_PREREQUISITE_COUNT
        and all(bool(row["pass"]) for row in resize_checks)
    )
    phase_a_scientific_gate_ready = (
        completion_valid
        and terminal_exact
        and outcome_validation_pass
        and pass_count >= minimum_valid
        and frozen_valid
    )
    return {
        "schema_version": "1.0.0",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "service": service,
        "service_state": state,
        "run_root": str(run_root),
        "phase_a_summary": {
            "status": completion_status,
            "expected": len(expected_units),
            "pass": pass_count,
            "fail": fail_count,
            "minimum_valid_required": minimum_valid,
            "frozen_valid_count": frozen_valid_count,
        },
        "safe_to_resize": safe_to_resize,
        "phase_a_scientific_gate_ready": phase_a_scientific_gate_ready,
        "gpu_equivalence_canary_ready_after_resize": safe_to_resize and gpu_package_pass,
        "pre_tractography_canary_ready": phase_a_scientific_gate_ready,
        "tractography_authorized": False,
        "matrix_generation_authorized": False,
        "full_cohort_authorized": False,
        "checks_passed": sum(bool(row["pass"]) for row in checks),
        "checks_total": len(checks),
        "resize_checks_passed": sum(bool(row["pass"]) for row in resize_checks),
        "resize_checks_total": len(resize_checks),
        "expected_resize_checks_total": EXPECTED_RESIZE_PREREQUISITE_COUNT,
        "checks": checks,
        "decision": (
            "SAFE_TO_RESIZE" if safe_to_resize else "NOT_SAFE_TO_RESIZE"
        ),
    }


def render_markdown(record: dict[str, object]) -> str:
    lines = [
        "# GPU transition gate",
        "",
        f"**Generated:** {record['generated_utc']}",
        f"**Decision:** {record['decision']}",
        f"**Phase-A scientific gate:** {'READY' if record['phase_a_scientific_gate_ready'] else 'NOT READY'}",
        f"**Resize prerequisites:** {record['resize_checks_passed']}/{record['resize_checks_total']}",
        f"**All tracked checks:** {record['checks_passed']}/{record['checks_total']}",
        "",
    ]
    summary = record["phase_a_summary"]
    lines.append(
        "Phase A aggregate: "
        f"status={summary['status']}; PASS={summary['pass']}; FAIL={summary['fail']}; "
        f"minimum valid={summary['minimum_valid_required']}; frozen valid={summary['frozen_valid_count']}."
    )
    lines.extend(("", "## Checks", ""))
    for check in record["checks"]:
        mark = "x" if check["pass"] else " "
        lines.append(
            f"- [{mark}] **{check['name']}** [{check['category']}] — {check['detail']}"
        )
    lines.extend(
        (
            "",
            "This gate never authorizes tractography, matrix generation, dashboard publication, or full-cohort execution.",
        )
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--decision", type=Path, default=DEFAULT_DECISION)
    parser.add_argument("--gpu-build", type=Path, default=DEFAULT_GPU_BUILD)
    parser.add_argument("--output-validation", type=Path, default=DEFAULT_OUTPUT_VALIDATION)
    parser.add_argument(
        "--retry3-package-validation",
        type=Path,
        default=DEFAULT_RETRY3_PACKAGE_VALIDATION,
    )
    parser.add_argument(
        "--retry3-dryrun-validation",
        type=Path,
        default=DEFAULT_RETRY3_DRYRUN_VALIDATION,
    )
    parser.add_argument("--service", default="scforge-h04a-r1-phase-a-retry2.service")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    record = build_record(
        run_root=args.run_root.resolve(),
        decision_path=args.decision.resolve(),
        gpu_build_path=args.gpu_build.resolve(),
        output_validation_path=args.output_validation.resolve(),
        retry3_package_validation_path=args.retry3_package_validation.resolve(),
        retry3_dryrun_validation_path=args.retry3_dryrun_validation.resolve(),
        service=args.service,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "gpu_transition_gate.json"
    md_path = args.output_dir / "gpu_transition_gate.md"
    json_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(record), encoding="utf-8")
    print(json.dumps({
        "decision": record["decision"],
        "phase_a_scientific_gate_ready": record["phase_a_scientific_gate_ready"],
        "resize_checks": f"{record['resize_checks_passed']}/{record['resize_checks_total']}",
        "all_checks": f"{record['checks_passed']}/{record['checks_total']}",
        "output": str(json_path),
    }))


if __name__ == "__main__":
    main()
