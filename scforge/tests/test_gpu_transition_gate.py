from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


PROJECT = Path(__file__).resolve().parents[2]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from research_audit import build_gpu_transition_gate as gate  # noqa: E402


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _record(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "size_bytes": path.stat().st_size,
    }


def _valid_fixture(tmp_path: Path) -> dict[str, Path]:
    run_root = tmp_path / "run"
    attempts = run_root / "attempts"
    publication = run_root / "publication"
    attempts.mkdir(parents=True)
    publication.mkdir(parents=True)

    units = [f"unit-{index:02d}" for index in range(15)]
    decision = tmp_path / "decision.json"
    _write_json(
        decision,
        {
            "units": units,
            "minimum_valid_response_calibration_units": 12,
            "fod_reconstruction_authorized": False,
            "tractography_authorized": False,
            "matrix_generation_authorized": False,
            "full_cohort_authorized": False,
        },
    )

    execution_log = attempts / "attempt.execution.log"
    execution_log.write_text("known terminal failure\n", encoding="utf-8")
    attempt_start = attempts / "attempt.start.json"
    _write_json(
        attempt_start,
        {
            "attempt_id": "attempt",
            "launcher_mode": "response-calibration-phase-a",
        },
    )
    attempt_end = attempts / "attempt.end.json"
    _write_json(
        attempt_end,
        {
            "attempt_id": "attempt",
            "launcher_mode": "response-calibration-phase-a",
            "status": "FAIL",
            "returncode": 1,
            "attempt_start": _record(attempt_start),
            "combined_execution_log": _record(execution_log),
        },
    )

    outcomes: dict[str, dict[str, object]] = {}
    for unit in units:
        outcome = run_root / "subjects" / unit / "05_model" / "response_calibration_outcome.json"
        _write_json(outcome, {"unit": unit, "status": "FAIL"})
        outcomes[unit] = _record(outcome)
    summary = {"expected": 15, "terminal": 15, "pass": 0, "fail": 15}
    manifest = publication / "response_calibration_phase_a_manifest.json"
    _write_json(
        manifest,
        {
            "expected_units": units,
            "summary": summary,
            "outcome_records": outcomes,
        },
    )
    completion = publication / "response_calibration_phase_a_completion.json"
    _write_json(
        completion,
        {
            "record_type": "response_calibration_phase_a_completion",
            "mode": "response-calibration-phase-a",
            "status": "FAIL",
            "production_overwrite": False,
            "full_run_terminal_publication": False,
            "recipe_id": "recipe",
            "run_id": "run",
            "snakemake_returncode": 1,
            "phase_a_summary": summary,
            "phase_a_manifest": _record(manifest),
            "terminal_attempt_start": _record(attempt_start),
            "terminal_attempt_end": _record(attempt_end),
            "combined_execution_log": _record(execution_log),
        },
    )

    gpu_build = tmp_path / "gpu_build.json"
    _write_json(
        gpu_build,
        {
            "status": "PASS",
            "execution_performed": False,
            "resize_performed": False,
            "runner_static_validation": {
                "checks": {f"check-{index}": True for index in range(11)}
            },
        },
    )
    output_builder_validation = tmp_path / "validation.json"
    _write_json(output_builder_validation, {"pass": True})
    output_validator = PROJECT / "research_audit" / "validate_refined_output_release_v1.py"
    output_validation = tmp_path / "output_validation.json"
    _write_json(
        output_validation,
        {
            "pass": True,
            "checks": {f"check-{index}": True for index in range(28)},
            "builder_validation_sha256": hashlib.sha256(
                output_builder_validation.read_bytes()
            ).hexdigest(),
            "validator_sha256": hashlib.sha256(output_validator.read_bytes()).hexdigest(),
        },
    )
    retry3_package = tmp_path / "retry3_package.json"
    _write_json(
        retry3_package,
        {
            "status": "PASS",
            "checks": {f"check-{index}": True for index in range(15)},
            "summary": {"passed": 15, "total": 15},
        },
    )
    retry3_dryrun = tmp_path / "retry3_dryrun.json"
    _write_json(
        retry3_dryrun,
        {
            "status": "PASS",
            "checks": {f"check-{index}": True for index in range(11)},
            "scheduled_rules": sorted(gate.EXPECTED_RETRY3_CONTINUATION_RULES),
            "allowed_rules": sorted(gate.EXPECTED_RETRY3_CONTINUATION_RULES),
            "unexpected_upstream_rules": [],
            "forbidden_downstream_rules": [],
            "missing_expected_continuation_rules": [],
        },
    )
    return {
        "run_root": run_root,
        "decision": decision,
        "gpu_build": gpu_build,
        "output_validation": output_validation,
        "output_builder_validation": output_builder_validation,
        "retry3_package": retry3_package,
        "retry3_dryrun": retry3_dryrun,
    }


def _build(paths: dict[str, Path]) -> dict[str, object]:
    return gate.build_record(
        run_root=paths["run_root"],
        decision_path=paths["decision"],
        gpu_build_path=paths["gpu_build"],
        output_validation_path=paths["output_validation"],
        retry3_package_validation_path=paths["retry3_package"],
        retry3_dryrun_validation_path=paths["retry3_dryrun"],
        service="synthetic.service",
    )


class GpuTransitionGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.paths = _valid_fixture(Path(self.temporary_directory.name))

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _record_with_state(self, state: str) -> dict[str, object]:
        with (
            mock.patch.object(gate, "service_state", return_value=state),
            mock.patch.object(gate, "active_imaging_processes", return_value=[]),
            mock.patch.object(gate, "validate_response_calibration_outcome", return_value=None),
        ):
            return _build(self.paths)

    def test_unknown_service_state_never_passes_resize_gate(self) -> None:
        record = self._record_with_state("unknown")

        self.assertEqual(record["decision"], "NOT_SAFE_TO_RESIZE")
        check = next(
            row
            for row in record["checks"]
            if row["name"] == "phase_a_service_is_known_terminal"
        )
        self.assertIs(check["pass"], False)

    def test_exact_closure_and_retry3_dryrun_are_required(self) -> None:
        record = self._record_with_state("inactive")
        self.assertEqual(record["decision"], "SAFE_TO_RESIZE")

        (self.paths["run_root"] / "attempts" / "extra.end.json").write_text(
            "{}\n", encoding="utf-8"
        )
        record = self._record_with_state("inactive")
        self.assertEqual(record["decision"], "NOT_SAFE_TO_RESIZE")
        check = next(
            row
            for row in record["checks"]
            if row["name"] == "exact_single_attempt_start_end_completion_closure"
        )
        self.assertIs(check["pass"], False)

    def test_missing_retry3_dryrun_never_passes_resize_gate(self) -> None:
        self.paths["retry3_dryrun"].unlink()
        record = self._record_with_state("failed")

        self.assertEqual(record["decision"], "NOT_SAFE_TO_RESIZE")
        check = next(
            row
            for row in record["checks"]
            if row["name"] == "retry3_continuation_dryrun_is_minimal_and_passes"
        )
        self.assertIs(check["pass"], False)

    def test_downstream_authorization_drift_never_passes_resize_gate(self) -> None:
        decision = json.loads(self.paths["decision"].read_text(encoding="utf-8"))
        decision["tractography_authorized"] = True
        _write_json(self.paths["decision"], decision)

        record = self._record_with_state("inactive")

        self.assertEqual(record["decision"], "NOT_SAFE_TO_RESIZE")
        self.assertEqual(record["resize_checks_total"], 13)
        check = next(
            row
            for row in record["checks"]
            if row["name"] == "decision_prohibits_downstream_execution"
        )
        self.assertIs(check["pass"], False)

    def test_output_readiness_hash_binding_detects_drift_without_blurring_resize_gate(self) -> None:
        self.paths["output_builder_validation"].write_text(
            '{"pass": false}\n', encoding="utf-8"
        )
        record = self._record_with_state("inactive")

        self.assertEqual(record["decision"], "SAFE_TO_RESIZE")
        check = next(
            row
            for row in record["checks"]
            if row["name"] == "refined_output_release_independent_validation_passes"
        )
        self.assertIs(check["pass"], False)
        self.assertIs(check["detail"]["builder_validation_hash_matches"], False)


if __name__ == "__main__":
    unittest.main()
