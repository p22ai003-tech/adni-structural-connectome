from __future__ import annotations

import copy
import sys
import tempfile
import unittest
from pathlib import Path

import yaml


WORKFLOW_DIR = Path(__file__).resolve().parents[1] / "workflow"
sys.path.insert(0, str(WORKFLOW_DIR))

from validate_environment_contract import (  # noqa: E402
    DEFAULT_CONFIG,
    DEFAULT_ENVIRONMENT,
    _hashed_artifacts,
    load_environment,
    load_recipe,
    audit_environment_contract,
    policy_consistency_checks,
    validate_hashed_artifact,
    workflow_integration_checks,
)


def _toolchain_installed() -> bool:
    """Are the executables the contract names actually on this host?

    Three tests below audit the host against its contract -- every binary
    present and matching its SHA-256. That is the right check on a machine set
    up to run the pipeline, and meaningless on one with no imaging tools, such
    as a CI runner; there they are skipped. The policy tests, which compare the
    files with each other, run everywhere.
    """
    environment = load_environment(DEFAULT_ENVIRONMENT)
    probes = (
        Path(environment["mrtrix3"]["prefix"]) / "bin" / "mrinfo",
        Path(environment["fsl"]["prefix"]) / "bin" / "flirt",
        Path(environment["ants"]["prefix"]) / "bin" / "antsRegistration",
    )
    return all(probe.is_file() for probe in probes)


requires_toolchain = unittest.skipUnless(
    _toolchain_installed(),
    "needs MRtrix3, FSL and ANTs installed (see IMAGING.md, `run_imaging.py lock-env`)",
)


class EnvironmentArtifactTests(unittest.TestCase):
    @requires_toolchain
    def test_all_declared_artifact_hashes_match_current_host(self) -> None:
        environment = load_environment(DEFAULT_ENVIRONMENT)
        mismatches = []
        for name, record, _ in _hashed_artifacts(environment):
            passed, observed = validate_hashed_artifact(
                Path(record["path"]), str(record["sha256"])
            )
            if not passed:
                mismatches.append((name, record["sha256"], observed))
        self.assertEqual(mismatches, [])

    def test_hash_mismatch_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "tool"
            artifact.write_bytes(b"locked bytes\n")
            passed, observed = validate_hashed_artifact(artifact, "0" * 64)
        self.assertFalse(passed)
        self.assertIsNotNone(observed)
        self.assertNotEqual(observed, "0" * 64)


class EnvironmentPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.environment = load_environment(DEFAULT_ENVIRONMENT)
        cls.config = load_recipe(DEFAULT_CONFIG, cls.environment)

    def test_cross_file_fail_closed_policy_has_no_errors(self) -> None:
        checks = policy_consistency_checks(self.environment, self.config)
        failures = [
            row["code"]
            for row in checks
            if not row["passed"] and row["severity"] == "error"
        ]
        self.assertEqual(failures, [])

    def test_cuda_mutation_breaks_single_cpu_policy(self) -> None:
        mutated = copy.deepcopy(self.config)
        mutated["dwi_preprocessing"]["eddy"]["executable"] = self.environment["fsl"][
            "binaries"
        ]["eddy_cuda11_0"]["path"]
        checks = policy_consistency_checks(self.environment, mutated)
        eddy = next(row for row in checks if row["code"] == "policy.eddy_single_cpu_executor")
        self.assertFalse(eddy["passed"])

    @requires_toolchain
    def test_workflow_resolves_locked_tools_and_cpu_only_eddy(self) -> None:
        checks = workflow_integration_checks(self.environment, self.config)
        failures = [
            (row["code"], row.get("observed"))
            for row in checks
            if not row["passed"] and row["severity"] == "error"
        ]
        self.assertEqual(failures, [])
        by_code = {row["code"]: row for row in checks}
        for command in ("dwifslpreproc", "tckgen", "tcksift2", "tck2connectome"):
            self.assertTrue(by_code[f"workflow.resolution_{command}"]["passed"])
        self.assertTrue(by_code["workflow.normal_path_excludes_tissue"]["passed"])
        self.assertTrue(by_code["workflow.eddy_cuda_not_discoverable"]["passed"])

    @requires_toolchain
    def test_full_read_only_audit_has_no_errors(self) -> None:
        report = audit_environment_contract()
        self.assertNotEqual(
            report["status"],
            "FAIL",
            yaml.safe_dump(report.get("failures", []), sort_keys=False),
        )


if __name__ == "__main__":
    unittest.main()
