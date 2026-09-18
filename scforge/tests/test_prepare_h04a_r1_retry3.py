from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT / "research_audit" / "prepare_h04a_r1_retry3.py"
SPEC = importlib.util.spec_from_file_location("prepare_h04a_r1_retry3_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
prepare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare)

DRYRUN_SCRIPT = PROJECT / "research_audit" / "validate_h04a_r1_retry3_dryrun.py"
DRYRUN_SPEC = importlib.util.spec_from_file_location(
    "validate_h04a_r1_retry3_dryrun_test", DRYRUN_SCRIPT
)
assert DRYRUN_SPEC is not None and DRYRUN_SPEC.loader is not None
dryrun = importlib.util.module_from_spec(DRYRUN_SPEC)
DRYRUN_SPEC.loader.exec_module(dryrun)


class Retry3PreparationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.source_root = Path(self.temporary_directory.name)
        self.units = [f"unit-{index:02d}" for index in range(15)]
        for unit in self.units:
            for relative in prepare.REQUIRED_SEED_PRODUCTS:
                product = self.source_root / "subjects" / unit / relative
                product.parent.mkdir(parents=True, exist_ok=True)
                product.write_bytes(b"validated-product")

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_complete_15_unit_seed_is_accepted(self) -> None:
        counts = prepare.require_complete_seed_inputs(
            self.units, source_root=self.source_root
        )

        self.assertEqual(set(counts.values()), {15})

    def test_missing_product_fails_before_destination_creation(self) -> None:
        missing = (
            self.source_root
            / "subjects"
            / self.units[-1]
            / "01_dwi/dwi_brain_mask.mif"
        )
        missing.unlink()

        with self.assertRaisesRegex(ValueError, "missing or invalid"):
            prepare.require_complete_seed_inputs(
                self.units, source_root=self.source_root
            )

    def test_symlink_product_is_rejected(self) -> None:
        product = (
            self.source_root
            / "subjects"
            / self.units[0]
            / "01_dwi/dwi_preproc.mif"
        )
        target = product.with_name("target.mif")
        product.rename(target)
        product.symlink_to(target)

        with self.assertRaisesRegex(ValueError, "missing or invalid"):
            prepare.require_complete_seed_inputs(
                self.units, source_root=self.source_root
            )

    def test_only_known_service_states_are_terminal(self) -> None:
        self.assertEqual(prepare.KNOWN_TERMINAL_SERVICE_STATES, {"inactive", "failed"})
        self.assertNotIn("unknown", prepare.KNOWN_TERMINAL_SERVICE_STATES)
        self.assertNotIn("active", prepare.KNOWN_TERMINAL_SERVICE_STATES)

    def test_all_11_pre_retry3_operational_checks_are_required(self) -> None:
        record = {
            "resize_checks_total": 13,
            "checks": [
                {"name": name, "pass": True}
                for name in sorted(prepare.PRE_RETRY3_OPERATIONAL_CHECKS)
            ],
        }

        passed = prepare.require_pre_retry3_operational_gate(record)

        self.assertEqual(len(passed), 11)

    def test_failed_pre_retry3_operational_check_is_rejected(self) -> None:
        rows = [
            {"name": name, "pass": True}
            for name in sorted(prepare.PRE_RETRY3_OPERATIONAL_CHECKS)
        ]
        rows[0]["pass"] = False

        with self.assertRaisesRegex(ValueError, "failed="):
            prepare.require_pre_retry3_operational_gate(
                {"resize_checks_total": 13, "checks": rows}
            )

    def test_missing_pre_retry3_operational_check_is_rejected(self) -> None:
        rows = [
            {"name": name, "pass": True}
            for name in sorted(prepare.PRE_RETRY3_OPERATIONAL_CHECKS)
        ][1:]

        with self.assertRaisesRegex(ValueError, "missing="):
            prepare.require_pre_retry3_operational_gate(
                {"resize_checks_total": 13, "checks": rows}
            )

    def test_dry_run_uses_only_supported_cli_flag(self) -> None:
        command = ["snakemake", "--cores", "32", "response_calibration_phase_a"]

        augmented = dryrun.add_supported_dry_run_flags(command)

        self.assertIn("--dry-run", augmented)
        self.assertNotIn("--reason", augmented)
        self.assertEqual(augmented[-1], "response_calibration_phase_a")

    def test_dry_run_workspace_is_inside_contract_run_root(self) -> None:
        run_root = self.source_root / "retry3"
        run_root.mkdir()

        with dryrun.temporary_dry_run_workspace(run_root) as directory:
            workspace = Path(directory).resolve()
            self.assertTrue(workspace.is_relative_to(run_root.resolve()))
            self.assertTrue(workspace.name.startswith(".h04a_retry3_dryrun_"))

        self.assertFalse(workspace.exists())

    def test_dry_run_contexts_exist_and_share_launcher_identity(self) -> None:
        run_root = self.source_root / "retry3-context"
        run_root.mkdir()

        with dryrun.temporary_dry_run_workspace(run_root) as directory:
            run_path, attempt_path = dryrun.write_dry_run_contexts(
                Path(directory),
                recipe_id="recipe-test",
                execution_binding={"kind": "execution"},
                recovery_extension_binding={"kind": "recovery"},
                run_root=run_root,
            )
            run_context = dryrun.json.loads(run_path.read_text(encoding="utf-8"))
            attempt_context = dryrun.json.loads(
                attempt_path.read_text(encoding="utf-8")
            )
            self.assertEqual(run_context["run_id"], attempt_context["run_id"])
            self.assertEqual(run_context["recipe_id"], attempt_context["recipe_id"])
            self.assertEqual(
                attempt_context["launcher_mode"],
                "response-calibration-phase-a",
            )
            self.assertEqual(run_context["status"], "DRY_RUN_ONLY")
            self.assertEqual(attempt_context["status"], "DRY_RUN_ONLY")
            self.assertEqual(run_context["run_root"], str(run_root.resolve()))


if __name__ == "__main__":
    unittest.main()
