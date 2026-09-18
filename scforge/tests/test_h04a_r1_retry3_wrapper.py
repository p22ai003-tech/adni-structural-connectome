from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


WRAPPER = Path(__file__).parents[1] / "workflow" / "run_h04a_r1_retry3.py"


class Retry3WrapperTests(unittest.TestCase):
    def test_wrapper_selects_versioned_sources_and_non_mtime_triggers(self) -> None:
        spec = importlib.util.spec_from_file_location("run_h04a_r1_retry3_test", WRAPPER)
        self.assertIsNotNone(spec)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        command = module.base.build_snakemake_command(
            snakemake=Path("/locked/snakemake"),
            resolved_config_path=Path("/attempt/config.yaml"),
            cores=32,
            mode="response-calibration-phase-a",
        )
        trigger_index = command.index("--rerun-triggers")
        self.assertEqual(command[trigger_index + 1 : trigger_index + 3], ["input", "params"])
        allowed_index = command.index("--allowed-rules")
        keep_going_index = command.index("--keep-going")
        self.assertEqual(
            command[allowed_index + 1 : keep_going_index],
            list(module.CONTINUATION_RULES),
        )
        self.assertEqual(
            set(module.CONTINUATION_RULES),
            {
                "select_fod_shells",
                "subject_response",
                "response_calibration_phase_a",
            },
        )
        self.assertNotIn("dwi_motion_eddy", module.CONTINUATION_RULES)
        self.assertEqual(command[-1], "response_calibration_phase_a")
        self.assertEqual(module.base.SNAKEFILE.name, "Snakefile_h04a_r1_retry3")
        self.assertEqual(module.base.NORMATIVE_CONFIG.name, "connectome_v2_retry3.yaml")
        self.assertEqual(
            module.base.ENVIRONMENT_CONTRACT.name,
            "environment_contract_retry3.yaml",
        )


if __name__ == "__main__":
    unittest.main()
