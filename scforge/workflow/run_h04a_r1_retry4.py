#!/usr/bin/env python3
"""Versioned launcher for the immutable-upstream SL-H04A-R1 retry4."""

from __future__ import annotations

import sys
from pathlib import Path


WORKFLOW_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = WORKFLOW_DIR.parents[1]
sys.path.insert(0, str(WORKFLOW_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "scforge"))

import run_h04a_r1_recovery as base  # noqa: E402
from scforge.h04a_r1_retry4 import (  # noqa: E402
    validate_recovery_extension_binding,
)


base.NORMATIVE_CONFIG = PROJECT_ROOT / "configs" / "connectome_v2_retry3.yaml"
base.ENVIRONMENT_CONTRACT = WORKFLOW_DIR / "environment_contract_retry3.yaml"
base.SNAKEFILE = WORKFLOW_DIR / "Snakefile_h04a_r1_retry4"
base.validate_recovery_extension_binding = validate_recovery_extension_binding

CONTINUATION_RULES = (
    "select_fod_shells",
    "subject_response",
    "response_calibration_phase_a",
)

_base_build_snakemake_command = base.build_snakemake_command


def _retry4_build_snakemake_command(**kwargs):
    command = _base_build_snakemake_command(**kwargs)
    trigger_index = command.index("--rerun-triggers")
    if command[trigger_index + 1] != "mtime":
        raise RuntimeError("base retry trigger changed unexpectedly")
    command[trigger_index + 1 : trigger_index + 2] = ["input", "params"]
    keep_going_index = command.index("--keep-going")
    command[keep_going_index:keep_going_index] = [
        "--allowed-rules",
        *CONTINUATION_RULES,
    ]
    return command


# Compatibility name used by the existing dry-run audit helper.
_retry3_build_snakemake_command = _retry4_build_snakemake_command
base.build_snakemake_command = _retry4_build_snakemake_command


if __name__ == "__main__":
    raise SystemExit(base.main())
