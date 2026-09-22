#!/usr/bin/env python3
"""Freeze diagnosis-blind phase-A response outcomes for phase-B reuse."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import yaml


WORKFLOW_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = WORKFLOW_DIR.parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "connectome_v2.yaml"
sys.path.insert(0, str(PROJECT_ROOT / "scforge"))

from scforge.response_calibration import (  # noqa: E402
    freeze_response_calibration_from_phase_a,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-a-completion", type=Path, required=True)
    parser.add_argument("--phase-a-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-valid-subjects", type=int, required=True)
    args = parser.parse_args()
    # The recipe names responsemean indirectly; resolve it against the run's
    # frozen environment when there is one, else against this machine's.
    from scforge.environment import expand, load_config, merged_contract

    frozen = args.output_dir.expanduser().resolve().parent / "contract" / "environment_contract.yaml"
    environment = (expand(yaml.safe_load(frozen.read_text(encoding="utf-8")), None)
                   if frozen.is_file() else merged_contract())
    config = load_config(CONFIG_PATH, environment)
    pooling = config["fod"]["response_estimation"]["calibration"]["pooling"]
    result = freeze_response_calibration_from_phase_a(
        args.phase_a_completion,
        args.phase_a_manifest,
        args.output_dir,
        minimum_valid_subjects=args.minimum_valid_subjects,
        generated_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        responsemean_executable=pooling["executable"],
        expected_responsemean_sha256=pooling["executable_sha256"],
    )
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "status",
                    "valid_subject_count",
                    "invalid_subject_count",
                    "manifest_path",
                    "manifest_sha256",
                    "phase_a_lineage",
                    "valid_pool_technical_diversity",
                    "valid_pool_technical_diversity_sha256",
                )
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
