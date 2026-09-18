#!/usr/bin/env python3
"""Compatibility entrypoint for active SC-Forge AAL3 source-contract probes."""

from pathlib import Path
import runpy


if __name__ == "__main__":
    target = Path(__file__).resolve().parent / "scripts" / "scforge" / "live" / "run_sc_aal3_source_contract_probe.py"
    runpy.run_path(str(target), run_name="__main__")
