#!/usr/bin/env python3
"""Compatibility entrypoint for SC-Forge final spatial-contract closeout logic."""

from pathlib import Path
import runpy


if __name__ == "__main__":
    target = Path(__file__).resolve().parent / "scripts" / "scforge" / "live" / "run_sc_final_spatial_contract_closeout.py"
    runpy.run_path(str(target), run_name="__main__")
