#!/usr/bin/env python3
"""Compatibility entrypoint for the active SC-Forge v1 density batch."""

from pathlib import Path
import runpy


if __name__ == "__main__":
    target = Path(__file__).resolve().parent / "scripts" / "scforge" / "live" / "run_scforge_v1_density_batch.py"
    runpy.run_path(str(target), run_name="__main__")
