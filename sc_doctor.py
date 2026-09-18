#!/usr/bin/env python
"""Check whether this machine can run the pipeline, before it tries to.

    python sc_doctor.py              # everything
    python sc_doctor.py --imaging    # only what the imaging stages need
    python sc_doctor.py --analysis   # only what the analysis stages need

Why this exists
---------------
The failure mode this prevents is a long run that dies in the middle. Two real
examples from this project: the v2 workflow reported "unable to extract WM-FOD
l=0 coefficient", which was not a data problem at all -- MRtrix was installed
but MRTRIX_BIN was unset and its bin directory was not on PATH. And the test
suite appeared to have 247 unrunnable tests, which was one missing package in
one of two virtual environments.

Neither was visible until something failed a long way in. Both are one line
here.

Exit status is 0 when nothing is missing, 1 when something required is absent,
so this can gate a run.
"""

from __future__ import annotations

import argparse
import importlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import sc_config  # noqa: E402

OK, WARN, BAD = "ok", "warn", "MISSING"

# (binary, which env var conventionally locates it, what needs it)
IMAGING_TOOLS = [
    ("mrconvert", "MRTRIX_BIN", "MRtrix3: conversion, FOD, tractography"),
    ("dwi2fod", "MRTRIX_BIN", "MRtrix3: constrained spherical deconvolution"),
    ("tckgen", "MRTRIX_BIN", "MRtrix3: streamline generation"),
    ("tck2connectome", "MRTRIX_BIN", "MRtrix3: connectome assembly"),
    ("dwifslpreproc", "MRTRIX_BIN", "MRtrix3: eddy / motion wrapper"),
    ("eddy_cpu", "FSLDIR", "FSL: eddy-current and motion correction"),
    ("bet", "FSLDIR", "FSL: brain extraction"),
    ("flirt", "FSLDIR", "FSL: linear registration"),
    ("antsRegistration", "ANTSPATH", "ANTs: nonlinear registration"),
    ("N4BiasFieldCorrection", "ANTSPATH", "ANTs: bias field correction"),
    ("dcm2niix", None, "DICOM -> NIfTI, and the acquisition sidecars"),
]

ANALYSIS_PACKAGES = [
    ("numpy", True), ("pandas", True), ("scipy", True), ("sklearn", True),
    ("matplotlib", True), ("seaborn", True), ("networkx", True),
    ("statsmodels", True), ("nibabel", True), ("shap", True), ("yaml", True),
    ("xgboost", False), ("tabpfn", False), ("pytorch_tabnet", False),
    ("jsonschema", False), ("pytest", False), ("pydicom", False),
]


def _print(status: str, label: str, detail: str = "") -> None:
    mark = {OK: "  ok   ", WARN: "  warn ", BAD: "  MISS "}[status]
    print(f"{mark} {label:<26} {detail}")


def tool_version(path: str) -> str:
    for flag in ("-version", "--version", "-v"):
        try:
            r = subprocess.run([path, flag], capture_output=True, text=True, timeout=20)
            for line in (r.stdout + r.stderr).splitlines():
                line = line.strip()
                if line and any(c.isdigit() for c in line):
                    return line[:58]
        except Exception:
            continue
    return ""


def check_paths() -> int:
    print("\nPaths")
    print("-" * 72)
    p = sc_config.paths()
    problems = 0
    # Roots that must exist for anything to work; the rest are created on demand.
    required = {"project_root", "data_root"}
    for name, value in p.as_dict().items():
        exists = Path(value).exists()
        if exists:
            _print(OK, name, value)
        elif name in required:
            _print(BAD, name, f"{value}   <- set SC_{name.upper()}")
            problems += 1
        else:
            _print(WARN, name, f"{value}   (absent; created on demand)")
    return problems


def check_tools(required: bool) -> int:
    print("\nImaging toolchain")
    print("-" * 72)
    t = sc_config.tools()
    problems = 0
    for binary, env_var, purpose in IMAGING_TOOLS:
        resolved = None
        base = {"MRTRIX_BIN": t.mrtrix_bin,
                "FSLDIR": (t.fsl_dir / "bin") if t.fsl_dir else None,
                "ANTSPATH": t.ants_path}.get(env_var) if env_var else None
        if base is not None and (base / binary).exists():
            resolved = str(base / binary)
        elif binary == "dcm2niix":
            # Resolved by the same helper the probe uses, so the two cannot
            # disagree about which binary is in play: this repo vendors it
            # under tools/dcm2niix/vX/ rather than expecting it on PATH.
            try:
                import sc_probe_acquisition
                resolved = sc_probe_acquisition.find_dcm2niix()
            except SystemExit:
                resolved = None
        else:
            resolved = shutil.which(binary)
        if resolved:
            _print(OK, binary, f"{tool_version(resolved) or resolved}")
        else:
            hint = f"set {env_var}, or put it on PATH" if env_var else "install it"
            _print(BAD if required else WARN, binary, f"{purpose}  ({hint})")
            problems += required
    if problems:
        print("\n  MRtrix, FSL and ANTs are frequently installed but not exported.")
        print("  Copy env.sh.example to env.sh, edit the three paths, then `source env.sh`.")
    return problems


def check_packages() -> int:
    print("\nPython packages")
    print("-" * 72)
    print(f"  interpreter: {sys.executable}  (Python {sys.version.split()[0]})")
    if sys.version_info < (3, 12):
        _print(BAD, "python", f"3.12 required, found {sys.version.split()[0]}")
    problems = 0
    for mod, required in ANALYSIS_PACKAGES:
        try:
            m = importlib.import_module(mod)
            _print(OK, mod, getattr(m, "__version__", ""))
        except ImportError:
            _print(BAD if required else WARN, mod,
                   "requirements/analysis.txt" if required else "optional")
            problems += required
    return problems


def check_cohort() -> int:
    print("\nCohort tables")
    print("-" * 72)
    folder = sc_config.paths().cohort_dir
    try:
        import sc_cohort
    except Exception as exc:
        _print(WARN, "sc_cohort", f"could not import: {exc}")
        return 0
    problems = 0
    for name, cols in sc_cohort.REQUIRED.items():
        path = folder / name
        if not path.is_file():
            _print(BAD, name, f"{path}   <- build with: python sc_cohort.py build --exports <folder>")
            problems += 1
            continue
        try:
            import pandas as pd
            head = pd.read_csv(path, nrows=0).columns
        except Exception as exc:
            _print(BAD, name, f"unreadable: {exc}")
            problems += 1
            continue
        missing = [c for c in cols if c not in head]
        if missing:
            _print(BAD, name, f"missing column(s): {', '.join(missing)}")
            problems += 1
        else:
            _print(OK, name, str(path))
    try:
        import sc_exclusions
        n = len(sc_exclusions.load())
        _print(OK if n else WARN, "exclusions",
               f"{n} subject(s) from {sc_exclusions.path()}" if n else
               f"{sc_exclusions.path()} absent; the ML stages will not reproduce the published numbers")
    except Exception:
        pass
    return problems


def check_manifest() -> int:
    print("\nInput contract")
    print("-" * 72)
    m = sc_config.paths().manifest
    if not Path(m).is_file():
        _print(WARN, "manifest", f"{m}  (absent; build it with discovery)")
        return 0
    try:
        import sc_manifest
        rc = sc_manifest.validate(Path(m))
        return 1 if rc else 0
    except Exception as exc:
        _print(WARN, "manifest", f"could not validate: {type(exc).__name__}: {exc}")
        return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--imaging", action="store_true", help="only the imaging requirements")
    ap.add_argument("--analysis", action="store_true", help="only the analysis requirements")
    args = ap.parse_args(argv)
    both = not (args.imaging or args.analysis)

    print("=" * 72)
    print("pipeline preflight")
    print("=" * 72)
    problems = check_paths()
    if both or args.analysis:
        problems += check_packages()
        problems += check_cohort()
    if both or args.imaging:
        problems += check_tools(required=args.imaging or both)
        problems += check_manifest()

    print("\n" + "=" * 72)
    if problems:
        print(f"{problems} problem(s). The pipeline will not complete until these are fixed.")
    else:
        print("no problems found.")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
