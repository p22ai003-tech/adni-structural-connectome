#!/usr/bin/env python3
"""Record where this machine's imaging tools are: configs/environment.local.yaml.

The recipe refers to tools indirectly; this writes the file that says where
they are here. For each toolkit it looks, in order, at the conventional
environment variable, then PATH, then the location recorded in the reference
contract, and it hashes every binary the workflow uses so a run can prove which
executables produced it.

    python run_imaging.py lock-env           # write configs/environment.local.yaml
    python run_imaging.py lock-env --check   # is the file still true?

Environment variables it reads:
    MRTRIX_BIN        MRtrix3 bin/ directory
    MRTRIX3TISSUE     MRtrix3Tissue install root (or MRTRIX3TISSUE_BIN for its bin/)
    FSLDIR            FSL install root
    ANTSPATH          ANTs bin/ directory
    C3D_BIN           Convert3D bin/ directory
    SC_MRTRIX_PYTHON  the Python MRtrix's scripts should run under
"""

from __future__ import annotations

import argparse
import copy
import datetime as _dt
import hashlib
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "scforge"))

from scforge.environment import LOCAL_CONTRACT, REFERENCE_CONTRACT, expand  # noqa: E402

TOOLS = {
    # section: (env var naming the prefix, env var naming bin/, a binary to find on PATH)
    "mrtrix3": (None, "MRTRIX_BIN", "tckgen"),
    "mrtrix3tissue": ("MRTRIX3TISSUE", "MRTRIX3TISSUE_BIN", "ss3t_csd_beta1"),
    "fsl": ("FSLDIR", None, "flirt"),
    "ants": (None, "ANTSPATH", "antsRegistration"),
    "convert3d": (None, "C3D_BIN", "c3d_affine_tool"),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _run(command: list[str]) -> str:
    try:
        done = subprocess.run(command, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return (done.stdout + "\n" + done.stderr).strip()


def _os_pretty_name() -> str:
    release = Path("/etc/os-release")
    if release.is_file():
        for line in release.read_text(encoding="utf-8").splitlines():
            if line.startswith("PRETTY_NAME="):
                return line.split("=", 1)[1].strip().strip('"')
    return platform.platform()


def _gpu_available() -> bool:
    if not shutil.which("nvidia-smi"):
        return False
    try:
        return subprocess.run(["nvidia-smi"], capture_output=True, timeout=10).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def find_prefix(section: str, reference_prefix: str) -> tuple[Path | None, str]:
    """Where a toolkit is installed, and how that was decided."""
    prefix_var, bin_var, probe = TOOLS[section]
    if prefix_var and os.environ.get(prefix_var):
        return Path(os.environ[prefix_var]).expanduser().resolve(), f"${prefix_var}"
    if bin_var and os.environ.get(bin_var):
        return Path(os.environ[bin_var]).expanduser().resolve().parent, f"${bin_var}"
    # MRtrix3Tissue ships binaries named like MRtrix3's, so PATH can only be
    # trusted for the one it alone has.
    found = shutil.which(probe)
    if found:
        return Path(found).resolve().parent.parent, f"PATH ({probe})"
    if reference_prefix and Path(reference_prefix).is_dir():
        return Path(reference_prefix).resolve(), "reference location"
    return None, "not found"


def find_script_python(mrtrix_prefix: Path, fsl_prefix: Path | None, reference: str) -> tuple[Path | None, str]:
    """The interpreter MRtrix's Python scripts should run under.

    Not FSL's: FSL bundles its own python3, and letting MRtrix scripts pick it
    up by PATH order is exactly the silent substitution the contract prevents.
    """
    if os.environ.get("SC_MRTRIX_PYTHON"):
        return Path(os.environ["SC_MRTRIX_PYTHON"]).expanduser(), "$SC_MRTRIX_PYTHON"
    if reference and Path(reference).is_file():
        return Path(reference), "reference location"
    for entry in os.environ.get("PATH", "").split(os.pathsep):
        candidate = Path(entry) / "python3"
        if not candidate.is_file():
            continue
        if fsl_prefix is not None and str(candidate.resolve()).startswith(str(fsl_prefix)):
            continue
        return candidate, "PATH (python3, excluding FSL's)"
    return None, "not found"


def version_of(section: str, prefix: Path) -> str:
    if section in ("mrtrix3", "mrtrix3tissue"):
        text = _run([str(prefix / "bin" / "mrinfo"), "-version"])
        m = re.search(r"==\s*mrinfo\s+(\S+)", text)
        return m.group(1) if m else "unknown"
    if section == "fsl":
        f = prefix / "etc" / "fslversion"
        return f.read_text().split(":")[0].strip() if f.is_file() else "unknown"
    if section == "ants":
        m = re.search(r"ANTs Version:\s*(\S+)", _run([str(prefix / "bin" / "antsRegistration"), "--version"]))
        return m.group(1) if m else "unknown"
    if section == "convert3d":
        m = re.search(r"(\d+\.\d+\.\d+)", _run([str(prefix / "bin" / "c3d"), "-version"]))
        return m.group(1) if m else "unknown"
    return "unknown"


def python_version(python: Path) -> str:
    return _run([str(python), "-c", "import platform; print(platform.python_version())"]).splitlines()[0] \
        if python.is_file() else "unknown"


def _swap(value, old: str, new: str):
    if isinstance(value, dict):
        return {k: _swap(v, old, new) for k, v in value.items()}
    if isinstance(value, list):
        return [_swap(v, old, new) for v in value]
    if isinstance(value, str) and value.startswith(old):
        return new + value[len(old):]
    return value


def build() -> tuple[dict, list[str], list[tuple]]:
    reference = expand(yaml.safe_load(REFERENCE_CONTRACT.read_text(encoding="utf-8")), None)
    problems: list[str] = []
    report: list[tuple] = []
    prefixes: dict[str, Path] = {}

    local: dict = {
        "schema_version": reference.get("schema_version"),
        "generated_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "generated_by": "sc_lock_environment.py",
        "reference_contract": str(REFERENCE_CONTRACT.relative_to(REPO)),
        # Observed the way validate_environment_contract.py observes a host,
        # so an audit on this machine compares like with like.
        "host": {
            "os": _os_pretty_name(),
            "architecture": platform.machine(),
            "kernel": platform.release(),
            # present-but-failing nvidia-smi (no driver) is not a GPU
            "gpu_driver_available": _gpu_available(),
        },
    }

    for section in TOOLS:
        ref_section = reference[section]
        prefix, how = find_prefix(section, ref_section.get("prefix", ""))
        if prefix is None:
            problems.append(f"{section}: not found (set {TOOLS[section][0] or TOOLS[section][1]})")
            report.append((section, "-", how, "-", ref_section.get("version", "")))
            continue
        prefixes[section] = prefix
        new = _swap(copy.deepcopy(ref_section), ref_section["prefix"], str(prefix))
        new["prefix"] = str(prefix)
        for name, entry in list(new.get("binaries", {}).items()):
            path = Path(entry["path"])
            if path.is_file():
                entry["sha256"] = sha256_file(path)
            elif entry.get("allowed_by_recipe") is False:
                del new["binaries"][name]   # recorded only to prove it is not used
            else:
                problems.append(f"{section}: {path} is missing")
        if section == "fsl":
            template = prefix / "data" / "standard" / "MNI152_T1_1mm.nii.gz"
            if template.is_file():
                new["mni_template"] = {"path": str(template), "sha256": sha256_file(template)}
            else:
                problems.append(f"fsl: registration template missing: {template}")
        found_version = version_of(section, prefix)
        new["version"] = found_version
        local[section] = new
        report.append((section, str(prefix), how, found_version, ref_section.get("version", "")))

    ref_policy = reference["execution_policy"]
    script_python, how = find_script_python(
        prefixes.get("mrtrix3", Path("/")), prefixes.get("fsl"),
        ref_policy["mrtrix_script_python"]["path"])
    policy = copy.deepcopy(ref_policy)
    if script_python is None or not script_python.is_file():
        problems.append("no Python for MRtrix scripts (set SC_MRTRIX_PYTHON)")
    elif all(k in prefixes for k in TOOLS):
        py_version = python_version(script_python)
        policy["mrtrix_script_python"] = {"path": str(script_python), "version": py_version,
                                          "sha256": sha256_file(script_python)}
        policy["eddy_executor"] = str(prefixes["fsl"] / "bin" / "eddy_cpu")
        policy["normal_tool_path_order"] = [
            str(prefixes["mrtrix3"] / "bin"), str(script_python.parent),
            str(prefixes["fsl"] / "bin"), str(prefixes["ants"] / "bin"),
            str(prefixes["convert3d"] / "bin"),
        ]
        policy["ss3t_invocation"] = {
            "python": str(script_python),
            "script": str(prefixes["mrtrix3tissue"] / "bin" / "ss3t_csd_beta1"),
            "path_prefix": [str(prefixes["mrtrix3tissue"] / "bin"),
                            str(prefixes["mrtrix3"] / "bin"), str(prefixes["fsl"] / "bin")],
        }
        report.append(("mrtrix python", str(script_python), how, py_version,
                       ref_policy["mrtrix_script_python"].get("version", "")))
    local["execution_policy"] = policy

    venv = REPO / ".venv_connectome_workflow" / "bin"
    workflow = {}
    for key, binary in (("python", "python"), ("snakemake", "snakemake")):
        path = venv / binary
        if not path.is_file():
            problems.append(f"workflow {key} missing: {path} (create the workflow venv; see IMAGING.md)")
            continue
        version = python_version(path) if key == "python" else _run([str(path), "--version"]).splitlines()[0]
        workflow[key] = {"path": str(path), "version": version, "sha256": sha256_file(path)}
        report.append((f"workflow {key}", str(path), "repository venv", version,
                       reference["workflow"][key].get("version", "")))
    local["workflow"] = workflow
    return local, problems, report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=LOCAL_CONTRACT)
    parser.add_argument("--check", action="store_true",
                        help="compare the existing file with what is installed, write nothing")
    args = parser.parse_args(argv)

    local, problems, report = build()
    print(f"{'tool':<16} {'version here':<22} {'reference':<22} found via")
    print("-" * 92)
    for tool, prefix, how, version, ref_version in report:
        flag = "" if (not ref_version or version == ref_version) else "   <- differs"
        print(f"{tool:<16} {version[:21]:<22} {str(ref_version)[:21]:<22} {how}{flag}")
        print(f"{'':<16} {prefix}")
    if problems:
        print("\ncannot write an environment file:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        return 1

    differs = [r for r in report if r[4] and r[3] != r[4]]
    if differs:
        print(f"\n{len(differs)} version(s) differ from the reference machine. The run will record "
              "what was used; results are comparable with the reference only where they match.")

    if args.check:
        if not args.out.is_file():
            print(f"\n{args.out} does not exist; run without --check to write it", file=sys.stderr)
            return 1
        current = yaml.safe_load(args.out.read_text(encoding="utf-8"))
        stale = []
        for section in list(TOOLS) + ["workflow"]:
            for name, entry in (current.get(section, {}).get("binaries", {})
                                if section != "workflow" else current.get("workflow", {})).items():
                path = Path(entry["path"])
                if not path.is_file() or sha256_file(path) != entry.get("sha256"):
                    stale.append(f"{section}.{name}")
        if stale:
            print(f"\n{len(stale)} recorded binaries changed or vanished: {', '.join(stale[:8])}")
            print("re-run: python run_imaging.py lock-env")
            return 1
        print(f"\n{args.out} matches what is installed.")
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    header = ("# This machine's imaging tools, written by `python run_imaging.py lock-env`.\n"
              "# Machine-specific and gitignored; regenerate after installing or upgrading a tool.\n"
              "# Sections here replace the same sections of the reference contract,\n"
              f"# {local['reference_contract']}.\n")
    args.out.write_text(header + yaml.safe_dump(local, sort_keys=False), encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
