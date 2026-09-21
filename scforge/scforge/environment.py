"""Where the tools are on this machine, kept out of the recipe.

The recipe (configs/connectome_v2.yaml) says *what* to run. Where MRtrix3,
FSL, ANTs, Convert3D and MRtrix3Tissue live differs on every machine, so the
recipe refers to them indirectly and they are resolved here:

``${repo}``
    the repository root, wherever it was cloned.
``${env:a.b.c}``
    a value from the environment contract -- for example
    ``${env:fsl.binaries.eddy_cpu.path}`` or ``${env:ants.prefix}``.

The environment contract comes in two parts:

``scforge/workflow/environment_contract.yaml``
    the reference record: every tool path, version and SHA-256 on the machine
    the thesis cohort was processed on. Committed, and never edited to suit
    another machine.
``configs/environment.local.yaml``
    this machine's tools, written by ``python run_imaging.py lock-env``.
    Gitignored. Its sections replace the reference's; anything it does not
    mention -- the dependency locks and the workflow source manifest, which
    belong to the repository rather than the machine -- comes from the
    reference.

``SC_ENVIRONMENT_CONTRACT`` names a different local file.
"""

from __future__ import annotations

import copy
import os
import re
from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parents[2]
REFERENCE_CONTRACT = REPO / "scforge" / "workflow" / "environment_contract.yaml"
LOCAL_CONTRACT = REPO / "configs" / "environment.local.yaml"

# Sections that describe the machine. Everything else in the contract
# describes the repository and is taken from the reference.
MACHINE_SECTIONS = ("host", "execution_policy", "mrtrix3", "mrtrix3tissue", "fsl", "ants", "convert3d")
MACHINE_WORKFLOW_KEYS = ("python", "snakemake")

_TOKEN = re.compile(r"\$\{(repo|env:[A-Za-z0-9_.]+)\}")

__all__ = [
    "REPO", "REFERENCE_CONTRACT", "LOCAL_CONTRACT", "EnvironmentError_",
    "local_contract_path", "merged_contract", "expand", "load_config",
]


class EnvironmentError_(RuntimeError):
    """The environment cannot resolve a value the recipe asks for."""


def local_contract_path() -> Path | None:
    override = os.environ.get("SC_ENVIRONMENT_CONTRACT")
    if override:
        return Path(override).expanduser().resolve()
    return LOCAL_CONTRACT if LOCAL_CONTRACT.is_file() else None


def _lookup(contract: dict, dotted: str) -> Any:
    node: Any = contract
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            raise EnvironmentError_(
                f"the environment contract has no value for {dotted!r}. "
                "Run `python run_imaging.py lock-env` on this machine."
            )
        node = node[part]
    return node


def expand(value: Any, contract: dict | None = None) -> Any:
    """Resolve ${repo} and ${env:...} throughout a nested structure."""
    if isinstance(value, dict):
        return {k: expand(v, contract) for k, v in value.items()}
    if isinstance(value, list):
        return [expand(v, contract) for v in value]
    if not isinstance(value, str) or "${" not in value:
        return value

    def substitute(match: re.Match) -> str:
        token = match.group(1)
        if token == "repo":
            return str(REPO)
        if contract is None:
            raise EnvironmentError_(f"{match.group(0)} needs an environment contract")
        found = _lookup(contract, token[4:])
        if isinstance(found, (dict, list)):
            raise EnvironmentError_(f"{match.group(0)} names a section, not a value")
        return str(found)

    whole = _TOKEN.fullmatch(value)
    if whole and whole.group(1).startswith("env:"):
        # a lone reference keeps the contract's own type (a number stays a number)
        if contract is None:
            raise EnvironmentError_(f"{value} needs an environment contract")
        return _lookup(contract, whole.group(1)[4:])
    return _TOKEN.sub(substitute, value)


def _read(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def merged_contract(local: Path | None = None, *, reference: Path = REFERENCE_CONTRACT) -> dict:
    """The reference contract with this machine's sections laid over it."""
    contract = _read(reference)
    local = local if local is not None else local_contract_path()
    if local is not None:
        overlay = _read(local)
        for section in MACHINE_SECTIONS:
            if section in overlay:
                contract[section] = copy.deepcopy(overlay[section])
        for key in MACHINE_WORKFLOW_KEYS:
            if key in overlay.get("workflow", {}):
                contract.setdefault("workflow", {})[key] = copy.deepcopy(overlay["workflow"][key])
        contract["local_contract"] = {"path": str(local), "generated_utc": overlay.get("generated_utc")}
    return expand(contract, None)


def load_config(path: str | Path, contract: dict | None = None) -> dict:
    """Read the recipe and resolve its machine references."""
    config = _read(Path(path))
    return expand(config, contract if contract is not None else merged_contract())
