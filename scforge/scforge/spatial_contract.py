from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any

from .config import SCForgeConfig


DEFAULT_PANEL: tuple[str, ...] = (
    "003_S_0908_I1249292",
    "127_S_5028_I401540",
    "168_S_6874_I1667523",
    "021_S_7092_I1597668",
    "003_S_6257_I974346",
    "031_S_4021_I1253150",
    "033_S_7114_I11063036",
    "014_S_4401_I1556672",
    "041_S_5141_I893581",
    "041_S_4427_I1243839",
)


def utc_stamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


@dataclass(frozen=True)
class SpatialCanaryPlan:
    tag: str
    run_root: Path
    command: list[str]
    variants: tuple[str, ...]
    subjects: tuple[str, ...]
    execute: bool
    launch_manifest: Path

    def to_dict(self) -> dict[str, Any]:
        return {
            "tag": self.tag,
            "run_root": str(self.run_root),
            "command": self.command,
            "variants": list(self.variants),
            "subjects": list(self.subjects),
            "execute": self.execute,
            "launch_manifest": str(self.launch_manifest),
        }


def _fsl_python(cfg: SCForgeConfig) -> str:
    candidate = cfg.project_root.parent / "fsl" / "bin" / "python"
    if candidate.exists():
        return str(candidate)
    return "/home/ec2-user/fsl/bin/python"


def _closeout_script(cfg: SCForgeConfig) -> Path:
    return cfg.project_root / "run_sc_final_spatial_contract_closeout.py"


def default_variants(include_forward80: bool = False) -> tuple[str, ...]:
    variants = ["default", "radial8", "forward40"]
    if include_forward80:
        variants.append("forward80")
    return tuple(variants)


def build_spatial_canary_plan(
    cfg: SCForgeConfig,
    *,
    tag: str | None = None,
    subjects: list[str] | tuple[str, ...] | None = None,
    variants: list[str] | tuple[str, ...] | None = None,
    include_forward80: bool = False,
    execute: bool = False,
) -> SpatialCanaryPlan:
    tag = tag or f"scforge_spatial_contract_{utc_stamp()}"
    subjects_tuple = tuple(subjects or DEFAULT_PANEL)
    variants_tuple = tuple(variants or default_variants(include_forward80=include_forward80))
    run_root = cfg.existing_derivatives_root / "qc" / "sc_matrix_qc" / tag
    script = _closeout_script(cfg)
    command = [
        _fsl_python(cfg),
        str(script),
        "--tag",
        tag,
        "--subjects",
        *subjects_tuple,
        "--variants",
        *variants_tuple,
    ]
    launch_dir = cfg.existing_derivatives_root / "qc" / "sc_matrix_qc" / "scforge_launches"
    launch_manifest = launch_dir / f"{tag}.json"
    return SpatialCanaryPlan(
        tag=tag,
        run_root=run_root,
        command=command,
        variants=variants_tuple,
        subjects=subjects_tuple,
        execute=execute,
        launch_manifest=launch_manifest,
    )


def write_launch_manifest(plan: SpatialCanaryPlan) -> Path:
    plan.launch_manifest.parent.mkdir(parents=True, exist_ok=True)
    payload = plan.to_dict()
    payload["created_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    tmp = plan.launch_manifest.with_suffix(plan.launch_manifest.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(plan.launch_manifest)
    return plan.launch_manifest


def run_spatial_canary(plan: SpatialCanaryPlan, cfg: SCForgeConfig) -> int:
    write_launch_manifest(plan)
    if not plan.execute:
        return 0
    script = _closeout_script(cfg)
    if not script.exists():
        raise FileNotFoundError(script)
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{cfg.project_root / 'scforge'}:{cfg.project_root}:" + env.get("PYTHONPATH", "")
    env["PATH"] = (
        f"{cfg.project_root.parent / 'mrtrix3' / 'bin'}:"
        f"{cfg.project_root.parent / 'fsl' / 'share' / 'fsl' / 'bin'}:"
        f"{cfg.project_root.parent / 'fsl' / 'bin'}:"
        + env.get("PATH", "")
    )
    proc = subprocess.run(plan.command, cwd=str(cfg.project_root), env=env, text=True)
    return int(proc.returncode)
