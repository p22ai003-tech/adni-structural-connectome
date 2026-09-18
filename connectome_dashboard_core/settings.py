from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    project_root: Path
    analysis_root: Path
    connectomes_root: Path
    outputs_root: Path
    aal_labels: Path
    hcp379_live_summary: Path
    hcp379_live_ledger: Path
    hcp379_release_topology: Path
    pipeline_status: Path
    pipeline_density: Path
    pipeline_manifest: Path
    table_default_limit: int = 200
    table_max_limit: int = 5_000


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    project = Path(
        os.environ.get("CONNECTOME_PROJECT_ROOT", "/home/ec2-user/exp")
    ).resolve()
    analysis = Path(
        os.environ.get(
            "CONNECTOME_ANALYSIS_ROOT",
            project / "data/derivatives/qc/analysis_cohort",
        )
    ).resolve()
    connectomes = Path(
        os.environ.get(
            "CONNECTOME_MATRICES_DIR",
            project / "data/derivatives/connectomes",
        )
    ).resolve()
    outputs = Path(
        os.environ.get("CONNECTOME_ENHANCED_ML_ROOT", project / "outputs")
    ).resolve()
    return Settings(
        project_root=project,
        analysis_root=analysis,
        connectomes_root=connectomes,
        outputs_root=outputs,
        aal_labels=project / "atlas/AAL/AAL3_labels.csv",
        hcp379_live_summary=(
            project
            / "research_audit/outputs/hcp379_live_530_status_density_v1/"
            "summary.json"
        ),
        hcp379_live_ledger=(
            project
            / "research_audit/outputs/"
            "hcp379_pretract_failure_recovery_ledger_v1/ledger.csv"
        ),
        hcp379_release_topology=(
            project
            / "research_audit/outputs/"
            "hcp379_balanced_release_topology_v1/topology.json"
        ),
        pipeline_status=Path(
            "/data/derivatives/qc/sc_matrix_qc/connectome_status.json"
        ),
        pipeline_density=Path(
            "/data/derivatives/qc/sc_matrix_qc/density_check.json"
        ),
        pipeline_manifest=Path(
            "/data/derivatives/qc/sc_matrix_qc/subject_manifest.csv"
        ),
    )
