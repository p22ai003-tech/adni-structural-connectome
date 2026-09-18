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
    aal_node_map: Path
    # HCP379 is a separate workstream and is excluded from the package; these
    # may legitimately be absent, so consumers must tolerate None.
    hcp379_live_summary: Path | None
    hcp379_live_ledger: Path | None
    hcp379_release_topology: Path | None
    pipeline_status: Path
    pipeline_density: Path
    pipeline_manifest: Path
    table_default_limit: int = 200
    table_max_limit: int = 5_000


def _opt(path: Path) -> Path | None:
    """Return the path if it exists, else None (HCP379 assets are optional)."""
    return path if Path(path).exists() else None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    # Resolve through the canonical resolver; CONNECTOME_* overrides still win.
    try:
        import sc_config  # noqa: PLC0415
    except ModuleNotFoundError:  # pragma: no cover
        import sys

        _root = Path(__file__).resolve().parents[1]
        if str(_root) not in sys.path:
            sys.path.insert(0, str(_root))
        import sc_config  # noqa: PLC0415
    _p = sc_config.paths()
    project = Path(
        os.environ.get("CONNECTOME_PROJECT_ROOT", _p.project_root)
    ).resolve()
    analysis = Path(
        os.environ.get(
            "CONNECTOME_ANALYSIS_ROOT",
            _p.analysis_root,
        )
    ).resolve()
    connectomes = Path(
        os.environ.get(
            "CONNECTOME_MATRICES_DIR",
            _p.connectomes_dir,
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
        # Correct 166-node LUT: AAL3_labels.csv is keyed on atlas value and
        # mislabels every matrix row from 37 onward.
        aal_node_map=_p.aal_node_map,
        hcp379_live_summary=_opt(
            project
            / "research_audit/outputs/hcp379_live_530_status_density_v1/"
            "summary.json"
        ),
        hcp379_live_ledger=_opt(
            project
            / "research_audit/outputs/"
            "hcp379_pretract_failure_recovery_ledger_v1/ledger.csv"
        ),
        hcp379_release_topology=_opt(
            project
            / "research_audit/outputs/"
            "hcp379_balanced_release_topology_v1/topology.json"
        ),
        pipeline_status=_p.qc_root / "sc_matrix_qc" / "connectome_status.json",
        pipeline_density=_p.qc_root / "sc_matrix_qc" / "density_check.json",
        pipeline_manifest=_p.subject_manifest_csv,
    )
