from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


GROUP_ORDER: List[str] = ["CN", "MCI", "AD"]
GROUP_PALETTE: Dict[str, str] = {
    "CN": "#1f77b4",
    "MCI": "#ff7f0e",
    "AD": "#d62728",
}


def _sc_paths():
    """Resolve project paths via the canonical resolver at the repository root.

    Imported lazily and defensively: this module is also used from scripts that
    run with a different working directory, and a missing resolver must not stop
    the analysis package from importing.
    """
    try:
        import sc_config  # noqa: PLC0415
    except ModuleNotFoundError:  # pragma: no cover - fallback for odd sys.path
        import sys

        root = Path(__file__).resolve().parents[1]
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        import sc_config  # noqa: PLC0415
    return sc_config.paths()


def _resolve_default_cohort_csv(name: str) -> Path:
    """Cohort CSV location. No foreign-machine fallback: SC_COHORT_DIR governs."""
    return _sc_paths().cohort_dir / name


def _default_deriv_root() -> Path:
    """Derivatives root. SC_DERIV_ROOT still wins, as it always did."""
    return _sc_paths().deriv_root


@dataclass
class AnalysisPaths:
    notebook_dir: Path
    deriv_root: Path = field(default_factory=_default_deriv_root)
    cohort_dti_csv: Path = field(
        default_factory=lambda: _resolve_default_cohort_csv("dti.csv")
    )
    cohort_mri_csv: Path = field(
        default_factory=lambda: _resolve_default_cohort_csv("mri.csv")
    )
    analysis_root_name: str = "analysis_cohort"
    # Explicit locations. When left as None they are derived from deriv_root,
    # which is what notebooks that pass deriv_root expect. get_analysis_paths()
    # fills them from sc_config when no deriv_root is passed, so SC_QC_ROOT,
    # SC_CONNECTOMES_DIR and SC_ANALYSIS_ROOT are honoured. Before this, the
    # analysis always wrote to <deriv_root>/qc/analysis_cohort whatever
    # SC_ANALYSIS_ROOT said, while the build_*.py subprocesses followed
    # sc_config -- two mechanisms that could point at different trees.
    qc_root: Path | None = None
    connectomes_root: Path | None = None
    analysis_root: Path | None = None

    @property
    def qc_dir(self) -> Path:
        return self.qc_root or self.deriv_root / "qc"

    @property
    def connectomes_dir(self) -> Path:
        return self.connectomes_root or self.deriv_root / "connectomes"

    @property
    def legacy_analysis_dir(self) -> Path:
        return self.qc_dir / "analysis"

    @property
    def output_root(self) -> Path:
        return self.analysis_root or self.qc_dir / self.analysis_root_name

    @property
    def figures_dir(self) -> Path:
        return self.output_root / "figures"

    @property
    def tables_dir(self) -> Path:
        return self.output_root / "tables"

    @property
    def exports_dir(self) -> Path:
        return self.output_root / "exports"

    @property
    def logs_dir(self) -> Path:
        return self.output_root / "logs"

    @property
    def master_dir(self) -> Path:
        return self.output_root / "00_master"

    @property
    def appendix_dir(self) -> Path:
        return self.output_root / "99_appendix"

    @property
    def metrics_candidates(self) -> List[Path]:
        return [
            self.qc_dir / "metrics_enriched.xlsx",
            self.qc_dir / "metrics_excel_new.xlsx",
            self.qc_dir / "metrics_excel.xlsx",
        ]

    def section_dir(self, prefix: str, slug: str) -> Path:
        return self.output_root / f"{prefix}_{slug}"

    def ensure(self) -> "AnalysisPaths":
        for path in [
            self.output_root,
            self.figures_dir,
            self.tables_dir,
            self.exports_dir,
            self.logs_dir,
            self.master_dir,
            self.appendix_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)
        return self


def get_analysis_paths(
    notebook_dir: str | Path | None = None,
    deriv_root: str | Path | None = None,
    cohort_dti_csv: str | Path | None = None,
    cohort_mri_csv: str | Path | None = None,
) -> AnalysisPaths:
    notebook_dir = Path(notebook_dir) if notebook_dir else Path(__file__).resolve().parent.parent
    paths = AnalysisPaths(notebook_dir=notebook_dir)
    if deriv_root:
        # An explicit derivatives root wins and everything is derived from it.
        paths.deriv_root = Path(deriv_root)
    else:
        # Otherwise take every location from sc_config, which honours the
        # SC_* environment variables, so the in-process stages and the
        # build_*.py subprocesses resolve exactly the same tree.
        p = _sc_paths()
        paths.qc_root = p.qc_root
        paths.connectomes_root = p.connectomes_dir
        paths.analysis_root = p.analysis_root
    if cohort_dti_csv:
        paths.cohort_dti_csv = Path(cohort_dti_csv)
    if cohort_mri_csv:
        paths.cohort_mri_csv = Path(cohort_mri_csv)
    return paths.ensure()


def apply_plot_theme() -> None:
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.figsize": (8.5, 5.5),
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "font.family": "DejaVu Sans",
            "savefig.dpi": 300,
        }
    )
