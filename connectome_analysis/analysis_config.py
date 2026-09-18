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


def _resolve_default_cohort_csv(name: str) -> Path:
    candidates = [
        Path.cwd() / "cohort" / name,
        Path("/Users/sabeesh/Desktop/exp/cohort") / name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[-1]


@dataclass
class AnalysisPaths:
    notebook_dir: Path
    deriv_root: Path = Path(
        os.environ.get("SC_DERIV_ROOT", "/Volumes/Seagate Hub/ADNI_Images/derivatives")
    )
    cohort_dti_csv: Path = field(
        default_factory=lambda: _resolve_default_cohort_csv("dti.csv")
    )
    cohort_mri_csv: Path = field(
        default_factory=lambda: _resolve_default_cohort_csv("mri.csv")
    )
    analysis_root_name: str = "analysis_cohort"

    @property
    def qc_dir(self) -> Path:
        return self.deriv_root / "qc"

    @property
    def connectomes_dir(self) -> Path:
        return self.deriv_root / "connectomes"

    @property
    def legacy_analysis_dir(self) -> Path:
        return self.qc_dir / "analysis"

    @property
    def output_root(self) -> Path:
        return self.qc_dir / self.analysis_root_name

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
        paths.deriv_root = Path(deriv_root)
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
