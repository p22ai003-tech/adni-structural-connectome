"""Single source of truth for every path this project needs.

Why this exists
---------------
The codebase grew on four different machines and carries 1,940 occurrences of
``/home/ec2-user`` across 453 files, plus defaults pointing at a Mac
(``/Volumes/Seagate Hub``, ``/Users/sabeesh``) and at a third machine
(``/home/pc``). None of those resolve anywhere else, so the project only runs
here, and only because a few environment variables happen to be set.

Everything is derived from at most three roots, and every value can be
overridden by one environment variable. Nothing here reads a file or creates a
directory at import time: resolution is pure, so importing this module can never
have a side effect on a machine where the paths do not exist.

The eleven names
----------------
    SC_PROJECT_ROOT     the repository (default: the directory holding this file)
    SC_DATA_ROOT        the data volume                  (default: $SC_PROJECT_ROOT/data)
    SC_RAW_IMAGES_ROOT  raw ADNI image tree              (default: $SC_DATA_ROOT/Images)
    SC_DERIV_ROOT       derivatives                      (default: $SC_DATA_ROOT/derivatives)
    SC_QC_ROOT          QC tree                          (default: $SC_DERIV_ROOT/qc)
    SC_ANALYSIS_ROOT    analysis outputs                 (default: $SC_QC_ROOT/analysis_cohort)
    SC_CONNECTOMES_DIR  connectome matrices              (default: $SC_DERIV_ROOT/connectomes)
    SC_COHORT_DIR       cohort CSVs (ADNI-restricted)    (default: $SC_PROJECT_ROOT/cohort)
    SC_ATLAS_ROOT       atlas assets                     (default: $SC_PROJECT_ROOT/atlas)
    SC_RUN_STATE_DIR    run state, replaces /tmp files   (default: $SC_QC_ROOT/run_state)
    SC_MANIFEST         acquisition manifest CSV         (default: $SC_PROJECT_ROOT/configs/acquisition_manifest.csv)

External toolchains use their own conventional variables (FSLDIR, ANTSPATH,
MRTRIX_BIN) and fall back to PATH lookup rather than to an absolute path.

Usage
-----
    from sc_config import paths, tools
    p = paths()
    p.connectomes_dir / f"SC_AAL166_{sid}_fd_sum.csv"
    tools().mrtrix("tckgen")
"""

from __future__ import annotations

import os
import shutil
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

__all__ = ["Paths", "Tools", "paths", "tools", "describe"]


def _env_path(name: str, default: Path) -> Path:
    """Read an override from the environment, else use the derived default.

    Deliberately does not resolve symlinks or require existence: a packaged run
    may point at a volume that is not mounted yet, and that should fail where the
    path is used, with a useful message, not at import.
    """
    raw = os.environ.get(name)
    return Path(raw).expanduser() if raw else default


@dataclass(frozen=True)
class Paths:
    project_root: Path
    data_root: Path
    raw_images_root: Path
    raw_dwi_root: Path
    raw_t1_root: Path
    deriv_root: Path
    qc_root: Path
    analysis_root: Path
    connectomes_dir: Path
    cohort_dir: Path
    atlas_root: Path
    aal_root: Path
    run_state_dir: Path
    manifest: Path

    # -- atlas assets: shipped with the code, so these are always derivable ----
    @property
    def aal_nifti_166(self) -> Path:
        return self.aal_root / "AAL3v1_1mm_166.nii.gz"

    @property
    def aal_node_map(self) -> Path:
        """The CORRECT 166-node look-up table, keyed on matrix index.

        Prefer this over AAL3_labels.csv, which is keyed on the original atlas
        value and therefore mislabels every matrix row from 37 onward.
        """
        return self.aal_root / "aal3_node_map_166.csv"

    @property
    def aal_network_mapping(self) -> Path:
        return self.aal_root / "AAL3_network_mapping.csv"

    # -- cohort tables (ADNI-restricted; never redistributed) -----------------
    @property
    def cohort_dti_csv(self) -> Path:
        return self.cohort_dir / "dti.csv"

    @property
    def cohort_mri_csv(self) -> Path:
        return self.cohort_dir / "mri.csv"

    # -- analysis-tree sections -----------------------------------------------
    # The analysis tree is numbered by stage. These are the sections that the
    # dashboard reads from and that the build_*.py scripts must therefore write
    # to. Writing anywhere else is what used to require a manual copy step.
    @property
    def master_dir(self) -> Path:
        return self.analysis_root / "00_master"

    @property
    def edr_exceptions_dir(self) -> Path:
        return self.analysis_root / "17_edr_exceptions"

    @property
    def network_dir(self) -> Path:
        return self.analysis_root / "19_network_analysis"

    @property
    def functional_network_dir(self) -> Path:
        """Network-level tables, keyed on the AAL3 -> Yeo functional mapping."""
        return self.network_dir / "functional"

    @property
    def exception_dir(self) -> Path:
        """Exception-specificity and ML artifacts; the dashboard's ML section."""
        return self.analysis_root / "20_exception_specificity"

    @property
    def figs_dir(self) -> Path:
        return self.project_root / "docs" / "figs"

    @property
    def master_cohort_csv(self) -> Path:
        return self.analysis_root / "00_master" / "master_cohort.csv"

    @property
    def subject_manifest_csv(self) -> Path:
        return self.qc_root / "sc_matrix_qc" / "subject_manifest.csv"

    def as_dict(self) -> dict[str, str]:
        return {k: str(v) for k, v in asdict(self).items()}


@dataclass(frozen=True)
class Tools:
    """External binaries, resolved by convention rather than absolute path."""

    fsl_dir: Path | None
    mrtrix_bin: Path | None
    ants_path: Path | None

    def _resolve(self, base: Path | None, name: str) -> str:
        if base is not None:
            candidate = base / name
            if candidate.exists():
                return str(candidate)
        found = shutil.which(name)
        if found:
            return found
        raise FileNotFoundError(
            f"{name!r} not found. Set FSLDIR / MRTRIX_BIN / ANTSPATH, or put it on PATH."
        )

    def fsl(self, name: str) -> str:
        return self._resolve(self.fsl_dir / "bin" if self.fsl_dir else None, name)

    def mrtrix(self, name: str) -> str:
        return self._resolve(self.mrtrix_bin, name)

    def ants(self, name: str) -> str:
        return self._resolve(self.ants_path, name)

    @property
    def mni_t1_1mm(self) -> Path | None:
        if self.fsl_dir is None:
            return None
        return self.fsl_dir / "data" / "standard" / "MNI152_T1_1mm.nii.gz"

    @property
    def mni_t1_1mm_brain(self) -> Path | None:
        if self.fsl_dir is None:
            return None
        return self.fsl_dir / "data" / "standard" / "MNI152_T1_1mm_brain.nii.gz"


@lru_cache(maxsize=1)
def paths() -> Paths:
    project_root = _env_path("SC_PROJECT_ROOT", Path(__file__).resolve().parent)
    data_root = _env_path("SC_DATA_ROOT", project_root / "data")
    raw_images_root = _env_path("SC_RAW_IMAGES_ROOT", data_root / "Images")
    deriv_root = _env_path("SC_DERIV_ROOT", data_root / "derivatives")
    qc_root = _env_path("SC_QC_ROOT", deriv_root / "qc")
    atlas_root = _env_path("SC_ATLAS_ROOT", project_root / "atlas")
    return Paths(
        project_root=project_root,
        data_root=data_root,
        raw_images_root=raw_images_root,
        raw_dwi_root=raw_images_root / "dti",
        raw_t1_root=raw_images_root / "mri",
        deriv_root=deriv_root,
        qc_root=qc_root,
        analysis_root=_env_path("SC_ANALYSIS_ROOT", qc_root / "analysis_cohort"),
        connectomes_dir=_env_path("SC_CONNECTOMES_DIR", deriv_root / "connectomes"),
        cohort_dir=_env_path("SC_COHORT_DIR", project_root / "cohort"),
        atlas_root=atlas_root,
        aal_root=atlas_root / "AAL",
        run_state_dir=_env_path("SC_RUN_STATE_DIR", qc_root / "run_state"),
        manifest=_env_path("SC_MANIFEST", project_root / "configs" / "acquisition_manifest.csv"),
    )


@lru_cache(maxsize=1)
def tools() -> Tools:
    def _opt(name: str) -> Path | None:
        raw = os.environ.get(name)
        return Path(raw).expanduser() if raw else None

    mrtrix = _opt("MRTRIX_BIN")
    if mrtrix is None:
        found = shutil.which("tckgen")
        mrtrix = Path(found).parent if found else None
    return Tools(fsl_dir=_opt("FSLDIR"), mrtrix_bin=mrtrix, ants_path=_opt("ANTSPATH"))


def describe() -> str:
    """Render the resolved configuration, marking anything that is missing."""
    p = paths()
    lines = ["Resolved configuration", "=" * 60]
    for key, value in p.as_dict().items():
        mark = " " if Path(value).exists() else "  <- does not exist"
        lines.append(f"  {key:<20} {value}{mark}")
    t = tools()
    lines.append("")
    lines.append("Toolchains")
    lines.append("-" * 60)
    for label, value in (("FSLDIR", t.fsl_dir), ("MRTRIX_BIN", t.mrtrix_bin), ("ANTSPATH", t.ants_path)):
        lines.append(f"  {label:<20} {value if value else '(unset; will resolve via PATH)'}")
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover
    print(describe())
