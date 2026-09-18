#!/usr/bin/env python3
"""Canonical EC2 path resolver for the structural-connectome workspace."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class PipelinePaths:
    project_root: Path
    data_root: Path
    raw_images_root: Path
    raw_dwi_root: Path
    raw_t1_root: Path
    deriv_root: Path
    cohort_dir: Path
    cohort_dti_csv: Path
    cohort_mri_csv: Path
    atlas_root: Path
    aal_root: Path
    aal_mni: Path

    def as_dict(self) -> dict[str, str]:
        return {key: str(value) for key, value in asdict(self).items()}

    def format_summary(self) -> str:
        lines = ["Pipeline paths"]
        lines.append("--------------")
        for key, value in self.as_dict().items():
            lines.append(f"{key:<16}: {value}")
        return "\n".join(lines)


def resolve_pipeline_paths(
    project_root: str | Path | None = None,
    *,
    create_layout: bool = False,
) -> PipelinePaths:
    root = Path(project_root) if project_root is not None else Path.home() / "exp"
    root = root.expanduser().resolve()

    data_root = root / "data"
    raw_images_root = data_root / "Images"
    raw_dwi_root = raw_images_root / "dti"
    raw_t1_root = raw_images_root / "mri"
    deriv_root = data_root / "derivatives"
    cohort_dir = root / "cohort"
    atlas_root = root / "atlas"
    aal_root = atlas_root / "AAL"

    if create_layout:
        for path in (
            root,
            data_root,
            raw_images_root,
            raw_dwi_root,
            raw_t1_root,
            deriv_root,
            cohort_dir,
            atlas_root,
            aal_root,
        ):
            path.mkdir(parents=True, exist_ok=True)

    return PipelinePaths(
        project_root=root,
        data_root=data_root,
        raw_images_root=raw_images_root,
        raw_dwi_root=raw_dwi_root,
        raw_t1_root=raw_t1_root,
        deriv_root=deriv_root,
        cohort_dir=cohort_dir,
        cohort_dti_csv=cohort_dir / "dti.csv",
        cohort_mri_csv=cohort_dir / "mri.csv",
        atlas_root=atlas_root,
        aal_root=aal_root,
        aal_mni=aal_root / "AAL3v1_1mm.nii.gz",
    )
