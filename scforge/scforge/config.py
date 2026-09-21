from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class AtlasConfig:
    name: str
    mni_image: Path
    labels_csv: Path
    max_original_label: int = 170
    expected_gaps: tuple[int, ...] = (35, 36, 81, 82)
    force_contiguous_nodes: bool = True
    required_labels: tuple[int, ...] = (7, 8)


@dataclass(frozen=True)
class DwiConfig:
    prefer_original_axial: bool = True
    reject_derived: bool = True
    pe_dir_default: str = "j-"
    eddy_options: str = "--slm=linear --data_is_shelled --repol"


@dataclass(frozen=True)
class SpatialRouteConfig:
    name: str
    promote: bool
    mni_to_t1: str | None = None
    t1_to_b0: str | None = None
    mni_to_b0: str | None = None


@dataclass(frozen=True)
class GateConfig:
    density_fd_sum_min: float = 0.15
    valid_zero_rows_max: int = 0
    missing_valid_labels_max: int = 0
    unique_assigned_nodes_min: int = 140
    top5_endpoint_fraction_max: float = 0.35


@dataclass(frozen=True)
class QCConfig:
    min_label_voxels_1mm: int = 50
    required_valid_nodes: int = 166
    density_fd_sum_min: float = 0.15
    strict_zero_valid_rows: int = 0
    warn_valid_zero_rows: int = 0
    min_endpoint_assignment_frac: float = 0.85
    min_label_inside_mask_frac: float = 0.80
    min_5tt_mask_dice: float = 0.70
    candidate_pass: GateConfig = field(
        default_factory=lambda: GateConfig(
            density_fd_sum_min=0.15,
            valid_zero_rows_max=5,
            missing_valid_labels_max=0,
            unique_assigned_nodes_min=120,
            top5_endpoint_fraction_max=0.50,
        )
    )
    publication_pass: GateConfig = field(default_factory=GateConfig)


@dataclass(frozen=True)
class TractographyConfig:
    preflight_streamlines: int = 500_000
    full_streamlines: int = 10_000_000
    algorithm: str = "iFOD2"
    maxlength: int = 250
    cutoff: float = 0.06
    assignment_ladder: tuple[str, ...] = ("radial4", "radial8", "reverse20", "forward40")


@dataclass(frozen=True)
class PublicationConfig:
    overwrite_existing: bool = False
    publish_warn: bool = False
    analysis_ready_statuses: tuple[str, ...] = ("PASS",)


@dataclass(frozen=True)
class SCForgeConfig:
    project_root: Path
    derivatives_root: Path
    existing_derivatives_root: Path
    atlas: AtlasConfig
    dwi: DwiConfig = field(default_factory=DwiConfig)
    spatial_routes: tuple[SpatialRouteConfig, ...] = ()
    qc: QCConfig = field(default_factory=QCConfig)
    tractography: TractographyConfig = field(default_factory=TractographyConfig)
    publication: PublicationConfig = field(default_factory=PublicationConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SCForgeConfig":
        atlas = data.get("atlas", {})
        dwi = data.get("dwi", {})
        qc = data.get("qc", {})
        tractography = data.get("tractography", {})
        publication = data.get("publication", {})
        routes = tuple(SpatialRouteConfig(**route) for route in data.get("spatial_routes", []))
        return cls(
            project_root=Path(data["project_root"]).expanduser(),
            derivatives_root=Path(data["derivatives_root"]).expanduser(),
            existing_derivatives_root=Path(data["existing_derivatives_root"]).expanduser(),
            atlas=AtlasConfig(
                name=str(atlas["name"]),
                mni_image=Path(atlas["mni_image"]).expanduser(),
                labels_csv=Path(atlas["labels_csv"]).expanduser(),
                max_original_label=int(atlas.get("max_original_label", 170)),
                expected_gaps=tuple(int(x) for x in atlas.get("expected_gaps", [35, 36, 81, 82])),
                force_contiguous_nodes=bool(atlas.get("force_contiguous_nodes", True)),
                required_labels=tuple(int(x) for x in atlas.get("required_labels", [7, 8])),
            ),
            dwi=DwiConfig(**dwi),
            spatial_routes=routes,
            qc=_parse_qc_config(qc),
            tractography=TractographyConfig(
                preflight_streamlines=int(tractography.get("preflight_streamlines", 500000)),
                full_streamlines=int(tractography.get("full_streamlines", 10000000)),
                algorithm=str(tractography.get("algorithm", "iFOD2")),
                maxlength=int(tractography.get("maxlength", 250)),
                cutoff=float(tractography.get("cutoff", 0.06)),
                assignment_ladder=tuple(str(x) for x in tractography.get("assignment_ladder", [])),
            ),
            publication=PublicationConfig(
                overwrite_existing=bool(publication.get("overwrite_existing", False)),
                publish_warn=bool(publication.get("publish_warn", False)),
                analysis_ready_statuses=tuple(publication.get("analysis_ready_statuses", ["PASS"])),
            ),
        )


def _parse_gate_config(data: dict[str, Any], default: GateConfig) -> GateConfig:
    if not isinstance(data, dict):
        data = {}
    return GateConfig(
        density_fd_sum_min=float(data.get("density_fd_sum_min", data.get("density_min", default.density_fd_sum_min))),
        valid_zero_rows_max=int(data.get("valid_zero_rows_max", default.valid_zero_rows_max)),
        missing_valid_labels_max=int(data.get("missing_valid_labels_max", data.get("missing_labels_max", default.missing_valid_labels_max))),
        unique_assigned_nodes_min=int(data.get("unique_assigned_nodes_min", default.unique_assigned_nodes_min)),
        top5_endpoint_fraction_max=float(data.get("top5_endpoint_fraction_max", default.top5_endpoint_fraction_max)),
    )


def _parse_qc_config(data: dict[str, Any]) -> QCConfig:
    if not isinstance(data, dict):
        data = {}
    base = {
        "min_label_voxels_1mm": int(data.get("min_label_voxels_1mm", 50)),
        "required_valid_nodes": int(data.get("required_valid_nodes", 166)),
        "density_fd_sum_min": float(data.get("density_fd_sum_min", 0.15)),
        "strict_zero_valid_rows": int(data.get("strict_zero_valid_rows", 0)),
        "warn_valid_zero_rows": int(data.get("warn_valid_zero_rows", 0)),
        "min_endpoint_assignment_frac": float(data.get("min_endpoint_assignment_frac", 0.85)),
        "min_label_inside_mask_frac": float(data.get("min_label_inside_mask_frac", 0.80)),
        "min_5tt_mask_dice": float(data.get("min_5tt_mask_dice", 0.70)),
    }
    candidate_default = QCConfig().candidate_pass
    publication_default = QCConfig().publication_pass
    return QCConfig(
        **base,
        candidate_pass=_parse_gate_config(data.get("candidate_pass", {}), candidate_default),
        publication_pass=_parse_gate_config(data.get("publication_pass", {}), publication_default),
    )


def load_config(path: str | Path) -> SCForgeConfig:
    path = Path(path).expanduser()
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Config did not parse as a mapping: {path}")
    # ${repo} is the repository root, wherever it was cloned
    from scforge.environment import expand

    return SCForgeConfig.from_dict(expand(data, None))
