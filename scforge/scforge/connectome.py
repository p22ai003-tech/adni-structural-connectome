from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .tractography import assignment_variant_args


REQUIRED_MATRICES = (
    "count",
    "fd_sum",
    "count_invnodevol",
    "len_mean",
    "invlen_mean",
    "fa_mean",
    "md_mean",
    "rd_mean",
    "ad_mean",
)


def derive_count_invnodevol(
    count: np.ndarray,
    node_volumes_mm3: np.ndarray,
    *,
    integer_tolerance: float = 1e-6,
    symmetry_tolerance: float = 1e-10,
) -> np.ndarray:
    """Derive raw-count inverse-node-volume weights without image-grid assumptions.

    For an off-diagonal edge ``(i, j)``, the result is
    ``2 * count[i, j] / (V_i + V_j)``, where the node volumes are physical
    volumes in mm3.  Inputs must already satisfy the frozen raw-count contract;
    this function deliberately does not crop, pad, symmetrise, round, clip, or
    impute them.
    """

    count_array = np.asarray(count)
    volume_array = np.asarray(node_volumes_mm3, dtype=np.float64)

    if count_array.ndim != 2 or count_array.shape[0] != count_array.shape[1]:
        raise ValueError("count must be an exact square matrix")
    if volume_array.ndim != 1 or volume_array.shape[0] != count_array.shape[0]:
        raise ValueError("node_volumes_mm3 must have exactly one value per count matrix node")
    if not np.issubdtype(count_array.dtype, np.number):
        raise TypeError("count must be numeric")

    count_float = count_array.astype(np.float64, copy=False)
    if not np.isfinite(count_float).all():
        raise ValueError("count must contain only finite values")
    if np.any(count_float < 0):
        raise ValueError("count must be nonnegative")
    if not np.allclose(count_float, np.rint(count_float), rtol=0.0, atol=integer_tolerance):
        raise ValueError("count must be integer-valued within tolerance")
    if not np.allclose(count_float, count_float.T, rtol=0.0, atol=symmetry_tolerance):
        raise ValueError("count must be symmetric")
    if not np.allclose(np.diag(count_float), 0.0, rtol=0.0, atol=integer_tolerance):
        raise ValueError("count must have a zero diagonal")

    if not np.isfinite(volume_array).all() or np.any(volume_array <= 0):
        raise ValueError("node_volumes_mm3 must contain only finite positive physical volumes")

    denominators = volume_array[:, None] + volume_array[None, :]
    return 2.0 * count_float / denominators


@dataclass(frozen=True)
class ConnectomeCommands:
    count: tuple[str, ...]
    fd_sum: tuple[str, ...]
    len_mean: tuple[str, ...]
    invlen_mean: tuple[str, ...]
    fa_sample: tuple[str, ...]
    fa_mean: tuple[str, ...]
    md_sample: tuple[str, ...]
    md_mean: tuple[str, ...]
    rd_sample: tuple[str, ...]
    rd_mean: tuple[str, ...]
    ad_sample: tuple[str, ...]
    ad_mean: tuple[str, ...]
    matrix_outputs: tuple[tuple[str, Path], ...]

    def to_records(self) -> list[dict[str, str]]:
        """Return executable commands while keeping output metadata declarative."""

        data = asdict(self)
        data.pop("matrix_outputs")
        return [{"stage": key, "command": " ".join(value)} for key, value in data.items()]

    def output_path(self, matrix_name: str) -> Path:
        """Return the declared path for one of the nine frozen matrix outputs."""

        try:
            return dict(self.matrix_outputs)[matrix_name]
        except KeyError as exc:
            raise KeyError(f"Unknown matrix output: {matrix_name}") from exc


def build_tck2connectome_command(
    tracks: Path,
    nodes: Path,
    out_csv: Path,
    *,
    assignment_variant: str = "radial4",
    tck_weights: Path | None = None,
    scale_file: Path | None = None,
    scale_length: bool = False,
    scale_invlength: bool = False,
    stat_edge: str | None = None,
    assignments_out: Path | None = None,
) -> tuple[str, ...]:
    command = [
        "tck2connectome",
        str(tracks),
        str(nodes),
        str(out_csv),
        "-symmetric",
        "-zero_diagonal",
        *assignment_variant_args(assignment_variant),
    ]
    if tck_weights:
        command.extend(["-tck_weights_in", str(tck_weights)])
    if scale_file:
        command.extend(["-scale_file", str(scale_file)])
    if scale_length:
        command.append("-scale_length")
    if scale_invlength:
        command.append("-scale_invlength")
    if stat_edge:
        command.extend(["-stat_edge", stat_edge])
    if assignments_out:
        command.extend(["-out_assignments", str(assignments_out)])
    return tuple(command)


def build_tcksample_command(tracks: Path, metric_mif: Path, out_tsf: Path) -> tuple[str, ...]:
    return ("tcksample", str(tracks), str(metric_mif), str(out_tsf), "-stat_tck", "mean")


def build_connectome_command_set(
    *,
    tracks: Path,
    nodes: Path,
    out_dir: Path,
    assignment_variant: str = "radial4",
    sift2_weights: Path | None = None,
    fa_mif: Path | None = None,
    md_mif: Path | None = None,
    rd_mif: Path | None = None,
    ad_mif: Path | None = None,
) -> ConnectomeCommands:
    if sift2_weights is None:
        raise ValueError("sift2_weights is required: fd_sum must be SIFT2-weighted")

    fa_tsf = out_dir / "fa_mean.tsf"
    md_tsf = out_dir / "md_mean.tsf"
    rd_tsf = out_dir / "rd_mean.tsf"
    ad_tsf = out_dir / "ad_mean.tsf"
    matrix_outputs = tuple((name, out_dir / f"SC_AAL_{name}.csv") for name in REQUIRED_MATRICES)
    output_by_name = dict(matrix_outputs)
    return ConnectomeCommands(
        count=build_tck2connectome_command(
            tracks,
            nodes,
            output_by_name["count"],
            assignment_variant=assignment_variant,
            stat_edge="sum",
            assignments_out=out_dir / "assignments_count.txt",
        ),
        fd_sum=build_tck2connectome_command(
            tracks,
            nodes,
            output_by_name["fd_sum"],
            assignment_variant=assignment_variant,
            tck_weights=sift2_weights,
            stat_edge="sum",
            assignments_out=out_dir / "assignments_fd_sum.txt",
        ),
        len_mean=build_tck2connectome_command(
            tracks,
            nodes,
            output_by_name["len_mean"],
            assignment_variant=assignment_variant,
            scale_length=True,
            stat_edge="mean",
        ),
        invlen_mean=build_tck2connectome_command(
            tracks,
            nodes,
            output_by_name["invlen_mean"],
            assignment_variant=assignment_variant,
            scale_invlength=True,
            stat_edge="mean",
        ),
        fa_sample=build_tcksample_command(tracks, fa_mif or Path("fa.mif"), fa_tsf),
        fa_mean=build_tck2connectome_command(
            tracks,
            nodes,
            output_by_name["fa_mean"],
            assignment_variant=assignment_variant,
            scale_file=fa_tsf,
            stat_edge="mean",
        ),
        md_sample=build_tcksample_command(tracks, md_mif or Path("md.mif"), md_tsf),
        md_mean=build_tck2connectome_command(
            tracks,
            nodes,
            output_by_name["md_mean"],
            assignment_variant=assignment_variant,
            scale_file=md_tsf,
            stat_edge="mean",
        ),
        rd_sample=build_tcksample_command(tracks, rd_mif or Path("rd.mif"), rd_tsf),
        rd_mean=build_tck2connectome_command(
            tracks,
            nodes,
            output_by_name["rd_mean"],
            assignment_variant=assignment_variant,
            scale_file=rd_tsf,
            stat_edge="mean",
        ),
        ad_sample=build_tcksample_command(tracks, ad_mif or Path("ad.mif"), ad_tsf),
        ad_mean=build_tck2connectome_command(
            tracks,
            nodes,
            output_by_name["ad_mean"],
            assignment_variant=assignment_variant,
            scale_file=ad_tsf,
            stat_edge="mean",
        ),
        matrix_outputs=matrix_outputs,
    )
