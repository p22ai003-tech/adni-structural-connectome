from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class AffineSanityQC:
    path: str
    readable: bool
    finite: bool
    shape_ok: bool
    determinant: float | None
    translation_norm: float | None
    status: str
    reasons: tuple[str, ...]

    def to_dict(self) -> dict:
        data = asdict(self)
        data["reasons"] = ";".join(self.reasons)
        return data


def qc_affine_matrix(
    path: str | Path,
    *,
    min_abs_det: float = 1e-6,
    max_abs_det: float = 1e6,
    max_translation_norm: float = 500.0,
) -> AffineSanityQC:
    """Basic sanity gate for 4x4 spatial transform matrices.

    This is intentionally conservative: it catches empty, malformed, singular,
    non-finite, or wildly translated transforms before they can enter the
    spatial-contract route selector. It does not claim anatomical alignment by
    itself; mask/label overlap gates remain responsible for that.
    """
    path = Path(path)
    reasons: list[str] = []
    try:
        matrix = np.loadtxt(path, dtype=float)
    except Exception as exc:
        return AffineSanityQC(
            path=str(path),
            readable=False,
            finite=False,
            shape_ok=False,
            determinant=None,
            translation_norm=None,
            status="FAIL",
            reasons=(f"read_error:{type(exc).__name__}:{exc}",),
        )

    shape_ok = matrix.shape == (4, 4)
    if not shape_ok:
        reasons.append(f"shape_{matrix.shape}_expected_4x4")
    finite = bool(np.isfinite(matrix).all())
    if not finite:
        reasons.append("nonfinite")
    if shape_ok and finite:
        determinant = float(np.linalg.det(matrix[:3, :3]))
        translation_norm = float(np.linalg.norm(matrix[:3, 3]))
        if abs(determinant) < min_abs_det:
            reasons.append("singular_or_near_singular")
        if abs(determinant) > max_abs_det:
            reasons.append("determinant_extreme")
        if translation_norm > max_translation_norm:
            reasons.append("translation_extreme")
    else:
        determinant = None
        translation_norm = None

    return AffineSanityQC(
        path=str(path),
        readable=True,
        finite=finite,
        shape_ok=shape_ok,
        determinant=determinant,
        translation_norm=translation_norm,
        status="PASS" if not reasons else "FAIL",
        reasons=tuple(reasons),
    )


def build_flirt6_t1_to_b0_command(t1_brain: Path, mean_b0: Path, out_mat: Path, out_image: Path) -> tuple[str, ...]:
    return (
        "flirt",
        "-in",
        str(t1_brain),
        "-ref",
        str(mean_b0),
        "-omat",
        str(out_mat),
        "-out",
        str(out_image),
        "-dof",
        "6",
        "-cost",
        "mutualinfo",
    )


def build_flirt_bbr_command(t1_brain: Path, mean_b0: Path, wmseg: Path, out_mat: Path, out_image: Path) -> tuple[str, ...]:
    return (
        "flirt",
        "-in",
        str(t1_brain),
        "-ref",
        str(mean_b0),
        "-omat",
        str(out_mat),
        "-out",
        str(out_image),
        "-dof",
        "6",
        "-cost",
        "bbr",
        "-wmseg",
        str(wmseg),
    )


def build_convert_xfm_inverse_command(in_mat: Path, out_mat: Path) -> tuple[str, ...]:
    return ("convert_xfm", "-omat", str(out_mat), "-inverse", str(in_mat))


def build_transformconvert_flirt_to_mrtrix_command(
    in_mat: Path,
    moving: Path,
    reference: Path,
    out_txt: Path,
) -> tuple[str, ...]:
    return (
        "transformconvert",
        str(in_mat),
        str(moving),
        str(reference),
        "flirt_import",
        str(out_txt),
    )


def build_ants_registration_mni_to_t1_command(
    *,
    fixed_t1: Path,
    moving_mni: Path,
    out_prefix: Path,
    fixed_mask: Path | None = None,
) -> tuple[str, ...]:
    command = [
        "antsRegistrationSyN.sh",
        "-d",
        "3",
        "-f",
        str(fixed_t1),
        "-m",
        str(moving_mni),
        "-o",
        str(out_prefix),
        "-t",
        "s",
    ]
    if fixed_mask:
        command.extend(["-x", str(fixed_mask)])
    return tuple(command)


def build_ants_apply_transforms_command(
    *,
    moving: Path,
    reference: Path,
    output: Path,
    transforms: list[Path],
    interpolation: str = "NearestNeighbor",
) -> tuple[str, ...]:
    command = [
        "antsApplyTransforms",
        "-d",
        "3",
        "-i",
        str(moving),
        "-r",
        str(reference),
        "-o",
        str(output),
        "-n",
        interpolation,
    ]
    for transform in transforms:
        command.extend(["-t", str(transform)])
    return tuple(command)


def build_mrtransform_linear_command(
    moving: Path,
    output: Path,
    linear_transform: Path,
    *,
    template: Path | None = None,
    interp: str | None = None,
) -> tuple[str, ...]:
    command = ["mrtransform", str(moving), str(output), "-linear", str(linear_transform)]
    if template:
        command.extend(["-template", str(template)])
    if interp:
        command.extend(["-interp", interp])
    return tuple(command)
