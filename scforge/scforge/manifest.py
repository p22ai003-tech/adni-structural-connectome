from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np


DERIVED_DWI_TOKENS = (
    "adc",
    "trace",
    "fa",
    "fractional",
    "md",
    "rd",
    "ad",
    "exp",
    "colfa",
    "derived",
)


@dataclass(frozen=True)
class GradientQC:
    bval_path: str
    bvec_path: str
    n_bvals: int
    n_bvecs: int
    n_b0: int
    n_diffusion: int
    shells: tuple[int, ...]
    bvec_norm_min: float | None
    bvec_norm_max: float | None
    status: str
    reasons: tuple[str, ...]

    def to_dict(self) -> dict:
        data = asdict(self)
        data["shells"] = ";".join(str(x) for x in self.shells)
        data["reasons"] = ";".join(self.reasons)
        return data


@dataclass(frozen=True)
class SubjectSourceContract:
    subject: str
    nifti: str
    bval: str
    bvec: str
    json_sidecar: str
    is_derived_dwi: bool
    gradient_status: str
    n_volumes: int | None
    n_b0: int
    n_diffusion: int
    shells: tuple[int, ...]
    phase_encoding_direction: str
    total_readout_time: str
    status: str
    reasons: tuple[str, ...]

    def to_dict(self) -> dict:
        data = asdict(self)
        data["shells"] = ";".join(str(x) for x in self.shells)
        data["reasons"] = ";".join(self.reasons)
        return data


def is_probably_derived_dwi(path: str | Path) -> bool:
    name = Path(path).name.lower()
    return any(token in name for token in DERIVED_DWI_TOKENS)


def _read_bvals(path: Path) -> np.ndarray:
    values = np.loadtxt(path, dtype=float)
    return np.atleast_1d(values).reshape(-1)


def _read_bvecs(path: Path) -> np.ndarray:
    values = np.loadtxt(path, dtype=float)
    if values.ndim == 1:
        if values.size % 3 != 0:
            raise ValueError("1D bvec file length is not divisible by 3")
        values = values.reshape(3, values.size // 3)
    if values.shape[0] == 3:
        return values
    if values.shape[1] == 3:
        return values.T
    raise ValueError(f"bvec shape is not 3xN or Nx3: {values.shape}")


def validate_bvals_bvecs(
    bval_path: str | Path,
    bvec_path: str | Path,
    *,
    b0_threshold: float = 50.0,
    shell_round: int = 100,
) -> GradientQC:
    bval_path = Path(bval_path)
    bvec_path = Path(bvec_path)
    reasons: list[str] = []
    try:
        bvals = _read_bvals(bval_path)
        bvecs = _read_bvecs(bvec_path)
    except Exception as exc:
        return GradientQC(
            bval_path=str(bval_path),
            bvec_path=str(bvec_path),
            n_bvals=0,
            n_bvecs=0,
            n_b0=0,
            n_diffusion=0,
            shells=(),
            bvec_norm_min=None,
            bvec_norm_max=None,
            status="FAIL",
            reasons=(f"read_error:{type(exc).__name__}:{exc}",),
        )

    n_bvals = int(bvals.size)
    n_bvecs = int(bvecs.shape[1])
    if n_bvals != n_bvecs:
        reasons.append(f"length_mismatch_bvals_{n_bvals}_bvecs_{n_bvecs}")
    b0_mask = bvals <= b0_threshold
    n_b0 = int(b0_mask.sum())
    n_diffusion = int((~b0_mask).sum())
    if n_b0 == 0:
        reasons.append("no_b0")
    if n_diffusion == 0:
        reasons.append("no_diffusion_volumes")
    norms = np.linalg.norm(bvecs, axis=0)
    if n_bvals == n_bvecs:
        b0_norms = norms[b0_mask]
        dwi_norms = norms[~b0_mask]
        if b0_norms.size and float(np.nanmax(b0_norms)) > 0.2:
            reasons.append("b0_bvec_norm_not_near_zero")
        if dwi_norms.size and float(np.nanmedian(np.abs(dwi_norms - 1.0))) > 0.2:
            reasons.append("diffusion_bvec_norm_not_near_one")
    rounded_shells = sorted({int(round(float(v) / shell_round) * shell_round) for v in bvals if v > b0_threshold})
    return GradientQC(
        bval_path=str(bval_path),
        bvec_path=str(bvec_path),
        n_bvals=n_bvals,
        n_bvecs=n_bvecs,
        n_b0=n_b0,
        n_diffusion=n_diffusion,
        shells=tuple(rounded_shells),
        bvec_norm_min=float(np.nanmin(norms)) if norms.size else None,
        bvec_norm_max=float(np.nanmax(norms)) if norms.size else None,
        status="PASS" if not reasons else "FAIL",
        reasons=tuple(reasons),
    )


def _read_json_metadata(path: Path | None) -> dict:
    if not path:
        return {}
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def build_source_contract(
    *,
    subject: str,
    nifti: str | Path,
    bval: str | Path,
    bvec: str | Path,
    json_sidecar: str | Path | None = None,
    expected_n_volumes: int | None = None,
    reject_derived: bool = True,
) -> SubjectSourceContract:
    nifti = Path(nifti)
    bval = Path(bval)
    bvec = Path(bvec)
    json_path = Path(json_sidecar) if json_sidecar else None
    reasons: list[str] = []
    for label, path in (("nifti", nifti), ("bval", bval), ("bvec", bvec)):
        if not path.exists():
            reasons.append(f"missing_{label}")
    if json_path and not json_path.exists():
        reasons.append("missing_json_sidecar")
    derived = is_probably_derived_dwi(nifti)
    if reject_derived and derived:
        reasons.append("derived_dwi_name")

    gradient = validate_bvals_bvecs(bval, bvec) if bval.exists() and bvec.exists() else GradientQC(
        bval_path=str(bval),
        bvec_path=str(bvec),
        n_bvals=0,
        n_bvecs=0,
        n_b0=0,
        n_diffusion=0,
        shells=(),
        bvec_norm_min=None,
        bvec_norm_max=None,
        status="FAIL",
        reasons=("missing_gradient_file",),
    )
    if gradient.status != "PASS":
        reasons.extend(f"gradient_{reason}" for reason in gradient.reasons)
    if expected_n_volumes is not None and gradient.n_bvals and gradient.n_bvals != expected_n_volumes:
        reasons.append(f"volume_gradient_mismatch_{expected_n_volumes}_{gradient.n_bvals}")

    meta = _read_json_metadata(json_path)
    pe_dir = str(meta.get("PhaseEncodingDirection", ""))
    readout = str(meta.get("TotalReadoutTime", ""))
    if not pe_dir:
        reasons.append("missing_phase_encoding_direction")
    if not readout:
        reasons.append("missing_total_readout_time")

    return SubjectSourceContract(
        subject=subject,
        nifti=str(nifti),
        bval=str(bval),
        bvec=str(bvec),
        json_sidecar=str(json_path or ""),
        is_derived_dwi=derived,
        gradient_status=gradient.status,
        n_volumes=expected_n_volumes,
        n_b0=gradient.n_b0,
        n_diffusion=gradient.n_diffusion,
        shells=gradient.shells,
        phase_encoding_direction=pe_dir,
        total_readout_time=readout,
        status="PASS" if not reasons else "FAIL",
        reasons=tuple(dict.fromkeys(reasons)),
    )
