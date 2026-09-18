from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from .config import DwiConfig


@dataclass(frozen=True)
class DwiPreprocCommands:
    mrconvert: tuple[str, ...]
    dwidenoise: tuple[str, ...]
    mrdegibbs: tuple[str, ...]
    dwifslpreproc: tuple[str, ...]
    dwibiascorrect: tuple[str, ...]
    mean_b0: tuple[str, ...]
    dwi_mask: tuple[str, ...]

    def to_records(self) -> list[dict]:
        return [{"stage": key, "command": " ".join(value)} for key, value in asdict(self).items()]


@dataclass(frozen=True)
class DwiPreprocPaths:
    out_dir: Path
    raw: Path
    denoised: Path
    noise: Path
    degibbs: Path
    preproc: Path
    biascorr: Path
    mean_b0: Path
    mask: Path


def dwi_preproc_paths(out_dir: Path) -> DwiPreprocPaths:
    return DwiPreprocPaths(
        out_dir=out_dir,
        raw=out_dir / "dwi_raw.mif",
        denoised=out_dir / "dwi_den.mif",
        noise=out_dir / "noise.mif",
        degibbs=out_dir / "dwi_den_degibbs.mif",
        preproc=out_dir / "dwi_preproc.mif",
        biascorr=out_dir / "dwi_preproc_biascorr.mif",
        mean_b0=out_dir / "mean_b0.mif",
        mask=out_dir / "dwi_mask.mif",
    )


def build_dwi_preproc_commands(
    *,
    nifti: Path,
    bvec: Path,
    bval: Path,
    json_sidecar: Path | None,
    out_dir: Path,
    config: DwiConfig,
    reverse_pe_epi: Path | None = None,
) -> DwiPreprocCommands:
    paths = dwi_preproc_paths(out_dir)
    mrconvert = ["mrconvert", str(nifti), str(paths.raw), "-fslgrad", str(bvec), str(bval)]
    if json_sidecar:
        mrconvert.extend(["-json_import", str(json_sidecar)])
    if reverse_pe_epi:
        dwifsl = [
            "dwifslpreproc",
            str(paths.degibbs),
            str(paths.preproc),
            "-rpe_pair",
            "-se_epi",
            str(reverse_pe_epi),
            "-pe_dir",
            config.pe_dir_default,
            "-eddy_options",
            config.eddy_options,
        ]
    else:
        dwifsl = [
            "dwifslpreproc",
            str(paths.degibbs),
            str(paths.preproc),
            "-rpe_none",
            "-pe_dir",
            config.pe_dir_default,
            "-eddy_options",
            config.eddy_options,
        ]
    return DwiPreprocCommands(
        mrconvert=tuple(mrconvert),
        dwidenoise=("dwidenoise", str(paths.raw), str(paths.denoised), "-noise", str(paths.noise)),
        mrdegibbs=("mrdegibbs", str(paths.denoised), str(paths.degibbs)),
        dwifslpreproc=tuple(dwifsl),
        dwibiascorrect=("dwibiascorrect", "ants", str(paths.preproc), str(paths.biascorr)),
        mean_b0=("bash", "-lc", f"dwiextract {paths.biascorr} - -bzero | mrmath - mean {paths.mean_b0} -axis 3"),
        dwi_mask=("dwi2mask", str(paths.biascorr), str(paths.mask)),
    )
