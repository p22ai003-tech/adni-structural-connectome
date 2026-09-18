from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FiveTTCommands:
    five_tt_primary: tuple[str, ...]
    five_tt_fallback: tuple[str, ...]
    primary_to_b0: tuple[str, ...]
    fallback_to_b0: tuple[str, ...]
    gmwmi_primary: tuple[str, ...]
    gmwmi_fallback: tuple[str, ...]


def build_5ttgen_freesurfer_command(subjects_dir_subject: Path, out_5tt_t1: Path) -> tuple[str, ...]:
    return ("5ttgen", "freesurfer", str(subjects_dir_subject), str(out_5tt_t1), "-nocrop")


def build_5ttgen_fsl_command(t1_native: Path, out_5tt_t1: Path) -> tuple[str, ...]:
    return ("5ttgen", "fsl", str(t1_native), str(out_5tt_t1), "-nocrop")


def build_5tt_to_b0_command(five_tt_t1: Path, t1_to_b0_mrtrix: Path, out_5tt_b0: Path) -> tuple[str, ...]:
    return ("mrtransform", str(five_tt_t1), str(out_5tt_b0), "-linear", str(t1_to_b0_mrtrix))


def build_gmwmi_command(five_tt_b0: Path, out_gmwmi: Path) -> tuple[str, ...]:
    return ("5tt2gmwmi", str(five_tt_b0), str(out_gmwmi))


def build_five_tt_command_set(
    *,
    subjects_dir_subject: Path,
    t1_native: Path,
    t1_to_b0_mrtrix: Path,
    out_dir: Path,
) -> FiveTTCommands:
    primary_t1 = out_dir / "5tt_t1_freesurfer.mif"
    fallback_t1 = out_dir / "5tt_t1_fsl.mif"
    primary_b0 = out_dir / "5tt_b0_freesurfer.mif"
    fallback_b0 = out_dir / "5tt_b0_fsl.mif"
    return FiveTTCommands(
        five_tt_primary=build_5ttgen_freesurfer_command(subjects_dir_subject, primary_t1),
        five_tt_fallback=build_5ttgen_fsl_command(t1_native, fallback_t1),
        primary_to_b0=build_5tt_to_b0_command(primary_t1, t1_to_b0_mrtrix, primary_b0),
        fallback_to_b0=build_5tt_to_b0_command(fallback_t1, t1_to_b0_mrtrix, fallback_b0),
        gmwmi_primary=build_gmwmi_command(primary_b0, out_dir / "gmwmi_b0_freesurfer.mif"),
        gmwmi_fallback=build_gmwmi_command(fallback_b0, out_dir / "gmwmi_b0_fsl.mif"),
    )
