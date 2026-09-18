from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import TractographyConfig


@dataclass(frozen=True)
class TractographyCommands:
    preflight_tckgen: tuple[str, ...]
    full_tckgen: tuple[str, ...]
    sift2: tuple[str, ...]


def build_tckgen_command(wmfod: Path, five_tt: Path, gmwmi: Path, out_tck: Path, config: TractographyConfig, *, preflight: bool) -> tuple[str, ...]:
    n_streamlines = config.preflight_streamlines if preflight else config.full_streamlines
    return (
        "tckgen",
        str(wmfod),
        str(out_tck),
        "-act",
        str(five_tt),
        "-backtrack",
        "-crop_at_gmwmi",
        "-seed_gmwmi",
        str(gmwmi),
        "-select",
        str(n_streamlines),
        "-maxlength",
        str(config.maxlength),
        "-cutoff",
        str(config.cutoff),
        "-algorithm",
        config.algorithm,
    )


def build_tcksift2_command(tracks: Path, wmfod: Path, five_tt: Path, out_weights: Path, out_mu: Path | None = None) -> tuple[str, ...]:
    command = ["tcksift2", str(tracks), str(wmfod), str(out_weights), "-act", str(five_tt)]
    if out_mu:
        command.extend(["-out_mu", str(out_mu)])
    return tuple(command)


def build_tractography_command_set(
    *,
    wmfod: Path,
    five_tt: Path,
    gmwmi: Path,
    out_dir: Path,
    config: TractographyConfig,
) -> TractographyCommands:
    preflight = out_dir / "tracks_preflight_500k.tck"
    full = out_dir / "tracks_10M.tck"
    weights = out_dir / "sift2_weights.csv"
    return TractographyCommands(
        preflight_tckgen=build_tckgen_command(wmfod, five_tt, gmwmi, preflight, config, preflight=True),
        full_tckgen=build_tckgen_command(wmfod, five_tt, gmwmi, full, config, preflight=False),
        sift2=build_tcksift2_command(full, wmfod, five_tt, weights, out_dir / "sift2_mu.txt"),
    )


def assignment_variant_args(variant: str) -> tuple[str, ...]:
    mapping = {
        "radial2": ("-assignment_radial_search", "2"),
        "radial4": ("-assignment_radial_search", "4"),
        "radial8": ("-assignment_radial_search", "8"),
        "reverse20": ("-assignment_reverse_search", "20"),
        "forward40": ("-assignment_forward_search", "40"),
        "forward80": ("-assignment_forward_search", "80"),
    }
    if variant not in mapping:
        raise ValueError(f"Unknown assignment variant: {variant}")
    return mapping[variant]
