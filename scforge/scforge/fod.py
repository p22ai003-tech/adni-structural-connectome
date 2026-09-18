from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FodCommands:
    response: tuple[str, ...]
    fod: tuple[str, ...]
    mtnormalise: tuple[str, ...]
    tensor: tuple[str, ...]
    tensor_metrics: tuple[str, ...]

def choose_fod_model(shells: list[int] | tuple[int, ...]) -> str:
    nonzero_shells = sorted({int(shell) for shell in shells if int(shell) > 50})
    if len(nonzero_shells) <= 1:
        return "ss3t_csd"
    return "msmt_csd"


def build_tensor_metric_commands(dwi_biascorr: str, mask: str, out_dir: str) -> list[tuple[str, ...]]:
    dt = f"{out_dir}/dt.mif"
    return [
        ("dwi2tensor", dwi_biascorr, dt, "-mask", mask),
        ("tensor2metric", dt, "-fa", f"{out_dir}/fa.mif", "-adc", f"{out_dir}/md.mif", "-ad", f"{out_dir}/ad.mif", "-rd", f"{out_dir}/rd.mif"),
    ]


def build_response_command(dwi_biascorr: Path, mask: Path, out_dir: Path) -> tuple[str, ...]:
    return (
        "dwi2response",
        "dhollander",
        str(dwi_biascorr),
        str(out_dir / "response_wm.txt"),
        str(out_dir / "response_gm.txt"),
        str(out_dir / "response_csf.txt"),
        "-mask",
        str(mask),
        "-voxels",
        str(out_dir / "response_voxels.mif"),
    )


def build_fod_command(dwi_biascorr: Path, mask: Path, out_dir: Path, *, model: str) -> tuple[str, ...]:
    if model == "ss3t_csd":
        return (
            "ss3t_csd_beta1",
            str(dwi_biascorr),
            str(out_dir / "response_wm.txt"),
            str(out_dir / "wmfod.mif"),
            str(out_dir / "response_gm.txt"),
            str(out_dir / "gm.mif"),
            str(out_dir / "response_csf.txt"),
            str(out_dir / "csf.mif"),
            "-mask",
            str(mask),
        )
    if model == "msmt_csd":
        return (
            "dwi2fod",
            "msmt_csd",
            str(dwi_biascorr),
            str(out_dir / "response_wm.txt"),
            str(out_dir / "wmfod.mif"),
            str(out_dir / "response_gm.txt"),
            str(out_dir / "gm.mif"),
            str(out_dir / "response_csf.txt"),
            str(out_dir / "csf.mif"),
            "-mask",
            str(mask),
        )
    raise ValueError(f"Unknown FOD model: {model}")


def build_mtnormalise_command(mask: Path, out_dir: Path) -> tuple[str, ...]:
    return (
        "mtnormalise",
        str(out_dir / "wmfod.mif"),
        str(out_dir / "wmfod_norm.mif"),
        str(out_dir / "gm.mif"),
        str(out_dir / "gm_norm.mif"),
        str(out_dir / "csf.mif"),
        str(out_dir / "csf_norm.mif"),
        "-mask",
        str(mask),
    )


def build_fod_command_set(dwi_biascorr: Path, mask: Path, out_dir: Path, shells: list[int] | tuple[int, ...]) -> FodCommands:
    model = choose_fod_model(shells)
    tensor_commands = build_tensor_metric_commands(str(dwi_biascorr), str(mask), str(out_dir))
    return FodCommands(
        response=build_response_command(dwi_biascorr, mask, out_dir),
        fod=build_fod_command(dwi_biascorr, mask, out_dir, model=model),
        mtnormalise=build_mtnormalise_command(mask, out_dir),
        tensor=tensor_commands[0],
        tensor_metrics=tensor_commands[1],
    )
