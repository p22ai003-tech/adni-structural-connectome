from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class T1Contract:
    t1_native: Path
    t1_brain: Path
    t1_brainmask: Path
    source: str
    status: str
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "t1_native": str(self.t1_native),
            "t1_brain": str(self.t1_brain),
            "t1_brainmask": str(self.t1_brainmask),
            "source": self.source,
            "status": self.status,
            "reason": self.reason,
        }


def classify_t1_source(path: str | Path) -> str:
    name = Path(path).name.lower()
    if name in {"nu.mgz", "brainmask.mgz"} or "freesurfer" in str(path).lower():
        return "native_freesurfer"
    if "b0" in name or "ras" in name or "resam" in name:
        return "resampled_or_b0_grid_review_required"
    return "native_candidate"


def build_t1_contract(
    *,
    t1_native: str | Path,
    t1_brain: str | Path,
    t1_brainmask: str | Path,
    allow_resampled: bool = False,
) -> T1Contract:
    native = Path(t1_native)
    brain = Path(t1_brain)
    mask = Path(t1_brainmask)
    reasons: list[str] = []
    for label, path in (("t1_native", native), ("t1_brain", brain), ("t1_brainmask", mask)):
        if not path.exists():
            reasons.append(f"missing_{label}")
    source = classify_t1_source(native)
    if source == "resampled_or_b0_grid_review_required" and not allow_resampled:
        reasons.append("t1_source_appears_resampled_or_b0_grid")
    return T1Contract(
        t1_native=native,
        t1_brain=brain,
        t1_brainmask=mask,
        source=source,
        status="PASS" if not reasons else "FAIL",
        reason=";".join(reasons),
    )


def build_mri_convert_command(input_image: str | Path, output_image: str | Path) -> tuple[str, ...]:
    return ("mri_convert", str(input_image), str(output_image))


def build_apply_mask_command(input_image: str | Path, mask_image: str | Path, output_image: str | Path) -> tuple[str, ...]:
    return ("fslmaths", str(input_image), "-mas", str(mask_image), str(output_image))
