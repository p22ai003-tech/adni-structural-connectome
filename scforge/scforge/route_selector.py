from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class SpatialContractQC:
    route_name: str
    label_survival_frac: float
    label_inside_mask_frac: float
    t1_b0_boundary_score: float
    five_tt_mask_dice: float
    transform_sanity_score: float
    aal_required_label_score: float
    endpoint_assignment_preflight_score: float = 0.0
    hard_fail_reasons: tuple[str, ...] = ()

    @property
    def hard_fail(self) -> bool:
        return bool(self.hard_fail_reasons)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["hard_fail_reasons"] = ";".join(self.hard_fail_reasons)
        data["score"] = score_spatial_contract(self)
        data["status"] = "FAIL" if self.hard_fail else "PASS"
        return data


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def score_spatial_contract(qc: SpatialContractQC) -> float:
    if qc.hard_fail:
        return 0.0
    return (
        0.30 * _clip01(qc.label_survival_frac)
        + 0.20 * _clip01(qc.label_inside_mask_frac)
        + 0.15 * _clip01(qc.t1_b0_boundary_score)
        + 0.15 * _clip01(qc.five_tt_mask_dice)
        + 0.10 * _clip01(qc.transform_sanity_score)
        + 0.05 * _clip01(qc.aal_required_label_score)
        + 0.05 * _clip01(qc.endpoint_assignment_preflight_score)
    )


def choose_best_spatial_contract(candidates: list[SpatialContractQC]) -> SpatialContractQC | None:
    passing = [candidate for candidate in candidates if not candidate.hard_fail]
    if not passing:
        return None
    return max(passing, key=score_spatial_contract)
