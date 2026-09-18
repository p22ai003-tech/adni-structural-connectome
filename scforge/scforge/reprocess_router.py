from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import GateConfig


@dataclass(frozen=True)
class RouteDecision:
    route: str
    publication_pass: bool
    candidate_pass: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "route": self.route,
            "publication_pass": int(self.publication_pass),
            "candidate_pass": int(self.candidate_pass),
            "reasons": ";".join(self.reasons),
        }


def _ival(data: dict[str, Any], key: str, default: int = 0) -> int:
    try:
        value = data.get(key, default)
        if value in ("", None):
            return default
        return int(float(value))
    except Exception:
        return default


def _fval(data: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = data.get(key, default)
        if value in ("", None):
            return default
        return float(value)
    except Exception:
        return default


def passes_gate(metrics: dict[str, Any], gate: GateConfig) -> tuple[bool, tuple[str, ...]]:
    reasons: list[str] = []
    if _fval(metrics, "density", -1.0) < gate.density_fd_sum_min:
        reasons.append(f"density_lt_{gate.density_fd_sum_min:g}")
    if _ival(metrics, "valid_zero_rows", 999999) > gate.valid_zero_rows_max:
        reasons.append(f"valid_zero_rows_gt_{gate.valid_zero_rows_max}")
    if _ival(metrics, "missing_valid_labels", 999999) > gate.missing_valid_labels_max:
        reasons.append(f"missing_valid_labels_gt_{gate.missing_valid_labels_max}")
    if _ival(metrics, "unique_assigned_nodes", 0) < gate.unique_assigned_nodes_min:
        reasons.append(f"unique_assigned_nodes_lt_{gate.unique_assigned_nodes_min}")
    if _fval(metrics, "top5_endpoint_fraction", 1.0) > gate.top5_endpoint_fraction_max:
        reasons.append(f"top5_endpoint_fraction_gt_{gate.top5_endpoint_fraction_max:g}")
    if _ival(metrics, "matrix_n", 0) != 166:
        reasons.append("matrix_not_166x166")
    return not reasons, tuple(reasons)


def route_subject(
    metrics: dict[str, Any],
    *,
    candidate_gate: GateConfig,
    publication_gate: GateConfig,
) -> RouteDecision:
    publication_pass, publication_reasons = passes_gate(metrics, publication_gate)
    candidate_pass, candidate_reasons = passes_gate(metrics, candidate_gate)
    if publication_pass:
        return RouteDecision("PUBLICATION_PASS", True, True, ())

    reasons = list(publication_reasons)
    if _ival(metrics, "missing_valid_labels", 0) > 0 or _ival(metrics, "matrix_n", 0) != 166:
        route = "BRANCH_B_REBUILD_AAL3_LABEL_CONTRACT"
    elif str(metrics.get("collapse_class", "")) in {"CATASTROPHIC_COLLAPSE", "SEVERE_COLLAPSE"}:
        route = "BRANCH_C_REGENERATE_5TT_FOD_TRACKS"
    elif _ival(metrics, "unique_assigned_nodes", 0) < candidate_gate.unique_assigned_nodes_min:
        route = "BRANCH_C_REGENERATE_OR_EXCLUDE"
    elif _ival(metrics, "valid_zero_rows", 0) > 0:
        route = "BRANCH_A_ENDPOINT_CATCHMENT_REPAIR"
    elif _fval(metrics, "density", 0.0) < candidate_gate.density_fd_sum_min:
        route = "BRANCH_C_REGENERATE_OR_EXCLUDE"
    else:
        route = "CANDIDATE_PASS_NOT_PUBLICATION_PASS" if candidate_pass else "RETRY_DIFFERENT_LANE"
        reasons.extend(candidate_reasons)

    return RouteDecision(route, False, candidate_pass, tuple(dict.fromkeys(reasons)))
