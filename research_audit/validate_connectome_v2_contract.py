#!/usr/bin/env python3
"""Validate the normative connectome-v2 processing and matrix contract.

This is a lightweight specification validator. It does not execute image
processing and it does not claim that the current host is canary-ready.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any

import yaml


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT / "configs" / "connectome_v2.yaml"
DEFAULT_DICTIONARY = PROJECT / "research_audit" / "matrix_data_dictionary.md"
DEFAULT_OUTPUT = PROJECT / "research_audit" / "outputs" / "connectome_v2_contract_validation.json"

EXPECTED_MATRICES = [
    "count",
    "fd_sum",
    "count_invnodevol",
    "len_mean",
    "invlen_mean",
    "fa_mean",
    "md_mean",
    "rd_mean",
    "ad_mean",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def nested(data: dict[str, Any], *keys: str) -> Any:
    value: Any = data
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            raise KeyError(".".join(keys))
        value = value[key]
    return value


def tool_version(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    try:
        proc = subprocess.run(
            [str(path), "-version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception as exc:  # pragma: no cover - diagnostic only
        return f"ERROR:{type(exc).__name__}:{exc}"
    lines = (proc.stdout or proc.stderr or "").strip().splitlines()
    return lines[0] if lines else f"exit={proc.returncode}"


def validate_contract(config_path: Path, dictionary_path: Path) -> dict[str, Any]:
    failures: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []
    checks: list[dict[str, Any]] = []

    def check(code: str, condition: bool, detail: str) -> None:
        row = {"code": code, "passed": bool(condition), "detail": detail}
        checks.append(row)
        if not condition:
            failures.append({"code": code, "detail": detail})

    def warn(code: str, condition: bool, detail: str) -> None:
        if condition:
            warnings.append({"code": code, "detail": detail})

    try:
        data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "status": "FAIL",
            "config": str(config_path),
            "failures": [{"code": "yaml_parse", "detail": f"{type(exc).__name__}:{exc}"}],
            "warnings": [],
            "checks": [],
        }
    check("yaml_mapping", isinstance(data, dict), "Configuration must parse as a mapping.")
    if not isinstance(data, dict):
        return {
            "status": "FAIL",
            "config": str(config_path),
            "failures": failures,
            "warnings": warnings,
            "checks": checks,
        }

    check(
        "recipe_status",
        nested(data, "contract", "status")
        == "implementation_candidate_pending_validation",
        "SL-H03-C1 validates an implementation candidate; it must not be labelled canary-frozen.",
    )
    check(
        "normative_recipe_non_authoritative_pending_h04a",
        nested(data, "contract", "imaging_execution_authorized") is False
        and nested(data, "contract", "current_human_gate") == "SL-H03-C1"
        and nested(data, "contract", "authorization_scope")
        == "design_only_no_imaging"
        and nested(data, "contract", "next_execution_gate") == "SL-H04A"
        and nested(data, "inputs", "approved_pair_manifest", "human_gate")
        == "SL-H03-C1",
        "The normative recipe must remain non-authoritative; only a separate exact signed SL-H04A decision may authorize bounded pre-tractography modes.",
    )
    launcher_path = PROJECT / "scforge" / "workflow" / "run_connectome_v2.py"
    launcher_source = (
        launcher_path.read_text(encoding="utf-8") if launcher_path.is_file() else ""
    )
    check(
        "signed_h04a_bounded_launcher_handshake",
        all(
            token in launcher_source
            for token in (
                "validate_normative_execution_lock",
                "H04A_AUTHORIZED_MODES",
                "require_h04a_mode_authorization",
                "prepare_h04a_resource_preflight",
                "execute_command_with_h04a_monitor",
                "H04A cannot authorize launcher mode",
                "H04A_MINIMUM_VALID_MANUFACTURER_FAMILIES = 2",
                "H04A_MINIMUM_VALID_T1_SOURCE_CLASSES = 2",
                "H04A_REQUIRED_VALID_T1_SOURCE_CLASSES",
                "validate_response_calibration_diversity_binding",
            )
        ),
        "The supported launcher must bind exact H04A authority and enforce the pre-tractography mode, resource limits, and valid-pool manufacturer/T1 diversity through continuation without changing the normative recipe.",
    )
    check(
        "fail_closed",
        nested(data, "contract", "failure_policy") == "fail_closed"
        and nested(data, "contract", "silent_fallbacks_allowed") is False,
        "Primary processing must fail closed with no silent fallback.",
    )
    check(
        "no_density_route_selection",
        nested(data, "contract", "select_route_by_connectome_density") is False
        and nested(data, "qc", "matrix", "density_used_as_inclusion_gate") is False,
        "Connectome density cannot select a route or primary inclusion.",
    )
    check(
        "no_padding_or_clipping",
        nested(data, "contract", "pad_missing_nodes_or_matrices") is False
        and nested(data, "contract", "clip_or_impute_tensor_metrics") is False
        and nested(data, "qc", "matrix", "padding_allowed") is False,
        "No node padding, tensor clipping, or tensor imputation is permitted.",
    )

    atlas = nested(data, "atlas")
    atlas_paths = {
        "source_image": "source_image_sha256",
        "contiguous_source_image": "contiguous_source_image_sha256",
        "node_table": "node_table_sha256",
        "node_map": "node_map_sha256",
    }
    for path_key, hash_key in atlas_paths.items():
        path = Path(atlas[path_key])
        exists = path.exists() and path.is_file()
        check(f"atlas_{path_key}_exists", exists, f"Required atlas artifact: {path}")
        if exists:
            observed = sha256_file(path)
            check(
                f"atlas_{path_key}_hash",
                observed == atlas[hash_key],
                f"Expected {atlas[hash_key]}, observed {observed}.",
            )

    try:
        with Path(atlas["node_map"]).open(newline="", encoding="utf-8") as handle:
            node_map = list(csv.DictReader(handle))
        new_ids = [int(row["new_id"]) for row in node_map]
        original_ids = [int(row["orig_value"]) for row in node_map]
    except Exception as exc:
        node_map, new_ids, original_ids = [], [], []
        failures.append({"code": "node_map_parse", "detail": f"{type(exc).__name__}:{exc}"})
    check("node_map_count", len(node_map) == 166, "Node map must contain exactly 166 rows.")
    check("node_map_contiguous", new_ids == list(range(1, 167)), "New node IDs must be exactly 1..166.")
    check(
        "node_map_original_labels",
        original_ids == [x for x in range(1, 171) if x not in {35, 36, 81, 82}],
        "Original AAL3 labels must be 1..170 excluding 35,36,81,82.",
    )
    check(
        "atlas_no_catchment",
        atlas["primary_endpoint_catchment_dilation_mm"] == 0
        and nested(data, "connectome", "assignment", "endpoint_catchment_dilation_mm") == 0,
        "The primary route must use direct labels with no endpoint-catchment dilation.",
    )

    matrices = nested(data, "connectome", "required_matrices")
    names = [row.get("name") for row in matrices]
    check("nine_matrix_names", names == EXPECTED_MATRICES, f"Observed matrix order: {names}")
    by_name = {row["name"]: row for row in matrices if isinstance(row, dict) and "name" in row}
    check(
        "count_is_raw",
        by_name.get("count", {}).get("streamline_weights") is None
        and by_name.get("count", {}).get("stat_edge") == "sum",
        "count must be an unweighted sum of assigned streamlines.",
    )
    check(
        "fd_sum_is_sift2",
        by_name.get("fd_sum", {}).get("streamline_weights") == "sift2_weights"
        and by_name.get("fd_sum", {}).get("stat_edge") == "sum",
        "fd_sum must be the sum of SIFT2 weights.",
    )
    check(
        "count_invnodevol_formula",
        by_name.get("count_invnodevol", {}).get("kind") == "derived_from_count_and_physical_node_volumes"
        and by_name.get("count_invnodevol", {}).get("formula")
        == "2*count_ij/(node_volume_i_mm3+node_volume_j_mm3)",
        "count_invnodevol must be independently derived from raw count and physical volumes.",
    )
    check(
        "length_definitions",
        by_name.get("len_mean", {}).get("scale") == "length"
        and by_name.get("invlen_mean", {}).get("scale") == "invlength"
        and by_name.get("len_mean", {}).get("stat_edge") == "mean"
        and by_name.get("invlen_mean", {}).get("stat_edge") == "mean",
        "Length and inverse-length matrices must be separate arithmetic edge means.",
    )
    metric_names = ["fa_mean", "md_mean", "rd_mean", "ad_mean"]
    check(
        "tensor_edge_means_unweighted",
        all(
            by_name.get(name, {}).get("kind") == "tcksample_then_tck2connectome"
            and by_name.get(name, {}).get("stat_tck") == "mean"
            and by_name.get(name, {}).get("stat_edge") == "mean"
            and by_name.get(name, {}).get("streamline_weights") is None
            for name in metric_names
        ),
        "All tensor matrices must use an unweighted mean-along-streamline then mean-within-edge definition.",
    )
    check(
        "axd_label",
        by_name.get("ad_mean", {}).get("publication_label") == "AxD",
        "The axial-diffusivity matrix filename may be ad_mean, but its publication label must be AxD.",
    )
    check(
        "assignment_frozen",
        nested(data, "connectome", "assignment", "method") == "radial_search"
        and nested(data, "connectome", "assignment", "radius_mm") == 4
        and nested(data, "connectome", "assignment", "apply_identically_to_every_subject_and_matrix") is True,
        "Primary endpoint assignment must be a uniform direct radial4 rule.",
    )
    check(
        "common_matrix_flags",
        nested(data, "connectome", "common_flags") == ["-symmetric", "-zero_diagonal"],
        "All matrices must be symmetric with a zero diagonal.",
    )

    check(
        "act_primary",
        nested(data, "tractography", "framework") == "ACT"
        and nested(data, "tractography", "act_enabled") is True
        and nested(data, "tractography", "automatic_act_to_noact_switch_allowed") is False,
        "ACT is the frozen starting route and cannot switch per subject.",
    )
    check(
        "tractography_fixed",
        nested(data, "tractography", "algorithm") == "iFOD2"
        and nested(data, "tractography", "select_streamlines") == 10_000_000
        and nested(data, "tractography", "minimum_length_mm") == 10
        and nested(data, "tractography", "maximum_length_mm") == 250
        and nested(data, "tractography", "cutoff") == 0.06,
        "Primary tractography must be iFOD2, 10M, 10-250 mm, cutoff 0.06.",
    )
    check(
        "sift2_fixed",
        nested(data, "sift2", "fd_scale_gm") is False
        and nested(data, "sift2", "retry_with_changed_flags_allowed") is False
        and nested(data, "sift2", "save_mu") is True,
        "SIFT2 flags must be fixed and mu retained.",
    )
    check(
        "tensor_no_fallback",
        nested(data, "tensor", "fit", "dtifit_fallback") is False
        and nested(data, "tensor", "invalid_value_policy") == "fail_qc_no_clipping_no_zero_fill",
        "Tensor failures cannot trigger a fitter switch, clipping, or zero fill.",
    )
    fod_lmax = int(nested(data, "fod", "lmax"))
    fod_unique_directions = int(nested(data, "fod", "minimum_unique_nonzero_directions"))
    required_even_sh_coefficients = (fod_lmax + 1) * (fod_lmax + 2) // 2
    check(
        "fod_lmax_direction_support",
        fod_lmax >= 2
        and fod_lmax % 2 == 0
        and fod_unique_directions >= required_even_sh_coefficients,
        f"lmax={fod_lmax} requires at least {required_even_sh_coefficients} unique antipodally canonicalised directions; configured minimum={fod_unique_directions}.",
    )

    dictionary_text = dictionary_path.read_text(encoding="utf-8") if dictionary_path.exists() else ""
    check("dictionary_exists", bool(dictionary_text), f"Matrix dictionary required: {dictionary_path}")
    for name in EXPECTED_MATRICES:
        check(f"dictionary_{name}", f"`{name}`" in dictionary_text, f"Dictionary must define {name}.")
    check(
        "dictionary_no_claim_ad_label",
        bool(re.search(r"AxD.*never.*AD", dictionary_text, flags=re.IGNORECASE | re.DOTALL)),
        "Dictionary must prevent AD/AxD terminology ambiguity.",
    )

    mrtrix = Path("/home/ec2-user/mrtrix3/bin/tck2connectome")
    ss3t = Path("/home/ec2-user/MRtrix3Tissue/bin/ss3t_csd_beta1")
    ss3t_python = Path("/usr/bin/python3.9")
    ants = Path("/home/ec2-user/exp/.envs/ants-2.6.5/bin/antsRegistration")
    check("mrtrix_host_present", mrtrix.exists(), f"Observed: {tool_version(mrtrix)}")
    check(
        "ss3t_environment_locked",
        ss3t.exists() and ss3t_python.exists(),
        "Pinned MRtrix3Tissue script and its Python 3.9 runtime must both exist.",
    )
    warn(
        "workflow_placeholders",
        any("Placeholder" in path.read_text(encoding="utf-8", errors="replace") for path in (PROJECT / "scforge" / "workflow").rglob("*.smk")),
        "The current SC-Forge workflow contains placeholder rules; SL-P0-12 must replace them.",
    )
    ants_lock = PROJECT / "scforge" / "workflow" / "locks" / "ants-2.6.5-linux-64.explicit.txt"
    check(
        "ants_environment_locked",
        ants.exists() and ants_lock.exists(),
        "The isolated ANTs executable and explicit Conda lock must both exist.",
    )
    environment_contract = PROJECT / "scforge" / "workflow" / "environment_contract.yaml"
    check(
        "host_environment_contract",
        environment_contract.exists(),
        "The exact host paths, versions, hashes, and executor policy must be recorded.",
    )
    check(
        "eddy_executor_frozen",
        nested(data, "dwi_preprocessing", "eddy", "executable") == "/home/ec2-user/fsl/bin/eddy_cpu"
        and nested(data, "dwi_preprocessing", "eddy", "automatic_cuda_to_cpu_fallback_allowed") is False,
        "The recipe must name one eddy executor and prohibit automatic CUDA-to-CPU switching.",
    )

    status = "FAIL" if failures else ("PASS_WITH_IMPLEMENTATION_BLOCKERS" if warnings else "PASS")
    return {
        "status": status,
        "recipe_id": nested(data, "contract", "recipe_id"),
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "dictionary": str(dictionary_path),
        "dictionary_sha256": sha256_file(dictionary_path) if dictionary_path.exists() else None,
        "matrix_names": names,
        "check_count": len(checks),
        "passed_check_count": sum(int(row["passed"]) for row in checks),
        "failure_count": len(failures),
        "warning_count": len(warnings),
        "failures": failures,
        "warnings": warnings,
        "checks": checks,
        "host_tools": {
            "tck2connectome": tool_version(mrtrix),
            "ss3t_csd_beta1": tool_version(ss3t),
            "ss3t_python": str(ss3t_python) if ss3t_python.exists() else None,
            "antsRegistration_path": str(ants) if ants.exists() else None,
            "flirt_path": shutil.which("flirt") or "/home/ec2-user/fsl/share/fsl/bin/flirt",
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dictionary", type=Path, default=DEFAULT_DICTIONARY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument("--strict-tools", action="store_true", help="Treat implementation blockers as nonzero exit.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = validate_contract(args.config.resolve(), args.dictionary.resolve())
    if not args.no_write:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(args.output)
    print(json.dumps({key: result[key] for key in ("status", "check_count", "passed_check_count", "failure_count", "warning_count")}, indent=2))
    if result["status"] == "FAIL":
        return 1
    if args.strict_tools and result["warning_count"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
