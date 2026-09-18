#!/home/ec2-user/fsl/bin/python
"""Run the Phase-B-selected HCP379 recipe on Recovery4 scale-up inputs.

The default mode is a non-imaging pre-Phase-B dry run.  ``--execute`` is
accepted only after all 216 Recovery4 pretract states pass, the genuine
15-subject visual review is bound, and the Phase-B stability validator has
selected one of 3M, 5M or 10M streamlines.  The runner then generates the
track-matched SIFT2 weights, endpoint assignments and all nine HCP379
matrices without using diagnosis, outcomes or subject-level density to select
or exclude a case.

The mature matrix/QC implementation in ``hcp379_scaleup_tractography_v2.py``
is reused only through a Recovery4 adapter.  That adapter verifies each final
pretract state and replaces every legacy path with the promoted selected
5TT, HCP379 atlas and separately bounded tensor maps.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping

import yaml


EXP = Path("/home/ec2-user/exp")
HCP_ROOT = Path("/data/derivatives/hcp379_v2")
ROOT = HCP_ROOT / "corrected_scaleup_recovery4"
RUN_ROOT = Path(
    "/data/derivatives/scforge_v2/h04a_r1_recovery_20260719_retry4"
)
PRETRACT_SOURCE = (
    EXP / "scripts/hcp/hcp379_scaleup_pretract_recovery4_v3.py"
)
ENGINE_SOURCE = EXP / "scripts/hcp/hcp379_scaleup_tractography_v2.py"
TCK_HEADER_COUNTER_SOURCE = (
    EXP / "scripts/hcp/tck_header_count_v1.py"
)
NUMERIC_VALUE_COUNTER_SOURCE = (
    EXP / "scripts/hcp/count_mrtrix_numeric_values_v1.py"
)
PHASE_RULES_SOURCE = (
    EXP
    / "scforge/workflow/extensions/"
    "h04a_r1_retry4_phase_b_hcp379/07_hcp379_stability.smk"
)
AUDIT_CSV = (
    EXP
    / "research_audit/outputs/hcp379_scaleup_input_audit_v2/"
    "scaleup_input_audit.csv"
)
AUDIT_SUMMARY = AUDIT_CSV.with_name("scaleup_input_audit_summary.json")
PRETRACT_SUMMARY = (
    ROOT / "manifests/pretract_recovery4_summary.json"
)
PRETRACT_PLAN = ROOT / "manifests/pretract_recovery4_plan.json"
TRACT_PLAN = (
    ROOT / "manifests/tractography_recovery4_selected_recipe_plan.json"
)
PHASE_MANIFEST = (
    RUN_ROOT / "publication/hcp379_recipe_stability_manifest.json"
)
PHASE_VALIDATION = (
    RUN_ROOT / "publication/hcp379_recipe_stability_validation.json"
)
CONFIG = EXP / "configs/connectome_v2_retry3.yaml"
MRTRIX = Path("/home/ec2-user/mrtrix3/bin")
EXPECTED_SCALEUP_N = 216
EXPECTED_CORRECTED_N = 227
EXPECTED_CANARY_N = 15
EXPECTED_CANARY_OVERLAP_N = 11
ALLOWED_COUNTS = (3_000_000, 5_000_000, 10_000_000)
MINIMUM_START_FREE_BYTES = 700_000_000_000


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PRETRACT = load_module(
    "hcp379_scaleup_pretract_recovery4_v3_for_tractography",
    PRETRACT_SOURCE,
)
ENGINE = load_module(
    "hcp379_scaleup_tractography_v2_recovery4_engine",
    ENGINE_SOURCE,
)


def pretract_compaction_path(root: Path, unit: str) -> Path:
    return root / "qc/pretract_compaction" / f"{unit}.json"


def validate_recovery4_pretract_compaction(
    root: Path, unit: str
) -> dict[str, Any] | None:
    path = pretract_compaction_path(root, unit)
    if not path.is_file():
        return None
    record = PRETRACT.load_json(path)
    if (
        record.get("record_type")
        != "diagnosis_blind_hcp379_pretract_compaction"
        or record.get("status") != "PASS_COMPACTED"
        or record.get("diagnosis_labels_used") is not False
        or record.get("unit") != unit
        or Path(str(record.get("root", ""))).resolve() != root.resolve()
        or record.get("recovery4_inputs") is not True
        or record.get("source_pretract_record_type")
        != "diagnosis_blind_hcp379_scaleup_pretract_recovery4"
        or record.get("source_pretract_status")
        != "PASS_PRETRACT_RECOVERY4"
        or record.get("raw_tensor_maps_preserved") is not True
        or record.get("bounded_tensor_maps_separate") is not True
    ):
        raise ValueError(f"{unit}:Recovery4 pretract compaction differs")
    PRETRACT.verify_file_record(
        record.get("source_pretract_state"),
        label=f"{unit}/compaction/source_pretract_state",
    )
    retained = record.get("retained_artifacts")
    deleted = record.get("deleted_artifacts")
    required = {
        "selected_five_tt",
        "selected_gmwmi",
        "selected_hcp379_nodes",
        "selected_hcp379_node_volumes",
        "selected_hcp379_atlas_qc",
        "wmfod_normalised",
        "fa_raw",
        "md_raw",
        "rd_raw",
        "ad_raw",
        "fa_bounded",
        "md_bounded",
        "rd_bounded",
        "ad_bounded",
        "spatial_route_qc",
        "scalar_recovery4_qc",
        "five_tt",
        "wmfod_norm",
        "hcp_nodes",
        "hcp_volumes",
        "fa",
        "md",
        "rd",
        "ad",
        "atlas_qc",
        "automated_qc",
    }
    if (
        not isinstance(retained, Mapping)
        or not required.issubset(retained)
        or not isinstance(deleted, Mapping)
    ):
        raise ValueError(f"{unit}:Recovery4 compacted artifacts differ")
    for name in required:
        PRETRACT.verify_file_record(
            retained[name], label=f"{unit}/compaction/retained/{name}"
        )
    dwi = deleted.get("dwi")
    if (
        not isinstance(dwi, Mapping)
        or dwi.get("deleted") is not True
        or Path(str(dwi.get("path", ""))).exists()
    ):
        raise ValueError(f"{unit}:Recovery4 compacted DWI differs")
    return record


def promoted_pretract_state(root: Path, unit: str) -> dict[str, Any]:
    path = PRETRACT.spatial_paths(root, unit)["state"]
    state = PRETRACT.load_json(path)
    if (
        state.get("record_type")
        != "diagnosis_blind_hcp379_scaleup_pretract_recovery4"
        or state.get("status") != "PASS_PRETRACT_RECOVERY4"
        or state.get("diagnosis_labels_used") is not False
        or state.get("unit") != unit
        or state.get("raw_tensor_maps_preserved") is not True
        or state.get("bounded_tensor_maps_separate") is not True
    ):
        raise ValueError(f"{unit}:Recovery4 pretract state differs")
    artifacts = state.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise TypeError(f"{unit}:Recovery4 artifacts are not a mapping")
    retained_required = {
        "selected_five_tt",
        "selected_gmwmi",
        "selected_hcp379_nodes",
        "selected_hcp379_node_volumes",
        "selected_hcp379_atlas_qc",
        "wmfod_normalised",
        "fa_raw",
        "md_raw",
        "rd_raw",
        "ad_raw",
        "fa_bounded",
        "md_bounded",
        "rd_bounded",
        "ad_bounded",
        "spatial_route_qc",
        "scalar_recovery4_qc",
    }
    working_required = {
        "dwi_bias_corrected",
        "dwi_mask",
        "gm_normalised",
        "csf_normalised",
    }
    required = retained_required | working_required
    if not required.issubset(artifacts):
        raise ValueError(
            f"{unit}:promoted artifacts absent: "
            f"{sorted(required - set(artifacts))}"
        )
    compaction = validate_recovery4_pretract_compaction(root, unit)
    verify_names = retained_required if compaction is not None else required
    for name in verify_names:
        PRETRACT.verify_file_record(
            artifacts[name], label=f"{unit}/pretract/{name}"
        )
    return state


class Recovery4PretractAdapter:
    """Minimal API expected by the mature tractography implementation."""

    utc_now = staticmethod(PRETRACT.utc_now)
    read_csv = staticmethod(PRETRACT.read_csv)
    file_record = staticmethod(PRETRACT.file_record)
    atomic_json = staticmethod(PRETRACT.atomic_json)
    atomic_csv = staticmethod(PRETRACT.atomic_csv)
    environment = staticmethod(PRETRACT.environment)

    @staticmethod
    def load_json(path: Path) -> dict[str, Any]:
        value = PRETRACT.load_json(path)
        if value.get("record_type") == (
            "diagnosis_blind_hcp379_scaleup_pretract_recovery4"
        ):
            translated = dict(value)
            translated["recovery4_status"] = value.get("status")
            if value.get("status") == "PASS_PRETRACT_RECOVERY4":
                translated["status"] = "PASS_PRETRACT"
            return translated
        return value

    @staticmethod
    def stage_paths(root: Path, unit: str) -> dict[str, Path]:
        paths = PRETRACT.spatial_paths(root, unit)
        state = promoted_pretract_state(root, unit)
        artifacts = state["artifacts"]
        compaction = validate_recovery4_pretract_compaction(root, unit)
        retained = (
            compaction["retained_artifacts"]
            if compaction is not None
            else {}
        )
        replacements = {
            "dwi": "dwi_bias_corrected",
            "mask": "dwi_mask",
            "five_tt": "selected_five_tt",
            "gmwmi": "selected_gmwmi",
            "hcp_nodes": "selected_hcp379_nodes",
            "hcp_volumes": "selected_hcp379_node_volumes",
            "atlas_qc": "selected_hcp379_atlas_qc",
            "wmfod_norm": "wmfod_normalised",
            "gm_norm": "gm_normalised",
            "csf_norm": "csf_normalised",
            "fa": "fa_bounded",
            "md": "md_bounded",
            "rd": "rd_bounded",
            "ad": "ad_bounded",
            "automated_qc": "scalar_recovery4_qc",
        }
        for key, artifact in replacements.items():
            record = retained.get(artifact, artifacts[artifact])
            paths[key] = Path(str(record["path"])).resolve()
        return paths


ADAPTER = Recovery4PretractAdapter()


def install_recovery4_adapter() -> None:
    ENGINE.PRETRACT = ADAPTER
    ENGINE.ROOT = ROOT
    ENGINE.PRETRACT_SUMMARY = PRETRACT_SUMMARY
    ENGINE.PRETRACT_GATE = PRETRACT_PLAN
    ENGINE.PRETRACT_SOURCE = PRETRACT_SOURCE
    ENGINE.PHASE_MANIFEST = PHASE_MANIFEST
    ENGINE.PHASE_VALIDATION = PHASE_VALIDATION


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def phase_status() -> dict[str, Any]:
    output: dict[str, Any] = {}
    for name, path in (
        ("manifest", PHASE_MANIFEST),
        ("validation", PHASE_VALIDATION),
    ):
        if path.is_file():
            record = PRETRACT.load_json(path)
            output[name] = {
                "present": True,
                "status": record.get("status"),
                "record": PRETRACT.file_record(path),
            }
        else:
            output[name] = {"present": False, "status": "NOT_RUN"}
    return output


def tool_contract() -> dict[str, dict[str, Any]]:
    tools = {
        name: MRTRIX / name
        for name in (
            "mrinfo",
            "tckgen",
            "tckinfo",
            "tcksift2",
            "tck2connectome",
            "tcksample",
        )
    }
    return {
        name: PRETRACT.file_record(path) for name, path in tools.items()
    }


def validate_recipe() -> tuple[dict[str, Any], str]:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise TypeError("recipe config is not a mapping")
    tractography = config["tractography"]
    required = {
        "algorithm": "iFOD2",
        "act_enabled": True,
        "backtrack": True,
        "crop_at_gmwmi": True,
    }
    for key, expected in required.items():
        if tractography.get(key) != expected:
            raise ValueError(f"locked tractography recipe differs: {key}")
    if (
        tractography.get("seeding", {}).get("method") != "seed_dynamic"
        or tractography.get("seeding", {}).get("image")
        != "normalised_wmfod"
        or tractography.get("seeding", {}).get("gmwmi_used_as_seed")
        is not False
        or int(tractography["maximum_seed_attempts"]) != 200_000_000
        or float(tractography["minimum_length_mm"]) != 10.0
        or float(tractography["maximum_length_mm"]) != 250.0
        or float(tractography["cutoff"]) != 0.06
        or float(tractography["maximum_angle_degrees"]) != 45.0
    ):
        raise ValueError("locked tractography parameter contract differs")
    return tractography, str(config["contract"]["recipe_id"])


def validate_prephase_scope() -> tuple[list[dict[str, str]], list[str]]:
    gate = PRETRACT.validate_recovery4_gate(
        human_qc=PRETRACT.DEFAULT_HUMAN_QC,
        execute=False,
    )
    rows, overlap = PRETRACT.validate_audit_rows(gate)
    if len(rows) != EXPECTED_SCALEUP_N:
        raise ValueError("Recovery4 tractography target set differs")
    return rows, overlap


def build_prephase_plan(
    *,
    root: Path,
    rows: list[dict[str, str]],
    overlap: list[str],
) -> dict[str, Any]:
    tractography, recipe_id = validate_recipe()
    source = ENGINE_SOURCE.read_text(encoding="utf-8")
    phase_source = PHASE_RULES_SOURCE.read_text(encoding="utf-8")
    required_fragments = (
        "tckgen",
        "-algorithm",
        "iFOD2",
        "-act",
        "-seed_dynamic",
        "tcksift2",
        "-reg_tikhonov",
        "0.0",
        "-reg_tv",
        "0.1",
        "-assignment_radial_search",
        "4",
        "tcksample",
        "-stat_tck",
        "mean",
        "tck_header_count_v1.py",
        "count_mrtrix_numeric_values_v1.py",
    )
    missing_engine = [
        value for value in required_fragments if value not in source
    ]
    missing_phase = [
        value for value in required_fragments if value not in phase_source
    ]
    if missing_engine or missing_phase:
        raise ValueError(
            f"tractography command parity differs: "
            f"engine={missing_engine};phase={missing_phase}"
        )
    if "wc -l" in source or "wc -l" in phase_source:
        raise ValueError("line-count assumptions remain in tractography code")
    plan = {
        "schema_version": "1.0.0",
        "record_type": (
            "diagnosis_blind_hcp379_recovery4_selected_recipe_"
            "tractography_plan"
        ),
        "status": "PRE_PHASE_DRY_RUN_PASS",
        "generated_utc": PRETRACT.utc_now(),
        "diagnosis_labels_used": False,
        "non_overwriting": True,
        "imaging_executed_by_plan": False,
        "execution_authorized": False,
        "output_root": str(root.resolve()),
        "target_noncanary_n": EXPECTED_SCALEUP_N,
        "canary_overlap_n": len(overlap),
        "corrected_release_n": EXPECTED_CORRECTED_N,
        "phase_b_status": phase_status(),
        "allowed_streamline_counts": list(ALLOWED_COUNTS),
        "selected_streamline_count": None,
        "selection_rule": (
            "smallest Phase-B density-qualified stable count; no "
            "subject-level recipe selection"
        ),
        "matrix_names": list(ENGINE.MATRIX_NAMES),
        "density_is_subject_inclusion_gate": False,
        "tractography_recipe": tractography,
        "recipe_id": recipe_id,
        "pretract_adapter_contract": {
            "required_status": "PASS_PRETRACT_RECOVERY4",
            "selected_five_tt": True,
            "selected_hcp379": True,
            "bounded_fa_md_rd_ad": True,
            "raw_tensor_maps_are_not_sampled": True,
        },
        "tool_contract": tool_contract(),
        "builder": PRETRACT.file_record(Path(__file__)),
        "matrix_engine": PRETRACT.file_record(ENGINE_SOURCE),
        "counter_helpers": {
            "tck_header_count": PRETRACT.file_record(
                TCK_HEADER_COUNTER_SOURCE
            ),
            "numeric_value_count": PRETRACT.file_record(
                NUMERIC_VALUE_COUNTER_SOURCE
            ),
        },
        "pretract_runner": PRETRACT.file_record(PRETRACT_SOURCE),
        "phase_b_rules": PRETRACT.file_record(PHASE_RULES_SOURCE),
        "config": PRETRACT.file_record(CONFIG),
        "units": [
            {
                "unit": row["unit"],
                "dti_source_id": row["dti_source_id"],
                "dti_raw_bundle_sha256": row["dti_raw_bundle_sha256"],
            }
            for row in rows
        ],
    }
    stamp = (
        PRETRACT.utc_now()
        .replace(":", "")
        .replace("-", "")
        .replace(".", "")
        + "-tractography-recovery4-plan.json"
    )
    PRETRACT.atomic_json(root / "manifests/attempts" / stamp, plan)
    PRETRACT.atomic_json(
        root
        / "manifests/tractography_recovery4_selected_recipe_plan.json",
        plan,
    )
    return plan


def validate_full_gate(root: Path) -> dict[str, Any]:
    rows, overlap = validate_prephase_scope()
    pretract_summary_path = root / PRETRACT_SUMMARY.relative_to(ROOT)
    pretract_plan_path = root / PRETRACT_PLAN.relative_to(ROOT)
    missing = [
        str(path)
        for path in (pretract_summary_path, pretract_plan_path)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Recovery4 pretract execution is not complete: "
            + ",".join(missing)
        )
    pretract = PRETRACT.load_json(pretract_summary_path)
    if (
        pretract.get("record_type")
        != "diagnosis_blind_hcp379_scaleup_pretract_recovery4_summary"
        or pretract.get("status") != "PASS"
        or pretract.get("target_n") != EXPECTED_SCALEUP_N
        or pretract.get("states_present_n") != EXPECTED_SCALEUP_N
        or pretract.get("status_counts", {}).get(
            "PASS_PRETRACT_RECOVERY4"
        )
        != EXPECTED_SCALEUP_N
    ):
        raise ValueError("216-subject Recovery4 pretract summary is not PASS")
    for row in rows:
        promoted_pretract_state(root, row["unit"])

    pretract_plan = PRETRACT.load_json(pretract_plan_path)
    if (
        pretract_plan.get("status") != "EXECUTION_GATE_PASS"
        or pretract_plan.get("execution_requested") is not True
        or pretract_plan.get("imaging_executed_by_plan") is not False
        or pretract_plan.get("target_n") != EXPECTED_SCALEUP_N
        or pretract_plan.get("selected_this_invocation_n")
        != EXPECTED_SCALEUP_N
    ):
        raise ValueError("Recovery4 pretract execution binding differs")
    human = PRETRACT.validate_human_qc(
        PRETRACT.DEFAULT_HUMAN_QC,
        set(PRETRACT.validate_recovery4_gate(
            human_qc=PRETRACT.DEFAULT_HUMAN_QC,
            execute=False,
        )["canary_units"]),
    )
    if pretract_plan.get("gate_records", {}).get(
        "human_visual_qc"
    ) != human:
        raise ValueError("pretract/human-QC binding differs")

    if not PHASE_MANIFEST.is_file() or not PHASE_VALIDATION.is_file():
        raise FileNotFoundError("Phase-B manifest/validation is not complete")
    manifest = PRETRACT.load_json(PHASE_MANIFEST)
    validation = PRETRACT.load_json(PHASE_VALIDATION)
    if (
        manifest.get("record_type")
        != "diagnosis_blind_hcp379_phase_b_recipe_stability_manifest"
        or manifest.get("diagnosis_labels_used") is not False
        or len(manifest.get("units", [])) != EXPECTED_CANARY_N
        or validation.get("status") != "PASS"
        or validation.get("diagnosis_labels_used") is not False
        or validation.get("manifest")
        != PRETRACT.file_record(PHASE_MANIFEST)
    ):
        raise ValueError("Phase-B stability evidence differs")
    selected = validation.get(
        "smallest_density_qualified_scale_up_streamline_count"
    )
    if selected not in ALLOWED_COUNTS:
        raise ValueError(
            f"Phase-B did not select an allowed streamline count: {selected}"
        )
    canary_units = sorted(
        str(record.get("unit")) for record in manifest["units"]
    )
    if (
        len(set(canary_units)) != EXPECTED_CANARY_N
        or len(set(canary_units).intersection(row["unit"] for row in rows))
        != 0
        or len(
            set(canary_units).intersection(
                row["unit"] for row in PRETRACT.read_csv(AUDIT_CSV)
            )
        )
        != EXPECTED_CANARY_OVERLAP_N
    ):
        raise ValueError("Phase-B/corrected-lane identity sets differ")

    tractography, recipe_id = validate_recipe()
    return {
        "rows": rows,
        "overlap": overlap,
        "selected_count": int(selected),
        "selected_run_id": f"primary_{int(selected / 1_000_000)}m",
        "canary_units": canary_units,
        "validation": validation,
        "manifest": manifest,
        "tractography": tractography,
        "recipe_id": recipe_id,
        "records": {
            "phase_validation": PRETRACT.file_record(PHASE_VALIDATION),
            "phase_manifest": PRETRACT.file_record(PHASE_MANIFEST),
            "pretract_summary": PRETRACT.file_record(
                root / PRETRACT_SUMMARY.relative_to(ROOT)
            ),
            "pretract_execution_plan": PRETRACT.file_record(
                root / PRETRACT_PLAN.relative_to(ROOT)
            ),
            "human_visual_qc": human,
            "input_audit": PRETRACT.file_record(AUDIT_CSV),
            "input_audit_summary": PRETRACT.file_record(AUDIT_SUMMARY),
            "config": PRETRACT.file_record(CONFIG),
        },
    }


def augment_pass_state(
    *,
    root: Path,
    unit: str,
    state: dict[str, Any],
    gate: Mapping[str, Any],
) -> dict[str, Any]:
    if state.get("status") != "PASS_ALL_NINE":
        return state
    pretract_path = PRETRACT.spatial_paths(root, unit)["state"]
    pretract_state = promoted_pretract_state(root, unit)
    compaction = validate_recovery4_pretract_compaction(root, unit)
    retained = (
        compaction["retained_artifacts"]
        if compaction is not None
        else pretract_state["artifacts"]
    )
    selected_names = (
        "selected_five_tt",
        "selected_hcp379_nodes",
        "selected_hcp379_node_volumes",
        "selected_hcp379_atlas_qc",
        "wmfod_normalised",
        "fa_bounded",
        "md_bounded",
        "rd_bounded",
        "ad_bounded",
    )
    selected_inputs = {
        name: retained[name] for name in selected_names
    }
    if compaction is None:
        selected_inputs = {
            "dwi_bias_corrected": pretract_state["artifacts"][
                "dwi_bias_corrected"
            ],
            "dwi_mask": pretract_state["artifacts"]["dwi_mask"],
            **selected_inputs,
        }
    value = {
        **state,
        "recovery4_input_binding": {
            "pretract_state": PRETRACT.file_record(pretract_path),
            "registration_route": pretract_state["registration_route"],
            "selected_inputs": selected_inputs,
            "pretract_compaction": (
                PRETRACT.file_record(pretract_compaction_path(root, unit))
                if compaction is not None
                else None
            ),
            "deleted_reproducible_working_inputs": (
                compaction["deleted_state_artifacts"]
                if compaction is not None
                else {}
            ),
            "dwi_spacing_mm": (
                compaction["dwi_spacing_mm"]
                if compaction is not None
                else None
            ),
            "phase_b_validation": gate["records"]["phase_validation"],
            "selected_streamline_count": gate["selected_count"],
            "selected_run_id": gate["selected_run_id"],
        },
        "bounded_tensor_maps_sampled": True,
        "raw_tensor_maps_sampled": False,
    }
    path = ENGINE.tract_paths(root, unit)["tract_state"]
    PRETRACT.atomic_json(path, value)
    return value


def process_unit(
    row: Mapping[str, str],
    *,
    root: Path,
    gate: Mapping[str, Any],
    nthreads: int,
) -> dict[str, Any]:
    state = ENGINE.process_unit(
        row,
        root=root,
        gate=gate,
        nthreads=nthreads,
    )
    return augment_pass_state(
        root=root,
        unit=str(row["unit"]),
        state=state,
        gate=gate,
    )


def write_execution_gate(
    root: Path,
    gate: Mapping[str, Any],
    *,
    compact: bool,
) -> None:
    PRETRACT.atomic_json(
        root / "manifests/tractography_recovery4_execution_gate.json",
        {
            "schema_version": "1.0.0",
            "record_type": (
                "diagnosis_blind_hcp379_recovery4_selected_recipe_"
                "tractography_gate"
            ),
            "status": "PASS",
            "generated_utc": PRETRACT.utc_now(),
            "diagnosis_labels_used": False,
            "selected_streamline_count": gate["selected_count"],
            "selected_run_id": gate["selected_run_id"],
            "density_is_subject_inclusion_gate": False,
            "compact_tractograms_after_pass": compact,
            "records": gate["records"],
            "builder": PRETRACT.file_record(Path(__file__)),
            "matrix_engine": PRETRACT.file_record(ENGINE_SOURCE),
            "counter_helpers": {
                "tck_header_count": PRETRACT.file_record(
                    TCK_HEADER_COUNTER_SOURCE
                ),
                "numeric_value_count": PRETRACT.file_record(
                    NUMERIC_VALUE_COUNTER_SOURCE
                ),
            },
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--pre-phase-dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--nthreads", type=int, default=16)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--compact-tractograms-after-pass", action="store_true"
    )
    args = parser.parse_args()
    if (
        not 1 <= args.workers <= 4
        or not 1 <= args.nthreads <= 16
        or args.workers * args.nthreads > (os.cpu_count() or 1)
    ):
        parser.error("worker/thread request exceeds the host contract")
    if args.limit is not None and not 1 <= args.limit <= EXPECTED_SCALEUP_N:
        parser.error(f"--limit must be in 1..{EXPECTED_SCALEUP_N}")

    install_recovery4_adapter()
    if not args.execute:
        rows, overlap = validate_prephase_scope()
        plan = build_prephase_plan(
            root=args.root, rows=rows, overlap=overlap
        )
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0

    if shutil.disk_usage(args.root).free < MINIMUM_START_FREE_BYTES:
        raise RuntimeError("storage is below the tractography start floor")
    gate = validate_full_gate(args.root)
    targets = list(gate["rows"])
    if args.limit is not None:
        targets = targets[: args.limit]
    write_execution_gate(
        args.root,
        gate,
        compact=args.compact_tractograms_after_pass,
    )
    for relative in (
        "qc/tractography",
        "qc/tractography_compaction",
        "qc/tractography_compaction_plans",
        "locks/tractography",
        "logs/tractography",
        "manifests",
    ):
        (args.root / relative).mkdir(parents=True, exist_ok=True)
    print(
        f"[{PRETRACT.utc_now()}] HCP379_RECOVERY4_TRACTOGRAPHY_START "
        f"n={len(targets)}/{EXPECTED_SCALEUP_N} "
        f"selected={gate['selected_count']} "
        f"workers={args.workers} threads={args.nthreads}",
        flush=True,
    )
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                process_unit,
                row,
                root=args.root,
                gate=gate,
                nthreads=args.nthreads,
            ): row["unit"]
            for row in targets
        }
        for index, future in enumerate(as_completed(futures), start=1):
            unit = futures[future]
            try:
                state = future.result()
            except Exception as exc:
                state = {
                    "unit": unit,
                    "status": "FAIL_TRACTOGRAPHY",
                    "error": f"{type(exc).__name__}:{exc}",
                }
            if (
                args.compact_tractograms_after_pass
                and state.get("status") == "PASS_ALL_NINE"
            ):
                compacted = ENGINE.compact_validated_tractogram(
                    root=args.root,
                    unit=str(state["unit"]),
                    state=state,
                )
                state = {
                    **state,
                    "tractogram_compaction_status": compacted["status"],
                    "tractogram_reclaimed_bytes": compacted[
                        "reclaimed_bytes"
                    ],
                }
            results.append(state)
            summary = ENGINE.write_summary(args.root)
            print(
                f"[{PRETRACT.utc_now()}] {index}/{len(targets)} "
                f"{unit} {state.get('status')} "
                f"density={state.get('edge_density')} "
                f"error={state.get('error')} "
                f"cohort_pass="
                f"{summary['status_counts'].get('PASS_ALL_NINE', 0)}",
                flush=True,
            )

    summary = ENGINE.write_summary(args.root)
    if args.limit is not None:
        return int(
            any(row.get("status") != "PASS_ALL_NINE" for row in results)
        )
    if summary.get("status") != "PASS":
        return 1
    manifest = ENGINE.build_corrected_manifest(
        root=args.root,
        gate=gate,
        audit_rows=gate["rows"],
    )
    return 0 if manifest.get("status") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
