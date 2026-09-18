#!/home/ec2-user/fsl/bin/python
"""Build the diagnosis-blind 530-subject HCP379 density recovery plan.

This is a planning and release-contract artifact.  It does not mutate imaging
data and it does not use diagnosis or outcomes.  Density is a technical
whole-cohort recipe/release requirement, never a rule for dropping subjects.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


EXP = Path("/home/ec2-user/exp")
ROOT = Path("/data/derivatives/hcp379_v2")
TRIAGE_ROOT = EXP / "research_audit/outputs/hcp_rerun_triage_v1"
TRIAGE = TRIAGE_ROOT / "hcp_rerun_triage_manifest.csv"
DENSITY = (
    EXP
    / "research_audit/outputs/hcp379_existing_302_density_v2/"
    "existing_302_density_subjects.csv"
)
SCALEUP_INPUTS = (
    EXP
    / "research_audit/outputs/hcp379_scaleup_input_audit_v2/"
    "scaleup_input_audit.csv"
)
ARCHIVE_INPUTS = (
    EXP
    / "research_audit/outputs/hcp379_archive_corrected_input_audit_v2/"
    "archive_corrected_input_audit.csv"
)
OUTPUT_ROOT = (
    EXP / "research_audit/outputs/hcp379_density_recovery_plan_v3"
)
OUTPUT_CSV = OUTPUT_ROOT / "hcp379_density_recovery_subjects.csv"
OUTPUT_JSON = OUTPUT_ROOT / "hcp379_density_recovery_plan.json"
THRESHOLD = 0.60
NODES = 379
POSSIBLE_EDGES = NODES * (NODES - 1) // 2

LANE_FILES = {
    "legacy_keep_214": ("keep_high_confidence_214.txt", 214),
    "archive_86": ("archive_recovery_86.txt", 86),
    "tensor_repair_2": ("tensor_weighted_repair_2.txt", 2),
    "corrected_track_rebuild_116": ("corrected_track_rebuild_116.txt", 116),
    "hcp_remap_probe_111": ("hcp_remap_probe_111.txt", 111),
    "fastsurfer_hcp_1": ("fastsurfer_retry_1.txt", 1),
}

GROUP_ORDER = (
    "D0_THRESHOLD_QUALIFIED",
    "D1_NEAR_THRESHOLD",
    "D2_MODERATE_LOW",
    "D3_SEVERE_LOW",
    "D4_CRITICAL_LOW",
    "U0_UNGENERATED",
)

GROUP_CONTRACT = {
    "D0_THRESHOLD_QUALIFIED": {
        "range": "[0.60, 1.00]",
        "meaning": (
            "Current matrix clears the numerical density target only; it is "
            "not automatically release-qualified."
        ),
        "action": "ROUTE_CONCORDANCE_AND_SEMANTIC_QC",
        "debug_checks": (
            "hash and 379x379 numerical contract; 379-node atlas survival; "
            "matched-weight support; route concordance against the locked "
            "corrected-ACT recipe; acquisition and processing provenance"
        ),
        "escalation": (
            "If any semantic, route, spatial or reproducibility gate fails, "
            "regenerate with the uniform corrected recipe."
        ),
    },
    "D1_NEAR_THRESHOLD": {
        "range": "[0.50, 0.60)",
        "meaning": "Near threshold but prohibited from release unchanged.",
        "action": "UNIFORM_CORRECTED_RECOVERY",
        "debug_checks": (
            "input identity and gradients; HCP379 node survival; endpoint "
            "assignment fraction; 5TT/GMWMI alignment; WMFOD and tensor QC; "
            "uniform selected-count tractography and all nine matrices"
        ),
        "escalation": (
            "Use the next cohort-level prespecified recipe only if the locked "
            "recipe fails; never tune streamline count for one subject."
        ),
    },
    "D2_MODERATE_LOW": {
        "range": "[0.40, 0.50)",
        "meaning": "Moderate support deficit; prohibited from release unchanged.",
        "action": "UNIFORM_CORRECTED_FULL_ROUTE",
        "debug_checks": (
            "all D1 checks plus parcel-volume and hemispheric balance, "
            "registration overlap, connected-node count and zero-row audit"
        ),
        "escalation": (
            "Escalate the locked route/recipe uniformly after technical RCA; "
            "do not fill missing edges."
        ),
    },
    "D3_SEVERE_LOW": {
        "range": "[0.20, 0.40)",
        "meaning": "Severe technical deficit; historical matrix is nonreleasable.",
        "action": "FULL_CORRECTED_ROUTE_WITH_ROOT_CAUSE",
        "debug_checks": (
            "D2 checks plus vanished/small parcels, transform direction and "
            "overlap, endpoint rejection, tissue-boundary placement, "
            "tractogram termination and assignment diagnostics"
        ),
        "escalation": (
            "Repair the identified upstream technical cause and rerun the "
            "uniform route; subject remains in the denominator."
        ),
    },
    "D4_CRITICAL_LOW": {
        "range": "[0.00, 0.20)",
        "meaning": "Critical technical failure; historical matrix is nonreleasable.",
        "action": "FULL_ROOT_CAUSE_AND_CORRECTED_REBUILD",
        "debug_checks": (
            "D3 checks plus exact DWI/T1 identity, atlas label histogram, "
            "empty islands, affine/warp sanity, streamline header/provenance "
            "and count-versus-assignment reconciliation"
        ),
        "escalation": (
            "A plateau below 0.60 after the maximum prespecified count rejects "
            "the recipe and requires a diagnosis-blind uniform recipe revision."
        ),
    },
    "U0_UNGENERATED": {
        "range": "no current HCP379-v2 nine-matrix bundle",
        "meaning": "Generate first, then classify by the same numerical contract.",
        "action": "GENERATE_WITH_UNIFORM_CORRECTED_ROUTE",
        "debug_checks": (
            "locked input identity; gradient and Eddy readiness; exact T1/HCP "
            "source; Recovery4 pretract QC; selected-count tractography; all "
            "nine matrices; independent-seed stability where prescribed"
        ),
        "escalation": (
            "Any density below 0.60 enters D1-D4 technical RCA without "
            "exclusion or diagnosis-aware tuning."
        ),
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise ValueError(f"required regular file missing: {resolved}")
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def read_units(path: Path, expected: int) -> set[str]:
    values = {
        value.strip()
        for value in path.read_text(encoding="utf-8").splitlines()
        if value.strip()
    }
    if len(values) != expected:
        raise ValueError(f"{path.name}: expected {expected}, found {len(values)}")
    return values


def optional_float(value: str | None) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    return float(value)


def density_group(value: float | None) -> str:
    if value is None:
        return "U0_UNGENERATED"
    if value >= 0.60:
        return "D0_THRESHOLD_QUALIFIED"
    if value >= 0.50:
        return "D1_NEAR_THRESHOLD"
    if value >= 0.40:
        return "D2_MODERATE_LOW"
    if value >= 0.20:
        return "D3_SEVERE_LOW"
    return "D4_CRITICAL_LOW"


def source_qc(triage: Mapping[str, str], lane: str) -> dict[str, str]:
    if lane == "archive_86":
        return {
            "historical_nodes_present": triage.get("archive_nodes_present", ""),
            "historical_assigned_streamlines": triage.get(
                "archive_assigned_streamlines", ""
            ),
            "historical_assignment_fraction": "",
            "historical_min_hub_degree": triage.get(
                "archive_min_hub_degree", ""
            ),
        }
    return {
        "historical_nodes_present": triage.get("local_nodes_present", ""),
        "historical_assigned_streamlines": triage.get(
            "local_assigned_streamlines", ""
        ),
        "historical_assignment_fraction": triage.get(
            "local_assignment_fraction", ""
        ),
        "historical_min_hub_degree": triage.get(
            "local_min_hub_degree", ""
        ),
    }


def input_readiness(
    unit: str,
    lane: str,
    scaleup: Mapping[str, Mapping[str, str]],
    archive: Mapping[str, Mapping[str, str]],
) -> tuple[str, str]:
    if unit in scaleup:
        row = scaleup[unit]
        return row["route"], row.get("repairable_actions", "")
    if unit in archive:
        row = archive[unit]
        return row["route"], row.get("repairable_actions", "")
    if lane == "fastsurfer_hcp_1":
        return "ANATOMY_HCP_RECOVERY_IN_PROGRESS", ""
    return "CORRECTED_ROUTE_INPUT_AUDIT_REQUIRED", ""


def action_for(lane: str, group: str) -> tuple[str, str]:
    if group == "D0_THRESHOLD_QUALIFIED":
        route_gate = (
            "archive-to-corrected concordance"
            if lane == "archive_86"
            else "legacy-to-corrected concordance"
        )
        return (
            "DENSITY_PASS_ROUTE_PENDING",
            (
                f"Retain as a candidate only after {route_gate}, matrix "
                "semantics, spatial and matched-weight gates pass. Otherwise "
                "send through the uniform corrected route."
            ),
        )
    if group == "U0_UNGENERATED":
        if lane == "fastsurfer_hcp_1":
            return (
                "ANATOMY_THEN_CORRECTED_GENERATION",
                (
                    "Complete exact-T1 FreeSurfer/HCP source recovery, then "
                    "run Recovery4 pretract, locked tractography and all nine "
                    "matrix families."
                ),
            )
        return (
            "CORRECTED_GENERATION",
            (
                "Use the audited Eddy/gradient route, rebuild corrected "
                "anatomy/HCP379/FOD/tensors, run the uniformly selected "
                "tractography recipe, and generate all nine matrices."
            ),
        )
    return (
        "CORRECTED_DENSITY_RECOVERY",
        (
            "Do not promote the historical matrix. Audit/reuse only validated "
            "upstream inputs, rebuild through Recovery4, apply the uniformly "
            "selected tractography recipe, regenerate all nine matrices, and "
            "rerun complete QC."
        ),
    )


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        encoding="utf-8",
        delete=False,
    ) as handle:
        json.dump(dict(value), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "unit",
        "inventory_lane",
        "current_bundle_state",
        "density_group",
        "current_density",
        "current_nonzero_undirected_edges",
        "possible_undirected_edges",
        "current_density_threshold_pass",
        "historical_provenance_class",
        "historical_nodes_present",
        "historical_assigned_streamlines",
        "historical_assignment_fraction",
        "historical_min_hub_degree",
        "historical_failure_signatures",
        "corrected_input_readiness",
        "corrected_input_repair_actions",
        "action_class",
        "immediate_action",
        "required_debug_checks",
        "escalation_rule",
        "final_release_rule",
        "diagnosis_or_outcomes_used",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        newline="",
        encoding="utf-8",
        delete=False,
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def build() -> dict[str, Any]:
    lane_by_unit: dict[str, str] = {}
    lane_sets: dict[str, set[str]] = {}
    for lane, (filename, expected) in LANE_FILES.items():
        units = read_units(TRIAGE_ROOT / filename, expected)
        overlap = set(lane_by_unit).intersection(units)
        if overlap:
            raise ValueError(f"lane overlap for {sorted(overlap)[:3]}")
        lane_sets[lane] = units
        lane_by_unit.update({unit: lane for unit in units})
    if len(lane_by_unit) != 530:
        raise ValueError(f"lane partition has {len(lane_by_unit)} units, not 530")

    triage_rows = read_csv(TRIAGE)
    triage = {row["fsid"]: row for row in triage_rows}
    if len(triage) != 530 or set(triage) != set(lane_by_unit):
        raise ValueError("triage identities do not match exact 530 lane partition")

    density_rows = read_csv(DENSITY)
    density = {row["unit"]: row for row in density_rows}
    expected_existing = (
        lane_sets["legacy_keep_214"]
        | lane_sets["archive_86"]
        | lane_sets["tensor_repair_2"]
    )
    if len(density) != 302 or set(density) != expected_existing:
        raise ValueError("density audit does not match exact 302 existing units")

    scaleup_rows = read_csv(SCALEUP_INPUTS)
    scaleup = {row["unit"]: row for row in scaleup_rows}
    archive_rows = read_csv(ARCHIVE_INPUTS)
    archive = {row["unit"]: row for row in archive_rows}
    expected_scaleup = (
        lane_sets["corrected_track_rebuild_116"]
        | lane_sets["hcp_remap_probe_111"]
    )
    if len(scaleup) != 227 or set(scaleup) != expected_scaleup:
        raise ValueError("scale-up input audit does not match exact 227 units")
    if len(archive) != 86 or set(archive) != lane_sets["archive_86"]:
        raise ValueError("archive input audit does not match exact 86 units")

    rows: list[dict[str, Any]] = []
    for unit in sorted(lane_by_unit):
        lane = lane_by_unit[unit]
        old = triage[unit]
        density_row = density.get(unit)
        value = (
            float(density_row["density"]) if density_row is not None else None
        )
        group = density_group(value)
        readiness, repair = input_readiness(unit, lane, scaleup, archive)
        action_class, immediate_action = action_for(lane, group)
        historical = source_qc(old, lane)
        contract = GROUP_CONTRACT[group]
        rows.append(
            {
                "unit": unit,
                "inventory_lane": lane,
                "current_bundle_state": (
                    "NINE_MATRIX_BUNDLE_PRESENT"
                    if density_row is not None
                    else "NOT_YET_GENERATED"
                ),
                "density_group": group,
                "current_density": "" if value is None else value,
                "current_nonzero_undirected_edges": (
                    ""
                    if density_row is None
                    else int(density_row["nonzero_undirected_edges"])
                ),
                "possible_undirected_edges": POSSIBLE_EDGES,
                "current_density_threshold_pass": (
                    value is not None and value >= THRESHOLD
                ),
                "historical_provenance_class": old.get(
                    "provenance_class", ""
                ),
                **historical,
                "historical_failure_signatures": old.get("reasons", ""),
                "corrected_input_readiness": readiness,
                "corrected_input_repair_actions": repair,
                "action_class": action_class,
                "immediate_action": immediate_action,
                "required_debug_checks": contract["debug_checks"],
                "escalation_rule": contract["escalation"],
                "final_release_rule": (
                    "density>=0.60 AND 379x379 semantic/numerical QC AND "
                    "matched nine-matrix support AND spatial/route/"
                    "reproducibility gates; never exclude to satisfy density"
                ),
                "diagnosis_or_outcomes_used": False,
            }
        )

    group_counts = Counter(row["density_group"] for row in rows)
    expected_groups = {
        "D0_THRESHOLD_QUALIFIED": 58,
        "D1_NEAR_THRESHOLD": 97,
        "D2_MODERATE_LOW": 61,
        "D3_SEVERE_LOW": 68,
        "D4_CRITICAL_LOW": 18,
        "U0_UNGENERATED": 228,
    }
    if dict(group_counts) != expected_groups:
        raise ValueError(
            f"density-group counts differ: {dict(group_counts)}"
        )

    lane_group: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        lane_group[row["inventory_lane"]][row["density_group"]] += 1
    threshold_pass_n = sum(
        bool(row["current_density_threshold_pass"]) for row in rows
    )
    if threshold_pass_n != 58:
        raise ValueError(f"threshold pass count is {threshold_pass_n}, not 58")

    atomic_csv(OUTPUT_CSV, rows)
    plan = {
        "record_type": "diagnosis_blind_hcp379_density_recovery_plan",
        "schema_version": "3.0.0",
        "generated_utc": utc_now(),
        "status": "PLAN_READY_EXECUTION_PENDING",
        "goal_acceptance_contract": {
            "target_subjects": 530,
            "atlas_nodes": NODES,
            "possible_undirected_edges": POSSIBLE_EDGES,
            "minimum_density_inclusive": THRESHOLD,
            "required_final_subjects_at_or_above_threshold": 530,
            "current_subjects_at_or_above_threshold": threshold_pass_n,
            "subject_exclusion_to_meet_density_allowed": False,
            "edge_imputation_or_artificial_filling_allowed": False,
            "diagnosis_or_outcome_aware_tuning_allowed": False,
            "per_subject_streamline_count_tuning_allowed": False,
            "historical_output_overwrite_allowed": False,
            "density_is_whole_cohort_technical_release_gate": True,
            "density_alone_establishes_anatomical_validity": False,
        },
        "current_inventory": {
            "exact_roster_n": len(rows),
            "existing_nine_matrix_bundles": len(density),
            "not_yet_generated": len(rows) - len(density),
            "existing_at_or_above_0_60": threshold_pass_n,
            "existing_below_0_60": len(density) - threshold_pass_n,
            "remaining_to_produce_or_recover_at_or_above_0_60": (
                len(rows) - threshold_pass_n
            ),
        },
        "density_groups": {
            name: {
                **GROUP_CONTRACT[name],
                "n": group_counts[name],
            }
            for name in GROUP_ORDER
        },
        "counts_by_inventory_lane_and_density_group": {
            lane: {
                group: lane_group[lane][group]
                for group in GROUP_ORDER
                if lane_group[lane][group]
            }
            for lane in LANE_FILES
        },
        "execution_policy": {
            "phase_b": (
                "Choose the smallest prespecified uniform streamline count "
                "whose complete blinded canary set and independent-seed "
                "replicates all meet density>=0.60 plus convergence, node, "
                "assignment, spatial and matrix-semantic gates."
            ),
            "no_qualified_phase_b_recipe": (
                "Return NO_DENSITY_QUALIFIED_RECIPE and do not scale."
            ),
            "bulk_recovery": (
                "Run D1-D4 and U0 through the locked corrected route without "
                "using diagnosis, outcomes or subject-specific count tuning."
            ),
            "bulk_outlier": (
                "A final density<0.60 is a technical failure requiring RCA. "
                "Escalate a prespecified route/recipe uniformly after "
                "validation; do not remove the subject or fabricate edges."
            ),
            "D0_retention": (
                "D0 matrices are candidates only. Retain them only if route "
                "concordance and complete semantic/spatial/matched-weight QC "
                "pass; otherwise regenerate with the corrected recipe."
            ),
            "final_gate": (
                "Release assembly must independently recompute count density "
                "and fail closed unless all 530 values are >=0.60."
            ),
        },
        "records": {
            "subject_plan": file_record(OUTPUT_CSV),
            "existing_density_audit": file_record(DENSITY),
            "triage": file_record(TRIAGE),
            "scaleup_input_audit": file_record(SCALEUP_INPUTS),
            "archive_input_audit": file_record(ARCHIVE_INPUTS),
            "builder": file_record(Path(__file__)),
        },
        "diagnosis_or_outcomes_used": False,
    }
    atomic_json(OUTPUT_JSON, plan)
    return plan


if __name__ == "__main__":
    result = build()
    print(
        json.dumps(
            {
                "status": result["status"],
                "groups": {
                    key: value["n"]
                    for key, value in result["density_groups"].items()
                },
                "subject_plan": result["records"]["subject_plan"]["path"],
                "plan": str(OUTPUT_JSON),
            },
            indent=2,
            sort_keys=True,
        )
    )
