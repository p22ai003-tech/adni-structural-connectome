#!/usr/bin/env python3
"""Emit a concise live status snapshot for the HCP379-v2 recovery."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("/data/derivatives/hcp379_v2")
PRETRACT_COMPLETION = (
    ROOT / "pretract_recovery4/pretract_recovery4_manifest.json"
)
VISUAL_REVIEW_MANIFEST = (
    ROOT
    / "review_recovery4_corrected_overlay/"
    "hcp379_visual_review_pack_recovery4_corrected_overlay_manifest.json"
)
CANDIDATE_VISUAL_REVIEW_MANIFEST = (
    ROOT
    / "review_recovery4_candidate_overlay_v6/"
    "hcp379_visual_review_pack_recovery4_candidate_v6_manifest.json"
)
CANDIDATE_PRETRACT_MASTER = (
    ROOT
    / "pretract_candidate_recovery4_v4/"
    "pretract_candidate_recovery4_v4_manifest.json"
)
CANDIDATE_PRETRACT_VALIDATION = Path(
    "/home/ec2-user/exp/research_audit/outputs/"
    "hcp379_009_candidate_pretract_v4/validation.json"
)
ROUND1_TRANSCRIPTION = (
    ROOT
    / "review_recovery4_corrected_overlay/"
    "human_visual_qc_recovery4_round1_transcription_manifest.json"
)
V6_COMPLETED_REVIEW = (
    ROOT
    / "review_recovery4_candidate_overlay_v6/"
    "human_visual_qc_recovery4_candidate_v6.csv"
)
V6_NUMBERS_TEMPLATE = (
    ROOT
    / "review_recovery4_candidate_overlay_v6/"
    "human_visual_qc_recovery4_candidate_v6_template.numbers"
)
V6_COMPLETED_NUMBERS = (
    ROOT
    / "review_recovery4_candidate_overlay_v6/"
    "human_visual_qc_recovery4_candidate_v6.numbers"
)
V6_NUMBERS_MANIFEST = (
    ROOT
    / "review_recovery4_candidate_overlay_v6/"
    "human_visual_qc_recovery4_candidate_v6_numbers_manifest.json"
)
V6_NUMBERS_TRANSCRIPTION = (
    ROOT
    / "review_recovery4_candidate_overlay_v6/"
    "human_visual_qc_recovery4_candidate_v6_numbers_transcription.json"
)
V6_NUMBERS_VALIDATION = Path(
    "/home/ec2-user/exp/research_audit/outputs/"
    "hcp379_v6_numbers_review_workflow/validation.json"
)
V6_NUMBERS_TRANSCRIBER = Path(
    "/home/ec2-user/exp/scripts/hcp/"
    "transcribe_hcp379_v6_numbers_review.py"
)
PROMOTION_BUILDER = Path(
    "/home/ec2-user/exp/scripts/hcp/"
    "promote_hcp379_candidate_review_v6.py"
)
PROMOTED_PRETRACT_MASTER = (
    ROOT
    / "pretract_promoted_recovery4_v6/"
    "pretract_promoted_recovery4_v6_manifest.json"
)
PROMOTION_RECEIPT = (
    ROOT / "pretract_promoted_recovery4_v6/promotion_receipt.json"
)
CORRECTED_OVERLAY_QUANTITATIVE_AUDIT = (
    ROOT
    / "review_recovery4_corrected_overlay/"
    "hcp379_corrected_overlay_quantitative_audit.json"
)
TARGETED_009_REGISTRATION_AUDIT = (
    ROOT
    / "diagnostics/registration_recovery4_targeted/"
    "009_S_4324_I1186579/targeted_registration_audit.json"
)
CANONICAL_CORRECTED_HUMAN_QC = (
    ROOT
    / "review_recovery4_corrected_overlay/"
    "human_visual_qc_recovery4_corrected_overlay.csv"
)
RECOVERY4_ATLAS_SUMMARY = (
    ROOT
    / "corrected_atlas_canary_recovery4/"
    "corrected_atlas_canary_recovery4_summary.json"
)
RECOVERY4_SCALAR_SUMMARY = (
    ROOT / "pretract_scalar_recovery4/scalar_recovery4_summary.json"
)
RECOVERY4_SPATIAL_SUMMARY = (
    ROOT / "pretract_spatial_recovery4/spatial_recovery4_summary.json"
)
PHASE_B_ROOT = ROOT / "phase_b_stability_recovery4"
PHASE_B_SUBJECTS_ROOT = Path(
    "/data/derivatives/scforge_v2/"
    "h04a_r1_recovery_20260719_retry4/subjects"
)
STABILITY_VALIDATION = Path(
    "/data/derivatives/scforge_v2/h04a_r1_recovery_20260719_retry4/"
    "publication/hcp379_recipe_stability_validation.json"
)
SCALEUP_INPUT_AUDIT = Path(
    "/home/ec2-user/exp/research_audit/outputs/"
    "hcp379_scaleup_input_audit_v2/scaleup_input_audit_summary.json"
)
ARCHIVE_CORRECTED_INPUT_AUDIT = Path(
    "/home/ec2-user/exp/research_audit/outputs/"
    "hcp379_archive_corrected_input_audit_v2/"
    "archive_corrected_input_audit_summary.json"
)
ARCHIVE_ROUTE_CONCORDANCE = (
    ROOT / "calibration/archive_route_concordance.json"
)
KEEP_ROUTE_CONCORDANCE = (
    ROOT / "calibration/keep_route_concordance.json"
)
KEEP_ROUTE_RUNNER = Path(
    "/home/ec2-user/exp/scripts/hcp/"
    "run_hcp379_keep_route_concordance_recovery4_v3.py"
)
KEEP_ROUTE_PLAN = (
    ROOT / "calibration/keep_route_concordance_recovery4_plan.json"
)
KEEP_ROUTE_VALIDATION = Path(
    "/home/ec2-user/exp/research_audit/outputs/"
    "hcp379_keep_route_concordance_recovery4_v3/validation.json"
)
ARCHIVE_CALIBRATION_RUNNERS = {
    "pretract": Path(
        "/home/ec2-user/exp/scripts/hcp/"
        "run_hcp379_archive_calibration_pretract_recovery4_v3.py"
    ),
    "primary_tractography": Path(
        "/home/ec2-user/exp/scripts/hcp/"
        "run_hcp379_archive_calibration_tractography_recovery4_v3.py"
    ),
    "independent_seed": Path(
        "/home/ec2-user/exp/scripts/hcp/"
        "run_hcp379_archive_calibration_replicate_recovery4_v3.py"
    ),
    "concordance_validator": Path(
        "/home/ec2-user/exp/research_audit/"
        "validate_hcp379_archive_route_concordance_recovery4_v3.py"
    ),
}
ARCHIVE_CALIBRATION_ROOT = (
    ROOT / "archive_corrected_calibration_recovery4"
)
ARCHIVE_CALIBRATION_PLANS = {
    "pretract": (
        ARCHIVE_CALIBRATION_ROOT
        / "manifests/archive_pretract_recovery4_plan.json"
    ),
    "primary_tractography": (
        ARCHIVE_CALIBRATION_ROOT
        / "manifests/archive_tractography_recovery4_plan.json"
    ),
    "independent_seed": (
        ARCHIVE_CALIBRATION_ROOT
        / "manifests/archive_replicate_recovery4_plan.json"
    ),
}
ARCHIVE_CALIBRATION_VALIDATION = Path(
    "/home/ec2-user/exp/research_audit/outputs/"
    "hcp379_archive_calibration_recovery4_v3/validation.json"
)
EXISTING_302_DENSITY_AUDIT = Path(
    "/home/ec2-user/exp/research_audit/outputs/"
    "hcp379_existing_302_density_v2/existing_302_density_audit.json"
)
LIVE_COHORT_INVENTORY_AUDIT = (
    ROOT
    / "audits/live_cohort_inventory_v1/"
    "hcp379_live_cohort_inventory_summary.json"
)
DENSITY_RECOVERY_PLAN = Path(
    "/home/ec2-user/exp/research_audit/outputs/"
    "hcp379_density_recovery_plan_v3/hcp379_density_recovery_plan.json"
)
DENSITY_EXECUTION_TOPOLOGY = Path(
    "/home/ec2-user/exp/research_audit/outputs/"
    "hcp379_density_execution_topology_v3/"
    "hcp379_density_execution_topology.json"
)
DENSITY_EXECUTION_VALIDATION = Path(
    "/home/ec2-user/exp/research_audit/outputs/"
    "hcp379_density_execution_topology_v3/validation.json"
)
LEGACY_DENSITY_INPUT_AUDIT = Path(
    "/home/ec2-user/exp/research_audit/outputs/"
    "hcp379_legacy_density_recovery_input_audit_v3/"
    "legacy_density_recovery_input_audit_summary.json"
)
LEGACY_DENSITY_PRETRACT_PLAN = (
    ROOT
    / "corrected_legacy_tensor_recovery4/manifests/"
    "legacy_tensor_density_pretract_recovery4_plan.json"
)
LEGACY_DENSITY_TRACT_PLAN = (
    ROOT
    / "corrected_legacy_tensor_recovery4/manifests/"
    "legacy_tensor_density_tractography_recovery4_plan.json"
)
LEGACY_DENSITY_VALIDATION = Path(
    "/home/ec2-user/exp/research_audit/outputs/"
    "hcp379_legacy_density_recovery4_v3/validation.json"
)
ARCHIVE_LOW_PRETRACT_PLAN = (
    ROOT
    / "corrected_archive_low_recovery4/manifests/"
    "archive_low_density_pretract_recovery4_plan.json"
)
ARCHIVE_LOW_TRACT_PLAN = (
    ROOT
    / "corrected_archive_low_recovery4/manifests/"
    "archive_low_density_tractography_recovery4_plan.json"
)
ARCHIVE_LOW_VALIDATION = Path(
    "/home/ec2-user/exp/research_audit/outputs/"
    "hcp379_archive_low_recovery4_v3/validation.json"
)
ARCHIVE_D0_PRETRACT_PLAN = (
    ROOT
    / "corrected_archive_D0_recovery4/manifests/"
    "archive_D0_pretract_recovery4_plan.json"
)
ARCHIVE_D0_TRACT_PLAN = (
    ROOT
    / "corrected_archive_D0_recovery4/manifests/"
    "archive_D0_tractography_recovery4_plan.json"
)
ARCHIVE_D0_VALIDATION = Path(
    "/home/ec2-user/exp/research_audit/outputs/"
    "hcp379_archive_D0_recovery4_v3/validation.json"
)
ARCHIVE_86_MERGE_PLAN = (
    ROOT
    / "corrected_archive_recovery4/manifests/"
    "corrected_archive_86_merge_plan.json"
)
ARCHIVE_86_MERGE_OUTPUT = (
    ROOT
    / "corrected_archive_recovery4/manifests/"
    "corrected_archive_86_release_manifest.json"
)
ARCHIVE_86_MERGE_VALIDATION = Path(
    "/home/ec2-user/exp/research_audit/outputs/"
    "hcp379_corrected_archive_86_recovery4_v3/validation.json"
)
SCALEUP_PRETRACT_SUMMARY = (
    ROOT
    / "corrected_scaleup_recovery4/manifests/"
    "pretract_recovery4_summary.json"
)
SCALEUP_PRETRACT_PLAN = (
    ROOT
    / "corrected_scaleup_recovery4/manifests/"
    "pretract_recovery4_plan.json"
)
SCALEUP_PRETRACT_VALIDATION = Path(
    "/home/ec2-user/exp/research_audit/outputs/"
    "hcp379_scaleup_pretract_recovery4_v3/validation.json"
)
SCALEUP_TRACT_SUMMARY = (
    ROOT
    / "corrected_scaleup_recovery4/manifests/"
    "tractography_summary.json"
)
SCALEUP_TRACT_PLAN = (
    ROOT
    / "corrected_scaleup_recovery4/manifests/"
    "tractography_recovery4_selected_recipe_plan.json"
)
SCALEUP_TRACT_VALIDATION = Path(
    "/home/ec2-user/exp/research_audit/outputs/"
    "hcp379_scaleup_tractography_recovery4_v3/validation.json"
)
CORRECTED_227_MANIFEST = (
    ROOT
    / "corrected_scaleup_recovery4/manifests/"
    "corrected_227_release_manifest.json"
)
FINAL_RELEASE_READINESS = (
    ROOT
    / "release_candidate_v2/hcp379_release_readiness.json"
)
FINAL_RELEASE_VALIDATION = Path(
    "/home/ec2-user/exp/research_audit/outputs/"
    "hcp379_final_release_recovery4_v3/validation.json"
)
INTEGRATION_VERIFY_PREFLIGHT = Path(
    "/home/ec2-user/exp/research_audit/outputs/"
    "hcp379_integration_release_verifier_v1/preflight.json"
)
INTEGRATION_VERIFY_VALIDATION = Path(
    "/home/ec2-user/exp/research_audit/outputs/"
    "hcp379_integration_release_verifier_v1/validation.json"
)
INTEGRATION_VERIFY_OUTPUT = (
    ROOT
    / "release_validation_v1/"
    "integration_verification.json"
)
ANALYSIS_HANDOFF_PREFLIGHT = Path(
    "/home/ec2-user/exp/research_audit/outputs/"
    "hcp379_verified_analysis_handoff_v1/preflight.json"
)
ANALYSIS_HANDOFF_VALIDATION = Path(
    "/home/ec2-user/exp/research_audit/outputs/"
    "hcp379_verified_analysis_handoff_v1/validation.json"
)
ANALYSIS_HANDOFF_RECEIPT = (
    ROOT
    / "analysis_handoff_v1/"
    "verified_handoff_receipt.json"
)
STORAGE_CAPACITY_FORECAST = (
    ROOT / "manifests/storage_capacity_forecast.json"
)
PARALLEL_PRETRACT_PLAN = (
    ROOT / "parallel_recovery4/parallel_phase_b_pretract_plan.json"
)
PARALLEL_PRETRACT_VALIDATION = Path(
    "/home/ec2-user/exp/research_audit/outputs/"
    "hcp379_parallel_pretract_recovery4_v3/validation.json"
)
PARALLEL_PRETRACT_ATTEMPTS = (
    ROOT / "parallel_recovery4/attempts"
)
PARALLEL_TRACT_RELEASE_PLAN = (
    ROOT
    / "parallel_recovery4/"
    "parallel_tractography_release_plan.json"
)
PARALLEL_TRACT_RELEASE_STATE = (
    ROOT
    / "parallel_recovery4/"
    "parallel_tractography_release_state.json"
)
PARALLEL_TRACT_RELEASE_VALIDATION = Path(
    "/home/ec2-user/exp/research_audit/outputs/"
    "hcp379_parallel_tractography_release_recovery4_v3/"
    "validation.json"
)
FREESURFER_SOURCE_ATTESTATION = (
    ROOT
    / "fastsurfer_repair/114_S_6347_I1344943/"
    "hcp_source_recovery_attestation_v3.json"
)
FREESURFER_PRETRACT_PLAN = (
    ROOT
    / "corrected_freesurfer_recovery4/manifests/"
    "freesurfer_pretract_recovery4_plan.json"
)
FREESURFER_TRACT_PLAN = (
    ROOT
    / "corrected_freesurfer_recovery4/manifests/"
    "freesurfer_tractography_recovery4_plan.json"
)
FREESURFER_CORRECTED_VALIDATION = Path(
    "/home/ec2-user/exp/research_audit/outputs/"
    "hcp379_freesurfer_recovery4_v3/validation.json"
)
LEGACY_SCALEUP_PRETRACT_RUNNER = Path(
    "/home/ec2-user/exp/scripts/hcp/hcp379_scaleup_pretract_v2.py"
)
RECOVERY4_SCALEUP_PRETRACT_RUNNER = Path(
    "/home/ec2-user/exp/scripts/hcp/hcp379_scaleup_pretract_recovery4_v3.py"
)
LEGACY_SCALEUP_TRACT_RUNNER = Path(
    "/home/ec2-user/exp/scripts/hcp/hcp379_scaleup_tractography_v2.py"
)
RECOVERY4_SCALEUP_TRACT_RUNNER = Path(
    "/home/ec2-user/exp/scripts/hcp/"
    "hcp379_scaleup_tractography_recovery4_v3.py"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def pid_alive(value: Any) -> bool:
    try:
        pid = int(value)
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    return True


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def tmux_sessions() -> list[str]:
    run = subprocess.run(
        ["tmux", "list-sessions", "-F", "#{session_name}"],
        check=False,
        capture_output=True,
        text=True,
    )
    return sorted(line.strip() for line in run.stdout.splitlines() if line.strip())


def recovery3_progress() -> dict[str, Any] | None:
    attempts = Path(
        "/data/derivatives/scforge_v2/"
        "h04a_r1_recovery_20260719_retry4/attempts"
    )
    logs = sorted(
        attempts.glob("*pre-tractography-recovery3-*.snakemake.log"),
        key=lambda path: path.stat().st_mtime_ns,
    )
    if not logs:
        return None
    path = logs[-1]
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    matches = re.findall(r"(\d+) of (\d+) steps", text)
    completed = int(matches[-1][0]) if matches else 0
    total = int(matches[-1][1]) if matches else 166
    return {
        "attempt_log": str(path),
        "completed_steps": completed,
        "total_steps": total,
        "progress_fraction": completed / total if total else 0.0,
        "error_rule_count": text.count("Error in rule"),
    }


def phase_b_live_status() -> dict[str, Any]:
    """Discover the current Phase-B DAG and count its retained artifacts.

    The original parallel supervisor may have exited while a deliberately
    resumed Phase-B wrapper remains active.  Process discovery therefore
    supplements, and when live supersedes, the historical supervisor state.
    """

    processes: list[dict[str, Any]] = []
    run = subprocess.run(
        ["ps", "-eo", "pid=,ppid=,state=,etime=,args="],
        check=False,
        capture_output=True,
        text=True,
    )
    line_pattern = re.compile(
        r"^\s*(\d+)\s+(\d+)\s+(\S+)\s+(\S+)\s+(.*)$"
    )
    for line in run.stdout.splitlines():
        match = line_pattern.match(line)
        if not match:
            continue
        pid, ppid, state, elapsed, command = match.groups()
        processes.append(
            {
                "pid": int(pid),
                "ppid": int(ppid),
                "state": state,
                "elapsed": elapsed,
                "command": command,
            }
        )

    wrappers = [
        row
        for row in processes
        if "run_hcp379_phase_b_stability_v2.py" in row["command"]
        and "--resume-existing-phase-b" in row["command"]
    ]
    dags = [
        row
        for row in processes
        if "Snakefile_h04a_r1_retry4_phase_b_hcp379_recovery4"
        in row["command"]
        and "hcp379_stability_evidence" in row["command"]
        and "--target-jobs" not in row["command"]
    ]

    active_jobs: list[dict[str, Any]] = []
    tool_patterns = (
        ("tckgen", "tckgen "),
        ("tcksift2", "tcksift2 "),
        ("tck2connectome", "tck2connectome "),
        ("tcksample", "tcksample "),
    )
    subject_run_pattern = re.compile(
        r"/subjects/([^/\s]+)/09_hcp379_stability/([^/\s]+)"
    )
    for row in processes:
        executable = row["command"].split(maxsplit=1)[0]
        if Path(executable).name in {"bash", "sh", "dash"}:
            continue
        tool = next(
            (
                name
                for name, pattern in tool_patterns
                if pattern in row["command"]
            ),
            None,
        )
        if tool is None:
            continue
        identity = subject_run_pattern.search(row["command"])
        unit = identity.group(1) if identity else None
        hcp_run = identity.group(2) if identity else None
        job = {
            "pid": row["pid"],
            "ppid": row["ppid"],
            "state": row["state"],
            "elapsed": row["elapsed"],
            "tool": tool,
            "unit": unit,
            "hcp_run": hcp_run,
        }
        if tool == "tckgen" and unit and hcp_run:
            log_path = (
                PHASE_B_SUBJECTS_ROOT
                / unit
                / "logs"
                / f"09_hcp379_{hcp_run}_tckgen.log"
            )
            try:
                with log_path.open("rb") as handle:
                    handle.seek(0, os.SEEK_END)
                    size = handle.tell()
                    handle.seek(max(0, size - 65_536))
                    tail = handle.read().decode(
                        "utf-8", errors="replace"
                    )
            except OSError:
                tail = ""
            selected_matches = re.findall(
                r"([\d,]+)\s+selected", tail
            )
            requested_match = re.search(r"(\d+)m$", hcp_run)
            requested = (
                int(requested_match.group(1)) * 1_000_000
                if requested_match
                else None
            )
            selected = (
                int(selected_matches[-1].replace(",", ""))
                if selected_matches
                else None
            )
            job["progress_log"] = str(log_path)
            job["selected_streamlines"] = selected
            job["requested_streamlines"] = requested
            job["selected_fraction"] = (
                selected / requested
                if selected is not None and requested
                else None
            )
        active_jobs.append(job)

    stability_roots = sorted(
        PHASE_B_SUBJECTS_ROOT.glob("*/09_hcp379_stability")
    )
    ten_m_pair_ready_units: list[str] = []
    ten_m_seed_ready_n = 0
    for stability_root in stability_roots:
        seed_ready: dict[str, bool] = {}
        for seed_class in ("primary", "independent"):
            run_root = stability_root / f"{seed_class}_10m"
            seed_ready[seed_class] = (
                (run_root / "tracks.tck").is_file()
                and (run_root / "tractography_parameters.json").is_file()
            )
            ten_m_seed_ready_n += int(seed_ready[seed_class])
        if all(seed_ready.values()):
            ten_m_pair_ready_units.append(stability_root.parent.name)
    artifact_counts = {
        "subject_roots": len(stability_roots),
        "planned_tractograms": 90,
        "planned_10m_seed_records": 30,
        "ten_m_seed_ready_n": ten_m_seed_ready_n,
        "planned_10m_pairs": 15,
        "ten_m_pair_ready_n": len(ten_m_pair_ready_units),
        "ten_m_pair_ready_units": ten_m_pair_ready_units,
        "tractograms": sum(
            1 for root in stability_roots for _ in root.glob("*/tracks.tck")
        ),
        "partial_tractograms": sum(
            1
            for root in stability_roots
            for _ in root.glob("*/tracks.tck.partial.tck")
        ),
        "tractography_parameter_records": sum(
            1
            for root in stability_roots
            for _ in root.glob("*/tractography_parameters.json")
        ),
        "sift2_weight_sets": sum(
            1
            for root in stability_roots
            for _ in root.glob("*/sift2_weights.txt")
        ),
        "run_records": sum(
            1
            for root in stability_roots
            for _ in root.glob("*/run_record.json")
        ),
        "matrix_csvs": sum(
            1
            for root in stability_roots
            for _ in root.glob("*/matrices/*.csv")
        ),
    }

    attempts = sorted(
        PHASE_B_ROOT.glob("*.attempt.json"),
        key=lambda path: path.stat().st_mtime_ns,
    )
    logs = sorted(
        PHASE_B_ROOT.glob("*.snakemake.log"),
        key=lambda path: path.stat().st_mtime_ns,
    )
    process_alive = bool(wrappers or dags)
    return {
        "status": "RUNNING" if process_alive else "NOT_RUNNING",
        "process_alive": process_alive,
        "live_dag_started": process_alive,
        "returncode": None if process_alive else None,
        "wrapper_pids": [row["pid"] for row in wrappers],
        "snakemake_pids": [row["pid"] for row in dags],
        "active_jobs": active_jobs,
        "artifact_counts": artifact_counts,
        "latest_attempt": str(attempts[-1]) if attempts else None,
        "latest_log": str(logs[-1]) if logs else None,
        "observed_utc": utc_now(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()
    root = args.root

    archive_states = []
    for path in sorted((root / "qc/subjects").glob("*.json")):
        state = load_json(path)
        if state and state.get("lane") == "ARCHIVE_RECOVERY":
            archive_states.append(state)
    archive_status = Counter(str(row.get("status", "UNKNOWN")) for row in archive_states)
    weighted_status = Counter(
        str(row.get("weighted_status", "NOT_STARTED")) for row in archive_states
    )

    tensor = {}
    for path in sorted((root / "tensor_repairs").glob("*/result.json")):
        state = load_json(path)
        if state:
            tensor[str(state.get("fsid", path.parent.name))] = {
                "status": state.get("status"),
                "error": state.get("error"),
            }

    eddy_state = load_json(
        root
        / "tensor_repairs/016_S_7002_I1493842/eddy_repair/"
        "eddy_repair_result.json"
    )
    freesurfer_state = load_json(
        root
        / "fastsurfer_repair/114_S_6347_I1344943/"
        "standard_freesurfer_result.json"
    )
    recovered_hcp_state = load_json(FREESURFER_SOURCE_ATTESTATION)
    corrected_atlas_state = load_json(RECOVERY4_ATLAS_SUMMARY)
    scalar_recovery4_state = load_json(RECOVERY4_SCALAR_SUMMARY)
    spatial_recovery4_state = load_json(RECOVERY4_SPATIAL_SUMMARY)
    visual_review_state = load_json(VISUAL_REVIEW_MANIFEST)
    candidate_visual_review_state = load_json(
        CANDIDATE_VISUAL_REVIEW_MANIFEST
    )
    candidate_pretract_state = load_json(CANDIDATE_PRETRACT_MASTER)
    candidate_pretract_validation = load_json(
        CANDIDATE_PRETRACT_VALIDATION
    )
    round1_transcription = load_json(ROUND1_TRANSCRIPTION)
    v6_numbers_manifest = load_json(V6_NUMBERS_MANIFEST)
    v6_numbers_validation = load_json(V6_NUMBERS_VALIDATION)
    v6_numbers_transcription = load_json(V6_NUMBERS_TRANSCRIPTION)
    promoted_pretract_state = load_json(PROMOTED_PRETRACT_MASTER)
    promotion_receipt_state = load_json(PROMOTION_RECEIPT)
    v6_promotion_ready = bool(
        promotion_receipt_state
        and promotion_receipt_state.get("status") == "PASS"
        and promoted_pretract_state
        and promoted_pretract_state.get("record_type")
        == "diagnosis_blind_hcp379_pretract_recovery4_manifest"
        and promoted_pretract_state.get("route_substitution", {}).get(
            "promotion_status"
        )
        == "HUMAN_VISUAL_QC_PASS_PROMOTED"
        and CANONICAL_CORRECTED_HUMAN_QC.is_file()
    )
    corrected_overlay_quantitative_state = load_json(
        CORRECTED_OVERLAY_QUANTITATIVE_AUDIT
    )
    targeted_009_registration_state = load_json(
        TARGETED_009_REGISTRATION_AUDIT
    )
    stability_state = load_json(STABILITY_VALIDATION)
    scaleup_input_state = load_json(SCALEUP_INPUT_AUDIT)
    archive_corrected_input_state = load_json(
        ARCHIVE_CORRECTED_INPUT_AUDIT
    )
    archive_concordance_state = load_json(ARCHIVE_ROUTE_CONCORDANCE)
    keep_concordance_state = load_json(KEEP_ROUTE_CONCORDANCE)
    keep_route_plan = load_json(KEEP_ROUTE_PLAN)
    keep_route_validation = load_json(KEEP_ROUTE_VALIDATION)
    keep_route_package_ready = (
        KEEP_ROUTE_RUNNER.is_file()
        and keep_route_plan is not None
        and keep_route_plan.get("status") == "PRE_PHASE_DRY_RUN_PASS"
        and keep_route_plan.get("execution_authorized") is False
        and keep_route_plan.get("imaging_executed_by_plan") is False
        and keep_route_validation is not None
        and keep_route_validation.get("status") == "PASS"
        and keep_route_validation.get("passed_checks") == 13
        and keep_route_validation.get("total_checks") == 13
    )
    archive_calibration_plans = {
        name: load_json(path)
        for name, path in ARCHIVE_CALIBRATION_PLANS.items()
    }
    archive_calibration_validation = load_json(
        ARCHIVE_CALIBRATION_VALIDATION
    )
    existing_302_density = load_json(EXISTING_302_DENSITY_AUDIT)
    live_cohort_inventory = load_json(LIVE_COHORT_INVENTORY_AUDIT)
    archive_plan_statuses = {
        "pretract": "PRE_HUMAN_DRY_RUN_PASS",
        "primary_tractography": "PRE_PHASE_DRY_RUN_PASS",
        "independent_seed": "PRE_PHASE_DRY_RUN_PASS",
    }
    archive_calibration_package_ready = (
        all(path.is_file() for path in ARCHIVE_CALIBRATION_RUNNERS.values())
        and all(
            archive_calibration_plans[name] is not None
            and archive_calibration_plans[name].get("status")
            == expected
            and archive_calibration_plans[name].get(
                "imaging_executed_by_plan"
            )
            is False
            for name, expected in archive_plan_statuses.items()
        )
        and archive_calibration_validation is not None
        and archive_calibration_validation.get("status") == "PASS"
        and archive_calibration_validation.get("passed_checks") == 20
        and archive_calibration_validation.get("total_checks") == 20
    )
    density_recovery_plan = load_json(DENSITY_RECOVERY_PLAN)
    density_execution_topology = load_json(
        DENSITY_EXECUTION_TOPOLOGY
    )
    density_execution_validation = load_json(
        DENSITY_EXECUTION_VALIDATION
    )
    legacy_density_input_audit = load_json(
        LEGACY_DENSITY_INPUT_AUDIT
    )
    legacy_density_pretract_plan = load_json(
        LEGACY_DENSITY_PRETRACT_PLAN
    )
    legacy_density_tract_plan = load_json(
        LEGACY_DENSITY_TRACT_PLAN
    )
    legacy_density_validation = load_json(
        LEGACY_DENSITY_VALIDATION
    )
    archive_low_pretract_plan = load_json(
        ARCHIVE_LOW_PRETRACT_PLAN
    )
    archive_low_tract_plan = load_json(ARCHIVE_LOW_TRACT_PLAN)
    archive_low_validation = load_json(ARCHIVE_LOW_VALIDATION)
    archive_D0_pretract_plan = load_json(ARCHIVE_D0_PRETRACT_PLAN)
    archive_D0_tract_plan = load_json(ARCHIVE_D0_TRACT_PLAN)
    archive_D0_validation = load_json(ARCHIVE_D0_VALIDATION)
    archive_86_merge_plan = load_json(ARCHIVE_86_MERGE_PLAN)
    archive_86_merge_output = load_json(ARCHIVE_86_MERGE_OUTPUT)
    archive_86_merge_validation = load_json(
        ARCHIVE_86_MERGE_VALIDATION
    )
    freesurfer_pretract_plan = load_json(FREESURFER_PRETRACT_PLAN)
    freesurfer_tract_plan = load_json(FREESURFER_TRACT_PLAN)
    freesurfer_corrected_validation = load_json(
        FREESURFER_CORRECTED_VALIDATION
    )
    scaleup_pretract_state = load_json(SCALEUP_PRETRACT_SUMMARY)
    scaleup_pretract_plan = load_json(SCALEUP_PRETRACT_PLAN)
    scaleup_pretract_validation = load_json(SCALEUP_PRETRACT_VALIDATION)
    scaleup_tract_state = load_json(SCALEUP_TRACT_SUMMARY)
    scaleup_tract_plan = load_json(SCALEUP_TRACT_PLAN)
    scaleup_tract_validation = load_json(SCALEUP_TRACT_VALIDATION)
    corrected_227_state = load_json(CORRECTED_227_MANIFEST)
    final_release_state = load_json(FINAL_RELEASE_READINESS)
    final_release_validation = load_json(FINAL_RELEASE_VALIDATION)
    integration_verify_preflight = load_json(
        INTEGRATION_VERIFY_PREFLIGHT
    )
    integration_verify_validation = load_json(
        INTEGRATION_VERIFY_VALIDATION
    )
    integration_verify_output = load_json(INTEGRATION_VERIFY_OUTPUT)
    analysis_handoff_preflight = load_json(
        ANALYSIS_HANDOFF_PREFLIGHT
    )
    analysis_handoff_validation = load_json(
        ANALYSIS_HANDOFF_VALIDATION
    )
    analysis_handoff_receipt = load_json(
        ANALYSIS_HANDOFF_RECEIPT
    )
    storage_capacity_state = load_json(STORAGE_CAPACITY_FORECAST)
    parallel_pretract_plan = load_json(PARALLEL_PRETRACT_PLAN)
    parallel_pretract_validation = load_json(
        PARALLEL_PRETRACT_VALIDATION
    )
    parallel_tract_release_plan = load_json(
        PARALLEL_TRACT_RELEASE_PLAN
    )
    parallel_tract_release_state = load_json(
        PARALLEL_TRACT_RELEASE_STATE
    )
    parallel_tract_release_validation = load_json(
        PARALLEL_TRACT_RELEASE_VALIDATION
    )
    parallel_states = sorted(
        PARALLEL_PRETRACT_ATTEMPTS.glob(
            "*/supervisor_state.json"
        ),
        key=lambda path: path.stat().st_mtime_ns,
    )
    parallel_pretract_state = (
        load_json(parallel_states[-1]) if parallel_states else None
    )
    phase_b_live = phase_b_live_status()
    if parallel_pretract_state:
        parallel_pretract_state = dict(parallel_pretract_state)
        observed_pretract = dict(
            parallel_pretract_state.get("pretract", {})
        )
        observed_active = dict(observed_pretract.get("active", {}))
        live_active = {
            name: row
            for name, row in observed_active.items()
            if isinstance(row, dict) and pid_alive(row.get("pid"))
        }
        stale_active = sorted(set(observed_active) - set(live_active))
        observed_pretract["active"] = live_active
        if stale_active:
            observed_pretract["stale_state_active_processes"] = (
                stale_active
            )
        parallel_pretract_state["pretract"] = observed_pretract
        legacy_phase_b = parallel_pretract_state.get("phase_b", {})
        parallel_pretract_state["legacy_supervisor_phase_b"] = (
            legacy_phase_b
        )
        phase_live = phase_b_live["process_alive"]
        parallel_pretract_state["phase_b"] = phase_b_live
        parallel_pretract_state["phase_b_process_alive"] = phase_live
        if phase_live:
            parallel_pretract_state["status"] = (
                "PHASE_B_RUNNING"
            )
        elif stale_active:
            parallel_pretract_state["status"] = (
                "INTERRUPTED_STALE_SUPERVISOR_STATE"
            )
    scaleup_live_counts: Counter[str] = Counter()
    scaleup_live_state_root = (
        ROOT / "corrected_scaleup_recovery4/qc/subjects"
    )
    if scaleup_live_state_root.is_dir():
        for state_path in scaleup_live_state_root.glob("*.json"):
            state = load_json(state_path)
            if state:
                scaleup_live_counts[
                    str(state.get("status", "UNKNOWN"))
                ] += 1
    pretract_state = load_json(PRETRACT_COMPLETION)
    phase_b_attempts = sorted(
        PHASE_B_ROOT.glob("*.attempt.json"),
        key=lambda path: path.stat().st_mtime_ns,
    )
    phase_b_dryrun_state = (
        load_json(phase_b_attempts[-1]) if phase_b_attempts else None
    )
    recovery3_live = recovery3_progress()
    sessions = tmux_sessions()
    disk = shutil.disk_usage(root)

    payload = {
        "updated_utc": utc_now(),
        "baseline_freeze": {
            "status": "PASS"
            if (root / "manifests/historical_freeze_manifest.csv").is_file()
            else "MISSING",
            "target_n": 530,
        },
        "existing_302_density": (
            {
                "status": existing_302_density.get("status"),
                "unit_count": existing_302_density.get("unit_count"),
                "definition": existing_302_density.get("definition"),
                "all_302": existing_302_density.get("all_302"),
                "lanes": existing_302_density.get("lanes"),
            }
            if existing_302_density
            else {"status": "NOT_AUDITED", "unit_count": 0}
        ),
        "live_cohort_inventory": (
            {
                "status": live_cohort_inventory.get("status"),
                "subject_count": live_cohort_inventory.get(
                    "subject_count"
                ),
                "matrix_family_count": live_cohort_inventory.get(
                    "matrix_family_count"
                ),
                "expected_final_matrix_files": (
                    live_cohort_inventory.get(
                        "expected_final_matrix_files"
                    )
                ),
                "final_release_subjects_verified": (
                    live_cohort_inventory.get(
                        "final_release_subjects_verified"
                    )
                ),
                "final_release_matrix_files_verified": (
                    live_cohort_inventory.get(
                        "final_release_matrix_files_verified"
                    )
                ),
                "historical_current_state": (
                    live_cohort_inventory.get(
                        "historical_current_state"
                    )
                ),
                "unbound_current_candidates": (
                    live_cohort_inventory.get(
                        "unbound_current_candidates"
                    )
                ),
                "completion_gap": live_cohort_inventory.get(
                    "completion_gap"
                ),
            }
            if live_cohort_inventory
            else {
                "status": "NOT_AUDITED",
                "subject_count": 530,
                "final_release_subjects_verified": 0,
                "final_release_matrix_files_verified": 0,
            }
        ),
        "density_recovery_530": (
            {
                "status": density_execution_topology.get("status"),
                "target_n": 530,
                "minimum_density_inclusive": 0.60,
                "current_historical_numeric_candidates_n": (
                    density_recovery_plan.get("current_inventory", {}).get(
                        "existing_at_or_above_0_60"
                    )
                ),
                "current_below_threshold_n": (
                    density_recovery_plan.get("current_inventory", {}).get(
                        "existing_below_0_60"
                    )
                ),
                "current_ungenerated_n": (
                    density_recovery_plan.get("current_inventory", {}).get(
                        "not_yet_generated"
                    )
                ),
                "requires_generation_or_recovery_n": (
                    density_recovery_plan.get("current_inventory", {}).get(
                        "remaining_to_produce_or_recover_at_or_above_0_60"
                    )
                ),
                "density_groups": {
                    name: {
                        "n": item.get("n"),
                        "range": item.get("range"),
                        "action": item.get("action"),
                    }
                    for name, item in density_recovery_plan.get(
                        "density_groups", {}
                    ).items()
                },
                "execution_lanes": (
                    density_execution_topology.get("lanes")
                ),
                "release_scenarios": (
                    density_execution_topology.get("release_scenarios")
                ),
                "whole_cohort_final_accepted_n": (
                    530
                    if final_release_state
                    and final_release_state.get("status") == "PASS"
                    else 0
                ),
                "topology_validation": (
                    {
                        "status": density_execution_validation.get(
                            "status"
                        ),
                        "passed": density_execution_validation.get(
                            "passed_checks"
                        ),
                        "total": density_execution_validation.get(
                            "total_checks"
                        ),
                    }
                    if density_execution_validation
                    else {"status": "NOT_RUN"}
                ),
                "lane_packages": {
                    "corrected_core_227": {
                        "status": (
                            "PREPARED_GATED"
                            if scaleup_pretract_validation
                            and scaleup_pretract_validation.get("status")
                            == "PASS"
                            and scaleup_tract_validation
                            and scaleup_tract_validation.get("status")
                            == "PASS"
                            else "HARDENING"
                        ),
                        "release_n": 227,
                        "new_execution_n": 216,
                    },
                    "corrected_legacy_tensor_216": {
                        "status": (
                            "PREPARED_GATED"
                            if legacy_density_validation
                            and legacy_density_validation.get("status")
                            == "PASS"
                            else "HARDENING"
                        ),
                        "input_audit_status": (
                            legacy_density_input_audit.get("status")
                            if legacy_density_input_audit
                            else "NOT_RUN"
                        ),
                        "release_n": 216,
                        "new_execution_n": 212,
                        "pretract_plan_status": (
                            legacy_density_pretract_plan.get("status")
                            if legacy_density_pretract_plan
                            else "NOT_RUN"
                        ),
                        "tractography_plan_status": (
                            legacy_density_tract_plan.get("status")
                            if legacy_density_tract_plan
                            else "NOT_RUN"
                        ),
                    },
                    "corrected_archive_low_30": {
                        "status": (
                            "PREPARED_GATED"
                            if archive_low_validation
                            and archive_low_validation.get("status")
                            == "PASS"
                            else "HARDENING"
                        ),
                        "release_n": 30,
                        "new_execution_n": 28,
                        "calibration_primary_reuse_n": 2,
                        "pretract_plan_status": (
                            archive_low_pretract_plan.get("status")
                            if archive_low_pretract_plan
                            else "NOT_RUN"
                        ),
                        "tractography_plan_status": (
                            archive_low_tract_plan.get("status")
                            if archive_low_tract_plan
                            else "NOT_RUN"
                        ),
                    },
                    "conditional_archive_D0_56": {
                        "status": (
                            archive_concordance_state.get("decision")
                            if archive_concordance_state
                            else (
                                "CONTINGENCY_PREPARED_GATED"
                                if archive_D0_validation
                                and archive_D0_validation.get("status")
                                == "PASS"
                                else "AWAITING_PHASE_B_AND_CALIBRATION"
                            )
                        ),
                        "n": 56,
                        "new_execution_n": 52,
                        "calibration_primary_reuse_n": 4,
                        "pretract_may_run_before_route_decision": True,
                        "pretract_plan_status": (
                            archive_D0_pretract_plan.get("status")
                            if archive_D0_pretract_plan
                            else "NOT_RUN"
                        ),
                        "tractography_plan_status": (
                            archive_D0_tract_plan.get("status")
                            if archive_D0_tract_plan
                            else "NOT_RUN"
                        ),
                        "package_validation": (
                            {
                                "status": archive_D0_validation.get(
                                    "status"
                                ),
                                "passed": archive_D0_validation.get(
                                    "passed_checks"
                                ),
                                "total": archive_D0_validation.get(
                                    "total_checks"
                                ),
                                "imaging_executed": (
                                    archive_D0_validation.get(
                                        "imaging_executed"
                                    )
                                ),
                            }
                            if archive_D0_validation
                            else {"status": "NOT_RUN"}
                        ),
                    },
                    "corrected_freesurfer_1": {
                        "status": (
                            "PREPARED_GATED"
                            if freesurfer_corrected_validation
                            and freesurfer_corrected_validation.get(
                                "status"
                            )
                            == "PASS"
                            else "HARDENING"
                        ),
                        "n": 1,
                        "source_status": (
                            recovered_hcp_state.get("status")
                            if recovered_hcp_state
                            else "NOT_RUN"
                        ),
                        "pretract_plan_status": (
                            freesurfer_pretract_plan.get("status")
                            if freesurfer_pretract_plan
                            else "NOT_RUN"
                        ),
                        "tractography_plan_status": (
                            freesurfer_tract_plan.get("status")
                            if freesurfer_tract_plan
                            else "NOT_RUN"
                        ),
                    },
                    "corrected_archive_86_derived_candidate": {
                        "status": (
                            archive_86_merge_output.get("status")
                            if archive_86_merge_output
                            else (
                                archive_86_merge_plan.get("status")
                                if archive_86_merge_plan
                                else "NOT_PREPARED"
                            )
                        ),
                        "n": 86,
                        "route_selected": False,
                        "archive_concordance_still_required": True,
                        "component_counts": (
                            archive_86_merge_plan.get(
                                "component_counts"
                            )
                            if archive_86_merge_plan
                            else {
                                "corrected_archive_low": 30,
                                "corrected_archive_D0": 56,
                            }
                        ),
                        "package_validation": (
                            {
                                "status": (
                                    archive_86_merge_validation.get(
                                        "status"
                                    )
                                ),
                                "passed": (
                                    archive_86_merge_validation.get(
                                        "passed_checks"
                                    )
                                ),
                                "total": (
                                    archive_86_merge_validation.get(
                                        "total_checks"
                                    )
                                ),
                                "imaging_executed": (
                                    archive_86_merge_validation.get(
                                        "imaging_executed"
                                    )
                                ),
                            }
                            if archive_86_merge_validation
                            else {"status": "NOT_RUN"}
                        ),
                    },
                },
            }
            if density_recovery_plan and density_execution_topology
            else {"status": "NOT_PREPARED"}
        ),
        "archive_recovery": {
            "target_n": 86,
            "states_present": len(archive_states),
            "status_counts": dict(sorted(archive_status.items())),
            "weighted_status_counts": dict(sorted(weighted_status.items())),
            "pass_all_nine": archive_status.get("PASS_ALL_NINE", 0),
            "remaining_n": 86 - archive_status.get("PASS_ALL_NINE", 0),
            "progress_fraction": archive_status.get("PASS_ALL_NINE", 0) / 86,
            "active_n": weighted_status.get(
                "RUNNING_MATCHED_REBUILD", 0
            ),
        },
        "keep_route_calibration": (
            {
                "status": keep_concordance_state.get("status"),
                "decision": keep_concordance_state.get("decision"),
                "calibration_n": keep_concordance_state.get(
                    "calibration_unit_count"
                ),
                "selected_streamline_count": (
                    keep_concordance_state.get(
                        "selected_streamline_count"
                    )
                ),
            }
            if keep_concordance_state
            else {
                "status": (
                    "RECOVERY4_DECISION_PREPARED_GATED"
                    if keep_route_package_ready
                    else "RECOVERY4_DECISION_HARDENING"
                ),
                "runner_ready": keep_route_package_ready,
                "plan_status": (
                    keep_route_plan.get("status")
                    if keep_route_plan
                    else "NOT_RUN"
                ),
                "keep_n": (
                    keep_route_plan.get("keep_subject_count")
                    if keep_route_plan
                    else 214
                ),
                "calibration_n": (
                    keep_route_plan.get("calibration_unit_count")
                    if keep_route_plan
                    else 0
                ),
                "calibration_units": (
                    keep_route_plan.get("calibration_units")
                    if keep_route_plan
                    else []
                ),
                "retention_requires_route_and_seed_pass": (
                    keep_route_plan.get(
                        "retention_requires_route_and_seed_pass"
                    )
                    if keep_route_plan
                    else None
                ),
                "package_validation": (
                    {
                        "status": keep_route_validation.get("status"),
                        "passed": keep_route_validation.get(
                            "passed_checks"
                        ),
                        "total": keep_route_validation.get(
                            "total_checks"
                        ),
                        "imaging_executed": (
                            keep_route_validation.get(
                                "imaging_executed_by_validation"
                            )
                        ),
                    }
                    if keep_route_validation
                    else {"status": "NOT_RUN"}
                ),
            }
        ),
        "archive_route_calibration": (
            {
                "status": archive_concordance_state.get("status"),
                "decision": archive_concordance_state.get("decision"),
                "calibration_n": archive_concordance_state.get(
                    "calibration_n"
                ),
            }
            if archive_concordance_state
            else {
                "status": (
                    "RECOVERY4_CALIBRATION_PREPARED_GATED"
                    if archive_corrected_input_state
                    and archive_corrected_input_state.get("status")
                    == "PASS"
                    and archive_calibration_package_ready
                    else (
                        "RECOVERY4_CALIBRATION_HARDENING"
                        if archive_corrected_input_state
                        and archive_corrected_input_state.get("status")
                        == "PASS"
                        else "PENDING_INPUT_AUDIT"
                    )
                ),
                "input_audit_status": (
                    archive_corrected_input_state.get("status")
                    if archive_corrected_input_state
                    else None
                ),
                "audited_n": (
                    archive_corrected_input_state.get("audited_n")
                    if archive_corrected_input_state
                    else 0
                ),
                "ready_n": (
                    archive_corrected_input_state.get("ready_n")
                    if archive_corrected_input_state
                    else 0
                ),
                "route_counts": (
                    archive_corrected_input_state.get("route_counts")
                    if archive_corrected_input_state
                    else {}
                ),
                "calibration_n": (
                    archive_corrected_input_state.get("calibration_n")
                    if archive_corrected_input_state
                    else 0
                ),
                "calibration_units": (
                    archive_corrected_input_state.get(
                        "calibration_units"
                    )
                    if archive_corrected_input_state
                    else []
                ),
                "archive_route_allowed_in_uniform_primary_release": False,
                "runners_ready": {
                    name: (
                        path.is_file()
                        and (
                            name == "concordance_validator"
                            or (
                                archive_calibration_plans.get(name)
                                is not None
                                and archive_calibration_plans[name].get(
                                    "status"
                                )
                                == archive_plan_statuses[name]
                            )
                        )
                    )
                    for name, path in ARCHIVE_CALIBRATION_RUNNERS.items()
                },
                "package_validation": (
                    {
                        "status": archive_calibration_validation.get(
                            "status"
                        ),
                        "passed": archive_calibration_validation.get(
                            "passed_checks"
                        ),
                        "total": archive_calibration_validation.get(
                            "total_checks"
                        ),
                        "imaging_executed": (
                            archive_calibration_validation.get(
                                "imaging_executed_by_validation"
                            )
                        ),
                    }
                    if archive_calibration_validation
                    else {"status": "NOT_RUN"}
                ),
                "plan_statuses": {
                    name: (
                        plan.get("status") if plan else "NOT_RUN"
                    )
                    for name, plan in archive_calibration_plans.items()
                },
                "legacy_v2_runners_present_but_not_launchable": True,
            }
        ),
        "remap_panel": {
            "status": "COMPLETE_ESCALATE_111"
            if (root / "manifests/remap_probe_summary.json").is_file()
            else "PENDING",
            "corrected_track_queue_n": 227,
        },
        "isolated_repairs": {
            "tensor": tensor,
            "eddy_016": (
                {
                    "status": eddy_state.get("status"),
                    "error": eddy_state.get("error"),
                }
                if eddy_state
                else {"status": "PENDING"}
            ),
            "freesurfer_114": (
                {
                    "status": freesurfer_state.get("status"),
                    "error": freesurfer_state.get("error"),
                }
                if freesurfer_state
                else {"status": "PENDING"}
            ),
            "hcp_build_114": (
                {
                    "status": recovered_hcp_state.get("status"),
                    "atlas_nodes_present": recovered_hcp_state.get(
                        "atlas_nodes_present"
                    ),
                    "old_existing_track_status": (
                        recovered_hcp_state.get(
                            "old_existing_track_status"
                        )
                    ),
                    "corrected_package_validation": (
                        {
                            "status": (
                                freesurfer_corrected_validation.get(
                                    "status"
                                )
                            ),
                            "passed": (
                                freesurfer_corrected_validation.get(
                                    "passed_checks"
                                )
                            ),
                            "total": (
                                freesurfer_corrected_validation.get(
                                    "total_checks"
                                )
                            ),
                        }
                        if freesurfer_corrected_validation
                        else {"status": "NOT_RUN"}
                    ),
                }
                if recovered_hcp_state
                else {"status": "PENDING"}
            ),
        },
        "corrected_pretract_canary": {
            "status": (
                pretract_state.get("status")
                if pretract_state
                else "PENDING"
            ),
            "target_n": 15,
            "passed_n": (
                pretract_state.get("unit_count") if pretract_state else 0
            ),
            "recovery3_diagnostic_execution": recovery3_live,
            "recovery4_scalar_qc": (
                {
                    "status": scalar_recovery4_state.get("status"),
                    "passed_n": scalar_recovery4_state.get("passed_n"),
                }
                if scalar_recovery4_state
                else {"status": "PENDING", "passed_n": 0}
            ),
            "recovery4_spatial_qc": (
                {
                    "status": spatial_recovery4_state.get("status"),
                    "passed_n": spatial_recovery4_state.get("passed_n"),
                }
                if spatial_recovery4_state
                else {"status": "PENDING", "passed_n": 0}
            ),
        },
        "corrected_hcp379_atlas_canary": (
            {
                "status": corrected_atlas_state.get("status"),
                "target_n": 15,
                "passed_n": corrected_atlas_state.get("passed_unit_count"),
            }
            if corrected_atlas_state
            else {
                "status": "PENDING",
                "target_n": 15,
                "passed_n": 0,
            }
        ),
        "blinded_visual_review": (
            {
                "status": (
                    "PASS_PROMOTED"
                    if v6_promotion_ready
                    else (
                        candidate_visual_review_state.get("status")
                        if candidate_visual_review_state
                        else visual_review_state.get("status")
                    )
                ),
                "unit_count": (
                    candidate_visual_review_state.get("unit_count")
                    if candidate_visual_review_state
                    else visual_review_state.get("unit_count")
                ),
                "human_visual_qc_inferred": (
                    candidate_visual_review_state.get(
                        "human_visual_qc_inferred"
                    )
                    if candidate_visual_review_state
                    else visual_review_state.get(
                        "human_visual_qc_inferred"
                    )
                ),
                "corrected_edge_overlay_pack_ready": (
                    visual_review_state.get("status")
                    == "READY_FOR_HUMAN_REVIEW"
                ),
                "candidate_v6_exact_input_pack_ready": (
                    candidate_visual_review_state is not None
                    and candidate_visual_review_state.get("status")
                    == "READY_FOR_EXACT_INPUT_HUMAN_REVIEW"
                ),
                "canonical_human_qc_ready": (
                    v6_promotion_ready
                ),
                "corrected_hcp379_panels_pending_n": (
                    0 if v6_promotion_ready else 15
                ),
                "submitted_round1_panel_counts": (
                    {
                        "B0_vs_T1": round1_transcription.get(
                            "b0_vs_t1_counts"
                        ),
                        "B0_vs_5TT": round1_transcription.get(
                            "b0_vs_5tt_counts"
                        ),
                        "superseded_HCP379_scalar_overlay": (
                            round1_transcription.get(
                                "prior_hcp379_scalar_overlay_counts"
                            )
                        ),
                        "status": round1_transcription.get("status"),
                    }
                    if round1_transcription
                    else {"status": "NOT_TRANSCRIBED"}
                ),
                "exact_v6_completed_review_present": (
                    V6_COMPLETED_REVIEW.is_file()
                ),
                "numbers_review_workflow": {
                    "status": (
                        "TRANSCRIBED"
                        if v6_numbers_transcription
                        else (
                            "COMPLETED_NUMBERS_PRESENT_AWAITING_TRANSCRIPTION"
                            if V6_COMPLETED_NUMBERS.is_file()
                            else "READY_FOR_HUMAN_REVIEW"
                        )
                    ),
                    "template_ready": (
                        V6_NUMBERS_TEMPLATE.is_file()
                        and v6_numbers_manifest is not None
                        and v6_numbers_manifest.get("status")
                        == "READY_FOR_HUMAN_REVIEW"
                    ),
                    "workflow_validation": (
                        {
                            "status": v6_numbers_validation.get("status"),
                            "passed": v6_numbers_validation.get(
                                "passed_checks"
                            ),
                            "total": v6_numbers_validation.get(
                                "total_checks"
                            ),
                        }
                        if v6_numbers_validation
                        else {"status": "NOT_VALIDATED"}
                    ),
                    "template": str(V6_NUMBERS_TEMPLATE),
                    "completed_numbers_expected": str(
                        V6_COMPLETED_NUMBERS
                    ),
                    "completed_numbers_present": (
                        V6_COMPLETED_NUMBERS.is_file()
                    ),
                    "transcriber_ready": (
                        V6_NUMBERS_TRANSCRIBER.is_file()
                    ),
                    "transcription_receipt_present": (
                        V6_NUMBERS_TRANSCRIPTION.is_file()
                    ),
                    "imaging_executed": False,
                },
                "promotion_gate": {
                    "status": (
                        "PASS_PROMOTED"
                        if v6_promotion_ready
                        else "AWAITING_COMPLETED_V6_HUMAN_REVIEW"
                    ),
                    "builder_ready": PROMOTION_BUILDER.is_file(),
                    "completed_review_expected": str(
                        V6_COMPLETED_REVIEW
                    ),
                    "promotion_receipt_present": (
                        PROMOTION_RECEIPT.is_file()
                    ),
                    "promoted_pretract_master_present": (
                        PROMOTED_PRETRACT_MASTER.is_file()
                    ),
                    "tractography_started_by_promotion": (
                        promotion_receipt_state.get(
                            "tractography_started"
                        )
                        if promotion_receipt_state
                        else False
                    ),
                },
                "unchanged_t1_5tt_pass_carried_n": (
                    candidate_visual_review_state.get(
                        "carried_human_evidence", {}
                    ).get("unchanged_units")
                    if candidate_visual_review_state
                    else None
                ),
                "original_quantitative_hard_holds": (
                    corrected_overlay_quantitative_state.get(
                        "hard_hold_units"
                    )
                    if corrected_overlay_quantitative_state
                    else None
                ),
                "targeted_009_recovery": (
                    {
                        "status": (
                            "PASS_PROMOTED"
                            if v6_promotion_ready
                            else targeted_009_registration_state.get(
                                "status"
                            )
                        ),
                        "recommended_candidate": (
                            targeted_009_registration_state.get(
                                "selection", {}
                            ).get("recommended_candidate")
                        ),
                        "promotion_status": (
                            (
                                promoted_pretract_state.get(
                                    "route_substitution", {}
                                ).get("promotion_status")
                                if promoted_pretract_state
                                else targeted_009_registration_state.get(
                                    "selection", {}
                                ).get("promotion_status")
                            )
                        ),
                        "candidate_pretract_status": (
                            candidate_pretract_state.get("status")
                            if candidate_pretract_state
                            else "NOT_PREPARED"
                        ),
                        "candidate_pretract_validation": (
                            {
                                "status": (
                                    candidate_pretract_validation.get(
                                        "status"
                                    )
                                ),
                                "passed": (
                                    candidate_pretract_validation.get(
                                        "passed_checks"
                                    )
                                ),
                                "total": (
                                    candidate_pretract_validation.get(
                                        "total_checks"
                                    )
                                ),
                            }
                            if candidate_pretract_validation
                            else {"status": "NOT_RUN"}
                        ),
                    }
                    if targeted_009_registration_state
                    else {"status": "NOT_RUN"}
                ),
            }
            if visual_review_state
            else {
                "status": "PENDING",
                "unit_count": 15,
                "human_visual_qc_inferred": False,
            }
        ),
        "hcp379_recipe_convergence": (
            {
                "status": stability_state.get("status"),
                "smallest_stable_streamline_count": stability_state.get(
                    "smallest_cohort_stable_streamline_count"
                ),
                "smallest_density_qualified_scale_up_streamline_count": (
                    stability_state.get(
                        "smallest_density_qualified_scale_up_streamline_count"
                    )
                ),
            }
            if stability_state
            else {
                "status": (
                    "PHASE_B_RUNNING"
                    if parallel_pretract_state
                    and parallel_pretract_state.get("phase_b", {}).get(
                        "live_dag_started"
                    )
                    is True
                    and parallel_pretract_state.get("phase_b", {}).get(
                        "returncode"
                    )
                    is None
                    else "AWAITING_PHASE_B_RESULT"
                    if v6_promotion_ready
                    else "AWAITING_GENUINE_HUMAN_QC"
                ),
                "pre_human_dry_run": (
                    {
                        "status": phase_b_dryrun_state.get("status"),
                        "scheduled_rules": len(
                            phase_b_dryrun_state.get(
                                "scheduled_rules", []
                            )
                        ),
                        "forbidden_scheduled_rules": (
                            phase_b_dryrun_state.get(
                                "forbidden_scheduled_rules"
                            )
                        ),
                    }
                    if phase_b_dryrun_state
                    else {"status": "NOT_RUN"}
                ),
                "streamline_levels": [3_000_000, 5_000_000, 10_000_000],
                "independent_seed_required": True,
                "density_0_60_scale_up_gate": True,
            }
        ),
        "parallel_phase_b_pretract": (
            {
                "status": parallel_pretract_state.get("status"),
                "attempt": str(parallel_states[-1].parent),
                "phase_b": parallel_pretract_state.get("phase_b"),
                "pretract": parallel_pretract_state.get("pretract"),
                "phase_b_process_alive": (
                    parallel_pretract_state.get(
                        "phase_b_process_alive"
                    )
                ),
                "updated_utc": parallel_pretract_state.get(
                    "updated_utc"
                ),
            }
            if parallel_pretract_state
            else {
                "status": (
                    parallel_pretract_plan.get("status")
                    if parallel_pretract_plan
                    else "NOT_PREPARED"
                ),
                "imaging_executed": False,
                "maximum_active_pretract_subjects": (
                    parallel_pretract_plan.get(
                        "resource_contract", {}
                    ).get("maximum_active_pretract_subjects")
                    if parallel_pretract_plan
                    else None
                ),
                "phase_b_cores": (
                    parallel_pretract_plan.get(
                        "resource_contract", {}
                    ).get("phase_b_cores")
                    if parallel_pretract_plan
                    else None
                ),
                "pretract_lane_n": (
                    len(
                        parallel_pretract_plan.get(
                            "pretract_lanes", []
                        )
                    )
                    if parallel_pretract_plan
                    else 0
                ),
                "pretract_target_n": (
                    sum(
                        int(row.get("target_n", 0))
                        for row in parallel_pretract_plan.get(
                            "pretract_lanes", []
                        )
                    )
                    if parallel_pretract_plan
                    else 0
                ),
                "package_validation": (
                    {
                        "status": (
                            parallel_pretract_validation.get(
                                "status"
                            )
                        ),
                        "passed": (
                            parallel_pretract_validation.get(
                                "passed_checks"
                            )
                        ),
                        "total": (
                            parallel_pretract_validation.get(
                                "total_checks"
                            )
                        ),
                        "imaging_executed": (
                            parallel_pretract_validation.get(
                                "imaging_executed"
                            )
                        ),
                    }
                    if parallel_pretract_validation
                    else {"status": "NOT_RUN"}
                ),
            }
        ),
        "phase_b_live": phase_b_live,
        "parallel_tractography_release": (
            {
                "status": parallel_tract_release_state.get("status"),
                "attempt": parallel_tract_release_state.get(
                    "attempt_root"
                ),
                "phase_b": parallel_tract_release_state.get("phase_b"),
                "upstream_gates": parallel_tract_release_state.get(
                    "upstream_gates"
                ),
                "tasks": parallel_tract_release_state.get("tasks"),
                "cross_supervisor_compute_limit": (
                    parallel_tract_release_state.get(
                        "cross_supervisor_compute_limit"
                    )
                ),
                "updated_utc": parallel_tract_release_state.get(
                    "updated_utc"
                ),
            }
            if parallel_tract_release_state
            else {
                "status": (
                    parallel_tract_release_plan.get("status")
                    if parallel_tract_release_plan
                    else "NOT_PREPARED"
                ),
                "imaging_executed": False,
                "task_n": (
                    len(
                        parallel_tract_release_plan.get("tasks", {})
                    )
                    if parallel_tract_release_plan
                    else 0
                ),
                "selected_streamline_count": (
                    parallel_tract_release_plan.get(
                        "selected_streamline_count"
                    )
                    if parallel_tract_release_plan
                    else None
                ),
                "maximum_active_compute_tasks": (
                    parallel_tract_release_plan.get(
                        "resources", {}
                    ).get("maximum_active_compute_tasks")
                    if parallel_tract_release_plan
                    else None
                ),
                "declared_peak_tractography_threads": (
                    parallel_tract_release_plan.get(
                        "resources", {}
                    ).get("declared_peak_tractography_threads")
                    if parallel_tract_release_plan
                    else None
                ),
                "cross_supervisor_cpu_guard": (
                    parallel_tract_release_plan.get(
                        "resources", {}
                    ).get("cross_supervisor_cpu_guard")
                    if parallel_tract_release_plan
                    else None
                ),
                "package_validation": (
                    {
                        "status": (
                            parallel_tract_release_validation.get(
                                "status"
                            )
                        ),
                        "passed": (
                            parallel_tract_release_validation.get(
                                "passed_checks"
                            )
                        ),
                        "total": (
                            parallel_tract_release_validation.get(
                                "total_checks"
                            )
                        ),
                        "imaging_executed": (
                            parallel_tract_release_validation.get(
                                "imaging_executed"
                            )
                        ),
                    }
                    if parallel_tract_release_validation
                    else {"status": "NOT_RUN"}
                ),
            }
        ),
        "corrected_tractography": {
            "status": (
                corrected_227_state.get("status")
                if corrected_227_state
                else scaleup_tract_state.get("status")
                if scaleup_tract_state
                else scaleup_pretract_state.get("status")
                if scaleup_pretract_state
                else "GATED_ON_CANARY_VISUAL_QC_AND_CONVERGENCE"
            ),
            "target_n": 227,
            "noncanary_pretract": (
                {
                    "status": (
                        "RECOVERY_CANARY_RUNNING"
                        if scaleup_live_counts.get("RUNNING", 0)
                        + scaleup_live_counts.get(
                            "RUNNING_RECOVERY4", 0
                        )
                        > 0
                        else scaleup_pretract_state.get("status")
                    ),
                    "target_n": scaleup_pretract_state.get("target_n"),
                    "states_present_n": max(
                        int(
                            scaleup_pretract_state.get(
                                "states_present_n", 0
                            )
                            or 0
                        ),
                        sum(scaleup_live_counts.values()),
                    ),
                    "status_counts": (
                        dict(sorted(scaleup_live_counts.items()))
                        if scaleup_live_counts
                        else scaleup_pretract_state.get(
                            "status_counts"
                        )
                    ),
                }
                if scaleup_pretract_state
                else {
                    "status": (
                        "RECOVERY_CANARY_RUNNING"
                        if scaleup_live_counts.get("RUNNING", 0)
                        + scaleup_live_counts.get(
                            "RUNNING_RECOVERY4", 0
                        )
                        > 0
                        else "RUNNING"
                        if parallel_pretract_state
                        and "corrected_core_216_new"
                        in parallel_pretract_state.get(
                            "pretract", {}
                        ).get("active", {})
                        else "PREPARED_GATED"
                        if RECOVERY4_SCALEUP_PRETRACT_RUNNER.is_file()
                        and scaleup_pretract_plan
                        and scaleup_pretract_plan.get("status")
                        in {
                            "PRE_HUMAN_DRY_RUN_PASS",
                            "EXECUTION_GATE_PASS",
                        }
                        and scaleup_pretract_validation
                        and scaleup_pretract_validation.get("status")
                        == "PASS"
                        else "RECOVERY4_RUNNER_HARDENING"
                    ),
                    "target_n": 216,
                    "states_present_n": sum(
                        scaleup_live_counts.values()
                    ),
                    "status_counts": dict(
                        sorted(scaleup_live_counts.items())
                    ),
                    "runner_ready": (
                        RECOVERY4_SCALEUP_PRETRACT_RUNNER.is_file()
                        and scaleup_pretract_plan is not None
                        and scaleup_pretract_plan.get("status")
                        in {
                            "PRE_HUMAN_DRY_RUN_PASS",
                            "EXECUTION_GATE_PASS",
                        }
                        and scaleup_pretract_validation is not None
                        and scaleup_pretract_validation.get("status")
                        == "PASS"
                    ),
                    "pre_human_dry_run_status": (
                        scaleup_pretract_plan.get("status")
                        if scaleup_pretract_plan
                        else "NOT_RUN"
                    ),
                    "validation_status": (
                        scaleup_pretract_validation.get("status")
                        if scaleup_pretract_validation
                        else "NOT_RUN"
                    ),
                    "validation_checks": (
                        {
                            "passed": scaleup_pretract_validation.get(
                                "passed_checks"
                            ),
                            "total": scaleup_pretract_validation.get(
                                "total_checks"
                            ),
                        }
                        if scaleup_pretract_validation
                        else None
                    ),
                    "legacy_v2_present_but_not_launchable": (
                        LEGACY_SCALEUP_PRETRACT_RUNNER.is_file()
                    ),
                }
            ),
            "noncanary_tracks_and_matrices": (
                {
                    "status": scaleup_tract_state.get("status"),
                    "target_n": scaleup_tract_state.get("target_n"),
                    "states_present_n": scaleup_tract_state.get(
                        "states_present_n"
                    ),
                    "status_counts": scaleup_tract_state.get(
                        "status_counts"
                    ),
                }
                if scaleup_tract_state
                else {
                    "status": (
                        "PREPARED_GATED_ON_PHASE_B"
                        if RECOVERY4_SCALEUP_TRACT_RUNNER.is_file()
                        and scaleup_tract_plan
                        and scaleup_tract_plan.get("status")
                        == "PRE_PHASE_DRY_RUN_PASS"
                        and scaleup_tract_validation
                        and scaleup_tract_validation.get("status")
                        == "PASS"
                        else "RECOVERY4_RUNNER_HARDENING"
                    ),
                    "target_n": 216,
                    "runner_ready": (
                        RECOVERY4_SCALEUP_TRACT_RUNNER.is_file()
                        and scaleup_tract_plan is not None
                        and scaleup_tract_plan.get("status")
                        == "PRE_PHASE_DRY_RUN_PASS"
                        and scaleup_tract_validation is not None
                        and scaleup_tract_validation.get("status")
                        == "PASS"
                    ),
                    "pre_phase_dry_run_status": (
                        scaleup_tract_plan.get("status")
                        if scaleup_tract_plan
                        else "NOT_RUN"
                    ),
                    "validation_status": (
                        scaleup_tract_validation.get("status")
                        if scaleup_tract_validation
                        else "NOT_RUN"
                    ),
                    "validation_checks": (
                        {
                            "passed": scaleup_tract_validation.get(
                                "passed_checks"
                            ),
                            "total": scaleup_tract_validation.get(
                                "total_checks"
                            ),
                        }
                        if scaleup_tract_validation
                        else None
                    ),
                    "legacy_v2_present_but_not_launchable": (
                        LEGACY_SCALEUP_TRACT_RUNNER.is_file()
                    ),
                }
            ),
            "corrected_227_release": (
                {
                    "status": corrected_227_state.get("status"),
                    "unit_count": corrected_227_state.get("unit_count"),
                    "selected_streamline_count": corrected_227_state.get(
                        "selected_streamline_count"
                    ),
                }
                if corrected_227_state
                else {"status": "PENDING", "unit_count": 0}
            ),
            "input_audit": (
                {
                    "status": scaleup_input_state.get("status"),
                    "audited_n": scaleup_input_state.get("audited_n"),
                    "route_counts": scaleup_input_state.get("route_counts"),
                }
                if scaleup_input_state
                else {"status": "PENDING"}
            ),
        },
        "final_hcp379_release": (
            {
                "status": final_release_state.get("status"),
                "expected_subjects": final_release_state.get(
                    "expected_subjects"
                ),
                "expected_matrix_files": final_release_state.get(
                    "expected_matrix_files"
                ),
                "blockers": final_release_state.get("blockers"),
                "matrix_bundles_present_by_lane": final_release_state.get(
                    "matrix_bundles_present_by_lane"
                ),
                "recovery4_assembler_validation": (
                    {
                        "status": final_release_validation.get("status"),
                        "passed": final_release_validation.get(
                            "passed_checks"
                        ),
                        "total": final_release_validation.get(
                            "total_checks"
                        ),
                    }
                    if final_release_validation
                    else {"status": "NOT_RUN"}
                ),
            }
            if final_release_state
            else {"status": "READINESS_NOT_EVALUATED"}
        ),
        "integration_release_verification": (
            {
                "status": integration_verify_output.get("status"),
                "subject_count": integration_verify_output.get(
                    "subject_count"
                ),
                "matrix_file_count": integration_verify_output.get(
                    "matrix_file_count"
                ),
                "selected_streamline_count": (
                    integration_verify_output.get(
                        "selected_streamline_count"
                    )
                ),
                "minimum_observed_edge_density": (
                    integration_verify_output.get(
                        "minimum_observed_edge_density"
                    )
                ),
                "minimum_observed_weighted_support": (
                    integration_verify_output.get(
                        "minimum_observed_weighted_support"
                    )
                ),
                "minimum_observed_assignment_fraction": (
                    integration_verify_output.get(
                        "minimum_observed_assignment_fraction"
                    )
                ),
            }
            if integration_verify_output
            else {
                "status": (
                    integration_verify_preflight.get("status")
                    if integration_verify_preflight
                    else "NOT_PREPARED"
                ),
                "verification_executed": False,
                "package_validation": (
                    {
                        "status": (
                            integration_verify_validation.get(
                                "status"
                            )
                        ),
                        "passed": (
                            integration_verify_validation.get(
                                "passed_checks"
                            )
                        ),
                        "total": (
                            integration_verify_validation.get(
                                "total_checks"
                            )
                        ),
                        "release_verification_executed": (
                            integration_verify_validation.get(
                                "release_verification_executed"
                            )
                        ),
                    }
                    if integration_verify_validation
                    else {"status": "NOT_RUN"}
                ),
            }
        ),
        "verified_analysis_handoff": (
            {
                "status": analysis_handoff_receipt.get("status"),
                "handoff_id": analysis_handoff_receipt.get(
                    "handoff_id"
                ),
                "subject_count": analysis_handoff_receipt.get(
                    "subject_count"
                ),
                "matrix_file_count": analysis_handoff_receipt.get(
                    "matrix_file_count"
                ),
                "matrix_shape": analysis_handoff_receipt.get(
                    "matrix_shape"
                ),
                "diagnosis_or_outcome_fields_present": (
                    analysis_handoff_receipt.get(
                        "diagnosis_or_outcome_fields_present"
                    )
                ),
                "live_dashboard_modified": (
                    analysis_handoff_receipt.get(
                        "live_dashboard_modified"
                    )
                ),
            }
            if analysis_handoff_receipt
            else {
                "status": (
                    analysis_handoff_preflight.get("status")
                    if analysis_handoff_preflight
                    else "NOT_PREPARED"
                ),
                "handoff_created": False,
                "live_dashboard_modified": False,
                "legacy_dashboard_compatible": (
                    analysis_handoff_preflight.get(
                        "compatibility", {}
                    ).get("legacy_dashboard_compatible")
                    if analysis_handoff_preflight
                    else None
                ),
                "package_validation": (
                    {
                        "status": (
                            analysis_handoff_validation.get("status")
                        ),
                        "passed": (
                            analysis_handoff_validation.get("passed")
                        ),
                        "total": (
                            analysis_handoff_validation.get("total")
                        ),
                        "imaging_executed": (
                            analysis_handoff_validation.get(
                                "imaging_executed"
                            )
                        ),
                    }
                    if analysis_handoff_validation
                    else {"status": "NOT_RUN"}
                ),
            }
        ),
        "tmux_sessions": sessions,
        "storage": {
            "total_bytes": disk.total,
            "used_bytes": disk.used,
            "free_bytes": disk.free,
            "used_fraction": disk.used / disk.total,
            "capacity_forecast": (
                {
                    "status": storage_capacity_state.get("status"),
                    "generated_utc": storage_capacity_state.get(
                        "generated_utc"
                    ),
                    "phase_b_safe": storage_capacity_state.get(
                        "phase_b_only", {}
                    ).get("safe_with_current_volume_and_reserve"),
                    "pretract_worst_case_515_safe": (
                        storage_capacity_state.get(
                        "pretract_worst_case_515_only", {}
                        ).get("safe_with_current_volume_and_reserve")
                    ),
                    "worst_case_new_processing_subject_n": (
                        storage_capacity_state.get(
                            "assumptions", {}
                        ).get("full_corrected_new_processing_n")
                    ),
                    "streaming_safe_streamline_counts": (
                        storage_capacity_state.get(
                            "streaming_safe_streamline_counts"
                        )
                    ),
                    "central_safe_streamline_counts": (
                        storage_capacity_state.get(
                            "central_safe_streamline_counts"
                        )
                    ),
                    "deletes_or_volume_changes_performed": (
                        storage_capacity_state.get(
                            "deletes_or_volume_changes_performed"
                        )
                    ),
                }
                if storage_capacity_state
                else {"status": "NOT_EVALUATED"}
            ),
        },
    }
    if not args.no_write:
        atomic_json(root / "manifests/live_status.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
