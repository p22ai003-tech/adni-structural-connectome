#!/home/ec2-user/fsl/bin/python
"""Prepare or launch Phase B plus storage-bounded Recovery4 pretract.

The default mode writes a non-imaging plan.  ``--launch`` requires a genuine
complete 15-case human-QC CSV, starts Phase B first, waits until its live DAG
has begun, and then supervises six disjoint pretract lanes with at most four
active subject workspaces.  Every pretract lane enables validated streaming
compaction.  This launcher never starts selected-count tractography.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import signal
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


EXP = Path("/home/ec2-user/exp")
HCP_ROOT = Path("/data/derivatives/hcp379_v2")
PYTHON = Path("/home/ec2-user/fsl/bin/python")
PHASE_SOURCE = (
    EXP / "scripts/hcp/run_hcp379_phase_b_stability_v2.py"
)
DEFAULT_HUMAN_QC = (
    HCP_ROOT
    / "review_recovery4_corrected_overlay/"
    "human_visual_qc_recovery4_corrected_overlay.csv"
)
ROOT = HCP_ROOT / "parallel_recovery4"
PLAN = ROOT / "parallel_phase_b_pretract_plan.json"
ATTEMPTS = ROOT / "attempts"
STATUS_SOURCE = EXP / "scripts/hcp/hcp379_v2_status.py"
STORAGE_FORECAST = HCP_ROOT / "manifests/storage_capacity_forecast.json"
TOPOLOGY_VALIDATION = (
    EXP
    / "research_audit/outputs/"
    "hcp379_density_execution_topology_v3/validation.json"
)
COMPACTOR_VALIDATION = (
    EXP
    / "research_audit/outputs/"
    "hcp379_pretract_compaction_recovery4_v3/validation.json"
)
MAX_ACTIVE_PRETRACT_SUBJECTS = 4
PRETRACT_THREADS_PER_SUBJECT = 8
PHASE_B_CORES = 24
MINIMUM_FREE_TO_LAUNCH_BYTES = 700_000_000_000
EMERGENCY_FREE_BYTES = 450_000_000_000
PHASE_LIVE_START_TIMEOUT_SECONDS = 900


def lane(
    *,
    name: str,
    source: str,
    validation: str,
    expected_checks: int,
    target_n: int,
    summary: str,
) -> dict[str, Any]:
    return {
        "name": name,
        "source": EXP / "scripts/hcp" / source,
        "validation": (
            EXP / "research_audit/outputs" / validation / "validation.json"
        ),
        "expected_checks": expected_checks,
        "target_n": target_n,
        "summary": HCP_ROOT / summary,
    }


LANES = (
    lane(
        name="corrected_core_216_new",
        source="hcp379_scaleup_pretract_recovery4_v3.py",
        validation="hcp379_scaleup_pretract_recovery4_v3",
        expected_checks=19,
        target_n=216,
        summary=(
            "corrected_scaleup_recovery4/manifests/"
            "pretract_recovery4_summary.json"
        ),
    ),
    lane(
        name="corrected_legacy_tensor_212_new",
        source="run_hcp379_legacy_density_pretract_recovery4_v3.py",
        validation="hcp379_legacy_density_recovery4_v3",
        expected_checks=9,
        target_n=212,
        summary=(
            "corrected_legacy_tensor_recovery4/manifests/"
            "pretract_recovery4_summary.json"
        ),
    ),
    lane(
        name="corrected_freesurfer_1_new",
        source="run_hcp379_freesurfer_pretract_recovery4_v3.py",
        validation="hcp379_freesurfer_recovery4_v3",
        expected_checks=11,
        target_n=1,
        summary=(
            "corrected_freesurfer_recovery4/manifests/"
            "pretract_recovery4_summary.json"
        ),
    ),
    lane(
        name="archive_calibration_6",
        source=(
            "run_hcp379_archive_calibration_pretract_recovery4_v3.py"
        ),
        validation="hcp379_archive_calibration_recovery4_v3",
        expected_checks=20,
        target_n=6,
        summary=(
            "archive_corrected_calibration_recovery4/manifests/"
            "pretract_recovery4_summary.json"
        ),
    ),
    lane(
        name="corrected_archive_low_28_new",
        source="run_hcp379_archive_low_pretract_recovery4_v3.py",
        validation="hcp379_archive_low_recovery4_v3",
        expected_checks=9,
        target_n=28,
        summary=(
            "corrected_archive_low_recovery4/manifests/"
            "pretract_recovery4_summary.json"
        ),
    ),
    lane(
        name="corrected_archive_D0_52_new",
        source="run_hcp379_archive_D0_pretract_recovery4_v3.py",
        validation="hcp379_archive_D0_recovery4_v3",
        expected_checks=12,
        target_n=52,
        summary=(
            "corrected_archive_D0_recovery4/manifests/"
            "pretract_recovery4_summary.json"
        ),
    ),
)


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PHASE = load_module(PHASE_SOURCE, "hcp379_parallel_phase_b_contract")
PRETRACT = load_module(
    Path(LANES[0]["source"]),
    "hcp379_parallel_pretract_hroi_policy_contract",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        encoding="utf-8",
        delete=False,
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def validation_record(
    path: Path, *, expected_checks: int
) -> dict[str, Any]:
    value = load_json(path)
    if (
        value.get("status") != "PASS"
        or value.get("passed_checks") != expected_checks
        or value.get("total_checks") != expected_checks
    ):
        raise ValueError(f"package validation differs: {path}")
    return PHASE.file_record(path)


def validate_static_contract() -> dict[str, Any]:
    if (
        MAX_ACTIVE_PRETRACT_SUBJECTS != 4
        or PRETRACT_THREADS_PER_SUBJECT != 8
        or PHASE_B_CORES != 24
        or PHASE_B_CORES
        + MAX_ACTIVE_PRETRACT_SUBJECTS
        * PRETRACT_THREADS_PER_SUBJECT
        > (os.cpu_count() or 1)
        or len(LANES) != 6
        or sum(int(item["target_n"]) for item in LANES) != 515
    ):
        raise ValueError("parallel resource or lane contract differs")
    topology = load_json(TOPOLOGY_VALIDATION)
    if (
        topology.get("status") != "PASS"
        or topology.get("passed_checks") != 13
        or topology.get("total_checks") != 13
    ):
        raise ValueError("density execution topology validation differs")
    compactor = load_json(COMPACTOR_VALIDATION)
    if (
        compactor.get("status") != "PASS"
        or compactor.get("passed_checks") != 11
        or compactor.get("total_checks") != 11
    ):
        raise ValueError("Recovery4 compactor validation differs")
    storage = load_json(STORAGE_FORECAST)
    assumptions = storage.get("assumptions", {})
    if (
        storage.get("status")
        != "SAFE_WITH_VALIDATED_STREAMING_COMPACTION"
        or storage.get("phase_b_only", {}).get(
            "safe_with_current_volume_and_reserve"
        )
        is not True
        or assumptions.get("streaming_pretract_workers")
        != MAX_ACTIVE_PRETRACT_SUBJECTS
        or assumptions.get("streaming_pretract_compaction_validated")
        is not True
        or set(storage.get("streaming_safe_streamline_counts", []))
        != {3_000_000, 5_000_000, 10_000_000}
    ):
        raise ValueError("storage streaming contract differs")
    sources = PHASE.active_sources()
    units = PHASE.units_from_pretract(
        Path(sources["pretract_manifest"])
    )
    upstream = PHASE.validate_upstream(units, sources=sources)
    hard_hold_units = list(
        upstream.pop("corrected_overlay_hard_hold_units", [])
    )
    raw_hard_hold_units = list(
        upstream.pop("corrected_overlay_raw_hard_hold_units", [])
    )
    source_mode = str(upstream.pop("source_mode"))
    promotion_receipt = upstream.pop("promotion_receipt", None)
    hroi_source_policy = PRETRACT.validate_hroi_policy_adoption(
        required=False
    )
    records: dict[str, Any] = {
        "phase_b_runner": PHASE.file_record(PHASE_SOURCE),
        "status_runner": PHASE.file_record(STATUS_SOURCE),
        "topology_validation": PHASE.file_record(TOPOLOGY_VALIDATION),
        "compactor_validation": PHASE.file_record(COMPACTOR_VALIDATION),
        "storage_forecast": PHASE.file_record(STORAGE_FORECAST),
        **upstream,
    }
    if promotion_receipt is not None:
        records["promotion_receipt"] = promotion_receipt
    for item in LANES:
        source = Path(item["source"])
        validation = Path(item["validation"])
        if not source.is_file():
            raise FileNotFoundError(source)
        records[f"{item['name']}_runner"] = PHASE.file_record(source)
        records[f"{item['name']}_validation"] = validation_record(
            validation,
            expected_checks=int(item["expected_checks"]),
        )
    return {
        "records": records,
        "canary_units": units,
        "storage": storage,
        "phase_b_source_mode": source_mode,
        "corrected_overlay_raw_hard_hold_units": raw_hard_hold_units,
        "corrected_overlay_hard_hold_units": hard_hold_units,
        "hroi_source_policy": {
            key: value
            for key, value in hroi_source_policy.items()
            if key != "units"
        },
    }


def phase_command(human_qc: Path) -> list[str]:
    return [
        str(PYTHON),
        str(PHASE_SOURCE),
        "--human-qc-manifest",
        str(human_qc.resolve()),
        "--cores",
        str(PHASE_B_CORES),
    ]


def lane_command(item: Mapping[str, Any], human_qc: Path) -> list[str]:
    return [
        str(PYTHON),
        str(item["source"]),
        "--execute",
        "--human-qc-manifest",
        str(human_qc.resolve()),
        "--workers",
        "1",
        "--nthreads",
        str(PRETRACT_THREADS_PER_SUBJECT),
        "--compact-reproducible-after-pass",
    ]


def preflight_payload(human_qc: Path) -> dict[str, Any]:
    static = validate_static_contract()
    if human_qc.is_file():
        human_record: Any = PHASE.validate_human_qc(
            human_qc, static["canary_units"]
        )
        status = (
            "READY_TO_LAUNCH_AFTER_GENUINE_HUMAN_QC_AND_HROI_POLICY"
            if static["hroi_source_policy"][
                "pretract_use_authorized"
            ]
            is True
            else "AWAITING_530_HROI_SOURCE_POLICY"
        )
    else:
        human_record = {
            "status": "MISSING_GENUINE_HUMAN_QC",
            "expected_path": str(human_qc.resolve()),
        }
        status = "AWAITING_GENUINE_HUMAN_QC"
    plan = {
        "schema_version": "1.0.0",
        "record_type": (
            "diagnosis_blind_hcp379_parallel_phase_b_pretract_plan"
        ),
        "status": status,
        "generated_utc": utc_now(),
        "diagnosis_labels_used": False,
        "diagnosis_or_outcomes_used": False,
        "non_overwriting": True,
        "imaging_executed_by_plan": False,
        "selected_count_tractography_authorized": False,
        "human_visual_qc": human_record,
        "hroi_source_policy": static["hroi_source_policy"],
        "phase_b_source_mode": static["phase_b_source_mode"],
        "corrected_overlay_gate": {
            "raw_hard_hold_units": static[
                "corrected_overlay_raw_hard_hold_units"
            ],
            "hard_hold_units": static[
                "corrected_overlay_hard_hold_units"
            ],
            "ready": not static[
                "corrected_overlay_hard_hold_units"
            ],
        },
        "resource_contract": {
            "host_logical_cpus": os.cpu_count(),
            "phase_b_cores": PHASE_B_CORES,
            "maximum_active_pretract_subjects": (
                MAX_ACTIVE_PRETRACT_SUBJECTS
            ),
            "pretract_threads_per_subject": (
                PRETRACT_THREADS_PER_SUBJECT
            ),
            "maximum_declared_parallel_threads": (
                PHASE_B_CORES
                + MAX_ACTIVE_PRETRACT_SUBJECTS
                * PRETRACT_THREADS_PER_SUBJECT
            ),
            "minimum_free_to_launch_bytes": (
                MINIMUM_FREE_TO_LAUNCH_BYTES
            ),
            "emergency_free_bytes": EMERGENCY_FREE_BYTES,
            "streaming_compaction_required": True,
        },
        "launch_order": {
            "phase_b_first": True,
            "wait_for_phase_b_live_dag_before_pretract": True,
            "pretract_queue": [str(item["name"]) for item in LANES],
            "maximum_simultaneous_lane_processes": (
                MAX_ACTIVE_PRETRACT_SUBJECTS
            ),
            "continue_pretract_if_phase_b_later_fails": True,
            "rationale": (
                "pretract is independent after genuine visual QC; "
                "selected-count tractography remains prohibited"
            ),
        },
        "phase_b": {
            "target_n": 15,
            "command": phase_command(human_qc),
        },
        "pretract_lanes": [
            {
                "name": item["name"],
                "target_n": item["target_n"],
                "summary_expected": str(item["summary"]),
                "command": lane_command(item, human_qc),
            }
            for item in LANES
        ],
        "records": {
            "launcher": PHASE.file_record(Path(__file__)),
            **static["records"],
        },
    }
    PLAN.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(PLAN, plan)
    return plan


def other_supervisors() -> list[str]:
    output = subprocess.run(
        ["pgrep", "-af", Path(__file__).name],
        check=False,
        capture_output=True,
        text=True,
    ).stdout
    own = str(os.getpid())
    return [
        line
        for line in output.splitlines()
        if not line.startswith(f"{own} ")
        and "--supervise" in line
        and "pgrep -af" not in line
    ]


def validate_launch(human_qc: Path) -> dict[str, Any]:
    plan = preflight_payload(human_qc)
    if plan["status"] == "AWAITING_GENUINE_HUMAN_QC":
        raise FileNotFoundError(
            f"genuine human-QC CSV is required: {human_qc.resolve()}"
        )
    if plan["status"] == "AWAITING_530_HROI_SOURCE_POLICY":
        raise FileNotFoundError(
            "530-subject HROI pretract-adoption receipt is required"
        )
    if (
        plan["status"]
        != "READY_TO_LAUNCH_AFTER_GENUINE_HUMAN_QC_AND_HROI_POLICY"
    ):
        raise ValueError("parallel pretract launch status differs")
    active = PHASE.active_imaging_processes()
    if active:
        raise RuntimeError(
            "another imaging process is active: " + " | ".join(active[:5])
        )
    supervisors = other_supervisors()
    if supervisors:
        raise RuntimeError(
            "parallel Recovery4 supervisor is already active: "
            + " | ".join(supervisors)
        )
    free = shutil.disk_usage(HCP_ROOT).free
    if free < MINIMUM_FREE_TO_LAUNCH_BYTES:
        raise RuntimeError("free storage is below the parallel launch floor")
    return plan


def update_state(
    path: Path, state: dict[str, Any], **updates: Any
) -> None:
    state.update(updates)
    state["updated_utc"] = utc_now()
    atomic_json(path, state)


def start_process(
    command: list[str], log_path: Path
) -> tuple[subprocess.Popen[Any], Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("x", encoding="utf-8")
    process = subprocess.Popen(
        command,
        cwd=EXP,
        stdout=handle,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    return process, handle


def summary_state(item: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(item["summary"])
    if not path.is_file():
        return {
            "status": "MISSING_SUMMARY",
            "path": str(path),
            "expected_n": item["target_n"],
        }
    value = load_json(path)
    passed = bool(
        value.get("status") == "PASS"
        and value.get("target_n") == item["target_n"]
        and value.get("states_present_n") == item["target_n"]
        and value.get("status_counts", {}).get(
            "PASS_PRETRACT_RECOVERY4"
        )
        == item["target_n"]
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "expected_n": item["target_n"],
        "summary": PHASE.file_record(path),
        "observed": value,
    }


def terminate_group(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return


def supervise(attempt_root: Path, human_qc: Path) -> int:
    static = validate_static_contract()
    human_record = PHASE.validate_human_qc(
        human_qc, static["canary_units"]
    )
    state_path = attempt_root / "supervisor_state.json"
    state: dict[str, Any] = {
        "schema_version": "1.0.0",
        "record_type": (
            "diagnosis_blind_hcp379_parallel_pretract_supervisor"
        ),
        "status": "STARTING_PHASE_B",
        "started_utc": utc_now(),
        "updated_utc": utc_now(),
        "diagnosis_labels_used": False,
        "human_visual_qc": human_record,
        "phase_b": {},
        "pretract": {
            "queue": [str(item["name"]) for item in LANES],
            "active": {},
            "terminal": {},
        },
    }
    atomic_json(state_path, state)
    phase_log = attempt_root / "phase_b_launcher.log"
    baseline_live = {
        path.resolve()
        for path in PHASE.PHASE_ROOT.glob("*.snakemake.log")
    }
    phase_process, phase_handle = start_process(
        phase_command(human_qc), phase_log
    )
    update_state(
        state_path,
        state,
        status="WAITING_FOR_PHASE_B_LIVE_DAG",
        phase_b={
            "pid": phase_process.pid,
            "command": phase_command(human_qc),
            "log": str(phase_log),
        },
    )
    deadline = time.monotonic() + PHASE_LIVE_START_TIMEOUT_SECONDS
    phase_live = False
    while time.monotonic() < deadline:
        current = {
            path.resolve()
            for path in PHASE.PHASE_ROOT.glob("*.snakemake.log")
        }
        if current - baseline_live:
            phase_live = True
            break
        returncode = phase_process.poll()
        if returncode is not None:
            phase_handle.close()
            update_state(
                state_path,
                state,
                status="PHASE_B_FAILED_BEFORE_LIVE_DAG",
                phase_b={
                    **state["phase_b"],
                    "returncode": returncode,
                },
            )
            return 1
        time.sleep(2)
    if not phase_live:
        terminate_group(phase_process)
        phase_handle.close()
        update_state(
            state_path,
            state,
            status="PHASE_B_LIVE_START_TIMEOUT",
        )
        return 1
    update_state(
        state_path,
        state,
        status="PHASE_B_AND_PRETRACT_RUNNING",
        phase_b={**state["phase_b"], "live_dag_started": True},
    )

    queue = list(LANES)
    active: dict[
        str, tuple[Mapping[str, Any], subprocess.Popen[Any], Any, Path]
    ] = {}
    terminal: dict[str, Any] = {}
    emergency = False
    while (
        queue
        or active
        or phase_process.poll() is None
    ):
        free = shutil.disk_usage(HCP_ROOT).free
        if free < EMERGENCY_FREE_BYTES:
            emergency = True
            for _, process, _, _ in active.values():
                terminate_group(process)
            terminate_group(phase_process)
        while (
            queue
            and len(active) < MAX_ACTIVE_PRETRACT_SUBJECTS
            and free >= MINIMUM_FREE_TO_LAUNCH_BYTES
            and not emergency
        ):
            item = queue.pop(0)
            name = str(item["name"])
            log_path = attempt_root / f"{name}.log"
            process, handle = start_process(
                lane_command(item, human_qc), log_path
            )
            active[name] = (item, process, handle, log_path)
        for name, (item, process, handle, log_path) in list(
            active.items()
        ):
            returncode = process.poll()
            if returncode is None:
                continue
            handle.close()
            terminal[name] = {
                "returncode": returncode,
                "log": str(log_path),
                "summary_state": summary_state(item),
            }
            del active[name]
        phase_returncode = phase_process.poll()
        update_state(
            state_path,
            state,
            status=(
                "EMERGENCY_STORAGE_STOP"
                if emergency
                else "PHASE_B_AND_PRETRACT_RUNNING"
            ),
            free_bytes=free,
            phase_b={
                **state["phase_b"],
                "returncode": phase_returncode,
            },
            pretract={
                "queue": [str(item["name"]) for item in queue],
                "active": {
                    name: {
                        "pid": process.pid,
                        "log": str(log_path),
                    }
                    for name, (_, process, _, log_path) in active.items()
                },
                "terminal": terminal,
            },
        )
        if emergency:
            break
        if not queue and not active and phase_returncode is not None:
            break
        time.sleep(15)

    if phase_process.poll() is None:
        phase_process.wait()
    phase_handle.close()
    phase_validation = (
        load_json(PHASE.STABILITY_OUTPUT)
        if PHASE.STABILITY_OUTPUT.is_file()
        else None
    )
    lane_pass = bool(
        len(terminal) == len(LANES)
        and all(
            row.get("returncode") == 0
            and row.get("summary_state", {}).get("status") == "PASS"
            for row in terminal.values()
        )
    )
    phase_pass = bool(
        phase_process.returncode == 0
        and phase_validation
        and phase_validation.get("status") == "PASS"
        and phase_validation.get(
            "smallest_density_qualified_scale_up_streamline_count"
        )
        in {3_000_000, 5_000_000, 10_000_000}
    )
    final_status = (
        "PASS_PHASE_B_AND_ALL_PRETRACT"
        if lane_pass and phase_pass and not emergency
        else "TERMINAL_WITH_FAILURES"
    )
    update_state(
        state_path,
        state,
        status=final_status,
        completed_utc=utc_now(),
        phase_b={
            **state["phase_b"],
            "returncode": phase_process.returncode,
            "validation": (
                PHASE.file_record(PHASE.STABILITY_OUTPUT)
                if PHASE.STABILITY_OUTPUT.is_file()
                else None
            ),
        },
        pretract={
            "queue": [],
            "active": {},
            "terminal": terminal,
        },
    )
    subprocess.run(
        [str(PYTHON), str(STATUS_SOURCE)],
        check=False,
        capture_output=True,
        text=True,
    )
    return 0 if final_status == "PASS_PHASE_B_AND_ALL_PRETRACT" else 1


def launch(human_qc: Path) -> dict[str, Any]:
    plan = validate_launch(human_qc)
    attempt_id = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        + f"-parallel-recovery4-{os.urandom(4).hex()}"
    )
    attempt_root = ATTEMPTS / attempt_id
    attempt_root.mkdir(parents=True, exist_ok=False)
    launch_record = {
        "schema_version": "1.0.0",
        "record_type": (
            "diagnosis_blind_hcp379_parallel_pretract_launch"
        ),
        "status": "STARTING_SUPERVISOR",
        "attempt_id": attempt_id,
        "created_utc": utc_now(),
        "diagnosis_labels_used": False,
        "human_visual_qc": PHASE.file_record(human_qc),
        "plan": PHASE.file_record(PLAN),
        "supervisor_command": [
            str(PYTHON),
            str(Path(__file__)),
            "--supervise",
            "--attempt-root",
            str(attempt_root),
            "--human-qc-manifest",
            str(human_qc.resolve()),
        ],
        "resource_contract": plan["resource_contract"],
    }
    launch_path = attempt_root / "launch.json"
    atomic_json(launch_path, launch_record)
    supervisor_log = attempt_root / "supervisor.log"
    handle = supervisor_log.open("x", encoding="utf-8")
    process = subprocess.Popen(
        launch_record["supervisor_command"],
        cwd=EXP,
        stdout=handle,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    handle.close()
    launch_record.update(
        {
            "status": "SUPERVISOR_STARTED",
            "supervisor_pid": process.pid,
            "supervisor_log": str(supervisor_log),
        }
    )
    atomic_json(launch_path, launch_record)
    return launch_record


def self_test() -> None:
    static = validate_static_contract()
    if (
        len(static["canary_units"]) != 15
        or len(LANES) != 6
        or sum(int(item["target_n"]) for item in LANES) != 515
        or [str(item["name"]) for item in LANES[:4]]
        != [
            "corrected_core_216_new",
            "corrected_legacy_tensor_212_new",
            "corrected_freesurfer_1_new",
            "archive_calibration_6",
        ]
        or any(
            "--compact-reproducible-after-pass"
            not in lane_command(item, DEFAULT_HUMAN_QC)
            for item in LANES
        )
    ):
        raise AssertionError("parallel pretract self-test differs")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--launch", action="store_true")
    mode.add_argument("--supervise", action="store_true")
    mode.add_argument("--self-test", action="store_true")
    parser.add_argument(
        "--human-qc-manifest", type=Path, default=DEFAULT_HUMAN_QC
    )
    parser.add_argument("--attempt-root", type=Path)
    args = parser.parse_args()

    if args.self_test:
        self_test()
        print("HCP379_PARALLEL_PRETRACT_RECOVERY4_SELF_TEST_PASS")
        return 0
    if args.supervise:
        if args.attempt_root is None:
            parser.error("--supervise requires --attempt-root")
        return supervise(
            args.attempt_root.resolve(),
            args.human_qc_manifest.resolve(),
        )
    payload = (
        launch(args.human_qc_manifest)
        if args.launch
        else preflight_payload(args.human_qc_manifest)
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
