#!/usr/bin/env python3
"""Launch the Recovery4-bound HCP379 3M/5M/10M stability canary.

This launcher is intentionally fail closed.  It requires the immutable 15/15
Recovery4 pre-tractography manifest, its blinded review pack, and a genuine
human-QC manifest with one PASS row per bound unit.  It then dry-runs the DAG
and rejects any attempt to schedule upstream imaging recovery before starting
tractography.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml


EXP = Path("/home/ec2-user/exp")
WORKFLOW = EXP / "scforge/workflow"
RUN_ROOT = Path(
    "/data/derivatives/scforge_v2/h04a_r1_recovery_20260719_retry4"
)
HCP_ROOT = Path("/data/derivatives/hcp379_v2")
PHASE_ROOT = HCP_ROOT / "phase_b_stability_recovery4"
SNAKEFILE = (
    WORKFLOW / "Snakefile_h04a_r1_retry4_phase_b_hcp379_recovery4"
)
EXTENSION = (
    WORKFLOW
    / "extensions/h04a_r1_retry4_phase_b_hcp379/"
    "07_hcp379_stability.smk"
)
TCK_HEADER_COUNT = (
    EXP / "scripts/hcp/tck_header_count_v1.py"
)
NUMERIC_VALUE_COUNT = (
    EXP / "scripts/hcp/count_mrtrix_numeric_values_v1.py"
)
BUILDER = WORKFLOW / "build_hcp379_phase_b_stability_manifest.py"
VALIDATOR = (
    EXP / "research_audit/validate_hcp379_phase_b_stability.py"
)
BASE_CONFIG = Path(
    "/data/derivatives/scforge_v2/"
    "h04a_r1_recovery_20260719_retry4/attempts/"
    "20260719T133038.809063Z-pre-tractography-recovery1-80a35e74."
    "resolved_config.yaml"
)
ORIGINAL_PRETRACT_MANIFEST = (
    HCP_ROOT / "pretract_recovery4/pretract_recovery4_manifest.json"
)
ORIGINAL_REVIEW_MANIFEST = (
    HCP_ROOT
    / "review_recovery4_corrected_overlay/"
    "hcp379_visual_review_pack_recovery4_corrected_overlay_manifest.json"
)
PROMOTED_ROOT = HCP_ROOT / "pretract_promoted_recovery4_v6"
PROMOTED_PRETRACT_MANIFEST = (
    PROMOTED_ROOT / "pretract_promoted_recovery4_v6_manifest.json"
)
PROMOTION_RECEIPT = PROMOTED_ROOT / "promotion_receipt.json"
EXACT_V6_REVIEW_MANIFEST = (
    HCP_ROOT
    / "review_recovery4_candidate_overlay_v6/"
    "hcp379_visual_review_pack_recovery4_candidate_v6_manifest.json"
)
CORRECTED_OVERLAY_QUANTITATIVE_AUDIT = (
    HCP_ROOT
    / "review_recovery4_corrected_overlay/"
    "hcp379_corrected_overlay_quantitative_audit.json"
)
DEFAULT_HUMAN_QC = (
    HCP_ROOT
    / "review_recovery4_corrected_overlay/"
    "human_visual_qc_recovery4_corrected_overlay.csv"
)
SNAKEMAKE = (
    EXP / ".venv_connectome_workflow/bin/snakemake"
)
STABILITY_OUTPUT = (
    RUN_ROOT / "publication/hcp379_recipe_stability_validation.json"
)
MINIMUM_FREE_BYTES_TO_START = 1_200_000_000_000
EMERGENCY_FREE_BYTES = 400_000_000_000
FORBIDDEN_SCHEDULED_RULES = {
    "normalize_dwi_source",
    "normalize_t1_source",
    "gradient_contract",
    "dwi_denoise",
    "dwi_degibbs",
    "dwi_motion_eddy",
    "dwi_bias_correct",
    "dwi_brain_mask",
    "select_fod_shells",
    "subject_response",
    "response_calibration_phase_a",
    "mni_registration_fixed_image",
    "five_tt_dwi",
    "mtnormalise",
    "automated_pre_tractography_qc",
    "visual_review_bundle",
    "connectome_count",
    "connectome_fd_sum",
    "connectome_len_mean",
    "connectome_invlen_mean",
    "tensor_streamline_samples",
    "tensor_connectome",
    "count_invnodevol",
    "matrix_qc",
    "provenance_sidecar",
    "publish_manifest",
}
ALLOWED_CONCURRENT_PRETRACT_ROOTS = (
    HCP_ROOT / "corrected_scaleup_recovery4",
    HCP_ROOT / "corrected_legacy_tensor_recovery4",
    HCP_ROOT / "archive_corrected_calibration_recovery4",
    HCP_ROOT / "corrected_archive_low_recovery4",
    HCP_ROOT / "corrected_archive_D0_recovery4",
    HCP_ROOT / "corrected_freesurfer_recovery4",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise ValueError(f"not a regular file: {resolved}")
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root is not an object: {path}")
    return value


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
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def verified_file_record(record: Mapping[str, Any]) -> Path:
    path = Path(str(record.get("path", ""))).resolve()
    if (
        not path.is_file()
        or path.is_symlink()
        or record.get("size_bytes") != path.stat().st_size
        or record.get("sha256") != sha256_file(path)
    ):
        raise ValueError(f"artifact binding differs: {path}")
    return path


def active_sources() -> dict[str, Any]:
    promoted_master_exists = PROMOTED_PRETRACT_MANIFEST.is_file()
    promotion_receipt_exists = PROMOTION_RECEIPT.is_file()
    if promoted_master_exists != promotion_receipt_exists:
        raise ValueError(
            "partial Recovery4 V6 promotion: promoted master and receipt "
            "must either both exist or both be absent"
        )
    if not promoted_master_exists:
        return {
            "mode": "ORIGINAL_AWAITING_CORRECTED_HUMAN_REVIEW",
            "pretract_manifest": ORIGINAL_PRETRACT_MANIFEST,
            "review_manifest": ORIGINAL_REVIEW_MANIFEST,
            "promotion_receipt": None,
        }
    if not EXACT_V6_REVIEW_MANIFEST.is_file():
        raise FileNotFoundError(EXACT_V6_REVIEW_MANIFEST)
    return {
        "mode": "V6_HUMAN_REVIEWED_PROMOTION",
        "pretract_manifest": PROMOTED_PRETRACT_MANIFEST,
        "review_manifest": EXACT_V6_REVIEW_MANIFEST,
        "promotion_receipt": PROMOTION_RECEIPT,
    }


def units_from_pretract(pretract_manifest: Path) -> list[str]:
    master = load_json(pretract_manifest)
    rows = master.get("units")
    if (
        master.get("record_type")
        != "diagnosis_blind_hcp379_pretract_recovery4_manifest"
        or master.get("status")
        != "AUTOMATED_PASS_AWAITING_HUMAN_VISUAL_QC"
        or master.get("diagnosis_labels_used") is not False
        or master.get("human_visual_qc_inferred") is not False
        or master.get("unit_count") != 15
        or not isinstance(rows, list)
    ):
        raise ValueError("Recovery4 pretract master differs")
    units = [str(row.get("unit", "")) for row in rows]
    if len(units) != 15 or len(set(units)) != 15 or not all(units):
        raise ValueError("Recovery4 pretract unit set differs")
    for row in rows:
        path = verified_file_record(row.get("manifest", {}))
        record = load_json(path)
        if (
            record.get("unit") != row.get("unit")
            or record.get("status")
            != "AUTOMATED_PASS_AWAITING_HUMAN_VISUAL_QC"
            or record.get("diagnosis_labels_used") is not False
            or record.get("human_visual_qc_inferred") is not False
        ):
            raise ValueError(
                f"Recovery4 unit manifest differs: {row.get('unit')}"
            )
    return sorted(units)


def validate_upstream(
    units: list[str],
    *,
    sources: Mapping[str, Any],
) -> dict[str, Any]:
    pretract_manifest = Path(sources["pretract_manifest"]).resolve()
    review_manifest = Path(sources["review_manifest"]).resolve()
    promotion_receipt_path = sources.get("promotion_receipt")
    master = load_json(pretract_manifest)
    review = load_json(review_manifest)
    corrected_overlay_audit = load_json(
        CORRECTED_OVERLAY_QUANTITATIVE_AUDIT
    )
    if (
        master.get("status")
        != "AUTOMATED_PASS_AWAITING_HUMAN_VISUAL_QC"
        or master.get("diagnosis_labels_used") is not False
        or master.get("human_visual_qc_inferred") is not False
        or master.get("tractography_started") is not False
        or master.get("matrix_generation_started") is not False
        or {
            str(row.get("unit")) for row in master.get("units", [])
        }
        != set(units)
    ):
        raise ValueError("Recovery4 pretract master is not an exact PASS")
    promoted = sources.get("mode") == "V6_HUMAN_REVIEWED_PROMOTION"
    expected_review_type = (
        "diagnosis_blind_hcp379_visual_review_pack_recovery4_candidate_v6"
        if promoted
        else (
            "diagnosis_blind_hcp379_visual_review_pack_"
            "recovery4_corrected_overlay"
        )
    )
    expected_review_status = (
        "READY_FOR_EXACT_INPUT_HUMAN_REVIEW"
        if promoted
        else "READY_FOR_HUMAN_REVIEW"
    )
    if (
        review.get("record_type") != expected_review_type
        or review.get("status") != expected_review_status
        or review.get("diagnosis_labels_used") is not False
        or review.get("human_visual_qc_inferred") is not False
        or review.get("unit_count") != 15
        or {str(row.get("unit")) for row in review.get("units", [])}
        != set(units)
    ):
        raise ValueError("blinded review pack differs")
    if (
        not promoted
        and review.get("pretract_manifest")
        != file_record(pretract_manifest)
    ):
        raise ValueError("review pack is not bound to the Recovery4 manifest")
    for row in review["units"]:
        verified_file_record(row.get("review_page", {}))
        for record in row.get("source_panels", {}).values():
            verified_file_record(record)
    if (
        corrected_overlay_audit.get("record_type")
        != (
            "diagnosis_blind_hcp379_recovery4_corrected_overlay_"
            "quantitative_audit"
        )
        or corrected_overlay_audit.get("diagnosis_labels_used") is not False
        or corrected_overlay_audit.get("unit_count") != 15
        or corrected_overlay_audit.get("all_units_have_379_labels")
        is not True
        or not isinstance(
            corrected_overlay_audit.get("hard_hold_units"), list
        )
    ):
        raise ValueError("corrected-overlay quantitative audit differs")
    raw_hard_holds = list(corrected_overlay_audit["hard_hold_units"])
    promotion_record = None
    effective_hard_holds = raw_hard_holds
    if promoted:
        if promotion_receipt_path is None:
            raise ValueError("promoted source lacks a promotion receipt")
        promotion_receipt_path = Path(promotion_receipt_path).resolve()
        receipt = load_json(promotion_receipt_path)
        records = receipt.get("records", {})
        hard_hold_resolution = receipt.get("hard_hold_resolution", {})
        if (
            receipt.get("record_type")
            != (
                "diagnosis_blind_hcp379_recovery4_v6_"
                "promotion_receipt"
            )
            or receipt.get("status") != "PASS"
            or receipt.get("diagnosis_labels_used") is not False
            or receipt.get("human_visual_qc_inferred") is not False
            or receipt.get("tractography_started") is not False
            or receipt.get("matrix_generation_started") is not False
            or receipt.get("unit_count") != 15
            or records.get("promoted_pretract_master")
            != file_record(pretract_manifest)
            or records.get("exact_input_review_manifest")
            != file_record(review_manifest)
            or records.get("quantitative_audit")
            != file_record(CORRECTED_OVERLAY_QUANTITATIVE_AUDIT)
            or hard_hold_resolution.get("source_hard_hold_units")
            != raw_hard_holds
            or hard_hold_resolution.get("resolved_units")
            != raw_hard_holds
            or hard_hold_resolution.get("remaining_hard_hold_units")
            != []
            or master.get("route_substitution", {}).get(
                "promotion_status"
            )
            != "HUMAN_VISUAL_QC_PASS_PROMOTED"
            or master.get("canonical_human_qc")
            != records.get("canonical_human_qc")
        ):
            raise ValueError("Recovery4 V6 promotion receipt differs")
        for label, record in records.items():
            verified_file_record(record)
        promotion_record = file_record(promotion_receipt_path)
        effective_hard_holds = []
    return {
        "recovery4_pretract_manifest": file_record(pretract_manifest),
        "visual_review_pack": file_record(review_manifest),
        "corrected_overlay_quantitative_audit": file_record(
            CORRECTED_OVERLAY_QUANTITATIVE_AUDIT
        ),
        "corrected_overlay_raw_hard_hold_units": raw_hard_holds,
        "corrected_overlay_hard_hold_units": effective_hard_holds,
        "promotion_receipt": promotion_record,
        "source_mode": sources.get("mode"),
    }


def validate_human_qc(path: Path, units: list[str]) -> dict[str, Any]:
    path = path.expanduser().resolve()
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = tuple(reader.fieldnames or ())
        forbidden = {
            "diagnosis",
            "diagnosis_at_dti",
            "group",
            "research_group",
            "outcome",
        }
        if forbidden.intersection(
            str(field).strip().lower() for field in fields
        ):
            raise ValueError("human-QC manifest contains a blinding field")
        required = {"unit", "status", "reviewer", "reviewed_utc"}
        if not required.issubset(fields):
            raise ValueError("human-QC manifest lacks required fields")
        rows = [
            {str(key): str(value or "").strip() for key, value in row.items()}
            for row in reader
        ]
    if (
        len(rows) != 15
        or len({row["unit"] for row in rows}) != 15
        or {row["unit"] for row in rows} != set(units)
    ):
        raise ValueError("human-QC rows differ from the exact 15-unit set")
    for row in rows:
        if (
            row["status"].upper() != "PASS"
            or not row["reviewer"]
            or not row["reviewed_utc"]
        ):
            raise ValueError(
                f"human visual QC is incomplete or not PASS for {row['unit']}"
            )
        reviewed_at = datetime.fromisoformat(
            row["reviewed_utc"][:-1] + "+00:00"
            if row["reviewed_utc"].endswith("Z")
            else row["reviewed_utc"]
        )
        if reviewed_at.tzinfo is None:
            raise ValueError(
                f"human visual-QC timestamp lacks timezone for {row['unit']}"
            )
        if reviewed_at > datetime.now(timezone.utc):
            raise ValueError(
                f"human visual-QC timestamp is in the future for {row['unit']}"
            )
    return file_record(path)


def active_imaging_processes() -> list[str]:
    output = subprocess.run(
        [
            "pgrep",
            "-af",
            "snakemake|eddy_cpu|eddy_cuda|dwifslpreproc|dwi2response|"
            "ss3t_csd_beta1|antsRegistration|tckgen|tcksift2",
        ],
        check=False,
        capture_output=True,
        text=True,
    ).stdout
    return [
        line
        for line in output.splitlines()
        if "pgrep -af" not in line
        and "run_hcp379_phase_b_stability_v2.py" not in line
    ]


def resume_process_contract(
    active: list[str],
) -> dict[str, Any]:
    allowed: list[str] = []
    unexpected: list[str] = []
    roots = tuple(str(path.resolve()) for path in ALLOWED_CONCURRENT_PRETRACT_ROOTS)
    for line in active:
        if any(root in line for root in roots):
            allowed.append(line)
        else:
            unexpected.append(line)
    failed_attempts = []
    for path in sorted(PHASE_ROOT.glob("*.attempt.json")):
        try:
            value = load_json(path)
        except Exception:
            continue
        if (
            value.get("record_type")
            == "diagnosis_blind_hcp379_phase_b_attempt"
            and value.get("status") == "FAIL"
            and value.get("completed_utc")
        ):
            failed_attempts.append(file_record(path))
    if unexpected:
        raise RuntimeError(
            "resume found unexpected imaging processes: "
            + " | ".join(unexpected[:5])
        )
    if not failed_attempts:
        raise RuntimeError(
            "resume requires a completed failed Phase-B attempt record"
        )
    return {
        "status": "PASS_CONTROLLED_PHASE_B_RESUME",
        "allowed_concurrent_pretract_processes": allowed,
        "unexpected_imaging_processes": [],
        "allowed_pretract_roots": list(roots),
        "prior_failed_attempts": failed_attempts,
    }


def write_resolved_config(
    human_qc: Path,
    units: list[str],
    *,
    execution_authorized: bool,
    pretract_manifest: Path,
    promotion_receipt: Path | None,
) -> Path:
    config = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise TypeError("base resolved config is not a mapping")
    config["human_qc_manifest"] = str(human_qc.resolve())
    config["launcher_mode"] = "phase-b"
    config["hcp379_pretract_manifest"] = str(
        pretract_manifest.resolve()
    )
    config["hcp379_promotion_receipt"] = (
        str(promotion_receipt.resolve())
        if promotion_receipt is not None
        else None
    )
    execution_binding = config.get("execution_binding")
    if not isinstance(execution_binding, dict):
        raise ValueError("base config lacks the canary execution binding")
    continuation = {
        "schema_version": "2.0.0",
        "binding_type": "tractography_continuation_binding",
        "execution_scope": "canary",
        "recipe_id": config["contract"]["recipe_id"],
        "execution_subset_manifest": execution_binding[
            "execution_subset_manifest"
        ],
        "approved_unit_count": len(units),
        "approved_units": list(units),
        "diagnosis_labels_used": False,
        "recovery4_pretract_manifest": file_record(pretract_manifest),
        "promotion_receipt": (
            file_record(promotion_receipt)
            if promotion_receipt is not None
            else None
        ),
        "human_qc_manifest": file_record(human_qc),
        "execution_authorized": execution_authorized,
        "authorization_scope": (
            "human_reviewed_phase_b"
            if execution_authorized
            else "pre_human_dry_run_only_no_execution"
        ),
    }
    config["tractography_continuation_binding"] = continuation
    suffix = (
        "resolved_config.yaml"
        if execution_authorized
        else "pre_human_dryrun_config.yaml"
    )
    source_suffix = (
        "promoted_v6_"
        if promotion_receipt is not None
        else "original_v6gate_"
    )
    output = (
        PHASE_ROOT
        / f"hcp379_phase_b_recovery4_{source_suffix}{suffix}"
    )
    payload = yaml.safe_dump(config, sort_keys=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        if output.read_text(encoding="utf-8") != payload:
            raise FileExistsError(
                "resolved HCP379 config exists with different content"
            )
        return output
    with tempfile.NamedTemporaryFile(
        mode="w",
        dir=output.parent,
        prefix=f".{output.name}.",
        suffix=".tmp",
        encoding="utf-8",
        delete=False,
    ) as handle:
        handle.write(payload)
        temporary = Path(handle.name)
    os.replace(temporary, output)
    return output


def command(config: Path, cores: int, *, dry_run: bool) -> list[str]:
    value = [
        str(SNAKEMAKE),
        "--snakefile",
        str(SNAKEFILE),
        "--configfile",
        str(config),
        "--cores",
        str(cores),
        "--rerun-incomplete",
        "--rerun-triggers",
        "input",
        "params",
        "--keep-going",
        "--printshellcmds",
        "--show-failed-logs",
    ]
    if dry_run:
        value.append("--dry-run")
    value.append("hcp379_stability_evidence")
    return value


def scheduled_rules(text: str) -> set[str]:
    rules = set()
    in_stats = False
    for line in text.splitlines():
        if line.strip() == "Job stats:":
            in_stats = True
            continue
        if not in_stats:
            continue
        match = re.match(r"^([A-Za-z0-9_]+)\s+(\d+)\s*$", line.strip())
        if match and match.group(1) != "total":
            rules.add(match.group(1))
    return rules


def validate_track_temporary_suffixes() -> dict[str, Any]:
    """Require every MRtrix track temporary to retain a final .tck suffix."""

    text = EXTENSION.read_text(encoding="utf-8")
    assignments = re.findall(
        r"^\s*partial=\{output\.tracks:q\}(\S+)\s*$",
        text,
        flags=re.MULTILINE,
    )
    expected = [".partial.tck", ".partial.tck", ".partial.tck"]
    if assignments != expected:
        raise ValueError(
            "Phase-B track temporary suffix contract differs: "
            f"{assignments}"
        )
    if "tracks.tck.partial\"" in text or ".tck.partial\n" in text:
        raise ValueError("legacy MRtrix-incompatible .tck.partial remains")
    if "tckinfo" in text or "wc -l" in text:
        raise ValueError(
            "unsafe MRtrix post-write or text line counter remains"
        )
    for helper in (
        "tck_header_count_v1.py",
        "count_mrtrix_numeric_values_v1.py",
    ):
        if helper not in text:
            raise ValueError(f"required bounded counter is absent: {helper}")
    return {
        "status": "PASS",
        "track_temporary_assignments": assignments,
        "mrtrix_recognized_final_suffix": ".tck",
        "tck_count_method": "bounded ASCII header parser",
        "numeric_text_count_method": (
            "finite numeric tokens excluding MRtrix comment headers"
        ),
        "assignment_count_method": (
            "comment-aware whitespace endpoint-pair parser"
        ),
    }


def run_live(
    live_command: list[str],
    *,
    log_path: Path,
) -> tuple[int, float, int]:
    started = time.monotonic()
    minimum_free = shutil.disk_usage(RUN_ROOT).free
    with log_path.open("x", encoding="utf-8") as log:
        process = subprocess.Popen(
            live_command,
            cwd=EXP,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        while process.poll() is None:
            free = shutil.disk_usage(RUN_ROOT).free
            minimum_free = min(minimum_free, free)
            if free < EMERGENCY_FREE_BYTES:
                log.write(
                    f"\n[{utc_now()}] EMERGENCY_STOP free_bytes={free}\n"
                )
                log.flush()
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                return 75, time.monotonic() - started, minimum_free
            time.sleep(30)
        return process.returncode, time.monotonic() - started, minimum_free


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--human-qc-manifest", type=Path, default=DEFAULT_HUMAN_QC
    )
    parser.add_argument("--cores", type=int, default=64)
    parser.add_argument("--dry-run-only", action="store_true")
    parser.add_argument(
        "--resume-existing-phase-b",
        action="store_true",
        help=(
            "Resume a failed Phase-B attempt while allowing only imaging "
            "processes rooted in the six known corrected Recovery4 pretract "
            "lanes."
        ),
    )
    parser.add_argument(
        "--pre-human-dry-run",
        action="store_true",
        help=(
            "validate DAG wiring against the blank review template only; "
            "never starts execution"
        ),
    )
    args = parser.parse_args()
    if not 1 <= args.cores <= os.cpu_count():
        raise ValueError(f"--cores must be in 1..{os.cpu_count()}")
    sources = active_sources()
    pretract_manifest = Path(sources["pretract_manifest"]).resolve()
    review_manifest = Path(sources["review_manifest"]).resolve()
    promotion_receipt = (
        Path(sources["promotion_receipt"]).resolve()
        if sources.get("promotion_receipt") is not None
        else None
    )
    for path in (
        SNAKEFILE,
        EXTENSION,
        TCK_HEADER_COUNT,
        NUMERIC_VALUE_COUNT,
        BUILDER,
        VALIDATOR,
        BASE_CONFIG,
        pretract_manifest,
        review_manifest,
        CORRECTED_OVERLAY_QUANTITATIVE_AUDIT,
        SNAKEMAKE,
        Path(__file__),
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    track_temporary_suffix_contract = (
        validate_track_temporary_suffixes()
    )
    existing = (
        load_json(STABILITY_OUTPUT) if STABILITY_OUTPUT.is_file() else None
    )
    if existing and existing.get("status") == "PASS":
        print(
            json.dumps(
                {
                    "status": "ALREADY_PASS",
                    "output": str(STABILITY_OUTPUT),
                    "smallest_density_qualified_scale_up_streamline_count": (
                        existing.get(
                            "smallest_density_qualified_scale_up_streamline_count"
                        )
                    ),
                },
                sort_keys=True,
            )
        )
        return 0
    active = active_imaging_processes()
    resume_contract = None
    if args.resume_existing_phase_b:
        resume_contract = resume_process_contract(active)
    elif active:
        raise RuntimeError(
            "another imaging process is active: " + " | ".join(active[:5])
        )
    if shutil.disk_usage(RUN_ROOT).free < MINIMUM_FREE_BYTES_TO_START:
        raise RuntimeError("free storage is below the HCP379 Phase-B start floor")
    units = units_from_pretract(pretract_manifest)
    upstream = validate_upstream(units, sources=sources)
    if args.pre_human_dry_run:
        review = load_json(review_manifest)
        template_record = (
            review.get("human_qc_template")
            if sources["mode"] == "V6_HUMAN_REVIEWED_PROMOTION"
            else review.get("blank_human_qc_template")
        )
        human_path = verified_file_record(
            template_record
        )
        human_record = {
            "status": "NOT_REVIEWED_BLANK_TEMPLATE",
            "template": file_record(human_path),
        }
        execution_authorized = False
    else:
        if upstream["corrected_overlay_hard_hold_units"]:
            raise ValueError(
                "corrected-overlay quantitative hard holds remain: "
                + ",".join(
                    str(unit)
                    for unit in upstream[
                        "corrected_overlay_hard_hold_units"
                    ]
                )
            )
        human_path = args.human_qc_manifest
        human_record = validate_human_qc(human_path, units)
        execution_authorized = True
    resolved = write_resolved_config(
        human_path,
        units,
        execution_authorized=execution_authorized,
        pretract_manifest=pretract_manifest,
        promotion_receipt=promotion_receipt,
    )
    PHASE_ROOT.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    attempt_id = (
        f"{timestamp}-hcp379-recovery4-phase-b-{os.urandom(4).hex()}"
    )
    dry_log = PHASE_ROOT / f"{attempt_id}.dryrun.log"
    live_log = PHASE_ROOT / f"{attempt_id}.snakemake.log"
    attempt_path = PHASE_ROOT / f"{attempt_id}.attempt.json"
    dry_command = command(resolved, args.cores, dry_run=True)
    dry = subprocess.run(
        dry_command,
        cwd=EXP,
        check=False,
        capture_output=True,
        text=True,
    )
    dry_text = (dry.stdout or "") + (
        "\n" + dry.stderr if dry.stderr else ""
    )
    dry_log.write_text(dry_text, encoding="utf-8")
    scheduled = scheduled_rules(dry_text)
    forbidden = sorted(scheduled & FORBIDDEN_SCHEDULED_RULES)
    if dry.returncode != 0 or forbidden:
        raise RuntimeError(
            f"HCP379 Phase-B dry-run failed: rc={dry.returncode}, "
            f"forbidden={forbidden}; see {dry_log}"
        )
    attempt = {
        "schema_version": "1.0.0",
        "record_type": "diagnosis_blind_hcp379_phase_b_attempt",
        "status": (
            "PRE_HUMAN_DRY_RUN_PASS"
            if args.pre_human_dry_run
            else "DRY_RUN_PASS"
        ),
        "attempt_id": attempt_id,
        "started_utc": utc_now(),
        "diagnosis_labels_used": False,
        "cores": args.cores,
        "minimum_start_free_bytes": MINIMUM_FREE_BYTES_TO_START,
        "emergency_free_bytes": EMERGENCY_FREE_BYTES,
        "upstream": upstream,
        "human_visual_qc": human_record,
        "sources": {
            "launcher": file_record(Path(__file__)),
            "snakefile": file_record(SNAKEFILE),
            "extension": file_record(EXTENSION),
            "tck_header_count": file_record(TCK_HEADER_COUNT),
            "numeric_value_count": file_record(NUMERIC_VALUE_COUNT),
            "builder": file_record(BUILDER),
            "validator": file_record(VALIDATOR),
            "resolved_config": file_record(resolved),
        },
        "dry_run_log": file_record(dry_log),
        "scheduled_rules": sorted(scheduled),
        "forbidden_scheduled_rules": forbidden,
        "track_temporary_suffix_contract": (
            track_temporary_suffix_contract
        ),
        "resume_existing_phase_b": args.resume_existing_phase_b,
        "controlled_resume_contract": resume_contract,
    }
    atomic_json(attempt_path, attempt)
    if args.dry_run_only or args.pre_human_dry_run:
        print(
            json.dumps(
                {
                    "status": (
                        "PRE_HUMAN_DRY_RUN_PASS"
                        if args.pre_human_dry_run
                        else "DRY_RUN_PASS"
                    ),
                    "scheduled_rules": len(scheduled),
                    "attempt": str(attempt_path),
                },
                sort_keys=True,
            )
        )
        return 0
    live_command = command(resolved, args.cores, dry_run=False)
    returncode, elapsed, minimum_free = run_live(
        live_command, log_path=live_log
    )
    validation = (
        load_json(STABILITY_OUTPUT) if STABILITY_OUTPUT.is_file() else None
    )
    attempt.update(
        {
            "status": (
                "PASS"
                if returncode == 0
                and validation
                and validation.get("status") == "PASS"
                else "FAIL"
            ),
            "completed_utc": utc_now(),
            "returncode": returncode,
            "elapsed_seconds": round(elapsed, 3),
            "minimum_observed_free_bytes": minimum_free,
            "execution_log": file_record(live_log),
            "stability_validation": (
                file_record(STABILITY_OUTPUT)
                if STABILITY_OUTPUT.is_file()
                else None
            ),
        }
    )
    atomic_json(attempt_path, attempt)
    print(
        json.dumps(
            {
                "status": attempt["status"],
                "attempt": str(attempt_path),
                "validation": (
                    str(STABILITY_OUTPUT)
                    if STABILITY_OUTPUT.is_file()
                    else None
                ),
                "smallest_density_qualified_scale_up_streamline_count": (
                    validation.get(
                        "smallest_density_qualified_scale_up_streamline_count"
                    )
                    if validation
                    else None
                ),
            },
            sort_keys=True,
        )
    )
    return 0 if attempt["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
