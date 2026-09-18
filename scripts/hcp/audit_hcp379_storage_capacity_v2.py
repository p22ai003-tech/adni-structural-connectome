#!/usr/bin/env python3
"""Forecast exact 530-subject HCP379-v2 recovery storage without mutation.

The forecast is diagnosis blind.  It measures the exact 15-unit Recovery3
working set and the completed 86-subject HCP379 3M tractograms, then projects
the remaining pre-tractography and selected-count tractography high-water
marks.  Both the selective archive-D0 retention case and the conservative
all-corrected archive case are projected from the frozen execution topology.
The report is operational evidence only; it never deletes data or changes an
EBS volume.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


EXP = Path("/home/ec2-user/exp")
HCP_ROOT = Path("/data/derivatives/hcp379_v2")
RUN_ROOT = Path(
    "/data/derivatives/scforge_v2/h04a_r1_recovery_20260719_retry4"
)
BINDING = (
    EXP
    / "research_audit/outputs/"
    "h04a_r1_retry4_pretract_recovery3_package_v1/"
    "recovery3_execution_binding.json"
)
ARCHIVE_SUMMARY = HCP_ROOT / "manifests/archive_restore_summary.json"
RECOVERY4_MASTER = (
    HCP_ROOT / "pretract_recovery4/pretract_recovery4_manifest.json"
)
COMPACTOR_VALIDATION = (
    EXP
    / "research_audit/outputs/"
    "hcp379_pretract_compaction_recovery4_v3/validation.json"
)
TRACT_VALIDATION = (
    EXP
    / "research_audit/outputs/"
    "hcp379_scaleup_tractography_recovery4_v3/validation.json"
)
DEFAULT_OUTPUT = HCP_ROOT / "manifests/storage_capacity_forecast.json"
EXPECTED_CANARY_N = 15
EXPECTED_ARCHIVE_N = 86
CORE_NONCANARY_PRETRACT_N = 216
LEGACY_NONCANARY_PRETRACT_N = 212
ARCHIVE_CALIBRATION_N = 6
ARCHIVE_LOW_NEW_N = 28
FREESURFER_NEW_N = 1
ARCHIVE_D0_FAIL_NEW_N = 52
SELECTIVE_ARCHIVE_NEW_PROCESSING_N = (
    CORE_NONCANARY_PRETRACT_N
    + LEGACY_NONCANARY_PRETRACT_N
    + ARCHIVE_CALIBRATION_N
    + ARCHIVE_LOW_NEW_N
    + FREESURFER_NEW_N
)
FULL_CORRECTED_NEW_PROCESSING_N = (
    SELECTIVE_ARCHIVE_NEW_PROCESSING_N + ARCHIVE_D0_FAIL_NEW_N
)
PHASE_B_N = 15
STREAMLINE_LEVELS = (3_000_000, 5_000_000, 10_000_000)
REFERENCE_STREAMLINES = 3_000_000
PHASE_B_REFERENCE_MULTIPLIER = 2.0 * (
    sum(STREAMLINE_LEVELS) / REFERENCE_STREAMLINES
)
TRACK_SIDECAR_OVERHEAD = 1.10
MISCELLANEOUS_RESERVE_BYTES = 50_000_000_000
SAFETY_FREE_BYTES = 500_000_000_000
CURRENT_EBS_GIB = 7000
STREAMING_PRETRACT_WORKERS = 4
STREAMING_TRACT_WORKERS = 3
STREAMING_UNCLASSIFIED_OVERHEAD_PER_SUBJECT_BYTES = 300_000_000
COMPACTED_TRACT_SIDECAR_FRACTION = 0.15


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root is not an object: {path}")
    return value


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


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path = path.expanduser().resolve()
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


def directory_size(path: Path) -> int:
    total = 0
    for root, _, files in os.walk(path):
        root_path = Path(root)
        for name in files:
            candidate = root_path / name
            if candidate.is_symlink():
                continue
            total += candidate.stat().st_size
    return total


def quantile(values: Iterable[int], probability: float) -> float:
    ordered = sorted(int(value) for value in values)
    if not ordered:
        raise ValueError("quantile requires at least one value")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("quantile probability is outside [0,1]")
    position = (len(ordered) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return float(
        ordered[lower] * (1.0 - fraction)
        + ordered[upper] * fraction
    )


def descriptive(values: list[int]) -> dict[str, Any]:
    return {
        "n": len(values),
        "minimum_bytes": min(values),
        "median_bytes": round(quantile(values, 0.50)),
        "p75_bytes": round(quantile(values, 0.75)),
        "p90_bytes": round(quantile(values, 0.90)),
        "maximum_bytes": max(values),
        "mean_bytes": round(sum(values) / len(values)),
        "total_bytes": sum(values),
    }


def round_up_ebs_gib(bytes_required: int) -> int:
    if bytes_required <= 0:
        return CURRENT_EBS_GIB
    extra_gib = math.ceil(bytes_required / (1024**3))
    target = CURRENT_EBS_GIB + extra_gib
    return min(16_384, int(math.ceil(target / 100.0) * 100))


def forecast(
    *,
    free_bytes: int,
    pretract_per_subject: int,
    track_3m_per_subject: int,
    streamline_count: int,
    processing_subject_n: int,
) -> dict[str, Any]:
    pretract = pretract_per_subject * processing_subject_n
    phase_b = round(
        track_3m_per_subject
        * PHASE_B_REFERENCE_MULTIPLIER
        * PHASE_B_N
        * TRACK_SIDECAR_OVERHEAD
    )
    scaleup_tracks = round(
        track_3m_per_subject
        * (streamline_count / REFERENCE_STREAMLINES)
        * processing_subject_n
        * TRACK_SIDECAR_OVERHEAD
    )
    retain_all = (
        pretract
        + phase_b
        + scaleup_tracks
        + MISCELLANEOUS_RESERVE_BYTES
    )
    prune_phase_b = (
        pretract
        + max(phase_b, scaleup_tracks)
        + MISCELLANEOUS_RESERVE_BYTES
    )
    required_with_reserve = prune_phase_b + SAFETY_FREE_BYTES
    shortfall = max(0, required_with_reserve - free_bytes)
    return {
        "selected_streamline_count": streamline_count,
        "new_processing_subject_n": processing_subject_n,
        "projected_pretract_bytes": pretract,
        "projected_phase_b_bytes": phase_b,
        "projected_full_scaleup_track_bytes": scaleup_tracks,
        "retain_all_projected_additional_bytes": retain_all,
        "prune_phase_b_projected_additional_bytes": prune_phase_b,
        "safety_free_bytes": SAFETY_FREE_BYTES,
        "free_bytes_after_pruned_schedule": free_bytes - prune_phase_b,
        "safe_with_current_volume_after_phase_b_pruning": (
            free_bytes >= required_with_reserve
        ),
        "additional_bytes_required_with_reserve": shortfall,
        "minimum_rounded_target_ebs_gib": round_up_ebs_gib(shortfall),
    }


def artifact_records(value: Any) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if isinstance(value, Mapping):
        if {
            "path",
            "size_bytes",
            "sha256",
        }.issubset(value):
            records.append(dict(value))
        else:
            for child in value.values():
                records.extend(artifact_records(child))
    elif isinstance(value, list):
        for child in value:
            records.extend(artifact_records(child))
    return records


def recovery4_retained_sizes() -> list[int]:
    master = load_json(RECOVERY4_MASTER)
    units = master.get("units")
    if (
        master.get("status")
        != "AUTOMATED_PASS_AWAITING_HUMAN_VISUAL_QC"
        or master.get("diagnosis_labels_used") is not False
        or not isinstance(units, list)
        or len(units) != EXPECTED_CANARY_N
    ):
        raise ValueError("Recovery4 master is not exact automated 15 PASS")
    sizes = []
    for row in units:
        if not isinstance(row, Mapping):
            raise TypeError("Recovery4 master unit is not a mapping")
        manifest_record = row.get("manifest")
        if not isinstance(manifest_record, Mapping):
            raise TypeError("Recovery4 unit manifest binding is absent")
        manifest_path = Path(str(manifest_record.get("path", "")))
        if file_record(manifest_path) != dict(manifest_record):
            raise ValueError("Recovery4 unit manifest binding differs")
        manifest = load_json(manifest_path)
        tract = manifest.get("tractography_inputs")
        if not isinstance(tract, Mapping):
            raise TypeError("Recovery4 tractography inputs differ")
        retained_tract = {
            name: tract[name]
            for name in ("five_tt", "gmwmi", "wmfod_normalised")
        }
        sections = {
            "tractography_inputs": retained_tract,
            "tensor_maps_bounded": manifest.get("tensor_maps_bounded"),
            "tensor_maps_raw_preserved": manifest.get(
                "tensor_maps_raw_preserved"
            ),
            "hcp379": manifest.get("hcp379"),
            "registration": manifest.get("registration"),
            "automated_qc": manifest.get("automated_qc"),
            "provenance": manifest.get("provenance"),
        }
        unique: dict[str, int] = {}
        for record in artifact_records(sections):
            path = Path(str(record.get("path", ""))).resolve()
            # External source parcellations already exist and are not copied
            # into the scale-up root.
            if path.is_relative_to(Path("/data/derivatives/parc_hcpmmp1")):
                continue
            observed = file_record(path)
            if observed != record:
                raise ValueError(f"Recovery4 retained binding differs: {path}")
            unique[str(path)] = int(record["size_bytes"])
        sizes.append(sum(unique.values()))
    return sizes


def streaming_forecast(
    *,
    free_bytes: int,
    retained_pretract_per_subject: int,
    active_pretract_per_subject: int,
    track_3m_per_subject: int,
    streamline_count: int,
    phase_b_bytes: int,
    processing_subject_n: int,
) -> dict[str, Any]:
    compacted_pretract = (
        retained_pretract_per_subject * processing_subject_n
    )
    active_pretract_high_water = (
        active_pretract_per_subject * STREAMING_PRETRACT_WORKERS
    )
    scaled_track = round(
        track_3m_per_subject
        * (streamline_count / REFERENCE_STREAMLINES)
    )
    retained_track_sidecars = round(
        scaled_track
        * COMPACTED_TRACT_SIDECAR_FRACTION
        * processing_subject_n
    )
    active_track_high_water = round(
        scaled_track
        * TRACK_SIDECAR_OVERHEAD
        * STREAMING_TRACT_WORKERS
    )
    archive_replicate_tracks = round(
        scaled_track
        * TRACK_SIDECAR_OVERHEAD
        * ARCHIVE_CALIBRATION_N
    )
    projected = (
        compacted_pretract
        + active_pretract_high_water
        + phase_b_bytes
        + retained_track_sidecars
        + active_track_high_water
        + archive_replicate_tracks
        + MISCELLANEOUS_RESERVE_BYTES
    )
    required = projected + SAFETY_FREE_BYTES
    shortfall = max(0, required - free_bytes)
    return {
        "selected_streamline_count": streamline_count,
        "new_processing_subject_n": processing_subject_n,
        "projected_compacted_pretract_bytes": compacted_pretract,
        "projected_active_pretract_high_water_bytes": (
            active_pretract_high_water
        ),
        "projected_phase_b_retained_bytes": phase_b_bytes,
        "projected_compacted_track_sidecars_bytes": (
            retained_track_sidecars
        ),
        "projected_active_track_high_water_bytes": active_track_high_water,
        "projected_archive_replicate_tracks_bytes": (
            archive_replicate_tracks
        ),
        "miscellaneous_reserve_bytes": MISCELLANEOUS_RESERVE_BYTES,
        "projected_additional_bytes": projected,
        "safety_free_bytes": SAFETY_FREE_BYTES,
        "projected_free_bytes_after_schedule": free_bytes - projected,
        "safe_with_current_volume_and_reserve": free_bytes >= required,
        "additional_bytes_required_with_reserve": shortfall,
        "minimum_rounded_target_ebs_gib": round_up_ebs_gib(shortfall),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    binding = load_json(BINDING)
    units = binding.get("units")
    if (
        binding.get("record_type")
        != "retry4_pretract_recovery3_execution_binding"
        or binding.get("diagnosis_labels_used") is not False
        or not isinstance(units, list)
        or len(units) != EXPECTED_CANARY_N
        or len(set(str(unit) for unit in units)) != EXPECTED_CANARY_N
    ):
        raise ValueError("Recovery3 binding identity differs")
    subject_sizes = []
    for unit in sorted(str(unit) for unit in units):
        subject = RUN_ROOT / "subjects" / unit
        if not subject.is_dir() or subject.is_symlink():
            raise FileNotFoundError(subject)
        subject_sizes.append(directory_size(subject))

    archive = load_json(ARCHIVE_SUMMARY)
    rows = archive.get("subjects")
    if (
        archive.get("status_counts") != {"PASS_ALL_NINE": EXPECTED_ARCHIVE_N}
        or not isinstance(rows, list)
        or len(rows) != EXPECTED_ARCHIVE_N
    ):
        raise ValueError("archive track-size reference is not 86/86 PASS")
    track_sizes = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise TypeError("archive row is not a mapping")
        path = Path(str(row.get("track_path", ""))).resolve()
        expected = int(row.get("track_size_bytes", -1))
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != expected
            or expected <= 0
        ):
            raise ValueError(
                f"archive track size binding differs for {row.get('fsid')}"
            )
        track_sizes.append(expected)

    usage = shutil.disk_usage(HCP_ROOT)
    pretract_stats = descriptive(subject_sizes)
    track_stats = descriptive(track_sizes)
    retained_pretract_stats = descriptive(recovery4_retained_sizes())
    compactor_validation = load_json(COMPACTOR_VALIDATION)
    tract_validation = load_json(TRACT_VALIDATION)
    compaction_validated = (
        compactor_validation.get("status") == "PASS"
        and compactor_validation.get("passed_checks") == 11
        and compactor_validation.get("total_checks") == 11
        and compactor_validation.get(
            "deletion_executed_on_real_subjects"
        )
        is False
        and tract_validation.get("status") == "PASS"
        and tract_validation.get("passed_checks") == 16
        and tract_validation.get("total_checks") == 16
    )
    central = {
        str(count): forecast(
            free_bytes=usage.free,
            pretract_per_subject=pretract_stats["p75_bytes"],
            track_3m_per_subject=track_stats["p75_bytes"],
            streamline_count=count,
            processing_subject_n=FULL_CORRECTED_NEW_PROCESSING_N,
        )
        for count in STREAMLINE_LEVELS
    }
    conservative = {
        str(count): forecast(
            free_bytes=usage.free,
            pretract_per_subject=pretract_stats["maximum_bytes"],
            track_3m_per_subject=track_stats["p90_bytes"],
            streamline_count=count,
            processing_subject_n=FULL_CORRECTED_NEW_PROCESSING_N,
        )
        for count in STREAMLINE_LEVELS
    }
    phase_b_only = round(
        track_stats["p75_bytes"]
        * PHASE_B_REFERENCE_MULTIPLIER
        * PHASE_B_N
        * TRACK_SIDECAR_OVERHEAD
        + MISCELLANEOUS_RESERVE_BYTES
    )
    conservative_phase_b = round(
        track_stats["p90_bytes"]
        * PHASE_B_REFERENCE_MULTIPLIER
        * PHASE_B_N
        * TRACK_SIDECAR_OVERHEAD
    )
    streaming_retained_per_subject = (
        retained_pretract_stats["p90_bytes"]
        + STREAMING_UNCLASSIFIED_OVERHEAD_PER_SUBJECT_BYTES
    )
    streaming = {
        str(count): streaming_forecast(
            free_bytes=usage.free,
            retained_pretract_per_subject=(
                streaming_retained_per_subject
            ),
            active_pretract_per_subject=pretract_stats["maximum_bytes"],
            track_3m_per_subject=track_stats["p90_bytes"],
            streamline_count=count,
            phase_b_bytes=conservative_phase_b,
            processing_subject_n=FULL_CORRECTED_NEW_PROCESSING_N,
        )
        for count in STREAMLINE_LEVELS
    }
    selective_streaming = {
        str(count): streaming_forecast(
            free_bytes=usage.free,
            retained_pretract_per_subject=(
                streaming_retained_per_subject
            ),
            active_pretract_per_subject=pretract_stats["maximum_bytes"],
            track_3m_per_subject=track_stats["p90_bytes"],
            streamline_count=count,
            phase_b_bytes=conservative_phase_b,
            processing_subject_n=SELECTIVE_ARCHIVE_NEW_PROCESSING_N,
        )
        for count in STREAMLINE_LEVELS
    }
    streaming_safe_counts = [
        int(count)
        for count, item in streaming.items()
        if item["safe_with_current_volume_and_reserve"]
    ]
    selective_streaming_safe_counts = [
        int(count)
        for count, item in selective_streaming.items()
        if item["safe_with_current_volume_and_reserve"]
    ]
    pretract_only = (
        pretract_stats["p75_bytes"]
        * FULL_CORRECTED_NEW_PROCESSING_N
        + MISCELLANEOUS_RESERVE_BYTES
    )
    safe_counts = [
        int(count)
        for count, item in central.items()
        if item["safe_with_current_volume_after_phase_b_pruning"]
    ]
    report = {
        "schema_version": "2.0.0",
        "record_type": (
            "diagnosis_blind_hcp379_storage_capacity_forecast"
        ),
        "generated_utc": utc_now(),
        "status": (
            "SAFE_WITH_VALIDATED_STREAMING_COMPACTION"
            if compaction_validated
            and len(streaming_safe_counts) == len(STREAMLINE_LEVELS)
            else "SAFE_FOR_ALL_STREAMLINE_LEVELS"
            if len(safe_counts) == len(STREAMLINE_LEVELS)
            else "CAPACITY_ACTION_REQUIRED_BEFORE_FULL_SCALEUP"
        ),
        "diagnosis_labels_used": False,
        "deletes_or_volume_changes_performed": False,
        "volume": {
            "total_bytes": usage.total,
            "used_bytes": usage.used,
            "free_bytes": usage.free,
            "current_ebs_gib": CURRENT_EBS_GIB,
            "safety_free_bytes": SAFETY_FREE_BYTES,
        },
        "measurements": {
            "recovery3_subject_working_set": pretract_stats,
            "archive_hcp379_3m_tractograms": track_stats,
            "recovery4_retained_scientific_artifacts": (
                retained_pretract_stats
            ),
        },
        "assumptions": {
            "core_noncanary_pretract_n": (
                CORE_NONCANARY_PRETRACT_N
            ),
            "legacy_noncanary_pretract_n": (
                LEGACY_NONCANARY_PRETRACT_N
            ),
            "archive_calibration_pretract_n": ARCHIVE_CALIBRATION_N,
            "archive_low_new_n": ARCHIVE_LOW_NEW_N,
            "freesurfer_new_n": FREESURFER_NEW_N,
            "conditional_archive_D0_additional_n": (
                ARCHIVE_D0_FAIL_NEW_N
            ),
            "selective_archive_new_processing_n": (
                SELECTIVE_ARCHIVE_NEW_PROCESSING_N
            ),
            "full_corrected_new_processing_n": (
                FULL_CORRECTED_NEW_PROCESSING_N
            ),
            "phase_b_n": PHASE_B_N,
            "phase_b_relative_3m_track_multiplier_per_subject": (
                PHASE_B_REFERENCE_MULTIPLIER
            ),
            "track_sidecar_overhead_multiplier": TRACK_SIDECAR_OVERHEAD,
            "miscellaneous_reserve_bytes": MISCELLANEOUS_RESERVE_BYTES,
            "central_pretract_estimator": "Recovery3 subject-directory p75",
            "central_track_estimator": "archive HCP379 3M track-size p75",
            "conservative_pretract_estimator": (
                "Recovery3 subject-directory maximum"
            ),
            "conservative_track_estimator": (
                "archive HCP379 3M track-size p90"
            ),
            "phase_b_pruning_is_implemented": False,
            "streaming_pretract_compaction_validated": (
                compaction_validated
            ),
            "streaming_tractogram_compaction_supported": True,
            "streaming_pretract_workers": STREAMING_PRETRACT_WORKERS,
            "streaming_tract_workers": STREAMING_TRACT_WORKERS,
            "streaming_unclassified_overhead_per_subject_bytes": (
                STREAMING_UNCLASSIFIED_OVERHEAD_PER_SUBJECT_BYTES
            ),
            "compacted_tract_sidecar_fraction": (
                COMPACTED_TRACT_SIDECAR_FRACTION
            ),
            "streaming_retained_estimator": (
                "Recovery4 retained-artifact p90 plus fixed "
                "unclassified-artifact overhead"
            ),
            "streaming_real_subject_compaction_empirically_measured": False,
        },
        "phase_b_only": {
            "projected_additional_bytes": phase_b_only,
            "safe_with_current_volume_and_reserve": (
                usage.free >= phase_b_only + SAFETY_FREE_BYTES
            ),
        },
        "pretract_worst_case_515_only": {
            "projected_additional_bytes": pretract_only,
            "safe_with_current_volume_and_reserve": (
                usage.free >= pretract_only + SAFETY_FREE_BYTES
            ),
        },
        "central_scenarios": central,
        "conservative_scenarios": conservative,
        "central_safe_streamline_counts": safe_counts,
        "streaming_compaction_scenarios": streaming,
        "streaming_safe_streamline_counts": streaming_safe_counts,
        "selective_archive_streaming_compaction_scenarios": (
            selective_streaming
        ),
        "selective_archive_streaming_safe_streamline_counts": (
            selective_streaming_safe_counts
        ),
        "workload_scenarios": {
            "archive_D0_concordance_pass": {
                "new_processing_subject_n": (
                    SELECTIVE_ARCHIVE_NEW_PROCESSING_N
                ),
                "final_corrected_ACT_n": 474,
                "retained_archive_no_ACT_n": 56,
                "final_total_n": 530,
            },
            "archive_D0_concordance_fail": {
                "new_processing_subject_n": (
                    FULL_CORRECTED_NEW_PROCESSING_N
                ),
                "final_corrected_ACT_n": 530,
                "retained_archive_no_ACT_n": 0,
                "final_total_n": 530,
            },
        },
        "sources": {
            "recovery3_binding": file_record(BINDING),
            "archive_summary": file_record(ARCHIVE_SUMMARY),
            "recovery4_master": file_record(RECOVERY4_MASTER),
            "compactor_validation": file_record(COMPACTOR_VALIDATION),
            "tractography_validation": file_record(TRACT_VALIDATION),
            "script": file_record(Path(__file__)),
        },
        "decision": (
            "The current volume is assessed against the conservative 515-new-"
            "processing-subject case, which covers a failed archive-D0 "
            "concordance decision and an all-corrected 530-subject release. "
            "It is storage-safe only when "
            "validated per-subject pretract and tractogram compaction are "
            "enabled and the 500 GB reserve is enforced. The first real "
            "compacted subject remains an empirical stop/check before broad "
            "continuation. Phase B's process-conflict guard and genuine "
            "human-QC gate still apply. Never use density or diagnosis to "
            "choose storage routing."
        ),
    }
    atomic_json(args.output, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "free_bytes": usage.free,
                "phase_b_safe": report["phase_b_only"][
                    "safe_with_current_volume_and_reserve"
                ],
                "pretract_worst_case_515_safe": report[
                    "pretract_worst_case_515_only"
                ][
                    "safe_with_current_volume_and_reserve"
                ],
                "central_safe_streamline_counts": safe_counts,
                "streaming_safe_streamline_counts": (
                    streaming_safe_counts
                ),
                "streaming_compaction_validated": compaction_validated,
                "worst_case_new_processing_subject_n": (
                    FULL_CORRECTED_NEW_PROCESSING_N
                ),
                "output": str(args.output.resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
