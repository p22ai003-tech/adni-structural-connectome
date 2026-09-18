"""Versioned terminal publication for the retry4 pretract hotfix recovery."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


LEDGER_NAME = "pre_tractography_canary_recovery1_ledger.jsonl"
MANIFEST_NAME = "pre_tractography_canary_recovery1_manifest.json"


def finalize_recovery1(
    *,
    base: Any,
    run_root: Path,
    execution_rows: list[dict[str, str]],
    execution_binding: dict[str, Any],
    run_id: str,
    lineage_id: str,
    recipe_id: str,
    run_context_path: Path,
    attempt_start_path: Path,
    attempt_end_path: Path,
    execution_log_path: Path,
    workflow_returncode: int,
    ended_utc: str,
    prior_failure_completion: dict[str, Any],
) -> dict[str, Any]:
    """Publish the recovery without overwriting the immutable failed release."""

    expected_units = tuple(row["unit"] for row in execution_rows)
    if list(expected_units) != execution_binding.get("units"):
        raise ValueError("recovery execution rows differ from bound canary units")
    expected_set = set(expected_units)
    row_by_unit = {row["unit"]: row for row in execution_rows}
    log_record = base.file_record(execution_log_path)
    failed_rules = base._failed_rule_evidence(
        execution_log_path.read_text(encoding="utf-8", errors="replace"),
        expected_set,
    )

    entries: list[dict[str, Any]] = []
    review_bundle_records: dict[str, dict[str, Any]] = {}
    automated_qc_records: dict[str, dict[str, Any]] = {}
    for unit in expected_units:
        review_dir = run_root / "subjects" / unit / "06_preflight"
        automated_qc_path = review_dir / "automated_pre_tractography_qc.json"
        review_paths = {
            "b0_t1": review_dir / "review_b0_vs_t1.png",
            "b0_5tt": review_dir / "review_b0_vs_5tt.png",
            "b0_atlas": review_dir / "review_b0_vs_aal3.png",
            "index": review_dir / "visual_review_index.csv",
        }
        present = base._existing_file_records(review_paths)
        automated_qc_record = (
            base.file_record(automated_qc_path)
            if automated_qc_path.is_file()
            else None
        )
        local_stage: str | None = None
        local_reason: str | None = None
        local_flags: list[str] = []
        automated_qc_pass = False
        if automated_qc_record is None:
            local_stage = "fod_tensor"
            local_reason = "automated_pre_tractography_qc_missing"
            local_flags = ["automated_qc_not_released_for_human_review"]
        else:
            try:
                automated_qc = base._load_json_mapping(
                    automated_qc_path,
                    label=f"automated pre-tractography QC for {unit}",
                )
            except (OSError, TypeError, ValueError) as exc:
                local_stage = "fod_tensor"
                local_reason = (
                    "automated_pre_tractography_qc_unreadable:"
                    f"{type(exc).__name__}"
                )
                local_flags = ["automated_qc_not_released_for_human_review"]
            else:
                automated_qc_pass = (
                    automated_qc.get("schema_version") == "2.0.0"
                    and automated_qc.get("record_type")
                    == "automated_pre_tractography_qc"
                    and automated_qc.get("status") == "PASS"
                    and automated_qc.get("unit") == unit
                    and automated_qc.get("diagnosis_labels_used") is False
                    and isinstance(automated_qc.get("image_ranges"), dict)
                    and isinstance(automated_qc.get("wmfod_qc"), dict)
                    and isinstance(automated_qc.get("commands"), list)
                    and automated_qc.get("failures") == []
                    and not {
                        "diagnosis",
                        "diagnosis_at_dti",
                        "group",
                        "research_group",
                    }.intersection(str(key).strip().lower() for key in automated_qc)
                )
                if not automated_qc_pass:
                    failures = automated_qc.get("failures")
                    first = (
                        str(failures[0]).strip()
                        if isinstance(failures, list) and failures
                        else "contract_or_status_differs"
                    )
                    local_stage = "fod_tensor"
                    local_reason = "automated_pre_tractography_qc_not_pass:" + first[:1000]
                    local_flags = ["automated_qc_not_released_for_human_review"]

        ready = set(present) == set(review_paths) and automated_qc_pass
        if ready:
            with review_paths["index"].open(newline="", encoding="utf-8-sig") as handle:
                rows = list(csv.DictReader(handle))
            if (
                len(rows) != 1
                or str(rows[0].get("unit", "")).strip() != unit
                or str(rows[0].get("status", "")).strip().upper() != "PENDING"
            ):
                ready = False
                local_stage = "visual_qc"
                local_reason = "visual_review_index_contract_differs"
                local_flags = ["review_bundle_not_released_for_human_qc"]
        elif local_reason is None:
            local_stage = "visual_qc"
            local_reason = "visual_review_bundle_incomplete"
            local_flags = ["review_bundle_not_released_for_human_qc"]

        stage: str | None = None
        reason: str | None = None
        flags: list[str] = []
        failure_markers: dict[str, dict[str, Any]] = {}
        if not ready:
            source_hashes = base._source_identity_hashes(row_by_unit[unit])
            structured = base._normalization_failure_evidence(
                run_root,
                unit,
                source_hashes,
                attempt_context_path=attempt_start_path,
            )
            if structured is not None:
                stage, reason, flags, failure_markers = structured
            elif local_reason is not None:
                stage, reason, flags = local_stage, local_reason, local_flags
            else:
                stage, reason, flags = base._rule_failure_for_unit(
                    unit, failed_rules, returncode=workflow_returncode
                )

        entry = {
            "schema_version": "2.0.0",
            "entry_type": "pre_tractography_canary_recovery_outcome",
            "run_id": run_id,
            "lineage_id": lineage_id,
            "recipe_id": recipe_id,
            "unit": unit,
            "status": "READY_FOR_HUMAN_QC" if ready else "FAIL",
            "failure_stage": stage,
            "primary_failure_reason": reason,
            "secondary_flags": list(
                dict.fromkeys(
                    [
                        *flags,
                        *(
                            [f"snakemake_returncode:{workflow_returncode}"]
                            if not ready
                            else []
                        ),
                    ]
                )
            ),
            "review_bundle": present,
            "automated_pre_tractography_qc": automated_qc_record,
            "failure_markers": failure_markers,
            "biological_inference_status": "NA",
            "matrix_outcomes_status": "NA",
            "tractography_started": False,
        }
        entries.append(entry)
        if ready:
            review_bundle_records[unit] = present
            automated_qc_records[unit] = automated_qc_record

    publication_dir = run_root / "publication"
    ledger_path = publication_dir / LEDGER_NAME
    manifest_path = publication_dir / MANIFEST_NAME
    if ledger_path.exists() or manifest_path.exists():
        raise FileExistsError("immutable pretract recovery publication already exists")
    base.write_immutable_jsonl(ledger_path, entries)
    ready_units = sorted(
        entry["unit"] for entry in entries if entry["status"] == "READY_FOR_HUMAN_QC"
    )
    fail_count = len(entries) - len(ready_units)
    outcome = (
        "PASS"
        if len(ready_units) == len(entries)
        else ("FAIL" if fail_count == len(entries) else "PARTIAL")
    )
    summary = {
        "expected": len(expected_units),
        "terminal": len(entries),
        "ready_for_human_qc": len(ready_units),
        "fail": fail_count,
    }
    manifest = {
        "schema_version": "2.0.0",
        "record_type": "pre_tractography_canary_recovery_manifest",
        "status": "COMPLETE",
        "workflow_state": "AWAITING_HUMAN_QC",
        "run_outcome": outcome,
        "generated_utc": ended_utc,
        "mode": "pre-tractography-canary-recovery1",
        "run_id": run_id,
        "lineage_id": lineage_id,
        "recipe_id": recipe_id,
        "execution_binding": execution_binding,
        "run_context": base.file_record(run_context_path),
        "attempt_start": base.file_record(attempt_start_path),
        "attempt_end": base.file_record(attempt_end_path),
        "combined_execution_log": log_record,
        "prior_failed_completion": prior_failure_completion,
        "expected_units": list(expected_units),
        "ready_units": ready_units,
        "summary": summary,
        "review_bundle_records": review_bundle_records,
        "automated_pre_tractography_qc_records": automated_qc_records,
        "pre_tractography_ledger": base.file_record(ledger_path),
        "human_qc_required_before_tractography": True,
        "tractography_started": False,
        "full_run_terminal_publication": False,
        "biological_inference_published": False,
        "matrix_outcomes_published": False,
    }
    base.write_immutable_json(manifest_path, manifest)
    return manifest
