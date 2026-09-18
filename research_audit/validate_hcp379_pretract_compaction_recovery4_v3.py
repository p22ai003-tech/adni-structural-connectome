#!/home/ec2-user/fsl/bin/python
"""Validate Recovery4 streaming compaction and tractography compatibility."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any


EXP = Path("/home/ec2-user/exp")
COMPACTOR_SOURCE = (
    EXP / "scripts/hcp/compact_hcp379_pretract_recovery4_v3.py"
)
PRETRACT_SOURCE = (
    EXP / "scripts/hcp/hcp379_scaleup_pretract_recovery4_v3.py"
)
TRACT_SOURCE = (
    EXP / "scripts/hcp/hcp379_scaleup_tractography_recovery4_v3.py"
)
ARCHIVE_SOURCE = (
    EXP
    / "scripts/hcp/"
    "run_hcp379_archive_calibration_pretract_recovery4_v3.py"
)
OUTPUT = (
    EXP
    / "research_audit/outputs/"
    "hcp379_pretract_compaction_recovery4_v3/validation.json"
)
PYTHON = Path("/home/ec2-user/fsl/bin/python")


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


COMPACTOR = load_module(
    COMPACTOR_SOURCE, "hcp379_recovery4_compactor_validation"
)
TRACT = load_module(
    TRACT_SOURCE, "hcp379_recovery4_tract_adapter_validation"
)


def check(
    checks: dict[str, dict[str, Any]],
    name: str,
    condition: bool,
    evidence: Any,
) -> None:
    checks[name] = {
        "status": "PASS" if condition else "FAIL",
        "evidence": evidence,
    }


def validate() -> dict[str, Any]:
    checks: dict[str, dict[str, Any]] = {}
    completed = subprocess.run(
        [str(PYTHON), str(COMPACTOR_SOURCE), "--self-test"],
        capture_output=True,
        text=True,
        check=False,
    )
    payload = (
        json.loads(completed.stdout)
        if completed.returncode == 0 and completed.stdout.strip()
        else {}
    )
    check(
        checks,
        "compactor_self_test",
        completed.returncode == 0 and payload.get("status") == "PASS",
        {
            "returncode": completed.returncode,
            "payload": payload,
            "stderr": completed.stderr,
        },
    )

    with tempfile.TemporaryDirectory(
        prefix="hcp379-recovery4-compaction-validation."
    ) as temporary:
        root = Path(temporary) / "corrected_scaleup_recovery4"
        unit = "synthetic_validation"
        state = COMPACTOR._synthetic_state(root, unit)
        paths = COMPACTOR.PRETRACT.spatial_paths(root, unit)
        future = paths["subject"] / "future_unknown_image.nii.gz"
        future.write_text("must be retained\n", encoding="utf-8")
        dry = COMPACTOR.compact_unit(
            root=root,
            unit=unit,
            execute=False,
            spacing_override=[2.0, 2.0, 2.0],
        )
        plan = COMPACTOR.PRETRACT.load_json(Path(dry["plan"]))
        check(
            checks,
            "dry_run_is_non_deleting",
            dry.get("status") == "DRY_RUN_READY"
            and paths["dwi"].is_file()
            and dry.get("reclaimable_bytes", 0) > 0,
            {
                "status": dry.get("status"),
                "dwi_present": paths["dwi"].is_file(),
                "reclaimable_bytes": dry.get("reclaimable_bytes"),
            },
        )
        check(
            checks,
            "unclassified_future_artifact_is_retained_by_plan",
            "future_unknown_image.nii.gz"
            in plan.get("unclassified_retained_artifacts", {}),
            sorted(plan.get("unclassified_retained_artifacts", {})),
        )

        result = COMPACTOR.compact_unit(
            root=root,
            unit=unit,
            execute=True,
            spacing_override=[2.0, 2.0, 2.0],
        )
        retained = result.get("retained_artifacts", {})
        check(
            checks,
            "write_ahead_compaction_passes",
            result.get("status") == "PASS_COMPACTED"
            and not paths["dwi"].exists()
            and future.is_file()
            and Path(retained["selected_five_tt"]["path"]).is_file()
            and Path(retained["wmfod_normalised"]["path"]).is_file(),
            {
                "status": result.get("status"),
                "dwi_deleted": not paths["dwi"].exists(),
                "future_artifact_retained": future.is_file(),
                "reclaimed_bytes": result.get("reclaimed_bytes"),
            },
        )
        raw_names = {"fa_raw", "md_raw", "rd_raw", "ad_raw"}
        check(
            checks,
            "raw_tensor_maps_are_preserved",
            result.get("raw_tensor_maps_preserved") is True
            and all(
                name in retained
                and retained[name] == state["artifacts"][name]
                and Path(retained[name]["path"]).is_file()
                for name in raw_names
            ),
            sorted(raw_names),
        )
        cached = COMPACTOR.PRETRACT.existing_final_pass(
            paths["state"], root=root, unit=unit
        )
        check(
            checks,
            "pretract_resume_accepts_valid_compaction",
            cached is not None
            and cached.get("status") == "PASS_PRETRACT_RECOVERY4",
            cached.get("status") if cached else None,
        )

        compacted = TRACT.validate_recovery4_pretract_compaction(root, unit)
        promoted = TRACT.promoted_pretract_state(root, unit)
        adapted = TRACT.ADAPTER.stage_paths(root, unit)
        TRACT.install_recovery4_adapter()
        step = TRACT.ENGINE.tractography_spacing(
            root=root,
            unit=unit,
            paths=adapted,
            env=TRACT.PRETRACT.environment(1),
        )
        check(
            checks,
            "tractography_adapter_accepts_compacted_recovery4",
            compacted is not None
            and promoted.get("status") == "PASS_PRETRACT_RECOVERY4"
            and not adapted["dwi"].exists()
            and adapted["five_tt"].is_file()
            and adapted["wmfod_norm"].is_file()
            and abs(step - 1.0) < 1.0e-12,
            {
                "compaction_status": (
                    compacted.get("status") if compacted else None
                ),
                "promoted_status": promoted.get("status"),
                "dwi_present": adapted["dwi"].exists(),
                "step_mm": step,
            },
        )
        aliases = set(COMPACTOR.LEGACY_TRACT_ALIASES)
        check(
            checks,
            "mature_engine_alias_contract_is_complete",
            aliases.issubset(retained),
            sorted(aliases),
        )

    unsafe_rejected = False
    try:
        COMPACTOR.ensure_scoped_root(Path("/data/derivatives/hcp379_v2"))
    except ValueError:
        unsafe_rejected = True
    check(
        checks,
        "broad_root_is_rejected",
        unsafe_rejected,
        "/data/derivatives/hcp379_v2",
    )

    pretract_text = PRETRACT_SOURCE.read_text(encoding="utf-8")
    archive_text = ARCHIVE_SOURCE.read_text(encoding="utf-8")
    check(
        checks,
        "streaming_compaction_is_explicit_opt_in",
        "--compact-reproducible-after-pass" in pretract_text
        and "if args.compact_reproducible_after_pass" in pretract_text
        and "--compact-reproducible-after-pass" in archive_text,
        {
            "central_runner": str(PRETRACT_SOURCE),
            "archive_wrapper": str(ARCHIVE_SOURCE),
        },
    )
    check(
        checks,
        "compactor_is_hash_bound",
        all(
            COMPACTOR.PRETRACT.file_record(path).get("sha256")
            for path in (
                COMPACTOR_SOURCE,
                PRETRACT_SOURCE,
                TRACT_SOURCE,
                ARCHIVE_SOURCE,
            )
        ),
        {
            path.name: COMPACTOR.PRETRACT.file_record(path)["sha256"]
            for path in (
                COMPACTOR_SOURCE,
                PRETRACT_SOURCE,
                TRACT_SOURCE,
                ARCHIVE_SOURCE,
            )
        },
    )

    passed = sum(row["status"] == "PASS" for row in checks.values())
    report = {
        "schema_version": "1.0.0",
        "record_type": (
            "hcp379_pretract_compaction_recovery4_v3_validation"
        ),
        "status": "PASS" if passed == len(checks) else "FAIL",
        "generated_utc": COMPACTOR.PRETRACT.utc_now(),
        "diagnosis_labels_used": False,
        "imaging_executed_on_real_subjects": False,
        "deletion_executed_on_real_subjects": False,
        "passed_checks": passed,
        "total_checks": len(checks),
        "checks": checks,
        "source": COMPACTOR.PRETRACT.file_record(COMPACTOR_SOURCE),
    }
    COMPACTOR.PRETRACT.atomic_json(OUTPUT, report)
    return report


def main() -> int:
    report = validate()
    print(json.dumps(report, indent=2, sort_keys=True))
    return int(report["status"] != "PASS")


if __name__ == "__main__":
    raise SystemExit(main())
