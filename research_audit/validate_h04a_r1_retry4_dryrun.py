#!/usr/bin/env python3
"""Run the proven retry3 dry-run audit against the retry4 package."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


ROOT = Path("/home/ec2-user/exp")
AUDIT = ROOT / "research_audit"
RUN_ROOT = Path("/data/derivatives/scforge_v2/h04a_r1_recovery_20260719_retry4")
PACKAGE = AUDIT / "outputs/h04a_r1_retry4_package_v1"
OUT = AUDIT / "outputs/h04a_r1_retry4_dryrun_v1"

sys.path.insert(0, str(AUDIT))
sys.path.insert(0, str(ROOT / "scforge/workflow"))
sys.path.insert(0, str(ROOT / "scforge"))

import validate_h04a_r1_retry3_dryrun as audit  # noqa: E402
import run_h04a_r1_retry4 as wrapper  # noqa: E402
from scforge import h04a_r1_retry4 as compatibility  # noqa: E402


sys.modules["run_h04a_r1_retry3"] = wrapper
sys.modules["scforge.h04a_r1_retry3"] = compatibility

audit.RUN_ROOT = RUN_ROOT
audit.BASE_PACKAGE = PACKAGE
audit.SAFETY_PACKAGE = PACKAGE
audit.DECISION = PACKAGE / "recovery_execution_decision_retry4.json"
audit.PACKAGE_VALIDATION = PACKAGE / "retry4_package_validation.json"
audit.SEED_MANIFEST = PACKAGE / "retry4_seed_manifest.json"
audit.BINDING = AUDIT / "outputs/h04a_r1_recovery_extension_binding_v7.json"
audit.OUT = OUT
audit.SERVICE = "scforge-h04a-r1-phase-a-retry3.service"

_original_workspace = audit.temporary_dry_run_workspace
_original_contexts = audit.write_dry_run_contexts


def _retry4_workspace(run_root: Path | None = None):
    return _original_workspace(RUN_ROOT if run_root is None else run_root)


def _retry4_contexts(
    workspace: Path,
    *,
    recipe_id: str,
    execution_binding: dict[str, object],
    recovery_extension_binding: dict[str, object],
    run_root: Path | None = None,
):
    return _original_contexts(
        workspace,
        recipe_id=recipe_id,
        execution_binding=execution_binding,
        recovery_extension_binding=recovery_extension_binding,
        run_root=RUN_ROOT if run_root is None else run_root,
    )


audit.temporary_dry_run_workspace = _retry4_workspace
audit.write_dry_run_contexts = _retry4_contexts


def main() -> int:
    result = audit.main()
    report_path = OUT / "validation.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    checks = report["checks"]
    checks["source_retry3_service_terminal"] = checks.pop("retry2_service_terminal")
    checks["retry4_package_pass"] = checks.pop("retry3_package_pass")
    seed = json.loads(audit.SEED_MANIFEST.read_text(encoding="utf-8"))
    checks["seed_excludes_mutable_continuation_paths"] = (
        seed.get("allowed_subject_areas") == ["00_inputs", "01_dwi"]
        and seed.get("continuation_paths_excluded") is True
    )

    dryrun_text = (OUT / "snakemake_dry_run.txt").read_text(
        encoding="utf-8", errors="replace"
    )
    job_counts: dict[str, int] = {}
    for rule in ("select_fod_shells", "subject_response", "response_calibration_phase_a"):
        match = re.search(rf"(?m)^{rule}\s+(\d+)\s*$", dryrun_text)
        if match:
            job_counts[rule] = int(match.group(1))
    total = re.search(r"(?m)^total\s+(\d+)\s*$", dryrun_text)
    job_counts["total"] = int(total.group(1)) if total else -1
    checks["exact_31_job_continuation"] = job_counts == {
        "select_fod_shells": 15,
        "subject_response": 15,
        "response_calibration_phase_a": 1,
        "total": 31,
    }
    report["scheduled_job_counts"] = job_counts
    report["status"] = "PASS" if all(checks.values()) else "FAIL"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# H04A retry4 immutable-upstream dry-run validation",
        "",
        f"**Generated:** {report['generated_utc']}",
        f"**Verdict:** {report['status']}",
        f"**Scheduled jobs:** {job_counts}",
        "",
    ]
    for name, passed in checks.items():
        lines.append(f"- [{'x' if passed else ' '}] {name}")
    (OUT / "validation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "scheduled_rules": report["scheduled_rules"],
                "scheduled_job_counts": job_counts,
            },
            indent=2,
        )
    )
    return 0 if result == 0 and report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
