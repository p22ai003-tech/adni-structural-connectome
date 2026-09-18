from __future__ import annotations

import difflib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / "workflow"
OLD_RULES = WORKFLOW / "rules/05_5tt_fod_retry3.smk"
NEW_RULES = WORKFLOW / "rules/05_5tt_fod_retry4_hotfix_h04a_r1.smk"
OLD_SNAKEFILE = WORKFLOW / "Snakefile_h04a_r1_retry4_pretract_v2"
NEW_SNAKEFILE = WORKFLOW / "Snakefile_h04a_r1_retry4_pretract_v3"
OLD_PREFLIGHT = WORKFLOW / "extensions/h04a_r1_retry4_pretract/00_manifest.smk"
NEW_PREFLIGHT = (
    WORKFLOW / "extensions/h04a_r1_retry4_pretract/00_manifest_recovery1.smk"
)
OLD_SPATIAL = WORKFLOW / "rules/03_spatial_contract.smk"
NEW_SPATIAL = WORKFLOW / "rules/03_spatial_contract_recovery1_h04a_r1.smk"


def _changed_lines(old: Path, new: Path) -> list[str]:
    return [
        line
        for line in difflib.unified_diff(
            old.read_text(encoding="utf-8").splitlines(),
            new.read_text(encoding="utf-8").splitlines(),
            lineterm="",
        )
        if line.startswith(("+", "-"))
        and line not in {"+", "-"}
        and not line.startswith(("+++", "---"))
    ]


def test_ss3t_hotfix_is_only_documentation_and_output_postcondition() -> None:
    changes = _changed_lines(OLD_RULES, NEW_RULES)
    assert changes == [
        '-\"\"\"FSL 5TT, diagnosis-blind pooled responses, SS3T-CSD, and tensor maps.\"\"\"',
        '+\"\"\"Retry4 pretract hotfix: 5TT, frozen-response SS3T-CSD, and tensor maps.',
        '+The only change from ``05_5tt_fod_retry3.smk`` is the SS3T output',
        '+postcondition.  The locked SS3T PATH intentionally excludes system binaries,',
        '+so use the shell builtin ``test`` rather than an unresolvable external grep.',
        '+\"\"\"',
        "-        grep -q . {output.wm:q}",
        "+        test -s {output.wm:q}",
        "+        test -s {output.gm:q}",
        "+        test -s {output.csf:q}",
    ]


def test_hotfix_uses_no_external_grep_in_restricted_ss3t_path() -> None:
    text = NEW_RULES.read_text(encoding="utf-8")
    assert "export PATH={SS3T_TOOL_PATH:q}" in text
    assert "grep -q" not in text
    assert "test -s {output.wm:q}" in text
    assert "test -s {output.gm:q}" in text
    assert "test -s {output.csf:q}" in text


def test_recovery_snakefile_changes_only_label_and_rule_include() -> None:
    changes = _changed_lines(OLD_SNAKEFILE, NEW_SNAKEFILE)
    assert changes == [
        '-\"\"\"Fail-closed SL-H04A-R1 retry4 pre-tractography workflow.',
        '+\"\"\"Fail-closed SL-H04A-R1 retry4 pre-tractography recovery workflow.',
        '-include: "extensions/h04a_r1_retry4_pretract/00_manifest.smk"',
        '+include: "extensions/h04a_r1_retry4_pretract/00_manifest_recovery1.smk"',
        '-include: "rules/03_spatial_contract.smk"',
        '+include: "rules/03_spatial_contract_recovery1_h04a_r1.smk"',
        '-include: "rules/05_5tt_fod_retry3.smk"',
        '+include: "rules/05_5tt_fod_retry4_hotfix_h04a_r1.smk"',
    ]


def test_recovery_preflight_changes_only_immutable_output_names() -> None:
    assert _changed_lines(OLD_PREFLIGHT, NEW_PREFLIGHT) == [
        '-\"\"\"Retry4 pre-tractography manifest freeze and fail-closed execution gate.\"\"\"',
        '+\"\"\"Versioned manifest/preflight outputs for retry4 pretract recovery1.\"\"\"',
        '-        manifest=run_path("contract", "approved_acquisition_manifest.csv"),',
        '-        validation=run_path("contract", "manifest_validation.json"),',
        '+        manifest=run_path("contract", "approved_acquisition_manifest_recovery1.csv"),',
        '+        validation=run_path("contract", "manifest_validation_recovery1.json"),',
        '-        run_path("logs", "00_freeze_manifest.log"),',
        '+        run_path("logs", "00_freeze_manifest_recovery1.log"),',
        '-        report=run_path("contract", "execution_preflight.json"),',
        '-        fsl_cpu_path=directory(run_path("contract", "fsl_cpu_path")),',
        '+        report=run_path("contract", "execution_preflight_recovery1.json"),',
        '+        fsl_cpu_path=directory(run_path("contract", "fsl_cpu_path_recovery1")),',
        '-            run_path("contract", "source_runtime_inventory")',
        '+            run_path("contract", "source_runtime_inventory_recovery1")',
        '-        run_path("logs", "00_execution_preflight.log"),',
        '+        run_path("logs", "00_execution_preflight_recovery1.log"),',
    ]


def test_spatial_hotfix_changes_only_native_image_output_contracts() -> None:
    assert _changed_lines(OLD_SPATIAL, NEW_SPATIAL) == [
        '-\"\"\"Frozen BBR and nonlinear MNI-to-T1 spatial transforms.\"\"\"',
        '+\"\"\"Recovery1 BBR output contract and frozen nonlinear MNI-to-T1 transforms.\"\"\"',
        '-        image=subject_path("{unit}", "03_spatial", "b0_in_t1_bbr.nii.gz"),',
        '+        image=subject_path("{unit}", "03_spatial", "b0_to_t1_bbr.nii.gz"),',
        '-        warped=subject_path("{unit}", "03_spatial", "mni_in_t1.nii.gz"),',
        '+        warped=subject_path("{unit}", "03_spatial", "mni_to_t1_Warped.nii.gz"),',
    ]
