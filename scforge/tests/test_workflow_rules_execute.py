"""Defects that only show when a rule actually executes.

A dry run plans the DAG from declared file names; it never runs a rule's
Python body and never checks that a tool writes the file the rule declares.
Every defect below passed dry runs and was found by a real run (or had been
patched only in a canary copy of the rules and never brought back). These
checks read the active rule files, so they run in seconds and without data.
"""

from __future__ import annotations

import re
from pathlib import Path

RULES = Path(__file__).resolve().parents[1] / "workflow" / "rules"
ACTIVE = sorted(p for p in RULES.glob("0[0-8]_*.smk") if not re.search(r"retry|recovery", p.name))


def _rule(name: str) -> str:
    for path in ACTIVE:
        match = re.search(rf"(?ms)^rule {name}:\n(.*?)(?=^rule |\Z)", path.read_text(encoding="utf-8"))
        if match:
            return match.group(1)
    raise AssertionError(f"rule {name} not found in the active rules")


def _declared(body: str, key: str) -> str:
    match = re.search(rf'(?m)^\s+{key}=subject_path\("\{{unit\}}", "[^"]+", "([^"]+)"\)', body)
    assert match, f"{key} is not declared as a subject_path"
    return match.group(1)


def test_the_active_rules_are_the_ones_the_snakefile_includes():
    snakefile = (RULES.parent / "Snakefile").read_text(encoding="utf-8")
    included = set(re.findall(r'(?m)^include: "rules/([^"]+)"', snakefile))
    assert included == {p.name for p in ACTIVE}


def test_no_rule_takes_another_rules_whole_output_list():
    """`rules.x.output` is a list; in a Python run block `Path(input.name)`
    then raises TypeError. Every input names the one file it means."""
    bare = re.compile(r"(?m)^\s+\w+=(ancient\()?rules\.\w+\.output\)?,?\s*$")
    offenders = [
        f"{path.name}: {line.strip()}"
        for path in ACTIVE
        for line in path.read_text(encoding="utf-8").splitlines()
        if bare.match(line)
    ]
    assert not offenders, offenders


def test_epi_reg_outputs_are_the_files_it_writes():
    body = _rule("b0_to_t1_bbr")
    prefix = re.search(r'"03_spatial", "([^"]+)"\)\)\)\.replace', body).group(1)
    assert _declared(body, "image") == f"{prefix}.nii.gz"
    assert _declared(body, "matrix") == f"{prefix}.mat"


def test_ants_outputs_are_the_files_it_writes():
    body = _rule("mni_to_t1_nonlinear")
    prefix = re.search(r'prefix=lambda wc: subject_path\(wc\.unit, "03_spatial", "([^"]+)"\)', body).group(1)
    assert _declared(body, "affine") == f"{prefix}0GenericAffine.mat"
    assert _declared(body, "warp") == f"{prefix}1Warp.nii.gz"
    assert _declared(body, "inverse_warp") == f"{prefix}1InverseWarp.nii.gz"
    assert _declared(body, "warped") == f"{prefix}Warped.nii.gz"


def test_mni_registration_is_tissue_masked_and_seeded():
    body = _rule("mni_to_t1_nonlinear")
    assert "moving=ancient(str(MNI_BRAIN_TEMPLATE))" in body
    assert "fixed=rules.mni_registration_fixed_image.output.brain" in body
    assert "fixed_mask=rules.mni_registration_fixed_image.output.mask" in body
    assert "export ANTS_RANDOM_SEED={params.seed}" in body
    fixed = _rule("mni_registration_fixed_image")
    assert "mrthreshold" in fixed and "-abs 0.5" in fixed


def test_tissue_fractions_are_resampled_linearly_and_clipped_before_the_check():
    body = _rule("five_tt_dwi")
    assert "-interp linear" in body
    clip, check = body.index("mrcalc "), body.rindex("5ttcheck ")
    assert "0 -max 1 -min" in body[clip:check]


def test_ss3t_checks_its_outputs_without_system_binaries():
    """SS3T runs with a PATH that excludes /usr/bin on purpose."""
    body = _rule("ss3t_csd")
    assert "SS3T_TOOL_PATH" in body
    shell = body[body.index('r"""'):]
    assert not re.search(r"(?m)^\s*(grep|awk|sed|cut)\b", shell)
    for output in ("wm", "gm", "csf"):
        assert f"test -s {{output.{output}:q}}" in shell
