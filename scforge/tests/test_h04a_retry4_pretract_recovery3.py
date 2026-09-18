from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / "workflow"
SNAKEFILE = WORKFLOW / "Snakefile_h04a_r1_retry4_pretract_v5"
SPATIAL = WORKFLOW / "rules/03_spatial_contract_recovery3_h04a_r1.smk"
ATLAS = WORKFLOW / "rules/04_atlas_recovery3_h04a_r1.smk"
MODEL = (
    WORKFLOW
    / "rules/05_5tt_fod_retry4_hotfix_recovery3_h04a_r1.smk"
)
PREFLIGHT = WORKFLOW / "rules/06_preflight_recovery3_h04a_r1.smk"


def test_recovery3_uses_tissue_masked_mni_registration() -> None:
    text = SPATIAL.read_text(encoding="utf-8")
    assert "rule mni_registration_fixed_image:" in text
    assert 'five_tt=subject_path("{unit}", "05_model", "5tt_t1_fsl.mif")' in text
    assert "mrmath {input.five_tt:q} sum" in text
    assert "mrthreshold" in text
    assert "fslmaths {input.t1:q} -mas {output.mask:q}" in text
    assert "moving=ancient(str(MNI_BRAIN_TEMPLATE))" in text
    assert "fixed=rules.mni_registration_fixed_image.output.brain" in text
    assert "fixed_mask=rules.mni_registration_fixed_image.output.mask" in text
    assert "rules.t1_brain_extract.output.mask" not in text.split(
        "rule mni_to_t1_nonlinear:", 1
    )[1]


def test_recovery3_5tt_resampling_preserves_partial_volumes_and_clips_roundoff() -> None:
    text = MODEL.read_text(encoding="utf-8")
    block = text.split("rule five_tt_dwi:", 1)[1].split("rule gmwmi_dwi:", 1)[0]
    assert "-interp linear" in block
    assert 'mrcalc "$raw" 0 -max 1 -min' in block
    assert "5ttcheck {output.five_tt:q}" in block
    assert "5tt_dwi_recovery3.mif" in block
    assert "-interp nearest" not in block


def test_recovery3_atlas_python_rules_receive_scalar_paths() -> None:
    text = ATLAS.read_text(encoding="utf-8")
    assert "atlas=rules.aal3_to_dwi_single_resample.output[0]" in text
    assert "reference=rules.b0_one_mm_world_grid.output[0]" in text
    assert "image = nib.load(atlas_path)" in text
    assert "reference = nib.load(reference_path)" in text
    assert "atlas_contract_qc_recovery3.json" in text


def test_recovery3_spatial_qc_keeps_hard_alignment_gates() -> None:
    text = PREFLIGHT.read_text(encoding="utf-8")
    assert "minimum_5tt_to_dwi_mask_dice" in text
    assert "minimum_atlas_label_inside_brain_fraction" in text
    assert "atlas=rules.aal3_to_dwi_single_resample.output[0]" in text
    assert "gradients=rules.gradient_contract.output[0]" in text
    assert "automated_pre_tractography_qc_recovery3.json" in text


def test_recovery3_wrapper_targets_only_versioned_pretract_outputs() -> None:
    text = SNAKEFILE.read_text(encoding="utf-8")
    for expected in (
        'include: "rules/03_spatial_contract_recovery3_h04a_r1.smk"',
        'include: "rules/04_atlas_recovery3_h04a_r1.smk"',
        'include: "rules/05_5tt_fod_retry4_hotfix_recovery3_h04a_r1.smk"',
        'include: "rules/06_preflight_recovery3_h04a_r1.smk"',
        "automated_pre_tractography_qc_recovery3.json",
        "visual_review_index_recovery3.csv",
        "gmwmi_dwi_recovery3.mif",
    ):
        assert expected in text
    assert "RECOVERY3_C3D_AFFINE_TOOL" in text
    assert "MNI_BRAIN_TEMPLATE" in text
