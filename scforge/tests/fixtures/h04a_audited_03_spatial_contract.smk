"""Frozen BBR and nonlinear MNI-to-T1 spatial transforms."""


rule mean_b0_nifti:
    input:
        rules.mean_b0.output,
    output:
        subject_path("{unit}", "03_spatial", "mean_b0.nii.gz"),
    log:
        subject_log("{unit}", "03_mean_b0_nifti.log"),
    shell:
        r"""
        set -euo pipefail
        mkdir -p "$(dirname {output:q})" "$(dirname {log:q})"
        export PATH={TOOL_PATH:q}
        mrconvert {input:q} {output:q} > {log:q} 2>&1
        """


rule b0_to_t1_bbr:
    input:
        b0=rules.mean_b0_nifti.output,
        t1=rules.t1_n4_bias_correct.output.t1,
        t1_brain=rules.t1_brain_extract.output.brain,
        wmseg=subject_path("{unit}", "05_model", "5tt_wmseg.nii.gz"),
    output:
        matrix=subject_path("{unit}", "03_spatial", "b0_to_t1_bbr.mat"),
        image=subject_path("{unit}", "03_spatial", "b0_in_t1_bbr.nii.gz"),
    params:
        prefix=lambda wc: str(Path(subject_path(wc.unit, "03_spatial", "b0_to_t1_bbr"))).replace(".nii.gz", ""),
    log:
        subject_log("{unit}", "03_epi_reg_bbr.log"),
    shell:
        r"""
        set -euo pipefail
        mkdir -p "$(dirname {output.matrix:q})" "$(dirname {log:q})"
        export PATH={TOOL_PATH:q}
        export FSLDIR={FSL_DIR:q}
        export FSLOUTPUTTYPE=NIFTI_GZ
        epi_reg --epi={input.b0:q} --t1={input.t1:q} --t1brain={input.t1_brain:q} \
          --wmseg={input.wmseg:q} --out={params.prefix:q} > {log:q} 2>&1
        test -s {output.matrix:q}
        test -s {output.image:q}
        """


rule invert_bbr_transform:
    input:
        matrix=rules.b0_to_t1_bbr.output.matrix,
        t1=rules.t1_n4_bias_correct.output.t1,
        b0=rules.mean_b0_nifti.output,
    output:
        fsl=subject_path("{unit}", "03_spatial", "t1_to_b0_bbr.mat"),
        mrtrix=subject_path("{unit}", "03_spatial", "t1_to_b0_bbr_mrtrix.txt"),
        itk=subject_path("{unit}", "03_spatial", "t1_to_b0_bbr_itk.txt"),
    log:
        subject_log("{unit}", "03_invert_and_convert_bbr.log"),
    shell:
        r"""
        set -euo pipefail
        mkdir -p "$(dirname {output.fsl:q})" "$(dirname {log:q})"
        export PATH={TOOL_PATH:q}
        export FSLDIR={FSL_DIR:q}
        convert_xfm -omat {output.fsl:q} -inverse {input.matrix:q} > {log:q} 2>&1
        transformconvert {output.fsl:q} {input.t1:q} {input.b0:q} flirt_import {output.mrtrix:q} \
          >> {log:q} 2>&1
        {C3D_AFFINE_TOOL:q} -ref {input.b0:q} -src {input.t1:q} {output.fsl:q} \
          -fsl2ras -oitk {output.itk:q} >> {log:q} 2>&1
        """


rule mni_to_t1_nonlinear:
    input:
        gate=ancient(rules.execution_preflight.output.report),
        moving=ancient(str(MNI_TEMPLATE)),
        fixed=rules.t1_n4_bias_correct.output.t1,
        fixed_mask=rules.t1_brain_extract.output.mask,
    output:
        affine=subject_path("{unit}", "03_spatial", "mni_to_t1_0GenericAffine.mat"),
        warp=subject_path("{unit}", "03_spatial", "mni_to_t1_1Warp.nii.gz"),
        inverse_warp=subject_path("{unit}", "03_spatial", "mni_to_t1_1InverseWarp.nii.gz"),
        warped=subject_path("{unit}", "03_spatial", "mni_in_t1.nii.gz"),
    params:
        prefix=lambda wc: subject_path(wc.unit, "03_spatial", "mni_to_t1_"),
        seed=lambda wc: subject_seed(wc.unit),
    log:
        subject_log("{unit}", "03_ants_mni_to_t1.log"),
    threads: 16
    shell:
        r"""
        set -euo pipefail
        mkdir -p "$(dirname {output.affine:q})" "$(dirname {log:q})"
        export PATH={TOOL_PATH:q}
        export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS={threads}
        export ANTS_RANDOM_SEED={params.seed}
        antsRegistrationSyN.sh -d 3 -f {input.fixed:q} -m {input.moving:q} \
          -o {params.prefix:q} -t s -x {input.fixed_mask:q} > {log:q} 2>&1
        test -s {output.affine:q}
        test -s {output.warp:q}
        test -s {output.inverse_warp:q}
        test -s {output.warped:q}
        """
