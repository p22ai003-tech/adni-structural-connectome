"""Native T1 bias correction and brain extraction."""


rule t1_n4_bias_correct:
    input:
        gate=rules.input_contract_gate.output,
        t1=rules.normalize_t1_source.output.t1,
    output:
        t1=subject_path("{unit}", "02_anat", "t1_n4.nii.gz"),
        bias=subject_path("{unit}", "02_anat", "t1_n4_bias.nii.gz"),
    log:
        subject_log("{unit}", "02_n4_bias.log"),
    threads: 8
    shell:
        r"""
        set -euo pipefail
        mkdir -p "$(dirname {output.t1:q})" "$(dirname {log:q})"
        export PATH={TOOL_PATH:q}
        export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS={threads}
        N4BiasFieldCorrection -d 3 -i {input.t1:q} \
          -o [{output.t1:q},{output.bias:q}] > {log:q} 2>&1
        """


rule t1_brain_extract:
    input:
        rules.t1_n4_bias_correct.output.t1,
    output:
        brain=subject_path("{unit}", "02_anat", "t1_brain.nii.gz"),
        mask=subject_path("{unit}", "02_anat", "t1_brain_mask.nii.gz"),
    log:
        subject_log("{unit}", "02_bet.log"),
    shell:
        r"""
        set -euo pipefail
        mkdir -p "$(dirname {output.brain:q})" "$(dirname {log:q})"
        export PATH={TOOL_PATH:q}
        export FSLDIR={FSL_DIR:q}
        export FSLOUTPUTTYPE=NIFTI_GZ
        bet {input:q} {output.brain:q} -m -R -f 0.30 > {log:q} 2>&1
        test -s {output.mask:q}
        """
