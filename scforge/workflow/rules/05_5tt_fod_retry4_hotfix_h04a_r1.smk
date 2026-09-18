"""Retry4 pretract hotfix: 5TT, frozen-response SS3T-CSD, and tensor maps.

The only change from ``05_5tt_fod_retry3.smk`` is the SS3T output
postcondition.  The locked SS3T PATH intentionally excludes system binaries,
so use the shell builtin ``test`` rather than an unresolvable external grep.
"""


def _extracted_shell_contract(image, expected_shells, expected_sizes):
    import subprocess

    observed = {}
    for name, flag, cast in (
        ("shells", "-shell_bvalues", float),
        ("sizes", "-shell_sizes", int),
    ):
        command = [
            str(MRTRIX_BIN / "mrinfo"),
            str(image),
            flag,
            "-config",
            "BZeroThreshold",
            str(BZERO_THRESHOLD),
        ]
        completed = subprocess.run(
            command,
            text=True,
            capture_output=True,
            check=False,
            env={**os.environ, "PATH": TOOL_PATH},
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"extracted shell query failed ({flag}): {completed.stderr.strip()}"
            )
        observed[name] = [cast(value) for value in completed.stdout.split()]
    if (
        len(observed["shells"]) != len(expected_shells)
        or any(
            abs(observed_value - expected_value) > 1e-6
            for observed_value, expected_value in zip(
                observed["shells"], expected_shells
            )
        )
        or observed["sizes"] != expected_sizes
    ):
        raise ValueError(
            f"extracted shell contract differs: observed={observed}, "
            f"expected_shells={expected_shells}, expected_sizes={expected_sizes}"
        )
    return observed


rule five_tt_t1:
    input:
        t1=rules.t1_n4_bias_correct.output.t1,
        brain_mask=rules.t1_brain_extract.output.mask,
    output:
        five_tt=subject_path("{unit}", "05_model", "5tt_t1_fsl.mif"),
        check=subject_path("{unit}", "05_model", "5tt_t1_check.done"),
    log:
        subject_log("{unit}", "05_5ttgen_fsl.log"),
    threads: 8
    shell:
        r"""
        set -euo pipefail
        mkdir -p "$(dirname {output.five_tt:q})" "$(dirname {log:q})"
        export PATH={TOOL_PATH:q}
        export FSLDIR=/home/ec2-user/fsl
        export FSLOUTPUTTYPE=NIFTI_GZ
        5ttgen fsl {input.t1:q} {output.five_tt:q} -nocrop -nthreads {threads} \
          > {log:q} 2>&1
        5ttcheck {output.five_tt:q} >> {log:q} 2>&1
        printf 'PASS\n' > {output.check:q}
        """


rule five_tt_wmseg:
    input:
        five_tt=rules.five_tt_t1.output.five_tt,
        check=rules.five_tt_t1.output.check,
    output:
        subject_path("{unit}", "05_model", "5tt_wmseg.nii.gz"),
    log:
        subject_log("{unit}", "05_5tt_wmseg.log"),
    shell:
        r"""
        set -euo pipefail
        mkdir -p "$(dirname {output:q})" "$(dirname {log:q})"
        export PATH={TOOL_PATH:q}
        mrconvert {input.five_tt:q} - -coord 3 2 2>> {log:q} \
          | mrthreshold - {output:q} -abs {config[t1_and_tissue][five_tissue_type][white_matter_segmentation_threshold]} \
            >> {log:q} 2>&1
        """


rule five_tt_dwi:
    input:
        five_tt=rules.five_tt_t1.output.five_tt,
        transform=rules.invert_bbr_transform.output.mrtrix,
        reference=rules.mean_b0.output,
    output:
        five_tt=subject_path("{unit}", "05_model", "5tt_dwi.mif"),
        check=subject_path("{unit}", "05_model", "5tt_dwi_check.done"),
    log:
        subject_log("{unit}", "05_5tt_to_dwi.log"),
    threads: 4
    shell:
        r"""
        set -euo pipefail
        mkdir -p "$(dirname {output.five_tt:q})" "$(dirname {log:q})"
        export PATH={TOOL_PATH:q}
        mrtransform {input.five_tt:q} {output.five_tt:q} -linear {input.transform:q} \
          -template {input.reference:q} -nthreads {threads} > {log:q} 2>&1
        5ttcheck {output.five_tt:q} >> {log:q} 2>&1
        printf 'PASS\n' > {output.check:q}
        """


rule gmwmi_dwi:
    input:
        five_tt=rules.five_tt_dwi.output.five_tt,
        check=rules.five_tt_dwi.output.check,
    output:
        subject_path("{unit}", "05_model", "gmwmi_dwi.mif"),
    log:
        subject_log("{unit}", "05_5tt2gmwmi.log"),
    shell:
        r"""
        set -euo pipefail
        mkdir -p "$(dirname {output:q})" "$(dirname {log:q})"
        export PATH={TOOL_PATH:q}
        5tt2gmwmi {input.five_tt:q} {output:q} > {log:q} 2>&1
        """


rule select_fod_shells:
    input:
        dwi=rules.dwi_bias_correct.output,
        contract=rules.gradient_contract.output[0],
    output:
        dwi=subject_path("{unit}", "05_model", "dwi_fod_shells.mif"),
        selection=subject_path("{unit}", "05_model", "fod_shell_selection.json"),
    log:
        subject_log("{unit}", "05_select_fod_shells.log"),
    threads: 4
    run:
        contract = json.loads(Path(input.contract).read_text(encoding="utf-8"))
        shells = contract.get("selected_fod_shells")
        if (
            contract.get("status") != "PASS"
            or not isinstance(shells, list)
            or len(shells) != 2
            or float(shells[0]) > BZERO_THRESHOLD
        ):
            raise ValueError(f"{wildcards.unit} has no valid frozen FOD shell set")
        shell_text = ",".join(str(value) for value in shells)
        shell(
            r"""
            set -euo pipefail
            mkdir -p "$(dirname {output.dwi:q})" "$(dirname {log:q})"
            export PATH={TOOL_PATH:q}
            dwiextract {input.dwi:q} {output.dwi:q} -shells {shell_text:q} \
              -config BZeroThreshold {BZERO_THRESHOLD} -nthreads {threads} \
              > {log:q} 2>&1
            """
        )
        observed = _extracted_shell_contract(
            output.dwi, shells, contract["selected_fod_shell_sizes"]
        )
        record = {
            "schema_version": "2.0.0",
            "status": "PASS",
            "unit": str(wildcards.unit),
            "purpose": "SS3T_CSD_and_response_estimation",
            "selected_shells_s_per_mm2": shells,
            "selected_shell_sizes": contract["selected_fod_shell_sizes"],
            "bzero_threshold_s_per_mm2": BZERO_THRESHOLD,
            "observed_output_shell_contract": observed,
            "selected_unique_direction_count": contract["selected_fod_unique_direction_count"],
            "selection_uses_diagnosis_labels": False,
            "gradient_contract_sha256": _sha256(input.contract),
        }
        Path(output.selection).write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


rule select_tensor_shells:
    input:
        dwi=rules.dwi_bias_correct.output,
        contract=rules.gradient_contract.output[0],
    output:
        dwi=subject_path("{unit}", "05_model", "dwi_tensor_shells.mif"),
        selection=subject_path("{unit}", "05_model", "tensor_shell_selection.json"),
    log:
        subject_log("{unit}", "05_select_tensor_shells.log"),
    threads: 4
    run:
        contract = json.loads(Path(input.contract).read_text(encoding="utf-8"))
        shells = contract.get("selected_tensor_shells")
        if (
            contract.get("status") != "PASS"
            or not isinstance(shells, list)
            or len(shells) < 2
            or float(shells[0]) > BZERO_THRESHOLD
        ):
            raise ValueError(f"{wildcards.unit} has no valid frozen tensor shell set")
        shell_text = ",".join(str(value) for value in shells)
        shell(
            r"""
            set -euo pipefail
            mkdir -p "$(dirname {output.dwi:q})" "$(dirname {log:q})"
            export PATH={TOOL_PATH:q}
            dwiextract {input.dwi:q} {output.dwi:q} -shells {shell_text:q} \
              -config BZeroThreshold {BZERO_THRESHOLD} -nthreads {threads} \
              > {log:q} 2>&1
            """
        )
        observed = _extracted_shell_contract(
            output.dwi, shells, contract["selected_tensor_shell_sizes"]
        )
        record = {
            "schema_version": "2.0.0",
            "status": "PASS",
            "unit": str(wildcards.unit),
            "purpose": "diffusion_tensor_fit",
            "selected_shells_s_per_mm2": shells,
            "selected_shell_sizes": contract["selected_tensor_shell_sizes"],
            "bzero_threshold_s_per_mm2": BZERO_THRESHOLD,
            "observed_output_shell_contract": observed,
            "selected_unique_direction_count": contract["selected_tensor_unique_direction_count"],
            "selection_uses_diagnosis_labels": False,
            "gradient_contract_sha256": _sha256(input.contract),
        }
        Path(output.selection).write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


rule subject_response:
    input:
        dwi=rules.select_fod_shells.output.dwi,
        fod_shell_selection=rules.select_fod_shells.output.selection,
        mask=rules.dwi_brain_mask.output,
        input_contract=rules.input_contract_gate.output[0],
        run_context=str(RUN_CONTEXT_PATH),
        attempt_context=str(ATTEMPT_CONTEXT_PATH),
    output:
        wm=subject_path("{unit}", "05_model", "response_wm.txt"),
        gm=subject_path("{unit}", "05_model", "response_gm.txt"),
        csf=subject_path("{unit}", "05_model", "response_csf.txt"),
        voxels=subject_path("{unit}", "05_model", "response_voxels.mif"),
        outcome=subject_path("{unit}", "05_model", "response_calibration_outcome.json"),
    log:
        subject_log("{unit}", "05_dwi2response_dhollander.log"),
    threads: 8
    run:
        import datetime
        import subprocess

        shell(
            r"""
            set -euo pipefail
            mkdir -p "$(dirname {output.wm:q})" "$(dirname {log:q})"
            export PATH={TOOL_PATH:q}
        dwi2response dhollander {input.dwi:q} {output.wm:q} {output.gm:q} {output.csf:q} \
              -mask {input.mask:q} -voxels {output.voxels:q} \
              -lmax {config[fod][response_estimation][shell_ordered_lmax]:q} \
              -config BZeroThreshold {BZERO_THRESHOLD} -nthreads {threads} \
              > {log:q} 2>&1
            """
        )
        input_contract = json.loads(
            Path(input.input_contract).read_text(encoding="utf-8")
        )
        run_context = json.loads(Path(input.run_context).read_text(encoding="utf-8"))
        attempt_context = json.loads(
            Path(input.attempt_context).read_text(encoding="utf-8")
        )
        unit = str(wildcards.unit)
        if (
            input_contract.get("status") != "PASS"
            or input_contract.get("unit") != unit
            or input_contract.get("recipe_id") != RECIPE_ID
        ):
            raise ValueError(f"{unit} input contract identity differs before response estimation")
        if (
            run_context.get("recipe_id") != RECIPE_ID
            or attempt_context.get("recipe_id") != RECIPE_ID
            or attempt_context.get("run_id") != run_context.get("run_id")
            or attempt_context.get("launcher_mode") != "response-calibration-phase-a"
        ):
            raise ValueError(f"{unit} phase-A run/attempt identity differs")
        voxel_count_command = [
            str(MRTRIX_BIN / "mrstats"),
            str(output.voxels),
            "-output",
            "count",
            "-ignorezero",
        ]
        voxel_count_result = subprocess.run(
            voxel_count_command,
            text=True,
            capture_output=True,
            check=False,
            env={**os.environ, "PATH": TOOL_PATH},
        )
        if voxel_count_result.returncode != 0:
            raise RuntimeError(
                f"{unit} response-voxel count failed: "
                + voxel_count_result.stderr.strip()
            )
        count_tokens = voxel_count_result.stdout.split()
        if len(count_tokens) != 3:
            raise ValueError(
                f"{unit} response-voxel count is not three CSF/GM/WM scalars: {count_tokens!r}"
            )
        numeric_counts = [float(token) for token in count_tokens]
        if any(not value.is_integer() or value <= 0 for value in numeric_counts):
            raise ValueError(
                f"{unit} response-voxel counts are not positive integers: {count_tokens!r}"
            )
        selected_voxel_counts = {
            "csf": int(numeric_counts[0]),
            "gm": int(numeric_counts[1]),
            "wm": int(numeric_counts[2]),
        }
        qc_contract = RESPONSE_CALIBRATION["response_qc"]
        response_paths = {
            "wm": output.wm,
            "gm": output.gm,
            "csf": output.csf,
        }
        response_qc = assess_response_calibration_candidate(
            response_paths,
            selected_voxel_counts=selected_voxel_counts,
            expected_shell_rows=int(qc_contract["expected_shell_rows"]),
            expected_coefficient_columns=qc_contract["expected_coefficient_columns"],
        )
        common = dict(
            unit=unit,
            recipe_id=RECIPE_ID,
            source_identity_hashes=input_contract["source_identity_hashes"],
            generated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            phase_a_run_id=str(run_context["run_id"]),
            phase_a_attempt_id=str(attempt_context["attempt_id"]),
            run_context_path=input.run_context,
            attempt_context_path=input.attempt_context,
        )
        if response_qc["status"] == "PASS":
            write_pass_response_calibration_outcome(
                output.outcome,
                **common,
                response_paths=response_paths,
                response_qc=response_qc,
                response_voxels_path=output.voxels,
                fod_shell_selection_path=input.fod_shell_selection,
            )
        else:
            write_failed_response_calibration_outcome(
                output.outcome,
                **common,
                primary_failure_reason="response_qc_failed:"
                + ";".join(response_qc["failure_reasons"]),
                response_qc=response_qc,
                candidate_response_paths=response_paths,
                response_voxels_path=output.voxels,
            )


rule response_calibration_phase_a:
    input:
        expand(
            run_path("subjects", "{unit}", "05_model", "response_calibration_outcome.json"),
            unit=UNITS,
        ),


rule pooled_response:
    input:
        manifest=ancient(str(FROZEN_RESPONSE_MANIFEST)),
        wm=ancient(str(FROZEN_RESPONSE_PATHS["wm"])),
        gm=ancient(str(FROZEN_RESPONSE_PATHS["gm"])),
        csf=ancient(str(FROZEN_RESPONSE_PATHS["csf"])),
    output:
        wm=run_path("05_group_response", "pooled_response_wm.txt"),
        gm=run_path("05_group_response", "pooled_response_gm.txt"),
        csf=run_path("05_group_response", "pooled_response_csf.txt"),
        manifest=run_path("05_group_response", "frozen_response_calibration_manifest.json"),
    log:
        run_path("logs", "05_pooled_response_median.log"),
    run:
        import shutil

        Path(output.wm).parent.mkdir(parents=True, exist_ok=True)
        Path(log[0]).parent.mkdir(parents=True, exist_ok=True)
        record = validate_frozen_response_calibration(
            input.manifest,
            expected_manifest_sha256=RESPONSE_CALIBRATION["frozen_manifest"]["sha256"],
            minimum_valid_subjects=int(RESPONSE_CALIBRATION["minimum_valid_subjects"]),
            expected_responses=RESPONSE_CALIBRATION["pooled_responses"],
            expected_technical_diversity=config["response_calibration_binding"][
                "valid_pool_technical_diversity"
            ],
        )
        for source, target in (
            (input.wm, output.wm),
            (input.gm, output.gm),
            (input.csf, output.csf),
            (input.manifest, output.manifest),
        ):
            temporary = Path(str(target) + ".partial")
            shutil.copyfile(source, temporary)
            temporary.replace(target)
        Path(log[0]).write_text(
            "internal: validate and snapshot diagnosis-blind frozen phase-B response calibration\n"
            + json.dumps(record, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


rule fod_shell_compatibility_gate:
    input:
        selection=rules.select_fod_shells.output.selection,
        calibration=rules.pooled_response.output.manifest,
    output:
        subject_path("{unit}", "05_model", "fod_shell_compatibility.json"),
    log:
        subject_log("{unit}", "05_fod_shell_compatibility.log"),
    run:
        selection = json.loads(Path(input.selection).read_text(encoding="utf-8"))
        calibration = json.loads(Path(input.calibration).read_text(encoding="utf-8"))
        expected = calibration.get("fod_shell_compatibility", {}).get(
            "selected_shells_s_per_mm2"
        )
        observed = selection.get("selected_shells_s_per_mm2")
        failures = []
        if selection.get("status") != "PASS":
            failures.append("subject_FOD_shell_selection_not_PASS")
        if expected != observed:
            failures.append(
                f"subject_shell_centroids_{observed}_differ_from_calibration_{expected}"
            )
        record = {
            "schema_version": "2.0.0",
            "status": "PASS" if not failures else "FAIL",
            "unit": str(wildcards.unit),
            "compatibility_rule": "exact_MRtrix_clustered_b0_and_nonzero_centroids",
            "expected_shells_s_per_mm2": expected,
            "observed_shells_s_per_mm2": observed,
            "subject_selection_sha256": _sha256(input.selection),
            "frozen_calibration_sha256": _sha256(input.calibration),
            "failures": failures,
        }
        Path(log[0]).parent.mkdir(parents=True, exist_ok=True)
        Path(log[0]).write_text(
            "internal: exact FOD shell compatibility gate\n"
            + json.dumps(record, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        if failures:
            raise ValueError("FOD shell compatibility failed: " + ";".join(failures))
        temporary = Path(str(output[0]) + ".partial")
        temporary.write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.replace(output[0])


rule ss3t_csd:
    input:
        dwi=rules.select_fod_shells.output.dwi,
        mask=rules.dwi_brain_mask.output,
        wm=rules.pooled_response.output.wm,
        gm=rules.pooled_response.output.gm,
        csf=rules.pooled_response.output.csf,
        shell_compatibility=rules.fod_shell_compatibility_gate.output,
    output:
        wm=subject_path("{unit}", "05_model", "wmfod.mif"),
        gm=subject_path("{unit}", "05_model", "gm.mif"),
        csf=subject_path("{unit}", "05_model", "csf.mif"),
    log:
        subject_log("{unit}", "05_ss3t_csd.log"),
    threads: 8
    shell:
        r"""
        set -euo pipefail
        mkdir -p "$(dirname {output.wm:q})" "$(dirname {log:q})"
        export PATH={SS3T_TOOL_PATH:q}
        {SS3T_PYTHON:q} {SS3T_SCRIPT:q} {input.dwi:q} {input.wm:q} {output.wm:q} \
          {input.gm:q} {output.gm:q} {input.csf:q} {output.csf:q} \
          -mask {input.mask:q} -lmax {config[fod][lmax]} \
          -config BZeroThreshold {BZERO_THRESHOLD} -nthreads {threads} > {log:q} 2>&1
        test -s {output.wm:q}
        test -s {output.gm:q}
        test -s {output.csf:q}
        """


rule mtnormalise:
    input:
        wm=rules.ss3t_csd.output.wm,
        gm=rules.ss3t_csd.output.gm,
        csf=rules.ss3t_csd.output.csf,
        mask=rules.dwi_brain_mask.output,
    output:
        wm=subject_path("{unit}", "05_model", "wmfod_norm.mif"),
        gm=subject_path("{unit}", "05_model", "gm_norm.mif"),
        csf=subject_path("{unit}", "05_model", "csf_norm.mif"),
    log:
        subject_log("{unit}", "05_mtnormalise.log"),
    threads: 8
    shell:
        r"""
        set -euo pipefail
        mkdir -p "$(dirname {output.wm:q})" "$(dirname {log:q})"
        export PATH={TOOL_PATH:q}
        mtnormalise {input.wm:q} {output.wm:q} {input.gm:q} {output.gm:q} \
          {input.csf:q} {output.csf:q} -mask {input.mask:q} -nthreads {threads} \
          > {log:q} 2>&1
        """


rule tensor_fit:
    input:
        dwi=rules.select_tensor_shells.output.dwi,
        mask=rules.dwi_brain_mask.output,
    output:
        subject_path("{unit}", "05_model", "tensor.mif"),
    log:
        subject_log("{unit}", "05_dwi2tensor.log"),
    threads: 8
    shell:
        r"""
        set -euo pipefail
        mkdir -p "$(dirname {output:q})" "$(dirname {log:q})"
        export PATH={TOOL_PATH:q}
        dwi2tensor {input.dwi:q} {output:q} -mask {input.mask:q} -iter 2 \
          -config BZeroThreshold {BZERO_THRESHOLD} -nthreads {threads} \
          > {log:q} 2>&1
        """


rule tensor_metrics:
    input:
        rules.tensor_fit.output,
    output:
        fa=subject_path("{unit}", "05_model", "fa.mif"),
        md=subject_path("{unit}", "05_model", "md.mif"),
        rd=subject_path("{unit}", "05_model", "rd.mif"),
        ad=subject_path("{unit}", "05_model", "ad.mif"),
    log:
        subject_log("{unit}", "05_tensor2metric.log"),
    threads: 4
    shell:
        r"""
        set -euo pipefail
        mkdir -p "$(dirname {output.fa:q})" "$(dirname {log:q})"
        export PATH={TOOL_PATH:q}
        tensor2metric {input:q} -fa {output.fa:q} -adc {output.md:q} \
          -rd {output.rd:q} -ad {output.ad:q} -nthreads {threads} > {log:q} 2>&1
        """
