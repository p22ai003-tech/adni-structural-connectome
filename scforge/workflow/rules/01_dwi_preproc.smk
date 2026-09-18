"""Locked DWI preprocessing after source normalization (rpe_none only)."""


rule gradient_contract:
    input:
        gate=rules.input_contract_gate.output,
        dwi=rules.normalize_dwi_source.output.dwi,
        bvec=rules.normalize_dwi_source.output.bvec,
        bval=rules.normalize_dwi_source.output.bval,
    output:
        subject_path("{unit}", "01_dwi", "gradient_contract.json"),
    log:
        subject_log("{unit}", "01_gradient_contract.log"),
    run:
        import subprocess
        import numpy as np

        Path(output[0]).parent.mkdir(parents=True, exist_ok=True)
        Path(log[0]).parent.mkdir(parents=True, exist_ok=True)
        command = [str(MRTRIX_BIN / "mrinfo"), str(input.dwi), "-size"]
        completed = subprocess.run(command, text=True, capture_output=True, check=False)
        failures = []
        if completed.returncode != 0:
            failures.append(f"mrinfo failed with exit {completed.returncode}")
            volume_count = None
        else:
            try:
                dimensions = [int(value) for value in completed.stdout.split()]
                volume_count = dimensions[3]
            except Exception:
                volume_count = None
                failures.append("unable to parse mrinfo -size")
        shell_commands = {
            "bvalues": [str(MRTRIX_BIN / "mrinfo"), str(input.dwi), "-shell_bvalues", "-config", "BZeroThreshold", str(BZERO_THRESHOLD)],
            "sizes": [str(MRTRIX_BIN / "mrinfo"), str(input.dwi), "-shell_sizes", "-config", "BZeroThreshold", str(BZERO_THRESHOLD)],
        }
        shell_results = {
            name: subprocess.run(cmd, text=True, capture_output=True, check=False)
            for name, cmd in shell_commands.items()
        }
        if any(result.returncode != 0 for result in shell_results.values()):
            failures.append("mrinfo clustered-shell query failed")
            shell_centroids = []
            shell_sizes = []
        else:
            try:
                shell_centroids = [float(value) for value in shell_results["bvalues"].stdout.split()]
                shell_sizes = [int(value) for value in shell_results["sizes"].stdout.split()]
                if (
                    not shell_centroids
                    or len(shell_centroids) != len(shell_sizes)
                    or any(size < 1 for size in shell_sizes)
                    or (volume_count is not None and sum(shell_sizes) != volume_count)
                ):
                    raise ValueError("clustered shell centroid/size shape differs")
            except Exception as exc:
                shell_centroids = []
                shell_sizes = []
                failures.append(f"unable to parse MRtrix clustered shells: {type(exc).__name__}:{exc}")
        bvals = np.loadtxt(input.bval, dtype=float).reshape(-1)
        bvecs = np.loadtxt(input.bvec, dtype=float)
        if bvecs.ndim != 2:
            failures.append("bvec array is not two-dimensional")
        elif bvecs.shape[0] == 3:
            pass
        elif bvecs.shape[1] == 3:
            bvecs = bvecs.T
        else:
            failures.append(f"bvec shape is not 3xN or Nx3: {bvecs.shape}")
        if bvecs.ndim == 2 and bvecs.shape[0] == 3:
            if len(bvals) != bvecs.shape[1]:
                failures.append("bval and bvec counts differ")
            if volume_count is not None and len(bvals) != volume_count:
                failures.append("gradient count differs from DWI volume count")
            norms = np.linalg.norm(bvecs, axis=0)
            nonzero = bvals > float(config["dwi_preprocessing"]["registration_reference"]["b0_threshold_s_per_mm2"])
            if np.any(np.abs(norms[nonzero] - 1.0) > 0.05):
                failures.append("nonzero-gradient vector norm outside 0.95..1.05")
            if np.any(norms[~nonzero] > 0.05):
                failures.append("b0 gradient vector norm exceeds 0.05")
            try:
                shell_selection = select_model_shells(
                    shell_centroids,
                    b0_threshold=float(config["dwi_preprocessing"]["registration_reference"]["b0_threshold_s_per_mm2"]),
                    fod_target=int(config["fod"]["target_nonzero_shell_s_per_mm2"]),
                    fod_tolerance=int(config["fod"]["shell_tolerance_s_per_mm2"]),
                    tensor_maximum=int(config["tensor"]["maximum_nonzero_shell_s_per_mm2"]),
                )
            except Exception as exc:
                failures.append(f"model shell selection failed: {type(exc).__name__}:{exc}")
                shell_selection = {
                    "observed_nonzero_shells": [],
                    "fod_shells": [],
                    "tensor_shells": [],
                    "selection_uses_diagnosis_labels": False,
                }
            selected_fod_shell = (
                shell_selection["fod_shells"][1]
                if len(shell_selection["fod_shells"]) == 2
                else None
            )
            if shell_centroids:
                centroid_array = np.asarray(shell_centroids, dtype=float)
                assigned_shells = centroid_array[
                    np.argmin(np.abs(bvals[:, None] - centroid_array[None, :]), axis=1)
                ]
            else:
                assigned_shells = np.full_like(bvals, np.nan)
            fod_mask = np.isclose(assigned_shells, selected_fod_shell, atol=1e-6) if selected_fod_shell is not None else np.zeros_like(nonzero)
            directions = bvecs[:, fod_mask].T
            direction_norms = np.linalg.norm(directions, axis=1)
            if np.any(direction_norms <= 0.05):
                unique_directions = 0
                failures.append("one or more selected FOD gradients has a near-zero direction vector")
            else:
                unit_directions = directions / direction_norms[:, None]
                canonical = unit_directions.copy()
                for index, vector in enumerate(canonical):
                    first_nonzero = np.flatnonzero(np.abs(vector) > 1e-8)
                    if first_nonzero.size and vector[first_nonzero[0]] < 0:
                        canonical[index] *= -1.0
                unique_directions = int(len(np.unique(np.round(canonical, decimals=4), axis=0)))
            if unique_directions < int(config["fod"]["minimum_unique_nonzero_directions"]):
                failures.append(f"too few selected FOD directions: {unique_directions}")
            tensor_mask = np.any(
                np.isclose(
                    assigned_shells[:, None],
                    np.asarray(shell_selection["tensor_shells"][1:], dtype=float)[None, :],
                    atol=1e-6,
                ),
                axis=1,
            ) if len(shell_selection["tensor_shells"]) > 1 else np.zeros_like(nonzero)
            tensor_directions = bvecs[:, tensor_mask].T
            tensor_norms = np.linalg.norm(tensor_directions, axis=1)
            if np.any(tensor_norms <= 0.05):
                tensor_unique_directions = 0
                failures.append("one or more tensor gradients has a near-zero direction vector")
            else:
                tensor_unit = tensor_directions / tensor_norms[:, None]
                tensor_canonical = tensor_unit.copy()
                for index, vector in enumerate(tensor_canonical):
                    first_nonzero = np.flatnonzero(np.abs(vector) > 1e-8)
                    if first_nonzero.size and vector[first_nonzero[0]] < 0:
                        tensor_canonical[index] *= -1.0
                tensor_unique_directions = int(
                    len(np.unique(np.round(tensor_canonical, decimals=4), axis=0))
                )
            if tensor_unique_directions < int(config["tensor"]["minimum_noncollinear_nonzero_directions"]):
                failures.append(f"too few tensor directions: {tensor_unique_directions}")
            try:
                tensor_design = tensor_design_diagnostics(tensor_directions)
                if tensor_design["rank"] != tensor_design["required_rank"]:
                    failures.append(
                        f"tensor design rank is {tensor_design['rank']}, required 6"
                    )
            except Exception as exc:
                tensor_design = {
                    "rank": 0,
                    "required_rank": 6,
                    "condition_number": None,
                    "singular_values": [],
                    "row_count": int(len(tensor_directions)),
                }
                failures.append(f"tensor design diagnostic failed: {type(exc).__name__}:{exc}")
            lmax = int(config["fod"]["lmax"])
            required_coefficients = (lmax + 1) * (lmax + 2) // 2
            if lmax < 2 or lmax % 2:
                failures.append(f"lmax must be an even integer of at least 2: {lmax}")
            if unique_directions < required_coefficients:
                failures.append(
                    f"{unique_directions} unique directions cannot support {required_coefficients} even-SH coefficients at lmax {lmax}"
                )
        else:
            shell_selection = {
                "observed_nonzero_shells": [],
                "fod_shells": [],
                "tensor_shells": [],
                "selection_uses_diagnosis_labels": False,
            }
            unique_directions = 0
            tensor_unique_directions = 0
            tensor_design = {
                "rank": 0,
                "required_rank": 6,
                "condition_number": None,
                "singular_values": [],
                "row_count": 0,
            }
        record = {
            "status": "PASS" if not failures else "FAIL",
            "unit": wildcards.unit,
            "volume_count": volume_count,
            "gradient_count": int(len(bvals)),
            "nonzero_shells": shell_selection["observed_nonzero_shells"],
            "mrtrix_clustered_shell_centroids_s_per_mm2": shell_centroids,
            "mrtrix_clustered_shell_sizes": shell_sizes,
            "selected_fod_shells": shell_selection["fod_shells"],
            "selected_tensor_shells": shell_selection["tensor_shells"],
            "selected_fod_shell_sizes": [
                shell_sizes[shell_centroids.index(value)]
                for value in shell_selection["fod_shells"]
            ] if shell_centroids and shell_selection["fod_shells"] else [],
            "selected_tensor_shell_sizes": [
                shell_sizes[shell_centroids.index(value)]
                for value in shell_selection["tensor_shells"]
            ] if shell_centroids and shell_selection["tensor_shells"] else [],
            "shell_selection_uses_diagnosis_labels": shell_selection["selection_uses_diagnosis_labels"],
            "selected_fod_unique_direction_count": unique_directions,
            "selected_tensor_unique_direction_count": tensor_unique_directions,
            "selected_tensor_design": tensor_design,
            "mrtrix_bzero_threshold_s_per_mm2": BZERO_THRESHOLD,
            "lmax": int(config["fod"]["lmax"]),
            "even_sh_coefficient_count": (int(config["fod"]["lmax"]) + 1) * (int(config["fod"]["lmax"]) + 2) // 2,
            "failures": failures,
            "commands": [" ".join(command)] + [" ".join(cmd) for cmd in shell_commands.values()],
        }
        Path(log[0]).write_text(
            "$ " + " ".join(command) + "\n" + completed.stdout + completed.stderr
            + "\n".join(
                "$ " + " ".join(shell_commands[name]) + "\n"
                + result.stdout + result.stderr
                for name, result in shell_results.items()
            )
            + json.dumps(record, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if failures:
            raise ValueError("Gradient contract failed: " + "; ".join(failures))
        tmp = Path(str(output[0]) + ".partial")
        tmp.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        tmp.replace(output[0])


rule dwi_denoise:
    input:
        dwi=rules.normalize_dwi_source.output.dwi,
        contract=rules.gradient_contract.output,
    output:
        dwi=subject_path("{unit}", "01_dwi", "dwi_denoised.mif"),
        noise=subject_path("{unit}", "01_dwi", "noise.mif"),
    log:
        subject_log("{unit}", "01_dwidenoise.log"),
    threads: 4
    shell:
        r"""
        set -euo pipefail
        mkdir -p "$(dirname {output.dwi:q})" "$(dirname {log:q})"
        export PATH={TOOL_PATH:q}
        dwidenoise {input.dwi:q} {output.dwi:q} -noise {output.noise:q} -nthreads {threads} \
          > {log:q} 2>&1
        """


rule dwi_degibbs:
    input:
        rules.dwi_denoise.output.dwi,
    output:
        subject_path("{unit}", "01_dwi", "dwi_denoised_degibbs.mif"),
    log:
        subject_log("{unit}", "01_mrdegibbs.log"),
    threads: 4
    shell:
        r"""
        set -euo pipefail
        mkdir -p "$(dirname {output:q})" "$(dirname {log:q})"
        export PATH={TOOL_PATH:q}
        mrdegibbs {input:q} {output:q} -nthreads {threads} > {log:q} 2>&1
        """


rule dwi_motion_eddy:
    input:
        dwi=rules.dwi_degibbs.output,
        source_metadata=rules.normalize_dwi_source.output.metadata,
        fsl_cpu_path=ancient(rules.execution_preflight.output.fsl_cpu_path),
    output:
        dwi=subject_path("{unit}", "01_dwi", "dwi_preproc.mif"),
        eddy_qc=directory(subject_path("{unit}", "01_dwi", "eddy_qc")),
    params:
        eddy_options="--slm=linear --data_is_shelled --repol --cnr_maps --residuals",
    log:
        subject_log("{unit}", "01_dwifslpreproc.log"),
    threads: 16
    shell:
        r"""
        set -euo pipefail
        mkdir -p "$(dirname {output.dwi:q})" "$(dirname {log:q})"
        export PATH={MRTRIX_BIN:q}:/usr/bin:/bin:{input.fsl_cpu_path:q}:{ANTS_BIN:q}:{C3D_AFFINE_TOOL.parent:q}
        export FSLDIR=/home/ec2-user/fsl
        export FSLOUTPUTTYPE=NIFTI_GZ
        export DWIFSLPREPROC_FORCE_CPU=1
        export DWIFSLPREPROC_NO_CPU_FALLBACK=1
        metadata_values="$({config[environment][locked_paths][workflow_python]:q} -c \
          'import json,sys; d=json.load(open(sys.argv[1])); pe=d.get("PhaseEncodingDirection"); ro=d.get("TotalReadoutTime"); assert pe in {{"i","i-","j","j-","k","k-"}}; ro=float(ro); assert 0.0 < ro < 1.0; print(pe, format(ro, ".12g"))' \
          {input.source_metadata:q})"
        set -- $metadata_values
        test "$#" -eq 2
        pe="$1"
        readout="$2"
        /usr/bin/python3.9 {MRTRIX_BIN:q}/dwifslpreproc {input.dwi:q} {output.dwi:q} \
          -rpe_none -pe_dir "$pe" -readout_time "$readout" \
          -config BZeroThreshold {BZERO_THRESHOLD} \
          -eddy_options {params.eddy_options:q} -eddyqc_all {output.eddy_qc:q} \
          -nthreads {threads} -debug > {log:q} 2>&1
        grep -F "eddy_cpu" {log:q} >/dev/null
        if grep -F "eddy_cuda" {log:q} >/dev/null; then
          printf 'FAIL: CUDA eddy appeared in the locked CPU execution log\n' >&2
          exit 1
        fi
        """


rule dwi_bias_correct:
    input:
        rules.dwi_motion_eddy.output.dwi,
    output:
        subject_path("{unit}", "01_dwi", "dwi_preproc_biascorr.mif"),
    log:
        subject_log("{unit}", "01_dwibiascorrect_ants.log"),
    threads: 8
    shell:
        r"""
        set -euo pipefail
        mkdir -p "$(dirname {output:q})" "$(dirname {log:q})"
        export PATH={TOOL_PATH:q}
        dwibiascorrect ants {input:q} {output:q} -nthreads {threads} > {log:q} 2>&1
        """


rule mean_b0:
    input:
        rules.dwi_bias_correct.output,
    output:
        subject_path("{unit}", "01_dwi", "mean_b0.mif"),
    log:
        subject_log("{unit}", "01_mean_b0.log"),
    threads: 4
    shell:
        r"""
        set -euo pipefail
        mkdir -p "$(dirname {output:q})" "$(dirname {log:q})"
        export PATH={TOOL_PATH:q}
        dwiextract {input:q} - -bzero -config BZeroThreshold {BZERO_THRESHOLD} -nthreads {threads} 2>> {log:q} \
          | mrmath - mean {output:q} -axis 3 -nthreads {threads} >> {log:q} 2>&1
        """


rule dwi_brain_mask:
    input:
        rules.dwi_bias_correct.output,
    output:
        subject_path("{unit}", "01_dwi", "dwi_brain_mask.mif"),
    log:
        subject_log("{unit}", "01_dwi2mask.log"),
    threads: 4
    shell:
        r"""
        set -euo pipefail
        mkdir -p "$(dirname {output:q})" "$(dirname {log:q})"
        export PATH={TOOL_PATH:q}
        dwi2mask {input:q} {output:q} -config BZeroThreshold {BZERO_THRESHOLD} -nthreads {threads} > {log:q} 2>&1
        """
