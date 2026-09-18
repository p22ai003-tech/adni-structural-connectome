"""Quantitative spatial gates and blinded visual-QC hand-off."""


rule spatial_qc_images:
    input:
        five_tt=rules.five_tt_dwi.output.five_tt,
        dwi_mask=rules.dwi_brain_mask.output,
        one_mm_reference=rules.b0_one_mm_world_grid.output,
    output:
        five_tt_mask=subject_path("{unit}", "06_preflight", "5tt_mask.nii.gz"),
        dwi_mask=subject_path("{unit}", "06_preflight", "dwi_mask.nii.gz"),
        dwi_mask_one_mm=subject_path("{unit}", "06_preflight", "dwi_mask_1mm.nii.gz"),
    log:
        subject_log("{unit}", "06_spatial_qc_images.log"),
    threads: 4
    shell:
        r"""
        set -euo pipefail
        mkdir -p "$(dirname {output.five_tt_mask:q})" "$(dirname {log:q})"
        export PATH={TOOL_PATH:q}
        five_tt_sum="$(dirname {output.five_tt_mask:q})/5tt_sum.partial.mif"
        trap 'rm -f "$five_tt_sum"' EXIT
        mrmath {input.five_tt:q} sum "$five_tt_sum" -axis 3 -nthreads {threads} > {log:q} 2>&1
        mrthreshold "$five_tt_sum" {output.five_tt_mask:q} -abs 0.5 >> {log:q} 2>&1
        mrconvert {input.dwi_mask:q} {output.dwi_mask:q} >> {log:q} 2>&1
        mrtransform {input.dwi_mask:q} {output.dwi_mask_one_mm:q} \
          -template {input.one_mm_reference:q} -interp nearest -nthreads {threads} >> {log:q} 2>&1
        """


rule spatial_quantitative_qc:
    input:
        five_tt_mask=rules.spatial_qc_images.output.five_tt_mask,
        dwi_mask=rules.spatial_qc_images.output.dwi_mask,
        dwi_mask_one_mm=rules.spatial_qc_images.output.dwi_mask_one_mm,
        atlas=rules.aal3_to_dwi_single_resample.output,
        atlas_contract=rules.atlas_contract_qc.output,
        bbr=rules.b0_to_t1_bbr.output.matrix,
    output:
        subject_path("{unit}", "06_preflight", "spatial_quantitative_qc.json"),
    log:
        subject_log("{unit}", "06_spatial_quantitative_qc.log"),
    run:
        import nibabel as nib
        import numpy as np

        Path(output[0]).parent.mkdir(parents=True, exist_ok=True)
        Path(log[0]).parent.mkdir(parents=True, exist_ok=True)
        failures = []
        five_tt = np.asanyarray(nib.load(input.five_tt_mask).dataobj) > 0
        dwi = np.asanyarray(nib.load(input.dwi_mask).dataobj) > 0
        if five_tt.shape != dwi.shape:
            failures.append(f"5TT/DWI-mask shape mismatch: {five_tt.shape} vs {dwi.shape}")
            dice = 0.0
        else:
            denominator = int(five_tt.sum() + dwi.sum())
            dice = 2.0 * float(np.logical_and(five_tt, dwi).sum()) / denominator if denominator else 0.0
        minimum_dice = float(config["registration"]["quantitative_qc"]["minimum_5tt_to_dwi_mask_dice"])
        if dice < minimum_dice:
            failures.append(f"5TT/DWI-mask Dice {dice:.6f} below {minimum_dice}")

        atlas = np.asanyarray(nib.load(input.atlas).dataobj) > 0
        dwi_one_mm = np.asanyarray(nib.load(input.dwi_mask_one_mm).dataobj) > 0
        if atlas.shape != dwi_one_mm.shape:
            failures.append(f"atlas/mask shape mismatch: {atlas.shape} vs {dwi_one_mm.shape}")
            atlas_inside = 0.0
        else:
            atlas_voxels = int(atlas.sum())
            atlas_inside = float(np.logical_and(atlas, dwi_one_mm).sum()) / atlas_voxels if atlas_voxels else 0.0
        minimum_inside = float(config["registration"]["quantitative_qc"]["minimum_atlas_label_inside_brain_fraction"])
        if atlas_inside < minimum_inside:
            failures.append(f"atlas-inside-brain fraction {atlas_inside:.6f} below {minimum_inside}")

        matrix = np.loadtxt(input.bbr, dtype=float)
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
            failures.append("BBR affine is not a finite 4x4 matrix")
            determinant = None
        else:
            determinant = float(np.linalg.det(matrix[:3, :3]))
            if abs(determinant) < 1e-6:
                failures.append("BBR affine is singular or near-singular")
        record = {
            "status": "PASS" if not failures else "FAIL",
            "unit": wildcards.unit,
            "five_tt_dwi_mask_dice": dice,
            "minimum_five_tt_dwi_mask_dice": minimum_dice,
            "atlas_inside_brain_fraction": atlas_inside,
            "minimum_atlas_inside_brain_fraction": minimum_inside,
            "bbr_affine_determinant": determinant,
            "failures": failures,
        }
        Path(log[0]).write_text(
            "internal: quantitative spatial contract\n" + json.dumps(record, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if failures:
            raise ValueError("Spatial contract failed: " + "; ".join(failures))
        tmp = Path(str(output[0]) + ".partial")
        tmp.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        tmp.replace(output[0])


rule t1_in_b0_for_visual_qc:
    input:
        t1=rules.t1_n4_bias_correct.output.t1,
        transform=rules.invert_bbr_transform.output.mrtrix,
        reference=rules.mean_b0_nifti.output,
    output:
        subject_path("{unit}", "06_preflight", "t1_in_b0_qc.nii.gz"),
    log:
        subject_log("{unit}", "06_t1_in_b0_qc.log"),
    shell:
        r"""
        set -euo pipefail
        mkdir -p "$(dirname {output:q})" "$(dirname {log:q})"
        export PATH={TOOL_PATH:q}
        mrtransform {input.t1:q} {output:q} -linear {input.transform:q} \
          -template {input.reference:q} -interp cubic > {log:q} 2>&1
        """


rule visual_review_bundle:
    input:
        b0=rules.mean_b0_nifti.output,
        t1=rules.t1_in_b0_for_visual_qc.output,
        five_tt=rules.spatial_qc_images.output.five_tt_mask,
        atlas=rules.atlas_native_grid_qc_copy.output,
        spatial_qc=rules.spatial_quantitative_qc.output,
    output:
        b0_t1=subject_path("{unit}", "06_preflight", "review_b0_vs_t1.png"),
        b0_5tt=subject_path("{unit}", "06_preflight", "review_b0_vs_5tt.png"),
        b0_atlas=subject_path("{unit}", "06_preflight", "review_b0_vs_aal3.png"),
        index=subject_path("{unit}", "06_preflight", "visual_review_index.csv"),
    log:
        subject_log("{unit}", "06_visual_review_bundle.log"),
    shell:
        r"""
        set -euo pipefail
        mkdir -p "$(dirname {output.b0_t1:q})" "$(dirname {log:q})"
        export PATH={TOOL_PATH:q}
        export FSLDIR=/home/ec2-user/fsl
        export FSLOUTPUTTYPE=NIFTI_GZ
        slices {input.b0:q} {input.t1:q} -o {output.b0_t1:q} > {log:q} 2>&1
        slices {input.b0:q} {input.five_tt:q} -o {output.b0_5tt:q} >> {log:q} 2>&1
        slices {input.b0:q} {input.atlas:q} -o {output.b0_atlas:q} >> {log:q} 2>&1
        printf 'unit,status,b0_t1,b0_5tt,b0_atlas\n%s,PENDING,%s,%s,%s\n' \
          {wildcards.unit:q} {output.b0_t1:q} {output.b0_5tt:q} {output.b0_atlas:q} > {output.index:q}
        """


rule automated_pre_tractography_qc:
    input:
        spatial=rules.spatial_quantitative_qc.output,
        atlas=rules.atlas_contract_qc.output,
        gradients=rules.gradient_contract.output,
        wmfod=rules.mtnormalise.output.wm,
        gm=rules.mtnormalise.output.gm,
        csf=rules.mtnormalise.output.csf,
        fa=rules.tensor_metrics.output.fa,
        md=rules.tensor_metrics.output.md,
        rd=rules.tensor_metrics.output.rd,
        ad=rules.tensor_metrics.output.ad,
        five_tt=rules.five_tt_dwi.output.five_tt,
        mask=rules.dwi_brain_mask.output,
    output:
        report=subject_path("{unit}", "06_preflight", "automated_pre_tractography_qc.json"),
    log:
        subject_log("{unit}", "06_automated_pre_tractography_qc.log"),
    run:
        import subprocess

        import numpy as np

        Path(output[0]).parent.mkdir(parents=True, exist_ok=True)
        Path(log[0]).parent.mkdir(parents=True, exist_ok=True)
        wmfod_l0_path = Path(output.report).with_name("wmfod_l0_qc.mif")
        wmfod_l0_partial = Path(str(wmfod_l0_path) + ".partial")
        wmfod_l0_path.unlink(missing_ok=True)
        wmfod_l0_partial.unlink(missing_ok=True)
        failures = []
        commands = []
        for qc_path in (input.spatial, input.atlas, input.gradients):
            record = json.loads(Path(qc_path).read_text(encoding="utf-8"))
            if record.get("status") != "PASS":
                failures.append(f"upstream QC is not PASS: {qc_path}")

        l0_command = [
            str(MRTRIX_BIN / "mrconvert"),
            str(input.wmfod),
            str(wmfod_l0_partial),
            "-coord",
            "3",
            "0",
        ]
        commands.append(" ".join(l0_command))
        l0_result = subprocess.run(
            l0_command, text=True, capture_output=True, check=False
        )
        if l0_result.returncode != 0:
            failures.append("unable to extract WM-FOD l=0 coefficient")
            wmfod_l0_partial.unlink(missing_ok=True)
        else:
            wmfod_l0_partial.replace(wmfod_l0_path)

        ranges = {}
        all_coeff_command = [
            str(MRTRIX_BIN / "mrstats"),
            str(input.wmfod),
            "-mask",
            str(input.mask),
            "-output",
            "min",
            "-output",
            "max",
        ]
        commands.append(" ".join(all_coeff_command))
        all_coeff_result = subprocess.run(
            all_coeff_command, text=True, capture_output=True, check=False
        )
        try:
            coefficient_ranges = [
                float(value) for value in all_coeff_result.stdout.split()
            ]
            if (
                all_coeff_result.returncode != 0
                or not coefficient_ranges
                or any(not np.isfinite(value) for value in coefficient_ranges)
            ):
                raise ValueError("nonfinite or empty WM-FOD coefficient ranges")
            ranges["wmfod_all_sh_coefficients"] = {
                "min": min(coefficient_ranges),
                "max": max(coefficient_ranges),
                "all_finite": True,
                "nonnegative_required": False,
            }
        except Exception as exc:
            failures.append(f"WM-FOD coefficient finite check failed: {type(exc).__name__}:{exc}")

        for name, path in (
            ("wmfod_l0", wmfod_l0_path),
            ("gm", input.gm),
            ("csf", input.csf),
            ("fa", input.fa),
            ("md", input.md),
            ("rd", input.rd),
            ("ad", input.ad),
        ):
            command = [
                str(MRTRIX_BIN / "mrstats"),
                str(path),
                "-mask",
                str(input.mask),
                "-output",
                "min",
                "-output",
                "max",
            ]
            commands.append(" ".join(command))
            completed = subprocess.run(command, text=True, capture_output=True, check=False)
            if completed.returncode != 0:
                failures.append(f"mrstats failed for {name}")
                continue
            try:
                minimum, maximum = [float(value) for value in completed.stdout.split()[-2:]]
            except Exception:
                failures.append(f"unable to parse mrstats for {name}")
                continue
            ranges[name] = {"min": minimum, "max": maximum}
            if not (np.isfinite(minimum) and np.isfinite(maximum)):
                failures.append(f"non-finite range for {name}")
        tensor_range_qc = config["qc"]["tensor_edge_values"]
        failures.extend(
            assess_pre_tractography_image_ranges(
                ranges,
                nonnegative_tolerance=float(
                    config["qc"]["matrix"]["nonnegative_absolute_tolerance"]
                ),
                range_tolerance=float(tensor_range_qc["range_absolute_tolerance"]),
                fa_minimum=float(tensor_range_qc["fa_min"]),
                fa_maximum=float(tensor_range_qc["fa_max"]),
                diffusivity_minimum=float(
                    tensor_range_qc["diffusivity_min_mm2_per_s"]
                ),
                diffusivity_maximum=float(
                    tensor_range_qc["diffusivity_hard_max_mm2_per_s"]
                ),
            )
        )
        record = {
            "schema_version": "2.0.0",
            "record_type": "automated_pre_tractography_qc",
            "status": "PASS" if not failures else "FAIL",
            "unit": wildcards.unit,
            "diagnosis_labels_used": False,
            "image_ranges": ranges,
            "wmfod_qc": {
                "all_sh_coefficients_must_be_finite": True,
                "all_sh_coefficients_must_be_nonnegative": False,
                "l0_must_be_nonnegative_and_nonzero_within_mask": True,
                "l0_image_path": str(wmfod_l0_path),
                "l0_image_sha256": (
                    _sha256(wmfod_l0_path) if wmfod_l0_path.is_file() else None
                ),
            },
            "range_tolerances": {
                "nonnegative_absolute_tolerance": float(
                    config["qc"]["matrix"]["nonnegative_absolute_tolerance"]
                ),
                "physical_range_absolute_tolerance": float(
                    tensor_range_qc["range_absolute_tolerance"]
                ),
            },
            "commands": commands,
            "failures": failures,
        }
        Path(log[0]).write_text(
            "\n".join("$ " + command for command in commands) + "\n"
            + json.dumps(record, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        tmp = Path(str(output.report) + ".partial")
        tmp.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        tmp.replace(output.report)


rule tractography_preflight:
    input:
        execution=ancient(rules.execution_preflight.output.report),
        automated=rules.automated_pre_tractography_qc.output.report,
        review_index=rules.visual_review_bundle.output.index,
        human_qc=str(HUMAN_QC_MANIFEST),
    output:
        subject_path("{unit}", "06_preflight", "tractography_preflight.json"),
    log:
        subject_log("{unit}", "06_tractography_preflight.log"),
    run:
        Path(output[0]).parent.mkdir(parents=True, exist_ok=True)
        Path(log[0]).parent.mkdir(parents=True, exist_ok=True)
        failures = []
        execution = json.loads(Path(input.execution).read_text(encoding="utf-8"))
        if (
            execution.get("status") != "PASS"
            or execution.get("recipe_id") != RECIPE_ID
            or execution.get("run_context_sha256") != _sha256(RUN_CONTEXT_PATH)
            or execution.get("attempt_context_sha256")
            != _sha256(ATTEMPT_CONTEXT_PATH)
            or execution.get("resolved_run_config_sha256")
            != _sha256(RESOLVED_RUN_CONFIG)
        ):
            failures.append(
                "execution preflight is not an exact current-mode, current-attempt PASS"
            )
        automated = json.loads(Path(input.automated).read_text(encoding="utf-8"))
        if (
            automated.get("record_type") != "automated_pre_tractography_qc"
            or automated.get("status") != "PASS"
            or automated.get("unit") != str(wildcards.unit)
            or automated.get("diagnosis_labels_used") is not False
        ):
            failures.append("automated pre-tractography QC is not a diagnosis-blind PASS")

        with Path(input.review_index).open(newline="", encoding="utf-8-sig") as handle:
            index_rows = list(csv.DictReader(handle))
        if (
            len(index_rows) != 1
            or str(index_rows[0].get("unit", "")).strip() != str(wildcards.unit)
            or str(index_rows[0].get("status", "")).strip().upper() != "PENDING"
        ):
            failures.append("visual review index contract differs")

        with Path(input.human_qc).open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            fields = tuple(reader.fieldnames or ())
            forbidden = {"diagnosis", "diagnosis_at_dti", "group", "research_group"}
            if forbidden.intersection(str(field).strip().lower() for field in fields):
                failures.append("human visual-QC manifest is not diagnosis blinded")
            required = {"unit", "status", "reviewer", "reviewed_utc"}
            if not required.issubset(fields):
                failures.append("human visual-QC manifest lacks required fields")
            review_rows = [
                {str(key): str(value or "").strip() for key, value in row.items()}
                for row in reader
                if str(row.get("unit", "")).strip() == str(wildcards.unit)
            ]
        review = review_rows[0] if len(review_rows) == 1 else {}
        if len(review_rows) != 1:
            failures.append("human visual-QC manifest requires exactly one row for unit")
        elif (
            review.get("status", "").upper() != "PASS"
            or not review.get("reviewer")
            or not review.get("reviewed_utc")
        ):
            failures.append("human visual QC is not a complete PASS")

        record = {
            "schema_version": "2.0.0",
            "record_type": "tractography_preflight",
            "status": "PASS" if not failures else "FAIL",
            "unit": str(wildcards.unit),
            "diagnosis_labels_used": False,
            "automated_pre_tractography_qc": {
                "path": str(Path(input.automated).resolve()),
                "sha256": _sha256(input.automated),
            },
            "visual_review_index": {
                "path": str(Path(input.review_index).resolve()),
                "sha256": _sha256(input.review_index),
            },
            "human_qc_manifest": {
                "path": str(Path(input.human_qc).resolve()),
                "sha256": _sha256(input.human_qc),
            },
            "human_review": review,
            "failures": failures,
        }
        Path(log[0]).write_text(
            "internal: automated and blinded-human tractography continuation gate\n"
            + json.dumps(record, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        if failures:
            raise ValueError("Tractography preflight failed: " + "; ".join(failures))
        tmp = Path(str(output[0]) + ".partial")
        tmp.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        tmp.replace(output[0])
