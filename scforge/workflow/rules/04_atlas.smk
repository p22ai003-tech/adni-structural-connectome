"""Single-resampling AAL3v1-to-DWI atlas route and label contract."""


rule b0_one_mm_world_grid:
    input:
        rules.mean_b0_nifti.output,
    output:
        subject_path("{unit}", "04_atlas", "b0_1mm_world_grid.nii.gz"),
    log:
        subject_log("{unit}", "04_b0_1mm_grid.log"),
    shell:
        r"""
        set -euo pipefail
        mkdir -p "$(dirname {output:q})" "$(dirname {log:q})"
        export PATH={TOOL_PATH:q}
        mrgrid {input:q} regrid -voxel 1.0 {output:q} > {log:q} 2>&1
        """


rule aal3_to_dwi_single_resample:
    input:
        atlas=ancient(str(ATLAS_IMAGE)),
        reference=rules.b0_one_mm_world_grid.output,
        t1_to_b0=rules.invert_bbr_transform.output.itk,
        warp=rules.mni_to_t1_nonlinear.output.warp,
        affine=rules.mni_to_t1_nonlinear.output.affine,
    output:
        subject_path("{unit}", "04_atlas", "aal3_nodes_166_dwi_1mm.nii.gz"),
    log:
        subject_log("{unit}", "04_ants_compose_atlas.log"),
    threads: 8
    shell:
        r"""
        set -euo pipefail
        mkdir -p "$(dirname {output:q})" "$(dirname {log:q})"
        export PATH={TOOL_PATH:q}
        export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS={threads}
        antsApplyTransforms -d 3 -i {input.atlas:q} -r {input.reference:q} \
          -o {output:q} -n GenericLabel \
          -t {input.t1_to_b0:q} -t {input.warp:q} -t {input.affine:q} \
          > {log:q} 2>&1
        """


rule atlas_contract_qc:
    input:
        atlas=rules.aal3_to_dwi_single_resample.output,
        reference=rules.b0_one_mm_world_grid.output,
        node_table=ancient(str(ATLAS_NODE_TABLE)),
        node_map=ancient(str(ATLAS_NODE_MAP)),
    output:
        subject_path("{unit}", "04_atlas", "atlas_contract_qc.json"),
    log:
        subject_log("{unit}", "04_atlas_contract_qc.log"),
    run:
        import nibabel as nib
        import numpy as np

        Path(output[0]).parent.mkdir(parents=True, exist_ok=True)
        Path(log[0]).parent.mkdir(parents=True, exist_ok=True)
        image = nib.load(input.atlas)
        reference = nib.load(input.reference)
        data = np.asanyarray(image.dataobj)
        rounded = np.rint(data)
        labels = sorted(int(value) for value in np.unique(rounded) if value > 0)
        expected = list(range(1, int(config["atlas"]["expected_nodes"]) + 1))
        counts = {str(label): int(np.count_nonzero(rounded == label)) for label in expected}
        failures = []
        if not np.allclose(data, rounded, atol=1e-6):
            failures.append("atlas contains non-integer values")
        if labels != expected:
            failures.append(
                f"atlas labels differ from 1..{len(expected)}: found {len(labels)} labels"
            )
        minimum = int(config["atlas"]["minimum_voxels_per_label_on_1mm_grid"])
        support_qc = assess_atlas_label_support(
            counts,
            expected_nodes=len(expected),
            hard_minimum_voxels=minimum,
        )
        failures.extend(support_qc["failures"])
        if image.shape != reference.shape:
            failures.append(f"atlas/reference shape mismatch: {image.shape} vs {reference.shape}")
        if not np.allclose(image.affine, reference.affine, atol=1e-5):
            failures.append("atlas/reference affine mismatch")
        record = {
            "status": "PASS" if not failures else "FAIL",
            "unit": wildcards.unit,
            "expected_nodes": len(expected),
            "labels_found": labels,
            "voxel_counts": counts,
            "voxel_support_qc": support_qc,
            "per_node_voxel_counts_recorded": True,
            "stronger_label_size_threshold_used_as_automatic_exclusion": False,
            "atlas_sha256": _sha256(input.atlas),
            "node_table_sha256": _sha256(input.node_table),
            "node_map_sha256": _sha256(input.node_map),
            "failures": failures,
        }
        Path(log[0]).write_text(
            "internal: validate AAL3 integer labels, affine, and physical support\n"
            + json.dumps(record, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if failures:
            raise ValueError("Atlas contract failed: " + "; ".join(failures))
        tmp = Path(str(output[0]) + ".partial")
        tmp.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        tmp.replace(output[0])


rule atlas_native_grid_qc_copy:
    input:
        atlas=rules.aal3_to_dwi_single_resample.output,
        reference=rules.mean_b0_nifti.output,
        contract=rules.atlas_contract_qc.output,
    output:
        subject_path("{unit}", "04_atlas", "aal3_nodes_166_native_b0_qc_only.nii.gz"),
    log:
        subject_log("{unit}", "04_atlas_native_grid_qc_copy.log"),
    shell:
        r"""
        set -euo pipefail
        mkdir -p "$(dirname {output:q})" "$(dirname {log:q})"
        export PATH={TOOL_PATH:q}
        mrtransform {input.atlas:q} {output:q} -template {input.reference:q} -interp nearest \
          > {log:q} 2>&1
        """
