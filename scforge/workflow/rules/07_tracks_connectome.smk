"""Locked 10M ACT tractography, SIFT2, and nine connectome matrices."""


def _subject_rng_seed(wildcards):
    # One definition of the per-subject seed, shared with registration; see
    # subject_seed() in the Snakefile.
    return subject_seed(wildcards.unit)


def _subject_rng_token(wildcards):
    row = ROW_BY_UNIT[str(wildcards.unit)]
    return f"{row['dti_source_id']}|{row['dti_raw_bundle_sha256']}|{RECIPE_ID}"


def _tensor_image(wildcards):
    return subject_path(wildcards.unit, "05_model", f"{wildcards.metric}.mif")


rule tractography_10m:
    input:
        preflight=rules.tractography_preflight.output,
        fod=rules.mtnormalise.output.wm,
        five_tt=rules.five_tt_dwi.output.five_tt,
        dwi=rules.dwi_bias_correct.output,
    output:
        tracks=subject_path("{unit}", "07_tractography", "tracks_10m.tck"),
        metadata=subject_path("{unit}", "07_tractography", "tractography_parameters.json"),
    params:
        rng_seed=_subject_rng_seed,
        rng_token=_subject_rng_token,
        requested=int(config["tractography"]["select_streamlines"]),
        max_seeds=int(config["tractography"]["maximum_seed_attempts"]),
        min_length=float(config["tractography"]["minimum_length_mm"]),
        max_length=float(config["tractography"]["maximum_length_mm"]),
        cutoff=float(config["tractography"]["cutoff"]),
        angle=float(config["tractography"]["maximum_angle_degrees"]),
    log:
        subject_log("{unit}", "07_tckgen_10m.log"),
    threads: int(config["tractography"]["nthreads_per_subject"])
    shell:
        r"""
        set -euo pipefail
        mkdir -p "$(dirname {output.tracks:q})" "$(dirname {log:q})"
        export PATH={TOOL_PATH:q}
        export MRTRIX_RNG_SEED={params.rng_seed}
        step="$(mrinfo {input.dwi:q} -spacing | awk '{{m=$1; if ($2<m) m=$2; if ($3<m) m=$3; printf "%.8f", m/2.0}}')"
        partial="$(dirname {output.tracks:q})/tracks_10m.partial.tck"
        trap 'rm -f "$partial"' EXIT
        tckgen {input.fod:q} "$partial" -algorithm iFOD2 -act {input.five_tt:q} \
          -backtrack -crop_at_gmwmi -seed_dynamic {input.fod:q} \
          -select {params.requested} -seeds {params.max_seeds} \
          -minlength {params.min_length} -maxlength {params.max_length} \
          -cutoff {params.cutoff} -step "$step" -angle {params.angle} \
          -nthreads {threads} > {log:q} 2>&1
        actual="$(tckinfo "$partial" -count 2>> {log:q} | awk '$1=="count:" {{print $2; exit}}')"
        if [ "$actual" != "{params.requested}" ]; then
          printf 'FAIL: requested %s streamlines, generated %s\n' {params.requested:q} "$actual" >> {log:q}
          exit 1
        fi
        mv "$partial" {output.tracks:q}
        metadata_partial={output.metadata:q}.partial
        printf '{{"status":"PASS","unit":"%s","algorithm":"iFOD2","rng_seed_token":"%s","rng_seed":%s,"actual_step_size_mm":%s,"requested_streamlines":%s,"actual_streamlines":%s,"maximum_seed_attempts":%s,"minimum_length_mm":%s,"maximum_length_mm":%s,"cutoff":%s,"maximum_angle_degrees":%s,"act":true,"backtrack":true,"crop_at_gmwmi":true,"seeding":"seed_dynamic"}}\n' \
          {wildcards.unit:q} {params.rng_token:q} {params.rng_seed:q} "$step" \
          {params.requested:q} "$actual" {params.max_seeds:q} {params.min_length:q} \
          {params.max_length:q} {params.cutoff:q} {params.angle:q} > "$metadata_partial"
        mv "$metadata_partial" {output.metadata:q}
        """


rule sift2_weights:
    input:
        tracks=rules.tractography_10m.output.tracks,
        fod=rules.mtnormalise.output.wm,
        five_tt=rules.five_tt_dwi.output.five_tt,
    output:
        weights=subject_path("{unit}", "07_tractography", "sift2_weights.txt"),
        mu=subject_path("{unit}", "07_tractography", "sift2_mu.txt"),
        iterations=subject_path("{unit}", "07_tractography", "sift2_iterations.csv"),
    log:
        subject_log("{unit}", "07_tcksift2.log"),
    threads: int(config["tractography"]["nthreads_per_subject"])
    shell:
        r"""
        set -euo pipefail
        mkdir -p "$(dirname {output.weights:q})" "$(dirname {log:q})"
        export PATH={TOOL_PATH:q}
        tcksift2 {input.tracks:q} {input.fod:q} {output.weights:q} \
          -act {input.five_tt:q} -out_mu {output.mu:q} -csv {output.iterations:q} \
          -reg_tikhonov 0.0 -reg_tv 0.1 -min_td_frac 0.1 \
          -min_iters 10 -max_iters 1000 -nthreads {threads} > {log:q} 2>&1
        test "$(wc -l < {output.weights:q})" -eq {config[tractography][select_streamlines]}
        """


rule connectome_count:
    input:
        tracks=rules.tractography_10m.output.tracks,
        atlas=rules.aal3_to_dwi_single_resample.output,
    output:
        matrix=matrix_path("{unit}", "count"),
        assignments=subject_path("{unit}", "07_connectome", "assignments.csv"),
    log:
        subject_log("{unit}", "07_connectome_count.log"),
    threads: 8
    shell:
        r"""
        set -euo pipefail
        mkdir -p "$(dirname {output.matrix:q})" "$(dirname {log:q})"
        export PATH={TOOL_PATH:q}
        tck2connectome {input.tracks:q} {input.atlas:q} {output.matrix:q} \
          -assignment_radial_search 4 -symmetric -zero_diagonal -stat_edge sum \
          -out_assignments {output.assignments:q} -nthreads {threads} > {log:q} 2>&1
        """


rule connectome_fd_sum:
    input:
        tracks=rules.tractography_10m.output.tracks,
        atlas=rules.aal3_to_dwi_single_resample.output,
        weights=rules.sift2_weights.output.weights,
    output:
        matrix_path("{unit}", "fd_sum"),
    log:
        subject_log("{unit}", "07_connectome_fd_sum.log"),
    threads: 8
    shell:
        r"""
        set -euo pipefail
        mkdir -p "$(dirname {output:q})" "$(dirname {log:q})"
        export PATH={TOOL_PATH:q}
        tck2connectome {input.tracks:q} {input.atlas:q} {output:q} \
          -assignment_radial_search 4 -symmetric -zero_diagonal -stat_edge sum \
          -tck_weights_in {input.weights:q} -nthreads {threads} > {log:q} 2>&1
        """


rule connectome_len_mean:
    input:
        tracks=rules.tractography_10m.output.tracks,
        atlas=rules.aal3_to_dwi_single_resample.output,
    output:
        matrix_path("{unit}", "len_mean"),
    log:
        subject_log("{unit}", "07_connectome_len_mean.log"),
    threads: 8
    shell:
        r"""
        set -euo pipefail
        mkdir -p "$(dirname {output:q})" "$(dirname {log:q})"
        export PATH={TOOL_PATH:q}
        tck2connectome {input.tracks:q} {input.atlas:q} {output:q} \
          -assignment_radial_search 4 -symmetric -zero_diagonal \
          -scale_length -stat_edge mean -nthreads {threads} > {log:q} 2>&1
        """


rule connectome_invlen_mean:
    input:
        tracks=rules.tractography_10m.output.tracks,
        atlas=rules.aal3_to_dwi_single_resample.output,
    output:
        matrix_path("{unit}", "invlen_mean"),
    log:
        subject_log("{unit}", "07_connectome_invlen_mean.log"),
    threads: 8
    shell:
        r"""
        set -euo pipefail
        mkdir -p "$(dirname {output:q})" "$(dirname {log:q})"
        export PATH={TOOL_PATH:q}
        tck2connectome {input.tracks:q} {input.atlas:q} {output:q} \
          -assignment_radial_search 4 -symmetric -zero_diagonal \
          -scale_invlength -stat_edge mean -nthreads {threads} > {log:q} 2>&1
        """


rule tensor_streamline_samples:
    input:
        tracks=rules.tractography_10m.output.tracks,
        image=_tensor_image,
    output:
        subject_path("{unit}", "07_connectome", "samples", "{metric}_per_streamline.csv"),
    log:
        subject_log("{unit}", "07_tcksample_{metric}.log"),
    threads: 8
    shell:
        r"""
        set -euo pipefail
        mkdir -p "$(dirname {output:q})" "$(dirname {log:q})"
        export PATH={TOOL_PATH:q}
        tcksample {input.tracks:q} {input.image:q} {output:q} -stat_tck mean -nthreads {threads} \
          > {log:q} 2>&1
        test "$(wc -l < {output:q})" -eq {config[tractography][select_streamlines]}
        """


rule tensor_connectome:
    input:
        tracks=rules.tractography_10m.output.tracks,
        atlas=rules.aal3_to_dwi_single_resample.output,
        samples=rules.tensor_streamline_samples.output,
    output:
        subject_path("{unit}", "07_connectome", "matrices", "{metric}_mean.csv"),
    log:
        subject_log("{unit}", "07_connectome_{metric}_mean.log"),
    threads: 8
    shell:
        r"""
        set -euo pipefail
        mkdir -p "$(dirname {output:q})" "$(dirname {log:q})"
        export PATH={TOOL_PATH:q}
        tck2connectome {input.tracks:q} {input.atlas:q} {output:q} \
          -assignment_radial_search 4 -symmetric -zero_diagonal \
          -scale_file {input.samples:q} -stat_edge mean -nthreads {threads} > {log:q} 2>&1
        """


rule count_invnodevol:
    input:
        count_matrix=rules.connectome_count.output.matrix,
        atlas=rules.aal3_to_dwi_single_resample.output,
        atlas_qc=rules.atlas_contract_qc.output,
    output:
        matrix=matrix_path("{unit}", "count_invnodevol"),
        node_volumes=subject_path("{unit}", "07_connectome", "node_volumes.csv"),
    log:
        subject_log("{unit}", "07_count_invnodevol.log"),
    run:
        import csv
        import nibabel as nib
        import numpy as np

        Path(output.matrix).parent.mkdir(parents=True, exist_ok=True)
        Path(log[0]).parent.mkdir(parents=True, exist_ok=True)
        count = np.loadtxt(input.count_matrix, delimiter=",")
        image = nib.load(input.atlas)
        labels = np.rint(np.asanyarray(image.dataobj)).astype(int)
        voxel_volume = abs(float(np.linalg.det(image.affine[:3, :3])))
        volumes = np.array(
            [np.count_nonzero(labels == node) * voxel_volume for node in range(1, 167)],
            dtype=float,
        )
        if count.shape != (166, 166):
            raise ValueError(f"count shape is {count.shape}, expected 166x166")
        if np.any(volumes <= 0):
            raise ValueError("one or more AAL3 nodes has non-positive physical volume")
        denominator = volumes[:, None] + volumes[None, :]
        derived = 2.0 * count / denominator
        np.fill_diagonal(derived, 0.0)
        matrix_tmp = Path(str(output.matrix) + ".partial")
        np.savetxt(matrix_tmp, derived, delimiter=",", fmt="%.12g")
        matrix_tmp.replace(output.matrix)
        volume_tmp = Path(str(output.node_volumes) + ".partial")
        with volume_tmp.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["node_index", "voxel_count", "voxel_volume_mm3", "node_volume_mm3"])
            for node, volume in enumerate(volumes, start=1):
                writer.writerow([node, int(np.count_nonzero(labels == node)), voxel_volume, volume])
        volume_tmp.replace(output.node_volumes)
        record = {
            "unit": wildcards.unit,
            "formula": "2*count_ij/(node_volume_i_mm3+node_volume_j_mm3)",
            "voxel_volume_mm3": voxel_volume,
            "minimum_node_volume_mm3": float(volumes.min()),
            "maximum_node_volume_mm3": float(volumes.max()),
        }
        Path(log[0]).write_text(
            "internal: derive count_invnodevol from count and physical AAL3 volumes\n"
            + json.dumps(record, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
