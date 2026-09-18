"""Diagnosis-blind 3M/5M/10M and independent-seed phase-B stability DAG."""

import shlex
import subprocess


STABILITY_RUNS = (
    "primary_3m",
    "primary_5m",
    "primary_10m",
    "independent_10m",
)
STABILITY_DERIVED_RUNS = (
    "primary_3m",
    "primary_5m",
    "independent_10m",
)
STABILITY_STREAMLINE_COUNTS = {
    "primary_3m": 3_000_000,
    "primary_5m": 5_000_000,
    "primary_10m": 10_000_000,
    "independent_10m": 10_000_000,
}
STABILITY_MATRIX_NAMES = (
    "count",
    "fd_sum",
    "fa_mean",
    "md_mean",
    "rd_mean",
    "ad_mean",
)
STABILITY_BUILDER = (
    WORKFLOW_DIR / "build_phase_b_stability_manifest.py"
).resolve()
STABILITY_VALIDATOR = (
    PROJECT_ROOT / "research_audit/validate_phase_b_recipe_stability.py"
).resolve()
THESIS_RELEASE_CONTRACT = (
    PROJECT_ROOT
    / "research_audit/outputs/thesis_grade_530_release_contract_v1/contract.json"
).resolve()
for _stability_source in (
    STABILITY_BUILDER,
    STABILITY_VALIDATOR,
    THESIS_RELEASE_CONTRACT,
):
    if not _stability_source.is_file():
        raise FileNotFoundError(_stability_source)


def stability_path(unit, stability_run, *parts):
    return subject_path(unit, "09_stability", stability_run, *parts)


def stability_tracks(wildcards):
    run_id = str(wildcards.stability_run)
    if run_id == "primary_10m":
        return rules.tractography_10m.output.tracks
    return stability_path(wildcards.unit, run_id, "tracks.tck")


def stability_metadata(wildcards):
    run_id = str(wildcards.stability_run)
    if run_id == "primary_10m":
        return rules.tractography_10m.output.metadata
    return stability_path(wildcards.unit, run_id, "tractography_parameters.json")


def stability_weights(wildcards):
    run_id = str(wildcards.stability_run)
    if run_id == "primary_10m":
        return rules.sift2_weights.output.weights
    return stability_path(wildcards.unit, run_id, "sift2_weights.txt")


def stability_assignments(wildcards):
    run_id = str(wildcards.stability_run)
    if run_id == "primary_10m":
        return rules.connectome_count.output.assignments
    return stability_path(wildcards.unit, run_id, "assignments.csv")


def stability_matrix_for(wildcards, matrix_name):
    run_id = str(wildcards.stability_run)
    if run_id == "primary_10m":
        return matrix_path(wildcards.unit, matrix_name)
    return stability_path(wildcards.unit, run_id, "matrices", f"{matrix_name}.csv")


def stability_seed_id(wildcards):
    token = _subject_rng_token(wildcards)
    if str(wildcards.stability_run) == "independent_10m":
        return f"{token}|independent-replicate-1"
    return token


def stability_seed_class(wildcards):
    return (
        "independent"
        if str(wildcards.stability_run) == "independent_10m"
        else "primary"
    )


def independent_rng_seed(wildcards):
    token = f"{_subject_rng_token(wildcards)}|independent-replicate-1".encode(
        "utf-8"
    )
    seed = int.from_bytes(hashlib.sha256(token).digest()[:4], "big") & 0x7FFFFFFF
    return seed or 1


rule stability_nested_primary_tracks:
    input:
        tracks=rules.tractography_10m.output.tracks,
    output:
        tracks=stability_path("{unit}", "{stability_run}", "tracks.tck"),
        metadata=stability_path(
            "{unit}", "{stability_run}", "tractography_parameters.json"
        ),
    params:
        requested=lambda wildcards: STABILITY_STREAMLINE_COUNTS[
            str(wildcards.stability_run)
        ],
        seed_id=stability_seed_id,
    log:
        subject_log("{unit}", "09_stability_{stability_run}_tckedit.log"),
    threads: 2
    wildcard_constraints:
        stability_run="primary_3m|primary_5m"
    shell:
        r"""
        set -euo pipefail
        mkdir -p "$(dirname {output.tracks:q})" "$(dirname {log:q})"
        export PATH={TOOL_PATH:q}
        partial={output.tracks:q}.partial
        trap 'rm -f "$partial" {output.metadata:q}.partial' EXIT
        tckedit {input.tracks:q} "$partial" -number {params.requested} -nthreads {threads} \
          > {log:q} 2>&1
        actual="$(tckinfo "$partial" -count 2>> {log:q} | awk '$1=="count:" {{print $2; exit}}')"
        test "$actual" = "{params.requested}"
        mv "$partial" {output.tracks:q}
        printf '{{"schema_version":"1.0.0","record_type":"phase_b_stability_tractography_parameters","diagnosis_labels_used":false,"unit":"%s","run_id":"%s","seed_class":"primary","seed_id":"%s","construction":"nested_prefix_of_primary_10m","requested_streamlines":%s,"actual_streamlines":%s}}\n' \
          {wildcards.unit:q} {wildcards.stability_run:q} {params.seed_id:q} \
          {params.requested:q} "$actual" > {output.metadata:q}.partial
        mv {output.metadata:q}.partial {output.metadata:q}
        """


rule stability_independent_10m_tracks:
    input:
        preflight=rules.tractography_preflight.output,
        fod=rules.mtnormalise.output.wm,
        five_tt=rules.five_tt_dwi.output.five_tt,
        dwi=rules.dwi_bias_correct.output,
    output:
        tracks=stability_path("{unit}", "independent_10m", "tracks.tck"),
        metadata=stability_path(
            "{unit}", "independent_10m", "tractography_parameters.json"
        ),
    params:
        rng_seed=independent_rng_seed,
        rng_token=lambda wildcards: (
            f"{_subject_rng_token(wildcards)}|independent-replicate-1"
        ),
        requested=10_000_000,
        max_seeds=int(config["tractography"]["maximum_seed_attempts"]),
        min_length=float(config["tractography"]["minimum_length_mm"]),
        max_length=float(config["tractography"]["maximum_length_mm"]),
        cutoff=float(config["tractography"]["cutoff"]),
        angle=float(config["tractography"]["maximum_angle_degrees"]),
    log:
        subject_log("{unit}", "09_stability_independent_10m_tckgen.log"),
    threads: int(config["tractography"]["nthreads_per_subject"])
    shell:
        r"""
        set -euo pipefail
        mkdir -p "$(dirname {output.tracks:q})" "$(dirname {log:q})"
        export PATH={TOOL_PATH:q}
        export MRTRIX_RNG_SEED={params.rng_seed}
        step="$(mrinfo {input.dwi:q} -spacing | awk '{{m=$1; if ($2<m) m=$2; if ($3<m) m=$3; printf "%.8f", m/2.0}}')"
        partial={output.tracks:q}.partial
        trap 'rm -f "$partial" {output.metadata:q}.partial' EXIT
        tckgen {input.fod:q} "$partial" -algorithm iFOD2 -act {input.five_tt:q} \
          -backtrack -crop_at_gmwmi -seed_dynamic {input.fod:q} \
          -select {params.requested} -seeds {params.max_seeds} \
          -minlength {params.min_length} -maxlength {params.max_length} \
          -cutoff {params.cutoff} -step "$step" -angle {params.angle} \
          -nthreads {threads} > {log:q} 2>&1
        actual="$(tckinfo "$partial" -count 2>> {log:q} | awk '$1=="count:" {{print $2; exit}}')"
        test "$actual" = "{params.requested}"
        mv "$partial" {output.tracks:q}
        printf '{{"schema_version":"1.0.0","record_type":"phase_b_stability_tractography_parameters","diagnosis_labels_used":false,"unit":"%s","run_id":"independent_10m","seed_class":"independent","seed_id":"%s","rng_seed":%s,"algorithm":"iFOD2","act":true,"backtrack":true,"crop_at_gmwmi":true,"seeding":"seed_dynamic","actual_step_size_mm":%s,"requested_streamlines":%s,"actual_streamlines":%s}}\n' \
          {wildcards.unit:q} {params.rng_token:q} {params.rng_seed:q} "$step" \
          {params.requested:q} "$actual" > {output.metadata:q}.partial
        mv {output.metadata:q}.partial {output.metadata:q}
        """


rule stability_sift2_weights:
    input:
        tracks=stability_tracks,
        fod=rules.mtnormalise.output.wm,
        five_tt=rules.five_tt_dwi.output.five_tt,
    output:
        weights=stability_path(
            "{unit}", "{stability_run}", "sift2_weights.txt"
        ),
        mu=stability_path("{unit}", "{stability_run}", "sift2_mu.txt"),
        iterations=stability_path(
            "{unit}", "{stability_run}", "sift2_iterations.csv"
        ),
    params:
        expected=lambda wildcards: STABILITY_STREAMLINE_COUNTS[
            str(wildcards.stability_run)
        ],
    log:
        subject_log("{unit}", "09_stability_{stability_run}_sift2.log"),
    threads: int(config["tractography"]["nthreads_per_subject"])
    wildcard_constraints:
        stability_run="primary_3m|primary_5m|independent_10m"
    shell:
        r"""
        set -euo pipefail
        mkdir -p "$(dirname {output.weights:q})" "$(dirname {log:q})"
        export PATH={TOOL_PATH:q}
        tcksift2 {input.tracks:q} {input.fod:q} {output.weights:q} \
          -act {input.five_tt:q} -out_mu {output.mu:q} -csv {output.iterations:q} \
          -reg_tikhonov 0.0 -reg_tv 0.1 -min_td_frac 0.1 \
          -min_iters 10 -max_iters 1000 -nthreads {threads} > {log:q} 2>&1
        test "$(wc -l < {output.weights:q})" -eq {params.expected}
        """


rule stability_connectome_count:
    input:
        tracks=stability_tracks,
        atlas=rules.aal3_to_dwi_single_resample.output,
    output:
        matrix=stability_path(
            "{unit}", "{stability_run}", "matrices", "count.csv"
        ),
        assignments=stability_path(
            "{unit}", "{stability_run}", "assignments.csv"
        ),
    log:
        subject_log("{unit}", "09_stability_{stability_run}_count.log"),
    threads: 8
    wildcard_constraints:
        stability_run="primary_3m|primary_5m|independent_10m"
    shell:
        r"""
        set -euo pipefail
        mkdir -p "$(dirname {output.matrix:q})" "$(dirname {log:q})"
        export PATH={TOOL_PATH:q}
        tck2connectome {input.tracks:q} {input.atlas:q} {output.matrix:q} \
          -assignment_radial_search 4 -symmetric -zero_diagonal -stat_edge sum \
          -out_assignments {output.assignments:q} -nthreads {threads} > {log:q} 2>&1
        """


rule stability_connectome_fd_sum:
    input:
        tracks=stability_tracks,
        atlas=rules.aal3_to_dwi_single_resample.output,
        weights=stability_weights,
    output:
        stability_path("{unit}", "{stability_run}", "matrices", "fd_sum.csv"),
    log:
        subject_log("{unit}", "09_stability_{stability_run}_fd_sum.log"),
    threads: 8
    wildcard_constraints:
        stability_run="primary_3m|primary_5m|independent_10m"
    shell:
        r"""
        set -euo pipefail
        mkdir -p "$(dirname {output:q})" "$(dirname {log:q})"
        export PATH={TOOL_PATH:q}
        tck2connectome {input.tracks:q} {input.atlas:q} {output:q} \
          -assignment_radial_search 4 -symmetric -zero_diagonal -stat_edge sum \
          -tck_weights_in {input.weights:q} -nthreads {threads} > {log:q} 2>&1
        """


rule stability_tensor_samples:
    input:
        tracks=stability_tracks,
        image=_tensor_image,
    output:
        stability_path(
            "{unit}",
            "{stability_run}",
            "samples",
            "{metric}_per_streamline.csv",
        ),
    params:
        expected=lambda wildcards: STABILITY_STREAMLINE_COUNTS[
            str(wildcards.stability_run)
        ],
    log:
        subject_log(
            "{unit}", "09_stability_{stability_run}_{metric}_tcksample.log"
        ),
    threads: 8
    wildcard_constraints:
        stability_run="primary_3m|primary_5m|independent_10m",
        metric="fa|md|rd|ad"
    shell:
        r"""
        set -euo pipefail
        mkdir -p "$(dirname {output:q})" "$(dirname {log:q})"
        export PATH={TOOL_PATH:q}
        tcksample {input.tracks:q} {input.image:q} {output:q} \
          -stat_tck mean -nthreads {threads} > {log:q} 2>&1
        test "$(wc -l < {output:q})" -eq {params.expected}
        """


rule stability_tensor_connectome:
    input:
        tracks=stability_tracks,
        atlas=rules.aal3_to_dwi_single_resample.output,
        samples=rules.stability_tensor_samples.output,
    output:
        stability_path(
            "{unit}", "{stability_run}", "matrices", "{metric}_mean.csv"
        ),
    log:
        subject_log(
            "{unit}", "09_stability_{stability_run}_{metric}_connectome.log"
        ),
    threads: 8
    wildcard_constraints:
        stability_run="primary_3m|primary_5m|independent_10m",
        metric="fa|md|rd|ad"
    shell:
        r"""
        set -euo pipefail
        mkdir -p "$(dirname {output:q})" "$(dirname {log:q})"
        export PATH={TOOL_PATH:q}
        tck2connectome {input.tracks:q} {input.atlas:q} {output:q} \
          -assignment_radial_search 4 -symmetric -zero_diagonal \
          -scale_file {input.samples:q} -stat_edge mean -nthreads {threads} \
          > {log:q} 2>&1
        """


rule stability_run_record:
    input:
        tracks=stability_tracks,
        metadata=stability_metadata,
        weights=stability_weights,
        assignments=stability_assignments,
        count_matrix=lambda wildcards: stability_matrix_for(wildcards, "count"),
        fd_sum_matrix=lambda wildcards: stability_matrix_for(wildcards, "fd_sum"),
        fa_mean_matrix=lambda wildcards: stability_matrix_for(wildcards, "fa_mean"),
        md_mean_matrix=lambda wildcards: stability_matrix_for(wildcards, "md_mean"),
        rd_mean_matrix=lambda wildcards: stability_matrix_for(wildcards, "rd_mean"),
        ad_mean_matrix=lambda wildcards: stability_matrix_for(wildcards, "ad_mean"),
    output:
        stability_path("{unit}", "{stability_run}", "run_record.json"),
    params:
        expected=lambda wildcards: STABILITY_STREAMLINE_COUNTS[
            str(wildcards.stability_run)
        ],
        seed_id=stability_seed_id,
        seed_class=stability_seed_class,
    log:
        subject_log("{unit}", "09_stability_{stability_run}_record.log"),
    wildcard_constraints:
        stability_run="primary_3m|primary_5m|primary_10m|independent_10m"
    run:
        Path(output[0]).parent.mkdir(parents=True, exist_ok=True)
        Path(log[0]).parent.mkdir(parents=True, exist_ok=True)
        count_output = subprocess.run(
            [str(MRTRIX_BIN / "tckinfo"), str(input.tracks), "-count"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        actual_tracks = None
        for line in count_output.splitlines():
            fields = line.split()
            if len(fields) >= 2 and fields[0] == "count:":
                actual_tracks = int(fields[1])
                break
        if actual_tracks is None:
            raise ValueError("tckinfo did not report a streamline count")

        def nonempty_lines(path):
            with Path(path).open(encoding="utf-8") as handle:
                return sum(bool(line.strip()) for line in handle)

        weight_count = nonempty_lines(input.weights)
        assignment_count = 0
        assigned_count = 0
        with Path(input.assignments).open(newline="", encoding="utf-8") as handle:
            for row in csv.reader(handle):
                if not row or not any(str(value).strip() for value in row):
                    continue
                if len(row) < 2:
                    raise ValueError("assignment row has fewer than two endpoints")
                left = int(str(row[0]).strip())
                right = int(str(row[1]).strip())
                assignment_count += 1
                assigned_count += int(left > 0 and right > 0)
        expected = int(params.expected)
        if not actual_tracks == weight_count == assignment_count == expected:
            raise ValueError(
                "tractogram/weight/assignment counts differ: "
                f"{actual_tracks}/{weight_count}/{assignment_count}/{expected}"
            )
        record = {
            "schema_version": "1.0.0",
            "record_type": "diagnosis_blind_phase_b_stability_run",
            "diagnosis_labels_used": False,
            "unit": str(wildcards.unit),
            "run_id": str(wildcards.stability_run),
            "seed_id": str(params.seed_id),
            "seed_class": str(params.seed_class),
            "streamline_count": expected,
            "tractogram_streamline_count": actual_tracks,
            "sift2_weight_count": weight_count,
            "assignment_row_count": assignment_count,
            "assigned_streamline_count": assigned_count,
            "endpoint_assignment_fraction": assigned_count / assignment_count,
            "matrices": {
                "count": str(Path(input.count_matrix).resolve()),
                "fd_sum": str(Path(input.fd_sum_matrix).resolve()),
                "fa_mean": str(Path(input.fa_mean_matrix).resolve()),
                "md_mean": str(Path(input.md_mean_matrix).resolve()),
                "rd_mean": str(Path(input.rd_mean_matrix).resolve()),
                "ad_mean": str(Path(input.ad_mean_matrix).resolve()),
            },
            "artifacts": {
                "tractogram": str(Path(input.tracks).resolve()),
                "sift2_weights": str(Path(input.weights).resolve()),
                "assignments": str(Path(input.assignments).resolve()),
                "tractography_metadata": str(Path(input.metadata).resolve()),
            },
        }
        temporary = Path(str(output[0]) + ".partial")
        temporary.write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output[0])
        Path(log[0]).write_text(
            "internal: diagnosis-blind phase-B stability run record\n"
            + json.dumps(record, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )


STABILITY_RUN_RECORDS = [
    stability_path(unit, run_id, "run_record.json")
    for unit in PHASE_B_UNITS
    for run_id in STABILITY_RUNS
]
STABILITY_RUN_RECORD_ARGS = " ".join(
    f"--run-record {shlex.quote(path)}" for path in STABILITY_RUN_RECORDS
)


rule phase_b_stability_manifest:
    input:
        records=STABILITY_RUN_RECORDS,
        builder=str(STABILITY_BUILDER),
    output:
        run_path("publication", "recipe_stability_manifest.json"),
    params:
        record_args=STABILITY_RUN_RECORD_ARGS,
    log:
        run_path("logs", "09_phase_b_stability_manifest.log"),
    shell:
        r"""
        set -euo pipefail
        mkdir -p "$(dirname {output:q})" "$(dirname {log:q})"
        {sys.executable:q} {input.builder:q} {params.record_args} --output {output:q} \
          > {log:q} 2>&1
        """


rule phase_b_stability_validation:
    input:
        manifest=rules.phase_b_stability_manifest.output,
        validator=str(STABILITY_VALIDATOR),
        contract=str(THESIS_RELEASE_CONTRACT),
    output:
        run_path("publication", "recipe_stability_validation.json"),
    log:
        run_path("logs", "09_phase_b_stability_validation.log"),
    shell:
        r"""
        set -euo pipefail
        mkdir -p "$(dirname {output:q})" "$(dirname {log:q})"
        {sys.executable:q} {input.validator:q} --manifest {input.manifest:q} \
          --contract {input.contract:q} --output {output:q} --require-pass \
          > {log:q} 2>&1
        """


rule phase_b_stability_evidence:
    input:
        rules.phase_b_stability_validation.output,
