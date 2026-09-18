"""Diagnosis-blind HCP379 3M/5M/10M and independent-seed Phase-B DAG."""

import shlex
import subprocess


HCP379_RUNS = (
    "primary_3m",
    "primary_5m",
    "primary_10m",
    "independent_3m",
    "independent_5m",
    "independent_10m",
)
HCP379_STREAMLINE_COUNTS = {
    "primary_3m": 3_000_000,
    "primary_5m": 5_000_000,
    "primary_10m": 10_000_000,
    "independent_3m": 3_000_000,
    "independent_5m": 5_000_000,
    "independent_10m": 10_000_000,
}
HCP379_MATRIX_NAMES = (
    "count",
    "fd_sum",
    "count_invnodevol",
    "len_mean",
    "invlen_mean",
    "fa_mean",
    "md_mean",
    "rd_mean",
    "ad_mean",
)
HCP379_CANARY_SUMMARY = Path(
    config.get(
        "hcp379_pretract_manifest",
        (
            "/data/derivatives/hcp379_v2/pretract_recovery4/"
            "pretract_recovery4_manifest.json"
        ),
    )
).resolve()
_hcp379_promotion_receipt_config = config.get(
    "hcp379_promotion_receipt"
)
HCP379_PROMOTION_RECEIPT = (
    Path(str(_hcp379_promotion_receipt_config)).resolve()
    if _hcp379_promotion_receipt_config
    else None
)
HCP379_BUILDER = (
    WORKFLOW_DIR / "build_hcp379_phase_b_stability_manifest.py"
).resolve()
HCP379_VALIDATOR = (
    PROJECT_ROOT / "research_audit/validate_hcp379_phase_b_stability.py"
).resolve()
HCP379_TCK_HEADER_COUNT = (
    PROJECT_ROOT / "scripts/hcp/tck_header_count_v1.py"
).resolve()
HCP379_NUMERIC_VALUE_COUNT = (
    PROJECT_ROOT / "scripts/hcp/count_mrtrix_numeric_values_v1.py"
).resolve()
HCP379_CONTRACT = (
    PROJECT_ROOT
    / "research_audit/outputs/thesis_grade_530_release_contract_v1/contract.json"
).resolve()
HCP379_VALIDATOR_PYTHON = Path(
    "/home/ec2-user/exp/.venv_connectome_app/bin/python"
).resolve()
for _hcp379_source in (
    HCP379_CANARY_SUMMARY,
    HCP379_BUILDER,
    HCP379_VALIDATOR,
    HCP379_TCK_HEADER_COUNT,
    HCP379_NUMERIC_VALUE_COUNT,
    HCP379_CONTRACT,
    HCP379_VALIDATOR_PYTHON,
):
    if not _hcp379_source.is_file():
        raise FileNotFoundError(_hcp379_source)
if (
    HCP379_PROMOTION_RECEIPT is not None
    and not HCP379_PROMOTION_RECEIPT.is_file()
):
    raise FileNotFoundError(HCP379_PROMOTION_RECEIPT)


def _hcp379_verified_artifact(record, label):
    path = Path(str(record.get("path", ""))).resolve()
    if (
        not path.is_file()
        or path.is_symlink()
        or record.get("size_bytes") != path.stat().st_size
        or record.get("sha256") != _sha256(path)
    ):
        raise ValueError(f"{label}: artifact binding differs")
    return path


with HCP379_CANARY_SUMMARY.open(encoding="utf-8") as _hcp379_handle:
    HCP379_PRETRACT_MASTER = json.load(_hcp379_handle)
if (
    HCP379_PRETRACT_MASTER.get("record_type")
    != "diagnosis_blind_hcp379_pretract_recovery4_manifest"
    or HCP379_PRETRACT_MASTER.get("status")
    != "AUTOMATED_PASS_AWAITING_HUMAN_VISUAL_QC"
    or HCP379_PRETRACT_MASTER.get("diagnosis_labels_used") is not False
    or HCP379_PRETRACT_MASTER.get("human_visual_qc_inferred") is not False
    or HCP379_PRETRACT_MASTER.get("unit_count") != 15
):
    raise ValueError("Recovery4 pretract master is not an exact automated PASS")
if HCP379_PROMOTION_RECEIPT is not None:
    with HCP379_PROMOTION_RECEIPT.open(
        encoding="utf-8"
    ) as _hcp379_handle:
        _hcp379_promotion = json.load(_hcp379_handle)
    _hcp379_summary_record = {
        "path": str(HCP379_CANARY_SUMMARY),
        "sha256": _sha256(HCP379_CANARY_SUMMARY),
        "size_bytes": HCP379_CANARY_SUMMARY.stat().st_size,
    }
    if (
        _hcp379_promotion.get("record_type")
        != (
            "diagnosis_blind_hcp379_recovery4_v6_"
            "promotion_receipt"
        )
        or _hcp379_promotion.get("status") != "PASS"
        or _hcp379_promotion.get("diagnosis_labels_used") is not False
        or _hcp379_promotion.get("tractography_started") is not False
        or _hcp379_promotion.get("matrix_generation_started") is not False
        or _hcp379_promotion.get("records", {}).get(
            "promoted_pretract_master"
        )
        != _hcp379_summary_record
        or _hcp379_promotion.get("hard_hold_resolution", {}).get(
            "remaining_hard_hold_units"
        )
        != []
    ):
        raise ValueError("Recovery4 V6 promotion receipt differs")
HCP379_PRETRACT_BY_UNIT = {}
HCP379_PRETRACT_MANIFEST_PATH_BY_UNIT = {}
for _hcp379_row in HCP379_PRETRACT_MASTER.get("units", []):
    _hcp379_unit = str(_hcp379_row.get("unit", ""))
    _hcp379_manifest_path = _hcp379_verified_artifact(
        _hcp379_row.get("manifest", {}),
        f"{_hcp379_unit}:unit_manifest",
    )
    with _hcp379_manifest_path.open(encoding="utf-8") as _hcp379_handle:
        _hcp379_manifest = json.load(_hcp379_handle)
    if (
        _hcp379_manifest.get("record_type")
        != "diagnosis_blind_hcp379_pretract_recovery4_unit_manifest"
        or _hcp379_manifest.get("status")
        != "AUTOMATED_PASS_AWAITING_HUMAN_VISUAL_QC"
        or _hcp379_manifest.get("diagnosis_labels_used") is not False
        or _hcp379_manifest.get("human_visual_qc_inferred") is not False
        or _hcp379_manifest.get("unit") != _hcp379_unit
    ):
        raise ValueError(f"{_hcp379_unit}: Recovery4 unit manifest differs")
    HCP379_PRETRACT_BY_UNIT[_hcp379_unit] = _hcp379_manifest
    HCP379_PRETRACT_MANIFEST_PATH_BY_UNIT[_hcp379_unit] = (
        _hcp379_manifest_path
    )
if set(HCP379_PRETRACT_BY_UNIT) != set(PHASE_B_UNITS):
    raise ValueError("Recovery4 pretract and Phase-B unit sets differ")


def hcp379_recorded_path(unit, *keys):
    value = HCP379_PRETRACT_BY_UNIT[str(unit)]
    for key in keys:
        value = value[key]
    return str(
        _hcp379_verified_artifact(
            value,
            f"{unit}:{'/'.join(str(key) for key in keys)}",
        )
    )


def hcp379_path(unit, hcp_run, *parts):
    return subject_path(unit, "09_hcp379_stability", hcp_run, *parts)


def hcp379_atlas_path(unit, name):
    if name == "selected_transform":
        registration = HCP379_PRETRACT_BY_UNIT[str(unit)][
            "registration"
        ]
        if "transform" in registration:
            record = registration["transform"]
        else:
            chain = registration.get("transform_chain", {})
            record = chain.get("warp") or chain.get("affine")
            if record is None:
                raise ValueError(
                    f"{unit}: no selected registration transform artifact"
                )
        return str(
            _hcp379_verified_artifact(
                record, f"{unit}:registration/selected_transform"
            )
        )
    keys = {
        "source_parcellation": ("hcp379", "source_parcellation"),
        "corrected_nodes": ("hcp379", "nodes_b0_1mm"),
        "node_volumes": ("hcp379", "node_volumes"),
        "source_node_volumes": ("hcp379", "source_node_volumes"),
        "atlas_qc": ("hcp379", "atlas_qc"),
        "visual_overlay": ("hcp379", "visual_overlay"),
        "atlas_result": ("automated_qc", "atlas"),
        "b0_reference": ("tractography_inputs", "mean_b0"),
    }[name]
    return hcp379_recorded_path(unit, *keys)


def hcp379_source_parcellation(wildcards):
    return hcp379_atlas_path(
        wildcards.unit, "source_parcellation"
    )


def hcp379_corrected_nodes(wildcards):
    return hcp379_atlas_path(wildcards.unit, "corrected_nodes")


def hcp379_node_volumes(wildcards):
    return hcp379_atlas_path(wildcards.unit, "node_volumes")


def hcp379_source_node_volumes(wildcards):
    return hcp379_atlas_path(wildcards.unit, "source_node_volumes")


def hcp379_atlas_qc(wildcards):
    return hcp379_atlas_path(wildcards.unit, "atlas_qc")


def hcp379_visual_overlay(wildcards):
    return hcp379_atlas_path(wildcards.unit, "visual_overlay")


def hcp379_wmfod(wildcards):
    return hcp379_recorded_path(
        wildcards.unit, "tractography_inputs", "wmfod_normalised"
    )


def hcp379_five_tt(wildcards):
    return hcp379_recorded_path(
        wildcards.unit, "tractography_inputs", "five_tt"
    )


def hcp379_dwi(wildcards):
    return hcp379_recorded_path(
        wildcards.unit, "tractography_inputs", "dwi_bias_corrected"
    )


def hcp379_tensor_image(wildcards):
    return hcp379_recorded_path(
        wildcards.unit, "tensor_maps_bounded", str(wildcards.metric)
    )


def hcp379_seed_id(wildcards):
    base_token = _subject_rng_token(wildcards)
    if str(wildcards.hcp_run).startswith("independent_"):
        return f"{base_token}|independent-replicate-1"
    return base_token


def hcp379_seed_class(wildcards):
    return (
        "independent"
        if str(wildcards.hcp_run).startswith("independent_")
        else "primary"
    )


def hcp379_independent_rng_seed(wildcards):
    token = (
        f"{_subject_rng_token(wildcards)}|independent-replicate-1"
    ).encode("utf-8")
    seed = int.from_bytes(hashlib.sha256(token).digest()[:4], "big")
    return (seed & 0x7FFFFFFF) or 1


def hcp379_rng_seed(wildcards):
    if str(wildcards.hcp_run).startswith("independent_"):
        return hcp379_independent_rng_seed(wildcards)
    return _subject_rng_seed(wildcards)


def hcp379_tracks(wildcards):
    return hcp379_path(
        wildcards.unit, str(wildcards.hcp_run), "tracks.tck"
    )


def hcp379_metadata(wildcards):
    return hcp379_path(
        wildcards.unit,
        str(wildcards.hcp_run),
        "tractography_parameters.json",
    )


def hcp379_weights(wildcards):
    return hcp379_path(
        wildcards.unit,
        str(wildcards.hcp_run),
        "sift2_weights.txt",
    )


def hcp379_matrix_for(wildcards, matrix_name):
    return hcp379_path(
        wildcards.unit,
        str(wildcards.hcp_run),
        "matrices",
        f"{matrix_name}.csv",
    )


rule hcp379_corrected_atlas_gate:
    input:
        summary=ancient(str(HCP379_CANARY_SUMMARY)),
        results=[
            ancient(str(HCP379_PRETRACT_MANIFEST_PATH_BY_UNIT[unit]))
            for unit in PHASE_B_UNITS
        ],
    output:
        run_path("contract", "hcp379_corrected_atlas_gate.json"),
    log:
        run_path("logs", "09_hcp379_corrected_atlas_gate.log"),
    run:
        Path(output[0]).parent.mkdir(parents=True, exist_ok=True)
        Path(log[0]).parent.mkdir(parents=True, exist_ok=True)
        summary = json.loads(Path(input.summary).read_text(encoding="utf-8"))
        failures = []
        if (
            summary.get("record_type")
            != "diagnosis_blind_hcp379_pretract_recovery4_manifest"
            or summary.get("status")
            != "AUTOMATED_PASS_AWAITING_HUMAN_VISUAL_QC"
            or summary.get("diagnosis_labels_used") is not False
            or summary.get("human_visual_qc_inferred") is not False
            or summary.get("unit_count") != 15
        ):
            failures.append(
                "Recovery4 pretract master is not an exact automated PASS"
            )
        result_evidence = {}
        for result_path in input.results:
            result = json.loads(Path(result_path).read_text(encoding="utf-8"))
            unit = str(result.get("unit", ""))
            if (
                unit not in PHASE_B_UNITS
                or result.get("record_type")
                != "diagnosis_blind_hcp379_pretract_recovery4_unit_manifest"
                or result.get("status")
                != "AUTOMATED_PASS_AWAITING_HUMAN_VISUAL_QC"
                or result.get("diagnosis_labels_used") is not False
                or result.get("human_visual_qc_inferred") is not False
                or result.get("registration", {}).get(
                    "selected_5tt_dwi_mask_dice", 0.0
                )
                < 0.70
            ):
                failures.append(
                    f"Recovery4 pretract unit manifest differs for {unit}"
                )
                continue
            for category in (
                "tractography_inputs",
                "tensor_maps_bounded",
                "hcp379",
                "automated_qc",
            ):
                for name, artifact in result.get(category, {}).items():
                    try:
                        _hcp379_verified_artifact(
                            artifact, f"{unit}:{category}/{name}"
                        )
                    except Exception as exc:
                        failures.append(str(exc))
            registration_artifacts = [
                (
                    "selected_t1_in_b0",
                    result["registration"]["selected_t1_in_b0"],
                )
            ]
            if "transform" in result["registration"]:
                registration_artifacts.append(
                    (
                        "selected_transform",
                        result["registration"]["transform"],
                    )
                )
            else:
                transform_chain = result["registration"].get(
                    "transform_chain", {}
                )
                if not transform_chain:
                    failures.append(
                        f"{unit}: registration transform is absent"
                    )
                registration_artifacts.extend(
                    (
                        f"selected_transform_chain/{name}",
                        artifact,
                    )
                    for name, artifact in transform_chain.items()
                )
            for name, artifact in registration_artifacts:
                try:
                    _hcp379_verified_artifact(artifact, f"{unit}:{name}")
                except Exception as exc:
                    failures.append(str(exc))
            result_evidence[unit] = {
                "path": str(Path(result_path).resolve()),
                "sha256": _sha256(result_path),
                "size_bytes": Path(result_path).stat().st_size,
            }
        if set(result_evidence) != set(PHASE_B_UNITS):
            failures.append("Recovery4 pretract unit-manifest set differs")
        record = {
            "schema_version": "1.0.0",
            "record_type": "diagnosis_blind_hcp379_corrected_atlas_gate",
            "status": "PASS" if not failures else "FAIL",
            "diagnosis_labels_used": False,
            "approved_units": list(PHASE_B_UNITS),
            "summary": {
                "path": str(Path(input.summary).resolve()),
                "sha256": _sha256(input.summary),
                "size_bytes": Path(input.summary).stat().st_size,
            },
            "result_evidence": dict(sorted(result_evidence.items())),
            "failures": failures,
        }
        Path(log[0]).write_text(
            "internal: Recovery4 HCP379 pre-tractography gate\n"
            + json.dumps(record, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        if failures:
            raise ValueError(
                "corrected HCP379 atlas gate failed: " + "; ".join(failures)
            )
        temporary = Path(str(output[0]) + ".partial")
        temporary.write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output[0])


rule hcp379_primary_nested_tracks:
    input:
        tracks=lambda wildcards: hcp379_path(
            wildcards.unit, "primary_10m", "tracks.tck"
        ),
    output:
        tracks=hcp379_path("{unit}", "{hcp_run}", "tracks.tck"),
        metadata=hcp379_path(
            "{unit}", "{hcp_run}", "tractography_parameters.json"
        ),
    params:
        requested=lambda wildcards: HCP379_STREAMLINE_COUNTS[
            str(wildcards.hcp_run)
        ],
        seed_id=hcp379_seed_id,
    log:
        subject_log("{unit}", "09_hcp379_{hcp_run}_tckedit.log"),
    threads: 2
    wildcard_constraints:
        hcp_run="primary_3m|primary_5m"
    shell:
        r"""
        set -euo pipefail
        mkdir -p "$(dirname {output.tracks:q})" "$(dirname {log:q})"
        export PATH={TOOL_PATH:q}
        partial={output.tracks:q}.partial.tck
        trap 'rm -f "$partial" {output.metadata:q}.partial' EXIT
        tckedit {input.tracks:q} "$partial" -number {params.requested} \
          -nthreads {threads} > {log:q} 2>&1
        actual="$(python /home/ec2-user/exp/scripts/hcp/tck_header_count_v1.py "$partial" --expect {params.requested} 2>> {log:q})"
        test "$actual" = "{params.requested}"
        mv "$partial" {output.tracks:q}
        printf '{{"schema_version":"1.0.0","record_type":"hcp379_stability_tractography_parameters","diagnosis_labels_used":false,"unit":"%s","run_id":"%s","seed_class":"primary","seed_id":"%s","construction":"nested_prefix_of_primary_10m","requested_streamlines":%s,"actual_streamlines":%s}}\n' \
          {wildcards.unit:q} {wildcards.hcp_run:q} {params.seed_id:q} \
          {params.requested:q} "$actual" > {output.metadata:q}.partial
        mv {output.metadata:q}.partial {output.metadata:q}
        """


rule hcp379_seeded_10m_tracks:
    input:
        atlas_gate=rules.hcp379_corrected_atlas_gate.output,
        fod=hcp379_wmfod,
        five_tt=hcp379_five_tt,
        dwi=hcp379_dwi,
    output:
        tracks=hcp379_path("{unit}", "{hcp_run}", "tracks.tck"),
        metadata=hcp379_path(
            "{unit}", "{hcp_run}", "tractography_parameters.json"
        ),
    params:
        rng_seed=hcp379_rng_seed,
        rng_token=hcp379_seed_id,
        seed_class=hcp379_seed_class,
        requested=10_000_000,
        max_seeds=int(config["tractography"]["maximum_seed_attempts"]),
        min_length=float(config["tractography"]["minimum_length_mm"]),
        max_length=float(config["tractography"]["maximum_length_mm"]),
        cutoff=float(config["tractography"]["cutoff"]),
        angle=float(config["tractography"]["maximum_angle_degrees"]),
    log:
        subject_log("{unit}", "09_hcp379_{hcp_run}_tckgen.log"),
    threads: int(config["tractography"]["nthreads_per_subject"])
    wildcard_constraints:
        hcp_run="primary_10m|independent_10m"
    shell:
        r"""
        set -euo pipefail
        mkdir -p "$(dirname {output.tracks:q})" "$(dirname {log:q})"
        export PATH={TOOL_PATH:q}
        export MRTRIX_RNG_SEED={params.rng_seed}
        step="$(mrinfo {input.dwi:q} -spacing | awk '{{m=$1; if ($2<m) m=$2; if ($3<m) m=$3; printf "%.8f", m/2.0}}')"
        partial={output.tracks:q}.partial.tck
        trap 'rm -f "$partial" {output.metadata:q}.partial' EXIT
        tckgen {input.fod:q} "$partial" -algorithm iFOD2 -act {input.five_tt:q} \
          -backtrack -crop_at_gmwmi -seed_dynamic {input.fod:q} \
          -select {params.requested} -seeds {params.max_seeds} \
          -minlength {params.min_length} -maxlength {params.max_length} \
          -cutoff {params.cutoff} -step "$step" -angle {params.angle} \
          -nthreads {threads} > {log:q} 2>&1
        actual="$(python /home/ec2-user/exp/scripts/hcp/tck_header_count_v1.py "$partial" --expect {params.requested} 2>> {log:q})"
        test "$actual" = "{params.requested}"
        mv "$partial" {output.tracks:q}
        printf '{{"schema_version":"1.0.0","record_type":"hcp379_stability_tractography_parameters","diagnosis_labels_used":false,"unit":"%s","run_id":"%s","seed_class":"%s","seed_id":"%s","rng_seed":%s,"algorithm":"iFOD2","act":true,"backtrack":true,"crop_at_gmwmi":true,"seeding":"seed_dynamic","actual_step_size_mm":%s,"requested_streamlines":%s,"actual_streamlines":%s}}\n' \
          {wildcards.unit:q} {wildcards.hcp_run:q} {params.seed_class:q} \
          {params.rng_token:q} {params.rng_seed:q} "$step" \
          {params.requested:q} "$actual" > {output.metadata:q}.partial
        mv {output.metadata:q}.partial {output.metadata:q}
        """


rule hcp379_independent_nested_tracks:
    input:
        tracks=lambda wildcards: hcp379_path(
            wildcards.unit, "independent_10m", "tracks.tck"
        ),
    output:
        tracks=hcp379_path("{unit}", "{hcp_run}", "tracks.tck"),
        metadata=hcp379_path(
            "{unit}", "{hcp_run}", "tractography_parameters.json"
        ),
    params:
        requested=lambda wildcards: HCP379_STREAMLINE_COUNTS[
            str(wildcards.hcp_run)
        ],
        seed_id=hcp379_seed_id,
    log:
        subject_log("{unit}", "09_hcp379_{hcp_run}_tckedit.log"),
    threads: 2
    wildcard_constraints:
        hcp_run="independent_3m|independent_5m"
    shell:
        r"""
        set -euo pipefail
        mkdir -p "$(dirname {output.tracks:q})" "$(dirname {log:q})"
        export PATH={TOOL_PATH:q}
        partial={output.tracks:q}.partial.tck
        trap 'rm -f "$partial" {output.metadata:q}.partial' EXIT
        tckedit {input.tracks:q} "$partial" -number {params.requested} \
          -nthreads {threads} > {log:q} 2>&1
        actual="$(python /home/ec2-user/exp/scripts/hcp/tck_header_count_v1.py "$partial" --expect {params.requested} 2>> {log:q})"
        test "$actual" = "{params.requested}"
        mv "$partial" {output.tracks:q}
        printf '{{"schema_version":"1.0.0","record_type":"hcp379_stability_tractography_parameters","diagnosis_labels_used":false,"unit":"%s","run_id":"%s","seed_class":"independent","seed_id":"%s","construction":"nested_prefix_of_independent_10m","requested_streamlines":%s,"actual_streamlines":%s}}\n' \
          {wildcards.unit:q} {wildcards.hcp_run:q} {params.seed_id:q} \
          {params.requested:q} "$actual" > {output.metadata:q}.partial
        mv {output.metadata:q}.partial {output.metadata:q}
        """


rule hcp379_stability_sift2:
    input:
        tracks=hcp379_tracks,
        fod=hcp379_wmfod,
        five_tt=hcp379_five_tt,
    output:
        weights=hcp379_path(
            "{unit}", "{hcp_run}", "sift2_weights.txt"
        ),
        mu=hcp379_path("{unit}", "{hcp_run}", "sift2_mu.txt"),
        iterations=hcp379_path(
            "{unit}", "{hcp_run}", "sift2_iterations.csv"
        ),
    params:
        expected=lambda wildcards: HCP379_STREAMLINE_COUNTS[
            str(wildcards.hcp_run)
        ],
    log:
        subject_log("{unit}", "09_hcp379_{hcp_run}_sift2.log"),
    threads: int(config["tractography"]["nthreads_per_subject"])
    wildcard_constraints:
        hcp_run="primary_3m|primary_5m|primary_10m|independent_3m|independent_5m|independent_10m"
    shell:
        r"""
        set -euo pipefail
        mkdir -p "$(dirname {output.weights:q})" "$(dirname {log:q})"
        export PATH={TOOL_PATH:q}
        tcksift2 {input.tracks:q} {input.fod:q} {output.weights:q} \
          -act {input.five_tt:q} -out_mu {output.mu:q} -csv {output.iterations:q} \
          -reg_tikhonov 0.0 -reg_tv 0.1 -min_td_frac 0.1 \
          -min_iters 10 -max_iters 1000 -nthreads {threads} > {log:q} 2>&1
        test "$(python {HCP379_NUMERIC_VALUE_COUNT:q} {output.weights:q} --expect {params.expected})" -eq {params.expected}
        """


rule hcp379_stability_count:
    input:
        gate=rules.hcp379_corrected_atlas_gate.output,
        tracks=hcp379_tracks,
        atlas=hcp379_corrected_nodes,
    output:
        matrix=hcp379_path(
            "{unit}", "{hcp_run}", "matrices", "count.csv"
        ),
        assignments=hcp379_path(
            "{unit}", "{hcp_run}", "assignments.csv"
        ),
    log:
        subject_log("{unit}", "09_hcp379_{hcp_run}_count.log"),
    threads: 8
    wildcard_constraints:
        hcp_run="primary_3m|primary_5m|primary_10m|independent_3m|independent_5m|independent_10m"
    shell:
        r"""
        set -euo pipefail
        mkdir -p "$(dirname {output.matrix:q})" "$(dirname {log:q})"
        export PATH={TOOL_PATH:q}
        tck2connectome {input.tracks:q} {input.atlas:q} {output.matrix:q} \
          -assignment_radial_search 4 -symmetric -zero_diagonal -stat_edge sum \
          -out_assignments {output.assignments:q} -nthreads {threads} \
          > {log:q} 2>&1
        """


rule hcp379_stability_fd_sum:
    input:
        tracks=hcp379_tracks,
        atlas=hcp379_corrected_nodes,
        weights=hcp379_weights,
    output:
        hcp379_path(
            "{unit}", "{hcp_run}", "matrices", "fd_sum.csv"
        ),
    log:
        subject_log("{unit}", "09_hcp379_{hcp_run}_fd_sum.log"),
    threads: 8
    wildcard_constraints:
        hcp_run="primary_3m|primary_5m|primary_10m|independent_3m|independent_5m|independent_10m"
    shell:
        r"""
        set -euo pipefail
        mkdir -p "$(dirname {output:q})" "$(dirname {log:q})"
        export PATH={TOOL_PATH:q}
        tck2connectome {input.tracks:q} {input.atlas:q} {output:q} \
          -assignment_radial_search 4 -symmetric -zero_diagonal -stat_edge sum \
          -tck_weights_in {input.weights:q} -nthreads {threads} > {log:q} 2>&1
        """


rule hcp379_stability_length:
    input:
        tracks=hcp379_tracks,
        atlas=hcp379_corrected_nodes,
    output:
        hcp379_path(
            "{unit}", "{hcp_run}", "matrices", "{length_metric}_mean.csv"
        ),
    params:
        scale=lambda wildcards: (
            "-scale_length"
            if str(wildcards.length_metric) == "len"
            else "-scale_invlength"
        ),
    log:
        subject_log(
            "{unit}",
            "09_hcp379_{hcp_run}_{length_metric}_mean.log",
        ),
    threads: 8
    wildcard_constraints:
        hcp_run="primary_3m|primary_5m|primary_10m|independent_3m|independent_5m|independent_10m",
        length_metric="len|invlen"
    shell:
        r"""
        set -euo pipefail
        mkdir -p "$(dirname {output:q})" "$(dirname {log:q})"
        export PATH={TOOL_PATH:q}
        tck2connectome {input.tracks:q} {input.atlas:q} {output:q} \
          -assignment_radial_search 4 -symmetric -zero_diagonal \
          {params.scale} -stat_edge mean -nthreads {threads} > {log:q} 2>&1
        """


rule hcp379_stability_tensor_samples:
    input:
        tracks=hcp379_tracks,
        image=hcp379_tensor_image,
    output:
        temp(
            hcp379_path(
                "{unit}",
                "{hcp_run}",
                "samples",
                "{metric}_per_streamline.csv",
            )
        ),
    params:
        expected=lambda wildcards: HCP379_STREAMLINE_COUNTS[
            str(wildcards.hcp_run)
        ],
    log:
        subject_log(
            "{unit}",
            "09_hcp379_{hcp_run}_{metric}_tcksample.log",
        ),
    threads: 8
    wildcard_constraints:
        hcp_run="primary_3m|primary_5m|primary_10m|independent_3m|independent_5m|independent_10m",
        metric="fa|md|rd|ad"
    shell:
        r"""
        set -euo pipefail
        mkdir -p "$(dirname {output:q})" "$(dirname {log:q})"
        export PATH={TOOL_PATH:q}
        tcksample {input.tracks:q} {input.image:q} {output:q} \
          -stat_tck mean -nthreads {threads} > {log:q} 2>&1
        test "$(python {HCP379_NUMERIC_VALUE_COUNT:q} {output:q} --expect {params.expected})" -eq {params.expected}
        """


rule hcp379_stability_tensor_connectome:
    input:
        tracks=hcp379_tracks,
        atlas=hcp379_corrected_nodes,
        samples=rules.hcp379_stability_tensor_samples.output,
    output:
        hcp379_path(
            "{unit}", "{hcp_run}", "matrices", "{metric}_mean.csv"
        ),
    log:
        subject_log(
            "{unit}",
            "09_hcp379_{hcp_run}_{metric}_connectome.log",
        ),
    threads: 8
    wildcard_constraints:
        hcp_run="primary_3m|primary_5m|primary_10m|independent_3m|independent_5m|independent_10m",
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


rule hcp379_stability_count_invnodevol:
    input:
        count_matrix=lambda wildcards: hcp379_matrix_for(wildcards, "count"),
        volumes=hcp379_node_volumes,
        atlas_qc=hcp379_atlas_qc,
    output:
        hcp379_path(
            "{unit}",
            "{hcp_run}",
            "matrices",
            "count_invnodevol.csv",
        ),
    log:
        subject_log(
            "{unit}", "09_hcp379_{hcp_run}_count_invnodevol.log"
        ),
    wildcard_constraints:
        hcp_run="primary_3m|primary_5m|primary_10m|independent_3m|independent_5m|independent_10m"
    run:
        import numpy as np

        Path(output[0]).parent.mkdir(parents=True, exist_ok=True)
        Path(log[0]).parent.mkdir(parents=True, exist_ok=True)
        count = np.loadtxt(input.count_matrix, delimiter=",")
        with Path(input.volumes).open(
            newline="", encoding="utf-8-sig"
        ) as handle:
            rows = list(csv.DictReader(handle))
        rows.sort(key=lambda row: int(row["node_id"]))
        if (
            count.shape != (379, 379)
            or len(rows) != 379
            or [int(row["node_id"]) for row in rows]
            != list(range(1, 380))
        ):
            raise ValueError("HCP379 count/node-volume contract differs")
        volumes = np.asarray(
            [float(row["volume_mm3"]) for row in rows], dtype=float
        )
        if not np.isfinite(volumes).all() or np.any(volumes <= 0):
            raise ValueError("HCP379 node volumes are nonfinite/nonpositive")
        denominator = volumes[:, None] + volumes[None, :]
        derived = 2.0 * count / denominator
        np.fill_diagonal(derived, 0.0)
        temporary = Path(str(output[0]) + ".partial")
        np.savetxt(temporary, derived, delimiter=",", fmt="%.12g")
        temporary.replace(output[0])
        Path(log[0]).write_text(
            "internal: derive HCP379 count_invnodevol\n"
            + json.dumps(
                {
                    "unit": str(wildcards.unit),
                    "run_id": str(wildcards.hcp_run),
                    "formula": (
                        "2*count_ij/(node_volume_i_mm3+node_volume_j_mm3)"
                    ),
                    "minimum_node_volume_mm3": float(volumes.min()),
                    "maximum_node_volume_mm3": float(volumes.max()),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )


rule hcp379_stability_run_record:
    input:
        tracks=hcp379_tracks,
        metadata=hcp379_metadata,
        weights=hcp379_weights,
        assignments=rules.hcp379_stability_count.output.assignments,
        count_matrix=lambda wildcards: hcp379_matrix_for(
            wildcards, "count"
        ),
        fd_sum=lambda wildcards: hcp379_matrix_for(wildcards, "fd_sum"),
        count_invnodevol=lambda wildcards: hcp379_matrix_for(
            wildcards, "count_invnodevol"
        ),
        len_mean=lambda wildcards: hcp379_matrix_for(
            wildcards, "len_mean"
        ),
        invlen_mean=lambda wildcards: hcp379_matrix_for(
            wildcards, "invlen_mean"
        ),
        fa_mean=lambda wildcards: hcp379_matrix_for(wildcards, "fa_mean"),
        md_mean=lambda wildcards: hcp379_matrix_for(wildcards, "md_mean"),
        rd_mean=lambda wildcards: hcp379_matrix_for(wildcards, "rd_mean"),
        ad_mean=lambda wildcards: hcp379_matrix_for(wildcards, "ad_mean"),
        source_parcellation=hcp379_source_parcellation,
        corrected_nodes=hcp379_corrected_nodes,
        node_volumes=hcp379_node_volumes,
        source_node_volumes=hcp379_source_node_volumes,
        atlas_qc=hcp379_atlas_qc,
        visual_overlay=hcp379_visual_overlay,
        atlas_result=lambda wildcards: hcp379_atlas_path(
            wildcards.unit, "atlas_result"
        ),
        corrected_atlas_gate=rules.hcp379_corrected_atlas_gate.output[0],
        t1_to_b0=lambda wildcards: hcp379_atlas_path(
            wildcards.unit, "selected_transform"
        ),
        b0_reference=lambda wildcards: hcp379_atlas_path(
            wildcards.unit, "b0_reference"
        ),
    output:
        hcp379_path("{unit}", "{hcp_run}", "run_record.json"),
    params:
        expected=lambda wildcards: HCP379_STREAMLINE_COUNTS[
            str(wildcards.hcp_run)
        ],
        seed_id=hcp379_seed_id,
        seed_class=hcp379_seed_class,
    log:
        subject_log("{unit}", "09_hcp379_{hcp_run}_record.log"),
    wildcard_constraints:
        hcp_run="primary_3m|primary_5m|primary_10m|independent_3m|independent_5m|independent_10m"
    run:
        Path(output[0]).parent.mkdir(parents=True, exist_ok=True)
        Path(log[0]).parent.mkdir(parents=True, exist_ok=True)
        def _single_input_path(name):
            value = getattr(input, name)
            while not isinstance(value, (str, Path)):
                values = list(value)
                if len(values) != 1:
                    raise ValueError(
                        f"{name}: expected one input path, found "
                        f"{len(values)}"
                    )
                value = values[0]
            return Path(str(value)).resolve()

        expected = int(params.expected)
        count_output = subprocess.run(
            [
                str(HCP379_VALIDATOR_PYTHON),
                str(HCP379_TCK_HEADER_COUNT),
                str(_single_input_path("tracks")),
                "--expect",
                str(expected),
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        actual_tracks = int(count_output.strip())

        weight_count = int(
            subprocess.run(
                [
                    str(HCP379_VALIDATOR_PYTHON),
                    str(HCP379_NUMERIC_VALUE_COUNT),
                    str(_single_input_path("weights")),
                    "--expect",
                    str(expected),
                ],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        assignment_count = 0
        assigned_count = 0
        with _single_input_path("assignments").open(
            encoding="utf-8"
        ) as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                fields = stripped.replace(",", " ").split()
                if len(fields) != 2:
                    raise ValueError(
                        "assignment row does not contain exactly two endpoints: "
                        f"line {line_number}"
                    )
                left, right = (int(value) for value in fields)
                if not 0 <= left <= 379 or not 0 <= right <= 379:
                    raise ValueError(
                        "assignment endpoint outside 0..379: "
                        f"line {line_number}"
                    )
                assignment_count += 1
                assigned_count += int(left > 0 and right > 0)
        if not actual_tracks == weight_count == assignment_count == expected:
            raise ValueError(
                "tractogram/weight/assignment counts differ: "
                f"{actual_tracks}/{weight_count}/"
                f"{assignment_count}/{expected}"
            )
        record = {
            "schema_version": "1.0.0",
            "record_type": (
                "diagnosis_blind_hcp379_phase_b_stability_run"
            ),
            "diagnosis_labels_used": False,
            "unit": str(wildcards.unit),
            "run_id": str(wildcards.hcp_run),
            "seed_id": str(params.seed_id),
            "seed_class": str(params.seed_class),
            "streamline_count": expected,
            "tractogram_streamline_count": actual_tracks,
            "sift2_weight_count": weight_count,
            "assignment_row_count": assignment_count,
            "assigned_streamline_count": assigned_count,
            "endpoint_assignment_fraction": (
                assigned_count / assignment_count
            ),
            "matrices": {
                name: str(
                    _single_input_path(
                        "count_matrix" if name == "count" else name
                    )
                )
                for name in HCP379_MATRIX_NAMES
            },
            "artifacts": {
                "tractogram": str(_single_input_path("tracks")),
                "sift2_weights": str(_single_input_path("weights")),
                "assignments": str(_single_input_path("assignments")),
                "tractography_metadata": str(
                    _single_input_path("metadata")
                ),
            },
            "atlas": {
                "source_parcellation": str(
                    _single_input_path("source_parcellation")
                ),
                "corrected_nodes": str(
                    _single_input_path("corrected_nodes")
                ),
                "node_volumes": str(
                    _single_input_path("node_volumes")
                ),
                "source_node_volumes": str(
                    Path(
                        hcp379_atlas_path(
                            wildcards.unit, "source_node_volumes"
                        )
                    ).resolve()
                ),
                "atlas_qc": str(_single_input_path("atlas_qc")),
                "visual_overlay": str(
                    _single_input_path("visual_overlay")
                ),
                "atlas_result": str(
                    _single_input_path("atlas_result")
                ),
                "corrected_atlas_gate": str(
                    _single_input_path("corrected_atlas_gate")
                ),
                "selected_t1_to_b0_transform_artifact": str(
                    _single_input_path("t1_to_b0")
                ),
                "corrected_b0_reference": str(
                    _single_input_path("b0_reference")
                ),
            },
        }
        temporary = Path(str(output[0]) + ".partial")
        temporary.write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output[0])
        Path(log[0]).write_text(
            "internal: diagnosis-blind HCP379 stability run record\n"
            + json.dumps(record, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )


HCP379_RUN_RECORDS = [
    hcp379_path(unit, run_id, "run_record.json")
    for unit in PHASE_B_UNITS
    for run_id in HCP379_RUNS
]
HCP379_RUN_RECORD_ARGS = " ".join(
    f"--run-record {shlex.quote(path)}" for path in HCP379_RUN_RECORDS
)


rule hcp379_stability_manifest:
    input:
        records=HCP379_RUN_RECORDS,
        builder=str(HCP379_BUILDER),
    output:
        run_path(
            "publication", "hcp379_recipe_stability_manifest.json"
        ),
    params:
        record_args=HCP379_RUN_RECORD_ARGS,
    log:
        run_path("logs", "09_hcp379_stability_manifest.log"),
    shell:
        r"""
        set -euo pipefail
        mkdir -p "$(dirname {output:q})" "$(dirname {log:q})"
        {HCP379_VALIDATOR_PYTHON:q} {input.builder:q} \
          {params.record_args} --output {output:q} > {log:q} 2>&1
        """


rule hcp379_stability_validation:
    input:
        manifest=rules.hcp379_stability_manifest.output,
        validator=str(HCP379_VALIDATOR),
        contract=str(HCP379_CONTRACT),
    output:
        run_path(
            "publication", "hcp379_recipe_stability_validation.json"
        ),
    log:
        run_path("logs", "09_hcp379_stability_validation.log"),
    shell:
        r"""
        set -euo pipefail
        mkdir -p "$(dirname {output:q})" "$(dirname {log:q})"
        {HCP379_VALIDATOR_PYTHON:q} {input.validator:q} \
          --manifest {input.manifest:q} --contract {input.contract:q} \
          --output {output:q} --require-pass > {log:q} 2>&1
        """


rule hcp379_stability_evidence:
    input:
        rules.hcp379_stability_validation.output,
