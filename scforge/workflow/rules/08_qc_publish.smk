"""Strict QC, mixed terminal provenance, and separated publication manifests."""


rule matrix_qc:
    input:
        count_matrix=matrix_path("{unit}", "count"),
        fd_sum=matrix_path("{unit}", "fd_sum"),
        count_invnodevol=matrix_path("{unit}", "count_invnodevol"),
        len_mean=matrix_path("{unit}", "len_mean"),
        invlen_mean=matrix_path("{unit}", "invlen_mean"),
        fa_mean=matrix_path("{unit}", "fa_mean"),
        md_mean=matrix_path("{unit}", "md_mean"),
        rd_mean=matrix_path("{unit}", "rd_mean"),
        ad_mean=matrix_path("{unit}", "ad_mean"),
        node_volumes=rules.count_invnodevol.output.node_volumes,
        assignments=rules.connectome_count.output.assignments,
        tracks=rules.tractography_10m.output.tracks,
        weights=rules.sift2_weights.output.weights,
    output:
        subject_path("{unit}", "08_qc", "matrix_qc.json"),
    log:
        subject_log("{unit}", "08_matrix_qc.log"),
    run:
        import csv
        import re as _re
        import subprocess
        import sys

        import numpy as np

        sys.path.insert(0, str(PROJECT_ROOT / "scforge"))
        from scforge.qc import qc_connectome_bundle

        Path(output[0]).parent.mkdir(parents=True, exist_ok=True)
        Path(log[0]).parent.mkdir(parents=True, exist_ok=True)
        if (
            config["qc"]["matrix"][
                "zero_count_nodes_used_as_automatic_exclusion"
            ]
            is not False
            or config["qc"]["assignment"][
                "support_metrics_used_as_automatic_exclusion"
            ]
            is not False
        ):
            raise ValueError(
                "pre-H04 support metrics must remain quantitative, non-gating QC"
            )
        matrices = {
            name: str(input.count_matrix if name == "count" else getattr(input, name))
            for name in EXPECTED_MATRICES
        }
        with Path(input.node_volumes).open(newline="", encoding="utf-8") as handle:
            volume_rows = sorted(csv.DictReader(handle), key=lambda row: int(row["node_index"]))
        volumes = np.array([float(row["node_volume_mm3"]) for row in volume_rows], dtype=float)
        result = qc_connectome_bundle(
            matrices,
            volumes,
            expected_nodes=int(config["atlas"]["expected_nodes"]),
            symmetry_tolerance=float(config["qc"]["matrix"]["symmetric_absolute_tolerance"]),
            zero_diagonal_tolerance=float(config["qc"]["matrix"]["zero_diagonal_absolute_tolerance"]),
            nonnegative_tolerance=float(config["qc"]["matrix"]["nonnegative_absolute_tolerance"]),
            integer_tolerance=float(config["qc"]["matrix"]["count_integer_absolute_tolerance"]),
            duplicate_rtol=float(config["qc"]["matrix"]["count_not_allclose_to_fd_sum"]["rtol"]),
            duplicate_atol=float(config["qc"]["matrix"]["count_not_allclose_to_fd_sum"]["atol"]),
            tensor_identity_tolerance=float(config["qc"]["tensor_edge_values"]["tensor_identity_absolute_tolerance_mm2_per_s"]),
            diffusivity_hard_max_mm2_per_s=float(config["qc"]["tensor_edge_values"]["diffusivity_hard_max_mm2_per_s"]),
        )
        record = result.to_dict()
        failures = list(record["failures"])
        quantitative_support_flags = []

        assignments = np.loadtxt(input.assignments, dtype=int)
        if assignments.ndim == 1:
            assignments = assignments.reshape(1, -1)
        if assignments.shape[1] != 2:
            failures.append(f"assignments:expected_two_columns_found_{assignments.shape}")
        track_command = [str(MRTRIX_BIN / "tckinfo"), str(input.tracks), "-count"]
        completed = subprocess.run(track_command, text=True, capture_output=True, check=False)
        match = _re.search(r"(?m)^\s*count:\s*([0-9]+)\s*$", completed.stdout + completed.stderr)
        track_count = int(match.group(1)) if completed.returncode == 0 and match else -1
        if track_count != int(config["tractography"]["select_streamlines"]):
            failures.append(f"tractogram_count:{track_count}")
        weight_count = sum(1 for line in Path(input.weights).open(encoding="utf-8") if line.strip())
        if weight_count != track_count:
            failures.append(f"sift2_weight_count:{weight_count}_track_count:{track_count}")
        if assignments.shape[0] != track_count:
            failures.append(f"assignment_count:{assignments.shape[0]}_track_count:{track_count}")

        if assignments.shape[1] == 2 and assignments.size:
            valid_endpoints = assignments > 0
            endpoint_fraction = float(valid_endpoints.sum() / assignments.size)
            assigned_nodes = assignments[valid_endpoints]
            unique_nodes = int(len(np.unique(assigned_nodes)))
            if assigned_nodes.size:
                counts = np.bincount(assigned_nodes, minlength=167)[1:167]
                top5_fraction = float(np.sort(counts)[-5:].sum() / counts.sum())
            else:
                top5_fraction = 1.0
            inter_node = int(np.count_nonzero(
                (assignments[:, 0] > 0)
                & (assignments[:, 1] > 0)
                & (assignments[:, 0] != assignments[:, 1])
            ))
            count_matrix = np.loadtxt(input.count_matrix, delimiter=",")
            upper_sum = float(np.triu(count_matrix, 1).sum())
            if not np.isclose(upper_sum, inter_node, rtol=0.0, atol=1e-6):
                failures.append(f"assignment_ledger_count_mismatch:{upper_sum}_vs_{inter_node}")
            if endpoint_fraction < float(
                config["qc"]["assignment"][
                    "candidate_reporting_threshold_minimum_endpoint_assignment_fraction"
                ]
            ):
                quantitative_support_flags.append(
                    f"endpoint_assignment_fraction_below_candidate_threshold:{endpoint_fraction}"
                )
            if unique_nodes < int(
                config["qc"]["assignment"][
                    "candidate_reporting_threshold_minimum_unique_assigned_nodes"
                ]
            ):
                quantitative_support_flags.append(
                    f"unique_assigned_nodes_below_candidate_threshold:{unique_nodes}"
                )
            if top5_fraction > float(
                config["qc"]["assignment"][
                    "candidate_reporting_threshold_maximum_top5_endpoint_fraction"
                ]
            ):
                quantitative_support_flags.append(
                    f"top5_endpoint_fraction_above_candidate_threshold:{top5_fraction}"
                )
        else:
            endpoint_fraction = 0.0
            unique_nodes = 0
            top5_fraction = 1.0
            inter_node = 0

        record.update(
            {
                "status": "PASS" if not failures else "FAIL",
                "pass_strict": not failures,
                "failures": list(dict.fromkeys(failures)),
                "density_is_reporting_only": True,
                "density_used_for_route_or_inclusion": False,
                "support_metrics_are_reporting_only_before_h04_sap": config[
                    "qc"
                ]["assignment"]["support_metrics_used_as_automatic_exclusion"]
                is False,
                "support_thresholds_release_state": config["qc"]["assignment"][
                    "support_thresholds_release_state"
                ],
                "quantitative_support_flags": quantitative_support_flags,
                "tractogram_streamlines": track_count,
                "sift2_weight_count": weight_count,
                "assignment_rows": int(assignments.shape[0]),
                "inter_node_assigned_streamlines": inter_node,
                "endpoint_assignment_fraction": endpoint_fraction,
                "unique_assigned_nodes": unique_nodes,
                "top5_endpoint_fraction": top5_fraction,
                "command": " ".join(track_command),
            }
        )
        Path(log[0]).write_text(
            "$ " + " ".join(track_command) + "\n" + completed.stdout + completed.stderr
            + json.dumps(record, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        tmp = Path(str(output[0]) + ".partial")
        tmp.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        tmp.replace(output[0])


rule provenance_sidecar:
    input:
        matrix_qc=rules.matrix_qc.output,
        spatial_qc=rules.spatial_quantitative_qc.output,
        tractography_preflight=rules.tractography_preflight.output,
        input_contract=rules.input_contract_gate.output[0],
        manifest_validation=rules.freeze_manifest.output.validation,
        execution_preflight=rules.execution_preflight.output.report,
        manifest=rules.freeze_manifest.output.manifest,
        normative_config=str(NORMATIVE_CONFIG),
        resolved_config=str(RESOLVED_RUN_CONFIG),
        environment=str(ENVIRONMENT_CONTRACT),
        workflow_source=str(WORKFLOW_SOURCE_MANIFEST),
        provenance_schema=str(PROVENANCE_SCHEMA),
        run_context=str(RUN_CONTEXT_PATH),
        attempt_context=str(ATTEMPT_CONTEXT_PATH),
        workflow_lock=str(Path(ENVIRONMENT_LOCK["workflow"]["requirements_lock"]["path"])),
        wheel_artifacts=str(Path(ENVIRONMENT_LOCK["workflow"]["wheel_artifact_manifest"]["path"])),
        dictionary=str(MATRIX_DICTIONARY),
        mni_template=str(MNI_TEMPLATE),
        atlas_node_table=str(ATLAS_NODE_TABLE),
        atlas_node_map=str(ATLAS_NODE_MAP),
        human_review_manifest=str(HUMAN_QC_MANIFEST),
        dwi=rules.normalize_dwi_source.output.dwi,
        bvec=rules.normalize_dwi_source.output.bvec,
        bval=rules.normalize_dwi_source.output.bval,
        dwi_metadata=rules.normalize_dwi_source.output.metadata,
        dwi_normalization=rules.normalize_dwi_source.output.record,
        t1=rules.normalize_t1_source.output.t1,
        t1_normalization=rules.normalize_t1_source.output.record,
        tracks=rules.tractography_10m.output.tracks,
        tractography_metadata=rules.tractography_10m.output.metadata,
        weights=rules.sift2_weights.output.weights,
        mu=rules.sift2_weights.output.mu,
        sift2_iterations=rules.sift2_weights.output.iterations,
        assignments=rules.connectome_count.output.assignments,
        node_volumes=rules.count_invnodevol.output.node_volumes,
        gradient_qc=rules.gradient_contract.output,
        atlas_qc=rules.atlas_contract_qc.output,
        b0_to_t1_fsl=rules.b0_to_t1_bbr.output.matrix,
        t1_to_b0_fsl=rules.invert_bbr_transform.output.fsl,
        t1_to_b0_mrtrix=rules.invert_bbr_transform.output.mrtrix,
        t1_to_b0_itk=rules.invert_bbr_transform.output.itk,
        mni_to_t1_affine=rules.mni_to_t1_nonlinear.output.affine,
        mni_to_t1_warp=rules.mni_to_t1_nonlinear.output.warp,
        mni_to_t1_inverse_warp=rules.mni_to_t1_nonlinear.output.inverse_warp,
        atlas_dwi=rules.aal3_to_dwi_single_resample.output,
        five_tt_dwi=rules.five_tt_dwi.output.five_tt,
        response_wm=rules.pooled_response.output.wm,
        response_gm=rules.pooled_response.output.gm,
        response_csf=rules.pooled_response.output.csf,
        wm_fod_norm=rules.mtnormalise.output.wm,
        gm_norm=rules.mtnormalise.output.gm,
        csf_norm=rules.mtnormalise.output.csf,
        tensor=rules.tensor_fit.output,
        fa=rules.tensor_metrics.output.fa,
        md=rules.tensor_metrics.output.md,
        rd=rules.tensor_metrics.output.rd,
        ad=rules.tensor_metrics.output.ad,
        overlay_b0_t1=rules.visual_review_bundle.output.b0_t1,
        overlay_b0_5tt=rules.visual_review_bundle.output.b0_5tt,
        overlay_b0_atlas=rules.visual_review_bundle.output.b0_atlas,
        matrices=lambda wc: all_matrix_paths(wc.unit),
    output:
        subject_path("{unit}", "08_qc", "terminal_record.json"),
    log:
        subject_log("{unit}", "08_provenance.log"),
    run:
        import datetime
        import sys

        Path(output[0]).parent.mkdir(parents=True, exist_ok=True)
        Path(log[0]).parent.mkdir(parents=True, exist_ok=True)
        sys.path.insert(0, str(PROJECT_ROOT / "scforge"))
        from scforge.provenance import (
            REQUIRED_OUTCOMES,
            file_record,
            matrix_qc_outcome_invalidity,
            validate_schema,
            validate_terminal_semantics,
            write_immutable_json,
        )

        row = dict(ROW_BY_UNIT[str(wildcards.unit)])
        source_paths = {
            "dwi_raw_mif": input.dwi,
            "dwi_bvec": input.bvec,
            "dwi_bval": input.bval,
            "dwi_source_metadata": input.dwi_metadata,
            "dwi_normalization": input.dwi_normalization,
            "t1_native_nifti": input.t1,
            "t1_normalization": input.t1_normalization,
        }
        complete_output_paths = {
            "tracks": input.tracks,
            "sift2_weights": input.weights,
            "sift2_mu": input.mu,
            **{metric: matrix_path(wildcards.unit, metric) for metric in EXPECTED_MATRICES},
        }
        intermediate_paths = {
            "b0_to_t1_fsl": input.b0_to_t1_fsl,
            "t1_to_b0_fsl": input.t1_to_b0_fsl,
            "t1_to_b0_mrtrix": input.t1_to_b0_mrtrix,
            "t1_to_b0_itk": input.t1_to_b0_itk,
            "mni_to_t1_affine": input.mni_to_t1_affine,
            "mni_to_t1_warp": input.mni_to_t1_warp,
            "mni_to_t1_inverse_warp": input.mni_to_t1_inverse_warp,
            "atlas_dwi": input.atlas_dwi,
            "five_tt_dwi": input.five_tt_dwi,
            "pooled_response_wm": input.response_wm,
            "pooled_response_gm": input.response_gm,
            "pooled_response_csf": input.response_csf,
            "wm_fod_norm": input.wm_fod_norm,
            "gm_norm": input.gm_norm,
            "csf_norm": input.csf_norm,
            "tensor": input.tensor,
            "fa": input.fa,
            "md": input.md,
            "rd": input.rd,
            "ad": input.ad,
            "sift2_iterations": input.sift2_iterations,
            "assignments": input.assignments,
            "node_volumes": input.node_volumes,
            "overlay_b0_t1": input.overlay_b0_t1,
            "overlay_b0_5tt": input.overlay_b0_5tt,
            "overlay_b0_atlas": input.overlay_b0_atlas,
        }
        log_dir = Path(subject_path(wildcards.unit, "logs"))
        # Only completed per-unit logs are immutable at this point.  Shared run
        # logs can still grow while other units execute and are therefore
        # hashed later by the launcher completion attestation, never here.
        log_paths = sorted(log_dir.glob("*.log"))
        command_logs = [
            file_record(path)
            for path in log_paths
            if path.resolve() != Path(log[0]).resolve()
        ]
        matrix_qc = json.loads(Path(input.matrix_qc).read_text(encoding="utf-8"))
        spatial_qc = json.loads(Path(input.spatial_qc).read_text(encoding="utf-8"))
        preflight_qc = json.loads(Path(input.tractography_preflight).read_text(encoding="utf-8"))
        gradient_qc = json.loads(Path(input.gradient_qc).read_text(encoding="utf-8"))
        atlas_qc = json.loads(Path(input.atlas_qc).read_text(encoding="utf-8"))
        input_contract = json.loads(Path(input.input_contract).read_text(encoding="utf-8"))
        execution_preflight = json.loads(Path(input.execution_preflight).read_text(encoding="utf-8"))
        run_context = json.loads(Path(input.run_context).read_text(encoding="utf-8"))
        attempt_context = json.loads(Path(input.attempt_context).read_text(encoding="utf-8"))
        tractography_parameters = json.loads(
            Path(input.tractography_metadata).read_text(encoding="utf-8")
        )
        failures = []
        failed_qc_names = []
        for name, qc_record in (
            ("gradient", gradient_qc),
            ("atlas", atlas_qc),
            ("spatial", spatial_qc),
            ("preflight", preflight_qc),
            ("matrix", matrix_qc),
        ):
            if qc_record.get("status") != "PASS":
                failed_qc_names.append(name)
                failures.append(f"{name}_qc_not_pass")
        if preflight_qc.get("human_visual_qc") != "PASS":
            failed_qc_names.append("visual")
            failures.append("human_visual_qc_not_pass")
        if run_context.get("recipe_id") != RECIPE_ID or attempt_context.get("recipe_id") != RECIPE_ID:
            failed_qc_names.append("publication")
            failures.append("run_attempt_recipe_identity_differs")
        if attempt_context.get("run_id") != run_context.get("run_id"):
            failed_qc_names.append("publication")
            failures.append("run_attempt_ids_differ")
        if tractography_parameters.get("requested_streamlines") != tractography_parameters.get("actual_streamlines"):
            failed_qc_names.append("tractography")
            failures.append("requested_actual_tractography_counts_differ")
        failures.extend(str(reason) for reason in matrix_qc.get("failures", []))
        failures = list(dict.fromkeys(failures))

        internal_commands = []
        for command_record in (
            execution_preflight,
            gradient_qc,
            spatial_qc,
            preflight_qc,
            matrix_qc,
        ):
            commands = command_record.get("commands", [])
            if isinstance(commands, list):
                internal_commands.extend(str(command) for command in commands)
            elif command_record.get("command"):
                internal_commands.append(str(command_record["command"]))

        generated_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
        source_records = {key: file_record(path) for key, path in source_paths.items()}
        computed_staged_input_hashes = {
            key: source_records[key]["sha256"]
            for key in (
                "dwi_raw_mif",
                "dwi_bvec",
                "dwi_bval",
                "dwi_source_metadata",
                "t1_native_nifti",
            )
        }
        staged_input_hashes = dict(input_contract.get("staged_input_hashes", {}))
        source_identity_hashes = dict(input_contract.get("source_identity_hashes", {}))
        if staged_input_hashes != computed_staged_input_hashes:
            raise ValueError("Input-contract staged hashes differ from terminal artifacts")
        expected_source_hashes = {
            key: row[key]
            for key in (
                "dti_raw_bundle_sha256",
                "t1_raw_bundle_sha256",
                "pair_content_bundle_sha256",
            )
        }
        if source_identity_hashes != expected_source_hashes:
            raise ValueError("Input-contract raw source hashes differ from acquisition manifest")

        all_upstream_pass = not any(
            name in failed_qc_names
            for name in ("gradient", "atlas", "spatial", "preflight", "visual", "publication")
        )
        matrix_invalidity = matrix_qc_outcome_invalidity(
            matrix_qc, required_matrices=EXPECTED_MATRICES
        )
        matrix_pass = matrix_qc.get("status") == "PASS" and not any(
            matrix_invalidity.values()
        )
        tract_failure_prefixes = (
            "tractogram_",
            "sift2_",
            "assignment_",
            "assignments:",
            "endpoint_",
            "unique_assigned_",
            "top5_",
            "requested_actual_tractography_",
        )
        tractography_pass = all_upstream_pass and not any(
            str(reason).startswith(tract_failure_prefixes) for reason in failures
        )
        biological_inference_release = (
            EXECUTION_BINDING.get("execution_scope") == "full"
            and config.get("provenance", {}).get(
                "biological_inference_release_authorized"
            )
            is True
        )

        if (
            all_upstream_pass
            and matrix_pass
            and tractography_pass
            and biological_inference_release
        ):
            terminal_status = "PASS"
            terminal_stage = "complete"
            primary_reason = None
            outcome_validity = {
                name: {"status": "VALID", "reason": None}
                for name in REQUIRED_OUTCOMES
            }
            for metric in EXPECTED_MATRICES:
                outcome_validity[metric]["artifact"] = file_record(
                    complete_output_paths[metric]
                )
            valid_output_paths = complete_output_paths
        elif all_upstream_pass:
            terminal_status = "PARTIAL"
            if not tractography_pass:
                terminal_stage = "tractography"
            elif not matrix_pass:
                terminal_stage = "matrix_qc"
            else:
                terminal_stage = "publication"
            detailed_failures = [
                reason for reason in failures if not str(reason).endswith("_qc_not_pass")
            ]
            if matrix_pass and tractography_pass and not biological_inference_release:
                primary_reason = (
                    "biological_inference_locked_pending_diagnosis_blind_h04_sap"
                )
            else:
                primary_reason = (
                    detailed_failures[0]
                    if detailed_failures
                    else (failures[0] if failures else "strict_outcome_qc_not_pass")
                )
            outcome_validity = {
                name: {"status": "NA", "reason": primary_reason}
                for name in REQUIRED_OUTCOMES
            }
            outcome_validity["processing"] = {"status": "VALID", "reason": None}
            valid_output_paths = {}
            if tractography_pass:
                outcome_validity["tractography"] = {"status": "VALID", "reason": None}
                valid_output_paths = {
                    key: complete_output_paths[key]
                    for key in ("tracks", "sift2_weights", "sift2_mu")
                }
                for metric in EXPECTED_MATRICES:
                    metric_reasons = matrix_invalidity[metric]
                    if metric_reasons:
                        outcome_validity[metric] = {
                            "status": "NA",
                            "reason": ";".join(metric_reasons),
                        }
                    else:
                        artifact = file_record(complete_output_paths[metric])
                        outcome_validity[metric] = {
                            "status": "VALID",
                            "reason": None,
                            "artifact": artifact,
                        }
                        valid_output_paths[metric] = complete_output_paths[metric]
                if matrix_pass:
                    outcome_validity["tensor"] = {
                        "status": "VALID",
                        "reason": None,
                    }
                    outcome_validity["matrices"] = {
                        "status": "VALID",
                        "reason": None,
                    }
        else:
            terminal_status = "FAIL"
            if "visual" in failed_qc_names:
                terminal_stage = "visual_qc"
            elif "publication" in failed_qc_names:
                terminal_stage = "publication"
            elif "preflight" in failed_qc_names:
                terminal_stage = "tractography_preflight"
            elif "atlas" in failed_qc_names:
                terminal_stage = "atlas_warp"
            elif "spatial" in failed_qc_names:
                terminal_stage = "registration"
            else:
                terminal_stage = "dwi_preprocessing"
            primary_reason = failures[0] if failures else "strict_upstream_qc_not_pass"
            outcome_validity = {
                name: {"status": "NA", "reason": primary_reason}
                for name in REQUIRED_OUTCOMES
            }
            valid_output_paths = {}

        record = {
            "schema_version": "2.0.0",
            "record_type": "connectome_subject_terminal_state",
            "status": terminal_status,
            "generated_utc": generated_utc,
            "started_utc": attempt_context["started_utc"],
            "ended_utc": generated_utc,
            "recipe_id": RECIPE_ID,
            "run_id": run_context["run_id"],
            "unit": str(wildcards.unit),
            "terminal_stage": terminal_stage,
            "primary_failure_reason": primary_reason,
            "secondary_flags": list(
                dict.fromkeys(
                    [
                        *(reason for reason in failures if reason != primary_reason),
                        *(
                            str(flag)
                            for flag in matrix_qc.get(
                                "quantitative_support_flags", []
                            )
                        ),
                    ]
                )
            ),
            "source_identity": row,
            "source_identity_hashes": source_identity_hashes,
            "staged_input_hashes": staged_input_hashes,
            "source_files": source_records,
            "contract_files": {
                "input_contract": file_record(input.input_contract),
                "manifest": file_record(input.manifest),
                "manifest_validation": file_record(input.manifest_validation),
                "normative_config": file_record(input.normative_config),
                "resolved_run_config": file_record(input.resolved_config),
                "environment": file_record(input.environment),
                "workflow_source_manifest": file_record(input.workflow_source),
                "execution_preflight": file_record(input.execution_preflight),
                "run_context": file_record(input.run_context),
                "attempt_context": file_record(input.attempt_context),
                "workflow_requirements_lock": file_record(input.workflow_lock),
                "wheel_artifact_manifest": file_record(input.wheel_artifacts),
                "matrix_dictionary": file_record(input.dictionary),
                "mni_template": file_record(input.mni_template),
                "atlas_node_table": file_record(input.atlas_node_table),
                "atlas_node_map": file_record(input.atlas_node_map),
                "human_review_manifest": file_record(input.human_review_manifest),
                "provenance_schema": file_record(input.provenance_schema),
            },
            "intermediate_files": {
                key: file_record(path) for key, path in intermediate_paths.items()
            },
            "outputs": {key: file_record(path) for key, path in valid_output_paths.items()},
            "outcome_validity": outcome_validity,
            "execution": {
                "attempt_started_utc": attempt_context["started_utc"],
                "terminal_utc": generated_utc,
                "host": run_context["host"],
                "tool_versions": execution_preflight["versions"],
                "snakemake_invocation": attempt_context["snakemake_invocation"],
                "command_capture": {
                    "mode": "snakemake_printshellcmds_combined_log",
                    "printshellcmds": True,
                    "final_log_hash_in_run_completion": True,
                },
                "command_logs": command_logs,
                "internal_commands": internal_commands,
            },
            "processing_contract": {
                "registration": config["registration"],
                "atlas": config["atlas"],
                "fod": config["fod"],
                "tensor": config["tensor"],
                "tractography": tractography_parameters,
                "sift2": config["sift2"],
                "assignment": config["connectome"]["assignment"],
            },
            "qc": {
                "gradient": gradient_qc,
                "atlas": atlas_qc,
                "matrix": matrix_qc,
                "spatial": spatial_qc,
                "preflight": preflight_qc,
                "input_contract": input_contract,
            },
            "safety": {
                "production_overwrite": False,
                "automatic_fallbacks": False,
                "density_route_selection": False,
                "invalid_values_replaced": False,
            },
        }
        schema_errors = validate_schema(record, input.provenance_schema)
        semantic_errors = validate_terminal_semantics(
            record,
            required_outcomes=REQUIRED_OUTCOMES,
            required_matrices=EXPECTED_MATRICES,
        )
        if schema_errors or semantic_errors:
            raise ValueError(
                "Terminal provenance validation failed: "
                + "; ".join(schema_errors + semantic_errors)
            )
        write_immutable_json(output[0], record)
        Path(log[0]).write_text(
            "internal: schema-validate and immutably create subject terminal provenance\n"
            + json.dumps(
                {
                    "status": terminal_status,
                    "schema_version": "2.0.0",
                    "run_id": run_context["run_id"],
                    "terminal_stage": terminal_stage,
                    "primary_failure_reason": primary_reason,
                    "command_log_count": len(command_logs),
                    "intermediate_file_count": len(intermediate_paths),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )


rule publish_manifest:
    input:
        sidecars=expand(subject_path("{unit}", "08_qc", "terminal_record.json"), unit=UNITS),
        schema=str(PROVENANCE_SCHEMA),
        ledger_schema=str(RUN_LEDGER_SCHEMA),
        run_context=str(RUN_CONTEXT_PATH),
    output:
        manifest=run_path("publication", "analysis_ready_manifest.csv"),
        outcome_manifest=run_path("publication", "outcome_validity_manifest.csv"),
        ledger=run_path("publication", "run_ledger.jsonl"),
        ledger_manifest=run_path("publication", "run_ledger_manifest.json"),
    log:
        run_path("logs", "08_publish_manifest.log"),
    run:
        import datetime
        import sys

        Path(output.manifest).parent.mkdir(parents=True, exist_ok=True)
        Path(log[0]).parent.mkdir(parents=True, exist_ok=True)
        sys.path.insert(0, str(PROJECT_ROOT / "scforge"))
        from scforge.provenance import (
            publish_terminal_ledger,
        )

        run_context = json.loads(Path(input.run_context).read_text(encoding="utf-8"))
        generated_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
        ledger_manifest = publish_terminal_ledger(
            input.sidecars,
            provenance_schema_path=input.schema,
            ledger_schema_path=input.ledger_schema,
            run_context_path=input.run_context,
            expected_units=UNITS,
            required_matrices=EXPECTED_MATRICES,
            analysis_manifest_path=output.manifest,
            outcome_manifest_path=output.outcome_manifest,
            ledger_path=output.ledger,
            ledger_manifest_path=output.ledger_manifest,
            generated_utc=generated_utc,
        )
        Path(log[0]).write_text(
            "internal: publish complete mixed terminal-state attrition ledger; "
            f"summary={json.dumps(ledger_manifest['summary'], sort_keys=True)}; "
            f"generated_utc={generated_utc}; run_id={run_context['run_id']}\n"
            "No files were copied into historical or production derivative directories.\n",
            encoding="utf-8",
        )
