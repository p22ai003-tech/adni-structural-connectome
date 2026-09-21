"""Immutable manifest snapshot and execution-environment gate."""


rule freeze_manifest:
    input:
        manifest=str(MANIFEST_PATH),
        execution_manifest=str(EXECUTION_MANIFEST_PATH),
        config=str(NORMATIVE_CONFIG),
        dictionary=str(MATRIX_DICTIONARY),
        environment=str(ENVIRONMENT_CONTRACT),
        workflow_source=str(WORKFLOW_SOURCE_MANIFEST),
        acquisition_schema=str(ACQUISITION_SCHEMA),
        source_projection=str(SOURCE_METADATA_PROJECTION),
        source_projection_validation=str(SOURCE_METADATA_PROJECTION_VALIDATION),
        resolved_config=str(RESOLVED_RUN_CONFIG),
        run_context=str(RUN_CONTEXT_PATH),
        attempt_context=str(ATTEMPT_CONTEXT_PATH),
        provenance_schema=str(PROVENANCE_SCHEMA),
    output:
        manifest=run_path("contract", "approved_acquisition_manifest.csv"),
        validation=run_path("contract", "manifest_validation.json"),
    log:
        run_path("logs", "00_freeze_manifest.log"),
    run:
        import datetime
        import shutil

        Path(output.manifest).parent.mkdir(parents=True, exist_ok=True)
        Path(log[0]).parent.mkdir(parents=True, exist_ok=True)
        source_hash = _sha256(input.manifest)
        configured_hash = str(config["inputs"]["approved_pair_manifest"].get("sha256", ""))
        record = {
            "status": "PASS",
            "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "recipe_id": RECIPE_ID,
            "manifest_source": str(MANIFEST_PATH),
            "manifest_sha256": source_hash,
            "configured_manifest_sha256": configured_hash,
            "config_path": str(NORMATIVE_CONFIG),
            "config_sha256": _sha256(input.config),
            "matrix_dictionary_path": str(MATRIX_DICTIONARY),
            "matrix_dictionary_sha256": _sha256(input.dictionary),
            "environment_contract_path": str(ENVIRONMENT_CONTRACT),
            "environment_contract_sha256": _sha256(input.environment),
            "workflow_source_manifest_path": str(WORKFLOW_SOURCE_MANIFEST),
            "workflow_source_manifest_sha256": _sha256(input.workflow_source),
            "acquisition_schema_path": str(ACQUISITION_SCHEMA),
            "acquisition_schema_sha256": _sha256(input.acquisition_schema),
            "source_metadata_projection_path": str(SOURCE_METADATA_PROJECTION),
            "source_metadata_projection_sha256": _sha256(input.source_projection),
            "source_metadata_projection_validation_path": str(
                SOURCE_METADATA_PROJECTION_VALIDATION
            ),
            "source_metadata_projection_validation_sha256": _sha256(
                input.source_projection_validation
            ),
            "resolved_run_config_path": str(RESOLVED_RUN_CONFIG),
            "resolved_run_config_sha256": _sha256(input.resolved_config),
            "run_context_path": str(RUN_CONTEXT_PATH),
            "run_context_sha256": _sha256(input.run_context),
            "attempt_context_path": str(ATTEMPT_CONTEXT_PATH),
            "attempt_context_sha256": _sha256(input.attempt_context),
            "provenance_schema_path": str(PROVENANCE_SCHEMA),
            "provenance_schema_sha256": _sha256(input.provenance_schema),
            "subject_series_units": list(UNITS),
            "scheduled_units": list(UNITS),
            "row_count": len(MANIFEST_ROWS),
            "parent_row_count": len(MANIFEST_ROWS),
            "execution_manifest_path": str(EXECUTION_MANIFEST_PATH),
            "execution_manifest_sha256": _sha256(input.execution_manifest),
            "execution_row_count": len(EXECUTION_ROWS),
            "execution_units": list(EXECUTION_UNITS),
            "phase_b_approved_units": list(PHASE_B_UNITS),
            "run_root": str(RUN_ROOT),
            "production_overwrite": False,
        }
        manifest_tmp = Path(str(output.manifest) + ".partial")
        validation_tmp = Path(str(output.validation) + ".partial")
        shutil.copyfile(input.manifest, manifest_tmp)
        manifest_tmp.replace(output.manifest)
        validation_tmp.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        validation_tmp.replace(output.validation)
        Path(log[0]).write_text(
            "internal: validate manifest schema, freeze exact bytes, and record SHA-256\n"
            + json.dumps(record, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )


rule execution_preflight:
    input:
        manifest=rules.freeze_manifest.output.manifest,
        validation=rules.freeze_manifest.output.validation,
        config=str(NORMATIVE_CONFIG),
        dictionary=str(MATRIX_DICTIONARY),
        environment=str(ENVIRONMENT_CONTRACT),
        workflow_source=str(WORKFLOW_SOURCE_MANIFEST),
        acquisition_schema=str(ACQUISITION_SCHEMA),
        locked_source_inventory=str(LOCKED_SOURCE_INVENTORY),
        source_projection=str(SOURCE_METADATA_PROJECTION),
        source_projection_validation=str(SOURCE_METADATA_PROJECTION_VALIDATION),
        resolved_config=str(RESOLVED_RUN_CONFIG),
        run_context=str(RUN_CONTEXT_PATH),
        attempt_context=str(ATTEMPT_CONTEXT_PATH),
        provenance_schema=str(PROVENANCE_SCHEMA),
        workflow_lock=str(Path(ENVIRONMENT_LOCK["workflow"]["requirements_lock"]["path"])),
        wheel_artifacts=str(Path(ENVIRONMENT_LOCK["workflow"]["wheel_artifact_manifest"]["path"])),
        ants_lock=str(Path(ENVIRONMENT_LOCK["ants"]["explicit_conda_lock"]["path"])),
        c3d_lock=str(C3D_LOCK),
        atlas=str(ATLAS_IMAGE),
        node_table=str(ATLAS_NODE_TABLE),
        node_map=str(ATLAS_NODE_MAP),
        mni=str(MNI_TEMPLATE),
    output:
        report=run_path("contract", "execution_preflight.json"),
        fsl_cpu_path=directory(run_path("contract", "fsl_cpu_path")),
        source_inventory=directory(run_path("contract", "source_runtime_inventory")),
    log:
        run_path("logs", "00_execution_preflight.log"),
    run:
        import datetime
        import shlex
        import subprocess

        Path(output.report).parent.mkdir(parents=True, exist_ok=True)
        Path(log[0]).parent.mkdir(parents=True, exist_ok=True)
        failures = []
        commands = []
        resolved_config = None

        contract = config.get("contract", {})
        if (
            contract.get("imaging_execution_authorized") is not False
            or contract.get("next_execution_gate") != "SL-H04A"
        ):
            failures.append(
                "normative recipe execution lock was mutated; bounded authority "
                "must come only from the signed H04A/H04B decision bindings"
            )
        h04a_authorization = EXECUTION_BINDING.get("h04a_authorization", {})
        h04a_expected = {
            "approval_mode": "H04A_BOUNDED_CANARY",
            "authorized_modes": [
                "response-calibration-phase-a",
                "pre-tractography-canary",
            ],
            "authorized_through": "pre_tractography_review_bundle_only",
            "maximum_cores": 4,
            "minimum_valid_response_calibration_units": 12,
            "minimum_valid_manufacturer_families": 2,
            "minimum_valid_t1_source_classes": 2,
            "required_valid_t1_source_classes": [
                "dicom_series",
                "nifti_single",
            ],
            "wall_clock_stop_hours": 72,
            "wall_clock_stop_seconds": 72 * 60 * 60,
            "storage_stop_gb": 150,
            "storage_stop_bytes": 150 * 1_000_000_000,
            "tractography_authorized": False,
            "matrix_generation_authorized": False,
            "full_cohort_authorized": False,
            "proposed_run_root": str(RUN_ROOT),
            "normative_config_sha256": _sha256(input.config),
            "workflow_source_manifest_sha256": _sha256(input.workflow_source),
            "environment_contract_sha256": _sha256(input.environment),
        }
        if not isinstance(h04a_authorization, dict):
            failures.append("execution binding lacks signed H04A authorization")
        else:
            for key, expected in h04a_expected.items():
                if h04a_authorization.get(key) != expected:
                    failures.append(f"signed H04A authorization differs at {key}")
            for key in ("approved_by", "approved_utc", "user_response"):
                if not isinstance(h04a_authorization.get(key), str) or not h04a_authorization.get(key):
                    failures.append(f"signed H04A authorization lacks {key}")
            if (
                LAUNCHER_MODE in {
                    "response-calibration-phase-a",
                    "pre-tractography-canary",
                }
                and LAUNCHER_MODE not in h04a_authorization.get("authorized_modes", [])
            ):
                failures.append("launcher mode is outside signed H04A scope")
            if LAUNCHER_MODE == "phase-b" and LAUNCHER_MODE in h04a_authorization.get("authorized_modes", []):
                failures.append("H04A authorization improperly includes phase-b")

        if str(ENVIRONMENT_LOCK.get("recipe_id")) != RECIPE_ID:
            failures.append("environment-contract recipe_id differs from normative recipe_id")
        policy = ENVIRONMENT_LOCK.get("execution_policy", {})
        if policy.get("fail_closed") is not True:
            failures.append("environment contract is not fail_closed")
        if policy.get("automatic_tool_or_algorithm_fallback") is not False:
            failures.append("environment contract permits automatic fallback")
        if policy.get("cuda_executor_allowed") is not False:
            failures.append("environment contract permits CUDA eddy")
        if Path(str(policy.get("eddy_executor", ""))).resolve() != EDDY_CPU:
            failures.append("environment contract eddy executor differs from locked eddy_cpu")
        normal_path = [str(path) for path in policy.get("normal_tool_path_order", [])]
        if normal_path != TOOL_PATH.split(":"):
            failures.append("runtime TOOL_PATH differs from the environment contract")
        if str(MRTRIX3TISSUE_BIN) in normal_path:
            failures.append("MRtrix3Tissue is forbidden on the normal global PATH")
        if not normal_path or normal_path[0] != str(MRTRIX_BIN):
            failures.append("locked MRtrix3 must be first on the normal PATH")
        if "/usr/bin" not in normal_path or str(FSL_BIN) not in normal_path or normal_path.index("/usr/bin") > normal_path.index(str(FSL_BIN)):
            failures.append("/usr/bin must precede FSL bin so MRtrix scripts use the locked system Python")

        try:
            normative_config = yaml.safe_load(Path(input.config).read_text(encoding="utf-8"))
            resolved_config = yaml.safe_load(Path(input.resolved_config).read_text(encoding="utf-8"))
            locked_run_context = json.loads(
                Path(input.run_context).read_text(encoding="utf-8")
            )
            validate_resolved_runtime_config(
                normative_config, resolved_config, locked_run_context,
                portable=bool(config.get("portable_mode", False)),
            )
        except Exception as exc:
            failures.append(f"unable to compare normative and resolved configs: {type(exc).__name__}:{exc}")

        try:
            run_context = json.loads(Path(input.run_context).read_text(encoding="utf-8"))
            attempt_context = json.loads(Path(input.attempt_context).read_text(encoding="utf-8"))
            if run_context.get("schema_version") != "1.0.0" or run_context.get("status") != "LOCKED":
                failures.append("run context is not a locked v1 record")
            if run_context.get("recipe_id") != RECIPE_ID:
                failures.append("run context recipe_id differs")
            if Path(str(run_context.get("run_root", ""))).resolve() != RUN_ROOT:
                failures.append("run context run_root differs")
            if run_context.get("acquisition_manifest", {}).get("sha256") != _sha256(input.manifest):
                failures.append("run context acquisition-manifest hash differs")
            if attempt_context.get("schema_version") != "1.0.0" or attempt_context.get("status") != "STARTED":
                failures.append("attempt context is not a STARTED v1 record")
            if attempt_context.get("run_id") != run_context.get("run_id"):
                failures.append("attempt and run context IDs differ")
            if attempt_context.get("recipe_id") != RECIPE_ID:
                failures.append("attempt context recipe_id differs")
            resolved_mode = (
                resolved_config.get("launcher_mode")
                if isinstance(resolved_config, dict)
                else None
            )
            if (
                resolved_mode not in {
                    "response-calibration-phase-a",
                    "pre-tractography-canary",
                    "phase-b",
                }
                or run_context.get("launcher_mode") != resolved_mode
                or attempt_context.get("launcher_mode") != resolved_mode
            ):
                failures.append("resolved/run/attempt launcher modes differ")
            attempt_resource_preflight = attempt_context.get(
                "h04a_resource_preflight"
            )
            if resolved_mode in {
                "response-calibration-phase-a",
                "pre-tractography-canary",
            }:
                if (
                    not isinstance(attempt_resource_preflight, dict)
                    or attempt_resource_preflight.get("status") != "PASS"
                    or attempt_resource_preflight.get("decision_sha256")
                    != EXECUTION_BINDING.get("execution_subset_decision", {}).get("sha256")
                    or attempt_resource_preflight.get("wall_clock_stop_seconds")
                    != h04a_authorization.get("wall_clock_stop_seconds")
                    or attempt_resource_preflight.get("storage_stop_bytes")
                    != h04a_authorization.get("storage_stop_bytes")
                    or attempt_resource_preflight.get("wall_clock_remaining_seconds", 0)
                    <= 0
                    or attempt_resource_preflight.get("storage_remaining_bytes", 0)
                    <= 0
                ):
                    failures.append("attempt lacks a valid signed H04A resource preflight")
            elif attempt_resource_preflight is not None:
                failures.append("phase-b improperly carries an H04A resource preflight")
            resolved_execution_binding = (
                resolved_config.get("execution_binding")
                if isinstance(resolved_config, dict)
                else None
            )
            if (
                not isinstance(resolved_execution_binding, dict)
                or resolved_execution_binding != run_context.get("execution_binding")
                or resolved_execution_binding != attempt_context.get("execution_binding")
            ):
                failures.append("canary execution binding differs across immutable contexts")
            resolved_binding = (
                resolved_config.get("response_calibration_binding")
                if isinstance(resolved_config, dict)
                else None
            )
            if resolved_mode == "response-calibration-phase-a":
                if (
                    resolved_binding is not None
                    or "response_calibration_binding" in run_context
                    or "response_calibration_binding" in attempt_context
                ):
                    failures.append("phase A contains a phase-B calibration binding")
            elif (
                not isinstance(resolved_binding, dict)
                or resolved_binding != run_context.get("response_calibration_binding")
                or resolved_binding != attempt_context.get("response_calibration_binding")
            ):
                failures.append("phase-B calibration binding differs across immutable contexts")
            resolved_continuation = (
                resolved_config.get("tractography_continuation_binding")
                if isinstance(resolved_config, dict)
                else None
            )
            if resolved_mode == "phase-b":
                if (
                    not isinstance(resolved_continuation, dict)
                    or resolved_continuation
                    != run_context.get("tractography_continuation_binding")
                    or resolved_continuation
                    != attempt_context.get("tractography_continuation_binding")
                ):
                    failures.append("tractography continuation binding differs across immutable contexts")
            elif (
                resolved_continuation is not None
                or "tractography_continuation_binding" in run_context
                or "tractography_continuation_binding" in attempt_context
            ):
                failures.append("pre-tractography mode contains continuation binding")
            if attempt_context.get("run_context", {}).get("sha256") != _sha256(input.run_context):
                failures.append("attempt context run-context hash differs")
            if attempt_context.get("resolved_run_config", {}).get("sha256") != _sha256(input.resolved_config):
                failures.append("attempt context resolved-config hash differs")
            invocation = [str(item) for item in attempt_context.get("snakemake_invocation", [])]
            for required_flag in ("--printshellcmds", "--snakefile", "--configfile"):
                if required_flag not in invocation:
                    failures.append(f"attempt invocation omits {required_flag}")
            try:
                cores_index = invocation.index("--cores")
                invocation_cores = int(invocation[cores_index + 1])
            except (ValueError, IndexError):
                failures.append("attempt invocation has no valid --cores value")
            else:
                if invocation_cores < 1:
                    failures.append("attempt invocation --cores must be positive")
                if (
                    resolved_mode in {
                        "response-calibration-phase-a",
                        "pre-tractography-canary",
                    }
                    and invocation_cores > int(h04a_authorization.get("maximum_cores", 0))
                ):
                    failures.append("attempt invocation exceeds signed H04A core maximum")
        except Exception as exc:
            failures.append(f"invalid run/attempt context: {type(exc).__name__}:{exc}")

        configured_manifest_hash = str(config["inputs"]["approved_pair_manifest"].get("sha256", ""))
        actual_manifest_hash = _sha256(input.manifest)
        if not re.fullmatch(r"[0-9a-f]{64}", configured_manifest_hash.lower()):
            failures.append("approved_pair_manifest.sha256 is not frozen")
        elif configured_manifest_hash.lower() != actual_manifest_hash:
            failures.append("approved_pair_manifest.sha256 does not match the frozen manifest")

        expected_static_hashes = {
            str(ATLAS_IMAGE): str(config["atlas"]["contiguous_source_image_sha256"]),
            str(ATLAS_NODE_TABLE): str(config["atlas"]["node_table_sha256"]),
            str(ATLAS_NODE_MAP): str(config["atlas"]["node_map_sha256"]),
            str(MNI_TEMPLATE): str(config["registration"]["mni_to_t1"]["template_sha256"]),
            str(ACQUISITION_SCHEMA): str(config["inputs"]["acquisition_schema"]["sha256"]),
            str(SOURCE_METADATA_PROJECTION): str(
                config["inputs"]["source_metadata_projection"]["sha256"]
            ),
            str(SOURCE_METADATA_PROJECTION_VALIDATION): str(
                config["inputs"]["source_metadata_projection"]["validation_sha256"]
            ),
        }
        for path, expected in expected_static_hashes.items():
            if _sha256(path).lower() != expected.lower():
                failures.append(f"static input SHA-256 mismatch: {path}")

        locked_files = {
            str(Path(ENVIRONMENT_LOCK["workflow"]["python"]["path"])): str(ENVIRONMENT_LOCK["workflow"]["python"]["sha256"]),
            str(Path(ENVIRONMENT_LOCK["workflow"]["snakemake"]["path"])): str(ENVIRONMENT_LOCK["workflow"]["snakemake"]["sha256"]),
            str(Path(ENVIRONMENT_LOCK["workflow"]["requirements_lock"]["path"])): str(ENVIRONMENT_LOCK["workflow"]["requirements_lock"]["sha256"]),
            str(Path(ENVIRONMENT_LOCK["workflow"]["wheel_artifact_manifest"]["path"])): str(ENVIRONMENT_LOCK["workflow"]["wheel_artifact_manifest"]["sha256"]),
            str(Path(ENVIRONMENT_LOCK["workflow"]["source_manifest"]["path"])): str(ENVIRONMENT_LOCK["workflow"]["source_manifest"]["sha256"]),
            str(Path(ENVIRONMENT_LOCK["execution_policy"]["mrtrix_script_python"]["path"])): str(ENVIRONMENT_LOCK["execution_policy"]["mrtrix_script_python"]["sha256"]),
            str(Path(ENVIRONMENT_LOCK["ants"]["explicit_conda_lock"]["path"])): str(ENVIRONMENT_LOCK["ants"]["explicit_conda_lock"]["sha256"]),
            str(C3D_LOCK): str(ENVIRONMENT_LOCK["convert3d"]["explicit_conda_lock"]["sha256"]),
        }
        locked_files[str(PROVENANCE_SCHEMA)] = _sha256(input.provenance_schema)
        # In portable mode the toolchain is checked for presence and version
        # (below) rather than by byte hash. Compiled binaries differ between
        # builds of the same release, and the interpreter and conda paths are
        # machine-specific, so hash-pinning them makes the workflow unrunnable
        # anywhere but the machine the contract was written on. Everything that
        # ships WITH the repository -- the atlas, the node tables, the schemas,
        # the MNI template -- is still hash-checked above, because those bytes
        # must be identical everywhere for the result to be comparable.
        portable = bool(config.get("portable_mode", False))
        tool_binaries = {}
        for section in ("mrtrix3", "mrtrix3tissue", "fsl", "ants", "convert3d"):
            for details in ENVIRONMENT_LOCK[section].get("binaries", {}).values():
                tool_binaries[str(Path(details["path"]))] = str(details["sha256"])
        if not portable:
            locked_files.update(tool_binaries)
        for path, expected in locked_files.items():
            candidate = Path(path)
            if not candidate.is_file():
                failures.append(f"locked environment file missing: {path}")
            elif not portable and _sha256(candidate).lower() != expected.lower():
                failures.append(f"locked environment SHA-256 mismatch: {path}")
        if portable:
            missing_tools = [p for p in tool_binaries if not Path(p).is_file()]
            if missing_tools:
                failures.append(
                    "toolchain binaries missing: " + ", ".join(sorted(missing_tools)[:6])
                )

        workflow_source_rows = []
        seen_source_paths = set()
        try:
            with Path(input.workflow_source).open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle, delimiter="\t")
                if reader.fieldnames != ["path", "sha256"]:
                    failures.append(
                        "workflow source manifest must have exactly the tab-delimited columns path and sha256"
                    )
                for row in reader:
                    declared_path = str(row.get("path", "")).strip()
                    expected_hash = str(row.get("sha256", "")).strip().lower()
                    candidate = Path(declared_path)
                    if not candidate.is_absolute():
                        candidate = PROJECT_ROOT / candidate
                    candidate = candidate.resolve()
                    if not declared_path or not _is_relative_to(candidate, PROJECT_ROOT):
                        failures.append(f"invalid workflow source path: {declared_path!r}")
                        continue
                    if candidate in seen_source_paths:
                        failures.append(f"duplicate workflow source path: {candidate}")
                        continue
                    seen_source_paths.add(candidate)
                    if not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
                        failures.append(f"invalid workflow source SHA-256: {candidate}")
                    elif not candidate.is_file():
                        failures.append(f"workflow source file missing: {candidate}")
                    elif _sha256(candidate).lower() != expected_hash:
                        failures.append(f"workflow source SHA-256 mismatch: {candidate}")
                    workflow_source_rows.append(
                        {"path": str(candidate), "sha256": expected_hash}
                    )
                if not workflow_source_rows:
                    failures.append("workflow source manifest contains no source rows")
        except Exception as exc:
            failures.append(
                f"invalid workflow source manifest: {type(exc).__name__}:{exc}"
            )
        missing_workflow_sources = sorted(
            str(path) for path in WORKFLOW_REQUIRED_SOURCE_PATHS - seen_source_paths
        )
        if missing_workflow_sources:
            failures.append(
                "workflow source manifest is incomplete: "
                + ", ".join(missing_workflow_sources)
            )
        declared_workflow_source_rows = int(
            ENVIRONMENT_LOCK["workflow"]["source_manifest"]["rows"]
        )
        if len(workflow_source_rows) != declared_workflow_source_rows:
            failures.append(
                "workflow source manifest row count differs from the environment contract: "
                f"{len(workflow_source_rows)} != {declared_workflow_source_rows}"
            )

        source_contract_rows = []
        runtime_metadata_failures = []
        try:
            runtime_inventory_index = build_runtime_inventory_slices(
                EXECUTION_ROWS,
                input.locked_source_inventory,
                output.source_inventory,
            )
            locked_inventory_contract = config["inputs"]["locked_file_inventory"]
            observed_inventory = runtime_inventory_index["locked_inventory"]
            if (
                observed_inventory["sha256"]
                != locked_inventory_contract["sha256"]
                or observed_inventory["row_count"]
                != int(locked_inventory_contract["row_count"])
            ):
                failures.append("locked source inventory hash/row count differs")
        except Exception as exc:
            runtime_inventory_index = {
                "status": "FAIL",
                "reason": f"{type(exc).__name__}:{exc}",
            }
            failures.append(
                "unable to build execution source-inventory slices: "
                + runtime_inventory_index["reason"]
            )
        for row in EXECUTION_ROWS:
            modality_records = {}
            for modality, kind_key, count_key, bytes_key, bundle_key in (
                ("dwi", "dti_source_kind", "dti_raw_file_count", "dti_raw_total_bytes", "dti_raw_bundle_sha256"),
                ("t1", "t1_source_kind", "t1_raw_file_count", "t1_raw_total_bytes", "t1_raw_bundle_sha256"),
            ):
                paths = list(raw_source_paths(row, modality))
                kind = row[kind_key]
                if kind == "dicom_series":
                    members = sorted(
                        path for path in paths[0].rglob("*") if path.is_file()
                    )
                else:
                    members = paths
                observed_count = len(members)
                observed_bytes = sum(path.stat().st_size for path in members)
                expected_count = int(row[count_key])
                expected_bytes = int(row[bytes_key])
                if observed_count != expected_count:
                    failures.append(
                        f"raw {modality} file-count mismatch for {row['unit']}: "
                        f"{observed_count} != {expected_count}"
                    )
                if observed_bytes != expected_bytes:
                    failures.append(
                        f"raw {modality} byte-count mismatch for {row['unit']}: "
                        f"{observed_bytes} != {expected_bytes}"
                    )
                if not re.fullmatch(r"[0-9a-f]{64}", row[bundle_key]):
                    failures.append(
                        f"invalid locked raw {modality} bundle hash for {row['unit']}"
                    )
                modality_records[modality] = {
                    "source_kind": kind,
                    "source_paths": [str(path) for path in paths],
                    "observed_file_count": observed_count,
                    "expected_file_count": expected_count,
                    "observed_total_bytes": observed_bytes,
                    "expected_total_bytes": expected_bytes,
                    "raw_bundle_sha256": row[bundle_key],
                }
            if row["dti_source_kind"] == "nifti_bundle":
                for path_key, hash_key in (
                    ("dwi_nifti_path", "dwi_nifti_sha256"),
                    ("dwi_bvec_path", "dwi_bvec_sha256"),
                    ("dwi_bval_path", "dwi_bval_sha256"),
                    ("dwi_json_path", "dwi_json_sha256"),
                ):
                    observed_hash = _sha256(row[path_key])
                    if observed_hash != row[hash_key]:
                        failures.append(
                            f"raw NIfTI-bundle member hash mismatch for {row['unit']}: {path_key}"
                        )
            try:
                require_eddy_metadata(row)
            except Exception as exc:
                # This is deliberately not a global preflight failure. The unit
                # remains in the 530-case denominator and normalize_dwi_source
                # emits the fail-closed terminal outcome without a default.
                runtime_metadata_failures.append(
                    {"unit": row["unit"], "reason": f"{type(exc).__name__}:{exc}"}
                )
            source_contract_rows.append(
                {
                    "unit": row["unit"],
                    "processing_authorized": True,
                    "normalization_readiness": row["normalization_readiness"],
                    "modalities": modality_records,
                }
            )

        tool_commands = {
            "mrtrix": [str(MRTRIX_BIN / "mrconvert"), "-version"],
            "fsl": [str(FSL_BIN / "flirt"), "-version"],
            "ants": [str(ANTS_BIN / "antsRegistration"), "--version"],
            "ss3t": [str(SS3T_PYTHON), str(SS3T_SCRIPT), "-version"],
        }
        versions = {}
        for name, command in tool_commands.items():
            commands.append(" ".join(shlex.quote(part) for part in command))
            if not Path(command[0]).is_file():
                failures.append(f"required executable missing: {command[0]}")
                continue
            completed = subprocess.run(command, text=True, capture_output=True, check=False)
            combined = (completed.stdout + "\n" + completed.stderr).strip()
            versions[name] = combined[:2000]
            if completed.returncode != 0:
                failures.append(f"version command failed for {name} with exit {completed.returncode}")

        record = {
            "status": "PASS" if not failures else "FAIL",
            "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "recipe_id": RECIPE_ID,
            "environment_contract": str(ENVIRONMENT_CONTRACT),
            "environment_contract_sha256": _sha256(input.environment),
            "resolved_run_config_sha256": _sha256(input.resolved_config),
            "run_context_sha256": _sha256(input.run_context),
            "attempt_context_sha256": _sha256(input.attempt_context),
            "locked_file_count": len(locked_files),
            "workflow_source_file_count": len(workflow_source_rows),
            "workflow_source_declared_file_count": declared_workflow_source_rows,
            "workflow_required_source_file_count": len(WORKFLOW_REQUIRED_SOURCE_PATHS),
            "workflow_source_files": workflow_source_rows,
            "acquisition_schema_sha256": _sha256(input.acquisition_schema),
            "source_contract_row_count": len(source_contract_rows),
            "source_contract_rows": source_contract_rows,
            "runtime_metadata_failure_count": len(runtime_metadata_failures),
            "runtime_metadata_failures": runtime_metadata_failures,
            "runtime_source_inventory": runtime_inventory_index,
            "commands": commands,
            "versions": versions,
            "failures": failures,
            "no_fallbacks": True,
            "density_route_selection": False,
        }
        Path(log[0]).write_text(
            "\n".join(f"$ {command}" for command in commands)
            + "\n"
            + json.dumps(record, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        if failures:
            raise ValueError("Execution preflight failed: " + "; ".join(failures))
        import shutil

        shim_dir = Path(output.fsl_cpu_path)
        if shim_dir.exists():
            shutil.rmtree(shim_dir)
        shim_dir.mkdir(parents=True)
        for candidate in FSL_BIN.iterdir():
            if candidate.name.startswith("eddy") and candidate.name != "eddy_cpu":
                continue
            if candidate.is_file() or candidate.is_symlink():
                (shim_dir / candidate.name).symlink_to(candidate.resolve())
        shim = shim_dir / "eddy_cpu"
        if not shim.is_symlink() or shim.resolve() != EDDY_CPU or any(path.name.startswith("eddy_cuda") for path in shim_dir.iterdir()):
            raise ValueError("Unable to construct an eddy_cpu-only PATH shim")
        tmp = Path(str(output.report) + ".partial")
        tmp.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        tmp.replace(output.report)
