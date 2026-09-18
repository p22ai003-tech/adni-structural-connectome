"""Versioned manifest/preflight outputs for retry4 pretract recovery1."""


rule freeze_manifest:
    input:
        manifest=str(MANIFEST_PATH),
        execution_manifest=str(EXECUTION_MANIFEST_PATH),
        config=str(NORMATIVE_CONFIG),
        environment=str(ENVIRONMENT_CONTRACT),
        workflow_source=str(WORKFLOW_SOURCE_MANIFEST),
        acquisition_schema=str(ACQUISITION_SCHEMA),
        resolved_config=str(RESOLVED_RUN_CONFIG),
        run_context=str(RUN_CONTEXT_PATH),
        attempt_context=str(ATTEMPT_CONTEXT_PATH),
    output:
        manifest=run_path("contract", "approved_acquisition_manifest_recovery1.csv"),
        validation=run_path("contract", "manifest_validation_recovery1.json"),
    log:
        run_path("logs", "00_freeze_manifest_recovery1.log"),
    run:
        import datetime
        import shutil

        Path(output.manifest).parent.mkdir(parents=True, exist_ok=True)
        Path(log[0]).parent.mkdir(parents=True, exist_ok=True)
        record = {
            "schema_version": "2.0.0",
            "record_type": "retry4_pretract_manifest_freeze",
            "status": "PASS",
            "generated_utc": datetime.datetime.now(
                datetime.timezone.utc
            ).isoformat(),
            "recipe_id": RECIPE_ID,
            "run_root": str(RUN_ROOT),
            "parent_manifest": {
                "path": str(MANIFEST_PATH),
                "sha256": _sha256(input.manifest),
                "size_bytes": Path(input.manifest).stat().st_size,
            },
            "execution_manifest": {
                "path": str(EXECUTION_MANIFEST_PATH),
                "sha256": _sha256(input.execution_manifest),
                "size_bytes": Path(input.execution_manifest).stat().st_size,
            },
            "execution_units": list(EXECUTION_UNITS),
            "execution_unit_count": len(EXECUTION_UNITS),
            "diagnosis_labels_used": False,
            "pretract_extension_binding": PRETRACT_EXTENSION_BINDING,
            "resolved_config_sha256": _sha256(input.resolved_config),
            "run_context_sha256": _sha256(input.run_context),
            "attempt_context_sha256": _sha256(input.attempt_context),
            "production_overwrite": False,
        }
        if (
            record["parent_manifest"]["sha256"]
            != config["inputs"]["approved_pair_manifest"]["sha256"]
            or EXECUTION_BINDING.get("execution_subset_manifest")
            != record["execution_manifest"]
            or len(EXECUTION_UNITS) != 15
        ):
            raise ValueError("retry4 pre-tractography manifest identity differs")
        manifest_tmp = Path(str(output.manifest) + ".partial")
        validation_tmp = Path(str(output.validation) + ".partial")
        shutil.copyfile(input.manifest, manifest_tmp)
        manifest_tmp.replace(output.manifest)
        validation_tmp.write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        validation_tmp.replace(output.validation)
        Path(log[0]).write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


rule execution_preflight:
    input:
        manifest=rules.freeze_manifest.output.manifest,
        validation=rules.freeze_manifest.output.validation,
        config=str(NORMATIVE_CONFIG),
        environment=str(ENVIRONMENT_CONTRACT),
        workflow_source=str(WORKFLOW_SOURCE_MANIFEST),
        acquisition_schema=str(ACQUISITION_SCHEMA),
        resolved_config=str(RESOLVED_RUN_CONFIG),
        run_context=str(RUN_CONTEXT_PATH),
        attempt_context=str(ATTEMPT_CONTEXT_PATH),
        atlas=str(ATLAS_IMAGE),
        node_table=str(ATLAS_NODE_TABLE),
        node_map=str(ATLAS_NODE_MAP),
        mni=str(MNI_TEMPLATE),
        frozen_response=str(FROZEN_RESPONSE_MANIFEST),
        pooled_wm=str(FROZEN_RESPONSE_PATHS["wm"]),
        pooled_gm=str(FROZEN_RESPONSE_PATHS["gm"]),
        pooled_csf=str(FROZEN_RESPONSE_PATHS["csf"]),
    output:
        report=run_path("contract", "execution_preflight_recovery1.json"),
        fsl_cpu_path=directory(run_path("contract", "fsl_cpu_path_recovery1")),
        source_inventory=directory(
            run_path("contract", "source_runtime_inventory_recovery1")
        ),
    log:
        run_path("logs", "00_execution_preflight_recovery1.log"),
    run:
        import csv
        import datetime
        import shutil

        Path(output.report).parent.mkdir(parents=True, exist_ok=True)
        Path(log[0]).parent.mkdir(parents=True, exist_ok=True)
        failures = []
        commands = []
        try:
            normative = yaml.safe_load(
                Path(input.config).read_text(encoding="utf-8")
            )
            resolved = yaml.safe_load(
                Path(input.resolved_config).read_text(encoding="utf-8")
            )
            run_context = json.loads(
                Path(input.run_context).read_text(encoding="utf-8")
            )
            attempt_context = json.loads(
                Path(input.attempt_context).read_text(encoding="utf-8")
            )
            validate_resolved_runtime_config(normative, resolved, run_context)
            if (
                run_context.get("schema_version") != "1.0.0"
                or run_context.get("status") != "LOCKED"
                or run_context.get("launcher_mode")
                != "pre-tractography-canary"
                or run_context.get("execution_binding") != EXECUTION_BINDING
                or run_context.get("pretract_extension_binding")
                != PRETRACT_EXTENSION_BINDING
                or run_context.get("response_calibration_binding")
                != config.get("response_calibration_binding")
            ):
                failures.append("locked pre-tractography run context differs")
            if (
                attempt_context.get("schema_version") != "1.0.0"
                or attempt_context.get("status") != "STARTED"
                or attempt_context.get("launcher_mode")
                != "pre-tractography-canary"
                or attempt_context.get("run_id") != run_context.get("run_id")
                or attempt_context.get("execution_binding") != EXECUTION_BINDING
                or attempt_context.get("pretract_extension_binding")
                != PRETRACT_EXTENSION_BINDING
                or attempt_context.get("response_calibration_binding")
                != config.get("response_calibration_binding")
            ):
                failures.append("started pre-tractography attempt context differs")
            resource = attempt_context.get("pretract_resource_preflight")
            if (
                not isinstance(resource, dict)
                or resource.get("status") != "PASS"
                or resource.get("maximum_cpu_cores") != 32
                or resource.get("wall_clock_stop_seconds") != 86400
                or resource.get("storage_stop_bytes") != 150_000_000_000
                or resource.get("wall_clock_remaining_seconds", 0) <= 0
                or resource.get("storage_remaining_bytes", 0) <= 0
            ):
                failures.append("attempt lacks valid pre-tractography resource cap")
            if PRETRACT_EXTENSION_BINDING.get("status") != "LOCKED":
                failures.append("dry-run-only binding cannot execute imaging")
        except Exception as exc:
            failures.append(
                "runtime binding validation failed: "
                f"{type(exc).__name__}:{exc}"
            )

        response_binding = config.get("response_calibration_binding")
        try:
            if (
                not isinstance(response_binding, dict)
                or response_binding.get("binding_type")
                != "response_calibration_phase_b_binding"
                or response_binding.get("valid_units") != list(EXECUTION_UNITS)
                or response_binding.get("diagnosis_labels_used") is not False
            ):
                raise ValueError("approved response binding semantics differ")
            validate_frozen_response_calibration(
                input.frozen_response,
                expected_manifest_sha256=RESPONSE_CALIBRATION[
                    "frozen_manifest"
                ]["sha256"],
                minimum_valid_subjects=int(
                    RESPONSE_CALIBRATION["minimum_valid_subjects"]
                ),
                expected_responses=RESPONSE_CALIBRATION["pooled_responses"],
                expected_technical_diversity=response_binding[
                    "valid_pool_technical_diversity"
                ],
            )
        except Exception as exc:
            failures.append(
                "frozen response validation failed: "
                f"{type(exc).__name__}:{exc}"
            )

        static_hashes = {
            str(ATLAS_IMAGE): str(
                config["atlas"]["contiguous_source_image_sha256"]
            ),
            str(ATLAS_NODE_TABLE): str(config["atlas"]["node_table_sha256"]),
            str(ATLAS_NODE_MAP): str(config["atlas"]["node_map_sha256"]),
            str(MNI_TEMPLATE): str(
                config["registration"]["mni_to_t1"]["template_sha256"]
            ),
            str(ACQUISITION_SCHEMA): str(
                config["inputs"]["acquisition_schema"]["sha256"]
            ),
        }
        for path, expected_hash in static_hashes.items():
            candidate = Path(path)
            if not candidate.is_file() or _sha256(candidate) != expected_hash:
                failures.append(f"static input identity differs: {path}")

        workflow_rows = []
        try:
            with Path(input.workflow_source).open(
                newline="", encoding="utf-8"
            ) as handle:
                reader = csv.DictReader(handle, delimiter="\t")
                if reader.fieldnames != ["path", "sha256"]:
                    raise ValueError("workflow source columns differ")
                for row in reader:
                    candidate = Path(str(row["path"]))
                    if not candidate.is_absolute():
                        candidate = PROJECT_ROOT / candidate
                    candidate = candidate.resolve()
                    if (
                        not candidate.is_file()
                        or _sha256(candidate) != row["sha256"]
                    ):
                        raise ValueError(
                            f"workflow source identity differs: {candidate}"
                        )
                    workflow_rows.append(str(candidate))
            if len(workflow_rows) != int(
                ENVIRONMENT_LOCK["workflow"]["source_manifest"]["rows"]
            ):
                raise ValueError("workflow source row count differs")
        except Exception as exc:
            failures.append(
                f"workflow source validation failed: {type(exc).__name__}:{exc}"
            )

        prerequisite_relatives = (
            "00_inputs/t1_native.nii",
            "00_inputs/input_contract.json",
            "01_dwi/dwi_preproc.mif",
            "01_dwi/dwi_preproc_biascorr.mif",
            "01_dwi/dwi_brain_mask.mif",
            "01_dwi/gradient_contract.json",
            "05_model/dwi_fod_shells.mif",
            "05_model/fod_shell_selection.json",
            "05_model/response_calibration_outcome.json",
        )
        prerequisite_count = 0
        for unit in EXECUTION_UNITS:
            for relative in prerequisite_relatives:
                candidate = RUN_ROOT / "subjects" / unit / relative
                if (
                    not candidate.is_file()
                    or candidate.is_symlink()
                    or candidate.stat().st_size < 1
                ):
                    failures.append(f"frozen prerequisite missing: {unit}/{relative}")
                else:
                    prerequisite_count += 1

        forbidden_paths = []
        for pattern in (
            "**/*.tck",
            "**/sift2_weights.txt",
            "subjects/*/07_connectome/matrices/*.csv",
        ):
            forbidden_paths.extend(str(path) for path in RUN_ROOT.glob(pattern))
        if forbidden_paths:
            failures.append("forbidden downstream artifacts already exist")

        source_inventory = Path(output.source_inventory)
        if source_inventory.exists():
            shutil.rmtree(source_inventory)
        source_inventory.mkdir(parents=True)
        (source_inventory / "pretract_reuse_index.json").write_text(
            json.dumps(
                {
                    "status": "PASS",
                    "unit_count": len(EXECUTION_UNITS),
                    "prerequisite_file_count": prerequisite_count,
                    "normalization_or_eddy_scheduled": False,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

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
        if (
            not shim.is_symlink()
            or shim.resolve() != EDDY_CPU
            or any(path.name.startswith("eddy_cuda") for path in shim_dir.iterdir())
        ):
            failures.append("unable to construct CPU-only FSL compatibility shim")

        record = {
            "schema_version": "2.0.0",
            "record_type": "retry4_pretract_execution_preflight",
            "status": "PASS" if not failures else "FAIL",
            "generated_utc": datetime.datetime.now(
                datetime.timezone.utc
            ).isoformat(),
            "recipe_id": RECIPE_ID,
            "run_root": str(RUN_ROOT),
            "execution_units": list(EXECUTION_UNITS),
            "execution_unit_count": len(EXECUTION_UNITS),
            "pretract_extension_binding": PRETRACT_EXTENSION_BINDING,
            "response_calibration_binding": response_binding,
            "workflow_source_file_count": len(workflow_rows),
            "frozen_prerequisite_file_count": prerequisite_count,
            "allowed_terminal_stage": "automated_qc_and_blinded_review_bundle",
            "normalization_or_eddy_scheduled": False,
            "response_reestimation_scheduled": False,
            "tractography_scheduled": False,
            "matrix_generation_scheduled": False,
            "diagnosis_labels_used": False,
            "commands": commands,
            "versions": {},
            "failures": failures,
        }
        Path(log[0]).write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if failures:
            raise ValueError(
                "retry4 pre-tractography preflight failed: "
                + "; ".join(failures)
            )
        temporary = Path(str(output.report) + ".partial")
        temporary.write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output.report)
