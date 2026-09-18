"""Deterministic raw-source normalization and staged-input hash contract."""


rule normalize_dwi_source:
    input:
        gate=ancient(rules.execution_preflight.output.report),
        attempt_context=ancient(str(ATTEMPT_CONTEXT_PATH)),
        runtime_inventory=ancient(rules.execution_preflight.output.source_inventory),
        raw=raw_modality_input("dwi"),
    output:
        dwi=subject_path("{unit}", "00_inputs", "dwi_raw.mif"),
        bvec=subject_path("{unit}", "00_inputs", "dwi.bvec"),
        bval=subject_path("{unit}", "00_inputs", "dwi.bval"),
        metadata=subject_path("{unit}", "00_inputs", "dwi_source_metadata.json"),
        record=subject_path("{unit}", "00_inputs", "dwi_normalization.json"),
    log:
        subject_log("{unit}", "00_normalize_dwi.log"),
    threads: 1
    run:
        import datetime
        import shlex
        import subprocess

        row = ROW_BY_UNIT[str(wildcards.unit)]
        attempt_context = json.loads(
            Path(input.attempt_context).read_text(encoding="utf-8")
        )
        attempt_id = str(attempt_context.get("attempt_id", ""))
        run_id = str(attempt_context.get("run_id", ""))
        failure_path = (
            Path(output.dwi).parent
            / "failures"
            / attempt_id
            / "dwi_normalization_failure.json"
        )

        def _fail(exc):
            write_normalization_failure_marker(
                failure_path,
                unit=str(wildcards.unit),
                normalizer="dwi",
                primary_failure_reason=f"{type(exc).__name__}: {exc}",
                source_identity_hashes={
                    "dti_raw_bundle_sha256": row["dti_raw_bundle_sha256"],
                    "pair_content_bundle_sha256": row["pair_content_bundle_sha256"],
                },
                generated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                run_id=run_id,
                attempt_id=attempt_id,
                attempt_context_path=input.attempt_context,
            )
            raise exc

        try:
            raw_paths = [Path(path) for path in input.raw]
            source_inventory_record = (
                Path(input.runtime_inventory) / f"{wildcards.unit}.dwi.json"
            )
            source_identity_before = verify_runtime_source_inventory(
                row, "dwi", source_inventory_record
            )
            output_paths = {
                "dwi_raw_mif": Path(output.dwi),
                "dwi_bvec": Path(output.bvec),
                "dwi_bval": Path(output.bval),
                "dwi_source_metadata": Path(output.metadata),
            }
            for path in output_paths.values():
                path.parent.mkdir(parents=True, exist_ok=True)
            Path(log[0]).parent.mkdir(parents=True, exist_ok=True)
            partials = {
                "dwi_raw_mif": Path(output.dwi).with_name("dwi_raw.partial.mif"),
                "dwi_bvec": Path(output.bvec).with_name("dwi.partial.bvec"),
                "dwi_bval": Path(output.bval).with_name("dwi.partial.bval"),
                "dwi_source_metadata": Path(output.metadata).with_name("dwi_source_metadata.partial.json"),
            }
            for path in partials.values():
                if path.exists():
                    path.unlink()

            command = [str(MRTRIX_BIN / "mrconvert")]
            if row["dti_source_kind"] == "dicom_series":
                if len(raw_paths) != 1 or not raw_paths[0].is_dir():
                    raise ValueError(f"{wildcards.unit} requires one DICOM source directory")
                command.extend([str(raw_paths[0]), str(partials["dwi_raw_mif"])])
            elif row["dti_source_kind"] == "nifti_bundle":
                if len(raw_paths) != 4 or any(not path.is_file() for path in raw_paths):
                    raise ValueError(f"{wildcards.unit} requires NIfTI, bvec, bval and JSON source files")
                for key, path in (
                    ("dwi_nifti_sha256", raw_paths[0]),
                    ("dwi_bvec_sha256", raw_paths[1]),
                    ("dwi_bval_sha256", raw_paths[2]),
                    ("dwi_json_sha256", raw_paths[3]),
                ):
                    observed = _sha256(path)
                    if observed != row[key]:
                        raise ValueError(f"{wildcards.unit} raw NIfTI bundle hash mismatch: {key}")
                command.extend(
                    [
                        str(raw_paths[0]),
                        str(partials["dwi_raw_mif"]),
                        "-fslgrad",
                        str(raw_paths[1]),
                        str(raw_paths[2]),
                        "-json_import",
                        str(raw_paths[3]),
                    ]
                )
            else:
                raise ValueError(f"Unsupported DWI source kind: {row['dti_source_kind']!r}")
            command.extend(
                [
                    "-json_export",
                    str(partials["dwi_source_metadata"]),
                    "-export_grad_fsl",
                    str(partials["dwi_bvec"]),
                    str(partials["dwi_bval"]),
                    "-strides",
                    "0,0,0,1",
                ]
            )
        except Exception as exc:
            _fail(exc)
        environment = os.environ.copy()
        environment["PATH"] = TOOL_PATH
        try:
            completed = subprocess.run(
                command,
                text=True,
                capture_output=True,
                check=False,
                env=environment,
            )
        except Exception as exc:
            _fail(exc)
        Path(log[0]).write_text(
            "$ " + shlex.join(command) + "\n" + completed.stdout + completed.stderr,
            encoding="utf-8",
        )
        if completed.returncode != 0:
            _fail(RuntimeError(
                f"DWI normalization failed for {wildcards.unit} with exit {completed.returncode}"
            ))
        try:
            metadata = json.loads(partials["dwi_source_metadata"].read_text(encoding="utf-8"))
            observed_direction = source_phase_encoding_token(
                metadata.get("PhaseEncodingDirection")
            )
            try:
                observed_readout = float(metadata["TotalReadoutTime"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"{wildcards.unit} normalized metadata lacks TotalReadoutTime"
                ) from exc
            normalized_row = dict(row)
            normalized_row["phase_encoding_direction"] = observed_direction
            normalized_row["total_readout_time"] = format(observed_readout, ".12g")
            direction, readout = require_eddy_metadata(normalized_row)
            declared_direction = row["phase_encoding_direction"].strip()
            declared_readout = row["total_readout_time"].strip()
            if declared_direction and declared_direction != direction:
                raise ValueError(
                    f"{wildcards.unit} source-declared PhaseEncodingDirection differs: "
                    f"{declared_direction!r} != {direction!r}"
                )
            if declared_readout and abs(float(declared_readout) - readout) > 1e-9:
                raise ValueError(
                    f"{wildcards.unit} source-declared TotalReadoutTime differs: "
                    f"{declared_readout} != {readout}"
                )
            source_identity_after = verify_runtime_source_inventory(
                row, "dwi", source_inventory_record
            )
        except Exception as exc:
            _fail(exc)
        for name, final in output_paths.items():
            partials[name].replace(final)
        staged = staged_hashes(output_paths)
        record = {
            "schema_version": "2.0.0",
            "status": "PASS",
            "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "unit": str(wildcards.unit),
            "recipe_id": RECIPE_ID,
            "source_kind": row["dti_source_kind"],
            "source_id": row["dti_source_id"],
            "raw_source_bundle": {
                "path": row["dti_source_path"],
                "sha256": source_identity_after["observed_bundle_sha256"],
                "file_count": int(row["dti_raw_file_count"]),
                "total_bytes": int(row["dti_raw_total_bytes"]),
                "hash_role": "locked_raw_bundle_identity",
            },
            "staged_files": staged,
            "staged_hash_role": "exact_post_normalization_file_identity",
            "raw_source_verification_before_conversion": source_identity_before,
            "raw_source_verification_after_conversion": source_identity_after,
            "phase_encoding_direction": direction,
            "total_readout_time": readout,
            "command": command,
        }
        record_path = Path(output.record)
        temporary_record = record_path.with_name(record_path.name + ".partial")
        temporary_record.write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary_record.replace(record_path)


rule normalize_t1_source:
    input:
        gate=ancient(rules.execution_preflight.output.report),
        attempt_context=ancient(str(ATTEMPT_CONTEXT_PATH)),
        runtime_inventory=ancient(rules.execution_preflight.output.source_inventory),
        raw=raw_modality_input("t1"),
    output:
        t1=subject_path("{unit}", "00_inputs", "t1_native.nii"),
        record=subject_path("{unit}", "00_inputs", "t1_normalization.json"),
    log:
        subject_log("{unit}", "00_normalize_t1.log"),
    threads: 1
    run:
        import datetime
        import shlex
        import subprocess

        row = ROW_BY_UNIT[str(wildcards.unit)]
        attempt_context = json.loads(
            Path(input.attempt_context).read_text(encoding="utf-8")
        )
        attempt_id = str(attempt_context.get("attempt_id", ""))
        run_id = str(attempt_context.get("run_id", ""))
        failure_path = (
            Path(output.t1).parent
            / "failures"
            / attempt_id
            / "t1_normalization_failure.json"
        )

        def _fail(exc):
            write_normalization_failure_marker(
                failure_path,
                unit=str(wildcards.unit),
                normalizer="t1",
                primary_failure_reason=f"{type(exc).__name__}: {exc}",
                source_identity_hashes={
                    "t1_raw_bundle_sha256": row["t1_raw_bundle_sha256"],
                    "pair_content_bundle_sha256": row["pair_content_bundle_sha256"],
                },
                generated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                run_id=run_id,
                attempt_id=attempt_id,
                attempt_context_path=input.attempt_context,
            )
            raise exc

        try:
            raw_paths = [Path(path) for path in input.raw]
            source_inventory_record = (
                Path(input.runtime_inventory) / f"{wildcards.unit}.t1.json"
            )
            source_identity_before = verify_runtime_source_inventory(
                row, "t1", source_inventory_record
            )
            if len(raw_paths) != 1:
                raise ValueError(f"{wildcards.unit} requires exactly one T1 source path")
            raw_path = raw_paths[0]
            if row["t1_source_kind"] == "dicom_series":
                if not raw_path.is_dir() or not row["t1_dicom_series_uid"]:
                    raise ValueError(f"{wildcards.unit} DICOM T1 requires a directory and SeriesInstanceUID")
            elif row["t1_source_kind"] == "nifti_single":
                if not raw_path.is_file() or row["t1_dicom_series_uid"]:
                    raise ValueError(f"{wildcards.unit} single-NIfTI T1 must have a file and null DICOM UID")
            else:
                raise ValueError(f"Unsupported T1 source kind: {row['t1_source_kind']!r}")
            final = Path(output.t1)
            final.parent.mkdir(parents=True, exist_ok=True)
            Path(log[0]).parent.mkdir(parents=True, exist_ok=True)
            partial = final.with_name("t1_native.partial.nii")
            if partial.exists():
                partial.unlink()
            command = [
                str(MRTRIX_BIN / "mrconvert"),
                str(raw_path),
                str(partial),
                "-strides",
                "+1,+2,+3",
            ]
        except Exception as exc:
            _fail(exc)
        environment = os.environ.copy()
        environment["PATH"] = TOOL_PATH
        try:
            completed = subprocess.run(
                command,
                text=True,
                capture_output=True,
                check=False,
                env=environment,
            )
        except Exception as exc:
            _fail(exc)
        Path(log[0]).write_text(
            "$ " + shlex.join(command) + "\n" + completed.stdout + completed.stderr,
            encoding="utf-8",
        )
        if completed.returncode != 0:
            _fail(RuntimeError(
                f"T1 normalization failed for {wildcards.unit} with exit {completed.returncode}"
            ))
        try:
            source_identity_after = verify_runtime_source_inventory(
                row, "t1", source_inventory_record
            )
            partial.replace(final)
            staged = staged_hashes({"t1_native_nifti": final})
        except Exception as exc:
            _fail(exc)
        record = {
            "schema_version": "2.0.0",
            "status": "PASS",
            "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "unit": str(wildcards.unit),
            "recipe_id": RECIPE_ID,
            "source_kind": row["t1_source_kind"],
            "source_id": row["t1_source_id"],
            "dicom_series_uid": row["t1_dicom_series_uid"] or None,
            "raw_source_bundle": {
                "path": row["t1_source_path"],
                "sha256": source_identity_after["observed_bundle_sha256"],
                "file_count": int(row["t1_raw_file_count"]),
                "total_bytes": int(row["t1_raw_total_bytes"]),
                "hash_role": "locked_raw_bundle_identity",
            },
            "staged_files": staged,
            "staged_hash_role": "exact_post_normalization_file_identity",
            "raw_source_verification_before_conversion": source_identity_before,
            "raw_source_verification_after_conversion": source_identity_after,
            "command": command,
        }
        record_path = Path(output.record)
        temporary_record = record_path.with_name(record_path.name + ".partial")
        temporary_record.write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary_record.replace(record_path)


rule input_contract_gate:
    input:
        dwi=rules.normalize_dwi_source.output.dwi,
        bvec=rules.normalize_dwi_source.output.bvec,
        bval=rules.normalize_dwi_source.output.bval,
        metadata=rules.normalize_dwi_source.output.metadata,
        dwi_record=rules.normalize_dwi_source.output.record,
        t1=rules.normalize_t1_source.output.t1,
        t1_record=rules.normalize_t1_source.output.record,
    output:
        subject_path("{unit}", "00_inputs", "input_contract.json"),
    log:
        subject_log("{unit}", "00_input_contract.log"),
    run:
        import datetime

        row = ROW_BY_UNIT[str(wildcards.unit)]
        dwi_record = json.loads(Path(input.dwi_record).read_text(encoding="utf-8"))
        t1_record = json.loads(Path(input.t1_record).read_text(encoding="utf-8"))
        if dwi_record.get("status") != "PASS" or t1_record.get("status") != "PASS":
            raise ValueError(f"{wildcards.unit} normalization records are not PASS")
        staged = staged_hashes(
            {
                "dwi_raw_mif": input.dwi,
                "dwi_bvec": input.bvec,
                "dwi_bval": input.bval,
                "dwi_source_metadata": input.metadata,
                "t1_native_nifti": input.t1,
            }
        )
        expected = {
            **dwi_record["staged_files"],
            **t1_record["staged_files"],
        }
        if staged != expected:
            raise ValueError(f"{wildcards.unit} staged input changed after normalization")
        record = {
            "schema_version": "2.0.0",
            "status": "PASS",
            "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "unit": str(wildcards.unit),
            "recipe_id": RECIPE_ID,
            "source_identity": {
                "subject_id": row["subject_id"],
                "dti_image_id": row["dti_image_id"],
                "dti_source_id": row["dti_source_id"],
                "dti_dicom_series_uid": row["dti_dicom_series_uid"] or None,
                "t1_image_id": row["t1_image_id"],
                "t1_source_id": row["t1_source_id"],
                "t1_dicom_series_uid": row["t1_dicom_series_uid"] or None,
            },
            "source_identity_hashes": {
                "dti_raw_bundle_sha256": row["dti_raw_bundle_sha256"],
                "t1_raw_bundle_sha256": row["t1_raw_bundle_sha256"],
                "pair_content_bundle_sha256": row["pair_content_bundle_sha256"],
            },
            "staged_input_hashes": {
                name: details["sha256"] for name, details in staged.items()
            },
            "staged_inputs": staged,
            "normalization_records": {
                "dwi": {"path": str(Path(input.dwi_record).resolve()), "sha256": _sha256(input.dwi_record)},
                "t1": {"path": str(Path(input.t1_record).resolve()), "sha256": _sha256(input.t1_record)},
            },
            "processing_authorized": True,
            "diagnosis_used_for_processing_route": False,
            "pair_gap_used_for_processing_rejection": False,
            "t1_source_kind_used_for_processing_rejection": False,
        }
        Path(log[0]).parent.mkdir(parents=True, exist_ok=True)
        Path(log[0]).write_text(
            "internal: verify raw/staged input identities remain separate\n"
            + json.dumps(record, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        temporary = Path(str(output[0]) + ".partial")
        temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(output[0])
