from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from jsonschema import Draft202012Validator, FormatChecker


LEGACY_PROVENANCE_SCHEMA_VERSION = "1.0.0"
LEGACY_LEDGER_SCHEMA_VERSION = "1.0.0"
PROVENANCE_SCHEMA_VERSION = "2.0.0"
LEDGER_SCHEMA_VERSION = "2.0.0"
SHA256_HEX_LENGTH = 64

TERMINAL_STATUSES = ("PASS", "PARTIAL", "FAIL")
OUTCOME_STATES = ("VALID", "NA")
CORE_OUTCOMES = (
    "processing",
    "tensor",
    "tractography",
    "matrices",
    "biological_inference",
)
MATRIX_OUTCOMES = (
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
REQUIRED_OUTCOMES = CORE_OUTCOMES + MATRIX_OUTCOMES

_COUNT_SUPPORTED_MEAN_OUTCOMES = (
    "len_mean",
    "invlen_mean",
    "fa_mean",
    "md_mean",
    "rd_mean",
    "ad_mean",
)
_TENSOR_OUTCOMES = ("md_mean", "rd_mean", "ad_mean")
_TRACTOGRAPHY_FAILURE_PREFIXES = (
    "tractogram_",
    "sift2_",
    "assignment_",
    "assignments:",
    "requested_actual_tractography_",
)
_REPORTING_ONLY_SUPPORT_PREFIXES = (
    "count:unexplained_zero_nodes",
    "endpoint_assignment_fraction:",
    "unique_assigned_nodes:",
    "top5_endpoint_fraction:",
    "density:",
    "density_",
)


def matrix_qc_outcome_invalidity(
    matrix_qc: Mapping[str, Any],
    *,
    required_matrices: Sequence[str] = MATRIX_OUTCOMES,
) -> dict[str, tuple[str, ...]]:
    """Map strict bundle-QC evidence to fail-closed per-matrix invalidity.

    A bundle failure does not automatically erase an independently valid
    matrix.  Cross-matrix checks are deliberately propagated to every metric
    that could be responsible, however, so an ambiguous disagreement can
    never release one side as valid.  Unknown or malformed QC evidence
    invalidates the complete matrix set.
    """

    required = tuple(dict.fromkeys(str(name) for name in required_matrices))
    invalid: dict[str, list[str]] = {name: [] for name in required}

    def add(names: Iterable[str], reason: str) -> None:
        for name in names:
            if name in invalid and reason not in invalid[name]:
                invalid[name].append(reason)

    def count_dependencies() -> set[str]:
        return {"count", "count_invnodevol", *_COUNT_SUPPORTED_MEAN_OUTCOMES}

    if not isinstance(matrix_qc, Mapping):
        add(required, "matrix_qc_record_not_a_mapping")
        return {name: tuple(reasons) for name, reasons in invalid.items()}

    expected_nodes = matrix_qc.get("expected_nodes")
    if (
        not isinstance(expected_nodes, int)
        or isinstance(expected_nodes, bool)
        or expected_nodes < 1
    ):
        add(required, "matrix_qc_expected_nodes_invalid")

    summaries = matrix_qc.get("matrix_summaries")
    if not isinstance(summaries, Mapping):
        add(required, "matrix_qc_summaries_missing_or_malformed")
        summaries = {}
    for name in required:
        summary_targets = count_dependencies() if name == "count" else {name}
        summary = summaries.get(name)
        if not isinstance(summary, Mapping):
            add(summary_targets, f"{name}:summary_missing_or_malformed")
            continue
        shape = summary.get("shape")
        if (
            not isinstance(shape, (list, tuple))
            or len(shape) != 2
            or expected_nodes is None
            or tuple(shape) != (expected_nodes, expected_nodes)
        ):
            add(summary_targets, f"{name}:summary_shape_not_exact")
        for field in ("finite", "symmetric", "zero_diagonal", "nonnegative"):
            if summary.get(field) is not True:
                add(summary_targets, f"{name}:summary_{field}_not_pass")

    matrix_names = matrix_qc.get("matrix_names")
    if not isinstance(matrix_names, (list, tuple)) or any(
        not isinstance(name, str) for name in matrix_names
    ):
        add(required, "matrix_qc_matrix_names_missing_or_malformed")
    else:
        named = set(matrix_names)
        for name in set(required) - named:
            targets = count_dependencies() if name == "count" else {name}
            add(targets, f"{name}:absent_from_matrix_names")
        if named - set(required):
            add(required, "matrix_qc_matrix_names_contain_unexpected_values")

    failures = matrix_qc.get("failures")
    if not isinstance(failures, (list, tuple)) or any(
        not isinstance(reason, str) or not reason.strip() for reason in failures
    ):
        add(required, "matrix_qc_failures_missing_or_malformed")
        failures = ()

    for raw_reason in failures:
        reason = raw_reason.strip()
        implicated: set[str] = set()
        if reason.startswith("missing_matrices:"):
            missing = {
                value.strip()
                for value in reason.split(":", 1)[1].split(",")
                if value.strip()
            }
            implicated.update(missing)
            if "count" in missing:
                implicated.update(count_dependencies())
        elif reason.startswith("unexpected_matrices:"):
            implicated.update(required)
        elif reason.startswith("count_and_fd_sum:"):
            implicated.update(("count", "fd_sum"))
        elif reason.startswith("node_volumes_mm3:"):
            implicated.add("count_invnodevol")
        elif reason.startswith("count_invnodevol:formula_mismatch"):
            implicated.update(("count", "count_invnodevol"))
        elif reason.startswith("tensor_order:") or reason.startswith(
            "tensor_identity:"
        ):
            implicated.update(_TENSOR_OUTCOMES)
        elif reason.startswith(_REPORTING_ONLY_SUPPORT_PREFIXES):
            # These are retained as quantitative flags for the blinded H04
            # threshold/SAP decision; they are not mathematical invalidity.
            continue
        elif reason.startswith(_TRACTOGRAPHY_FAILURE_PREFIXES):
            # Every matrix is downstream of the tractogram/weights/assignments.
            implicated.update(required)
        else:
            direct_name = next(
                (name for name in required if reason.startswith(f"{name}:")),
                None,
            )
            if direct_name == "count":
                if reason.startswith("count:fractional"):
                    implicated.update(("count", "count_invnodevol"))
                else:
                    # A structurally invalid count matrix cannot establish the
                    # support used by edge means or its volume-normalised form.
                    implicated.update(count_dependencies())
            elif direct_name in _COUNT_SUPPORTED_MEAN_OUTCOMES and (
                reason.endswith(":nonzero_outside_count_support")
                or reason.endswith(":zero_on_connected_edge")
            ):
                # The disagreement is ambiguous: either this mean is wrong or
                # the shared count support is wrong.  In the latter case no
                # edge-mean support mask is established, so all means are
                # implicated rather than only the one that exposed the defect.
                implicated.update(("count", *_COUNT_SUPPORTED_MEAN_OUTCOMES))
            elif direct_name is not None:
                implicated.add(direct_name)
            else:
                # Novel QC checks must be classified explicitly before any
                # matrix can be released from a failed bundle.
                implicated.update(required)
                reason = f"unmapped_matrix_qc_failure:{reason}"
        add(implicated, reason)

    status = matrix_qc.get("status")
    pass_strict = matrix_qc.get("pass_strict")
    if status == "PASS":
        if pass_strict is not True or failures:
            add(required, "matrix_qc_pass_contract_inconsistent")
    elif status == "FAIL":
        if pass_strict is not False:
            add(required, "matrix_qc_fail_contract_inconsistent")
        if not failures:
            add(required, "matrix_qc_failed_without_reason")
    else:
        add(required, "matrix_qc_status_invalid")

    return {name: tuple(reasons) for name, reasons in invalid.items()}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: str | Path) -> dict[str, Any]:
    candidate = Path(path).resolve()
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    return {
        "path": str(candidate),
        "sha256": sha256_file(candidate),
        "size_bytes": candidate.stat().st_size,
    }


def canonical_json_bytes(record: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(record, indent=2, sort_keys=True, separators=(",", ": ")) + "\n"
    ).encode("utf-8")


def write_immutable_bytes(path: str | Path, payload: bytes) -> Path:
    """Create one immutable evidence file without any overwrite window."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"immutable evidence already exists: {destination}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".partial", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, destination)
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def write_immutable_json(path: str | Path, record: Mapping[str, Any]) -> Path:
    return write_immutable_bytes(path, canonical_json_bytes(record))


def write_immutable_jsonl(path: str | Path, records: Iterable[Mapping[str, Any]]) -> Path:
    payload = b"".join(
        json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
        for record in records
    )
    if not payload:
        raise ValueError("immutable JSONL ledger cannot be empty")
    return write_immutable_bytes(path, payload)


def validate_schema(record: Mapping[str, Any], schema_path: str | Path) -> list[str]:
    schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    errors = []
    for error in sorted(validator.iter_errors(record), key=lambda item: list(item.path)):
        location = ".".join(str(part) for part in error.absolute_path) or "$"
        errors.append(f"{location}: {error.message}")
    return errors


def verify_file_record(record: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    path = Path(str(record.get("path", "")))
    expected_hash = str(record.get("sha256", "")).lower()
    expected_size = record.get("size_bytes")
    if not path.is_file():
        return [f"missing file: {path}"]
    if len(expected_hash) != SHA256_HEX_LENGTH or any(
        char not in "0123456789abcdef" for char in expected_hash
    ):
        errors.append(f"invalid SHA-256: {path}")
    elif sha256_file(path) != expected_hash:
        errors.append(f"SHA-256 mismatch: {path}")
    if not isinstance(expected_size, int) or expected_size < 0:
        errors.append(f"invalid size_bytes: {path}")
    elif path.stat().st_size != expected_size:
        errors.append(f"size mismatch: {path}")
    return errors


def verify_provenance_artifacts(record: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for section in ("source_files", "contract_files", "intermediate_files", "outputs"):
        artifacts = record.get(section, {})
        if not isinstance(artifacts, Mapping):
            errors.append(f"{section} is not an artifact mapping")
            continue
        for name, artifact in artifacts.items():
            if not isinstance(artifact, Mapping):
                errors.append(f"{section}.{name} is not a file record")
                continue
            errors.extend(f"{section}.{name}: {error}" for error in verify_file_record(artifact))
    execution = record.get("execution", {})
    command_logs = execution.get("command_logs", []) if isinstance(execution, Mapping) else []
    for index, artifact in enumerate(command_logs):
        if not isinstance(artifact, Mapping):
            errors.append(f"execution.command_logs.{index} is not a file record")
            continue
        errors.extend(
            f"execution.command_logs.{index}: {error}"
            for error in verify_file_record(artifact)
        )
    return errors


def _build_legacy_run_ledger_entries(
    sidecar_paths: Sequence[str | Path],
    *,
    schema_path: str | Path,
    required_matrices: Sequence[str],
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    seen_units: set[str] = set()
    required_outputs = {"tracks", "sift2_weights", "sift2_mu", *required_matrices}
    run_ids: set[str] = set()
    recipe_ids: set[str] = set()
    for sidecar_path in sorted(Path(path).resolve() for path in sidecar_paths):
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        schema_errors = validate_schema(sidecar, schema_path)
        if schema_errors:
            raise ValueError(
                f"invalid provenance sidecar {sidecar_path}: " + "; ".join(schema_errors)
            )
        if sidecar.get("status") != "PASS":
            raise ValueError(f"non-PASS provenance cannot enter the run ledger: {sidecar_path}")
        artifact_errors = verify_provenance_artifacts(sidecar)
        if artifact_errors:
            raise ValueError(
                f"provenance artifact verification failed for {sidecar_path}: "
                + "; ".join(artifact_errors)
            )
        unit = str(sidecar["unit"])
        if unit in seen_units:
            raise ValueError(f"duplicate provenance unit: {unit}")
        seen_units.add(unit)
        output_names = set(sidecar["outputs"])
        if output_names != required_outputs:
            raise ValueError(
                f"provenance outputs differ from the frozen contract for {unit}: "
                f"{sorted(output_names)}"
            )
        run_ids.add(str(sidecar["run_id"]))
        recipe_ids.add(str(sidecar["recipe_id"]))
        identity = sidecar["source_identity"]
        entries.append(
            {
                "schema_version": LEGACY_LEDGER_SCHEMA_VERSION,
                "run_id": sidecar["run_id"],
                "recipe_id": sidecar["recipe_id"],
                "unit": unit,
                "subject_id": identity["subject_id"],
                "dti_image_id": identity["dti_image_id"],
                "t1_image_id": identity["t1_image_id"],
                "status": "PASS",
                "provenance": file_record(sidecar_path),
                "outputs": sidecar["outputs"],
                "matrix_qc_status": sidecar["qc"]["matrix"]["status"],
                "human_visual_qc": sidecar["qc"]["preflight"]["human_visual_qc"],
            }
        )
    if not entries:
        raise ValueError("run ledger requires at least one PASS provenance sidecar")
    if len(run_ids) != 1 or len(recipe_ids) != 1:
        raise ValueError("all run-ledger entries must share one run_id and recipe_id")
    return sorted(entries, key=lambda row: row["unit"])


def _parse_datetime(value: Any, *, field: str) -> datetime:
    text = str(value)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field} is not an ISO-8601 timestamp: {text!r}") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{field} must include a timezone: {text!r}")
    return parsed.astimezone(timezone.utc)


def validate_terminal_semantics(
    record: Mapping[str, Any],
    *,
    required_outcomes: Sequence[str] = REQUIRED_OUTCOMES,
    required_matrices: Sequence[str] = MATRIX_OUTCOMES,
) -> list[str]:
    """Validate cross-field terminal-state rules that JSON Schema cannot express.

    A terminal record is an attrition fact, not a value-repair mechanism.  An
    outcome is either backed by its immutable artifact or explicitly ``NA``;
    no missing/invalid outcome is converted to a numeric sentinel.
    """

    errors: list[str] = []
    status = str(record.get("status", ""))
    if status not in TERMINAL_STATUSES:
        errors.append(f"invalid terminal status: {status!r}")

    outcomes = record.get("outcome_validity", {})
    if not isinstance(outcomes, Mapping):
        return errors + ["outcome_validity is not a mapping"]
    required = tuple(dict.fromkeys(str(name) for name in required_outcomes))
    missing = sorted(set(required) - set(outcomes))
    unexpected = sorted(set(outcomes) - set(required))
    if missing:
        errors.append("missing required outcomes: " + ", ".join(missing))
    if unexpected:
        errors.append("unexpected outcomes: " + ", ".join(unexpected))

    states: list[str] = []
    outputs = record.get("outputs", {})
    if not isinstance(outputs, Mapping):
        errors.append("outputs is not a mapping")
        outputs = {}
    for name in required:
        outcome = outcomes.get(name)
        if not isinstance(outcome, Mapping):
            continue
        state = str(outcome.get("status", ""))
        reason = outcome.get("reason")
        states.append(state)
        if state not in OUTCOME_STATES:
            errors.append(f"{name}: invalid outcome status {state!r}")
            continue
        if state == "VALID":
            if reason is not None:
                errors.append(f"{name}: VALID outcome must have null reason")
            if name in required_matrices:
                artifact = outcome.get("artifact")
                if not isinstance(artifact, Mapping):
                    errors.append(f"{name}: VALID matrix outcome requires an artifact")
                elif outputs.get(name) != artifact:
                    errors.append(
                        f"{name}: outcome artifact differs from outputs.{name}"
                    )
        else:
            if not isinstance(reason, str) or not reason.strip():
                errors.append(f"{name}: NA outcome requires a reason")
            if "artifact" in outcome:
                errors.append(f"{name}: NA outcome cannot carry an artifact")
            if name in required_matrices and name in outputs:
                errors.append(f"{name}: NA outcome cannot appear in valid outputs")

    valid_count = states.count("VALID")
    na_count = states.count("NA")
    primary_reason = record.get("primary_failure_reason")
    stage = record.get("terminal_stage")
    if status == "PASS":
        if na_count or valid_count != len(required):
            errors.append("PASS requires every required outcome to be VALID")
        if primary_reason is not None:
            errors.append("PASS requires null primary_failure_reason")
        if stage != "complete":
            errors.append("PASS requires terminal_stage=complete")
        expected_outputs = {
            "tracks",
            "sift2_weights",
            "sift2_mu",
            *required_matrices,
        }
        if set(outputs) != expected_outputs:
            errors.append(
                "PASS outputs differ from the frozen analytical contract: "
                + ", ".join(sorted(set(outputs) ^ expected_outputs))
            )
    elif status == "PARTIAL":
        if not valid_count or not na_count:
            errors.append("PARTIAL requires at least one VALID and one NA outcome")
        if outcomes.get("biological_inference", {}).get("status") != "NA":
            errors.append("PARTIAL cannot be valid for biological inference")
        if not isinstance(primary_reason, str) or not primary_reason.strip():
            errors.append("PARTIAL requires primary_failure_reason")
        if stage == "complete":
            errors.append("PARTIAL cannot use terminal_stage=complete")
    elif status == "FAIL":
        if valid_count or na_count != len(required):
            errors.append("FAIL requires every required outcome to be NA")
        if outputs:
            errors.append("FAIL cannot expose any valid analytical outputs")
        if not isinstance(primary_reason, str) or not primary_reason.strip():
            errors.append("FAIL requires primary_failure_reason")
        if stage == "complete":
            errors.append("FAIL cannot use terminal_stage=complete")

    if outcomes.get("tractography", {}).get("status") == "VALID":
        tract_outputs = {"tracks", "sift2_weights", "sift2_mu"}
        if not tract_outputs.issubset(outputs):
            errors.append(
                "VALID tractography requires tracks, sift2_weights, and sift2_mu outputs"
            )

    try:
        started = _parse_datetime(record.get("started_utc"), field="started_utc")
        ended = _parse_datetime(record.get("ended_utc"), field="ended_utc")
        if ended < started:
            errors.append("ended_utc precedes started_utc")
    except ValueError as exc:
        errors.append(str(exc))

    return errors


def _build_terminal_run_ledger_entries(
    sidecar_paths: Sequence[str | Path],
    *,
    schema_path: str | Path,
    required_matrices: Sequence[str],
    expected_units: Sequence[str] | None,
    required_outcomes: Sequence[str],
) -> list[dict[str, Any]]:
    if not sidecar_paths:
        raise ValueError("terminal run ledger cannot be empty")
    entries: list[dict[str, Any]] = []
    seen_units: set[str] = set()
    run_ids: set[str] = set()
    recipe_ids: set[str] = set()
    for sidecar_path in sorted(Path(path).resolve() for path in sidecar_paths):
        if not sidecar_path.is_file():
            raise FileNotFoundError(sidecar_path)
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        schema_errors = validate_schema(sidecar, schema_path)
        semantic_errors = validate_terminal_semantics(
            sidecar,
            required_outcomes=required_outcomes,
            required_matrices=required_matrices,
        )
        if schema_errors or semantic_errors:
            raise ValueError(
                f"invalid terminal record {sidecar_path}: "
                + "; ".join(schema_errors + semantic_errors)
            )
        artifact_errors = verify_provenance_artifacts(sidecar)
        for name, outcome in sidecar["outcome_validity"].items():
            artifact = outcome.get("artifact")
            # Matrix outcome artifacts are the same immutable records already
            # checked under outputs.  Do not read large matrices twice.
            if artifact is not None and sidecar["outputs"].get(name) != artifact:
                artifact_errors.extend(
                    f"outcome_validity.{name}: {error}"
                    for error in verify_file_record(artifact)
                )
        if artifact_errors:
            raise ValueError(
                f"terminal-record artifact verification failed for {sidecar_path}: "
                + "; ".join(artifact_errors)
            )
        unit = str(sidecar["unit"])
        if unit in seen_units:
            raise ValueError(f"duplicate terminal-record unit: {unit}")
        seen_units.add(unit)
        run_ids.add(str(sidecar["run_id"]))
        recipe_ids.add(str(sidecar["recipe_id"]))
        identity = sidecar["source_identity"]
        entries.append(
            {
                "schema_version": LEDGER_SCHEMA_VERSION,
                "entry_type": "connectome_subject_terminal_state",
                "run_id": sidecar["run_id"],
                "recipe_id": sidecar["recipe_id"],
                "unit": unit,
                "subject_id": identity["subject_id"],
                "dti_image_id": identity["dti_image_id"],
                "t1_image_id": identity["t1_image_id"],
                "status": sidecar["status"],
                "terminal_stage": sidecar["terminal_stage"],
                "primary_failure_reason": sidecar["primary_failure_reason"],
                "secondary_flags": sidecar["secondary_flags"],
                "started_utc": sidecar["started_utc"],
                "ended_utc": sidecar["ended_utc"],
                "terminal_record": file_record(sidecar_path),
                "outcome_validity": sidecar["outcome_validity"],
                "outputs": sidecar["outputs"],
            }
        )

    if len(run_ids) != 1 or len(recipe_ids) != 1:
        raise ValueError("all terminal records must share one run_id and recipe_id")
    if expected_units is not None:
        expected = tuple(str(unit) for unit in expected_units)
        if len(expected) != len(set(expected)):
            raise ValueError("expected_units contains duplicates")
        missing = sorted(set(expected) - seen_units)
        unexpected = sorted(seen_units - set(expected))
        if missing or unexpected:
            details = []
            if missing:
                details.append("missing=" + ",".join(missing))
            if unexpected:
                details.append("unexpected=" + ",".join(unexpected))
            raise ValueError(
                "terminal-record units differ from expected attempted units: "
                + "; ".join(details)
            )
        order = {unit: index for index, unit in enumerate(expected)}
        entries.sort(key=lambda row: order[row["unit"]])
    else:
        entries.sort(key=lambda row: row["unit"])
    return entries


def build_run_ledger_entries(
    sidecar_paths: Sequence[str | Path],
    *,
    schema_path: str | Path,
    required_matrices: Sequence[str],
    expected_units: Sequence[str] | None = None,
    required_outcomes: Sequence[str] = REQUIRED_OUTCOMES,
) -> list[dict[str, Any]]:
    """Build either a preserved v1 PASS ledger or the v2 terminal ledger.

    Version dispatch preserves the immutable v2.0.11/v1-schema evidence while
    making mixed PASS/PARTIAL/FAIL publication explicit for v2.1 and later.
    """

    schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))
    version = schema.get("properties", {}).get("schema_version", {}).get("const")
    if version == LEGACY_PROVENANCE_SCHEMA_VERSION:
        if expected_units is not None:
            raise ValueError("expected_units is supported only by terminal schema v2")
        return _build_legacy_run_ledger_entries(
            sidecar_paths,
            schema_path=schema_path,
            required_matrices=required_matrices,
        )
    if version != PROVENANCE_SCHEMA_VERSION:
        raise ValueError(f"unsupported provenance schema version: {version!r}")
    return _build_terminal_run_ledger_entries(
        sidecar_paths,
        schema_path=schema_path,
        required_matrices=required_matrices,
        expected_units=expected_units,
        required_outcomes=required_outcomes,
    )


ANALYSIS_READY_FIELDS = (
    "run_id",
    "unit",
    "subject_id",
    "dti_image_id",
    "t1_image_id",
    "recipe_id",
    "status",
    "terminal_record",
    "terminal_record_sha256",
    *MATRIX_OUTCOMES,
)
OUTCOME_VALIDITY_FIELDS = (
    "run_id",
    "unit",
    "subject_id",
    "recipe_id",
    "terminal_status",
    "terminal_stage",
    "outcome",
    "validity",
    "reason",
    "artifact",
    "artifact_sha256",
)


def _csv_payload(fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(fieldnames), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def build_publication_rows(
    entries: Sequence[Mapping[str, Any]],
    *,
    required_outcomes: Sequence[str] = REQUIRED_OUTCOMES,
    required_matrices: Sequence[str] = MATRIX_OUTCOMES,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Separate analysis-ready PASS units from the complete validity audit."""

    analysis_rows: list[dict[str, Any]] = []
    outcome_rows: list[dict[str, Any]] = []
    for entry in entries:
        outcomes = entry["outcome_validity"]
        for outcome_name in required_outcomes:
            outcome = outcomes[outcome_name]
            artifact = outcome.get("artifact") or {}
            outcome_rows.append(
                {
                    "run_id": entry["run_id"],
                    "unit": entry["unit"],
                    "subject_id": entry["subject_id"],
                    "recipe_id": entry["recipe_id"],
                    "terminal_status": entry["status"],
                    "terminal_stage": entry["terminal_stage"],
                    "outcome": outcome_name,
                    "validity": outcome["status"],
                    "reason": outcome["reason"] or "",
                    "artifact": artifact.get("path", ""),
                    "artifact_sha256": artifact.get("sha256", ""),
                }
            )
        if (
            entry["status"] == "PASS"
            and outcomes["biological_inference"]["status"] == "VALID"
        ):
            outputs = entry["outputs"]
            analysis_rows.append(
                {
                    "run_id": entry["run_id"],
                    "unit": entry["unit"],
                    "subject_id": entry["subject_id"],
                    "dti_image_id": entry["dti_image_id"],
                    "t1_image_id": entry["t1_image_id"],
                    "recipe_id": entry["recipe_id"],
                    "status": "PASS",
                    "terminal_record": entry["terminal_record"]["path"],
                    "terminal_record_sha256": entry["terminal_record"]["sha256"],
                    **{name: outputs[name]["path"] for name in required_matrices},
                }
            )
    return analysis_rows, outcome_rows


def _run_outcome(entries: Sequence[Mapping[str, Any]]) -> str:
    statuses = [entry["status"] for entry in entries]
    if statuses and all(status == "PASS" for status in statuses):
        return "PASS"
    if statuses and all(status == "FAIL" for status in statuses):
        return "FAIL"
    return "PARTIAL"


def _validate_existing_publication(
    *,
    entries: Sequence[Mapping[str, Any]],
    expected_units: Sequence[str],
    analysis_payload: bytes,
    outcome_payload: bytes,
    ledger_payload: bytes,
    analysis_manifest_path: Path,
    outcome_manifest_path: Path,
    ledger_path: Path,
    ledger_manifest_path: Path,
    ledger_schema_path: str | Path,
    run_context_path: str | Path,
) -> dict[str, Any]:
    expected_payloads = {
        analysis_manifest_path: analysis_payload,
        outcome_manifest_path: outcome_payload,
        ledger_path: ledger_payload,
    }
    for path, payload in expected_payloads.items():
        if path.read_bytes() != payload:
            raise ValueError(f"existing immutable publication differs: {path}")
    manifest = json.loads(ledger_manifest_path.read_text(encoding="utf-8"))
    schema_errors = validate_schema(manifest, ledger_schema_path)
    if schema_errors:
        raise ValueError(
            "existing run-ledger manifest fails schema: " + "; ".join(schema_errors)
        )
    if manifest.get("expected_units") != list(expected_units):
        raise ValueError("existing run-ledger expected_units differ")
    if manifest.get("run_id") != entries[0]["run_id"]:
        raise ValueError("existing run-ledger run_id differs")
    if manifest.get("recipe_id") != entries[0]["recipe_id"]:
        raise ValueError("existing run-ledger recipe_id differs")
    expected_summary = {
        "expected": len(expected_units),
        "terminal": len(entries),
        "pass": sum(entry["status"] == "PASS" for entry in entries),
        "partial": sum(entry["status"] == "PARTIAL" for entry in entries),
        "fail": sum(entry["status"] == "FAIL" for entry in entries),
        "analysis_ready": sum(
            entry["status"] == "PASS"
            and entry["outcome_validity"]["biological_inference"]["status"] == "VALID"
            for entry in entries
        ),
        "outcome_valid_rows": sum(
            outcome["status"] == "VALID"
            for entry in entries
            for outcome in entry["outcome_validity"].values()
        ),
    }
    if manifest.get("summary") != expected_summary:
        raise ValueError("existing run-ledger summary differs from terminal records")
    if manifest.get("run_outcome") != _run_outcome(entries):
        raise ValueError("existing run-ledger run_outcome differs from terminal records")
    if manifest.get("run_context") != file_record(run_context_path):
        raise ValueError("existing run-ledger run_context differs")
    terminal_records = {
        entry["unit"]: entry["terminal_record"] for entry in entries
    }
    if manifest.get("subject_terminal_records") != terminal_records:
        raise ValueError("existing run-ledger terminal-record map differs")
    for field in (
        "run_context",
        "run_ledger",
        "analysis_ready_manifest",
        "outcome_validity_manifest",
    ):
        errors = verify_file_record(manifest[field])
        if errors:
            raise ValueError(
                f"existing run-ledger {field} verification failed: "
                + "; ".join(errors)
            )
    return manifest


def publish_terminal_ledger(
    terminal_record_paths: Sequence[str | Path],
    *,
    provenance_schema_path: str | Path,
    ledger_schema_path: str | Path,
    run_context_path: str | Path,
    expected_units: Sequence[str],
    required_matrices: Sequence[str],
    analysis_manifest_path: str | Path,
    outcome_manifest_path: str | Path,
    ledger_path: str | Path,
    ledger_manifest_path: str | Path,
    generated_utc: str | None = None,
) -> dict[str, Any]:
    """Atomically publish a complete all-attempt attrition ledger.

    Existing evidence is accepted only when every byte and every referenced
    digest matches the records being finalized.  A partial prior publication
    is always rejected rather than repaired in place.
    """

    expected = tuple(str(unit) for unit in expected_units)
    entries = build_run_ledger_entries(
        terminal_record_paths,
        schema_path=provenance_schema_path,
        required_matrices=required_matrices,
        expected_units=expected,
        required_outcomes=REQUIRED_OUTCOMES,
    )
    analysis_rows, outcome_rows = build_publication_rows(
        entries,
        required_outcomes=REQUIRED_OUTCOMES,
        required_matrices=required_matrices,
    )
    analysis_payload = _csv_payload(ANALYSIS_READY_FIELDS, analysis_rows)
    outcome_payload = _csv_payload(OUTCOME_VALIDITY_FIELDS, outcome_rows)
    ledger_payload = b"".join(
        json.dumps(entry, sort_keys=True, separators=(",", ":")).encode("utf-8")
        + b"\n"
        for entry in entries
    )
    if not ledger_payload:
        raise ValueError("terminal run ledger cannot be empty")

    analysis_path = Path(analysis_manifest_path)
    outcome_path = Path(outcome_manifest_path)
    ledger_file = Path(ledger_path)
    ledger_manifest_file = Path(ledger_manifest_path)
    publication_paths = (
        analysis_path,
        outcome_path,
        ledger_file,
        ledger_manifest_file,
    )
    existing = [path.exists() for path in publication_paths]
    if any(existing):
        if not all(existing):
            missing = [str(path) for path, present in zip(publication_paths, existing) if not present]
            raise ValueError(
                "partial immutable publication detected; missing: " + ", ".join(missing)
            )
        return _validate_existing_publication(
            entries=entries,
            expected_units=expected,
            analysis_payload=analysis_payload,
            outcome_payload=outcome_payload,
            ledger_payload=ledger_payload,
            analysis_manifest_path=analysis_path,
            outcome_manifest_path=outcome_path,
            ledger_path=ledger_file,
            ledger_manifest_path=ledger_manifest_file,
            ledger_schema_path=ledger_schema_path,
            run_context_path=run_context_path,
        )

    for path in publication_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
    write_immutable_bytes(analysis_path, analysis_payload)
    write_immutable_bytes(outcome_path, outcome_payload)
    write_immutable_bytes(ledger_file, ledger_payload)

    status_counts = {
        status.lower(): sum(entry["status"] == status for entry in entries)
        for status in TERMINAL_STATUSES
    }
    terminal_records = {
        entry["unit"]: entry["terminal_record"] for entry in entries
    }
    manifest = {
        "schema_version": LEDGER_SCHEMA_VERSION,
        "ledger_type": "connectome_run_ledger",
        "status": "COMPLETE",
        "run_outcome": _run_outcome(entries),
        "generated_utc": generated_utc
        or datetime.now(timezone.utc).isoformat(),
        "run_id": entries[0]["run_id"],
        "recipe_id": entries[0]["recipe_id"],
        "run_context": file_record(run_context_path),
        "expected_units": list(expected),
        "summary": {
            "expected": len(expected),
            "terminal": len(entries),
            "pass": status_counts["pass"],
            "partial": status_counts["partial"],
            "fail": status_counts["fail"],
            "analysis_ready": len(analysis_rows),
            "outcome_valid_rows": sum(
                row["validity"] == "VALID" for row in outcome_rows
            ),
        },
        "run_ledger": file_record(ledger_file),
        "analysis_ready_manifest": file_record(analysis_path),
        "outcome_validity_manifest": file_record(outcome_path),
        "subject_terminal_records": terminal_records,
        "terminal_statuses": list(TERMINAL_STATUSES),
        "analysis_ready_statuses": ["PASS"],
        "external_digest_anchor_required_for_release": True,
    }
    if manifest["summary"]["expected"] != manifest["summary"]["terminal"]:
        raise ValueError("expected and terminal unit counts differ")
    if (
        manifest["summary"]["pass"]
        + manifest["summary"]["partial"]
        + manifest["summary"]["fail"]
        != manifest["summary"]["terminal"]
    ):
        raise ValueError("terminal status counts do not sum to terminal count")
    schema_errors = validate_schema(manifest, ledger_schema_path)
    if schema_errors:
        raise ValueError(
            "run-ledger schema validation failed: " + "; ".join(schema_errors)
        )
    write_immutable_json(ledger_manifest_file, manifest)
    return manifest


def make_failure_terminal_record(
    *,
    run_id: str,
    recipe_id: str,
    unit: str,
    terminal_stage: str,
    primary_failure_reason: str,
    started_utc: str,
    ended_utc: str,
    source_identity: Mapping[str, Any],
    source_identity_hashes: Mapping[str, str],
    staged_input_hashes: Mapping[str, str],
    source_files: Mapping[str, Mapping[str, Any]],
    contract_files: Mapping[str, Mapping[str, Any]],
    intermediate_files: Mapping[str, Mapping[str, Any]] | None = None,
    processing_contract: Mapping[str, Any] | None = None,
    command_logs: Sequence[Mapping[str, Any]] = (),
    secondary_flags: Sequence[str] = (),
) -> dict[str, Any]:
    """Build a conservative FAIL record; all analytical outcomes remain NA."""

    reason = str(primary_failure_reason).strip()
    if not reason:
        raise ValueError("failure terminal record requires a reason")
    outcomes = {
        name: {"status": "NA", "reason": reason} for name in REQUIRED_OUTCOMES
    }
    return {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "record_type": "connectome_subject_terminal_state",
        "status": "FAIL",
        "generated_utc": ended_utc,
        "started_utc": started_utc,
        "ended_utc": ended_utc,
        "recipe_id": recipe_id,
        "run_id": run_id,
        "unit": unit,
        "terminal_stage": terminal_stage,
        "primary_failure_reason": reason,
        "secondary_flags": list(dict.fromkeys(str(flag) for flag in secondary_flags if str(flag))),
        "source_identity": dict(source_identity),
        "source_identity_hashes": dict(source_identity_hashes),
        "staged_input_hashes": dict(staged_input_hashes),
        "source_files": dict(source_files),
        "contract_files": dict(contract_files),
        "intermediate_files": dict(intermediate_files or {}),
        "outputs": {},
        "outcome_validity": outcomes,
        "execution": {
            "attempt_started_utc": started_utc,
            "terminal_utc": ended_utc,
            "command_logs": list(command_logs),
            "internal_commands": [],
        },
        "processing_contract": dict(
            processing_contract
            or {
                key: {}
                for key in (
                    "registration",
                    "atlas",
                    "fod",
                    "tensor",
                    "tractography",
                    "sift2",
                    "assignment",
                )
            }
        ),
        "qc": {},
        "safety": {
            "production_overwrite": False,
            "automatic_fallbacks": False,
            "density_route_selection": False,
            "invalid_values_replaced": False,
        },
    }
