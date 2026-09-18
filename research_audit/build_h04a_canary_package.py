#!/usr/bin/env python3
"""Build the non-executable SL-P0-15 / SL-H04A canary decision package.

This builder is intentionally metadata-only.  It reads CSV/JSON/text metadata,
filesystem identities and file sizes.  It never opens an image payload, invokes
MRtrix/Snakemake, or creates an executable APPROVED decision.
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path("/home/ec2-user/exp")
AUDIT_ROOT = ROOT / "research_audit"
PARENT_MANIFEST = AUDIT_ROOT / "outputs/connectome_v2_input_manifest_v2.csv"
SOURCE_PROJECTION = (
    AUDIT_ROOT
    / "outputs/source_metadata_workflow_preflight_v1/source_metadata_workflow_projection_v1.csv"
)
SOURCE_PROJECTION_VALIDATION = (
    AUDIT_ROOT
    / "outputs/source_metadata_workflow_preflight_v1/source_metadata_workflow_projection_validation_v1.json"
)
PAIR_MANIFEST = AUDIT_ROOT / "outputs/available_data_pair_manifest_v2.csv"
CONTENT_INVENTORY = (
    AUDIT_ROOT
    / "outputs/available_data_content_lock_v2/available_data_file_content_inventory_v2.csv"
)
RECIPE_CONFIG = ROOT / "configs/connectome_v2.yaml"
ACQUISITION_SCHEMA = ROOT / "scforge/workflow/schemas/acquisition_manifest_v2.schema.json"
WORKFLOW_SOURCE_MANIFEST = ROOT / "scforge/workflow/workflow_source_manifest.tsv"
ENVIRONMENT_CONTRACT = ROOT / "scforge/workflow/environment_contract.yaml"
MATRIX_DICTIONARY = AUDIT_ROOT / "matrix_data_dictionary.md"

DEFAULT_OUTPUT = AUDIT_ROOT / "outputs/h04a_canary_package_v1"
PROPOSED_RUN_ROOT = Path(
    "/data/derivatives/scforge_v2/h04a_canary_20260718_v1"
)
SELECTION_SEED = "SL-P0-15|technical-only|h04a-canary-selection-v1"
CANARY_SIZE = 24

# These are the only row-level variables available to the selection objective.
# diagnosis_at_dti and analysis_role are deliberately absent.  The diagnosis
# field remains in the exact parent row because the acquisition schema requires
# byte-equivalent rows, but it is not consulted until the selection is locked.
SELECTION_SOURCE_FIELDS = frozenset(
    {
        "pair_content_bundle_sha256",
        "subject_id",
        "dti_image_id",
        "t1_image_id",
        "dti_raw_total_bytes",
        "t1_raw_total_bytes",
        "t1_source_kind",
        "timing_stratum",
        "phase",
        "site",
        "manufacturer",
        "scanner_model",
        "protocol",
        "dti_series_uid_syntax_status",
        "dti_dicom_in_plane_phase_encoding_status",
        "dti_public_gradient_headers_present",
        "dti_public_b_value_headers_present",
        "legacy_sc_matrix_qc_status",
        "legacy_mask_quality",
    }
)
FORBIDDEN_SELECTION_FIELDS = frozenset(
    {
        "diagnosis_at_dti",
        "analysis_role",
        "diagnosis_reconciled_harmonized",
        "diagnosis_reconciled_primary_analysis_group",
        "group",
        "group_corrected",
        "legacy_connectome_density",
        "legacy_registration_ncc",
        "legacy_label_survival",
        "corrected_rerun_analysis_eligible",
    }
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def generated_file_record(staged_path: Path, final_path: Path) -> dict[str, Any]:
    """Record final identity while bytes are still in the atomic staging dir."""

    return {
        "path": str(final_path.resolve()),
        "sha256": sha256_file(staged_path),
        "size_bytes": staged_path.stat().st_size,
    }


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader]
        fields = list(reader.fieldnames or ())
    if not rows or not fields:
        raise ValueError(f"empty CSV: {path}")
    return rows, fields


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def stable_unit(row: Mapping[str, str]) -> str:
    return f"{row['subject_id']}_I{row['dti_image_id']}"


def pair_id(row: Mapping[str, str]) -> str:
    return f"{row['subject_id']}_I{row['dti_image_id']}_I{row['t1_image_id']}"


def manufacturer_family(value: str) -> str:
    normalized = value.strip().upper()
    if "SIEMENS" in normalized:
        return "SIEMENS"
    if normalized.startswith("GE"):
        return "GE"
    if "PHILIPS" in normalized:
        return "PHILIPS"
    return "OTHER"


def positive_header_class(value: str) -> str:
    try:
        return "present" if int(value) > 0 else "absent"
    except (TypeError, ValueError):
        return "absent"


def legacy_mask_class(value: str) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "missing"
    if not math.isfinite(numeric):
        return "missing"
    if numeric < 0.4:
        return "low_lt_0.4"
    if numeric < 0.6:
        return "mid_0.4_0.6"
    return "high_ge_0.6"


def equal_rank_quartiles(
    rows: Sequence[Mapping[str, str]], value_field: str
) -> dict[str, str]:
    ordered = sorted(
        rows,
        key=lambda row: (
            int(row[value_field]),
            hashlib.sha256(
                (SELECTION_SEED + "|quartile|" + row["pair_content_bundle_sha256"]).encode()
            ).hexdigest(),
        ),
    )
    result: dict[str, str] = {}
    for index, row in enumerate(ordered):
        quartile = min(4, (4 * index) // len(ordered) + 1)
        result[pair_id(row)] = f"Q{quartile}"
    return result


def build_candidates() -> tuple[list[dict[str, Any]], list[str], list[dict[str, str]]]:
    if SELECTION_SOURCE_FIELDS & FORBIDDEN_SELECTION_FIELDS:
        raise AssertionError("selection source field allowlist contains forbidden fields")
    parent, parent_fields = read_csv(PARENT_MANIFEST)
    projection, _ = read_csv(SOURCE_PROJECTION)
    pair_rows, _ = read_csv(PAIR_MANIFEST)
    if len(parent) != 530 or len(projection) != 530 or len(pair_rows) != 530:
        raise ValueError("H04A builder requires three exact 530-row inputs")
    projection_by_pair = {row["pair_id"]: row for row in projection}
    pair_by_pair = {row["pair_id"]: row for row in pair_rows}
    if len(projection_by_pair) != 530 or len(pair_by_pair) != 530:
        raise ValueError("duplicate pair identifiers in metadata inputs")

    dti_quartiles = equal_rank_quartiles(parent, "dti_raw_total_bytes")
    t1_quartiles = equal_rank_quartiles(parent, "t1_raw_total_bytes")
    candidates: list[dict[str, Any]] = []
    for row in parent:
        pid = pair_id(row)
        if pid not in projection_by_pair or pid not in pair_by_pair:
            raise ValueError(f"metadata join missing for {pid}")
        projection_row = projection_by_pair[pid]
        pair_row = pair_by_pair[pid]
        technical = {
            "manufacturer_family": manufacturer_family(row["manufacturer"]),
            "t1_source_kind": row["t1_source_kind"],
            "timing_stratum": row["timing_stratum"],
            "phase": row["phase"],
            "protocol": row["protocol"],
            "site": row["site"],
            "scanner_model": row["scanner_model"],
            "legacy_qc_status": pair_row["legacy_sc_matrix_qc_status"],
            "uid_syntax": projection_row["dti_series_uid_syntax_status"],
            "pe_header": projection_row[
                "dti_dicom_in_plane_phase_encoding_status"
            ],
            "gradient_public": positive_header_class(
                projection_row["dti_public_gradient_headers_present"]
            ),
            "bvalue_public": positive_header_class(
                projection_row["dti_public_b_value_headers_present"]
            ),
            "legacy_mask_risk": legacy_mask_class(
                pair_row["legacy_mask_quality"]
            ),
            "dti_size_quartile": dti_quartiles[pid],
            "t1_size_quartile": t1_quartiles[pid],
        }
        tie_break = hashlib.sha256(
            (
                SELECTION_SEED
                + "|"
                + row["pair_content_bundle_sha256"]
            ).encode("utf-8")
        ).hexdigest()
        candidates.append(
            {
                "row": row,
                "pair_id": pid,
                "unit": stable_unit(row),
                "technical": technical,
                "tie_break": tie_break,
            }
        )
    return candidates, parent_fields, parent


TARGETS: dict[str, dict[str, int]] = {
    "manufacturer_family": {"SIEMENS": 8, "GE": 8, "PHILIPS": 8},
    "t1_source_kind": {"dicom_series": 12, "nifti_single": 12},
    "timing_stratum": {"le_90_days": 8, "days_91_180": 3, "gt_180_days": 13},
    "phase": {"ADNI 2": 4, "ADNI 3": 16, "ADNI 4": 4},
    "legacy_qc_status": {
        "PASS": 4,
        "FAIL_REGISTRATION": 5,
        "WARN_SPARSE_NODE": 5,
        "FAIL_LABEL_COVERAGE": 5,
        "FAIL_STREAMLINE_ASSIGNMENT": 5,
    },
    "uid_syntax": {"VALID": 12, "NONCONFORMANT_LEADING_ZERO_COMPONENT": 12},
    "pe_header": {"PASS": 22, "FAIL_IN_PLANE_PE_MISSING": 2},
    "gradient_public": {"present": 6, "absent": 18},
    "bvalue_public": {"present": 6, "absent": 18},
    "legacy_mask_risk": {
        "missing": 6,
        "low_lt_0.4": 6,
        "mid_0.4_0.6": 6,
        "high_ge_0.6": 6,
    },
    "dti_size_quartile": {f"Q{index}": 6 for index in range(1, 5)},
    "t1_size_quartile": {f"Q{index}": 6 for index in range(1, 5)},
}
TARGET_WEIGHTS: dict[str, float] = {
    "manufacturer_family": 4.0,
    "t1_source_kind": 3.0,
    "timing_stratum": 2.0,
    "phase": 2.0,
    "legacy_qc_status": 2.0,
    "uid_syntax": 1.5,
    "pe_header": 2.0,
    "gradient_public": 2.0,
    "bvalue_public": 1.0,
    "legacy_mask_risk": 1.5,
    "dti_size_quartile": 1.0,
    "t1_size_quartile": 1.0,
}
UNIQUE_COVERAGE_WEIGHTS = {"protocol": 7.0, "scanner_model": 2.0, "site": 0.5}


def selection_score(selected: Sequence[Mapping[str, Any]]) -> float:
    penalty = 0.0
    for feature, target in TARGETS.items():
        counts = collections.Counter(item["technical"][feature] for item in selected)
        penalty += TARGET_WEIGHTS[feature] * sum(
            abs(counts[value] - expected) for value, expected in target.items()
        )
    reward = 0.0
    for feature, weight in UNIQUE_COVERAGE_WEIGHTS.items():
        reward += weight * len(
            {item["technical"][feature] for item in selected}
        )
    site_counts = collections.Counter(
        item["technical"]["site"] for item in selected
    )
    penalty += 2.0 * sum(max(0, count - 2) for count in site_counts.values())
    return penalty - reward


def select_canary(candidates: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    remaining = list(candidates)
    selected: list[dict[str, Any]] = []
    while len(selected) < CANARY_SIZE:
        best = min(
            remaining,
            key=lambda item: (
                selection_score([*selected, item]),
                item["tie_break"],
            ),
        )
        selected.append(best)
        remaining.remove(best)

    # Deterministic one-swap local improvement.  Only technical features enter
    # selection_score; the tie-break derives from locked source identity.
    for _ in range(100):
        current_score = selection_score(selected)
        best_swap: tuple[tuple[Any, ...], int, dict[str, Any], dict[str, Any]] | None = None
        for index, outgoing in enumerate(selected):
            base = selected[:index] + selected[index + 1 :]
            for incoming in remaining:
                score = selection_score([*base, incoming])
                key = (score, incoming["tie_break"], outgoing["tie_break"])
                if score < current_score and (
                    best_swap is None or key < best_swap[0]
                ):
                    best_swap = (key, index, incoming, outgoing)
        if best_swap is None:
            break
        _, index, incoming, outgoing = best_swap
        selected[index] = incoming
        remaining.remove(incoming)
        remaining.append(outgoing)
    return selected


def parse_duration(value: str) -> int:
    pieces = [int(item) for item in value.split(":")]
    if len(pieces) == 2:
        return pieces[0] * 60 + pieces[1]
    if len(pieces) == 3:
        return pieces[0] * 3600 + pieces[1] * 60 + pieces[2]
    raise ValueError(f"unsupported duration: {value}")


def numeric_summary(values: Iterable[int]) -> dict[str, Any]:
    ordered = sorted(int(value) for value in values)
    if not ordered:
        return {"n": 0}

    def quantile(fraction: float) -> int:
        return ordered[round((len(ordered) - 1) * fraction)]

    return {
        "n": len(ordered),
        "minimum": ordered[0],
        "p10": quantile(0.10),
        "median": quantile(0.50),
        "p90": quantile(0.90),
        "maximum": ordered[-1],
        "mean": sum(ordered) / len(ordered),
    }


def runtime_evidence() -> dict[str, Any]:
    evidence: dict[str, Any] = {}
    eddy_path = Path("/data/derivatives/qc/eddy_runtime_notes.txt")
    step7_path = Path("/data/derivatives/qc/step7_runtime_notes.txt")
    if eddy_path.is_file():
        text = eddy_path.read_text(encoding="utf-8", errors="replace")
        durations = [
            parse_duration(match.group(1))
            for match in re.finditer(
                r"= ([0-9]+:[0-9]+(?::[0-9]+)?) \| status=ok", text
            )
        ]
        evidence["legacy_eddy_success_seconds"] = {
            **numeric_summary(durations),
            "source": file_record(eddy_path),
            "comparability": "historical heterogeneous workflow; timing evidence only",
        }
    if step7_path.is_file():
        text = step7_path.read_text(encoding="utf-8", errors="replace")
        for stage in ("prep", "fod", "tracks", "post"):
            durations = [
                parse_duration(match.group(1))
                for match in re.finditer(
                    rf"stage={stage} \| time taken: ([0-9]+:[0-9]+(?::[0-9]+)?)",
                    text,
                )
            ]
            evidence[f"legacy_step7_{stage}_seconds"] = {
                **numeric_summary(durations),
                "source": file_record(step7_path),
                "comparability": "historical heterogeneous workflow; timing evidence only",
            }
    return evidence


def du_bytes(path: Path) -> int | None:
    if not path.exists():
        return None
    completed = subprocess.run(
        ["du", "-sb", str(path)],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0 or not completed.stdout.strip():
        return None
    return int(completed.stdout.split()[0])


def storage_evidence() -> dict[str, Any]:
    derivative_root = Path("/data/derivatives")
    stage_names = (
        "dti",
        "mif_dwi",
        "mif_denoised",
        "mif_unringed",
        "eddy",
        "biascorr_1",
        "t1_anat",
        "t1_fast",
        "dwi_t1_bbr",
        "parc",
        "fod",
        "tracks",
        "connectomes",
    )
    stage_bytes = {
        name: du_bytes(derivative_root / name) for name in stage_names
    }
    legacy_10m_sizes = [
        path.stat().st_size
        for path in (derivative_root / "qc/sc_matrix_qc").rglob("tracks_10M.tck")
        if path.is_file()
    ]
    legacy_3m_sizes = [
        path.stat().st_size
        for path in (derivative_root / "tracks").glob(
            "*/tracks_final_3000k.tck"
        )
        if path.is_file()
    ]
    disk = shutil.disk_usage(derivative_root)
    return {
        "same_filesystem_identity": {
            "/data/derivatives": f"{os.stat(derivative_root).st_dev}:{os.stat(derivative_root).st_ino}",
            "/home/ec2-user/exp/data/derivatives": (
                f"{os.stat(ROOT / 'data/derivatives').st_dev}:"
                f"{os.stat(ROOT / 'data/derivatives').st_ino}"
            ),
        },
        "historical_stage_root_bytes": stage_bytes,
        "historical_completed_10m_tck_bytes": numeric_summary(legacy_10m_sizes),
        "historical_completed_3m_tck_bytes": numeric_summary(legacy_3m_sizes),
        "filesystem_capacity_bytes": {
            "total": disk.total,
            "used": disk.used,
            "free": disk.free,
        },
        "unit_denominator_for_stage_root_context": 709,
        "comparability": (
            "Historical roots contain a heterogeneous legacy recipe and may include "
            "retained intermediates. They support conservative capacity planning, not "
            "a claim of exact v2 resource use."
        ),
    }


def balance_rows(
    candidates: Sequence[Mapping[str, Any]],
    selected: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    features = [*TARGETS, *UNIQUE_COVERAGE_WEIGHTS]
    for feature in features:
        cohort_counts = collections.Counter(
            item["technical"][feature] for item in candidates
        )
        selected_counts = collections.Counter(
            item["technical"][feature] for item in selected
        )
        for value in sorted(cohort_counts):
            target = TARGETS.get(feature, {}).get(value, "coverage_only")
            records.append(
                {
                    "feature": feature,
                    "value": value,
                    "cohort_count": cohort_counts[value],
                    "selected_count": selected_counts[value],
                    "target_count": target,
                    "target_met": (
                        "NA"
                        if target == "coverage_only"
                        else str(selected_counts[value] == target).lower()
                    ),
                }
            )
    return records


def make_markdown(
    selected: Sequence[Mapping[str, Any]],
    subset_hash: str,
    input_records: Mapping[str, Mapping[str, Any]],
    diagnosis_counts: Mapping[str, int],
    runtime: Mapping[str, Any],
    storage: Mapping[str, Any],
) -> str:
    technical_counts = {
        feature: dict(
            sorted(
                collections.Counter(
                    item["technical"][feature] for item in selected
                ).items()
            )
        )
        for feature in (
            "manufacturer_family",
            "t1_source_kind",
            "timing_stratum",
            "phase",
            "legacy_qc_status",
            "pe_header",
            "gradient_public",
        )
    }
    track = storage.get("historical_completed_10m_tck_bytes", {})
    return f"""# SL-H04A canary-start decision package v1

**Status:** PENDING USER DECISION; non-executable  
**Proposed exact subset:** {len(selected)} units  
**Subset SHA-256:** `{subset_hash}`  
**Proposed isolated run root:** `{PROPOSED_RUN_ROOT}`  
**Recipe:** `connectome-v2.1.0-closed-world-canary-candidate`  

## Recommendation

Approve only a staged, conversion-first canary if the user accepts the exact subset, a minimum of **12 valid and shell-compatible response-calibration units**, a **72-hour H04A wall-clock stop**, and a **150 GB H04A storage reservation**. The approval should stop automatically after the pre-tractography automated-QC and blinded visual-review bundles. It must not authorize tractography, SIFT2, matrices, group inference, dashboard release, or the 530-unit run.

This package does **not** itself grant approval. No launcher-compatible `APPROVED` decision has been created.

If a separate human-signed `APPROVED` record is created without changing any
bound field, the launcher will keep the normative recipe flag false, enforce the
four-core ceiling, reserve the remaining 150-GB envelope, count cumulative
Phase-A plus pre-tractography attempt time against 72 hours, and monitor both
limits while the isolated process group runs. A breach is terminated and
attested as failure. Human H04A/H04B decisions and blinded image review remain
manual. If the launcher or host is forcibly killed before an immutable attempt
end is written, the next invocation stops on the unclosed attempt until a human
resolves it.

## Why this subset is diagnosis-blind

Selection used only prespecified technical metadata and source-identity hashes. Diagnosis, analysis role, connectome values, group contrasts, and corrected biological outcomes were excluded from the selection objective. The exact parent rows necessarily retain the schema-required diagnosis column, but the selector never accesses it. After the subset was immutably selected, a separate aggregate check found: `{json.dumps(dict(diagnosis_counts), sort_keys=True)}`. This check did not alter the subset.

The canary is a deliberately balanced technical stress test, not a miniature prevalence sample. Its technical composition is `{json.dumps(technical_counts, sort_keys=True)}`. It covers {len({item['technical']['protocol'] for item in selected})} protocol keys, {len({item['technical']['scanner_model'] for item in selected})} scanner models, and {len({item['technical']['site'] for item in selected})} sites.

## Conversion-first fail-closed stage

All {len(selected)} rows are source-identity ready but conversion-pending. No row currently has an authoritative BIDS phase-encoding direction or total readout time in the parent manifest. The proposed first stage therefore:

1. Revalidates exact locked source identity before conversion.
2. Converts the exact DTI and T1 sources into the isolated run root and writes new staged-file hashes.
3. Reads phase-encoding direction and total readout time only from the frozen conversion output; exports b-vectors/b-values from that same conversion.
4. Fails each unit independently if direction, readout, gradient count, vector norms/rotation, or shell checks are missing or inconsistent. No default, guess, or imputation is permitted.
5. Proceeds to response estimation only for technically valid units. It freezes no pooled response unless at least 12 shell-compatible PASS units remain and those exact PASS units include at least two manufacturer families plus both signed T1 source classes (`dicom_series` and `nifti_single`).

Two selected rows intentionally represent the 14-row higher-risk class lacking a public DICOM in-plane phase-encoding field; six represent rows with public gradient headers and 18 represent the conversion-dependent class. Even the lower-risk rows remain conversion-pending until the exact conversion proves their normalized metadata.

## Compute and storage envelope

Historical local evidence is heterogeneous and not an exact benchmark for v2. The successful legacy Eddy sample contains {runtime.get('legacy_eddy_success_seconds', {}).get('n', 0)} observations, with median {runtime.get('legacy_eddy_success_seconds', {}).get('median', 'NA')} seconds and p90 {runtime.get('legacy_eddy_success_seconds', {}).get('p90', 'NA')} seconds. Legacy Step-7 timing shows long-tailed tractography; the current 10M TCK sample has n={track.get('n', 0)}, median={track.get('median', 'NA')} bytes, p90={track.get('p90', 'NA')} bytes, and maximum={track.get('maximum', 'NA')} bytes.

The conservative planning envelope is:

- H04A conversion plus response calibration: approximately 6–24 wall-clock hours at no more than four concurrent units on the 64-vCPU host.
- H04A pre-tractography processing and review bundles: an additional 12–36 wall-clock hours; combined H04A hard stop at 72 hours.
- H04A retained storage: reserve 150 GB and stop before exceeding it.
- H04B primary-only tractography, if later approved: plan another 8–36 hours per continued unit before parallelism, with roughly 12 GB per 10M tractogram plus SIFT2/matrix overhead. Reserve roughly 450 GB for 24 primary-only units. The configured sensitivity suite could materially exceed this and requires its own bounded retention/concurrency decision.

## Threshold decisions requested at H04A

### Response calibration

- **Recommended:** at least 12 valid units, with exact clustered-shell compatibility and at least two manufacturer families plus both T1 source classes represented among the valid set. If not met, stop; do not silently reduce or repool.
- More conservative option: require 15 valid units. This reduces sensitivity to a few atypical responses but increases the chance that the canary cannot proceed without expansion.
- Rejected permissive option: fewer than 12. That is too fragile for a pooled calibration intended for later reuse.

### Atlas parcel survival

- Keep the current hard rule unchanged: exact contiguous labels 1–166 and at least one voxel for every label.
- Recommended stronger action: use `<8 voxels` only as a reporting alert tied to the canonical source-atlas minimum; do not make it an automatic exclusion before blinded canary review.
- Do not revive the legacy 50-voxel rule. The canonical AAL3 source includes parcels smaller than 50 voxels, so that rule can reject valid atlas anatomy by construction.

### Endpoint/support metrics

- Keep the current hard semantic invariants: exact 166x166 matrices, positive node volumes, count/support consistency, finite connected-edge metrics, symmetry, and zero diagonal.
- Keep endpoint assignment fraction 0.85, 166 uniquely assigned nodes, top-five endpoint fraction 0.35, zero-count nodes, and density as reporting alerts only during this canary.
- Do not choose hard support thresholds from diagnosis separation or matrix density. Any promotion to a hard continuation threshold belongs at H04B/H04 after blinded spatial review and requires a new frozen decision if it changes the recipe.

## What approval would authorize

- The exact subset manifest with SHA-256 `{subset_hash}`.
- Source-identity verification, write-once normalization, DWI/T1 voxel operations, response-calibration Phase A, pooled-response freeze only if the minimum is met, pre-tractography processing, automated QC, and blinded review-bundle generation inside `{PROPOSED_RUN_ROOT}`.
- Four-unit maximum concurrency, 72-hour H04A elapsed stop, and 150 GB H04A storage stop.

## What approval would not authorize

- Tractography, SIFT2, any of the nine connectome matrices, or H04B continuation.
- Diagnosis-based route selection, biological unblinding, threshold tuning using group separation, or population-effect claims from the canary.
- Full-cohort processing, overwrite of legacy/production derivatives, dashboard inference release, or manuscript confirmation claims.

## Exact upstream identities

{os.linesep.join(f"- `{name}`: `{record['sha256']}` — `{record['path']}`" for name, record in input_records.items())}
"""


def build_package(output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing package: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.", dir=str(output.parent))
    )
    try:
        candidates, parent_fields, parent_rows = build_candidates()
        selected = select_canary(candidates)
        selected.sort(key=lambda item: item["unit"])
        if len(selected) != CANARY_SIZE or len({item["unit"] for item in selected}) != CANARY_SIZE:
            raise AssertionError("canary selection is not 24 unique units")

        subset_path = temporary / "candidate_canary_manifest_v1.csv"
        write_csv(subset_path, [item["row"] for item in selected], parent_fields)
        subset_hash = sha256_file(subset_path)
        subset_record = generated_file_record(
            subset_path, output / subset_path.name
        )

        audit_fields = [
            "selection_order",
            "unit",
            "pair_id",
            "selection_tie_break_sha256",
            "manufacturer_family",
            "protocol",
            "scanner_model",
            "site",
            "phase",
            "t1_source_kind",
            "timing_stratum",
            "legacy_qc_status",
            "legacy_mask_risk",
            "uid_syntax",
            "pe_header",
            "gradient_public",
            "bvalue_public",
            "dti_size_quartile",
            "t1_size_quartile",
            "dti_raw_total_bytes",
            "t1_raw_total_bytes",
            "source_projection_status",
            "canary_ready_before_conversion",
        ]
        projection_rows, _ = read_csv(SOURCE_PROJECTION)
        projection_by_pair = {row["pair_id"]: row for row in projection_rows}
        audit_rows: list[dict[str, Any]] = []
        for index, item in enumerate(selected, start=1):
            source = projection_by_pair[item["pair_id"]]
            audit_rows.append(
                {
                    "selection_order": index,
                    "unit": item["unit"],
                    "pair_id": item["pair_id"],
                    "selection_tie_break_sha256": item["tie_break"],
                    **item["technical"],
                    "dti_raw_total_bytes": item["row"]["dti_raw_total_bytes"],
                    "t1_raw_total_bytes": item["row"]["t1_raw_total_bytes"],
                    "source_projection_status": source["workflow_projection_status"],
                    "canary_ready_before_conversion": source["canary_ready"],
                }
            )
        write_csv(
            temporary / "candidate_selection_audit_v1.csv", audit_rows, audit_fields
        )
        balance = balance_rows(candidates, selected)
        write_csv(
            temporary / "candidate_balance_summary_v1.csv",
            balance,
            (
                "feature",
                "value",
                "cohort_count",
                "selected_count",
                "target_count",
                "target_met",
            ),
        )

        # Diagnosis is accessed only after selected is fixed and is not passed
        # back to the selection objective.  This is an aggregate coverage audit.
        diagnosis_counts = collections.Counter(
            item["row"]["diagnosis_at_dti"] for item in selected
        )
        postselection = {
            "schema_version": "1.0.0",
            "record_type": "postselection_aggregate_group_coverage",
            "selection_was_locked_before_this_check": True,
            "diagnosis_labels_used_for_selection": False,
            "subset_manifest_sha256": subset_hash,
            "row_level_mapping_published": False,
            "aggregate_counts": dict(sorted(diagnosis_counts.items())),
            "subset_changed_after_check": False,
        }
        (temporary / "postselection_group_coverage_v1.json").write_text(
            json.dumps(postselection, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        conversion_stage = {
            "schema_version": "1.0.0",
            "record_type": "h04a_conversion_first_fail_closed_stage",
            "status": "PROPOSED_NOT_AUTHORIZED",
            "subset_manifest_sha256": subset_hash,
            "all_units_source_identity_ready": True,
            "all_units_conversion_pending": True,
            "phase_encoding_or_readout_guessed": False,
            "preconversion_authoritative_bids_pe_and_readout_count": 0,
            "selected_public_in_plane_pe_missing_count": sum(
                item["technical"]["pe_header"] == "FAIL_IN_PLANE_PE_MISSING"
                for item in selected
            ),
            "selected_public_gradient_present_count": sum(
                item["technical"]["gradient_public"] == "present"
                for item in selected
            ),
            "ordered_stages": [
                "runtime_source_identity_revalidation",
                "write_once_dwi_and_t1_normalization",
                "derive_normalized_metadata_from_same_conversion",
                "fail_unit_if_pe_readout_or_gradients_are_missing_or_inconsistent",
                "attempt_response_estimation_only_for_valid_units",
                "freeze_pool_only_if_minimum_12_shell_compatible_pass_units_include_2_manufacturer_families_and_both_t1_source_classes",
                "pre_tractography_processing_and_blinded_review_bundle",
                "terminal_stop_before_tractography",
            ],
            "minimum_valid_response_calibration_units_recommended": 12,
            "minimum_valid_technical_diversity": {
                "manufacturer_families": 2,
                "t1_source_classes": 2,
            },
            "no_default_no_imputation": True,
        }
        (temporary / "conversion_first_stage_v1.json").write_text(
            json.dumps(conversion_stage, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        runtime = runtime_evidence()
        storage = storage_evidence()
        evidence = {
            "schema_version": "1.0.0",
            "record_type": "h04a_runtime_storage_local_evidence",
            "runtime": runtime,
            "storage": storage,
        }
        (temporary / "runtime_storage_evidence_v1.json").write_text(
            json.dumps(evidence, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        estimate_rows = [
            {
                "stage": "H04A_conversion_and_response_phase_a",
                "authorization_gate": "SL-H04A",
                "units": 24,
                "max_concurrent_units": 4,
                "wall_clock_low_hours": 6,
                "wall_clock_high_hours": 24,
                "storage_reservation_gb": 50,
                "hard_stop_hours": 72,
                "hard_stop_storage_gb": 150,
                "status": "PROPOSED_NOT_AUTHORIZED",
                "basis": "legacy Eddy/prep/FOD timing plus v2 conversion and QC uncertainty",
            },
            {
                "stage": "H04A_pre_tractography_and_review",
                "authorization_gate": "SL-H04A",
                "units": 24,
                "max_concurrent_units": 4,
                "wall_clock_low_hours": 12,
                "wall_clock_high_hours": 36,
                "storage_reservation_gb": 100,
                "hard_stop_hours": 72,
                "hard_stop_storage_gb": 150,
                "status": "PROPOSED_NOT_AUTHORIZED",
                "basis": "legacy derivative footprint plus registration/FOD/tensor/atlas uncertainty",
            },
            {
                "stage": "H04B_primary_10M_tractography_SIFT2_matrices",
                "authorization_gate": "SL-H04B",
                "units": "only automated-and-human-QC-approved units",
                "max_concurrent_units": 4,
                "wall_clock_low_hours": "8_per_unit",
                "wall_clock_high_hours": "36_per_unit",
                "storage_reservation_gb": 450,
                "hard_stop_hours": "REQUIRES_H04B_DECISION",
                "hard_stop_storage_gb": "REQUIRES_H04B_DECISION",
                "status": "NOT_AUTHORIZED_BY_H04A",
                "basis": "legacy long-tailed track timing and 5 observed 10M TCK sizes",
            },
        ]
        write_csv(
            temporary / "runtime_storage_estimate_v1.csv",
            estimate_rows,
            tuple(estimate_rows[0]),
        )

        input_paths = {
            "parent_manifest": PARENT_MANIFEST,
            "source_projection": SOURCE_PROJECTION,
            "source_projection_validation": SOURCE_PROJECTION_VALIDATION,
            "content_inventory": CONTENT_INVENTORY,
            "recipe_config": RECIPE_CONFIG,
            "acquisition_schema": ACQUISITION_SCHEMA,
            "workflow_source_manifest": WORKFLOW_SOURCE_MANIFEST,
            "environment_contract": ENVIRONMENT_CONTRACT,
            "matrix_dictionary": MATRIX_DICTIONARY,
        }
        input_records = {name: file_record(path) for name, path in input_paths.items()}
        decision_draft = {
            "schema_version": "2.0.0",
            "decision_type": "connectome_canary_execution_subset_approval",
            "status": "PENDING_USER_DECISION",
            "approval_mode": "H04A_BOUNDED_CANARY",
            "approved_by": None,
            "approved_utc": None,
            "user_response": None,
            "execution_scope": "canary",
            "recipe_id": "connectome-v2.1.0-closed-world-canary-candidate",
            "parent_acquisition_manifest_sha256": input_records["parent_manifest"]["sha256"],
            "execution_subset_manifest_sha256": subset_hash,
            "approved_unit_count": 24,
            "diagnosis_labels_used": False,
            "selection_locked": True,
            "proposed_run_root": str(PROPOSED_RUN_ROOT),
            "authorized_modes": [
                "response-calibration-phase-a",
                "pre-tractography-canary",
            ],
            "minimum_valid_response_calibration_units": 12,
            "minimum_valid_manufacturer_families": 2,
            "minimum_valid_t1_source_classes": 2,
            "required_valid_t1_source_classes": [
                "dicom_series",
                "nifti_single",
            ],
            "maximum_cores": 4,
            "h04a_wall_clock_stop_hours": 72,
            "h04a_storage_stop_gb": 150,
            "authorized_through": "pre_tractography_review_bundle_only",
            "tractography_authorized": False,
            "matrix_generation_authorized": False,
            "full_cohort_authorized": False,
            "normative_config_sha256": input_records["recipe_config"]["sha256"],
            "workflow_source_manifest_sha256": input_records[
                "workflow_source_manifest"
            ]["sha256"],
            "environment_contract_sha256": input_records[
                "environment_contract"
            ]["sha256"],
            "note": (
                "This draft is intentionally not launcher-usable. A separate signed "
                "record must set APPROVED plus approved_by, approved_utc, and the exact "
                "user_response after explicit SL-H04A approval; all other fields remain "
                "byte-for-byte bound to this package."
            ),
        }
        (temporary / "h04a_decision_draft_v1.json").write_text(
            json.dumps(decision_draft, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        markdown = make_markdown(
            selected,
            subset_hash,
            input_records,
            diagnosis_counts,
            runtime,
            storage,
        )
        (temporary / "h04a_decision_package_v1.md").write_text(
            markdown, encoding="utf-8"
        )

        run_manifest = {
            "schema_version": "1.0.0",
            "record_type": "h04a_canary_package_build",
            "status": "PASS_NONEXECUTABLE_DECISION_PACKAGE",
            "builder": file_record(Path(__file__)),
            "selection_seed": SELECTION_SEED,
            "selection_size": 24,
            "selection_score": selection_score(selected),
            "selection_source_field_allowlist": sorted(SELECTION_SOURCE_FIELDS),
            "forbidden_selection_fields": sorted(FORBIDDEN_SELECTION_FIELDS),
            "diagnosis_labels_used_for_selection": False,
            "inputs": input_records,
            "candidate_subset_manifest": subset_record,
            "proposed_run_root": str(PROPOSED_RUN_ROOT),
            "imaging_or_workflow_commands_executed": False,
            "approved_execution_decision_created": False,
        }
        (temporary / "run_manifest.json").write_text(
            json.dumps(run_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        parent_by_unit = {stable_unit(row): row for row in parent_rows}
        exact_rows = all(
            item["row"] == parent_by_unit[item["unit"]] for item in selected
        )
        checks = {
            "exact_24_unique_rows": len(selected) == 24
            and len({item["unit"] for item in selected}) == 24,
            "all_subset_rows_bytefield_equal_parent_rows": exact_rows,
            "selection_allowlist_excludes_forbidden_fields": not (
                SELECTION_SOURCE_FIELDS & FORBIDDEN_SELECTION_FIELDS
            ),
            "diagnosis_not_in_selection_allowlist": "diagnosis_at_dti"
            not in SELECTION_SOURCE_FIELDS,
            "all_three_primary_diagnoses_present_postselection": all(
                diagnosis_counts.get(label, 0) > 0 for label in ("CN", "MCI", "AD")
            ),
            "all_selected_rows_conversion_pending": all(
                projection_by_pair[item["pair_id"]]["canary_ready"].lower()
                == "false"
                for item in selected
            ),
            "no_approved_decision_created": decision_draft["status"]
            == "PENDING_USER_DECISION",
            "decision_authorizes_only_h04a_pretractography_modes": (
                decision_draft["authorized_modes"]
                == [
                    "response-calibration-phase-a",
                    "pre-tractography-canary",
                ]
                and decision_draft["authorized_through"]
                == "pre_tractography_review_bundle_only"
                and decision_draft["tractography_authorized"] is False
                and decision_draft["matrix_generation_authorized"] is False
                and decision_draft["full_cohort_authorized"] is False
            ),
            "decision_resource_limits_are_exact": (
                decision_draft["maximum_cores"] == 4
                and decision_draft[
                    "minimum_valid_response_calibration_units"
                ]
                == 12
                and decision_draft["minimum_valid_manufacturer_families"] == 2
                and decision_draft["minimum_valid_t1_source_classes"] == 2
                and decision_draft["required_valid_t1_source_classes"]
                == ["dicom_series", "nifti_single"]
                and decision_draft["h04a_wall_clock_stop_hours"] == 72
                and decision_draft["h04a_storage_stop_gb"] == 150
            ),
            "decision_binds_current_recipe_source_environment": (
                decision_draft["normative_config_sha256"]
                == sha256_file(RECIPE_CONFIG)
                and decision_draft["workflow_source_manifest_sha256"]
                == sha256_file(WORKFLOW_SOURCE_MANIFEST)
                and decision_draft["environment_contract_sha256"]
                == sha256_file(ENVIRONMENT_CONTRACT)
            ),
            "proposed_run_root_does_not_exist": not PROPOSED_RUN_ROOT.exists(),
            "manufacturer_balance_exact": collections.Counter(
                item["technical"]["manufacturer_family"] for item in selected
            )
            == collections.Counter({"SIEMENS": 8, "GE": 8, "PHILIPS": 8}),
            "t1_source_balance_exact": collections.Counter(
                item["technical"]["t1_source_kind"] for item in selected
            )
            == collections.Counter({"dicom_series": 12, "nifti_single": 12}),
            "all_protocol_keys_represented": len(
                {item["technical"]["protocol"] for item in selected}
            )
            == len({item["technical"]["protocol"] for item in candidates}),
        }
        validation = {
            "schema_version": "1.0.0",
            "status": "PASS" if all(checks.values()) else "FAIL",
            "checks": checks,
            "counts": {
                "parent_rows": len(parent_rows),
                "selected_rows": len(selected),
                "selected_protocol_keys": len(
                    {item["technical"]["protocol"] for item in selected}
                ),
                "selected_sites": len(
                    {item["technical"]["site"] for item in selected}
                ),
                "selected_scanner_models": len(
                    {item["technical"]["scanner_model"] for item in selected}
                ),
                "postselection_diagnosis_counts": dict(
                    sorted(diagnosis_counts.items())
                ),
            },
            "candidate_subset_manifest_sha256": subset_hash,
            "execution_authorized": False,
        }
        (temporary / "release_validation.json").write_text(
            json.dumps(validation, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if validation["status"] != "PASS":
            raise AssertionError(json.dumps(checks, sort_keys=True))

        artifact_rows = []
        for path in sorted(temporary.iterdir()):
            if path.is_file():
                artifact_rows.append(
                    {
                        "relative_path": path.name,
                        "size_bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                    }
                )
        write_csv(
            temporary / "artifact_manifest.csv",
            artifact_rows,
            ("relative_path", "size_bytes", "sha256"),
        )
        temporary.replace(output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    build_package(args.output.resolve())


if __name__ == "__main__":
    main()
