#!/usr/bin/env python3
"""Audit historical DWI metadata proxies without reading voxel arrays.

The selected-ID derivatives under ``/data/derivatives/mif_dwi`` predate the
new fixed-data release.  Their MIF headers and gradient text files can expose
whether the current workflow preflight is likely to succeed, but they are not
authoritative evidence for the locked raw DICOM sources.  This audit therefore
labels every result as a historical proxy and leaves raw-source validation as a
required canary-preparation step.

No voxel array is loaded.  ``mrinfo -property`` reads MIF header properties;
NumPy reads only ``.bval``/``.bvec`` text; nibabel reads only NIfTI headers for
the available processed-T1 UID-carrier check.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd

from research_audit.run_fixed_data_feasibility import (
    DEFAULT_DECISION,
    DEFAULT_DIAGNOSIS,
    DEFAULT_MANIFEST,
    DEFAULT_MANIFEST_VALIDATION,
    sha256_file,
    validate_inputs,
)


PROJECT = Path("/home/ec2-user/exp")
DEFAULT_DERIVATIVE_ROOT = Path("/data/derivatives/mif_dwi")
DEFAULT_MRINFO = Path("/home/ec2-user/mrtrix3/bin/mrinfo")
DEFAULT_OUTPUT = PROJECT / "research_audit" / "outputs"
OUTPUT_ROWS = "historical_source_preflight_proxy_v1.csv"
OUTPUT_VALIDATION = "historical_source_preflight_proxy_validation_v1.json"
OUTPUT_SUMMARY = "historical_source_preflight_proxy_summary_v1.md"
UID_RE = re.compile(r"(?<!\d)(?:[0-9]+\.){3,}[0-9]+(?!\d)")


def _file_sha256(path: Path) -> str:
    return sha256_file(path)


def _gradient_arrays(bval_path: Path, bvec_path: Path) -> tuple[np.ndarray, np.ndarray]:
    bvals = np.atleast_1d(np.loadtxt(bval_path, dtype=float)).reshape(-1)
    bvecs = np.asarray(np.loadtxt(bvec_path, dtype=float))
    if bvecs.ndim == 1:
        if bvecs.size % 3:
            raise ValueError("1D bvec length is not divisible by 3")
        bvecs = bvecs.reshape(3, -1)
    elif bvecs.shape[0] == 3:
        pass
    elif bvecs.shape[1] == 3:
        bvecs = bvecs.T
    else:
        raise ValueError(f"bvec shape is not 3xN or Nx3: {bvecs.shape}")
    if bvals.size != bvecs.shape[1]:
        raise ValueError(f"bval/bvec length mismatch {bvals.size}/{bvecs.shape[1]}")
    if not np.isfinite(bvals).all() or not np.isfinite(bvecs).all():
        raise ValueError("non-finite gradient values")
    return bvals, bvecs


def antipodal_unique_directions(
    bvals: np.ndarray, bvecs: np.ndarray, *, b0_threshold: float = 50.0
) -> int:
    directions: list[tuple[float, float, float]] = []
    for vector in bvecs[:, bvals > b0_threshold].T:
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-8:
            continue
        unit = vector / norm
        for component in unit:
            if abs(float(component)) > 1e-8:
                if component < 0:
                    unit = -unit
                break
        directions.append(tuple(np.round(unit, 4)))
    return len(set(directions))


def gradient_proxy(bval_path: Path, bvec_path: Path) -> dict[str, Any]:
    bvals, bvecs = _gradient_arrays(bval_path, bvec_path)
    b0_mask = bvals <= 50
    norms = np.linalg.norm(bvecs, axis=0)
    nonzero_shells = sorted(
        {int(round(float(value) / 50.0) * 50) for value in bvals[~b0_mask]}
    )
    return {
        "n_bvals": int(bvals.size),
        "n_b0": int(b0_mask.sum()),
        "n_nonzero": int((~b0_mask).sum()),
        "unique_nonzero_directions_antipodal_round4": antipodal_unique_directions(
            bvals, bvecs
        ),
        "shells_round50": ";".join(str(value) for value in nonzero_shells),
        "n_nonzero_shells": len(nonzero_shells),
        "b0_vector_norm_gt_0_05_count": int((norms[b0_mask] > 0.05).sum()),
        "b0_vector_norm_max": float(norms[b0_mask].max()) if b0_mask.any() else None,
        "bval_sha256": _file_sha256(bval_path),
        "bvec_sha256": _file_sha256(bvec_path),
    }


def mrinfo_property(mrinfo: Path, image: Path, property_name: str) -> str:
    completed = subprocess.run(
        [str(mrinfo), str(image), "-property", property_name],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"mrinfo failed for {image} {property_name}: "
            f"{(completed.stderr or completed.stdout).strip()}"
        )
    return completed.stdout.strip()


def processed_t1_header_proxy(series_dir: Path) -> dict[str, Any]:
    files = sorted(path for path in series_dir.iterdir() if path.is_file())
    nifti = [path for path in files if path.name.lower().endswith((".nii", ".nii.gz"))]
    if len(nifti) != 1:
        return {
            "processed_t1_header_status": "FAIL_NOT_ONE_NIFTI",
            "processed_t1_nifti_path": "",
            "processed_t1_extensions": "",
            "processed_t1_aux_file": "",
            "processed_t1_description": "",
            "processed_t1_uid_text_found": "",
            "processed_t1_header_sha256": "",
        }
    image = nib.load(str(nifti[0]), mmap=True)
    header = image.header
    aux = bytes(header["aux_file"]).rstrip(b"\x00").decode("latin-1", errors="replace")
    description = bytes(header["descrip"]).rstrip(b"\x00").decode(
        "latin-1", errors="replace"
    )
    extension_payloads: list[bytes] = []
    for extension in header.extensions:
        try:
            content = extension.get_content()
            extension_payloads.append(
                content if isinstance(content, bytes) else str(content).encode("utf-8")
            )
        except Exception:
            extension_payloads.append(b"<UNREADABLE_EXTENSION>")
    header_payload = b"\0".join(
        [header.binaryblock, aux.encode(), description.encode(), *extension_payloads]
    )
    searchable = "\n".join(
        [aux, description]
        + [payload.decode("latin-1", errors="replace") for payload in extension_payloads]
    )
    return {
        "processed_t1_header_status": "PASS_HEADER_ONLY",
        "processed_t1_nifti_path": str(nifti[0]),
        "processed_t1_extensions": len(header.extensions),
        "processed_t1_aux_file": aux,
        "processed_t1_description": description,
        "processed_t1_uid_text_found": bool(UID_RE.search(searchable)),
        "processed_t1_header_sha256": hashlib.sha256(header_payload).hexdigest(),
    }


def build_proxy(
    frame: pd.DataFrame,
    derivative_root: Path,
    mrinfo: Path,
) -> pd.DataFrame:
    if not mrinfo.is_file():
        raise FileNotFoundError(f"mrinfo not found: {mrinfo}")
    rows: list[dict[str, Any]] = []
    for source in frame.itertuples(index=False):
        unit = f"{source.subject_id}_I{source.dti_image_id}"
        base = derivative_root / unit
        mif = base.with_suffix(".mif")
        bval = base.with_suffix(".bval")
        bvec = base.with_suffix(".bvec")
        for path in (mif, bval, bvec):
            if not path.is_file():
                raise FileNotFoundError(f"historical proxy input missing: {path}")
        phase_encoding = mrinfo_property(mrinfo, mif, "PhaseEncodingDirection")
        readout = mrinfo_property(mrinfo, mif, "TotalReadoutTime")
        gradient = gradient_proxy(bval, bvec)
        record: dict[str, Any] = {
            "evidence_role": "historical_selected_id_derivative_proxy_not_locked_raw_source",
            "subject_id": source.subject_id,
            "dti_image_id": source.dti_image_id,
            "current_t1_image_id": source.current_t1_image_id,
            "group": source.group,
            "primary_eligible": source.primary_eligible,
            "dti_manufacturer": source.dti_manufacturer,
            "dti_protocol_key": source.dti_protocol_key,
            "manifest_gradient_directions": source.dti_gradient_directions,
            "current_t1_type": source.current_t1_type,
            "historical_mif_path": str(mif),
            "historical_mif_size_bytes": mif.stat().st_size,
            "historical_mif_mtime_ns": mif.stat().st_mtime_ns,
            "phase_encoding_direction_proxy": phase_encoding,
            "total_readout_time_proxy": readout,
            "pe_and_readout_complete_proxy": bool(phase_encoding and readout),
            **gradient,
        }
        if source.current_t1_type == "Processed":
            record.update(processed_t1_header_proxy(Path(source.local_t1_series_dir)))
        else:
            record.update(
                {
                    "processed_t1_header_status": "NOT_APPLICABLE_ORIGINAL_T1",
                    "processed_t1_nifti_path": "",
                    "processed_t1_extensions": "",
                    "processed_t1_aux_file": "",
                    "processed_t1_description": "",
                    "processed_t1_uid_text_found": "",
                    "processed_t1_header_sha256": "",
                }
            )
        rows.append(record)
    return pd.DataFrame(rows)


def summary_markdown(
    rows: pd.DataFrame,
    generated_utc: str,
    manifest_sha256: str,
    mrinfo_version: str,
) -> str:
    complete = rows["pe_and_readout_complete_proxy"].astype(bool)
    low_directions = rows["unique_nonzero_directions_antipodal_round4"] < 30
    b0_issue = rows["b0_vector_norm_gt_0_05_count"] > 0
    multishell = rows["n_nonzero_shells"] > 1
    processed = rows["current_t1_type"].eq("Processed")
    missing_metadata = rows.loc[~complete, "dti_manufacturer"].value_counts().to_dict()
    low_by_manufacturer = rows.loc[low_directions, "dti_manufacturer"].value_counts().to_dict()
    lines = [
        "# Historical source-preflight proxy v1",
        "",
        f"**Generated:** {generated_utc}",
        "**Evidence role:** historical selected-ID derivative headers/gradient text; not locked raw-source evidence.",
        "**Safety:** no voxel array was loaded; no source or derivative was modified.",
        "",
        "## Results",
        "",
        f"- Historical MIF/.bval/.bvec sets present: {len(rows)}/{len(rows)}.",
        f"- Phase-encoding direction plus TotalReadoutTime present in historical MIF headers: {int(complete.sum())}/{len(rows)}; missing {int((~complete).sum())}.",
        f"- Missing PE/readout by manufacturer: `{json.dumps(missing_metadata, sort_keys=True)}`.",
        f"- Fewer than 30 antipodally canonicalized unique nonzero directions: {int(low_directions.sum())}/{len(rows)}; by manufacturer `{json.dumps(low_by_manufacturer, sort_keys=True)}`.",
        f"- More than one rounded nonzero shell: {int(multishell.sum())}/{len(rows)}. Shell combinations: `{json.dumps(rows.loc[multishell, 'shells_round50'].value_counts().to_dict(), sort_keys=True)}`.",
        f"- At least one b<=50 vector norm >0.05: {int(b0_issue.sum())}/{len(rows)}; by manufacturer `{json.dumps(rows.loc[b0_issue, 'dti_manufacturer'].value_counts().to_dict(), sort_keys=True)}`.",
        f"- Processed T1 NIfTI headers checked: {int(processed.sum())}; UID-like text found: {int(rows.loc[processed, 'processed_t1_uid_text_found'].astype(bool).sum())}; zero-extension headers: {int((pd.to_numeric(rows.loc[processed, 'processed_t1_extensions']) == 0).sum())}.",
        "",
        "## Interpretation boundary",
        "",
        "These observations are canary-preparation warnings. They do not prove that the locked raw DICOM sources contain or lack the same fields, and they do not authorize hardcoded phase/readout values. Raw-source header extraction, within-series consistency, exact conversion provenance, and gradient rotation/count checks remain mandatory.",
        "",
        "For 225 processed T1 inputs, the NIfTI header proxy found no UID-like text. The workflow contract must therefore allow a nullable `t1_dicom_series_uid` and require a stable `t1_source_id` based on the ADNI Image ID plus locked source-bundle hash. A UID must not be fabricated and these cases must not be discarded solely because an authoritative DICOM UID is unavailable.",
        "",
        "## Provenance",
        "",
        f"- Available-data manifest SHA-256: `{manifest_sha256}`.",
        f"- mrinfo: `{mrinfo_version}`.",
        "- MIF evidence is property output plus path/size/mtime, not a full MIF content hash; bval/bvec evidence uses exact file SHA-256.",
        "",
    ]
    return "\n".join(lines)


def run_proxy(
    manifest_path: Path,
    validation_path: Path,
    diagnosis_path: Path,
    decision_path: Path,
    derivative_root: Path,
    mrinfo: Path,
    output_dir: Path,
    generated_utc: str | None = None,
) -> dict[str, Any]:
    generated = generated_utc or datetime.now(timezone.utc).isoformat()
    frame, evidence = validate_inputs(
        manifest_path, validation_path, diagnosis_path, decision_path
    )
    target_paths = [
        output_dir / OUTPUT_ROWS,
        output_dir / OUTPUT_VALIDATION,
        output_dir / OUTPUT_SUMMARY,
    ]
    if any(path.exists() for path in target_paths):
        raise FileExistsError("refusing to overwrite historical preflight proxy release")
    completed = subprocess.run(
        [str(mrinfo), "-version"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    mrinfo_version = completed.stdout.strip().splitlines()[0]
    rows = build_proxy(frame, derivative_root, mrinfo)
    if len(rows) != 530 or rows["subject_id"].nunique() != 530:
        raise ValueError("historical proxy did not cover exactly 530 subjects")
    complete = rows["pe_and_readout_complete_proxy"].astype(bool)
    low_directions = rows["unique_nonzero_directions_antipodal_round4"] < 30
    b0_issue = rows["b0_vector_norm_gt_0_05_count"] > 0
    processed = rows["current_t1_type"].eq("Processed")
    if int(complete.sum()) != 380 or int(low_directions.sum()) != 10 or int(b0_issue.sum()) != 51:
        raise ValueError("historical proxy invariants changed; inspect before release")
    if int(processed.sum()) != 225:
        raise ValueError("processed T1 count changed")
    if rows.loc[processed, "processed_t1_uid_text_found"].astype(bool).any():
        raise ValueError("a processed T1 UID carrier was found; contract classification must be reviewed")

    output_dir.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".historical-preflight-", dir=output_dir))
    try:
        row_path = stage / OUTPUT_ROWS
        summary_path = stage / OUTPUT_SUMMARY
        rows.to_csv(row_path, index=False, lineterminator="\n")
        summary_path.write_text(
            summary_markdown(
                rows, generated, evidence["manifest"]["sha256"], mrinfo_version
            ),
            encoding="utf-8",
        )
        validation = {
            "schema_version": "1.0",
            "status": "PASS",
            "release_status": "HISTORICAL_PROXY_ONLY_RAW_SOURCE_VALIDATION_REQUIRED",
            "generated_utc": generated,
            "input_manifest": evidence["manifest"],
            "methods": {
                "mif_properties": "mrinfo -property PhaseEncodingDirection/TotalReadoutTime; header only",
                "gradient_unique_directions": "b>50; unit normalize; antipodal canonicalize; round 4 decimals",
                "shells": "nonzero b-values rounded to nearest 50",
                "b0_vector_flag": "any norm(bvec[:, b<=50]) > 0.05",
                "processed_t1_uid_carrier": "nibabel header/extension text only; UID regex; no dataobj access",
            },
            "mrinfo": {"path": str(mrinfo), "version": mrinfo_version},
            "counts": {
                "rows": len(rows),
                "pe_and_readout_complete": int(complete.sum()),
                "pe_or_readout_missing": int((~complete).sum()),
                "unique_directions_lt_30": int(low_directions.sum()),
                "multiple_nonzero_shells": int((rows["n_nonzero_shells"] > 1).sum()),
                "b0_vector_norm_issue": int(b0_issue.sum()),
                "processed_t1_headers": int(processed.sum()),
                "processed_t1_uid_text_found": int(
                    rows.loc[processed, "processed_t1_uid_text_found"].astype(bool).sum()
                ),
            },
            "outputs": {
                OUTPUT_ROWS: {
                    "sha256": _file_sha256(row_path),
                    "size_bytes": row_path.stat().st_size,
                    "rows": len(rows),
                },
                OUTPUT_SUMMARY: {
                    "sha256": _file_sha256(summary_path),
                    "size_bytes": summary_path.stat().st_size,
                },
            },
            "safety": {
                "raw_source_voxel_arrays_read": 0,
                "historical_derivative_voxel_arrays_read": 0,
                "historical_mif_headers_read": 530,
                "gradient_text_pairs_read": 530,
                "processed_t1_headers_read": 225,
                "files_modified_or_deleted": 0,
            },
            "limitations": [
                "Historical selected-ID derivative metadata may differ from locked raw-source headers.",
                "MIF files are bound by path/size/mtime plus extracted property values, not by full content hash, to avoid reading voxel payloads.",
                "Original-T1 and raw-DTI within-series UID/model consistency remain untested here.",
            ],
        }
        validation_path_out = stage / OUTPUT_VALIDATION
        validation_path_out.write_text(
            json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        for path in (row_path, summary_path, validation_path_out):
            os.replace(path, output_dir / path.name)
        return validation
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--manifest-validation", type=Path, default=DEFAULT_MANIFEST_VALIDATION)
    parser.add_argument("--diagnosis", type=Path, default=DEFAULT_DIAGNOSIS)
    parser.add_argument("--decision", type=Path, default=DEFAULT_DECISION)
    parser.add_argument("--derivative-root", type=Path, default=DEFAULT_DERIVATIVE_ROOT)
    parser.add_argument("--mrinfo", type=Path, default=DEFAULT_MRINFO)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_proxy(
        args.manifest,
        args.manifest_validation,
        args.diagnosis,
        args.decision,
        args.derivative_root,
        args.mrinfo,
        args.output_dir,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "release_status": result["release_status"],
                "validation": str((args.output_dir / OUTPUT_VALIDATION).resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
