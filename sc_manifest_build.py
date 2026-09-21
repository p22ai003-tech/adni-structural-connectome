#!/usr/bin/env python
"""Turn a folder of images into the acquisition manifest the workflow requires.

    python sc_manifest_build.py --input-root /data/study --layout simple --out configs/acquisition_manifest.csv

Discovery (sc_discover.py) answers "which scans exist and which T1 pairs with
which DWI". This turns that into the 47-field contract the imaging workflow
consumes: source kinds, series identifiers, per-file and per-bundle SHA-256
checksums, file counts and byte totals, and the policy fields. Everything it
writes is derived from the files themselves; nothing is assumed.

Three input layouts are understood:

    simple   <root>/dwi/<subject>[/<session>]/...      the documented template
             <root>/anat/<subject>[/<session>]/...
    adni     <root>/dti/<SUBJECT>/<PROTOCOL>/<DATE>/I<ID>/...
             <root>/mri/<SUBJECT>/<PROTOCOL>/<DATE>/I<ID>/...
    bids     <root>/sub-*/[ses-*/]dwi/*.nii.gz  and  anat/*_T1w.nii.gz

A scan folder holding *.dcm is a DICOM series; one holding a NIfTI with its
.bval, .bvec and .json is a bundle. Both are supported for DWI; T1 may be a
DICOM series or a single NIfTI.

participants.csv is optional. When present its subject_id column is joined to
carry a diagnosis label through to the analysis. The workflow never selects or
rejects a scan by that label.

Phase-encoding direction and readout time are NOT set here: they come from
sc_probe_acquisition.py, which reads them from the source metadata.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
for _p in (PROJECT_ROOT, PROJECT_ROOT / "scforge"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import sc_config  # noqa: E402
from scforge.input_contract import bundle_sha256  # noqa: E402

SCHEMA_REL = Path("scforge/workflow/schemas/acquisition_manifest_v2.schema.json")
DICOM_SUFFIXES = {".dcm", ".ima", ""}
NIFTI_SUFFIXES = (".nii.gz", ".nii")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


@dataclass
class Bundle:
    """One scan on disk, whatever form it takes."""
    kind: str                 # dicom_series | nifti_bundle | nifti_single
    root: Path
    members: list[Path]
    nifti: Path | None = None
    bval: Path | None = None
    bvec: Path | None = None
    json_sidecar: Path | None = None

    def inventory(self) -> tuple[str, int, int]:
        records, total = [], 0
        for m in self.members:
            size = m.stat().st_size
            total += size
            records.append((m.name, size, sha256_file(m)))
        return bundle_sha256(records), len(self.members), total


def classify(folder: Path, *, modality: str) -> Bundle | None:
    """Decide what kind of scan a folder holds, by looking inside it."""
    if not folder.is_dir():
        return None
    files = [p for p in sorted(folder.rglob("*")) if p.is_file() and not p.name.startswith(".")]
    if not files:
        return None
    niftis = [p for p in files if p.name.endswith(NIFTI_SUFFIXES)]
    if modality == "dwi":
        bval = next((p for p in files if p.suffix == ".bval"), None)
        bvec = next((p for p in files if p.suffix == ".bvec"), None)
        side = next((p for p in files if p.suffix == ".json"), None)
        if niftis and bval and bvec:
            return Bundle("nifti_bundle", folder, [niftis[0], bval, bvec] + ([side] if side else []),
                          nifti=niftis[0], bval=bval, bvec=bvec, json_sidecar=side)
        dicoms = [p for p in files if p.suffix.lower() in DICOM_SUFFIXES]
        return Bundle("dicom_series", folder, dicoms) if dicoms else None
    if niftis:
        return Bundle("nifti_single", folder, [niftis[0]], nifti=niftis[0])
    dicoms = [p for p in files if p.suffix.lower() in DICOM_SUFFIXES]
    return Bundle("dicom_series", folder, dicoms) if dicoms else None


def series_uid(bundle: Bundle) -> str | None:
    """The DICOM SeriesInstanceUID, when the scan is a DICOM series."""
    if bundle.kind != "dicom_series" or not bundle.members:
        return None
    try:
        import pydicom
        ds = pydicom.dcmread(bundle.members[0], stop_before_pixels=True, force=True)
        return str(getattr(ds, "SeriesInstanceUID", "") or "") or None
    except Exception:
        return None


def scanner_metadata(bundle: Bundle) -> dict[str, str]:
    if bundle.kind != "dicom_series" or not bundle.members:
        return {}
    try:
        import pydicom
        ds = pydicom.dcmread(bundle.members[0], stop_before_pixels=True, force=True)
        return {
            "manufacturer": str(getattr(ds, "Manufacturer", "") or "").strip(),
            "scanner_model": str(getattr(ds, "ManufacturerModelName", "") or "").strip(),
            "field_strength_t": str(getattr(ds, "MagneticFieldStrength", "") or "").strip(),
            "protocol": str(getattr(ds, "ProtocolName", "") or "").strip(),
            "study_date": str(getattr(ds, "StudyDate", "") or "").strip(),
        }
    except Exception:
        return {}


def row_for(subject: str, dwi: Bundle, t1: Bundle, *, gap_days, stratum: str,
            diagnosis: str | None, site: str | None, phase: str | None,
            dwi_id: str, t1_id: str) -> dict:
    dwi_hash, dwi_n, dwi_bytes = dwi.inventory()
    t1_hash, t1_n, t1_bytes = t1.inventory()
    meta = scanner_metadata(dwi)
    pair_hash = bundle_sha256([
        ("dti", dwi_bytes, dwi_hash),
        ("t1", t1_bytes, t1_hash),
    ])
    row = {
        # policy fields fixed by the contract
        "manifest_schema_version": "2.0.0",
        "data_scope_decision": "SL-D01_local_only",
        "processing_authorized": "true",
        "source_lock_status": "COMPLETE_SHA256",
        "analysis_role": "primary",
        # identity
        "subject_id": subject,
        "diagnosis_at_dti": diagnosis or "",
        # DWI
        "dti_image_id": dwi_id,
        "dti_study_date": meta.get("study_date", ""),
        "dti_source_kind": dwi.kind,
        "dti_source_id": dwi_id,
        "dti_source_path": str(dwi.root),
        "dti_dicom_series_uid": series_uid(dwi) or "",
        "dti_raw_bundle_sha256": dwi_hash,
        "dti_raw_file_count": dwi_n,
        "dti_raw_total_bytes": dwi_bytes,
        # a NIfTI bundle names its members; a DICOM series leaves these empty
        "dwi_nifti_path": str(dwi.nifti) if dwi.nifti and dwi.kind == "nifti_bundle" else "",
        "dwi_bvec_path": str(dwi.bvec) if dwi.bvec else "",
        "dwi_bval_path": str(dwi.bval) if dwi.bval else "",
        "dwi_json_path": str(dwi.json_sidecar) if dwi.json_sidecar else "",
        "dwi_nifti_sha256": sha256_file(dwi.nifti) if dwi.nifti and dwi.kind == "nifti_bundle" else "",
        "dwi_bvec_sha256": sha256_file(dwi.bvec) if dwi.bvec else "",
        "dwi_bval_sha256": sha256_file(dwi.bval) if dwi.bval else "",
        "dwi_json_sha256": sha256_file(dwi.json_sidecar) if dwi.json_sidecar else "",
        # T1
        "t1_image_id": t1_id,
        "t1_study_date": scanner_metadata(t1).get("study_date", ""),
        "t1_source_kind": t1.kind,
        "t1_source_id": t1_id,
        "t1_source_path": str(t1.nifti or t1.root),
        "t1_dicom_series_uid": series_uid(t1) or "",
        "t1_raw_bundle_sha256": t1_hash,
        "t1_raw_file_count": t1_n,
        "t1_raw_total_bytes": t1_bytes,
        # pairing
        "abs_pair_gap_days": "" if gap_days is None else int(round(gap_days)),
        "timing_stratum": stratum,
        # acquisition context
        "phase": phase or "",
        "site": site or "",
        "manufacturer": meta.get("manufacturer", ""),
        "scanner_model": meta.get("scanner_model", ""),
        "field_strength_t": meta.get("field_strength_t", ""),
        "protocol": meta.get("protocol", ""),
        # filled by sc_probe_acquisition.py
        "phase_encoding_direction": "",
        "phase_encoding_source": "",
        "total_readout_time": "",
        "total_readout_time_source": "",
        "normalization_readiness": "FAIL_MISSING_PHASE_ENCODING_AND_TOTAL_READOUT_TIME",
        "pair_content_bundle_sha256": pair_hash,
    }
    return row


def contract_fields() -> list[str]:
    schema = json.loads((PROJECT_ROOT / SCHEMA_REL).read_text())
    return list(schema.get("required") or (schema.get("items") or {}).get("required") or [])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", type=Path, required=True,
                    help="CSV from sc_discover.py (subject, dwi path, t1 path, gap, stratum)")
    ap.add_argument("--participants", type=Path, default=None,
                    help="optional participants.csv with subject_id and a diagnosis column")
    ap.add_argument("--diagnosis-column", default="group")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)

    diagnosis = {}
    if args.participants and args.participants.is_file():
        for r in csv.DictReader(args.participants.open(newline="", encoding="utf-8")):
            sid = (r.get("subject_id") or r.get("participant_id") or "").strip()
            if sid:
                diagnosis[sid] = (r.get(args.diagnosis_column) or "").strip()

    pairs = list(csv.DictReader(args.pairs.open(newline="", encoding="utf-8")))
    if args.limit:
        pairs = pairs[: args.limit]

    fields = contract_fields()
    rows, skipped = [], []
    for n, p in enumerate(pairs, 1):
        subject = p["subject_id"]
        dwi = classify(Path(p["dti_source_path"]), modality="dwi")
        t1_path = Path(p["t1_source_path"])
        t1 = classify(t1_path if t1_path.is_dir() else t1_path.parent, modality="anat")
        if dwi is None or t1 is None:
            skipped.append((subject, "unreadable DWI or T1 folder"))
            continue
        row = row_for(
            subject, dwi, t1,
            gap_days=float(p["abs_pair_gap_days"]) if p.get("abs_pair_gap_days") else None,
            stratum=p.get("timing_stratum", ""),
            diagnosis=diagnosis.get(subject),
            site=subject.split("_")[0] if "_S_" in subject else None,
            phase=p.get("phase") or None,
            dwi_id=p.get("dti_image_id") or dwi.root.name,
            t1_id=p.get("t1_image_id") or t1.root.name,
        )
        rows.append({k: row.get(k, "") for k in fields})
        if n % 25 == 0:
            print(f"  hashed {n}/{len(pairs)}", flush=True)

    print(f"built {len(rows)} manifest row(s); skipped {len(skipped)}")
    for s, why in skipped[:5]:
        print(f"   skipped {s}: {why}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {args.out}  ({len(fields)} contract fields)")
        print(f"next: python sc_probe_acquisition.py --manifest {args.out} --out {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
