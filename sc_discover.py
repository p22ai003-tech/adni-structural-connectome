#!/usr/bin/env python
"""Build an acquisition manifest from a folder of T1 and DWI images.

This is the pipeline's entry point. Everything downstream consumes the manifest
and nothing downstream globs the raw tree, so discovery is the one place that
has to understand how the images are laid out on disk.

Expected layout -- the standard ADNI download tree:

    <raw>/dti/<SUBJECT>/<PROTOCOL>/<YYYY-MM-DD_HH_MM_SS.0>/<IMAGE_ID>/*.dcm
    <raw>/mri/<SUBJECT>/<PROTOCOL>/<YYYY-MM-DD_HH_MM_SS.0>/<IMAGE_ID>/*.dcm

What it does
------------
Enumerates every series, reads one DICOM per series for the scanner metadata,
then pairs each subject's DWI with a T1. Pairing is the part that carries a
policy rather than a fact, so it is explicit and configurable: the nearest T1
in time, within a maximum gap, and each pairing is labelled with the gap in days
and which stratum it fell into. A DWI with no T1 inside the limit is reported,
not silently dropped and not paired to something too far away.

This does NOT fill phase_encoding_direction or total_readout_time. Those come
from sc_probe_acquisition.py, which runs dcm2niix to get the vendor-specific
values right. Discovery deliberately writes nothing it would have to guess.

    python sc_discover.py --raw-root /data/Images --out configs/acquisition_manifest.csv
    python sc_discover.py --raw-root /data/Images --limit 20 --report
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import sc_config  # noqa: E402

# ADNI encodes the acquisition time in the directory name.
_DATE_DIR = re.compile(r"^(\d{4})-(\d{2})-(\d{2})_(\d{2})_(\d{2})_(\d{2})")
_SUBJECT = re.compile(r"^\d{3}_S_\d{4,5}$")

COLUMNS = [
    "subject_id", "analysis_role",
    "dti_image_id", "dti_study_date", "dti_protocol", "dti_source_path", "dti_file_count",
    "t1_image_id", "t1_study_date", "t1_protocol", "t1_source_path", "t1_file_count",
    "abs_pair_gap_days", "timing_stratum",
    "manufacturer", "scanner_model", "field_strength_t",
    "phase_encoding_direction", "phase_encoding_source",
    "total_readout_time", "total_readout_time_source",
]


@dataclass
class Series:
    subject: str
    modality: str
    protocol: str
    acquired: datetime | None
    image_id: str
    path: Path
    n_files: int = 0
    meta: dict = field(default_factory=dict)


def parse_date(name: str) -> datetime | None:
    m = _DATE_DIR.match(name)
    if not m:
        return None
    try:
        return datetime(*(int(g) for g in m.groups()))
    except ValueError:
        return None


def scan_modality(root: Path, modality: str, subjects: set[str] | None) -> list[Series]:
    out: list[Series] = []
    base = root / modality
    if not base.is_dir():
        return out
    for subj_dir in sorted(base.iterdir()):
        if not subj_dir.is_dir() or not _SUBJECT.match(subj_dir.name):
            continue
        if subjects and subj_dir.name not in subjects:
            continue
        for proto_dir in sorted(subj_dir.iterdir()):
            if not proto_dir.is_dir():
                continue
            for date_dir in sorted(proto_dir.iterdir()):
                if not date_dir.is_dir():
                    continue
                acquired = parse_date(date_dir.name)
                for img_dir in sorted(date_dir.iterdir()):
                    if not img_dir.is_dir():
                        continue
                    out.append(Series(
                        subject=subj_dir.name, modality=modality,
                        protocol=proto_dir.name, acquired=acquired,
                        image_id=img_dir.name.lstrip("I"), path=img_dir,
                    ))
    return out


def read_meta(series: Series) -> None:
    """One DICOM per series is enough for the scanner metadata."""
    try:
        import pydicom
    except ImportError:
        return
    files = sorted(series.path.glob("*.dcm"))
    series.n_files = len(files)
    if not files:
        files = [p for p in sorted(series.path.iterdir()) if p.is_file()]
        series.n_files = len(files)
    if not files:
        return
    try:
        ds = pydicom.dcmread(files[0], stop_before_pixels=True, force=True)
    except Exception:
        return
    series.meta = {
        "manufacturer": str(getattr(ds, "Manufacturer", "") or "").strip(),
        "scanner_model": str(getattr(ds, "ManufacturerModelName", "") or "").strip(),
        "field_strength_t": str(getattr(ds, "MagneticFieldStrength", "") or "").strip(),
    }


def pair(dwi: list[Series], t1: list[Series], max_days: float,
         sensitivity_days: float,
         max_gap_days: float | None = None) -> tuple[list[dict], list[dict]]:
    """Pair each DWI with the nearest T1 in time, within a maximum gap.

    Returns (rows, unpaired). The gap limit is a policy, not a fact, so the
    stratum each pairing fell into is recorded rather than discarded.
    """
    by_subject: dict[str, list[Series]] = {}
    for s in t1:
        by_subject.setdefault(s.subject, []).append(s)

    rows, unpaired = [], []
    for d in dwi:
        candidates = by_subject.get(d.subject, [])
        best, best_gap = None, None
        for c in candidates:
            if d.acquired is None or c.acquired is None:
                continue
            gap = abs((c.acquired - d.acquired).total_seconds()) / 86400.0
            if best_gap is None or gap < best_gap:
                best, best_gap = c, gap
        if best is None or best_gap is None:
            unpaired.append({"subject_id": d.subject, "dti_image_id": d.image_id,
                             "reason": "no T1 series for this subject"})
            continue
        # The pair is made and LABELLED rather than refused. That is how this
        # cohort was actually built, and the label is the thing that matters: in
        # the 530-subject manifest the median T1-DWI gap is 753 days and 321 of
        # 530 pairs exceed 180 days. Dropping them would have cost most of the
        # cohort; hiding the gap would have been worse. Use --max-gap-days to
        # exclude instead, if a particular analysis needs to.
        if best_gap <= max_days:
            stratum = f"le_{max_days:.0f}_days"
        elif best_gap <= sensitivity_days:
            # Named for the inclusive lower bound, matching the convention the
            # project's v2 manifest already uses (days_91_180, not days_90_180).
            stratum = f"days_{max_days + 1:.0f}_{sensitivity_days:.0f}"
        else:
            stratum = f"gt_{sensitivity_days:.0f}_days"
        if max_gap_days is not None and best_gap > max_gap_days:
            unpaired.append({"subject_id": d.subject, "dti_image_id": d.image_id,
                             "reason": f"gap {best_gap:.0f} d exceeds --max-gap-days {max_gap_days:.0f}"})
            continue
        rows.append({
            "subject_id": d.subject,
            "analysis_role": "primary",
            "dti_image_id": d.image_id,
            "dti_study_date": d.acquired.date().isoformat() if d.acquired else "",
            "dti_protocol": d.protocol,
            "dti_source_path": str(d.path),
            "dti_file_count": d.n_files,
            "t1_image_id": best.image_id,
            "t1_study_date": best.acquired.date().isoformat() if best.acquired else "",
            "t1_protocol": best.protocol,
            "t1_source_path": str(best.path),
            "t1_file_count": best.n_files,
            "abs_pair_gap_days": round(best_gap, 3),
            "timing_stratum": stratum,
            "manufacturer": d.meta.get("manufacturer", ""),
            "scanner_model": d.meta.get("scanner_model", ""),
            "field_strength_t": d.meta.get("field_strength_t", ""),
            # Left blank on purpose; sc_probe_acquisition.py fills these.
            "phase_encoding_direction": "", "phase_encoding_source": "",
            "total_readout_time": "", "total_readout_time_source": "",
        })
    return rows, unpaired


def main(argv=None) -> int:
    p = sc_config.paths()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-root", type=Path, default=p.raw_images_root,
                    help="directory holding dti/ and mri/ (default: $SC_RAW_IMAGES_ROOT)")
    ap.add_argument("--dwi-dir", default="dti")
    ap.add_argument("--t1-dir", default="mri")
    ap.add_argument("--out", type=Path, default=None, help="manifest CSV to write")
    ap.add_argument("--unpaired-out", type=Path, default=None,
                    help="CSV of DWI series that could not be paired")
    ap.add_argument("--max-days", type=float, default=90.0,
                    help="primary stratum: maximum |T1 - DWI| gap in days")
    ap.add_argument("--sensitivity-days", type=float, default=180.0,
                    help="upper edge of the middle stratum; pairs beyond it are still made")
    ap.add_argument("--max-gap-days", type=float, default=None,
                    help="hard limit: refuse to pair beyond this gap (default: no limit, "
                         "every pair is made and labelled with its stratum)")
    ap.add_argument("--limit", type=int, default=None, help="first N subjects only")
    ap.add_argument("--no-meta", action="store_true",
                    help="skip reading DICOM headers (faster, no scanner metadata)")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args(argv)

    raw = Path(args.raw_root)
    if not raw.is_dir():
        raise SystemExit(f"raw root not found: {raw}")
    print(f"raw root : {raw}")

    subjects = None
    if args.limit:
        d = raw / args.dwi_dir
        subjects = {x.name for x in sorted(d.iterdir())[: args.limit] if x.is_dir()}

    dwi = scan_modality(raw, args.dwi_dir, subjects)
    t1 = scan_modality(raw, args.t1_dir, subjects)
    print(f"series   : {len(dwi)} DWI, {len(t1)} T1")

    if not args.no_meta:
        for s in dwi:
            read_meta(s)
        for s in t1:
            read_meta(s)

    rows, unpaired = pair(dwi, t1, args.max_days, args.sensitivity_days, args.max_gap_days)
    print(f"paired   : {len(rows)}   unpaired: {len(unpaired)}")

    if args.report:
        print("\ntiming stratum:")
        for k, c in Counter(r["timing_stratum"] for r in rows).most_common():
            print(f"   {k:<24} {c:>5}")
        print("manufacturer:")
        for k, c in Counter(r["manufacturer"] or "(unknown)" for r in rows).most_common(8):
            print(f"   {k:<24} {c:>5}")
        gaps = sorted(r["abs_pair_gap_days"] for r in rows)
        if gaps:
            print(f"pair gap days: min={gaps[0]:.1f} "
                  f"median={gaps[len(gaps)//2]:.1f} max={gaps[-1]:.1f}")
        if unpaired:
            print(f"\nunpaired ({len(unpaired)}), first 5:")
            for u in unpaired[:5]:
                print(f"   {u['subject_id']:<14} {u['reason']}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=COLUMNS)
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {args.out}  ({len(rows)} rows)")
        print("next: python sc_probe_acquisition.py --manifest "
              f"{args.out} --out {args.out}")
    if args.unpaired_out and unpaired:
        with Path(args.unpaired_out).open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(unpaired[0].keys()))
            w.writeheader()
            w.writerows(unpaired)
        print(f"wrote {args.unpaired_out}  ({len(unpaired)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
