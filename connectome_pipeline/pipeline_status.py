#!/usr/bin/env python3
"""
Small helpers to report pipeline stage counts.

Current scope:
- raw DWI MIFs           : derivatives/mif_dwi/*.mif
- denoised MIFs          : derivatives/mif_denoised/*_den.mif
- Gibbs-unringed MIFs    : derivatives/mif_unringed/*_den_unr.mif
- T1 preprocessing       : derivatives/t1_anat + derivatives/t1_fast
- DWI bias correction    : derivatives/biascorr_1
"""

from __future__ import annotations

import re
import csv
import os
import subprocess as sp
from functools import lru_cache
from html import escape
from pathlib import Path
from typing import Dict, Optional

from connectome_pipeline.pipeline_paths import resolve_pipeline_paths


DEFAULT_PATHS = resolve_pipeline_paths()
DEFAULT_DERIV_ROOT = DEFAULT_PATHS.deriv_root
DEFAULT_COHORT_DTI_CSV = DEFAULT_PATHS.cohort_dti_csv
SYNCED_STAGE_LOCATIONS_PATH = DEFAULT_PATHS.project_root / "synced_ec2_s3_locations.txt"
GROUP_ORDER = ("CN", "MCI", "AD")
GROUP_STAGE_ORDER = ("DICOM->MIF", "Denoise", "Gibbs", "Eddy")
ALL_STAGE_ORDER = (
    "DICOM->MIF",
    "Denoise",
    "Gibbs",
    "Eddy",
    "T1 Prep",
    "BiasCorr",
    "BBR",
    "Step7 Prep",
    "FOD",
    "Tracks",
    "Parc",
    "DTI",
    "Connectomes",
)
CANONICAL_DERIV_ROOT_DIRS = (
    "biascorr_1",
    "connectomes",
    "dti",
    "dwi_t1_bbr",
    "eddy",
    "fod",
    "mif_denoised",
    "mif_dwi",
    "mif_unringed",
    "parc",
    "qc",
    "t1_anat",
    "t1_fast",
    "tracks",
)
DEFAULT_S3_ROOT = os.environ.get("EXP_S3_ROOT", "s3://sabeesh/exp").rstrip("/")
STAGE_LOCATION_DIRS = {
    "DICOM->MIF": "mif_dwi",
    "Denoise": "mif_denoised",
    "Gibbs": "mif_unringed",
    "Eddy": "eddy",
    "T1 Prep": "t1_anat+t1_fast",
    "BiasCorr": "biascorr_1",
    "BBR": "dwi_t1_bbr",
    "Step7 Prep": "fod",
    "FOD": "fod",
    "Tracks": "tracks",
    "Parc": "parc",
    "DTI": "dti",
    "Connectomes": "connectomes",
}
STAGE_LOCATION_FOLDERS = {
    stage: tuple(folder.split("+"))
    for stage, folder in STAGE_LOCATION_DIRS.items()
}
STAGE_FOLDER_KEYS = tuple(
    dict.fromkeys(
        folder
        for folders in STAGE_LOCATION_FOLDERS.values()
        for folder in folders
    )
)


SERIES_RE = re.compile(r"\d{3}_S_\d{4}_I\d+")
S3_URI_RE = re.compile(r"^s3://(?P<bucket>[^/]+)(?:/(?P<prefix>.*))?$")


def _visible_glob(path: Path, pattern: str):
    for p in path.glob(pattern):
        if p.name.startswith("._") or p.name.startswith("."):
            continue
        if not p.exists():
            continue
        yield p


def _series_ids_from_raw(raw_dir: Path) -> set[str]:
    return {p.stem for p in _visible_glob(raw_dir, "*.mif")}


def _series_ids_from_denoised(den_dir: Path) -> set[str]:
    suffix = "_den.mif"
    return {p.name[: -len(suffix)] for p in _visible_glob(den_dir, f"*{suffix}")}


def _series_ids_from_unringed(unr_dir: Path) -> set[str]:
    suffix = "_den_unr.mif"
    return {p.name[: -len(suffix)] for p in _visible_glob(unr_dir, f"*{suffix}")}


def _count_visible(folder: Path, pattern: str) -> int:
    return sum(1 for _ in _visible_glob(folder, pattern)) if folder.exists() else 0


def _extract_suffix_ids(folder: Path, pattern: str, suffix: str) -> set[str]:
    return {p.name[: -len(suffix)] for p in _visible_glob(folder, pattern)}


def _extract_prefix_ids(folder: Path, pattern: str, prefix: str) -> set[str]:
    ids = set()
    for p in _visible_glob(folder, pattern):
        name = p.name
        if name.startswith(prefix):
            ids.add(name[len(prefix):].split(".nii.gz", 1)[0].split(".nii", 1)[0])
    return ids


def _t1_fast_ids(folder: Path, suffix: str) -> set[str]:
    ids = set()
    for p in _visible_glob(folder, f"*{suffix}"):
        if p.name.startswith("fast_"):
            ids.add(p.name[len("fast_") : -len(suffix)])
    return ids


def _series_ids_from_eddy(eddy_dir: Path) -> set[str]:
    ids = set()
    ids |= _extract_suffix_ids(eddy_dir, "*_preproc.mif", "_preproc.mif")
    ids |= _extract_suffix_ids(eddy_dir, "*_eddy.mif", "_eddy.mif")
    return ids


def _subject_ids_from_series(series_ids: set[str]) -> set[str]:
    return {sid.split("_I", 1)[0] for sid in series_ids if "_I" in sid}


def _subject_root_from_text(value: str) -> str:
    text = str(value or "")
    match = re.search(r"\d{3}_S_\d{4}", text)
    if match:
        return match.group(0)
    parts = text.split("_")
    return "_".join(parts[:3]) if len(parts) >= 3 else text


def _strip_prefix_suffix(name: str, prefix: str, suffix: str) -> Optional[str]:
    if not name.startswith(prefix) or not name.endswith(suffix):
        return None
    return name[len(prefix) : -len(suffix)]


def _t1_complete_subjects_from_names(anat_names: set[str], fast_names: set[str]) -> set[str]:
    t1_from_dicom = {
        base
        for name in anat_names
        if (base := _strip_prefix_suffix(name, "t1_from_dicom_", ".nii.gz")) is not None
    }
    corrected = {
        base
        for name in anat_names
        if (base := _strip_prefix_suffix(name, "corrected_T1_", ".nii.gz")) is not None
    }
    skull_stripped = {
        base
        for name in anat_names
        if (base := _strip_prefix_suffix(name, "T1_ss_", ".nii.gz")) is not None
    }
    brainmask = {
        base
        for name in anat_names
        if (base := _strip_prefix_suffix(name, "brainmask_", ".nii.gz")) is not None
    }
    fast_pve2 = {
        base
        for name in fast_names
        if (base := _strip_prefix_suffix(name, "fast_", "_pve_2.nii.gz")) is not None
    }
    fast_seg = {
        base
        for name in fast_names
        if (base := _strip_prefix_suffix(name, "fast_", "_seg.nii.gz")) is not None
    }
    complete_bases = t1_from_dicom & corrected & skull_stripped & brainmask & fast_pve2 & fast_seg
    return {_subject_root_from_text(base) for base in complete_bases}


def _bbr_ready_series_ids_from_names(names: set[str]) -> set[str]:
    b0mean = {
        sid
        for name in names
        if (sid := _strip_prefix_suffix(name, "", "_b0mean_ras.nii.gz")) is not None
    }
    t1_ras = {
        sid
        for name in names
        if (sid := _strip_prefix_suffix(name, "", "_t1_ras.nii.gz")) is not None
    }
    t1brain = {
        sid
        for name in names
        if (sid := _strip_prefix_suffix(name, "", "_t1brain_ras.nii.gz")) is not None
    }
    mat = {
        sid
        for name in names
        if (sid := _strip_prefix_suffix(name, "", "_t12b0_bbr.mat")) is not None
    }
    return b0mean & t1_ras & t1brain & mat


def _collect_step7_subject_sets_from_local(
    deriv_root: Path,
    *,
    select_streamlines: int = 3_000_000,
) -> Dict[str, set[str]]:
    prep_fixed: set[str] = set()
    prep_mask: set[str] = set()
    prep_gmwmi: set[str] = set()
    fod_final: set[str] = set()
    fod_mode: set[str] = set()
    tracks_ready: set[str] = set()
    parc_ready: set[str] = set()
    dti_fa: set[str] = set()
    dti_md: set[str] = set()
    dti_ad: set[str] = set()
    dti_rd: set[str] = set()
    connectomes_ready: set[str] = set()

    fod_dir = Path(deriv_root) / "fod"
    if fod_dir.exists():
        for sid_dir in fod_dir.iterdir():
            if sid_dir.name.startswith(".") or not sid_dir.is_dir():
                continue
            sid = sid_dir.name
            if (sid_dir / "5tt_b0_fixed.mif").exists():
                prep_fixed.add(sid)
            if (sid_dir / "mask_5tt_brain.mif").exists():
                prep_mask.add(sid)
            if (sid_dir / "gmwmi.mif").exists():
                prep_gmwmi.add(sid)
            if (sid_dir / "wmfod_final.mif").exists():
                fod_final.add(sid)
            if (sid_dir / "deconv_mode.txt").exists():
                fod_mode.add(sid)

    tracks_dir = Path(deriv_root) / "tracks"
    track_name = f"tracks_final_{select_streamlines // 1000}k.tck"
    if tracks_dir.exists():
        for sid_dir in tracks_dir.iterdir():
            if sid_dir.name.startswith(".") or not sid_dir.is_dir():
                continue
            if (sid_dir / track_name).exists():
                tracks_ready.add(sid_dir.name)

    parc_dir = Path(deriv_root) / "parc"
    if parc_dir.exists():
        for sid_dir in parc_dir.iterdir():
            if sid_dir.name.startswith(".") or not sid_dir.is_dir():
                continue
            if (sid_dir / "AAL_b0.nii.gz").exists():
                parc_ready.add(sid_dir.name)

    dti_dir = Path(deriv_root) / "dti"
    if dti_dir.exists():
        for sid_dir in dti_dir.iterdir():
            if sid_dir.name.startswith(".") or not sid_dir.is_dir():
                continue
            sid = sid_dir.name
            if (sid_dir / "fa.mif").exists():
                dti_fa.add(sid)
            if (sid_dir / "md.mif").exists():
                dti_md.add(sid)
            if (sid_dir / "ad.mif").exists():
                dti_ad.add(sid)
            if (sid_dir / "rd.mif").exists():
                dti_rd.add(sid)

    conn_dir = Path(deriv_root) / "connectomes"
    if conn_dir.exists():
        for path in conn_dir.glob("SC_AAL_*_ALL.csv"):
            if path.name.startswith("."):
                continue
            sid = path.name[len("SC_AAL_") : -len("_ALL.csv")]
            connectomes_ready.add(sid)

    return {
        "Step7 Prep": {_subject_root_from_text(sid) for sid in (prep_fixed & prep_mask & prep_gmwmi)},
        "FOD": {_subject_root_from_text(sid) for sid in (fod_final & fod_mode)},
        "Tracks": {_subject_root_from_text(sid) for sid in tracks_ready},
        "Parc": {_subject_root_from_text(sid) for sid in parc_ready},
        "DTI": {_subject_root_from_text(sid) for sid in (dti_fa & dti_md & dti_ad & dti_rd)},
        "Connectomes": {_subject_root_from_text(sid) for sid in connectomes_ready},
    }


def _collect_ec2_all_stage_inventory(
    deriv_root: Path,
    *,
    select_streamlines: int = 3_000_000,
) -> Dict[str, object]:
    deriv_root = Path(deriv_root)

    raw_series_ids = _series_ids_from_raw(deriv_root / "mif_dwi")
    den_series_ids = _series_ids_from_denoised(deriv_root / "mif_denoised")
    unr_series_ids = _series_ids_from_unringed(deriv_root / "mif_unringed")
    eddy_series_ids = _series_ids_from_eddy(deriv_root / "eddy")

    anat_names = {p.name for p in _visible_glob(deriv_root / "t1_anat", "*.nii.gz")}
    fast_names = {p.name for p in _visible_glob(deriv_root / "t1_fast", "*.nii.gz")}
    t1_ready_subjects = _t1_complete_subjects_from_names(anat_names, fast_names)

    bias_series_ids = _extract_suffix_ids(deriv_root / "biascorr_1", "*_unbiased.mif", "_unbiased.mif")
    bias_subjects = _subject_ids_from_series(bias_series_ids)

    bbr_names = {p.name for p in _visible_glob(deriv_root / "dwi_t1_bbr", "*")}
    bbr_ready_series_ids = _bbr_ready_series_ids_from_names(bbr_names)
    bbr_ready_subjects = _subject_ids_from_series(bbr_ready_series_ids)

    step7_sets = _collect_step7_subject_sets_from_local(
        deriv_root,
        select_streamlines=select_streamlines,
    )

    stage_subjects = {
        "DICOM->MIF": _subject_ids_from_series(raw_series_ids),
        "Denoise": _subject_ids_from_series(den_series_ids),
        "Gibbs": _subject_ids_from_series(unr_series_ids),
        "Eddy": _subject_ids_from_series(eddy_series_ids),
        "T1 Prep": t1_ready_subjects,
        "BiasCorr": bias_subjects,
        "BBR": bbr_ready_subjects,
        **step7_sets,
    }

    stage_series_counts = {
        "DICOM->MIF": len(raw_series_ids),
        "Denoise": len(den_series_ids),
        "Gibbs": len(unr_series_ids),
        "Eddy": len(eddy_series_ids),
        "T1 Prep": len(t1_ready_subjects),
        "BiasCorr": len(bias_series_ids),
        "BBR": len(bbr_ready_series_ids),
        "Step7 Prep": len(step7_sets["Step7 Prep"]),
        "FOD": len(step7_sets["FOD"]),
        "Tracks": len(step7_sets["Tracks"]),
        "Parc": len(step7_sets["Parc"]),
        "DTI": len(step7_sets["DTI"]),
        "Connectomes": len(step7_sets["Connectomes"]),
    }

    return {
        "raw_series_ids": raw_series_ids,
        "stage_subjects": stage_subjects,
        "stage_series_counts": stage_series_counts,
        "stage_artifact_counts": {
            "DICOM->MIF": _collect_stage_inventory(
                deriv_root / "mif_dwi",
                patterns=("*.mif",),
                extractor=_series_ids_from_raw,
            ),
            "Denoise": _collect_stage_inventory(
                deriv_root / "mif_denoised",
                patterns=("*_den.mif",),
                extractor=_series_ids_from_denoised,
            ),
            "Gibbs": _collect_stage_inventory(
                deriv_root / "mif_unringed",
                patterns=("*_den_unr.mif",),
                extractor=_series_ids_from_unringed,
            ),
            "Eddy": _collect_stage_inventory(
                deriv_root / "eddy",
                patterns=("*_preproc.mif", "*_eddy.mif"),
                extractor=_series_ids_from_eddy,
            ),
        },
    }


def _normalize_inventory_location(location: str) -> str:
    normalized = str(location or "ec2").strip().lower()
    if normalized not in {"ec2", "s3"}:
        raise ValueError("location must be 'ec2' or 's3'")
    return normalized


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    cleaned = str(uri or "").strip().rstrip("/")
    match = S3_URI_RE.match(cleaned)
    if not match:
        raise ValueError(f"Invalid S3 URI: {uri}")
    bucket = match.group("bucket") or ""
    prefix = (match.group("prefix") or "").strip("/")
    return bucket, prefix


def _with_trailing_slash(value: str) -> str:
    cleaned = str(value or "").strip()
    if not cleaned:
        return ""
    return cleaned if cleaned.endswith("/") else f"{cleaned}/"


@lru_cache(maxsize=1)
def _load_synced_stage_locations() -> Dict[str, Dict[str, str]]:
    if not SYNCED_STAGE_LOCATIONS_PATH.exists():
        return {}

    entries: Dict[str, Dict[str, str]] = {}
    current_name: Optional[str] = None
    current_locations: Dict[str, str] = {}

    with SYNCED_STAGE_LOCATIONS_PATH.open(encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                if current_name and current_locations:
                    entries[current_name] = dict(current_locations)
                current_name = None
                current_locations = {}
                continue

            if line.startswith((
                "Synced folder locations",
                "Generated from ",
                "EC2 root:",
                "S3 root:",
                "Note:",
            )):
                continue

            if line.startswith("EC2:"):
                current_locations["ec2"] = _with_trailing_slash(line.split(":", 1)[1].strip())
                continue
            if line.startswith("S3:"):
                current_locations["s3"] = _with_trailing_slash(line.split(":", 1)[1].strip())
                continue
            if ":" in line:
                continue

            if current_name and current_locations:
                entries[current_name] = dict(current_locations)
                current_locations = {}
            current_name = line

    if current_name and current_locations:
        entries[current_name] = dict(current_locations)

    return entries


def _format_project_relative(path: Path) -> str:
    path = Path(path).expanduser()
    try:
        absolute = path if path.is_absolute() else (Path.cwd() / path)
        relative = absolute.relative_to(DEFAULT_PATHS.project_root)
        return str(Path(DEFAULT_PATHS.project_root.name) / relative).replace("\\", "/") + ("/" if path.is_dir() else "")
    except Exception:
        return str(path)


def _build_folder_locations(
    *,
    deriv_root: Path,
    location: str,
    s3_root: str,
) -> Dict[str, str]:
    normalized = _normalize_inventory_location(location)
    if normalized == "ec2":
        locations = {
            folder: _with_trailing_slash(_format_project_relative(Path(deriv_root) / folder))
            for folder in STAGE_FOLDER_KEYS
        }
    else:
        s3_root = str(s3_root or DEFAULT_S3_ROOT).rstrip("/")
        locations = {
            folder: f"{s3_root}/data/derivatives/{folder}/"
            for folder in STAGE_FOLDER_KEYS
        }

    synced_locations = _load_synced_stage_locations()
    for folder in STAGE_FOLDER_KEYS:
        synced_value = synced_locations.get(folder, {}).get(normalized)
        if synced_value:
            locations[folder] = _with_trailing_slash(synced_value)
    return locations


def _build_stage_locations(
    *,
    deriv_root: Path,
    location: str,
    s3_root: str,
    overrides: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    _normalize_inventory_location(location)
    overrides = overrides or {}
    folder_locations = _build_folder_locations(
        deriv_root=deriv_root,
        location=location,
        s3_root=s3_root,
    )
    locations: Dict[str, str] = {}
    for stage, folders in STAGE_LOCATION_FOLDERS.items():
        locations[stage] = " + ".join(folder_locations[folder] for folder in folders)
    locations.update(overrides)
    return locations


def _run_aws_s3_ls_recursive(prefix_uri: str) -> list[str]:
    proc = sp.run(
        ["aws", "s3", "ls", prefix_uri, "--recursive"],
        stdout=sp.PIPE,
        stderr=sp.PIPE,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        raise RuntimeError(f"aws s3 ls failed for {prefix_uri}: {stderr or 'unknown error'}")

    keys: list[str] = []
    for raw_line in proc.stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = raw_line.split(maxsplit=3)
        if len(parts) != 4:
            continue
        key = parts[3].strip()
        if not key:
            continue
        name = Path(key).name
        if name.startswith(".") or name.startswith("._"):
            continue
        keys.append(key)
    return keys


def _safe_run_aws_s3_ls_recursive(prefix_uri: str) -> list[str]:
    try:
        return _run_aws_s3_ls_recursive(prefix_uri)
    except Exception:
        return []


def _relative_s3_keys(prefix_uri: str) -> list[str]:
    _, prefix = _parse_s3_uri(prefix_uri)
    base_prefix = f"{prefix.rstrip('/')}/" if prefix else ""
    relative_keys: list[str] = []
    for key in _safe_run_aws_s3_ls_recursive(prefix_uri):
        rel_key = key[len(base_prefix):] if base_prefix and key.startswith(base_prefix) else key
        rel_key = rel_key.lstrip("/")
        if rel_key:
            relative_keys.append(rel_key)
    return relative_keys


def _collect_s3_all_stage_inventory(
    *,
    s3_root: str,
    deriv_root: Path = DEFAULT_DERIV_ROOT,
    select_streamlines: int = 3_000_000,
) -> Dict[str, object]:
    folder_locations = _build_folder_locations(
        deriv_root=deriv_root,
        location="s3",
        s3_root=s3_root,
    )
    track_filename = f"tracks_final_{select_streamlines // 1000}k.tck"

    raw_series_ids: set[str] = set()
    den_series_ids: set[str] = set()
    unr_series_ids: set[str] = set()
    eddy_series_ids: set[str] = set()
    t1_anat_names: set[str] = set()
    t1_fast_names: set[str] = set()
    canonical_bias_series_ids: set[str] = set()
    canonical_bbr_names: set[str] = set()
    prep_fixed: set[str] = set()
    prep_mask: set[str] = set()
    prep_gmwmi: set[str] = set()
    fod_final: set[str] = set()
    fod_mode: set[str] = set()
    tracks_ready: set[str] = set()
    parc_ready: set[str] = set()
    dti_fa: set[str] = set()
    dti_md: set[str] = set()
    dti_ad: set[str] = set()
    dti_rd: set[str] = set()
    connectomes_ready: set[str] = set()

    for rel_key in _relative_s3_keys(folder_locations["mif_dwi"]):
        name = Path(rel_key).name
        if name.endswith(".mif"):
            raw_series_ids.add(name[:-len(".mif")])

    for rel_key in _relative_s3_keys(folder_locations["mif_denoised"]):
        name = Path(rel_key).name
        if name.endswith("_den.mif"):
            den_series_ids.add(name[:-len("_den.mif")])

    for rel_key in _relative_s3_keys(folder_locations["mif_unringed"]):
        name = Path(rel_key).name
        if name.endswith("_den_unr.mif"):
            unr_series_ids.add(name[:-len("_den_unr.mif")])

    for rel_key in _relative_s3_keys(folder_locations["eddy"]):
        name = Path(rel_key).name
        if name.endswith("_preproc.mif"):
            eddy_series_ids.add(name[:-len("_preproc.mif")])
        elif name.endswith("_eddy.mif"):
            eddy_series_ids.add(name[:-len("_eddy.mif")])

    for rel_key in _relative_s3_keys(folder_locations["t1_anat"]):
        name = Path(rel_key).name
        if name.endswith(".nii.gz"):
            t1_anat_names.add(name)

    for rel_key in _relative_s3_keys(folder_locations["t1_fast"]):
        name = Path(rel_key).name
        if name.endswith(".nii.gz"):
            t1_fast_names.add(name)

    for rel_key in _relative_s3_keys(folder_locations["biascorr_1"]):
        name = Path(rel_key).name
        if name.endswith("_unbiased.mif"):
            canonical_bias_series_ids.add(name[:-len("_unbiased.mif")])

    for rel_key in _relative_s3_keys(folder_locations["dwi_t1_bbr"]):
        canonical_bbr_names.add(Path(rel_key).name)

    for rel_key in _relative_s3_keys(folder_locations["fod"]):
        parts = Path(rel_key).parts
        if len(parts) < 2:
            continue
        sid = parts[0]
        name = parts[-1]
        if name == "5tt_b0_fixed.mif":
            prep_fixed.add(sid)
        elif name == "mask_5tt_brain.mif":
            prep_mask.add(sid)
        elif name == "gmwmi.mif":
            prep_gmwmi.add(sid)
        elif name == "wmfod_final.mif":
            fod_final.add(sid)
        elif name == "deconv_mode.txt":
            fod_mode.add(sid)

    for rel_key in _relative_s3_keys(folder_locations["tracks"]):
        parts = Path(rel_key).parts
        if len(parts) < 2:
            continue
        sid = parts[0]
        name = parts[-1]
        if name == track_filename:
            tracks_ready.add(sid)

    for rel_key in _relative_s3_keys(folder_locations["parc"]):
        parts = Path(rel_key).parts
        if len(parts) < 2:
            continue
        sid = parts[0]
        name = parts[-1]
        if name == "AAL_b0.nii.gz":
            parc_ready.add(sid)

    for rel_key in _relative_s3_keys(folder_locations["dti"]):
        parts = Path(rel_key).parts
        if len(parts) < 2:
            continue
        sid = parts[0]
        name = parts[-1]
        if name == "fa.mif":
            dti_fa.add(sid)
        elif name == "md.mif":
            dti_md.add(sid)
        elif name == "ad.mif":
            dti_ad.add(sid)
        elif name == "rd.mif":
            dti_rd.add(sid)

    for rel_key in _relative_s3_keys(folder_locations["connectomes"]):
        name = Path(rel_key).name
        if name.startswith("SC_AAL_") and name.endswith("_ALL.csv"):
            connectomes_ready.add(name[len("SC_AAL_") : -len("_ALL.csv")])

    bias_series_ids = canonical_bias_series_ids
    bbr_names = canonical_bbr_names

    prep_ready = {_subject_root_from_text(sid) for sid in (prep_fixed & prep_mask & prep_gmwmi)}
    fod_ready = {_subject_root_from_text(sid) for sid in (fod_final & fod_mode)}
    tracks_subjects = {_subject_root_from_text(sid) for sid in tracks_ready}
    parc_subjects = {_subject_root_from_text(sid) for sid in parc_ready}
    dti_subjects = {_subject_root_from_text(sid) for sid in (dti_fa & dti_md & dti_ad & dti_rd)}
    connectomes_subjects = {_subject_root_from_text(sid) for sid in connectomes_ready}
    t1_ready_subjects = _t1_complete_subjects_from_names(t1_anat_names, t1_fast_names)
    bias_subjects = _subject_ids_from_series(bias_series_ids)
    bbr_ready_series_ids = _bbr_ready_series_ids_from_names(bbr_names)
    bbr_ready_subjects = _subject_ids_from_series(bbr_ready_series_ids)
    stage_subjects = {
        "DICOM->MIF": _subject_ids_from_series(raw_series_ids),
        "Denoise": _subject_ids_from_series(den_series_ids),
        "Gibbs": _subject_ids_from_series(unr_series_ids),
        "Eddy": _subject_ids_from_series(eddy_series_ids),
        "T1 Prep": t1_ready_subjects,
        "BiasCorr": bias_subjects,
        "BBR": bbr_ready_subjects,
        "Step7 Prep": prep_ready,
        "FOD": fod_ready,
        "Tracks": tracks_subjects,
        "Parc": parc_subjects,
        "DTI": dti_subjects,
        "Connectomes": connectomes_subjects,
    }
    stage_series_counts = {
        "DICOM->MIF": len(raw_series_ids),
        "Denoise": len(den_series_ids),
        "Gibbs": len(unr_series_ids),
        "Eddy": len(eddy_series_ids),
        "T1 Prep": len(t1_ready_subjects),
        "BiasCorr": len(bias_series_ids),
        "BBR": len(bbr_ready_series_ids),
        "Step7 Prep": len(prep_ready),
        "FOD": len(fod_ready),
        "Tracks": len(tracks_subjects),
        "Parc": len(parc_subjects),
        "DTI": len(dti_subjects),
        "Connectomes": len(connectomes_subjects),
    }

    return {
        "raw_series_ids": raw_series_ids,
        "stage_subjects": stage_subjects,
        "stage_series_counts": stage_series_counts,
        "stage_artifact_counts": {
            "DICOM->MIF": {
                "matched_paths": len(raw_series_ids),
                "usable_existing_paths": len(raw_series_ids),
                "broken_symlink_paths": 0,
                "hidden_ignored_paths": 0,
                "usable_series_ids": len(raw_series_ids),
            },
            "Denoise": {
                "matched_paths": len(den_series_ids),
                "usable_existing_paths": len(den_series_ids),
                "broken_symlink_paths": 0,
                "hidden_ignored_paths": 0,
                "usable_series_ids": len(den_series_ids),
            },
            "Gibbs": {
                "matched_paths": len(unr_series_ids),
                "usable_existing_paths": len(unr_series_ids),
                "broken_symlink_paths": 0,
                "hidden_ignored_paths": 0,
                "usable_series_ids": len(unr_series_ids),
            },
            "Eddy": {
                "matched_paths": len(eddy_series_ids),
                "usable_existing_paths": len(eddy_series_ids),
                "broken_symlink_paths": 0,
                "hidden_ignored_paths": 0,
                "usable_series_ids": len(eddy_series_ids),
            },
        },
    }


def _resolve_cohort_dti_csv(cohort_dti_csv: Optional[Path] = None) -> Path:
    candidates = []
    if cohort_dti_csv is not None:
        candidates.append(Path(cohort_dti_csv))
    candidates.append(DEFAULT_COHORT_DTI_CSV)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    joined = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not find cohort dti csv. Tried: {joined}")


def _recode_group(value: str) -> Optional[str]:
    normalized = str(value or "").strip().upper()
    if not normalized:
        return None
    if normalized in {"CN", "NORMAL"}:
        return "CN"
    if normalized == "AD":
        return "AD"
    if normalized in {"MCI", "EMCI", "LMCI", "SMC"}:
        return "MCI"
    return None


def _normalize_group_filter(group: str) -> str:
    normalized = _recode_group(group)
    if normalized is not None:
        return normalized
    raw = str(group or "").strip().lower()
    if raw == "all":
        return "all"
    raise ValueError(f"Unsupported group filter: {group}")


def _load_subject_group_map(cohort_dti_csv: Optional[Path] = None) -> Dict[str, str]:
    resolved = _resolve_cohort_dti_csv(cohort_dti_csv)
    mapping: Dict[str, str] = {}
    with resolved.open(newline="", encoding="utf-8", errors="replace") as handle:
        for row in csv.DictReader(handle):
            subject_id = str(
                row.get("Subject ID")
                or row.get("subject_id")
                or row.get("subject")
                or ""
            ).strip()
            group = _recode_group(
                str(row.get("Research Group") or row.get("group") or "")
            )
            if subject_id and group:
                mapping[subject_id] = group
    return mapping


def _collect_stage_inventory(
    folder: Path,
    *,
    patterns: tuple[str, ...],
    extractor,
) -> Dict[str, int]:
    matched_paths = 0
    usable_existing_paths = 0
    broken_symlink_paths = 0
    hidden_ignored_paths = 0

    if folder.exists():
        for pattern in patterns:
            for path in folder.glob(pattern):
                if path.name.startswith("._") or path.name.startswith("."):
                    hidden_ignored_paths += 1
                    continue
                matched_paths += 1
                if path.is_symlink() and not path.exists():
                    broken_symlink_paths += 1
                    continue
                if path.exists():
                    usable_existing_paths += 1

    return {
        "matched_paths": matched_paths,
        "usable_existing_paths": usable_existing_paths,
        "broken_symlink_paths": broken_symlink_paths,
        "hidden_ignored_paths": hidden_ignored_paths,
        "usable_series_ids": len(extractor(folder)) if folder.exists() else 0,
    }


def collect_derivatives_root_diagnostics(deriv_root: Path = DEFAULT_DERIV_ROOT) -> Dict[str, object]:
    deriv_root = Path(deriv_root)
    canonical_dirs: list[str] = []
    stray_dirs: list[str] = []
    stray_files: list[str] = []
    broken_entries: list[str] = []

    if not deriv_root.exists():
        return {
            "deriv_root": str(deriv_root),
            "exists": False,
            "canonical_dirs": canonical_dirs,
            "stray_dirs": stray_dirs,
            "stray_files": stray_files,
            "broken_entries": broken_entries,
        }

    for entry in sorted(deriv_root.iterdir(), key=lambda p: p.name.lower()):
        if entry.name.startswith("."):
            continue
        if entry.is_symlink() and not entry.exists():
            broken_entries.append(entry.name)
            continue
        if entry.is_dir():
            if entry.name in CANONICAL_DERIV_ROOT_DIRS:
                canonical_dirs.append(entry.name)
            else:
                stray_dirs.append(entry.name)
        elif entry.is_file():
            stray_files.append(entry.name)
        else:
            stray_files.append(entry.name)

    return {
        "deriv_root": str(deriv_root),
        "exists": True,
        "canonical_dirs": canonical_dirs,
        "stray_dirs": stray_dirs,
        "stray_files": stray_files,
        "broken_entries": broken_entries,
    }


def print_derivatives_root_diagnostics(deriv_root: Path = DEFAULT_DERIV_ROOT) -> Dict[str, object]:
    diag = collect_derivatives_root_diagnostics(deriv_root)
    print("Derivatives root diagnostics")
    print("---------------------------")
    print(f"root: {diag['deriv_root']}")
    print(f"exists: {diag['exists']}")
    print(f"canonical dirs ({len(diag['canonical_dirs'])}): {', '.join(diag['canonical_dirs']) or '(none)'}")
    print(f"stray dirs ({len(diag['stray_dirs'])}): {', '.join(diag['stray_dirs']) or '(none)'}")
    print(f"stray files ({len(diag['stray_files'])}): {', '.join(diag['stray_files']) or '(none)'}")
    print(f"broken root entries ({len(diag['broken_entries'])}): {', '.join(diag['broken_entries']) or '(none)'}")
    return diag


def collect_pending_eddy_series(
    deriv_root: Path = DEFAULT_DERIV_ROOT,
    *,
    cohort_dti_csv: Optional[Path] = None,
    group: str = "all",
) -> list[str]:
    deriv_root = Path(deriv_root)
    normalized_group = _normalize_group_filter(group)
    unringed_ids = _series_ids_from_unringed(deriv_root / "mif_unringed")
    eddy_ids = _series_ids_from_eddy(deriv_root / "eddy")
    pending = sorted(unringed_ids - eddy_ids)
    if normalized_group == "all":
        return pending

    subject_group = _load_subject_group_map(cohort_dti_csv)
    return [
        series
        for series in pending
        if subject_group.get(series.split("_I", 1)[0]) == normalized_group
    ]


def collect_group_stage_status(
    deriv_root: Path = DEFAULT_DERIV_ROOT,
    *,
    cohort_dti_csv: Optional[Path] = None,
) -> Dict[str, object]:
    deriv_root = Path(deriv_root)
    resolved_csv = _resolve_cohort_dti_csv(cohort_dti_csv)
    subject_group = _load_subject_group_map(resolved_csv)

    raw_series_ids = _series_ids_from_raw(deriv_root / "mif_dwi")
    den_series_ids = _series_ids_from_denoised(deriv_root / "mif_denoised")
    unr_series_ids = _series_ids_from_unringed(deriv_root / "mif_unringed")
    eddy_series_ids = _series_ids_from_eddy(deriv_root / "eddy")

    raw_subjects = _subject_ids_from_series(raw_series_ids)
    den_subjects = _subject_ids_from_series(den_series_ids)
    unr_subjects = _subject_ids_from_series(unr_series_ids)
    eddy_subjects = _subject_ids_from_series(eddy_series_ids)

    stage_subjects = {
        "DICOM->MIF": raw_subjects,
        "Denoise": den_subjects,
        "Gibbs": unr_subjects,
        "Eddy": eddy_subjects,
    }

    base_subjects = sorted(raw_subjects)
    group_subjects = {
        group: {subject for subject in base_subjects if subject_group.get(subject) == group}
        for group in GROUP_ORDER
    }

    rows = []
    for group in GROUP_ORDER:
        total = len(group_subjects[group])
        stage_counts = {}
        for stage in GROUP_STAGE_ORDER:
            done = len(group_subjects[group] & stage_subjects[stage])
            stage_counts[stage] = {
                "done": done,
                "remaining": max(total - done, 0),
            }
        rows.append(
            {
                "group": group,
                "total": total,
                "stages": stage_counts,
            }
        )

    total_subjects = set().union(*group_subjects.values()) if group_subjects else set()
    total_row = {
        "group": "Total",
        "total": len(total_subjects),
        "stages": {
            stage: {
                "done": len(total_subjects & stage_subjects[stage]),
                "remaining": max(len(total_subjects) - len(total_subjects & stage_subjects[stage]), 0),
            }
            for stage in GROUP_STAGE_ORDER
        },
    }

    unmapped_subjects = sorted(subject for subject in base_subjects if subject not in subject_group)
    stage_artifact_counts = {
        "DICOM->MIF": _collect_stage_inventory(
            deriv_root / "mif_dwi",
            patterns=("*.mif",),
            extractor=_series_ids_from_raw,
        ),
        "Denoise": _collect_stage_inventory(
            deriv_root / "mif_denoised",
            patterns=("*_den.mif",),
            extractor=_series_ids_from_denoised,
        ),
        "Gibbs": _collect_stage_inventory(
            deriv_root / "mif_unringed",
            patterns=("*_den_unr.mif",),
            extractor=_series_ids_from_unringed,
        ),
        "Eddy": _collect_stage_inventory(
            deriv_root / "eddy",
            patterns=("*_preproc.mif", "*_eddy.mif"),
            extractor=_series_ids_from_eddy,
        ),
    }

    return {
        "cohort_dti_csv": str(resolved_csv),
        "deriv_root": str(deriv_root),
        "stage_order": list(GROUP_STAGE_ORDER),
        "rows": rows,
        "total_row": total_row,
        "unmapped_subjects": unmapped_subjects,
        "raw_subject_universe": len(base_subjects),
        "raw_series_universe": len(raw_series_ids),
        "stage_series_counts": {
            "DICOM->MIF": len(raw_series_ids),
            "Denoise": len(den_series_ids),
            "Gibbs": len(unr_series_ids),
            "Eddy": len(eddy_series_ids),
        },
        "stage_artifact_counts": stage_artifact_counts,
        "deriv_root_diagnostics": collect_derivatives_root_diagnostics(deriv_root),
        "missing_stage_labels": ["Prep", "FOD"],
    }


def print_group_stage_status(
    deriv_root: Path = DEFAULT_DERIV_ROOT,
    *,
    cohort_dti_csv: Optional[Path] = None,
) -> Dict[str, object]:
    status = collect_group_stage_status(deriv_root, cohort_dti_csv=cohort_dti_csv)
    headers = ["Group", "Total", *status["stage_order"]]
    widths = {
        "Group": 7,
        "Total": 5,
    }
    for stage in status["stage_order"]:
        widths[stage] = max(len(stage), 20)

    print("Pipeline group status (subject counts)")
    print("--------------------------------------")
    print(
        "  ".join(
            f"{header:<{widths.get(header, len(header))}}"
            for header in headers
        )
    )
    for row in [*status["rows"], status["total_row"]]:
        parts = [
            f"{row['group']:<{widths['Group']}}",
            f"{row['total']:<{widths['Total']}}",
        ]
        for stage in status["stage_order"]:
            cell = row["stages"][stage]
            text = f"{cell['done']} present"
            parts.append(f"{text:<{widths[stage]}}")
        print("  ".join(parts))
    if status["unmapped_subjects"]:
        joined = ", ".join(status["unmapped_subjects"][:10])
        print(f"unmapped raw-mif subjects: {joined}")
    noteworthy = []
    for stage, counts in status.get("stage_artifact_counts", {}).items():
        broken = int(counts.get("broken_symlink_paths", 0))
        if broken:
            noteworthy.append(f"{stage} broken symlinks ignored: {broken}")
    if noteworthy:
        for line in noteworthy:
            print(line)
    return status


def display_group_stage_status(
    deriv_root: Path = DEFAULT_DERIV_ROOT,
    *,
    cohort_dti_csv: Optional[Path] = None,
    title: str = "Pipeline group status (pre-Eddy stages)",
):
    status = collect_group_stage_status(deriv_root, cohort_dti_csv=cohort_dti_csv)
    try:
        from IPython.display import HTML, display
    except Exception:
        print_group_stage_status(deriv_root, cohort_dti_csv=cohort_dti_csv)
        return status

    table_rows = [*status["rows"], status["total_row"]]
    html_parts = [
        "<div style=\"font-family: Avenir Next, Segoe UI, Helvetica, Arial, sans-serif; "
        "background:#111; color:#f5f5f5; border-radius:16px; padding:18px 18px 14px 18px;\">",
        f"<div style=\"font-size:20px; font-weight:800; margin-bottom:4px;\">{escape(title)}</div>",
        "<div style=\"font-size:12px; color:#c8c8c8; margin-bottom:12px;\">"
        "Showing subject counts over the current raw DICOM->MIF universe available on this EC2 dataset.</div>",
        "<table style=\"border-collapse:collapse; width:100%; font-size:13px;\">",
        "<thead><tr>",
        "<th style=\"text-align:left; padding:10px 12px; background:#2a2a2a;\">Group</th>",
        "<th style=\"text-align:left; padding:10px 12px; background:#2a2a2a;\">Total</th>",
    ]
    for stage in status["stage_order"]:
        html_parts.append(
            f"<th style=\"text-align:left; padding:10px 12px; background:#2a2a2a;\">{escape(stage)}</th>"
        )
    html_parts.append("</tr></thead><tbody>")

    for row in table_rows:
        group_weight = "800" if row["group"] == "Total" else "700"
        html_parts.append("<tr>")
        html_parts.append(
            f"<td style=\"padding:10px 12px; border-top:1px solid #2a2a2a; font-weight:{group_weight};\">"
            f"{escape(str(row['group']))}</td>"
        )
        html_parts.append(
            f"<td style=\"padding:10px 12px; border-top:1px solid #2a2a2a; font-weight:{group_weight};\">"
            f"{row['total']}</td>"
        )
        for stage in status["stage_order"]:
            cell = row["stages"][stage]
            html_parts.append(
                "<td style=\"padding:10px 12px; border-top:1px solid #2a2a2a;\">"
                f"<span style=\"font-weight:700;\">{cell['done']} present</span></td>"
            )
        html_parts.append("</tr>")

    footer_bits = [
        f"Source: {escape(status['cohort_dti_csv'])}",
        f"Derivatives root: {escape(status['deriv_root'])}",
        f"Universe subjects: {status['raw_subject_universe']}",
        f"Raw DICOM->MIF series: {status['raw_series_universe']}",
    ]
    noteworthy = []
    for stage, counts in status.get("stage_artifact_counts", {}).items():
        broken = int(counts.get("broken_symlink_paths", 0))
        if broken:
            noteworthy.append(f"{stage} broken symlinks ignored: {broken}")
    stray_files = status.get("deriv_root_diagnostics", {}).get("stray_files", [])
    if stray_files:
        noteworthy.append(f"Root stray files: {len(stray_files)}")
    if status.get("missing_stage_labels"):
        footer_bits.append(
            "Not available on this EC2 dataset: " +
            ", ".join(escape(label) for label in status["missing_stage_labels"])
        )
    footer_bits.extend(noteworthy)

    html_parts.extend(
        [
            "</tbody></table>",
            "<div style=\"font-size:12px; color:#c8c8c8; margin-top:12px; line-height:1.45;\">"
            + " | ".join(footer_bits)
            + "</div>",
            "</div>",
        ]
    )
    display(HTML("".join(html_parts)))
    return status


def collect_all_stage_group_status(
    deriv_root: Path = DEFAULT_DERIV_ROOT,
    *,
    cohort_dti_csv: Optional[Path] = None,
    aal_mni: Optional[Path] = None,
    scope: str = "full",
    location: str = "ec2",
    s3_root: str = DEFAULT_S3_ROOT,
) -> Dict[str, object]:
    deriv_root = Path(deriv_root)
    resolved_csv = _resolve_cohort_dti_csv(cohort_dti_csv)
    resolved_aal = Path(aal_mni) if aal_mni is not None else DEFAULT_PATHS.aal_mni
    normalized_location = _normalize_inventory_location(location)
    subject_group = _load_subject_group_map(resolved_csv)

    if normalized_location == "s3":
        inventory = _collect_s3_all_stage_inventory(s3_root=s3_root, deriv_root=deriv_root)
        deriv_root_diagnostics = {
            "deriv_root": f"{str(s3_root).rstrip('/')}/",
            "exists": True,
            "canonical_dirs": sorted({folder for folder in STAGE_LOCATION_DIRS.values() if "+" not in folder}),
            "stray_dirs": [],
            "stray_files": [],
            "broken_entries": [],
        }
    else:
        inventory = _collect_ec2_all_stage_inventory(deriv_root)
        deriv_root_diagnostics = collect_derivatives_root_diagnostics(deriv_root)

    stage_locations = _build_stage_locations(
        deriv_root=deriv_root,
        location=normalized_location,
        s3_root=s3_root,
        overrides=inventory.get("stage_location_overrides", {}),
    )

    raw_series_ids = set(inventory["raw_series_ids"])
    stage_subjects = {
        stage: set(inventory["stage_subjects"].get(stage, set()))
        for stage in ALL_STAGE_ORDER
    }
    base_subjects = sorted(stage_subjects["DICOM->MIF"])
    group_subjects = {
        group: {subject for subject in base_subjects if subject_group.get(subject) == group}
        for group in GROUP_ORDER
    }

    ordered_rows = []
    for group in GROUP_ORDER:
        total = len(group_subjects[group])
        ordered_rows.append(
            {
                "group": group,
                "total": total,
                "stages": {
                    stage: {
                        "done": len(group_subjects[group] & stage_subjects[stage]),
                        "remaining": max(total - len(group_subjects[group] & stage_subjects[stage]), 0),
                    }
                    for stage in ALL_STAGE_ORDER
                },
            }
        )

    total_subjects = sum(int(row["total"]) for row in ordered_rows)
    total_row = {
        "group": "Total",
        "total": total_subjects,
        "stages": {
            stage: {
                "done": sum(int(row["stages"][stage]["done"]) for row in ordered_rows),
                "remaining": max(
                    total_subjects - sum(int(row["stages"][stage]["done"]) for row in ordered_rows),
                    0,
                ),
            }
            for stage in ALL_STAGE_ORDER
        },
    }

    unmapped_subjects = sorted(subject for subject in base_subjects if subject not in subject_group)

    return {
        "cohort_dti_csv": str(resolved_csv),
        "aal_mni": str(resolved_aal),
        "deriv_root": str(deriv_root),
        "inventory_location": normalized_location,
        "s3_root": str(s3_root).rstrip("/"),
        "stage_locations": stage_locations,
        "stage_order": list(ALL_STAGE_ORDER),
        "rows": ordered_rows,
        "total_row": total_row,
        "unmapped_subjects": unmapped_subjects,
        "raw_subject_universe": len(base_subjects),
        "raw_series_universe": len(raw_series_ids),
        "series_in_current_view": len(raw_series_ids),
        "stage_series_counts": dict(inventory.get("stage_series_counts", {})),
        "stage_artifact_counts": dict(inventory.get("stage_artifact_counts", {})),
        "deriv_root_diagnostics": deriv_root_diagnostics,
        "step7_error": None,
    }


def display_all_stage_group_status(
    deriv_root: Path = DEFAULT_DERIV_ROOT,
    *,
    cohort_dti_csv: Optional[Path] = None,
    aal_mni: Optional[Path] = None,
    scope: str = "full",
    location: str = "ec2",
    s3_root: str = DEFAULT_S3_ROOT,
    return_status: bool = True,
    title: str = "Pipeline group status (All stages)",
) -> Optional[Dict[str, object]]:
    status = collect_all_stage_group_status(
        deriv_root,
        cohort_dti_csv=cohort_dti_csv,
        aal_mni=aal_mni,
        scope=scope,
        location=location,
        s3_root=s3_root,
    )
    try:
        from IPython.display import HTML, display
    except Exception:
        print_group_stage_status(deriv_root, cohort_dti_csv=cohort_dti_csv)
        return status

    table_rows = [*status["rows"], status["total_row"]]
    html_parts = [
        "<div style=\"margin:12px 0 18px 0;padding:14px 16px;border:1px solid #333;border-radius:10px;background:#111;color:#f5f5f5;\">",
        f"<div style=\"font-weight:700;font-size:14px;margin-bottom:8px;\">{escape(title)}</div>",
        "<div style=\"font-size:12px;color:#c8c8c8;margin-bottom:10px;\">"
        "Showing counts over the full raw DICOM-&gt;MIF universe.</div>",
        "<table style=\"border-collapse:collapse;width:100%;font-size:12px;\">",
        "<thead><tr>",
        "<th style=\"padding:7px 10px;text-align:left;border-bottom:1px solid #333;background:#2a2a2a;\">Group</th>",
        "<th style=\"padding:7px 10px;text-align:left;border-bottom:1px solid #333;background:#2a2a2a;\">Total</th>",
    ]
    for stage in status["stage_order"]:
        html_parts.append(
            f"<th style=\"padding:7px 10px;text-align:left;border-bottom:1px solid #333;background:#2a2a2a;\">{escape(stage)}</th>"
        )
    html_parts.append("</tr></thead><tbody>")

    html_parts.append("<tr>")
    html_parts.append(
        "<td style=\"padding:8px 10px;border-bottom:1px solid #222;background:#151515;font-weight:700;\">Location</td>"
    )
    html_parts.append(
        f"<td style=\"padding:8px 10px;border-bottom:1px solid #222;background:#151515;font-weight:700;\">"
        f"{escape(str(status.get('inventory_location', 'ec2')).upper())}</td>"
    )
    for stage in status["stage_order"]:
        html_parts.append(
            "<td style=\"padding:8px 10px;border-bottom:1px solid #222;background:#151515;"
            "font-size:11px;color:#c8c8c8;white-space:normal;word-break:break-word;\">"
            f"{escape(str(status.get('stage_locations', {}).get(stage, '')))}"
            "</td>"
        )
    html_parts.append("</tr>")

    for row in table_rows:
        is_total = str(row["group"]).lower() == "total"
        row_bg = "#1a1a1a" if is_total else "transparent"
        row_border = "2px solid #555" if is_total else "1px solid #222"
        row_weight = "700" if is_total else "400"
        html_parts.append("<tr>")
        html_parts.append(
            f"<td style=\"padding:8px 10px;border-bottom:{row_border};background:{row_bg};font-weight:{row_weight};\">"
            f"{escape(str(row['group']))}</td>"
        )
        html_parts.append(
            f"<td style=\"padding:8px 10px;border-bottom:{row_border};background:{row_bg};font-weight:{row_weight};\">"
            f"{row['total']}</td>"
        )
        for stage in status["stage_order"]:
            cell = row["stages"][stage]
            html_parts.append(
                f"<td style=\"padding:8px 10px;border-bottom:{row_border};background:{row_bg};\">"
                f"<span style=\"font-weight:700;\">{cell['done']} present</span>"
                "</td>"
            )
        html_parts.append("</tr>")

    footer_bits = [
        f"Source: {escape(status['cohort_dti_csv'])}",
        f"Inventory location: {escape(str(status.get('inventory_location', 'ec2')).upper())}",
        "Universe: full raw DICOM-&gt;MIF universe",
        f"series in current view: {status['series_in_current_view']}",
        f"global raw DICOM-&gt;MIF series: {status['raw_series_universe']}",
    ]
    gibbs_broken = int(status.get("stage_artifact_counts", {}).get("Gibbs", {}).get("broken_symlink_paths", 0))
    if gibbs_broken:
        footer_bits.append(f"Gibbs broken symlinks ignored: {gibbs_broken}")
    if status.get("step7_error"):
        footer_bits.append(f"Later-stage quick scan unavailable: {escape(str(status['step7_error']))}")

    html_parts.extend(
        [
            "</tbody></table>",
            "<div style=\"font-size:12px;color:#c8c8c8;margin-top:10px;line-height:1.45;\">"
            + " | ".join(footer_bits)
            + "</div>",
            "</div>",
        ]
    )
    display(HTML("".join(html_parts)))
    return status if return_status else None


def collect_preproc_status(deriv_root: Path = DEFAULT_DERIV_ROOT) -> Dict[str, object]:
    deriv_root = Path(deriv_root)
    raw_dir = deriv_root / "mif_dwi"
    den_dir = deriv_root / "mif_denoised"
    unr_dir = deriv_root / "mif_unringed"

    raw_ids = _series_ids_from_raw(raw_dir) if raw_dir.exists() else set()
    den_ids = _series_ids_from_denoised(den_dir) if den_dir.exists() else set()
    unr_ids = _series_ids_from_unringed(unr_dir) if unr_dir.exists() else set()

    noise_count = _count_visible(den_dir, "*_noise.mif")

    missing_denoised = sorted(raw_ids - den_ids)
    missing_unringed = sorted(den_ids - unr_ids)

    return {
        "raw_count": len(raw_ids),
        "denoised_count": len(den_ids),
        "noise_count": noise_count,
        "unringed_count": len(unr_ids),
        "missing_denoised_count": len(missing_denoised),
        "missing_unringed_count": len(missing_unringed),
        "missing_denoised_examples": missing_denoised[:10],
        "missing_unringed_examples": missing_unringed[:10],
    }


def collect_denoise_status(deriv_root: Path = DEFAULT_DERIV_ROOT) -> Dict[str, object]:
    status = collect_preproc_status(deriv_root)
    return {
        "raw_count": status["raw_count"],
        "denoised_count": status["denoised_count"],
        "noise_count": status["noise_count"],
        "missing_denoised_count": status["missing_denoised_count"],
        "missing_denoised_examples": status["missing_denoised_examples"],
    }


def print_denoise_status(deriv_root: Path = DEFAULT_DERIV_ROOT) -> Dict[str, object]:
    status = collect_denoise_status(deriv_root)

    print("Denoise status")
    print("--------------")
    print(f"raw mif files  : {status['raw_count']}")
    print(f"denoised       : {status['denoised_count']}")
    print(f"noise maps     : {status['noise_count']}")
    print(f"missing denoise: {status['missing_denoised_count']}")

    if status["missing_denoised_examples"]:
        joined = ", ".join(status["missing_denoised_examples"])
        print(f"examples missing denoise: {joined}")

    return status


def collect_eddy_status(deriv_root: Path = DEFAULT_DERIV_ROOT) -> Dict[str, object]:
    deriv_root = Path(deriv_root)
    unr_dir = deriv_root / "mif_unringed"
    eddy_dir = deriv_root / "eddy"
    quarantine_csv = deriv_root / "qc" / "eddy_quarantine.csv"

    unringed_ids = _series_ids_from_unringed(unr_dir) if unr_dir.exists() else set()
    eddy_ids = _series_ids_from_eddy(eddy_dir) if eddy_dir.exists() else set()
    missing_eddy = sorted(unringed_ids - eddy_ids)
    quarantine_rows = []
    if quarantine_csv.exists():
        with quarantine_csv.open(newline="") as fp:
            quarantine_rows = [
                row
                for row in csv.DictReader(fp)
                if (row.get("series") or "").strip()
            ]

    return {
        "unringed_count": len(unringed_ids),
        "eddy_count": len(eddy_ids),
        "missing_eddy_count": len(missing_eddy),
        "missing_eddy_examples": missing_eddy[:10],
        "quarantine_count": len(quarantine_rows),
        "quarantine_examples": [
            f"{row.get('series', '')}:{row.get('failure_class', '')}"
            for row in quarantine_rows[:10]
        ],
    }


def print_eddy_status(deriv_root: Path = DEFAULT_DERIV_ROOT) -> Dict[str, object]:
    status = collect_eddy_status(deriv_root)

    print("Eddy status")
    print("-----------")
    print(f"unringed inputs : {status['unringed_count']}")
    print(f"eddy outputs    : {status['eddy_count']}")
    print(f"missing eddy    : {status['missing_eddy_count']}")
    print(f"quarantined     : {status['quarantine_count']}")

    if status["missing_eddy_examples"]:
        joined = ", ".join(status["missing_eddy_examples"])
        print(f"examples missing: {joined}")
    if status["quarantine_examples"]:
        joined = ", ".join(status["quarantine_examples"])
        print(f"examples quarant: {joined}")

    return status


def print_preproc_status(deriv_root: Path = DEFAULT_DERIV_ROOT) -> Dict[str, object]:
    status = collect_preproc_status(deriv_root)

    print("Preprocessing status")
    print("--------------------")
    print(f"raw files      : {status['raw_count']}")
    print(f"denoised       : {status['denoised_count']}")
    print(f"noise maps     : {status['noise_count']}")
    print(f"unringed       : {status['unringed_count']}")
    print(f"missing denoise: {status['missing_denoised_count']}")
    print(f"missing unring : {status['missing_unringed_count']}")

    if status["missing_denoised_examples"]:
        joined = ", ".join(status["missing_denoised_examples"])
        print(f"examples missing denoise: {joined}")
    if status["missing_unringed_examples"]:
        joined = ", ".join(status["missing_unringed_examples"])
        print(f"examples missing unring : {joined}")

    return status


def collect_t1_status(deriv_root: Path = DEFAULT_DERIV_ROOT) -> Dict[str, object]:
    deriv_root = Path(deriv_root)
    eddy_dir = deriv_root / "eddy"
    t1_anat = deriv_root / "t1_anat"
    t1_fast = deriv_root / "t1_fast"

    eddy_series = _series_ids_from_eddy(eddy_dir) if eddy_dir.exists() else set()
    eddy_subjects = _subject_ids_from_series(eddy_series)

    raw_ids = _extract_prefix_ids(t1_anat, "t1_from_dicom_*.nii*", "t1_from_dicom_")
    corr_ids = _extract_prefix_ids(t1_anat, "corrected_T1_*.nii*", "corrected_T1_")
    ss_ids = _extract_prefix_ids(t1_anat, "T1_ss_*.nii*", "T1_ss_")
    mask_ids = _extract_prefix_ids(t1_anat, "brainmask_*.nii*", "brainmask_")
    fast_pve2_ids = _t1_fast_ids(t1_fast, "_pve_2.nii.gz")
    fast_seg_ids = _t1_fast_ids(t1_fast, "_seg.nii.gz")

    complete_ids = corr_ids & ss_ids & mask_ids & fast_pve2_ids & fast_seg_ids
    complete_subjects = _subject_ids_from_series(complete_ids)
    missing_subjects = eddy_subjects - complete_subjects

    return {
        "eddy_series_count": len(eddy_series),
        "eddy_subject_count": len(eddy_subjects),
        "t1_raw_count": len(raw_ids),
        "t1_corrected_count": len(corr_ids),
        "t1_ss_count": len(ss_ids),
        "t1_mask_count": len(mask_ids),
        "t1_fast_pve2_count": len(fast_pve2_ids),
        "t1_fast_seg_count": len(fast_seg_ids),
        "t1_complete_count": len(complete_ids),
        "t1_complete_subject_count": len(complete_subjects),
        "t1_missing_subject_count": len(missing_subjects),
        "t1_missing_subject_examples": sorted(missing_subjects)[:10],
    }


def print_t1_status(deriv_root: Path = DEFAULT_DERIV_ROOT) -> Dict[str, object]:
    status = collect_t1_status(deriv_root)

    print("T1 status")
    print("---------")
    print(f"eddy series         : {status['eddy_series_count']}")
    print(f"eddy subjects       : {status['eddy_subject_count']}")
    print(f"t1 raw nifti        : {status['t1_raw_count']}")
    print(f"t1 corrected        : {status['t1_corrected_count']}")
    print(f"t1 skull-stripped   : {status['t1_ss_count']}")
    print(f"t1 brain masks      : {status['t1_mask_count']}")
    print(f"fast pve2           : {status['t1_fast_pve2_count']}")
    print(f"fast seg            : {status['t1_fast_seg_count']}")
    print(f"complete t1 series  : {status['t1_complete_count']}")
    print(f"complete subjects   : {status['t1_complete_subject_count']}")
    print(f"missing subjects    : {status['t1_missing_subject_count']}")

    if status["t1_missing_subject_examples"]:
        joined = ", ".join(status["t1_missing_subject_examples"])
        print(f"examples missing t1 : {joined}")

    return status


def collect_biascorr_status(deriv_root: Path = DEFAULT_DERIV_ROOT) -> Dict[str, object]:
    deriv_root = Path(deriv_root)
    eddy_dir = deriv_root / "eddy"
    bias_dir = deriv_root / "biascorr_1"

    eddy_ids = _series_ids_from_eddy(eddy_dir) if eddy_dir.exists() else set()
    ub_mif_ids = _extract_suffix_ids(bias_dir, "*_unbiased.mif", "_unbiased.mif")
    ub_nii_ids = _extract_suffix_ids(bias_dir, "*_unbiased.nii.gz", "_unbiased.nii.gz")
    biasfield_ids = _extract_suffix_ids(bias_dir, "*_biasfield.mif", "_biasfield.mif")
    b0_ids = _extract_suffix_ids(bias_dir, "*_b0.nii.gz", "_b0.nii.gz")

    complete_ids = ub_mif_ids & ub_nii_ids & biasfield_ids & b0_ids

    return {
        "eddy_series_count": len(eddy_ids),
        "unbiased_mif_count": len(ub_mif_ids),
        "unbiased_nii_count": len(ub_nii_ids),
        "biasfield_count": len(biasfield_ids),
        "b0_count": len(b0_ids),
        "complete_count": len(complete_ids),
        "missing_complete_count": len(eddy_ids - complete_ids),
        "missing_complete_examples": sorted(eddy_ids - complete_ids)[:10],
    }


def print_biascorr_status(deriv_root: Path = DEFAULT_DERIV_ROOT) -> Dict[str, object]:
    status = collect_biascorr_status(deriv_root)

    print("Bias correction status")
    print("----------------------")
    print(f"eddy series        : {status['eddy_series_count']}")
    print(f"unbiased mif       : {status['unbiased_mif_count']}")
    print(f"unbiased nii       : {status['unbiased_nii_count']}")
    print(f"bias fields        : {status['biasfield_count']}")
    print(f"b0 nifti           : {status['b0_count']}")
    print(f"complete bias sets : {status['complete_count']}")
    print(f"missing complete   : {status['missing_complete_count']}")

    if status["missing_complete_examples"]:
        joined = ", ".join(status["missing_complete_examples"])
        print(f"examples missing   : {joined}")

    return status


def collect_bbr_status(deriv_root: Path = DEFAULT_DERIV_ROOT) -> Dict[str, object]:
    deriv_root = Path(deriv_root)
    bias_dir = deriv_root / "biascorr_1"
    bbr_dir = deriv_root / "dwi_t1_bbr"

    bias_ids = _extract_suffix_ids(bias_dir, "*_unbiased.mif", "_unbiased.mif")
    b0mean_ids = _extract_suffix_ids(bbr_dir, "*_b0mean_ras.nii.gz", "_b0mean_ras.nii.gz")
    t1_ras_ids = _extract_suffix_ids(bbr_dir, "*_t1_ras.nii.gz", "_t1_ras.nii.gz")
    t1brain_ids = _extract_suffix_ids(bbr_dir, "*_t1brain_ras.nii.gz", "_t1brain_ras.nii.gz")
    mat_ids = _extract_suffix_ids(bbr_dir, "*_t12b0_bbr.mat", "_t12b0_bbr.mat")
    mask_ids = _extract_suffix_ids(bbr_dir, "*_mask_on_b0_bbr.nii.gz", "_mask_on_b0_bbr.nii.gz")

    key_complete_ids = bias_ids & b0mean_ids & t1_ras_ids & mat_ids
    missing_key = sorted(bias_ids - key_complete_ids)

    return {
        "bias_ready_count": len(bias_ids),
        "b0mean_count": len(b0mean_ids),
        "t1_ras_count": len(t1_ras_ids),
        "t1brain_ras_count": len(t1brain_ids),
        "mat_count": len(mat_ids),
        "mask_b0_count": len(mask_ids),
        "key_complete_count": len(key_complete_ids),
        "missing_key_count": len(missing_key),
        "missing_key_examples": missing_key[:10],
    }


def print_bbr_status(deriv_root: Path = DEFAULT_DERIV_ROOT) -> Dict[str, object]:
    status = collect_bbr_status(deriv_root)

    print("BBR bridge status")
    print("-----------------")
    print(f"bias-ready series  : {status['bias_ready_count']}")
    print(f"b0mean ras         : {status['b0mean_count']}")
    print(f"t1 ras             : {status['t1_ras_count']}")
    print(f"t1brain ras        : {status['t1brain_ras_count']}")
    print(f"t12b0 mat          : {status['mat_count']}")
    print(f"mask on b0         : {status['mask_b0_count']}")
    print(f"complete key sets  : {status['key_complete_count']}")
    print(f"missing key sets   : {status['missing_key_count']}")

    if status["missing_key_examples"]:
        joined = ", ".join(status["missing_key_examples"])
        print(f"examples missing   : {joined}")

    return status


def collect_step6_summary(
    deriv_root: Path = DEFAULT_DERIV_ROOT,
    *,
    raw_dwi_root: Path | None = None,
    discovered_total: int | None = None,
    discovered_raw_dwi: int | None = None,
    discovered_derived: int | None = None,
) -> Dict[str, object]:
    if raw_dwi_root is not None:
        from connectome_pipeline.dwi_convert import find_all_series

        _, discovery_counts = find_all_series(Path(raw_dwi_root))
        discovered_total = discovery_counts["total"]
        discovered_raw_dwi = discovery_counts["raw_dwi"]
        discovered_derived = discovery_counts["derived"]

    pre = collect_preproc_status(deriv_root)
    eddy = collect_eddy_status(deriv_root)
    t1 = collect_t1_status(deriv_root)
    bias = collect_biascorr_status(deriv_root)
    bbr = collect_bbr_status(deriv_root)

    return {
        "discovered_total": discovered_total,
        "discovered_raw_dwi": discovered_raw_dwi,
        "discovered_derived": discovered_derived,
        "raw_mif_count": pre["raw_count"],
        "denoised_count": pre["denoised_count"],
        "unringed_count": pre["unringed_count"],
        "eddy_count": eddy["eddy_count"],
        "eddy_quarantine_count": eddy["quarantine_count"],
        "t1_complete_subject_count": t1["t1_complete_subject_count"],
        "bias_complete_count": bias["complete_count"],
        "bbr_complete_count": bbr["key_complete_count"],
    }


def print_step6_summary(
    deriv_root: Path = DEFAULT_DERIV_ROOT,
    *,
    raw_dwi_root: Path | None = None,
    discovered_total: int | None = None,
    discovered_raw_dwi: int | None = None,
    discovered_derived: int | None = None,
) -> Dict[str, object]:
    status = collect_step6_summary(
        deriv_root,
        raw_dwi_root=raw_dwi_root,
        discovered_total=discovered_total,
        discovered_raw_dwi=discovered_raw_dwi,
        discovered_derived=discovered_derived,
    )

    def _fmt(count: int | None, base: int | None) -> str:
        if count is None:
            return "n/a"
        if base and base > 0:
            return f"{count} ({count/base:.1%})"
        return str(count)

    print("Pipeline counts through Step 6")
    print("------------------------------")
    if discovered_total is not None:
        print(f"step1 discovered total : {_fmt(discovered_total, discovered_total)}")
    if discovered_raw_dwi is not None:
        print(f"step1 raw dwi series   : {_fmt(discovered_raw_dwi, discovered_raw_dwi)}")
    if discovered_derived is not None:
        print(f"step1 derived series   : {_fmt(discovered_derived, discovered_total)}")

    raw_base = discovered_raw_dwi
    print(f"step1 raw mif outputs  : {_fmt(status['raw_mif_count'], raw_base)}")
    print(f"step2 denoised         : {_fmt(status['denoised_count'], raw_base)}")
    print(f"step3 unringed         : {_fmt(status['unringed_count'], raw_base)}")
    print(f"step4 eddy             : {_fmt(status['eddy_count'], raw_base)}")
    print(f"step4 quarantined      : {_fmt(status['eddy_quarantine_count'], raw_base)}")
    print(f"step5 complete t1      : {_fmt(status['t1_complete_subject_count'], status['eddy_count'])}")
    print(f"step6a bias complete   : {_fmt(status['bias_complete_count'], status['eddy_count'])}")
    print(f"step6d bbr complete    : {_fmt(status['bbr_complete_count'], status['bias_complete_count'])}")

    return status


if __name__ == "__main__":
    print_preproc_status()
