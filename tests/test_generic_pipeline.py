"""The pipeline has to work for a study that is not ours.

Everything here builds a small synthetic study on disk and drives the real
tools over it. No ADNI paths, no machine-specific locations, no imaging
toolchain: these are the steps that run before anything is computed, which are
exactly the steps a new user hits first and the ones that used to assume our
cohort's shape.
"""

from __future__ import annotations

import csv
import os
import json
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import sc_discover  # noqa: E402
import sc_source_lock  # noqa: E402
import sc_study  # noqa: E402
from scforge.input_contract import stable_unit  # noqa: E402


# --------------------------------------------------------------------------
# study configuration
# --------------------------------------------------------------------------

def test_template_config_is_usable_once_the_paths_are_filled_in(tmp_path):
    target = tmp_path / "study.yaml"
    sc_study.write_template(target)
    text = target.read_text()
    assert "layout: simple" in text
    assert "source: local" in text
    # The template is not itself runnable: it names placeholder paths on
    # purpose, so a user who forgets to edit it is told rather than silently
    # running against nothing.
    with pytest.raises(sc_study.StudyError):
        sc_study.load_study(target).ensure_available()


def test_missing_settings_are_all_reported_at_once(tmp_path):
    config = tmp_path / "study.yaml"
    config.write_text("study: {layout: nope}\ninput: {source: s3}\noutput: {}\n")
    with pytest.raises(sc_study.StudyError) as error:
        sc_study.load_study(config)
    message = str(error.value)
    for expected in ("study.layout", "input.uri", "input.stage_root", "output.data_root"):
        assert expected in message


def test_s3_input_needs_a_uri_and_a_staging_directory(tmp_path):
    config = tmp_path / "study.yaml"
    config.write_text(
        "study: {name: s, layout: bids}\n"
        f"input: {{source: s3, uri: 's3://bucket/prefix', stage_root: {tmp_path / 'stage'}}}\n"
        f"output: {{data_root: {tmp_path / 'work'}}}\n"
    )
    study = sc_study.load_study(config)
    assert study.staged_root == tmp_path / "stage"
    # Nothing downstream should see the URI: the pipeline reads the staged copy.
    assert study.environment()["SC_RAW_IMAGES_ROOT"] == str(tmp_path / "stage")


def test_outputs_derive_from_one_root(tmp_path):
    config = tmp_path / "study.yaml"
    config.write_text(
        "study: {name: s, layout: simple}\n"
        f"input: {{source: local, path: {tmp_path}}}\n"
        f"output: {{data_root: {tmp_path / 'work'}}}\n"
    )
    environment = sc_study.load_study(config).environment()
    derivatives = Path(environment["SC_DERIV_ROOT"])
    assert derivatives == tmp_path / "work" / "derivatives"
    assert Path(environment["SC_QC_ROOT"]).is_relative_to(derivatives)
    assert Path(environment["SC_CONNECTOMES_DIR"]).is_relative_to(derivatives)


# --------------------------------------------------------------------------
# layouts and pairing
# --------------------------------------------------------------------------

def _scan(root: Path, subject: str, session: str | None, *, layout: str) -> None:
    """Create a minimal DWI + T1 pair for one subject in the given layout."""
    if layout == "simple":
        places = {"dwi": root / "dwi", "anat": root / "anat"}
        for modality, base in places.items():
            folder = base / subject / session if session else base / subject
            folder.mkdir(parents=True, exist_ok=True)
            (folder / f"{modality}.nii").write_bytes(b"\0" * 16)
            if modality == "dwi":
                for extension in ("bval", "bvec", "json"):
                    (folder / f"dwi.{extension}").write_text("0\n")
    elif layout == "bids":
        holder = root / subject / session if session else root / subject
        for modality, name in (("dwi", "dwi"), ("anat", "T1w")):
            folder = holder / modality
            folder.mkdir(parents=True, exist_ok=True)
            stem = f"{subject}_{session}_{name}" if session else f"{subject}_{name}"
            (folder / f"{stem}.nii").write_bytes(b"\0" * 16)
            if modality == "dwi":
                for extension in ("bval", "bvec", "json"):
                    (folder / f"{stem}.{extension}").write_text("0\n")
    else:  # adni
        for modality in ("dti", "mri"):
            folder = root / modality / subject / "Protocol" / "2020-01-01_00_00_00.0" / "I100"
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "0001.dcm").write_bytes(b"\0" * 16)


@pytest.mark.parametrize("layout", ["simple", "bids"])
def test_every_layout_finds_and_pairs_its_scans(tmp_path, layout):
    root = tmp_path / "raw"
    subjects = ["sub-001", "sub-002"] if layout != "adni" else ["XXX_S_NNNN"]
    for subject in subjects:
        _scan(root, subject, "ses-01", layout=layout)

    dwi = sc_discover.scan_modality(root, "dwi", None, layout)
    t1 = sc_discover.scan_modality(root, "anat", None, layout)
    assert len(dwi) == len(subjects)
    assert len(t1) == len(subjects)

    rows, unpaired = sc_discover.pair(dwi, t1, 90.0, 180.0, None)
    assert len(rows) == len(subjects)
    assert unpaired == []


def test_sessions_without_dates_still_pair(tmp_path):
    """A generic layout need not encode when a scan was taken."""
    root = tmp_path / "raw"
    _scan(root, "sub-001", "ses-01", layout="simple")
    dwi = sc_discover.scan_modality(root, "dwi", None, "simple")
    t1 = sc_discover.scan_modality(root, "anat", None, "simple")
    rows, unpaired = sc_discover.pair(dwi, t1, 90.0, 180.0, None)
    assert not unpaired
    assert rows[0]["timing_stratum"] == "undated"
    # The gap is left empty rather than invented.
    assert rows[0]["abs_pair_gap_days"] == ""


def test_a_subject_subset_restricts_the_scan(tmp_path):
    root = tmp_path / "raw"
    for subject in ("sub-001", "sub-002", "sub-003"):
        _scan(root, subject, None, layout="simple")
    found = sc_discover.scan_modality(root, "dwi", {"sub-002"}, "simple")
    assert [series.subject for series in found] == ["sub-002"]


def test_participants_supply_the_grouping_variable(tmp_path):
    participants = tmp_path / "participants.csv"
    participants.write_text("subject_id,group,age,sex\nsub-001,CN,71,F\n")
    rows = [{"subject_id": "sub-001", "diagnosis": "", "age": "", "sex": ""}]
    assert sc_discover.attach_participants(rows, participants) == 1
    assert rows[0]["diagnosis"] == "CN"
    assert rows[0]["age"] == "71"


def test_participants_file_without_a_subject_column_is_rejected(tmp_path):
    participants = tmp_path / "participants.csv"
    participants.write_text("name,group\nsub-001,CN\n")
    with pytest.raises(SystemExit):
        sc_discover.attach_participants([], participants)


# --------------------------------------------------------------------------
# the input contract
# --------------------------------------------------------------------------

def test_unit_ids_accept_a_generic_image_id():
    assert stable_unit({"subject_id": "sub-001", "dti_image_id": "ses-01"}) == "sub-001_Ises-01"
    assert stable_unit({"subject_id": "XXX_S_NNNN", "dti_image_id": "863064"}) == "XXX_S_NNNN_I863064"


def test_unit_ids_reject_what_would_be_ambiguous():
    for row in (
        {"subject_id": "sub-001", "dti_image_id": ""},
        {"subject_id": "sub-001", "dti_image_id": "ses_I01"},
        {"subject_id": "sub-001", "dti_image_id": "../escape"},
        {"subject_id": "../escape", "dti_image_id": "1"},
    ):
        with pytest.raises(ValueError):
            stable_unit(row)


# --------------------------------------------------------------------------
# the source content lock
# --------------------------------------------------------------------------

def _manifest_row(folder: Path) -> dict:
    return {
        "subject_id": "sub-001",
        "dti_image_id": "ses-01",
        "t1_image_id": "ses-01",
        "dti_source_kind": "dicom_series",
        "dti_source_path": str(folder / "dwi"),
        "t1_source_kind": "nifti_single",
        "t1_source_path": str(folder / "t1.nii"),
        "normalization_readiness": "READY",
    }


def _write_manifest(path: Path, row: dict) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def test_content_lock_records_every_file_and_notices_a_change(tmp_path):
    study = tmp_path / "study"
    (study / "dwi").mkdir(parents=True)
    for index in range(3):
        (study / "dwi" / f"{index:04d}.dcm").write_bytes(bytes([index]) * 32)
    (study / "t1.nii").write_bytes(b"\1" * 64)

    manifest = tmp_path / "manifest.csv"
    _write_manifest(manifest, _manifest_row(study))

    first = sc_source_lock.build(manifest, tmp_path / "lock1")
    assert first["inventory_row_count"] == 4  # three DICOM files and one T1
    assert first["missing_files"] == []

    (study / "dwi" / "0001.dcm").write_bytes(b"different")
    second = sc_source_lock.build(manifest, tmp_path / "lock2")
    assert second["inventory_sha256"] != first["inventory_sha256"]


def test_content_lock_reports_a_missing_source(tmp_path):
    study = tmp_path / "study"
    (study / "dwi").mkdir(parents=True)
    (study / "dwi" / "0001.dcm").write_bytes(b"x")
    manifest = tmp_path / "manifest.csv"
    _write_manifest(manifest, _manifest_row(study))  # t1.nii never created

    result = sc_source_lock.build(manifest, tmp_path / "lock")
    assert len(result["missing_files"]) == 1
    validation = json.loads(result["validation_path"].read_text())
    assert validation["checks"]["all_manifest_files_present"] is False


def test_projection_carries_the_manifest_and_inventory_identities(tmp_path):
    study = tmp_path / "study"
    (study / "dwi").mkdir(parents=True)
    (study / "dwi" / "0001.dcm").write_bytes(b"x")
    (study / "t1.nii").write_bytes(b"y")
    manifest = tmp_path / "manifest.csv"
    _write_manifest(manifest, _manifest_row(study))

    result = sc_source_lock.build(manifest, tmp_path / "lock")
    rows = list(csv.DictReader(result["projection_path"].open(encoding="utf-8")))
    assert len(rows) == 1
    assert rows[0]["locked_inventory_sha256"] == result["inventory_sha256"]
    assert rows[0]["dti_inventory_file_count"] == "1"


# --------------------------------------------------------------------------
# the command-line surface
# --------------------------------------------------------------------------

@pytest.mark.parametrize("command", [
    "doctor", "discover", "probe", "validate", "approve",
    "freeze", "review", "continue", "publish", "run",
])
def test_every_step_documents_itself(command):
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "run_imaging.py"), command, "--help"],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


def test_phase_b_cannot_be_approved_directly():
    """Phase B opens through `continue`, after preflight and a human review."""
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "run_imaging.py"), "approve",
         "--run-root", "/tmp/scforge_v2_nonexistent", "--by", "test", "--mode", "phase-b"],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode != 0
    assert "phase-b" in result.stderr or "invalid choice" in result.stderr


# --------------------------------------------------------------------------
# acquisition parameters
# --------------------------------------------------------------------------

def test_a_supplied_sidecar_is_read_rather_than_re_derived(tmp_path):
    """A DWI given as NIfTI already carries what dcm2niix would have produced."""
    import sc_probe_acquisition

    sidecar = tmp_path / "dwi.json"
    sidecar.write_text(json.dumps({
        "PhaseEncodingDirection": "j-",
        "TotalReadoutTime": 0.0333,
        "Manufacturer": "GE",
    }))
    result = sc_probe_acquisition.read_sidecar(sidecar)
    assert "error" not in result
    values = sc_probe_acquisition.row_values(result["sidecar"], origin=result["origin"])
    assert values["phase_encoding_direction"] == "j-"
    assert values["normalization_readiness"] == "READY"
    # Where the value came from is recorded, not flattened into "dicom_header".
    assert values["phase_encoding_source"] == "supplied_sidecar"


def test_an_axis_without_a_polarity_never_becomes_ready(tmp_path):
    import sc_probe_acquisition

    for origin in ("dicom_header", "supplied_sidecar"):
        values = sc_probe_acquisition.row_values(
            {"PhaseEncodingAxis": "j", "TotalReadoutTime": 0.05}, origin=origin)
        assert values["phase_encoding_direction"] == "j"
        assert values["normalization_readiness"] == "FAIL_MISSING_PHASE_ENCODING"


def test_a_broken_sidecar_is_an_error_not_a_guess(tmp_path):
    import sc_probe_acquisition

    broken = tmp_path / "dwi.json"
    broken.write_text("{not json")
    assert "error" in sc_probe_acquisition.read_sidecar(broken)
    assert "error" in sc_probe_acquisition.read_sidecar(tmp_path / "absent.json")


# --------------------------------------------------------------------------
# the analysis side's cohort table
# --------------------------------------------------------------------------

def test_a_participants_table_stands_in_for_the_cohort_exports(tmp_path):
    """A study that is not ADNI has one small table, not two IDA exports."""
    from connectome_analysis.analysis_cohort import _load_cohort_csv

    participants = tmp_path / "participants.csv"
    participants.write_text(
        "participant_id,diagnosis,age,gender\n"
        "sub-001,CN,71,F\n"
        "sub-002,AD,78,M\n"
    )
    table = _load_cohort_csv(participants)
    assert list(table["subject_id"]) == ["sub-001", "sub-002"]
    assert list(table["group"]) == ["CN", "AD"]
    assert list(table["Age"]) == [71, 78]
    assert list(table["Sex"]) == ["F", "M"]


def test_the_adni_column_names_still_win(tmp_path):
    from connectome_analysis.analysis_cohort import _load_cohort_csv

    export = tmp_path / "dti.csv"
    export.write_text("Subject ID,Research Group,Age,Sex\nXXX_S_NNNN,CN,70,F\n")
    table = _load_cohort_csv(export)
    assert list(table["subject_id"]) == ["XXX_S_NNNN"]
    assert list(table["group"]) == ["CN"]


def test_a_table_with_no_subject_column_says_what_it_wanted(tmp_path):
    from connectome_analysis.analysis_cohort import _load_cohort_csv

    bad = tmp_path / "participants.csv"
    bad.write_text("name,diagnosis\nsub-001,CN\n")
    with pytest.raises(ValueError) as error:
        _load_cohort_csv(bad)
    assert "participant_id" in str(error.value)


def test_the_study_file_beats_an_exported_variable(tmp_path, monkeypatch):
    """`source env.sh` must not quietly redirect a study's output."""
    config = tmp_path / "study.yaml"
    participants = tmp_path / "participants.csv"
    participants.write_text("subject_id,diagnosis\nsub-001,CN\n")
    config.write_text(
        "study: {name: s, layout: simple}\n"
        f"input: {{source: local, path: {tmp_path}, participants: participants.csv}}\n"
        f"output: {{data_root: {tmp_path / 'work'}}}\n"
    )
    monkeypatch.setenv("SC_DATA_ROOT", "/somewhere/else")
    monkeypatch.setenv("SC_COHORT_DIR", "/another/study/cohort")
    sc_study.load_study(config).apply_environment()
    assert os.environ["SC_DATA_ROOT"] == str(tmp_path / "work")
    # A study with its own participants table owns its cohort directory, so it
    # can never join against another study's subjects.
    assert os.environ["SC_COHORT_DIR"] == str(tmp_path)
