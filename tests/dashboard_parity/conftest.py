from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from apps.connectome_api.main import app
from connectome_dashboard_core.lr_sr import (
    edr_exception_count_summary,
    edr_exception_inference,
    model_feature_eda,
    tract_length_distribution_data,
)
from connectome_dashboard_core.settings import get_settings


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_ROOT = (
    PROJECT_ROOT
    / "research_audit/outputs/connectome_dashboard_reference_v1"
)


def _study_outputs_present() -> bool:
    """These tests compare the dashboard against this study's own results.

    They need the analysis outputs of the ADNI cohort and the frozen reference
    captured from the earlier Streamlit dashboard, neither of which is part of
    the repository. Where they are absent there is nothing to compare, so the
    tests are skipped rather than failed.
    """
    try:
        analysis_root = get_settings().analysis_root
    except Exception:
        return False
    return (
        (REFERENCE_ROOT / "section_11.json").is_file()
        and (Path(analysis_root) / "00_master" / "master_cohort.csv").is_file()
    )


def pytest_collection_modifyitems(config, items):
    if _study_outputs_present():
        return
    skip = pytest.mark.skip(
        reason="needs this study's analysis outputs and the frozen dashboard "
               "reference, which are not part of the repository"
    )
    here = Path(__file__).resolve().parent
    for item in items:
        if Path(str(item.fspath)).resolve().is_relative_to(here):
            item.add_marker(skip)


@pytest.fixture(scope="session")
def settings():
    return get_settings()


@pytest.fixture(scope="session")
def reference_lr_sr():
    return json.loads(
        (REFERENCE_ROOT / "section_11.json").read_text(
            encoding="utf-8"
        )
    )


@pytest.fixture(scope="session")
def core_snapshot(settings):
    samples, tract = tract_length_distribution_data(settings)
    edr, thresholds, subjects = edr_exception_count_summary(
        settings,
        "fd_sum",
    )
    plot, inference, metadata = edr_exception_inference(
        settings,
        "fd_sum",
    )
    features, families, feature_subject_n = model_feature_eda(settings)
    return {
        "samples": samples,
        "tract": tract,
        "edr": edr,
        "thresholds": thresholds,
        "subjects": subjects,
        "plot": plot,
        "inference": inference,
        "inference_metadata": metadata,
        "features": features,
        "families": families,
        "feature_subject_n": feature_subject_n,
    }


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="session")
def a_subject():
    """Any subject present in the connectome tree.

    Discovered rather than hard-coded: an ADNI participant identifier is
    restricted under the Data Use Agreement and does not belong in a tracked
    test file. The test needs some subject that exists, not a specific one.
    """
    import re

    import sc_config

    pattern = re.compile(r"SC_AAL166_(\d{3}_S_\d{4,5})_I\d+_fd_sum\.csv$")
    for path in sorted(sc_config.paths().connectomes_dir.glob("SC_AAL166_*_fd_sum.csv")):
        m = pattern.search(path.name)
        if m:
            return m.group(1)
    pytest.skip("no connectome matrices available to exercise the endpoint")
