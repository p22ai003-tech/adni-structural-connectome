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


PROJECT_ROOT = Path("/home/ec2-user/exp")
REFERENCE_ROOT = (
    PROJECT_ROOT
    / "research_audit/outputs/connectome_dashboard_reference_v1"
)


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
