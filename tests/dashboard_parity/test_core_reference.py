from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from connectome_dashboard_core.artifacts import Artifact, artifact_path
from connectome_dashboard_core.sections import SECTIONS


EXPECTED_SECTION_LABELS = [
    "Demographics",
    "Novel Findings",
    "Global DTI",
    "Local / ROI DTI",
    "Global Graph",
    "SC Matrix Viewer",
    "Node Metrics",
    "Coupling",
    "Coupling-AAL",
    "Brain Age",
    "LR / SR",
    "LR-SR Analysis",
    "EDR Exceptions",
    "ML Diagnostics",
    "Delay",
    "Advanced",
    "Network Analysis",
    "Functional Pending",
]


def test_section_registry_matches_frozen_reference():
    assert len(SECTIONS) == 18
    assert [section.label for section in SECTIONS] == EXPECTED_SECTION_LABELS
    assert len({section.id for section in SECTIONS}) == 18


def test_tract_summary_matches_frozen_streamlit(
    core_snapshot,
    reference_lr_sr,
):
    tract = core_snapshot["tract"].set_index("Panel")
    frozen = reference_lr_sr["dataframes"][2]
    frozen_rows = frozen["head"] + frozen["tail"]
    expected = {
        str(row["Cohort"]): row
        for row in frozen_rows
    }
    assert set(tract.index) == {"Overall", "CN", "MCI", "AD"}
    for panel, row in expected.items():
        assert int(tract.loc[panel, "n_subjects"]) == int(row["Cases"])
        assert int(tract.loc[panel, "n_edges"]) == int(
            row["Positive edges"]
        )
        assert float(tract.loc[panel, "median"]) == pytest.approx(
            float(row["Median length (mm)"]),
            abs=1e-12,
        )
        assert float(tract.loc[panel, "q1"]) == pytest.approx(
            float(row["Q1 (mm)"]),
            abs=1e-12,
        )
        assert float(tract.loc[panel, "q3"]) == pytest.approx(
            float(row["Q3 (mm)"]),
            abs=1e-12,
        )
    assert core_snapshot["samples"].shape == (230_000, 3)


def test_edr_exception_definition_and_counts_match_reference(core_snapshot):
    thresholds = core_snapshot["thresholds"]
    assert float(thresholds["short_max_mm"]) == pytest.approx(78.6856)
    assert float(thresholds["long_min_mm"]) == pytest.approx(171.021)
    assert int(thresholds["min_bin_edges"]) == 120
    assert len(core_snapshot["subjects"]) == 530

    edr = core_snapshot["edr"].set_index(["Cohort", "Range"])
    assert int(edr.loc[("Overall", "SR"), "Candidate edges"]) == 1_336_215
    assert int(edr.loc[("Overall", "SR"), "Exception edges"]) == 32_545
    assert float(
        edr.loc[("Overall", "SR"), "Pooled exception (%)"]
    ) == pytest.approx(2.4356110356492033)
    assert float(
        edr.loc[("MCI", "LR"), "Median exception measure"]
    ) == pytest.approx(40.8944)
    assert float(
        edr.loc[("AD", "LR"), "Median exception measure"]
    ) == pytest.approx(47.3813)


def test_requested_strength_tests_match_frozen_values(core_snapshot):
    table = core_snapshot["inference"]
    requested = table[
        pd.to_numeric(
            table["Requested 4-test Holm p"],
            errors="coerce",
        ).notna()
    ].copy()
    assert len(requested) == 4
    assert set(requested["Requested contrast supported"]) == {"YES"}
    assert set(requested["Observed direction"]) == {"AD higher"}
    assert np.allclose(
        pd.to_numeric(
            requested["Requested 4-test Holm p"],
            errors="raise",
        ),
        0.03193000110770622,
        atol=1e-15,
    )

    lookup = requested.set_index(["Metric", "Contrast"])
    expected = {
        ("Exception strength", "MCI vs AD"): (
            40.8944,
            47.3813,
            -0.2016,
        ),
        ("Non-exception strength", "MCI vs AD"): (
            2.12262,
            2.64919,
            -0.2047,
        ),
        ("Exception − non-exception strength", "MCI vs AD"): (
            38.03389,
            44.941922,
            -0.2013,
        ),
        ("Exception − non-exception strength", "CN vs AD"): (
            36.31887,
            44.941922,
            -0.1773,
        ),
    }
    for key, (first, second, effect) in expected.items():
        row = lookup.loc[key]
        assert float(row["Median A"]) == pytest.approx(first, abs=1e-5)
        assert float(row["Median B"]) == pytest.approx(second, abs=1e-5)
        assert float(row["Effect"]) == pytest.approx(effect, abs=5e-5)

    metadata = core_snapshot["inference_metadata"]
    assert metadata["exception_omnibus_adjusted_p"] == pytest.approx(
        0.08649321694836706
    )
    assert metadata["change_mci_ad_adjusted_p"] == pytest.approx(
        0.03193000110770622
    )


def test_feature_family_counts_match_streamlit_reference(core_snapshot):
    assert core_snapshot["feature_subject_n"] == 530
    assert len(core_snapshot["features"]) == 2_299
    families = core_snapshot["families"].set_index(
        "Requested feature family"
    )
    assert int(
        families.loc[
            "Graph / network topology",
            "Candidate features",
        ]
    ) == 527
    assert int(
        families.loc[
            "Connectome edge / matrix",
            "Candidate features",
        ]
    ) == 182
    assert int(
        families.loc[
            "Network-wise systems (DMN/limbic/etc.)",
            "Candidate features",
        ]
    ) == 0
    assert (
        families.loc[
            "Network-wise systems (DMN/limic/etc.)"
            if "Network-wise systems (DMN/limic/etc.)" in families.index
            else "Network-wise systems (DMN/limbic/etc.)",
            "Current model binding",
        ]
        == "NOT EXPLICITLY PRESENT"
    )


def test_artifact_path_cannot_escape_analysis_root(settings):
    malicious = Artifact(
        id="malicious",
        relative_path="../../../../etc/passwd",
        name="passwd",
        suffix=".csv",
        size_bytes=1,
        mtime_ns=1,
        mime_type="text/csv",
        kind="table",
    )
    with pytest.raises(ValueError):
        artifact_path(settings, malicious)
