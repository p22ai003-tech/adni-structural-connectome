from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_health_and_metadata(client):
    live = client.get("/api/v1/health/live")
    ready = client.get("/api/v1/health/ready")
    metadata = client.get("/api/v1/metadata/app")
    assert live.status_code == 200
    assert ready.status_code == 200
    assert ready.json()["status"] == "ready"
    assert metadata.status_code == 200
    assert metadata.json()["data"]["read_only"] is True
    assert metadata.json()["data"]["analysis_section_n"] == 18
    assert live.headers["x-content-type-options"] == "nosniff"
    assert live.headers["x-request-id"]


def test_api_is_read_only(client):
    for method in ("post", "put", "patch", "delete"):
        response = getattr(client, method)("/api/v1/health/live")
        assert response.status_code == 405
        assert response.json()["detail"] == "This service is read-only"


def test_section_and_artifact_contracts(client):
    response = client.get("/api/v1/sections")
    assert response.status_code == 200
    sections = response.json()["data"]
    assert len(sections) == 18
    assert {section["id"] for section in sections} >= {
        "global-dti",
        "global-graph",
        "lr-sr-analysis",
        "network-analysis",
    }
    assert client.get("/api/v1/sections/not-real/artifacts").status_code == 404
    assert client.get("/api/v1/artifacts/not-real/table").status_code == 404


def test_artifact_download_bytes_match_allow_listed_source(client, settings):
    inventory = client.get("/api/v1/sections/demographics/artifacts")
    assert inventory.status_code == 200
    artifact = next(
        row
        for row in inventory.json()["data"]
        if row["relative_path"] == "00_master/qc_snapshot.csv"
    )
    response = client.get(
        f"/api/v1/artifacts/{artifact['id']}/download"
    )
    assert response.status_code == 200
    source = settings.analysis_root / artifact["relative_path"]
    assert response.content == source.read_bytes()
    assert response.headers["content-disposition"].startswith(
        "attachment;"
    )
    assert client.get(
        "/api/v1/artifacts/..%2F..%2Fetc%2Fpasswd/download"
    ).status_code in {404, 422}


def test_lr_sr_api_matches_requested_inference(client):
    ranges = client.get("/api/v1/lr-sr/tract-length/ranges")
    assert ranges.status_code == 200
    range_data = ranges.json()["data"]
    assert range_data["population_median_mm"] == pytest.approx(
        124.040688885913
    )
    assert range_data["short_max_mm"] == pytest.approx(94.4099)
    assert range_data["medium_max_mm"] == pytest.approx(154.102)
    assert [row["code"] for row in range_data["ranges"]] == [
        "SR",
        "MR",
        "LR",
    ]

    response = client.get(
        "/api/v1/lr-sr/exceptions/inference"
        "?measure=fd_sum&table=requested"
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["provenance"]["calculation"]
    rows = payload["data"]["rows"]
    assert len(rows) == 4
    assert {row["Observed direction"] for row in rows} == {"AD higher"}
    assert all(
        row["Requested 4-test Holm p"]
        == pytest.approx(0.03193000110770622)
        for row in rows
    )
    assert all(
        row["Requested contrast supported"] == "YES"
        for row in rows
    )
    full = client.get(
        "/api/v1/lr-sr/exceptions/inference"
        "?measure=fd_sum&table=full"
    )
    assert full.status_code == 200
    across_group = [
        row
        for row in full.json()["data"]["rows"]
        if row["Question"] == "Across groups"
        and row["Test"] == "Mann-Whitney U"
    ]
    assert len(across_group) == 9
    assert {
        row["Metric"] for row in across_group
    } == {
        "Exception strength",
        "Non-exception strength",
        "Exception − non-exception strength",
    }
    assert {
        row["Contrast"] for row in across_group
    } == {"CN vs MCI", "CN vs AD", "MCI vs AD"}
    interaction = full.json()["data"]["interaction_rows"]
    assert len(interaction) == 4
    omnibus = next(
        row
        for row in interaction
        if row["Test"] == "Kruskal-Wallis"
    )
    assert omnibus["p"] == pytest.approx(0.033373191984145796)
    cn_ad = next(
        row
        for row in interaction
        if row["Contrast"] == "CN vs AD"
    )
    assert cn_ad["Observed direction"] == "AD higher"
    assert cn_ad["Adjusted p"] == pytest.approx(
        0.03433826519611171
    )


def test_matrix_endpoint_is_allow_listed(client, a_subject):
    valid = client.get(
        f"/api/v1/connectomes/{a_subject}/fd_sum/summary"
    )
    assert valid.status_code == 200
    summary = valid.json()["data"]["summary"]
    assert summary["positive_edges"] > 0
    assert 0 < summary["density"] <= 1
    invalid = client.get(
        f"/api/v1/connectomes/{a_subject}/arbitrary/summary"
    )
    assert invalid.status_code == 422
    unknown = client.get(
        "/api/v1/connectomes/not_a_subject/fd_sum/summary"
    )
    assert unknown.status_code == 404


def test_pipeline_poll_payload_is_compact_and_read_only(client):
    response = client.get("/api/v1/pipeline/release-status")
    assert response.status_code == 200
    assert len(response.content) < 100_000
    payload = response.json()["data"]
    assert payload["subject_rows"] == []
    assert payload["summary"]["total_n"] == 530
    assert payload["ledger_summary"]


def test_json_contract_has_no_nonstandard_nan(client):
    for path in (
        "/api/v1/lr-sr/tract-length/summary",
        "/api/v1/lr-sr/exceptions/summary",
        "/api/v1/models/features/families",
        "/api/v1/models/tasks/diagnosis",
        (
            "/api/v1/networks/analysis/distribution"
            "?scheme=functional&family=microstructure&metric=fa_mean"
        ),
        "/api/v1/findings/summary",
        "/api/v1/pipeline/live-status",
    ):
        response = client.get(path)
        assert response.status_code == 200
        text = response.text
        assert "NaN" not in text
        assert "Infinity" not in text
        json.loads(text)


def test_scientific_section_profiles_are_not_artifact_only(client):
    global_dti = client.get(
        "/api/v1/sections/global-dti/analysis/profile"
    )
    local_dti = client.get(
        "/api/v1/sections/local-roi-dti/analysis/profile"
    )
    coupling = client.get(
        "/api/v1/sections/coupling/analysis/profile"
    )
    assert global_dti.status_code == 200
    assert local_dti.status_code == 200
    assert coupling.status_code == 200
    global_profile = global_dti.json()["data"]
    local_profile = local_dti.json()["data"]
    coupling_profile = coupling.json()["data"]
    assert global_profile["subject"]["available"] is True
    assert global_profile["subject"]["subject_n"] == 529
    assert global_profile["subject"]["metrics"][0]["id"] == (
        "fa_mean_edge_mean"
    )
    assert local_profile["node"]["available"] is True
    assert local_profile["node"]["nodes"] == 166
    assert coupling_profile["subject"]["available"] is True
    assert {
        metric["id"] for metric in coupling_profile["subject"]["metrics"]
    } >= {"strength_fa_mean_rho", "nodal_eff_rd_mean_rho"}


def test_subject_analysis_uses_stored_statistics(client):
    response = client.get(
        "/api/v1/sections/global-dti/analysis/subject/"
        "fa_mean_edge_mean"
    )
    assert response.status_code == 200
    payload = response.json()["data"]
    assert len(payload["rows"]) == 529
    assert len(payload["display_rows"]) < len(payload["rows"])
    assert payload["statistic_source"] == "stored analysis outputs"
    descriptives = {
        row["group"]: row for row in payload["descriptives"]
    }
    assert descriptives["CN"]["median"] == pytest.approx(
        0.3920536838853838
    )
    assert len(payload["pairwise"]) == 3
    assert payload["display_filter"].startswith("Per-group")


def test_node_rankings_and_region_values_are_subject_level(client):
    ranking = client.get(
        "/api/v1/sections/local-roi-dti/analysis/nodes/fa_mean/"
        "ranking?group_a=CN&group_b=MCI"
    )
    assert ranking.status_code == 200
    rows = ranking.json()["data"]["rows"]
    assert len(rows) == 166
    assert [row["p_value"] for row in rows] == sorted(
        row["p_value"] for row in rows
    )
    assert rows[0]["atlas_label"]
    assert "Benjamini-Hochberg" in ranking.json()["data"]["method"]

    node = int(rows[0]["node"])
    values = client.get(
        f"/api/v1/sections/local-roi-dti/analysis/nodes/fa_mean/{node}"
    )
    assert values.status_code == 200
    payload = values.json()["data"]
    assert payload["node"] == node
    assert {row["group"] for row in payload["rows"]} <= {
        "CN",
        "MCI",
        "AD",
    }
    assert len(payload["descriptives"]) == 3


def test_demographics_contract_carries_all_groups_and_stored_age_tests(
    client,
):
    response = client.get("/api/v1/demographics/summary")
    assert response.status_code == 200
    payload = response.json()["data"]
    assert sum(int(row["n_subjects"]) for row in payload["cohort"]) == 530
    assert len(payload["age_rows"]) == 530
    assert len(payload["age_descriptives"]) == 3
    assert len(payload["age_pairwise"]) == 3
    assert {row["group"] for row in payload["cohort"]} == {
        "CN",
        "MCI",
        "AD",
    }


def test_recorded_ml_tasks_are_exposed_without_refitting(client):
    catalog = client.get("/api/v1/models/tasks")
    diagnosis = client.get("/api/v1/models/tasks/diagnosis")
    assert catalog.status_code == 200
    assert diagnosis.status_code == 200
    assert {task["id"] for task in catalog.json()["data"]["tasks"]} == {
        "diagnosis",
        "cdr",
        "score",
    }
    payload = diagnosis.json()["data"]
    assert payload["kind"] == "classification"
    assert len(payload["performance"]) == 5
    assert len(payload["predictions"]) == 2650
    assert "not refitted" in (
        diagnosis.json()["provenance"]["calculation"]
    )


def test_network_analysis_carries_values_stats_and_blocks(client):
    catalog = client.get("/api/v1/networks/analysis/catalog")
    distribution = client.get(
        "/api/v1/networks/analysis/distribution"
        "?scheme=functional&family=microstructure&metric=fa_mean"
    )
    blocks = client.get(
        "/api/v1/networks/analysis/blocks"
        "?scheme=functional&metric=mean_weight"
    )
    assert catalog.status_code == 200
    assert distribution.status_code == 200
    assert blocks.status_code == 200
    assert len(catalog.json()["data"]["schemes"]) == 2
    values = distribution.json()["data"]
    assert len(values["rows"]) == 4804
    assert len(values["statistics"]) == 10
    assert {
        row["group"] for row in values["rows"]
    } == {"CN", "MCI", "AD"}
    block_payload = blocks.json()["data"]
    assert len(block_payload["networks"]) == 10
    assert {
        row["group"] for row in block_payload["rows"]
    } == {"CN", "MCI", "AD"}


def test_coupling_aal_ranking_is_subject_level_and_fdr_corrected(client):
    response = client.get(
        "/api/v1/coupling-aal/ranking"
        "?topology_metric=strength"
        "&microstructure_metric=fa_mean"
        "&value_metric=coupling_index"
        "&group_a=CN&group_b=MCI"
    )
    assert response.status_code == 200
    rows = response.json()["data"]["rows"]
    assert len(rows) == 166
    assert rows[0]["atlas_label"] == "Precuneus_R"
    assert rows[0]["p_value"] == pytest.approx(
        0.0077081650591445845
    )
    assert rows[0]["q_value"] == pytest.approx(
        0.9230610341089435
    )
    assert rows[0]["n_CN"] > 100
    assert rows[0]["n_MCI"] > 100


def test_findings_endpoint_preserves_claim_boundaries(client):
    response = client.get("/api/v1/findings/summary")
    assert response.status_code == 200
    payload = response.json()["data"]
    assert len(payload["cards"]) == 4
    assert len(payload["catalog"]) == 210
    assert [row["q"] for row in payload["catalog"]] == sorted(
        row["q"] for row in payload["catalog"]
    )
    assert "exploratory hypotheses" in payload["contract"]["claim_scope"]
    assert all(card["caveat"] for card in payload["cards"])


def test_frontend_production_build_is_shadow_scoped():
    dist = Path(
        "/home/ec2-user/exp/apps/connectome_web/dist"
    )
    index = (dist / "index.html").read_text(encoding="utf-8")
    assert "/next/assets/" in index
    assert "<div id=\"root\"></div>" in index
    assert list((dist / "assets").glob("*.js"))
    assert list((dist / "assets").glob("*.css"))
    assert not list((dist / "assets").glob("*.map"))
