from __future__ import annotations

import hashlib
import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from connectome_dashboard_core.artifacts import (
    artifact_by_id,
    artifact_content,
    artifact_inventory,
    artifact_path,
    section_artifacts,
    subject_metric_catalog,
    subject_metric_values,
    table_payload,
)
from connectome_dashboard_core.advanced_analytics import (
    coupling_aal_catalog,
    coupling_aal_ranking,
    ml_task_catalog,
    ml_task_data,
    network_affectedness,
    network_blocks,
    network_catalog,
    network_distribution,
    novel_findings_summary,
)
from connectome_dashboard_core.connectomes import (
    ALLOWED_MATRIX_WEIGHTS,
    connectome_subjects,
    matrix_edges,
    matrix_payload,
    matrix_summary,
)
from connectome_dashboard_core.lr_sr import (
    EDR_MEASURES,
    edr_exception_count_summary,
    edr_exception_example_data,
    edr_exception_inference,
    edr_exception_subject_table,
    model_feature_eda,
    model_feature_family_metric_inventory,
    network_mapping_summary,
    tract_length_distribution_data,
    tract_length_range_summary,
)
from connectome_dashboard_core.network_ranked import (
    network_measure_ranked_table,
)
from connectome_dashboard_core.exception_ml import (
    exception_specificity_payload,
)
from connectome_dashboard_core.pipeline import (
    legacy_pipeline_status,
    pipeline_subjects,
    release_status,
)
from connectome_dashboard_core.sections import SECTIONS, section_or_raise
from connectome_dashboard_core.section_analytics import (
    demographics_summary,
    node_catalog,
    node_cross_comparison,
    node_metric_profile,
    node_pairwise_ranking,
    node_values,
    subject_metric_analysis,
    subject_metric_profile,
)
from connectome_dashboard_core.serialization import json_safe, records
from connectome_dashboard_core.settings import Settings, get_settings

from .schemas import ApiEnvelope, ErrorResponse, HealthResponse


LOGGER = logging.getLogger("connectome_api")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
SCHEMA_VERSION = "1.0"
API_PREFIX = "/api/v1"
APP_FILE = Path(__file__).resolve()
REFERENCE_FILE = (
    Path("/home/ec2-user/exp")
    / "research_audit/outputs/connectome_dashboard_reference_v1/reference.json"
)


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _provenance(
    *,
    sources: list[Path] | None = None,
    calculation: str | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    if calculation:
        result["calculation"] = calculation
    if sources:
        result["sources"] = [
            {
                "path": str(path),
                "available": path.is_file(),
                "mtime_ns": (
                    path.stat().st_mtime_ns if path.is_file() else None
                ),
            }
            for path in sources
        ]
    return result


def envelope(
    data: Any,
    *,
    warnings: list[str] | None = None,
    sources: list[Path] | None = None,
    calculation: str | None = None,
) -> ApiEnvelope:
    return ApiEnvelope(
        schema_version=SCHEMA_VERSION,
        data=json_safe(data),
        warnings=warnings or [],
        provenance=_provenance(
            sources=sources,
            calculation=calculation,
        ),
    )


app = FastAPI(
    title="Structural Connectome Dashboard API",
    description=(
        "Read-only, versioned API over the existing structural-connectome "
        "analysis artifacts. It does not write scientific outputs."
    ),
    version=SCHEMA_VERSION,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)


@app.middleware("http")
async def request_boundary(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
    started = time.perf_counter()
    if request.method not in {"GET", "HEAD", "OPTIONS"}:
        return JSONResponse(
            status_code=405,
            content={
                "detail": "This service is read-only",
                "request_id": request_id,
            },
            headers={"x-request-id": request_id},
        )
    response = await call_next(request)
    response.headers["x-request-id"] = request_id
    response.headers["x-content-type-options"] = "nosniff"
    response.headers["cache-control"] = (
        "no-store"
        if request.url.path.startswith(f"{API_PREFIX}/pipeline")
        else "private, max-age=60"
    )
    elapsed_ms = (time.perf_counter() - started) * 1000
    LOGGER.info(
        "request_id=%s method=%s path=%s status=%s elapsed_ms=%.1f",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    return response


@app.exception_handler(KeyError)
async def key_error_handler(request: Request, error: KeyError):
    return JSONResponse(
        status_code=404,
        content={
            "detail": str(error).strip("'"),
            "request_id": request.headers.get("x-request-id"),
        },
    )


@app.exception_handler(FileNotFoundError)
async def not_found_handler(request: Request, error: FileNotFoundError):
    return JSONResponse(
        status_code=404,
        content={
            "detail": str(error),
            "request_id": request.headers.get("x-request-id"),
        },
    )


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, error: ValueError):
    return JSONResponse(
        status_code=422,
        content={
            "detail": str(error),
            "request_id": request.headers.get("x-request-id"),
        },
    )


@app.get(
    f"{API_PREFIX}/health/live",
    response_model=HealthResponse,
    tags=["health"],
)
def health_live() -> HealthResponse:
    return HealthResponse(
        status="ok",
        service="connectome-api",
        schema_version=SCHEMA_VERSION,
    )


@app.get(
    f"{API_PREFIX}/health/ready",
    response_model=HealthResponse,
    responses={503: {"model": ErrorResponse}},
    tags=["health"],
)
def health_ready() -> HealthResponse:
    settings = get_settings()
    required = (
        settings.analysis_root / "00_master/master_cohort.csv",
        settings.connectomes_root,
        REFERENCE_FILE,
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise HTTPException(
            status_code=503,
            detail=f"required sources unavailable: {missing}",
        )
    return HealthResponse(
        status="ready",
        service="connectome-api",
        schema_version=SCHEMA_VERSION,
    )


@app.get(
    f"{API_PREFIX}/metadata/app",
    response_model=ApiEnvelope,
    tags=["metadata"],
)
def metadata_app() -> ApiEnvelope:
    settings = get_settings()
    return envelope(
        {
            "name": "Structural Connectome Dashboard",
            "architecture": "FastAPI + React shadow migration",
            "read_only": True,
            "analysis_section_n": len(SECTIONS),
            "matrix_weights": list(ALLOWED_MATRIX_WEIGHTS),
            "live_reference": "Streamlit remains authoritative during parity",
            "api_docs": "/api/docs",
        },
        sources=[settings.analysis_root / "00_master/master_cohort.csv"],
    )


@app.get(
    f"{API_PREFIX}/metadata/provenance",
    response_model=ApiEnvelope,
    tags=["metadata"],
)
def metadata_provenance() -> ApiEnvelope:
    settings = get_settings()
    master = settings.analysis_root / "00_master/master_cohort.csv"
    return envelope(
        {
            "reference_fixture": str(REFERENCE_FILE),
            "reference_sha256": _sha256(REFERENCE_FILE),
            "api_implementation_sha256": _sha256(APP_FILE),
            "analysis_root": str(settings.analysis_root),
            "connectomes_root": str(settings.connectomes_root),
            "master_sha256": _sha256(master),
            "artifact_count": len(artifact_inventory(settings)),
        },
        sources=[REFERENCE_FILE, master],
    )


@app.get(
    f"{API_PREFIX}/sections",
    response_model=ApiEnvelope,
    tags=["sections"],
)
def sections() -> ApiEnvelope:
    settings = get_settings()
    data: list[dict[str, Any]] = []
    for section in SECTIONS:
        item = section.to_dict()
        item["artifact_count"] = len(
            section_artifacts(settings, section)
        )
        item["metrics"] = subject_metric_catalog(
            settings,
            section,
        )
        data.append(item)
    return envelope(data)


@app.get(
    f"{API_PREFIX}/sections/{{section_id}}/artifacts",
    response_model=ApiEnvelope,
    tags=["sections"],
)
def artifacts_for_section(section_id: str) -> ApiEnvelope:
    settings = get_settings()
    section = section_or_raise(section_id)
    return envelope(
        [
            artifact.to_dict()
            for artifact in section_artifacts(settings, section)
        ]
    )


@app.get(
    f"{API_PREFIX}/sections/{{section_id}}/metrics",
    response_model=ApiEnvelope,
    tags=["sections"],
)
def metrics_for_section(section_id: str) -> ApiEnvelope:
    settings = get_settings()
    section = section_or_raise(section_id)
    return envelope(subject_metric_catalog(settings, section))


@app.get(
    f"{API_PREFIX}/sections/{{section_id}}/metrics/{{metric}}",
    response_model=ApiEnvelope,
    tags=["sections"],
)
def metric_values(section_id: str, metric: str) -> ApiEnvelope:
    settings = get_settings()
    section = section_or_raise(section_id)
    return envelope(
        subject_metric_values(settings, section, metric),
        calculation="source subject table values; no client-side recoding",
    )


@app.get(
    f"{API_PREFIX}/sections/{{section_id}}/analysis/profile",
    response_model=ApiEnvelope,
    tags=["section-analysis"],
)
def section_analysis_profile(section_id: str) -> ApiEnvelope:
    settings = get_settings()
    section = section_or_raise(section_id)
    return envelope(
        {
            "section": section.to_dict(),
            "subject": subject_metric_profile(settings, section),
            "node": node_metric_profile(settings, section),
            "statistical_contract": {
                "group_order": ["CN", "MCI", "AD"],
                "visual_filter": (
                    "Per-group 1.5-IQR filtering is display-only."
                ),
                "subject_inference": (
                    "Use the stored descriptives and pairwise output files "
                    "when present; no client-side hypothesis tests."
                ),
                "node_inference": (
                    "Two-sided Mann-Whitney U per node with "
                    "Benjamini-Hochberg correction within each contrast."
                ),
            },
        }
    )


@app.get(
    f"{API_PREFIX}/sections/{{section_id}}/analysis/subject/{{metric}}",
    response_model=ApiEnvelope,
    tags=["section-analysis"],
)
def section_subject_analysis(
    section_id: str,
    metric: str,
) -> ApiEnvelope:
    settings = get_settings()
    section = section_or_raise(section_id)
    payload = subject_metric_analysis(settings, section, metric)
    sources = [
        settings.analysis_root / source
        for source in payload.pop("sources", [])
    ]
    return envelope(
        payload,
        sources=sources,
        calculation=(
            "authoritative subject rows; stored descriptives and pairwise "
            "statistics are used when available; IQR filtering is visual only"
        ),
    )


@app.get(
    f"{API_PREFIX}/sections/{{section_id}}/analysis/nodes/{{metric}}",
    response_model=ApiEnvelope,
    tags=["section-analysis"],
)
def section_node_catalog(
    section_id: str,
    metric: str,
) -> ApiEnvelope:
    settings = get_settings()
    section = section_or_raise(section_id)
    return envelope(
        node_catalog(settings, section, metric),
        calculation=(
            "AAL3 node inventory with stored Kruskal-Wallis/FDR columns "
            "when the section supplies them"
        ),
    )


@app.get(
    (
        f"{API_PREFIX}/sections/{{section_id}}/analysis/nodes/"
        "{metric}/ranking"
    ),
    response_model=ApiEnvelope,
    tags=["section-analysis"],
)
def section_node_ranking(
    section_id: str,
    metric: str,
    group_a: Literal["CN", "MCI"] = Query("CN"),
    group_b: Literal["MCI", "AD"] = Query("MCI"),
) -> ApiEnvelope:
    settings = get_settings()
    section = section_or_raise(section_id)
    return envelope(
        node_pairwise_ranking(
            settings,
            section,
            metric,
            group_a,
            group_b,
        ),
        calculation=(
            "two-sided Mann-Whitney U for each AAL3 node, Cliff delta, "
            "and Benjamini-Hochberg correction across nodes"
        ),
    )


@app.get(
    (
        f"{API_PREFIX}/sections/{{section_id}}/analysis/nodes/"
        "{metric}/repeated"
    ),
    response_model=ApiEnvelope,
    tags=["section-analysis"],
)
def section_node_repeated(
    section_id: str,
    metric: str,
    top_n: int = Query(20, ge=1, le=166),
) -> ApiEnvelope:
    settings = get_settings()
    section = section_or_raise(section_id)
    return envelope(
        node_cross_comparison(
            settings,
            section,
            metric,
            top_n,
        ),
        calculation=(
            "repeat frequency across the three pairwise Top-N node "
            "rankings; FDR status is retained explicitly"
        ),
    )


@app.get(
    (
        f"{API_PREFIX}/sections/{{section_id}}/analysis/nodes/"
        "{metric}/{node}"
    ),
    response_model=ApiEnvelope,
    tags=["section-analysis"],
)
def section_node_values(
    section_id: str,
    metric: str,
    node: int,
) -> ApiEnvelope:
    settings = get_settings()
    section = section_or_raise(section_id)
    return envelope(
        node_values(settings, section, metric, node),
        calculation=(
            "authoritative subject-level node values; per-group IQR "
            "filtering is visual only"
        ),
    )


@app.get(
    f"{API_PREFIX}/demographics/summary",
    response_model=ApiEnvelope,
    tags=["demographics"],
)
def demographics_summary_endpoint() -> ApiEnvelope:
    settings = get_settings()
    payload = demographics_summary(settings)
    sources = [
        settings.analysis_root / source
        for source in payload.pop("sources", [])
    ]
    return envelope(
        payload,
        sources=sources,
        calculation=(
            "authoritative master-cohort rows joined only to recorded "
            "completeness and stored demographic inference outputs"
        ),
    )


@app.get(
    f"{API_PREFIX}/artifacts/{{artifact_id}}/table",
    response_model=ApiEnvelope,
    tags=["artifacts"],
)
def artifact_table(
    artifact_id: str,
    offset: int = Query(0, ge=0),
    limit: int = Query(200, ge=1, le=5_000),
) -> ApiEnvelope:
    settings = get_settings()
    artifact = artifact_by_id(settings, artifact_id)
    return envelope(
        table_payload(settings, artifact, offset, limit),
        sources=[artifact_path(settings, artifact)],
    )


@app.get(
    f"{API_PREFIX}/artifacts/{{artifact_id}}/content",
    response_model=ApiEnvelope,
    tags=["artifacts"],
)
def artifact_text_content(artifact_id: str) -> ApiEnvelope:
    settings = get_settings()
    artifact = artifact_by_id(settings, artifact_id)
    return envelope(
        artifact_content(settings, artifact),
        sources=[artifact_path(settings, artifact)],
    )


@app.get(
    f"{API_PREFIX}/artifacts/{{artifact_id}}/download",
    response_class=FileResponse,
    tags=["artifacts"],
)
def artifact_download(artifact_id: str) -> FileResponse:
    settings = get_settings()
    artifact = artifact_by_id(settings, artifact_id)
    path = artifact_path(settings, artifact)
    return FileResponse(
        path=path,
        media_type=artifact.mime_type,
        filename=artifact.name,
    )


@app.get(
    f"{API_PREFIX}/cohort/summary",
    response_model=ApiEnvelope,
    tags=["cohort"],
)
def cohort_summary() -> ApiEnvelope:
    settings = get_settings()
    path = settings.analysis_root / "00_master/master_cohort.csv"
    frame = pd.read_csv(path)
    group_counts = (
        frame["group"].astype(str).value_counts().to_dict()
        if "group" in frame.columns
        else {}
    )
    return envelope(
        {
            "subjects": int(len(frame)),
            "group_counts": {
                str(key): int(value)
                for key, value in group_counts.items()
            },
            "columns": [str(column) for column in frame.columns],
            "diffusivity_subjects": (
                int(
                    pd.to_numeric(
                        frame.get("has_conn_fa_mean"),
                        errors="coerce",
                    )
                    .fillna(0)
                    .gt(0)
                    .sum()
                )
                if "has_conn_fa_mean" in frame.columns
                else None
            ),
        },
        sources=[path],
    )


@app.get(
    f"{API_PREFIX}/lr-sr/tract-length/summary",
    response_model=ApiEnvelope,
    tags=["lr-sr"],
)
def tract_length_summary() -> ApiEnvelope:
    settings = get_settings()
    _samples, summary = tract_length_distribution_data(settings)
    return envelope(
        records(summary),
        calculation=(
            "complete positive upper-triangle AAL3 len_mean edges; Tukey "
            "fences are descriptive and do not remove observations"
        ),
    )


@app.get(
    f"{API_PREFIX}/lr-sr/tract-length/ranges",
    response_model=ApiEnvelope,
    tags=["lr-sr"],
)
def tract_length_ranges() -> ApiEnvelope:
    result = tract_length_range_summary(get_settings())
    if not result:
        raise KeyError("LR/SR tract-length thresholds are unavailable")
    return envelope(
        result,
        calculation=(
            "stored LR/SR pooled-CN tertile thresholds plus the complete "
            "cohort positive-edge tract-length median"
        ),
    )


@app.get(
    f"{API_PREFIX}/lr-sr/tract-length/distribution",
    response_model=ApiEnvelope,
    tags=["lr-sr"],
)
def tract_length_distribution(
    panel: Literal["Overall", "CN", "MCI", "AD"] = "Overall",
) -> ApiEnvelope:
    settings = get_settings()
    samples, summary = tract_length_distribution_data(settings)
    values = samples.loc[
        samples["Panel"].astype(str).eq(panel),
        "Length_mm",
    ].to_numpy(dtype=float)
    summary_row = summary[
        summary["Panel"].astype(str).eq(panel)
    ]
    return envelope(
        {
            "panel": panel,
            "sample_seed": (
                20260507
                if panel == "Overall"
                else 20260507 + ("CN", "MCI", "AD").index(panel)
            ),
            "display_sample_n": int(values.size),
            "values": values.tolist(),
            "full_summary": (
                records(summary_row)[0]
                if not summary_row.empty
                else None
            ),
        },
        calculation=(
            "deterministic display sample; full percentiles and counts are "
            "calculated from all loaded edges"
        ),
    )


@app.get(
    f"{API_PREFIX}/lr-sr/exceptions/config",
    response_model=ApiEnvelope,
    tags=["lr-sr"],
)
def exception_config() -> ApiEnvelope:
    settings = get_settings()
    _subject, thresholds = edr_exception_subject_table(
        settings,
        "fd_sum",
    )
    return envelope(
        {
            "measures": EDR_MEASURES,
            "default_measure": "fd_sum",
            "thresholds": thresholds,
            "rule": (
                "Within each subject and adaptive length bin, flag selected "
                "positive edge measure > bin mean + 3 sample SD. One-sided."
            ),
            "unit_of_inference": (
                "one subject median per edge class for group inference"
            ),
        },
        calculation="current Streamlit EDR definition, extracted unchanged",
    )


@app.get(
    f"{API_PREFIX}/lr-sr/exceptions/summary",
    response_model=ApiEnvelope,
    tags=["lr-sr"],
)
def exception_summary(
    measure: Literal[
        "fd_sum",
        "count",
        "count_invnodevol",
    ] = "fd_sum",
) -> ApiEnvelope:
    settings = get_settings()
    summary, thresholds, _subject = edr_exception_count_summary(
        settings,
        measure,
    )
    return envelope(
        {
            "measure": measure,
            "measure_definition": EDR_MEASURES[measure],
            "thresholds": thresholds,
            "rows": records(summary),
        },
        calculation=(
            "within-subject adaptive length bins; mean plus three sample SD"
        ),
    )


@app.get(
    f"{API_PREFIX}/lr-sr/exceptions/inference",
    response_model=ApiEnvelope,
    tags=["lr-sr"],
)
def exception_inference(
    measure: Literal[
        "fd_sum",
        "count",
        "count_invnodevol",
    ] = "fd_sum",
    table: Literal["requested", "full"] = "requested",
) -> ApiEnvelope:
    settings = get_settings()
    plot, results, metadata = edr_exception_inference(
        settings,
        measure,
    )
    if table == "requested":
        keep = (
            results["Requested 4-test Holm p"].notna()
            if "Requested 4-test Holm p" in results.columns
            else pd.Series(False, index=results.index)
        )
        visible = results[keep].copy()
    else:
        visible = results
    interaction = results[
        results["Question"].eq(
            "Edge-class × diagnosis interaction"
        )
    ].copy()
    return envelope(
        {
            "measure": measure,
            "table_scope": table,
            "rows": records(visible),
            "interaction_rows": records(interaction),
            "plot_rows": records(plot),
            "metadata": metadata,
            "limitations": [
                "Exploratory subject-level tests",
                "No adjustment for age, sex, acquisition, or site",
                (
                    "The paired exception contrast is expected by "
                    "construction because exceptions are high-strength "
                    "within their length bin"
                ),
                (
                    "A supported edge-class by diagnosis interaction is a "
                    "candidate incremental signal, not a validated biomarker"
                ),
            ],
        },
        calculation=(
            "Mann-Whitney U with Cliff delta; paired Wilcoxon with paired "
            "rank-biserial; edge-class by diagnosis interaction from the "
            "subject log-strength contrast; explicit Holm families exactly "
            "as documented"
        ),
    )


@app.get(
    f"{API_PREFIX}/lr-sr/exceptions/subjects",
    response_model=ApiEnvelope,
    tags=["lr-sr"],
)
def exception_subjects(
    measure: Literal[
        "fd_sum",
        "count",
        "count_invnodevol",
    ] = "fd_sum",
) -> ApiEnvelope:
    settings = get_settings()
    subject, _thresholds = edr_exception_subject_table(
        settings,
        measure,
    )
    columns = [
        column
        for column in (
            "subject_id",
            "group",
            "n_edges_valid",
            "n_exceptions",
            "lr_edge_count",
            "lr_exception_count",
            "lr_exception_pct",
            "lr_nonexception_measure_median",
            "lr_exception_measure_median",
        )
        if column in subject.columns
    ]
    return envelope(records(subject[columns]))


@app.get(
    f"{API_PREFIX}/lr-sr/exceptions/{{subject_id}}/example",
    response_model=ApiEnvelope,
    tags=["lr-sr"],
)
def exception_example(
    subject_id: str,
    measure: Literal[
        "fd_sum",
        "count",
        "count_invnodevol",
    ] = "fd_sum",
) -> ApiEnvelope:
    settings = get_settings()
    points, curve, metadata = edr_exception_example_data(
        settings,
        subject_id,
        measure,
    )
    if not metadata:
        raise KeyError(
            f"EDR example unavailable for subject {subject_id}"
        )
    return envelope(
        {
            "measure": measure,
            "metadata": metadata,
            "points": records(points),
            "adaptive_bins": records(curve),
        },
        calculation=(
            "all displayed exception points plus deterministic candidate "
            "sample; threshold is mean plus three sample SD in each bin"
        ),
    )


@app.get(
    f"{API_PREFIX}/models/features/families",
    response_model=ApiEnvelope,
    tags=["models"],
)
def model_feature_families() -> ApiEnvelope:
    settings = get_settings()
    feature_eda, family_summary, subject_n = model_feature_eda(settings)
    inventory = model_feature_family_metric_inventory(
        feature_eda,
        family_summary,
    )
    return envelope(
        {
            "subjects": subject_n,
            "candidate_features": int(len(feature_eda)),
            "families": records(family_summary),
            "metric_inventory": records(inventory),
        }
    )


@app.get(
    f"{API_PREFIX}/models/features/eda",
    response_model=ApiEnvelope,
    tags=["models"],
)
def model_feature_eda_endpoint(
    offset: int = Query(0, ge=0),
    limit: int = Query(200, ge=1, le=2_500),
    family: str | None = None,
    status: str | None = None,
    search: str | None = None,
) -> ApiEnvelope:
    settings = get_settings()
    feature_eda, _summary, subject_n = model_feature_eda(settings)
    filtered = feature_eda
    if family:
        filtered = filtered[
            filtered["Dashboard family"].astype(str).eq(family)
        ]
    if status:
        filtered = filtered[
            filtered["Model status"].astype(str).eq(status)
        ]
    if search:
        filtered = filtered[
            filtered["Feature"]
            .astype(str)
            .str.contains(search, case=False, regex=False)
        ]
    page = filtered.iloc[offset : offset + limit]
    return envelope(
        {
            "subjects": subject_n,
            "offset": offset,
            "limit": limit,
            "total_rows": int(len(filtered)),
            "columns": [str(column) for column in filtered.columns],
            "rows": records(page),
        }
    )


@app.get(
    f"{API_PREFIX}/models/tasks",
    response_model=ApiEnvelope,
    tags=["models"],
)
def model_tasks() -> ApiEnvelope:
    settings = get_settings()
    return envelope(ml_task_catalog(settings))


@app.get(
    f"{API_PREFIX}/models/tasks/{{task}}",
    response_model=ApiEnvelope,
    tags=["models"],
)
def model_task(task: str) -> ApiEnvelope:
    settings = get_settings()
    payload = ml_task_data(settings, task)
    sources = [
        settings.analysis_root / source
        for source in payload.pop("sources", [])
    ]
    return envelope(
        payload,
        sources=sources,
        calculation=(
            "recorded cross-validation outputs; models are not refitted "
            "by the dashboard"
        ),
    )


@app.get(
    f"{API_PREFIX}/networks/mapping",
    response_model=ApiEnvelope,
    tags=["networks"],
)
def network_mapping(detail: bool = False) -> ApiEnvelope:
    settings = get_settings()
    mapping = network_mapping_summary(settings)
    return envelope(
        {
            "status": mapping["status"],
            "functional": records(mapping["functional"]),
            "anatomical": records(mapping["anatomical"]),
            "mapping": (
                records(mapping["mapping"]) if detail else []
            ),
        },
        calculation=(
            "analysis-defined Yeo-7-inspired crosswalk; mapping-sensitive "
            "and not native voxelwise AAL3-to-Yeo overlap"
        ),
    )


@app.get(
    f"{API_PREFIX}/networks/exception-specificity",
    response_model=ApiEnvelope,
    tags=["networks"],
)
def network_exception_specificity() -> ApiEnvelope:
    result = exception_specificity_payload(get_settings())
    return envelope(
        result,
        calculation=(
            "precomputed Part A (like-for-like |Cliff's delta| tier gains, "
            "exception-architecture group stats with ADNI-3 sensitivity) and "
            "Part B (repeated 5x5-fold CV incremental ladder F0..F4 for "
            "nearest-visit MMSE regression and binary CDR, ElasticNet/Huber + "
            "HistGradientBoosting + ExtraTrees, permutation-tested; outlier "
            "one outlier subject excluded; no model is fitted at request time)"
        ),
    )


@app.get(
    f"{API_PREFIX}/networks/measure-ranked",
    response_model=ApiEnvelope,
    tags=["networks"],
)
def network_measure_ranked(
    contrast: Literal["cn_mci", "cn_ad", "mci_ad"] = "cn_ad",
) -> ApiEnvelope:
    result = network_measure_ranked_table(get_settings(), contrast)
    if result.get("status") != "ok":
        raise KeyError("network x measure ranked table sources are unavailable")
    return envelope(
        result,
        calculation=(
            f"per-measure network ranking by {contrast} |Cliff's delta|; graph/"
            "microstructure/coupling cells reuse the precomputed section-19 "
            "group statistics (Brunner-Munzel BH-FDR q); EDR LR-exception "
            "burden/strength are aggregated from the node-level section-17 "
            "artifact via the functional-network crosswalk and tested "
            "CN-vs-AD with the same statistics (exploratory; exception "
            "identification follows Chakraborty et al. mean+3SD-per-distance "
            "with CN-referenced Q1/Q3 range cutoffs)"
        ),
    )


@app.get(
    f"{API_PREFIX}/networks/analysis/catalog",
    response_model=ApiEnvelope,
    tags=["networks"],
)
def network_analysis_catalog() -> ApiEnvelope:
    settings = get_settings()
    return envelope(network_catalog(settings))


@app.get(
    f"{API_PREFIX}/networks/analysis/distribution",
    response_model=ApiEnvelope,
    tags=["networks"],
)
def network_analysis_distribution(
    scheme: Literal["functional", "anatomical"] = "functional",
    family: Literal["microstructure", "graph", "coupling"] = (
        "microstructure"
    ),
    metric: str = Query(..., min_length=1),
) -> ApiEnvelope:
    settings = get_settings()
    payload = network_distribution(
        settings,
        scheme,
        family,
        metric,
    )
    sources = [
        settings.analysis_root / source
        for source in payload.pop("sources", [])
    ]
    return envelope(
        payload,
        sources=sources,
        calculation=(
            "precomputed subject-by-network values and recorded "
            "network-wise statistical output"
        ),
    )


@app.get(
    f"{API_PREFIX}/networks/analysis/affectedness",
    response_model=ApiEnvelope,
    tags=["networks"],
)
def network_analysis_affectedness(
    scheme: Literal["functional", "anatomical"] = "functional",
    family: str | None = None,
) -> ApiEnvelope:
    settings = get_settings()
    return envelope(
        network_affectedness(settings, scheme, family),
        calculation=(
            "recorded CN-AD effect sizes and FDR results, ranked by "
            "absolute Cliff delta"
        ),
    )


@app.get(
    f"{API_PREFIX}/networks/analysis/blocks",
    response_model=ApiEnvelope,
    tags=["networks"],
)
def network_analysis_blocks(
    scheme: Literal["functional", "anatomical"] = "functional",
    metric: Literal["density", "mean_weight"] = "mean_weight",
) -> ApiEnvelope:
    settings = get_settings()
    payload = network_blocks(settings, scheme, metric)
    sources = [
        settings.analysis_root / source
        for source in payload.pop("sources", [])
    ]
    return envelope(
        payload,
        sources=sources,
        calculation=(
            "recorded group-mean within/between-network structural "
            "connectivity blocks"
        ),
    )


@app.get(
    f"{API_PREFIX}/coupling-aal/catalog",
    response_model=ApiEnvelope,
    tags=["coupling-aal"],
)
def coupling_aal_catalog_endpoint() -> ApiEnvelope:
    settings = get_settings()
    return envelope(coupling_aal_catalog(settings))


@app.get(
    f"{API_PREFIX}/coupling-aal/ranking",
    response_model=ApiEnvelope,
    tags=["coupling-aal"],
)
def coupling_aal_ranking_endpoint(
    topology_metric: Literal["strength", "degree", "nodal_eff"] = (
        "strength"
    ),
    microstructure_metric: Literal[
        "fa_mean",
        "md_mean",
        "ad_mean",
        "rd_mean",
    ] = "fa_mean",
    value_metric: Literal[
        "coupling_index",
        "micro_topology_ratio",
    ] = "coupling_index",
    group_a: Literal["CN", "MCI"] = "CN",
    group_b: Literal["MCI", "AD"] = "MCI",
) -> ApiEnvelope:
    settings = get_settings()
    payload = coupling_aal_ranking(
        settings,
        topology_metric,
        microstructure_metric,
        value_metric,
        group_a,
        group_b,
    )
    sources: list[Path] = []
    for source in payload.pop("sources", []):
        path = Path(source)
        sources.append(
            path
            if path.is_absolute()
            else settings.analysis_root / path
        )
    return envelope(
        payload,
        sources=sources,
        calculation=(
            "within-subject regional z-score product or ratio; "
            "Mann-Whitney U, Cliff delta and BH-FDR across AAL3 nodes"
        ),
    )


@app.get(
    f"{API_PREFIX}/findings/summary",
    response_model=ApiEnvelope,
    tags=["findings"],
)
def findings_summary() -> ApiEnvelope:
    settings = get_settings()
    payload = novel_findings_summary(settings)
    sources = [
        settings.analysis_root / source
        for source in payload.pop("sources", [])
    ]
    return envelope(
        payload,
        sources=sources,
        calculation=(
            "recorded FDR-supported findings and curated "
            "literature-grounded synthesis cards"
        ),
    )


@app.get(
    f"{API_PREFIX}/connectomes/subjects",
    response_model=ApiEnvelope,
    tags=["connectomes"],
)
def matrix_subjects() -> ApiEnvelope:
    settings = get_settings()
    return envelope(records(connectome_subjects(settings)))


@app.get(
    f"{API_PREFIX}/connectomes/{{subject_id}}/{{weight}}/summary",
    response_model=ApiEnvelope,
    tags=["connectomes"],
)
def connectome_summary(subject_id: str, weight: str) -> ApiEnvelope:
    return envelope(
        matrix_summary(get_settings(), subject_id, weight)
    )


@app.get(
    f"{API_PREFIX}/connectomes/{{subject_id}}/{{weight}}/matrix",
    response_model=ApiEnvelope,
    tags=["connectomes"],
)
def connectome_matrix(subject_id: str, weight: str) -> ApiEnvelope:
    return envelope(
        matrix_payload(get_settings(), subject_id, weight)
    )


@app.get(
    f"{API_PREFIX}/connectomes/{{subject_id}}/{{weight}}/edges",
    response_model=ApiEnvelope,
    tags=["connectomes"],
)
def connectome_edges(
    subject_id: str,
    weight: str,
    positive_only: bool = True,
    offset: int = Query(0, ge=0),
    limit: int = Query(500, ge=1, le=5_000),
) -> ApiEnvelope:
    return envelope(
        matrix_edges(
            get_settings(),
            subject_id,
            weight,
            positive_only=positive_only,
            offset=offset,
            limit=limit,
        )
    )


@app.get(
    f"{API_PREFIX}/pipeline/release-status",
    response_model=ApiEnvelope,
    tags=["pipeline"],
)
def pipeline_release_status(detail: bool = False) -> ApiEnvelope:
    result = release_status(get_settings())
    warnings = result.pop("warnings", [])
    if not detail:
        result["subject_rows"] = []
        topology = result.get("topology")
        if isinstance(topology, dict):
            result["topology"] = {
                key: value
                for key, value in topology.items()
                if key
                not in {
                    "canaries",
                    "production",
                    "records",
                }
            }
    return envelope(result, warnings=warnings)


@app.get(
    f"{API_PREFIX}/pipeline/live-status",
    response_model=ApiEnvelope,
    tags=["pipeline"],
)
def pipeline_live_status() -> ApiEnvelope:
    result = legacy_pipeline_status(get_settings())
    warnings = result.pop("warnings", [])
    return envelope(result, warnings=warnings)


@app.get(
    f"{API_PREFIX}/pipeline/subjects",
    response_model=ApiEnvelope,
    tags=["pipeline"],
)
def pipeline_subject_table(
    source: Literal["hcp379", "legacy", "recovery-ledger"] = "hcp379",
    offset: int = Query(0, ge=0),
    limit: int = Query(200, ge=1, le=5_000),
) -> ApiEnvelope:
    return envelope(
        pipeline_subjects(
            get_settings(),
            source=source,
            offset=offset,
            limit=limit,
        )
    )


FRONTEND_DIST = (
    Path(__file__).resolve().parents[1] / "connectome_web/dist"
)
if FRONTEND_DIST.is_dir():
    app.mount(
        "/next",
        StaticFiles(directory=FRONTEND_DIST, html=True),
        name="connectome-web",
    )
