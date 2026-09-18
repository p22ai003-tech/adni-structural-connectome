from __future__ import annotations

from functools import lru_cache
import math
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
import pandas as pd

from .settings import Settings
from .statistics import GROUP_ORDER, edr_lr_measure_inference


EDR_MEASURES: dict[str, dict[str, str]] = {
    "fd_sum": {
        "label": "SIFT2 edge strength (fd_sum) — paper/default",
        "short_label": "SIFT2 edge strength",
        "status": "PRIMARY / PAPER DEFINITION",
        "interpretation": (
            "SIFT2 fibre-density-weighted structural connection strength. "
            "This is the implemented paper/default EDR exception basis."
        ),
    },
    "count": {
        "label": "Streamline count — sensitivity view",
        "short_label": "Streamline count",
        "status": "SENSITIVITY VIEW",
        "interpretation": (
            "Raw assigned streamline count. It is tractography- and "
            "parcel-size-sensitive and is not substituted for the paper's "
            "fd_sum definition."
        ),
    },
    "count_invnodevol": {
        "label": "Node-volume-normalized count — sensitivity view",
        "short_label": "Node-volume-normalized count",
        "status": "SENSITIVITY VIEW",
        "interpretation": (
            "Streamline count scaled by inverse endpoint parcel volume. It is "
            "a sensitivity view, not the paper's primary EDR definition."
        ),
    },
}

_TRACT_CACHE_LOCK = Lock()
_EDR_CACHE_LOCK = Lock()
_FEATURE_CACHE_LOCK = Lock()


def _mtime(path: Path) -> int:
    try:
        return path.stat().st_mtime_ns
    except FileNotFoundError:
        return 0


def deterministic_sample(
    values: np.ndarray,
    max_values: int,
    seed: int,
) -> np.ndarray:
    values = values[np.isfinite(values)]
    if len(values) <= max_values:
        return values
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(values), size=max_values, replace=False)
    return values[np.sort(indices)]


def aal3_connectome_matrix_path(
    connectomes_dir: Path,
    subject_id: str,
    image_id: object,
    matrix_type: str,
) -> Path | None:
    candidates: list[Path] = []
    image_text = ""
    if image_id is not None and not pd.isna(image_id):
        try:
            image_text = str(int(float(image_id)))
        except (TypeError, ValueError):
            image_text = str(image_id).strip().removeprefix("I")
    if image_text:
        candidates.append(
            connectomes_dir
            / f"SC_AAL166_{subject_id}_I{image_text}_{matrix_type}.csv"
        )
    candidates.extend(
        sorted(
            connectomes_dir.glob(
                f"SC_AAL166_{subject_id}_I*_{matrix_type}.csv"
            )
        )
    )
    return next((candidate for candidate in candidates if candidate.is_file()), None)


def tract_length_summary_row(
    panel: str,
    values: np.ndarray,
    n_subjects: int,
) -> dict[str, object]:
    q1 = float(np.percentile(values, 25))
    median_value = float(np.percentile(values, 50))
    q3 = float(np.percentile(values, 75))
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr
    low_outlier_n = int(np.count_nonzero(values < lower_fence))
    high_outlier_n = int(np.count_nonzero(values > upper_fence))
    outlier_n = low_outlier_n + high_outlier_n
    return {
        "Panel": panel,
        "n_subjects": int(n_subjects),
        "n_edges": int(values.size),
        "q1": q1,
        "median": median_value,
        "q3": q3,
        "iqr": iqr,
        "lower_fence": lower_fence,
        "upper_fence": upper_fence,
        "low_outlier_n": low_outlier_n,
        "high_outlier_n": high_outlier_n,
        "outlier_n": outlier_n,
        "outlier_pct": 100.0 * outlier_n / int(values.size),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def _tract_signature(settings: Settings) -> tuple[int, ...]:
    return (
        _mtime(settings.analysis_root / "00_master/master_cohort.csv"),
        _mtime(
            settings.analysis_root
            / "12_length_delay/lr_sr_subject_level_fd_sum_len_mean.csv"
        ),
        _mtime(settings.connectomes_root),
    )


def tract_length_distribution_data(
    settings: Settings,
    max_group_values: int = 50_000,
    max_overall_values: int = 80_000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # lru_cache is thread-safe for its internal dictionary but does not
    # coalesce simultaneous misses. The API requests summary and panels in
    # parallel, so serialize the expensive first fill to avoid reading the
    # 5.4-million-edge source more than once.
    with _TRACT_CACHE_LOCK:
        return _tract_length_distribution_data_cached(
            settings,
            max_group_values,
            max_overall_values,
            _tract_signature(settings),
        )


def tract_length_range_summary(settings: Settings) -> dict[str, Any]:
    threshold_path = (
        settings.analysis_root
        / "12_length_delay/lr_sr_thresholds.csv"
    )
    if not threshold_path.is_file():
        return {}
    threshold_frame = pd.read_csv(threshold_path)
    if threshold_frame.empty:
        return {}
    threshold = threshold_frame.iloc[0]
    short_max = float(threshold["short_max_mm"])
    medium_max = float(threshold["medium_max_mm"])
    _samples, summary = tract_length_distribution_data(settings)
    overall = summary[
        summary["Panel"].astype(str).eq("Overall")
    ]
    population_median = (
        float(overall.iloc[0]["median"])
        if not overall.empty
        else math.nan
    )
    return {
        "population_median_mm": population_median,
        "short_max_mm": short_max,
        "medium_max_mm": medium_max,
        "source_group": str(threshold.get("source_group", "")),
        "rule": str(threshold.get("rule", "")),
        "ranges": [
            {
                "code": "SR",
                "label": f"0 < L ≤ {short_max:.2f} mm",
            },
            {
                "code": "MR",
                "label": (
                    f"{short_max:.2f} < L ≤ {medium_max:.2f} mm"
                ),
            },
            {
                "code": "LR",
                "label": f"L > {medium_max:.2f} mm",
            },
        ],
    }


@lru_cache(maxsize=4)
def _tract_length_distribution_data_cached(
    settings: Settings,
    max_group_values: int,
    max_overall_values: int,
    signature: tuple[int, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    del signature
    master_path = settings.analysis_root / "00_master/master_cohort.csv"
    if not master_path.is_file() or not settings.connectomes_root.is_dir():
        return pd.DataFrame(), pd.DataFrame()
    master = pd.read_csv(master_path)
    required = {"subject_id", "group", "Image ID"}
    if not required.issubset(master.columns):
        return pd.DataFrame(), pd.DataFrame()
    lr_subject_path = (
        settings.analysis_root
        / "12_length_delay/lr_sr_subject_level_fd_sum_len_mean.csv"
    )
    if lr_subject_path.is_file():
        lr_subjects = pd.read_csv(
            lr_subject_path,
            usecols=lambda column: column in {"subject_id", "group"},
        )
        if {"subject_id", "group"}.issubset(lr_subjects.columns):
            lr_subjects = lr_subjects.rename(columns={"group": "lr_sr_group"})
            master = master.merge(lr_subjects, on="subject_id", how="inner")
            master["group"] = master["lr_sr_group"].fillna(master["group"])
    if "has_conn_len_mean" in master.columns:
        ready = (
            pd.to_numeric(master["has_conn_len_mean"], errors="coerce")
            .fillna(0)
            .gt(0)
        )
        master = master[ready].copy()
    master["group"] = master["group"].astype(str)
    master = master[master["group"].isin(GROUP_ORDER)].copy()

    values_by_group: dict[str, list[np.ndarray]] = {
        group: [] for group in GROUP_ORDER
    }
    subjects_by_group = {group: 0 for group in GROUP_ORDER}
    for _, row in master.iterrows():
        subject_id = str(row.get("subject_id", "")).strip()
        group = str(row.get("group", "")).strip()
        if not subject_id or group not in values_by_group:
            continue
        matrix_path = aal3_connectome_matrix_path(
            settings.connectomes_root,
            subject_id,
            row.get("Image ID"),
            "len_mean",
        )
        if matrix_path is None:
            continue
        try:
            matrix = np.loadtxt(matrix_path, delimiter=",")
        except (OSError, ValueError):
            continue
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            continue
        upper = matrix[np.triu_indices_from(matrix, k=1)]
        upper = upper[np.isfinite(upper) & (upper > 0)]
        if upper.size == 0:
            continue
        values_by_group[group].append(upper.astype(float, copy=False))
        subjects_by_group[group] += 1

    sample_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    all_values: list[np.ndarray] = []
    for group in GROUP_ORDER:
        if not values_by_group[group]:
            continue
        values = np.concatenate(values_by_group[group])
        all_values.append(values)
        sample = deterministic_sample(
            values,
            max_group_values,
            seed=20260507 + GROUP_ORDER.index(group),
        )
        sample_frames.append(
            pd.DataFrame(
                {"Panel": group, "Group": group, "Length_mm": sample}
            )
        )
        summary_rows.append(
            tract_length_summary_row(
                group,
                values,
                subjects_by_group[group],
            )
        )
    if all_values:
        overall = np.concatenate(all_values)
        sample = deterministic_sample(
            overall,
            max_overall_values,
            seed=20260507,
        )
        sample_frames.append(
            pd.DataFrame(
                {
                    "Panel": "Overall",
                    "Group": "Overall",
                    "Length_mm": sample,
                }
            )
        )
        summary_rows.insert(
            0,
            tract_length_summary_row(
                "Overall",
                overall,
                sum(subjects_by_group.values()),
            ),
        )
    if not sample_frames:
        return pd.DataFrame(), pd.DataFrame()
    return (
        pd.concat(sample_frames, ignore_index=True),
        pd.DataFrame(summary_rows),
    )


def edr_adaptive_exception_arrays(
    edge_lengths: np.ndarray,
    edge_measure: np.ndarray,
    min_bin_edges: int = 120,
    max_bins: int = 30,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    length_valid = np.isfinite(edge_lengths) & (edge_lengths > 0)
    valid = (
        length_valid
        & np.isfinite(edge_measure)
        & (edge_measure > 0)
    )
    bin_ids = np.full(edge_lengths.shape, -1, dtype=int)
    n_length = int(length_valid.sum())
    if n_length >= max(20, min_bin_edges):
        n_bins = int(np.clip(n_length // min_bin_edges, 3, max_bins))
        edges = np.unique(
            np.nanquantile(
                edge_lengths[length_valid],
                np.linspace(0.0, 1.0, n_bins + 1),
            )
        )
        if len(edges) >= 3:
            bin_ids[length_valid] = np.searchsorted(
                edges[1:-1],
                edge_lengths[length_valid],
                side="right",
            ).astype(int)
    exception = np.zeros(edge_measure.shape, dtype=bool)
    threshold = np.full(edge_measure.shape, np.nan, dtype=float)
    for bin_id in sorted(set(bin_ids[valid].tolist())):
        if bin_id < 0:
            continue
        mask = valid & (bin_ids == bin_id)
        values = edge_measure[mask]
        if values.size < 20:
            continue
        mean = float(np.nanmean(values))
        sd = (
            float(np.nanstd(values, ddof=1))
            if values.size > 1
            else np.nan
        )
        cut = mean + 3.0 * sd if np.isfinite(sd) else np.nan
        threshold[mask] = cut
        if np.isfinite(cut):
            exception[mask] = edge_measure[mask] > cut
    return exception, threshold, bin_ids


def _exception_signature(
    settings: Settings,
    measure_type: str,
) -> tuple[int, ...]:
    section = settings.analysis_root / "17_edr_exceptions"
    return (
        _mtime(settings.analysis_root / "00_master/master_cohort.csv"),
        _mtime(section / "edr_exception_thresholds.csv"),
        _mtime(
            section
            / "edr_exception_subject_level_fd_sum_len_mean.csv"
        ),
        _mtime(settings.connectomes_root),
        hash(measure_type),
    )


def edr_exception_subject_table(
    settings: Settings,
    measure_type: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if measure_type not in EDR_MEASURES:
        raise ValueError(f"unsupported EDR measure: {measure_type}")
    with _EDR_CACHE_LOCK:
        return _edr_exception_subject_table_cached(
            settings,
            measure_type,
            _exception_signature(settings, measure_type),
        )


@lru_cache(maxsize=8)
def _edr_exception_subject_table_cached(
    settings: Settings,
    measure_type: str,
    signature: tuple[int, ...],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    del signature
    section_root = settings.analysis_root / "17_edr_exceptions"
    threshold_path = section_root / "edr_exception_thresholds.csv"
    thresholds: dict[str, Any] = {}
    if threshold_path.is_file():
        threshold_frame = pd.read_csv(threshold_path)
        if not threshold_frame.empty:
            thresholds = threshold_frame.iloc[0].to_dict()

    precomputed: pd.DataFrame | None = None
    if measure_type == "fd_sum":
        subject_path = (
            section_root
            / "edr_exception_subject_level_fd_sum_len_mean.csv"
        )
        if not subject_path.is_file():
            return pd.DataFrame(), thresholds
        precomputed = pd.read_csv(subject_path)

    master_path = settings.analysis_root / "00_master/master_cohort.csv"
    if not master_path.is_file() or not settings.connectomes_root.is_dir():
        return (
            precomputed if precomputed is not None else pd.DataFrame(),
            thresholds,
        )
    short_max = pd.to_numeric(
        pd.Series([thresholds.get("short_max_mm")]),
        errors="coerce",
    ).iloc[0]
    long_min = pd.to_numeric(
        pd.Series([thresholds.get("long_min_mm")]),
        errors="coerce",
    ).iloc[0]
    if pd.isna(short_max) or pd.isna(long_min):
        return (
            precomputed if precomputed is not None else pd.DataFrame(),
            thresholds,
        )
    master = pd.read_csv(master_path)
    if not {"subject_id", "group", "Image ID"}.issubset(master.columns):
        return (
            precomputed if precomputed is not None else pd.DataFrame(),
            thresholds,
        )

    rows: list[dict[str, object]] = []
    for _, master_row in master.iterrows():
        subject_id = str(master_row["subject_id"]).strip()
        length_path = aal3_connectome_matrix_path(
            settings.connectomes_root,
            subject_id,
            master_row.get("Image ID"),
            "len_mean",
        )
        measure_path = aal3_connectome_matrix_path(
            settings.connectomes_root,
            subject_id,
            master_row.get("Image ID"),
            measure_type,
        )
        if length_path is None or measure_path is None:
            continue
        try:
            length_matrix = np.loadtxt(length_path, delimiter=",")
            measure_matrix = np.loadtxt(measure_path, delimiter=",")
        except (OSError, ValueError):
            continue
        if (
            length_matrix.ndim != 2
            or length_matrix.shape[0] != length_matrix.shape[1]
            or measure_matrix.shape != length_matrix.shape
        ):
            continue
        upper = np.triu_indices_from(length_matrix, k=1)
        lengths = length_matrix[upper]
        measures = measure_matrix[upper]
        valid = (
            np.isfinite(lengths)
            & (lengths > 0)
            & np.isfinite(measures)
            & (measures > 0)
        )
        exception, _threshold, _bin_ids = edr_adaptive_exception_arrays(
            lengths,
            measures,
        )
        row: dict[str, object] = {
            "subject_id": subject_id,
            "group": str(master_row["group"]),
            "n_edges_valid": int(valid.sum()),
            "n_exceptions": int(exception.sum()),
        }
        for code in ("sr", "mr", "lr"):
            if code == "sr":
                in_range = lengths <= float(short_max)
            elif code == "mr":
                in_range = (
                    (lengths > float(short_max))
                    & (lengths <= float(long_min))
                )
            else:
                in_range = lengths > float(long_min)
            candidates = valid & in_range
            exceptions = candidates & exception
            nonexceptions = candidates & ~exception
            candidate_n = int(candidates.sum())
            exception_n = int(exceptions.sum())
            row[f"{code}_edge_count"] = candidate_n
            row[f"{code}_exception_count"] = exception_n
            row[f"{code}_exception_pct"] = (
                exception_n / candidate_n if candidate_n else np.nan
            )
            row[f"{code}_nonexception_measure_median"] = (
                float(np.nanmedian(measures[nonexceptions]))
                if nonexceptions.any()
                else np.nan
            )
            row[f"{code}_exception_measure_median"] = (
                float(np.nanmedian(measures[exceptions]))
                if exceptions.any()
                else np.nan
            )
        rows.append(row)
    computed = pd.DataFrame(rows)
    if precomputed is None:
        return computed, thresholds

    median_columns = [
        "subject_id",
        *[
            f"{code}_{kind}_measure_median"
            for code in ("sr", "mr", "lr")
            for kind in ("nonexception", "exception")
        ],
    ]
    available = [
        column for column in median_columns if column in computed.columns
    ]
    if available == ["subject_id"]:
        return precomputed, thresholds
    return (
        precomputed.merge(
            computed[available],
            on="subject_id",
            how="left",
            validate="one_to_one",
        ),
        thresholds,
    )


def edr_exception_count_summary(
    settings: Settings,
    measure_type: str,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    subject, thresholds = edr_exception_subject_table(
        settings,
        measure_type,
    )
    if subject.empty or "group" not in subject.columns:
        return pd.DataFrame(), thresholds, subject
    rows: list[dict[str, object]] = []
    cohorts = [("Overall", subject)] + [
        (
            group,
            subject[subject["group"].astype(str).eq(group)].copy(),
        )
        for group in GROUP_ORDER
    ]
    for cohort, cohort_frame in cohorts:
        for code, label in (("sr", "SR"), ("mr", "MR"), ("lr", "LR")):
            candidate_column = f"{code}_edge_count"
            exception_column = f"{code}_exception_count"
            rate_column = f"{code}_exception_pct"
            if not {
                candidate_column,
                exception_column,
            }.issubset(cohort_frame.columns):
                continue
            candidates = pd.to_numeric(
                cohort_frame[candidate_column],
                errors="coerce",
            )
            exceptions = pd.to_numeric(
                cohort_frame[exception_column],
                errors="coerce",
            )
            rates = (
                pd.to_numeric(
                    cohort_frame[rate_column],
                    errors="coerce",
                )
                if rate_column in cohort_frame.columns
                else exceptions / candidates.replace(0, np.nan)
            )
            nonexception_measure = pd.to_numeric(
                cohort_frame.get(
                    f"{code}_nonexception_measure_median",
                    pd.Series(index=cohort_frame.index, dtype=float),
                ),
                errors="coerce",
            )
            exception_measure = pd.to_numeric(
                cohort_frame.get(
                    f"{code}_exception_measure_median",
                    pd.Series(index=cohort_frame.index, dtype=float),
                ),
                errors="coerce",
            )
            total_candidates = int(candidates.fillna(0).sum())
            total_exceptions = int(exceptions.fillna(0).sum())
            rows.append(
                {
                    "Cohort": cohort,
                    "Range": label,
                    "Cases": int(len(cohort_frame)),
                    "Candidate edges": total_candidates,
                    "Exception edges": total_exceptions,
                    "Median non-exception measure": float(
                        nonexception_measure.median()
                    ),
                    "Median exception measure": float(
                        exception_measure.median()
                    ),
                    "Pooled exception (%)": (
                        100.0 * total_exceptions / total_candidates
                        if total_candidates
                        else np.nan
                    ),
                    "Median exceptions / case": float(
                        exceptions.median()
                    ),
                    "Median subject rate (%)": float(
                        100.0 * rates.median()
                    ),
                }
            )
    return pd.DataFrame(rows), thresholds, subject


def edr_exception_inference(
    settings: Settings,
    measure_type: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    subject, _thresholds = edr_exception_subject_table(
        settings,
        measure_type,
    )
    return edr_lr_measure_inference(subject)


def edr_exception_example_data(
    settings: Settings,
    subject_id: str,
    measure_type: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if measure_type not in EDR_MEASURES:
        raise ValueError(f"unsupported EDR measure: {measure_type}")
    master_path = settings.analysis_root / "00_master/master_cohort.csv"
    if not master_path.is_file():
        return pd.DataFrame(), pd.DataFrame(), {}
    master = pd.read_csv(master_path)
    selected = master[
        master["subject_id"].astype(str).eq(str(subject_id))
    ]
    if selected.empty:
        return pd.DataFrame(), pd.DataFrame(), {}
    row = selected.iloc[0]
    length_path = aal3_connectome_matrix_path(
        settings.connectomes_root,
        subject_id,
        row.get("Image ID"),
        "len_mean",
    )
    measure_path = aal3_connectome_matrix_path(
        settings.connectomes_root,
        subject_id,
        row.get("Image ID"),
        measure_type,
    )
    if length_path is None or measure_path is None:
        return pd.DataFrame(), pd.DataFrame(), {}
    length_matrix = np.loadtxt(length_path, delimiter=",")
    measure_matrix = np.loadtxt(measure_path, delimiter=",")
    if measure_matrix.shape != length_matrix.shape:
        return pd.DataFrame(), pd.DataFrame(), {}
    upper = np.triu_indices_from(length_matrix, k=1)
    lengths = length_matrix[upper]
    measures = measure_matrix[upper]
    exception, threshold, bin_ids = edr_adaptive_exception_arrays(
        lengths,
        measures,
    )
    valid = (
        np.isfinite(lengths)
        & (lengths > 0)
        & np.isfinite(measures)
        & (measures > 0)
        & np.isfinite(threshold)
    )
    valid_indices = np.where(valid)[0]
    exception_indices = valid_indices[exception[valid_indices]]
    ordinary_indices = valid_indices[~exception[valid_indices]]
    seed = int(
        sum(
            (index + 1) * value
            for index, value in enumerate(subject_id.encode())
        )
        % (2**32 - 1)
    )
    rng = np.random.default_rng(seed)
    if ordinary_indices.size > 3500:
        ordinary_indices = np.sort(
            rng.choice(
                ordinary_indices,
                size=3500,
                replace=False,
            )
        )
    if exception_indices.size > 1500:
        exception_indices = np.sort(
            rng.choice(
                exception_indices,
                size=1500,
                replace=False,
            )
        )
    display_indices = np.concatenate(
        (ordinary_indices, exception_indices)
    )
    display = pd.DataFrame(
        {
            "Length (mm)": lengths[display_indices],
            "Measure": measures[display_indices],
            "Threshold": threshold[display_indices],
            "Class": np.where(
                exception[display_indices],
                "Exception",
                "Candidate",
            ),
        }
    )
    curve_rows: list[dict[str, float]] = []
    for bin_id in sorted(set(bin_ids[valid].tolist())):
        mask = valid & (bin_ids == bin_id)
        if not mask.any():
            continue
        curve_rows.append(
            {
                "Minimum length": float(np.nanmin(lengths[mask])),
                "Maximum length": float(np.nanmax(lengths[mask])),
                "Median": float(np.nanmedian(measures[mask])),
                "Threshold": float(np.nanmedian(threshold[mask])),
            }
        )
    metadata = {
        "subject_id": subject_id,
        "group": str(row.get("group", "")),
        "candidate_n": int(valid.sum()),
        "exception_n": int((valid & exception).sum()),
    }
    return display, pd.DataFrame(curve_rows), metadata


def model_dashboard_family(feature_group: object) -> str:
    group = str(feature_group)
    if group in {"Global graph", "Node metrics", "Coupling"}:
        return "Graph / network topology"
    if group in {"Raw high-variance edges", "Raw matrix summaries"}:
        return "Connectome edge / matrix"
    if group in {"Global DTI", "Local ROI DTI"}:
        return "Structural diffusion (FA/MD/RD/AxD)"
    if group in {"EDR exceptions", "AAL EDR exceptions"}:
        return "EDR exceptions"
    if group in {
        "LR/SR and delay",
        "Delay",
        "Advanced structural",
        "Brain age",
    }:
        return "LR/SR, delay, and advanced geometry"
    return "Other"


def _feature_signature(settings: Settings) -> tuple[int, ...]:
    root = settings.analysis_root / "18_ml_diagnostics"
    return (
        _mtime(root / "ml_feature_catalog.csv"),
        _mtime(root / "ml_subject_feature_matrix.csv"),
    )


def model_feature_eda(
    settings: Settings,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    with _FEATURE_CACHE_LOCK:
        return _model_feature_eda_cached(
            settings,
            _feature_signature(settings),
        )


@lru_cache(maxsize=4)
def _model_feature_eda_cached(
    settings: Settings,
    signature: tuple[int, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    del signature
    root = settings.analysis_root / "18_ml_diagnostics"
    catalog_path = root / "ml_feature_catalog.csv"
    matrix_path = root / "ml_subject_feature_matrix.csv"
    if not catalog_path.is_file() or not matrix_path.is_file():
        return pd.DataFrame(), pd.DataFrame(), 0
    catalog = pd.read_csv(catalog_path)
    if "feature" not in catalog.columns:
        return pd.DataFrame(), pd.DataFrame(), 0
    feature_names = catalog["feature"].astype(str).tolist()
    feature_set = set(feature_names)
    matrix = pd.read_csv(
        matrix_path,
        usecols=lambda column: column in feature_set,
    )
    matrix = matrix.apply(pd.to_numeric, errors="coerce")
    row_count = int(len(matrix))
    non_null = matrix.notna().sum()
    nulls = matrix.isna().sum()
    zeros = matrix.eq(0).sum()
    statistics = pd.DataFrame(
        {
            "n": row_count,
            "Non-null": non_null,
            "Nulls": nulls,
            "Nulls (%)": (
                100.0 * nulls / row_count if row_count else np.nan
            ),
            "Zeros": zeros,
            "Zeros (%)": (
                100.0 * zeros / row_count if row_count else np.nan
            ),
            "Mean": matrix.mean(axis=0),
            "Median": matrix.median(axis=0),
            "Variance": matrix.var(axis=0, ddof=1),
            "SD": matrix.std(axis=0, ddof=1),
            "Min": matrix.min(axis=0),
            "Max": matrix.max(axis=0),
            "Unique": matrix.nunique(dropna=True),
        }
    )
    catalog = catalog.copy()
    catalog["Feature"] = catalog["feature"].astype(str)
    catalog["Dashboard family"] = catalog["feature_group"].map(
        model_dashboard_family
    )
    catalog["Feature group"] = catalog["feature_group"].fillna(
        "not recorded"
    )
    catalog["Source section"] = catalog["source_section"].fillna(
        "not recorded"
    )
    catalog["Source column"] = catalog["source_column"].fillna("")
    catalog["Model status"] = catalog["final_status"].fillna(
        "not recorded"
    )
    catalog["Selected CV folds"] = (
        pd.to_numeric(catalog["selected_fold_count"], errors="coerce")
        .fillna(0)
        .astype(int)
    )
    catalog["Elastic-net nonzero folds"] = (
        pd.to_numeric(
            catalog["elastic_net_nonzero_fold_count"],
            errors="coerce",
        )
        .fillna(0)
        .astype(int)
    )
    catalog["Drop reason"] = catalog["drop_reason"].fillna("")
    feature_eda = catalog[
        [
            "Feature",
            "Dashboard family",
            "Feature group",
            "Source section",
            "Source column",
            "Model status",
            "Selected CV folds",
            "Elastic-net nonzero folds",
            "Drop reason",
        ]
    ].join(statistics, on="Feature")

    requested_families = [
        "Graph / network topology",
        "Connectome edge / matrix",
        "Network-wise systems (DMN/limbic/etc.)",
        "Structural diffusion (FA/MD/RD/AxD)",
        "EDR exceptions",
        "LR/SR, delay, and advanced geometry",
    ]
    family_rows: list[dict[str, object]] = []
    for family in requested_families:
        subset = feature_eda[
            feature_eda["Dashboard family"].eq(family)
        ]
        status = subset["Model status"].astype(str)
        network_gap = family.startswith("Network-wise systems")
        family_rows.append(
            {
                "Requested feature family": family,
                "Candidate features": int(len(subset)),
                "Retained after filtering": int(
                    status.isin(
                        ("selected_in_cv", "retained_not_selected")
                    ).sum()
                ),
                "Selected in CV": int(
                    status.eq("selected_in_cv").sum()
                ),
                "Dropped": int(
                    status.str.startswith("dropped").sum()
                ),
                "Current model binding": (
                    "NOT EXPLICITLY PRESENT"
                    if network_gap
                    else "PRESENT"
                ),
                "Interpretation": (
                    "AAL3-to-network mapping exists, but subject-level "
                    "DMN/limbic within/between-network aggregates are not "
                    "columns in the current model matrix."
                    if network_gap
                    else ""
                ),
            }
        )
    return feature_eda, pd.DataFrame(family_rows), row_count


def model_feature_family_metric_inventory(
    feature_eda: pd.DataFrame,
    family_summary: pd.DataFrame,
) -> pd.DataFrame:
    specifications = {
        "Graph / network topology": {
            "High-level metrics captured": (
                "Density; total/mean strength; characteristic path length; "
                "global efficiency; nodal degree, strength and efficiency; "
                "topology-microstructure Spearman coupling"
            ),
            "Representation": (
                "Whole-connectome summaries plus 166-node metric blocks; "
                "individual AAL IDs are hidden in this overview"
            ),
        },
        "Connectome edge / matrix": {
            "High-level metrics captured": (
                "Selected high-variance edge weights; positive-edge "
                "count/density; edge sum, mean, median, SD, IQR, q10/q90 "
                "and min/max across count, fd_sum, length and "
                "diffusion-weight matrices"
            ),
            "Representation": (
                "Selected edge pairs plus whole-matrix distribution summaries"
            ),
        },
        "Network-wise systems (DMN/limbic/etc.)": {
            "High-level metrics captured": (
                "Not a direct model block yet. The Network Analysis tab "
                "separately computes network-aggregated FA/MD/RD/AxD, graph "
                "metrics, coupling, and within/between-network weight and "
                "density"
            ),
            "Representation": (
                "Mapped analysis outputs only; zero direct columns in the "
                "current model matrix"
            ),
        },
        "Structural diffusion (FA/MD/RD/AxD)": {
            "High-level metrics captured": (
                "Whole-connectome and regional FA, MD, RD and AxD; edge mean, "
                "median, IQR and edge-support counts"
            ),
            "Representation": (
                "Global summaries plus 166-region metric blocks"
            ),
        },
        "EDR exceptions": {
            "High-level metrics captured": (
                "EDR lambda/slope/intercept/correlation; SR/MR/LR candidate "
                "count, density, exception count/rate, exception strength and "
                "weight; regional exception count/rate/strength"
            ),
            "Representation": (
                "Subject-level EDR summaries plus 166-region exception blocks"
            ),
        },
        "LR/SR, delay, and advanced geometry": {
            "High-level metrics captured": (
                "SR/MR/LR length, weight, density, efficiency and delay; "
                "SR-LR difference/ratio; EDR residuals; long-range "
                "vulnerability; short-range preservation; compensation; "
                "brain age/BAG and delay burden"
            ),
            "Representation": (
                "Subject-level geometric, delay, resilience and brain-age "
                "summaries"
            ),
        },
    }
    family_counts = (
        family_summary.set_index("Requested feature family")[
            "Candidate features"
        ].to_dict()
        if not family_summary.empty
        else {}
    )
    rows: list[dict[str, object]] = []
    for family, specification in specifications.items():
        subset = feature_eda[
            feature_eda["Dashboard family"].eq(family)
        ]
        source_blocks = sorted(
            {
                str(value)
                for value in subset["Feature group"].dropna()
                if str(value).strip()
            }
        )
        rows.append(
            {
                "Feature family": family,
                "Candidate features": int(
                    family_counts.get(family, 0)
                ),
                **specification,
                "Actual source blocks": (
                    ", ".join(source_blocks)
                    if source_blocks
                    else "None in current model matrix"
                ),
            }
        )
    return pd.DataFrame(rows)


def network_mapping_summary(
    settings: Settings,
) -> dict[str, pd.DataFrame | str]:
    path = (
        settings.analysis_root
        / "19_network_analysis/network_mapping_used.csv"
    )
    if not path.is_file():
        return {
            "mapping": pd.DataFrame(),
            "functional": pd.DataFrame(),
            "anatomical": pd.DataFrame(),
            "status": "missing",
        }
    mapping = pd.read_csv(path)
    functional = (
        mapping.groupby("functional_network", dropna=False)
        .size()
        .rename("Mapped AAL3 nodes")
        .reset_index()
        .rename(columns={"functional_network": "Functional network"})
        .sort_values("Functional network")
        if "functional_network" in mapping.columns
        else pd.DataFrame()
    )
    anatomical = (
        mapping.groupby("anatomical_system", dropna=False)
        .size()
        .rename("Mapped AAL3 nodes")
        .reset_index()
        .rename(columns={"anatomical_system": "Anatomical system"})
        .sort_values("Anatomical system")
        if "anatomical_system" in mapping.columns
        else pd.DataFrame()
    )
    return {
        "mapping": mapping,
        "functional": functional,
        "anatomical": anatomical,
        "status": (
            "analysis-defined Yeo-7-inspired approximation; not an official "
            "native AAL3-to-Yeo assignment"
        ),
    }
