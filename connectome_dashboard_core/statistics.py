from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


GROUP_ORDER = ("CN", "MCI", "AD")


def holm_adjusted_p_values(values: list[float]) -> list[float]:
    adjusted = np.full(len(values), np.nan, dtype=float)
    finite = [
        index for index, value in enumerate(values) if np.isfinite(value)
    ]
    if not finite:
        return adjusted.tolist()
    ordered = sorted(finite, key=lambda index: float(values[index]))
    running = 0.0
    family_size = len(ordered)
    for rank, index in enumerate(ordered):
        candidate = min(
            1.0,
            (family_size - rank) * float(values[index]),
        )
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted.tolist()


def paired_rank_biserial(first: np.ndarray, second: np.ndarray) -> float:
    difference = np.asarray(first, dtype=float) - np.asarray(
        second, dtype=float
    )
    difference = difference[
        np.isfinite(difference) & ~np.isclose(difference, 0.0)
    ]
    if difference.size == 0:
        return math.nan
    ranks = stats.rankdata(np.abs(difference))
    positive = float(ranks[difference > 0].sum())
    negative = float(ranks[difference < 0].sum())
    denominator = positive + negative
    return (
        (positive - negative) / denominator
        if denominator
        else math.nan
    )


def _direction(
    first_group: str,
    second_group: str,
    first_median: float,
    second_median: float,
) -> str:
    if not np.isfinite(first_median) or not np.isfinite(second_median):
        return "Not estimable"
    if first_median > second_median:
        return f"{first_group} higher"
    if second_median > first_median:
        return f"{second_group} higher"
    return "Equal medians"


def edr_lr_measure_inference(
    subject_table: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    required = {
        "subject_id",
        "group",
        "lr_nonexception_measure_median",
        "lr_exception_measure_median",
    }
    if subject_table.empty or not required.issubset(subject_table.columns):
        return pd.DataFrame(), pd.DataFrame(), {}
    analysis = subject_table[
        [
            "subject_id",
            "group",
            "lr_nonexception_measure_median",
            "lr_exception_measure_median",
        ]
    ].copy()
    analysis["group"] = analysis["group"].astype(str)
    for column in (
        "lr_nonexception_measure_median",
        "lr_exception_measure_median",
    ):
        analysis[column] = pd.to_numeric(analysis[column], errors="coerce")
        analysis.loc[analysis[column] <= 0, column] = np.nan
    analysis = analysis[analysis["group"].isin(GROUP_ORDER)].copy()
    analysis["lr_exception_strength_change"] = (
        analysis["lr_exception_measure_median"]
        - analysis["lr_nonexception_measure_median"]
    )
    analysis["lr_log_edge_class_contrast"] = (
        np.log(analysis["lr_exception_measure_median"])
        - np.log(analysis["lr_nonexception_measure_median"])
    )

    plot_rows: list[dict[str, Any]] = []
    for _, row in analysis.iterrows():
        for label, column in (
            ("Non-exception", "lr_nonexception_measure_median"),
            ("Exception", "lr_exception_measure_median"),
            (
                "Exception strength change",
                "lr_exception_strength_change",
            ),
        ):
            value = row[column]
            if pd.notna(value):
                plot_rows.append(
                    {
                        "Subject": str(row["subject_id"]),
                        "Group": str(row["group"]),
                        "Edge class": label,
                        "Subject median": float(value),
                    }
                )
    plot_data = pd.DataFrame(plot_rows)

    results: list[dict[str, Any]] = []
    within_indices: list[int] = []
    within_p: list[float] = []
    for group in GROUP_ORDER:
        paired = analysis[analysis["group"].eq(group)].dropna(
            subset=[
                "lr_nonexception_measure_median",
                "lr_exception_measure_median",
            ]
        )
        ordinary = paired["lr_nonexception_measure_median"].to_numpy(
            dtype=float
        )
        exceptional = paired["lr_exception_measure_median"].to_numpy(
            dtype=float
        )
        p_value = math.nan
        statistic = math.nan
        if len(paired) >= 2 and not np.allclose(exceptional, ordinary):
            test = stats.wilcoxon(
                exceptional,
                ordinary,
                alternative="two-sided",
                zero_method="wilcox",
                method="auto",
            )
            statistic = float(test.statistic)
            p_value = float(test.pvalue)
        index = len(results)
        within_indices.append(index)
        within_p.append(p_value)
        results.append(
            {
                "Question": "Within group",
                "Metric": "Exception vs non-exception strength",
                "Contrast": f"{group}: exception vs non-exception",
                "N": f"{len(paired)} paired",
                "Median A": (
                    float(np.median(exceptional))
                    if exceptional.size
                    else math.nan
                ),
                "Median B": (
                    float(np.median(ordinary))
                    if ordinary.size
                    else math.nan
                ),
                "Median difference": (
                    float(np.median(exceptional - ordinary))
                    if exceptional.size
                    else math.nan
                ),
                "Test": "Paired Wilcoxon",
                "Statistic": statistic,
                "Effect": paired_rank_biserial(exceptional, ordinary),
                "Effect type": "Paired rank-biserial",
                "Observed direction": (
                    "Exception higher"
                    if np.nanmedian(exceptional) > np.nanmedian(ordinary)
                    else "Non-exception higher"
                ),
                "p": p_value,
                "Adjusted p": math.nan,
                "Family omnibus p": math.nan,
                "Family omnibus adjusted p": math.nan,
                "Correction family": "Holm across 3 within-group tests",
            }
        )
    for index, adjusted in zip(
        within_indices, holm_adjusted_p_values(within_p)
    ):
        results[index]["Adjusted p"] = adjusted

    metric_families = (
        ("Exception strength", "lr_exception_measure_median"),
        ("Non-exception strength", "lr_nonexception_measure_median"),
        (
            "Exception − non-exception strength",
            "lr_exception_strength_change",
        ),
    )
    omnibus_rows: list[int] = []
    omnibus_p_values: list[float] = []
    pairwise_rows_by_metric: dict[str, list[int]] = {}
    for metric_label, metric_column in metric_families:
        values_by_group = {
            group: analysis.loc[
                analysis["group"].eq(group), metric_column
            ]
            .dropna()
            .to_numpy(dtype=float)
            for group in GROUP_ORDER
        }
        available_groups = [
            group
            for group in GROUP_ORDER
            if values_by_group[group].size >= 2
        ]
        kw_statistic = math.nan
        kw_p = math.nan
        epsilon_squared = math.nan
        if len(available_groups) >= 2:
            test = stats.kruskal(
                *[values_by_group[group] for group in available_groups]
            )
            kw_statistic = float(test.statistic)
            kw_p = float(test.pvalue)
            total_n = sum(
                values_by_group[group].size for group in available_groups
            )
            if total_n > len(available_groups):
                epsilon_squared = max(
                    0.0,
                    (
                        kw_statistic - len(available_groups) + 1
                    )
                    / (total_n - len(available_groups)),
                )
        omnibus_index = len(results)
        omnibus_rows.append(omnibus_index)
        omnibus_p_values.append(kw_p)
        results.append(
            {
                "Question": "Across groups",
                "Metric": metric_label,
                "Contrast": "CN vs MCI vs AD",
                "N": " / ".join(
                    f"{group} {values_by_group[group].size}"
                    for group in GROUP_ORDER
                ),
                "Median A": math.nan,
                "Median B": math.nan,
                "Median difference": math.nan,
                "Test": "Kruskal-Wallis",
                "Statistic": kw_statistic,
                "Effect": epsilon_squared,
                "Effect type": "Epsilon-squared",
                "Observed direction": "Omnibus; no pairwise direction",
                "p": kw_p,
                "Adjusted p": math.nan,
                "Family omnibus p": kw_p,
                "Family omnibus adjusted p": math.nan,
                "Correction family": (
                    "Holm across 3 metric-family omnibus tests"
                ),
            }
        )

        pair_indices: list[int] = []
        pair_p: list[float] = []
        for first_group, second_group in (
            ("CN", "MCI"),
            ("CN", "AD"),
            ("MCI", "AD"),
        ):
            first = values_by_group[first_group]
            second = values_by_group[second_group]
            statistic = math.nan
            p_value = math.nan
            effect = math.nan
            if first.size >= 2 and second.size >= 2:
                test = stats.mannwhitneyu(
                    first,
                    second,
                    alternative="two-sided",
                )
                statistic = float(test.statistic)
                p_value = float(test.pvalue)
                effect = (
                    2.0
                    * statistic
                    / float(first.size * second.size)
                    - 1.0
                )
            first_median = (
                float(np.median(first)) if first.size else math.nan
            )
            second_median = (
                float(np.median(second)) if second.size else math.nan
            )
            index = len(results)
            pair_indices.append(index)
            pair_p.append(p_value)
            results.append(
                {
                    "Question": "Across groups",
                    "Metric": metric_label,
                    "Contrast": f"{first_group} vs {second_group}",
                    "First group": first_group,
                    "Second group": second_group,
                    "N": f"{first.size} / {second.size}",
                    "Median A": first_median,
                    "Median B": second_median,
                    "Median difference": (
                        first_median - second_median
                        if np.isfinite(first_median)
                        and np.isfinite(second_median)
                        else math.nan
                    ),
                    "Test": "Mann-Whitney U",
                    "Statistic": statistic,
                    "Effect": effect,
                    "Effect type": (
                        "Cliff delta (first group minus second group)"
                    ),
                    "Observed direction": _direction(
                        first_group,
                        second_group,
                        first_median,
                        second_median,
                    ),
                    "p": p_value,
                    "Adjusted p": math.nan,
                    "Family omnibus p": kw_p,
                    "Family omnibus adjusted p": math.nan,
                    "Correction family": (
                        f"Holm across 3 {metric_label.lower()} "
                        "pairwise group tests"
                    ),
                }
            )
        for index, adjusted in zip(
            pair_indices, holm_adjusted_p_values(pair_p)
        ):
            results[index]["Adjusted p"] = adjusted
        pairwise_rows_by_metric[metric_label] = pair_indices

    adjusted_omnibus = holm_adjusted_p_values(omnibus_p_values)
    for metric_spec, omnibus_index, adjusted in zip(
        metric_families, omnibus_rows, adjusted_omnibus
    ):
        metric_label = metric_spec[0]
        results[omnibus_index]["Adjusted p"] = adjusted
        results[omnibus_index][
            "Family omnibus adjusted p"
        ] = adjusted
        for pair_index in pairwise_rows_by_metric[metric_label]:
            results[pair_index][
                "Family omnibus adjusted p"
            ] = adjusted

    # A diagnosis effect that is equally present in exception and
    # non-exception strengths is not exception-specific. On the positive,
    # strongly skewed strength scale, the subject-level log contrast
    # log(exception) - log(non-exception) is the scale-appropriate
    # edge-class contrast. Comparing it across diagnostic groups is the
    # non-parametric edge-class-by-diagnosis interaction test.
    interaction_values = {
        group: analysis.loc[
            analysis["group"].eq(group),
            "lr_log_edge_class_contrast",
        ]
        .dropna()
        .to_numpy(dtype=float)
        for group in GROUP_ORDER
    }
    interaction_groups = [
        group
        for group in GROUP_ORDER
        if interaction_values[group].size >= 2
    ]
    interaction_statistic = math.nan
    interaction_p = math.nan
    interaction_effect = math.nan
    if len(interaction_groups) >= 2:
        interaction_test = stats.kruskal(
            *[interaction_values[group] for group in interaction_groups]
        )
        interaction_statistic = float(interaction_test.statistic)
        interaction_p = float(interaction_test.pvalue)
        interaction_n = sum(
            interaction_values[group].size
            for group in interaction_groups
        )
        if interaction_n > len(interaction_groups):
            interaction_effect = max(
                0.0,
                (
                    interaction_statistic
                    - len(interaction_groups)
                    + 1
                )
                / (interaction_n - len(interaction_groups)),
            )
    results.append(
        {
            "Question": "Edge-class × diagnosis interaction",
            "Metric": "Within-subject log strength contrast",
            "Contrast": "CN vs MCI vs AD",
            "N": " / ".join(
                f"{group} {interaction_values[group].size}"
                for group in GROUP_ORDER
            ),
            "Median A": math.nan,
            "Median B": math.nan,
            "Median difference": math.nan,
            "Test": "Kruskal-Wallis",
            "Statistic": interaction_statistic,
            "Effect": interaction_effect,
            "Effect type": "Epsilon-squared",
            "Observed direction": "Omnibus; no pairwise direction",
            "p": interaction_p,
            "Adjusted p": interaction_p,
            "Family omnibus p": interaction_p,
            "Family omnibus adjusted p": interaction_p,
            "Correction family": "Single prespecified interaction test",
        }
    )
    interaction_pair_indices: list[int] = []
    interaction_pair_p: list[float] = []
    for first_group, second_group in (
        ("CN", "MCI"),
        ("CN", "AD"),
        ("MCI", "AD"),
    ):
        first = interaction_values[first_group]
        second = interaction_values[second_group]
        statistic = math.nan
        p_value = math.nan
        effect = math.nan
        if first.size >= 2 and second.size >= 2:
            interaction_pair_test = stats.mannwhitneyu(
                first,
                second,
                alternative="two-sided",
            )
            statistic = float(interaction_pair_test.statistic)
            p_value = float(interaction_pair_test.pvalue)
            effect = (
                2.0
                * statistic
                / float(first.size * second.size)
                - 1.0
            )
        first_median = (
            float(np.median(first)) if first.size else math.nan
        )
        second_median = (
            float(np.median(second)) if second.size else math.nan
        )
        interaction_pair_indices.append(len(results))
        interaction_pair_p.append(p_value)
        results.append(
            {
                "Question": "Edge-class × diagnosis interaction",
                "Metric": "Within-subject log strength contrast",
                "Contrast": f"{first_group} vs {second_group}",
                "First group": first_group,
                "Second group": second_group,
                "N": f"{first.size} / {second.size}",
                "Median A": first_median,
                "Median B": second_median,
                "Median difference": (
                    first_median - second_median
                    if np.isfinite(first_median)
                    and np.isfinite(second_median)
                    else math.nan
                ),
                "Test": "Mann-Whitney U",
                "Statistic": statistic,
                "Effect": effect,
                "Effect type": (
                    "Cliff delta (first group minus second group)"
                ),
                "Observed direction": _direction(
                    first_group,
                    second_group,
                    first_median,
                    second_median,
                ),
                "p": p_value,
                "Adjusted p": math.nan,
                "Family omnibus p": interaction_p,
                "Family omnibus adjusted p": interaction_p,
                "Correction family": (
                    "Holm across 3 interaction pairwise tests"
                ),
            }
        )
    for index, adjusted in zip(
        interaction_pair_indices,
        holm_adjusted_p_values(interaction_pair_p),
    ):
        results[index]["Adjusted p"] = adjusted

    for row in results:
        row["Requested 4-test Holm p"] = math.nan
    requested_keys = (
        ("Exception strength", "MCI vs AD"),
        ("Non-exception strength", "MCI vs AD"),
        ("Exception − non-exception strength", "MCI vs AD"),
        ("Exception − non-exception strength", "CN vs AD"),
    )
    index_by_key = {
        (str(row["Metric"]), str(row["Contrast"])): index
        for index, row in enumerate(results)
        if row["Question"] == "Across groups"
        and row["Test"] == "Mann-Whitney U"
    }
    requested_indices = [
        index_by_key[key] for key in requested_keys if key in index_by_key
    ]
    requested_adjusted = holm_adjusted_p_values(
        [float(results[index]["p"]) for index in requested_indices]
    )
    for index, adjusted in zip(requested_indices, requested_adjusted):
        results[index]["Requested 4-test Holm p"] = adjusted

    table = pd.DataFrame(results)
    if not table.empty:
        adjusted = pd.to_numeric(table["Adjusted p"], errors="coerce")
        omnibus_adjusted = pd.to_numeric(
            table["Family omnibus adjusted p"], errors="coerce"
        )
        supported = adjusted < 0.05
        pairwise = table["Question"].eq("Across groups") & table[
            "Test"
        ].eq("Mann-Whitney U")
        supported.loc[pairwise] = (
            supported.loc[pairwise]
            & (omnibus_adjusted.loc[pairwise] < 0.05)
        )
        table["Statistically supported after correction"] = supported.map(
            {True: "YES", False: "NO"}
        )
        table["Requested contrast supported"] = (
            pd.to_numeric(
                table["Requested 4-test Holm p"], errors="coerce"
            )
            < 0.05
        ).map({True: "YES", False: "NO"})

    requested_p = {
        key: float(
            results[index_by_key[key]]["Requested 4-test Holm p"]
        )
        for key in requested_keys
        if key in index_by_key
    }
    metadata = {
        "exception_omnibus_p": float(
            results[omnibus_rows[0]]["p"]
        ),
        "exception_omnibus_adjusted_p": float(adjusted_omnibus[0]),
        "change_omnibus_p": float(results[omnibus_rows[2]]["p"]),
        "change_omnibus_adjusted_p": float(adjusted_omnibus[2]),
        "change_mci_ad_adjusted_p": requested_p.get(
            ("Exception − non-exception strength", "MCI vs AD"),
            math.nan,
        ),
        "change_cn_ad_adjusted_p": requested_p.get(
            ("Exception − non-exception strength", "CN vs AD"),
            math.nan,
        ),
        "interaction_omnibus_p": interaction_p,
        "interaction_cn_mci_adjusted_p": float(
            results[interaction_pair_indices[0]]["Adjusted p"]
        ),
        "interaction_cn_ad_adjusted_p": float(
            results[interaction_pair_indices[1]]["Adjusted p"]
        ),
        "interaction_mci_ad_adjusted_p": float(
            results[interaction_pair_indices[2]]["Adjusted p"]
        ),
    }
    return plot_data, table, metadata
