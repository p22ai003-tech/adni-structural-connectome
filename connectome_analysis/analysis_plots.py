from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from connectome_analysis.analysis_config import GROUP_ORDER, GROUP_PALETTE


def save_dataframe(df: pd.DataFrame, out_path: str | Path) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return out


def write_inference_markdown(lines: Iterable[str], out_path: str | Path) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    clean = [str(line).strip() for line in lines if str(line).strip()]
    text = "\n".join(f"- {line}" for line in clean) + ("\n" if clean else "")
    out.write_text(text, encoding="utf-8")
    return out


def _ordered_groups(series: pd.Series) -> list[str]:
    present = [str(v) for v in series.dropna().unique()]
    groups = [g for g in GROUP_ORDER if g in present]
    groups.extend(sorted(g for g in present if g not in groups))
    return groups


def _finish_figure(fig, out_path: str | Path) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    try:
        import matplotlib.pyplot as plt

        plt.close(fig)
    except Exception:
        pass
    return out


def boxplot_with_points(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    ylabel: str,
    out_path: str | Path,
    omnibus_text: str | None = None,
    pairwise_lines: list[str] | None = None,
) -> Path:
    import matplotlib.pyplot as plt
    import seaborn as sns

    data = df[[x_col, y_col]].copy() if {x_col, y_col}.issubset(df.columns) else pd.DataFrame()
    if not data.empty:
        data[y_col] = pd.to_numeric(data[y_col], errors="coerce")
        data = data.dropna(subset=[x_col, y_col])
    order = _ordered_groups(data[x_col]) if not data.empty else GROUP_ORDER
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    if data.empty:
        ax.text(0.5, 0.5, "No valid data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
    else:
        palette = {g: GROUP_PALETTE.get(g, "#4c78a8") for g in order}
        sns.boxplot(
            data=data,
            x=x_col,
            y=y_col,
            hue=x_col,
            order=order,
            hue_order=order,
            palette=palette,
            legend=False,
            showfliers=False,
            width=0.55,
            ax=ax,
        )
        sns.stripplot(
            data=data,
            x=x_col,
            y=y_col,
            order=order,
            color="#222222",
            alpha=0.55,
            size=3,
            jitter=0.22,
            ax=ax,
        )
        ax.set_xlabel("")
        ax.set_ylabel(ylabel)
    ax.set_title(title)
    annotation = []
    if omnibus_text:
        annotation.append(omnibus_text)
    if pairwise_lines:
        annotation.extend(pairwise_lines[:4])
    if annotation:
        ax.text(
            0.02,
            0.98,
            "\n".join(annotation),
            ha="left",
            va="top",
            transform=ax.transAxes,
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "#dddddd", "alpha": 0.85},
        )
    return _finish_figure(fig, out_path)


def manhattan_plot(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    sig_col: str,
    threshold: float,
    title: str,
    ylabel: str,
    out_path: str | Path,
) -> Path:
    import matplotlib.pyplot as plt

    cols = list(dict.fromkeys([x_col, y_col, sig_col]))
    data = df[cols].copy() if set(cols).issubset(df.columns) else pd.DataFrame()
    if not data.empty:
        data[y_col] = pd.to_numeric(data[y_col], errors="coerce")
        data[sig_col] = pd.to_numeric(data[sig_col], errors="coerce")
        data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=[x_col, y_col])
    fig, ax = plt.subplots(figsize=(9.5, 4.5))
    if data.empty:
        ax.text(0.5, 0.5, "No valid data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
    else:
        x = pd.to_numeric(data[x_col], errors="coerce")
        if x.isna().any():
            x = pd.Series(np.arange(1, len(data) + 1), index=data.index)
        vals = data[y_col].to_numpy(dtype=float)
        plot_vals = -np.log10(np.clip(vals, 1e-300, 1.0))
        sig = data[sig_col].to_numpy(dtype=float) < threshold
        ax.scatter(x, plot_vals, c=np.where(sig, "#d62728", "#4c78a8"), s=18, alpha=0.85)
        if np.isfinite(threshold) and threshold > 0:
            ax.axhline(-np.log10(threshold), color="#444444", linestyle="--", linewidth=1)
        ax.set_xlabel(x_col)
        ax.set_ylabel(f"-log10({ylabel})")
    ax.set_title(title)
    return _finish_figure(fig, out_path)


def heatmap_matrix(
    matrix: np.ndarray,
    title: str,
    out_path: str | Path,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
) -> Path:
    import matplotlib.pyplot as plt

    arr = np.asarray(matrix, dtype=float)
    fig, ax = plt.subplots(figsize=(7.0, 6.0))
    if arr.size == 0:
        ax.text(0.5, 0.5, "No valid data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
    else:
        im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        fig.colorbar(im, ax=ax, shrink=0.8)
        ax.set_xlabel("Node")
        ax.set_ylabel("Node")
    ax.set_title(title)
    return _finish_figure(fig, out_path)


def scatter_with_group_fit(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    group_col: str,
    title: str,
    xlabel: str,
    ylabel: str,
    out_path: str | Path,
    annotation_lines: list[str] | None = None,
) -> Path:
    import matplotlib.pyplot as plt

    data = df[[x_col, y_col, group_col]].copy() if {x_col, y_col, group_col}.issubset(df.columns) else pd.DataFrame()
    if not data.empty:
        data[x_col] = pd.to_numeric(data[x_col], errors="coerce")
        data[y_col] = pd.to_numeric(data[y_col], errors="coerce")
        data = data.dropna(subset=[x_col, y_col, group_col])
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    if data.empty:
        ax.text(0.5, 0.5, "No valid data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
    else:
        for group in _ordered_groups(data[group_col]):
            sub = data.loc[data[group_col] == group]
            color = GROUP_PALETTE.get(group, "#4c78a8")
            ax.scatter(sub[x_col], sub[y_col], label=group, alpha=0.7, s=24, color=color)
            if len(sub) >= 3 and sub[x_col].nunique() >= 2:
                coeff = np.polyfit(sub[x_col].to_numpy(dtype=float), sub[y_col].to_numpy(dtype=float), deg=1)
                xs = np.linspace(sub[x_col].min(), sub[x_col].max(), 100)
                ax.plot(xs, np.polyval(coeff, xs), color=color, linewidth=1.5)
        ax.legend(frameon=False)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    ax.set_title(title)
    if annotation_lines:
        ax.text(
            0.02,
            0.98,
            "\n".join(annotation_lines[:4]),
            ha="left",
            va="top",
            transform=ax.transAxes,
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "#dddddd", "alpha": 0.85},
        )
    return _finish_figure(fig, out_path)
