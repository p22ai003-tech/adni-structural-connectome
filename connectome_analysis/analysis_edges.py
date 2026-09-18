from __future__ import annotations

import re
from collections import deque
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from connectome_analysis.analysis_config import GROUP_ORDER, AnalysisPaths
from connectome_analysis.analysis_plots import heatmap_matrix, save_dataframe, write_inference_markdown
from connectome_analysis.analysis_stats import bh_fdr


@dataclass
class ConnectomeStack:
    subject_ids: list[str]
    groups: list[str]
    matrices: np.ndarray
    nodes: int


def _connectome_path(paths: AnalysisPaths, subject_id: str, connectome_type: str) -> Path | None:
    pattern = f"SC_AAL166_{subject_id}_I*_{connectome_type}.csv"
    matches = sorted(
        p for p in paths.connectomes_dir.glob(pattern) if p.is_file() and not p.name.startswith("._")
    )
    return matches[0] if matches else None


def load_connectome_stack(
    paths: AnalysisPaths,
    master: pd.DataFrame,
    connectome_type: str = "fd_sum",
) -> ConnectomeStack:
    subject_ids = []
    groups = []
    matrices = []
    shape_rows = []
    for _, row in master.drop_duplicates("subject_id").iterrows():
        if "has_conn_ALL" in row.index:
            all_flag = pd.to_numeric(pd.Series([row.get("has_conn_ALL")]), errors="coerce").iloc[0]
            if not np.isfinite(all_flag) or int(all_flag) != 1:
                continue
        path = _connectome_path(paths, row["subject_id"], connectome_type)
        if path is None:
            continue
        mat = pd.read_csv(path, header=None).to_numpy(dtype=float)
        if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
            shape_rows.append(
                {
                    "subject_id": row["subject_id"],
                    "group": row["group"],
                    "connectome_type": connectome_type,
                    "shape": "non_square",
                    "path": str(path),
                    "used": 0,
                }
            )
            continue
        subject_ids.append(row["subject_id"])
        groups.append(row["group"])
        matrices.append(mat)
        shape_rows.append(
            {
                "subject_id": row["subject_id"],
                "group": row["group"],
                "connectome_type": connectome_type,
                "shape": f"{mat.shape[0]}x{mat.shape[1]}",
                "path": str(path),
                "used": 1,
            }
        )
    if not matrices:
        raise FileNotFoundError(f"No connectomes found for type {connectome_type}")

    shape_counts = Counter(mat.shape for mat in matrices)
    modal_shape, _ = shape_counts.most_common(1)[0]
    kept_subject_ids = []
    kept_groups = []
    kept_matrices = []
    for sid, group, mat in zip(subject_ids, groups, matrices):
        if mat.shape != modal_shape:
            continue
        kept_subject_ids.append(sid)
        kept_groups.append(group)
        kept_matrices.append(mat)

    if not kept_matrices:
        raise FileNotFoundError(f"No shape-consistent connectomes found for type {connectome_type}")

    if shape_rows:
        shape_df = pd.DataFrame(shape_rows)
        if not shape_df.empty:
            modal_label = f"{modal_shape[0]}x{modal_shape[1]}"
            shape_df["modal_shape"] = modal_label
            shape_df["used"] = (
                shape_df["used"].astype(int).eq(1) & shape_df["shape"].eq(modal_label)
            ).astype(int)
            paths.master_dir.mkdir(parents=True, exist_ok=True)
            shape_df.to_csv(
                paths.master_dir / f"connectome_stack_shape_inventory_{connectome_type}.csv",
                index=False,
            )

    arr = np.stack(kept_matrices, axis=0)
    return ConnectomeStack(subject_ids=kept_subject_ids, groups=kept_groups, matrices=arr, nodes=arr.shape[1])


def _upper_triangle(arr: np.ndarray) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    iu = np.triu_indices(arr.shape[1], k=1)
    return arr[:, iu[0], iu[1]], iu


def _welch_vectorized(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n1, n2 = a.shape[0], b.shape[0]
    mean1, mean2 = np.nanmean(a, axis=0), np.nanmean(b, axis=0)
    var1, var2 = np.nanvar(a, axis=0, ddof=1), np.nanvar(b, axis=0, ddof=1)
    denom = np.sqrt(var1 / n1 + var2 / n2)
    denom[denom == 0] = np.nan
    tvals = (mean1 - mean2) / denom
    with np.errstate(divide="ignore", invalid="ignore"):
        df = (var1 / n1 + var2 / n2) ** 2 / (
            (var1**2) / ((n1**2) * (n1 - 1)) + (var2**2) / ((n2**2) * (n2 - 1))
        )
        pvals = 2.0 * stats.t.sf(np.abs(tvals), df)
    return tvals, pvals


def _kruskal_per_edge(values: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    uniq = [g for g in GROUP_ORDER if g in set(groups)]
    stats_out = np.full(values.shape[1], np.nan, dtype=float)
    p_out = np.full(values.shape[1], np.nan, dtype=float)
    masks = [groups == g for g in uniq]
    for idx in range(values.shape[1]):
        vecs = [values[m, idx] for m in masks]
        vecs = [v[np.isfinite(v)] for v in vecs if np.isfinite(v).sum() > 1]
        if len(vecs) < 2:
            continue
        try:
            stat, p = stats.kruskal(*vecs)
            stats_out[idx] = stat
            p_out[idx] = p
        except Exception:
            continue
    return stats_out, p_out


def _adjacency_from_mask(mask: np.ndarray, n_nodes: int, iu: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    adj = np.zeros((n_nodes, n_nodes), dtype=bool)
    adj[iu[0], iu[1]] = mask
    adj |= adj.T
    return adj


def _components(adj: np.ndarray) -> list[list[int]]:
    visited = np.zeros(adj.shape[0], dtype=bool)
    comps = []
    for start in range(adj.shape[0]):
        if visited[start] or not adj[start].any():
            continue
        queue = deque([start])
        visited[start] = True
        comp = []
        while queue:
            node = queue.popleft()
            comp.append(node)
            nbrs = np.where(adj[node])[0]
            for nbr in nbrs:
                if not visited[nbr]:
                    visited[nbr] = True
                    queue.append(nbr)
        if len(comp) > 1:
            comps.append(sorted(comp))
    return comps


def _component_mass(component: list[int], stat_matrix: np.ndarray) -> float:
    sub = stat_matrix[np.ix_(component, component)]
    return float(np.nansum(np.abs(np.triu(sub, 1))))


def _nbs_pairwise(
    edge_values: np.ndarray,
    groups: np.ndarray,
    group_a: str,
    group_b: str,
    n_nodes: int,
    iu: tuple[np.ndarray, np.ndarray],
    threshold: float = 3.1,
    n_perm: int = 1000,
    seed: int = 42,
) -> tuple[pd.DataFrame, np.ndarray]:
    a = edge_values[groups == group_a]
    b = edge_values[groups == group_b]
    tvals, pvals = _welch_vectorized(a, b)
    mask = np.abs(tvals) >= threshold
    stat_matrix = np.zeros((n_nodes, n_nodes), dtype=float)
    stat_matrix[iu[0], iu[1]] = tvals
    stat_matrix += stat_matrix.T
    adj = _adjacency_from_mask(mask, n_nodes, iu)
    comps = _components(adj)
    rng = np.random.default_rng(seed)
    observed = []
    for comp_id, comp in enumerate(comps, start=1):
        observed.append(
            {
                "component_id": comp_id,
                "nodes": " ".join(str(i + 1) for i in comp),
                "n_nodes": len(comp),
                "mass": _component_mass(comp, stat_matrix),
            }
        )
    if not observed:
        return pd.DataFrame(columns=["component_id", "nodes", "n_nodes", "mass", "p_perm"]), stat_matrix

    labels = np.array([group_a] * len(a) + [group_b] * len(b))
    subset = np.vstack([a, b])
    perm_masses = []
    for _ in range(n_perm):
        perm = rng.permutation(labels)
        t_perm, _ = _welch_vectorized(subset[perm == group_a], subset[perm == group_b])
        adj_perm = _adjacency_from_mask(np.abs(t_perm) >= threshold, n_nodes, iu)
        comps_perm = _components(adj_perm)
        if not comps_perm:
            perm_masses.append(0.0)
            continue
        stat_perm = np.zeros((n_nodes, n_nodes), dtype=float)
        stat_perm[iu[0], iu[1]] = t_perm
        stat_perm += stat_perm.T
        perm_masses.append(max(_component_mass(comp, stat_perm) for comp in comps_perm))
    perm_masses = np.asarray(perm_masses, dtype=float)
    rows = []
    for row in observed:
        p_perm = (np.sum(perm_masses >= row["mass"]) + 1) / (len(perm_masses) + 1)
        row["p_perm"] = p_perm
        rows.append(row)
    return pd.DataFrame(rows), stat_matrix


def run_edgewise_inference(
    paths: AnalysisPaths,
    master: pd.DataFrame,
    connectome_type: str = "fd_sum",
    threshold: float = 3.1,
    n_perm: int = 1000,
) -> dict:
    out_dir = paths.section_dir("10", f"edgewise_{connectome_type}")
    out_dir.mkdir(parents=True, exist_ok=True)
    stack = load_connectome_stack(paths, master, connectome_type=connectome_type)
    edge_values, iu = _upper_triangle(stack.matrices)
    groups = np.asarray(stack.groups)
    kw_stat, kw_p = _kruskal_per_edge(edge_values, groups)
    edge_table = pd.DataFrame(
        {
            "i": iu[0] + 1,
            "j": iu[1] + 1,
            "kw_stat": kw_stat,
            "kw_p": kw_p,
        }
    )
    edge_table["kw_q"] = bh_fdr(edge_table["kw_p"].to_numpy())
    save_dataframe(edge_table, out_dir / f"edges_{connectome_type}_omnibus.csv")

    component_tables = []
    for group_a, group_b in [("CN", "MCI"), ("CN", "AD"), ("MCI", "AD")]:
        if group_a not in groups or group_b not in groups:
            continue
        tvals, pvals = _welch_vectorized(edge_values[groups == group_a], edge_values[groups == group_b])
        pair_table = pd.DataFrame(
            {
                "i": iu[0] + 1,
                "j": iu[1] + 1,
                "welch_t": tvals,
                "welch_p": pvals,
            }
        )
        pair_table["welch_q"] = bh_fdr(pair_table["welch_p"].to_numpy())
        pair_name = f"{group_a}vs{group_b}"
        save_dataframe(pair_table, out_dir / f"edges_{connectome_type}_{pair_name}.csv")
        comp_df, stat_matrix = _nbs_pairwise(
            edge_values,
            groups,
            group_a,
            group_b,
            n_nodes=stack.nodes,
            iu=iu,
            threshold=threshold,
            n_perm=n_perm,
        )
        save_dataframe(comp_df, out_dir / f"nbs_{connectome_type}_{pair_name}_components.csv")
        heatmap_matrix(
            stat_matrix,
            title=f"{connectome_type}: {group_a} vs {group_b} Welch t map",
            out_path=out_dir / f"welch_t_{connectome_type}_{pair_name}.png",
            cmap="coolwarm",
        )
        if not comp_df.empty:
            component_tables.append(comp_df.assign(pair=pair_name))
    comp_all = pd.concat(component_tables, ignore_index=True) if component_tables else pd.DataFrame()
    save_dataframe(comp_all, out_dir / f"nbs_{connectome_type}_all_components.csv")
    write_inference_markdown(
        [
            "Omnibus edge-wise inference uses Kruskal-Wallis across groups with BH-FDR across edges.",
            "Pairwise component inference uses Welch t-statistics with an NBS-style permutation correction on supra-threshold components.",
            "This is intentionally reported as NBS-like / TFNBS-oriented rather than a strict canonical TFNBS implementation.",
        ],
        out_dir / "edgewise_inference.md",
    )
    return {"stack": stack, "edges": edge_table, "components": comp_all, "out_dir": out_dir}
