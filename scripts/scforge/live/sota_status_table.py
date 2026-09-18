#!/usr/bin/env python3
"""Group-wise SOTA status table: CN/MCI/AD count-density + zero-rows, current vs
projected-after-promotion. Density = COUNT (fiber/streamline count) based:
fraction of the 166 valid AAL3 region-pairs that have >=1 streamline.

Re-run any time to refresh the overall view.
"""
from __future__ import annotations
import csv, glob, json, os, sys
import numpy as np

DERIV = "/data/derivatives"
GROUP_CSV = f"{DERIV}/sc_forge_v1/group_qc/current_connectome_subject_availability_with_group.csv"
PROD = f"{DERIV}/connectomes/SC_AAL_{{sid}}_count.csv"
GAPS = {35, 36, 81, 82}
ORDER = ["CN", "MCI", "AD", "ALL"]


def group_map() -> dict:
    g = {}
    with open(GROUP_CSV) as h:
        for r in csv.DictReader(h):
            sid = r["sid"].strip()
            grp = (r.get("group") or "").strip().upper()
            if sid and grp:
                g[sid] = grp
    return g


def qc_count(path: str):
    M = np.loadtxt(path, delimiter=","); np.fill_diagonal(M, 0)
    n = M.shape[0]; deg = M.sum(1)
    zv = sum(1 for i in range(n) if deg[i] == 0 and (i + 1) not in GAPS)
    dens = (np.triu(M, 1) > 0).sum() / (166 * 165 / 2)
    return zv, float(dens)


def collect_promotions(run_glob: str) -> dict:
    """sid -> candidate count matrix path, for SOTA runs that PROMOTE."""
    promoted = {}
    for rj in glob.glob(run_glob):
        try:
            r = json.load(open(rj))
        except Exception:
            continue
        if r.get("status") != "ok" or not (r.get("decision", {}) or {}).get("promote"):
            continue
        sid = r["sid"]
        cdir = os.path.join(os.path.dirname(rj), "connectomes")
        cand = os.path.join(cdir, f"SC_AAL3cc_{sid}_count.csv")
        if os.path.exists(cand):
            promoted[sid] = cand
    return promoted


def agg(rows):
    """rows: list of (group, zero_rows, density). Returns per-group stats."""
    out = {}
    for grp in ORDER:
        sub = rows if grp == "ALL" else [r for r in rows if r[0] == grp]
        if not sub:
            continue
        dens = np.array([r[2] for r in sub]); zr = np.array([r[1] for r in sub])
        out[grp] = {
            "n": len(sub),
            "mean_density": float(dens.mean()),
            "median_density": float(np.median(dens)),
            "mean_zero_rows": float(zr.mean()),
            "ready": int((zr <= 15).sum()),
        }
    return out


def print_table(title, stats):
    print(f"\n{title}")
    print(f"{'GROUP':5s} {'n':>4s} {'mean_density':>13s} {'median':>8s} {'mean_zero_rows':>15s} {'analysis_ready(<=15z)':>22s}")
    for grp in ORDER:
        s = stats.get(grp)
        if not s:
            continue
        print(f"{grp:5s} {s['n']:>4d} {s['mean_density']:>13.3f} {s['median_density']:>8.3f} "
              f"{s['mean_zero_rows']:>15.1f} {s['ready']:>15d} ({100*s['ready']/s['n']:.0f}%)")


def print_corrected_table(title, corr):
    """corr: dict group -> list of (old_density, new_density, new_zero_rows).
    Shows the ISOLATED benefit on only the corrected subjects (undiluted)."""
    print(f"\n{title}")
    print(f"{'GROUP':5s} {'n_fixed':>7s} {'old_med_dens':>12s} {'new_med_dens':>12s} "
          f"{'new_mean_dens':>13s} {'ready_now(<=15z)':>16s}")
    for grp in ORDER:
        sub = corr.get(grp) if grp != "ALL" else [x for g in ORDER[:-1] for x in corr.get(g, [])]
        if not sub:
            continue
        old = np.array([x[0] for x in sub]); new = np.array([x[1] for x in sub])
        zr = np.array([x[2] for x in sub])
        print(f"{grp:5s} {len(sub):>7d} {np.median(old):>12.3f} {np.median(new):>12.3f} "
              f"{new.mean():>13.3f} {int((zr<=15).sum()):>10d} ({100*(zr<=15).mean():.0f}%)")


def main():
    gm = group_map()
    promoted = collect_promotions(f"{DERIV}/qc/sc_matrix_qc/route_sota*/*/result.json")
    cur_rows, proj_rows = [], []
    corr = {g: [] for g in ORDER[:-1]}  # per-group (old_d, new_d, new_zero) for fixed subjects
    for sid, grp in gm.items():
        p = PROD.format(sid=sid)
        if not os.path.exists(p):
            continue
        try:
            zv, d = qc_count(p)
        except Exception:
            continue
        cur_rows.append((grp, zv, d))
        if sid in promoted:
            try:
                zv2, d2 = qc_count(promoted[sid])
                proj_rows.append((grp, zv2, d2))
                if grp in corr:
                    corr[grp].append((d, d2, zv2))
            except Exception:
                proj_rows.append((grp, zv, d))
        else:
            proj_rows.append((grp, zv, d))
    print("=" * 84)
    print("SOTA STATUS  |  density = COUNT (fiber-count) based, over 166 valid AAL3 nodes")
    print(f"Claude has corrected & promoted {len(promoted)} subjects so far (gated; never regresses).")
    print_table("[1] CURRENT production (all subjects, dilutes the benefit):", agg(cur_rows))
    if promoted:
        print_corrected_table("[2] CLAUDE-CORRECTED (only the fixed subjects, ISOLATED benefit):", corr)
        print_table("[3] PROJECTED cohort (current + corrected merged in):", agg(proj_rows))
    print("=" * 84)


if __name__ == "__main__":
    main()
