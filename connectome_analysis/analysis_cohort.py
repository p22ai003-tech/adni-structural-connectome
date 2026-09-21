from __future__ import annotations

import os
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from connectome_analysis.analysis_config import GROUP_ORDER, AnalysisPaths


SID_RE = re.compile(r"(?P<sid>\d{3}_S_\d+(?:_I\d+)?)")


def recode_group(value: str) -> str:
    if pd.isna(value):
        return value
    if value in {"EMCI", "LMCI", "SMC"}:
        return "MCI"
    return value


def subject_id_from_text(value: str | float | int | None) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value)
    match = SID_RE.search(text)
    if match:
        sid = match.group("sid")
        return "_".join(sid.split("_")[:3])
    if re.match(r"^\d{3}_S_\d+$", text):
        return text
    return None


def _choose_metrics_table(paths: AnalysisPaths) -> Path | None:
    for candidate in paths.metrics_candidates:
        if candidate.exists():
            return candidate
    return None


def _read_metrics_table(metrics_path: Path | None) -> pd.DataFrame:
    if metrics_path is None:
        return pd.DataFrame()
    if metrics_path.suffix.lower() == ".csv":
        metrics = pd.read_csv(metrics_path)
    else:
        metrics = pd.read_excel(metrics_path)
    if "subject_id" not in metrics.columns:
        if "subject_root" in metrics.columns:
            metrics["subject_id"] = metrics["subject_root"].astype(str).str.split("_").str[:3].str.join("_")
        elif "sid" in metrics.columns:
            metrics["subject_id"] = metrics["sid"].apply(subject_id_from_text)
        elif "Subject ID" in metrics.columns:
            metrics["subject_id"] = metrics["Subject ID"].astype(str)
    if "sid" not in metrics.columns:
        if "subject_root" in metrics.columns:
            metrics["sid"] = metrics["subject_root"]
        elif "subject_id" in metrics.columns:
            metrics["sid"] = metrics["subject_id"]
    return metrics


# What a generic participants table calls a column, mapped onto what the ADNI
# exports call it. Everything downstream reads the ADNI names.
_PARTICIPANT_ALIASES = {
    "participant_id": "subject_id",
    "subject": "subject_id",
    "diagnosis": "group",
    "dx": "group",
    "research_group": "group",
    "age": "Age",
    "sex": "Sex",
    "gender": "Sex",
}


def _load_cohort_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    lower = {str(name).strip().lower(): name for name in df.columns}
    for alias, canonical in _PARTICIPANT_ALIASES.items():
        if canonical not in df.columns and alias in lower:
            df[canonical] = df[lower[alias]]
    if "Subject ID" in df.columns:
        df["subject_id"] = df["Subject ID"].astype(str)
    elif "subject_id" in df.columns:
        df["subject_id"] = df["subject_id"].astype(str)
    else:
        raise ValueError(
            f"Could not find a subject column in {path}; expected one of "
            f"Subject ID, subject_id, participant_id"
        )
    if "Research Group" in df.columns:
        df["group"] = df["Research Group"].map(recode_group)
    elif "group" in df.columns:
        df["group"] = df["group"].map(recode_group)
    return df


def _participants_table(paths: AnalysisPaths) -> Path | None:
    """The generic stand-in for the ADNI cohort exports.

    A study that is not ADNI has one small table saying who its subjects are --
    subject_id and whichever of diagnosis, age and sex it has -- rather than
    the two IDA exports this cohort was built from. Either is enough to give
    every later stage its grouping variable.
    """
    candidates = [
        paths.cohort_dti_csv.parent / "participants.csv",
        Path(os.environ["SC_STUDY"]).parent / "participants.csv"
        if os.environ.get("SC_STUDY") else None,
        Path(os.environ["SC_RAW_IMAGES_ROOT"]) / "participants.csv"
        if os.environ.get("SC_RAW_IMAGES_ROOT") else None,
    ]
    for candidate in candidates:
        if candidate is not None and candidate.is_file():
            return candidate
    return None


def build_master_cohort(paths: AnalysisPaths) -> pd.DataFrame:
    paths.ensure()
    if not paths.cohort_dti_csv.is_file():
        participants = _participants_table(paths)
        if participants is None:
            raise SystemExit(
                f"no cohort table found.\n"
                f"  looked for {paths.cohort_dti_csv}\n"
                f"  and participants.csv beside it, beside the study file, or in "
                f"the raw image root.\n"
                f"A participants.csv needs a subject_id column plus whichever of "
                f"diagnosis, age and sex you have."
            )
        dti = mri = _load_cohort_csv(participants)
    else:
        dti = _load_cohort_csv(paths.cohort_dti_csv)
        mri = _load_cohort_csv(paths.cohort_mri_csv)
    metrics_path = _choose_metrics_table(paths)
    metrics = _read_metrics_table(metrics_path)

    keep_dti = [
        "subject_id",
        "Subject ID",
        "Sex",
        "Age",
        "Weight",
        "Research Group",
        "group",
        "Phase",
        "APOE A1",
        "APOE A2",
        "Global CDR",
        "MMSE Total Score",
        "NPI-Q Total Score",
        "GDSCALE Total Score",
        "FAQ Total Score",
        "Image ID",
        "Description",
        "Type",
    ]
    dti_keep = [c for c in keep_dti if c in dti.columns]
    master = dti[dti_keep].copy()
    mri_cols = [c for c in ["subject_id", "Image ID", "Description", "Type", "Phase"] if c in mri.columns]
    if mri_cols:
        master = master.merge(
            mri[mri_cols].drop_duplicates("subject_id"),
            on="subject_id",
            how="left",
            suffixes=("", "_mri"),
        )
    if not metrics.empty:
        metrics = metrics.copy()
        metrics["subject_id"] = metrics["subject_id"].astype(str)
        metrics = metrics.drop_duplicates("subject_id")
        master = master.merge(metrics, on="subject_id", how="left", suffixes=("", "_metrics"))

    master["group"] = master["group"].map(recode_group)
    master = master.loc[master["group"].isin(GROUP_ORDER)].copy()
    master["sex"] = master.get("Sex", pd.Series(index=master.index)).astype(str).replace({"nan": np.nan})
    master["phase"] = master.get("Phase", pd.Series(index=master.index)).astype(str).replace({"nan": np.nan})
    master["age"] = pd.to_numeric(master.get("Age"), errors="coerce")
    master["weight"] = pd.to_numeric(master.get("Weight"), errors="coerce")

    if "APOE A1" in master.columns and "APOE A2" in master.columns:
        def _e4_count(row: pd.Series) -> float:
            alleles = [str(row.get("APOE A1", "")), str(row.get("APOE A2", ""))]
            if all(a == "nan" for a in alleles):
                return np.nan
            return sum(a.strip() == "4" for a in alleles)

        master["apoe_e4_count"] = master.apply(_e4_count, axis=1)
        master["apoe_e4_carrier"] = master["apoe_e4_count"].apply(
            lambda v: np.nan if pd.isna(v) else int(v > 0)
        )

    connectome_info = connectome_inventory(paths)
    if not connectome_info.empty:
        master = master.merge(connectome_info, on="subject_id", how="left")
    master = merge_sc_matrix_qc_gate(paths, master)

    # --- DENSE COHORT: recompute count-density from the connectomes and keep ONLY the dense (>=0.6)
    # set (the 530 "good" subjects). The main connectomes/ dir holds only dense matrices (mid/low are
    # quarantined to _midband/_lowband_archive), so this yields the canonical dense cohort. Every
    # downstream analysis + the summary then reflect n=530. See docs/DECISIONS.md.
    def _count_density(sid: str) -> float:
        matches = sorted(paths.connectomes_dir.glob(f"SC_AAL166_{sid}_I*_count.csv"))
        if not matches:
            return float("nan")
        try:
            m = np.loadtxt(matches[0], delimiter=","); n = m.shape[0]
            off = m[~np.eye(n, dtype=bool)]
            return float((off > 0).sum() / off.size)
        except Exception:
            return float("nan")

    master["density"] = master["subject_id"].astype(str).map(_count_density)
    master["is_dense"] = master["density"].apply(lambda d: bool(pd.notna(d) and d >= 0.6))
    master = master.loc[master["is_dense"]].copy()
    master["cohort_label"] = "dense"

    out_path = paths.master_dir / "master_cohort.csv"
    master.to_csv(out_path, index=False)
    return master


def load_master_cohort(paths: AnalysisPaths, rebuild: bool = False) -> pd.DataFrame:
    out_path = paths.master_dir / "master_cohort.csv"
    if rebuild or not out_path.exists():
        return build_master_cohort(paths)
    master = pd.read_csv(out_path)
    return merge_sc_matrix_qc_gate(paths, master)


def merge_sc_matrix_qc_gate(paths: AnalysisPaths, master: pd.DataFrame) -> pd.DataFrame:
    """Attach SC-matrix QC status without removing rows.

    The detailed QC is series-level because connectome files are named with full
    scan IDs. Most analysis cohort tables are subject-root level, so this merges
    a conservative subject-level summary while preserving the exact series-level
    decision table under qc/sc_matrix_qc/.
    """
    qc_path = paths.qc_dir / "sc_matrix_qc" / "sc_matrix_qc_decisions.csv"
    if not qc_path.exists():
        if "sc_matrix_qc_status" not in master.columns:
            master = master.copy()
            master["sc_matrix_qc_status"] = "NOT_RUN"
            master["sc_matrix_qc_include"] = np.nan
            master["sc_matrix_qc_gate"] = "not_run"
        return master
    qc = pd.read_csv(qc_path)
    if qc.empty or "subject_id" not in qc.columns:
        return master
    qc = qc.copy()
    qc["sc_matrix_qc_include"] = qc.get("analysis_gate", "").astype(str).eq("include")
    fail_mask = ~qc["sc_matrix_qc_include"].fillna(False)
    severe_order = {
        "FAIL_SOURCE": 5,
        "FAIL_REGISTRATION": 4,
        "FAIL_LABEL_COVERAGE": 3,
        "FAIL_STREAMLINE_ASSIGNMENT": 2,
        "WARN_SPARSE_NODE": 1,
        "PASS": 0,
    }
    qc["_severity"] = qc["sc_matrix_qc_status"].map(severe_order).fillna(0)

    rows = []
    for subject_id, sub in qc.groupby("subject_id", dropna=False):
        include_any = bool(sub["sc_matrix_qc_include"].any())
        fail_any = bool(fail_mask.loc[sub.index].any())
        worst = sub.sort_values("_severity", ascending=False).iloc[0]
        if include_any and fail_any:
            status = "MIXED_REVIEW"
            gate = "include_with_series_review"
        elif include_any:
            status = str(worst["sc_matrix_qc_status"])
            gate = "include"
        else:
            status = str(worst["sc_matrix_qc_status"])
            gate = "exclude_pending_repair"
        rows.append(
            {
                "subject_id": subject_id,
                "sc_matrix_qc_status": status,
                "sc_matrix_qc_gate": gate,
                "sc_matrix_qc_include": include_any,
                "n_sc_qc_series": int(sub["sid"].nunique()) if "sid" in sub.columns else int(len(sub)),
                "n_sc_qc_failed_series": int(fail_mask.loc[sub.index].sum()),
                "sc_matrix_qc_statuses": ";".join(sorted(set(sub["sc_matrix_qc_status"].astype(str)))),
            }
        )
    summary = pd.DataFrame(rows)
    drop_cols = [c for c in summary.columns if c in master.columns and c != "subject_id"]
    if drop_cols:
        master = master.drop(columns=drop_cols)
    return master.merge(summary, on="subject_id", how="left")


def connectome_inventory(paths: AnalysisPaths) -> pd.DataFrame:
    rows = []
    if not paths.connectomes_dir.exists():
        return pd.DataFrame()
    suffix_counter: Counter[str] = Counter()
    subject_types: dict[str, set[str]] = {}
    for path in paths.connectomes_dir.glob("SC_AAL166_*.csv"):
        if path.name.startswith("._"):
            continue
        stem = path.stem
        match = re.match(r"^SC_AAL166_(\d{3}_S_\d+(?:_I\d+)?)_(.+)$", stem)
        if not match:
            continue
        sid = match.group(1)
        subject_id = "_".join(sid.split("_")[:3])
        suffix = match.group(2)
        suffix_counter[suffix] += 1
        subject_types.setdefault(subject_id, set()).add(suffix)
    for subject_id, suffixes in sorted(subject_types.items()):
        row = {"subject_id": subject_id, "n_connectome_types": len(suffixes)}
        for suffix in sorted(suffixes):
            row[f"has_conn_{suffix}"] = 1
        rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.fillna(0)
    summary = pd.DataFrame(
        [{"connectome_type": key, "n_files": val} for key, val in suffix_counter.most_common()]
    )
    summary.to_csv(paths.master_dir / "connectome_type_inventory.csv", index=False)
    if not df.empty:
        df.to_csv(paths.master_dir / "subject_connectome_inventory.csv", index=False)
    return df


def qc_snapshot(paths: AnalysisPaths, master: pd.DataFrame | None = None) -> pd.DataFrame:
    master = master if master is not None else load_master_cohort(paths)
    rows = []
    rows.append({"stage": "cohort_subjects", "n": int(master["subject_id"].nunique())})
    rows.append({"stage": "group_CN", "n": int((master["group"] == "CN").sum())})
    rows.append({"stage": "group_MCI", "n": int((master["group"] == "MCI").sum())})
    rows.append({"stage": "group_AD", "n": int((master["group"] == "AD").sum())})
    for candidate in paths.metrics_candidates:
        if candidate.exists():
            rows.append({"stage": f"metrics_{candidate.name}", "n": 1})
    inventory_path = paths.master_dir / "subject_connectome_inventory.csv"
    if inventory_path.exists():
        conn_inv = pd.read_csv(inventory_path)
        rows.append({"stage": "subjects_with_connectomes", "n": int(conn_inv["subject_id"].nunique())})
    out = pd.DataFrame(rows)
    out.to_csv(paths.master_dir / "qc_snapshot.csv", index=False)
    return out


def cohort_counts_table(master: pd.DataFrame) -> pd.DataFrame:
    counts = (
        master.groupby("group")["subject_id"]
        .nunique()
        .reindex(GROUP_ORDER)
        .rename("n_subjects")
        .reset_index()
    )
    counts["mean_age"] = (
        master.groupby("group")["age"].mean().reindex(GROUP_ORDER).to_numpy()
    )
    counts["female_n"] = (
        master.assign(is_female=master["sex"].astype(str).str.upper().str.startswith("F"))
        .groupby("group")["is_female"]
        .sum()
        .reindex(GROUP_ORDER)
        .to_numpy()
    )
    return counts


def run_demographics_overview(paths: AnalysisPaths, master: pd.DataFrame) -> dict:
    from connectome_analysis.analysis_plots import boxplot_with_points, save_dataframe, write_inference_markdown
    from connectome_analysis.analysis_stats import group_descriptives, kruskal_summary, pairwise_robust_tests

    out_dir = paths.section_dir("02", "demographics")
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = cohort_counts_table(master)
    save_dataframe(counts, out_dir / "group_counts.csv")

    age_df = master[["subject_id", "group", "age"]].dropna()
    age_desc = group_descriptives(age_df, "age")
    age_pairwise = pairwise_robust_tests(age_df, "age")
    age_omni = kruskal_summary(age_df, "age")
    save_dataframe(age_desc, out_dir / "age_descriptives.csv")
    save_dataframe(age_pairwise, out_dir / "age_pairwise.csv")
    pair_lines = []
    for _, row in age_pairwise.sort_values("bm_p").head(3).iterrows():
        pair_lines.append(
            f"{row.group_a} vs {row.group_b}: BM q={row.bm_q:.3g}, Cliff's d={row.cliffs_delta:.3g}"
        )
    boxplot_with_points(
        age_df,
        "group",
        "age",
        title="Age distribution by group",
        ylabel="Age",
        out_path=out_dir / "age_boxplot.png",
        omnibus_text=f"Kruskal p={age_omni.pvalue:.3g}",
        pairwise_lines=pair_lines,
    )

    clinical_cols = [
        c
        for c in ["MMSE Total Score", "Global CDR", "NPI-Q Total Score", "GDSCALE Total Score", "FAQ Total Score"]
        if c in master.columns
    ]
    clinical_rows = []
    for col in clinical_cols:
        sub = master[["subject_id", "group", col]].dropna()
        if sub.empty:
            continue
        desc = group_descriptives(sub, col)
        pairwise = pairwise_robust_tests(sub, col)
        omni = kruskal_summary(sub, col)
        save_dataframe(desc, out_dir / f"{col.replace(' ', '_').lower()}_descriptives.csv")
        save_dataframe(pairwise, out_dir / f"{col.replace(' ', '_').lower()}_pairwise.csv")
        clinical_rows.append({"metric": col, "kw_p": omni.pvalue, "n": omni.n_total})
    clinical_summary = pd.DataFrame(clinical_rows)
    save_dataframe(clinical_summary, out_dir / "clinical_summary.csv")
    write_inference_markdown(
        [
            "Demographic and clinical context is summarized before connectome inference so group effects can be interpreted against cohort composition.",
            "Age and clinical scales use the same robust omnibus and pairwise framework as the imaging analyses.",
        ],
        out_dir / "demographics_inference.md",
    )
    return {"counts": counts, "clinical_summary": clinical_summary, "out_dir": out_dir}


def run_data_completeness(
    paths: AnalysisPaths,
    master: pd.DataFrame,
    strict_tracks_count: bool = False,
    track_minimum: int = 3_000_000,
) -> dict:
    from connectome_analysis.analysis_plots import save_dataframe, write_inference_markdown
    import shutil
    import subprocess as sp

    out_dir = paths.section_dir("01", "qc")
    out_dir.mkdir(parents=True, exist_ok=True)
    subjects = master[["subject_id", "group"]].drop_duplicates().copy()

    def _parse_weights_txt(path: Path) -> bool:
        if not path.exists():
            return False
        try:
            if path.stat().st_size <= 0:
                return False
            with path.open("r", encoding="utf-8", errors="ignore") as handle:
                sample = handle.read(8192)
            for line in sample.splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                for token in line.split():
                    try:
                        value = float(token)
                    except ValueError:
                        continue
                    if np.isfinite(value):
                        return True
            # SIFT2 files can contain a long command-history header followed by
            # millions of weights. A nonempty file with a normal size is enough
            # for the notebook-level inventory; post-stage QC owns strict checks.
            return path.stat().st_size > 1024
        except Exception:
            return False

    def _tck_count_ok(path: Path, minimum: int = 3_000_000) -> bool:
        if not path.exists():
            return False
        if not strict_tracks_count:
            return True
        tckinfo = shutil.which("tckinfo")
        if not tckinfo:
            return True
        try:
            out = sp.run(
                [tckinfo, "-count", str(path)],
                text=True,
                stdout=sp.PIPE,
                stderr=sp.DEVNULL,
                check=False,
            ).stdout
            for line in out.splitlines():
                if line.strip().startswith("count:"):
                    return int(line.split()[-1]) >= minimum
        except Exception:
            return True
        return True

    bias_subject_map: dict[str, Path] = {}
    for dwi in sorted(paths.deriv_root.joinpath("biascorr_1").glob("*_unbiased.mif")):
        if dwi.name.startswith("._"):
            continue
        sid = dwi.name.replace("_unbiased.mif", "")
        subject_key = sid.split("_I", 1)[0]
        bias_subject_map.setdefault(subject_key, dwi)
    # Resolve the full sid (with _I image-id) from the connectome too, so subjects whose biascorr
    # intermediate was cleaned still resolve their stage dirs (the connectome is the source of truth).
    conn_sid_map: dict[str, str] = {}
    for cc in sorted(paths.deriv_root.joinpath("connectomes").glob("SC_AAL166_*_count.csv")):
        if cc.name.startswith("._") or "_invnodevol" in cc.name:
            continue
        m = re.match(r"SC_AAL166_(\d{3}_S_\d+_I\d+)_count\.csv$", cc.name)
        if m:
            conn_sid_map.setdefault(m.group(1).split("_I", 1)[0], m.group(1))

    def _paths_for_subject(subject_id: str) -> dict[str, Path]:
        dwi = bias_subject_map.get(subject_id)
        sid = dwi.name.replace("_unbiased.mif", "") if dwi else conn_sid_map.get(subject_id)
        if not sid:
            return {}
        dwi = dwi if dwi else (paths.deriv_root / "biascorr_1" / f"{sid}_unbiased.mif")
        bbr_root = paths.deriv_root / "dwi_t1_bbr"
        fod_dir = paths.deriv_root / "fod" / sid
        tracks_dir = paths.deriv_root / "tracks" / sid
        dti_dir = paths.deriv_root / "dti" / sid
        parc_dir = paths.deriv_root / "parc" / sid
        conn_dir = paths.connectomes_dir
        return {
            "sid": Path(sid),
            "DWI": dwi,
            "B0MEAN": bbr_root / f"{sid}_b0mean_ras.nii.gz",
            "T1_RAS": bbr_root / f"{sid}_t1_ras.nii.gz",
            "T1B": bbr_root / f"{sid}_t1brain_ras.nii.gz",
            "T12B0": bbr_root / f"{sid}_t12b0_bbr.mat",
            "FIVE_FIX": fod_dir / "5tt_b0_fixed.mif",
            "MASK_5TT_BRAIN": fod_dir / "mask_5tt_brain.mif",
            "GMWMI": fod_dir / "gmwmi.mif",
            "WMFOD_FINAL": fod_dir / "wmfod_final.mif",
            "TRACKS_FINAL": tracks_dir / "tracks_final_3000k.tck",
            "WEIGHTS": tracks_dir / "sift_weights.txt",
            "AAL_B0": parc_dir / "AAL_b0.nii.gz",
            "FA_MIF": dti_dir / "fa.mif",
            "MD_MIF": dti_dir / "md.mif",
            "AD_MIF": dti_dir / "ad.mif",
            "RD_MIF": dti_dir / "rd.mif",
            "SC_ALL": conn_dir / f"SC_AAL166_{sid}_ALL.csv",
            "SC_COUNT": conn_dir / f"SC_AAL166_{sid}_count.csv",
        }

    per_subject_rows = []
    for subject_id, group in subjects[["subject_id", "group"]].itertuples(index=False):
        pmap = _paths_for_subject(subject_id)
        if not pmap:
            per_subject_rows.append(
                {
                    "subject_id": subject_id,
                    "group": group,
                    "has_bias_subject": 0,
                    "bbr_ready": 0,
                    "step7_prep_done": 0,
                    "final_connectome_ready": 0,
                }
            )
            continue
        bbr_ready = all(pmap[key].exists() for key in ("DWI", "B0MEAN", "T1_RAS", "T1B", "T12B0"))
        prep_done = all(pmap[key].exists() for key in ("FIVE_FIX", "MASK_5TT_BRAIN", "GMWMI"))
        fod_done = pmap["WMFOD_FINAL"].exists()
        tracks_done = _tck_count_ok(pmap["TRACKS_FINAL"], minimum=track_minimum)
        sift2_done = _parse_weights_txt(pmap["WEIGHTS"])
        parc_done = pmap["AAL_B0"].exists()
        dti_done = all(pmap[key].exists() for key in ("FA_MIF", "MD_MIF", "AD_MIF", "RD_MIF"))
        final_ready = pmap["SC_COUNT"].exists()  # the canonical per-subject connectome (the _ALL stack is legacy/absent)
        per_subject_rows.append(
            {
                "subject_id": subject_id,
                "group": group,
                "has_bias_subject": 1,
                "bbr_ready": int(bbr_ready),
                "step7_prep_done": int(prep_done),
                "step7_fod_done": int(fod_done),
                "step7_tracks_done": int(tracks_done),
                "step7_sift2_done": int(sift2_done),
                "step7_parc_done": int(parc_done),
                "step7_dti_done": int(dti_done),
                "final_connectome_ready": int(final_ready),
            }
        )
    merged = pd.DataFrame(per_subject_rows)
    completeness = (
        merged.groupby("group")
        .agg(
            n_subjects=("subject_id", "nunique"),
            n_bias_subjects=("has_bias_subject", "sum"),
            n_bbr_ready=("bbr_ready", "sum"),
            n_step7_prep_done=("step7_prep_done", "sum"),
            n_step7_fod_done=("step7_fod_done", "sum"),
            n_step7_tracks_done=("step7_tracks_done", "sum"),
            n_step7_sift2_done=("step7_sift2_done", "sum"),
            n_step7_parc_done=("step7_parc_done", "sum"),
            n_step7_dti_done=("step7_dti_done", "sum"),
            n_with_final_connectomes=("final_connectome_ready", "sum"),
        )
        .reindex(GROUP_ORDER)
        .reset_index()
    )
    for col in [
        "n_bias_subjects",
        "n_bbr_ready",
        "n_step7_prep_done",
        "n_step7_fod_done",
        "n_step7_tracks_done",
        "n_step7_sift2_done",
        "n_step7_parc_done",
        "n_step7_dti_done",
        "n_with_final_connectomes",
    ]:
        completeness[f"{col}_pct"] = np.where(
            completeness["n_subjects"] > 0,
            100.0 * completeness[col] / completeness["n_subjects"],
            np.nan,
        )
    total_row = {"group": "TOTAL"}
    for col in [
        "n_subjects",
        "n_bias_subjects",
        "n_bbr_ready",
        "n_step7_prep_done",
        "n_step7_fod_done",
        "n_step7_tracks_done",
        "n_step7_sift2_done",
        "n_step7_parc_done",
        "n_step7_dti_done",
        "n_with_final_connectomes",
    ]:
        total_row[col] = int(completeness[col].sum())
        total_row[f"{col}_pct"] = (
            100.0 * total_row[col] / total_row["n_subjects"] if total_row["n_subjects"] else np.nan
        )
    completeness = pd.concat([completeness, pd.DataFrame([total_row])], ignore_index=True)
    save_dataframe(merged, out_dir / "data_completeness_subject_level.csv")
    save_dataframe(completeness, out_dir / "data_completeness.csv")
    write_inference_markdown(
        [
            "Completeness is tracked at the subject level so later group differences can be interpreted against data availability.",
            "This table separates bias-subject availability, BBR readiness, Step 7 prep/FOD/tracks/SIFT2/parcellation/DTI completion, and final connectome completion.",
            "The moving Step 7 progress bar in notebook B is not a final connectome count; only the final SC_ALL files contribute to n_with_final_connectomes.",
        ],
        out_dir / "qc_inference.md",
    )
    return {
        "completeness": completeness,
        "out_dir": out_dir,
        "strict_tracks_count": strict_tracks_count,
        "track_minimum": track_minimum,
    }
