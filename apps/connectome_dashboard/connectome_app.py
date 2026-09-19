from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from html import escape
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from scipy import stats


APP_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = APP_ROOT.parent.parent
ANALYSIS_ROOT = Path(os.environ.get("CONNECTOME_ANALYSIS_ROOT", PROJECT_ROOT / "data" / "derivatives" / "qc" / "analysis_cohort"))
CONNECTOMES_DIR = Path(os.environ.get("CONNECTOME_MATRICES_DIR", PROJECT_ROOT / "data" / "derivatives" / "connectomes"))
ENHANCED_ML_ROOT = Path(os.environ.get("CONNECTOME_ENHANCED_ML_ROOT", PROJECT_ROOT / "outputs"))
HCP379_LIVE_SUMMARY = (
    PROJECT_ROOT
    / "research_audit/outputs/hcp379_live_530_status_density_v1/summary.json"
)
HCP379_LIVE_LEDGER = (
    PROJECT_ROOT
    / "research_audit/outputs/hcp379_pretract_failure_recovery_ledger_v1/ledger.csv"
)
HCP379_RELEASE_TOPOLOGY = (
    PROJECT_ROOT
    / "research_audit/outputs/hcp379_balanced_release_topology_v1/topology.json"
)
HCP379_ROOT_CAUSE_SUMMARY = (
    PROJECT_ROOT
    / "research_audit/outputs/hcp379_backward_root_cause_audit_v1/summary.json"
)
HCP379_GMWMI_DECISION = (
    PROJECT_ROOT
    / "research_audit/outputs/hcp379_gmwmi_seeding_counterfactual_v1/"
    "decision.json"
)
HCP379_GMWMI_ATTEMPTS = Path(
    "/data/derivatives/hcp379_v2/"
    "phase_b_hroi_gmwmi_seeding_counterfactual_v1/attempts"
)
HCP379_FOD_CUTOFF_PREDECLARATION = (
    PROJECT_ROOT
    / "research_audit/outputs/hcp379_fod_cutoff_counterfactual_v1/"
    "predeclaration.json"
)
HCP379_FOD_CUTOFF_VALIDATION = (
    PROJECT_ROOT
    / "research_audit/outputs/hcp379_fod_cutoff_counterfactual_v1/"
    "predeclaration_validation.json"
)
HCP379_FOD_CUTOFF_ATTEMPTS = Path(
    "/data/derivatives/hcp379_v2/"
    "phase_b_hroi_fod_cutoff_counterfactual_v1/attempts"
)
AAL3_RESCUE_BATCH_ROOT = Path(
    os.environ.get("AAL3_RESCUE_BATCH_ROOT", "/data/derivatives/qc/sc_matrix_qc/aal3_force_connectome_ad_batch")
)
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "structural_connectome_C.ipynb"
# Keyed on the matrix row (node 1..166). AAL3_labels.csv is keyed on the
# original atlas value and named every row from 35 onward wrongly.
AAL_LABEL_CSV = PROJECT_ROOT / "atlas" / "AAL" / "aal3_labels_166.csv"
AAL_ATLAS_NII = PROJECT_ROOT / "atlas" / "AAL" / "AAL3v1_1mm.nii.gz"
MNI_BRAIN_MASK_ENV = os.environ.get("CONNECTOME_MNI_BRAIN_MASK")
MNI_BRAIN_MASK_CANDIDATES = tuple([Path(MNI_BRAIN_MASK_ENV)] if MNI_BRAIN_MASK_ENV else []) + (
    Path("/home/ec2-user/fsl/data/standard/MNI152_T1_2mm_brain_mask.nii.gz"),
    Path("/home/ec2-user/fsl/data/standard/MNI152_T1_1mm_brain_mask.nii.gz"),
    AAL_ATLAS_NII,
)


@dataclass(frozen=True)
class Section:
    label: str
    description: str
    folders: tuple[str, ...]
    preferred_tables: tuple[str, ...] = ()
    show_subject_tables: bool = False


SECTIONS = [
    Section("Demographics", "Group counts, demographics, QC, and clinical score summaries.", ("00_master", "01_qc", "02_demographics")),
    Section(
        "Novel Findings",
        "PPT-style synthesis of positive exploratory findings across AAL3, distance/EDR, and clinical prediction.",
        (
            "03_global_microstructure_live",
            "04_node_microstructure_live",
            "06_global_graph_live",
            "07_node_graph_live",
            "10_edgewise_fd_sum",
            "11_brain_age_live",
            "12_length_delay",
            "13_delay",
            "15_advanced_structural",
            "17_edr_exceptions",
            "18_ml_diagnostics",
        ),
    ),
    Section("Global DTI", "Global FA/MD/RD/AD live microstructure summaries.", ("03_global_microstructure_live", "03_global_rd")),
    Section("Local / ROI DTI", "Node-wise FA/MD live microstructure with FDR-controlled ROI summaries.", ("04_node_microstructure_live", "04_local_rd")),
    Section("Global Graph", "Global strength, density, efficiency, and path-length summaries.", ("06_global_graph_live", "06_global_graph")),
    Section("SC Matrix Viewer", "Single-subject structural-connectome matrix QC and summaries.", ("00_master", "01_qc", "03_global_microstructure_live", "06_global_graph_live", "07_node_graph_live", "09_coupling_live", "11_brain_age_live", "12_length_delay", "13_delay", "17_edr_exceptions", "18_ml_diagnostics")),
    Section("Node Metrics", "Node-wise graph metrics and centrality-style live summaries.", ("07_node_graph_live", "07_node_metrics", "08_centrality")),
    Section("Coupling", "Structure-microstructure coupling across AAL nodes; structural only, not FC.", ("09_coupling_live", "09_coupling")),
    Section("Coupling-AAL", "Regional AAL-level structure-microstructure coupling rankings.", ("09_coupling_live", "04_node_microstructure_live", "07_node_graph_live")),
    Section("Brain Age", "Live structural brain-age and brain-age-gap summaries.", ("11_brain_age_live", "11_brain_age")),
    Section("LR / SR", "Short-range vs long-range structural pathway analyses.", ("12_length_delay",)),
    Section(
        "LR-SR Analysis",
        "Compact work-along analysis of tract lengths, model inputs, feature EDA, and selected LR/SR metrics.",
        ("12_length_delay", "18_ml_diagnostics"),
    ),
    Section("EDR Exceptions", "Edge-distance relationship exceptions and structural compensation signatures.", ("17_edr_exceptions",)),
    Section("ML Diagnostics", "Cross-validated structural prediction models and feature signatures.", ("18_ml_diagnostics",)),
    Section("Delay", "Length-derived delay-proxy analyses.", ("13_delay",)),
    Section("Advanced", "EDR residuals, compensation index, long-range vulnerability, and disease-gradient maps.", ("15_advanced_structural",)),
    Section("Network Analysis", "Network/system-level aggregation of all feature families + within/between connectivity; which network is most affected CN→AD.", ("19_network_analysis",)),
    Section("Functional Pending", "Functional coupling interface placeholder until real FC/EEG/MEG matrices are supplied.", ("16_functional_placeholder",)),
]

SECTION_GROUPS = (
    ("General", ("Demographics", "Novel Findings")),
    ("Graph Measures", ("Global DTI", "Local / ROI DTI", "Global Graph", "Node Metrics")),
    ("Functional Measures", ("Coupling", "Coupling-AAL", "Delay")),
    ("Tractography", ("SC Matrix Viewer", "LR / SR", "LR-SR Analysis", "EDR Exceptions")),
    ("Network Measures", ("Network Analysis",)),
    ("Advanced", ("Brain Age", "ML Diagnostics", "Advanced", "Functional Pending")),
)


RAW_TABLE_HINTS = (
    "master_cohort",
    "subject_level",
    "subject_table",
    "subject_connectome",
    "edge_level",
    "node_level",
    "predictions",
    "long.csv",
    "node_graph_long",
    "node_microstructure_long",
)

GROUP_ORDER = ("CN", "MCI", "AD")
GROUP_COLORS = ["#5dade2", "#9ccc65", "#ef5350"]
REGRESSION_SCORE_TARGETS = ("MMSE Total Score",)
CDR_CLASSIFICATION_TARGET = "Global CDR"
ACTIVE_SCORE_TARGETS = (CDR_CLASSIFICATION_TARGET, *REGRESSION_SCORE_TARGETS)
CDR_CLASS_ORDER = ("0", "0.5", "1", "2+")
PRIMARY_SCORE_MODE = "primary_scan_aligned_730d"
HISTORY_SCORE_MODE = "sensitivity_all_available_history"
SCORE_MODE_ORDER = (HISTORY_SCORE_MODE, PRIMARY_SCORE_MODE)
ENHANCED_ML_RUNS = (
    ("clinical assisted", "enhanced_goal_search_clinical_only"),
    ("combined disease fast", "enhanced_goal_search_disease_combined_fast"),
    ("primary score combined fast", "enhanced_goal_search_score_primary_combined_fast"),
    ("neural clinical", "enhanced_neural_benchmarks_clinical"),
    ("neural structural compact", "enhanced_neural_benchmarks_structural_compact"),
)
CONNECTOME_WEIGHTS = ("fd_sum", "count", "count_invnodevol", "invlen_mean", "len_mean", "fa_mean", "md_mean", "ad_mean", "rd_mean")
AAL3_RESCUE_ROUTE_SUFFIXES = (
    "",
    "_route2",
    "_route3_assignment",
    "_route4_label_rescue",
    "_route5_deep_assignment",
    "_route6_all_voxels_diagnostic",
    "_route7_selective_track_rescue",
    "_route8_full_t1_fnirt",
    "_route9_route8_assignment_retry",
)
EDGEWISE_PAIRS = ("CNvsAD", "MCIvsAD", "CNvsMCI")
NODE_GRAPH_METRICS = ("strength", "degree", "nodal_eff")
EDR_NODE_RANK_METRICS = (
    "sr_exception_rate",
    "lr_exception_rate",
    "sr_exception_strength",
    "lr_exception_strength",
    "sr_lr_exception_rate_diff",
)
EDR_SUBJECT_METRIC_GROUPS = (
    ("EDR Fit", ("edr_lambda", "edr_slope", "edr_intercept", "edr_r", "edr_p")),
    (
        "Exception Rate",
        ("exception_pct", "sr_exception_pct", "mr_exception_pct", "lr_exception_pct", "sr_lr_exception_pct_diff"),
    ),
    (
        "Exception Strength",
        (
            "sr_exception_strength",
            "mr_exception_strength",
            "lr_exception_strength",
            "sr_lr_exception_strength_ratio",
        ),
    ),
)
DELAY_SUBJECT_METRIC_GROUPS = (
    ("Delay Distribution", ("mean_delay_ms", "median_delay_ms", "p90_delay_ms")),
    ("Burden / Long Delay", ("delay_burden_ms", "long_delay_fraction")),
    ("Weighted / Path", ("delay_weighted_strength", "delay_path_ms")),
)
DELAY_NODE_RANK_METRICS = (
    "node_mean_delay_ms",
    "node_p90_delay_ms",
    "node_delay_burden_ms",
    "node_long_delay_fraction",
    "node_delay_weighted_strength",
)
ADVANCED_SUBJECT_METRIC_GROUPS = (
    ("Compensation", ("structural_compensation_index", "short_range_preservation")),
    ("Long-Range Vulnerability", ("long_range_vulnerability", "edr_residual_long")),
    ("CN-Reference Residual Sensitivity", ("edr_residual_mean", "edr_residual_short", "edr_residual_medium")),
)
BRAIN_AGE_SUBJECT_METRIC_GROUPS = (
    ("BAG", ("BAG",)),
    ("Corrected BAG", ("BAG_age_corrected",)),
    ("Prediction", ("brain_age_pred",)),
)
LR_SR_SUBJECT_METRIC_GROUPS = (
    ("Weight Mean", ("short_w_mean", "medium_w_mean", "long_w_mean", "intra_w_mean", "inter_w_mean")),
    ("Weight Sum", ("short_w_sum", "medium_w_sum", "long_w_sum")),
    ("Density", ("short_density", "medium_density", "long_density")),
    ("Efficiency", ("short_global_eff", "medium_global_eff", "long_global_eff")),
    ("Balance", ("sr_lr_w_diff", "sr_lr_w_ratio", "sr_lr_w_normdiff")),
    ("Delay Proxy", ("short_mean_delay_path_ms", "medium_mean_delay_path_ms", "long_mean_delay_path_ms")),
)
COUPLING_NODAL_METRICS = (
    ("degree", "Topology metric: Degree"),
    ("strength", "Topology metric: Strength"),
    ("nodal_eff", "Topology metric: Nodal Efficiency"),
)
COUPLING_MICRO_METRICS = (("md_mean", "MD"), ("fa_mean", "FA"))
COUPLING_AAL_VALUE_METRICS = ("coupling_index", "micro_topology_ratio")
CHART_BACKGROUND = "#f8fafc"
CHART_TEXT = "#0f172a"
CHART_MUTED = "#475569"
CHART_GRID = "#dbe3ef"
CHART_STROKE = "#cbd5e1"
LR_SR_CHART_GRID = "#94a3b8"
LR_SR_CHART_STROKE = "#475569"

CONNECTOME_WEIGHT_INFO = {
    "fd_sum": {
        "name": "Fibre-density weighted structural strength",
        "definition": "Sum of SIFT2 streamline weights assigned to an ROI pair. This is the primary structural connection weight used for the current graph metrics and edge-wise group tests.",
        "formula": "W_ij = sum_{s in E_ij} w_s",
        "detail": "E_ij is the set of streamlines assigned to ROIs i and j; w_s is the SIFT2 weight for streamline s.",
    },
    "count": {
        "name": "Streamline count",
        "definition": "Number of reconstructed streamlines assigned to an ROI pair, before SIFT2 weighting.",
        "formula": "W_ij = |E_ij|",
        "detail": "This is useful for QC, but raw counts are strongly affected by tractography and parcel-size biases.",
    },
    "count_invnodevol": {
        "name": "Node-volume normalized count",
        "definition": "Streamline count scaled by the inverse of the two endpoint node volumes using MRtrix scale_invnodevol.",
        "formula": "W_ij = sum_{s in E_ij} 2 / (V_i + V_j)",
        "detail": "V_i and V_j are the two AAL node volumes; this reduces parcel-size effects.",
    },
    "invlen_mean": {
        "name": "Mean inverse streamline length",
        "definition": "Average inverse length of streamlines assigned to an ROI pair.",
        "formula": "W_ij = mean_{s in E_ij}(1 / L_s)",
        "detail": "L_s is streamline length in mm. Larger values favor shorter pathways.",
    },
    "len_mean": {
        "name": "Mean streamline length",
        "definition": "Average streamline length for an ROI pair.",
        "formula": "W_ij = mean_{s in E_ij}(L_s)",
        "detail": "This is a pathway-length attribute, not a connection-strength measure.",
    },
    "fa_mean": {
        "name": "Mean edge FA",
        "definition": "Mean fractional anisotropy sampled along streamlines assigned to an ROI pair.",
        "formula": "W_ij = mean_{s in E_ij}(mean_{x in s} FA(x))",
        "detail": "This is an edge-wise microstructure attribute. It is not the same as structural strength.",
    },
    "md_mean": {
        "name": "Mean edge MD",
        "definition": "Mean diffusivity sampled along streamlines assigned to an ROI pair.",
        "formula": "W_ij = mean_{s in E_ij}(mean_{x in s} MD(x))",
        "detail": "This is an edge-wise microstructure attribute. Higher MD often indicates greater diffusivity or tissue rarefaction.",
    },
    "ad_mean": {
        "name": "Mean edge AD",
        "definition": "Axial diffusivity sampled along streamlines assigned to an ROI pair.",
        "formula": "W_ij = mean_{s in E_ij}(mean_{x in s} AD(x))",
        "detail": "AD is the principal diffusion-tensor eigenvalue. This is an edge-wise microstructure attribute, not a strength weight.",
    },
    "rd_mean": {
        "name": "Mean edge RD",
        "definition": "Radial diffusivity sampled along streamlines assigned to an ROI pair.",
        "formula": "W_ij = mean_{s in E_ij}(mean_{x in s} RD(x))",
        "detail": "RD is the average of the two non-principal diffusion-tensor eigenvalues. This is an edge-wise microstructure attribute.",
    },
}

GLOBAL_GRAPH_FORMULAS = (
    "node strength: s_i = sum_j W_ij",
    "mean strength: mean_i(s_i)",
    "total strength: 0.5 * sum_i(s_i)",
    "density: |{i < j: W_ij > 0}| / (N(N - 1) / 2)",
    "shortest-path cost: c_ij = 1 / W_ij for W_ij > 0",
    "global efficiency: mean(1 / d_ij) over finite shortest paths",
    "characteristic path length: mean(d_ij) over finite shortest paths",
)

SUBJECT_TABLE_BY_SECTION = {
    "Global DTI": "live_global_microstructure_subject_table.csv",
    "Global Graph": "live_global_graph_subject_table.csv",
    "SC Matrix Viewer": "live_global_graph_subject_table.csv",
    "Coupling": "live_subject_level_coupling.csv",
    "Brain Age": "live_brain_age_predictions.csv",
    "LR / SR": "lr_sr_subject_level_fd_sum_len_mean.csv",
    "LR-SR Analysis": "lr_sr_subject_level_fd_sum_len_mean.csv",
    "EDR Exceptions": "edr_exception_subject_level_fd_sum_len_mean.csv",
    "Delay": "delay_subject_level_fd_sum_len_mean.csv",
    "Advanced": "edr_subject_level_fd_sum_len_mean.csv",
}

LONG_NODE_TABLE_BY_SECTION = {
    "Local / ROI DTI": "live_node_microstructure_long.csv",
    "Node Metrics": "live_node_graph_long.csv",
}

PREFERRED_METRICS_BY_SECTION = {
    "Global DTI": (
        "fa_mean_edge_mean",
        "fa_mean_edge_median",
        "md_mean_edge_mean",
        "md_mean_edge_median",
        "rd_mean_edge_mean",
        "rd_mean_edge_median",
        "ad_mean_edge_mean",
        "ad_mean_edge_median",
    ),
    "Global Graph": ("mean_strength", "density", "global_efficiency", "charpath_len"),
    "Coupling": ("strength_fa_mean_rho", "strength_md_mean_rho", "degree_fa_mean_rho", "nodal_eff_fa_mean_rho"),
    "Brain Age": ("BAG", "BAG_age_corrected", "brain_age_pred"),
    "LR / SR": ("short_w_mean", "long_w_mean", "sr_lr_w_diff", "sr_lr_w_ratio", "short_global_eff", "long_global_eff"),
    "LR-SR Analysis": ("short_len_median", "long_len_median", "short_w_mean", "long_w_mean", "sr_lr_w_ratio"),
    "EDR Exceptions": ("edr_lambda", "sr_exception_pct", "lr_exception_pct", "sr_exception_strength", "lr_exception_strength", "sr_lr_exception_pct_diff"),
    "Delay": ("mean_delay_ms", "median_delay_ms", "p90_delay_ms", "delay_burden_ms", "long_delay_fraction", "delay_weighted_strength", "delay_path_ms"),
    "Advanced": ("edr_residual_mean", "edr_residual_short", "edr_residual_long", "structural_compensation_index", "long_range_vulnerability"),
}

NON_METRIC_COLUMNS = {
    "subject_id",
    "group",
    "age",
    "sex",
    "phase",
    "velocity_mm_per_ms",
    "p_subject",
}

DTI_FORMULAS = (
    r"\mathrm{AD} = \lambda_1",
    r"\mathrm{RD} = \frac{\lambda_2 + \lambda_3}{2}",
    r"\mathrm{MD} = \frac{\lambda_1 + \lambda_2 + \lambda_3}{3}",
    (
        r"\mathrm{FA} = \sqrt{\frac{3}{2}}"
        r"\frac{\sqrt{(\lambda_1-\mathrm{MD})^2 + (\lambda_2-\mathrm{MD})^2 + (\lambda_3-\mathrm{MD})^2}}"
        r"{\sqrt{\lambda_1^2 + \lambda_2^2 + \lambda_3^2}}"
    ),
)


st.set_page_config(
    page_title="Connectome C Dashboard",
    page_icon="SC",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
	    <style>
	    .stApp { background: #111316; color: #f3f4f6; }
	    .app-title {
	        text-align: center;
	        color: #facc15;
	        font-size: clamp(2.1rem, 4vw, 4rem);
	        font-weight: 950;
	        letter-spacing: 0;
	        margin: 0.4rem 0 1.2rem 0;
	    }
	    div[data-testid="stMetric"] { background: #1a1d22; border: 1px solid #2c313a; border-radius: 8px; padding: 12px; }
    div[data-testid="stDataFrame"] { border: 1px solid #2c313a; border-radius: 8px; }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-color: rgba(72, 148, 255, 0.28);
        background: linear-gradient(145deg, rgba(10, 18, 31, 0.94), rgba(16, 22, 30, 0.92));
        box-shadow: 0 0 0 1px rgba(72, 148, 255, 0.08), 0 0 18px rgba(37, 99, 235, 0.16);
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:hover {
        border-color: rgba(96, 165, 250, 0.58);
        box-shadow: 0 0 0 1px rgba(96, 165, 250, 0.18), 0 0 24px rgba(59, 130, 246, 0.28);
    }
    div.stButton > button {
        min-height: 3.5rem;
        padding: 0.35rem 0.65rem;
        border-radius: 8px;
        font-size: 0.95rem;
        font-weight: 850;
        background: linear-gradient(145deg, rgba(36, 112, 214, 0.28), rgba(13, 34, 63, 0.94));
        border: 1px solid rgba(96, 165, 250, 0.52);
        color: #eef7ff;
        box-shadow: 0 0 18px rgba(59, 130, 246, 0.18);
    }
    div.stButton > button:hover {
        border-color: rgba(125, 211, 252, 0.72);
        box-shadow: 0 0 18px rgba(56, 189, 248, 0.28);
        color: #ffffff;
    }
    div.stButton > button[kind="primary"] {
        background: linear-gradient(145deg, rgba(22, 163, 74, 0.9), rgba(13, 92, 55, 0.95));
        border-color: rgba(134, 239, 172, 0.82);
        color: #f3fff7;
        box-shadow: 0 0 22px rgba(34, 197, 94, 0.36);
    }
    div.stButton > button[kind="primary"]:hover {
        border-color: rgba(187, 247, 208, 0.95);
        box-shadow: 0 0 28px rgba(34, 197, 94, 0.48);
    }
    .section-note { color: #cbd5e1; font-size: 0.95rem; }
    .section-card-title {
        font-size: 0.88rem;
        font-weight: 800;
        color: #f8fafc;
        margin-bottom: 0.12rem;
        line-height: 1.1;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .section-card-desc {
        color: #cbd5e1;
        font-size: 0.72rem;
        line-height: 1.25;
        min-height: 2.25rem;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        overflow: hidden;
    }
    .section-card-meta {
        color: #9fb7d4;
        font-size: 0.68rem;
        line-height: 1.15;
        margin-top: 0.18rem;
        margin-bottom: 0.18rem;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .section-card-selected {
        color: #93c5fd;
        font-size: 0.66rem;
        font-weight: 800;
        letter-spacing: 0;
        margin-bottom: 0.16rem;
    }
    .analysis-group-title {
        color: #facc15;
        font-size: 0.98rem;
        font-weight: 950;
        letter-spacing: 0;
        text-transform: uppercase;
        margin: 0.1rem 0 0.7rem 0;
        text-align: center;
    }
    .analysis-group-note {
        color: #94a3b8;
        font-size: 0.72rem;
        line-height: 1.25;
        min-height: 2.1rem;
        text-align: center;
        margin: -0.25rem 0 0.8rem 0;
    }
    .lr-sr-section-title {
        color: #facc15;
        font-size: 1.38rem;
        font-weight: 950;
        letter-spacing: 0.075em;
        line-height: 1.2;
        margin: 0.55rem 0 0.65rem 0;
        text-align: center;
        text-transform: uppercase;
    }
    .lr-sr-section-title.compact {
        font-size: 1.12rem;
        margin: 0.2rem 0 0.5rem 0;
    }
    .lr-sr-mapping-note {
        background: #fff7d6;
        border: 1px solid #facc15;
        border-left: 5px solid #eab308;
        border-radius: 8px;
        color: #3f2f03;
        font-size: 0.92rem;
        line-height: 1.42;
        margin: 0.45rem 0 0.8rem 0;
        padding: 0.8rem 0.95rem;
    }
    .analysis-vsplit {
        width: 1px;
        min-height: 20rem;
        margin: 0.35rem auto 0 auto;
        background: linear-gradient(180deg, rgba(250, 204, 21, 0.0), rgba(250, 204, 21, 0.42), rgba(96, 165, 250, 0.22), rgba(250, 204, 21, 0.0));
    }
	    .small-muted { color: #94a3b8; font-size: 0.86rem; }
	    .chart-finding {
	        background: #eff6ff;
	        border: 1px solid #bfdbfe;
	        border-radius: 8px;
	        color: #0f172a;
	        padding: 0.9rem 1rem;
	        font-size: 1rem;
	        line-height: 1.45;
	    }
	    .result-card {
	        background: #f8fafc;
	        border: 1px solid #cbd5e1;
	        border-radius: 8px;
	        color: #0f172a;
	        padding: 1rem;
	    }
	    .result-card h4 {
	        margin: 0 0 0.45rem 0;
	        color: #0f172a;
	    }
		    .metric-explain {
		        color: #334155;
		        font-size: 0.9rem;
		        line-height: 1.35;
		        margin-bottom: 0.8rem;
		    }
		    .edr-definition-box {
		        background: #f8fafc;
		        border: 1px solid #cbd5e1;
		        border-radius: 8px;
		        color: #0f172a;
		        padding: 0.9rem 1rem;
		        margin-bottom: 0.85rem;
		    }
		    .edr-definition-box h5 {
		        color: #0f172a;
		        font-size: 1.02rem;
		        line-height: 1.2;
		        margin: 0 0 0.45rem 0;
		    }
		    .edr-definition-box p {
		        color: #334155;
		        font-size: 0.92rem;
		        line-height: 1.4;
		        margin: 0;
		    }
		    .edr-inference-box {
		        background: #fef3c7;
		        border: 1px solid #facc15;
		        border-left: 5px solid #eab308;
		        border-radius: 8px;
		        color: #3f2f03;
		        padding: 0.9rem 1rem;
		        font-size: 1rem;
		        font-weight: 700;
		        line-height: 1.45;
		        margin-bottom: 1.05rem;
		    }
		    .edr-table-gap {
		        height: 0.15rem;
		    }
		    .graph-guide-card h5 {
	        color: #1d4ed8;
	        font-size: 0.92rem;
	        margin: 0.85rem 0 0.25rem 0;
	    }
	    .formula-line {
	        background: #e0f2fe;
	        border: 1px solid #bae6fd;
	        border-radius: 6px;
	        color: #0f172a;
	        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
	        font-size: 0.82rem;
	        line-height: 1.35;
	        margin: 0.28rem 0;
	        padding: 0.35rem 0.5rem;
	    }
	    .formula-list {
	        margin-top: 0.35rem;
	    }
	    .formula-list .formula-line {
	        background: #f1f5f9;
	        border-color: #dbe3ef;
	        font-size: 0.78rem;
	    }
	    .definition-card {
	        background: #f8fafc;
	        border: 1px solid #cbd5e1;
	        border-radius: 8px;
	        color: #0f172a;
	        padding: 0.85rem 1rem;
	        margin: 0.35rem 0 0.85rem 0;
	    }
	    .definition-card h4 {
	        margin: 0 0 0.35rem 0;
	        color: #0f172a;
	    }
	    .definition-card p {
	        color: #334155;
	        font-size: 0.9rem;
	        line-height: 1.35;
	        margin: 0 0 0.45rem 0;
	    }
		    .definition-card .metric-formula {
		        background: #e0f2fe;
		        border: 1px solid #bae6fd;
	        border-radius: 6px;
	        color: #0f172a;
	        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
	        font-size: 0.82rem;
		        line-height: 1.35;
		        padding: 0.38rem 0.52rem;
		    }
		    .definition-table {
		        width: 100%;
		        border-collapse: collapse;
		        margin-top: 0.5rem;
		        font-size: 0.88rem;
		    }
		    .definition-table th {
		        background: #e2e8f0;
		        border: 1px solid #cbd5e1;
		        color: #0f172a;
		        font-weight: 800;
		        padding: 0.5rem 0.55rem;
		        text-align: left;
		    }
		    .definition-table td {
		        border: 1px solid #dbe3ef;
		        color: #1e293b;
		        padding: 0.48rem 0.55rem;
		        vertical-align: top;
		    }
		    .definition-table td.metric-name {
		        font-weight: 800;
		        white-space: nowrap;
		    }
		    .definition-table td.formula-cell {
		        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
		        font-size: 0.82rem;
		        color: #0f172a;
		    }
	    .definition-grid {
	        display: grid;
	        grid-template-columns: minmax(0, 1.35fr) minmax(320px, 0.9fr);
	        gap: 1rem;
	        align-items: start;
	    }
	    .definition-grid .definition-table {
	        table-layout: fixed;
	    }
	    .definition-grid .definition-table th,
	    .definition-grid .definition-table td {
	        white-space: normal;
	        overflow-wrap: anywhere;
	        word-break: break-word;
	    }
	    .definition-grid .definition-table td.metric-name {
	        white-space: normal;
	    }
	    @media (max-width: 1200px) {
	        .definition-grid {
	            grid-template-columns: 1fr;
	        }
	    }
	    .median-row {
	        display: grid;
	        grid-template-columns: minmax(150px, 1fr) 110px 130px;
	        column-gap: 1rem;
	        align-items: center;
	        border-top: 1px solid #e2e8f0;
	        padding: 0.35rem 0;
	        font-size: 0.95rem;
	    }
	    .median-value {
	        text-align: right;
	        font-variant-numeric: tabular-nums;
	        font-weight: 700;
	    }
	    .median-status {
	        display: flex;
	        justify-content: flex-end;
	    }
	    .median-chip {
	        border-radius: 999px;
	        padding: 0.08rem 0.5rem;
	        font-weight: 800;
	        min-width: 5.5rem;
	        text-align: center;
	    }
	    .chip-green { background: #dcfce7; color: #166534; }
	    .chip-red { background: #fee2e2; color: #991b1b; }
	    .chip-neutral { background: #e2e8f0; color: #334155; }
	    .roi-subsection {
	        border: 1px solid #dbeafe;
	        border-left: 4px solid #2563eb;
	        border-radius: 8px;
	        background: #f8fbff;
	        padding: 0.75rem 0.85rem;
	        margin-top: 0.85rem;
	    }
	    .roi-subsection h5 {
	        color: #1d4ed8;
	        font-size: 1rem;
	        font-weight: 900;
	        margin: 0 0 0.35rem 0;
	    }
	    .roi-inference-card {
	        background: linear-gradient(180deg, #f8fafc 0%, #eef6ff 100%);
	    }
	    .roi-inference-card h4 {
	        color: #12306b;
	        font-size: 1.2rem;
	        font-weight: 950;
	        border-bottom: 2px solid #bfdbfe;
	        padding-bottom: 0.45rem;
	    }
	    .roi-inference-lede {
	        color: #334155;
	        font-size: 0.88rem;
	        line-height: 1.35;
	        margin: 0.55rem 0 0.75rem 0;
	    }
	    .roi-result {
	        color: #0f172a;
	        background: #ecfdf5;
	        border: 1px solid #bbf7d0;
	        border-radius: 6px;
	        padding: 0.45rem 0.55rem;
	        margin: 0.35rem 0 0.45rem 0;
	    }
	    .roi-result b { color: #166534; }
	    .roi-subsection ol {
	        margin: 0.25rem 0 0 1.25rem;
	        padding: 0;
	    }
	    .roi-subsection li {
	        margin: 0.16rem 0;
	    }
	    .novel-hero {
	        border: 1px solid rgba(250, 204, 21, 0.45);
	        border-radius: 8px;
	        background: linear-gradient(135deg, #07111f 0%, #10243d 46%, #1a2b1f 100%);
	        color: #f8fafc;
	        padding: 1.35rem 1.45rem;
	        margin: 0 0 1rem 0;
	        box-shadow: 0 18px 42px rgba(0,0,0,0.34);
	    }
	    .novel-hero h2 {
	        color: #facc15;
	        font-size: clamp(1.85rem, 3vw, 3.2rem);
	        font-weight: 1000;
	        letter-spacing: 0;
	        margin: 0.15rem 0 0.45rem 0;
	        line-height: 1.05;
	    }
	    .novel-kicker {
	        color: #93c5fd;
	        font-size: 0.82rem;
	        font-weight: 1000;
	        letter-spacing: 0;
	        text-transform: uppercase;
	    }
	    .novel-claim {
	        color: #e5edf8;
	        font-size: 1.05rem;
	        line-height: 1.45;
	        max-width: 980px;
	    }
	    .novel-grid {
	        display: grid;
	        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
	        gap: 0.75rem;
	        margin-top: 1rem;
	    }
	    .novel-stat {
	        border: 1px solid rgba(148, 163, 184, 0.35);
	        border-radius: 8px;
	        background: rgba(15, 23, 42, 0.78);
	        padding: 0.82rem 0.9rem;
	    }
	    .novel-stat-label {
	        color: #bae6fd;
	        font-size: 0.72rem;
	        font-weight: 950;
	        text-transform: uppercase;
	    }
	    .novel-stat-value {
	        color: #ffffff;
	        font-size: 1.85rem;
	        font-weight: 1000;
	        line-height: 1.05;
	        margin-top: 0.22rem;
	    }
	    .novel-stat-note {
	        color: #cbd5e1;
	        font-size: 0.78rem;
	        line-height: 1.25;
	        margin-top: 0.3rem;
	    }
	    .novel-slide {
	        border: 1px solid #cbd5e1;
	        border-radius: 8px;
	        background: #f8fafc;
	        color: #0f172a;
	        padding: 1rem 1.1rem;
	        margin: 1rem 0;
	    }
	    .novel-slide h3 {
	        color: #0f172a;
	        font-size: 1.45rem;
	        font-weight: 1000;
	        letter-spacing: 0;
	        margin: 0.2rem 0 0.45rem 0;
	    }
	    .novel-slide p {
	        color: #334155;
	        font-size: 0.98rem;
	        line-height: 1.42;
	        margin: 0 0 0.7rem 0;
	    }
	    .novel-pill-row {
	        display: flex;
	        flex-wrap: wrap;
	        gap: 0.45rem;
	        margin: 0.7rem 0 0.25rem 0;
	    }
	    .novel-pill {
	        border: 1px solid #bfdbfe;
	        border-radius: 999px;
	        background: #eff6ff;
	        color: #1d4ed8;
	        font-size: 0.78rem;
	        font-weight: 900;
	        padding: 0.18rem 0.55rem;
	    }
	    .novel-source-list a {
	        color: #2563eb;
	        font-weight: 850;
	        text-decoration: none;
	    }
	    .novel-source-list li {
	        color: #334155;
	        margin: 0.25rem 0;
	    }
	    .novel-mechanism {
	        border: 1px solid #c7d2fe;
	        border-radius: 8px;
	        background: #f8fafc;
	        padding: 1rem;
	        margin: 1rem 0;
	    }
	    .novel-mechanism-title {
	        color: #0f172a;
	        font-size: 1.02rem;
	        font-weight: 1000;
	        margin: 0 0 0.75rem 0;
	    }
	    .novel-mechanism-flow {
	        display: grid;
	        grid-template-columns: repeat(5, minmax(120px, 1fr));
	        gap: 0.55rem;
	        align-items: stretch;
	    }
	    .novel-mechanism-step {
	        border: 1px solid #bfdbfe;
	        border-radius: 8px;
	        background: linear-gradient(180deg, #eff6ff 0%, #ffffff 100%);
	        color: #0f172a;
	        padding: 0.75rem;
	        min-height: 92px;
	        position: relative;
	    }
	    .novel-mechanism-step:not(:last-child)::after {
	        content: ">";
	        position: absolute;
	        right: -0.47rem;
	        top: 50%;
	        transform: translateY(-50%);
	        color: #2563eb;
	        font-weight: 1000;
	        font-size: 1.1rem;
	        z-index: 2;
	    }
	    .novel-mechanism-step b {
	        display: block;
	        color: #1d4ed8;
	        font-size: 0.8rem;
	        text-transform: uppercase;
	        margin-bottom: 0.35rem;
	    }
	    .novel-mechanism-step span {
	        color: #334155;
	        font-size: 0.86rem;
	        line-height: 1.28;
	    }
	    .novel-evidence-strip {
	        border: 1px solid #bbf7d0;
	        border-left: 5px solid #16a34a;
	        border-radius: 8px;
	        background: #f0fdf4;
	        color: #14532d;
	        padding: 0.62rem 0.8rem;
	        margin: -0.25rem 0 0.85rem 0;
	        display: grid;
	        grid-template-columns: minmax(180px, 0.55fr) minmax(240px, 1fr);
	        gap: 0.65rem;
	        align-items: center;
	    }
	    .novel-evidence-strip b {
	        color: #052e16;
	        font-weight: 1000;
	    }
	    .novel-evidence-source {
	        font-size: 0.78rem;
	        line-height: 1.25;
	    }
	    .novel-evidence-result {
	        font-size: 0.9rem;
	        line-height: 1.3;
	        font-weight: 850;
	    }
	    .novel-ribbon {
	        display: grid;
	        grid-template-columns: repeat(auto-fit, minmax(165px, 1fr));
	        gap: 0.65rem;
	        margin: 0.65rem 0 0.95rem 0;
	    }
	    .novel-ribbon-item {
	        border: 1px solid #bfdbfe;
	        border-radius: 8px;
	        background: #eff6ff;
	        color: #1e3a8a;
	        padding: 0.72rem 0.8rem;
	        font-size: 0.88rem;
	        font-weight: 1000;
	        text-align: center;
	    }
	    .novel-edge-guide {
	        display: grid;
	        grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
	        gap: 0.65rem;
	        margin: 0.75rem 0 0.95rem 0;
	    }
	    .novel-edge-guide-item {
	        border: 1px solid #cbd5e1;
	        border-radius: 8px;
	        background: #ffffff;
	        color: #0f172a;
	        padding: 0.74rem 0.82rem;
	    }
	    .novel-edge-guide-item b {
	        display: block;
	        color: #0f172a;
	        font-size: 0.78rem;
	        text-transform: uppercase;
	        margin-bottom: 0.28rem;
	    }
	    .novel-edge-guide-item span {
	        color: #334155;
	        font-size: 0.86rem;
	        line-height: 1.3;
	    }
	    .novel-figure-caption {
	        color: #cbd5e1;
	        font-size: 0.86rem;
	        line-height: 1.35;
	        margin: 0.2rem 0 0.8rem 0;
	    }
	    @media (max-width: 950px) {
	        .novel-mechanism-flow {
	            grid-template-columns: 1fr;
	        }
	        .novel-mechanism-step:not(:last-child)::after {
	            content: "";
	        }
	        .novel-evidence-strip {
	            grid-template-columns: 1fr;
	        }
	    }
	    </style>
    """,
    unsafe_allow_html=True,
)


def _safe_read_csv(path: Path) -> pd.DataFrame | None:
    try:
        return pd.read_csv(path)
    except Exception as exc:
        st.warning(f"Could not read `{path.name}`: {exc}")
        return None


def current_csv_path(csv_files: list[Path], name: str) -> Path | None:
    matches = [p for p in csv_files if p.name == name]
    if not matches:
        return None
    live = []
    for path in matches:
        try:
            parts = path.relative_to(ANALYSIS_ROOT).parts
        except ValueError:
            parts = path.parts
        if "archive" not in parts:
            live.append(path)
    candidates = live or matches
    return sorted(candidates, key=lambda p: (len(p.parts), p.as_posix()))[0]


@st.cache_data(ttl=120, show_spinner=False)
def enhanced_ml_table(file_names: tuple[str, ...]) -> pd.DataFrame:
    frames = []
    for label, folder in ENHANCED_ML_RUNS:
        run_dir = ENHANCED_ML_ROOT / folder
        for file_name in file_names:
            path = run_dir / file_name
            if not path.exists():
                continue
            try:
                df = pd.read_csv(path)
            except Exception:
                continue
            if df.empty:
                continue
            df = df.copy()
            df["run_source"] = label
            df["source_file"] = path.relative_to(PROJECT_ROOT).as_posix() if path.is_relative_to(PROJECT_ROOT) else str(path)
            frames.append(df)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


@st.cache_data(ttl=60)
def inventory(root_str: str) -> dict:
    root = Path(root_str)
    files = sorted(p for p in root.rglob("*") if p.is_file()) if root.exists() else []
    return {
        "root": root,
        "files": files,
        "png": [p for p in files if p.suffix.lower() == ".png"],
        "csv": [p for p in files if p.suffix.lower() == ".csv"],
        "md": [p for p in files if p.suffix.lower() == ".md"],
        "txt": [p for p in files if p.suffix.lower() == ".txt"],
    }


def files_for_folders(files: list[Path], folders: tuple[str, ...], suffixes: tuple[str, ...] | None = None) -> list[Path]:
    out = []
    for file in files:
        rel = file.relative_to(ANALYSIS_ROOT).as_posix()
        top = rel.split("/", 1)[0]
        if "" in folders or top in folders:
            if suffixes is None or file.suffix.lower() in suffixes:
                out.append(file)
    return sorted(out)


def is_raw_subject_table(path: Path) -> bool:
    name = path.name.lower()
    rel = path.relative_to(ANALYSIS_ROOT).as_posix().lower()
    return any(hint in name or hint in rel for hint in RAW_TABLE_HINTS)


def read_markdown(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore").strip()
    except Exception as exc:
        return f"Could not read {path.name}: {exc}"


@st.cache_data(ttl=60)
def group_summary(root_str: str) -> pd.DataFrame:
    root = Path(root_str)
    demographics_path = root / "02_demographics" / "group_counts.csv"
    completeness_path = root / "01_qc" / "data_completeness.csv"
    master_path = root / "00_master" / "master_cohort.csv"

    if demographics_path.exists():
        summary = pd.read_csv(demographics_path)
    elif master_path.exists():
        master = pd.read_csv(master_path)
        summary = master.groupby("group", dropna=False).agg(n_subjects=("subject_id", "nunique")).reset_index()
    else:
        return pd.DataFrame()

    summary["group"] = summary["group"].astype(str)
    summary = summary[summary["group"].isin(GROUP_ORDER)].copy()
    if completeness_path.exists():
        completeness = pd.read_csv(completeness_path)
        completeness["group"] = completeness["group"].astype(str)
        keep_cols = [
            "group",
            "n_step7_tracks_done",
            "n_step7_sift2_done",
            "n_step7_parc_done",
            "n_step7_dti_done",
            "n_with_final_connectomes",
        ]
        existing = [c for c in keep_cols if c in completeness.columns]
        summary = summary.merge(completeness[existing], on="group", how="left")

    order_map = {name: idx for idx, name in enumerate(GROUP_ORDER)}
    summary["_order"] = summary["group"].map(order_map)
    summary = summary.sort_values("_order").drop(columns="_order")
    return summary


@st.cache_data(ttl=60)
def refresh_status(root_str: str) -> dict:
    status_path = Path(root_str) / "00_master" / "dashboard_refresh_status.json"
    if not status_path.exists():
        return {}
    try:
        import json

        return json.loads(status_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def metric_label(metric: str) -> str:
    label = metric.replace("_", " ")
    replacements = {
        "fa": "FA",
        "md": "MD",
        "rd": "RD",
        "ad": "AD",
        "bag": "BAG",
        "edr": "EDR",
        "lr": "LR",
        "sr": "SR",
        "aal": "AAL",
    }
    words = [replacements.get(part.lower(), part) for part in label.split()]
    return " ".join(words)


def format_number(value: float | int | str | None, digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return "NA"
    try:
        value_float = float(value)
    except Exception:
        return str(value)
    if abs(value_float) >= 100:
        return f"{value_float:,.1f}"
    if 0 < abs(value_float) < 0.001:
        return f"{value_float:.2e}"
    return f"{value_float:.{digits}g}"


def format_p(value: float | int | str | None) -> str:
    if value is None or pd.isna(value):
        return "NA"
    try:
        value_float = float(value)
    except Exception:
        return str(value)
    if value_float < 0.001:
        return f"{value_float:.2e}"
    return f"{value_float:.3f}"


def group_count_text(counts: dict[str, int], plotted_counts: dict[str, int] | None = None) -> str:
    parts = []
    for group in GROUP_ORDER:
        total = int(counts.get(group, 0))
        plotted = int((plotted_counts or counts).get(group, total))
        if plotted != total:
            parts.append(f"{group}: n={plotted}/{total}")
        else:
            parts.append(f"{group}: n={total}")
    return " | ".join(parts)


def clean_table_for_display(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned = cleaned.replace(r"^\s*$", pd.NA, regex=True)
    cleaned = cleaned.dropna(axis=0, how="all").dropna(axis=1, how="all")
    return cleaned.reset_index(drop=True)


def compact_table_height(n_rows: int) -> int:
    return min(max(38 * (n_rows + 1), 120), 420)


def show_analysis_notes(paths: list[Path]) -> None:
    if not paths:
        return
    with st.expander("Additional analysis notes", expanded=False):
        for path in paths:
            st.caption(path.relative_to(ANALYSIS_ROOT).as_posix())
            st.markdown(read_markdown(path))


def numeric_metric_columns(df: pd.DataFrame, section_label: str) -> list[str]:
    preferred = list(PREFERRED_METRICS_BY_SECTION.get(section_label, ()))
    numeric_cols = []
    for col in df.columns:
        if col in NON_METRIC_COLUMNS:
            continue
        if col.startswith("n_") or col.endswith("_n_edges") or col.endswith("_count") or col.endswith("_edge_count"):
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        if series.notna().sum() >= 3:
            numeric_cols.append(col)
    ordered = [m for m in preferred if m in numeric_cols]
    ordered.extend([m for m in numeric_cols if m not in ordered])
    return ordered


def subject_table_for_section(section: Section, csv_files: list[Path]) -> Path | None:
    target = SUBJECT_TABLE_BY_SECTION.get(section.label)
    if not target:
        return None
    matches = [p for p in csv_files if p.name == target]
    return matches[0] if matches else None


def long_node_table_for_section(section: Section, csv_files: list[Path]) -> Path | None:
    target = LONG_NODE_TABLE_BY_SECTION.get(section.label)
    if not target:
        return None
    matches = [p for p in csv_files if p.name == target]
    return matches[0] if matches else None


@st.cache_data(ttl=300)
def aal_label_lookup(label_csv: str) -> pd.DataFrame:
    path = Path(label_csv)
    if not path.exists():
        return pd.DataFrame(columns=["node", "node_name", "atlas_label", "atlas_version"])
    df = pd.read_csv(path)
    if "node" in df.columns:
        df["node"] = pd.to_numeric(df["node"], errors="coerce").astype("Int64")
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def aal3_valid_label_values(label_csv: str) -> list[int]:
    labels = aal_label_lookup(label_csv)
    if labels.empty or "atlas_value" not in labels.columns:
        return []
    values = pd.to_numeric(labels["atlas_value"], errors="coerce").dropna().astype(int)
    return sorted(set(int(v) for v in values if int(v) > 0))


@st.cache_data(ttl=3600, show_spinner=False)
def aal_centroids(atlas_nii: str, label_csv: str) -> pd.DataFrame:
    atlas_path = Path(atlas_nii)
    labels = aal_label_lookup(label_csv)
    if not atlas_path.exists() or labels.empty or "atlas_value" not in labels.columns:
        return pd.DataFrame()
    try:
        import nibabel as nib
        import numpy as np
    except Exception:
        return pd.DataFrame()
    try:
        img = nib.load(str(atlas_path))
        data = np.asanyarray(img.dataobj)
        rows = []
        for _, label in labels.iterrows():
            value = pd.to_numeric(pd.Series([label.get("atlas_value")]), errors="coerce").iloc[0]
            if pd.isna(value):
                continue
            vox = np.argwhere(data == int(value))
            if vox.size == 0:
                continue
            x, y, z = nib.affines.apply_affine(img.affine, vox.mean(axis=0))
            rows.append(
                {
                    "node": int(label.get("node")),
                    "node_name": str(label.get("node_name")),
                    "atlas_label": str(label.get("atlas_label")),
                    "x": float(x),
                    "y": float(y),
                    "z": float(z),
                }
            )
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame()


def mni_brain_mask_path() -> Path | None:
    for path in MNI_BRAIN_MASK_CANDIDATES:
        if path.exists():
            return path
    return None


@st.cache_data(ttl=3600, show_spinner=False)
def mni_brain_shell_mesh(mask_nii: str, step: int = 2) -> dict[str, np.ndarray]:
    path = Path(mask_nii)
    if not path.exists():
        return {}
    try:
        import nibabel as nib
        from skimage.measure import marching_cubes
    except Exception:
        return {}
    try:
        img = nib.load(str(path))
        data = np.asanyarray(img.dataobj)
        mask = np.nan_to_num(data, nan=0.0) > 0
        if not mask.any():
            return {}
        downsample = max(1, int(step))
        if downsample > 1:
            mask = mask[::downsample, ::downsample, ::downsample]
        affine = img.affine @ np.diag([downsample, downsample, downsample, 1.0])
        verts, faces, _, _ = marching_cubes(mask.astype(np.float32), level=0.5)
        xyz = nib.affines.apply_affine(affine, verts)
        return {
            "x": xyz[:, 0].astype(float),
            "y": xyz[:, 1].astype(float),
            "z": xyz[:, 2].astype(float),
            "i": faces[:, 0].astype(np.int32),
            "j": faces[:, 1].astype(np.int32),
            "k": faces[:, 2].astype(np.int32),
        }
    except Exception:
        return {}


def related_metric_file(csv_files: list[Path], metric: str, suffix: str) -> Path | None:
    expected = f"{metric}{suffix}"
    matches = [p for p in csv_files if p.name == expected]
    return matches[0] if matches else None


def read_metric_table(csv_files: list[Path], metric: str, suffix: str) -> pd.DataFrame | None:
    path = related_metric_file(csv_files, metric, suffix)
    if path is None:
        return None
    df = _safe_read_csv(path)
    if df is None or df.empty:
        return None
    return df


def summary_row_for_metric(csv_files: list[Path], metric: str) -> pd.Series | None:
    for path in csv_files:
        if "summary" not in path.name.lower():
            continue
        df = _safe_read_csv(path)
        if df is None or "metric" not in df.columns:
            continue
        rows = df[df["metric"].astype(str) == metric]
        if not rows.empty:
            return rows.iloc[0]
    return None


def compute_descriptives_from_data(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if data.empty:
        return pd.DataFrame(columns=["group", "n", "mean", "median", "q1", "q3"])
    for group in GROUP_ORDER:
        values = pd.to_numeric(data.loc[data["Group"] == group, "Value"], errors="coerce").dropna()
        rows.append(
            {
                "group": group,
                "n": int(values.size),
                "mean": values.mean() if not values.empty else pd.NA,
                "median": values.quantile(0.50) if not values.empty else pd.NA,
                "q1": values.quantile(0.25) if not values.empty else pd.NA,
                "q3": values.quantile(0.75) if not values.empty else pd.NA,
            }
        )
    return pd.DataFrame(rows)


def metric_descriptives(csv_files: list[Path], metric: str, data: pd.DataFrame | None = None) -> pd.DataFrame:
    desc = read_metric_table(csv_files, metric, "_descriptives.csv")
    if desc is not None and {"group", "median", "q1", "q3"}.issubset(desc.columns):
        cols = [c for c in ("group", "n", "mean", "median", "q1", "q3") if c in desc.columns]
        out = desc[cols].copy()
        out["group"] = out["group"].astype(str)
        return out[out["group"].isin(GROUP_ORDER)].copy()
    if data is not None:
        return compute_descriptives_from_data(data)
    return pd.DataFrame(columns=["group", "n", "median", "q1", "q3"])


def metric_pairwise(csv_files: list[Path], metric: str) -> pd.DataFrame | None:
    return read_metric_table(csv_files, metric, "_pairwise.csv")


def pairwise_q_col(pairwise: pd.DataFrame | None) -> str | None:
    if pairwise is None:
        return None
    return next((col for col in ("bm_q", "mwu_q", "q", "welch_q", "kw_q") if col in pairwise.columns), None)


def pairwise_p_col(pairwise: pd.DataFrame | None) -> str | None:
    if pairwise is None:
        return None
    return next((col for col in ("bm_p", "mwu_p", "welch_p", "kw_p", "p_perm", "p") if col in pairwise.columns), None)


def summary_subtitle(csv_files: list[Path], metric: str) -> str:
    summary = summary_row_for_metric(csv_files, metric)
    if summary is None:
        return ""
    parts = []
    if "kw_p" in summary:
        parts.append(f"KW p={format_p(summary.get('kw_p'))}")
    if "perm_p" in summary:
        parts.append(f"perm p={format_p(summary.get('perm_p'))}")
    if "best_q" in summary:
        parts.append(f"best q={format_p(summary.get('best_q'))}")
    if "n_fdr_sig" in summary:
        parts.append(f"FDR nodes={format_number(summary.get('n_fdr_sig'), digits=0)}")
    return "Omnibus: " + " | ".join(parts) if parts else ""


def pairwise_subtitle(pairwise: pd.DataFrame | None) -> str:
    if pairwise is None or pairwise.empty:
        return ""
    q_col = pairwise_q_col(pairwise)
    p_col = pairwise_p_col(pairwise)
    stat_col = q_col or p_col
    if stat_col is None:
        return ""
    parts = []
    for _, row in pairwise.head(3).iterrows():
        group_a = str(row.get("group_a", "")).strip()
        group_b = str(row.get("group_b", "")).strip()
        label = f"{group_a}-{group_b}".strip("-") or "pair"
        prefix = "q" if stat_col == q_col and q_col else "p"
        effect = ""
        if "cliffs_delta" in row:
            effect = f", delta={format_number(row.get('cliffs_delta'), digits=2)}"
        parts.append(f"{label} {prefix}={format_p(row.get(stat_col))}{effect}")
    return "Pairwise: " + " | ".join(parts) if parts else ""


def q_value_lines(pairwise: pd.DataFrame | None) -> list[str]:
    if pairwise is None or pairwise.empty:
        return []
    q_col = pairwise_q_col(pairwise)
    if q_col is None:
        return []
    rows = pairwise.copy()
    rows["_q"] = pd.to_numeric(rows[q_col], errors="coerce")
    rows = rows[rows["_q"].notna()].sort_values("_q")
    if rows.empty:
        return []
    stars_by_rank = ["***", "**", "*"]
    lines = []
    for idx, (_, row) in enumerate(rows.head(3).iterrows()):
        group_a = str(row.get("group_a", "")).strip()
        group_b = str(row.get("group_b", "")).strip()
        pair = f"{group_a}-{group_b}".strip("-") or "pair"
        stars = stars_by_rank[idx] if idx < len(stars_by_rank) else "*"
        lines.append(f"{pair}: q={format_p(row.get(q_col))} {stars}")
    return lines


def chart_stat_subtitles(csv_files: list[Path], metric: str, extra: list[str] | None = None) -> list[str]:
    subtitles = [line for line in q_value_lines(metric_pairwise(csv_files, metric)) if line]
    if extra:
        subtitles.extend([line for line in extra if line])
    return subtitles[:3]


def significant_pairwise(pairwise: pd.DataFrame | None) -> pd.DataFrame:
    if pairwise is None or pairwise.empty:
        return pd.DataFrame()
    q_col = pairwise_q_col(pairwise)
    p_col = pairwise_p_col(pairwise)
    stat_col = q_col or p_col
    if stat_col is None:
        return pd.DataFrame()
    out = pairwise.copy()
    out["_stat"] = pd.to_numeric(out[stat_col], errors="coerce")
    threshold = 0.05
    return out[out["_stat"] < threshold].copy()


def one_line_finding(section: Section, metric: str, desc: pd.DataFrame, pairwise: pd.DataFrame | None) -> str:
    label = metric_label(metric)
    sig = significant_pairwise(pairwise)
    if sig.empty:
        return f"No FDR-significant pairwise group difference is detected for {label}."
    medians = {}
    if not desc.empty and {"group", "median"}.issubset(desc.columns):
        for _, row in desc.iterrows():
            group = str(row.get("group"))
            if group in GROUP_ORDER:
                value = pd.to_numeric(pd.Series([row.get("median")]), errors="coerce").iloc[0]
                if pd.notna(value):
                    medians[group] = float(value)
    if len(medians) >= 2:
        low_group = min(medians, key=medians.get)
        high_group = max(medians, key=medians.get)
        if "AD" in medians and medians["AD"] == medians[low_group]:
            return f"AD shows lower {label} than the comparison groups, with FDR-significant pairwise differences."
        if "AD" in medians and medians["AD"] == medians[high_group]:
            return f"AD shows higher {label} than the comparison groups, with FDR-significant pairwise differences."
        return f"{low_group} shows lower {label} than {high_group}, with FDR-significant pairwise evidence."
    first = sig.iloc[0]
    return f"{first.get('group_a')} differs from {first.get('group_b')} for {label}, with FDR-significant pairwise evidence."


def metric_direction(metric: str) -> str:
    lower = metric.lower()
    higher_better_tokens = ("fa_", "_fa_", "strength", "degree", "eff", "density", "rho", "compensation")
    lower_better_tokens = ("md_", "_md_", "rd_", "_rd_", "delay", "charpath", "path_len", "bag", "vulnerability")
    if any(token in lower for token in lower_better_tokens):
        return "lower_better"
    if any(token in lower for token in higher_better_tokens):
        return "higher_better"
    return "neutral"


def metric_explanation(section: Section, metric: str) -> str:
    label = metric_label(metric)
    lower = metric.lower()
    if "edge_mean" in lower:
        return (
            f"{label} is the subject-level average across valid AAL connectome edges after the edge matrix is built; "
            "for DTI this means tensor-derived FA/MD values were sampled along streamlines and summarized per edge first."
        )
    if "edge_median" in lower:
        return (
            f"{label} is the subject-level median across valid AAL connectome edges after the edge matrix is built, "
            "so it is less sensitive to a small number of extreme edges than the edge mean."
        )
    if lower == "strength":
        return "Node strength is the sum of fd_sum structural weights connected to one AAL3 region: s_i = sum_j W_ij."
    if lower == "degree":
        return "Node degree is the number of nonzero structural edges incident on one AAL3 region: k_i = count_j(W_ij > 0)."
    if lower == "nodal_eff":
        return "Nodal efficiency is the mean inverse shortest-path distance from one AAL3 region to the rest of the weighted structural graph."
    if lower.startswith("mean_strength") or lower == "total_strength":
        return "Strength summarizes the total or average structural connection weight attached to the network for each subject."
    if lower == "density":
        return "Density is the fraction of possible AAL edges that are present after connectome construction."
    if "efficiency" in lower or lower.endswith("_eff"):
        return "Efficiency summarizes how easily information can traverse the weighted structural network through short paths."
    if "charpath" in lower or "path_len" in lower:
        return "Characteristic path length summarizes the average shortest-path distance through the weighted structural network."
    if lower.endswith("_rho"):
        nodal_metric, micro_metric = coupling_metric_parts(metric)
        if nodal_metric and micro_metric:
            return (
                f"{label} is a subject-level Spearman coupling coefficient: it correlates each AAL node's "
                f"{metric_label(nodal_metric)} value with that same node's {metric_label(micro_metric)} value."
            )
        return "Rho is the subject-level Spearman correlation between node-wise graph topology and local FA/MD microstructure."
    if lower == "bag":
        return "BAG is brain-age gap: predicted structural brain age minus chronological age."
    if lower == "bag_age_corrected":
        return "Age-corrected BAG is the residual brain-age gap after removing linear dependence on chronological age."
    if lower == "brain_age_pred":
        return "Predicted brain age is estimated from structural connectome features by the refreshed brain-age model."
    if lower == "mean_delay_ms":
        return "Mean delay is the subject-level average of D_ij = L_ij / v across positive weighted AAL3 edges."
    if lower == "median_delay_ms":
        return "Median delay is the subject-level median of D_ij = L_ij / v across positive weighted AAL3 edges."
    if lower == "p90_delay_ms":
        return "P90 delay is the 90th percentile of edge delay values and highlights the slowest tract-length-derived pathways."
    if lower == "delay_burden_ms":
        return "Delay burden is the fd_sum-weighted mean edge delay, so stronger structural edges contribute more to the subject summary."
    if lower == "long_delay_fraction":
        return "Long-delay fraction is the fraction of positive weighted edges whose delay exceeds the CN-reference long-delay threshold."
    if lower == "delay_weighted_strength":
        return "Delay weighted strength sums W_ij / D_ij and is larger when strong edges have shorter delay."
    if lower == "delay_path_ms":
        return "Delay path is the average shortest-path delay through the structural graph using D_ij as edge cost."
    if lower.startswith("node_") and "delay" in lower:
        return f"{label} is an AAL-region-level delay summary computed over positive weighted edges incident on that region."
    if lower == "edr_residual_mean":
        return (
            "For each ROI pair, the CN distance-decay model predicts the expected log edge weight from tract length. "
            "This metric averages observed-minus-expected residuals across edges; positive values mean stronger-than-CN-expected structure, "
            "and negative values mean weaker-than-CN-expected structure after accounting for edge length."
        )
    if lower in {"edr_residual_short", "edr_residual_medium", "edr_residual_long"}:
        return (
            f"{label} is the same CN-reference observed-minus-expected residual, but summarized only for {label.split()[-1].lower()}-range edges. "
            "This separates short local pathways from longer vulnerable pathways instead of mixing all tract lengths together."
        )
    if lower == "structural_compensation_index":
        return (
            "Structural compensation index is short-range residual minus long-range residual. A higher value means short-range structure is relatively preserved "
            "while long-range structure is relatively reduced, which is a structural compensation-style pattern."
        )
    if lower == "long_range_vulnerability":
        return (
            "Long-range vulnerability is the negative of the long-range residual. Higher values mean long-range edges are weaker than expected from the CN reference curve "
            "after accounting for their length."
        )
    if lower == "short_range_preservation":
        return (
            "Short-range preservation is the short-range residual from the CN-reference distance-decay model. Higher values mean local/short edges are stronger than expected "
            "for their length compared with the CN reference."
        )
    if lower == "edr_lambda":
        return "EDR lambda is the subject-wise distance-decay coefficient from log(W_ij) = alpha - lambda * L_ij; larger lambda means steeper weight decay with edge length."
    if lower == "edr_slope":
        return "EDR slope is the fitted slope of log edge weight against edge length for each subject; more negative values mean stronger distance decay."
    if lower == "edr_intercept":
        return "EDR intercept is the fitted alpha term in the subject-wise edge-distance model; it reflects the baseline log edge weight at zero length extrapolation."
    if lower == "edr_r":
        return "EDR r is the correlation from the subject-wise edge-distance regression; values farther from zero indicate a stronger length-weight relationship."
    if lower == "edr_p":
        return "EDR p is the p-value for the subject-wise edge-distance regression slope; it summarizes whether length and log edge weight are related within that subject."
    if lower == "exception_pct":
        return "Exception pct is n_exceptions / n_valid_edges across all valid AAL edges in a subject. The plotted value is a fraction, so 0.02 means about 2% of valid edges are unusually high-weight exceptions."
    if lower == "sr_exception_rate":
        return (
            "SR exception rate is the AAL-region-level fraction of short-range candidate edges attached to that region "
            "that were flagged as unusually high-weight EDR exceptions."
        )
    if lower == "lr_exception_rate":
        return (
            "LR exception rate is the AAL-region-level fraction of long-range candidate edges attached to that region "
            "that were flagged as unusually high-weight EDR exceptions."
        )
    if lower == "sr_exception_pct":
        return (
            "SR exception pct is sr_exception_count / sr_edge_count for short-range candidate edges. "
            "The plotted value is a fraction, so 0.026 means about 2.6%. An exception is not merely a normal strong short edge; "
            "it is an edge whose fd_sum is unusually high relative to that subject's other edges of similar length."
        )
    if lower == "lr_exception_pct":
        return (
            "LR exception pct is lr_exception_count / lr_edge_count for long-range candidate edges. "
            "The plotted value is a fraction; low values are expected because the exception rule is deliberately stringent."
        )
    if lower == "mr_exception_pct":
        return (
            "MR exception pct is mr_exception_count / mr_edge_count for medium-range candidate edges. "
            "It measures the fraction of medium-length edges that are unusually high-weight for that subject."
        )
    if lower in {"sr_exception_strength", "mr_exception_strength", "lr_exception_strength"}:
        if lower.startswith("sr_"):
            range_name = "short-range"
        elif lower.startswith("mr_"):
            range_name = "medium-range"
        else:
            range_name = "long-range"
        return (
            f"{label} is the sum of fd_sum weights over flagged {range_name} exception edges. "
            "It measures burden/intensity of unusually high-weight edges, not the total weight of all edges in that range."
        )
    if lower == "sr_lr_exception_pct_diff":
        return "SR LR exception pct diff is sr_exception_pct - lr_exception_pct; positive values mean a subject has a higher fraction of short-range than long-range EDR exceptions."
    if lower == "sr_lr_exception_rate_diff":
        return "SR LR exception rate diff is the AAL-region-level short-range exception rate minus long-range exception rate; positive values indicate more short-range exception enrichment around that region."
    if lower == "sr_lr_exception_strength_ratio":
        return "SR LR exception strength ratio is sr_exception_strength / lr_exception_strength; larger values indicate exception burden is more short-range dominated."
    if "exception" in lower:
        return (
            "EDR exception metrics count or weight edges whose fd_sum is unusually high for their tract length within a "
            "subject-specific length bin, using the mean + 3 SD rule."
        )
    range_prefix = None
    if lower.startswith("short_"):
        range_prefix = "short-range"
    elif lower.startswith("medium_"):
        range_prefix = "medium-range"
    elif lower.startswith("long_"):
        range_prefix = "long-range"
    if range_prefix and lower.endswith("_w_mean"):
        return (
            f"{label} is the average fd_sum structural weight across length-defined {range_prefix} AAL3 candidate edges for each subject. "
            "Zero weights are retained here, so this reflects both connection presence and connection strength; density reports the nonzero-edge fraction."
        )
    if range_prefix and lower.endswith("_w_median"):
        return f"{label} is the median fd_sum structural weight across valid {range_prefix} AAL3 edges for each subject."
    if range_prefix and lower.endswith("_w_sum"):
        return f"{label} is the total fd_sum structural weight summed over length-defined {range_prefix} AAL3 candidate edges for each subject."
    if range_prefix and lower.endswith("_density"):
        return f"{label} is the fraction of possible {range_prefix} AAL3 edges that are present with nonzero fd_sum weight."
    if range_prefix and lower.endswith("_global_eff"):
        return f"{label} is global efficiency computed within the {range_prefix} subgraph using inverse fd_sum as path cost."
    if range_prefix and lower.endswith("_mean_delay_path_ms"):
        return f"{label} is a tract-length-derived conduction-delay proxy summarized within the {range_prefix} subgraph."
    if lower == "sr_lr_w_diff":
        return "SR LR w diff is short_w_mean - long_w_mean; positive values indicate stronger average short-range than long-range structural weights."
    if lower == "sr_lr_w_ratio":
        return "SR LR w ratio is short_w_mean / long_w_mean; larger values indicate short-range weight dominance relative to long-range edges."
    if lower == "sr_lr_w_normdiff":
        return "SR LR w normdiff is a normalized short-versus-long weight contrast, reducing scale effects from overall connection strength."
    if lower == "intra_w_mean":
        return "Intra w mean is the average fd_sum weight for within-hemisphere AAL3 edges for each subject."
    if lower == "inter_w_mean":
        return "Inter w mean is the average fd_sum weight for between-hemisphere AAL3 edges for each subject."
    if "short_" in lower or "long_" in lower or "sr_lr" in lower:
        return "Short-range/long-range metrics split edges by tract length, then summarize connection weight or topology within each range."
    if "delay" in lower:
        return "Delay metrics are length-derived proxies using tract length and assumed conduction velocity to summarize slower structural pathways."
    if "edr" in lower:
        return "EDR metrics summarize deviations from the control edge-distance relationship, highlighting distance-adjusted structural vulnerability."
    if "compensation" in lower:
        return "The compensation index summarizes preserved or increased structural organization relative to disease-vulnerable edge patterns."
    return f"{label} is calculated per subject from the refreshed CSV table for this section and compared across diagnostic groups."


def coupling_metric_parts(metric: str) -> tuple[str | None, str | None]:
    lower = metric.lower()
    if not lower.endswith("_rho"):
        return None, None
    stem = lower.removesuffix("_rho")
    for micro_metric in ("fa_mean", "md_mean", "ad_mean", "rd_mean"):
        suffix = f"_{micro_metric}"
        if stem.endswith(suffix):
            return stem.removesuffix(suffix), micro_metric
    return None, None


NODE_METRIC_DEFINITIONS = {
    "strength": {
        "title": "Node Strength",
        "definition": "For each AAL3 region, strength is the total structural connection weight attached to that node. In this dashboard it is computed from the fd_sum structural matrix.",
        "formula": r"s_i = \sum_{j=1}^{N} W_{ij}",
        "note": "W_ij is the SIFT2/fibre-density weighted edge between AAL nodes i and j.",
    },
    "degree": {
        "title": "Node Degree",
        "definition": "For each AAL3 region, degree is the number of other AAL nodes with a nonzero structural edge to that region.",
        "formula": r"k_i = \sum_{j=1}^{N} \mathbf{1}(W_{ij} > 0)",
        "note": "Degree is topological presence/absence; it ignores edge magnitude once the edge is nonzero.",
    },
    "nodal_eff": {
        "title": "Nodal Efficiency",
        "definition": "For each AAL3 region, nodal efficiency is the average inverse shortest-path distance from that node to all other reachable nodes.",
        "formula": r"E_i = \frac{1}{N-1}\sum_{j\ne i}\frac{1}{d_{ij}},\quad c_{ij}=1/W_{ij}",
        "note": "d_ij is the shortest-path distance using inverse edge weight as path cost.",
    },
    "rho": {
        "title": "Spearman Rho",
        "definition": "Rho is the within-subject rank correlation between a node-wise topology map and a node-wise microstructure map over matched AAL3 regions.",
        "formula": r"\rho_s = \mathrm{corr}_{rank}\left(T_i, M_i\right)_{i=1}^{N}",
        "note": "Positive rho means regions with higher topology values also tend to have higher microstructure values within that subject; negative rho means the two maps tend to vary in opposite directions.",
    },
}


def definition_table_html(rows: list[tuple[str, str, str]], headers: tuple[str, str, str]) -> str:
    return (
        "<table class='definition-table'>"
        "<thead><tr>"
        f"<th>{escape(headers[0])}</th><th>{escape(headers[1])}</th><th>{escape(headers[2])}</th>"
        "</tr></thead><tbody>"
        + "".join(
            "<tr>"
            f"<td class='metric-name'>{escape(first)}</td>"
            f"<td>{escape(second)}</td>"
            f"<td class='formula-cell'>{escape(third)}</td>"
            "</tr>"
            for first, second, third in rows
        )
        + "</tbody></table>"
    )


def definition_grid_html(left_title: str, left_html: str, right_title: str, right_html: str) -> str:
    return (
        "<div class='definition-grid'>"
        "<div>"
        f"<h5>{escape(left_title)}</h5>"
        f"{left_html}"
        "</div>"
        "<div>"
        f"<h5>{escape(right_title)}</h5>"
        f"{right_html}"
        "</div>"
        "</div>"
    )


def lr_sr_edge_count_summary(subject_path: Path, prefix: str) -> str:
    if not subject_path.exists():
        return "not available"
    subject = _safe_read_csv(subject_path)
    if subject is None or subject.empty:
        return "not available"
    col = f"{prefix}_edge_count"
    if col not in subject.columns:
        return "not available"
    values = pd.to_numeric(subject[col], errors="coerce").dropna()
    if values.empty:
        return "not available"
    med = values.median()
    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    total = values.sum()
    return (
        f"median {format_number(med, digits=1)} "
        f"[IQR {format_number(q1, digits=1)}-{format_number(q3, digits=1)}]; "
        f"total subject-edge observations {format_number(total, digits=0)}"
    )


def lr_sr_formula_for_metric(metric: str) -> str:
    lower = metric.lower()
    range_sets = {"short": "E_S", "medium": "E_M", "long": "E_L"}
    range_key = next((key for key in range_sets if lower.startswith(f"{key}_")), None)
    edge_set = range_sets.get(range_key or "", "E")
    if range_key and lower.endswith("_w_mean"):
        return f"mean_{{(i,j) in {edge_set}}}(W_ij)"
    if range_key and lower.endswith("_w_sum"):
        return f"sum_{{(i,j) in {edge_set}}}(W_ij)"
    if range_key and lower.endswith("_density"):
        return f"|{{(i,j) in {edge_set}: W_ij > 0}}| / |{edge_set}|"
    if range_key and lower.endswith("_global_eff"):
        return f"mean(1 / d_ij) within {edge_set}, with c_ij = 1 / W_ij"
    if range_key and lower.endswith("_mean_delay_path_ms"):
        return f"mean_{{(i,j) in {edge_set}}}(L_ij / v)"
    if lower == "sr_lr_w_diff":
        return "short_w_mean - long_w_mean"
    if lower == "sr_lr_w_ratio":
        return "short_w_mean / long_w_mean"
    if lower == "sr_lr_w_normdiff":
        return "(short_w_mean - long_w_mean) / (short_w_mean + long_w_mean)"
    if lower == "intra_w_mean":
        return "mean(W_ij) for same-hemisphere AAL3 edge candidates"
    if lower == "inter_w_mean":
        return "mean(W_ij) for opposite-hemisphere AAL3 edge candidates"
    return "see definition"


def lr_sr_metric_group_table_html(section: Section, metrics: list[str]) -> str:
    rows = [
        (metric_label(metric), metric_explanation(section, metric), lr_sr_formula_for_metric(metric))
        for metric in metrics
    ]
    return (
        "<div class='result-card graph-guide-card'>"
        "<h4>Selected LR/SR metric definitions</h4>"
        f"{definition_table_html(rows, ('Metric', 'Definition', 'Formula'))}"
        "</div>"
    )


def deterministic_sample(values: np.ndarray, max_values: int, seed: int) -> np.ndarray:
    values = values[np.isfinite(values)]
    if len(values) <= max_values:
        return values
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(values), size=max_values, replace=False)
    return values[np.sort(idx)]


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
        except Exception:
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
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def connectome_len_matrix_path(connectomes_dir: Path, subject_id: str, image_id: object) -> Path | None:
    return aal3_connectome_matrix_path(
        connectomes_dir,
        subject_id,
        image_id,
        "len_mean",
    )


def tract_length_summary_row(panel: str, values: np.ndarray, n_subjects: int) -> dict[str, object]:
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


@st.cache_data(ttl=1800, show_spinner=False)
def lr_sr_tract_length_distribution_data(
    root_str: str,
    connectomes_dir_str: str,
    max_group_values: int = 50_000,
    max_overall_values: int = 80_000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = Path(root_str)
    connectomes_dir = Path(connectomes_dir_str)
    master_path = root / "00_master" / "master_cohort.csv"
    if not master_path.exists() or not connectomes_dir.exists():
        return pd.DataFrame(), pd.DataFrame()
    master = pd.read_csv(master_path)
    required = {"subject_id", "group", "Image ID"}
    if not required.issubset(master.columns):
        return pd.DataFrame(), pd.DataFrame()
    lr_sr_subject_path = root / "12_length_delay" / "lr_sr_subject_level_fd_sum_len_mean.csv"
    if lr_sr_subject_path.exists():
        lr_subjects = pd.read_csv(lr_sr_subject_path, usecols=lambda col: col in {"subject_id", "group"})
        if {"subject_id", "group"}.issubset(lr_subjects.columns):
            lr_subjects = lr_subjects.rename(columns={"group": "lr_sr_group"})
            master = master.merge(lr_subjects, on="subject_id", how="inner")
            master["group"] = master["lr_sr_group"].fillna(master["group"])
    if "has_conn_len_mean" in master.columns:
        ready = pd.to_numeric(master["has_conn_len_mean"], errors="coerce").fillna(0) > 0
        master = master[ready].copy()
    master["group"] = master["group"].astype(str)
    master = master[master["group"].isin(GROUP_ORDER)].copy()

    values_by_group: dict[str, list[np.ndarray]] = {group: [] for group in GROUP_ORDER}
    subjects_by_group: dict[str, int] = {group: 0 for group in GROUP_ORDER}
    for _, row in master.iterrows():
        subject_id = str(row.get("subject_id", "")).strip()
        group = str(row.get("group", "")).strip()
        if not subject_id or group not in values_by_group:
            continue
        matrix_path = connectome_len_matrix_path(connectomes_dir, subject_id, row.get("Image ID"))
        if matrix_path is None:
            continue
        try:
            matrix = np.loadtxt(matrix_path, delimiter=",")
        except Exception:
            continue
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            continue
        tri = matrix[np.triu_indices_from(matrix, k=1)]
        tri = tri[np.isfinite(tri) & (tri > 0)]
        if tri.size == 0:
            continue
        values_by_group[group].append(tri.astype(float, copy=False))
        subjects_by_group[group] += 1

    sample_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    all_values: list[np.ndarray] = []
    for group in GROUP_ORDER:
        if not values_by_group[group]:
            continue
        values = np.concatenate(values_by_group[group])
        all_values.append(values)
        sample = deterministic_sample(values, max_group_values, seed=20260507 + GROUP_ORDER.index(group))
        sample_frames.append(pd.DataFrame({"Panel": group, "Group": group, "Length_mm": sample}))
        summary_rows.append(tract_length_summary_row(group, values, subjects_by_group[group]))
    if all_values:
        overall = np.concatenate(all_values)
        sample = deterministic_sample(overall, max_overall_values, seed=20260507)
        sample_frames.append(pd.DataFrame({"Panel": "Overall", "Group": "Overall", "Length_mm": sample}))
        summary_rows.insert(
            0,
            tract_length_summary_row(
                "Overall",
                overall,
                int(sum(subjects_by_group.values())),
            ),
        )
    if not sample_frames:
        return pd.DataFrame(), pd.DataFrame()
    return pd.concat(sample_frames, ignore_index=True), pd.DataFrame(summary_rows)


def render_lr_sr_section_title(
    title: str, *, compact: bool = False
) -> None:
    class_name = "lr-sr-section-title compact" if compact else "lr-sr-section-title"
    st.markdown(
        f"<div class='{class_name}'>{escape(title)}</div>",
        unsafe_allow_html=True,
    )


def tract_length_histogram_figure(values: pd.Series, summary: pd.Series, title: str, color: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=values,
            nbinsx=36,
            marker_color=color,
            opacity=0.85,
            hovertemplate="Length=%{x:.2f} mm<br>Edges=%{y}<extra></extra>",
            showlegend=False,
        )
    )
    q1 = float(summary.get("q1", np.nan))
    med = float(summary.get("median", np.nan))
    q3 = float(summary.get("q3", np.nan))
    for label, value, dash in (("Q1", q1, "dot"), ("Median", med, "solid"), ("Q3", q3, "dot")):
        if np.isfinite(value):
            fig.add_vline(
                x=value,
                line={"color": "#111827", "width": 2 if label == "Median" else 1.3, "dash": dash},
                annotation_text=f"{label} {format_number(value, digits=3)}",
                annotation_position="top",
                annotation_font={"size": 9, "color": CHART_TEXT},
    )
    n_edges = summary.get("n_edges", 0)
    n_subjects = summary.get("n_subjects", 0)
    n_edges_text = f"{int(n_edges):,}" if pd.notna(n_edges) else "NA"
    n_subjects_text = f"{int(n_subjects):,}" if pd.notna(n_subjects) else "NA"
    fig.add_annotation(
        x=0.98,
        y=0.96,
        xref="paper",
        yref="paper",
        text=f"n={n_edges_text} edges<br>{n_subjects_text} subjects",
        showarrow=False,
        align="right",
        bgcolor="rgba(255,255,255,0.86)",
        bordercolor="#cbd5e1",
        borderpad=3,
        font={"size": 9, "color": CHART_TEXT},
    )
    fig.update_layout(
        title={"text": title, "x": 0.02, "xanchor": "left", "font": {"size": 14, "color": CHART_TEXT}},
        template="plotly_white",
        plot_bgcolor=CHART_BACKGROUND,
        paper_bgcolor=CHART_BACKGROUND,
        height=270,
        margin={"l": 42, "r": 18, "t": 54, "b": 48},
        bargap=0.03,
        font={"color": CHART_TEXT},
        xaxis={
            "title": "Length (mm)",
            "gridcolor": LR_SR_CHART_GRID,
            "linecolor": LR_SR_CHART_STROKE,
            "zerolinecolor": LR_SR_CHART_STROKE,
            "tickfont": {"size": 10, "color": CHART_TEXT},
        },
        yaxis={
            "title": "Edges",
            "gridcolor": LR_SR_CHART_GRID,
            "linecolor": LR_SR_CHART_STROKE,
            "zerolinecolor": LR_SR_CHART_STROKE,
            "tickfont": {"size": 10, "color": CHART_TEXT},
        },
    )
    return fig


def tract_length_box_figure(samples: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    color_map = dict(zip(GROUP_ORDER, GROUP_COLORS))
    for group in ("CN", "AD", "MCI"):
        values = samples.loc[samples["Panel"] == group, "Length_mm"]
        if values.empty:
            continue
        fig.add_trace(
            go.Box(
                y=values,
                name=group,
                marker_color=color_map.get(group, "#64748b"),
                boxpoints=False,
                hovertemplate=f"{group}<br>Length=%{{y:.2f}} mm<extra></extra>",
            )
        )
    fig.update_layout(
        title={"text": "Percentile box plot", "x": 0.02, "xanchor": "left", "font": {"size": 14, "color": CHART_TEXT}},
        template="plotly_white",
        plot_bgcolor=CHART_BACKGROUND,
        paper_bgcolor=CHART_BACKGROUND,
        height=270,
        margin={"l": 46, "r": 16, "t": 54, "b": 46},
        showlegend=False,
        font={"color": CHART_TEXT},
        xaxis={
            "title": "",
            "tickfont": {"size": 10, "color": CHART_TEXT},
            "gridcolor": LR_SR_CHART_GRID,
            "linecolor": LR_SR_CHART_STROKE,
            "zerolinecolor": LR_SR_CHART_STROKE,
        },
        yaxis={
            "title": "Length (mm)",
            "gridcolor": LR_SR_CHART_GRID,
            "linecolor": LR_SR_CHART_STROKE,
            "zerolinecolor": LR_SR_CHART_STROKE,
            "tickfont": {"size": 10, "color": CHART_TEXT},
        },
    )
    return fig


def render_lr_sr_tract_length_distribution_panel() -> None:
    samples, summary = lr_sr_tract_length_distribution_data(str(ANALYSIS_ROOT), str(CONNECTOMES_DIR))
    if samples.empty or summary.empty:
        st.info("Tract-length distribution plots are not available yet because len_mean connectome matrices were not found.")
        return
    st.subheader("Tract Length Distribution")
    st.caption(
        "Live positive upper-triangle AAL3 edge lengths from `SC_AAL166_*_len_mean.csv`; plots use deterministic samples for display, while percentile labels use the full loaded edge distribution."
    )
    color_map = {"Overall": "#64748b", "CN": GROUP_COLORS[0], "AD": GROUP_COLORS[2], "MCI": GROUP_COLORS[1]}
    cols = st.columns(5, gap="small")
    for col, panel in zip(cols[:4], ("Overall", "CN", "AD", "MCI")):
        panel_values = samples.loc[samples["Panel"] == panel, "Length_mm"]
        panel_summary = summary.loc[summary["Panel"] == panel]
        with col:
            if panel_values.empty or panel_summary.empty:
                st.info(f"{panel} length data unavailable.")
            else:
                st.plotly_chart(
                    tract_length_histogram_figure(panel_values, panel_summary.iloc[0], f"{panel} length", color_map[panel]),
                    width="stretch",
                    config={"displaylogo": False},
                )
    with cols[4]:
        st.plotly_chart(tract_length_box_figure(samples), width="stretch", config={"displaylogo": False})


def tract_length_compact_histogram_figure(samples: pd.DataFrame, summary: pd.DataFrame) -> go.Figure:
    panels = ("Overall", "CN", "AD", "MCI")
    colors = {
        "Overall": "#64748b",
        "CN": GROUP_COLORS[0],
        "AD": GROUP_COLORS[2],
        "MCI": GROUP_COLORS[1],
    }
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=panels,
        horizontal_spacing=0.09,
        vertical_spacing=0.16,
    )
    for index, panel in enumerate(panels):
        row = index // 2 + 1
        col = index % 2 + 1
        values = samples.loc[samples["Panel"].eq(panel), "Length_mm"]
        panel_summary = summary.loc[summary["Panel"].eq(panel)]
        if values.empty or panel_summary.empty:
            continue
        stats_row = panel_summary.iloc[0]
        fig.add_trace(
            go.Histogram(
                x=values,
                nbinsx=32,
                marker_color=colors[panel],
                opacity=0.86,
                showlegend=False,
                hovertemplate="Length=%{x:.2f} mm<br>Sampled edges=%{y}<extra></extra>",
            ),
            row=row,
            col=col,
        )
        fig.add_vline(
            x=float(stats_row["median"]),
            line={"color": "#111827", "width": 2},
            row=row,
            col=col,
        )
        fig.add_annotation(
            x=0.98,
            y=0.92,
            xref=f"x{index + 1 if index else ''} domain",
            yref=f"y{index + 1 if index else ''} domain",
            text=(
                f"median {float(stats_row['median']):.1f} mm"
                f"<br>n={int(stats_row['n_subjects']):,} subjects"
            ),
            showarrow=False,
            xanchor="right",
            align="right",
            bgcolor="rgba(255,255,255,0.84)",
            bordercolor=CHART_STROKE,
            borderpad=3,
            font={"size": 9, "color": CHART_TEXT},
        )
    fig.update_annotations(font={"color": CHART_TEXT, "size": 12})
    fig.update_xaxes(
        title_text="Length (mm)",
        gridcolor=LR_SR_CHART_GRID,
        linecolor=LR_SR_CHART_STROKE,
        zerolinecolor=LR_SR_CHART_STROKE,
        tickfont={"color": CHART_TEXT, "size": 10},
        title_font={"color": CHART_TEXT, "size": 11},
    )
    fig.update_yaxes(
        title_text="Sampled edges",
        gridcolor=LR_SR_CHART_GRID,
        linecolor=LR_SR_CHART_STROKE,
        zerolinecolor=LR_SR_CHART_STROKE,
        tickfont={"color": CHART_TEXT, "size": 10},
        title_font={"color": CHART_TEXT, "size": 11},
    )
    fig.update_layout(
        template="plotly_white",
        plot_bgcolor=CHART_BACKGROUND,
        paper_bgcolor=CHART_BACKGROUND,
        height=500,
        margin={"l": 54, "r": 18, "t": 48, "b": 48},
        bargap=0.03,
        showlegend=False,
        font={"color": CHART_TEXT},
    )
    return fig


def lr_sr_length_summary_display(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    display = summary.copy()
    order = {"Overall": 0, "CN": 1, "MCI": 2, "AD": 3}
    display["_order"] = display["Panel"].map(order).fillna(99)
    display = display.sort_values("_order").drop(columns="_order")
    display = display.rename(
        columns={
            "Panel": "Cohort",
            "n_subjects": "Cases",
            "n_edges": "Positive edges",
            "median": "Median length (mm)",
            "q1": "Q1 (mm)",
            "q3": "Q3 (mm)",
            "iqr": "IQR (mm)",
            "lower_fence": "Lower fence (mm)",
            "upper_fence": "Upper fence (mm)",
            "low_outlier_n": "Low outliers",
            "high_outlier_n": "High outliers",
            "outlier_n": "Outlier edges",
            "outlier_pct": "Outliers (%)",
        }
    )
    return display[
        [
            "Cohort",
            "Cases",
            "Positive edges",
            "Median length (mm)",
            "Q1 (mm)",
            "Q3 (mm)",
            "IQR (mm)",
            "Lower fence (mm)",
            "Upper fence (mm)",
            "Low outliers",
            "High outliers",
        ]
    ]


def lr_sr_summary_field_dictionary() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Table": "Tract length",
                "Field": "Cases",
                "Definition": "Number of subjects contributing at least one positive finite len_mean edge.",
                "Formula / unit": "Distinct subjects",
            },
            {
                "Table": "Tract length",
                "Field": "Positive edges",
                "Definition": "All positive finite upper-triangle AAL3 tract-length observations pooled over the displayed cohort.",
                "Formula / unit": "Edge observations",
            },
            {
                "Table": "Tract length",
                "Field": "Median, Q1, Q3",
                "Definition": "50th, 25th, and 75th percentiles of the pooled positive edge-length distribution.",
                "Formula / unit": "Millimetres",
            },
            {
                "Table": "Tract length",
                "Field": "IQR",
                "Definition": "Middle-half spread of the edge-length distribution.",
                "Formula / unit": "Q3 - Q1, millimetres",
            },
            {
                "Table": "Tract length",
                "Field": "Lower fence",
                "Definition": "Conventional Tukey lower outlier threshold; it is not the observed minimum and can be negative.",
                "Formula / unit": "Q1 - 1.5 x IQR",
            },
            {
                "Table": "Tract length",
                "Field": "Upper fence",
                "Definition": "Conventional Tukey upper outlier threshold; it is not the observed maximum and can exceed the tractography length cap.",
                "Formula / unit": "Q3 + 1.5 x IQR",
            },
            {
                "Table": "Tract length",
                "Field": "Low / high outliers",
                "Definition": "Edge observations below the lower fence or above the upper fence. Zero is valid when both fences lie outside the observed range.",
                "Formula / unit": "Counts",
            },
            {
                "Table": "EDR exceptions",
                "Field": "Range",
                "Definition": "Short-, medium-, or long-range EDR bin. These bins use the EDR analysis thresholds, not the separate LR/SR tertile thresholds.",
                "Formula / unit": "SR / MR / LR",
            },
            {
                "Table": "EDR exceptions",
                "Field": "Candidate edges",
                "Definition": "All subject-edge observations eligible for the stated range-specific exception test.",
                "Formula / unit": "Sum over subjects",
            },
            {
                "Table": "EDR exceptions",
                "Field": "Exception edges",
                "Definition": (
                    "Candidate edges whose selected positive edge measure is greater than their "
                    "subject-specific adaptive-bin mean plus 3 SD. The paper/default measure is "
                    "SIFT2 fd_sum; count variants are sensitivity views."
                ),
                "Formula / unit": "One-sided high-for-length count",
            },
            {
                "Table": "EDR exceptions",
                "Field": "Median non-exception measure",
                "Definition": (
                    "Across-subject median of each subject's median selected edge measure among "
                    "eligible edges that were not flagged. With the default basis this is SIFT2 fd_sum strength."
                ),
                "Formula / unit": "Median of subject medians; selected basis units",
            },
            {
                "Table": "EDR exceptions",
                "Field": "Median exception measure",
                "Definition": (
                    "Across-subject median of each subject's median selected edge measure among flagged edges. "
                    "With the default basis this is SIFT2 fd_sum strength."
                ),
                "Formula / unit": "Median of subject medians; selected basis units",
            },
            {
                "Table": "EDR exceptions",
                "Field": "Pooled exception (%)",
                "Definition": "Exception percentage after pooling all candidate and exception edges; subjects with more candidates receive more weight.",
                "Formula / unit": "100 x sum(exception edges) / sum(candidate edges)",
            },
            {
                "Table": "EDR exceptions",
                "Field": "Median exceptions / case",
                "Definition": "Median number of exception edges carried by one subject in that cohort and range.",
                "Formula / unit": "Median subject count",
            },
            {
                "Table": "EDR exceptions",
                "Field": "Median subject rate (%)",
                "Definition": "Median of each subject's own exception percentage; every subject receives equal weight.",
                "Formula / unit": "Median[100 x subject exceptions / subject candidates]",
            },
            {
                "Table": "EDR exceptions",
                "Field": "Exception burden (grouped header)",
                "Definition": (
                    "Visual grouping for pooled exception rate, median exception count per case, "
                    "and median subject exception rate. The middle quantity is a count, not a percentage."
                ),
                "Formula / unit": "Two rates (%) plus one count (edges/case)",
            },
        ]
    )


def render_lr_sr_summary_field_dictionary() -> None:
    with st.expander(
        "Definitions for every tract-summary and EDR-exception field",
        expanded=False,
    ):
        definitions = lr_sr_summary_field_dictionary()
        for table_name, fields in definitions.groupby(
            "Table", sort=False
        ):
            st.markdown(f"**{str(table_name).upper()}**")
            glossary_lines: list[str] = []
            for _, field in fields.iterrows():
                glossary_lines.extend(
                    [
                        (
                            f"- **{field['Field']}** — "
                            f"{field['Definition']}"
                        ),
                        (
                            "  - *Formula / unit:* "
                            f"{field['Formula / unit']}"
                        ),
                    ]
                )
            st.markdown("\n".join(glossary_lines))
        st.caption(
            "Why are all displayed length outlier counts zero? The current Tukey lower fences are below 0 mm "
            "and upper fences are above the observed tract-length range, so no loaded edge crosses them. "
            "The fences are statistical thresholds, not acquisition limits."
        )


EDR_EXCEPTION_MEASURE_INFO = {
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
            "Raw assigned streamline count. It is tractography- and parcel-size-sensitive "
            "and is not substituted for the paper's fd_sum definition."
        ),
    },
    "count_invnodevol": {
        "label": "Node-volume-normalized count — sensitivity view",
        "short_label": "Node-volume-normalized count",
        "status": "SENSITIVITY VIEW",
        "interpretation": (
            "Streamline count scaled by inverse endpoint parcel volume. It is a sensitivity "
            "view, not the paper's primary EDR definition."
        ),
    },
}


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
        n_bins = int(
            np.clip(n_length // min_bin_edges, 3, max_bins)
        )
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


@st.cache_data(ttl=300, show_spinner=False)
def edr_exception_subject_table(
    root_str: str,
    connectomes_dir_str: str,
    measure_type: str,
) -> tuple[pd.DataFrame, dict]:
    root = Path(root_str)
    section_root = root / "17_edr_exceptions"
    threshold_path = section_root / "edr_exception_thresholds.csv"
    thresholds: dict = {}
    if threshold_path.exists():
        threshold_df = pd.read_csv(threshold_path)
        if not threshold_df.empty:
            thresholds = threshold_df.iloc[0].to_dict()
    precomputed_subject: pd.DataFrame | None = None
    if measure_type == "fd_sum":
        subject_path = (
            section_root
            / "edr_exception_subject_level_fd_sum_len_mean.csv"
        )
        if not subject_path.exists():
            return pd.DataFrame(), thresholds
        precomputed_subject = pd.read_csv(subject_path)

    master_path = root / "00_master/master_cohort.csv"
    connectomes_dir = Path(connectomes_dir_str)
    if not master_path.exists() or not connectomes_dir.exists():
        return (
            precomputed_subject
            if precomputed_subject is not None
            else pd.DataFrame()
        ), thresholds
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
            precomputed_subject
            if precomputed_subject is not None
            else pd.DataFrame()
        ), thresholds
    master = pd.read_csv(master_path)
    if not {"subject_id", "group", "Image ID"}.issubset(
        master.columns
    ):
        return (
            precomputed_subject
            if precomputed_subject is not None
            else pd.DataFrame()
        ), thresholds

    rows: list[dict[str, object]] = []
    for _, master_row in master.iterrows():
        subject_id = str(master_row["subject_id"]).strip()
        length_path = aal3_connectome_matrix_path(
            connectomes_dir,
            subject_id,
            master_row.get("Image ID"),
            "len_mean",
        )
        measure_path = aal3_connectome_matrix_path(
            connectomes_dir,
            subject_id,
            master_row.get("Image ID"),
            measure_type,
        )
        if length_path is None or measure_path is None:
            continue
        try:
            length_matrix = np.loadtxt(length_path, delimiter=",")
            measure_matrix = np.loadtxt(measure_path, delimiter=",")
        except Exception:
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
        exception, _threshold, _bin_ids = (
            edr_adaptive_exception_arrays(lengths, measures)
        )
        row: dict[str, object] = {
            "subject_id": subject_id,
            "group": str(master_row["group"]),
            "n_edges_valid": int(valid.sum()),
            "n_exceptions": int(exception.sum()),
        }
        for code, label in (("sr", "SR"), ("mr", "MR"), ("lr", "LR")):
            if label == "SR":
                in_range = lengths <= float(short_max)
            elif label == "MR":
                in_range = (
                    (lengths > float(short_max))
                    & (lengths <= float(long_min))
                )
            else:
                in_range = lengths > float(long_min)
            candidates = valid & in_range
            exceptions = candidates & exception
            candidate_n = int(candidates.sum())
            exception_n = int(exceptions.sum())
            nonexceptions = candidates & ~exception
            row[f"{code}_edge_count"] = candidate_n
            row[f"{code}_exception_count"] = exception_n
            row[f"{code}_exception_pct"] = (
                exception_n / candidate_n
                if candidate_n
                else np.nan
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
    if precomputed_subject is None:
        return computed, thresholds
    median_columns = [
        "subject_id",
        *[
            f"{code}_{kind}_measure_median"
            for code in ("sr", "mr", "lr")
            for kind in ("nonexception", "exception")
        ],
    ]
    available_median_columns = [
        column for column in median_columns if column in computed.columns
    ]
    if available_median_columns == ["subject_id"]:
        return precomputed_subject, thresholds
    enriched = precomputed_subject.merge(
        computed[available_median_columns],
        on="subject_id",
        how="left",
        validate="one_to_one",
    )
    return enriched, thresholds


@st.cache_data(ttl=300, show_spinner=False)
def edr_exception_count_summary(
    root_str: str,
    connectomes_dir_str: str,
    measure_type: str,
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    subject, thresholds = edr_exception_subject_table(
        root_str,
        connectomes_dir_str,
        measure_type,
    )
    if subject.empty or "group" not in subject.columns:
        return pd.DataFrame(), thresholds, subject
    rows: list[dict[str, object]] = []
    cohorts = [("Overall", subject)] + [
        (group, subject[subject["group"].astype(str).eq(group)].copy())
        for group in GROUP_ORDER
    ]
    for cohort, cohort_df in cohorts:
        for code, label in (("sr", "SR"), ("mr", "MR"), ("lr", "LR")):
            candidate_col = f"{code}_edge_count"
            exception_col = f"{code}_exception_count"
            rate_col = f"{code}_exception_pct"
            if not {candidate_col, exception_col}.issubset(cohort_df.columns):
                continue
            candidates = pd.to_numeric(cohort_df[candidate_col], errors="coerce")
            exceptions = pd.to_numeric(cohort_df[exception_col], errors="coerce")
            rates = (
                pd.to_numeric(cohort_df[rate_col], errors="coerce")
                if rate_col in cohort_df.columns
                else exceptions / candidates.replace(0, np.nan)
            )
            nonexception_measure = pd.to_numeric(
                cohort_df.get(
                    f"{code}_nonexception_measure_median",
                    pd.Series(index=cohort_df.index, dtype=float),
                ),
                errors="coerce",
            )
            exception_measure = pd.to_numeric(
                cohort_df.get(
                    f"{code}_exception_measure_median",
                    pd.Series(index=cohort_df.index, dtype=float),
                ),
                errors="coerce",
            )
            total_candidates = int(candidates.fillna(0).sum())
            total_exceptions = int(exceptions.fillna(0).sum())
            rows.append(
                {
                    "Cohort": cohort,
                    "Range": label,
                    "Cases": int(len(cohort_df)),
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
                    "Median exceptions / case": float(exceptions.median()),
                    "Median subject rate (%)": float(100.0 * rates.median()),
                }
            )
    return pd.DataFrame(rows), thresholds, subject


def edr_exception_rate_figure(
    summary: pd.DataFrame,
    measure_label: str,
) -> go.Figure:
    plot = summary[
        summary["Cohort"].isin(GROUP_ORDER)
        & summary["Range"].isin(("SR", "MR", "LR"))
    ].copy()
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=("SHORT RANGE", "MEDIUM RANGE", "LONG RANGE"),
        shared_yaxes=True,
        horizontal_spacing=0.075,
    )
    color_map = dict(zip(GROUP_ORDER, GROUP_COLORS))
    for col, range_label in enumerate(("SR", "MR", "LR"), start=1):
        subset = (
            plot[plot["Range"].eq(range_label)]
            .set_index("Cohort")
            .reindex(GROUP_ORDER)
            .reset_index()
        )
        fig.add_trace(
            go.Bar(
                x=subset["Cohort"],
                y=subset["Pooled exception (%)"],
                marker_color=[
                    color_map.get(str(group), "#64748b")
                    for group in subset["Cohort"]
                ],
                text=[
                    (
                        f"{value:.2f}%"
                        if pd.notna(value)
                        else ""
                    )
                    for value in subset["Pooled exception (%)"]
                ],
                textposition="outside",
                showlegend=False,
                hovertemplate=(
                    "%{x}<br>Pooled exception rate=%{y:.3f}%"
                    "<extra></extra>"
                ),
            ),
            row=1,
            col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=subset["Cohort"],
                y=subset["Median subject rate (%)"],
                mode="markers",
                marker={
                    "symbol": "diamond",
                    "size": 10,
                    "color": "#111827",
                    "line": {"color": "#ffffff", "width": 1},
                },
                name="Median subject rate",
                showlegend=col == 1,
                hovertemplate=(
                    "%{x}<br>Median subject rate=%{y:.3f}%"
                    "<extra></extra>"
                ),
            ),
            row=1,
            col=col,
        )
    fig.update_annotations(font={"color": CHART_TEXT, "size": 11})
    fig.update_xaxes(
        gridcolor=LR_SR_CHART_GRID,
        linecolor=LR_SR_CHART_STROKE,
        tickfont={"color": CHART_TEXT},
    )
    fig.update_yaxes(
        title_text="Exception rate (%)",
        gridcolor=LR_SR_CHART_GRID,
        linecolor=LR_SR_CHART_STROKE,
        zerolinecolor=LR_SR_CHART_STROKE,
        rangemode="tozero",
        tickfont={"color": CHART_TEXT},
    )
    fig.update_layout(
        title={
            "text": f"Exception-rate profile · {measure_label}",
            "x": 0.01,
            "font": {"color": CHART_TEXT, "size": 15},
        },
        template="plotly_white",
        plot_bgcolor=CHART_BACKGROUND,
        paper_bgcolor=CHART_BACKGROUND,
        font={"color": CHART_TEXT},
        height=360,
        margin={"l": 54, "r": 18, "t": 72, "b": 48},
        legend={
            "orientation": "h",
            "y": 1.13,
            "x": 0.62,
        },
    )
    return fig


def holm_adjusted_p_values(values: list[float]) -> list[float]:
    adjusted = np.full(len(values), np.nan, dtype=float)
    finite = [
        index
        for index, value in enumerate(values)
        if np.isfinite(value)
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


def paired_rank_biserial(
    first: np.ndarray, second: np.ndarray
) -> float:
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


def edr_lr_measure_inference(
    subject_table: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    required = {
        "subject_id",
        "group",
        "lr_nonexception_measure_median",
        "lr_exception_measure_median",
    }
    if subject_table.empty or not required.issubset(
        subject_table.columns
    ):
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
        analysis[column] = pd.to_numeric(
            analysis[column], errors="coerce"
        )
        analysis.loc[analysis[column] <= 0, column] = np.nan
    analysis = analysis[
        analysis["group"].isin(GROUP_ORDER)
    ].copy()
    analysis["lr_exception_strength_change"] = (
        analysis["lr_exception_measure_median"]
        - analysis["lr_nonexception_measure_median"]
    )

    plot_rows: list[dict[str, object]] = []
    for _, row in analysis.iterrows():
        for label, column in (
            (
                "Non-exception",
                "lr_nonexception_measure_median",
            ),
            ("Exception", "lr_exception_measure_median"),
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
        strength_change = row["lr_exception_strength_change"]
        if pd.notna(strength_change):
            plot_rows.append(
                {
                    "Subject": str(row["subject_id"]),
                    "Group": str(row["group"]),
                    "Edge class": "Exception strength change",
                    "Subject median": float(strength_change),
                }
            )
    plot_data = pd.DataFrame(plot_rows)

    results: list[dict[str, object]] = []
    within_indices: list[int] = []
    within_p: list[float] = []
    for group in GROUP_ORDER:
        paired = analysis[
            analysis["group"].eq(group)
        ].dropna(
            subset=[
                "lr_nonexception_measure_median",
                "lr_exception_measure_median",
            ]
        )
        ordinary = paired[
            "lr_nonexception_measure_median"
        ].to_numpy(dtype=float)
        exceptional = paired[
            "lr_exception_measure_median"
        ].to_numpy(dtype=float)
        p_value = math.nan
        statistic = math.nan
        if len(paired) >= 2 and not np.allclose(
            exceptional, ordinary
        ):
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
                "Contrast": (
                    f"{group}: exception vs non-exception"
                ),
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
                    float(
                        np.median(
                            exceptional - ordinary
                        )
                    )
                    if exceptional.size
                    else math.nan
                ),
                "Test": "Paired Wilcoxon",
                "Statistic": statistic,
                "Effect": paired_rank_biserial(
                    exceptional, ordinary
                ),
                "Effect type": "Paired rank-biserial",
                "Observed direction": (
                    "Exception higher"
                    if np.nanmedian(exceptional)
                    > np.nanmedian(ordinary)
                    else "Non-exception higher"
                ),
                "p": p_value,
                "Adjusted p": math.nan,
                "Family omnibus p": math.nan,
                "Family omnibus adjusted p": math.nan,
                "Correction family": (
                    "Holm across 3 within-group tests"
                ),
            }
        )
    for index, adjusted in zip(
        within_indices, holm_adjusted_p_values(within_p)
    ):
        results[index]["Adjusted p"] = adjusted

    metric_families = (
        (
            "Exception strength",
            "lr_exception_measure_median",
        ),
        (
            "Non-exception strength",
            "lr_nonexception_measure_median",
        ),
        (
            "Exception − non-exception strength",
            "lr_exception_strength_change",
        ),
    )
    omnibus_rows: list[int] = []
    omnibus_p_values: list[float] = []
    pairwise_rows_by_metric: dict[
        str, list[int]
    ] = {}
    for metric_label, metric_column in metric_families:
        values_by_group = {
            group: analysis.loc[
                analysis["group"].eq(group),
                metric_column,
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
                *[
                    values_by_group[group]
                    for group in available_groups
                ]
            )
            kw_statistic = float(test.statistic)
            kw_p = float(test.pvalue)
            total_n = int(
                sum(
                    values_by_group[group].size
                    for group in available_groups
                )
            )
            if total_n > len(available_groups):
                epsilon_squared = max(
                    0.0,
                    (
                        kw_statistic
                        - len(available_groups)
                        + 1
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

        pairwise_indices: list[int] = []
        pairwise_p_values: list[float] = []
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
                float(np.median(first))
                if first.size
                else math.nan
            )
            second_median = (
                float(np.median(second))
                if second.size
                else math.nan
            )
            if np.isfinite(first_median) and np.isfinite(
                second_median
            ):
                if first_median > second_median:
                    direction = f"{first_group} higher"
                elif second_median > first_median:
                    direction = f"{second_group} higher"
                else:
                    direction = "Equal medians"
            else:
                direction = "Not estimable"
            index = len(results)
            pairwise_indices.append(index)
            pairwise_p_values.append(p_value)
            results.append(
                {
                    "Question": "Across groups",
                    "Metric": metric_label,
                    "Contrast": (
                        f"{first_group} vs {second_group}"
                    ),
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
                        "Cliff delta (first group minus "
                        "second group)"
                    ),
                    "Observed direction": direction,
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
            pairwise_indices,
            holm_adjusted_p_values(pairwise_p_values),
        ):
            results[index]["Adjusted p"] = adjusted
        pairwise_rows_by_metric[metric_label] = pairwise_indices

    adjusted_omnibus = holm_adjusted_p_values(
        omnibus_p_values
    )
    for (
        metric_spec,
        omnibus_index,
        adjusted,
    ) in zip(metric_families, omnibus_rows, adjusted_omnibus):
        metric_label = metric_spec[0]
        results[omnibus_index]["Adjusted p"] = adjusted
        results[omnibus_index][
            "Family omnibus adjusted p"
        ] = adjusted
        for pairwise_index in pairwise_rows_by_metric[
            metric_label
        ]:
            results[pairwise_index][
                "Family omnibus adjusted p"
            ] = adjusted

    for row in results:
        row["Requested 4-test Holm p"] = math.nan
    requested_keys = (
        ("Exception strength", "MCI vs AD"),
        ("Non-exception strength", "MCI vs AD"),
        (
            "Exception − non-exception strength",
            "MCI vs AD",
        ),
        (
            "Exception − non-exception strength",
            "CN vs AD",
        ),
    )
    requested_index_by_key = {
        (str(row["Metric"]), str(row["Contrast"])): index
        for index, row in enumerate(results)
        if row["Question"] == "Across groups"
        and row["Test"] == "Mann-Whitney U"
    }
    primary_requested_indices = [
        requested_index_by_key[key]
        for key in requested_keys
        if key in requested_index_by_key
    ]
    primary_requested_adjusted = holm_adjusted_p_values(
        [
            float(results[index]["p"])
            for index in primary_requested_indices
        ]
    )
    for index, adjusted in zip(
        primary_requested_indices,
        primary_requested_adjusted,
    ):
        results[index]["Requested 4-test Holm p"] = adjusted

    result_table = pd.DataFrame(results)
    if not result_table.empty:
        adjusted = pd.to_numeric(
            result_table["Adjusted p"], errors="coerce"
        )
        omnibus_adjusted = pd.to_numeric(
            result_table["Family omnibus adjusted p"],
            errors="coerce",
        )
        supported = adjusted < 0.05
        across_pairwise = (
            result_table["Question"].eq("Across groups")
            & result_table["Test"].eq("Mann-Whitney U")
        )
        supported.loc[across_pairwise] = (
            supported.loc[across_pairwise]
            & (
                omnibus_adjusted.loc[across_pairwise]
                < 0.05
            )
        )
        result_table[
            "Statistically supported after correction"
        ] = supported.map({True: "YES", False: "NO"})
        result_table["Requested contrast supported"] = (
            pd.to_numeric(
                result_table["Requested 4-test Holm p"],
                errors="coerce",
            )
            < 0.05
        ).map({True: "YES", False: "NO"})
    exception_omnibus_p = math.nan
    change_omnibus_p = math.nan
    if omnibus_rows:
        exception_omnibus_p = float(
            results[omnibus_rows[0]]["p"]
        )
        change_omnibus_p = float(
            results[omnibus_rows[2]]["p"]
        )
    requested_adjusted_by_key = {
        key: float(
            results[requested_index_by_key[key]][
                "Requested 4-test Holm p"
            ]
        )
        for key in requested_keys
        if key in requested_index_by_key
    }
    return (
        plot_data,
        result_table,
        {
            "exception_omnibus_p": exception_omnibus_p,
            "exception_omnibus_adjusted_p": float(
                adjusted_omnibus[0]
            ),
            "change_omnibus_p": change_omnibus_p,
            "change_omnibus_adjusted_p": float(
                adjusted_omnibus[2]
            ),
            "change_mci_ad_adjusted_p": (
                requested_adjusted_by_key.get(
                    (
                        "Exception − non-exception strength",
                        "MCI vs AD",
                    ),
                    math.nan,
                )
            ),
            "change_cn_ad_adjusted_p": (
                requested_adjusted_by_key.get(
                    (
                        "Exception − non-exception strength",
                        "CN vs AD",
                    ),
                    math.nan,
                )
            ),
        },
    )


def edr_lr_measure_inference_figure(
    plot_data: pd.DataFrame,
    measure_label: str,
    metadata: Mapping[str, float],
) -> go.Figure:
    figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "SUBJECT MEDIANS: EXCEPTION VS NON-EXCEPTION",
            (
                "RAW EXCEPTION − NON-EXCEPTION CHANGE"
                " · MCI–AD Holm p="
                f"{format_p(metadata.get('change_mci_ad_adjusted_p', math.nan))}"
                " · CN–AD Holm p="
                f"{format_p(metadata.get('change_cn_ad_adjusted_p', math.nan))}"
            ),
        ),
        horizontal_spacing=0.12,
    )
    class_colors = {
        "Non-exception": "#94a3b8",
        "Exception": "#dc2626",
    }
    for edge_class in ("Non-exception", "Exception"):
        subset = plot_data[
            plot_data["Edge class"].eq(edge_class)
        ]
        figure.add_trace(
            go.Box(
                x=subset["Group"],
                y=subset["Subject median"],
                name=edge_class,
                legendgroup=edge_class,
                marker_color=class_colors[edge_class],
                line_color=class_colors[edge_class],
                boxpoints="outliers",
                offsetgroup=edge_class,
                hovertemplate=(
                    "%{x}<br>Subject median=%{y:.4g}"
                    f"<extra>{edge_class}</extra>"
                ),
            ),
            row=1,
            col=1,
        )
    for group, color in zip(GROUP_ORDER, GROUP_COLORS):
        subset = plot_data[
            plot_data["Group"].eq(group)
            & plot_data["Edge class"].eq(
                "Exception strength change"
            )
        ]
        figure.add_trace(
            go.Violin(
                x=[group] * len(subset),
                y=subset["Subject median"],
                name=group,
                legendgroup=f"exception-{group}",
                showlegend=False,
                line_color=color,
                fillcolor=color,
                opacity=0.70,
                box_visible=True,
                meanline_visible=False,
                points="outliers",
                hovertemplate=(
                    f"{group}<br>Exception − non-exception="
                    "%{y:.4g}"
                    "<extra></extra>"
                ),
            ),
            row=1,
            col=2,
        )
    figure.update_annotations(
        font={"color": CHART_TEXT, "size": 13}
    )
    figure.update_xaxes(
        categoryorder="array",
        categoryarray=list(GROUP_ORDER),
        gridcolor=LR_SR_CHART_GRID,
        linecolor=LR_SR_CHART_STROKE,
    )
    figure.update_yaxes(
        type="log",
        title_text=f"Subject-level median {measure_label}",
        gridcolor=LR_SR_CHART_GRID,
        linecolor=LR_SR_CHART_STROKE,
        zeroline=False,
        row=1,
        col=1,
    )
    figure.update_yaxes(
        title_text=(
            f"Exception − non-exception {measure_label}"
        ),
        gridcolor=LR_SR_CHART_GRID,
        linecolor=LR_SR_CHART_STROKE,
        zeroline=True,
        row=1,
        col=2,
    )
    figure.add_hline(
        y=0.0,
        line_dash="dash",
        line_color="#475569",
        annotation_text="No strength change",
        row=1,
        col=2,
    )
    figure.update_layout(
        template="plotly_white",
        plot_bgcolor=CHART_BACKGROUND,
        paper_bgcolor=CHART_BACKGROUND,
        font={"color": CHART_TEXT},
        boxmode="group",
        violinmode="group",
        height=470,
        margin={"l": 68, "r": 24, "t": 82, "b": 54},
        legend={
            "orientation": "h",
            "y": 1.13,
            "x": 0.02,
        },
    )
    return figure


def render_edr_lr_measure_inference(
    context: tuple[pd.DataFrame, dict, pd.DataFrame, str] | None,
) -> None:
    if context is None:
        return
    _summary, _thresholds, subject_table, measure_type = context
    plot_data, tests, metadata = edr_lr_measure_inference(
        subject_table
    )
    if plot_data.empty or tests.empty:
        st.info(
            "Subject-level LR exception and non-exception measures "
            "are not available for inference."
        )
        return
    measure_label = str(
        EDR_EXCEPTION_MEASURE_INFO[measure_type]["short_label"]
    )
    render_lr_sr_section_title(
        "2B · Long-range exception-strength comparisons",
        compact=True,
    )
    st.plotly_chart(
        edr_lr_measure_inference_figure(
            plot_data,
            measure_label,
            metadata,
        ),
        width="stretch",
        config={"displaylogo": False},
    )
    mci_ad = tests[
        tests["Question"].eq("Across groups")
        & (
            (
                tests["Contrast"].eq("MCI vs AD")
                & tests["Metric"].isin(
                    [
                        "Exception strength",
                        "Non-exception strength",
                        "Exception − non-exception strength",
                    ]
                )
            )
            | (
                tests["Contrast"].eq("CN vs AD")
                & tests["Metric"].eq(
                    "Exception − non-exception strength"
                )
            )
        )
    ].copy()
    mci_ad = mci_ad.rename(
        columns={
            "Median A": "Group A median",
            "Median B": "Group B median",
            "Median difference": "A − B median difference",
            "Adjusted p": "Pairwise Holm p",
            "Family omnibus adjusted p": "Omnibus Holm p",
        }
    )
    for column in (
        "Group A median",
        "Group B median",
        "A − B median difference",
        "Effect",
    ):
        mci_ad[column] = mci_ad[column].map(
            lambda value: format_number(value, digits=4)
        )
    for column in (
        "p",
        "Pairwise Holm p",
        "Omnibus Holm p",
        "Requested 4-test Holm p",
    ):
        mci_ad[column] = mci_ad[column].map(format_p)
    st.markdown(
        "**REQUESTED STRENGTH COMPARISONS — PRIMARY ANSWERS**"
    )
    st.dataframe(
        mci_ad[
            [
                "Metric",
                "First group",
                "Second group",
                "N",
                "Group A median",
                "Group B median",
                "Observed direction",
                "Effect",
                "Effect type",
                "Requested 4-test Holm p",
                "Requested contrast supported",
            ]
        ],
        width="stretch",
        hide_index=True,
        height=176,
        key=f"lr-edr-mci-ad-inference-{measure_type}",
    )
    display = tests.copy()
    for column in (
        "Median A",
        "Median B",
        "Median difference",
        "Effect",
    ):
        display[column] = display[column].map(
            lambda value: format_number(value, digits=4)
        )
    for column in (
        "p",
        "Adjusted p",
        "Family omnibus p",
        "Family omnibus adjusted p",
        "Requested 4-test Holm p",
    ):
        display[column] = display[column].map(format_p)
    display = display.drop(columns=["Statistic"])
    with st.expander(
        "Full CN / MCI / AD inference table",
        expanded=False,
    ):
        st.dataframe(
            display,
            width="stretch",
            hide_index=True,
            height=510,
            key=f"lr-edr-strength-inference-{measure_type}",
        )
    st.caption(
        "Unit of inference: one subject median per edge class. Within-group tests are paired Wilcoxon "
        "tests with Holm correction across CN/MCI/AD. Across-group inference separately tests exception "
        "strength, non-exception strength, and each subject's raw exception-minus-non-exception strength "
        "change. Omnibus tests "
        "are Holm-corrected across those three metric families; pairwise tests are Holm-corrected across "
        "CN–MCI, CN–AD and MCI–AD within each family. The visible primary table answers the three requested "
        "strength questions directly and Holm-corrects the four requested p-values as one family: MCI–AD "
        "exception strength, MCI–AD non-exception strength, and the raw subject-level exception-minus-"
        "non-exception strength change for MCI–AD and CN–AD. It does not require "
        "a prior omnibus gate. The full exploratory table additionally reports the broader all-group tests. "
        "Negative Cliff delta for "
        "MCI minus AD means AD tends higher, and the table now states that observed direction explicitly. "
        "The raw change is calculated within each subject before group testing; it is not obtained by merely "
        "subtracting the displayed group medians. Comparing that change between groups is the requested "
        "difference-in-differences-style test. The paired exception contrast is expected by construction because "
        "exceptions are selected for high strength within their length bin. These remain exploratory, "
        "unadjusted tests and do not control for age, sex, acquisition or site."
    )


@st.cache_data(ttl=300, show_spinner=False)
def edr_exception_example_data(
    root_str: str,
    connectomes_dir_str: str,
    subject_id: str,
    measure_type: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    root = Path(root_str)
    master_path = root / "00_master/master_cohort.csv"
    if not master_path.exists():
        return pd.DataFrame(), pd.DataFrame(), {}
    master = pd.read_csv(master_path)
    selected = master[
        master["subject_id"].astype(str).eq(str(subject_id))
    ]
    if selected.empty:
        return pd.DataFrame(), pd.DataFrame(), {}
    row = selected.iloc[0]
    connectomes_dir = Path(connectomes_dir_str)
    length_path = aal3_connectome_matrix_path(
        connectomes_dir,
        subject_id,
        row.get("Image ID"),
        "len_mean",
    )
    measure_path = aal3_connectome_matrix_path(
        connectomes_dir,
        subject_id,
        row.get("Image ID"),
        measure_type,
    )
    if length_path is None or measure_path is None:
        return pd.DataFrame(), pd.DataFrame(), {}
    lengths_matrix = np.loadtxt(length_path, delimiter=",")
    measures_matrix = np.loadtxt(measure_path, delimiter=",")
    if measures_matrix.shape != lengths_matrix.shape:
        return pd.DataFrame(), pd.DataFrame(), {}
    upper = np.triu_indices_from(lengths_matrix, k=1)
    lengths = lengths_matrix[upper]
    measures = measures_matrix[upper]
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
        sum((index + 1) * value for index, value in enumerate(subject_id.encode()))
        % (2**32 - 1)
    )
    rng = np.random.default_rng(seed)
    if ordinary_indices.size > 3500:
        ordinary_indices = np.sort(
            rng.choice(ordinary_indices, size=3500, replace=False)
        )
    if exception_indices.size > 1500:
        exception_indices = np.sort(
            rng.choice(exception_indices, size=1500, replace=False)
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


def render_edr_exception_controls(
) -> tuple[pd.DataFrame, dict, pd.DataFrame, str] | None:
    measure_type = st.selectbox(
        "Exception basis for the EDR table and plots",
        list(EDR_EXCEPTION_MEASURE_INFO),
        index=0,
        format_func=lambda key: EDR_EXCEPTION_MEASURE_INFO[key][
            "label"
        ],
        help=(
            "The paper/default uses SIFT2 fd_sum edge strength. Count-based "
            "alternatives are sensitivity views. Node strength is not offered "
            "because it is a node aggregate, whereas this is an edge-level test."
        ),
        key="lr-sr-workalong-exception-basis",
    )
    measure_info = EDR_EXCEPTION_MEASURE_INFO[measure_type]
    summary, thresholds, subject_table = edr_exception_count_summary(
        str(ANALYSIS_ROOT),
        str(CONNECTOMES_DIR),
        measure_type,
    )
    if summary.empty:
        st.info("EDR exception subject summaries are not available.")
        return None
    st.caption(
        f"{measure_info['status']}: {measure_info['interpretation']} "
        "Within each subject and adaptive length bin, an edge is flagged only when its selected positive "
        "edge measure is greater than bin mean + 3xSD. The flag is one-sided: it captures abnormally "
        "high-for-length edges, not weak edges and not every long edge. The example plot also shows the "
        "adaptive-bin median as descriptive context; the median is not used to flag exceptions."
    )
    return summary, thresholds, subject_table, measure_type


def edr_exception_summary_display(
    summary: pd.DataFrame,
    measure_type: str,
) -> pd.io.formats.style.Styler:
    measure_group = {
        "fd_sum": "MEDIAN SIFT2 STRENGTH",
        "count": "MEDIAN STREAMLINE COUNT",
        "count_invnodevol": "MEDIAN NORMALIZED COUNT",
    }.get(measure_type, "MEDIAN SELECTED MEASURE")
    display = summary[
        [
            "Cohort",
            "Range",
            "Cases",
            "Candidate edges",
            "Exception edges",
            "Median non-exception measure",
            "Median exception measure",
            "Pooled exception (%)",
            "Median exceptions / case",
            "Median subject rate (%)",
        ]
    ].copy()
    display.columns = pd.MultiIndex.from_tuples(
        [
            ("IDENTITY", "Cohort"),
            ("IDENTITY", "Range"),
            ("IDENTITY", "Cases"),
            ("EDGE COUNTS", "Candidates"),
            ("EDGE COUNTS", "Exceptions"),
            (measure_group, "Non-exception"),
            (measure_group, "Exception"),
            ("EXCEPTION BURDEN", "Pooled rate (%)"),
            ("EXCEPTION BURDEN", "Median count/case"),
            ("EXCEPTION BURDEN", "Subject rate (%)"),
        ]
    )
    formats = {
        ("IDENTITY", "Cases"): "{:,.0f}",
        ("EDGE COUNTS", "Candidates"): "{:,.0f}",
        ("EDGE COUNTS", "Exceptions"): "{:,.0f}",
        (measure_group, "Non-exception"): "{:,.4g}",
        (measure_group, "Exception"): "{:,.4g}",
        ("EXCEPTION BURDEN", "Pooled rate (%)"): "{:.3f}",
        ("EXCEPTION BURDEN", "Median count/case"): "{:,.1f}",
        ("EXCEPTION BURDEN", "Subject rate (%)"): "{:.3f}",
    }
    burden_columns = [
        ("EXCEPTION BURDEN", "Pooled rate (%)"),
        ("EXCEPTION BURDEN", "Median count/case"),
        ("EXCEPTION BURDEN", "Subject rate (%)"),
    ]
    return (
        display.style.format(formats, na_rep="NA")
        .set_properties(
            subset=burden_columns,
            **{
                "background-color": "#3a3208",
                "color": "#fde68a",
            },
        )
    )


def render_edr_exception_count_table(
    context: tuple[pd.DataFrame, dict, pd.DataFrame, str] | None,
    *,
    table_height: int = 380,
) -> None:
    if context is None:
        st.info("EDR exception subject summaries are not available.")
        return
    summary, thresholds, _subject_table, measure_type = context
    short_max = pd.to_numeric(
        pd.Series([thresholds.get("short_max_mm")]),
        errors="coerce",
    ).iloc[0]
    long_min = pd.to_numeric(
        pd.Series([thresholds.get("long_min_mm")]),
        errors="coerce",
    ).iloc[0]
    short_text = f"{float(short_max):.2f} mm" if pd.notna(short_max) else "Q1"
    long_text = f"{float(long_min):.2f} mm" if pd.notna(long_min) else "Q3"
    st.dataframe(
        edr_exception_summary_display(summary, measure_type),
        width="stretch",
        height=table_height,
        hide_index=True,
        key=f"lr-sr-workalong-edr-exception-summary-{measure_type}",
    )
    st.caption(
        "The grouped yellow cells are the EXCEPTION BURDEN block. Pooled rate and subject rate are "
        "percentages; median count/case is an edge count. Strength/count medians are medians of "
        "subject-level edge medians, so each subject contributes equally."
    )
    st.caption(
        f"EDR range bins are CN-referenced quartiles: SR ≤ {short_text}, MR between the cut-points, "
        f"and LR > {long_text}. These are intentionally different from the existing LR/SR tertile cut-points. "
        "The +3xSD rule is not an empirical percentile filter; under an ideal Gaussian model it is approximately "
        "a one-sided 99.865th-percentile threshold."
    )


def render_edr_exception_visualisation_panel(
    context: tuple[pd.DataFrame, dict, pd.DataFrame, str] | None,
) -> None:
    if context is None:
        return
    summary, _thresholds, subject_table, measure_type = context
    measure_info = EDR_EXCEPTION_MEASURE_INFO[measure_type]
    render_lr_sr_section_title(
        "2A · Exception visualisation", compact=True
    )
    st.plotly_chart(
        edr_exception_rate_figure(
            summary,
            str(measure_info["short_label"]),
        ),
        width="stretch",
        config={"displaylogo": False},
    )
    st.caption(
        "Bars show the candidate-edge-weighted pooled exception percentage. "
        "Black diamonds show the median subject's own exception percentage, so the plot makes the "
        "difference between pooled and subject-weighted summaries visible."
    )
    render_edr_lr_measure_inference(context)
    if subject_table.empty or "subject_id" not in subject_table.columns:
        return
    subjects = subject_table.copy()
    if "n_exceptions" not in subjects.columns:
        exception_columns = [
            column
            for column in (
                "sr_exception_count",
                "mr_exception_count",
                "lr_exception_count",
            )
            if column in subjects.columns
        ]
        subjects["n_exceptions"] = (
            subjects[exception_columns]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0)
            .sum(axis=1)
            if exception_columns
            else 0
        )
    subjects["n_exceptions"] = pd.to_numeric(
        subjects["n_exceptions"], errors="coerce"
    ).fillna(0)
    subjects = subjects.sort_values(
        ["n_exceptions", "subject_id"],
        ascending=[False, True],
    )
    options = subjects["subject_id"].astype(str).drop_duplicates().tolist()
    if not options:
        return
    with st.expander(
        "See the adaptive-bin median and mean + 3 SD edge threshold for one subject",
        expanded=True,
    ):
        selected_subject = st.selectbox(
            "Example subject",
            options,
            index=0,
            help=(
                "The default is the subject with the most exceptions for the selected measure, "
                "chosen only to make the visual rule easy to see."
            ),
            key=f"lr-sr-edr-example-subject-{measure_type}",
        )
        display, curve, metadata = edr_exception_example_data(
            str(ANALYSIS_ROOT),
            str(CONNECTOMES_DIR),
            selected_subject,
            measure_type,
        )
        if display.empty or curve.empty:
            st.info("No aligned positive length/measure edges are available for this subject.")
            return
        figure = go.Figure()
        ordinary = display[display["Class"].eq("Candidate")]
        exceptions = display[display["Class"].eq("Exception")]
        figure.add_trace(
            go.Scattergl(
                x=ordinary["Length (mm)"],
                y=ordinary["Measure"],
                mode="markers",
                name="Candidate edge",
                marker={
                    "color": "#94a3b8",
                    "size": 4,
                    "opacity": 0.30,
                },
                hovertemplate=(
                    "Length=%{x:.2f} mm<br>Measure=%{y:.4g}"
                    "<extra>Candidate</extra>"
                ),
            )
        )
        figure.add_trace(
            go.Scattergl(
                x=exceptions["Length (mm)"],
                y=exceptions["Measure"],
                mode="markers",
                name="Exception edge",
                marker={
                    "color": "#dc2626",
                    "size": 7,
                    "symbol": "x",
                    "opacity": 0.90,
                },
                hovertemplate=(
                    "Length=%{x:.2f} mm<br>Measure=%{y:.4g}"
                    "<extra>Exception</extra>"
                ),
            )
        )
        threshold_x: list[float | None] = []
        threshold_y: list[float | None] = []
        for _, threshold_row in curve.iterrows():
            threshold_x.extend(
                [
                    float(threshold_row["Minimum length"]),
                    float(threshold_row["Maximum length"]),
                    None,
                ]
            )
            threshold_y.extend(
                [
                    float(threshold_row["Threshold"]),
                    float(threshold_row["Threshold"]),
                    None,
                ]
            )
        figure.add_trace(
            go.Scatter(
                x=threshold_x,
                y=threshold_y,
                mode="lines",
                name="Adaptive-bin mean + 3 SD threshold",
                line={
                    "color": "#111827",
                    "width": 2.2,
                    "dash": "dash",
                },
                hoverinfo="skip",
            )
        )
        median_x: list[float | None] = []
        median_y: list[float | None] = []
        for _, median_row in curve.iterrows():
            median_x.extend(
                [
                    float(median_row["Minimum length"]),
                    float(median_row["Maximum length"]),
                    None,
                ]
            )
            median_y.extend(
                [
                    float(median_row["Median"]),
                    float(median_row["Median"]),
                    None,
                ]
            )
        figure.add_trace(
            go.Scatter(
                x=median_x,
                y=median_y,
                mode="lines",
                name="Adaptive-bin median (descriptive)",
                line={
                    "color": "#2563eb",
                    "width": 2.0,
                    "dash": "dot",
                },
                hoverinfo="skip",
            )
        )
        figure.update_layout(
            title={
                "text": (
                    f"{selected_subject} · {measure_info['short_label']}"
                ),
                "x": 0.01,
                "font": {"color": CHART_TEXT, "size": 15},
            },
            template="plotly_white",
            plot_bgcolor=CHART_BACKGROUND,
            paper_bgcolor=CHART_BACKGROUND,
            font={"color": CHART_TEXT},
            height=460,
            margin={"l": 62, "r": 20, "t": 62, "b": 54},
            xaxis={
                "title": "Edge tract length (mm)",
                "gridcolor": LR_SR_CHART_GRID,
                "linecolor": LR_SR_CHART_STROKE,
            },
            yaxis={
                "title": str(measure_info["short_label"]),
                "type": "log",
                "gridcolor": LR_SR_CHART_GRID,
                "linecolor": LR_SR_CHART_STROKE,
            },
            legend={"orientation": "h", "y": 1.10, "x": 0.38},
        )
        st.plotly_chart(
            figure,
            width="stretch",
            config={"displaylogo": False},
        )
        st.caption(
            f"{metadata.get('group', '')} · "
            f"{int(metadata.get('exception_n', 0)):,} exceptions among "
            f"{int(metadata.get('candidate_n', 0)):,} eligible positive edges. "
            "Grey points are a deterministic display sample; all red exceptions are retained up to the "
            "documented display cap. The black dashed step is the subject-specific mean + 3 SD exception "
            "threshold; the blue dotted step is the bin median for descriptive context and does not define the flag."
        )


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
    if group in {"LR/SR and delay", "Delay", "Advanced structural", "Brain age"}:
        return "LR/SR, delay, and advanced geometry"
    return "Other"


@st.cache_data(ttl=300, show_spinner=False)
def network_mapping_model_gap_summary(root_str: str) -> pd.DataFrame:
    path = Path(root_str) / "19_network_analysis" / "network_mapping_used.csv"
    if not path.exists():
        return pd.DataFrame()
    mapping = pd.read_csv(path)
    if "functional_network" not in mapping.columns:
        return pd.DataFrame()
    summary = (
        mapping.groupby("functional_network", dropna=False)
        .size()
        .rename("Mapped AAL3 nodes")
        .reset_index()
        .rename(columns={"functional_network": "Functional network"})
        .sort_values(["Mapped AAL3 nodes", "Functional network"], ascending=[False, True])
    )
    summary["Current model binding"] = "Mapping only — no explicit network aggregate feature"
    return summary.reset_index(drop=True)


def model_feature_family_metric_inventory(
    feature_eda: pd.DataFrame,
    family_summary: pd.DataFrame,
) -> pd.DataFrame:
    specifications = {
        "Graph / network topology": {
            "High-level metrics captured": (
                "Density; total/mean strength; characteristic path length; global efficiency; "
                "nodal degree, strength and efficiency; topology-microstructure Spearman coupling"
            ),
            "Representation": (
                "Whole-connectome summaries plus 166-node metric blocks; individual AAL IDs are hidden in this overview"
            ),
        },
        "Connectome edge / matrix": {
            "High-level metrics captured": (
                "Selected high-variance edge weights; positive-edge count/density; edge sum, mean, median, "
                "SD, IQR, q10/q90 and min/max across count, fd_sum, length and diffusion-weight matrices"
            ),
            "Representation": "Selected edge pairs plus whole-matrix distribution summaries",
        },
        "Network-wise systems (DMN/limbic/etc.)": {
            "High-level metrics captured": (
                "Not a direct model block yet. The Network Analysis tab separately computes network-aggregated "
                "FA/MD/RD/AxD, graph metrics, coupling, and within/between-network weight and density"
            ),
            "Representation": "Mapped analysis outputs only; zero direct columns in the current model matrix",
        },
        "Structural diffusion (FA/MD/RD/AxD)": {
            "High-level metrics captured": (
                "Whole-connectome and regional FA, MD, RD and AxD; edge mean, median, IQR and edge-support counts"
            ),
            "Representation": "Global summaries plus 166-region metric blocks",
        },
        "EDR exceptions": {
            "High-level metrics captured": (
                "EDR lambda/slope/intercept/correlation; SR/MR/LR candidate count, density, exception count/rate, "
                "exception strength and weight; regional exception count/rate/strength"
            ),
            "Representation": "Subject-level EDR summaries plus 166-region exception blocks",
        },
        "LR/SR, delay, and advanced geometry": {
            "High-level metrics captured": (
                "SR/MR/LR length, weight, density, efficiency and delay; SR-LR difference/ratio; EDR residuals; "
                "long-range vulnerability; short-range preservation; compensation; brain age/BAG and delay burden"
            ),
            "Representation": "Subject-level geometric, delay, resilience and brain-age summaries",
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
                "Candidate features": int(family_counts.get(family, 0)),
                **specification,
                "Actual source blocks": (
                    ", ".join(source_blocks)
                    if source_blocks
                    else "None in current model matrix"
                ),
            }
        )
    return pd.DataFrame(rows)


def render_network_mapping_provenance_panel() -> None:
    mapping_path = (
        ANALYSIS_ROOT
        / "19_network_analysis/network_mapping_used.csv"
    )
    notes_path = (
        PROJECT_ROOT / "atlas/AAL/AAL3_network_mapping_notes.md"
    )
    mapping = _safe_read_csv(mapping_path)
    with st.expander(
        "How AAL3 nodes are assigned to anatomical systems and functional networks",
        expanded=True,
    ):
        st.markdown(
            "<div class='lr-sr-mapping-note'>"
            "<b>This is not an official native AAL3-to-Yeo assignment.</b> "
            "The anatomical-system column is a deterministic label-based grouping of AAL3 regions. "
            "The functional-network column is an analysis-defined, Yeo-7-inspired crosswalk with explicit "
            "region-level rationales. It is approximate: an AAL3 parcel can cross functional boundaries, "
            "the current table was not generated by voxelwise overlap with a native Yeo atlas, and structural "
            "tractography does not establish functional connectivity. Functional-network results should "
            "therefore be treated as secondary and mapping-sensitive."
            "</div>",
            unsafe_allow_html=True,
        )
        if mapping is None or mapping.empty:
            st.info("The 166-node network crosswalk is not available.")
            return
        functional = (
            mapping.groupby("functional_network", dropna=False)
            .size()
            .rename("Mapped AAL3 nodes")
            .reset_index()
            .rename(columns={"functional_network": "Functional network"})
            .sort_values("Functional network")
        )
        anatomical = (
            mapping.groupby("anatomical_system", dropna=False)
            .size()
            .rename("Mapped AAL3 nodes")
            .reset_index()
            .rename(columns={"anatomical_system": "Anatomical system"})
            .sort_values("Anatomical system")
        )
        left, right = st.columns(2, gap="large")
        with left:
            st.markdown("**Approximate functional crosswalk**")
            st.dataframe(
                functional,
                width="stretch",
                height=300,
                hide_index=True,
                key="lr-sr-functional-network-membership-counts",
            )
        with right:
            st.markdown("**Deterministic anatomical grouping**")
            st.dataframe(
                anatomical,
                width="stretch",
                height=300,
                hide_index=True,
                key="lr-sr-anatomical-system-membership-counts",
            )
        st.caption(
            f"Auditable sources: `{mapping_path.relative_to(PROJECT_ROOT)}` and "
            f"`{notes_path.relative_to(PROJECT_ROOT)}`. Join by matrix_idx (1–166), "
            "not by the display alias AAL_001…AAL_166."
        )
        if st.toggle(
            "Show the complete node-to-network crosswalk and rationale",
            value=False,
            key="lr-sr-show-network-crosswalk",
        ):
            available = [
                column
                for column in (
                    "matrix_idx",
                    "atlas_value",
                    "atlas_label",
                    "functional_network",
                    "anatomical_system",
                    "rationale",
                )
                if column in mapping.columns
            ]
            st.dataframe(
                mapping[available],
                width="stretch",
                height=480,
                hide_index=True,
                key="lr-sr-network-crosswalk-detail",
            )


@st.cache_data(ttl=300, show_spinner=False)
def model_feature_eda(root_str: str) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    root = Path(root_str) / "18_ml_diagnostics"
    catalog_path = root / "ml_feature_catalog.csv"
    matrix_path = root / "ml_subject_feature_matrix.csv"
    if not catalog_path.exists() or not matrix_path.exists():
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
    n_rows = int(len(matrix))
    non_null = matrix.notna().sum()
    nulls = matrix.isna().sum()
    zeros = matrix.eq(0).sum()
    stats_frame = pd.DataFrame(
        {
            "n": n_rows,
            "Non-null": non_null,
            "Nulls": nulls,
            "Nulls (%)": 100.0 * nulls / n_rows if n_rows else np.nan,
            "Zeros": zeros,
            "Zeros (%)": 100.0 * zeros / n_rows if n_rows else np.nan,
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
    catalog["Dashboard family"] = catalog["feature_group"].map(model_dashboard_family)
    catalog["Feature group"] = catalog["feature_group"].fillna("not recorded")
    catalog["Source section"] = catalog["source_section"].fillna("not recorded")
    catalog["Source column"] = catalog["source_column"].fillna("")
    catalog["Model status"] = catalog["final_status"].fillna("not recorded")
    catalog["Selected CV folds"] = pd.to_numeric(
        catalog["selected_fold_count"],
        errors="coerce",
    ).fillna(0).astype(int)
    catalog["Elastic-net nonzero folds"] = pd.to_numeric(
        catalog["elastic_net_nonzero_fold_count"],
        errors="coerce",
    ).fillna(0).astype(int)
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
    ].join(stats_frame, on="Feature")

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
        subset = feature_eda[feature_eda["Dashboard family"].eq(family)]
        status = subset["Model status"].astype(str)
        explicit_network_gap = family.startswith("Network-wise systems")
        family_rows.append(
            {
                "Requested feature family": family,
                "Candidate features": int(len(subset)),
                "Retained after filtering": int(
                    status.isin(("selected_in_cv", "retained_not_selected")).sum()
                ),
                "Selected in CV": int(status.eq("selected_in_cv").sum()),
                "Dropped": int(status.str.startswith("dropped").sum()),
                "Current model binding": (
                    "NOT EXPLICITLY PRESENT"
                    if explicit_network_gap
                    else "PRESENT"
                ),
                "Interpretation": (
                    "AAL3-to-network mapping exists, but subject-level DMN/limbic within/between-network aggregates "
                    "are not columns in the current model matrix."
                    if explicit_network_gap
                    else ""
                ),
            }
        )
    return feature_eda, pd.DataFrame(family_rows), n_rows


def render_lr_sr_feature_eda_panel() -> None:
    feature_eda, family_summary, cohort_n = model_feature_eda(str(ANALYSIS_ROOT))
    render_lr_sr_section_title("3 · Modelling features")
    if feature_eda.empty:
        st.info("The model feature catalog or subject feature matrix is not available.")
        return
    status = feature_eda["Model status"].fillna("not recorded").astype(str)
    selected_n = int(status.eq("selected_in_cv").sum())
    retained_n = int(status.isin(("selected_in_cv", "retained_not_selected")).sum())
    metric_cols = st.columns(4)
    metric_cols[0].metric("Candidate inputs", f"{len(feature_eda):,}")
    metric_cols[1].metric("Subjects", f"{cohort_n:,}")
    metric_cols[2].metric("Retained after filtering", f"{retained_n:,}")
    metric_cols[3].metric("Selected in CV", f"{selected_n:,}")
    st.caption(
        "The family summary separates graph/network topology, raw connectome edge/matrix features, structural "
        "diffusion measures, and EDR exceptions. It also makes the current network-system gap explicit: DMN, "
        "limbic, and other within/between-network aggregates have been mapped in the network-analysis tab but "
        "are not yet direct inputs to this model matrix."
    )
    st.dataframe(
        family_summary,
        width="stretch",
        height=250,
        hide_index=True,
        key="lr-sr-workalong-feature-family-summary",
    )
    render_lr_sr_section_title(
        "High-level metrics captured by each feature family",
        compact=True,
    )
    metric_inventory = model_feature_family_metric_inventory(
        feature_eda,
        family_summary,
    )
    st.dataframe(
        metric_inventory,
        width="stretch",
        height=400,
        hide_index=True,
        column_config={
            "Feature family": st.column_config.TextColumn(
                help="Scientifically coherent family of candidate model inputs."
            ),
            "Candidate features": st.column_config.NumberColumn(
                format="%d",
                help="Exact number of columns from this family in the current candidate feature catalog.",
            ),
            "High-level metrics captured": st.column_config.TextColumn(
                help="Metric types represented, summarized without listing individual AAL node IDs."
            ),
            "Representation": st.column_config.TextColumn(
                help="Whether the family is represented globally, regionally, edge-wise, or only as a mapped analysis output."
            ),
            "Actual source blocks": st.column_config.TextColumn(
                help="Exact feature-group blocks contributing columns to the current candidate model matrix."
            ),
        },
        key="lr-sr-workalong-feature-metric-inventory",
    )
    render_network_mapping_provenance_panel()
    st.caption(
        "Candidate inputs are columns presented to the preprocessing/selection workflow. "
        "`selected_in_cv` is a fold-selection record, not a causal or biological importance claim. "
        "Variance and SD below use sample ddof=1."
    )
    render_lr_sr_section_title("Feature-level EDA", compact=True)
    family_options = ["All families"] + [
        family
        for family in family_summary["Requested feature family"].astype(str)
        if not family.startswith("Network-wise systems")
    ]
    filter_family, filter_status, filter_search = st.columns((1, 1, 1.2), gap="small")
    with filter_family:
        selected_family = st.selectbox(
            "Feature family",
            family_options,
            index=0,
            key="lr-sr-workalong-feature-family",
        )
    status_options = sorted(status.unique())
    with filter_status:
        selected_status = st.multiselect(
            "Model status",
            status_options,
            default=status_options,
            key="lr-sr-workalong-feature-status",
        )
    with filter_search:
        search = st.text_input(
            "Find feature",
            value="",
            placeholder="e.g. DMN, limbic, FA, density, exception",
            key="lr-sr-workalong-feature-search",
        ).strip().lower()
    filtered = feature_eda[
        feature_eda["Model status"].astype(str).isin(selected_status)
    ].copy()
    if selected_family != "All families":
        filtered = filtered[filtered["Dashboard family"].eq(selected_family)]
    if search:
        filtered = filtered[
            filtered["Feature"].astype(str).str.lower().str.contains(search, regex=False)
            | filtered["Feature group"].astype(str).str.lower().str.contains(search, regex=False)
            | filtered["Source column"].astype(str).str.lower().str.contains(search, regex=False)
        ]
    essential = [
        "Feature",
        "Dashboard family",
        "Feature group",
        "Model status",
        "Selected CV folds",
        "n",
        "Variance",
        "SD",
        "Nulls",
        "Zeros",
    ]
    st.caption(f"Showing {len(filtered):,} feature rows.")
    st.dataframe(
        filtered[essential],
        width="stretch",
        height=420,
        hide_index=True,
        column_config={
            "Variance": st.column_config.NumberColumn(format="%.5g"),
            "SD": st.column_config.NumberColumn(format="%.5g"),
        },
        key="lr-sr-workalong-feature-eda-essential",
    )
    with st.expander("Show the complete EDA columns for the filtered features", expanded=False):
        st.dataframe(
            filtered,
            width="stretch",
            height=440,
            hide_index=True,
            column_config={
                "Variance": st.column_config.NumberColumn(format="%.5g"),
                "SD": st.column_config.NumberColumn(format="%.5g"),
                "Mean": st.column_config.NumberColumn(format="%.5g"),
                "Median": st.column_config.NumberColumn(format="%.5g"),
                "Min": st.column_config.NumberColumn(format="%.5g"),
                "Max": st.column_config.NumberColumn(format="%.5g"),
                "Nulls (%)": st.column_config.NumberColumn(format="%.2f"),
                "Zeros (%)": st.column_config.NumberColumn(format="%.2f"),
            },
            key="lr-sr-workalong-feature-eda-full",
        )


def render_lr_sr_compact_metric_panel(
    section: Section,
    csv_files: list[Path],
    subject_df: pd.DataFrame,
    hide_visual_outliers: bool,
) -> None:
    available = numeric_metric_columns(subject_df, section.label)
    if not available:
        st.info("No numeric LR/SR subject metrics are available.")
        return
    default_metric = "short_len_median" if "short_len_median" in available else available[0]
    render_lr_sr_section_title("4 · LR/SR metrics")
    controls_left, controls_right = st.columns((1.35, 1), gap="small")
    with controls_left:
        metric = st.selectbox(
            "Metric",
            available,
            index=available.index(default_metric),
            format_func=metric_label,
            key="lr-sr-workalong-selected-metric",
        )
    with controls_right:
        st.caption(
            "One metric is shown at a time to keep this work-along page compact. "
            "The existing LR / SR tab remains available for the full metric-family view."
        )
    data = subject_metric_frame(subject_df, metric, value_col=metric)
    if data.empty:
        st.info(f"No plottable values are available for {metric_label(metric)}.")
        return
    desc = metric_descriptives(csv_files, metric, data=data)
    pairwise = metric_pairwise(csv_files, metric)
    chart, _, _, removed_counts = live_distribution_chart(
        data,
        metric_label(metric),
        hide_visual_outliers=hide_visual_outliers,
        stats_df=desc,
        subtitle_lines=chart_stat_subtitles(csv_files, metric),
    )
    chart.update_layout(height=430)
    chart.update_xaxes(
        gridcolor=LR_SR_CHART_GRID,
        linecolor=LR_SR_CHART_STROKE,
        tickfont={"color": CHART_TEXT},
    )
    chart.update_yaxes(
        gridcolor=LR_SR_CHART_GRID,
        linecolor=LR_SR_CHART_STROKE,
        zerolinecolor=LR_SR_CHART_STROKE,
        tickfont={"color": CHART_TEXT},
    )
    left, right = st.columns((1.15, 1), gap="small")
    with left:
        st.plotly_chart(chart, width="stretch", config={"displaylogo": False})
        if sum(removed_counts.values()):
            st.caption(
                "Display-only 1.5-IQR trim is active; the statistical tables remain unchanged. "
                f"Hidden points: {group_count_text(removed_counts)}."
            )
    with right:
        render_metric_definition_inference_table(
            section,
            metric,
            desc,
            pairwise,
            key=f"lr-sr-workalong-stats-{metric}",
        )


def render_lr_sr_workalong_layout(
    section: Section,
    csv_files: list[Path],
    hide_visual_outliers: bool,
) -> bool:
    samples, length_summary = lr_sr_tract_length_distribution_data(
        str(ANALYSIS_ROOT),
        str(CONNECTOMES_DIR),
    )
    render_lr_sr_section_title("1 · Tract-length distributions")
    st.caption(
        "Computed live from positive upper-triangle AAL3 `len_mean` edges. Histograms use deterministic "
        "display samples; all medians, quartiles, fences, and outlier counts use the complete loaded edge set."
    )
    length_available = not samples.empty and not length_summary.empty
    if not length_available:
        st.info("Tract-length matrices are not available.")
    else:
        plot_left, plot_right = st.columns((1.6, 0.85), gap="small")
        with plot_left:
            st.plotly_chart(
                tract_length_compact_histogram_figure(samples, length_summary),
                width="stretch",
                config={"displaylogo": False},
            )
        with plot_right:
            box = tract_length_box_figure(samples)
            box.update_layout(
                height=500,
                title={
                    "text": "<b>CN / MCI / AD PERCENTILES</b>",
                    "x": 0.03,
                    "xanchor": "left",
                    "font": {"size": 20, "color": CHART_TEXT},
                },
            )
            st.plotly_chart(box, width="stretch", config={"displaylogo": False})

    edr_context = render_edr_exception_controls()
    summary_left, summary_right = st.columns((1, 1), gap="small")
    with summary_left:
        render_lr_sr_section_title(
            "Tract-length summary",
            compact=True,
        )
        if not length_available:
            st.info("No tract-length summary is available.")
        else:
            st.dataframe(
                lr_sr_length_summary_display(length_summary),
                width="stretch",
                height=400,
                hide_index=True,
                column_config={
                    "Cohort": st.column_config.TextColumn(
                        help="Complete cohort or diagnostic group."
                    ),
                    "Cases": st.column_config.NumberColumn(
                        format="%d",
                        help="Subjects contributing at least one positive finite len_mean edge.",
                    ),
                    "Positive edges": st.column_config.NumberColumn(
                        format="%d",
                        help="Positive finite upper-triangle AAL3 length observations.",
                    ),
                    "Median length (mm)": st.column_config.NumberColumn(
                        format="%.2f",
                        help="50th percentile of pooled positive edge lengths.",
                    ),
                    "Q1 (mm)": st.column_config.NumberColumn(
                        format="%.2f",
                        help="25th percentile of pooled positive edge lengths.",
                    ),
                    "Q3 (mm)": st.column_config.NumberColumn(
                        format="%.2f",
                        help="75th percentile of pooled positive edge lengths.",
                    ),
                    "IQR (mm)": st.column_config.NumberColumn(
                        format="%.2f",
                        help="Q3 minus Q1.",
                    ),
                    "Lower fence (mm)": st.column_config.NumberColumn(
                        format="%.2f",
                        help="Q1 minus 1.5 times IQR; a threshold, not the observed minimum.",
                    ),
                    "Upper fence (mm)": st.column_config.NumberColumn(
                        format="%.2f",
                        help="Q3 plus 1.5 times IQR; a threshold, not the observed maximum.",
                    ),
                    "Low outliers": st.column_config.NumberColumn(
                        format="%d",
                        help="Edge observations below the lower Tukey fence.",
                    ),
                    "High outliers": st.column_config.NumberColumn(
                        format="%d",
                        help="Edge observations above the upper Tukey fence.",
                    ),
                },
                key="lr-sr-workalong-length-summary",
            )
            st.caption(
                "Outliers are edge-level observations outside Q1 - 1.5xIQR or Q3 + 1.5xIQR. "
                "They are reported, not removed from source data or inferential analyses."
            )
    with summary_right:
        render_lr_sr_section_title(
            "2 · EDR exceptions",
            compact=True,
        )
        render_edr_exception_count_table(
            edr_context,
            table_height=400,
        )

    render_lr_sr_summary_field_dictionary()
    render_edr_exception_visualisation_panel(edr_context)
    st.divider()
    render_lr_sr_feature_eda_panel()
    subject_path = subject_table_for_section(section, csv_files)
    if subject_path is None:
        st.info("The LR/SR subject-level metric table is not available.")
        return True
    subject_df = _safe_read_csv(subject_path)
    if subject_df is None or subject_df.empty or "group" not in subject_df.columns:
        st.info("The LR/SR subject-level metric table is empty or lacks disease-group labels.")
        return True
    st.divider()
    render_lr_sr_compact_metric_panel(
        section,
        csv_files,
        subject_df,
        hide_visual_outliers,
    )
    with st.expander("LR/SR definitions, thresholds, and formulas", expanded=False):
        render_lr_sr_definition_panel()
    return True


def render_node_metric_definition(metric: str) -> None:
    info = NODE_METRIC_DEFINITIONS.get(metric)
    if not info:
        return
    st.markdown(
        "<div class='definition-card'>"
        f"<h4>{escape(info['title'])}</h4>"
        f"<p>{escape(info['definition'])}</p>"
        f"<div class='metric-formula'>{escape(info['formula'])}</div>"
        f"<p>{escape(info['note'])}</p>"
        "</div>",
        unsafe_allow_html=True,
    )


def render_coupling_explainer() -> None:
    st.caption(
        "Coupling is not a duplicate strength or degree table. For each subject, the app correlates a node-wise topology map "
        "T_i (strength, degree, or nodal efficiency) with a node-wise microstructure map M_i (FA/MD/AD/RD) over matched AAL3 regions."
    )
    st.markdown(
        "<div class='definition-card'>"
        "<h4>Structure-Microstructure Coupling</h4>"
        "<p>The coupling value is a within-subject Spearman rank correlation across AAL3 nodes. "
        "A positive value means nodes with higher structural topology also tend to have higher local microstructure values in that same subject.</p>"
        "<div class='metric-formula'>rho_s = corr_rank(T_i, M_i), i = 1..N matched AAL3 nodes</div>"
        "<p>The words strength, degree, and nodal efficiency identify the topology variable T_i used in the correlation, not a second copy of the raw node-metric distribution.</p>"
        "</div>",
        unsafe_allow_html=True,
    )
    cols = st.columns(4)
    with cols[0]:
        render_node_metric_definition("degree")
    with cols[1]:
        render_node_metric_definition("strength")
    with cols[2]:
        render_node_metric_definition("nodal_eff")
    with cols[3]:
        render_node_metric_definition("rho")


def render_lr_sr_definition_panel() -> None:
    thresholds_path = ANALYSIS_ROOT / "12_length_delay" / "lr_sr_thresholds.csv"
    subject_path = ANALYSIS_ROOT / "12_length_delay" / "lr_sr_subject_level_fd_sum_len_mean.csv"
    short_text = "theta_S"
    medium_text = "theta_M"
    rule = "not available"
    if thresholds_path.exists():
        thresholds = _safe_read_csv(thresholds_path)
        if thresholds is not None and not thresholds.empty:
            row = thresholds.iloc[0]
            short_val = pd.to_numeric(pd.Series([row.get("short_max_mm")]), errors="coerce").iloc[0]
            medium_val = pd.to_numeric(pd.Series([row.get("medium_max_mm")]), errors="coerce").iloc[0]
            if pd.notna(short_val):
                short_text = f"{float(short_val):.2f} mm"
            if pd.notna(medium_val):
                medium_text = f"{float(medium_val):.2f} mm"
            rule = str(row.get("rule", rule))
    else:
        st.warning("LR/SR threshold CSV is missing; formulas are shown generically until the analysis refresh regenerates it.")

    formula_rows = [
        (
            "Short-range set",
            "AAL3 ROI-pair edges with positive tract length at or below the short-range threshold.",
            "E_S = {(i,j): i < j, 0 < L_ij <= theta_S}",
        ),
        (
            "Medium-range set",
            "AAL3 ROI-pair edges longer than the short threshold and at or below the medium threshold.",
            "E_M = {(i,j): i < j, theta_S < L_ij <= theta_M}",
        ),
        (
            "Long-range set",
            "AAL3 ROI-pair edges longer than the medium threshold.",
            "E_L = {(i,j): i < j, L_ij > theta_M}",
        ),
        (
            "short_w_mean",
            "Mean fd_sum weight across short-range candidate edges.",
            "mean_{(i,j) in E_S}(W_ij)",
        ),
        (
            "long_w_mean",
            "Mean fd_sum weight across long-range candidate edges.",
            "mean_{(i,j) in E_L}(W_ij)",
        ),
        (
            "sr_lr_w_diff",
            "Absolute short-minus-long average weight contrast.",
            "short_w_mean - long_w_mean",
        ),
        (
            "sr_lr_w_ratio",
            "Relative short-versus-long average weight ratio.",
            "short_w_mean / long_w_mean",
        ),
        (
            "global_eff_range",
            "Range-specific global efficiency using inverse structural weight as shortest-path cost.",
            "mean(1 / d_ij), c_ij = 1 / W_ij",
        ),
    ]
    formula_table_html = definition_table_html(formula_rows, ("Metric", "Definition", "Formula"))
    threshold_rows = [
        (
            "SR",
            f"0 < L_ij <= {short_text}",
            lr_sr_edge_count_summary(subject_path, "short"),
        ),
        (
            "MR",
            f"{short_text} < L_ij <= {medium_text}",
            lr_sr_edge_count_summary(subject_path, "medium"),
        ),
        (
            "LR",
            f"L_ij > {medium_text}",
            lr_sr_edge_count_summary(subject_path, "long"),
        ),
    ]
    threshold_table_html = definition_table_html(
        threshold_rows,
        ("Range", "Length rule", "AAL edge candidates per subject"),
    )
    symbol_rows = [
        ("L_ij", "AAL3 ROI-pair tract-length summary from len_mean.", "millimetres"),
        ("W_ij", "SIFT2 fibre-density structural weight from fd_sum.", "edge weight"),
        ("theta_S", "Short-range threshold read from short_max_mm.", short_text),
        ("theta_M", "Medium/long threshold read from medium_max_mm.", medium_text),
        ("E_S, E_M, E_L", "Short-, medium-, and long-range AAL3 edge sets.", "defined by L_ij"),
        ("c_ij", "Inverse structural path cost used for shortest paths.", "1 / W_ij"),
        ("d_ij", "Shortest-path distance after converting weights to inverse-weight costs.", "path distance"),
        ("v", "Assumed conduction velocity used by the delay proxy.", "delay_ij = L_ij / v"),
    ]
    symbol_table_html = definition_table_html(
        symbol_rows,
        ("Symbol", "Meaning", "Units / formula"),
    )

    st.markdown(
        "<div class='result-card graph-guide-card'>"
        "<h4>LR / SR Definition</h4>"
        "<div class='metric-explain'>Short-, medium-, and long-range edges are defined from the live "
        "<code>12_length_delay/lr_sr_thresholds.csv</code> thresholds. The current dashboard keeps the existing "
        "tertile-based LR/SR analysis unchanged.</div>"
        "<h5>Current thresholds and candidate edges</h5>"
        f"{threshold_table_html}"
        "<div class='metric-explain'>Counts are AAL3 connectome ROI-pair edge candidates summarized across subjects, "
        "not raw streamline tract counts. They vary by subject because only finite length-defined candidate edges are included.</div>"
        "<h5>Live data source</h5>"
        f"<div class='metric-explain'>Subject metrics: <code>{escape(subject_path.relative_to(ANALYSIS_ROOT).as_posix())}</code>. "
        "Edge length matrix: <code>len_mean</code>. Edge weight matrix: <code>fd_sum</code>. "
        f"Threshold CSV fields: <code>short_max_mm</code> and <code>medium_max_mm</code>. "
        f"Threshold rule: {escape(rule)}.</div>"
        f"{definition_grid_html('Metric definitions and formulas', formula_table_html, 'Symbols', symbol_table_html)}"
        "<div class='metric-explain'><b>Caveat:</b> LR/SR here is a tract-length proxy from AAL3 connectome edges, "
        "not a direct superficial white matter or U-fiber extraction.</div>"
        "</div>",
        unsafe_allow_html=True,
    )


def render_edr_exception_definition() -> None:
    thresholds_path = ANALYSIS_ROOT / "17_edr_exceptions" / "edr_exception_thresholds.csv"
    subject_path = ANALYSIS_ROOT / "17_edr_exceptions" / "edr_exception_subject_level_fd_sum_len_mean.csv"
    rule = "within-subject adaptive length bin; weight > bin mean + 3*bin sd"
    short_text = "Q1"
    long_text = "Q3"
    if thresholds_path.exists():
        thresholds = _safe_read_csv(thresholds_path)
        if thresholds is not None and not thresholds.empty:
            row = thresholds.iloc[0]
            short_val = pd.to_numeric(pd.Series([row.get("short_max_mm")]), errors="coerce").iloc[0]
            long_val = pd.to_numeric(pd.Series([row.get("long_min_mm")]), errors="coerce").iloc[0]
            if pd.notna(short_val):
                short_text = f"{float(short_val):.2f} mm"
            if pd.notna(long_val):
                long_text = f"{float(long_val):.2f} mm"
            rule = str(row.get("exception_rule", rule))
    else:
        st.warning("EDR exception outputs are not generated yet. Run the full dashboard refresh to populate this section.")
    count_summary = edr_range_count_summary_html(subject_path)
    st.markdown(
        "<div class='result-card graph-guide-card'>"
        "<h4>EDR Exception Definition</h4>"
        "<div class='metric-explain'>This section implements the structural part of the edge-distance relationship exception analysis. "
        "Each subject is fitted independently, then edges that are unusually high-weight for that subject's own length distribution are flagged.</div>"
        "<div class='formula-list'>"
        "<div class='formula-line'>log(W_ij) = alpha_s - lambda_s L_ij + epsilon_ij</div>"
        "<div class='formula-line'>exception_ij = 1 if W_ij &gt; mean_bin(W) + 3 * sd_bin(W)</div>"
        "<div class='formula-line'>SR: L_ij &lt;= Q1; MR: Q1 &lt; L_ij &lt;= Q3; LR: L_ij &gt; Q3</div>"
        "</div>"
        f"<div class='chart-finding'><b>Current Q1/Q3 thresholds:</b> SR <= {escape(short_text)}; LR > {escape(long_text)}. "
        f"Exception rule: {escape(rule)}.</div>"
        f"{count_summary}"
        "<div class='metric-explain'>W_ij is <code>fd_sum</code>; L_ij is <code>len_mean</code>; lambda_s is the subject-specific distance-decay coefficient. "
        "SR exception pct is a fraction of candidate SR edges flagged by the 3 SD rule; for example, 0.026 means about 2.6% of SR candidate edges are exceptions. "
        "This remains structural-only: no Kuramoto/metastability or SC-FC claim is made without real functional matrices.</div>"
        "</div>",
        unsafe_allow_html=True,
    )


def edr_range_count_summary_html(subject_path: Path) -> str:
    if not subject_path.exists():
        return ""
    subject = _safe_read_csv(subject_path)
    if subject is None or subject.empty:
        return ""
    pieces = []
    for prefix, label in (("sr", "SR"), ("mr", "MR"), ("lr", "LR")):
        edge_col = f"{prefix}_edge_count"
        exc_col = f"{prefix}_exception_count"
        pct_col = f"{prefix}_exception_pct"
        if edge_col not in subject.columns:
            continue
        edges = pd.to_numeric(subject[edge_col], errors="coerce").dropna()
        if edges.empty:
            continue
        q1 = edges.quantile(0.25)
        med = edges.median()
        q3 = edges.quantile(0.75)
        exc_text = ""
        if exc_col in subject.columns:
            exc = pd.to_numeric(subject[exc_col], errors="coerce").dropna()
            if not exc.empty:
                exc_text = f"; median exceptions {format_number(exc.median(), digits=2)}"
        pct_text = ""
        if pct_col in subject.columns:
            pct = pd.to_numeric(subject[pct_col], errors="coerce").dropna()
            if not pct.empty:
                pct_text = f" ({format_number(pct.median() * 100.0, digits=2)}%)"
        pieces.append(
            f"<b>{label}</b>: candidate edges median {escape(format_number(med, digits=2))} "
            f"[IQR {escape(format_number(q1, digits=2))}-{escape(format_number(q3, digits=2))}]"
            f"{escape(exc_text)}{escape(pct_text)}"
        )
    if not pieces:
        return ""
    return (
        "<div class='metric-explain'><b>Current candidate-edge counts per subject:</b> "
        + "; ".join(pieces)
        + ". Counts vary by subject because only finite positive fd_sum/len_mean edges are included.</div>"
    )


def median_summary_html(metric: str, desc: pd.DataFrame) -> str:
    if desc.empty or "median" not in desc.columns:
        return ""
    rows = []
    medians = {}
    for _, row in desc.iterrows():
        group = str(row.get("group"))
        if group not in GROUP_ORDER:
            continue
        value = pd.to_numeric(pd.Series([row.get("median")]), errors="coerce").iloc[0]
        if pd.notna(value):
            medians[group] = float(value)
    if not medians:
        return ""
    direction = metric_direction(metric)
    reference = medians.get("CN")
    for group in GROUP_ORDER:
        if group not in medians:
            continue
        value = medians[group]
        chip_class = "chip-neutral"
        chip_text = "reference" if group == "CN" else "comparable"
        if reference is not None and group != "CN":
            if math.isclose(value, reference, rel_tol=1e-6, abs_tol=1e-12):
                chip_text = "comparable"
            elif value > reference:
                chip_text = "higher"
                chip_class = "chip-green" if direction == "higher_better" else "chip-red" if direction == "lower_better" else "chip-neutral"
            else:
                chip_text = "lower"
                chip_class = "chip-red" if direction == "higher_better" else "chip-green" if direction == "lower_better" else "chip-neutral"
        rows.append(
            "<div class='median-row'>"
            f"<span><b>{group}</b> median</span>"
            f"<span class='median-value'>{format_number(value)}</span>"
            f"<span class='median-status'><span class='median-chip {chip_class}'>{chip_text}</span></span>"
            "</div>"
        )
    return "".join(rows)


def result_card_html(section: Section, metric: str, desc: pd.DataFrame, pairwise: pd.DataFrame | None) -> str:
    return (
        "<div class='result-card'>"
        f"<h4>{metric_label(metric)}</h4>"
        f"<div class='metric-explain'>{metric_explanation(section, metric)}</div>"
        f"<div class='chart-finding'>{one_line_finding(section, metric, desc, pairwise)}</div>"
        f"{median_summary_html(metric, desc)}"
        "</div>"
    )


def pairwise_median_direction(row: pd.Series) -> str:
    group_a = str(row.get("group_a", "")).strip()
    group_b = str(row.get("group_b", "")).strip()
    value_a = pd.to_numeric(pd.Series([row.get("median_a")]), errors="coerce").iloc[0]
    value_b = pd.to_numeric(pd.Series([row.get("median_b")]), errors="coerce").iloc[0]
    if not group_a or not group_b or pd.isna(value_a) or pd.isna(value_b):
        return f"{group_a} vs {group_b}".strip()
    if math.isclose(float(value_a), float(value_b), rel_tol=1e-6, abs_tol=1e-12):
        return f"{group_a} = {group_b}"
    return f"{group_a} > {group_b}" if float(value_a) > float(value_b) else f"{group_a} < {group_b}"


def metric_stats_display_table(desc: pd.DataFrame, pairwise: pd.DataFrame | None) -> pd.DataFrame:
    rows = []
    if desc is not None and not desc.empty:
        for _, row in desc.iterrows():
            group = str(row.get("group", "")).strip()
            if group not in GROUP_ORDER:
                continue
            n_value = pd.to_numeric(pd.Series([row.get("n")]), errors="coerce").iloc[0]
            rows.append(
                {
                    "row": group,
                    "n": str(int(n_value)) if pd.notna(n_value) else "",
                    "mean": row.get("mean"),
                    "median": row.get("median"),
                    "q1": row.get("q1"),
                    "q3": row.get("q3"),
                    "direction": "",
                    "p": pd.NA,
                    "BH q": pd.NA,
                    "delta": pd.NA,
                }
            )
    if pairwise is not None and not pairwise.empty:
        p_col = pairwise_p_col(pairwise)
        q_col = pairwise_q_col(pairwise)
        for _, row in pairwise.iterrows():
            group_a = str(row.get("group_a", "")).strip()
            group_b = str(row.get("group_b", "")).strip()
            n_a = pd.to_numeric(pd.Series([row.get("n_a")]), errors="coerce").iloc[0]
            n_b = pd.to_numeric(pd.Series([row.get("n_b")]), errors="coerce").iloc[0]
            n_text = (
                f"{int(n_a)}/{int(n_b)}"
                if pd.notna(n_a) and pd.notna(n_b)
                else ""
            )
            rows.append(
                {
                    "row": f"{group_a} vs {group_b}".strip(),
                    "n": n_text,
                    "mean": pd.NA,
                    "median": pd.NA,
                    "q1": pd.NA,
                    "q3": pd.NA,
                    "direction": pairwise_median_direction(row),
                    "p": row.get(p_col) if p_col else pd.NA,
                    "BH q": row.get(q_col) if q_col else pd.NA,
                    "delta": row.get("cliffs_delta", pd.NA),
                }
            )
    return pd.DataFrame(rows)


def render_metric_definition_inference_table(
    section: Section,
    metric: str,
    desc: pd.DataFrame,
    pairwise: pd.DataFrame | None,
    key: str,
) -> None:
    st.markdown(
        "<div class='edr-definition-box'>"
        f"<h5>{escape(metric_label(metric))}</h5>"
        f"<p>{escape(metric_explanation(section, metric))}</p>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div class='edr-inference-box'>{escape(one_line_finding(section, metric, desc, pairwise))}</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<div class='edr-table-gap'></div>", unsafe_allow_html=True)
    stats_table = metric_stats_display_table(desc, pairwise)
    if stats_table.empty:
        st.info("No statistical table is available for this metric.")
        return
    st.dataframe(
        stats_table,
        width="stretch",
        height=315,
        hide_index=True,
        column_config={
            "row": "Group / contrast",
            "n": "n",
            "mean": st.column_config.NumberColumn("mean", format="%.4g"),
            "median": st.column_config.NumberColumn("median", format="%.4g"),
            "q1": st.column_config.NumberColumn("Q1", format="%.4g"),
            "q3": st.column_config.NumberColumn("Q3", format="%.4g"),
            "direction": "Median direction",
            "p": st.column_config.NumberColumn("p", format="%.3g"),
            "BH q": st.column_config.NumberColumn("BH q", format="%.3g"),
            "delta": st.column_config.NumberColumn("Cliff's delta", format="%.3f"),
        },
        key=key,
    )


def bh_fdr(p_values: list[float]) -> list[float]:
    indexed = sorted(
        [(idx, float(value)) for idx, value in enumerate(p_values) if pd.notna(value)],
        key=lambda item: item[1],
    )
    q_values = [pd.NA] * len(p_values)
    m = len(indexed)
    running = 1.0
    for rank_from_end, (idx, p_value) in enumerate(reversed(indexed), start=1):
        rank = m - rank_from_end + 1
        running = min(running, p_value * m / rank)
        q_values[idx] = min(running, 1.0)
    return q_values


@st.cache_data(ttl=300, show_spinner=False)
def local_roi_pairwise_rankings(long_csv: str, metric: str, group_a: str, group_b: str, label_csv: str) -> pd.DataFrame:
    from scipy.stats import mannwhitneyu

    df = pd.read_csv(long_csv)
    needed = {"subject_id", "group", "node", "node_name", "metric", "value"}
    if not needed.issubset(df.columns):
        return pd.DataFrame()
    df = df[df["metric"].astype(str) == metric].copy()
    df = df[df["group"].astype(str).isin([group_a, group_b])].copy()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df[df["value"].notna()].copy()

    rows = []
    for (node, node_name), sub in df.groupby(["node", "node_name"], dropna=False):
        a = sub.loc[sub["group"].astype(str) == group_a, "value"].dropna()
        b = sub.loc[sub["group"].astype(str) == group_b, "value"].dropna()
        if len(a) < 5 or len(b) < 5:
            continue
        try:
            test = mannwhitneyu(a, b, alternative="two-sided")
            p_value = float(test.pvalue)
            u_value = float(test.statistic)
            cliffs_delta = (2.0 * u_value / (len(a) * len(b))) - 1.0
        except Exception:
            p_value = math.nan
            cliffs_delta = math.nan
        median_a = float(a.median())
        median_b = float(b.median())
        rows.append(
            {
                "node": int(node),
                "node_name": str(node_name),
                "metric": metric,
                "comparison": f"{group_a} vs {group_b}",
                f"n_{group_a}": int(len(a)),
                f"n_{group_b}": int(len(b)),
                f"median_{group_a}": median_a,
                f"median_{group_b}": median_b,
                "median_diff_a_minus_b": median_a - median_b,
                "higher_group": group_a if median_a > median_b else group_b if median_b > median_a else "tie",
                "p_value": p_value,
                "cliffs_delta": cliffs_delta,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["q_value"] = bh_fdr(list(out["p_value"]))
    out = out.sort_values(["p_value", "q_value", "node"], na_position="last").reset_index(drop=True)
    out.insert(0, "rank", range(1, len(out) + 1))

    labels = aal_label_lookup(label_csv)
    if not labels.empty and {"node", "atlas_label"}.issubset(labels.columns):
        out = out.merge(labels[["node", "atlas_label", "atlas_version"]], on="node", how="left")
    else:
        out["atlas_label"] = out["node_name"]
        out["atlas_version"] = "unmapped"
    out["atlas_label"] = out["atlas_label"].fillna(out["node_name"])
    return out


def roi_direction_label(rankings: pd.DataFrame, group_a: str, group_b: str, top_k: int | None = None) -> str:
    if rankings.empty or "higher_group" not in rankings.columns:
        return f"{group_a} vs {group_b}"
    rows = rankings.head(top_k).copy() if top_k else rankings.copy()
    counts = rows["higher_group"].astype(str).value_counts()
    a_count = int(counts.get(group_a, 0))
    b_count = int(counts.get(group_b, 0))
    if a_count > b_count:
        return f"{group_a} > {group_b}"
    if b_count > a_count:
        return f"{group_a} < {group_b}"
    return f"{group_a} = {group_b}"


def pairwise_direction_sentence(rankings: pd.DataFrame, metric: str, group_a: str, group_b: str, top_k: int = 3) -> str:
    label = metric_label(metric)
    rows = rankings.head(top_k).copy()
    if rows.empty:
        return f"No {label} ROI ranking is available for {group_a} vs {group_b}."
    direction = roi_direction_label(rows, group_a, group_b)
    counts = rows["higher_group"].astype(str).value_counts()
    higher = group_a if int(counts.get(group_a, 0)) >= int(counts.get(group_b, 0)) else group_b
    lower = group_b if higher == group_a else group_a
    if direction.endswith("="):
        return f"{label} has mixed median direction across the top {len(rows)} ROIs for {group_a} vs {group_b}."
    return f"{label} median is generally lower in {lower} than {higher} across the top {len(rows)} p-ranked ROIs ({direction})."


def top_roi_items(rankings: pd.DataFrame, top_k: int = 3) -> str:
    items = []
    for _, row in rankings.head(top_k).iterrows():
        label = str(row.get("atlas_label") or row.get("node_name") or "ROI")
        node = str(row.get("node_name") or "")
        p_value = format_p(row.get("p_value"))
        suffix = f" ({node}, p={p_value})" if node and node != label else f" (p={p_value})"
        items.append(f"<li>{escape(label)}{escape(suffix)}</li>")
    return "<ol>" + "".join(items) + "</ol>" if items else "<p>No ranked ROI rows available.</p>"


def roi_pairwise_inference_html(metric: str, pair_rankings: dict[tuple[str, str], pd.DataFrame]) -> str:
    blocks = []
    for group_a, group_b in [("CN", "MCI"), ("CN", "AD"), ("MCI", "AD")]:
        rows = pair_rankings.get((group_a, group_b), pd.DataFrame())
        if rows.empty:
            body = "<p>No rows available.</p>"
            sentence = f"No {metric_label(metric)} rows are available for {group_a} vs {group_b}."
        else:
            sentence = pairwise_direction_sentence(rows, metric, group_a, group_b, top_k=3)
            body = top_roi_items(rows, top_k=3)
        blocks.append(
            "<div class='roi-subsection'>"
            f"<h5>{escape(group_a)} vs {escape(group_b)} [Top 3]</h5>"
            f"<div class='roi-result'><b>Result:</b> {escape(sentence)}</div>"
            f"{body}"
            "</div>"
        )
    return (
        "<div class='result-card roi-inference-card'>"
        "<h4>Pairwise ROI inference</h4>"
        "<div class='roi-inference-lede'>Top 3 lists use the selected metric and rank ROIs by Mann-Whitney U p-value. BH q remains the FDR-control column for thesis claims.</div>"
        + "".join(blocks)
        + "</div>"
    )


def row_median_direction(row: pd.Series, group_a: str, group_b: str) -> str:
    value_a = pd.to_numeric(pd.Series([row.get(f"median_{group_a}")]), errors="coerce").iloc[0]
    value_b = pd.to_numeric(pd.Series([row.get(f"median_{group_b}")]), errors="coerce").iloc[0]
    if pd.isna(value_a) or pd.isna(value_b):
        return f"{group_a} vs {group_b}"
    if math.isclose(float(value_a), float(value_b), rel_tol=1e-6, abs_tol=1e-12):
        return f"{group_a} = {group_b}"
    return f"{group_a} > {group_b}" if float(value_a) > float(value_b) else f"{group_a} < {group_b}"


@st.cache_data(ttl=300, show_spinner=False)
def local_roi_cross_comparison_summary(long_csv: str, metric: str, top_n: int, label_csv: str) -> pd.DataFrame:
    frames = []
    for group_a, group_b in [("CN", "MCI"), ("CN", "AD"), ("MCI", "AD")]:
        rankings = local_roi_pairwise_rankings(long_csv, metric, group_a, group_b, label_csv)
        if rankings.empty:
            continue
        top = rankings.head(top_n).copy()
        top["pair_rank"] = top["rank"]
        top["direction"] = top.apply(
            lambda row: f"{group_a}>{group_b}" if row.get("higher_group") == group_a else f"{group_a}<{group_b}" if row.get("higher_group") == group_b else f"{group_a}={group_b}",
            axis=1,
        )
        frames.append(top)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    combined["p_value"] = pd.to_numeric(combined["p_value"], errors="coerce")
    combined["q_value"] = pd.to_numeric(combined["q_value"], errors="coerce")
    combined["nominal_p_lt_0_05"] = combined["p_value"] < 0.05
    combined["fdr_q_lt_0_05"] = combined["q_value"] < 0.05
    significant = combined[combined["nominal_p_lt_0_05"]].copy()
    if significant.empty:
        significant = combined.copy()

    summary = (
        significant.groupby(["node", "node_name", "atlas_label"], dropna=False)
        .agg(
            top_n_occurrences=("comparison", "nunique"),
            fdr_significant_comparisons=("fdr_q_lt_0_05", "sum"),
            best_p=("p_value", "min"),
            best_q=("q_value", "min"),
            comparisons=("comparison", lambda values: "; ".join(dict.fromkeys(map(str, values)))),
            directions=("direction", lambda values: "; ".join(dict.fromkeys(map(str, values)))),
        )
        .reset_index()
    )
    summary = summary.sort_values(
        ["fdr_significant_comparisons", "top_n_occurrences", "best_p", "best_q"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)
    summary.insert(0, "rank", range(1, len(summary) + 1))
    return summary


def ranked_roi_chart(rankings: pd.DataFrame, group_a: str, group_b: str) -> go.Figure:
    chart_df = rankings.copy()
    chart_df["minus_log10_p"] = -pd.to_numeric(chart_df["p_value"], errors="coerce").clip(lower=1e-300).map(math.log10)
    chart_df["label"] = chart_df.apply(
        lambda row: f"{int(row['rank'])}. {row['atlas_label']}",
        axis=1,
    )
    chart_df["p_label"] = chart_df["p_value"].map(lambda value: f"p={format_p(value)}")
    direction = roi_direction_label(chart_df, group_a, group_b, top_k=len(chart_df))
    title = f"Top ROI differences: {metric_label(str(chart_df['metric'].iloc[0]))}, {group_a} vs {group_b} ({direction})"
    fig = go.Figure()
    fig.add_bar(
        x=chart_df["minus_log10_p"],
        y=chart_df["label"],
        orientation="h",
        marker_color="#5dade2",
        text=chart_df["p_label"],
        textposition="outside",
        cliponaxis=False,
        hovertemplate=(
            "Rank=%{customdata[0]}<br>"
            "AAL=%{customdata[1]}<br>"
            "Atlas=%{customdata[2]}<br>"
            "Higher median=%{customdata[3]}<br>"
            "p=%{customdata[4]:.3g}<br>"
            "BH q=%{customdata[5]:.3g}<extra></extra>"
        ),
        customdata=chart_df[["rank", "node_name", "atlas_label", "higher_group", "p_value", "q_value"]],
    )
    max_x = float(chart_df["minus_log10_p"].max()) if not chart_df.empty else 1.0
    fig.update_layout(
        title={"text": f"<b>{escape(title)}</b><br><sup>Ranked by Mann-Whitney U p-value; BH q is shown in the table.</sup>", "x": 0.0, "xanchor": "left"},
        template="plotly_white",
        paper_bgcolor=CHART_BACKGROUND,
        plot_bgcolor=CHART_BACKGROUND,
        font={"color": CHART_TEXT, "size": 13},
        title_font={"color": CHART_TEXT, "size": 17},
        height=min(max(32 * len(chart_df) + 120, 340), 920),
        margin={"l": 260, "r": 130, "t": 82, "b": 58},
        showlegend=False,
        xaxis={
            "title": {"text": "-log10 p", "font": {"color": CHART_TEXT, "size": 13}},
            "gridcolor": CHART_GRID,
            "linecolor": CHART_STROKE,
            "tickfont": {"color": CHART_TEXT, "size": 12},
            "range": [0, max_x * 1.16 if max_x > 0 else 1],
        },
        yaxis={
            "title": "",
            "autorange": "reversed",
            "automargin": True,
            "tickfont": {"color": CHART_TEXT, "size": 12},
            "linecolor": CHART_STROKE,
        },
    )
    fig.update_traces(textfont={"color": CHART_TEXT, "size": 11})
    return fig


@st.fragment
def render_local_aal_viewer(long_path_str: str, hide_visual_outliers: bool) -> bool:
    long_path = Path(long_path_str)
    df = _safe_read_csv(long_path)
    if df is None or df.empty or not {"group", "node_name", "metric", "value"}.issubset(df.columns):
        return False

    st.subheader("AAL Region Viewer")
    st.caption("Select one AAL3 region to visualize. This viewer runs as a Streamlit fragment, so changing the selector does not rerun the whole dashboard page.")

    metrics = [m for m in ("fa_mean", "md_mean", "ad_mean", "rd_mean") if m in set(df["metric"].astype(str))]
    if not metrics:
        metrics = sorted(df["metric"].dropna().astype(str).unique())
    if not metrics:
        return False

    c1, c2 = st.columns((0.9, 2.1))
    with c1:
        metric = st.selectbox(
            "Metric to visualize",
            metrics,
            index=0,
            format_func=metric_label,
            key="local-aal-viewer-metric",
        )

    metric_df = df[df["metric"].astype(str) == metric].copy()
    labels = aal_label_lookup(str(AAL_LABEL_CSV))
    node_lookup = metric_df[["node", "node_name"]].drop_duplicates().copy()
    node_lookup["node"] = pd.to_numeric(node_lookup["node"], errors="coerce").astype("Int64")
    if not labels.empty and {"node", "atlas_label"}.issubset(labels.columns):
        node_lookup = node_lookup.merge(labels[["node", "atlas_label"]], on="node", how="left")
    node_lookup["atlas_label"] = node_lookup.get("atlas_label", node_lookup["node_name"]).fillna(node_lookup["node_name"])
    node_lookup = node_lookup.sort_values(["node", "node_name"]).reset_index(drop=True)
    node_options = [
        (str(row.node_name), f"{row.node_name} | {row.atlas_label}")
        for row in node_lookup.itertuples(index=False)
    ]
    if not node_options:
        return False
    option_labels = [label for _, label in node_options]
    option_to_node = {label: node for node, label in node_options}
    with c2:
        selected_label = st.selectbox(
            "AAL3 region",
            option_labels,
            index=0,
            key=f"local-aal-viewer-node-{metric}",
        )
    node = option_to_node[selected_label]
    atlas_label = selected_label.split(" | ", 1)[1] if " | " in selected_label else node

    node_df = metric_df[metric_df["node_name"].astype(str) == node].copy()
    data = subject_metric_frame(node_df, metric, value_col="value")
    if data.empty:
        st.info("No subject values were found for the selected AAL region.")
        return True

    desc = compute_descriptives_from_data(data)
    chart, counts, plotted_counts, removed_counts = live_distribution_chart(
        data,
        f"{metric_label(metric)} at {node} - {atlas_label}",
        hide_visual_outliers=hide_visual_outliers,
        stats_df=desc,
        subtitle_lines=[],
    )
    left, right = st.columns((1.65, 1), gap="large")
    with left:
        st.plotly_chart(chart, width="stretch", config={"displaylogo": False})
        st.caption(f"Live chart from `{long_path.relative_to(ANALYSIS_ROOT).as_posix()}`.")
        if sum(removed_counts.values()):
            st.caption(
                "Visual-only 1.5-IQR trim active; statistics and CSV tables remain unchanged. "
                f"Hidden points: {group_count_text(removed_counts)}."
            )
        render_dti_formula_block(metric)
    with right:
        st.markdown(
            "<div class='result-card'>"
            f"<h4>{escape(metric_label(metric))} at {escape(node)}</h4>"
            f"<div class='metric-explain'><b>AAL3 label:</b> {escape(atlas_label)}</div>"
            f"<div class='chart-finding'>{escape(one_line_finding(Section('Local / ROI DTI', '', ()), metric, desc, None))}</div>"
            f"{median_summary_html(metric, desc)}"
            "</div>",
            unsafe_allow_html=True,
        )
    return True


def render_local_roi_pairwise_panel(csv_files: list[Path], hide_visual_outliers: bool) -> bool:
    long_path = next((p for p in csv_files if p.name == "live_node_microstructure_long.csv"), None)
    if long_path is None:
        return False
    df_head = _safe_read_csv(long_path)
    if df_head is None or df_head.empty or "metric" not in df_head.columns or "group" not in df_head.columns:
        return False

    render_node_network_aggregate(df_head, "micro", "local-roi")
    st.subheader("Pairwise ROI Ranking")
    c1, c2, c3, c4 = st.columns((1.1, 1.1, 1.1, 1))
    available_metrics = [m for m in ("fa_mean", "md_mean", "ad_mean", "rd_mean") if m in set(df_head["metric"].astype(str))]
    if not available_metrics:
        available_metrics = sorted(df_head["metric"].dropna().astype(str).unique())
    with c1:
        metric = st.selectbox(
            "Metric",
            available_metrics,
            index=0,
            format_func=metric_label,
            key="local-roi-pairwise-metric",
        )
    pair_options = [("CN", "MCI"), ("CN", "AD"), ("MCI", "AD")]
    with c2:
        pair_label = st.selectbox(
            "Group comparison",
            [f"{a} vs {b}" for a, b in pair_options],
            index=0,
            key="local-roi-pairwise-groups",
        )
    group_a, group_b = pair_label.split(" vs ")
    with c3:
        top_n = st.slider("Top N ROIs", min_value=5, max_value=80, value=20, step=5, key="local-roi-pairwise-topn")
    with c4:
        p_only = st.checkbox("Only p<0.05", value=True, key="local-roi-pairwise-p")

    all_pair_rankings = {
        pair: local_roi_pairwise_rankings(str(long_path), metric, pair[0], pair[1], str(AAL_LABEL_CSV))
        for pair in pair_options
    }
    rankings = all_pair_rankings.get((group_a, group_b), pd.DataFrame())
    if rankings.empty:
        st.info("No pairwise ROI ranking could be computed for the selected options.")
        return True
    display_rankings = rankings[pd.to_numeric(rankings["p_value"], errors="coerce") < 0.05].copy() if p_only else rankings.copy()
    display_rankings = display_rankings.head(top_n).copy()
    if display_rankings.empty:
        st.info("No ROIs have p<0.05 for this comparison; disable the p<0.05 filter to inspect the full p-value ranking.")
        return True

    cross_summary = local_roi_cross_comparison_summary(str(long_path), metric, top_n, str(AAL_LABEL_CSV))
    st.markdown("#### Top Regions Repeated Across Pairwise Comparisons")
    summary_col, cn_mci_col = st.columns((1.35, 1), gap="large")
    with summary_col:
        st.markdown("##### Repeated Top-N Regions")
        if cross_summary.empty:
            st.info("No repeated Top-N ROI summary could be computed across CN vs MCI, CN vs AD, and MCI vs AD.")
        else:
            cross_display = cross_summary.head(top_n).copy()
            st.caption("AAL3 regions appearing in the selected Top N across CN vs MCI, CN vs AD, and MCI vs AD.")
            st.dataframe(
                cross_display[
                    [
                        "rank",
                        "node_name",
                        "atlas_label",
                        "top_n_occurrences",
                        "fdr_significant_comparisons",
                        "best_p",
                        "best_q",
                        "comparisons",
                        "directions",
                    ]
                ],
                width="stretch",
                height=390,
                hide_index=True,
                column_config={
                    "rank": st.column_config.NumberColumn("Rank", format="%d"),
                    "top_n_occurrences": st.column_config.NumberColumn("Top-N hits", format="%d"),
                    "fdr_significant_comparisons": st.column_config.NumberColumn("BH q<0.05 hits", format="%d"),
                    "best_p": st.column_config.NumberColumn("Best p", format="%.3g"),
                    "best_q": st.column_config.NumberColumn("Best BH q", format="%.3g"),
                },
            )
    with cn_mci_col:
        st.markdown("##### CN vs MCI Top 10")
        cn_mci_rankings = all_pair_rankings.get(("CN", "MCI"), pd.DataFrame()).head(10).copy()
        if cn_mci_rankings.empty:
            st.info("No CN vs MCI ranking rows are available for the selected metric.")
        else:
            cn_mci_rankings["fdr_q_lt_0_05"] = pd.to_numeric(cn_mci_rankings["q_value"], errors="coerce") < 0.05
            cn_mci_rankings["median_direction"] = cn_mci_rankings.apply(lambda row: row_median_direction(row, "CN", "MCI"), axis=1)
            st.caption("Top 10 CN vs MCI ROIs ranked by Mann-Whitney U p-value.")
            st.dataframe(
                cn_mci_rankings[
                    [
                        "rank",
                        "node_name",
                        "atlas_label",
                        "median_CN",
                        "median_MCI",
                        "median_diff_a_minus_b",
                        "median_direction",
                        "p_value",
                        "q_value",
                        "fdr_q_lt_0_05",
                    ]
                ],
                width="stretch",
                height=390,
                hide_index=True,
                column_config={
                    "rank": st.column_config.NumberColumn("Rank", format="%d"),
                    "median_CN": st.column_config.NumberColumn("CN median", format="%.4f"),
                    "median_MCI": st.column_config.NumberColumn("MCI median", format="%.4f"),
                    "median_diff_a_minus_b": st.column_config.NumberColumn("CN-MCI diff", format="%.4f"),
                    "median_direction": "Median direction",
                    "p_value": st.column_config.NumberColumn("p", format="%.3g"),
                    "q_value": st.column_config.NumberColumn("BH q", format="%.3g"),
                    "fdr_q_lt_0_05": st.column_config.CheckboxColumn("BH q<0.05"),
                },
            )

    st.markdown("#### Selected Pairwise Ranking")
    chart_col, inference_col = st.columns((1.35, 1), gap="large")
    with chart_col:
        st.plotly_chart(ranked_roi_chart(display_rankings, group_a, group_b), width="stretch", config={"displaylogo": False})
        st.caption(f"Live ranking from `{long_path.relative_to(ANALYSIS_ROOT).as_posix()}`. AAL3 atlas path: `{AAL_ATLAS_NII}`.")
    with inference_col:
        st.markdown(roi_pairwise_inference_html(metric, all_pair_rankings), unsafe_allow_html=True)
    st.divider()
    render_local_aal_viewer(str(long_path), hide_visual_outliers)
    return True


def node_location_html(row: pd.Series | None) -> str:
    if row is None or row.empty:
        return "<div class='metric-explain'><b>AAL3 location:</b> atlas label is not mapped for this node.</div>"
    atlas_label = str(row.get("atlas_label") or row.get("node_name") or "unmapped")
    node_name = str(row.get("node_name") or "")
    atlas_value = row.get("atlas_value", pd.NA)
    atlas_version = str(row.get("atlas_version") or "AAL3")
    hemi = "left" if atlas_label.endswith("_L") else "right" if atlas_label.endswith("_R") else "midline/bilateral"
    coord_bits = []
    for axis in ("x", "y", "z"):
        value = pd.to_numeric(pd.Series([row.get(axis)]), errors="coerce").iloc[0]
        if pd.notna(value):
            coord_bits.append(f"{axis.upper()}={format_number(value, digits=1)}")
    coord_text = ", ".join(coord_bits) if coord_bits else "centroid unavailable"
    return (
        "<div class='metric-explain'>"
        f"<b>AAL3 location:</b> {escape(atlas_label)} ({escape(node_name)}), "
        f"atlas value {escape(format_number(atlas_value, digits=0))}, {escape(hemi)}; "
        f"MNI centroid {escape(coord_text)}. "
        f"<span class='small-muted'>{escape(atlas_version)}</span>"
        "</div>"
    )


def node_graph_lookup(df: pd.DataFrame) -> pd.DataFrame:
    lookup = df[["node", "node_name"]].drop_duplicates().copy()
    lookup["node"] = pd.to_numeric(lookup["node"], errors="coerce").astype("Int64")
    labels = aal_label_lookup(str(AAL_LABEL_CSV))
    if not labels.empty and {"node", "atlas_label"}.issubset(labels.columns):
        keep = [c for c in ("node", "atlas_label", "atlas_value", "atlas_version") if c in labels.columns]
        lookup = lookup.merge(labels[keep], on="node", how="left")
    centroids = aal_centroids(str(AAL_ATLAS_NII), str(AAL_LABEL_CSV))
    if not centroids.empty and {"node", "x", "y", "z"}.issubset(centroids.columns):
        lookup = lookup.merge(centroids[["node", "x", "y", "z"]], on="node", how="left")
    lookup["atlas_label"] = lookup.get("atlas_label", lookup["node_name"]).fillna(lookup["node_name"])
    lookup["atlas_version"] = lookup.get("atlas_version", pd.Series(["AAL3"] * len(lookup))).fillna("AAL3")
    return lookup.sort_values(["node", "node_name"]).reset_index(drop=True)


def node_graph_metric_options(lookup: pd.DataFrame, tests_df: pd.DataFrame | None = None) -> list[tuple[str, str]]:
    ranked = lookup.copy()
    if tests_df is not None and not tests_df.empty and "node_name" in tests_df.columns:
        order = tests_df[["node_name"]].drop_duplicates().reset_index().rename(columns={"index": "_test_order"})
        ranked = ranked.merge(order, on="node_name", how="left")
        ranked["_test_order"] = pd.to_numeric(ranked["_test_order"], errors="coerce").fillna(len(ranked) + 1)
        ranked = ranked.sort_values(["_test_order", "node"])
    out = []
    for row in ranked.itertuples(index=False):
        node_name = str(row.node_name)
        atlas_label = str(getattr(row, "atlas_label", "") or node_name)
        out.append((node_name, f"{node_name} | {atlas_label}"))
    return out


def render_node_metric_panel(
    df: pd.DataFrame,
    csv_files: list[Path],
    metric: str,
    long_path: Path,
    hide_visual_outliers: bool,
) -> None:
    tests_path = related_metric_file(csv_files, f"{metric}_node", "_tests.csv")
    tests_df = _safe_read_csv(tests_path) if tests_path is not None else None
    lookup = node_graph_lookup(df)
    options = node_graph_metric_options(lookup, tests_df)
    if not options:
        st.info(f"No AAL3 nodes are available for {metric_label(metric)}.")
        return
    option_labels = [label for _, label in options]
    option_to_node = {label: node for node, label in options}
    selected_label = st.selectbox(
        "AAL3 region",
        option_labels,
        index=0,
        key=f"node-metric-viewer-{metric}",
    )
    node_name = option_to_node[selected_label]
    node_df = df[df["node_name"].astype(str) == str(node_name)].copy()
    data = subject_metric_frame(node_df, metric, value_col=metric)
    if data.empty:
        st.info(f"No subject values were found for {metric_label(metric)} at {node_name}.")
        return
    desc = compute_descriptives_from_data(data)
    node_stat_line = node_test_subtitle(tests_df, node_name)
    node_sig = False
    if tests_df is not None and not tests_df.empty and {"node_name", "kw_q"}.issubset(tests_df.columns):
        rows = tests_df[tests_df["node_name"].astype(str) == str(node_name)]
        if not rows.empty:
            node_sig = bool(pd.to_numeric(pd.Series([rows.iloc[0].get("kw_q")]), errors="coerce").iloc[0] < 0.05)
    atlas_row = lookup[lookup["node_name"].astype(str) == str(node_name)]
    atlas_series = atlas_row.iloc[0] if not atlas_row.empty else None
    atlas_label = str(atlas_series.get("atlas_label") if atlas_series is not None else node_name)
    chart, counts, plotted_counts, removed_counts = live_distribution_chart(
        data,
        f"{metric_label(metric)} at {node_name}",
        hide_visual_outliers=hide_visual_outliers,
        stats_df=desc,
        subtitle_lines=[node_stat_line] if node_stat_line else [],
    )
    st.plotly_chart(chart, width="stretch", config={"displaylogo": False})
    st.caption(f"Live node table from `{long_path.relative_to(ANALYSIS_ROOT).as_posix()}`.")
    if sum(removed_counts.values()):
        st.caption(
            "Visual-only 1.5-IQR trim active; statistics and CSV tables remain unchanged. "
            f"Hidden points: {group_count_text(removed_counts)}."
        )
    finding = (
        f"{atlas_label} shows FDR-significant group variation for {metric_label(metric)}."
        if node_sig
        else f"No FDR-significant node-level group effect is detected for {metric_label(metric)} at {atlas_label}."
    )
    st.markdown(
        "<div class='result-card'>"
        f"<h4>{escape(metric_label(metric))} at {escape(node_name)}</h4>"
        f"<div class='metric-explain'>{escape(metric_explanation(Section('Node Metrics', '', ()), metric))}</div>"
        f"{node_location_html(atlas_series)}"
        f"<div class='chart-finding'>{escape(finding)}</div>"
        f"{median_summary_html(metric, desc)}"
        "</div>",
        unsafe_allow_html=True,
    )


@st.cache_data(ttl=300, show_spinner=False)
def node_graph_pairwise_rankings(long_csv: str, metric: str, group_a: str, group_b: str, label_csv: str) -> pd.DataFrame:
    df = pd.read_csv(long_csv)
    needed = {"subject_id", "group", "node", "node_name", metric}
    if not needed.issubset(df.columns):
        return pd.DataFrame()
    df = df[df["group"].astype(str).isin([group_a, group_b])].copy()
    df[metric] = pd.to_numeric(df[metric], errors="coerce")
    df = df[df[metric].notna()].copy()
    rows = []
    for (node, node_name), sub in df.groupby(["node", "node_name"], dropna=False):
        a = sub.loc[sub["group"].astype(str) == group_a, metric].dropna()
        b = sub.loc[sub["group"].astype(str) == group_b, metric].dropna()
        if len(a) < 5 or len(b) < 5:
            continue
        try:
            test = stats.mannwhitneyu(a, b, alternative="two-sided")
            p_value = float(test.pvalue)
            u_value = float(test.statistic)
            cliffs_delta = (2.0 * u_value / (len(a) * len(b))) - 1.0
        except Exception:
            p_value = math.nan
            cliffs_delta = math.nan
        median_a = float(a.median())
        median_b = float(b.median())
        rows.append(
            {
                "node": int(node),
                "node_name": str(node_name),
                "metric": metric,
                "comparison": f"{group_a} vs {group_b}",
                f"n_{group_a}": int(len(a)),
                f"n_{group_b}": int(len(b)),
                f"median_{group_a}": median_a,
                f"median_{group_b}": median_b,
                "median_diff_a_minus_b": median_a - median_b,
                "higher_group": group_a if median_a > median_b else group_b if median_b > median_a else "tie",
                "p_value": p_value,
                "cliffs_delta": cliffs_delta,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["q_value"] = bh_fdr(list(out["p_value"]))
    out = out.sort_values(["p_value", "q_value", "node"], na_position="last").reset_index(drop=True)
    out.insert(0, "rank", range(1, len(out) + 1))
    labels = aal_label_lookup(label_csv)
    if not labels.empty and {"node", "atlas_label"}.issubset(labels.columns):
        keep = [c for c in ("node", "atlas_label", "atlas_value", "atlas_version") if c in labels.columns]
        out = out.merge(labels[keep], on="node", how="left")
    else:
        out["atlas_label"] = out["node_name"]
        out["atlas_version"] = "unmapped"
    out["atlas_label"] = out["atlas_label"].fillna(out["node_name"])
    return out


@st.cache_data(ttl=300, show_spinner=False)
def node_graph_cross_comparison_summary(long_csv: str, metric: str, top_n: int, label_csv: str) -> pd.DataFrame:
    frames = []
    for group_a, group_b in [("CN", "MCI"), ("CN", "AD"), ("MCI", "AD")]:
        rankings = node_graph_pairwise_rankings(long_csv, metric, group_a, group_b, label_csv)
        if rankings.empty:
            continue
        top = rankings.head(top_n).copy()
        top["pair_rank"] = top["rank"]
        top["direction"] = top.apply(
            lambda row: f"{group_a}>{group_b}" if row.get("higher_group") == group_a else f"{group_a}<{group_b}" if row.get("higher_group") == group_b else f"{group_a}={group_b}",
            axis=1,
        )
        frames.append(top)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    combined["p_value"] = pd.to_numeric(combined["p_value"], errors="coerce")
    combined["q_value"] = pd.to_numeric(combined["q_value"], errors="coerce")
    combined["nominal_p_lt_0_05"] = combined["p_value"] < 0.05
    combined["fdr_q_lt_0_05"] = combined["q_value"] < 0.05
    significant = combined[combined["nominal_p_lt_0_05"]].copy()
    if significant.empty:
        significant = combined.copy()
    summary = (
        significant.groupby(["node", "node_name", "atlas_label"], dropna=False)
        .agg(
            top_n_occurrences=("comparison", "nunique"),
            fdr_significant_comparisons=("fdr_q_lt_0_05", "sum"),
            best_p=("p_value", "min"),
            best_q=("q_value", "min"),
            comparisons=("comparison", lambda values: "; ".join(dict.fromkeys(map(str, values)))),
            directions=("direction", lambda values: "; ".join(dict.fromkeys(map(str, values)))),
        )
        .reset_index()
    )
    summary = summary.sort_values(
        ["fdr_significant_comparisons", "top_n_occurrences", "best_p", "best_q"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)
    summary.insert(0, "rank", range(1, len(summary) + 1))
    return summary


def top_node_items(rankings: pd.DataFrame, top_k: int = 3) -> str:
    items = []
    for _, row in rankings.head(top_k).iterrows():
        label = str(row.get("atlas_label") or row.get("node_name") or "AAL node")
        node = str(row.get("node_name") or "")
        p_value = format_p(row.get("p_value"))
        suffix = f" ({node}, p={p_value})" if node and node != label else f" (p={p_value})"
        items.append(f"<li>{escape(label)}{escape(suffix)}</li>")
    return "<ol>" + "".join(items) + "</ol>" if items else "<p>No ranked node rows available.</p>"


def node_pairwise_inference_html(metric: str, pair_rankings: dict[tuple[str, str], pd.DataFrame]) -> str:
    blocks = []
    for group_a, group_b in [("CN", "MCI"), ("CN", "AD"), ("MCI", "AD")]:
        rows = pair_rankings.get((group_a, group_b), pd.DataFrame())
        if rows.empty:
            sentence = f"No {metric_label(metric)} node rows are available for {group_a} vs {group_b}."
            body = "<p>No rows available.</p>"
        else:
            sentence = pairwise_direction_sentence(rows, metric, group_a, group_b, top_k=3)
            body = top_node_items(rows, top_k=3)
        blocks.append(
            "<div class='roi-subsection'>"
            f"<h5>{escape(group_a)} vs {escape(group_b)} [Top 3]</h5>"
            f"<div class='roi-result'><b>Result:</b> {escape(sentence)}</div>"
            f"{body}"
            "</div>"
        )
    return (
        "<div class='result-card roi-inference-card'>"
        "<h4>Pairwise node inference</h4>"
        "<div class='roi-inference-lede'>Top 3 lists use the selected node metric and rank AAL3 regions by Mann-Whitney U p-value. BH q remains the FDR-control column for thesis claims.</div>"
        + "".join(blocks)
        + "</div>"
    )


def ranked_node_chart(rankings: pd.DataFrame, group_a: str, group_b: str) -> go.Figure:
    chart_df = rankings.copy()
    chart_df["minus_log10_p"] = -pd.to_numeric(chart_df["p_value"], errors="coerce").clip(lower=1e-300).map(math.log10)
    chart_df["label"] = chart_df.apply(lambda row: f"{int(row['rank'])}. {row['atlas_label']}", axis=1)
    chart_df["p_label"] = chart_df["p_value"].map(lambda value: f"p={format_p(value)}")
    direction = roi_direction_label(chart_df, group_a, group_b, top_k=len(chart_df))
    title = f"Top AAL node differences: {metric_label(str(chart_df['metric'].iloc[0]))}, {group_a} vs {group_b} ({direction})"
    fig = go.Figure()
    fig.add_bar(
        x=chart_df["minus_log10_p"],
        y=chart_df["label"],
        orientation="h",
        marker_color="#5dade2",
        text=chart_df["p_label"],
        textposition="outside",
        cliponaxis=False,
        hovertemplate=(
            "Rank=%{customdata[0]}<br>"
            "AAL=%{customdata[1]}<br>"
            "Atlas=%{customdata[2]}<br>"
            "Higher median=%{customdata[3]}<br>"
            "p=%{customdata[4]:.3g}<br>"
            "BH q=%{customdata[5]:.3g}<extra></extra>"
        ),
        customdata=chart_df[["rank", "node_name", "atlas_label", "higher_group", "p_value", "q_value"]],
    )
    max_x = float(chart_df["minus_log10_p"].max()) if not chart_df.empty else 1.0
    fig.update_layout(
        title={"text": f"<b>{escape(title)}</b><br><sup>Ranked by Mann-Whitney U p-value; BH q is shown in the table.</sup>", "x": 0.0, "xanchor": "left"},
        template="plotly_white",
        paper_bgcolor=CHART_BACKGROUND,
        plot_bgcolor=CHART_BACKGROUND,
        font={"color": CHART_TEXT, "size": 13},
        title_font={"color": CHART_TEXT, "size": 17},
        height=min(max(32 * len(chart_df) + 120, 340), 920),
        margin={"l": 260, "r": 130, "t": 82, "b": 58},
        showlegend=False,
        xaxis={
            "title": {"text": "-log10 p", "font": {"color": CHART_TEXT, "size": 13}},
            "gridcolor": CHART_GRID,
            "linecolor": CHART_STROKE,
            "tickfont": {"color": CHART_TEXT, "size": 12},
            "range": [0, max_x * 1.16 if max_x > 0 else 1],
        },
        yaxis={"title": "", "autorange": "reversed", "automargin": True, "tickfont": {"color": CHART_TEXT, "size": 12}, "linecolor": CHART_STROKE},
    )
    fig.update_traces(textfont={"color": CHART_TEXT, "size": 11})
    return fig


def split_comparison(comparison: str) -> tuple[str, str]:
    parts = [part.strip() for part in str(comparison).split(" vs ") if part.strip()]
    if len(parts) >= 2:
        return parts[0], parts[1]
    return "", ""


def row_direction_from_comparison(row: pd.Series, comparison: str) -> str:
    group_a, group_b = split_comparison(comparison)
    if not group_a or not group_b:
        higher = str(row.get("higher_group", "")).strip()
        return higher or "not available"
    direction = row_median_direction(row, group_a, group_b)
    if " vs " not in direction:
        return direction
    higher = str(row.get("higher_group", "")).strip()
    if higher in {group_a, group_b}:
        lower = group_b if higher == group_a else group_a
        return f"{higher} > {lower}"
    return direction


def edr_rank_selected_inference(metric: str, comparison: str, selected: pd.DataFrame, only_fdr: bool) -> str:
    if selected.empty:
        return ""
    q_values = pd.to_numeric(selected.get("q_value"), errors="coerce")
    fdr_rows = selected[q_values < 0.05].copy()
    table_for_claim = fdr_rows if not fdr_rows.empty else selected
    top = table_for_claim.iloc[0]
    label = str(top.get("atlas_label") or top.get("node_name") or "AAL region")
    node_name = str(top.get("node_name") or "")
    direction = row_direction_from_comparison(top, comparison)
    p_text = format_p(top.get("p_value"))
    q_text = format_p(top.get("q_value"))
    top_labels = [
        str(row.get("atlas_label") or row.get("node_name") or "AAL region")
        for _, row in table_for_claim.head(3).iterrows()
    ]
    top_text = ", ".join(top_labels)
    node_suffix = f" ({node_name})" if node_name and node_name != label else ""
    if not fdr_rows.empty:
        return (
            f"For {metric_label(metric)} in {comparison}, the strongest FDR-supported AAL signal is "
            f"{label}{node_suffix}, direction {direction}, q={q_text}, p={p_text}. "
            f"Top regions in the current table: {top_text}."
        )
    if only_fdr:
        return ""
    return (
        f"For {metric_label(metric)} in {comparison}, no displayed top-N AAL region reaches BH q<0.05; "
        f"the strongest screening signal is {label}{node_suffix}, direction {direction}, q={q_text}, p={p_text}."
    )


def edr_repeated_inference(metric: str, rep: pd.DataFrame) -> str:
    if rep.empty:
        return ""
    sig_count = 0
    if "fdr_significant_comparisons" in rep.columns:
        sig_count = int((pd.to_numeric(rep["fdr_significant_comparisons"], errors="coerce").fillna(0) > 0).sum())
    top = rep.iloc[0]
    label = str(top.get("atlas_label") or top.get("node_name") or "AAL region")
    node_name = str(top.get("node_name") or "")
    hits = pd.to_numeric(pd.Series([top.get("top_n_occurrences")]), errors="coerce").iloc[0]
    hit_text = str(int(hits)) if pd.notna(hits) else "multiple"
    comparisons = str(top.get("comparisons") or "the pairwise comparisons")
    directions = str(top.get("directions") or "direction not available")
    q_text = format_p(top.get("best_q"))
    node_suffix = f" ({node_name})" if node_name and node_name != label else ""
    if sig_count:
        return (
            f"For {metric_label(metric)}, {sig_count} repeated AAL regions have at least one FDR-supported comparison; "
            f"the top repeated region is {label}{node_suffix}, appearing in {hit_text} top-N comparisons "
            f"({comparisons}; {directions}; best q={q_text})."
        )
    return (
        f"For {metric_label(metric)}, repeated regions are visible across top-N screens but none have BH q<0.05; "
        f"the top recurrent region is {label}{node_suffix}, appearing in {hit_text} comparisons ({comparisons})."
    )


def render_node_graph_pairwise_panel(long_path: Path) -> None:
    df = _safe_read_csv(long_path)
    if df is None or df.empty:
        return
    st.subheader("Pairwise AAL Ranking")
    c1, c2, c3, c4 = st.columns((1.1, 1.1, 1.1, 1))
    available_metrics = [m for m in NODE_GRAPH_METRICS if m in df.columns]
    with c1:
        metric = st.selectbox("Metric", available_metrics, index=0, format_func=metric_label, key="node-graph-pairwise-metric")
    pair_options = [("CN", "MCI"), ("CN", "AD"), ("MCI", "AD")]
    with c2:
        pair_label = st.selectbox("Group comparison", [f"{a} vs {b}" for a, b in pair_options], index=0, key="node-graph-pairwise-groups")
    group_a, group_b = pair_label.split(" vs ")
    with c3:
        top_n = st.slider("Top N AALs", min_value=5, max_value=80, value=20, step=5, key="node-graph-pairwise-topn")
    with c4:
        p_only = st.checkbox("Only p<0.05", value=True, key="node-graph-pairwise-p")

    all_pair_rankings = {
        pair: node_graph_pairwise_rankings(str(long_path), metric, pair[0], pair[1], str(AAL_LABEL_CSV))
        for pair in pair_options
    }
    rankings = all_pair_rankings.get((group_a, group_b), pd.DataFrame())
    if rankings.empty:
        st.info("No pairwise AAL node ranking could be computed for the selected options.")
        return
    display_rankings = rankings[pd.to_numeric(rankings["p_value"], errors="coerce") < 0.05].copy() if p_only else rankings.copy()
    display_rankings = display_rankings.head(top_n).copy()
    if display_rankings.empty:
        st.info("No AAL nodes have p<0.05 for this comparison; disable the p<0.05 filter to inspect the full p-value ranking.")
        return

    selected_col, common_col = st.columns((1.15, 1), gap="large")
    with selected_col:
        st.markdown(f"##### {group_a} vs {group_b} Top {min(top_n, len(display_rankings))}: {metric_label(metric)}")
        table = display_rankings.copy()
        table["fdr_q_lt_0_05"] = pd.to_numeric(table["q_value"], errors="coerce") < 0.05
        table["median_direction"] = table.apply(lambda row: row_median_direction(row, group_a, group_b), axis=1)
        st.dataframe(
            table[
                [
                    "rank",
                    "node_name",
                    "atlas_label",
                    f"median_{group_a}",
                    f"median_{group_b}",
                    "median_diff_a_minus_b",
                    "median_direction",
                    "p_value",
                    "q_value",
                    "fdr_q_lt_0_05",
                ]
            ],
            width="stretch",
            height=390,
            hide_index=True,
            column_config={
                "rank": st.column_config.NumberColumn("Rank", format="%d"),
                f"median_{group_a}": st.column_config.NumberColumn(f"{group_a} median", format="%.4g"),
                f"median_{group_b}": st.column_config.NumberColumn(f"{group_b} median", format="%.4g"),
                "median_diff_a_minus_b": st.column_config.NumberColumn(f"{group_a}-{group_b} diff", format="%.4g"),
                "median_direction": "Median direction",
                "p_value": st.column_config.NumberColumn("p", format="%.3g"),
                "q_value": st.column_config.NumberColumn("BH q", format="%.3g"),
                "fdr_q_lt_0_05": st.column_config.CheckboxColumn("BH q<0.05"),
            },
        )
    with common_col:
        st.markdown("##### Repeated Top-N Across All Three Comparisons")
        cross_summary = node_graph_cross_comparison_summary(str(long_path), metric, top_n, str(AAL_LABEL_CSV))
        if cross_summary.empty:
            st.info("No repeated Top-N AAL summary could be computed across CN vs MCI, CN vs AD, and MCI vs AD.")
        else:
            st.dataframe(
                cross_summary.head(top_n)[
                    [
                        "rank",
                        "node_name",
                        "atlas_label",
                        "top_n_occurrences",
                        "fdr_significant_comparisons",
                        "best_p",
                        "best_q",
                        "comparisons",
                        "directions",
                    ]
                ],
                width="stretch",
                height=390,
                hide_index=True,
                column_config={
                    "rank": st.column_config.NumberColumn("Rank", format="%d"),
                    "top_n_occurrences": st.column_config.NumberColumn("Top-N hits", format="%d"),
                    "fdr_significant_comparisons": st.column_config.NumberColumn("BH q<0.05 hits", format="%d"),
                    "best_p": st.column_config.NumberColumn("Best p", format="%.3g"),
                    "best_q": st.column_config.NumberColumn("Best BH q", format="%.3g"),
                },
            )

    chart_col, inference_col = st.columns((1.25, 1), gap="large")
    with chart_col:
        st.plotly_chart(ranked_node_chart(display_rankings, group_a, group_b), width="stretch", config={"displaylogo": False})
        st.caption(f"Live ranking from `{long_path.relative_to(ANALYSIS_ROOT).as_posix()}`.")
    with inference_col:
        st.markdown(node_pairwise_inference_html(metric, all_pair_rankings), unsafe_allow_html=True)


def render_node_metrics_layout(csv_files: list[Path], hide_visual_outliers: bool) -> bool:
    long_path = next((p for p in csv_files if p.name == "live_node_graph_long.csv"), None)
    if long_path is None:
        return False
    df = _safe_read_csv(long_path)
    if df is None or df.empty or not {"group", "node", "node_name"}.issubset(df.columns):
        return False
    if not all(metric in df.columns for metric in NODE_GRAPH_METRICS):
        return False

    render_node_network_aggregate(df, "graph", "node-metrics")
    st.subheader("Node Metric Definitions")
    st.caption("These metrics are computed from the fd_sum structural connectome matrix for each AAL3 node.")
    cols = st.columns(3, gap="large")
    for col, metric in zip(cols, ("strength", "degree", "nodal_eff")):
        with col:
            render_node_metric_definition(metric)

    st.subheader("Strength, Degree, And Nodal Efficiency")
    metric_cols = st.columns(3, gap="large")
    for col, metric in zip(metric_cols, ("strength", "degree", "nodal_eff")):
        with col:
            st.markdown(f"### {metric_label(metric)}")
            render_node_metric_panel(df, csv_files, metric, long_path, hide_visual_outliers)
    st.divider()
    render_node_graph_pairwise_panel(long_path)
    return True


def node_test_subtitle(test_df: pd.DataFrame | None, node: str) -> str:
    if test_df is None or test_df.empty:
        return ""
    if "node_name" in test_df.columns:
        rows = test_df[test_df["node_name"].astype(str) == str(node)]
    else:
        rows = pd.DataFrame()
    if rows.empty:
        return ""
    row = rows.iloc[0]
    parts = []
    if "kw_q" in row:
        parts.append(f"q={format_p(row.get('kw_q'))}")
    if "n" in row:
        parts.append(f"n={format_number(row.get('n'), digits=0)}")
    return "Node test: " + " | ".join(parts) if parts else ""


def is_dti_metric(metric: str) -> bool:
    lower = metric.lower()
    return lower.startswith(("fa_", "md_", "rd_", "ad_")) or any(token in lower for token in ("_fa_", "_md_", "_rd_", "_ad_"))


def render_dti_formula_block(metric: str) -> None:
    if not is_dti_metric(metric):
        return
    with st.expander("DTI definitions and formulas", expanded=True):
        st.latex(DTI_FORMULAS[0])
        st.latex(DTI_FORMULAS[1])
        st.latex(DTI_FORMULAS[2])
        st.latex(DTI_FORMULAS[3])
        if "edge_mean" in metric.lower():
            st.latex(r"\mathrm{Subject\ edge\ mean} = \frac{1}{|E|}\sum_{(i,j)\in E} M_{ij}")
        if "edge_median" in metric.lower():
            st.latex(r"\mathrm{Subject\ edge\ median} = \operatorname{median}_{(i,j)\in E}(M_{ij})")
        st.caption(
            "For FA/MD edge matrices, streamline samples are collapsed into each AAL edge value M_ij; "
            "edge mean and edge median then summarize all valid subject-level edges E. RD/AD formulas are shown for tensor context."
        )


def iqr_filtered_for_display(data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int], dict[str, int]]:
    pieces = []
    plotted_counts: dict[str, int] = {}
    removed_counts: dict[str, int] = {}
    for group in GROUP_ORDER:
        group_df = data[data["Group"] == group].copy()
        total = len(group_df)
        if total < 4:
            kept = group_df
        else:
            q1 = group_df["Value"].quantile(0.25)
            q3 = group_df["Value"].quantile(0.75)
            iqr = q3 - q1
            if pd.isna(iqr) or iqr <= 0:
                kept = group_df
            else:
                low = q1 - 1.5 * iqr
                high = q3 + 1.5 * iqr
                kept = group_df[(group_df["Value"] >= low) & (group_df["Value"] <= high)]
        plotted_counts[group] = len(kept)
        removed_counts[group] = total - len(kept)
        pieces.append(kept)
    if not pieces:
        return data, plotted_counts, removed_counts
    return pd.concat(pieces, ignore_index=True), plotted_counts, removed_counts


def subject_metric_frame(df: pd.DataFrame, metric: str, value_col: str | None = None) -> pd.DataFrame:
    source_col = value_col or metric
    data = df[["group", source_col] + (["subject_id"] if "subject_id" in df.columns else [])].copy()
    data = data.rename(columns={"group": "Group", source_col: "Value", "subject_id": "Subject"})
    data["Group"] = data["Group"].astype(str)
    data["Value"] = pd.to_numeric(data["Value"], errors="coerce")
    data = data[data["Group"].isin(GROUP_ORDER)]
    data = data[data["Value"].notna()].copy()
    return data


def live_distribution_chart(
    data: pd.DataFrame,
    title: str,
    hide_visual_outliers: bool,
    stats_df: pd.DataFrame | None = None,
    subtitle_lines: list[str] | None = None,
) -> tuple[go.Figure, dict[str, int], dict[str, int], dict[str, int]]:
    total_counts = {group: int((data["Group"] == group).sum()) for group in GROUP_ORDER}
    chart_data = data.copy()
    plotted_counts = total_counts.copy()
    removed_counts = {group: 0 for group in GROUP_ORDER}
    if hide_visual_outliers:
        chart_data, plotted_counts, removed_counts = iqr_filtered_for_display(chart_data)
    if chart_data.empty:
        empty = go.Figure()
        return empty, total_counts, plotted_counts, removed_counts

    stats = stats_df.copy() if stats_df is not None and not stats_df.empty else compute_descriptives_from_data(data)
    stats["group"] = stats["group"].astype(str)
    y_min = float(chart_data["Value"].min())
    y_max = float(chart_data["Value"].max())
    if y_min == y_max:
        pad = abs(y_min) * 0.1 if y_min else 1.0
    else:
        pad = (y_max - y_min) * 0.16
    label_pad = pad if pad else 1.0
    y_upper = y_max + label_pad * 0.78
    y_lower = y_min - label_pad * 0.18
    stats_by_group = {str(row["group"]): row for _, row in stats.iterrows()}
    group_x = {group: idx for idx, group in enumerate(GROUP_ORDER)}

    fig = go.Figure()
    for group, color in zip(GROUP_ORDER, GROUP_COLORS):
        group_values = chart_data.loc[chart_data["Group"] == group, "Value"]
        group_subjects = chart_data.loc[chart_data["Group"] == group, "Subject"] if "Subject" in chart_data.columns else None
        fig.add_trace(
            go.Violin(
                y=group_values,
                x=[group_x[group]] * len(group_values),
                name=group,
                fillcolor=color,
                line_color=color,
                opacity=0.82,
                points=False,
                box_visible=False,
                meanline_visible=False,
                width=0.72,
                text=list(group_subjects) if group_subjects is not None else None,
                hovertemplate=f"Group={group}<br>Value=%{{y:.4g}}<extra></extra>",
                showlegend=False,
            )
        )

    stat_styles = {
        "Q3": {"dash": "dot", "width": 1.4, "x0": -0.20, "x1": 0.20, "text_x": 0.24, "anchor": "left"},
        "Median": {"dash": "solid", "width": 3.0, "x0": -0.26, "x1": 0.26, "text_x": 0.30, "anchor": "left"},
        "Mean": {"dash": "dash", "width": 2.0, "x0": -0.20, "x1": 0.20, "text_x": -0.24, "anchor": "right"},
        "Q1": {"dash": "dot", "width": 1.4, "x0": -0.20, "x1": 0.20, "text_x": -0.24, "anchor": "right"},
    }
    stat_keys = (("Q3", "q3"), ("Median", "median"), ("Mean", "mean"), ("Q1", "q1"))

    for group, color in zip(GROUP_ORDER, GROUP_COLORS):
        row = stats_by_group.get(group, {})
        x_center = group_x[group]
        n_label = (
            f"n={plotted_counts.get(group, 0)}/{total_counts.get(group, 0)}"
            if plotted_counts.get(group, 0) != total_counts.get(group, 0)
            else f"n={total_counts.get(group, 0)}"
        )
        fig.add_annotation(
            x=x_center,
            y=0.045,
            xref="x",
            yref="paper",
            text=n_label,
            showarrow=False,
            font={"size": 12, "color": CHART_TEXT},
            align="center",
            bgcolor="rgba(248,250,252,0.72)",
            bordercolor="rgba(203,213,225,0.65)",
            borderpad=2,
        )
        for stat_label, stat_col in stat_keys:
            raw_value = row.get(stat_col) if hasattr(row, "get") else pd.NA
            value = pd.to_numeric(pd.Series([raw_value]), errors="coerce").iloc[0]
            if pd.isna(value):
                continue
            style = stat_styles[stat_label]
            fig.add_shape(
                type="line",
                x0=x_center + style["x0"],
                x1=x_center + style["x1"],
                y0=float(value),
                y1=float(value),
                line={"color": color if stat_label != "Mean" else "#111827", "width": style["width"], "dash": style["dash"]},
                layer="above",
            )
            fig.add_annotation(
                x=x_center + style["text_x"],
                y=float(value),
                text=f"{stat_label} {format_number(value)}",
                showarrow=False,
                xanchor=style["anchor"],
                yanchor="middle",
                font={"size": 10, "color": CHART_TEXT},
                align="left",
                bgcolor="rgba(248,250,252,0.80)",
                bordercolor="rgba(203,213,225,0.70)",
                borderpad=1,
            )

    q_lines = subtitle_lines or []
    if q_lines:
        fig.add_annotation(
            x=0.02,
            y=0.96,
            xref="paper",
            yref="paper",
            text="<br>".join(q_lines),
            showarrow=False,
            align="left",
            bgcolor="rgba(255,255,255,0.90)",
            bordercolor="#cbd5e1",
            borderpad=5,
            font={"size": 12, "color": CHART_TEXT},
        )

    fig.update_layout(
        title={"text": title, "x": 0.01, "xanchor": "left", "font": {"size": 18, "color": CHART_TEXT}},
        plot_bgcolor=CHART_BACKGROUND,
        paper_bgcolor=CHART_BACKGROUND,
        height=450,
        margin={"l": 70, "r": 48, "t": 60, "b": 60},
        xaxis={
            "tickmode": "array",
            "tickvals": list(group_x.values()),
            "ticktext": list(GROUP_ORDER),
            "title": "",
            "tickfont": {"color": CHART_TEXT, "size": 13},
            "showgrid": False,
            "linecolor": CHART_STROKE,
            "range": [-0.65, len(GROUP_ORDER) - 0.35],
        },
        yaxis={
            "title": {"text": title, "font": {"color": CHART_TEXT}},
            "range": [y_lower, y_upper],
            "gridcolor": CHART_GRID,
            "linecolor": CHART_STROKE,
            "tickfont": {"color": CHART_TEXT},
            "zerolinecolor": CHART_GRID,
        },
        boxmode="group",
    )
    return fig, total_counts, plotted_counts, removed_counts


def age_metric_association_table(df: pd.DataFrame, x_col: str, y_col: str) -> pd.DataFrame:
    work = df[["group", x_col, y_col]].copy()
    work["group"] = work["group"].astype(str)
    work[x_col] = pd.to_numeric(work[x_col], errors="coerce")
    work[y_col] = pd.to_numeric(work[y_col], errors="coerce")
    work = work[work["group"].isin(GROUP_ORDER) & work[x_col].notna() & work[y_col].notna()].copy()
    rows: list[dict] = []
    groups: list[tuple[str, pd.DataFrame]] = [("Overall", work)]
    groups.extend((group, work[work["group"] == group].copy()) for group in GROUP_ORDER)
    for group, group_df in groups:
        row: dict = {"Group": group, "n": len(group_df)}
        if len(group_df) >= 3 and group_df[x_col].nunique() >= 2 and group_df[y_col].nunique() >= 2:
            x = group_df[x_col].to_numpy(dtype=float)
            y = group_df[y_col].to_numpy(dtype=float)
            lin = stats.linregress(x, y)
            rho, rho_p = stats.spearmanr(x, y, nan_policy="omit")
            try:
                pearson_r, pearson_p = stats.pearsonr(x, y)
            except Exception:
                pearson_r, pearson_p = np.nan, np.nan
            row.update(
                {
                    "Spearman rho": float(rho) if np.isfinite(rho) else np.nan,
                    "Spearman p": float(rho_p) if np.isfinite(rho_p) else np.nan,
                    "Pearson r": float(pearson_r) if np.isfinite(pearson_r) else np.nan,
                    "Pearson p": float(pearson_p) if np.isfinite(pearson_p) else np.nan,
                    "Linear slope": float(lin.slope) if np.isfinite(lin.slope) else np.nan,
                    "Intercept": float(lin.intercept) if np.isfinite(lin.intercept) else np.nan,
                }
            )
        else:
            row.update(
                {
                    "Spearman rho": np.nan,
                    "Spearman p": np.nan,
                    "Pearson r": np.nan,
                    "Pearson p": np.nan,
                    "Linear slope": np.nan,
                    "Intercept": np.nan,
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def scatter_with_group_trends(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    x_title: str,
    y_title: str,
    identity_line: bool = False,
    trend_labels: dict[str, str] | None = None,
    intersections: pd.DataFrame | None = None,
) -> go.Figure:
    work = df[["group", x_col, y_col] + (["subject_id"] if "subject_id" in df.columns else [])].copy()
    work["group"] = work["group"].astype(str)
    work[x_col] = pd.to_numeric(work[x_col], errors="coerce")
    work[y_col] = pd.to_numeric(work[y_col], errors="coerce")
    work = work[work["group"].isin(GROUP_ORDER) & work[x_col].notna() & work[y_col].notna()].copy()
    fig = go.Figure()
    color_map = dict(zip(GROUP_ORDER, GROUP_COLORS))
    for group in GROUP_ORDER:
        group_df = work[work["group"] == group].copy()
        if group_df.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=group_df[x_col],
                y=group_df[y_col],
                mode="markers",
                name=group,
                marker={
                    "size": 9,
                    "color": color_map[group],
                    "opacity": 0.84,
                    "line": {"color": "#ffffff", "width": 0.9},
                },
                text=group_df["subject_id"].astype(str) if "subject_id" in group_df.columns else None,
                hovertemplate=(
                    f"Group={group}<br>{escape(x_title)}=%{{x:.3g}}<br>{escape(y_title)}=%{{y:.4g}}"
                    "<br>%{text}<extra></extra>"
                ),
            )
        )
        if len(group_df) >= 3 and group_df[x_col].nunique() >= 2:
            lin = stats.linregress(group_df[x_col].to_numpy(dtype=float), group_df[y_col].to_numpy(dtype=float))
            x_line = np.linspace(float(group_df[x_col].min()), float(group_df[x_col].max()), 60)
            y_line = lin.intercept + lin.slope * x_line
            fig.add_trace(
                go.Scatter(
                    x=x_line,
                    y=y_line,
                    mode="lines",
                    name=trend_labels.get(group, f"{group} trend") if trend_labels else f"{group} trend",
                    line={"color": color_map[group], "width": 3.8},
                    hovertemplate=f"{group} trend<extra></extra>",
                    showlegend=bool(trend_labels),
                )
            )
    if intersections is not None and not intersections.empty and not work.empty:
        y_min = float(work[y_col].min())
        y_max = float(work[y_col].max())
        x_min = float(work[x_col].min())
        x_max = float(work[x_col].max())
        dash_colors = {"CN-MCI": "#facc15", "CN-AD": "#c084fc", "MCI-AD": "#2dd4bf"}
        for _, row in intersections.iterrows():
            age_value = pd.to_numeric(pd.Series([row.get("intersection_age")]), errors="coerce").iloc[0]
            pair = str(row.get("pair", "intersection"))
            p_value = row.get("slope_diff_p")
            if pd.isna(age_value) or age_value < x_min or age_value > x_max:
                fig.add_trace(
                    go.Scatter(
                        x=[None],
                        y=[None],
                        mode="lines",
                        name=f"{pair} p={format_p(p_value)}, x out",
                        line={"color": dash_colors.get(pair, "#94a3b8"), "width": 2, "dash": "dot"},
                        hovertemplate=f"{escape(pair)} slope p={format_p(p_value)}<br>intersection outside plotted age range<extra></extra>",
                    )
                )
                continue
            fig.add_trace(
                go.Scatter(
                    x=[float(age_value), float(age_value)],
                    y=[y_min, y_max],
                    mode="lines",
                    name=f"{pair} x={age_value:.1f}, p={format_p(p_value)}",
                    line={"color": dash_colors.get(pair, "#94a3b8"), "width": 2, "dash": "dot"},
                    hovertemplate=f"{escape(pair)} intersection age={age_value:.2f}<br>slope p={format_p(p_value)}<extra></extra>",
                )
            )
    if identity_line and not work.empty:
        min_axis = float(min(work[x_col].min(), work[y_col].min()))
        max_axis = float(max(work[x_col].max(), work[y_col].max()))
        fig.add_trace(
            go.Scatter(
                x=[min_axis, max_axis],
                y=[min_axis, max_axis],
                mode="lines",
                name="identity",
                line={"color": "#64748b", "width": 2, "dash": "dash"},
                hovertemplate="Predicted = chronological age<extra></extra>",
            )
        )
    fig.update_layout(
        title={"text": f"<b>{escape(title)}</b>", "x": 0.01, "xanchor": "left", "font": {"size": 20, "color": CHART_TEXT}},
        template="plotly_white",
        plot_bgcolor=CHART_BACKGROUND,
        paper_bgcolor=CHART_BACKGROUND,
        height=520,
        margin={"l": 82, "r": 38, "t": 86, "b": 82},
        font={"color": CHART_TEXT, "size": 13},
        showlegend=True,
        legend={
            "title": {"text": "Group", "font": {"color": CHART_TEXT, "size": 12}},
            "orientation": "h",
            "yanchor": "top",
            "y": 0.99,
            "xanchor": "left",
            "x": 0.01,
            "bgcolor": "rgba(248,250,252,0.92)",
            "bordercolor": CHART_STROKE,
            "borderwidth": 1,
            "font": {"color": CHART_TEXT, "size": 13},
        },
        xaxis={
            "title": {"text": x_title, "font": {"color": CHART_TEXT, "size": 15}},
            "gridcolor": CHART_GRID,
            "linecolor": CHART_STROKE,
            "zerolinecolor": CHART_GRID,
            "tickfont": {"color": CHART_TEXT, "size": 12},
            "ticks": "outside",
            "showline": True,
            "mirror": True,
        },
        yaxis={
            "title": {"text": y_title, "font": {"color": CHART_TEXT, "size": 15}},
            "gridcolor": CHART_GRID,
            "linecolor": CHART_STROKE,
            "zerolinecolor": CHART_GRID,
            "tickfont": {"color": CHART_TEXT, "size": 12},
            "ticks": "outside",
            "showline": True,
            "mirror": True,
        },
    )
    return fig


def age_coupling_inference(title: str, table: pd.DataFrame) -> str:
    if table.empty:
        return f"{title}: no age-coupling association table is available."
    group_rows = table[table["Group"].isin(GROUP_ORDER)].copy()
    group_rows["Spearman p"] = pd.to_numeric(group_rows["Spearman p"], errors="coerce")
    pieces = []
    for group in GROUP_ORDER:
        match = group_rows[group_rows["Group"].astype(str) == group]
        if match.empty:
            continue
        row = match.iloc[0]
        rho = pd.to_numeric(pd.Series([row.get("Spearman rho")]), errors="coerce").iloc[0]
        p_val = pd.to_numeric(pd.Series([row.get("Spearman p")]), errors="coerce").iloc[0]
        slope = pd.to_numeric(pd.Series([row.get("Linear slope")]), errors="coerce").iloc[0]
        if pd.isna(rho) or pd.isna(p_val) or pd.isna(slope):
            pieces.append(f"{group}: insufficient age variation")
            continue
        direction = "increases" if slope > 0 else "decreases"
        evidence = "significant" if p_val < 0.05 else "not significant"
        pieces.append(f"{group}: {direction} with age ({evidence}; rho={format_number(rho)}, p={format_p(p_val)})")
    if pieces:
        return f"{title}: " + "; ".join(pieces) + "."
    return f"{title}: no group-specific age association is detected in the current refreshed cohort."


def age_direction_cell(row: pd.Series | None) -> str:
    if row is None:
        return "not available"
    rho = pd.to_numeric(pd.Series([row.get("Spearman rho")]), errors="coerce").iloc[0]
    p_val = pd.to_numeric(pd.Series([row.get("Spearman p")]), errors="coerce").iloc[0]
    slope = pd.to_numeric(pd.Series([row.get("Linear slope")]), errors="coerce").iloc[0]
    if pd.isna(rho) or pd.isna(p_val) or pd.isna(slope):
        return "insufficient data"
    direction = "increases with age" if slope > 0 else "decreases with age"
    evidence = "sig" if p_val < 0.05 else "ns"
    return f"{direction} ({evidence}; rho={format_number(rho)}, p={format_p(p_val)})"


def age_coupling_summary_table(metric_df: pd.DataFrame, available: list[tuple[str, str, str, str]]) -> pd.DataFrame:
    rows = []
    for label, metric, _, _ in available:
        filtered = metric_df[metric_df["metric_key"].astype(str) == metric].copy()
        filtered["rho"] = pd.to_numeric(filtered["rho"], errors="coerce")
        filtered["age"] = pd.to_numeric(filtered["age"], errors="coerce")
        filtered = filtered[filtered["rho"].notna() & filtered["age"].notna()].copy()
        stats_table = age_metric_association_table(filtered, "age", "rho")
        row = {"Metric": f"{label} coupling"}
        for group in GROUP_ORDER:
            match = stats_table[stats_table["Group"].astype(str) == group]
            row[group] = age_direction_cell(match.iloc[0] if not match.empty else None)
        rows.append(row)
    return pd.DataFrame(rows)


def render_age_association_table(table: pd.DataFrame, key: str) -> None:
    if table.empty:
        st.info("No age-association statistics are available.")
        return
    display = table.drop(columns=[c for c in ("Intercept",) if c in table.columns]).copy()
    st.dataframe(
        display,
        width="stretch",
        height=compact_table_height(len(display)),
        hide_index=True,
        column_config={
            "n": st.column_config.NumberColumn("n", format="%d"),
            "Spearman rho": st.column_config.NumberColumn("Spearman rho", format="%.3f"),
            "Spearman p": st.column_config.NumberColumn("Spearman p", format="%.3g"),
            "Pearson r": st.column_config.NumberColumn("Pearson r", format="%.3f"),
            "Pearson p": st.column_config.NumberColumn("Pearson p", format="%.3g"),
            "Linear slope": st.column_config.NumberColumn("Linear slope", format="%.4g"),
        },
        key=key,
    )


def age_analysis_frame(csv_files: list[Path]) -> pd.DataFrame:
    feature_path = next((p for p in csv_files if p.name == "ml_subject_feature_matrix.csv"), None)
    master_path = next((p for p in csv_files if p.name == "master_cohort.csv"), ANALYSIS_ROOT / "00_master" / "master_cohort.csv")
    if feature_path is None:
        return pd.DataFrame()
    features = _safe_read_csv(feature_path)
    master = _safe_read_csv(master_path) if master_path is not None else None
    if features is None or features.empty or "subject_id" not in features.columns:
        return pd.DataFrame()
    work = features.copy()
    if master is not None and not master.empty and "subject_id" in master.columns:
        age_col = "age" if "age" in master.columns else ("Age" if "Age" in master.columns else None)
        join_cols = ["subject_id"]
        if age_col:
            join_cols.append(age_col)
        if "group" in master.columns:
            join_cols.append("group")
        meta = master[join_cols].copy().drop_duplicates(subset=["subject_id"])
        rename = {}
        if age_col:
            rename[age_col] = "age_from_master"
        if "group" in meta.columns:
            rename["group"] = "group_from_master"
        meta = meta.rename(columns=rename)
        work = work.merge(meta, on="subject_id", how="left")
    master_age = pd.to_numeric(work["age_from_master"], errors="coerce") if "age_from_master" in work.columns else pd.Series(np.nan, index=work.index)
    if "age" not in work.columns:
        work["age"] = master_age
    else:
        work["age"] = pd.to_numeric(work["age"], errors="coerce").fillna(master_age)
    master_group = work["group_from_master"] if "group_from_master" in work.columns else pd.Series("", index=work.index)
    if "group" not in work.columns:
        work["group"] = master_group
    else:
        work["group"] = work["group"].fillna(master_group)
    work["age"] = pd.to_numeric(work["age"], errors="coerce")
    work["group"] = work["group"].astype(str)
    return work[work["age"].notna() & work["group"].isin(GROUP_ORDER)].copy()


def age_analysis_metric_label(metric: str) -> str:
    return metric_label(age_analysis_metric_key(metric))


def age_analysis_metric_key(metric: str) -> str:
    label = metric
    for prefix in ("global_dti__", "global_graph__", "lr_sr__", "delay__", "brain_age__"):
        if label.startswith(prefix):
            label = label[len(prefix) :]
            break
    return label


def age_analysis_metric_meaning(metric: str) -> str:
    key = age_analysis_metric_key(metric).lower()
    dti_names = {
        "fa": "FA",
        "md": "MD",
        "ad": "AD",
        "rd": "RD",
    }
    dti_match = re.match(r"^(fa|md|ad|rd)_mean_(edge_mean|edge_median|edge_iqr|n_edges)$", key)
    if dti_match:
        dti, stat = dti_match.groups()
        stat_labels = {
            "edge_mean": "mean across edges",
            "edge_median": "median across edges",
            "edge_iqr": "edge spread IQR",
            "n_edges": "valid-edge count",
        }
        return f"{dti_names[dti]} {stat_labels[stat]}"
    graph_labels = {
        "mean_strength": "average connection strength",
        "total_strength": "total connection strength",
        "density": "nonzero-edge fraction",
        "global_efficiency": "whole-network efficiency",
        "charpath_len": "characteristic path length",
    }
    if key in graph_labels:
        return graph_labels[key]
    brain_age_labels = {
        "brain_age_pred": "predicted structural brain age",
        "bag": "brain age gap",
        "bag_age_corrected": "age-adjusted brain age gap",
    }
    if key in brain_age_labels:
        return brain_age_labels[key]
    range_names = {
        "short": "short-range",
        "medium": "medium-range",
        "long": "long-range",
    }
    for prefix, range_label in range_names.items():
        if key == f"{prefix}_edge_count":
            return f"{range_label} edge count"
        if key == f"{prefix}_len_mean":
            return f"{range_label} mean tract length"
        if key == f"{prefix}_len_median":
            return f"{range_label} median tract length"
        if key == f"{prefix}_w_mean":
            return f"{range_label} mean edge weight"
        if key == f"{prefix}_w_median":
            return f"{range_label} median edge weight"
        if key == f"{prefix}_w_sum":
            return f"{range_label} total edge weight"
        if key == f"{prefix}_density":
            return f"{range_label} nonzero-edge fraction"
        if key == f"{prefix}_global_eff":
            return f"{range_label} network efficiency"
        if key == f"{prefix}_mean_delay_path_ms":
            return f"{range_label} mean delay proxy"
    hemi_labels = {
        "intra_len_median": "within-hemisphere median length",
        "intra_w_mean": "within-hemisphere mean weight",
        "inter_len_median": "between-hemisphere median length",
        "inter_w_mean": "between-hemisphere mean weight",
    }
    if key in hemi_labels:
        return hemi_labels[key]
    sr_lr_labels = {
        "sr_lr_w_diff": "short minus long mean weight",
        "sr_lr_w_ratio": "short/long weight ratio",
        "sr_lr_w_normdiff": "normalized short-long contrast",
    }
    if key in sr_lr_labels:
        return sr_lr_labels[key]
    delay_labels = {
        "velocity_mm_per_ms": "assumed conduction velocity",
        "mean_delay_ms": "mean tract-length delay",
        "median_delay_ms": "median tract-length delay",
        "p90_delay_ms": "90th-percentile delay",
        "delay_burden_ms": "weight-averaged delay burden",
        "long_delay_fraction": "fraction of long-delay edges",
        "delay_weighted_strength": "strength adjusted by delay",
        "delay_path_ms": "shortest-path delay",
    }
    return delay_labels.get(key, "")


def age_analysis_metric_title(metric: str) -> str:
    label = age_analysis_metric_label(metric)
    meaning = age_analysis_metric_meaning(metric)
    return f"{label} ({meaning})" if meaning else label


def age_analysis_metric_catalog(df: pd.DataFrame) -> dict[str, list[str]]:
    if df.empty:
        return {}
    numeric_cols = []
    forbidden = ("mmse", "cdr", "faq", "gdscale", "npi", "adas", "diagnosis", "dx", "clinical", "subject_id", "group")
    for col in df.columns:
        lower = col.lower()
        if col in {"subject_id", "group", "age"} or any(token in lower for token in forbidden):
            continue
        if pd.to_numeric(df[col], errors="coerce").notna().sum() < 10:
            continue
        numeric_cols.append(col)
    catalog = {
        "DTI": [col for col in numeric_cols if col.startswith("global_dti__")],
        "Graph": [col for col in numeric_cols if col.startswith("global_graph__")],
        "LR/SR + Delay": [col for col in numeric_cols if col.startswith(("lr_sr__", "delay__"))],
        "Brain Age": [
            col
            for col in numeric_cols
            if col.startswith("brain_age__") and any(token in col.lower() for token in ("brain_age_pred", "bag", "age_corrected"))
        ],
    }
    return {family: cols for family, cols in catalog.items() if cols}


def age_analysis_core_metrics(metrics: list[str], family: str) -> list[str]:
    if family == "DTI":
        core = [metric for metric in metrics if metric.endswith("_edge_mean")]
    elif family == "LR/SR + Delay":
        keep_tokens = (
            "_w_mean",
            "_density",
            "_global_eff",
            "mean_delay_ms",
            "delay_burden_ms",
            "long_delay_fraction",
            "delay_path_ms",
        )
        core = [metric for metric in metrics if any(token in metric for token in keep_tokens)]
    else:
        core = metrics
    return core or metrics


def render_age_association_cards(stats_table: pd.DataFrame) -> None:
    if stats_table.empty:
        return
    cards = []
    for group in GROUP_ORDER:
        match = stats_table[stats_table["Group"].astype(str).eq(group)]
        if match.empty:
            continue
        row = match.iloc[0]
        cards.append(
            "<div style='border:1px solid rgba(96,165,250,0.38);border-radius:8px;background:#0f172a;padding:0.82rem;'>"
            f"<div style='color:#facc15;font-weight:1000;font-size:1rem;margin-bottom:0.35rem;'>{escape(group)}</div>"
            f"<div style='color:#ffffff;font-weight:900;font-size:1.35rem;'>rho {format_number(row.get('Spearman rho'), digits=3)}</div>"
            f"<div style='color:#cbd5e1;font-size:0.88rem;'>r {format_number(row.get('Pearson r'), digits=3)} | slope {format_number(row.get('Linear slope'), digits=3)}</div>"
            f"<div style='color:#94a3b8;font-size:0.82rem;'>p {format_p(row.get('Spearman p'))} | n {format_number(row.get('n'), digits=0)}</div>"
            "</div>"
        )
    if cards:
        st.markdown(
            "<div style='display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:0.75rem;margin:0.7rem 0 0.9rem 0;'>"
            + "".join(cards)
            + "</div>",
            unsafe_allow_html=True,
        )


def age_iqr_trim_for_metric(df: pd.DataFrame, metric: str) -> tuple[pd.DataFrame, dict[str, int]]:
    work = df[["subject_id", "group", "age", metric]].copy()
    work[metric] = pd.to_numeric(work[metric], errors="coerce")
    work["age"] = pd.to_numeric(work["age"], errors="coerce")
    work = work[work["group"].astype(str).isin(GROUP_ORDER) & work["age"].notna() & work[metric].notna()].copy()
    pieces = []
    removed_counts: dict[str, int] = {}
    for group in GROUP_ORDER:
        group_df = work[work["group"].astype(str).eq(group)].copy()
        total = len(group_df)
        if total >= 4:
            q1 = group_df[metric].quantile(0.25)
            q3 = group_df[metric].quantile(0.75)
            iqr = q3 - q1
            if pd.notna(iqr) and iqr > 0:
                low = q1 - 1.5 * iqr
                high = q3 + 1.5 * iqr
                group_df = group_df[group_df[metric].between(low, high, inclusive="both")].copy()
        removed_counts[group] = total - len(group_df)
        pieces.append(group_df)
    kept = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(columns=work.columns)
    return kept, removed_counts


def age_metric_frame_for_view(df: pd.DataFrame, metric: str, data_view: str) -> tuple[pd.DataFrame, dict[str, int]]:
    work = df[["subject_id", "group", "age", metric]].copy()
    work[metric] = pd.to_numeric(work[metric], errors="coerce")
    work["age"] = pd.to_numeric(work["age"], errors="coerce")
    work = work[work["group"].astype(str).isin(GROUP_ORDER) & work["age"].notna() & work[metric].notna()].copy()
    if data_view == "Outlier removed":
        return age_iqr_trim_for_metric(df, metric)
    return work, {group: 0 for group in GROUP_ORDER}


def age_scaled_metric_frame(df: pd.DataFrame, metric: str, scale: str) -> tuple[pd.DataFrame, str, str]:
    work = df.copy()
    work["_plot_value"] = pd.to_numeric(work[metric], errors="coerce")
    label = age_analysis_metric_label(metric)
    if scale == "Log10 positive":
        work = work[work["_plot_value"] > 0].copy()
        work["_plot_value"] = np.log10(work["_plot_value"])
        return work, f"log10({label})", "log10 positive scale; non-positive values are excluded."
    if scale == "Robust z-score":
        median = work["_plot_value"].median()
        iqr = work["_plot_value"].quantile(0.75) - work["_plot_value"].quantile(0.25)
        if pd.isna(iqr) or iqr <= 0:
            iqr = work["_plot_value"].std()
        if pd.isna(iqr) or iqr <= 0:
            return work, label, "raw scale; robust scaling was not possible for this metric."
        work["_plot_value"] = (work["_plot_value"] - median) / iqr
        return work, f"robust z-score {label}", "robust z-score = (value - cohort median) / cohort IQR."
    if scale == "Percent from median":
        median = work["_plot_value"].median()
        if pd.isna(median) or abs(float(median)) < 1e-12:
            return work, label, "raw scale; percent-from-median was not possible because the median is near zero."
        work["_plot_value"] = 100.0 * ((work["_plot_value"] / median) - 1.0)
        return work, f"% from median {label}", "percent from cohort median; this expands small absolute differences into relative change."
    return work, label, "raw values."


def age_group_line_stats(df: pd.DataFrame, x_col: str, y_col: str) -> pd.DataFrame:
    rows = []
    for group in GROUP_ORDER:
        group_df = df[df["group"].astype(str).eq(group)].copy()
        group_df[x_col] = pd.to_numeric(group_df[x_col], errors="coerce")
        group_df[y_col] = pd.to_numeric(group_df[y_col], errors="coerce")
        group_df = group_df[group_df[x_col].notna() & group_df[y_col].notna()].copy()
        row = {
            "Group": group,
            "n": len(group_df),
            "slope": np.nan,
            "intercept": np.nan,
            "slope_p": np.nan,
            "slope_se": np.nan,
        }
        if len(group_df) >= 3 and group_df[x_col].nunique() >= 2 and group_df[y_col].nunique() >= 2:
            lin = stats.linregress(group_df[x_col].to_numpy(dtype=float), group_df[y_col].to_numpy(dtype=float))
            row.update(
                {
                    "slope": float(lin.slope),
                    "intercept": float(lin.intercept),
                    "slope_p": float(lin.pvalue),
                    "slope_se": float(lin.stderr) if lin.stderr is not None else np.nan,
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def age_pairwise_line_comparisons(line_stats: pd.DataFrame, age_min: float, age_max: float) -> pd.DataFrame:
    rows = []
    for group_a, group_b in (("CN", "MCI"), ("CN", "AD"), ("MCI", "AD")):
        a = line_stats[line_stats["Group"].astype(str).eq(group_a)]
        b = line_stats[line_stats["Group"].astype(str).eq(group_b)]
        if a.empty or b.empty:
            continue
        a_row = a.iloc[0]
        b_row = b.iloc[0]
        slope_a = pd.to_numeric(pd.Series([a_row.get("slope")]), errors="coerce").iloc[0]
        slope_b = pd.to_numeric(pd.Series([b_row.get("slope")]), errors="coerce").iloc[0]
        intercept_a = pd.to_numeric(pd.Series([a_row.get("intercept")]), errors="coerce").iloc[0]
        intercept_b = pd.to_numeric(pd.Series([b_row.get("intercept")]), errors="coerce").iloc[0]
        se_a = pd.to_numeric(pd.Series([a_row.get("slope_se")]), errors="coerce").iloc[0]
        se_b = pd.to_numeric(pd.Series([b_row.get("slope_se")]), errors="coerce").iloc[0]
        n_a = pd.to_numeric(pd.Series([a_row.get("n")]), errors="coerce").iloc[0]
        n_b = pd.to_numeric(pd.Series([b_row.get("n")]), errors="coerce").iloc[0]
        slope_diff = slope_a - slope_b if pd.notna(slope_a) and pd.notna(slope_b) else np.nan
        p_value = np.nan
        if pd.notna(slope_diff) and pd.notna(se_a) and pd.notna(se_b):
            se_diff = math.sqrt(float(se_a) ** 2 + float(se_b) ** 2)
            df = int(n_a + n_b - 4) if pd.notna(n_a) and pd.notna(n_b) else 0
            if se_diff > 0 and df > 0:
                t_stat = float(slope_diff) / se_diff
                p_value = float(2.0 * stats.t.sf(abs(t_stat), df))
        intersection_age = np.nan
        intersection_y = np.nan
        if pd.notna(slope_diff) and abs(float(slope_diff)) > 1e-12 and pd.notna(intercept_a) and pd.notna(intercept_b):
            intersection_age = float((intercept_b - intercept_a) / slope_diff)
            intersection_y = float(intercept_a + slope_a * intersection_age)
        rows.append(
            {
                "pair": f"{group_a}-{group_b}",
                "slope_diff": slope_diff,
                "slope_diff_p": p_value,
                "intersection_age": intersection_age,
                "intersection_y": intersection_y,
                "intersection_in_age_range": bool(pd.notna(intersection_age) and age_min <= float(intersection_age) <= age_max),
            }
        )
    return pd.DataFrame(rows)


def age_association_table_html(
    stats_table: pd.DataFrame,
    removed_counts: dict[str, int],
    line_stats: pd.DataFrame,
    pairwise: pd.DataFrame,
) -> str:
    rows = []
    for group in GROUP_ORDER:
        match = stats_table[stats_table["Group"].astype(str).eq(group)] if not stats_table.empty else pd.DataFrame()
        row = match.iloc[0] if not match.empty else {}
        line_match = line_stats[line_stats["Group"].astype(str).eq(group)] if line_stats is not None and not line_stats.empty else pd.DataFrame()
        line_row = line_match.iloc[0] if not line_match.empty else {}
        rows.append(
            "<tr>"
            f"<td style='color:#e5e7eb;padding:0.26rem;border-bottom:1px solid rgba(51,65,85,0.55);'>{escape(group)}</td>"
            f"<td style='color:#e5e7eb;padding:0.26rem;border-bottom:1px solid rgba(51,65,85,0.55);'>{format_number(row.get('n'), digits=0) if hasattr(row, 'get') else 'NA'}</td>"
            f"<td style='color:#e5e7eb;padding:0.26rem;border-bottom:1px solid rgba(51,65,85,0.55);'>{removed_counts.get(group, 0)}</td>"
            f"<td style='color:#e5e7eb;padding:0.26rem;border-bottom:1px solid rgba(51,65,85,0.55);'>{format_number(row.get('Spearman rho'), digits=3) if hasattr(row, 'get') else 'NA'}</td>"
            f"<td style='color:#e5e7eb;padding:0.26rem;border-bottom:1px solid rgba(51,65,85,0.55);'>{format_p(row.get('Spearman p')) if hasattr(row, 'get') else 'NA'}</td>"
            f"<td style='color:#e5e7eb;padding:0.26rem;border-bottom:1px solid rgba(51,65,85,0.55);'>{format_number(row.get('Pearson r'), digits=3) if hasattr(row, 'get') else 'NA'}</td>"
            f"<td style='color:#e5e7eb;padding:0.26rem;border-bottom:1px solid rgba(51,65,85,0.55);'>{format_number(row.get('Linear slope'), digits=3) if hasattr(row, 'get') else 'NA'}</td>"
            f"<td style='color:#e5e7eb;padding:0.26rem;border-bottom:1px solid rgba(51,65,85,0.55);'>{format_p(line_row.get('slope_p')) if hasattr(line_row, 'get') else 'NA'}</td>"
            "</tr>"
        )
    header_style = "color:#facc15;text-align:left;padding:0.26rem;border-bottom:1px solid rgba(51,65,85,0.8);"
    pair_rows = []
    if pairwise is not None and not pairwise.empty:
        for _, row in pairwise.iterrows():
            in_range = "yes" if bool(row.get("intersection_in_age_range")) else "no"
            pair_rows.append(
                "<tr>"
                f"<td style='color:#e5e7eb;padding:0.26rem;border-bottom:1px solid rgba(51,65,85,0.55);'>{escape(str(row.get('pair', '')))}</td>"
                f"<td style='color:#e5e7eb;padding:0.26rem;border-bottom:1px solid rgba(51,65,85,0.55);'>{format_number(row.get('slope_diff'), digits=3)}</td>"
                f"<td style='color:#e5e7eb;padding:0.26rem;border-bottom:1px solid rgba(51,65,85,0.55);'>{format_p(row.get('slope_diff_p'))}</td>"
                f"<td style='color:#e5e7eb;padding:0.26rem;border-bottom:1px solid rgba(51,65,85,0.55);'>{format_number(row.get('intersection_age'), digits=2)}</td>"
                f"<td style='color:#e5e7eb;padding:0.26rem;border-bottom:1px solid rgba(51,65,85,0.55);'>{in_range}</td>"
                "</tr>"
            )
    pair_table = (
        "<div style='color:#facc15;font-weight:950;margin:0.55rem 0 0.2rem 0;'>Pairwise trend-line comparison</div>"
        "<table style='width:100%;border-collapse:collapse;background:transparent;font-size:0.78rem;margin-top:0.1rem;'>"
        "<thead><tr>"
        f"<th style='{header_style}'>Pair</th>"
        f"<th style='{header_style}'>slope diff</th>"
        f"<th style='{header_style}'>slope p</th>"
        f"<th style='{header_style}'>intersect age</th>"
        f"<th style='{header_style}'>in range</th>"
        "</tr></thead><tbody>"
        + "".join(pair_rows)
        + "</tbody></table>"
        if pair_rows
        else ""
    )
    return (
        "<div style='color:#facc15;font-weight:950;margin:0.35rem 0 0.2rem 0;'>Group age trend</div>"
        "<table style='width:100%;border-collapse:collapse;background:transparent;font-size:0.78rem;margin-top:0.35rem;'>"
        "<thead><tr>"
        f"<th style='{header_style}'>Group</th>"
        f"<th style='{header_style}'>n</th>"
        f"<th style='{header_style}'>out</th>"
        f"<th style='{header_style}'>rho</th>"
        f"<th style='{header_style}'>p</th>"
        f"<th style='{header_style}'>r</th>"
        f"<th style='{header_style}'>slope</th>"
        f"<th style='{header_style}'>slope p</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
        + pair_table
    )


def render_age_metric_card(age_df: pd.DataFrame, metric: str, scale: str, data_view: str) -> None:
    title_label = age_analysis_metric_title(metric)
    kept, removed_counts = age_metric_frame_for_view(age_df, metric, data_view)
    if kept.empty:
        st.info(f"No non-outlier values are available for {age_analysis_metric_label(metric)}.")
        return
    scaled, y_title, scale_note = age_scaled_metric_frame(kept, metric, scale)
    scaled = scaled[scaled["_plot_value"].notna()].copy()
    if scaled.empty:
        st.info(f"No plottable values are available for {age_analysis_metric_label(metric)} after applying {scale}.")
        return
    plot_source = scaled.rename(columns={"_plot_value": "plot_value"})
    stats_table = age_metric_association_table(plot_source, "age", "plot_value")
    line_stats = age_group_line_stats(plot_source, "age", "plot_value")
    age_min = float(plot_source["age"].min())
    age_max = float(plot_source["age"].max())
    pairwise = age_pairwise_line_comparisons(line_stats, age_min, age_max)
    trend_labels = {
        str(row["Group"]): f"{row['Group']} trend p={format_p(row.get('slope_p'))}"
        for _, row in line_stats.iterrows()
        if str(row.get("Group")) in GROUP_ORDER
    }
    fig = scatter_with_group_trends(
        plot_source,
        "age",
        "plot_value",
        f"Age vs {title_label}",
        "Age",
        y_title,
        trend_labels=trend_labels,
        intersections=pairwise,
    )
    fig.update_layout(
        height=470,
        margin={"l": 58, "r": 22, "t": 68, "b": 145},
        legend={
            "title": {"text": ""},
            "orientation": "h",
            "yanchor": "top",
            "y": -0.23,
            "xanchor": "left",
            "x": 0,
            "bgcolor": "rgba(248,250,252,0.95)",
            "bordercolor": CHART_STROKE,
            "borderwidth": 1,
            "font": {"color": CHART_TEXT, "size": 10},
            "tracegroupgap": 6,
        },
    )
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False})
    view_note = (
        f"Group-wise 1.5x IQR outlier trim applied before plotting/stats; removed {sum(removed_counts.values())} rows."
        if data_view == "Outlier removed"
        else "Normal/raw view: no IQR outlier removal is applied; rows are excluded only when age or metric is missing."
    )
    st.caption(
        f"{view_note} Scale: {scale_note} Pairwise p-values test whether two group trend slopes differ; intersection ages mark where fitted lines cross."
    )
    st.markdown(age_association_table_html(stats_table, removed_counts, line_stats, pairwise), unsafe_allow_html=True)


def render_age_analysis_panel(csv_files: list[Path]) -> bool:
    age_df = age_analysis_frame(csv_files)
    if age_df.empty:
        st.info("Age analysis data are not available. The ML feature matrix and master cohort age table are required.")
        return False
    catalog = age_analysis_metric_catalog(age_df)
    if not catalog:
        st.info("No eligible structural metrics are available for age analysis.")
        return False

    render_yellow_center_heading("AGE ANALYSIS")
    st.caption(
        "Descriptive age association only: age is plotted against structural metrics for CN/MCI/AD and is not used as a predictor in clinical outcome models."
    )
    family_options = [family for family in ("DTI", "Graph", "LR/SR + Delay", "Brain Age") if family in catalog]
    controls_left, metric_set_col, data_view_col, controls_right = st.columns((0.42, 0.18, 0.20, 0.20), gap="large")
    with controls_left:
        selected_family = render_button_selector(
            family_options,
            "age-analysis-family",
            default=family_options[0],
            max_cols=min(len(family_options), 4),
        )
    with metric_set_col:
        metric_set = st.selectbox(
            "Metric set",
            ["Core summaries", "All summaries"],
            index=0,
            help="Core summaries reduce repeated mean/median/IQR variants. All summaries shows every eligible metric in the family.",
            key="age-analysis-metric-set",
        )
    with data_view_col:
        data_view = st.selectbox(
            "Data view",
            ["Outlier removed", "Normal/raw"],
            index=0,
            help="Outlier removed uses group-wise 1.5x IQR trimming for plotting and displayed stats. Normal/raw keeps all non-missing values.",
            key="age-analysis-data-view",
        )
    with controls_right:
        scale = st.selectbox(
            "Y-axis display scale",
            ["Raw", "Log10 positive", "Robust z-score", "Percent from median"],
            index=2,
            help=(
                "Raw keeps the original units. Log10 helps positive skewed metrics. "
                "Robust z-score expands small differences using the cohort IQR. "
                "Percent from median shows relative change from the cohort median."
            ),
            key="age-analysis-y-scale",
        )
    all_metric_options = catalog.get(selected_family, [])
    metric_options = age_analysis_core_metrics(all_metric_options, selected_family) if metric_set == "Core summaries" else all_metric_options
    st.caption(
        f"Showing {len(metric_options)} of {len(all_metric_options)} {selected_family} metrics, two plots per row. "
        f"Data view: {data_view}. Trend-line p-values and pairwise line intersections are computed from the displayed view."
    )
    for start in range(0, len(metric_options), 2):
        cols = st.columns(2, gap="large")
        for col, metric in zip(cols, metric_options[start : start + 2]):
            with col:
                with st.container(border=True):
                    st.markdown(f"#### {age_analysis_metric_title(metric)}")
                    render_age_metric_card(age_df, metric, scale, data_view)
    return True


def render_subject_metric_charts(
    section: Section,
    csv_files: list[Path],
    max_charts: int,
    hide_visual_outliers: bool,
) -> bool:
    subject_path = subject_table_for_section(section, csv_files)
    if subject_path is None:
        return False
    df = _safe_read_csv(subject_path)
    if df is None or df.empty or "group" not in df.columns:
        return False

    metric_df = df.copy()
    metric_options: list[str]
    metric_key_to_filter: dict[str, pd.Series] = {}
    value_col_by_metric: dict[str, str] = {}
    if section.label == "Coupling" and {"nodal_metric", "micro_metric", "rho"}.issubset(metric_df.columns):
        metric_df["metric_key"] = metric_df["nodal_metric"].astype(str) + "_" + metric_df["micro_metric"].astype(str) + "_rho"
        metric_options = list(metric_df["metric_key"].dropna().astype(str).unique())
        preferred = [m for m in PREFERRED_METRICS_BY_SECTION.get(section.label, ()) if m in metric_options]
        metric_options = preferred + [m for m in sorted(metric_options) if m not in preferred]
        for metric in metric_options:
            metric_key_to_filter[metric] = metric_df["metric_key"] == metric
            value_col_by_metric[metric] = "rho"
    else:
        metric_options = numeric_metric_columns(metric_df, section.label)
        for metric in metric_options:
            metric_key_to_filter[metric] = pd.Series(True, index=metric_df.index)
            value_col_by_metric[metric] = metric

    if not metric_options:
        return False

    if section.label == "Coupling":
        render_coupling_explainer()

    st.subheader("Live CSV Charts")
    default_metrics = metric_options[:max_charts]
    selected = st.multiselect(
        "Choose live charts",
        metric_options,
        default=default_metrics,
        format_func=metric_label,
        key=f"{section_key(section.label)}-live-metrics-{hash(tuple(metric_options))}",
    )
    for metric in selected:
        filtered = metric_df[metric_key_to_filter[metric]].copy()
        value_col = value_col_by_metric[metric]
        data = subject_metric_frame(filtered, metric, value_col=value_col)
        if data.empty:
            continue
        desc = metric_descriptives(csv_files, metric, data=data)
        pairwise = metric_pairwise(csv_files, metric)
        chart, counts, plotted_counts, removed_counts = live_distribution_chart(
            data,
            metric_label(metric),
            hide_visual_outliers=hide_visual_outliers,
            stats_df=desc,
            subtitle_lines=chart_stat_subtitles(csv_files, metric),
        )
        left, right = st.columns((1.65, 1), gap="large")
        with left:
            st.plotly_chart(chart, width="stretch", config={"displaylogo": False})
            st.caption(f"Live chart from `{subject_path.relative_to(ANALYSIS_ROOT).as_posix()}`.")
            if sum(removed_counts.values()):
                st.caption(
                    "Visual-only 1.5-IQR trim active; statistics and CSV tables remain unchanged. "
                    f"Hidden points: {group_count_text(removed_counts)}."
                )
            render_dti_formula_block(metric)
        with right:
            st.markdown(
                result_card_html(section, metric, desc, pairwise),
                unsafe_allow_html=True,
            )
    return True


def render_edr_subject_metric_tabs(section: Section, csv_files: list[Path], hide_visual_outliers: bool) -> bool:
    subject_path = subject_table_for_section(section, csv_files)
    if subject_path is None:
        return False
    df = _safe_read_csv(subject_path)
    if df is None or df.empty or "group" not in df.columns:
        return False

    available_metrics = set(numeric_metric_columns(df, section.label))
    metric_groups = [
        (label, [metric for metric in metrics if metric in available_metrics])
        for label, metrics in EDR_SUBJECT_METRIC_GROUPS
    ]
    metric_groups = [(label, metrics) for label, metrics in metric_groups if metrics]
    if not metric_groups:
        return False

    st.subheader("EDR Exceptions")
    tabs = st.tabs([label for label, _ in metric_groups])
    for tab, (group_label, metrics) in zip(tabs, metric_groups):
        with tab:
            st.caption(f"{group_label} metrics are rendered live from `{subject_path.relative_to(ANALYSIS_ROOT).as_posix()}`.")
            for metric in metrics:
                data = subject_metric_frame(df, metric, value_col=metric)
                if data.empty:
                    continue
                desc = metric_descriptives(csv_files, metric, data=data)
                pairwise = metric_pairwise(csv_files, metric)
                chart, _, _, removed_counts = live_distribution_chart(
                    data,
                    metric_label(metric),
                    hide_visual_outliers=hide_visual_outliers,
                    stats_df=desc,
                    subtitle_lines=chart_stat_subtitles(csv_files, metric),
                )
                st.markdown(f"### {metric_label(metric)}")
                left, right = st.columns((1.25, 1), gap="large")
                with left:
                    st.plotly_chart(chart, width="stretch", config={"displaylogo": False})
                    if sum(removed_counts.values()):
                        st.caption(
                            "Visual-only 1.5-IQR trim active; statistics and CSV tables remain unchanged. "
                            f"Hidden points: {group_count_text(removed_counts)}."
                        )
                with right:
                    render_metric_definition_inference_table(
                        section,
                        metric,
                        desc,
                        pairwise,
                        key=f"edr-stats-{section_key(group_label)}-{metric}",
                    )
    return True


def render_lr_sr_layout(section: Section, csv_files: list[Path], hide_visual_outliers: bool) -> bool:
    render_lr_sr_definition_panel()
    subject_path = subject_table_for_section(section, csv_files)
    if subject_path is None:
        st.info("LR/SR subject-level table is not available yet.")
        return True
    df = _safe_read_csv(subject_path)
    if df is None or df.empty or "group" not in df.columns:
        st.info("LR/SR subject-level table is empty or missing group labels.")
        return True

    available_metrics = set(numeric_metric_columns(df, section.label))
    metric_groups = [
        (label, [metric for metric in metrics if metric in available_metrics])
        for label, metrics in LR_SR_SUBJECT_METRIC_GROUPS
    ]
    metric_groups = [(label, metrics) for label, metrics in metric_groups if metrics]
    if not metric_groups:
        st.info("No numeric LR/SR metrics are available for plotting yet.")
        return True

    render_lr_sr_tract_length_distribution_panel()
    st.subheader("LR / SR Metrics")
    st.caption(f"Metrics are rendered live from `{subject_path.relative_to(ANALYSIS_ROOT).as_posix()}`.")
    tabs = st.tabs([label for label, _ in metric_groups])
    for tab, (group_label, metrics) in zip(tabs, metric_groups):
        with tab:
            st.markdown(lr_sr_metric_group_table_html(section, metrics), unsafe_allow_html=True)
            for metric in metrics:
                data = subject_metric_frame(df, metric, value_col=metric)
                if data.empty:
                    continue
                desc = metric_descriptives(csv_files, metric, data=data)
                pairwise = metric_pairwise(csv_files, metric)
                chart, _, _, removed_counts = live_distribution_chart(
                    data,
                    metric_label(metric),
                    hide_visual_outliers=hide_visual_outliers,
                    stats_df=desc,
                    subtitle_lines=chart_stat_subtitles(csv_files, metric),
                )
                st.markdown(f"### {metric_label(metric)}")
                left, right = st.columns((1.25, 1), gap="large")
                with left:
                    st.plotly_chart(chart, width="stretch", config={"displaylogo": False})
                    if sum(removed_counts.values()):
                        st.caption(
                            "Visual-only 1.5-IQR trim active; statistics and CSV tables remain unchanged. "
                            f"Hidden points: {group_count_text(removed_counts)}."
                        )
                with right:
                    render_metric_definition_inference_table(
                        section,
                        metric,
                        desc,
                        pairwise,
                        key=f"lr-sr-stats-{section_key(group_label)}-{metric}",
                    )
    return True


def global_dti_metric_specs() -> list[tuple[str, str, str]]:
    return [
        ("FA", "fa_mean_edge_mean", "fa_mean_edge_median"),
        ("MD", "md_mean_edge_mean", "md_mean_edge_median"),
        ("RD", "rd_mean_edge_mean", "rd_mean_edge_median"),
        ("AD", "ad_mean_edge_mean", "ad_mean_edge_median"),
    ]


def render_global_dti_stats_table(table: pd.DataFrame, key: str) -> None:
    if table.empty:
        st.info("No statistical table is available for this metric.")
        return
    st.dataframe(
        table,
        width="stretch",
        height=285,
        hide_index=True,
        column_config={
            "row": "Group / contrast",
            "n": "n",
            "mean": st.column_config.NumberColumn("mean", format="%.4g"),
            "median": st.column_config.NumberColumn("median", format="%.4g"),
            "q1": st.column_config.NumberColumn("Q1", format="%.4g"),
            "q3": st.column_config.NumberColumn("Q3", format="%.4g"),
            "direction": "Median direction",
            "p": st.column_config.NumberColumn("p", format="%.3g"),
            "BH q": st.column_config.NumberColumn("BH q", format="%.3g"),
            "delta": st.column_config.NumberColumn("Cliff's delta", format="%.3f"),
        },
        key=key,
    )


def render_global_dti_layout(section: Section, csv_files: list[Path], hide_visual_outliers: bool) -> bool:
    subject_path = subject_table_for_section(section, csv_files)
    if subject_path is None:
        return False
    df = _safe_read_csv(subject_path)
    if df is None or df.empty or "group" not in df.columns:
        return False

    st.subheader("Global DTI")
    st.caption(
        "Each DTI family is arranged as edge mean and edge median side by side. "
        "The paired statistical tables below each chart pair use the same refreshed CSV/statistical outputs."
    )
    available_cols = set(df.columns)
    for family, mean_metric, median_metric in global_dti_metric_specs():
        st.markdown(f"### {family}")
        metric_pair = [("Edge mean", mean_metric), ("Edge median", median_metric)]
        present = [(label, metric) for label, metric in metric_pair if metric in available_cols]
        if not present:
            st.info(
                f"{family} edge mean/median columns are not present in the current live Global DTI subject table. "
                "This section will populate once matching edge-sampled matrices/statistical tables are generated."
            )
            continue

        graph_cols = st.columns(2, gap="large")
        table_payload: list[tuple[str, str, pd.DataFrame]] = []
        for idx, (summary_label, metric) in enumerate(metric_pair):
            with graph_cols[idx]:
                if metric not in available_cols:
                    st.info(f"{summary_label} is not available for {family}.")
                    table_payload.append((summary_label, metric, pd.DataFrame()))
                    continue
                data = subject_metric_frame(df, metric, value_col=metric)
                desc = metric_descriptives(csv_files, metric, data=data)
                pairwise = metric_pairwise(csv_files, metric)
                chart, counts, plotted_counts, removed_counts = live_distribution_chart(
                    data,
                    f"{family} {summary_label.lower()}",
                    hide_visual_outliers=hide_visual_outliers,
                    stats_df=desc,
                    subtitle_lines=chart_stat_subtitles(csv_files, metric),
                )
                st.plotly_chart(chart, width="stretch", config={"displaylogo": False})
                if sum(removed_counts.values()):
                    st.caption(
                        "Visual-only 1.5-IQR trim active; statistics and CSV tables remain unchanged. "
                        f"Hidden points: {group_count_text(removed_counts)}."
                    )
                table_payload.append((summary_label, metric, metric_stats_display_table(desc, pairwise)))

        table_cols = st.columns(2, gap="large")
        for idx, (summary_label, metric, table) in enumerate(table_payload):
            with table_cols[idx]:
                st.markdown(f"##### {family} {summary_label} Table")
                render_global_dti_stats_table(table, key=f"global-dti-table-{metric}")

    st.caption(f"Live Global DTI source: `{subject_path.relative_to(ANALYSIS_ROOT).as_posix()}`.")
    render_dti_formula_block("fa_md_rd_edge_mean_edge_median")
    return True


def coupling_metric_key(nodal_metric: str, micro_metric: str) -> str:
    return f"{nodal_metric}_{micro_metric}_rho"


def coupling_chart_title(nodal_metric: str, micro_metric: str) -> str:
    return f"{metric_label(nodal_metric)} vs {metric_label(micro_metric)} coupling"


def render_single_coupling_chart(
    filtered: pd.DataFrame,
    metric: str,
    title: str,
    csv_files: list[Path],
    subject_path: Path,
    hide_visual_outliers: bool,
) -> None:
    data = subject_metric_frame(filtered, metric, value_col="rho")
    if data.empty:
        st.info(f"No live coupling values are available for {title}.")
        return
    desc = metric_descriptives(csv_files, metric, data=data)
    pairwise = metric_pairwise(csv_files, metric)
    chart, counts, plotted_counts, removed_counts = live_distribution_chart(
        data,
        title,
        hide_visual_outliers=hide_visual_outliers,
        stats_df=desc,
        subtitle_lines=chart_stat_subtitles(csv_files, metric),
    )
    st.plotly_chart(chart, width="stretch", config={"displaylogo": False})
    if sum(removed_counts.values()):
        st.caption(
            "Visual-only 1.5-IQR trim active; statistics and CSV tables remain unchanged. "
            f"Hidden points: {group_count_text(removed_counts)}."
        )
    st.markdown(
        "<div class='chart-finding'>"
        f"{escape(one_line_finding(Section('Coupling', '', ()), metric, desc, pairwise))}"
        "</div>",
        unsafe_allow_html=True,
    )
    st.caption(f"Live chart from `{subject_path.relative_to(ANALYSIS_ROOT).as_posix()}`.")
    render_global_dti_stats_table(
        metric_stats_display_table(desc, pairwise),
        key=f"coupling-table-{metric}",
    )


def render_coupling_age_panel(metric_df: pd.DataFrame, subject_path: Path) -> None:
    if "age" not in metric_df.columns:
        st.info("Age is not available in the live coupling table, so age-coupling scatter plots cannot be rendered.")
        return
    specs = (
        ("Strength-FA", "strength", "fa_mean"),
        ("Strength-MD", "strength", "md_mean"),
        ("Degree-FA", "degree", "fa_mean"),
        ("Degree-MD", "degree", "md_mean"),
        ("Nodal efficiency-FA", "nodal_eff", "fa_mean"),
        ("Nodal efficiency-MD", "nodal_eff", "md_mean"),
    )
    available = []
    metric_keys = set(metric_df["metric_key"].dropna().astype(str)) if "metric_key" in metric_df.columns else set()
    for label, nodal_metric, micro_metric in specs:
        metric = coupling_metric_key(nodal_metric, micro_metric)
        if metric in metric_keys:
            available.append((label, metric, nodal_metric, micro_metric))
    if not available:
        st.info("No main age-coupling pairs are available in the current live coupling CSV.")
        return

    st.subheader("Age vs Coupling")
    st.caption(f"Scatter plots are rendered live from `{subject_path.relative_to(ANALYSIS_ROOT).as_posix()}`.")
    tabs = st.tabs([label for label, _, _, _ in available])
    for tab, (label, metric, nodal_metric, micro_metric) in zip(tabs, available):
        with tab:
            filtered = metric_df[metric_df["metric_key"].astype(str) == metric].copy()
            filtered["rho"] = pd.to_numeric(filtered["rho"], errors="coerce")
            filtered["age"] = pd.to_numeric(filtered["age"], errors="coerce")
            filtered = filtered[filtered["rho"].notna() & filtered["age"].notna()].copy()
            if filtered.empty:
                st.info(f"No age/rho values are available for {label}.")
                continue
            title = f"Age vs {metric_label(nodal_metric)}-{metric_label(micro_metric)} coupling"
            stats_table = age_metric_association_table(filtered, "age", "rho")
            left, right = st.columns((1.35, 1), gap="large")
            with left:
                st.plotly_chart(
                    scatter_with_group_trends(filtered, "age", "rho", title, "Age", "Spearman rho"),
                    width="stretch",
                    config={"displaylogo": False},
                )
                st.caption(
                    f"No outlier trimming is applied to this scatter. Plotted n={len(filtered):,}; "
                    "rows are excluded only when age or rho is missing."
                )
            with right:
                st.markdown(
                    "<div class='edr-definition-box'>"
                    f"<h5>{escape(title)}</h5>"
                    "<p>Each point is one subject. Rho is the within-subject Spearman correlation between the selected node-wise "
                    "structural topology map and the selected node-wise DTI microstructure map over matched AAL3 regions; the lines are group-specific linear age trends.</p>"
                    "<p>No visual outlier removal is applied in this age scatter; missing age/rho rows only are omitted.</p>"
                    "</div>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"<div class='edr-inference-box'>{escape(age_coupling_inference(title, stats_table))}</div>",
                    unsafe_allow_html=True,
                )
                render_age_association_table(stats_table, key=f"age-coupling-table-{metric}")
    summary = age_coupling_summary_table(metric_df, available)
    if not summary.empty:
        st.markdown("#### Age-coupling Summary")
        st.caption(
            "Direction is based on the group-specific linear age slope; rho/p are Spearman age associations from the same live CSV."
        )
        st.dataframe(
            summary,
            width="stretch",
            hide_index=True,
            height=compact_table_height(len(summary)),
        )


def render_coupling_layout(section: Section, csv_files: list[Path], hide_visual_outliers: bool) -> bool:
    subject_path = subject_table_for_section(section, csv_files)
    if subject_path is None:
        return False
    df = _safe_read_csv(subject_path)
    if df is None or df.empty or not {"group", "nodal_metric", "micro_metric", "rho"}.issubset(df.columns):
        return False

    render_coupling_explainer()
    metric_df = df.copy()
    metric_df["metric_key"] = metric_df["nodal_metric"].astype(str) + "_" + metric_df["micro_metric"].astype(str) + "_rho"
    available_keys = set(metric_df["metric_key"].dropna().astype(str))
    render_coupling_age_panel(metric_df, subject_path)
    for nodal_metric, nodal_label in COUPLING_NODAL_METRICS:
        st.markdown(f"### {nodal_label}")
        cols = st.columns(2, gap="large")
        for idx, (micro_metric, micro_label) in enumerate(COUPLING_MICRO_METRICS):
            metric = coupling_metric_key(nodal_metric, micro_metric)
            title = f"{nodal_label} vs {micro_label} coupling"
            with cols[idx]:
                st.markdown(f"#### {title}")
                if metric not in available_keys:
                    st.info(f"{title} is not present in the current coupling CSV outputs.")
                    continue
                filtered = metric_df[metric_df["metric_key"].astype(str) == metric].copy()
                render_single_coupling_chart(filtered, metric, title, csv_files, subject_path, hide_visual_outliers)
    return True


def brain_age_prediction_stats(df: pd.DataFrame) -> pd.DataFrame:
    needed = ["group", "age", "brain_age_pred"]
    optional = [col for col in ("BAG", "BAG_age_corrected") if col in df.columns]
    work = df[needed + optional].copy()
    work["group"] = work["group"].astype(str)
    for col in ["age", "brain_age_pred", *optional]:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work = work[work["group"].isin(GROUP_ORDER) & work["age"].notna() & work["brain_age_pred"].notna()].copy()
    rows: list[dict] = []
    groups: list[tuple[str, pd.DataFrame]] = [("Overall", work)]
    groups.extend((group, work[work["group"] == group].copy()) for group in GROUP_ORDER)
    for group, group_df in groups:
        row: dict = {"Group": group, "n": len(group_df)}
        if not group_df.empty:
            error = group_df["brain_age_pred"] - group_df["age"]
            row.update(
                {
                    "Median age": group_df["age"].median(),
                    "Median predicted age": group_df["brain_age_pred"].median(),
                    "MAE": error.abs().mean(),
                    "RMSE": math.sqrt(float(np.mean(np.square(error)))),
                    "Median BAG": group_df["BAG"].median() if "BAG" in group_df.columns else error.median(),
                    "Median age-corrected BAG": group_df["BAG_age_corrected"].median() if "BAG_age_corrected" in group_df.columns else np.nan,
                }
            )
        if len(group_df) >= 3 and group_df["age"].nunique() >= 2 and group_df["brain_age_pred"].nunique() >= 2:
            pearson_r, pearson_p = stats.pearsonr(group_df["age"], group_df["brain_age_pred"])
            row["Pearson r"] = float(pearson_r)
            row["Pearson p"] = float(pearson_p)
        else:
            row["Pearson r"] = np.nan
            row["Pearson p"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def brain_age_prediction_inference(stats_table: pd.DataFrame) -> str:
    if stats_table.empty:
        return "Brain-age prediction statistics are not available in the current refreshed CSV."
    overall = stats_table[stats_table["Group"].astype(str) == "Overall"]
    overall_text = ""
    if not overall.empty:
        row = overall.iloc[0]
        overall_text = (
            f"overall r={format_number(row.get('Pearson r'))}, "
            f"MAE={format_number(row.get('MAE'))} years"
        )
    group_rows = stats_table[stats_table["Group"].isin(GROUP_ORDER)].copy()
    if "Median BAG" in group_rows.columns and not group_rows.empty:
        group_rows["Median BAG"] = pd.to_numeric(group_rows["Median BAG"], errors="coerce")
        if group_rows["Median BAG"].notna().any():
            top = group_rows.sort_values("Median BAG", ascending=False).iloc[0]
            return (
                f"Predicted structural brain age tracks chronological age ({overall_text}); "
                f"{top['Group']} has the highest median BAG ({format_number(top.get('Median BAG'))} years)."
            )
    return f"Predicted structural brain age tracks chronological age ({overall_text})."


def brain_age_aux_csv(subject_path: Path, name: str) -> pd.DataFrame:
    path = subject_path.parent / name
    df = _safe_read_csv(path)
    return df if df is not None else pd.DataFrame()


def brain_age_model_performance(subject_path: Path, fallback: pd.DataFrame) -> pd.DataFrame:
    perf = brain_age_aux_csv(subject_path, "live_brain_age_model_performance.csv")
    if not perf.empty:
        rename = {
            "group": "Group",
            "r2": "R²",
            "mae_years": "MAE years",
            "rmse_years": "RMSE years",
            "median_bag_years": "Median BAG",
            "mean_bag_years": "Mean BAG",
            "pearson_r": "Pearson r",
            "pearson_p": "Pearson p",
            "spearman_rho": "Spearman rho",
            "spearman_p": "Spearman p",
            "pred_min": "Pred min",
            "pred_max": "Pred max",
            "age_min": "Age min",
            "age_max": "Age max",
        }
        perf = perf.rename(columns=rename)
        preferred = [
            "Group",
            "n",
            "R²",
            "MAE years",
            "RMSE years",
            "Median BAG",
            "Mean BAG",
            "Pearson r",
            "Pearson p",
            "Spearman rho",
            "Spearman p",
            "Age min",
            "Age max",
            "Pred min",
            "Pred max",
        ]
        return perf[[col for col in preferred if col in perf.columns]].copy()
    return fallback.copy()


def brain_age_feature_catalog(subject_path: Path) -> pd.DataFrame:
    catalog = brain_age_aux_csv(subject_path, "live_brain_age_feature_catalog.csv")
    if catalog.empty:
        return pd.DataFrame()
    rename = {
        "feature": "Feature",
        "feature_group": "Feature group",
        "source_file": "Source file",
        "used_as_predictor": "Used as predictor",
        "model_role": "Model role",
        "missingness": "Missingness",
        "variance": "Variance",
        "n_unique": "Unique values",
        "high_corr_with": "Highly correlated with",
        "high_corr_abs_r": "|rho|",
        "redundancy_handling": "Redundancy handling",
    }
    catalog = catalog.rename(columns=rename)
    preferred = [
        "Feature",
        "Feature group",
        "Used as predictor",
        "Missingness",
        "Unique values",
        "Highly correlated with",
        "|rho|",
        "Redundancy handling",
        "Source file",
    ]
    return catalog[[col for col in preferred if col in catalog.columns]].copy()


def brain_age_age_correction(subject_path: Path) -> pd.DataFrame:
    correction = brain_age_aux_csv(subject_path, "live_brain_age_age_correction.csv")
    if correction.empty:
        return correction
    rename = {
        "correction": "Correction",
        "input_metric": "Input metric",
        "covariate": "Covariate",
        "slope": "Slope",
        "intercept": "Intercept",
        "formula": "Formula",
        "interpretation": "Interpretation",
    }
    return correction.rename(columns=rename)


def brain_age_model_summary(perf: pd.DataFrame) -> str:
    if perf.empty or "Group" not in perf.columns:
        return "Brain-age model performance is not available in the current refreshed CSV."
    overall = perf[perf["Group"].astype(str) == "Overall"]
    if overall.empty:
        overall = perf.head(1)
    row = overall.iloc[0]
    r2 = row.get("R²", row.get("r2", np.nan))
    mae = row.get("MAE years", row.get("MAE", np.nan))
    rmse = row.get("RMSE years", row.get("RMSE", np.nan))
    pearson = row.get("Pearson r", np.nan)
    return (
        "This is a regression model, so the primary performance metrics are "
        f"R²={format_number(r2, digits=3)}, MAE={format_number(mae, digits=2)} years, "
        f"RMSE={format_number(rmse, digits=2)} years, and Pearson r={format_number(pearson, digits=3)}."
    )


def brain_age_performance_inference(perf: pd.DataFrame) -> str:
    if perf.empty:
        return "Brain-age performance metrics are not available in the current refreshed CSV."
    overall = perf[perf["Group"].astype(str) == "Overall"]
    if overall.empty:
        overall = perf.head(1)
    row = overall.iloc[0]
    r2 = pd.to_numeric(pd.Series([row.get("R²", np.nan)]), errors="coerce").iloc[0]
    mae = row.get("MAE years", np.nan)
    rmse = row.get("RMSE years", np.nan)
    if pd.notna(r2) and r2 < 0.05:
        return (
            f"Cross-validated brain-age prediction is weak as an age regressor "
            f"(R²={format_number(r2, digits=3)}, MAE={format_number(mae, digits=2)} years, "
            f"RMSE={format_number(rmse, digits=2)} years), so BAG and corrected BAG are the more interpretable views."
        )
    return (
        f"Cross-validated brain-age prediction explains measurable chronological-age variance "
        f"(R²={format_number(r2, digits=3)}, MAE={format_number(mae, digits=2)} years)."
    )


def render_brain_age_stats_table(table: pd.DataFrame, key: str) -> None:
    if table.empty:
        st.info("No brain-age prediction statistics are available.")
        return
    st.dataframe(
        table,
        width="stretch",
        height=compact_table_height(len(table)),
        hide_index=True,
        column_config={
            "n": st.column_config.NumberColumn("n", format="%d"),
            "R²": st.column_config.NumberColumn("R²", format="%.3f"),
            "MAE years": st.column_config.NumberColumn("MAE years", format="%.2f"),
            "RMSE years": st.column_config.NumberColumn("RMSE years", format="%.2f"),
            "Median age": st.column_config.NumberColumn("Median age", format="%.1f"),
            "Median predicted age": st.column_config.NumberColumn("Median predicted age", format="%.1f"),
            "MAE": st.column_config.NumberColumn("MAE", format="%.2f"),
            "RMSE": st.column_config.NumberColumn("RMSE", format="%.2f"),
            "Median BAG": st.column_config.NumberColumn("Median BAG", format="%.2f"),
            "Mean BAG": st.column_config.NumberColumn("Mean BAG", format="%.2f"),
            "Median age-corrected BAG": st.column_config.NumberColumn("Median age-corrected BAG", format="%.3f"),
            "Pearson r": st.column_config.NumberColumn("Pearson r", format="%.3f"),
            "Pearson p": st.column_config.NumberColumn("Pearson p", format="%.3g"),
            "Spearman rho": st.column_config.NumberColumn("Spearman rho", format="%.3f"),
            "Spearman p": st.column_config.NumberColumn("Spearman p", format="%.3g"),
            "Pred min": st.column_config.NumberColumn("Pred min", format="%.1f"),
            "Pred max": st.column_config.NumberColumn("Pred max", format="%.1f"),
            "Age min": st.column_config.NumberColumn("Age min", format="%.1f"),
            "Age max": st.column_config.NumberColumn("Age max", format="%.1f"),
        },
        key=key,
    )


def brain_age_scatter_view(work: pd.DataFrame, y_col: str, title: str, y_title: str, *, identity_line: bool = False) -> go.Figure:
    fig = scatter_with_group_trends(
        work,
        "age",
        y_col,
        title,
        "Chronological age",
        y_title,
        identity_line=identity_line,
    )
    if y_col in {"BAG", "BAG_age_corrected"}:
        fig.add_hline(
            y=0,
            line_dash="dash",
            line_color="#334155",
            line_width=2,
            annotation_text="zero gap",
            annotation_font_color=CHART_TEXT,
        )
    return fig


def render_brain_age_prediction_panel(subject_path: Path, df: pd.DataFrame) -> bool:
    required = {"group", "age", "brain_age_pred"}
    if not required.issubset(df.columns):
        return False
    work = df.copy()
    work["age"] = pd.to_numeric(work["age"], errors="coerce")
    work["brain_age_pred"] = pd.to_numeric(work["brain_age_pred"], errors="coerce")
    for col in ("BAG", "BAG_age_corrected"):
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")
    work = work[work["group"].astype(str).isin(GROUP_ORDER) & work["age"].notna() & work["brain_age_pred"].notna()].copy()
    if work.empty:
        return False
    stats_table = brain_age_prediction_stats(work)
    perf_table = brain_age_model_performance(subject_path, stats_table)
    feature_catalog = brain_age_feature_catalog(subject_path)
    correction_table = brain_age_age_correction(subject_path)
    st.subheader("Live Structural Brain-Age Prediction")
    st.caption(f"Scatter plot is rendered live from `{subject_path.relative_to(ANALYSIS_ROOT).as_posix()}`.")
    st.markdown(
        "<div class='definition-card'>"
        "<h4>Model audit</h4>"
        "<p><strong>Model:</strong> repeated 5-fold cross-validated pipeline with median imputation, standardization, PCA compression, and RidgeCV regression. "
        "The target is chronological age. The model uses structural connectome predictors only; group, subject ID, sex, and phase are not used as predictors in this brain-age model.</p>"
        f"<p>{escape(brain_age_model_summary(perf_table))}</p>"
        "<p>The dashed diagonal in the predicted-age plot is the identity line where predicted age equals chronological age. "
        "Colored lines are group-specific linear trend lines. No outlier trimming is applied to these scatter plots.</p>"
        "</div>",
        unsafe_allow_html=True,
    )
    if not feature_catalog.empty:
        predictors = feature_catalog[feature_catalog["Used as predictor"].astype(str).str.lower().isin(("true", "1", "yes"))].copy()
        st.markdown("#### Brain-age predictors")
        st.dataframe(
            predictors,
            width="stretch",
            height=compact_table_height(min(len(predictors), 13)),
            hide_index=True,
            column_config={
                "Missingness": st.column_config.NumberColumn("Missingness", format="%.3f"),
                "|rho|": st.column_config.NumberColumn("|rho|", format="%.3f"),
            },
            key="brain-age-feature-catalog-predictors",
        )
        with st.expander("Metadata/non-predictor audit and redundancy handling", expanded=False):
            st.dataframe(
                feature_catalog,
                width="stretch",
                height=compact_table_height(len(feature_catalog)),
                hide_index=True,
                column_config={
                    "Missingness": st.column_config.NumberColumn("Missingness", format="%.3f"),
                    "|rho|": st.column_config.NumberColumn("|rho|", format="%.3f"),
                },
                key="brain-age-feature-catalog-all",
            )
    left, right = st.columns((1.35, 1), gap="large")
    with left:
        plot_tabs = st.tabs(["Predicted age", "BAG", "Age-corrected BAG"])
        with plot_tabs[0]:
            st.plotly_chart(
                brain_age_scatter_view(
                    work,
                    "brain_age_pred",
                    "Live structural brain-age prediction",
                    "Predicted brain age",
                    identity_line=True,
                ),
                width="stretch",
                config={"displaylogo": False},
            )
        with plot_tabs[1]:
            if "BAG" in work.columns:
                st.plotly_chart(
                    brain_age_scatter_view(work, "BAG", "Brain-age gap vs chronological age", "BAG years"),
                    width="stretch",
                    config={"displaylogo": False},
                )
            else:
                st.info("BAG is not present in the current brain-age prediction CSV.")
        with plot_tabs[2]:
            if "BAG_age_corrected" in work.columns:
                st.plotly_chart(
                    brain_age_scatter_view(
                        work,
                        "BAG_age_corrected",
                        "Age-corrected BAG vs chronological age",
                        "Age-corrected BAG years",
                    ),
                    width="stretch",
                    config={"displaylogo": False},
                )
            else:
                st.info("Age-corrected BAG is not present in the current brain-age prediction CSV.")
    with right:
        st.markdown(
            "<div class='edr-definition-box'>"
            "<h5>Predicted brain age, BAG, and scale</h5>"
            "<p>The predicted-age panel stays close to the cohort mean because the cross-validated age signal is weak. "
            "A log scale is not appropriate for age gaps; the BAG and age-corrected BAG tabs show deviations on an interpretable year scale.</p>"
            "</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div class='edr-inference-box'>{escape(brain_age_performance_inference(perf_table))}</div>",
            unsafe_allow_html=True,
        )
        render_brain_age_stats_table(perf_table, key="brain-age-prediction-performance")
        if not correction_table.empty:
            with st.expander("Age-corrected BAG residualization", expanded=True):
                row = correction_table.iloc[0]
                slope = row.get("Slope", np.nan)
                intercept = row.get("Intercept", np.nan)
                st.markdown(
                    "<div class='edr-definition-box'>"
                    "<h5>What linear dependence means</h5>"
                    "<p>We fit the ordinary least-squares line <code>BAG = beta0 + beta1 * chronological_age</code>. "
                    "The age-corrected BAG is the residual after subtracting that fitted line, so the displayed value is less driven by chronological age.</p>"
                    f"<p><code>BAG_age_corrected = BAG - ({format_number(slope, digits=4)} * age + {format_number(intercept, digits=4)})</code></p>"
                    "</div>",
                    unsafe_allow_html=True,
                )
                st.dataframe(
                    correction_table,
                    width="stretch",
                    height=compact_table_height(len(correction_table)),
                    hide_index=True,
                    column_config={
                        "Slope": st.column_config.NumberColumn("Slope", format="%.4f"),
                        "Intercept": st.column_config.NumberColumn("Intercept", format="%.4f"),
                    },
                    key="brain-age-correction-table",
                )
    return True


def metric_formula(metric: str) -> str:
    formulas = {
        "mean_delay_ms": "mean(D_ij), D_ij = L_ij / v",
        "median_delay_ms": "median(D_ij), D_ij = L_ij / v",
        "p90_delay_ms": "percentile_90(D_ij), D_ij = L_ij / v",
        "delay_burden_ms": "sum(W_ij D_ij) / sum(W_ij)",
        "long_delay_fraction": "|{D_ij >= theta_delay}| / |E_+|",
        "delay_weighted_strength": "sum(W_ij / D_ij)",
        "delay_path_ms": "mean shortest-path distance using D_ij as edge cost",
        "BAG": "brain_age_pred - chronological_age",
        "BAG_age_corrected": "BAG - (beta_1 * chronological_age + beta_0)",
        "brain_age_pred": "f_theta(structural connectome features)",
        "edr_residual_mean": "mean(log1p(W_ij) - f_CN(log L_ij))",
        "edr_residual_short": "mean residual over CN-reference short edges",
        "edr_residual_medium": "mean residual over CN-reference medium edges",
        "edr_residual_long": "mean residual over CN-reference long edges",
        "structural_compensation_index": "edr_residual_short - edr_residual_long",
        "long_range_vulnerability": "-edr_residual_long",
        "short_range_preservation": "edr_residual_short",
    }
    return formulas.get(metric, "")


def metric_group_table_html(section: Section, metrics: list[str]) -> str:
    rows = []
    for metric in metrics:
        rows.append(
            "<tr>"
            f"<td><strong>{escape(metric_label(metric))}</strong></td>"
            f"<td>{escape(metric_explanation(section, metric))}</td>"
            f"<td><code>{escape(metric_formula(metric) or metric)}</code></td>"
            "</tr>"
        )
    body = "".join(rows)
    return (
        "<div class='definition-card compact-definition-card'>"
        "<h4>Metric definitions and formulas</h4>"
        "<table class='definition-table'><thead><tr>"
        "<th>Metric</th><th>Definition</th><th>Formula</th>"
        f"</tr></thead><tbody>{body}</tbody></table>"
        "</div>"
    )


def render_grouped_subject_metric_tabs(
    section: Section,
    csv_files: list[Path],
    subject_path: Path,
    metric_groups: tuple[tuple[str, tuple[str, ...]], ...],
    hide_visual_outliers: bool,
    *,
    show_definition_table: bool = True,
) -> bool:
    df = _safe_read_csv(subject_path)
    if df is None or df.empty or "group" not in df.columns:
        return False
    available_metrics = set(numeric_metric_columns(df, section.label))
    groups = [
        (label, [metric for metric in metrics if metric in available_metrics])
        for label, metrics in metric_groups
    ]
    groups = [(label, metrics) for label, metrics in groups if metrics]
    if not groups:
        return False
    tabs = st.tabs([label for label, _ in groups])
    for tab, (group_label, metrics) in zip(tabs, groups):
        with tab:
            if show_definition_table:
                st.markdown(metric_group_table_html(section, metrics), unsafe_allow_html=True)
            for metric in metrics:
                data = subject_metric_frame(df, metric, value_col=metric)
                if data.empty:
                    continue
                desc = metric_descriptives(csv_files, metric, data=data)
                pairwise = metric_pairwise(csv_files, metric)
                chart, _, _, removed_counts = live_distribution_chart(
                    data,
                    metric_label(metric),
                    hide_visual_outliers=hide_visual_outliers,
                    stats_df=desc,
                    subtitle_lines=chart_stat_subtitles(csv_files, metric),
                )
                st.markdown(f"### {metric_label(metric)}")
                left, right = st.columns((1.25, 1), gap="large")
                with left:
                    st.plotly_chart(chart, width="stretch", config={"displaylogo": False})
                    st.caption(f"Live chart from `{subject_path.relative_to(ANALYSIS_ROOT).as_posix()}`.")
                    if sum(removed_counts.values()):
                        st.caption(
                            "Visual-only 1.5-IQR trim active; statistics and CSV tables remain unchanged. "
                            f"Hidden points: {group_count_text(removed_counts)}."
                        )
                with right:
                    render_metric_definition_inference_table(
                        section,
                        metric,
                        desc,
                        pairwise,
                        key=f"{section_key(section.label)}-{section_key(group_label)}-{metric}",
                    )
    return True


def render_delay_definition_panel(subject_path: Path | None = None) -> None:
    velocity = None
    if subject_path is not None:
        df = _safe_read_csv(subject_path)
        if df is not None and "velocity_mm_per_ms" in df.columns:
            vals = pd.to_numeric(df["velocity_mm_per_ms"], errors="coerce").dropna()
            if not vals.empty:
                velocity = float(vals.iloc[0])
    velocity_text = format_number(velocity) if velocity is not None else "configured velocity"
    rows = [
        ("Edge delay", "Length-derived conduction-delay proxy for an AAL3 edge.", "D_ij = L_ij / v"),
        ("Delay burden", "Weight-averaged subject delay; stronger edges contribute more.", "sum(W_ij D_ij) / sum(W_ij)"),
        ("Long-delay fraction", "Fraction of positive edges above the CN-reference 75th percentile delay threshold.", "|{D_ij >= theta_delay}| / |E_+|"),
        ("Delay path", "Shortest-path delay through the structural graph using edge delay as cost.", "mean shortest path using D_ij"),
    ]
    table_rows = "".join(
        f"<tr><td><strong>{escape(name)}</strong></td><td>{escape(defn)}</td><td><code>{escape(formula)}</code></td></tr>"
        for name, defn, formula in rows
    )
    st.markdown(
        "<div class='definition-card'>"
        "<h3>Delay Methodology</h3>"
        "<p>Delay is a structural proxy, not a directly measured conduction velocity. "
        "It uses <code>len_mean</code> as AAL3 tract-length summary, <code>fd_sum</code> as structural weight, "
        f"and v={escape(velocity_text)} mm/ms when available.</p>"
        "<table class='definition-table'><thead><tr><th>Metric</th><th>Definition</th><th>Formula</th></tr></thead>"
        f"<tbody>{table_rows}</tbody></table>"
        "</div>",
        unsafe_allow_html=True,
    )


def render_brain_age_definition_panel() -> None:
    rows = [
        ("Brain-age model", "Repeated cross-validated PCA plus RidgeCV regression trained to predict chronological age from structural connectome summaries.", "age_hat = Ridge(PCA(zscore(SC features)))"),
        ("Predicted brain age", "Age estimated from structural connectome features by the refreshed brain-age model.", "brain_age_pred = age_hat"),
        ("BAG", "Brain-age gap; positive values indicate an older-appearing structural connectome.", "BAG = brain_age_pred - age"),
        ("Age-corrected BAG", "Residual BAG after subtracting the fitted linear relationship between BAG and chronological age.", "BAG_age_corrected = BAG - (beta_1 * age + beta_0)"),
        ("Normative z-score", "Optional standardized deviation from the normative model when available.", "z = (observed - expected) / sigma"),
    ]
    table_rows = "".join(
        f"<tr><td><strong>{escape(name)}</strong></td><td>{escape(defn)}</td><td><code>{escape(formula)}</code></td></tr>"
        for name, defn, formula in rows
    )
    st.markdown(
        "<div class='definition-card'>"
        "<h3>Brain-Age Methodology</h3>"
        "<p>The brain-age block asks whether the structural connectome resembles an older or younger brain than expected from chronological age. "
        "BAG and corrected BAG are descriptive biomarkers for downstream clinical association, not standalone diagnosis labels.</p>"
        "<table class='definition-table'><thead><tr><th>Metric</th><th>Definition</th><th>Formula</th></tr></thead>"
        f"<tbody>{table_rows}</tbody></table>"
        "</div>",
        unsafe_allow_html=True,
    )


def render_advanced_definition_panel() -> None:
    st.markdown(
        "<div class='definition-card'>"
        "<h3>Advanced Structural Metrics</h3>"
        "<p><strong>Why distance adjustment is needed:</strong> structural connectome weights naturally depend on tract length. "
        "Short edges usually have stronger weights and long edges usually have weaker weights, even in healthy controls. "
        "If we compare raw weights directly, disease effects can be confused with this normal distance effect.</p>"
        "<p><strong>What the CN-reference model is:</strong> using the CN group only, the app fits a normal distance-decay curve "
        "<code>f_CN(log L_ij)</code>. This curve predicts the expected log structural weight for an ROI-pair edge of length <code>L_ij</code>. "
        "Then every subject and edge is compared against that CN expectation.</p>"
        "<p><strong>How to read a residual:</strong> <code>resid_ij = log1p(W_ij) - f_CN(log L_ij)</code>. "
        "A positive residual means the edge is stronger than expected for its length; a negative residual means it is weaker than expected for its length. "
        "This Advanced section is therefore a CN-reference residual sensitivity analysis. It is different from the paper-style outlier/exception detector in <strong>EDR Exceptions</strong>.</p>"
        "<table class='definition-table'><thead><tr><th>Family</th><th>Definition</th><th>Formula</th></tr></thead><tbody>"
        "<tr><td><strong>CN-reference residual sensitivity</strong></td><td>Average observed-minus-expected log edge weight after correcting for tract length using the CN distance-decay curve. Positive means stronger than expected; negative means weaker than expected.</td><td><code>resid_ij = log1p(W_ij) - f_CN(log L_ij)</code></td></tr>"
        "<tr><td><strong>Short / medium / long residuals</strong></td><td>The same residual summarized separately within short-, medium-, or long-range edge sets. This helps identify whether disease effects are concentrated in local or long-distance pathways.</td><td><code>mean(resid_ij for edges in selected length range)</code></td></tr>"
        "<tr><td><strong>Structural compensation</strong></td><td>Short-range residual preservation relative to long-range residual loss. Higher values suggest short-range structure is relatively preserved while long-range structure is reduced.</td><td><code>SCI = residual_short - residual_long</code></td></tr>"
        "<tr><td><strong>Long-range vulnerability</strong></td><td>Transforms negative long-range residuals into a positive vulnerability score. Higher values mean long-range edges are weaker than expected after length adjustment.</td><td><code>LR vulnerability = -residual_long</code></td></tr>"
        "<tr><td><strong>Disease-gradient edges</strong></td><td>ROI pairs whose residuals change monotonically across CN, MCI, and AD. This ranks edges that follow a disease-stage gradient after distance adjustment.</td><td><code>rho(edge residual, CN/MCI/AD ordinal)</code></td></tr>"
        "</tbody></table>"
        "<table class='definition-table'><thead><tr><th>Symbol</th><th>Meaning</th></tr></thead><tbody>"
        "<tr><td><code>W_ij</code></td><td>Structural edge weight between AAL regions i and j, usually the SIFT2/fd_sum weight.</td></tr>"
        "<tr><td><code>L_ij</code></td><td>Mean tract length for the same ROI-pair edge, from the length matrix.</td></tr>"
        "<tr><td><code>log1p(W_ij)</code></td><td>Log-transformed edge weight, using log(1 + W) to handle small or zero-like values safely.</td></tr>"
        "<tr><td><code>f_CN(log L_ij)</code></td><td>The expected log edge weight at that tract length, learned from CN controls only.</td></tr>"
        "<tr><td><code>resid_ij</code></td><td>The distance-adjusted difference between observed and expected edge weight.</td></tr>"
        "</tbody></table>"
        "</div>",
        unsafe_allow_html=True,
    )


def render_brain_age_panel(section: Section, csv_files: list[Path], hide_visual_outliers: bool) -> bool:
    subject_path = subject_table_for_section(section, csv_files)
    if subject_path is None:
        return False
    df = _safe_read_csv(subject_path)
    if df is None or df.empty:
        return False
    render_brain_age_definition_panel()
    rendered = render_brain_age_prediction_panel(subject_path, df)
    st.subheader("Brain-Age Metrics")
    rendered = (
        render_grouped_subject_metric_tabs(
            section,
            csv_files,
            subject_path,
            BRAIN_AGE_SUBJECT_METRIC_GROUPS,
            hide_visual_outliers,
            show_definition_table=True,
        )
        or rendered
    )
    return rendered


def render_brain_age_layout(section: Section, csv_files: list[Path], max_charts: int, hide_visual_outliers: bool) -> bool:
    return render_brain_age_panel(section, csv_files, hide_visual_outliers)


def atlas_label_map() -> dict[str, str]:
    labels = aal_label_lookup(str(AAL_LABEL_CSV))
    if labels.empty or not {"node_name", "atlas_label"}.issubset(labels.columns):
        return {}
    return {str(row["node_name"]): str(row["atlas_label"]) for _, row in labels.iterrows()}


def aal_node_number_maps() -> tuple[dict[int, str], dict[int, str]]:
    labels = aal_label_lookup(str(AAL_LABEL_CSV))
    if labels.empty or not {"node", "node_name", "atlas_label"}.issubset(labels.columns):
        return {}, {}
    node_names: dict[int, str] = {}
    atlas_labels: dict[int, str] = {}
    for _, row in labels.dropna(subset=["node"]).iterrows():
        node = int(row["node"])
        node_name = str(row.get("node_name") or f"AAL_{node:03d}")
        atlas_label = str(row.get("atlas_label") or node_name)
        node_names[node] = node_name
        atlas_labels[node] = atlas_label
    return node_names, atlas_labels


def add_atlas_labels(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    labels = atlas_label_map()
    if "atlas_label" not in out.columns and "node_name" in out.columns:
        out["atlas_label"] = out["node_name"].astype(str).map(labels).fillna(out["node_name"].astype(str))
    return out


def rank_table_for_metric_comparison(
    ranking_df: pd.DataFrame,
    metric: str,
    comparison: str,
    top_n: int,
    only_fdr: bool,
) -> pd.DataFrame:
    if ranking_df.empty:
        return pd.DataFrame()
    work = ranking_df[
        (ranking_df["metric"].astype(str) == metric)
        & (ranking_df["comparison"].astype(str) == comparison)
    ].copy()
    if work.empty:
        return pd.DataFrame()
    q_col = "BH q" if "BH q" in work.columns else next((c for c in ("q", "bm_q", "mwu_q") if c in work.columns), None)
    p_col = "p" if "p" in work.columns else next((c for c in ("p_value", "bm_p", "mwu_p") if c in work.columns), None)
    if only_fdr and q_col:
        work = work[pd.to_numeric(work[q_col], errors="coerce") < 0.05].copy()
    sort_cols = [c for c in (q_col, p_col) if c]
    if sort_cols:
        work = work.sort_values(sort_cols, na_position="last")
    work = add_atlas_labels(work.head(top_n).copy())
    if work.empty:
        return pd.DataFrame()
    work.insert(0, "Rank", range(1, len(work) + 1))
    rename = {}
    if q_col and q_col != "BH q":
        rename[q_col] = "BH q"
    if p_col and p_col != "p":
        rename[p_col] = "p"
    work = work.rename(columns=rename)
    keep = [
        c for c in (
            "Rank",
            "node_name",
            "atlas_label",
            "higher_group",
            "Median diff CN-MCI",
            "Median diff CN-AD",
            "Median diff MCI-AD",
            "p",
            "BH q",
            "Cliff's delta",
        )
        if c in work.columns
    ]
    return work[keep].copy()


def repeated_table_for_metric(repeated_df: pd.DataFrame, metric: str, top_n: int, only_fdr: bool) -> pd.DataFrame:
    if repeated_df.empty or "metric" not in repeated_df.columns:
        return pd.DataFrame()
    work = repeated_df[repeated_df["metric"].astype(str) == metric].copy()
    if work.empty:
        return pd.DataFrame()
    if only_fdr and "BH q<0.05 hits" in work.columns:
        work = work[pd.to_numeric(work["BH q<0.05 hits"], errors="coerce").fillna(0) > 0].copy()
    work = work.sort_values(
        [c for c in ("BH q<0.05 hits", "Top-N hits", "Best p", "Best BH q") if c in work.columns],
        ascending=[False, False, True, True][: len([c for c in ("BH q<0.05 hits", "Top-N hits", "Best p", "Best BH q") if c in work.columns])],
        na_position="last",
    ).head(top_n)
    work = add_atlas_labels(work)
    keep = [
        c for c in (
            "Rank",
            "node_name",
            "atlas_label",
            "Top-N hits",
            "BH q<0.05 hits",
            "Best p",
            "Best BH q",
            "comparisons",
            "directions",
        )
        if c in work.columns
    ]
    return work[keep].copy()


def aal_ranking_inference(metric: str, comparison: str, table: pd.DataFrame) -> str:
    if table.empty:
        return f"No AAL regions pass the current {metric_label(metric)} filter for {comparison}."
    top = table.iloc[0]
    label = str(top.get("atlas_label") or top.get("node_name") or "AAL region")
    q_value = top.get("BH q", pd.NA)
    p_value = top.get("p", pd.NA)
    higher = str(top.get("higher_group", "")).strip()
    direction = f"; higher median group: {higher}" if higher else ""
    return (
        f"Top {metric_label(metric)} region for {comparison} is {label} "
        f"(p={format_p(p_value)}, q={format_p(q_value)}{direction})."
    )


def repeated_ranking_inference(metric: str, table: pd.DataFrame) -> str:
    if table.empty:
        return f"No repeated top-N AAL regions pass the current {metric_label(metric)} filter."
    top = table.iloc[0]
    label = str(top.get("atlas_label") or top.get("node_name") or "AAL region")
    hits = top.get("Top-N hits", pd.NA)
    comparisons = str(top.get("comparisons", ""))
    return f"For {metric_label(metric)}, the most recurrent top-N region is {label}, appearing in {format_number(hits, digits=0)} comparisons ({comparisons})."


def render_delay_aal_rankings(csv_files: list[Path]) -> bool:
    rank_path = next((p for p in csv_files if p.name == "delay_pairwise_roi_rankings.csv"), None)
    repeated_path = next((p for p in csv_files if p.name == "delay_repeated_top_regions.csv"), None)
    if rank_path is None:
        st.info("AAL-level delay ranking tables have not been generated yet. Run a full dashboard refresh to populate them.")
        return False
    rankings = _safe_read_csv(rank_path)
    repeated = _safe_read_csv(repeated_path) if repeated_path is not None else pd.DataFrame()
    if rankings is None or rankings.empty or "metric" not in rankings.columns:
        st.info("AAL-level delay ranking table is empty.")
        return False

    st.subheader("AAL-Level Delay Ranking")
    top_col, q_col = st.columns((1.0, 1.0))
    with top_col:
        top_n = st.slider("Top N AAL regions", min_value=5, max_value=50, value=20, step=5, key="delay-aal-top-n")
    with q_col:
        only_fdr = st.checkbox("Only q<0.05", value=False, key="delay-aal-only-fdr")
    metrics = [metric for metric in DELAY_NODE_RANK_METRICS if metric in set(rankings["metric"].astype(str))]
    if not metrics:
        metrics = sorted(rankings["metric"].dropna().astype(str).unique())
    metric_tabs = st.tabs([metric_label(metric) for metric in metrics])
    for metric_tab, metric in zip(metric_tabs, metrics):
        with metric_tab:
            st.markdown(metric_group_table_html(Section("Delay", "", ()), [metric]), unsafe_allow_html=True)
            comparison_tabs = st.tabs(["CN vs MCI", "CN vs AD", "MCI vs AD"])
            for comparison_tab, comparison in zip(comparison_tabs, ["CN vs MCI", "CN vs AD", "MCI vs AD"]):
                with comparison_tab:
                    left, right = st.columns((1.1, 1), gap="large")
                    table = rank_table_for_metric_comparison(rankings, metric, comparison, top_n, only_fdr)
                    repeated_table = repeated_table_for_metric(repeated if repeated is not None else pd.DataFrame(), metric, top_n, only_fdr)
                    with left:
                        st.markdown(f"#### {comparison} Top {top_n}: {metric_label(metric)}")
                        if table.empty:
                            st.info(f"No AAL regions pass the current filter for {comparison}.")
                        else:
                            st.dataframe(
                                table,
                                width="stretch",
                                height=430,
                                hide_index=True,
                                column_config={
                                    "p": st.column_config.NumberColumn("p", format="%.3g"),
                                    "BH q": st.column_config.NumberColumn("BH q", format="%.3g"),
                                    "Cliff's delta": st.column_config.NumberColumn("Cliff's delta", format="%.3f"),
                                },
                                key=f"delay-rank-{metric}-{comparison}-{top_n}-{only_fdr}",
                            )
                        st.markdown(
                            f"<div class='edr-inference-box'>{escape(aal_ranking_inference(metric, comparison, table))}</div>",
                            unsafe_allow_html=True,
                        )
                    with right:
                        st.markdown("#### Repeated Top-N AAL Regions")
                        if repeated_table.empty:
                            st.info("No repeated AAL regions pass the current filter.")
                        else:
                            st.dataframe(
                                repeated_table,
                                width="stretch",
                                height=430,
                                hide_index=True,
                                column_config={
                                    "Best p": st.column_config.NumberColumn("Best p", format="%.3g"),
                                    "Best BH q": st.column_config.NumberColumn("Best BH q", format="%.3g"),
                                },
                                key=f"delay-repeated-{metric}-{comparison}-{top_n}-{only_fdr}",
                            )
                        st.markdown(
                            f"<div class='edr-inference-box'>{escape(repeated_ranking_inference(metric, repeated_table))}</div>",
                            unsafe_allow_html=True,
                        )
    st.caption(f"Live AAL delay rankings from `{rank_path.relative_to(ANALYSIS_ROOT).as_posix()}`.")
    return True


def render_delay_panel(section: Section, csv_files: list[Path], hide_visual_outliers: bool) -> bool:
    subject_path = subject_table_for_section(section, csv_files)
    render_delay_definition_panel(subject_path)
    rendered = False
    if subject_path is not None:
        st.subheader("Delay Metrics")
        st.caption(f"Metrics are rendered live from `{subject_path.relative_to(ANALYSIS_ROOT).as_posix()}`.")
        rendered = render_grouped_subject_metric_tabs(
            section,
            csv_files,
            subject_path,
            DELAY_SUBJECT_METRIC_GROUPS,
            hide_visual_outliers,
            show_definition_table=True,
        )
    rendered = render_delay_aal_rankings(csv_files) or rendered
    return rendered or True


def render_disease_gradient_panel(csv_files: list[Path]) -> bool:
    edge_path = next((p for p in csv_files if p.name == "top_disease_gradient_edges.csv"), None)
    if edge_path is None:
        edge_path = next((p for p in csv_files if p.name == "edr_edge_disease_gradient_fd_sum_len_mean.csv"), None)
    if edge_path is None:
        return False
    table = _safe_read_csv(edge_path)
    if table is None or table.empty:
        return False
    st.subheader("Disease-Gradient Edges")
    work = table.copy().head(50)
    rename = {
        "roi_a_label": "ROI A",
        "roi_b_label": "ROI B",
        "disease_gradient_rho": "rho",
        "disease_gradient_p": "p",
        "disease_gradient_q": "BH q",
    }
    work = work.rename(columns={k: v for k, v in rename.items() if k in work.columns})
    if {"i", "j"}.issubset(work.columns) and not {"ROI A", "ROI B"}.issubset(work.columns):
        node_names, atlas_labels = aal_node_number_maps()

        def _node_label(value: object, lookup: dict[int, str], fallback_prefix: str) -> str:
            node = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
            if pd.isna(node):
                return ""
            node_int = int(node)
            return lookup.get(node_int, f"{fallback_prefix}_{node_int:03d}")

        work.insert(0, "ROI A", work["i"].map(lambda value: _node_label(value, atlas_labels, "AAL")))
        work.insert(1, "ROI B", work["j"].map(lambda value: _node_label(value, atlas_labels, "AAL")))
        work.insert(2, "Node A", work["i"].map(lambda value: _node_label(value, node_names, "AAL")))
        work.insert(3, "Node B", work["j"].map(lambda value: _node_label(value, node_names, "AAL")))
    left, right = st.columns((1.1, 1), gap="large")
    with left:
        display_cols = [
            c
            for c in ("ROI A", "ROI B", "Node A", "Node B", "rho", "p", "BH q", "n", "length_mm", "weight")
            if c in work.columns
        ]
        if not display_cols:
            display_cols = list(work.columns[:8])
        st.dataframe(
            work[display_cols],
            width="stretch",
            height=450,
            hide_index=True,
            column_config={
                "rho": st.column_config.NumberColumn("rho", format="%.3f"),
                "p": st.column_config.NumberColumn("p", format="%.3g"),
                "BH q": st.column_config.NumberColumn("BH q", format="%.3g"),
            },
        )
    with right:
        sig = work[pd.to_numeric(work.get("BH q"), errors="coerce") < 0.05] if "BH q" in work.columns else pd.DataFrame()
        st.markdown(
            "<div class='edr-definition-box'>"
            "<h5>Disease-gradient edges</h5>"
            "<p>These ROI pairs are ranked by how their CN-reference residuals vary across diagnostic stage. "
            "This is an exploratory residual-gradient map, not the paper-style exception detector used in EDR Exceptions.</p>"
            "</div>",
            unsafe_allow_html=True,
        )
        inference = (
            f"{len(sig):,} displayed disease-gradient edges are FDR-significant."
            if not sig.empty
            else "No displayed disease-gradient edge reaches FDR q<0.05 in the current table."
        )
        if not work.empty and {"ROI A", "ROI B", "rho", "p", "BH q"}.issubset(work.columns):
            top = work.iloc[0]
            rho = pd.to_numeric(pd.Series([top.get("rho")]), errors="coerce").iloc[0]
            direction = (
                "decreases from CN to MCI to AD"
                if pd.notna(rho) and float(rho) < 0
                else "increases from CN to MCI to AD"
                if pd.notna(rho) and float(rho) > 0
                else "changes across CN, MCI, and AD"
            )
            inference += (
                f" Top pair: {top.get('ROI A')} - {top.get('ROI B')} "
                f"{direction} (rho={format_number(rho, digits=3)}, "
                f"p={format_p(top.get('p'))}, q={format_p(top.get('BH q'))})."
            )
        st.markdown(f"<div class='edr-inference-box'>{escape(inference)}</div>", unsafe_allow_html=True)
        st.caption(f"Live edge table from `{edge_path.relative_to(ANALYSIS_ROOT).as_posix()}`.")
    return True


def render_advanced_panel(section: Section, csv_files: list[Path], hide_visual_outliers: bool) -> bool:
    subject_path = subject_table_for_section(section, csv_files)
    render_advanced_definition_panel()
    rendered = False
    tabs = st.tabs(["Compensation", "Long-Range Vulnerability", "CN-Reference Residual Sensitivity", "Disease-Gradient Edges"])
    for tab, (label, metrics) in zip(tabs[:3], ADVANCED_SUBJECT_METRIC_GROUPS):
        with tab:
            if subject_path is None:
                st.info("Advanced subject-level table is not available.")
                continue
            st.markdown(metric_group_table_html(section, list(metrics)), unsafe_allow_html=True)
            rendered = render_grouped_subject_metric_tabs(
                section,
                csv_files,
                subject_path,
                ((label, metrics),),
                hide_visual_outliers,
                show_definition_table=False,
            ) or rendered
    with tabs[3]:
        rendered = render_disease_gradient_panel(csv_files) or rendered
    return rendered or True


def coupling_aal_value_label(metric: str) -> str:
    labels = {
        "coupling_index": "Regional coupling index",
        "micro_topology_ratio": "Microstructure/topology ratio",
    }
    return labels.get(metric, metric_label(metric))


@st.cache_data(ttl=300, show_spinner=False)
def coupling_aal_long(
    graph_csv: str,
    micro_csv: str,
    topology_metric: str,
    micro_metric: str,
    label_csv: str,
) -> pd.DataFrame:
    graph_needed = {"subject_id", "group", "node", "node_name", topology_metric}
    micro_needed = {"subject_id", "group", "node", "node_name", "metric", "value"}
    try:
        graph = pd.read_csv(graph_csv, usecols=lambda col: col in graph_needed)
        micro = pd.read_csv(micro_csv, usecols=lambda col: col in micro_needed)
    except Exception:
        return pd.DataFrame()
    if not graph_needed.issubset(graph.columns) or not micro_needed.issubset(micro.columns):
        return pd.DataFrame()

    graph = graph.rename(columns={"group": "group_graph", topology_metric: "topology_value"})
    micro = micro[micro["metric"].astype(str) == micro_metric].copy()
    micro = micro.rename(columns={"group": "group_micro", "value": "micro_value"})
    merged = graph.merge(
        micro,
        on=["subject_id", "node", "node_name"],
        how="inner",
    )
    if merged.empty:
        return pd.DataFrame()
    merged["group"] = merged["group_graph"].fillna(merged["group_micro"]).astype(str)
    merged = merged[merged["group"].isin(GROUP_ORDER)].copy()
    merged["topology_value"] = pd.to_numeric(merged["topology_value"], errors="coerce")
    merged["micro_value"] = pd.to_numeric(merged["micro_value"], errors="coerce")
    merged = merged[merged["topology_value"].notna() & merged["micro_value"].notna()].copy()
    if merged.empty:
        return pd.DataFrame()

    topo_mean = merged.groupby("subject_id")["topology_value"].transform("mean")
    micro_mean = merged.groupby("subject_id")["micro_value"].transform("mean")
    topo_std = merged.groupby("subject_id")["topology_value"].transform(lambda s: s.std(ddof=0))
    micro_std = merged.groupby("subject_id")["micro_value"].transform(lambda s: s.std(ddof=0))
    merged["topology_z"] = np.where(topo_std > 0, (merged["topology_value"] - topo_mean) / topo_std, np.nan)
    merged["micro_z"] = np.where(micro_std > 0, (merged["micro_value"] - micro_mean) / micro_std, np.nan)
    merged["coupling_index"] = merged["topology_z"] * merged["micro_z"]
    merged["micro_topology_ratio"] = np.where(
        merged["topology_value"].abs() > 0,
        merged["micro_value"] / merged["topology_value"],
        np.nan,
    )

    labels = aal_label_lookup(label_csv)
    if not labels.empty and {"node", "atlas_label"}.issubset(labels.columns):
        keep = [c for c in ("node", "atlas_label", "atlas_value", "atlas_version") if c in labels.columns]
        merged = merged.merge(labels[keep], on="node", how="left")
    else:
        merged["atlas_label"] = merged["node_name"]
        merged["atlas_version"] = "unmapped"
    merged["atlas_label"] = merged["atlas_label"].fillna(merged["node_name"])
    merged["topology_metric"] = topology_metric
    merged["micro_metric"] = micro_metric
    return merged[
        [
            "subject_id",
            "group",
            "node",
            "node_name",
            "atlas_label",
            "topology_metric",
            "micro_metric",
            "topology_value",
            "micro_value",
            "topology_z",
            "micro_z",
            "coupling_index",
            "micro_topology_ratio",
        ]
    ].copy()


@st.cache_data(ttl=300, show_spinner=False)
def coupling_aal_pairwise_rankings(
    graph_csv: str,
    micro_csv: str,
    topology_metric: str,
    micro_metric: str,
    value_metric: str,
    group_a: str,
    group_b: str,
    label_csv: str,
) -> pd.DataFrame:
    df = coupling_aal_long(graph_csv, micro_csv, topology_metric, micro_metric, label_csv)
    if df.empty or value_metric not in df.columns:
        return pd.DataFrame()
    df = df[df["group"].astype(str).isin([group_a, group_b])].copy()
    df[value_metric] = pd.to_numeric(df[value_metric], errors="coerce")
    df["micro_topology_ratio"] = pd.to_numeric(df["micro_topology_ratio"], errors="coerce")
    df = df[df[value_metric].notna()].copy()
    rows = []
    for (node, node_name, atlas_label), sub in df.groupby(["node", "node_name", "atlas_label"], dropna=False):
        a = sub.loc[sub["group"].astype(str) == group_a, value_metric].dropna()
        b = sub.loc[sub["group"].astype(str) == group_b, value_metric].dropna()
        if len(a) < 5 or len(b) < 5:
            continue
        try:
            test = stats.mannwhitneyu(a, b, alternative="two-sided")
            p_value = float(test.pvalue)
            u_value = float(test.statistic)
            cliffs_delta = (2.0 * u_value / (len(a) * len(b))) - 1.0
        except Exception:
            p_value = math.nan
            cliffs_delta = math.nan
        median_a = float(a.median())
        median_b = float(b.median())
        ratio_a = sub.loc[sub["group"].astype(str) == group_a, "micro_topology_ratio"].dropna()
        ratio_b = sub.loc[sub["group"].astype(str) == group_b, "micro_topology_ratio"].dropna()
        median_ratio_a = float(ratio_a.median()) if not ratio_a.empty else math.nan
        median_ratio_b = float(ratio_b.median()) if not ratio_b.empty else math.nan
        rows.append(
            {
                "node": int(node),
                "node_name": str(node_name),
                "atlas_label": str(atlas_label),
                "metric": f"{topology_metric}_vs_{micro_metric}_{value_metric}",
                "comparison": f"{group_a} vs {group_b}",
                f"n_{group_a}": int(len(a)),
                f"n_{group_b}": int(len(b)),
                f"median_{group_a}": median_a,
                f"median_{group_b}": median_b,
                "median_diff_a_minus_b": median_a - median_b,
                f"median_ratio_{group_a}": median_ratio_a,
                f"median_ratio_{group_b}": median_ratio_b,
                "ratio_direction": (
                    f"{group_a} > {group_b}"
                    if median_ratio_a > median_ratio_b
                    else f"{group_a} < {group_b}"
                    if median_ratio_b > median_ratio_a
                    else f"{group_a} = {group_b}"
                ),
                "higher_group": group_a if median_a > median_b else group_b if median_b > median_a else "tie",
                "p_value": p_value,
                "cliffs_delta": cliffs_delta,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["q_value"] = bh_fdr(list(out["p_value"]))
    out = out.sort_values(["p_value", "q_value", "node"], na_position="last").reset_index(drop=True)
    out.insert(0, "rank", range(1, len(out) + 1))
    return out


@st.cache_data(ttl=300, show_spinner=False)
def coupling_aal_cross_comparison_summary(
    graph_csv: str,
    micro_csv: str,
    topology_metric: str,
    micro_metric: str,
    value_metric: str,
    top_n: int,
    label_csv: str,
) -> pd.DataFrame:
    frames = []
    for group_a, group_b in [("CN", "MCI"), ("CN", "AD"), ("MCI", "AD")]:
        rankings = coupling_aal_pairwise_rankings(
            graph_csv,
            micro_csv,
            topology_metric,
            micro_metric,
            value_metric,
            group_a,
            group_b,
            label_csv,
        )
        if rankings.empty:
            continue
        top = rankings.head(top_n).copy()
        top["pair_rank"] = top["rank"]
        top["direction"] = top.apply(
            lambda row: f"{group_a}>{group_b}" if row.get("higher_group") == group_a else f"{group_a}<{group_b}" if row.get("higher_group") == group_b else f"{group_a}={group_b}",
            axis=1,
        )
        frames.append(top)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    combined["p_value"] = pd.to_numeric(combined["p_value"], errors="coerce")
    combined["q_value"] = pd.to_numeric(combined["q_value"], errors="coerce")
    combined["nominal_p_lt_0_05"] = combined["p_value"] < 0.05
    combined["fdr_q_lt_0_05"] = combined["q_value"] < 0.05
    significant = combined[combined["nominal_p_lt_0_05"]].copy()
    if significant.empty:
        significant = combined.copy()
    summary = (
        significant.groupby(["node", "node_name", "atlas_label"], dropna=False)
        .agg(
            top_n_occurrences=("comparison", "nunique"),
            fdr_significant_comparisons=("fdr_q_lt_0_05", "sum"),
            best_p=("p_value", "min"),
            best_q=("q_value", "min"),
            comparisons=("comparison", lambda values: "; ".join(dict.fromkeys(map(str, values)))),
            directions=("direction", lambda values: "; ".join(dict.fromkeys(map(str, values)))),
            ratio_directions=("ratio_direction", lambda values: "; ".join(dict.fromkeys(map(str, values)))),
        )
        .reset_index()
    )
    summary = summary.sort_values(
        ["fdr_significant_comparisons", "top_n_occurrences", "best_p", "best_q"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)
    summary.insert(0, "rank", range(1, len(summary) + 1))
    return summary


def render_coupling_aal_definition(topology_metric: str, micro_metric: str, value_metric: str) -> None:
    st.markdown(
        "<div class='definition-card'>"
        "<h4>Regional AAL Coupling Calculation</h4>"
        f"<p>For each subject and each AAL3 node, the app joins {escape(metric_label(topology_metric))} from the node-graph table "
        f"with {escape(metric_label(micro_metric))} from the node-microstructure table.</p>"
        f"<p><b>T_i</b> is the selected topology value for AAL3 node <i>i</i> in one subject "
        f"({escape(metric_label(topology_metric))}). <b>M_i</b> is the selected microstructure value for the same AAL3 node "
        f"({escape(metric_label(micro_metric))}). The index <b>i</b> runs over the matched AAL3 regions within that subject.</p>"
        "<div class='metric-formula'>zT_i = (T_i - mean(T)) / sd(T); zM_i = (M_i - mean(M)) / sd(M)</div>"
        "<div class='metric-formula'>regional coupling index_i = zT_i * zM_i</div>"
        "<div class='metric-formula'>microstructure/topology ratio_i = M_i / T_i, shown when T_i != 0</div>"
        f"<p>Current ranking metric: <b>{escape(coupling_aal_value_label(value_metric))}</b>. "
        "Pairwise AAL tables use Mann-Whitney U p-values and BH q across AAL3 nodes.</p>"
        "</div>",
        unsafe_allow_html=True,
    )


def render_coupling_aal_panel(csv_files: list[Path], hide_visual_outliers: bool) -> bool:
    graph_path = next((p for p in csv_files if p.name == "live_node_graph_long.csv"), None)
    micro_path = next((p for p in csv_files if p.name == "live_node_microstructure_long.csv"), None)
    if graph_path is None or micro_path is None:
        return False
    try:
        graph_head = pd.read_csv(graph_path, nrows=1)
        micro_metrics = pd.read_csv(micro_path, usecols=["metric"])["metric"].dropna().astype(str).unique().tolist()
    except Exception:
        return False
    topology_options = [m for m, _ in COUPLING_NODAL_METRICS if m in graph_head.columns]
    micro_options = [m for m, _ in COUPLING_MICRO_METRICS if m in set(micro_metrics)]
    if not topology_options or not micro_options:
        return False

    st.subheader("Regional Coupling-AAL Ranking")
    render_network_coupling_breakdown("coupling-aal")
    c1, c2, c3, c4, c5, c6 = st.columns((0.95, 0.95, 1.15, 1.0, 1.0, 0.8))
    with c1:
        topology_metric = st.selectbox("Topology metric", topology_options, index=0, format_func=metric_label, key="coupling-aal-topology")
    with c2:
        micro_metric = st.selectbox("Microstructure metric", micro_options, index=0, format_func=metric_label, key="coupling-aal-micro")
    with c3:
        value_metric = st.selectbox("Regional value", COUPLING_AAL_VALUE_METRICS, index=0, format_func=coupling_aal_value_label, key="coupling-aal-value")
    pair_options = [("CN", "MCI"), ("CN", "AD"), ("MCI", "AD")]
    with c4:
        pair_label = st.selectbox("Group comparison", [f"{a} vs {b}" for a, b in pair_options], index=0, key="coupling-aal-pair")
    group_a, group_b = pair_label.split(" vs ")
    with c5:
        top_n = st.slider("Top N AALs", min_value=5, max_value=80, value=20, step=5, key="coupling-aal-topn")
    with c6:
        p_only = st.checkbox("Only p<0.05", value=True, key="coupling-aal-p-only")

    render_coupling_aal_definition(topology_metric, micro_metric, value_metric)
    all_pair_rankings = {
        pair: coupling_aal_pairwise_rankings(
            str(graph_path),
            str(micro_path),
            topology_metric,
            micro_metric,
            value_metric,
            pair[0],
            pair[1],
            str(AAL_LABEL_CSV),
        )
        for pair in pair_options
    }
    rankings = all_pair_rankings.get((group_a, group_b), pd.DataFrame())
    if rankings.empty:
        st.info("No regional AAL coupling ranking could be computed for the selected options.")
        return True
    display_rankings = rankings[pd.to_numeric(rankings["p_value"], errors="coerce") < 0.05].copy() if p_only else rankings.copy()
    display_rankings = display_rankings.head(top_n).copy()
    if display_rankings.empty:
        st.info("No AAL coupling rows have p<0.05 for this comparison; disable the p<0.05 filter to inspect the full p-value ranking.")
        return True

    st.markdown("#### Top AAL Coupling Regions")
    repeated_col, selected_col = st.columns((1.15, 1), gap="large")
    with repeated_col:
        st.markdown("##### Repeated Top-N Across Pairwise Comparisons")
        cross_summary = coupling_aal_cross_comparison_summary(
            str(graph_path),
            str(micro_path),
            topology_metric,
            micro_metric,
            value_metric,
            top_n,
            str(AAL_LABEL_CSV),
        )
        if cross_summary.empty:
            st.info("No repeated Top-N AAL coupling summary could be computed across CN vs MCI, CN vs AD, and MCI vs AD.")
        else:
            st.dataframe(
                cross_summary.head(top_n)[
                    [
                        "rank",
                        "node_name",
                        "atlas_label",
                        "top_n_occurrences",
                        "fdr_significant_comparisons",
                        "best_p",
                        "best_q",
                        "comparisons",
                        "directions",
                        "ratio_directions",
                    ]
                ],
                width="stretch",
                height=390,
                hide_index=True,
                column_config={
                    "rank": st.column_config.NumberColumn("Rank", format="%d"),
                    "top_n_occurrences": st.column_config.NumberColumn("Top-N hits", format="%d"),
                    "fdr_significant_comparisons": st.column_config.NumberColumn("BH q<0.05 hits", format="%d"),
                    "best_p": st.column_config.NumberColumn("Best p", format="%.3g"),
                    "best_q": st.column_config.NumberColumn("Best BH q", format="%.3g"),
                    "ratio_directions": "Ratio directions",
                },
            )
    with selected_col:
        st.markdown(f"##### {group_a} vs {group_b} Top {min(top_n, len(display_rankings))}")
        table = display_rankings.copy()
        table["fdr_q_lt_0_05"] = pd.to_numeric(table["q_value"], errors="coerce") < 0.05
        table["median_direction"] = table.apply(lambda row: row_median_direction(row, group_a, group_b), axis=1)
        st.dataframe(
            table[
                [
                    "rank",
                    "node_name",
                    "atlas_label",
                    f"median_{group_a}",
                    f"median_{group_b}",
                    "median_direction",
                    f"median_ratio_{group_a}",
                    f"median_ratio_{group_b}",
                    "ratio_direction",
                    "p_value",
                    "q_value",
                    "fdr_q_lt_0_05",
                ]
            ],
            width="stretch",
            height=390,
            hide_index=True,
            column_config={
                "rank": st.column_config.NumberColumn("Rank", format="%d"),
                f"median_{group_a}": st.column_config.NumberColumn(f"{group_a} median", format="%.4g"),
                f"median_{group_b}": st.column_config.NumberColumn(f"{group_b} median", format="%.4g"),
                f"median_ratio_{group_a}": st.column_config.NumberColumn(f"{group_a} ratio", format="%.3g"),
                f"median_ratio_{group_b}": st.column_config.NumberColumn(f"{group_b} ratio", format="%.3g"),
                "p_value": st.column_config.NumberColumn("p", format="%.3g"),
                "q_value": st.column_config.NumberColumn("BH q", format="%.3g"),
                "fdr_q_lt_0_05": st.column_config.CheckboxColumn("BH q<0.05"),
            },
        )

    chart_col, inference_col = st.columns((1.25, 1), gap="large")
    with chart_col:
        st.plotly_chart(ranked_node_chart(display_rankings, group_a, group_b), width="stretch", config={"displaylogo": False})
        st.caption(
            f"Computed live from `{graph_path.relative_to(ANALYSIS_ROOT).as_posix()}` and "
            f"`{micro_path.relative_to(ANALYSIS_ROOT).as_posix()}`."
        )
    with inference_col:
        st.markdown(node_pairwise_inference_html(f"{topology_metric}_vs_{micro_metric}_{value_metric}", all_pair_rankings), unsafe_allow_html=True)
    return True


def render_node_long_charts(
    section: Section,
    csv_files: list[Path],
    max_charts: int,
    hide_visual_outliers: bool,
) -> bool:
    long_path = long_node_table_for_section(section, csv_files)
    if long_path is None:
        return False
    df = _safe_read_csv(long_path)
    if df is None or df.empty or "group" not in df.columns or "node_name" not in df.columns:
        return False

    chart_specs: list[tuple[str, str, str]] = []
    if section.label == "Local / ROI DTI" and {"metric", "value"}.issubset(df.columns):
        metrics = [m for m in ("fa_mean", "md_mean", "ad_mean", "rd_mean") if m in set(df["metric"].astype(str))]
        for metric in metrics:
            tests = related_metric_file(csv_files, f"{metric}_node", "_tests.csv")
            if tests is not None:
                test_df = _safe_read_csv(tests)
                nodes = list(test_df.get("node_name", pd.Series(dtype=str)).astype(str).head(2)) if test_df is not None else []
            else:
                nodes = list(df.loc[df["metric"].astype(str) == metric, "node_name"].dropna().astype(str).unique()[:2])
            for node in nodes:
                chart_specs.append((metric, node, "value"))
    elif section.label == "Node Metrics":
        metrics = [m for m in ("strength", "degree", "nodal_eff") if m in df.columns]
        for metric in metrics:
            tests = related_metric_file(csv_files, f"{metric}_node", "_tests.csv")
            if tests is not None:
                test_df = _safe_read_csv(tests)
                nodes = list(test_df.get("node_name", pd.Series(dtype=str)).astype(str).head(2)) if test_df is not None else []
            else:
                nodes = list(df["node_name"].dropna().astype(str).unique()[:2])
            for node in nodes:
                chart_specs.append((metric, node, metric))

    if not chart_specs:
        return False

    st.subheader("Live CSV Charts")
    option_labels = [f"{metric_label(metric)} | {node}" for metric, node, _ in chart_specs]
    selected_labels = st.multiselect(
        "Choose live ROI/node charts",
        option_labels,
        default=option_labels[:max_charts],
        key=f"{section_key(section.label)}-node-live-{hash(tuple(option_labels))}",
    )
    selected_specs = [chart_specs[option_labels.index(label)] for label in selected_labels]
    for metric, node, value_col in selected_specs:
        node_df = df[df["node_name"].astype(str) == node].copy()
        if "metric" in node_df.columns:
            node_df = node_df[node_df["metric"].astype(str) == metric]
        data = subject_metric_frame(node_df, metric, value_col=value_col)
        if data.empty:
            continue
        desc = compute_descriptives_from_data(data)
        tests_path = related_metric_file(csv_files, f"{metric}_node", "_tests.csv")
        tests_df = _safe_read_csv(tests_path) if tests_path is not None else None
        node_stat_line = node_test_subtitle(tests_df, node)
        node_sig = False
        if tests_df is not None and not tests_df.empty and "kw_q" in tests_df.columns and "node_name" in tests_df.columns:
            node_rows = tests_df[tests_df["node_name"].astype(str) == str(node)].copy()
            if not node_rows.empty:
                node_sig = bool(pd.to_numeric(node_rows.iloc[0].get("kw_q"), errors="coerce") < 0.05)
        chart, counts, plotted_counts, removed_counts = live_distribution_chart(
            data,
            f"{metric_label(metric)} at {node}",
            hide_visual_outliers=hide_visual_outliers,
            stats_df=desc,
            subtitle_lines=[node_stat_line] if node_stat_line else [],
        )
        left, right = st.columns((1.65, 1), gap="large")
        with left:
            st.plotly_chart(chart, width="stretch", config={"displaylogo": False})
            st.caption(f"Live chart from `{long_path.relative_to(ANALYSIS_ROOT).as_posix()}`.")
            if sum(removed_counts.values()):
                st.caption(
                    "Visual-only 1.5-IQR trim active; statistics and CSV tables remain unchanged. "
                    f"Hidden points: {group_count_text(removed_counts)}."
                )
            render_dti_formula_block(metric)
        with right:
            finding = (
                f"{node} shows FDR-significant group variation for {metric_label(metric)}."
                if node_sig
                else f"No FDR-significant node-level group effect is detected for {metric_label(metric)} at {node}."
            )
            st.markdown(
                "<div class='result-card'>"
                f"<h4>{metric_label(metric)} at {node}</h4>"
                f"<div class='metric-explain'>{metric_explanation(section, metric)}</div>"
                f"<div class='chart-finding'>{finding}</div>"
                f"{median_summary_html(metric, desc)}"
                "</div>",
                unsafe_allow_html=True,
            )
    return True


def render_ranked_csv_charts(section: Section, csv_files: list[Path], max_charts: int) -> bool:
    ranked_paths = [
        p for p in csv_files
        if (
            p.name.endswith("_tests.csv")
            or p.name.endswith("_summary.csv")
            or p.name.startswith("edges_")
            or p.name.startswith("top_")
            or "disease_gradient" in p.name
            or p.name.endswith("_components.csv")
        )
    ]
    if not ranked_paths:
        return False
    st.subheader("Live CSV Charts")
    selected_paths = st.multiselect(
        "Choose ranked CSV charts",
        ranked_paths,
        default=ranked_paths[:max_charts],
        format_func=lambda p: p.relative_to(ANALYSIS_ROOT).as_posix(),
        key=f"{section_key(section.label)}-ranked-live-{hash(tuple(str(p) for p in ranked_paths))}",
    )
    for path in selected_paths:
        df = _safe_read_csv(path)
        if df is None or df.empty:
            continue
        q_col = next((c for c in ("kw_q", "welch_q", "disease_gradient_q", "bm_q", "mwu_q", "q") if c in df.columns), None)
        p_col = next((c for c in ("kw_p", "welch_p", "disease_gradient_p", "perm_p", "p_perm", "p") if c in df.columns), None)
        if q_col is None and p_col is None:
            continue
        score_col = q_col or p_col
        chart_df = df.copy()
        chart_df["score"] = pd.to_numeric(chart_df[score_col], errors="coerce")
        chart_df = chart_df[chart_df["score"].notna()].sort_values("score").head(30).copy()
        if chart_df.empty:
            continue
        chart_df["rank"] = range(1, len(chart_df) + 1)
        def feature_name(row: pd.Series) -> str:
            for col in ("metric", "node_name", "pair", "component_id", "node"):
                if col in row and pd.notna(row.get(col)) and str(row.get(col)).strip():
                    value = str(row.get(col)).strip()
                    return f"component {value}" if col == "component_id" else value
            if "i" in row and "j" in row:
                return f"{row.get('i')}-{row.get('j')}"
            return f"rank {row.get('rank')}"

        chart_df["feature"] = chart_df.apply(feature_name, axis=1)
        chart_df["minus_log10"] = -chart_df["score"].clip(lower=1e-300).map(math.log10)
        chart_df["score_label"] = chart_df["score"].map(lambda value: f"{'q' if q_col else 'p'}={format_p(value)}")
        tooltip = [
            alt.Tooltip("feature:N", title="Feature"),
            alt.Tooltip("score:Q", title=score_col, format=".3g"),
        ]
        if "n" in chart_df.columns:
            tooltip.append(alt.Tooltip("n:Q", title="N", format=",.0f"))
        base = alt.Chart(chart_df).encode(
            y=alt.Y("feature:N", sort="-x", axis=alt.Axis(title=None)),
            x=alt.X("minus_log10:Q", axis=alt.Axis(title=f"-log10({score_col})")),
            tooltip=tooltip,
        )
        bars = base.mark_bar(color="#5dade2", cornerRadiusTopRight=3, cornerRadiusBottomRight=3)
        labels = base.mark_text(
            align="left",
            baseline="middle",
            dx=5,
            color=CHART_TEXT,
            fontSize=11,
            fontWeight="bold",
        ).encode(text="score_label:N")
        chart = (bars + labels).properties(
            height=min(max(26 * len(chart_df), 180), 620),
            background=CHART_BACKGROUND,
            title=alt.TitleParams(
                text=path.stem.replace("_", " "),
                subtitle=[f"Ranked by {score_col}; higher bars indicate smaller p/q values."],
                anchor="start",
                color=CHART_TEXT,
                subtitleColor=CHART_MUTED,
                fontSize=16,
                subtitleFontSize=12,
            ),
        ).configure_axis(
            labelColor=CHART_TEXT,
            titleColor=CHART_TEXT,
            gridColor=CHART_GRID,
            domainColor=CHART_STROKE,
            tickColor=CHART_STROKE,
        ).configure_view(fill=CHART_BACKGROUND, stroke=CHART_STROKE)
        left, right = st.columns((1.65, 1), gap="large")
        with left:
            st.altair_chart(chart, width="stretch")
            st.caption(f"Live ranked chart from `{path.relative_to(ANALYSIS_ROOT).as_posix()}`.")
        with right:
            top = chart_df.iloc[0]
            score_name = "q" if q_col else "p"
            finding = f"Top ranked feature is {top['feature']} with {score_name}={format_p(top['score'])} in the refreshed CSV."
            st.markdown(f"<div class='chart-finding'>{finding}</div>", unsafe_allow_html=True)
    return True


def render_edr_exception_rankings(csv_files: list[Path]) -> bool:
    rankings_path = next((p for p in csv_files if p.name == "edr_exception_pairwise_roi_rankings.csv"), None)
    repeated_path = next((p for p in csv_files if p.name == "edr_exception_repeated_top_regions.csv"), None)
    edge_map_path = next((p for p in csv_files if p.name == "edr_exception_edge_group_map_fd_sum_len_mean.csv"), None)

    rendered = False
    rankings = _safe_read_csv(rankings_path) if rankings_path is not None else None
    repeated = _safe_read_csv(repeated_path) if repeated_path is not None else None
    if rankings is not None and not rankings.empty:
        rendered = True
        st.subheader("AAL-Level EDR Exception Ranking")
        metric_options = [m for m in EDR_NODE_RANK_METRICS if m in set(rankings["metric"].astype(str))]
        metric_options.extend(sorted(m for m in rankings["metric"].dropna().astype(str).unique() if m not in metric_options))
        comparison_options = [p for p in ("CN vs MCI", "CN vs AD", "MCI vs AD") if p in set(rankings["comparison"].astype(str))]
        if not comparison_options:
            comparison_options = sorted(rankings["comparison"].dropna().astype(str).unique())
        edr_section = next((section for section in SECTIONS if section.label == "EDR Exceptions"), Section("EDR Exceptions", "", ()))
        c1, c2, _ = st.columns((0.24, 0.22, 0.54))
        with c1:
            top_n = st.slider("Top N", min_value=5, max_value=80, value=20, step=5, key="edr-exception-topn")
        with c2:
            only_fdr = st.checkbox("Only q<0.05", value=True, key="edr-exception-rank-fdr")

        metric_tabs = st.tabs([metric_label(metric) for metric in metric_options])
        for metric_tab, metric in zip(metric_tabs, metric_options):
            with metric_tab:
                comparison_tabs = st.tabs(comparison_options)
                for comparison_tab, comparison in zip(comparison_tabs, comparison_options):
                    with comparison_tab:
                        st.markdown(
                            "<div class='edr-definition-box'>"
                            f"<h5>{escape(metric_label(metric))}</h5>"
                            f"<p>{escape(metric_explanation(edr_section, metric))}</p>"
                            "</div>",
                            unsafe_allow_html=True,
                        )
                        selected = rankings[
                            (rankings["metric"].astype(str) == metric)
                            & (rankings["comparison"].astype(str) == comparison)
                        ].copy()
                        selected["q_value"] = pd.to_numeric(selected.get("q_value"), errors="coerce")
                        selected["p_value"] = pd.to_numeric(selected.get("p_value"), errors="coerce")
                        if only_fdr:
                            selected = selected[selected["q_value"] < 0.05].copy()
                        selected = selected.sort_values(["q_value", "p_value"], na_position="last").head(top_n).copy()
                        if "rank" in selected.columns:
                            selected["rank"] = range(1, len(selected) + 1)
                        if repeated is None or repeated.empty:
                            rep = pd.DataFrame()
                        else:
                            rep = repeated[repeated["metric"].astype(str) == metric].copy() if "metric" in repeated.columns else repeated.copy()
                            rep = rep.head(top_n).copy()
                        left, right = st.columns((0.62, 0.38), gap="large")
                        with left:
                            st.markdown(f"#### {comparison} Top {top_n}: {metric_label(metric)}")
                            if selected.empty:
                                st.info("No AAL regions pass the current EDR exception ranking filter.")
                            else:
                                display_cols = [
                                    c
                                    for c in (
                                        "rank",
                                        "node_name",
                                        "atlas_label",
                                        "higher_group",
                                        "median_diff_a_minus_b",
                                        "p_value",
                                        "q_value",
                                        "cliffs_delta",
                                    )
                                    if c in selected.columns
                                ]
                                st.dataframe(
                                    selected[display_cols],
                                    width="stretch",
                                    height=430,
                                    hide_index=True,
                                    column_config={
                                        "rank": st.column_config.NumberColumn("Rank", format="%d"),
                                        "median_diff_a_minus_b": st.column_config.NumberColumn("Median diff", format="%.4g"),
                                        "p_value": st.column_config.NumberColumn("p", format="%.3g"),
                                        "q_value": st.column_config.NumberColumn("BH q", format="%.3g"),
                                        "cliffs_delta": st.column_config.NumberColumn("Cliff's delta", format="%.3f"),
                                    },
                                )
                                selected_finding = edr_rank_selected_inference(metric, comparison, selected, only_fdr)
                                if selected_finding:
                                    st.markdown("<div class='edr-table-gap'></div>", unsafe_allow_html=True)
                                    st.markdown(
                                        f"<div class='edr-inference-box'>{escape(selected_finding)}</div>",
                                        unsafe_allow_html=True,
                                    )
                            st.caption(f"Live AAL ranking from `{rankings_path.relative_to(ANALYSIS_ROOT).as_posix()}`.")
                        with right:
                            st.markdown("#### Repeated Top-N AAL Regions")
                            if rep.empty:
                                st.info("No repeated Top-N EDR exception summary is available yet.")
                            else:
                                display_cols = [
                                    c
                                    for c in (
                                        "rank",
                                        "node_name",
                                        "atlas_label",
                                        "top_n_occurrences",
                                        "fdr_significant_comparisons",
                                        "best_p",
                                        "best_q",
                                        "comparisons",
                                        "directions",
                                    )
                                    if c in rep.columns
                                ]
                                st.dataframe(
                                    rep[display_cols],
                                    width="stretch",
                                    height=430,
                                    hide_index=True,
                                    column_config={
                                        "rank": st.column_config.NumberColumn("Rank", format="%d"),
                                        "top_n_occurrences": st.column_config.NumberColumn("Top-N hits", format="%d"),
                                        "fdr_significant_comparisons": st.column_config.NumberColumn("BH q<0.05 hits", format="%d"),
                                        "best_p": st.column_config.NumberColumn("Best p", format="%.3g"),
                                        "best_q": st.column_config.NumberColumn("Best BH q", format="%.3g"),
                                    },
                                )
                                st.caption(
                                    "AAL regions repeated across CN vs MCI, CN vs AD, and MCI vs AD for the selected exception metric."
                                )
                                repeated_finding = edr_repeated_inference(metric, rep)
                                if repeated_finding:
                                    st.markdown("<div class='edr-table-gap'></div>", unsafe_allow_html=True)
                                    st.markdown(
                                        f"<div class='edr-inference-box'>{escape(repeated_finding)}</div>",
                                        unsafe_allow_html=True,
                                    )

    edge_map = _safe_read_csv(edge_map_path) if edge_map_path is not None else None
    if edge_map is not None and not edge_map.empty:
        rendered = True
        st.subheader("Edge-Level EDR Exception Maps")
        c1, c2, c3, c4 = st.columns((1.1, 0.9, 0.9, 0.8))
        comparisons = sorted(edge_map["comparison"].dropna().astype(str).unique()) if "comparison" in edge_map.columns else []
        classes = ["All"] + [c for c in ("SR", "MR", "LR") if "edr_class" in edge_map.columns and c in set(edge_map["edr_class"].astype(str))]
        with c1:
            comparison = st.selectbox("Edge comparison", comparisons, index=0, key="edr-edge-comparison") if comparisons else ""
        with c2:
            edr_class = st.selectbox("EDR class", classes, index=0, key="edr-edge-class")
        with c3:
            top_edges = st.slider("Top ROI pairs", min_value=5, max_value=80, value=20, step=5, key="edr-edge-topn")
        with c4:
            only_fdr = st.checkbox("Only q<0.05", value=True, key="edr-edge-fdr")
        edge_display = edge_map.copy()
        if comparison:
            edge_display = edge_display[edge_display["comparison"].astype(str) == comparison].copy()
        if edr_class != "All" and "edr_class" in edge_display.columns:
            edge_display = edge_display[edge_display["edr_class"].astype(str) == edr_class].copy()
        edge_display["q_value"] = pd.to_numeric(edge_display.get("q_value"), errors="coerce")
        edge_display["p_value"] = pd.to_numeric(edge_display.get("p_value"), errors="coerce")
        if only_fdr:
            edge_display = edge_display[edge_display["q_value"] < 0.05].copy()
        edge_display = edge_display.sort_values(["q_value", "p_value"], na_position="last").head(top_edges).copy()
        if edge_display.empty:
            st.info("No edge-level EDR exception rows pass the current filter.")
        else:
            comparison_groups = [part.strip() for part in str(comparison).split(" vs ") if part.strip()]
            group_rate_cols = [f"exception_rate_{group}" for group in comparison_groups if f"exception_rate_{group}" in edge_display.columns]
            group_strength_cols = [
                f"median_exception_strength_{group}"
                for group in comparison_groups
                if f"median_exception_strength_{group}" in edge_display.columns
            ]
            display_cols = [
                c
                for c in (
                    "roi_a",
                    "roi_b",
                    "edr_class",
                    "length_median_mm",
                    "length_valid_subjects",
                    *group_rate_cols,
                    *group_strength_cols,
                    "disease_direction",
                    "p_value",
                    "q_value",
                )
                if c in edge_display.columns
            ]
            st.dataframe(
                edge_display[display_cols],
                width="stretch",
                height=420,
                hide_index=True,
                column_config={
                    "length_median_mm": st.column_config.NumberColumn("Positive length median mm", format="%.2f"),
                    "length_valid_subjects": st.column_config.NumberColumn("Subjects with positive length", format="%d"),
                    "p_value": st.column_config.NumberColumn("p", format="%.3g"),
                    "q_value": st.column_config.NumberColumn("BH q", format="%.3g"),
                    **{
                        f"exception_rate_{group}": st.column_config.NumberColumn(f"{group} exception rate", format="%.3f")
                        for group in comparison_groups
                        if f"exception_rate_{group}" in edge_display.columns
                    },
                    **{
                        f"median_exception_strength_{group}": st.column_config.NumberColumn(
                            f"{group} median exception weight",
                            format="%.3g",
                        )
                        for group in comparison_groups
                        if f"median_exception_strength_{group}" in edge_display.columns
                    },
                },
            )
            st.caption(f"Live edge map from `{edge_map_path.relative_to(ANALYSIS_ROOT).as_posix()}`.")
    return rendered


def ml_target_label(target: str) -> str:
    return str(target).replace("_vs_", " vs ")


def ml_target_groups(target: str) -> tuple[str, ...]:
    groups = tuple(part.strip() for part in str(target).split("_vs_") if part.strip())
    return groups if groups else tuple(GROUP_ORDER)


def ml_feature_set_label(feature_set: str) -> str:
    mapping = {
        "comprehensive_structural": "Comprehensive structural",
        "comprehensive_structural_population": "Comprehensive structural + population",
        "structural_edr": "Structural EDR/LR/SR",
        "structural_edr_age_sex": "Structural EDR/LR/SR + age/sex",
        "clinical": "Clinical",
        "clinical_structural": "Clinical + structural",
    }
    return mapping.get(str(feature_set), metric_label(str(feature_set)))


def ml_model_label(model: str) -> str:
    mapping = {
        "multinomial_elastic_net": "Multinomial elastic-net",
        "xgboost_multiclass": "XGBoost multiclass",
        "extra_trees_multiclass": "Extra Trees multiclass",
        "random_forest_multiclass": "Random forest multiclass",
        "mlp_multiclass": "Neural net MLP multiclass",
        "elastic_net_regression": "Elastic-net regression",
        "xgboost_regression": "XGBoost regression",
        "extra_trees_regression": "Extra Trees regression",
        "random_forest_regression": "Random forest regression",
        "hist_gradient_boosting_regression": "Histogram gradient boosting regression",
        "mlp_regression": "Neural net MLP regression",
        "mlp_augmented_regression": "Augmented neural net MLP regression",
        "ridge_select": "Ridge regression",
        "elastic_net_select": "Elastic-net regression",
        "pca_ridge": "PCA ridge regression",
        "svr_rbf": "RBF SVR regression",
        "mlp_deep_regression": "Deep MLP regression",
        "mlp_deep": "Deep MLP",
        "mlp_deep_augmented": "Augmented deep MLP",
        "ft_transformer": "FT-Transformer",
        "tabm": "TabM",
        "tabpfn_regressor": "TabPFN regressor",
        "tabnet_regressor": "TabNet regressor",
        "ft_transformer_classifier": "FT-Transformer classifier",
        "tabm_classifier": "TabM classifier",
        "tabpfn_classifier": "TabPFN classifier",
        "tabnet_classifier": "TabNet classifier",
        "elastic_net_logistic": "Elastic-net logistic",
        "random_forest": "Random forest",
        "gradient_boosting": "Gradient boosting",
        "search_ridge_select": "Search ridge",
        "search_extra_trees_regression": "Search Extra Trees regression",
        "search_hist_gradient_boosting_regression": "Search histogram boosting regression",
        "search_svr_rbf": "Search RBF SVR regression",
        "search_xgboost_regression": "Search XGBoost regression",
        "search_multinomial_elastic_net": "Search multinomial elastic-net",
        "search_extra_trees_multiclass": "Search Extra Trees multiclass",
        "search_svc_rbf": "Search RBF SVC",
        "search_xgboost_multiclass": "Search XGBoost multiclass",
    }
    return mapping.get(str(model), metric_label(str(model)))


def ml_feature_category(feature: str) -> str:
    lower = str(feature).lower()
    if lower.startswith("population"):
        return "Population"
    if lower.startswith("global_dti") or "fa_mean" in lower or "md_mean" in lower or "rd_mean" in lower or "ad_mean" in lower:
        return "DTI microstructure"
    if lower.startswith("local_roi_dti"):
        return "AAL DTI"
    if lower.startswith("global_graph"):
        return "Global graph"
    if lower.startswith("node_metrics"):
        return "AAL graph"
    if lower.startswith("coupling"):
        return "Coupling"
    if lower.startswith("brain_age"):
        return "Brain age"
    if lower.startswith("advanced"):
        return "Advanced structural"
    if lower.startswith("delay"):
        return "Delay"
    if lower.startswith("matrix_summary"):
        return "Matrix summaries"
    if lower.startswith("raw_edge"):
        return "Raw edge features"
    if lower.startswith("edr_aal"):
        return "AAL EDR exceptions"
    if lower.startswith("edr_"):
        return "EDR fit"
    if "exception" in lower:
        return "Exception burden"
    if lower.startswith(("sr_", "mr_", "lr_", "sr_lr_")):
        return "LR/SR range"
    if "density" in lower or "w_mean" in lower or "w_sum" in lower:
        return "Range structure"
    return "Other"


def ml_feature_label_map() -> dict[str, str]:
    labels = aal_label_lookup(str(AAL_LABEL_CSV))
    if labels.empty or "node_name" not in labels.columns or "atlas_label" not in labels.columns:
        return {}
    return {str(row["node_name"]): str(row["atlas_label"]) for _, row in labels.iterrows()}


def ml_feature_display_name(feature: str, label_map: dict[str, str] | None = None) -> str:
    feature = str(feature)
    label_map = label_map or {}
    match = re.search(r"AAL_(\d{3})", feature)
    if match:
        aal = f"AAL_{match.group(1)}"
        region = label_map.get(aal, aal)
        prefix = feature[: match.start()].removeprefix("node_").strip("_")
        return f"{metric_label(prefix)} - {region} ({aal})"
    return metric_label(feature)


def apply_ml_plot_theme(fig: go.Figure) -> go.Figure:
    axis_style = {
        "title_font": {"color": CHART_TEXT, "size": 13},
        "tickfont": {"color": CHART_TEXT, "size": 12},
        "linecolor": CHART_STROKE,
        "zerolinecolor": CHART_STROKE,
        "gridcolor": CHART_GRID,
    }
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor=CHART_BACKGROUND,
        plot_bgcolor=CHART_BACKGROUND,
        font={"color": CHART_TEXT, "size": 13},
        title_font={"color": CHART_TEXT, "size": 18},
        legend={
            "font": {"color": CHART_TEXT, "size": 12},
            "bgcolor": "rgba(248,250,252,0.92)",
            "bordercolor": CHART_STROKE,
            "borderwidth": 1,
        },
    )
    fig.update_xaxes(**axis_style)
    fig.update_yaxes(**axis_style)
    fig.update_traces(
        selector={"type": "bar"},
        textfont={"color": CHART_TEXT, "size": 11},
        insidetextfont={"color": "#ffffff", "size": 11},
        outsidetextfont={"color": CHART_TEXT, "size": 11},
    )
    fig.update_traces(selector={"type": "heatmap"}, textfont={"color": CHART_TEXT, "size": 15})
    for trace in fig.data:
        colorbar = getattr(trace, "colorbar", None)
        if colorbar is not None:
            colorbar.tickfont = {"color": CHART_TEXT, "size": 11}
            colorbar.title = {"font": {"color": CHART_TEXT, "size": 12}}
    return fig


def ml_best_row(perf: pd.DataFrame) -> pd.Series | None:
    if perf.empty:
        return None
    work = perf.copy()
    for col in ("auroc_mean", "balanced_accuracy_mean"):
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")
    score_col = "auroc_mean" if "auroc_mean" in work.columns and work["auroc_mean"].notna().any() else "balanced_accuracy_mean"
    if score_col not in work.columns or not work[score_col].notna().any():
        return work.iloc[0]
    return work.loc[work[score_col].idxmax()]


def ml_performance_inference(target: str, perf: pd.DataFrame) -> str:
    best = ml_best_row(perf)
    if best is None:
        return "No cross-validated model performance is available for this target yet."
    auroc = format_number(best.get("auroc_mean"), digits=3)
    bacc = format_number(best.get("balanced_accuracy_mean"), digits=3)
    model = ml_model_label(str(best.get("model", "model")))
    feature_set = ml_feature_set_label(str(best.get("feature_set", "features")))
    return (
        f"Best current {ml_target_label(target)} diagnostic prediction is {model} with {feature_set}: "
        f"AUROC={auroc}, balanced accuracy={bacc}; treat this as exploratory cross-validation, not a deployment model."
    )


def ml_performance_figure(perf: pd.DataFrame, target: str) -> go.Figure:
    plot_df = perf.copy()
    plot_df["feature_model"] = [
        f"{ml_feature_set_label(fs)}<br>{ml_model_label(model)}"
        for fs, model in zip(plot_df.get("feature_set", ""), plot_df.get("model", ""))
    ]
    fig = go.Figure()
    metric_specs = [
        ("auroc_mean", "auroc_sd", "AUROC", "#2563eb"),
        ("balanced_accuracy_mean", "balanced_accuracy_sd", "Balanced accuracy", "#16a34a"),
    ]
    for value_col, sd_col, label, color in metric_specs:
        if value_col not in plot_df.columns:
            continue
        y = pd.to_numeric(plot_df[value_col], errors="coerce")
        err = pd.to_numeric(plot_df[sd_col], errors="coerce") if sd_col in plot_df.columns else None
        fig.add_trace(
            go.Bar(
                name=label,
                x=plot_df["feature_model"],
                y=y,
                error_y={"type": "data", "array": err, "visible": True} if err is not None else None,
                marker_color=color,
                text=[format_number(v, digits=3) for v in y],
                textposition="outside",
                cliponaxis=False,
            )
        )
    fig.update_layout(
        title=f"{ml_target_label(target)} model performance",
        barmode="group",
        height=440,
        paper_bgcolor=CHART_BACKGROUND,
        plot_bgcolor=CHART_BACKGROUND,
        font={"color": CHART_TEXT},
        legend={"orientation": "h", "y": 1.12, "x": 0},
        margin={"l": 55, "r": 30, "t": 90, "b": 120},
        yaxis={"range": [0, 1.05], "title": "Cross-validated score", "gridcolor": CHART_GRID},
        xaxis={"title": "", "tickangle": -18},
    )
    return apply_ml_plot_theme(fig)


def ml_importance_figure(importance: pd.DataFrame, top_n: int, label_map: dict[str, str]) -> go.Figure:
    plot_df = importance.copy()
    plot_df["importance"] = pd.to_numeric(plot_df["importance"], errors="coerce")
    plot_df = plot_df.dropna(subset=["importance"]).sort_values("importance", ascending=False).head(top_n)
    plot_df["display_feature"] = [ml_feature_display_name(f, label_map) for f in plot_df["feature"]]
    plot_df["category"] = [ml_feature_category(f) for f in plot_df["feature"]]
    plot_df = plot_df.sort_values("importance", ascending=True)
    colors = {
        "AAL exception burden": "#38bdf8",
        "EDR fit": "#a78bfa",
        "Exception burden": "#f97316",
        "LR/SR range": "#22c55e",
        "Range structure": "#14b8a6",
        "Population": "#facc15",
        "DTI microstructure": "#60a5fa",
        "AAL DTI": "#38bdf8",
        "Global graph": "#34d399",
        "AAL graph": "#22c55e",
        "Coupling": "#f59e0b",
        "Brain age": "#fb7185",
        "Advanced structural": "#c084fc",
        "Delay": "#a3e635",
        "Matrix summaries": "#2dd4bf",
        "Raw edge features": "#94a3b8",
        "AAL EDR exceptions": "#f97316",
        "Other": "#94a3b8",
    }
    fig = go.Figure()
    for category, sub in plot_df.groupby("category", sort=False):
        fig.add_trace(
            go.Bar(
                x=sub["importance"],
                y=sub["display_feature"],
                orientation="h",
                name=category,
                marker_color=colors.get(category, "#94a3b8"),
                text=[format_number(v, digits=3) for v in sub["importance"]],
                textposition="outside",
                cliponaxis=False,
            )
        )
    fig.update_layout(
        title="Feature importance / SHAP-style ranking",
        height=min(max(32 * len(plot_df) + 130, 360), 760),
        paper_bgcolor=CHART_BACKGROUND,
        plot_bgcolor=CHART_BACKGROUND,
        font={"color": CHART_TEXT},
        margin={"l": 260, "r": 50, "t": 75, "b": 55},
        xaxis={"title": "importance", "gridcolor": CHART_GRID},
        yaxis={"title": ""},
        legend={"orientation": "h", "y": 1.08, "x": 0},
    )
    return apply_ml_plot_theme(fig)


def ml_confusion_matrix_figure(predictions: pd.DataFrame, target: str) -> go.Figure:
    labels = list(ml_target_groups(target))
    for col in ("true_group", "predicted_group"):
        for value in predictions.get(col, pd.Series(dtype=str)).dropna().astype(str).unique():
            if value not in labels:
                labels.append(value)
    cm = pd.crosstab(predictions["true_group"], predictions["predicted_group"])
    cm = cm.reindex(index=labels, columns=labels, fill_value=0)
    text = cm.astype(int).astype(str).values
    fig = go.Figure(
        data=go.Heatmap(
            z=cm.values,
            x=cm.columns,
            y=cm.index,
            colorscale="Blues",
            text=text,
            texttemplate="%{text}",
            textfont={"color": CHART_TEXT, "size": 15},
            hovertemplate="True=%{y}<br>Predicted=%{x}<br>n=%{z}<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"{ml_target_label(target)} cross-validated confusion matrix",
        height=390,
        paper_bgcolor=CHART_BACKGROUND,
        plot_bgcolor=CHART_BACKGROUND,
        font={"color": CHART_TEXT},
        margin={"l": 70, "r": 30, "t": 70, "b": 60},
        xaxis={"title": "Predicted group"},
        yaxis={"title": "True group"},
    )
    return fig


def ml_probability_figure(predictions: pd.DataFrame, target: str) -> go.Figure | None:
    groups = ml_target_groups(target)
    if len(groups) < 2:
        return None
    positive = groups[-1]
    prob_col = f"prob_{positive}"
    if prob_col not in predictions.columns:
        return None
    plot_df = predictions.copy()
    plot_df[prob_col] = pd.to_numeric(plot_df[prob_col], errors="coerce")
    plot_df = plot_df.dropna(subset=[prob_col, "true_group"])
    if plot_df.empty:
        return None
    fig = go.Figure()
    for group, color in zip(GROUP_ORDER, GROUP_COLORS):
        sub = plot_df[plot_df["true_group"].astype(str) == group]
        if sub.empty:
            continue
        fig.add_trace(
            go.Histogram(
                x=sub[prob_col],
                name=group,
                opacity=0.58,
                marker_color=color,
                nbinsx=24,
            )
        )
    fig.update_layout(
        title=f"Predicted probability for {positive}",
        barmode="overlay",
        height=330,
        paper_bgcolor=CHART_BACKGROUND,
        plot_bgcolor=CHART_BACKGROUND,
        font={"color": CHART_TEXT},
        margin={"l": 55, "r": 25, "t": 65, "b": 55},
        xaxis={"title": f"P({positive})", "range": [0, 1], "gridcolor": CHART_GRID},
        yaxis={"title": "Subjects", "gridcolor": CHART_GRID},
    )
    return fig


def ml_feature_inventory(features: pd.DataFrame) -> pd.DataFrame:
    metadata_cols = {"subject_id", "group", "age", "sex", "phase"}
    feature_cols = [c for c in features.columns if c not in metadata_cols]
    rows = []
    for category in sorted({ml_feature_category(c) for c in feature_cols}):
        cols = [c for c in feature_cols if ml_feature_category(c) == category]
        rows.append(
            {
                "Feature class": category,
                "n_features": len(cols),
                "examples": ", ".join(cols[:4]),
            }
        )
    return pd.DataFrame(rows)


def render_ml_context_cards(features: pd.DataFrame | None) -> None:
    n_subjects = 0 if features is None or features.empty else int(features["subject_id"].nunique()) if "subject_id" in features.columns else len(features)
    html = (
        "<div class='definition-grid'>"
        "<div class='edr-definition-box'>"
        "<h5>What is predicted?</h5>"
        "<p>The target is one multiclass diagnosis label: CN, MCI, or AD. "
        "The model predicts the held-out subject's diagnostic group from structural connectome features only.</p>"
        "</div>"
        "<div class='edr-definition-box'>"
        "<h5>What features are used?</h5>"
        f"<p>The live feature table contains {n_subjects:,} subjects and one comprehensive structural feature set spanning "
        "DTI, graph metrics, node metrics, coupling, brain age, LR/SR, delay, advanced structural metrics, EDR exceptions, and raw matrix summaries.</p>"
        "</div>"
        "<div class='edr-definition-box'>"
        "<h5>Leakage control</h5>"
        "<p>The master cohort CSV is used only for subject labels/universe. ADNI phase, population columns, APOE, age, sex, weight, clinical scores, and group-derived p/q/Welch outputs are not predictors.</p>"
        "</div>"
        "<div class='edr-definition-box'>"
        "<h5>Feature selection</h5>"
        "<p>Missingness, sparse presence, near-zero variance, and high-correlation filters are unsupervised. Mutual-information selection and model regularization happen inside cross-validation.</p>"
        "</div>"
        "</div>"
    )
    st.markdown(html, unsafe_allow_html=True)


def ml_diagnostics_best_model(perf: pd.DataFrame) -> str:
    if perf.empty or "model" not in perf.columns:
        return ""
    work = perf.copy()
    for col in ("macro_auroc_ovr", "balanced_accuracy", "macro_f1"):
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")
    score_cols = [c for c in ("macro_auroc_ovr", "balanced_accuracy", "macro_f1") if c in work.columns]
    if not score_cols:
        return str(work["model"].iloc[0])
    work["_score"] = work[score_cols].mean(axis=1, skipna=True)
    return str(work.loc[work["_score"].idxmax(), "model"])


def ml_diagnostics_performance_figure(perf: pd.DataFrame) -> go.Figure:
    plot_df = perf.copy()
    plot_df["Model"] = [ml_model_label(m) for m in plot_df["model"].astype(str)]
    fig = go.Figure()
    specs = (
        ("balanced_accuracy", "Balanced accuracy", "#16a34a"),
        ("macro_auroc_ovr", "Macro AUROC", "#2563eb"),
        ("macro_f1", "Macro F1", "#f97316"),
    )
    for col, label, color in specs:
        if col not in plot_df.columns:
            continue
        values = pd.to_numeric(plot_df[col], errors="coerce")
        fig.add_trace(
            go.Bar(
                x=plot_df["Model"],
                y=values,
                name=label,
                marker_color=color,
                text=[format_number(v, digits=3) for v in values],
                textposition="outside",
                cliponaxis=False,
            )
        )
    fig.update_layout(
        title="Multiclass CN/MCI/AD model performance",
        barmode="group",
        height=430,
        paper_bgcolor=CHART_BACKGROUND,
        plot_bgcolor=CHART_BACKGROUND,
        font={"color": CHART_TEXT},
        margin={"l": 55, "r": 30, "t": 80, "b": 75},
        legend={"orientation": "h", "y": 1.12, "x": 0},
        yaxis={"title": "Cross-validated score", "range": [0, 1.05], "gridcolor": CHART_GRID},
        xaxis={"title": ""},
    )
    return apply_ml_plot_theme(fig)


def ml_diagnostics_performance_inference(perf: pd.DataFrame) -> str:
    if perf.empty:
        return "No multiclass diagnostic model performance is available yet."
    model = ml_diagnostics_best_model(perf)
    row = perf[perf["model"].astype(str).eq(model)].iloc[0]
    return (
        f"Best multiclass CN/MCI/AD model is {ml_model_label(model)} with "
        f"balanced accuracy={format_number(row.get('balanced_accuracy'), 3)}, "
        f"macro AUROC={format_number(row.get('macro_auroc_ovr'), 3)}, and "
        f"macro F1={format_number(row.get('macro_f1'), 3)}."
    )


def ml_diagnostics_confusion_figure(confusion: pd.DataFrame, model: str) -> go.Figure:
    labels = list(GROUP_ORDER)
    sub = confusion[confusion["model"].astype(str).eq(model)].copy()
    if sub.empty:
        mat = pd.DataFrame(0, index=labels, columns=labels)
    else:
        mat = sub.pivot_table(index="true_group", columns="predicted_group", values="n", aggfunc="sum", fill_value=0)
        mat = mat.reindex(index=labels, columns=labels, fill_value=0)
    fig = go.Figure(
        data=go.Heatmap(
            z=mat.to_numpy(),
            x=mat.columns,
            y=mat.index,
            colorscale="Blues",
            text=mat.astype(int).astype(str).to_numpy(),
            texttemplate="%{text}",
            textfont={"color": CHART_TEXT, "size": 16},
            hovertemplate="True=%{y}<br>Predicted=%{x}<br>n=%{z}<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"{ml_model_label(model)} confusion matrix",
        height=390,
        paper_bgcolor=CHART_BACKGROUND,
        plot_bgcolor=CHART_BACKGROUND,
        font={"color": CHART_TEXT},
        margin={"l": 70, "r": 30, "t": 70, "b": 60},
        xaxis={"title": "Predicted group"},
        yaxis={"title": "True group"},
    )
    return apply_ml_plot_theme(fig)


def ml_diagnostics_roc_figure(roc: pd.DataFrame, model: str) -> go.Figure:
    fig = go.Figure()
    sub = roc[roc["model"].astype(str).eq(model)].copy() if not roc.empty and "model" in roc.columns else pd.DataFrame()
    for group, color in zip(GROUP_ORDER, GROUP_COLORS):
        curve = sub[sub["class"].astype(str).eq(group)].copy() if not sub.empty else pd.DataFrame()
        if curve.empty:
            continue
        curve["fpr"] = pd.to_numeric(curve["fpr"], errors="coerce")
        curve["tpr"] = pd.to_numeric(curve["tpr"], errors="coerce")
        fig.add_trace(
            go.Scatter(
                x=curve["fpr"],
                y=curve["tpr"],
                mode="lines",
                name=f"{group} one-vs-rest",
                line={"color": color, "width": 3},
            )
        )
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="chance", line={"color": "#64748b", "dash": "dash"}))
    fig.update_layout(
        title=f"{ml_model_label(model)} one-vs-rest ROC curves",
        height=390,
        paper_bgcolor=CHART_BACKGROUND,
        plot_bgcolor=CHART_BACKGROUND,
        font={"color": CHART_TEXT},
        margin={"l": 60, "r": 30, "t": 70, "b": 60},
        xaxis={"title": "False positive rate", "range": [0, 1], "gridcolor": CHART_GRID},
        yaxis={"title": "True positive rate", "range": [0, 1], "gridcolor": CHART_GRID},
        legend={"orientation": "h", "y": 1.1, "x": 0},
    )
    return apply_ml_plot_theme(fig)


def ml_diagnostics_importance_figure(importance: pd.DataFrame, model: str, top_n: int, value_col: str = "importance") -> go.Figure:
    sub = importance[importance["model"].astype(str).eq(model)].copy() if not importance.empty and "model" in importance.columns else pd.DataFrame()
    if sub.empty or value_col not in sub.columns:
        return go.Figure()
    sub[value_col] = pd.to_numeric(sub[value_col], errors="coerce")
    sub = sub.dropna(subset=[value_col]).sort_values(value_col, ascending=False).head(top_n)
    if "feature_group" not in sub.columns:
        sub["feature_group"] = [ml_feature_category(f) for f in sub["feature"]]
    sub["Feature"] = [ml_feature_display_name(f, ml_feature_label_map()) for f in sub["feature"]]
    sub = sub.sort_values(value_col, ascending=True)
    colors = {
        "Demographics/population": "#facc15",
        "Global DTI": "#60a5fa",
        "Local ROI DTI": "#38bdf8",
        "Global graph": "#34d399",
        "Node metrics": "#22c55e",
        "Coupling": "#f59e0b",
        "Brain age": "#fb7185",
        "LR/SR and delay": "#a3e635",
        "Delay": "#bef264",
        "Advanced structural": "#c084fc",
        "EDR exceptions": "#f97316",
        "AAL EDR exceptions": "#fb923c",
        "Whole-connectome distribution summaries": "#2dd4bf",
        "Raw high-variance edges": "#94a3b8",
    }
    fig = go.Figure()
    for group, group_df in sub.groupby("feature_group", sort=False):
        fig.add_trace(
            go.Bar(
                x=group_df[value_col],
                y=group_df["Feature"],
                orientation="h",
                name=str(group),
                marker_color=colors.get(str(group), "#94a3b8"),
                text=[format_number(v, digits=3) for v in group_df[value_col]],
                textposition="outside",
                cliponaxis=False,
            )
        )
    title = "TreeSHAP contributions" if value_col == "mean_abs_contribution" else "Feature importance"
    fig.update_layout(
        title=f"{ml_model_label(model)} {title}",
        height=min(max(32 * len(sub) + 130, 360), 760),
        paper_bgcolor=CHART_BACKGROUND,
        plot_bgcolor=CHART_BACKGROUND,
        font={"color": CHART_TEXT},
        margin={"l": 290, "r": 60, "t": 75, "b": 55},
        xaxis={"title": value_col.replace("_", " "), "gridcolor": CHART_GRID},
        yaxis={"title": ""},
        legend={"orientation": "h", "y": 1.08, "x": 0},
    )
    return apply_ml_plot_theme(fig)


def score_prediction_best_model(perf: pd.DataFrame, target: str, target_mode: str) -> str:
    if perf.empty:
        return ""
    perf = add_model_identity(perf)
    work = perf[
        perf["target"].astype(str).eq(str(target))
        & perf["target_mode"].astype(str).eq(str(target_mode))
        & perf.get("status", pd.Series("", index=perf.index)).astype(str).eq("ok")
    ].copy()
    if work.empty:
        return ""
    for col in ("mae", "rmse", "r2", "spearman_r"):
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")
    work["_rank_r2"] = work["r2"].rank(method="min", ascending=False) if "r2" in work.columns else 1
    work["_rank_spearman"] = work["spearman_r"].rank(method="min", ascending=False) if "spearman_r" in work.columns else 1
    work["_score"] = work[["_rank_r2", "_rank_spearman"]].mean(axis=1, skipna=True)
    return str(work.loc[work["_score"].idxmin(), "model_key"])


def clinical_outcome_production_perf(perf: pd.DataFrame, target: str, target_mode: str, *, task_type: str) -> pd.DataFrame:
    if perf is None or perf.empty:
        return pd.DataFrame()
    work = add_model_identity(filter_no_explicit_age_policy(perf))
    if "target" in work.columns:
        work = work[work["target"].astype(str).eq(str(target))].copy()
    work = work[
        work["target_mode"].astype(str).eq(str(target_mode))
        & work.get("status", pd.Series("", index=work.index)).astype(str).eq("ok")
    ].copy()
    if work.empty:
        return work
    if task_type == "classification":
        sort_cols = [col for col in ("accuracy", "balanced_accuracy", "macro_f1") if col in work.columns]
        for col in sort_cols:
            work[col] = pd.to_numeric(work[col], errors="coerce")
        if sort_cols:
            work = work.sort_values(sort_cols, ascending=[False] * len(sort_cols), na_position="last")
    else:
        for col in ("r2", "spearman_r", "rmse"):
            if col in work.columns:
                work[col] = pd.to_numeric(work[col], errors="coerce")
        sort_cols = [col for col in ("r2", "spearman_r") if col in work.columns]
        if "rmse" in work.columns:
            sort_cols.append("rmse")
        if sort_cols:
            ascending = [False if col != "rmse" else True for col in sort_cols]
            work = work.sort_values(sort_cols, ascending=ascending, na_position="last")
    return work.head(1).copy()


def score_prediction_performance_figure(perf: pd.DataFrame, target: str, target_mode: str) -> go.Figure:
    perf = add_model_identity(perf)
    plot_df = perf[
        perf["target"].astype(str).eq(str(target))
        & perf["target_mode"].astype(str).eq(str(target_mode))
        & perf.get("status", pd.Series("", index=perf.index)).astype(str).eq("ok")
    ].copy()
    for col in ("r2", "mae", "rmse", "spearman_r"):
        if col in plot_df.columns:
            plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce")
    sort_cols = [col for col in ("r2", "spearman_r") if col in plot_df.columns]
    if sort_cols:
        plot_df = plot_df.sort_values(sort_cols, ascending=[False] * len(sort_cols), na_position="last").head(8)
    plot_df["Model"] = plot_df.get("model_display", pd.Series(dtype=str)).astype(str)
    fig = go.Figure()
    specs = (
        ("mae", "MAE", "#2563eb"),
        ("rmse", "RMSE", "#f97316"),
    )
    for col, label, color in specs:
        if col not in plot_df.columns:
            continue
        values = pd.to_numeric(plot_df[col], errors="coerce")
        fig.add_trace(
            go.Bar(
                x=plot_df["Model"],
                y=values,
                name=label,
                marker_color=color,
                text=[format_number(v, digits=3) for v in values],
                textposition="outside",
                cliponaxis=False,
            )
        )
    fig.update_layout(
        title=f"{metric_label(target)} score prediction error",
        barmode="group",
        height=390,
        paper_bgcolor=CHART_BACKGROUND,
        plot_bgcolor=CHART_BACKGROUND,
        font={"color": CHART_TEXT},
        margin={"l": 55, "r": 30, "t": 75, "b": 80},
        legend={"orientation": "h", "y": 1.12, "x": 0},
        yaxis={"title": "Lower is better", "gridcolor": CHART_GRID},
        xaxis={"title": ""},
    )
    return apply_ml_plot_theme(fig)


def score_prediction_true_pred_figure(predictions: pd.DataFrame, target: str, target_mode: str, model: str) -> go.Figure:
    predictions = add_model_identity(predictions)
    plot_df = predictions[
        predictions["target"].astype(str).eq(str(target))
        & predictions["target_mode"].astype(str).eq(str(target_mode))
        & predictions["model_key"].astype(str).eq(str(model))
    ].copy()
    plot_df["true_score"] = pd.to_numeric(plot_df["true_score"], errors="coerce")
    plot_df["predicted_score"] = pd.to_numeric(plot_df["predicted_score"], errors="coerce")
    fig = go.Figure()
    color_map = dict(zip(GROUP_ORDER, GROUP_COLORS))
    for group in GROUP_ORDER:
        sub = plot_df[plot_df["group"].astype(str).eq(group)]
        if sub.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=sub["true_score"],
                y=sub["predicted_score"],
                mode="markers",
                name=group,
                marker={"color": color_map.get(group, "#64748b"), "size": 8, "opacity": 0.78},
                hovertemplate="Subject=%{customdata}<br>True=%{x:.2f}<br>Predicted=%{y:.2f}<extra></extra>",
                customdata=sub["subject_id"],
            )
        )
    finite = plot_df[["true_score", "predicted_score"]].replace([np.inf, -np.inf], np.nan).dropna()
    if not finite.empty:
        lo = float(np.nanmin(finite.to_numpy()))
        hi = float(np.nanmax(finite.to_numpy()))
        fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="ideal", line={"color": "#64748b", "dash": "dash"}))
    fig.update_layout(
        title="MMSE true vs predicted",
        height=430,
        paper_bgcolor=CHART_BACKGROUND,
        plot_bgcolor=CHART_BACKGROUND,
        font={"color": CHART_TEXT},
        margin={"l": 60, "r": 30, "t": 75, "b": 60},
        xaxis={"title": "True MMSE", "gridcolor": CHART_GRID},
        yaxis={"title": "Predicted MMSE", "gridcolor": CHART_GRID},
        legend={"orientation": "h", "y": 1.12, "x": 0},
    )
    return apply_ml_plot_theme(fig)


def score_prediction_residual_figure(predictions: pd.DataFrame, target: str, target_mode: str, model: str) -> go.Figure:
    predictions = add_model_identity(predictions)
    plot_df = predictions[
        predictions["target"].astype(str).eq(str(target))
        & predictions["target_mode"].astype(str).eq(str(target_mode))
        & predictions["model_key"].astype(str).eq(str(model))
    ].copy()
    plot_df["residual"] = pd.to_numeric(plot_df["residual"], errors="coerce")
    fig = go.Figure()
    color_map = dict(zip(GROUP_ORDER, GROUP_COLORS))
    for group in GROUP_ORDER:
        sub = plot_df[plot_df["group"].astype(str).eq(group)]
        if sub.empty:
            continue
        fig.add_trace(
            go.Box(
                y=sub["residual"],
                name=group,
                marker_color=color_map.get(group, "#64748b"),
                boxpoints="all",
                jitter=0.25,
                pointpos=0,
            )
        )
    fig.add_hline(y=0, line={"color": "#64748b", "dash": "dash"})
    fig.update_layout(
        title=f"{metric_label(target)} residuals by group",
        height=360,
        paper_bgcolor=CHART_BACKGROUND,
        plot_bgcolor=CHART_BACKGROUND,
        font={"color": CHART_TEXT},
        margin={"l": 60, "r": 30, "t": 75, "b": 55},
        yaxis={"title": "True - predicted", "gridcolor": CHART_GRID},
        xaxis={"title": ""},
    )
    return apply_ml_plot_theme(fig)


def filter_active_score_targets(df: pd.DataFrame | None) -> pd.DataFrame | None:
    if df is None or df.empty or "target" not in df.columns:
        return df
    return df[df["target"].astype(str).isin(ACTIVE_SCORE_TARGETS)].copy()


def filter_regression_score_targets(df: pd.DataFrame | None) -> pd.DataFrame | None:
    if df is None or df.empty or "target" not in df.columns:
        return df
    return df[df["target"].astype(str).isin(REGRESSION_SCORE_TARGETS)].copy()


def filter_structural_score_policy(df: pd.DataFrame | None) -> pd.DataFrame | None:
    if df is None or df.empty or "feature_policy" not in df.columns:
        return df
    return df[df["feature_policy"].astype(str).eq("structural_connectome_only")].copy()


def filter_no_explicit_age_policy(df: pd.DataFrame | None) -> pd.DataFrame | None:
    if df is None or df.empty or "feature_policy" not in df.columns:
        return df
    policy = df["feature_policy"].astype(str)
    explicit_age = policy.eq("age_only_experimental") | policy.str.endswith("_plus_age_experimental") | policy.str.contains(
        "plus_age_experimental", regex=False
    )
    return df[~explicit_age].copy()


MODEL_KEY_SEPARATOR = "|||"


def add_model_identity(df: pd.DataFrame | None, default_feature_policy: str = "current_dashboard") -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame() if df is None else df
    out = df.copy()
    if "feature_policy" not in out.columns:
        out["feature_policy"] = default_feature_policy
    out["feature_policy"] = out["feature_policy"].fillna(default_feature_policy).astype(str)
    if "model" not in out.columns:
        out["model"] = ""
    out["model"] = out["model"].fillna("").astype(str)
    out["model_key"] = out["feature_policy"] + MODEL_KEY_SEPARATOR + out["model"]
    out["model_display"] = [
        ml_model_label(model) if policy == "current_dashboard" else f"{ml_model_label(model)} | {metric_label(policy)}"
        for policy, model in zip(out["feature_policy"], out["model"])
    ]
    return out


def model_display_map(perf: pd.DataFrame) -> dict[str, str]:
    if perf is None or perf.empty or "model_key" not in perf.columns:
        return {}
    rows = perf[["model_key", "model_display"]].drop_duplicates("model_key", keep="first")
    return dict(zip(rows["model_key"].astype(str), rows["model_display"].astype(str)))


def selected_model_rows(df: pd.DataFrame | None, selected_model_key: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    work = df.copy()
    if "model_key" in work.columns:
        return work[work["model_key"].astype(str).eq(str(selected_model_key))].copy()
    return work[work["model"].astype(str).eq(str(selected_model_key))].copy()


def normalize_cdr_class_label(value: object) -> str | None:
    text = str(value).strip()
    if text in CDR_CLASS_ORDER:
        return text
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return None
    return cdr_class_from_value(numeric)


def cdr_confusion_from_predictions(predictions: pd.DataFrame | None) -> pd.DataFrame:
    if predictions is None or predictions.empty:
        return pd.DataFrame()
    needed = {"target_mode", "true_class", "predicted_class"}
    if not needed.issubset(predictions.columns):
        return pd.DataFrame()
    work = add_model_identity(predictions)
    work["true_class"] = work["true_class"].map(normalize_cdr_class_label)
    work["predicted_class"] = work["predicted_class"].map(normalize_cdr_class_label)
    work = work.dropna(subset=["true_class", "predicted_class"])
    group_cols = ["target", "target_mode", "feature_policy", "model", "model_key", "model_display", "true_class", "predicted_class"]
    group_cols = [col for col in group_cols if col in work.columns]
    return work.groupby(group_cols, dropna=False).size().reset_index(name="n")


def cdr_class_metrics_from_predictions(predictions: pd.DataFrame | None) -> pd.DataFrame:
    if predictions is None or predictions.empty:
        return pd.DataFrame()
    work = add_model_identity(predictions)
    if not {"target_mode", "true_class", "predicted_class"}.issubset(work.columns):
        return pd.DataFrame()
    work["true_class"] = work["true_class"].map(normalize_cdr_class_label)
    work["predicted_class"] = work["predicted_class"].map(normalize_cdr_class_label)
    work = work.dropna(subset=["true_class", "predicted_class"])
    rows = []
    for keys, sub in work.groupby(["target", "target_mode", "feature_policy", "model", "model_key", "model_display"], dropna=False):
        target, target_mode, feature_policy, model, model_key, model_display = keys
        y_true = sub["true_class"].astype(str).to_numpy()
        y_pred = sub["predicted_class"].astype(str).to_numpy()
        for label in CDR_CLASS_ORDER:
            tp = int(((y_true == label) & (y_pred == label)).sum())
            fp = int(((y_true != label) & (y_pred == label)).sum())
            fn = int(((y_true == label) & (y_pred != label)).sum())
            tn = int(((y_true != label) & (y_pred != label)).sum())
            precision = tp / (tp + fp) if (tp + fp) else np.nan
            recall = tp / (tp + fn) if (tp + fn) else np.nan
            specificity = tn / (tn + fp) if (tn + fp) else np.nan
            f1 = (2 * precision * recall / (precision + recall)) if np.isfinite(precision) and np.isfinite(recall) and (precision + recall) else np.nan
            rows.append(
                {
                    "target": target,
                    "target_mode": target_mode,
                    "feature_policy": feature_policy,
                    "model": model,
                    "model_key": model_key,
                    "model_display": model_display,
                    "class": label,
                    "support": int((y_true == label).sum()),
                    "precision": precision,
                    "recall_sensitivity": recall,
                    "specificity": specificity,
                    "f1": f1,
                }
            )
    return pd.DataFrame(rows)


def render_yellow_center_heading(text: str) -> None:
    st.markdown(
        f"<h2 style='text-align:center;color:#facc15;letter-spacing:0;margin:0.8rem 0 1rem 0;'>{escape(text)}</h2>",
        unsafe_allow_html=True,
    )


def render_yellow_left_heading(text: str) -> None:
    st.markdown(
        f"<h3 style='text-align:left;color:#facc15;letter-spacing:0;margin:1rem 0 0.55rem 0;font-size:1.18rem;font-weight:900;'>{escape(text)}</h3>",
        unsafe_allow_html=True,
    )


def score_mode_display(mode: str) -> str:
    if mode == PRIMARY_SCORE_MODE:
        return "Primary scan-aligned"
    if mode == HISTORY_SCORE_MODE:
        return "All-history sensitivity"
    return metric_label(mode)


def score_importance_for_selection(
    importance: pd.DataFrame | None,
    target: str,
    target_mode: str,
    model: str,
    method: str | None = None,
) -> pd.DataFrame:
    if importance is None or importance.empty:
        return pd.DataFrame()
    imp = importance[
        importance["target"].astype(str).eq(str(target))
        & importance["target_mode"].astype(str).eq(str(target_mode))
        & importance["model"].astype(str).eq(str(model))
    ].copy()
    if imp.empty:
        return imp
    if "importance_method" in imp.columns:
        if method is None:
            methods = imp["importance_method"].dropna().astype(str).drop_duplicates().tolist()
            preferred = ("tree_feature_importance", "xgboost_treeshap_abs_contribution", "abs_elastic_net_coef")
            method = next((candidate for candidate in preferred if candidate in methods), methods[0] if methods else None)
        if method is not None:
            imp = imp[imp["importance_method"].astype(str).eq(str(method))].copy()
    if "rank" in imp.columns:
        imp["rank"] = pd.to_numeric(imp["rank"], errors="coerce")
        imp = imp.sort_values(["rank", "importance"], ascending=[True, False], na_position="last")
    elif "importance" in imp.columns:
        imp["importance"] = pd.to_numeric(imp["importance"], errors="coerce")
        imp = imp.sort_values("importance", ascending=False, na_position="last")
    return imp


def feature_card_group(row: pd.Series) -> str:
    group = str(row.get("feature_group", "") or "")
    feature = str(row.get("feature", "") or "")
    lower = f"{group} {feature}".lower()
    if "clinical" in lower:
        return "Clinical"
    if "metadata" in lower:
        return "Metadata"
    if "coupling" in lower:
        return "Coupling"
    if "node_metrics" in lower or "node metrics" in lower or "nodal" in lower:
        return "Node metrics"
    if "matrix_summary" in lower or "matrix summaries" in lower:
        return "Whole-connectome distribution summaries"
    if "raw_edge" in lower:
        return "AAL edges"
    if "aal" in lower or "roi" in lower:
        return "AAL / ROI"
    if group:
        return group
    return "Other"


def render_top_predictor_scroll(importance: pd.DataFrame, height_px: int = 360) -> None:
    if importance.empty:
        st.info("Top predictors are not available for the selected target/model.")
        return
    rows = []
    for _, row in importance.head(100).iterrows():
        rank = int(pd.to_numeric(pd.Series([row.get("rank")]), errors="coerce").fillna(len(rows) + 1).iloc[0])
        feature = escape(str(row.get("feature", "")))
        group = escape(feature_card_group(row))
        value = pd.to_numeric(pd.Series([row.get("importance")]), errors="coerce").iloc[0]
        value_text = f"{value:.4g}" if pd.notna(value) else ""
        rows.append(
            "<div style='padding:0.45rem 0;border-bottom:1px solid #1f2937;'>"
            f"<div style='font-weight:700;color:#facc15;'>#{rank} {feature}</div>"
            f"<div style='font-size:0.83rem;color:#cbd5e1;'>{group} {value_text}</div>"
            "</div>"
        )
    st.markdown(
        f"<div style='height:{height_px}px;overflow-y:auto;border:1px solid #334155;border-radius:8px;padding:0.55rem;background:#0f172a;'>"
        + "".join(rows)
        + "</div>",
        unsafe_allow_html=True,
    )


def _aal_region_text(node_id: str, label_map: dict[str, str]) -> str:
    aal = f"AAL_{int(node_id):03d}"
    return f"{label_map.get(aal, aal)} ({aal})"


def _diffusion_metric_meaning(metric: str) -> str:
    mapping = {
        "fa_mean": "fractional anisotropy, a directional white-matter organization marker",
        "md_mean": "mean diffusivity, an overall water-diffusion marker that often rises with tissue disruption",
        "rd_mean": "radial diffusivity, a diffusion marker often interpreted in relation to myelin integrity",
        "ad_mean": "axial diffusivity, a diffusion marker along the main fiber direction",
        "count": "streamline count, a tractography-derived connection presence or weight proxy",
        "fd_sum": "summed fixel-density/streamline-derived structural connection strength",
        "len_mean": "mean tract length between connected AAL3 regions",
    }
    return mapping.get(metric, metric_label(metric).lower())


def _edge_stat_meaning(stat: str) -> str:
    mapping = {
        "edge_mean": "average across positive connectome edges",
        "edge_median": "median across positive connectome edges",
        "edge_q10": "lower-tail 10th percentile across positive connectome edges",
        "edge_q90": "upper-tail 90th percentile across positive connectome edges",
        "edge_sd": "between-edge variability",
        "edge_iqr": "between-edge interquartile spread",
        "edge_min": "minimum observed positive edge value",
        "edge_max": "maximum observed positive edge value",
        "edge_sum": "total summed edge value",
        "positive_density": "fraction of possible AAL3 edges that are present",
        "positive_edge_count": "number of positive AAL3 edges",
        "n_edges": "number of valid positive AAL3 edges",
    }
    return mapping.get(stat, metric_label(stat).lower())


def feature_neuroscience_label(row: pd.Series) -> str:
    feature = str(row.get("feature", "")).lower()
    group = str(row.get("feature_group", "") or feature_card_group(row)).lower()
    text = f"{group} {feature}"
    if "coupling" in text:
        return "Coupling"
    if "nodal_eff" in text or "efficiency" in text or "global_eff" in text:
        return "Efficiency"
    if "degree" in text or "strength" in text:
        return "Hubness" if "node" in text else "Connectivity"
    if "fa_mean" in text:
        return "Microstructure"
    if "md_mean" in text:
        return "Diffusivity"
    if "rd_mean" in text:
        return "Myelin"
    if "ad_mean" in text:
        return "Axonal"
    if "density" in text or "count" in text or "n_edges" in text:
        return "Density"
    if "len" in text or "length" in text:
        return "Length"
    if "delay" in text or "lr_sr" in text:
        return "Delay"
    if "edr" in text or "exception" in text:
        return "Vulnerability"
    if "brain_age" in text or "age" in text:
        return "Aging"
    if "compens" in text:
        return "Compensation"
    return "Connectivity"


def feature_scientific_meaning(row: pd.Series, label_map: dict[str, str]) -> str:
    feature = str(row.get("feature", ""))
    group = str(row.get("feature_group", "") or feature_card_group(row))
    lower = feature.lower()

    edge_match = re.search(r"raw_edge__([a-z0-9_]+)__n(\d{3})_n(\d{3})", lower)
    if edge_match:
        metric, node_a, node_b = edge_match.groups()
        return (
            f"Single AAL3 edge between {_aal_region_text(node_a, label_map)} and {_aal_region_text(node_b, label_map)}. "
            f"The value is {_diffusion_metric_meaning(metric)}, so it captures a specific tract-level connection signal."
        )

    node_match = re.search(r"node_metrics__(degree|strength|nodal_eff)__aal_(\d{3})", lower)
    if node_match:
        metric, node = node_match.groups()
        region = _aal_region_text(node, label_map)
        if metric == "degree":
            return f"Number of nonzero structural connections incident on {region}; scientifically this is local network connectedness."
        if metric == "strength":
            return f"Total structural connection weight attached to {region}; higher values indicate a stronger local connectome hub signal."
        return f"Nodal efficiency of {region}; this reflects how efficiently that region can communicate with the rest of the structural network."

    roi_match = re.search(r"local_roi_dti__([a-z0-9_]+)__aal_(\d{3})", lower)
    if roi_match:
        metric, node = roi_match.groups()
        return f"Local ROI {_diffusion_metric_meaning(metric)} in {_aal_region_text(node, label_map)}; this is a regional microstructure marker."

    edr_aal_match = re.search(r"edr_aal__([a-z0-9_]+)__aal_(\d{3})", lower)
    if edr_aal_match:
        metric, node = edr_aal_match.groups()
        return f"AAL3 regional EDR exception feature for {_aal_region_text(node, label_map)}; it summarizes where expected edge-distance behavior deviates locally."

    matrix_match = re.search(r"matrix_summary__([a-z0-9_]+)__([a-z0-9_]+)", lower)
    if matrix_match:
        metric, stat = matrix_match.groups()
        return f"Whole-connectome matrix summary: {_edge_stat_meaning(stat)} of {_diffusion_metric_meaning(metric)}. It captures global distributional burden rather than one ROI."

    global_dti_match = re.search(r"global_dti__([a-z0-9_]+)_(edge_[a-z0-9]+|n_edges)", lower)
    if global_dti_match:
        metric, stat = global_dti_match.groups()
        return f"Global DTI feature: {_edge_stat_meaning(stat)} of {_diffusion_metric_meaning(metric)} across connectome edges."

    coupling_match = re.search(r"coupling__(rho|n_nodes)__(degree|strength|nodal_eff)__([a-z0-9_]+)", lower)
    if coupling_match:
        kind, topology, micro = coupling_match.groups()
        topology_text = metric_label(topology).lower()
        micro_text = _diffusion_metric_meaning(micro)
        if kind == "rho":
            return f"Within-subject Spearman coupling between node {topology_text} and node-wise {micro_text}; it tests whether network hubs align with microstructure."
        return f"Number of AAL3 nodes contributing to the {topology_text} to {micro_text} coupling estimate."

    if lower.startswith("global_graph__"):
        metric = lower.replace("global_graph__", "")
        return f"Global graph topology summary: {metric_label(metric).lower()}, describing whole-network integration or density."
    if lower.startswith("brain_age__"):
        metric = lower.replace("brain_age__", "")
        return f"Brain-age/connectome-age derived feature: {metric_label(metric).lower()}, summarizing whether structural features resemble older or younger brains."
    if lower.startswith("delay__"):
        metric = lower.replace("delay__", "")
        return f"Length-derived conduction-delay proxy: {metric_label(metric).lower()}, based on tract length and assumed signal velocity."
    if lower.startswith("lr_sr__"):
        metric = lower.replace("lr_sr__", "")
        return f"Short-range/long-range organization feature: {metric_label(metric).lower()}, summarizing how connectivity is distributed by tract length."
    if lower.startswith("edr__"):
        metric = lower.replace("edr__", "")
        return f"Edge-distance relationship feature: {metric_label(metric).lower()}, measuring how connection strength changes with distance or where it deviates."
    if lower.startswith("advanced__"):
        metric = lower.replace("advanced__", "")
        return f"Advanced structural summary: {metric_label(metric).lower()}, designed to capture compensatory or vulnerability patterns in the connectome."

    return f"{metric_label(group)} feature from the structural connectome. Interpret it as a model-selected signal, not as a standalone causal biomarker."


def render_predictor_interpretation_table(importance: pd.DataFrame) -> None:
    st.markdown("### Top Predictor Interpretation")
    if importance.empty or "feature" not in importance.columns:
        st.info("Predictor interpretation is not available for this selected target/model.")
        return
    work = importance.head(100).copy().reset_index(drop=True)
    if "rank" in work.columns:
        ranks = pd.to_numeric(work["rank"], errors="coerce")
    else:
        ranks = pd.Series(np.arange(1, len(work) + 1), index=work.index)
    table = pd.DataFrame(
        {
            "Rank": ranks.fillna(pd.Series(np.arange(1, len(work) + 1), index=work.index)).astype(int),
            "Feature name": work["feature"].astype(str),
            "What it means": [feature_neuroscience_label(row) for _, row in work.iterrows()],
        }
    )
    st.dataframe(
        table,
        width="stretch",
        hide_index=True,
        height=420,
        column_config={
            "Rank": st.column_config.NumberColumn("Rank", format="%d", width="small"),
            "Feature name": st.column_config.TextColumn("Feature name", width="medium"),
            "What it means": st.column_config.TextColumn("What it means", width="large"),
        },
    )


def importance_value_note(method: object) -> str:
    method_text = str(method or "").strip()
    if method_text == "tree_feature_importance":
        return "Relative tree-ensemble reliance; larger means the model used this feature more to reduce prediction error."
    if method_text == "xgboost_treeshap_abs_contribution":
        return "Mean absolute TreeSHAP contribution; larger means bigger average effect on the model prediction."
    if method_text == "abs_elastic_net_coef":
        return "Absolute standardized linear coefficient; larger means a stronger regularized linear association."
    return "Model-specific importance; compare values within the selected target, model, and method only."


def render_feature_group_feature_box(catalog: pd.DataFrame | None) -> None:
    if catalog is None or catalog.empty or not {"feature_group", "feature"}.issubset(catalog.columns):
        st.info("Feature group details are not available for this refresh.")
        return
    work = catalog.copy()
    if "final_status" in work.columns:
        work = work[~work["final_status"].fillna("").astype(str).str.startswith("dropped")].copy()
    if work.empty:
        st.info("No model-ready feature groups are available after preprocessing.")
        return
    if "selected_fold_count" in work.columns:
        work["selected_fold_count"] = pd.to_numeric(work["selected_fold_count"], errors="coerce").fillna(0)
    else:
        work["selected_fold_count"] = 0
    groups = []
    for group, sub in work.groupby("feature_group", dropna=False):
        sub = sub.sort_values(["selected_fold_count", "feature"], ascending=[False, True])
        features = [escape(str(value)) for value in sub["feature"].dropna().astype(str).head(8)]
        display_group = "Whole-connectome distribution summaries" if str(group) == "Raw matrix summaries" else str(group)
        groups.append((display_group, int(len(sub)), int(sub["selected_fold_count"].gt(0).sum()), features))
    groups = sorted(groups, key=lambda item: (item[2], item[1]), reverse=True)
    blocks = []
    for group, n_features, n_selected, features in groups[:10]:
        items = "".join(f"<li>{feature}</li>" for feature in features)
        selected_text = f"{n_selected} CV-selected" if n_selected else "model-ready"
        blocks.append(
            "<div style='min-width:0;padding:0.65rem;border:1px solid #334155;border-radius:8px;background:#111827;'>"
            f"<div style='display:flex;justify-content:space-between;gap:0.75rem;align-items:baseline;'>"
            f"<h5 style='margin:0;color:#facc15;font-size:0.95rem;'>{escape(group)}</h5>"
            f"<span style='color:#94a3b8;font-size:0.75rem;white-space:nowrap;'>{n_features} features | {selected_text}</span>"
            "</div>"
            "<ul style='margin:0.45rem 0 0 1rem;padding:0;color:#e5e7eb;font-size:0.78rem;line-height:1.35;'>"
            + items
            + "</ul></div>"
        )
    render_yellow_left_heading("Feature Groups And Features")
    st.markdown(
        "<div style='margin:0 0 0.3rem 0;border:1px solid #334155;border-radius:8px;background:#0f172a;padding:0.85rem;'>"
        "<div style='display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:0.75rem;max-height:360px;overflow-y:auto;'>"
        + "".join(blocks)
        + "</div></div>",
        unsafe_allow_html=True,
    )


def render_feature_importance_cards(importance: pd.DataFrame) -> None:
    if importance.empty:
        st.info("Feature-importance rows are not available for this target/model.")
        return
    work = importance.copy()
    work["card_group"] = work.apply(feature_card_group, axis=1)
    work["importance"] = pd.to_numeric(work.get("importance"), errors="coerce")
    grouped = []
    for group, sub in work.groupby("card_group", dropna=False):
        sub = sub.sort_values("importance", ascending=False, na_position="last").head(4)
        grouped.append((str(group), sub))
    grouped = sorted(grouped, key=lambda item: float(item[1]["importance"].fillna(0).max()), reverse=True)
    for start in range(0, len(grouped), 4):
        cols = st.columns(4)
        for col, (group, sub) in zip(cols, grouped[start : start + 4]):
            rows = []
            method_note = importance_value_note(sub["importance_method"].iloc[0] if "importance_method" in sub.columns and not sub.empty else "")
            for _, row in sub.iterrows():
                value = pd.to_numeric(pd.Series([row.get("importance")]), errors="coerce").iloc[0]
                value_text = f"{value:.3g}" if pd.notna(value) else ""
                rows.append(
                    "<tr>"
                    f"<td style='vertical-align:top;color:#e5e7eb;border-bottom:1px solid rgba(51,65,85,0.72);padding:0.34rem 0.28rem;overflow-wrap:anywhere;'>{escape(str(row.get('feature', '')))}</td>"
                    f"<td style='vertical-align:top;color:#facc15;border-bottom:1px solid rgba(51,65,85,0.72);padding:0.34rem 0.28rem;text-align:right;font-weight:800;white-space:nowrap;'>{escape(value_text)}</td>"
                    f"<td style='vertical-align:top;color:#cbd5e1;border-bottom:1px solid rgba(51,65,85,0.72);padding:0.34rem 0.28rem;overflow-wrap:anywhere;'>{escape(feature_neuroscience_label(row))}</td>"
                    "</tr>"
                )
            with col:
                st.markdown(
                    "<div style='border:1px solid #334155;border-radius:8px;padding:0.75rem;min-height:300px;background:#0f172a;'>"
                    f"<h4 style='margin:0 0 0.55rem 0;color:#facc15;text-align:center;'>{escape(group)}</h4>"
                    f"<div style='font-size:0.72rem;color:#cbd5e1;margin:0 0 0.55rem 0;text-align:center;'>{escape(method_note)}</div>"
                    "<div style='max-height:280px;overflow-y:auto;overflow-x:auto;'>"
                    "<table style='width:100%;border-collapse:collapse;background:transparent;font-size:0.72rem;line-height:1.25;'>"
                    "<thead><tr>"
                    "<th style='text-align:left;color:#f8fafc;border-bottom:1px solid #334155;padding:0.3rem;'>Feature</th>"
                    "<th style='text-align:right;color:#f8fafc;border-bottom:1px solid #334155;padding:0.3rem;'>Importance</th>"
                    "<th style='text-align:left;color:#f8fafc;border-bottom:1px solid #334155;padding:0.3rem;'>Meaning</th>"
                    "</tr></thead>"
                    "<tbody>"
                    + "".join(rows)
                    + "</tbody></table></div></div>",
                    unsafe_allow_html=True,
                )


def cdr_class_from_value(value: object) -> str | None:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return None
    if numeric <= 0:
        return "0"
    if numeric <= 0.5:
        return "0.5"
    if numeric <= 1:
        return "1"
    return "2+"


def clinical_target_distribution_figure(matches: pd.DataFrame | None, target: str, target_mode: str) -> go.Figure:
    fig = go.Figure()
    if matches is None or matches.empty:
        fig.update_layout(title="Target distribution unavailable", height=330)
        return apply_ml_plot_theme(fig)
    plot_df = matches[
        matches["target"].astype(str).eq(str(target))
        & matches["target_mode"].astype(str).eq(str(target_mode))
    ].copy()
    plot_df["target_value"] = pd.to_numeric(plot_df.get("target_value"), errors="coerce")
    plot_df = plot_df.dropna(subset=["target_value"])
    color_map = dict(zip(GROUP_ORDER, GROUP_COLORS))
    if target == CDR_CLASSIFICATION_TARGET:
        plot_df["cdr_class"] = plot_df["target_value"].map(cdr_class_from_value)
        for group in GROUP_ORDER:
            sub = plot_df[plot_df["group"].astype(str).eq(group)]
            counts = sub["cdr_class"].value_counts()
            fig.add_trace(
                go.Bar(
                    x=list(CDR_CLASS_ORDER),
                    y=[int(counts.get(label, 0)) for label in CDR_CLASS_ORDER],
                    name=group,
                    marker_color=color_map.get(group, "#64748b"),
                    text=[int(counts.get(label, 0)) for label in CDR_CLASS_ORDER],
                    textposition="outside",
                    cliponaxis=False,
                )
            )
        fig.update_layout(
            title="Global CDR class distribution",
            barmode="group",
            yaxis={"title": "Subjects", "gridcolor": CHART_GRID},
            xaxis={"title": "CDR class"},
        )
    else:
        for group in GROUP_ORDER:
            sub = plot_df[plot_df["group"].astype(str).eq(group)]
            if sub.empty:
                continue
            fig.add_trace(
                go.Histogram(
                    x=sub["target_value"],
                    name=group,
                    marker_color=color_map.get(group, "#64748b"),
                    opacity=0.75,
                    nbinsx=14,
                )
            )
        fig.update_layout(
            title=f"{metric_label(target)} distribution",
            barmode="overlay",
            yaxis={"title": "Subjects", "gridcolor": CHART_GRID},
            xaxis={"title": "Clinical score", "gridcolor": CHART_GRID},
        )
    fig.update_layout(
        height=330,
        paper_bgcolor=CHART_BACKGROUND,
        plot_bgcolor=CHART_BACKGROUND,
        font={"color": CHART_TEXT},
        margin={"l": 55, "r": 25, "t": 70, "b": 50},
        legend={"orientation": "h", "y": 1.1, "x": 0},
    )
    return apply_ml_plot_theme(fig)


def set_selector_value(key: str, value: str) -> None:
    st.session_state[key] = value


def render_button_selector(options: list[str], key: str, label_func=None, default: str | None = None, max_cols: int = 8) -> str:
    if not options:
        return ""
    option_values = [str(option) for option in options]
    default_value = str(default) if default is not None else None
    if key not in st.session_state or str(st.session_state[key]) not in option_values:
        st.session_state[key] = default_value if default_value in option_values else option_values[0]
    selected = str(st.session_state[key])
    cols = st.columns(min(len(options), max_cols))
    for idx, option in enumerate(options):
        option_value = str(option)
        with cols[idx % len(cols)]:
            label = label_func(option_value) if label_func else option_value
            st.button(
                str(label),
                key=f"{key}-{idx}-{option_value}",
                type="primary" if option_value == selected else "secondary",
                use_container_width=True,
                on_click=set_selector_value,
                args=(key, option_value),
            )
    return str(st.session_state[key])


def _percentage_text(value: object) -> str:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return "NA"
    return f"{numeric * 100:.1f}%"


def render_selected_model_metrics(
    perf: pd.DataFrame,
    selected_target: str,
    selected_mode: str,
    selected_model: str,
    *,
    task_type: str,
) -> None:
    if perf is None or perf.empty:
        return
    perf = add_model_identity(perf)
    work = perf[
        perf["target_mode"].astype(str).eq(str(selected_mode))
        & perf["model_key"].astype(str).eq(str(selected_model))
    ].copy()
    if "target" in work.columns:
        work = work[work["target"].astype(str).eq(str(selected_target))]
    if work.empty:
        return
    row = work.iloc[0]
    if task_type == "classification":
        metrics = [
            ("Accuracy", _percentage_text(row.get("accuracy"))),
            ("Balanced accuracy", _percentage_text(row.get("balanced_accuracy"))),
            ("Macro F1", _percentage_text(row.get("macro_f1"))),
            ("Sensitivity", _percentage_text(row.get("macro_recall_sensitivity"))),
            ("Specificity", _percentage_text(row.get("macro_specificity"))),
            ("AUROC OVR", format_number(row.get("macro_auroc_ovr"), digits=3)),
            ("Subjects", format_number(row.get("n_subjects"), digits=0)),
        ]
    else:
        metrics = [
            ("R2", format_number(row.get("r2"), digits=3)),
            ("MAE", format_number(row.get("mae"), digits=2)),
            ("RMSE", format_number(row.get("rmse"), digits=2)),
            ("Spearman r", format_number(row.get("spearman_r"), digits=3)),
            ("Subjects", format_number(row.get("n_subjects"), digits=0)),
        ]
    render_selected_metric_cards(
        "Selected model metrics",
        str(row.get("model_display", ml_model_label(str(row.get("model", selected_model))))),
        score_mode_display(selected_mode),
        metrics,
    )


def render_selected_metric_cards(title: str, model_name: str, subtitle: str, metrics: list[tuple[str, str]]) -> None:
    cards = []
    for label, value in metrics:
        cards.append(
            "<div style='min-width:160px;border:1px solid rgba(250,204,21,0.42);border-radius:8px;"
            "background:linear-gradient(180deg,#101827 0%,#050b16 100%);padding:1.05rem 1.08rem;"
            "box-shadow:0 15px 32px rgba(0,0,0,0.32);'>"
            f"<div style='color:#facc15;font-size:0.78rem;font-weight:1000;text-transform:uppercase;letter-spacing:0;'>{escape(label)}</div>"
            f"<div style='color:#ffffff;font-size:2.25rem;font-weight:1000;line-height:1.02;margin-top:0.36rem;'>{escape(str(value))}</div>"
            "</div>"
        )
    st.markdown(
        "<div style='margin:0.9rem 0 0.95rem 0;border:1px solid rgba(250,204,21,0.28);"
        "border-radius:8px;background:#020617;padding:1rem;'>"
        "<div style='display:flex;align-items:flex-end;justify-content:space-between;gap:1rem;flex-wrap:wrap;margin-bottom:0.75rem;'>"
        "<div>"
        f"<div style='color:#facc15;font-size:0.82rem;font-weight:1000;text-transform:uppercase;letter-spacing:0;'>{escape(title)}</div>"
        f"<div style='color:#ffffff;font-size:1.55rem;font-weight:1000;line-height:1.15;margin-top:0.18rem;'>{escape(model_name)}</div>"
        "</div>"
        f"<div style='color:#cbd5e1;font-size:0.92rem;font-weight:850;'>{escape(subtitle)}</div>"
        "</div>"
        "<div style='display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:0.75rem;'>"
        + "".join(cards)
        + "</div></div>",
        unsafe_allow_html=True,
    )


def render_disease_selected_model_metrics(
    perf: pd.DataFrame,
    selected_model: str,
    class_metrics: pd.DataFrame | None = None,
) -> None:
    if perf is None or perf.empty:
        return
    work = perf[perf["model"].astype(str).eq(str(selected_model))].copy()
    if "status" in work.columns:
        ok_work = work[work["status"].astype(str).eq("ok")].copy()
        if not ok_work.empty:
            work = ok_work
    if work.empty:
        return
    row = work.iloc[0]
    specificity = None
    if class_metrics is not None and not class_metrics.empty and "specificity" in class_metrics.columns:
        cmet = class_metrics[class_metrics["model"].astype(str).eq(str(selected_model))].copy()
        if not cmet.empty:
            specificity = pd.to_numeric(cmet["specificity"], errors="coerce").mean()
    metrics = [
        ("Accuracy", _percentage_text(row.get("accuracy"))),
        ("Balanced accuracy", _percentage_text(row.get("balanced_accuracy"))),
        ("Macro F1", _percentage_text(row.get("macro_f1"))),
        ("Sensitivity", _percentage_text(row.get("macro_recall"))),
        ("Specificity", _percentage_text(specificity)),
        ("AUROC OVR", format_number(row.get("macro_auroc_ovr"), digits=3)),
        ("Subjects", format_number(row.get("n_subjects"), digits=0)),
    ]
    render_selected_metric_cards(
        "Selected disease model metrics",
        ml_model_label(selected_model),
        "CN / MCI / AD classifier",
        metrics,
    )


def render_top_models_table(perf: pd.DataFrame, selected_target: str, selected_mode: str, *, task_type: str, top_n: int | str = 10) -> None:
    if perf is None or perf.empty:
        return
    perf = add_model_identity(perf)
    work = perf[perf["target_mode"].astype(str).eq(str(selected_mode))].copy()
    if "target" in work.columns:
        work = work[work["target"].astype(str).eq(str(selected_target))]
    if "status" in work.columns:
        work = work[work["status"].astype(str).eq("ok")].copy()
    if work.empty:
        return
    if task_type == "classification":
        sort_cols = [col for col in ("accuracy", "balanced_accuracy", "macro_f1") if col in work.columns]
        for col in sort_cols:
            work[col] = pd.to_numeric(work[col], errors="coerce")
        if sort_cols:
            work = work.sort_values(sort_cols, ascending=[False] * len(sort_cols), na_position="last")
        if isinstance(top_n, int):
            work = work.head(top_n)
        columns = [
            ("Model", work.get("model_display", pd.Series([""] * len(work), index=work.index)).astype(str).tolist()),
            ("Feature policy", [metric_label(v) for v in work.get("feature_policy", pd.Series(["current_dashboard"] * len(work), index=work.index)).astype(str)]),
            ("Accuracy", [_percentage_text(v) for v in work.get("accuracy", pd.Series(index=work.index))]),
            ("Balanced Acc.", [_percentage_text(v) for v in work.get("balanced_accuracy", pd.Series(index=work.index))]),
            ("Macro F1", [_percentage_text(v) for v in work.get("macro_f1", pd.Series(index=work.index))]),
            ("Sensitivity", [_percentage_text(v) for v in work.get("macro_recall_sensitivity", pd.Series(index=work.index))]),
            ("Specificity", [_percentage_text(v) for v in work.get("macro_specificity", pd.Series(index=work.index))]),
            ("AUROC OVR", [format_number(v, digits=3) for v in work.get("macro_auroc_ovr", pd.Series(index=work.index))]),
        ]
    else:
        sort_cols = [col for col in ("r2", "spearman_r") if col in work.columns]
        for col in ("r2", "mae", "rmse", "spearman_r"):
            if col in work.columns:
                work[col] = pd.to_numeric(work[col], errors="coerce")
        work = work.sort_values(sort_cols or ["model"], ascending=[False] * len(sort_cols) if sort_cols else [True], na_position="last")
        if isinstance(top_n, int):
            work = work.head(top_n)
        columns = [
            ("Model", work.get("model_display", pd.Series([""] * len(work), index=work.index)).astype(str).tolist()),
            ("Feature policy", [metric_label(v) for v in work.get("feature_policy", pd.Series(["current_dashboard"] * len(work), index=work.index)).astype(str)]),
            ("R2", [format_number(v, digits=3) for v in work.get("r2", pd.Series(index=work.index))]),
            ("MAE", [format_number(v, digits=2) for v in work.get("mae", pd.Series(index=work.index))]),
            ("RMSE", [format_number(v, digits=2) for v in work.get("rmse", pd.Series(index=work.index))]),
            ("Spearman r", [format_number(v, digits=3) for v in work.get("spearman_r", pd.Series(index=work.index))]),
        ]
    rows = []
    for row_idx in range(len(work)):
        cells = []
        for col_idx, (_, values) in enumerate(columns):
            color = "#ffffff" if col_idx == 0 else "#f8fafc"
            weight = "900" if col_idx == 0 else "850"
            cells.append(
                f"<td style='padding:0.48rem 0.6rem;border-bottom:1px solid rgba(51,65,85,0.7);color:{color};font-weight:{weight};'>{escape(str(values[row_idx]))}</td>"
            )
        rows.append("<tr>" + "".join(cells) + "</tr>")
    header = "".join(
        f"<th style='text-align:left;padding:0.45rem 0.6rem;border-bottom:1px solid #334155;color:#facc15;font-weight:950;'>{escape(label)}</th>"
        for label, _ in columns
    )
    st.markdown(
        "<div style='margin:0.75rem 0 1rem 0;border:1px solid #334155;border-radius:8px;background:#0f172a;padding:0.75rem;'>"
        f"<div style='color:#facc15;font-size:1.05rem;font-weight:950;margin-bottom:0.55rem;'>Top Models ({len(work)})</div>"
        "<div style='overflow-x:auto;'>"
        "<table style='width:100%;border-collapse:collapse;background:transparent;font-size:0.9rem;'>"
        f"<thead><tr>{header}</tr></thead><tbody>{''.join(rows)}</tbody></table>"
        "</div></div>",
        unsafe_allow_html=True,
    )


def selected_model_row(perf: pd.DataFrame, selected_target: str, selected_mode: str, selected_model: str) -> pd.Series | None:
    if perf is None or perf.empty:
        return None
    work = add_model_identity(perf)
    if "target" in work.columns:
        work = work[work["target"].astype(str).eq(str(selected_target))]
    work = work[
        work["target_mode"].astype(str).eq(str(selected_mode))
        & work["model_key"].astype(str).eq(str(selected_model))
    ].copy()
    if work.empty:
        return None
    return work.iloc[0]


def ensemble_rank_metric(model_name: str, task_type: str) -> str:
    if task_type == "classification":
        if "_balanced_" in model_name:
            return "balanced_accuracy"
        if "_f1_" in model_name:
            return "macro_f1"
        return "accuracy"
    return "r2"


def ensemble_member_count(model_name: str) -> int | None:
    match = re.search(r"_top(\d+)_", model_name)
    if match:
        return int(match.group(1))
    if "_all_" in model_name:
        return None
    return None


def ensemble_base_models(perf: pd.DataFrame, selected_target: str, selected_mode: str, model_name: str, *, task_type: str) -> pd.DataFrame:
    if perf is None or perf.empty:
        return pd.DataFrame()
    rank_metric = ensemble_rank_metric(model_name, task_type)
    n_take = ensemble_member_count(model_name)
    work = add_model_identity(filter_no_explicit_age_policy(perf))
    if "target" in work.columns:
        work = work[work["target"].astype(str).eq(str(selected_target))]
    work = work[
        work["target_mode"].astype(str).eq(str(selected_mode))
        & work.get("status", pd.Series("", index=work.index)).astype(str).eq("ok")
        & ~work["feature_policy"].astype(str).eq("posthoc_oof_ensemble_exploratory")
        & ~work["model"].astype(str).str.startswith("ensemble_oof")
    ].copy()
    if work.empty or rank_metric not in work.columns:
        return pd.DataFrame()
    work[rank_metric] = pd.to_numeric(work[rank_metric], errors="coerce")
    sort_cols = [rank_metric]
    for col in ("balanced_accuracy", "macro_f1", "macro_auroc_ovr", "spearman_r", "rmse"):
        if col in work.columns and col not in sort_cols:
            work[col] = pd.to_numeric(work[col], errors="coerce")
            sort_cols.append(col)
    ascending = [False if col != "rmse" else True for col in sort_cols]
    work = work.sort_values(sort_cols, ascending=ascending, na_position="last")
    if n_take is not None:
        work = work.head(n_take)
    return work.copy()


def render_ensemble_explainer(perf: pd.DataFrame, selected_target: str, selected_mode: str, selected_model: str, *, task_type: str) -> None:
    row = selected_model_row(perf, selected_target, selected_mode, selected_model)
    if row is None:
        return
    model_name = str(row.get("model", ""))
    if not model_name.startswith("ensemble_oof"):
        return
    rank_metric = ensemble_rank_metric(model_name, task_type)
    n_take = ensemble_member_count(model_name)
    n_text = f"top {n_take}" if n_take is not None else "all"
    calibrated = "calibrated" in model_name
    base = ensemble_base_models(perf, selected_target, selected_mode, model_name, task_type=task_type)
    if base.empty:
        detail = str(row.get("best_params_by_fold", ""))
        st.info(f"Ensemble details: {detail}" if detail else "This is a post-hoc out-of-fold ensemble; base model details are unavailable.")
        return
    table_cols = [
        c
        for c in (
            "model_display",
            "feature_policy",
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "macro_auroc_ovr",
            "r2",
            "mae",
            "rmse",
            "spearman_r",
        )
        if c in base.columns
    ]
    display = base[table_cols].copy()
    display = display.rename(columns={"model_display": "model", "feature_policy": "feature policy"})
    calibration_text = "with fold-safe class-prior calibration" if calibrated and task_type == "classification" else "without post-hoc calibration"
    st.markdown(
        "<div style='border:1px solid rgba(250,204,21,0.35);border-radius:8px;background:#07111f;padding:0.85rem;margin:0.4rem 0 0.9rem 0;'>"
        f"<div style='color:#facc15;font-weight:1000;font-size:0.95rem;margin-bottom:0.25rem;'>What is this ensemble?</div>"
        f"<div style='color:#e5e7eb;font-size:0.92rem;'>This model averages out-of-fold predictions from the {escape(n_text)} base models ranked by "
        f"<strong>{escape(metric_label(rank_metric))}</strong>, {escape(calibration_text)}. The base models are listed below.</div>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.dataframe(
        display,
        width="stretch",
        hide_index=True,
        height=compact_table_height(len(display)),
        column_config={
            "accuracy": st.column_config.NumberColumn("accuracy", format="%.3f"),
            "balanced_accuracy": st.column_config.NumberColumn("balanced accuracy", format="%.3f"),
            "macro_f1": st.column_config.NumberColumn("macro F1", format="%.3f"),
            "macro_auroc_ovr": st.column_config.NumberColumn("AUROC OVR", format="%.3f"),
            "r2": st.column_config.NumberColumn("R2", format="%.3f"),
            "mae": st.column_config.NumberColumn("MAE", format="%.3f"),
            "rmse": st.column_config.NumberColumn("RMSE", format="%.3f"),
            "spearman_r": st.column_config.NumberColumn("Spearman r", format="%.3f"),
        },
        key=f"ensemble-base-models-{selected_target}-{selected_mode}-{selected_model}",
    )


def cdr_classification_best_model(perf: pd.DataFrame, target_mode: str) -> str:
    if perf.empty:
        return ""
    perf = add_model_identity(perf)
    work = perf[
        perf["target_mode"].astype(str).eq(str(target_mode))
        & perf.get("status", pd.Series("", index=perf.index)).astype(str).eq("ok")
    ].copy()
    if work.empty:
        return ""
    for col in ("accuracy", "balanced_accuracy", "macro_f1"):
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")
    sort_cols = [col for col in ("accuracy", "balanced_accuracy", "macro_f1") if col in work.columns]
    work = work.sort_values(sort_cols, ascending=[False] * len(sort_cols), na_position="last")
    return str(work.iloc[0]["model_key"])


def cdr_classification_performance_figure(perf: pd.DataFrame, target_mode: str) -> go.Figure:
    perf = add_model_identity(perf)
    plot_df = perf[
        perf["target_mode"].astype(str).eq(str(target_mode))
        & perf.get("status", pd.Series("", index=perf.index)).astype(str).eq("ok")
    ].copy()
    for col in ("accuracy", "balanced_accuracy", "macro_f1"):
        if col in plot_df.columns:
            plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce")
    sort_cols = [col for col in ("accuracy", "balanced_accuracy", "macro_f1") if col in plot_df.columns]
    if sort_cols:
        plot_df = plot_df.sort_values(sort_cols, ascending=[False] * len(sort_cols), na_position="last").head(8)
    plot_df["Model"] = plot_df.get("model_display", pd.Series(dtype=str)).astype(str)
    fig = go.Figure()
    specs = (
        ("accuracy", "Accuracy", "#2563eb"),
        ("balanced_accuracy", "Balanced accuracy", "#f97316"),
        ("macro_f1", "Macro F1", "#22c55e"),
    )
    for col, label, color in specs:
        if col not in plot_df.columns:
            continue
        values = pd.to_numeric(plot_df[col], errors="coerce")
        fig.add_trace(go.Bar(x=plot_df["Model"], y=values, name=label, marker_color=color, text=[format_number(v, 3) for v in values], textposition="outside"))
    fig.update_layout(
        title="Global CDR classifier performance",
        barmode="group",
        height=380,
        paper_bgcolor=CHART_BACKGROUND,
        plot_bgcolor=CHART_BACKGROUND,
        font={"color": CHART_TEXT},
        margin={"l": 55, "r": 25, "t": 75, "b": 95},
        yaxis={"title": "Higher is better", "range": [0, 1], "gridcolor": CHART_GRID},
        xaxis={"title": ""},
        legend={"orientation": "h", "y": 1.13, "x": 0},
    )
    return apply_ml_plot_theme(fig)


def cdr_confusion_figure(confusion: pd.DataFrame | None, target_mode: str, model: str) -> go.Figure:
    fig = go.Figure()
    if confusion is None or confusion.empty:
        fig.update_layout(title="Confusion matrix unavailable", height=380)
        return apply_ml_plot_theme(fig)
    confusion = add_model_identity(confusion)
    plot_df = confusion[
        confusion["target_mode"].astype(str).eq(str(target_mode))
        & confusion["model_key"].astype(str).eq(str(model))
    ].copy()
    if plot_df.empty:
        fig.update_layout(title="Confusion matrix unavailable", height=380)
        return apply_ml_plot_theme(fig)
    plot_df["true_class"] = plot_df["true_class"].map(normalize_cdr_class_label)
    plot_df["predicted_class"] = plot_df["predicted_class"].map(normalize_cdr_class_label)
    labels = [label for label in CDR_CLASS_ORDER if label in set(plot_df["true_class"].astype(str)) or label in set(plot_df["predicted_class"].astype(str))]
    matrix = np.zeros((len(labels), len(labels)), dtype=float)
    for _, row in plot_df.iterrows():
        if str(row.get("true_class")) in labels and str(row.get("predicted_class")) in labels:
            matrix[labels.index(str(row.get("true_class"))), labels.index(str(row.get("predicted_class")))] = pd.to_numeric(pd.Series([row.get("n")]), errors="coerce").fillna(0).iloc[0]
    fig.add_trace(
        go.Heatmap(
            z=matrix,
            x=labels,
            y=labels,
            colorscale="Blues",
            text=matrix.astype(int),
            texttemplate="%{text}",
            showscale=False,
            hovertemplate="True=%{y}<br>Predicted=%{x}<br>n=%{z}<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"{plot_df['model_display'].iloc[0] if 'model_display' in plot_df.columns and not plot_df.empty else ml_model_label(model)} confusion matrix",
        height=380,
        paper_bgcolor=CHART_BACKGROUND,
        plot_bgcolor=CHART_BACKGROUND,
        font={"color": CHART_TEXT},
        margin={"l": 65, "r": 25, "t": 75, "b": 55},
        xaxis={"title": "Predicted CDR class"},
        yaxis={"title": "True CDR class", "autorange": "reversed"},
    )
    return apply_ml_plot_theme(fig)


def cdr_class_metrics_figure(class_metrics: pd.DataFrame | None, target_mode: str, model: str) -> go.Figure:
    fig = go.Figure()
    if class_metrics is None or class_metrics.empty:
        fig.update_layout(title="Class metrics unavailable", height=380)
        return apply_ml_plot_theme(fig)
    class_metrics = add_model_identity(class_metrics)
    plot_df = class_metrics[
        class_metrics["target_mode"].astype(str).eq(str(target_mode))
        & class_metrics["model_key"].astype(str).eq(str(model))
    ].copy()
    if plot_df.empty:
        fig.update_layout(title="Class metrics unavailable", height=380)
        return apply_ml_plot_theme(fig)
    plot_df["class"] = pd.Categorical(plot_df["class"].map(normalize_cdr_class_label), categories=list(CDR_CLASS_ORDER), ordered=True)
    plot_df = plot_df.sort_values("class")
    for col, label, color in (("recall_sensitivity", "Sensitivity", "#22c55e"), ("specificity", "Specificity", "#facc15")):
        values = pd.to_numeric(plot_df[col], errors="coerce") if col in plot_df.columns else pd.Series(dtype=float)
        fig.add_trace(go.Bar(x=plot_df["class"].astype(str), y=values, name=label, marker_color=color, text=[format_number(v, 3) for v in values], textposition="outside"))
    fig.update_layout(
        title="Per-class sensitivity/specificity",
        barmode="group",
        height=380,
        paper_bgcolor=CHART_BACKGROUND,
        plot_bgcolor=CHART_BACKGROUND,
        font={"color": CHART_TEXT},
        margin={"l": 55, "r": 25, "t": 75, "b": 55},
        yaxis={"title": "Metric", "range": [0, 1], "gridcolor": CHART_GRID},
        xaxis={"title": "CDR class"},
        legend={"orientation": "h", "y": 1.13, "x": 0},
    )
    return apply_ml_plot_theme(fig)


def cdr_roc_figure(predictions: pd.DataFrame | None, target_mode: str, model: str) -> go.Figure:
    fig = go.Figure()
    if predictions is None or predictions.empty:
        fig.update_layout(title="CDR ROC unavailable", height=380)
        return apply_ml_plot_theme(fig)
    predictions = add_model_identity(predictions)
    plot_df = predictions[
        predictions["target_mode"].astype(str).eq(str(target_mode))
        & predictions["model_key"].astype(str).eq(str(model))
    ].copy()
    if plot_df.empty or "true_class" not in plot_df.columns:
        fig.update_layout(title="CDR ROC unavailable", height=380)
        return apply_ml_plot_theme(fig)
    labels = [label for label in CDR_CLASS_ORDER if f"prob_{label}" in plot_df.columns]
    if not labels:
        fig.update_layout(title="CDR ROC unavailable", height=380)
        return apply_ml_plot_theme(fig)
    try:
        from sklearn.metrics import auc, roc_curve
    except Exception:
        fig.update_layout(title="CDR ROC requires scikit-learn", height=380)
        return apply_ml_plot_theme(fig)
    plot_df["true_class"] = plot_df["true_class"].map(normalize_cdr_class_label)
    colors = ["#2563eb", "#f97316", "#22c55e", "#facc15"]
    for label, color in zip(labels, colors):
        y_true = plot_df["true_class"].astype(str).eq(label).astype(int).to_numpy()
        y_score = pd.to_numeric(plot_df[f"prob_{label}"], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(y_score)
        y_true = y_true[finite]
        y_score = y_score[finite]
        if len(np.unique(y_true)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        fig.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                mode="lines",
                name=f"CDR {label} AUC={roc_auc:.3f}",
                line={"color": color, "width": 3},
                hovertemplate="FPR=%{x:.3f}<br>TPR=%{y:.3f}<extra></extra>",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="chance",
            line={"color": "#64748b", "width": 2, "dash": "dash"},
            hovertemplate="Chance<extra></extra>",
        )
    )
    fig.update_layout(
        title="Global CDR one-vs-rest ROC",
        height=450,
        paper_bgcolor=CHART_BACKGROUND,
        plot_bgcolor=CHART_BACKGROUND,
        font={"color": CHART_TEXT},
        margin={"l": 55, "r": 25, "t": 75, "b": 125},
        xaxis={"title": "False positive rate", "range": [0, 1], "gridcolor": CHART_GRID},
        yaxis={"title": "True positive rate", "range": [0, 1], "gridcolor": CHART_GRID},
        legend={
            "orientation": "h",
            "y": -0.28,
            "x": 0,
            "yanchor": "top",
            "xanchor": "left",
            "bgcolor": "rgba(248,250,252,0.94)",
            "bordercolor": CHART_STROKE,
            "borderwidth": 1,
        },
    )
    return apply_ml_plot_theme(fig)


def aggregate_feature_importance(importance: pd.DataFrame) -> pd.DataFrame:
    if importance is None or importance.empty:
        return pd.DataFrame()
    frames = []
    group_cols = [c for c in ("model", "importance_method") if c in importance.columns]
    for _, sub in importance.groupby(group_cols, dropna=False):
        work = sub.copy()
        work["rank"] = pd.to_numeric(work.get("rank"), errors="coerce")
        work["importance"] = pd.to_numeric(work.get("importance"), errors="coerce")
        if work["rank"].notna().any():
            max_rank = float(work["rank"].max())
            work["_consensus_score"] = (max_rank - work["rank"] + 1.0) / max(max_rank, 1.0)
        else:
            max_imp = float(work["importance"].max()) if work["importance"].notna().any() else 1.0
            work["_consensus_score"] = work["importance"] / max(max_imp, 1e-12)
        frames.append(work[["feature", "feature_group", "_consensus_score"]])
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    agg = (
        combined.groupby(["feature", "feature_group"], dropna=False)
        .agg(importance=("_consensus_score", "mean"), model_method_count=("_consensus_score", "size"))
        .reset_index()
        .sort_values("importance", ascending=False, na_position="last")
        .reset_index(drop=True)
    )
    agg["rank"] = np.arange(1, len(agg) + 1)
    agg["target"] = importance["target"].iloc[0] if "target" in importance.columns and not importance.empty else ""
    agg["target_mode"] = importance["target_mode"].iloc[0] if "target_mode" in importance.columns and not importance.empty else ""
    agg["model"] = "Overall"
    agg["importance_method"] = "aggregate_model_consensus"
    agg["selected_fold_count"] = agg["model_method_count"]
    return agg


def render_feature_importance_model_cards(importance: pd.DataFrame, key: str) -> tuple[str, pd.DataFrame]:
    if importance is None or importance.empty:
        return "", pd.DataFrame()
    models = importance["model"].dropna().astype(str).drop_duplicates().tolist() if "model" in importance.columns else []
    options = ["Overall", *models]
    selected = render_button_selector(options, key, label_func=ml_model_label, default="Overall", max_cols=max(len(options), 1))
    if selected == "Overall":
        return selected, aggregate_feature_importance(importance)
    return selected, score_importance_for_selection(importance, str(importance["target"].iloc[0]), str(importance["target_mode"].iloc[0]), selected)


def render_overall_top_predictors_card(importance: pd.DataFrame) -> None:
    if importance is None or importance.empty:
        return
    rows = []
    for _, row in importance.head(10).iterrows():
        value = pd.to_numeric(pd.Series([row.get("importance")]), errors="coerce").iloc[0]
        rows.append(
            "<tr>"
            f"<td style='padding:0.35rem;border-bottom:1px solid rgba(51,65,85,0.7);color:#facc15;font-weight:800;text-align:right;'>{int(row.get('rank', len(rows)+1))}</td>"
            f"<td style='padding:0.35rem;border-bottom:1px solid rgba(51,65,85,0.7);color:#e5e7eb;overflow-wrap:anywhere;'>{escape(str(row.get('feature', '')))}</td>"
            f"<td style='padding:0.35rem;border-bottom:1px solid rgba(51,65,85,0.7);color:#facc15;text-align:right;font-weight:800;'>{format_number(value, 4)}</td>"
            f"<td style='padding:0.35rem;border-bottom:1px solid rgba(51,65,85,0.7);color:#cbd5e1;'>{escape(feature_neuroscience_label(row))}</td>"
            "</tr>"
        )
    st.markdown(
        "<div style='border:1px solid #334155;border-radius:8px;padding:0.75rem;background:#0f172a;margin:0.5rem 0 1rem 0;'>"
        "<h4 style='margin:0 0 0.65rem 0;color:#facc15;text-align:center;'>Overall Top Predictors</h4>"
        "<table style='width:100%;border-collapse:collapse;background:transparent;font-size:0.78rem;'>"
        "<thead><tr><th style='text-align:right;color:#f8fafc;padding:0.3rem;'>Rank</th><th style='text-align:left;color:#f8fafc;padding:0.3rem;'>Feature</th><th style='text-align:right;color:#f8fafc;padding:0.3rem;'>Importance</th><th style='text-align:left;color:#f8fafc;padding:0.3rem;'>Meaning</th></tr></thead>"
        "<tbody>"
        + "".join(rows)
        + "</tbody></table></div>",
        unsafe_allow_html=True,
    )


def render_score_prediction_panel(csv_files: list[Path]) -> bool:
    audit_path = next((p for p in csv_files if p.name == "clinical_score_target_audit.csv"), None)
    matches_path = next((p for p in csv_files if p.name == "score_prediction_target_matches.csv"), None)
    perf_path = next((p for p in csv_files if p.name == "score_prediction_model_performance.csv"), None)
    pred_path = next((p for p in csv_files if p.name == "score_prediction_cv_predictions.csv"), None)
    imp_path = next((p for p in csv_files if p.name == "score_prediction_feature_importance.csv"), None)
    group_path = next((p for p in csv_files if p.name == "score_prediction_group_performance.csv"), None)
    mixed_path = next((p for p in csv_files if p.name == "score_prediction_mixed_model.csv"), None)
    outlier_path = next((p for p in csv_files if p.name == "score_prediction_outlier_audit.csv"), None)
    status_path = next((p for p in csv_files if p.name == "score_prediction_model_status.csv"), None)
    cdr_perf_path = next((p for p in csv_files if p.name == "cdr_classification_model_performance.csv"), None)
    cdr_pred_path = next((p for p in csv_files if p.name == "cdr_classification_cv_predictions.csv"), None)
    cdr_confusion_path = next((p for p in csv_files if p.name == "cdr_classification_confusion_matrix.csv"), None)
    cdr_class_path = next((p for p in csv_files if p.name == "cdr_classification_class_metrics.csv"), None)
    cdr_imp_path = next((p for p in csv_files if p.name == "cdr_classification_feature_importance.csv"), None)
    cdr_fold_path = next((p for p in csv_files if p.name == "cdr_classification_cv_fold_best_params.csv"), None)
    cdr_status_path = next((p for p in csv_files if p.name == "cdr_classification_model_status.csv"), None)
    search_perf_path = current_csv_path(csv_files, "clinical_outcome_model_search_performance.csv")
    search_pred_path = current_csv_path(csv_files, "clinical_outcome_model_search_cv_predictions.csv")
    preprocessing_path = next((p for p in csv_files if p.name == "ml_preprocessing_audit.csv"), None)
    catalog_path = next((p for p in csv_files if p.name == "ml_feature_catalog.csv"), None)
    catalog = _safe_read_csv(catalog_path) if catalog_path is not None else None
    matches = _safe_read_csv(matches_path) if matches_path is not None else None
    audit = _safe_read_csv(audit_path) if audit_path is not None else None
    perf = _safe_read_csv(perf_path) if perf_path is not None else None
    predictions = _safe_read_csv(pred_path) if pred_path is not None else None
    importance = _safe_read_csv(imp_path) if imp_path is not None else None
    group_perf = _safe_read_csv(group_path) if group_path is not None else None
    mixed = _safe_read_csv(mixed_path) if mixed_path is not None else None
    outliers = _safe_read_csv(outlier_path) if outlier_path is not None else None
    status = _safe_read_csv(status_path) if status_path is not None else None
    cdr_perf = _safe_read_csv(cdr_perf_path) if cdr_perf_path is not None else None
    cdr_predictions = _safe_read_csv(cdr_pred_path) if cdr_pred_path is not None else None
    cdr_confusion = _safe_read_csv(cdr_confusion_path) if cdr_confusion_path is not None else None
    cdr_class_metrics = _safe_read_csv(cdr_class_path) if cdr_class_path is not None else None
    cdr_importance = _safe_read_csv(cdr_imp_path) if cdr_imp_path is not None else None
    cdr_fold_params = _safe_read_csv(cdr_fold_path) if cdr_fold_path is not None else None
    cdr_status = _safe_read_csv(cdr_status_path) if cdr_status_path is not None else None
    search_perf = _safe_read_csv(search_perf_path) if search_perf_path is not None else None
    search_predictions = _safe_read_csv(search_pred_path) if search_pred_path is not None else None
    preprocessing = _safe_read_csv(preprocessing_path) if preprocessing_path is not None else None
    audit = filter_active_score_targets(audit)
    matches = filter_active_score_targets(matches)
    perf = filter_regression_score_targets(perf)
    predictions = filter_regression_score_targets(predictions)
    importance = filter_regression_score_targets(importance)
    group_perf = filter_regression_score_targets(group_perf)
    mixed = filter_regression_score_targets(mixed)
    outliers = filter_regression_score_targets(outliers)

    render_yellow_center_heading("CLINICAL OUTCOME PREDICTION")
    st.markdown(
        "<div class='definition-grid'>"
        "<div class='edr-definition-box'><h5>What is predicted?</h5>"
        "<p>MMSE Total Score is modeled as a continuous regression target. Global CDR is modeled as ordinal CDR classes: 0, 0.5, 1, and 2+.</p></div>"
        "<div class='edr-definition-box'><h5>Outlier / class policy</h5>"
        "<p>MMSE regression removes group-wise 1.5x IQR target outliers. CDR classes are not outlier-filtered because high values are severity classes.</p></div>"
        "</div>",
        unsafe_allow_html=True,
    )
    render_feature_group_feature_box(catalog)
    st.caption(
        "Whole-connectome distribution summaries are retained: they summarize full AAL3 edge distributions, while node, ROI, and edge features describe local or edge-specific structure. "
        "The correlation filter removes duplicate summaries before CV."
    )
    st.caption(
        "Preprocessing: median imputation, missingness/sparsity/near-zero-variance/correlation filters, nested-CV feature selection, "
        "and normalization for linear and MLP models. Tree and boosting models use the same filtered features without requiring scaling."
    )

    ok_reg_perf = add_model_identity(perf[perf.get("status", pd.Series("", index=perf.index)).astype(str).eq("ok")].copy()) if perf is not None and not perf.empty else pd.DataFrame()
    ok_cdr_perf = add_model_identity(cdr_perf[cdr_perf.get("status", pd.Series("", index=cdr_perf.index)).astype(str).eq("ok")].copy()) if cdr_perf is not None and not cdr_perf.empty else pd.DataFrame()
    predictions = add_model_identity(predictions) if predictions is not None and not predictions.empty else pd.DataFrame()
    cdr_predictions = add_model_identity(cdr_predictions) if cdr_predictions is not None and not cdr_predictions.empty else pd.DataFrame()
    search_reg_perf = pd.DataFrame()
    search_cdr_perf = pd.DataFrame()
    search_reg_predictions = pd.DataFrame()
    search_cdr_predictions = pd.DataFrame()
    if search_perf is not None and not search_perf.empty and "task_type" in search_perf.columns:
        search_perf = search_perf[search_perf.get("status", pd.Series("", index=search_perf.index)).astype(str).eq("ok")].copy()
        search_perf = filter_no_explicit_age_policy(search_perf)
        search_reg_perf = add_model_identity(search_perf[search_perf["task_type"].astype(str).eq("regression")].copy())
        search_cdr_perf = add_model_identity(search_perf[search_perf["task_type"].astype(str).eq("classification")].copy())
    if search_predictions is not None and not search_predictions.empty and "task_type" in search_predictions.columns:
        search_predictions = filter_no_explicit_age_policy(search_predictions)
        search_reg_predictions = add_model_identity(search_predictions[search_predictions["task_type"].astype(str).eq("regression")].copy())
        search_cdr_predictions = add_model_identity(search_predictions[search_predictions["task_type"].astype(str).eq("classification")].copy())
    ok_reg_perf = pd.concat([ok_reg_perf, search_reg_perf], ignore_index=True, sort=False)
    ok_cdr_perf = pd.concat([ok_cdr_perf, search_cdr_perf], ignore_index=True, sort=False)
    predictions = pd.concat([predictions, search_reg_predictions], ignore_index=True, sort=False)
    cdr_predictions = pd.concat([cdr_predictions, search_cdr_predictions], ignore_index=True, sort=False)
    cdr_confusion = cdr_confusion_from_predictions(cdr_predictions)
    cdr_class_metrics = cdr_class_metrics_from_predictions(cdr_predictions)
    if ok_reg_perf.empty and ok_cdr_perf.empty:
        st.info("Clinical outcome model outputs are not available yet. Run the full dashboard refresh after the code update.")
        return bool(audit is not None and not audit.empty)

    st.info(
        "Primary scan-aligned uses the nearest clinical score within 730 days of the DTI scan and is the preferred paper model. "
        "All-history sensitivity uses the nearest available historical score regardless of scan distance and is exploratory."
    )

    controls, dist_col = st.columns((0.46, 0.54), gap="large")
    target_options = [target for target in ACTIVE_SCORE_TARGETS if (target in REGRESSION_SCORE_TARGETS and not ok_reg_perf.empty) or (target == CDR_CLASSIFICATION_TARGET and not ok_cdr_perf.empty)]
    with controls:
        render_yellow_left_heading("Clinical Score Target")
        selected_target = render_button_selector(target_options, "clinical-outcome-target", label_func=metric_label, default=target_options[0])
        target_perf = ok_cdr_perf if selected_target == CDR_CLASSIFICATION_TARGET else ok_reg_perf[ok_reg_perf["target"].astype(str).eq(selected_target)].copy()
        available_modes = target_perf["target_mode"].dropna().astype(str).drop_duplicates().tolist() if not target_perf.empty else []
        mode_options = [mode for mode in SCORE_MODE_ORDER if mode in available_modes] or available_modes
        st.markdown("### Target matching")
        selected_mode = render_button_selector(
            mode_options,
            f"clinical-outcome-mode-{selected_target}",
            label_func=score_mode_display,
            default=HISTORY_SCORE_MODE if HISTORY_SCORE_MODE in mode_options else (mode_options[0] if mode_options else None),
        )
    with dist_col:
        st.plotly_chart(clinical_target_distribution_figure(matches, selected_target, selected_mode), width="stretch", config={"displaylogo": False})

    if selected_target == CDR_CLASSIFICATION_TARGET:
        all_mode_perf = ok_cdr_perf.loc[ok_cdr_perf["target_mode"].astype(str).eq(selected_mode)].copy()
        mode_perf = clinical_outcome_production_perf(ok_cdr_perf, selected_target, selected_mode, task_type="classification")
        if mode_perf.empty:
            mode_perf = all_mode_perf.copy()
        for col in ("accuracy", "balanced_accuracy", "macro_f1"):
            if col in mode_perf.columns:
                mode_perf[col] = pd.to_numeric(mode_perf[col], errors="coerce")
        sort_cols = [col for col in ("accuracy", "balanced_accuracy", "macro_f1") if col in mode_perf.columns]
        if sort_cols:
            mode_perf = mode_perf.sort_values(sort_cols, ascending=[False] * len(sort_cols), na_position="last")
        mode_perf = mode_perf.head(1).copy()
        model_options = mode_perf["model_key"].dropna().astype(str).drop_duplicates().tolist()
        model_labels = model_display_map(mode_perf)
        best_model = cdr_classification_best_model(ok_cdr_perf, selected_mode)
    else:
        all_mode_perf = ok_reg_perf.loc[
            ok_reg_perf["target"].astype(str).eq(selected_target) & ok_reg_perf["target_mode"].astype(str).eq(selected_mode),
        ].copy()
        mode_perf = clinical_outcome_production_perf(ok_reg_perf, selected_target, selected_mode, task_type="regression")
        if mode_perf.empty:
            mode_perf = all_mode_perf.copy()
        for col in ("r2", "spearman_r", "mae", "rmse"):
            if col in mode_perf.columns:
                mode_perf[col] = pd.to_numeric(mode_perf[col], errors="coerce")
        sort_cols = [col for col in ("r2", "spearman_r") if col in mode_perf.columns]
        if sort_cols:
            mode_perf = mode_perf.sort_values(sort_cols, ascending=[False] * len(sort_cols), na_position="last")
        mode_perf = mode_perf.head(1).copy()
        model_options = mode_perf["model_key"].dropna().astype(str).drop_duplicates().tolist()
        model_labels = model_display_map(mode_perf)
        best_model = score_prediction_best_model(ok_reg_perf, selected_target, selected_mode)
    if not model_options:
        st.warning("No model rows are available for the selected clinical outcome and matching mode.")
        return True
    st.markdown("### Model")
    selected_model = render_button_selector(
        model_options,
        f"clinical-outcome-model-{selected_target}-{selected_mode}",
        label_func=lambda key: model_labels.get(str(key), ml_model_label(str(key).split(MODEL_KEY_SEPARATOR)[-1])),
        default=best_model if best_model in model_options else model_options[0],
    )
    if selected_target == CDR_CLASSIFICATION_TARGET:
        render_selected_model_metrics(ok_cdr_perf, selected_target, selected_mode, selected_model, task_type="classification")
        render_ensemble_explainer(ok_cdr_perf, selected_target, selected_mode, selected_model, task_type="classification")
    else:
        render_selected_model_metrics(ok_reg_perf, selected_target, selected_mode, selected_model, task_type="regression")
        render_ensemble_explainer(ok_reg_perf, selected_target, selected_mode, selected_model, task_type="regression")

    top_n_choice = st.selectbox(
        "Show top model comparison",
        [5, 10, 20, 50, "All"],
        index=1,
        key=f"clinical-outcome-top-n-{selected_target}-{selected_mode}",
        help="The main card stays production-focused; this table lets you inspect more candidate model metrics without changing the selected production model.",
    )
    if selected_target == CDR_CLASSIFICATION_TARGET:
        render_top_models_table(all_mode_perf, selected_target, selected_mode, task_type="classification", top_n=top_n_choice)
    else:
        render_top_models_table(all_mode_perf, selected_target, selected_mode, task_type="regression", top_n=top_n_choice)

    outlier_count = 0
    if selected_target != CDR_CLASSIFICATION_TARGET and outliers is not None and not outliers.empty and "removed_as_outlier" in outliers.columns:
        out_sub = outliers[
            outliers["target"].astype(str).eq(selected_target)
            & outliers["target_mode"].astype(str).eq(selected_mode)
            & outliers["removed_as_outlier"].astype(str).str.lower().isin(("true", "1", "yes"))
        ]
        outlier_count = int(out_sub["subject_id"].nunique()) if "subject_id" in out_sub.columns else int(len(out_sub))
    st.caption(f"Current detail mode: {score_mode_display(selected_mode)} | Outliers removed: {outlier_count if selected_target != CDR_CLASSIFICATION_TARGET else 'not applied to CDR classes'}")

    plot_cols = st.columns(3, gap="large")
    if selected_target == CDR_CLASSIFICATION_TARGET:
        with plot_cols[0]:
            st.plotly_chart(cdr_classification_performance_figure(mode_perf, selected_mode), width="stretch", config={"displaylogo": False})
        with plot_cols[1]:
            st.plotly_chart(cdr_roc_figure(cdr_predictions, selected_mode, selected_model), width="stretch", config={"displaylogo": False})
        with plot_cols[2]:
            st.plotly_chart(cdr_class_metrics_figure(cdr_class_metrics, selected_mode, selected_model), width="stretch", config={"displaylogo": False})
    else:
        with plot_cols[0]:
            st.plotly_chart(score_prediction_performance_figure(mode_perf, selected_target, selected_mode), width="stretch", config={"displaylogo": False})
        with plot_cols[1]:
            st.plotly_chart(score_prediction_true_pred_figure(predictions, selected_target, selected_mode, selected_model), width="stretch", config={"displaylogo": False})
            st.caption("Dashed line is the ideal y=x reference; points are held-out MMSE predictions.")
        with plot_cols[2]:
            st.plotly_chart(score_prediction_residual_figure(predictions, selected_target, selected_mode, selected_model), width="stretch", config={"displaylogo": False})

    render_yellow_center_heading("FEATURE IMPORTANCE")
    target_importance = pd.DataFrame()
    if selected_target == CDR_CLASSIFICATION_TARGET and cdr_importance is not None and not cdr_importance.empty:
        target_importance = cdr_importance[cdr_importance["target_mode"].astype(str).eq(selected_mode)].copy()
    elif importance is not None and not importance.empty:
        target_importance = importance[
            importance["target"].astype(str).eq(selected_target)
            & importance["target_mode"].astype(str).eq(selected_mode)
        ].copy()
    if target_importance.empty:
        st.info("No feature-importance rows are available for the selected clinical outcome.")
    else:
        target_importance = add_model_identity(target_importance)
        model_specific_importance = target_importance[target_importance["model_key"].astype(str).eq(str(selected_model))].copy()
        if model_specific_importance.empty:
            selected_importance = aggregate_feature_importance(target_importance)
            st.caption(
                "Production ensemble feature importance is not separately available in the current artifacts, "
                "so this shows an aggregate structural-feature consensus. Model-specific experimental rows are in Advanced."
            )
        else:
            selected_importance = model_specific_importance
            st.caption("Feature importance is shown for the selected production model.")
        render_overall_top_predictors_card(selected_importance)
        render_feature_importance_cards(selected_importance)
        render_predictor_interpretation_table(selected_importance)

    render_yellow_center_heading("ADVANCED ANALYTIC APPENDIX")
    with st.expander("Target availability audit", expanded=False):
        if audit is not None and not audit.empty:
            st.dataframe(audit, width="stretch", hide_index=True, height=420)
        else:
            st.info("Clinical target availability audit is not available.")
    with st.expander("Held-out rows and per-group details", expanded=False):
        if selected_target == CDR_CLASSIFICATION_TARGET:
            if cdr_predictions is not None and not cdr_predictions.empty:
                show_pred = cdr_predictions[
                    cdr_predictions["target_mode"].astype(str).eq(selected_mode)
                    & cdr_predictions["model_key"].astype(str).eq(selected_model)
                ].copy()
                st.markdown("**CDR held-out prediction rows**")
                st.dataframe(show_pred.head(800), width="stretch", hide_index=True, height=360)
            if cdr_class_metrics is not None and not cdr_class_metrics.empty:
                st.markdown("**CDR per-class metrics**")
                sub_class = cdr_class_metrics[
                    cdr_class_metrics["target_mode"].astype(str).eq(selected_mode)
                    & cdr_class_metrics["model_key"].astype(str).eq(selected_model)
                ].copy()
                st.dataframe(sub_class, width="stretch", hide_index=True, height=compact_table_height(len(sub_class)))
        else:
            if predictions is not None and not predictions.empty:
                show_pred = predictions[
                    predictions["target"].astype(str).eq(selected_target)
                    & predictions["target_mode"].astype(str).eq(selected_mode)
                    & predictions["model_key"].astype(str).eq(selected_model)
                ].copy()
                display_cols = [c for c in ("fold", "subject_id", "group", "true_score", "predicted_score", "residual", "abs_day_delta", "clinical_source") if c in show_pred.columns]
                st.markdown("**MMSE held-out prediction rows**")
                st.dataframe(show_pred[display_cols].head(800), width="stretch", hide_index=True, height=360)
            if group_perf is not None and not group_perf.empty:
                group_perf = add_model_identity(group_perf)
                sub_group = group_perf[
                    group_perf["target"].astype(str).eq(selected_target)
                    & group_perf["target_mode"].astype(str).eq(selected_mode)
                    & group_perf["model_key"].astype(str).eq(selected_model)
                ].copy()
                st.markdown("**Per-group regression error**")
                st.dataframe(sub_group, width="stretch", hide_index=True, height=compact_table_height(len(sub_group)))
    with st.expander("Mixed model, outlier audit, and preprocessing", expanded=False):
        if selected_target != CDR_CLASSIFICATION_TARGET and mixed is not None and not mixed.empty:
            mixed = add_model_identity(mixed)
            sub_mixed = mixed[
                mixed["target"].astype(str).eq(selected_target)
                & mixed["target_mode"].astype(str).eq(selected_mode)
                & mixed["model_key"].astype(str).eq(selected_model)
            ].copy()
            st.markdown("**Mixed-model / robust OLS check**")
            st.dataframe(sub_mixed, width="stretch", hide_index=True, height=compact_table_height(len(sub_mixed)))
        if selected_target != CDR_CLASSIFICATION_TARGET and outliers is not None and not outliers.empty:
            st.markdown("**MMSE outlier audit**")
            sub_outliers = outliers[
                outliers["target"].astype(str).eq(selected_target)
                & outliers["target_mode"].astype(str).eq(selected_mode)
            ].copy()
            st.dataframe(sub_outliers, width="stretch", hide_index=True, height=360)
        if preprocessing is not None and not preprocessing.empty:
            st.markdown("**Preprocessing audit**")
            st.dataframe(preprocessing, width="stretch", hide_index=True, height=compact_table_height(len(preprocessing)))
    with st.expander("Model availability and fold parameters", expanded=False):
        active_status = cdr_status if selected_target == CDR_CLASSIFICATION_TARGET else status
        score_fold_path = next((p for p in csv_files if p.name == "score_prediction_cv_fold_best_params.csv"), None)
        active_fold = cdr_fold_params if selected_target == CDR_CLASSIFICATION_TARGET else (_safe_read_csv(score_fold_path) if score_fold_path is not None else None)
        if active_status is not None and not active_status.empty:
            st.markdown("**Model availability**")
            st.dataframe(active_status, width="stretch", hide_index=True, height=compact_table_height(len(active_status)))
        if active_fold is not None and not active_fold.empty:
            st.markdown("**Fold parameters**")
            st.dataframe(active_fold, width="stretch", hide_index=True, height=360)
    with st.expander("Hidden model experiments and search rows", expanded=False):
        hidden_perf = all_mode_perf.copy() if "all_mode_perf" in locals() else pd.DataFrame()
        if not hidden_perf.empty:
            st.markdown("**All no-age candidate rows for this target/mode**")
            show_cols = [
                c
                for c in (
                    "task_type",
                    "target",
                    "target_mode",
                    "analysis_role",
                    "feature_policy",
                    "model",
                    "status",
                    "n_subjects",
                    "accuracy",
                    "balanced_accuracy",
                    "macro_f1",
                    "macro_recall_sensitivity",
                    "macro_specificity",
                    "r2",
                    "mae",
                    "rmse",
                    "spearman_r",
                )
                if c in hidden_perf.columns
            ]
            st.dataframe(hidden_perf[show_cols].head(500), width="stretch", hide_index=True, height=420)
        else:
            st.info("No hidden model experiment rows are available for this selection.")
        if "target_importance" in locals() and target_importance is not None and not target_importance.empty:
            st.markdown("**All available feature-importance rows for this target/mode**")
            st.dataframe(target_importance.head(800), width="stretch", hide_index=True, height=420)
    with st.expander("Additional analysis notes", expanded=False):
        st.markdown(
            "- MMSE is the only active continuous score-regression target.\n"
            "- Global CDR is displayed as ordinal multiclass classification, so accuracy, balanced accuracy, macro F1, sensitivity, and specificity replace R2.\n"
            "- Matrix summaries remain in training because they are full-connectome distribution summaries and are correlation-filtered before supervised selection."
        )
    if perf_path is not None or cdr_perf_path is not None:
        source = cdr_perf_path if selected_target == CDR_CLASSIFICATION_TARGET else perf_path
        if source is not None:
            st.caption(f"Live clinical-outcome source: `{source.relative_to(ANALYSIS_ROOT).as_posix()}`.")
    render_enhanced_score_results()
    return True


def render_enhanced_score_results() -> bool:
    enhanced_perf = enhanced_ml_table(("enhanced_score_model_performance.csv", "neural_score_model_performance.csv"))
    enhanced_imp = enhanced_ml_table(("enhanced_score_feature_importance.csv", "neural_score_feature_importance.csv"))
    enhanced_perf = filter_regression_score_targets(enhanced_perf)
    enhanced_imp = filter_regression_score_targets(enhanced_imp)
    enhanced_perf = filter_structural_score_policy(enhanced_perf)
    enhanced_imp = filter_structural_score_policy(enhanced_imp)
    if enhanced_perf.empty:
        return False
    with st.expander("Advanced details: enhanced goal-search and neural score models", expanded=False):
        st.caption(
            "Experimental score rows are filtered to structural-connectome feature policies only. "
            "Age, demographics, diagnosis labels, APOE, phase, and clinical-score predictors are excluded from active score reporting."
        )
        show = enhanced_perf.copy()
        for col in ("r2", "mae", "rmse", "spearman_r", "pearson_r"):
            if col in show.columns:
                show[col] = pd.to_numeric(show[col], errors="coerce")
        if "r2" in show.columns:
            show = show.sort_values(["r2", "spearman_r"], ascending=[False, False], na_position="last")
        display_cols = [
            c
            for c in (
                "run_source",
                "target",
                "target_mode",
                "feature_set",
                "feature_policy",
                "model",
                "status",
                "n_subjects",
                "n_subjects_before_outlier_filter",
                "n_outliers_removed",
                "n_features",
                "r2",
                "mae",
                "rmse",
                "spearman_r",
                "pearson_r",
            )
            if c in show.columns
        ]
        st.dataframe(
            show[display_cols].head(120),
            width="stretch",
            hide_index=True,
            height=420,
            column_config={
                "r2": st.column_config.NumberColumn("R2", format="%.3f"),
                "mae": st.column_config.NumberColumn("MAE", format="%.3f"),
                "rmse": st.column_config.NumberColumn("RMSE", format="%.3f"),
                "spearman_r": st.column_config.NumberColumn("Spearman r", format="%.3f"),
                "pearson_r": st.column_config.NumberColumn("Pearson r", format="%.3f"),
            },
        )
        if not enhanced_imp.empty:
            st.markdown("**Enhanced score feature importance**")
            st.dataframe(
                enhanced_imp.head(800),
                width="stretch",
                hide_index=True,
                height=420,
                column_config={"importance": st.column_config.NumberColumn("Importance", format="%.4f")},
            )
    return True


def render_ml_diagnostics_panel(csv_files: list[Path]) -> bool:
    st.markdown(
        "<div style='display:flex;justify-content:center;margin:0.1rem 0 0.7rem 0;'>"
        "<div style='color:#cbd5e1;font-weight:800;font-size:0.86rem;letter-spacing:0;'>Choose ML workflow</div>"
        "</div>",
        unsafe_allow_html=True,
    )
    left_pad, selector_col, right_pad = st.columns((0.75, 3.8, 0.75), gap="medium")
    workflow_labels = {
        "score": "Clinical Outcome Prediction",
        "disease": "Disease Prediction",
        "age": "Age Analysis",
    }
    with selector_col:
        selected_view = render_button_selector(
            ["score", "disease", "age"],
            "ml-diagnostics-view",
            label_func=lambda key: workflow_labels.get(str(key), str(key)),
            default="score",
            max_cols=3,
        )
    st.markdown("<div style='height:1px;background:#2c313a;margin:0.35rem 0 1rem 0;'></div>", unsafe_allow_html=True)

    if selected_view == "disease":
        return render_disease_prediction_panel(csv_files)
    if selected_view == "age":
        return render_age_analysis_panel(csv_files)
    return render_score_prediction_panel(csv_files)


def render_disease_prediction_panel(csv_files: list[Path]) -> bool:
    perf_path = next((p for p in csv_files if p.name == "ml_model_performance.csv"), None)
    status_path = next((p for p in csv_files if p.name == "ml_model_status.csv"), None)
    group_path = next((p for p in csv_files if p.name == "ml_feature_group_summary.csv"), None)
    catalog_path = next((p for p in csv_files if p.name == "ml_feature_catalog.csv"), None)
    imp_path = next((p for p in csv_files if p.name == "ml_feature_importance.csv"), None)
    shap_path = next((p for p in csv_files if p.name == "ml_shap_contributions.csv"), None)
    pred_path = next((p for p in csv_files if p.name == "ml_cross_validated_predictions.csv"), None)
    confusion_path = next((p for p in csv_files if p.name == "ml_confusion_matrix.csv"), None)
    roc_path = next((p for p in csv_files if p.name == "ml_roc_curve.csv"), None)
    class_path = next((p for p in csv_files if p.name == "ml_class_metrics.csv"), None)
    feature_path = next((p for p in csv_files if p.name == "ml_subject_feature_matrix.csv"), None)
    perf = _safe_read_csv(perf_path) if perf_path is not None else None
    model_status = _safe_read_csv(status_path) if status_path is not None else None
    group_summary = _safe_read_csv(group_path) if group_path is not None else None
    catalog = _safe_read_csv(catalog_path) if catalog_path is not None else None
    importance = _safe_read_csv(imp_path) if imp_path is not None else None
    shap = _safe_read_csv(shap_path) if shap_path is not None else None
    predictions = _safe_read_csv(pred_path) if pred_path is not None else None
    confusion = _safe_read_csv(confusion_path) if confusion_path is not None else None
    roc = _safe_read_csv(roc_path) if roc_path is not None else None
    class_metrics = _safe_read_csv(class_path) if class_path is not None else None
    features = _safe_read_csv(feature_path) if feature_path is not None else None
    if perf is None or perf.empty:
        st.info("ML diagnostic outputs are not available yet. Run the full dashboard refresh after structural sections finish.")
        return False

    st.subheader("ML Diagnostic Prediction")
    render_ml_context_cards(features)
    if feature_path is not None:
        st.caption(f"Live ML feature source: `{feature_path.relative_to(ANALYSIS_ROOT).as_posix()}`.")

    if group_summary is not None and not group_summary.empty:
        st.markdown("### Feature Audit")
        st.dataframe(
            group_summary,
            width="stretch",
            hide_index=True,
            height=compact_table_height(len(group_summary)),
            column_config={
                "feature_group": "Feature group",
                "features": st.column_config.NumberColumn("Features", format="%d"),
                "selected": st.column_config.NumberColumn("Selected", format="%d"),
                "dropped": st.column_config.NumberColumn("Dropped", format="%d"),
                "reason": "Reason",
            },
        )

    st.markdown("### Model Performance")
    left, right = st.columns((0.58, 0.42), gap="large")
    best_model = ml_diagnostics_best_model(perf)
    with left:
        st.plotly_chart(ml_diagnostics_performance_figure(perf), width="stretch", config={"displaylogo": False})
    with right:
        st.markdown(
            f"<div class='edr-inference-box'>{escape(ml_diagnostics_performance_inference(perf))}</div>",
            unsafe_allow_html=True,
        )
        perf_cols = [
            c
            for c in (
                "model",
                "status",
                "n_subjects",
                "n_features_input",
                "n_outer_folds",
                "n_inner_folds",
                "accuracy",
                "balanced_accuracy",
                "macro_auroc_ovr",
                "log_loss",
                "macro_f1",
                "weighted_f1",
                "macro_precision",
                "macro_recall",
            )
            if c in perf.columns
        ]
        st.dataframe(
            perf[perf_cols],
            width="stretch",
            hide_index=True,
            height=compact_table_height(len(perf)),
            column_config={
                "model": st.column_config.TextColumn("Model"),
                "accuracy": st.column_config.NumberColumn("Accuracy", format="%.3f"),
                "balanced_accuracy": st.column_config.NumberColumn("Balanced accuracy", format="%.3f"),
                "macro_auroc_ovr": st.column_config.NumberColumn("Macro AUROC", format="%.3f"),
                "log_loss": st.column_config.NumberColumn("Log loss", format="%.3f"),
                "macro_f1": st.column_config.NumberColumn("Macro F1", format="%.3f"),
                "weighted_f1": st.column_config.NumberColumn("Weighted F1", format="%.3f"),
                "macro_precision": st.column_config.NumberColumn("Macro precision", format="%.3f"),
                "macro_recall": st.column_config.NumberColumn("Macro recall", format="%.3f"),
            },
        )
        if model_status is not None and not model_status.empty:
            with st.expander("Classifier model availability and neural architecture log", expanded=False):
                st.dataframe(model_status, width="stretch", hide_index=True, height=compact_table_height(len(model_status)))

    model_options = perf["model"].dropna().astype(str).drop_duplicates().tolist()
    st.markdown("### Model")
    selected_model = render_button_selector(
        model_options,
        "ml-diagnostics-disease-model-details",
        label_func=ml_model_label,
        default=best_model if best_model in model_options else (model_options[0] if model_options else None),
    )
    render_disease_selected_model_metrics(perf, selected_model, class_metrics)

    cm_col, roc_col = st.columns((0.5, 0.5), gap="large")
    with cm_col:
        if confusion is not None and not confusion.empty:
            st.plotly_chart(ml_diagnostics_confusion_figure(confusion, selected_model), width="stretch", config={"displaylogo": False})
        else:
            st.info("Confusion matrix output is not available yet.")
    with roc_col:
        if roc is not None and not roc.empty:
            st.plotly_chart(ml_diagnostics_roc_figure(roc, selected_model), width="stretch", config={"displaylogo": False})
        else:
            st.info("ROC curve output is not available yet.")

    if class_metrics is not None and not class_metrics.empty:
        st.markdown("### Class-wise Metrics")
        cmets = class_metrics[class_metrics["model"].astype(str).eq(selected_model)].copy()
        st.dataframe(
            cmets,
            width="stretch",
            hide_index=True,
            height=compact_table_height(len(cmets)),
            column_config={
                "precision": st.column_config.NumberColumn("Precision", format="%.3f"),
                "recall_sensitivity": st.column_config.NumberColumn("Sensitivity / recall", format="%.3f"),
                "specificity": st.column_config.NumberColumn("Specificity", format="%.3f"),
                "f1": st.column_config.NumberColumn("F1", format="%.3f"),
            },
        )

    st.markdown("### Feature Importance")
    top_n = st.slider("Top features to show", 10, 60, 25, 5, key="ml-diagnostics-top-features")
    imp_col, shap_col = st.columns((0.5, 0.5), gap="large")
    with imp_col:
        if importance is not None and not importance.empty:
            st.plotly_chart(
                ml_diagnostics_importance_figure(importance, selected_model, top_n, value_col="importance"),
                width="stretch",
                config={"displaylogo": False},
            )
        else:
            st.info("Feature importance output is not available yet.")
    with shap_col:
        if shap is not None and not shap.empty and "mean_abs_contribution" in shap.columns:
            st.plotly_chart(
                ml_diagnostics_importance_figure(shap, selected_model, top_n, value_col="mean_abs_contribution"),
                width="stretch",
                config={"displaylogo": False},
            )
        else:
            st.info("TreeSHAP contribution output is not available; use model feature importance for this refresh.")

    if catalog is not None and not catalog.empty:
        st.markdown("### Selected And Dropped Features")
        with st.expander("Feature catalog with filtering reasons", expanded=True):
            show = catalog.copy()
            if "selected_fold_count" in show.columns:
                show["selected_fold_count"] = pd.to_numeric(show["selected_fold_count"], errors="coerce").fillna(0)
                show = show.sort_values(["selected_fold_count", "final_status", "feature_group"], ascending=[False, True, True])
            display_cols = [
                c
                for c in (
                    "feature_group",
                    "feature",
                    "source_file",
                    "missingness",
                    "variance",
                    "sparse_presence",
                    "correlation_cluster",
                    "selected_fold_count",
                    "elastic_net_nonzero_fold_count",
                    "final_status",
                    "drop_reason",
                )
                if c in show.columns
            ]
            st.dataframe(
                show[display_cols].head(700),
                width="stretch",
                height=520,
                hide_index=True,
                column_config={
                    "missingness": st.column_config.NumberColumn("Missingness", format="%.3f"),
                    "variance": st.column_config.NumberColumn("Variance", format="%.3g"),
                    "sparse_presence": st.column_config.NumberColumn("Presence", format="%.3f"),
                    "selected_fold_count": st.column_config.NumberColumn("Selected folds", format="%d"),
                    "elastic_net_nonzero_fold_count": st.column_config.NumberColumn("Elastic-net nonzero folds", format="%d"),
                },
            )

    st.markdown("### Subject-level Predictions")
    if predictions is not None and not predictions.empty:
        selected_predictions = predictions[predictions["model"].astype(str).eq(selected_model)].copy()
        prediction_cols = [
            c
            for c in ("fold", "subject_id", "true_group", "predicted_group", "prob_CN", "prob_MCI", "prob_AD")
            if c in selected_predictions.columns
        ]
        st.dataframe(
            selected_predictions[prediction_cols].head(800),
            width="stretch",
            height=430,
            hide_index=True,
            column_config={
                "prob_CN": st.column_config.NumberColumn("P(CN)", format="%.3f"),
                "prob_MCI": st.column_config.NumberColumn("P(MCI)", format="%.3f"),
                "prob_AD": st.column_config.NumberColumn("P(AD)", format="%.3f"),
            },
        )
    else:
        st.info("Cross-validated prediction rows are not available yet.")
    if perf_path is not None:
        st.caption(f"Live ML performance source: `{perf_path.relative_to(ANALYSIS_ROOT).as_posix()}`.")
    render_enhanced_disease_results()
    return True


def render_enhanced_disease_results() -> bool:
    enhanced_perf = enhanced_ml_table(("enhanced_disease_model_performance.csv", "neural_disease_model_performance.csv"))
    enhanced_class = enhanced_ml_table(("enhanced_disease_class_metrics.csv", "neural_disease_class_metrics.csv"))
    enhanced_imp = enhanced_ml_table(("enhanced_disease_feature_importance.csv", "neural_disease_feature_importance.csv"))
    if enhanced_perf.empty:
        return False
    st.markdown("### Enhanced Goal-search Disease Models")
    st.caption(
        "Experimental feature-set sweep including structural-only, metadata, clinical-assisted, combined, neural, and augmented-model rows. "
        "Use the feature policy column to separate structural-only from clinically assisted performance."
    )
    show = enhanced_perf.copy()
    for col in ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "macro_auroc_ovr"):
        if col in show.columns:
            show[col] = pd.to_numeric(show[col], errors="coerce")
    if "accuracy" in show.columns:
        show = show.sort_values(["accuracy", "macro_f1"], ascending=[False, False], na_position="last")
    display_cols = [
        c
        for c in (
            "run_source",
            "feature_set",
            "feature_policy",
            "model",
            "status",
            "n_subjects",
            "n_features",
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "weighted_f1",
            "macro_auroc_ovr",
        )
        if c in show.columns
    ]
    st.dataframe(
        show[display_cols].head(120),
        width="stretch",
        hide_index=True,
        height=420,
        column_config={
            "accuracy": st.column_config.NumberColumn("Accuracy", format="%.3f"),
            "balanced_accuracy": st.column_config.NumberColumn("Balanced accuracy", format="%.3f"),
            "macro_f1": st.column_config.NumberColumn("Macro F1", format="%.3f"),
            "weighted_f1": st.column_config.NumberColumn("Weighted F1", format="%.3f"),
            "macro_auroc_ovr": st.column_config.NumberColumn("Macro AUROC", format="%.3f"),
        },
    )
    if not enhanced_class.empty:
        with st.expander("Enhanced class-wise F1 / recall / precision", expanded=False):
            st.dataframe(
                enhanced_class.head(800),
                width="stretch",
                hide_index=True,
                height=420,
                column_config={
                    "precision": st.column_config.NumberColumn("Precision", format="%.3f"),
                    "recall": st.column_config.NumberColumn("Recall", format="%.3f"),
                    "recall_sensitivity": st.column_config.NumberColumn("Recall", format="%.3f"),
                    "f1": st.column_config.NumberColumn("F1", format="%.3f"),
                },
            )
    if not enhanced_imp.empty:
        with st.expander("Enhanced disease feature importance", expanded=False):
            st.dataframe(
                enhanced_imp.head(800),
                width="stretch",
                hide_index=True,
                height=420,
                column_config={"importance": st.column_config.NumberColumn("Importance", format="%.4f")},
            )
    return True


def render_edr_exceptions_panel(section: Section, csv_files: list[Path], max_charts: int, hide_visual_outliers: bool) -> bool:
    render_edr_exception_definition()
    rendered = render_edr_subject_metric_tabs(section, csv_files, hide_visual_outliers)
    ranked = render_edr_exception_rankings(csv_files)
    return rendered or ranked


NOVEL_Q_THRESHOLD = 0.05
NOVEL_Q_COLUMNS = ("kw_q", "welch_q", "bm_q", "mwu_q", "q_value", "disease_gradient_q", "best_q")
NOVEL_THALAMIC_GRAPH_NODES = ("Thal_MDm_L", "Thal_IL_L", "Thal_MDl_L", "Thal_VL_L", "Thal_VA_L", "Thal_PuM_L")
NOVEL_LIMBIC_CINGULATE_NODES = (
    "orbitofrontal/OFC",
    "ACC_sub_R",
    "nucleus accumbens",
    "insula",
    "hippocampus/parahippocampal",
    "amygdala",
    "LC",
    "raphe",
)
NOVEL_GLOBAL_DISCONNECTION_METRICS = (
    "global FA lower in AD",
    "MD/RD/AD higher in AD",
    "density lower in AD",
    "global efficiency lower in AD",
    "mean strength lower in AD",
)
NOVEL_DISTANCE_METRICS = (
    "short-range and long-range efficiency",
    "length-derived delay proxies",
    "EDR residual mean",
    "EDR lambda",
    "disease-gradient edges",
)
NOVEL_SYSTEM_COLORS = {
    "Thalamic": "#f97316",
    "Limbic / neuromodulatory": "#14b8a6",
    "Cingulate": "#8b5cf6",
    "Cerebellar": "#64748b",
    "Posterior / sensorimotor": "#0ea5e9",
    "Other": "#94a3b8",
}
NOVEL_READABLE_TEXT = "#000000"
NOVEL_LR_SR_CLASS_COLORS = {"SR": "#06b6d4", "LR": "#a855f7"}
NOVEL_LR_SR_SIGNAL_OPTIONS = ("Exception rate", "Exception strength", "All-edge structural weight")
NOVEL_LR_SR_RANGE_OPTIONS = ("SR + LR", "SR only", "LR only")
NOVEL_LITERATURE_ANCHORS = (
    (
        "ADNI DWI connectomics: diffusion/tractography has progression signal, but DWI-only prediction is limited.",
        "https://www.mdpi.com/2076-3417/14/16/7001",
    ),
    (
        "Whole-brain graph meta-analysis: AD spectrum shows reduced structural integration and segregation.",
        "https://www.sciencedirect.com/science/article/pii/S0149763425001745",
    ),
    (
        "AAL3 graph/ML AD paper: 166-node AAL3 graph features and thalamic/subcortical signals are relevant.",
        "https://www.frontiersin.org/journals/neuroinformatics/articles/10.3389/fninf.2024.1384720/full",
    ),
    (
        "Thalamic nuclei in AD stages: medial/MDm thalamic nuclei are affected across disease stages.",
        "https://pubmed.ncbi.nlm.nih.gov/39029273/",
    ),
    (
        "AAL3 atlas: fine thalamic, ACC, neuromodulatory, and subcortical parcellation basis.",
        "https://www.sciencedirect.com/science/article/pii/S1053811919307803",
    ),
)


def novel_csv_path(csv_files: list[Path], name: str) -> Path | None:
    return current_csv_path(csv_files, name)


def novel_csv(csv_files: list[Path], name: str) -> pd.DataFrame:
    path = novel_csv_path(csv_files, name)
    if path is None:
        return pd.DataFrame()
    df = _safe_read_csv(path)
    return df if df is not None else pd.DataFrame()


def novel_numeric(df: pd.DataFrame, columns: tuple[str, ...] | list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def novel_node_label_map() -> dict[int, str]:
    labels = aal_label_lookup(str(AAL_LABEL_CSV))
    if labels.empty or not {"node", "atlas_label"}.issubset(labels.columns):
        return {}
    rows = labels.dropna(subset=["node"]).copy()
    return {int(row["node"]): str(row["atlas_label"]) for _, row in rows.iterrows()}


def novel_region_system(region: str) -> str:
    lower = str(region).lower()
    if "thal" in lower:
        return "Thalamic"
    if lower.startswith("n_acc") or "n_acc" in lower:
        return "Limbic / neuromodulatory"
    if "cingulate" in lower or lower.startswith("acc") or "acc_" in lower:
        return "Cingulate"
    limbic_terms = (
        "hippocampus",
        "parahippocampal",
        "amygdala",
        "insula",
        "ofc",
        "frontal_med_orb",
        "rectus",
        "temporal",
        "olfactory",
        "n_acc",
        "lc_",
        "raphe",
    )
    if any(term in lower for term in limbic_terms):
        return "Limbic / neuromodulatory"
    if "cerebellum" in lower or "vermis" in lower:
        return "Cerebellar"
    if "precuneus" in lower or "postcentral" in lower or "precentral" in lower:
        return "Posterior / sensorimotor"
    return "Other"


def novel_add_node_labels(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "node" not in df.columns:
        return df
    out = df.copy()
    label_map = novel_node_label_map()
    out["node"] = pd.to_numeric(out["node"], errors="coerce").astype("Int64")
    out["AAL3 region"] = [label_map.get(int(node), f"AAL_{int(node):03d}") if pd.notna(node) else "NA" for node in out["node"]]
    out["System"] = out["AAL3 region"].map(novel_region_system)
    out["AAL node"] = [f"AAL_{int(node):03d}" if pd.notna(node) else "NA" for node in out["node"]]
    return out


def novel_q_filtered(df: pd.DataFrame) -> pd.DataFrame:
    q_cols = [col for col in NOVEL_Q_COLUMNS if col in df.columns]
    if not q_cols:
        return pd.DataFrame()
    out = novel_numeric(df, q_cols)
    out["Evidence q"] = out[q_cols].min(axis=1)
    return out[out["Evidence q"].lt(NOVEL_Q_THRESHOLD)].copy()


def novel_node_evidence(csv_files: list[Path], specs: tuple[tuple[str, str, str], ...], top_per_metric: int = 5) -> pd.DataFrame:
    frames = []
    for file_name, metric, family in specs:
        df = novel_csv(csv_files, file_name)
        if df.empty or "node" not in df.columns or "kw_q" not in df.columns:
            continue
        work = novel_q_filtered(df)
        if work.empty:
            continue
        work = work.sort_values("Evidence q", na_position="last").head(top_per_metric).copy()
        work["Metric"] = metric
        work["Family"] = family
        frames.append(work)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True, sort=False)
    out = novel_add_node_labels(out)
    out["KW p"] = pd.to_numeric(out.get("kw_p"), errors="coerce")
    out["KW stat"] = pd.to_numeric(out.get("kw_stat"), errors="coerce")
    display_cols = ["Family", "Metric", "System", "AAL3 region", "AAL node", "n", "KW stat", "KW p", "Evidence q"]
    return out[[col for col in display_cols if col in out.columns]].copy()


def novel_direction_text(row: pd.Series) -> str:
    higher = row.get("higher_group")
    if isinstance(higher, str) and higher.strip():
        return f"{higher.strip()} higher"
    group_a = str(row.get("group_a", "A")).strip()
    group_b = str(row.get("group_b", "B")).strip()
    diff = pd.to_numeric(pd.Series([row.get("median_diff_a_minus_b")]), errors="coerce").iloc[0]
    if pd.isna(diff):
        return "direction shown by medians"
    if diff > 0:
        return f"{group_a} higher"
    if diff < 0:
        return f"{group_b} higher"
    return "similar medians"


def novel_pairwise_evidence(
    csv_files: list[Path],
    specs: tuple[tuple[str, str, str], ...],
    *,
    top_per_metric: int = 2,
    ad_only: bool = True,
) -> pd.DataFrame:
    frames = []
    for file_name, metric, message in specs:
        df = novel_csv(csv_files, file_name)
        if df.empty or not {"group_a", "group_b"}.issubset(df.columns):
            continue
        work = novel_q_filtered(df)
        if work.empty:
            continue
        if ad_only:
            ad_mask = work["group_a"].astype(str).eq("AD") | work["group_b"].astype(str).eq("AD")
            work = work[ad_mask].copy()
        if work.empty:
            continue
        work = novel_numeric(work, ("median_a", "median_b", "median_diff_a_minus_b", "cliffs_delta", "Evidence q"))
        work = work.sort_values("Evidence q", na_position="last").head(top_per_metric).copy()
        work["Finding"] = metric
        work["Story role"] = message
        work["Comparison"] = work["group_a"].astype(str) + " vs " + work["group_b"].astype(str)
        work["Direction"] = work.apply(novel_direction_text, axis=1)
        frames.append(work)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True, sort=False)
    display_cols = [
        "Finding",
        "Comparison",
        "Direction",
        "median_a",
        "median_b",
        "median_diff_a_minus_b",
        "cliffs_delta",
        "Evidence q",
        "Story role",
    ]
    return out[[col for col in display_cols if col in out.columns]].copy()


def novel_edge_gradient_evidence(csv_files: list[Path], top_n: int = 8) -> pd.DataFrame:
    df = novel_csv(csv_files, "top_disease_gradient_edges.csv")
    if df.empty:
        df = novel_csv(csv_files, "edr_edge_disease_gradient_fd_sum_len_mean.csv")
    if df.empty or not {"i", "j", "disease_gradient_q"}.issubset(df.columns):
        return pd.DataFrame()
    work = novel_q_filtered(df)
    if work.empty:
        return pd.DataFrame()
    label_map = novel_node_label_map()
    work = novel_numeric(work, ("i", "j", "disease_gradient_rho", "disease_gradient_p", "disease_gradient_q", "n"))
    work = work.sort_values("Evidence q", na_position="last").head(top_n).copy()
    work["ROI A"] = [label_map.get(int(node), f"AAL_{int(node):03d}") if pd.notna(node) else "NA" for node in work["i"]]
    work["ROI B"] = [label_map.get(int(node), f"AAL_{int(node):03d}") if pd.notna(node) else "NA" for node in work["j"]]
    work["Direction"] = np.where(pd.to_numeric(work["disease_gradient_rho"], errors="coerce") < 0, "declines CN to MCI to AD", "rises CN to MCI to AD")
    display_cols = ["i", "j", "ROI A", "ROI B", "Direction", "disease_gradient_rho", "disease_gradient_p", "disease_gradient_q", "n"]
    return work[[col for col in display_cols if col in work.columns]].copy()


def novel_model_sort(work: pd.DataFrame, sort_cols: tuple[str, ...], ascending: tuple[bool, ...]) -> pd.DataFrame:
    out = work.copy()
    for col in sort_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    existing = [col for col in sort_cols if col in out.columns]
    if existing:
        asc = [ascending[sort_cols.index(col)] for col in existing]
        out = out.sort_values(existing, ascending=asc, na_position="last")
    return out


def novel_regression_model_evidence(csv_files: list[Path]) -> pd.DataFrame:
    frames = []
    score = novel_csv(csv_files, "score_prediction_model_performance.csv")
    if not score.empty:
        work = score[
            score.get("status", pd.Series("", index=score.index)).astype(str).eq("ok")
            & score.get("target", pd.Series("", index=score.index)).astype(str).eq("MMSE Total Score")
            & score.get("target_mode", pd.Series("", index=score.index)).astype(str).eq(PRIMARY_SCORE_MODE)
        ].copy()
        work = novel_model_sort(work, ("r2", "spearman_r", "rmse"), (False, False, True)).head(3)
        if not work.empty:
            work["Workflow"] = "Current structural CV"
            frames.append(work)
    search = novel_csv(csv_files, "clinical_outcome_model_search_performance.csv")
    if not search.empty:
        work = search[
            search.get("status", pd.Series("", index=search.index)).astype(str).eq("ok")
            & search.get("task_type", pd.Series("", index=search.index)).astype(str).eq("regression")
            & search.get("target", pd.Series("", index=search.index)).astype(str).eq("MMSE Total Score")
            & search.get("target_mode", pd.Series("", index=search.index)).astype(str).eq(PRIMARY_SCORE_MODE)
            & search.get("feature_policy", pd.Series("", index=search.index)).astype(str).eq("posthoc_oof_ensemble_exploratory")
        ].copy()
        work = novel_model_sort(work, ("r2", "spearman_r", "rmse"), (False, False, True)).head(3)
        if not work.empty:
            work["Workflow"] = "Posthoc OOF ensemble"
            frames.append(work)
    enhanced = enhanced_ml_table(("enhanced_score_model_performance.csv", "neural_score_model_performance.csv"))
    enhanced = filter_regression_score_targets(enhanced)
    enhanced = filter_structural_score_policy(enhanced)
    if enhanced is not None and not enhanced.empty:
        work = enhanced[enhanced.get("status", pd.Series("", index=enhanced.index)).astype(str).eq("ok")].copy()
        work = novel_model_sort(work, ("r2", "spearman_r", "rmse"), (False, False, True)).head(3)
        if not work.empty:
            work["Workflow"] = work.get("run_source", pd.Series(["Enhanced structural compact"] * len(work), index=work.index)).astype(str)
            frames.append(work)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True, sort=False)
    for col in ("r2", "mae", "rmse", "spearman_r", "pearson_r", "n_subjects", "n_features", "n_features_input"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    out["Model"] = out["model"].map(ml_model_label)
    if "n_features" not in out.columns and "n_features_input" in out.columns:
        out["n_features"] = out["n_features_input"]
    display_cols = ["Workflow", "target", "target_mode", "Model", "n_subjects", "n_features", "r2", "mae", "rmse", "spearman_r", "pearson_r"]
    return out[[col for col in display_cols if col in out.columns]].copy()


def novel_cdr_model_evidence(csv_files: list[Path]) -> pd.DataFrame:
    frames = []
    cdr = novel_csv(csv_files, "cdr_classification_model_performance.csv")
    if not cdr.empty:
        work = cdr[
            cdr.get("status", pd.Series("", index=cdr.index)).astype(str).eq("ok")
            & cdr.get("target", pd.Series("", index=cdr.index)).astype(str).eq(CDR_CLASSIFICATION_TARGET)
            & cdr.get("target_mode", pd.Series("", index=cdr.index)).astype(str).eq(PRIMARY_SCORE_MODE)
        ].copy()
        work = novel_model_sort(work, ("macro_auroc_ovr", "accuracy", "balanced_accuracy"), (False, False, False)).head(3)
        if not work.empty:
            work["Workflow"] = "Current structural CV"
            frames.append(work)
    search = novel_csv(csv_files, "clinical_outcome_model_search_performance.csv")
    if not search.empty:
        work = search[
            search.get("status", pd.Series("", index=search.index)).astype(str).eq("ok")
            & search.get("task_type", pd.Series("", index=search.index)).astype(str).eq("classification")
            & search.get("target", pd.Series("", index=search.index)).astype(str).eq(CDR_CLASSIFICATION_TARGET)
            & search.get("feature_policy", pd.Series("", index=search.index)).astype(str).eq("posthoc_oof_ensemble_exploratory")
        ].copy()
        work = novel_model_sort(work, ("accuracy", "balanced_accuracy", "macro_f1"), (False, False, False)).head(3)
        if not work.empty:
            work["Workflow"] = "Posthoc OOF ensemble"
            frames.append(work)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True, sort=False)
    for col in ("accuracy", "balanced_accuracy", "macro_f1", "macro_auroc_ovr", "n_subjects", "n_features", "n_features_input"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    out["Model"] = out["model"].map(ml_model_label)
    if "n_features" not in out.columns and "n_features_input" in out.columns:
        out["n_features"] = out["n_features_input"]
    display_cols = ["Workflow", "target", "target_mode", "Model", "n_subjects", "n_features", "accuracy", "balanced_accuracy", "macro_f1", "macro_auroc_ovr"]
    return out[[col for col in display_cols if col in out.columns]].copy()


def novel_signal_feature_evidence(csv_files: list[Path], top_n: int = 10) -> pd.DataFrame:
    df = novel_csv(csv_files, "clinical_outcome_signal_feature_audit.csv")
    if df.empty or "feature" not in df.columns:
        return pd.DataFrame()
    work = novel_numeric(df, ("abs_spearman_r", "spearman_r", "selected_fold_count", "feature_missingness_in_target_cohort", "n_subjects", "eta_squared_by_cdr_class"))
    if "selected_fold_count" in work.columns:
        selected = work["selected_fold_count"].fillna(0).gt(0)
        if selected.any():
            work = work[selected].copy()
    work = work.sort_values(["abs_spearman_r", "selected_fold_count"], ascending=[False, False], na_position="last").head(top_n).copy()
    label_map = novel_node_label_map()

    def label_feature(feature: str) -> str:
        match = re.search(r"aal[_-]?(\d+)", str(feature).lower())
        if not match:
            return metric_label(str(feature))
        node = int(match.group(1))
        region = label_map.get(node, f"AAL_{node:03d}")
        return f"{metric_label(str(feature).split('__')[0])} {region} (AAL_{node:03d})"

    work["Feature label"] = work["feature"].map(label_feature)
    display_cols = [
        "target",
        "target_mode",
        "feature_group",
        "Feature label",
        "n_subjects",
        "selected_fold_count",
        "spearman_r",
        "abs_spearman_r",
        "feature_missingness_in_target_cohort",
        "eta_squared_by_cdr_class",
    ]
    return work[[col for col in display_cols if col in work.columns]].copy()


def novel_count_fdr_rows(csv_files: list[Path], names: tuple[str, ...]) -> int:
    count = 0
    for name in names:
        df = novel_csv(csv_files, name)
        if df.empty:
            continue
        count += len(novel_q_filtered(df))
    return count


def novel_metric_card_html(label: str, value: str, note: str) -> str:
    return (
        "<div class='novel-stat'>"
        f"<div class='novel-stat-label'>{escape(label)}</div>"
        f"<div class='novel-stat-value'>{escape(value)}</div>"
        f"<div class='novel-stat-note'>{escape(note)}</div>"
        "</div>"
    )


def novel_join(items: tuple[str, ...] | list[str]) -> str:
    return ", ".join(str(item) for item in items)


def render_novel_mechanism_map() -> None:
    steps = (
        ("AAL3 fine parcellation", "Subdivides thalamus, ACC, nucleus accumbens, LC, raphe, and related regions."),
        ("Vulnerable nodes", "FDR-supported thalamic graph and limbic/cingulate local DTI signals."),
        ("Network disconnection", "Reduced density, global efficiency, and weighted strength with DTI disruption."),
        ("Distance/EDR layer", "Pathway length, delay proxies, distance-decay residuals, and gradient edges."),
        ("Clinical severity", "Structural-only models predict MMSE and support Global CDR classification."),
    )
    step_html = "".join(
        "<div class='novel-mechanism-step'>"
        f"<b>{escape(title)}</b>"
        f"<span>{escape(text)}</span>"
        "</div>"
        for title, text in steps
    )
    st.markdown(
        "<div class='novel-mechanism'>"
        "<div class='novel-mechanism-title'>Manuscript mechanism model</div>"
        f"<div class='novel-mechanism-flow'>{step_html}</div>"
        "</div>",
        unsafe_allow_html=True,
    )


def render_novel_evidence_strip(source: str, result: str) -> None:
    st.markdown(
        "<div class='novel-evidence-strip'>"
        f"<div class='novel-evidence-source'><b>Evidence:</b> {escape(source)}</div>"
        f"<div class='novel-evidence-result'>{escape(result)}</div>"
        "</div>",
        unsafe_allow_html=True,
    )


def render_novel_ribbon(items: tuple[str, ...] | list[str]) -> None:
    html = "".join(f"<div class='novel-ribbon-item'>{escape(item)}</div>" for item in items)
    st.markdown(f"<div class='novel-ribbon'>{html}</div>", unsafe_allow_html=True)


def render_novel_gradient_edge_explainer(gradient_rows: pd.DataFrame) -> None:
    work = gradient_rows.copy()
    if not work.empty:
        if "disease_gradient_rho" in work.columns:
            work["disease_gradient_rho"] = pd.to_numeric(work["disease_gradient_rho"], errors="coerce")
        if "disease_gradient_q" in work.columns:
            work["disease_gradient_q"] = pd.to_numeric(work["disease_gradient_q"], errors="coerce")
    mapped_edges = len(work)
    decline_edges = int(work["disease_gradient_rho"].lt(0).sum()) if "disease_gradient_rho" in work.columns else 0
    rise_edges = int(work["disease_gradient_rho"].gt(0).sum()) if "disease_gradient_rho" in work.columns else 0
    node_cols = [col for col in ("i", "j") if col in work.columns]
    unique_nodes = 0
    if node_cols:
        node_values = pd.Series(work[node_cols].to_numpy().ravel())
        unique_nodes = int(node_values.dropna().nunique())
    top_edge = "Top edge unavailable"
    if not work.empty and "disease_gradient_q" in work.columns:
        top = work.sort_values("disease_gradient_q", na_position="last").iloc[0]
        top_edge = (
            f"{top.get('ROI A', 'ROI A')} - {top.get('ROI B', 'ROI B')} "
            f"(q={format_p(top.get('disease_gradient_q'))})"
        )
    shell_path = mni_brain_mask_path()
    shell_text = "Translucent MNI152 template brain mask; anatomical context only, not a subject CT/MRI or tract surface."
    if shell_path is None:
        shell_text = "MNI brain shell unavailable; points remain atlas-centroid coordinates."
    items = (
        ("What is plotted", f"{mapped_edges} FDR-supported gradient edges across {unique_nodes} AAL3 endpoint nodes."),
        ("Color direction", f"Blue edges decline from CN to MCI to AD; red edges rise. Current map: {decline_edges} blue, {rise_edges} red."),
        ("Line thickness", "Thicker lines have lower FDR q-values, so they are stronger disease-gradient evidence."),
        ("MRI context", shell_text),
        ("Strongest displayed edge", top_edge),
    )
    html = "".join(
        "<div class='novel-edge-guide-item'>"
        f"<b>{escape(title)}</b>"
        f"<span>{escape(text)}</span>"
        "</div>"
        for title, text in items
    )
    st.markdown(f"<div class='novel-edge-guide'>{html}</div>", unsafe_allow_html=True)


def novel_lr_sr_subject_profile(csv_files: list[Path]) -> pd.DataFrame:
    specs = (
        ("lr_sr_subject_level_fd_sum_len_mean.csv", "short_w_mean", "SR weight", "Weight", "SR"),
        ("lr_sr_subject_level_fd_sum_len_mean.csv", "long_w_mean", "LR weight", "Weight", "LR"),
        ("lr_sr_subject_level_fd_sum_len_mean.csv", "short_global_eff", "SR efficiency", "Efficiency", "SR"),
        ("lr_sr_subject_level_fd_sum_len_mean.csv", "long_global_eff", "LR efficiency", "Efficiency", "LR"),
        ("edr_exception_subject_level_fd_sum_len_mean.csv", "sr_exception_pct", "SR exception rate", "Exception rate", "SR"),
        ("edr_exception_subject_level_fd_sum_len_mean.csv", "lr_exception_pct", "LR exception rate", "Exception rate", "LR"),
    )
    frames = []
    for file_name, metric, label, domain, distance_class in specs:
        df = novel_csv(csv_files, file_name)
        if df.empty or metric not in df.columns or "group" not in df.columns:
            continue
        work = df[["group", metric]].copy()
        work[metric] = pd.to_numeric(work[metric], errors="coerce")
        work = work[work["group"].isin(GROUP_ORDER) & work[metric].notna()].copy()
        if work.empty:
            continue
        summary = work.groupby("group", observed=False)[metric].agg(["median", "count"]).reset_index()
        cn_median = pd.to_numeric(summary.loc[summary["group"].eq("CN"), "median"], errors="coerce")
        baseline = float(cn_median.iloc[0]) if not cn_median.empty and pd.notna(cn_median.iloc[0]) else np.nan
        if not np.isfinite(baseline) or baseline == 0:
            summary["CN-normalized median"] = summary["median"]
        else:
            summary["CN-normalized median"] = summary["median"] / baseline
        summary["Metric"] = label
        summary["Domain"] = domain
        summary["Range"] = distance_class
        frames.append(summary)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True, sort=False)
    out["group"] = pd.Categorical(out["group"], categories=GROUP_ORDER, ordered=True)
    metric_order = {name: idx for idx, name in enumerate([spec[2] for spec in specs])}
    out["metric_order"] = out["Metric"].map(metric_order)
    return out.sort_values(["metric_order", "group"]).reset_index(drop=True)


def novel_lr_sr_group_profile_figure(csv_files: list[Path]) -> go.Figure:
    frame = novel_lr_sr_subject_profile(csv_files)
    if frame.empty:
        return novel_empty_figure("LR/SR group profile", "No LR/SR subject-level rows are available.", height=430)
    fig = go.Figure()
    color_map = dict(zip(GROUP_ORDER, GROUP_COLORS))
    for group in GROUP_ORDER:
        group_df = frame[frame["group"].astype(str).eq(group)].copy()
        if group_df.empty:
            continue
        fig.add_trace(
            go.Bar(
                x=group_df["Metric"],
                y=group_df["CN-normalized median"],
                name=group,
                marker_color=color_map.get(group, "#64748b"),
                customdata=group_df[["Range", "Domain", "median", "count"]],
                hovertemplate=(
                    "%{x}<br>%{customdata[0]} / %{customdata[1]}"
                    "<br>median=%{customdata[2]:.4g}<br>n=%{customdata[3]:.0f}"
                    "<br>CN-normalized=%{y:.3g}<extra></extra>"
                ),
            )
        )
    fig.add_hline(y=1.0, line={"color": "#475569", "width": 1.2, "dash": "dot"})
    fig.update_layout(
        title={
            "text": "<b>LR/SR group profile</b><br><sup>Values are group medians normalized to CN within each metric.</sup>",
            "x": 0.01,
            "xanchor": "left",
        },
        height=470,
        margin={"l": 70, "r": 28, "t": 78, "b": 112},
        barmode="group",
        xaxis={"title": "", "tickangle": -22, "gridcolor": CHART_GRID},
        yaxis={"title": "CN-normalized median", "gridcolor": CHART_GRID},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1.0,
            "font": {"color": NOVEL_READABLE_TEXT, "size": 12},
        },
    )
    return novel_apply_readable_layout(fig)


def novel_lr_sr_exception_edge_frame(csv_files: list[Path], top_per_class: int = 10) -> pd.DataFrame:
    df = novel_csv(csv_files, "edr_exception_edge_group_map_fd_sum_len_mean.csv")
    if df.empty or not {"i", "j", "edr_class", "comparison"}.issubset(df.columns):
        return pd.DataFrame()
    work = df[df["comparison"].isin(("CN vs AD", "MCI vs AD")) & df["edr_class"].isin(("SR", "LR"))].copy()
    if work.empty:
        return pd.DataFrame()
    for col in (
        "i",
        "j",
        "length_median_mm",
        "exception_rate_CN",
        "exception_rate_MCI",
        "exception_rate_AD",
        "median_exception_strength_CN",
        "median_exception_strength_MCI",
        "median_exception_strength_AD",
        "p_value",
        "q_value",
    ):
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")

    def other_group(comparison: str) -> str:
        return "CN" if str(comparison).startswith("CN") else "MCI"

    work["Comparator"] = work["comparison"].map(other_group)
    work["Comparator exception rate"] = np.where(
        work["Comparator"].eq("CN"),
        work.get("exception_rate_CN"),
        work.get("exception_rate_MCI"),
    )
    work["AD exception rate"] = work.get("exception_rate_AD")
    work["AD minus comparator"] = work["AD exception rate"] - work["Comparator exception rate"]
    work["Abs exception-rate delta"] = work["AD minus comparator"].abs()
    work["Comparator exception strength"] = np.where(
        work["Comparator"].eq("CN"),
        work.get("median_exception_strength_CN"),
        work.get("median_exception_strength_MCI"),
    )
    work["AD exception strength"] = work.get("median_exception_strength_AD")
    work["Strength delta"] = work["AD exception strength"] - work["Comparator exception strength"]
    work["Evidence q"] = pd.to_numeric(work.get("q_value"), errors="coerce")
    work["Evidence p"] = pd.to_numeric(work.get("p_value"), errors="coerce")
    work = work[work[["i", "j", "AD exception rate", "Comparator exception rate"]].notna().all(axis=1)].copy()
    if work.empty:
        return pd.DataFrame()
    ranked = []
    for edr_class, class_df in work.groupby("edr_class", dropna=False):
        ranked.append(
            class_df.sort_values(
                ["Evidence q", "Evidence p", "Abs exception-rate delta"],
                ascending=[True, True, False],
                na_position="last",
            ).head(top_per_class)
        )
    out = pd.concat(ranked, ignore_index=True, sort=False) if ranked else pd.DataFrame()
    if out.empty:
        return out
    out["i"] = out["i"].astype(int)
    out["j"] = out["j"].astype(int)
    out["Class label"] = out["edr_class"].map({"SR": "Short-range exception", "LR": "Long-range exception"}).fillna(out["edr_class"].astype(str))
    return out.sort_values(["edr_class", "Evidence q", "Evidence p"], na_position="last").reset_index(drop=True)


def render_novel_lr_sr_exception_explainer(
    csv_files: list[Path],
    signal: str = "Exception rate",
    range_class: str = "SR + LR",
    max_edges_per_group: int = 150,
) -> None:
    signal_frame = novel_lr_sr_three_group_signal_frame(csv_files, signal)
    visible = novel_lr_sr_filter_signal_frame(signal_frame, range_class, max_edges_per_group) if not signal_frame.empty else pd.DataFrame()
    profile = novel_lr_sr_subject_profile(csv_files)
    profile_msg = "Subject-level SR/LR rows unavailable."
    if not profile.empty:
        ad_sr = profile[profile["group"].astype(str).eq("AD") & profile["Metric"].eq("SR weight")]
        ad_lr_eff = profile[profile["group"].astype(str).eq("AD") & profile["Metric"].eq("LR efficiency")]
        parts = []
        if not ad_sr.empty:
            parts.append(f"AD SR weight is {float(ad_sr['CN-normalized median'].iloc[0]):.2g}x CN")
        if not ad_lr_eff.empty:
            parts.append(f"AD LR efficiency is {float(ad_lr_eff['CN-normalized median'].iloc[0]):.2g}x CN")
        if parts:
            profile_msg = "; ".join(parts) + "."
    visible_counts = visible.groupby("group").size().to_dict() if not visible.empty else {}
    visible_msg = ", ".join(f"{group}: {int(visible_counts.get(group, 0))}" for group in GROUP_ORDER)
    if signal == "Exception rate":
        signal_source = "EDR exception group map"
    elif signal == "Exception strength":
        signal_source = "EDR exception edge-level weights when group-map medians are zero"
    else:
        signal_source = "cached fd_sum/len_mean matrix aggregation"
    items = (
        ("Group profile", profile_msg),
        ("Three-panel map", f"{signal}; {range_class}; visible edges per panel after Top-N cap: {visible_msg}."),
        ("Color coding", "Cyan = short-range exception edge; purple = long-range exception edge; yellow = AAL3 endpoint node."),
        ("Thickness and nodes", "Edge width/opacity scales with selected signal; node size is incident visible-edge count; node color is cumulative signal."),
        ("Signal source", signal_source),
    )
    html = "".join(
        "<div class='novel-edge-guide-item'>"
        f"<b>{escape(title)}</b>"
        f"<span>{escape(text)}</span>"
        "</div>"
        for title, text in items
    )
    st.markdown(f"<div class='novel-edge-guide'>{html}</div>", unsafe_allow_html=True)


def novel_lr_sr_thresholds(csv_files: list[Path]) -> tuple[float, float] | None:
    df = novel_csv(csv_files, "lr_sr_thresholds.csv")
    if df.empty or not {"short_max_mm", "medium_max_mm"}.issubset(df.columns):
        return None
    short_max = pd.to_numeric(df["short_max_mm"], errors="coerce").dropna()
    medium_max = pd.to_numeric(df["medium_max_mm"], errors="coerce").dropna()
    if short_max.empty or medium_max.empty:
        return None
    return float(short_max.iloc[0]), float(medium_max.iloc[0])


def novel_lr_sr_class_from_length(lengths: np.ndarray, short_max_mm: float, medium_max_mm: float) -> np.ndarray:
    classes = np.full(len(lengths), "MR", dtype=object)
    finite = np.isfinite(lengths)
    classes[finite & (lengths <= short_max_mm)] = "SR"
    classes[finite & (lengths > medium_max_mm)] = "LR"
    return classes


def novel_lr_sr_exception_group_signal_frame(csv_files: list[Path], signal: str) -> pd.DataFrame:
    df = novel_csv(csv_files, "edr_exception_edge_group_map_fd_sum_len_mean.csv")
    if df.empty or not {"i", "j", "edr_class"}.issubset(df.columns):
        return pd.DataFrame()
    value_prefix = {
        "Exception rate": "exception_rate",
        "Exception strength": "median_exception_strength",
    }.get(signal)
    if value_prefix is None:
        return pd.DataFrame()
    work = df[df["edr_class"].isin(("SR", "LR"))].copy()
    if work.empty:
        return pd.DataFrame()
    for col in ("i", "j", "length_median_mm", "length_valid_subjects", "p_value", "q_value"):
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")
    frames = []
    for group in GROUP_ORDER:
        value_col = f"{value_prefix}_{group}"
        if value_col not in work.columns:
            continue
        group_df = work.copy()
        group_df["group"] = group
        group_df["Signal value"] = pd.to_numeric(group_df[value_col], errors="coerce")
        group_df = group_df[group_df["Signal value"].notna() & group_df["Signal value"].gt(0)].copy()
        if group_df.empty:
            continue
        frames.append(group_df)
    if not frames:
        return pd.DataFrame()
    long_df = pd.concat(frames, ignore_index=True, sort=False)
    agg = (
        long_df.groupby(["group", "i", "j", "edr_class", "roi_a", "roi_b"], dropna=False)
        .agg(
            **{
                "Signal value": ("Signal value", "max"),
                "Median length mm": ("length_median_mm", "median"),
                "n_subjects": ("length_valid_subjects", "max"),
                "Evidence q": ("q_value", "min"),
                "Evidence p": ("p_value", "min"),
                "Comparison support": ("comparison", lambda s: ", ".join(dict.fromkeys(map(str, s.dropna())))),
            }
        )
        .reset_index()
    )
    agg["Signal"] = signal
    agg["Source"] = "EDR exception group map"
    return agg


def novel_lr_sr_exception_strength_edge_level_frame(csv_files: list[Path]) -> pd.DataFrame:
    df = novel_csv(csv_files, "edr_exception_edge_level_fd_sum_len_mean.csv")
    if df.empty or not {"group", "i", "j", "edr_class", "weight"}.issubset(df.columns):
        return pd.DataFrame()
    work = df[df["group"].astype(str).isin(GROUP_ORDER) & df["edr_class"].isin(("SR", "LR"))].copy()
    if work.empty:
        return pd.DataFrame()
    for col in ("i", "j", "length_mm", "weight"):
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")
    work = work[work[["i", "j", "weight"]].notna().all(axis=1) & work["weight"].gt(0)].copy()
    if work.empty:
        return pd.DataFrame()
    subject_col = "subject_id" if "subject_id" in work.columns else "group"
    out = (
        work.groupby(["group", "i", "j", "edr_class", "roi_a", "roi_b"], dropna=False)
        .agg(
            **{
                "Signal value": ("weight", "median"),
                "Median length mm": ("length_mm", "median"),
                "n_subjects": (subject_col, "nunique"),
            }
        )
        .reset_index()
    )
    out["Signal"] = "Exception strength"
    out["Source"] = "EDR exception edge-level weights"
    out["Nonzero fraction"] = np.nan
    out["Evidence q"] = np.nan
    out["Evidence p"] = np.nan
    out["Comparison support"] = "group median exception-edge weight"
    return out


@st.cache_data(ttl=3600, show_spinner="Aggregating all-edge LR/SR structural weights from connectome matrices...")
def novel_lr_sr_weight_edge_frame_cached(
    subject_table_path: str,
    connectomes_dir: str,
    short_max_mm: float,
    medium_max_mm: float,
) -> pd.DataFrame:
    subject_path = Path(subject_table_path)
    matrix_dir = Path(connectomes_dir)
    if not subject_path.exists() or not matrix_dir.exists():
        return pd.DataFrame()
    subjects = pd.read_csv(subject_path)
    if subjects.empty or not {"subject_id", "group"}.issubset(subjects.columns):
        return pd.DataFrame()
    subjects = subjects[["subject_id", "group"]].dropna().drop_duplicates()
    subjects = subjects[subjects["group"].astype(str).isin(GROUP_ORDER)].copy()
    if subjects.empty:
        return pd.DataFrame()

    def matrix_path(subject_id: str, weight: str) -> Path | None:
        matches = sorted(matrix_dir.glob(f"SC_AAL166_{subject_id}_*_{weight}.csv"))
        return matches[0] if matches else None

    tri_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    group_weights: dict[str, dict[int, list[np.ndarray]]] = {group: {} for group in GROUP_ORDER}
    group_lengths: dict[str, dict[int, list[np.ndarray]]] = {group: {} for group in GROUP_ORDER}
    edge_nodes: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    for row in subjects.itertuples(index=False):
        subject_id = str(getattr(row, "subject_id"))
        group = str(getattr(row, "group"))
        fd_path = matrix_path(subject_id, "fd_sum")
        len_path = matrix_path(subject_id, "len_mean")
        if fd_path is None or len_path is None:
            continue
        try:
            fd = pd.read_csv(fd_path, header=None).to_numpy(dtype=np.float32, copy=False)
            length = pd.read_csv(len_path, header=None).to_numpy(dtype=np.float32, copy=False)
        except Exception:
            continue
        if fd.ndim != 2 or length.ndim != 2:
            continue
        n = int(min(fd.shape[0], fd.shape[1], length.shape[0], length.shape[1]))
        if n < 2:
            continue
        if n not in tri_cache:
            tri_cache[n] = np.triu_indices(n, k=1)
            edge_nodes[n] = (tri_cache[n][0] + 1, tri_cache[n][1] + 1)
        tri = tri_cache[n]
        group_weights[group].setdefault(n, []).append(np.asarray(fd[:n, :n][tri], dtype=np.float32))
        group_lengths[group].setdefault(n, []).append(np.asarray(length[:n, :n][tri], dtype=np.float32))

    frames = []
    for group in GROUP_ORDER:
        if not group_weights[group] or not group_lengths[group]:
            continue
        for n, weight_arrays in group_weights[group].items():
            length_arrays = group_lengths[group].get(n, [])
            if not weight_arrays or not length_arrays:
                continue
            weights = np.vstack(weight_arrays)
            lengths = np.vstack(length_arrays)
            node_i, node_j = edge_nodes.get(n, (None, None))
            if node_i is None or node_j is None:
                continue
            median_length = np.nanmedian(lengths, axis=0)
            edge_class = novel_lr_sr_class_from_length(median_length, short_max_mm, medium_max_mm)
            keep = np.isin(edge_class, ("SR", "LR"))
            if not keep.any():
                continue
            median_weight = np.nanmedian(weights, axis=0)
            nonzero_fraction = np.nanmean(np.where(np.isfinite(weights), weights > 0, np.nan), axis=0)
            finite_count = np.isfinite(weights).sum(axis=0)
            frame = pd.DataFrame(
                {
                    "group": group,
                    "i": node_i,
                    "j": node_j,
                    "edr_class": edge_class,
                    "Signal value": median_weight,
                    "Median length mm": median_length,
                    "n_subjects": finite_count,
                    "Nonzero fraction": nonzero_fraction,
                }
            )
            frame = frame[keep & frame["Signal value"].notna() & frame["Signal value"].gt(0)].copy()
            if frame.empty:
                continue
            frame["Signal"] = "All-edge structural weight"
            frame["Source"] = "fd_sum matrices grouped by len_mean SR/LR class"
            frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def novel_lr_sr_weight_edge_frame(csv_files: list[Path]) -> pd.DataFrame:
    subject_path = novel_csv_path(csv_files, "lr_sr_subject_level_fd_sum_len_mean.csv")
    thresholds = novel_lr_sr_thresholds(csv_files)
    if subject_path is None or thresholds is None:
        return pd.DataFrame()
    frame = novel_lr_sr_weight_edge_frame_cached(str(subject_path), str(CONNECTOMES_DIR), thresholds[0], thresholds[1])
    if frame.empty:
        return frame
    labels = novel_node_label_map()
    frame = frame.copy()
    frame["roi_a"] = frame["i"].map(labels).fillna(frame["i"].map(lambda value: f"AAL_{int(value):03d}"))
    frame["roi_b"] = frame["j"].map(labels).fillna(frame["j"].map(lambda value: f"AAL_{int(value):03d}"))
    frame["Evidence q"] = np.nan
    frame["Evidence p"] = np.nan
    frame["Comparison support"] = "group median all-edge weight"
    return frame


def novel_lr_sr_three_group_signal_frame(csv_files: list[Path], signal: str) -> pd.DataFrame:
    if signal == "All-edge structural weight":
        frame = novel_lr_sr_weight_edge_frame(csv_files)
    elif signal == "Exception strength":
        frame = novel_lr_sr_exception_group_signal_frame(csv_files, signal)
        if frame.empty:
            frame = novel_lr_sr_exception_strength_edge_level_frame(csv_files)
    else:
        frame = novel_lr_sr_exception_group_signal_frame(csv_files, signal)
    if frame.empty:
        return pd.DataFrame()
    out = frame.copy()
    out["i"] = pd.to_numeric(out["i"], errors="coerce")
    out["j"] = pd.to_numeric(out["j"], errors="coerce")
    out["Signal value"] = pd.to_numeric(out["Signal value"], errors="coerce")
    out = out[out["group"].astype(str).isin(GROUP_ORDER) & out["edr_class"].isin(("SR", "LR")) & out["i"].notna() & out["j"].notna() & out["Signal value"].notna()]
    if out.empty:
        return pd.DataFrame()
    out["i"] = out["i"].astype(int)
    out["j"] = out["j"].astype(int)
    out["Signal"] = signal
    return out.reset_index(drop=True)


def novel_lr_sr_filter_signal_frame(frame: pd.DataFrame, range_class: str, max_edges_per_group: int) -> pd.DataFrame:
    if frame.empty:
        return frame
    if range_class == "SR only":
        work = frame[frame["edr_class"].eq("SR")].copy()
    elif range_class == "LR only":
        work = frame[frame["edr_class"].eq("LR")].copy()
    else:
        work = frame[frame["edr_class"].isin(("SR", "LR"))].copy()
    if work.empty:
        return work
    selected = []
    for group in GROUP_ORDER:
        group_df = work[work["group"].astype(str).eq(group)].copy()
        if group_df.empty:
            continue
        if range_class == "SR + LR":
            per_class = max(1, int(max_edges_per_group) // 2)
            chosen_parts = []
            chosen_index: set[int] = set()
            for edr_class in ("SR", "LR"):
                class_df = group_df[group_df["edr_class"].eq(edr_class)].sort_values("Signal value", ascending=False).head(per_class)
                chosen_parts.append(class_df)
                chosen_index.update(map(int, class_df.index))
            chosen = pd.concat(chosen_parts, ignore_index=False, sort=False) if chosen_parts else pd.DataFrame()
            remaining_slots = int(max_edges_per_group) - len(chosen)
            if remaining_slots > 0:
                fill = group_df.loc[~group_df.index.isin(chosen_index)].sort_values("Signal value", ascending=False).head(remaining_slots)
                chosen = pd.concat([chosen, fill], ignore_index=False, sort=False)
        else:
            chosen = group_df.sort_values("Signal value", ascending=False).head(int(max_edges_per_group))
        selected.append(chosen)
    return pd.concat(selected, ignore_index=True, sort=False) if selected else pd.DataFrame()


def novel_node_number_series(df: pd.DataFrame) -> pd.Series:
    if "node" in df.columns:
        return pd.to_numeric(df["node"], errors="coerce")
    if "AAL node" in df.columns:
        return pd.to_numeric(df["AAL node"].astype(str).str.extract(r"(\d+)")[0], errors="coerce")
    return pd.Series(np.nan, index=df.index)


def novel_node_plot_frame(nodes: pd.DataFrame) -> pd.DataFrame:
    if nodes.empty:
        return pd.DataFrame()
    work = nodes.copy()
    work["node_num"] = novel_node_number_series(work)
    work["Evidence q"] = pd.to_numeric(work.get("Evidence q"), errors="coerce")
    work = work[work["node_num"].notna() & work["Evidence q"].notna()].copy()
    if work.empty:
        return pd.DataFrame()
    work["node_num"] = work["node_num"].astype(int)
    work["neg_log10_q"] = -np.log10(work["Evidence q"].clip(lower=1e-300))
    grouped = (
        work.groupby(["node_num", "AAL3 region", "System"], dropna=False)
        .agg(
            **{
                "Best q": ("Evidence q", "min"),
                "-log10(q)": ("neg_log10_q", "max"),
                "Families": ("Family", lambda s: ", ".join(dict.fromkeys(map(str, s.dropna())))),
                "Metrics": ("Metric", lambda s: ", ".join(dict.fromkeys(map(str, s.dropna())))),
            }
        )
        .reset_index()
    )
    coords = aal_centroids(str(AAL_ATLAS_NII), str(AAL_LABEL_CSV))
    if coords.empty or not {"node", "x", "y", "z"}.issubset(coords.columns):
        return pd.DataFrame()
    coords = coords[["node", "x", "y", "z"]].copy()
    coords["node"] = pd.to_numeric(coords["node"], errors="coerce").astype("Int64")
    out = grouped.merge(coords, left_on="node_num", right_on="node", how="left")
    out = out[out[["x", "y", "z"]].notna().all(axis=1)].copy()
    if out.empty:
        return pd.DataFrame()
    score = pd.to_numeric(out["-log10(q)"], errors="coerce").fillna(1.0)
    if score.max() > score.min():
        out["Marker size"] = 7 + 16 * (score - score.min()) / (score.max() - score.min())
    else:
        out["Marker size"] = 11
    return out


def novel_empty_figure(title: str, message: str, height: int = 420) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font={"color": CHART_TEXT, "size": 16},
    )
    fig.update_layout(
        title={"text": f"<b>{escape(title)}</b>", "x": 0.01, "xanchor": "left"},
        paper_bgcolor=CHART_BACKGROUND,
        plot_bgcolor=CHART_BACKGROUND,
        font={"color": CHART_TEXT},
        height=height,
        margin={"l": 20, "r": 20, "t": 64, "b": 20},
    )
    return novel_apply_readable_layout(fig)


def novel_apply_readable_layout(fig: go.Figure, *, height: int | None = None, margin: dict | None = None) -> go.Figure:
    layout: dict[str, object] = {
        "paper_bgcolor": CHART_BACKGROUND,
        "plot_bgcolor": CHART_BACKGROUND,
        "font": {"color": NOVEL_READABLE_TEXT, "size": 13},
        "title_font": {"color": NOVEL_READABLE_TEXT, "size": 18},
    }
    if height is not None:
        layout["height"] = height
    if margin is not None:
        layout["margin"] = margin
    fig.update_layout(**layout)
    axis_common = {
        "automargin": True,
        "color": NOVEL_READABLE_TEXT,
        "gridcolor": CHART_GRID,
        "linecolor": CHART_STROKE,
        "zerolinecolor": CHART_STROKE,
        "tickfont": {"color": NOVEL_READABLE_TEXT, "size": 12},
        "title_font": {"color": NOVEL_READABLE_TEXT, "size": 14},
    }
    fig.update_xaxes(**axis_common)
    fig.update_yaxes(**axis_common)
    return fig


def novel_apply_readable_scene(fig: go.Figure) -> go.Figure:
    scene_axis = {
        "backgroundcolor": CHART_BACKGROUND,
        "gridcolor": CHART_GRID,
        "linecolor": CHART_STROKE,
        "zerolinecolor": CHART_STROKE,
        "tickfont": {"color": NOVEL_READABLE_TEXT, "size": 11},
        "title_font": {"color": NOVEL_READABLE_TEXT, "size": 12},
    }
    fig.update_layout(
        scene={
            "xaxis": {**scene_axis, "title": "MNI x"},
            "yaxis": {**scene_axis, "title": "MNI y"},
            "zaxis": {**scene_axis, "title": "MNI z"},
            "aspectmode": "data",
        }
    )
    return fig


def novel_brain_shell_trace(opacity: float = 0.12) -> go.Mesh3d | None:
    path = mni_brain_mask_path()
    if path is None:
        return None
    mesh = mni_brain_shell_mesh(str(path), step=2)
    if not mesh or len(mesh.get("x", [])) == 0:
        return None
    return go.Mesh3d(
        x=mesh["x"],
        y=mesh["y"],
        z=mesh["z"],
        i=mesh["i"],
        j=mesh["j"],
        k=mesh["k"],
        name="MNI152 brain shell",
        color="#94a3b8",
        opacity=opacity,
        flatshading=True,
        showscale=False,
        showlegend=True,
        hoverinfo="skip",
        lighting={"ambient": 0.72, "diffuse": 0.55, "roughness": 0.9, "specular": 0.08},
        lightposition={"x": -120, "y": -80, "z": 180},
    )


def novel_node_brainspace_figure(nodes: pd.DataFrame, title: str = "AAL3 FDR node signals in brain space") -> go.Figure:
    frame = novel_node_plot_frame(nodes)
    if frame.empty:
        return novel_empty_figure(title, "No mappable AAL3 node evidence is available.")
    fig = go.Figure()
    for system, system_df in frame.groupby("System", dropna=False):
        system_name = str(system)
        fig.add_trace(
            go.Scatter3d(
                x=system_df["x"],
                y=system_df["y"],
                z=system_df["z"],
                mode="markers+text",
                name=system_name,
                marker={
                    "size": system_df["Marker size"],
                    "color": NOVEL_SYSTEM_COLORS.get(system_name, NOVEL_SYSTEM_COLORS["Other"]),
                    "opacity": 0.9,
                    "line": {"color": "#0f172a", "width": 1.0},
                },
                text=system_df["AAL3 region"],
                textfont={"color": CHART_TEXT, "size": 10},
                textposition="top center",
                customdata=system_df[["Families", "Metrics", "Best q", "-log10(q)"]],
                hovertemplate=(
                    "%{text}<br>%{customdata[0]}<br>%{customdata[1]}"
                    "<br>best q=%{customdata[2]:.3g}<br>-log10(q)=%{customdata[3]:.2f}<extra></extra>"
                ),
            )
        )
    fig.update_layout(
        title={"text": f"<b>{escape(title)}</b><br><sup>Node size is -log10(FDR q); position is AAL3 atlas centroid.</sup>", "x": 0.0, "xanchor": "left"},
        paper_bgcolor=CHART_BACKGROUND,
        plot_bgcolor=CHART_BACKGROUND,
        font={"color": CHART_TEXT},
        height=620,
        margin={"l": 0, "r": 0, "t": 82, "b": 0},
        legend={
            "orientation": "h",
            "yanchor": "top",
            "y": 0.98,
            "xanchor": "left",
            "x": 0.01,
            "bgcolor": "rgba(248,250,252,0.9)",
            "bordercolor": CHART_STROKE,
            "borderwidth": 1,
            "font": {"color": CHART_TEXT, "size": 11},
        },
        scene={
            "xaxis": {"title": "MNI x", "backgroundcolor": CHART_BACKGROUND, "gridcolor": CHART_GRID, "zerolinecolor": CHART_STROKE},
            "yaxis": {"title": "MNI y", "backgroundcolor": CHART_BACKGROUND, "gridcolor": CHART_GRID, "zerolinecolor": CHART_STROKE},
            "zaxis": {"title": "MNI z", "backgroundcolor": CHART_BACKGROUND, "gridcolor": CHART_GRID, "zerolinecolor": CHART_STROKE},
            "aspectmode": "data",
        },
    )
    novel_apply_readable_layout(fig)
    return novel_apply_readable_scene(fig)


def novel_node_family_count_figure(nodes: pd.DataFrame) -> go.Figure:
    frame = novel_node_plot_frame(nodes)
    if frame.empty:
        return novel_empty_figure("Node-family counts", "No node-family counts are available.", height=360)
    counts = frame["System"].value_counts().rename_axis("System").reset_index(name="FDR nodes")
    counts["color"] = counts["System"].map(lambda name: NOVEL_SYSTEM_COLORS.get(str(name), NOVEL_SYSTEM_COLORS["Other"]))
    fig = go.Figure(
        go.Bar(
            x=counts["System"],
            y=counts["FDR nodes"],
            text=counts["FDR nodes"],
            textposition="outside",
            textfont={"color": CHART_TEXT, "size": 13},
            cliponaxis=False,
            marker_color=counts["color"],
            hovertemplate="%{x}<br>nodes=%{y}<extra></extra>",
        )
    )
    y_max = float(pd.to_numeric(counts["FDR nodes"], errors="coerce").max())
    fig.update_layout(
        title={"text": "<b>FDR nodes by anatomical family</b>", "x": 0.01, "xanchor": "left"},
        paper_bgcolor=CHART_BACKGROUND,
        plot_bgcolor=CHART_BACKGROUND,
        font={"color": CHART_TEXT},
        height=360,
        margin={"l": 72, "r": 30, "t": 72, "b": 118},
        xaxis={"title": "", "gridcolor": CHART_GRID, "tickangle": -25},
        yaxis={"title": "Unique AAL3 nodes", "gridcolor": CHART_GRID, "range": [0, max(1.0, y_max * 1.22)]},
        showlegend=False,
    )
    return novel_apply_readable_layout(fig)


def novel_top_node_bar_figure(nodes: pd.DataFrame, top_n: int = 14) -> go.Figure:
    if nodes.empty:
        return novel_empty_figure("Top AAL3 node findings", "No node rows are available.", height=430)
    work = nodes.copy()
    work["Evidence q"] = pd.to_numeric(work.get("Evidence q"), errors="coerce")
    work = work[work["Evidence q"].notna()].sort_values("Evidence q").head(top_n).copy()
    if work.empty:
        return novel_empty_figure("Top AAL3 node findings", "No node rows are available.", height=430)
    work["-log10(q)"] = -np.log10(work["Evidence q"].clip(lower=1e-300))
    work["Label"] = work["Metric"].astype(str) + " | " + work["AAL3 region"].astype(str)
    work = work.iloc[::-1].copy()
    x_max = float(pd.to_numeric(work["-log10(q)"], errors="coerce").max())
    fig = go.Figure(
        go.Bar(
            x=work["-log10(q)"],
            y=work["Label"],
            orientation="h",
            text=work["-log10(q)"].map(lambda value: f"{value:.1f}"),
            textposition="outside",
            textfont={"color": CHART_TEXT, "size": 12},
            cliponaxis=False,
            marker_color=work["System"].map(lambda name: NOVEL_SYSTEM_COLORS.get(str(name), NOVEL_SYSTEM_COLORS["Other"])),
            customdata=work[["System", "Evidence q"]],
            hovertemplate="%{y}<br>%{customdata[0]}<br>q=%{customdata[1]:.3g}<extra></extra>",
        )
    )
    fig.update_layout(
        title={"text": "<b>Top FDR node signals</b>", "x": 0.01, "xanchor": "left"},
        paper_bgcolor=CHART_BACKGROUND,
        plot_bgcolor=CHART_BACKGROUND,
        font={"color": CHART_TEXT},
        height=510,
        margin={"l": 310, "r": 58, "t": 72, "b": 68},
        xaxis={"title": "-log10(FDR q)", "gridcolor": CHART_GRID, "range": [0, max(1.0, x_max * 1.14)]},
        yaxis={"title": "", "gridcolor": CHART_GRID},
        showlegend=False,
    )
    return novel_apply_readable_layout(fig)


def novel_pairwise_direction_figure(rows: pd.DataFrame, title: str = "AD-oriented group effects") -> go.Figure:
    if rows.empty or "cliffs_delta" not in rows.columns:
        return novel_empty_figure(title, "No FDR-supported pairwise rows are available.", height=450)
    work = rows.copy()
    work["cliffs_delta"] = pd.to_numeric(work["cliffs_delta"], errors="coerce")
    work = work[work["cliffs_delta"].notna()].copy()
    if work.empty:
        return novel_empty_figure(title, "No finite effect sizes are available.", height=450)

    def ad_effect(row: pd.Series) -> float:
        comparison = str(row.get("Comparison", ""))
        delta = float(row.get("cliffs_delta"))
        if comparison.endswith("vs AD"):
            return -delta
        if comparison.startswith("AD vs"):
            return delta
        return delta

    work["AD-oriented Cliff delta"] = work.apply(ad_effect, axis=1)
    work["Label"] = work["Finding"].astype(str) + " | " + work["Comparison"].astype(str)
    work = work.sort_values("AD-oriented Cliff delta", key=lambda s: s.abs(), ascending=False).head(18).iloc[::-1].copy()
    colors = np.where(work["AD-oriented Cliff delta"] >= 0, "#dc2626", "#2563eb")
    x_vals = pd.to_numeric(work["AD-oriented Cliff delta"], errors="coerce")
    x_abs_max = float(x_vals.abs().max()) if not x_vals.empty else 1.0
    fig = go.Figure(
        go.Bar(
            x=work["AD-oriented Cliff delta"],
            y=work["Label"],
            orientation="h",
            text=work["AD-oriented Cliff delta"].map(lambda value: f"{value:+.2f}"),
            textposition="outside",
            textfont={"color": NOVEL_READABLE_TEXT, "size": 12},
            cliponaxis=False,
            marker_color=colors,
            customdata=work[["Direction", "Evidence q"]],
            hovertemplate="%{y}<br>%{customdata[0]}<br>AD-oriented Cliff delta=%{x:.3f}<br>q=%{customdata[1]:.3g}<extra></extra>",
        )
    )
    fig.add_vline(x=0, line={"color": "#64748b", "width": 1})
    fig.update_layout(
        title={"text": f"<b>{escape(title)}</b>", "x": 0.01, "xanchor": "left"},
        paper_bgcolor=CHART_BACKGROUND,
        plot_bgcolor=CHART_BACKGROUND,
        font={"color": CHART_TEXT},
        height=520,
        margin={"l": 330, "r": 88, "t": 72, "b": 78},
        xaxis={
            "title": "AD-oriented Cliff delta (red=AD higher, blue=AD lower)",
            "gridcolor": CHART_GRID,
            "range": [-max(0.1, x_abs_max * 1.18), max(0.1, x_abs_max * 1.18)],
        },
        yaxis={"title": "", "gridcolor": CHART_GRID},
        showlegend=False,
    )
    novel_apply_readable_layout(fig)
    fig.update_xaxes(tickfont={"color": NOVEL_READABLE_TEXT, "size": 13}, title_font={"color": NOVEL_READABLE_TEXT, "size": 15})
    fig.update_yaxes(tickfont={"color": NOVEL_READABLE_TEXT, "size": 13}, title_font={"color": NOVEL_READABLE_TEXT, "size": 15})
    fig.update_layout(font={"color": NOVEL_READABLE_TEXT, "size": 13}, title_font={"color": NOVEL_READABLE_TEXT, "size": 18})
    return fig


def novel_gradient_edge_brainspace_figure(gradient_rows: pd.DataFrame) -> go.Figure:
    if gradient_rows.empty or not {"i", "j"}.issubset(gradient_rows.columns):
        return novel_empty_figure("Disease-gradient edges in AAL3 space", "No disease-gradient edges are available.", height=560)
    coords = aal_centroids(str(AAL_ATLAS_NII), str(AAL_LABEL_CSV))
    if coords.empty or not {"node", "x", "y", "z"}.issubset(coords.columns):
        return novel_empty_figure("Disease-gradient edges in AAL3 space", "AAL3 centroids are unavailable.", height=560)
    coord_map = {
        int(row["node"]): (float(row["x"]), float(row["y"]), float(row["z"]))
        for _, row in coords.dropna(subset=["node", "x", "y", "z"]).iterrows()
    }
    fig = go.Figure()
    shell = novel_brain_shell_trace(opacity=0.11)
    if shell is not None:
        fig.add_trace(shell)
    node_ids: set[int] = set()
    legend_seen = {"decline": False, "rise": False}
    for _, row in gradient_rows.iterrows():
        i = pd.to_numeric(pd.Series([row.get("i")]), errors="coerce").iloc[0]
        j = pd.to_numeric(pd.Series([row.get("j")]), errors="coerce").iloc[0]
        rho = pd.to_numeric(pd.Series([row.get("disease_gradient_rho")]), errors="coerce").iloc[0]
        q_val = pd.to_numeric(pd.Series([row.get("disease_gradient_q")]), errors="coerce").iloc[0]
        if pd.isna(i) or pd.isna(j) or int(i) not in coord_map or int(j) not in coord_map:
            continue
        node_ids.update([int(i), int(j)])
        a = coord_map[int(i)]
        b = coord_map[int(j)]
        width = 2.0
        if pd.notna(q_val) and q_val > 0:
            width = min(8.0, max(2.0, 1.4 + -math.log10(float(q_val))))
        color = "#2563eb" if pd.notna(rho) and rho < 0 else "#dc2626"
        legend_key = "decline" if pd.notna(rho) and rho < 0 else "rise"
        legend_name = "Declines CN to MCI to AD" if legend_key == "decline" else "Rises CN to MCI to AD"
        fig.add_trace(
            go.Scatter3d(
                x=[a[0], b[0]],
                y=[a[1], b[1]],
                z=[a[2], b[2]],
                mode="lines",
                line={"color": color, "width": width},
                hovertemplate=(
                    f"{escape(str(row.get('ROI A', 'ROI A')))} - {escape(str(row.get('ROI B', 'ROI B')))}"
                    f"<br>rho={format_number(rho, digits=3)}<br>q={format_p(q_val)}<extra></extra>"
                ),
                name=legend_name,
                showlegend=not legend_seen[legend_key],
            )
        )
        legend_seen[legend_key] = True
    if not node_ids:
        return novel_empty_figure("Disease-gradient edges in AAL3 space", "No mappable disease-gradient edges are available.", height=560)
    node_rows = coords[pd.to_numeric(coords["node"], errors="coerce").isin(node_ids)].copy()
    fig.add_trace(
        go.Scatter3d(
            x=node_rows["x"],
            y=node_rows["y"],
            z=node_rows["z"],
            mode="markers+text",
            marker={"size": 6, "color": "#facc15", "line": {"color": "#0f172a", "width": 0.8}},
            text=node_rows["atlas_label"],
            textfont={"color": NOVEL_READABLE_TEXT, "size": 10},
            textposition="top center",
            hovertemplate="%{text}<extra></extra>",
            name="AAL3 endpoint nodes",
            showlegend=True,
        )
    )
    fig.update_layout(
        title={"text": "<b>Disease-gradient edges over MNI template brain</b><br><sup>Yellow nodes are AAL3 centroids; gray shell is MNI152 brain mask context.</sup>", "x": 0.0, "xanchor": "left"},
        paper_bgcolor=CHART_BACKGROUND,
        plot_bgcolor=CHART_BACKGROUND,
        font={"color": CHART_TEXT},
        height=650,
        margin={"l": 0, "r": 0, "t": 92, "b": 0},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 0.01,
            "xanchor": "left",
            "x": 0.01,
            "bgcolor": "rgba(248,250,252,0.86)",
            "bordercolor": CHART_STROKE,
            "borderwidth": 1,
            "font": {"color": NOVEL_READABLE_TEXT, "size": 11},
        },
        scene={
            "xaxis": {"title": "MNI x", "backgroundcolor": CHART_BACKGROUND, "gridcolor": CHART_GRID, "zerolinecolor": CHART_STROKE},
            "yaxis": {"title": "MNI y", "backgroundcolor": CHART_BACKGROUND, "gridcolor": CHART_GRID, "zerolinecolor": CHART_STROKE},
            "zaxis": {"title": "MNI z", "backgroundcolor": CHART_BACKGROUND, "gridcolor": CHART_GRID, "zerolinecolor": CHART_STROKE},
            "aspectmode": "data",
        },
        scene_camera={"eye": {"x": -2.35, "y": -0.25, "z": 0.42}, "up": {"x": 0, "y": 0, "z": 1}},
        showlegend=True,
    )
    novel_apply_readable_layout(fig)
    return novel_apply_readable_scene(fig)


def novel_lr_sr_exception_brainspace_figure(csv_files: list[Path]) -> go.Figure:
    edge_rows = novel_lr_sr_exception_edge_frame(csv_files)
    if edge_rows.empty:
        return novel_empty_figure("LR/SR exception edges in MNI space", "No LR/SR exception edge rows are available.", height=560)
    coords = aal_centroids(str(AAL_ATLAS_NII), str(AAL_LABEL_CSV))
    if coords.empty or not {"node", "x", "y", "z"}.issubset(coords.columns):
        return novel_empty_figure("LR/SR exception edges in MNI space", "AAL3 centroids are unavailable.", height=560)
    coord_map = {
        int(row["node"]): (float(row["x"]), float(row["y"]), float(row["z"]))
        for _, row in coords.dropna(subset=["node", "x", "y", "z"]).iterrows()
    }
    class_colors = {"SR": "#06b6d4", "LR": "#a855f7"}
    fig = go.Figure()
    shell = novel_brain_shell_trace(opacity=0.10)
    if shell is not None:
        fig.add_trace(shell)
    max_delta = pd.to_numeric(edge_rows["Abs exception-rate delta"], errors="coerce").max()
    max_delta = float(max_delta) if pd.notna(max_delta) and max_delta > 0 else 1.0
    node_ids: set[int] = set()
    seen_classes: set[str] = set()
    for _, row in edge_rows.iterrows():
        i = int(row.get("i"))
        j = int(row.get("j"))
        if i not in coord_map or j not in coord_map:
            continue
        node_ids.update([i, j])
        a = coord_map[i]
        b = coord_map[j]
        edr_class = str(row.get("edr_class", ""))
        delta = pd.to_numeric(pd.Series([row.get("Abs exception-rate delta")]), errors="coerce").iloc[0]
        delta = float(delta) if pd.notna(delta) else 0.0
        width = 2.0 + 7.0 * min(1.0, delta / max_delta)
        color = class_colors.get(edr_class, "#64748b")
        label = str(row.get("Class label", edr_class))
        ad_delta = pd.to_numeric(pd.Series([row.get("AD minus comparator")]), errors="coerce").iloc[0]
        direction_text = "AD higher exception rate" if pd.notna(ad_delta) and ad_delta > 0 else "AD lower exception rate"
        hover_payload = [
            row.get("comparison"),
            direction_text,
            row.get("AD exception rate"),
            row.get("Comparator exception rate"),
            row.get("Evidence q"),
            row.get("length_median_mm"),
        ]
        fig.add_trace(
            go.Scatter3d(
                x=[a[0], b[0]],
                y=[a[1], b[1]],
                z=[a[2], b[2]],
                mode="lines",
                line={"color": color, "width": width},
                name=label,
                showlegend=edr_class not in seen_classes,
                customdata=[hover_payload, hover_payload],
                hovertemplate=(
                    f"{escape(str(row.get('roi_a', 'ROI A')))} - {escape(str(row.get('roi_b', 'ROI B')))}"
                    "<br>%{customdata[0]} / %{customdata[1]}"
                    "<br>AD exception rate=%{customdata[2]:.3f}"
                    "<br>Comparator exception rate=%{customdata[3]:.3f}"
                    "<br>q=%{customdata[4]:.3g}"
                    "<br>median length=%{customdata[5]:.1f} mm<extra></extra>"
                ),
            )
        )
        seen_classes.add(edr_class)
    if not node_ids:
        return novel_empty_figure("LR/SR exception edges in MNI space", "No mappable LR/SR exception edges are available.", height=560)
    node_rows = coords[pd.to_numeric(coords["node"], errors="coerce").isin(node_ids)].copy()
    fig.add_trace(
        go.Scatter3d(
            x=node_rows["x"],
            y=node_rows["y"],
            z=node_rows["z"],
            mode="markers+text",
            marker={"size": 6, "color": "#facc15", "line": {"color": "#0f172a", "width": 0.8}},
            text=node_rows["atlas_label"],
            textfont={"color": NOVEL_READABLE_TEXT, "size": 10},
            textposition="top center",
            hovertemplate="%{text}<extra></extra>",
            name="AAL3 endpoint nodes",
            showlegend=True,
        )
    )
    fig.update_layout(
        title={
            "text": "<b>Short- and long-range exception edges over MNI template brain</b><br><sup>Edge class uses existing EDR SR/LR labels; thickness is AD-vs-comparator exception-rate separation.</sup>",
            "x": 0.0,
            "xanchor": "left",
        },
        paper_bgcolor=CHART_BACKGROUND,
        plot_bgcolor=CHART_BACKGROUND,
        font={"color": NOVEL_READABLE_TEXT},
        height=650,
        margin={"l": 0, "r": 0, "t": 96, "b": 0},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 0.01,
            "xanchor": "left",
            "x": 0.01,
            "bgcolor": "rgba(248,250,252,0.86)",
            "bordercolor": CHART_STROKE,
            "borderwidth": 1,
            "font": {"color": NOVEL_READABLE_TEXT, "size": 11},
        },
        scene={
            "xaxis": {"title": "MNI x", "backgroundcolor": CHART_BACKGROUND, "gridcolor": CHART_GRID, "zerolinecolor": CHART_STROKE},
            "yaxis": {"title": "MNI y", "backgroundcolor": CHART_BACKGROUND, "gridcolor": CHART_GRID, "zerolinecolor": CHART_STROKE},
            "zaxis": {"title": "MNI z", "backgroundcolor": CHART_BACKGROUND, "gridcolor": CHART_GRID, "zerolinecolor": CHART_STROKE},
            "aspectmode": "data",
        },
        scene_camera={"eye": {"x": -2.15, "y": -0.45, "z": 0.46}, "up": {"x": 0, "y": 0, "z": 1}},
        showlegend=True,
    )
    novel_apply_readable_layout(fig)
    return novel_apply_readable_scene(fig)


def novel_mni_scene_ranges() -> dict[str, list[float]]:
    path = mni_brain_mask_path()
    mesh = mni_brain_shell_mesh(str(path), step=2) if path is not None else {}
    if mesh and len(mesh.get("x", [])) > 0:
        ranges = {}
        for axis in ("x", "y", "z"):
            values = np.asarray(mesh[axis], dtype=float)
            finite = values[np.isfinite(values)]
            if finite.size == 0:
                continue
            pad = max(4.0, float(finite.max() - finite.min()) * 0.04)
            ranges[axis] = [float(finite.min() - pad), float(finite.max() + pad)]
        if {"x", "y", "z"}.issubset(ranges):
            return ranges
    return {"x": [-95.0, 95.0], "y": [-130.0, 100.0], "z": [-80.0, 100.0]}


def novel_lr_sr_panel_title(group: str, rows: pd.DataFrame) -> str:
    if rows.empty:
        return f"{group}<br><sup>0 edges</sup>"
    median_signal = pd.to_numeric(rows["Signal value"], errors="coerce").median()
    return f"{group}<br><sup>{len(rows):,} edges; median={format_number(median_signal, digits=3)}</sup>"


def novel_lr_sr_edge_width_bin(values: pd.Series, scale_max: float) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").fillna(0.0)
    if not np.isfinite(scale_max) or scale_max <= 0:
        return pd.Series(0, index=values.index)
    normalized = (numeric / scale_max).clip(lower=0, upper=1)
    return pd.cut(normalized, bins=[-0.001, 0.25, 0.5, 0.75, 1.0], labels=[0, 1, 2, 3]).astype(int)


def novel_lr_sr_group_node_stats(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    endpoints = []
    for node_col, roi_col in (("i", "roi_a"), ("j", "roi_b")):
        part = rows[[node_col, roi_col, "Signal value"]].copy()
        part.columns = ["node", "atlas_label", "Signal value"]
        endpoints.append(part)
    out = pd.concat(endpoints, ignore_index=True, sort=False)
    out["node"] = pd.to_numeric(out["node"], errors="coerce")
    out["Signal value"] = pd.to_numeric(out["Signal value"], errors="coerce")
    out = out[out["node"].notna() & out["Signal value"].notna()].copy()
    if out.empty:
        return pd.DataFrame()
    stats_df = (
        out.groupby(["node", "atlas_label"], dropna=False)
        .agg(
            **{
                "Incident edges": ("Signal value", "size"),
                "Node signal": ("Signal value", "sum"),
            }
        )
        .reset_index()
    )
    stats_df["node"] = stats_df["node"].astype(int)
    return stats_df


def novel_lr_sr_three_group_brainspace_figure(
    csv_files: list[Path],
    signal: str,
    range_class: str,
    max_edges_per_group: int,
) -> go.Figure:
    frame = novel_lr_sr_three_group_signal_frame(csv_files, signal)
    if frame.empty:
        return novel_empty_figure("CN/MCI/AD LR/SR brain-space comparison", "No LR/SR edge rows are available.", height=620)
    visible = novel_lr_sr_filter_signal_frame(frame, range_class, max_edges_per_group)
    if visible.empty:
        return novel_empty_figure("CN/MCI/AD LR/SR brain-space comparison", "No LR/SR edges match the selected controls.", height=620)
    coords = aal_centroids(str(AAL_ATLAS_NII), str(AAL_LABEL_CSV))
    if coords.empty or not {"node", "x", "y", "z"}.issubset(coords.columns):
        return novel_empty_figure("CN/MCI/AD LR/SR brain-space comparison", "AAL3 centroids are unavailable.", height=620)
    coord_map = {
        int(row["node"]): (float(row["x"]), float(row["y"]), float(row["z"]))
        for _, row in coords.dropna(subset=["node", "x", "y", "z"]).iterrows()
    }
    visible = visible[visible["i"].isin(coord_map) & visible["j"].isin(coord_map)].copy()
    if visible.empty:
        return novel_empty_figure("CN/MCI/AD LR/SR brain-space comparison", "No selected LR/SR edges are mappable to AAL3 centroids.", height=620)

    visible["Signal value"] = pd.to_numeric(visible["Signal value"], errors="coerce")
    scale_max = float(visible["Signal value"].max()) if visible["Signal value"].notna().any() else 1.0
    scale_max = scale_max if np.isfinite(scale_max) and scale_max > 0 else 1.0
    visible["Width bin"] = novel_lr_sr_edge_width_bin(visible["Signal value"], scale_max)
    width_map = {0: 1.5, 1: 2.8, 2: 4.8, 3: 7.2}
    opacity_map = {0: 0.35, 1: 0.48, 2: 0.64, 3: 0.82}
    visible_by_group = {group: visible[visible["group"].astype(str).eq(group)].copy() for group in GROUP_ORDER}
    subplot_titles = [novel_lr_sr_panel_title(group, visible_by_group[group]) for group in GROUP_ORDER]
    fig = make_subplots(
        rows=1,
        cols=3,
        specs=[[{"type": "scene"}, {"type": "scene"}, {"type": "scene"}]],
        subplot_titles=subplot_titles,
        horizontal_spacing=0.015,
    )
    scene_names = {1: "scene", 2: "scene2", 3: "scene3"}
    show_shell_legend = True
    shown_class_legend: set[str] = set()
    node_signal_max = 1.0
    node_stats_by_group: dict[str, pd.DataFrame] = {}
    for group, rows in visible_by_group.items():
        stats_df = novel_lr_sr_group_node_stats(rows)
        node_stats_by_group[group] = stats_df
        if not stats_df.empty:
            node_signal_max = max(node_signal_max, float(pd.to_numeric(stats_df["Node signal"], errors="coerce").max()))

    for col, group in enumerate(GROUP_ORDER, start=1):
        scene_name = scene_names[col]
        shell = novel_brain_shell_trace(opacity=0.08)
        if shell is not None:
            shell.update(scene=scene_name, showlegend=show_shell_legend)
            fig.add_trace(shell, row=1, col=col)
            show_shell_legend = False

        group_rows = visible_by_group[group]
        for edr_class in ("SR", "LR"):
            class_rows = group_rows[group_rows["edr_class"].eq(edr_class)].copy()
            if class_rows.empty:
                continue
            for width_bin, bin_rows in class_rows.groupby("Width bin", dropna=False):
                x_vals: list[float | None] = []
                y_vals: list[float | None] = []
                z_vals: list[float | None] = []
                customdata: list[list[object] | None] = []
                for _, edge in bin_rows.iterrows():
                    a = coord_map[int(edge["i"])]
                    b = coord_map[int(edge["j"])]
                    payload = [
                        group,
                        edr_class,
                        edge.get("roi_a", f"AAL_{int(edge['i']):03d}"),
                        edge.get("roi_b", f"AAL_{int(edge['j']):03d}"),
                        edge.get("Signal value"),
                        edge.get("Median length mm"),
                        edge.get("n_subjects"),
                        edge.get("Nonzero fraction"),
                        edge.get("Evidence q"),
                        edge.get("Source"),
                    ]
                    x_vals.extend([a[0], b[0], None])
                    y_vals.extend([a[1], b[1], None])
                    z_vals.extend([a[2], b[2], None])
                    customdata.extend([payload, payload, None])
                label = "Short-range edges" if edr_class == "SR" else "Long-range edges"
                fig.add_trace(
                    go.Scatter3d(
                        x=x_vals,
                        y=y_vals,
                        z=z_vals,
                        mode="lines",
                        line={"color": NOVEL_LR_SR_CLASS_COLORS[edr_class], "width": width_map.get(int(width_bin), 2.0)},
                        opacity=opacity_map.get(int(width_bin), 0.5),
                        name=label,
                        legendgroup=edr_class,
                        showlegend=edr_class not in shown_class_legend,
                        scene=scene_name,
                        customdata=customdata,
                        hovertemplate=(
                            "%{customdata[0]} / %{customdata[1]}"
                            "<br>%{customdata[2]} - %{customdata[3]}"
                            "<br>signal=%{customdata[4]:.4g}"
                            "<br>median length=%{customdata[5]:.1f} mm"
                            "<br>n=%{customdata[6]:.0f}"
                            "<br>nonzero fraction=%{customdata[7]:.3f}"
                            "<br>q=%{customdata[8]:.3g}"
                            "<br>%{customdata[9]}<extra></extra>"
                        ),
                    ),
                    row=1,
                    col=col,
                )
                shown_class_legend.add(edr_class)

        node_stats = node_stats_by_group[group]
        if not node_stats.empty:
            node_stats = node_stats[node_stats["node"].isin(coord_map)].copy()
            node_stats["x"] = node_stats["node"].map(lambda node: coord_map[int(node)][0])
            node_stats["y"] = node_stats["node"].map(lambda node: coord_map[int(node)][1])
            node_stats["z"] = node_stats["node"].map(lambda node: coord_map[int(node)][2])
            max_count = max(1, int(pd.to_numeric(node_stats["Incident edges"], errors="coerce").max()))
            node_stats["Marker size"] = 4 + 14 * pd.to_numeric(node_stats["Incident edges"], errors="coerce").fillna(0) / max_count
            fig.add_trace(
                go.Scatter3d(
                    x=node_stats["x"],
                    y=node_stats["y"],
                    z=node_stats["z"],
                    mode="markers",
                    marker={
                        "size": node_stats["Marker size"],
                        "color": node_stats["Node signal"],
                        "colorscale": "YlOrRd",
                        "cmin": 0,
                        "cmax": node_signal_max,
                        "opacity": 0.94,
                        "line": {"color": "#0f172a", "width": 0.7},
                        "showscale": col == 3,
                        "colorbar": {"title": "Node signal", "len": 0.42, "thickness": 12, "x": 1.02},
                    },
                    name="AAL3 endpoint nodes",
                    legendgroup="nodes",
                    showlegend=col == 1,
                    scene=scene_name,
                    customdata=node_stats[["atlas_label", "Incident edges", "Node signal"]],
                    hovertemplate="%{customdata[0]}<br>incident edges=%{customdata[1]:.0f}<br>node signal=%{customdata[2]:.4g}<extra></extra>",
                ),
                row=1,
                col=col,
            )

    ranges = novel_mni_scene_ranges()
    camera = {"eye": {"x": -2.15, "y": -0.45, "z": 0.46}, "up": {"x": 0, "y": 0, "z": 1}}
    axis_template = {
        "visible": False,
        "range": None,
        "backgroundcolor": CHART_BACKGROUND,
        "gridcolor": CHART_GRID,
        "zerolinecolor": CHART_STROKE,
    }
    scene_layout = {
        "xaxis": {**axis_template, "range": ranges["x"]},
        "yaxis": {**axis_template, "range": ranges["y"]},
        "zaxis": {**axis_template, "range": ranges["z"]},
        "aspectmode": "data",
        "camera": camera,
    }
    fig.update_layout(
        title={
            "text": f"<b>CN / MCI / AD {escape(signal)} in SR/LR brain space</b><br><sup>Top-N controls visibility only; edge color is pathway range, edge thickness/opacity is selected signal, node size is visible incident-edge count.</sup>",
            "x": 0.01,
            "xanchor": "left",
        },
        paper_bgcolor=CHART_BACKGROUND,
        plot_bgcolor=CHART_BACKGROUND,
        font={"color": NOVEL_READABLE_TEXT, "size": 12},
        title_font={"color": NOVEL_READABLE_TEXT, "size": 18},
        height=720,
        margin={"l": 0, "r": 18, "t": 112, "b": 8},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 0.01,
            "xanchor": "left",
            "x": 0.01,
            "bgcolor": "rgba(248,250,252,0.86)",
            "bordercolor": CHART_STROKE,
            "borderwidth": 1,
            "font": {"color": NOVEL_READABLE_TEXT, "size": 11},
        },
        scene=scene_layout,
        scene2=scene_layout,
        scene3=scene_layout,
        showlegend=True,
    )
    for annotation in fig.layout.annotations:
        annotation.font = {"color": NOVEL_READABLE_TEXT, "size": 14}
    return fig


def novel_distance_metric_visuals(csv_files: list[Path]) -> bool:
    specs = (
        ("lr_sr_subject_level_fd_sum_len_mean.csv", "short_global_eff", "Short-range efficiency"),
        ("lr_sr_subject_level_fd_sum_len_mean.csv", "long_global_eff", "Long-range efficiency"),
        ("delay_subject_level_fd_sum_len_mean.csv", "mean_delay_ms", "Mean delay proxy"),
        ("edr_subject_level_fd_sum_len_mean.csv", "edr_residual_mean", "EDR residual mean"),
        ("edr_exception_subject_level_fd_sum_len_mean.csv", "edr_lambda", "EDR lambda"),
    )
    rendered = False
    cols = st.columns(2, gap="large")
    for idx, (file_name, metric, title) in enumerate(specs):
        df = novel_csv(csv_files, file_name)
        if df.empty or metric not in df.columns or "group" not in df.columns:
            continue
        data = subject_metric_frame(df, metric, value_col=metric)
        if data.empty:
            continue
        desc = metric_descriptives(csv_files, metric, data=data)
        pairwise = metric_pairwise(csv_files, metric)
        chart, _, _, _ = live_distribution_chart(
            data,
            title,
            hide_visual_outliers=False,
            stats_df=desc,
            subtitle_lines=chart_stat_subtitles(csv_files, metric),
        )
        chart.update_layout(height=390, margin={"l": 74, "r": 24, "t": 74, "b": 70})
        novel_apply_readable_layout(chart)
        with cols[idx % 2]:
            st.plotly_chart(chart, width="stretch", config={"displaylogo": False}, theme=None)
        rendered = True
    return rendered


def render_novel_brain_age_visuals(csv_files: list[Path]) -> bool:
    df = novel_csv(csv_files, "live_brain_age_predictions.csv")
    if df.empty or not {"group", "age", "brain_age_pred", "BAG_age_corrected"}.issubset(df.columns):
        return False
    work = df.copy()
    for col in ("age", "brain_age_pred", "BAG", "BAG_age_corrected"):
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")
    work = work[work["group"].astype(str).isin(GROUP_ORDER) & work["age"].notna()].copy()
    if work.empty:
        return False
    left, right = st.columns(2, gap="large")
    with left:
        scatter_fig = brain_age_scatter_view(
            work,
            "brain_age_pred",
            "Structural predicted age vs chronological age",
            "Predicted structural age",
            identity_line=True,
        )
        novel_apply_readable_layout(scatter_fig)
        st.plotly_chart(
            scatter_fig,
            width="stretch",
            config={"displaylogo": False},
            theme=None,
        )
    with right:
        data = subject_metric_frame(work, "BAG_age_corrected", value_col="BAG_age_corrected")
        desc = metric_descriptives(csv_files, "BAG_age_corrected", data=data)
        pairwise = metric_pairwise(csv_files, "BAG_age_corrected")
        chart, _, _, _ = live_distribution_chart(
            data,
            "Age-corrected brain-age gap",
            hide_visual_outliers=False,
            stats_df=desc,
            subtitle_lines=chart_stat_subtitles(csv_files, "BAG_age_corrected"),
        )
        chart.update_layout(height=520, margin={"l": 74, "r": 24, "t": 74, "b": 70})
        novel_apply_readable_layout(chart)
        st.plotly_chart(chart, width="stretch", config={"displaylogo": False}, theme=None)
    return True


def novel_slide_header(title: str, kicker: str, claim: str, pills: tuple[str, ...] = ()) -> None:
    pill_html = "".join(f"<span class='novel-pill'>{escape(pill)}</span>" for pill in pills)
    st.markdown(
        "<div class='novel-slide'>"
        f"<div class='novel-kicker'>{escape(kicker)}</div>"
        f"<h3>{escape(title)}</h3>"
        f"<p>{escape(claim)}</p>"
        f"<div class='novel-pill-row'>{pill_html}</div>"
        "</div>",
        unsafe_allow_html=True,
    )


def novel_table_config() -> dict:
    return {
        "Evidence q": st.column_config.NumberColumn("BH/FDR q", format="%.3g"),
        "KW p": st.column_config.NumberColumn("KW p", format="%.3g"),
        "KW stat": st.column_config.NumberColumn("KW stat", format="%.3f"),
        "median_a": st.column_config.NumberColumn("Median A", format="%.4g"),
        "median_b": st.column_config.NumberColumn("Median B", format="%.4g"),
        "median_diff_a_minus_b": st.column_config.NumberColumn("Median A-B", format="%.4g"),
        "cliffs_delta": st.column_config.NumberColumn("Cliff delta", format="%.3f"),
        "disease_gradient_rho": st.column_config.NumberColumn("Disease rho", format="%.3f"),
        "disease_gradient_p": st.column_config.NumberColumn("p", format="%.3g"),
        "disease_gradient_q": st.column_config.NumberColumn("q", format="%.3g"),
        "r2": st.column_config.NumberColumn("R2", format="%.3f"),
        "mae": st.column_config.NumberColumn("MAE", format="%.3f"),
        "rmse": st.column_config.NumberColumn("RMSE", format="%.3f"),
        "spearman_r": st.column_config.NumberColumn("Spearman r", format="%.3f"),
        "pearson_r": st.column_config.NumberColumn("Pearson r", format="%.3f"),
        "accuracy": st.column_config.NumberColumn("Accuracy", format="%.3f"),
        "balanced_accuracy": st.column_config.NumberColumn("Balanced acc.", format="%.3f"),
        "macro_f1": st.column_config.NumberColumn("Macro F1", format="%.3f"),
        "macro_auroc_ovr": st.column_config.NumberColumn("AUROC OVR", format="%.3f"),
        "abs_spearman_r": st.column_config.NumberColumn("|Spearman r|", format="%.3f"),
        "feature_missingness_in_target_cohort": st.column_config.NumberColumn("Missingness", format="%.3f"),
        "eta_squared_by_cdr_class": st.column_config.NumberColumn("Eta2 by CDR", format="%.3f"),
    }


FINDINGS_DIR = ANALYSIS_ROOT / "20_findings"


@st.cache_data(ttl=600, show_spinner=False)
def _load_novelty_cards() -> list:
    import json
    try:
        return json.loads((FINDINGS_DIR / "novelty_cards.json").read_text())
    except Exception:
        return []


def _citation_li(rec: dict) -> str:
    title = str(rec.get("title", "")).strip()
    authors = str(rec.get("authors", "")).split(";")
    first = authors[0].strip() if authors and authors[0].strip() else ""
    etal = " et al." if len([a for a in authors if a.strip()]) > 1 else ""
    year = rec.get("year", "")
    venue = str(rec.get("venue", "")).strip()
    cites = rec.get("cited_by_count", 0)
    url = str(rec.get("doi") or rec.get("url") or "").strip()
    meta = " · ".join([x for x in [f"{first}{etal}" if first else "", str(year), f"*{venue}*" if venue else "",
                                   f"{cites} cites" if cites else ""] if x])
    label = f"{title} — {meta}" if meta else title
    if url:
        return f"<li><a href='{url}' target='_blank'>{label}</a></li>"
    return f"<li>{label}</li>"


def _citation_list_html(records: list, tier: str) -> str:
    items = [_citation_li(r) for r in records if str(r.get("tier", "")) == tier]
    if not items:
        return ""
    return f"<ul class='novel-source-list'>{''.join(items)}</ul>"


def render_literature_novelty_workbench() -> None:
    """Literature-grounded novelty: positions each significant finding against retrieved 2024-26 literature.
    Renders synthesized novelty cards if present, else a findings+evidence browser. Citations are only ever
    real retrieved records (OpenAlex)."""
    cards = _load_novelty_cards()
    cat = _net_csv(FINDINGS_DIR / "findings_catalog.csv")
    lit = _net_csv(FINDINGS_DIR / "literature_records.csv")
    if not cards and (cat is None or cat.empty):
        return
    st.markdown("## 🔬 Literature-grounded novelty")
    st.caption(
        "Every FDR-significant network finding positioned against the latest literature (OpenAlex, ≥2024). "
        "Two tiers — **landmark** (high-citation, established) and **frontier** (2025–26, venue-ranked, since "
        "new papers aren't cited yet). Citations are real retrieved records, never fabricated. Novelty is a "
        "literature-validated hypothesis, not a proven discovery."
    )
    if not cat.empty:
        n_sig = len(cat)
        n_lit = 0 if lit is None or lit.empty else int(lit["finding_id"].nunique())
        st.caption(f"{n_sig} significant findings · literature retrieved for {n_lit} · "
                   f"{0 if lit is None or lit.empty else len(lit)} records.")
    _report = FINDINGS_DIR / "novelty_report.md"
    if _report.exists():
        try:
            st.download_button("⬇ Download paper-grade novelty report (.md)", _report.read_text(),
                               file_name="novelty_report.md", mime="text/markdown", key="novelty-report-dl")
        except Exception:
            pass

    if cards:
        types = ["all"] + sorted({str(c.get("novelty_type", "")) for c in cards if c.get("novelty_type")})
        pick = st.selectbox("Novelty type", types, key="novelty-type-filter")
        for c in cards:
            if pick != "all" and str(c.get("novelty_type", "")) != pick:
                continue
            with st.container(border=True):
                st.markdown(f"### {c.get('theme', c.get('finding_id', 'Finding'))}")
                badges = " ".join(f"<span class='novel-pill'>{b}</span>" for b in
                                  [c.get("novelty_type", ""), f"confidence: {c.get('confidence', '?')}"] if b)
                st.markdown(f"<div class='novel-pill-row'>{badges}</div>", unsafe_allow_html=True)
                st.markdown(f"**Our finding.** {c.get('our_finding', '')}")
                if c.get("prior_literature"):
                    st.markdown(f"**What's known (landmark).** {c['prior_literature']}")
                if c.get("frontier_context"):
                    st.markdown(f"**Frontier (2025–26).** {c['frontier_context']}")
                st.markdown(f"**Novelty.** {c.get('novelty_claim', '')}")
                if c.get("caveat"):
                    st.caption(f"⚠️ {c['caveat']}")
                recs = c.get("citations", [])
                lm, fr = _citation_list_html(recs, "landmark"), _citation_list_html(recs, "frontier")
                if lm:
                    st.markdown("**Landmark references**", help="≥2024, ranked by citations")
                    st.markdown(lm, unsafe_allow_html=True)
                if fr:
                    st.markdown("**Frontier references (2025–26)**")
                    st.markdown(fr, unsafe_allow_html=True)
        st.divider()
        return

    # fallback: findings + evidence browser (before synthesis is authored)
    st.info("Novelty cards not yet synthesized — browsing findings with their retrieved literature.")
    fam = st.selectbox("Feature family", sorted(cat["feature_family"].dropna().unique()), key="nov-fb-fam")
    sub = cat[cat["feature_family"] == fam].copy().sort_values("q")
    for r in sub.head(25).itertuples():
        with st.expander(f"{r.plain_statement}"):
            if lit is not None and not lit.empty:
                recs = lit[lit["finding_id"] == r.finding_id].to_dict("records")
                lm, fr = _citation_list_html(recs, "landmark"), _citation_list_html(recs, "frontier")
                if lm:
                    st.markdown("**Landmark (≥2024, by citations)**"); st.markdown(lm, unsafe_allow_html=True)
                if fr:
                    st.markdown("**Frontier (2025–26)**"); st.markdown(fr, unsafe_allow_html=True)
                if not lm and not fr:
                    st.caption("No literature records retrieved for this finding.")
            else:
                st.caption("Run the literature retrieval step to populate citations.")
    st.divider()


def render_novel_findings_panel(csv_files: list[Path]) -> bool:
    render_literature_novelty_workbench()
    local_specs = (
        ("fa_mean_node_tests.csv", "FA node mean", "Local DTI"),
        ("md_mean_node_tests.csv", "MD node mean", "Local DTI"),
        ("rd_mean_node_tests.csv", "RD node mean", "Local DTI"),
        ("ad_mean_node_tests.csv", "AD node mean", "Local DTI"),
    )
    graph_specs = (
        ("degree_node_tests.csv", "Degree", "Node graph"),
        ("strength_node_tests.csv", "Strength", "Node graph"),
        ("nodal_eff_node_tests.csv", "Nodal efficiency", "Node graph"),
    )
    global_specs = (
        ("fa_mean_edge_mean_pairwise.csv", "Global edge FA", "white-matter integrity separates AD from CN/MCI"),
        ("md_mean_edge_mean_pairwise.csv", "Global edge MD", "diffusivity rises in AD"),
        ("rd_mean_edge_mean_pairwise.csv", "Global edge RD", "radial diffusivity rises in AD"),
        ("ad_mean_edge_mean_pairwise.csv", "Global edge AD", "axial diffusivity rises in AD"),
        ("density_pairwise.csv", "Graph density", "structural graph sparsity separates AD"),
        ("global_efficiency_pairwise.csv", "Global efficiency", "network integration drops in AD"),
        ("mean_strength_pairwise.csv", "Mean strength", "weighted structural strength drops in AD"),
        ("BAG_age_corrected_pairwise.csv", "Corrected brain-age gap", "structural age gap increases in AD"),
    )
    distance_specs = (
        ("short_w_mean_pairwise.csv", "Short-range weight", "short-range structural burden differs in AD"),
        ("short_global_eff_pairwise.csv", "Short-range efficiency", "short-range efficiency is lower in AD"),
        ("long_global_eff_pairwise.csv", "Long-range efficiency", "long-range efficiency is near-collapsed in AD"),
        ("mean_delay_ms_pairwise.csv", "Mean delay proxy", "surviving-path delay topology shifts in AD"),
        ("long_delay_fraction_pairwise.csv", "Long-delay fraction", "long-delay fraction is lower in AD"),
        ("edr_residual_mean_pairwise.csv", "EDR residual mean", "CN-reference distance residuals rise in AD"),
        ("edr_lambda_pairwise.csv", "EDR lambda", "distance-decay fit changes in AD"),
        ("sr_exception_pct_pairwise.csv", "SR exception rate", "short-range exceptions separate AD"),
    )

    local_nodes = novel_node_evidence(csv_files, local_specs, top_per_metric=4)
    graph_nodes = novel_node_evidence(csv_files, graph_specs, top_per_metric=4)
    global_rows = novel_pairwise_evidence(csv_files, global_specs, top_per_metric=2)
    distance_rows = novel_pairwise_evidence(csv_files, distance_specs, top_per_metric=2)
    gradient_rows = novel_edge_gradient_evidence(csv_files)
    regression_rows = novel_regression_model_evidence(csv_files)
    cdr_rows = novel_cdr_model_evidence(csv_files)
    feature_rows = novel_signal_feature_evidence(csv_files)

    node_fdr_count = novel_count_fdr_rows(csv_files, tuple(spec[0] for spec in local_specs + graph_specs))
    edge_fdr_count = novel_count_fdr_rows(csv_files, ("edges_fd_sum_omnibus.csv", "edges_fd_sum_CNvsAD.csv", "edges_fd_sum_MCIvsAD.csv"))
    best_r2 = pd.to_numeric(regression_rows.get("r2", pd.Series(dtype=float)), errors="coerce").max() if not regression_rows.empty else np.nan
    best_cdr = pd.to_numeric(cdr_rows.get("macro_auroc_ovr", pd.Series(dtype=float)), errors="coerce").max() if not cdr_rows.empty else np.nan

    cards = [
        novel_metric_card_html("FDR-significant node tests", f"{node_fdr_count:,}", "local DTI + graph nodes at q<0.05"),
        novel_metric_card_html("FDR-significant edge tests", f"{edge_fdr_count:,}", "omnibus / AD pairwise fd_sum edges"),
        novel_metric_card_html("Best structural MMSE prediction", format_number(best_r2, digits=3), "status-ok cross-validated R2"),
        novel_metric_card_html("Best structural CDR prediction", format_number(best_cdr, digits=3), "status-ok cross-validated AUROC"),
    ]
    st.markdown(
        "<div class='novel-hero'>"
        "<div class='novel-kicker'>Novel findings synthesis</div>"
        "<h2>AAL3-resolved structural connectomics identifies a subcortical-limbic vulnerability pattern linked to network disconnection and cognitive severity in AD.</h2>"
        "<div class='novel-claim'>"
        "Using FDR-controlled AAL3 node, edge, graph, distance-aware, and cross-validated clinical prediction analyses, "
        "the current cohort shows convergent structural evidence for AD-related disconnection centered on thalamic, "
        "limbic, cingulate, and neuromodulatory systems. Main claims use FDR-supported statistics or status-ok "
        "cross-validated model rows; weaker top-N signals remain supporting only."
        "</div>"
        f"<div class='novel-grid'>{''.join(cards)}</div>"
        "</div>",
        unsafe_allow_html=True,
    )
    render_novel_mechanism_map()

    novel_slide_header(
        "Result 1: AAL3 localizes the signal to subcortical-limbic and cingulate nodes",
        "AAL3 node localization",
        "FDR-controlled node tests identify complementary anatomic components: thalamic nuclei in graph topology and limbic, cingulate, orbitofrontal, temporal, and neuromodulatory regions in local DTI and node graph features.",
        ("AAL3 166-node atlas", "FDR q<0.05", "local DTI + node graph"),
    )
    render_novel_evidence_strip(
        "AAL3 node-level Kruskal-Wallis tests, BH/FDR q<0.05.",
        f"Thalamic graph nodes: {novel_join(NOVEL_THALAMIC_GRAPH_NODES)}; limbic/cingulate nodes include {novel_join(NOVEL_LIMBIC_CINGULATE_NODES)}.",
    )
    node_frames = [df for df in (local_nodes, graph_nodes) if not df.empty]
    all_nodes = pd.concat(node_frames, ignore_index=True, sort=False) if node_frames else pd.DataFrame()
    node_tabs = st.tabs(["All FDR nodes", "Local DTI", "Graph topology"])
    for tab, label, frame in (
        (node_tabs[0], "All AAL3 FDR node signals", all_nodes),
        (node_tabs[1], "Local DTI AAL3 node signals", local_nodes),
        (node_tabs[2], "Graph topology AAL3 node signals", graph_nodes),
    ):
        with tab:
            st.plotly_chart(novel_node_brainspace_figure(frame, label), width="stretch", config={"displaylogo": False}, theme=None)
    node_plot_cols = st.columns((0.58, 0.42), gap="large")
    with node_plot_cols[0]:
        st.plotly_chart(novel_top_node_bar_figure(all_nodes), width="stretch", config={"displaylogo": False}, theme=None)
    with node_plot_cols[1]:
        st.plotly_chart(novel_node_family_count_figure(all_nodes), width="stretch", config={"displaylogo": False}, theme=None)
    with st.expander("Supporting AAL3 node tables", expanded=False):
        table_cols = st.columns(2, gap="large")
        with table_cols[0]:
            st.markdown("**Top local DTI nodes**")
            if local_nodes.empty:
                st.info("No FDR-supported local DTI node rows are available.")
            else:
                display = local_nodes.head(14)
                st.dataframe(display, width="stretch", hide_index=True, height=compact_table_height(len(display)), column_config=novel_table_config())
        with table_cols[1]:
            st.markdown("**Top graph topology nodes**")
            if graph_nodes.empty:
                st.info("No FDR-supported graph node rows are available.")
            else:
                focus = graph_nodes[graph_nodes["System"].isin(("Thalamic", "Cingulate", "Limbic / neuromodulatory"))].copy()
                display = focus.head(14) if not focus.empty else graph_nodes.head(14)
                st.dataframe(display, width="stretch", hide_index=True, height=compact_table_height(len(display)), column_config=novel_table_config())

    novel_slide_header(
        "Result 2: Regional findings converge with global structural disconnection",
        "Network disconnection",
        "The regional AAL3 findings align with a whole-network phenotype in which AD shows tissue-level DTI disruption and graph-level loss of density, efficiency, and weighted strength.",
        ("global DTI", "density", "efficiency", "brain-age gap"),
    )
    render_novel_ribbon(("FA down in AD", "MD / RD / AD up in AD", "Density + efficiency down", "Mean strength down"))
    render_novel_evidence_strip(
        "AD-involving global pairwise tables, BH/FDR q<0.05.",
        "Microstructural edge integrity and graph communication capacity shift together, supporting a structural disconnection phenotype.",
    )
    if global_rows.empty:
        st.info("No FDR-supported global pairwise rows are available.")
    else:
        st.plotly_chart(novel_pairwise_direction_figure(global_rows, "Global structural disconnection: AD-oriented effects"), width="stretch", config={"displaylogo": False}, theme=None)
        with st.expander("Supporting global disconnection table", expanded=False):
            st.dataframe(global_rows, width="stretch", hide_index=True, height=compact_table_height(len(global_rows)), column_config=novel_table_config())

    st.markdown("#### Age-related structural brain-age signal")
    raw_bag_positive = len(novel_q_filtered(novel_csv(csv_files, "BAG_pairwise.csv"))) > 0
    raw_bag_text = "raw BAG also has FDR-supported rows" if raw_bag_positive else "raw BAG is not FDR-positive"
    render_novel_evidence_strip(
        "Brain-age pairwise outputs: BAG_age_corrected_pairwise.csv and BAG_pairwise.csv.",
        f"Age-corrected BAG separates AD from CN/MCI with FDR support; {raw_bag_text}, so age correction is the positive finding.",
    )
    if not render_novel_brain_age_visuals(csv_files):
        st.info("Brain-age visual rows are not available in the current CSV outputs.")

    novel_slide_header(
        "Result 3: Distance-aware analysis reveals pathway-length vulnerability",
        "Distance / EDR signature",
        "Distance-aware metrics test whether disconnection depends on anatomical pathway length: short/long-range efficiency, length-derived delay, EDR residuals, EDR lambda, and disease-gradient edges.",
        ("LR/SR", "delay proxy", "EDR residuals", "disease gradient"),
    )
    render_novel_ribbon(("LR/SR = physical pathway range", "Delay = length-derived proxy", "EDR = distance-decay of weight", "Gradient edges = CN to MCI to AD trend"))
    render_novel_evidence_strip(
        "Distance, delay, EDR residual, and gradient-edge outputs with FDR q<0.05.",
        "The AD phenotype is distance-sensitive: pathway-length summaries and CN-reference distance-decay residuals carry positive signal.",
    )
    if not novel_distance_metric_visuals(csv_files):
        st.info("No subject-level distance/EDR visual rows are available.")
    st.markdown("#### LR/SR group differences and exception localization")
    control_cols = st.columns((0.36, 0.34, 0.30), gap="large")
    with control_cols[0]:
        lr_sr_signal = st.selectbox(
            "Signal",
            list(NOVEL_LR_SR_SIGNAL_OPTIONS),
            index=0,
            key="novel_lr_sr_signal",
        )
    with control_cols[1]:
        lr_sr_range = st.segmented_control(
            "Range class",
            options=list(NOVEL_LR_SR_RANGE_OPTIONS),
            default="SR + LR",
            key="novel_lr_sr_range",
        )
        if not lr_sr_range:
            lr_sr_range = "SR + LR"
    with control_cols[2]:
        lr_sr_max_edges = st.slider(
            "Max visible edges per group",
            min_value=25,
            max_value=500,
            value=150,
            step=25,
            key="novel_lr_sr_max_edges",
        )
    render_novel_lr_sr_exception_explainer(csv_files, lr_sr_signal, lr_sr_range, lr_sr_max_edges)
    st.plotly_chart(novel_lr_sr_group_profile_figure(csv_files), width="stretch", config={"displaylogo": False}, theme=None)
    st.plotly_chart(
        novel_lr_sr_three_group_brainspace_figure(csv_files, lr_sr_signal, lr_sr_range, lr_sr_max_edges),
        width="stretch",
        config={"displaylogo": False},
        theme=None,
    )
    st.markdown("#### Disease-gradient edge localization")
    render_novel_gradient_edge_explainer(gradient_rows)
    st.plotly_chart(novel_gradient_edge_brainspace_figure(gradient_rows), width="stretch", config={"displaylogo": False}, theme=None)
    with st.expander("Supporting distance, EDR, and disease-gradient tables", expanded=False):
        dist_cols = st.columns((0.58, 0.42), gap="large")
        with dist_cols[0]:
            st.markdown("**Subject-level distance and EDR evidence**")
            if distance_rows.empty:
                st.info("No FDR-supported distance/EDR pairwise rows are available.")
            else:
                st.dataframe(distance_rows, width="stretch", hide_index=True, height=compact_table_height(len(distance_rows)), column_config=novel_table_config())
        with dist_cols[1]:
            st.markdown("**Top disease-gradient edges**")
            if gradient_rows.empty:
                st.info("No FDR-supported disease-gradient edges are available.")
            else:
                display_gradient = gradient_rows.drop(columns=[col for col in ("i", "j") if col in gradient_rows.columns])
                st.dataframe(display_gradient, width="stretch", hide_index=True, height=compact_table_height(len(display_gradient)), column_config=novel_table_config())
        st.markdown("**Selected LR/SR brain-map source rows**")
        lr_sr_source_rows = novel_lr_sr_three_group_signal_frame(csv_files, lr_sr_signal)
        if lr_sr_source_rows.empty:
            st.info("No LR/SR source rows are available for the selected signal.")
        else:
            display_cols = [
                "group",
                "edr_class",
                "roi_a",
                "roi_b",
                "Signal value",
                "Median length mm",
                "n_subjects",
                "Nonzero fraction",
                "Evidence q",
                "Source",
            ]
            display_lrsr = lr_sr_source_rows[[col for col in display_cols if col in lr_sr_source_rows.columns]].sort_values(
                ["group", "Signal value"], ascending=[True, False], na_position="last"
            ).head(200)
            st.dataframe(display_lrsr, width="stretch", hide_index=True, height=compact_table_height(min(len(display_lrsr), 18)), column_config=novel_table_config())

    novel_slide_header(
        "Result 4: Structural signatures predict cognitive severity",
        "Clinical prediction",
        "Status-ok cross-validated structural models show measurable MMSE prediction and supportive Global CDR classification, linking the structural connectome signature to clinical severity.",
        ("MMSE", "Global CDR", "cross-validation", "structural-only"),
    )
    render_novel_evidence_strip(
        "Status-ok cross-validated clinical model rows.",
        f"Best displayed MMSE R2={format_number(best_r2, digits=3)}; best displayed Global CDR AUROC={format_number(best_cdr, digits=3)}.",
    )
    with st.expander("Supporting clinical prediction tables", expanded=False):
        model_cols = st.columns((0.58, 0.42), gap="large")
        with model_cols[0]:
            st.markdown("**MMSE regression models**")
            if regression_rows.empty:
                st.info("No status-ok MMSE regression model rows are available.")
            else:
                st.dataframe(regression_rows.head(12), width="stretch", hide_index=True, height=compact_table_height(min(len(regression_rows), 12)), column_config=novel_table_config())
        with model_cols[1]:
            st.markdown("**Global CDR classifiers**")
            if cdr_rows.empty:
                st.info("No status-ok CDR classification model rows are available.")
            else:
                st.dataframe(cdr_rows.head(8), width="stretch", hide_index=True, height=compact_table_height(min(len(cdr_rows), 8)), column_config=novel_table_config())
    if not feature_rows.empty:
        with st.expander("Supporting model-linked structural features", expanded=False):
            st.dataframe(feature_rows, width="stretch", hide_index=True, height=compact_table_height(len(feature_rows)), column_config=novel_table_config())

    novel_slide_header(
        "Integrated interpretation",
        "Core discovery",
        "Existing literature supports AD as a network-disconnection syndrome. The current positive findings add a fine-grained AAL3, distance-aware, clinical-score-linked structural-connectome synthesis from this cohort.",
        ("exploratory", "positive findings only", "figure-led synthesis"),
    )
    st.markdown(
        "<div class='novel-slide'>"
        "<p><b>Core discovery:</b> These findings support an exploratory model in which AD-related structural disconnection is expressed first as fine-grained subcortical-limbic and cingulate vulnerability, then as global network inefficiency and distance-sensitive edge disruption, with measurable association to cognitive severity.</p>"
        "</div>",
        unsafe_allow_html=True,
    )
    source_items = "".join(
        f"<li><a href='{escape(url)}' target='_blank'>{escape(text)}</a></li>"
        for text, url in NOVEL_LITERATURE_ANCHORS
    )
    st.markdown(
        "<div class='novel-slide novel-source-list'>"
        "<div class='novel-kicker'>Literature anchors</div>"
        f"<ul>{source_items}</ul>"
        "</div>",
        unsafe_allow_html=True,
    )
    return True


def connectome_matrix_path(subject_id: str, weight: str) -> Path | None:
    if not CONNECTOMES_DIR.exists():
        return None
    matches = sorted(CONNECTOMES_DIR.glob(f"SC_AAL166_{subject_id}_*_{weight}.csv"))
    return matches[0] if matches else None


def latest_aal3_rescue_root(rescue_parent: Path = AAL3_RESCUE_BATCH_ROOT) -> Path | None:
    latest_file = rescue_parent / "latest_tag.txt"
    if latest_file.exists():
        tag = latest_file.read_text(encoding="utf-8", errors="ignore").strip()
        if tag and (rescue_parent / tag).exists():
            return rescue_parent / tag
    if not rescue_parent.exists():
        return None
    candidates = [
        path
        for path in rescue_parent.iterdir()
        if path.is_dir() and not any(path.name.endswith(suffix) for suffix in AAL3_RESCUE_ROUTE_SUFFIXES[1:])
    ]
    if not candidates:
        return None
    return sorted(candidates, key=lambda path: path.stat().st_mtime, reverse=True)[0]


def aal3_rescue_base_roots(rescue_parent: Path) -> list[Path]:
    if not rescue_parent.exists():
        return []
    if any((rescue_parent / name).exists() for name in ("subjects.csv", "run_manifest.json", "status.json")):
        return [rescue_parent]
    route_suffixes = AAL3_RESCUE_ROUTE_SUFFIXES[1:]
    roots = [
        path
        for path in rescue_parent.iterdir()
        if path.is_dir()
        and not any(path.name.endswith(suffix) for suffix in route_suffixes)
        and any((path / name).exists() for name in ("subjects.csv", "run_manifest.json", "status.json", "route_subject_ledger.csv"))
    ]
    return sorted(roots, key=lambda path: path.name)


@st.cache_data(ttl=60)
def aal3_qc_promoted_subject_ids(rescue_parent_str: str) -> set[str]:
    rescue_parent = Path(rescue_parent_str)
    promoted: set[str] = set()

    def add_promoted_ids(values: pd.Series) -> None:
        for raw_sid in values.dropna().astype(str):
            sid = raw_sid.strip()
            if not sid:
                continue
            promoted.add(sid)
            base_sid = re.sub(r"_I\d+$", "", sid)
            if base_sid:
                promoted.add(base_sid)

    for run_root in aal3_rescue_base_roots(rescue_parent):
        for suffix in AAL3_RESCUE_ROUTE_SUFFIXES:
            route_root = run_root.parent / f"{run_root.name}{suffix}"
            manifest = route_root / "production_promotion_manifest.csv"
            if not manifest.exists() or manifest.stat().st_size <= 0:
                continue
            rows = _safe_read_csv(manifest)
            if rows is None or rows.empty or not {"sid", "status"}.issubset(rows.columns):
                continue
            add_promoted_ids(rows.loc[rows["status"].astype(str).eq("promoted"), "sid"])
    return promoted


def claude_repaired_subject_ids(manifest_path: str = "/data/derivatives/qc/sc_matrix_qc/claude_repaired_manifest.csv") -> set[str]:
    """Subjects Claude repaired via the SOTA route and promoted to production.
    Drives the ' -- cl' viewer tag. Mirrors the older ' -- qc' rescue tagging."""
    repaired: set[str] = set()
    path = Path(manifest_path)
    if not path.exists() or path.stat().st_size <= 0:
        return repaired
    rows = _safe_read_csv(path)
    if rows is None or rows.empty or "sid" not in rows.columns:
        return repaired
    for raw_sid in rows["sid"].dropna().astype(str):
        sid = raw_sid.strip()
        if not sid:
            continue
        repaired.add(sid)
        base_sid = re.sub(r"_I\d+$", "", sid)
        if base_sid:
            repaired.add(base_sid)
    return repaired


def subject_viewer_label(row: object, qc_promoted_subjects: set[str],
                         cl_repaired_subjects: set[str] | None = None) -> str:
    sid = str(getattr(row, "subject_id"))
    cl_repaired_subjects = cl_repaired_subjects or set()
    # Claude-repaired ('cl') takes precedence over the older 'qc' rescue tag.
    if sid in cl_repaired_subjects:
        suffix = " -- cl"
    elif sid in qc_promoted_subjects:
        suffix = " -- qc"
    else:
        suffix = ""
    if hasattr(row, "group"):
        return f"{sid} ({getattr(row, 'group')}){suffix}"
    return f"{sid}{suffix}"


@st.cache_data(ttl=300, show_spinner=False)
def connectome_matrix(matrix_path: str) -> pd.DataFrame:
    return pd.read_csv(matrix_path, header=None)


def graph_subject_options(subject_path: Path) -> pd.DataFrame:
    df = _safe_read_csv(subject_path)
    if df is None or df.empty or "subject_id" not in df.columns:
        return pd.DataFrame()
    cols = [c for c in ("subject_id", "group", "age", "sex") if c in df.columns]
    out = df[cols].copy()
    out["subject_id"] = out["subject_id"].astype(str)
    if "group" in out.columns:
        out["group"] = out["group"].astype(str)
    return out.sort_values(["group", "subject_id"] if "group" in out.columns else ["subject_id"]).reset_index(drop=True)


def graph_weight_info_html(weight: str, subject_id: str, node_count: int, edge_count: int) -> str:
    info = CONNECTOME_WEIGHT_INFO.get(weight, CONNECTOME_WEIGHT_INFO["fd_sum"])
    metric_formula_html = "".join(
        f"<div class='formula-line'>{escape(line)}</div>"
        for line in GLOBAL_GRAPH_FORMULAS
    )
    relation = (
        "The dropdown selects the matrix used in this viewer. The group plots below are whole-network graph metrics "
        "computed from the current fd_sum structural-strength matrix, so they are derived summaries, not separate raw measurements."
    )
    if weight != "fd_sum":
        relation += " Changing this viewer to another weight does not change those lower fd_sum-based cohort statistics."
    return (
        "<div class='result-card graph-guide-card'>"
        "<h4>How to read this graph</h4>"
        "<div class='metric-explain'>Each node is an AAL3 ROI positioned at its atlas centroid in MNI space. "
        "Node color and size reflect total node strength in the selected matrix. Lines show only the strongest selected nonzero edges, "
        "so the display stays readable; the full matrix is available in the heatmap below.</div>"
        f"<div class='chart-finding'>This is an individual-subject graph view for {escape(subject_id)} with "
        f"{node_count:,} matched AAL nodes and {edge_count:,} displayed top edges. Cohort inference remains in the distribution and edge-wise tests.</div>"
        f"<h5>Selected weight: {escape(weight)} - {escape(info['name'])}</h5>"
        f"<div class='metric-explain'>{escape(info['definition'])}</div>"
        f"<div class='formula-line'>{escape(info['formula'])}</div>"
        f"<div class='metric-explain'>{escape(info['detail'])}</div>"
        "<h5>Relation to strength, density, efficiency, and path length</h5>"
        f"<div class='metric-explain'>{escape(relation)}</div>"
        f"<div class='formula-list'>{metric_formula_html}</div>"
        "</div>"
    )


def pair_label(pair: str) -> str:
    if "vs" not in pair:
        return pair
    a, b = pair.split("vs", 1)
    return f"{a} vs {b}"


def edgewise_pair_path(weight: str, pair: str) -> Path | None:
    folder = ANALYSIS_ROOT / f"10_edgewise_{weight}"
    path = folder / f"edges_{weight}_{pair}.csv"
    if path.exists():
        return path
    return None


def connectome_weight_has_matrices(weight: str) -> bool:
    if not CONNECTOMES_DIR.exists():
        return False
    try:
        next(CONNECTOMES_DIR.glob(f"SC_AAL166_*_{weight}.csv"))
        return True
    except StopIteration:
        return False


@st.cache_data(ttl=300, show_spinner=False)
def edgewise_pair_table(path_str: str, label_csv: str) -> pd.DataFrame:
    path = Path(path_str)
    table = pd.read_csv(path)
    labels = aal_label_lookup(label_csv)
    label_map = {}
    if not labels.empty and "node" in labels.columns and "atlas_label" in labels.columns:
        label_map = {
            int(row["node"]): str(row["atlas_label"])
            for _, row in labels.dropna(subset=["node"]).iterrows()
        }
    out = table.copy()
    out["ROI A"] = pd.to_numeric(out["i"], errors="coerce").astype("Int64").map(lambda n: label_map.get(int(n), f"AAL_{int(n):03d}") if pd.notna(n) else "")
    out["ROI B"] = pd.to_numeric(out["j"], errors="coerce").astype("Int64").map(lambda n: label_map.get(int(n), f"AAL_{int(n):03d}") if pd.notna(n) else "")
    if "welch_t" in out.columns:
        return out
    return pd.DataFrame()


@st.cache_data(ttl=600, show_spinner=False)
def live_edgewise_pair_table(weight: str, pair: str, subject_csv: str, label_csv: str) -> pd.DataFrame:
    subjects = pd.read_csv(subject_csv)
    if "subject_id" not in subjects.columns or "group" not in subjects.columns:
        return pd.DataFrame()
    if "vs" not in pair:
        return pd.DataFrame()
    group_a, group_b = pair.split("vs", 1)
    labels = aal_label_lookup(label_csv)
    label_map = {}
    if not labels.empty and "node" in labels.columns and "atlas_label" in labels.columns:
        label_map = {
            int(row["node"]): str(row["atlas_label"])
            for _, row in labels.dropna(subset=["node"]).iterrows()
        }

    arrays: list[np.ndarray] = []
    groups: list[str] = []
    n_nodes: int | None = None
    for row in subjects[subjects["group"].isin([group_a, group_b])].itertuples(index=False):
        sid = str(getattr(row, "subject_id"))
        matrix_path = connectome_matrix_path(sid, weight)
        if matrix_path is None:
            continue
        try:
            mat = pd.read_csv(matrix_path, header=None).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        except Exception:
            continue
        if mat.ndim != 2 or mat.shape[0] < 2 or mat.shape[1] < 2:
            continue
        n = min(mat.shape)
        mat = mat[:n, :n]
        mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)
        mat = (mat + mat.T) / 2.0
        np.fill_diagonal(mat, 0.0)
        if n_nodes is None:
            n_nodes = n
        if n != n_nodes:
            n_use = min(n, n_nodes)
            mat = mat[:n_use, :n_use]
            arrays = [old[:n_use, :n_use] for old in arrays]
            n_nodes = n_use
        arrays.append(mat)
        groups.append(str(getattr(row, "group")))
    if not arrays or n_nodes is None:
        return pd.DataFrame()
    group_arr = np.array(groups)
    if np.sum(group_arr == group_a) < 3 or np.sum(group_arr == group_b) < 3:
        return pd.DataFrame()
    stack = np.stack(arrays, axis=0)
    upper_i, upper_j = np.triu_indices(n_nodes, k=1)
    values = stack[:, upper_i, upper_j]
    a_vals = values[group_arr == group_a, :]
    b_vals = values[group_arr == group_b, :]
    with np.errstate(invalid="ignore", divide="ignore"):
        test = stats.ttest_ind(a_vals, b_vals, axis=0, equal_var=False, nan_policy="omit")
    p_values = np.asarray(test.pvalue, dtype=float)
    t_values = np.asarray(test.statistic, dtype=float)
    q_values = bh_fdr([float(p) if np.isfinite(p) else pd.NA for p in p_values])
    mean_a = np.nanmean(a_vals, axis=0)
    mean_b = np.nanmean(b_vals, axis=0)
    out = pd.DataFrame(
        {
            "i": upper_i + 1,
            "j": upper_j + 1,
            "welch_t": t_values,
            "welch_p": p_values,
            "welch_q": q_values,
            f"mean_{group_a}": mean_a,
            f"mean_{group_b}": mean_b,
            f"n_{group_a}": int(np.sum(group_arr == group_a)),
            f"n_{group_b}": int(np.sum(group_arr == group_b)),
        }
    )
    out = out[pd.to_numeric(out["welch_p"], errors="coerce").notna()].copy()
    out["ROI A"] = out["i"].map(lambda n: label_map.get(int(n), f"AAL_{int(n):03d}"))
    out["ROI B"] = out["j"].map(lambda n: label_map.get(int(n), f"AAL_{int(n):03d}"))
    return out


def edgewise_table_for_weight_pair(weight: str, pair: str, subject_csv: str) -> tuple[pd.DataFrame, str]:
    path = edgewise_pair_path(weight, pair)
    if path is not None:
        return (
            edgewise_pair_table(str(path), str(AAL_LABEL_CSV)),
            f"precomputed table `{path.relative_to(ANALYSIS_ROOT).as_posix()}`",
        )
    return (
        live_edgewise_pair_table(weight, pair, subject_csv, str(AAL_LABEL_CSV)),
        f"live calculation from `{weight}` matrices in `{CONNECTOMES_DIR}`",
    )


def edgewise_welch_heatmap_figure(table: pd.DataFrame, pair: str) -> go.Figure:
    if table.empty or not {"i", "j", "welch_t"}.issubset(table.columns):
        return go.Figure()
    labels = aal_label_lookup(str(AAL_LABEL_CSV))
    label_map = {}
    if not labels.empty and {"node", "atlas_label"}.issubset(labels.columns):
        label_map = {
            int(row["node"]): str(row["atlas_label"])
            for _, row in labels.dropna(subset=["node"]).iterrows()
        }
    max_node = int(
        max(
            pd.to_numeric(table["i"], errors="coerce").max(),
            pd.to_numeric(table["j"], errors="coerce").max(),
            max(label_map) if label_map else 0,
        )
    )
    if max_node <= 1:
        return go.Figure()
    z = np.full((max_node, max_node), np.nan, dtype=float)
    roi_a = np.empty((max_node, max_node), dtype=object)
    roi_b = np.empty((max_node, max_node), dtype=object)
    for idx in range(max_node):
        label = label_map.get(idx + 1, f"AAL_{idx + 1:03d}")
        roi_a[:, idx] = label
        roi_b[idx, :] = label
    for _, row in table.iterrows():
        i = pd.to_numeric(pd.Series([row.get("i")]), errors="coerce").iloc[0]
        j = pd.to_numeric(pd.Series([row.get("j")]), errors="coerce").iloc[0]
        t_value = pd.to_numeric(pd.Series([row.get("welch_t")]), errors="coerce").iloc[0]
        if pd.isna(i) or pd.isna(j) or pd.isna(t_value):
            continue
        i0 = int(i) - 1
        j0 = int(j) - 1
        if 0 <= i0 < max_node and 0 <= j0 < max_node:
            z[i0, j0] = float(t_value)
            z[j0, i0] = float(t_value)
    z_abs = float(np.nanmax(np.abs(z))) if np.isfinite(z).any() else 1.0
    z_abs = max(z_abs, 1.0)
    custom = np.dstack([roi_b, roi_a])
    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=list(range(1, max_node + 1)),
            y=list(range(1, max_node + 1)),
            zmin=-z_abs,
            zmax=z_abs,
            zmid=0,
            colorscale=[[0.0, "#2563eb"], [0.5, "#f8fafc"], [1.0, "#dc2626"]],
            colorbar={"title": "Welch t"},
            customdata=custom,
            hovertemplate="ROI A=%{customdata[0]}<br>ROI B=%{customdata[1]}<br>Welch t=%{z:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        title={"text": f"<b>fd_sum: {escape(pair_label(pair))} Welch t map</b>", "x": 0.01, "xanchor": "left"},
        paper_bgcolor=CHART_BACKGROUND,
        plot_bgcolor=CHART_BACKGROUND,
        font={"color": CHART_TEXT},
        height=660,
        margin={"l": 70, "r": 28, "t": 72, "b": 65},
        xaxis={"title": "AAL3 node", "showgrid": False, "tickfont": {"color": CHART_TEXT}},
        yaxis={"title": "AAL3 node", "showgrid": False, "autorange": "reversed", "tickfont": {"color": CHART_TEXT}},
    )
    return fig


def edgewise_heatmap_inference(table: pd.DataFrame, pair: str) -> str:
    if table.empty:
        return f"No edge-wise Welch t table is available for {pair_label(pair)}."
    work = table.copy()
    work["welch_q"] = pd.to_numeric(work.get("welch_q"), errors="coerce")
    work["welch_p"] = pd.to_numeric(work.get("welch_p"), errors="coerce")
    work["welch_t"] = pd.to_numeric(work.get("welch_t"), errors="coerce")
    significant = work[work["welch_q"] < 0.05].copy()
    a, b = pair.split("vs", 1) if "vs" in pair else ("first group", "second group")
    if significant.empty:
        nominal = int((work["welch_p"] < 0.05).sum())
        return (
            f"{pair_label(pair)} has no FDR-significant fd_sum edges at q<0.05; "
            f"{nominal:,} nominal p<0.05 edges are shown as unthresholded Welch t values."
        )
    pos = int((significant["welch_t"] > 0).sum())
    neg = int((significant["welch_t"] < 0).sum())
    direction = f"{a} > {b}" if pos >= neg else f"{a} < {b}"
    return f"{pair_label(pair)} has {len(significant):,} FDR-significant fd_sum edges; the dominant direction is {direction}."


def edgewise_top_table(table: pd.DataFrame, pair: str, top_n: int = 12) -> pd.DataFrame:
    if table.empty:
        return pd.DataFrame()
    work = table.copy()
    a, b = pair.split("vs", 1) if "vs" in pair else ("A", "B")
    work["welch_q"] = pd.to_numeric(work.get("welch_q"), errors="coerce")
    work["welch_p"] = pd.to_numeric(work.get("welch_p"), errors="coerce")
    work["welch_t"] = pd.to_numeric(work.get("welch_t"), errors="coerce")
    work["Direction"] = work["welch_t"].map(
        lambda value: f"{a} > {b}" if pd.notna(value) and value > 0 else (f"{a} < {b}" if pd.notna(value) and value < 0 else f"{a} = {b}")
    )
    sort_cols = [col for col in ("welch_q", "welch_p") if col in work.columns]
    if sort_cols:
        work = work.sort_values(sort_cols, na_position="last")
    display_cols = [col for col in ("ROI A", "ROI B", "Direction", "welch_t", "welch_p", "welch_q") if col in work.columns]
    return work[display_cols].head(top_n).copy()


def render_edgewise_welch_heatmap_panel(subject_csv: str) -> None:
    available_pairs = [pair for pair in EDGEWISE_PAIRS if edgewise_pair_path("fd_sum", pair) is not None]
    if not available_pairs:
        return
    st.subheader("Welch t Edge Heatmaps")
    st.caption("ROI-by-ROI maps are reconstructed live from fd_sum edge-wise Welch t CSVs; red means the first group has larger fd_sum, blue means the second group has larger fd_sum.")
    tabs = st.tabs([pair_label(pair) for pair in available_pairs])
    for tab, pair in zip(tabs, available_pairs):
        with tab:
            table, source_text = edgewise_table_for_weight_pair("fd_sum", pair, subject_csv)
            if table.empty:
                st.info(f"No fd_sum edge-wise table is available for {pair_label(pair)}.")
                continue
            left, right = st.columns((1.35, 1), gap="large")
            with left:
                st.plotly_chart(edgewise_welch_heatmap_figure(table, pair), width="stretch", config={"displaylogo": False})
            with right:
                st.markdown(
                    "<div class='edr-definition-box'>"
                    f"<h5>{escape(pair_label(pair))} Welch t map</h5>"
                    "<p>Each heatmap cell is one AAL3 ROI pair. Welch t compares fd_sum structural edge weights between diagnostic groups; "
                    "positive t means the first group has the larger edge weight and negative t means the second group has the larger edge weight.</p>"
                    "</div>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"<div class='edr-inference-box'>{escape(edgewise_heatmap_inference(table, pair))}</div>",
                    unsafe_allow_html=True,
                )
                top = edgewise_top_table(table, pair, top_n=12)
                if top.empty:
                    st.info("No ranked ROI-pair table is available for this comparison.")
                else:
                    st.dataframe(
                        top,
                        width="stretch",
                        height=430,
                        hide_index=True,
                        column_config={
                            "welch_t": st.column_config.NumberColumn("Welch t", format="%.3f"),
                            "welch_p": st.column_config.NumberColumn("p", format="%.3g"),
                            "welch_q": st.column_config.NumberColumn("BH q", format="%.3g"),
                        },
                        key=f"welch-heatmap-top-{pair}",
                    )
                st.caption(f"Source: {source_text}.")


def repeated_edgewise_top_pairs(weight: str, pairs: list[str], subject_csv: str, top_n: int, only_fdr: bool) -> pd.DataFrame:
    rows: list[dict] = []
    for pair in pairs:
        table, _ = edgewise_table_for_weight_pair(weight, pair, subject_csv)
        if table.empty or "vs" not in pair:
            continue
        a, b = pair.split("vs", 1)
        work = table.copy()
        if only_fdr and "welch_q" in work.columns:
            work = work[pd.to_numeric(work["welch_q"], errors="coerce") < 0.05].copy()
        sort_cols = [col for col in ("welch_q", "welch_p") if col in work.columns]
        if sort_cols:
            work = work.sort_values(sort_cols, na_position="last")
        for _, row in work.head(top_n).iterrows():
            try:
                i = int(row["i"])
                j = int(row["j"])
            except Exception:
                continue
            t_value = pd.to_numeric(pd.Series([row.get("welch_t")]), errors="coerce").iloc[0]
            direction = f"{a}>{b}" if pd.notna(t_value) and t_value > 0 else (f"{a}<{b}" if pd.notna(t_value) and t_value < 0 else f"{a}={b}")
            rows.append(
                {
                    "i": min(i, j),
                    "j": max(i, j),
                    "ROI A": row.get("ROI A", f"AAL_{i:03d}"),
                    "ROI B": row.get("ROI B", f"AAL_{j:03d}"),
                    "comparison": pair_label(pair),
                    "direction": direction,
                    "p": pd.to_numeric(pd.Series([row.get("welch_p")]), errors="coerce").iloc[0],
                    "BH q": pd.to_numeric(pd.Series([row.get("welch_q")]), errors="coerce").iloc[0],
                }
            )
    if not rows:
        return pd.DataFrame()
    ranked = pd.DataFrame(rows)
    grouped = (
        ranked.groupby(["i", "j", "ROI A", "ROI B"], dropna=False)
        .agg(
            **{
                "Top-N hits": ("comparison", "count"),
                "BH q<0.05 hits": ("BH q", lambda s: int((pd.to_numeric(s, errors="coerce") < 0.05).sum())),
                "Best p": ("p", "min"),
                "Best BH q": ("BH q", "min"),
                "comparisons": ("comparison", lambda s: "; ".join(dict.fromkeys(map(str, s)))),
                "directions": ("direction", lambda s: "; ".join(dict.fromkeys(map(str, s)))),
            }
        )
        .reset_index()
        .sort_values(["BH q<0.05 hits", "Top-N hits", "Best p", "Best BH q"], ascending=[False, False, True, True])
        .head(top_n)
        .copy()
    )
    grouped.insert(0, "Rank", range(1, len(grouped) + 1))
    return grouped.drop(columns=["i", "j"])


def render_edgewise_group_comparison(weight: str, subject_csv: str) -> None:
    stats_weight = weight
    stats_info = CONNECTOME_WEIGHT_INFO.get(stats_weight, CONNECTOME_WEIGHT_INFO["fd_sum"])
    precomputed_pairs = [pair for pair in EDGEWISE_PAIRS if edgewise_pair_path(stats_weight, pair) is not None]
    live_available = connectome_weight_has_matrices(stats_weight)
    available_pairs = precomputed_pairs if precomputed_pairs else (list(EDGEWISE_PAIRS) if live_available else [])
    if not available_pairs:
        st.info(
            f"Group-level ROI-vs-ROI edge inference is not available for `{stats_weight}` ({stats_info['name']}) yet, "
            "and no matching connectome matrices are available for a live calculation."
        )
        return
    st.subheader(
        f"Between-Group ROI-vs-ROI Edge Tests: {metric_label(stats_weight)} - {stats_info['name']}"
    )
    if not precomputed_pairs:
        st.caption(
            f"No precomputed corrected edge table was found for `{stats_weight}`. "
            "This table is computed live from the selected matrix weight's subject connectome CSVs and cached for the dashboard."
        )

    c1, c2, c3 = st.columns((1.1, 1.0, 1.0))
    with c1:
        pair = st.selectbox("Group comparison", available_pairs, format_func=pair_label, key=f"edgewise-pair-{weight}-{stats_weight}")
    with c2:
        top_n = st.slider("Top ROI pairs", min_value=5, max_value=50, value=15, step=5, key=f"edgewise-top-{weight}-{stats_weight}")
    with c3:
        only_fdr = st.checkbox("Only q<0.05", value=True, key=f"edgewise-fdr-{weight}-{stats_weight}")

    table, source_text = edgewise_table_for_weight_pair(stats_weight, pair, subject_csv)
    if table.empty:
        st.info(f"No readable edge-wise table found for `{stats_weight}` {pair_label(pair)}.")
        return
    a, b = pair.split("vs", 1)
    if only_fdr and "welch_q" in table.columns:
        table = table[pd.to_numeric(table["welch_q"], errors="coerce") < 0.05].copy()
    sort_cols = [col for col in ("welch_q", "welch_p") if col in table.columns]
    if sort_cols:
        table = table.sort_values(sort_cols, na_position="last")
    table = table.head(top_n).copy()
    table["Direction"] = pd.to_numeric(table["welch_t"], errors="coerce").map(
        lambda value: f"{a} > {b}" if pd.notna(value) and value > 0 else (f"{a} < {b}" if pd.notna(value) and value < 0 else f"{a} = {b}")
    )
    display_cols = ["ROI A", "ROI B", "Direction", "welch_t", "welch_p", "welch_q"]
    display_cols = [col for col in display_cols if col in table.columns]
    left, right = st.columns((0.6, 0.4), gap="large")
    with left:
        st.markdown(f"#### {pair_label(pair)} Top {top_n} ROI Pairs")
        if table.empty:
            st.info(f"No `{weight}` ROI pairs pass the current filter for {pair_label(pair)}.")
        else:
            st.dataframe(
                table[display_cols],
                width="stretch",
                height=430,
                hide_index=True,
                column_config={
                    "welch_t": st.column_config.NumberColumn("Welch t", format="%.3f"),
                    "welch_p": st.column_config.NumberColumn("p", format="%.3g"),
                    "welch_q": st.column_config.NumberColumn("BH q", format="%.3g"),
                },
            )
        st.caption(
            f"Group edge table from {source_text}. "
            f"Metric compared: `{stats_weight}` ({stats_info['name']}). Positive Welch t means the first group has the larger edge weight."
        )
    with right:
        st.markdown("#### Repeated Top-N ROI Pairs Across Comparisons")
        repeated = repeated_edgewise_top_pairs(stats_weight, list(EDGEWISE_PAIRS), subject_csv, top_n, only_fdr)
        if repeated.empty:
            st.info("No repeated ROI pairs are available under the current Top-N/q filter.")
        else:
            st.dataframe(
                repeated,
                width="stretch",
                height=430,
                hide_index=True,
                column_config={
                    "Best p": st.column_config.NumberColumn("Best p", format="%.3g"),
                    "Best BH q": st.column_config.NumberColumn("Best BH q", format="%.3g"),
                    "Top-N hits": st.column_config.NumberColumn("Top-N hits", format="%d"),
                    "BH q<0.05 hits": st.column_config.NumberColumn("BH q<0.05 hits", format="%d"),
                },
            )
        st.caption("ROI pairs appearing in the selected Top N across CN vs MCI, CN vs AD, and MCI vs AD.")


def connectome_edges_and_nodes(matrix_df: pd.DataFrame, top_edges: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    import numpy as np

    matrix = matrix_df.apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(dtype=float)
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    if matrix.shape[0] != matrix.shape[1]:
        n = min(matrix.shape)
        matrix = matrix[:n, :n]
    matrix = (matrix + matrix.T) / 2.0
    np.fill_diagonal(matrix, 0.0)

    coords = aal_centroids(str(AAL_ATLAS_NII), str(AAL_LABEL_CSV))
    if coords.empty:
        return pd.DataFrame(), pd.DataFrame()
    n = matrix.shape[0]
    coords = coords[pd.to_numeric(coords["node"], errors="coerce").between(1, n)].copy()
    coords["matrix_index"] = pd.to_numeric(coords["node"], errors="coerce").astype(int) - 1
    valid_indices = set(coords["matrix_index"].astype(int))

    edges = []
    upper_i, upper_j = np.triu_indices(n, k=1)
    values = matrix[upper_i, upper_j]
    order = np.argsort(values)[::-1]
    for idx in order:
        value = float(values[idx])
        if value <= 0:
            break
        i = int(upper_i[idx])
        j = int(upper_j[idx])
        if i not in valid_indices or j not in valid_indices:
            continue
        edges.append({"i": i, "j": j, "weight": value})
        if len(edges) >= top_edges:
            break

    edge_df = pd.DataFrame(edges)
    coords["strength"] = coords["matrix_index"].map(lambda idx: float(matrix[int(idx), :].sum()))
    return edge_df, coords


def connectome_graph_figure(edge_df: pd.DataFrame, node_df: pd.DataFrame, title: str) -> go.Figure:
    import numpy as np

    fig = go.Figure()
    if not edge_df.empty:
        by_index = node_df.set_index("matrix_index")
        xs, ys, zs = [], [], []
        for _, edge in edge_df.iterrows():
            a = by_index.loc[int(edge["i"])]
            b = by_index.loc[int(edge["j"])]
            xs.extend([a["x"], b["x"], None])
            ys.extend([a["y"], b["y"], None])
            zs.extend([a["z"], b["z"], None])
        fig.add_trace(
            go.Scatter3d(
                x=xs,
                y=ys,
                z=zs,
                mode="lines",
                line={"color": "rgba(71, 85, 105, 0.34)", "width": 2},
                hoverinfo="skip",
                name="Top edges",
            )
        )

    strengths = pd.to_numeric(node_df["strength"], errors="coerce").fillna(0)
    if strengths.max() > strengths.min():
        sizes = 4 + 14 * (strengths - strengths.min()) / (strengths.max() - strengths.min())
    else:
        sizes = pd.Series(np.full(len(node_df), 7), index=node_df.index)
    fig.add_trace(
        go.Scatter3d(
            x=node_df["x"],
            y=node_df["y"],
            z=node_df["z"],
            mode="markers",
            marker={
                "size": sizes,
                "color": strengths,
                "colorscale": "Viridis",
                "colorbar": {"title": "Node strength", "len": 0.62},
                "line": {"color": "#0f172a", "width": 0.5},
                "opacity": 0.92,
            },
            text=node_df["atlas_label"],
            customdata=node_df[["node_name", "strength"]],
            hovertemplate="AAL=%{customdata[0]}<br>%{text}<br>Strength=%{customdata[1]:,.3g}<extra></extra>",
            name="AAL3 nodes",
        )
    )
    fig.update_layout(
        title={"text": f"<b>{escape(title)}</b><br><sup>Nodes are AAL3 regions; edges are strongest selected connectome weights.</sup>", "x": 0.0, "xanchor": "left"},
        paper_bgcolor=CHART_BACKGROUND,
        plot_bgcolor=CHART_BACKGROUND,
        font={"color": CHART_TEXT},
        height=650,
        margin={"l": 0, "r": 0, "t": 72, "b": 0},
        scene={
            "xaxis": {"title": "MNI x", "backgroundcolor": CHART_BACKGROUND, "gridcolor": CHART_GRID, "zerolinecolor": CHART_STROKE},
            "yaxis": {"title": "MNI y", "backgroundcolor": CHART_BACKGROUND, "gridcolor": CHART_GRID, "zerolinecolor": CHART_STROKE},
            "zaxis": {"title": "MNI z", "backgroundcolor": CHART_BACKGROUND, "gridcolor": CHART_GRID, "zerolinecolor": CHART_STROKE},
            "aspectmode": "data",
        },
        showlegend=False,
    )
    return fig


def connectome_heatmap_figure(matrix_df: pd.DataFrame, title: str) -> go.Figure:
    matrix = matrix_df.apply(pd.to_numeric, errors="coerce").fillna(0)
    fig = go.Figure(
        data=go.Heatmap(
            z=matrix.to_numpy(),
            colorscale="Viridis",
            colorbar={"title": "Weight"},
            hovertemplate="row=%{y}<br>col=%{x}<br>weight=%{z:.3g}<extra></extra>",
        )
    )
    fig.update_layout(
        title={"text": f"<b>{escape(title)}</b>", "x": 0.0, "xanchor": "left"},
        paper_bgcolor=CHART_BACKGROUND,
        plot_bgcolor=CHART_BACKGROUND,
        font={"color": CHART_TEXT},
        height=420,
        margin={"l": 45, "r": 20, "t": 58, "b": 45},
        xaxis={"title": "AAL matrix column", "showticklabels": False},
        yaxis={"title": "AAL matrix row", "showticklabels": False, "autorange": "reversed"},
    )
    return fig


def connectome_labels_for_matrix(n: int) -> list[str]:
    labels = aal_label_lookup(str(AAL_LABEL_CSV))
    if labels.empty:
        return [f"AAL_{idx + 1:03d}" for idx in range(n)]
    labels = labels.copy()
    labels["node"] = pd.to_numeric(labels["node"], errors="coerce")
    labels = labels.dropna(subset=["node"]).sort_values("node")
    label_values = []
    for _, row in labels.head(n).iterrows():
        atlas = str(row.get("atlas_label") or "").strip()
        node_name = str(row.get("node_name") or f"AAL_{int(row['node']):03d}").strip()
        label_values.append(f"{node_name} | {atlas}" if atlas and atlas != node_name else node_name)
    while len(label_values) < n:
        label_values.append(f"AAL_{len(label_values) + 1:03d}")
    return label_values[:n]


def sc_matrix_figure(matrix_df: pd.DataFrame, title: str, labels: list[str]) -> go.Figure:
    matrix = matrix_df.apply(pd.to_numeric, errors="coerce").fillna(0)
    arr = matrix.to_numpy(dtype=float)
    n = min(arr.shape) if arr.ndim == 2 else 0
    arr = arr[:n, :n]
    labels = labels[:n]
    vmax = float(np.nanpercentile(arr[np.isfinite(arr)], 99)) if np.isfinite(arr).any() else 1.0
    fig = go.Figure(
        data=go.Heatmap(
            z=arr,
            x=labels,
            y=labels,
            colorscale="Viridis",
            zmin=0,
            zmax=vmax if vmax > 0 else None,
            colorbar={"title": "Weight"},
            hovertemplate="ROI A=%{y}<br>ROI B=%{x}<br>weight=%{z:.4g}<extra></extra>",
        )
    )
    fig.update_layout(
        title={"text": f"<b>{escape(title)}</b>", "x": 0.0, "xanchor": "left"},
        paper_bgcolor=CHART_BACKGROUND,
        plot_bgcolor=CHART_BACKGROUND,
        font={"color": CHART_TEXT, "size": 12},
        height=720,
        margin={"l": 80, "r": 20, "t": 62, "b": 70},
        xaxis={"title": "AAL3 ROI", "showticklabels": False, "ticks": ""},
        yaxis={"title": "AAL3 ROI", "showticklabels": False, "ticks": "", "autorange": "reversed"},
    )
    return fig


def sc_matrix_metrics(matrix_df: pd.DataFrame) -> dict[str, float]:
    arr = matrix_df.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    n = min(arr.shape) if arr.ndim == 2 else 0
    if n == 0:
        return {}
    arr = np.nan_to_num(arr[:n, :n], nan=0.0, posinf=0.0, neginf=0.0)
    arr = (arr + arr.T) / 2.0
    np.fill_diagonal(arr, 0.0)
    upper = arr[np.triu_indices(n, 1)]
    finite = upper[np.isfinite(upper)]
    positive = finite[finite > 0]
    possible = n * (n - 1) / 2
    node_strength = arr.sum(axis=1)
    return {
        "AAL nodes": float(n),
        "possible undirected edges": float(possible),
        "nonzero edges": float(len(positive)),
        "density": float(len(positive) / possible) if possible else np.nan,
        "matrix sparsity": float(1.0 - (len(positive) / possible)) if possible else np.nan,
        "total strength": float(np.nansum(positive)),
        "mean edge weight": float(np.nanmean(positive)) if len(positive) else np.nan,
        "median edge weight": float(np.nanmedian(positive)) if len(positive) else np.nan,
        "mean node strength": float(np.nanmean(node_strength)) if len(node_strength) else np.nan,
        "max edge weight": float(np.nanmax(positive)) if len(positive) else np.nan,
    }


def sc_square_array(matrix_df: pd.DataFrame, valid_labels: list[int] | None = None) -> np.ndarray:
    arr = matrix_df.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    n = min(arr.shape) if arr.ndim == 2 else 0
    if n == 0:
        return np.array([], dtype=float)
    arr = np.nan_to_num(arr[:n, :n], nan=0.0, posinf=0.0, neginf=0.0)
    arr = (arr + arr.T) / 2.0
    np.fill_diagonal(arr, 0.0)
    if valid_labels:
        padded = np.zeros((len(valid_labels), len(valid_labels)), dtype=float)
        present = [(out_idx, int(label) - 1) for out_idx, label in enumerate(valid_labels) if 0 <= int(label) - 1 < n]
        for out_i, src_i in present:
            for out_j, src_j in present:
                padded[out_i, out_j] = arr[src_i, src_j]
        arr = padded
    return arr


def sc_upper_values(matrix_df: pd.DataFrame, valid_labels: list[int] | None = None) -> np.ndarray:
    arr = sc_square_array(matrix_df, valid_labels)
    n = min(arr.shape) if arr.ndim == 2 else 0
    if n == 0:
        return np.array([], dtype=float)
    values = arr[np.triu_indices(n, 1)]
    return values[np.isfinite(values)]


def sc_count_density_summary(count_matrix_df: pd.DataFrame, valid_labels: list[int] | None = None) -> dict[str, float]:
    metrics = sc_matrix_metrics(count_matrix_df)
    valid_arr = sc_square_array(count_matrix_df, valid_labels)
    values = sc_upper_values(count_matrix_df, valid_labels)
    positive = values[values > 0]
    valid_n = min(valid_arr.shape) if valid_arr.ndim == 2 and valid_arr.size else 0
    possible = valid_n * (valid_n - 1) / 2 if valid_n else np.nan
    density = float(len(positive) / possible) if possible and np.isfinite(possible) else np.nan
    row_signal = np.abs(valid_arr).sum(axis=0) + np.abs(valid_arr).sum(axis=1) if valid_n else np.array([])
    out = {
        "matrix rows": metrics.get("AAL nodes", np.nan),
        "valid AAL3 ROIs": float(valid_n) if valid_n else np.nan,
        "possible ROI pairs": possible,
        "ROI pairs with >=1 tract": float(len(positive)),
        "tract-count density": density,
        "zero-pair fraction": float(1.0 - density) if np.isfinite(density) else np.nan,
        "zero ROI rows": float((row_signal == 0).sum()) if len(row_signal) else np.nan,
        "total assigned tract count": float(np.nansum(positive)) if len(positive) else 0.0,
        "mean tracts per connected pair": float(np.nanmean(positive)) if len(positive) else np.nan,
        "median tracts per connected pair": float(np.nanmedian(positive)) if len(positive) else np.nan,
        "max tract count in one ROI pair": float(np.nanmax(positive)) if len(positive) else np.nan,
    }
    return out


@st.cache_data(ttl=600, show_spinner=False)
def live_tract_count_density_distribution(subject_csv: str, connectomes_dir: str, label_csv: str) -> pd.DataFrame:
    subjects = pd.read_csv(subject_csv)
    if "subject_id" not in subjects.columns:
        return pd.DataFrame()
    valid_labels = aal3_valid_label_values(label_csv)
    rows: list[dict] = []
    for row in subjects.itertuples(index=False):
        sid = str(getattr(row, "subject_id"))
        path = connectome_matrix_path(sid, "count")
        if path is None:
            continue
        try:
            matrix = pd.read_csv(path, header=None)
        except Exception:
            continue
        metrics = sc_count_density_summary(matrix, valid_labels)
        rows.append(
            {
                "subject_id": sid,
                "group": str(getattr(row, "group", "")),
                "tract_count_density": metrics.get("tract-count density", np.nan),
                "connected_roi_pairs": metrics.get("ROI pairs with >=1 tract", np.nan),
                "possible_roi_pairs": metrics.get("possible ROI pairs", np.nan),
                "zero_roi_rows": metrics.get("zero ROI rows", np.nan),
                "total_assigned_tract_count": metrics.get("total assigned tract count", np.nan),
                "median_tracts_per_connected_pair": metrics.get("median tracts per connected pair", np.nan),
                "matrix_path": str(path),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    for col in (
        "tract_count_density",
        "connected_roi_pairs",
        "possible_roi_pairs",
        "zero_roi_rows",
        "total_assigned_tract_count",
        "median_tracts_per_connected_pair",
    ):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def tract_count_density_distribution_figure(df: pd.DataFrame, selected_subject_id: str | None = None) -> go.Figure:
    fig = go.Figure()
    if df.empty or "tract_count_density" not in df.columns:
        return fig
    for group, color in zip(GROUP_ORDER, GROUP_COLORS):
        group_df = df[df["group"].astype(str) == group].copy()
        values = pd.to_numeric(group_df["tract_count_density"], errors="coerce").dropna()
        if values.empty:
            continue
        fig.add_trace(
            go.Violin(
                y=values,
                name=group,
                box_visible=True,
                meanline_visible=True,
                points="all",
                jitter=0.28,
                marker={"color": color, "size": 5, "opacity": 0.62},
                line={"color": color},
                hovertemplate=f"Group={group}<br>density=%{{y:.4f}}<extra></extra>",
            )
        )
    if selected_subject_id and "subject_id" in df.columns:
        selected = df[df["subject_id"].astype(str) == str(selected_subject_id)]
        if not selected.empty:
            y = pd.to_numeric(selected["tract_count_density"], errors="coerce").dropna()
            if not y.empty:
                group = str(selected.iloc[0].get("group", "Selected"))
                x = group if group in GROUP_ORDER else GROUP_ORDER[0]
                fig.add_trace(
                    go.Scatter(
                        x=[x],
                        y=[float(y.iloc[0])],
                        mode="markers",
                        marker={"color": "#facc15", "size": 15, "line": {"color": "#0f172a", "width": 1.5}},
                        name="selected subject",
                        hovertemplate="Selected subject<br>density=%{y:.4f}<extra></extra>",
                    )
                )
    fig.update_layout(
        title={"text": "<b>Tract-count density distribution</b>", "x": 0.0, "xanchor": "left"},
        paper_bgcolor=CHART_BACKGROUND,
        plot_bgcolor=CHART_BACKGROUND,
        font={"color": CHART_TEXT},
        height=390,
        margin={"l": 58, "r": 20, "t": 64, "b": 55},
        yaxis={"title": "ROI-pair density from count matrix", "gridcolor": CHART_GRID, "zerolinecolor": CHART_STROKE},
        xaxis={"title": "Diagnostic group"},
        showlegend=True,
        legend={"orientation": "h", "y": 1.08, "x": 0.01, "bgcolor": "rgba(248,250,252,0.78)"},
    )
    return fig


def tract_count_edge_distribution_figure(
    count_matrix_df: pd.DataFrame, subject_label: str, valid_labels: list[int] | None = None
) -> go.Figure:
    values = sc_upper_values(count_matrix_df, valid_labels)
    positive = values[values > 0]
    fig = go.Figure()
    if len(positive):
        fig.add_trace(
            go.Histogram(
                x=positive,
                nbinsx=50,
                marker={"color": "#38bdf8", "line": {"color": "#0f172a", "width": 0.35}},
                hovertemplate="tracts=%{x}<br>ROI pairs=%{y}<extra></extra>",
                name="connected ROI pairs",
            )
        )
    fig.update_layout(
        title={"text": f"<b>{escape(subject_label)} tract-count edge distribution</b>", "x": 0.0, "xanchor": "left"},
        paper_bgcolor=CHART_BACKGROUND,
        plot_bgcolor=CHART_BACKGROUND,
        font={"color": CHART_TEXT},
        height=390,
        margin={"l": 58, "r": 20, "t": 64, "b": 55},
        xaxis={"title": "Streamline count assigned to one ROI pair", "gridcolor": CHART_GRID, "type": "log" if len(positive) and np.nanmax(positive) > 100 else "linear"},
        yaxis={"title": "Number of ROI pairs", "gridcolor": CHART_GRID, "zerolinecolor": CHART_STROKE},
        showlegend=False,
    )
    return fig


def sc_metrics_table(metrics: dict[str, float]) -> pd.DataFrame:
    rows = []
    for metric, value in metrics.items():
        rows.append({"Metric": metric, "Value": format_number(value)})
    return pd.DataFrame(rows)


def sc_top_edges(matrix_df: pd.DataFrame, top_n: int, min_percentile: float, labels: list[str]) -> pd.DataFrame:
    arr = matrix_df.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    n = min(arr.shape) if arr.ndim == 2 else 0
    if n == 0:
        return pd.DataFrame()
    arr = np.nan_to_num(arr[:n, :n], nan=0.0, posinf=0.0, neginf=0.0)
    arr = (arr + arr.T) / 2.0
    np.fill_diagonal(arr, 0.0)
    i, j = np.triu_indices(n, 1)
    values = arr[i, j]
    valid = np.isfinite(values) & (values > 0)
    i, j, values = i[valid], j[valid], values[valid]
    if not len(values):
        return pd.DataFrame()
    threshold = np.nanpercentile(values, min_percentile) if min_percentile > 0 else -np.inf
    keep = values >= threshold
    i, j, values = i[keep], j[keep], values[keep]
    order = np.argsort(values)[::-1][:top_n]
    rows = []
    for rank, idx in enumerate(order, start=1):
        rows.append(
            {
                "Rank": rank,
                "ROI A": labels[int(i[idx])] if int(i[idx]) < len(labels) else f"AAL_{int(i[idx]) + 1:03d}",
                "ROI B": labels[int(j[idx])] if int(j[idx]) < len(labels) else f"AAL_{int(j[idx]) + 1:03d}",
                "Weight": float(values[idx]),
            }
        )
    return pd.DataFrame(rows)


def table_row_for_subject(csv_files: list[Path], filename: str, subject_id: str) -> tuple[Path | None, pd.Series | None]:
    path = next((p for p in csv_files if p.name == filename), None)
    if path is None:
        return None, None
    df = _safe_read_csv(path)
    if df is None or df.empty or "subject_id" not in df.columns:
        return path, None
    rows = df[df["subject_id"].astype(str) == str(subject_id)]
    if rows.empty:
        return path, None
    return path, rows.iloc[0]


def subject_metric_summary_table(row: pd.Series | None, columns: tuple[str, ...]) -> pd.DataFrame:
    if row is None:
        return pd.DataFrame()
    rows = []
    for col in columns:
        if col in row.index and pd.notna(row[col]):
            rows.append({"Metric": metric_label(col), "Value": format_number(row[col]) if pd.api.types.is_number(row[col]) else str(row[col])})
    return pd.DataFrame(rows)


def render_subject_summary_table(title: str, row: pd.Series | None, columns: tuple[str, ...], *, height: int = 260) -> bool:
    table = subject_metric_summary_table(row, columns)
    st.markdown(f"#### {title}")
    if table.empty:
        st.info("No matching subject-level values are available for this block.")
        return False
    st.dataframe(table, width="stretch", height=height, hide_index=True)
    return True


@st.cache_data(ttl=3600, show_spinner="Computing cohort density…")
def cohort_density_frame(subject_csv: str, label_csv: str, metric: str) -> pd.DataFrame:
    """Per-subject connectome density (count-based) + group + age + mean edge-magnitude for a chosen
    metric. Density is the nonzero off-diagonal fraction (same edge set across weights), so it's reported
    once; the metric column carries the magnitude (mean positive edge weight) for FA/MD/fd_sum/etc."""
    subs = pd.read_csv(subject_csv)
    if "subject_id" not in subs.columns:
        return pd.DataFrame()
    valid = aal3_valid_label_values(label_csv)
    rows: list[dict] = []
    for r in subs.itertuples(index=False):
        sid = str(getattr(r, "subject_id"))
        cpath = connectome_matrix_path(sid, "count")
        if cpath is None:
            continue
        try:
            cm = pd.read_csv(cpath, header=None)
        except Exception:
            continue
        dens = sc_count_density_summary(cm, valid).get("tract-count density", np.nan)
        mval = np.nan
        mpath = connectome_matrix_path(sid, metric)
        if mpath is not None:
            try:
                mm = pd.read_csv(mpath, header=None).to_numpy(dtype=float)
                iu = np.triu_indices(mm.shape[0], 1)
                v = mm[iu]; v = v[np.isfinite(v) & (v > 0)]
                mval = float(np.mean(v)) if v.size else np.nan
            except Exception:
                pass
        rows.append({
            "subject_id": sid, "group": str(getattr(r, "group", "")),
            "age": pd.to_numeric(getattr(r, "age", np.nan), errors="coerce"),
            "tract_count_density": dens, "metric_edge_mean": mval,
        })
    return pd.DataFrame(rows)


def render_cohort_density_panel(subject_path: Path) -> None:
    """Phase-4 density analysis: connectome density + per-metric edge magnitude, by group / age / cohort."""
    import plotly.graph_objects as _go
    from scipy import stats as _ss
    with st.expander("📈 Cohort density analysis — density & metric magnitude by group / age", expanded=False):
        weights = [w for w in CONNECTOME_WEIGHTS if connectome_weight_has_matrices(w)]
        metric = st.selectbox("Edge-magnitude metric", weights,
                              index=(weights.index("fa_mean") if "fa_mean" in weights else 0),
                              key="cohort-density-metric")
        df = cohort_density_frame(str(subject_path), str(AAL_LABEL_CSV), metric)
        if df.empty:
            st.info("No density data available.")
            return
        st.caption(f"Dense cohort n={df['subject_id'].nunique()} · density = nonzero off-diagonal fraction (count) · "
                   f"magnitude = mean positive edge weight of `{metric}`")
        ca, cb = st.columns(2, gap="large")
        with ca:
            st.markdown("**Connectome density by group**")
            st.plotly_chart(tract_count_density_distribution_figure(df), width="stretch", config={"displaylogo": False})
        with cb:
            st.markdown("**Density vs age**")
            fig = _go.Figure()
            for g, c in zip(GROUP_ORDER, GROUP_COLORS):
                gd = df[df["group"].astype(str) == g]
                fig.add_trace(_go.Scatter(x=gd["age"], y=gd["tract_count_density"], mode="markers",
                                          name=g, marker={"color": c, "size": 6, "opacity": 0.6}))
            fig.update_layout(xaxis_title="age", yaxis_title="density", height=360,
                              legend={"orientation": "h"}, margin={"l": 40, "r": 10, "t": 10, "b": 40})
            st.plotly_chart(fig, width="stretch", config={"displaylogo": False})
        # group summary + Kruskal-Wallis on density and on the chosen metric magnitude
        def _kw(col: str) -> str:
            vals = [df[df["group"].astype(str) == g][col].dropna() for g in GROUP_ORDER]
            vals = [v for v in vals if len(v) > 1]
            try:
                return f"{_ss.kruskal(*vals).pvalue:.3g}" if len(vals) >= 2 else "NA"
            except Exception:
                return "NA"
        summ = (df.groupby("group")[["tract_count_density", "metric_edge_mean", "age"]]
                .agg(["mean", "median", "count"]))
        summ.columns = [f"{a}_{b}" for a, b in summ.columns]
        summ = summ.reindex(GROUP_ORDER).round(4).reset_index()
        st.markdown(f"**Group summary** · density KW p={_kw('tract_count_density')} · "
                    f"`{metric}` magnitude KW p={_kw('metric_edge_mean')}")
        st.dataframe(summ, hide_index=True, width="stretch")


NETWORK_MAPPING_CSV = PROJECT_ROOT / "atlas" / "AAL" / "AAL3_network_mapping.csv"
NETWORK_SCHEME_COL = {"Functional networks": "functional_network", "Anatomical systems": "anatomical_system"}


@st.cache_data(ttl=3600, show_spinner=False)
def network_mapping() -> pd.DataFrame:
    try:
        m = pd.read_csv(NETWORK_MAPPING_CSV)
        m["matrix_idx"] = pd.to_numeric(m["matrix_idx"], errors="coerce").astype("Int64")
        return m
    except Exception:
        return pd.DataFrame()


def node_network_lookup(scheme_col: str) -> dict[int, str]:
    """matrix index (1..166) -> network/system label for a scheme column."""
    m = network_mapping()
    if m.empty or scheme_col not in m.columns:
        return {}
    return {int(r.matrix_idx): str(getattr(r, scheme_col))
            for r in m.itertuples() if pd.notna(r.matrix_idx)}


def _net_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


_RANK_LINE = {1: "#f1c40f", 2: "#3498db", 3: "#2ecc71"}          # yellow / blue / green borders
_RANK_FILL = {1: "rgba(241,196,15,0.12)", 2: "rgba(52,152,219,0.12)", 3: "rgba(46,204,113,0.12)"}


def _add_ranked_pair_highlights(fig, networks_order: list, stats_df: pd.DataFrame, metric: str,
                                allowed_pairs: list, ranks: set) -> None:
    """Highlight the top-3 network × group-pair differences for `metric` (max |Cliff's δ| among BH-FDR q<0.05,
    restricted to `allowed_pairs`): #1 yellow, #2 blue, #3 green. Only the ranks in `ranks` are drawn."""
    if stats_df is None or stats_df.empty or "metric" not in stats_df.columns or not networks_order:
        return
    tagmap = {"CN-MCI": ("CN", "MCI", "cn_mci"), "CN-AD": ("CN", "AD", "cn_ad"), "MCI-AD": ("MCI", "AD", "mci_ad")}
    s = stats_df[stats_df["metric"] == metric]
    cands = []
    for _, r in s.iterrows():
        for lbl in allowed_pairs:
            ga, gb, tag = tagmap[lbl]
            q = r.get(f"{tag}_bm_q"); d = r.get(f"{tag}_cliffs_delta")
            if pd.notna(q) and q < 0.05 and pd.notna(d):
                cands.append((str(r["network"]), ga, gb, float(d), float(q)))
    cands.sort(key=lambda t: abs(t[3]), reverse=True)
    ctr = {"CN": -0.233, "MCI": 0.0, "AD": 0.233}  # sub-box centres for 3 grouped boxes
    for rank, (net, ga, gb, d, q) in enumerate(cands[:3], start=1):
        if rank not in ranks or net not in networks_order:
            continue
        k = networks_order.index(net)
        x0 = min(ctr[ga], ctr[gb]) - 0.10 + k
        x1 = max(ctr[ga], ctr[gb]) + 0.10 + k
        fig.add_shape(type="rect", xref="x", yref="paper", x0=x0, x1=x1, y0=0, y1=1, layer="below",
                      line={"color": _RANK_LINE[rank], "width": 2.5}, fillcolor=_RANK_FILL[rank])
        fig.add_annotation(x=(x0 + x1) / 2, xref="x", y=1.0, yref="paper", yanchor="bottom", showarrow=False,
                           text=f"#{rank} {net} {ga}–{gb} (δ={d:+.2f}, q={q:.2g})",
                           font={"size": 9, "color": _RANK_LINE[rank]})
    if cands:
        fig.update_xaxes(categoryorder="array", categoryarray=networks_order)


@st.cache_data(ttl=300, show_spinner=False)
def cohort_provenance_caption() -> str:
    """Provenance stamp so each tab visibly states it's on the fresh 530-dense cohort (addresses #10)."""
    import datetime
    try:
        gcp = ANALYSIS_ROOT / "02_demographics" / "group_counts.csv"
        gc = pd.read_csv(gcp)
        total = int(gc["n_subjects"].sum())
        per = " / ".join(f"{r['group']} {int(r['n_subjects'])}" for _, r in gc.iterrows())
        ts = datetime.datetime.utcfromtimestamp(gcp.stat().st_mtime).strftime("%Y-%m-%d %H:%M UTC")
        return f"📌 Cohort: dense **n={total}** ({per}) · diffusivity n=529 · numbers refreshed {ts}"
    except Exception:
        return ""


def render_node_network_aggregate(long_df: pd.DataFrame, kind: str, key_prefix: str) -> None:
    """Lightweight network/system aggregate of a node-level long table (kind='micro' or 'graph').
    Collapses the per-node view to per-network using the same mapping as the Network Analysis tab."""
    m = network_mapping()
    if long_df is None or long_df.empty or m.empty or "node" not in long_df.columns:
        return
    with st.expander("🧠 Aggregate by network / system", expanded=False):
        scheme_label = st.radio("Scheme", list(NETWORK_SCHEME_COL.keys()), horizontal=True,
                                key=f"{key_prefix}-net-scheme")
        lut = node_network_lookup(NETWORK_SCHEME_COL[scheme_label])
        df = long_df.copy()
        df["network"] = pd.to_numeric(df["node"], errors="coerce").map(lut)
        df = df.dropna(subset=["network"])
        if df.empty or "subject_id" not in df.columns:
            st.info("No mappable node data.")
            return
        if kind == "micro":
            avail = [x for x in ("fa_mean", "md_mean", "ad_mean", "rd_mean")
                     if x in set(df.get("metric", pd.Series(dtype=str)).astype(str))]
            if not avail:
                st.info("No microstructure metrics.")
                return
            metric = st.selectbox("Metric", avail, format_func=metric_label, key=f"{key_prefix}-net-metric")
            d = df[df["metric"].astype(str) == metric].copy()
            w = d["n_edges"].clip(lower=0) if "n_edges" in d.columns else 1.0
            d["_wv"] = d["value"] * w
            d["_w"] = w
            agg = d.groupby(["subject_id", "group", "network"], as_index=False).agg(_wv=("_wv", "sum"), _w=("_w", "sum"))
            agg["value"] = agg["_wv"] / agg["_w"].replace(0, np.nan)
            ylab = metric_label(metric)
        else:
            avail = [x for x in ("strength", "degree", "nodal_eff") if x in df.columns]
            metric = st.selectbox("Metric", avail, format_func=metric_label, key=f"{key_prefix}-net-metric")
            agg = df.groupby(["subject_id", "group", "network"], as_index=False)[metric].mean().rename(
                columns={metric: "value"})
            ylab = metric_label(metric)
        if agg.empty:
            st.info("No data to aggregate.")
            return
        fig = go.Figure()
        for g, c in zip(GROUP_ORDER, GROUP_COLORS):
            gd = agg[agg["group"].astype(str) == g]
            fig.add_trace(go.Box(x=gd["network"], y=gd["value"], name=g, marker_color=c, boxpoints=False))
        fig.update_layout(boxmode="group", height=420, yaxis_title=ylab, legend={"orientation": "h"},
                          margin={"l": 50, "r": 10, "t": 10, "b": 90})
        st.plotly_chart(fig, width="stretch", config={"displaylogo": False})
        st.caption(f"{scheme_label}: node {ylab} aggregated within each network per subject.")
        scheme_key = "functional" if scheme_label == "Functional networks" else "anatomical"
        family = "microstructure" if kind == "micro" else "graph"
        st.divider()
        render_groupwise_stats_table(scheme_key, family, metric)
        render_systemwise_stats(scheme_key, family, metric, key_prefix)


def render_network_breakdown(kind: str, key_prefix: str) -> None:
    """Load the node-level long table from ANALYSIS_ROOT and show the network-aggregate expander.
    For feature tabs whose csv_files don't include the node long table."""
    fn = {"micro": ("04_node_microstructure_live", "live_node_microstructure_long.csv"),
          "graph": ("07_node_graph_live", "live_node_graph_long.csv")}.get(kind)
    if not fn:
        return
    df = _net_csv(ANALYSIS_ROOT / fn[0] / fn[1])
    render_node_network_aggregate(df, kind, key_prefix)


def render_network_coupling_breakdown(key_prefix: str) -> None:
    """Within-network structure-microstructure coupling (ρ) violins by group, for the coupling tabs."""
    with st.expander("🧠 Coupling aggregated by network / system", expanded=False):
        scheme_label = st.radio("Scheme", list(NETWORK_SCHEME_COL.keys()), horizontal=True,
                                key=f"{key_prefix}-netcoup-scheme")
        scheme_key = "functional" if scheme_label == "Functional networks" else "anatomical"
        d = _net_csv(ANALYSIS_ROOT / "19_network_analysis" / scheme_key / "network_coupling_subject.csv")
        if d.empty or not {"network", "metric", "value", "group"}.issubset(d.columns):
            st.info("Network coupling table not available — run the network analysis step.")
            return
        metric = st.selectbox("Coupling pair", sorted(d["metric"].dropna().unique()), key=f"{key_prefix}-netcoup-metric")
        dd = d[d["metric"] == metric]
        fig = go.Figure()
        for g, c in zip(GROUP_ORDER, GROUP_COLORS):
            gd = dd[dd["group"].astype(str) == g]
            fig.add_trace(go.Violin(x=gd["network"], y=gd["value"], name=g, line_color=c,
                                    box_visible=True, meanline_visible=True, points=False, scalemode="width"))
        fig.update_layout(violinmode="group", height=420, yaxis_title="Spearman ρ", legend={"orientation": "h"},
                          margin={"l": 50, "r": 10, "t": 10, "b": 100})
        st.plotly_chart(fig, width="stretch", config={"displaylogo": False})
        st.caption("Within-network Spearman ρ(node graph metric, node DTI).")
        st.divider()
        render_groupwise_stats_table(scheme_key, "coupling", metric)
        render_systemwise_stats(scheme_key, "coupling", metric, key_prefix)


def _q_stars(q: float) -> str:
    if not (isinstance(q, (int, float)) and np.isfinite(q)):
        return ""
    return "✱✱✱" if q < 1e-3 else "✱✱" if q < 1e-2 else "✱" if q < 0.05 else ""


def _pair_star_map(row: pd.Series) -> dict:
    """{'CN-AD': '✱✱', ...} from the bm_q columns of an affectedness row."""
    out = {}
    for tag, lab in (("cn_mci", "CN-MCI"), ("cn_ad", "CN-AD"), ("mci_ad", "MCI-AD")):
        col = f"{tag}_bm_q"
        if col in row and np.isfinite(row.get(col, np.nan)):
            s = _q_stars(float(row[col]))
            if s:
                out[lab] = s
    return out


def _direction_word(delta: float) -> str:
    if not np.isfinite(delta):
        return "differs"
    return "lower in AD" if delta > 0 else "higher in AD"


def network_interpretation(sub: pd.DataFrame, family: str, scheme_label: str) -> str:
    """Templated-from-stats interpretation (+literature link) for an affectedness feature-family slice.
    Every clause is built from the actual numbers — no free invention (guardrail)."""
    if sub is None or sub.empty or "cn_ad_cliffs_delta" not in sub.columns:
        return ""
    s = sub.dropna(subset=["cn_ad_cliffs_delta"]).copy()
    if s.empty:
        return ""
    s["abs"] = s["cn_ad_cliffs_delta"].abs()
    top = s.sort_values("abs", ascending=False).iloc[0]
    net, metric = top["network"], top["metric"]
    delta, q = float(top["cn_ad_cliffs_delta"]), float(top.get("q_kw", np.nan))
    sig_pairs = _pair_star_map(top)
    sig_txt = ", ".join(f"{k} {v}" for k, v in sig_pairs.items()) or "no pairwise comparison survives FDR"
    n_sig = int((s["q_kw"] < 0.05).sum()) if "q_kw" in s.columns else 0
    # (1) templated facts
    line = (f"**Interpretation.** In **{family}** ({scheme_label.lower()}), **{net}** shows the largest CN→AD "
            f"effect — `{metric}` is **{_direction_word(delta)}** (Cliff's δ={delta:+.2f}"
            f"{', q=' + format(q, '.2g') if np.isfinite(q) else ''}). Significant pairwise: {sig_txt}. "
            f"{n_sig} of {len(s)} {family} measures are FDR-significant.")
    # (2) literature link
    try:
        cards = _load_novelty_cards()
        hit = next((c for c in cards if str(net).split(" ")[0].lower() in c.get("theme", "").lower()
                    or any(str(net).lower() in str(fid).lower() for fid in c.get("finding_ids", []))), None)
        if hit:
            line += f" *Consistent with recent literature* ({hit.get('novelty_type','')}; see **Novel Findings**)."
    except Exception:
        pass
    return line


def render_groupwise_stats_table(scheme_key: str, family: str, metric: str) -> None:
    """Group-wise (CN/MCI/AD, between-subjects) stats per network — READ precomputed network_{family}_stats.csv."""
    s = _net_csv(ANALYSIS_ROOT / "19_network_analysis" / scheme_key / f"network_{family}_stats.csv")
    if s.empty or "metric" not in s.columns:
        return
    s = s[s["metric"] == metric].copy()
    if s.empty:
        return
    s = s.reindex(s["cn_ad_cliffs_delta"].abs().sort_values(ascending=False).index) if "cn_ad_cliffs_delta" in s.columns else s
    s["signif."] = [", ".join(f"{k}{v}" for k, v in _pair_star_map(r).items()) for _, r in s.iterrows()]
    cols = [c for c in ["network", "mean_CN", "mean_MCI", "mean_AD", "kw_p",
                        "cn_mci_cliffs_delta", "cn_ad_cliffs_delta", "mci_ad_cliffs_delta", "signif."] if c in s.columns]
    ren = {"cn_mci_cliffs_delta": "δ CN-MCI", "cn_ad_cliffs_delta": "δ CN-AD",
           "mci_ad_cliffs_delta": "δ MCI-AD", "kw_p": "p (KW)"}
    st.markdown("**Group contrast per network (CN vs MCI vs AD, between-subjects)**")
    st.dataframe(s[cols].rename(columns=ren).round(4), hide_index=True, width="stretch")
    st.caption("Kruskal–Wallis across groups; δ = Cliff's δ (>0 lower in AD); signif. = Brunner–Munzel BH-FDR "
               "stars (✱ q<.05, ✱✱ q<.01, ✱✱✱ q<.001) per pair.")


def render_systemwise_stats(scheme_key: str, family: str, metric: str, key_prefix: str) -> None:
    """System-wise (network-vs-network, WITHIN-subject paired) stats: Friedman omnibus + Wilcoxon matrix
    (r_rb + FDR q) with a group selector, + the ART group×network interaction. READ precomputed CSVs."""
    base = ANALYSIS_ROOT / "19_network_analysis" / scheme_key
    pw = _net_csv(base / "network_systemwise_pairwise.csv")
    om = _net_csv(base / "network_systemwise_omnibus.csv")
    it = _net_csv(base / "network_systemwise_interaction.csv")
    if pw.empty or not {"family", "metric", "group"}.issubset(pw.columns):
        return
    st.markdown("**Between-network comparisons — which brain systems differ (within-subject / paired)**")
    grp = st.radio("Group", ["CN", "MCI", "AD", "Pooled"], horizontal=True, key=f"{key_prefix}-sw-group")
    sub = pw[(pw["family"] == family) & (pw["metric"] == metric) & (pw["group"] == grp)]
    if sub.empty:
        st.info("No system-wise data for this selection.")
        return
    nets = sorted(set(sub["net_a"]) | set(sub["net_b"]))
    R = pd.DataFrame(np.nan, index=nets, columns=nets)
    Q = pd.DataFrame(np.nan, index=nets, columns=nets)
    for r in sub.itertuples():
        R.loc[r.net_a, r.net_b] = r.r_rb
        R.loc[r.net_b, r.net_a] = -r.r_rb
        Q.loc[r.net_a, r.net_b] = r.wilcoxon_q
        Q.loc[r.net_b, r.net_a] = r.wilcoxon_q
    fig = go.Figure(go.Heatmap(z=R.values, x=nets, y=nets, colorscale="RdBu", zmid=0,
        colorbar={"title": "r_rb"}, hovertemplate="%{y} vs %{x}<br>r_rb=%{z:.2f}<extra></extra>"))
    for na in nets:
        for nb in nets:
            s = _q_stars(Q.loc[na, nb]) if pd.notna(Q.loc[na, nb]) else ""
            if s:
                fig.add_annotation(x=nb, y=na, text=s, showarrow=False, font={"size": 9, "color": "black"})
    fig.update_layout(height=max(360, 30 * len(nets) + 130), margin={"l": 10, "r": 10, "t": 10, "b": 90})
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False})
    omr = om[(om["family"] == family) & (om["metric"] == metric) & (om["group"] == grp)] if not om.empty else pd.DataFrame()
    if not omr.empty:
        o = omr.iloc[0]
        excl = o["excluded_networks"] if isinstance(o.get("excluded_networks"), str) and o["excluded_networks"] else "none"
        st.caption(f"Friedman omnibus ({grp}): χ²={o['chi2']:.1f}, df={int(o['df'])}, p={o['p']:.2g}, "
                   f"Kendall's W={o['kendalls_w']:.2f}, complete-n={int(o['n_complete'])}. "
                   f"Excluded networks (>30% missing): {excl}.")
    itr = it[(it["family"] == family) & (it["metric"] == metric)] if not it.empty else pd.DataFrame()
    if not itr.empty:
        i = itr.iloc[0]
        qtxt = f", q={i['art_q']:.2g}" if "art_q" in itr.columns and pd.notna(i.get("art_q")) else ""
        st.caption(f"Group×network interaction (ART, exploratory): F={i['art_F']:.2f}, p={i['art_p']:.2g}{qtxt} "
                   "— whether the network profile itself shifts CN→AD.")
    st.caption("Cell = matched-pairs rank-biserial r_rb (row vs col; blue = row higher, red = col higher). "
               "Stars = Wilcoxon signed-rank BH-FDR q (✱ q<.05, ✱✱ q<.01, ✱✱✱ q<.001). Networks are a "
               "within-subject factor, so paired tests are used.")


@st.cache_data(ttl=600, show_spinner=False)
def graph_metric_concordance(pair: str):
    """Per-region group contrast (Cliff's δ) for strength/degree/nodal_eff between two groups, and the
    Spearman concordance of those region-wise contrast profiles across the 3 metrics. δ>0 = lower in 2nd group."""
    from scipy import stats as _ss
    nl = _net_csv(ANALYSIS_ROOT / "07_node_graph_live" / "live_node_graph_long.csv")
    mets = ["strength", "degree", "nodal_eff"]
    if nl.empty or not {"group", "node", *mets}.issubset(nl.columns):
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    g1, g2 = [s.strip() for s in pair.split("vs")]

    def _cd(a, b):
        a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
        if len(a) < 3 or len(b) < 3:
            return np.nan
        try:
            u = _ss.mannwhitneyu(a, b, alternative="two-sided").statistic
            return 2.0 * u / (len(a) * len(b)) - 1.0
        except Exception:
            return np.nan

    rows = []
    for node, nd in nl.groupby("node"):
        rec = {"node": node, "node_name": str(nd["node_name"].iloc[0]) if "node_name" in nd.columns else str(node)}
        for m in mets:
            rec[f"d_{m}"] = _cd(nd[nd["group"] == g1][m].to_numpy(float), nd[nd["group"] == g2][m].to_numpy(float))
        rows.append(rec)
    df = pd.DataFrame(rows)
    corr = pd.DataFrame(np.eye(3), index=mets, columns=mets)
    pval = pd.DataFrame(np.zeros((3, 3)), index=mets, columns=mets)
    for i, mi in enumerate(mets):
        for j, mj in enumerate(mets):
            if i < j:
                r, p = _ss.spearmanr(df[f"d_{mi}"], df[f"d_{mj}"], nan_policy="omit")
                corr.iloc[i, j] = corr.iloc[j, i] = r
                pval.iloc[i, j] = pval.iloc[j, i] = p
    return df, corr, pval


_NET_PALETTE = ["#5dade2", "#e67e22", "#27ae60", "#e74c3c", "#9b59b6", "#16a085",
                "#f1c40f", "#7f8c8d", "#34495e", "#e84393", "#00bcd4", "#8bc34a"]


@st.fragment
def render_graph_metric_concordance() -> None:
    """3x3 frame: columns = group contrasts (CN-MCI/CN-AD/MCI-AD), rows = metric pairs. Each cell scatters the
    166 regions (δ of two graph metrics) coloured by network, and fits a regression line WITHIN each network
    separately — drawn only if that network's fit is significant (p<0.05). Fragment → the top-N control updates
    the plot without a page refresh."""
    from plotly.subplots import make_subplots
    from scipy import stats as _ss
    if not (ANALYSIS_ROOT / "07_node_graph_live" / "live_node_graph_long.csv").exists():
        return
    st.markdown("#### Graph-metric relationship — region-contrast concordance (3×3 frame)")
    st.caption("Each dot = one of the 166 regions; x/y = that region's group contrast (Cliff's δ) for two graph "
               "metrics. **Columns = group contrasts; rows = metric pairs.** Within **each network separately** "
               "(Limbic, DMN, …) a regression line is fit and drawn **only if significant (p<0.05)**, coloured to its "
               "network. δ>0 → metric lower in the 2nd group.")
    pairs = [("CN", "MCI"), ("CN", "AD"), ("MCI", "AD")]
    mpairs = [("strength", "nodal_eff"), ("strength", "degree"), ("degree", "nodal_eff")]
    lab = {"strength": "Strength", "degree": "Degree", "nodal_eff": "Nodal eff"}
    col_titles = [f"{a} vs {b}" for a, b in pairs]
    row_titles = [f"{lab[a]} vs {lab[b]}" for a, b in mpairs]
    lut = node_network_lookup("functional_network")
    dfs = {}
    for g1, g2 in pairs:
        d, _, _ = graph_metric_concordance(f"{g1} vs {g2}")
        if not d.empty:
            d = d.copy()
            d["network"] = pd.to_numeric(d["node"], errors="coerce").map(lut) if lut else "all"
        dfs[(g1, g2)] = d
    nets = sorted({n for d in dfs.values() if not d.empty and "network" in d for n in d["network"].dropna().unique()})
    cmap = {n: _NET_PALETTE[i % len(_NET_PALETTE)] for i, n in enumerate(nets)}
    topn = st.slider("Top-N network lines per cell (ranked by |correlation| among significant)", 1,
                     max(1, len(nets)), value=3, key="gm-concord-topn")
    fig = make_subplots(rows=3, cols=3, column_titles=col_titles, row_titles=row_titles,
                        horizontal_spacing=0.06, vertical_spacing=0.11)
    seen_leg, sig_grid = set(), {}
    MIN_N = 6
    for ri, (ma, mb) in enumerate(mpairs):
        for ci, (g1, g2) in enumerate(pairs):
            d = dfs[(g1, g2)]
            if d.empty:
                continue
            xc, yc = f"d_{ma}", f"d_{mb}"
            sub = d.dropna(subset=[xc, yc])
            fits = []
            for nw in nets:
                dd = sub[sub["network"] == nw]
                if dd.empty:
                    continue
                show = nw not in seen_leg
                seen_leg.add(nw)
                col = cmap.get(nw, "#888")
                fig.add_trace(go.Scatter(x=dd[xc], y=dd[yc], mode="markers", name=nw, legendgroup=nw,
                    showlegend=show, marker={"color": col, "size": 5, "opacity": 0.7},
                    text=dd["node_name"], hovertemplate="%{text}<br>δx=%{x:.2f} δy=%{y:.2f}<extra>" + nw + "</extra>"),
                    row=ri + 1, col=ci + 1)
                # per-NETWORK regression: keep only significant fits, rank later by |r|
                xv, yv = dd[xc].to_numpy(), dd[yc].to_numpy()
                mk = np.isfinite(xv) & np.isfinite(yv); xv, yv = xv[mk], yv[mk]
                if len(xv) >= MIN_N and np.ptp(xv) > 0:
                    sl, ic, rr, p, _ = _ss.linregress(xv, yv)
                    if np.isfinite(p) and p < 0.05:
                        fits.append((abs(rr), nw, col, float(sl), float(ic), float(xv.min()), float(xv.max()), float(rr), float(p)))
            fits.sort(key=lambda t: t[0], reverse=True)  # strongest |r| first
            shown = fits[:topn]
            for _, nw, col, sl, ic, xmn, xmx, rr, p in shown:
                xs = np.array([xmn, xmx])
                fig.add_trace(go.Scatter(x=xs, y=sl * xs + ic, mode="lines", legendgroup=nw, showlegend=False,
                    line={"color": col, "width": 2.5},
                    hovertemplate=f"{nw}: slope={sl:+.2f}, r={rr:+.2f}, p={p:.2g}<extra></extra>"),
                    row=ri + 1, col=ci + 1)
            sig_grid[(ma, mb, g1, g2)] = len(fits)
            fig.add_annotation(text=f"top {len(shown)} of {len(fits)} sig", xref="x domain",
                               yref="y domain", x=0.03, y=0.99, xanchor="left", yanchor="top", showarrow=False,
                               font={"size": 10, "color": "#fff"}, bgcolor="rgba(0,0,0,0.55)",
                               row=ri + 1, col=ci + 1)
            # per-cell mini-legend: which network each drawn line is (colour-matched), updates with Top-N
            for i, (_, nw, col, *_rest) in enumerate(shown[:6]):
                fig.add_annotation(text=f"▬ {nw}", xref="x domain", yref="y domain", x=0.03, y=0.90 - 0.075 * i,
                                   xanchor="left", yanchor="top", showarrow=False,
                                   font={"size": 9, "color": col}, bgcolor="rgba(0,0,0,0.4)",
                                   row=ri + 1, col=ci + 1)
            fig.update_xaxes(title_text=f"{lab[ma]} δ", title_font={"size": 10}, row=ri + 1, col=ci + 1)
            fig.update_yaxes(title_text=f"{lab[mb]} δ", title_font={"size": 10}, row=ri + 1, col=ci + 1)
    fig.update_layout(height=1000, margin={"l": 50, "r": 20, "t": 40, "b": 30},
                      legend={"orientation": "h", "font": {"size": 9}, "y": -0.05})
    fig.update_xaxes(tickfont={"size": 8}); fig.update_yaxes(tickfont={"size": 8})
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False})
    st.caption("Each dot = a region; **a regression line is fit WITHIN each network separately** (Limbic, DMN, … — "
               "networks with ≥6 regions) and drawn **only if that network's fit is significant (p<0.05)**, coloured to "
               "match its dots. **Slope direction:** a line sloping **up (left→right) = positive = the two metrics rise "
               "and fall together across that network's regions (concordant)**; down = they trade off; no line = not "
               "significant. Hover a line for its slope + p. The count in each cell = how many networks are significant.")
    # NL interpretation from the per-network significant counts
    def _cnt(ma, mb):
        return {f"{g1}-{g2}": sig_grid.get((ma, mb, g1, g2), 0) for g1, g2 in pairs}
    se, sd, dn = _cnt("strength", "nodal_eff"), _cnt("strength", "degree"), _cnt("degree", "nodal_eff")
    def _fmt(c):
        return ", ".join(f"{k} {v}/{len(nets)}" for k, v in c.items())
    st.markdown(
        f"**Interpretation (networks with a significant within-network trend, of {len(nets)}).** "
        f"**Strength ↔ Nodal-eff** — {_fmt(se)}: nearly every network shows the concordance at every stage. "
        f"**Strength ↔ Degree** — {_fmt(sd)}. "
        f"**Degree ↔ Nodal-eff** — {_fmt(dn)}: far fewer networks, and it thins out toward AD — binary connectivity "
        "decouples from efficiency region-by-region as disease advances.")


@st.fragment
def render_graph_metric_group_frame() -> None:
    """Strength / degree / nodal-eff box plots per network × group, each highlighting the top-ranked significant
    differences (#1 yellow / #2 blue / #3 green). Fragment → controls update the plots without a page refresh."""
    st.markdown("#### Graph measures by network — group comparison + significance highlight")
    st.caption("Box plots of **strength / degree / nodal-efficiency** per network and group (CN/MCI/AD). Highlight "
               "boxes mark the network × group-pair with the **largest significant difference** for each metric — "
               "**#1 yellow, #2 blue, #3 green** (rank = |Cliff's δ| among BH-FDR q<0.05). Controls update instantly.")
    c0, c1, c2 = st.columns([1.1, 1.1, 1.3])
    scheme_label = c0.radio("Scheme", list(NETWORK_SCHEME_COL.keys()), key="gmf-scheme")
    ranks_sel = c1.multiselect("Show ranks", [1, 2, 3], default=[1, 2, 3],
                               format_func=lambda r: {1: "#1 (yellow)", 2: "#2 (blue)", 3: "#3 (green)"}[r],
                               key="gmf-ranks")
    pair_sel = c2.multiselect("Rank significance based on", ["CN-MCI", "CN-AD", "MCI-AD"],
                              default=["CN-MCI", "CN-AD", "MCI-AD"], key="gmf-pairs")
    scheme_key = "functional" if scheme_label == "Functional networks" else "anatomical"
    base = ANALYSIS_ROOT / "19_network_analysis" / scheme_key
    subj = _net_csv(base / "network_graph_subject.csv")
    stats_df = _net_csv(base / "network_graph_stats.csv")
    if subj.empty or not {"network", "metric", "value", "group"}.issubset(subj.columns):
        st.info("Network graph table unavailable — run the dashboard refresh.")
        return
    pairs = pair_sel or ["CN-MCI", "CN-AD", "MCI-AD"]
    lab = {"strength": "Strength", "degree": "Degree", "nodal_eff": "Nodal efficiency"}
    networks = sorted(subj["network"].dropna().unique())
    for m in ("strength", "degree", "nodal_eff"):
        d = subj[subj["metric"] == m]
        if d.empty:
            continue
        fig = go.Figure()
        for g, c in zip(GROUP_ORDER, GROUP_COLORS):
            gd = d[d["group"].astype(str) == g]
            fig.add_trace(go.Box(x=gd["network"], y=gd["value"], name=g, marker_color=c, boxpoints=False))
        fig.update_layout(boxmode="group", height=340, yaxis_title=lab.get(m, m),
                          legend={"orientation": "h"}, margin={"l": 55, "r": 10, "t": 34, "b": 70})
        _add_ranked_pair_highlights(fig, networks, stats_df, m, pairs, set(ranks_sel))
        st.plotly_chart(fig, width="stretch", config={"displaylogo": False}, key=f"gmf-chart-{m}")


_COUP_LAB = {"strength__fa_mean": "Strength–FA", "strength__md_mean": "Strength–MD",
             "nodal_eff__fa_mean": "NodalEff–FA", "nodal_eff__md_mean": "NodalEff–MD"}
_COUP_METRICS = ["strength__fa_mean", "strength__md_mean", "nodal_eff__fa_mean", "nodal_eff__md_mean"]


@st.fragment
def render_coupling_metric_group_frame() -> None:
    """Coupling (within-network structure–function ρ) box plots per network × group, with ranked significance
    highlights (#1 yellow / #2 blue / #3 green). Fragment → controls update the plots without a page refresh."""
    st.markdown("#### Coupling by network — group comparison + significance highlight")
    st.caption("Within-network structure–function coupling (Spearman ρ) per network and group (CN/MCI/AD). Highlight "
               "boxes mark the network × group-pair with the **largest significant difference** for each coupling "
               "metric — **#1 yellow, #2 blue, #3 green** (|Cliff's δ| among BH-FDR q<0.05).")
    c0, c1, c2 = st.columns([1.1, 1.1, 1.3])
    scheme_label = c0.radio("Scheme", list(NETWORK_SCHEME_COL.keys()), key="cmf-scheme")
    ranks_sel = c1.multiselect("Show ranks", [1, 2, 3], default=[1, 2, 3],
                               format_func=lambda r: {1: "#1 (yellow)", 2: "#2 (blue)", 3: "#3 (green)"}[r],
                               key="cmf-ranks")
    pair_sel = c2.multiselect("Rank significance based on", ["CN-MCI", "CN-AD", "MCI-AD"],
                              default=["CN-MCI", "CN-AD", "MCI-AD"], key="cmf-pairs")
    scheme_key = "functional" if scheme_label == "Functional networks" else "anatomical"
    base = ANALYSIS_ROOT / "19_network_analysis" / scheme_key
    subj = _net_csv(base / "network_coupling_subject.csv")
    stats_df = _net_csv(base / "network_coupling_stats.csv")
    if subj.empty or not {"network", "metric", "value", "group"}.issubset(subj.columns):
        st.info("Network coupling table unavailable — run the dashboard refresh.")
        return
    pairs = pair_sel or ["CN-MCI", "CN-AD", "MCI-AD"]
    for m in _COUP_METRICS:
        d = subj[(subj["metric"] == m) & subj["value"].notna()]   # drop networks with no data (e.g. DorsalAttention)
        if d.empty:
            continue
        nets_m = sorted(d["network"].unique())
        fig = go.Figure()
        for g, c in zip(GROUP_ORDER, GROUP_COLORS):
            gd = d[d["group"].astype(str) == g]
            fig.add_trace(go.Box(x=gd["network"], y=gd["value"], name=g, marker_color=c, boxpoints=False))
        fig.update_layout(boxmode="group", height=320, yaxis_title=f"{_COUP_LAB.get(m, m)} ρ",
                          legend={"orientation": "h"}, margin={"l": 55, "r": 10, "t": 34, "b": 70})
        _add_ranked_pair_highlights(fig, nets_m, stats_df, m, pairs, set(ranks_sel))
        st.plotly_chart(fig, width="stretch", config={"displaylogo": False}, key=f"cmf-chart-{m}")


@st.fragment
def render_coupling_vs_age() -> None:
    """Coupling (ρ) vs age for ALL networks at once (small-multiples grid), per group, with a per-group
    regression line drawn only if significant. Empty networks are hidden."""
    from plotly.subplots import make_subplots
    from scipy import stats as _ss
    st.markdown("#### Coupling vs age — all networks (group comparison)")
    st.caption("Structure–function coupling (Spearman ρ, y) against age (x) for **every network at once**, coloured by "
               "group. A regression line is drawn **per group only if significant (p<0.05)** — compare whether coupling "
               "changes with age differently in CN / MCI / AD. (Cross-sectional; age available for all 530.)")
    c0, c1 = st.columns([1, 1.6])
    scheme_label = c0.radio("Scheme", list(NETWORK_SCHEME_COL.keys()), key="cva-scheme")
    scheme_key = "functional" if scheme_label == "Functional networks" else "anatomical"
    subj = _net_csv(ANALYSIS_ROOT / "19_network_analysis" / scheme_key / "network_coupling_subject.csv")
    if subj.empty or not {"network", "metric", "value", "group", "subject_id"}.issubset(subj.columns):
        st.info("Network coupling table unavailable.")
        return
    metric = c1.selectbox("Coupling metric", _COUP_METRICS, format_func=lambda m: _COUP_LAB.get(m, m), key="cva-metric")
    master = _net_csv(ANALYSIS_ROOT / "00_master" / "master_cohort.csv")
    age = master[["subject_id", "age"]] if "age" in master.columns else pd.DataFrame(columns=["subject_id", "age"])
    d = subj[subj["metric"] == metric].merge(age, on="subject_id", how="left").dropna(subset=["value", "age"])
    if d.empty:
        st.info("No data for this selection.")
        return
    networks = sorted(d["network"].unique())   # only non-empty networks (DorsalAttention etc. dropped)
    ncol = 3
    nrow = (len(networks) + ncol - 1) // ncol
    fig = make_subplots(rows=nrow, cols=ncol, subplot_titles=networks,
                        horizontal_spacing=0.06, vertical_spacing=max(0.06, 0.16 / max(1, nrow)))
    seen, summ = set(), []
    for idx, net in enumerate(networks):
        rr, cc = idx // ncol + 1, idx % ncol + 1
        dn = d[d["network"] == net]
        anns = []
        for g, col in zip(GROUP_ORDER, GROUP_COLORS):
            gd = dn[dn["group"].astype(str) == g]
            if gd.empty:
                continue
            show = g not in seen
            seen.add(g)
            fig.add_trace(go.Scatter(x=gd["age"], y=gd["value"], mode="markers", name=g, legendgroup=g,
                          showlegend=show, marker={"color": col, "size": 4, "opacity": 0.4}), row=rr, col=cc)
            if len(gd) >= 8 and gd["age"].nunique() > 3:
                sl, ic, _, p, _ = _ss.linregress(gd["age"], gd["value"])
                rho, _ = _ss.spearmanr(gd["value"], gd["age"])
                sig = bool(np.isfinite(p) and p < 0.05)
                # draw ALL three group lines so you can compare — solid = significant, dotted = not
                xs = np.array([gd["age"].min(), gd["age"].max()])
                fig.add_trace(go.Scatter(x=xs, y=sl * xs + ic, mode="lines", legendgroup=g, showlegend=False,
                              opacity=1.0 if sig else 0.5,
                              line={"color": col, "width": 2.6 if sig else 1.4, "dash": "solid" if sig else "dot"},
                              hovertemplate=f"{g}: ρ={rho:+.2f}, p={p:.2g}<extra></extra>"), row=rr, col=cc)
                anns.append((g, col, rho, sig))
                if sig:
                    summ.append(net)
        # per-panel ρ readout for every group (colour-matched; * = significant) to compare across CN/MCI/AD
        for j, (g, col, rho, sig) in enumerate(anns):
            fig.add_annotation(text=f"{g} ρ={rho:+.2f}{'*' if sig else ''}", xref="x domain", yref="y domain",
                               x=0.02, y=0.98 - 0.11 * j, xanchor="left", yanchor="top", showarrow=False,
                               font={"size": 8, "color": col}, row=rr, col=cc)
    fig.update_layout(height=250 * nrow + 60, margin={"l": 40, "r": 10, "t": 40, "b": 40},
                      legend={"orientation": "h", "y": -0.04})
    fig.update_xaxes(tickfont={"size": 7}); fig.update_yaxes(tickfont={"size": 7})
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False}, key="cva-grid")
    st.caption("x = age (years), y = coupling ρ. **All three group regression lines are drawn** — **solid = "
               "significant (p<0.05)**, **dotted = not significant**. Each panel lists per-group Spearman ρ "
               "(★ = significant) so you can compare the age–coupling slope across CN / MCI / AD. "
               "Significant somewhere: " + (", ".join(sorted(set(summ))) if summ else "none") + ".")


def render_network_analysis_panel(csv_files: list[Path]) -> bool:
    st.subheader("Network / System Analysis")
    st.markdown(
        "<div class='definition-card'><h3>Which brain network / system is affected?</h3>"
        "<p>Node-level features (DTI microstructure, graph metrics, structure-microstructure coupling) and the "
        "166×166 structural connectome are aggregated into brain networks. <b>Affectedness</b> ranks networks "
        "by the CN-vs-AD Cliff's δ within each feature family (BH-FDR across networks). Structural connectomes "
        "only — not functional connectivity.</p></div>",
        unsafe_allow_html=True,
    )
    scheme_label = st.radio("Grouping scheme", list(NETWORK_SCHEME_COL.keys()), horizontal=True, key="net-scheme")
    scheme_key = "functional" if scheme_label == "Functional networks" else "anatomical"
    ndir = ANALYSIS_ROOT / "19_network_analysis" / scheme_key
    if not ndir.exists():
        st.info("Network analysis tables not found — run the dashboard refresh (`network_analysis` section).")
        return True
    if scheme_label == "Functional networks":
        st.caption("Functional = Yeo-7 (DMN, Limbic, Salience/VAN, DorsalAttention, Frontoparietal, Somatomotor, "
                   "Visual) + Subcortical / Cerebellar / Brainstem. Cortex→network is a documented approximation "
                   "on an anatomical atlas; tiny brainstem/thalamic nuclei have low edge support (treat as exploratory).")

    # --- affectedness ranking (table LEFT / heatmap RIGHT) ---
    aff = _net_csv(ndir / "network_affectedness_summary.csv")
    if not aff.empty and "feature_family" in aff.columns:
        st.markdown("#### Affectedness ranking (CN → MCI → AD)")
        fam = st.selectbox("Feature family", list(aff["feature_family"].dropna().unique()), key="net-aff-fam")
        sub = aff[aff["feature_family"] == fam].copy()
        sub = sub.reindex(sub["cn_ad_cliffs_delta"].abs().sort_values(ascending=False).index)
        left, right = st.columns([1.35, 1], gap="large")
        with left:
            cols = [c for c in ["network", "metric", "mean_CN", "mean_MCI", "mean_AD",
                                "cn_mci_cliffs_delta", "cn_ad_cliffs_delta", "mci_ad_cliffs_delta",
                                "kw_p", "q_kw"] if c in sub.columns]
            ren = {"cn_mci_cliffs_delta": "δ CN-MCI", "cn_ad_cliffs_delta": "δ CN-AD",
                   "mci_ad_cliffs_delta": "δ MCI-AD", "kw_p": "p (KW)", "q_kw": "q (FDR)"}
            st.dataframe(sub[cols].rename(columns=ren).round(4), hide_index=True, width="stretch")
            st.caption("δ>0 → lower in AD; δ<0 → higher in AD (e.g. diffusivity). p = Kruskal-Wallis, q = BH-FDR. "
                       "Per-pair significance stars are on the violins below.")
        with right:
            try:
                piv = sub.pivot_table(index="network", columns="metric", values="cn_ad_cliffs_delta", aggfunc="mean")
                if not piv.empty:
                    fig = go.Figure(go.Heatmap(z=piv.values, x=list(piv.columns), y=list(piv.index),
                        colorscale="RdBu", zmid=0, colorbar={"title": "CN−AD δ"},
                        hovertemplate="%{y} / %{x}<br>δ=%{z:.3f}<extra></extra>"))
                    fig.update_layout(height=max(280, 26 * len(piv.index) + 120),
                                      margin={"l": 10, "r": 10, "t": 10, "b": 60})
                    st.plotly_chart(fig, width="stretch", config={"displaylogo": False})
            except Exception as exc:
                st.caption(f"heatmap unavailable: {exc}")
        st.markdown(network_interpretation(sub, fam, scheme_label))

    # --- per-network distributions (VIOLINS + significance stars) ---
    st.markdown("#### Per-network distributions by group")
    family_files = {
        "Microstructure (FA/MD/AD/RD)": ("network_microstructure_subject.csv", "network_microstructure_stats.csv"),
        "Graph (strength/degree/eff)": ("network_graph_subject.csv", "network_graph_stats.csv"),
        "Coupling (within-network ρ)": ("network_coupling_subject.csv", "network_coupling_stats.csv"),
    }
    fam2 = st.selectbox("Feature family ", list(family_files.keys()), key="net-box-fam")
    subj_file, stats_file = family_files[fam2]
    subj = _net_csv(ndir / subj_file)
    stats_df = _net_csv(ndir / stats_file)
    if not subj.empty and {"network", "metric", "value", "group"}.issubset(subj.columns):
        metric = st.selectbox("Metric", sorted(subj["metric"].dropna().unique()), key="net-box-metric")
        d = subj[subj["metric"] == metric]
        networks = sorted(d["network"].dropna().unique())
        fig = go.Figure()
        for g, c in zip(GROUP_ORDER, GROUP_COLORS):
            gd = d[d["group"].astype(str) == g]
            fig.add_trace(go.Violin(x=gd["network"], y=gd["value"], name=g, line_color=c,
                                    box_visible=True, meanline_visible=True, points=False, scalemode="width"))
        # significance stars per network (from the matching stats file, this metric)
        if not stats_df.empty and "metric" in stats_df.columns:
            srow = stats_df[stats_df["metric"] == metric].set_index("network")
            for net in networks:
                if net in srow.index:
                    stars = _pair_star_map(srow.loc[net])
                    txt = "<br>".join(f"{k} {v}" for k, v in stars.items())
                    if txt:
                        fig.add_annotation(x=net, xref="x", yref="paper", y=1.0, yanchor="bottom",
                                           text=txt, showarrow=False, font={"size": 9}, align="center")
        fig.update_layout(violinmode="group", height=460, yaxis_title=metric, legend={"orientation": "h"},
                          margin={"l": 50, "r": 10, "t": 40, "b": 100})
        st.plotly_chart(fig, width="stretch", config={"displaylogo": False})
        st.caption("Significance stars above each network (Brunner–Munzel, BH-FDR): ✱ q<0.05 · ✱✱ q<0.01 · "
                   "✱✱✱ q<0.001. Pairs shown: CN-MCI, CN-AD, MCI-AD (only significant ones appear).")
        # system-wise (network vs network) comparisons for this family + metric
        st.markdown("#### Between-network comparisons")
        _fam_key = {"Microstructure (FA/MD/AD/RD)": "microstructure",
                    "Graph (strength/degree/eff)": "graph",
                    "Coupling (within-network ρ)": "coupling"}.get(fam2, "microstructure")
        render_groupwise_stats_table(scheme_key, _fam_key, metric)
        render_systemwise_stats(scheme_key, _fam_key, metric, "netpanel")

    # --- within / between-network connectivity block matrices ---
    st.markdown("#### Within / between-network connectivity")
    metric_file = {"mean edge weight": "block_groupmean_mean_weight.csv", "density": "block_groupmean_density.csv"}
    bm_label = st.radio("Block metric", list(metric_file.keys()), horizontal=True, key="net-block-metric")
    block = _net_csv(ndir / metric_file[bm_label])
    value_col = "mean_weight" if bm_label == "mean edge weight" else "density"
    if not block.empty and {"group", "net_a", "net_b", value_col}.issubset(block.columns):
        networks = sorted(set(block["net_a"]) | set(block["net_b"]))

        def _mat(group: str) -> pd.DataFrame:
            g = block[block["group"] == group]
            M = pd.DataFrame(np.nan, index=networks, columns=networks)
            for r in g.itertuples():
                M.loc[r.net_a, r.net_b] = getattr(r, value_col)
                M.loc[r.net_b, r.net_a] = getattr(r, value_col)
            return M

        mats = {g: _mat(g) for g in GROUP_ORDER}
        cols = st.columns(3)
        for col, g in zip(cols, GROUP_ORDER):
            with col:
                st.caption(f"{g}")
                M = mats[g]
                fig = go.Figure(go.Heatmap(z=M.values, x=networks, y=networks, colorscale="Viridis",
                    showscale=False, hovertemplate="%{y} | %{x}<br>%{z:.3g}<extra></extra>"))
                fig.update_layout(height=300, margin={"l": 4, "r": 4, "t": 4, "b": 60})
                st.plotly_chart(fig, width="stretch", config={"displaylogo": False})
        st.caption("CN − AD difference (blue = reduced in AD)")
        diff = mats["CN"] - mats["AD"]
        fig = go.Figure(go.Heatmap(z=diff.values, x=networks, y=networks, colorscale="RdBu", zmid=0,
            colorbar={"title": "CN−AD"}, hovertemplate="%{y} | %{x}<br>Δ=%{z:.3g}<extra></extra>"))
        fig.update_layout(height=400, margin={"l": 4, "r": 4, "t": 4, "b": 60})
        st.plotly_chart(fig, width="stretch", config={"displaylogo": False})
        # block NL interpretation (templated from block_stats)
        bstats = _net_csv(ndir / "block_stats.csv")
        if not bstats.empty and "cn_ad_cliffs_delta" in bstats.columns:
            bs = bstats.dropna(subset=["cn_ad_cliffs_delta"]).copy()
            if not bs.empty:
                bs["abs"] = bs["cn_ad_cliffs_delta"].abs()
                t = bs.sort_values("abs", ascending=False).iloc[0]
                pair = f"{t.get('net_a','')}–{t.get('net_b','')} ({t.get('kind','')})"
                nsig = int((bs["q_kw"] < 0.05).sum()) if "q_kw" in bs.columns else 0
                st.markdown(f"**Interpretation.** The most group-discriminating connection block is **{pair}** "
                            f"(δ={float(t['cn_ad_cliffs_delta']):+.2f}"
                            f"{', q=' + format(float(t['q_kw']), '.2g') if 'q_kw' in t and np.isfinite(t['q_kw']) else ''}); "
                            f"{nsig} of {len(bs)} blocks are FDR-significant. Positive δ = connectivity reduced in AD.")

    # --- mechanistic / compensation chains (mediation) ---
    render_mediation_panel()
    return True


def render_mediation_panel() -> None:
    """Phase R5: formal mediation X->M->Y chains with bootstrapped indirect effects."""
    med = _net_csv(ANALYSIS_ROOT / "21_mediation" / "mediation_results.csv")
    st.markdown("#### Mechanistic & compensation chains (mediation)")
    if med.empty:
        st.info("Mediation results not found — run the `analysis_mediation` step.")
        return
    st.caption("Formal mediation X→M→Y (X=structural metric, M=graph mediator, Y=diagnosis / brain-age gap / "
               "another network = compensation). Indirect effect a·b with bootstrap 95% CI; covaries age+sex. "
               "⚠️ Cross-sectional → associational, **not proof of causation**; 'compensation' is a hypothesis. "
               "Cognition scores are too sparse (≈26/530) so diagnosis / brain-age gap are used as outcomes.")
    for kind in ("disease", "compensation"):
        rows = med[med["kind"] == kind]
        if rows.empty:
            continue
        st.markdown(f"**{'Disease-outcome chains' if kind == 'disease' else 'Compensation chains (cross-network)'}**")
        for r in rows.itertuples():
            sig = "✅ significant indirect effect" if getattr(r, "sig", False) else "— no significant mediation"
            with st.container(border=True):
                st.markdown(f"**{r.chain}** &nbsp; {sig}")
                st.markdown(getattr(r, "statement", ""))
        show = rows[[c for c in ["chain", "Y", "a", "b", "indirect_ab", "ci_lo", "ci_hi",
                                 "prop_mediated", "n", "outcome_type"] if c in rows.columns]]
        st.dataframe(show.round(3), hide_index=True, width="stretch")


def render_sc_matrix_viewer_panel(csv_files: list[Path]) -> bool:
    subject_path = next((p for p in csv_files if p.name == SUBJECT_TABLE_BY_SECTION["SC Matrix Viewer"]), None)
    if subject_path is None:
        st.info("SC Matrix Viewer needs `live_global_graph_subject_table.csv` to list subjects.")
        return True
    subjects = graph_subject_options(subject_path)
    if subjects.empty:
        st.info("No subjects are available for SC Matrix Viewer.")
        return True
    qc_promoted_subjects = aal3_qc_promoted_subject_ids(str(AAL3_RESCUE_BATCH_ROOT))
    cl_repaired_subjects = claude_repaired_subject_ids()

    st.subheader("SC Matrix Viewer")
    st.markdown(
        "<div class='definition-card'>"
        "<h3>Single-Subject Structural Connectome QC</h3>"
        "<p>This viewer reads one subject's live <code>SC_AAL</code> matrix and reports descriptive QC and subject-level summaries. "
        "It does not make group-inference claims for the selected individual.</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    try:
        render_cohort_density_panel(subject_path)
    except Exception as _e:
        st.caption(f"cohort density analysis unavailable: {_e}")

    available_weights = [weight for weight in CONNECTOME_WEIGHTS if connectome_weight_has_matrices(weight)]
    if not available_weights:
        st.info(f"No connectome matrices were found in `{CONNECTOMES_DIR}`.")
        return True

    c1, c2, c3, c4 = st.columns((0.8, 1.4, 0.9, 0.9), gap="large")
    groups = ["All"] + [g for g in GROUP_ORDER if "group" in subjects.columns and g in set(subjects["group"])]
    with c1:
        group = st.selectbox("Group", groups, key="sc-viewer-group")
    filtered = subjects if group == "All" or "group" not in subjects.columns else subjects[subjects["group"] == group].copy()
    subject_rows = list(filtered.itertuples(index=False))
    labels = [subject_viewer_label(row, qc_promoted_subjects, cl_repaired_subjects) for row in subject_rows]
    label_map = {label: str(row.subject_id) for label, row in zip(labels, subject_rows)}
    if "len_mean" in available_weights and not st.session_state.get("sc-viewer-len-mean-default-applied"):
        st.session_state["sc-viewer-weight"] = "len_mean"
        st.session_state["sc-viewer-len-mean-default-applied"] = True
    with c2:
        selected = st.selectbox("Subject", labels, key=f"sc-viewer-subject-{group}")
    with c3:
        weight = st.selectbox("Matrix weight", available_weights, key="sc-viewer-weight")
    with c4:
        top_n = st.slider("Top ROI pairs", min_value=10, max_value=100, value=25, step=5, key="sc-viewer-top-n")
    threshold_pct = st.slider("Minimum displayed edge percentile", min_value=0, max_value=99, value=0, step=5, key="sc-viewer-edge-threshold")

    subject_id = label_map[selected]
    display_subject_label = selected
    matrix_path = connectome_matrix_path(subject_id, weight)
    if matrix_path is None:
        st.info(f"No `{weight}` matrix found for `{subject_id}`.")
        return True

    matrix_df = connectome_matrix(str(matrix_path))
    n = min(matrix_df.shape)
    roi_labels = connectome_labels_for_matrix(n)
    metrics = sc_matrix_metrics(matrix_df)
    valid_aal3_labels = aal3_valid_label_values(str(AAL_LABEL_CSV))
    count_matrix_path = connectome_matrix_path(subject_id, "count")
    count_matrix_df = connectome_matrix(str(count_matrix_path)) if count_matrix_path is not None else None
    count_metrics = sc_count_density_summary(count_matrix_df, valid_aal3_labels) if count_matrix_df is not None else {}
    top_edges = sc_top_edges(matrix_df, top_n, threshold_pct, roi_labels)

    left, right = st.columns((1.3, 1.0), gap="large")
    with left:
        st.plotly_chart(
            sc_matrix_figure(matrix_df, f"{display_subject_label} {weight} SC matrix", roi_labels),
            width="stretch",
            config={"displaylogo": False},
        )
        st.caption(f"Matrix source: `{matrix_path}`.")
        if subject_id in qc_promoted_subjects:
            st.caption("`-- qc` marks a subject whose production connectome was refreshed by the AAL3 rescue promotion workflow.")
    with right:
        info = CONNECTOME_WEIGHT_INFO.get(weight, CONNECTOME_WEIGHT_INFO["fd_sum"])
        st.markdown(
            "<div class='edr-definition-box'>"
            f"<h5>{escape(weight)} - {escape(info['name'])}</h5>"
            f"<p>{escape(info['definition'])}</p>"
            f"<div class='formula-line'>{escape(info['formula'])}</div>"
            f"<p>{escape(info['detail'])}</p>"
            "</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div class='edr-inference-box'>"
            f"Primary tract-count density: {format_number(count_metrics.get('tract-count density'))} "
            f"from {format_number(count_metrics.get('ROI pairs with >=1 tract'), digits=0)} connected ROI pairs "
            f"out of {format_number(count_metrics.get('possible ROI pairs'), digits=0)} possible. "
            f"The selected {escape(weight)} matrix is shown for inspection, but density here is based on raw streamline-count presence "
            "across valid AAL3 ROI labels."
            "</div>",
            unsafe_allow_html=True,
        )
        if count_metrics:
            st.dataframe(sc_metrics_table(count_metrics), width="stretch", height=360, hide_index=True)
        else:
            st.warning("No tract-count matrix is available for this subject; selected-matrix metrics are shown as fallback.")
            st.dataframe(sc_metrics_table(metrics), width="stretch", height=360, hide_index=True)

    density_df = live_tract_count_density_distribution(str(subject_path), str(CONNECTOMES_DIR), str(AAL_LABEL_CSV))
    if count_matrix_df is not None or not density_df.empty:
        st.markdown("#### Tract-Count Density Distribution")
        st.caption(
            "Density is computed only from `count`: "
            "number of valid AAL3 ROI pairs with at least one assigned streamline divided by all possible undirected valid AAL3 ROI pairs. "
            "The heatmap can show the full 170-row matrix index, but density uses the 166 real AAL3 labels from the atlas table. "
            "fd_sum, FA, MD, and length matrices are not used for this density chart."
        )
        dist_left, dist_right = st.columns(2, gap="large")
        with dist_left:
            if density_df.empty:
                st.info("No cohort count-density rows could be computed from the available count matrices.")
            else:
                st.plotly_chart(
                    tract_count_density_distribution_figure(density_df, subject_id),
                    width="stretch",
                    config={"displaylogo": False},
                )
        with dist_right:
            if count_matrix_df is None:
                st.info(f"No `count` matrix found for `{subject_id}`.")
            else:
                st.plotly_chart(
                    tract_count_edge_distribution_figure(count_matrix_df, display_subject_label, valid_aal3_labels),
                    width="stretch",
                    config={"displaylogo": False},
                )
                if count_matrix_path is not None:
                    st.caption(f"Tract-count source: `{count_matrix_path}`.")

    edge_left, edge_right = st.columns((1.15, 1), gap="large")
    with edge_left:
        st.markdown("#### Ranked ROI-vs-ROI Weights")
        if top_edges.empty:
            st.info("No positive edges are available under the current threshold.")
        else:
            st.dataframe(
                top_edges,
                width="stretch",
                height=420,
                hide_index=True,
                column_config={"Weight": st.column_config.NumberColumn("Weight", format="%.4g")},
            )
    with edge_right:
        st.markdown("#### Subject Context")
        master_path, master_row = table_row_for_subject(csv_files, "master_cohort.csv", subject_id)
        qc_cols = tuple(c for c in ("group", "age", "sex", "phase", "weight", "apoe_e4_count", "apoe_e4_carrier", "n_connectome_types", "has_conn_fd_sum", "has_conn_fa_mean", "has_conn_md_mean", "has_conn_rd_mean", "has_conn_ad_mean") if master_row is not None and c in master_row.index)
        render_subject_summary_table("Demographics / Availability", master_row, qc_cols, height=315)
        if master_path is not None:
            st.caption(f"Context source: `{master_path.relative_to(ANALYSIS_ROOT).as_posix()}`.")

    tabs = st.tabs(["DTI", "Graph", "LR/SR + Delay", "Brain Age", "EDR + ML"])
    with tabs[0]:
        _, row = table_row_for_subject(csv_files, "live_global_microstructure_subject_table.csv", subject_id)
        render_subject_summary_table(
            "Global DTI Summary",
            row,
            (
                "fa_mean_edge_mean",
                "fa_mean_edge_median",
                "md_mean_edge_mean",
                "md_mean_edge_median",
                "rd_mean_edge_mean",
                "rd_mean_edge_median",
                "ad_mean_edge_mean",
                "ad_mean_edge_median",
            ),
        )
    with tabs[1]:
        _, row = table_row_for_subject(csv_files, "live_global_graph_subject_table.csv", subject_id)
        render_subject_summary_table(
            "Global Graph Summary",
            row,
            ("mean_strength", "total_strength", "density", "global_efficiency", "charpath_len"),
        )
    with tabs[2]:
        c1, c2 = st.columns(2, gap="large")
        _, lr_row = table_row_for_subject(csv_files, "lr_sr_subject_level_fd_sum_len_mean.csv", subject_id)
        _, delay_row = table_row_for_subject(csv_files, "delay_subject_level_fd_sum_len_mean.csv", subject_id)
        with c1:
            render_subject_summary_table(
                "LR/SR",
                lr_row,
                ("short_w_mean", "long_w_mean", "sr_lr_w_diff", "sr_lr_w_ratio", "short_global_eff", "long_global_eff"),
            )
        with c2:
            render_subject_summary_table(
                "Delay",
                delay_row,
                ("mean_delay_ms", "median_delay_ms", "p90_delay_ms", "delay_burden_ms", "long_delay_fraction", "delay_path_ms"),
            )
    with tabs[3]:
        _, row = table_row_for_subject(csv_files, "live_brain_age_predictions.csv", subject_id)
        render_subject_summary_table("Brain Age / BAG", row, ("age", "brain_age_pred", "BAG", "BAG_age_corrected"))
    with tabs[4]:
        c1, c2 = st.columns(2, gap="large")
        _, edr_row = table_row_for_subject(csv_files, "edr_exception_subject_level_fd_sum_len_mean.csv", subject_id)
        with c1:
            render_subject_summary_table(
                "EDR Exceptions",
                edr_row,
                ("edr_lambda", "exception_pct", "sr_exception_pct", "lr_exception_pct", "sr_exception_strength", "lr_exception_strength"),
            )
        with c2:
            pred_path = next((p for p in csv_files if p.name == "ml_cross_validated_predictions.csv"), None)
            pred_df = _safe_read_csv(pred_path) if pred_path is not None else None
            if pred_df is None or pred_df.empty or "subject_id" not in pred_df.columns:
                st.info("ML prediction rows are not available for this subject.")
            else:
                pred_rows = pred_df[pred_df["subject_id"].astype(str) == subject_id].copy()
                st.markdown("#### ML Prediction Probabilities")
                if pred_rows.empty:
                    st.info("No cross-validated ML prediction row is available for this subject.")
                else:
                    cols = [c for c in ("model", "true_group", "predicted_group", "prob_CN", "prob_MCI", "prob_AD") if c in pred_rows.columns]
                    st.dataframe(
                        pred_rows[cols],
                        width="stretch",
                        height=compact_table_height(len(pred_rows)),
                        hide_index=True,
                        column_config={
                            "prob_CN": st.column_config.NumberColumn("P(CN)", format="%.3f"),
                            "prob_MCI": st.column_config.NumberColumn("P(MCI)", format="%.3f"),
                            "prob_AD": st.column_config.NumberColumn("P(AD)", format="%.3f"),
                        },
                    )
    return True


@st.fragment
def render_connectome_graph_viewer(subject_path_str: str) -> bool:
    subject_path = Path(subject_path_str)
    subjects = graph_subject_options(subject_path)
    if subjects.empty:
        return False
    qc_promoted_subjects = aal3_qc_promoted_subject_ids(str(AAL3_RESCUE_BATCH_ROOT))
    cl_repaired_subjects = claude_repaired_subject_ids()

    current_weight = st.session_state.get("graph-viewer-weight", CONNECTOME_WEIGHTS[0])
    render_edgewise_welch_heatmap_panel(str(subject_path))
    st.markdown("<div style='height: 1.25rem;'></div>", unsafe_allow_html=True)
    render_edgewise_group_comparison(str(current_weight), str(subject_path))
    st.markdown("<div style='height: 1.25rem;'></div>", unsafe_allow_html=True)

    st.subheader("Network Graph Viewer")
    st.caption("A graph here is the structural connectome: AAL3 regions are nodes and tractography-derived matrix values are weighted edges.")
    c1, c2, c3, c4 = st.columns((0.9, 1.5, 1.0, 1.0))
    groups = ["All"] + [g for g in GROUP_ORDER if "group" in subjects.columns and g in set(subjects["group"])]
    with c1:
        group = st.selectbox("Group", groups, key="graph-viewer-group")
    filtered = subjects if group == "All" or "group" not in subjects.columns else subjects[subjects["group"] == group].copy()
    subject_rows = list(filtered.itertuples(index=False))
    subject_labels = [subject_viewer_label(row, qc_promoted_subjects, cl_repaired_subjects) for row in subject_rows]
    subject_map = {label: str(row.subject_id) for label, row in zip(subject_labels, subject_rows)}
    with c2:
        selected_subject_label = st.selectbox("Subject", subject_labels, key=f"graph-viewer-subject-{group}")
    with c3:
        weight = st.selectbox(
            "Matrix weight",
            CONNECTOME_WEIGHTS,
            index=0,
            key="graph-viewer-weight",
            help="fd_sum/count are structural connection weights. fa_mean/md_mean are edge-wise microstructure attributes sampled on streamlines.",
        )
    with c4:
        top_edges = st.slider("Top edges", min_value=50, max_value=600, value=160, step=50, key="graph-viewer-top-edges")

    subject_id = subject_map[selected_subject_label]
    display_subject_label = selected_subject_label
    matrix_path = connectome_matrix_path(subject_id, weight)
    if matrix_path is None:
        st.info(f"No `{weight}` connectome matrix found for `{subject_id}` in `{CONNECTOMES_DIR}`.")
        return True

    matrix_df = connectome_matrix(str(matrix_path))
    edge_df, node_df = connectome_edges_and_nodes(matrix_df, top_edges)
    if edge_df.empty or node_df.empty:
        st.info("Could not build a graph view from this matrix and AAL3 atlas.")
        return True

    graph_title = f"{display_subject_label} {weight} graph, strongest {len(edge_df)} edges"
    left, right = st.columns((1.45, 1), gap="large")
    with left:
        st.plotly_chart(connectome_graph_figure(edge_df, node_df, graph_title), width="stretch", config={"displaylogo": False})
        st.caption(f"Matrix source: `{matrix_path}`.")
        if subject_id in qc_promoted_subjects:
            st.caption("`-- qc` marks a subject whose production connectome was refreshed by the AAL3 rescue promotion workflow.")
        st.markdown(graph_weight_info_html(weight, display_subject_label, len(node_df), len(edge_df)), unsafe_allow_html=True)
    with right:
        table_rows = min(max(20, top_edges // 2), 100, len(edge_df))
        top_edges_df = edge_df.head(table_rows).copy()
        node_lookup = node_df.set_index("matrix_index")
        top_edges_df["ROI A"] = top_edges_df["i"].map(lambda idx: node_lookup.loc[int(idx), "atlas_label"])
        top_edges_df["ROI B"] = top_edges_df["j"].map(lambda idx: node_lookup.loc[int(idx), "atlas_label"])
        st.markdown("#### Ranked Individual ROI-vs-ROI Edges")
        st.dataframe(
            top_edges_df[["ROI A", "ROI B", "weight"]],
            width="stretch",
            height=820,
            hide_index=True,
            column_config={"weight": st.column_config.NumberColumn("Weight", format="%.3g")},
        )
        st.caption(
            f"Showing the top {len(top_edges_df):,} ROI pairs from the selected individual matrix. "
            f"AAL3 contributes {len(node_df):,} labelled nodes here; the table is intentionally ranked/truncated, not a full ROI-by-ROI listing."
        )
    with st.expander("Adjacency Matrix Heatmap", expanded=False):
        st.plotly_chart(connectome_heatmap_figure(matrix_df, f"{display_subject_label} {weight} adjacency matrix"), width="stretch", config={"displaylogo": False})
    return True


def render_live_charts(
    section: Section,
    csv_files: list[Path],
    max_charts: int,
    hide_visual_outliers: bool,
) -> bool:
    _prov = cohort_provenance_caption()
    if _prov:
        st.caption(_prov)
    if section.label == "Novel Findings":
        return render_novel_findings_panel(csv_files)
    if section.label == "Demographics":
        render_demographics_extras()
        return True
    if section.label == "Global DTI":
        ok = render_global_dti_layout(section, csv_files, hide_visual_outliers)
        render_network_breakdown("micro", "global-dti")
        return ok
    if section.label == "Local / ROI DTI":
        return render_local_roi_pairwise_panel(csv_files, hide_visual_outliers)
    if section.label == "Global Graph":
        subject_path = subject_table_for_section(section, csv_files)
        if subject_path is not None:
            render_connectome_graph_viewer(str(subject_path))
        render_network_breakdown("graph", "global-graph")
        st.divider()
        render_graph_metric_group_frame()
        st.divider()
        render_graph_metric_concordance()
    if section.label == "SC Matrix Viewer":
        return render_sc_matrix_viewer_panel(csv_files)
    if section.label == "Network Analysis":
        return render_network_analysis_panel(csv_files)
    if section.label == "Node Metrics":
        return render_node_metrics_layout(csv_files, hide_visual_outliers)
    if section.label == "Coupling":
        ok = render_coupling_layout(section, csv_files, hide_visual_outliers)
        render_network_coupling_breakdown("coupling")
        st.divider()
        render_coupling_metric_group_frame()
        st.divider()
        render_coupling_vs_age()
        return ok
    if section.label == "Coupling-AAL":
        return render_coupling_aal_panel(csv_files, hide_visual_outliers)
    if section.label == "Brain Age":
        return render_brain_age_layout(section, csv_files, max_charts, hide_visual_outliers)
    if section.label == "LR / SR":
        return render_lr_sr_layout(section, csv_files, hide_visual_outliers)
    if section.label == "LR-SR Analysis":
        return render_lr_sr_workalong_layout(section, csv_files, hide_visual_outliers)
    if section.label == "EDR Exceptions":
        return render_edr_exceptions_panel(section, csv_files, max_charts, hide_visual_outliers)
    if section.label == "ML Diagnostics":
        return render_ml_diagnostics_panel(csv_files)
    if section.label == "Delay":
        return render_delay_panel(section, csv_files, hide_visual_outliers)
    if section.label == "Advanced":
        return render_advanced_panel(section, csv_files, hide_visual_outliers)
    if render_subject_metric_charts(section, csv_files, max_charts, hide_visual_outliers):
        return True
    if render_node_long_charts(section, csv_files, max_charts, hide_visual_outliers):
        return True
    return render_ranked_csv_charts(section, csv_files, max_charts)


def show_tables(paths: list[Path], allow_raw: bool, key_prefix: str) -> None:
    if not paths:
        return
    summary_like = [
        p for p in paths
        if any(token in p.name.lower() for token in ("summary", "pairwise", "descriptives", "tests", "components", "availability", "counts", "snapshot", "manifest", "requirements"))
    ]
    other = [p for p in paths if p not in summary_like]
    table_paths = summary_like + other
    if not allow_raw:
        table_paths = [p for p in table_paths if not is_raw_subject_table(p)]
    st.subheader("Statistical Tables")
    if not table_paths:
        st.info("Only subject-level/raw tables were found here. Enable raw table display in the sidebar to show them.")
        return
    with st.expander("Browse statistical tables", expanded=False):
        selected = st.selectbox(
            "Table",
            table_paths,
            format_func=lambda p: p.relative_to(ANALYSIS_ROOT).as_posix(),
            key=f"{key_prefix}-table-{hash(tuple(str(p) for p in table_paths))}",
        )
        df = _safe_read_csv(selected)
        if df is not None:
            display_df = clean_table_for_display(df)
            st.caption(f"{selected.relative_to(ANALYSIS_ROOT)} | rows={len(display_df):,} cols={len(display_df.columns):,}")
            st.dataframe(
                display_df,
                width="stretch",
                height=compact_table_height(len(display_df)),
                hide_index=True,
            )
            st.download_button(
                "Download selected table",
                data=selected.read_bytes(),
                file_name=selected.name,
                mime="text/csv",
                key=f"{key_prefix}-download-{selected}",
            )


def section_status_table() -> pd.DataFrame | None:
    p = ANALYSIS_ROOT / "exports" / "section_status.csv"
    return _safe_read_csv(p) if p.exists() else None


def section_key(label: str) -> str:
    return (
        label.lower()
        .replace(" / ", "-")
        .replace("/", "-")
        .replace(" ", "-")
        .replace("+", "plus")
    )


def live_chart_count_for_section(section: Section, csv_files: list[Path]) -> int:
    if section.label == "Demographics":
        return 2
    subject_path = subject_table_for_section(section, csv_files)
    if subject_path is not None:
        if section.label == "Coupling":
            summary_paths = [p for p in csv_files if p.name == "live_coupling_summary.csv"]
            if summary_paths:
                summary = _safe_read_csv(summary_paths[0])
                if summary is not None and "metric" in summary.columns:
                    return len(summary["metric"].dropna().unique())
        try:
            header = pd.read_csv(subject_path, nrows=0)
            return len(numeric_metric_columns(header, section.label))
        except Exception:
            return 0
    long_path = long_node_table_for_section(section, csv_files)
    if long_path is not None:
        return len([p for p in csv_files if p.name.endswith("_tests.csv")])
    return len([
        p for p in csv_files
        if (
            p.name.endswith("_tests.csv")
            or p.name.startswith("edges_")
            or p.name.startswith("top_")
            or "disease_gradient" in p.name
            or p.name.endswith("_components.csv")
        )
    ])


def section_output_counts(section: Section, inv: dict) -> tuple[int, int, int]:
    folders = section.folders
    csv_files = files_for_folders(inv["csv"], folders, (".csv",))
    return (
        len(files_for_folders(inv["md"], folders, (".md",))),
        len(csv_files),
        live_chart_count_for_section(section, csv_files),
    )


def top_summary_metrics(inv: dict) -> None:
    snapshot_path = ANALYSIS_ROOT / "analysis_snapshot.csv"
    qc_path = ANALYSIS_ROOT / "01_qc" / "data_completeness.csv"
    section_status = section_status_table()
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("PNG figures", len(inv["png"]))
    c2.metric("CSV tables", len(inv["csv"]))
    c3.metric("Inference notes", len(inv["md"]))
    if snapshot_path.exists():
        snap = _safe_read_csv(snapshot_path)
        if snap is not None and not snap.empty:
            c4.metric("Connectomes", int(snap.get("n_ALL", pd.Series([0])).iloc[0]))
            c5.metric("Mode", str(snap.get("mode", pd.Series(["unknown"])).iloc[0]))
    elif qc_path.exists():
        c4.metric("Completeness table", "ready")
    if section_status is not None:
        ok_count = int((section_status["status"] == "ok").sum())
        skipped = int(section_status["status"].astype(str).str.startswith("skipped").sum())
        st.caption(f"Section execution: {ok_count} ok, {skipped} legacy sections skipped and replaced where live fallbacks exist.")


@st.cache_data(ttl=60, show_spinner=False)
def aal3_rescue_cohort_counts(rescue_parent_str: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    seen: dict[str, set[str]] = {group: set() for group in GROUP_ORDER}
    for run_root in aal3_rescue_base_roots(Path(rescue_parent_str)):
        subject_path = run_root / "subjects.csv"
        if not subject_path.exists() or subject_path.stat().st_size <= 0:
            continue
        subjects = _safe_read_csv(subject_path)
        if subjects is None or subjects.empty or not {"sid", "group"}.issubset(subjects.columns):
            continue
        for row in subjects[["sid", "group"]].dropna().itertuples(index=False):
            group = str(row.group).strip().upper()
            sid = str(row.sid).strip()
            if group in seen and sid:
                seen[group].add(sid)
    for group in GROUP_ORDER:
        rows.append({"group": group, "n_rescue_subjects": len(seen[group])})
    return pd.DataFrame(rows)


def render_group_summary() -> None:
    summary = group_summary(str(ANALYSIS_ROOT))
    if summary.empty:
        return
    rescue_counts = aal3_rescue_cohort_counts(str(AAL3_RESCUE_BATCH_ROOT))

    _total = int(summary["n_subjects"].sum())
    st.subheader(f"Cohort Counts — Dense cohort (n={_total:,})")
    st.caption(
        "Analysis cohort = the dense connectomes (count-density ≥ 0.6). Every statistic, p-value and model "
        "below is computed on this set. Diffusivity tabs (Global/Local DTI, Coupling, Brain Age) use the "
        "subset with FA/MD/AD/RD matrices (n=529; one subject is structural-only — kept in graph analyses)."
    )
    cols = st.columns(4)
    by_group = summary.set_index("group")
    for idx, group in enumerate(GROUP_ORDER):
        n = int(by_group.loc[group, "n_subjects"]) if group in by_group.index else 0
        ready = ""
        if group in by_group.index and "n_with_final_connectomes" in by_group.columns and pd.notna(by_group.loc[group, "n_with_final_connectomes"]):
            ready = f"{int(by_group.loc[group, 'n_with_final_connectomes'])} final connectomes"
        cols[idx].metric(group, f"{n:,}", ready)
    cols[3].metric("Total (dense)", f"{_total:,}")

    chart_cols = st.columns((1, 1, 1)) if not rescue_counts.empty else st.columns((1, 1))
    cohort_chart = summary[["group", "n_subjects"]].rename(columns={"group": "Group", "n_subjects": "Subjects"})
    with chart_cols[0]:
        st.caption("Live chart from `02_demographics/group_counts.csv`.")
        st.altair_chart(
            labeled_bar_chart(cohort_chart, x_col="Group", y_col="Subjects", y_title="Subjects"),
            width="stretch",
        )
    if "n_with_final_connectomes" in summary.columns:
        final_chart = (
            summary[["group", "n_with_final_connectomes"]]
            .fillna(0)
            .rename(columns={"group": "Group", "n_with_final_connectomes": "Final connectomes"})
        )
        with chart_cols[1]:
            st.caption("Production final-connectome count from `01_qc/data_completeness.csv`.")
            st.altair_chart(
                labeled_bar_chart(final_chart, x_col="Group", y_col="Final connectomes", y_title="Final connectomes"),
                width="stretch",
            )
    if not rescue_counts.empty:
        rescue_chart = rescue_counts.rename(columns={"group": "Group", "n_rescue_subjects": "AAL3 rescue subjects"})
        with chart_cols[-1]:
            st.caption("Active AAL3 rescue cohort from rescue-batch `subjects.csv` files.")
            st.altair_chart(
                labeled_bar_chart(
                    rescue_chart,
                    x_col="Group",
                    y_col="AAL3 rescue subjects",
                    y_title="AAL3 rescue subjects",
                ),
                width="stretch",
            )

    show_cols = [
        c for c in (
            "group",
            "n_subjects",
            "mean_age",
            "female_n",
            "n_step7_tracks_done",
            "n_step7_sift2_done",
            "n_step7_parc_done",
            "n_step7_dti_done",
            "n_with_final_connectomes",
        )
        if c in summary.columns
    ]
    with st.expander("Cohort count table", expanded=False):
        st.dataframe(summary[show_cols], width="stretch", hide_index=True)
    if not rescue_counts.empty:
        with st.expander("AAL3 rescue cohort count table", expanded=False):
            st.dataframe(rescue_counts, width="stretch", hide_index=True)


def render_data_provenance() -> None:
    status = refresh_status(str(ANALYSIS_ROOT))
    refreshed = status.get("last_refresh_utc", "unknown")
    refresh_mode = status.get("mode", "unknown")
    connectomes_dir = status.get("connectomes_dir", str(ANALYSIS_ROOT.parent.parent / "connectomes"))
    st.info(
        f"Data source: generated analysis snapshot in `{ANALYSIS_ROOT}`. Last refresh: `{refreshed}` "
        f"({refresh_mode}). Inputs are cohort CSVs plus connectome files in `{connectomes_dir}`. "
        "Counts and section charts are computed live from refreshed CSV tables with a 60-second app cache. "
        "Stored PNG exports are ignored by this web view."
    )


def labeled_bar_chart(data: pd.DataFrame, x_col: str, y_col: str, y_title: str) -> alt.LayerChart:
    chart_data = data.copy()
    chart_data[y_col] = pd.to_numeric(chart_data[y_col], errors="coerce").fillna(0)
    y_max = max(float(chart_data[y_col].max()), 1.0)
    y_domain = [0, y_max * 1.18]
    base = alt.Chart(chart_data).encode(
        x=alt.X(f"{x_col}:N", sort=list(GROUP_ORDER), axis=alt.Axis(title=None, labelAngle=0)),
        y=alt.Y(f"{y_col}:Q", scale=alt.Scale(domain=y_domain), axis=alt.Axis(title=y_title)),
        tooltip=[
            alt.Tooltip(f"{x_col}:N", title="Group"),
            alt.Tooltip(f"{y_col}:Q", title=y_title, format=",.0f"),
        ],
    )
    bars = base.mark_bar(color="#7ec8ff", cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
    labels = base.mark_text(
        align="center",
        baseline="bottom",
        dy=-5,
        color=CHART_TEXT,
        fontSize=13,
        fontWeight="bold",
    ).encode(text=alt.Text(f"{y_col}:Q", format=",.0f"))
    return (bars + labels).properties(height=240, background=CHART_BACKGROUND).configure_axis(
        labelColor=CHART_TEXT,
        titleColor=CHART_TEXT,
        gridColor=CHART_GRID,
        domainColor=CHART_STROKE,
        tickColor=CHART_STROKE,
    ).configure_view(fill=CHART_BACKGROUND, stroke=CHART_STROKE)


def complete_group_sex_counts(data: pd.DataFrame) -> pd.DataFrame:
    if data.empty:
        return pd.DataFrame({"Group": [], "Sex": [], "Count": []})
    rows = []
    for group in GROUP_ORDER:
        for sex in ("Female", "Male"):
            match = data[(data["Group"] == group) & (data["Sex"] == sex)]
            count = int(match["Count"].iloc[0]) if not match.empty else 0
            rows.append({"Group": group, "Sex": sex, "Count": count})
    return pd.DataFrame(rows)


def sex_distribution_chart(chart_data: pd.DataFrame, title: str) -> alt.LayerChart:
    chart_data = complete_group_sex_counts(chart_data)
    if chart_data.empty:
        return alt.Chart(pd.DataFrame({"Group": [], "Sex": [], "Count": []})).mark_bar()
    y_max = max(float(chart_data["Count"].max()), 1.0)
    base = alt.Chart(chart_data).encode(
        x=alt.X("Group:N", sort=list(GROUP_ORDER), axis=alt.Axis(title=None, labelAngle=0)),
        xOffset=alt.XOffset("Sex:N"),
        y=alt.Y("Count:Q", scale=alt.Scale(domain=[0, y_max * 1.24]), axis=alt.Axis(title=title)),
        color=alt.Color(
            "Sex:N",
            scale=alt.Scale(domain=["Female", "Male"], range=["#7ec8ff", "#4f83ff"]),
            legend=alt.Legend(orient="top", title=None),
        ),
        tooltip=["Group:N", "Sex:N", alt.Tooltip("Count:Q", format=",.0f")],
    )
    bars = base.mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
    labels = base.mark_text(
        align="center",
        baseline="bottom",
        dy=-4,
        color=CHART_TEXT,
        fontSize=12,
        fontWeight="bold",
    ).encode(text=alt.Text("Count:Q", format=",.0f"))
    return (bars + labels).properties(
        height=260,
        background=CHART_BACKGROUND,
        title=alt.TitleParams(text=title, anchor="start", color=CHART_TEXT, fontSize=15),
    ).configure_axis(
        labelColor=CHART_TEXT,
        titleColor=CHART_TEXT,
        gridColor=CHART_GRID,
        domainColor=CHART_STROKE,
        tickColor=CHART_STROKE,
    ).configure_legend(
        labelColor=CHART_TEXT,
        titleColor=CHART_TEXT,
        orient="top",
    ).configure_view(fill=CHART_BACKGROUND, stroke=CHART_STROKE)


def normalize_sex_label(value: object) -> str | None:
    sex = str(value).strip().upper()
    if sex.startswith("F"):
        return "Female"
    if sex.startswith("M"):
        return "Male"
    return None


def age_test_note(data: pd.DataFrame, group_col: str) -> str:
    if data.empty or group_col not in data.columns or "Age" not in data.columns:
        return ""
    pieces = []
    groups = [g for g in data[group_col].dropna().astype(str).unique().tolist() if g]
    if group_col == "Group":
        groups = [g for g in GROUP_ORDER if g in set(data["Group"].astype(str))]
    if group_col == "Sex":
        groups = [g for g in ("Female", "Male") if g in set(data["Sex"].astype(str))]
    values = [
        pd.to_numeric(data.loc[data[group_col].astype(str) == group, "Age"], errors="coerce").dropna()
        for group in groups
    ]
    values = [v for v in values if len(v) >= 3]
    try:
        if len(values) >= 3:
            pieces.append(f"Kruskal-Wallis p={format_p(stats.kruskal(*values).pvalue)}")
        elif len(values) == 2:
            pieces.append(f"Mann-Whitney p={format_p(stats.mannwhitneyu(values[0], values[1], alternative='two-sided').pvalue)}")
    except Exception:
        pass
    med_parts = []
    for group in groups:
        series = pd.to_numeric(data.loc[data[group_col].astype(str) == group, "Age"], errors="coerce").dropna()
        if not series.empty:
            med_parts.append(f"{group} median={format_number(series.median(), digits=1)}")
    if med_parts:
        pieces.append("; ".join(med_parts))
    return " | ".join(pieces)


def age_violin_chart(chart_data: pd.DataFrame, x_col: str, title: str, order: list[str], colors: dict[str, str]) -> go.Figure:
    data = chart_data.copy()
    data["Age"] = pd.to_numeric(data["Age"], errors="coerce")
    data = data[data["Age"].notna() & data[x_col].notna()].copy()
    fig = go.Figure()
    all_ages = data["Age"].dropna()
    if all_ages.empty:
        return fig
    y_min = max(0, float(all_ages.min()) - 4)
    y_max = float(all_ages.max()) + 7
    for item in order:
        vals = data.loc[data[x_col].astype(str) == str(item), "Age"].dropna()
        if vals.empty:
            continue
        fig.add_trace(
            go.Violin(
                x=[item] * len(vals),
                y=vals,
                name=item,
                points=False,
                box_visible=False,
                meanline_visible=True,
                line={"color": colors.get(item, "#5dade2"), "width": 1.5},
                fillcolor=colors.get(item, "#5dade2"),
                opacity=0.72,
                hovertemplate=f"{x_col}={item}<br>Age=%{{y:.1f}}<extra></extra>",
            )
        )
        q1 = vals.quantile(0.25)
        median = vals.quantile(0.50)
        q3 = vals.quantile(0.75)
        fig.add_annotation(
            x=item,
            y=y_max - 1.0,
            text=f"n={len(vals)}<br>Med={median:.1f}<br>Q1={q1:.1f} | Q3={q3:.1f}",
            showarrow=False,
            align="center",
            bgcolor="rgba(255,255,255,0.84)",
            bordercolor="#cbd5e1",
            borderpad=4,
            font={"size": 11, "color": CHART_TEXT},
        )
    subtitle = age_test_note(data, x_col)
    fig.update_layout(
        title={"text": f"<b>{escape(title)}</b>" + (f"<br><sup>{escape(subtitle)}</sup>" if subtitle else ""), "x": 0.0, "xanchor": "left"},
        template="plotly_white",
        paper_bgcolor=CHART_BACKGROUND,
        plot_bgcolor=CHART_BACKGROUND,
        font={"color": CHART_TEXT, "size": 13},
        title_font={"color": CHART_TEXT, "size": 16},
        height=430,
        margin={"l": 58, "r": 28, "t": 78, "b": 55},
        showlegend=False,
        yaxis={
            "title": {"text": "Age", "font": {"color": CHART_TEXT}},
            "range": [y_min, y_max],
            "gridcolor": CHART_GRID,
            "linecolor": CHART_STROKE,
            "tickfont": {"color": CHART_TEXT},
        },
        xaxis={
            "title": "",
            "categoryorder": "array",
            "categoryarray": order,
            "tickfont": {"color": CHART_TEXT, "size": 12},
            "linecolor": CHART_STROKE,
        },
    )
    return fig


@st.cache_data(ttl=60)
def age_distribution_tables(root_str: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = Path(root_str)
    master_path = root / "00_master" / "master_cohort.csv"
    completeness_path = root / "01_qc" / "data_completeness_subject_level.csv"
    empty = pd.DataFrame(columns=["subject_id", "Group", "Sex", "Age"])
    if not master_path.exists():
        return empty, empty
    master = pd.read_csv(master_path)
    age_col = "age" if "age" in master.columns else "Age" if "Age" in master.columns else None
    sex_col = "sex" if "sex" in master.columns else "Sex" if "Sex" in master.columns else None
    if not {"subject_id", "group"}.issubset(master.columns) or age_col is None or sex_col is None:
        return empty, empty
    subjects = master[["subject_id", "group", sex_col, age_col]].copy()
    subjects = subjects.rename(columns={"group": "Group", sex_col: "Sex", age_col: "Age"})
    subjects["Group"] = subjects["Group"].astype(str)
    subjects["Sex"] = subjects["Sex"].map(normalize_sex_label)
    subjects["Age"] = pd.to_numeric(subjects["Age"], errors="coerce")
    subjects = subjects[subjects["Group"].isin(GROUP_ORDER) & subjects["Sex"].notna() & subjects["Age"].notna()].copy()
    subjects = subjects.drop_duplicates("subject_id").reset_index(drop=True)
    available = empty.copy()
    if completeness_path.exists():
        completeness = pd.read_csv(completeness_path)
        if {"subject_id", "final_connectome_ready"}.issubset(completeness.columns):
            ready = completeness[pd.to_numeric(completeness["final_connectome_ready"], errors="coerce").fillna(0) > 0]
            available = subjects.merge(ready[["subject_id"]].drop_duplicates(), on="subject_id", how="inner")
    return subjects, available


@st.cache_data(ttl=60)
def sex_distribution_tables(root_str: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = Path(root_str)
    master_path = root / "00_master" / "master_cohort.csv"
    completeness_path = root / "01_qc" / "data_completeness_subject_level.csv"
    group_counts_path = root / "02_demographics" / "group_counts.csv"

    full_counts = pd.DataFrame({"Group": [], "Sex": [], "Count": []})
    available_counts = pd.DataFrame({"Group": [], "Sex": [], "Count": []})

    if master_path.exists():
        master = pd.read_csv(master_path)
        if {"subject_id", "group"}.issubset(master.columns):
            sex_col = "sex" if "sex" in master.columns else "Sex" if "Sex" in master.columns else None
            if sex_col:
                subjects = master[["subject_id", "group", sex_col]].copy()
                subjects = subjects.rename(columns={"group": "Group", sex_col: "Sex"})
                subjects["Group"] = subjects["Group"].astype(str)
                subjects["Sex"] = subjects["Sex"].map(normalize_sex_label)
                subjects = subjects[subjects["Group"].isin(GROUP_ORDER) & subjects["Sex"].notna()]
                subjects = subjects.drop_duplicates("subject_id")
                full_counts = (
                    subjects.groupby(["Group", "Sex"], as_index=False)
                    .agg(Count=("subject_id", "nunique"))
                )
                if completeness_path.exists():
                    completeness = pd.read_csv(completeness_path)
                    if {"subject_id", "final_connectome_ready"}.issubset(completeness.columns):
                        ready = completeness[pd.to_numeric(completeness["final_connectome_ready"], errors="coerce").fillna(0) > 0]
                        ready_subjects = subjects.merge(ready[["subject_id"]].drop_duplicates(), on="subject_id", how="inner")
                        available_counts = (
                            ready_subjects.groupby(["Group", "Sex"], as_index=False)
                            .agg(Count=("subject_id", "nunique"))
                        )

    if full_counts.empty and group_counts_path.exists():
        summary = pd.read_csv(group_counts_path)
        if {"group", "n_subjects", "female_n"}.issubset(summary.columns):
            rows = []
            for _, row in summary.iterrows():
                group = str(row["group"])
                total = int(row["n_subjects"])
                female = int(row["female_n"])
                rows.append({"Group": group, "Sex": "Female", "Count": female})
                rows.append({"Group": group, "Sex": "Male", "Count": max(total - female, 0)})
            full_counts = pd.DataFrame(rows)

    return complete_group_sex_counts(full_counts), complete_group_sex_counts(available_counts)


def render_demographics_extras() -> None:
    total_sex, available_sex = sex_distribution_tables(str(ANALYSIS_ROOT))
    total_age, available_age = age_distribution_tables(str(ANALYSIS_ROOT))
    if total_sex.empty and available_sex.empty and total_age.empty and available_age.empty:
        return
    if not (total_sex.empty and available_sex.empty):
        st.subheader("Sex Distribution")
        cols = st.columns((1, 1), gap="large")
        with cols[0]:
            st.caption("Total cohort: live chart from `00_master/master_cohort.csv`.")
            st.altair_chart(sex_distribution_chart(total_sex, "Total subjects"), width="stretch")
        with cols[1]:
            st.caption("Available cohort: subjects with `final_connectome_ready=1` in `01_qc/data_completeness_subject_level.csv`.")
            st.altair_chart(sex_distribution_chart(available_sex, "Available final connectomes"), width="stretch")

    if not (total_age.empty and available_age.empty):
        st.subheader("Age Distribution")
        age_mode = st.radio(
            "Age cohort",
            ["Available final connectomes", "Total cohort"],
            index=0 if not available_age.empty else 1,
            horizontal=True,
            key="demographics-age-cohort",
        )
        age_df = available_age if age_mode == "Available final connectomes" and not available_age.empty else total_age
        st.caption(
            "Age violins are rendered live from `00_master/master_cohort.csv`; the available view filters to "
            "`final_connectome_ready=1` from `01_qc/data_completeness_subject_level.csv`."
        )
        cols = st.columns((1, 1), gap="large")
        with cols[0]:
            st.plotly_chart(
                age_violin_chart(
                    age_df,
                    "Group",
                    f"Age by disease group - {age_mode}",
                    list(GROUP_ORDER),
                    dict(zip(GROUP_ORDER, GROUP_COLORS)),
                ),
                width="stretch",
                config={"displaylogo": False},
            )
        with cols[1]:
            st.plotly_chart(
                age_violin_chart(
                    age_df,
                    "Sex",
                    f"Age by sex - {age_mode}",
                    ["Female", "Male"],
                    {"Female": "#7ec8ff", "Male": "#4f83ff"},
                ),
                width="stretch",
                config={"displaylogo": False},
            )


def render_section_cards(inv: dict) -> Section:
    if "selected_section_idx" not in st.session_state:
        st.session_state.selected_section_idx = 0
    if int(st.session_state.selected_section_idx) >= len(SECTIONS):
        st.session_state.selected_section_idx = 0

    st.subheader("Analysis Sections")
    section_by_label = {section.label: (idx, section) for idx, section in enumerate(SECTIONS)}
    # Build one column per SECTION_GROUP (with thin separators) — dynamic so adding a group can never
    # silently drop the last group from the navigation.
    n_groups = len(SECTION_GROUPS)
    spec = []
    for i in range(n_groups):
        if i > 0:
            spec.append(0.035)
        spec.append(1)
    layout_cols = st.columns(tuple(spec), gap="small")
    group_cols = [layout_cols[i * 2] for i in range(n_groups)]
    for i in range(n_groups - 1):
        with layout_cols[i * 2 + 1]:
            st.markdown("<div class='analysis-vsplit'></div>", unsafe_allow_html=True)

    for group_col, (group_label, section_labels) in zip(group_cols, SECTION_GROUPS):
        with group_col:
            with st.container(border=True):
                st.markdown(f"<div class='analysis-group-title'>{escape(group_label)}</div>", unsafe_allow_html=True)
                for label in section_labels:
                    if label not in section_by_label:
                        continue
                    idx, section = section_by_label[label]
                    is_selected = idx == st.session_state.selected_section_idx
                    if st.button(
                        section.label,
                        key=f"section-card-{idx}-{section_key(section.label)}",
                        type="primary" if is_selected else "secondary",
                        width="stretch",
                    ):
                        if int(st.session_state.selected_section_idx) != idx:
                            st.session_state.selected_section_idx = idx
                            st.rerun()

    return SECTIONS[int(st.session_state.selected_section_idx)]


def render_section(
    section: Section,
    inv: dict,
    allow_raw: bool,
    max_charts: int,
    hide_visual_outliers: bool,
) -> None:
    st.header(section.label)
    folders = section.folders
    md_files = files_for_folders(inv["md"], folders, (".md",))
    csv_files = files_for_folders(inv["csv"], folders, (".csv",))
    if not (md_files or csv_files):
        st.warning("No generated outputs found for this section yet.")
        return
    key_prefix = section_key(section.label)
    if section.label == "Overview":
        show_tables(csv_files, allow_raw=allow_raw, key_prefix=key_prefix)
        return
    rendered_charts = render_live_charts(
        section,
        csv_files,
        max_charts=max_charts,
        hide_visual_outliers=hide_visual_outliers,
    )
    if not rendered_charts:
        st.info("No live CSV chart recipe is available for this section yet. Tables are still shown below.")
    if section.label in ("ML Diagnostics", "Novel Findings", "LR-SR Analysis"):
        return
    show_analysis_notes(md_files)
    show_tables(csv_files, allow_raw=allow_raw, key_prefix=key_prefix)


def _load_json_object(path: Path) -> dict:
    try:
        value = __import__("json").loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def _latest_gmwmi_counterfactual_state() -> dict:
    decision = _load_json_object(HCP379_GMWMI_DECISION)
    if decision:
        return {
            "status": str(decision.get("status", "DECISION_AVAILABLE")),
            "completed": 4,
            "total": 4,
            "attempt": Path(str(decision.get("summary", ""))).parent.name,
        }
    if not HCP379_GMWMI_ATTEMPTS.exists():
        return {}
    attempts = sorted(
        (path for path in HCP379_GMWMI_ATTEMPTS.iterdir() if path.is_dir()),
        reverse=True,
    )
    for attempt in attempts:
        summary = _load_json_object(attempt / "summary.json")
        if summary:
            units = summary.get("units", {})
            return {
                "status": str(summary.get("status", "COMPLETE")),
                "completed": 4 if isinstance(units, dict) and len(units) == 2 else len(units),
                "total": 4,
                "attempt": attempt.name,
            }
        state = _load_json_object(attempt / "state.json")
        if state:
            completed = state.get("completed_jobs", [])
            remaining = state.get("remaining_jobs", [])
            observed_outputs = len(list(attempt.glob("*/*/count_hroi.csv")))
            return {
                "status": str(state.get("status", "UNKNOWN")),
                "completed": max(
                    len(completed) if isinstance(completed, list) else 0,
                    observed_outputs,
                ),
                "total": (
                    len(completed) + len(remaining)
                    if isinstance(completed, list) and isinstance(remaining, list)
                    else 4
                ),
                "attempt": attempt.name,
            }
    return {}


def _latest_fod_cutoff_counterfactual_state() -> dict:
    declaration = _load_json_object(HCP379_FOD_CUTOFF_PREDECLARATION)
    validation = _load_json_object(HCP379_FOD_CUTOFF_VALIDATION)
    predeclared = (
        declaration.get("status")
        == "PREDECLARED_READY_FOR_BOUNDED_EXECUTION"
        and validation.get("status") == "PASS"
    )
    validation_label = (
        f"{int(validation.get('passed_checks', 0) or 0)}/"
        f"{int(validation.get('total_checks', 0) or 0)}"
        if validation
        else "unavailable"
    )
    if HCP379_FOD_CUTOFF_ATTEMPTS.exists():
        attempts = sorted(
            (
                path
                for path in HCP379_FOD_CUTOFF_ATTEMPTS.iterdir()
                if path.is_dir()
            ),
            reverse=True,
        )
        for attempt in attempts:
            summary = _load_json_object(attempt / "summary.json")
            if summary:
                units = summary.get("units", {})
                return {
                    "status": str(summary.get("status", "COMPLETE")),
                    "completed": (
                        4
                        if isinstance(units, dict) and len(units) == 2
                        else len(units)
                    ),
                    "total": 4,
                    "attempt": attempt.name,
                    "predeclared": predeclared,
                    "validation": validation_label,
                }
            state = _load_json_object(attempt / "state.json")
            if state:
                completed = state.get("completed_jobs", [])
                remaining = state.get("remaining_jobs", [])
                observed_outputs = len(
                    list(attempt.glob("*/*/count_hroi.csv"))
                )
                return {
                    "status": str(state.get("status", "UNKNOWN")),
                    "completed": max(
                        (
                            len(completed)
                            if isinstance(completed, list)
                            else 0
                        ),
                        observed_outputs,
                    ),
                    "total": (
                        len(completed) + len(remaining)
                        if isinstance(completed, list)
                        and isinstance(remaining, list)
                        else 4
                    ),
                    "attempt": attempt.name,
                    "predeclared": predeclared,
                    "validation": validation_label,
                }
    if declaration:
        return {
            "status": str(declaration.get("status", "PREDECLARED")),
            "completed": 0,
            "total": 4,
            "attempt": "",
            "predeclared": predeclared,
            "validation": validation_label,
        }
    return {}


def hcp379_release_tracker_data() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    summary = _load_json_object(HCP379_LIVE_SUMMARY)
    root_cause = _load_json_object(HCP379_ROOT_CAUSE_SUMMARY)
    topology = _load_json_object(HCP379_RELEASE_TOPOLOGY)
    base_rows = summary.get("rows", [])
    if not isinstance(base_rows, list) or not base_rows:
        return pd.DataFrame(), pd.DataFrame(), {}

    lane_labels = {
        "archive_calibration": "Archive calibration",
        "corrected_archive_low": "Archive-low recovery",
        "corrected_archive_D0": "Archive-D0 recovery",
        "corrected_core": "Corrected core",
        "corrected_legacy_tensor": "Corrected legacy/tensor",
        "corrected_freesurfer": "Isolated FreeSurfer recovery",
    }
    units_by_label: dict[str, list[str]] = {
        "Phase-B canaries (reused)": [
            str(row.get("unit"))
            for row in topology.get("canaries", [])
            if isinstance(row, dict) and row.get("unit")
        ]
    }
    for row in topology.get("production", []):
        if not isinstance(row, dict):
            continue
        label = lane_labels.get(str(row.get("lane")))
        unit = row.get("unit")
        if label and unit:
            units_by_label.setdefault(label, []).append(str(unit))

    ledger_by_unit: dict[str, dict] = {}
    if HCP379_LIVE_LEDGER.exists():
        try:
            ledger = pd.read_csv(HCP379_LIVE_LEDGER, dtype=str).fillna("")
            if "unit" in ledger.columns:
                ledger_by_unit = {
                    str(row["unit"]): row.to_dict()
                    for _, row in ledger.iterrows()
                }
        except Exception:
            ledger_by_unit = {}

    working_rows = [dict(row) for row in base_rows if isinstance(row, dict)]
    for row in working_rows:
        label = str(row.get("processing_group", ""))
        if label == "Phase-B canaries (reused)":
            continue
        units = units_by_label.get(label, [])
        if label == "Full release total" or not units or not ledger_by_unit:
            continue
        statuses = [
            str(ledger_by_unit.get(unit, {}).get("status", "WAITING"))
            for unit in units
        ]
        row["pretract_completed_n"] = statuses.count("PASS_COMPACTED")
        row["running_n"] = statuses.count("RUNNING")
        row["corrective_hold_n"] = statuses.count("FAILED")
        row["waiting_n"] = statuses.count("WAITING")
        row["completion_fraction"] = (
            statuses.count("PASS_COMPACTED") / len(units) if units else 0.0
        )
    detail_rows = [
        row for row in working_rows
        if str(row.get("processing_group")) != "Full release total"
    ]
    total_row = next(
        (
            row for row in working_rows
            if str(row.get("processing_group")) == "Full release total"
        ),
        {},
    )
    if detail_rows and total_row:
        total_row["pretract_completed_n"] = sum(
            int(row.get("pretract_completed_n", 0)) for row in detail_rows
        )
        total_row["running_n"] = sum(int(row.get("running_n", 0)) for row in detail_rows)
        total_row["corrective_hold_n"] = sum(
            int(row.get("corrective_hold_n", 0)) for row in detail_rows
        )
        total_row["waiting_n"] = sum(int(row.get("waiting_n", 0)) for row in detail_rows)
        total_row["completion_fraction"] = (
            int(total_row["pretract_completed_n"])
            / max(1, int(total_row.get("target_n", 530)))
        )

    corrected_state = root_cause.get("corrected_15_canary_count_only_state", {})
    density_pass_n = int(corrected_state.get("density_pass_n", 0) or 0)
    all_nine_n = int(corrected_state.get("all_nine_qc_complete_n", 0) or 0)
    final_n = int(
        root_cause.get("current_530_state", {}).get(
            "corrected_hcp379_release_available_n",
            0,
        )
        or 0
    )
    cutoff = _latest_fod_cutoff_counterfactual_state()
    gmwmi = _latest_gmwmi_counterfactual_state()
    if cutoff:
        counterfactual_action = (
            f"FOD cutoff .06→.05 screen "
            f"{cutoff.get('completed', 0)}/{cutoff.get('total', 4)} "
            f"replicates · {cutoff.get('status', 'UNKNOWN')} · "
            f"predeclaration {cutoff.get('validation', 'unavailable')}"
        )
    elif gmwmi:
        counterfactual_action = (
            f"GMWMI screen {gmwmi.get('completed', 0)}/{gmwmi.get('total', 4)} "
            f"replicates · {gmwmi.get('status', 'UNKNOWN')}"
        )
    else:
        counterfactual_action = "Uniform-recipe counterfactual pending"

    display_rows: list[dict[str, object]] = []
    for row in working_rows:
        label = str(row.get("processing_group", ""))
        target = int(row.get("target_n", 0) or 0)
        pretract = int(row.get("pretract_completed_n", 0) or 0)
        running = int(row.get("running_n", 0) or 0)
        hold = int(row.get("corrective_hold_n", 0) or 0)
        waiting = int(row.get("waiting_n", 0) or 0)
        historical_available = int(row.get("existing_hcp379_available_n", 0) or 0)
        historical_pass = int(row.get("existing_hcp379_pass_0_60_n", 0) or 0)
        historical_median = pd.to_numeric(
            pd.Series([row.get("existing_hcp379_median_density")]),
            errors="coerce",
        ).iloc[0]
        historical_text = (
            f"{historical_pass}/{historical_available} ≥.60; med {float(historical_median):.3f}"
            if historical_available and pd.notna(historical_median)
            else "not available"
        )
        is_canary = label == "Phase-B canaries (reused)"
        is_total = label == "Full release total"
        if is_canary or is_total:
            corrected_density = (
                f"{density_pass_n}/15 ≥.60 (count-only; not final)"
            )
            all_nine = f"{all_nine_n}/15"
            stability = "0/15"
        else:
            corrected_density = "pending uniform recipe"
            all_nine = f"0/{target}"
            stability = f"0/{target}"
        if is_canary:
            action = counterfactual_action
        elif pretract < target:
            action = (
                f"Pretract {pretract}/{target}; "
                f"{running} running, {hold} hold, {waiting} waiting"
            )
        else:
            action = "Pretract clear; waiting for uniform recipe gate"
        if is_total:
            action = (
                f"{counterfactual_action}; pretract {pretract}/{target}"
            )
        lane_final_n = final_n if is_total else 0
        display_rows.append(
            {
                "Lane": label,
                "Target": target,
                "Pretract complete": f"{pretract}/{target}",
                "Running": running,
                "Hold": hold,
                "Waiting": waiting,
                "Historical HCP379 ≥.60 (reference only)": historical_text,
                "Corrected density": corrected_density,
                "All-nine QC": all_nine,
                "Seed stability": stability,
                "Current action / last cleared": action,
                "Completed status": f"{lane_final_n}/{target} FINAL",
            }
        )

    localization_path = (
        PROJECT_ROOT
        / "research_audit/outputs/hcp379_backward_root_cause_audit_v1/"
        "subject_stage_localization.csv"
    )
    localization = pd.DataFrame()
    if localization_path.exists():
        try:
            localization = pd.read_csv(localization_path)
        except Exception:
            localization = pd.DataFrame()

    def passed(column: str) -> int:
        if localization.empty or column not in localization.columns:
            return 0
        return int(localization[column].astype(str).eq("PASS").sum())

    pretract_total = int(total_row.get("pretract_completed_n", 0) or 0)
    gate_rows = [
        {
            "Gate": "1 · Locked identity and source roster",
            "Current": "530/530",
            "Status": "PASS",
            "What clears it": "Exact disjoint 15 canary + 515 production scan identities",
        },
        {
            "Gate": "2 · Uniform pretract inputs",
            "Current": f"{pretract_total}/530",
            "Status": "ACTIVE" if pretract_total < 530 else "PASS",
            "What clears it": "Gradient/Eddy, b0, tensor/FOD, 5TT and compacted provenance",
        },
        {
            "Gate": "3 · HROI source labels and parcel volumes",
            "Current": f"{passed('source_label_status')}/15 canaries",
            "Status": "PASS" if passed("source_label_status") == 15 else "ACTIVE",
            "What clears it": "All 379 labels present with preserved parcel support",
        },
        {
            "Gate": "4 · Registration and final atlas geometry",
            "Current": f"{passed('registration_status')}/15 canaries",
            "Status": "PASS" if passed("registration_status") == 15 else "ACTIVE",
            "What clears it": "T1-to-b0 transform and atlas-inside-mask geometry pass",
        },
        {
            "Gate": "5 · Endpoint assignment",
            "Current": f"{passed('endpoint_assignment_status')}/15 canaries",
            "Status": "PASS" if passed("endpoint_assignment_status") == 15 else "ACTIVE",
            "What clears it": "Predeclared endpoint-assignment fraction passes",
        },
        {
            "Gate": "6 · 379-node connectivity",
            "Current": f"{passed('connected_node_status')}/15 canaries",
            "Status": "ACTIVE" if passed("connected_node_status") < 15 else "PASS",
            "What clears it": "Every HCP379 parcel has supported connectivity",
        },
        {
            "Gate": "7 · Count density ≥0.60",
            "Current": f"{density_pass_n}/15 canaries",
            "Status": "ACTIVE" if density_pass_n < 15 else "PASS",
            "What clears it": "One diagnosis-blind uniform recipe passes every canary",
        },
        {
            "Gate": "8 · All-nine matched matrices",
            "Current": f"{all_nine_n}/15 canaries",
            "Status": "PENDING" if all_nine_n == 0 else "ACTIVE",
            "What clears it": "Shape, symmetry, support, weighting and cross-family QC",
        },
        {
            "Gate": "9 · Two-seed stability and final release",
            "Current": f"{final_n}/530 final",
            "Status": "PENDING" if final_n < 530 else "PASS",
            "What clears it": "Seed stability, blinded review, hashes and independent release replay",
        },
    ]
    metadata = {
        "generated_utc": summary.get("generated_utc", "unknown"),
        "ledger_updated_utc": (
            pd.Timestamp(HCP379_LIVE_LEDGER.stat().st_mtime, unit="s", tz="UTC").isoformat()
            if HCP379_LIVE_LEDGER.exists()
            else "unknown"
        ),
        "pretract_complete_n": pretract_total,
        "historical_available_n": int(
            total_row.get("existing_hcp379_available_n", 0) or 0
        ),
        "historical_pass_n": int(
            total_row.get("existing_hcp379_pass_0_60_n", 0) or 0
        ),
        "corrected_density_pass_n": density_pass_n,
        "final_n": final_n,
        "counterfactual": counterfactual_action,
    }
    return pd.DataFrame(display_rows), pd.DataFrame(gate_rows), metadata


def render_hcp379_release_tracker() -> None:
    st.markdown("### HCP379-v2 · 530-subject release tracker")
    lane_table, gate_table, metadata = hcp379_release_tracker_data()
    if lane_table.empty:
        st.warning("The HCP379-v2 status artifacts are not available.")
        return
    metrics = st.columns(4)
    metrics[0].metric(
        "Pretract complete",
        f"{int(metadata['pretract_complete_n']):,}/530",
    )
    metrics[1].metric(
        "Historical ≥0.60",
        f"{int(metadata['historical_pass_n']):,}/{int(metadata['historical_available_n']):,}",
        help="Reference/exploratory matrices only; recipes are heterogeneous.",
    )
    metrics[2].metric(
        "Corrected count-density screen",
        f"{int(metadata['corrected_density_pass_n']):,}/15",
        help="Count-only canary evidence; this is not an all-nine final release.",
    )
    metrics[3].metric("Final inferential cohort", f"{int(metadata['final_n']):,}/530")
    st.progress(
        min(1.0, max(0.0, float(metadata["pretract_complete_n"]) / 530.0)),
        text=(
            f"Uniform pretract inputs: {int(metadata['pretract_complete_n']):,}/530 · "
            f"current isolation experiment: {metadata['counterfactual']}"
        ),
    )
    st.warning(
        "The 58 historical density passes are retained as exploratory/reference evidence only. "
        "They do not enter the final inferential cohort until the same corrected recipe and every downstream gate pass."
    )
    st.markdown("##### Lane-wise progress")
    st.dataframe(
        lane_table,
        width="stretch",
        height=340,
        hide_index=True,
        column_config={
            "Target": st.column_config.NumberColumn(format="%d"),
            "Running": st.column_config.NumberColumn(format="%d"),
            "Hold": st.column_config.NumberColumn(format="%d"),
            "Waiting": st.column_config.NumberColumn(format="%d"),
        },
        key="hcp379-release-lane-tracker",
    )
    st.caption(
        "The last column is deliberately final-release completion, not merely pretract or density completion. "
        f"Ledger read live at {metadata['ledger_updated_utc']}; density snapshot generated "
        f"{metadata['generated_utc']}. This panel refreshes every 20 seconds."
    )
    with st.expander("Nine ordered release gates and the all-nine matrix contract", expanded=True):
        st.dataframe(
            gate_table,
            width="stretch",
            height=355,
            hide_index=True,
            key="hcp379-release-gate-tracker",
        )
        st.caption(
            "All-nine matrix families: count, SIFT2 fd_sum, inverse-node-volume count, mean length, "
            "mean inverse length, FA, MD, RD, and AxD. A density pass alone does not clear this contract."
        )


@st.fragment(run_every="20s")
def render_pipeline_monitor() -> None:
    """Live pipeline monitor — reads the canonical status file written by status_writer.py."""
    import json, os
    PYBIN = "/home/ec2-user/exp/.venv_connectome_app/bin/python"
    STATUS_FILE = "/data/derivatives/qc/sc_matrix_qc/connectome_status.json"
    render_hcp379_release_tracker()
    st.divider()
    st.markdown("### Existing AAL3-166 pipeline monitor")
    try:
        s = json.load(open(STATUS_FILE))
    except Exception as exc:
        st.warning(f"Live status not available yet: {exc}")
        return
    o = s["overall"]; coh = s.get("cohort", {})
    def _bar(p, h=7):
        p = max(0.0, min(1.0, p))
        return (f'<div style="background:#2b2b2b;border-radius:4px;height:{h}px;width:100%;margin-top:8px">'
                f'<div style="background:#22c55e;width:{p*100:.1f}%;height:{h}px;border-radius:4px"></div></div>')
    def _box(html, h=40):
        return (f'<div style="border:1px solid rgba(255,255,255,0.30);border-radius:5px;padding:6px 9px;'
                f'min-height:{h}px;font-size:0.85em;margin:1px">{html}</div>')
    if s.get("alerts"):
        st.error("⚠️ " + "  |  ".join(s["alerts"]))
    else:
        st.caption(f"✓ nominal · updated {s.get('ts','?')} · auto-refresh 20s")
    _dc = {}
    try:
        _dc = json.load(open("/data/derivatives/qc/sc_matrix_qc/density_check.json"))
    except Exception:
        pass
    _cst = open("/tmp/density_check.status").read().strip() if os.path.exists("/tmp/density_check.status") else "idle"
    W = [1.25, 1.5, 1.05, 1.05, 0.8, 0.8, 0.8, 0.8, 0.55]
    hdr = st.columns(W)
    for col, t in zip(hdr, ["Task", "Progress", "EC2 m/med", "S3 m/med", "tck E/S", "tsf E/S", "csv E/S", "sift E/S", ""]):
        col.markdown(f"<span style='color:#9aa3ad;font-size:0.76em'>{t}</span>", unsafe_allow_html=True)
    def _files(dd, t):
        x = dd.get("files", {}).get(t, {})
        return f"{x.get('ec2','–')}/{x.get('s3','–')}"
    rows = [("CONNECTOMES", "overall", o["done"], o["total"], o.get("mean_d"), o.get("median_d"))]
    for g in ["AD", "MCI", "CN"]:
        v = coh.get(g, {})
        rows.append((g, g, v.get("done", 0), v.get("total", 0), v.get("mean_d"), v.get("median_d")))
    for label, key, done, total, em, emd in rows:
        c = st.columns(W); pc = done / total if total else 0
        dd = _dc.get(key, {}); s3d = dd.get("s3", {})
        _run = (sum(cc.get('running',0) for cc in coh.values()) if key=='overall' else coh.get(key,{}).get('running',0))
        c[0].markdown(_box(f"<b>{label}</b><br><span style='color:#9aa3ad;font-size:0.8em'>{done}/{total} ({pc*100:.0f}%)</span>  <span style='color:#facc15;font-weight:700'>▶{_run}</span>"), unsafe_allow_html=True)
        c[1].markdown(_box(_bar(pc)), unsafe_allow_html=True)
        c[2].markdown(_box(f"<b>{em}</b>/{emd}"), unsafe_allow_html=True)
        c[3].markdown(_box(f"<b>{s3d.get('mean','–')}</b>/{s3d.get('median','–')}"), unsafe_allow_html=True)
        c[4].markdown(_box(_files(dd, "tck")), unsafe_allow_html=True)
        c[5].markdown(_box(_files(dd, "tsf")), unsafe_allow_html=True)
        c[6].markdown(_box(_files(dd, "csv")), unsafe_allow_html=True)
        c[7].markdown(_box(_files(dd, "sift")), unsafe_allow_html=True)
        if c[8].button("🔄", key=f"recalc_{key}", disabled=(_cst == "running"), help="recompute EC2+S3 density+files (no page reload)"):
            import subprocess as _sp
            _sp.Popen([PYBIN, "/home/ec2-user/exp/pipeline/audit/derive_density_check.py"]); st.toast(f"Recalculating {label}…")
    sp, s3t = s.get("s3_pushed", 0), s.get("s3_total", 1)
    rg = s.get("runs", {}).get("regen216", {})
    ov = _dc.get("overall", {})
    c = st.columns(W); pcb = sp / s3t if s3t else 0
    c[0].markdown(_box(f"<b>S3 backup</b><br><span style='color:#9aa3ad;font-size:0.8em'>{sp}/{s3t} ({pcb*100:.0f}%)</span>"), unsafe_allow_html=True)
    c[1].markdown(_box(_bar(pcb)), unsafe_allow_html=True)
    c[2].markdown(_box(f"<b>{ov.get('ec2',{}).get('mean','–')}</b>/{ov.get('ec2',{}).get('median','–')}"), unsafe_allow_html=True)
    c[3].markdown(_box(f"<b>{ov.get('s3',{}).get('mean','–')}</b>/{ov.get('s3',{}).get('median','–')}"), unsafe_allow_html=True)
    c[4].markdown(_box(_files(ov,'tck')), unsafe_allow_html=True)
    c[5].markdown(_box(_files(ov,'tsf')), unsafe_allow_html=True)
    c[6].markdown(_box(_files(ov,'csv')), unsafe_allow_html=True)
    c[7].markdown(_box(_files(ov,'sift')), unsafe_allow_html=True)
    c[8].markdown(_box("<span style='color:#666'>–</span>"), unsafe_allow_html=True)
    rgf = _dc.get("regen", {})
    c = st.columns(W); pcr = rg.get("done", 0) / 216
    c[0].markdown(_box(f"<b>Regen rebuilt</b><br><span style='color:#9aa3ad;font-size:0.8em'>{rg.get('done',0)}/216 ({pcr*100:.0f}%)</span>"), unsafe_allow_html=True)
    c[1].markdown(_box(_bar(pcr)), unsafe_allow_html=True)
    _rge = rgf.get('ec2',{}); _rgs = rgf.get('s3',{})
    c[2].markdown(_box(f"<b>{_rge.get('mean', rg.get('mean_d','–'))}</b>/{_rge.get('median', rg.get('median_d','–'))}"), unsafe_allow_html=True)
    c[3].markdown(_box(f"<b>{_rgs.get('mean','–')}</b>/{_rgs.get('median','–')}"), unsafe_allow_html=True)
    c[4].markdown(_box(_files(rgf,'tck')), unsafe_allow_html=True)
    c[5].markdown(_box(_files(rgf,'tsf')), unsafe_allow_html=True)
    c[6].markdown(_box(_files(rgf,'csv')), unsafe_allow_html=True)
    c[7].markdown(_box(_files(rgf,'sift')), unsafe_allow_html=True)
    c[8].markdown(_box("<span style='color:#666'>–</span>"), unsafe_allow_html=True)
    st.caption((f"S3 density+files computed {_dc.get('ts','?')}" if _dc else "not computed yet — click 🔄") + ("  ·  ⏳ running" if _cst == "running" else ""))
    _totrun = sum(cc.get("running",0) for cc in coh.values())
    st.markdown(f"<span style='color:#facc15;font-weight:700;font-size:1.05em'>▶ Total running: {_totrun}</span>", unsafe_allow_html=True)
    # ---- stage funnel + per-subject manifest table (single source of truth: subject_manifest.csv) ----
    try:
        import pandas as _pd
        _ff = _pd.read_csv("/data/derivatives/qc/sc_matrix_qc/stage_funnel.csv")
        _funnel = "  →  ".join(f"{r['stage']} {r['total']}" for _, r in _ff.iterrows()
                               if r["stage"] in ("mif_dwi", "eddy", "fod", "tck", "connectome_good"))
        st.caption(f"**Stage funnel:** {_funnel}")
    except Exception:
        pass
    st.markdown("##### Per-subject manifest  ·  density from production count.csv · stages · radial · mask · EC2/S3")
    try:
        import pandas as _pd
        _adf = _pd.read_csv("/data/derivatives/qc/sc_matrix_qc/subject_manifest.csv")
        _fc1, _fc2 = st.columns(2)
        _bandsel = _fc1.selectbox("Quality band", ["All", "good (≥0.6)", "mid (0.4–0.6)", "low (<0.4)", "none"], key="audit_band")
        _cohsel = _fc2.selectbox("Cohort", ["All", "AD", "MCI", "CN"], key="audit_cohort")
        _f = _adf.copy()
        _bm = {"good (≥0.6)": "good", "mid (0.4–0.6)": "mid", "low (<0.4)": "low", "none": "none"}
        if _bandsel in _bm:
            _f = _f[_f["band"] == _bm[_bandsel]]
        if _cohsel != "All":
            _f = _f[_f["group"] == _cohsel]
        for _nc in ("density", "mask_quality", "reg_ncc"):
            _f[_nc] = _pd.to_numeric(_f[_nc], errors="coerce")
        _f["csv_ec2"] = (_f["conn_location"].astype(str) != "").astype(int)
        for _t, _e in (("tck", "tck"), ("tsf", "tsf"), ("sift", "sift"), ("csv", "csv_ec2")):
            _f[_t + "_ES"] = _f[_e].astype(str) + "/" + _f["s3_" + _t].astype(str)
        _f["s3_complete"] = _f["s3_complete"].astype(bool)
        _cols = ["sid", "group", "density", "band", "radial", "stages_present", "n_weights",
                 "mask_quality", "reg_ncc", "best_recipe", "tck_ES", "tsf_ES", "csv_ES", "sift_ES", "s3_complete"]
        _show = _f[_cols].sort_values("density", ascending=False, na_position="last")
        st.caption(f"{len(_f)} subjects · E/S = EC2/S3 present per type · radial 4=production, 2=recovery · "
                   f"good {(_adf['band']=='good').sum()} / mid {(_adf['band']=='mid').sum()} / low {(_adf['band']=='low').sum()} / none {(_adf['band']=='none').sum()}")
        st.dataframe(_show, width="stretch", hide_index=True,
                     height=min(620, 70 + 28 * min(len(_show), 19)),
                     column_config={
                         "density": st.column_config.NumberColumn("density", format="%.3f"),
                         "mask_quality": st.column_config.NumberColumn("mask_q", format="%.3f"),
                         "reg_ncc": st.column_config.NumberColumn("reg_ncc", format="%.3f"),
                         "s3_complete": st.column_config.CheckboxColumn("s3_ok"),
                     }, key="audit_table")
    except Exception as _e:
        st.caption(f"per-subject manifest unavailable: {_e}")
    # ---- data locations (EC2 + S3) ----
    with st.expander("📍 Data locations — EC2 & S3", expanded=False):
        st.markdown(
            "**EC2**  (`/data` = symlink `~/exp/data`)\n"
            "- **Connectomes** (530 good): `/data/derivatives/connectomes/SC_AAL166_<sid>_<weight>.csv`  ·  "
            "9 weights: count, count_invnodevol, fd_sum, len_mean, invlen_mean, fa_mean, md_mean, ad_mean, rd_mean\n"
            "  - mid-band (0.4–0.6) → `connectomes/_midband/`  ·  low/error (<0.4) → `connectomes/_lowband_archive/`\n"
            "- **Tracks + tsf + sift**: `/data/derivatives/tracks/<sid>/`  (tracks_final_3000k.tck, {fa,md,ad,rd}_mean.tsf, sift_weights.txt, assignments_aal.csv)\n"
            "- **Audit / source of truth**: `/data/derivatives/qc/sc_matrix_qc/subject_manifest.csv`\n"
            "- **Source inputs**: `/data/derivatives/{eddy, fod, dti, mif_dwi, biascorr_1, t1_anat, parc}`\n\n"
            "**S3**  (`s3://sabeesh/exp/`)\n"
            "- **Connectomes**: `s3://sabeesh/exp/connectomes/`  (+ `_midband/`, `_lowband_archive/`)\n"
            "- **Tracks + tsf + sift**: `s3://sabeesh/exp/tracks/<sid>/`\n"
            "- **Source inputs**: `s3://sabeesh/exp/{eddy, fod, dti, mif_dwi, …}`\n\n"
            "_EC2 ↔ S3 are mirrored for all 530 good (tck/tsf/csv/sift all match). Re-run "
            "`/tmp/subject_audit.py` then `/tmp/derive_density_check.py` after any connectome change._"
        )
    # ---- full detailed lane cascade: exact terminal tables (AD | MCI | CN), live ----
    st.markdown("##### Detailed lane cascade — AD | MCI | CN (live)")
    try:
        import subprocess as _sp, re as _re
        _out = _sp.run([PYBIN, "/tmp/lane_frame.py"], capture_output=True, text=True, timeout=25).stdout
        _out = _re.sub(r"\x1b\[[0-9;]*m", "", _out)
        _out = "\n".join(_out.split("\n")[4:])
        st.code(_out, language=None)
    except Exception as _e:
        st.caption(f"cascade detail unavailable: {_e}")
    h = s.get("host", {})
    st.caption(f"host: load {h.get('load_ratio')}×/{h.get('ncpu')} · RAM {h.get('ram_free')}/{h.get('ram_total')}G · swap {h.get('swap_used_g','?')}G · /tmp(tmpfs) {h.get('tmp_used_g','?')}G · disk {h.get('disk_free_g')}G · tckgen {h.get('tckgen')}")


def main() -> None:
    st.markdown("<h1 class='app-title'>Analysis of Structural connectomes</h1>", unsafe_allow_html=True)
    _view = st.sidebar.radio("View", ["📊 Analysis", "🔴 Live Pipeline Monitor"], index=0)
    if _view == "🔴 Live Pipeline Monitor":
        st.markdown("## 🧠 Connectome Pipeline — Live Monitor")
        render_pipeline_monitor()
        return
    if not ANALYSIS_ROOT.exists():
        st.error(f"Analysis root not found: {ANALYSIS_ROOT}")
        return
    inv = inventory(str(ANALYSIS_ROOT))
    with st.sidebar:
        with st.expander("Controls", expanded=False):
            st.write(f"Root: `{ANALYSIS_ROOT}`")
            st.write(f"Notebook: `{NOTEBOOK_PATH}`")
            allow_raw = st.checkbox(
                "Show subject-level/raw tables",
                value=False,
                help="Off by default to reduce accidental exposure of subject-level rows in a public dashboard.",
            )
            max_charts = st.slider("Default live charts per section", 1, 12, 4)
            hide_visual_outliers = st.checkbox(
                "Hide visual outliers in live charts",
                value=True,
                help="Display-only 1.5-IQR filtering. It does not modify the CSV tables or statistical results.",
            )
            if st.button("Refresh inventory", icon=":material/refresh:", width="stretch"):
                st.cache_data.clear()
                st.rerun()
            st.markdown("---")
            st.markdown("Security: app is read-only and should be served only behind nginx HTTPS + basic auth.")
    render_group_summary()
    render_data_provenance()
    st.markdown("---")
    selected_section = render_section_cards(inv)
    st.markdown("---")
    render_section(
        selected_section,
        inv,
        allow_raw=allow_raw,
        max_charts=max_charts,
        hide_visual_outliers=hide_visual_outliers,
    )


if __name__ == "__main__":
    main()
