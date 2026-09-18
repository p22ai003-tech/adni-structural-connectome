from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Section:
    id: str
    label: str
    group: str
    description: str
    folders: tuple[str, ...]
    subject_table: str | None = None
    node_table: str | None = None

    def to_dict(self) -> dict:
        value = asdict(self)
        value["folders"] = list(self.folders)
        return value


SECTIONS = (
    Section(
        "demographics",
        "Demographics",
        "General",
        "Group counts, demographics, QC, and clinical score summaries.",
        ("00_master", "01_qc", "02_demographics"),
    ),
    Section(
        "novel-findings",
        "Novel Findings",
        "General",
        "PPT-style synthesis of positive exploratory findings.",
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
    Section(
        "global-dti",
        "Global DTI",
        "Graph Measures",
        "Global FA, MD, RD and AxD microstructure summaries.",
        ("03_global_microstructure_live", "03_global_rd"),
        "live_global_microstructure_subject_table.csv",
    ),
    Section(
        "local-roi-dti",
        "Local / ROI DTI",
        "Graph Measures",
        "Node-wise diffusion microstructure and ROI inference.",
        ("04_node_microstructure_live", "04_local_rd"),
        node_table="live_node_microstructure_long.csv",
    ),
    Section(
        "global-graph",
        "Global Graph",
        "Graph Measures",
        "Global strength, density, efficiency and path-length summaries.",
        ("06_global_graph_live", "06_global_graph"),
        "live_global_graph_subject_table.csv",
    ),
    Section(
        "sc-matrix-viewer",
        "SC Matrix Viewer",
        "Tractography",
        "Single-subject structural-connectome matrix QC and summaries.",
        (
            "00_master",
            "01_qc",
            "03_global_microstructure_live",
            "06_global_graph_live",
            "07_node_graph_live",
            "09_coupling_live",
            "11_brain_age_live",
            "12_length_delay",
            "13_delay",
            "17_edr_exceptions",
            "18_ml_diagnostics",
        ),
        "live_global_graph_subject_table.csv",
    ),
    Section(
        "node-metrics",
        "Node Metrics",
        "Graph Measures",
        "Node-wise graph metrics and centrality summaries.",
        ("07_node_graph_live", "07_node_metrics", "08_centrality"),
        node_table="live_node_graph_long.csv",
    ),
    Section(
        "coupling",
        "Coupling",
        "Functional Measures",
        "Structural topology-microstructure coupling; not functional connectivity.",
        ("09_coupling_live", "09_coupling"),
        "live_subject_level_coupling.csv",
    ),
    Section(
        "coupling-aal",
        "Coupling-AAL",
        "Functional Measures",
        "Regional AAL-level structure-microstructure coupling rankings.",
        (
            "09_coupling_live",
            "04_node_microstructure_live",
            "07_node_graph_live",
        ),
    ),
    Section(
        "brain-age",
        "Brain Age",
        "Advanced",
        "Structural brain-age and brain-age-gap summaries.",
        ("11_brain_age_live", "11_brain_age"),
        "live_brain_age_predictions.csv",
    ),
    Section(
        "lr-sr",
        "LR / SR",
        "Tractography",
        "Short-range versus long-range structural pathway analyses.",
        ("12_length_delay",),
        "lr_sr_subject_level_fd_sum_len_mean.csv",
    ),
    Section(
        "lr-sr-analysis",
        "LR-SR Analysis",
        "Tractography",
        "Compact tract-length, exception, feature and modelling work-along.",
        ("12_length_delay", "18_ml_diagnostics"),
        "lr_sr_subject_level_fd_sum_len_mean.csv",
    ),
    Section(
        "edr-exceptions",
        "EDR Exceptions",
        "Tractography",
        "Edge-distance relationship exceptions and compensation signatures.",
        ("17_edr_exceptions",),
        "edr_exception_subject_level_fd_sum_len_mean.csv",
    ),
    Section(
        "ml-diagnostics",
        "ML Diagnostics",
        "Advanced",
        "Cross-validated structural prediction models and feature signatures.",
        ("18_ml_diagnostics",),
    ),
    Section(
        "delay",
        "Delay",
        "Functional Measures",
        "Length-derived delay-proxy analyses.",
        ("13_delay",),
        "delay_subject_level_fd_sum_len_mean.csv",
    ),
    Section(
        "advanced",
        "Advanced",
        "Advanced",
        "EDR residuals, compensation and long-range vulnerability.",
        ("15_advanced_structural",),
        "edr_subject_level_fd_sum_len_mean.csv",
    ),
    Section(
        "network-analysis",
        "Network Analysis",
        "Network Measures",
        "Network/system aggregation and within/between connectivity.",
        ("19_network_analysis",),
    ),
    Section(
        "functional-pending",
        "Functional Pending",
        "Advanced",
        "Placeholder until functional matrices are supplied.",
        ("16_functional_placeholder",),
    ),
)

SECTION_BY_ID = {section.id: section for section in SECTIONS}
SECTION_BY_LABEL = {section.label: section for section in SECTIONS}


def section_or_raise(section_id: str) -> Section:
    try:
        return SECTION_BY_ID[section_id]
    except KeyError as error:
        raise KeyError(f"unknown section: {section_id}") from error
