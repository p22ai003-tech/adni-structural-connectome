"""The analysis DAG, declared as data.

Every stage names what it needs and what it produces. That is the whole point:
the pipeline used to be a straight-line script in which the order was implicit
and two real defects hid in it.

  * ``mediation`` reads ``11_brain_age_live/live_brain_age_predictions.csv`` but
    ran *before* ``live_brain_age`` produced it. The read is guarded by
    ``if ba.exists()``, so on a clean tree the brain-age mediator was silently
    dropped and the analysis still reported success. It only ever worked here
    because an earlier run had left the file on disk.

  * ``build_shap_overall`` overwrites four tables that ``build_ml_explain_v2``
    and ``build_explain_v3`` also write. Whichever runs last wins. The correct
    order -- shap_overall last, because it is the seed-averaged rewrite computed
    over all 105 features rather than a top-k truncation -- lived only in shell
    history.

Declaring ``needs`` and ``produces`` makes both structural rather than
remembered: the runner refuses to start a stage whose declared inputs are
absent, and it topologically sorts rather than trusting the order of this file.

Every path in ``produces`` is relative to the analysis root and was verified
against the tree this project actually produced. They are the resume contract:
a wrong path here would make the runner re-run finished work, or worse, declare
a stage complete that never ran.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["Stage", "STAGES", "by_name", "topo_order", "with_dependencies", "GROUPS"]

GROUPS = ("cohort", "measures", "inference", "ml", "build")


@dataclass(frozen=True)
class Stage:
    name: str
    group: str
    summary: str
    needs: tuple[str, ...] = ()
    produces: tuple[str, ...] = ()
    # Stages whose return value a later stage consumes in memory. The runner
    # memoises these and re-runs the producer if a resumed run needs one.
    provides_value: bool = False
    optional: bool = False
    # A build_*.py script run as a subprocess rather than an in-process call.
    script: str | None = None


# --------------------------------------------------------------------------
# Group 1: cohort assembly and QC. Everything else depends on master_cohort.
# --------------------------------------------------------------------------
_COHORT = [
    Stage("analysis_snapshot", "cohort",
          "Freeze the input snapshot (provisional or final).",
          produces=("00_master/analysis_snapshot.csv",)),
    Stage("master_cohort", "cohort",
          "Build the master cohort table; the spine of every later stage.",
          needs=("analysis_snapshot",),
          produces=("00_master/master_cohort.csv",),
          provides_value=True),
    Stage("qc_snapshot", "cohort",
          "Per-subject QC status.",
          needs=("master_cohort",),
          produces=("00_master/qc_snapshot.csv",)),
    Stage("data_completeness", "cohort",
          "Which modalities and derivatives exist per subject.",
          needs=("master_cohort",),
          produces=("01_qc/data_completeness.csv",
                    "01_qc/data_completeness_subject_level.csv")),
    Stage("demographics", "cohort",
          "Group counts, age and sex distributions, clinical summary.",
          needs=("master_cohort",),
          produces=("02_demographics/group_counts.csv",
                    "02_demographics/clinical_summary.csv")),
    Stage("connectome_qc", "cohort",
          "Matrix-level QC: availability, density, dimension checks.",
          needs=("master_cohort",),
          produces=("05_connectome_qc/subject_connectome_availability.csv",
                    "05_connectome_qc/connectome_availability_by_group.csv")),
    Stage("functional_placeholder", "cohort",
          "Note that the functional arm is a mapping, not an fMRI measurement.",
          needs=("master_cohort",),
          produces=("19_network_analysis/functional/README_functional.txt",)),
]

# --------------------------------------------------------------------------
# Group 2: live per-subject measures, read straight from the connectome tree.
# --------------------------------------------------------------------------
_MEASURES = [
    Stage("live_global_microstructure", "measures",
          "Whole-brain FA / MD / RD / AxD per subject.",
          needs=("master_cohort",),
          produces=("03_global_microstructure_live/live_global_microstructure_subject_table.csv",
                    "03_global_microstructure_live/live_global_microstructure_summary.csv"),
          provides_value=True),
    Stage("live_node_microstructure", "measures",
          "Node-wise microstructure per subject.",
          needs=("master_cohort",),
          produces=("04_node_microstructure_live/live_node_microstructure_long.csv",
                    "04_node_microstructure_live/live_node_microstructure_summary.csv"),
          provides_value=True),
    Stage("live_multimetric", "measures",
          "Multimetric overview combining global and node-wise microstructure.",
          needs=("live_global_microstructure", "live_node_microstructure"),
          produces=("05_multimetric_live/live_multimetric_global_summary.csv",
                    "05_multimetric_live/live_multimetric_node_summary.csv")),
    Stage("live_global_graph", "measures",
          "Global graph metrics per subject.",
          needs=("master_cohort",),
          produces=("06_global_graph_live/live_global_graph_subject_table.csv",
                    "06_global_graph_live/live_global_graph_summary.csv")),
    Stage("live_node_graph", "measures",
          "Node-wise graph metrics per subject.",
          needs=("master_cohort",),
          produces=("07_node_graph_live/live_node_graph_long.csv",
                    "07_node_graph_live/live_node_graph_summary.csv"),
          provides_value=True),
    Stage("live_coupling", "measures",
          "Structure-microstructure coupling.",
          needs=("live_node_graph", "live_node_microstructure"),
          produces=("09_coupling_live/live_subject_level_coupling.csv",
                    "09_coupling_live/live_coupling_summary.csv")),
    Stage("network_analysis", "measures",
          "AAL3 -> Yeo network tables; feeds the whole network family.",
          needs=("master_cohort", "live_node_microstructure", "live_node_graph"),
          produces=("19_network_analysis/network_mapping_used.csv",
                    "19_network_analysis/functional/network_microstructure_stats.csv",
                    "19_network_analysis/functional/network_microstructure_subject.csv",
                    "19_network_analysis/functional/network_graph_stats.csv")),
]

# --------------------------------------------------------------------------
# Group 3: inference. edr_exceptions is the thesis's central stage.
# --------------------------------------------------------------------------
_INFERENCE = [
    Stage("edgewise_fd_sum", "inference",
          "Edge-wise group inference with permutation control.",
          needs=("master_cohort",),
          produces=("10_edgewise_fd_sum/edges_fd_sum_omnibus.csv",
                    "10_edgewise_fd_sum/edges_fd_sum_CNvsAD.csv")),
    Stage("live_brain_age", "inference",
          "Brain-age model and the per-subject brain-age gap (BAG).",
          needs=("master_cohort",),
          produces=("11_brain_age_live/live_brain_age_predictions.csv",
                    "11_brain_age_live/live_brain_age_model_performance.csv")),
    Stage("lr_sr", "inference",
          "Long-range / short-range split on the CN length distribution.",
          needs=("master_cohort",),
          produces=("12_length_delay/lr_sr_summary.csv",)),
    # NOTE: this does NOT depend on lr_sr. It derives its own thresholds from the
    # pooled CN upper triangle (Q1 = 78.6856 mm, Q3 = 171.021 mm), which is a
    # different split from the 33/67 tertiles that lr_sr uses. See the SR/LR note
    # in configs/analysis.yaml -- the two definitions coexist on purpose and are
    # reported under different column names.
    Stage("edr_exceptions", "inference",
          "EDR exceptions: within-subject, per-length-bin, mean + k SD.",
          needs=("master_cohort",),
          produces=("17_edr_exceptions/edr_exception_thresholds.csv",
                    "17_edr_exceptions/edr_exception_subject_level_fd_sum_len_mean.csv",
                    "17_edr_exceptions/edr_exception_edge_level_fd_sum_len_mean.csv",
                    "17_edr_exceptions/edr_exception_node_level_fd_sum_len_mean.csv")),
    Stage("delay", "inference",
          "Conduction-delay proxy analysis.",
          needs=("master_cohort",),
          produces=("13_delay/delay_summary.csv",)),
    Stage("clinical", "inference",
          "Clinical covariate review (MMSE / CDR / APOE).",
          needs=("master_cohort",),
          produces=("14_clinical_sensitivity/clinical_summary_files.csv",)),
    Stage("advanced_structural", "inference",
          "Structural innovation analyses, including the EDR model fit.",
          needs=("master_cohort",),
          produces=("15_advanced_structural/advanced_structural_summary.csv",
                    "15_advanced_structural/edr_model_parameters.csv")),
    Stage("findings_catalog", "inference",
          "Machine-readable catalogue of every statistical finding.",
          needs=("network_analysis", "edr_exceptions"),
          produces=("20_findings/findings_catalog.csv",)),
    # Ordered AFTER live_brain_age: it reads the BAG predictions, and its read is
    # guarded by exists(), so running it early degrades silently.
    Stage("mediation", "inference",
          "Mediation of group -> outcome through network measures and BAG.",
          needs=("network_analysis", "live_brain_age"),
          produces=("21_mediation/mediation_results.csv",)),
    Stage("literature_retrieval", "inference",
          "OpenAlex retrieval for the findings catalogue.",
          needs=("findings_catalog",),
          produces=("20_findings/literature_records.csv",),
          optional=True),
]

# --------------------------------------------------------------------------
# Group 4: in-process ML diagnostics.
# --------------------------------------------------------------------------
_ML = [
    Stage("ml_diagnostics", "ml",
          "Diagnostic ML sweep over the connectome feature families.",
          needs=("master_cohort", "edr_exceptions"),
          produces=("18_ml_diagnostics/cdr_classification_model_performance.csv",)),
    Stage("clinical_outcome_search", "ml",
          "Model search against clinical outcomes on the dense cohort.",
          needs=("ml_diagnostics",)),
]

# --------------------------------------------------------------------------
# Group 5: the build_*.py layer. Previously run by hand, then copied by hand.
# Order matters: build_shap_overall must land last (see the module docstring).
# --------------------------------------------------------------------------
_BUILD = [
    Stage("build_range_restricted_networks", "build",
          "Range-restricted and exception network tables.",
          needs=("edr_exceptions", "network_analysis"),
          script="build_range_restricted_networks.py",
          produces=("19_network_analysis/functional/network_range_restricted_stats.csv",
                    "19_network_analysis/functional/network_range_restricted_subject.csv",
                    "19_network_analysis/functional/network_exception_measures_stats.csv",
                    "19_network_analysis/functional/network_exception_measures_subject.csv",
                    "19_network_analysis/functional/network_exception_range_stats.csv")),
    Stage("build_exception_specificity", "build",
          "Exception tier deltas and per-subject exception architecture.",
          needs=("build_range_restricted_networks",),
          script="build_exception_specificity.py",
          produces=("20_exception_specificity/exception_tier_delta.csv",
                    "20_exception_specificity/exception_architecture_stats.csv",
                    "20_exception_specificity/exception_architecture_subject.csv")),
    Stage("build_simple_ml", "build",
          "The ML feature matrix, targets and the model ladder.",
          needs=("build_exception_specificity",),
          script="build_simple_ml.py",
          produces=("20_exception_specificity/ml_feature_matrix.csv",
                    "20_exception_specificity/ml_targets.csv",
                    "20_exception_specificity/ml_ladder_results.csv",
                    "20_exception_specificity/ml_predictions.csv",
                    "20_exception_specificity/ml_missingness.csv")),
    Stage("build_model_interpretation", "build",
          "R-squared summary, per-target SHAP values, beeswarm sources and ICE curves.",
          needs=("build_simple_ml",), script="build_model_interpretation.py",
          produces=("20_exception_specificity/ml_r2_summary.csv",
                    "20_exception_specificity/ml_shap_values_mmse.csv",
                    "20_exception_specificity/ml_shap_values_cdr_bin.csv",
                    "20_exception_specificity/ml_shap_beeswarm_mmse.csv",
                    "20_exception_specificity/ml_shap_beeswarm_cdr_bin.csv",
                    "20_exception_specificity/ml_ice_mmse.csv",
                    "20_exception_specificity/ml_ice_cdr_bin.csv")),
    Stage("build_mmse_buckets", "build",
          "MMSE tertile bands and their class-wise performance.",
          needs=("build_simple_ml",), script="build_mmse_buckets.py",
          produces=("20_exception_specificity/mmse_bucket_results.csv",
                    "20_exception_specificity/mmse_bucket_definition.csv",
                    "20_exception_specificity/mmse_bucket_classwise.csv",
                    "20_exception_specificity/mmse_bucket_curves.csv",
                    "20_exception_specificity/mmse_bucket_shap_classwise.csv")),
    Stage("build_final_model_metrics", "build",
          "The accepted final model: AUC / PR-AUC, overall and class-wise.",
          needs=("build_simple_ml",), script="build_final_model_metrics.py",
          produces=("20_exception_specificity/final_model_table.csv",
                    "20_exception_specificity/final_model_metrics.csv",
                    "20_exception_specificity/final_model_classwise.csv",
                    "20_exception_specificity/final_model_curves.csv")),
    Stage("build_pdp_range_family", "build",
          "Partial dependence across the SR / LR / SR-exc / LR-exc family.",
          needs=("build_simple_ml",), script="build_pdp_range_family.py",
          produces=("20_exception_specificity/ml_pdp_range_family.csv",)),
    # This also writes ml_shap_class_group.csv, ml_shap_class_group_dir.csv and
    # ml_pdp_class_group.csv, but those are NOT declared here: build_shap_overall
    # rewrites all three, so they are declared there, on the stage that has the
    # last word. Declaring them in both places would let resume call this stage
    # current on the strength of files a later stage had replaced.
    Stage("build_ml_explain_v2", "build",
          "SHAP and PDP by class and diagnostic group; the preprocessing audit.",
          needs=("build_simple_ml",), script="build_ml_explain_v2.py",
          produces=("20_exception_specificity/ml_pipeline_steps.csv",
                    "20_exception_specificity/ml_preprocessing_audit.csv",
                    "20_exception_specificity/ml_feature_dictionary.csv",
                    "20_exception_specificity/ml_selection_experiment.csv")),
    # ml_shap_beeswarm_class.csv is likewise declared on build_shap_overall.
    Stage("build_explain_v3", "build",
          "CDR three-level curves and the class beeswarm source table.",
          needs=("build_simple_ml",), script="build_explain_v3.py",
          produces=("20_exception_specificity/cdr_classwise.csv",
                    "20_exception_specificity/cdr_curves.csv")),
    # LAST on purpose: it rewrites ml_shap_class_group.csv, ml_shap_class_group_dir.csv,
    # ml_pdp_class_group.csv and ml_shap_beeswarm_class.csv with the seed-averaged
    # version computed over all 105 features. Running it earlier leaves the
    # single-seed, top-k-truncated tables in place, which overstate the shares.
    Stage("build_shap_overall", "build",
          "Seed-averaged SHAP over all features; supersedes the earlier tables.",
          needs=("build_ml_explain_v2", "build_explain_v3"),
          script="build_shap_overall.py",
          produces=("20_exception_specificity/ml_shap_class_group.csv",
                    "20_exception_specificity/ml_shap_class_group_dir.csv",
                    "20_exception_specificity/ml_shap_overall_beeswarm.csv",
                    "20_exception_specificity/ml_shap_family_group.csv",
                    "20_exception_specificity/ml_shap_beeswarm_class.csv",
                    "20_exception_specificity/ml_pdp_class_group.csv")),
]

STAGES: tuple[Stage, ...] = tuple(_COHORT + _MEASURES + _INFERENCE + _ML + _BUILD)


def by_name(name: str) -> Stage:
    for s in STAGES:
        if s.name == name:
            return s
    raise KeyError(f"unknown stage {name!r}. Run with --list to see all {len(STAGES)}.")


def topo_order(names: list[str] | None = None) -> list[Stage]:
    """Topologically sort the requested stages, pulling in nothing extra.

    Raises on a cycle rather than silently dropping a stage, because a dropped
    stage would leave whatever an earlier run wrote in place and look like success.
    """
    want = {s.name for s in STAGES} if names is None else set(names)
    unknown = want - {s.name for s in STAGES}
    if unknown:
        raise KeyError(f"unknown stage(s): {', '.join(sorted(unknown))}")

    ordered: list[Stage] = []
    placed: set[str] = set()
    remaining = [s for s in STAGES if s.name in want]
    while remaining:
        progressed = False
        for s in list(remaining):
            if all(n in placed or n not in want for n in s.needs):
                ordered.append(s)
                placed.add(s.name)
                remaining.remove(s)
                progressed = True
        if not progressed:
            stuck = ", ".join(s.name for s in remaining)
            raise ValueError(f"cycle or unsatisfiable dependency among: {stuck}")
    return ordered


def with_dependencies(names: list[str]) -> list[str]:
    """Close a stage selection over its transitive ``needs``."""
    out: set[str] = set()
    stack = list(names)
    while stack:
        n = stack.pop()
        if n in out:
            continue
        out.add(n)
        stack.extend(by_name(n).needs)
    return sorted(out)
