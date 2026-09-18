import type { components } from "./generated/api-schema";

type GeneratedApiEnvelope = components["schemas"]["ApiEnvelope"];

export type HealthResponse = components["schemas"]["HealthResponse"];

export type ApiEnvelope<T> = Omit<
  GeneratedApiEnvelope,
  "data" | "warnings" | "provenance"
> & {
  data: T;
  warnings: string[];
  provenance: {
    generated_utc?: string;
    calculation?: string;
    sources?: Array<{
      path: string;
      available: boolean;
      mtime_ns?: number | null;
    }>;
  };
};

export interface MetricCatalog {
  available: boolean;
  source?: string;
  subject_n?: number;
  metrics: string[];
}

export interface Section {
  id: string;
  label: string;
  group: string;
  description: string;
  folders: string[];
  subject_table?: string | null;
  node_table?: string | null;
  artifact_count: number;
  metrics: MetricCatalog;
}

export interface Artifact {
  id: string;
  relative_path: string;
  name: string;
  suffix: string;
  size_bytes: number;
  mtime_ns: number;
  mime_type: string;
  kind: "table" | "json" | "note" | "figure";
}

export interface TablePayload {
  artifact: Artifact;
  offset: number;
  limit: number;
  total_rows: number;
  columns: string[];
  rows: Record<string, unknown>[];
}

export interface MetricPayload {
  source: string;
  metric: string;
  rows: Record<string, unknown>[];
}

export interface AnalysisMetric {
  id: string;
  label: string;
  description: string;
}

export interface AnalysisProfile {
  section: {
    id: string;
    label: string;
    group: string;
    description: string;
    folders: string[];
    subject_table?: string | null;
    node_table?: string | null;
  };
  subject: {
    available: boolean;
    source?: string | null;
    subject_n: number;
    metrics: AnalysisMetric[];
  };
  node: {
    available: boolean;
    source?: string | null;
    layout?: string;
    nodes: number;
    metrics: AnalysisMetric[];
  };
  statistical_contract: {
    group_order: string[];
    visual_filter: string;
    subject_inference: string;
    node_inference: string;
  };
}

export interface SubjectAnalysis {
  metric: string;
  description?: string | null;
  source: string;
  statistic_source: string;
  rows: Record<string, unknown>[];
  display_rows: Record<string, unknown>[];
  display_removed: Record<string, number>;
  descriptives: Record<string, unknown>[];
  pairwise: Record<string, unknown>[];
  display_filter: string;
}

export interface NodeCatalog {
  metric: string;
  source: string;
  nodes: Record<string, unknown>[];
}

export interface NodeRanking {
  metric: string;
  group_a: string;
  group_b: string;
  source: string;
  method: string;
  rows: Record<string, unknown>[];
}

export interface NodeValues {
  metric: string;
  node: number;
  node_name: string;
  source: string;
  rows: Record<string, unknown>[];
  display_rows: Record<string, unknown>[];
  display_removed: Record<string, number>;
  descriptives: Record<string, unknown>[];
  display_filter: string;
}

export interface DemographicsSummary {
  cohort: Record<string, unknown>[];
  age_rows: Record<string, unknown>[];
  age_descriptives: Record<string, unknown>[];
  age_pairwise: Record<string, unknown>[];
  clinical: Array<{
    metric: string;
    descriptives: Record<string, unknown>[];
    pairwise: Record<string, unknown>[];
  }>;
  definitions: Record<string, string>;
}

export interface ModelTaskCatalog {
  tasks: Array<{
    id: string;
    label: string;
    kind: "classification" | "regression";
    available: boolean;
    models: string[];
    targets: string[];
    roles: string[];
  }>;
  contract: Record<string, string>;
}

export interface ModelTaskData {
  id: string;
  label: string;
  kind: "classification" | "regression";
  performance: Record<string, unknown>[];
  status: Record<string, unknown>[];
  detail: Record<string, unknown>[];
  confusion: Record<string, unknown>[];
  predictions: Record<string, unknown>[];
  importance: Record<string, unknown>[];
  roc: Record<string, unknown>[];
  shap: Record<string, unknown>[];
  stage: Record<string, unknown>[];
}

export interface NetworkAnalysisCatalog {
  schemes: Array<{
    id: "functional" | "anatomical";
    label: string;
    families: Array<{
      id: "microstructure" | "graph" | "coupling";
      label: string;
      available: boolean;
      metrics: string[];
      subject_source: string;
      stats_source: string;
    }>;
  }>;
  mapping_contract: Record<string, string>;
}

export interface NetworkDistribution {
  scheme: string;
  family: string;
  family_label: string;
  metric: string;
  rows: Record<string, unknown>[];
  statistics: Record<string, unknown>[];
}

export interface NetworkAffectedness {
  scheme: string;
  families: string[];
  rows: Record<string, unknown>[];
  source: string;
}

export interface NetworkBlocks {
  scheme: string;
  metric: string;
  networks: string[];
  rows: Record<string, unknown>[];
  statistics: Record<string, unknown>[];
}

export interface CouplingAalCatalog {
  topology_metrics: string[];
  microstructure_metrics: string[];
  value_metrics: string[];
  definition: Record<string, string>;
}

export interface CouplingAalRanking {
  topology_metric: string;
  microstructure_metric: string;
  value_metric: string;
  group_a: string;
  group_b: string;
  rows: Record<string, unknown>[];
}

export interface NoveltyCitation {
  title: string;
  authors?: string;
  year?: number;
  venue?: string;
  doi?: string;
  url?: string;
  tier?: string;
  cited_by_count?: number | null;
}

export interface NoveltyCard {
  theme: string;
  finding_ids: string[];
  novelty_type: string;
  confidence: string;
  our_finding: string;
  prior_literature: string;
  frontier_context: string;
  novelty_claim: string;
  caveat: string;
  citations: NoveltyCitation[];
}

export interface NovelFindingsSummary {
  cards: NoveltyCard[];
  catalog: Record<string, unknown>[];
  family_counts: Record<string, unknown>[];
  confidence_counts: Record<string, number>;
  report_available: boolean;
  contract: Record<string, string>;
}

export interface ExceptionSpecificity {
  status: string;
  tier_delta?: Record<string, unknown>[];
  tier_delta_summary?: Array<{
    contrast: string;
    mean_gain_vs_range: number;
    pop_ups: number;
    drop_outs: number;
    cells: number;
  }>;
  architecture?: Record<string, unknown>[];
  ml_ladder?: Record<string, unknown>[];
  ml_predictions?: Array<{
    subject_id: string;
    target: string;
    model: string;
    y_true: number;
    y_oof: number;
  }>;
  ml_permutation?: Record<
    string,
    { model: string; metric: string; observed: number; perm_p: number; null_mean: number; n_perm: number }
  >;
  ml_cohort?: {
    n: number;
    mmse_n: number;
    cdr_n: number;
    cdr_balance: number[];
    median_lag_days: number;
  };
  ml_status?: string;
  ml_r2?: Array<{ target: string; rung: string; metric: string; value: number; sd: number; n: number; n_features: number }>;
  ml_shap?: Array<{ target: string; feature: string; mean_abs_shap: number; family: string }>;
  ml_shap_family_share?: Array<{ target: string; family: string; pct: number }>;
  ml_ice?: Array<{ target: string; feature: string; grid: number; pdp: number; ice_lo: number; ice_hi: number }>;
  mmse_bucket_results?: Array<{ rung: string; model: string; macro_auc: number; macro_pr_auc: number; bal_acc: number; n_features: number }>;
  mmse_bucket_definition?: Array<{ class: number; label: string; range: string; n: number; CN: number; MCI: number; AD: number }>;
  mmse_bucket_classwise?: Array<{ cls: number; label: string; support: number; auc: number; pr_auc: number; baseline_pr: number; model: string; rung: string }>;
  mmse_bucket_curves?: Array<{ kind: string; cls: number; label: string; x: number; y: number }>;
  mmse_bucket_shap?: Array<{ cls: number; label: string; feature: string; mean_abs_shap: number; family: string }>;
  feature_dictionary?: Array<{ feature: string; family: string; construction: string }>;
  pipeline_steps?: Array<{ order: number; step: string; performed: string; detail: string }>;
  preprocessing_audit?: Array<Record<string, unknown>>;
  selection_experiment?: Array<{ target: string; selection: string; metric: string; value: number; sd: number }>;
  shap_overall?: Array<{ target: string; target_label: string; short: string; feature: string; family: string; share_pct: number; pos_pct: number; n_pos: number; n_neg: number; subject: string; group: string; shap: number; feature_pctl: number }>;
  shap_family_group?: Array<{ target: string; cls: number; cls_label: string; group: string; family: string; share_pct: number }>;
  shap_class_group?: Array<{ target: string; cls: number; cls_label: string; group: string; n: number; feature: string; family: string; mean_abs_shap: number; share_pct: number; share_sd: number }>;
  shap_class_group_dir?: Array<{ target: string; cls: number; cls_label: string; group: string; feature: string; mean_signed_shap: number }>;
  pdp_range_family?: Array<{ target: string; feature: string; range_label: string; range_order: number; cls: number; cls_label: string; grid: number; prob: number; missing_pct: number }>;
  pdp_class_group?: Array<{ target: string; feature: string; cls: number; cls_label: string; group: string; grid: number; prob: number }>;
  final_model_table?: Array<{ target: string; row_order: number; row_label: string; col_order: number; col_label: string; value: string }>;
  final_model_metrics?: Array<{ target: string; scope: string; cls: number; cls_label: string; auc: number | null; pr_auc: number | null; ci_lo: number | null; ci_hi: number | null; n_pos: number; n_neg: number; reason: string }>;
  final_model_classwise?: Array<{ target: string; cls: number; label: string; auc: number | null; pr_auc: number | null; baseline_pr: number; support: number }>;
  final_model_curves?: Array<{ target: string; kind: string; cls: number; label: string; x: number; y: number }>;
  auc_class_diaggroup?: Array<{ target: string; cls: number; cls_label: string; auc_All: number | null; n_All: string; auc_CN: number | null; n_CN: string; auc_MCI: number | null; n_MCI: string; auc_AD: number | null; n_AD: string }>;
  improvement_mmse?: Array<{ metric: string; baseline: number; residualised: number }>;
  improvement_check?: Array<{ target: string; config: string; macro_auc: number; bal_acc: number; auc_0: number; auc_05: number; auc_ge1: number }>;
  cdr_threeclass?: Array<{ cls: string; n: number; auc: number }>;
  cdr_level_counts?: Array<{ level: number; CN: number; MCI: number; AD: number; total: number }>;
  cdr_curves?: Array<{ kind: string; cls: number; label: string; x: number; y: number }>;
  cdr_classwise?: Array<{ cls: number; label: string; support: number; auc: number; pr_auc: number; baseline_pr: number }>;
  shap_beeswarm_class?: Array<{ target: string; cls: number; cls_label: string; feature: string; subject: string; group: string; shap: number; feature_pctl: number }>;
  inference_summary?: {
    per_class_top: Array<{ target: string; cls: string; top: string; top_family: string }>;
    family_share_by_group: Array<{ target: string; group: string; family: string; pct: number }>;
    agreement: string[];
  };
}

export interface NetworkMeasureRankedCell {
  network: string;
  family: string;
  measure: string;
  available: boolean;
  rank?: number;
  mean_CN?: number;
  mean_MCI?: number;
  mean_AD?: number;
  cliffs_delta?: number;
  kw_p?: number;
  bm_q?: number;
  direction?: "up" | "down";
  significant?: boolean;
}

export interface NetworkMeasureRankedGroup {
  key: string;
  label: string;
  description: string;
  measures: string[];
  cells: NetworkMeasureRankedCell[];
}

export interface NetworkMeasureRanked {
  status: string;
  n_subjects: number;
  contrast: string;
  contrast_label: string;
  contrasts: Array<{ key: string; label: string }>;
  measures: Array<{ key: string; label: string; family: string }>;
  networks: string[];
  cells: NetworkMeasureRankedCell[];
  groups: NetworkMeasureRankedGroup[];
  note: string;
}

export interface TractSummary {
  Panel: string;
  n_subjects: number;
  n_edges: number;
  q1: number;
  median: number;
  q3: number;
  iqr: number;
  lower_fence: number;
  upper_fence: number;
  low_outlier_n: number;
  high_outlier_n: number;
  outlier_n: number;
  outlier_pct: number;
  min: number;
  max: number;
}

export interface TractRanges {
  population_median_mm: number;
  short_max_mm: number;
  medium_max_mm: number;
  source_group: string;
  rule: string;
  ranges: Array<{
    code: "SR" | "MR" | "LR";
    label: string;
  }>;
}

export interface TractDistribution {
  panel: "Overall" | "CN" | "MCI" | "AD";
  sample_seed: number;
  display_sample_n: number;
  values: number[];
  full_summary: TractSummary;
}

export interface ExceptionConfig {
  measures: Record<
    string,
    {
      label: string;
      short_label: string;
      status: string;
      interpretation: string;
    }
  >;
  default_measure: string;
  thresholds: Record<string, unknown>;
  rule: string;
  unit_of_inference: string;
}

export interface ExceptionSummary {
  measure: string;
  measure_definition: Record<string, string>;
  thresholds: Record<string, unknown>;
  rows: Record<string, unknown>[];
}

export interface ExceptionInference {
  measure: string;
  table_scope: string;
  rows: Record<string, unknown>[];
  interaction_rows: Record<string, unknown>[];
  plot_rows: Array<{
    Subject: string;
    Group: string;
    "Edge class": string;
    "Subject median": number;
  }>;
  metadata: Record<string, number>;
  limitations: string[];
}

export interface ExceptionSubject {
  subject_id: string;
  group: string;
  n_edges_valid?: number;
  n_exceptions?: number;
  lr_edge_count?: number;
  lr_exception_count?: number;
  lr_exception_pct?: number;
  lr_nonexception_measure_median?: number;
  lr_exception_measure_median?: number;
}

export interface ExceptionExample {
  measure: string;
  metadata: {
    subject_id: string;
    group: string;
    candidate_n: number;
    exception_n: number;
  };
  points: Array<{
    "Length (mm)": number;
    Measure: number;
    Threshold: number;
    Class: string;
  }>;
  adaptive_bins: Array<{
    "Minimum length": number;
    "Maximum length": number;
    Median: number;
    Threshold: number;
  }>;
}

export interface FeatureFamilies {
  subjects: number;
  candidate_features: number;
  families: Record<string, unknown>[];
  metric_inventory: Record<string, unknown>[];
}

export interface FeatureEda {
  subjects: number;
  offset: number;
  limit: number;
  total_rows: number;
  columns: string[];
  rows: Record<string, unknown>[];
}

export interface NetworkMapping {
  status: string;
  functional: Record<string, unknown>[];
  anatomical: Record<string, unknown>[];
  mapping: Record<string, unknown>[];
}

export interface ConnectomeSubject {
  subject_id: string;
  group: string;
  "Image ID": number;
  sex?: string;
  age?: number;
  phase?: string;
  density?: number;
  is_dense?: boolean;
  n_connectome_types?: number;
  available_weights: string[];
}

export interface MatrixSummary {
  metadata: Record<string, unknown>;
  summary: Record<string, number | boolean | null>;
}

export interface MatrixPayload {
  metadata: Record<string, unknown>;
  labels: Array<{
    matrix_idx: number;
    node_name?: string;
    label: string;
    atlas_value?: number | null;
  }>;
  matrix: Array<Array<number | null>>;
}

export interface PipelineRelease {
  summary: Record<string, unknown> | null;
  group_rows: Record<string, unknown>[];
  subject_rows: Record<string, unknown>[];
  ledger_summary: Record<string, unknown>[];
  topology: Record<string, unknown> | null;
  sources: Record<string, unknown>[];
}

export interface PipelineLegacy {
  status: Record<string, unknown> | null;
  density: Record<string, unknown> | null;
  manifest_summary: Record<string, unknown> | null;
  sources: Record<string, unknown>[];
}
