from __future__ import annotations

import pandas as pd

from connectome_analysis.analysis_config import AnalysisPaths
from connectome_analysis.analysis_plots import save_dataframe, write_inference_markdown


def run_clinical_covariate_review(paths: AnalysisPaths) -> dict:
    out_dir = paths.section_dir("14", "clinical_sensitivity")
    out_dir.mkdir(parents=True, exist_ok=True)
    ancova_path = paths.legacy_analysis_dir / "analysis4_clinical_revised" / "global_group_ancova.csv"
    ancova = pd.read_csv(ancova_path) if ancova_path.exists() else pd.DataFrame()
    if not ancova.empty:
        ancova = ancova.sort_values("p_diag", na_position="last")
        save_dataframe(ancova, out_dir / "global_group_ancova_review.csv")
    sensitivity_rows = []
    for summary_file in (paths.legacy_analysis_dir / "analysis4_clinical_revised").glob("*_summary.txt"):
        sensitivity_rows.append({"file": summary_file.name, "path": str(summary_file)})
    sensitivity = pd.DataFrame(sensitivity_rows)
    save_dataframe(sensitivity, out_dir / "clinical_summary_files.csv")
    write_inference_markdown(
        [
            "This section preserves the age- and covariate-aware analyses from the legacy notebook and surfaces them in one place.",
            "The ANCOVA-style tables are treated as sensitivity analyses alongside the permutation-adjusted primary statistics used elsewhere in notebook C.",
        ],
        out_dir / "clinical_inference.md",
    )
    return {"ancova": ancova, "sensitivity": sensitivity, "out_dir": out_dir}

