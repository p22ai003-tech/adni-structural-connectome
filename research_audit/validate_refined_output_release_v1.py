#!/usr/bin/env python3
"""Independent validation of the refined output-only release."""

from __future__ import annotations

import hashlib
import json
import math
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image
import numpy as np
import pandas as pd

import build_multivariate_stage_vector_v1 as multivar


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "refined_output_release_v1"
DATA = OUT / "source_data"
FIG = OUT / "figures"
STAGE = ROOT / "outputs" / "stage_specific_diffusivity_v2"
SOUNDNESS = ROOT / "outputs" / "primary_inference_soundness_v2"
EXTENSION = ROOT / "outputs" / "manuscript_multiscale_extension_v2"
ML = ROOT / "outputs" / "network_resolved_site_validation_v2"
CLAIMS = ROOT / "outputs" / "all_evidence_inventory_v2" / "claim_status_matrix.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def frames_close(left: pd.DataFrame, right: pd.DataFrame, keys: list[str]) -> bool:
    left = left.sort_values(keys).reset_index(drop=True)
    right = right.sort_values(keys).reset_index(drop=True)
    if list(left.columns) != list(right.columns) or left.shape != right.shape:
        return False
    for column in left.columns:
        if pd.api.types.is_numeric_dtype(left[column]) and pd.api.types.is_numeric_dtype(right[column]):
            if not np.allclose(
                left[column].to_numpy(float),
                right[column].to_numpy(float),
                rtol=1e-12,
                atol=1e-14,
                equal_nan=True,
            ):
                return False
        else:
            if not left[column].fillna("<NA>").astype(str).equals(
                right[column].fillna("<NA>").astype(str)
            ):
                return False
    return True


def ellipse_summary(samples: np.ndarray) -> dict[str, float]:
    covariance = np.cov(samples.T)
    values, vectors = np.linalg.eigh(covariance)
    order = values.argsort()[::-1]
    values = values[order]
    vectors = vectors[:, order]
    scale95 = math.sqrt(5.991464547)
    return {
        "ellipse_width_95": float(2.0 * scale95 * math.sqrt(max(values[0], 0.0))),
        "ellipse_height_95": float(2.0 * scale95 * math.sqrt(max(values[1], 0.0))),
        "ellipse_angle_degrees": float(
            math.degrees(math.atan2(vectors[1, 0], vectors[0, 0]))
        ),
    }


def main() -> None:
    build_validation = json.loads((OUT / "validation.json").read_text(encoding="utf-8"))

    tensor_export = pd.read_csv(DATA / "01_whole_brain_tensor_context.csv")
    tensor_upstream = pd.read_csv(EXTENSION / "global_microstructure_site_tests.csv")
    tensor_upstream = tensor_upstream[
        tensor_upstream["data_validity"].eq("hard_range_valid")
        & tensor_upstream["scope"].eq("full_site_fe")
    ].copy()
    tensor_upstream["holm_pass"] = tensor_upstream[
        "holm_p_12_within_validity_scope"
    ].le(0.05)
    tensor_upstream = tensor_upstream[tensor_export.columns]

    primary_export = pd.read_csv(DATA / "02_adjacent_stage_effects.csv")
    primary_locked = pd.read_csv(STAGE / "primary_contrasts.csv")
    primary_audit = pd.read_csv(SOUNDNESS / "primary_small_cluster_inference.csv")
    audit_columns = [
        "endpoint",
        "scope",
        "n",
        "n_sites",
        "beta",
        "cr1_se",
        "p_cr1_t_gminus1",
        "p_restricted_wild_cluster_bootstrap_t",
        "wild_bootstrap_valid_iterations",
        "holm_p_cr1_t_gminus1",
        "holm_p_restricted_wild_cluster_bootstrap_t",
    ]
    primary_upstream = primary_locked.merge(
        primary_audit[audit_columns],
        on=["endpoint", "scope", "n", "n_sites", "beta"],
        validate="1:1",
    )
    primary_upstream["wild_holm_pass"] = primary_upstream[
        "holm_p_restricted_wild_cluster_bootstrap_t"
    ].le(0.05)
    primary_upstream = primary_upstream[primary_export.columns]

    sensitivity_export = pd.read_csv(DATA / "04_completeness_acquisition_sensitivity.csv")
    sensitivity_upstream = pd.read_csv(STAGE / "sensitivity_contrasts.csv")
    sensitivity_upstream = sensitivity_upstream[
        sensitivity_upstream["stratum"].isin(sensitivity_export["stratum"].unique())
    ].copy()
    sensitivity_upstream["holm_pass"] = sensitivity_upstream["holm_p_two_endpoints"].lt(0.05)
    sensitivity_upstream = sensitivity_upstream[sensitivity_export.columns]

    systems_export = pd.read_csv(DATA / "03_system_effects.csv")
    systems_upstream = multivar.network_effect_frame()[systems_export.columns]

    centroid_export = pd.read_csv(DATA / "03_stage_centroids.csv")
    cohort = multivar.load_exact_cohort()
    metrics = multivar.collect_metrics()
    composites, _ = multivar.build_composites(
        cohort, metrics, multivar.PRIMARY_NETWORKS
    )
    residualized = multivar.nuisance_residuals(composites)
    centroid_bootstrap = multivar.cluster_bootstrap_centroids(residualized)
    centroid_rows = []
    for group in ["CN", "MCI", "AD"]:
        subset = residualized[residualized["group"].eq(group)]
        row = {
            "group": group,
            "n": int(len(subset)),
            "axial_residual_mean": float(subset["axial_residual"].mean()),
            "radial_residual_mean": float(subset["radial_residual"].mean()),
        }
        row.update(ellipse_summary(centroid_bootstrap[group]))
        centroid_rows.append(row)
    centroid_upstream = pd.DataFrame(centroid_rows)[centroid_export.columns]

    class_export = pd.read_csv(DATA / "03_classification_summary_AD_vs_MCI.csv")
    class_upstream = pd.read_csv(ML / "classification_summary.csv")
    class_upstream = class_upstream[class_upstream["contrast"].eq("AD_vs_MCI")][class_export.columns]

    increments_export = pd.read_csv(DATA / "03_classification_increments_AD_vs_MCI.csv")
    increments_upstream = pd.read_csv(ML / "classification_increment_bootstrap.csv")
    increments_upstream = increments_upstream[
        increments_upstream["contrast"].eq("AD_vs_MCI")
        & increments_upstream["reference"].eq("baseline")
        & increments_upstream["comparator"].isin(["endpoint_composite", "endpoint_networks"])
    ][increments_export.columns]

    loo = pd.read_csv(DATA / "03_anonymous_leave_one_site_out.csv")
    loo_upstream, loo_summary = multivar.leave_one_site_out_pairs(composites)
    loo_upstream.insert(
        0, "anonymous_deletion_index", np.arange(1, len(loo_upstream) + 1)
    )
    loo_upstream = loo_upstream[loo.columns]
    ledger = pd.read_csv(OUT / "results_status_ledger.csv")
    claim_source = pd.read_csv(CLAIMS)
    status_map = {
        "A": "retained_primary_candidate",
        "B": "retained_secondary",
        "C": "exploratory_only",
        "D": "not_supported_or_invalid",
    }
    expected_ledger = claim_source[
        [
            "claim_id",
            "claim_name",
            "status_code",
            "status_label",
            "evidence_summary",
            "evidence_path",
            "required_before_vfinal",
        ]
    ].copy()
    expected_ledger.insert(3, "output_disposition", expected_ledger["status_code"].map(status_map))

    core_export = pd.read_csv(OUT / "core_results_table.csv")
    core_rows: list[dict[str, object]] = []
    for row in primary_export.itertuples():
        core_rows.append(
            {
                "result_family": "primary_adjacent_stage",
                "result": f"{row.endpoint} | {row.scope}",
                "estimate": row.beta,
                "ci95_low": row.ci95_low,
                "ci95_high": row.ci95_high,
                "adjusted_p_or_q": row.holm_p_restricted_wild_cluster_bootstrap_t,
                "status": "retained" if row.wild_holm_pass else "sensitivity_only",
                "source": "primary_inference_soundness_v2/primary_small_cluster_inference.csv",
            }
        )
    network_model = class_export[
        class_export["model"].eq("endpoint_networks")
    ].iloc[0]
    network_increment = increments_export[
        increments_export["comparator"].eq("endpoint_networks")
    ].iloc[0]
    core_rows.extend(
        [
            {
                "result_family": "system_participation",
                "result": "AxD MCI-CN systems passing endpoint-specific FDR",
                "estimate": int(systems_export["axial_q"].le(0.05).sum()),
                "ci95_low": np.nan,
                "ci95_high": np.nan,
                "adjusted_p_or_q": np.nan,
                "status": "retained",
                "source": "stage_specific_diffusivity_v2/network_contrasts.csv",
            },
            {
                "result_family": "system_participation",
                "result": "RD AD-MCI systems passing endpoint-specific FDR",
                "estimate": int(systems_export["radial_q"].le(0.05).sum()),
                "ci95_low": np.nan,
                "ci95_high": np.nan,
                "adjusted_p_or_q": np.nan,
                "status": "retained",
                "source": "stage_specific_diffusivity_v2/network_contrasts.csv",
            },
            {
                "result_family": "site_influence",
                "result": "Paired site deletions retaining both positive and nominally significant effects",
                "estimate": int(loo_summary["iterations"]),
                "ci95_low": np.nan,
                "ci95_high": np.nan,
                "adjusted_p_or_q": np.nan,
                "status": "retained",
                "source": "refined_output_release_v1/source_data/03_anonymous_leave_one_site_out.csv",
            },
            {
                "result_family": "held_out_site_classification",
                "result": "AD-MCI 17-system RD ROC AUC",
                "estimate": float(network_model["roc_auc"]),
                "ci95_low": np.nan,
                "ci95_high": np.nan,
                "adjusted_p_or_q": np.nan,
                "status": "secondary",
                "source": "network_resolved_site_validation_v2/classification_summary.csv",
            },
            {
                "result_family": "held_out_site_classification",
                "result": "AD-MCI 17-system RD delta AUC over baseline",
                "estimate": float(network_increment["delta_auc_median"]),
                "ci95_low": float(network_increment["delta_auc_ci95_low"]),
                "ci95_high": float(network_increment["delta_auc_ci95_high"]),
                "adjusted_p_or_q": np.nan,
                "status": "secondary",
                "source": "network_resolved_site_validation_v2/classification_increment_bootstrap.csv",
            },
        ]
    )
    core_upstream = pd.DataFrame(core_rows)[core_export.columns]

    figure_files = sorted(FIG.glob("*"))
    pngs = sorted(FIG.glob("*.png"))
    svgs = sorted(FIG.glob("*.svg"))
    pdfs = sorted(FIG.glob("*.pdf"))
    png_dimensions = {}
    for path in pngs:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            png_dimensions[path.name] = list(image.size)

    svg_valid = True
    for path in svgs:
        try:
            ET.parse(path)
        except ET.ParseError:
            svg_valid = False
    pdf_valid = all(path.read_bytes()[:5] == b"%PDF-" for path in pdfs)

    hash_replay = True
    for relative, record in {
        **build_validation["figures"],
        **build_validation["source_data"],
        **build_validation["tables"],
    }.items():
        path = OUT / relative
        if not path.is_file() or sha256(path) != record["sha256"]:
            hash_replay = False
            break

    exported_columns = {
        column
        for path in sorted(DATA.glob("*.csv"))
        for column in pd.read_csv(path, nrows=0).columns
    }
    prohibited = {"participant_id", "subject_id", "site", "site_code", "prediction", "fold"}

    checks = {
        "builder_validation_passed": build_validation.get("pass") is True,
        "builder_source_hash_replays": build_validation.get("builder", {}).get(
            "sha256"
        )
        == sha256(ROOT / "build_refined_output_release_v1.py"),
        "tensor_export_exactly_matches_upstream": frames_close(
            tensor_export, tensor_upstream, ["metric", "contrast"]
        ),
        "primary_export_exactly_matches_locked_and_audited_sources": frames_close(
            primary_export, primary_upstream, ["endpoint", "scope"]
        ),
        "sensitivity_export_exactly_matches_upstream": frames_close(
            sensitivity_export, sensitivity_upstream, ["endpoint", "stratum"]
        ),
        "system_export_exactly_matches_upstream": frames_close(
            systems_export, systems_upstream, ["network"]
        ),
        "stage_centroid_export_exactly_recomputes": frames_close(
            centroid_export, centroid_upstream, ["group"]
        ),
        "classification_export_exactly_matches_upstream": frames_close(
            class_export, class_upstream, ["contrast", "model"]
        ),
        "classification_increment_export_exactly_matches_upstream": frames_close(
            increments_export,
            increments_upstream,
            ["contrast", "comparator", "reference"],
        ),
        "all_six_primary_rows_have_9999_wild_draws": bool(
            primary_export["wild_bootstrap_valid_iterations"].eq(9999).all()
        ),
        "full_and_pair_common_rows_pass_wild_holm": bool(
            primary_export[primary_export["scope"].isin(["full_site_fe", "pair_common_sites"])][
                "wild_holm_pass"
            ].all()
        ),
        "all_three_rows_are_sensitivity_only": bool(
            (~primary_export[primary_export["scope"].eq("all_three_group_sites")]["wild_holm_pass"]).all()
        ),
        "anonymous_site_deletion_has_51_rows": len(loo) == 51,
        "anonymous_site_deletion_exactly_recomputes": frames_close(
            loo, loo_upstream, ["anonymous_deletion_index"]
        ),
        "anonymous_site_deletion_indices_are_complete": loo[
            "anonymous_deletion_index"
        ].tolist() == list(range(1, 52)),
        "all_anonymous_site_deletions_retain_both_positive_effects": bool(
            (loo[["axial_beta", "radial_beta"]] > 0).all().all()
        ),
        "all_anonymous_site_deletions_retain_both_nominal_p_below_005": bool(
            (loo[["axial_p", "radial_p"]] < 0.05).all().all()
        ),
        "claim_ledger_exactly_matches_authoritative_inventory": frames_close(
            ledger, expected_ledger, ["claim_id"]
        ),
        "core_results_table_exactly_reconstructs_from_sources": frames_close(
            core_export, core_upstream, ["result_family", "result"]
        ),
        "four_png_files_decode": len(pngs) == 4 and len(png_dimensions) == 4,
        "all_pngs_are_publication_scale": all(
            width >= 2500 and height >= 1400 for width, height in png_dimensions.values()
        ),
        "four_svgs_parse_as_xml": len(svgs) == 4 and svg_valid,
        "four_pdfs_have_valid_magic": len(pdfs) == 4 and pdf_valid,
        "builder_hashes_replay": hash_replay,
        "no_direct_identifier_or_site_label_column": not bool(exported_columns & prohibited),
        "exactly_eight_source_data_csv_files": len(list(DATA.glob("*.csv"))) == 8,
        "exactly_twelve_figure_files": len(figure_files) == 12,
        "core_results_table_has_11_rows": len(pd.read_csv(OUT / "core_results_table.csv")) == 11,
    }
    checks = {name: bool(value) for name, value in checks.items()}
    payload = {
        "schema_version": "1.0.0",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "pass": all(checks.values()),
        "checks": checks,
        "png_dimensions": png_dimensions,
        "builder_validation_sha256": sha256(OUT / "validation.json"),
        "validator_sha256": sha256(Path(__file__).resolve()),
    }
    destination = OUT / "independent_validation.json"
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Independent refined-output validation",
        "",
        f"**Result:** {'PASS' if payload['pass'] else 'FAIL'}",
        "",
    ]
    lines.extend(
        f"- [{'x' if passed else ' '}] {name.replace('_', ' ')}"
        for name, passed in checks.items()
    )
    (OUT / "independent_validation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"pass": payload["pass"], "checks": len(checks), "output": str(destination)}))
    if not payload["pass"]:
        failed = [name for name, passed in checks.items() if not passed]
        raise SystemExit(f"Independent refined-output validation failed: {failed}")


if __name__ == "__main__":
    main()
