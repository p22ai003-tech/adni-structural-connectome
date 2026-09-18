from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, is_dataclass
import json
import os
from pathlib import Path
import subprocess
import sys

from .atlas import build_aal3_node_table, write_aal3_node_table
from .config import load_config
from .connectome import build_connectome_command_set
from .dwi import build_dwi_preproc_commands
from .five_tt import build_five_tt_command_set
from .fod import build_fod_command_set
from .manifest import build_source_contract, validate_bvals_bvecs
from .qc import aal3_label_summary, matrix_qc_from_atlas_config
from .registration import qc_affine_matrix
from .runner import ensure_dirs
from .spatial_contract import build_spatial_canary_plan, run_spatial_canary, write_launch_manifest
from .tractography import build_tractography_command_set


V2_DEFAULT_SUBJECTS = (
    "003_S_0908_I1249292",
    "014_S_4401_I1556672",
    "041_S_5141_I893581",
    "041_S_4427_I1243839",
    "021_S_7092_I1597668",
    "031_S_4021_I1253150",
    "033_S_7114_I11063036",
    "003_S_6257_I974346",
    "127_S_5028_I401540",
    "168_S_6874_I1667523",
)


def _default_config() -> Path:
    return Path(__file__).resolve().parents[1] / "configs" / "scforge.yaml"


def _project_root() -> Path:
    return Path("/home/ec2-user/exp")


def _latest_v2_root() -> Path | None:
    latest = _project_root() / "data" / "derivatives" / "qc" / "sc_matrix_qc" / "scforge_v2_latest.txt"
    if latest.exists():
        value = latest.read_text(encoding="utf-8", errors="replace").strip()
        if value:
            return Path(value)
    roots = sorted((latest.parent).glob("scforge_v2_*")) if latest.parent.exists() else []
    return roots[-1] if roots else None


def _latest_v1_density_root() -> Path | None:
    latest = _project_root() / "data" / "derivatives" / "qc" / "sc_matrix_qc" / "scforge_v1_density_latest.txt"
    if latest.exists():
        value = latest.read_text(encoding="utf-8", errors="replace").strip()
        if value:
            return Path(value)
    roots = sorted((latest.parent).glob("scforge_v1_density_batch_*")) if latest.parent.exists() else []
    return roots[-1] if roots else None


def _print_json(data: object) -> None:
    print(json.dumps(data, indent=2, sort_keys=True))


def _command_records(command_object: object) -> list[dict]:
    if hasattr(command_object, "to_records"):
        return command_object.to_records()
    if is_dataclass(command_object):
        data = asdict(command_object)
    elif isinstance(command_object, dict):
        data = command_object
    else:
        data = {}
    return [{"stage": key, "command": " ".join(str(x) for x in value)} for key, value in data.items()]


def _write_or_print(data: object, output: str | None = None) -> None:
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, sort_keys=True)
            handle.write("\n")
    else:
        _print_json(data)


def cmd_status(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    routes = [route.name for route in cfg.spatial_routes]
    data = {
        "project_root": str(cfg.project_root),
        "derivatives_root": str(cfg.derivatives_root),
        "existing_derivatives_root": str(cfg.existing_derivatives_root),
        "atlas_image_exists": cfg.atlas.mni_image.exists(),
        "atlas_labels_exists": cfg.atlas.labels_csv.exists(),
        "spatial_routes": routes,
        "overwrite_existing": cfg.publication.overwrite_existing,
    }
    _print_json(data)
    return 0


def cmd_aal3_summary(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    _print_json(aal3_label_summary(cfg.atlas))
    return 0


def cmd_matrix_qc(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    result = matrix_qc_from_atlas_config(args.matrix, cfg.qc, cfg.atlas)
    _print_json(result.to_dict())
    return 0 if result.readable else 2


def cmd_gradient_qc(args: argparse.Namespace) -> int:
    result = validate_bvals_bvecs(args.bval, args.bvec)
    _print_json(result.to_dict())
    return 0 if result.status == "PASS" else 1


def cmd_affine_qc(args: argparse.Namespace) -> int:
    result = qc_affine_matrix(args.matrix)
    _print_json(result.to_dict())
    return 0 if result.status == "PASS" else 1


def cmd_init_derivatives(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    dirs = [
        cfg.derivatives_root,
        cfg.derivatives_root / "raw_contracts",
        cfg.derivatives_root / "subjects",
        cfg.derivatives_root / "group_qc",
        cfg.derivatives_root / "quarantine",
    ]
    _print_json({"execute": bool(args.execute), "directories": ensure_dirs(dirs, execute=args.execute)})
    return 0


def cmd_source_contract(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    contract = build_source_contract(
        subject=args.subject,
        nifti=args.nifti,
        bval=args.bval,
        bvec=args.bvec,
        json_sidecar=args.json_sidecar,
        expected_n_volumes=args.expected_n_volumes,
        reject_derived=cfg.dwi.reject_derived,
    )
    _write_or_print(contract.to_dict(), args.output)
    return 0 if contract.status == "PASS" else 1


def cmd_write_aal3_node_table(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    out = write_aal3_node_table(cfg.atlas, args.output)
    _print_json({"output": str(out), "rows": int(len(build_aal3_node_table(cfg.atlas)))})
    return 0


def cmd_plan_dwi(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    commands = build_dwi_preproc_commands(
        nifti=Path(args.nifti),
        bvec=Path(args.bvec),
        bval=Path(args.bval),
        json_sidecar=Path(args.json_sidecar) if args.json_sidecar else None,
        out_dir=Path(args.out_dir),
        config=cfg.dwi,
        reverse_pe_epi=Path(args.reverse_pe_epi) if args.reverse_pe_epi else None,
    )
    _write_or_print({"subject": args.subject, "commands": _command_records(commands)}, args.output)
    return 0


def cmd_plan_five_tt(args: argparse.Namespace) -> int:
    commands = build_five_tt_command_set(
        subjects_dir_subject=Path(args.subjects_dir_subject),
        t1_native=Path(args.t1_native),
        t1_to_b0_mrtrix=Path(args.t1_to_b0_mrtrix),
        out_dir=Path(args.out_dir),
    )
    _write_or_print({"subject": args.subject, "commands": _command_records(commands)}, args.output)
    return 0


def cmd_plan_fod(args: argparse.Namespace) -> int:
    commands = build_fod_command_set(
        dwi_biascorr=Path(args.dwi_biascorr),
        mask=Path(args.mask),
        out_dir=Path(args.out_dir),
        shells=[int(x) for x in args.shells],
    )
    _write_or_print({"subject": args.subject, "commands": _command_records(commands)}, args.output)
    return 0


def cmd_plan_preflight(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    commands = build_tractography_command_set(
        wmfod=Path(args.wmfod),
        five_tt=Path(args.five_tt),
        gmwmi=Path(args.gmwmi),
        out_dir=Path(args.out_dir),
        config=cfg.tractography,
    )
    _write_or_print({"subject": args.subject, "commands": _command_records(commands)}, args.output)
    return 0


def cmd_plan_connectome(args: argparse.Namespace) -> int:
    commands = build_connectome_command_set(
        tracks=Path(args.tracks),
        nodes=Path(args.nodes),
        out_dir=Path(args.out_dir),
        assignment_variant=args.assignment_variant,
        sift2_weights=Path(args.sift2_weights) if args.sift2_weights else None,
        fa_mif=Path(args.fa_mif) if args.fa_mif else None,
        md_mif=Path(args.md_mif) if args.md_mif else None,
        rd_mif=Path(args.rd_mif) if args.rd_mif else None,
        ad_mif=Path(args.ad_mif) if args.ad_mif else None,
    )
    _write_or_print({"subject": args.subject, "commands": _command_records(commands)}, args.output)
    return 0


def cmd_reindex_aal3_labels(args: argparse.Namespace) -> int:
    try:
        import nibabel as nib
        import numpy as np
        import pandas as pd
    except Exception as exc:
        _print_json({"status": "FAIL", "reason": f"missing_python_dependency:{type(exc).__name__}:{exc}"})
        return 2
    source = Path(args.orig_label_image)
    table = pd.read_csv(args.node_table_tsv, sep="\t")
    mapping = {
        int(row["original_aal_label"]): int(row["node_index"])
        for _, row in table.iterrows()
        if not pd.isna(row.get("original_aal_label")) and not pd.isna(row.get("node_index"))
    }
    image = nib.load(str(source))
    data = np.asarray(image.get_fdata(), dtype=np.int32)
    out = np.zeros_like(data, dtype=np.int16)
    for original_label, node_index in mapping.items():
        out[data == original_label] = node_index
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(out, image.affine, image.header), str(output))
    _print_json({"status": "PASS", "output": str(output), "mapped_labels": len(mapping)})
    return 0


def cmd_spatial_canary(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    variants = args.variants
    if variants and args.include_forward80 and "forward80" not in variants:
        variants = [*variants, "forward80"]
    plan = build_spatial_canary_plan(
        cfg,
        tag=args.tag,
        subjects=args.subjects,
        variants=variants,
        include_forward80=bool(args.include_forward80),
        execute=bool(args.execute),
    )
    write_launch_manifest(plan)
    _print_json({"status": "RUNNING" if args.execute else "DRY_RUN", **plan.to_dict()})
    return run_spatial_canary(plan, cfg)


def cmd_v2_plan(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    data = {
        "status": "READY",
        "scratch_only": True,
        "runner": str(_project_root() / "run_scforge_v2_canary.py"),
        "launcher": str(_project_root() / "launch_scforge_v2_canary.sh"),
        "monitor": str(_project_root() / "watch_scforge_v2_monitor.sh"),
        "subjects": list(args.subjects or V2_DEFAULT_SUBJECTS),
        "candidate_pass": asdict(cfg.qc.candidate_pass),
        "publication_pass": asdict(cfg.qc.publication_pass),
        "attach_runner": "tmux attach -t scforge_v2_canary",
        "attach_monitor": "tmux attach -t scforge_v2_monitor",
    }
    _print_json(data)
    return 0


def cmd_v2_canary(args: argparse.Namespace) -> int:
    script = _project_root() / "run_scforge_v2_canary.py"
    command = [sys.executable, str(script), "--config", str(args.config)]
    if args.tag:
        command.extend(["--tag", args.tag])
    if args.subjects:
        command.append("--subjects")
        command.extend(args.subjects)
    if args.execute:
        command.append("--execute")
    if args.launch_tmux:
        launcher = _project_root() / "launch_scforge_v2_canary.sh"
        launcher_command = [str(launcher)]
        if args.tag:
            launcher_command = ["env", f"SCFORGE_V2_TAG={args.tag}", *launcher_command]
        return subprocess.call(launcher_command)
    return subprocess.call(command)


def cmd_v2_qc(args: argparse.Namespace) -> int:
    root = Path(args.root) if args.root else _latest_v2_root()
    if not root:
        _print_json({"status": "MISSING", "reason": "no_scforge_v2_run_found"})
        return 1
    status_path = root / "status.json"
    summary_path = root / "scforge_v2_canary_summary.csv"
    payload: dict[str, object] = {"root": str(root), "status_file": str(status_path), "summary_file": str(summary_path)}
    if status_path.exists():
        payload["status"] = json.loads(status_path.read_text(encoding="utf-8", errors="replace"))
    else:
        payload["status"] = {"phase": "unknown"}
    if summary_path.exists():
        import pandas as pd

        payload["summary"] = pd.read_csv(summary_path).to_dict(orient="records")
    _print_json(payload)
    return 0


def cmd_v2_summary(args: argparse.Namespace) -> int:
    root = Path(args.root) if args.root else _latest_v2_root()
    if not root:
        print("No SC-Forge v2 run found.")
        return 1
    handoff = root / "scforge_v2_gpt_handoff.md"
    if handoff.exists():
        print(handoff.read_text(encoding="utf-8", errors="replace"))
        return 0
    return cmd_v2_qc(argparse.Namespace(root=str(root)))


def _latest_completed_connectome_qc() -> Path | None:
    root = _project_root() / "data" / "derivatives" / "qc" / "sc_matrix_qc"
    files = sorted(root.glob("completed_connectomes_after_gap_*/sc_matrix_integrity_subjects.csv"))
    if files:
        return files[-1]
    fallback = root / "sc_matrix_integrity_subjects.csv"
    return fallback if fallback.exists() else None


def cmd_v1_density_plan(args: argparse.Namespace) -> int:
    qc_path = _latest_completed_connectome_qc()
    fd_rows = 0
    if qc_path:
        with qc_path.open(newline="", encoding="utf-8", errors="replace") as handle:
            for row in csv.DictReader(handle):
                if row.get("weight") == "fd_sum" and not str(row.get("read_error", "")).strip():
                    fd_rows += 1
    data = {
        "status": "READY" if qc_path else "MISSING_QC",
        "scratch_only": True,
        "qc_source": str(qc_path) if qc_path else "",
        "fd_sum_subjects": fd_rows,
        "runner": str(_project_root() / "run_scforge_v1_density_batch.py"),
        "launcher": str(_project_root() / "launch_scforge_v1_density_batch.sh"),
        "monitor": str(_project_root() / "watch_scforge_v1_density_monitor.sh"),
        "variants": ["default", "radial8", "forward40"],
        "required_metrics": ["count", "fd_sum", "len_mean", "fa_mean", "md_mean", "rd_mean", "ad_mean"],
        "attach_runner": "tmux attach -t scforge_v1_density_batch",
        "attach_monitor": "tmux attach -t scforge_v1_density_monitor",
        "production_overwrite": False,
    }
    _print_json(data)
    return 0 if qc_path else 1


def cmd_v1_density_batch(args: argparse.Namespace) -> int:
    launcher = _project_root() / "launch_scforge_v1_density_batch.sh"
    script = _project_root() / "run_scforge_v1_density_batch.py"
    if args.launch_tmux:
        command = [str(launcher)]
        env = dict(**os.environ)
        if args.tag:
            env["SCFORGE_V1_TAG"] = args.tag
        if args.phase:
            env["SCFORGE_V1_PHASE"] = args.phase
        return subprocess.call(command, env=env)
    command = [sys.executable, str(script), "--phase", args.phase]
    if args.tag:
        command.extend(["--tag", args.tag])
    if args.subjects:
        command.append("--subjects")
        command.extend(args.subjects)
    if args.limit:
        command.extend(["--limit", str(args.limit)])
    if args.execute:
        command.append("--execute")
    return subprocess.call(command)


def cmd_v1_density_summary(args: argparse.Namespace) -> int:
    root = Path(args.root) if args.root else _latest_v1_density_root()
    if not root:
        print("No SC-Forge v1 density batch run found.")
        return 1
    status_path = root / "status.json"
    summary_path = root / "scforge_v1_canary_summary.csv"
    payload: dict[str, object] = {"root": str(root)}
    if status_path.exists():
        payload["status"] = json.loads(status_path.read_text(encoding="utf-8", errors="replace"))
    if summary_path.exists():
        with summary_path.open(newline="", encoding="utf-8", errors="replace") as handle:
            payload["summary"] = list(csv.DictReader(handle))
    _print_json(payload)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="scforge")
    parser.add_argument("--config", default=str(_default_config()), help="Path to scforge.yaml")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_config_option(child: argparse.ArgumentParser) -> None:
        child.add_argument("--config", default=argparse.SUPPRESS, help="Path to scforge.yaml")

    status = sub.add_parser("status", help="Show config and path status")
    add_config_option(status)
    status.set_defaults(func=cmd_status)

    aal3 = sub.add_parser("aal3-summary", help="Summarize AAL3 label contract")
    add_config_option(aal3)
    aal3.set_defaults(func=cmd_aal3_summary)

    matrix = sub.add_parser("matrix-qc", help="Run matrix QC on one CSV")
    matrix.add_argument("matrix")
    add_config_option(matrix)
    matrix.set_defaults(func=cmd_matrix_qc)

    grad = sub.add_parser("gradient-qc", help="Validate one bval/bvec pair")
    grad.add_argument("bval")
    grad.add_argument("bvec")
    add_config_option(grad)
    grad.set_defaults(func=cmd_gradient_qc)

    affine = sub.add_parser("affine-qc", help="Validate one 4x4 transform matrix")
    affine.add_argument("matrix")
    add_config_option(affine)
    affine.set_defaults(func=cmd_affine_qc)

    init = sub.add_parser("init-derivatives", help="List or create SC-Forge derivative directories")
    init.add_argument("--execute", action="store_true", help="Actually create directories")
    add_config_option(init)
    init.set_defaults(func=cmd_init_derivatives)

    init_alias = sub.add_parser("init", help="Alias for init-derivatives")
    init_alias.add_argument("--execute", action="store_true", help="Actually create directories")
    add_config_option(init_alias)
    init_alias.set_defaults(func=cmd_init_derivatives)

    source = sub.add_parser("source-contract", help="Validate one subject source DWI contract")
    source.add_argument("--subject", required=True)
    source.add_argument("--nifti", required=True)
    source.add_argument("--bval", required=True)
    source.add_argument("--bvec", required=True)
    source.add_argument("--json-sidecar")
    source.add_argument("--expected-n-volumes", type=int)
    source.add_argument("--output")
    add_config_option(source)
    source.set_defaults(func=cmd_source_contract)

    node_table = sub.add_parser("write-aal3-node-table", help="Write contiguous AAL3 node table")
    node_table.add_argument("output")
    add_config_option(node_table)
    node_table.set_defaults(func=cmd_write_aal3_node_table)

    plan_dwi = sub.add_parser("plan-dwi", help="Emit dry-run DWI preprocessing commands")
    plan_dwi.add_argument("--subject", required=True)
    plan_dwi.add_argument("--nifti", required=True)
    plan_dwi.add_argument("--bval", required=True)
    plan_dwi.add_argument("--bvec", required=True)
    plan_dwi.add_argument("--json-sidecar")
    plan_dwi.add_argument("--reverse-pe-epi")
    plan_dwi.add_argument("--out-dir", required=True)
    plan_dwi.add_argument("--output")
    add_config_option(plan_dwi)
    plan_dwi.set_defaults(func=cmd_plan_dwi)

    plan_5tt = sub.add_parser("plan-5tt", help="Emit dry-run 5TT/GMWMI commands")
    plan_5tt.add_argument("--subject", required=True)
    plan_5tt.add_argument("--subjects-dir-subject", required=True)
    plan_5tt.add_argument("--t1-native", required=True)
    plan_5tt.add_argument("--t1-to-b0-mrtrix", required=True)
    plan_5tt.add_argument("--out-dir", required=True)
    plan_5tt.add_argument("--output")
    add_config_option(plan_5tt)
    plan_5tt.set_defaults(func=cmd_plan_five_tt)

    plan_fod = sub.add_parser("plan-fod", help="Emit dry-run FOD/DTI commands")
    plan_fod.add_argument("--subject", required=True)
    plan_fod.add_argument("--dwi-biascorr", required=True)
    plan_fod.add_argument("--mask", required=True)
    plan_fod.add_argument("--out-dir", required=True)
    plan_fod.add_argument("--shells", nargs="+", required=True)
    plan_fod.add_argument("--output")
    add_config_option(plan_fod)
    plan_fod.set_defaults(func=cmd_plan_fod)

    plan_preflight = sub.add_parser("plan-preflight", help="Emit dry-run tractography commands")
    plan_preflight.add_argument("--subject", required=True)
    plan_preflight.add_argument("--wmfod", required=True)
    plan_preflight.add_argument("--five-tt", required=True)
    plan_preflight.add_argument("--gmwmi", required=True)
    plan_preflight.add_argument("--out-dir", required=True)
    plan_preflight.add_argument("--output")
    add_config_option(plan_preflight)
    plan_preflight.set_defaults(func=cmd_plan_preflight)

    plan_conn = sub.add_parser("plan-connectome", help="Emit dry-run connectome commands")
    plan_conn.add_argument("--subject", required=True)
    plan_conn.add_argument("--tracks", required=True)
    plan_conn.add_argument("--nodes", required=True)
    plan_conn.add_argument("--out-dir", required=True)
    plan_conn.add_argument("--assignment-variant", default="radial4")
    plan_conn.add_argument("--sift2-weights")
    plan_conn.add_argument("--fa-mif")
    plan_conn.add_argument("--md-mif")
    plan_conn.add_argument("--rd-mif")
    plan_conn.add_argument("--ad-mif")
    plan_conn.add_argument("--output")
    add_config_option(plan_conn)
    plan_conn.set_defaults(func=cmd_plan_connectome)

    reindex = sub.add_parser("reindex-aal3-labels", help="Convert original AAL3 labels to contiguous 1..166 nodes")
    reindex.add_argument("orig_label_image")
    reindex.add_argument("node_table_tsv")
    reindex.add_argument("output")
    add_config_option(reindex)
    reindex.set_defaults(func=cmd_reindex_aal3_labels)

    spatial = sub.add_parser(
        "spatial-canary",
        help="Run the scratch-only AAL3 spatial-contract canary and validation closeout",
    )
    spatial.add_argument("--execute", action="store_true", help="Actually run the scratch canary")
    spatial.add_argument("--tag", help="Output tag under qc/sc_matrix_qc")
    spatial.add_argument("--subjects", nargs="*", help="Subject IDs to test; defaults to the fixed SC-Forge panel")
    spatial.add_argument(
        "--variants",
        nargs="*",
        help="Assignment variants to test; defaults to default radial8 forward40",
    )
    spatial.add_argument(
        "--include-forward80",
        action="store_true",
        help="Also test forward80 assignment as an audit variant",
    )
    add_config_option(spatial)
    spatial.set_defaults(func=cmd_spatial_canary)

    v2_plan = sub.add_parser("v2-plan", help="Show SC-Forge v2 scratch-canary plan and gates")
    v2_plan.add_argument("--subjects", nargs="*", help="Optional subject IDs; defaults to the fixed v2 canary panel")
    add_config_option(v2_plan)
    v2_plan.set_defaults(func=cmd_v2_plan)

    v2_canary = sub.add_parser("v2-canary", help="Run SC-Forge v2 scratch-only canary")
    v2_canary.add_argument("--execute", action="store_true", help="Actually run tck2connectome commands")
    v2_canary.add_argument("--tag", help="Output tag under qc/sc_matrix_qc")
    v2_canary.add_argument("--subjects", nargs="*", help="Optional subject IDs; defaults to the fixed v2 canary panel")
    v2_canary.add_argument("--launch-tmux", action="store_true", help="Launch runner and monitor tmux sessions")
    add_config_option(v2_canary)
    v2_canary.set_defaults(func=cmd_v2_canary)

    v2_qc = sub.add_parser("v2-qc", help="Show latest SC-Forge v2 status and summary")
    v2_qc.add_argument("--root", help="Specific SC-Forge v2 run root; defaults to latest")
    add_config_option(v2_qc)
    v2_qc.set_defaults(func=cmd_v2_qc)

    v2_summary = sub.add_parser("v2-summary", help="Print latest SC-Forge v2 handoff summary")
    v2_summary.add_argument("--root", help="Specific SC-Forge v2 run root; defaults to latest")
    add_config_option(v2_summary)
    v2_summary.set_defaults(func=cmd_v2_summary)

    v1_plan = sub.add_parser("v1-density-plan", help="Show SC-Forge v1 all-subject density batch plan")
    add_config_option(v1_plan)
    v1_plan.set_defaults(func=cmd_v1_density_plan)

    v1_batch = sub.add_parser("v1-density-batch", help="Run SC-Forge v1 scratch-only density improvement batch")
    v1_batch.add_argument("--execute", action="store_true", help="Actually run probes and scratch candidate generation")
    v1_batch.add_argument("--tag", help="Output tag under qc/sc_matrix_qc")
    v1_batch.add_argument("--phase", choices=["inventory", "smoke", "canary", "full", "all"], default="all")
    v1_batch.add_argument("--subjects", nargs="*", help="Optional subject IDs")
    v1_batch.add_argument("--limit", type=int, help="Optional subject count limit")
    v1_batch.add_argument("--launch-tmux", action="store_true", help="Launch runner and monitor tmux sessions")
    add_config_option(v1_batch)
    v1_batch.set_defaults(func=cmd_v1_density_batch)

    v1_summary = sub.add_parser("v1-density-summary", help="Show latest SC-Forge v1 density batch status")
    v1_summary.add_argument("--root", help="Specific SC-Forge v1 density run root; defaults to latest")
    add_config_option(v1_summary)
    v1_summary.set_defaults(func=cmd_v1_density_summary)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
