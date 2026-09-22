"""The imaging run lifecycle, driven through the workflow's own launcher.

A run goes through three automated phases separated by human decisions, and
``scforge/workflow/run_connectome_v2.py`` enforces every step of it:

    approve     which subjects, which limits                    (a person)
    phase A     preprocessing, one response function per subject
    freeze      pool the responses; approve the calibration     (a person)
    pre-tract   FOD, tissue segmentation, registration, QC bundles
    review      look at the QC bundles                          (a person)
    continue    authorise tractography                          (a person)
    phase B     tractography, SIFT2, the nine matrices

This module does not re-implement any of those checks. Each decision is
*drafted by the launcher itself* -- the same function that later validates it
returns, in draft mode, exactly the fields it will require -- and only the
signature is added here. The formats cannot drift apart because there is only
one definition of each.

Files a run accumulates, under ``<run>/contract/``:

    execution_subset.csv                   the approved units
    execution_subset_decision.json         the approval
    source_lock/                           content hashes of every raw file
    environment_contract.yaml              the tools frozen at approval
    approval_inputs.json                   which manifest was approved
    response_calibration_decision.json     the calibration approval
    tractography_continuation_decision.json

and under ``<run>/review/human_visual_qc.csv`` the reviewer's verdicts.
"""

from __future__ import annotations

import csv
import datetime as _dt
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent
WORKFLOW_DIR = PROJECT_ROOT / "scforge" / "workflow"
for _extra in (PROJECT_ROOT / "scforge", WORKFLOW_DIR):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

LAUNCHER = WORKFLOW_DIR / "run_connectome_v2.py"
FREEZER = WORKFLOW_DIR / "freeze_response_calibration.py"
NORMATIVE_CONFIG = PROJECT_ROOT / "configs" / "connectome_v2.yaml"
WORKFLOW_PYTHON = PROJECT_ROOT / ".venv_connectome_workflow" / "bin" / "python"

PHASES = (
    ("response-calibration-phase-a", "response_calibration_phase_a_completion.json"),
    ("pre-tractography-canary", "pre_tractography_canary_completion.json"),
    ("phase-b", "run_completion.json"),
)


class LifecycleError(RuntimeError):
    """A step cannot run yet, with a message saying what comes first."""


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _launcher():
    import run_connectome_v2 as launcher

    return launcher


def _recipe() -> dict[str, Any]:
    return yaml.safe_load(NORMATIVE_CONFIG.read_text(encoding="utf-8"))


def _paths(run_root: Path) -> dict[str, Path]:
    contract = run_root / "contract"
    publication = run_root / "publication"
    return {
        "contract": contract,
        "subset": contract / "execution_subset.csv",
        "decision": contract / "execution_subset_decision.json",
        "inputs": contract / "approval_inputs.json",
        "environment": contract / "environment_contract.yaml",
        "calibration_decision": contract / "response_calibration_decision.json",
        "continuation_decision": contract / "tractography_continuation_decision.json",
        "human_qc": run_root / "review" / "human_visual_qc.csv",
        "frozen_dir": run_root / "05_group_response_frozen",
        "frozen_manifest": run_root / "05_group_response_frozen" / "frozen_response_calibration_manifest.json",
        "phase_a_completion": publication / "response_calibration_phase_a_completion.json",
        "phase_a_manifest": publication / "response_calibration_phase_a_manifest.json",
        "pre_completion": publication / "pre_tractography_canary_completion.json",
        "pre_manifest": publication / "pre_tractography_canary_manifest.json",
        "run_completion": publication / "run_completion.json",
    }


def _schema(environment: dict) -> Path:
    from scforge.environment import expand

    return Path(expand(_recipe()["inputs"]["acquisition_schema"]["path"], environment))


def _environment(run_root: Path) -> dict:
    from scforge.environment import expand, merged_contract

    frozen = _paths(run_root)["environment"]
    return expand(yaml.safe_load(frozen.read_text(encoding="utf-8")), None) if frozen.is_file() else merged_contract()


def _parent_rows(manifest: Path, run_root: Path) -> list[dict[str, str]]:
    from scforge.input_contract import load_acquisition_manifest

    rows = list(load_acquisition_manifest(manifest, _schema(_environment(run_root)), run_root=run_root))
    rows.sort(key=lambda row: row["unit"])
    return rows


def _binding(run_root: Path) -> tuple[dict, Path]:
    """The launcher's own execution binding for an approved run."""
    p = _paths(run_root)
    if not p["decision"].is_file():
        raise LifecycleError(f"{run_root} has not been approved; run `approve` first")
    inputs = json.loads(p["inputs"].read_text(encoding="utf-8"))
    manifest = Path(inputs["manifest"])
    binding, _ = _launcher().prepare_execution_binding(
        recipe_id=_recipe()["contract"]["recipe_id"],
        parent_manifest_path=manifest,
        parent_rows=_parent_rows(manifest, run_root),
        execution_subset_manifest_path=p["subset"],
        execution_subset_decision_path=p["decision"],
        acquisition_schema_path=_schema(_environment(run_root)),
        run_root=run_root,
        environment_contract_path=p["environment"] if p["environment"].is_file() else None,
    )
    return binding, manifest


def _sign(draft: dict, *, by: str, utc: str | None, note: str) -> dict:
    return {**draft, "approved_by": by, "approved_utc": utc or _now(),
            "user_response": note or f"approved with run_imaging.py by {by}"}


def _write_json(path: Path, record: dict) -> None:
    if path.exists():
        raise LifecycleError(f"{path.name} already exists; decisions are not overwritten")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- approve

def approve(*, manifest: Path, run_root: Path, units: list[str], by: str, utc: str | None,
            note: str, max_cores: int, wall_clock_hours: int, storage_gb: int,
            min_valid_calibration: int | None, min_manufacturers: int) -> dict:
    """Approve units for phase A and pre-tractography, and prove the launcher accepts it."""
    import sc_source_lock
    from scforge.environment import merged_contract
    from scforge.input_contract import PORTABLE_APPROVAL_MODE
    from scforge.response_calibration import manufacturer_family

    run_root = run_root.resolve()
    p = _paths(run_root)
    if p["decision"].exists():
        raise LifecycleError(f"{run_root} is already approved; use a new run root")
    rows = _parent_rows(manifest, run_root)
    by_unit = {row["unit"]: row for row in rows}
    missing = [u for u in units if u not in by_unit]
    if missing:
        raise LifecycleError(f"not in the manifest: {', '.join(missing[:5])}")
    chosen = sorted(set(units))
    if not chosen:
        raise LifecycleError("no units selected")

    p["contract"].mkdir(parents=True, exist_ok=True)
    fields = [name for name in next(csv.reader(manifest.open(encoding="utf-8")))]
    with p["subset"].open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for unit in chosen:
            writer.writerow(by_unit[unit])

    lock = sc_source_lock.build(p["subset"], p["contract"] / "source_lock", rehash=True)
    if lock["missing_files"]:
        raise LifecycleError(f"{len(lock['missing_files'])} approved source file(s) are missing, "
                             f"e.g. {lock['missing_files'][0]}")
    p["environment"].write_text(
        "# Environment frozen for this run by `run_imaging.py approve`.\n"
        + yaml.safe_dump(merged_contract(), sort_keys=False), encoding="utf-8")
    p["inputs"].write_text(json.dumps({"manifest": str(manifest.resolve())}, indent=2) + "\n",
                           encoding="utf-8")

    # A portable approval states what this batch actually is. The calibration
    # later requires the valid pool to contain exactly these T1 source classes
    # and at least this many scanner families.
    t1_classes = sorted({by_unit[u]["t1_source_kind"] for u in chosen})
    families = sorted({manufacturer_family(by_unit[u]["manufacturer"]) for u in chosen})
    minimum_valid = min_valid_calibration or max(1, math.ceil(len(chosen) / 2))
    if min_manufacturers > len(families):
        raise LifecycleError(f"the batch has {len(families)} scanner famil(ies) ({', '.join(families)}); "
                             f"cannot require {min_manufacturers}")
    request = {
        "approval_mode": PORTABLE_APPROVAL_MODE,
        "maximum_cores": max_cores,
        "minimum_valid_response_calibration_units": minimum_valid,
        "minimum_valid_manufacturer_families": min_manufacturers,
        "minimum_valid_t1_source_classes": len(t1_classes),
        "required_valid_t1_source_classes": t1_classes,
        "wall_clock_stop_hours": wall_clock_hours,
        "storage_stop_gb": storage_gb,
        "tractography_authorized": False,
        "matrix_generation_authorized": False,
        "full_cohort_authorized": False,
    }
    launcher = _launcher()
    draft, _ = launcher.prepare_execution_binding(
        recipe_id=_recipe()["contract"]["recipe_id"],
        parent_manifest_path=manifest.resolve(),
        parent_rows=rows,
        execution_subset_manifest_path=p["subset"],
        execution_subset_decision_path=p["decision"],
        acquisition_schema_path=_schema(_environment(run_root)),
        run_root=run_root,
        environment_contract_path=p["environment"],
        draft_policy=request,
    )
    _write_json(p["decision"], _sign(draft, by=by, utc=utc, note=note))
    binding, _ = _binding(run_root)          # the launcher accepts it, or this raises
    return {"units": chosen, "families": families, "t1_classes": t1_classes,
            "minimum_valid": minimum_valid, "source_files": lock["inventory_row_count"],
            "binding_units": binding["units"]}


# --------------------------------------------------------------------------- state

def next_phase(run_root: Path) -> tuple[str | None, str]:
    """Which launcher mode runs next, or what a person has to do first."""
    p = _paths(run_root)
    if not p["decision"].is_file():
        return None, "not approved yet: run `approve`"
    if not p["phase_a_completion"].is_file():
        return "response-calibration-phase-a", "phase A: preprocessing and response functions"
    if not p["calibration_decision"].is_file():
        return None, "phase A is complete: run `freeze` to pool and approve the calibration"
    if not p["pre_completion"].is_file():
        return "pre-tractography-canary", "pre-tractography: FODs, tissue segmentation, registration, QC bundles"
    if not p["human_qc"].is_file():
        return None, "pre-tractography is complete: look at the QC bundles, then run `review`"
    if not p["continuation_decision"].is_file():
        return None, "review recorded: run `continue` to authorise tractography"
    if not p["run_completion"].is_file():
        return "phase-b", "phase B: tractography, SIFT2 and the nine matrices"
    return None, "the run is complete: run `publish`"


def launcher_command(run_root: Path, mode: str, cores: int) -> list[str]:
    p = _paths(run_root)
    inputs = json.loads(p["inputs"].read_text(encoding="utf-8"))
    decision = json.loads(p["decision"].read_text(encoding="utf-8"))
    python = str(WORKFLOW_PYTHON if WORKFLOW_PYTHON.is_file() else Path(sys.executable))
    command = [python, str(LAUNCHER), "--mode", mode, "--manifest", inputs["manifest"],
               "--run-root", str(run_root), "--execution-subset-manifest", str(p["subset"]),
               "--execution-subset-decision", str(p["decision"]), "--cores", str(cores)]
    if mode in ("pre-tractography-canary", "phase-b"):
        command += ["--response-calibration-manifest", str(p["frozen_manifest"]),
                    "--response-calibration-minimum",
                    str(decision["minimum_valid_response_calibration_units"]),
                    "--response-calibration-decision", str(p["calibration_decision"])]
    if mode == "phase-b":
        command += ["--human-qc-manifest", str(p["human_qc"]),
                    "--tractography-continuation-decision", str(p["continuation_decision"])]
    return command


# --------------------------------------------------------------------------- freeze

def freeze(*, run_root: Path, by: str, utc: str | None, note: str) -> dict:
    """Pool phase A's valid responses and sign the calibration the launcher drafts."""
    run_root = run_root.resolve()
    p = _paths(run_root)
    if not p["phase_a_completion"].is_file():
        raise LifecycleError("phase A has not completed; run the workflow first")
    decision = json.loads(p["decision"].read_text(encoding="utf-8"))
    minimum = int(decision["minimum_valid_response_calibration_units"])
    if not p["frozen_manifest"].is_file():
        python = str(WORKFLOW_PYTHON if WORKFLOW_PYTHON.is_file() else Path(sys.executable))
        done = subprocess.run(
            [python, str(FREEZER), "--phase-a-completion", str(p["phase_a_completion"]),
             "--phase-a-manifest", str(p["phase_a_manifest"]),
             "--output-dir", str(p["frozen_dir"]), "--minimum-valid-subjects", str(minimum)],
            capture_output=True, text=True)
        if done.returncode != 0:
            raise LifecycleError("freezing the calibration failed:\n" + (done.stderr or done.stdout).strip()[-2000:])
    binding, _ = _binding(run_root)
    launcher = _launcher()
    kwargs = dict(recipe_id=_recipe()["contract"]["recipe_id"],
                  phase_a_completion_path=p["phase_a_completion"],
                  response_calibration_manifest_path=p["frozen_manifest"],
                  minimum_valid_subjects=minimum, execution_binding=binding)
    draft = launcher.prepare_phase_b_binding(**kwargs, response_calibration_decision_path=None, draft=True)
    _write_json(p["calibration_decision"], _sign(draft, by=by, utc=utc, note=note))
    launcher.prepare_phase_b_binding(**kwargs, response_calibration_decision_path=p["calibration_decision"])
    frozen = json.loads(p["frozen_manifest"].read_text(encoding="utf-8"))
    return {"valid": frozen.get("valid_subject_count"), "invalid": frozen.get("invalid_subject_count"),
            "valid_units": frozen.get("valid_units", [])}


# --------------------------------------------------------------------------- review

def reviewable_units(run_root: Path) -> list[str]:
    p = _paths(run_root)
    if not p["pre_manifest"].is_file():
        raise LifecycleError("pre-tractography has not completed; there is nothing to review")
    return list(json.loads(p["pre_manifest"].read_text(encoding="utf-8")).get("ready_units", []))


def review(*, run_root: Path, reviewer: str, status: str, units: list[str] | None,
           utc: str | None, note: str) -> Path:
    """Record a person's verdict on the QC bundles of the reviewable units."""
    run_root = run_root.resolve()
    ready = reviewable_units(run_root)
    chosen = units or ready
    unknown = sorted(set(chosen) - set(ready))
    if unknown:
        raise LifecycleError(f"not reviewable in this run: {', '.join(unknown)}")
    path = _paths(run_root)["human_qc"]
    rows: dict[str, dict] = {}
    if path.is_file():
        with path.open(newline="", encoding="utf-8-sig") as handle:
            rows = {row["unit"]: row for row in csv.DictReader(handle)}
    stamp = utc or _now()
    for unit in chosen:
        rows[unit] = {"unit": unit, "status": status.upper(), "reviewer": reviewer,
                      "reviewed_utc": stamp, "note": note}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["unit", "status", "reviewer", "reviewed_utc", "note"])
        writer.writeheader()
        for unit in sorted(rows):
            writer.writerow(rows[unit])
    return path


# --------------------------------------------------------------------------- continue

def continue_to_tractography(*, run_root: Path, by: str, utc: str | None, note: str) -> dict:
    """Sign the continuation decision the launcher drafts from the review."""
    run_root = run_root.resolve()
    p = _paths(run_root)
    for key, what in (("pre_completion", "pre-tractography"), ("human_qc", "the review"),
                      ("calibration_decision", "the calibration approval")):
        if not p[key].is_file():
            raise LifecycleError(f"{what} is not in place yet")
    binding, _ = _binding(run_root)
    decision = json.loads(p["decision"].read_text(encoding="utf-8"))
    launcher = _launcher()
    calibration = launcher.prepare_phase_b_binding(
        recipe_id=_recipe()["contract"]["recipe_id"],
        phase_a_completion_path=p["phase_a_completion"],
        response_calibration_manifest_path=p["frozen_manifest"],
        minimum_valid_subjects=int(decision["minimum_valid_response_calibration_units"]),
        response_calibration_decision_path=p["calibration_decision"],
        execution_binding=binding,
    )
    kwargs = dict(recipe_id=_recipe()["contract"]["recipe_id"], execution_binding=binding,
                  response_calibration_binding=calibration,
                  pre_tractography_completion_path=p["pre_completion"],
                  human_qc_manifest_path=p["human_qc"])
    draft = launcher.prepare_tractography_continuation_binding(**kwargs, continuation_decision_path=None, draft=True)
    _write_json(p["continuation_decision"], _sign(draft, by=by, utc=utc, note=note))
    bound = launcher.prepare_tractography_continuation_binding(
        **kwargs, continuation_decision_path=p["continuation_decision"])
    return {"units": bound["approved_units"]}
