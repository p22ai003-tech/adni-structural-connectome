"""Cohort settings: density floor, SMC handling, label hiding, primary sample, length measure.

The analysis pipeline (connectome_analysis, hcp_analysis) and the revision scripts (revision/tools, run from
docs/iit_dipanjan/tools) used to fix these in code: the density floor lived in three places and SMC participants were
always counted as MCI. They are settings now, and they live here, so that one chain cannot read two different values.

Every setting has a default that reproduces the published behaviour exactly. Resolution order:

    1. the environment variable below;
    2. <IIT_ROOT>/chain_settings.json, when a chain root is set (written once by tools/iit_chain.py init);
    3. the default.

    SC_DENSITY_FLOOR     0.6        minimum count density of a connectome; 0 or "none" switches the floor off
    SC_SMC               mci        "mci": subjective memory concern counted as MCI; "hold_out": kept out of every group
    SC_LABELS            (unset)    "shuffled": diagnosis and clinical scores permuted between participants (SC_LABEL_SEED
                                    is required and recorded); "real": real labels, the explicit flag
    IIT_PRIMARY_SAMPLE   adni34     "adni34": ADNI-3 and ADNI-4; "adni3": ADNI-3 alone; "pooled": everyone
    IIT_LENGTH_MEASURE   per_person "per_person": each person's mean streamline length; "mni_distance": MNI centre distance;
                                    "control_median_act": control-median ACT length per edge
    IIT_R4_FIELD, IIT_R4_MIN  reg_fa_corr, 0.4   the registration check of the exclusion rule; "none" states there is no such check
    SC_N_NODES           166        nodes analysed: the first n of the 166 AAL3 nodes. 154 leaves out the 12 brainstem nuclei, which
                                    the recipe v4 rule calls not estimable; the matrices stay 166 x 166 with those rows at zero

Real labels are joined without the flag only on the legacy route (no chain root, connectomes without a recorded recipe
hash), which is what keeps the published tree reproducible. Anywhere else, a run that has not said which labels it wants
stops before it reads a single label.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

DEFAULT_FLOOR = 0.6
ENV = {
    "density_floor": "SC_DENSITY_FLOOR",
    "smc": "SC_SMC",
    "labels": "SC_LABELS",
    "label_seed": "SC_LABEL_SEED",
    "primary_sample": "IIT_PRIMARY_SAMPLE",
    "length_measure": "IIT_LENGTH_MEASURE",
    "r4_field": "IIT_R4_FIELD",
    "r4_min": "IIT_R4_MIN",
    "n_nodes": "SC_N_NODES",
}
DEFAULTS = {
    "density_floor": DEFAULT_FLOOR,
    "smc": "mci",
    "labels": "",
    "label_seed": "",
    "primary_sample": "adni34",
    "length_measure": "per_person",
    "r4_field": "reg_fa_corr",
    "r4_min": 0.4,
    "n_nodes": 166,
}
CHOICES = {
    "smc": ("mci", "hold_out"),
    "labels": ("", "real", "shuffled"),
    "primary_sample": ("adni34", "adni3", "pooled"),
    "length_measure": ("per_person", "mni_distance", "control_median_act"),
}
SAMPLE_LABEL = {"adni34": "ADNI-3/4", "adni3": "ADNI-3", "pooled": "Pooled"}
SAMPLE_PHASES = {"adni34": ("ADNI-3", "ADNI-4"), "adni3": ("ADNI-3",), "pooled": None}

# Columns that carry diagnosis or a clinical score. They move together, as one block, when labels are shuffled.
_CLINICAL = re.compile(
    r"^(group|research group|global cdr|mmse.*|npi.*|gdscale.*|faq.*|apoe.*|cdr.*|diagnos.*|dx.*)$", re.IGNORECASE
)
LABEL_MODE_FILE = "label_mode.json"
DONOR_MAP_FILE = "label_donor_map.csv"
SETTINGS_FILE = "cohort_settings.json"


def chain_root() -> Path | None:
    raw = os.environ.get("IIT_ROOT", "").strip()
    return Path(raw) if raw else None


def chain_settings() -> dict:
    root = chain_root()
    if root is None:
        return {}
    file = root / "chain_settings.json"
    if not file.is_file():
        return {}
    return (json.loads(file.read_text()) or {}).get("settings", {})


def get(name: str):
    """A setting's value: environment, then the chain's settings file, then the default."""
    raw = os.environ.get(ENV[name])
    if raw is None or raw == "":
        value = chain_settings().get(name, DEFAULTS[name])
    else:
        value = raw
    if name in CHOICES and value not in CHOICES[name]:
        raise SystemExit(f"{ENV[name]}={value!r}: expected one of {', '.join(c or '(unset)' for c in CHOICES[name])}")
    return value


def density_floor() -> float:
    """The density floor. 'none', 'off' and 0 all mean no floor."""
    raw = get("density_floor")
    if isinstance(raw, str) and raw.strip().lower() in {"none", "off", "no"}:
        return 0.0
    try:
        floor = float(raw)
    except (TypeError, ValueError):
        raise SystemExit(f"{ENV['density_floor']}={raw!r}: expected a number between 0 and 1, or 'none'")
    if not 0.0 <= floor <= 1.0:
        raise SystemExit(f"{ENV['density_floor']}={raw!r}: expected a number between 0 and 1, or 'none'")
    return floor


def n_nodes() -> int:
    return int(get("n_nodes"))


def smc_held_out() -> bool:
    return get("smc") == "hold_out"


def non_default() -> dict:
    """The cohort settings that differ from the published ones, for the record beside the master cohort."""
    out = {}
    if density_floor() != DEFAULT_FLOOR:
        out["density_floor"] = density_floor()
    if smc_held_out():
        out["smc"] = "hold_out"
    if get("labels"):
        out["labels"] = get("labels")
    if n_nodes() != DEFAULTS["n_nodes"]:
        out["n_nodes"] = n_nodes()
    return out


# ------------------------------------------------------------------------------------------------ labels
def _recipe_recorded(connectomes_dir: Path | None) -> bool:
    if connectomes_dir is None or not Path(connectomes_dir).is_dir():
        return False
    for file in Path(connectomes_dir).glob("*_qc.json"):
        try:
            return "recipe_sha256" in json.loads(file.read_text())
        except (OSError, ValueError):
            return False
    return False


def labels_are_guarded(connectomes_dir: Path | None = None) -> bool:
    """True where real labels need the flag: a chain root, a recipe_v4 path, or connectomes with a recorded recipe hash."""
    if chain_root() is not None:
        return True
    if connectomes_dir is not None and "recipe_v4" in str(connectomes_dir):
        return True
    return _recipe_recorded(connectomes_dir)


def label_mode(connectomes_dir: Path | None = None) -> str:
    """'real' or 'shuffled'. Stops when the run has not said which, on any route but the legacy one."""
    mode = get("labels")
    if mode == "shuffled":
        if str(get("label_seed")).strip() == "":
            raise SystemExit("SC_LABELS=shuffled needs SC_LABEL_SEED: the seed is part of the record.")
        return "shuffled"
    if mode == "real":
        return "real"
    if labels_are_guarded(connectomes_dir):
        raise SystemExit(
            "LABEL GUARD: this run would join real diagnoses and no flag says it may.\n"
            "  SC_LABELS=shuffled SC_LABEL_SEED=<n>   permuted labels, for debugging\n"
            "  SC_LABELS=real                         real labels, once the recipe and the analysis plan are frozen"
        )
    return "real"


def clinical_columns(columns) -> list[str]:
    return [c for c in columns if _CLINICAL.match(str(c))]


def shuffled_donors(subject_ids, seed) -> dict[str, str]:
    """subject -> the subject whose clinical block they carry. Depends only on the seed and the sorted subject list."""
    import numpy as np

    ids = sorted(str(s) for s in subject_ids)
    perm = np.random.default_rng(int(seed)).permutation(len(ids))
    return {s: ids[int(k)] for s, k in zip(ids, perm)}


def apply_labels(master, connectomes_dir: Path | None, master_dir: Path):
    """Apply the label mode to the master cohort and record it. On the legacy route this changes and writes nothing."""
    import pandas as pd

    mode = label_mode(connectomes_dir)
    master_dir = Path(master_dir)
    if mode == "real":
        if labels_are_guarded(connectomes_dir):
            (master_dir / LABEL_MODE_FILE).write_text(
                json.dumps({"mode": "real", "flag": "SC_LABELS=real", "n": int(len(master))}, indent=1) + "\n"
            )
        return master

    seed = int(get("label_seed"))
    ids = master["subject_id"].astype(str).tolist()
    if len(set(ids)) != len(ids):
        raise SystemExit("shuffled labels need one master row per participant")
    donors = shuffled_donors(ids, seed)
    block = clinical_columns(master.columns)
    by_id = master.set_index(master["subject_id"].astype(str))[block]
    shuffled = master.copy()
    for col in block:
        shuffled[col] = [by_id.at[donors[s], col] for s in ids]
    changed = int((shuffled["group"].astype(str).to_numpy() != master["group"].astype(str).to_numpy()).sum())
    pd.DataFrame({"subject_id": sorted(donors), "donor_id": [donors[s] for s in sorted(donors)]}).to_csv(
        master_dir / DONOR_MAP_FILE, index=False
    )
    (master_dir / LABEL_MODE_FILE).write_text(
        json.dumps(
            {
                "mode": "shuffled",
                "seed": seed,
                "n": len(ids),
                "share_with_a_different_label": round(changed / max(len(ids), 1), 3),
                "columns": block,
                "donor_map_sha256": hashlib.sha256((master_dir / DONOR_MAP_FILE).read_bytes()).hexdigest(),
            },
            indent=1,
        )
        + "\n"
    )
    return shuffled


def donor_map(analysis_root: Path | None) -> dict[str, str] | None:
    """subject -> donor when this tree's labels are shuffled, else None."""
    if analysis_root is None:
        return None
    file = Path(analysis_root) / "00_master" / DONOR_MAP_FILE
    mode_file = Path(analysis_root) / "00_master" / LABEL_MODE_FILE
    if not mode_file.is_file() or json.loads(mode_file.read_text()).get("mode") != "shuffled":
        return None
    out = {}
    for line in file.read_text().splitlines()[1:]:
        subject, donor = line.split(",")
        out[subject] = donor
    return out


def hide_clinical_rows(df, id_col: str, analysis_root: Path | None):
    """Clinical rows read straight from the cohort exports, with labels shuffled: each participant gets their donor's rows.

    Participants outside the shuffled cohort lose their rows. With real labels the frame comes back untouched.
    """
    donors = donor_map(analysis_root)
    if donors is None:
        return df
    recipient_of = {d: s for s, d in donors.items()}
    out = df[df[id_col].astype(str).isin(recipient_of)].copy()
    out[id_col] = out[id_col].astype(str).map(recipient_of)
    return out


def write_cohort_record(master_dir: Path, considered: int, kept: int) -> None:
    """Beside a master cohort built with non-default settings, say so. A default build writes nothing new."""
    settings = non_default()
    if settings:
        record = dict(settings, n_considered=int(considered), n_kept=int(kept))
        (Path(master_dir) / SETTINGS_FILE).write_text(json.dumps(record, indent=1) + "\n")
