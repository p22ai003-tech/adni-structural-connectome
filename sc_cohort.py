#!/usr/bin/env python
"""Build and check the cohort tables the analysis reads.

    python sc_cohort.py build --exports ~/Downloads/adni_exports --out cohort/
    python sc_cohort.py check                     # checks $SC_COHORT_DIR (default: cohort/)

What goes in
------------
Three CSV files you download from the ADNI image archive (IDA, "Advanced Image
Search", export as CSV). Put them in one folder:

    dti_master.csv   search: Modality = DTI.  Every DTI image of every subject.
    mri_master.csv   search: Modality = MRI.  Every MRI image of every subject.
    all_mri.csv      search: Modality = MRI, all descriptions (the T1 candidates).
                     Optional; see below.

Tick at least these result columns when you export: Subject ID, Phase, Sex,
Research Group, Visit, Study Date, Age, Image ID, Description, Type,
MMSE Total Score, Global CDR. APOE A1/A2, Weight, FAQ, GDSCALE and NPI-Q are
used when present.

What comes out
--------------
    dti.csv          one row per subject: the DTI image the analysis uses
    mri.csv          one row per subject: the T1 image
    dti_master.csv   copied through; the ML stages read clinical scores from it
    mri_master.csv   copied through, same reason

The selection rule
------------------
This reproduces the published cohort tables byte for byte; nothing else in the
repository ever wrote them.

    dti.csv  the LAST row per Subject ID in dti_master.csv, in file order, with
             Study Date rewritten as an ISO date.
    mri.csv  rows of all_mri.csv whose subject is in dti.csv and whose
             Description is in configs/cohort_rules/Ranked_OK_T1.csv; per
             subject the lowest rank wins (1 = best), ties going to the row
             that comes first in the file. The ranking itself is a hand-made
             list of acceptable T1 descriptions, derived from the patterns in
             adni_t1_coreg_whitelist.csv.

The analysis reads only the Subject ID column of mri.csv, so without
all_mri.csv a header-only mri.csv is written and the results are unchanged.

Diagnostic groups: EMCI, LMCI and SMC are grouped with MCI, and only CN, MCI
and AD are analysed (connectome_analysis/analysis_cohort.py, recode_group).
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import sc_config  # noqa: E402

RULES = PROJECT_ROOT / "configs" / "cohort_rules"
RANKING = RULES / "Ranked_OK_T1.csv"

# Columns a consumer actually reads; a missing one fails somewhere in the run.
REQUIRED = {
    "dti.csv": ["Subject ID", "Research Group", "Image ID", "Study Date", "Phase", "Sex", "Age"],
    "mri.csv": ["Subject ID"],
    "dti_master.csv": ["Subject ID", "Study Date", "MMSE Total Score", "Global CDR"],
    "mri_master.csv": ["Subject ID", "Study Date", "MMSE Total Score", "Global CDR"],
}
USED_IF_PRESENT = {
    "dti.csv": ["MMSE Total Score", "Global CDR", "APOE A1", "APOE A2", "Weight",
                "FAQ Total Score", "GDSCALE Total Score", "NPI-Q Total Score"],
}


def _read(path: Path):
    import pandas as pd
    return pd.read_csv(path, low_memory=False)


def build(exports: Path, out: Path) -> int:
    import pandas as pd

    dm_path = exports / "dti_master.csv"
    mm_path = exports / "mri_master.csv"
    for p in (dm_path, mm_path):
        if not p.is_file():
            print(f"missing: {p}\n(see `python sc_cohort.py --help` for what to download)")
            return 2
    out.mkdir(parents=True, exist_ok=True)

    dm = _read(dm_path)
    dm["Study Date"] = pd.to_datetime(dm["Study Date"], format="%m/%d/%Y")
    dti = dm.drop_duplicates("Subject ID", keep="last")
    dti.to_csv(out / "dti.csv")           # the index column is part of the published format
    print(f"dti.csv         {len(dti):>6} subjects  (from {len(dm)} DTI rows)")

    all_mri = exports / "all_mri.csv"
    if all_mri.is_file():
        am = _read(all_mri)
        ranked = pd.read_csv(RANKING)
        rank = dict(zip(ranked["description"], ranked["rank"]))
        cand = am[am["Subject ID"].isin(dti["Subject ID"]) & am["Description"].isin(rank)].copy()
        cand["rank"] = cand["Description"].map(rank)
        cand["_i"] = cand.index
        best = cand.sort_values(["rank", "_i"]).drop_duplicates("Subject ID").sort_values("_i")
        best.drop(columns="_i").to_csv(out / "mri.csv")
        print(f"mri.csv         {len(best):>6} subjects  (ranked T1 choice)")
        lost = len(set(dti["Subject ID"]) - set(best["Subject ID"]))
        if lost:
            print(f"                {lost} DTI subject(s) have no acceptable T1 description")
    else:
        (out / "mri.csv").write_text("Subject ID\n")
        print("mri.csv         header only (no all_mri.csv given; the analysis reads only Subject ID)")

    for src, name in ((dm_path, "dti_master.csv"), (mm_path, "mri_master.csv")):
        if src.resolve() != (out / name).resolve():
            shutil.copy2(src, out / name)
        print(f"{name:<15} copied")
    print(f"\nwrote {out}")
    print("next:  python sc_cohort.py check" + ("" if out == sc_config.paths().cohort_dir
                                               else f"   (with SC_COHORT_DIR={out})"))
    return 0


def check(folder: Path) -> int:
    print(f"cohort folder: {folder}\n")
    problems = 0
    for name, cols in REQUIRED.items():
        path = folder / name
        if not path.is_file():
            print(f"  MISSING  {name}")
            problems += 1
            continue
        df = _read(path)
        missing = [c for c in cols if c not in df.columns]
        extra = [c for c in USED_IF_PRESENT.get(name, []) if c not in df.columns]
        dup = ""
        if name in ("dti.csv", "mri.csv") and len(df) and "Subject ID" in df.columns:
            n_dup = int(df["Subject ID"].duplicated().sum())
            if n_dup:
                dup = f"; {n_dup} duplicate subject row(s) -- each subject must appear once"
                problems += 1
        if missing:
            print(f"  BAD      {name:<16} {len(df):>6} rows; missing required column(s): {', '.join(missing)}")
            problems += 1
        else:
            print(f"  ok       {name:<16} {len(df):>6} rows{dup}")
        if extra:
            print(f"           optional columns absent: {', '.join(extra)}")
    print(f"\n{'no problems found.' if not problems else f'{problems} problem(s).'}")
    return 1 if problems else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="build the cohort tables from ADNI exports")
    b.add_argument("--exports", type=Path, required=True,
                   help="folder holding dti_master.csv, mri_master.csv and optionally all_mri.csv")
    b.add_argument("--out", type=Path, default=None, help="default: $SC_COHORT_DIR (cohort/)")
    c = sub.add_parser("check", help="check the cohort tables are complete")
    c.add_argument("--dir", type=Path, default=None, help="default: $SC_COHORT_DIR (cohort/)")
    args = ap.parse_args(argv)
    if args.cmd == "build":
        return build(args.exports, args.out or sc_config.paths().cohort_dir)
    return check(args.dir or sc_config.paths().cohort_dir)


if __name__ == "__main__":
    raise SystemExit(main())
