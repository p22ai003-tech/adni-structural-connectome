"""Phase 1 step 3b — assemble the per-condition engine outputs into effects_long.csv, sr_lr_summary.csv and
condition_notes.json (canonical condition order), then move the per-condition files to phase1/runs/."""
import sys as _sys; from pathlib import Path as _Path; _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import iit_chain as chain; chain.boot()      # IIT_ROOT and the chain settings; no effect when IIT_ROOT is unset
import os
import json, shutil
from pathlib import Path
import pandas as pd

O = Path(os.environ.get("IIT_P1", "/home/ec2-user/exp/docs/iit_dipanjan/phase1")); RUNS = O / "runs"; RUNS.mkdir(exist_ok=True)
ORDER = ["C0", "C0q", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10", "C11"]
eff, srl, notes = [], [], {}
for c in ORDER:
    for src in (O, RUNS):
        f = src / f"effects_long_{c}.csv"
        if f.is_file():
            eff.append(pd.read_csv(f)); srl.append(pd.read_csv(src / f"sr_lr_summary_{c}.csv"))
            notes.update(json.load(open(src / f"condition_notes_{c}.json"))); break
    else:
        print(f"WARNING: no engine output for {c} (condition skipped; see engine_{c}.log)")
pd.concat(eff, ignore_index=True).to_csv(O / "effects_long.csv", index=False)
pd.concat(srl, ignore_index=True).to_csv(O / "sr_lr_summary.csv", index=False)
json.dump(notes, open(O / "condition_notes.json", "w"), indent=1)
for c in ORDER:
    for stem in ("effects_long", "sr_lr_summary", "condition_notes"):
        f = O / f"{stem}_{c}.{'json' if stem == 'condition_notes' else 'csv'}"
        if f.is_file(): shutil.move(f, RUNS / f.name)
print("merged", len(ORDER), "conditions")
