"""Phase 1 step 4c (rebuilt connectomes only) — Figure P1-4: published against uniformly rebuilt connectomes.

  left    share of connected edges carrying diffusion values, by group (published / rebuilt)
  middle  median edge length by published processing route (published / rebuilt)
  right   whole-brain FA per participant, published against rebuilt, coloured by published route

Primary sample of the rebuilt run (ADNI-3/4, passing quality control). Reads both audits (IIT_P1_COMPARE and IIT_P1).
"""
import sys as _sys; from pathlib import Path as _Path; _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import iit_chain as chain; chain.boot()      # IIT_ROOT and the chain settings; no effect when IIT_ROOT is unset
import os
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

O = Path(os.environ["IIT_P1"]); A1 = Path(os.environ.get("IIT_P1_COMPARE", "/home/ec2-user/exp/docs/iit_dipanjan/phase1"))
cov = pd.read_csv(O / "covariates.csv").set_index("subject_id")
qc = pd.read_csv(O / "qc.csv", index_col=0).reindex(cov.index)
new = pd.read_csv(O / "audit/audit_participants.csv", index_col=0).reindex(cov.index)
old = pd.read_csv(A1 / "audit/audit_participants.csv", index_col=0).reindex(cov.index)
P = cov.index[(chain.primary(cov) == 1) & (qc.qc_fail == 0)]
route = qc.loc[P, "route"] if chain.root() is None else pd.read_csv(A1 / "qc.csv", index_col=0).route.reindex(P)   # the earlier analysis's route
GCOL = {"CN": "#38bdf8", "MCI": "#84cc16", "AD": "#fb7185"}
RCOL = {"legacy_step7": "#a855f7", "legacy_r4": "#f59e0b", "new": "#2563eb"}
RLAB = {"legacy_step7": "legacy route A", "legacy_r4": "legacy route B", "new": "current route"}
rng = np.random.default_rng(1)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
ax = axes[0]
for gi, g in enumerate(["CN", "MCI", "AD"]):
    ids = P[cov.loc[P, "group"] == g]
    for k, (src, lab) in enumerate([(old, "published"), (new, "rebuilt")]):
        v = 100 * src.loc[ids, "md_coverage"].dropna(); x = gi * 2.6 + k
        ax.scatter(x + rng.uniform(-0.3, 0.3, len(v)), v, s=6, color=GCOL[g], alpha=0.55 if k == 0 else 0.9, lw=0)
        ax.plot([x - 0.38, x + 0.38], [v.median()] * 2, color="#111827", lw=2)
ax.set_xticks([gi * 2.6 + k for gi in range(3) for k in range(2)])
ax.set_xticklabels([f"{g}\n{lab}" for g in ["CN", "MCI", "AD"] for lab in ("published", "rebuilt")], fontsize=8.5)
ax.set_ylabel("connected edges carrying diffusion values (%)"); ax.set_title("Microstructure coverage", fontsize=10)

ax = axes[1]
for ri, r in enumerate(["legacy_step7", "legacy_r4", "new"]):
    ids = route.index[route == r]
    for k, src in enumerate([old, new]):
        v = src.loc[ids, "len_median"].dropna(); x = ri * 2.6 + k
        ax.scatter(x + rng.uniform(-0.3, 0.3, len(v)), v, s=6, color=RCOL[r], alpha=0.55 if k == 0 else 0.9, lw=0)
        ax.plot([x - 0.38, x + 0.38], [v.median()] * 2, color="#111827", lw=2)
ax.set_xticks([ri * 2.6 + k for ri in range(3) for k in range(2)])
SHORT = {"legacy_step7": "legacy A", "legacy_r4": "legacy B", "new": "current"}
ax.set_xticklabels([f"{SHORT[r]}\n{lab}" for r in ["legacy_step7", "legacy_r4", "new"] for lab in ("published", "rebuilt")], fontsize=8.5)
ax.set_ylabel("median edge length per participant (mm)"); ax.set_title("Edge length by published processing route", fontsize=10)

ax = axes[2]
for r in ["legacy_step7", "legacy_r4", "new"]:
    ids = route.index[route == r]
    ax.scatter(old.loc[ids, "wb_fa_median"], new.loc[ids, "wb_fa_median"], s=10, color=RCOL[r], alpha=0.8, lw=0, label=RLAB[r])
lo = np.nanmin([old.loc[P, "wb_fa_median"].min(), new.loc[P, "wb_fa_median"].min()]) * 0.95
hi = np.nanmax([old.loc[P, "wb_fa_median"].max(), new.loc[P, "wb_fa_median"].max()]) * 1.02
ax.plot([lo, hi], [lo, hi], color="#9ca3af", ls="--", lw=1); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
ax.set_xlabel("whole-brain FA, published connectomes"); ax.set_ylabel("whole-brain FA, rebuilt connectomes")
ax.set_title("Whole-brain FA per participant", fontsize=10); ax.legend(fontsize=8, title="published route", title_fontsize=8)
plt.tight_layout(); plt.savefig(O / "fig_p1_rebuild.png", dpi=300); plt.close()
print("wrote", O / "fig_p1_rebuild.png")
