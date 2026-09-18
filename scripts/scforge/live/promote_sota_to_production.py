#!/usr/bin/env python3
"""Promote SOTA-route Claude-repaired connectomes into production (parallel).

For every subject the gated SOTA route PROMOTED, this:
  1. assembles the full required weight set on the NEW 1mm-world-space 166-node
     atlas -- REUSING the weights the runner already computed
     (count, fd_sum, count_invnodevol, len_mean) and generating only the missing
     ones (invlen_mean, fa/md/rd/ad_mean) via tck2connectome + the existing .tsf,
  2. remaps each 166-node matrix to the production 170-node convention
     (gap labels 35/36/81/82 stay zero) so the app/analysis is unchanged,
  3. backs up the existing production matrix ONCE (never overwrites a prior backup,
     so the true original is preserved across re-runs),
  4. writes the repaired matrices into production,
  5. records the subject in claude_repaired_manifest.csv (drives the [cl] tag).

Idempotent, reversible, parallel.
"""
from __future__ import annotations
import argparse, csv, glob, json, os, shutil, subprocess, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np

ROOT = Path("/home/ec2-user/exp")
DERIV = Path("/data/derivatives")
MRBIN = Path.home() / "mrtrix3" / "bin"
PROD = DERIV / "connectomes"
NODE_MAP = ROOT / "atlas" / "AAL" / "aal3_node_map_166.csv"
MANIFEST = DERIV / "qc" / "sc_matrix_qc" / "claude_repaired_manifest.csv"
# single canonical backup dir reused across runs (backup-once guard protects originals)
BACKUP = DERIV / "connectomes_backup_sota_20260612T082420Z"

# weight suffix -> tck2connectome args used when the runner did NOT already emit it
GEN_ARGS = {
    "invlen_mean": ["-scale_invlength", "-stat_edge", "mean"],
    "fa_mean": ["-scale_file", "{fa}", "-stat_edge", "mean"],
    "md_mean": ["-scale_file", "{md}", "-stat_edge", "mean"],
    "ad_mean": ["-scale_file", "{ad}", "-stat_edge", "mean"],
    "rd_mean": ["-scale_file", "{rd}", "-stat_edge", "mean"],
}
# weights the runner already wrote (SC_AAL3cc_<sid>_<suffix>.csv) -> reuse by remap
REUSE = ("count", "fd_sum", "count_invnodevol", "len_mean")
ALL_WEIGHTS = REUSE + tuple(GEN_ARGS)


def utc():
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def load_node_map():
    return {int(r["new_id"]): int(r["orig_value"]) for r in csv.DictReader(open(NODE_MAP))}


def remap_166_to_170(m, nm):
    out = np.zeros((170, 170), dtype=m.dtype)
    idx = [nm[i + 1] - 1 for i in range(166)]
    for a in range(166):
        out[np.ix_([idx[a]], idx)] = m[a]
    return out


def latest_promotions(run_glob):
    best = {}
    for rj in glob.glob(run_glob):
        try:
            r = json.load(open(rj))
        except Exception:
            continue
        if r.get("status") != "ok" or not (r.get("decision", {}) or {}).get("promote"):
            continue
        sid = r["sid"]; mt = os.path.getmtime(rj)
        if sid not in best or mt > best[sid][1]:
            best[sid] = (os.path.dirname(rj), mt, {"aal_qc": r.get("aal_qc", {}),
                                                   "baseline_qc": r.get("baseline_qc", {}),
                                                   "reg_ncc": r.get("reg_ncc", "")})
    return best


NM = load_node_map()


def promote_one(sid, sdir, meta):
    sdir = Path(sdir)
    parc = sdir / "aal3_166_dwi.mif"
    rconn = sdir / "connectomes"
    tdir = DERIV / "tracks" / sid
    tck = tdir / "tracks_final_3000k.tck"; sift = tdir / "sift_weights.txt"
    tsf = {k: str(tdir / f"{k}_mean.tsf") for k in ("fa", "md", "ad", "rd")}
    if not (parc.exists() and tck.exists()):
        return {"sid": sid, "ok": False, "reason": "missing parc/tracks"}
    wrote = []
    for suffix in ALL_WEIGHTS:
        out_csv = PROD / f"SC_AAL_{sid}_{suffix}.csv"
        # obtain the 166-node matrix: reuse runner output or generate
        if suffix in REUSE:
            src = rconn / f"SC_AAL3cc_{sid}_{suffix}.csv"
            if not src.exists():
                continue
            m = np.loadtxt(src, delimiter=",")
        else:
            args = [a.format(**tsf) for a in GEN_ARGS[suffix]]
            if "-scale_file" in args:
                f = args[args.index("-scale_file") + 1]
                if not Path(f).exists():
                    continue
            tmp = rconn / f"_gen_{suffix}.csv"
            cmd = [str(MRBIN / "tck2connectome"), str(tck), str(parc), str(tmp),
                   "-symmetric", "-zero_diagonal", "-assignment_radial_search", "4",
                   "-nthreads", "3", "-force", *args]
            p = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if p.returncode != 0 or not tmp.exists():
                continue
            m = np.loadtxt(tmp, delimiter=",")
        if m.shape != (166, 166):
            continue
        m170 = remap_166_to_170(m, NM)
        # backup ONCE (never overwrite an existing original backup)
        bkp = BACKUP / out_csv.name
        if out_csv.exists() and not bkp.exists():
            shutil.copy2(out_csv, bkp)
        np.savetxt(out_csv, m170, delimiter=",", fmt="%.6g")
        wrote.append(suffix)
    aq = meta["aal_qc"]; bq = meta["baseline_qc"]
    return {"sid": sid, "ok": True, "tag": "cl", "weights_written": ";".join(wrote),
            "old_density": round(bq.get("density", 0), 4), "new_density": round(aq.get("density", 0), 4),
            "old_zero_rows": bq.get("zero_rows", ""), "new_zero_rows": aq.get("zero_rows", ""),
            "reg_ncc": meta.get("reg_ncc", ""), "promoted_utc": utc()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-glob", default=str(DERIV / "qc/sc_matrix_qc/route_sota*/*/result.json"))
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    BACKUP.mkdir(parents=True, exist_ok=True)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    promos = latest_promotions(args.run_glob)
    print(f"[{utc()}] promoting {len(promos)} subjects | backup={BACKUP} | workers={args.workers}", flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(promote_one, sid, sdir, meta): sid for sid, (sdir, _, meta) in promos.items()}
        done = 0
        for fut in as_completed(futs):
            r = fut.result(); done += 1
            if r.get("ok"):
                rows.append(r)
            if done % 25 == 0:
                print(f"  ... {done}/{len(promos)} ({len(rows)} ok)", flush=True)
    fields = ["sid", "tag", "weights_written", "old_density", "new_density",
              "old_zero_rows", "new_zero_rows", "reg_ncc", "promoted_utc"]
    # merge with the existing manifest so scoped/per-group promotions ACCUMULATE
    # (each lane writes only its own rows; without merge they clobber prior groups'
    # [cl] tags). Existing rows are kept; rows for re-promoted sids are refreshed.
    merged = {}
    if MANIFEST.exists():
        for r in csv.DictReader(open(MANIFEST)):
            merged[r["sid"]] = {k: r.get(k, "") for k in fields}
    for r in rows:
        merged[r["sid"]] = {k: r.get(k, "") for k in fields}
    with open(MANIFEST, "w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=fields)
        w.writeheader()
        for sid in sorted(merged):
            w.writerow(merged[sid])
    print(f"[{utc()}] DONE: promoted {len(rows)} this run; manifest now {len(merged)} total -> {MANIFEST}", flush=True)


if __name__ == "__main__":
    main()
