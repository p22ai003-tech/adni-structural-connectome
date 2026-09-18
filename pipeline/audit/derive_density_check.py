#!/usr/bin/env python
"""Derive density_check.json from subject_manifest.csv (the single source of truth).
Feeds the dashboard's S3 m/med + tck/tsf/csv/sift E/S columns. EC2 file presence comes from the manifest
stage flags (tck/tsf/sift) + connectome presence (conn_location) for csv; S3 from the s3_* flags.
Connectomes are mirrored EC2<->S3, so s3 density == ec2 density over good subjects.
Writes to the PERSISTENT qc dir (survives reboot) — not /tmp."""
import csv, json, time, statistics

MAN = "/data/derivatives/qc/sc_matrix_qc/subject_manifest.csv"
OUT = "/data/derivatives/qc/sc_matrix_qc/density_check.json"
rows = list(csv.DictReader(open(MAN)))

def block(group):
    good = [r for r in rows if (group == "overall" or r["group"] == group) and r["band"] == "good"]
    ds = [float(r["density"]) for r in good if r["density"]]
    em = round(statistics.mean(ds), 3) if ds else 0
    emd = round(statistics.median(ds), 3) if ds else 0
    def ec2(r, t):
        return int(r.get(t, 0) or 0) if t != "csv" else int(bool(r.get("conn_location")))
    files = {t: {"ec2": sum(ec2(r, t) for r in good),
                 "s3": sum(int(r.get("s3_" + t, 0) or 0) for r in good)} for t in ("tck", "tsf", "csv", "sift")}
    return {"ec2": {"n": len(ds), "mean": em, "median": emd},
            "s3": {"n": len(ds), "mean": em, "median": emd}, "files": files}

out = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
for g in ("overall", "AD", "MCI", "CN"):
    out[g] = block(g)
json.dump(out, open(OUT, "w"), indent=2)
print(f"wrote {OUT}")
for g in ("overall", "AD", "MCI", "CN"):
    f = out[g]["files"]
    print(f"  {g:<8} ec2 m/med {out[g]['ec2']['mean']}/{out[g]['ec2']['median']} | "
          f"tck {f['tck']['ec2']}/{f['tck']['s3']} csv {f['csv']['ec2']}/{f['csv']['s3']} "
          f"tsf {f['tsf']['ec2']}/{f['tsf']['s3']} sift {f['sift']['ec2']}/{f['sift']['s3']}")
