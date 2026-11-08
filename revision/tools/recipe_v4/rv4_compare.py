"""Recipe v4 bench: apply bench/RULE.md mechanically. Reads only bench/qc/*.json, the session list (no diagnosis column) and, for arm D, field-map images
if they exist. No diagnosis, group, CDR or MMSE is read. Density and edge counts are printed, never used. Sessions are shown as S01..S24, not by identifier.

Writes bench/compare.md and bench/compare.json. --final marks the report final (driver, after every run); without it the report says INTERIM.
Usage: rv4_compare.py [--out /data/derivatives/recipe_v4/bench] [--final]
"""
import argparse, glob, hashlib, json, os, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import binomtest

ap = argparse.ArgumentParser(); ap.add_argument("--out", default="/data/derivatives/recipe_v4/bench"); ap.add_argument("--final", action="store_true"); a = ap.parse_args()
B = Path(a.out); L = Path("/data/derivatives/recipe_v4/lists"); U = pd.read_csv(L / "bench_units.csv", dtype=str)[["unit", "subj", "phase", "vendor"]]
U["stratum"] = U.phase + " " + U.vendor; U["sid"] = [f"S{k + 1:02d}" for k in range(len(U))]; N = len(U)


def qc(armtag, unit):
    f = B / "qc" / f"{armtag}_{unit}.json"
    return json.load(open(f)) if f.is_file() else None


Q = {arm: {u: qc(arm, u) for u in U.unit} for arm in ["A", "B", "C", "B_seed2", "C_seed2", "B_10M", "C_10M"]}
ok = lambda d: d is not None and d.get("status") == "ok"
fin = lambda d: d is not None and d.get("status") in ("ok", "failed")
med = lambda x: float(np.median(x)) if len(x) else float("nan")
fm = lambda x, k=3: "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{k}f}"
PF = lambda b: "PASS" if b else "FAIL"
md = []; add = md.append
n_b, n_c = sum(fin(Q["B"][u]) for u in U.unit), sum(fin(Q["C"][u]) for u in U.unit); complete_1m = n_b == N and n_c == N
add(f"# Recipe v4 bench: {'final' if a.final else 'INTERIM'} comparison\n")
add(f"Written {time.strftime('%F %H:%M UTC', time.gmtime())}. RULE.md sha256 {hashlib.sha256(open(B / 'RULE.md', 'rb').read()).hexdigest()[:16]}. Finished at 1M: A {sum(fin(Q['A'][u]) for u in U.unit)}, B {n_b}, C {n_c} of {N}.\n")

# ---------------- adoption rule, C over B
rows = []
for u, sid, st, v in zip(U.unit, U.sid, U.stratum, U.vendor):
    b, c = Q["B"][u], Q["C"][u]
    rows.append(dict(sid=sid, stratum=st, vendor=v, b_ok=ok(b), c_ok=ok(c), c_failed=fin(c) and not ok(c), gain=(c["wm_dice"] - b["wm_dice"]) if ok(b) and ok(c) else np.nan,
                     gain_r=(c["wm_dice_regional"] - b["wm_dice_regional"]) if ok(b) and ok(c) else np.nan))
R = pd.DataFrame(rows); g = R.gain.dropna().values; gr = R.gain_r.dropna().values
nz = g[g != 0]; pos = int((nz > 0).sum()); p = float(binomtest(pos, len(nz), 0.5).pvalue) if len(nz) else 1.0
vend = R.dropna(subset=["gain"]).groupby("vendor").gain.median(); lost = int(R.c_failed.sum() + (R.gain < -0.03).sum())
fm_ids = [x for x in open(L / "fieldmap_bench_ids.txt").read().strip().split(",") if x]
fm_disk = sorted({p_ for i in fm_ids for s in U.subj for p_ in glob.glob(f"/data/Images/*/{s}/*/*/I{i}")}) + sorted({str(p_) for i in fm_ids for p_ in Path("/data/adni_downloads").rglob(f"I{i}")} if Path("/data/adni_downloads").is_dir() else [])
c1 = med(g) >= 0.02 and p < 0.05; c2 = med(gr) >= 0.02; c3 = lost <= 1; c4 = bool(len(vend)) and bool((vend >= 0).all())
armd = json.load(open(B / "armD.json")) if (B / "armD.json").is_file() else None     # {"median_r": .., "n": ..} from the arm D comparison; the rule: once field maps exist, D must have been read
c5 = None if not fm_disk else (armd is not None and armd["median_r"] >= 0.50 and armd["n"] >= 8)
adopt = bool(complete_1m and c1 and c2 and c3 and c4 and (c5 is None or c5))
add("## Adoption rule for distortion correction (C over B)\n")
add("| # | Condition | Value | Bar | Result |\n|---|---|---|---|---|")
add(f"| 1 | Median paired gain, whole-brain WM-to-FA>0.25 Dice; sign test | {fm(med(g))}; {pos}/{len(nz)} up, p={p:.4f} | >= +.02 and p < .05 | {PF(c1)} |")
add(f"| 2 | Median paired gain, same Dice in the orbitofrontal, temporal-pole, medial-temporal mask | {fm(med(gr))} | >= +.02 | {PF(c2)} |")
add(f"| 3 | Sessions where C fails or loses > .03 of whole-brain Dice | {lost} (failed {int(R.c_failed.sum())}) | <= 1 of {N} | {PF(c3)} |")
add(f"| 4 | Median gain by vendor | {', '.join(f'{k} {v:+.3f}' for k, v in vend.items()) or 'n/a'} | no median loss | {PF(c4)} |")
add(f"| 5 | Synb0 vs measured displacement (arm D) | {f'median r {armd['median_r']:.3f}, n={armd['n']}' if armd else f'arm D not run; {len(fm_disk)} of {len(fm_ids)} field-map series on disk'} | r >= .50, n >= 8, if D exists | {'not applicable' if not fm_disk else PF(c5) if armd else 'FIELD MAPS ON DISK: C cannot be adopted until arm D is read'} |\n")
arm = "C" if adopt else "B"
add(f"**Recipe under the rule: arm {arm}.** " + ("All conditions hold." if adopt else ("Not all 1M runs have finished, so C cannot be adopted yet." if not complete_1m else "Conditions 1-4 hold, but arm D has not been read, so C cannot be adopted yet." if c1 and c2 and c3 and c4 and fm_disk and not armd else "At least one condition fails, so the recipe is B and the absence of distortion correction is disclosed.")) + "\n")

# ---------------- go/no-go for the chosen arm
Qa = Q[arm]; good = [u for u in U.unit if ok(Qa[u])]; fails = [u for u in U.unit if fin(Qa[u]) and not ok(Qa[u])]; pending = N - len(good) - len(fails)
vals = lambda k, us: [Qa[u][k] for u in us]
acc, dic, emp = vals("acceptance", good), vals("wm_dice", good), vals("empty_nodes_154", good)
vmap = pd.read_csv(B / "variants.csv") if (B / "variants.csv").is_file() else pd.DataFrame(columns=["unit", "variant"])
u10 = list(vmap[vmap.variant == "10M"].unit); u2 = list(vmap[vmap.variant == "seed2"].unit)
q10 = {u: Q[f"{arm}_10M"][u] for u in u10}; q2 = {u: Q[f"{arm}_seed2"][u] for u in u2}
ten_ok = sum(ok(d) for d in q10.values()); ten_fin = sum(fin(d) for d in q10.values())


def seed_table(units):
    """Whole-brain r between the two seeds, per measure, over these sessions."""
    t = {}
    if len(units) < 3: return t
    for lab in ["FA", "MD", "RD", "AxD"]:
        nets = sorted(Qa[units[0]]["wb_net_values"][lab]); rs = []
        for n_ in nets:
            x = np.array([Qa[u]["wb_net_values"][lab][n_] for u in units]); y = np.array([q2[u]["wb_net_values"][lab][n_] for u in units])
            if x.std() > 0 and y.std() > 0: rs.append(float(np.corrcoef(x, y)[0, 1]))
        t[lab] = dict(median_r=med(rs), min_r=float(min(rs)) if rs else float("nan"), n_networks=len(rs),
                      global_r=float(np.corrcoef([Qa[u]["wb_global"][lab] for u in units], [q2[u]["wb_global"][lab] for u in units])[0, 1]))
    return t


g2 = [u for u in u2 if ok(q2[u]) and ok(Qa[u])]; rr = seed_table(g2)
seed_v = min(v["median_r"] for v in rr.values()) if rr else float("nan")
crit = [("Sessions failing after one retry", f"{len(fails)}" + (f" (+{pending} not finished)" if pending else ""), f"<= 2 of {N}", None if pending and len(fails) <= 2 else len(fails) <= 2),
        ("Acceptance, median", fm(med(acc)), ">= .40", med(acc) >= 0.40 if len(good) else None),
        ("Whole-brain WM Dice, median", fm(med(dic)), ">= .65", med(dic) >= 0.65 if len(good) else None),
        ("Empty nodes of 154, median", fm(med(emp), 1), "<= 2", med(emp) <= 2 if len(good) else None),
        (f"10M end to end, sessions finished (arm {arm})", f"{ten_ok} of {len(u10)}", "all 4", (ten_ok == 4) if ten_fin == len(u10) and len(u10) == 4 else None),
        (f"Second seed, whole-brain value r (arm {arm}, n={len(g2)})", fm(seed_v, 4), ">= .99", (seed_v >= 0.99 and len(g2) == 6) if len(g2) == 6 or all(fin(q2[u]) for u in u2) and u2 else None)]
add(f"## Go/no-go for the frozen recipe (arm {arm})\n")
add("| Criterion | Value | Bar | Result |\n|---|---|---|---|")
for c_, v_, b_, r_ in crit: add(f"| {c_} | {v_} | {b_} | {'PENDING' if r_ is None else PF(r_)} |")
add("")
# strata
add("By acquisition stratum (rule: if only one stratum fails, it moves to sensitivity):\n")
add("| Stratum | n ok / n | Fails | Acceptance | WM Dice | Regional Dice | Empty of 154 | Stratum result |\n|---|---|---|---|---|---|---|---|")
st_fail = []
for st, gu in U.groupby("stratum"):
    us = [u for u in gu.unit if ok(Qa[u])]; fl = [u for u in gu.unit if fin(Qa[u]) and not ok(Qa[u])]
    A, D, E = med(vals("acceptance", us)), med(vals("wm_dice", us)), med(vals("empty_nodes_154", us))
    res = (len(fl) <= 2 and A >= 0.40 and D >= 0.65 and E <= 2) if us else (False if fl and len(fl) == len(gu) else None)      # every session failed: a failed stratum, not a pending one
    if res is False: st_fail.append(st)
    add(f"| {st} | {len(us)} / {len(gu)} | {len(fl)} | {fm(A)} | {fm(D)} | {fm(med(vals('wm_dice_regional', us)))} | {fm(E, 1)} | {'PENDING' if res is None else PF(res)} |")
res_all = [c_[3] for c_ in crit]; st_of = dict(zip(U.unit, U.stratum))
if len(st_fail) == 1:                                    # the rule: that stratum moves to sensitivity, so the rest, its own variants included, is what gets judged
    keep = [u for u in U.unit if st_of[u] != st_fail[0]]; okk = [u for u in keep if ok(Qa[u])]; flk = [u for u in keep if fin(Qa[u]) and not ok(Qa[u])]
    k10 = [u for u in u10 if st_of[u] != st_fail[0]]; k2 = [u for u in u2 if st_of[u] != st_fail[0]]; g2k = [u for u in k2 if ok(q2[u]) and ok(Qa[u])]; rk = seed_table(g2k)
    if len(okk) + len(flk) < len(keep) or not all(fin(q10[u]) for u in k10) or not all(fin(q2[u]) for u in k2): verdict = "PENDING"
    else:
        rest_ok = len(flk) <= 2 and med(vals("acceptance", okk)) >= 0.40 and med(vals("wm_dice", okk)) >= 0.65 and med(vals("empty_nodes_154", okk)) <= 2 and all(ok(q10[u]) for u in k10) and len(g2k) == len(k2) and bool(rk) and min(v["median_r"] for v in rk.values()) >= 0.99
        verdict = f"GO, with {st_fail[0]} moved to sensitivity" if rest_ok else "NO-GO: no-ACT stays primary and the question closes"
elif a.final and not st_fail and any(r is False for r in res_all): verdict = "NO-GO: no-ACT stays primary and the question closes"     # what is left unfinished cannot rescue a criterion that has failed
elif any(r is None for r in res_all): verdict = "PENDING"
elif all(res_all) and not st_fail: verdict = "GO"
else: verdict = "NO-GO: no-ACT stays primary and the question closes"
add(f"\n**Verdict under the rule: {verdict}.**\n")

# ---------------- reference: arms side by side (reported, not criteria unless named above)
add("## Arms side by side at 1M, medians\n\nDensity is printed for the record; the rule says it is not a criterion.\n")
add("| Arm | ok/fail | Acceptance | WM Dice | Regional Dice | Empty of 154 | Density of 154 | FA along streamlines | BBR chosen/computed/runs | Abs motion mm | Rel motion mm | Outlier slices % |\n|---|---|---|---|---|---|---|---|---|---|---|---|")
for k in ["A", "B", "C"]:
    d = [Q[k][u] for u in U.unit if ok(Q[k][u])]; f_ = sum(fin(Q[k][u]) and not ok(Q[k][u]) for u in U.unit)
    gv = lambda key: med([x[key] for x in d])
    ed = [x["eddy"] for x in d if x.get("eddy")]
    add(f"| {k} | {len(d)}/{f_} | {fm(gv('acceptance'))} | {fm(gv('wm_dice'))} | {fm(gv('wm_dice_regional'))} | {fm(gv('empty_nodes_154'), 1)} | {fm(gv('density_154'))} | {fm(gv('fa_along_mean'))} | {sum(x['registration']['chosen'] == 'bbr' for x in d)}/{sum(x['registration'].get('dice_bbr') is not None for x in d)}/{len(d)} | {fm(med([e['abs_rms_mean'] for e in ed]), 2)} | {fm(med([e['rel_rms_mean'] for e in ed]), 2)} | {fm(med([e['outlier_pct'] for e in ed]), 2)} |")
add("\nPer session (B and C whole-brain Dice; gain; regional gain; reason for any failure):\n")
add("| Session | Stratum | B | C | Gain | Regional gain | Note |\n|---|---|---|---|---|---|---|")
for u, sid, st in zip(U.unit, U.sid, U.stratum):
    b, c, a_ = Q["B"][u], Q["C"][u], Q["A"][u]; note = []
    for tag, d in (("A", a_), ("B", b), ("C", c)):
        if fin(d) and not ok(d): note.append(f"{tag} failed at {d.get('failed_stage')}")
    add(f"| {sid} | {st} | {fm(b['wm_dice']) if ok(b) else '-'} | {fm(c['wm_dice']) if ok(c) else '-'} | {fm(c['wm_dice'] - b['wm_dice']) if ok(b) and ok(c) else '-'} | {fm(c['wm_dice_regional'] - b['wm_dice_regional']) if ok(b) and ok(c) else '-'} | {'; '.join(note)} |")

# ---------------- second seed
add(f"\n## Second seed (arm {arm}, 1M, n={len(g2)} of {len(u2)})\n")
if rr:
    add("Pearson r across sessions of the whole-brain network value (definition of p5_features/p5_act: per network, per measure, on 154 nodes). Criterion: lowest per-measure median.\n")
    add("| Measure | Median r over networks | Lowest network r | r of the all-edge mean |\n|---|---|---|---|")
    for lab, v in rr.items(): add(f"| {lab} | {fm(v['median_r'], 4)} | {fm(v['min_r'], 4)} | {fm(v['global_r'], 4)} |")
    vb = [q2[u]["vs_base"] for u in g2 if q2[u].get("vs_base")]
    if vb: add(f"\nFor information, session by session: log-weight r on shared edges, median {fm(med([x['log_fd_r_on_shared_edges'] for x in vb]))}.")
else:
    add("Not finished.")

# ---------------- 10M
add(f"\n## 10M runs (arm {arm})\n\n| Session | Stratum | Status | Hours | Threads | Peak work disk GB | Acceptance | Empty of 154 | Density of 154 |\n|---|---|---|---|---|---|---|---|---|")
sid_of = dict(zip(U.unit, U.sid))
for u in u10:
    d = q10[u]
    if d is None: add(f"| {sid_of[u]} | {st_of[u]} | not started | | | | | | |"); continue
    sec = sum(v["sec"] for k, v in d["stages"].items() if k.startswith(("s6", "s7"))); pk = max((v["peak_gb"] for k, v in d["stages"].items() if k.startswith("s6")), default=0)
    add(f"| {sid_of[u]} | {st_of[u]} | {d['status']} | {sec / 3600:.1f} | {d['threads']} | {pk:.1f} | {fm(d.get('acceptance'))} | {d.get('empty_nodes_154', '')} | {fm(d.get('density_154'))} |")

# ---------------- time and disk
add("\n## Time and disk by stage (finished runs, first computation of each stage only)\n")
rows = []
for k in ["A", "B", "C"]:
    for u, st in zip(U.unit, U.stratum):
        d = Q[k][u]
        if d is None: continue
        for s_, v in d["stages"].items():
            if not (v.get("reused") and v.get("shared")): rows.append(dict(arm=k, stratum=st, stage=s_, sec=v["sec"], peak=v["peak_gb"], rss=v.get("rss_so_far_gb", np.nan)))   # a stage kept from an earlier attempt of this arm counts; one built by another arm does not
if rows:
    T_ = pd.DataFrame(rows); t = T_.groupby(["stage"]).agg(n=("sec", "size"), sec_median=("sec", "median"), sec_max=("sec", "max"), peak_gb_max=("peak", "max"), rss_gb_max=("rss", "max")).round(1).reset_index()
    add("| Stage | Runs | Median s | Max s | Peak GB (largest) | Largest process RSS so far GB |\n|---|---|---|---|---|---|")
    for r in t.itertuples(): add(f"| {r.stage} | {r.n} | {r.sec_median:.0f} | {r.sec_max:.0f} | {r.peak_gb_max} | {r.rss_gb_max} |")
    add("\nPer stratum, median seconds per stage:\n")
    pv = T_.pivot_table(index="stage", columns="stratum", values="sec", aggfunc="median").round(0)
    add("| Stage | " + " | ".join(pv.columns) + " |\n|---" * 1 + "|---" * len(pv.columns) + "|")
    for s_, r in pv.iterrows(): add(f"| {s_} | " + " | ".join("" if np.isnan(x) else f"{x:.0f}" for x in r.values) + " |")
add("\n## Facts about the inputs that the rule does not test\n")
acq = [d["acq"] for u in U.unit for d in [Q["B"][u]] if ok(d)]
if acq:
    add(f"- Polarity of phase encoding not stored (axis only, + used): {sum(not x['pe_polarity_known'] for x in acq)} of {len(acq)} sessions. Readout time an estimate from the water-fat shift: {sum('Estimated' in x['readout_source'] for x in acq)}.")
    add(f"- Zero-filled in-plane (reconstruction matrix larger than acquisition matrix): {sum(bool(x['zero_filled']) for x in acq)} of {len(acq)}; returned to the acquired lines before dwidenoise and mrdegibbs (RULE.md, change of 3 Nov): {sum(bool(x.get('cropped')) for x in acq)}.")
    mixed = sorted({sid_of[u] for k in "BC" for u in U.unit if ok(Q[k][u]) and Q[k][u]["acq"]["zero_filled"] and not Q[k][u]["acq"].get("cropped")})
    if mixed: add(f"- **Zero-filled sessions with a run that skipped that step, which the rule no longer allows: {', '.join(mixed)}. These results cannot be used as they stand.**")
    add(f"- Partial Fourier recorded in the sidecar (mrdegibbs documents reduced validity): {sum(x['partial_fourier'] is not None for x in acq)} of {len(acq)}.")
    add(f"- Multi-shell in this sample: {sum(bool(x['multishell']) for x in acq)} of {len(acq)}.")
open(B / "compare.md", "w").write("\n".join(md) + "\n")
json.dump(dict(final=a.final, recipe_arm=arm, adopt_c=adopt, complete_1m=complete_1m, verdict=verdict, median_gain=med(g), median_gain_regional=med(gr), sign_p=p, lost=lost,
               go_nogo={c_[0]: dict(value=c_[1], bar=c_[2], result=c_[3]) for c_ in crit}, strata_failing=st_fail, second_seed=rr), open(B / "compare.json", "w"), indent=1, default=float)
print("compare:", "final" if a.final else "interim", "| arm", arm, "|", verdict)
