"""Tests of the chain's guards, on invented tables in a temporary folder. No participant data is read.

    tools/.venv/bin/python -I tools/test_chain_guards.py
"""
import json, os, subprocess, sys, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
PY = sys.executable
RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok)))
    print(f"{'pass' if ok else 'FAIL'}  {name}" + (f"  [{detail}]" if detail and not ok else ""))


def run(args, env=None, cwd=None):
    e = {k: v for k, v in os.environ.items() if not k.startswith(("IIT_", "SC_"))}
    e.update(env or {})
    p = subprocess.run([PY, "-I", str(HERE / "iit_chain.py"), *args], capture_output=True, text=True, env=e, cwd=cwd)
    return p.returncode, p.stdout + p.stderr


def snippet(code, env=None):
    e = {k: v for k, v in os.environ.items() if not k.startswith(("IIT_", "SC_"))}
    e.update(env or {})
    head = f"import sys; sys.path.insert(0, {str(HERE)!r}); sys.path.append({str(HERE.parents[2])!r})\n"
    p = subprocess.run([PY, "-I", "-c", head + code], capture_output=True, text=True, env=e)
    return p.returncode, p.stdout + p.stderr


with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    # ---------------------------------------------------------------- the signed recipe
    conn = tmp / "conn"; conn.mkdir()
    signed = tmp / "SIGNED_RECIPE.json"
    code, out = run(["check-recipe", str(conn), str(signed)])
    check("recipe: a missing signed file refuses", code == 3 and "does not exist" in out, out)
    signed.write_text(json.dumps({"recipe_sha256": "a" * 64}))
    code, out = run(["check-recipe", str(conn), str(signed)])
    check("recipe: connectomes without qc.json refuse", code == 3, out)
    (conn / "SC_AAL166_X_qc.json").write_text(json.dumps({"recipe": "no hash"}))
    code, out = run(["check-recipe", str(conn), str(signed)])
    check("recipe: a qc.json with no recorded hash refuses", code == 3 and "none recorded" in out, out)
    (conn / "SC_AAL166_X_qc.json").write_text(json.dumps({"recipe_sha256": "b" * 64}))
    code, out = run(["check-recipe", str(conn), str(signed)])
    check("recipe: a different hash refuses", code == 3 and "bbbbbbbbbbbb" in out, out)
    (conn / "SC_AAL166_X_qc.json").write_text(json.dumps({"recipe_sha256": "a" * 64}))
    (conn / "SC_AAL166_Y_qc.json").write_text(json.dumps({"recipe_sha256": "b" * 64}))
    code, out = run(["check-recipe", str(conn), str(signed)])
    check("recipe: one odd connectome among signed ones refuses", code == 3, out)
    (conn / "SC_AAL166_Y_qc.json").write_text(json.dumps({"recipe_sha256": "a" * 64}))
    code, out = run(["check-recipe", str(conn), str(signed)])
    check("recipe: matching hashes pass", code == 0, out)
    import hashlib
    whole = hashlib.sha256(signed.read_bytes()).hexdigest()           # what rv4_subject.py records: the signature file's own hash
    for u in "XY": (conn / f"SC_AAL166_{u}_qc.json").write_text(json.dumps({"recipe_sha256": whole}))
    code, out = run(["check-recipe", str(conn), str(signed)])
    check("recipe: the signature file's own hash passes", code == 0, out)
    (conn / "SC_AAL166_Y_qc.json").write_text(json.dumps({"recipe_sha256": "a" * 64}))
    code, out = run(["check-recipe", str(conn), str(signed)])
    check("recipe: a mix of the two hashes refuses", code == 3, out)
    signed.write_text(json.dumps({"recipe_sha256": "a" * 64, "changed": 1}))
    (conn / "SC_AAL166_Y_qc.json").write_text(json.dumps({"recipe_sha256": whole}))
    code, out = run(["check-recipe", str(conn), str(signed)])
    check("recipe: the hash of an earlier signature file refuses", code == 3, out)
    signed.write_text(json.dumps({"recipe_sha256": "a" * 64}))
    for u in "XY": (conn / f"SC_AAL166_{u}_qc.json").write_text(json.dumps({"recipe_sha256": "a" * 64}))
    code, out = run(["guard", "test"], env={"IIT_CONN": str(conn)})
    check("recipe: the launcher guard refuses with the real path while unsigned", code == 3, out)
    code, out = run(["guard", "test"], env={"IIT_REHEARSAL": "1"})
    check("recipe: a rehearsal without its own root refuses", code == 3 and "root of its own" in out, out)

    # ---------------------------------------------------------------- a recipe run's matrices, as the chain reads them
    import numpy as np
    np.savez_compressed(tmp / "m.npz", count=np.eye(3), fd_sum=np.eye(3) * 2, len_mean=np.eye(3) * 3)
    (tmp / "q.json").write_text(json.dumps({"recipe_sha256": "a" * 64, "wm_dice": 0.7}))
    code, out = run(["export-matrices", str(tmp / "m.npz"), str(tmp / "q.json"), str(tmp / "exp"), "TEST_UNIT"])
    check("export: csv per kind and the qc.json", code == 0 and (tmp / "exp/SC_AAL166_TEST_UNIT_fd_sum.csv").is_file() and (tmp / "exp/SC_AAL166_TEST_UNIT_qc.json").is_file(), out)
    code, out = run(["check-recipe", str(tmp / "exp"), str(signed)])
    check("export: the exported session passes the recipe check", code == 0, out)

    # ---------------------------------------------------------------- settings
    root = tmp / "root"
    stated = dict(SC_DENSITY_FLOOR="none", SC_SMC="hold_out", IIT_PRIMARY_SAMPLE="adni3", IIT_LENGTH_MEASURE="mni_distance", SC_N_NODES="154")
    base = dict(stated, IIT_ROOT=str(root), SC_LABELS="shuffled", SC_LABEL_SEED="1")
    code, out = run(["init", "--connectomes", str(conn), "--rehearsal"], env=dict(IIT_ROOT=str(root), SC_LABELS="shuffled", SC_LABEL_SEED="1"))
    check("init: a chain must state its settings", code != 0 and "states its settings" in out, out)
    code, out = run(["init", "--connectomes", str(conn)], env=base)
    check("init: refused without the rehearsal flag while unsigned", code == 3, out)
    code, out = run(["init", "--connectomes", str(conn), "--rehearsal"], env=base)
    check("init: rehearsal starts and records the bypass", code == 0 and json.loads((root / "chain_settings.json").read_text())["rehearsal"] is True, out)
    check("init: the bypass is written beside the outputs", (root / "REHEARSAL_UNSIGNED.txt").is_file())
    code, out = run(["init", "--connectomes", str(conn), "--rehearsal"], env=dict(base, SC_DENSITY_FLOOR="0.6"))
    check("init: a second start with another floor refuses", code != 0 and "different settings" in out, out)
    code, out = snippet("import iit_chain as c; c.boot(); print('booted')", env=dict(IIT_ROOT=str(root), SC_DENSITY_FLOOR="0.6"))
    check("settings: a step with another floor stops", code != 0 and "SETTINGS GUARD" in out, out)
    code, out = snippet("import iit_chain as c, os; c.boot(); print(os.environ['IIT_P1'])", env=dict(IIT_ROOT=str(root)))
    check("settings: a step needs only IIT_ROOT", code == 0 and str(root / "phase1") in out, out)
    code, out = snippet("import iit_chain as c, os; c.boot(); print(c.P(3)); print(c.docs_dir()); print('IIT_P1' in os.environ)")
    check("settings: with no root boot changes nothing", code == 0 and "phase3_uniform" in out and "iit_dipanjan" in out and "False" in out, out)
    signed_root = tmp / "signed_root"; signed_root.mkdir()
    (signed_root / "chain_settings.json").write_text(json.dumps(dict(settings=dict(density_floor=0.6), rehearsal=False)))
    code, out = snippet("import iit_chain as c; print(c.rehearsal())", env=dict(IIT_ROOT=str(signed_root), IIT_REHEARSAL="1"))
    check("recipe: the bypass cannot be added to a chain started without it", code == 0 and "False" in out, out)
    # ---------------------------------------------------------------- the existing no-ACT connectomes
    NOACT = "/data/derivatives/connectomes_uniform_v3/matrices"
    na = dict(stated, IIT_ROOT=str(tmp / "noact"), SC_LABELS="real")
    code, out = run(["init", "--connectomes", NOACT], env=na)
    check("no-ACT base: refused without the flag", code == 3 and not (tmp / "noact/chain_settings.json").exists(), out)
    code, out = run(["init", "--connectomes", str(conn), "--no-act-base"], env=na)
    check("no-ACT base: the flag on any other folder refuses", code == 3 and "covers only" in out and not (tmp / "noact/chain_settings.json").exists(), out)
    code, out = run(["init", "--connectomes", NOACT, "--no-act-base", "--rehearsal"], env=na)
    check("no-ACT base: refuses together with --rehearsal", code != 0 and "exclude each other" in out, out)
    code, out = run(["init", "--connectomes", NOACT, "--no-act-base"], env={k: v for k, v in na.items() if k != "SC_LABELS"})
    check("no-ACT base: still has to say which labels", code != 0 and "LABEL GUARD" in out, out)
    code, out = run(["init", "--connectomes", NOACT, "--no-act-base"], env=na)
    rec = json.loads((tmp / "noact/chain_settings.json").read_text())
    check("no-ACT base: starts and records it", code == 0 and rec.get("no_act_base") is True and rec["rehearsal"] is False and rec["signed_recipe_check"].startswith("NOT SIGNED"), out)
    check("no-ACT base: written beside the outputs", (tmp / "noact/NO_ACT_BASE_UNSIGNED.txt").is_file())
    code, out = run(["guard", "test"], env=na)
    check("no-ACT base: the launcher guard lets this chain through", code == 0 and "not signed" in out, out)
    code, out = run(["guard", "test"], env=dict(na, IIT_CONN_RAW=str(conn)))
    check("no-ACT base: the guard refuses when the connectomes are repointed", code == 3 and ("covers only" in out or "escape the recipe check" in out), out)
    code, out = run(["init", "--connectomes", NOACT], env=na)
    check("no-ACT base: a restart without the flag refuses", code != 0 and "different settings" in out, out)
    code, out = run(["init", "--connectomes", NOACT, "--no-act-base"], env=na)
    check("no-ACT base: a restart with the flag carries on", code == 0, out)
    code, out = snippet("import iit_chain as c; print(c.rehearsal(), c.no_act_base())", env=na)
    check("no-ACT base: is not a rehearsal, so no banner and no recording of contradictions", "False True" in out, out)
    code, out = run(["guard", "test"], env=dict(stated, IIT_ROOT=str(tmp / "noact_never_started")))
    check("no-ACT base: the flag cannot come from nowhere: another root still refuses", code == 3, out)
    code, out = run(["init", "--connectomes", str(conn), "--rehearsal"], env=dict(stated, IIT_ROOT="/home/ec2-user/exp/docs/iit_dipanjan/x", SC_LABELS="shuffled", SC_LABEL_SEED="1"))
    check("init: a root inside today's outputs refuses", code != 0, out)

    # ---------------------------------------------------------------- labels
    code, out = snippet("import sc_cohort_settings as S; print(S.label_mode(None))")
    check("labels: the legacy route stays real without a flag", code == 0 and "real" in out, out)
    root2 = tmp / "root2"
    code, out = run(["init", "--connectomes", str(conn), "--rehearsal"], env=dict(stated, IIT_ROOT=str(root2)))
    check("labels: a chain with no label flag stops at init", code != 0 and "LABEL GUARD" in out, out)
    code, out = snippet("import sc_cohort_settings as S; print(S.label_mode(None))", env=dict(IIT_ROOT=str(root2)))
    check("labels: under a root real labels need the flag", code != 0 and "LABEL GUARD" in out, out)
    code, out = snippet("import sc_cohort_settings as S; print(S.label_mode(None))", env=dict(IIT_ROOT=str(root2), SC_LABELS="real"))
    check("labels: SC_LABELS=real is the flag", code == 0 and "real" in out, out)
    code, out = snippet("import sc_cohort_settings as S; print(S.label_mode(None))", env=dict(SC_LABELS="shuffled"))
    check("labels: shuffled without a seed stops", code != 0 and "SEED" in out, out)
    code, out = snippet("""
import sc_cohort_settings as S, pandas as pd, tempfile, json, pathlib
d = pathlib.Path(tempfile.mkdtemp())
m = pd.DataFrame(dict(subject_id=[f'{i:03d}_S_{i:04d}' for i in range(30)], group=(['CN','MCI','AD']*10), age=range(30), **{'MMSE Total Score': range(30)}))
a = S.apply_labels(m, None, d); b = S.apply_labels(m, None, d)
assert a.group.value_counts().to_dict() == m.group.value_counts().to_dict()
assert a.equals(b), 'not reproducible'
assert (a.age == m.age).all(), 'age must not move'
assert (a.group != m.group).any()
j = json.loads((d/'label_mode.json').read_text()); assert j['seed'] == 7 and j['mode'] == 'shuffled'
assert S.donor_map(d.parent) is None
print('ok')
""", env=dict(SC_LABELS="shuffled", SC_LABEL_SEED="7"))
    check("labels: a shuffle keeps group sizes, moves scores with diagnosis, leaves age, repeats with the seed", code == 0 and "ok" in out, out)

    # ---------------------------------------------------------------- settings resolve to today's values
    code, out = snippet("""
import sc_cohort_settings as S
assert S.density_floor() == 0.6 and not S.smc_held_out() and S.get('primary_sample') == 'adni34' and S.get('length_measure') == 'per_person'
print('ok')""")
    check("defaults: floor 0.6, SMC in MCI, ADNI-3/4, per-person length", code == 0 and "ok" in out, out)
    for raw, want in (("none", 0.0), ("0", 0.0), ("0.45", 0.45)):
        code, out = snippet("import sc_cohort_settings as S; print(S.density_floor())", env=dict(SC_DENSITY_FLOOR=raw))
        check(f"floor: {raw} reads as {want}", code == 0 and str(want) in out, out)
    code, out = snippet("import sc_cohort_settings as S; S.density_floor()", env=dict(SC_DENSITY_FLOOR="1.5"))
    check("floor: an impossible value stops", code != 0, out)
    code, out = snippet("""
import iit_chain as c, pandas as pd
d = pd.DataFrame(dict(adni34=[1,1,0], phase=['ADNI-3','ADNI-4','ADNI-2']))
print(list(c.primary(d)))""", env=dict(IIT_PRIMARY_SAMPLE="adni3"))
    check("sample: ADNI-3 alone", code == 0 and "[1, 0, 0]" in out, out)
    code, out = snippet("""
import iit_chain as c, pandas as pd
d = pd.DataFrame(dict(adni34=[1,1,0], phase=['ADNI-3','ADNI-4','ADNI-2']))
print(list(c.primary(d)))""", env=dict(IIT_PRIMARY_SAMPLE="pooled"))
    check("sample: pooled", code == 0 and "[1, 1, 1]" in out, out)

    # ---------------------------------------------------------------- Word documents
    docs = tmp / "docs"; docs.mkdir()
    docenv = dict(IIT_DOCS=str(docs))
    code, out = snippet("import iit_chain as c; c.doc_path('X')", env=docenv)
    check("documents: no version name, no build", code != 0 and "version name" in out, out)
    (docs / "X_v9.docx").write_text("exists")
    code, out = snippet("import iit_chain as c; c.doc_path('X')", env=dict(docenv, IIT_DOC_VERSION="v9"))
    check("documents: an existing version refuses", code != 0 and "never overwritten" in out, out)
    code, out = snippet("""
from docx import Document
import iit_chain as c
d = Document(); d.add_paragraph('Nothing guarded here.')
c.save_new(d, c.doc_path('New')); print('saved')
c.save_new(d, c.docs_dir() / 'New_v1.docx')""", env=dict(docenv, IIT_DOC_VERSION="v1"))
    check("documents: saves once, refuses the second time", code != 0 and "saved" in out and "never overwritten" in out, out)
    for tool in ("docx_edit.py", "phase7/p7_post.py", "phase1/p1_docx_update_rebuilt.py", "phase1/p1_docx_appendix.py", "phase1/p1_docx_patch.py"):
        e = {k: v for k, v in os.environ.items() if not k.startswith(("IIT_", "SC_"))}; e["IIT_P1"] = str(tmp)
        args = [str(docs / "X_v9.docx")] + ([str(tmp / "ops.json")] if tool == "docx_edit.py" else [])
        p = subprocess.run([PY, "-I", str(HERE / tool), *args], capture_output=True, text=True, env=e)
        said = p.stdout + p.stderr
        check(f"documents: {tool} refuses a real file", p.returncode != 0 and ("not a working copy" in said or "superseded" in said), said[-200:])
    p = subprocess.run([PY, "-I", str(HERE / "phase1/p1_docx.py")], capture_output=True, text=True)
    check("documents: p1_docx.py refuses to run", p.returncode != 0 and "Never run" in p.stdout + p.stderr, (p.stdout + p.stderr)[-200:])
    for tool in ("phase2/p2_docx.py", "phase3/p3_docx.py", "phase4/p4_docx.py", "phase5/p5_docx.py", "phase6/p6_docx.py", "phase7/p7_supplement.py"):
        e = {k: v for k, v in os.environ.items() if not k.startswith(("IIT_", "SC_"))}; e["IIT_DOCS"] = str(docs)
        p = subprocess.run([PY, "-I", str(HERE / tool), "uniform_v3"], capture_output=True, text=True, env=e)
        check(f"documents: {tool} needs a version name", p.returncode != 0 and "version name" in p.stdout + p.stderr, (p.stdout + p.stderr)[-200:])

    # ---------------------------------------------------------------- narrative guard
    import pandas as pd

    P1, P2, P3, P5 = (tmp / f"p{k}" for k in (1, 2, 3, 5))
    for d in (P1, P2 / "tables", P3 / "results", P5 / "tables", P5 / "act/act45"):
        d.mkdir(parents=True, exist_ok=True)

    def tables(slopes, sl, dice=0.95, r=0.85, signs=(1, 1)):
        pd.DataFrame([dict(condition="P2", region="Whole brain", measure=m, contrast=ct, I_delta=v[0], I_delta_lo=v[1], I_delta_hi=v[2])
                      for (m, ct), v in slopes.items()]).to_csv(P2 / "tables/interaction_uniform_v3.csv", index=False)
        pd.DataFrame([dict(condition="C2 x", kind="TR", contrast=ct, diff=v[0], ci_lo=v[1], ci_hi=v[2]) for ct, v in sl.items()]).to_csv(P1 / "sr_lr_summary.csv", index=False)
        pd.DataFrame([dict(condition="P2", sample=s, dice_median=dice) for s in ("whole sample", "CN only")]).to_csv(P3 / "results/C_split_half_uniform_v3.csv", index=False)
        pd.DataFrame([{"Edges": "whole brain", "Measure": m, "Pearson r, median [range]": f"{r:.2f} [0.1, 0.9]"} for m in ("FA", "MD", "AxD", "RD")]).to_csv(P5 / "tables/act_agreement.csv", index=False)
        pd.DataFrame([{"Measure": f"m{k}", "unconstrained: CN–MCI": "+0.10", "ACT: CN–MCI": "+0.20" if k else "−0.20", "unconstrained: CN–AD": "+0.10", "ACT: CN–AD": "+0.30"} for k in range(7)]).to_csv(P5 / "tables/act_key_effects.csv", index=False)
        pd.DataFrame(dict(subject_id=range(530))).to_csv(P1 / "covariates.csv", index=False)

    good = {(m, ct): (-0.04, -0.08, -0.01) for m in ("FA", "MD", "RD", "AxD") for ct in ("CN-MCI", "CN-AD", "MCI-AD")}
    flat = {ct: (0.01, -0.03, 0.05) for ct in ("CN-MCI", "CN-AD", "MCI-AD")}
    guard_env = dict(IIT_P1=str(P1), IIT_P2=str(P2), IIT_P3=str(P3), IIT_P5=str(P5), IIT_DOCS=str(docs))

    def guarded(sentence, env=None):
        code = f"""
from docx import Document
import narrative_guard as g
d = Document(); d.add_paragraph({sentence!r})
g.check_document(d, 'test.docx'); print('passed')"""
        return snippet(code, env=dict(guard_env, **(env or {})))

    tables(good, flat)
    code, out = guarded("Effects were larger on short connections at every stage; there was no crossover.")
    check("guard: true sentences pass", code == 0 and "passed" in out, out)
    bad = dict(good); bad[("MD", "CN-MCI")] = (0.05, 0.01, 0.09)
    tables(bad, flat)
    code, out = guarded("There was no crossover from long- to short-range vulnerability.")
    check("guard: 'no crossover' stops on a significant reversal", code == 3 and "NARRATIVE GUARD" in out and "reverses" in out, out)
    code, out = guarded("Mean and radial diffusivity effects were larger on short connections at every stage.")
    check("guard: 'at every stage' stops when one stage has the other sign", code == 3 and "MD CN-MCI" in out, out)
    code, out = guarded("In a different use, this holds at every stage.")
    check("guard: 'every stage' with no test stops", code == 3 and "no test covers" in out, out)
    tables(good, {"CN-MCI": (-0.05, -0.09, -0.01), "CN-AD": (0.05, 0.01, 0.09), "MCI-AD": (0.01, -0.03, 0.05)})
    code, out = guarded("The data do not support a crossover from long- to short-range vulnerability.")
    check("guard: 'do not support a crossover' stops on a reversal in short minus long", code == 3 and "short minus long reverses" in out, out)
    code, out = guarded("Short and long connections are affected to a similar degree at every stage.")
    check("guard: 'similar degree at every stage' stops when an interval excludes zero", code == 3 and "excludes zero" in out, out)
    tables(good, flat)
    code, out = guarded("And 13 of 14 leading group effects kept their sign.")
    check("guard: '13 of 14' passes when the table says 13 of 14", code == 0, out)
    tables(good, flat)
    pd.read_csv(P5 / "tables/act_key_effects.csv").head(3).to_csv(P5 / "tables/act_key_effects.csv", index=False)
    code, out = guarded("And 13 of 14 leading group effects kept their sign.")
    check("guard: '13 of 14' stops when the table says otherwise", code == 3 and "5 of 6" in out, out)
    tables(good, flat, r=0.77)
    code, out = guarded("Whole-brain values agreed with the main reconstruction.")
    check("guard: 'agreed' stops at r 0.77", code == 3 and "below 0.8" in out, out)
    tables(good, flat, r=0.85)
    code, out = guarded("Whole-brain values agreed with the main reconstruction.")
    check("guard: 'agreed' passes at r 0.85", code == 0, out)
    tables(good, flat, dice=0.4)
    code, out = guarded("A stable core of 12 exception edges was expressed less in AD.")
    check("guard: 'stable' stops at Dice 0.40", code == 3 and "Dice below" in out, out)
    code, out = guarded("These findings were robust to acquisition protocol, demographics and parcellation.")
    check("guard: 'robust to' stops when the effects table is missing", code == 3 and "does not exist" in out, out)
    code, out = guarded("The core has near-perfect split-half stability.")
    check("guard: 'near-perfect' stops at Dice 0.40", code == 3, out)
    code, out = guarded("Rank-based statistics were used because they are comparatively robust to these non-Gaussian distributions.")
    check("guard: a statistical property is not a claim", code == 0, out)
    tables(good, flat)
    code, out = guarded("Participants were scanned under the ADNI-3 and ADNI-4 protocols.", env=dict(IIT_PRIMARY_SAMPLE="adni3"))
    check("guard: an ADNI-3/4 sentence stops when the sample is ADNI-3", code == 3 and "primary sample" in out, out)
    code, out = guarded("A connectome was excluded when its edge density was below 0.6.", env=dict(SC_DENSITY_FLOOR="none"))
    check("guard: the 0.6 sentence stops when there is no floor", code == 3 and "density floor" in out, out)
    code, out = guarded("Nodes were the 166 regions of the AAL3 atlas, in ten functional networks.", env=dict(SC_N_NODES="154"))
    check("guard: the 166-node sentence stops at 154 nodes", code == 3 and "154" in out, out)
    code, out = guarded("The effect was δ = nan, q = nan.")
    check("guard: a number printed as nan stops", code == 3 and "could not be estimated" in out, out)
    code, out = guarded("Diffusion MRI from 530 participants was reconstructed.")
    check("guard: '530' passes with 530 sessions", code == 0, out)
    pd.DataFrame(dict(subject_id=range(45))).to_csv(P1 / "covariates.csv", index=False)
    code, out = guarded("Diffusion MRI from 530 participants was reconstructed.")
    check("guard: '530' stops with 45 sessions", code == 3 and "45 sessions" in out, out)
    code, out = snippet("""
from docx import Document
import narrative_guard as g, sys
d = Document(); r = d.add_paragraph().add_run('There was no crossover.'); r.font.strike = True
print(g.live_text(d))""", env=guard_env)
    check("guard: struck text is ignored", code == 0 and "[]" in out, out)
    code, out = guarded("Results on ADNI-3/4 were repeated as a sensitivity analysis.", env=dict(IIT_PRIMARY_SAMPLE="adni3"))
    check("guard: ADNI-3/4 named as a sensitivity analysis passes", code == 0, out)
    for words, claim in (("We found no evidence of a crossover.", "no crossover"), ("No analysis showed larger early effects on long connections.", "no crossover"),
                         ("Effects were larger at each stage.", "at every stage"), ("Whole-brain values agree with the ACT run.", "agreed")):
        code, out = snippet(f"import narrative_guard as g; print([l for l, p, _ in g.CLAIMS if p.search({words!r})])", env=guard_env)
        check(f"guard: {words!r} reaches the test for '{claim}'", code == 0 and claim in out, out)
    code, out = snippet("import narrative_guard as g; print(g.t_agreed('The two recipes did not agree.', g.Evidence()))", env=guard_env)
    check("guard: saying the recipes do not agree is not the claim 'agreed'", code == 0 and "None" in out, out)
    (tmp / "guard_root").mkdir(exist_ok=True); (tmp / "guard_root" / "chain_settings.json").write_text(json.dumps(dict(settings={}, no_act_base=True)))
    pd.DataFrame(dict(qc_fail=[False] * 504 + [True] * 15 + [True] * 11, qc_reason=[None] * 504 + ["R0 excluded before the analysis"] * 15 + ["R3 reconstruction"] * 11)).to_csv(P1 / "qc.csv", index=False)
    code, out = guarded("The flow ended at 519 participants before SMC were held out.", env=dict(IIT_ROOT=str(tmp / "guard_root")))
    check("guard: '519 participants before the SMC hold-out' passes when the QC table gives 519", code == 0 and "passed" in out, out)
    pd.DataFrame(dict(qc_fail=[False] * 500 + [True] * 15, qc_reason=[None] * 500 + ["R0 excluded before the analysis"] * 15)).to_csv(P1 / "qc.csv", index=False)
    code, out = guarded("The flow ended at 519 participants before SMC were held out.", env=dict(IIT_ROOT=str(tmp / "guard_root")))
    check("guard: '519 participants' stops when the QC table gives another count", code == 3 and "this chain's QC gives 515" in out, out)
    code, out = guarded("Whole-brain values agreed with the main reconstruction.", env=dict(IIT_NARRATIVE_GUARD="record"))
    tables(good, flat, r=0.5)
    code, out = guarded("Whole-brain values agreed with the main reconstruction.", env=dict(IIT_NARRATIVE_GUARD="record"))
    check("guard: recording instead of stopping is refused outside a rehearsal", code != 0 and "only in a rehearsal" in out, out)

with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    code, out = run(["guard", "test"], env=dict(IIT_ROOT=str(tmp / "fresh"), IIT_REHEARSAL="1"))
    check("recipe: IIT_REHEARSAL before a chain starts does not bypass the check", code == 3 and "bypassed" not in out, out)
    code, out = snippet("import iit_chain as c; print(c.rehearsal())", env=dict(IIT_ROOT=str(tmp / "fresh"), IIT_REHEARSAL="1"))
    check("recipe: a root without init --rehearsal is no rehearsal", code == 0 and "False" in out, out)
    code, out = run(["guard", "test"], env=dict(IIT_ROOT=str(tmp / "fresh"), IIT_CONN_RAW=str(tmp / "fresh/connectomes/matrices"), IIT_CONN="/elsewhere/matrices"))
    check("recipe: an explicit IIT_CONN outside the root refuses", code == 3 and "escape the recipe check" in out, out)
    code, out = run(["guard", "test"], env=dict(IIT_CONN_RAW="/signed/matrices", IIT_CONN_QC="/elsewhere/matrices_qc"))
    check("recipe: without a root, an IIT_CONN_QC other than the checked folder refuses", code == 3 and "escape the recipe check" in out, out)
    dkt = tmp / "dkt"; dkt.mkdir(); (dkt / "SC_DKT76_X_count.csv").write_text("0")
    code, out = run(["init", "--connectomes", "/data/derivatives/connectomes_uniform_v3/matrices", "--dkt", str(dkt), "--no-act-base"],
                    env=dict(IIT_ROOT=str(tmp / "base"), SC_DENSITY_FLOOR="0.6", SC_SMC="hold_out", SC_LABELS="real", IIT_PRIMARY_SAMPLE="adni3", IIT_LENGTH_MEASURE="mni_distance", SC_N_NODES="166"))
    check("no-ACT base: a DKT folder holding other files refuses", code == 3 and "come from elsewhere" in out, out)
    for launcher in ("phase7/build_ms.sh", "phase7/build_thesis.sh", "rebuild/docs_rebuilt.sh", "run_chain.sh"):
        text = (HERE / launcher).read_text().splitlines(); k = next(i for i, l in enumerate(text) if l.startswith("chain_env "))
        check(f"recipe: {launcher} exits after a path test skipped the guard", "IIT_PRINT_PATHS" in text[k + 1] and "exit 0" in text[k + 1])

fails = [n for n, ok in RESULTS if not ok]
print(f"\n{len(RESULTS) - len(fails)} of {len(RESULTS)} passed")
sys.exit(1 if fails else 0)
