"""One place for where a chain reads and writes, and for what it may not do.

The revision scripts (Phases 1-7, build_tree.sh, refresh_all.sh) grew with their paths written in, so a second recipe could
only run by repointing about twenty files and a mistake mixed two recipes without any error. Now:

  IIT_ROOT       the one root variable. Unset: every script resolves today's paths, exactly as before. Set: a chain reads its
                 connectomes from and writes everything under this root (layout below), and nothing else may be written.
  chain_settings.json   the settings of that chain, written once by `iit_chain.py init`. A later step run with a different
                 value in its environment stops, so a chain cannot change settings half way.

Layout of a root:
  connectomes/matrices      this chain's matrices (a link to where the recipe wrote them); qc.json files carry recipe_sha256
  connectomes/matrices_view the same with len_mean replaced by the chosen length measure and nodes beyond SC_N_NODES at zero
                            (only when those settings ask for it)
  connectomes/matrices_qc   links to the above minus the exclusions: what the analysis tree reads
  connectomes/matrices_dkt  the DKT76 matrices
  exclusions.yaml           written from the rule by `iit_chain.py exclusions`
  analysis_tree/  phase1/ ... phase7/  notebooks/  docs/  qc/  logs/

Guards kept here:
  recipe    launchers refuse when the connectomes' recorded recipe hash differs from SIGNED_RECIPE.json, or that file is
            missing. IIT_REHEARSAL=1 bypasses this, only with a root of its own, and the bypass is written into the outputs.
            --no-act-base is the one other way through: it covers only the existing no-ACT connectomes (NO_ACT_BASE), which
            predate any signature, needs a root of its own, and is written into the outputs. Everything else still refuses.
  documents builders take a new version name and never overwrite a Word file (doc_path, save_new, work_copy, publish).
  labels    sc_cohort_settings.py: real diagnoses are joined only with SC_LABELS=real.
  narrative narrative_guard.py: builders stop when computed numbers contradict a fixed sentence.

Command line:
  iit_chain.py init --connectomes DIR [--dkt DIR] [--rehearsal | --no-act-base]   start a chain under IIT_ROOT (settings from the environment)
  iit_chain.py paths                                              print the resolved locations
  iit_chain.py shell-env                                          print export lines for a launcher
  iit_chain.py guard NAME                                         exit 3 unless the recipe check passes
  iit_chain.py check-recipe CONN SIGNED                           the same check on explicit paths (tests)
  iit_chain.py exclusions                                         exclusions.yaml from the rule
  iit_chain.py views                                              matrices_qc (and matrices_view)
  iit_chain.py export-matrices NPZ QC.json OUTDIR UNIT            one session's matrices as the CSV files and qc.json this chain reads
  iit_chain.py work-copy SRC STEM                                 a working copy of SRC for the in-place editors; prints its path
  iit_chain.py publish WORK STEM                                  run the narrative guard, then move WORK to STEM_<version>.docx
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))        # python -I leaves the script's folder off the path
for _up in TOOLS.parents:
    if (_up / "sc_cohort_settings.py").is_file():
        EXP = _up
        break
else:
    EXP = Path("/home/ec2-user/exp")
if str(EXP) not in sys.path:
    sys.path.append(str(EXP))
import sc_cohort_settings as S  # noqa: E402

U0 = Path("/home/ec2-user/exp/docs/iit_dipanjan")
SIGNED_RECIPE = Path("/data/derivatives/recipe_v4/SIGNED_RECIPE.json")
NO_ACT_BASE = Path("/data/derivatives/connectomes_uniform_v3/matrices")   # the existing unconstrained connectomes: no recipe hash, ever
SETTINGS_NAME = "chain_settings.json"
EXIT_REFUSED = 3
RECIPE_KEY = "recipe_sha256"
STATED = ("density_floor", "smc", "primary_sample", "length_measure", "n_nodes")   # no chain may start on these by default


def root() -> Path | None:
    return S.chain_root()


def settings_file() -> dict:
    r = root()
    if r is None or not (r / SETTINGS_NAME).is_file():
        return {}
    return json.loads((r / SETTINGS_NAME).read_text())


def rehearsal() -> bool:
    """A started chain is what its settings file says, whatever the environment says; the bypass cannot be added later."""
    saved = settings_file()
    if saved:
        return bool(saved.get("rehearsal"))
    return os.environ.get("IIT_REHEARSAL", "") not in ("", "0")


def no_act_base() -> bool:
    """A chain started with --no-act-base. Read from its settings file only, so this cannot be added later or from the environment."""
    return bool(settings_file().get("no_act_base"))


def layout(r: Path) -> dict[str, str]:
    """Environment of a chain under root r. Explicit IIT_* values still win (Phase 5 runs variants that way)."""
    conn = r / "connectomes"
    plain = S.get("length_measure") == "per_person" and S.n_nodes() == S.DEFAULTS["n_nodes"]
    out = {
        "IIT_AROOT": r / "analysis_tree",
        "IIT_CONN": conn / ("matrices" if plain else "matrices_view"),
        "IIT_CONN_RAW": conn / "matrices",
        "IIT_CONN_QC": conn / "matrices_qc",
        "IIT_CONN_DKT": conn / "matrices_dkt",
        "IIT_EXCLUSIONS": r / "exclusions.yaml",
        "IIT_DOCS": r / "docs",
        "IIT_NOTEBOOKS": r / "notebooks",
        "IIT_LOGS": r / "logs",
        "IIT_SUFFIX": "",
        "IIT_SRC": "uniform_v3",                       # the tag in the phase file names, not a statement about the recipe
        "IIT_P1_COMPARE": U0 / "phase1",               # earlier analysis: read only, for the 'what changed' tables
        "IIT_P2_COMPARE": U0 / "phase2",
        "IIT_P3_COMPARE": U0 / "phase3",
        "SC_QC_ROOT": r / "qc",                        # keeps the old tree's QC gate and metrics out of this chain
        "SC_DERIV_ROOT": r,
    }
    for k in range(1, 8):
        out[f"IIT_P{k}"] = r / f"phase{k}"
    return {k: str(v) for k, v in out.items()}


def boot() -> None:
    """First call in every phase script. With no root this does nothing at all, which is what keeps today's paths."""
    r = root()
    if r is None:
        return
    if not (r / SETTINGS_NAME).is_file():
        raise SystemExit(f"IIT_ROOT={r} has no {SETTINGS_NAME}: start the chain with iit_chain.py init")
    saved = settings_file().get("settings", {})
    for name, env in S.ENV.items():
        raw = os.environ.get(env)
        now, was = (S.density_floor(), float(saved.get(name))) if name == "density_floor" else (raw, saved.get(name))
        if raw not in (None, "") and str(now) != str(was):
            raise SystemExit(f"SETTINGS GUARD: {env}={raw!r} but this chain was started with {was!r}. "
                             f"A chain keeps one set of settings; use a new root.")
    for k, v in layout(r).items():
        os.environ.setdefault(k, v)
    _require_recipe("phase script")


def P(k: int) -> Path:
    """Phase k's folder: IIT_P<k>, else today's phase<k>_uniform."""
    return Path(os.environ.get(f"IIT_P{k}", U0 / f"phase{k}_uniform"))


def aroot(default: str = "/data/derivatives/qc/analysis_cohort_uniform") -> Path:
    return Path(os.environ.get("IIT_AROOT", default))


def docs_dir() -> Path:
    return Path(os.environ.get("IIT_DOCS", U0))


def dkt_map(for_writing: bool = False) -> Path:
    """DKT region -> network. A chain reads its own map when it has one, else today's (read only), and writes only under its root."""
    today = Path("/data/derivatives/connectomes_uniform_v3/dkt76_network_mapping.csv")
    r = root()
    own = r / "connectomes/dkt76_network_mapping.csv" if r is not None else None
    if "IIT_DKT_MAP" in os.environ:
        return Path(os.environ["IIT_DKT_MAP"])
    if own is not None and (for_writing or own.is_file()):
        return own
    return today


# ------------------------------------------------------------------------------------- primary sample
def sample() -> str:
    return S.get("primary_sample")


def sample_label() -> str:
    return S.SAMPLE_LABEL[sample()]


def primary(df):
    """The primary-sample flag as a 0/1 Series. The default reads the adni34 column exactly as before."""
    import pandas as pd

    s = sample()
    if s == "adni34":
        return df["adni34"]
    if s == "pooled":
        return pd.Series(1, index=df.index)
    return df["phase"].isin(S.SAMPLE_PHASES[s]).astype(int)


# ------------------------------------------------------------------------------------- recipe guard
def recipe_status(conn: Path, signed: Path = SIGNED_RECIPE) -> tuple[bool, str]:
    """(passes, why). Fails on a missing signed file, on any connectome without a recorded hash, and on any difference."""
    signed, conn = Path(signed), Path(conn)
    if not signed.is_file():
        return False, f"{signed} does not exist: no recipe has been signed"
    try:
        want = json.loads(signed.read_text()).get(RECIPE_KEY)
    except ValueError as exc:
        return False, f"{signed} cannot be read: {exc}"
    if not want:
        return False, f"{signed} has no {RECIPE_KEY}"
    files = sorted(conn.glob("SC_*_qc.json"))
    if not files:
        return False, f"{conn} holds no qc.json, so no recipe hash is recorded"
    seen: Counter = Counter()
    for f in files:
        seen[json.loads(f.read_text()).get(RECIPE_KEY)] += 1
    if set(seen) != {want}:
        rows = ", ".join(f"{str(k)[:12] if k else 'none recorded'}: {v}" for k, v in seen.most_common(5))
        return False, f"{conn}: signed {want[:12]}, recorded {rows}"
    return True, f"{sum(seen.values())} connectomes carry the signed recipe {want[:12]}"


def _conn_for_guard() -> Path:
    return Path(os.environ.get("IIT_CONN_RAW") or os.environ.get("IIT_CONN") or "/data/derivatives/connectomes_uniform_v3/matrices")


def _require_recipe(who: str) -> str:
    if rehearsal():
        if root() is None:
            print("RECIPE GUARD: a rehearsal needs a root of its own (IIT_ROOT), or it would write over "
                  "today's outputs without the signed-recipe check.", file=sys.stderr)
            raise SystemExit(EXIT_REFUSED)
        return "bypassed: rehearsal"
    if no_act_base():
        if _conn_for_guard().resolve() != NO_ACT_BASE.resolve():
            print(f"RECIPE GUARD ({who}): refusing to run. --no-act-base covers only {NO_ACT_BASE}, not {_conn_for_guard()}.", file=sys.stderr)
            raise SystemExit(EXIT_REFUSED)
        return "not signed: the existing no-ACT connectomes, by --no-act-base"
    ok, why = recipe_status(_conn_for_guard())
    dkt = os.environ.get("IIT_CONN_DKT", "")
    if ok and root() is not None and dkt and Path(dkt).is_dir():       # the second parcellation must come from the same recipe
        ok, why = recipe_status(Path(dkt))
    if not ok:
        print(f"RECIPE GUARD ({who}): refusing to run. {why}\n"
              f"  The analysis runs only on connectomes whose recorded {RECIPE_KEY} equals {SIGNED_RECIPE}.", file=sys.stderr)
        raise SystemExit(EXIT_REFUSED)
    return why


# ------------------------------------------------------------------------------------- chain start
def _git_head() -> str:
    try:
        return subprocess.run(["git", "-C", str(EXP), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _tools_hash() -> str:
    """The analysis scripts are not all in git (the document builders stay out), so a chain records what it ran."""
    h = hashlib.sha256()
    names = [*TOOLS.glob("*.py"), *TOOLS.glob("*.sh"), *TOOLS.glob("phase*/*.py"), *TOOLS.glob("phase*/*.sh"), *TOOLS.glob("rebuild/*.sh")]
    for f in sorted(names):
        h.update(f.relative_to(TOOLS).as_posix().encode()); h.update(f.read_bytes())
    return h.hexdigest()


def init(connectomes: str, dkt: str | None, rehearse: bool, base: bool = False) -> None:
    r = root()
    if r is None:
        raise SystemExit("init needs IIT_ROOT")
    if rehearse and base:
        raise SystemExit("--rehearsal and --no-act-base exclude each other: a chain is one or the other.")
    if base and Path(connectomes).resolve() != NO_ACT_BASE.resolve():
        print(f"RECIPE GUARD (init): refusing to run. --no-act-base covers only {NO_ACT_BASE}, not {connectomes}.", file=sys.stderr)
        raise SystemExit(EXIT_REFUSED)
    if str(r) == str(U0) or str(r).startswith(str(U0) + "/") or "analysis_cohort" in str(r):
        raise SystemExit(f"IIT_ROOT={r}: a chain root must not be inside today's outputs")
    settings = {name: S.get(name) for name in S.ENV}
    settings["density_floor"] = S.density_floor()
    os.environ["IIT_REHEARSAL"] = "1" if rehearse else os.environ.get("IIT_REHEARSAL", "")
    file = r / SETTINGS_NAME
    if not file.is_file():
        unstated = [env for name, env in S.ENV.items() if name in STATED and not os.environ.get(env)]
        if unstated:
            raise SystemExit("SETTINGS GUARD: a chain states its settings; today's defaults belong to today's outputs. "
                             f"Set {', '.join(unstated)}.")
    if file.is_file():
        old = json.loads(file.read_text())
        same = {k: str(v) for k, v in (old.get("settings") or {}).items()} == {k: str(v) for k, v in settings.items()}
        if not same or bool(old.get("rehearsal")) != bool(rehearse) or bool(old.get("no_act_base")) != bool(base):
            raise SystemExit(f"{file} already holds different settings: a chain keeps one set; use a new root.")
    else:
        S.label_mode(Path(connectomes))                 # stops now when no flag says which labels this chain may join
        if not rehearse and not base:
            ok, why = recipe_status(Path(connectomes))
            if not ok:
                print(f"RECIPE GUARD (init): refusing to run. {why}", file=sys.stderr)
                raise SystemExit(EXIT_REFUSED)
        record = dict(
            settings=settings, rehearsal=bool(rehearse),
            signed_recipe_check=("BYPASSED: rehearsal flag" if rehearse else
                                 "NOT SIGNED: existing no-ACT connectomes, by --no-act-base" if base else recipe_status(Path(connectomes))[1]),
            connectomes=str(connectomes), dkt=dkt, git_head=_git_head(), tools_sha256=_tools_hash(), started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
        if base:
            record["no_act_base"] = True
        r.mkdir(parents=True, exist_ok=True)
        file.write_text(json.dumps(record, indent=1) + "\n")
        if base:
            (r / "NO_ACT_BASE_UNSIGNED.txt").write_text(
                f"This tree reads the existing no-ACT connectomes ({NO_ACT_BASE}), which predate the signed recipe: --no-act-base was set, "
                "so the signed-recipe check did not apply.\nThese are results under unconstrained tracking, not recipe v4.\n"
                f"Labels: {settings['labels']}.\n")
        if rehearse:
            (r / "REHEARSAL_UNSIGNED.txt").write_text(
                "This tree came from a rehearsal: IIT_REHEARSAL bypassed the signed-recipe check. Nothing here is a result.\n"
                f"Labels: {settings['labels']}, seed {settings['label_seed']}.\n")
    for k, v in layout(r).items():
        os.environ.setdefault(k, v)
    for name in ("connectomes", "analysis_tree", "docs", "notebooks", "logs", "qc", *[f"phase{k}" for k in range(1, 8)]):
        (r / name).mkdir(parents=True, exist_ok=True)
    _link(Path(connectomes), r / "connectomes/matrices")
    if dkt:
        _link(Path(dkt), r / "connectomes/matrices_dkt")
    print(f"chain at {r}: {json.dumps(settings)} rehearsal={bool(rehearse)}" + (" no_act_base=True" if base else ""))


def _link(target: Path, link: Path) -> None:
    if link.is_symlink() or link.exists():
        if link.resolve() != target.resolve():
            raise SystemExit(f"{link} already points to {link.resolve()}, not {target}")
        return
    link.symlink_to(target)


# ------------------------------------------------------------------------------------- exclusions: the rule, once
def _count_density(file: Path) -> float:
    import numpy as np

    k = S.n_nodes(); m = np.loadtxt(file, delimiter=",")[:k, :k]; off = m[~np.eye(m.shape[0], dtype=bool)]
    return float((off > 0).sum() / off.size)


def r4_field() -> str | None:
    """The registration measure behind rule R4. IIT_R4_FIELD=none says plainly that this chain has none."""
    raw = str(S.get("r4_field"))
    return None if raw.strip().lower() in ("", "none", "off") else raw


def r4_min() -> float:
    return float(S.get("r4_min"))


def write_exclusions() -> Path:
    """exclude_subjects from the rule (registration below its minimum, or density below the floor), with its reasons."""
    import yaml

    r = root()
    if r is None:
        raise SystemExit("exclusions needs IIT_ROOT")
    for k, v in layout(r).items():
        os.environ.setdefault(k, v)
    conn = Path(os.environ["IIT_CONN_RAW"]); floor = S.density_floor(); field = r4_field(); low = r4_min()
    reasons = {}
    missing = []
    for count in sorted(conn.glob("SC_AAL166_*_count.csv")):
        unit = count.name[len("SC_AAL166_"):-len("_count.csv")]; sid = unit[:10]
        why = []
        d = _count_density(count)
        if d < floor:
            why.append(f"density {d:.2f} below {floor:g}")
        if field:
            q = conn / f"SC_AAL166_{unit}_qc.json"
            val = json.loads(q.read_text()).get(field) if q.is_file() else None
            if val is None:
                missing.append(sid)
            elif float(val) < low:
                why.append(f"{field} {float(val):.2f} below {low:g}")
        if why:
            reasons[sid] = "; ".join(why)
    if missing:
        raise SystemExit(f"R4 GUARD: {len(missing)} connectomes in {conn} lack {field} in their qc.json, so the registration rule "
                         f"cannot run. Record it in the recipe, or set IIT_R4_FIELD=none to state that this chain has no such check.")
    local = EXP / "configs/exclusions.local.yaml"
    lists = (yaml.safe_load(local.read_text()) or {}).get("subject_lists") if local.is_file() else None
    out = r / "exclusions.yaml"
    body = dict(rule=dict(density_floor=floor, registration_field=field, registration_min=low if field else None),
                exclude_subjects=sorted(reasons), reasons=reasons, subject_lists=lists or {})
    out.write_text("# Written by iit_chain.py exclusions from the rule below. Holds ADNI identifiers: stays off git.\n"
                   + yaml.safe_dump(body, sort_keys=False))
    print(f"exclusions: {len(reasons)} of {len(list(conn.glob('SC_AAL166_*_count.csv')))} (floor {floor:g}, registration {field or 'none'})")
    return out


def check_exclusions_floor(path: Path) -> None:
    """The exclusions file must have been written under this chain's floor (today's file is the 0.6 one)."""
    import yaml

    rule = (yaml.safe_load(Path(path).read_text()) or {}).get("rule") or {}
    written = float(rule.get("density_floor", S.DEFAULT_FLOOR))
    if written != S.density_floor():
        raise SystemExit(f"FLOOR GUARD: {path} was written for a density floor of {written:g}, this chain uses {S.density_floor():g}.")


# ------------------------------------------------------------------------------------- length measure
def mni_distance_edges(n_nodes: int = 166):
    """MNI centre-to-centre distance per edge, as in the length audit (AAL3 1 mm atlas, voxel means in world coordinates)."""
    import nibabel as nib
    import numpy as np
    import pandas as pd

    img = nib.load(EXP / "atlas/AAL/AAL3v1_1mm.nii.gz")
    lab = np.rint(np.asarray(img.dataobj)).astype(int)
    nodes = pd.read_csv(EXP / "atlas/AAL/aal3_node_map_166.csv")
    if "orig_value" not in nodes:
        raise SystemExit("aal3_node_map_166.csv has no orig_value column")
    ctr = np.full((n_nodes, 3), np.nan)
    for k, value in enumerate(nodes.orig_value.astype(int).to_numpy()[:n_nodes]):
        vox = np.argwhere(lab == value)
        if len(vox):
            ctr[k] = img.affine[:3, :3] @ vox.mean(0) + img.affine[:3, 3]
    iu = np.triu_indices(n_nodes, 1)
    return np.sqrt(((ctr[iu[0]] - ctr[iu[1]]) ** 2).sum(-1))


def control_median_edges(ref_conn: Path, master_csv: Path, min_share: float):
    """Per edge, the median ACT length over controls of the primary sample having the edge; NaN below min_share of them."""
    import numpy as np
    import pandas as pd

    m = pd.read_csv(master_csv, low_memory=False)
    phase = m.phase.astype(str).str.replace("ADNI ", "ADNI-")
    m = m.assign(phase=phase, adni34=phase.isin(["ADNI-3", "ADNI-4"]).astype(int))
    cn = m[(m.group == "CN") & (primary(m) == 1)].subject_id.astype(str).tolist()
    stack = []
    for sid in cn:
        files = sorted(Path(ref_conn).glob(f"SC_AAL166_{sid}_I*_len_mean.csv"))
        if files:
            L = np.loadtxt(files[0], delimiter=","); stack.append(L[np.triu_indices(L.shape[0], 1)])
    if not stack:
        raise SystemExit(f"no control has a len_mean file in {ref_conn}")
    X = np.array(stack, dtype=float); X[X <= 0] = np.nan
    present = np.isfinite(X).mean(0)
    with np.errstate(all="ignore"):
        med = np.nanmedian(X, 0)
    med[present < min_share] = np.nan
    return med, len(stack)


def build_ruler() -> Path | None:
    """The fixed per-edge values of the chosen measure, saved once. None for per_person."""
    import numpy as np

    measure = S.get("length_measure")
    if measure == "per_person":
        return None
    r = root()
    file = r / f"connectomes/ruler_{measure}.npz"
    if file.is_file():
        return file
    if measure == "mni_distance":
        value = mni_distance_edges(); note = dict(source="AAL3v1_1mm.nii.gz centroids in MNI space", n_controls=None)
    else:
        given = os.environ.get("IIT_LENGTH_REF")          # the no-ACT arm takes the ACT arm's ruler
        if given:
            shutil.copy2(given, file)
            (file.parent / f"ruler_{measure}.json").write_text(json.dumps(dict(measure=measure, taken_from=given), indent=1) + "\n")
            return file
        share = float(os.environ.get("IIT_LENGTH_MIN_SHARE", "0.5"))
        value, n = control_median_edges(Path(os.environ["IIT_CONN_RAW"]), Path(os.environ["IIT_AROOT"]) / "00_master/master_cohort.csv", share)
        note = dict(source=str(os.environ["IIT_CONN_RAW"]), n_controls=n, min_share=share, labels=S.get("labels"))
    np.savez_compressed(file, value=value)
    note.update(measure=measure, edges=int(value.size), edges_defined=int(np.isfinite(value).sum()))
    (file.parent / f"ruler_{measure}.json").write_text(json.dumps(note, indent=1) + "\n")
    print(f"ruler {measure}: {note['edges_defined']} of {note['edges']} edges defined")
    return file


def build_view(ruler: Path | None) -> Path:
    """matrices_view/: links to every matrix; real files where this chain differs: len_mean = the ruler on the edges this person
    has, and, with SC_N_NODES below 166, every matrix with the rows and columns beyond it at zero."""
    import numpy as np

    conn = Path(os.environ["IIT_CONN_RAW"]); view = Path(os.environ["IIT_CONN"]); view.mkdir(parents=True, exist_ok=True)
    value = np.load(ruler)["value"] if ruler is not None else None
    k = S.n_nodes(); cut = k != S.DEFAULTS["n_nodes"]

    def write(arr, target):
        tmp = target.with_suffix(f".tmp{os.getpid()}")
        np.savetxt(tmp, arr, delimiter=",", fmt="%.10g"); os.replace(tmp, target)

    for f in sorted(conn.glob("SC_AAL166_*")):
        target = view / f.name
        if target.is_file() and not target.is_symlink():
            continue
        is_len = f.name.endswith("_len_mean.csv") and value is not None
        if is_len:
            L = np.loadtxt(f, delimiter=","); W = np.loadtxt(conn / f.name.replace("_len_mean.csv", "_fd_sum.csv"), delimiter=",")
            n = L.shape[0]; iu = np.triu_indices(n, 1); out = np.zeros((n, n))
            out[iu] = np.where((L[iu] > 0) & (W[iu] > 0) & np.isfinite(value), value, 0.0)
            out = out + out.T
            out[k:, :] = 0; out[:, k:] = 0
            write(out, target)
        elif cut and f.suffix == ".csv":
            m = np.loadtxt(f, delimiter=","); m[k:, :] = 0; m[:, k:] = 0
            write(m, target)
        elif not os.path.lexists(target):
            target.symlink_to(f)
    return view


def build_qc_view() -> Path:
    """matrices_qc/: what the analysis tree reads. Today's rule, now under the chain's root."""
    import yaml

    source = Path(os.environ["IIT_CONN"]); qc = Path(os.environ["IIT_CONN_QC"]); qc.mkdir(parents=True, exist_ok=True)
    check_exclusions_floor(Path(os.environ["IIT_EXCLUSIONS"]))
    excl = set((yaml.safe_load(Path(os.environ["IIT_EXCLUSIONS"]).read_text()) or {}).get("exclude_subjects") or [])
    for f in qc.glob("*"):                        # drop links of newly excluded participants and links to another source
        if f.name[10:20] in excl or not f.is_symlink() or Path(os.readlink(f)).parent != source:
            f.unlink()
    for f in sorted(source.glob("SC_AAL166_*")):
        target = qc / f.name
        if f.name[10:20] not in excl and not os.path.lexists(target):
            target.symlink_to(source / f.name)
    n = len({f.name[10:20] for f in qc.glob("*_count.csv")})
    print(f"matrices_qc: {n} participants")
    return qc


def views(ruler_stage: bool) -> None:
    r = root()
    if r is None:
        raise SystemExit("views needs IIT_ROOT")
    for k, v in layout(r).items():
        os.environ.setdefault(k, v)
    measure = S.get("length_measure")
    plain = measure == "per_person" and S.n_nodes() == S.DEFAULTS["n_nodes"]
    if ruler_stage:
        build_view(build_ruler())
        build_qc_view()
        return
    if measure == "control_median_act" and not os.environ.get("IIT_LENGTH_REF"):
        # The ruler needs the controls, which the master cohort names; the master cohort reads only count matrices.
        # So the first matrices_qc holds the matrices as they are; build_tree.sh rebuilds it after the master cohort.
        os.environ["IIT_CONN"] = os.environ["IIT_CONN_RAW"]
        build_qc_view()
    else:
        if not plain:
            build_view(build_ruler())
        build_qc_view()


def export_matrices(npz: Path, qc_json: Path, outdir: Path, unit: str) -> None:
    """One session of a recipe run (matrices_aal166.npz, its QC json) as the SC_AAL166_<unit>_<kind>.csv files the chain reads."""
    import numpy as np

    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    z = np.load(npz)
    for kind in z.files:
        tmp = outdir / f".SC_AAL166_{unit}_{kind}.tmp{os.getpid()}"
        np.savetxt(tmp, z[kind], delimiter=",", fmt="%.10g"); os.replace(tmp, outdir / f"SC_AAL166_{unit}_{kind}.csv")
    shutil.copy2(qc_json, outdir / f"SC_AAL166_{unit}_qc.json")


# ------------------------------------------------------------------------------------- Word documents
_VERSION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,40}$")


def doc_version() -> str:
    """The new version name every document build must be given: --version=NAME or IIT_DOC_VERSION."""
    name = os.environ.get("IIT_DOC_VERSION", "")
    for a in list(sys.argv[1:]):
        if a.startswith("--version="):
            name = a.split("=", 1)[1]; sys.argv.remove(a)
    if not name:
        raise SystemExit("DOCUMENT GUARD: this builder needs a new version name (--version=NAME or IIT_DOC_VERSION). "
                         "Nothing was written.")
    if not _VERSION.match(name):
        raise SystemExit(f"DOCUMENT GUARD: version name {name!r} may hold only letters, digits, '.', '_' and '-'.")
    return name


def doc_path(stem: str) -> Path:
    """<docs>/<stem>_<version>.docx. Stops at once when that file exists, so nothing is computed for a refused build."""
    path = docs_dir() / f"{stem}_{doc_version()}.docx"
    if path.exists():
        raise SystemExit(f"DOCUMENT GUARD: {path} exists. Word files are never overwritten; choose a new version name.")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _exclusive_move(tmp: Path, path: Path) -> None:
    try:
        os.link(tmp, path)                   # fails if path exists: no race and no overwrite
    except FileExistsError:
        raise SystemExit(f"DOCUMENT GUARD: {path} exists. Word files are never overwritten; choose a new version name.")
    finally:
        tmp.unlink(missing_ok=True)


def save_new(doc, path: Path, guard: bool = True) -> None:
    """Save a python-docx document as a new file, after the narrative guard."""
    path = Path(path)
    if guard:
        import narrative_guard
        narrative_guard.check_document(doc, path.name)
    _rehearsal_banner(doc)
    fd, name = tempfile.mkstemp(suffix=".docx", dir=path.parent); os.close(fd)
    doc.save(name)
    _exclusive_move(Path(name), path)
    print("wrote", path)


def _rehearsal_banner(doc) -> None:
    if not rehearsal():
        return
    if doc.paragraphs:
        doc.paragraphs[0].insert_paragraph_before(
            f"REHEARSAL, NOT A RESULT: signed-recipe check bypassed; labels {S.get('labels') or 'real'} "
            f"(seed {S.get('label_seed') or 'none'}). Every number below is a plumbing test.")


def work_copy(src: Path, stem: str) -> Path:
    """A working copy for the in-place editors (docx_edit.py, p7_post.py and others), which refuse any other path."""
    version = doc_version()
    final = docs_dir() / f"{stem}_{version}.docx"
    if final.exists():
        raise SystemExit(f"DOCUMENT GUARD: {final} exists. Word files are never overwritten; choose a new version name.")
    work = docs_dir() / ".work" / f"{stem}_{version}.{os.getpid()}.work.docx"
    work.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, work)
    return work


def assert_work_copy(path) -> None:
    p = Path(path)
    if not (p.name.endswith(".work.docx") and p.parent.name == ".work"):
        raise SystemExit(f"DOCUMENT GUARD: {p} is not a working copy. In-place editing of a Word file is refused; make one with "
                         f"iit_chain.py work-copy, then publish it under a new version name.")


def publish(work: Path, stem: str) -> Path:
    """Narrative guard on the finished working copy, then move it to its final name. Never overwrites."""
    assert_work_copy(work)
    from docx import Document
    import narrative_guard

    doc = Document(work)
    final = docs_dir() / f"{stem}_{doc_version()}.docx"
    try:
        narrative_guard.check_document(doc, final.name)
    except SystemExit:
        Path(work).unlink(missing_ok=True)      # a refused build leaves no Word file behind; docs/.guard/ keeps the reasons
        raise
    if rehearsal():
        _rehearsal_banner(doc); doc.save(work)
    _exclusive_move(Path(work), final)
    print("wrote", final)
    return final


# ------------------------------------------------------------------------------------- command line
def _paths_table() -> dict[str, str]:
    """Where a launcher reads and writes when nothing is set: the values these scripts held before the root variable."""
    r = root()
    if r is not None:
        return layout(r)
    return {
        "IIT_AROOT": "/data/derivatives/qc/analysis_cohort_uniform",
        "IIT_CONN": "/data/derivatives/connectomes_uniform_v3/matrices",
        "IIT_CONN_QC": "/data/derivatives/connectomes_uniform_v3/matrices_qc",
        "IIT_CONN_DKT": "/data/derivatives/connectomes_uniform_v3/matrices_dkt",
        "IIT_EXCLUSIONS": str(U0 / "tools/rebuild/exclusions_uniform.yaml"),
        **{f"IIT_P{k}": str(U0 / f"phase{k}_uniform") for k in range(1, 8)},
        "IIT_DOCS": str(U0),
        "IIT_NOTEBOOKS": str(U0 / "notebooks"),
    }


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    cmd, args = argv[0], argv[1:]
    if cmd in ("work-copy", "publish"):
        boot()
    if cmd == "init":
        def val(flag):
            return args[args.index(flag) + 1] if flag in args else None
        if not val("--connectomes"):
            raise SystemExit("init needs --connectomes DIR")
        init(val("--connectomes"), val("--dkt"), "--rehearsal" in args, "--no-act-base" in args)
    elif cmd == "paths":
        print(json.dumps(_paths_table(), indent=1))
    elif cmd == "shell-env":
        r = root()
        if r is None:
            raise SystemExit("shell-env needs IIT_ROOT")
        for k, v in layout(r).items():
            print(f"export {k}={json.dumps(os.environ.get(k, v))}")
        floor = S.density_floor()
        for name, env in S.ENV.items():
            value = f"{floor:g}" if name == "density_floor" else S.get(name)
            if str(value) != "":
                print(f"export {env}={value}")
        if rehearsal():
            print("export IIT_REHEARSAL=1")
    elif cmd == "guard":
        r = root()
        if r is not None:
            for k, v in layout(r).items():
                os.environ.setdefault(k, v)
        if r is None and S.density_floor() != S.DEFAULT_FLOOR:
            print(f"FLOOR GUARD: today's exclusions file and trees belong to the 0.6 floor; SC_DENSITY_FLOOR={S.density_floor():g} "
                  "needs a chain root of its own (IIT_ROOT).", file=sys.stderr)
            raise SystemExit(EXIT_REFUSED)
        print(f"recipe check ({args[0] if args else 'launcher'}): {_require_recipe(args[0] if args else 'launcher')}")
    elif cmd == "check-recipe":
        ok, why = recipe_status(Path(args[0]), Path(args[1]))
        print(("OK: " if ok else "REFUSED: ") + why)
        return 0 if ok else EXIT_REFUSED
    elif cmd == "dkt-map":
        print(dkt_map())
    elif cmd == "exclusions":
        write_exclusions()
    elif cmd == "views":
        views("--ruler" in args)
    elif cmd == "export-matrices":
        export_matrices(Path(args[0]), Path(args[1]), Path(args[2]), args[3])
    elif cmd == "work-copy":
        print(work_copy(Path(args[0]), args[1]))
    elif cmd == "publish":
        publish(Path(args[0]), args[1])
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
