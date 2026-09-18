"""Subject-level exclusions, kept out of the repository.

The published results exclude one subject, flagged by the project's outlier
audit for mean diffusivity around ten times normal -- CSF-level, i.e. the
diffusion signal is not brain tissue. That exclusion is part of the method and
has to survive, but the identifier cannot live in the repository: ADNI
participant identifiers are restricted under the Data Use Agreement.

So the list lives in a file that git ignores, and this module loads it. The
default location is ``configs/exclusions.local.yaml``, overridable with
``SC_EXCLUSIONS``:

    exclude_subjects:
      - XXX_S_XXXX          # CSF-level MD, ~10x normal

If the file is absent the exclusion set is empty and ``warn_if_empty`` says so,
loudly, because a run without it will not reproduce the published numbers and
should not appear to.
"""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT = PROJECT_ROOT / "configs" / "exclusions.local.yaml"

__all__ = ["load", "warn_if_empty", "path", "subject_list"]


def path() -> Path:
    raw = os.environ.get("SC_EXCLUSIONS")
    return Path(raw).expanduser() if raw else DEFAULT


@lru_cache(maxsize=1)
def load() -> frozenset[str]:
    p = path()
    if not p.is_file():
        return frozenset()
    try:
        import yaml
        data = yaml.safe_load(p.read_text()) or {}
    except Exception as exc:  # a malformed list must not silently mean "none"
        raise SystemExit(f"could not read {p}: {type(exc).__name__}: {exc}")
    subjects = data.get("exclude_subjects") or []
    if isinstance(subjects, str):
        subjects = [subjects]
    return frozenset(str(s).strip() for s in subjects if str(s).strip())


def warn_if_empty(context: str = "") -> frozenset[str]:
    """Load the set, and say so plainly when it is empty."""
    excl = load()
    if not excl:
        where = f" ({context})" if context else ""
        print(
            f"NOTE{where}: no subject exclusions loaded from {path()}.\n"
            "      The published results exclude one subject (CSF-level mean\n"
            "      diffusivity, ~10x normal). Without that file this run will not\n"
            "      reproduce them. See sc_exclusions.py.",
            file=sys.stderr,
        )
    return excl


@lru_cache(maxsize=8)
def subject_list(name: str) -> tuple[str, ...]:
    """A named list of subjects from the same gitignored local file.

    Several modules carried a hard-coded default cohort -- a canary set, an
    expected-unit list, a QC panel. Those are defaults, not logic, and they name
    real ADNI participants, so they live beside the exclusions:

        subject_lists:
          v2_default_subjects: [ ... ]
          spatial_contract_panel: [ ... ]

    An absent list is empty. Every caller treats empty as "no default set", not
    as an error, so the code runs anywhere and only does less.
    """
    p = path()
    if not p.is_file():
        return ()
    try:
        import yaml
        data = yaml.safe_load(p.read_text()) or {}
    except Exception as exc:
        raise SystemExit(f"could not read {p}: {type(exc).__name__}: {exc}")
    items = (data.get("subject_lists") or {}).get(name) or []
    if isinstance(items, str):
        items = [items]
    return tuple(str(x).strip() for x in items if str(x).strip())
