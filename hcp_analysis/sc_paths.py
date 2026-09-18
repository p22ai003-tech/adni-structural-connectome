"""Path resolution and a common CLI for the ``build_*.py`` analysis scripts.

Why this exists
---------------
Each build script used to open with its own block of absolute constants::

    OUT      = Path("/home/ec2-user/exp/hcp_analysis")
    ANALYSIS = Path("/home/ec2-user/exp/data/derivatives/qc/analysis_cohort")
    FUNC     = ANALYSIS / "19_network_analysis/functional"

Two things were wrong with that. The paths only resolve on one machine, and
more importantly the scripts *wrote* to ``hcp_analysis/`` while both the
dashboard and the other build scripts *read* from the analysis tree. Closing
that gap took two manual copy steps -- five ``network_*`` tables into
``19_network_analysis/functional/`` and forty-three files into
``20_exception_specificity/`` -- which lived only in shell history. Miss one and
the dashboard silently serves stale numbers, because every reader falls back to
whatever is already on disk.

Here the default output directory *is* the directory the consumer reads, so
there is nothing left to copy. ``--out`` overrides it for a scratch run.

Usage
-----
    import sc_paths
    P = sc_paths.resolve("ml")          # or "network"
    P.out / "ml_targets.csv"            # -> .../20_exception_specificity/
    P.analysis / "00_master/master_cohort.csv"
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    import sc_config
except ModuleNotFoundError:  # running as a loose script from hcp_analysis/
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import sc_config

__all__ = ["BuildPaths", "resolve", "resolve_for_script", "add_common_args"]

# Which analysis-tree section each family of build scripts writes into. The
# network family has to land in the functional subdirectory because that is
# where network_ranked.py and two sibling build scripts look for it.
_FAMILY_OUT = {
    "ml": "exception_dir",
    "network": "functional_network_dir",
}


@dataclass(frozen=True)
class BuildPaths:
    out: Path               # where this script writes
    analysis: Path          # analysis_cohort root
    func: Path              # 19_network_analysis/functional
    exceptions: Path        # 17_edr_exceptions
    connectomes: Path
    figs: Path
    cohort: Path           # ADNI-restricted clinical tables; never redistributed
    aal_node_map: Path     # the correct 166-node LUT, keyed on matrix index

    def ensure(self) -> "BuildPaths":
        """Create the output and figure directories. Inputs are never created:
        a missing input is a real error and must not be masked by an empty dir."""
        self.out.mkdir(parents=True, exist_ok=True)
        self.figs.mkdir(parents=True, exist_ok=True)
        return self


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--out", type=Path, default=None,
        help="output directory (default: the analysis-tree section the dashboard reads)",
    )
    parser.add_argument(
        "--analysis-root", type=Path, default=None,
        help="analysis_cohort root (default: $SC_ANALYSIS_ROOT)",
    )
    parser.add_argument(
        "--figs", type=Path, default=None,
        help="figure output directory (default: <project>/docs/figs)",
    )
    return parser


def resolve(family: str = "ml", args: argparse.Namespace | None = None) -> BuildPaths:
    """Resolve every path a build script needs.

    ``family`` selects the default output section: "ml" writes the
    exception-specificity / ML artifacts, "network" writes the network tables.
    An explicit ``--out`` always wins.
    """
    if family not in _FAMILY_OUT:
        raise ValueError(f"unknown family {family!r}; expected one of {sorted(_FAMILY_OUT)}")

    p = sc_config.paths()
    analysis = getattr(args, "analysis_root", None) or p.analysis_root
    # Re-derive the sections when the caller overrides the analysis root, so an
    # override moves the whole tree rather than half of it.
    if analysis != p.analysis_root:
        func = analysis / "19_network_analysis" / "functional"
        exceptions = analysis / "17_edr_exceptions"
        default_out = analysis / ("19_network_analysis/functional" if family == "network"
                                  else "20_exception_specificity")
    else:
        func = p.functional_network_dir
        exceptions = p.edr_exceptions_dir
        default_out = getattr(p, _FAMILY_OUT[family])

    return BuildPaths(
        out=getattr(args, "out", None) or default_out,
        analysis=Path(analysis),
        func=func,
        exceptions=exceptions,
        connectomes=p.connectomes_dir,
        figs=getattr(args, "figs", None) or p.figs_dir,
        cohort=p.cohort_dir,
        aal_node_map=p.aal_node_map,
    )


def resolve_for_script(family: str = "ml") -> BuildPaths:
    """Resolve paths at module level in a build script.

    Uses ``parse_known_args`` so that importing the script -- from a test, or
    from the runner -- cannot fail on an argv it did not expect. Unrecognised
    arguments belong to whoever is driving the process, not to us.
    """
    parser = add_common_args(argparse.ArgumentParser(add_help=False))
    known, _ = parser.parse_known_args()
    return resolve(family, known).ensure()


if __name__ == "__main__":  # pragma: no cover
    ap = add_common_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--family", default="ml", choices=sorted(_FAMILY_OUT))
    a = ap.parse_args()
    bp = resolve(a.family, a)
    for name, value in bp.__dict__.items():
        mark = "" if Path(value).exists() else "   <- does not exist"
        print(f"  {name:<12} {value}{mark}")
