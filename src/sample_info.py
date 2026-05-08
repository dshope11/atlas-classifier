"""Sample registry for the H->WW->2lnunu 0-jet analysis.

Maps friendly sample keys to ATLAS Open Data 2025 release DSIDs and the
``is_signal`` label used at training time. All other per-sample metadata
(``xsec``, ``filteff``, ``kfac``, ``sum_of_weights``) is stored in the
ntuple itself in the 2025 release - no external ``infofile.py`` needed
(unlike the 2020 release).

Source for sample selection rationale and the per-DSID metadata table: see
``notes/data_sources.md``.

Sample list discovered via:
    import atlasopenmagic as atom
    atom.set_release("2025e-13tev-beta")
    atom.get_all_metadata()
"""

from __future__ import annotations

from dataclasses import dataclass

ATLAS_OPENDATA_RELEASE = "2025e-13tev-beta"
ATLAS_OPENDATA_SKIM = "2to4lep"


@dataclass(frozen=True)
class SampleInfo:
    """Metadata for one MC sample."""

    dsid: int
    is_signal: bool
    description: str


# H->WW->lvlv signal (all production modes - ggF dominates ~85%).
# Selection rationale: full Higgs-mass-relevant production set; mixture
# weighted by per-sample xsec at load time.
# DSIDs verified against atlasopenmagic 2026-04-26.
SAMPLES: dict[str, SampleInfo] = {
    # Signal - H->WW->lvlv
    "ggH125_WWlvlv": SampleInfo(
        dsid=345324,
        is_signal=True,
        description="Powheg+Pythia8 NNLOPS ggH125 -> WW -> lvlv (dominant production mode)",
    ),
    "VBFH125_WWlvlv": SampleInfo(
        dsid=345948,
        is_signal=True,
        description="Powheg+Pythia8 VBFH125 -> WW -> lvlv",
    ),
    "WpH125_qqWWlvlv": SampleInfo(
        dsid=345325,
        is_signal=True,
        description="Powheg+Pythia8 MINLO W+H125 (W->qq), H -> WW -> lvlv",
    ),
    "WmH125_qqWWlvlv": SampleInfo(
        dsid=345433,
        is_signal=True,
        description="Powheg+Pythia8 MINLO W-H125 (W->qq), H -> WW -> lvlv",
    ),
    "ZH125_vvWWlvlv": SampleInfo(
        dsid=345445,
        is_signal=True,
        description="Powheg+Pythia8 MINLO ZH125 (Z->nunu), H -> WW -> lvlv",
    ),
    "ggZH125_ZinclWWlvlv": SampleInfo(
        dsid=346524,
        is_signal=True,
        description="Powheg+Pythia8 MINLO gg-induced ZH125 (Z inclusive), H -> WW -> lvlv",
    ),
    # Background - fully-leptonic WW (irreducible diboson) + dileptonic ttbar
    # (suppressed by 0-jet veto but still contributes residually)
    "WW_llvv": SampleInfo(
        dsid=700602,
        is_signal=False,
        description="Sherpa 2.2.12 WW -> lvlv (opposite-sign), the irreducible diboson background",
    ),
    "ttbar_nonallhad": SampleInfo(
        dsid=410470,
        is_signal=False,
        description="Powheg+Pythia8 A14 ttbar non-all-hadronic; 0-jet veto strongly suppresses this",
    ),
}


def signal_keys() -> list[str]:
    """Return the keys of all signal samples."""
    return [k for k, s in SAMPLES.items() if s.is_signal]


def background_keys() -> list[str]:
    """Return the keys of all background samples."""
    return [k for k, s in SAMPLES.items() if not s.is_signal]


def dsids() -> list[int]:
    """Return the list of all DSIDs (signal + background) in registration order."""
    return [s.dsid for s in SAMPLES.values()]
