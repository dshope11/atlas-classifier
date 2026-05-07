"""ROOT → HDF5 data loading for the H→WW→2lνν 0-jet analysis.

For each sample registered in :mod:`src.sample_info`:

1. Read needed branches from the local ROOT file via ``uproot``, batched by
   ``uproot.TTree.iterate`` so the 1.9 GB ttbar file does not blow memory.
2. Filter to events with exactly two leptons (otherwise lep[:, 0/1] indexing
   would be unsafe — the ``2to4lep`` skim guarantees ≥2 but allows up to 4).
3. Compute the pre-cut scalars (``lead_lep_pt``, ``sublead_lep_pt``,
   ``lep_charge_product``) needed by the cut DSL.
4. Apply event-selection cuts via :func:`src.utils.evaluate_cuts` against the
   ``cuts`` block in ``config.yaml``.
5. Compute the per-event physics weight using the canonical 2025-release
   formula (see ``notes/data_sources.md``); tag ``is_signal`` from the
   sample registry.

All sample chunks are concatenated and written to a single HDF5 dataset
``events`` as a structured numpy record array. The output file is validated
immediately afterward.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import awkward as ak
import h5py
import numpy as np
import uproot

# Make src.* importable when this module is run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import TrainingConfig, load_config  # noqa: E402
from src.sample_info import SAMPLES, SampleInfo  # noqa: E402
from src.utils import chronomat, evaluate_cuts, print_timings, setup_logging  # noqa: E402

LOGGER = logging.getLogger(Path(__file__).stem)

# ROOT branches needed for both selection and weight computation
_BRANCHES: list[str] = [
    "lep_pt", "lep_eta", "lep_phi", "lep_e", "lep_charge", "lep_type", "lep_n",
    "met", "met_phi",
    "jet_n",
    "mcWeight",
    "ScaleFactor_PILEUP", "ScaleFactor_ELE", "ScaleFactor_MUON", "ScaleFactor_LepTRIGGER",
    # Per-event sample metadata (replicated as columns in the 2025 release)
    "xsec", "filteff", "kfac", "sum_of_weights",
]

# Output schema — one structured-array record per surviving event
OUTPUT_DTYPE: np.dtype = np.dtype([
    ("lep_pt_lead",        "f4"),
    ("lep_pt_sublead",     "f4"),
    ("lep_eta_lead",       "f4"),
    ("lep_eta_sublead",    "f4"),
    ("lep_phi_lead",       "f4"),
    ("lep_phi_sublead",    "f4"),
    ("lep_e_lead",         "f4"),
    ("lep_e_sublead",      "f4"),
    ("lep_charge_lead",    "i4"),
    ("lep_charge_sublead", "i4"),
    ("lep_type_lead",      "i4"),
    ("lep_type_sublead",   "i4"),
    ("met",                "f4"),
    ("met_phi",            "f4"),
    ("jet_n",              "i4"),
    ("event_weight",       "f8"),
    ("is_signal",          "i1"),
])

# Iterate the tree in ~100 MB chunks; ttbar is 1.9 GB so this matters.
_ITERATE_STEP_SIZE = "100 MB"


def _process_chunk(
    arrays: ak.Array,
    info: SampleInfo,
    lumi: float,
    cuts: dict[str, Any],
) -> np.ndarray:
    """Apply selection + weight to one ``uproot.iterate`` chunk."""
    # Pre-filter to lep_n == 2 so lep[:, 0/1] is always safe
    two_lep_mask = arrays["lep_n"] == 2
    arrays = arrays[two_lep_mask]
    if len(arrays) == 0:
        return np.zeros(0, dtype=OUTPUT_DTYPE)

    # Pre-cut scalars referenced by config.yaml's cuts DSL
    lep_pt = arrays["lep_pt"]
    lep_charge = arrays["lep_charge"]
    cut_inputs: dict[str, np.ndarray] = {
        "lep_n": ak.to_numpy(arrays["lep_n"]),
        "jet_n": ak.to_numpy(arrays["jet_n"]),
        "lead_lep_pt": ak.to_numpy(lep_pt[:, 0]),
        "sublead_lep_pt": ak.to_numpy(lep_pt[:, 1]),
        "lep_charge_product": ak.to_numpy(lep_charge[:, 0] * lep_charge[:, 1]),
    }
    mask = evaluate_cuts(cut_inputs, cuts)
    arrays = arrays[mask]
    if len(arrays) == 0:
        return np.zeros(0, dtype=OUTPUT_DTYPE)

    # Per-event physics weight — see notes/data_sources.md for derivation.
    # source: 2025 ATLAS Open Data ttbar_analysis.ipynb (calc_weight)
    # generalised from single-lepton (El/Mu trigger split) to 2-lepton
    # (LepTRIGGER) by multiplying both flavour SFs (1.0 for absent flavour).
    xsec = ak.to_numpy(arrays["xsec"])
    filteff = ak.to_numpy(arrays["filteff"])
    kfac = ak.to_numpy(arrays["kfac"])
    sumw = ak.to_numpy(arrays["sum_of_weights"])
    mcweight = ak.to_numpy(arrays["mcWeight"])
    sf_pileup = ak.to_numpy(arrays["ScaleFactor_PILEUP"])
    sf_ele = ak.to_numpy(arrays["ScaleFactor_ELE"])
    sf_muon = ak.to_numpy(arrays["ScaleFactor_MUON"])
    sf_trig = ak.to_numpy(arrays["ScaleFactor_LepTRIGGER"])
    # lumi is in fb⁻¹; ntuple xsec is in pb → multiply lumi by 1000
    event_weight = (
        (lumi * 1000.0 * xsec * kfac * filteff / sumw)
        * mcweight
        * sf_pileup * sf_ele * sf_muon * sf_trig
    )

    n = len(arrays)
    out = np.empty(n, dtype=OUTPUT_DTYPE)
    out["lep_pt_lead"]        = ak.to_numpy(arrays["lep_pt"][:, 0])
    out["lep_pt_sublead"]     = ak.to_numpy(arrays["lep_pt"][:, 1])
    out["lep_eta_lead"]       = ak.to_numpy(arrays["lep_eta"][:, 0])
    out["lep_eta_sublead"]    = ak.to_numpy(arrays["lep_eta"][:, 1])
    out["lep_phi_lead"]       = ak.to_numpy(arrays["lep_phi"][:, 0])
    out["lep_phi_sublead"]    = ak.to_numpy(arrays["lep_phi"][:, 1])
    out["lep_e_lead"]         = ak.to_numpy(arrays["lep_e"][:, 0])
    out["lep_e_sublead"]      = ak.to_numpy(arrays["lep_e"][:, 1])
    out["lep_charge_lead"]    = ak.to_numpy(arrays["lep_charge"][:, 0])
    out["lep_charge_sublead"] = ak.to_numpy(arrays["lep_charge"][:, 1])
    out["lep_type_lead"]      = ak.to_numpy(arrays["lep_type"][:, 0])
    out["lep_type_sublead"]   = ak.to_numpy(arrays["lep_type"][:, 1])
    out["met"]                = ak.to_numpy(arrays["met"])
    out["met_phi"]            = ak.to_numpy(arrays["met_phi"])
    out["jet_n"]              = ak.to_numpy(arrays["jet_n"])
    out["event_weight"]       = event_weight
    out["is_signal"]          = int(info.is_signal)
    return out


def _process_sample(
    path: Path,
    key: str,
    info: SampleInfo,
    lumi: float,
    cuts: dict[str, Any],
) -> np.ndarray:
    """Load one sample's ROOT file in chunks, apply cuts, return structured array."""
    LOGGER.info("Loading %s  DSID=%d  file=%s", key, info.dsid, path.name)
    chunks: list[np.ndarray] = []
    with uproot.open(path) as f:
        tree = f["analysis"]
        for arr in tree.iterate(_BRANCHES, library="ak", step_size=_ITERATE_STEP_SIZE):
            chunks.append(_process_chunk(arr, info, lumi, cuts))
    if not chunks:
        return np.zeros(0, dtype=OUTPUT_DTYPE)
    out = np.concatenate(chunks)
    LOGGER.info("  %s: %d events kept (is_signal=%s)", key, len(out), info.is_signal)
    return out


@chronomat
def load_to_hdf5(config: TrainingConfig) -> None:
    """Process all SAMPLES into a single HDF5 ``events`` dataset.

    Output schema: one structured array named ``events`` with fields per
    :data:`OUTPUT_DTYPE`. File-level attrs record ``lumi_fb_inv`` and
    summary counts.
    """
    raw_dir = Path(config.raw_dir)
    output = Path(config.processed_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    chunks: list[np.ndarray] = []
    for key, info in SAMPLES.items():
        path = raw_dir / f"mc_{info.dsid}.root"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing ROOT file: {path}. Run scripts/download_data.py first."
            )
        chunks.append(_process_sample(path, key, info, config.lumi, config.cuts))

    events = np.concatenate(chunks) if chunks else np.zeros(0, dtype=OUTPUT_DTYPE)
    n_sig = int(events["is_signal"].sum())
    n_bkg = len(events) - n_sig
    LOGGER.info(
        "Total events after selection: %d (signal: %d, background: %d)",
        len(events), n_sig, n_bkg,
    )
    if n_sig == 0 or n_bkg == 0:
        raise RuntimeError(
            f"Empty class after selection — n_signal={n_sig}, n_background={n_bkg}. "
            "Check cuts in config.yaml."
        )

    with h5py.File(output, "w") as f:
        f.create_dataset("events", data=events, compression="lzf")
        f.attrs["lumi_fb_inv"] = config.lumi
        f.attrs["n_signal"] = n_sig
        f.attrs["n_background"] = n_bkg

    # Validate output — catch silent write failures
    with h5py.File(output, "r") as f:
        if "events" not in f:
            raise RuntimeError(f"Output validation failed: 'events' missing in {output}")
        if f["events"].dtype != OUTPUT_DTYPE:
            raise RuntimeError(
                f"Output validation failed: dtype mismatch in {output}: "
                f"{f['events'].dtype} != {OUTPUT_DTYPE}"
            )
        if len(f["events"]) != len(events):
            raise RuntimeError(
                f"Output validation failed: length mismatch in {output}: "
                f"{len(f['events'])} != {len(events)}"
            )

    LOGGER.info("Wrote %s (%d events)", output, len(events))


def main() -> int:
    config = load_config("config.yaml")
    setup_logging(config.log_path)
    load_to_hdf5(config)
    print_timings(LOGGER)
    return 0


if __name__ == "__main__":
    sys.exit(main())
