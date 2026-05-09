"""ROOT -> HDF5 data loading for the H->WW->2lnunu 0-jet analysis.

For each sample registered in :mod:`src.sample_info`:

1. Read needed branches from the local ROOT file via ``uproot``, batched by
   ``uproot.TTree.iterate`` so the 1.9 GB ttbar file does not blow memory.
2. Filter to events with exactly two leptons (otherwise lep[:, 0/1] indexing
   would be unsafe - the ``2to4lep`` skim guarantees >= 2 but allows up to 4).
3. Compute the pre-cut scalars needed by the cut DSL:
   ``lead_lep_pt``, ``sublead_lep_pt``, ``lep_charge_product``,
   ``lep_type_product`` (143 for emu), ``met``, ``mll``, ``dphi_ll_met``.
4. Apply event-selection cuts via :func:`src.utils.evaluate_cuts` against the
   ``cuts`` block in ``config.yaml``.
5. Compute the per-event physics weight using the canonical 2025-release
   formula (see ``notes/data_sources.md``); tag ``is_signal`` from the
   sample registry.

Selected events are appended directly to a resizable HDF5 dataset named
``events`` as structured numpy records. The output file is validated
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

# Output schema - one structured-array record per surviving event
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
    lep_pt     = arrays["lep_pt"]
    lep_charge = arrays["lep_charge"]
    lep_type   = arrays["lep_type"]

    pt0  = ak.to_numpy(lep_pt[:, 0]).astype(np.float64)
    pt1  = ak.to_numpy(lep_pt[:, 1]).astype(np.float64)
    phi0 = ak.to_numpy(arrays["lep_phi"][:, 0]).astype(np.float64)
    phi1 = ak.to_numpy(arrays["lep_phi"][:, 1]).astype(np.float64)
    eta0 = ak.to_numpy(arrays["lep_eta"][:, 0]).astype(np.float64)
    eta1 = ak.to_numpy(arrays["lep_eta"][:, 1]).astype(np.float64)
    e0   = ak.to_numpy(arrays["lep_e"][:, 0]).astype(np.float64)
    e1   = ak.to_numpy(arrays["lep_e"][:, 1]).astype(np.float64)

    # Dilepton system - shared intermediates for mll and dphi_ll_met
    px_ll = pt0 * np.cos(phi0) + pt1 * np.cos(phi1)
    py_ll = pt0 * np.sin(phi0) + pt1 * np.sin(phi1)
    pz_ll = pt0 * np.sinh(eta0) + pt1 * np.sinh(eta1)
    m_sq  = (e0 + e1) ** 2 - (px_ll ** 2 + py_ll ** 2 + pz_ll ** 2)
    mll   = np.sqrt(np.clip(m_sq, 0.0, None))

    phi_ll      = np.arctan2(py_ll, px_ll)
    met_phi_np  = ak.to_numpy(arrays["met_phi"]).astype(np.float64)
    raw_dphi    = phi_ll - met_phi_np
    dphi_ll_met = np.abs(np.arctan2(np.sin(raw_dphi), np.cos(raw_dphi)))

    cut_inputs: dict[str, np.ndarray] = {
        "lep_n":              ak.to_numpy(arrays["lep_n"]),
        "jet_n":              ak.to_numpy(arrays["jet_n"]),
        "lead_lep_pt":        pt0,
        "sublead_lep_pt":     pt1,
        "lep_charge_product": ak.to_numpy(lep_charge[:, 0] * lep_charge[:, 1]),
        "lep_type_product":   ak.to_numpy(lep_type[:, 0] * lep_type[:, 1]),
        "met":                ak.to_numpy(arrays["met"]),
        "mll":                mll,
        "dphi_ll_met":        dphi_ll_met,
    }
    mask = evaluate_cuts(cut_inputs, cuts)
    arrays = arrays[mask]
    if len(arrays) == 0:
        return np.zeros(0, dtype=OUTPUT_DTYPE)

    # Per-event physics weight - see notes/data_sources.md for derivation.
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
    # lumi is in fb^-1; ntuple xsec is in pb -> multiply lumi by 1000
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
    events: h5py.Dataset,
) -> tuple[int, int]:
    """Load one sample's ROOT file in chunks and append selected events to HDF5."""
    LOGGER.info("Loading %s  DSID=%d  file=%s", key, info.dsid, path.name)
    n_sample = 0
    n_signal = 0
    with uproot.open(path) as f:
        tree = f["analysis"]
        for arr in tree.iterate(_BRANCHES, library="ak", step_size=_ITERATE_STEP_SIZE):
            chunk = _process_chunk(arr, info, lumi, cuts)
            if len(chunk) == 0:
                continue
            _append_events(events, chunk)
            n_sample += len(chunk)
            n_signal += int(chunk["is_signal"].sum())
    LOGGER.info("  %s: %d events kept (is_signal=%s)", key, n_sample, info.is_signal)
    return n_sample, n_signal


def _append_events(events: h5py.Dataset, chunk: np.ndarray) -> None:
    """Append one processed chunk to the resizable ``events`` dataset."""
    if chunk.dtype != OUTPUT_DTYPE:
        raise TypeError(f"Unexpected event dtype: {chunk.dtype} != {OUTPUT_DTYPE}")
    old_size = len(events)
    new_size = old_size + len(chunk)
    events.resize((new_size,))
    events[old_size:new_size] = chunk


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

    tmp_output = output.with_name(f"{output.name}.tmp")
    if tmp_output.exists():
        tmp_output.unlink()

    total_events = 0
    n_sig = 0
    try:
        with h5py.File(tmp_output, "w") as f:
            events = f.create_dataset(
                "events",
                shape=(0,),
                maxshape=(None,),
                dtype=OUTPUT_DTYPE,
                chunks=True,
                compression="lzf",
            )
            for key, info in SAMPLES.items():
                path = raw_dir / f"mc_{info.dsid}.root"
                if not path.exists():
                    raise FileNotFoundError(
                        f"Missing ROOT file: {path}. Run scripts/download_data.py first."
                    )
                n_sample, n_sample_sig = _process_sample(
                    path, key, info, config.lumi, config.cuts, events
                )
                total_events += n_sample
                n_sig += n_sample_sig

            n_bkg = total_events - n_sig
            LOGGER.info(
                "Total events after selection: %d (signal: %d, background: %d)",
                total_events, n_sig, n_bkg,
            )
            if n_sig == 0 or n_bkg == 0:
                raise RuntimeError(
                    f"Empty class after selection - n_signal={n_sig}, n_background={n_bkg}. "
                    "Check cuts in config.yaml."
                )

            f.attrs["lumi_fb_inv"] = config.lumi
            f.attrs["n_signal"] = n_sig
            f.attrs["n_background"] = n_bkg

        # Validate output - catch silent write failures
        with h5py.File(tmp_output, "r") as f:
            if "events" not in f:
                raise RuntimeError(f"Output validation failed: 'events' missing in {tmp_output}")
            if f["events"].dtype != OUTPUT_DTYPE:
                raise RuntimeError(
                    f"Output validation failed: dtype mismatch in {tmp_output}: "
                    f"{f['events'].dtype} != {OUTPUT_DTYPE}"
                )
            if len(f["events"]) != total_events:
                raise RuntimeError(
                    f"Output validation failed: length mismatch in {tmp_output}: "
                    f"{len(f['events'])} != {total_events}"
                )
            actual_sig = 0
            step = 100_000
            for start in range(0, total_events, step):
                actual_sig += int(f["events"]["is_signal"][start : start + step].sum())
            if actual_sig != n_sig:
                raise RuntimeError(
                    f"Output validation failed: n_signal mismatch in {tmp_output}: "
                    f"data has {actual_sig} but expected {n_sig}"
                )

        tmp_output.replace(output)
    except Exception:
        if tmp_output.exists():
            tmp_output.unlink()
        raise
    LOGGER.info("Wrote %s (%d events)", output, total_events)


def main() -> int:
    config = load_config("config.yaml")
    setup_logging(config.log_path)
    load_to_hdf5(config)
    print_timings(LOGGER)
    return 0


if __name__ == "__main__":
    sys.exit(main())
