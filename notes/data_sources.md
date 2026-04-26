# Data Sources & Sample Metadata

This file documents where every piece of sample metadata in `src/sample_info.py` came from, the luminosity weighting formula and its correspondence to the ATLAS Open Data conventions, and the rationale behind sample-selection decisions. Companion to the code; intended to be self-documenting for anyone reading the repo.

---

## Source: ATLAS Open Data 2025 release (`2025e-13tev-beta`)

We use the **2025e-13tev-beta** release of the ATLAS Open Data, accessed via the official `atlasopenmagic` Python library. This is the current Education & Outreach release, accompanying the 2025 ATLAS Open Data publication. Underlying data is the same 2015+2016 13 TeV pp run, but the ntuple format and metadata have been modernised since the 2020 release that older notebooks reference.

```python
import atlasopenmagic as atom
atom.set_release('2025e-13tev-beta')
```

Per-sample metadata used to be in a separate `infofile.py`; in the 2025 release the equivalent values are stored **as branches in the ntuple itself** (`xsec`, `filteff`, `kfac`, `sum_of_weights`). The data loader reads these from the per-event branches at load time — no external metadata file needed.

Skim used: **`2to4lep`** (events with 2–4 reconstructed leptons after preselection).

---

## DSID table

For an H→WW→2lνν 0-jet analysis, signal is **H→WW→lvlv** in all production modes (ggF, VBF, WH, ZH, ggZH); background is **fully-leptonic WW (Sh_2212_llvv_os)** plus **dileptonic ttbar (PhPy8 nonallhad)**.

### Signal samples (H→WW→2lνν, all production modes)

| Key | DSID | Generator | xsec (pb) | nEvents |
|---|---|---|---|---|
| ggH125_WWlvlv | 345324 | Powheg+Pythia8 NNLOPS | 1.11 | 167,000 |
| VBFH125_WWlvlv | 345948 | Powheg+Pythia8 | 0.08602 | 180,000 |
| WpH125_qqWWlvlv | 345325 | Powheg+Pythia8 MINLO | 0.86215 | 100,000 |
| WmH125_qqWWlvlv | 345433 | Powheg+Pythia8 MINLO | 0.54016 | 120,000 |
| ZH125_vvWWlvlv | 345445 | Powheg+Pythia8 MINLO | 0.1501 | 50,000 |
| ggZH125_ZinclWWlvlv | 346524 | Powheg+Pythia8 MINLO | 0.05744 | 40,000 |

### Background samples

| Key | DSID | Generator | xsec (pb) | nEvents | Notes |
|---|---|---|---|---|---|
| WW_llvv | 700602 | Sherpa 2.2.12 | 12.079 | 900,000 | Fully-leptonic WW→lνlν, opposite-sign |
| ttbar_nonallhad | 410470 | Powheg+Pythia8 A14 | 729.77 | 32,890,000 | Includes dileptonic + semileptonic; 0-jet veto suppresses semileptonic |

> **Note on metadata source:** The xsec values shown above are from `atlasopenmagic.get_all_metadata()`. The ntuple branches (`xsec`, `filteff`, `kfac`, `sum_of_weights`) sometimes report different "raw" generator-level values; the **canonical weight formula uses the ntuple branches** (see below), so what's in the table is for human reference only.

---

## Sample selection rationale

For an H→WW→2lνν analysis (2-lepton channel):

- **All H→WW production modes are signal**: ggF dominates (~85%), then VBF, then WH/ZH/ggZH. We include them all, weighted by their respective cross sections, to get the realistic signal mixture an analysis would target.
- **WW→lνlν** (`Sh_2212_llvv_os`, DSID 700602) is the irreducible diboson background — same final state as the signal, distinguished only by kinematics. This is the equivalent of the old release's `llvv` (DSID 363492).
- **Dileptonic ttbar** (`PhPy8 nonallhad`, DSID 410470) is the dominant non-diboson background. After the 0-jet veto in our event selection, ttbar is heavily suppressed (top → b-jet; vetoing jets removes most ttbar) but residual contribution is correctly weighted.

Samples to **avoid**:
- DSIDs 341456/341458/341460 (ZH→VνWW→lνqq) — semi-leptonic WW final state, wrong for 2-lepton channel
- Same-sign WW (`Sh_2212_llvv_ss`, DSID 700603) — produces same-sign leptons, not the OS signature we cut for

---

## Luminosity weighting formula

Per-event physics weight (saved to HDF5, used in evaluation only — see "Weight Design" below):

```python
event_weight = (lumi * xsec * kfac * filteff / sum_of_weights) * mcWeight \
             * ScaleFactor_PILEUP * ScaleFactor_ELE * ScaleFactor_MUON * ScaleFactor_LepTRIGGER
```

This is the canonical formula from the 2025 ATLAS Open Data `ttbar_analysis.ipynb`, generalised from single-lepton (electron-or-muon scale factors) to the 2-lepton case (multiply both flavour scale factors — `ScaleFactor_ELE` is 1.0 for events with no electrons and similarly for muons, so the product is safe).

### Field-by-field correspondence

| Term | Source in ntuple | Notes |
|---|---|---|
| `lumi` | external constant | 36.1 fb⁻¹ × 1000 = 36100 pb⁻¹ (see below) |
| `xsec` | per-event branch (replicated value) | "Raw" generator-level cross section in pb |
| `kfac` | per-event branch (replicated value) | k-factor (NLO/NNLO correction) |
| `filteff` | per-event branch (replicated value) | Generator-level filter efficiency |
| `sum_of_weights` | per-event branch (replicated value) | Sum of mcWeights over all generated events for this sample |
| `mcWeight` | per-event branch | Generator weight for this specific event |
| `ScaleFactor_*` | per-event branches | Pre-computed product over leptons in event; 1.0 for absent flavour |

The ttbar reference notebook also includes `ScaleFactor_FTAG` (b-tagging SF). We omit it since the 0-jet veto means no jets to b-tag.

---

## Luminosity value: 36.1 fb⁻¹

We use `lumi = 36.1 fb⁻¹` as the default in `data_loading.py`. The picture across sources:

| Source | Value | Notes |
|---|---|---|
| ATLAS-published precise value (2015+2016) | **36.1 fb⁻¹** | From the ATLAS luminosity publication; what we use |
| Official ATLAS O&E docs | 36 fb⁻¹ | Rounded for general readers |
| 2025 release `ttbar_analysis.ipynb` | 36000 pb⁻¹ = 36 fb⁻¹ | Notebook convention; rounded |
| Older 2020 release `HZZAnalysis.ipynb` | 36.6 fb⁻¹ | Pre-revision estimate — do not use |

`lumi` is a configurable parameter in `TrainingConfig` (in `config.yaml`), so this can be revisited without code changes.

---

## Variable names and units (2025 release)

| Branch | Type | Units | Notes |
|---|---|---|---|
| `lep_n` | `int32_t` | — | Number of selected leptons (2 in 2to4lep skim) |
| `lep_pt` | `RVec<float>` | GeV | Per-lepton (variable length) |
| `lep_eta`, `lep_phi`, `lep_e` | `RVec<float>` | rad/GeV | Same |
| `lep_charge` | `RVec<int32_t>` | ±1 | Same |
| `lep_type` | `RVec<int32_t>` | — | 11 = electron, 13 = muon |
| `met` | `float` | GeV | Missing transverse energy magnitude (renamed from old `met_et`) |
| `met_phi` | `float` | rad | Missing transverse energy direction |
| `jet_n` | `int32_t` | — | Number of jets (renamed from old `n_jets`) |
| `mcWeight` | `float` | — | Per-event generator weight |
| `xsec`, `kfac`, `filteff`, `sum_of_weights` | scalar (per-file) | pb / unitless | Replicated as a column across all events |

> **Units:** the 2025 release stores all energies/momenta in **GeV** (older releases used MeV). All cuts in `config.yaml` are in GeV.

---

## Weight design (for orientation)

Physics weights (`event_weight`) are computed at load time and saved to HDF5, but they are **not** used in training. The training loss uses counts-based class balancing (`pos_weight = n_background / n_signal` in `BCEWithLogitsLoss`); physics weights are applied only in evaluation (yields, score-distribution plots, significance calculations).

Rationale: signal has small cross section by construction. Weighting the loss by `xsec × lumi` would suppress signal contributions to gradient updates, defeating the purpose of training a discriminator. See the project README for the full discussion.

---

## Pattern attribution

Loading and weighting patterns adapted from the 2025 ATLAS Open Data notebook collection:
- **`ttbar_analysis.ipynb`** — clearest implementation of the per-event `calc_weight` formula using the new in-ntuple metadata branches
- **`atlasopenmagic` library** — used to discover sample URLs and metadata for a given release + skim

Sample selection (which DSIDs constitute "signal" vs. "background" for the 2-lepton 0-jet channel) is our own decision based on the physics of the H→WW→2lνν channel — there is no template H→WW notebook in the O&E repo to copy from.
