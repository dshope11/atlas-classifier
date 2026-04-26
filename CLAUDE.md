# ATLAS Open Data Classifier — Claude Code Schema

## Project Goal

End-to-end ML pipeline for binary signal/background classification on ATLAS Open Data.
Demonstrates physics-motivated feature engineering, a PyTorch DNN, and rigorous evaluation
including a cut-based statistical baseline for quantitative "before/after ML" comparison.

**Success criteria:** GitHub repo with a working pipeline, ROC curve, KS overtraining check,
and a measurable significance improvement over the cut-based baseline.

---

## Locked Decisions — Do Not Re-Derive

| Decision | Choice | Reason |
|---|---|---|
| Channel | H→WW→2lνν, signal vs. WW+tt̄ | No reconstructable invariant mass → ML has maximum leverage; WW is the irreducible background that defines the analysis |
| Jet category | 0-jet (veto ≥1 jet) | tt̄ negligible after jet veto; WW dominant; Δφ_ll most powerful; cleanest ML story |
| Background normalisation | MC luminosity weights only | No NFs from CRs — simplification noted in writeup; adds complexity without changing the ML demonstration |
| Dataset | ATLAS O&E 2025e-13tev-beta release, `2to4lep` skim | Python-native, fits on laptop, pre-slimmed ntuples |
| Intermediate format | HDF5 | Avoid re-running uproot each session |
| Loss function | BCEWithLogitsLoss + pos_weight=n_bg/n_sig | Raw logit output; class-count balancing in loss only |
| Framework | PyTorch (`nn.Module`) | No Lightning; manual training loop |
| Systematics | Scoped out of v1 | O&E tier; one sentence in writeup suffices |
| Stats baseline | Cut-based, before ML | Quantitative benchmark for significance improvement |

---

## Directory Structure

```
atlas-classifier/
  config.yaml            ← single source of truth for all hyperparameters + cut DSL
  notes/
    data_sources.md      ← DSID table, weight formula, sample selection rationale
  src/
    config.py            ← TrainingConfig dataclass + YAML loader, type aliases
    sample_info.py       ← SAMPLES registry (DSIDs + is_signal labels)
    utils.py             ← asimov_significance, evaluate_cuts (cut DSL), chronomat
    data_loading.py      ← uproot + awkward → HDF5 with selection + weights
    preprocessing.py     ← feature engineering, normalization, 3-way split
    model.py             ← HWWClassifier (in-model RobustScaler, raw-logit output)
    train.py             ← manual training loop
    evaluate.py          ← full evaluation suite
  scripts/
    download_data.py     ← atlasopenmagic-driven ROOT file download
    inspect_h5.py        ← quick HDF5 structure dump
  tests/
    test_preprocessing.py
    test_model.py
    test_utils.py
  data/
    raw/            ← ROOT files (gitignored)
    processed/      ← HDF5 + checkpoint + plots (gitignored)
  logs/             ← training.log (appended)
  requirements.txt
  CLAUDE.md
```

---

## Pipeline

```
0. Download (scripts/download_data.py)
   - Fetches 8 ROOT files (6 H→WW modes + WW + tt̄) from 2025e-13tev-beta via atlasopenmagic
   - Output: data/raw/mc_<DSID>.root  (idempotent; --force to re-download)

1. Load data (src/data_loading.py → data/processed/events.h5)
   - 2to4lep skim; filter to exactly 2 leptons (skim allows up to 4)
   - Compute pre-cut scalars (lead/sublead pT, charge product)
   - Apply cut DSL from config.yaml: 0-jet, OS leptons, pT thresholds
   - Assign is_signal from SAMPLES registry in sample_info.py
   - Compute per-event physics weight:
       event_weight = mcWeight × xsec × filteff × kfac × (lumi / sum_of_weights) × scalefactors
   - Output: flat structured HDF5 array; composite features NOT stored here

2. Preprocess (src/preprocessing.py → data/processed/split.h5)
   - Compute 5 composite features: m_ll, pT_ll, dphi_ll, dphi_ll_met, m_T
   - 70/15/15 stratified split (train/val/test) — stratified on is_signal
   - RobustScaler stats (median, IQR) from train split only → saved in checkpoint
   - Output: split.h5 with raw (un-normalised) X/y/w per split + scaler stats

3. Train (src/train.py → data/processed/best_model.pt + loss_history.json)
   - BCEWithLogitsLoss with pos_weight=n_bg/n_sig (counts-based class balance)
   - Adam + ReduceLROnPlateau + early stopping on val loss (patience=15)
   - Per-epoch Asimov Z diagnostic on val set at 30% signal efficiency WP
   - Saves best checkpoint (lowest val loss) — self-describing, no external sidecar

4. Evaluate (src/evaluate.py → data/processed/eval/)
   - Cut-based baseline: m_T < 125 GeV AND dphi_ll < 1.8 rad → Asimov Z "before ML"
   - ROC + AUC with Clopper–Pearson 1σ bands; cut WP overlaid as marker
   - Score distributions: signal vs background, train/test overlaid, KS stat annotated
   - Feature importance: permutation (global AUC drop) + perturbation (local ±0.01σ)
   - Training curves: twin-axis val loss + val Asimov Z per epoch
```

---

## Feature Engineering

Composite variables must be computed — they are not stored in the ntuples.

**Key raw branches (`2to4lep` skim, 2025e release):**
- `lep_pt`, `lep_eta`, `lep_phi`, `lep_e`, `lep_charge` — variable-length per event; indexed [lead, sublead] after 2-lepton filter
- `met`, `met_phi`
- `jet_n`

**Composite variables to compute:**
- `m_ll` — dilepton invariant mass (4-vector sum)
- `pT_ll` — dilepton system pT
- `dphi_ll` — Δφ between the two leptons; small for signal (Higgs spin-0 → collinear leptons)
- `dphi_ll_met` — Δφ between dilepton system and MET
- `m_T` — WW transverse mass:
    `m_T = sqrt(2 * pT_ll * met * (1 - cos(dphi_ll_met)))`
    This is the primary cut-based discriminant. Kinematic edge at m_W for WW/top backgrounds;
    extends to m_H (~125 GeV) for signal. The stats baseline cut lives here.

---

## Evaluation

**Required outputs (all implemented):**
1. ROC curve + AUC with Clopper–Pearson 1σ bands; cut-based WP overlaid as marker
2. Score distributions — signal vs background, train (filled) vs test (line), weighted;
   KS stat + p-value for signal and background annotated directly on the plot (flag if p < 0.05)
3. Asimov significance at working point (Cowan et al. 2011, Eq. 97) — compare DNN vs cut-based baseline
4. Train/val loss curves per epoch — twin axis with val Asimov Z
5. Feature ranking — permutation importance (global AUC drop after shuffling) AND
   perturbation importance (local mean |Δscore| at ±0.01σ shift) — side-by-side bar chart

**Expected result:** Δφ_ll should rank highly — it is the primary discriminant between
signal (Higgs spin-0 → collinear leptons, small Δφ_ll) and WW background (back-to-back
leptons, large Δφ_ll). If Δφ_ll does not rank near the top, investigate.

**Reference (ATLAS-restricted, for documentation only):**
Run 3 VH(WW) NRML Workshop 2025 slides:
https://indico.cern.ch/event/1498256/contributions/6454020/attachments/3056092/5403155/NRML%20Workshop%202025%20--%20Intro+3L.pdf

**Neyman-Pearson note:** A well-trained binary classifier approximates the likelihood ratio
L(signal|x)/L(background|x), which is the theoretically optimal test statistic by the
Neyman-Pearson lemma. The significance improvement over the cut-based baseline is a
quantitative demonstration of this.

---

## Workflow Rules

- **Plan mode first** — before writing or editing any code, enter plan mode. No code without a plan.
- **Self-validation hook** — configured in `.claude/settings.json`; runs pytest + mypy after
  every file edit. Treat failures as blockers. Fix before moving on.
- **HDF5 intermediate** — never re-run the full uproot pipeline mid-session. Always load from
  `data/processed/`. Regenerate only if the schema changes.
- **No Lightning** — plain `nn.Module` and manual training loop only.
- **Writeup-driven** — every design decision should be explainable in one sentence for the README.
- **Python environment** — use `conda run -n atlas-classifier python` for any ad-hoc Python
  commands in Bash. The post-edit hook already uses the explicit env path; bare `python` resolves
  to system Python and will fail with missing-module errors.

---

## Reference Links

- ATLAS Open Data: https://opendata.atlas.cern/
- O&E dataset details (13 TeV 25 ns): https://opendata.atlas.cern/docs/data/for_education/13TeV25_details
- Notebook repo (reference for data loading + sample labelling):
  https://github.com/atlas-outreach-data-tools/notebooks-collection-opendata
- atlasopenmagic: https://opendata.atlas.cern/docs/atlasopenmagic
- CERN Open Data Forum: https://opendata-forum.cern.ch
