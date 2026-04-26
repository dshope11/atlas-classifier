# atlas-classifier

End-to-end PyTorch DNN pipeline for **ATLAS Open Data H→WW→2lνν** signal-vs-background classification in the 0-jet category. The 0-jet signal region is dominated by irreducible WW diboson background and has no reconstructable invariant mass (two neutrinos in the final state), so a discriminating classifier is where the analysis sensitivity comes from.

The pipeline goes from ATLAS Open Data ROOT ntuples → engineered features → trained DNN → quantitative significance comparison vs. a cut-based baseline, with a full set of evaluation diagnostics (ROC with confidence bands, KS overtraining check, weighted score distributions, two complementary feature-importance methods, twin-axis training curves with per-epoch Asimov significance).

> Developed using Claude Code; `CLAUDE.md` contains the technical context file for AI-assisted sessions.

---

## Headline result

On the held-out test set (15% of 173k events after selection):

| Metric | Cut-based baseline | DNN (this work) | Δ |
|---|---|---|---|
| Signal efficiency | 53% | 30% (working point) | — |
| Background rejection | 82% | 99.4% | — |
| Asimov significance Z₀ | **7.28σ** | **8.43σ** | **+1.15σ** |
| AUC | n/a | 0.890 | — |
| KS overtraining (signal / background p-value) | n/a | 0.86 / 0.12 | both > 0.05 ✓ |

Asimov significance follows Cowan et al. 2011, Eq. 97 (`√(2·[(s+b)·log(1+s/b) − s])`), reduces to S/√B in the s ≪ b limit and is the value ATLAS publications report. All yield-based numbers use the full physics weight `xsec × lumi × mcWeight × scale-factors`; ROC and KS are unweighted (they measure discriminator quality, not yields).

---

## Reproducing the results from scratch

The pipeline runs end-to-end in well under 10 minutes on Apple Silicon (~5 min for the data download, ~5 min for training; everything else is seconds).

### 1. Set up the Python environment (conda)

```bash
conda create -n atlas-classifier python=3.12 -y
conda activate atlas-classifier
pip install -r requirements.txt
```

PyTorch's MPS backend is used automatically on Apple Silicon; CUDA is used if available; otherwise CPU.

### 2. Download the ATLAS Open Data ROOT files (~2.5 GB, idempotent)

```bash
python scripts/download_data.py
```

This uses the `atlasopenmagic` library to discover the correct URLs for the 8 samples (6 H→WW production modes + WW + ttbar) in the **2to4lep** skim of the **2025e-13tev-beta** release, and saves them to `data/raw/mc_<DSID>.root`. Re-runs are no-ops; pass `--force` to re-download.

### 3. ROOT → HDF5 conversion (~10 s)

```bash
python src/data_loading.py
```

For each sample, opens the local ROOT file, applies the event-selection cuts (defined declaratively in `config.yaml`), computes per-event physics weights using the canonical 2025-release formula, and writes a single structured HDF5 dataset. Output: `data/processed/events.h5`.

### 4. Feature engineering and 70/15/15 split (<1 s)

```bash
python src/preprocessing.py
```

Computes the 5 composite features (`m_ll`, `pT_ll`, `dphi_ll`, `dphi_ll_met`, `m_T`), splits stratified by `is_signal`, and computes RobustScaler normalisation stats from the **train split only** (no test-set leakage). Stats are saved into `data/processed/split.h5` and ultimately travel inside the model checkpoint via `register_buffer`. Output: `data/processed/split.h5`.

### 5. Train the DNN (~5 min on M1 Max)

```bash
python src/train.py
```

Trains the DNN with `BCEWithLogitsLoss(pos_weight=n_bg/n_sig)`, Adam, `ReduceLROnPlateau`, early stopping on val loss, NaN guard, and per-epoch Asimov significance diagnostic on the val set. Restores best weights and writes a self-describing checkpoint plus a JSON history. Outputs:
- `data/processed/best_model.pt` (architecture + weights + norm stats + feature schema)
- `data/processed/loss_history.json` (epoch / train_loss / val_loss / val_auc / val_significance / lr)

### 6. Evaluate (~2 s)

```bash
python src/evaluate.py
```

Generates pre-fit and post-fit plots and prints the test-set summary. All written to `data/processed/eval/`:
- `features_signal_vs_background.png` — per-feature distributions, signal vs background
- `feature_correlation.png` — correlation heatmap on signal events
- `training_curves.png` — twin-axis val loss + val Asimov significance per epoch
- `roc.png` — ROC curve with Clopper–Pearson 1σ confidence bands; cut-based working point overlaid as a marker
- `score_distributions.png` — weighted score distributions; train (filled) vs test (line)
- `feature_importance.png` — permutation (global) and perturbation (local) side-by-side

### 7. Run the test suite

```bash
pytest tests/ -v
```

36 tests covering the cut DSL evaluator, Asimov significance formula, Δφ wrapping, `m_T` formula, stratified split correctness, scaler train-only fit, model forward shape and logit semantics, BCE numerical stability, and constructor validation.

---

## Repository layout

```
atlas-classifier/
├── config.yaml              # Single source of truth for all hyperparameters
├── notes/data_sources.md    # DSID table, weight formula, sample selection rationale
├── src/
│   ├── config.py            # TrainingConfig dataclass + YAML loader, type aliases
│   ├── sample_info.py       # SAMPLES registry (DSIDs + is_signal labels)
│   ├── utils.py             # asimov_significance, evaluate_cuts (cut DSL), chronomat
│   ├── data_loading.py      # ROOT (uproot) → HDF5 with selection + weights
│   ├── preprocessing.py     # Composite features + stratified split + scaler stats
│   ├── model.py             # HWWClassifier (in-model RobustScaler, raw-logit output)
│   ├── train.py             # Manual training loop with all the diagnostics
│   └── evaluate.py          # Full evaluation suite
├── scripts/
│   ├── download_data.py     # atlasopenmagic-driven ROOT download
│   └── inspect_h5.py        # Quick HDF5 structure dump
├── tests/                   # pytest suite (36 tests)
├── data/raw/                # ROOT files (gitignored)
├── data/processed/          # HDF5 + checkpoint + plots (gitignored)
└── logs/                    # training.log (appended)
```

---

## Intermediate data formats

The pipeline produces three disk artifacts; in-memory formats are transient.

| Artifact | Produced by | Contents |
|---|---|---|
| `data/raw/mc_<DSID>.root` | `download_data.py` | Raw ATLAS Open Data ntuples |
| `data/processed/events.h5` | `data_loading.py` | Selected events: flat per-event scalars (lead/sublead lepton kinematics, MET, jet count), `event_weight`, `is_signal`. Composite features are **not** stored here. |
| `data/processed/split.h5` | `preprocessing.py` | Computed feature matrix X (raw, not normalised), labels y, physics weights w — one group per split (train/val/test). Scaler stats (median, IQR) from the train split only. |
| `data/processed/best_model.pt` | `train.py` | PyTorch checkpoint: weights + in-model normalisation stats (`register_buffer`) + feature schema. Self-describing — no external sidecar needed at inference time. |

A few non-obvious details worth calling out:

- **Uproot iterates in 100 MB chunks.** The ttbar file is 1.9 GB; loading it in one shot would blow memory. Awkward arrays handle the ragged per-object branches (variable-length `lep_pt` per event); after the 2-lepton selection the ragged structure resolves to flat lead/sublead scalars.
- **Composite features are computed at preprocessing time, not stored in `events.h5`.** This keeps the HDF5 schema close to the raw ntuple and makes adding a feature a one-file change (extend `FEATURE_FNS` in `preprocessing.py`, re-run preprocessing).
- **Normalisation lives inside the model, not in `split.h5`.** The split file stores raw feature values; the model's `forward()` applies `(x − median) / IQR` via `register_buffer` tensors. This means the checkpoint is fully self-describing and the scaler stats can't drift out of sync with the weights.

---

## Key design decisions

### Weight handling: counts-based class balancing in the loss; physics weights in evaluation only

Signal events are rare by construction (small cross section). Naively passing the full physics weight (`xsec × luminosity × mcWeight × scale-factors`) to `BCEWithLogitsLoss` would suppress signal contributions to gradient updates, producing a "classifier collapse" toward predicting background everywhere. Instead:

| Stage | Weighting |
|---|---|
| Class balance in the loss | `pos_weight = n_background / n_signal` from event **counts** (no physics rates) |
| Per-event loss weighting | none |
| Feature normalisation | unweighted |
| Yield estimates / significance / score-distribution plots | full `event_weight` (xsec × lumi × …) |

The heuristic: *if including a weight changes the relative balance between signal and background, keep it out of training.* Physics weights belong where physics yields belong — in evaluation.

### `RobustScaler` (median ± IQR) over `StandardScaler`

HEP kinematic distributions are heavy-tailed (pT, mass). Mean and standard deviation are pulled by outliers; median and inter-quartile range are not. The choice has small but real impact on training stability with extreme-pT events.

### Feature functions are self-contained and independently testable

Each composite feature (`m_ll`, `pT_ll`, `dphi_ll`, `dphi_ll_met`, `m_T`) is a standalone function in `preprocessing.py` that takes the full structured event array and returns one column. Some intermediate quantities (e.g. dilepton system px/py) are recomputed across functions rather than cached. This is intentional: preprocessing completes in under a second, and self-contained functions can be unit-tested in isolation without shared mutable state. The alternative — building a mutable intermediate dict of computed columns — would be more efficient but would couple the functions together and complicate the test suite.

### In-model normalisation via `register_buffer`

Median and IQR live as non-parameter tensors *inside* the model, registered via `register_buffer`. They move with `.to(device)` and are saved in the checkpoint automatically. The `.pt` file is fully self-describing for inference: no external scaler-stats sidecar is needed, and feature-order mismatches are caught at load time via the saved `feature_names` attribute.

### Asimov significance, not S/√B

S/√B is a small-signal approximation. The Asimov formula `Z = √(2·[(s+b)·log(1+s/b) − s])` (Cowan et al. 2011, Eq. 97) handles the s ~ b regime correctly and is what ATLAS publications report. Reduces to S/√B when s ≪ b. One extra line of code, more defensible number.

### Two complementary feature-importance methods

- **Permutation importance** (global): shuffle each feature across the test set, measure AUC drop.
- **Perturbation importance** (local): shift each feature by ±0.01σ at each event's actual value, measure mean |Δscore|.

They measure different things — global discrimination vs. local gradient sensitivity. Reporting both makes disagreements interpretable: a feature that ranks high on perturbation but low on permutation suggests sharp local gradients in a region that doesn't contribute much global separation.

### ROC unweighted; significance weighted

The ROC curve measures the discriminator's ranking quality as a pure function — it's a property of the classifier, not the experiment. The Asimov significance number is what carries the physics: yields × cross-section × luminosity. Mixing the two would conflate "is this classifier good" with "is the analysis sensitive". Clopper–Pearson 1σ binomial confidence bands on the ROC make the statistical uncertainty visible.

### Cut DSL in `config.yaml`

Event-selection cuts live as a small nested boolean dict in `config.yaml`, evaluated by a 30-line recursive function. The selection is auditable and version-controlled without touching code:

```yaml
cuts:
  and:
    - ["jet_n", "== 0"]
    - ["lep_n", "== 2"]
    - ["lep_charge_product", "< 0"]
    - ["lead_lep_pt", "> 25"]
    - ["sublead_lep_pt", "> 15"]
```

### 3-way 70/15/15 split (no k-fold cross-validation)

Single stratified train/val/test split. Train teaches; val is the only signal from outside the loop (early stopping, LR scheduling, per-epoch significance); test is touched once for final reported metrics. K-fold would buy little at this dataset size (170k events) and would conflate the role of val with that of test.

### Training runs to completion; one checkpoint saved

The training loop runs in a single session (~5 min on M1 Max) and saves only the best checkpoint — the weights from the epoch with the lowest validation loss. Intermediate per-epoch checkpoints are not written. The checkpoint is for evaluation and inference only: optimizer state, LR scheduler state, and epoch counter are not saved, so training cannot be resumed from it. This is a deliberate simplification: at this dataset size a single session is cheap enough that resume capability isn't worth the added complexity.

---

## What this repo doesn't do

Scoped out of v1 with one-line writeup notes:

- **Systematic uncertainties.** The Education-and-Outreach data tier doesn't expose per-event systematic variations as branches; treating them properly would require re-running with separate systematics ROOT files. Aware of the gap.
- **Background normalisation factors from control regions.** Real ATLAS analyses fit MC normalisations from data sidebands. We use luminosity weights only. Adds complexity without changing the ML demonstration.
- **Hyperparameter optimisation.** Defaults are manually validated. The infrastructure for an Optuna sweep (TPE search over learning rate, architecture, dropout, batch size) is sketched in the planning notes but not implemented in v1.
- **Real collision data.** ATLAS Open Data includes real pp collision data alongside the MC, but this pipeline uses MC only. In a real analysis the classifier would run over collision data, and a profile likelihood fit (`data = μ × signal_MC + background_MC`) would extract the signal strength μ — significance is then how many sigma μ is from zero. Here, significance is computed entirely from MC yields using the Asimov approximation (treating the MC prediction as a stand-in for data). This is standard practice for feasibility studies and classifier development before unblinding.
- **BDT comparison.** XGBoost on the same feature set is a natural addition (and Gini importance contrasts cleanly with the two DNN-importance methods); planned but not in v1.

