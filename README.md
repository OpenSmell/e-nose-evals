# OpenSmell e-nose-evals

Reproducible evaluation suite for electronic-nose machine learning with the
OpenSmell framework: six experiments (U1–U6) run on public datasets through one
shared harness, scored under a recording-fair protocol, every headline carrying
its null baseline in the same table.

**This is an evaluation suite with a fixed protocol.** Six experiments, one shared
harness, recorded with the same rules every time: recording-fair grouping (no
window leaks across the train/test boundary), every headline carrying its null
baseline in the same table, and committed result artifacts with provenance. Some
numbers are seed-averaged (U5's k-shot curve averages 5 seeds); the suite is
built to be re-run end-to-end, so every claim is checkable against the checked-in
artifacts in this repo.

## The six experiments (U1–U6)

| Exp | Question | Data | Headline (all recording-fair) |
|-----|----------|------|------------------------------|
| U1 | Harness end-to-end on synthetic data | — | `python selftest.py` green |
| U2 | Gas leak / mixed-gas identification | UCI turbulent (309), UCI dynamic (322) | 94.1% gas-present; R² 0.92–0.95 dynamic ppm |
| U3 | Food spoilage against microbial ground truth | Harvard Dataverse beef cuts | TVC R² 0.793, MAE 0.384 log₁₀ CFU/g |
| U4 | Indoor-air event detection | UCI-362 home activity | 87.6% stimulus detection (balanced 74.1%) |
| U5 | Rig chemoprinting + per-rig calibration | UCI drift (146), 10 batches | rig fingerprint 78.5% (chance 25%); zero-shot 52.3% vs ceiling 99.5% |
| U6 | Smell taxonomy: identity vs perceptual family | SmellNet offline + OSMO | 50-substance identity 89.4%; family (LSO) 40.2% ≈ majority 38.1% |

Full numbers, per-class detail, and the analysis are in each
`uN_*/results/*_metrics.json` and `*_analysis.md`.

## Pipeline (`harness/`)

```
dataset loader  ->  windowing + framework feature extraction  ->  evaluation  ->  report
harness.loaders     harness.features (opensmell.features)         harness.evaluate   harness.report
```

| Module | Role |
|--------|------|
| `loaders.py` | Normalize public datasets to `(X, channels, time, meta, sr)` bundles; sensor readings converted to resistance (kOhm) so the framework feature extractor sees physically meaningful units. |
| `features.py` | Sliding-window extraction of the full framework feature set (device-agnostic + absolute + temporal + health + hardware + decay + saturation) per window, parallelized. |
| `evaluate.py` | Classification (chance + majority baselines) and regression (mean-predictor baseline, Spearman) under `StratifiedGroupKFold` so windows from the same recording never leak across folds. |
| `report.py` | Emits `metrics.json` (with `generated_utc` provenance) and `analysis.md` per experiment. |

The feature extractor comes from the [OpenSmell SDK](https://github.com/opensmell/opensmell)
(`opensmell.features`); this suite evaluates what that framework does on real public data.

## Install & run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python selftest.py                                    # U1 — no dataset needed
python u2_gas_leak/run_experiment.py                  # needs data/turbulent-mixtures/
python u2_gas_leak/run_event_detection.py             # needs data/turbulent-mixtures/
python u2_gas_leak/run_dynamic_experiment.py          # needs data/dynamic-mixtures/
python u3_food/run_experiment.py                      # bundled data/beef-spoilage/
python u4_indoor_air/run_experiment.py                # needs data/indoor-air/
python u5_chemoprint/run_experiment.py                # bundled data/uci-drift/
python u6_taxonomy/run_experiment.py                  # bundled data/smellnet-offline/
```

Each script writes `results/*_metrics.json` and `results/*_analysis.md`.

## Data

Bundled in this repo (small, derived, research/validation-only): UCI drift
(10 batch files), SmellNet `offline_training` (250 recordings), the OSMO
substance→family map, and the 12-cut beef-spoilage sheets.

Re-downloadable, never committed (~1.5 GB of binaries): UCI turbulent mixtures
(309), UCI dynamic mixtures (322), UCI-362 indoor air. Provenance, licensing,
conversion formulas, and download pointers: `data/DATASETS.md`.

## Honesty rules

- No window leaks across the train/test boundary (grouped CV by recording /
  cut / induction / substance — the group key is an explicit, versioned
  argument, and U6's family split was audited after a grouping bug was caught).
- Headline numbers always carry their null baseline in the same table
  (chance, majority class, mean predictor).
- The power-law calibration claim is limited to reference-point per-rig
  calibration — never zero-shot cross-device.
- The drift batches are one physical array aging over time; results describe
  these batches/this array, not distinct manufactured hardware.

## License

MIT (see `LICENSE`). Note: the UCI corpora and SmellNet data are
**research/validation-only** even where their license badge reads CC BY 4.0 —
see `data/DATASETS.md`.

## See also

- [OpenSmell project](https://github.com/opensmell) — SDK, web, hardware, science layer
- [OpenSmell Academy](https://opensmell.onrender.com) — the "U-suite" essay walks this suite and its summary table
- `session-invariance` / `encoder` / `interoperability` — companion evaluation and representation-learning work
