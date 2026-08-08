# U2c — Dynamic Gas Mixtures (ethylene_methane.txt)

## Protocol

ethylene_methane.txt: 417851 samples at 10 Hz (decimated from 100 Hz). Windows 60 s at 30 s stride, 1392 windows. Concentration setpoints change every 80-120 s; 348 episodes. Labels and episode groups come from the window centre time. Clean air 32.8% of the time.

## Leak detection — any gas flowing (binary)

- **Evaluation:** StratifiedGroupKFold
- **Accuracy:** 98.1% (chance 50.0%, majority-class 67.2%)
- **Balanced accuracy:** 97.7%
- **F1 (macro):** 0.978
- **Edge over chance:** +48.1 pp
- **Edge over majority:** +30.8 pp
- **n:** 1392

| class | n | accuracy |
|---|---|---|
| False | 456 | 96.7% |
| True | 936 | 98.7% |

## Four-state discrimination (air / ethylene / gas2 / both)

- **Evaluation:** StratifiedGroupKFold
- **Accuracy:** 96.7% (chance 25.0%, majority-class 32.8%)
- **Balanced accuracy:** 96.6%
- **F1 (macro):** 0.967
- **Edge over chance:** +71.7 pp
- **Edge over majority:** +63.9 pp
- **n:** 1392

| class | n | accuracy |
|---|---|---|
| 0 | 456 | 97.1% |
| 1 | 337 | 98.2% |
| 2 | 317 | 95.0% |
| 3 | 282 | 96.1% |

## Ethylene ppm regression

- **Evaluation:** GroupKFold
- **R²:** 0.922 (mean-predictor baseline 0.000)
- **MAE:** 0.621 (mean-predictor baseline 4.974)
- **RMSE:** 1.546
- **Spearman ρ:** 0.886
- **MAE improvement over mean predictor:** +87.5%
- **n:** 1392

## Gas2 ppm regression

- **Evaluation:** GroupKFold
- **R²:** 0.952 (mean-predictor baseline 0.000)
- **MAE:** 5.682 (mean-predictor baseline 64.768)
- **RMSE:** 16.875
- **Spearman ρ:** 0.949
- **MAE improvement over mean predictor:** +91.2%
- **n:** 1392
