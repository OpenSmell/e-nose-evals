# U2c — Dynamic Gas Mixtures (ethylene_CO.txt)

## Protocol

ethylene_CO.txt: 420827 samples at 10 Hz (decimated from 100 Hz). Windows 60 s at 30 s stride, 1051 windows. Concentration setpoints change every 80-120 s; 274 episodes. Labels and episode groups come from the window centre time. Clean air 31.4% of the time.

## Leak detection — any gas flowing (binary)

- **Evaluation:** StratifiedGroupKFold
- **Accuracy:** 98.1% (chance 50.0%, majority-class 68.6%)
- **Balanced accuracy:** 97.9%
- **F1 (macro):** 0.978
- **Edge over chance:** +48.1 pp
- **Edge over majority:** +29.5 pp
- **n:** 1051

| class | n | accuracy |
|---|---|---|
| False | 330 | 97.3% |
| True | 721 | 98.5% |

## Four-state discrimination (air / ethylene / gas2 / both)

- **Evaluation:** StratifiedGroupKFold
- **Accuracy:** 95.8% (chance 25.0%, majority-class 31.4%)
- **Balanced accuracy:** 95.8%
- **F1 (macro):** 0.956
- **Edge over chance:** +70.8 pp
- **Edge over majority:** +64.4 pp
- **n:** 1051

| class | n | accuracy |
|---|---|---|
| 0 | 330 | 97.3% |
| 1 | 196 | 98.0% |
| 2 | 300 | 94.7% |
| 3 | 225 | 93.3% |

## Ethylene ppm regression

- **Evaluation:** GroupKFold
- **R²:** 0.924 (mean-predictor baseline 0.000)
- **MAE:** 0.719 (mean-predictor baseline 5.227)
- **RMSE:** 1.583
- **Spearman ρ:** 0.906
- **MAE improvement over mean predictor:** +86.2%
- **n:** 1051

## Gas2 ppm regression

- **Evaluation:** GroupKFold
- **R²:** 0.918 (mean-predictor baseline 0.000)
- **MAE:** 19.167 (mean-predictor baseline 146.738)
- **RMSE:** 45.669
- **Spearman ρ:** 0.900
- **MAE improvement over mean predictor:** +86.9%
- **n:** 1051
