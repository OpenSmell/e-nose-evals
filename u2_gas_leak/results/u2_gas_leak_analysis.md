# U2 — Gas-Leak / Mixed-Gas Identification

## Dataset

180 wind-tunnel recordings, 8 Figaro TGS sensors, 60 s windows at 30 s stride from the exposure phase onward. Ethylene 0-96 ppm; CO 0-460 ppm; Methane 0-131 ppm.

## Power-law reference calibration

**ethylene:** best channel 4 → a = 0.742, b = 0.147, log-log R² = 0.319 (median R² across 8 channels 0.276)
**CO:** best channel 1 → a = 0.403, b = 0.175, log-log R² = 0.593 (median R² across 8 channels 0.320)
**Me:** best channel 1 → a = 0.444, b = 0.221, log-log R² = 0.604 (median R² across 8 channels 0.067)

## Leave-one-configuration-out concentration recovery

**ethylene:** median |log10(pred/true)| 0.19 decades (p75 0.37), 100% within one decade, n=288, config-order Spearman 1.000
**CO:** median |log10(pred/true)| 0.16 decades (p75 0.26), 94% within one decade, n=144, config-order Spearman 0.500
**Me:** median |log10(pred/true)| 0.30 decades (p75 0.41), 98% within one decade, n=96, config-order Spearman 0.500

Each decade is a factor of 10 in ppm: median errors of 0.16-0.30 decades mean typical single-channel power-law predictions are only within ~x1.4-2 of the true concentration. This is a rough dose-response reference for detection screening, **not** a certified quantification pipeline (see the framework regression tables for concentration estimation).

## Mixture identification (ethylene-only / gas2-only / both)

- **Evaluation:** stratified-group-6fold (n_groups=180)
- **Accuracy:** 89.2% (chance 25.0%, majority-class 60.0%)
- **Balanced accuracy:** 83.9%
- **F1 (macro):** 0.869
- **Edge over chance:** +64.2 pp
- **Edge over majority:** +29.2 pp
- **n:** 1260

| class | n | accuracy |
|---|---|---|
| CO | 126 | 73.0% |
| Me | 126 | 93.7% |
| both | 756 | 96.8% |
| ethylene | 252 | 72.2% |

## Secondary-gas identity (CO vs Methane, exposure only)

- **Evaluation:** stratified-group-6fold (n_groups=144)
- **Accuracy:** 94.9% (chance 50.0%, majority-class 50.0%)
- **Balanced accuracy:** 94.9%
- **F1 (macro):** 0.949
- **Edge over chance:** +44.9 pp
- **Edge over majority:** +44.9 pp
- **n:** 1008

| class | n | accuracy |
|---|---|---|
| CO | 504 | 95.2% |
| Me | 504 | 94.6% |

## ppm regression from framework features

- **Evaluation:** stratified-group-6fold (n_groups=180)
- **R²:** 0.678 (mean-predictor baseline 0.000)
- **MAE:** 14.610 (mean-predictor baseline 26.596)
- **RMSE:** 19.237
- **Spearman ρ:** 0.851
- **MAE improvement over mean predictor:** +45.1%
- **n:** 1260

---

- **Evaluation:** stratified-group-6fold (n_groups=180)
- **R²:** 0.854 (mean-predictor baseline 0.000)
- **MAE:** 40.847 (mean-predictor baseline 148.640)
- **RMSE:** 63.029
- **Spearman ρ:** 0.902
- **MAE improvement over mean predictor:** +72.5%
- **n:** 1260
