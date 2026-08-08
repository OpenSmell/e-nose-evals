# Harness Self-Test

## Pipeline

Synthetic smoke test: 36 recordings at 3 levels, 8 channels, framework features per 60 s window.

## Classification

- **Evaluation:** stratified-group-6fold (n_groups=36)
- **Accuracy:** 100.0% (chance 33.3%, majority-class 33.3%)
- **Balanced accuracy:** 100.0%
- **F1 (macro):** 1.000
- **Edge over chance:** +66.7 pp
- **Edge over majority:** +66.7 pp
- **n:** 108

| class | n | accuracy |
|---|---|---|
| 0 | 36 | 100.0% |
| 1 | 36 | 100.0% |
| 2 | 36 | 100.0% |

## Regression

- **Evaluation:** stratified-group-6fold (n_groups=36)
- **R²:** 0.998 (mean-predictor baseline 0.000)
- **MAE:** 0.014 (mean-predictor baseline 0.667)
- **RMSE:** 0.039
- **Spearman ρ:** 0.960
- **MAE improvement over mean predictor:** +97.8%
- **n:** 108
