# U3 — Food-Spoilage Monitoring

## Dataset

12 beef cuts x 2220 min at 1 min resolution, 11 MQ sensors, TVC measured hourly (range 1.88-5.76 log10 CFU/g). 60 min windows at 30 min stride -> 876 windows; labels are the current-hour TVC / its freshness recoding at the window centre.

## Sensor signal

- Pooled per-channel Spearman with the current-hour TVC:
  MQ5 +0.93, MQ4 -0.90, MQ137 +0.83, MQ135 -0.63, MQ3 -0.60, MQ8 -0.57, MQ136 -0.49, MQ9 -0.45, MQ6 +0.36, MQ2 -0.31, MQ138 -0.25

Direction differs by sensor model (MQ5/MQ137 rise while MQ4/MQ135 fall with TVC), as expected for a mixed MOX chamber. The reported tasks test whether the full framework window features give reliable cross-cut prediction.

## TVC regression (log10 CFU/g)

- **Evaluation:** LeaveOneGroupOut
- **R²:** 0.793 (mean-predictor baseline 0.000)
- **MAE:** 0.384 (mean-predictor baseline 0.935)
- **RMSE:** 0.495
- **Spearman ρ:** 0.849
- **MAE improvement over mean predictor:** +59.0%
- **n:** 876

## Four-class freshness (Label 1-4)

- **Evaluation:** LeaveOneGroupOut
- **Accuracy:** 78.4% (chance 25.0%, majority-class 60.3%)
- **Balanced accuracy:** 64.5%
- **F1 (macro):** 0.642
- **Edge over chance:** +53.4 pp
- **Edge over majority:** +18.2 pp
- **n:** 876

| class | n | accuracy |
|---|---|---|
| 1 | 106 | 81.1% |
| 2 | 138 | 66.7% |
| 3 | 104 | 17.3% |
| 4 | 528 | 93.0% |

## Binary spoiled? (TVC >= 5)

- **Evaluation:** LeaveOneGroupOut
- **Accuracy:** 85.0% (chance 50.0%, majority-class 60.3%)
- **Balanced accuracy:** 84.8%
- **F1 (macro):** 0.845
- **Edge over chance:** +35.0 pp
- **Edge over majority:** +24.8 pp
- **n:** 876

| class | n | accuracy |
|---|---|---|
| 0 | 348 | 83.3% |
| 1 | 528 | 86.2% |

## Per-cut holdout (leave-one-cut-out, 4-class)

Accuracy on each held-out cut (model trained only on the other 11).

| cut | n | accuracy |
|---|---|---|
| 1.Inside-Outside | 73 | 80.8% |
| 10.Clod_Chuck | 73 | 58.9% |
| 11.Shin | 73 | 83.6% |
| 12.Fat | 73 | 71.2% |
| 2.Round | 73 | 86.3% |
| 3.Top_Sirloin | 73 | 86.3% |
| 4.Tenderloin | 73 | 91.8% |
| 5.Flap_meat | 73 | 75.3% |
| 6.Striploin | 73 | 86.3% |
| 7.Rib_eye | 73 | 84.9% |
| 8.Skirt_meat | 73 | 54.8% |
| 9.Brisket | 73 | 84.9% |

Pooled: 78.8% (chance 25.0%, majority class 60.3%)

## Honesty notes

Label 1-4 is the dataset's own deterministic recoding of hourly TVC, so classification and regression are two lenses on the same signal. MQ columns are raw readings in arbitrary units whose response direction to spoilage differs across sensor models (MQ4/MQ135 fall while MQ5/MQ137 rise with TVC); the framework features are scale-invariant ratios, so relative features are robust to the unknown transducer gain, but any absolute-ppm features are uncalibrated for this dataset. The intermediate freshness class 3 (TVC 4-5) is badly separated in this dataset (per-class accuracy ~17%), so the four-class number is driven by the easy fresh/spoiled endpoints — the binary spoiled? task is the more decision-relevant read-out. All models were validated leave-one-cut-out (trained on 11 cuts, tested on a whole new cut), so within-cut autocorrelation and the monotone spoilage trend cannot leak. With 12 cuts from a single lab/rig and per-hour TVC plateaus, these numbers describe **this study's rig**; cross-device or cross-lab generalization is untested and not claimed.
