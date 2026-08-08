# U2B — Temporal Gas-Leak Event Detection

## Protocol

Each recording: 60 s clean air, 180 s gas release, 60 s recovery. 20 s windows at 10 s stride. Labels come from the known timeline: gas flowing when the window centre lies in [60, 240] s.

## Gas-flowing window classification

- **Evaluation:** stratified-group-6fold (n_groups=180)
- **Accuracy:** 94.1% (chance 50.0%, majority-class 65.5%)
- **Balanced accuracy:** 94.4%
- **F1 (macro):** 0.935
- **Edge over chance:** +44.1 pp
- **Edge over majority:** +28.5 pp
- **n:** 5220

| class | n | accuracy |
|---|---|---|
| False | 1800 | 95.4% |
| True | 3420 | 93.3% |

## Onset detection on held-out recordings

Windows within ±20 s of the 60 s onset.

- **Evaluation:** stratified-group-6fold (n_groups=180)
- **Accuracy:** 84.2% (chance 50.0%, majority-class 60.0%)
- **Balanced accuracy:** 85.2%
- **F1 (macro):** 0.840
- **Edge over chance:** +34.2 pp
- **Edge over majority:** +24.2 pp
- **n:** 900

| class | n | accuracy |
|---|---|---|
| False | 360 | 90.0% |
| True | 540 | 80.4% |

## Detection latency

- Recordings: 180
- Detected on held-out recordings: 100.0%
- Median onset latency: 10.0 s
- 75th percentile latency: 10.0 s
