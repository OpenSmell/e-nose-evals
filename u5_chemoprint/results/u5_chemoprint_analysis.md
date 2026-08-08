# U5 — Chemoprint Per Rig

## Dataset

UCI Gas Sensor Array Drift: 16 MOX sensors, 6 gases, 10 batches over 36 months, 128 pre-extracted features per measurement (13910 measurements total). Batches model rigs / device-time states; batches [6, 7, 9, 10] cover all six gases with >=20 samples/gas and are used for fingerprinting, and the canonical early/late split ([1, 2, 3, 4, 5] -> [6, 7, 8, 9, 10]) drives the calibration curve.

## Rig fingerprinting (leave-gas-out)

- **Evaluation:** leave-gas-out (train on 5 gases, test on the 6th)
- **Pooled accuracy:** 78.5% (chance 25.0%)

| held-out gas | n | accuracy | chance | majority class |
|---|---|---|---|---|
| Ethanol | 1824 | 78.1% | 25.0% | 35.6% |
| Ethylene | 1891 | 68.8% | 25.0% | 35.0% |
| Ammonia | 1170 | 79.4% | 25.0% | 51.3% |
| Acetaldehyde | 1448 | 91.5% | 25.0% | 51.4% |
| Acetone | 1914 | 77.7% | 25.0% | 32.9% |
| Toluene | 1736 | 79.1% | 25.0% | 34.6% |

Rig identification generalizes across a never-seen gas if the per-rig response pattern (chemoprint) is gas-independent.

## Per-rig calibration curve

- **Source rigs:** 1, 2, 3, 4, 5 (3633 measurements)
- **Target rigs:** 6, 7, 8, 9, 10 (10277 measurements)
- **Chance:** 16.7% | **Majority class:** 20.0%
- **Zero-shot transfer:** 52.3%
- **In-target ceiling:** 99.5% (balanced 99.4%)

| k labeled samples/gas | accuracy (mean ± std) |
|---|---|
| 5 | 65.6% ± 1.5 |
| 10 | 73.4% ± 2.3 |
| 25 | 83.5% ± 0.9 |
| 50 | 91.2% ± 0.4 |

Each k averaged over 5 sampling seeds; the k samples are added to the source training set and the remaining target measurements are tested.

## Honesty notes

The drift batches are one physical array aging over time, so they model rig identity and device-time shift, not distinct manufactured hardware. Rig fingerprinting is evaluated leave-gas-out so the rig pattern must generalize to a never-seen gas to count as a real chemoprint; within-batch autocorrelation is not an issue because the split is by gas (a strict split). The calibration curve shows zero-shot cross-rig gas identification stays far below the per-rig ceiling (52% vs 99.5%), per-rig calibration with labeled samples is required to approach it, and the in-target ceiling is the realistic upper bound — numbers describe **these batches/this array** and should not be read as a guarantee for other hardware.
