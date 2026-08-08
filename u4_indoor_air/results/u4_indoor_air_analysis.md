# U4 — Indoor-Air Monitoring

## Dataset

99 inductions over home conditions (wine: 36, banana: 33, background: 30), 8 Figaro MOX channels (R1, R2, R3, R4, R5, R6, R7, R8), 60 s windows at 60 s stride (evenly capped at 40 per induction) -> 3960 windows. Temp./Humidity excluded so the claim stays MOX-only. Per-induction class counts: background 30, banana 33, wine 36.

## Binary event detection (stimulus vs background, leave-one-induction-out)

- **Evaluation:** LeaveOneGroupOut
- **Accuracy:** 87.6% (chance 50.0%, majority-class 81.7%)
- **Balanced accuracy:** 74.1%
- **F1 (macro):** 0.768
- **Edge over chance:** +37.6 pp
- **Edge over majority:** +5.9 pp
- **n:** 3960

| class | n | accuracy |
|---|---|---|
| 0 | 3234 | 95.4% |
| 1 | 726 | 52.8% |

## Three-class discrimination (background / wine / banana, leave-one-induction-out)

- **Evaluation:** LeaveOneGroupOut
- **Accuracy:** 86.0% (chance 33.3%, majority-class 81.7%)
- **Balanced accuracy:** 55.0%
- **F1 (macro):** 0.561
- **Edge over chance:** +52.7 pp
- **Edge over majority:** +4.4 pp
- **n:** 3960

| class | n | accuracy |
|---|---|---|
| 0 | 3234 | 96.8% |
| 1 | 414 | 63.8% |
| 2 | 312 | 4.5% |

## Honesty notes

Both tasks use leave-one-induction-out, so every test window comes from a continuous recording whose sensor response was never seen in training — the strictest honest split for indoor monitoring. The binary task is the product-relevant claim (an odor event is present or not); the three-class task asks which event. These numbers describe **this Figaro array and these three home conditions**; the product rig is a different MQ array, so cross-device transfer is a separate, weaker result (see interoperability experiments) and is not claimed here.
