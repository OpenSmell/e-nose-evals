# U6 — Smell Taxonomy

## Dataset

250 recordings of 50 substances, 6 MOX channels (NO2, C2H5OH, VOC, CO, Alcohol, LPG), 60 s windows at 30 s stride -> 4903 windows. OSMO families: Citrus 2, Floral 1, Fruity 10, Green 8, Herbal 7, Mineral 1, Soulful 2, Woody 19.

## Fine substance identity (50 classes, reference ceiling)

- **Evaluation:** stratified-group-6fold (n_groups=250)
- **Accuracy:** 89.4% (chance 2.0%, majority-class 2.3%)
- **Balanced accuracy:** 89.5%
- **F1 (macro):** 0.895
- **Edge over chance:** +87.4 pp
- **Edge over majority:** +87.2 pp
- **n:** 4903

| class | n | accuracy |
|---|---|---|
| 0 | 95 | 94.7% |
| 1 | 105 | 81.9% |
| 2 | 105 | 77.1% |
| 3 | 99 | 97.0% |
| 4 | 95 | 87.4% |
| 5 | 95 | 97.9% |
| 6 | 91 | 83.5% |
| 7 | 97 | 80.4% |
| 8 | 86 | 97.7% |
| 9 | 95 | 95.8% |
| 10 | 95 | 88.4% |
| 11 | 105 | 81.0% |
| 12 | 97 | 100.0% |
| 13 | 105 | 81.9% |
| 14 | 102 | 86.3% |
| 15 | 103 | 100.0% |
| 16 | 97 | 92.8% |
| 17 | 93 | 91.4% |
| 18 | 101 | 93.1% |
| 19 | 100 | 82.0% |
| 20 | 102 | 90.2% |
| 21 | 96 | 94.8% |
| 22 | 99 | 85.9% |
| 23 | 98 | 96.9% |
| 24 | 97 | 95.9% |
| 25 | 98 | 100.0% |
| 26 | 101 | 70.3% |
| 27 | 101 | 77.2% |
| 28 | 99 | 91.9% |
| 29 | 93 | 94.6% |
| 30 | 93 | 96.8% |
| 31 | 100 | 69.0% |
| 32 | 90 | 97.8% |
| 33 | 96 | 63.5% |
| 34 | 94 | 94.7% |
| 35 | 98 | 83.7% |
| 36 | 104 | 92.3% |
| 37 | 100 | 100.0% |
| 38 | 92 | 98.9% |
| 39 | 95 | 98.9% |
| 40 | 101 | 99.0% |
| 41 | 92 | 78.3% |
| 42 | 111 | 100.0% |
| 43 | 94 | 92.6% |
| 44 | 101 | 93.1% |
| 45 | 105 | 83.8% |
| 46 | 89 | 100.0% |
| 47 | 100 | 65.0% |
| 48 | 96 | 94.8% |
| 49 | 107 | 86.9% |

## Perceptual family (8 OSMO grand families, leave-one-substance-out)

- **Evaluation:** leave-one-substance-out
- **Accuracy:** 40.2% (chance 12.5%, majority-class 38.1%)
- **Edge over chance:** +27.7 pp
- **Fine-to-coarse collapse accuracy:** 35.7%
- **n:** 4903 (groups: 50 substances)

| family | n | accuracy |
|---|---|---|
| Citrus | 202 | 0.0% |
| Floral | 105 | 0.0% |
| Fruity | 990 | 28.5% |
| Green | 772 | 41.7% |
| Herbal | 686 | 21.7% |
| Mineral | 99 | 0.0% |
| Soulful | 181 | 0.0% |
| Woody | 1868 | 65.2% |

## Honesty notes

The fine task is the rig's reliable capability (recording-fair CV). The family task is the honest taxonomy claim: an unseen substance's OSMO perceptual category is much harder than its identity, because perceptual families span chemically diverse substances and MOX features capture redox response, not the molecular structure that drives human categories. The fine-to-coarse collapse shows whether family structure already exists inside fine predictions. Families are imbalanced (Woody 19/50, Fruity 10/50, ...), so the majority baseline is strict. These numbers describe **this rig and these 50 substances**; cross-device taxonomy transfer is a separate, weaker result (see interoperability experiments) and is not claimed here.
