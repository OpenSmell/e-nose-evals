# Verification: UCI 322, Gas Sensor Array under Dynamic Gas Mixtures

Companion to `../DATASETS.md`, which records what this dataset *is*. This file
records what was **measured** from the two files after download, and one
consequence of the measurement that is not obvious from the dataset description.

Descriptive facts — chamber volume, flow rate, heater voltage, sensor types,
ppm ranges, licensing — are in `../DATASETS.md` and are deliberately not
repeated here.

- **Source:** Fonollosa, J. (2015). *Gas sensor array under dynamic gas
  mixtures* [Dataset]. UCI Machine Learning Repository.
  <https://doi.org/10.24432/C5WP4C> (donated 2015-03-19)

## Integrity

| File | Bytes | Size | Rows | Span |
|------|-------|------|------|------|
| `ethylene_CO.txt` | 642,945,087 | 613.2 MiB | 4,208,261 | 42,087.55 s |
| `ethylene_methane.txt` | 638,311,247 | 608.7 MiB | 4,178,504 | 41,790.19 s |

Sizes match the 613.2 MB / 608.7 MB the UCI listing reports.

```
sha256  ethylene_CO.txt        eb979ebf40f22bb7206b4fdd335fad6da51ad0289d2a37fdb1fff2f6e425487e
sha256  ethylene_methane.txt   a00f383f2734850cf61370dc687cb09e9b400648549567a2f6a43e5c6a99aaf2
```

Re-download and re-hash to confirm a fresh copy is intact. UCI serves no Range
headers, so a partial download cannot be resumed — it restarts from zero, which
is why these hashes are worth recording.

## Acquisition rate is 100 Hz

Measured median inter-sample gap is **0.0100 s in both files**: exactly
100.00 Hz. The gap histogram contains 0.01, 0.02, 0.03 … so the logger
occasionally emits extra rows. The median is the right estimator here and the
mean is not.

Several project documents describe analyses of this corpus as running "at
1 Hz". The acquisition is 100 Hz; **1 Hz is the analysis grid the benchmarks
downsampled onto.** The distinction is load-bearing:

- at 100 Hz the minimum resolvable event is 20 ms;
- a transition feature computed on a 1 Hz grid has a 2 s floor;
- anything reported as a 1 Hz result inherits the *analysis grid's* resolution,
  not the sensor's.

Per `opensmell/mox/timing.py::min_detectable_duration`, the resolvable floor is
`2 Δt`, not `Δt`.

## Recovered transition schedule

The commanded concentration columns change in discrete steps, so the ground-truth
schedule is recoverable from the data with no manual annotation. Detected by
step changes in those two columns.

| Statistic | `ethylene_CO` | `ethylene_methane` |
|-----------|---------------|--------------------|
| commanded changes | 372 | 351 |
| inter-transition intervals | 371 | 350 |
| min interval | 1 s | 2 s |
| p10 | 76 s | 79 s |
| **median interval** | **100 s** | **100 s** |
| p90 | 189 s | 200 s |
| max interval | 870 s | 613 s |
| intervals < 80 s | 42 | 35 |
| intervals in 80–120 s | 277 | 258 |
| intervals > 120 s | 52 | 57 |

The median matches the documented random draw from 80–120 s. The tails are
wider than that description implies: intervals run from 1 s to 870 s, and about
11% fall *below* the stated 80 s minimum.

## Consequence: the median gap in this corpus is not a clean gap

Recovery fits on this corpus put the slow time constant at **τ_slow ≈ 45–60 s**
(`opensmell-rs/docs/sensor-memory-and-identifiability.md` §2). Measured against
that:

| Interval | In units of τ_slow | Residue still present |
|----------|--------------------|-----------------------|
| p10 = 76–79 s | ~1.4 τ | substantial |
| median = 100 s | **~1.7–2.2 τ** | **decayed ~5–9×, not returned to baseline** |
| p90 = 189–200 s | ~3–4 τ | measurable |
| > 300 s (≈10% of intervals) | >5 τ | approaching clean |

So the "clean gap before an exposure" that the published gap-modulation analysis
correlates against is, at the median transition, **not clean**. This is the most
likely reason the measured response-vs-gap correlation is as strongly negative as
it is — median *r* = −0.361 and −0.414, negative in 75% and 61.5% of
configurations: the schedule seldom provides the washout needed to separate a
fresh response from residue of the previous one.

Two things follow:

1. The published gap analysis is a valid measurement of **history dependence
   under realistic scheduling**. It is **not** a measurement of transition
   response in isolation and must not be cited as the latter.
2. Isolating a transition needs a **designed** gap sweep spanning at least
   30–600 s. This corpus cannot supply one: its schedule is random, not
   factorial. See `docs/transition-protocol-ab.md` in the OpenSmell root.

## Caveats

- The concentration columns are **commanded setpoints**, not measurements.
  Anything about *achieved* concentration needs the delivery-platform
  calibration, which is not in this archive.
- No temperature or humidity columns. Chamber conditions are constant by
  construction and unmeasured, so environmental confounding cannot be checked
  from these files.
- The 1–2 s intervals at the low end sit inside the fast recovery mode
  (τ_fast ≈ 8–25 s). Those transitions are real but not resolvable as separate
  events on a 1 Hz analysis grid. Filter them, or report at the grid you used.