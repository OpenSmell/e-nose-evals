# A→B Transition Recording Protocol

> **Canonical source.** This copy lives in `e-nose-evals` and is versioned
> there. The readable hub is `OpenSmell/docs/`.

**Status:** specification, not yet collected. Supersedes nothing; complements
`docs/protocols/wire-protocol.md` (wire format), `electronic-nose/SAMPLING_CONTRACT.md` (cadence),
and `data-commons/FORMAT.md` (submission schema).

**Why this exists.** Every dataset currently in the project measures a *steady
state*: a substance is presented, the response is allowed to settle, and the
settled value is used. That is the easy case. A deployed e-nose sitting in a room
encounters transitions — someone opens a perfume bottle — and the interesting
question is what the array does *during* the change. The existing evidence says
this is not a small effect:

- Sensor memory is bi-exponential. On the UCI dynamic-mixtures archive a
  two-exponential fit beat a single-exponential fit in 97.3% of 2928 fits (8/8
  sensor families across 2 files, worst family 0.889), with fast τ ≈ 8–25 s and
  slow τ ≈ 45–60 s (`reports/bench_dynamic_memory.json`,
  `opensmell-rs/docs/sensor-memory-and-identifiability.md` §2).
- A single-exponential memory model is therefore wrong for this hardware, and
  any transition classifier built on one will mispredict the tail.
- Priming is real but **not a single scalar**: self-priming *raised* the next
  response in 27/32 config-channel groups, with median Δ = +0.206 and +0.128 on
  the two files, but the sign is reversed in the remaining 5 groups. It must be
  tracked per channel, and it must not be assumed positive.
- On the Wörner Smelldect corpus, ≈12% of the deflection is still unstabilised
  after a 5-minute air gap, and ≈15% after 246 s
  (`reports/woerner_memory_validation.md`). A protocol with a short air gap
  between transitions will read that residue as signal for B.

The last point is the design constraint that drives everything below. **The
transition is not separable from the carryover unless the gap is long enough and
the gap itself is measured.** An A→B design that omits a matched B-alone control
cannot distinguish "the array responds to a change" from "the array responds to
whatever is still adsorbed from A."

---

## 1. Design

### 1.1 Conditions

Four conditions, each a full recording on the standard protocol:

| ID | Sequence | Purpose |
|----|----------|---------|
| `A-ON` | air → A | Reference response to A alone |
| `B-ON` | air → B | Reference response to B alone |
| `AB` | air → A → B | The transition of interest |
| `BA` | air → B → A | Reciprocal transition, same hardware, same session |

`A-ON` and `B-ON` are not optional controls. They are what makes the AB
transition interpretable: without them, any difference between `AB`'s B-phase
and a standalone B cannot be attributed to the transition rather than to
session-to-session drift, which the same corpus shows is substantial.

Both orderings are collected because the priming sign is channel- and
gas-specific. Running only `AB` would leave open the possibility that the effect
is asymmetric, and an asymmetric effect cannot be modelled from one direction.

### 1.2 Substance selection

A and B must differ on **at least one sensor channel's selectivity profile**, and
that difference must be larger than the within-substance day-to-day spread of
that channel. Concretely, before committing to a pair:

1. Collect `A-ON` and `B-ON` on at least 3 separate days.
2. Compute, per channel, the standardized mean difference
   (Hedges' *g*) between A and B responses.
3. Keep pairs where at least one channel has |*g*| > 2 across days and the sign
   is stable in every replicate.

Rejecting a pair on this criterion is cheaper than collecting a transition whose
B-phase signal is indistinguishable from baseline. If no pair on the available
array clears |*g*| > 2, the honest conclusion is that the array cannot resolve
the pair at all, which is itself a result worth reporting.

### 1.3 Timing

All durations are wall-clock and must be recorded, not assumed:

| Phase | Duration | Rationale |
|-------|----------|-----------|
| Pre-baseline in clean air | ≥ 300 s | Must exceed the slow τ (45–60 s) by ≥5× so R0 is not contaminated by prior history |
| Exposure A | 60 s | Long enough for the fast mode to complete (τ_fast ≈ 8–25 s), short enough to stay pre-saturation |
| **Air gap between A and B** | **≥ 300 s** | See §1.4. This is the load-bearing parameter. |
| Exposure B | 60 s | Same as A |
| Post-recovery | ≥ 300 s | To observe the tail; short of the slow τ, the fit is censored |

The gap is a variable, not a constant — see §1.4.

### 1.4 The gap sweep

A single gap length cannot distinguish a transition effect from memory. Run the
AB sequence at a minimum of three gap lengths spanning the memory timescale:

| Arm | Air gap | Expected residue from A (bi-exp fit) |
|-----|---------|----------------------------------------|
| `AB-short` | 30 s | Substantial — 0.5–0.7 τ_slow |
| `AB-mid` | 120 s | ~2 τ_slow; decayed but not baseline |
| `AB-long` | 600 s | >10 τ_slow; near the noise floor |

The arms are placed relative to τ_slow (45–60 s), not chosen roundly: 30 s is
inside the slow mode, 120 s is roughly the median gap that made the UCI
analysis uninterpretable as a transition measurement, and 600 s is far enough
out that a residual response cannot be attributed to memory.

The measurement that matters is the **slope of the B-phase peak response against
gap length**. Under a pure-memory account with no separate transition effect,
that slope is negative and flattens toward zero as the gap grows. A residual
positive response at `AB-long`, above the `B-ON` reference, is evidence for a
transition effect that memory does not explain.

This is the same design used in the UCI dynamic-mixtures gap analysis, where the
response-vs-gap correlation was negative in 75% and 61.5% of configurations
(median *r* = −0.361 and −0.414, over 336 and 208 configurations). That corpus
did not sweep gap length as a designed factor, only as whatever the schedule
happened to produce, so the slope is not directly comparable — but the sign is
the right prior, and it held across two independently generated gas pairs.

**Why a designed sweep is necessary rather than merely better.** Recovering the
commanded schedule from that corpus
(`e-nose-evals/data/dynamic-mixtures/PROVENANCE.md`) shows why the existing
analysis cannot answer the question this protocol asks. Its inter-transition
intervals have a median of **100 s** and a p90 of 189–200 s, against a measured
τ_slow of 45–60 s. The median gap is therefore only **1.7–2.2 τ_slow**: the
previous stimulus has decayed five- to ninefold but has not returned to
baseline. Only about 10% of intervals exceed 300 s.

So in that corpus the "clean gap" is not clean at the median transition, and the
strength of the negative correlation is largely a statement about how short the
schedule's gaps were. The result is a valid measurement of history dependence
under realistic scheduling — but it cannot separate a fresh response from residue,
because it rarely provides the washout. A random schedule cannot be analysed into
a designed one. That is the gap this protocol fills.

### 1.5 Replicates and counterbalancing

- ≥ 5 replicates per condition per day, ≥ 3 days.
- Randomize the order of conditions within a day. A fixed order confounds
  condition with drift: whatever is measured last is always measured on the most
  drifted sensor.
- **Randomize A and B physical positions** (left/right inlet, near/far) between
  replicates, or position effects become a fixed bias attributed to the gas.
- Log ambient temperature and relative humidity with every recording. The
  environment audit found drift, environment, and mixture nonlinearity are
  first-order corruptors for MOX arrays (`reports/README.md` findings 4–6, 13).

### 1.6 Session structure

```
[≥300 s clean air baseline] → [60 s A] → [gap: 30|120|600 s clean air]
   → [60 s B] → [≥300 s recovery]
```

Repeat the whole sequence with A and B exchanged, then with positions exchanged.
Between sequences, return to clean air for ≥ 600 s so the next sequence's
baseline is not contaminated by the previous one's B.

---

## 2. Required metadata

Every recording must carry these, or the transition timing is unrecoverable —
which is exactly the failure mode of the August 2026 reference recordings:

| Field | Why |
|-------|-----|
| `sampling_rate_hz` | Declared cadence; validated against timestamps at submission |
| `phase_labels` | Per-sample `baseline` / `exposure_a` / `gap` / `exposure_b` / `recovery` |
| `phase_transition_times` | Exact host-clock time of each phase change |
| `substance_a`, `substance_b` | Identity and concentration of each |
| `concentration_a`, `concentration_b` | Response is dose-dependent; a transition at unequal doses is a different experiment |
| `inlet_position_a`, `inlet_position_b` | Randomized; must be recoverable to test for position bias |
| `temperature_celsius`, `humidity_percent` | Recorded with the recording, not reconstructed afterwards |
| `device_id`, `channel_map` | Channel-to-physical-sensor mapping; selectivity ratios are meaningless without it |

`phase_labels` is the field that makes the dataset usable. Without it a consumer
must infer phase boundaries from the response itself, and the response is what is
being measured — the inference is circular and will assign the transition to the
wrong sample range.

If the host cannot emit per-sample labels, it must at minimum emit the exact
transition timestamps, so labels can be reconstructed by index arithmetic. A
transition recorded to the nearest sample rather than the nearest second carries a
timing error of up to one sample period, which at 2 Hz is ±500 ms against a
transition whose fast mode completes in 8–25 s.

---

## 3. Features to compute

Static steady-state features are computed on the A-phase and B-phase windows
separately and compared. The transition-specific features are the point:

| Feature | Definition | What it discriminates |
|---------|------------|----------------------|
| `transition_latency_s` | Interpolated time from phase change to 10% of the B-phase step | Whether the array responds to the *change* or only to the new steady state |
| `transition_overshoot` | max response after phase change / final B-phase value | Relaxation overshoot; distinguishes MOX from a simple RC filter |
| `bilinear_degree` | Fit y = a·x_A + b·x_B + c·x_A·x_B + d to (x_A, x_B, y) across the four arms | Whether the array is linear in a mixture at all |
| `carryover_index` | (AB B-phase peak − B-ON peak) / (B-ON peak − baseline) | Direct measure of residue from A; must fall toward 0 as the gap grows |
| `gap_slope` | Slope of B-phase peak vs gap length, per channel | The §1.4 measurement |
| `hysteresis_path_ratio` | Adsorption vs desorption time integral, dimensionless | Asymmetric approach and release; see `framework_features.compute_channel_health` |

`carryover_index` is the headline quantity and the reason the matched controls
exist. It is undefined without `B-ON` and meaningless without a gap sweep, so it
cannot be computed retrospectively from a steady-state dataset.

Two features must **not** be reported from this protocol:

- Anything requiring sub-`2Δt` event resolution. At 2 Hz the minimum
  resolvable event is ≈1 s.
- Absolute concentration estimates. Nothing here is calibrated; §4 states the
  position on that.

---

## 4. Claims this protocol does and does not support

**Supports:**

- Whether a transition response is separable from carryover, and at what gap.
- Whether the effect is symmetric, from the paired `AB` / `BA` arms.
- Whether relaxation is bi-exponential in the transition setting specifically,
  as it is in the steady-state recovery fits.
- Per-channel priming direction, which is not reducible to a scalar.

**Does not support:**

- Absolute concentration. Two-point calibration per device is required first
  (`calibration.py`), and this protocol does not include reference concentrations
  for that purpose.
- Cross-device transfer. Theorem 2 in the paper establishes this is
  mathematically impossible without calibration; nothing in a single-device
  protocol changes that.
- Generalization beyond A and B. Two substances are not a mixture space.

---

## 5. Analysis plan

Pre-registered before collection, to keep the analysis from being chosen after
seeing the gap sweep:

1. **Primary:** per-channel `carryover_index` vs gap length. Fit a line per
   channel. Report the slope and its confidence interval. Prediction under a
   pure-memory account: slope negative, consistent with zero at `AB-long`.
2. **Secondary:** `transition_latency_s` distribution, `AB` vs `BA`, and whether
   the difference exceeds within-session replicate spread.
3. **Tertiary:** bilinear fit on the four arms; report `c` with an interval and
   do not claim significance from a single fit.
4. **Falsification:** if `carryover_index` at `AB-long` is indistinguishable
   from `B-ON` within replicate noise, the transition effect is not resolved by
   this protocol and the result is reported as a null, not as a small effect.
5. **Stop rule:** if the A/B pair fails the |*g*| > 2 screen in §1.2 mid-collection,
   stop and report the separability failure rather than collecting the full design
   on a pair the array cannot resolve.

---

## 6. Implementation status

| Item | Status |
|------|--------|
| Protocol specification | This document |
| Timed phase/event records in `.osmell` | **Implemented** — `SessionEvent(label, start_ms, end_ms)` (`opensmell/opensmell/types.py:270`), serialized to and from `events.json` (`opensmell/opensmell/io.py:45`) |
| Session-level role/label | **Implemented but insufficient** — `SessionDescriptor.role` is a single value per session (`types.py:184`); a transition recording needs five ordered phases within one session |
| Event message on the wire | **Not implemented** — `wire-protocol.md` defines `OSM`, `INFO`, `CAL`, `ERR`, `PING`, `PONG`, `SWAP`. No message type carries a phase boundary, so a live host cannot emit phase boundaries into the stream |
| Phase-label validation at submission | **Implemented** — `data-commons/validate_submission.py::_check_events` rejects overlaps, inverted intervals, and a ~1000× unit mismatch against the CSV timestamp column, and reports off-grid boundaries |
| `events` in the exchange schema | **Implemented** — `data-commons/FORMAT.md` §"Optional: phase events"; accepts `start_ms`/`end_ms` or `start_s`/`end_s` |
| §2 environmental/randomization fields in schema | **Partial** — `temperature_celsius` and `humidity_percent` exist as scalar metadata; `inlet_position_*` and `concentration_*` do not exist |
| Gap-sweep driver | **Not implemented** |
| Data | None collected |

The container for phase labels already exists and needed no redesign:
`events.json` with ordered `SessionEvent` records (`opensmell/opensmell/types.py:270`,
serialized at `io.py:45`) covers §2's `phase_labels` exactly, and the 600 s
`AB-long` arm is an event boundary rather than a different file. What is still
missing is everything that *produces* the labels: no wire message emits an event,
and no driver sequences the arms — a host has to annotate phase boundaries
against its own clock after the fact, which reintroduces exactly the estimated-
boundary error the validator now reports.

The gap between this document and the code is deliberate and worth stating
plainly: the design is settled enough to collect against, but nothing has been
collected, and no claim about transitions is currently supported by this
project's own data.