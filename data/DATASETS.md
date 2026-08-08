# Dataset Registry — OpenSmell e-nose-evals

All public datasets here are for **research and validation only** — never for
product training. This registry records provenance, licensing, and status for
every dataset the evaluation suite consumes. Small derived datasets (drift,
SmellNet offline, OSMO taxonomy, beef-spoilage) are **bundled in this repo** so
the suite reproduces standalone; the large raw corpora (turbulent, dynamic,
indoor-air) are re-downloadable and never committed (~1.5 GB of binaries).

| Dataset | Dir | Sensors | Source | License | Status |
|---------|-----|---------|--------|---------|--------|
| UCI gas sensor array exposed to turbulent gas mixtures | `turbulent-mixtures/` | 8 (TGS2600/02/10/11/12, TGS2620) | UCI id 309, Fonollosa 2014 | CC BY 4.0* | re-downloadable (180 recordings) |
| UCI gas sensor array under dynamic gas mixtures | `dynamic-mixtures/` | 16 (TGS2600/02/10/20 ×4) | UCI id 322, Fonollosa 2015 | CC BY 4.0* | re-downloadable (2 × 12 h) |
| Beef spoilage electronic nose | `beef-spoilage/` | 11 MQ | Harvard Dataverse DOI 10.7910/DVN/XNFVTS | CC0 1.0 | ✅ bundled (12 cut sheets) |
| UCI gas sensor array drift | `uci-drift/` | 16 | UCI id 146 | CC BY 4.0* | ✅ bundled (10 batches) |
| SmellNet offline_training | `smellnet-offline/` | 6 MOX | HF `DeweiFeng/smell-net` | research | ✅ bundled (250 recordings) |
| OSMO taxonomy | `taxonomy/` | — | OSMO/Google | research | ✅ bundled |
| UCI gas sensors for home activity monitoring | `indoor-air/` | 8 Figaro (TGS) + Temp/Humidity | UCI id 362, Huerta 2016 | CC BY 4.0* | re-downloadable (99 inductions) |

\* UCI pages state "research purposes only; commercial use excluded" in the
description even where the license badge reads CC BY 4.0. Treat as
**research-only** — including the bundled drift files.

## Turbulent mixtures (UCI 309)

- 180 recordings = 30 configurations (Ethylene + CO or Methane at 4 flow
  levels: zero/low/medium/high) × 6 repeats.
- Protocol per recording: 60 s clean air → 180 s gas release → 60 s recovery.
- 11 columns: time (s), temperature (°C), RH (%), 8 sensor ADC counts.
  Convert to resistance: `Rs(kOhm) = 10·(3110 − A)/A`.
- Mean ppm (GC-MS): Ethylene 0/31/46/96; CO 0/270/397/460; Methane
  0/51/115/131.
- Files: `dataset_twosources_raw/` (20 ms) and `dataset_twosources_downsampled/`
  (100 ms). Filename encodes levels: `007_Et_L_Me_H`.

## Dynamic mixtures (UCI 322)

- Two ~12 h continuous recordings at 100 Hz: `ethylene_CO.txt` (643 MB) and
  `ethylene_methane.txt` (638 MB), ~4.21 M samples each.
- 19 columns: time (s), gas2 conc (ppm), ethylene conc (ppm), 16 sensor
  readings. Convert: `Rs(kOhm) = 40000/S` (the UCI page writes `40.000/S_i`;
  ~0.1-150 kOhm from the data confirms the 40,000 scale). Sensor order:
  TGS2602×2, TGS2600×2, TGS2610×2, TGS2620×2, then the same 8 again.
- The files include a one-line header (comma-separated) that must be skipped
  when parsing with a whitespace separator. A short startup transient and
  transition spikes give non-positive readings that convert to NaN (~1.2% of
  samples).
- Concentration ranges: Ethylene 0-20 ppm; CO 0-600 ppm; Methane 0-300 ppm.
  Transitions every 80-120 s to random levels — ideal for temporal leak-event
  detection and concentration regression.
- Zip (369 MB) downloaded and verified via a retrying background loop (UCI
  serves no Range headers, so failed attempts restart from zero).

## Beef spoilage (Harvard Dataverse 10.7910/DVN/XNFVTS)

- "Dataset for electronic nose from various beef cuts" (Wijaya et al., Telkom
  University), CC0 1.0, v2 released 2022-04-02. Downloaded as the original
  multi-sheet xlsx and as the Dataverse-converted `.tab`.
- 12 sheets, one per beef cut (Inside-Outside, Round, Top_Sirloin, Tenderloin,
  Flap_meat, Striploin, Rib_eye, Skirt_meat, Brisket, Clod_Chuck, Shin, Fat).
  Each cut: 2220 minute-by-minute rows sampled once per minute at room
  temperature while the meat spoils.
- Columns: `Minute, TVC, Label, MQ135, MQ136, MQ137, MQ138, MQ2, MQ3, MQ4,
  MQ5, MQ6, MQ8, MQ9`. TVC = total viable count in log10 CFU/g, measured
  hourly and held constant within each hour block.
- `Label` is a deterministic Excel-formula recoding of TVC against
  microbiological thresholds: <3 -> 1 (fresh), 3-4 -> 2, 4-5 -> 3, >=5 -> 4
  (spoiled). Per-cut label shares are e.g. 300/360/240/1320 for most cuts.
- TVC range across cuts: ~1.88-5.76 log10 CFU/g, monotone non-decreasing
  within each cut. MQ columns are raw readings in arbitrary units; response
  direction to spoilage differs by sensor model.
- Derived per-cut CSVs in `cuts/` (extracted by `extract.py` from the xlsx,
  which is the source of record; the single-cut converted `.tab` is dropped
  as redundant).

## Indoor air, home activity monitoring (UCI 362)

- "Gas sensors for home activity monitoring" (Huerta et al. 2016, UCSD/IBEC),
  Figaro TGS array (R1 TGS2611, R2 TGS2612, R3 TGS2610, R4 TGS2600,
  R5/R6 TGS2602, R7/R8 TGS2620) plus Sensirion SHT75 temperature/humidity.
- 100 inductions (99 present in the data; id 95 missing) of background home
  activity, with wine and banana presentations placed near the sensors:
  36 wine, 33 banana, 31 background. 928,991 one-second rows total.
- Two whitespace-separated files: `HT_Sensor_metadata.dat` (`id, date, class,
  t0, dt` — `dt` is the stimulus duration in hours) and `HT_Sensor_dataset.dat`
  (`id, time, R1..R8, Temp., Humidity`). `time=0` is stimulus onset; every
  induction includes ~1 h of pre/post-stimulus background.
- Windows are labeled *stimulus* only while `0 <= center_time < dt`; all other
  windows (pre/post and the background inductions) are *background*. The U4
  experiment uses the 8 MOX channels and excludes Temp./Humidity so the claim
  stays MOX-only.
- Mirror note: `archive.ics.uci.edu` is unreachable from this machine; the
  canonical `HT_Sensor_UCIsubmission.zip` was obtained from the GitHub mirror
  `AlejandroSantorum/Gas_sensors_home_activity_monitoring`
  (`datasets/raw/HT_Sensor_UCIsubmission.zip`). Source zip kept in this dir
  for provenance (sha256 `7c143b9f4402a8205ebe8072c2ce0967c25741f1865e23d650e981053294f395`).
