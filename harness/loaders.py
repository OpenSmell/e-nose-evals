"""Per-dataset loaders for the use-case experiments.

Every loader returns a normalized bundle:
    {
        "X":        ndarray (n_samples, n_channels, n_time) float64,
        "sr":       int sampling rate (Hz),
        "channel_names": list[str],
        "meta":     pandas DataFrame, one row per sample,
        "name":     str dataset key,
    }

Sensor readings are converted to resistance (kOhm) so the framework feature
extractor sees the same units it was designed for (higher resistance = more
reducing gas).
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"

TURBULENT_RAW = DATA / "turbulent-mixtures" / "dataset_twosources_raw"
TURBULENT_DOWN = DATA / "turbulent-mixtures" / "dataset_twosources_downsampled"
DYNAMIC_DIR = DATA / "dynamic-mixtures"
BEEF_DIR = DATA / "beef-spoilage"
BEEF_CUTS_DIR = BEEF_DIR / "cuts"

# UCI drift (10 batch files) is bundled in this repo's data dir so U5 runs
# standalone. It is the public UCI gas sensor array drift dataset (id 146),
# research/validation-only — see data/DATASETS.md.
DRIFT_DIR = DATA / "uci-drift"

DRIFT_GAS_NAMES = {1: "Ethanol", 2: "Ethylene", 3: "Ammonia",
                   4: "Acetaldehyde", 5: "Acetone", 6: "Toluene"}

# SmellNet offline_training (250 recordings, 50 substances) is bundled in this
# repo's data dir so U6 runs standalone. See data/DATASETS.md.
SMELLNET_OFFLINE = DATA / "smellnet-offline"

# Gas channels used for features. Benzene is excluded: its column carries a
# sentinel (2^32-1) in ~32% of rows. Gas_Resistance is BME680's raw estimate
# (not an MQ/MOX response) and Temperature/Pressure/Humidity/Altitude are
# environmental — all excluded to keep the chemoprint MOX-only.
SMELLNET_CHANNELS = ["NO2", "C2H5OH", "VOC", "CO", "Alcohol", "LPG"]

TURBULENT_CHANNELS = [
    "TGS2600", "TGS2602a", "TGS2602b", "TGS2620a",
    "TGS2612", "TGS2620b", "TGS2611", "TGS2610",
]

TURBULENT_PPM = {
    "ethylene": {"n": 0, "L": 31, "M": 46, "H": 96},
    "CO": {"n": 0, "L": 270, "M": 397, "H": 460},
    "Me": {"n": 0, "L": 51, "M": 115, "H": 131},
}

TURBULENT_NAME_RE = re.compile(
    r"^(?P<idx>\d{3})_Et_(?P<et>[nLMH])_(?P<gas2>CO|Me)_(?P<gas2level>[nLMH])$"
)

DYNAMIC_CHANNELS = [
    "TGS2602", "TGS2602", "TGS2600", "TGS2600",
    "TGS2610", "TGS2610", "TGS2620", "TGS2620",
    "TGS2602", "TGS2602", "TGS2600", "TGS2600",
    "TGS2610", "TGS2610", "TGS2620", "TGS2620",
]

BEEF_CHANNELS = [
    "MQ135", "MQ136", "MQ137", "MQ138", "MQ2", "MQ3",
    "MQ4", "MQ5", "MQ6", "MQ8", "MQ9",
]

# Dataset's own freshness recoding of hourly TVC (log10 CFU/g).
BEEF_LABEL_BINS = [3.0, 4.0, 5.0]   # <3 -> 1, 3-4 -> 2, 4-5 -> 3, >=5 -> 4

SR_RAW = 50
SR_DOWN = 10


def _to_resistance_8ch(arr):
    """Convert turbulent ADC counts A to Rs (kOhm) = 10 * (3110 - A) / A."""
    a = np.where(arr <= 0, np.nan, arr)
    r = 10.0 * (3110.0 - a) / a
    return r


def _to_resistance_16ch(arr):
    """Convert dynamic conductance values S to Rs (kOhm) = 40000 / S.

    The UCI dynamic-mixtures page gives 'Rs(KOhms) = 40.000/S_i' (European
    decimal notation for 40,000); the resulting ~0.1-150 kOhm range matches
    Figaro TGS sensor resistance, so 40000 is the intended scale.
    """
    s = np.where(arr <= 0, np.nan, arr)
    r = 40000.0 / s
    return r


def load_turbulent_mixtures(use_downsampled=False):
    """Load all 180 turbulent-mixture recordings (8 MOX sensors)."""
    root = TURBULENT_DOWN if use_downsampled else TURBULENT_RAW
    if not root.is_dir():
        raise FileNotFoundError(
            f"turbulent mixtures not found at {root} — download and extract the "
            "UCI dataset into e-nose-evals/data/turbulent-mixtures/"
        )
    sr = SR_DOWN if use_downsampled else SR_RAW

    samples, rows = [], []
    for path in sorted(root.iterdir()):
        if not path.is_file():
            continue
        m = TURBULENT_NAME_RE.match(path.stem)
        if not m:
            continue
        df = pd.read_csv(path, header=None, dtype=np.float64)
        values = df.values
        if values.shape[1] < 11:
            continue
        sensors = _to_resistance_8ch(values[:, 3:11])
        et_level = m.group("et")
        gas2 = m.group("gas2")
        gas2_level = m.group("gas2level")
        samples.append(sensors.T)
        rows.append({
            "file": path.stem,
            "gas2": gas2,
            "ethylene_level": et_level,
            "gas2_level": gas2_level,
            "ethylene_ppm": TURBULENT_PPM["ethylene"][et_level],
            "gas2_ppm": TURBULENT_PPM[gas2][gas2_level],
            "seconds": float(values[0, 0]),
            "duration_s": float(values[-1, 0] - values[0, 0]),
        })

    if not samples:
        raise ValueError(f"no valid recordings found in {root}")

    meta = pd.DataFrame(rows)
    meta["mixture"] = np.where(
        (meta["ethylene_ppm"] > 0) & (meta["gas2_ppm"] > 0), "both",
        np.where(meta["ethylene_ppm"] > 0, "ethylene",
                 np.where(meta["gas2_ppm"] > 0, meta["gas2"], "air")))
    X = np.stack(samples)
    return {
        "X": X, "sr": sr, "channel_names": list(TURBULENT_CHANNELS),
        "meta": meta, "name": "turbulent-mixtures",
    }


def load_dynamic_mixtures(file_name=None):
    """Load UCI dynamic-mixtures continuous time series (16 MOX sensors).

    file_name selects one mixture file: 'ethylene_CO.txt' or
    'ethylene_methane.txt' (default: use whichever exists).
    """
    if not DYNAMIC_DIR.is_dir():
        raise FileNotFoundError(
            f"dynamic mixtures not found at {DYNAMIC_DIR} — the UCI download is "
            "e-nose-evals/data/dynamic-mixtures/"
        )
    if file_name is None:
        candidates = [p for p in DYNAMIC_DIR.glob("*.txt")]
        if not candidates:
            raise FileNotFoundError(f"no *.txt files in {DYNAMIC_DIR}")
        file_name = candidates[0].name

    path = DYNAMIC_DIR / file_name
    if not path.is_file():
        raise FileNotFoundError(path)

    df = pd.read_csv(path, sep=r"\s+", header=None, skiprows=1, comment=None)
    if df.shape[1] < 19:
        raise ValueError(f"{file_name}: expected 19 columns, got {df.shape[1]}")

    time_s = df[0].values
    gas2_ppm = df[1].values
    et_ppm = df[2].values
    sensors = _to_resistance_16ch(df.iloc[:, 3:19].values)

    meta = pd.DataFrame({
        "file": file_name,
        "gas2_ppm": gas2_ppm,
        "ethylene_ppm": et_ppm,
    })
    return {
        "X": sensors, "sr": 100,
        "channel_names": list(DYNAMIC_CHANNELS), "meta": meta,
        "name": f"dynamic-mixtures/{file_name}",
    }


def _beef_label_from_tvc(tvc):
    """Recode hourly TVC (log10 CFU/g) into the dataset's freshness labels."""
    return np.digitize(tvc, BEEF_LABEL_BINS) + 1


def load_beef_spoilage():
    """Load the 12-cut beef-spoilage dataset (11 MQ sensors, per-minute).

    Each cut is one continuous 2220-minute spoilage series sampled once per
    minute; TVC was measured hourly and held constant within each hour block,
    so the per-minute label is the current hour's TVC and its recoded freshness
    class. The MQ columns are raw readings in arbitrary units whose response
    direction to spoilage differs across sensor models (MQ4/MQ135 fall, while
    MQ5/MQ137 rise with TVC); they are passed through unmodified — the
    framework features are scale-invariant ratios, so relative features are
    robust to the unknown transducer gain.
    """
    if not BEEF_CUTS_DIR.is_dir():
        raise FileNotFoundError(
            f"beef-spoilage cuts not found at {BEEF_CUTS_DIR} — extract the "
            "Dataverse xlsx with e-nose-evals/data/beef-spoilage/extract.py"
        )
    samples, rows = [], []
    for path in sorted(BEEF_CUTS_DIR.glob("*.csv")):
        df = pd.read_csv(path)
        sensors = df[BEEF_CHANNELS].to_numpy(dtype=np.float64)
        tvc = df["TVC"].to_numpy(dtype=np.float64)
        samples.append(sensors.T)
        rows.append({
            "cut": path.stem,
            "minutes": int(len(df)),
            "tvc_series": tvc,
            "label_series": _beef_label_from_tvc(tvc).astype(np.int64),
        })
    meta = pd.DataFrame(rows)
    return {
        "X": np.stack(samples), "sr": 1,
        "channel_names": list(BEEF_CHANNELS),
        "meta": meta, "name": "beef-spoilage",
    }


INDOOR_DIR = DATA / "indoor-air"

# UCI "Gas sensors for home activity monitoring" (dataset 362): 8 Figaro MOX
# channels R1-R8; Temp. and Humidity are environmental sensors and are
# excluded so the claim stays MOX-only, matching the other use cases.
INDOOR_CHANNELS = ["R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8"]

INDOOR_CLASSES = ["background", "wine", "banana"]


def load_indoor_air():
    """Load UCI-362 home-activity monitoring recordings (8 MOX channels, 1 Hz).

    The dataset ships two whitespace-separated files: HT_Sensor_metadata.dat
    (per-induction class, t0, dt) and HT_Sensor_dataset.dat (per-second rows:
    id, time, R1-R8, Temp., Humidity). For each induction, time=0 marks the
    stimulus onset and dt is the stimulus duration, so a window is "stimulus"
    only while 0 <= center_time < dt; everything else (pre/post stimulus and
    the pure-background inductions) is background. Recordings differ in
    length, so X is a list of (n_channels, n_time) arrays and `time` holds the
    per-induction time axes (hours, relative to stimulus onset).
    """
    if not INDOOR_DIR.is_dir():
        raise FileNotFoundError(
            f"indoor-air data not found at {INDOOR_DIR} — extract UCI dataset "
            "362 (HT_Sensor_UCIsubmission.zip) into data/indoor-air/"
        )
    meta = pd.read_csv(INDOOR_DIR / "HT_Sensor_metadata.dat", sep=r"\s+")
    dat = pd.read_csv(INDOOR_DIR / "HT_Sensor_dataset.dat", sep=r"\s+")

    samples, times, rows = [], [], []
    for ind_id, g in dat.groupby("id", sort=True):
        ind_meta = meta[meta["id"] == ind_id]
        if ind_meta.empty:
            continue
        g = g.sort_values("time")
        sensors = g[INDOOR_CHANNELS].to_numpy(dtype=np.float64).T
        samples.append(sensors)
        times.append(g["time"].to_numpy(dtype=np.float64))
        r = ind_meta.iloc[0]
        rows.append({
            "induction": int(ind_id),
            "class": r["class"],
            "t0": float(r["t0"]),
            "dt": float(r["dt"]),
            "n_rows": int(len(g)),
        })
    if not samples:
        raise ValueError(f"no inductions found in {INDOOR_DIR}")
    return {
        "X": samples, "time": times, "sr": 1,
        "channel_names": list(INDOOR_CHANNELS),
        "meta": pd.DataFrame(rows), "name": "indoor-air",
    }


def load_drift_batches():
    """Load the UCI Gas Sensor Array Drift dataset as rigs (batches).

    Returns a dict {batch_number: (X, y)} where X is (n, 128) — the
    dataset's own 8 features x 16 sensors — and y is the gas id (1..6).
    This is feature-level data (the original release ships pre-extracted
    features, not raw time series), so the framework window pipeline does
    not apply; the rig/chemoprint experiments consume it directly.
    """
    if not DRIFT_DIR.is_dir():
        raise FileNotFoundError(
            f"UCI drift not found at {DRIFT_DIR} — it lives in this "
            "repo's data/uci-drift/"
        )
    batches = {}
    for path in sorted(DRIFT_DIR.glob("batch*.dat")):
        b = int(path.stem.replace("batch", ""))
        X, y = [], []
        with open(path) as f:
            for line in f:
                parts = line.split()
                if not parts:
                    continue
                label = int(parts[0])
                feats = [float(p.split(":")[1]) for p in parts[1:129]]
                if len(feats) != 128:
                    continue
                X.append(feats)
                y.append(label)
        batches[b] = (np.array(X, dtype=np.float64),
                      np.array(y, dtype=np.int64))
    return batches


def load_smellnet_offline():
    """Load SmellNet offline_training recordings (6 MOX channels, ~1 Hz).

    Each of 50 substances has 5 recordings; every recording is one continuous
    exposure series sampled about once per second. The MQ/CCS811 raw readings
    are passed through as arbitrary units. Recordings have different lengths,
    so X is a list of (n_channels, n_time) arrays (one per recording); the
    consumer windows each recording with its own length.
    """
    if not SMELLNET_OFFLINE.is_dir():
        raise FileNotFoundError(
            f"SmellNet not found at {SMELLNET_OFFLINE} — it lives in this "
            "repo's data/smellnet-offline/."
        )
    samples, rows = [], []
    for sub in sorted(SMELLNET_OFFLINE.iterdir()):
        if not sub.is_dir():
            continue
        for path in sorted(sub.glob("*.csv")):
            df = pd.read_csv(path)
            sensors = df[SMELLNET_CHANNELS].to_numpy(dtype=np.float64)
            samples.append(sensors.T)
            rows.append({
                "substance": sub.name,
                "recording": path.stem,
                "n_rows": int(len(df)),
            })
    if not samples:
        raise ValueError(f"no SmellNet recordings found in {SMELLNET_OFFLINE}")
    meta = pd.DataFrame(rows)
    return {
        "X": samples, "sr": 1,
        "channel_names": list(SMELLNET_CHANNELS),
        "meta": meta, "name": "smellnet-offline",
    }


def registry():
    """Report which datasets are available on disk."""
    entries = {
        "turbulent-mixtures": TURBULENT_RAW.is_dir(),
        "dynamic-mixtures": DYNAMIC_DIR.is_dir(),
        "beef-spoilage": BEEF_CUTS_DIR.is_dir(),
        "uci-drift": DRIFT_DIR.is_dir(),
        "smellnet-offline": SMELLNET_OFFLINE.is_dir(),
        "indoor-air": INDOOR_DIR.is_dir(),
    }
    if entries["turbulent-mixtures"]:
        n_raw = len(list(TURBULENT_RAW.iterdir()))
        n_down = len(list(TURBULENT_DOWN.iterdir()))
        entries["turbulent-mixtures"] = f"present ({n_raw} raw, {n_down} downsampled)"
    if entries["dynamic-mixtures"]:
        files = [p.name for p in DYNAMIC_DIR.glob("*.txt")]
        entries["dynamic-mixtures"] = f"present ({', '.join(files)})"
    if entries["beef-spoilage"]:
        n = len(list(BEEF_CUTS_DIR.glob("*.csv")))
        entries["beef-spoilage"] = f"present ({n} cut sheets extracted)"
    if entries["uci-drift"]:
        n = len(list(DRIFT_DIR.glob("batch*.dat")))
        entries["uci-drift"] = f"present ({n} batches)"
    if entries["smellnet-offline"]:
        n = len(list(SMELLNET_OFFLINE.glob("*/*.csv")))
        entries["smellnet-offline"] = f"present ({n} recordings)"
    if entries["indoor-air"]:
        present = [p for p in INDOOR_DIR.iterdir() if p.name.startswith("HT_Sensor")]
        entries["indoor-air"] = "present (" + ", ".join(p.name for p in sorted(present)) + ")"
    return entries


if __name__ == "__main__":
    for key, status in registry().items():
        print(f"{key}: {status}")
