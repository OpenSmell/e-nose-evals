"""U4 — indoor-air monitoring: can MOX features detect odor events at home?

UCI-362 "Gas sensors for home activity monitoring" (Figaro TGS array, 8 MOX
channels + Temp./Humidity) recorded 100 inductions of background home
activity, with wine and banana presentations placed near the sensors. For
each induction, time=0 marks the stimulus onset and dt its duration, so the
per-window class is "stimulus" only while 0 <= center_time < dt, and
background otherwise (pre/post stimulus and the pure-background inductions).

U4 asks the product-scoping question for an indoor-air monitor: can framework
window features tell when an odor event is happening in an otherwise normal
indoor environment? Two tasks (null baselines always in the same table):

1. Binary event detection: any stimulus (wine OR banana) vs background — the
   "is there an odor event right now" product question. Chance = 50%.
2. Three-class discrimination: background vs wine vs banana — "which event".
   Chance = 33.3%; majority = the background share.

Both are validated leave-one-induction-out (99 groups) so no window from the
same continuous recording leaks into training and every induction is
completely unseen at test time. Windows are non-overlapping 60 s blocks.

These numbers describe **this Figaro array and these three home conditions**;
the product rig is a different MQ array, so cross-device transfer is a
separate, weaker result (see interoperability experiments) and is not claimed
here.
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
USECASES = HERE.parent
sys.path.insert(0, str(USECASES))

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import LeaveOneGroupOut

from harness.loaders import load_indoor_air
from harness.features import extract_window_features, window_indices
from harness.evaluate import evaluate_classification
from harness.report import (
    analysis_markdown, classification_table,
    write_analysis, write_metrics,
)

OUT = HERE / "results"

WINDOW_S = 60.0
STRIDE_S = 60.0     # non-overlapping: independent statistical units
MAX_WINDOWS = 40    # per-induction even-subsample cap (autocorrelated anyway)
N_JOBS = -1


def _pick_windows(windows, cap):
    """Evenly subsample a list of (s, e) windows down to at most `cap`."""
    if len(windows) <= cap:
        return windows
    idx = np.unique(np.linspace(0, len(windows) - 1, cap).round().astype(int))
    return [windows[i] for i in idx]


def window_all(bundle):
    """Window every induction (its own length), extract features, keep the
    center time so windows can be labeled stimulus vs background.

    Windows are evenly subsampled to at most MAX_WINDOWS per induction:
    adjacent 60 s blocks in a slowly drifting indoor series are strongly
    autocorrelated, so the cap bounds runtime without discarding information,
    and the leave-one-induction-out split stays the honest guard.
    """
    rows, rec_ids, names, centers = [], [], None, []

    def _extract(rec):
        ts = bundle["X"][rec]
        t_axis = bundle["time"][rec]
        out = []
        for s, e in _pick_windows(
                window_indices(ts.shape[1], bundle["sr"],
                               WINDOW_S, STRIDE_S), MAX_WINDOWS):
            if e - s < 5:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                out.append((extract_window_features(
                    ts[:, s:e], bundle["sr"]), rec,
                    float(t_axis[min((s + e) // 2, len(t_axis) - 1)])))
        return out

    results = Parallel(n_jobs=N_JOBS, verbose=0)(
        delayed(_extract)(r) for r in range(len(bundle["X"])))
    for batch in results:
        for feats, rec, center in batch:
            if names is None:
                names = sorted(feats.keys())
            rows.append([feats.get(k, -1.0) for k in names])
            rec_ids.append(rec)
            centers.append(center)
    return (np.asarray(rows, dtype=np.float64), np.asarray(rec_ids),
            names, np.asarray(centers, dtype=np.float64))


def main():
    t0 = time.time()
    bundle = load_indoor_air()
    meta = bundle["meta"]
    print(f"loaded {len(bundle['X'])} inductions, "
          f"{meta['class'].nunique()} conditions, "
          f"{len(bundle['channel_names'])} MOX channels")

    Xw, rec_ids, names, centers = window_all(bundle)
    dt = meta["dt"].to_numpy()[rec_ids]
    cond = meta["class"].to_numpy()[rec_ids]
    in_stim = (centers >= 0.0) & (centers < dt)
    y_stim = (in_stim & (cond != "background")).astype(np.int64)
    y_3 = np.where(in_stim & (cond == "wine"), 1,
                   np.where(in_stim & (cond == "banana"), 2, 0))

    print(f"{len(Xw)} windows x {len(names)} features, "
          f"{meta['class'].value_counts().to_dict()} inductions")

    cls_binary = evaluate_classification(
        Xw, y_stim, groups=rec_ids, cv=LeaveOneGroupOut(),
        n_estimators=200, n_jobs=N_JOBS)
    cls_3 = evaluate_classification(
        Xw, y_3, groups=rec_ids, cv=LeaveOneGroupOut(),
        n_estimators=200, n_jobs=N_JOBS)

    metrics = {
        "dataset": "uci-362-indoor-air",
        "inductions": int(len(bundle["X"])),
        "conditions": list(meta["class"].unique()),
        "channels": list(bundle["channel_names"]),
        "windows": int(len(Xw)),
        "window_s": WINDOW_S,
        "stride_s": STRIDE_S,
        "max_windows_per_induction": MAX_WINDOWS,
        "classification": {
            "binary_stimulus_detection": cls_binary,
            "three_class": cls_3,
        },
        "runtime_s": round(time.time() - t0, 1),
    }

    sections = [
        ("Dataset",
         f"{len(bundle['X'])} inductions over home conditions "
         f"({', '.join(f'{k}: {v}' for k, v in meta['class'].value_counts().items())}), "
         f"{len(bundle['channel_names'])} Figaro MOX channels "
         f"({', '.join(bundle['channel_names'])}), {WINDOW_S:g} s windows at "
         f"{STRIDE_S:g} s stride (evenly capped at {MAX_WINDOWS} per induction) "
         f"-> {len(Xw)} windows. Temp./Humidity excluded "
         "so the claim stays MOX-only. Per-induction class counts: "
         + ", ".join(f"{k} {v}" for k, v in sorted(
             dict(meta["class"].value_counts()).items())) + "."),
        ("Binary event detection (stimulus vs background, leave-one-induction-out)",
         classification_table(cls_binary)),
        ("Three-class discrimination (background / wine / banana, leave-one-induction-out)",
         classification_table(cls_3)),
        ("Honesty notes",
         "Both tasks use leave-one-induction-out, so every test window comes "
         "from a continuous recording whose sensor response was never seen in "
         "training — the strictest honest split for indoor monitoring. The "
         "binary task is the product-relevant claim (an odor event is "
         "present or not); the three-class task asks which event. These "
         "numbers describe **this Figaro array and these three home "
         "conditions**; the product rig is a different MQ array, so "
         "cross-device transfer is a separate, weaker result (see "
         "interoperability experiments) and is not claimed here."),
    ]
    write_metrics(OUT / "u4_indoor_air_metrics.json", metrics,
                  title="U4 indoor-air monitoring",
                  dataset="UCI-362 gas sensors for home activity monitoring",
                  experiment="u4_indoor_air", version="0.1.0")
    write_analysis(OUT / "u4_indoor_air_analysis.md",
                   analysis_markdown("U4 — Indoor-Air Monitoring", sections))
    print(f"wrote results to {OUT} in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
