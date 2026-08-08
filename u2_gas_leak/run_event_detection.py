"""U2 Part B — temporal gas-leak event detection (onset + exposure).

On the turbulent wind-tunnel recordings each measurement follows a fixed
protocol: 60 s clean air -> 180 s gas release -> 60 s recovery. That known
ground-truth timeline lets us train and honestly evaluate a *temporal* leak
detector: per-window binary "gas flowing?" classification, evaluated on
hold-out recordings (StratifiedGroupKFold so no window leaks across the
train/test boundary from the same recording).

Also reports detection latency: on held-out recordings, how long after the
60 s onset does the detector first fire.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
USECASES = HERE.parent
sys.path.insert(0, str(USECASES))

import numpy as np

from harness.loaders import load_turbulent_mixtures
from harness.features import feature_matrix, feature_frame
from harness.evaluate import evaluate_classification
from harness.report import (
    analysis_markdown, classification_table,
    write_analysis, write_metrics,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.ensemble import RandomForestClassifier

OUT = HERE / "results"

WINDOW_S = 20.0
STRIDE_S = 10.0
ONSET_S = 60.0
RELEASE_END_S = 240.0


def main():
    t0 = time.time()
    bundle = load_turbulent_mixtures(use_downsampled=True)
    sr = bundle["sr"]
    n_rec = bundle["X"].shape[0]
    print(f"loaded {n_rec} recordings, sr={sr}")

    X, ids, names, start_s, center_s = feature_matrix(bundle["X"], sr, WINDOW_S, STRIDE_S)
    frame = feature_frame(X, ids, names, bundle["meta"], start_s, center_s)

    centers = frame["window_center_s"].to_numpy()
    starts = frame["window_start_s"].to_numpy()
    y = (centers >= ONSET_S) & (centers <= RELEASE_END_S)
    frame["gas_flowing"] = y.astype(int)

    cls = evaluate_classification(X, y, groups=frame["sample_id"].to_numpy(),
                                  n_estimators=200)

    latency = detection_latency(X, ids, names, start_s, center_s, bundle["meta"])
    onset_accuracy = evaluate_onset_accuracy(X, y, starts, centers,
                                             frame["sample_id"].to_numpy())
    metrics = {
        "dataset": "turbulent-mixtures",
        "recordings": int(n_rec),
        "windows": int(len(X)),
        "window_s": WINDOW_S,
        "stride_s": STRIDE_S,
        "gas_fraction": float(y.mean()),
        "classification": cls,
        "onset_latency": latency,
        "onset_window_accuracy": onset_accuracy,
        "runtime_s": round(time.time() - t0, 1),
    }

    sections = [
        ("Protocol",
         f"Each recording: 60 s clean air, 180 s gas release, 60 s recovery. "
         f"{WINDOW_S:g} s windows at {STRIDE_S:g} s stride. Labels come from the "
         f"known timeline: gas flowing when the window centre lies in "
         f"[{ONSET_S:g}, {RELEASE_END_S:g}] s."),
        ("Gas-flowing window classification",
         classification_table(cls)),
        ("Onset detection on held-out recordings",
         onset_section(onset_accuracy)),
        ("Detection latency",
         latency_section(latency)),
    ]
    write_metrics(OUT / "u2_event_detection_metrics.json", metrics,
                  title="U2B gas-leak temporal event detection",
                  dataset="UCI turbulent gas mixtures",
                  experiment="u2_event_detection", version="0.1.0")
    write_analysis(OUT / "u2_event_detection_analysis.md",
                   analysis_markdown("U2B — Temporal Gas-Leak Event Detection", sections))
    print(f"wrote results to {OUT} in {time.time() - t0:.1f}s")
    return 0


def detection_latency(X, ids, names, start_s, center_s, meta):
    """For each recording, find the first window (from 60 s on) the detector
    would flag, using an RF trained on other recordings only."""
    frame = feature_frame(X, ids, names, meta, start_s, center_s)
    centers = frame["window_center_s"].to_numpy()
    y = ((centers >= ONSET_S) & (centers <= RELEASE_END_S)).astype(int)

    rng = np.random.default_rng(0)
    cv = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=0)
    model = RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1)
    probs = np.full(len(y), np.nan)
    groups = frame["sample_id"].to_numpy()
    for tr, te in cv.split(X, y, groups):
        model.fit(X[tr], y[tr])
        probs[te] = model.predict_proba(X[te])[:, 1]

    frame["p_gas"] = probs
    latencies = []
    for sid, sub in frame.groupby("sample_id"):
        onset_rows = sub[(sub["window_center_s"] >= ONSET_S)]
        if len(onset_rows) == 0:
            continue
        first_fire = onset_rows[onset_rows["p_gas"] >= 0.5]
        if len(first_fire) == 0:
            latencies.append(np.inf)
        else:
            latencies.append(float(first_fire["window_center_s"].iloc[0] - ONSET_S))
    latencies = np.array(latencies)
    return {
        "n_recordings": int(len(latencies)),
        "n_detected": int(np.isfinite(latencies).sum()),
        "detection_rate_pct": float(np.isfinite(latencies).mean() * 100.0),
        "median_latency_s": float(np.median(latencies[np.isfinite(latencies)])),
        "p75_latency_s": float(np.percentile(latencies[np.isfinite(latencies)], 75)),
    }


def evaluate_onset_accuracy(X, y, starts, centers, groups):
    """Accuracy of the per-window gas labels restricted to a window around the
    onset (baseline + early exposure), so the reported number is about the
    *transition*, not the long steady-state exposure."""
    early = (centers >= ONSET_S - WINDOW_S) & (centers <= ONSET_S + WINDOW_S)
    if early.sum() < 20:
        return None
    return evaluate_classification(X[early], y[early], groups=groups[early],
                                   n_estimators=200)


def onset_section(result):
    if result is None:
        return "_not computed_"
    lines = [
        f"Windows within ±{WINDOW_S:g} s of the {ONSET_S:g} s onset.",
        classification_table(result),
    ]
    return "\n\n".join(lines)


def latency_section(lat):
    lines = [
        f"- Recordings: {lat['n_recordings']}",
        f"- Detected on held-out recordings: {lat['detection_rate_pct']:.1f}%",
        f"- Median onset latency: {lat['median_latency_s']:.1f} s",
        f"- 75th percentile latency: {lat['p75_latency_s']:.1f} s",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
