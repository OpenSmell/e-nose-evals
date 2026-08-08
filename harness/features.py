"""Windowing + framework-feature extraction for use-case time series.

Reuses the product SDK's framework feature extractor
(``opensmell.features.extract_all_framework_features``) on per-channel,
per-window resistance time series, and aligns the resulting per-window feature
dicts into a fixed feature matrix.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from opensmell.features import extract_all_framework_features


def extract_window_features(window, sr):
    """Extract the framework feature dict for one (n_channels, n_time) window."""
    w = np.asarray(window, dtype=np.float64)
    if w.ndim == 1:
        w = w.reshape(-1, 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return extract_all_framework_features(
            w.T, r0_samples=max(3, int(0.1 * w.shape[1])), sr=sr)


def window_indices(n, sr, window_s, stride_s):
    """Return (start, end) index pairs for sliding windows over a series."""
    win = int(round(window_s * sr))
    step = int(round(stride_s * sr))
    if n < win:
        return [(0, n)]
    starts = list(range(0, n - win + 1, step))
    if n - win > 0 and starts[-1] != n - win:
        starts.append(n - win)
    return [(s, s + win) for s in starts]


def feature_matrix(samples, sr, window_s=60.0, stride_s=30.0, sample_ids=None,
                   n_jobs=-1):
    """Window every sample and extract features.

    Returns (X, y_ids, feature_columns, start_s, center_s) where X is
    (n_windows, n_features), y_ids aligns each row with the source sample
    index in `samples`, and start_s / center_s give the window timing in
    seconds relative to each sample's origin.
    """
    if samples.ndim != 3:
        raise ValueError(f"expected (n_samples, n_channels, n_time), got {samples.shape}")

    jobs = []
    for i in range(samples.shape[0]):
        ts = samples[i]
        for (s, e) in window_indices(ts.shape[1], sr, window_s, stride_s):
            if e - s < 5:
                continue
            jobs.append((i, ts[:, s:e], s, e))

    def _extract(job):
        i, window, s, e = job
        feats = extract_window_features(window, sr)
        return i, feats, s, e

    rows, ids, names, starts, centers = [], [], None, [], []
    results = Parallel(n_jobs=n_jobs, verbose=0)(delayed(_extract)(j) for j in jobs)
    for i, feats, s, e in results:
        if names is None:
            names = sorted(feats.keys())
        rows.append([feats.get(k, -1.0) for k in names])
        ids.append(sample_ids[i] if sample_ids is not None else i)
        starts.append(s / sr)
        centers.append((s + e) / 2.0 / sr)
    X = np.asarray(rows, dtype=np.float64)
    return (X, np.asarray(ids, dtype=np.int64), names,
            np.asarray(starts, dtype=np.float64),
            np.asarray(centers, dtype=np.float64))


def feature_frame(X, ids, names, meta, start_s=None, center_s=None):
    """Join window features with sample metadata into one tidy DataFrame."""
    df = pd.DataFrame(X, columns=names)
    df["sample_id"] = ids
    if start_s is not None:
        df["window_start_s"] = start_s
    if center_s is not None:
        df["window_center_s"] = center_s
    meta = meta.copy()
    if "sample_id" not in meta.columns:
        meta = meta.reset_index().rename(columns={"index": "sample_id"})
    return df.merge(meta, on="sample_id", how="left")


def validate_extractor():
    """Smoke-check the feature extractor on a tiny synthetic window."""
    rng = np.random.default_rng(7)
    n_ch, n_time = 8, 200
    t = np.arange(n_time) / 10.0
    sig = np.zeros((n_ch, n_time))
    for c in range(n_ch):
        r0 = 10.0 + c
        sig[c] = r0 * (1 + 0.5 * (1 - np.exp(-t / 15.0)))
        sig[c] += rng.normal(0, 0.05, n_time)
    feats = extract_window_features(sig, sr=10)
    return feats
