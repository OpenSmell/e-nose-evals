"""U2c — temporal leak-event detection + concentration regression on the UCI
dynamic-mixtures time series (16 MOX sensors, ~12 h continuous each).

The two recordings (Ethylene + CO and Ethylene + Methane) present the hardest
realistic monitoring scenario in the suite: concentrations change to random
setpoints every 80-120 s, sensors never reach steady state, and drift accrues
over 12 h. Ground truth is the *setpoint* ppm columns that ship with the files
(the actual delivered concentration).

Tasks (per file, so the two sessions are never conflated)
---------------------------------------------------------
1. Leak detection: per-window binary "any gas flowing?" + 4-state
   (air / ethylene-only / gas2-only / both), grouped CV over concentration
   episodes (contiguous constant-setpoint stretches) so temporally
   autocorrelated windows never leak across the train/test boundary.
2. Concentration regression: ethylene ppm and gas2 ppm from framework window
   features, against the mean-predictor baseline, GroupKFold over episodes.

All headline numbers carry null baselines (chance, majority class, mean
predictor).
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
USECASES = HERE.parent
sys.path.insert(0, str(USECASES))

import numpy as np
from joblib import Parallel, delayed
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

from harness.loaders import load_dynamic_mixtures
from harness.features import extract_window_features, window_indices
from harness.evaluate import evaluate_classification, evaluate_regression
from harness.report import (
    analysis_markdown, classification_table, regression_table,
    write_analysis, write_metrics,
)

OUT = HERE / "results"

SR_DOWN = 10          # decimated sampling rate used for features
WINDOW_S = 60.0
STRIDE_S = 30.0
MAX_NAN_FRAC = 0.05   # drop a channel-window with more than this fraction NaN


def prepare_series(file_name):
    """Load, decimate, and segment one dynamic recording.

    Returns (X, sr, meta) with X (n_time, n_channels) decimated to SR_DOWN,
    meta a dict of per-sample arrays: time_s, gas2_ppm, ethylene_ppm,
    state (int 0..3), episode (int id of the constant-setpoint stretch).
    """
    bundle = load_dynamic_mixtures(file_name)
    X_full = bundle["X"]  # (n_time, n_channels)
    sr_full = bundle["sr"]
    dec = max(1, round(sr_full / SR_DOWN))
    X = X_full[::dec]
    time_s = np.arange(X.shape[0], dtype=np.float64) / SR_DOWN

    g2 = bundle["meta"]["gas2_ppm"].to_numpy()[::dec]
    et = bundle["meta"]["ethylene_ppm"].to_numpy()[::dec]

    state = np.zeros(X.shape[0], dtype=np.int64)
    state[(et == 0) & (g2 > 0)] = 1
    state[(et > 0) & (g2 == 0)] = 2
    state[(et > 0) & (g2 > 0)] = 3

    keys = np.round(et, 2) * 1000 + np.round(g2, 2)
    keys = np.rint(keys).astype(np.int64)
    episode = np.zeros(X.shape[0], dtype=np.int64)
    ep = 0
    for i in range(1, X.shape[0]):
        if keys[i] != keys[i - 1]:
            ep += 1
        episode[i] = ep

    return X, SR_DOWN, {
        "time_s": time_s, "gas2_ppm": g2, "ethylene_ppm": et,
        "state": state, "episode": episode, "file": file_name,
    }


def clean_window(window):
    """Interpolate isolated NaNs per channel; return None if too sparse."""
    out = np.empty_like(window)
    n_t = window.shape[1]
    for c in range(window.shape[0]):
        col = window[c]
        nan = ~np.isfinite(col)
        if nan.sum() / n_t > MAX_NAN_FRAC:
            return None
        if nan.any():
            x = np.arange(n_t)
            col = np.interp(x, x[~nan], col[~nan])
        out[c] = col
    return out


def extract_windows(X, sr, meta, window_s, stride_s, n_jobs=-1):
    """Sliding windows + framework features, aligned to center-time labels."""
    n = X.shape[0]
    spans = window_indices(n, sr, window_s, stride_s)
    centers = np.array([(s + e) / 2.0 / sr for s, e in spans])

    def _extract(span):
        s, e = span
        w = clean_window(X[s:e].T)
        if w is None:
            return None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            feats = extract_window_features(w, sr)
        return feats

    results = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(_extract)(sp) for sp in spans)

    rows, keep = [], []
    names = None
    for i, feats in enumerate(results):
        if feats is None:
            continue
        if names is None:
            names = sorted(feats.keys())
        rows.append([feats.get(k, -1.0) for k in names])
        keep.append(i)
    keep = np.array(keep)
    Xw = np.asarray(rows, dtype=np.float64)

    def _at(center_s, array):
        idx = np.minimum(np.rint(center_s * sr).astype(np.int64),
                         array.size - 1)
        return array[idx]

    return {
        "X": Xw,
        "names": names,
        "center_s": centers[keep],
        "episode": _at(centers[keep], meta["episode"]),
        "state": _at(centers[keep], meta["state"]),
        "gas2_ppm": _at(centers[keep], meta["gas2_ppm"]),
        "ethylene_ppm": _at(centers[keep], meta["ethylene_ppm"]),
    }


def run_file(file_name, n_jobs=-1):
    t0 = time.time()
    X, sr, meta = prepare_series(file_name)
    n_ch = X.shape[1]
    print(f"[{file_name}] {X.shape[0]} samples, {n_ch} ch, sr={sr}", flush=True)

    ds = extract_windows(X, sr, meta, WINDOW_S, STRIDE_S, n_jobs=n_jobs)
    Xw, names = ds["X"], ds["names"]
    groups = ds["episode"]
    any_gas = (ds["gas2_ppm"] > 0) | (ds["ethylene_ppm"] > 0)

    cls_any = evaluate_classification(
        Xw, any_gas, groups=groups,
        cv=StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=42))
    cls_state = evaluate_classification(
        Xw, ds["state"], groups=groups,
        cv=StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=42))

    cv_reg = GroupKFold(n_splits=6)
    reg_et = evaluate_regression(Xw, ds["ethylene_ppm"], groups=groups, cv=cv_reg)
    reg_g2 = evaluate_regression(Xw, ds["gas2_ppm"], groups=groups, cv=cv_reg)

    metrics = {
        "dataset": f"dynamic-mixtures/{file_name}",
        "channels": int(n_ch),
        "samples": int(X.shape[0]),
        "windows": int(len(Xw)),
        "window_s": WINDOW_S,
        "stride_s": STRIDE_S,
        "n_episodes": int(len(np.unique(groups))),
        "air_fraction": float(np.mean(ds["state"] == 0)),
        "classification": {"any_gas": cls_any, "four_state": cls_state},
        "regression": {"ethylene_ppm": reg_et, "gas2_ppm": reg_g2},
        "runtime_s": round(time.time() - t0, 1),
    }

    sections = [
        ("Protocol",
         f"{file_name}: {X.shape[0]} samples at {sr} Hz (decimated from 100 Hz). "
         f"Windows {WINDOW_S:g} s at {STRIDE_S:g} s stride, {len(Xw)} windows. "
         f"Concentration setpoints change every 80-120 s; {metrics['n_episodes']} "
         f"episodes. Labels and episode groups come from the window centre "
         f"time. Clean air {metrics['air_fraction']*100:.1f}% of the time."),
        ("Leak detection — any gas flowing (binary)",
         classification_table(cls_any)),
        ("Four-state discrimination (air / ethylene / gas2 / both)",
         classification_table(cls_state)),
        ("Ethylene ppm regression",
         regression_table(reg_et)),
        ("Gas2 ppm regression",
         regression_table(reg_g2)),
    ]
    stem = file_name.replace(".txt", "")
    write_metrics(OUT / f"u2_dynamic_{stem}_metrics.json", metrics,
                  title=f"U2c — dynamic mixtures ({file_name})",
                  dataset=f"UCI dynamic gas mixtures ({file_name})",
                  experiment=f"u2_dynamic_{stem}", version="0.1.0")
    write_analysis(OUT / f"u2_dynamic_{stem}_analysis.md",
                   analysis_markdown(f"U2c — Dynamic Gas Mixtures ({file_name})", sections))
    print(f"[{file_name}] wrote results in {time.time() - t0:.1f}s", flush=True)
    return metrics


def main():
    t0 = time.time()
    results = {}
    for fn in ("ethylene_CO.txt", "ethylene_methane.txt"):
        results[fn] = run_file(fn)
    summary = {
        "title": "U2c dynamic mixtures — summary",
        "experiment": "u2_dynamic",
        "generated_utc": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc).isoformat(timespec="seconds"),
        "metrics": {"files": list(results.keys()),
                    "any_gas_acc": {k: v["classification"]["any_gas"]["accuracy"]
                                    for k, v in results.items()},
                    "any_gas_chance": {k: v["classification"]["any_gas"]["chance"]
                                       for k, v in results.items()},
                    "any_gas_majority": {k: v["classification"]["any_gas"]["majority_baseline"]
                                         for k, v in results.items()},
                    "total_runtime_s": round(time.time() - t0, 1)},
    }
    write_metrics(OUT / "u2_dynamic_summary.json", summary, title="U2c dynamic summary",
                  dataset="UCI dynamic gas mixtures", experiment="u2_dynamic",
                  version="0.1.0")
    print(f"all files done in {time.time() - t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
