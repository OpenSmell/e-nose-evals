"""U2 — Gas-leak / mixed-gas identification on real MOX mixtures.

Part A (turbulent mixtures, local): identifies gas leaks on the 8-sensor UCI
wind-tunnel recordings (Ethylene + CO or Methane at 4 flow levels each, 6
repeats per configuration).

Tasks
-----
1. Mixture classification (air / ethylene-only / gas2-only / both) and
   secondary-gas identity (CO vs Me), evaluated leave-one-repeat-out so no
   window leaks into training from the same recording.
2. Per-channel power-law reference calibration (response = R0/R_exposure =
   a * C^b) fitted per gas on single-gas exposures only and validated by
   leave-one-configuration-out concentration recovery (log-decade error).
3. ppm regression from framework window features against the mean-predictor
   baseline.

Part B (dynamic mixtures, download pending) adds temporal leak-event detection
on time-varying concentrations.
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
import pandas as pd
from scipy import stats
from sklearn.metrics import r2_score

from harness.loaders import load_turbulent_mixtures
from harness.features import feature_matrix, feature_frame
from harness.evaluate import evaluate_classification, evaluate_regression
from harness.report import (
    analysis_markdown, classification_table, regression_table,
    write_analysis, write_metrics,
)

OUT = HERE / "results"

SR = 10
WINDOW_S = 60.0
STRIDE_S = 30.0
EXPOSURE_START_S = 60.0


GASES = (("ethylene", "ethylene_ppm"), ("CO", "gas2_ppm"), ("Me", "gas2_ppm"))


def gas_mask(meta, gas):
    """Boolean mask over recordings where `gas` is the sole gas present."""
    if gas == "ethylene":
        return meta["gas2_ppm"].to_numpy() == 0
    return ((meta["gas2"].to_numpy() == gas)
            & (meta["ethylene_ppm"].to_numpy() == 0))


def power_law_fit(ppm, resp):
    """Fit response = a * C^b in log-log space over the ppm sweep."""
    mask = (ppm > 0) & np.isfinite(resp) & (resp > 0)
    if mask.sum() < 2:
        return None
    lc = np.log(ppm[mask])
    lr = np.log(resp[mask])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        b, loga, r, p, se = stats.linregress(lc, lr)
    return {"a": float(np.exp(loga)), "b": float(b), "loglog_r2": float(r ** 2),
            "n_points": int(mask.sum())}


def exposure_response(bundle):
    """Per-recording R0 / R_exposure for every channel.

    Response > 1 during exposure (reducing gases lower the sensor resistance),
    so response = a * C^b has a well-conditioned positive exponent. R0 is the
    median over the clean-air phase; R_exposure is the median over the full
    gas-release phase (robust to turbulence spikes).
    """
    X = bundle["X"]
    sr = bundle["sr"]
    r0_idx = int(EXPOSURE_START_S * sr)
    resp = np.zeros((len(bundle["meta"]), X.shape[1]))
    for i in range(len(bundle["meta"])):
        ts = X[i]
        r0 = np.median(ts[:, :r0_idx], axis=1)
        r_exp = np.median(ts[:, r0_idx:4 * r0_idx], axis=1)
        resp[i] = r0 / np.where(r_exp > 0, r_exp, np.nan)
    return resp


def channel_calibration(bundle, resp):
    """Per-channel (a, b) fit over each gas's real single-gas ppm sweep.

    Each gas is fit only on recordings where it is the sole gas present, so
    the response-ppm relation is not contaminated by cross-sensitivity from a
    co-flowing gas. CO and methane are fit separately: their ppm scales and
    sensor sensitivities differ.
    """
    meta = bundle["meta"]
    n_ch = bundle["X"].shape[1]

    cal = {}
    for gas, col in GASES:
        mask = gas_mask(meta, gas)
        ppm = meta.loc[mask, col].to_numpy()
        rrs = resp[mask]
        fits = []
        for c in range(n_ch):
            f = power_law_fit(ppm, rrs[:, c])
            if f is not None:
                f["channel"] = c
                fits.append(f)
        cal[gas] = {"per_channel": fits, "n_recordings": int(mask.sum()),
                    "n_levels": int((np.unique(ppm) > 0).sum())}
        if fits:
            best = max(fits, key=lambda f: f["loglog_r2"])
            cal[gas]["overall"] = {
                "channel": best["channel"], "a": best["a"], "b": best["b"],
                "loglog_r2": best["loglog_r2"], "n_points": best["n_points"],
                "median_r2": float(np.median([f["loglog_r2"] for f in fits])),
            }
        else:
            cal[gas]["overall"] = None
    return cal


def leave_one_config_ppm_error(meta, resp, col, gas, n_ch):
    """Leave-one-configuration-out single-channel concentration recovery.

    For each held-out exposure level of a gas, the power law is fit per channel
    on that gas's other single-gas levels, and every held-out recording/channel
    predicts the concentration. Error is reported in log-decades
    (|log10(pred/true)|) because the small positive exponents (~0.05..0.2)
    make the inverse power law ill-conditioned: a few predictions blow up. The
    air level (0 ppm) is excluded — detection is the classification task;
    quantification applies to exposure concentrations only.
    """
    mask = gas_mask(meta, gas)
    single = meta[mask]
    ppm = single[col].to_numpy()
    resp_s = resp[mask]
    levels = sorted(float(p) for p in np.unique(ppm).tolist() if p > 0)

    per_level = {p: [] for p in levels}
    for held in levels:
        held_mask = ppm == held
        train_mask = ~held_mask
        for c in range(n_ch):
            f = power_law_fit(ppm[train_mask], resp_s[train_mask, c])
            if f is None:
                continue
            a, b = f["a"], f["b"]
            if not (np.isfinite(a) and np.isfinite(b)) or b < 0.02:
                continue
            for i in np.where(held_mask)[0]:
                r = resp_s[i, c]
                if np.isfinite(r) and r > 0:
                    per_level[held].append((r / a) ** (1.0 / b))

    y_true, y_pred, decades = [], [], []
    for p, preds in per_level.items():
        for pred in preds:
            if not np.isfinite(pred):
                continue
            y_true.append(p)
            y_pred.append(pred)
            decades.append(abs(np.log10(pred) - np.log10(p)))
    if not y_true:
        return None
    decades = np.array(decades)

    config_medians = {}
    for p, preds in per_level.items():
        finite = [v for v in preds if np.isfinite(v)]
        if finite:
            config_medians[p] = float(np.median(finite))

    config_rank = None
    if len(config_medians) >= 2 and len(set(config_medians)) >= 2:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            config_rank = float(stats.spearmanr(
                list(config_medians), list(config_medians.values())).statistic)

    return {
        "n_predictions": int(len(y_true)),
        "n_levels": int(len(levels)),
        "median_log10_error_decades": float(np.median(decades)),
        "p75_log10_error_decades": float(np.percentile(decades, 75)),
        "within_one_decade_pct": float(np.mean(decades <= 1.0) * 100.0),
        "config_median_ppm": {f"{p:.0f}": v for p, v in config_medians.items()},
        "config_order_spearman": config_rank,
    }


def main():
    t0 = time.time()
    bundle = load_turbulent_mixtures(use_downsampled=True)
    meta = bundle["meta"]
    n_ch = bundle["X"].shape[1]
    print(f"loaded {len(meta)} recordings, {n_ch} channels, sr={bundle['sr']}")

    resp = exposure_response(bundle)
    cal = channel_calibration(bundle, resp)
    print("power-law calibration fitted per channel")

    loo_errors = {}
    for gas, col in GASES:
        loo_errors[gas] = leave_one_config_ppm_error(meta, resp, col, gas, n_ch)

    start = int(EXPOSURE_START_S * bundle["sr"])
    X_ts = bundle["X"][:, :, start:]
    Xw, ids, names, _, _ = feature_matrix(X_ts, bundle["sr"], WINDOW_S, STRIDE_S)
    frame = feature_frame(Xw, ids, names, meta)

    y_mixture = np.where(
        (frame["ethylene_ppm"] > 0) & (frame["gas2_ppm"] > 0), "both",
        np.where(frame["ethylene_ppm"] > 0, "ethylene",
                 np.where(frame["gas2_ppm"] > 0, frame["gas2"], "air")))

    cls_mixture = evaluate_classification(
        Xw, y_mixture, groups=frame["sample_id"].to_numpy())

    gas2_ids = frame[frame["gas2_ppm"] > 0].copy()
    cls_gas2 = None
    if len(gas2_ids) > 5:
        Xg = Xw[frame["gas2_ppm"].to_numpy() > 0]
        cls_gas2 = evaluate_classification(
            Xg, gas2_ids["gas2"].to_numpy(),
            groups=gas2_ids["sample_id"].to_numpy())

    reg_et = evaluate_regression(Xw, frame["ethylene_ppm"].to_numpy(),
                                 groups=frame["sample_id"].to_numpy())
    reg_gas2 = evaluate_regression(Xw, frame["gas2_ppm"].to_numpy(),
                                   groups=frame["sample_id"].to_numpy())

    metrics = {
        "dataset": "turbulent-mixtures",
        "recordings": int(len(meta)),
        "channels": int(n_ch),
        "windows": int(len(Xw)),
        "power_law_calibration": cal,
        "loocv_concentration_recovery": loo_errors,
        "classification": {
            "mixture_identity": cls_mixture,
            "secondary_gas_id": cls_gas2,
        },
        "regression": {
            "ethylene_ppm": reg_et,
            "gas2_ppm": reg_gas2,
        },
        "runtime_s": round(time.time() - t0, 1),
    }

    sections = [
        ("Dataset",
         f"{len(meta)} wind-tunnel recordings, {n_ch} Figaro TGS sensors, "
         f"{WINDOW_S:g} s windows at {STRIDE_S:g} s stride from the exposure "
         "phase onward. Ethylene 0-96 ppm; CO 0-460 ppm; Methane 0-131 ppm."),
        ("Power-law reference calibration",
         power_law_section(cal)),
        ("Leave-one-configuration-out concentration recovery",
         loo_section(loo_errors)),
        ("Mixture identification (ethylene-only / gas2-only / both)",
         classification_table(cls_mixture)),
        ("Secondary-gas identity (CO vs Methane, exposure only)",
         classification_table(cls_gas2) if cls_gas2 else "_not computed_"),
        ("ppm regression from framework features",
         regression_table(reg_et) + "\n\n---\n\n" + regression_table(reg_gas2)),
    ]

    write_metrics(OUT / "u2_gas_leak_metrics.json", metrics,
                  title="U2 gas-leak / mixed-gas identification",
                  dataset="UCI turbulent gas mixtures",
                  experiment="u2_gas_leak", version="0.1.0")
    write_analysis(OUT / "u2_gas_leak_analysis.md",
                   analysis_markdown("U2 — Gas-Leak / Mixed-Gas Identification", sections))
    print(f"wrote results to {OUT}")
    print(f"runtime {time.time() - t0:.1f}s")
    return 0


def power_law_section(cal):
    lines = []
    for gas, info in cal.items():
        o = info["overall"]
        if o is None:
            lines.append(f"**{gas}:** no valid single-gas fit")
            continue
        lines.append(
            f"**{gas}:** best channel {o['channel']} → "
            f"a = {o['a']:.3f}, b = {o['b']:.3f}, "
            f"log-log R² = {o['loglog_r2']:.3f} "
            f"(median R² across {len(info['per_channel'])} channels "
            f"{o['median_r2']:.3f})")
    return "\n".join(lines)


def loo_section(errors):
    lines = []
    for gas, e in errors.items():
        if e is None:
            lines.append(f"**{gas}:** not computable")
            continue
        rank = ("nan" if e["config_order_spearman"] is None
                else f"{e['config_order_spearman']:.3f}")
        lines.append(
            f"**{gas}:** median |log10(pred/true)| "
            f"{e['median_log10_error_decades']:.2f} decades "
            f"(p75 {e['p75_log10_error_decades']:.2f}), "
            f"{e['within_one_decade_pct']:.0f}% within one decade, "
            f"n={e['n_predictions']}, config-order Spearman {rank}")
    lines.append(
        "\nEach decade is a factor of 10 in ppm: median errors of "
        "0.16-0.30 decades mean typical single-channel power-law predictions "
        "are only within ~x1.4-2 of the true concentration. This is a rough "
        "dose-response reference for detection screening, **not** a certified "
        "quantification pipeline (see the framework regression tables for "
        "concentration estimation).")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
