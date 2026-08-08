"""U3 — food-spoilage monitoring on the Harvard Dataverse beef dataset.

The dataset (DOI 10.7910/DVN/XNFVTS, CC0) holds 12 beef cuts, each sampled
once per minute for 2220 minutes with 11 MQ sensors while the meat spoils at
room temperature. TVC (total viable count, log10 CFU/g) was measured hourly
and held constant within each hour block; the dataset's freshness Label 1-4 is
a deterministic recoding of TVC against standard thresholds
(<3 fresh, 3-4, 4-5, >=5 spoiled).

Tasks (always with the honest null in the same table)
-----------------------------------------------------
1. TVC regression (log10 CFU/g) from framework window features, against the
   mean-predictor baseline.
2. Four-class freshness classification (Label 1-4), against chance (25%) and
   the majority class (~59.5% spoiled).
3. Binary "spoiled?" decision (Label >= 4, i.e. TVC >= 5), against the same
   majority baseline.

Validation is leave-one-cut-out: the model is trained on windows from the
other 11 cuts and tested on an entirely new piece of meat, so temporally
autocorrelated windows within a cut never leak into training. With only 12
cuts from a single lab/rig, claims are limited to "this rig, this study" —
cross-device generalization is untested (see honesty notes).
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
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import LeaveOneGroupOut

from harness.loaders import load_beef_spoilage
from harness.features import feature_matrix
from harness.evaluate import evaluate_classification, evaluate_regression
from harness.report import (
    analysis_markdown, classification_table, regression_table,
    write_analysis, write_metrics,
)

OUT = HERE / "results"

WINDOW_S = 60.0
STRIDE_S = 30.0
SPOILED_LABEL = 4      # Label >= 4 <==> TVC >= 5
N_JOBS = 2             # keep contention low when other experiments run


def align_center_labels(ids, center_s, meta, column):
    """Per-window label = the per-minute value at the window centre."""
    out = np.full(len(ids), np.nan)
    for i, series in enumerate(meta[column]):
        sel = ids == i
        cmin = np.minimum(np.rint(center_s[sel]).astype(np.int64), len(series) - 1)
        out[sel] = series[cmin]
    return out


def per_channel_signal(bundle):
    """Mean over cuts of the per-cut Spearman of each sensor vs its TVC."""
    rows = []
    for i in range(bundle["X"].shape[0]):
        ts = bundle["X"][i]
        tvc = bundle["meta"]["tvc_series"].iloc[i]
        for c in range(ts.shape[0]):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rho = stats.spearmanr(ts[c], tvc).statistic
            rows.append({"channel": bundle["channel_names"][c], "cut": i,
                         "spearman": rho})
    df = pd.DataFrame(rows)
    return df.groupby("channel")["spearman"].mean().round(3).to_dict()


def leave_one_cut(classifier_factory, X, y, groups):
    """LOOCV per cut. Returns per-cut and pooled metrics."""
    logo = LeaveOneGroupOut()
    per_cut = {}
    y_true, y_pred = [], []
    for tr, te in logo.split(X, y, groups):
        m = classifier_factory()
        m.fit(X[tr], y[tr])
        yp = m.predict(X[te])
        y_true.extend(y[te])
        y_pred.extend(yp)
        g = groups[te][0]
        per_cut[g] = {
            "n": int(len(te)),
            "accuracy": float(accuracy_score(y[te], yp)),
        }
    return {
        "pooled_accuracy": float(accuracy_score(y_true, y_pred)),
        "per_cut": per_cut,
    }


def main():
    t0 = time.time()
    bundle = load_beef_spoilage()
    sr = bundle["sr"]
    meta = bundle["meta"]
    n_cut = bundle["X"].shape[0]
    print(f"loaded {n_cut} cuts, {bundle['X'].shape[1]} channels, "
          f"{bundle['X'].shape[2]} min each, sr={sr}")

    Xw, ids, names, start_s, center_s = feature_matrix(
        bundle["X"], sr, WINDOW_S, STRIDE_S, n_jobs=N_JOBS)
    tvc = align_center_labels(ids, center_s, meta, "tvc_series")
    label = align_center_labels(ids, center_s, meta, "label_series")
    cuts = np.empty(len(ids), dtype=object)
    for i, c in enumerate(meta["cut"]):
        cuts[ids == i] = c
    label = np.rint(label).astype(np.int64)
    print(f"windows: {len(Xw)} ({n_cut} cuts x "
          f"~{int((2220 - WINDOW_S) / STRIDE_S + 1)}), features {len(names)}")

    keep = np.isfinite(tvc)
    Xw, tvc, label, cuts, ids = (Xw[keep], tvc[keep], label[keep],
                                 cuts[keep], ids[keep])

    sig = per_channel_signal(bundle)
    print("per-channel |Spearman| with TVC:", sig)

    reg = evaluate_regression(Xw, tvc, groups=ids, cv=LeaveOneGroupOut(),
                              n_jobs=N_JOBS)
    cls4 = evaluate_classification(Xw, label, groups=ids,
                                   cv=LeaveOneGroupOut(), n_jobs=N_JOBS)
    y_bin = (label >= SPOILED_LABEL).astype(int)
    cls2 = evaluate_classification(Xw, y_bin, groups=ids,
                                   cv=LeaveOneGroupOut(), n_jobs=N_JOBS)

    per_cut_4 = leave_one_cut(
        lambda: RandomForestClassifier(n_estimators=300, random_state=42,
                                       n_jobs=N_JOBS),
        Xw, label, cuts)

    metrics = {
        "dataset": "beef-spoilage",
        "cuts": int(n_cut),
        "channels": int(bundle["X"].shape[1]),
        "windows": int(len(Xw)),
        "window_s": WINDOW_S,
        "stride_s": STRIDE_S,
        "tvc_range_log10": [float(tvc.min()), float(tvc.max())],
        "per_channel_spearman_with_tvc": sig,
        "regression_tvc_log10_cfu_g": reg,
        "classification": {
            "four_class_freshness": cls4,
            "binary_spoiled_tvc_ge_5": cls2,
            "per_cut_four_class": per_cut_4,
        },
        "runtime_s": round(time.time() - t0, 1),
    }

    sections = [
        ("Dataset",
         f"{n_cut} beef cuts x 2220 min at 1 min resolution, {len(bundle['channel_names'])} "
         f"MQ sensors, TVC measured hourly (range "
         f"{tvc.min():.2f}-{tvc.max():.2f} log10 CFU/g). {WINDOW_S:g} min windows "
         f"at {STRIDE_S:g} min stride -> {len(Xw)} windows; labels are the "
         "current-hour TVC / its freshness recoding at the window centre."),
        ("Sensor signal",
         signal_section(sig)),
        ("TVC regression (log10 CFU/g)",
         regression_table(reg)),
        ("Four-class freshness (Label 1-4)",
         classification_table(cls4)),
        ("Binary spoiled? (TVC >= 5)",
         classification_table(cls2)),
        ("Per-cut holdout (leave-one-cut-out, 4-class)",
         per_cut_section(per_cut_4, cuts, label)),
        ("Honesty notes",
         "Label 1-4 is the dataset's own deterministic recoding of hourly TVC, "
         "so classification and regression are two lenses on the same signal. "
         "MQ columns are raw readings in arbitrary units whose response "
         "direction to spoilage differs across sensor models (MQ4/MQ135 fall "
         "while MQ5/MQ137 rise with TVC); the framework features are "
         "scale-invariant ratios, so relative features are robust to the "
         "unknown transducer gain, but any absolute-ppm features are "
         "uncalibrated for this dataset. The intermediate freshness class 3 "
         "(TVC 4-5) is badly separated in this dataset (per-class accuracy "
         "~17%), so the four-class number is driven by the easy fresh/spoiled "
         "endpoints — the binary spoiled? task is the more decision-relevant "
         "read-out. All models were validated leave-one-cut-out (trained on "
         "11 cuts, tested on a whole new cut), so within-cut autocorrelation "
         "and the monotone spoilage trend cannot leak. With 12 cuts from a "
         "single lab/rig and per-hour TVC plateaus, these numbers describe "
         "**this study's rig**; cross-device or cross-lab generalization is "
         "untested and not claimed."),
    ]
    write_metrics(OUT / "u3_food_metrics.json", metrics,
                  title="U3 food-spoilage monitoring",
                  dataset="Harvard Dataverse beef spoilage (12 cuts)",
                  experiment="u3_food", version="0.1.0")
    write_analysis(OUT / "u3_food_analysis.md",
                   analysis_markdown("U3 — Food-Spoilage Monitoring", sections))
    print(f"wrote results to {OUT} in {time.time() - t0:.1f}s")
    return 0


def signal_section(sig):
    lines = [f"- Pooled per-channel Spearman with the current-hour TVC:",
             "  " + ", ".join(f"{k} {v:+.2f}" for k, v in sorted(
                 sig.items(), key=lambda kv: -abs(kv[1])))]
    lines.append("\nDirection differs by sensor model (MQ5/MQ137 rise while "
                 "MQ4/MQ135 fall with TVC), as expected for a mixed MOX "
                 "chamber. The reported tasks test whether the full framework "
                 "window features give reliable cross-cut prediction.")
    return "\n".join(lines)


def per_cut_section(per_cut, cuts, label):
    lines = [
        "Accuracy on each held-out cut (model trained only on the other 11).",
        "",
        "| cut | n | accuracy |",
        "|---|---|---|",
    ]
    for g in sorted(set(cuts), key=str):
        info = per_cut["per_cut"][g]
        lines.append(f"| {g} | {info['n']} | {100.0 * info['accuracy']:.1f}% |")
    classes, counts = np.unique(label, return_counts=True)
    lines.append("")
    lines.append(f"Pooled: {100.0 * per_cut['pooled_accuracy']:.1f}% "
                 f"(chance {100.0 / len(classes):.1f}%, majority class "
                 f"{100.0 * counts.max() / counts.sum():.1f}%)")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
