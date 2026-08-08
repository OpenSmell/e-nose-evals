"""Harness self-test: end-to-end pipeline on synthetic data.

Verifies loader-independent pieces (feature extraction, windowing, evaluation,
report emission) run green without any dataset on disk. Writes outputs to
``e-nose-evals/selftest/``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from harness.features import feature_frame, feature_matrix, validate_extractor
from harness.evaluate import evaluate_classification, evaluate_regression
from harness.report import (
    analysis_markdown, classification_table, regression_table,
    write_analysis, write_metrics,
)

HERE = Path(__file__).resolve().parent
OUT = HERE / "selftest"


def make_synthetic_samples(rng, n_per_level, n_ch=8, sr=10):
    """Synthetic MOX-like resistance responses at 3 'gas' levels."""
    levels = [0.0, 0.3, 0.9]
    samples, labels = [], []
    for li, frac in enumerate(levels):
        for _ in range(n_per_level):
            base = 10.0 + rng.uniform(-0.5, 0.5, n_ch)
            n = sr * 120
            t = np.arange(n) / sr
            s = np.zeros((n_ch, n))
            for c in range(n_ch):
                r0 = base[c]
                peak = r0 * (1.0 + frac * (0.3 + 0.2 * rng.uniform()))
                resp = r0 + (peak - r0) * (1 - np.exp(-t / 12.0))
                s[c] = resp + rng.normal(0, 0.02 * r0, n)
            samples.append(s)
            labels.append(li)
    return np.stack(samples), np.asarray(labels)


def main():
    rng = np.random.default_rng(3)
    samples, labels = make_synthetic_samples(rng, n_per_level=12)
    sr = 10

    feats = validate_extractor()
    assert len(feats) > 40, "extractor returned unexpectedly few features"

    X, ids, names, _, _ = feature_matrix(samples, sr, window_s=60.0, stride_s=30.0)
    meta = pd.DataFrame({"level": labels})
    frame = feature_frame(X, ids, names, meta)
    y = frame["level"].to_numpy()
    Xf = frame[names].to_numpy()

    cls = evaluate_classification(Xf, y, groups=frame["sample_id"].to_numpy())
    reg = evaluate_regression(Xf, y.astype(np.float64), groups=frame["sample_id"].to_numpy())

    metrics = {
        "extractor_features": len(feats),
        "windows": int(len(Xf)),
        "classification": cls,
        "regression": reg,
    }

    sections = [
        ("Pipeline", "Synthetic smoke test: 36 recordings at 3 levels, "
         "8 channels, framework features per 60 s window."),
        ("Classification", classification_table(cls)),
        ("Regression", regression_table(reg)),
    ]
    write_metrics(OUT / "selftest_metrics.json", metrics,
                  title="Harness self-test", dataset="synthetic",
                  experiment="harness-selftest", version="0.1.0")
    write_analysis(OUT / "selftest_analysis.md",
                   analysis_markdown("Harness Self-Test", sections))
    print("selftest green: extractor OK, %d windows, classification OK, regression OK"
          % len(Xf))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
