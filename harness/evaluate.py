"""Standardized, LOO-fair evaluation for the use-case experiments.

Classification and regression are always compared against honest baselines
(chance, majority class, mean predictor) so a headline number cannot be read
without its null reference.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, r2_score, mean_absolute_error,
    mean_squared_error,
)
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold, StratifiedKFold

from .features import feature_matrix

CHANCE_STRATIFIED = "stratified chance"
MAJORITY = "majority class"
MEAN_PREDICTOR = "mean predictor"


def evaluate_classification(X, y, groups=None, cv=None, seed=42, n_jobs=-1,
                            n_estimators=200):
    """Grouped or shuffled CV classification with chance and majority baselines.

    If `groups` is given and has more than one unique group, a
    StratifiedGroupKFold (default 6 folds) is used so no windows from the same
    recording leak into training. Pass an explicit `cv` (e.g.
    LeaveOneGroupOut()) for stricter schemes.
    """
    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    n_classes = len(classes)

    if cv is not None:
        splitter = lambda: cv.split(X, y, groups) if groups is not None else cv.split(X, y)
        eval_label = type(cv).__name__
    elif groups is not None and len(np.unique(groups)) > 1:
        cv = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=seed)
        splitter = lambda: cv.split(X, y, groups)
        eval_label = f"stratified-group-6fold (n_groups={len(np.unique(groups))})"
    else:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        splitter = lambda: skf.split(X, y)
        eval_label = "stratified-5fold"

    model = RandomForestClassifier(n_estimators=n_estimators, random_state=seed, n_jobs=n_jobs)
    y_true, y_pred, oof_group = [], [], []
    for tr, te in splitter():
        model.fit(X[tr], y[tr])
        y_true.extend(y[te])
        y_pred.extend(model.predict(X[te]))
        if groups is not None:
            oof_group.extend(np.asarray(groups)[te])
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    chance = 1.0 / n_classes
    majority = counts.max() / counts.sum()
    acc = accuracy_score(y_true, y_pred)
    bal = balanced_accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")

    per_class = {}
    for c in classes:
        m = y_true == c
        per_class[str(c)] = {
            "n": int(m.sum()),
            "accuracy": float(accuracy_score(y_true[m], y_pred[m])) if m.sum() else None,
        }

    result = {
        "n_samples": int(len(y_true)),
        "n_classes": int(n_classes),
        "classes": [str(c) for c in classes],
        "accuracy": float(acc),
        "balanced_accuracy": float(bal),
        "f1_macro": float(f1_macro),
        "chance": float(chance),
        "majority_baseline": float(majority),
        "edge_over_chance_pp": float((acc - chance) * 100.0),
        "edge_over_majority_pp": float((acc - majority) * 100.0),
        "per_class": per_class,
    }
    if groups is not None:
        result["n_groups"] = int(len(np.unique(groups)))
    result["evaluation"] = eval_label
    return result


def evaluate_regression(X, y, groups=None, cv=None, seed=42, n_jobs=-1,
                        n_estimators=300):
    """Grouped or shuffled CV regression against the mean predictor baseline.

    `y` should be a physical quantity (ppm) in linear space. R² is reported
    against the baseline 'predict the mean', which is the honest null for
    concentration estimation.
    """
    y = np.asarray(y, dtype=np.float64)

    if cv is not None:
        splitter = lambda: cv.split(X, y, groups) if groups is not None else cv.split(X, y)
        eval_label = type(cv).__name__
    elif groups is not None and len(np.unique(groups)) > 1:
        cv = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=seed)
        splitter = lambda: cv.split(X, y, groups)
        eval_label = f"stratified-group-6fold (n_groups={len(np.unique(groups))})"
    else:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(y))
        n = len(idx)
        folds = np.array_split(idx, 5)
        def splitter():
            for i in range(5):
                te = folds[i]
                tr = np.concatenate([f for j, f in enumerate(folds) if j != i])
                yield tr, te
        eval_label = "shuffled-5fold"

    model = RandomForestRegressor(n_estimators=n_estimators, random_state=seed, n_jobs=n_jobs)
    y_true, y_pred = [], []
    for tr, te in splitter():
        model.fit(X[tr], y[tr])
        y_true.extend(y[te])
        y_pred.extend(model.predict(X[te]))
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mean_pred = np.full_like(y_true, np.mean(y_true))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sp = stats.spearmanr(y_true, y_pred).statistic
    return {
        "evaluation": eval_label,
        "n_samples": int(len(y_true)),
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mean_predictor_r2": float(r2_score(y_true, mean_pred)),
        "mean_predictor_mae": float(mean_absolute_error(y_true, mean_pred)),
        "edge_over_mean_predictor_mae_pct": float(
            100.0 * (mean_absolute_error(y_true, mean_pred) - mean_absolute_error(y_true, y_pred))
            / max(mean_absolute_error(y_true, mean_pred), 1e-12)),
        "spearman": float(sp),
    }


def build_window_dataset(samples, sr, meta, label_column, window_s=60.0,
                         stride_s=30.0, max_windows_per_sample=None):
    """Window samples, extract framework features, and align labels.

    Returns (X, y, ids, names, frame). Windows are dropped where the label
    column is missing.
    """
    X, ids, names, start_s, center_s = feature_matrix(samples, sr, window_s, stride_s)
    frame = feature_frame(X, ids, names, meta, start_s, center_s)
    if max_windows_per_sample is not None:
        keep = frame.groupby("sample_id").head(max_windows_per_sample).index
        frame = frame.loc[keep]
    frame = frame.dropna(subset=[label_column])
    y = frame[label_column].to_numpy(dtype=np.float64)
    X = frame[names].to_numpy(dtype=np.float64)
    ids = frame["sample_id"].to_numpy(dtype=np.int64)
    return X, y, ids, names, frame
