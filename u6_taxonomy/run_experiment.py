"""U6 — smell taxonomy: do MOX features encode OSMO perceptual families?

SmellNet (sibling repo, HF `DeweiFeng/smell-net`) recorded 50 everyday
substances on a 6-sensor MOX rig; the OSMO taxonomy (bundled,
data/taxonomy/smellnet_osmo_labels.csv) maps each substance to a perceptual grand family
(Woody, Fruity, Green, Herbal, Citrus, ...). U6 asks the product-scoping
question: can framework window features classify the *perceptual category* of
an unseen substance, or only its fine identity?

Tasks (null baselines always in the same table)
------------------------------------------------
1. Fine substance identity (50 classes) — reference ceiling: how well a rig
   resolves individual substances, validated recording-fair
   (StratifiedGroupKFold over recordings so no window from the same recording
   leaks into training). Chance = 2%.
2. Perceptual family classification (8 OSMO grand families) — the honest
   taxonomy claim, validated leave-one-substance-out (train on 49 substances,
   test on a never-seen substance). Chance = 1/8; majority = the largest
   family's share (Woody ~38%).
3. Fine-to-coarse collapse: what coarse accuracy do fine predictions give when
   collapsed onto families — as a comparison against direct coarse training.

Prior work (opensmell/research taxonomy work) found only a *marginal*
intra-family/intra vs inter-family cosine separation (ratio ~0.95), so the
family-level result is expected to be weak or near baseline; reporting it
honestly scopes the product claim.
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold

from harness.loaders import load_smellnet_offline
from harness.features import extract_window_features, window_indices
from harness.evaluate import evaluate_classification
from harness.report import (
    analysis_markdown, classification_table,
    write_analysis, write_metrics,
)

OUT = HERE / "results"

WINDOW_S = 60.0
STRIDE_S = 30.0
N_JOBS = -1

# OSMO grand families + substance->family map (bundled in data/taxonomy).
TAXONOMY_CSV = HERE.parent / "data" / "taxonomy" / "smellnet_osmo_labels.csv"


def load_family_map():
    df = pd.read_csv(TAXONOMY_CSV)
    return dict(zip(df["substance"], df["grand_family"]))


def window_all(bundle):
    """Window every recording with its own length and extract features.

    Returns X (n_windows, n_features), plus per-window recording index and the
    per-recording centre time (unused here but kept for provenance).
    """
    rows, rec_ids, names = [], [], None

    def _extract(rec):
        ts = bundle["X"][rec]
        out = []
        for s, e in window_indices(ts.shape[1], bundle["sr"],
                                   WINDOW_S, STRIDE_S):
            if e - s < 5:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                out.append((extract_window_features(
                    ts[:, s:e], bundle["sr"]), rec, s, e))
        return out

    results = Parallel(n_jobs=N_JOBS, verbose=0)(
        delayed(_extract)(r) for r in range(len(bundle["X"])))
    for batch in results:
        for feats, rec, s, e in batch:
            if names is None:
                names = sorted(feats.keys())
            rows.append([feats.get(k, -1.0) for k in names])
            rec_ids.append(rec)
    return np.asarray(rows, dtype=np.float64), np.asarray(rec_ids), names


def main():
    t0 = time.time()
    bundle = load_smellnet_offline()
    family = load_family_map()
    meta = bundle["meta"]
    n_rec = len(bundle["X"])
    print(f"loaded {n_rec} recordings, {meta['substance'].nunique()} "
          f"substances, {len(bundle['channel_names'])} channels")

    Xw, rec_ids, names = window_all(bundle)
    substance = meta["substance"].to_numpy()[rec_ids]
    y_sub = pd.Series(substance).astype("category").cat.codes.to_numpy()
    y_fam = np.array([family[s] for s in substance])

    cls_sub = evaluate_classification(
        Xw, y_sub, groups=rec_ids, n_estimators=200, n_jobs=N_JOBS)

    logo = LeaveOneGroupOut()
    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=N_JOBS)
    y_fam_true, y_fam_pred = [], []
    per_family = {}
    for tr, te in logo.split(Xw, y_fam, substance):
        clf.fit(Xw[tr], y_fam[tr])
        yp = clf.predict(Xw[te])
        y_fam_true.extend(y_fam[te])
        y_fam_pred.extend(yp)
    y_fam_true = np.asarray(y_fam_true)
    y_fam_pred = np.asarray(y_fam_pred)
    for f in np.unique(y_fam_true):
        m = y_fam_true == f
        per_family[f] = {
            "n": int(m.sum()),
            "accuracy": float(accuracy_score(y_fam_true[m], y_fam_pred[m])),
        }

    # Fine-to-coarse collapse: the fine classifier's leave-one-substance-out
    # predictions, collapsed onto families, scored against the true family.
    # Both OOF arrays are kept in original window order so they align.
    logo_sub = LeaveOneGroupOut()
    y_sub_oof = np.empty(len(Xw), dtype=np.int64)
    for tr, te in logo_sub.split(Xw, y_sub, substance):
        clf.fit(Xw[tr], y_sub[tr])
        y_sub_oof[te] = clf.predict(Xw[te])
    cats = pd.Series(substance).astype("category").cat.categories
    fam_by_code = np.array([family[c] for c in cats])
    collapse_acc = float(accuracy_score(y_fam, fam_by_code[y_sub_oof]))

    classes_f, counts_f = np.unique(y_fam_true, return_counts=True)
    cls_fam = {
        "evaluation": "leave-one-substance-out",
        "n_samples": int(len(y_fam_true)),
        "n_classes": int(len(classes_f)),
        "classes": [str(c) for c in classes_f],
        "accuracy": float(accuracy_score(y_fam_true, y_fam_pred)),
        "chance": float(1.0 / len(classes_f)),
        "majority_baseline": float(counts_f.max() / counts_f.sum()),
        "edge_over_chance_pp": float(
            (accuracy_score(y_fam_true, y_fam_pred) - 1.0 / len(classes_f)) * 100.0),
        "per_class": per_family,
        "fine_to_coarse_collapse_accuracy": collapse_acc,
        "n_groups": int(meta["substance"].nunique()),
    }

    metrics = {
        "dataset": "smellnet-offline",
        "recordings": int(n_rec),
        "substances": int(meta["substance"].nunique()),
        "channels": list(bundle["channel_names"]),
        "windows": int(len(Xw)),
        "window_s": WINDOW_S,
        "stride_s": STRIDE_S,
        "classification": {
            "fine_substance": cls_sub,
            "perceptual_family": cls_fam,
        },
        "runtime_s": round(time.time() - t0, 1),
    }

    sections = [
        ("Dataset",
         f"{n_rec} recordings of {meta['substance'].nunique()} substances, "
         f"{len(bundle['channel_names'])} MOX channels "
         f"({', '.join(bundle['channel_names'])}), {WINDOW_S:g} s windows at "
         f"{STRIDE_S:g} s stride -> {len(Xw)} windows. OSMO families: "
         + ", ".join(f"{k} {v}" for k, v in sorted(
             dict(pd.Series(family).value_counts()).items()))
         + "."),
        ("Fine substance identity (50 classes, reference ceiling)",
         classification_table(cls_sub)),
        ("Perceptual family (8 OSMO grand families, leave-one-substance-out)",
         family_section(cls_fam)),
        ("Honesty notes",
         "The fine task is the rig's reliable capability (recording-fair CV). "
         "The family task is the honest taxonomy claim: an unseen substance's "
         "OSMO perceptual category is much harder than its identity, because "
         "perceptual families span chemically diverse substances and MOX "
         "features capture redox response, not the molecular structure that "
         "drives human categories. The fine-to-coarse collapse shows whether "
         "family structure already exists inside fine predictions. Families "
         "are imbalanced (Woody 19/50, Fruity 10/50, ...), so the majority "
         "baseline is strict. These numbers describe **this rig and these 50 "
         "substances**; cross-device taxonomy transfer is a separate, weaker "
         "result (see interoperability experiments) and is not claimed here."),
    ]
    write_metrics(OUT / "u6_taxonomy_metrics.json", metrics,
                  title="U6 smell taxonomy",
                  dataset="SmellNet offline + OSMO taxonomy",
                  experiment="u6_taxonomy", version="0.1.0")
    write_analysis(OUT / "u6_taxonomy_analysis.md",
                   analysis_markdown("U6 — Smell Taxonomy", sections))
    print(f"wrote results to {OUT} in {time.time() - t0:.1f}s")
    return 0


def family_section(cls):
    acc = cls["accuracy"]
    lines = [
        f"- **Evaluation:** {cls['evaluation']}",
        f"- **Accuracy:** {100.0 * acc:.1f}% (chance "
        f"{100.0 * cls['chance']:.1f}%, majority-class "
        f"{100.0 * cls['majority_baseline']:.1f}%)",
        f"- **Edge over chance:** {cls['edge_over_chance_pp']:+.1f} pp",
        f"- **Fine-to-coarse collapse accuracy:** "
        f"{100.0 * cls['fine_to_coarse_collapse_accuracy']:.1f}%",
        f"- **n:** {cls['n_samples']} (groups: {cls['n_groups']} substances)",
        "",
        "| family | n | accuracy |",
        "|---|---|---|",
    ]
    for f, info in cls["per_class"].items():
        lines.append(f"| {f} | {info['n']} | "
                     f"{100.0 * info['accuracy']:.1f}% |")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
