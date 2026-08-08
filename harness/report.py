"""metrics.json + analysis.md emitter for use-case experiments.

Every use-case experiment writes a machine-readable metrics file and a human
readable analysis markdown into its own directory under `e-nose-evals/`.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def write_metrics(path, metrics, title, dataset, experiment, version):
    """Write a metrics.json with provenance."""
    doc = {
        "title": title,
        "experiment": experiment,
        "dataset": dataset,
        "version": version,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "metrics": metrics,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(doc, f, indent=2, default=_json_default)
    return path


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def pct(x):
    return f"{100.0 * x:.1f}%"


def fmt_r2(x):
    return f"{x:.3f}"


def analysis_markdown(title, sections):
    """Render a list of (heading, markdown-body) pairs as markdown."""
    lines = [f"# {title}", ""]
    for heading, body in sections:
        lines.append(f"## {heading}")
        lines.append("")
        lines.append(body.strip())
        lines.append("")
    return "\n".join(lines)


def write_analysis(path, markdown):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown)
    return path


def classification_table(result):
    acc = result["accuracy"]
    chance = result["chance"]
    majority = result["majority_baseline"]
    ev = result.get("evaluation", "stratified-5fold")
    lines = [
        f"- **Evaluation:** {ev}",
        f"- **Accuracy:** {pct(acc)} (chance {pct(chance)}, majority-class {pct(majority)})",
        f"- **Balanced accuracy:** {pct(result['balanced_accuracy'])}",
        f"- **F1 (macro):** {result['f1_macro']:.3f}",
        f"- **Edge over chance:** {result['edge_over_chance_pp']:+.1f} pp",
        f"- **Edge over majority:** {result['edge_over_majority_pp']:+.1f} pp",
        f"- **n:** {result['n_samples']}",
        "",
        "| class | n | accuracy |",
        "|---|---|---|",
    ]
    for cls, info in result["per_class"].items():
        acc_c = info["accuracy"]
        lines.append(f"| {cls} | {info['n']} | {pct(acc_c) if acc_c is not None else '—'} |")
    return "\n".join(lines)


def regression_table(result):
    r2 = result["r2"]
    mae = result["mae"]
    mp_r2 = result["mean_predictor_r2"]
    mp_mae = result["mean_predictor_mae"]
    edge = result["edge_over_mean_predictor_mae_pct"]
    lines = [
        f"- **Evaluation:** {result['evaluation']}",
        f"- **R²:** {r2:.3f} (mean-predictor baseline {mp_r2:.3f})",
        f"- **MAE:** {mae:.3f} (mean-predictor baseline {mp_mae:.3f})",
        f"- **RMSE:** {result['rmse']:.3f}",
        f"- **Spearman ρ:** {result['spearman']:.3f}",
        f"- **MAE improvement over mean predictor:** {edge:+.1f}%",
        f"- **n:** {result['n_samples']}",
    ]
    return "\n".join(lines)
