"""Common harness for OpenSmell use-case experiments.

Pipeline: dataset loaders -> windowing + framework feature extraction ->
honest-baseline evaluation -> metrics.json + analysis.md.

Reuse across the U2-U6 experiments:
    from harness.loaders import load_turbulent_mixtures
    from harness.evaluate import build_window_dataset, evaluate_classification
    from harness.report import write_metrics, write_analysis
"""

from . import evaluate, features, loaders, report
from .evaluate import (
    build_window_dataset,
    evaluate_classification,
    evaluate_regression,
)
from .features import feature_matrix, validate_extractor
from .loaders import (
    load_dynamic_mixtures,
    load_turbulent_mixtures,
    registry,
)
from .report import (
    analysis_markdown,
    classification_table,
    regression_table,
    write_analysis,
    write_metrics,
)

__all__ = [
    "analysis_markdown",
    "build_window_dataset",
    "classification_table",
    "evaluate",
    "evaluate_classification",
    "evaluate_regression",
    "feature_matrix",
    "features",
    "load_dynamic_mixtures",
    "load_turbulent_mixtures",
    "loaders",
    "regression_table",
    "registry",
    "report",
    "validate_extractor",
    "write_analysis",
    "write_metrics",
]
