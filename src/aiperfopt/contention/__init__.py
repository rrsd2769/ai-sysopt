"""Contention Prediction (Section 7.1, item 6): a genuinely trained
classifier (Section 10.4) consuming forecast outputs + current state +
anomaly score (FR-6.1), not raw metrics directly.

- ``config``           -> ContentionLabelingConfig (training-label thresholds)
- ``dataset``            -> build_contention_dataset() — forecast+state+anomaly join (FR-6.1)
- ``model``               -> ContentionClassifier — logistic regression + coefficient attribution
- ``processes``            -> contributing-processes lookup (FR-6.2)
- ``prediction``             -> live inference: anomaly dampening + full record assembly
- ``persistence``             -> save/load classifier artifacts
- ``evaluation``               -> walk-forward precision/recall/F1
- ``training_pipeline``          -> the Phase 5 entrypoint
"""

from aiperfopt.contention.config import ContentionLabelingConfig
from aiperfopt.contention.dataset import build_contention_dataset
from aiperfopt.contention.evaluation import (
    ContentionFoldMetrics,
    mean_contention_metrics,
    run_contention_walk_forward_evaluation,
)
from aiperfopt.contention.model import ContentionClassifier, ContributingMetric
from aiperfopt.contention.persistence import (
    contention_artifact_exists,
    load_contention_classifier,
    save_contention_classifier,
)
from aiperfopt.contention.prediction import predict_contention
from aiperfopt.contention.processes import ContributingProcess, get_contributing_processes
from aiperfopt.contention.training_pipeline import (
    ContentionTrainingSummary,
    train_contention_and_anomaly_pipeline,
)

__all__ = [
    "ContentionLabelingConfig",
    "build_contention_dataset",
    "ContentionFoldMetrics",
    "mean_contention_metrics",
    "run_contention_walk_forward_evaluation",
    "ContentionClassifier",
    "ContributingMetric",
    "contention_artifact_exists",
    "load_contention_classifier",
    "save_contention_classifier",
    "predict_contention",
    "ContributingProcess",
    "get_contributing_processes",
    "ContentionTrainingSummary",
    "train_contention_and_anomaly_pipeline",
]
