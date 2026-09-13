from __future__ import annotations

import pytest
from sqlalchemy import Engine

from aiperfopt.contention.training_pipeline import ContentionTrainingSummary
from aiperfopt.forecasting.training_pipeline import MetricHorizonOutcome, TrainingRunSummary
from aiperfopt.retraining.comparison import compare_candidate_to_active
from aiperfopt.storage import insert_evaluation
from aiperfopt.storage.records import EvaluationRecord


def _outcome(metric_name="cpu_percent", horizon_label="short", mae=1.0, rmse=2.0, directional=0.8):
    return MetricHorizonOutcome(
        metric_name=metric_name, horizon_label=horizon_label, horizon_seconds=45,
        walk_forward_summary={}, selected_forecaster="lightgbm",
        holdout_mae=mae, holdout_rmse=rmse, holdout_directional_accuracy=directional,
    )


@pytest.mark.behavioural
def test_no_active_version_recommends_no_active_baseline(storage_engine: Engine):
    forecast_summary = TrainingRunSummary(model_version="v1", outcomes=[_outcome()])

    comparison = compare_candidate_to_active(
        storage_engine, "v1", forecast_summary, None, active_version=None, regression_tolerance=0.1
    )

    assert comparison.recommendation == "no_active_baseline"
    assert comparison.active_version is None
    assert len(comparison.reasons) == 1


@pytest.mark.behavioural
def test_improved_candidate_recommends_deploy(storage_engine: Engine):
    insert_evaluation(
        storage_engine,
        EvaluationRecord(
            evaluated_at=100.0, model_name="lightgbm", model_version="active-v1",
            metric_name="cpu_percent", horizon_label="short", dataset_tag="holdout_real_usage",
            mae=2.0, rmse=3.0, directional_accuracy=0.7,
        ),
    )
    forecast_summary = TrainingRunSummary(
        model_version="v2", outcomes=[_outcome(mae=1.0, rmse=1.5, directional=0.9)]
    )

    comparison = compare_candidate_to_active(
        storage_engine, "v2", forecast_summary, None, active_version="active-v1", regression_tolerance=0.1
    )

    assert comparison.recommendation == "deploy"
    assert comparison.metric_comparisons[0].regressed is False
    assert comparison.metric_comparisons[0].baseline_mae == 2.0


@pytest.mark.behavioural
def test_regressed_mae_recommends_reject(storage_engine: Engine):
    insert_evaluation(
        storage_engine,
        EvaluationRecord(
            evaluated_at=100.0, model_name="lightgbm", model_version="active-v1",
            metric_name="cpu_percent", horizon_label="short", dataset_tag="holdout_real_usage",
            mae=1.0, rmse=1.5, directional_accuracy=0.9,
        ),
    )
    # MAE more than doubled -- well beyond a 10% tolerance.
    forecast_summary = TrainingRunSummary(
        model_version="v2", outcomes=[_outcome(mae=3.0, rmse=4.0, directional=0.9)]
    )

    comparison = compare_candidate_to_active(
        storage_engine, "v2", forecast_summary, None, active_version="active-v1", regression_tolerance=0.1
    )

    assert comparison.recommendation == "reject"
    assert comparison.metric_comparisons[0].regressed is True
    assert len(comparison.reasons) == 1


@pytest.mark.behavioural
def test_regressed_directional_accuracy_recommends_reject(storage_engine: Engine):
    insert_evaluation(
        storage_engine,
        EvaluationRecord(
            evaluated_at=100.0, model_name="lightgbm", model_version="active-v1",
            metric_name="cpu_percent", horizon_label="short", dataset_tag="holdout_real_usage",
            mae=1.0, rmse=1.5, directional_accuracy=0.9,
        ),
    )
    # MAE holds steady but directional accuracy collapses.
    forecast_summary = TrainingRunSummary(
        model_version="v2", outcomes=[_outcome(mae=1.0, rmse=1.5, directional=0.3)]
    )

    comparison = compare_candidate_to_active(
        storage_engine, "v2", forecast_summary, None, active_version="active-v1", regression_tolerance=0.1
    )

    assert comparison.recommendation == "reject"
    assert comparison.metric_comparisons[0].regressed is True


@pytest.mark.behavioural
def test_no_baseline_for_metric_does_not_block_deployment(storage_engine: Engine):
    # active-v1 never trained "mem_percent" -- nothing to compare against for it.
    insert_evaluation(
        storage_engine,
        EvaluationRecord(
            evaluated_at=100.0, model_name="lightgbm", model_version="active-v1",
            metric_name="cpu_percent", horizon_label="short", dataset_tag="holdout_real_usage",
            mae=1.0, rmse=1.5, directional_accuracy=0.9,
        ),
    )
    forecast_summary = TrainingRunSummary(
        model_version="v2",
        outcomes=[
            _outcome(metric_name="cpu_percent", mae=1.0, rmse=1.5, directional=0.9),
            _outcome(metric_name="mem_percent", mae=5.0, rmse=6.0, directional=0.5),
        ],
    )

    comparison = compare_candidate_to_active(
        storage_engine, "v2", forecast_summary, None, active_version="active-v1", regression_tolerance=0.1
    )

    assert comparison.recommendation == "deploy"
    mem_comparison = [c for c in comparison.metric_comparisons if c.metric_name == "mem_percent"][0]
    assert mem_comparison.baseline_mae is None
    assert mem_comparison.regressed is False


@pytest.mark.behavioural
def test_contention_f1_regression_blocks_deployment(storage_engine: Engine):
    insert_evaluation(
        storage_engine,
        EvaluationRecord(
            evaluated_at=100.0, model_name="contention_classifier", model_version="active-v1",
            horizon_label="short", dataset_tag="holdout_real_usage",
            extra={"precision": 0.8, "recall": 0.8, "f1": 0.8},
        ),
    )
    forecast_summary = TrainingRunSummary(model_version="v2", outcomes=[_outcome()])
    contention_summary = ContentionTrainingSummary(
        model_version="v2", holdout_metrics={"precision": 0.2, "recall": 0.2, "f1": 0.2}
    )

    comparison = compare_candidate_to_active(
        storage_engine, "v2", forecast_summary, contention_summary,
        active_version="active-v1", regression_tolerance=0.1,
    )

    assert comparison.recommendation == "reject"
    assert comparison.contention_comparison.regressed is True
    assert comparison.contention_comparison.baseline_f1 == 0.8


@pytest.mark.plumbing
def test_as_dict_is_json_serializable(storage_engine: Engine):
    import json

    forecast_summary = TrainingRunSummary(model_version="v1", outcomes=[_outcome()])
    comparison = compare_candidate_to_active(
        storage_engine, "v1", forecast_summary, None, active_version=None, regression_tolerance=0.1
    )
    json.dumps(comparison.as_dict())  # must not raise
