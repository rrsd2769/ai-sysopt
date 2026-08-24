from __future__ import annotations

from sqlalchemy import Engine

from aiperfopt.storage import (
    ContentionPredictionRecord,
    DiagnosisRecord,
    EvaluationRecord,
    FeatureVectorRecord,
    ForecastPredictionRecord,
    LogEvent,
    ModelDeploymentRecord,
    ProcessEvent,
    ProcessMetricSample,
    RecommendationRecord,
    SystemMetricSample,
    get_contention_predictions_range,
    get_diagnoses_range,
    get_evaluations_by_version,
    get_evaluations_range,
    get_features_range,
    get_forecast_predictions_range,
    get_latest_feature_vector,
    get_latest_model_deployment,
    get_log_events_range,
    get_model_deployments_range,
    get_process_events_range,
    get_process_metrics_range,
    get_recent_system_metrics,
    get_recommendations_range,
    get_system_metrics_range,
    insert_contention_prediction,
    insert_diagnosis,
    insert_evaluation,
    insert_feature_vector,
    insert_forecast_prediction,
    insert_log_event,
    insert_model_deployment,
    insert_process_event,
    insert_process_metric_sample,
    insert_process_metric_samples_batch,
    insert_recommendation,
    insert_system_metric_sample,
    insert_system_metric_samples_batch,
)


def _sample(ts: float, cpu: float = 10.0) -> SystemMetricSample:
    return SystemMetricSample(timestamp=ts, cpu_percent=cpu, mem_percent=42.0)


def test_system_metric_round_trip(storage_engine: Engine):
    row_id = insert_system_metric_sample(storage_engine, _sample(1000.0, cpu=55.5))
    assert row_id is not None

    rows = get_system_metrics_range(storage_engine, start=999.0, end=1001.0)
    assert len(rows) == 1
    assert rows[0].cpu_percent == 55.5
    assert rows[0].mem_percent == 42.0
    assert rows[0].id == row_id


def test_system_metrics_range_is_chronologically_ordered(storage_engine: Engine):
    # Insert out of order deliberately.
    insert_system_metric_samples_batch(
        storage_engine, [_sample(300.0), _sample(100.0), _sample(200.0)]
    )

    rows = get_system_metrics_range(storage_engine)

    assert [r.timestamp for r in rows] == [100.0, 200.0, 300.0]


def test_get_recent_system_metrics_windowing(storage_engine: Engine):
    insert_system_metric_samples_batch(
        storage_engine, [_sample(t) for t in (0.0, 50.0, 100.0, 150.0, 200.0)]
    )

    recent = get_recent_system_metrics(storage_engine, window_seconds=60, as_of=200.0)

    assert [r.timestamp for r in recent] == [150.0, 200.0]


def test_process_metric_round_trip_and_pid_filter(storage_engine: Engine):
    insert_process_metric_samples_batch(
        storage_engine,
        [
            ProcessMetricSample(timestamp=10.0, pid=111, name="chrome", cpu_percent=20.0),
            ProcessMetricSample(timestamp=10.0, pid=222, name="finder", cpu_percent=1.0),
            ProcessMetricSample(timestamp=20.0, pid=111, name="chrome", cpu_percent=25.0),
        ],
    )

    all_rows = get_process_metrics_range(storage_engine)
    assert len(all_rows) == 3

    pid_rows = get_process_metrics_range(storage_engine, pid=111)
    assert len(pid_rows) == 2
    assert all(r.pid == 111 for r in pid_rows)


def test_process_event_round_trip(storage_engine: Engine):
    insert_process_event(
        storage_engine, ProcessEvent(timestamp=5.0, pid=999, name="stress_test", event_type="launch")
    )

    rows = get_process_events_range(storage_engine)
    assert len(rows) == 1
    assert rows[0].event_type == "launch"
    assert rows[0].pid == 999


def test_feature_vector_round_trip_and_latest(storage_engine: Engine):
    insert_feature_vector(
        storage_engine,
        FeatureVectorRecord(
            timestamp=10.0, computed_at=10.1, feature_set_version="v1",
            feature_values={"cpu_lag_1": 12.3, "cpu_roll_mean_30s": 15.0},
        ),
    )
    insert_feature_vector(
        storage_engine,
        FeatureVectorRecord(
            timestamp=20.0, computed_at=20.1, feature_set_version="v1",
            feature_values={"cpu_lag_1": 15.0},
        ),
    )

    rows = get_features_range(storage_engine)
    assert len(rows) == 2
    assert rows[0].feature_values["cpu_lag_1"] == 12.3

    latest = get_latest_feature_vector(storage_engine)
    assert latest is not None
    assert latest.timestamp == 20.0


def test_forecast_prediction_round_trip_and_metric_filter(storage_engine: Engine):
    insert_forecast_prediction(
        storage_engine,
        ForecastPredictionRecord(
            generated_at=100.0, target_timestamp=140.0, metric_name="cpu_percent",
            horizon_label="short", horizon_seconds=40, predicted_value=72.5,
            confidence=0.81, model_version="lgbm-v1",
        ),
    )
    insert_forecast_prediction(
        storage_engine,
        ForecastPredictionRecord(
            generated_at=100.0, target_timestamp=140.0, metric_name="mem_percent",
            horizon_label="short", horizon_seconds=40, predicted_value=60.0,
            confidence=0.75, model_version="lgbm-v1",
        ),
    )

    cpu_rows = get_forecast_predictions_range(storage_engine, metric_name="cpu_percent")
    assert len(cpu_rows) == 1
    assert cpu_rows[0].predicted_value == 72.5


def test_contention_prediction_round_trip_with_json_fields(storage_engine: Engine):
    insert_contention_prediction(
        storage_engine,
        ContentionPredictionRecord(
            generated_at=100.0, probability=0.88, threshold_used=0.75, triggered=True,
            contributing_metrics=[{"metric": "cpu_percent", "contribution": 0.6}],
            contributing_processes=[{"pid": 42, "name": "chrome", "contribution": 0.5}],
            anomaly_score=0.12, model_version="contention-v1",
        ),
    )

    rows = get_contention_predictions_range(storage_engine)
    assert len(rows) == 1
    assert rows[0].triggered is True
    assert rows[0].contributing_metrics[0]["metric"] == "cpu_percent"
    assert rows[0].contributing_processes[0]["pid"] == 42


def test_diagnosis_round_trip_with_json_fields(storage_engine: Engine):
    insert_diagnosis(
        storage_engine,
        DiagnosisRecord(
            generated_at=100.0,
            category="cpu_bound_single_process",
            is_actionable=True,
            confidence=0.91,
            explanation_tier="shap",
            trigger_reason="high_confidence_contention",
            technical_explanation="Diagnosis: CPU-Bound Single Process...",
            plain_language_explanation="chrome is using a large share of your CPU...",
            model_version="diagnosis-v1",
            forecast_explanations=[
                {"metric_name": "cpu_percent", "forecaster_kind": "lightgbm", "tier": "shap",
                 "top_features": [{"feature_name": "cpu_percent_lag_30s", "value": 12.3}], "note": None}
            ],
            contributing_metrics=[{"metric": "forecast_cpu_percent", "contribution": 0.6}],
            contributing_processes=[{"pid": 42, "name": "chrome", "value": 91.2}],
            anomaly_score=0.05,
        ),
    )

    rows = get_diagnoses_range(storage_engine)
    assert len(rows) == 1
    assert rows[0].category == "cpu_bound_single_process"
    assert rows[0].is_actionable is True
    assert rows[0].forecast_explanations[0]["metric_name"] == "cpu_percent"
    assert rows[0].contributing_processes[0]["name"] == "chrome"


def test_recommendation_round_trip(storage_engine: Engine):
    diag_id = insert_diagnosis(
        storage_engine,
        DiagnosisRecord(
            generated_at=100.0, category="disk_io_bottleneck", is_actionable=True, confidence=0.8,
            explanation_tier="lightweight", trigger_reason="routine", technical_explanation="tech",
            plain_language_explanation="plain", model_version="v1",
        ),
    )
    insert_recommendation(
        storage_engine,
        RecommendationRecord(
            generated_at=100.0,
            diagnosis_id=diag_id,
            diagnosis_category="disk_io_bottleneck",
            actions=[{"title": "t", "description": "d", "rationale": "r"}],
            catalog_version="v1",
        ),
    )

    rows = get_recommendations_range(storage_engine)
    assert len(rows) == 1
    assert rows[0].diagnosis_id == diag_id
    assert rows[0].diagnosis_category == "disk_io_bottleneck"
    assert rows[0].actions[0]["title"] == "t"


def test_evaluation_round_trip(storage_engine: Engine):
    insert_evaluation(
        storage_engine,
        EvaluationRecord(
            evaluated_at=500.0, model_name="lightgbm", model_version="v1",
            metric_name="cpu_percent", horizon_label="short", fold_index=2,
            dataset_tag="walk_forward", mae=1.2, rmse=2.1, directional_accuracy=0.7,
            baseline_name=None,
        ),
    )
    insert_evaluation(
        storage_engine,
        EvaluationRecord(
            evaluated_at=500.0, model_name="naive_persistence", model_version="baseline",
            metric_name="cpu_percent", horizon_label="short", fold_index=2,
            dataset_tag="walk_forward", mae=3.5, rmse=4.4, directional_accuracy=0.5,
            baseline_name="naive_persistence",
        ),
    )

    rows = get_evaluations_range(storage_engine)
    assert len(rows) == 2
    baseline_rows = [r for r in rows if r.baseline_name is not None]
    assert len(baseline_rows) == 1
    assert baseline_rows[0].mae == 3.5


def test_evaluations_by_version_filters_correctly(storage_engine: Engine):
    insert_evaluation(
        storage_engine,
        EvaluationRecord(
            evaluated_at=500.0, model_name="lightgbm", model_version="v1",
            metric_name="cpu_percent", horizon_label="short", fold_index=None,
            dataset_tag="holdout_scenario", mae=1.0, rmse=2.0, directional_accuracy=0.8,
        ),
    )
    insert_evaluation(
        storage_engine,
        EvaluationRecord(
            evaluated_at=600.0, model_name="lightgbm", model_version="v2",
            metric_name="cpu_percent", horizon_label="short", fold_index=None,
            dataset_tag="holdout_scenario", mae=0.5, rmse=1.0, directional_accuracy=0.9,
        ),
    )

    v1_rows = get_evaluations_by_version(storage_engine, "v1")
    assert len(v1_rows) == 1
    assert v1_rows[0].mae == 1.0

    v2_rows = get_evaluations_by_version(storage_engine, "v2")
    assert len(v2_rows) == 1
    assert v2_rows[0].mae == 0.5

    assert get_evaluations_by_version(storage_engine, "v3") == []


def test_model_deployment_round_trip(storage_engine: Engine):
    insert_model_deployment(
        storage_engine,
        ModelDeploymentRecord(
            decided_at=100.0, candidate_version="v1", previous_version=None,
            decision="deployed", recommendation="deploy", confirmed_by="human",
            comparison_summary={"reasons": ["no active version to compare against"]},
        ),
    )
    insert_model_deployment(
        storage_engine,
        ModelDeploymentRecord(
            decided_at=200.0, candidate_version="v2", previous_version="v1",
            decision="rejected", recommendation="reject", confirmed_by="human",
            comparison_summary={"reasons": ["mae regressed beyond tolerance"]},
        ),
    )

    rows = get_model_deployments_range(storage_engine)
    assert len(rows) == 2
    assert rows[0].candidate_version == "v1"
    assert rows[1].previous_version == "v1"
    assert rows[1].decision == "rejected"

    latest = get_latest_model_deployment(storage_engine)
    assert latest is not None
    assert latest.candidate_version == "v2"
    assert latest.comparison_summary["reasons"] == ["mae regressed beyond tolerance"]


def test_log_event_round_trip(storage_engine: Engine):
    insert_log_event(
        storage_engine,
        LogEvent(
            timestamp=10.0, level="ERROR", logger="aiperfopt.retraining",
            message="model deployment failed validation",
            context={"model_version": "v3"},
        ),
    )

    rows = get_log_events_range(storage_engine)
    assert len(rows) == 1
    assert rows[0].level == "ERROR"
    assert rows[0].context["model_version"] == "v3"


def test_range_query_limit_is_respected(storage_engine: Engine):
    insert_system_metric_samples_batch(
        storage_engine, [_sample(float(t)) for t in range(10)]
    )
    rows = get_system_metrics_range(storage_engine, limit=3)
    assert len(rows) == 3
