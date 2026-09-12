from __future__ import annotations

from pathlib import Path

from sqlalchemy import inspect

from aiperfopt.storage import create_all_tables, create_storage_engine, verify_schema
from aiperfopt.storage.schema import metadata


def test_create_all_tables_creates_every_expected_table(tmp_path: Path):
    engine = create_storage_engine(tmp_path / "schema_test.db")
    create_all_tables(engine)

    inspector = inspect(engine)
    existing = set(inspector.get_table_names())
    expected = set(metadata.tables.keys())

    assert expected.issubset(existing)
    assert expected == {
        "raw_system_metrics",
        "raw_process_metrics",
        "raw_process_metrics_other",
        "process_events",
        "raw_system_metrics_rollup",
        "features",
        "process_feature_summaries",
        "forecast_predictions",
        "contention_predictions",
        "diagnoses",
        "recommendations",
        "evaluations",
        "model_deployments",
        "app_logs",
    }
    engine.dispose()


def test_create_all_tables_is_idempotent(tmp_path: Path):
    engine = create_storage_engine(tmp_path / "schema_test.db")
    create_all_tables(engine)
    create_all_tables(engine)  # must not raise
    assert verify_schema(engine) is True
    engine.dispose()


def test_verify_schema_false_on_empty_database(tmp_path: Path):
    engine = create_storage_engine(tmp_path / "empty.db")
    assert verify_schema(engine) is False
    engine.dispose()


def test_timestamp_columns_are_indexed(tmp_path: Path):
    engine = create_storage_engine(tmp_path / "schema_test.db")
    create_all_tables(engine)
    inspector = inspect(engine)

    expected_indexed_columns = {
        "raw_system_metrics": "timestamp",
        "raw_process_metrics": "timestamp",
        "raw_process_metrics_other": "timestamp",
        "process_events": "timestamp",
        "raw_system_metrics_rollup": "bucket_start",
        "features": "timestamp",
        "process_feature_summaries": "timestamp",
        "forecast_predictions": "generated_at",
        "contention_predictions": "generated_at",
        "evaluations": "evaluated_at",
        "model_deployments": "decided_at",
        "app_logs": "timestamp",
    }

    for table_name, column_name in expected_indexed_columns.items():
        indexes = inspector.get_indexes(table_name)
        indexed_columns = {col for idx in indexes for col in idx["column_names"]}
        assert column_name in indexed_columns, f"{table_name}.{column_name} is not indexed"

    engine.dispose()
