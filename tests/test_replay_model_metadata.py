"""Regression test for a real bug surfaced by an actual live replay run:
the live-inference loop reconstructs deployed forecasters from persisted
`evaluations` rows (`resolve_deployed_forecasters`), but `aiperfopt
replay` created a replay database containing only raw system/process
samples -- never the evaluation history a server pointed at that database
needs to rebuild the active model's forecasters. Every live-inference
tick then failed with "model version '<x>' has no usable short-horizon
forecasters", even though the model artifacts on disk were fine.

Standalone file (not folded into `tests/test_replay.py`) to avoid
touching a file already being edited elsewhere in this session.
"""

from __future__ import annotations

import pytest

from aiperfopt.replay.player import copy_model_metadata
from aiperfopt.storage import create_all_tables, create_storage_engine
from aiperfopt.storage.reader import get_evaluations_by_version, get_latest_model_deployment
from aiperfopt.storage.records import EvaluationRecord, ModelDeploymentRecord
from aiperfopt.storage.writer import insert_evaluation, insert_model_deployment

pytestmark = pytest.mark.behavioural


def test_copy_model_metadata_makes_evaluations_reachable_in_the_replay_db(tmp_path):
    primary_engine = create_storage_engine(tmp_path / "primary.db")
    create_all_tables(primary_engine)
    replay_engine = create_storage_engine(tmp_path / "replay.db")
    create_all_tables(replay_engine)

    insert_evaluation(primary_engine, EvaluationRecord(
        evaluated_at=1000.0, model_name="lightgbm", model_version="v1",
        dataset_tag="holdout_final", metric_name="cpu_percent", horizon_label="short",
        mae=1.5, rmse=2.0, directional_accuracy=0.7,
    ))
    insert_model_deployment(primary_engine, ModelDeploymentRecord(
        decided_at=1000.0, candidate_version="v1", decision="deployed",
    ))

    # Replay db starts genuinely empty of this metadata -- reproduces the bug.
    assert get_evaluations_by_version(replay_engine, "v1") == []

    copy_model_metadata(primary_engine, replay_engine)

    copied = get_evaluations_by_version(replay_engine, "v1")
    assert len(copied) == 1
    assert copied[0].metric_name == "cpu_percent"
    assert copied[0].mae == 1.5

    deployment = get_latest_model_deployment(replay_engine)
    assert deployment is not None
    assert deployment.candidate_version == "v1"


def test_copy_model_metadata_is_a_noop_on_an_empty_primary_db(tmp_path):
    primary_engine = create_storage_engine(tmp_path / "primary.db")
    create_all_tables(primary_engine)
    replay_engine = create_storage_engine(tmp_path / "replay.db")
    create_all_tables(replay_engine)

    copy_model_metadata(primary_engine, replay_engine)  # must not raise

    assert get_evaluations_by_version(replay_engine, "anything") == []
    assert get_latest_model_deployment(replay_engine) is None
