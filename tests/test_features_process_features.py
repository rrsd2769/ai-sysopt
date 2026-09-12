from __future__ import annotations

import math

import pandas as pd

from aiperfopt.features.process_features import (
    PROCESS_SUMMARY_COLUMNS,
    align_process_summary_to_timestamps,
    compute_process_summary_series,
)


def _metrics_df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _events_df(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["timestamp", "event_type"])
    return pd.DataFrame(rows)


def _aggregates_df(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["timestamp", "other_count", "other_cpu_sum", "other_mem_rss_sum"])
    return pd.DataFrame(rows)


def test_empty_process_metrics_yields_empty_frame_with_expected_columns():
    result = compute_process_summary_series(
        _metrics_df([]), _events_df([]), _aggregates_df([])
    )
    assert result.empty
    assert list(result.columns) == ["timestamp", *PROCESS_SUMMARY_COLUMNS]


def test_one_row_per_distinct_poll_timestamp():
    metrics = _metrics_df(
        [
            {"timestamp": 100.0, "pid": 1, "cpu_percent": 10.0, "memory_rss": 1000, "num_threads": 2, "create_time": 50.0},
            {"timestamp": 103.0, "pid": 1, "cpu_percent": 12.0, "memory_rss": 1000, "num_threads": 2, "create_time": 50.0},
        ]
    )
    result = compute_process_summary_series(metrics, _events_df([]), _aggregates_df([]))
    assert sorted(result["timestamp"].tolist()) == [100.0, 103.0]


def test_causality_a_later_row_never_changes_an_earlier_summary():
    """Adding a future sample (t=200) must not change the summary computed at t=100."""
    base_metrics = _metrics_df(
        [{"timestamp": 100.0, "pid": 1, "cpu_percent": 40.0, "memory_rss": 500, "num_threads": 1, "create_time": 0.0}]
    )
    extended_metrics = pd.concat(
        [
            base_metrics,
            _metrics_df(
                [{"timestamp": 200.0, "pid": 1, "cpu_percent": 99.0, "memory_rss": 999, "num_threads": 9, "create_time": 0.0}]
            ),
        ],
        ignore_index=True,
    )
    events = _events_df([{"timestamp": 150.0, "event_type": "launch"}])

    before = compute_process_summary_series(base_metrics, events, _aggregates_df([]))
    after = compute_process_summary_series(extended_metrics, events, _aggregates_df([]))

    row_before = before[before["timestamp"] == 100.0].iloc[0]
    row_after = after[after["timestamp"] == 100.0].iloc[0]
    for col in PROCESS_SUMMARY_COLUMNS:
        assert row_before[col] == row_after[col], col

    # And the future event (t=150, after t=100) must not have counted
    # towards the launch window ending at t=100.
    assert row_before["proc_launch_count_30s"] == 0


def test_launch_counts_are_backward_looking_from_each_timestamp():
    metrics = _metrics_df(
        [{"timestamp": 200.0, "pid": 1, "cpu_percent": 5.0, "memory_rss": 0, "num_threads": 1, "create_time": 0.0}]
    )
    events = _events_df(
        [
            {"timestamp": 175.0, "event_type": "launch"},  # 25s before -> within 30s window
            {"timestamp": 100.0, "event_type": "launch"},  # 100s before -> within 120s, not 30s
            {"timestamp": 190.0, "event_type": "terminate"},  # 10s before
        ]
    )
    result = compute_process_summary_series(metrics, events, _aggregates_df([])).iloc[0]

    assert result["proc_launch_count_30s"] == 1
    assert result["proc_launch_count_120s"] == 2
    assert result["proc_terminate_count_30s"] == 1


def test_launch_burst_10s_finds_the_densest_10s_subwindow():
    metrics = _metrics_df(
        [{"timestamp": 100.0, "pid": 1, "cpu_percent": 5.0, "memory_rss": 0, "num_threads": 1, "create_time": 0.0}]
    )
    # Three launches within a 2-second span (a burst), one isolated far earlier.
    events = _events_df(
        [
            {"timestamp": 90.0, "event_type": "launch"},
            {"timestamp": 90.5, "event_type": "launch"},
            {"timestamp": 91.0, "event_type": "launch"},
            {"timestamp": 75.0, "event_type": "launch"},
        ]
    )
    result = compute_process_summary_series(metrics, events, _aggregates_df([])).iloc[0]
    assert result["proc_launch_burst_10s"] == 3


def test_young_process_features_reflect_only_recently_created_processes():
    metrics = _metrics_df(
        [
            {"timestamp": 100.0, "pid": 1, "cpu_percent": 20.0, "memory_rss": 0, "num_threads": 1, "create_time": 90.0},  # 10s old -> young
            {"timestamp": 100.0, "pid": 2, "cpu_percent": 30.0, "memory_rss": 0, "num_threads": 1, "create_time": 0.0},   # ancient -> not young
        ]
    )
    result = compute_process_summary_series(metrics, _events_df([]), _aggregates_df([])).iloc[0]
    assert result["proc_young_count"] == 1
    assert result["proc_young_cpu_sum"] == 20.0


def test_young_process_features_are_zero_with_no_recent_launches():
    metrics = _metrics_df(
        [{"timestamp": 100.0, "pid": 1, "cpu_percent": 20.0, "memory_rss": 0, "num_threads": 1, "create_time": 0.0}]
    )
    result = compute_process_summary_series(metrics, _events_df([]), _aggregates_df([])).iloc[0]
    assert result["proc_young_count"] == 0
    assert result["proc_young_cpu_sum"] == 0.0


def test_aggregate_row_corrects_whole_population_sums_but_not_top_k():
    """proc_cpu_sum/proc_mem_rss_sum/proc_count_active must include the
    'everything else' aggregate; proc_cpu_top1 must not (top-K is exact
    from the retained top-N rows alone by construction)."""
    metrics = _metrics_df(
        [
            {"timestamp": 100.0, "pid": 1, "cpu_percent": 50.0, "memory_rss": 1000, "num_threads": 2, "create_time": 0.0},
            {"timestamp": 100.0, "pid": 2, "cpu_percent": 10.0, "memory_rss": 200, "num_threads": 1, "create_time": 0.0},
        ]
    )
    aggregates = _aggregates_df([{"timestamp": 100.0, "other_count": 5, "other_cpu_sum": 15.0, "other_mem_rss_sum": 500}])

    result = compute_process_summary_series(metrics, _events_df([]), aggregates).iloc[0]

    assert result["proc_cpu_sum"] == 50.0 + 10.0 + 15.0
    assert result["proc_mem_rss_sum"] == 1000 + 200 + 500
    assert result["proc_count_active"] == 2 + 5
    assert result["proc_cpu_top1"] == 50.0  # unaffected by the aggregate
    assert result["proc_cpu_top1_share"] == 50.0 / 75.0


def test_align_uses_the_most_recent_poll_at_or_before_each_eval_timestamp():
    metrics = _metrics_df(
        [
            {"timestamp": 100.0, "pid": 1, "cpu_percent": 10.0, "memory_rss": 0, "num_threads": 1, "create_time": 0.0},
            {"timestamp": 110.0, "pid": 1, "cpu_percent": 40.0, "memory_rss": 0, "num_threads": 1, "create_time": 0.0},
        ]
    )
    summary = compute_process_summary_series(metrics, _events_df([]), _aggregates_df([]))

    aligned = align_process_summary_to_timestamps(summary, [101.0, 112.0], tolerance_seconds=5.0)

    row_101 = aligned[aligned["timestamp"] == 101.0].iloc[0]
    row_112 = aligned[aligned["timestamp"] == 112.0].iloc[0]
    assert row_101["proc_cpu_sum"] == 10.0  # nearest poll <= 101 is t=100
    assert row_112["proc_cpu_sum"] == 40.0  # nearest poll <= 112 is t=110


def test_align_never_uses_a_future_poll():
    """A poll strictly after the eval timestamp must never be used (causality)."""
    metrics = _metrics_df(
        [{"timestamp": 200.0, "pid": 1, "cpu_percent": 99.0, "memory_rss": 0, "num_threads": 1, "create_time": 0.0}]
    )
    summary = compute_process_summary_series(metrics, _events_df([]), _aggregates_df([]))

    aligned = align_process_summary_to_timestamps(summary, [199.0], tolerance_seconds=50.0)

    row = aligned.iloc[0]
    assert all(pd.isna(row[col]) for col in PROCESS_SUMMARY_COLUMNS)


def test_align_yields_nan_not_a_stale_value_beyond_tolerance():
    metrics = _metrics_df(
        [{"timestamp": 100.0, "pid": 1, "cpu_percent": 10.0, "memory_rss": 0, "num_threads": 1, "create_time": 0.0}]
    )
    summary = compute_process_summary_series(metrics, _events_df([]), _aggregates_df([]))

    aligned = align_process_summary_to_timestamps(summary, [100.0, 200.0], tolerance_seconds=5.0)

    row_100 = aligned[aligned["timestamp"] == 100.0].iloc[0]
    row_200 = aligned[aligned["timestamp"] == 200.0].iloc[0]
    assert row_100["proc_cpu_sum"] == 10.0
    assert all(pd.isna(row_200[col]) for col in PROCESS_SUMMARY_COLUMNS)


def test_align_with_empty_summary_yields_all_nan_rows():
    empty_summary = pd.DataFrame(columns=["timestamp", *PROCESS_SUMMARY_COLUMNS])
    aligned = align_process_summary_to_timestamps(empty_summary, [1.0, 2.0], tolerance_seconds=5.0)
    assert sorted(aligned["timestamp"].tolist()) == [1.0, 2.0]
    for col in PROCESS_SUMMARY_COLUMNS:
        assert aligned[col].isna().all()


def test_align_is_identical_whether_called_for_training_or_live():
    """The same (summary, eval_timestamps, tolerance) must produce the
    same output regardless of which caller invokes it -- the property
    that makes the training and live paths structurally incapable of
    diverging (both storage_bridge functions route through this one
    call)."""
    metrics = _metrics_df(
        [
            {"timestamp": 100.0, "pid": 1, "cpu_percent": 10.0, "memory_rss": 0, "num_threads": 1, "create_time": 0.0},
            {"timestamp": 103.0, "pid": 1, "cpu_percent": 12.0, "memory_rss": 0, "num_threads": 1, "create_time": 0.0},
        ]
    )
    summary = compute_process_summary_series(metrics, _events_df([]), _aggregates_df([]))
    eval_ts = [100.0, 101.5, 103.0, 106.0]

    result_a = align_process_summary_to_timestamps(summary, eval_ts, tolerance_seconds=3.0)
    result_b = align_process_summary_to_timestamps(summary, list(eval_ts), tolerance_seconds=3.0)

    pd.testing.assert_frame_equal(result_a, result_b)


def test_gap_between_polls_does_not_interpolate_a_missing_timestamp():
    """No row is fabricated for a timestamp process_metrics never sampled --
    this module produces one row per *observed* poll, no reindexing/ffill."""
    metrics = _metrics_df(
        [
            {"timestamp": 100.0, "pid": 1, "cpu_percent": 5.0, "memory_rss": 0, "num_threads": 1, "create_time": 0.0},
            {"timestamp": 500.0, "pid": 1, "cpu_percent": 5.0, "memory_rss": 0, "num_threads": 1, "create_time": 0.0},
        ]
    )
    result = compute_process_summary_series(metrics, _events_df([]), _aggregates_df([]))
    assert sorted(result["timestamp"].tolist()) == [100.0, 500.0]
