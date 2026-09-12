from __future__ import annotations

import pandas as pd

from aiperfopt.features.process_features import (
    PROCESS_SUMMARY_COLUMNS,
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
