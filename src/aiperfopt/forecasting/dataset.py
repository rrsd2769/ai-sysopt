"""
Supervised dataset construction for Forecasting.

Builds a training/evaluation dataset by joining Feature Engineering's
output (:func:`aiperfopt.features.compute_feature_matrix` — one row per
raw sample, computed *causally*: every feature uses only data at or
before its own timestamp) with a forward-looking label: the metric's
actual value ``horizon_seconds`` in the future.

Why the label lookup is a separate function from feature computation
--------------------------------------------------------------------------
:func:`_value_near_target_time` deliberately looks *forward* in time and
is used *only* here, for constructing training targets — never exposed
alongside :mod:`aiperfopt.features.engine`'s feature-computation helpers.
Mixing a forward-looking lookup into the feature-computation module would
create a real risk that a future change accidentally uses it to build a
live-inference feature, which would leak future information into the
model (the model would appear to perform far better in evaluation than it
possibly could at real inference time, since it isn't causal). Keeping
this function physically in a different module, with a docstring warning,
is a deliberate defense against that class of bug.

Counter metrics label the future *rate*, not the future raw counter value
-----------------------------------------------------------------------------
This was a real bug, found and fixed while building Phase 5: for a
counter metric (e.g. ``disk_read_bytes``, a cumulative OS counter that
can reach into the millions), the *feature* columns available to the
model are all small-magnitude, rate-transformed values (FR-4.1's
counter-rate transform — see :mod:`aiperfopt.features.engine`). But the
original version of this function set ``target`` to the raw future
counter value directly. A model with no feature carrying the current
large baseline has no way to predict a similarly large future value from
small-magnitude rate features alone — it was structurally set up to fail,
which is exactly what made every baseline (which *does* have access to
``current_value`` or a rolling-rate mean) look unbeatable for these
metrics. ``is_counter=True`` now rate-transforms the raw series (via
:func:`aiperfopt.features.engine.counter_rate_per_second`) *before*
building both ``current_value`` and ``target`` — so a counter metric's
label becomes "the rate ``horizon_seconds`` from now," a small,
stationary, genuinely learnable quantity, consistent with what its
engineered features already describe.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Sequence

import pandas as pd

from aiperfopt.common.timeutils import epoch_seconds_to_utc_datetime
from aiperfopt.features.engine import counter_rate_per_second
from aiperfopt.storage.records import SystemMetricSample


def _value_near_target_time(
    raw_series: pd.Series, target_times: pd.DatetimeIndex, tolerance_seconds: float
) -> pd.Series:
    """Look up ``raw_series``'s value nearest each of ``target_times``.

    FOR LABEL CONSTRUCTION ONLY. This looks forward in time relative to
    the row being labeled and must never be used to compute a live
    inference feature — see the module docstring.
    """
    if raw_series.empty:
        return pd.Series([float("nan")] * len(target_times))

    source = raw_series.rename("value").reset_index().rename(columns={raw_series.index.name or "index": "sample_time"})
    targets = pd.DataFrame({"target_time": target_times})

    merged = pd.merge_asof(
        targets.sort_values("target_time"),
        source.sort_values("sample_time"),
        left_on="target_time",
        right_on="sample_time",
        direction="nearest",
        tolerance=pd.Timedelta(seconds=tolerance_seconds),
    )
    merged.index = targets.sort_values("target_time").index
    return merged["value"].reindex(range(len(target_times)))


def build_supervised_dataset(
    samples: Sequence[SystemMetricSample],
    feature_matrix: pd.DataFrame,
    metric_name: str,
    horizon_seconds: int,
    tolerance_seconds: float,
    is_counter: bool = False,
    max_gap_seconds: float = 30.0,
) -> pd.DataFrame:
    """Join a feature matrix with forward-looking labels for one metric/horizon.

    Returns a DataFrame with every column from ``feature_matrix`` plus:

    - ``current_value``: the metric's own value at this row's timestamp
      (used by the naive-persistence baseline and as the reference point
      for directional-accuracy scoring). For a counter metric, this is
      the *rate* at this row's timestamp, not the raw cumulative value —
      see the module docstring.
    - ``target``: the metric's actual value ``horizon_seconds`` later
      (or actual rate, for a counter metric).

    Rows with no valid target within ``tolerance_seconds`` (inevitably
    the tail of the series, where "the future" hasn't happened yet in the
    input data) are dropped — training or evaluating against a fabricated
    label would be exactly the kind of silent corruption FR-4.3 is about,
    even though FR-4.3 itself is scoped to feature computation rather
    than label construction.

    Parameters
    ----------
    is_counter:
        Set for cumulative OS counters (disk/net I/O byte counts). See
        the module docstring for why this matters — get it wrong and the
        resulting model is not just suboptimal but structurally unable
        to learn.
    max_gap_seconds:
        Passed through to the counter-rate transform when ``is_counter``
        is set; ignored otherwise.
    """
    if feature_matrix.empty or not samples:
        return pd.DataFrame()

    raw_df = pd.DataFrame.from_records([asdict(s) for s in samples])
    raw_df = raw_df.drop_duplicates(subset="timestamp", keep="last").sort_values("timestamp")
    if metric_name not in raw_df.columns:
        raise KeyError(f"'{metric_name}' is not a raw_system_metrics column")
    raw_df["_dt"] = epoch_seconds_to_utc_datetime(raw_df["timestamp"])
    raw_series = raw_df.set_index("_dt")[metric_name].astype(float)

    if is_counter:
        raw_series = counter_rate_per_second(raw_series, max_gap_seconds)

    result = feature_matrix.copy()
    result["_dt"] = epoch_seconds_to_utc_datetime(result["timestamp"])

    target_times = pd.DatetimeIndex(result["_dt"] + pd.Timedelta(seconds=horizon_seconds))
    result["target"] = _value_near_target_time(raw_series, target_times, tolerance_seconds).to_numpy()

    # current_value: exact match, since result['_dt'] is derived from the
    # same raw samples the feature matrix was built from.
    result["current_value"] = raw_series.reindex(result["_dt"]).to_numpy()

    result = result.dropna(subset=["target", "current_value"])
    return result.drop(columns=["_dt"]).reset_index(drop=True)
