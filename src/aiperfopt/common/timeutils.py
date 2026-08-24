"""
Shared time-conversion utility.

Fixes a real pandas 3.0 behavior: ``pd.to_datetime(x, unit='s')`` infers
second-level resolution (``datetime64[s]``) when every input value
happens to be a whole number, but arithmetic against a ``pd.Timedelta``
(e.g. ``index - pd.Timedelta(seconds=30)``) upcasts the result to
microsecond resolution. A subsequent ``pd.merge_asof`` between the two
then raises ``MergeError: incompatible merge keys ... must be the same
type``.

This is invisible with real wall-clock timestamps (``time.time()``
always has a fractional component, which pandas infers as higher
resolution from the start) but reliably reproduces with synthetic or
test data built from whole-second timestamps — exactly what
``synthetic`` generates. Pinning every
timestamp conversion to a fixed resolution (nanoseconds) here means the
resolution can never depend on whether the input floats happen to be
whole numbers.
"""

from __future__ import annotations

import pandas as pd


def epoch_seconds_to_utc_datetime(values) -> pd.DatetimeIndex | pd.Series:
    """Convert epoch-second floats to UTC datetimes at a fixed (ns) resolution."""
    converted = pd.to_datetime(values, unit="s", utc=True)
    if isinstance(converted, pd.Series):
        return converted.dt.as_unit("ns")
    return converted.as_unit("ns")
