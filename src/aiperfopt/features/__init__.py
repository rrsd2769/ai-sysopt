"""Feature Engineering (Section 7.1, item 4).

- ``config``            -> FeatureEngineeringConfig (which metrics, windows, tolerances)
- ``engine``             -> pure computation functions, the FR-4.2 single source of truth
- ``process_features``   -> pure process summary series computation (Phase 2)
- ``storage_bridge``     -> storage-aware convenience wrappers used by live inference / training
"""

from aiperfopt.features.config import FEATURE_SET_VERSION, FeatureEngineeringConfig
from aiperfopt.features.engine import (
    compute_feature_matrix,
    compute_latest_feature_vector,
    counter_rate_per_second,
    feature_matrix_to_records,
)
from aiperfopt.features.process_features import PROCESS_SUMMARY_COLUMNS, compute_process_summary_series
from aiperfopt.features.storage_bridge import (
    backfill_feature_matrix,
    compute_and_store_live_feature_vector,
    compute_live_feature_vector,
    summarize_and_persist_process_features,
)

__all__ = [
    "FEATURE_SET_VERSION",
    "PROCESS_SUMMARY_COLUMNS",
    "FeatureEngineeringConfig",
    "backfill_feature_matrix",
    "compute_and_store_live_feature_vector",
    "compute_feature_matrix",
    "compute_latest_feature_vector",
    "compute_process_summary_series",
    "counter_rate_per_second",
    "compute_live_feature_vector",
    "feature_matrix_to_records",
    "summarize_and_persist_process_features",
]
