"""
Frontend (PySide6), Section 7.1 item 13 / FR-13.1-13.3 / ADL-16.

This package is a pure API client. Nothing under ``aiperfopt.frontend``
imports from ``aiperfopt.storage``, ``aiperfopt.forecasting``,
``aiperfopt.contention``, ``aiperfopt.diagnosis``,
``aiperfopt.recommendation``, or ``aiperfopt.lifecycle`` -- FR-13.2
requires the frontend to consume all data exclusively through the API
layer (Phase 9), matching FR-12.4's "the API is the only access path to
backend functionality." Every widget in this package renders data that
already arrived as a REST response or a ``/ws/live`` broadcast; none of
it reaches past ``api_client``/``live_client`` to fetch anything itself.

General Mode vs. Developer Mode (Section 4) is purely a rendering
distinction, not a data-access one -- both modes are built from the same
``LiveStateOut`` payload and the same historical endpoints; Developer
Mode widgets simply surface additional fields (SHAP attribution,
contributing metrics/processes, model versions, confidence internals)
that are already present in that same payload rather than requiring a
separate "detailed" endpoint.
"""
