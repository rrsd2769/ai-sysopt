"""
Thin synchronous REST client over the Phase 9 API layer (FR-13.2).

Deliberately dumb: every method is a one-line ``self._client.get/post``
plus response-JSON return. No caching, no retry logic, no business rules
-- those all already live on the backend (Phases 1-9); this module's only
job is "make the HTTP call and hand back parsed JSON," matching FR-12.4's
"the API is the only access path" from the client side, not layering a
second copy of any pipeline decision.

Deliberately framework-agnostic (no PySide6 import here) so it's testable
in complete isolation and reusable from a future non-Qt client (e.g. a
CLI) without dragging in a Qt dependency -- ``live_client.py`` is the only
module in this package that needs an event loop, and it wraps this class
rather than duplicating its request logic.
"""

from __future__ import annotations

import httpx

from aiperfopt.frontend.config import FrontendSettings


class ApiError(RuntimeError):
    """Raised when the backend returns a non-2xx response.

    Carries the status code and parsed detail (FastAPI's standard
    ``{"detail": ...}`` error shape) so a caller -- typically a Qt
    status-bar/dialog -- can show something more useful than a raw
    exception string.
    """

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(f"API error {status_code}: {detail}")
        self.status_code = status_code
        self.detail = detail


class ApiClient:
    """A REST client bound to one backend instance.

    Parameters
    ----------
    settings:
        Connection settings (host/port/timeout). Defaults to
        :func:`aiperfopt.frontend.config.load_frontend_settings`.
    transport:
        Optional ``httpx`` transport override, mainly for advanced test
        setups (e.g. request recording). Ordinary tests instead point
        ``settings`` at a real backend running on a real local port
        (``tests/frontend/server_fixture.py``) -- ``httpx.ASGITransport``
        is async-only and can't back this class's synchronous client the
        way ``fastapi.testclient.TestClient`` fakes around internally, so
        a real socket is used rather than an in-process shortcut.
    """

    def __init__(
        self,
        settings: FrontendSettings | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.settings = settings or FrontendSettings()
        self._client = httpx.Client(
            base_url=self.settings.rest_base_url,
            timeout=self.settings.request_timeout_seconds,
            transport=transport,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "ApiClient":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # --- internal ----------------------------------------------------------

    def _get(self, path: str, params: dict | None = None):
        response = self._client.get(path, params=_clean(params))
        return _unwrap(response)

    def _post(self, path: str, json: dict | None = None, timeout: float | None = None):
        response = self._client.post(path, json=json, timeout=timeout)
        return _unwrap(response)

    # --- system (Phase 9 api/routers/system.py) -----------------------------

    def health(self) -> dict:
        return self._get("/health")

    def get_config(self) -> dict:
        return self._get("/config")

    def monitoring_status(self) -> dict:
        return self._get("/monitoring/status")

    def start_monitoring(self) -> dict:
        return self._post("/monitoring/start")

    def stop_monitoring(self) -> dict:
        return self._post("/monitoring/stop")

    def live_status(self) -> dict:
        return self._get("/live/status")

    def get_readiness(self) -> dict:
        return self._get("/readiness")

    def start_live_inference(self) -> dict:
        return self._post("/live/start")

    def stop_live_inference(self) -> dict:
        return self._post("/live/stop")

    # --- live state (REST fallback for non-WebSocket callers) --------------

    def get_live_state(self) -> dict | None:
        return self._get("/live/state")

    # --- history (Phase 9 api/routers/history.py) ---------------------------

    def get_system_metrics(
        self, start: float | None = None, end: float | None = None, limit: int | None = None
    ) -> list[dict]:
        return self._get("/metrics/system", {"start": start, "end": end, "limit": limit})

    def get_features(
        self, start: float | None = None, end: float | None = None, limit: int | None = None
    ) -> list[dict]:
        return self._get("/features", {"start": start, "end": end, "limit": limit})

    def get_forecasts(
        self, start: float | None = None, end: float | None = None, limit: int | None = None
    ) -> list[dict]:
        return self._get("/forecasts", {"start": start, "end": end, "limit": limit})

    def get_contention_predictions(
        self, start: float | None = None, end: float | None = None, limit: int | None = None
    ) -> list[dict]:
        return self._get("/contention", {"start": start, "end": end, "limit": limit})

    def get_diagnoses(
        self, start: float | None = None, end: float | None = None, limit: int | None = None
    ) -> list[dict]:
        return self._get("/diagnoses", {"start": start, "end": end, "limit": limit})

    def get_recommendations(
        self, start: float | None = None, end: float | None = None, limit: int | None = None
    ) -> list[dict]:
        return self._get("/recommendations", {"start": start, "end": end, "limit": limit})

    def get_evaluations(
        self, start: float | None = None, end: float | None = None, limit: int | None = None
    ) -> list[dict]:
        return self._get("/evaluations", {"start": start, "end": end, "limit": limit})

    def get_evaluations_by_version(self, model_version: str) -> list[dict]:
        return self._get(f"/evaluations/by-version/{model_version}")

    def get_deployments(
        self, start: float | None = None, end: float | None = None, limit: int | None = None
    ) -> list[dict]:
        return self._get("/deployments", {"start": start, "end": end, "limit": limit})

    def get_latest_deployment(self) -> dict | None:
        return self._get("/deployments/latest")

    # --- reporting (Phase 10's own small addition) ---------------------------

    def get_predicted_vs_actual(
        self,
        metric_name: str,
        start: float | None = None,
        end: float | None = None,
        horizon_label: str | None = None,
        limit: int | None = None,
    ) -> list[dict]:
        return self._get(
            "/reporting/predicted-vs-actual",
            {
                "metric_name": metric_name, "start": start, "end": end,
                "horizon_label": horizon_label, "limit": limit,
            },
        )

    def get_reportable_metric_names(self) -> list[str]:
        return self._get("/reporting/metrics")

    def get_report_summary(
        self,
        period: str | None = None,
        start: float | None = None,
        end: float | None = None,
    ) -> dict:
        return self._get("/reporting/summary", {"period": period, "start": start, "end": end})

    # --- retraining (Phase 9 api/routers/retraining.py) ----------------------

    def get_active_model_version(self) -> dict:
        return self._get("/retraining/active-version")

    def run_retraining(
        self,
        new_model_version: str,
        data_source: str = "real_usage",
        target_metrics: list[str] | None = None,
        start: float | None = None,
        end: float | None = None,
        scenario_duration_seconds: float = 900.0,
        interval_seconds: float = 2.0,
    ) -> dict:
        # PROJECT_HANDOFF.md flags /retraining/run as synchronous and
        # potentially multi-minute (Section 14's own performance notes) --
        # a much longer per-request timeout than the default is used here
        # deliberately, rather than the caller's UI-responsive default;
        # the caller (RetrainingPanel) already runs this off the UI thread
        # via workers.run_in_background so a long wait here doesn't freeze
        # anything.
        return self._post(
            "/retraining/run",
            {
                "new_model_version": new_model_version, "data_source": data_source,
                "target_metrics": target_metrics, "start": start, "end": end,
                "scenario_duration_seconds": scenario_duration_seconds,
                "interval_seconds": interval_seconds,
            },
            timeout=600.0,
        )

    def confirm_deployment(
        self, candidate_version: str, comparison: dict, gate: dict, confirmed: bool, confirmed_by: str = "human"
    ) -> dict:
        return self._post(
            "/retraining/confirm",
            {
                "candidate_version": candidate_version, "comparison": comparison, "gate": gate,
                "confirmed": confirmed, "confirmed_by": confirmed_by,
            },
        )

    def rollback(self, reason: str) -> dict:
        return self._post("/retraining/rollback", {"reason": reason})


def _clean(params: dict | None) -> dict | None:
    """Drop ``None`` values so httpx doesn't serialize them as the literal string 'None'."""
    if params is None:
        return None
    return {k: v for k, v in params.items() if v is not None}


def _unwrap(response: httpx.Response):
    if response.status_code >= 400:
        detail = response.text
        try:
            detail = response.json().get("detail", detail)
        except Exception:
            pass
        raise ApiError(response.status_code, detail)
    if response.status_code == 204 or not response.content:
        return None
    return response.json()
