"""
Frontend connection settings.

Every other operational parameter in this project (Section 4's mode
selection included -- ``AppSettings.mode.active_mode``) is read from the
backend over the API (``GET /config``), per FR-13.2. Host/port cannot
be, by construction: the frontend needs *some* address to make its first
request to before any API response exists. This is therefore the one
piece of frontend configuration that lives locally, and it gets the same
externally-configurable treatment (NFR-12.1) as everything else via
``pydantic-settings`` env vars -- just without Section 15's YAML-layering
machinery, since there is exactly one small settings object here, not an
entire subsystem's worth of tunables.
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class FrontendSettings(BaseSettings):
    """Where to find the backend API (Phase 9). Override via
    ``AIPERFOPT_FRONTEND__API_HOST`` / ``AIPERFOPT_FRONTEND__API_PORT`` /
    ``AIPERFOPT_FRONTEND__USE_TLS``, matching the rest of the project's
    ``AIPERFOPT_`` env-var convention.
    """

    model_config = SettingsConfigDict(env_prefix="AIPERFOPT_FRONTEND__", extra="forbid")

    api_host: str = "127.0.0.1"
    api_port: int = Field(default=8000, gt=0, le=65535)
    use_tls: bool = False
    # How long to wait for a REST call before giving up (seconds).
    # Deliberately short -- every endpoint this client calls is a local,
    # already-fast SQLite read or a pre-computed live-inference snapshot
    # (Phase 9's own NFR-3.3 target is <= 500ms); a hung request should
    # surface quickly rather than freezing the UI thread indefinitely.
    request_timeout_seconds: float = 5.0

    @property
    def rest_base_url(self) -> str:
        scheme = "https" if self.use_tls else "http"
        return f"{scheme}://{self.api_host}:{self.api_port}"

    @property
    def ws_url(self) -> str:
        scheme = "wss" if self.use_tls else "ws"
        return f"{scheme}://{self.api_host}:{self.api_port}/ws/live"


def load_frontend_settings() -> FrontendSettings:
    return FrontendSettings()
