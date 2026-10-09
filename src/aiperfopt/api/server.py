"""
Run with: ``python -m aiperfopt.api.server``

The real server entrypoint -- builds the app with ``auto_start=True`` (so
Monitoring and live inference are running from the moment the server is
up, matching the flagship demo's expectation of a continuously-running
monitor) and serves it with uvicorn on the configured host/port.

``start_monitoring=False`` (``aiperfopt serve --no-monitor``) skips only
the bundled Monitoring pipeline, keeping the live-inference loop running
against whatever is already in the configured database -- this is what a
replay demo needs: ``aiperfopt replay`` is the sole writer of raw samples,
and this server must not also poll the real, current machine into the
same (replay) database concurrently.
"""

from __future__ import annotations

import uvicorn

from aiperfopt.api.app import create_app
from aiperfopt.bootstrap import bootstrap


def main(start_monitoring: bool = True) -> None:
    ctx = bootstrap()
    app = create_app(ctx, auto_start=True, start_monitoring=start_monitoring)
    uvicorn.run(app, host=ctx.settings.api.host, port=ctx.settings.api.port)


if __name__ == "__main__":
    main()
