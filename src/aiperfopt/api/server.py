"""
Run with: ``python -m aiperfopt.api.server``

The real server entrypoint -- builds the app with ``auto_start=True`` (so
Monitoring and live inference are running from the moment the server is
up, matching the flagship demo's expectation of a continuously-running
monitor, Section 6) and serves it with uvicorn on the configured host/port.
"""

from __future__ import annotations

import uvicorn

from aiperfopt.api.app import create_app
from aiperfopt.bootstrap import bootstrap


def main() -> None:
    ctx = bootstrap()
    app = create_app(ctx, auto_start=True)
    uvicorn.run(app, host=ctx.settings.api.host, port=ctx.settings.api.port)


if __name__ == "__main__":
    main()
