"""
Run with: ``python -m aiperfopt.frontend.main``

Starts the Qt event loop, builds an ``ApiClient``/``LiveWebSocketClient``
pair against the locally configured backend (``frontend/config.py``),
and shows the main window. This process assumes the backend
(``python -m aiperfopt.api.server``) is already running or will become
reachable shortly -- ``LiveWebSocketClient`` retries indefinitely rather than requiring a particular startup order.
"""

from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from aiperfopt.frontend.config import load_frontend_settings
from aiperfopt.frontend.main_window import MainWindow


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("AI System Performance Optimizer")

    settings = load_frontend_settings()
    window = MainWindow(settings=settings)
    window.resize(1200, 800)
    window.show()
    window.start()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
