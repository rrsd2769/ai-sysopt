"""Resolution of the application's data root, and of every path derived from it.

Why this module exists
----------------------
Every runtime path in ``default.yaml`` (``data/aiperfopt.db``, ``models``,
``logs``, ``data/active_model_version.json``) used to be interpreted
relative to the *current working directory*. Launching the server from
one directory and the training script from another silently produced two
separate, empty databases -- a failure with no error message anywhere.
Paths are now resolved exactly once, against a single application home,
and are absolute from that point on.

Platform conventions
--------------------
The default home follows each platform's own convention rather than
hardcoding a POSIX dotfile directory, because this application targets
Windows and macOS equally:

- Windows: ``%LOCALAPPDATA%\\aiperfopt`` (falling back to
  ``%USERPROFILE%\\AppData\\Local\\aiperfopt`` if the variable is unset)
- macOS:   ``~/Library/Application Support/aiperfopt``
- other:   ``$XDG_DATA_HOME/aiperfopt`` or ``~/.local/share/aiperfopt``

``AIPERFOPT_HOME`` overrides all of the above, which is what tests and
side-by-side installs use.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

APP_DIR_NAME = "aiperfopt"
HOME_ENV_VAR = "AIPERFOPT_HOME"


def default_app_home() -> Path:
    """The platform-conventional data directory, ignoring ``AIPERFOPT_HOME``."""
    if sys.platform == "win32":
        local_app_data = os.environ.get("LOCALAPPDATA")
        base = Path(local_app_data) if local_app_data else Path.home() / "AppData" / "Local"
        return base / APP_DIR_NAME
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / APP_DIR_NAME
    xdg = os.environ.get("XDG_DATA_HOME")
    base = Path(xdg) if xdg else Path.home() / ".local" / "share"
    return base / APP_DIR_NAME


def app_home() -> Path:
    """The active application home: ``AIPERFOPT_HOME`` if set, else the platform default.

    Always absolute. Not created here -- creation is the caller's job, so
    that merely reading configuration never has a filesystem side effect.
    """
    override = os.environ.get(HOME_ENV_VAR)
    if override:
        return Path(override).expanduser().resolve()
    return default_app_home()


def resolve_under_home(value: Path | str) -> Path:
    """Make one configured path absolute.

    An absolute path (or one starting with ``~``) is the user stating
    exactly where they want something and is honoured as given. A
    relative path is interpreted against :func:`app_home` -- never
    against the current working directory.
    """
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return app_home() / path
