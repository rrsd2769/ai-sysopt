"""
The active-model-version pointer — a small on-disk fact ("which
model_version is currently deployed") that the live inference path
(the API layer) will read on every cycle, and that only a confirmed
deployment decision (:mod:`aiperfopt.lifecycle.deployment`) ever writes.

Deliberately a plain JSON file rather than a database row: it's a single
fact needed on the hot inference path, not a query. The durable,
queryable *history* of every deployment decision still lives in the
``model_deployments`` storage table (the audit trail) — this file
is only ever a cache of "what does that history currently say is
active", written atomically so a crash mid-write can never leave the
live pipeline reading a corrupted or half-written pointer (a retraining-side failure must not be able to corrupt what
the live pipeline reads).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from aiperfopt.applog import get_logger
from aiperfopt.common.exceptions import RetrainingError

logger = get_logger(__name__)

_VERSION_KEY = "active_model_version"
_UPDATED_AT_KEY = "updated_at"


def read_active_version(path: Path) -> str | None:
    """Return the currently active model_version, or None if never set.

    None is a legitimate, expected state (a fresh install before the
    first retraining/deployment has ever run) — not an error.
    """
    if not path.exists():
        return None

    try:
        payload: dict[str, Any] = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RetrainingError(
            f"Active model version file at {path} is unreadable or corrupted",
            recoverable=False,
            context={"path": str(path)},
        ) from exc

    return payload.get(_VERSION_KEY)


def write_active_version(path: Path, model_version: str) -> None:
    """Atomically record ``model_version`` as the active version.

    Writes to a sibling temp file and renames it into place — on POSIX
    filesystems ``Path.replace`` is atomic, so a reader can never observe
    a partially-written file (matching how the Data Ingestion consumer
    treats write-then-persist as one indivisible step).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {_VERSION_KEY: model_version, _UPDATED_AT_KEY: time.time()}

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2))
    tmp_path.replace(path)

    logger.info("Active model version updated", extra={"model_version": model_version, "path": str(path)})
