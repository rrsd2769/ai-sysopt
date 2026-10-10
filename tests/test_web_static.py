"""The web dashboard is served by the API and every asset it references resolves."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tests.api_conftest import build_test_client

WEB_DIR = Path(__file__).resolve().parent.parent / "src" / "aiperfopt" / "web" / "static"

pytestmark = pytest.mark.plumbing


def _index_asset_urls() -> list[str]:
    html = (WEB_DIR / "index.html").read_text()
    return re.findall(r'<(?:script|link)[^>]+(?:src|href)="([^"]+)"', html)


def _js_import_urls() -> list[str]:
    urls = []
    for js in WEB_DIR.joinpath("js").rglob("*.js"):
        rel_dir = js.parent.relative_to(WEB_DIR).as_posix()
        for spec in re.findall(r'^\s*(?:import|export)\b[^"\']*?from\s+["\']([^"\']+)["\']', js.read_text(), re.M):
            resolved = Path("/static", rel_dir, spec).as_posix()
            # Collapse "a/b/../c" without touching the filesystem.
            parts: list[str] = []
            for part in resolved.split("/"):
                if part == "..":
                    parts.pop()
                elif part not in ("", "."):
                    parts.append(part)
            urls.append("/" + "/".join(parts))
    return urls


def test_index_served_at_root(tmp_path: Path):
    with build_test_client(tmp_path) as client:
        r = client.get("/")
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/html")
        assert '<div id="app">' in r.text


def test_every_referenced_asset_resolves(tmp_path: Path):
    urls = _index_asset_urls() + _js_import_urls()
    assert any(u.endswith("/js/app.js") for u in urls)
    for view in ("live", "accuracy", "reports", "model"):
        assert any(u.endswith(f"/js/views/{view}.js") for u in urls), view
    with build_test_client(tmp_path) as client:
        for url in urls:
            assert client.get(url).status_code == 200, url


def test_static_mount_does_not_shadow_api_routes(tmp_path: Path):
    with build_test_client(tmp_path) as client:
        assert client.get("/health").json() == {"status": "ok"}
        assert client.get("/static/does-not-exist.js").status_code == 404
