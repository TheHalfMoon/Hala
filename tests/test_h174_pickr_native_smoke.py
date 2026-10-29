"""H174 synthetic negative tests: no donor, browser, private media, or model calls."""
from __future__ import annotations

from hashlib import sha256
import pytest
from h174.pickr_native_smoke import allow_loopback_request, verify_derived_assets


def _dist(tmp_path, *, script="./assets/app.js", style="./assets/app.css", base=""):
    root = tmp_path / "dist"
    (root / "assets").mkdir(parents=True)
    (root / "assets" / "app.js").write_text("console.log('synthetic')", encoding="utf-8")
    (root / "assets" / "app.css").write_text("body { color: red }", encoding="utf-8")
    (root / "index.html").write_text(
        f'<html><head>{base}<link rel="stylesheet" href="{style}">'
        '<link rel="stylesheet" href="https://fonts.example/test.css">'
        f'<script type="module" src="{script}"></script></head></html>',
        encoding="utf-8",
    )
    return root


def test_h174_pickr_build_asset_sha(tmp_path):
    root = _dist(tmp_path)
    found = verify_derived_assets(root)
    assert found["js"] == sha256((root / "assets" / "app.js").read_bytes()).hexdigest()
    assert found["css"] == sha256((root / "assets" / "app.css").read_bytes()).hexdigest()


@pytest.mark.parametrize("url", [
    "https://127.0.0.1:8123/index.html",
    "http://localhost:8123/index.html",
    "http://127.0.0.1:8124/index.html",
    "http://evil.example:8123/index.html",
    "file:///C:/private.png",
    "data:text/plain,hi",
    "http://user@127.0.0.1:8123/index.html",
    "http://127.0.0.1:bad/index.html",
])
def test_h174_pickr_rejects_external_requests(url):
    assert not allow_loopback_request(url, 8123)


def test_h174_pickr_allows_exact_ephemeral_loopback():
    assert allow_loopback_request("http://127.0.0.1:8123/assets/app.js", 8123)


@pytest.mark.parametrize("script", [
    "https://evil.example/x.js",
    "./assets/../private.js",
    "./assets/%2e%2e/private.js",
    "./assets/app.js?cache=1",
])
def test_h174_pickr_rejects_unpinned_scripts(tmp_path, script):
    with pytest.raises((AssertionError, FileNotFoundError)):
        verify_derived_assets(_dist(tmp_path, script=script))


def test_h174_pickr_rejects_base_url(tmp_path):
    with pytest.raises(AssertionError, match="base"):
        verify_derived_assets(_dist(tmp_path, base='<base href="https://evil.example/">' ))


def test_h174_pickr_rejects_missing_css(tmp_path):
    root = _dist(tmp_path)
    (root / "assets" / "app.css").unlink()
    with pytest.raises(FileNotFoundError):
        verify_derived_assets(root)


def test_h174_pickr_rejects_missing_js(tmp_path):
    root = _dist(tmp_path)
    (root / "assets" / "app.js").unlink()
    with pytest.raises(FileNotFoundError):
        verify_derived_assets(root)


def test_h174_pickr_rejects_empty_js(tmp_path):
    root = _dist(tmp_path)
    (root / "assets" / "app.js").write_bytes(b"")
    with pytest.raises(AssertionError, match="Empty"):
        verify_derived_assets(root)


def test_h174_pickr_rejects_symlinked_scratch_ancestor(tmp_path):
    actual = tmp_path / "actual"
    _dist(actual)
    alias = tmp_path / "alias"
    try:
        alias.symlink_to(actual, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("OS does not permit symlink creation")
    with pytest.raises(ValueError, match="symlink or junction"):
        verify_derived_assets(alias / "dist")


def test_h174_pickr_rejects_symlinked_derived_asset(tmp_path):
    root = _dist(tmp_path)
    target = root / "assets" / "app.js"
    target.unlink()
    elsewhere = tmp_path / "external.js"
    elsewhere.write_text("console.log('outside')", encoding="utf-8")
    try:
        target.symlink_to(elsewhere)
    except (OSError, NotImplementedError):
        pytest.skip("OS does not permit symlink creation")
    with pytest.raises(ValueError, match="symlink or junction"):
        verify_derived_assets(root)
