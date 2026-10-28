"""Private H173 source attempts are one-shot even when the browser crashes.

Synthetic failure injection only: no owner data, prior OOD, or GPU.
"""
from contextlib import contextmanager
import json
import sys
from types import ModuleType, SimpleNamespace

import pytest

from h173 import capture_ood


@pytest.mark.parametrize("mode", ("capture", "verify"))
def test_failed_original_browser_attempt_is_irrevocably_quarantined(
    tmp_path, monkeypatch, mode
):
    destination = tmp_path / "new-private-h173-ood"
    if mode == "verify":
        destination.mkdir()
        (destination / "manifest.json").write_text(
            json.dumps({"schema": capture_ood.MANIFEST, "count": 32}),
            encoding="utf8",
        )

    monkeypatch.setattr(capture_ood, "verified_original_source", lambda _: {})
    monkeypatch.setattr(
        capture_ood, "previous_image_hashes", lambda *args: set()
    )

    @contextmanager
    def fake_serve(_):
        yield "http://127.0.0.1/"

    from h138 import probe
    monkeypatch.setattr(probe, "serve", fake_serve)

    class Browser:
        def close(self):
            pass

    @contextmanager
    def fake_playwright():
        yield SimpleNamespace(
            chromium=SimpleNamespace(launch=lambda **kwargs: Browser())
        )

    fake_package = ModuleType("playwright")
    fake_package.__path__ = []
    fake_api = ModuleType("playwright.sync_api")
    fake_api.sync_playwright = fake_playwright
    monkeypatch.setitem(sys.modules, "playwright", fake_package)
    monkeypatch.setitem(sys.modules, "playwright.sync_api", fake_api)

    attempted = []

    def failed_scene(*args, **kwargs):
        attempted.append("first-scene")
        raise RuntimeError("injected original-browser failure")

    monkeypatch.setattr(capture_ood, "original_native_scene", failed_scene)

    def attempt():
        return capture_ood.run(
            tmp_path / "owner", tmp_path / "upstream",
            tmp_path / "h168", tmp_path / "h169",
            tmp_path / "h171", tmp_path / "h172",
            destination, mode,
        )

    with pytest.raises(RuntimeError, match="injected original-browser failure"):
        attempt()
    assert destination.is_dir()
    lock_name = (
        "H173_CAPTURE_ATTEMPT_RESERVED.lock"
        if mode == "capture"
        else "H173_INDEPENDENT_REPLAY_ATTEMPT_CONSUMED.lock"
    )
    assert (destination / lock_name).is_file()
    assert not (destination / "owner-source-replay-receipt.json").exists()
    assert attempted == ["first-scene"]

    # The same failed source must NEVER be re-captured or re-replayed.
    with pytest.raises(FileExistsError):
        attempt()
    assert attempted == ["first-scene"]


def test_existing_owner_directory_is_never_modified_by_capture(
    tmp_path, monkeypatch
):
    destination = tmp_path / "quarantined-original-ood"
    destination.mkdir()
    sentinel = destination / "private-evidence.txt"
    sentinel.write_text("do not modify", encoding="utf8")
    monkeypatch.setattr(capture_ood, "verified_original_source", lambda _: {})
    monkeypatch.setattr(
        capture_ood, "previous_image_hashes", lambda *args: set()
    )
    with pytest.raises(FileExistsError):
        capture_ood.run(
            tmp_path / "owner", tmp_path / "upstream",
            tmp_path / "h168", tmp_path / "h169",
            tmp_path / "h171", tmp_path / "h172",
            destination, "capture",
        )
    assert sentinel.read_text("utf8") == "do not modify"
