"""Synthetic-only tests: never read original media or owner-private data."""
import subprocess
from pathlib import Path

import pytest

from h253.original_app_asset_rights import audit, _css_resources


def _git(root: Path, *args: str) -> str:
    p = subprocess.run(["git", "-C", str(root), *args], check=True,
                       capture_output=True, text=True, timeout=20)
    return p.stdout.strip()


@pytest.fixture
def original(tmp_path: Path):
    root = tmp_path / "original"
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.name", "Synthetic Test")
    _git(root, "config", "user.email", "synthetic@example.invalid")
    (root / "LICENSE").write_text("MIT License\nSynthetic fictional owner\n", encoding="utf-8")
    (root / "index.html").write_text(
        '<link rel="stylesheet" href="style.css">'
        '<link rel="preconnect" href="https://fonts.example.invalid">'
        '<link rel="stylesheet" href="https://cdn.example.invalid/icon.css">'
        '<img src="logo.png"><svg></svg>', encoding="utf-8")
    (root / "style.css").write_text(
        '.a {background:url("bg.webp")} @font-face{src:url(font.woff2)}', encoding="utf-8")
    for name in ("logo.png", "bg.webp", "font.woff2"):
        (root / name).write_bytes(b"synthetic-placeholder-not-media")
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "Synthetic source")
    return root, _git(root, "rev-parse", "HEAD")


def test_public_source_asset_inventory_is_not_admission(original):
    root, sha = original
    result = audit(root, sha, ["index.html"])
    assert result["status"] == "INCOMPLETE_NO_ADMISSION"
    assert result["referenced_local_media"] == ["bg.webp", "font.woff2", "logo.png"]
    assert result["remote_runtime_dependencies"] == ["https://cdn.example.invalid/icon.css"]
    assert result["remote_hints_not_proof_of_load"] == ["https://fonts.example.invalid"]
    assert result["inline_svg_elements"] == 1
    assert result["visual_media_rights_proven"] is False
    assert result["official_train_dev_ood_admitted"] is False


def test_dirty_original_fails_closed(original):
    root, sha = original
    (root / "index.html").write_text("<p>modified</p>", encoding="utf-8")
    with pytest.raises(ValueError, match="not clean"):
        audit(root, sha, ["index.html"])


def test_wrong_commit_fails_closed(original):
    root, _ = original
    with pytest.raises(ValueError, match="HEAD differs"):
        audit(root, "0" * 40, ["index.html"])


def test_missing_local_asset_is_reported_not_admitted(original):
    root, sha = original
    (root / "logo.png").unlink()
    _git(root, "add", "-u")
    _git(root, "commit", "-qm", "Synthetic missing asset")
    sha = _git(root, "rev-parse", "HEAD")
    result = audit(root, sha, ["index.html"])
    assert "logo.png" in result["missing_local_resources"]
    assert not result["official_train_dev_ood_admitted"]


def test_unsafe_escape_and_opaque_urls_are_flagged(original):
    root, sha = original
    (root / "index.html").write_text(
        '<img src="../../private.png"><img src="data:image/png;base64,AAAA">',
        encoding="utf-8")
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "Synthetic malicious fixture")
    sha = _git(root, "rev-parse", "HEAD")
    result = audit(root, sha, ["index.html"])
    assert "img:path-escape" in result["unsafe_or_embedded_resources"]
    assert "img:embedded-or-opaque" in result["unsafe_or_embedded_resources"]


def test_css_import_is_detected():
    assert ("css-import", "theme.css") in _css_resources('@import "theme.css";')


def test_missing_entry_rejected(original):
    root, sha = original
    with pytest.raises(ValueError, match="HTML entry missing"):
        audit(root, sha, ["no.html"])


def test_no_media_read_needed(original):
    root, sha = original
    (root / "logo.png").chmod(0)
    try:
        result = audit(root, sha, ["index.html"])
        assert "logo.png" in result["referenced_local_media"]
    finally:
        (root / "logo.png").chmod(0o666)
