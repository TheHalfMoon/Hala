"""Synthetic only: H174 medium-zoom source-native gate, no pictures/models."""
from pathlib import Path
import ast
import inspect
import pytest

from h174.mediumzoom_native_smoke import (
    verify_mediumzoom_dist, run_native_mediumzoom_smoke,
)


def _dist(tmp_path):
    root = tmp_path / "dist"
    (root / "assets").mkdir(parents=True)
    (root / "index.html").write_text(
        '<!doctype html><title>Medium Zoom | Vanilla JavaScript demo</title>'
        '<img src="assets/one.jpg"><img src="assets/two.jpg">',
        encoding="utf-8",
    )
    for name in ("one.jpg", "two.jpg"):
        (root / "assets" / name).write_bytes(b"synthetic jpg fixture only")
    return root


def test_h174_mediumzoom_generated_scratch_contract(tmp_path):
    assert verify_mediumzoom_dist(_dist(tmp_path)) == tmp_path / "dist"


@pytest.mark.parametrize("violation", [
    "missing-html", "missing-assets", "wrong-title", "missing-photos", "node_modules",
])
def test_h174_mediumzoom_dist_fails_closed(tmp_path, violation):
    root = _dist(tmp_path)
    html = root / "index.html"
    if violation == "missing-html":
        html.unlink()
    elif violation == "missing-assets":
        html.write_text(
            '<title>Medium Zoom | Vanilla JavaScript demo</title>',
            encoding="utf-8",
        )
    elif violation == "wrong-title":
        html.write_text(
            '<title>Different app</title><img src="assets/one.jpg">',
            encoding="utf-8",
        )
    elif violation == "missing-photos":
        (root / "assets" / "two.jpg").unlink()
    elif violation == "node_modules":
        (root / "node_modules").mkdir()
    with pytest.raises((ValueError, FileNotFoundError), match="H174 medium-zoom"):
        verify_mediumzoom_dist(root)


def test_h174_mediumzoom_native_source_uses_actual_click_and_rgba_replay():
    tree = ast.parse(inspect.getsource(run_native_mediumzoom_smoke))
    attrs = [node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)]
    assert "click" in attrs
    assert "screenshot" in attrs
    assert "convert" in attrs
    assert "create_isolated_smoke_context" not in attrs
    assert "route" in attrs
    assert not any(isinstance(node, ast.Assert) for node in ast.walk(tree))


def test_h174_mediumzoom_refuses_symlinked_scratch_ancestor(tmp_path):
    actual = tmp_path / "actual"
    _dist(actual)
    alias = tmp_path / "alias"
    try:
        alias.symlink_to(actual, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("OS cannot create symlink in this context")
    with pytest.raises(ValueError, match="symlink or junction"):
        verify_mediumzoom_dist(alias / "dist")


@pytest.mark.parametrize("system,expected", [
    ("Windows", "windows"), ("Linux", "linux"),
])
def test_h174_mediumzoom_receipt_truthfully_names_real_platform(
    monkeypatch, capsys, system, expected
):
    import sys
    from h174 import mediumzoom_native_smoke as smoke
    observed = []
    monkeypatch.setattr(smoke.platform, "system", lambda: system)

    def fake_native(dist_dir, channel):
        observed.append((dist_dir, channel))
        return "e" * 64, 2

    monkeypatch.setattr(smoke, "run_native_mediumzoom_smoke", fake_native)
    monkeypatch.setattr(sys, "argv", [
        "h174.mediumzoom_native_smoke", "--dist-dir", "synthetic-dist",
        "--browser-channel", "msedge",
    ])
    smoke.main()
    captured = capsys.readouterr().out
    assert observed == [(Path("synthetic-dist"), "msedge")]
    assert "H174_MEDIUMZOOM_ORIGINAL_NATIVE_PASS platform=" + expected in captured
    assert "rgba_sha256=" + "e" * 64 in captured
    assert "H174_MEDIUMZOOM_ORIGINAL_LINUX_NATIVE_PASS" not in captured
