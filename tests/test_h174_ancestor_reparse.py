"""Synthetic ancestor-reparse regressions for H174 donor source preflight."""

import pytest

from h174.source_candidates import _reject_reparse


@pytest.mark.parametrize("kind", ["symlink", "junction"])
def test_ancestor_reparse_must_be_rejected_before_checkout_access(tmp_path, monkeypatch, kind):
    """A safe-looking leaf can still be reached through an unsafe parent."""
    root = tmp_path / "nested" / "source"
    root.mkdir(parents=True)
    ancestor = root.parent
    path_type = type(root)
    method = "is_symlink" if kind == "symlink" else "is_junction"
    original = getattr(path_type, method, lambda path: False)

    def synthetic_reparse(path):
        return path == ancestor or original(path)

    monkeypatch.setattr(path_type, method, synthetic_reparse, raising=False)
    with pytest.raises(ValueError, match="symlink or junction"):
        _reject_reparse(root)


@pytest.mark.parametrize("kind", ["symlink", "junction"])
def test_verifier_rejects_ancestor_before_directory_probe(tmp_path, monkeypatch, kind):
    """No checkout or .git access may occur through a redirected parent."""
    from h174.source_candidates import verify_local_source_only

    root = tmp_path / "redirected" / "source"
    root.mkdir(parents=True)
    path_type = type(root)
    method = "is_symlink" if kind == "symlink" else "is_junction"
    original = getattr(path_type, method, lambda path: False)
    monkeypatch.setattr(
        path_type, method,
        lambda path: path == root.parent or original(path), raising=False,
    )
    original_is_dir = path_type.is_dir

    def forbid_source_probe(path):
        if path == root:
            raise AssertionError("source directory probed before ancestor check")
        return original_is_dir(path)

    monkeypatch.setattr(path_type, "is_dir", forbid_source_probe)
    with pytest.raises(ValueError, match="symlink or junction"):
        verify_local_source_only("pickr", root)
