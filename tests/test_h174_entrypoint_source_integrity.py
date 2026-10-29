"""Synthetic-only H174 HTML resolution and pinned asset provenance negatives."""

import pytest

import h174.entrypoint_inventory as inventory
from test_h174_entrypoint_inventory import _set_html
from test_h174_source_candidates import pristine


def test_html_base_href_cannot_relabel_external_script_as_local(pristine):
    root = _set_html(
        pristine,
        '<base href="https://cdn.example.invalid/"><script src="app.js"></script>',
    )
    (root / "app.js").write_text("local decoy", encoding="utf-8")
    with pytest.raises(ValueError, match="base href"):
        inventory.inventory_entrypoint("pickr", root)


def test_ignored_untracked_script_cannot_count_as_pinned_source(pristine, monkeypatch):
    root = _set_html(pristine, '<script src="ignored.js"></script>')
    (root / "ignored.js").write_text("ignored local artifact", encoding="utf-8")

    def not_tracked(*args):
        raise ValueError("H174 candidate original file not tracked")

    monkeypatch.setattr(inventory, "_git", not_tracked, raising=False)
    result = inventory.inventory_entrypoint("pickr", root)
    assert result["local_present"] == []
    assert result["local_untracked"] == ["ignored.js"]


def test_tracked_script_is_source_present_but_not_native_qualified(pristine):
    root, _, answers = pristine
    root = _set_html(pristine, '<script src="tracked.js"></script>')
    (root / "tracked.js").write_text("tracked source", encoding="utf-8")
    answers[("ls-files", "--error-unmatch", "--", "tracked.js")] = "tracked.js"
    result = inventory.inventory_entrypoint("pickr", root)
    assert result["local_present"] == ["tracked.js"]
    assert result["local_untracked"] == []
    assert result["source_native_qa_qualified"] is False
