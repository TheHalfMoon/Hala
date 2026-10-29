"""Synthetic H174 entrypoint references: no browser, gold, or old OOD."""
import hashlib

import pytest

from h174.entrypoint_inventory import inventory_entrypoint
from test_h174_source_candidates import pristine  # reuse synthetic-only fixture


def _set_html(pristine, content):
    root, pin, _ = pristine
    raw = content.encode("utf-8")
    (root / pin["html_path"]).write_bytes(raw)
    pin["html_sha256"] = hashlib.sha256(raw).hexdigest()
    return root


def test_missing_original_script_is_reported_not_native_qualified(pristine):
    root = _set_html(pristine, '<script type="module" src="./missing.js"></script>')
    result = inventory_entrypoint("pickr", root)
    assert result["local_missing"] == ["./missing.js"]
    assert result["source_native_qa_qualified"] is False


def test_present_local_script_and_css_are_only_static_evidence(pristine):
    root = _set_html(pristine, '<script src="app.js"></script><link rel="stylesheet" href="app.css">')
    (root / "app.js").write_text("console.log('new source')", encoding="utf-8")
    (root / "app.css").write_text("body{}", encoding="utf-8")
    result = inventory_entrypoint("pickr", root)
    assert result["local_missing"] == []
    assert result["local_present"] == ["app.css", "app.js"]
    assert result["source_native_qa_qualified"] is False


def test_external_dependency_is_not_fetched(pristine):
    root = _set_html(pristine, '<script src="https://example.invalid/x.js"></script>')
    result = inventory_entrypoint("pickr", root)
    assert result["external_references"] == ["https://example.invalid/x.js"]
    assert result["model_calls"] == 0


def test_scss_is_not_browser_ready_css(pristine):
    root = _set_html(pristine, '<link rel="stylesheet" href="theme.scss">')
    (root / "theme.scss").write_text("a{color:red}", encoding="utf-8")
    result = inventory_entrypoint("pickr", root)
    assert result["nonbrowser_stylesheets"] == ["theme.scss"]


@pytest.mark.parametrize("reference", ["../private.js", "%2e%2e/private.js", "..\\private.js"])
def test_traversal_never_reads_outside_checkout(pristine, reference):
    root = _set_html(pristine, '<script src="' + reference + '"></script>')
    with pytest.raises(ValueError, match="unsafe local asset"):
        inventory_entrypoint("pickr", root)
