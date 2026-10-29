"""H174 bounded static browser import edge contract — synthetic only."""
import pytest
import h174.module_edges as edges


def _fixture(tmp_path, monkeypatch, html=None, main=None, css=None, tracked=None):
    (tmp_path / "index.html").write_text(
        html or '<script type="module" src="./main.js"></script>', encoding="utf-8")
    (tmp_path / "main.js").write_text(
        main if main is not None else 'import "./inner.js"\n', encoding="utf-8")
    (tmp_path / "inner.js").write_text("export const a = 1\n", encoding="utf-8")
    (tmp_path / "style.css").write_text(css or "", encoding="utf-8")
    monkeypatch.setitem(edges.CANDIDATES, "synthetic",
                        {"html_path": "index.html"})
    monkeypatch.setattr(edges, "inventory_entrypoint",
                        lambda name, checkout: {
                            "local_missing": [], "local_untracked": [],
                            "external_references": [], "nonbrowser_stylesheets": [],
                        })
    saved = tracked or {"main.js", "inner.js", "style.css"}
    monkeypatch.setattr(
        edges, "_git",
        lambda root, *args: args[-1] if args[-1] in saved else "",
    )
    return tmp_path


def test_tracked_recursive_js_import_graph(tmp_path, monkeypatch):
    p = _fixture(tmp_path, monkeypatch)
    result = edges.inventory_module_edges("synthetic", p)
    assert result["tracked_edges"] == ["main.js -> ./inner.js"]
    assert result["inspected_tracked_files"] == 2
    assert result["requires_build_or_resolution"] is False
    assert result["native_browser_qualified"] is False


def test_bare_package_never_treated_as_browser_resolved(tmp_path, monkeypatch):
    p = _fixture(tmp_path, monkeypatch, main="import interact from '@interactjs/interactjs'\n")
    result = edges.inventory_module_edges("synthetic", p)
    assert result["bare_packages"] == ["main.js -> @interactjs/interactjs"]
    assert result["requires_build_or_resolution"] is True


def test_root_absolute_js_import_is_original_checkout_root(tmp_path, monkeypatch):
    p = _fixture(tmp_path, monkeypatch, main="import App from '/inner.js'\n")
    result = edges.inventory_module_edges("synthetic", p)
    assert result["tracked_edges"] == ["main.js -> /inner.js"]


def test_missing_relative_module_not_silently_resolved(tmp_path, monkeypatch):
    p = _fixture(tmp_path, monkeypatch, main="import './nonexistent.js'\n")
    assert edges.inventory_module_edges("synthetic", p)["missing_local"] == [
        "main.js -> ./nonexistent.js"
    ]


def test_ignored_or_untracked_module_is_never_trusted(tmp_path, monkeypatch):
    p = _fixture(tmp_path, monkeypatch, tracked={"main.js"})
    assert edges.inventory_module_edges("synthetic", p)["untracked_local"] == [
        "main.js -> ./inner.js"
    ]


def test_dynamic_variable_import_is_unparsed_not_approved(tmp_path, monkeypatch):
    p = _fixture(tmp_path, monkeypatch, main="const x = import(target)\n")
    r = edges.inventory_module_edges("synthetic", p)
    assert "main.js" in r["unparsed_edges"] and r["requires_build_or_resolution"]


def test_literal_dynamic_import_is_visible(tmp_path, monkeypatch):
    p = _fixture(tmp_path, monkeypatch, main='const x = import("./inner.js")\n')
    assert edges.inventory_module_edges("synthetic", p)["tracked_edges"] == [
        "main.js -> ./inner.js"
    ]


def test_external_remote_import_never_local(tmp_path, monkeypatch):
    p = _fixture(tmp_path, monkeypatch, main='import "https://example.org/mod.js"\n')
    assert edges.inventory_module_edges("synthetic", p)["external_edges"]


@pytest.mark.parametrize("ref", [
    "./../private.js", "./%2e%2e/secret.js", "/../hidden.js",
    "./%252e%252e/private.js", "./%2fprivate.js",
])
def test_encoded_and_relative_traversal_fail_closed(tmp_path, monkeypatch, ref):
    p = _fixture(tmp_path, monkeypatch, main=f'import "{ref}"\n')
    with pytest.raises(ValueError, match="traversal|unsafe"):
        edges.inventory_module_edges("synthetic", p)


def test_css_import_detected_as_dependency(tmp_path, monkeypatch):
    p = _fixture(tmp_path, monkeypatch,
                 html='<link rel="stylesheet" href="./style.css">',
                 css='@import url("./inner.css");')
    assert edges.inventory_module_edges("synthetic", p)["missing_local"] == [
        "style.css -> ./inner.css"
    ]

def test_css_relative_import_without_dot_slash_is_local(tmp_path, monkeypatch):
    p = _fixture(tmp_path, monkeypatch,
                 html='<link rel="stylesheet" href="./style.css">',
                 css='@import "theme.css";')
    r = edges.inventory_module_edges("synthetic", p)
    assert r["missing_local"] == ["style.css -> theme.css"]
    assert not r["bare_packages"]


def test_direct_html_missing_is_reported_separately(tmp_path, monkeypatch):
    p = _fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(edges, "inventory_entrypoint",
                        lambda name, checkout: {
                            "local_missing": ["./missing-entry.js"],
                            "local_untracked": [],
                            "external_references": [],
                            "nonbrowser_stylesheets": [],
                        })
    r = edges.inventory_module_edges("synthetic", p)
    assert r["direct_html_missing"] == ["./missing-entry.js"]
    assert r["requires_build_or_resolution"] and not r["native_browser_qualified"]
