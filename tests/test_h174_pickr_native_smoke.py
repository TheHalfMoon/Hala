"""H174 synthetic negative tests: no donor, browser, private media, or model calls."""
from __future__ import annotations

from hashlib import sha256
import ast
import inspect
import os
from pathlib import Path
import subprocess
import sys
import h174.pickr_native_smoke as source_module
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
    with pytest.raises((ValueError, FileNotFoundError)):
        verify_derived_assets(_dist(tmp_path, script=script))


def test_h174_pickr_rejects_base_url(tmp_path):
    with pytest.raises(ValueError, match="base"):
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
    with pytest.raises(ValueError, match="Empty"):
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


def test_h174_pickr_rejects_any_unexpected_inline_script(tmp_path):
    root = _dist(tmp_path)
    html = root / "index.html"
    html.write_text(html.read_text(encoding="utf-8").replace(
        "</head>", '<script>window.pwned = true</script></head>'
    ), encoding="utf-8")
    with pytest.raises(ValueError, match="inline script"):
        verify_derived_assets(root)


def test_h174_pickr_rejects_src_less_module_script(tmp_path):
    root = _dist(tmp_path)
    html = root / "index.html"
    html.write_text(html.read_text(encoding="utf-8").replace(
        "</head>", '<script type="module">import "./assets/app.js"</script></head>'
    ), encoding="utf-8")
    with pytest.raises(ValueError, match="inline script"):
        verify_derived_assets(root)


@pytest.mark.parametrize("violation", ["empty-js", "inline", "base", "traversal"])
def test_h174_pickr_fail_closed_even_with_python_optimized_mode(tmp_path, violation):
    root = _dist(tmp_path)
    html = root / "index.html"
    if violation == "empty-js":
        (root / "assets" / "app.js").write_bytes(b"")
    elif violation == "inline":
        html.write_text(html.read_text(encoding="utf-8").replace(
            "</head>", '<script>window.unsafe=true</script></head>'
        ), encoding="utf-8")
    elif violation == "base":
        html.write_text(html.read_text(encoding="utf-8").replace(
            "<head>", '<head><base href="https://evil.example/">'
        ), encoding="utf-8")
    else:
        html.write_text(html.read_text(encoding="utf-8").replace(
            "./assets/app.js", "./assets/../app.js"
        ), encoding="utf-8")
    package_root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(package_root) + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, "-O", "-c",
         "from pathlib import Path; from h174.pickr_native_smoke import verify_derived_assets; "
         + "verify_derived_assets(Path(" + repr(str(root)) + "))"],
        cwd=package_root, env=env, capture_output=True, text=True, timeout=20,
        check=False,
    )
    assert result.returncode != 0, "Optimized interpreter silently bypassed asset admission"
    assert "ValueError" in result.stderr, result.stderr


def test_h174_pickr_module_has_no_optimization_skippable_assert_guards():
    module_ast = ast.parse(inspect.getsource(source_module))
    assert not any(isinstance(node, ast.Assert) for node in ast.walk(module_ast))


@pytest.mark.parametrize("injected_html", [
    '<button onclick="window.unsafe=true">Click</button>',
    '<img src="./assets/app.js" onerror="window.unsafe=true">',
    '<iframe src="./assets/app.js"></iframe>',
    '<object data="./assets/app.js"></object>',
    '<embed src="./assets/app.js">',
    '<meta http-equiv="refresh" content="0;url=./assets/app.js">',
    '<script src="./assets/app.js" src="https://evil.example/other.js"></script>',
    '<a href="javascript:alert(1)">Run</a>',
    '<link rel="import" href="./assets/app.js">',
    '<svg onload="window.unsafe=true"></svg>',
])
def test_h174_pickr_refuses_ambiguous_or_active_derived_html(tmp_path, injected_html):
    root = _dist(tmp_path)
    html = root / "index.html"
    html.write_text(html.read_text(encoding="utf-8").replace(
        "</head>", injected_html + "</head>"
    ), encoding="utf-8")
    with pytest.raises(ValueError, match="Unsafe active HTML"):
        verify_derived_assets(root)


@pytest.mark.parametrize("injected_html", [
    '<button onclick="window.unsafe=true">Click</button>',
    '<iframe src="./assets/app.js"></iframe>',
    '<meta http-equiv="refresh" content="0;url=./assets/app.js">',
])
def test_h174_pickr_active_html_stays_rejected_in_python_optimized_mode(tmp_path, injected_html):
    root = _dist(tmp_path)
    html = root / "index.html"
    html.write_text(html.read_text(encoding="utf-8").replace(
        "</head>", injected_html + "</head>"
    ), encoding="utf-8")
    package_root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(package_root) + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, "-O", "-c",
         "from pathlib import Path; from h174.pickr_native_smoke import verify_derived_assets; "
         + "verify_derived_assets(Path(" + repr(str(root)) + "))"],
        cwd=package_root, env=env, capture_output=True, text=True, timeout=20,
        check=False,
    )
    assert result.returncode != 0, "Optimized Python bypassed active HTML safeguard"
    assert "ValueError: Unsafe active HTML" in result.stderr, result.stderr


class _FakeIsolatedContext:
    def __init__(self):
        self.websocket_route = None

    def route_web_socket(self, pattern, handler):
        self.websocket_route = (pattern, handler)


class _FakeBrowserForIsolation:
    def __init__(self):
        self.options = None
        self.context = _FakeIsolatedContext()

    def new_context(self, **kwargs):
        self.options = kwargs
        return self.context


def test_h174_pickr_native_context_disables_service_workers_and_downloads():
    """Source-only spy checks exact browser options without launching Edge."""
    fake = _FakeBrowserForIsolation()
    context = source_module.create_isolated_smoke_context(fake)
    assert context is fake.context
    assert fake.options["service_workers"] == "block"
    assert fake.options["accept_downloads"] is False
    assert fake.options["viewport"] == {"width": 960, "height": 720}
    assert fake.options["device_scale_factor"] == 1
    assert fake.options["locale"] == "en-US"
    assert fake.options["reduced_motion"] == "reduce"


def test_h174_pickr_websocket_connections_closed_without_connecting():
    """WebSocket route does not proxy private data to any endpoint."""
    fake = _FakeBrowserForIsolation()
    context = source_module.create_isolated_smoke_context(fake)
    pattern, handler = context.websocket_route
    assert pattern == "**/*"

    class FakeWebSocket:
        def __init__(self):
            self.closed = False
            self.connected = False

        def close(self):
            self.closed = True

        def connect(self):
            self.connected = True

    socket = FakeWebSocket()
    handler(socket)
    assert socket.closed is True
    assert socket.connected is False


def test_h174_pickr_native_smoke_uses_isolated_context_factory():
    """Protect against accidentally reverting the original browser callsite."""
    tree = ast.parse(inspect.getsource(source_module.run_native_smoke))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    assert any(
        isinstance(c.func, ast.Name)
        and c.func.id == "create_isolated_smoke_context"
        for c in calls
    )
    assert not any(
        isinstance(c.func, ast.Attribute) and c.func.attr == "new_context"
        for c in calls
    )


def test_h174_pickr_isolation_factory_survives_python_optimized_mode(tmp_path):
    """Explicitly check isolation is not an assert-only policy."""
    package_root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(package_root) + os.pathsep + env.get("PYTHONPATH", "")
    program = (
        "from h174.pickr_native_smoke import create_isolated_smoke_context\n"
        "class FakeContext:\n"
        " def route_web_socket(self, pattern, handler):\n"
        "  self.pattern=pattern; self.handler=handler\n"
        "class FakeBrowser:\n"
        " def new_context(self, **kwargs):\n"
        "  if kwargs['service_workers']!='block' or kwargs['accept_downloads'] is not False:\n"
        "   raise ValueError('isolation context unexpectedly relaxed')\n"
        "  return FakeContext()\n"
        "ctx=create_isolated_smoke_context(FakeBrowser());\n"
        "if ctx.pattern!='**/*': raise ValueError('websocket route missing')\n"
    )
    result = subprocess.run(
        [sys.executable, "-O", "-c", program], cwd=package_root,
        env=env, capture_output=True, text=True, timeout=20, check=False,
    )
    assert result.returncode == 0, result.stderr
