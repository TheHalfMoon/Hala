"""H174 interact source-only synthetic contract; never opens donor GUI."""
import ast
import inspect

import pytest
from h174.interact_native_smoke import run_native_dropzone_smoke


@pytest.mark.parametrize("port", [-1, 0, 65536, 2.5, "8181", True])
def test_h174_interact_refuses_untrusted_port_before_browser_access(port):
    with pytest.raises(ValueError, match="invalid localhost port"):
        run_native_dropzone_smoke(port, "chromium")


def test_h174_interact_original_drop_is_nonmocked_source_native():
    src = inspect.getsource(run_native_dropzone_smoke)
    tree = ast.parse(src)
    assert "dropzones/index.html" in src
    assert "#drag3" in src and "#drop1" in src
    assert "Dropped" in src
    attrs = [n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)]
    assert "down" in attrs and "up" in attrs and "move" in attrs
    assert "screenshot" in attrs and "convert" in attrs
    assert not any(isinstance(n, ast.Assert) for n in ast.walk(tree))
    assert "allow_loopback_request" in src
    assert "create_isolated_smoke_context" in src


def test_h174_interact_vite_navigation_does_not_wait_for_hmr_network_idle():
    """Prove no new indefinite HMR wait is introduced into native replay."""
    src = inspect.getsource(run_native_dropzone_smoke)
    tree = ast.parse(src)
    navigations = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        and n.func.attr == "goto"
    ]
    assert len(navigations) == 1
    options = {k.arg: k.value for k in navigations[0].keywords}
    assert isinstance(options["wait_until"], ast.Constant)
    assert options["wait_until"].value == "domcontentloaded"
    assert isinstance(options["timeout"], ast.Constant)
    assert options["timeout"].value == 25000
    assert "networkidle" not in src
