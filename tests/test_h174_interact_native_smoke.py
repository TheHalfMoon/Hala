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
