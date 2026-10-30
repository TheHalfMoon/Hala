"""Synthetic-only H243 regression: fail-closed offline before any browser page."""
import ast
from pathlib import Path
import unittest

from h243.browser_offline import require_browser_offline

ROOT = Path(__file__).resolve().parents[1]
CAPTURES = (
    ("h231/native_replay.py", "ctx"),
    ("h236/original_dragula_native.py", "ctx"),
    ("h242/private_dragula_prospects.py", "context"),
)


class FakeContext:
    def __init__(self, error=None):
        self.calls = []
        self.error = error

    def set_offline(self, value):
        self.calls.append(value)
        if self.error is not None:
            raise self.error


class H243OfflineTests(unittest.TestCase):
    def test_enables_offline_true(self):
        ctx = FakeContext()
        self.assertIsNone(require_browser_offline(ctx))
        self.assertEqual(ctx.calls, [True])

    def test_missing_control_fails_closed(self):
        with self.assertRaisesRegex(RuntimeError, "no offline control"):
            require_browser_offline(object())

    def test_failing_control_fails_closed(self):
        ctx = FakeContext(RuntimeError("synthetic backend failure"))
        with self.assertRaisesRegex(RuntimeError, "offline setup failed"):
            require_browser_offline(ctx)
        self.assertEqual(ctx.calls, [True])

    def test_none_setter_fails_closed(self):
        class Missing:
            set_offline = None
        with self.assertRaisesRegex(RuntimeError, "no offline control"):
            require_browser_offline(Missing())

    def test_all_native_capture_paths_require_offline_before_page(self):
        for relative, context_name in CAPTURES:
            with self.subTest(path=relative):
                source = (ROOT / relative).read_text(encoding="utf-8")
                tree = ast.parse(source)
                calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
                offline = [n for n in calls if isinstance(n.func, ast.Name)
                           and n.func.id == "require_browser_offline"
                           and len(n.args) == 1
                           and isinstance(n.args[0], ast.Name)
                           and n.args[0].id == context_name]
                pages = [n for n in calls if isinstance(n.func, ast.Attribute)
                         and n.func.attr == "new_page"
                         and isinstance(n.func.value, ast.Name)
                         and n.func.value.id == context_name]
                self.assertEqual(len(offline), 1)
                self.assertGreaterEqual(len(pages), 1)
                self.assertTrue(all(offline[0].lineno < p.lineno for p in pages))
                self.assertIn('service_workers="block"', source)
                self.assertIn('.route("**/*"', source)


if __name__ == "__main__":
    unittest.main()
