"""H61 test for a partial browser capture.

A browser can write a screenshot that is not a readable image when a capture
races a navigation. The corpus must reject that scene with a reason and
never stop on it, because the corpus is built inside a free session that
would otherwise be lost.
"""
import unittest
from pathlib import Path

from h61.firstparty_drag import audit_scene, build_scenes


class PartialCaptureTests(unittest.TestCase):
    def test_an_unreadable_screenshot_is_a_rejection_and_not_a_crash(self):
        with self.TemporaryDirectory() as directory:
            path = directory / "broken.png"
            path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\0" * 32)
            scene = build_scenes(1)[0]
            problems = audit_scene(path, scene)
            self.assertTrue(problems)
            self.assertIn("readable", problems[0])

    def test_an_empty_screenshot_is_a_rejection(self):
        with self.TemporaryDirectory() as directory:
            path = directory / "empty.png"
            path.write_bytes(b"")
            problems = audit_scene(path, build_scenes(1)[0])
            self.assertTrue(problems)

    def TemporaryDirectory(self):
        import tempfile

        class _Holder(tempfile.TemporaryDirectory):
            def __enter__(self):
                value = super().__enter__()
                return Path(value)

        return _Holder()


if __name__ == "__main__":
    unittest.main()
