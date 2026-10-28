"""H173 shared original-browser renderer contract, CPU-only negative guards."""
import unittest
from pathlib import Path
from unittest.mock import patch

from h173.browser_runtime import BROWSER_ARGS, original_browser_options

ROOT = Path(__file__).resolve().parents[1] / "h173"


class H173OriginalBrowserRuntimeTests(unittest.TestCase):
    def test_freeze_software_compositing_for_replay(self):
        self.assertEqual(BROWSER_ARGS, ("--disable-gpu",))
        with patch("h173.browser_runtime._chromium_binary", return_value=None):
            self.assertEqual(
                original_browser_options(),
                {"headless": True, "args": ["--disable-gpu"]},
            )

    def test_use_explicit_local_binary_when_present(self):
        with patch(
            "h173.browser_runtime._chromium_binary",
            return_value=r"C:\original\chromium.exe",
        ):
            options = original_browser_options()
        self.assertEqual(options["executable_path"], r"C:\original\chromium.exe")
        self.assertEqual(options["args"], ["--disable-gpu"])

    def test_same_runtime_for_capture_replay_native_and_public_qa(self):
        for name in (
            "capture_ood.py",
            "ci_source_smoke.py",
            "ci_native_actuator_smoke.py",
            "native_reward.py",
        ):
            code = (ROOT / name).read_text(encoding="utf8")
            self.assertIn("original_browser_options()", code, name)
            self.assertNotIn("from h130.browser_env import _chromium_binary", code, name)

    def test_public_qa_requires_distinct_browser_launches(self):
        code = (ROOT / "ci_source_smoke.py").read_text(encoding="utf8")
        self.assertIn('for phase in ("capture", "replay"):', code)
        self.assertIn("independent_chromium_processes", code)
        self.assertIn("browser.close()", code)
        self.assertIn("same_exact_png_pixels(saved, raw)", code)


if __name__ == "__main__":
    unittest.main()
