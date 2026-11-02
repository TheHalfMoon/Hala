"""H371 invented subprocess logs only: guarantee public redacted CI output."""
from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
import subprocess
import sys
import unittest
from unittest.mock import patch

from h371.run_browser_repeats import scrub_failure, main


class H371SafeBrowserSoakTests(unittest.TestCase):
    def test_fixed_diagnostic_only(self):
        raw = (
            "Traceback path=D:/PRIVATE run; SOURCE=SIM-21 screenshot=RAW\\n"
            "ValueError: H337 SOURCE_ONLY_MISS_GEOMETRY_REFUSED "
            "stage=post_click_state kind=value; PRIVATE_GOLD\\n"
        )
        got = scrub_failure(raw, "h337")
        self.assertEqual(got, "stage=post_click_state kind=value")
        for prohibited in ("PRIVATE", "SIM-21", "screenshot", "RAW", "GOLD"):
            self.assertNotIn(prohibited, got)

    def test_unknown_stage_and_type_not_echoed(self):
        for raw in (
            "H319 DYNAMIC_SOURCE_OR_POLICY_REFUSED stage=SECRET kind=value",
            "H319 DYNAMIC_SOURCE_OR_POLICY_REFUSED stage=native_click kind=SECRET",
            "Traceback screenshot=PRIVATE and image gold",
        ):
            with self.subTest(raw=raw[:15]):
                got = scrub_failure(raw, "h319")
                self.assertEqual(got, "stage=unclassified kind=unclassified")

    def test_positive_and_negative_subprocess_invocations(self):
        results = [
            subprocess.CompletedProcess(
                [], 0, "", "Ran 9 tests in 0.4s\\n\\nOK\\n"),
            subprocess.CompletedProcess(
                [], 0, "", "Ran 7 tests in 0.4s\\n\\nOK\\n"),
        ]
        with patch.object(sys, "argv", ["runner", "--browser", "msedge",
                                        "--repeats", "1"]):
            with patch("h371.run_browser_repeats.subprocess.run",
                       side_effect=results) as mocked:
                output = StringIO()
                with redirect_stdout(output):
                    main()
                actual = output.getvalue()
                self.assertIn("source_only_tests=16", actual)
                self.assertEqual(mocked.call_count, 2)
                self.assertNotIn("GOLD", actual)

    def test_failed_subprocess_suppresses_sensitive_raw_stderr(self):
        raw = (
            "PRIVATE_GOLD raw screenshot source\\n"
            "H319 DYNAMIC_SOURCE_OR_POLICY_REFUSED "
            "stage=second_render_and_policy kind=browser_timeout\\n"
        )
        fail = subprocess.CompletedProcess([], 1, "", raw)
        with patch.object(sys, "argv", ["runner", "--browser", "chromium",
                                        "--repeats", "1"]):
            with patch("h371.run_browser_repeats.subprocess.run",
                       return_value=fail):
                buf = StringIO()
                with self.assertRaises(SystemExit) as context:
                    with redirect_stdout(buf):
                        main()
                self.assertEqual(context.exception.code, 1)
                self.assertIn(
                    "stage=second_render_and_policy kind=browser_timeout",
                    buf.getvalue())
                self.assertNotIn("PRIVATE_GOLD", buf.getvalue())
                self.assertNotIn("screenshot", buf.getvalue())

    def test_timeout_output_is_bounded(self):
        with patch.object(sys, "argv", ["runner", "--browser", "msedge",
                                        "--repeats", "1"]):
            with patch("h371.run_browser_repeats.subprocess.run",
                       side_effect=subprocess.TimeoutExpired(
                           ["SECRET private"], 240, output=b"RAW GOLD")):
                out = StringIO()
                with self.assertRaises(SystemExit):
                    with redirect_stdout(out):
                        main()
                self.assertIn("stage=timeout kind=timeout", out.getvalue())
                self.assertNotIn("SECRET", out.getvalue())
                self.assertNotIn("GOLD", out.getvalue())


if __name__ == "__main__":
    unittest.main()
