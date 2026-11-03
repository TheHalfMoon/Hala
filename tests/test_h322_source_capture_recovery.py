"""H322 invented-only screenshot recovery tests. No real or private frames."""
import unittest
from unittest.mock import patch

from h322.source_capture import SourceCaptureRefused, source_capture_png


class Error(Exception):
    """Local fake with the Playwright error class name; no browser needed."""


TRANSIENT = (
    "Page.screenshot: Protocol error (Page.captureScreenshot): "
    "Unable to capture screenshot"
)


class H322CaptureTests(unittest.TestCase):
    def test_exact_chromium_transient_waits_once_before_second_fresh_capture(self):
        class FakePage:
            def __init__(self):
                self.waits = []
            def wait_for_timeout(self, milliseconds):
                self.waits.append(milliseconds)
        page = FakePage()
        captured = []
        def capture(_page):
            captured.append(True)
            if len(captured) == 1:
                raise Error(TRANSIENT)
            return b"fresh-valid-source-frame"
        self.assertEqual(source_capture_png(page, capture=capture),
                         b"fresh-valid-source-frame")
        self.assertEqual(page.waits, [150])
        self.assertEqual(len(captured), 2)

    def test_failed_chromium_settle_never_delivers_any_frame(self):
        class FakePage:
            def wait_for_timeout(self, milliseconds):
                raise RuntimeError("PRIVATE_BROWSER_STATE_DO_NOT_LEAK")
        seen = []
        def capture(page):
            seen.append(True)
            if len(seen) == 1:
                raise Error(TRANSIENT)
            return b"MUST_NOT_BE_CONSUMED"
        with self.assertRaises(SourceCaptureRefused) as caught:
            source_capture_png(FakePage(), capture=capture)
        self.assertEqual(str(caught.exception),
                         "H322 SOURCE_SCREENSHOT_CAPTURE_REFUSED")
        self.assertIsNone(caught.exception.__cause__)
        self.assertEqual(len(seen), 1)

    def test_second_exact_error_after_settle_still_refuses(self):
        class FakePage:
            def __init__(self):
                self.waits = []
            def wait_for_timeout(self, milliseconds):
                self.waits.append(milliseconds)
        page = FakePage()
        calls = []
        def capture(_page):
            calls.append(True)
            raise Error(TRANSIENT + " PRIVATE_URL_MUST_NOT_LEAK")
        with self.assertRaises(SourceCaptureRefused) as caught:
            source_capture_png(page, capture=capture)
        self.assertEqual(page.waits, [150])
        self.assertEqual(len(calls), 2)
        self.assertEqual(str(caught.exception),
                         "H322 SOURCE_SCREENSHOT_CAPTURE_REFUSED")
        self.assertIsNone(caught.exception.__cause__)

    def test_other_error_never_schedules_browser_wait(self):
        class FakePage:
            def wait_for_timeout(self, milliseconds):
                raise AssertionError("Never wait for unexpected capture failure")
        seen = []
        def capture(page):
            seen.append(True)
            raise Error("Page.screenshot: error unrelated to known CDP symptom")
        with self.assertRaises(SourceCaptureRefused):
            source_capture_png(FakePage(), capture=capture)
        self.assertEqual(len(seen), 1)

    def test_first_valid_capture_never_retried(self):
        calls = []
        def capture(page):
            calls.append(page)
            return b"fictional"
        self.assertEqual(source_capture_png(object(), capture=capture), b"fictional")
        self.assertEqual(len(calls), 1)

    def test_exact_known_transient_retries_once_then_requires_real_capture(self):
        calls = []
        def capture(page):
            calls.append(page)
            if len(calls) == 1:
                raise Error(TRANSIENT)
            return b"fictional-valid-after-original-validator"
        self.assertEqual(
            source_capture_png(object(), capture=capture),
            b"fictional-valid-after-original-validator",
        )
        self.assertEqual(len(calls), 2)

    def test_second_transient_fails_with_fixed_sanitized_error(self):
        calls = []
        def capture(page):
            calls.append(page)
            raise Error(TRANSIENT + " SECRET_DO_NOT_LEAK")
        with self.assertRaises(SourceCaptureRefused) as caught:
            source_capture_png(object(), capture=capture)
        self.assertEqual(len(calls), 2)
        self.assertEqual(str(caught.exception),
                         "H322 SOURCE_SCREENSHOT_CAPTURE_REFUSED")
        self.assertIsNone(caught.exception.__cause__)

    def test_unexpected_exception_is_not_retried_or_leaked(self):
        calls = []
        def capture(page):
            calls.append(page)
            raise RuntimeError("PRIVATE_FRAME_SOURCE")
        with self.assertRaises(SourceCaptureRefused) as caught:
            source_capture_png(object(), capture=capture)
        self.assertEqual(len(calls), 1)
        self.assertNotIn("PRIVATE_FRAME_SOURCE", str(caught.exception))

    def test_similarly_named_non_playwright_error_does_not_retry(self):
        calls = []
        def capture(page):
            calls.append(page)
            raise ValueError(TRANSIENT)
        with self.assertRaises(SourceCaptureRefused):
            source_capture_png(object(), capture=capture)
        self.assertEqual(len(calls), 1)

    def test_original_validation_still_runs_and_rejects_invalid_image(self):
        with patch("h322.source_capture._observed_png",
                   side_effect=ValueError("INVALID_MODEL_FRAME")) as validator:
            with self.assertRaises(SourceCaptureRefused):
                source_capture_png(object())
            self.assertEqual(validator.call_count, 1)

    def test_source_research_paths_use_shared_helper(self):
        # Static import/call-site check without invoking optional browser deps.
        from pathlib import Path
        root = Path(__file__).resolve().parents[1]
        for source, expected in (
            ("h285/train_only_browser_candidates.py", 2),
            ("h309/exposed_source_rgb_split.py", 1),
        ):
            with self.subTest(source=source):
                body = (root / source).read_text(encoding="utf-8")
                self.assertIn(
                    "from h322.source_capture import source_capture_png", body)
                self.assertEqual(body.count("source_capture_png(page)"), expected)


if __name__ == "__main__":
    unittest.main()
