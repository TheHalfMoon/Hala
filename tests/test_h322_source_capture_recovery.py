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
        from h285 import train_only_browser_candidates as train
        from h309 import exposed_source_rgb_split as split
        self.assertIs(train.source_capture_png, source_capture_png)
        self.assertIs(split.source_capture_png, source_capture_png)


if __name__ == "__main__":
    unittest.main()
